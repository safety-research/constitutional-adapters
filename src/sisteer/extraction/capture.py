"""Capture residual-stream activations at chosen layers and positions."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch

from ..models import get_decoder_layers

POSITION_SPECS = ("last", "mean", "completion_mean", "completion_last", "all")

# Position specs whose pooled positions provably sit in the SUFFIX of a
# shared-prefix batch, so a prefix-KV-cached capture (suffix-width hidden
# states) can serve them. "mean" is excluded on purpose: it pools every real
# position, prefix included, and a suffix-only capture would silently pool a
# truncated support — prefix_cache=True REFUSES it rather than approximating.
_PREFIX_CACHEABLE_POSITIONS = ("last", "completion_mean", "completion_last")


@contextlib.contextmanager
def capture_layer_outputs(model, layers: list[int], site: str = "post"):
    """Yields a dict that fills with {layer: hidden [B, S, d]} on each forward.

    site="post" (default): residual stream leaving block l — the harness-wide
    convention. site="pre": residual stream entering block l (ActAdd's
    hook_resid_pre; note resid_pre of block l == resid_post of block l-1).
    """
    captured: dict[int, torch.Tensor] = {}
    decoder = get_decoder_layers(model)
    handles = []

    def make_post_hook(idx):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx] = hidden.detach()

        return hook

    def make_pre_hook(idx):
        def hook(module, args, kwargs):
            hidden = kwargs.get("hidden_states", args[0] if args else None)
            captured[idx] = hidden.detach()
            return None

        return hook

    for idx in layers:
        if site == "post":
            handles.append(decoder[idx].register_forward_hook(make_post_hook(idx)))
        elif site == "pre":
            handles.append(
                decoder[idx].register_forward_pre_hook(
                    make_pre_hook(idx), with_kwargs=True
                )
            )
        else:
            raise ValueError(f"site must be 'pre' or 'post', got {site!r}")
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()


def _last_logit_kwargs(model) -> dict:
    """Keep the forward's logits at ``[B, 1, V]`` for a CAPTURE pass.

    This function reads hooked hidden states and DISCARDS the model output
    entirely, but a plain forward still materializes ``[B, T, V]`` logits —
    at Llama-3's 128k vocab and an SI-rendered ~6.9k-token prompt that is
    ~1.78 GB PER ROW, which is what caps the capture batch size (the
    hidden states themselves are ~30x smaller). Asking for one logit
    position is VALUE-IDENTICAL here (nothing reads them) and lifts the
    ceiling by roughly an order of magnitude.

    Version-matched across transformers releases — the kwarg was renamed ``num_logits_to_keep`` -> ``logits_to_keep``
    — and degrades to a plain forward on a signature that has neither, so
    an older transformers keeps working (just at the old ceiling)."""
    import inspect

    params = inspect.signature(model.forward).parameters
    if "logits_to_keep" in params:
        return {"logits_to_keep": 1}
    if "num_logits_to_keep" in params:
        return {"num_logits_to_keep": 1}
    return {}


@dataclass
class ActivationBatch:
    """Pooled activations: {layer: [N, d]} for N examples."""

    acts: dict[int, torch.Tensor]


def _pool(
    hidden: torch.Tensor,  # [B, S, d]
    attn_mask: torch.Tensor,  # [B, S]
    position: str,
    prompt_lens: torch.Tensor | None = None,  # [B], required for completion_*
) -> torch.Tensor:
    B, S, _ = hidden.shape
    idx = torch.arange(S, device=hidden.device)[None, :]  # [1, S]
    if position == "last":
        last = attn_mask.sum(dim=1) - 1  # right-padding assumed
        return hidden[torch.arange(B, device=hidden.device), last]
    if position == "mean":
        m = attn_mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1)
    if position.startswith("lastk:"):
        # k-th token from the end of the real sequence (lastk:1 == last;
        # lastk:2 == CAA's answer-letter position in "... (A)").
        k = int(position.split(":", 1)[1])
        pos = (attn_mask.sum(dim=1) - k).clamp(min=0)
        return hidden[torch.arange(B, device=hidden.device), pos]
    if position in ("completion_mean", "completion_last"):
        if prompt_lens is None:
            raise ValueError(f"{position} pooling requires prompt_lens")
        comp = (idx >= prompt_lens[:, None].to(hidden.device)) & attn_mask.bool()
        if position == "completion_last":
            last = attn_mask.sum(dim=1) - 1
            return hidden[torch.arange(B, device=hidden.device), last]
        m = comp.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1)
    raise ValueError(f"position must be one of {POSITION_SPECS}")


@torch.no_grad()
def collect_pooled_activations(
    model,
    tok,
    texts: list[str],
    layers: list[int],
    position: str = "last",
    prompt_lens: list[int] | None = None,
    batch_size: int = 8,
    prefix_cache: bool = False,
) -> ActivationBatch:
    """Forward `texts`, pool residual activations per example.

    `prompt_lens[i]` = number of tokens of texts[i] that belong to the
    prompt (the remainder is the completion), needed for completion_* specs.
    Tokenization is right-padded here (pooling assumes it).

    ``prefix_cache=True`` (teacher-side prefix KV caching): when the texts
    share a literal common token prefix — a rendered SI at the head of
    every prompt, ~95% of all tokens for the long SIs — the prefix runs
    through the model ONCE and every chunk forwards only its suffix
    against the cached KV (:mod:`.prefix_cache`). Semantically exact,
    fp-tolerance-only difference (kernel shapes change); pooled positions
    must sit in the suffix, so only ``_PREFIX_CACHEABLE_POSITIONS`` are
    accepted — "mean" REFUSES loudly rather than silently pooling a
    truncated support. The default ``False`` keeps every existing caller
    byte-identical.

    Under an ambient fit-parallel context (extraction.ddp — set only by the
    arm-task entrypoint, never at eval time) the TEXT LIST is sharded
    across the arm's ranks and the pooled rows are gathered back in
    original order, so every rank returns the identical full tensor. The
    downstream closed-form solves then run replicated on identical inputs.
    (Batch composition within a shard differs from the single-process
    chunking, so pooled values can differ from an unsharded run at fp
    reduction level — never across ranks.) The prefix-cached path is
    deliberately NOT sharded: it runs the full list replicated on every
    rank (rank 0 owns any downstream save), so ``prefix_cache=True``
    composes with an active context without its no-common-prefix fallback
    re-sharding an already-sharded list.
    """
    if prefix_cache:
        if position not in _PREFIX_CACHEABLE_POSITIONS:
            raise ValueError(
                f"prefix_cache=True cannot serve position={position!r}: its "
                "pooled positions are not confined to the suffix of a "
                f"shared-prefix batch (cacheable: "
                f"{_PREFIX_CACHEABLE_POSITIONS}). Run it uncached.")
        return _collect_pooled_prefix_cached(
            model, tok, texts, layers, position, prompt_lens, batch_size)
    from . import ddp as _ddp

    par = _ddp.active()
    n_total = len(texts)
    my_texts = _ddp.shard_items(list(texts), par)
    my_lens = (_ddp.shard_items(list(prompt_lens), par)
               if prompt_lens is not None else None)

    device = next(model.parameters()).device
    out: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    prev_side = tok.padding_side
    tok.padding_side = "right"
    try:
        for i in range(0, len(my_texts), batch_size):
            chunk = my_texts[i : i + batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True).to(device)
            lens = (
                torch.tensor(my_lens[i : i + batch_size])
                if my_lens is not None
                else None
            )
            with capture_layer_outputs(model, layers) as captured:
                model(**enc, **_last_logit_kwargs(model))
            for l in layers:
                pooled = _pool(
                    captured[l], enc["attention_mask"], position, lens
                )
                out[l].append(pooled.float().cpu())
    finally:
        tok.padding_side = prev_side
    if par is None or not par.enabled:
        return ActivationBatch(
            {l: torch.cat(v, dim=0) for l, v in out.items()})
    full: dict[int, torch.Tensor] = {}
    for l in layers:
        local = torch.cat(out[l], dim=0) if out[l] else torch.empty(0)
        local_rows = [local[j] for j in range(local.shape[0])]
        rows = _ddp.gather_sharded(local_rows, n_total, par)
        full[l] = torch.stack(rows, dim=0)
    return ActivationBatch(full)


def _collect_pooled_prefix_cached(
    model, tok, texts, layers, position, prompt_lens, batch_size,
) -> ActivationBatch:
    """The prefix-cached body of :func:`collect_pooled_activations`.

    Tokenizes every text once up front (the SAME ``add_special_tokens=True``
    call the uncached path's batched ``tok(chunk, padding=True)`` performs
    per row), derives the longest common token prefix P over ALL rows —
    capped so every pooled position lands in the suffix: at
    ``min(len) - 1`` (a "last"-pooled row must keep its last token in the
    suffix, and every row needs a non-empty suffix to forward at all) and
    additionally at ``min(prompt_lens)`` for the completion_* specs (their
    pooled positions start at each row's own prompt_len) — then builds the
    prefix KV once and pools each chunk's suffix-width hidden states with
    the mask/lens re-based by P. Falls back to the exact uncached path when
    no common prefix exists (P < 1): that is 'nothing to cache', not an
    error."""
    from .prefix_cache import TeacherPrefixCache

    device = next(model.parameters()).device
    all_ids = [tok(t, add_special_tokens=True)["input_ids"] for t in texts]
    if not all_ids:
        return ActivationBatch({l: torch.empty(0) for l in layers})
    cap = min(len(r) for r in all_ids) - 1
    if prompt_lens is not None:
        cap = min(cap, min(prompt_lens))
    p = 0
    ref = all_ids[0]
    while p < cap and all(len(r) > p and r[p] == ref[p] for r in all_ids):
        p += 1
    if p < 1:
        return collect_pooled_activations(
            model, tok, texts, layers, position, prompt_lens, batch_size,
            prefix_cache=False)
    pad_id = tok.pad_token_id
    if pad_id is None:
        raise ValueError(
            "prefix_cache=True needs tok.pad_token_id for the manual "
            "right-padded collate (the uncached path's tok(chunk, "
            "padding=True) would refuse the same way)")
    cache = TeacherPrefixCache.build(model, ref[:p])
    out: dict[int, list[torch.Tensor]] = {l: [] for l in layers}
    for i in range(0, len(all_ids), batch_size):
        chunk = all_ids[i : i + batch_size]
        width = max(len(r) for r in chunk)
        ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), width), dtype=torch.long)
        for j, r in enumerate(chunk):
            ids[j, : len(r)] = torch.tensor(r)
            attn[j, : len(r)] = 1
        ids, attn = ids.to(device), attn.to(device)
        with capture_layer_outputs(model, layers) as captured:
            # The suffix forward, through the prefix_cache module's ONE
            # model-call convention (offset seam included) — the hooks
            # capture suffix-width hidden states as a side effect.
            cache.forward_ids(model, ids, attn, p)
        lens = (
            torch.tensor(prompt_lens[i : i + batch_size]) - p
            if prompt_lens is not None
            else None
        )
        for l in layers:
            # Suffix-width hidden + suffix-width mask: "last" resolves to
            # (real_len - p) - 1, completion_* boundaries to prompt_len - p
            # — the same positions the full forward pooled, re-based by P.
            pooled = _pool(captured[l], attn[:, p:], position, lens)
            out[l].append(pooled.float().cpu())
    return ActivationBatch({l: torch.cat(v, dim=0) for l, v in out.items()})
