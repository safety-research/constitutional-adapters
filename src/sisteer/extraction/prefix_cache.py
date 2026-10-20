"""Teacher-side prefix KV caching.

Every teacher prompt for a given SI is the SAME rendered prefix (chat header
+ ``system=<SI text>``) followed by a per-pair attack suffix — a literal
common token prefix at positions ``0..P-1`` of every row under the
right-padded collate. Measured SI lengths: si_constitution 5,872 tokens,
si_irrelevant 6,380, safety_refusal 88 — so for the long SIs ~95% of all
teacher tokens per epoch recompute an identical prefix thousands of times
per arm-group. This module runs those P tokens through the model ONCE per
(SI, group) under ``no_grad``, keeps the ``past_key_values``, and replays
them per batch: the model is fed only suffix+completion ids with
``cache_position``/``position_ids`` offset by P, the prefix KV handed in
expanded across the batch dim.

MEMORY (corrected 2026-08-24 — this docstring originally claimed the batch
expansion stayed a zero-copy view, which was WRONG in the unsafe
direction): we hand ``Tensor.expand`` views to ``DynamicCache``, but
transformers 5.14.1 initialises each layer with ``torch.cat`` against an
empty tensor, materialising a contiguous BATCH-SIZED COPY at cache
construction (verified against the locked env: in-cache tensor
contiguous, different storage pointer, exactly B x the source bytes). So
per-batch prefix-KV memory scales with B (~0.72 GB/row at P=5,886 on
Llama-3.1-8B GQA bf16, plus the cat'ed B x (P+S) update during the
forward), TRANSIENT — freed with the batch's cache — on top of the one
persistent P-sized build. The NET peak effect is nonetheless NEGATIVE:
the cached forward never materialises the full-width logits tensor
([B, S, 128k]), and that saving exceeds the KV copies — measured,
instrument-validated (job 190735; job 190490's positive deltas were VOID
harness contamination, see scripts/bench_prefix_cache.py's post-mortem):
uncached 24.42 -> cached 19.34 GiB peak at bs=4, 33.85 -> 22.98 GiB at
bs=8. The copy is a real per-batch cost all the same, and it is
library-version-dependent behaviour; tests/test_prefix_cache.py pins the
storage semantics so a transformers change flags this note as stale
instead of silently un-truing it.

TEACHER LEGS ONLY, GRADIENT-TRAINING FORWARDS ONLY. The student is
SI-absent (no long prefix to share) and carries gradients — nothing in
this module may touch a student forward; :meth:`TeacherPrefixCache.forward`
runs under a forced ``torch.no_grad()`` for exactly that reason. And no
DECISION-BEARING pass may read a cached forward (review finding
2026-08-24): the floor probes (whose numbers the ``floor_ratio_max`` bar
rules on — a rejection writes no artifact, so a cache-flipped verdict is
unrecoverable at decision time) and held-out epoch selection are computed
by the plain full forward inside each trainer, flag or no flag — pinned
by the cache-call-count tests.

Exactness: causal attention makes prefix K/V independent of the suffix, so
a cached forward is SEMANTICALLY exact — same distribution, same floors,
same guards. It is NOT guaranteed bitwise: kernel shapes differ between one
long forward and prefix+suffix (the same fp-tolerance class the
row-sharding contract already accepts for reduction order). The config flag
``teacher_prefix_cache: false`` (configs/two_si.yaml) is the escape hatch
if any arm's guard verdict sits near a boundary.

THE OFF-BY-P HAZARD — why this module exists as one seam rather than inline
edits at each teacher forward: ``distill._gather_completion_logits`` derives
completion-position indices from the FULL sequence. A teacher forward that
sees only suffix ids shifts every gather index by P, and an off-by-P there
returns perfectly well-formed logits over the WRONG positions — the KL
still computes, the floors still look plausible, nothing raises (the same
silent failure class as the serve-seam masking hazard). Defences, all in
this file so they cannot drift apart:

- the position offset and the gather offset each come from ONE function
  (:func:`_cache_offset` / :func:`_gather_offset`), monkeypatched by the
  parity mutation tests (tests/test_prefix_cache.py) to prove the parity
  gate actually goes red on a misaligned offset;
- :func:`gather_completion_logits_suffix` REFUSES (never approximates) when
  any completion token sits at or before the prefix boundary — the one
  configuration where the needed logit was never computed;
- the per-batch prefix length is re-derived from THAT batch's rows against
  the stored prefix ids (:meth:`TeacherPrefixCache.match_len`), so a batch
  whose rows do not share the prefix (held-out rows, another SI) can never
  silently reuse a wrong cache — it gets a cropped cache (exact: causal
  attention makes prefix KV at position j depend only on tokens ``0..j``)
  or, when nothing matches, the plain uncached forward.
"""

from __future__ import annotations

import torch

from .traindata import Batch


def _cache_offset(p: int) -> int:
    """The position offset the suffix forward runs at: ``cache_position``
    and ``position_ids`` both start here, so suffix token ``i`` is RoPE- and
    mask-positioned at ``p + i`` — exactly where the full forward put it.
    A SEAM on purpose: the parity mutation test monkeypatches this to
    ``p ± 1`` and asserts the gathered-logit parity test goes red, proving
    the parity gate can detect the silent-misalignment failure class."""
    return p


def _gather_offset(p: int) -> int:
    """The offset the suffix-coordinate completion gather subtracts from
    full-sequence indices. Same seam rationale as :func:`_cache_offset`;
    kept separate because the two offsets are consumed by different code
    (the model call vs the label slice) and the mutation test must be able
    to break their AGREEMENT, not just their shared value."""
    return p


def gather_completion_logits_suffix(
    logits: torch.Tensor, batch: Batch, p: int,
    gather: torch.Tensor | None = None,
) -> torch.Tensor:
    """``distill._gather_completion_logits`` for a suffix forward: logits at
    positions predicting completion tokens → ``[N_comp_tokens, V]``, from
    suffix-width logits (``[B, S-p, V]``, positions ``p..S-1`` of the full
    sequence). Selects EXACTLY the (row, position) pairs the full-width
    gather selects, in the same row-major flatten order — so a flattened
    ``keep`` mask computed against the full-width gather (lockstep's
    ``_probe_pair``, the mask→loss wiring) applies unchanged.

    ``gather`` (optional ``[B, S-1]`` bool, FULL-sequence coordinates)
    overrides the default completion mask, mirroring the full-width
    function's parameter.

    Fail-loudly guard (the off-by-P hazard): a completion token at index
    ``<= p`` is predicted by a logit at ``< p`` that the suffix forward
    never computed. That can only happen when the prefix was derived
    wrongly (it is capped at ``min(prompt_lens) - 1``), so it raises
    rather than returning well-formed logits over the wrong positions."""
    off = _gather_offset(p)
    labels = batch.labels
    if bool((labels[:, : off + 1] != -100).any()):
        raise ValueError(
            f"prefix cache off-by-P guard: a completion token sits at or "
            f"before the prefix boundary (p={p}), so its predicting logit "
            "was never computed by the suffix forward. The prefix length "
            "must be capped at min(prompt_lens) - 1 — refusing to gather "
            "well-formed logits over the wrong positions.")
    if gather is None:
        # labels[:, off+1:][i] is the token at full index off+1+i, predicted
        # by suffix logit i (full position off+i) — the [:, :-1] shift of the
        # full-width gather, re-based to suffix coordinates.
        mask = labels[:, off + 1:] != -100
    else:
        if bool(gather[:, :off].any()):
            raise ValueError(
                f"prefix cache off-by-P guard: the explicit gather mask "
                f"selects a logit position inside the cached prefix "
                f"(p={p}) — the suffix forward never computed it.")
        mask = gather[:, off:]
    return logits[:, :-1][mask]


def completion_token_mask_suffix(batch: Batch, p: int) -> torch.Tensor:
    """``[B, S-p]`` mask of completion-TOKEN positions in suffix
    coordinates — the ACTIVATION-side analogue of
    :func:`gather_completion_logits_suffix` (the positions themselves, not
    the logits predicting them; train_psr's mse loss reads hidden states AT
    completion positions). Same fail-loudly off-by-P guard: a completion
    token inside the prefix has no captured suffix activation."""
    labels = batch.labels
    off = _gather_offset(p)
    if bool((labels[:, :off] != -100).any()):
        raise ValueError(
            f"prefix cache off-by-P guard: a completion token sits inside "
            f"the cached prefix (p={p}), so its activation was never "
            "captured by the suffix forward — refusing the wrong support.")
    return labels[:, off:] != -100


def _kv_pairs(past) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Per-layer (K, V) tensors out of whatever cache object this
    transformers version returned. Three API generations are live across
    the grading host (5.x: ``Cache.layers``) and older cluster/pod envs
    (4.36-4.5x: ``key_cache``/``value_cache`` lists; before that: a legacy
    tuple) — resolved here once so no caller carries version branches."""
    layers = getattr(past, "layers", None)
    if layers is not None:
        return [(l.keys, l.values) for l in layers]
    if hasattr(past, "key_cache"):
        return list(zip(past.key_cache, past.value_cache))
    return [(k, v) for k, v in past]  # legacy tuple-of-tuples


def _make_cache(pairs: list[tuple[torch.Tensor, torch.Tensor]]):
    """A fresh ``DynamicCache`` holding ``pairs`` — built PER BATCH because
    the model APPENDS the suffix KV to whatever cache it is handed (the
    cache object is consumed). The SOURCE prefix tensors are never mutated
    or freed (construction and update both allocate new storage — on
    transformers 5.14.1 construction itself cats against an empty tensor,
    i.e. COPIES the handed-in expand views; the module docstring's MEMORY
    note quantifies it), so they stay valid across batches. Same version
    split as :func:`_kv_pairs`."""
    from transformers import DynamicCache

    try:
        return DynamicCache(ddp_cache_data=pairs)  # >= 4.54 / 5.x
    except TypeError:
        return DynamicCache.from_legacy_cache(tuple(pairs))


class TeacherPrefixCache:
    """The prefix KV of one (SI, group): built ONCE by :meth:`build`, then
    :meth:`forward` replays it per batch. Teacher legs only (forced
    ``no_grad`` — see the module docstring)."""

    def __init__(self, prefix_ids: torch.Tensor,
                 kv: list[tuple[torch.Tensor, torch.Tensor]]):
        self.prefix_ids = prefix_ids  # [P] long, on the model's device
        self._kv = kv                 # per layer: [1, H_kv, P, D_head]

    def __len__(self) -> int:
        return int(self.prefix_ids.numel())

    @classmethod
    def build(cls, model, prefix_ids) -> "TeacherPrefixCache":
        """ONE ``no_grad`` forward of the P prefix tokens (batch 1). The
        returned KV is the PERSISTENT per-group memory cost (~0.77 GB bf16
        at P~5,900 on Llama-3.1-8B GQA); each batch additionally pays the
        transient B-sized copy _batch_cache documents."""
        device = next(model.parameters()).device
        ids = torch.as_tensor(prefix_ids, dtype=torch.long,
                              device=device).reshape(1, -1)
        if ids.numel() < 1:
            raise ValueError("TeacherPrefixCache.build got an empty prefix")
        with torch.no_grad():
            out = model(input_ids=ids,
                        attention_mask=torch.ones_like(ids),
                        use_cache=True)
        return cls(ids[0], _kv_pairs(out.past_key_values))

    def match_len(self, batch: Batch) -> int:
        """The usable prefix length for THIS batch: the longest run of
        tokens every row shares with the stored prefix, capped at
        ``min(prompt_lens) - 1`` so every row keeps at least one prompt
        token in the suffix (the logit predicting the FIRST completion
        token — position ``prompt_len - 1`` — must be computed by the
        suffix forward). Re-derived per batch from the batch's own rows —
        never assumed — so held-out batches, probe batches, or a stray
        batch from another SI are each served exactly the prefix they
        actually share (0 = nothing; the caller falls back to the plain
        forward). 0 also when any prefix column is padding, which the
        right-padded collate makes impossible for real prompt tokens —
        checked rather than trusted."""
        ids = batch.input_ids
        n = min(len(self), ids.shape[1] - 1, int(batch.prompt_lens.min()) - 1)
        if n < 1:
            return 0
        pre = self.prefix_ids[:n].to(ids.device)
        neq = ids[:, :n] != pre.unsqueeze(0)
        if bool(neq.any()):
            first = neq.int().argmax(dim=1)  # first mismatch where any
            has = neq.any(dim=1)
            n = int(torch.where(has, first,
                                torch.full_like(first, n)).min())
        if n < 1:
            return 0
        if not bool(batch.attention_mask[:, :n].all()):
            return 0
        return n

    def _batch_cache(self, batch_size: int, p: int):
        """A fresh per-batch ``DynamicCache`` of the prefix KV, cropped to
        ``p`` (a slice view — exact because causal attention makes prefix
        KV at position j depend only on tokens ``0..j``, so the first p
        entries of a longer prefix's cache ARE the p-prefix's cache) and
        expanded across the batch dim. NOTE the expansion does NOT stay a
        view: transformers 5.14.1 materialises a contiguous batch-sized
        copy at construction (the module docstring's MEMORY note —
        ~B x P x 128 KiB transient per batch on the production model).
        The [1, ...] SOURCE tensors are never copied from in place or
        mutated, so they stay valid across batches either way."""
        return _make_cache([
            (k[:, :, :p].expand(batch_size, -1, -1, -1),
             v[:, :, :p].expand(batch_size, -1, -1, -1))
            for k, v in self._kv
        ])

    def forward_ids(self, model, input_ids: torch.Tensor,
                    attention_mask: torch.Tensor, p: int) -> torch.Tensor:
        """The cached suffix forward on raw tensors (``p >= 1`` — the caller
        derived and validated the prefix length): suffix-width logits
        ``[B, S-p, V]`` for positions ``p..S-1``. The ONE place the model is
        called with a prefix cache, so the position-offset convention (and
        its mutation-test seam, :func:`_cache_offset`) cannot fork between
        the logit path and the capture path. Forced ``no_grad``: teacher
        legs only, by construction."""
        b, s = input_ids.shape
        off = _cache_offset(p)
        # cache_position/position_ids: suffix token i sits at absolute
        # position off+i — identical to the full forward's default
        # arange(S) (right padding gets column-index positions there too;
        # padded rows are excluded by the gather/pool either way).
        pos = torch.arange(off, off + (s - p), device=input_ids.device)
        with torch.no_grad():
            return model(input_ids=input_ids[:, p:],
                         # The full-width mask IS the cached call's mask:
                         # transformers expects past_len + suffix_len
                         # columns, and columns 0..p-1 are real prompt
                         # tokens (all ones) under the right-padded collate
                         # — asserted by match_len / the capture collate.
                         attention_mask=attention_mask,
                         past_key_values=self._batch_cache(b, p),
                         cache_position=pos,
                         position_ids=pos.unsqueeze(0)).logits

    def forward(self, model, batch: Batch) -> tuple[torch.Tensor, int]:
        """One teacher forward over ``batch``, cached where possible.
        Returns ``(logits, p)``: ``p >= 1`` → suffix-width logits
        (``[B, S-p, V]``, positions ``p..S-1``; gather with
        :func:`gather_completion_logits_suffix`); ``p == 0`` → the plain
        full-width forward (nothing cacheable for this batch — an exact
        fallback, not an approximation). Forced ``no_grad``: this is a
        teacher leg by construction."""
        p = self.match_len(batch)
        if p < 1:
            with torch.no_grad():
                logits = model(input_ids=batch.input_ids,
                               attention_mask=batch.attention_mask).logits
            return logits, 0
        return self.forward_ids(model, batch.input_ids,
                                batch.attention_mask, p), p

    def completion_logits(self, model, batch: Batch,
                          gather: torch.Tensor | None = None) -> torch.Tensor:
        """The gathered completion-position teacher logits
        (``[N_comp, V]``) — the drop-in replacement for
        ``_gather_completion_logits(model(...).logits, batch, gather)``
        at every teacher leg. Same tensor semantics, same flatten order,
        fp-tolerance-exact values (see the module docstring)."""
        logits, p = self.forward(model, batch)
        if p == 0:
            # Deferred import: distill imports this module at the top
            # level, so the reverse edge must stay function-local.
            from .distill import _gather_completion_logits

            return _gather_completion_logits(logits, batch, gather=gather)
        return gather_completion_logits_suffix(logits, batch, p,
                                               gather=gather)


def common_prefix_ids(batches: list[Batch]) -> torch.Tensor | None:
    """The longest token prefix EVERY row of EVERY batch shares, capped at
    ``min(prompt_lens) - 1`` across all rows (see
    :meth:`TeacherPrefixCache.match_len` for why the cap) — the prefix a
    group-level :class:`TeacherPrefixCache` should be built from. ``None``
    when there is nothing to cache (no batches, or no shared prefix), so a
    caller can skip the build entirely."""
    if not batches:
        return None
    ref = batches[0].input_ids[0]
    p = min(int(batches[0].prompt_lens[0]) - 1, ref.shape[0] - 1)
    for b in batches:
        if p < 1:
            return None
        n = min(p, b.input_ids.shape[1] - 1, int(b.prompt_lens.min()) - 1)
        if n < 1:
            return None
        pre = ref[:n].to(b.input_ids.device)
        neq = b.input_ids[:, :n] != pre.unsqueeze(0)
        if bool(neq.any()):
            first = neq.int().argmax(dim=1)
            has = neq.any(dim=1)
            n = int(torch.where(has, first,
                                torch.full_like(first, n)).min())
        p = min(p, n)
    if p < 1:
        return None
    return ref[:p]


def prefix_cache_for_batches(model, batches: list[Batch],
                             ) -> TeacherPrefixCache | None:
    """Build the group-level cache for ``batches`` (teacher batches of one
    trainer call), or ``None`` when nothing is cacheable — the single
    helper every trainer's opt-in goes through, so 'flag on but no shared
    prefix' degrades to the exact uncached path everywhere the same way."""
    ids = common_prefix_ids(batches)
    if ids is None:
        return None
    return TeacherPrefixCache.build(model, ids)
