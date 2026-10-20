"""Soft-token serving WITHOUT a hook: vLLM's prompt-embeddings input.

The soft-token kinds (``token`` — extraction/objects.SoftTokenObject;
``token_appended`` — extraction/tokens.AppendedSoftTokenObject) are
UNSERVABLE on the lens path (``lens_adapters.UNSERVABLE_KIND_REASONS``):
their intervention happens at the input-embedding layer, which is not a
decoder block output. But they need no hook AT ALL. The intervention is,
in full: (1) splice a reserved sentinel token into the rendered prompt
(``obj.inject``), and (2) replace that sentinel position's input
embedding with ``scale * param`` (the class's embedding forward hook).
Both steps are functions of the REQUEST, not of the model's forward — so
this module computes the input-embedding sequence client-side and
submits it to vLLM as an embeddings prompt. The intervention is applied
before the engine ever runs: exactly, with no hook, no plugin, no
``enforce_eager`` penalty and no prefix-cache disablement. That makes
these arms servable on the FAST lens-free server.

THE vLLM 0.25.1 INTERFACE (the repo's pin) — every claim here was
established by READING the v0.25.1 source (tag v0.25.1), not by running
it; this box cannot import vllm. File/line citations are into that tag:

- Engine arg: ``enable_prompt_embeds`` (``vllm/config/model.py:255``;
  CLI ``--enable-prompt-embeds``, ``vllm/engine/arg_utils.py:426/849``).
  Default False. The config docstring WARNS the engine may crash on a
  wrong embedding shape ("Only enable this flag for trusted users!") —
  the offline processor validates almost nothing (below), so the
  validation this module performs is load-bearing.
- Offline request shape (``LLM.generate``): an ``EmbedsPrompt`` TypedDict
  (``vllm/inputs/llm.py:125-147``, a member of ``DecoderOnlyPrompt``):
  ``{"prompt_embeds": torch.Tensor}`` of shape ``(seq_len, hidden)``
  (a ``(1, seq_len, hidden)`` batch-of-one is squeezed). The processor
  (``vllm/renderers/base.py:804-832``) REFUSES when the engine was not
  started with the flag ("You must set `--enable-prompt-embeds` ..."),
  requires 2-D after the squeeze, moves the tensor to CPU — and checks
  NEITHER hidden size NOR dtype on the offline path. The runner's
  persistent buffer is in the MODEL dtype and the copy into it casts
  silently (``vllm/v1/worker/gpu_model_runner.py:780-782``), so this
  module emits embeddings ALREADY in the embedding matrix's dtype.
- Mixed mode (ids + mask + embeds): ``EmbedsPrompt`` also takes
  ``prompt_token_ids`` + ``prompt_is_token_ids`` (``vllm/inputs/
  llm.py:130-143``): positions masked True use the real token id (the
  ENGINE embeds them from its own weights); False positions read the
  pre-computed row out of ``prompt_embeds``, which is FULL-LENGTH with
  the True-position rows ignored (the canonical construction zero-fills
  them: ``_build_mixed_prompt_embeds``, ``vllm/renderers/hf.py:192-211``;
  the recombination is ``vllm/v1/worker/gpu_model_runner.py:3540-3548``).
  For a soft token this is the sharper form: only the sentinel row is
  client-supplied, every other embedding comes from the served weights.
- Online request shape (OpenAI-compatible COMPLETIONS API only):
  ``prompt_embeds: bytes | list[bytes]`` (``vllm/entrypoints/openai/
  completion/protocol.py:99``) — a base64-encoded ``torch.save`` payload,
  decoded server-side by ``safe_load_prompt_embeds`` (``vllm/renderers/
  embed_utils.py:16-80``): 2-D, hidden size pinned to the model config,
  floating dtype required, cast to the model dtype. ``echo`` and
  ``prompt_logprobs`` are refused with embeds (``vllm/renderers/
  online_renderer.py:235-240``).

WHAT SURVIVES (the whole point — read, not executed):

- PREFIX CACHING stays ON and stays CORRECT: v0.25.1 folds a per-block
  sha256 of the embedding BYTES into every block hash
  (``_gen_prompt_embeds_extra_hash_keys`` + ``generate_block_hash_extra_
  keys``, ``vllm/v1/core/kv_cache_utils.py:513-535,563-568``). Two
  requests sharing an identical embedding prefix share KV blocks; a
  different trained vector or scale hashes differently, so no cross-arm
  contamination is possible through the cache. (Embeds requests only
  share blocks with other embeds requests: token-id requests carry no
  embeds hash key, so a scale-0 embeds prompt and the same ids submitted
  as a TokensPrompt do not share cache — a throughput note, not a
  correctness one.)
- CUDA GRAPHS stay ON: the V1 GPU model runner feeds the decoder
  ``inputs_embeds`` for every forward when the flag is set, in normal
  execution (``vllm/v1/worker/gpu_model_runner.py:3527-3552``) AND in
  the dummy/capture run (``:5958-5961``) — nothing forces eager. The
  cost is that the embedding gather for token-id positions runs eagerly
  outside the captured graph, engine-wide, for every request on that
  engine (the in-code TODO at ``:3530-3539`` documents exactly this).
- The one engine-level fallback: ``enable_prompt_embeds`` is on the V2
  model runner's unsupported list (``vllm/config/vllm.py:2117-2118``),
  so the engine warns and uses the V1 model runner (``:546,2142``) — a
  performance nuance, not a functional loss.
- Pure-embeds requests have NO real prompt token ids engine-side (the
  request substitutes zeros: ``vllm/v1/request.py:135-139``), so prompt
  echo/detokenization of the prompt is meaningless there — use the mixed
  form when the transcript must show real prompt tokens.

SCALE SEMANTICS (the load-bearing detail — from the class's own hook,
extraction/objects.py::SoftTokenObject.apply):

    if scale == 0.0: return output            # pass through, UNCHANGED
    output[mask] = (scale * self.param).to(output.dtype)   # otherwise

- ``scale != 0``: the sentinel row is REPLACED by ``scale * param`` —
  computed in fp32 (``param`` is stored fp32 by ``save()``) and then
  cast to the embedding dtype. A replacement scaled by ``scale``: not an
  interpolation with, and not an addition to, the native embedding.
  Note the cliff at zero: ``scale=1e-6`` replaces the row with a
  NEAR-ZERO vector (nothing like the native embedding), while
  ``scale=0.0`` (exact float equality, so ``-0.0`` included) restores
  the native row entirely. That cliff is the class's own semantics and
  is reproduced here deliberately.
- ``scale == 0.0``: the hook passes the sentinel's NATIVE embedding
  through (the class's own comment: scale 0 must be the UNSTEERED
  MODEL, bitwise — so it must not zero the row). This module reproduces
  that exactly: at scale 0 the sentinel position keeps the embedding
  matrix's own row for the sentinel id.

THE HONEST NO-OP. Scale 0 is the unsteered model *evaluated on the
INJECTED prompt*: the sentinel is still a real token in the context —
one extra position the model attends over — carrying its native
embedding. That is NOT the same forward as the raw (un-injected) prompt,
and no scale value can make it so. So the no-op LEG for a soft-token arm
comes in two inequivalent flavours, and the caller must pick the one its
contract means:
- "unsteered MODEL, same request": scale 0 through this module (or,
  identically, the injected prompt's token ids as a plain TokensPrompt).
- "the baseline request": the RAW rendered prompt's token ids — do not
  inject and do not call this module at all.
This module never papers over the difference; the tests pin both facts.

STILL REQUIRING GPU VERIFICATION (read-not-executed; nothing in this
module has run against a real vLLM engine — the CPU tests prove the
embedding construction equals the class's own hooked forward bit-exactly
on a tiny HF model, which is this module's whole claim):
1. Every request-format claim cited above, end to end on the pinned
   engine (offline EmbedsPrompt; online base64 completions field).
2. Prefix-cache block sharing between two soft-token requests sharing an
   embeds prefix (the sha256 mechanism was read, not run).
3. CUDA-graph capture + replay with ``--enable-prompt-embeds`` on the
   pinned engine and model.
4. The mixed-mode fields threaded from the OFFLINE ``EmbedsPrompt``
   (their docstrings describe the chat content-part path; the plumbing
   accepts them from offline too, but that route was only read).
5. That the client-side embedding matrix bytes equal the served
   checkpoint's ``embed_tokens`` weights in the served dtype — the
   deployment precondition for exactness of the non-sentinel rows on the
   pure-embeds path (gather from the SAME checkpoint revision; the mixed
   form sidesteps this for all but the sentinel row).
6. The online-path refusals (echo / prompt_logprobs) and behavior of
   returned prompt fields for pure-embeds requests.
"""

from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch


class SoftTokenError(RuntimeError):
    """A soft-token serving refusal (malformed artifact, a sentinel that
    cannot be located, an engine started without the flag) — loud, never
    a silently-wrong embedding sequence."""


#: the two soft-token object kinds (each class's own ``kind`` attribute;
#: pinned against the real classes by tests/test_petri_soft_token.py).
KIND_TOKEN = "token"
KIND_TOKEN_APPENDED = "token_appended"
SOFT_TOKEN_KINDS = (KIND_TOKEN, KIND_TOKEN_APPENDED)

#: the engine precondition, in both spellings (vllm/config/model.py:255;
#: vllm/engine/arg_utils.py:849 at v0.25.1).
ENABLE_PROMPT_EMBEDS_ARG = "enable_prompt_embeds"
ENABLE_PROMPT_EMBEDS_FLAG = "--enable-prompt-embeds"

#: the hook site recorded for these classes when they are fit: the input
#: embedding layer's OUTPUT. Tolerated (and cross-checked) when present
#: in the blob/spec; these kinds carry no other site.
EMBEDDING_SITE = "embedding_out"

#: SoftTokenObject.save writes exactly these keys; ``site`` is tolerated
#: when a fit lane stamps it. Anything else is not this artifact.
_BLOB_REQUIRED_KEYS = frozenset({"kind", "param", "sentinel"})
_BLOB_OPTIONAL_KEYS = frozenset({"site"})


# --- artifact loading + validation (lens_adapters house style) ---------------


@dataclass(frozen=True)
class SoftTokenArtifact:
    """One validated soft-token object: the trained fp32 parameter and
    the sentinel text that carries it. ``param`` is kept fp32 — the
    class's hook multiplies in fp32 and casts the PRODUCT to the
    embedding dtype, and reproducing that order bit-exactly is this
    module's contract."""

    kind: str
    sentinel_text: str
    param: torch.Tensor  # fp32, [d]


def _spec_check(kind: str, spec: Mapping[str, Any], key: str,
                expected: Any) -> None:
    """F9-style cross-check (same rule as lens_adapters._spec_check):
    when the resolved spec carries ``key`` it must agree with the
    artifact's own stored value."""
    if key in spec and spec[key] != expected:
        raise SoftTokenError(
            f"kind {kind!r}: resolved_spec {key}={spec[key]!r} != artifact "
            f"{key}={expected!r} (F9-style cross-check) — the spec and the "
            "bytes disagree; refusing")


def load_soft_token_artifact(kind: str, resolved_spec: Mapping[str, Any],
                             artifact_bytes: bytes,
                             sidecar_bytes: bytes = b"", *,
                             embedding_dim: Optional[int] = None
                             ) -> SoftTokenArtifact:
    """Validate a SoftTokenObject / AppendedSoftTokenObject ``save()``
    blob into a served :class:`SoftTokenArtifact`.

    Same argument shape as a lens adapter (resolved_spec + artifact bytes
    + sidecar bytes) so the caller's plumbing carries the hashed identity
    unchanged — but this is deliberately NOT a lens adapter and is never
    registered with ``lens_adapters.register_adapter`` (these kinds stay
    in ``UNSERVABLE_KIND_REASONS``: they are still not block-output ops;
    this module is their committed path AROUND the lens, not through it).

    ``embedding_dim``: the served model's hidden size, when the caller
    knows it — the parameter must match it exactly."""
    if kind not in SOFT_TOKEN_KINDS:
        raise SoftTokenError(
            f"kind {kind!r} is not a soft-token kind "
            f"(soft-token kinds: {list(SOFT_TOKEN_KINDS)})")
    if not isinstance(resolved_spec, Mapping):
        raise SoftTokenError(
            f"kind {kind!r}: resolved_spec must be a mapping, got "
            f"{type(resolved_spec).__name__}")
    if sidecar_bytes:
        raise SoftTokenError(
            f"kind {kind!r} takes no sidecar bytes, got "
            f"{len(bytes(sidecar_bytes))} — the sidecar is hashed into the "
            "intervention identity and must never be silently dropped; a "
            "behavior-bearing sidecar on this kind means the entry is not "
            "what this loader translates")
    if not artifact_bytes:
        raise SoftTokenError(f"kind {kind!r}: empty artifact bytes")
    try:
        blob = torch.load(io.BytesIO(bytes(artifact_bytes)),
                          map_location="cpu", weights_only=True)
    except Exception as e:  # torch raises several types here
        raise SoftTokenError(
            f"kind {kind!r}: artifact bytes did not load as a torch "
            f"payload: {e}") from e
    if not isinstance(blob, dict) or "kind" not in blob:
        raise SoftTokenError(
            f"kind {kind!r}: artifact is not a torch blob with a 'kind' "
            "field (the load_object contract)")
    if blob["kind"] != kind:
        raise SoftTokenError(
            f"kind {kind!r}: artifact stores kind {blob['kind']!r} — this "
            "cell is mapped to the wrong loader; refusing")
    extra = set(blob) - _BLOB_REQUIRED_KEYS - _BLOB_OPTIONAL_KEYS
    missing = _BLOB_REQUIRED_KEYS - set(blob)
    if extra or missing:
        raise SoftTokenError(
            f"kind {kind!r}: artifact keys {sorted(blob)} are not the "
            f"SoftTokenObject.save schema (required {sorted(_BLOB_REQUIRED_KEYS)},"
            f" optional {sorted(_BLOB_OPTIONAL_KEYS)}) — refusing keys the "
            "loader would silently ignore")
    if "site" in blob and blob["site"] != EMBEDDING_SITE:
        raise SoftTokenError(
            f"kind {kind!r}: artifact site {blob['site']!r} != "
            f"{EMBEDDING_SITE!r} — a soft token lives at the input "
            "embedding output and nowhere else; refusing")
    sentinel = blob["sentinel"]
    if not isinstance(sentinel, str) or not sentinel:
        raise SoftTokenError(
            f"kind {kind!r}: 'sentinel' must be a non-empty string, got "
            f"{sentinel!r}")
    param = blob["param"]
    if not isinstance(param, torch.Tensor) or param.dim() != 1:
        raise SoftTokenError(
            f"kind {kind!r}: 'param' must be a 1-D tensor, got "
            + (f"shape {tuple(param.shape)}" if isinstance(param, torch.Tensor)
               else type(param).__name__))
    if not param.is_floating_point():
        raise SoftTokenError(
            f"kind {kind!r}: 'param' has non-floating dtype {param.dtype}")
    if not torch.isfinite(param).all():
        raise SoftTokenError(
            f"kind {kind!r}: 'param' carries non-finite values — refusing "
            "to serve a corrupt tensor")
    if embedding_dim is not None and param.shape[0] != embedding_dim:
        raise SoftTokenError(
            f"kind {kind!r}: param dim {param.shape[0]} != the served "
            f"model's embedding dim {embedding_dim} — this object was fit "
            "on a different model; refusing")
    _spec_check(kind, resolved_spec, "kind", kind)
    _spec_check(kind, resolved_spec, "sentinel", sentinel)
    _spec_check(kind, resolved_spec, "site", blob.get("site", EMBEDDING_SITE))
    return SoftTokenArtifact(kind=kind, sentinel_text=sentinel,
                             param=param.detach().to(torch.float32))


# --- the splice + the located sentinel ----------------------------------------


def inject_sentinel(kind: str, sentinel_text: str,
                    rendered_prompt: str) -> str:
    """The kind's own sentinel splice, DELEGATED to the real class's
    ``inject`` (called unbound on a carrier) so the serving-side splice
    can never drift from the training-side one: ``token`` inserts right
    after the BOS text (or prepends), ``token_appended`` appends."""
    from sisteer.extraction.objects import SoftTokenObject
    from sisteer.extraction.tokens import AppendedSoftTokenObject

    cls = {KIND_TOKEN: SoftTokenObject,
           KIND_TOKEN_APPENDED: AppendedSoftTokenObject}.get(kind)
    if cls is None:
        raise SoftTokenError(
            f"kind {kind!r} is not a soft-token kind "
            f"(soft-token kinds: {list(SOFT_TOKEN_KINDS)})")
    return cls.inject(SimpleNamespace(sentinel_text=sentinel_text),
                      rendered_prompt)


def _token_ids(tok: Any, text: str, add_special_tokens: bool
               ) -> Tuple[int, ...]:
    out = tok(text, add_special_tokens=add_special_tokens)
    try:
        ids = out["input_ids"]
    except (TypeError, KeyError) as e:
        raise SoftTokenError(
            f"tokenizer returned no 'input_ids' for {text!r}: "
            f"{type(out).__name__}") from e
    if not isinstance(ids, (list, tuple)) or any(
            not isinstance(i, int) or isinstance(i, bool) for i in ids):
        raise SoftTokenError(
            f"tokenizer 'input_ids' must be a flat list of ints, got "
            f"{ids!r}")
    return tuple(ids)


def resolve_sentinel_id(tok: Any, sentinel_text: str) -> int:
    """The sentinel's single token id — the same discipline as
    ``SoftTokenObject.load_blob``: a sentinel that does not tokenize to
    exactly one id cannot carry the embedding."""
    ids = _token_ids(tok, sentinel_text, add_special_tokens=False)
    if len(ids) != 1:
        raise SoftTokenError(
            f"sentinel {sentinel_text!r} is not a single token under this "
            f"tokenizer (got {len(ids)} ids: {list(ids)}) — the trained "
            "embedding occupies exactly one position; refusing")
    return ids[0]


def _locate_sentinel(kind: str, raw_ids: Tuple[int, ...],
                     injected_ids: Tuple[int, ...], sentinel_id: int) -> int:
    """LOCATE the sentinel by matching token ids — never by arithmetic
    from an assumed layout (the response mask's own discipline, for the
    same reason: a rendered prompt can shift, and an index computed from
    an assumption is silently wrong). The injected ids must be the raw
    ids with EXACTLY ONE insertion of the sentinel id; anything else
    (an absent sentinel, a duplicate, a tokenizer that merged or moved a
    neighbour across the splice) refuses."""
    n = injected_ids.count(sentinel_id)
    if n == 0:
        raise SoftTokenError(
            f"kind {kind!r}: sentinel id {sentinel_id} is ABSENT from the "
            "injected prompt's tokenization — the tokenizer swallowed or "
            "rewrote the sentinel; the trained embedding has nowhere to "
            "go; refusing")
    if n > 1:
        raise SoftTokenError(
            f"kind {kind!r}: sentinel id {sentinel_id} appears {n} times "
            "in the injected prompt's tokenization — exactly one is "
            "expected (obj.inject inserts one, and the reserved token "
            "must never occur in natural text); refusing")
    if len(injected_ids) != len(raw_ids) + 1:
        raise SoftTokenError(
            f"kind {kind!r}: injecting the sentinel changed the "
            f"tokenization beyond one insertion ({len(raw_ids)} -> "
            f"{len(injected_ids)} ids) — the splice altered neighbouring "
            "tokens, so the served positions would not be the trained "
            "ones; refusing")
    p = 0
    while p < len(raw_ids) and injected_ids[p] == raw_ids[p]:
        p += 1
    if injected_ids[p] != sentinel_id or injected_ids[p + 1:] != raw_ids[p:]:
        raise SoftTokenError(
            f"kind {kind!r}: the injected tokenization is not the raw "
            "tokenization with one inserted sentinel — neighbouring "
            "tokens changed across the splice; refusing")
    return p


def _check_expected_position(kind: str, k: int, raw_ids: Tuple[int, ...],
                             injected_ids: Tuple[int, ...],
                             injected_prompt: str, sentinel_text: str,
                             tok: Any, add_special_tokens: bool) -> None:
    """The located position must not contradict the object's own
    expectation. ``token_appended``: the sentinel occupies the FINAL
    prompt position ([E_x, e_b] — the class's stated layout).
    ``token``: everything before the sentinel must be exactly the
    tokenization of the string ``inject`` kept ahead of it (the BOS text
    it matched, or nothing) under the SAME tokenization settings — which
    also covers a tokenizer-added BOS. Derived from the injected string
    itself, never from a duplicated BOS list."""
    if kind == KIND_TOKEN_APPENDED:
        if k != len(raw_ids):
            raise SoftTokenError(
                f"kind {kind!r}: sentinel located at position {k} of "
                f"{len(injected_ids)} ids, but the appended soft token "
                "occupies the FINAL prompt position ([E_x, e_b]) — a "
                "tokenizer-appended suffix (or a mangled splice) "
                "contradicts the object's own layout; refusing")
        return
    lead = injected_prompt.split(sentinel_text, 1)[0]
    expected_prefix = _token_ids(tok, lead, add_special_tokens)
    if injected_ids[:k] != expected_prefix:
        raise SoftTokenError(
            f"kind {kind!r}: sentinel located at position {k}, but the "
            f"{k} ids before it {list(injected_ids[:k])} are not the "
            f"tokenization of the splice lead {lead!r} "
            f"{list(expected_prefix)} — the sentinel is not immediately "
            "after the BOS region the object put it after; refusing")


# --- the embedding sequence ----------------------------------------------------


@dataclass(frozen=True)
class SoftTokenEmbeds:
    """One request's embedding sequence, ready for submission.

    ``embeds`` is ``[T, d]`` in the embedding MATRIX's dtype (= the
    served model dtype — the runner buffer casts silently, so the cast
    happens here, exactly where the class's hook does it). At
    ``scale == 0.0`` the sentinel row is the matrix's own native row
    (see SCALE SEMANTICS in the module docstring); otherwise it is
    ``(scale * param_fp32).to(dtype)``, a REPLACEMENT."""

    kind: str
    embeds: torch.Tensor              # [T, d], embedding-matrix dtype
    token_ids: Tuple[int, ...]        # the injected prompt's ids
    sentinel_index: int
    sentinel_id: int
    scale: float
    injected_prompt: str


def build_soft_token_embeds(artifact: SoftTokenArtifact, tok: Any,
                            embed_weight: torch.Tensor,
                            rendered_prompt: str, *, scale: float,
                            add_special_tokens: bool) -> SoftTokenEmbeds:
    """The pure construction: tokenize the injected prompt, LOCATE the
    sentinel, and substitute the trained vector — reproducing the
    class's hooked forward bit-exactly (the equivalence the tests pin on
    a real tiny model).

    ``embed_weight`` is the base model's input-embedding matrix
    ``[vocab, d]`` — for exactness it must be the SERVED checkpoint's
    ``embed_tokens`` weight in the served dtype (module docstring, GPU
    item 5). ``add_special_tokens`` must be passed explicitly and match
    however the surrounding pipeline tokenizes rendered prompts; the
    position cross-check covers a tokenizer-added BOS either way."""
    if not isinstance(artifact, SoftTokenArtifact):
        raise SoftTokenError(
            f"artifact must be a SoftTokenArtifact (from "
            f"load_soft_token_artifact), got {type(artifact).__name__}")
    if not (isinstance(scale, (int, float)) and not isinstance(scale, bool)
            and math.isfinite(float(scale))):
        raise SoftTokenError(
            f"scale must be a finite number, got {scale!r}")
    scale = float(scale)
    if not isinstance(embed_weight, torch.Tensor) or embed_weight.dim() != 2:
        raise SoftTokenError(
            "embed_weight must be the 2-D input-embedding matrix "
            f"[vocab, d], got "
            + (f"shape {tuple(embed_weight.shape)}"
               if isinstance(embed_weight, torch.Tensor)
               else type(embed_weight).__name__))
    if not embed_weight.is_floating_point():
        raise SoftTokenError(
            f"embed_weight has non-floating dtype {embed_weight.dtype}")
    d = artifact.param.shape[0]
    if embed_weight.shape[1] != d:
        raise SoftTokenError(
            f"embed_weight dim {embed_weight.shape[1]} != the trained "
            f"param dim {d} — wrong model for this object; refusing")
    if not isinstance(rendered_prompt, str) or not rendered_prompt:
        raise SoftTokenError(
            f"rendered_prompt must be a non-empty string, got "
            f"{rendered_prompt!r}")
    if artifact.sentinel_text in rendered_prompt:
        raise SoftTokenError(
            f"the rendered prompt already contains the sentinel text "
            f"{artifact.sentinel_text!r} — the reserved token must never "
            "occur in natural text (the class's hook would replace every "
            "occurrence); refusing")

    sentinel_id = resolve_sentinel_id(tok, artifact.sentinel_text)
    raw_ids = _token_ids(tok, rendered_prompt, add_special_tokens)
    if sentinel_id in raw_ids:
        raise SoftTokenError(
            f"the raw prompt's tokenization already contains the sentinel "
            f"id {sentinel_id} — refusing an ambiguous splice")
    injected_prompt = inject_sentinel(artifact.kind, artifact.sentinel_text,
                                      rendered_prompt)
    injected_ids = _token_ids(tok, injected_prompt, add_special_tokens)
    k = _locate_sentinel(artifact.kind, raw_ids, injected_ids, sentinel_id)
    _check_expected_position(artifact.kind, k, raw_ids, injected_ids,
                             injected_prompt, artifact.sentinel_text, tok,
                             add_special_tokens)

    vocab = embed_weight.shape[0]
    if max(injected_ids) >= vocab:
        raise SoftTokenError(
            f"token id {max(injected_ids)} is out of range for the "
            f"embedding matrix ({vocab} rows) — tokenizer/matrix "
            "mismatch; refusing")
    idx = torch.tensor(injected_ids, dtype=torch.long,
                       device=embed_weight.device)
    # a gather of rows — bit-identical to nn.Embedding's F.embedding.
    embeds = embed_weight[idx]
    if scale != 0.0:
        # the class's exact arithmetic and cast ORDER: fp32 product,
        # THEN cast to the embedding dtype (objects.py hook line).
        embeds[k] = (scale * artifact.param.to(embed_weight.device)
                     ).to(embeds.dtype)
    # scale == 0.0: the native row stays — the hook's pass-through.
    # See THE HONEST NO-OP in the module docstring: this is
    # the unsteered MODEL on the INJECTED prompt, not the raw prompt.
    if not torch.isfinite(embeds).all():
        raise SoftTokenError(
            "the built embedding sequence carries non-finite values "
            "(a corrupt embedding matrix, or a scale*param overflow at "
            "this dtype) — the hooked reference would serve the same "
            "corruption silently; this path refuses instead")
    return SoftTokenEmbeds(kind=artifact.kind, embeds=embeds,
                           token_ids=injected_ids, sentinel_index=k,
                           sentinel_id=sentinel_id, scale=scale,
                           injected_prompt=injected_prompt)


# --- submission: the vLLM 0.25.1 request shapes --------------------------------


def require_prompt_embeds_enabled(
        engine_args: Union[Mapping[str, Any], Sequence[str]]) -> None:
    """The engine precondition, CHECKED rather than assumed: refuse at
    submission when the engine was not started with
    ``enable_prompt_embeds`` — vLLM would refuse anyway
    (renderers/base.py:805-808 at v0.25.1), but this names the flag at
    the point that owns the request instead of surfacing an engine error
    after the round-trip.

    ``engine_args``: either the engine-arg mapping the engine was
    constructed with (must carry ``enable_prompt_embeds: True``), or the
    ``vllm serve`` argv (must carry the literal ``--enable-prompt-embeds``
    token, ``=true``-suffixed accepted)."""
    if isinstance(engine_args, Mapping):
        val = engine_args.get(ENABLE_PROMPT_EMBEDS_ARG)
        if val is True:
            return
        raise SoftTokenError(
            f"the engine was not started with {ENABLE_PROMPT_EMBEDS_ARG}="
            f"True (found {val!r}) — prompt-embeds requests need the "
            f"engine arg {ENABLE_PROMPT_EMBEDS_ARG}=True "
            f"(CLI {ENABLE_PROMPT_EMBEDS_FLAG}); refusing at submission")
    if isinstance(engine_args, (list, tuple)):
        for token in engine_args:
            if not isinstance(token, str):
                continue
            if token == ENABLE_PROMPT_EMBEDS_FLAG:
                return
            if token.startswith(ENABLE_PROMPT_EMBEDS_FLAG + "="):
                if token.split("=", 1)[1].lower() in ("1", "true", "yes",
                                                      "on"):
                    return
        raise SoftTokenError(
            f"the server launch args carry no {ENABLE_PROMPT_EMBEDS_FLAG} "
            "— prompt-embeds requests need the engine started with that "
            "flag; refusing at submission")
    raise SoftTokenError(
        "engine_args must be the engine-arg mapping or the launch argv, "
        f"got {type(engine_args).__name__}")


def to_embeds_prompt(built: SoftTokenEmbeds, *,
                     engine_args: Union[Mapping[str, Any], Sequence[str]],
                     mixed: bool = False,
                     cache_salt: Optional[str] = None) -> Dict[str, Any]:
    """The OFFLINE request: an ``EmbedsPrompt`` dict for
    ``LLM.generate`` (vllm/inputs/llm.py:125-147 at v0.25.1).

    ``mixed=False`` (default): the pure form ``{"prompt_embeds": [T, d]}``
    — every row client-supplied.
    ``mixed=True``: real token ids everywhere with ONLY the sentinel
    position masked ``False`` and its row supplied — the engine embeds
    every other position from its own weights (exact by construction at
    every scale, the native sentinel row included at scale 0). The
    non-sentinel rows of the full-length tensor are zero-filled, exactly
    like vLLM's own ``_build_mixed_prompt_embeds``
    (vllm/renderers/hf.py:192-211): the engine overwrites True-position
    rows, and deterministic filler keeps prefix-cache hashes stable.
    Mixed mode from the offline path is GPU-verification item 4."""
    require_prompt_embeds_enabled(engine_args)
    prompt: Dict[str, Any]
    if not mixed:
        prompt = {"prompt_embeds": built.embeds}
    else:
        k = built.sentinel_index
        full = torch.zeros_like(built.embeds)
        full[k] = built.embeds[k]
        mask: List[bool] = [True] * len(built.token_ids)
        mask[k] = False
        prompt = {"prompt_embeds": full,
                  "prompt_token_ids": list(built.token_ids),
                  "prompt_is_token_ids": mask}
    if cache_salt is not None:
        if not isinstance(cache_salt, str) or not cache_salt:
            raise SoftTokenError(
                f"cache_salt must be a non-empty string, got {cache_salt!r}")
        prompt["cache_salt"] = cache_salt
    return prompt


def encode_prompt_embeds_for_openai(
        built: SoftTokenEmbeds, *,
        engine_args: Union[Mapping[str, Any], Sequence[str]]) -> str:
    """The ONLINE request payload: the base64-encoded ``torch.save``
    bytes for the COMPLETIONS API's ``prompt_embeds`` field
    (vllm/entrypoints/openai/completion/protocol.py:99; decoded
    server-side by renderers/embed_utils.py:16-80 at v0.25.1 — which
    pins hidden size to the model config, requires a floating dtype, and
    casts to the model dtype). Pure-embeds only: the mixed fields are
    chat content-part machinery online. ``echo`` and ``prompt_logprobs``
    are refused by the server with embeds
    (renderers/online_renderer.py:235-240)."""
    require_prompt_embeds_enabled(engine_args)
    buf = io.BytesIO()
    torch.save(built.embeds.cpu(), buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")
