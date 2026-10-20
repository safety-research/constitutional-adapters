"""Tier 0 probes.

Four probe families, all bespoke, in-process, teacher-forced except the
canary:

* **Windowed raw NLL** (§4.2): 1024-token windows over the pinned text
  corpora; per-window ``nll_sum`` + ``token_count`` rows so aggregates
  recompute Σnum/Σden from rows (F14).
* **Assistant-masked chat NLL** (§4.2): content tokens ONLY — the
  terminal EOT and all header/user/system scaffold tokens are excluded.
  Spans are located by the marker primitives in
  :mod:`sisteer.chat_markers`: each span must sit immediately after the
  probe-derived assistant HEADER marker (where one is locatable) and
  end exactly where the probe-derived CLOSING scaffold begins — never
  derived by subtracting two renders (the NousResearch template
  appends the assistant header unconditionally, which made the old
  suffix arithmetic swallow the terminal EOT + trailing header).
  Boundary retokenization or any other structural surprise raises
  :class:`MaskError` instead of guessing (F11).
* **Coarsened KL(base‖ablation)** (§4.3 as revised): float64 log-space
  atoms end-to-end, top-k atom sets with deterministic ascending-id tie
  break, tail LOG-mass via masked logsumexp, NO floors or clamps
  anywhere (F4). A ``q_tail = −inf`` position with ``p_tail > 0``
  makes the corpus KL **+inf** — never excluded, never capped — and
  emits an automatic flag row plus an inf-position count row. The
  statistic is the exact KL of the induced (k+1)-cell partition, a true
  lower bound on full KL by the data-processing inequality: reported as
  ``kl_coarse_lb``, never as "KL". Since 2026-08-18 the magnitude is an
  ANNOTATION — it feeds no red flag, no screen, and no verdict (the old
  ``flag_kl_gt_kappa``/``dist_shift`` path is gone); its remaining roles
  are the no-op exact-zero certificate, reported magnitudes, and triage.
* **Coherence canary** (§4.4): 300 pinned prompts, greedy generation,
  five indicators; thresholds are OWNED BY CALIBRATION (never hardcoded
  here); ≥2 tripped indicators fail the canary, and a failure at
  attempt 0 requests a fresh-subprocess confirmation rerun — the DRIVER
  owns the process mechanics (§11.1 re-exec), this module owns the
  state machine.
* **KL localization** (§4.5, display-only): base atoms for the
  target-relevant slice are built IN THE ARM'S OWN CELL with hooks
  disabled (arm=None ⇒ no steering ctx ⇒ exactly the base model),
  cached per-arm under the same digest scheme; trial 0 is not
  involved. Skipping (no target-relevant corpus declared) is a
  recorded notice row, never a silent omission.

Cache addressing (F5/r2-F5): ``<cache_dir>/<digest16>.safetensors`` +
``<digest16>.json`` where digest = sha256 over exactly the sidecar field
list :data:`CACHE_SIDECAR_FIELDS` — model_revision and engine included,
nothing in the sidecar sits outside the digest. Atomic publish = tmp
FILES in the destination directory + two plain file renames, sidecar
FIRST, then the ``.safetensors`` — a consumer that sees the tensors file
always finds its sidecar in place (r2-new-3). Consumers re-verify every
sidecar field against their own resolved spec and refuse on mismatch.

Base lane only: torch / safetensors / the sisteer core — no evalplus,
no vllm.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import torch
from safetensors.torch import load_file, save_file

from sisteer.chat_markers import (
    ChatMarkerError,
    assistant_content_spans,
    control_token_ids,
    derive_assistant_markers,
    divergence as _divergence,
    render_ids as _render_ids,
)
from sisteer.capeval.manifest import canonical_json, sha256_hex16
from sisteer.capeval.store import ResultRow
# the metric-name contract shared with the stats module (κ derivation reads
# per-window METRIC_KL rows; canary thresholds read the per-indicator
# aggregate rows at CANARY_AGG_ITEM)
from sisteer.capeval.stats.noninferiority import (
    CANARY_AGG_ITEM,
    CANARY_DIRECTIONS,
    METRIC_KL,
    METRIC_NLL_SUM,
    METRIC_TOKEN_COUNT,
)
from sisteer.eval.capability import distinct_ngram_ratio
from sisteer.generate import encode_kwargs
from sisteer.models import build_chat_prompt

# Bumped whenever the Tier-0 math / windowing / masking conventions
# change — it is a KL-cache digest input, so a bump orphans stale caches
# instead of silently reusing them. Window geometry (window_len, BOS
# convention, masking policy) rides on this constant: change any of it,
# bump this.
TIER0_CODE_VERSION = 1

#: The frozen cache digest field list == the sidecar field list (r2-F5).
#: Nothing recorded in the sidecar sits outside the digest.
CACHE_SIDECAR_FIELDS = (
    "model_revision",
    "engine",
    "corpus_id",
    "corpus_revision",
    "slice_hash",
    "tokenizer_hash",
    "dtype",
    "top_k",
    "code_version",
)


class Tier0Error(RuntimeError):
    """A Tier-0 refusal (loud, never a silent fallback)."""


class MaskError(Tier0Error):
    """Assistant-mask boundary retokenization: the incremental render is
    not a token prefix of the fuller render — the run fails loudly
    rather than guessing a mask (F11)."""


class CacheMismatchError(Tier0Error):
    """A KL atom cache whose sidecar does not match the consumer's
    resolved spec (F5)."""


# ---------------------------------------------------------------------------
# Scored-corpus containers (the `corpus` type — engines
# duck-type: they only touch .items / .n_positions and each item's fields)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoredItem:
    """One teacher-forced scoring unit (a window / conversation / prompt).

    ``score_positions[j] = i`` means token ``input_ids[i]`` is scored,
    conditioned on ``input_ids[:i]`` — every i >= 1."""

    item_id: str
    input_ids: Tuple[int, ...]
    score_positions: Tuple[int, ...]


@dataclass(frozen=True)
class WindowedCorpus:
    """A tokenized, windowed corpus in CANONICAL order (the order every
    KL atom cache is stored in)."""

    name: str
    items: Tuple[ScoredItem, ...]

    @property
    def n_positions(self) -> int:
        return sum(len(it.score_positions) for it in self.items)


@dataclass(frozen=True)
class AtomScores:
    """Per-position §4.3 atom quantities (the ``atom_probs`` member of
    the engine protocol's PositionScore). Log-space, float64 end-to-end
    (r2-F4): ``ids`` int32 [k], ``logp`` float64 [k], ``logp_tail`` a
    python float (the masked-logsumexp tail LOG-mass)."""

    ids: torch.Tensor
    logp: torch.Tensor
    logp_tail: float


# ---------------------------------------------------------------------------
# Windowing (raw text)
# ---------------------------------------------------------------------------


def windows_from_texts(
    tok,
    name: str,
    texts: Sequence[str],
    window_len: int = 1024,
    token_budget: Optional[int] = None,
    min_window_tokens: int = 32,
) -> WindowedCorpus:
    """1024-token windows over the concatenated document token stream
    (documents tokenized independently, concatenated in slice order).

    Each window is prefixed with BOS (when the tokenizer has one) as an
    UNscored conditioning token, so every window token is scored:
    ``token_count == len(window)``. A trailing fragment shorter than
    ``min_window_tokens`` is dropped (deterministic). Window geometry is
    part of TIER0_CODE_VERSION."""
    stream: List[int] = []
    for t in texts:
        stream.extend(tok(t, add_special_tokens=False)["input_ids"])
        if token_budget is not None and len(stream) >= token_budget:
            break
    if token_budget is not None:
        stream = stream[:token_budget]
    bos = getattr(tok, "bos_token_id", None)
    items: List[ScoredItem] = []
    for w, start in enumerate(range(0, len(stream), window_len)):
        chunk = stream[start:start + window_len]
        if len(chunk) < min_window_tokens:
            break
        ids = ([bos] if bos is not None else []) + chunk
        items.append(ScoredItem(
            item_id=f"{name}/w{w:05d}",
            input_ids=tuple(ids),
            # position 0 (BOS, or the first stream token without one) is
            # the unscored conditioning token
            score_positions=tuple(range(1, len(ids))),
        ))
    return WindowedCorpus(name=name, items=tuple(items))


# ---------------------------------------------------------------------------
# Assistant-content mask (chat NLL, §4.2 / F11)
# ---------------------------------------------------------------------------


def assistant_content_mask(
    tok, messages: Sequence[Mapping[str, str]]
) -> Tuple[List[int], List[int]]:
    """(full-conversation token ids, sorted assistant CONTENT positions).

    Spans come from :func:`sisteer.chat_markers.assistant_content_spans`,
    which LOCATES structure instead of subtracting renders: the
    probe-derived assistant HEADER marker must sit immediately before each
    span (where the template structure lets one be located), and the span
    ends exactly where the probe-derived CLOSING scaffold begins — verified
    token-for-token — so it can never contain the terminal EOT, trailing
    glue, or an unconditional trailing header, and (with the control-token
    scan) can never cross a turn boundary. Rationale: the NousResearch
    Llama-3.1-8B-Instruct template appends
    ``<|start_header_id|>assistant<|end_header_id|>\n\n`` UNCONDITIONALLY
    (``add_generation_prompt`` is never consulted), so the old
    prefix/suffix arithmetic placed the terminal ``<|eot_id|>`` plus that
    trailing header inside the CONTENT span — silently diluting the
    teacher-forced NLL toward the scaffold's near-zero loss (and the
    coarsened KL rode on the same mask). Any structural surprise raises
    :class:`MaskError` (F11: fail loudly, never guess); the mask keeps
    CONTENT tokens only and an empty-content assistant turn
    contributes no positions.

    Defense in depth, independent of the span derivation: if any CONTROL
    token id survives inside a computed span, raise :class:`MaskError`
    naming the offending tokens rather than return it. The control set is
    ``all_special_ids`` UNION the added-special tokens — on the Llama-3.1
    mirrors ``all_special_ids`` is only ``[<|begin_of_text|>,
    <|eot_id|>]``, so a specials-only guard would wave
    ``<|start_header_id|>``/role/``<|end_header_id|>`` scaffold straight
    through (the measured empty-assistant corruption was four scaffold
    positions containing zero special ids)."""
    try:
        markers = derive_assistant_markers(tok) if any(
            m.get("role") == "assistant" for m in messages) else None
        full, spans = assistant_content_spans(tok, messages, markers=markers)
    except ChatMarkerError as e:
        raise MaskError(f"{e} (F11/D7)") from e
    ctrl = control_token_ids(tok)
    positions: List[int] = []
    for lo, hi in spans:
        bad = [p for p in range(lo, hi) if full[p] in ctrl]
        if bad:
            names = [(p, tok.convert_ids_to_tokens(full[p])) for p in bad]
            raise MaskError(
                f"assistant content span [{lo}, {hi}) still contains "
                f"control tokens {names!r} after span derivation; refusing "
                "a scaffold-contaminated mask (F11/D7)")
        positions.extend(range(lo, hi))
    return full, positions


def windows_from_conversations(
    tok,
    name: str,
    conversations: Sequence[Sequence[Mapping[str, str]]],
    item_ids: Sequence[str],
    assistant_token_budget: Optional[int] = None,
    max_tokens: int = 4096,
) -> WindowedCorpus:
    """One ScoredItem per conversation (the §4.2 bootstrap unit): scored
    positions are the assistant CONTENT tokens, truncated at
    ``max_tokens``. Conversations are consumed in slice order until the
    assistant-token budget is met."""
    if len(conversations) != len(item_ids):
        raise Tier0Error("windows_from_conversations: item_ids must align "
                         "with conversations")
    items: List[ScoredItem] = []
    budget_used = 0
    for conv, iid in zip(conversations, item_ids):
        full, positions = assistant_content_mask(tok, conv)
        positions = [p for p in positions if 1 <= p < max_tokens]
        if not positions:
            continue
        items.append(ScoredItem(
            item_id=iid,
            input_ids=tuple(full[:max_tokens]),
            score_positions=tuple(positions),
        ))
        budget_used += len(positions)
        if assistant_token_budget is not None and budget_used >= assistant_token_budget:
            break
    return WindowedCorpus(name=name, items=tuple(items))


def windows_from_prompts(
    tok,
    name: str,
    prompts: Sequence[str],
    item_ids: Sequence[str],
    max_tokens: int = 4096,
) -> WindowedCorpus:
    """One ScoredItem per chat-rendered prompt (the §4.5 target-relevant
    slice): every position of the rendered prompt is scored."""
    if len(prompts) != len(item_ids):
        raise Tier0Error("windows_from_prompts: item_ids must align with prompts")
    items = []
    for p, iid in zip(prompts, item_ids):
        rendered = build_chat_prompt(tok, p)
        ids = tok(rendered, **encode_kwargs(tok, rendered))["input_ids"][:max_tokens]
        if len(ids) < 2:
            continue
        items.append(ScoredItem(
            item_id=iid,
            input_ids=tuple(int(i) for i in ids),
            score_positions=tuple(range(1, len(ids))),
        ))
    return WindowedCorpus(name=name, items=tuple(items))


# ---------------------------------------------------------------------------
# KL atom cache (digest-keyed, atomically published — §4.3/F5)
# ---------------------------------------------------------------------------


def tokenizer_hash(tok) -> str:
    """Content hash of the tokenizer: vocab + chat template + special
    tokens. A cache digest input — two tokenizers that tokenize
    differently can never share atom caches."""
    payload = {
        "vocab": {str(k): int(v) for k, v in tok.get_vocab().items()},
        "chat_template": getattr(tok, "chat_template", None) or "",
        "special_tokens": sorted(str(t) for t in tok.all_special_tokens),
    }
    return sha256_hex16(canonical_json(payload).encode("utf-8"))


def cache_spec(
    *,
    model_revision: str,
    engine: str,
    corpus_id: str,
    corpus_revision: str,
    slice_hash: str,
    tokenizer_hash: str,
    dtype: str,
    top_k: int,
    code_version: int = TIER0_CODE_VERSION,
) -> Dict[str, Any]:
    """The resolved cache spec — keys are exactly CACHE_SIDECAR_FIELDS
    (keyword-only so a positional slip cannot scramble the digest)."""
    return {
        "model_revision": model_revision,
        "engine": engine,
        "corpus_id": corpus_id,
        "corpus_revision": corpus_revision,
        "slice_hash": slice_hash,
        "tokenizer_hash": tokenizer_hash,
        "dtype": dtype,
        "top_k": int(top_k),
        "code_version": int(code_version),
    }


def cache_digest(spec: Mapping[str, Any]) -> str:
    """16-hex digest over EXACTLY the sidecar field list (r2-F5): extra
    or missing keys are refusals, so nothing can ride in the sidecar
    outside the digest or vice versa."""
    extra = sorted(set(spec) - set(CACHE_SIDECAR_FIELDS))
    missing = sorted(set(CACHE_SIDECAR_FIELDS) - set(spec))
    if extra or missing:
        raise Tier0Error(
            f"cache spec keys must be exactly {CACHE_SIDECAR_FIELDS}: "
            f"extra={extra} missing={missing}")
    return sha256_hex16(
        canonical_json({k: spec[k] for k in CACHE_SIDECAR_FIELDS}).encode("utf-8"))


def cache_paths(cache_dir: Union[str, Path], spec: Mapping[str, Any]) -> Tuple[Path, Path]:
    digest = cache_digest(spec)
    d = Path(cache_dir)
    return d / f"{digest}.safetensors", d / f"{digest}.json"


def publish_atom_cache(
    cache_dir: Union[str, Path],
    spec: Mapping[str, Any],
    ids: torch.Tensor,
    logp: torch.Tensor,
    logp_tail: torch.Tensor,
) -> Path:
    """Atomic publish, serialized per digest:

    * publication holds an exclusive ``flock`` on ``<digest>.lock`` in
      the destination directory, so concurrent arm cells that computed
      the same slice can never interleave renames (parallel array cells
      begin the same base pass together);
    * a writer that takes the lock and finds the cache already complete
      REUSES it — the winner's files are never clobbered;
    * temporary files are WRITER-UNIQUE (pid + random token), so no
      process can rename another's still-open tmp;
    * then the r2-new-3 discipline: two plain file renames, the sidecar
      FIRST, then the ``.safetensors`` — never a directory rename. A
      consumer that sees the tensors file therefore always finds its
      sidecar already in place. Arrays are stored float64 end-to-end
      (r2-F4). The ``.lock`` file is left behind deliberately (unlinking
      it would race the next locker)."""
    st_path, sc_path = cache_paths(cache_dir, spec)
    k = int(spec["top_k"])
    if ids.dtype != torch.int32 or logp.dtype != torch.float64 \
            or logp_tail.dtype != torch.float64:
        raise Tier0Error(
            f"atom cache dtypes must be ids=int32/logp=float64/tail=float64, "
            f"got {ids.dtype}/{logp.dtype}/{logp_tail.dtype} (r2-F4)")
    if ids.shape != logp.shape or ids.dim() != 2 or ids.shape[1] != k \
            or logp_tail.shape != (ids.shape[0],):
        raise Tier0Error(
            f"atom cache shapes must be ids/logp [N,{k}] and tail [N], got "
            f"{tuple(ids.shape)}/{tuple(logp.shape)}/{tuple(logp_tail.shape)}")
    st_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = st_path.with_name(st_path.stem + ".lock")
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            if st_path.exists() and sc_path.exists():
                return st_path  # a concurrent writer won: reuse, never clobber
            token = f"{os.getpid()}-{uuid.uuid4().hex[:8]}"
            sc_tmp = sc_path.with_name(f"{sc_path.name}.{token}.tmp")
            st_tmp = st_path.with_name(f"{st_path.name}.{token}.tmp")
            sc_tmp.write_text(
                json.dumps({key: spec[key] for key in CACHE_SIDECAR_FIELDS},
                           indent=1, sort_keys=True), encoding="utf-8")
            save_file({"ids": ids.contiguous(), "logp": logp.contiguous(),
                       "logp_tail": logp_tail.contiguous()}, str(st_tmp))
            os.replace(sc_tmp, sc_path)  # sidecar first (r2-new-3)
            os.replace(st_tmp, st_path)
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
    return st_path


def atom_cache_exists(cache_dir: Union[str, Path], spec: Mapping[str, Any]) -> bool:
    st_path, _ = cache_paths(cache_dir, spec)
    return st_path.exists()


def load_atom_cache(
    cache_dir: Union[str, Path],
    spec: Mapping[str, Any],
    expected_positions: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load + VERIFY a cache against the consumer's own resolved spec:
    every sidecar field must equal the spec's (refusal naming both
    values), the sidecar key set must be exactly the digest field list,
    and — when given — the position count must match the corpus (a
    window-geometry drift alarm)."""
    st_path, sc_path = cache_paths(cache_dir, spec)
    if not st_path.exists():
        raise Tier0Error(
            f"KL atom cache missing: {st_path} — the base atom cache is "
            "built by calibration trial 0 (§4.3) or, for target-relevant "
            "slices, by the arm's own cell (§4.5); refusing to proceed "
            "without it")
    if not sc_path.exists():
        raise CacheMismatchError(
            f"{st_path} exists but its sidecar {sc_path.name} does not — "
            "publish-order violation (sidecar must land first, r2-new-3); "
            "delete the tensors file and rebuild")
    sidecar = json.loads(sc_path.read_text(encoding="utf-8"))
    if set(sidecar) != set(CACHE_SIDECAR_FIELDS):
        raise CacheMismatchError(
            f"{sc_path}: sidecar keys {sorted(sidecar)} != digest field list "
            f"{sorted(CACHE_SIDECAR_FIELDS)} (r2-F5)")
    for k in CACHE_SIDECAR_FIELDS:
        if sidecar[k] != spec[k]:
            raise CacheMismatchError(
                f"{sc_path}: sidecar {k!r} = {sidecar[k]!r} but the resolved "
                f"spec says {spec[k]!r} — refusing the cache (F5)")
    tensors = load_file(str(st_path))
    ids, logp, tail = tensors["ids"], tensors["logp"], tensors["logp_tail"]
    if expected_positions is not None and ids.shape[0] != expected_positions:
        raise CacheMismatchError(
            f"{st_path}: cache has {ids.shape[0]} positions but the corpus "
            f"windows to {expected_positions} — window geometry drifted "
            "without a TIER0_CODE_VERSION bump; refusing")
    return ids, logp, tail


# ---------------------------------------------------------------------------
# Coarsened-KL math (float64, no clamps — §4.3/F4)
# ---------------------------------------------------------------------------


def kl_coarse_window(
    logp: torch.Tensor,
    logp_tail: torch.Tensor,
    logq: torch.Tensor,
    logq_tail: torch.Tensor,
) -> torch.Tensor:
    """Per-position coarsened KL over one window (float64 [n,k] atoms +
    [n] tails) -> float64 [n].

    KL_t = Σ_j exp(logp_j)·(logp_j − logq_j) + exp(logp_tail)·(logp_tail
    − logq_tail). Terms with p_j = 0 (logp = −inf) contribute 0. A
    position with q-mass 0 where p-mass > 0 (atom or tail) is +inf —
    NEVER capped (F4). NaN anywhere is a hard numerical failure and a
    refusal: NaN compares false against κ, so letting it through would
    silently evade the distribution-shift red flag — −inf/+inf stay
    legitimate, NaN never is."""
    for t in (logp, logp_tail, logq, logq_tail):
        if t.dtype != torch.float64:
            raise Tier0Error(f"kl_coarse_window takes float64, got {t.dtype} (F4)")
        if torch.isnan(t).any():
            raise Tier0Error(
                "kl_coarse_window: NaN in atom log-probabilities — a "
                "numerically destroyed forward must refuse the cell, not "
                "evade the κ red flag (r1 finding 10)")
    p = torch.exp(logp)
    zero = torch.zeros((), dtype=torch.float64)
    atom_terms = torch.where(p == 0, zero, p * (logp - logq))
    total = atom_terms.sum(dim=-1)
    p_tail = torch.exp(logp_tail)
    tail_term = torch.where(p_tail == 0, zero, p_tail * (logp_tail - logq_tail))
    return total + tail_term


def kl_coarse_position(
    logp: Sequence[float], logp_tail: float,
    logq: Sequence[float], logq_tail: float,
) -> float:
    """Scalar convenience wrapper over kl_coarse_window (golden tests)."""
    out = kl_coarse_window(
        torch.tensor([list(logp)], dtype=torch.float64),
        torch.tensor([logp_tail], dtype=torch.float64),
        torch.tensor([list(logq)], dtype=torch.float64),
        torch.tensor([logq_tail], dtype=torch.float64),
    )
    return float(out[0])


# ---------------------------------------------------------------------------
# Coherence canary (§4.4): indicators, calibration-owned thresholds,
# ≥2-indicator rule, confirmation state machine
# ---------------------------------------------------------------------------

#: The five §4.4 indicators, in their reporting order — derived from the
#: stats side's CANARY_DIRECTIONS so the indicator set and each one's
#: harmful direction have exactly one authoritative copy.
CANARY_INDICATORS = tuple(CANARY_DIRECTIONS)

#: +1 = larger is worse, -1 = smaller is worse. nonascii_frac is
#: base-RELATIVE (§4.4: a French-style target legitimately shifts the
#: absolute value): its value is compared as a delta vs the calibration
#: base reference.
CANARY_HARMFUL_SIGN = {
    ind: (+1 if d == "up" else -1) for ind, d in CANARY_DIRECTIONS.items()
}

CANARY_PASS = "pass"
CANARY_NEEDS_CONFIRMATION = "needs_confirmation"
CANARY_CONFIRMED_FAIL = "confirmed_fail"
CANARY_NOT_REPRODUCED = "not_reproduced"
#: Attempt 1 tripped >=2 indicators but canary confirmation was NOT
#: requested (the attempt-0 red flag was another probe's): NEW
#: information, filed as an UNCONFIRMED flag row — never *_confirmed.
CANARY_NEW_UNCONFIRMED = "new_unconfirmed"


def max_consecutive_ngram_run(tokens: Sequence[Any],
                              ns: Sequence[int] = (1, 2, 3)) -> int:
    """max over n of the longest consecutive-repeat run of any token
    n-gram: 'a b a b a b' -> 3 at n=2 (the gram (a,b) occurs 3 times
    back-to-back). Consecutive repeats sit n gram-positions apart, so
    each offset chain r, r+n, r+2n, ... is walked separately (a merged
    scan would double-count interleaved chains). A healthy text scores
    1."""
    if not tokens:
        return 0
    best = 0
    for n in ns:
        if len(tokens) < n:
            continue
        grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
        best_n = 1
        for r in range(min(n, len(grams))):
            run = 1
            for i in range(r + n, len(grams), n):
                if grams[i] == grams[i - n]:
                    run += 1
                    best_n = max(best_n, run)
                else:
                    run = 1
        best = max(best, best_n)
    return best


def nonascii_frac(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ord(c) > 127) / len(text)


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, math.ceil(0.95 * len(s)) - 1)
    return float(s[max(idx, 0)])


def canary_gen_stats(out: Any, *, abrupt_min_tokens: int = 8,
                     rep_ns: Sequence[int] = (1, 2, 3)) -> Dict[str, Any]:
    """Per-generation §4.4 stats from one GenOut — THE one per-prompt
    implementation (canary_values and the production path must never
    disagree). Semantics:

    * repetition and content length come from the RAW generated token
      ids (``gen_token_ids``), never a re-tokenization of the decoded
      text — a special-token degeneration (256 BOS tokens) decodes to
      empty text but is fully visible in the ids;
    * the terminal EOS is excluded from the content tokens, so seven
      content tokens + EOS count as 7, not 8;
    * ``abrupt`` requires ``ended_with_eos`` — a short generation that
      merely hit max_new_tokens is not an abrupt EOS."""
    ids = [int(t) for t in out.gen_token_ids]
    content = ids[:-1] if (out.ended_with_eos and ids) else ids
    return {
        "n_tokens": len(content),
        "n_gen": len(out.gen_token_logprobs),
        "max_rep_run": max_consecutive_ngram_run(content, rep_ns),
        "nonascii_frac": nonascii_frac(out.text),
        "abrupt": int(bool(out.ended_with_eos)
                      and len(content) < abrupt_min_tokens),
        "ended_with_eos": int(out.ended_with_eos),
        "logprob_sum": float(sum(out.gen_token_logprobs)),
        "text": out.text,
    }


def canary_values_from_stats(stats: Sequence[Mapping[str, Any]]
                             ) -> Dict[str, float]:
    """The five §4.4 indicator values from per-generation stats
    (:func:`canary_gen_stats` rows — live or recovered from the store's
    extra_json on resume). ONE implementation, used by both
    :func:`canary_values` and the production ``_run_canary`` path
    (simplify item 4). nonascii_frac is the unweighted mean of
    per-prompt fractions (deterministic and resume-stable — the
    per-prompt fractions ride in the stat rows)."""
    if not stats:
        raise Tier0Error("canary_values_from_stats: no per-prompt stats")
    return {
        "max_rep_run_p95": _p95([float(s["max_rep_run"]) for s in stats]),
        "distinct3": distinct_ngram_ratio([s["text"] for s in stats], n=3),
        "mean_tok_logprob": (
            sum(float(s["logprob_sum"]) for s in stats)
            / max(sum(int(s["n_gen"]) for s in stats), 1)),
        "nonascii_frac": (sum(float(s["nonascii_frac"]) for s in stats)
                          / len(stats)),
        "abrupt_eos_rate": sum(int(s["abrupt"]) for s in stats) / len(stats),
    }


def canary_values(
    outs: Sequence[Any],
    abrupt_min_tokens: int = 8,
    rep_ns: Sequence[int] = (1, 2, 3),
) -> Dict[str, float]:
    """The five §4.4 indicator values for one run of generations
    (``outs``: GenOut-shaped objects). A thin composition of
    :func:`canary_gen_stats` + :func:`canary_values_from_stats`, so this
    test-covered entry point and the production path share one
    implementation exactly (simplify item 4)."""
    return canary_values_from_stats([
        canary_gen_stats(o, abrupt_min_tokens=abrupt_min_tokens,
                         rep_ns=rep_ns) for o in outs])


@dataclass(frozen=True)
class CanaryThresholds:
    """Calibration-owned fail lines (§4.4/§7.1 — derived from the base
    self-runs, NEVER hardcoded here). ``fail_lines[ind]`` is compared in
    the indicator's harmful direction. ``nonascii_base`` implements the
    §4.4 base-relative convention for nonascii_frac (the value is
    compared as a delta vs this reference); v0 calibration derives its
    lines on the absolute scale — worst-base-plus-spread, which is
    base-derived — so the adapter sets the reference to 0.0."""

    fail_lines: Dict[str, float]
    nonascii_base: float = 0.0

    def __post_init__(self):
        missing = sorted(set(CANARY_INDICATORS) - set(self.fail_lines))
        if missing:
            raise Tier0Error(f"CanaryThresholds missing indicators {missing}")

    @classmethod
    def from_calibration(cls, lines: Mapping[str, Mapping[str, float]]
                         ) -> "CanaryThresholds":
        """Adapter over stats.calibrate.canary_thresholds_from_values
        output ({indicator: {fail_line, worst_base, null_spread_p99}})."""
        return cls(fail_lines={k: float(v["fail_line"])
                               for k, v in lines.items()})


def derive_canary_thresholds(
    base_runs: Sequence[Mapping[str, float]],
    reference_index: int = 0,
) -> CanaryThresholds:
    """Thresholds from k base self-runs. Derivation is OWNED BY
    calibration (stats.calibrate.canary_thresholds_from_values — worst
    base value + p99 null spread in the harmful direction, the
    reference-contrast structure); this is the consuming adapter, kept
    here so the gate and its threshold shape live together."""
    from sisteer.capeval.stats.calibrate import canary_thresholds_from_values

    for r in base_runs:
        missing = sorted(set(CANARY_INDICATORS) - set(r))
        if missing:
            raise Tier0Error(f"base run missing indicators {missing}")
    per_trial = {ind: [float(r[ind]) for r in base_runs]
                 for ind in CANARY_INDICATORS}
    lines = canary_thresholds_from_values(per_trial,
                                          reference_index=reference_index)
    return CanaryThresholds.from_calibration(lines)


def tripped_indicators(values: Mapping[str, float],
                       thresholds: CanaryThresholds) -> Tuple[str, ...]:
    """Indicators past their fail line, harmful direction only. A NaN
    indicator value is a refusal: NaN compares false against every
    fail line, so letting it through would silently pass a numerically
    destroyed run."""
    out = []
    for ind in CANARY_INDICATORS:
        v = values[ind]
        if math.isnan(v):
            raise Tier0Error(
                f"canary indicator {ind!r} is NaN — a numerical failure "
                "must refuse the cell, not evade the red-flag comparisons "
                "(r1 finding 10)")
        if ind == "nonascii_frac":
            v = v - thresholds.nonascii_base  # base-relative (§4.4)
        if CANARY_HARMFUL_SIGN[ind] * (v - thresholds.fail_lines[ind]) > 0:
            out.append(ind)
    return tuple(out)


def canary_outcome(tripped_attempt0: Sequence[str],
                   tripped_attempt1: Optional[Sequence[str]] = None) -> str:
    """The §4.4 state machine. Canary FAILS on >=2 indicators; a failure
    triggers a confirmation rerun in a FRESH SUBPROCESS (fresh model
    load / seed / item order, same allocation — the driver owns those
    mechanics, §11.1/F16); only a reproduced failure (>=2 indicators
    again on attempt 1) is a confirmed red flag. Both attempts' rows
    persist."""
    failed0 = len(tripped_attempt0) >= 2
    if not failed0:
        return CANARY_PASS
    if tripped_attempt1 is None:
        return CANARY_NEEDS_CONFIRMATION
    return (CANARY_CONFIRMED_FAIL if len(tripped_attempt1) >= 2
            else CANARY_NOT_REPRODUCED)


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------


@dataclass
class Tier0CorpusCfg:
    """One neutral corpus: a corpora.CorpusSlice (kind text|chat) plus
    its registered task names."""

    slice: Any  # sisteer.capeval.corpora.CorpusSlice
    nll_task: str
    kl_task: str
    token_budget: Optional[int] = None            # text kinds
    assistant_token_budget: Optional[int] = None  # chat kinds


@dataclass
class CanaryCfg:
    slice: Any  # prompts CorpusSlice
    task: str = "tier0/canary"
    gen: Dict[str, Any] = field(default_factory=lambda: {
        "max_new_tokens": 256, "do_sample": False})
    thresholds: Optional[CanaryThresholds] = None  # calibration-owned
    abrupt_min_tokens: int = 8
    rep_ns: Tuple[int, ...] = (1, 2, 3)


@dataclass
class Tier0Spec:
    """Everything one tier0.run(...) needs beyond the engine/arm/writer.

    ``template`` supplies every cell-constant ResultRow column (sweep,
    run_id, model_id, model_revision, arm_name, kind, coeff,
    intervention_hash, engine, trial_id, seed, attempt,
    harness_config_hash, env_digest, slurm_job_id) — same convention as
    rows_to_result_rows. ``domain_of`` maps every emitted task to
    its ONE registry domain (registered-metric rule): the driver builds
    it from domains.Registry."""

    template: Dict[str, Any]
    domain_of: Mapping[str, str]
    corpora: List[Tier0CorpusCfg] = field(default_factory=list)
    canary: Optional[CanaryCfg] = None
    relevant_corpus: Optional[Any] = None  # CorpusSlice | None (§4.5)
    localization_task: str = "tier0/kl_localization"
    top_k: int = 64
    window_len: int = 1024
    min_window_tokens: int = 32
    chat_max_tokens: int = 4096
    code_version: int = TIER0_CODE_VERSION
    build_cache: bool = False  # calibration trial 0 ONLY (§4.3)


_TEMPLATE_KEYS = (
    "sweep", "run_id", "model_id", "model_revision", "arm_name", "kind",
    "coeff", "intervention_hash", "engine", "trial_id", "seed", "attempt",
    "harness_config_hash", "env_digest", "slurm_job_id",
)


def load_tier0_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load + validate configs/capeval/tier0.yaml (structure only —
    corpus revisions live in pins.yaml, resolved by the Phase-0 fetch
    script; this file owns window/top_k/canary/task-name settings)."""
    import yaml

    raw = yaml.safe_load(Path(path).read_text())
    # "kappa_floor" was removed from the vocabulary when the κ flag
    # path was dropped — a yaml still carrying it refuses loudly below.
    allowed = {"version", "code_version", "top_k", "window_len",
               "min_window_tokens", "chat_max_tokens",
               "corpora", "canary", "localization"}
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise Tier0Error(f"{path}: unknown top-level key(s) {unknown}")
    if raw.get("code_version") != TIER0_CODE_VERSION:
        raise Tier0Error(
            f"{path}: code_version {raw.get('code_version')} != module "
            f"TIER0_CODE_VERSION {TIER0_CODE_VERSION} — bump them together "
            "(the yaml records the convention the caches were built under)")
    corpora = raw.get("corpora")
    if not isinstance(corpora, list) or not corpora:
        raise Tier0Error(f"{path}: 'corpora' must be a non-empty list")
    for c in corpora:
        for k in ("name", "kind", "nll_task", "kl_task"):
            if k not in c:
                raise Tier0Error(f"{path}: corpus entry missing {k!r}: {c}")
        if c["kind"] not in ("text", "chat"):
            raise Tier0Error(f"{path}: corpus kind {c['kind']!r} not in "
                             "('text','chat')")
    can = raw.get("canary")
    if not isinstance(can, Mapping) or "task" not in can:
        raise Tier0Error(f"{path}: 'canary' must be a mapping with a task")
    return raw


# ---------------------------------------------------------------------------
# tier0.run
# ---------------------------------------------------------------------------


def _template_check(template: Mapping[str, Any]) -> Dict[str, Any]:
    missing = sorted(set(_TEMPLATE_KEYS) - set(template))
    extra = sorted(set(template) - set(_TEMPLATE_KEYS))
    if missing or extra:
        raise Tier0Error(f"tier0 template keys must be exactly "
                         f"{_TEMPLATE_KEYS}: missing={missing} extra={extra}")
    return dict(template)


class _Emitter:
    """Row construction with the cell-constant template.

    Idempotent per (task, item_id, metric): rows already in the store
    (a resumed cell) and rows already emitted this run are never
    duplicated — calibration's readers require exactly one aggregate
    row per (metric, trial) (stats.calibrate), so resume must not
    double-file the "_corpus"/"__aggregate__" rows. Attempts and trials
    land in their own cells (attempt/trial are cell_key fields), so
    within-cell dedup can never swallow a legitimate re-measurement."""

    def __init__(self, writer, template: Dict[str, Any],
                 domain_of: Mapping[str, str], task_version: str):
        self.writer = writer
        self.template = template
        self.domain_of = domain_of
        self.task_version = task_version
        self._seen: Set[Tuple[str, str, str]] = set()
        parsed = getattr(writer, "_parsed_rows", None)
        if parsed is not None:
            for r in parsed():
                self._seen.add((r.get("task"), r.get("item_id"),
                                r.get("metric")))

    def domain(self, task: str) -> str:
        try:
            return self.domain_of[task]
        except KeyError:
            raise Tier0Error(
                f"task {task!r} resolves to no registered domain — the "
                "registered-metric rule refuses unmapped emissions "
                "(DESIGN §3)") from None

    def emit(self, *, task: str, item_id: str, metric: str, value: float,
             tier: int = 0, domain: Optional[str] = None,
             extra: Optional[Mapping[str, Any]] = None,
             position_order: Optional[int] = None,
             wall_clock_s: Optional[float] = None) -> None:
        key = (task, item_id, metric)
        if key in self._seen:
            return
        if not math.isfinite(float(value)):
            # §PW-nanfix-1 backstop: the store's row writer allows NaN, so a
            # non-finite value would be written as the JSON constant `NaN`
            # and read back as a number by every consumer. A measurement that
            # is not a number is not a row.
            raise Tier0Error(
                f"emit({task!r}, {item_id!r}, {metric!r}): value is {value!r}, not a "
                "finite number — a row is a measurement; a producer with nothing to "
                "measure files no row and says so in its summary counts")
        self._seen.add(key)
        row = ResultRow(
            **self.template,
            tier=tier,
            task=task,
            task_version=self.task_version,
            domain=domain if domain is not None else self.domain(task),
            item_id=item_id,
            position_order=position_order,
            metric=metric,
            value=float(value),
            extra_json=canonical_json(dict(extra)) if extra else None,
            created_at=time.time(),
            wall_clock_s=wall_clock_s,
        )
        self.writer.append_row(row)


def _completed_items(writer, task: str, expected_metrics: Set[str]) -> Dict[str, Dict[str, float]]:
    """item_id -> {metric: value} for items of ``task`` that already have
    ALL expected metrics filed (per-item resume, §2.3). Uses the store's
    parsed rows when available (CellStore); a plain stub writer without
    them resumes nothing.

    ONE value per (item, metric): the store is append-only and
    :class:`_Emitter` never files a key twice, so a (task, item, metric)
    that appears twice with DIFFERENT values can only mean two writers
    raced on the store (a duplicate driver) — and a resume that took
    whichever row came later in the file would silently pick one of two
    measurements. That is refused here (2026-09-02); a same-value repeat
    (a crash between an item's companion rows, re-filed identically) is
    harmless and resumes."""
    parsed = getattr(writer, "_parsed_rows", None)
    if parsed is None:
        return {}
    got: Dict[str, Dict[str, float]] = {}
    for r in parsed():
        if r.get("task") == task and r.get("metric") in expected_metrics:
            item, metric, value = r["item_id"], r["metric"], r["value"]
            have = got.setdefault(item, {})
            if metric in have and have[metric] != value:
                raise Tier0Error(
                    f"{task}: item {item!r} metric {metric!r} is filed twice "
                    f"with DIFFERENT values ({have[metric]!r}, then {value!r}) "
                    "— two writers raced on this store; refusing to resume "
                    "from rows that cannot say which measurement is real. "
                    "Inspect rows.jsonl and repair the store deliberately.")
            have[metric] = value
    return {i: m for i, m in got.items() if set(m) == expected_metrics}


def _windowed(tok, c: Tier0CorpusCfg, spec: Tier0Spec) -> WindowedCorpus:
    sl = c.slice
    if sl.kind == "text":
        return windows_from_texts(
            tok, sl.name, list(sl.items), window_len=spec.window_len,
            token_budget=c.token_budget,
            min_window_tokens=spec.min_window_tokens)
    if sl.kind == "chat":
        return windows_from_conversations(
            tok, sl.name, list(sl.items), list(sl.item_ids),
            assistant_token_budget=c.assistant_token_budget,
            max_tokens=spec.chat_max_tokens)
    raise Tier0Error(f"corpus {sl.name!r}: unsupported kind {sl.kind!r} for "
                     "the NLL/KL probes (text|chat)")


def _corpus_revision(sl) -> str:
    pin = (sl.spec or {}).get("pin") or {}
    if "revision" in pin:
        return str(pin["revision"])
    src = (sl.spec or {}).get("source") or {}
    if "revision" in src and src["revision"]:
        return str(src["revision"])
    return "local"


def _score_pass(engine, windowed: WindowedCorpus, arm, atom_request):
    """Consume engine.score_corpus grouped back per item:
    yields (item, [nll...], AtomScores-list-or-None)."""
    it = iter(engine.score_corpus(windowed, arm, atom_request))
    for item in windowed.items:
        n = len(item.score_positions)
        nlls, atoms = [], []
        for _ in range(n):
            ps = next(it)
            nlls.append(float(ps.token_nll))
            atoms.append(ps.atom_probs)
        yield item, nlls, (atoms if atoms and atoms[0] is not None else None)
    # the engine must be exactly exhausted
    leftover = next(it, None)
    if leftover is not None:
        raise Tier0Error("engine.score_corpus yielded more positions than "
                         "the corpus declares — position alignment is broken")


def _stack_atoms(atom_lists: List[List[AtomScores]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ids = torch.stack([a.ids for atoms in atom_lists for a in atoms]).to(torch.int32)
    logp = torch.stack([a.logp for atoms in atom_lists for a in atoms]).to(torch.float64)
    tail = torch.tensor([a.logp_tail for atoms in atom_lists for a in atoms],
                        dtype=torch.float64)
    return ids, logp, tail


def _cache_spec_for(sl, cfg: Tier0Spec, tok_hash: str, dtype: str) -> Dict[str, Any]:
    """The resolved §4.3 cache spec for one corpus slice under this run."""
    return cache_spec(
        model_revision=cfg.template["model_revision"],
        engine=cfg.template["engine"],
        corpus_id=sl.name,
        corpus_revision=_corpus_revision(sl),
        slice_hash=sl.slice_hash,
        tokenizer_hash=tok_hash,
        dtype=dtype,
        top_k=cfg.top_k,
        code_version=cfg.code_version,
    )


def _build_base_atoms(engine, windowed: WindowedCorpus, cache_dir: Path,
                      spec: Mapping[str, Any], top_k: int):
    """Base pass (arm=None): NLL + top-k atoms in one sweep, published to
    the cache. Returns (per_item, ids, logp, tail)."""
    per_item = list(_score_pass(engine, windowed, None, int(top_k)))
    ids, logp, tail = _stack_atoms([atoms for _, _, atoms in per_item])
    publish_atom_cache(cache_dir, spec, ids, logp, tail)
    return per_item, ids, logp, tail


def _atoms_kl(base_logp: torch.Tensor, base_tail: torch.Tensor,
              atoms: List[AtomScores]) -> torch.Tensor:
    """Per-position coarsened KL of one item's ablation atoms vs the base
    slices (float64 end-to-end, r2-F4)."""
    return kl_coarse_window(
        base_logp, base_tail,
        torch.stack([a.logp for a in atoms]).to(torch.float64),
        torch.tensor([a.logp_tail for a in atoms], dtype=torch.float64))


#: The probes whose attempt-0 red flags request a confirmation rerun —
#: the vocabulary of ``confirm_probes`` (the driver persists them in
#: confirmation_required.json). "dist_shift" can no longer be
#: REQUESTED (the κ flag path is gone), but it stays in the vocabulary
#: so older persisted markers replay attempt 1 without a refusal; a
#: rerun it requested simply confirms nothing KL-side.
CONFIRMABLE_PROBES = frozenset({"canary", "dist_shift"})


def run(engine, arm, cfg: Tier0Spec, writer, caches: Union[str, Path],
        confirm_probes: Optional[frozenset] = None) -> Dict[str, Any]:
    """The §4 Tier-0 entry point: NLL + coarsened-KL per corpus, canary,
    localization; per-probe wall-clock recorded on the summary rows.

    Returns the phase summary consumed by the driver, including
    ``confirmation_needed`` entries for any Tier-0 red flag raised at
    attempt 0 (canary >=2 indicators — now the ONLY red flag;
    kl_coarse_lb magnitudes are annotations and trip nothing) — the
    driver reruns the cell in a fresh subprocess as attempt 1
    (§11.1/F16), passing the attempt-0 tripped probes back in as
    ``confirm_probes`` (from confirmation_required.json — the frozen
    contract). On attempt 1, ONLY the probes in ``confirm_probes``
    evaluate confirmation outcomes; a fresh trip of an unrequested
    probe files a NEW flag row (unconfirmed), never a ``*_confirmed``
    one.

    Resume semantics: teacher-forced windows are RE-SCORED on a partial
    resume (deterministic, order-invariant at batch=1; only duplicate
    row emission is skipped), so corpus aggregates are always computed
    over the full corpus. Canary GENERATION is skipped per completed
    prompt (the expensive part) and per-prompt stats are recovered from
    the filed rows' extra_json."""
    template = _template_check(cfg.template)
    tok = engine.tokenizer
    em = _Emitter(writer, template, cfg.domain_of,
                  task_version=f"tier0_v{cfg.code_version}")
    cache_dir = Path(caches)
    tok_hash = tokenizer_hash(tok)
    dtype = str(next(engine.model.parameters()).dtype)
    attempt = int(template["attempt"])
    seed = int(template["seed"])

    confirm = frozenset(confirm_probes) if confirm_probes is not None else None
    if confirm is not None:
        unknown_probes = sorted(confirm - CONFIRMABLE_PROBES)
        if unknown_probes:
            raise Tier0Error(f"confirm_probes {unknown_probes} are not "
                             f"confirmable probes {sorted(CONFIRMABLE_PROBES)}")
    if attempt == 0 and confirm:
        raise Tier0Error(
            "confirm_probes passed at attempt 0 — the confirmation set only "
            "exists once attempt 0 has tripped (driver contract, r1 "
            "finding 5)")
    if attempt > 0 and confirm is None:
        raise Tier0Error(
            f"attempt {attempt} without confirm_probes — the driver must "
            "pass the attempt-0 tripped-probe set (from "
            "confirmation_required.json) so only requested probes can file "
            "*_confirmed outcomes (r1 finding 5)")

    # Teacher-forced probes never apply arm.transform_prompt — an arm
    # whose intervention lives in the prompt (a token kind) would be
    # scored against the UNTRANSFORMED corpus and report base numbers.
    # Arm-transformed scored corpora (with recomputed masks and
    # transformed-corpus cache identity) are a v1 item; until then,
    # refuse loudly rather than file silently-base NLL/KL rows.
    if arm is not None and (cfg.corpora or cfg.relevant_corpus is not None):
        sentinel = "capeval transform-probe sentinel"
        if arm.transform_prompt(sentinel) != sentinel:
            raise Tier0Error(
                f"arm {getattr(arm, 'name', '?')!r} transforms prompts (a "
                "token-kind intervention): the teacher-forced Tier-0 NLL/KL/"
                "localization probes would score the untransformed corpus "
                "and report base numbers — arm-transformed scored corpora "
                "are a v1 item (r1 finding 3); refusing")

    summary: Dict[str, Any] = {"nll_per_token": {}, "kl_coarse_lb": {},
                               "confirmation_needed": [], "flags": []}
    neutral_kl_sum, neutral_kl_pos = 0.0, 0

    # ---- neutral corpora: one arm pass per corpus scores NLL + KL ------
    for c in cfg.corpora:
        t_start = time.time()
        windowed = _windowed(tok, c, cfg)
        if not windowed.items:
            raise Tier0Error(f"corpus {c.slice.name!r} windowed to zero items")
        spec = _cache_spec_for(c.slice, cfg, tok_hash, dtype)

        built_this_pass = False
        if atom_cache_exists(cache_dir, spec):
            base_ids, base_logp, base_tail = load_atom_cache(
                cache_dir, spec, expected_positions=windowed.n_positions)
        elif cfg.build_cache:
            if arm is not None:
                raise Tier0Error(
                    f"corpus {c.slice.name!r}: build_cache is set but an arm "
                    "is active — base atom caches are built by the BASE pass "
                    "of calibration trial 0 only (§4.3)")
            per_item, base_ids, base_logp, base_tail = _build_base_atoms(
                engine, windowed, cache_dir, spec, cfg.top_k)
            built_this_pass = True
        else:
            raise Tier0Error(
                f"corpus {c.slice.name!r}: no base atom cache under "
                f"{cache_dir} and build_cache is not set — calibration "
                "trial 0 builds the neutral-corpus caches; run it first "
                "(§4.3/§11.2)")

        nll_metrics = {METRIC_NLL_SUM, METRIC_TOKEN_COUNT}
        kl_metrics = {METRIC_KL, "kl_sum", "kl_pos_count"}
        done_nll = _completed_items(writer, c.nll_task, nll_metrics)
        done_kl = _completed_items(writer, c.kl_task, kl_metrics)

        if not built_this_pass:
            # arm pass (or non-reference base pass): exact logq at the
            # cached ids + tail by the same masked logsumexp
            per_item = list(_score_pass(engine, windowed, arm, base_ids))

        # per-window rows + corpus aggregates
        offset = 0
        corpus_nll, corpus_tok = 0.0, 0
        corpus_kl, corpus_pos, inf_positions = 0.0, 0, 0
        for order, (item, nlls, atoms) in enumerate(per_item):
            n = len(item.score_positions)
            if built_this_pass:
                # trial 0 vs its own atoms: KL is identically zero (same
                # process, same logits) — filed as the reference leg.
                kl_vals = torch.zeros(n, dtype=torch.float64)
            else:
                q_ids = torch.stack([a.ids for a in atoms])
                if not torch.equal(q_ids.to(torch.int64),
                                   base_ids[offset:offset + n].to(torch.int64)):
                    raise Tier0Error(
                        f"{c.slice.name}/{item.item_id}: engine returned "
                        "atoms at ids other than the cached base ids — "
                        "position alignment is broken")
                kl_vals = _atoms_kl(base_logp[offset:offset + n],
                                    base_tail[offset:offset + n], atoms)
            offset += n

            nll_sum = float(sum(nlls))
            kl_sum = float(kl_vals.sum())
            n_inf = int(torch.isinf(kl_vals).sum())
            corpus_nll += nll_sum
            corpus_tok += n
            corpus_kl += kl_sum
            corpus_pos += n
            inf_positions += n_inf
            if item.item_id not in done_nll:
                em.emit(task=c.nll_task, item_id=item.item_id,
                        metric=METRIC_NLL_SUM, value=nll_sum,
                        position_order=order)
                em.emit(task=c.nll_task, item_id=item.item_id,
                        metric=METRIC_TOKEN_COUNT, value=float(n),
                        position_order=order)
            if item.item_id not in done_kl:
                # METRIC_KL per window = the window's mean nats/position
                # (the item value κ derivation bootstraps); kl_sum +
                # kl_pos_count keep the recompute-from-rows discipline (F14)
                em.emit(task=c.kl_task, item_id=item.item_id,
                        metric=METRIC_KL, value=kl_sum / max(n, 1),
                        position_order=order)
                em.emit(task=c.kl_task, item_id=item.item_id,
                        metric="kl_sum", value=kl_sum, position_order=order)
                em.emit(task=c.kl_task, item_id=item.item_id,
                        metric="kl_pos_count", value=float(n),
                        position_order=order)

        wall = time.time() - t_start
        mean_nll = corpus_nll / max(corpus_tok, 1)
        mean_kl = corpus_kl / max(corpus_pos, 1)  # +inf propagates, never excluded
        em.emit(task=c.nll_task, item_id="_corpus", metric="nll_per_token",
                value=mean_nll, wall_clock_s=wall)
        # deliberately NOT metric kl_coarse_lb: that name is the per-window
        # item contract κ derivation consumes (METRIC_KL)
        em.emit(task=c.kl_task, item_id="_corpus", metric="kl_coarse_lb_corpus",
                value=mean_kl)
        em.emit(task=c.kl_task, item_id="_corpus", metric="kl_inf_positions",
                value=float(inf_positions))
        if inf_positions > 0:
            # automatic flag row — surfaced in the gate log, never capped
            # (F4). Note this indicts numerics/mass destruction visibly;
            # it is display/gate-log evidence only and (like every
            # flag_kl_* metric) never reaches the screen or a verdict.
            em.emit(task=c.kl_task, item_id="_corpus", metric="flag_kl_inf",
                    value=1.0, extra={"corpus": c.slice.name,
                                      "inf_positions": inf_positions})
        summary["nll_per_token"][c.nll_task] = mean_nll
        summary["kl_coarse_lb"][c.kl_task] = mean_kl
        neutral_kl_sum += corpus_kl
        neutral_kl_pos += corpus_pos

    # There is deliberately NO dist-shift red flag here anymore. The old
    # path flagged kl_coarse_lb > κ on >=2 corpora and requested a
    # confirmation rerun; with κ degenerate at its floor on deterministic
    # stacks the flag classified every active arm (the harmless contrast
    # included), so KL was demoted to an annotation. The per-window and
    # per-corpus kl_coarse_lb magnitudes above are still filed in full.

    # ---- canary (§4.4) --------------------------------------------------
    if cfg.canary is not None:
        summary["canary"] = _run_canary(engine, arm, cfg, em, writer,
                                        attempt, seed, summary, confirm)

    # ---- KL localization (§4.5, display-only) --------------------------
    if cfg.relevant_corpus is None:
        # recorded notice, never a silent omission (§4.5)
        em.emit(task=cfg.localization_task, item_id="_run", metric="notice",
                value=0.0,
                extra={"reason": "not computed: no target-relevant corpus "
                                 "declared (mode is control/calibration)"})
        summary["kl_concentration"] = None
    else:
        summary["kl_concentration"] = _run_localization(
            engine, arm, cfg, em, cache_dir, tok, tok_hash, dtype,
            neutral_kl_sum, neutral_kl_pos)

    return summary


def _run_canary(engine, arm, cfg: Tier0Spec, em: _Emitter, writer,
                attempt: int, seed: int, summary: Dict[str, Any],
                confirm: Optional[frozenset]) -> Dict[str, Any]:
    can = cfg.canary
    t_start = time.time()
    tok = engine.tokenizer
    prompts = [str(p) for p in can.slice.items]
    item_ids = list(can.slice.item_ids)
    rendered = [build_chat_prompt(tok, p) for p in prompts]
    if arm is not None:
        rendered = [arm.transform_prompt(r) for r in rendered]

    # per-prompt resume: one atomic row per prompt (metric canary_gen)
    done = _completed_items(writer, can.task, {"canary_gen"})
    # the item-order permutation the calibration null covers (§7.1):
    # batched generation is the one Tier-0 path where order genuinely
    # enters (batch composition), so the permutation is applied here.
    order = list(range(len(prompts)))
    random.Random(seed).shuffle(order)
    todo = [i for i in order if item_ids[i] not in done]

    # the configured batch_size rides in gen_cfg: the ENGINE chunks the
    # prompt list and enters a fresh arm ctx per chunk — never one giant
    # batch.
    gen_cfg = dict(can.gen)
    gen_cfg.setdefault("batch_size", 16)
    rank = {i: r for r, i in enumerate(order)}
    per_prompt_extra: Dict[str, Dict[str, Any]] = {}

    outs = engine.generate([rendered[i] for i in todo], arm, gen_cfg) \
        if todo else []
    if len(outs) != len(todo):
        raise Tier0Error(f"canary: {len(outs)} generations for "
                         f"{len(todo)} prompts")
    for i, out in zip(todo, outs):
        # per-prompt stats from the RAW generated ids, via the ONE
        # shared implementation. The text rides in extra_json so
        # distinct3 survives a per-item resume (recovered below); live
        # store data only, never repo data.
        stats = canary_gen_stats(out, abrupt_min_tokens=can.abrupt_min_tokens,
                                 rep_ns=can.rep_ns)
        em.emit(task=can.task, item_id=item_ids[i], metric="canary_gen",
                value=float(stats["n_tokens"]),
                position_order=rank[i],
                extra=stats)
        per_prompt_extra[item_ids[i]] = stats

    # aggregates need every prompt's stats: recover resumed ones from the
    # store's parsed rows (extra_json carries them)
    parsed = getattr(writer, "_parsed_rows", None)
    if parsed is not None:
        for r in parsed():
            if (r.get("task") == can.task and r.get("metric") == "canary_gen"
                    and r["item_id"] not in per_prompt_extra
                    and r.get("extra_json")):
                per_prompt_extra[r["item_id"]] = json.loads(r["extra_json"])
    missing = [iid for iid in item_ids if iid not in per_prompt_extra]
    if missing:
        raise Tier0Error(
            f"canary: {len(missing)} prompts have no recoverable per-prompt "
            "stats (resumed rows without extra_json?) — cannot aggregate "
            "indicators honestly; refusing")

    stats_in_order = [per_prompt_extra[iid] for iid in item_ids]
    values = canary_values_from_stats(stats_in_order)
    # per-indicator aggregate rows at CANARY_AGG_ITEM, metric = the bare
    # indicator name — the exact rows calibration derives thresholds from
    # (stats.calibrate._canary_thresholds_from_rows iterates every metric
    # under this (task, item), so nothing else may be filed there)
    for ind in CANARY_INDICATORS:
        em.emit(task=can.task, item_id=CANARY_AGG_ITEM, metric=ind,
                value=values[ind])

    result: Dict[str, Any] = {"values": values, "outcome": CANARY_PASS}
    if can.thresholds is not None:
        tripped = tripped_indicators(values, can.thresholds)
        for ind in tripped:
            em.emit(task=can.task, item_id="_run",
                    metric=f"canary/tripped_{ind}", value=1.0)
        em.emit(task=can.task, item_id="_run", metric="canary/tripped_count",
                value=float(len(tripped)))
        if attempt == 0:
            outcome = canary_outcome(tripped)
        elif "canary" in confirm:
            # the requested confirmation: reproduced (>=2 again) or not
            outcome = (CANARY_CONFIRMED_FAIL if len(tripped) >= 2
                       else CANARY_NOT_REPRODUCED)
        else:
            # canary confirmation was NOT requested (another probe's
            # rerun): a fresh >=2 trip is NEW information — filed
            # unconfirmed, never *_confirmed
            outcome = (CANARY_NEW_UNCONFIRMED if len(tripped) >= 2
                       else CANARY_PASS)
        result.update({"tripped": list(tripped), "outcome": outcome})
        if outcome == CANARY_NEEDS_CONFIRMATION:
            summary["confirmation_needed"].append(
                {"probe": "canary", "tripped": list(tripped)})
        if outcome == CANARY_CONFIRMED_FAIL:
            em.emit(task=can.task, item_id="_run",
                    metric="flag_canary_confirmed", value=1.0,
                    extra={"tripped": list(tripped), "attempt": attempt})
            summary["flags"].append({"probe": "canary",
                                     "tripped": list(tripped)})
        if outcome == CANARY_NEW_UNCONFIRMED:
            em.emit(task=can.task, item_id="_run",
                    metric="flag_canary_unconfirmed", value=1.0,
                    extra={"tripped": list(tripped), "attempt": attempt})
            summary["flags"].append({"probe": "canary",
                                     "tripped": list(tripped),
                                     "confirmed": False})
    wall = time.time() - t_start
    em.emit(task=can.task, item_id="_run", metric="canary_wall_clock",
            value=wall, wall_clock_s=wall)
    return result


def _run_localization(engine, arm, cfg: Tier0Spec, em: _Emitter,
                      cache_dir: Path, tok, tok_hash: str, dtype: str,
                      neutral_kl_sum: float, neutral_kl_pos: int) -> Optional[float]:
    sl = cfg.relevant_corpus
    t_start = time.time()
    windowed = windows_from_prompts(tok, sl.name, [str(p) for p in sl.items],
                                    list(sl.item_ids),
                                    max_tokens=cfg.chat_max_tokens)
    if not windowed.items:
        raise Tier0Error("localization: target-relevant slice windowed to "
                         "zero items")
    spec = _cache_spec_for(sl, cfg, tok_hash, dtype)
    # r2-new-1: base atoms built IN THIS CELL with hooks disabled
    # (arm=None => no steering ctx => exactly the base model); cached
    # per-arm under the slice's own hash. Trial 0 is not involved.
    if atom_cache_exists(cache_dir, spec):
        base_ids, base_logp, base_tail = load_atom_cache(
            cache_dir, spec, expected_positions=windowed.n_positions)
    else:
        _, base_ids, base_logp, base_tail = _build_base_atoms(
            engine, windowed, cache_dir, spec, cfg.top_k)

    per_item = list(_score_pass(engine, windowed, arm, base_ids))
    offset, total_kl, total_pos = 0, 0.0, 0
    for order, (item, _nlls, atoms) in enumerate(per_item):
        n = len(item.score_positions)
        kl_vals = _atoms_kl(base_logp[offset:offset + n],
                            base_tail[offset:offset + n], atoms)
        offset += n
        em.emit(task=cfg.localization_task, item_id=item.item_id,
                metric="kl_sum", value=float(kl_vals.sum()),
                position_order=order)
        em.emit(task=cfg.localization_task, item_id=item.item_id,
                metric="kl_pos_count", value=float(n), position_order=order)
        total_kl += float(kl_vals.sum())
        total_pos += n

    target_mean = total_kl / max(total_pos, 1)
    em.emit(task=cfg.localization_task, item_id="_run",
            metric="kl_target_mean", value=target_mean,
            wall_clock_s=time.time() - t_start)
    neutral_mean = neutral_kl_sum / max(neutral_kl_pos, 1)
    if neutral_kl_pos == 0 or neutral_mean == 0.0:
        em.emit(task=cfg.localization_task, item_id="_run", metric="notice",
                value=0.0,
                extra={"reason": "kl_concentration not computed: neutral "
                                 "kl_coarse_lb is zero"})
        return None
    concentration = target_mean / neutral_mean
    em.emit(task=cfg.localization_task, item_id="_run",
            metric="kl_concentration", value=concentration)
    return concentration
