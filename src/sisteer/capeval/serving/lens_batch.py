"""Segmenting the flattened continuous-batching forward back into
per-request rows, per-request :class:`~.lens_adapters.HookContext`
construction, and the post-run prefill ledger (petri-bloom Phase-2
serving; the vllm-lens hook wiring's position layer).

WHY THIS EXISTS. The vllm-lens hook fires at each decoder layer's block
output with ONE flattened ``[num_tokens, d]`` tensor covering EVERY
request scheduled into that forward (vLLM V1 continuous batching; no
padding). Position/phase-dependent ops (``lens_adapters.
op_needs_context``) must know which rows belong to which request and
where each request's prompt boundary sits. This module derives that
from vLLM V1's attention metadata (``query_start_loc`` / ``seq_lens`` /
``num_actual_tokens``), classifies every segment, builds the
per-request :class:`~.lens_adapters.HookContext`, and keeps the
run-level multiset accounting that catches the two failure modes no
per-forward check can see (below).

THE CLASSIFICATION LATTICE (ported from an earlier backend's
``derive_batch_plan`` — reimplemented here, never imported: that
module lives on a different branch). Per segment, with ``q`` the
query length and ``computed`` the tokens already in the KV cache
before this forward:

- ``q >= 2, computed == 0`` — a COMPLETE prefill: the whole prompt in
  one forward, rows at absolute positions ``0..q-1``.
- ``q == 1, computed >= 1`` — a decode step: one generated token at
  absolute position ``computed``.
- ``q >= 2, computed > 0`` — a chunked-prefill continuation or a
  prefix-cache hit. REFUSED by default (the pinned geometry makes
  chunking impossible and a cache-off engine never resumes); under
  ``allow_resumed=True`` — passed by the hook wiring exactly when the
  LIVE engine config says prefix caching is on — it is classified as a
  RESUMED segment instead: kept in ``segments`` (a mixed forward's row
  accounting must still cover its rows) and given a HookContext ONLY
  when the caller names it a resumed PREFILL of a request whose ops may
  resume (``build_segment_contexts(..., resumed_prefill=...)``); every
  other resumed segment gets ``ctx=None``. A selector-LESS request may
  always arrive resumed — cache hits are exactly the win the unhooked
  prompted/base legs are supposed to get, and no op runs on their rows.
  A TRACKED (selector-carrying, plan-needing) resumed segment is judged
  by :func:`check_tracked_segment` against the request's own prompt
  length AND its ops' ``lens_adapters.resumed_prefill_ok`` (the
  ``resumed_ok`` argument): a resumable cell's slice that ends EXACTLY at
  the prompt end is a RESUMED PREFILL (:data:`SEG_RESUMED_PREFILL`) whose
  context carries the tail's absolute positions and ``cached =
  computed``; any other tracked resume — a non-resumable cell (the salt
  discipline broke, or a foreign client bypassed the proxy), a slice
  that overruns the prompt (a preemption recompute under caching), or one
  that stops short of it (a chunked continuation) — refuses.
- ``q == 1, computed == N-1`` for a request of prompt length N — the
  FULLY prefix-cached prompt: vLLM leaves the last prompt token to be
  computed, so the shape is decode-like but the row is a PROMPT row.
  Shape alone cannot tell it from a decode step or from the 1-token
  remnant of a chunked prefill; :func:`check_tracked_segment` can, and
  for a resumable cell files it as a one-row RESUMED PREFILL (the avg@k
  shape: an identical prompt served twice under one salt); for every
  other tracked request it stays the refusal it always was.
- ``q == 1, computed == 0`` — REFUSED: a 1-token prefill — either a
  sub-2-token prompt or the 1-token remnant of a chunked prefill;
  refusing to guess which.

TWO FAILURE MODES NO SHAPE-ONLY CHECK CATCHES. Both pass every local
check above; both are caught by :func:`check_tracked_segment`, which
compares each tracked segment against ITS OWN request's submitted
prompt-token length (read off the runner's request state, so it needs
no run lifecycle and no cross-request state):

(a) THE 1-TOKEN FINAL CHUNK of a chunked prefill. A length-N prompt
    split N-1 then 1: the final chunk arrives with ``q == 1,
    computed == N-1`` — bit-for-bit the legal decode-step shape, and
    the positions cross-check passes because that row genuinely sits at
    position N-1. Shape alone cannot tell it from a decode row; the
    prompt length can: a GENUINE decode step of that request has
    ``computed >= N``, the final chunk has ``computed == N-1 < N``.
(b) A PREEMPTION-RECOMPUTE. A preempted request's KV blocks are
    evicted and the engine later re-runs prompt PLUS already-generated
    tokens as one fresh prefill: ``q == N+g, computed == 0`` — a
    structurally perfect complete prefill whose positions start at 0.
    Steering it as a prefill puts prompt-boundary-dependent edits on
    rows that are generated tokens, silently. Against the request's own
    prompt length it is loud: ``q == N+g != N``. (The same comparison
    catches a FIRST chunk of a chunked prefill, ``q < N, computed ==
    0``, which the lattice alone also cannot tell from a short
    complete prefill.)

:class:`PrefillLedger` — the run-scoped multiset accounting over
submitted prompt lengths — predates the per-request check and survives
as an OPTIONAL, additional cross-check for the in-process exact-replay
harness (one ``llm.generate`` call == one run, so a run boundary
exists there). It is consumed only while ARMED (``begin_run``): the
HTTP serving path has no run boundary — requests overlap arbitrarily —
so it never arms the ledger and relies on :func:`check_tracked_segment`
alone, which subsumes the ledger's two failure modes per request. An
unarmed ledger is therefore NOT an error on the serving path; arming
is the in-proc harness's explicit act.

DELIBERATE DIVERGENCES FROM THE ORIGINAL BACKEND:

- The positions cross-check there was CONDITIONAL — silently skipped
  when the positions arg was absent or not a 1-D tensor, so a
  call-convention change degraded it to a no-op. Here ``positions`` is
  a REQUIRED argument and a missing/malformed one is a LOUD refusal
  (repo rule: fail loudly, never degrade silently).
- The cross-check verifies EVERY row of every segment sits at its
  expected absolute position (``computed_i .. computed_i + q_i - 1``),
  not just each segment's first row — same cost class, strictly more
  metadata drift caught.
- Whether the plan machinery applies is driven off the OPS
  (:func:`ops_need_plan`, reusing ``lens_adapters.op_needs_context``),
  not a hard-coded kind list.
"""

from __future__ import annotations

import operator
from collections import Counter
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import torch

from sisteer.capeval.serving import lens_adapters as la


class LensBatchError(RuntimeError):
    """A flattened-forward shape, positions cross-check, or prefill-
    accounting invariant the position logic cannot honor — loud, never
    a guessed row and never a silently skipped check."""


# --- metadata coercion (accepts lists or 1-D tensors, loudly) ------------------


def _to_int(name: str, x: Any) -> int:
    """``x`` as a plain int; bools, floats, and non-scalars refuse."""
    if torch.is_tensor(x):
        if (x.numel() != 1 or x.is_floating_point() or x.is_complex()
                or x.dtype == torch.bool):
            raise LensBatchError(
                f"{name} must be a single integer, got a {x.dtype} tensor "
                f"of shape {tuple(x.shape)}")
        return int(x.item())
    if isinstance(x, bool):
        raise LensBatchError(f"{name} must be an int, got the bool {x!r}")
    try:
        return operator.index(x)
    except TypeError:
        raise LensBatchError(
            f"{name} must be an int, got {type(x).__name__} {x!r}"
        ) from None


def _int_rows(name: str, val: Any) -> List[int]:
    """``val`` as a list of ints. Accepts a 1-D integer tensor or a
    list/tuple of ints; ``None`` and every other shape refuse loudly —
    this is what makes the cross-check un-skippable (module docstring)."""
    if val is None:
        raise LensBatchError(
            f"{name} is required and got None — the original backend "
            "silently skipped its cross-check when this went missing; "
            "this module refuses instead (fail loudly, never degrade)")
    if torch.is_tensor(val):
        if val.dim() != 1:
            raise LensBatchError(
                f"{name} must be a 1-D tensor, got shape "
                f"{tuple(val.shape)}")
        if (val.is_floating_point() or val.is_complex()
                or val.dtype == torch.bool):
            raise LensBatchError(
                f"{name} must be an integer tensor, got dtype {val.dtype} "
                "— a non-integer positions/ids convention is metadata "
                "drift, not something to round through")
        return [int(x) for x in val.tolist()]
    if isinstance(val, (list, tuple)):
        return [_to_int(f"{name}[{i}]", x) for i, x in enumerate(val)]
    raise LensBatchError(
        f"{name} must be a 1-D tensor or a list/tuple of ints, got "
        f"{type(val).__name__}")


# --- the batch plan --------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    """One request's rows in a flattened ``[num_tokens, d]`` forward."""

    start: int      # first flattened row (inclusive)
    end: int        # one past the last flattened row
    q: int          # query length == end - start
    computed: int   # tokens already in the KV cache before this forward
    seq_len: int    # computed + q

    def __post_init__(self):
        # Internal consistency only — the classification lattice lives in
        # derive_batch_plan; this keeps a hand-built Segment honest.
        if self.start < 0 or self.end <= self.start:
            raise LensBatchError(
                f"Segment rows [{self.start}, {self.end}) are not a "
                "non-empty forward range")
        if self.q != self.end - self.start:
            raise LensBatchError(
                f"Segment.q == {self.q} != end - start == "
                f"{self.end - self.start}")
        if self.computed < 0:
            raise LensBatchError(f"Segment.computed == {self.computed} < 0")
        if self.seq_len != self.computed + self.q:
            raise LensBatchError(
                f"Segment.seq_len == {self.seq_len} != computed + q == "
                f"{self.computed + self.q}")

    @property
    def rows(self) -> slice:
        """The flattened rows this segment governs
        (``hidden[seg.rows]``)."""
        return slice(self.start, self.end)

    @property
    def is_prefill(self) -> bool:
        """True iff this is the lattice's complete-prefill cell.
        Meaningful only on plan-derived segments — derive_batch_plan
        refuses every shape that is neither a complete prefill, a decode
        step, nor (under ``allow_resumed``) a resumed segment."""
        return self.computed == 0

    @property
    def is_resumed(self) -> bool:
        """True iff this is a RESUMED shape (a prefix-cache hit's
        continuation: multi-token forward with tokens already in the KV
        cache) — legal under ``allow_resumed`` for selector-less requests
        and, as a RESUMED PREFILL, for resumable cells (module docstring;
        :func:`check_tracked_segment`). NB the fully-cached prompt's last
        row is decode-SHAPED (q == 1) and is not this."""
        return self.q >= 2 and self.computed > 0


@dataclass(frozen=True)
class BatchPlan:
    """Classified segments of one forward pass. Under
    :func:`derive_batch_plan`'s lattice every segment is either a
    COMPLETE prefill (``q >= 2, computed == 0``) or a decode step
    (``q == 1, computed >= 1``)."""

    num_tokens: int
    segments: Tuple[Segment, ...]
    full_prefills: Tuple[Segment, ...]
    n_decode: int
    #: resumed (cache-hit) segments tolerated under ``allow_resumed`` —
    #: never in ``full_prefills`` (the ledger's credits are COMPLETE
    #: prefills; the in-proc exact-replay harness runs caching-off and
    #: never observes these).
    n_resumed: int = 0


def derive_batch_plan(query_start_loc: Any, seq_lens: Any,
                      num_actual_tokens: Any, *,
                      allow_resumed: bool = False) -> BatchPlan:
    """Classify one vLLM V1 forward's flattened rows into segments.

    ``query_start_loc``: ``[num_reqs+1]`` cumulative query lengths;
    ``seq_lens``: ``[num_reqs]`` context+query per request (both accept
    lists or 1-D integer tensors). Raises :class:`LensBatchError` on ANY
    shape the position semantics cannot honor — the classification
    lattice and its rationale live in the module docstring — rather
    than steering wrong rows."""
    qsl = _int_rows("query_start_loc", query_start_loc)
    lens = _int_rows("seq_lens", seq_lens)
    nat = _to_int("num_actual_tokens", num_actual_tokens)
    if len(qsl) < 2:
        raise LensBatchError(
            f"query_start_loc has {len(qsl)} entries — no requests in the "
            "forward; refusing to guess")
    if len(lens) != len(qsl) - 1:
        raise LensBatchError(
            f"metadata mismatch: {len(lens)} seq_lens for {len(qsl) - 1} "
            "query segments")
    if qsl[0] != 0:
        raise LensBatchError(
            f"query_start_loc must start at 0, got {qsl[0]}")
    if qsl[-1] != nat:
        raise LensBatchError(
            f"query_start_loc[-1] == {qsl[-1]} != num_actual_tokens == "
            f"{nat} — padded or truncated metadata; refusing to index "
            "rows")

    segments: List[Segment] = []
    full_prefills: List[Segment] = []
    n_decode = 0
    n_resumed = 0
    for i, seq_len in enumerate(lens):
        start, end = qsl[i], qsl[i + 1]
        q = end - start
        if q < 1:
            raise LensBatchError(
                f"segment {i}: non-increasing query_start_loc "
                f"({start} -> {end})")
        computed = seq_len - q
        if computed < 0:
            raise LensBatchError(
                f"segment {i}: seq_len {seq_len} < query length {q}")
        seg = Segment(start=start, end=end, q=q, computed=computed,
                      seq_len=seq_len)
        if q >= 2 and computed == 0:
            full_prefills.append(seg)
        elif q == 1 and computed >= 1:
            n_decode += 1
        elif q >= 2 and allow_resumed:  # computed > 0: a cache hit
            n_resumed += 1
        elif q >= 2:  # computed > 0, resumed shapes not allowed
            raise LensBatchError(
                f"segment {i}: multi-token forward with {computed} tokens "
                "already computed — a chunked-prefill continuation or a "
                "prefix-cache hit. The prompt-boundary position logic "
                "cannot honor a partial prompt; the composed geometry "
                "makes chunking impossible, and every selector-carrying "
                "request is served under a REQUEST-UNIQUE cache salt "
                "(proxy._rewrite; Phase-0 assertion 7 proves the "
                "isolation), so a steered prefill can never resume from "
                "cache. This server is not the pinned configuration, or "
                "the salt discipline broke.")
        else:  # q == 1 and computed == 0
            raise LensBatchError(
                f"segment {i}: 1-token prefill — either a sub-2-token "
                "prompt (refused at submission) or the 1-token remnant of "
                "a chunked prefill. Refusing to guess which.")
        segments.append(seg)
    return BatchPlan(num_tokens=nat, segments=tuple(segments),
                     full_prefills=tuple(full_prefills), n_decode=n_decode,
                     n_resumed=n_resumed)


def cross_check_positions(plan: BatchPlan, positions: Any) -> None:
    """Verify the model-input positions tensor agrees with the attention
    metadata: segment ``i``'s rows must sit at absolute positions
    ``computed_i .. computed_i + q_i - 1`` (in particular its FIRST row
    at exactly ``computed_i``; checking every row is this module's
    deliberate strengthening of the original backend's first-row-only
    check — module docstring).

    ``positions`` is REQUIRED (the original backend silently skipped this
    check when the arg was absent or not a 1-D tensor — a
    call-convention change degraded it to a no-op; here that is a LOUD
    refusal). Accepts a 1-D integer tensor or a list/tuple of ints.
    Rows beyond ``plan.num_tokens`` are tolerated (cudagraph padding
    rows sit past ``num_actual_tokens``); too FEW rows refuse."""
    pos = _int_rows("positions", positions)
    if len(pos) < plan.num_tokens:
        raise LensBatchError(
            f"positions has {len(pos)} rows but the plan covers "
            f"{plan.num_tokens} actual tokens — refusing to index rows "
            "the tensor does not carry")
    for i, seg in enumerate(plan.segments):
        got = pos[seg.start:seg.end]
        want = list(range(seg.computed, seg.computed + seg.q))
        if got != want:
            bad = next(j for j, (g, w) in enumerate(zip(got, want))
                       if g != w)
            raise LensBatchError(
                f"positions tensor disagrees with attention metadata: "
                f"segment {i} row {seg.start + bad} sits at position "
                f"{got[bad]}, expected {want[bad]} (the segment covers "
                f"absolute positions {seg.computed}.."
                f"{seg.computed + seg.q - 1}) — refusing to steer rows "
                "the two metadata sources place differently")


# --- per-request HookContext construction ---------------------------------------


@dataclass(frozen=True)
class SegmentContext:
    """One segment paired with the :class:`~.lens_adapters.HookContext`
    governing EXACTLY its rows: the context applies to
    ``hidden[sc.rows]`` (== ``hidden[sc.segment.start:sc.segment.end]``)
    of the flattened ``[num_tokens, d]`` forward, and to nothing else.

    ``ctx`` is ``None`` exactly for a RESUMED (cache-hit) segment the
    caller did not file as a resumable cell's RESUMED PREFILL: a
    selector-less request's hit (no op ever consumes that context — and
    ``apply_hook_ops`` refuses loudly if a context-needing op somehow
    reaches a ``None`` ctx, so the asymmetry cannot rot silently). A filed
    resumed prefill carries a prefill context with ``cached > 0``."""

    segment: Segment
    ctx: Optional["la.HookContext"]

    @property
    def rows(self) -> slice:
        return self.segment.rows


def build_segment_contexts(plan: BatchPlan, positions: Any,
                           token_ids: Any = None, *,
                           resumed_prefill: Sequence[int] = (),
                           cached_ids: Optional[Mapping[int, Sequence[int]]] = None
                           ) -> Tuple[SegmentContext, ...]:
    """The per-request contexts for one classified forward, in segment
    order. Runs :func:`cross_check_positions` first — the required-
    positions check cannot be skipped on this path.

    A complete-prefill segment yields ``phase="prefill"`` with positions
    ``0..q-1``; a decode segment yields ``phase="decode"`` with its
    single absolute position (``computed``); a resumed segment yields
    ``ctx=None`` — UNLESS its index is in ``resumed_prefill``, the
    caller's list of segments :func:`check_tracked_segment` filed as
    :data:`SEG_RESUMED_PREFILL` (a resumable cell's prompt tail, or its
    fully-cached prompt's decode-shaped last row): those yield
    ``phase="prefill"`` at absolute positions ``computed..seq_len-1``
    with ``cached=computed`` and the matching token-id slice — plus, when
    ``cached_ids`` names the segment, the CACHED region's ids
    (``HookContext.cached_ids``: the request's prompt ids ``[0,
    computed)``, which a ``positions="response"`` op verifies header-free
    before masking — the ``conditional`` policy's per-request decision,
    re-verified at the apply). An index naming a complete prefill
    (``computed == 0``) refuses — nothing was cached, so the caller's
    classification is wrong; ``cached_ids`` for an unfiled segment refuse
    too.

    ``token_ids`` is the flattened per-row token-id tensor (1-D, aligned
    with the hidden rows) or ``None`` when the wiring could not obtain
    it — sliced per segment exactly the way the hidden rows are, into
    ``HookContext.token_ids``. A PRESENT-but-malformed ``token_ids``
    refuses loudly rather than degrading to ``None``: the silent
    downgrade would hide the very wiring breakage it signals, and the
    ops that need ids would then refuse somewhere far from the cause."""
    cross_check_positions(plan, positions)
    ids: Optional[List[int]] = None
    if token_ids is not None:
        ids = _int_rows("token_ids", token_ids)
        if len(ids) < plan.num_tokens:
            raise LensBatchError(
                f"token_ids has {len(ids)} rows but the plan covers "
                f"{plan.num_tokens} actual tokens — a truncated id tensor "
                "cannot be sliced per request; refusing (never degraded "
                "to token_ids=None)")
    resumed_idx = set()
    cached_map = dict(cached_ids or {})
    for i in resumed_prefill:
        j = _to_int("resumed_prefill", i)
        if not 0 <= j < len(plan.segments):
            raise LensBatchError(
                f"resumed_prefill names segment {j} but the plan has "
                f"{len(plan.segments)} segments; refusing")
        if plan.segments[j].computed == 0:
            raise LensBatchError(
                f"resumed_prefill names segment {j}, a COMPLETE prefill "
                "(computed == 0) — nothing of it was cached, so filing it "
                "as resumed is a classification error; refusing")
        resumed_idx.add(j)
    for j in cached_map:
        if _to_int("cached_ids", j) not in resumed_idx:
            raise LensBatchError(
                f"cached_ids names segment {j}, which resumed_prefill does not "
                "file as a resumed prefill; refusing an unattached cached region")
    out: List[SegmentContext] = []
    for i, seg in enumerate(plan.segments):
        cached = 0
        if i in resumed_idx:
            # a RESUMED PREFILL of a resumable cell (module docstring): the
            # prompt tail at its absolute positions, the cached prefix
            # declared, the ids sliced like the hidden rows.
            phase = la.PHASE_PREFILL
            ctx_positions: Tuple[int, ...] = tuple(range(seg.computed,
                                                         seg.seq_len))
            cached = seg.computed
        elif seg.is_resumed:
            # a cache-hit continuation of a request whose ops never run
            # on it (selector-less), or one the caller did not file as a
            # resumed prefill: NO HookContext. A context-needing op
            # reaching this None refuses in apply_hook_ops.
            out.append(SegmentContext(segment=seg, ctx=None))
            continue
        elif seg.is_prefill:
            phase = la.PHASE_PREFILL
            ctx_positions = tuple(range(seg.q))
        else:
            phase = la.PHASE_DECODE
            ctx_positions = (seg.computed,)
        seg_ids = (tuple(ids[seg.start:seg.end])
                   if ids is not None else None)
        region = cached_map.get(i)
        ctx = la.HookContext(phase=phase, positions=ctx_positions,
                             token_ids=seg_ids, cached=cached,
                             cached_ids=(None if region is None
                                         else tuple(int(t) for t in region)))
        out.append(SegmentContext(segment=seg, ctx=ctx))
    return tuple(out)


# --- per-request prefill invariants (the always-on defense) ----------------------


#: what :func:`check_tracked_segment` files a tracked segment as
SEG_PREFILL = "prefill"                    # the complete prompt, q == prompt_len
SEG_DECODE = "decode"                      # one generated row, computed >= prompt_len
SEG_RESUMED_PREFILL = "resumed_prefill"    # the prompt TAIL after a prefix-cache hit
SEGMENT_CLASSES = (SEG_PREFILL, SEG_DECODE, SEG_RESUMED_PREFILL)


def check_tracked_segment(seg: Segment, *, prompt_len: Any,
                          req_id: Any, resumed_ok: bool = False) -> str:
    """The per-request prefill invariants for ONE tracked segment of a
    derived plan, against that request's OWN submitted prompt-token
    length — the always-on defense against the module docstring's two
    shape-invisible failure modes, (a) the 1-token final chunk and (b)
    the preemption-recompute. Needs NO run lifecycle and no
    cross-request state, so it holds under arbitrarily overlapping HTTP
    requests where the run-scoped :class:`PrefillLedger` cannot apply.
    Returns the segment's class (:data:`SEGMENT_CLASSES`) so the caller
    builds the matching context.

    Under :func:`derive_batch_plan`'s lattice ``seg`` is a COMPLETE
    prefill (``computed == 0, q >= 2``), a decode-shaped step
    (``q == 1, computed >= 1``), or — under ``allow_resumed`` — a RESUMED
    shape (``q >= 2, computed > 0``):

    - a complete prefill must have ``q == prompt_len`` exactly —
      ``q > prompt_len`` is a preemption-recompute (generated rows about
      to be steered as prompt), ``q < prompt_len`` is the FIRST chunk of
      a chunked prefill (a partial prompt about to be steered as
      complete);
    - a decode step must have ``computed >= prompt_len`` — a smaller
      ``computed`` is the 1-token final chunk of a chunked prefill (or a
      fully-prefix-cached prompt's first scheduled token): a PROMPT row
      about to be steered as a generated one — UNLESS ``resumed_ok`` and
      ``computed == prompt_len - 1``, the fully-cached prompt of a
      RESUMABLE cell: a one-row RESUMED PREFILL (module docstring);
    - a resumed shape refuses unless ``resumed_ok`` (the caller's fact:
      the cell's ``lens_adapters.resumed_prefill_policy`` is ``always``,
      or ``conditional`` and THIS request's cached region is header-free
      — the worker decides that before calling; an unsafe conditional
      resume never reaches here, it is contained per request) AND the
      slice ends exactly at the prompt end (``computed + q ==
      prompt_len``): overrunning it is a
      preemption recompute under caching (generated rows about to be
      steered as prompt — the same class as (b)), stopping short of it
      is a chunked continuation the pinned geometry never produces.

    Raises :class:`LensBatchError` on any violation; the hook wiring
    condemns the worker on it, exactly as for every other refusal."""
    n = _to_int(f"prompt_len({req_id!r})", prompt_len)
    if seg.is_resumed:
        if not resumed_ok:
            raise LensBatchError(
                f"request {req_id!r}: TRACKED segment arrived RESUMED "
                f"(q={seg.q}, computed={seg.computed}) — a prefix-cache "
                "hit on a steered request whose ops CANNOT resume "
                "(lens_adapters.resumed_prefill_ok is False: a broadcast, "
                "positions='last'/'response', or phase-restricted op). Its "
                "cached rows skipped the forward, so the hooks never fired "
                "on them, and their writer may have edited them "
                "differently; the per-request cache-salt discipline "
                "(proxy._rewrite) that makes this impossible has broken, "
                "and this request's output must be DISCARDED.")
        if seg.seq_len > n:
            raise LensBatchError(
                f"request {req_id!r}: RESUMED segment covers absolute "
                f"positions {seg.computed}..{seg.seq_len - 1} but the "
                f"request's own prompt is {n} tokens — the slice runs past "
                "the prompt into generated rows: a preemption recompute "
                "under caching (the engine re-ran the uncached prompt tail "
                "PLUS already-generated tokens in one forward), and "
                "steering it as a prompt tail would treat generated rows "
                "as prompt: this request's output must be DISCARDED.")
        if seg.seq_len < n:
            raise LensBatchError(
                f"request {req_id!r}: RESUMED segment stops at absolute "
                f"position {seg.seq_len - 1}, SHORT of the request's own "
                f"{n}-token prompt — a chunked-prefill continuation (the "
                "serving config is supposed to make chunking impossible; "
                "this server is not the pinned geometry), not a "
                "prefix-cache resume: this request's output must be "
                "DISCARDED.")
        return SEG_RESUMED_PREFILL
    if n < 2:
        raise LensBatchError(
            f"request {req_id!r}: submitted prompt length {n} < 2 can "
            "NEVER be observed as a complete prefill (the plan lattice "
            "refuses 1-token prefills), so a tracked request with it is "
            "unservable — refuse such prompts at submission instead")
    if seg.is_prefill:
        if seg.q != n:
            if seg.q > n:
                raise LensBatchError(
                    f"request {req_id!r}: complete prefill of {seg.q} "
                    f"tokens (rows {seg.start}:{seg.end}) exceeds the "
                    f"request's own {n}-token prompt — a preemption "
                    "recompute (the engine re-ran prompt + already-"
                    "generated tokens as one fresh prefill), and prompt-"
                    "boundary-dependent steering would treat generated "
                    "rows as prompt: this request's output (and any run "
                    "containing it) must be DISCARDED.")
            raise LensBatchError(
                f"request {req_id!r}: complete-prefill-shaped segment of "
                f"{seg.q} tokens (rows {seg.start}:{seg.end}) is SHORTER "
                f"than the request's own {n}-token prompt — the first "
                "chunk of a chunked prefill (the serving config is "
                "supposed to make chunking impossible; this server is "
                "not the pinned geometry), and steering it as a complete "
                "prompt would mis-place every boundary-dependent edit: "
                "this request's output must be DISCARDED.")
        return SEG_PREFILL
    if seg.computed < n:
        if resumed_ok and seg.computed == n - 1:
            # the FULLY prefix-cached prompt of a resumable cell: vLLM
            # schedules the last prompt token alone (module docstring)
            return SEG_RESUMED_PREFILL
        raise LensBatchError(
            f"request {req_id!r}: decode-shaped segment at absolute "
            f"position {seg.computed} sits INSIDE the request's own "
            f"{n}-token prompt — the 1-token final chunk of a chunked "
            "prefill (or a fully-prefix-cached prompt's first scheduled "
            "token" + (", which a RESUMABLE cell may serve only when it is "
                       "exactly the last prompt token" if resumed_ok else
                       " on a cell whose ops cannot resume") + "), i.e. a "
            "PROMPT row about to be steered as a generated one: this "
            "request's output must be DISCARDED.")
    return SEG_DECODE


# --- the post-run multiset accounting (OPTIONAL in-proc cross-check) --------------


class PrefillLedger:
    """Run-level multiset accounting over COMPLETE prefills — an
    OPTIONAL, additional cross-check against the module docstring's two
    failure modes, (a) the 1-token final chunk of a chunked prefill and
    (b) the preemption-recompute, for harnesses that HAVE a run
    boundary (the in-process exact-replay smoke: one ``llm.generate``
    call == one run). The always-on, per-request defense is
    :func:`check_tracked_segment`, which subsumes both failure modes
    without a run lifecycle; the serving path never arms this ledger
    (requests overlap arbitrarily over HTTP — there is no run), and an
    UNARMED ledger is simply not consulted, never an error.

    Mechanism: :meth:`begin_run` arms a ``Counter`` of credits keyed by
    the SUBMITTED prompt token-lengths. Every complete prefill the plan
    step observes (:meth:`observe`) decrements its length's credit IN
    FLIGHT and refuses immediately when the credit is already spent —
    that is case (b), including the collision where a recompute's
    length matches another submitted prompt (whoever draws second
    over-draws). :meth:`finish_run` then refuses any credit never
    consumed — case (a): the length-N prompt was only ever seen as a
    shorter "prefill" plus decode-shaped remnants.

    REACHABILITY of ``finish_run``: an in-flight :meth:`observe` raise
    means ``finish_run`` is never reached — that is fine, and by
    design: the run already failed and its outputs are already
    condemned. Do NOT call ``finish_run`` from a ``finally`` block,
    where its leftover refusal would mask the in-flight diagnosis; call
    it only at a SUCCESSFUL run end. The exception-path cleanup is
    :meth:`disarm` — it drops the armed credits WITHOUT checking them,
    so an aborted run's leftovers can never poison the next run's
    :meth:`begin_run` with a stale "already armed" refusal.

    Whether a run needs this accounting at all is :func:`ops_need_plan`
    — a purely row-local op sequence is chunking-safe and skips the
    plan machinery entirely."""

    def __init__(self) -> None:
        self._credit: Optional[Counter] = None

    @property
    def armed(self) -> bool:
        return self._credit is not None

    def begin_run(self, prompt_lens: Sequence[int]) -> None:
        """Arm the accounting with the submitted prompt token-lengths."""
        if self._credit is not None:
            raise LensBatchError(
                "PrefillLedger.begin_run while a run is already armed — "
                "re-arming would silently discard outstanding prompt "
                "credits; finish_run() the previous run first")
        lens_list = [_to_int("prompt_lens", n) for n in prompt_lens]
        for n in lens_list:
            if n < 2:
                raise LensBatchError(
                    f"submitted prompt length {n} < 2 can NEVER be "
                    "observed as a complete prefill (the plan lattice "
                    "refuses 1-token prefills), so arming it guarantees a "
                    "corrupt-run refusal at finish_run — refuse such "
                    "prompts at submission instead")
        self._credit = Counter(lens_list)

    def disarm(self) -> None:
        """Drop any armed credits WITHOUT checking them — the
        exception-path cleanup (class docstring). Never a substitute
        for :meth:`finish_run` on a successful run; a no-op when
        nothing is armed, so it is always safe to call."""
        self._credit = None

    def observe(self, plan: BatchPlan) -> None:
        """Consume one credit per complete prefill in ``plan`` — called
        by the plan step on every derived plan OF AN ARMED RUN, so the
        refusal fires on the forward that over-draws. (The hook wiring
        consults the ledger only while :attr:`armed`; a DIRECT unarmed
        call still refuses below — that is a wiring bug, not the
        serving path's legitimate no-run state.)"""
        if self._credit is None:
            raise LensBatchError(
                "PrefillLedger.observe with no armed run — the wiring "
                "must begin_run(prompt_lens) before the first forward; an "
                "unarmed observe means the accounting was silently "
                "skipped, which this module exists to prevent")
        for seg in plan.full_prefills:
            if self._credit[seg.q] <= 0:
                raise LensBatchError(
                    f"complete prefill of {seg.q} tokens (rows "
                    f"{seg.start}:{seg.end}) matches NO outstanding "
                    "submitted prompt length — a preemption recompute "
                    "(the engine re-ran prompt + already-generated tokens "
                    "as one fresh prefill) or a recompute/chunk colliding "
                    "with another prompt's length and over-drawing its "
                    "credit. Prompt-boundary-dependent steering has "
                    "treated generated rows as a prompt (the collision is "
                    "detected only at the over-draw): the run's outputs "
                    "must be DISCARDED.")
            self._credit[seg.q] -= 1

    def finish_run(self) -> None:
        """Refuse any submitted prompt length never observed as exactly
        one complete prefill; disarms the ledger either way. Reached
        only when no :meth:`observe` refused (class docstring)."""
        if self._credit is None:
            raise LensBatchError(
                "PrefillLedger.finish_run with no armed run — begin_run "
                "was never called (or finish_run ran twice); the "
                "accounting cannot vouch for a run it never observed")
        # +Counter drops the fully-consumed (non-positive) entries, so
        # the report names only the lengths still owed a prefill.
        leftover, self._credit = +self._credit, None
        if leftover:
            raise LensBatchError(
                f"run finished with submitted prompt lengths never "
                f"observed as a complete prefill: {dict(leftover)} — each "
                "such prompt was chunked (a length-N prompt split N-1 + 1 "
                "is seen only as an N-1 'prefill' plus a decode-shaped "
                "1-token remnant), prefix-cached, or preempted-and-"
                "recomputed, and any prompt-boundary-dependent steering "
                "it received landed on the wrong rows: the run's outputs "
                "must be DISCARDED.")


# --- when the plan machinery applies ---------------------------------------------


def ops_need_plan(ops: Sequence["la.HookOp"]) -> bool:
    """Whether serving ``ops`` needs :func:`derive_batch_plan`, the
    positions cross-check, and the :class:`PrefillLedger` at all.

    An op needs them exactly when its effect depends on where the
    prompt boundary sits — any phase restriction, any position mode
    other than ``"all"``, any BOS skip. That is precisely
    :func:`lens_adapters.op_needs_context`, which is REUSED rather than
    restated (two copies of the vocabulary would drift). A sequence of
    purely row-local ops (positions ``"all"``, phase ``"both"``, no BOS
    skip) is chunking-safe — every row gets the same edit wherever the
    prompt boundary sits, so mis-segmentation cannot change the math —
    and deliberately skips the whole plan machinery (the original
    backend's asymmetry, driven off the ops instead of a kind list)."""
    return any(la.op_needs_context(op) for op in ops)
