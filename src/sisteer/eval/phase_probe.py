"""Empirical serving-PHASE verification — measure which positions an object
ACTUALLY steers, never infer it from the arm name or a per-arm allowlist, and
compare the measured position SET against the requested phase AND the method's
DEFINED footprint (never a coarse "all vs response" label).

WHY EMPIRICAL, NOT DECLARATIVE. A per-arm allowlist goes stale the moment a new
arm lands or an arm's fit-side position support outruns its serve-side helper
(exactly what happened to eval/jailbreak.py's object-type gate, which predates the
fit-side PositionMaskedFit extension). Worse, some arms PERSIST their fit positions
in the artifact and re-hydrate applying their own mask (diffmean reads
self.fit_positions; LoReFT treats positions as part of what the object IS), while a
non-persisting arm re-hydrates ALL-POSITION and silently runs the wrong method. So
we LOCATE the applied phase the way the chat-marker fix locates markers: measure
it. One short forward with the object applied and one without, on a synthetic
prompt with a known prompt/generated split; the set of positions whose hidden
state CHANGED is the object's applied footprint, measured rather than asserted.

WHY FOOTPRINT SETS, NOT COARSE LABELS (the antidote trap). A coarse check that
bins into ALL / RESPONSE / PROMPT by "did ANY prompt position change AND did ANY
generated position change" CANNOT tell a full-header decode serve apart from
antidote's ONE-prompt-token + decode footprint: both change some prompt-side and
all generated positions, so both read as the SAME coarse label. antidote served
under an all-position request would then read coarse-ALL and pass AS a matched
all-position cell, when it actually steers last-prompt + decode — a NOVEL footprint,
not all-position. So the measurement is a structured FOOTPRINT over three regions —
CONTENT (the prompt minus the assistant header), the HEADER (the H assistant-header
tokens, located via :mod:`sisteer.chat_markers` so our boundary agrees with the
Petri and fit lanes by construction), and GENERATED — with per-region coverage
(all / partial / none). antidote's HEADER coverage comes back PARTIAL (only the
final header token, not all H) → the measured footprint differs from any clean
phase → refused. The three-token difference between antidote's footprint and a
cross-phase serve IS the assistant header (H=4 header positions + generated for a
cross-phase serve; 1 prompt position + decode for antidote).

THE THREE-WAY CHECK (and the footgun that makes two quantities insufficient).
``_set_response_positions`` now ACTIVELY applies "all" (resetting a persisted
response-fit object) rather than early-returning, so an explicit all-position
request no longer silently serves decode-only. But verification still needs THREE
quantities, all agreeing, because a persisted response object served under an
all-position request would otherwise be recorded as the all-position leg while
actually being the decode-only one — two legs of a comparison identical while
labelled different, corrupting exactly the phase axis the probe protects:

  1. REQUESTED — from the sweep condition (what the cell claims to be).
  2. FIT       — ``positions_at_fit`` from the object's fit-time provenance.
  3. MEASURED  — the empirical footprint probe (which positions actually changed).

TAXONOMY — the phase axis' defining property is ``measured != fit`` (every arm on
the decode-only run list is a fit-ALL-POSITION object SERVED DECODE-ONLY, so
``measured != fit`` is exactly the cross-phase being measured — refusing it would
silently empty the axis):
  1. requested != measured -> MISLABELLED CELL. REFUSE ALWAYS. The row would lie
     about its own phase (a name-based check cannot catch it — every declaration
     involved is individually correct). This is the case the three-way check
     exists for, and the footprint measurement is what makes it catch antidote.
  2. measured ⊄ defined footprint -> RELOCATION (a decode-only serve of a
     prompt-only method, an all-position request of antidote's last+decode).
     REFUSE — the served positions are ones the method is defined to AVOID, i.e.
     a different method under this arm's name (decode_capability.MEANINGLESS_SERVE).
  3. measured == requested, measured != fit, fit all-position -> DELIBERATE
     CROSS-PHASE TRANSFER TEST. RUN IT, labelled ``phase_relation = cross_phase``.
  4. measured == fit == requested -> MATCHED CELL. RUN IT, ``phase_relation =
     matched``.
Default-deny stands for anything unverifiable (no fit provenance, unresolvable
probe). Arms that refuse at DEFENSE CONSTRUCTION (loreft's SERVE_MASK_REFUSAL, the
soft tokens' FIT_MASK_REFUSAL) never reach the probe — their reason is read from
:mod:`sisteer.eval.decode_capability` (the declared-reason fallback).

The ``phase_relation`` label is LOAD-BEARING, not bookkeeping: a matched cell and a
cross-phase cell ANSWER DIFFERENT QUESTIONS, so the aggregate must NEVER pool them.

SCOPE. Decode-only is a ``generate()``-path-only feature. Under the engine-phase
gate the FIRST generated token is produced by the UNSTEERED prefill and steering
begins at token 2 — so a decode-only column is "tokens 2..N steered". A decode-only
cell with max_new_tokens == 1 never enters the ctx and is 100% baseline regardless
of arm — refuse that configuration (see :func:`assert_decode_tokens_reachable`).
This module measures the OBJECT's intrinsic applied footprint (where its ctx/mask
acts on ONE full forward); the token-1 serving-mechanism gate is separate,
documented, not folded in here. The position-mask path DOES shape token 1 (the
assistant header producing the first generated token is steered) — a property the
forward-pair probe can PROVE by measuring the last-prompt/header position as
changed, not merely assume.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# applied-phase labels (match the sweep's mask vocabulary: "all" / "generated")
ALL = "all"
RESPONSE = "response"      # decode/generated-only (the sweep mask "generated")
PROMPT = "prompt"          # prompt/prefill-only
NONE = "none"              # no measurable effect — a no-op
MIXED = "mixed"            # a non-canonical footprint (partial region coverage,
#                            e.g. antidote's last-prompt+decode) — never a clean
#                            phase, so it can never MATCH a requested clean phase.

#: mask value -> canonical phase. The sweep condition mask is "all"/"generated";
#: "generated" is the response/decode phase.
_MASK_TO_PHASE = {"all": ALL, "generated": RESPONSE, "response": RESPONSE,
                  "prompt": PROMPT}

#: per-region coverage values.
_COV_ALL = "all"
_COV_PARTIAL = "partial"
_COV_NONE = "none"
_COV_NA = "n/a"        # the region has no positions on this sequence (vacuous)


class PhaseVerificationError(ValueError):
    """A decode-only/serving-phase check failed — refused rather than served."""


def phase_of_mask(mask: str) -> str:
    """Canonical applied-phase for a sweep condition mask ('all' / 'generated')."""
    return _MASK_TO_PHASE.get(str(mask), str(mask))


def _max_abs_diff(a, b) -> float:
    """max|a - b| over one position's hidden vector, tensor- or list-agnostic."""
    try:
        return float((a - b).abs().max())
    except (AttributeError, TypeError):
        return max(abs(float(x) - float(y)) for x, y in zip(a, b))


def changed_positions(hidden_baseline, hidden_steered, *, atol: float = 1e-5):
    """The SET of position indices whose hidden state changed by more than
    ``atol`` between the without-object and with-object forwards. The raw measured
    footprint, before any region classification."""
    seq = len(hidden_baseline)
    if len(hidden_steered) != seq:
        raise PhaseVerificationError(
            f"probe hidden-state length mismatch: baseline {seq} vs steered "
            f"{len(hidden_steered)} — cannot resolve a footprint")
    return frozenset(
        i for i in range(seq)
        if _max_abs_diff(hidden_steered[i], hidden_baseline[i]) > atol)


# --------------------------------------------------------------------------- #
# Structured footprint over three regions (content | header | generated).      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Footprint:
    """Per-region coverage of a measured or DEFINED footprint over one sequence.

    ``content`` — coverage of the prompt MINUS the H assistant-header tokens
    (``[0, prompt_len - header_len)`` minus ignored/BOS positions);
    ``header`` — coverage of the H assistant-header tokens
    (``[prompt_len - header_len, prompt_len)``); ``generated`` — coverage of
    ``[prompt_len, seq_len)``. Each is ``"all"`` / ``"partial"`` / ``"none"`` /
    ``"n/a"`` (a region with no positions on this sequence). Grouping the HEADER
    as its own region is what tells antidote (one header token) apart from a
    cross-phase serve (all H header tokens) — the difference a coarse label loses.
    """

    content: str
    header: str
    generated: str

    def label(self) -> str:
        """The coarse phase label, or :data:`MIXED` for a non-canonical footprint.

        A canonical footprint has NO partial region and the header grouped with
        the response side: ALL (content+header+generated all steered), RESPONSE
        (content untouched, header+generated all steered — the decode-only serve
        including the header that shapes token 1), PROMPT (content+header steered,
        generated untouched), NONE (nothing). Anything else — a partial region, a
        header that disagrees with generated — is :data:`MIXED`."""
        c, h, g = self.content, self.header, self.generated
        if _COV_PARTIAL in (c, h, g):
            return MIXED
        content_all = c == _COV_ALL
        content_none = c in (_COV_NONE, _COV_NA)
        header_all = h in (_COV_ALL, _COV_NA)
        header_none = h in (_COV_NONE, _COV_NA)
        gen_all = g == _COV_ALL
        gen_none = g in (_COV_NONE, _COV_NA)
        if content_all and header_all and gen_all:
            return ALL
        if content_none and header_all and gen_all:
            return RESPONSE
        if content_all and header_all and gen_none:
            return PROMPT
        if content_none and header_none and gen_none:
            return NONE
        return MIXED

    def __le__(self, other: "Footprint") -> bool:
        """SUBSET: ``self ⊆ other`` region-by-region. A meaningful RESTRICTION has
        the SERVED footprint a subset of the DEFINED one; a RELOCATION does not.
        Per region: ``none``/``n/a`` ⊆ anything; ``all`` ⊆ ``all``; ``partial`` ⊆
        ``all``/``partial``; ``all`` ⊄ ``partial``/``none``."""
        return all(_cov_subset(a, b) for a, b in (
            (self.content, other.content), (self.header, other.header),
            (self.generated, other.generated)))


def _cov_subset(a: str, b: str) -> bool:
    """``a ⊆ b`` for one region's coverage."""
    if a in (_COV_NONE, _COV_NA):
        return True
    if a == _COV_ALL:
        return b == _COV_ALL
    # a == partial
    return b in (_COV_ALL, _COV_PARTIAL)


def footprints_equal(a: Footprint, b: Footprint) -> bool:
    """Region-by-region equality, treating ``n/a`` (an empty region) as matching
    the corresponding non-``partial`` coverage — an empty region carries no
    information, so it must not spuriously distinguish two footprints."""
    return all(_cov_equal(x, y) for x, y in (
        (a.content, b.content), (a.header, b.header),
        (a.generated, b.generated)))


def _cov_equal(x: str, y: str) -> bool:
    if x == y:
        return True
    # n/a (empty region) matches none (no coverage) — both mean "nothing here".
    na_none = {_COV_NA, _COV_NONE}
    return x in na_none and y in na_none


def _region_coverage(changed, lo: int, hi: int, ignore) -> str:
    """Coverage of the half-open region ``[lo, hi)`` (minus ``ignore``) given the
    ``changed`` position set."""
    region = frozenset(i for i in range(max(lo, 0), max(hi, 0)) if i not in ignore)
    if not region:
        return _COV_NA
    hit = region & changed
    if hit == region:
        return _COV_ALL
    if not hit:
        return _COV_NONE
    return _COV_PARTIAL


def measure_footprint(hidden_baseline, hidden_steered, *, prompt_len: int,
                      seq_len: Optional[int] = None, header_len: int = 0,
                      ignore_positions=(), atol: float = 1e-5) -> Footprint:
    """The MEASURED :class:`Footprint` from per-position hidden-state diffs.

    ``hidden_baseline`` / ``hidden_steered`` are per-position hidden states from
    one forward WITHOUT and one WITH the object applied over the SAME synthetic
    sequence. ``prompt_len`` is the FULL prompt/prefill width (the prompt/generated
    split): the assistant HEADER is the prompt's LAST ``header_len`` positions
    (``[prompt_len - header_len, prompt_len)``), its own region so the LABEL can
    count it on the response side (a position-mask serve steers it to shape token
    1) while antidote's single-header-token footprint stays distinguishable.
    Passing the CONTENT boundary here instead of the full width slides the header
    window onto unsteered content and mislabels a clean response serve as MIXED —
    the exact bug the first GPU validation caught (2026-08-21). ``header_len`` (H)
    is the assistant-header token count (from
    :func:`sisteer.chat_markers.derive_assistant_markers`); with ``H=0`` the
    header region is empty and the measurement degenerates to prompt/generated.
    ``ignore_positions`` excludes positions never steered by construction (the true
    BOS under ``skip_bos``), so an all-position skip-BOS serve still reads as
    full CONTENT coverage rather than ``partial``."""
    changed = changed_positions(hidden_baseline, hidden_steered, atol=atol)
    n = seq_len if seq_len is not None else len(hidden_baseline)
    ignore = frozenset(int(i) for i in ignore_positions)
    header_lo = prompt_len - int(header_len)
    return Footprint(
        content=_region_coverage(changed, 0, header_lo, ignore),
        header=_region_coverage(changed, header_lo, prompt_len, ignore),
        generated=_region_coverage(changed, prompt_len, n, ignore),
    )


def expand_footprint(descriptor: str, *, prompt_len: int, seq_len: int,
                     header_len: int = 0) -> Footprint:
    """A DEFINED-footprint descriptor
    (:mod:`sisteer.eval.decode_capability` ``FOOTPRINT_*``) -> the
    :class:`Footprint` it implies over ``(prompt_len, seq_len, header_len)``, for
    the subset test ``measured ⊆ expand_footprint(defined)``.

    - ``all``               -> every region fully covered.
    - ``response``          -> the decode-only serve: HEADER + generated, content
      none (the header is included — it shapes token 1 under position-mask serving).
    - ``prompt``            -> content + header, generated none (loreft's identity).
    - ``last_prompt+decode``-> the final prompt token (header, PARTIAL when H>1) +
      generated; content none. antidote's intrinsic deployment."""
    from .decode_capability import (FOOTPRINT_ALL,
                                    FOOTPRINT_LAST_PLUS_DECODE, FOOTPRINT_PROMPT,
                                    FOOTPRINT_RESPONSE)
    has_content = (prompt_len - int(header_len)) > 0
    has_header = int(header_len) > 0
    has_gen = seq_len > prompt_len
    _c = (lambda cov: cov if has_content else _COV_NA)
    _h = (lambda cov: cov if has_header else _COV_NA)
    _g = (lambda cov: cov if has_gen else _COV_NA)
    if descriptor == FOOTPRINT_ALL:
        return Footprint(_c(_COV_ALL), _h(_COV_ALL), _g(_COV_ALL))
    if descriptor == FOOTPRINT_RESPONSE:
        return Footprint(_c(_COV_NONE), _h(_COV_ALL), _g(_COV_ALL))
    if descriptor == FOOTPRINT_PROMPT:
        return Footprint(_c(_COV_ALL), _h(_COV_ALL), _g(_COV_NONE))
    if descriptor == FOOTPRINT_LAST_PLUS_DECODE:
        # the final prompt token is the last HEADER token; with H>1 that is
        # partial header coverage (the discriminator against a cross-phase serve).
        return Footprint(_c(_COV_NONE),
                         _h(_COV_PARTIAL if int(header_len) > 1 else _COV_ALL),
                         _g(_COV_ALL))
    raise ValueError(f"unknown footprint descriptor {descriptor!r}")


def measure_applied_phase(hidden_baseline, hidden_steered, *, prompt_len: int,
                          header_len: int = 0, ignore_positions=(),
                          atol: float = 1e-5) -> str:
    """The coarse MEASURED phase label (:data:`ALL` / :data:`RESPONSE` /
    :data:`PROMPT` / :data:`NONE` / :data:`MIXED`) — :meth:`Footprint.label` of
    the measured footprint. :data:`MIXED` is a non-canonical footprint (a partial
    region, e.g. antidote's last-prompt+decode) that can never match a clean
    requested phase, so a coarse comparison still refuses it as mislabelled."""
    return measure_footprint(
        hidden_baseline, hidden_steered, prompt_len=prompt_len,
        header_len=header_len, ignore_positions=ignore_positions,
        atol=atol).label()


#: phase_relation labels (LOAD-BEARING — the aggregate must never pool these).
MATCHED = "matched"          # measured == fit == requested
CROSS_PHASE = "cross_phase"  # fit-all object served at a narrower phase (cross-phase)


def verify_phase(requested, fit, measured, *, what: str = "") -> str:
    """Verify the three phases (coarse labels) and RETURN the ``phase_relation``
    label (:data:`MATCHED` / :data:`CROSS_PHASE`), or raise
    :class:`PhaseVerificationError`. See the module taxonomy. This is the
    coarse-label 3-way; :func:`verify_footprint` is the fuller SET-based gate that
    ALSO applies the subset (relocation) test. ``measured == MIXED`` never equals a
    clean requested phase, so it refuses here as mislabelled.

    ``requested`` (condition), ``fit`` (positions_at_fit provenance), ``measured``
    (empirical probe). Any ``None`` or a no-op probe (``measured == NONE``) ->
    default-deny."""
    tag = f" [{what}]" if what else ""
    for name, val in (("requested", requested), ("fit", fit),
                      ("measured", measured)):
        if val is None:
            raise PhaseVerificationError(
                f"serving phase unverifiable{tag}: {name} phase is unknown "
                "(None) — refusing (default-deny; e.g. the object exposes no "
                "positions_at_fit provenance, or the probe could not resolve).")
    if measured == NONE:
        raise PhaseVerificationError(
            f"the object probes as a NO-OP{tag} (no position's hidden state "
            "changed) — refusing rather than serving baseline numbers that read "
            "as a real collapse-at-phase finding (default-deny).")
    # CASE 1 — MISLABELLED. Always refuse; the row would lie about its phase.
    if requested != measured:
        raise PhaseVerificationError(
            f"cell MISLABELLED{tag}: requested {requested}-position, measured "
            f"{measured} — the row would lie about its own phase (e.g. a "
            "persisted response-mode object served under an all-position "
            "request silently serves decode-only; or an intrinsic-footprint arm "
            "like antidote measured as a non-canonical MIXED footprint). Refusing "
            "rather than recording it under the wrong phase.")
    # CASE 4 — MATCHED. measured == fit == requested.
    if measured == fit:
        return MATCHED
    # CASE 3 — measured == requested != fit. A CROSS-PHASE serving, allowed at the
    # axis level ONLY for a fit-all-position object (Class A/B).
    if fit == ALL:
        return CROSS_PHASE
    raise PhaseVerificationError(
        f"cross-phase refused{tag}: measured {measured} == requested but the "
        f"object was fitted {fit} (not all-position). A fit-all object may be "
        "served cross-phase (cross-phase test), but a decode-only/Class-C fit must "
        "be served MATCHED. This is also the signature of a NON-PERSISTING "
        "artifact that re-hydrated all-position instead of applying its "
        f"{fit}-mode fit positions — refusing rather than running the wrong "
        "method.")


def verify_footprint(*, requested: Footprint, defined: Footprint,
                     measured: Footprint, fit: Optional[Footprint] = None,
                     what: str = "") -> str:
    """The SET-based serving-phase gate: compare measured position footprints (not
    coarse labels), applying BOTH the mislabel test (``measured == requested``) AND
    the subset/relocation test (``measured ⊆ defined``). Returns the
    ``phase_relation`` label (:data:`MATCHED` / :data:`CROSS_PHASE`) or raises
    :class:`PhaseVerificationError` with the DISTINCT failure.

    ``requested`` — the footprint the cell claims (from the mask + H). ``defined``
    — the method's identity footprint (:func:`expand_footprint` of the arm's
    ``decode_capability.defined_footprint``). ``measured`` — the probe. ``fit`` —
    the fit-time footprint (optional; needed to distinguish matched from
    cross_phase). This is what catches antidote: its measured footprint (partial
    header + generated) equals NEITHER a clean all nor a clean response request,
    and ``all ⊄ last+decode`` fails the subset test besides."""
    tag = f" [{what}]" if what else ""
    if measured.label() == NONE:
        raise PhaseVerificationError(
            f"the object probes as a NO-OP{tag} (no position's hidden state "
            "changed) — refusing rather than serving baseline numbers that read "
            "as a real collapse-at-phase finding (default-deny).")
    # CASE 1 — MISLABELLED. The measured footprint is not the one the cell claims.
    if not footprints_equal(measured, requested):
        raise PhaseVerificationError(
            f"cell MISLABELLED{tag}: requested footprint {requested}, measured "
            f"{measured} — the row would lie about its own phase (e.g. antidote's "
            "one-prompt-token+decode footprint under an all-position or "
            "decode-only request: the measured header coverage is PARTIAL, not "
            "the full header a clean phase has). Refusing rather than recording it "
            "under the wrong phase.")
    # CASE 2 — RELOCATION. The served footprint is not a subset of the method's
    # defined footprint: a decode-only serve of a prompt-only method, or an
    # all-position request of an intrinsic last+decode arm. A different method
    # under this arm's name (decode_capability.MEANINGLESS_SERVE).
    if not (measured <= defined):
        raise PhaseVerificationError(
            f"serve is a RELOCATION{tag}, not a restriction: measured footprint "
            f"{measured} is NOT a subset of the method's defined footprint "
            f"{defined} — the served positions are ones the method is defined to "
            "AVOID (e.g. a decode-only serve of a prompt-only method, or an "
            "all-position request of antidote's last-prompt+decode). Refusing: "
            "this runs a different method under this arm's name.")
    if fit is None:
        # No fit provenance to classify matched vs cross_phase.
        raise PhaseVerificationError(
            f"serving phase unverifiable{tag}: no fit footprint (default-deny; "
            "the object exposes no positions_at_fit provenance).")
    # CASE 4 — MATCHED. measured == fit == requested.
    if footprints_equal(measured, fit):
        return MATCHED
    # CASE 3 — cross-phase, allowed only for a fit-all-position object.
    if fit.label() == ALL:
        return CROSS_PHASE
    raise PhaseVerificationError(
        f"cross-phase refused{tag}: measured footprint {measured} == requested "
        f"but the object was fitted {fit} (not all-position). A fit-all object may "
        "be served cross-phase; a decode-only/Class-C fit must be served MATCHED. "
        "This is also the signature of a NON-PERSISTING artifact that re-hydrated "
        "all-position instead of applying its fit positions.")


def assert_decode_tokens_reachable(max_new_tokens) -> None:
    """Refuse a decode-only cell whose ``max_new_tokens <= 1``. Under the
    engine-phase gate the single generated token is produced by the UNSTEERED
    prefill and the arm ctx is never entered, so the cell is 100% baseline
    regardless of arm — a guaranteed-false "the method does nothing" result.
    Refuse the CONFIGURATION rather than emit that number."""
    if int(max_new_tokens) <= 1:
        raise PhaseVerificationError(
            f"decode-only cell with max_new_tokens={max_new_tokens} never enters "
            "the intervention ctx (the first generated token comes from the "
            "unsteered prefill; steering begins at token 2). It is 100% baseline "
            "regardless of arm — refusing rather than emitting a guaranteed-false "
            "collapse-at-decode number. Use max_new_tokens >= 2.")


def probe_object_phase(capture_hidden, *, prompt_len: int, seq_len: int,
                       header_len: int = 0, ignore_positions=(),
                       atol: float = 1e-5) -> str:
    """Serve-time empirical probe -> the coarse phase label. ``capture_hidden(
    applied: bool)`` runs ONE short forward over a fixed synthetic sequence and
    returns that forward's per-position hidden states WITH the object's ctx applied
    (``applied=True``) and WITHOUT (``applied=False``). Kept caller-injected (a
    callable, not a model handle) so the pure classification is unit-testable
    without a model and the real forward pair lives at the one GPU call site."""
    return probe_object_footprint(
        capture_hidden, prompt_len=prompt_len, seq_len=seq_len,
        header_len=header_len, ignore_positions=ignore_positions,
        atol=atol).label()


def classify_batched_row_mask(row_absdiff, *, header_cols, pad_cols,
                             steer_atol: float = 0.5) -> dict:
    """Classify ONE batch row's per-column max-abs hidden diff into the mask it
    actually applied, MAGNITUDE-aware (not a binary atol). Returns the booleans
    plus the separating magnitudes, so a caller records evidence rather than a
    verdict.

    WHY MAGNITUDE-AWARE (finding, slurm 173568 on H200 bf16): a binary
    ``diff > 1e-5`` conflates REAL steering (magnitude ~= the serving scale,
    several units) with H200 CUDA bf16 BATCH NON-INVARIANCE — steering one row
    perturbs a causally-independent row's columns (incl. its padding) at the
    kernel-noise level (~1e-3), because batched bf16 matmul reductions are
    value-dependent (independently reproduced on a bare engine with no
    steering). That noise is FIVE+ orders below the real edit, so a
    threshold between them (``steer_atol``, default 0.5) separates a genuine
    mask-in-padding (a real mis-mask, magnitude ~scale) from kernel churn. The
    magnitudes are returned so the separation is auditable and a too-close call
    is visible rather than silently bucketed.

    ``row_absdiff`` is a length-S sequence of per-column max-abs diffs;
    ``header_cols`` / ``pad_cols`` are index sets for THIS row."""
    diff = [float(x) for x in row_absdiff]
    steered = {c for c in range(len(diff)) if diff[c] > steer_atol}
    header = set(header_cols)
    pad = set(pad_cols)
    def _mx(cols):
        return max((diff[c] for c in cols), default=0.0)
    def _mn(cols):
        return min((diff[c] for c in cols), default=0.0)
    off_header = set(range(len(diff))) - header
    return {
        "header_steered_exactly": steered == header,
        "mask_in_padding": bool(steered & pad),
        "min_header_diff": round(_mn(header), 5),
        "max_pad_diff": round(_mx(pad), 5),
        "max_offheader_diff": round(_mx(off_header), 5),
        "steer_atol": steer_atol,
    }


def probe_object_footprint(capture_hidden, *, prompt_len: int, seq_len: int,
                           header_len: int = 0, ignore_positions=(),
                           atol: float = 1e-5) -> Footprint:
    """Serve-time empirical probe -> the structured :class:`Footprint`. Same
    ``capture_hidden(applied)`` contract as :func:`probe_object_phase`; returns the
    footprint (for :func:`verify_footprint`'s subset test) rather than the coarse
    label."""
    baseline = capture_hidden(False)
    steered = capture_hidden(True)
    return measure_footprint(baseline, steered, prompt_len=prompt_len,
                             seq_len=seq_len, header_len=header_len,
                             ignore_positions=ignore_positions, atol=atol)
