"""Per-arm decode-only (serving-PHASE) capability — the training-side audit, a
CROSS-CHECK reference AND the home of the meaningfulness RULE. NOT the serve gate.

The serve gate is the EMPIRICAL per-object phase probe in
:mod:`sisteer.eval.phase_probe` (measure which positions actually change; compare
the measured position SET against the requested and the method's DEFINED
footprint). A per-arm allowlist goes stale the moment a new arm lands or an arm's
fit-side position support outruns its serve-side helper — so it must not decide
serving. This table records the training side's EXPECTED per-arm behaviour so the
probe's measurement can be CROSS-CHECKED against it: the audit says which arms
SHOULD round-trip, the probe says which DO, and a disagreement is itself a finding
worth surfacing. Do not gate serving on it.

THE TRAP THE PROBE PREVENTS (recorded here for context). Under generated-only
serving the plain base model serves the prompt/prefill and the arm's intervention
context is entered for DECODE forwards only. An arm whose ENTIRE effect is on
prompt/prefill positions therefore becomes a NO-OP under decode-only serving — and
a no-op does NOT error: it returns BASELINE numbers, which read exactly like "this
method collapses when restricted to decode." That is a strong, publishable-
looking, entirely FALSE finding. So an arm with no decode-position analogue must
REFUSE the decode-only cell rather than silently serve a no-op.

THE UNDERLYING RULE — encoded here so it classifies a NEW arm automatically
instead of waiting for someone to remember. A phase cell (a decode-only, or
otherwise cross-phase, serve) is MEANINGFUL only if BOTH hold:

  1. EXPRESSIBLE — the arm can actually apply the requested position mask (else
     :data:`NOT_EXPRESSIBLE`), and its footprint VARIES with the request (else
     :data:`PHASE_INVARIANT`: it applies one hardcoded footprint whatever is
     asked, so there is no phase contrast to measure).
  2. SUBSET — the SERVED footprint is a SUBSET of the method's DEFINED footprint.
     A restriction is meaningful ("does the method survive being narrowed?"); a
     RELOCATION — serving positions the method is defined to AVOID — is a
     different method wearing this arm's name (:data:`MEANINGLESS_SERVE`).
     The subset test is POSITIONAL and has a second half, DERIVATION LOCALITY:
     the edit must be computable FROM the positions it is applied to. A
     compute-once-and-broadcast edit (alphasteer: derived from the last prompt
     token) served masked would apply a prompt-derived vector to generated
     positions — the function changed even though the footprint shrank. No
     current run-list arm has that shape (all compute per position from the
     local activation), so this is a documented criterion
     (``classify_phase_cell(derivation_local=...)``), not a detector.

Checked against the known cases (all four must come out right, which is how we
know it is the right rule):
  - psr* / vector / lofit / lora* / diffmean / bidpo / alphakl_*: DEFINED
    all-position, so a decode-only serve is ``decode ⊆ all`` → meaningful
    RESTRICTION → RUNS (:data:`SUPPORTED`).
  - loreft: DEFINED prompt-only (it intervenes on prompt positions ONLY BY
    IDENTITY — decode steps pass through untouched, that is the method, not a
    default). A decode-only serve is ``decode ⊄ prompt`` → RELOCATION, not
    restriction → REFUSE (:data:`MEANINGLESS_SERVE`). It CAN express the mask and
    its masked FIT stays supported; what it cannot do is MEAN anything served
    decode-only. It refuses at DEFENSE CONSTRUCTION (``SERVE_MASK_REFUSAL`` on
    ``LoReFTObject``, checked in ``jailbreak._set_response_positions`` before the
    mask seams), so the probe sees a construction-time refusal, not a mislabelled
    cell — the "fall back to the declared reason" branch of the rule.
  - antidote: DEFINED last-prompt-token + every decode token, HARDCODED
    (``positions="last"`` inside apply(), not a settable attribute), so its
    footprint does NOT vary with the request → :data:`PHASE_INVARIANT`. An
    ALL-POSITION antidote request ALSO fails SUBSET (``all ⊄ last+decode``), which
    is the independent reason it is a NOVEL arm rather than a phase cell — the
    rule reproduces that conclusion, a good sign it is the right rule.
  - token_bos / token_faithful: the intervention IS a sentinel embedded in the
    PROMPT, with no per-position notion at decode → cannot express a decode mask →
    :data:`NOT_EXPRESSIBLE`. Mechanical incapacity.

THREE DISTINCT REFUSAL REASONS, never one bucket — they fail in different ways and
a reader must not see them as one category: :data:`NOT_EXPRESSIBLE` (mechanical
incapacity), :data:`PHASE_INVARIANT` (hardcoded, no contrast),
:data:`MEANINGLESS_SERVE` (relocation, not restriction).
The subset half is checkable at RUNTIME from the probe's measured footprint
(``measured_served ⊆ defined``); this registry supplies the DEFINED footprint and
the declared reason for arms that refuse before a probe is possible.

Anything NOT registered here is DEFAULT-DENY (:data:`_UNVERIFIED`): decode-only
serving is refused with an "unverified" reason until the arm's decode behaviour is
established in code, because a wrong ``SUPPORTED`` is exactly the false-finding
trap above. That is the fail-safe direction.
"""

from __future__ import annotations

# -- serve-time phase-cell verdicts (the rule's output) --------------------- #
#: served footprint ⊆ defined footprint — a genuine RESTRICTION of the method;
#: the phase cell is meaningful and RUNS.
SUPPORTED = "supported"
#: the arm cannot apply the requested position mask at all (no seam / a
#: prompt-embedded sentinel) — mechanical incapacity. REFUSE.
NOT_EXPRESSIBLE = "not_expressible"
#: the arm applies ONE hardcoded footprint whatever is requested, so there is no
#: phase contrast to measure (no distinct matched vs cross-phase leg). REFUSE.
PHASE_INVARIANT = "phase_invariant"
#: the arm CAN express the mask, but the served footprint is NOT a subset of the
#: method's defined footprint — a RELOCATION onto positions the method is defined
#: to avoid, i.e. a different method under this arm's name. REFUSE.
MEANINGLESS_SERVE = "meaningless_serve"

#: The three distinct refusal reason codes (never collapsed into one bucket).
REFUSAL_REASONS = (NOT_EXPRESSIBLE, PHASE_INVARIANT, MEANINGLESS_SERVE)

# Back-compat alias: the old umbrella "no decode analogue" name maps to the
# precise "cannot express the mask" reason (a superset of the old meaning). Kept
# so any external reference keeps resolving; new code should use the precise code.
NO_DECODE_ANALOGUE = NOT_EXPRESSIBLE

# -- footprint descriptors (what positions a method is defined to touch, or a
#    cell requests). Expanded to a per-region Footprint by
#    phase_probe.expand_footprint. -- #
#: every position (prompt content + assistant header + decode) — the all-position
#: arms, and the ``mask="all"`` request.
FOOTPRINT_ALL = "all"
#: the decode-only SERVE via position_mask (``positions="response"``): the
#: assistant HEADER + every generated token, prompt CONTENT untouched. This is the
#: ``mask="generated"`` request — the header is INCLUDED because it is exactly what
#: shapes the first generated token (that is the property the position-mask path
#: guarantees). Named "response" to match the sweep's decode-only leg, NOT
#: "generated-only" (which would wrongly exclude the header).
FOOTPRINT_RESPONSE = "response"
#: prompt/prefill positions only — content + header, no decode (loreft's
#: defining footprint, by IDENTITY: decode steps pass through untouched).
FOOTPRINT_PROMPT = "prompt"
#: the final prompt token + every decode token (antidote's intrinsic deployment):
#: content untouched, header only its LAST token (partial when H>1 — the
#: discriminator against a full-header decode serve), generated all.
FOOTPRINT_LAST_PLUS_DECODE = "last_prompt+decode"

_UNVERIFIED = (
    NOT_EXPRESSIBLE,
    "decode-only capability UNVERIFIED for this arm — not yet established in the "
    "intervention code; refusing rather than risk serving a silent no-op "
    "(default-deny). Classify it explicitly in decode_capability before serving.",
)


def classify_phase_cell(*, expressible: bool, footprint_varies: bool,
                        served_footprint, defined_footprint,
                        derivation_local: bool = True) -> str:
    """The MEANINGFULNESS rule, encoded so a NEW arm classifies automatically.

    Returns one of :data:`SUPPORTED` / :data:`NOT_EXPRESSIBLE` /
    :data:`PHASE_INVARIANT` / :data:`MEANINGLESS_SERVE`.

    - ``expressible`` — can the arm apply the requested position mask at all?
    - ``footprint_varies`` — does the applied footprint change with the request
      (vs one hardcoded footprint)?
    - ``served_footprint`` / ``defined_footprint`` — position SETS (any objects
      supporting ``<=`` subset, e.g. ``frozenset`` of indices, or a
      :class:`~sisteer.eval.phase_probe.Footprint`). The subset test is what
      separates a meaningful RESTRICTION (``served ⊆ defined``) from a RELOCATION.
    - ``derivation_local`` — the SECOND half of the meaningfulness test: the
      edit must be computable FROM the positions it is applied to. The subset
      test is POSITIONAL and misses a relocation of the FUNCTION: an edit
      computed once at position X and applied at positions Y is a different
      function even when Y sits inside the defined set — the footprint shrank
      but the function changed. The example that motivates it is alphasteer:
      its edit is computed ONCE from the last prompt token and broadcast, so a
      response-masked serve would apply a prompt-DERIVED vector to generated
      positions — a relocation the positional subset test alone would pass.
      (alphasteer is independently ``not_expressible`` here — prefill-only,
      gates on ``hidden.shape[1] <= 1`` — which is why this predicate changes
      NOTHING for any current arm: all eleven run-list arms compute per
      position from that position's own activation — psr gates and alphakl
      adds evaluate locally; vector/diffmean/bidpo/lofit read nothing;
      lora_r1/bidpo_lora are position-wise output edits on their own input;
      antidote is already phase_invariant. The flag defaults local and is a
      documented criterion for a FUTURE compute-once-and-broadcast arm,
      deliberately not a detection mechanism.)

    The order is load-bearing: expressibility gates first (a non-expressible arm
    has no footprint to compare), then phase-invariance (a hardcoded footprint has
    no contrast), then derivation locality (a non-local edit is a relocation of
    the function regardless of the footprint), then the subset test (restriction
    vs relocation of positions)."""
    if not expressible:
        return NOT_EXPRESSIBLE
    if not footprint_varies:
        return PHASE_INVARIANT
    if not derivation_local:
        return MEANINGLESS_SERVE
    if not (served_footprint <= defined_footprint):
        return MEANINGLESS_SERVE
    return SUPPORTED


#: arm -> (status, defined_footprint, expressible, footprint_varies, reason).
#: Keyed on ``methods.ARMS`` names. ``status`` is the verdict the rule yields for
#: a decode-only serve of that arm; ``defined_footprint`` is the method's identity
#: footprint (for the runtime subset test); ``expressible`` / ``footprint_varies``
#: are the rule inputs (so :func:`classify_phase_cell` reproduces the status).
#: ``bidpo_lora`` / ``caa_static_r`` are included.
DECODE_ONLY_CAPABILITY: dict[str, tuple] = {
    # -- SUPPORTED: DEFINED all-position, so decode-only is a real restriction -- #
    "alphakl_null": (
        SUPPORTED, FOOTPRINT_ALL, True, True,
        "per-position ReLU-clamped gate applied at prompt AND decode "
        "(defense_cell CLASS_A_ARMS note; the deployment-matched rung); a "
        "serve-side position mask restricts it to decode."),
    "alphakl_soft": (
        SUPPORTED, FOOTPRINT_ALL, True, True,
        "same per-position gate mechanism as alphakl_null, soft-penalty rung — "
        "acts at every decode step; masked serve = a restriction."),
    "psr_kl_all": (
        SUPPORTED, FOOTPRINT_ALL, True, True,
        "learned per-position gate; positions='response' selects the decode-only "
        "restriction (extraction/psr.py)."),
    "psr_mse_all": (
        SUPPORTED, FOOTPRINT_ALL, True, True,
        "learned per-position gate; positions='response' decode-only restriction "
        "(extraction/psr.py)."),
    # psr_xfer_* (2026-08-28, psr-transfer behavioral eval): cross-model
    # TRANSFERRED PSR units — ordinary PSRObject blobs (kind 'psr'), the
    # SAME learned per-position gate mechanism as psr_kl_all, so the same
    # verdict applies. Registered WITH the cell registration: without an
    # entry, defined_footprint falls to the prompt-only default and every
    # mask='generated' serve of these arms refuses at the phase probe's
    # subset test (the exact trap lora_r1_all hit on 2026-08-28).
    "psr_xfer_from_e2b": (
        SUPPORTED, FOOTPRINT_ALL, True, True,
        "transferred PSR unit (kind 'psr'): learned per-position gate, "
        "positions='response' selects the decode-only restriction — the "
        "psr_kl_all mechanism under a transfer-identity arm name."),
    "psr_xfer_from_e2b_rw1": (
        SUPPORTED, FOOTPRINT_ALL, True, True,
        "ridge-ablation variant of psr_xfer_from_e2b — same PSRObject "
        "mechanism, same verdict."),
    "psr_xfer_from_e4b": (
        SUPPORTED, FOOTPRINT_ALL, True, True,
        "transferred PSR unit (kind 'psr'): learned per-position gate, "
        "positions='response' selects the decode-only restriction — the "
        "psr_kl_all mechanism under a transfer-identity arm name."),
    "psr_xfer_from_e4b_rw1": (
        SUPPORTED, FOOTPRINT_ALL, True, True,
        "ridge-ablation variant of psr_xfer_from_e4b — same PSRObject "
        "mechanism, same verdict."),
    "vector": (
        SUPPORTED, FOOTPRINT_ALL, True, True,
        "additive residual applied at every steered position, including each "
        "KV-cached decode step; masked serve restricts the same edit."),
    "diffmean": (
        SUPPORTED, FOOTPRINT_ALL, True, True,
        "additive CAA-family vector applied all-position (defense_cell diffmean "
        "DEPLOYMENT note); positions='response' persists the decode-only "
        "restriction (extraction/diffmean.py)."),
    "bidpo": (
        SUPPORTED, FOOTPRINT_ALL, True, True,
        "DPO-tuned weights act at every forward, decode included (not a "
        "position-scoped hook); a masked serve is a real restriction."),
    "lora_r1": (
        SUPPORTED, FOOTPRINT_ALL, True, True,
        "LoRA weight delta acts at every forward, decode included."),
    "lofit": (
        SUPPORTED, FOOTPRINT_ALL, True, True,
        "positions='response' selects the decode-only fit/serve restriction "
        "(extraction/lofit.py)."),
    "bidpo_lora": (
        SUPPORTED, FOOTPRINT_ALL, True, True,
        "RankOneLoRAObject — a position-wise o_proj output edit with NO "
        "prefill/decode gating, so decode-only genuinely restricts the same edit "
        "(training lane, f024e12 + position-masked fits ec7a1f2)."),
    "lora_r1_all": (
        SUPPORTED, FOOTPRINT_ALL, True, True,
        "MultiLayerLoRAObject — lora_r1's position-wise o_proj output edit on "
        "EVERY layer (rank-general since 2026-08-28: TrainTest's rank-k "
        "artifacts and the repo's rank-1 fits share the kind), no "
        "prefill/decode gating, PositionMaskedFit seam; decode-only genuinely "
        "restricts the same edit. Registered 2026-08-28 with the cell "
        "registration — without this entry defined_footprint fell to the "
        "prompt-only default and the JB legs of every <si>__lora_r1_all "
        "condition refused at the phase probe's subset test. LEGACY spelling "
        "of lora_r4 (renamed the same day); kept so legacy-named cells keep "
        "their footprint."),
    "lora_r4": (
        SUPPORTED, FOOTPRINT_ALL, True, True,
        "The CANONICAL name of lora_r1_all above (user rename 2026-08-28: "
        "the staged objects are rank 4, so the rank is named and 'all' is "
        "implicit). Identical entry by construction — same object class, "
        "same position-wise o_proj output edit, same PositionMaskedFit "
        "seam; the alias is recorded in methods.LEGACY_ARM_ALIASES."),
    # -- PHASE_INVARIANT: one hardcoded footprint, no phase contrast ---------- #
    "antidote": (
        PHASE_INVARIANT, FOOTPRINT_LAST_PLUS_DECODE, True, False,
        "positions='last' is HARDCODED inside apply() (extraction/antidote.py:"
        "19-21, not a settable attribute), so it serves ONE footprint whatever is "
        "requested: the final real PROMPT token (during prefill) plus every "
        "KV-cached DECODE token. That footprint has no phase CONTRAST — no "
        "distinct matched leg, no distinct cross-phase leg — so it cannot be a "
        "phase cell. Its footprint is ADJACENT to, not identical with, a "
        "cross-phase serve: a cross-phase serve steers the H=4 assistant-header "
        "positions plus generated, antidote steers 1 prompt position plus decode, "
        "and the three-token difference IS the assistant header. Consequence: an "
        "antidote ALL-POSITION cell is ALSO refused (by the mislabelled-cell rule "
        "and by SUBSET — requested all, measured last+decode, all ⊄ last+decode), "
        "which will surprise someone."),
    # -- MEANINGLESS_SERVE: expressible, but a relocation not a restriction --- #
    "loreft": (
        MEANINGLESS_SERVE, FOOTPRINT_PROMPT, True, True,
        "LoReFT intervenes on PROMPT positions only BY IDENTITY — decode steps "
        "pass through untouched, that is the method, not a default. It CAN express "
        "the mask (a set_fit_positions seam) and its masked FIT stays supported; "
        "but a decode-only SERVE is not a RESTRICTION of the method, it is the "
        "COMPLEMENT: the edit relocates onto exactly the positions the method is "
        "defined to AVOID (decode ⊄ prompt), which runs a different method under "
        "LoReFT's name. Refused at defense construction (SERVE_MASK_REFUSAL on "
        "LoReFTObject, extraction/reft.py), so the probe never runs — this is the "
        "declared-reason fallback."),
    # -- NOT_EXPRESSIBLE: cannot apply a decode mask at all ------------------- #
    "alphasteer": (
        NOT_EXPRESSIBLE, FOOTPRINT_PROMPT, False, False,
        "prefill-only / last-token-broadcast / unclamped closed-form Class B "
        "object with NO maskable seam; the edit reaches generation only through "
        "the KV cache and never touches a decode forward, and it cannot be told to "
        "apply a decode mask (defense_cell CLASS_B note)."),
    "caa_static_r": (
        NOT_EXPRESSIBLE, FOOTPRINT_PROMPT, False, False,
        "DEPLOYMENT='prefill_only_resid_pre_broadcast_all_prompt_positions_10layer' "
        "and apply() gates on hidden.shape[1] <= 1, so decode is reached only via "
        "the KV cache and no seam can mask it to decode (extraction/caa_static_r.py"
        ":111 + the shape gate)."),
    "token_bos": (
        NOT_EXPRESSIBLE, FOOTPRINT_PROMPT, False, False,
        "soft-prompt sentinel embedded in the PROMPT (prefill only); the "
        "intervention has no per-position notion at decode, so a decode mask is "
        "inexpressible."),
    "token_faithful": (
        NOT_EXPRESSIBLE, FOOTPRINT_PROMPT, False, False,
        "soft-prompt sentinel embedded in the PROMPT (prefill only); the "
        "intervention has no per-position notion at decode, so a decode mask is "
        "inexpressible."),
}


def decode_capability(arm: str) -> tuple[str, str]:
    """``(status, reason)`` for serving ``arm`` decode-only. Unregistered arms are
    DEFAULT-DENY (:data:`NOT_EXPRESSIBLE` with an 'unverified' reason) — the
    fail-safe direction, since a wrong ``SUPPORTED`` serves a silent no-op that
    reads as a real (false) collapse-at-decode finding."""
    entry = DECODE_ONLY_CAPABILITY.get(arm)
    if entry is None:
        return _UNVERIFIED
    status, _defined, _expr, _var, reason = entry
    return status, reason


def defined_footprint(arm: str) -> str:
    """The method's DEFINED footprint descriptor for ``arm`` (for the runtime
    subset test: ``measured_served ⊆ expand_footprint(defined_footprint(arm))``).
    Unregistered arms default to the most restrictive assumption (prompt-only), so
    any generated-only serve of an unknown arm fails the subset test."""
    entry = DECODE_ONLY_CAPABILITY.get(arm)
    return entry[1] if entry is not None else FOOTPRINT_PROMPT


def refusal_reason_code(arm: str) -> str:
    """The DISTINCT refusal reason code for a decode-only serve of ``arm``
    (:data:`NOT_EXPRESSIBLE` / :data:`PHASE_INVARIANT` / :data:`MEANINGLESS_SERVE`),
    or :data:`SUPPORTED` when the serve is a meaningful restriction. This is what a
    caller records so the three reasons never collapse into one bucket."""
    return decode_capability(arm)[0]


def supports_decode_only(arm: str) -> bool:
    """True iff a decode-only serve of ``arm`` is a MEANINGFUL restriction
    (:data:`SUPPORTED`). Every refusal reason (:data:`NOT_EXPRESSIBLE` /
    :data:`PHASE_INVARIANT` / :data:`MEANINGLESS_SERVE`) is False — antidote
    (phase_invariant) and loreft (meaningless_serve) are NOT servable phase
    cells, a change from the earlier PARTIAL/analogue lumping."""
    return refusal_reason_code(arm) == SUPPORTED


class NoDecodeAnalogueError(ValueError):
    """Raised when a decode-only cell is requested for an arm whose serve is not a
    meaningful restriction — refused rather than served as a no-op / a relocation /
    a phase-invariant footprint. The message carries the DISTINCT reason code."""


def assert_decode_capable(arm: str) -> str:
    """Return :data:`SUPPORTED` for ``arm``, or raise
    :class:`NoDecodeAnalogueError` naming the arm, the DISTINCT reason code, and
    the reason when a decode-only serve is not a meaningful restriction. This is
    the cross-check refusal — the sweep's real gate is the empirical footprint
    probe (:mod:`phase_probe`), but this refuses the same set of arms from the
    registry so a no-op / relocation / phase-invariant serve never returns a
    number that reads as a real collapse-at-decode finding."""
    status, reason = decode_capability(arm)
    if status != SUPPORTED:
        raise NoDecodeAnalogueError(
            f"arm {arm!r} cannot be served as a decode-only phase cell "
            f"[{status}]: {reason} Serving it decode-only would return a number "
            "nobody should compare (a silent no-op, a relocation, or a "
            "phase-invariant footprint). Run this arm all-position, or (if you "
            "must force it) label the output by-construction, never a number.")
    return status
