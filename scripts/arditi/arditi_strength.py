#!/usr/bin/env python
"""The Arditi strength lane's ONE table of strengths, view names and paths.

DELIBERATELY THE SAME GRAMMAR AS ``scripts/alphasteer_f22/as22_strength.py``:
the strength frontier's Figure 5 joins three method lanes, and two sibling
lanes spelling their staged ids differently is a trap for whoever writes the
join. So: view ``ar_d<tok>``, staged id ``<view>__arditi``, per-target bank
``<root>/<target>/strength``, and the dose parses out of the VIEW name, which
is what the paper reader reads. The predecessor's spelling (``s0p1__arditi``
under ``<root>/<target>``) worked and is deliberately abandoned before
anything was staged, because it cost nothing to change then and would cost a
special case forever.

THE DOSE UNIT IS A FRACTION OF THE RESIDUAL STREAM — and
this REPLACED an earlier nominal-raw-coefficient ladder, so read the unit, not
the number. The served edit is

    alpha * ||h||_ref * r_hat

the UNIT refusal direction scaled so its magnitude is a fixed fraction of the
residual stream's own norm at the served layer. ``alpha`` is therefore the DOSE:
the fraction of the residual stream the edit adds. That is what makes it
comparable across models whose RAW direction norms span 3.29 (qwen35_4b) to
278.2 (nemotron3nano4b) — a factor of 85 — and comparable in principle to any
other method reported in the same unit.

``||h||_ref`` is MEASURED, per target, at the served layer, on the evaluation
prompt distribution under the served rendering (scripts/arditi/measure_href.py).
Not on the fit data and not on the paper's splits: this lane has already been
burned by that exact substitution, where the analogous quantity read 1.09 on the
fit capture against 0.75 on eval prompts and reversed a written conclusion.

THE GRID: alpha in {-0.10, 0.05, 0.10, 0.20, 0.40} — from below the AlphaSteer
object's own working point (which measures 0.03-0.14 in this unit depending on
attack family) to well past where degeneration began in this campaign (around
a median of 0.15), because that span is what the frontier's tails need. The
NEGATIVE rung is the sign control: if the direction is what we think it is,
steering against it should REDUCE refusal rather than do nothing. All five are
baked and screened: the stop rule governs whether a rung proceeds to the
EXPENSIVE legs, never whether it is screened.

THE UNIT IS DATA, NOT LORE. Two sibling lanes spelling different units with the
same letter is a join that succeeds and means the wrong thing, so the view
letter is `r` (residual-norm fraction) where the LoRA lane's is `d`, AND an
explicit ``dose_unit`` key rides in the roster, the pins and every object
sidecar. See DOSE_UNIT below for where it cannot ride and why.

A DEVIATION FROM ARDITI, NAMED: the paper's activation-addition half uses the RAW
difference of means at coeff +-1.0, precisely so that the coefficient's magnitude
is the measured distance between the class means (verified in their code —
``select_direction`` returns the candidate unmodified and only the ABLATION hook
normalises). We serve the unit direction at a measured fraction of ||h|| instead.
The GEOMETRY is identical (same direction, same layer, same positions, same
phases); only the magnitude convention differs, and it differs in order to make
one number mean one thing across four models.

ONLY ACTIVATION ADDITION IS ON THIS AXIS. Arditi is two operations
sharing one direction: activation addition (``h += c*r`` at one layer, every
position, prefill AND decode) and directional ablation
(``h -= r_hat r_hat^T h`` at every residual-writing site). Only the first has
a coefficient that IS a dose. Scaling a projection interpolates between "no
removal" and "full removal" of a subspace, and a NEGATIVE scale amplifies the
component rather than reversing a steer, so the two can never share an axis
labelled "strength". No ablation object exists in this lane.

THE STRENGTH IS BAKED INTO THE BYTES, never a serve-time flag: the served
``OP_ADD_VECTOR`` carries ``coeff = base_scale = 1.0`` and the lens's
``effective_scale`` is binary 0/1, so a rung is ``rs[fit_layer] <- alpha *
||h||_ref * r_hat`` and each rung is a different object with its own sha.

THE LADDER SERVES PREFILL-ONLY, WHICH IS A DECLARED DEVIATION FROM ARDITI
 — see ``SERVED_KIND`` and ``DEVIATIONS_FROM_THE_PAPER`` below for
the three declarations, the kind that makes prefill-only reachable, and the
layer convention that inverts with it.

Import-light on purpose (stdlib only): the producers run on the grading host and the
stager drives clusters over ssh; neither needs torch to know where things go.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from arditi_targets import CELL_ID, TARGETS, paper_lane  # noqa: F401  (re-exported)

#: the dose grid, in view order. alpha = the fraction of
#: the residual stream's norm the edit adds.
ALPHAS: Tuple[float, ...] = (-0.10, 0.05, 0.10, 0.20, 0.40)
#: the old name, kept pointing at the same tuple so no reader silently gets the
#: retired raw-coefficient ladder
STRENGTHS = ALPHAS

#: THE FIT-TIME LADDER PROBE'S COEFFICIENTS — RAW multipliers of the fitted
#: direction, and NOT a dose in this lane's unit. Deliberately its own name
#.
#:
#: The probe in ``fit_arditi.py`` runs ``h += c * direction`` with the RAW
#: direction, at fit time, before ``||h||_ref`` has been measured — so it cannot
#: be run at the served grid, whose values are fractions of a quantity the fit
#: job does not yet know. Feeding :data:`ALPHAS` here would run the probe at
#: raw coefficients of 0.05-0.40, which on nemotron3nano4b (``||r|| = 278``) is a
#: wildly different intervention from the rung of the same number: two units,
#: one symbol, the exact trap the `r`-vs-`d` view letter exists to prevent.
#:
#: These are the values the campaign's four existing fits were probed at, so a
#: fifth target's probe row is COMPARABLE with theirs; reproducing that
#: comparison is the probe's only job. It is a DIAGNOSTIC for ranking rungs by
#: risk, never a gate and never quotable as a served number.
#:
#: (``fit_arditi.py`` imported this grid from ``arditi_targets`` until the
#: strength grammar moved out on 2026-09-10, and the import was never
#: repointed — so the fit script has raised ``ImportError`` on every branch
#: since. Found 2026-09-12 while adding the fifth target;.)
LADDER_PROBE_COEFFS: Tuple[float, ...] = (-0.5, 0.1, 0.3, 0.5, 1.0)

#: THE UNIT, as a string, used verbatim everywhere it appears. `r` in the view
#: name is this unit; the LoRA lane's `d` is a write-basis multiplier and
#: AlphaSteer's is a nominal base_scale, so the letters must not be shared.
DOSE_UNIT = "residual_norm_fraction"
DOSE_UNIT_DESCRIPTION = (
    "alpha = ||edit|| / ||h||_ref at the served layer, where ||h||_ref is the median residual-"
    "stream norm measured on the campaign's confirmatory goal set plus benign controls under the "
    "served rendering. The served vector is alpha * ||h||_ref * r_hat with r_hat the unit Arditi "
    "refusal direction.")
#: WHERE `dose_unit` CANNOT RIDE, and why — so nobody re-attempts it. The Petri
#: MANIFEST cannot carry it: capeval.manifest refuses unknown keys at BOTH levels
#: (`_check_unknown_keys(raw, ("petri",))` and again against PetriSpec's fields),
#: and PetriSpec has no spare field. Adding one would change a schema every
#: campaign manifest is validated against and whose hash is baked into every
#: recorded audit id — a deliberate shared change for all three lanes at once,
#: not something a method lane does to itself. It rides instead in the roster
#: (top level AND per view), the pins, the serve config, and every object's
#: fit_provenance.json / provenance / staged_from.json, all of which travel with
#: the object and are read by any join.
DOSE_UNIT_NOT_IN_MANIFEST = (
    "capeval.manifest.load_petri_spec refuses unknown keys and PetriSpec has no field for it; "
    "changing that schema would move every campaign manifest's harness hash")

# --- the served KIND and the served PHASE -------------------------------------
#
# THE LADDER SERVES PREFILL-ONLY, and that is a DECLARED DEVIATION
# FROM ARDITI — see DEVIATIONS_FROM_THE_PAPER below. Two consequences live here
# because they are VOCABULARY, not implementation.

#: the on-disk kind of every object of this lane. `static_r` is the ONE
#: registered adapter that emits an additive residual vector at PREFILL ONLY
#: (`lens_adapters._adapt_static_r`); `alphasteer` is a matmul op and `loreft`
#: an affine subspace op, so neither can carry this intervention.
#: `methods._KIND_LOADERS["static_r"]` is the additive alias that makes such a
#: blob hydratable; the CLASS's own on-disk kind is `caa_static_r` and
#: the adapter REFUSES that spelling, which is why the alias had to exist.
#:
#: THE LAYER CONVENTION INVERTS WITH THE KIND, and that is the trap of the
#: swap. A `vector_all` blob stores the SERVED block-output layer and the
#: adapter serves it as stored. A `static_r` blob at `site="pre"` stores the
#: FIT layer and the adapter SHIFTS it (`_block_output_layer` -> l-1). So `rs`
#: is keyed by the FIT layer and the served block output lands at
#: `fit_layer - 1` — the same site the retired `vector_all` object hooked.
#: Storing the served layer here would steer one block EARLIER than the fit,
#: with every other signal looking healthy.
SERVED_KIND = "static_r"
#: the blob's own deployment keys, cross-checked by the adapter against the
#: resolved spec (`_spec_check`). `prefill_all` + `pre` is the ONE combination
#: that yields an all-position, prefill-only add at a block output.
SERVED_STEER_POSITIONS = "prefill_all"
SERVED_SITE = "pre"
#: the served PHASE, as data. `lens_adapters.PHASE_PREFILL`'s value, restated
#: so this table needs no torch import; asserted equal to the adapter's
#: constant in tests/test_arditi_strength_objects.py.
SERVED_PHASE = "prefill"
#: the served POSITION mask and BOS convention — UNCHANGED from Arditi, whose
#: hook adds at every position of the forward, BOS included. Only the PHASE
#: deviates.
SERVED_POSITIONS = "all"
SERVED_SKIP_BOS = False
#: THE TWO KEYS A static_r BLOB MUST NOT CARRY. `lens_adapters.
#: _BLOB_POSITION_SCHEMAS` declares no position-record schema for this kind, so
#: `recorded_fit_positions` REFUSES a static_r blob carrying `positions` or
#: `fit_positions` ("an undeclared vocabulary read as any particular one is a
#: guess"). The retired vector_all blob carried BOTH; the producer asserts
#: their absence rather than relying on remembering.
FORBIDDEN_BLOB_KEYS: Tuple[str, ...] = ("positions", "fit_positions")

#: THE THREE DECLARED DEVIATIONS, as data rather than prose, so they ride any
#: row or artifact that names this lane an "Arditi baseline". A reader
#: must not be able to infer from our output that we ran their method
#: unmodified.
DEVIATIONS_FROM_THE_PAPER: Tuple[Tuple[str, str], ...] = (
    ("phase",
     "PREFILL-ONLY. Arditi's activation-addition hook is a plain forward "
     "pre-hook, ungated on sequence length, so his intervention fires at the "
     "prefill AND at every decode step. We serve the prefill only, so "
     "generated tokens see the edit ONLY through the steered prompt KV cache: "
     "we apply half his intervention. Chosen to PHASE-MATCH the AlphaSteer "
     "lane, because both measure the same shared dose pool and a both-phases "
     "Arditi against a prefill-only AlphaSteer would vary two things at once."),
    ("rendering",
     "apply_chat_template through the harness's one renderer "
     "(sisteer.models.build_chat_prompt), not their hardcoded per-family "
     "template strings."),
    ("unit",
     "the UNIT direction at a MEASURED fraction of ||h||, not the raw "
     "difference of means. Arditi's addition half uses the raw vector at coeff "
     "+-1.0 (verified in their code: only the ABLATION hook normalises), "
     "precisely so the coefficient's magnitude IS the measured distance "
     "between the class means. The GEOMETRY is identical — same direction, "
     "layer, positions and BOS convention; only the magnitude convention "
     "differs, adopted so one number means one thing across four models whose "
     "||r|| spans 3.29 to 278.2."),
)

#: THE ROW CONTRACT. `phase` is a REQUIRED row field beside
#: `dose_unit`: a row omitting either is MALFORMED, never silently defaulted.
#: Same reasoning that put `dose_unit` there — a convention living only in
#: prose is one a join cannot read — and of the three deviations the phase is
#: the only one that changes what the model computes at decode time.
ROW_REQUIRED_FIELDS: Tuple[str, ...] = ("dose_unit", "phase")

#: MEASURED 2026-09-11, and carried beside every number: ||h|| is not
#: constant across tokens, so `alpha` is the dose at the MEDIAN row and drifts
#: elsewhere. gemma4e4b 1.29x and qwen35_4b 1.55x p95/p05 are tight;
#: glm46vflash spans 8.78x (its realised dose at the p95 token is ~0.15x alpha)
#: and nemotron3nano4b 2.76x. Neither of those two may be described as "dosed
#: at alpha" unqualified.
DOSE_AXIS_IS_A_MEDIAN = (
    "alpha is the realised dose ||edit||/||h|| at the MEDIAN row of the reference pool, not a "
    "constant: the edit magnitude is fixed while ||h|| varies by token and by prompt, so the "
    "realised dose is |alpha| * ||h||_ref / ||h||. Every row carries the measured p05/p95 spread "
    "and the dose realised at each end; a target whose spread is wide must never be described as "
    "'dosed at alpha' without it.")
#: above this p95/p05 spread the single scalar is a POOR summary and the
#: qualifier is MANDATORY rather than advisory (glm46vflash 8.78x and
#: nemotron3nano4b 2.76x exceed it; gemma4e4b 1.29x and qwen35_4b 1.55x do not)
DOSE_SPREAD_QUALIFIER_ABOVE = 2.0

VIEW_PREFIX = "ar_"
#: `r` = residual-norm fraction (see DOSE_UNIT)
VIEW_UNIT_LETTER = "r"
#: the per-target bank subdir — as22_strength.BANK_SUBDIR's spelling, so both
#: lanes' banks are <root>/<target>/strength and one reader walks both
BANK_SUBDIR = "strength"
#: the object root under $SHARED on either cluster (our own; nothing of the
#: wave's or the AlphaSteer campaign's is touched)
BANK_ROOT = "exp/objects/arditi_strength"
#: the run-tag namespace: arditi_strength_<target>_<view> (spec: Tag = <target>_<view>)
RUN_TAG_PREFIX = "arditi_strength"

#: THE DECODE CAPS THIS LANE CARRIES — all three the campaign body's, unchanged.
#: BoN is 512 plus a 512 -> 4,096 CLIMB, NOT a native 4,096 pass, and that holds
#: for every method lane: Figure 5 plots defence against capability with METHOD
#: as the series, so a BoN component that is a native pass for one method and a
#: climb counterfactual for another makes the methods incomparable with each
#: other — which is the one comparison the figure exists to make. The estimand
#: belongs to the FIGURE, not to the lane. The climb tooling also refuses a
#: store whose cap is not 512.
#:
#: BON_MAX_NEW_TOKENS IS THE LANE DEFAULT, NOT EVERY TARGET'S CAP. The BoN
#: instrument is a per-TARGET campaign fact and is not this lane's to choose:
#: llama8b has NO 512 store, and the canonical-battery rule makes it a NATIVE
#: 4,096 pass for every lane on that target, so its body carries 4,096 and so
#: must our config for it. The per-target value is READ OFF EACH TARGET'S WAVE
#: BODY (``arditi_targets.body_bon_cap``) into the roster's
#: ``targets.<t>.bon_max_new_tokens``, with the route beside it; this constant
#: remains what a CLIMBED target must equal, and the config generator refuses a
#: climbed target whose cap is anything else. The estimand still belongs to the
#: figure — what changed is that for llama8b the figure's route IS the native
#: pass, for all three method lanes alike.
BON_MAX_NEW_TOKENS = 512
TARGET_MAX_NEW_TOKENS = 512
STATIC_MAX_NEW_TOKENS = 4096

# =============================================================================
# THE SECOND DOSE UNIT: raw_coefficient
# =============================================================================
#
# Everything ABOVE this line is the residual-norm-fraction unit and is
# UNCHANGED, to the letter: same grid, same `r`, same `static_r`, same
# prefill-only phase, same bank root, same forbidden blob keys. The lane's
# transplanted tests are the gate on that, and nothing below re-enters anything above.
#
# WHY A SECOND UNIT AND NOT A FLAG. This unit restores Arditi's OWN dose
# convention FOR ONE ARM on ONE model: the RAW difference of means at
# coefficient `c` (`c = 1.0` the paper's nominal), added at EVERY position of
# EVERY forward. That is a different dose in a different unit served by a
# different adapter at a different phase — four facts at once — so it gets its
# own unit record rather than a parameter on this one. The lane already paid
# once for two units sharing a symbol (the retired `ar_d` ladder, module header).
#
# PRESERVE 1 — THE REFUSAL STAYS TOTAL, PER UNIT. Each unit's row contract
# refuses everything but its own: `assert_row_fields` (above) still refuses any
# phase but `prefill` and any unit but `residual_norm_fraction`, and
# :func:`assert_row_fields_raw_coefficient` refuses any phase but `both` and any
# unit but `raw_coefficient`. Neither was relaxed to admit the other. Two strict
# units beat one permissive one; a dispatcher (:func:`assert_row_fields_in`)
# chooses between them by the unit the row DECLARES, and a row declaring no unit
# is refused by both.
#
# PRESERVE 2 — THE UNIT LETTER IS LOAD-BEARING. `ar_c<c>` against
# `ar_r<alpha>`: two doses of the same number in different units are a join that
# succeeds and means the wrong thing. The view letter is the unit, so the letter
# is what a reader (and every path below) keys on.

#: the two units this module knows, by the string that rides every row
UNIT_RESIDUAL_NORM_FRACTION = DOSE_UNIT
UNIT_RAW_COEFFICIENT = "raw_coefficient"
DOSE_UNITS: Tuple[str, ...] = (UNIT_RESIDUAL_NORM_FRACTION, UNIT_RAW_COEFFICIENT)

RAW_COEFFICIENT_DESCRIPTION = (
    "c = the multiplier on the RAW Arditi difference of means: the served vector is c * r_raw at "
    "the served layer, with c = 1.0 the paper's nominal (arXiv 2406.11717; verified in their code, "
    "where select_direction returns the candidate unmodified and only the ABLATION hook "
    "normalises). Unlike residual_norm_fraction this is NOT normalised by ||h||, so the same c on "
    "two models is two different interventions -- which is why this unit is served on ONE model.")

#: the ladder. The 0.1 floor keeps the bracket around the prefill-only arm's
#: equivalent, c ~ 0.142, from being degenerate. A refinement round may add
#: rungs; the inventory a reader consumes is DECLARED, never inferred from what
#: happens to be on disk.
RAW_COEFFS: Tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 1.0)

#: the Class B cell id this unit's objects are served under — a NEW id, so the
#: prefill-only `arditi` cells keep their identity and nothing aliases two
#: interventions under one name across the export's supersession mechanisms.
RAW_CELL_ID = "arditi_allpos"
#: `c` = raw coefficient. NOT `r` (residual-norm fraction) and NOT `d` (the
#: retired raw ladder's letter, and the LoRA lane's write-basis multiplier).
RAW_VIEW_UNIT_LETTER = "c"
#: the SERVED kind: `vector_all` is the registered adapter (`lens_adapters.
#: _adapt_vector_all`) that emits an additive residual vector at EVERY position
#: of BOTH phases, which is Arditi's own ungated forward pre-hook. `static_r`
#: cannot carry it: that adapter is prefill-only by construction.
RAW_SERVED_KIND = "vector_all"
#: the served PHASE, as data. `lens_adapters.PHASE_BOTH`'s value, restated so
#: this table needs no torch import; asserted equal to the adapter's constant by
#: test, exactly as SERVED_PHASE is.
RAW_SERVED_PHASE = "both"
RAW_SERVED_POSITIONS = "all"
RAW_SERVED_SKIP_BOS = False
#: the FIT leg: fitted at every position, deployed at every position. `lora_export
#: ._residual_vector_all` REFUSES a `response` fit served at `all`, so this is a
#: fact the blob must carry rather than a default anyone may assume.
RAW_FIT_POSITIONS = "all"
#: THE BLOB-KEY RULE, INVERTED FROM static_r's, and that inversion is the point.
#: `lens_adapters._BLOB_POSITION_SCHEMAS` declares NO position schema for
#: `static_r`, so :data:`FORBIDDEN_BLOB_KEYS` refuses those two keys there; it
#: DOES declare one for `vector_all`, whose deployment is read from exactly those
#: two keys. A blob carrying neither is a vector_all object whose position
#: semantics are a guess, so here they are REQUIRED. Same strictness, opposite
#: sign: this is why the two units' contracts could not be merged into one
#: permissive rule (Preserve 1 above).
RAW_REQUIRED_BLOB_KEYS: Tuple[str, ...] = ("positions", "fit_positions")
#: the stored-layer convention INVERTS WITH THE KIND, and confusing the two
#: steers a block early with every other signal healthy. A `static_r` blob at
#: site="pre" stores the FIT layer and the adapter shifts it to fit_layer - 1; a
#: `vector_all` blob stores the SERVED block-output layer and the adapter serves
#: it as stored. So on qwen35_4b (fit layer 13, served block output 12) the
#: static_r rungs key `rs` by 13 and these rungs key `vectors` by 12.
RAW_STORED_LAYER_IS_SERVED_LAYER = True
#: this unit's own bank root and per-target subdir: a NEW namespace, so nothing
#: of the prefill-only lane's is read or written and nothing existing is
#: overwritten. `<root>/<target>/ladder/cells/<staged id>`.
RAW_BANK_ROOT = "exp/objects/arditi_allpos"
RAW_BANK_SUBDIR = "ladder"
RAW_RUN_TAG_PREFIX = "arditi_allpos"
#: the generator of this unit's serve config / manifest pairs, named so a job's
#: "missing config" fatal points at THIS unit's producer and never at the
#: prefill-only lane's (whose ``make_arditi_configs.py`` refuses an ``ar_c`` view
#: by design). It is Phase B's output -- the raw unit's twin of
#: ``make_arditi_configs.py`` over ``configs/arditi_allpos_roster.yaml`` -- and
#: is NOT on this tree until the ladder is baked; the shell says so when the
#: file is absent.
RAW_CONFIG_GENERATOR = "scripts/arditi/make_arditi_allpos_configs.py"

#: THE DEVIATIONS THIS UNIT DOES **NOT** HAVE, as data beside
#: :data:`DEVIATIONS_FROM_THE_PAPER`, so a row of this unit cannot be read as
#: carrying the prefill-only lane's declarations. Phase and unit are CORRECTED
#: here; the rendering deviation remains, because it is the harness's one
#: renderer and not a choice this arm makes.
RAW_DEVIATIONS_FROM_THE_PAPER: Tuple[Tuple[str, str], ...] = (
    ("rendering",
     "apply_chat_template through the harness's one renderer "
     "(sisteer.models.build_chat_prompt), not their hardcoded per-family "
     "template strings."),
)
#: the CROSS-MODEL caveat every row of this unit carries: the
#: other three families' Arditi arms are the prefill-only residual_norm_fraction
#: ones, so a table placing them side by side is comparing two interventions.
RAW_CROSS_MODEL_CAVEAT = (
    "Served in the paper's own configuration -- the raw difference-of-means vector at coefficient "
    "c, added at every position of every forward (prefill AND decode). The other families' Arditi "
    "arms are served PREFILL-ONLY at alpha, a fraction of the residual norm. The two are not the "
    "same intervention and their doses are not the same quantity.")


class StrengthRefusal(SystemExit):
    def __init__(self, msg: str) -> None:
        super().__init__(f"arditi_strength: REFUSED — {msg}")


def validate_strength(s) -> float:
    """An alpha is a finite, non-zero real number. 0 is refused (a zero edit is
    the `none` anchor, not a dose); bools, NaN and infinities are refused;
    anything that is not a number is refused."""
    if isinstance(s, bool) or not isinstance(s, (int, float)):
        raise StrengthRefusal(f"strength {s!r} is not a number")
    f = float(s)
    if not math.isfinite(f):
        raise StrengthRefusal(f"strength {s!r} is not finite")
    if f == 0.0:
        raise StrengthRefusal("strength 0 is refused — a zero edit is the `none` anchor, not a dose")
    return f


def parse_strength(text: str) -> float:
    try:
        return validate_strength(float(str(text).strip()))
    except ValueError as e:
        raise StrengthRefusal(f"strength {text!r} does not parse as a number") from e


def format_strength(s) -> str:
    """`1.0` -> `1`, `0.1` -> `0p1`, `-0.5` -> `m0p5`, `0.25` -> `0p25`."""
    f = validate_strength(s)
    text = f"{abs(f):.10g}"
    if "e" in text or "E" in text:
        raise StrengthRefusal(
            f"strength {s!r} has no positional decimal spelling ({text}); refusing to name a view for it")
    return ("m" if f < 0 else "") + text.replace(".", "p")


def view_name(s) -> str:
    """`ar_r0p1` — the unit letter is `r` (residual-norm fraction), NOT the LoRA
    lane's `d`: same grammar, different unit, so the letter says which."""
    return f"{VIEW_PREFIX}{VIEW_UNIT_LETTER}{format_strength(s)}"


def strength_of_view(view: str) -> float:
    """The inverse of :func:`view_name`; refuses anything that is not the
    CANONICAL spelling of its alpha (`ar_r0p10`, `ar_r+0p1`, a foreign
    prefix), so a typo can never serve a dose by accident."""
    if not isinstance(view, str) or not view.startswith(VIEW_PREFIX + VIEW_UNIT_LETTER):
        raise StrengthRefusal(
            f"view {view!r} is not a dose view of this lane ({VIEW_PREFIX}{VIEW_UNIT_LETTER}...). "
            f"Note `{VIEW_PREFIX}d...` was this lane's RETIRED raw-coefficient spelling and is "
            "refused on purpose: the same number means a different intervention in that unit")
    body = view[len(VIEW_PREFIX) + 1:]
    neg = body.startswith("m")
    if neg:
        body = body[1:]
    if not body or not all(c.isdigit() or c == "p" for c in body):
        raise StrengthRefusal(f"view {view!r}: dose spelling {body!r} is not digits with one `p`")
    s = parse_strength(("-" if neg else "") + body.replace("p", "."))
    if view_name(s) != view:
        raise StrengthRefusal(f"view {view!r} is not the canonical spelling of {s!r} ({view_name(s)})")
    return s


def is_anti_refusal(s) -> bool:
    """A negative alpha is the fitted direction REVERSED — the SIGN CONTROL: if
    the direction is what we think it is, steering against it should REDUCE
    refusal rather than do nothing. The same geometry, a different intervention;
    labelled as such everywhere and never averaged with, or interpolated
    through, the positive rungs."""
    return validate_strength(s) < 0


def assert_row_fields(row, where: str) -> None:
    """Refuse a row of this lane that omits ``dose_unit`` or ``phase``, or
    spells either wrong.

    Called by EVERY writer of this lane — the producer's three sidecars and its
    ledger, the stager's pins and staging-manifest rows, the roster generator —
    so the contract is enforced at the write rather than remembered. A missing
    field is NOT filled in with a default: a defaulted phase is exactly the
    silent-fallback shape this lane keeps being bitten by, and the value it
    would default to (``prefill``) is the very deviation we are obliged to
    declare."""
    if not hasattr(row, "get"):
        raise StrengthRefusal(
            f"{where}: expected a mapping row, got {type(row).__name__}")
    missing = [k for k in ROW_REQUIRED_FIELDS if k not in row]
    if missing:
        raise StrengthRefusal(
            f"{where}: row omits the REQUIRED field(s) {missing} (every row of this lane "
            f"carries {list(ROW_REQUIRED_FIELDS)}). Such a row is malformed, not defaulted: "
            "the dose number means nothing without its unit, and the served phase is a "
            "DECLARED DEVIATION from Arditi that a join must be able to read")
    if row["dose_unit"] != DOSE_UNIT:
        raise StrengthRefusal(f"{where}: dose_unit {row['dose_unit']!r} != {DOSE_UNIT!r}")
    if row["phase"] != SERVED_PHASE:
        raise StrengthRefusal(
            f"{where}: phase {row['phase']!r} != {SERVED_PHASE!r} — this lane serves "
            "prefill-only on every rung of every target, so a row claiming another phase "
            "describes an object this lane never baked")


def realised_dose(alpha, h_ref: float, h: float) -> float:
    """The dose actually realised at a row whose residual norm is ``h``:
    ``|alpha| * ||h||_ref / h``. Equals ``|alpha|`` at ``h == ||h||_ref`` BY
    CONSTRUCTION, which is what makes the served re-measurement a
    self-consistency check rather than a new assumption."""
    a = abs(validate_strength(alpha))
    for name, x in (("h_ref", h_ref), ("h", h)):
        if not (isinstance(x, (int, float)) and not isinstance(x, bool)
                and math.isfinite(float(x)) and float(x) > 0):
            raise StrengthRefusal(f"realised_dose: {name} must be positive and finite, got {x!r}")
    return a * float(h_ref) / float(h)


def dose_spread_block(alpha, h_ref_record: dict) -> dict:
    """The spread half of a row, from a ``measure_href.py`` record's ``h_ref``
    block: what the dose ACTUALLY is at the tails, and whether the single
    scalar is a poor enough summary that the qualifier is mandatory.

    Read off the record that describes the quantity, never recomputed from a
    remembered number."""
    h = {k: float(h_ref_record[k]) for k in ("value", "p05", "p95")}
    spread = h["p95"] / h["p05"]
    return {
        "h_ref": h["value"], "h_ref_p05": h["p05"], "h_ref_p95": h["p95"],
        "h_ref_spread_p95_over_p05": spread,
        # a SMALL ||h|| means a LARGE realised dose, so p05 of ||h|| is the
        # high-dose tail; naming both ends by the ||h|| quantile they come
        # from keeps that from being read backwards
        "realised_dose_at_h_p05": realised_dose(alpha, h["value"], h["p05"]),
        "realised_dose_at_h_p95": realised_dose(alpha, h["value"], h["p95"]),
        "dose_axis": "median",
        "dose_axis_note": DOSE_AXIS_IS_A_MEDIAN,
        "spread_qualifier_mandatory": bool(spread > DOSE_SPREAD_QUALIFIER_ABOVE),
    }


def staged_id_of_strength(s) -> str:
    """`<view>__arditi` — as22_strength.staged_id's grammar."""
    return f"{view_name(s)}__{CELL_ID}"


#: as22_strength spells this `staged_id`; both names resolve here so a reader
#: of either lane finds the same function.
staged_id = staged_id_of_strength


def strength_of_staged_id(sid: str) -> float:
    suffix = f"__{CELL_ID}"
    if not isinstance(sid, str) or not sid.endswith(suffix):
        raise StrengthRefusal(f"staged id {sid!r} does not end in {suffix!r}")
    return strength_of_view(sid[: -len(suffix)])


def run_tag(target: str, s) -> str:
    paper_lane(target)                 # refuses an unknown target loudly
    return f"{RUN_TAG_PREFIX}_{target}_{view_name(s)}"


# --- the objects root layout --------------------------------------------------

def strength_bank_dir(root: Path, target: str) -> Path:
    """`<root>/<target>/strength/` — what make_bank_view.py is passed (`--bank`)."""
    return Path(root) / target / BANK_SUBDIR


def strength_cells_dir(root: Path, target: str) -> Path:
    return strength_bank_dir(root, target) / "cells"


def strength_cell_dir(root: Path, target: str, s) -> Path:
    return strength_cells_dir(root, target) / staged_id(s)


def strength_view_dir(root: Path, target: str, s) -> Path:
    """The per-strength DEFENSES DIR a standup is pointed at
    (`--defenses-dir` / `TT_TWO_SI_DEFENSES_DIR`):
    `<root>/<target>/strength/cells_view/<view>` (built by make_bank_view.py)."""
    return strength_bank_dir(root, target) / "cells_view" / view_name(s)


# --- the generated files ------------------------------------------------------

def serve_config_rel(target: str, s) -> str:
    return f"configs/arditi_strength_{target}_{view_name(s)}.yaml"


def manifest_rel(target: str, s) -> str:
    return f"configs/capeval/manifests/arditi_strength_{target}_{view_name(s)}.yaml"


def defenses_dir_placeholder(target: str, s) -> str:
    """The config's `defenses_dir` value: the wave's `out/...` placeholder
    shape, resolved on the cluster by the job's TT_TWO_SI_DEFENSES_DIR export to
    $SHARED/exp/objects/arditi_strength/<target>/strength/cells_view/<view>."""
    return f"out/{RUN_TAG_PREFIX}/{target}/{BANK_SUBDIR}/cells_view/{view_name(s)}"


# =============================================================================
# THE UNIT-KEYED SURFACE (arditi-qwen-all-positions)
# =============================================================================
#
# One record per unit, and every path, name and contract above derived from it
# by the SAME arithmetic the existing functions use. The existing functions are
# untouched and remain the residual-norm-fraction unit's entry points;
# `tests/test_arditi_allpos_arm.py` pins each of them EQUAL to its unit-keyed
# twin over the whole probe grid, so the two can never drift apart silently.
#
# A caller that knows its unit calls the `_in` form. A caller that reads a unit
# off a row calls it with that string, and an unknown unit REFUSES rather than
# defaulting to either — a defaulted unit is the silent-fallback shape this lane
# keeps being bitten by, and it would pick a dose convention on the reader's
# behalf.


@dataclass(frozen=True)
class DoseUnitSpec:
    """One dose unit, whole: what the number means, what the letter is, what is
    served, and where it lives. Every field is DATA a row or an artifact can
    carry, so a join never has to know which lane wrote it."""

    unit: str
    description: str
    view_letter: str
    cell_id: str
    served_kind: str
    served_phase: str
    served_positions: str
    served_skip_bos: bool
    #: the fit leg, where the kind records one (`vector_all`); None where the
    #: kind declares no position schema at all (`static_r`)
    fit_positions: Optional[str]
    #: blob keys this kind's adapter REQUIRES / REFUSES — exactly one of the two
    #: is non-empty per unit, and which one is the inversion documented above
    required_blob_keys: Tuple[str, ...]
    forbidden_blob_keys: Tuple[str, ...]
    #: True when the blob stores the SERVED block layer (`vector_all`); False
    #: when it stores the FIT layer and the adapter shifts it (`static_r`)
    stored_layer_is_served_layer: bool
    bank_root: str
    bank_subdir: str
    run_tag_prefix: str
    grid: Tuple[float, ...]
    #: the producer of this unit's serve config / manifest pairs (what a job's
    #: "missing config" refusal names -- each unit its own, never the other's)
    config_generator: str


UNITS: Dict[str, DoseUnitSpec] = {
    UNIT_RESIDUAL_NORM_FRACTION: DoseUnitSpec(
        unit=UNIT_RESIDUAL_NORM_FRACTION, description=DOSE_UNIT_DESCRIPTION,
        view_letter=VIEW_UNIT_LETTER, cell_id=CELL_ID,
        served_kind=SERVED_KIND, served_phase=SERVED_PHASE,
        served_positions=SERVED_POSITIONS, served_skip_bos=SERVED_SKIP_BOS,
        fit_positions=None,
        required_blob_keys=(), forbidden_blob_keys=FORBIDDEN_BLOB_KEYS,
        stored_layer_is_served_layer=False,
        bank_root=BANK_ROOT, bank_subdir=BANK_SUBDIR,
        run_tag_prefix=RUN_TAG_PREFIX, grid=ALPHAS,
        config_generator="scripts/arditi/make_arditi_configs.py"),
    UNIT_RAW_COEFFICIENT: DoseUnitSpec(
        unit=UNIT_RAW_COEFFICIENT, description=RAW_COEFFICIENT_DESCRIPTION,
        view_letter=RAW_VIEW_UNIT_LETTER, cell_id=RAW_CELL_ID,
        served_kind=RAW_SERVED_KIND, served_phase=RAW_SERVED_PHASE,
        served_positions=RAW_SERVED_POSITIONS, served_skip_bos=RAW_SERVED_SKIP_BOS,
        fit_positions=RAW_FIT_POSITIONS,
        required_blob_keys=RAW_REQUIRED_BLOB_KEYS, forbidden_blob_keys=(),
        stored_layer_is_served_layer=RAW_STORED_LAYER_IS_SERVED_LAYER,
        bank_root=RAW_BANK_ROOT, bank_subdir=RAW_BANK_SUBDIR,
        run_tag_prefix=RAW_RUN_TAG_PREFIX, grid=RAW_COEFFS,
        config_generator=RAW_CONFIG_GENERATOR),
}


def unit_spec(unit: str) -> DoseUnitSpec:
    """The unit's record. An unknown unit REFUSES — never a default: picking one
    would pick a dose convention, a served phase and a bank root on the caller's
    behalf, and the two conventions differ in all three."""
    try:
        return UNITS[unit]
    except (KeyError, TypeError):
        raise StrengthRefusal(
            f"dose unit {unit!r} is not one of this lane's units {list(DOSE_UNITS)} — the unit says "
            "what the number MEANS (a fraction of the residual norm, or a raw multiplier of the "
            "difference of means) and there is no defaulting between them") from None


def unit_of_cell_id(cell_id: str) -> str:
    """The unit a served cell id belongs to. The cell id is the export's and the
    registry's handle on the object, so this is how a reader with only a row's
    cell id finds the convention its dose is in."""
    for spec in UNITS.values():
        if spec.cell_id == cell_id:
            return spec.unit
    raise StrengthRefusal(
        f"cell id {cell_id!r} is not a cell of this lane "
        f"({sorted(s.cell_id for s in UNITS.values())})")


def unit_of_view(view: str) -> str:
    """The unit whose LETTER the view carries (`ar_r...` / `ar_c...`), found by
    asking each unit's own strict reader; a view of no unit — the retired `ar_d`
    ladder included — REFUSES. The shell's `ar_unit_of_view` is this call."""
    for unit in DOSE_UNITS:
        try:
            strength_of_view_in(unit, view)
        except StrengthRefusal:
            continue
        return unit
    raise StrengthRefusal(
        f"view {view!r} is not a dose view of any unit of this lane ({list(DOSE_UNITS)})")


def view_name_in(unit: str, s) -> str:
    """`ar_r0p1` (residual-norm fraction) / `ar_c0p1` (raw coefficient). The
    LETTER IS THE UNIT: the same number under two letters is two doses, and a
    join that drops the letter silently pools them (Preserve 2 above)."""
    return f"{VIEW_PREFIX}{unit_spec(unit).view_letter}{format_strength(s)}"


def strength_of_view_in(unit: str, view: str) -> float:
    """The inverse of :func:`view_name_in`, refusing anything that is not the
    CANONICAL spelling of its dose in THIS unit — including the OTHER unit's
    letter, which is the whole point: `ar_r0p1` handed to the raw-coefficient
    reader is a real view of a real object at a dose this unit's number does not
    describe, so it must refuse rather than parse the digits."""
    spec = unit_spec(unit)
    prefix = VIEW_PREFIX + spec.view_letter
    if not isinstance(view, str) or not view.startswith(prefix):
        others = {u: VIEW_PREFIX + s.view_letter for u, s in UNITS.items() if u != unit}
        raise StrengthRefusal(
            f"view {view!r} is not a dose view of the {spec.unit} unit ({prefix}...). The other "
            f"unit(s) spell {others} and are refused here on purpose: the same number in another "
            f"unit is another intervention. ({VIEW_PREFIX}d... is the lane's RETIRED raw ladder.)")
    body = view[len(prefix):]
    neg = body.startswith("m")
    if neg:
        body = body[1:]
    if not body or not all(c.isdigit() or c == "p" for c in body):
        raise StrengthRefusal(f"view {view!r}: dose spelling {body!r} is not digits with one `p`")
    s = parse_strength(("-" if neg else "") + body.replace("p", "."))
    if view_name_in(unit, s) != view:
        raise StrengthRefusal(
            f"view {view!r} is not the canonical spelling of {s!r} ({view_name_in(unit, s)})")
    return s


def staged_id_in(unit: str, s) -> str:
    """`<view>__<cell id>`: `ar_r0p1__arditi` / `ar_c0p1__arditi_allpos`. The
    CELL ID is in the staged id as well as the letter, so a staged dir names
    both the dose and the intervention it is a dose of."""
    return f"{view_name_in(unit, s)}__{unit_spec(unit).cell_id}"


def strength_of_staged_id_in(unit: str, sid: str) -> float:
    suffix = f"__{unit_spec(unit).cell_id}"
    if not isinstance(sid, str) or not sid.endswith(suffix):
        raise StrengthRefusal(f"staged id {sid!r} does not end in {suffix!r}")
    return strength_of_view_in(unit, sid[: -len(suffix)])


def bank_dir_in(unit: str, root: Path, target: str) -> Path:
    """`<root>/<target>/<bank subdir>/` — what make_bank_view.py is passed."""
    return Path(root) / target / unit_spec(unit).bank_subdir


def cells_dir_in(unit: str, root: Path, target: str) -> Path:
    return bank_dir_in(unit, root, target) / "cells"


def cell_dir_in(unit: str, root: Path, target: str, s) -> Path:
    return cells_dir_in(unit, root, target) / staged_id_in(unit, s)


def view_dir_in(unit: str, root: Path, target: str, s) -> Path:
    """The per-dose DEFENSES DIR a standup is pointed at."""
    return bank_dir_in(unit, root, target) / "cells_view" / view_name_in(unit, s)


def run_tag_in(unit: str, target: str, s) -> str:
    paper_lane(target)                 # refuses an unknown target loudly
    return f"{unit_spec(unit).run_tag_prefix}_{target}_{view_name_in(unit, s)}"


def serve_config_rel_in(unit: str, target: str, s) -> str:
    return f"configs/{run_tag_in(unit, target, s)}.yaml"


def manifest_rel_in(unit: str, target: str, s) -> str:
    return f"configs/capeval/manifests/{run_tag_in(unit, target, s)}.yaml"


def defenses_dir_placeholder_in(unit: str, target: str, s) -> str:
    """The config's `defenses_dir`: the wave's `out/...` placeholder shape,
    resolved on the cluster by TT_TWO_SI_DEFENSES_DIR."""
    spec = unit_spec(unit)
    return (f"out/{spec.run_tag_prefix}/{target}/{spec.bank_subdir}/cells_view/"
            f"{view_name_in(unit, s)}")


def assert_row_fields_raw_coefficient(row, where: str) -> None:
    """THE RAW-COEFFICIENT UNIT'S OWN ROW CONTRACT — equally strict, and TOTAL
    (Preserve 1 above). It refuses everything but its own unit and its own
    phase, exactly as :func:`assert_row_fields` refuses everything but
    ``residual_norm_fraction`` / ``prefill``. Neither rule was widened to admit
    the other: a permissive rule that accepted both would accept the ONE
    combination that is a real object served under the wrong label — a
    prefill-only static_r blob filed as a row of this arm.

    Called by every writer of this unit, so the contract is enforced at the
    write rather than remembered."""
    if not hasattr(row, "get"):
        raise StrengthRefusal(f"{where}: expected a mapping row, got {type(row).__name__}")
    missing = [k for k in ROW_REQUIRED_FIELDS if k not in row]
    if missing:
        raise StrengthRefusal(
            f"{where}: row omits the REQUIRED field(s) {missing} (every row of this lane "
            f"carries {list(ROW_REQUIRED_FIELDS)}). Such a row is malformed, not defaulted: "
            "the dose number means nothing without its unit, and the served phase is what "
            "distinguishes the paper's own configuration from the prefill-only deviation")
    if row["dose_unit"] != UNIT_RAW_COEFFICIENT:
        raise StrengthRefusal(
            f"{where}: dose_unit {row['dose_unit']!r} != {UNIT_RAW_COEFFICIENT!r} — a row of the "
            f"{UNIT_RESIDUAL_NORM_FRACTION!r} unit is a DIFFERENT dose of a DIFFERENT "
            "intervention and is refused here rather than pooled")
    if row["phase"] != RAW_SERVED_PHASE:
        raise StrengthRefusal(
            f"{where}: phase {row['phase']!r} != {RAW_SERVED_PHASE!r} — this unit serves the "
            "paper's own ungated hook, prefill AND every decode step, on every rung; a row "
            "claiming another phase describes an object this unit never baked")


#: each unit's TOTAL row contract, by unit — a table rather than an if/else, so
#: the dispatch is total by construction: a third unit gets its own entry or a
#: KeyError, never whichever branch happens to be last
_ROW_CONTRACTS = {UNIT_RESIDUAL_NORM_FRACTION: assert_row_fields,
                  UNIT_RAW_COEFFICIENT: assert_row_fields_raw_coefficient}


def assert_row_fields_in(unit: str, row, where: str) -> None:
    """Dispatch a row to ITS unit's total contract. The unit argument is the
    unit the CALLER expects; a row declaring another one is refused by that
    unit's own rule, never quietly re-dispatched to the rule it would pass."""
    _ROW_CONTRACTS[unit_spec(unit).unit](row, where)


def assert_blob_keys_in(unit: str, blob_keys, where: str) -> None:
    """The unit's BLOB-KEY contract, on the keys of the object about to be
    written or served. `static_r` FORBIDS ``positions`` / ``fit_positions``
    (its adapter declares no position schema, so an undeclared vocabulary read
    as any particular one is a guess); `vector_all` REQUIRES them (its adapter
    reads the deployment out of exactly those two keys). Same strictness,
    opposite sign — which is why a blob that satisfies one unit's rule is a
    refusal under the other's, and why the arm validator routes by unit."""
    spec = unit_spec(unit)
    keys = set(blob_keys)
    present = [k for k in spec.forbidden_blob_keys if k in keys]
    if present:
        raise StrengthRefusal(
            f"{where}: blob carries {present}, which the {spec.served_kind!r} adapter refuses "
            "(this kind declares no position schema)")
    absent = [k for k in spec.required_blob_keys if k not in keys]
    if absent:
        raise StrengthRefusal(
            f"{where}: blob lacks {absent}, which the {spec.served_kind!r} adapter READS to "
            "resolve its deployment — an absent key would be read as this kind's default "
            f"rather than as this object's fact (want positions={spec.served_positions!r}, "
            f"fit_positions={spec.fit_positions!r})")


#: THE PINS' ABSOLUTE LAYER BINDING (arditi-qwen-all-positions Phase A review, round 2, item 2).
#: Every rung's entry in the raw unit's pins (configs/arditi_allpos_pins.json) carries the FIT
#: RECORD's own layers -- arditi_fit.json `selected.{layer, served_layer, direction_sha256}` -- so
#: a serve-time gate can hold an object to WHERE the fit put it, not merely to a relation between
#: its own indices. The relational checks (stored == served == layers[0] == fit.selected.layer - 1)
#: pass a coherent object baked a block off: fit 12 / served 11 / stored 11 is exactly as
#: self-consistent as fit 13 / served 12 / stored 12, and only the pins, derived from the fit
#: record and never typed, can tell the two apart. The producer (Phase B) writes these four keys
#: from the fit record; every consumer below REFUSES a pins entry lacking any of them BY NAME,
#: because a pins file without layers cannot certify one. `direction_sha256` is the fit's binding
#: to the direction's VALUES (sha256_text(json.dumps(direction.tolist())), plan 5.1), which the
#: provenance's `fit.selected` carries verbatim.
RAW_PIN_LAYER_KEYS: Tuple[str, ...] = ("fit_layer", "served_layer", "stored_layer", "direction_sha256")


def raw_pin_entry(pins: dict, target: str, staged_id: str, where: str) -> dict:
    """The pins entry for ONE staged rung, `pins[<target>][<staged id>]`, refusing an absent target
    or rung BY NAME (an unpinned rung is unservable; an unpinned target is a pins file of another
    lane or an unstaged bank)."""
    if not isinstance(pins, dict) or target not in pins or not isinstance(pins[target], dict):
        raise StrengthRefusal(f"{where}: no pins for target {target!r} (targets pinned: "
                              f"{sorted(pins) if isinstance(pins, dict) else type(pins).__name__})")
    if staged_id not in pins[target]:
        raise StrengthRefusal(f"{where}: no pins entry for {target}.{staged_id} (rungs pinned: "
                              f"{sorted(pins[target])}) -- an unpinned rung has no bytes and no layers to serve")
    entry = pins[target][staged_id]
    if not isinstance(entry, dict):
        raise StrengthRefusal(f"{where}: pins entry {target}.{staged_id} is a {type(entry).__name__}, not a mapping")
    return entry


def raw_pin_layers(pin: dict, where: str) -> dict:
    """The layer binding read off ONE pins entry: the four keys, each present; the three layers
    integers; the sha a 64-hex string; and the two relations this kind's fit and blob imply --
    `served_layer == fit_layer - 1` (the fit record's own rule: the lens applies at a block's
    OUTPUT, arditi_fit) and `stored_layer == served_layer` (RAW_STORED_LAYER_IS_SERVED_LAYER: a
    vector_all blob stores the SERVED layer). A pins entry violating any of these is refused BY
    NAME rather than read for whichever key happens to be present."""
    missing = [k for k in RAW_PIN_LAYER_KEYS if k not in pin]
    if missing:
        raise StrengthRefusal(
            f"{where}: pins entry lacks {missing} (required: {list(RAW_PIN_LAYER_KEYS)}) -- the pins bind each "
            "rung's ABSOLUTE layers from the fit record (arditi_fit.json selected.layer / served_layer / "
            "direction_sha256); a pins file without them cannot certify where an object's edit lands, and a "
            "coherent object baked a block off passes every relational check")
    layers = {}
    for k in ("fit_layer", "served_layer", "stored_layer"):
        v = pin[k]
        if isinstance(v, bool) or not isinstance(v, int):
            raise StrengthRefusal(f"{where}: pins {k}={v!r} is not an integer")
        layers[k] = int(v)
    sha = pin["direction_sha256"]
    if not (isinstance(sha, str) and len(sha) == 64 and all(ch in "0123456789abcdef" for ch in sha)):
        raise StrengthRefusal(f"{where}: pins direction_sha256={sha!r} is not a 64-hex sha256")
    if layers["served_layer"] != layers["fit_layer"] - 1:
        raise StrengthRefusal(
            f"{where}: pins served_layer {layers['served_layer']} != fit_layer {layers['fit_layer']} - 1 -- the fit "
            "record serves one block BELOW the fitted block (arditi_fit selected.served_layer = layer - 1)")
    if layers["stored_layer"] != layers["served_layer"]:
        raise StrengthRefusal(
            f"{where}: pins stored_layer {layers['stored_layer']} != served_layer {layers['served_layer']} -- a "
            f"{RAW_SERVED_KIND} blob stores the SERVED layer (the static_r relation stored == fit layer is the "
            "OTHER unit's)")
    return {**layers, "direction_sha256": sha}


def raw_provenance_layer_problems(prov: dict, binding: dict, where: str) -> list:
    """Hold ONE staged object's `object.pt.provenance.json` to the pins' binding, every problem
    naming BOTH numbers: `fit.selected.layer == fit_layer`, `fit.selected.served_layer ==
    served_layer`, `stored_layer == stored_layer`, `layers == [served_layer]`, and
    `fit.selected.direction_sha256 == direction_sha256`. An absent field is a problem, never a
    pass: a provenance that does not say where its fit was cannot be certified."""
    problems = []
    sel = (prov.get("fit") or {}).get("selected") if isinstance(prov.get("fit"), dict) else None
    sel = sel if isinstance(sel, dict) else {}
    for prov_key, pin_key, what in (("layer", "fit_layer", "the FIT layer"),
                                    ("served_layer", "served_layer", "the SERVED layer")):
        got = sel.get(prov_key)
        if got is None:
            problems.append(f"{where}: fit.selected has no {prov_key!r} -- the pins bind {what} to {binding[pin_key]} "
                            "and a provenance that does not state it cannot be certified")
        elif int(got) != binding[pin_key]:
            problems.append(f"{where}: fit.selected.{prov_key} {int(got)} != the pins' {pin_key} {binding[pin_key]} -- "
                            f"{what} of this object is not where the fit record put it (a coherent object baked a "
                            "block off passes every relational check; only the pins' absolute layers catch it)")
    stored = prov.get("stored_layer")
    if stored is None:
        problems.append(f"{where}: no stored_layer -- the pins bind it to {binding['stored_layer']}")
    elif int(stored) != binding["stored_layer"]:
        problems.append(f"{where}: stored_layer {int(stored)} != the pins' stored_layer {binding['stored_layer']}")
    layers = prov.get("layers")
    if not isinstance(layers, list) or sorted(int(x) for x in layers) != [binding["served_layer"]]:
        problems.append(f"{where}: layers {layers!r} != [{binding['served_layer']}] (the pins' served_layer)")
    got_sha = sel.get("direction_sha256")
    if got_sha is None:
        problems.append(f"{where}: fit.selected has no direction_sha256 -- the pins bind the fitted direction's values "
                        f"to {binding['direction_sha256'][:12]}… and an unbound direction cannot be certified")
    elif str(got_sha) != binding["direction_sha256"]:
        problems.append(f"{where}: fit.selected.direction_sha256 {str(got_sha)[:12]}… != the pins' "
                        f"{binding['direction_sha256'][:12]}… -- this object was baked from a different fitted direction")
    return problems


#: THE APPROVED FIT IS DECLARED IN THE ARMS TABLE (arditi-qwen-all-positions Phase A review, round 3,
#: item 1). The pins above are the authority at every serve-time gate -- and a pins file is Phase B's
#: OUTPUT, written from whichever fit dir Phase B was pointed at (~/arditi-stage/fits/ holds several
#: models' fits). Nothing bound the PINS to the fit that was approved: coherent 12/11/11 pins plus a
#: coherent 12/11/11 object passed every gate with `problems: []`. So the CHOICE of fit is data in
#: `configs/sdfbase_full22_arms.yaml` under `arms.<arm>.fits.<target>` (RAW_FIT_KEYS: the fit record's
#: sha, its direction.pt's file sha, the direction's VALUES hash the producer binds to, and the layers the
#: record selected), exactly as `object_sha256` on an operating point declares the choice of bytes. Every
#: pins entry is held to that block BEFORE the pins certify any object (raw_pin_binding, the ONE read every
#: consumer makes); a target with no `fits` entry refuses by name -- a pins file cannot be the authority
#: for a fit the table never approved. The layers still COME FROM the fit record; the table names it.
ARMS_TABLE_REL = "configs/sdfbase_full22_arms.yaml"
RAW_FIT_KEYS: Tuple[str, ...] = ("fit_record_sha256", "direction_pt_sha256", "direction_sha256",
                                 "fit_layer", "served_layer", "stored_layer", "fit_pos", "source")
#: the pins keys the approved-fit block binds (the layer binding, RAW_PIN_LAYER_KEYS, key for key)
RAW_FIT_BOUND_PIN_KEYS: Tuple[str, ...] = RAW_PIN_LAYER_KEYS


def _is_sha256_hex(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(ch in "0123456789abcdef" for ch in v)


def load_arms_table(path) -> dict:
    """The arms table (configs/sdfbase_full22_arms.yaml) as a dict; an absent file refuses BY NAME (a
    checkout without the table approves no fit and pins nothing)."""
    import yaml
    p = Path(path)
    if not p.is_file():
        raise StrengthRefusal(f"no arms table at {p} -- the approved fit (arms.{RAW_CELL_ID}.fits.<target>) lives there, "
                              "and without it no pins entry can be validated")
    table = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(table, dict) or not isinstance(table.get("arms"), dict):
        raise StrengthRefusal(f"{p} is not an arms table (no `arms` mapping)")
    return table


def raw_fit_block(arms: dict, target: str, where: str, *, arm: str = RAW_CELL_ID) -> dict:
    """The APPROVED FIT for `target` -- `arms.<arm>.fits.<target>` of the loaded arms table -- with every
    RAW_FIT_KEYS key present, the three layers integers, the three shas 64-hex, and the block's own
    relations (served == fit - 1, the fit record's rule; stored == served, this kind's). A target the table
    approves no fit for is refused BY NAME: the pins cannot stand in for a choice the table never made."""
    arm_block = (arms.get("arms") or {}).get(arm) if isinstance(arms, dict) else None
    if not isinstance(arm_block, dict):
        raise StrengthRefusal(f"{where}: the arms table has no `arms.{arm}` block")
    fits = arm_block.get("fits")
    if not isinstance(fits, dict) or target not in fits or not isinstance(fits[target], dict):
        raise StrengthRefusal(
            f"{where}: the arms table approves no {arm} fit for {target!r} (arms.{arm}.fits has "
            f"{sorted(fits) if isinstance(fits, dict) else 'no entries'}) -- a pins file cannot be the authority for a fit "
            "the table never approved; declare the approved fit record there (fit_record_sha256 / direction_pt_sha256 / "
            "direction_sha256 / fit_layer / served_layer / stored_layer / fit_pos / source) before any rung of this target "
            "is served")
    fit = fits[target]
    missing = [k for k in RAW_FIT_KEYS if k not in fit]
    if missing:
        raise StrengthRefusal(f"{where}: arms.{arm}.fits.{target} lacks {missing} (required: {list(RAW_FIT_KEYS)})")
    for k in ("fit_layer", "served_layer", "stored_layer", "fit_pos"):
        if isinstance(fit[k], bool) or not isinstance(fit[k], int):
            raise StrengthRefusal(f"{where}: arms.{arm}.fits.{target}.{k}={fit[k]!r} is not an integer")
    for k in ("fit_record_sha256", "direction_pt_sha256", "direction_sha256"):
        if not _is_sha256_hex(fit[k]):
            raise StrengthRefusal(f"{where}: arms.{arm}.fits.{target}.{k}={fit[k]!r} is not a 64-hex sha256")
    if fit["served_layer"] != fit["fit_layer"] - 1:
        raise StrengthRefusal(f"{where}: arms.{arm}.fits.{target} served_layer {fit['served_layer']} != fit_layer "
                              f"{fit['fit_layer']} - 1 -- the fit record serves one block BELOW the fitted block")
    if fit["stored_layer"] != fit["served_layer"]:
        raise StrengthRefusal(f"{where}: arms.{arm}.fits.{target} stored_layer {fit['stored_layer']} != served_layer "
                              f"{fit['served_layer']} -- a {RAW_SERVED_KIND} blob stores the SERVED layer")
    return dict(fit)


#: HOW THE OPERATING POINT WAS PICKED, as a WORD on the row.
#: The AlphaSteer arm already carries `selection_outcome` (`chosen` / `a4_fallback`); the raw unit needs its
#: own vocabulary because this arm's point was NOT the rule's output. The words, and what each asserts:
#:   * `chosen` -- the budget rule's own answer: the highest dose with NO dose at or below it over the
#:     item budget, the point admissible under the rule as written.
#:   * `user_override_over_budget_2026-09-16` -- A DOSE THE RULE REJECTS WAS SELECTED: the selected rung
#:     measures OVER the budget and the mechanical rule returns a LOWER one. A row carrying
#:     this word must also carry the `selection` block below, so both numbers travel with it and no reader
#:     can mistake the point for a clean pick.
#: Never a bare `ruled_<date>`: that word is the AlphaSteer ladder's and the prefill-only `arditi` entry's
#: (g8k_cells.ARDITI_SELECTION_OUTCOMES), and re-using it here would say a ruling moved a point between two
#: rungs the rule ranked, which is not what happened.
RAW_SELECTION_OUTCOME_CHOSEN = "chosen"
RAW_SELECTION_OUTCOME_USER_OVERRIDE = "user_override_over_budget_2026-09-16"
RAW_SELECTION_OUTCOMES: Tuple[str, ...] = (RAW_SELECTION_OUTCOME_CHOSEN, RAW_SELECTION_OUTCOME_USER_OVERRIDE)
#: the words that mean "the rule did not pick this" -- an OVERRIDE, which must carry both readings
RAW_SELECTION_OVERRIDE_OUTCOMES: Tuple[str, ...] = (RAW_SELECTION_OUTCOME_USER_OVERRIDE,)
#: the `selection` block an operating point carries: the rule as written, the SELECTED point's measured
#: reading, and the reading of the point the RULE returns. Both readings are REQUIRED on every raw row,
#: override or not -- on a `chosen` row the two agree and saying so costs a line; on an override row the
#: pair IS the finding, and a field that can hold only one of them would drop the one that embarrasses it.
RAW_SELECTION_KEYS: Tuple[str, ...] = ("rule", "statistic", "cap_read", "selected", "rule_answer", "why", "record")
#: each of `selected` / `rule_answer`: which rung, its dose, and its measured delta in BOTH units
RAW_SELECTION_POINT_KEYS: Tuple[str, ...] = ("view", "c", "delta_items_gate_vs_none", "delta_pp_gate_vs_none")


def raw_selection_problems(op: dict, where: str, *, budget_items: float) -> list:
    """Every way a raw-coefficient operating point's SELECTION declaration fails to stand on its own.

    The point of this function is the OVERRIDE case: c = 0.3 at 25.54 items was selected against a
    25.0-item budget, and the rule as written returns c = 0.2 at 12.55. Nothing may record
    that as a clean pick, and nothing may record it carrying only one of the two numbers -- so:
      * `selection_outcome` is one of RAW_SELECTION_OUTCOMES, and the `selection` block is present with
        every RAW_SELECTION_KEYS key and both points fully stated (RAW_SELECTION_POINT_KEYS).
      * `selected` names the row's OWN view and dose -- a selection block describing another rung is a
        record of a different decision.
      * the arithmetic and the word AGREE: `chosen` requires the selected delta within budget AND the
        rule's answer to be the selected point itself; an override requires the selected delta OVER
        budget AND a rule answer that differs. A `chosen` on an over-budget point is the single failure
        this vocabulary exists to make impossible.
    The MEASUREMENTS are not re-derived here (that is the selection record's job, from the stores); this
    holds the declaration to itself and to the row it sits on."""
    problems = []
    outcome = op.get("selection_outcome")
    if outcome not in RAW_SELECTION_OUTCOMES:
        problems.append(f"{where}.selection_outcome: {outcome!r} is not one of {list(RAW_SELECTION_OUTCOMES)} -- a raw "
                        "operating point states HOW it was picked, and an unknown word asserts nothing")
    sel = op.get("selection")
    if not isinstance(sel, dict):
        return problems + [f"{where}.selection: missing -- a raw operating point carries the rule it was picked "
                           "under, the SELECTED point's reading and the reading of the point the RULE returns "
                           f"(keys {list(RAW_SELECTION_KEYS)})"]
    missing = [k for k in RAW_SELECTION_KEYS if k not in sel]
    if missing:
        problems.append(f"{where}.selection lacks {missing} (required: {list(RAW_SELECTION_KEYS)})")
    pts = {}
    for role in ("selected", "rule_answer"):
        blk = sel.get(role)
        if not isinstance(blk, dict):
            problems.append(f"{where}.selection.{role}: missing -- BOTH readings travel, always")
            continue
        gone = [k for k in RAW_SELECTION_POINT_KEYS if k not in blk]
        if gone:
            problems.append(f"{where}.selection.{role} lacks {gone} (required: {list(RAW_SELECTION_POINT_KEYS)})")
            continue
        pts[role] = blk
    if len(pts) != 2:
        return problems
    got, rule = pts["selected"], pts["rule_answer"]
    for k in ("view", "c"):
        if op.get(k if k != "view" else "bank_view") != got[k]:
            problems.append(f"{where}.selection.selected.{k}={got[k]!r} is not the row's own "
                            f"{'bank_view' if k == 'view' else k}={op.get(k if k != 'view' else 'bank_view')!r} -- "
                            "this block describes a different rung than the row serves")
    over = float(got["delta_items_gate_vs_none"]) > float(budget_items)
    same = got["view"] == rule["view"]
    if outcome == RAW_SELECTION_OUTCOME_CHOSEN and (over or not same):
        problems.append(
            f"{where}.selection_outcome={outcome!r} but the selected point measures "
            f"{got['delta_items_gate_vs_none']} items against the {budget_items} budget and the rule returns "
            f"{rule['view']} -- {'a point over budget' if over else 'a point the rule does not return'} is NOT a "
            f"clean pick; an override declares itself ({RAW_SELECTION_OUTCOME_USER_OVERRIDE})")
    if outcome in RAW_SELECTION_OVERRIDE_OUTCOMES:
        if not over:
            problems.append(f"{where}.selection_outcome={outcome!r} but the selected point measures "
                            f"{got['delta_items_gate_vs_none']} items, WITHIN the {budget_items} budget -- an override "
                            "records a point the rule rejects; a point the rule admits is `chosen`")
        if same:
            problems.append(f"{where}.selection.rule_answer names the selected rung {rule['view']!r} itself -- an "
                            "override supersedes a DIFFERENT point, and both must travel together")
    return problems


def raw_pin_fit_problems(binding: dict, fit: dict, where: str, target: str, *, arm: str = RAW_CELL_ID) -> list:
    """Hold ONE pins entry's layer binding (raw_pin_layers) to the table's approved fit, key for key:
    fit_layer, served_layer, stored_layer, direction_sha256. Every problem names BOTH values and says
    which is the authority -- the arms table -- because pins that are merely self-consistent (12/11/11
    is as coherent as 13/12/12) are exactly what this catches."""
    problems = []
    for k in RAW_FIT_BOUND_PIN_KEYS:
        got, want = binding[k], fit[k]
        if got != want:
            show = (lambda v: f"{str(v)[:12]}…") if k == "direction_sha256" else (lambda v: str(v))
            problems.append(
                f"{where}: pins {k} {show(got)} != the approved fit's {k} {show(want)} (arms table arms.{arm}.fits.{target}, "
                f"the AUTHORITY for which fit is approved) -- these pins were written from a fit the table did not approve "
                "(another model's fit dir, or a stale record); a pins file cannot certify a fit the table never approved")
    return problems


def raw_pin_binding(pins: dict, arms: dict, target: str, staged_id: str, where: str) -> Tuple[dict, dict]:
    """THE ONE VALIDATED READ of a rung's pins: `(pins entry, layer binding)` for `<target>.<staged id>`, or a
    refusal BY NAME. In order: the entry exists (raw_pin_entry); it carries the four keys and its own relations
    hold (raw_pin_layers); the arms table approves a fit for this target (raw_fit_block); and the entry EQUALS
    that fit key for key (raw_pin_fit_problems). Only then is the binding returned -- so every consumer
    (arm_cell_check.verify, ar_resolve_config, ar_bank_view, the grader) holds an object to pins that were
    themselves held to the approved fit first. The binding carries the table's block as `approved_fit`."""
    pin = raw_pin_entry(pins, target, staged_id, where)
    here = f"{where} [{target}.{staged_id}]"
    binding = raw_pin_layers(pin, here)
    fit = raw_fit_block(arms, target, here)
    problems = raw_pin_fit_problems(binding, fit, here, target)
    if problems:
        raise StrengthRefusal("PINS vs ARMS TABLE REFUSED:\n  " + "\n  ".join(problems))
    return pin, {**binding, "approved_fit": fit}


def raw_pin_object_sha256(pin: dict, where: str) -> str:
    """The pins entry's `object_sha256` -- the staged object's byte identity -- a 64-hex string, or a refusal
    by name (an entry without it pins nothing)."""
    sha = pin.get("object_sha256")
    if not _is_sha256_hex(sha):
        raise StrengthRefusal(f"{where}: pins object_sha256={sha!r} is not a 64-hex sha256 -- the entry pins no bytes")
    return sha


def alpha_equiv(c, raw_direction_norm: float, h_ref: float) -> float:
    """The residual-norm fraction ARITHMETICALLY equivalent to raw coefficient
    ``c``: ``|c| * ||r_raw|| / ||h||_ref``. On qwen35_4b (||r|| 3.2904739,
    ||h||_ref 9.364958763) that is ``c * 0.35136``.

    A LABELLED CONVERSION, NEVER A REALISED-DOSE CLAIM.
    ``||h||_ref`` was measured on PROMPT tokens under PREFILL-ONLY serving; this
    unit writes at decode positions too, where the residual norm differs. The
    number exists so a reader can place a rung beside the prefill-only ladder,
    and every artifact that carries it must carry that sentence with it."""
    a = abs(validate_strength(c))
    for name, x in (("raw_direction_norm", raw_direction_norm), ("h_ref", h_ref)):
        if not (isinstance(x, (int, float)) and not isinstance(x, bool)
                and math.isfinite(float(x)) and float(x) > 0):
            raise StrengthRefusal(f"alpha_equiv: {name} must be positive and finite, got {x!r}")
    return a * float(raw_direction_norm) / float(h_ref)


#: the sentence :func:`alpha_equiv`'s output must be labelled with, verbatim
ALPHA_EQUIV_IS_ARITHMETIC = (
    "alpha_equiv is an ARITHMETIC conversion c * ||r_raw|| / ||h||_ref, where ||h||_ref was "
    "measured on PROMPT tokens under PREFILL-ONLY serving. This unit writes at decode positions "
    "too, where ||h|| differs, so the figure is for comparability with the prefill-only ladder and "
    "is NEVER a claim about the dose realised at decode. The measured realised dose is recorded "
    "beside it, per rung.")


if __name__ == "__main__":
    import json
    table = {"dose_unit": DOSE_UNIT, "dose_unit_description": DOSE_UNIT_DESCRIPTION,
             "alphas": list(ALPHAS), "targets": {}}
    for t in TARGETS:
        table["targets"][t] = {
            view_name(a): {"alpha": a, "staged_id": staged_id_of_strength(a),
                           "config": serve_config_rel(t, a),
                           "manifest": manifest_rel(t, a),
                           "is_sign_control": is_anti_refusal(a)}
            for a in ALPHAS}
    print(json.dumps(table, indent=1))
