"""``paper_rows`` — the flat per-row export of the two capability analyses.

THE DEFECT THIS CLOSES. Each capability report files, per cell, a ``point`` whose denominator is
``n_scored``: the CONDITIONAL estimand, accuracy over the items whose draw TERMINATED. A draw that
hits the cap inside an unterminated think block is UNSCORED at the store level — it leaves the
denominator, it is not graded wrong. A reader that takes ``point`` with ``n = n_scored`` as "the
accuracy" therefore reads the model over the problems it finished, and a configuration that makes
generations long looks BETTER the more of them it cuts off. The paper's capability panel asks the
other question — of the problems posed, how many did this configuration solve — which is the
SCORE-AS-IS estimand: ``n_correct / n_paired`` with every unscored draw counted wrong.

TWO PRIMARIES, BOTH STATED IN THE OPEN. Conditional accuracy stays the REPORT primary; the
score-as-is branch is filed beside it on every record, with the censoring differential.
Score-as-is is the FIGURE primary the paper draws. This export carries both, with
the four count fields (``n_paired`` / ``n_scored`` / ``n_unscored`` / ``n_at_cap``) beside them so
a reader can see the censoring, and never changes an estimator: every number is READ from the
existing report (or, where a count is not in the report, from the store's ``gen_stats.json``,
named in ``source``).

SHAPE. ``paper_rows.json`` = ``{"schema", "fields", "n_rows", "not_exported", "rows"}`` where every
row has EXACTLY the keys of :data:`FIELDS`, in that order; ``paper_rows.csv`` is its twin (one line
per row, the same columns; ``unscored_reason`` JSON-encoded). Both are written beside ``report.json``
by the analysis's production path and by its ``--paper-rows-only REPORT`` mode.

VOCABULARIES (pinned by the tests):
``rung_state``: ``complete`` (a rung was due and its store is read) · ``owed`` (a rung is due, no
store yet) · ``not_applicable`` (no rung due: at-cap share under the trigger, or nothing eligible, or
an instrument with one cap) · ``refused`` (the cell or its rung was refused by name) · ``never_run``
(a rung was due and no mechanism ran it). ``object_generation``: ``batch8`` | ``batch64`` (the
stamp READ from the record) · ``none`` (no object: bare / prompted) · ``n/a`` (a fitted object whose
record carries no generation stamp — every record written before the stamp existed). ``selection_outcome``:
``chosen`` | ``a4_fallback`` | the steer-admission record's dated words ``ruled_2026-09-14`` | ``chosen_confirmed_2026-09-14`` |
``ruled_2026-09-14_second`` | ``candidate_2026-09-14`` | ``user_override_over_budget_2026-09-16`` (READ from the record,
verbatim, never relabelled; the last is the all-positions Arditi arm's DECLARED override of its selection rule — see
``selection`` below) · ``None`` (no selection on this arm).
``unscored_reason``: counts by reason — "the budget ran out" vs "the grader could not score it" is
the difference between a censoring correction and a missing measurement, and a reader cannot
recover it from a count; ``n_unscored == sum(unscored_reason.values())`` on every row (a remainder
lands in ``other`` and is named in ``source``).

``paper_arm`` (schema ``paper_rows/2``): the paper's CANONICAL arm key, emitted by this producer from the DECLARED
table :data:`PAPER_ARM_KEY_BY_CELL` (matrix cell slug -> paper key) and never inferred by a reader from the slug's
spelling. Only the seven slugs the paper draws map to a key; every other slug (``prompted__safety_refusal``, the PSR /
si12 / distilled arms, any GLM-only cell) carries ``null`` — an absent key is a statement, not a guess. The bare
``alphasteer`` slug resolves to ``alphasteer_rebake`` explicitly: ``alphasteer_as_fit`` is a DIFFERENT arm (the as-fit
object) and must never be written for the rebake cells this export carries.

THE THREE PROVENANCE FIELDS (schema ``paper_rows/3``; appended after ``source``, null on every row they do not apply to):
``selection`` — the arm's selection READING as the admission record declares it, both readings and the reading's identity:
``{outcome, rule_returns, rule_reading_pp, override_reading_pp, budget_pp, ladder_cap, single_trial, dose: {value, unit},
alpha_equiv: {value, note}, replicate_note}`` (:data:`SELECTION_FIELDS`). ``ladder_cap`` is part of the reading's IDENTITY
(at another cap the rule returns another point), so a block without it is not a reading of anything and refuses; a block
whose ``outcome`` disagrees with the row's ``selection_outcome`` refuses; a block claiming the override word while
``override_reading_pp <= budget_pp`` or while ``rule_returns`` names the selected dose refuses (nothing may imply the rule
chose the override point). The dose is a ``raw_coefficient`` on the all-positions arm — NEVER an alpha
(``residual_norm_fraction`` is another arm's unit on another intervention; ``alpha_equiv`` is arithmetic convenience,
recorded as such, and never shares an axis with the alpha doses). Required on every row whose ``selection_outcome`` is the
override word. The ladder behind it is SINGLE-TRIAL: every margin is a ratio against a crude floor, never a significance
claim, and ``single_trial`` says so on the block.
``serving_commit`` — the commit of the SERVING STACK the leg actually served under, READ from the store's record, never typed
into a config and never inferred. Required on a row WITH A READING of an arm in :data:`CELLS_REQUIRING_SERVING_COMMIT`; a
row of that arm with no reading (nothing was served) carries null. As of 2026-09-17 NO field of the store record names it:
``producer.tree_sha`` is the DRIVER's (client) tree, the standup's ``serving_config.json`` records config / manifest digests
and a repo PATH, and the native ``serving_attestation.json`` records no tree — so the readers file null and this export
REFUSES the arm's rows by name until a serving-side writer stamps the commit. Refusing is the point: a driver tree written
under this name would be a different quantity wearing the field's label.
``fit_provenance`` — what the served direction was FITTED on, verbatim from the admission record's block:
:data:`FIT_PROVENANCE_FIELDS`, every key present and non-null (the three ``*_sha256`` keys 64-hex; the two count / gate
sub-blocks non-empty mappings), each satisfied under its canonical name OR a DECLARED producer spelling of the same
quantity (:data:`FIT_PROVENANCE_ALIASES`: the record spells the fit items' commit ``item_source_repo_sha``) — a naming
difference is not a missing fact. Required on every row of an arm in :data:`CELLS_REQUIRING_FIT_PROVENANCE`; missing or
incomplete REFUSES the row — never exported with a null, never a guessed value, and never another field standing in for
one the record does not carry. As of 2026-09-17 the admission record names no ``capture_positions`` (the positions the fit
ACTIVATIONS were captured at — stated only inside the ``rendering`` prose), so this export refuses the arm's rows by name
until the record carries it; the served position mode on the row is a different quantity and is not a substitute.
"""
from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "paper_rows/4"          # /4: + ``n_refusal_shaped`` last (/3: the three provenance fields after ``source``; /2: ``paper_arm``)
#: the documents :func:`read_paper_rows` reads: /4 as written; a /3 document (the same rows minus ``n_refusal_shaped``) and a /2 one
#: (minus the three provenance fields too) are read with the absent fields filled null — a /2 or /3 row carries no refusal-shaped
#: count because its producer never counted one (absence, not zero) — and verified under the /4 contract
SCHEMA_2 = "paper_rows/2"
SCHEMA_3 = "paper_rows/3"
READABLE_SCHEMAS = (SCHEMA_2, SCHEMA_3, SCHEMA)
JSON_NAME, CSV_NAME = "paper_rows.json", "paper_rows.csv"

#: the row's keys, in order — EXACTLY these (the base field list + ``ci_method`` + ``unscored_reason`` +
#: the paper's ``paper_arm`` key right after ``arm``, schema /2; the three provenance fields, schema /3; ``n_refusal_shaped``, schema /4)
FIELDS: tuple[str, ...] = (
    "instrument", "model_key", "view", "cell", "arm", "paper_arm", "cap", "leg",
    "n_paired", "n_scored", "n_unscored", "n_at_cap", "at_cap_rate", "n_unterminated_not_at_cap", "n_correct",
    "point_conditional", "ci_conditional_low", "ci_conditional_high",
    "point_score_as_is", "ci_score_as_is_low", "ci_score_as_is_high", "ci_method",
    "rung_state", "object_generation", "selection_outcome", "unscored_reason", "censoring_note", "source",
    "selection", "serving_commit", "fit_provenance",
    "n_refusal_shaped", "n_refusal_shaped_of",
)
#: schema /3's three provenance fields (appended after ``source``); ``make_row`` fills them null unless given, and the verifier
#: REQUIRES them only where they apply (:func:`row_problems`)
PROVENANCE_FIELDS: tuple[str, ...] = ("selection", "serving_commit", "fit_provenance")
#: schema /4: the REFUSAL-SHAPED count WITH ITS DENOMINATOR — a count with its n is auditable by a reader in a way
#: a share is not. ``n_refusal_shaped``: the scored draws of the reading whose emitted CONTENT channel matched
#: the campaign's fixed, judge-free pattern list (``sisteer.capeval.tinycap.refusal_shaped``; the count is that module's ``readout``);
#: ``n_refusal_shaped_of``: the responses the classifier actually EXAMINED for the row — the scored draws whose content channel is on
#: record (rows.jsonl carries the scored draws; a draw with no content channel is neither examined nor counted) — never a share. Filed
#: BESIDE ``n_correct`` / ``n_unscored`` so a capability cost can be named as over-refusal, wrong arithmetic or degeneration; they change no
#: denominator and no grader value (``n_refusal_shaped <= n_refusal_shaped_of <= n_scored``, the way ``n_correct <= n_scored``). The pair
#: is null TOGETHER: on a row with no reading, and on a row read from a /2 or /3 document or a pre-class record (its producer never counted
#: the class — absence, never zero); a count without its denominator, or a denominator without its count, refuses.
REFUSAL_SHAPED_FIELD = "n_refusal_shaped"
REFUSAL_SHAPED_OF_FIELD = "n_refusal_shaped_of"
REFUSAL_SHAPED_FIELDS: tuple[str, ...] = (REFUSAL_SHAPED_FIELD, REFUSAL_SHAPED_OF_FIELD)
#: the fields that are null together on a row with no reading (a refused / missing cell)
COUNT_FIELDS: tuple[str, ...] = ("n_paired", "n_scored", "n_unscored", "n_at_cap", "at_cap_rate", "n_unterminated_not_at_cap", "n_correct",
                                 "point_conditional", "ci_conditional_low", "ci_conditional_high",
                                 "point_score_as_is", "ci_score_as_is_low", "ci_score_as_is_high", "ci_method", "unscored_reason",
                                 *REFUSAL_SHAPED_FIELDS)

INSTRUMENT_GSM8K, INSTRUMENT_MATH500 = "gsm8k_paired200", "math500"
INSTRUMENTS = (INSTRUMENT_GSM8K, INSTRUMENT_MATH500)

#: the literal sentence every row carries
CENSORING_NOTE = ("at-cap draws that end inside an unterminated think block are UNSCORED and excluded from point_conditional; "
                  "point_score_as_is counts every such draw as wrong")

RUNG_COMPLETE, RUNG_OWED, RUNG_NOT_APPLICABLE, RUNG_REFUSED, RUNG_NEVER_RUN = "complete", "owed", "not_applicable", "refused", "never_run"
RUNG_STATES = (RUNG_COMPLETE, RUNG_OWED, RUNG_NOT_APPLICABLE, RUNG_REFUSED, RUNG_NEVER_RUN)

GEN_BATCH8, GEN_BATCH64, GEN_NONE, GEN_NA = "batch8", "batch64", "none", "n/a"
OBJECT_GENERATIONS = (GEN_BATCH8, GEN_BATCH64, GEN_NONE, GEN_NA)
#: the stamps a record may carry for a FITTED object (anything else is a vocabulary the export does not know)
OBJECT_GENERATION_STAMPS = (GEN_BATCH8, GEN_BATCH64)

SEL_CHOSEN, SEL_A4_FALLBACK = "chosen", "a4_fallback"
#: the steer-admission record's DATED rulings on an AlphaSteer row (the launcher's vocabulary, `g8k_cells.SELECTION_OUTCOMES`,
#: mirrored here word for word — the export module lives under src/ and imports no launcher; the test holds the two equal):
#: a ruled operating point, a confirmed choice, a SECOND ruling of the same day (gemma lam3 -> lam2, spelled `_second` by the
#: oracle), and an interim CANDIDATE (not a ruling; row 47's store was produced under it). Read VERBATIM from the record;
#: never relabelled: a candidate is exported as one.
SEL_RULED, SEL_CHOSEN_CONFIRMED = "ruled_2026-09-14", "chosen_confirmed_2026-09-14"
SEL_RULED_SECOND, SEL_CANDIDATE = "ruled_2026-09-14_second", "candidate_2026-09-14"
#: the all-positions Arditi arm's ONE outcome word (the launcher's `g8k_cells.SELECTION_OUTCOME_ALLPOS_USER_OVERRIDE`, mirrored word
#: for word): the operating point c = 0.3 is a DECLARED OVERRIDE of the selection rule, which returns c = 0.2 at 5.04 pp
#: over-refusal delta where 0.3 measures 10.26 pp against a 10 pp budget. Read VERBATIM from the record, never relabelled; a row
#: carrying it must carry the ``selection`` block that states both readings (:func:`selection_problems`). Nothing here may imply
#: the rule chose 0.3 — a null on that arm would read as the rule's own pick and is refused (:data:`ARDITI_ALLPOS_SELECTION_OUTCOMES`).
SEL_USER_OVERRIDE_OVER_BUDGET = "user_override_over_budget_2026-09-16"
SELECTION_OUTCOMES = (SEL_CHOSEN, SEL_A4_FALLBACK, SEL_RULED, SEL_CHOSEN_CONFIRMED, SEL_RULED_SECOND, SEL_CANDIDATE,
                      SEL_USER_OVERRIDE_OVER_BUDGET, None)
#: the outcome words a row of the all-positions Arditi cell may carry: exactly the override word (never null, never another arm's)
ARDITI_ALLPOS_SELECTION_OUTCOMES = (SEL_USER_OVERRIDE_OVER_BUDGET,)

#: the ``selection`` block's keys (the admission record's block, carried verbatim; extra keys ride along, these are REQUIRED):
#: both readings (``rule_reading_pp`` at the rule's return, ``override_reading_pp`` at the selected point) against ``budget_pp``,
#: the reading's identity (``ladder_cap``), the single-trial flag, the dose with its unit, the alpha arithmetic labelled as such
SELECTION_FIELDS: tuple[str, ...] = ("outcome", "rule_returns", "rule_reading_pp", "override_reading_pp", "budget_pp", "ladder_cap",
                                     "single_trial", "dose", "alpha_equiv", "replicate_note")
SELECTION_DOSE_FIELDS: tuple[str, ...] = ("value", "unit")
SELECTION_ALPHA_EQUIV_FIELDS: tuple[str, ...] = ("value", "note")
#: the dose units the block may spell: the all-positions arm's raw coefficient (c scales the direction; the dose is in the bytes)
#: and the alpha arms' residual-norm fraction — a DIFFERENT unit on a DIFFERENT intervention, refused on the all-positions arm
DOSE_UNIT_RAW_COEFFICIENT, DOSE_UNIT_ALPHA = "raw_coefficient", "residual_norm_fraction"
#: the cells whose ``selection.dose`` must be a raw coefficient (an alpha-shaped dose there puts two units on one axis)
CELLS_DOSED_IN_RAW_COEFFICIENT: tuple[str, ...] = ("arditi_allpos",)
#: the lane's coefficient VIEW spelling (``ar_c0p2`` = c 0.2, ``ar_c0p3`` = c 0.3): the one way ``rule_returns`` may name a dose
#: besides a bare number; anything else the export cannot read refuses by name (never a guess at which dose the rule returned)
_COEFFICIENT_VIEW_RE = re.compile(r"^[a-z]+_c(\d+)p(\d+)$")

#: ``fit_provenance``: every key the block must carry, non-null — what the direction was fitted on (model + revision), how the
#: activations were rendered and at which positions they were captured, the fit items (source + its sha) and the REALISED
#: counts, the direction's two digests, the fit layer / position / norm, the SERVED layer (a `vector_all` blob is served as
#: stored: served = fit - 1 on this arm), the fit-quality gate and the fit record's digest
FIT_PROVENANCE_FIELDS: tuple[str, ...] = (
    "fitted_on_model", "fitted_on_revision", "rendering", "capture_positions", "item_source", "item_source_sha", "realised_counts",
    "direction_sha256", "direction_pt_sha256", "fit_layer", "fit_position", "direction_norm", "served_layer", "fit_quality_gate",
    "fit_record_sha256")
#: the PRODUCER's own spelling of a canonical key, where the two names are the same quantity (the admission record's
#: `steer_admission_ruling.FIT_PROVENANCE_KEYS` spells the fit items' commit ``item_source_repo_sha``). A block satisfies a
#: canonical key by carrying EITHER name — a naming difference is not a missing fact, and refusing over one would reject the
#: authoritative record itself. Every entry here is a DECLARED equivalence between two names for one quantity, never a
#: substitution of a DIFFERENT quantity: nothing stands in for a fact the record does not carry (see ``capture_positions``,
#: which the record names nowhere and which therefore refuses).
FIT_PROVENANCE_ALIASES: dict[str, tuple[str, ...]] = {"item_source_sha": ("item_source_repo_sha",)}
FIT_PROVENANCE_SHA256_FIELDS: tuple[str, ...] = ("direction_sha256", "direction_pt_sha256", "fit_record_sha256")
#: the block's two GIT commits (40-hex, not content digests): the checkpoint the direction was fit on, and the tree the fit
#: items were pinned at — checked in their own shape so a content digest cannot pass as a commit or the other way round
FIT_PROVENANCE_GIT_SHA_FIELDS: tuple[str, ...] = ("fitted_on_revision", "item_source_sha")
FIT_PROVENANCE_MAPPING_FIELDS: tuple[str, ...] = ("realised_counts", "fit_quality_gate")
#: the cells whose rows REQUIRE a complete ``fit_provenance`` block (the export refuses the row otherwise)
CELLS_REQUIRING_FIT_PROVENANCE: tuple[str, ...] = ("arditi_allpos",)
#: the cells whose rows WITH A READING require ``serving_commit`` (a row with no reading served nothing and carries null)
CELLS_REQUIRING_SERVING_COMMIT: tuple[str, ...] = ("arditi_allpos",)
_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: where a row's CIs come from: the report's own percentile bootstrap, or the exact binomial where the report has none
CI_BOOTSTRAP, CI_EXACT = "percentile_bootstrap(report)", "exact_binomial_clopper_pearson"
CI_METHODS = (CI_BOOTSTRAP, CI_EXACT)

UNSCORED_UNTERMINATED, UNSCORED_HIT_CAP_MID_SOLUTION, UNSCORED_REFUSAL, UNSCORED_OTHER = (
    "unterminated_think_block", "hit_cap_mid_solution", "refusal", "other")
UNSCORED_REASONS = (UNSCORED_UNTERMINATED, UNSCORED_HIT_CAP_MID_SOLUTION, UNSCORED_REFUSAL, UNSCORED_OTHER)

#: the paper's arm name per cell id — DECLARED, never inferred from the id's spelling (the MATH-500 report names
#: its arms itself through ``comparisons[*].leg_keys``; the GSM8K report has no arm vocabulary, so it takes this table)
PAPER_ARM_BY_CELL: dict[str, str] = {
    "none": "bare",
    "prompted__safety_refusal": "prompted",
    "prompted__si_constitution": "prompted_constitution",
    "constv2ctl2-full__lora_r4_ce": "lora_ctl2",
    "constv2-mix-full__lora_r4_ce": "lora_mix",
    "constv2-mix-full__lora_r4_ce_minus_ctl": "object",
    "constv2-mix-full__psr_ce_all": "psr_mix",
    "constv2ctl2-full__psr_ce_all": "psr_ctl2",
    "constv2-mix-full__psr_ce_all_minus_ctl_distilled": "psr_object_distilled",
    "safety_refusal__psr_kl_all": "si12_psr",
    "safety_refusal__lora_r4": "si12_lora",
}

#: the paper's CANONICAL arm key per matrix cell slug — DECLARED by this producer, never inferred by a reader from the
#: slug's spelling (the paper's owner, 2026-09-14). Exactly the seven arms the paper draws from this export; every other
#: slug maps to ``None`` (see :func:`paper_arm_of`). HAZARD, named on purpose: the bare ``alphasteer`` slug is the REBAKE
#: object (``alphasteer_rebake``); ``alphasteer_as_fit`` is a different arm of another lane and is never a value here.
PAPER_ARM_KEY_BY_CELL: dict[str, str] = {
    "none": "none",
    "alphasteer": "alphasteer_rebake",
    "arditi": "arditi",
    "prompted__si_constitution": "si_constitution",
    "constv2-mix-full__lora_r4_ce": "const_lora",
    "constv2ctl2-full__lora_r4_ce": "ctl_lora",
    "constv2-mix-full__lora_r4_ce_minus_ctl": "diff_lora",
    # HAZARD, named on purpose: `arditi_allpos` is NOT `arditi`. They are two different interventions — `arditi` is the
    # PREFILL-ONLY refusal-direction arm dosed as an ALPHA (a residual-norm fraction on a `static_r` object); `arditi_allpos`
    # is the ALL-POSITIONS direction (prefill AND every decode step) dosed as a RAW COEFFICIENT on a `vector_all` object,
    # served on the native residual slot. The two must never collapse to one paper key, share a dose axis, or be pooled;
    # the `arditi` entry above stays where it is.
    "arditi_allpos": "arditi_allpos",
}
PAPER_ARM_KEYS: tuple[str, ...] = tuple(PAPER_ARM_KEY_BY_CELL.values())


def paper_arm_of(cell: Any) -> str | None:
    """The paper's canonical arm key for a matrix cell slug, from :data:`PAPER_ARM_KEY_BY_CELL`; ``None`` for every
    slug the paper does not draw from this export (a declared absence, never a guess from the spelling)."""
    if not isinstance(cell, str):
        return None
    return PAPER_ARM_KEY_BY_CELL.get(cell)


#: the tolerance of the round-trip identities: a filed point times its denominator must be an integer count
ROUND_TRIP_TOL = 1e-6


class PaperRowsError(ValueError):
    """The export cannot be written truthfully — a row failed its round-trip identity, a vocabulary or the schema."""


# --------------------------------------------------------------------------- #
# small readers                                                                #
# --------------------------------------------------------------------------- #


def ratio(k: int | None, n: int | None) -> float | None:
    """``k / n``, or ``None`` when there is no denominator (a leg with nothing scored, an absent count)."""
    if k is None or n is None or n == 0:
        return None
    return k / n


def exact_binomial_ci(k: int, n: int, level: float = 0.95) -> tuple[float | None, float | None]:
    """The Clopper-Pearson (exact binomial) ``level`` interval for ``k`` successes in ``n`` trials — used ONLY
    where the report files no bootstrap for a leg (a rung leg read over its redrawn subset). ``n == 0`` -> no
    interval. Imported lazily: the module must import without SciPy (the CSV / JSON writers need none)."""
    if n <= 0:
        return None, None
    if not 0 <= k <= n:
        raise PaperRowsError(f"exact_binomial_ci: k={k} outside 0..{n}")
    from scipy.stats import beta  # noqa: PLC0415 — lazy on purpose (see the docstring)

    alpha = 1.0 - level
    low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return low, high


def unscored_reason(n_unscored: int, *, unterminated_think_block: int, hit_cap_mid_solution: int = 0,
                    refusal: int = 0) -> tuple[dict[str, int], str | None]:
    """The ``unscored_reason`` map for a row and the note ``source`` must carry when a remainder lands in
    ``other``. The named reasons are READ counts (today the store's ``n_think_block_unterminated`` accounts for
    every unscored draw); ``other`` is whatever ``n_unscored`` they do not account for — never negative."""
    named = int(unterminated_think_block) + int(hit_cap_mid_solution) + int(refusal)
    other = int(n_unscored) - named
    if other < 0:
        raise PaperRowsError(f"unscored_reason: the named reasons ({named}) exceed n_unscored ({n_unscored})")
    out = {UNSCORED_UNTERMINATED: int(unterminated_think_block), UNSCORED_HIT_CAP_MID_SOLUTION: int(hit_cap_mid_solution),
           UNSCORED_REFUSAL: int(refusal), UNSCORED_OTHER: other}
    note = None if other == 0 else f"unscored_reason: {other} unscored draw(s) not accounted for by the store's named reasons -> other"
    return out, note


def source_text(report_path: Path | str | None, *, store_pin: str | None, analysis_code_sha: str | None,
                reads: Sequence[str]) -> str:
    """The ``source`` field: the report (path + its store pin + the analysis code's sha) and, in order, what was
    read for the row — every field name, every store file, every derivation and every note."""
    head = [f"report={report_path}", f"store_pin={store_pin}", f"analysis_code_sha={analysis_code_sha}"]
    return "; ".join(head) + "; read: " + " | ".join(str(r) for r in reads)


def make_row(**fields: Any) -> dict[str, Any]:
    """A row with EXACTLY :data:`FIELDS`, in order. ``censoring_note`` is filled by the schema and ``paper_arm`` by the
    declared table (:func:`paper_arm_of` of the row's ``cell``; a caller may pass it, the verifier holds it to the
    table); every other field must be given (``None`` is a value — the absent-reading rows pass it explicitly)."""
    fields.setdefault("censoring_note", CENSORING_NOTE)
    fields.setdefault("paper_arm", paper_arm_of(fields.get("cell")))
    for f in PROVENANCE_FIELDS:         # /3: null unless the producer read one; the verifier requires them only where they apply
        fields.setdefault(f, None)
    for f in REFUSAL_SHAPED_FIELDS:      # /4: null TOGETHER unless the producer COUNTED the class (the two math readers do; absence, never zero)
        fields.setdefault(f, None)
    missing = [f for f in FIELDS if f not in fields]
    extra = [f for f in fields if f not in FIELDS]
    if missing or extra:
        raise PaperRowsError(f"make_row: missing {missing}, unknown {extra}")
    return {f: fields[f] for f in FIELDS}


# --------------------------------------------------------------------------- #
# the verifier: every row must round-trip to its counts                        #
# --------------------------------------------------------------------------- #


def _is_int_multiple(point: float | None, n: int | None, k: int | None) -> bool:
    if point is None or n is None or k is None:
        return False
    return math.isclose(point * n, k, abs_tol=ROUND_TRIP_TOL) and math.isclose(round(point * n), k, abs_tol=ROUND_TRIP_TOL)


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def dose_of_rule_return(rule_returns: Any) -> float | None:
    """The dose ``selection.rule_returns`` names: a bare number, or the lane's coefficient VIEW spelling (``ar_c0p2`` -> 0.2,
    :data:`_COEFFICIENT_VIEW_RE`). ``None`` for anything else — the export cannot read which dose the rule returned and
    says so (:func:`selection_problems`) rather than guess."""
    if _is_number(rule_returns):
        return float(rule_returns)
    if isinstance(rule_returns, str):
        m = _COEFFICIENT_VIEW_RE.match(rule_returns.strip())
        if m:
            return float(f"{m.group(1)}.{m.group(2)}")
    return None


def selection_problems(selection: Any, *, tag: str, selection_outcome: Any, cell: Any) -> list[str]:
    """Every way a ``selection`` block fails its contract, by name (``[]`` for a sound block). The block carries BOTH readings and
    the reading's identity: every :data:`SELECTION_FIELDS` key present; ``outcome`` equal to the row's ``selection_outcome``
    (one word, in both places); ``ladder_cap`` a positive integer (the reading's IDENTITY — without it the block is not a
    reading of anything); ``single_trial`` exactly true (no margin here is a significance claim); the three ``*_pp`` numeric;
    ``dose`` = ``{value: number, unit}`` with a known unit — a RAW COEFFICIENT on a cell in :data:`CELLS_DOSED_IN_RAW_COEFFICIENT`
    (an alpha there puts two units on one axis), and never an ``alpha`` key inside the dose; ``alpha_equiv`` null or
    ``{value: number, note: text}`` (arithmetic convenience, labelled); ``replicate_note`` text. Under the OVERRIDE word:
    ``override_reading_pp > budget_pp`` (else the word is false — the point was within budget), ``rule_reading_pp <= budget_pp``
    (the rule's return is within budget by construction), and ``rule_returns`` must not name the selected dose (else the rule
    chose it and there was no override); a ``rule_returns`` the export cannot read (:func:`dose_of_rule_return`) refuses."""
    out: list[str] = []
    if not isinstance(selection, Mapping):
        return [f"{tag}: selection {selection!r} is not a mapping"]
    missing = [k for k in SELECTION_FIELDS if k not in selection]
    if missing:
        out.append(f"{tag}: selection lacks {missing} (the block carries {list(SELECTION_FIELDS)})")
    if "ladder_cap" in selection and not (isinstance(selection["ladder_cap"], int) and not isinstance(selection["ladder_cap"], bool)
                                          and selection["ladder_cap"] > 0):
        out.append(f"{tag}: selection.ladder_cap {selection['ladder_cap']!r} is not a positive integer — the cap is part of the "
                   "reading's IDENTITY (at another cap the rule returns another point); without it the block is not a reading of anything")
    if selection.get("outcome") != selection_outcome:
        out.append(f"{tag}: selection.outcome {selection.get('outcome')!r} != the row's selection_outcome {selection_outcome!r} — one word, in both places")
    if selection.get("outcome") not in SELECTION_OUTCOMES:
        out.append(f"{tag}: selection.outcome {selection.get('outcome')!r} not in {SELECTION_OUTCOMES}")
    if "single_trial" in selection and selection["single_trial"] is not True:
        out.append(f"{tag}: selection.single_trial {selection['single_trial']!r} is not true — the ladder was run ONCE; the block says so, "
                   "so no margin on it reads as a significance claim")
    for k in ("rule_reading_pp", "override_reading_pp", "budget_pp"):
        if k in selection and not _is_number(selection[k]):
            out.append(f"{tag}: selection.{k} {selection[k]!r} is not a number")
    dose = selection.get("dose")
    dose_value: float | None = None
    if not isinstance(dose, Mapping):
        out.append(f"{tag}: selection.dose {dose!r} is not a {{value, unit}} mapping")
    else:
        dmissing = [k for k in SELECTION_DOSE_FIELDS if k not in dose]
        if dmissing:
            out.append(f"{tag}: selection.dose lacks {dmissing}")
        if "alpha" in dose:
            out.append(f"{tag}: selection.dose carries an `alpha` key ({dose['alpha']!r}) — a dose is one value in one unit; an alpha here "
                       "puts two units on one axis")
        if not _is_number(dose.get("value")):
            out.append(f"{tag}: selection.dose.value {dose.get('value')!r} is not a number")
        else:
            dose_value = float(dose["value"])
        unit = dose.get("unit")
        if unit not in (DOSE_UNIT_RAW_COEFFICIENT, DOSE_UNIT_ALPHA):
            out.append(f"{tag}: selection.dose.unit {unit!r} not in {(DOSE_UNIT_RAW_COEFFICIENT, DOSE_UNIT_ALPHA)}")
        elif cell in CELLS_DOSED_IN_RAW_COEFFICIENT and unit != DOSE_UNIT_RAW_COEFFICIENT:
            out.append(f"{tag}: selection.dose.unit {unit!r} on {cell!r} — this arm's dose is a {DOSE_UNIT_RAW_COEFFICIENT!r} (c scales the "
                       f"direction), NEVER an alpha: {DOSE_UNIT_ALPHA!r} is another arm's unit on another intervention")
    alpha_equiv = selection.get("alpha_equiv")
    if alpha_equiv is not None:
        if not isinstance(alpha_equiv, Mapping) or [k for k in SELECTION_ALPHA_EQUIV_FIELDS if k not in alpha_equiv]:
            out.append(f"{tag}: selection.alpha_equiv {alpha_equiv!r} is not a {{value, note}} mapping (arithmetic convenience, labelled as such)")
        else:
            if not _is_number(alpha_equiv.get("value")):
                out.append(f"{tag}: selection.alpha_equiv.value {alpha_equiv.get('value')!r} is not a number")
            if not isinstance(alpha_equiv.get("note"), str) or not alpha_equiv["note"].strip():
                out.append(f"{tag}: selection.alpha_equiv.note is empty — the alpha arithmetic must say it is not the dose")
    if "replicate_note" in selection and (not isinstance(selection["replicate_note"], str) or not selection["replicate_note"].strip()):
        out.append(f"{tag}: selection.replicate_note is empty")
    if selection.get("outcome") == SEL_USER_OVERRIDE_OVER_BUDGET:
        over, rule_pp, budget = selection.get("override_reading_pp"), selection.get("rule_reading_pp"), selection.get("budget_pp")
        if _is_number(over) and _is_number(budget) and not over > budget:
            out.append(f"{tag}: selection claims {SEL_USER_OVERRIDE_OVER_BUDGET!r} but override_reading_pp {over} <= budget_pp {budget} — the "
                       "selected point is within budget, so the word is false")
        if _is_number(rule_pp) and _is_number(budget) and rule_pp > budget:
            out.append(f"{tag}: selection.rule_reading_pp {rule_pp} > budget_pp {budget} — the rule's return is within budget by construction; "
                       "this is not the rule's reading")
        rr = selection.get("rule_returns")
        rr_dose = dose_of_rule_return(rr)
        if rr_dose is None:
            out.append(f"{tag}: selection.rule_returns {rr!r} names no dose the export can read (a number or the lane's coefficient view "
                       "spelling, e.g. ar_c0p2) — refusing to guess which point the rule returned")
        elif dose_value is not None and math.isclose(rr_dose, dose_value, abs_tol=1e-9):
            out.append(f"{tag}: selection.rule_returns {rr!r} names the selected dose {dose_value} — then the rule chose it and there was no "
                       f"override; {SEL_USER_OVERRIDE_OVER_BUDGET!r} is false")
    return out


def fit_provenance_value(fp: Mapping[str, Any], key: str) -> Any:
    """The value a block carries for a canonical key, under that name or a DECLARED producer spelling of the same quantity
    (:data:`FIT_PROVENANCE_ALIASES`); ``None`` when it carries neither (an absent FACT, which refuses)."""
    for name in (key, *FIT_PROVENANCE_ALIASES.get(key, ())):
        v = fp.get(name)
        if v is not None and not (isinstance(v, str) and not v.strip()):
            return v
    return None


def fit_provenance_problems(fp: Any, *, tag: str) -> list[str]:
    """Every way a ``fit_provenance`` block fails its contract, by name: a mapping carrying every :data:`FIT_PROVENANCE_FIELDS` key
    NON-NULL (no empty string) under that name or a declared producer spelling of the same quantity
    (:data:`FIT_PROVENANCE_ALIASES`, :func:`fit_provenance_value`), the three digests 64-hex, the count / gate sub-blocks
    non-empty mappings, ``fit_layer`` and ``served_layer`` integers, ``direction_norm`` a number. Extra keys ride along
    verbatim. Never a default, and never another field standing in for a fact the record does not carry."""
    if not isinstance(fp, Mapping):
        return [f"{tag}: fit_provenance {fp!r} is not a mapping"]
    out: list[str] = []
    missing = [k for k in FIT_PROVENANCE_FIELDS if fit_provenance_value(fp, k) is None]
    if missing:
        out.append(f"{tag}: fit_provenance is missing {missing} (every key of {list(FIT_PROVENANCE_FIELDS)} is required, non-null, "
                   f"under that name or a declared producer spelling {FIT_PROVENANCE_ALIASES}; the export never guesses a fit fact)")
    for k in FIT_PROVENANCE_SHA256_FIELDS:
        v = fit_provenance_value(fp, k)
        if v is not None and not (isinstance(v, str) and _SHA256_RE.match(v)):
            out.append(f"{tag}: fit_provenance.{k} {v!r} is not a 64-hex sha256")
    for k in FIT_PROVENANCE_GIT_SHA_FIELDS:
        v = fit_provenance_value(fp, k)
        if v is not None and not (isinstance(v, str) and _GIT_SHA_RE.match(v)):
            out.append(f"{tag}: fit_provenance.{k} {v!r} is not a 40-hex git commit")
    for k in FIT_PROVENANCE_MAPPING_FIELDS:
        v = fit_provenance_value(fp, k)
        if v is not None and (not isinstance(v, Mapping) or not v):
            out.append(f"{tag}: fit_provenance.{k} {v!r} is not a non-empty mapping")
    for k in ("fit_layer", "served_layer"):
        v = fit_provenance_value(fp, k)
        if v is not None and (not isinstance(v, int) or isinstance(v, bool)):
            out.append(f"{tag}: fit_provenance.{k} {v!r} is not an integer")
    norm = fit_provenance_value(fp, "direction_norm")
    if norm is not None and not _is_number(norm):
        out.append(f"{tag}: fit_provenance.direction_norm {norm!r} is not a number")
    return out


def serving_commit_problems(value: Any, *, tag: str, cell: Any, has_reading: bool) -> list[str]:
    """``serving_commit``: null, or a 40-hex git commit. REQUIRED (non-null) on a row WITH A READING of a cell in
    :data:`CELLS_REQUIRING_SERVING_COMMIT` — that row was served under SOME stack, and a store record that cannot name it is
    refused by name rather than exported with a null or a stand-in (see the module docstring for what the record does and
    does not carry today). A row of that cell with no reading served nothing and carries null."""
    if value is None:
        if cell in CELLS_REQUIRING_SERVING_COMMIT and has_reading:
            return [f"{tag}: serving_commit is null on a served row of {cell!r} — the serving stack's commit is required here and no "
                    "authoritative field of the store record names it (producer.tree_sha is the DRIVER's tree; serving_config.json "
                    "records digests and a repo path; the native serving attestation records no tree) — refusing rather than inventing one"]
        return []
    if not (isinstance(value, str) and _GIT_SHA_RE.match(value)):
        return [f"{tag}: serving_commit {value!r} is not a 40-hex git commit"]
    return []


def row_problems(row: Mapping[str, Any]) -> list[str]:
    """Every way ``row`` fails the export's contract, by name (``[]`` for a sound row): the exact field set and
    order; the vocabularies (``paper_arm`` held to the declared table for the row's ``cell``); the /3 provenance fields
    where they apply (:func:`selection_problems`, :func:`fit_provenance_problems`, :func:`serving_commit_problems` —
    a block that is present is verified on every row; a block is REQUIRED on the all-positions Arditi rows and under the
    override word); the count identities (``n_paired == n_scored + n_unscored``, ``n_unscored == sum of
    unscored_reason``); the ROUND-TRIP identities ``point_conditional x n_scored == n_correct`` and
    ``point_score_as_is x n_paired == n_correct`` (integers); the CIs bracketing their points; ``score_as_is <=
    conditional`` (equality exactly when nothing is unscored)."""
    out: list[str] = []
    if tuple(row.keys()) != FIELDS:
        out.append(f"fields are {list(row.keys())}, not {list(FIELDS)}")
        return out
    tag = f"{row['instrument']}:{row['model_key']}/{row['view']}/{row['cell']}@{row['leg']}"
    if row["instrument"] not in INSTRUMENTS:
        out.append(f"{tag}: instrument {row['instrument']!r} not in {INSTRUMENTS}")
    if row["paper_arm"] is not None and row["paper_arm"] not in PAPER_ARM_KEYS:
        out.append(f"{tag}: paper_arm {row['paper_arm']!r} not in {PAPER_ARM_KEYS} (nor null)")
    if row["paper_arm"] != paper_arm_of(row["cell"]):
        out.append(f"{tag}: paper_arm {row['paper_arm']!r} != the declared key {paper_arm_of(row['cell'])!r} for cell {row['cell']!r}")
    if row["rung_state"] not in RUNG_STATES:
        out.append(f"{tag}: rung_state {row['rung_state']!r} not in {RUNG_STATES}")
    if row["object_generation"] not in OBJECT_GENERATIONS:
        out.append(f"{tag}: object_generation {row['object_generation']!r} not in {OBJECT_GENERATIONS}")
    if row["selection_outcome"] not in SELECTION_OUTCOMES:
        out.append(f"{tag}: selection_outcome {row['selection_outcome']!r} not in {SELECTION_OUTCOMES}")
    if row["censoring_note"] != CENSORING_NOTE:
        out.append(f"{tag}: censoring_note is not the literal sentence")
    if not isinstance(row["source"], str) or not row["source"]:
        out.append(f"{tag}: source is empty")
    out += provenance_problems(row, tag=tag)
    if row["n_paired"] is None:
        # a row with no reading: every count field null together
        stray = [f for f in COUNT_FIELDS if row[f] is not None]
        if stray:
            out.append(f"{tag}: no reading (n_paired null) but {stray} carry values")
        return out
    n, ns, nu, nc = row["n_paired"], row["n_scored"], row["n_unscored"], row["n_correct"]
    for f in ("n_paired", "n_scored", "n_unscored", "n_at_cap", "n_unterminated_not_at_cap", "n_correct"):
        if not isinstance(row[f], int) or isinstance(row[f], bool) or row[f] < 0:
            out.append(f"{tag}: {f}={row[f]!r} is not a non-negative integer")
    if out:
        return out
    if n != ns + nu:
        out.append(f"{tag}: n_paired {n} != n_scored {ns} + n_unscored {nu}")
    if nc > ns:
        out.append(f"{tag}: n_correct {nc} > n_scored {ns}")
    if row["n_at_cap"] > n:
        out.append(f"{tag}: n_at_cap {row['n_at_cap']} > n_paired {n}")
    nrs, nrs_of = row[REFUSAL_SHAPED_FIELD], row[REFUSAL_SHAPED_OF_FIELD]
    if (nrs is None) != (nrs_of is None):
        # /4: the count and its denominator are null TOGETHER (never counted) or present together — a count without its n is not auditable,
        # a denominator without its count is a claim nobody made
        out.append(f"{tag}: {REFUSAL_SHAPED_FIELD}={nrs!r} and {REFUSAL_SHAPED_OF_FIELD}={nrs_of!r} must be null together or present together")
    elif nrs is not None:
        # the class lives INSIDE the scored draws the classifier examined: count <= examined <= n_scored (the way n_correct <= n_scored)
        bad = [f for f, v in ((REFUSAL_SHAPED_FIELD, nrs), (REFUSAL_SHAPED_OF_FIELD, nrs_of)) if not isinstance(v, int) or isinstance(v, bool) or v < 0]
        if bad:
            out.append(f"{tag}: {bad} not non-negative integers")
        else:
            if nrs > nrs_of:
                out.append(f"{tag}: {REFUSAL_SHAPED_FIELD} {nrs} > {REFUSAL_SHAPED_OF_FIELD} {nrs_of}")
            if nrs_of > ns:
                out.append(f"{tag}: {REFUSAL_SHAPED_OF_FIELD} {nrs_of} > n_scored {ns}")
    if row["ci_method"] not in CI_METHODS:
        out.append(f"{tag}: ci_method {row['ci_method']!r} not in {CI_METHODS}")
    if not math.isclose(row["at_cap_rate"], row["n_at_cap"] / n, abs_tol=1e-6):
        out.append(f"{tag}: at_cap_rate {row['at_cap_rate']} != n_at_cap / n_paired = {row['n_at_cap'] / n}")
    ur = row["unscored_reason"]
    if not isinstance(ur, Mapping) or tuple(ur.keys()) != UNSCORED_REASONS:
        out.append(f"{tag}: unscored_reason keys {list(ur.keys()) if isinstance(ur, Mapping) else ur!r} != {list(UNSCORED_REASONS)}")
    elif sum(int(v) for v in ur.values()) != nu:
        out.append(f"{tag}: n_unscored {nu} != sum(unscored_reason) {sum(int(v) for v in ur.values())}")
    elif any(int(v) < 0 for v in ur.values()):
        out.append(f"{tag}: a negative unscored_reason count")
    if ns == 0:
        if row["point_conditional"] is not None or row["ci_conditional_low"] is not None or row["ci_conditional_high"] is not None:
            out.append(f"{tag}: nothing scored but point_conditional / its CI carry values")
    else:
        if not _is_int_multiple(row["point_conditional"], ns, nc):
            out.append(f"{tag}: ROUND TRIP point_conditional {row['point_conditional']} x n_scored {ns} != n_correct {nc}")
        _ci_problems(out, tag, "conditional", row["point_conditional"], row["ci_conditional_low"], row["ci_conditional_high"])
    if not _is_int_multiple(row["point_score_as_is"], n, nc):
        out.append(f"{tag}: ROUND TRIP point_score_as_is {row['point_score_as_is']} x n_paired {n} != n_correct {nc}")
    _ci_problems(out, tag, "score_as_is", row["point_score_as_is"], row["ci_score_as_is_low"], row["ci_score_as_is_high"])
    if ns and row["point_conditional"] is not None and row["point_score_as_is"] is not None:
        if nu == 0 and not math.isclose(row["point_conditional"], row["point_score_as_is"], abs_tol=ROUND_TRIP_TOL):
            out.append(f"{tag}: nothing unscored yet conditional {row['point_conditional']} != score_as_is {row['point_score_as_is']}")
        if nu > 0 and nc > 0 and not row["point_score_as_is"] < row["point_conditional"]:
            out.append(f"{tag}: {nu} unscored yet score_as_is {row['point_score_as_is']} is not below conditional {row['point_conditional']}")
    return out


def provenance_problems(row: Mapping[str, Any], *, tag: str) -> list[str]:
    """The /3 provenance fields' problems on one row: a ``selection`` block is REQUIRED when the row's ``selection_outcome`` is
    the override word (and a row of the all-positions cell must carry that word — a null there would read as the rule's own
    pick); ``fit_provenance`` is REQUIRED on :data:`CELLS_REQUIRING_FIT_PROVENANCE`; ``serving_commit`` on a served row of
    :data:`CELLS_REQUIRING_SERVING_COMMIT`. A block that is PRESENT is verified on every row, whatever the cell."""
    out: list[str] = []
    cell, sel_outcome = row["cell"], row["selection_outcome"]
    if cell in CELLS_REQUIRING_FIT_PROVENANCE and sel_outcome not in ARDITI_ALLPOS_SELECTION_OUTCOMES:
        out.append(f"{tag}: selection_outcome {sel_outcome!r} on {cell!r} — this arm's operating point is a DECLARED USER OVERRIDE and "
                   f"says so in exactly one word, {SEL_USER_OVERRIDE_OVER_BUDGET!r}; a null would read as the rule's own pick")
    if row["selection"] is None:
        if sel_outcome == SEL_USER_OVERRIDE_OVER_BUDGET:
            out.append(f"{tag}: selection_outcome {sel_outcome!r} but no selection block — the override word requires the block that states "
                       "both readings, the budget and the ladder cap")
    else:
        out += selection_problems(row["selection"], tag=tag, selection_outcome=sel_outcome, cell=cell)
    if row["fit_provenance"] is None:
        if cell in CELLS_REQUIRING_FIT_PROVENANCE:
            out.append(f"{tag}: fit_provenance is null on {cell!r} — the block is required on this arm (every key of "
                       f"{list(FIT_PROVENANCE_FIELDS)}); the export never fills it with a null or a guess")
    else:
        out += fit_provenance_problems(row["fit_provenance"], tag=tag)
    out += serving_commit_problems(row["serving_commit"], tag=tag, cell=cell, has_reading=row["n_paired"] is not None)
    return out


def _ci_problems(out: list[str], tag: str, name: str, point: float | None, low: float | None, high: float | None) -> None:
    if point is None:
        return
    if low is None or high is None:
        out.append(f"{tag}: {name} has a point but no CI")
        return
    if not (low - 1e-9 <= point <= high + 1e-9):
        out.append(f"{tag}: {name} CI [{low}, {high}] does not bracket the point {point}")
    if not (0.0 - 1e-9 <= low <= high <= 1.0 + 1e-9):
        out.append(f"{tag}: {name} CI [{low}, {high}] is not an interval in [0, 1]")


def verify_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """Refuse (:class:`PaperRowsError`) unless every row passes :func:`row_problems` and no two rows share a key
    ``(instrument, model_key, view, cell, leg)``."""
    problems: list[str] = []
    seen: dict[tuple[Any, ...], int] = {}
    for i, row in enumerate(rows):
        problems += row_problems(row)
        if tuple(row.keys()) == FIELDS:
            key = (row["instrument"], row["model_key"], row["view"], row["cell"], row["leg"])
            if key in seen:
                problems.append(f"rows {seen[key]} and {i} share the key {key}")
            seen.setdefault(key, i)
    if problems:
        raise PaperRowsError(f"{len(problems)} paper-row problem(s): " + "; ".join(problems[:12]) + (" ..." if len(problems) > 12 else ""))


# --------------------------------------------------------------------------- #
# the writers                                                                  #
# --------------------------------------------------------------------------- #


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return json.dumps(value, sort_keys=False, separators=(",", ":"))
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_paper_rows(rows: Sequence[Mapping[str, Any]], out_dir: Path | str, *, instrument: str, report_path: Path | str | None,
                     not_exported: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Verify ``rows`` and write ``paper_rows.json`` + ``paper_rows.csv`` into ``out_dir`` (created if absent).
    Nothing else in ``out_dir`` is touched. Returns ``{"json", "csv", "n_rows", "n_not_exported"}``."""
    verify_rows(rows)
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = {"schema": SCHEMA, "instrument": instrument, "report": str(report_path) if report_path is not None else None,
           "fields": list(FIELDS), "n_rows": len(rows), "censoring_note": CENSORING_NOTE,
           "not_exported": [dict(x) for x in not_exported], "rows": [dict(r) for r in rows]}
    jp, cp = out_dir / JSON_NAME, out_dir / CSV_NAME
    jp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    with cp.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for r in rows:
            w.writerow([_csv_cell(r[k]) for k in FIELDS])
    return {"json": str(jp), "csv": str(cp), "n_rows": len(rows), "n_not_exported": len(not_exported)}


def read_paper_rows(path: Path | str) -> list[dict[str, Any]]:
    """The rows of a ``paper_rows.json`` (its ``rows`` list), verified under the current contract. A /2 document (written
    before the three provenance fields) reads with ``selection`` / ``serving_commit`` / ``fit_provenance`` filled null — no arm
    a /2 producer knew requires them, and a /2 row of an arm that does refuses in :func:`row_problems` by name (its
    ``paper_arm`` cannot match the declared key either). Any other schema refuses."""
    doc = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(doc, Mapping) or doc.get("schema") not in READABLE_SCHEMAS or not isinstance(doc.get("rows"), list):
        raise PaperRowsError(f"{path}: not a {' / '.join(READABLE_SCHEMAS)} document")
    rows = [dict(r) for r in doc["rows"]]
    if doc["schema"] == SCHEMA_2:
        rows = [{**r, **{f: None for f in PROVENANCE_FIELDS if f not in r}} for r in rows]
    if doc["schema"] in (SCHEMA_2, SCHEMA_3):
        # a /2 or /3 producer never counted the refusal-shaped class: the count and its denominator absent (null) TOGETHER, never zero
        rows = [{**r, **{f: None for f in REFUSAL_SHAPED_FIELDS if f not in r}} for r in rows]
    rows = [{f: r[f] for f in FIELDS} if set(r) == set(FIELDS) else r for r in rows]
    verify_rows(rows)
    return rows


def large_gap_rows(rows: Sequence[Mapping[str, Any]], *, pp: float = 5.0) -> list[dict[str, Any]]:
    """The rows whose conditional and score-as-is points differ by at least ``pp`` percentage points — the cells
    where which primary a figure draws changes what it says."""
    out = []
    for r in rows:
        if r["point_conditional"] is None or r["point_score_as_is"] is None:
            continue
        gap = (r["point_conditional"] - r["point_score_as_is"]) * 100.0
        if gap >= pp - 1e-9:
            out.append({**dict(r), "gap_pp": gap})
    return out
