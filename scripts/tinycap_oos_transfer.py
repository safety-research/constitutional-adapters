"""Read out the fit-support-transfer 2x2 (experiment 1: thinking transfer) and file the verdict.

Single-shot: Qwen3.5-4B x {none, safety_refusal__psr_kl_all} x {thinking off, on}
on GSM8K paired-200, k = 1.

    PYTHONPATH=src python scripts/tinycap_oos_transfer.py --gen-root <local copy of
        /vol/tinycap/gen> --sha <sha7> [--analysis-sha <sha>] [--wave-root <copy> --wave-sha c7e1796]
        [--prompt-budget oos_prompt_budget.json] [--wandb-mode online|disabled]
        [--figs-dir DIR] [--out report.json] [--mirror-dir DIR | --no-mirror]

``--sha`` is the PRODUCER/STORE pin (the run's ``oos-<sha7>`` out-suffix); the report also
records ``analysis_code_sha``, the git sha of the tree this code ran from (read at runtime,
or ``--analysis-sha``) — a re-analysis of old stores moves the second and not the first.

The four stores are ``<gen-root>/qwen35_4b/<cell>/thinking_<arm>/oos-<sha7>/cells/<key>/``.
Every per-leg number is keyed on the FULL (cell, arm) leg — never on
``petri_stats.classify_cell_id``, which folds a ``@thinking`` variant into its base.
The SUMMARY is authoritative for every rate and the ``outcomes/<task>.json``
record for every mask (verified against the summary before any statistic);
``rows.jsonl``, the ``.eval`` logs, the usage ledger and ``evidence.json`` are
cross-checks, never denominators. Every leg must be DONE (a PARTIAL marker is a
limited run — diagnostics only) over the frozen paired-200 population
(:func:`population_check`; ``--paired-ids``). The estimand and the rule:
:func:`estimands`, :func:`classify_word`, :func:`verdict_from_bounds`; the gates:
:func:`instrumentation_gates`.
"""
from __future__ import annotations

import argparse
import filecmp
import json
import math
import re
import shutil
import subprocess
import sys
import textwrap
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, NamedTuple

from sisteer.capeval.tiers import pairing as pr
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap.completion import CompletionError

# --------------------------------------------------------------------------- #
# the experiment's vocabulary                                                 #
# --------------------------------------------------------------------------- #

#: The authorised tuple's spellings (contracts.AUTHORISED_OOS_THINKING).
MODEL: str = C.AUTHORISED_OOS_THINKING.model
CELLS: tuple[str, ...] = tuple(C.AUTHORISED_OOS_THINKING.cells)
ARMS: tuple[str, ...] = pr.LEG_ARMS                  # (off, on): pairing's leg vocabulary
TASK: str = C.GSM8K_PAIRED_TASK
#: the estimand's cell KIND per authorised cell (contracts.OOS_CELL_KIND, beside
#: the tuple; pairing.LEG_CELLS' vocabulary) and its inverse
KIND_CELL: Mapping[str, str] = {v: k for k, v in C.OOS_CELL_KIND.items()}
#: The frozen paired-200 population every leg must have attempted:
#: ``--paired-ids plan`` derives the id set from the task plan itself.
PAIRED_IDS_FROM_PLAN = "plan"
OUT_SUFFIX_PREFIX = "oos-"
WAVE_SHA_DEFAULT = "c7e1796"
#: The thinking-OFF comparator at the thinking-ON cap: two more
#: legs — `none` / `psr` thinking-OFF at cap 8192 — under their OWN out-suffix and
#: store pin, ``offcap-<sha7>`` (``--offcap-sha``), read only when asked.
OFFCAP_OUT_SUFFIX_PREFIX = "offcap-"
#: The comparator's OFF completion cap: the ON cap under the OFF render — the
#: contract's own constant (``contracts.OFF_CAP_OVERRIDE_MAX_TOKENS``, pinned there
#: to ``OOS_THINKING_MAX_TOKENS`` by reference); the executed cap
#: per task comes from the authorised comparator tuple through
#: ``contracts.off_cap_override_max_tokens`` (:func:`expected_cap`).
OFFCAP_CAP: int = C.OFF_CAP_OVERRIDE_MAX_TOKENS
#: the comparator store's ONE optional identity key and its value:
#: the full task name -> the cap the authorised comparator names for it
OFFCAP_MAX_TOKENS_BY_TASK: Mapping[str, int] = {
    TASK: int(C.off_cap_override_max_tokens(TASK.removeprefix(C.INSPECT_TASK_PREFIX)))}
#: The driver's evidence block on the comparator's base-route (`none`) leg
#: is read under the contract's ONE spelling,
#: ``contracts.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED`` — one negative
#: probe: the base route must answer the thinking-ON declaration key with 400
#: ``thinking_on_not_lens`` BEFORE any forward — ``{status, code, ok}``; this
#: is its verdict field. Read ONLY on a leg whose VALIDATED comparator
#: classification is true (``leg["cap_override"]``), never on cap-key presence:
#: the frozen 2x2's ON manifests carry ``max_tokens_by_task`` too and have no
#: probe.
EVIDENCE_BASE_ROUTE_PROBE_OK = "ok"
#: an ON at-cap rate above this means the cap was too small
CAP_GATE_AT_CAP_RATE = 0.02
#: P2-c: an ON-leg accuracy within this of 0 or 1 is a floor / ceiling that
#: limits interpretation (stated separately from the gates)
INTERPRETATION_EDGE = 0.05

BRANCH_PRIMARY = "primary"
BRANCH_SCORE_AS_IS = "score_as_is"
BRANCHES: tuple[str, ...] = (BRANCH_PRIMARY, BRANCH_SCORE_AS_IS)

WORD_NOT_RESOLVED = "in-support effect not resolved"
WORD_SIGN_FLIP = "does_not_transfer (sign flip)"
WORD_TRANSFERS = "transfers"
WORD_COLLAPSE = "does_not_transfer (collapse toward zero)"
WORD_AMPLIFIES = "amplifies"
WORDS: tuple[str, ...] = (WORD_NOT_RESOLVED, WORD_SIGN_FLIP, WORD_TRANSFERS,
                          WORD_COLLAPSE, WORD_AMPLIFIES)
NOT_DECIDABLE_CENSORING = "not decidable at this cap (censoring)"
NOT_DECIDABLE_BRANCHES = "not decidable at this cap (branch disagreement)"
NOT_MEASURABLE = "not measurable at this cap / with this instrument"
SIGN_NOT_IDENTIFIED = "sign not identified (censoring)"
#: What ``sign_identified`` / ``signs.*.identified`` MEANS: the bound is
#: a SAMPLE censoring-sensitivity interval — the observed sign survives every
#: assignment of the censored items — never a population statement (the
#: bound's own endpoints carry bootstrap CIs, ``bound_ci_low`` / ``bound_ci_high``).
SIGN_IDENTIFIED_MEANING = ("observed sign survives every censoring assignment (sample), not a "
                           "population statement")
BOUND_LABEL = "censoring-sensitivity interval (sample bound)"
#: An OFF CI covering zero is an UNRESOLVED effect, never equivalence
#: ("free", "costs nothing") — no equivalence margin was preregistered.
OFF_NOT_RESOLVED_TEXT = "no net OFF accuracy effect was resolved at this sample size and cap"
#: The exact tests are SENSITIVITY checks beside the preregistered
#: percentile CI, never the decision.
MCNEMAR_LABEL = ("exact two-sided McNemar on the discordant pairs — a sensitivity check beside the "
                 "preregistered percentile CI, never the decision")
#: The exact test is a COMPLETE-CASE sensitivity under a scoring
#: branch — never "primary p", which reads as the primary estimand's own p.
MCNEMAR_BRANCH_LABEL = "complete-case exact sensitivity under {branch} scoring"
BRANCH_NAMES: Mapping[str, str] = {BRANCH_PRIMARY: "primary", BRANCH_SCORE_AS_IS: "score-as-is"}
#: What a one-sided discordance does and does NOT imply. "Cannot draw
#: an improvement" is a fact about an EMPIRICAL bootstrap of the complete-case
#: pairs; the registered conditional bootstrap resamples censoring masks and
#: divides by per-replicate denominators, so its replicates can land on either
#: side of zero (the real run: 9,961 negative / 20 zero / 19 positive of
#: 10,000 for effect_on) — the counts are filed beside the note
#: (``replicate_signs``, from the bootstrap's own replicate arrays).
ONE_SIDED_NOTE_REGISTERED = ("the registered conditional bootstrap also resamples censoring masks and uses "
                             "per-replicate denominators, so this property does not apply literally to its "
                             "replicates")
#: The analysis review's M1: that statement is FALSE for a contrast whose two
#: legs are fully scored under the branch — every censoring mask is one and
#: every replicate denominator is n, so the registered bootstrap IS the
#: empirical bootstrap of the pairs and cannot generate a contrast of the
#: opposite sign (the real run's psr cap relief: 0 negative / 196 zero / 9,804
#: positive of 10,000). The note therefore reads the contrast's ACTUAL masks
#: (``fully_scored`` on the McNemar record) and wears this tail when they are
#: all one: an interval endpoint on the discordances' side is then a property
#: of the sparse empirical distribution, not corroboration.
ONE_SIDED_NOTE_FULLY_SCORED = ("both legs are fully scored under this branch (every censoring mask one, every "
                               "replicate denominator {n}), so the registered conditional bootstrap is that empirical "
                               "bootstrap here and the constraint applies to its replicates — an interval endpoint on "
                               "the discordances' side is a property of the sparse empirical distribution ({d} "
                               "discordant of {n}), not corroboration")
_NUMBER_WORDS: Mapping[int, str] = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven",
                                    8: "eight", 9: "nine", 10: "ten"}
NOT_ROBUST_TEXT = "not a robust steering-harms-thinking claim"
#: The ON-vs-OFF shift changes the render AND the completion cap.
RENDER_AND_CAP_TEXT = ("this comparison changes both the render and the completion cap, so it does not "
                       "isolate thinking from relief of OFF truncation")
SELECTED_SUBSET_LABEL = "selected-subset diagnostic (exploratory, not an estimate)"
#: EVERY cross-arm contrast — shift_none, shift_psr AND the DiD —
#: compares the OFF render at the table's cap with the ON render at its own
#: cap. Each is a render+cap contrast; none is a thinking-only one, so an
#: interaction the DiD might show is with the combined render-plus-cap
#: configuration, never "ON-specific" (that word is banned from the report).
RENDER_CAP_CONTRAST_LABEL = "render+cap contrast"
#: A ceiling on the ON legs is DIRECTIONAL — it limits the evidence a
#: benefit could leave (little room above the unsteered cell, few discordant
#: errors) and says nothing about a loss, which has the whole scale below it.
#: "Little room in either direction" biased against the observed negative
#: effect and is banned.
CEILING_TEXT = ("limits evidence for a benefit ({room} of room) and leaves few discordant errors; "
                "it does not constrain a steering-induced loss")
FLOOR_TEXT = ("limits evidence for a loss ({room} of room) and leaves few discordant successes; "
              "it does not constrain a steering-induced gain")
#: the closed-think capped item: reasoning completed, the answer truncated
CLOSED_THINK_CAPPED_POLICY = ("truncated answer after completed reasoning; the registered policy keeps "
                              "it scored — recoding it censored is a sensitivity, not the estimand")
PARITY_NO_DRIFT = "no drift detected by the registered screen"
PARITY_NOTE = ("a passed screen is not equivalence and never reads as parity established: like-for-like "
               "on the checked task and serving settings, not identical producer execution")

GATE_PASS = "PASSED"
GATE_FAIL = "FAILED"
GATE_PENDING = "PENDING"   # a stamp this tree does not write yet (plan S3) — named, never a pass
#: the gates (+ the population gate), in report order
GATE_NAMES: tuple[str, ...] = ("population", "cap", "outcomes", "split_anomalies", "declaration")

# -- the comparator's vocabulary -----------------------------------------------------
#: The three fixed mappings every :func:`estimands` call takes — the natural 2x2
#: (MAP_ORIG), the cap-relief relabelling (the OFF/8192 legs in the ``on`` slots:
#: MAP_CAP) and the matched-budget relabelling (the OFF/8192 legs in the ``off``
#: slots: MAP_MATCHED). Two relabelled calls of the UNCHANGED four-leg bootstrap,
#: whose replicate draws are a pure function of (seed, n, B) and so coherent.
MAP_ORIG, MAP_CAP, MAP_MATCHED = "orig", "cap", "matched"
MAPPINGS: tuple[str, ...] = (MAP_ORIG, MAP_CAP, MAP_MATCHED)
#: what each relabelled call's internal contrast IS, by mapping
CONTRAST_NAMES_BY_MAPPING: Mapping[str, Mapping[str, str]] = {
    MAP_ORIG: {c: c for c in pr.CONTRASTS},
    MAP_CAP: {"effect_off": "effect_off", "effect_on": "effect_off_8192", "shift_none": "cap_relief_none",
              "shift_psr": "cap_relief_psr", "did": "did_cap_relief"},
    MAP_MATCHED: {"effect_off": "effect_off_8192", "effect_on": "effect_on", "shift_none": "shift_none_matched",
                  "shift_psr": "shift_psr_matched", "did": "did_matched"},
}
#: the six comparator contrasts the spec names (the cap-relief interaction
#: ``did_cap_relief`` is reported in its block, not a finding)
OFFCAP_CONTRASTS: tuple[str, ...] = ("cap_relief_none", "cap_relief_psr", "shift_none_matched",
                                     "shift_psr_matched", "did_matched", "effect_off_8192")
#: the three served configurations of the six-leg design, by key, and which one
#: each internal 2x2 slot holds under each mapping
CFG_OFF, CFG_OFF_OVERRIDE, CFG_ON = "off", "off_override", "on"
SLOT_CONFIGURATIONS: Mapping[str, Mapping[str, str]] = {
    MAP_ORIG: {C.THINKING_OFF: CFG_OFF, C.THINKING_ON: CFG_ON},
    MAP_CAP: {C.THINKING_OFF: CFG_OFF, C.THINKING_ON: CFG_OFF_OVERRIDE},
    MAP_MATCHED: {C.THINKING_OFF: CFG_OFF_OVERRIDE, C.THINKING_ON: CFG_ON},
}
VERDICT_SENSITIVITY_8192_LABEL = ("the transfer rule keyed on the OFF/8192 leg — a sensitivity, not the registered "
                                  "input (spec: the rule's input stays the 512 leg)")
#: At the comparator's cap: an unresolved OFF/8192 effect is said so, at that cap
OFF_NOT_RESOLVED_TEXT_8192 = f"{OFF_NOT_RESOLVED_TEXT} {OFFCAP_CAP}"
#: §6.5 [R1]: the OFF/8192 legs take no parity read; the transition description is not one
OFFCAP_PARITY_NOTE = ("the OFF/8192 legs are not screened against the wave: max_tokens is a parity field, so the "
                      "pair is non-comparable by the screen's own rule (a different completion cap is a different "
                      "instrument); the OFF/512 → OFF/8192 transition description is exploratory and is not parity "
                      "evidence")
#: §6.5 [R1]/[R2]: the OFF/512 → OFF/8192 block is a transition DESCRIPTION — counts split
#: by the first draw's at-cap status, no expectation, no noise-floor or parity reading
OFF512_TO_OFF8192_LABEL = ("exploratory transition description (OFF/512 → OFF/8192 per cell): counts only, split by "
                           "the first draw's at-cap status; no expectation attached")
OFF512_TO_OFF8192_WHY = ("no expectation is attached because the split conditions on the FIRST draw's stopping "
                         "behaviour, which correlates with correctness (an item that stopped before 512 is one the "
                         "model answered concisely on that draw), so the not-capped subset's second-draw difference "
                         "is not zero-centred even under identical serving and identical seeds, and no noise floor "
                         "can be read off it; this description is not parity evidence")
#: the block's caveat sentences — the ONLY places its vocabulary may name what it is NOT
OFF512_TO_OFF8192_CAVEATS: tuple[str, ...] = (OFF512_TO_OFF8192_LABEL, OFF512_TO_OFF8192_WHY, OFFCAP_PARITY_NOTE)
OFFCAP_PROVENANCE_NOTE = ("three shas, kept apart: `sha` is the 2x2 store pin (the oos-<sha7> out-suffix), "
                          "`offcap_sha` the comparator store pin (the offcap-<sha7> out-suffix), "
                          "`analysis_code_sha` the tree this code ran from; a re-analysis moves the third and "
                          "never the first two")

#: The experiment's FIXED sampling row, read from the ONE place contracts
#: defines the campaign row (``contracts.sampling_for`` over the tuple's model
#: and sampling mode — never re-spelled here), and the summary fields it must
#: appear under on EVERY leg (a leg drawn under another row is another
#: instrument, not a control).
FIXED_SAMPLING: C.SamplingRow = C.sampling_for(C.model_row(MODEL).model_key,
                                               C.AUTHORISED_OOS_THINKING.sampling_mode)
SAMPLING_FIELDS: tuple[str, ...] = ("temperature", "top_p", "top_k")
#: The authorised model's registry row (``contracts.model_row`` over the
#: tuple's slug): the served target and its PINNED revision every leg's
#: manifest must name POSITIVELY (equality across the legs is not identity:
#: four stores at another model agree with each other), and the family whose
#: reasoning-parser pin (``contracts.REASONING_PARSER_BY_FAMILY``; Qwen3.5:
#: none, the thought stays in ``content``) enters the think-split derivation.
MODEL_ROW: C.ModelRow = C.model_row(MODEL)
FAMILY_REASONING_PARSER: str | None = C.REASONING_PARSER_BY_FAMILY.get(MODEL_ROW.family)
#: The nuisance parameters the within-arm design holds CONSTANT across the
#: four legs (steered minus none within an arm, then across arms):
#: the summary's, and the manifest identity's (served geometry, reduction
#: mode, the model and its revision). The caps are per ARM by design (8192 ON,
#: the table's OFF) and are asserted per leg (:func:`expected_cap`), not here.
NUISANCE_SUMMARY_FIELDS: tuple[str, ...] = (
    "temperature", "top_p", "top_k", "samples_per_item", "item_set", "task", "reasoning_parser",
    "unterminated_policy")
NUISANCE_IDENTITY_FIELDS: tuple[str, ...] = (
    "served.max_model_len", "served.max_num_seqs", "served.concurrency", C.REDUCTION_MODE_FIELD,
    "model_id", "model_revision")
NUISANCE_FIELDS: tuple[str, ...] = NUISANCE_SUMMARY_FIELDS + NUISANCE_IDENTITY_FIELDS
#: The fitted OBJECT's identity (the analysis review's M3): the manifest's
#: ``provenance.artifact_sha256`` — the sha256 of the registry artifact the
#: driver served (``cell_controls.artifact_sha256``; the same digest the
#: launcher holds the standup to under ``--expect-object-sha256`` and the
#: standup attests as ``object_artifact_sha256``). NOT the intervention hash,
#: which legitimately differs on the declared ON route. Recorded on every leg
#: (``identity.object_artifact_sha256``) and, under the comparator, required
#: to agree within every cell across its configurations
#: (:func:`object_identity_check`): a changed object under the same cell name
#: passed every other admission check.
PROVENANCE_ARTIFACT_SHA256_FIELD = "artifact_sha256"
OBJECT_IDENTITY_FIELD = f"{C.MANIFEST_KEY_PROVENANCE}.{PROVENANCE_ARTIFACT_SHA256_FIELD}"

#: The three states of a stamped evidence value, kept DISTINCT: the key absent
#: (the producer has not landed), an explicit JSON null (the producer landed
#: and the certification did not run), a present value of the contract's type —
#: and a present value of ANY OTHER type, which is a hard failure naming the
#: key and the value, never a pending stamp.
EVIDENCE_ABSENT, EVIDENCE_NULL, EVIDENCE_STAMPED, EVIDENCE_OFF_CONTRACT = (
    "absent", "null", "stamped", "off_contract")
#: the certification flags of the header block, all JSON booleans by contract
HEADER_CERTIFICATION_KEYS: tuple[str, ...] = (
    C.OOS_EVIDENCE_HEADER_CERTIFIED, C.OOS_EVIDENCE_PROMPT_OPENED_THINK_CERTIFIED,
    C.OOS_EVIDENCE_HEADER_ATTESTED_AGREES)


def stamp_name(block: str, key: str) -> str:
    """A producer stamp's ONE spelling in the pending lists: ``<block>.<key>``."""
    return f"{block}.{key}"


#: The task summary's LEDGER-derived count of the task's rejected
#: calls (``contracts.SUMMARY_PAYLOAD_FIELDS``), written by the Inspect tier
#: from the same file the ledger confirmation reads here; the two must agree.
SUMMARY_REJECTED_FIELD = "n_http_rejected_retried"


#: The ONLY pending items ``--allow-pending-gates`` may waive: the producer
#: stamps Phase B's driver writes (plan S3.3 / S3.4 / S3.6 / S3.8) under the
#: contract's names — the header certifications, the probe's render arm, the
#: live prompt-budget pass, and the summary's §3.4 regime triple. An operator
#: input that is missing (``--paired-ids``, ``--prompt-budget``) or anything
#: else pending withholds the word regardless of the flag.
WAIVABLE_PENDING_STAMPS: tuple[str, ...] = (
    *(stamp_name(C.OOS_EVIDENCE_HEADER_BLOCK, k) for k in HEADER_CERTIFICATION_KEYS),
    stamp_name(C.OOS_EVIDENCE_BITES_BLOCK, C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM),
    stamp_name(C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK, C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_MAX),
    stamp_name(C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK, C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_P99),
    *(stamp_name("summary", f) for f in C.EXPECTED_STAMP_TRIPLE_FIELDS),
    # The Inspect tier's ledger-derived rejected-call count — a producer
    # stamp this tree's driver writes; a store from before it is pending,
    # and the ledger read here still decides censoring on its own
    stamp_name("summary", SUMMARY_REJECTED_FIELD),
)
#: the pending entry of an operator input (never waivable)
NOT_A_PRODUCER_STAMP = None

#: The text-signature rule (moved in-repo from the retired ``analyze_oos.py``,
#: F4): a completion is called degenerate when one repeated n-gram covers this
#: share of its whitespace tokens. A SIGNATURE beside the counts, not a verdict
#: ([[small-model-degeneration-fake-safety]]).
REPEAT_NGRAM = 8
REPEAT_SHARE = 0.5

#: the W&B purpose: the project is sisteer.wandb_entity.wandb_project(WANDB_PURPOSE) (the site's SISTEER_WANDB_PROJECT_TINYCAP first)
WANDB_PURPOSE = "tinycap"
NUMBERS_FILENAME = "final_e1_figures_numbers.json"
TABLES_FILENAME = "final_e1_tables.json"
#: the comparator's figure set (``final_e1b_*``) and its own JSON pair
NUMBERS_OFFCAP_FILENAME = "final_e1b_figures_numbers.json"
TABLES_OFFCAP_FILENAME = "final_e1b_tables.json"


class TransferAnalysisError(RuntimeError):
    """A refusal of the analysis — loud, never a silent fallback."""


# --------------------------------------------------------------------------- #
# store reading                                                               #
# --------------------------------------------------------------------------- #


def leg_key(cell: str, arm: str, *, cap_override: bool = False) -> str:
    """The FULL leg id — ``<cell id>|<arm>`` — every per-leg record is keyed on.
    A variant cell (``…@thinking``) keys on its own full id, never its base. A
    comparator leg keys ``<cell>|off|cap8192``: the plain key would COLLIDE
    with the OFF/512 leg's."""
    return f"{cell}|{arm}" + (f"|cap{OFFCAP_CAP}" if cap_override else "")


def out_root_of(gen_root: Path, sha: str, *, cell: str, arm: str, prefix: str = OUT_SUFFIX_PREFIX) -> Path:
    return Path(C.cell_out_dir(str(gen_root), cell, arm, model_slug=MODEL)) / f"{prefix}{sha}"


def _read_json(p: Path) -> Any:
    """The parsed JSON at ``p``, or ``None`` ONLY when the file is absent. A
    file that is present but unreadable or malformed refuses with its cause —
    a torn ``evidence.json`` must never read as "no evidence" (silently
    PENDING) when it is a broken store."""
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise TransferAnalysisError(f"{p}: unreadable ({type(e).__name__}: {str(e)[:160]})") from e


def _mapping(v: Any) -> Mapping[str, Any]:
    """``v`` when it is a JSON object, else ``{}`` — the one coercion every
    optional sub-block of a manifest / evidence document goes through."""
    return v if isinstance(v, Mapping) else {}


def _iter_jsonl(path: Path):
    """The JSON objects of a ``.jsonl`` file, one per non-blank line; a torn
    line (the store's own reading of an interrupted write) is skipped."""
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict):
            yield r


def _cell_dir(root: Path) -> tuple[Path | None, str | None]:
    """The single ``cells/<key>/`` store under an out-root, or (None, reason).
    Two of them is a real condition (two identities under one root) and is
    named, never resolved by picking the first."""
    cells_root = root / "cells"
    cells = sorted(p for p in cells_root.glob("*") if (p / "manifest.json").is_file()) \
        if cells_root.is_dir() else []
    if not cells:
        return None, f"no cells/*/manifest.json under {root}"
    if len(cells) > 1:
        return None, (f"{len(cells)} stores under {root} — two identities in one out-root; "
                      f"name the one you mean: {[c.name for c in cells]}")
    return cells[0], None


def _rows_for_task(cell_dir: Path, task: str) -> list[dict[str, Any]]:
    """The per-item ``acc`` rows of ``task`` (never the aggregate rows), each
    with its parsed ``extra_json`` under ``_extra``."""
    out: list[dict[str, Any]] = []
    p = cell_dir / "rows.jsonl"
    if not p.is_file():
        return out
    for r in _iter_jsonl(p):
        if r.get("task") != task or r.get("item_id") == C.AGG_ITEM_ID:
            continue
        if r.get("metric") != C.METRIC_ACC:
            continue
        extra = r.get("extra_json")
        if isinstance(extra, str):
            try:
                extra = json.loads(extra)
            except ValueError:
                extra = {}
        r["_extra"] = _mapping(extra)
        out.append(r)
    return out


def rows_cross_check(rows: Sequence[Mapping[str, Any]], items: Mapping[str, Mapping[str, Any]]
                     ) -> dict[str, Any]:
    """``rows.jsonl`` (the SCORED subset) against the outcome record: the row
    item set must equal ``{i: scored}`` and every row's value the record's.
    Any difference is named; the rows are never a denominator."""
    scored = {i for i, rec in items.items() if rec["scored"]}
    by_id: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by_id.setdefault(str(r["item_id"]), []).append(r)
    dup = sorted(i for i, rs in by_id.items() if len(rs) > 1)
    only_rows = sorted(set(by_id) - scored)
    only_record = sorted(scored - set(by_id))
    value_diff = sorted(
        i for i in set(by_id) & scored
        if float(by_id[i][0].get("value")) != float(items[i]["value"]))
    problems = []
    if dup:
        problems.append(f"{len(dup)} items filed more than once (first {dup[:3]})")
    if only_rows:
        problems.append(f"{len(only_rows)} rows for items the record marks unscored or never "
                        f"attempted (first {only_rows[:3]})")
    if only_record:
        problems.append(f"{len(only_record)} scored items with no row (first {only_record[:3]})")
    if value_diff:
        problems.append(f"{len(value_diff)} items whose row value differs from the record "
                        f"(first {value_diff[:3]})")
    return {"n_rows": len(rows), "n_scored_in_record": len(scored),
            "agrees": not problems, "detail": "; ".join(problems) or None}


def _looks_repetitive(text: str) -> bool:
    toks = re.findall(r"\S+", text)
    if len(toks) < 4 * REPEAT_NGRAM:
        return False
    grams = Counter(tuple(toks[i:i + REPEAT_NGRAM]) for i in range(len(toks) - REPEAT_NGRAM + 1))
    return (max(grams.values()) * REPEAT_NGRAM) / len(toks) >= REPEAT_SHARE


def _eval_logs(cell_dir: Path, task: str) -> list[Path]:
    return sorted(cell_dir.glob(f"{C.INSPECT_LOGS_SUBDIR}/{task.replace('/', '__')}__s*/*.eval"))


def _read_eval_results(cell_dir: Path, task: str) -> tuple[list[Any] | None, list[str] | None,
                                                           str | None]:
    """(results, completion texts, reason) off the retained ``.eval`` logs:
    ``inspect_served.results_from_log`` over every sample's log (the same
    re-derivation the runner does), plus every completion string. ``(None,
    None, reason)`` when there is no log or the client venv cannot read one —
    the outcome record then stands alone, and the reason is reported."""
    logs = _eval_logs(cell_dir, task)
    if not logs:
        return None, None, (f"no Inspect .eval log under {cell_dir / C.INSPECT_LOGS_SUBDIR} — the "
                            "rows carry no completion text; the text signature and the log "
                            "re-derivation are unavailable; the outcome record stands alone")
    try:
        from inspect_ai.log import read_eval_log

        from sisteer.capeval.tiers import inspect_served as isv
    except ImportError as e:            # pragma: no cover - venv-dependent
        return None, None, f"inspect_ai / inspect_served not importable here ({e})"
    name = task.removeprefix(C.INSPECT_TASK_PREFIX)
    results: list[Any] = []
    texts: list[str] = []
    for p in logs:
        m = re.search(r"__s(\d+)(?:__a\d+)?$", p.parent.name)
        sidx = int(m.group(1)) if m else 0
        try:
            log = read_eval_log(str(p))
            results.extend(isv.results_from_log(
                log, sample_idx=sidx, primary_scorer=isv.PRIMARY_SCORER.get(name)))
        except Exception as e:                                   # noqa: BLE001
            return None, None, f"{p}: {type(e).__name__}: {str(e)[:200]}"
        for s in (log.samples or ()):
            out = getattr(s, "output", None)
            comp = getattr(out, "completion", None) if out is not None else None
            if comp is not None:
                texts.append(str(comp))
    return results, texts, None


def text_signature(texts: Sequence[str] | None, reason: str | None) -> dict[str, Any]:
    if texts is None:
        return {"n_completions": None, "n_empty": None, "empty_rate": None,
                "n_repetitive": None, "repetition_rate": None, "unavailable": reason}
    n = len(texts)
    empty = sum(1 for t in texts if not t.strip())
    rep = sum(1 for t in texts if _looks_repetitive(t))
    return {"n_completions": n, "n_empty": empty, "empty_rate": (empty / n) if n else None,
            "n_repetitive": rep, "repetition_rate": (rep / n) if n else None,
            "repetition_rule": (f"one {REPEAT_NGRAM}-gram covering >= {REPEAT_SHARE:.0%} of the "
                                "completion's whitespace tokens")}


def counts_from_results(results: Sequence[Any]) -> dict[str, Any]:
    """The outcome header's counts re-derived from ``ItemResult``-like objects
    (``inspect_served.results_from_log``'s output) — the independent reading
    the log cross-check compares to the record."""
    unscored = [r for r in results if getattr(r, "unscored", False)]
    return {
        "n_attempted_items": len({r.item_id for r in results}),
        "n_attempted_samples": len(results),
        "n_unscored_samples": len(unscored),
        "n_scored_samples": len(results) - len(unscored),
        "n_at_cap": sum(1 for r in results if getattr(r, "at_cap", False)),
        "n_think_block_present": sum(1 for r in results if getattr(r, "think_block_present", False)),
        "n_think_block_unterminated": sum(
            1 for r in results if getattr(r, "think_block_unterminated", False)),
    }


def _score_reading(scored: bool, value: Any) -> dict[str, Any]:
    """One draw's ``(availability, value)`` in the outcome record's own terms
    (``inspect_served.outcome_record``: ``scored = not unscored``, ``value`` the
    float where scored and ``None`` where not)."""
    return {"scored": bool(scored), "value": (float(value) if scored else None)}


def per_item_log_check(results: Sequence[Any], items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The retained log's results against the outcome record's items, ONE BY
    ONE, keyed by ``(item_id, sample_idx)``: score availability (``scored`` vs
    the result's ``unscored``) and score value (``value`` vs
    ``float(result.value)`` where scored) must agree on every draw, and the
    two key sets must be the same — a draw the log lacks (``missing_in_log``)
    or the record lacks (``extra_in_log``) is named. The aggregate counts
    (:func:`counts_from_results`) cannot see a store whose rows and outcomes
    swapped one correct and one incorrect draw: every marginal survives that
    swap, and it moved the comparator's cap-relief discordances 7 / 1 -> 6 / 0.
    A log carrying the same draw twice is refused outright (two results cannot
    both be the draw). The derivation of ``scored`` / ``value`` is the writer's
    (``inspect_served.outcome_record``), so the comparison is exact."""
    by_key: dict[tuple[str, int], Any] = {}
    for r in results:
        key = (str(r.item_id), int(getattr(r, "sample_idx", 0)))
        if key in by_key:
            raise TransferAnalysisError(f"the retained log carries draw {key!r} twice — two results cannot both be the draw")
        by_key[key] = r
    record = {(str(it["item_id"]), int(it.get("sample_idx", 0))): it for it in items}
    missing = sorted(k for k in record if k not in by_key)
    extra = sorted(k for k in by_key if k not in record)
    mismatches: list[dict[str, Any]] = []
    n_compared = 0
    for key in sorted(k for k in record if k in by_key):
        r, it = by_key[key], record[key]
        n_compared += 1
        log_side = _score_reading(not bool(getattr(r, "unscored", False)), r.value)
        rec_side = _score_reading(bool(it["scored"]), it.get("value"))
        if log_side != rec_side:
            mismatches.append({"item_id": key[0], "sample_idx": key[1], "log": log_side, "record": rec_side})
    return {"n_log": len(by_key), "n_record": len(record), "n_compared": n_compared, "n_mismatched": len(mismatches),
            "mismatches": mismatches, "missing_in_log": [list(k) for k in missing], "extra_in_log": [list(k) for k in extra],
            "agrees": not mismatches and not missing and not extra}


def _per_item_detail(per: Mapping[str, Any], *, show: int = 5) -> str | None:
    if per["agrees"]:
        return None
    parts = []
    if per["mismatches"]:
        shown = "; ".join(f"{m['item_id']}/{m['sample_idx']}: log scored {m['log']['scored']} value {m['log']['value']} != "
                          f"record scored {m['record']['scored']} value {m['record']['value']}" for m in per["mismatches"][:show])
        more = per["n_mismatched"] - min(show, per["n_mismatched"])
        parts.append(f"{per['n_mismatched']} of {per['n_compared']} draws differ item by item ({shown}"
                     + (f"; and {more} more" if more else "") + ")")
    if per["missing_in_log"]:
        parts.append(f"{len(per['missing_in_log'])} record draw(s) absent from the log ({per['missing_in_log'][:show]})")
    if per["extra_in_log"]:
        parts.append(f"{len(per['extra_in_log'])} log draw(s) absent from the record ({per['extra_in_log'][:show]})")
    return "; ".join(parts)


def eval_cross_check(cell_dir: Path, task: str, header: Mapping[str, Any],
                     items: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The ``.eval`` re-derivation against the outcome record (plan S5.2):
    the header's counts AND — the analysis review's M2 — every draw's score
    availability and value by ``(item_id, sample_idx)``
    (:func:`per_item_log_check`, when ``items`` is given); disagreement in
    either is a REFUSAL; absence is reported. ``log_present`` says whether a
    retained ``.eval`` log EXISTS under the store: an absent log leaves the
    record standing alone (nothing to re-derive from); a log that is present
    but could not be read here (the reader venv lacks inspect_ai, or the file
    is torn) is a re-derivation NOT RUN, which the outcomes gate records as a
    named PENDING entry rather than passing silently (the validation of the
    censored run: every leg's log was present and unread in the figure venv,
    and the gate read PASSED)."""
    present = bool(_eval_logs(cell_dir, task))
    results, texts, reason = _read_eval_results(cell_dir, task)
    sig = text_signature(texts, reason)
    if results is None:
        return {"available": False, "reason": reason, "agrees": None, "text": sig,
                "log_present": present}
    derived = counts_from_results(results)
    diffs = {k: (v, header.get(k)) for k, v in derived.items()
             if header.get(k) is not None and header.get(k) != v}
    count_detail = "; ".join(f"{k}: log {a} != record {b}" for k, (a, b) in diffs.items()) or None
    per = per_item_log_check(results, items) if items is not None else None
    per_detail = _per_item_detail(per) if per is not None else None
    detail = "; ".join(d for d in (count_detail, per_detail) if d) or None
    return {"available": True, "reason": None, "agrees": not diffs and (per is None or per["agrees"]),
            "log_present": present, "derived": derived, "count_detail": count_detail, "per_item": per,
            "detail": detail, "text": sig}


# --------------------------------------------------------------------------- #
# the ledger confirmation (plan S5.6)                                          #
# --------------------------------------------------------------------------- #


def _ledger_rows(path: Path, task: str) -> list[dict[str, Any]]:
    return [r for r in _iter_jsonl(path) if r.get("task") == task]


def checked_prompt_budget(budget: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The CPU prompt-budget artifact (plan S4.6, ``oos_prompt_budget.json``)
    as the analysis accepts it: ``None`` when none was supplied, else the
    document validated through the contract's ONE rule
    (``contracts.validate_oos_prompt_budget`` — exactly
    ``OOS_PROMPT_BUDGET_FIELDS``, statistics consistent with ``lengths``) and
    required to be the ON arm's: the three-way agreement it enters is the ON
    legs' (S5.6). A foreign, torn or OFF-arm document refuses by name — the
    tolerant reader that once probed three key spellings is gone."""
    if budget is None:
        return None
    try:
        doc = C.validate_oos_prompt_budget(budget, where="--prompt-budget artifact")
    except C.OutOfSupportPromptBudgetError as e:
        raise TransferAnalysisError(str(e)) from e
    if doc["arm"] != C.THINKING_ON:
        raise TransferAnalysisError(
            f"--prompt-budget artifact: arm {doc['arm']!r} — the analysis takes the ON arm's "
            f"artifact (the three-way prompt-max agreement is an ON-leg check, plan S5.6)")
    return doc


def checked_prompt_budget_off(budget: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The comparator's CPU artifact (``--prompt-budget-off``): the contract's
    ONE validation rule, then required to be the OFF arm's, written for the
    OVERRIDE cap (8192) on the paired-200 task — a table-cap (512) OFF
    artifact, an ON artifact or a foreign document refuses by name. It is the
    third prompt-max source on the OFF/8192 legs; the ON artifact
    (``--prompt-budget``) stays the ON legs' and is still refused if it is not
    ON (:func:`checked_prompt_budget`)."""
    if budget is None:
        return None
    try:
        doc = C.validate_oos_prompt_budget(budget, where="--prompt-budget-off artifact")
    except C.OutOfSupportPromptBudgetError as e:
        raise TransferAnalysisError(str(e)) from e
    if doc["arm"] != C.THINKING_OFF:
        raise TransferAnalysisError(
            f"--prompt-budget-off artifact: arm {doc['arm']!r} — the comparator's artifact tokenizes the OFF "
            "render (the three-way prompt-max agreement on the OFF/8192 legs)")
    if doc["cap"] != OFFCAP_CAP:
        raise TransferAnalysisError(
            f"--prompt-budget-off artifact: cap {doc['cap']} != the comparator's OFF cap {OFFCAP_CAP} "
            "(contracts.AUTHORISED_OFF_CAP_COMPARATOR.max_tokens_off) — a table-cap OFF artifact does not "
            f"stand for the OFF/{OFFCAP_CAP} legs; the preflight writes it with --off-cap-override")
    if doc["cap_task"] != TASK:
        raise TransferAnalysisError(
            f"--prompt-budget-off artifact: cap_task {doc['cap_task']!r} != {TASK!r}")
    return doc


#: the two statistics the live tokenize pass stamps (``prompt_budget_live``)
#: and the CPU artifact carries under the SAME names — read and checked alike
LIVE_PROMPT_STATS: tuple[str, ...] = (C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_MAX,
                                      C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_P99)
def _artifact_not_supplied(flag: str) -> str:
    """The pending reason when no artifact rode ``flag`` (``--prompt-budget`` /
    ``--prompt-budget-off``) — one spelling for the reader and the tests."""
    return f"no {flag} artifact supplied"


ARTIFACT_NOT_SUPPLIED = _artifact_not_supplied("--prompt-budget")


def _artifact_stat(budget: Mapping[str, Any] | None, key: str, *, flag: str = "--prompt-budget"
                   ) -> tuple[int | None, str | None]:
    """The validated artifact's ``key`` (``max`` / ``p99``, its ONE spelling
    each — the contract validated the statistics against ``lengths``), or
    ``(None, why)`` when no artifact was supplied (``flag`` names the operator
    input that would supply it: the ON artifact's, or ``--prompt-budget-off``)."""
    if budget is None:
        return None, _artifact_not_supplied(flag)
    return int(budget[key]), None


def _artifact_prompt_max(budget: Mapping[str, Any] | None, *, flag: str = "--prompt-budget"
                         ) -> tuple[int | None, str | None]:
    return _artifact_stat(budget, C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_MAX, flag=flag)


class LiveMeasurement(NamedTuple):
    """One stamped statistic of the driver's live tokenize pass
    (``evidence.prompt_budget_live.<key>``, plan S3.8), read in the three
    states of :func:`evidence_state`: ``value`` is the integer when STAMPED and
    ``None`` otherwise — an off-contract raw value is NEVER carried as the
    metric; ``problem`` names the key and the value when OFF_CONTRACT."""

    state: str
    value: int | None
    problem: str | None


def live_measurement(block: Any, key: str) -> LiveMeasurement:
    state, raw = evidence_state(block, key, kind="int")
    problem = (off_contract_problem(C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK, key, raw,
                                    expected="a JSON integer (a token count)")
               if state == EVIDENCE_OFF_CONTRACT else None)
    return LiveMeasurement(state, raw if state == EVIDENCE_STAMPED else None, problem)


def live_measurements(block: Any) -> dict[str, LiveMeasurement]:
    """Both :data:`LIVE_PROMPT_STATS` of the live block, symmetrically."""
    return {key: live_measurement(block, key) for key in LIVE_PROMPT_STATS}


def ledger_confirmation(out_root: Path, *, ledger_name: str, task: str, arm: str,
                        summary: Mapping[str, Any], header: Mapping[str, Any],
                        max_model_len: int | None, live_prompt: Mapping[str, LiveMeasurement],
                        prompt_budget: Mapping[str, Any] | None, budgeted: bool,
                        artifact_flag: str = "--prompt-budget") -> dict[str, Any]:
    """The out-root's usage-ledger rows of ``task`` against the leg: call count ==
    attempted draws; one cap (the summary's) and one arm; every ``error_class``
    ``none`` and nothing rejected (no client clamp exists on this path — an
    overflow surfaces as ``context_length``) — a rejected row FAILS the leg by
    name as CENSORING (the client's retry re-sampled that item and the second
    draw is what was scored), and the summary's own ledger-derived
    ``n_http_rejected_retried`` must equal the count read here (absent on a
    store from a producer before the field: pending, a waivable producer
    stamp); ``count(finish_reason == length)
    == summary n_at_cap`` (a null ``n_at_cap`` FAILS, never coerced to 0);
    ``max(prompt_tokens_pre) + cap <= served max_model_len`` — on EVERY leg.
    On the ON legs ONLY (plan S5.6): the three-way prompt-max agreement ledger
    == live tokenize pass == the CPU artifact's ``max``, and the artifact's own
    fit at this leg's cap and window (``contracts.prompt_budget_fit``). The
    artifact tokenizes the ON render; an OFF prompt is a different length on
    the same messages (two tokens longer on this family), so comparing it on
    an OFF leg would fail a correctly executed experiment — an OFF leg records
    the comparison as not applicable, neither pending nor failed. A source this
    tree does not write yet is PENDING, named (``pending_stamps`` carries the
    producer stamp per entry, ``None`` for a missing operator input); a
    disagreement FAILS, and so does a live statistic that is PRESENT but not
    an integer (``live_prompt`` — :func:`live_measurements`: an unreadable
    measurement is a failed one, never pending). The live pass's ``max`` and
    ``p99`` are handled ALIKE: each absent / null is its own named producer
    stamp, each off-contract value its own unwaivable failure, and the ``p99``
    is compared with the artifact's ``p99`` on the ON legs — the same 200
    messages under the same render, so the two tokenizations must agree
    (``prompt_p99``, beside ``prompt_max``).
    ``budgeted`` names a leg held to an artifact — the ON legs, and the comparator's
    OFF/8192 legs too: their third source is the OFF
    artifact (``artifact_flag`` ``--prompt-budget-off``), and the fit is read at
    their cap; the OFF/512 legs keep the two-source check."""
    path = out_root / ledger_name
    failures: list[str] = []
    pending: list[str] = []
    pending_stamps: list[str | None] = []
    if not path.is_file():
        return {"available": False, "path": str(path), "passed": False,
                "failures": [f"no {ledger_name} at the out-root"], "pending": [], "pending_stamps": []}
    rows = _ledger_rows(path, task)
    expected_calls = header["n_attempted_samples"]
    n_calls = len(rows)
    if n_calls != expected_calls:
        failures.append(f"{n_calls} ledger calls for {task}, expected {expected_calls} "
                        "(k = 1, no retry on GSM8K)")
    caps = sorted({r.get("max_tokens_requested") for r in rows}, key=lambda v: (v is None, v))
    want_cap = summary.get("max_tokens")
    cap_ok = bool(rows) and caps == [want_cap]
    if rows and not cap_ok:
        failures.append(f"max_tokens_requested {caps} != the summary's cap {want_cap!r}")
    thinks = sorted({r.get("thinking") for r in rows}, key=lambda v: (v is None, bool(v)))
    want_think = C.enable_thinking(arm)
    think_ok = bool(rows) and thinks == [want_think]
    if rows and not think_ok:
        failures.append(f"thinking {thinks} != the arm's {want_think!r}")
    classes = Counter(str(r.get("error_class")) for r in rows)
    if rows and set(classes) != {C.LEDGER_ERROR_CLASS_NONE}:
        failures.append(f"error_class counts {dict(classes)} — every call must be "
                        f"{C.LEDGER_ERROR_CLASS_NONE!r}")
    n_rejected = sum(1 for r in rows if r.get("rejected") is True)
    if n_rejected:
        # A rejected call is one the upstream refused AFTER it was
        # attempted (at the lens proxy's upstream timeout: a cap-bound decode);
        # the client's retry re-drew the item at the card temperature and the
        # SECOND sample is what the summary scored — the leg's scored
        # population is not its first-draw population, named as censoring
        statuses = sorted({str(r.get("status")) for r in rows if r.get("rejected") is True})
        failures.append(
            f"censored: {n_rejected} rejected calls re-sampled by the client's retry (upstream HTTP "
            f"{', '.join(statuses)}) — each rejected draw was refused by the upstream after it was "
            "attempted (at the proxy's upstream timeout, a cap-bound decode) and the score on file is "
            "the client's SECOND sample of that item; the scored population is not the first-draw "
            "population (D73 (d))")
    # The summary's ledger-derived count must agree with the ledger read
    # here; a summary without the field is a producer before it (pending, the
    # ledger still decides censoring above); a disagreement names the split
    summary_rejected = summary.get(SUMMARY_REJECTED_FIELD, EVIDENCE_ABSENT)
    if summary_rejected is EVIDENCE_ABSENT:
        pending.append(f"summary {SUMMARY_REJECTED_FIELD} (summary.{SUMMARY_REJECTED_FIELD} — the Inspect "
                       "tier's ledger-derived rejected-call count, D73 (d); not stamped: a producer "
                       "before it)")
        pending_stamps.append(stamp_name("summary", SUMMARY_REJECTED_FIELD))
    elif summary_rejected != n_rejected:
        failures.append(f"summary {SUMMARY_REJECTED_FIELD} {summary_rejected!r} != the ledger's "
                        f"{n_rejected} rejected calls for {task} — the summary and the ledger disagree "
                        "on how many draws were re-sampled")
    n_length = sum(1 for r in rows if r.get("finish_reason") == "length")
    n_at_cap = summary.get("n_at_cap")
    if n_at_cap is None:
        length_agrees: bool | None = None
        failures.append(f"the summary's n_at_cap is null (no draw recorded a stop reason) while "
                        f"the ledger counts {n_length} finish_reason == length")
    else:
        length_agrees = n_length == n_at_cap
        if not length_agrees:
            failures.append(f"ledger finish_reason == length {n_length} != summary n_at_cap {n_at_cap}")
    pre = [r.get("prompt_tokens_pre") for r in rows]
    pre_ints = [p for p in pre if isinstance(p, int) and not isinstance(p, bool)]
    ledger_max = max(pre_ints) if pre_ints and len(pre_ints) == len(rows) else None
    if rows and ledger_max is None:
        failures.append("a ledger row carries no integer prompt_tokens_pre")
    window: dict[str, Any] = {"prompt_max": ledger_max, "cap": want_cap,
                              "max_model_len": max_model_len, "fits": None}
    if ledger_max is not None and isinstance(want_cap, int) and isinstance(max_model_len, int):
        window["fits"] = ledger_max + want_cap <= max_model_len
        if not window["fits"]:
            failures.append(f"max(prompt_tokens_pre) {ledger_max} + cap {want_cap} > served "
                            f"max_model_len {max_model_len}")
    elif rows:
        failures.append("the window check needs the ledger max, the summary cap and the manifest's "
                        "served max_model_len")
    MAX, P99 = C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_MAX, C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_P99
    live_max, live_p99 = live_prompt[MAX], live_prompt[P99]
    on_leg = bool(budgeted)
    three: dict[str, Any] = {"ledger": ledger_max, "live": live_max.value, "artifact": None,
                             "agrees": None, "detail": None, "applicable": on_leg}
    p99: dict[str, Any] = {"live": live_p99.value, "artifact": None, "agrees": None, "detail": None,
                           "applicable": on_leg}
    artifact_fit: dict[str, Any] | None = None
    if not on_leg:
        three["detail"] = p99["detail"] = (
            "not compared on an OFF leg: the CPU artifact measures the ON render, and the "
            "prompt-budget agreement is an ON-leg check (plan S5.6)")
    else:
        artifact_prompt_max, artifact_reason = _artifact_prompt_max(prompt_budget, flag=artifact_flag)
        artifact_p99, _same_reason = _artifact_stat(prompt_budget, P99, flag=artifact_flag)
        three["artifact"], p99["artifact"] = artifact_prompt_max, artifact_p99
        # the live pass's two statistics, alike: off-contract FAILS (unwaivable),
        # absent / null is that statistic's OWN producer stamp, pending
        for key, m in ((MAX, live_max), (P99, live_p99)):
            if m.state == EVIDENCE_OFF_CONTRACT:
                failures.append(f"live prompt {key} off-contract: {m.problem}")
            elif m.value is None:
                pending.append(f"live prompt {key} (evidence.{stamp_name(C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK, key)}"
                               " — the driver's live tokenize pass, plan S3.8, "
                               + ("stamped null: the pass did not run)" if m.state == EVIDENCE_NULL
                                  else "not stamped)"))
                pending_stamps.append(stamp_name(C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK, key))
        if prompt_budget is None:
            pending.append(f"CPU artifact prompt max ({artifact_reason})")
            pending_stamps.append(NOT_A_PRODUCER_STAMP)          # an operator input, never waived
        elif isinstance(want_cap, int) and isinstance(max_model_len, int):
            # the artifact's OWN fit at this leg's cap and window — the contract's
            # one gate, so the analysis and the preflight read the same rule
            try:
                artifact_fit = {"passed": True, **C.prompt_budget_fit(
                    prompt_budget, arm=arm, cap=want_cap, max_model_len=max_model_len,
                    where=f"the {artifact_flag} artifact against this leg")}
            except C.OutOfSupportPromptBudgetError as e:
                artifact_fit = {"passed": False, "detail": str(e)}
                failures.append(f"the CPU artifact does not stand for this leg: {e}")
        if ledger_max is not None and live_max.value is not None and artifact_prompt_max is not None:
            three["agrees"] = ledger_max == live_max.value == artifact_prompt_max
            if not three["agrees"]:
                three["detail"] = (f"ledger {ledger_max} / live {live_max.value} / artifact "
                                   f"{artifact_prompt_max} disagree")
                failures.append("three-way prompt-max disagreement: " + three["detail"])
        # the p99 beside the max: the live pass and the artifact tokenize the SAME
        # 200 messages under the ON render (the ledger has no p99 of its own)
        if live_p99.value is not None and artifact_p99 is not None:
            p99["agrees"] = live_p99.value == artifact_p99
            if not p99["agrees"]:
                p99["detail"] = f"live {live_p99.value} / artifact {artifact_p99} disagree"
                failures.append("live / artifact prompt-p99 disagreement: " + p99["detail"])
    return {"available": True, "path": str(path), "n_calls": n_calls, "artifact_fit": artifact_fit,
            "expected_calls": expected_calls, "calls_agree": n_calls == expected_calls,
            "max_tokens_requested": {"values": caps, "expected": want_cap, "agrees": cap_ok},
            "thinking": {"values": thinks, "expected": want_think, "agrees": think_ok},
            "error_class": dict(classes), "n_rejected": n_rejected,
            "summary_n_http_rejected_retried": (None if summary_rejected is EVIDENCE_ABSENT
                                                else summary_rejected),
            "n_length": n_length, "summary_n_at_cap": n_at_cap, "length_agrees": length_agrees,
            "window": window, "prompt_max": three, "prompt_p99": p99,
            "passed": not failures, "failures": failures, "pending": pending,
            "pending_stamps": pending_stamps}


# --------------------------------------------------------------------------- #
# the §3.4 regime triple per leg (optional at this tree — plan S3 stamps it)  #
# --------------------------------------------------------------------------- #

def regime_vocabulary(field: str) -> tuple[str, ...]:
    """The closed vocabulary of one stamp field, the contract's:
    ``salt_regime`` and ``render_regime`` are contract-FINAL literals
    (``contracts.SALT_REGIME_LITERALS`` / ``RENDER_REGIME_LITERALS``);
    ``resumed_policy`` is the contract's eight labels
    (``contracts.RESUMED_POLICY_LABELS``: the proxy's three attested policies
    plus its five non-values, the two the lens half added among them). Keyed on
    ``StampRegimes``' field names, never a second spelling of them."""
    salt, policy, render = C.EXPECTED_STAMP_TRIPLE_FIELDS
    if field == salt:
        return tuple(C.SALT_REGIME_LITERALS)
    if field == render:
        return tuple(C.RENDER_REGIME_LITERALS)
    if field == policy:
        return tuple(C.RESUMED_POLICY_LABELS)
    raise KeyError(f"{field!r} is not a stamp field ({C.EXPECTED_STAMP_TRIPLE_FIELDS})")


def render_regime_for_arm(arm: str) -> str:
    """The render regime a leg of ``arm`` MUST have been served under —
    decidable from the arm alone (contract §2.5.5 / §2.5.11, beside
    ``contracts.RENDER_REGIME_BY_ARM``, which is exactly this map): the harness
    always sends the ``enable_thinking`` kwarg, so an ON leg renders
    ``thinking_on`` and an OFF leg ``thinking_off``; ``unspecified`` is never
    an expected value."""
    return C.RENDER_REGIME_BY_ARM[C.thinking_mode(C.enable_thinking(arm))]


def regime_check(summary: Mapping[str, Any], *, lens: bool, arm: str) -> dict[str, Any]:
    """Each leg carries exactly ONE value of each stamp field, equal to its
    expected triple (``contracts.expected_stamp_regimes``). A PRESENT value is
    validated BEFORE any unresolved expectation is consulted: a value outside
    the field's contract vocabulary (:func:`regime_vocabulary`) is
    off-contract — an unwaivable FAIL naming key and value; a ``render_regime``
    off its arm's (:func:`render_regime_for_arm` — ``thinking_off`` on an ON
    leg, ``thinking_on`` on an OFF leg, ``unspecified`` anywhere) FAILS now,
    decidable from the arm alone; a value off a KNOWN expectation FAILS; MIXED
    (the leg's distinct-value count is not 1) FAILS.
    PENDING only when a stamp is absent / null ("not stamped yet": the driver's
    stamp check writes it) or is present, valid and arm-consistent
    but its expected member reads ``None`` ("not reconciled" — never on the
    shipped contracts module, which fills these from the contract; the branch
    stays so a mutated expectation can never read as a pass)."""
    expected = C.expected_stamp_regimes(lens=lens, arm=arm)._asdict()   # by name, never by position
    observed = {f: summary.get(f) for f in C.EXPECTED_STAMP_TRIPLE_FIELDS}
    counts = {f: summary.get(C.STAMP_REGIME_COUNT_FIELD[f]) for f in C.EXPECTED_STAMP_TRIPLE_FIELDS}
    render_field = C.EXPECTED_STAMP_TRIPLE_FIELDS[-1]
    failures, unreconciled, unstamped = [], [], []
    for f in C.EXPECTED_STAMP_TRIPLE_FIELDS:
        v = observed[f]
        if v is None:
            unstamped.append(f)                      # absent or explicit null: the producer's stamp
            continue
        vocab = regime_vocabulary(f)
        if not isinstance(v, str) or v not in vocab:
            failures.append(f"{f}: {v!r} ({type(v).__name__}) is off-contract — not in the vocabulary "
                            f"{list(vocab)}; an unreadable stamp is a failed one, never a pending one")
            continue
        if counts[f] != 1:
            failures.append(f"{f}: {counts[f]!r} distinct values over the leg (MIXED; must be 1)")
        if f == render_field and v != render_regime_for_arm(arm):
            failures.append(f"{f}: {v!r} on the {arm} arm — this arm renders "
                            f"{render_regime_for_arm(arm)!r} (the harness always sends the kwargs key; "
                            "decidable from the arm alone)")
            continue
        if expected[f] is None:
            unreconciled.append(f)
        elif v != expected[f]:
            failures.append(f"{f}: {v!r} != expected {expected[f]!r}")
    stamped = len(unstamped) < len(C.EXPECTED_STAMP_TRIPLE_FIELDS)
    status = GATE_FAIL if failures else (GATE_PENDING if (unstamped or unreconciled) else GATE_PASS)
    if failures:
        detail: str | None = "; ".join(failures)
    else:
        parts = []
        if unstamped:
            parts.append(f"not stamped yet: the summary carries no {unstamped} (the driver's stamp check, "
                         "plan S3, writes them)")
        if unreconciled:
            parts.append(f"expected member(s) {unreconciled} not reconciled (the contracts' expectation "
                         "reads None — a mutated module; B0 filled every contract constant)")
        detail = "; ".join(parts) or None
    return {"stamped": stamped, "expected": expected, "observed": observed if stamped else None,
            "counts": counts, "status": status, "detail": detail,
            "pending_stamps": ([stamp_name("summary", f) for f in (*unstamped, *unreconciled)]
                               if status == GATE_PENDING else [])}


# --------------------------------------------------------------------------- #
# collect_legs — the four stores, each source allowed to say only its own    #
# thing (plan S5.2)                                                           #
# --------------------------------------------------------------------------- #


_EVIDENCE_KINDS: Mapping[str, Callable[[Any], bool]] = {
    "bool": lambda x: isinstance(x, bool),
    "int": lambda x: isinstance(x, int) and not isinstance(x, bool),
    "arm": lambda x: isinstance(x, str) and x in C.THINKING_MODES,
}


def evidence_state(block: Any, key: str, *, kind: str) -> tuple[str, Any]:
    """``(state, value)`` of one stamped evidence key — the three states kept
    distinct: :data:`EVIDENCE_ABSENT` (the key is not there: the producer has
    not landed), :data:`EVIDENCE_NULL` (an explicit JSON null: the producer
    landed, the certification did not run), :data:`EVIDENCE_STAMPED` (a value
    of the contract's type — ``kind`` ``bool`` / ``int`` / ``arm``) or
    :data:`EVIDENCE_OFF_CONTRACT` (present, of another type: the string
    ``"false"``, an int where a boolean belongs, a list). Only the first two
    may read as pending; the last is a hard failure at the gate. A block that
    is present but not a JSON object puts every key of it off-contract."""
    if block is None:
        return EVIDENCE_ABSENT, None
    if not isinstance(block, Mapping):
        return EVIDENCE_OFF_CONTRACT, block
    if key not in block:
        return EVIDENCE_ABSENT, None
    v = block[key]
    if v is None:
        return EVIDENCE_NULL, None
    return (EVIDENCE_STAMPED if _EVIDENCE_KINDS[kind](v) else EVIDENCE_OFF_CONTRACT), v


def off_contract_problem(block_name: str, key: str, value: Any, *, expected: str) -> str:
    return (f"{stamp_name(block_name, key)} is {value!r} ({type(value).__name__}) — off-contract: "
            f"{expected} or null was expected; an unreadable stamp is a failed one, never a pending one")


def sampling_row_problem(summary: Mapping[str, Any]) -> str | None:
    """Why the leg's summary is NOT drawn under the experiment's fixed sampling
    row (:data:`FIXED_SAMPLING`) — ``None`` when it is. Exact equality of the
    three numbers, as the summary records them."""
    got = tuple(summary.get(f) for f in SAMPLING_FIELDS)
    want = (FIXED_SAMPLING.temperature, FIXED_SAMPLING.top_p, FIXED_SAMPLING.top_k)
    if got == want and all(type(g) is type(w) or isinstance(g, (int, float)) and not isinstance(g, bool)
                           for g, w in zip(got, want)):
        return None
    return (f"sampling row {dict(zip(SAMPLING_FIELDS, got))} != the experiment's fixed row "
            f"{dict(zip(SAMPLING_FIELDS, want))} (contracts.sampling_for({C.model_row(MODEL).model_key!r}, "
            f"{C.AUTHORISED_OOS_THINKING.sampling_mode!r})) — not this experiment: a leg drawn under "
            "another row is another instrument, never a control")


def expected_cap(arm: str, *, cap_override: bool = False) -> int:
    """The per-arm completion cap the experiment runs the paired-200 GSM8K at:
    ON = the tuple's raised cap (``contracts.oos_max_tokens_on``), OFF = the
    in-support table's (``inspect_served.MAX_TOKENS_OFF``, the ONE place it is
    defined; imported here, the eval stack is not otherwise loaded); under the
    comparator's ``cap_override`` OFF = :data:`OFFCAP_CAP`
    (the ON cap under the OFF render) — an OFF fact, refused on the ON arm."""
    name = TASK.removeprefix(C.INSPECT_TASK_PREFIX)
    if cap_override:
        if arm != C.THINKING_OFF:
            raise TransferAnalysisError(f"the OFF cap override is an OFF fact; requested on the {arm!r} arm")
        cap = C.off_cap_override_max_tokens(name)
        if cap is None:
            raise TransferAnalysisError(
                f"contracts.AUTHORISED_OFF_CAP_COMPARATOR names no OFF cap for {name!r}")
        return int(cap)
    if arm == C.THINKING_ON:
        cap = C.oos_max_tokens_on(name)
        if cap is None:
            raise TransferAnalysisError(f"contracts.AUTHORISED_OOS_THINKING names no ON cap for {name!r}")
        return int(cap)
    from sisteer.capeval.tiers import inspect_served as isv

    return int(isv.MAX_TOKENS_OFF[name])


def cap_problem(summary: Mapping[str, Any], arm: str, *, cap_override: bool = False) -> str | None:
    want = expected_cap(arm, cap_override=cap_override)
    got = summary.get("max_tokens")
    if got == want and isinstance(got, int) and not isinstance(got, bool):
        return None
    if cap_override:
        return (f"max_tokens {got!r} != the comparator's {arm} cap {want} "
                "(contracts.AUTHORISED_OFF_CAP_COMPARATOR.max_tokens_off) — not this comparator")
    return (f"max_tokens {got!r} != the experiment's {arm} cap {want} "
            f"({'contracts.AUTHORISED_OOS_THINKING.max_tokens_on' if arm == C.THINKING_ON else 'inspect_served.MAX_TOKENS_OFF'}) "
            "— not this experiment")


def expected_think_split_mode(arm: str) -> str:
    """The think-block split mode the experiment's scorer MUST have run under on
    ``arm`` — the contract's ONE derivation (``contracts.think_split_mode``) over
    the arm, the family's reasoning-parser pin (:data:`FAMILY_REASONING_PARSER`;
    Qwen3.5: none, so the thought stays in ``content``) and the authorised
    tuple's DECLARED render fact (``prompt_opens_think_on``): OFF ->
    ``record_only``, ON -> ``implicit_open``. Per ARM: the two legitimately
    differ, so this is never a cross-leg equality."""
    return C.think_split_mode(arm=arm, reasoning_parser=FAMILY_REASONING_PARSER,
                              prompt_opened_think=C.AUTHORISED_OOS_THINKING.prompt_opens_think_on)


def think_split_problem(summary: Mapping[str, Any], arm: str) -> str | None:
    """Why the leg was NOT scored under its arm's think-split mode — ``None``
    when it was. The summary's ``reasoning_parser`` must be the family's pin
    (the derivation's premise: a parser-separated leg is another instrument)
    and its ``think_split_mode`` the derived one; the outcomes header's copy is
    held equal to the summary's by
    ``completion.verify_outcomes_against_summary``. An ON leg scored
    ``record_only`` / ``explicit_tags`` under a prompt that opened the block
    may have read an intermediate reasoning number as the GSM8K answer —
    before this check it was displayed and admitted."""
    parser = summary.get("reasoning_parser")
    if parser != FAMILY_REASONING_PARSER:
        return (f"reasoning_parser {parser!r} != the family's pin {FAMILY_REASONING_PARSER!r} "
                f"(contracts.REASONING_PARSER_BY_FAMILY, family {MODEL_ROW.family!r}) — a differently "
                "split instrument, not this experiment")
    want = expected_think_split_mode(arm)
    got = summary.get("think_split_mode")
    if got == want:
        return None
    return (f"think_split_mode {got!r} != the {arm} arm's expected {want!r} "
            f"(contracts.think_split_mode(arm={arm!r}, reasoning_parser={FAMILY_REASONING_PARSER!r}, "
            f"prompt_opened_think={C.AUTHORISED_OOS_THINKING.prompt_opens_think_on!r})) — the completions "
            "were split under another rule, so the scored values are not this experiment's")


def expected_route_vmodel(cell: str, *, arm: str) -> str:
    """The COMPLETE virtual-model spelling the leg's route must wear —
    ``contracts.cell_spellings`` at the leg's door: the base route (``none``)
    is bare on either arm (``petri__base``); a lens cell is bare on the OFF arm
    and wears the DECLARED ``@thinking`` spelling on the ON arm (the
    out-of-support standup's cell spelling IS the declaration, contract
    §2.5.2). Raises ``contracts.OutOfSupportDeclarationUnreconciled`` should a
    declared-spelling constant read ``None`` (a mutated contracts module — the
    shipped one fills both from the contract): an ON lens leg's route identity
    is then UNRESOLVED — never accepted under either spelling. A
    variant-carrying id on the OFF arm has no spelling
    (``contracts.CellNameError``)."""
    declared = C.is_declared_configuration(cell, arm=arm, out_of_support=True)
    return C.cell_spellings(cell, declared=declared).vmodel


def identity_problem(manifest: Mapping[str, Any], *, cell: str, arm: str) -> str | None:
    """Why the store under a leg's out-root is NOT the authorised model's run of
    this leg's cell on this arm — ``None`` when it is. POSITIVE validation, per
    leg and before the cross-leg :func:`nuisance_check` (equality across the
    legs is not identity: four stores at another model agree with each other):
    the manifest's ``model_id`` / ``model_revision`` must be :data:`MODEL_ROW`'s
    ``hf`` / ``rev``; its ``cell_id`` (the TRUE id the driver records — the
    store key carries only the ``__`` -> ``--`` token) must be the requested
    cell; and its route's virtual model must be EXACTLY the contract's spelling
    of that cell on that arm (:func:`expected_route_vmodel`) — a spelling that
    cannot be derived yet is a named refusal, not a pass."""
    got_id, got_rev = manifest.get("model_id"), manifest.get("model_revision")
    if (got_id, got_rev) != (MODEL_ROW.hf, MODEL_ROW.rev):
        return (f"manifest model {got_id!r} @ {got_rev!r} != the authorised model's {MODEL_ROW.hf!r} @ "
                f"{MODEL_ROW.rev!r} (contracts.model_row({MODEL!r})) — another model's store, not this "
                "experiment")
    cell_id = manifest.get("cell_id")
    if cell_id != cell:
        return (f"manifest cell_id {cell_id!r} != the leg's cell {cell!r} — another cell's store under "
                "this leg's out-root")
    try:
        want_vmodel = expected_route_vmodel(cell, arm=arm)
    except C.OutOfSupportDeclarationUnreconciled as e:
        return (f"the {arm} leg's route identity is UNRESOLVED: the declared spelling of cell {cell!r} "
                f"cannot be derived yet ({e}) — refused from the estimand rather than accepted under "
                "either spelling")
    except (C.CellNameError, C.ServeDeclarationError) as e:
        return f"cell {cell!r} has no route spelling on the {arm} arm: {e}"
    vmodel = _mapping(manifest.get(C.MANIFEST_KEY_ROUTE)).get("virtual_model")
    if vmodel != want_vmodel:
        return (f"manifest route virtual_model {vmodel!r} != the contract's spelling of cell {cell!r} on "
                f"the {arm} arm ({want_vmodel!r}, contracts.cell_spellings) — another route under this "
                "leg's out-root")
    return None


def model_label(legs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The report's model label, read off the legs whose manifests VALIDATED
    against the registry row (:func:`identity_problem`) — never the tuple's
    slug alone, which the report once printed over four stores of another
    model. ``hf`` / ``revision`` are ``None`` with no validated leg."""
    validated = sorted(k for k, leg in legs.items() if leg.get("identity_validated") is True)
    return {"slug": MODEL if validated else None,
            "hf": MODEL_ROW.hf if validated else None, "revision": MODEL_ROW.rev if validated else None,
            "validated_legs": validated,
            "source": (f"manifest model_id / model_revision of {len(validated)} leg(s) validated against "
                       f"contracts.model_row({MODEL!r})" if validated else
                       f"no leg's manifest validated against contracts.model_row({MODEL!r}) — unlabelled")}


def collect_leg(gen_root: Path, sha: str, *, cell: str, arm: str,
                prompt_budget: Mapping[str, Any] | None = None,
                read_eval_logs: bool = True, cap_override: bool = False,
                out_suffix_prefix: str = OUT_SUFFIX_PREFIX) -> dict[str, Any]:
    """One leg's record. ``missing`` names a leg the run did not produce
    (no out-root, no store, no summary, no outcome record); ``refused`` names
    an inconsistency between sources (outcomes vs summary, rows vs outcomes,
    log vs outcomes, identity vs arm), a store that is not the authorised
    model's run of this cell (:func:`identity_problem`), a leg off the fixed
    row / cap / arm-specific think-split mode, or an INCOMPLETE task — a PARTIAL marker
    (a limited run), no done marker, or a population that is not the frozen
    paired-200 (``n != contracts.GSM8K_PAIRED_N``). Either stops the estimand;
    a refused leg still carries its summary for the controls table (partial
    diagnostics, verdict withheld). ``kind`` places the leg in the 2x2 through
    ``contracts.oos_cell_kind`` (a variant id through its base); ``lens`` is
    the cell grammar's verdict (``contracts.is_lens_cell``: a fitted object,
    never ``cell != "none"``).

    ``cap_override`` REQUESTS a comparator leg (the OFF/8192 store under
    ``offcap-<sha>``, ``out_suffix_prefix``); the record's
    ``cap_override`` is the VALIDATED comparator classification — True only
    when the leg was requested as one, its arm is OFF and its optional-key
    SHAPE validated (``max_tokens_by_task == {task: 8192}``, no
    ``out_of_support_thinking``, no ``serve_out_of_support``) — and False on
    every 2x2 leg whatever keys it carries (the frozen ON stores carry the cap
    key too). ``arm`` stays ``"off"``: the RENDER arm every per-arm check reads."""
    kind = C.oos_cell_kind(cell)
    root = out_root_of(gen_root, sha, cell=cell, arm=arm, prefix=out_suffix_prefix)
    leg: dict[str, Any] = {"cell": cell, "arm": arm, "kind": kind, "leg": leg_key(cell, arm, cap_override=cap_override),
                           "lens": C.is_lens_cell(cell), "cap_override": False, "out_root": str(root),
                           "missing": None, "refused": None}
    if cap_override and arm != C.THINKING_OFF:
        leg["refused"] = (f"a comparator leg is an OFF fact (the thinking-OFF render at cap {OFFCAP_CAP}); "
                          f"requested on the {arm!r} arm")
        return leg
    if not root.is_dir():
        leg["missing"] = f"no out-root at {root}"
        return leg
    cd, why = _cell_dir(root)
    if cd is None:
        leg["missing"] = why
        return leg
    leg["cell_dir"] = str(cd)
    leg["cell_key"] = cd.name
    manifest = completion.read_store_manifest(cd)
    if manifest is None:
        leg["refused"] = f"{cd / 'manifest.json'} is not a JSON object"
        return leg
    tinycap = _mapping(manifest.get(C.MANIFEST_KEY_TINYCAP))
    served = _mapping(tinycap.get("served"))
    route = _mapping(manifest.get(C.MANIFEST_KEY_ROUTE))
    producer = _mapping(manifest.get(C.MANIFEST_KEY_PRODUCER))
    provenance = _mapping(manifest.get(C.MANIFEST_KEY_PROVENANCE))
    leg["identity"] = {
        "cell_id": manifest.get("cell_id"), "harness_cell_id": manifest.get("harness_cell_id"),
        "virtual_model": route.get("virtual_model"),
        "model_id": manifest.get("model_id"), "model_revision": manifest.get("model_revision"),
        "intervention_hash": manifest.get("intervention_hash"),
        "harness_config_hash": manifest.get("harness_config_hash"),
        "producer_tree_sha": producer.get("tree_sha"),
        # M3: the fitted object's artifact digest (None on the base route, or unrecorded)
        "object_artifact_sha256": provenance.get(PROVENANCE_ARTIFACT_SHA256_FIELD),
        C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING: tinycap.get(C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING),
        C.IDENTITY_KEY_MAX_TOKENS_BY_TASK: tinycap.get(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK),
        C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT: tinycap.get(C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT),
        C.REDUCTION_MODE_FIELD: served.get(C.REDUCTION_MODE_FIELD, route.get(C.REDUCTION_MODE_FIELD)),
        "served": {k: served.get(k) for k in ("max_model_len", "max_num_seqs", "concurrency",
                                              "reasoning_parser")},
        "route_declarations": route.get(C.ROUTE_KEY_SERVE_DECLARATIONS),
        "thinking": tinycap.get("thinking"),
    }
    # the store is POSITIVELY the authorised model's run of THIS cell — its
    # model id / revision against the registry row, its cell identity against
    # the requested leg — before anything else is read off it (the cross-leg
    # nuisance_check can only say the four agree, not what they are)
    problem = identity_problem(manifest, cell=cell, arm=arm)
    leg["identity_validated"] = problem is None
    if problem:
        leg["refused"] = problem
        return leg
    oos = tinycap.get(C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING)
    if arm == C.THINKING_ON and oos is not True:
        leg["refused"] = (f"the ON leg's store is not positively marked "
                          f"{C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING}: true (got {oos!r}) — not "
                          "the authorised out-of-support store")
        return leg
    if arm == C.THINKING_OFF and cap_override:
        # the comparator's identity carries EXACTLY ONE optional key — the executed
        # cap — and neither of the out-of-support store's other two
        if C.IDENTITY_KEY_MAX_TOKENS_BY_TASK not in tinycap:
            leg["refused"] = (f"the comparator leg's store carries no {C.IDENTITY_KEY_MAX_TOKENS_BY_TASK} — not the "
                              f"OFF/{OFFCAP_CAP} comparator store (a table-cap store, or a producer before the override)")
            return leg
        caps_key = tinycap.get(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK)
        if caps_key != dict(OFFCAP_MAX_TOKENS_BY_TASK):
            leg["refused"] = (f"the comparator leg's store carries {C.IDENTITY_KEY_MAX_TOKENS_BY_TASK} = {caps_key!r} "
                              f"!= the comparator's {dict(OFFCAP_MAX_TOKENS_BY_TASK)!r} "
                              "(contracts.AUTHORISED_OFF_CAP_COMPARATOR.max_tokens_off) — not this comparator")
            return leg
        extra = [k for k in (C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING, C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT) if k in tinycap]
        if extra:
            leg["refused"] = (f"the comparator leg's store carries {extra} = {[tinycap[k] for k in extra]!r} — an "
                              f"out-of-support store, not the OFF/{OFFCAP_CAP} comparator (which carries "
                              f"{C.IDENTITY_KEY_MAX_TOKENS_BY_TASK} alone)")
            return leg
        leg["cap_override"] = True
    elif arm == C.THINKING_OFF:
        if C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING in tinycap:
            leg["refused"] = (f"the OFF leg's store carries {C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING} "
                              f"= {oos!r} — an in-support store carries no such key")
            return leg
        other = [k for k in (C.IDENTITY_KEY_MAX_TOKENS_BY_TASK, C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT) if k in tinycap]
        if other:
            leg["refused"] = (f"the OFF leg's store carries {other} = {[tinycap[k] for k in other]!r} — an in-support "
                              "store at the table's cap carries no optional identity key (a comparator store "
                              "reads under --offcap-sha)")
            return leg
    summary = completion.read_task_summary(cd, TASK)
    if summary is None:
        leg["missing"] = (f"no {C.SUMMARY_DIR}/{C.marker_filename(TASK)} under {cd} — the leg "
                          "filed no summary (it did not complete)")
        return leg
    leg["summary"] = summary
    if summary.get("thinking") != arm:
        leg["refused"] = f"summary thinking {summary.get('thinking')!r} != the leg's arm {arm!r}"
        return leg
    if summary.get("task") != TASK:
        leg["refused"] = f"summary task {summary.get('task')!r} != {TASK!r}"
        return leg
    if summary.get("item_set") != C.GSM8K_ITEM_SET_PAIRED:
        leg["refused"] = (f"summary item_set {summary.get('item_set')!r} != {C.GSM8K_ITEM_SET_PAIRED!r} "
                          "— not the paired-200 instrument")
        return leg
    # the experiment's FIXED parameters: the sampling row and this arm's cap —
    # a leg that was not drawn under them is not this experiment (refused from
    # the estimand; its summary stays for the controls table); and this ARM's
    # think-split mode: a leg split under another rule scored other values
    # (the header's copy is verified against the summary below)
    leg["think_split_mode_expected"] = expected_think_split_mode(arm)
    problem = (sampling_row_problem(summary) or cap_problem(summary, arm, cap_override=leg["cap_override"])
               or think_split_problem(summary, arm))
    if problem:
        leg["refused"] = problem
        return leg
    # the task must be DONE — a PARTIAL marker is a limited run (finish_task
    # writes the summary and the record for it too), never the complete experiment
    partial_path = completion.partial_marker_path(cd, TASK)
    done_path = completion.done_marker_path(cd, TASK)
    if partial_path.is_file():
        partial = _mapping(_read_json(partial_path))
        leg["partial"] = {"limits": partial.get("limits"), "n_items": partial.get("n_items"),
                          "n_expected_full": partial.get("n_expected_full")}
        leg["refused"] = (f"PARTIAL marker at {partial_path}: a limited run (limits "
                          f"{partial.get('limits')!r}, {partial.get('n_items')!r} of "
                          f"{partial.get('n_expected_full')!r} items) — not the complete experiment; "
                          "diagnostics only, no estimand")
        return leg
    if not done_path.is_file():
        leg["refused"] = (f"no done marker at {done_path} — the task did not complete in this store; "
                          "diagnostics only, no estimand")
        return leg
    leg["done_marker"] = _mapping(_read_json(done_path))
    try:
        outcomes = completion.read_task_outcomes(cd, TASK)
    except CompletionError as e:
        leg["refused"] = f"outcomes/{C.task_outcomes_filename(TASK)}: {e}"
        return leg
    if outcomes is None:
        leg["missing"] = (f"MISSING-OUTCOMES: no {C.TASK_OUTCOMES_DIR}/{C.task_outcomes_filename(TASK)} "
                          f"under {cd} — the all-attempted record is required for every leg")
        return leg
    try:
        completion.verify_outcomes_against_summary(outcomes, summary)
    except CompletionError as e:
        leg["refused"] = str(e)
        return leg
    header = {k: v for k, v in outcomes.items() if k != "items"}
    leg["outcomes_header"] = header
    k = int(header["samples_per_item"])
    if k != 1:
        leg["refused"] = (f"samples_per_item {k}: the estimand of plan §6.1 is defined at k = 1 "
                          "(one mask per item) — a k > 1 store is not this experiment's")
        return leg
    if header["n_attempted_items"] != C.GSM8K_PAIRED_N:
        leg["refused"] = (f"n_attempted_items {header['n_attempted_items']} != the frozen "
                          f"{C.GSM8K_PAIRED_N} (contracts.GSM8K_PAIRED_N) — not the paired-200 population")
        return leg
    items = {rec["item_id"]: rec for rec in outcomes["items"]}
    leg["items"] = items
    leg["attempted_ids"] = sorted(items)
    rows = _rows_for_task(cd, TASK)
    leg["rows_check"] = rows_cross_check(rows, items)
    if not leg["rows_check"]["agrees"]:
        leg["refused"] = f"rows.jsonl disagrees with the outcome record: {leg['rows_check']['detail']}"
        return leg
    leg["rows_path"] = str(cd / "rows.jsonl")
    leg["eval_check"] = (eval_cross_check(cd, TASK, header, outcomes["items"]) if read_eval_logs
                         else {"available": False, "reason": "log reading disabled", "agrees": None,
                               "text": text_signature(None, "log reading disabled"),
                               # the operator's explicit choice (--no-eval-logs): not a
                               # re-derivation that was owed and not run
                               "log_present": None})
    if leg["eval_check"]["available"] and not leg["eval_check"]["agrees"]:
        leg["refused"] = f"the .eval log re-derivation disagrees with the record: {leg['eval_check']['detail']}"
        return leg
    try:
        evidence = _mapping(_read_json(cd / "evidence.json"))
    except TransferAnalysisError as e:
        leg["refused"] = str(e)
        return leg
    # the evidence keys under their ONE spelling (contracts.OOS_EVIDENCE_*): the
    # driver's stamps and this reader share them. Every certification / live
    # measurement is read in THREE states (evidence_state): absent, null, or a
    # value of the contract's type — a present value of another type is
    # off-contract and fails at the gate, never "not stamped"
    hdr_raw = evidence.get(C.OOS_EVIDENCE_HEADER_BLOCK)
    bites_raw = evidence.get(C.OOS_EVIDENCE_BITES_BLOCK)
    live_raw = evidence.get(C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK)
    hdr, bites = _mapping(hdr_raw), _mapping(bites_raw)
    cert = {f: evidence_state(hdr_raw, f, kind="bool") for f in HEADER_CERTIFICATION_KEYS}
    pra_state, pra_value = evidence_state(bites_raw, C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM, kind="arm")
    # the live pass's max AND p99, read alike: the metric carried is the stamped
    # integer or None — an off-contract raw value is named, never serialised
    live = live_measurements(live_raw)
    leg["evidence"] = {
        "present": bool(evidence),
        "header": {C.OOS_EVIDENCE_HEADER_MOVES: hdr.get(C.OOS_EVIDENCE_HEADER_MOVES),
                   C.OOS_EVIDENCE_HEADER_IDS_BY_ARM: hdr.get(C.OOS_EVIDENCE_HEADER_IDS_BY_ARM),
                   C.OOS_EVIDENCE_HEADER_SOURCE: hdr.get(C.OOS_EVIDENCE_HEADER_SOURCE),
                   C.OOS_EVIDENCE_HEADER_VERIFIED: hdr.get(C.OOS_EVIDENCE_HEADER_VERIFIED),
                   **{f: cert[f][1] for f in HEADER_CERTIFICATION_KEYS},
                   C.OOS_EVIDENCE_ATTESTED_SWITCH: hdr.get(C.OOS_EVIDENCE_ATTESTED_SWITCH),
                   C.OOS_EVIDENCE_ATTESTED_MOVES: hdr.get(C.OOS_EVIDENCE_ATTESTED_MOVES)},
        "header_state": {f: cert[f][0] for f in HEADER_CERTIFICATION_KEYS},
        "bites": {**{f: bites.get(f) for f in (
            C.OOS_EVIDENCE_BITES_APPLICABLE, C.OOS_EVIDENCE_BITES_VERIFIED, C.OOS_EVIDENCE_BITES_TWIN,
            C.OOS_EVIDENCE_BITES_N_DIFFERING, C.OOS_EVIDENCE_BITES_ERROR)},
                  C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM: pra_value},
        "bites_state": {C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM: pra_state},
        C.OOS_EVIDENCE_PROMPT_BUDGET_LIVE_BLOCK: {
            **{f: m.value for f, m in live.items()},
            "state": {f: m.state for f, m in live.items()},
            "problem": {f: m.problem for f, m in live.items()}},
    }
    if leg["cap_override"]:
        # the base-route probe, read in the three states ONLY on a
        # validated comparator leg; the gate holds the `none` leg to it
        probe_raw = evidence.get(C.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED)
        probe_state, probe_ok = evidence_state(probe_raw, EVIDENCE_BASE_ROUTE_PROBE_OK, kind="bool")
        probe = _mapping(probe_raw)
        leg["evidence"][C.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED] = {
            "status": probe.get("status"), "code": probe.get("code"),
            EVIDENCE_BASE_ROUTE_PROBE_OK: probe_ok, "state": probe_state}
    ledger_name = Path(str(manifest.get(C.MANIFEST_KEY_USAGE_LEDGER) or C.LEDGER_FILENAME)).name
    leg["ledger"] = ledger_confirmation(
        root, ledger_name=ledger_name, task=TASK, arm=arm, summary=summary, header=header,
        max_model_len=leg["identity"]["served"].get("max_model_len"),
        live_prompt=live, prompt_budget=prompt_budget,
        budgeted=(arm == C.THINKING_ON or leg["cap_override"]),
        artifact_flag="--prompt-budget-off" if leg["cap_override"] else "--prompt-budget")
    leg["regimes"] = regime_check(summary, lens=leg["lens"], arm=arm)
    return leg


def collect_legs(gen_root: Path, sha: str, *, cells: Sequence[str] = CELLS,
                 arms: Sequence[str] = ARMS, prompt_budget: Mapping[str, Any] | None = None,
                 read_eval_logs: bool = True) -> dict[str, dict[str, Any]]:
    """The four legs, keyed by :func:`leg_key` (the FULL cell id and the arm).
    Replaces ``analyze_oos.collect`` (F4). A MISSING leg is a record, never
    dropped."""
    sha = sha.removeprefix(OUT_SUFFIX_PREFIX)
    return {leg_key(cell, arm): collect_leg(gen_root, sha, cell=cell, arm=arm,
                                           prompt_budget=prompt_budget, read_eval_logs=read_eval_logs)
            for cell in cells for arm in arms}


def collect_offcap_legs(gen_root: Path, sha: str, *, cells: Sequence[str] = CELLS,
                        prompt_budget_off: Mapping[str, Any] | None = None,
                        read_eval_logs: bool = True) -> dict[str, dict[str, Any]]:
    """The two comparator records — `none` / `psr` thinking-OFF
    at cap 8192 under ``<offcap gen root>/…/thinking_off/offcap-<sha>`` — keyed
    ``leg_key(cell, off, cap_override=True)``; the OFF artifact is their third
    prompt-max source. Kept in a dict of their own: the estimand consumers take
    explicit four-key mappings, never a six-leg dict."""
    sha = sha.removeprefix(OFFCAP_OUT_SUFFIX_PREFIX)
    return {leg_key(cell, C.THINKING_OFF, cap_override=True): collect_leg(
        gen_root, sha, cell=cell, arm=C.THINKING_OFF, prompt_budget=prompt_budget_off,
        read_eval_logs=read_eval_logs, cap_override=True, out_suffix_prefix=OFFCAP_OUT_SUFFIX_PREFIX)
        for cell in cells}


def nuisance_values(leg: Mapping[str, Any]) -> dict[str, Any] | None:
    """The leg's :data:`NUISANCE_FIELDS` values (``None`` for a leg without a
    summary — a MISSING leg has nothing to compare)."""
    if "summary" not in leg:
        return None
    s, ident = leg["summary"], _mapping(leg.get("identity"))
    served = _mapping(ident.get("served"))
    out: dict[str, Any] = {f: s.get(f) for f in NUISANCE_SUMMARY_FIELDS}
    out.update({"served.max_model_len": served.get("max_model_len"),
                "served.max_num_seqs": served.get("max_num_seqs"),
                "served.concurrency": served.get("concurrency"),
                C.REDUCTION_MODE_FIELD: ident.get(C.REDUCTION_MODE_FIELD),
                "model_id": ident.get("model_id"), "model_revision": ident.get("model_revision")})
    return out


def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True, default=str)


def nuisance_check(legs: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, str]]:
    """EVERY nuisance parameter (:data:`NUISANCE_FIELDS`) equal across the legs
    that carry a summary. Returns the per-field values by leg with the
    disagreeing fields named, and ``{leg key: reason}`` for every leg that is
    NOT this experiment: on a disagreement the legs off the strict majority
    value are refused; with no strict majority every leg carrying the field is
    (the four are one experiment or none of them is)."""
    per = {k: nuisance_values(leg) for k, leg in legs.items()}
    present = {k: v for k, v in per.items() if v is not None}
    fields: dict[str, Any] = {}
    refuse: dict[str, str] = {}
    for f in NUISANCE_FIELDS:
        by_leg = {k: v[f] for k, v in present.items()}
        counts = Counter(_canon(v) for v in by_leg.values())
        agree = len(counts) <= 1
        fields[f] = {"values": by_leg, "agree": agree}
        if agree:
            continue
        top, n_top = counts.most_common(1)[0]
        majority = top if n_top > len(by_leg) / 2 else None
        for k, v in by_leg.items():
            if majority is None or _canon(v) != majority:
                refuse.setdefault(k, (
                    f"nuisance parameter {f} = {v!r} disagrees across the legs "
                    f"({ {k2: v2 for k2, v2 in by_leg.items()} }) — not one experiment: the within-arm "
                    "design holds every nuisance constant, so this leg is not a control of it"))
    return {"fields": fields, "disagreeing": [f for f, r in fields.items() if not r["agree"]],
            "refused": dict(refuse)}, refuse


_NO_MAJORITY = object()


def object_identity_check(legs: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, str]]:
    """The fitted object's identity across the legs (the analysis review's
    M3): every leg that read a manifest is placed by its cell kind, and within
    a cell the :data:`OBJECT_IDENTITY_FIELD` digests must AGREE across its
    configurations — a lens cell's legs each carry one 64-hex digest, the base
    route's carry none. Refusals, by name: a lens leg whose digest is not a
    64-hex sha256 (unrecorded identity is never "agrees with nothing"); a
    base-route leg carrying a digest (a fitted object served on the base
    route); and, on a disagreement within a cell, the legs off the strict
    majority digest — with no strict majority every leg of the cell (the
    :func:`nuisance_check` convention: one experiment or none of it). Returns
    the record for the report and ``{leg key: reason}``."""
    by_leg: dict[str, Any] = {}
    refuse: dict[str, str] = {}
    cells: dict[str, dict[str, Any]] = {}
    for k, leg in legs.items():
        ident = leg.get("identity")
        if ident is None:
            continue
        digest = ident.get("object_artifact_sha256")
        by_leg[k] = digest
        lens = bool(leg.get("lens"))
        if lens and not C.is_sha256_hex(digest):
            refuse[k] = (f"fitted-object artifact digest {OBJECT_IDENTITY_FIELD} = {digest!r} is not a 64-hex sha256 on a "
                         f"lens leg — the object's identity is unrecorded, so this leg cannot be shown to be the same "
                         "fitted object as its configurations' (not admitted)")
        elif not lens and digest is not None:
            refuse[k] = (f"fitted-object artifact digest {OBJECT_IDENTITY_FIELD} = {digest!r} on the base-route leg — a "
                         "fitted object served on the base route is another experiment")
        cells.setdefault(str(leg.get("kind")), {"lens": lens, "legs": [], "digests": []})
        cells[str(leg.get("kind"))]["legs"].append(k)
    for kind, cell in cells.items():
        cell["legs"] = sorted(cell["legs"])
        values = {k: by_leg[k] for k in cell["legs"]}
        counts = Counter(values.values())
        cell["digests"] = sorted(counts, key=lambda v: (v is None, str(v)))
        cell["agree"] = len(counts) <= 1
        if cell["agree"]:
            continue
        top, n_top = counts.most_common(1)[0]
        # the base route's digest IS None, so "no strict majority" needs its own sentinel
        majority = top if n_top > len(values) / 2 else _NO_MAJORITY
        cell_id = KIND_CELL.get(kind, kind)
        for k, v in values.items():
            if majority is _NO_MAJORITY or v != majority:
                refuse.setdefault(k, (
                    f"fitted-object artifact digest {OBJECT_IDENTITY_FIELD} = {v!r} disagrees across the {cell_id!r} legs "
                    f"({values!r}) — another fitted object under the same cell name, not this experiment's"))
    return {"field": OBJECT_IDENTITY_FIELD, "by_leg": by_leg, "by_cell": cells,
            "agree": all(c["agree"] for c in cells.values()) and not refuse, "refused": dict(refuse),
            "note": ("the fitted object's identity is its artifact digest, compared within every cell across its "
                     "configurations; the intervention hash is not compared here — it legitimately differs on the "
                     "declared ON route")}, refuse


def unusable_legs(legs: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """``{leg key: reason}`` for every leg that cannot enter the estimand."""
    return {k: (leg["missing"] or leg["refused"]) for k, leg in legs.items()
            if leg.get("missing") or leg.get("refused")}


def common_attempted_ids(legs: Mapping[str, Mapping[str, Any]]) -> tuple[list[str] | None, str | None]:
    """The ONE attempted id set (identical across the four legs), or (None,
    the difference named)."""
    sets = {k: set(leg["attempted_ids"]) for k, leg in legs.items() if "attempted_ids" in leg}
    if len(sets) != len(legs):
        return None, "a leg has no attempted ids (missing or refused)"
    union = set.union(*sets.values())
    common = set.intersection(*sets.values())
    if union != common:
        detail = "; ".join(f"{k} lacks {sorted(union - s)[:5]}" for k, s in sets.items() if union - s)
        return None, f"the {_NUMBER_WORDS.get(len(sets), str(len(sets)))} attempted-id sets differ: {detail}"
    return sorted(common), None


def frozen_paired_ids_from_plan(repo_root: Path | None = None) -> list[str]:
    """The frozen paired-200 id set as the TASK PLAN names it — the ids the
    driver requested (``inspect_served.plan_task(gsm8k_paired200).sample_ids``,
    the same derivation the prompt-budget script renders its messages from).
    Needs ``inspect_ai`` and the pinned inputs; refuses when it cannot be built
    here rather than guessing."""
    try:
        from sisteer.capeval.tiers import inspect_served as isv
    except ImportError as e:                                  # pragma: no cover - venv-dependent
        raise TransferAnalysisError(f"the frozen paired-200 ids need inspect_served ({e})") from e
    kw = {"repo_root": repo_root} if repo_root is not None else {}
    plan = isv.plan_task(C.GSM8K_PAIRED_TASK_NAME, grader_model=None, thinking=C.THINKING_OFF, **kw)
    ids = sorted(str(i) for i in (plan.sample_ids or []))
    if len(ids) != C.GSM8K_PAIRED_N:
        raise TransferAnalysisError(
            f"the paired-200 plan names {len(ids)} ids, not the frozen {C.GSM8K_PAIRED_N}")
    return ids


def population_check(legs: Mapping[str, Mapping[str, Any]],
                     paired_ids: Sequence[str] | None) -> dict[str, Any]:
    """The population gate (ONE common set of the 200 paired ids,
    identical across the four legs): every leg usable (DONE, verified), the
    four attempted-id sets identical, ``n == contracts.GSM8K_PAIRED_N``, and
    — when the frozen set is supplied (``--paired-ids``) — the ids ARE that
    set. Without the frozen set the identity is PENDING, named (the estimand
    may run as diagnostics; the word is withheld — an operator input is not a
    producer stamp, so ``--allow-pending-gates`` never waives it); a mismatch
    FAILS and no estimand runs."""
    ids, why = common_attempted_ids(legs)
    rec: dict[str, Any] = {"n": len(ids) if ids is not None else None, "reason": why,
                           "frozen_set": None, "n_frozen": None}
    if ids is None:
        return _gate(GATE_FAIL, why, **rec)
    if len(ids) != C.GSM8K_PAIRED_N:
        detail = (f"the common attempted set has {len(ids)} ids, not the frozen {C.GSM8K_PAIRED_N} "
                  "(contracts.GSM8K_PAIRED_N)")
        return _gate(GATE_FAIL, detail, **{**rec, "reason": detail})
    if paired_ids is None:
        return _gate(GATE_PENDING,
                     f"n == {C.GSM8K_PAIRED_N} and the {_NUMBER_WORDS.get(len(legs), str(len(legs)))} legs agree, "
                     "but no frozen paired-200 id "
                     f"set was supplied to compare against (--paired-ids <json list> | "
                     f"{PAIRED_IDS_FROM_PLAN}); the population's identity is not verified",
                     **{**rec, "frozen_set": "not supplied"})
    want = {str(i) for i in paired_ids}
    rec["n_frozen"] = len(want)
    if len(want) != C.GSM8K_PAIRED_N:
        detail = f"the supplied frozen id set has {len(want)} ids, not {C.GSM8K_PAIRED_N}"
        return _gate(GATE_FAIL, detail, **{**rec, "frozen_set": "invalid"})
    have = set(ids)
    if have != want:
        detail = (f"the attempted ids are not the frozen paired-200 set: {len(have - want)} attempted "
                  f"but not frozen (first {sorted(have - want)[:3]}), {len(want - have)} frozen but not "
                  f"attempted (first {sorted(want - have)[:3]})")
        return _gate(GATE_FAIL, detail, **{**rec, "frozen_set": "mismatch"})
    return _gate(GATE_PASS, None, **{**rec, "frozen_set": "verified"})


# ---------------------------------------------------------------------------- #
# the estimand under both branches                                              #
# ---------------------------------------------------------------------------- #


def natural_mapping(legs: Mapping[str, Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    """The natural 2x2 mapping ``{(kind, arm): leg record}`` of the FOUR 2x2
    legs (MAP_ORIG). A comparator leg has no slot here (its arm is ``off`` like
    the OFF/512 leg's — the two would silently overwrite each other), and two
    legs claiming one slot refuse by name: the four-leg wrapper never sees six
    keys."""
    out: dict[tuple[str, str], Mapping[str, Any]] = {}
    for k, leg in legs.items():
        if leg.get("cap_override") is True:
            raise TransferAnalysisError(
                f"{k}: a comparator leg has no slot in the natural 2x2 mapping — it enters through the cap-relief / "
                "matched relabellings (offcap plan §6.3)")
        key = (leg["kind"], leg["arm"])
        if key in out:
            raise TransferAnalysisError(f"two legs claim the 2x2 slot {pr.leg_label(key)}: {out[key]['leg']} and {k}")
        out[key] = leg
    return out


def _branch_items(leg: Mapping[str, Any], branch: str, *, exclude_ids: Sequence[str] = ()
                  ) -> dict[str, tuple[int, float | None]]:
    """One leg's ``pairing`` input for one scoring branch: PRIMARY masks an
    unscored draw out (``(0, None)``); SCORE-AS-IS scores every draw, an
    unscored one as wrong (``(1, 0.0)``)."""
    if branch not in BRANCHES:
        raise TransferAnalysisError(f"branch must be one of {BRANCHES}, got {branch!r}")
    drop = set(exclude_ids)
    d: dict[str, tuple[int, float | None]] = {}
    for i, rec in leg["items"].items():
        if i in drop:
            continue
        if rec["scored"]:
            d[i] = (1, float(rec["value"]))
        elif branch == BRANCH_PRIMARY:
            d[i] = (0, None)
        else:
            d[i] = (1, 0.0)
    return d


def branch_legs_from(mapping: Mapping[tuple[str, str], Mapping[str, Any]], branch: str, *,
                     exclude_ids: Sequence[str] = ()) -> dict[tuple[str, str], dict[str, tuple[int, float | None]]]:
    """The ``pairing`` input for one scoring branch over an EXPLICIT mapping
    ``{pairing key: leg record}`` — the relabelling is the caller's (MAP_ORIG /
    MAP_CAP / MAP_MATCHED), never inferred from the legs."""
    return {key: _branch_items(leg, branch, exclude_ids=exclude_ids) for key, leg in mapping.items()}


def branch_legs(legs: Mapping[str, Mapping[str, Any]], branch: str, *,
                exclude_ids: Sequence[str] = ()) -> dict[tuple[str, str], dict[str, tuple[int, float | None]]]:
    """:func:`branch_legs_from` over the natural 2x2 mapping of ``legs``."""
    return branch_legs_from(natural_mapping(legs), branch, exclude_ids=exclude_ids)


def estimands(mapping: Mapping[tuple[str, str], Mapping[str, Any]], *, n_resamples: int = pr.DEFAULT_BOOTSTRAP,
              seed: int = pr.DEFAULT_BOOTSTRAP_SEED, ci: float = pr.DEFAULT_CI,
              exclude_ids: Sequence[str] = ()) -> dict[str, dict[str, Any]]:
    """``{branch: pairing.paired_bootstrap_conditional_2x2(...)}`` — the same
    function, both branches; nothing else changes between them. Takes a
    MAPPING ``{pairing key: leg record}`` (:func:`natural_mapping`, or one of
    the comparator's two relabellings), never a legs dict: exactly the four
    pairing keys, refused by name otherwise."""
    if not isinstance(mapping, Mapping) or set(mapping) != set(pr.LEG_KEYS):
        got = sorted(map(str, mapping)) if isinstance(mapping, Mapping) else type(mapping).__name__
        raise TransferAnalysisError(
            f"estimands takes a mapping over exactly the four pairing keys {[pr.leg_label(k) for k in pr.LEG_KEYS]}, "
            f"got {got}")
    return {b: pr.paired_bootstrap_conditional_2x2(branch_legs_from(mapping, b, exclude_ids=exclude_ids),
                                                   n_resamples=n_resamples, seed=seed, ci=ci)
            for b in BRANCHES}


def three_mappings(legs4: Mapping[str, Mapping[str, Any]], legs_off: Mapping[str, Mapping[str, Any]] | None = None
                   ) -> dict[str, dict[tuple[str, str], Mapping[str, Any]]]:
    """The fixed mappings every :func:`estimands` call takes:
    MAP_ORIG = the natural 2x2; with the two comparator records, MAP_CAP puts
    the OFF/8192 legs in the ``on`` slots (its shifts ARE cap relief, its
    ``effect_on`` IS effect_off_8192) and MAP_MATCHED puts them in the ``off``
    slots (its shifts ARE the matched shifts, its DiD the matched DiD)."""
    orig = natural_mapping(legs4)
    if legs_off is None:
        return {MAP_ORIG: orig}
    off, on = C.THINKING_OFF, C.THINKING_ON
    # the legs by CONFIGURATION; each relabelled mapping fills its two slots from
    # the table (:data:`SLOT_CONFIGURATIONS`) — ONE declaration of which
    # configuration a slot holds, for the estimand and the labels alike
    by_cfg = {CFG_OFF: {kind: orig[(kind, off)] for kind in pr.LEG_CELLS},
              CFG_ON: {kind: orig[(kind, on)] for kind in pr.LEG_CELLS},
              CFG_OFF_OVERRIDE: {kind: legs_off[leg_key(KIND_CELL[kind], off, cap_override=True)]
                                 for kind in pr.LEG_CELLS}}
    return {MAP_ORIG: orig,
            **{name: {(kind, arm): by_cfg[SLOT_CONFIGURATIONS[name][arm]][kind]
                      for arm in (off, on) for kind in pr.LEG_CELLS}
               for name in (MAP_CAP, MAP_MATCHED)}}


# --------------------------------------------------------------------------- #
# the verdict rule and its two decidability checks                           #
# --------------------------------------------------------------------------- #


def _sign(x: float) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def classify_word(e_off: float, lo: float, hi: float, e_on: float) -> str:
    """The verdict rule, one branch: ``[lo, hi]`` is effect_off's CI, its
    edges INCLUSIVE (``lo <= e_on <= hi`` transfers). Every operand is the ONE
    representation pairing returns — the correctly rounded float of an exact
    rational over the integer counts, for the points, the bound ends AND the
    replicate statistics the CI's order statistics are drawn from (the closing
    review's P2: a CI edge that was the float difference of two rounded
    accuracies, ``0.21999999999999997``, sat a hair below an exact ON effect of
    ``0.22`` and read "amplifies"; both are ``44/200`` as one float now, and the
    edge reads "transfers"). :func:`decidability_over_bounds` evaluates the
    same inclusive rule, through this function, on the same representation."""
    if lo <= 0.0 <= hi:
        return WORD_NOT_RESOLVED
    if e_on == 0.0:
        return WORD_COLLAPSE            # zero is not a flip; it is a collapse (0 is outside the CI here)
    if _sign(e_on) != _sign(e_off):
        return WORD_SIGN_FLIP
    if lo <= e_on <= hi:
        return WORD_TRANSFERS
    if abs(e_on) < abs(e_off):
        return WORD_COLLAPSE
    return WORD_AMPLIFIES


def _breakpoints_on(e_off: float, lo: float, hi: float) -> list[tuple[str, float]]:
    return [("the sign", 0.0), ("the OFF CI's lower edge", lo), ("the OFF CI's upper edge", hi),
            ("the collapse/amplify line", -abs(e_off)), ("the collapse/amplify line", abs(e_off))]


def decidability_over_bounds(*, off_ci: tuple[float, float], e_off_bound: tuple[float, float],
                             e_on_bound: tuple[float, float], eps: float = 1e-9) -> dict[str, Any]:
    """The rule re-evaluated over the identified set (P1-a): the rectangle
    ``e_off_bound x e_on_bound`` (a segment when the OFF bound is a point — no
    OFF censoring), at its corners and on both sides of every breakpoint of
    the rule inside it (``E_on in {0, lo, hi, -|E_off|, +|E_off|}`` and the
    matching ``E_off`` lines), since the classification is piecewise constant
    between breakpoints. ``decidable`` iff the word is the SAME everywhere;
    ``crossings`` names the boundaries the set crosses (for the doc). The
    corners and the ``±eps`` probes are classified through
    :func:`classify_word` itself, so the CI's edges are the same INCLUSIVE
    edges on the same representation (pairing's correctly rounded rationals):
    a point bound sitting exactly on ``hi`` is one corner, classified once,
    "transfers" — never a probe a hair past the edge."""
    lo, hi = off_ci
    fo_lo, fo_hi = e_off_bound
    on_lo, on_hi = e_on_bound
    if fo_lo > fo_hi or on_lo > on_hi:
        raise TransferAnalysisError("a bound's low end exceeds its high end")
    off_cands = {fo_lo, fo_hi}
    for b in (0.0, on_lo, -on_lo, on_hi, -on_hi):
        if fo_lo < b < fo_hi:
            off_cands.update({b - eps, b + eps})
    points: list[dict[str, Any]] = []
    crossings: list[str] = []
    for eo in sorted(off_cands):
        on_cands = {on_lo, on_hi}
        for name, b in _breakpoints_on(eo, lo, hi):
            if on_lo < b < on_hi:
                on_cands.update({b - eps, b + eps})
                if classify_word(eo, lo, hi, b - eps) != classify_word(eo, lo, hi, b + eps) \
                        and name not in crossings:
                    crossings.append(name)
        for en in sorted(on_cands):
            points.append({"e_off": eo, "e_on": en, "word": classify_word(eo, lo, hi, en)})
    if fo_lo < 0.0 < fo_hi and "the sign of the OFF effect" not in crossings:
        words_off = {p["word"] for p in points}
        if len(words_off) > 1:
            crossings.append("the sign of the OFF effect")
    words = sorted({p["word"] for p in points})
    return {"decidable": len(words) == 1, "words": words, "points": points, "crossings": crossings}


def verdict_from_bounds(e_off: float, ci: tuple[float, float],
                        points_by_branch: Mapping[str, Mapping[str, Any]],
                        bound: tuple[float, float], *,
                        e_off_bound: tuple[float, float] | None = None) -> dict[str, Any]:
    """The rule over the two branches. ``e_off`` / ``ci``: the PRIMARY branch's
    effect_off point and CI; ``points_by_branch = {branch: {"effect_off",
    "effect_on", "did"[, "ci": (lo, hi)]}}``; ``bound`` (and the optional
    ``e_off_bound``, default a point): the sharp identified interval of
    effect_on (effect_off). Returns the word per branch, ``agree_branches``
    (same word and no +/- sign split on any of the three contrasts),
    ``decidable_bounds`` (:func:`decidability_over_bounds`) and the verdict:
    the PRIMARY word iff both hold, else the named "not decidable" reason."""
    words: dict[str, str] = {}
    for b, pt in points_by_branch.items():
        lo, hi = tuple(pt.get("ci", ci))
        words[b] = classify_word(float(pt["effect_off"]), float(lo), float(hi), float(pt["effect_on"]))
    branches = list(points_by_branch)
    same_word = len(set(words.values())) == 1
    # sign agreement: the sign of each named contrast must be IDENTICAL
    # across the two branches, zero being its own value (a +1.5 pp primary DiD
    # beside a score-as-is DiD of exactly 0 is a disagreement, not a wildcard).
    # The exact sign rides in from pairing (`point_sign`, integer arithmetic);
    # a caller handing bare floats gets the float's sign.
    signs_by_contrast = {
        k: {b: int(points_by_branch[b].get("signs", {}).get(k, _sign(float(points_by_branch[b][k]))))
            for b in branches}
        for k in ("effect_off", "effect_on", "did")}
    disagreeing = [k for k, s in signs_by_contrast.items() if len(set(s.values())) > 1]
    same_signs = not disagreeing
    agree = same_word and same_signs
    dec = decidability_over_bounds(off_ci=tuple(ci), e_off_bound=e_off_bound or (e_off, e_off),
                                   e_on_bound=tuple(bound))
    word_primary = words.get(BRANCH_PRIMARY, next(iter(words.values())))
    if agree and dec["decidable"]:
        verdict, reason = word_primary, None
    elif not dec["decidable"]:
        verdict = NOT_DECIDABLE_CENSORING
        reason = ("the identified interval of effect_on crosses "
                  + (", ".join(dec["crossings"]) or "a classification boundary")
                  + f" — words over the set: {dec['words']}")
    else:
        verdict = NOT_DECIDABLE_BRANCHES
        parts = []
        if not same_word:
            parts.append(f"words {words}")
        if disagreeing:
            parts.append("the sign of " + ", ".join(
                f"{k} ({'/'.join(str(signs_by_contrast[k][b]) for b in branches)})" for k in disagreeing))
        reason = "the two branches disagree on " + "; ".join(parts)
    return {"word_by_branch": words, "agree_branches": agree, "same_word": same_word,
            "same_signs": same_signs, "signs_by_contrast": signs_by_contrast,
            "disagreeing_contrasts": disagreeing, "decidable_bounds": dec["decidable"],
            "decidability": dec, "verdict": verdict, "reason": reason}


def _pp(x: float) -> str:
    return f"{100 * x:+.2f} pp"


def _ci(s: Mapping[str, Any], lo: str = "ci_low", hi: str = "ci_high") -> str:
    return f"[{_pp(s[lo])}, {_pp(s[hi])}]"


def contrast_text(name: str, s: Mapping[str, Any], *, level: float) -> str:
    """One contrast as THREE labelled quantities: the point, its sampling
    percentile CI, and the censoring-sensitivity interval (the sample bound
    over the censored items' assignments) with the bound endpoints' own
    bootstrap CI — never a containment claim between the CI and the bound."""
    pct = f"{round(100 * level)} %"
    bound = (f"{BOUND_LABEL} {_ci(s, 'bound_low', 'bound_high')}"
             + (" — a point: no censored item on these legs" if s["bound_low"] == s["bound_high"]
                else f", its endpoints' bootstrap CI {_ci(s, 'bound_ci_low', 'bound_ci_high')}"))
    return f"{name} = {_pp(s['point'])} (point); sampling {pct} CI {_ci(s)}; {bound}"


def _mcnemar_text(m: Mapping[str, Mapping[str, Any]]) -> str:
    """Both branches' exact tests, each labelled for what it is: a
    complete-case exact sensitivity under that branch's scoring."""
    parts = []
    for b in BRANCHES:
        r = m.get(b)
        if not r:
            continue
        pv = "n/a (no discordant pair)" if r["p_value"] is None else f"{r['p_value']:.4g}"
        parts.append(f"{MCNEMAR_BRANCH_LABEL.format(branch=BRANCH_NAMES[b])}: p = {pv} "
                     f"(C→I {r['C->I']} / I→C {r['I->C']}, n = {r['n_items']})")
    return "; ".join(parts)


def one_sided_note(rec: Mapping[str, Any] | None) -> str | None:
    """The one-sided-discordance note for ONE branch's McNemar record, or
    ``None`` when the discordance is not one-sided. Says exactly what holds:
    every complete-case discordance runs one way, so an EMPIRICAL bootstrap of
    those pairs cannot generate a change the other way — and then what that
    means for the REGISTERED bootstrap, read off the contrast's ACTUAL masks:
    with both legs fully scored under the branch (``fully_scored``) the
    registered bootstrap is that empirical bootstrap and the constraint applies
    to its replicates (:data:`ONE_SIDED_NOTE_FULLY_SCORED`); with masks varying
    it is not, and the property does not bind them
    (:data:`ONE_SIDED_NOTE_REGISTERED`). A record that does not say
    (``fully_scored`` absent) gets the weaker statement. The replicates' actual
    signs stand beside the record (``replicate_signs``,
    :func:`attach_registered_replicate_signs`) and in the statement
    (:func:`replicate_signs_text`)."""
    if not rec or not rec.get("one_sided_discordance"):
        return None
    d, n = rec["n_discordant"], rec["n_items"]
    losses = rec["I->C"] == 0
    other = "an improvement" if losses else "a decline"
    if d == 1:
        head = f"The one complete-case discordance is a {'loss' if losses else 'gain'}"
    else:
        head = f"All {_NUMBER_WORDS.get(d, str(d))} complete-case discordances are {'losses' if losses else 'gains'}"
    tail = (ONE_SIDED_NOTE_FULLY_SCORED.format(n=n, d=d) if rec.get("fully_scored") is True
            else ONE_SIDED_NOTE_REGISTERED)
    return f"{head}, so an empirical bootstrap of those {n} pairs cannot generate {other}; {tail}"


def first_one_sided_note(m: Mapping[str, Mapping[str, Any]] | None) -> str | None:
    """The note of the first branch (PRIMARY first) whose discordance runs one way."""
    for b in BRANCHES:
        note = one_sided_note((m or {}).get(b))
        if note:
            return note
    return None


def replicate_signs_text(est: Mapping[str, Mapping[str, Any]], contrast: str = "effect_on") -> str:
    """Where the registered bootstrap's replicates of ``contrast`` actually
    fell — negative / zero / positive counts per branch, read from
    ``pairing``'s ``replicate_signs`` (computed from its own replicate
    arrays, exact zeros) — the fact that stands beside the one-sided note."""
    parts = []
    for b in BRANCHES:
        rs = est[b]["stats"][contrast]["replicate_signs"]
        parts.append(f"{BRANCH_NAMES[b]}: {rs['negative']:,} negative / {rs['zero']:,} zero / "
                     f"{rs['positive']:,} positive of {est[b]['n_replicates_used']:,}")
    return f"registered bootstrap replicate signs for {contrast} — " + "; ".join(parts)


def verdict(est: Mapping[str, Mapping[str, Any]], *, mcnemar: Mapping[str, Any] | None = None,
            transitions: Mapping[str, Any] | None = None,
            caps: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The verdict block from the two branches' estimands: the rule per branch,
    the two decidability checks, the sign decisions for the shifts and the DiD
    (interval excludes 0 -> filed; else "sign not identified (censoring)"),
    and the statement the doc quotes. Gates are applied by the caller
    (:func:`instrumentation_gates`), which may withhold the word. ``mcnemar``
    (:func:`all_mcnemar`) puts the exact tests beside effect_on as sensitivity
    checks; ``transitions`` (:func:`all_transitions`) puts the OFF discordances
    beside effect_off; ``caps`` (:func:`arm_caps`) names the two configurations
    every cross-arm contrast compares."""
    p = est[BRANCH_PRIMARY]["stats"]
    pts = {b: {"effect_off": est[b]["stats"]["effect_off"]["point"],
               "effect_on": est[b]["stats"]["effect_on"]["point"],
               "did": est[b]["stats"]["did"]["point"],
               # the EXACT signs (pairing's integer arithmetic), never re-derived from the floats
               "signs": {k: est[b]["stats"][k]["point_sign"] for k in ("effect_off", "effect_on", "did")},
               "ci": (est[b]["stats"]["effect_off"]["ci_low"], est[b]["stats"]["effect_off"]["ci_high"])}
           for b in BRANCHES}
    v = verdict_from_bounds(
        p["effect_off"]["point"], (p["effect_off"]["ci_low"], p["effect_off"]["ci_high"]), pts,
        (p["effect_on"]["bound_low"], p["effect_on"]["bound_high"]),
        e_off_bound=(p["effect_off"]["bound_low"], p["effect_off"]["bound_high"]))
    signs = {}
    for c in ("shift_none", "shift_psr", "did"):
        s = p[c]
        signs[c] = {"sign": ("+" if s["bound_low"] > 0 else "-") if s["sign_identified"] else None,
                    "identified": s["sign_identified"],
                    "identified_meaning": SIGN_IDENTIFIED_MEANING,
                    # each of the three is an OFF/<cap> vs ON/<cap> contrast
                    "contrast": f"{RENDER_CAP_CONTRAST_LABEL} ({render_cap_configurations(caps)})",
                    "reading": (("+" if s["bound_low"] > 0 else "-") if s["sign_identified"]
                                else SIGN_NOT_IDENTIFIED),
                    "point": s["point"], "ci": [s["ci_low"], s["ci_high"]],
                    "bound": [s["bound_low"], s["bound_high"]],
                    "bound_endpoints_ci": [s["bound_ci_low"], s["bound_ci_high"]]}
    level = float(est[BRANCH_PRIMARY].get("level", pr.DEFAULT_CI))
    eo, en, dd = p["effect_off"], p["effect_on"], p["did"]
    # effect_off: an unresolved effect is said so, with the OFF discordances
    # beside it — never "free" / "costs nothing"
    off = contrast_text("effect_off", eo, level=level)
    if eo["ci_low"] <= 0.0 <= eo["ci_high"]:
        off += f" — {OFF_NOT_RESOLVED_TEXT}"
    t_off = (transitions or {}).get(f"none_to_psr__{C.THINKING_OFF}")
    if t_off:
        cnt = t_off["counts"]
        off += f" (OFF discordances none→psr: C→I {cnt['C->I']} / I→C {cnt['I->C']})"
    # effect_on: the exact tests as sensitivity checks, both branches
    on = contrast_text("effect_on", en, level=level)
    m_on = (mcnemar or {}).get(f"none_to_psr__{C.THINKING_ON}")
    if m_on:
        on += f"; exact two-sided McNemar: {_mcnemar_text(m_on)}"
        note = first_one_sided_note(m_on)
        if note:
            # the note says what an EMPIRICAL bootstrap of the complete-case
            # pairs could not do; the registered bootstrap's replicates stand beside it
            on += f" — {note} ({replicate_signs_text(est)})"
    # the DiD is a render+cap contrast too: what it would show, if nonzero, is
    # an interaction with the combined configuration, never "ON-specific"
    did = (contrast_text("DiD", dd, level=level)
           + f" ({RENDER_CAP_CONTRAST_LABEL}: effect_on at {arm_cap_label(C.THINKING_ON, caps)} − "
             f"effect_off at {arm_cap_label(C.THINKING_OFF, caps)})")
    if dd["ci_low"] <= 0.0 <= dd["ci_high"]:
        did += f" — the DiD CI includes zero: no {did_interaction_text(caps)} is established"
    statement = (f"{v['verdict']}: {off}; {on}; {did}"
                 + (f" — {v['reason']}" if v["reason"] else ""))
    return {**v, "signs": signs, "statement": statement}


def verdict_sensitivity_off_8192(ests: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """The transfer rule re-run with ``effect_off_8192`` as E_off (its CI, its
    point bound) and ``effect_on`` as E_on, both branches, with the two
    decidability checks — a SENSITIVITY (the rule's registered input stays the
    512 leg), never the headline. Reads the two relabelled blocks:
    ``effect_off_8192`` is MAP_CAP's ``effect_on`` (equal to MAP_MATCHED's
    ``effect_off``, asserted by :func:`offcap_consistency`); ``effect_on`` and
    ``did_matched`` are MAP_MATCHED's ``effect_on`` / ``did``."""
    cap, matched = ests[MAP_CAP], ests[MAP_MATCHED]
    p8 = cap[BRANCH_PRIMARY]["stats"]["effect_on"]
    pts = {b: {"effect_off": cap[b]["stats"]["effect_on"]["point"],
               "effect_on": matched[b]["stats"]["effect_on"]["point"],
               "did": matched[b]["stats"]["did"]["point"],
               "signs": {"effect_off": cap[b]["stats"]["effect_on"]["point_sign"],
                         "effect_on": matched[b]["stats"]["effect_on"]["point_sign"],
                         "did": matched[b]["stats"]["did"]["point_sign"]},
               "ci": (cap[b]["stats"]["effect_on"]["ci_low"], cap[b]["stats"]["effect_on"]["ci_high"])}
           for b in BRANCHES}
    e_on = matched[BRANCH_PRIMARY]["stats"]["effect_on"]
    v = verdict_from_bounds(p8["point"], (p8["ci_low"], p8["ci_high"]), pts,
                            (e_on["bound_low"], e_on["bound_high"]),
                            e_off_bound=(p8["bound_low"], p8["bound_high"]))
    return {"label": VERDICT_SENSITIVITY_8192_LABEL, "headline": False,
            "inputs": {"e_off": "effect_off_8192", "e_on": "effect_on", "did": "did_matched"},
            **v, "word": v["verdict"] if v["verdict"] in WORDS else None}


# --------------------------------------------------------------------------- #
# instrumentation gates (these, and ONLY these, invalidate an arm)             #
# --------------------------------------------------------------------------- #


def _gate(status: str, detail: str | None = None, **more: Any) -> dict[str, Any]:
    return {"status": status, "detail": detail, **more}


def cap_gate(legs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Gate 1: the SUMMARY's ``at_cap_rate`` on each ON leg (never a row-derived
    rate — the P1-b fixture reads 2.5 % here where rows read 0 %) and, with the
    same 2 % threshold, on each OFF/8192 comparator leg (they ARE the
    adequate-budget legs; the OFF/512 legs stay ungated — their at-cap rate is
    the thing being relieved). A null rate (no draw recorded a stop reason)
    cannot clear the gate."""
    per: dict[str, Any] = {}
    failures = []
    for k, leg in legs.items():
        comparator = leg.get("cap_override") is True
        if (leg["arm"] != C.THINKING_ON and not comparator) or "summary" not in leg:
            continue
        s = leg["summary"]
        rate = s.get("at_cap_rate")
        rec = {"at_cap_rate": rate, "n_at_cap": s.get("n_at_cap"), "n_stop_reason": s.get("n_stop_reason"),
               "threshold": CAP_GATE_AT_CAP_RATE}
        if rate is None:
            rec["status"] = GATE_FAIL
            failures.append(f"{k}: at_cap_rate is null (no draw recorded a stop reason)")
        elif rate > CAP_GATE_AT_CAP_RATE:
            rec["status"] = GATE_FAIL
            failures.append(f"{k}: at_cap_rate {rate:.3%} > {CAP_GATE_AT_CAP_RATE:.0%} — "
                            + (f"the OFF budget was not adequate at {s.get('max_tokens')}; not a capability reading "
                               "at this cap" if comparator else
                               "cap too small; not a capability reading; re-cost at a re-measured geometry"))
        else:
            rec["status"] = GATE_PASS
        per[k] = rec
    return _gate(GATE_FAIL if failures else GATE_PASS, "; ".join(failures) or None, legs=per)


def outcomes_gate(legs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Gate 2: every leg present with a verified outcome record, the four
    attempted-id sets identical, OFF legs fully scored (§6.1 (ii)).

    PENDING, named per leg, when a retained ``.eval`` log is PRESENT under a
    store but the independent re-derivation did not run (the reader venv
    cannot import inspect_ai, or the log could not be read): the record then
    stands unverified by its log, which is not a pass. An ABSENT log is
    "the record stands alone" (nothing owed, reported on the leg, not
    pending); ``--no-eval-logs`` is the operator's explicit choice (not
    pending). The entry is not a producer stamp, so ``--allow-pending-gates``
    does not waive it: run the analysis where the log can be read."""
    problems = dict(unusable_legs(legs))
    pending: list[str] = []
    if not problems:
        _ids, why = common_attempted_ids(legs)
        if why:
            problems["attempted_ids"] = why
        for k, leg in legs.items():
            if leg["arm"] == C.THINKING_OFF and leg["summary"].get("n_unscored") != 0:
                problems[k] = (f"the OFF leg has n_unscored = {leg['summary'].get('n_unscored')!r} "
                               "(no think block under record_only; match always returns a value)")
            ec = leg.get("eval_check") or {}
            if ec.get("available") is False and ec.get("log_present") is True:
                pending.append(f"{k}: independent .eval re-derivation not run: {ec.get('reason')}")
    status = GATE_FAIL if problems else (GATE_PENDING if pending else GATE_PASS)
    detail = ("; ".join(f"{k}: {v}" for k, v in problems.items()) if problems
              else ("; ".join(pending) or None))
    return _gate(status, detail, problems=problems, pending=pending,
                 pending_stamps=[NOT_A_PRODUCER_STAMP] * len(pending))


def anomaly_ids(legs: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    """Per leg: items whose think block was unterminated but did NOT stop at the
    cap (the model ended its turn inside its thoughts), items unscored without
    an unterminated block (unscored for some other reason), and items carrying
    a STRAY open tag (a literal ``<think>`` inside a completion whose block the
    prompt opened — the record's ``stray_open_tag``; such an item may well be
    SCORED, and a scored anomaly can move the word)."""
    out: dict[str, list[str]] = {}
    for k, leg in legs.items():
        ids = []
        for i, rec in leg.get("items", {}).items():
            if (rec["think_block_unterminated"] and not rec["at_cap"]) or \
                    (not rec["scored"] and not rec["think_block_unterminated"]) or \
                    rec["stray_open_tag"]:
                ids.append(i)
        out[k] = sorted(ids)
    return out


def anomaly_counts(legs: Mapping[str, Mapping[str, Any]], ids_by_leg: Mapping[str, Sequence[str]]
                   ) -> dict[str, dict[str, Any]]:
    """Per leg, the counts the gate reports beside its verdict: the summary's
    ``n_unterminated_not_at_cap``, the record's stray-tag count (and how many
    of those were scored), and the anomalous item total."""
    out: dict[str, dict[str, Any]] = {}
    for k, leg in legs.items():
        if "summary" not in leg:
            continue
        items = leg.get("items", {})
        stray = [i for i, rec in items.items() if rec["stray_open_tag"]]
        out[k] = {"n_unterminated_not_at_cap": leg["summary"].get("n_unterminated_not_at_cap"),
                  "n_stray_open_tag": len(stray),
                  "n_stray_open_tag_scored": sum(1 for i in stray if items[i]["scored"]),
                  "n_anomalous_items": len(ids_by_leg.get(k, []))}
    return out


def split_anomaly_gate(legs: Mapping[str, Mapping[str, Any]], est: Mapping[str, Mapping[str, Any]], *,
                       n_resamples: int, seed: int, ci: float) -> dict[str, Any]:
    """Gate 3: split anomalies (an unterminated block not at the cap, an
    unscored item without one, a stray open tag — :func:`anomaly_ids`) are
    REPORTED with their counts; they invalidate only if the affected items
    change the verdict word under either branch — the rule is re-run with
    those items masked out of every leg. ``legs`` are the FOUR 2x2 legs, re-run
    through their natural 2x2 mapping (the registered rule runs on MAP_ORIG;
    the comparator's re-estimation is :func:`offcap_split_anomaly_gate`)."""
    ids_by_leg = anomaly_ids(legs)
    counts = anomaly_counts(legs, ids_by_leg)
    drop = sorted(set().union(*ids_by_leg.values())) if ids_by_leg else []
    if not drop:
        return _gate(GATE_PASS, None, counts=counts, masked_out=[], word_moves=False)
    before = verdict(est)
    try:
        after = verdict(estimands(natural_mapping(legs), n_resamples=n_resamples, seed=seed, ci=ci,
                                  exclude_ids=drop))
    except pr.PairingError as e:
        return _gate(GATE_FAIL, f"re-running the rule without {len(drop)} anomalous items failed: {e}",
                     counts=counts, masked_out=drop, word_moves=None)
    moves = {b: (before["word_by_branch"][b], after["word_by_branch"][b]) for b in BRANCHES
             if before["word_by_branch"][b] != after["word_by_branch"][b]}
    moves_verdict = before["verdict"] != after["verdict"]
    moved = bool(moves) or moves_verdict
    return _gate(GATE_FAIL if moved else GATE_PASS,
                 (f"{len(drop)} anomalous items change the word: {moves or {'verdict': (before['verdict'], after['verdict'])}} "
                  "— the arm is not measurable" if moved else
                  f"{len(drop)} anomalous items reported; the word does not move without them"),
                 counts=counts, masked_out=drop, word_moves=moved,
                 word_without={b: after["word_by_branch"][b] for b in BRANCHES},
                 verdict_without=after["verdict"])


def _resolved(stat: Mapping[str, Any]) -> bool:
    """The first-branch reading effect_off gets: the percentile CI excludes zero."""
    return not (stat["ci_low"] <= 0.0 <= stat["ci_high"])


def offcap_anomaly_ids(legs6: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    """The comparator's anomalous items: :func:`anomaly_ids`' three classes over
    all six legs PLUS, on every thinking-OFF leg, each item whose record says a
    think block was PRESENT. Under ``record_only`` no prompt opens a block, so
    a completion that emits a CLOSED ``<think>…</think>`` of its own is recorded
    ``think_block_present True, think_block_unterminated False, stray_open_tag
    False`` (``think_split.split_metadata``) — terminated, no stray tag, and
    usually SCORED, so the legacy predicate never counts it (under
    ``implicit_open`` every ON item has a block, which is why it cannot). The
    gate reported the summary count of such items while leaving them in both
    re-estimated mappings (the code review's A1); they are anomalous ids here,
    masked with the rest. :func:`anomaly_ids` itself is untouched: the
    registered four-leg rule (:func:`split_anomaly_gate`) reads it byte for byte."""
    out = anomaly_ids(legs6)
    for k, leg in legs6.items():
        if leg.get("arm") != C.THINKING_OFF:
            continue
        present = {i for i, rec in leg.get("items", {}).items() if rec["think_block_present"]}
        if present:
            out[k] = sorted(set(out.get(k, ())) | present)
    return out


def offcap_split_anomaly_gate(legs6: Mapping[str, Mapping[str, Any]],
                              ests: Mapping[str, Mapping[str, Mapping[str, Any]]],
                              maps: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
                              cfgs: Mapping[str, Mapping[str, Any]], *,
                              n_resamples: int, seed: int, ci: float) -> dict[str, Any]:
    """The comparator's anomaly re-estimation:
    the anomaly ids over all SIX legs (:func:`offcap_anomaly_ids` — the legacy
    classes plus every present think block on an OFF leg), the two relabelled
    mappings re-run with the union masked out, and whether any comparator
    contrast's SIGN or the ``effect_off_8192`` resolved reading moves —
    reported, never a verdict word. Each OFF leg's summary count of present
    blocks (``n_think_block_present``, expected 0 under ``record_only``) is
    reported beside the item counts."""
    ids_by_leg = offcap_anomaly_ids(legs6)
    counts = anomaly_counts(legs6, ids_by_leg)
    for k, leg in legs6.items():
        if k in counts and leg["arm"] == C.THINKING_OFF:
            counts[k]["n_think_block_present"] = leg["summary"].get("n_think_block_present")
    drop = sorted(set().union(*ids_by_leg.values())) if ids_by_leg else []
    if not drop:
        return _gate(GATE_PASS, None, counts=counts, masked_out=[], sign_moves={}, resolved_moves=False, without=None)
    before = {name: rename_contrasts(ests[name], name, cfgs) for name in (MAP_CAP, MAP_MATCHED)}
    try:
        after = {name: rename_contrasts(estimands(maps[name], n_resamples=n_resamples, seed=seed, ci=ci,
                                                  exclude_ids=drop), name, cfgs)
                 for name in (MAP_CAP, MAP_MATCHED)}
    except pr.PairingError as e:
        return _gate(GATE_FAIL, f"re-running the comparator contrasts without {len(drop)} anomalous items failed: {e}",
                     counts=counts, masked_out=drop, sign_moves=None, resolved_moves=None, without=None)
    sign_moves: dict[str, tuple[int, int]] = {}
    for name in (MAP_CAP, MAP_MATCHED):
        for b in BRANCHES:
            for c, s in after[name][b]["stats"].items():
                was = before[name][b]["stats"][c]["point_sign"]
                if c in OFFCAP_CONTRASTS and s["point_sign"] != was:
                    sign_moves[f"{b}/{c}"] = (was, s["point_sign"])
    res_before = _resolved(before[MAP_CAP][BRANCH_PRIMARY]["stats"]["effect_off_8192"])
    res_after = _resolved(after[MAP_CAP][BRANCH_PRIMARY]["stats"]["effect_off_8192"])
    resolved_moves = res_before != res_after
    moved = bool(sign_moves) or resolved_moves
    without = {name: {b: {c: {f: s[f] for f in ("point", "ci_low", "ci_high", "point_sign")}
                          for c, s in after[name][b]["stats"].items()} for b in BRANCHES}
               for name in (MAP_CAP, MAP_MATCHED)}
    detail = (f"{len(drop)} anomalous items move a comparator contrast: signs {sign_moves}"
              + (f"; effect_off_8192 resolved {res_before} -> {res_after}" if resolved_moves else "")
              if moved else f"{len(drop)} anomalous items reported; no comparator contrast's sign or the "
                            "effect_off_8192 reading moves without them")
    return _gate(GATE_FAIL if moved else GATE_PASS, detail, counts=counts, masked_out=drop, sign_moves=sign_moves,
                 resolved_moves=resolved_moves, without=without)


def declaration_gate(legs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Gate 4: header / render-fact certification on the ON legs, the bites
    probe on the psr legs (and its render arm on psr/on), the ledger
    confirmation, the regime triple. A stamp this tree does not write is
    PENDING (named), a stamped ``False`` or a disagreement FAILS. The REC
    stream is not read here: a refusal REC ends the run before any store is
    complete, so four complete stores imply none fired."""
    failures, pending, pending_stamps = [], [], []
    per: dict[str, Any] = {}

    def pend(msg: str, stamp: str | None) -> None:
        pending.append(msg)
        pending_stamps.append(stamp)

    for k, leg in legs.items():
        if "ledger" not in leg:
            continue
        rec: dict[str, Any] = {}
        ev = leg["evidence"]
        if leg["arm"] == C.THINKING_ON:
            for f in HEADER_CERTIFICATION_KEYS:
                state, v = ev["header_state"][f], ev["header"].get(f)
                rec[f] = v
                stamp = stamp_name(C.OOS_EVIDENCE_HEADER_BLOCK, f)
                if state == EVIDENCE_ABSENT:
                    pend(f"{k}: {f} not stamped (the driver's certification, plan S3.4)", stamp)
                elif state == EVIDENCE_NULL:
                    pend(f"{k}: {f} stamped null (the driver recorded the certification as not run, "
                         "plan S3.4)", stamp)
                elif state == EVIDENCE_OFF_CONTRACT:
                    failures.append(f"{k}: " + off_contract_problem(
                        C.OOS_EVIDENCE_HEADER_BLOCK, f, v, expected="a JSON boolean"))
                elif v is not True:
                    failures.append(f"{k}: {f} is {v!r}")
        if leg["lens"]:
            b = ev["bites"]
            applicable, verified = b.get(C.OOS_EVIDENCE_BITES_APPLICABLE), b.get(C.OOS_EVIDENCE_BITES_VERIFIED)
            error, pra = b.get(C.OOS_EVIDENCE_BITES_ERROR), b.get(C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM)
            pra_state = ev["bites_state"][C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM]
            rec["bites"] = {C.OOS_EVIDENCE_BITES_APPLICABLE: applicable, C.OOS_EVIDENCE_BITES_VERIFIED: verified,
                            C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM: pra}
            if applicable is not True or verified is not True:
                failures.append(f"{k}: steering bites not verified on a lens leg "
                                f"(applicable={applicable!r}, verified={verified!r}"
                                + (f", error={str(error)[:120]}" if error else "") + ")")
            if leg["arm"] == C.THINKING_ON:
                stamp = stamp_name(C.OOS_EVIDENCE_BITES_BLOCK, C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM)
                if pra_state == EVIDENCE_ABSENT:
                    pend(f"{k}: {C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM} not stamped (plan S3.3)", stamp)
                elif pra_state == EVIDENCE_NULL:
                    pend(f"{k}: {C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM} stamped null (plan S3.3)", stamp)
                elif pra_state == EVIDENCE_OFF_CONTRACT:
                    failures.append(f"{k}: " + off_contract_problem(
                        C.OOS_EVIDENCE_BITES_BLOCK, C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM, pra,
                        expected=f"one of {list(C.THINKING_MODES)}"))
                elif pra != C.THINKING_ON:
                    failures.append(f"{k}: {C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM} {pra!r} != {C.THINKING_ON!r}")
        led = leg["ledger"]
        rec["ledger"] = {"passed": led["passed"], "failures": led["failures"], "pending": led["pending"]}
        failures.extend(f"{k}: ledger: {f}" for f in led["failures"])
        for msg, stamp in zip(led["pending"], led.get("pending_stamps") or [None] * len(led["pending"])):
            pend(f"{k}: ledger: {msg}", stamp)
        rg = leg["regimes"]
        rec["regimes"] = {"status": rg["status"], "detail": rg["detail"]}
        if rg["status"] == GATE_FAIL:
            failures.append(f"{k}: regime triple: {rg['detail']}")
        elif rg["status"] == GATE_PENDING:
            pending.append(f"{k}: regime triple: {rg['detail']}")
            pending_stamps.append(list(rg.get("pending_stamps") or []))
        if leg.get("cap_override") is True and not leg["lens"]:
            # the comparator's base-route leg must carry the
            # live refusal of the declaration key — bound to the VALIDATED comparator
            # classification, never to cap-key presence (the frozen ON stores carry the
            # key and no probe); absent / null / off-contract / not-refused all FAIL
            probe = _mapping(ev.get(C.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED))
            state, ok = probe.get("state"), probe.get(EVIDENCE_BASE_ROUTE_PROBE_OK)
            rec[C.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED] = {"status": probe.get("status"), "code": probe.get("code"),
                                                   EVIDENCE_BASE_ROUTE_PROBE_OK: ok}
            where = f"{k}: evidence.{C.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED}.{EVIDENCE_BASE_ROUTE_PROBE_OK}"
            if state == EVIDENCE_ABSENT:
                failures.append(f"{where} absent — the comparator's none leg must record the base route's live "
                                "refusal of the thinking-ON declaration key (400 thinking_on_not_lens, plan §4.8); "
                                "a store without the probe was not produced by the comparator's driver")
            elif state == EVIDENCE_NULL:
                failures.append(f"{where} stamped null — the probe did not run; the comparator's driver runs it "
                                "before generation (plan §4.8)")
            elif state == EVIDENCE_OFF_CONTRACT:
                failures.append(f"{k}: " + off_contract_problem(C.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED,
                                                                EVIDENCE_BASE_ROUTE_PROBE_OK, ok,
                                                                expected="a JSON boolean"))
            elif ok is not True:
                failures.append(f"{where} is {ok!r} (status {probe.get('status')}, code {probe.get('code')}) — "
                                "the base route did not refuse the thinking-ON declaration key")
        per[k] = rec
    status = GATE_FAIL if failures else (GATE_PENDING if pending else GATE_PASS)
    return _gate(status, "; ".join(failures) or None, legs=per, failures=failures, pending=pending,
                 pending_stamps=pending_stamps)


def instrumentation_gates(legs: Mapping[str, Mapping[str, Any]],
                          est: Mapping[str, Mapping[str, Any]] | None, *,
                          n_resamples: int, seed: int, ci: float,
                          population: Mapping[str, Any] | None = None,
                          verdict_legs: Mapping[str, Mapping[str, Any]] | None = None
                          ) -> dict[str, Any]:
    """The population gate, then the four gates in order, then the
    roll-up: ``failed`` names the gates that FAILED (the arm is not measurable
    — no word); ``pending`` names stamps this tree does not write yet (and a
    frozen id set not supplied). ``legs`` are every leg the generic gates run
    over (the six under the comparator); the anomaly re-run of the REGISTERED
    rule takes ``verdict_legs`` — the four 2x2 legs, through their natural
    mapping, MAP_ORIG (default: ``legs``)."""
    gates: dict[str, Any] = {
        "population": dict(population) if population is not None else _gate(
            GATE_PENDING, "not evaluated"),
        "cap": cap_gate(legs), "outcomes": outcomes_gate(legs)}
    if est is not None and gates["outcomes"]["status"] != GATE_FAIL:
        # a PENDING outcomes gate (a log present but not re-derived here) still
        # has every record it needs for the masking re-run; only a FAILED one has not
        gates["split_anomalies"] = split_anomaly_gate(legs if verdict_legs is None else verdict_legs, est,
                                                      n_resamples=n_resamples, seed=seed, ci=ci)
    else:
        gates["split_anomalies"] = _gate(GATE_PENDING, "not evaluated: the estimand did not run")
    gates["declaration"] = declaration_gate(legs)
    failed = sorted(k for k, g in gates.items() if g["status"] == GATE_FAIL)
    pending = sorted(k for k, g in gates.items() if g["status"] == GATE_PENDING)
    gates["failed"] = failed
    gates["pending"] = pending
    gates["pending_items"] = pending_items(gates)
    gates["pending_unwaivable"] = [it for it in gates["pending_items"] if not it["waivable"]]
    return gates


def pending_items(gates: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every PENDING entry across the gates with the producer stamp it stands
    for (``None`` for a missing operator input or an unevaluated gate) and
    whether ``--allow-pending-gates`` may waive it: exactly the stamps in
    :data:`WAIVABLE_PENDING_STAMPS`, nothing else."""
    items: list[dict[str, Any]] = []

    def add(gate: str, detail: str, stamps: Any) -> None:
        names = stamps if isinstance(stamps, list) else [stamps]
        waivable = bool(names) and all(n in WAIVABLE_PENDING_STAMPS for n in names)
        items.append({"gate": gate, "detail": detail, "stamps": names, "waivable": waivable})

    for gate in GATE_NAMES:
        g = gates.get(gate)
        if not isinstance(g, Mapping) or g.get("status") != GATE_PENDING:
            continue
        named = g.get("pending")
        if isinstance(named, list) and named:
            # a gate that names its pending entries (the declaration gate's
            # stamps; the outcomes gate's un-run re-derivations) files each one
            for detail, stamp in zip(named, g.get("pending_stamps") or [None] * len(named)):
                add(gate, detail, stamp)
        else:
            add(gate, str(g.get("detail")), NOT_A_PRODUCER_STAMP)
    return items


def _edge_limit(edge: str, legs: Sequence[tuple[str, float]], *, none_on_acc: float | None) -> str:
    """One ceiling (or floor) line over the ON legs at it — DIRECTIONAL: the
    room a benefit has is what lies between the unsteered ON cell and 1.0
    (a loss has the whole scale below it); at the floor the roles swap."""
    named = ", ".join(f"{leg} {acc:.3f}" for leg, acc in legs)
    ref = none_on_acc if none_on_acc is not None else (max if edge == "ceiling" else min)(a for _l, a in legs)
    room = (1.0 - ref) if edge == "ceiling" else -ref
    body = (CEILING_TEXT if edge == "ceiling" else FLOOR_TEXT).format(room=f"~{100 * room:+.1f} pp")
    return f"the ON legs are near the {edge} ({named}): the {edge} {body}"


def interpretation_limits(est: Mapping[str, Mapping[str, Any]], *,
                          mcnemar: Mapping[str, Any] | None = None,
                          caps: Mapping[str, Any] | None = None) -> list[str]:
    """A floor / ceiling that limits interpretation (acc near 0 or 1 on an ON
    leg) — stated SEPARATELY from the instrument gates (P2-c), and
    DIRECTIONALLY: a ceiling limits the evidence for a benefit and leaves few
    discordant errors; it does not constrain a loss. And the headline caveat:
    an ON-only effect whose percentile CI excludes zero while the OFF effect
    is unresolved is SUGGESTIVE — the exact McNemar p-values stand beside it
    and the DiD CI decides whether an interaction with the combined
    render-plus-cap configuration (never "ON-specific") is established — not
    a robust steering-harms-thinking claim."""
    out = []
    p = est[BRANCH_PRIMARY]
    none_on = pr.leg_label(("none", C.THINKING_ON))
    at_edge: dict[str, list[tuple[str, float]]] = {"floor": [], "ceiling": []}
    for leg, a in p["acc_by_leg"].items():
        # the arm's cap beside the leg, the way every other arm mention reads (OFF/512, ON/8192)
        kind, arm = leg.split("/")
        name = f"{kind}/{arm_cap_label(arm, caps)}"
        if a["point"] <= INTERPRETATION_EDGE:
            at_edge["floor"].append((name, a["point"]))
        elif a["point"] >= 1.0 - INTERPRETATION_EDGE:
            at_edge["ceiling"].append((name, a["point"]))
    none_on_acc = p["acc_by_leg"].get(none_on, {}).get("point")
    for edge in ("floor", "ceiling"):
        if at_edge[edge]:
            out.append(_edge_limit(edge, at_edge[edge], none_on_acc=none_on_acc))
    eo, en, dd = p["stats"]["effect_off"], p["stats"]["effect_on"], p["stats"]["did"]
    if eo["ci_low"] <= 0.0 <= eo["ci_high"] and en["excludes_zero"]:
        m_on = (mcnemar or {}).get(f"none_to_psr__{C.THINKING_ON}")
        mc = f"; exact two-sided McNemar: {_mcnemar_text(m_on)}" if m_on else ""
        note = first_one_sided_note(m_on)
        if note:
            mc += f" — {note} ({replicate_signs_text(est)})"
        did = (f"the DiD CI {_ci(dd)} includes zero, so no {did_interaction_text(caps)} is established"
               if dd["ci_low"] <= 0.0 <= dd["ci_high"] else
               f"the DiD CI {_ci(dd)} excludes zero — an {did_interaction_text(caps)}, not a thinking-specific one")
        out.append(f"effect_on = {_pp(en['point'])} {_ci(en)} excludes zero while the OFF effect is "
                   f"unresolved ({OFF_NOT_RESOLVED_TEXT}){mc}; {did} — a suggestive ON-arm steering "
                   f"cost, {NOT_ROBUST_TEXT}")
    return out


def _configuration(arm: str, caps: Mapping[str, Any] | None) -> str:
    cap = (caps or {}).get(arm)
    return f"thinking-{arm.upper()}" + (f"/{cap}" if cap is not None else "")


def arm_cap_label(arm: str, caps: Mapping[str, Any] | None) -> str:
    """``OFF/512`` / ``ON/8192`` — how an arm is named wherever it is named:
    the render together with its completion cap, because the two change
    together between the arms. Without a known cap, the bare arm."""
    cap = (caps or {}).get(arm)
    return arm.upper() + (f"/{cap}" if cap is not None else "")


def render_cap_configurations(caps: Mapping[str, Any] | None) -> str:
    """``OFF/512 vs ON/8192`` — the two configurations every cross-arm contrast compares."""
    return f"{arm_cap_label(C.THINKING_OFF, caps)} vs {arm_cap_label(C.THINKING_ON, caps)}"


def did_interaction_text(caps: Mapping[str, Any] | None) -> str:
    """What a nonzero DiD would be: an "interaction with the combined
    render-plus-cap configuration (OFF/512 vs ON/8192)" — never an
    "ON-specific" one. No article, so it reads after "no" and after "an"."""
    return f"interaction with the combined render-plus-cap configuration ({render_cap_configurations(caps)})"


def cross_arm_contrasts(est: Mapping[str, Mapping[str, Any]], *,
                        caps: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The three cross-arm contrasts, each qualified beside its own
    number as a render+cap contrast — ``shift_psr`` the same way ``shift_none``
    is (before this only the ``none`` shift carried the qualifier, in the
    finding) — and what the DiD would be if it were nonzero."""
    p = est[BRANCH_PRIMARY]["stats"]
    off, on = arm_cap_label(C.THINKING_OFF, caps), arm_cap_label(C.THINKING_ON, caps)
    lines = {
        "shift_none": (f"shift_none = {_pp(p['shift_none']['point'])} {_ci(p['shift_none'])} — "
                       f"{RENDER_CAP_CONTRAST_LABEL}: acc(none, {on}) − acc(none, {off}); {RENDER_AND_CAP_TEXT}"),
        "shift_psr": (f"shift_psr = {_pp(p['shift_psr']['point'])} {_ci(p['shift_psr'])} — "
                      f"{RENDER_CAP_CONTRAST_LABEL}: acc(psr, {on}) − acc(psr, {off}); {RENDER_AND_CAP_TEXT}"),
        "did": (f"DiD = {_pp(p['did']['point'])} {_ci(p['did'])} — {RENDER_CAP_CONTRAST_LABEL}: "
                f"effect_on ({on}) − effect_off ({off}); a nonzero DiD would be an {did_interaction_text(caps)}, "
                "not a thinking-specific one"),
    }
    return {"label": RENDER_CAP_CONTRAST_LABEL, "configurations": render_cap_configurations(caps),
            "note": RENDER_AND_CAP_TEXT, "lines": lines}


def findings(est: Mapping[str, Mapping[str, Any]], *, diagnostic: Mapping[str, Any] | None = None,
             caps: Mapping[str, Any] | None = None) -> list[str]:
    """P2-c: a ``none`` ON-OFF shift with clean termination is a FINDING,
    reported with its CI as shift_none — as a RENDER + CAP statement: the
    ON/OFF configurations differ in the render AND the completion cap
    (``caps`` = each arm's max_tokens), so the comparison does not isolate
    thinking from relief of OFF truncation. ``diagnostic``
    (:func:`selected_subset_diagnostic`) adds the exploratory restriction to
    the items not capped under the OFF leg, labelled as such."""
    s = est[BRANCH_PRIMARY]["stats"]["shift_none"]
    on_cfg, off_cfg = _configuration(C.THINKING_ON, caps), _configuration(C.THINKING_OFF, caps)
    ci = f"shift_none = {_pp(s['point'])} {_ci(s)}"
    out = []
    if s["ci_high"] < 0:
        out.append(f"thinking changed the unsteered cell's capability: the {on_cfg} configuration scored "
                   f"lower than {off_cfg} on the unsteered cell: {ci} (a finding, not an instrument "
                   f"failure) — {RENDER_AND_CAP_TEXT}")
    elif s["ci_low"] > 0:
        out.append(f"the {on_cfg} configuration scored higher than {off_cfg} on the unsteered cell: {ci} "
                   f"— {RENDER_AND_CAP_TEXT}")
    d = ((diagnostic or {}).get("cells") or {}).get("none")
    if out and d:
        sub = d["subset_not_capped_off"]
        a_off, a_on = sub[C.THINKING_OFF], sub[C.THINKING_ON]
        out[-1] += (f". {SELECTED_SUBSET_LABEL}: {d['n_i_to_c_at_cap_off']} of the {d['n_i_to_c']} I→C "
                    f"transitions were at the cap under OFF; on the {sub['n']} items not capped under "
                    f"none/OFF, OFF {a_off['correct']}/{a_off['scored']} vs ON {a_on['correct']}/{a_on['scored']}")
    return out


# --------------------------------------------------------------------------- #
# controls table                                                               #
# --------------------------------------------------------------------------- #

_CONTROLS_SUMMARY_FIELDS: tuple[str, ...] = (
    "acc", "n", "n_scored", "n_unscored", "unparseable_rate", "n_unparseable", "n_answered",
    "at_cap_rate", "n_at_cap", "n_stop_reason", "mean_completion_tokens", "p99_completion_tokens",
    "n_completion_tokens", "n_think_block_present", "n_think_block_unterminated",
    "n_unterminated_not_at_cap", "think_split_mode", "unterminated_policy", "max_tokens",
    "reasoning_parser", "temperature", "top_p", "top_k", "samples_per_item", "wall_clock_s",
    "n_requests", "render_regime", "salt_regime", "resumed_policy", "n_render_regimes",
    "n_salt_regimes", "n_resumed_policies")


def controls_table(legs: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per cell x arm, each value with the SOURCE it came from."""
    rows = []
    for k, leg in legs.items():
        row: dict[str, Any] = {"leg": k, "cell": leg["cell"], "arm": leg["arm"], "kind": leg["kind"],
                               "missing": leg.get("missing"), "refused": leg.get("refused")}
        s = leg.get("summary") or {}
        row["summary"] = {f: s.get(f) for f in _CONTROLS_SUMMARY_FIELDS}
        h = leg.get("outcomes_header") or {}
        row["outcomes"] = {f: h.get(f) for f in ("n_attempted_items", "n_attempted_samples",
                                                 "producer_tree_sha")}
        row["outcomes"]["n_censored"] = (h.get("n_unscored") if h else None)
        row["text"] = (leg.get("eval_check") or {}).get("text")
        row["evidence"] = leg.get("evidence")
        row["identity"] = leg.get("identity")
        led = leg.get("ledger") or {}
        row["ledger"] = {f: led.get(f) for f in ("n_calls", "max_tokens_requested", "error_class",
                                                 "n_rejected", "n_length", "prompt_max", "prompt_p99",
                                                 "window", "passed", "pending")}
        row["regimes"] = leg.get("regimes")
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# complete-case over rows (secondary) and the serving-parity cross-check     #
# --------------------------------------------------------------------------- #


def _side(label: str, rows_path: str) -> pr.Side:
    return pr.Side(label=label, rows_path=Path(rows_path), task=TASK, engine="vllm", metric=C.METRIC_ACC)


def complete_case_pairs() -> dict[str, tuple[str, str]]:
    """The four legacy complete-case pairs (B minus A), by contrast name."""
    off, on = C.THINKING_OFF, C.THINKING_ON
    return {"effect_off": (leg_key(KIND_CELL["none"], off), leg_key(KIND_CELL["psr"], off)),
            "effect_on": (leg_key(KIND_CELL["none"], on), leg_key(KIND_CELL["psr"], on)),
            "shift_none": (leg_key(KIND_CELL["none"], off), leg_key(KIND_CELL["none"], on)),
            "shift_psr": (leg_key(KIND_CELL["psr"], off), leg_key(KIND_CELL["psr"], on))}


def offcap_complete_case_pairs() -> dict[str, tuple[str, str]]:
    """The five comparator pairs, by contrast name: the two
    cap reliefs (OFF/512 → OFF/8192), the two matched shifts (OFF/8192 → ON)
    and the steering effect at OFF/8192 (none → psr)."""
    off, on = C.THINKING_OFF, C.THINKING_ON
    k8 = {kind: leg_key(KIND_CELL[kind], off, cap_override=True) for kind in pr.LEG_CELLS}
    return {"cap_relief_none": (leg_key(KIND_CELL["none"], off), k8["none"]),
            "cap_relief_psr": (leg_key(KIND_CELL["psr"], off), k8["psr"]),
            "shift_none_matched": (k8["none"], leg_key(KIND_CELL["none"], on)),
            "shift_psr_matched": (k8["psr"], leg_key(KIND_CELL["psr"], on)),
            "effect_off_8192": (k8["none"], k8["psr"])}


def complete_case_rows(legs: Mapping[str, Mapping[str, Any]], *, n_resamples: int, seed: int,
                       ci: float, pairs: Mapping[str, tuple[str, str]] | None = None) -> dict[str, Any]:
    """The complete-case comparisons over ``rows.jsonl`` through the repo's
    pairing tool (``pairing.compare_sides``, the ``compare_served_hf_task.py``
    path) — steered vs base within each arm, ON vs OFF within each cell —
    labelled :data:`pairing.COMPLETE_CASE_LABEL`, SECONDARY. Unmatched items
    (unscored on one side) are LISTED, never dropped quietly. ``pairs`` names
    the leg pairs (default :func:`complete_case_pairs`; the comparator's are
    :func:`offcap_complete_case_pairs`)."""
    pairs = complete_case_pairs() if pairs is None else pairs
    out: dict[str, Any] = {"label": pr.COMPLETE_CASE_LABEL}
    for name, (a, b) in pairs.items():
        la, lb = legs.get(a, {}), legs.get(b, {})
        if "rows_path" not in la or "rows_path" not in lb:
            out[name] = {"skipped": f"{a}: {la.get('missing') or la.get('refused') or 'ok'}; "
                                    f"{b}: {lb.get('missing') or lb.get('refused') or 'ok'}"}
            continue
        rep = pr.compare_sides(_side(f"{name}:A", la["rows_path"]), _side(f"{name}:B", lb["rows_path"]),
                               allow_unmatched=True, bootstrap=n_resamples, bootstrap_seed=seed, ci=ci)
        out[name] = {"n_items": rep["n_items"], "delta": rep["delta_b_minus_a"],
                     "mean_a": rep["mean_a"], "mean_b": rep["mean_b"],
                     "paired_bootstrap": rep["paired_bootstrap"], "mcnemar": rep["mcnemar"],
                     "n_discordant": rep["n_discordant"], "only_a": rep["only_a"], "only_b": rep["only_b"]}
    return out


#: What both sides of the serving-parity pair must agree on before a delta is
#: read: the summary's task / item set / arm / k / sampling row / cap, and the
#: manifest's model and revision.
PARITY_SUMMARY_FIELDS: tuple[str, ...] = (
    "task", "item_set", "thinking", "samples_per_item", "temperature", "top_p", "top_k", "max_tokens")
PARITY_MANIFEST_FIELDS: tuple[str, ...] = ("model_id", "model_revision")
PARITY_FIELDS: tuple[str, ...] = PARITY_SUMMARY_FIELDS + PARITY_MANIFEST_FIELDS


def parity_findings(parity: Mapping[str, Any]) -> list[str]:
    """One line per cell whose fresh-minus-wave paired delta EXCLUDES zero — a
    serving-parity finding, reported before the ON numbers are read."""
    out = []
    for cell, rec in (parity.get("cells") or {}).items():
        if rec.get("drift_beyond_ci") is True:
            out.append(f"serving-parity drift beyond its CI on {cell} OFF: fresh - wave = "
                       f"{_pp(rec['delta_fresh_minus_wave'])} [{_pp(rec['ci'][0])}, {_pp(rec['ci'][1])}] "
                       f"over {rec['n_items']} items ({rec['regime_label']}) — a serving difference to "
                       "explain before any ON number is read")
    return out


def serving_parity(legs: Mapping[str, Mapping[str, Any]], wave_root: Path | None, *,
                   wave_sha: str = WAVE_SHA_DEFAULT, n_resamples: int, seed: int, ci: float
                   ) -> dict[str, Any]:
    """The fresh OFF legs vs the wave's OFF stores on the same task, items, sampling
    row and cap: the paired delta with its CI (complete-case is exact here — no
    unscored item under OFF, asserted). BEFORE any statistic the two sides are
    compared on :data:`PARITY_FIELDS` — task, item set, arm, samples per item,
    the sampling row, the cap, the model and its revision; a mismatch
    REFUSES the statistic and labels the pair ``non-comparable (<fields>)`` —
    never like-for-like. Then the served regime (``max_model_len`` /
    ``max_num_seqs`` / ``concurrency``) is compared; a difference labels the
    comparison "across regimes" (PW5-4). Like-for-like on cache salt on the
    psr pair (F9): the ``unattested`` -> ``conditional`` label change is recorded
    when the stamps exist; "across cache regimes" is never claimed.
    ``drift_beyond_ci`` (the paired CI excludes zero) is promoted by
    :func:`analyze` into ``findings`` and ``top_items``."""
    if wave_root is None:
        return {"available": False, "reason": "no --wave-root given"}
    out: dict[str, Any] = {"available": True, "wave_sha": wave_sha, "cells": {}}
    for cell in CELLS:
        k = leg_key(cell, C.THINKING_OFF)
        fresh = legs.get(k, {})
        rec: dict[str, Any] = {"fresh_leg": k}
        wave_out = Path(C.cell_out_dir(str(wave_root), cell, C.THINKING_OFF, model_slug=MODEL)) / wave_sha
        wcd, why = _cell_dir(wave_out)
        if wcd is None:
            rec["missing"] = why
            out["cells"][cell] = rec
            continue
        if "rows_path" not in fresh:
            rec["missing"] = f"fresh OFF leg unusable: {fresh.get('missing') or fresh.get('refused')}"
            out["cells"][cell] = rec
            continue
        wm = _read_json(wcd / "manifest.json") or {}
        wt = wm.get(C.MANIFEST_KEY_TINYCAP) if isinstance(wm.get(C.MANIFEST_KEY_TINYCAP), Mapping) else {}
        ws = wt.get("served") if isinstance(wt.get("served"), Mapping) else {}
        wsum = completion.read_task_summary(wcd, TASK) or {}
        fsum = fresh["summary"]
        fident = fresh["identity"]
        # comparability first: what the two sides MEASURED must be the same
        # instrument before any delta between them means anything
        comparison = {f: {"wave": wsum.get(f), "fresh": fsum.get(f)} for f in PARITY_SUMMARY_FIELDS}
        comparison.update({f: {"wave": wm.get(f), "fresh": fident.get(f)} for f in PARITY_MANIFEST_FIELDS})
        non_comparable = sorted(f for f, v in comparison.items() if _canon(v["wave"]) != _canon(v["fresh"]))
        rec["comparison"] = comparison
        rec["comparable"] = not non_comparable
        rec["non_comparable_fields"] = non_comparable
        fserved = fident["served"]
        regime = {f: {"wave": ws.get(f), "fresh": fserved.get(f)}
                  for f in ("max_model_len", "max_num_seqs", "concurrency")}
        same = all(v["wave"] == v["fresh"] for v in regime.values())
        rec["regime"] = regime
        if non_comparable:
            rec["regime_label"] = "non-comparable (" + ", ".join(
                f"{f} {comparison[f]['wave']!r} vs {comparison[f]['fresh']!r}" for f in non_comparable) + ")"
            rec["refused"] = ("not the same instrument on both sides — no paired delta is computed; "
                              "the differing fields are " + ", ".join(non_comparable))
            rec["drift_beyond_ci"] = None
            out["cells"][cell] = rec
            continue
        rec["regime_label"] = ("like-for-like (same window, served S and client width)" if same else
                               "across regimes (" + ", ".join(
                                   f"{f} {v['wave']!r} -> {v['fresh']!r}" for f, v in regime.items()
                                   if v["wave"] != v["fresh"]) + "; PW5-4: a different S is a different "
                               "scheduling regime)")
        rec["unscored"] = {"wave": wsum.get("n_unscored"), "fresh": fresh["summary"].get("n_unscored")}
        rec["salt_label"] = {"wave": wsum.get("resumed_policy", "not stamped"),
                             "fresh": fresh["summary"].get("resumed_policy", "not stamped"),
                             "note": ("the psr pair is like-for-like on cache salt: both request-unique; "
                                      "only the policy label changed (unattested -> conditional)")
                             if cell != C.ROUTE_NONE_CELL else "base route: cell-stable on both"}
        try:
            rep = pr.compare_sides(_side("wave", str(wcd / "rows.jsonl")), _side("fresh", fresh["rows_path"]),
                                   allow_unmatched=False, bootstrap=n_resamples, bootstrap_seed=seed, ci=ci)
            pb = rep["paired_bootstrap"] or {}
            rec["delta_fresh_minus_wave"] = rep["delta_b_minus_a"]
            rec["ci"] = [pb.get("ci_low"), pb.get("ci_high")]
            rec["n_items"] = rep["n_items"]
            rec["mcnemar"] = rep["mcnemar"]
            rec["drift_beyond_ci"] = bool(pb.get("excludes_zero"))
        except pr.PairingError as e:
            rec["refused"] = str(e)
        out["cells"][cell] = rec
    return out


# --------------------------------------------------------------------------- #
# the item-transition tables (for the figure; complete-case, secondary)      #
# --------------------------------------------------------------------------- #


#: the five per-item transition classes, in the order every counts dict and
#: transition panel files them
TRANSITION_KEYS: tuple[str, ...] = ("C->C", "C->I", "I->C", "I->I", "->U")


def _transition_counts(a_items: Mapping[str, Mapping[str, Any]], b_items: Mapping[str, Mapping[str, Any]],
                       ids: Sequence[str]) -> dict[str, int]:
    """Per-item transitions a -> b over ``ids`` (:data:`TRANSITION_KEYS`): C->C,
    C->I, I->C, I->I and ->U (unscored on either side)."""
    t: Counter = Counter()
    for i in ids:
        ra, rb = a_items[i], b_items[i]
        if not ra["scored"] or not rb["scored"]:
            t["->U"] += 1
            continue
        t[("C" if int(ra["value"]) else "I") + "->" + ("C" if int(rb["value"]) else "I")] += 1
    return {k: t.get(k, 0) for k in TRANSITION_KEYS}


def transitions(legs: Mapping[str, Mapping[str, Any]], a: str, b: str) -> dict[str, Any]:
    """Per-item transitions a -> b over the common attempted set
    (:func:`_transition_counts`); McNemar over the items both legs scored
    (labelled complete-case)."""
    la, lb = legs[a]["items"], legs[b]["items"]
    ids = sorted(set(la) & set(lb))
    both = {i: (int(la[i]["value"]), int(lb[i]["value"])) for i in ids if la[i]["scored"] and lb[i]["scored"]}
    mc = None
    if both:
        from sisteer.eval.realign_stats import mcnemar
        mc = mcnemar({i: v[0] for i, v in both.items()}, {i: v[1] for i, v in both.items()})
    return {"from": a, "to": b, "n_items": len(ids), "counts": _transition_counts(la, lb, ids),
            "mcnemar": mc, "label": pr.COMPLETE_CASE_LABEL}


def transition_pairs() -> dict[str, tuple[str, str]]:
    """The four transitions (OFF -> ON per cell, none -> psr per arm), keyed
    the way :func:`all_transitions` and :func:`all_mcnemar` file them."""
    pairs: dict[str, tuple[str, str]] = {}
    for kind in pr.LEG_CELLS:
        cell = KIND_CELL[kind]
        pairs[f"off_to_on__{kind}"] = (leg_key(cell, C.THINKING_OFF), leg_key(cell, C.THINKING_ON))
    for arm in ARMS:
        pairs[f"none_to_psr__{arm}"] = (leg_key(KIND_CELL["none"], arm), leg_key(KIND_CELL["psr"], arm))
    return pairs


def offcap_transition_pairs() -> dict[str, tuple[str, str]]:
    """The five comparator transitions: OFF/512 → OFF/8192
    per cell, OFF/8192 → ON per cell, none → psr at OFF/8192."""
    off, on = C.THINKING_OFF, C.THINKING_ON
    pairs: dict[str, tuple[str, str]] = {}
    for kind in pr.LEG_CELLS:
        cell = KIND_CELL[kind]
        pairs[f"off512_to_off8192__{kind}"] = (leg_key(cell, off), leg_key(cell, off, cap_override=True))
    for kind in pr.LEG_CELLS:
        cell = KIND_CELL[kind]
        pairs[f"off8192_to_on__{kind}"] = (leg_key(cell, off, cap_override=True), leg_key(cell, on))
    pairs["none_to_psr__off8192"] = (leg_key(KIND_CELL["none"], off, cap_override=True),
                                     leg_key(KIND_CELL["psr"], off, cap_override=True))
    return pairs


def all_transitions(legs: Mapping[str, Mapping[str, Any]],
                    pairs: Mapping[str, tuple[str, str]] | None = None) -> dict[str, Any]:
    return {k: transitions(legs, a, b) for k, (a, b) in (transition_pairs() if pairs is None else pairs).items()}


# --------------------------------------------------------------------------- #
# The exact tests as sensitivity checks, the selected-subset diagnostic, #
# the closed-think capped recoding, the parity wording                        #
# --------------------------------------------------------------------------- #


def _branch_outcome(rec: Mapping[str, Any], branch: str) -> int | None:
    """The item's 0/1 under ``branch``: PRIMARY scores nothing for an unscored
    draw (``None``); SCORE-AS-IS scores it wrong."""
    if rec["scored"]:
        return int(rec["value"])
    return None if branch == BRANCH_PRIMARY else 0


def mcnemar_by_branch(legs: Mapping[str, Mapping[str, Any]], a: str, b: str) -> dict[str, Any]:
    """Exact two-sided McNemar over the discordant pairs of ``a -> b`` under
    BOTH scoring branches (PRIMARY = the items both legs scored, the
    complete-case n; SCORE-AS-IS = every common item, an unscored one wrong),
    with the discordance counts (``C->I`` / ``I->C``) and whether it runs one
    way. One-sided discordance means an EMPIRICAL bootstrap of these
    complete-case pairs could not generate a change the other way (there a
    replicate is at exactly zero only when it holds no discordant item —
    ``p_replicate_without_discordant_item`` is that chance under item
    resampling). Whether that binds the REGISTERED conditional bootstrap is a
    fact about the contrast's masks under the branch, recorded as
    ``fully_scored``: every common item scored on both legs — every mask one,
    every replicate denominator ``n_common`` — and the registered bootstrap IS
    that empirical bootstrap; a censored item on either leg (PRIMARY only:
    SCORE-AS-IS scores every item) and it resamples masks and per-replicate
    denominators, which the property does not bind (the note says which,
    :func:`one_sided_note`). A SENSITIVITY check beside the preregistered CI,
    never the decision (:data:`MCNEMAR_LABEL`); each branch's p is a
    complete-case exact sensitivity under that branch's scoring
    (:data:`MCNEMAR_BRANCH_LABEL`)."""
    from sisteer.eval.realign_stats import mcnemar

    la, lb = legs[a]["items"], legs[b]["items"]
    ids = sorted(set(la) & set(lb))
    n_common = len(ids)
    out: dict[str, Any] = {"from": a, "to": b, "label": MCNEMAR_LABEL}
    for branch in BRANCHES:
        xa: dict[str, int] = {}
        xb: dict[str, int] = {}
        for i in ids:
            va, vb = _branch_outcome(la[i], branch), _branch_outcome(lb[i], branch)
            if va is None or vb is None:
                continue
            xa[i], xb[i] = va, vb
        n = len(xa)
        c_to_i = sum(1 for i in xa if xa[i] and not xb[i])
        i_to_c = sum(1 for i in xa if xb[i] and not xa[i])
        d = c_to_i + i_to_c
        mc = mcnemar(xa, xb) if n else None
        one_sided = d > 0 and min(c_to_i, i_to_c) == 0
        rec = {
            "n_items": n, "n_common": n_common, "C->I": c_to_i, "I->C": i_to_c, "n_discordant": d,
            "p_value": mc["p_value"] if mc else None,
            "label": MCNEMAR_BRANCH_LABEL.format(branch=BRANCH_NAMES[branch]),
            "one_sided_discordance": one_sided,
            # M1: the masks under THIS branch — all one iff every common item is scored on both legs
            "fully_scored": n == n_common,
            "p_replicate_without_discordant_item": ((1.0 - d / n) ** n if n else None),
        }
        rec["note"] = one_sided_note(rec)
        out[branch] = rec
    return out


#: which REGISTERED contrast each transition pair IS — the legacy four
#: (MAP_ORIG's names) and the comparator's five (the renamed contrasts of the
#: relabelled calls, with the mapping they come from)
LEGACY_PAIR_CONTRAST: Mapping[str, str] = {
    "off_to_on__none": "shift_none", "off_to_on__psr": "shift_psr",
    "none_to_psr__off": "effect_off", "none_to_psr__on": "effect_on"}
OFFCAP_PAIR_CONTRAST: Mapping[str, tuple[str, str]] = {
    "off512_to_off8192__none": (MAP_CAP, "cap_relief_none"), "off512_to_off8192__psr": (MAP_CAP, "cap_relief_psr"),
    "off8192_to_on__none": (MAP_MATCHED, "shift_none_matched"), "off8192_to_on__psr": (MAP_MATCHED, "shift_psr_matched"),
    "none_to_psr__off8192": (MAP_CAP, "effect_off_8192")}


def attach_registered_replicate_signs(mc: dict[str, Any], contrasts: Mapping[str, tuple[Mapping[str, Mapping[str, Any]], str]]
                                      ) -> dict[str, Any]:
    """Beside every McNemar record (M1): the REGISTERED bootstrap's replicate
    signs of the contrast that pair IS — ``pairing``'s ``replicate_signs``,
    computed from its own replicate arrays — per branch, with the replicate
    count, so the one-sided note is read against the fact it is about.
    ``contrasts`` = ``{pair: (estimands block whose stats carry the contrast,
    contrast name)}``; a pair not named is left as it is."""
    for pair, (est, contrast) in contrasts.items():
        rec = mc[pair]
        rec["registered_contrast"] = contrast
        for b in BRANCHES:
            rec[b]["replicate_signs"] = dict(est[b]["stats"][contrast]["replicate_signs"])
            rec[b]["n_replicates_used"] = est[b]["n_replicates_used"]
    return mc


def all_mcnemar(legs: Mapping[str, Mapping[str, Any]],
                pairs: Mapping[str, tuple[str, str]] | None = None) -> dict[str, Any]:
    return {k: mcnemar_by_branch(legs, a, b) for k, (a, b) in (transition_pairs() if pairs is None else pairs).items()}


def _subset_acc(items: Mapping[str, Mapping[str, Any]], subset: Sequence[str]) -> dict[str, Any]:
    """The leg's conditional accuracy over ``subset`` (its own scored denominator)."""
    scored = [i for i in subset if items[i]["scored"]]
    correct = sum(1 for i in scored if items[i]["value"])
    return {"correct": correct, "scored": len(scored), "acc": (correct / len(scored)) if scored else None}


def selected_subset_diagnostic(legs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Per cell: how many of the OFF -> ON I->C transitions were AT THE CAP
    under the OFF leg, and the two legs' accuracies restricted to the items
    NOT capped under OFF (each leg's own scored denominator). An EXPLORATORY,
    post-selected restriction — labelled so — never a corrected estimate:
    isolating thinking from OFF truncation needs an adequate-budget OFF
    comparator, not a subset."""
    out: dict[str, Any] = {"label": SELECTED_SUBSET_LABEL, "cells": {}}
    for kind in pr.LEG_CELLS:
        cell = KIND_CELL[kind]
        off, on = legs[leg_key(cell, C.THINKING_OFF)], legs[leg_key(cell, C.THINKING_ON)]
        lo, ln = off["items"], on["items"]
        ids = sorted(set(lo) & set(ln))
        capped_off = {i for i in ids if lo[i]["at_cap"]}
        i_to_c = [i for i in ids if lo[i]["scored"] and ln[i]["scored"]
                  and not lo[i]["value"] and ln[i]["value"]]
        subset = [i for i in ids if i not in capped_off]
        a_off, a_on = _subset_acc(lo, subset), _subset_acc(ln, subset)
        shift = ((a_on["acc"] - a_off["acc"]) if a_on["acc"] is not None and a_off["acc"] is not None
                 else None)
        out["cells"][kind] = {
            "cap": {C.THINKING_OFF: off["summary"].get("max_tokens"),
                    C.THINKING_ON: on["summary"].get("max_tokens")},
            "n_items": len(ids), "n_at_cap_off": len(capped_off),
            "n_at_cap_off_incorrect": sum(1 for i in capped_off if lo[i]["scored"] and not lo[i]["value"]),
            "n_i_to_c": len(i_to_c), "n_i_to_c_at_cap_off": sum(1 for i in i_to_c if i in capped_off),
            "subset_not_capped_off": {"n": len(subset), C.THINKING_OFF: a_off, C.THINKING_ON: a_on,
                                      "shift": shift},
        }
    return out


def closed_think_capped_items(legs: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    """``{leg key: [item ids]}`` of the SCORED at-cap items whose think block
    CLOSED before the cap — a truncated answer after completed reasoning,
    scored under the registered policy (:data:`CLOSED_THINK_CAPPED_POLICY`)."""
    out: dict[str, list[str]] = {}
    for k, leg in legs.items():
        ids = sorted(i for i, r in leg.get("items", {}).items()
                     if r["scored"] and r["at_cap"] and r.get("think_block_present")
                     and not r["think_block_unterminated"])
        if ids:
            out[k] = ids
    return out


def closed_think_capped_sensitivity(legs: Mapping[str, Mapping[str, Any]],
                                    est: Mapping[str, Mapping[str, Any]], *,
                                    n_resamples: int, seed: int, ci: float) -> dict[str, Any]:
    """The estimand RE-RUN with every closed-think capped item recoded censored
    (unscored) — the same machinery, the same draws — filed as a SENSITIVITY
    beside the registered policy that keeps them scored: per branch the five
    contrasts (point / CI / bound), the recoded legs' accuracies, the word, and
    how much wider effect_on's bound gets."""
    items = closed_think_capped_items(legs)
    rec: dict[str, Any] = {"policy": CLOSED_THINK_CAPPED_POLICY, "items": items,
                           "n_items": sum(len(v) for v in items.values())}
    if not items:
        rec["recoded"] = None
        return rec
    recoded_legs = _recode_censored(legs, items)
    re_est = estimands(natural_mapping(recoded_legs), n_resamples=n_resamples, seed=seed, ci=ci)
    fields = SENSITIVITY_FIELDS
    rec["recoded"] = {b: {c: {f: re_est[b]["stats"][c][f] for f in fields} for c in pr.CONTRASTS}
                      for b in BRANCHES}
    rec["acc_by_leg"] = {b: re_est[b]["acc_by_leg"] for b in BRANCHES}
    v = verdict(re_est)
    rec["word_by_branch"] = v["word_by_branch"]
    rec["verdict"] = v["verdict"]
    rec["effect_on_bound_widening"] = {
        b: ((re_est[b]["stats"]["effect_on"]["bound_high"] - re_est[b]["stats"]["effect_on"]["bound_low"])
            - (est[b]["stats"]["effect_on"]["bound_high"] - est[b]["stats"]["effect_on"]["bound_low"]))
        for b in BRANCHES}
    return rec


def _recode_censored(legs: Mapping[str, Mapping[str, Any]], items: Mapping[str, Sequence[str]]
                     ) -> dict[str, dict[str, Any]]:
    """``legs`` with every ``items[leg]`` item recoded censored (unscored)."""
    return {k: {**leg, "items": {i: ({**r, "scored": False, "value": None} if i in items.get(k, ()) else r)
                                 for i, r in leg["items"].items()}}
            for k, leg in legs.items()}


def offcap_closed_think_capped_sensitivity(legs4: Mapping[str, Mapping[str, Any]],
                                           legs_off: Mapping[str, Mapping[str, Any]],
                                           ests: Mapping[str, Mapping[str, Mapping[str, Any]]],
                                           cfgs: Mapping[str, Mapping[str, Any]], *,
                                           n_resamples: int, seed: int, ci: float) -> dict[str, Any]:
    """The comparator's closed-think capped sensitivity: the SAME recoded ON
    records re-run through MAP_MATCHED, filed apart
    from the legacy block (which stays the four legs' / MAP_ORIG's). MAP_CAP has
    no ON leg, so the recoding cannot touch it — asserted (``map_cap_unchanged``:
    no OFF leg, 512 or 8192, carries a closed-think capped item; a think block
    on an OFF leg is an anomaly under ``record_only``), and only when that
    assertion FAILS is MAP_CAP recomputed too, so nothing reads stale."""
    legs6 = {**legs4, **legs_off}
    items = closed_think_capped_items(legs6)
    off_keys = [k for k, leg in legs6.items() if leg["arm"] == C.THINKING_OFF]
    map_cap_unchanged = not any(k in items for k in off_keys)
    rec: dict[str, Any] = {
        "policy": CLOSED_THINK_CAPPED_POLICY, "items": items, "n_items": sum(len(v) for v in items.values()),
        "map_cap_unchanged": map_cap_unchanged,
        "map_cap_note": ("MAP_CAP has no ON leg: the recoding touches ON items only, so the cap-relief contrasts "
                         "are unchanged — asserted, not recomputed" if map_cap_unchanged else
                         "an OFF leg carries a closed-think capped item (an anomaly: no think block is expected "
                         "under record_only) — MAP_CAP was recomputed too"),
        "recoded": None}
    if not items:
        return rec
    recoded6 = _recode_censored(legs6, items)
    maps_r = three_mappings({k: recoded6[k] for k in legs4}, {k: recoded6[k] for k in legs_off})
    changed = [MAP_MATCHED] + ([] if map_cap_unchanged else [MAP_CAP])
    re_raw = {name: estimands(maps_r[name], n_resamples=n_resamples, seed=seed, ci=ci) for name in changed}
    renamed = {name: rename_contrasts(re_raw[name], name, cfgs) for name in changed}
    fields = SENSITIVITY_FIELDS
    rec["recoded"] = {name: {b: {c: {f: s[f] for f in fields} for c, s in renamed[name][b]["stats"].items()}
                             for b in BRANCHES} for name in changed}
    rec["acc_by_leg"] = {name: {b: renamed[name][b]["acc_by_leg"] for b in BRANCHES} for name in changed}
    rec["effect_on_bound_widening"] = {
        b: ((re_raw[MAP_MATCHED][b]["stats"]["effect_on"]["bound_high"]
             - re_raw[MAP_MATCHED][b]["stats"]["effect_on"]["bound_low"])
            - (ests[MAP_MATCHED][b]["stats"]["effect_on"]["bound_high"]
               - ests[MAP_MATCHED][b]["stats"]["effect_on"]["bound_low"]))
        for b in BRANCHES}
    rec["verdict_sensitivity_off_8192"] = verdict_sensitivity_off_8192(
        {MAP_CAP: re_raw.get(MAP_CAP, ests[MAP_CAP]), MAP_MATCHED: re_raw[MAP_MATCHED]})
    return rec


def parity_wording(parity: Mapping[str, Any]) -> dict[str, Any]:
    """What the serving-parity screen may be SAID to show: a screen that
    found no drift is "no drift detected by the registered screen" — never
    "parity established" (:data:`PARITY_NOTE`)."""
    if not parity.get("available"):
        return {"wording": f"no serving-parity screen was run ({parity.get('reason')})", "note": PARITY_NOTE,
                "drift_cells": [], "unscreened_cells": []}
    cells = parity.get("cells") or {}
    drift = sorted(c for c, r in cells.items() if r.get("drift_beyond_ci") is True)
    unscreened = sorted(c for c, r in cells.items() if r.get("drift_beyond_ci") is not False)
    if drift:
        wording = f"drift beyond its CI detected by the registered screen on {drift}"
    elif unscreened:
        wording = f"the registered screen did not run on every cell (unscreened: {unscreened})"
    else:
        wording = PARITY_NO_DRIFT
    return {"wording": wording, "note": PARITY_NOTE, "drift_cells": drift, "unscreened_cells": unscreened}


def arm_caps(legs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Each arm's completion cap as the ``none`` legs' summaries state it."""
    return {arm: (legs.get(leg_key(KIND_CELL["none"], arm), {}).get("summary") or {}).get("max_tokens")
            for arm in ARMS}


#: The two shas a report carries, and why they differ
ANALYSIS_CODE_NOTE = ("the git sha of the tree this analysis code ran from — distinct from `sha`, the "
                      "producer/store pin (the run's oos-<sha7> out-suffix): a re-analysis of old stores "
                      "moves this sha and not that one")
_SHA_RE = re.compile(r"[0-9a-f]{7,40}")


def analysis_code_provenance(explicit: str | None = None) -> dict[str, Any]:
    """The analysis code's own git sha, recorded beside the producer/store pin
    so a report says which code read the stores. Read at runtime from the tree
    this file lives in — ``git rev-parse HEAD`` plus whether that tree had
    uncommitted TRACKED changes — or taken from ``--analysis-sha`` for a copy
    that is not a checkout. Neither available → refuse by name; a null here
    would be a silent gap."""
    if explicit is not None:
        if not _SHA_RE.fullmatch(explicit):
            raise TransferAnalysisError(f"--analysis-sha {explicit!r}: expected 7-40 hex digits")
        return {"sha": explicit, "source": "--analysis-sha", "dirty": None, "tree": None, "note": ANALYSIS_CODE_NOTE}
    here = Path(__file__).resolve().parent

    def git(*args: str) -> str:
        return subprocess.run(["git", "-C", str(here), *args], check=True, capture_output=True,
                              text=True).stdout.strip()

    try:
        sha, top = git("rev-parse", "HEAD"), git("rev-parse", "--show-toplevel")
        dirty = git("status", "--porcelain", "--untracked-files=no") != ""
    except (OSError, subprocess.CalledProcessError) as e:
        raise TransferAnalysisError(
            f"the analysis code's git sha could not be read from {here} ({e}); pass --analysis-sha "
            "<sha> when running from a copy that is not a git checkout") from e
    return {"sha": sha, "source": "git rev-parse HEAD of the script's tree", "dirty": dirty, "tree": top,
            "note": ANALYSIS_CODE_NOTE}


# --------------------------------------------------------------------------- #
# the thinking-OFF comparator at cap 8192                                      #
# --------------------------------------------------------------------------- #


def configuration_caps(legs4: Mapping[str, Mapping[str, Any]],
                       legs_off: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    """The served configurations' caps and labels — ``OFF/512``, ``OFF/8192``,
    ``ON/8192`` — as the `none` legs' summaries state them (:func:`arm_caps`
    for the two arms; the comparator's `none` leg for the override cap)."""
    caps = arm_caps(legs4)
    out = {CFG_OFF: {"cap": caps[C.THINKING_OFF], "label": arm_cap_label(C.THINKING_OFF, caps)},
           CFG_ON: {"cap": caps[C.THINKING_ON], "label": arm_cap_label(C.THINKING_ON, caps)}}
    if legs_off is not None:
        cap8 = (legs_off.get(leg_key(KIND_CELL["none"], C.THINKING_OFF, cap_override=True), {}).get("summary")
                or {}).get("max_tokens")
        out[CFG_OFF_OVERRIDE] = {"cap": cap8, "label": arm_cap_label(C.THINKING_OFF, {C.THINKING_OFF: cap8})}
    return out


def contrast_varies(cfgs: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """What each of the contrasts VARIES: every contrast is labelled by what
    it varies — budget only / render only / both; the figures' axis labels
    read it."""
    off, off8, on = cfgs[CFG_OFF]["label"], cfgs[CFG_OFF_OVERRIDE]["label"], cfgs[CFG_ON]["label"]
    cap_off, cap8 = cfgs[CFG_OFF]["cap"], cfgs[CFG_OFF_OVERRIDE]["cap"]
    both = f"render+cap ({off} vs {on})"
    budget = f"budget only (OFF render, {cap_off} → {cap8})"
    return {"effect_off": f"steering at {off}", "effect_on": f"steering at {on}",
            "shift_none": both, "shift_psr": both, "did": both,
            "cap_relief_none": budget, "cap_relief_psr": budget,
            "did_cap_relief": f"budget only (OFF render, {cap_off} → {cap8}; the steering interaction)",
            "shift_none_matched": f"render only (cap {cap8})", "shift_psr_matched": f"render only (cap {cap8})",
            "did_matched": f"render only (matched cap {cap8})", "effect_off_8192": f"steering at {off8}"}


def rename_contrasts(est: Mapping[str, Mapping[str, Any]], mapping_name: str,
                     cfgs: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """One relabelled call's output under the contrast names it IS
    (:data:`CONTRAST_NAMES_BY_MAPPING`) and with every internal slot label
    replaced by the configuration it held (``none/OFF/8192``), plus a
    ``pseudo_2x2`` record so nobody reads the internal ``on`` slot as thinking
    and the ``varies`` label per contrast."""
    names = CONTRAST_NAMES_BY_MAPPING[mapping_name]
    slots = SLOT_CONFIGURATIONS[mapping_name]
    label_of = {pr.leg_label((kind, arm)): f"{kind}/{cfgs[slots[arm]]['label']}" for kind, arm in pr.LEG_KEYS}
    varies = contrast_varies(cfgs)
    out: dict[str, dict[str, Any]] = {}
    for b in BRANCHES:
        e = est[b]
        out[b] = {
            "B": e["B"], "seed": e["seed"], "level": e["level"], "n_items": e["n_items"],
            "n_scored_by_leg": {label_of[k]: v for k, v in e["n_scored_by_leg"].items()},
            "n_censored_by_leg": {label_of[k]: v for k, v in e["n_censored_by_leg"].items()},
            "n_correct_by_leg": {label_of[k]: v for k, v in e["n_correct_by_leg"].items()},
            "n_degenerate_replicates": e["n_degenerate_replicates"], "n_replicates_used": e["n_replicates_used"],
            "acc_by_leg": {label_of[k]: v for k, v in e["acc_by_leg"].items()},
            "stats": {names[c]: {**s, "legs": {label_of[k]: sg for k, sg in s["legs"].items()}}
                      for c, s in e["stats"].items()},
            "complete_case": {"label": e["complete_case"]["label"],
                              **{names[c]: e["complete_case"][c] for c in pr.CONTRASTS}},
            "pseudo_2x2": {C.THINKING_OFF: cfgs[slots[C.THINKING_OFF]]["label"],
                           C.THINKING_ON: cfgs[slots[C.THINKING_ON]]["label"],
                           "note": "the bootstrap's internal 2x2 slots held these configurations; a slot is not "
                                   "a render unless its label says ON/"},
            "varies": {names[c]: varies[names[c]] for c in pr.CONTRASTS},
        }
    return out


#: a contrast's point, percentile CI and censoring-sensitivity bound — the five
#: numbers every figure and the W&B summary log per contrast
STAT_FIELDS: tuple[str, ...] = ("point", "ci_low", "ci_high", "bound_low", "bound_high")
#: ... plus the bound's own CI: what the closed-think sensitivities carry per contrast
SENSITIVITY_FIELDS: tuple[str, ...] = (*STAT_FIELDS, "bound_ci_low", "bound_ci_high")
#: the fields two calls sharing a contrast must agree on bit for bit (``legs``
#: is the slot label and differs by construction)
CONSISTENCY_FIELDS: tuple[str, ...] = (*SENSITIVITY_FIELDS, "replicate_signs", "excludes_zero", "point_sign",
                                       "sign_identified")


def offcap_consistency(ests: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """The three cross-call equalities, asserted at runtime:
    effect_off is one number in MAP_ORIG and MAP_CAP; effect_on in MAP_ORIG
    and MAP_MATCHED; effect_off_8192 in MAP_CAP (its ``effect_on``) and
    MAP_MATCHED (its ``effect_off``) — on point / CI / bounds / replicate signs.
    Refuses otherwise: the three calls did not share their replicate draws."""
    checks: list[dict[str, Any]] = []
    for a, ca, b, cb, what in (
            (MAP_CAP, "effect_off", MAP_ORIG, "effect_off", "effect_off: the OFF/512 legs in the cap-relief and the original call"),
            (MAP_MATCHED, "effect_on", MAP_ORIG, "effect_on", "effect_on: the ON legs in the matched and the original call"),
            (MAP_MATCHED, "effect_off", MAP_CAP, "effect_on", ("effect_off_8192: the OFF/8192 legs in the matched call's "
                                                              "off slots and the cap-relief call's on slots"))):
        for br in BRANCHES:
            sa, sb = ests[a][br]["stats"][ca], ests[b][br]["stats"][cb]
            differing = [f for f in CONSISTENCY_FIELDS if sa[f] != sb[f]]
            checks.append({"what": what, "branch": br, "a": f"{a}.{ca}", "b": f"{b}.{cb}",
                           "equal": not differing, "differing": differing})
    bad = [c for c in checks if not c["equal"]]
    if bad:
        raise TransferAnalysisError(
            f"offcap consistency: the relabelled calls disagree on a shared contrast — {bad}; the three calls "
            "did not share their replicate draws (seed, n, B must be identical)")
    return {"ok": True, "fields": list(CONSISTENCY_FIELDS), "checks": checks}


def _replicate_leg_counts(legs_by_name: Mapping[str, Mapping[str, Any]], branch: str, *,
                          n_resamples: int, seed: int) -> tuple[dict[str, tuple[Any, Any]], int]:
    """Per named leg the replicates' integer ``(correct, scored)`` counts over
    the SAME index draws ``pairing.paired_bootstrap_conditional_2x2`` makes —
    ``default_rng(seed).integers(0, n, size=(m, n))`` per chunk, the same chunk
    order — so a contrast rebuilt from them as ONE division of integer counts
    is that call's replicate array. Ties the decomposition's replicate identity
    to pairing's own draw (the CI comparison in :func:`decomposition` refuses
    a drift)."""
    import numpy as np

    ids: list[str] | None = None
    masks: dict[str, Any] = {}
    hits: dict[str, Any] = {}
    for name, leg in legs_by_name.items():
        d = _branch_items(leg, branch)
        if ids is None:
            ids = sorted(d)
        elif sorted(d) != ids:
            raise TransferAnalysisError(f"{name}: attempted ids differ from the first leg's")
        m = np.asarray([int(d[i][0]) for i in ids], dtype=np.int64)
        y = np.asarray([int(d[i][1]) if d[i][0] == 1 else 0 for i in ids], dtype=np.int64)
        masks[name], hits[name] = m, m * y
    n = len(ids or [])
    rng = np.random.default_rng(seed)
    out = {name: (np.empty(n_resamples, dtype=np.int64), np.empty(n_resamples, dtype=np.int64))
           for name in legs_by_name}
    chunk = max(1, 4_000_000 // n)
    done = 0
    while done < n_resamples:
        m = min(chunk, n_resamples - done)
        idx = rng.integers(0, n, size=(m, n))
        sl = slice(done, done + m)
        for name in legs_by_name:
            out[name][0][sl] = hits[name][idx].sum(axis=1)
            out[name][1][sl] = masks[name][idx].sum(axis=1)
        done += m
    return out, n


def _replicate_contrast(cnt: Mapping[str, tuple[Any, Any]], a_name: str, c_name: str) -> Any:
    """``c − a`` per replicate, EXACTLY as pairing forms a two-leg contrast: ONE
    float division of exact integer counts, ``(c_c·s_a − c_a·s_c) / (s_c·s_a)``."""
    import numpy as np

    (ca, sa), (cc, sc) = cnt[a_name], cnt[c_name]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (cc * sa - ca * sc) / (sc * sa)


def _replicates_kept(cnt: Mapping[str, tuple[Any, Any]], *leg_names: str) -> Any:
    """The replicates a four-leg call KEEPS: every named leg's denominator > 0."""
    import numpy as np

    kept = None
    for nm in leg_names:
        ok = cnt[nm][1] > 0
        kept = ok if kept is None else (kept & ok)
    return kept if kept is not None else np.ones(0, dtype=bool)


def _plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def discordance_qualifier(rec: Mapping[str, Any], *, interval: tuple[float, float] | None = None) -> str:
    """The sparse-discordance / exact-test qualification the cap-only contrast
    wears BESIDE the headline: how many discordant items of how many, gains /
    losses, the exact two-sided McNemar p (a complete-case sensitivity), and
    what the contrast's masks imply for the registered bootstrap — both legs
    fully scored: it is the empirical bootstrap of those pairs, so a one-way
    discordance means it cannot generate a contrast of the opposite sign;
    masks varying: the qualified statement. The endpoint on the discordances'
    side is READ off ``interval`` (the budget term's percentile CI), never
    assumed (1 gain of 200 leaves a zero lower endpoint — about 36.7 % of
    replicates carry no gain): positive, it reflects the sparse empirical
    distribution, not corroboration; zero, said so with the share of
    replicates carrying no discordant item (the registered replicate signs
    when the record carries them, :func:`attach_registered_replicate_signs`,
    else the analytic ``(1 − d/n)^n``); without an interval the wording is
    conditional."""
    d, n, gains, losses = rec["n_discordant"], rec["n_items"], rec["I->C"], rec["C->I"]
    p = rec["p_value"]
    p_text = "n/a (no discordant pair)" if p is None else f"{p:.4g}"
    head = (f"The cap-only contrast rests on {d} discordant {_plural(d, 'item', 'items')} of {n} "
            f"({gains} {_plural(gains, 'gain', 'gains')} / {losses} {_plural(losses, 'loss', 'losses')}; exact two-sided "
            f"McNemar p = {p_text}, a complete-case sensitivity)")
    rs, n_rep = rec.get("replicate_signs"), rec.get("n_replicates_used")
    signs = (f"replicate signs {rs['negative']:,} negative / {rs['zero']:,} zero / {rs['positive']:,} positive"
             + (f" of {n_rep:,}" if n_rep else "")) if rs else None
    if rec.get("fully_scored") is not True:
        return f"{head}; a leg carries censored draws under this branch, so {ONE_SIDED_NOTE_REGISTERED}" + (
            f" ({signs})" if signs else "")
    if rec["one_sided_discordance"]:
        gains_only = losses == 0
        side, sign_word, item_word = ("lower", "positive", "gain") if gains_only else ("upper", "negative", "loss")
        conditional = (f"a {sign_word} {side} endpoint, when present, reflects this sparse empirical distribution, not "
                       "corroboration")
        if interval is None:
            tail = f"— {conditional}"
        else:
            endpoint = float(interval[0] if gains_only else interval[1])
            if (endpoint > 0) if gains_only else (endpoint < 0):
                tail = f"— its {sign_word} {side} endpoint ({_pp(endpoint)}) reflects this sparse empirical distribution, not corroboration"
            else:
                if rs and n_rep:
                    share = f"{rs['zero']:,} of {n_rep:,} replicates carry no {item_word}"
                else:
                    share = f"about {100 * rec['p_replicate_without_discordant_item']:.1f} % of replicates carry no {item_word}"
                tail = f"— its {side} endpoint is {_pp(endpoint)}: {share}, so the interval reaches zero; {conditional}"
        return (f"{head}; both legs are fully scored under this branch, so the registered bootstrap cannot generate a "
                f"contrast of the opposite sign" + (f" ({signs})" if signs else "") + f" {tail}")
    return (f"{head}; both legs are fully scored under this branch, so the registered bootstrap is the empirical "
            f"bootstrap of those pairs and its interval reflects that sparse distribution" + (f" ({signs})" if signs else ""))


def decomposition(ests: Mapping[str, Mapping[str, Mapping[str, Any]]],
                  maps: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]],
                  cfgs: Mapping[str, Mapping[str, Any]], *, n_resamples: int, seed: int, ci: float,
                  budget_mcnemar: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Per cell and branch: ``shift_c = cap_relief_c +
    shift_c_matched`` EXACTLY — on the full set (the rationals) and inside
    every replicate (the OFF/8192 accuracy cancels; the three calls share the
    index draw) — with each term's own CI, a DESCRIPTIVE budget share (no
    interval: no new statistics beyond the contrasts) and the
    statement. ``identity_replicates_to_an_ulp`` is read off replicate arrays
    rebuilt from pairing's own draw (:func:`_replicate_leg_counts`) and every
    rebuilt array's percentile pair must equal the call's filed CI — a drift
    from pairing refuses rather than passing a check on other numbers.

    The statement (the analysis review's M1) reads "the observed cap-only
    contrast was +x pp [CI]" — never "+x pp is budget" — names the identity as
    one between OBSERVED estimates (an accounting identity, not a causal
    partition) and carries the cap-only contrast's sparse-discordance / exact
    McNemar qualification (:func:`discordance_qualifier`) beside the numbers;
    ``budget_mcnemar`` = ``{cell kind: the OFF/512 → OFF/8192 McNemar record of
    that cell}`` (:func:`mcnemar_by_branch`, signs attached), filed per branch
    as ``budget_discordance`` so the qualification sits in this block and in
    the findings, not only under ``sensitivity``."""
    import numpy as np

    off, off8, on = cfgs[CFG_OFF]["label"], cfgs[CFG_OFF_OVERRIDE]["label"], cfgs[CFG_ON]["label"]
    alpha = 1.0 - ci
    # the six legs by configuration: the 2x2's four (MAP_ORIG) and the two
    # OFF/8192 legs (MAP_CAP's ``on`` slots), ONE index draw for all — the
    # rebuilt replicate counts and the kept masks depend on the BRANCH alone,
    # so they are drawn once per branch and read per cell
    legs_named = {"none_off": maps[MAP_ORIG][("none", C.THINKING_OFF)],
                  "psr_off": maps[MAP_ORIG][("psr", C.THINKING_OFF)],
                  "none_on": maps[MAP_ORIG][("none", C.THINKING_ON)],
                  "psr_on": maps[MAP_ORIG][("psr", C.THINKING_ON)],
                  "none_off8": maps[MAP_CAP][("none", C.THINKING_ON)],
                  "psr_off8": maps[MAP_CAP][("psr", C.THINKING_ON)]}
    counts = {b: _replicate_leg_counts(legs_named, b, n_resamples=n_resamples, seed=seed)[0] for b in BRANCHES}
    kept = {b: (_replicates_kept(cnt, "none_off", "psr_off", "none_on", "psr_on"),
                _replicates_kept(cnt, "none_off", "psr_off", "none_off8", "psr_off8"),
                _replicates_kept(cnt, "none_off8", "psr_off8", "none_on", "psr_on"))
            for b, cnt in counts.items()}
    out: dict[str, Any] = {}
    for kind in pr.LEG_CELLS:
        cell_word = "unsteered" if kind == "none" else "steered (psr)"
        out[kind] = {}
        for b in BRANCHES:
            total = ests[MAP_ORIG][b]["stats"][f"shift_{kind}"]
            budget = ests[MAP_CAP][b]["stats"][f"shift_{kind}"]
            render = ests[MAP_MATCHED][b]["stats"][f"shift_{kind}"]
            # the exact identity on the full set, from the integer counts the calls filed
            a_off = ests[MAP_ORIG][b]["acc_by_leg"][pr.leg_label((kind, C.THINKING_OFF))]
            a_on = ests[MAP_ORIG][b]["acc_by_leg"][pr.leg_label((kind, C.THINKING_ON))]
            a_8 = ests[MAP_CAP][b]["acc_by_leg"][pr.leg_label((kind, C.THINKING_ON))]
            fr = {k: Fraction(a["n_correct"], a["n_scored"]) for k, a in (("off", a_off), ("on", a_on), ("off8", a_8))}
            exact = (fr["on"] - fr["off"] == (fr["off8"] - fr["off"]) + (fr["on"] - fr["off8"])
                     and float(fr["on"] - fr["off"]) == total["point"] and float(fr["off8"] - fr["off"]) == budget["point"]
                     and float(fr["on"] - fr["off8"]) == render["point"])
            # the identity inside every replicate, on arrays rebuilt from pairing's draw
            cnt = counts[b]
            r_total = _replicate_contrast(cnt, f"{kind}_off", f"{kind}_on")
            r_budget = _replicate_contrast(cnt, f"{kind}_off", f"{kind}_off8")
            r_render = _replicate_contrast(cnt, f"{kind}_off8", f"{kind}_on")
            k_orig, k_cap, k_matched = kept[b]
            for arr, kp, filed, which in ((r_total, k_orig, total, "shift"), (r_budget, k_cap, budget, "cap_relief"),
                                          (r_render, k_matched, render, "matched shift")):
                lo, hi = np.quantile(arr[kp], [alpha / 2.0, 1.0 - alpha / 2.0])
                if (float(lo), float(hi)) != (filed["ci_low"], filed["ci_high"]):
                    raise TransferAnalysisError(
                        f"decomposition ({kind}, {b}): the rebuilt {which} replicates' percentile pair "
                        f"({float(lo)}, {float(hi)}) != the filed CI ({filed['ci_low']}, {filed['ci_high']}) — the "
                        "re-derivation does not reproduce pairing's draw; refusing the identity check")
            all_kept = k_orig & k_cap & k_matched
            ulp = bool(np.abs(r_total[all_kept] - (r_budget[all_kept] + r_render[all_kept])).max() <= 1e-12) \
                if all_kept.any() else None
            share = (budget["point"] / total["point"]) if total["point"] != 0 else None
            direction = "gain" if total["point"] > 0 else ("loss" if total["point"] < 0 else "change")
            statement = (f"Of the {cell_word} {off} → {on} {direction} of {_pp(total['point'])} {_ci(total)}, the "
                         f"observed cap-only contrast ({off} → {off8}, same render) was {_pp(budget['point'])} "
                         f"{_ci(budget)} and the observed render-only contrast ({off8} → {on}, same cap) was "
                         f"{_pp(render['point'])} {_ci(render)}; the two point estimates sum to the total exactly (a "
                         "decomposition identity between observed estimates, not a causal partition); the intervals "
                         "are each term's own and do not add.")
            mc_rec = (budget_mcnemar or {}).get(kind, {}).get(b)
            discordance = None
            if mc_rec is not None:
                qualifier = discordance_qualifier(mc_rec, interval=(budget["ci_low"], budget["ci_high"]))
                statement += f" {qualifier}."
                discordance = {
                    "transition": f"off512_to_off8192__{kind}", "registered_contrast": f"cap_relief_{kind}",
                    **{f: mc_rec.get(f) for f in ("n_items", "n_common", "C->I", "I->C", "n_discordant", "p_value",
                                                  "one_sided_discordance", "fully_scored", "label", "note",
                                                  "replicate_signs", "n_replicates_used")},
                    "qualifier": qualifier}
            out[kind][b] = {
                "total": total, "budget": budget, "render": render,
                "identity_exact_full_set": bool(exact), "identity_replicates_to_an_ulp": ulp,
                "n_replicates_compared": int(all_kept.sum()),
                "budget_share_point": share,
                "budget_share_note": ("a descriptive ratio of the two point estimates, no interval (spec: no new "
                                      "statistics beyond the contrasts); the uncertainty statement is the two "
                                      "terms' own CIs"),
                "budget_discordance": discordance,
                "statement": statement}
    return out


def did_matched_reading(ests: Mapping[str, Mapping[str, Mapping[str, Any]]],
                        cfgs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The matched DiD: ``effect_on − effect_off_8192 =
    shift_psr_matched − shift_none_matched`` — render only, at the matched cap."""
    s = ests[MAP_MATCHED][BRANCH_PRIMARY]["stats"]["did"]
    off8, on = cfgs[CFG_OFF_OVERRIDE]["label"], cfgs[CFG_ON]["label"]
    level = float(ests[MAP_MATCHED][BRANCH_PRIMARY].get("level", pr.DEFAULT_CI))
    text = (contrast_text("did_matched", s, level=level)
            + f" — render only at the matched cap: effect_on ({on}) − effect_off_8192 ({off8}) = shift_psr_matched − "
              "shift_none_matched; a nonzero value would be an interaction of steering with the render at a "
              "matched budget")
    if not _resolved(s):
        text += "; the CI includes zero: no such interaction is established"
    return {"point": s["point"], "ci": [s["ci_low"], s["ci_high"]], "bound": [s["bound_low"], s["bound_high"]],
            "excludes_zero": s["excludes_zero"], "sign_identified": s["sign_identified"],
            "varies": contrast_varies(cfgs)["did_matched"], "statement": text}


def effect_off_8192_reading(ests: Mapping[str, Mapping[str, Mapping[str, Any]]],
                            cfgs: Mapping[str, Mapping[str, Any]],
                            transitions_off: Mapping[str, Any] | None) -> dict[str, Any]:
    """``effect_off_8192`` with the same first-branch reading ``effect_off``
    gets, verdict words only where the original rule applies: resolved iff the
    percentile CI excludes zero, else :data:`OFF_NOT_RESOLVED_TEXT_8192`, with
    the OFF/8192 discordances beside it."""
    s = ests[MAP_CAP][BRANCH_PRIMARY]["stats"]["effect_on"]
    level = float(ests[MAP_CAP][BRANCH_PRIMARY].get("level", pr.DEFAULT_CI))
    resolved = _resolved(s)
    text = contrast_text("effect_off_8192", s, level=level)
    if not resolved:
        text += f" — {OFF_NOT_RESOLVED_TEXT_8192}"
    t = (transitions_off or {}).get("none_to_psr__off8192")
    if t:
        cnt = t["counts"]
        text += f" ({cfgs[CFG_OFF_OVERRIDE]['label']} discordances none→psr: C→I {cnt['C->I']} / I→C {cnt['I->C']})"
    return {"resolved": resolved, "point": s["point"], "ci": [s["ci_low"], s["ci_high"]],
            "bound": [s["bound_low"], s["bound_high"]], "point_sign": s["point_sign"],
            "varies": contrast_varies(cfgs)["effect_off_8192"], "text": text,
            "reading": ("the same first-branch reading as effect_off (D80 (2)); the registered transfer rule's "
                        "input stays the 512 leg — the 8192 reading is a sensitivity "
                        "(verdict_sensitivity_off_8192)")}


def off512_to_off8192_description(legs4: Mapping[str, Mapping[str, Any]],
                                  legs_off: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The EXPLORATORY OFF/512 → OFF/8192 transition description — per cell
    the transition counts over all items, the same
    counts split by the OFF/512 draw's at-cap status, the formerly capped
    items' OFF/8192 outcomes and completion lengths, and how many not-capped
    items changed completion length between the two draws. NOT a diagnostic
    with an expectation and NOT parity evidence: :data:`OFF512_TO_OFF8192_WHY`."""
    out: dict[str, Any] = {"label": OFF512_TO_OFF8192_LABEL, "why_no_expectation": OFF512_TO_OFF8192_WHY, "cells": {}}
    for kind in pr.LEG_CELLS:
        cell = KIND_CELL[kind]
        a = legs4[leg_key(cell, C.THINKING_OFF)]["items"]
        b = legs_off[leg_key(cell, C.THINKING_OFF, cap_override=True)]["items"]
        ids = sorted(set(a) & set(b))
        capped = [i for i in ids if a[i]["at_cap"]]
        not_capped = [i for i in ids if not a[i]["at_cap"]]
        lengths = sorted(int(b[i]["completion_tokens"]) for i in capped if b[i].get("completion_tokens") is not None)
        compared = [i for i in not_capped
                    if a[i].get("completion_tokens") is not None and b[i].get("completion_tokens") is not None]
        out["cells"][kind] = {
            "n_items": len(ids), "counts": _transition_counts(a, b, ids),
            "by_first_draw_stop": {
                "capped_at_512": {"n": len(capped), "counts": _transition_counts(a, b, capped)},
                "not_capped_at_512": {"n": len(not_capped), "counts": _transition_counts(a, b, not_capped)}},
            "capped_at_512_second_draw": {
                "n": len(capped),
                "n_correct_at_8192": sum(1 for i in capped if b[i]["scored"] and int(b[i]["value"]) == 1),
                "n_at_cap_at_8192": sum(1 for i in capped if b[i]["at_cap"]),
                "completion_tokens_at_8192": {
                    "min": lengths[0] if lengths else None, "p50": (lengths[len(lengths) // 2] if lengths else None),
                    "max": lengths[-1] if lengths else None, "values": lengths}},
            "not_capped_at_512": {
                "n": len(not_capped), "n_compared": len(compared),
                "n_completion_tokens_differ": sum(1 for i in compared
                                                  if int(a[i]["completion_tokens"]) != int(b[i]["completion_tokens"]))},
        }
    return out


def offcap_block(*, legs4: Mapping[str, Mapping[str, Any]], legs_off: Mapping[str, Mapping[str, Any]],
                 legs6: Mapping[str, Mapping[str, Any]], ests: Mapping[str, Mapping[str, Mapping[str, Any]]] | None,
                 maps: Mapping[str, Mapping[tuple[str, str], Mapping[str, Any]]] | None,
                 prompt_budget_off: Mapping[str, Any] | None, sha: str, offcap_sha: str, analysis_code_sha: str,
                 n_resamples: int, seed: int, ci: float,
                 object_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The report's ``offcap`` block: the two records, the
    renamed contrasts of the two relabelled calls, the consistency and
    decomposition blocks, the effect_off_8192 reading and the verdict
    sensitivity, the five comparator transitions / McNemar / complete-case
    pairs, the exploratory description, the comparator's gates and controls,
    provenance with three shas kept apart. Without an estimand (a leg missing
    or refused: the six-leg population gate is all-or-nothing) only the leg
    records, the artifact, the ledger, the probe and the notes are filed."""
    cfgs = configuration_caps(legs4, legs_off)
    k_none = leg_key(KIND_CELL["none"], C.THINKING_OFF, cap_override=True)
    probe_leg = legs_off.get(k_none, {})
    probe = _mapping(_mapping(probe_leg.get("evidence")).get(C.OOS_EVIDENCE_BASE_ROUTE_DECLARATION_REFUSED))
    block: dict[str, Any] = {
        "provenance": {"sha": sha, "offcap_sha": offcap_sha, "analysis_code_sha": analysis_code_sha,
                       "note": OFFCAP_PROVENANCE_NOTE},
        "legs": legs_off,
        # M3: the fitted object's artifact digest per leg and its agreement within every cell
        "object_identity": object_identity,
        "configurations": cfgs,
        "contrast_varies": contrast_varies(cfgs) if cfgs.get(CFG_OFF_OVERRIDE, {}).get("cap") is not None else None,
        "prompt_budget_off": ({k: prompt_budget_off[k] for k in C.OOS_PROMPT_BUDGET_FIELDS if k != "lengths"}
                              if prompt_budget_off is not None else None),
        "cap_gate": cap_gate(legs_off),
        "ledger": {k: {f: leg["ledger"].get(f) for f in ("passed", "failures", "pending", "n_calls", "max_tokens_requested",
                                                            "thinking", "prompt_max", "prompt_p99", "window", "artifact_fit")}
                   for k, leg in legs_off.items() if "ledger" in leg},
        "base_route_declaration_probe": {"leg": k_none, "status": probe.get("status"), "code": probe.get("code"),
                                         EVIDENCE_BASE_ROUTE_PROBE_OK: probe.get(EVIDENCE_BASE_ROUTE_PROBE_OK),
                                         "state": probe.get("state", EVIDENCE_ABSENT)},
        "parity_note": OFFCAP_PARITY_NOTE,
        "censoring_note": ("cap_relief_none, cap_relief_psr and effect_off_8192 are points in the censoring-sensitivity "
                           "sense: no OFF leg carries a censored item (n_unscored == 0 on every OFF leg, asserted by "
                           "the outcomes gate; a rejected or unscorable OFF row is named by the ledger confirmation "
                           "and unparseable_rate); shift_none_matched / shift_psr_matched inherit the ON leg's "
                           "censoring width and did_matched inherits effect_on's"),
    }
    if ests is None or maps is None:
        block["findings"] = []
        return block
    renamed = {name: rename_contrasts(ests[name], name, cfgs) for name in (MAP_CAP, MAP_MATCHED)}
    block["estimands"] = renamed
    block["consistency"] = offcap_consistency(ests)
    pairs = offcap_transition_pairs()
    # the five comparator McNemar records with the registered contrasts' replicate
    # signs beside them (M1); the decomposition quotes the cap-only pair's
    mc = attach_registered_replicate_signs(all_mcnemar(legs6, pairs),
                                           {p: (renamed[m], c) for p, (m, c) in OFFCAP_PAIR_CONTRAST.items()})
    block["decomposition"] = decomposition(ests, maps, cfgs, n_resamples=n_resamples, seed=seed, ci=ci,
                                           budget_mcnemar={kind: mc[f"off512_to_off8192__{kind}"] for kind in pr.LEG_CELLS})
    block["did_matched"] = did_matched_reading(ests, cfgs)
    block["transitions"] = all_transitions(legs6, pairs)
    block["effect_off_8192"] = effect_off_8192_reading(ests, cfgs, block["transitions"])
    block["verdict_sensitivity_off_8192"] = verdict_sensitivity_off_8192(ests)
    block["sensitivity"] = {
        "mcnemar_exact": mc,
        "closed_think_capped": offcap_closed_think_capped_sensitivity(
            legs4, legs_off, ests, cfgs, n_resamples=n_resamples, seed=seed, ci=ci)}
    block["complete_case_rows"] = complete_case_rows(legs6, n_resamples=n_resamples, seed=seed, ci=ci,
                                                     pairs=offcap_complete_case_pairs())
    block["off512_to_off8192"] = off512_to_off8192_description(legs4, legs_off)
    block["gates"] = {"split_anomalies": offcap_split_anomaly_gate(legs6, ests, maps, cfgs, n_resamples=n_resamples,
                                                                   seed=seed, ci=ci)}
    findings = [block["decomposition"][kind][BRANCH_PRIMARY]["statement"] for kind in pr.LEG_CELLS]
    findings.append(block["did_matched"]["statement"])
    e8 = block["effect_off_8192"]
    findings.append(("resolved: " if e8["resolved"] else "not resolved: ") + e8["text"])
    v8 = block["verdict_sensitivity_off_8192"]
    findings.append(f"{v8['label']}: {v8['verdict']}" + (f" — {v8['reason']}" if v8.get("reason") else ""))
    if block["gates"]["split_anomalies"]["status"] == GATE_FAIL:
        findings.append("comparator anomaly re-estimation: " + str(block["gates"]["split_anomalies"]["detail"]))
    if block["cap_gate"]["status"] == GATE_FAIL:
        findings.append("comparator cap gate: " + str(block["cap_gate"]["detail"]))
    block["findings"] = findings
    return block


# --------------------------------------------------------------------------- #
# the report                                                                  #
# --------------------------------------------------------------------------- #


def analyze(gen_root: Path, sha: str, *, wave_root: Path | None = None,
            wave_sha: str = WAVE_SHA_DEFAULT, prompt_budget: Mapping[str, Any] | None = None,
            paired_ids: Sequence[str] | None = None,
            n_resamples: int = pr.DEFAULT_BOOTSTRAP, seed: int = pr.DEFAULT_BOOTSTRAP_SEED,
            ci: float = pr.DEFAULT_CI, allow_pending_gates: bool = False,
            read_eval_logs: bool = True, analysis_sha: str | None = None,
            offcap_sha: str | None = None, offcap_gen_root: Path | None = None,
            prompt_budget_off: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Verify -> population -> ledger -> gates -> estimand under both branches
    with sharp bounds -> the two decidability checks -> verdict.
    No estimand runs unless every leg is DONE (never PARTIAL) with a verified
    outcome record and the four attempted-id sets are the same 200; no word is
    filed unless, in addition, the ids are the frozen paired-200 set
    (``paired_ids``; not supplied -> PENDING), both decidability checks pass
    and no gate FAILED; a PENDING gate withholds it, always named —
    ``allow_pending_gates`` waives ONLY the enumerated producer stamps Phase B
    writes (:data:`WAIVABLE_PENDING_STAMPS`); a missing operator input (the
    frozen set, the artifact) is never waived. Every leg must also be drawn
    under the experiment's fixed row and per-arm cap, with every nuisance
    parameter equal across the legs (:func:`nuisance_check`) — a leg off them
    is refused as not this experiment; before that cross-leg check every leg's
    manifest is validated POSITIVELY against the registry row and its cell
    (:func:`identity_problem`) and its scoring against the arm's think-split
    mode (:func:`think_split_problem`). ``prompt_budget`` is validated through
    the contract (:func:`checked_prompt_budget`) before anything is read.

    ``offcap_sha`` adds the two thinking-OFF comparator legs at
    cap 8192 from ``offcap_gen_root`` (default ``gen_root``) with their OFF
    artifact ``prompt_budget_off``: TWO named dicts kept apart for the whole
    run — ``legs4`` (today's four, keyed as today) and ``legs`` (plus the two
    comparator records). The GENERIC consumers (nuisance, population, the
    cap / outcomes / declaration gates, controls) take ``legs``; every
    ESTIMAND consumer takes an explicit four-key mapping (:func:`three_mappings`)
    — never the six-leg dict — and the legacy blocks are computed over ``legs4`` /
    MAP_ORIG exactly as without the flag. Absent, every block is byte-identical
    to today's."""
    sha = sha.removeprefix(OUT_SUFFIX_PREFIX)
    prompt_budget = checked_prompt_budget(prompt_budget)
    legs4 = collect_legs(gen_root, sha, prompt_budget=prompt_budget, read_eval_logs=read_eval_logs)
    offcap = offcap_sha is not None
    legs_off: dict[str, dict[str, Any]] | None = None
    if offcap:
        offcap_sha = str(offcap_sha).removeprefix(OFFCAP_OUT_SUFFIX_PREFIX)
        offcap_gen_root = Path(offcap_gen_root) if offcap_gen_root is not None else Path(gen_root)
        prompt_budget_off = checked_prompt_budget_off(prompt_budget_off)
        legs_off = collect_offcap_legs(offcap_gen_root, offcap_sha, prompt_budget_off=prompt_budget_off,
                                       read_eval_logs=read_eval_logs)
    # every leg the GENERIC consumers run over: the four, plus the two comparator
    # records under the flag (the estimand consumers take explicit mappings)
    legs: dict[str, dict[str, Any]] = {**legs4, **legs_off} if offcap else legs4
    # Every nuisance parameter equal across the legs — a leg off the
    # others is not this experiment (refused from the estimand; its summary
    # stays in the controls table)
    nuisance, refuse = nuisance_check(legs)
    for k, reason in refuse.items():
        legs[k].setdefault("nuisance_refused", reason)
        if not legs[k].get("refused") and not legs[k].get("missing"):
            legs[k]["refused"] = reason
    object_identity: dict[str, Any] | None = None
    if offcap:
        # M3: the fitted object's artifact digest must agree within every cell across its
        # three configurations — a changed object under the same cell name passed every
        # other admission check; the record is filed under offcap.object_identity
        object_identity, refuse_obj = object_identity_check(legs)
        for k, reason in refuse_obj.items():
            legs[k].setdefault("object_identity_refused", reason)
            if not legs[k].get("refused") and not legs[k].get("missing"):
                legs[k]["refused"] = reason
    code = analysis_code_provenance(analysis_sha)
    report: dict[str, Any] = {
        # the label is what the VALIDATED manifests say, never the tuple's slug alone —
        # over EVERY leg read (six under the comparator), not the 2x2's four alone
        "model": model_label(legs), "task": TASK, "sha": sha,
        # the producer/store pin (`sha`) and the analysis code's sha are two commits
        "analysis_code_sha": code["sha"], "analysis_code": code,
        "gen_root": str(gen_root),
        # the comparator's store pin and root beside them — three shas, kept apart
        **({"offcap_sha": offcap_sha, "offcap_gen_root": str(offcap_gen_root)} if offcap else {}),
        "estimand": ("conditional accuracy with each leg's OWN denominator over the ONE common "
                     "attempted set; effect_a = acc(psr,a) - acc(none,a); shift_c = acc(c,on) - "
                     "acc(c,off); DiD = effect_on - effect_off; B replicates, one draw per replicate "
                     "applied to all four legs carrying each leg's mask; percentile CIs; sharp "
                     "contrast bounds from leg-wise adverse imputations of the censored items"),
        "bootstrap": {"B": n_resamples, "seed": seed, "level": ci},
        "sampling_deviation": (f"k = 1 at the model's NON-thinking card {FIXED_SAMPLING.temperature}/"
                               f"{FIXED_SAMPLING.top_p}/{FIXED_SAMPLING.top_k} for BOTH arms — the "
                               "thinking-mode card is 0.6; held constant as a shared nuisance (asserted "
                               "at admission on every leg: sampling_row_problem, nuisance_check)"),
        "fixed_sampling": FIXED_SAMPLING._asdict(),
        "nuisance": nuisance,
        "legs": legs4,
        "unusable_legs": unusable_legs(legs),
    }
    population = (population_check(legs, paired_ids) if not report["unusable_legs"]
                  else _gate(GATE_FAIL, f"unusable legs: {report['unusable_legs']}", n=None,
                             reason="unusable legs", frozen_set=None, n_frozen=None))
    report["population"] = population
    report["common_attempted_ids"] = {"n": population["n"], "reason": population["reason"]}
    est: dict[str, dict[str, Any]] | None = None
    ests: dict[str, dict[str, dict[str, Any]]] | None = None
    maps: dict[str, dict[tuple[str, str], Mapping[str, Any]]] | None = None
    if population["status"] != GATE_FAIL:
        # ONE loop over the fixed mappings: MAP_ORIG alone without the
        # comparator; the two relabelled calls beside it with it — the same function,
        # the same seed, the same index draw
        maps = three_mappings(legs4, legs_off)
        ests = {name: estimands(mapping, n_resamples=n_resamples, seed=seed, ci=ci) for name, mapping in maps.items()}
        est = ests[MAP_ORIG]
        report["estimands"] = est
        report["transitions"] = all_transitions(legs4)
        mc = attach_registered_replicate_signs(all_mcnemar(legs4), {p: (est, c) for p, c in LEGACY_PAIR_CONTRAST.items()})
        caps = arm_caps(legs4)
        report["verdict_rule"] = verdict(est, mcnemar=mc, transitions=report["transitions"], caps=caps)
        report["complete_case_rows"] = complete_case_rows(legs4, n_resamples=n_resamples, seed=seed, ci=ci)
        report["selected_subset_diagnostic"] = selected_subset_diagnostic(legs4)
        report["interpretation_limits"] = interpretation_limits(est, mcnemar=mc, caps=caps)
        report["findings"] = findings(est, diagnostic=report["selected_subset_diagnostic"], caps=caps)
        # every cross-arm contrast named for what it compares, shift_psr like shift_none
        report["cross_arm_contrasts"] = cross_arm_contrasts(est, caps=caps)
        # the sensitivity checks, filed apart from the estimand and the verdict
        report["sensitivity"] = {
            "mcnemar_exact": mc,
            "closed_think_capped": closed_think_capped_sensitivity(legs4, est, n_resamples=n_resamples,
                                                                   seed=seed, ci=ci)}
    gates = instrumentation_gates(legs, est, n_resamples=n_resamples, seed=seed, ci=ci,
                                  population=population, verdict_legs=legs4)
    report["gates"] = gates
    report["controls"] = controls_table(legs)
    report["serving_parity"] = serving_parity(legs4, wave_root, wave_sha=wave_sha, n_resamples=n_resamples,
                                              seed=seed, ci=ci)
    report.setdefault("sensitivity", {})["parity"] = parity_wording(report["serving_parity"])
    # a parity drift beyond its CI is a FINDING, read before the ON numbers
    parity_lines = parity_findings(report["serving_parity"])
    report["findings"] = parity_lines + list(report.get("findings") or [])
    # -- the filed verdict
    if est is None:
        filed = {"word": None, "verdict": NOT_MEASURABLE,
                 "reason": f"the estimand did not run: {population['detail']}; "
                           f"unusable legs: {report['unusable_legs']}"}
    elif gates["failed"]:
        filed = {"word": None, "verdict": NOT_MEASURABLE,
                 "reason": "gate(s) FAILED: " + "; ".join(f"{g}: {gates[g]['detail']}" for g in gates["failed"])}
    elif gates["pending"] and (not allow_pending_gates or gates["pending_unwaivable"]):
        # a pending gate withholds the word; --allow-pending-gates waives ONLY the
        # enumerated producer stamps (WAIVABLE_PENDING_STAMPS) — a missing operator
        # input or anything else pending is named as not waivable
        unwaivable = gates["pending_unwaivable"]
        filed = {"word": None, "verdict": NOT_MEASURABLE + " (instrumentation stamps pending)",
                 "reason": "gate(s) PENDING: " + "; ".join(
                     f"{g}: {gates[g].get('detail') or gates[g].get('pending')}" for g in gates["pending"])
                 + (" — not waivable by --allow-pending-gates (not a producer stamp Phase B writes): "
                    + "; ".join(f"{it['gate']}: {it['detail']}" for it in unwaivable)
                    if allow_pending_gates and unwaivable else ""),
                 "pending_unwaivable": unwaivable}
    else:
        v = report["verdict_rule"]
        filed = {"word": v["verdict"] if v["verdict"] in WORDS else None, "verdict": v["verdict"],
                 "reason": v["reason"], "statement": v["statement"],
                 "gates_pending": gates["pending"]}
    report["verdict"] = filed
    top: list[str] = list(parity_lines)          # a serving difference comes before any ON reading
    if est is not None:
        v = report["verdict_rule"]
        if not v["agree_branches"]:
            top.append("the two scoring branches DISAGREE (D11) — top artifact item: " + str(v["word_by_branch"]))
        if not v["decidable_bounds"]:
            top.append("the identified set crosses a classification boundary (D16 P1-a): " + str(v["reason"]))
    top.extend(f for f in report["findings"] if f not in parity_lines)
    report["top_items"] = top
    if offcap:
        assert legs_off is not None
        report["offcap"] = offcap_block(legs4=legs4, legs_off=legs_off, legs6=legs, ests=ests, maps=maps,
                                        prompt_budget_off=prompt_budget_off, sha=sha, offcap_sha=str(offcap_sha),
                                        analysis_code_sha=code["sha"], n_resamples=n_resamples, seed=seed, ci=ci,
                                        object_identity=object_identity)
    return report


# --------------------------------------------------------------------------- #
# figures (house style = make_audit_figs.py's palette)                        #
# --------------------------------------------------------------------------- #

#: The dataviz reference palette (light surface) the audit figures use.
SURFACE, INK, INK2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAT = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100"}
#: hue = cell kind — slot 1 blue `none`, slot 2 orange `psr`; hatch = thinking ON
KIND_COLOR = {"none": CAT["blue"], "psr": CAT["orange"]}
ON_HATCH = "////"
DPI = 200


def _mpl():
    try:
        import matplotlib
    except ImportError as e:    # pragma: no cover - venv-dependent
        raise TransferAnalysisError(
            "matplotlib is not importable in this venv — run with --no-figs, or use a venv that "
            "carries it (the figures are not optional for the results doc)") from e
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "text.color": INK, "text.parse_math": False,
        "axes.labelcolor": INK2, "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
        "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-", "hatch.linewidth": 0.7,
        "pdf.fonttype": 42})
    return plt, mcolors, Patch


def _recessive(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="both", length=0, pad=4)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)


def _bar_kw(mcolors, fc: str, on: bool) -> dict[str, Any]:
    if on:
        return {"facecolor": mcolors.to_rgba(fc, 0.22), "edgecolor": fc, "hatch": ON_HATCH,
                "linewidth": 0.8, "zorder": 3}
    return {"facecolor": fc, "edgecolor": "none", "linewidth": 0, "zorder": 3}


def _save(fig, out_dir: Path, stem: str) -> list[Path]:
    """PNG + PDF of ``fig`` under ``stem``; the figure is closed afterwards."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"
    # tight: a below-axes legend or a long title is never clipped at the edge
    fig.savefig(png, dpi=DPI, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return [png, pdf]


def _num(numbers: list[dict[str, Any]], name: str, value: Any, source: str) -> None:
    numbers.append({"name": name, "value": value, "source": source})


def report_path(report: Mapping[str, Any], path: str) -> Any:
    """The value at a dotted report-JSON path — dict keys by name, list entries
    by integer index — the ONE spelling the comparator sidecar's ``source``
    uses (the analysis review's L1). An unresolvable path is a refusal naming
    the part that failed, never a silently null record."""
    node: Any = report
    for part in path.split("."):
        if isinstance(node, Mapping):
            if part not in node:
                raise TransferAnalysisError(f"sidecar source {path!r}: no key {part!r} at that level of the report")
            node = node[part]
        elif isinstance(node, (list, tuple)):
            try:
                node = node[int(part)]
            except (ValueError, IndexError) as e:
                raise TransferAnalysisError(f"sidecar source {path!r}: {part!r} does not index the list there ({e})") from e
        else:
            raise TransferAnalysisError(f"sidecar source {path!r}: {part!r} descends into a leaf ({node!r})")
    return node


def _same_leaf(a: Any, b: Any) -> bool:
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _num_at(numbers: list[dict[str, Any]], report: Mapping[str, Any], name: str, source: str | Sequence[str], value: Any,
            *, transform: str | None = None) -> None:
    """A comparator-sidecar record (L1): ``source`` is a report-JSON path (or
    the list of paths a transform reads), each RESOLVED in ``report`` here;
    ``transform`` names any arithmetic between the source leaves and the value
    (``None`` = the leaf as filed, in which case the value must BE that leaf —
    a path pointing at another, valid leaf is refused by name)."""
    paths = [source] if isinstance(source, str) else list(source)
    resolved = [report_path(report, p) for p in paths]
    if transform is None:
        if len(paths) != 1:
            raise TransferAnalysisError(f"sidecar record {name!r}: an untransformed record reads exactly one path, got {paths}")
        if not _same_leaf(value, resolved[0]):
            raise TransferAnalysisError(
                f"sidecar record {name!r}: the plotted value {value!r} is not the leaf at {paths[0]!r} ({resolved[0]!r})")
    numbers.append({"name": name, "value": value, "source": source, "transform": transform})


def _cfg_leg_path(kind: str, cfg: str) -> str:
    """The report path of the leg a configuration names (:func:`_cfg_leg`'s spelling)."""
    cell = KIND_CELL[kind]
    if cfg == CFG_OFF_OVERRIDE:
        return f"offcap.legs.{leg_key(cell, C.THINKING_OFF, cap_override=True)}"
    return f"legs.{leg_key(cell, C.THINKING_OFF if cfg == CFG_OFF else C.THINKING_ON)}"


def _cfg_acc_path(report: Mapping[str, Any], kind: str, cfg: str, branch: str = BRANCH_PRIMARY) -> str:
    """The report path of the configuration's ``acc_by_leg`` record (:func:`_cfg_acc`'s spelling)."""
    if cfg == CFG_OFF_OVERRIDE:
        _off, off8, _on = _cfg_labels(report)
        return f"offcap.estimands.{MAP_CAP}.{branch}.acc_by_leg.{kind}/{off8}"
    arm = C.THINKING_OFF if cfg == CFG_OFF else C.THINKING_ON
    return f"estimands.{branch}.acc_by_leg.{pr.leg_label((kind, arm))}"


def _hue_handles(Patch) -> list:
    """The two cell-kind (hue) legend handles every figure shares."""
    return [Patch(facecolor=KIND_COLOR["none"], edgecolor="none", label="none (unsteered)"),
            Patch(facecolor=KIND_COLOR["psr"], edgecolor="none", label="safety_refusal__psr_kl_all")]


def _kind_handles(Patch, mcolors, caps: Mapping[str, Any] | None = None) -> list:
    """The three legend handles every per-leg figure wears: the two cell kinds
    (hue) and thinking ON (hatch) — the arm named with its cap."""
    on = arm_cap_label(C.THINKING_ON, caps)
    return [*_hue_handles(Patch),
            Patch(facecolor=mcolors.to_rgba(INK2, 0.18), edgecolor=INK2, hatch=ON_HATCH,
                  label=f"thinking ON ({on})" if "/" in on else "thinking ON")]


def _legend(Patch, mcolors, ax, *, loc: str = "upper left", anchor: tuple = (0.0, 1.0),
            caps: Mapping[str, Any] | None = None):
    return ax.legend(handles=_kind_handles(Patch, mcolors, caps), loc=loc, bbox_to_anchor=anchor,
                     frameon=False, fontsize=7.4, handlelength=1.6, handleheight=0.9, ncol=3,
                     columnspacing=1.8)


def _leg_tick(kind: str, arm: str, caps: Mapping[str, Any] | None) -> str:
    """A per-leg tick label: the cell over the arm WITH its cap (``none`` / ``OFF/512``)."""
    return f"{kind}\n{arm_cap_label(arm, caps)}"


def _titles(fig, title: str, subtitle: str, *, sub_y: float = 0.895,
            linespacing: float | None = None) -> None:
    """The house title + subtitle block every figure wears."""
    fig.text(0.035, 0.965, title, fontsize=12.5, fontweight="bold", color=INK, ha="left", va="top")
    extra = {"linespacing": linespacing} if linespacing is not None else {}
    fig.text(0.035, sub_y, subtitle, fontsize=7.6, color=INK2, ha="left", va="top", **extra)


def _leg_slots(gap: float):
    """``(x, kind, arm)`` for the four legs in plotting order — cells side by
    side, the two arms within a cell, ``gap`` between cells."""
    x = 0.0
    for kind in pr.LEG_CELLS:
        for arm in ARMS:
            yield x, kind, arm
            x += 1.0
        x += gap


def fig_acc_2x2(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, mcolors, Patch = _mpl()
    est = report["estimands"][BRANCH_PRIMARY]
    numbers: list[dict[str, Any]] = []
    caps = arm_caps(report["legs"])
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    fig.subplots_adjust(left=0.1, right=0.98, top=0.78, bottom=0.27)
    xs, labels = [], []
    for x, kind, arm in _leg_slots(0.5):
        leg = report["legs"][leg_key(KIND_CELL[kind], arm)]
        s = leg["summary"]
        a = est["acc_by_leg"][pr.leg_label((kind, arm))]
        acc, n_scored = s["acc"], s["n_scored"]
        se = math.sqrt(acc * (1 - acc) / n_scored) if n_scored else 0.0
        ax.bar([x], [acc], width=0.7, **_bar_kw(mcolors, KIND_COLOR[kind], arm == C.THINKING_ON))
        ax.plot([x, x], [acc - se, acc + se], color=INK2, lw=0.9, zorder=6)
        if a["acc_minus"] != a["acc_plus"]:
            ax.plot([x + 0.42, x + 0.42], [a["acc_minus"], a["acc_plus"]], color=INK, lw=1.4, zorder=7)
            for y in (a["acc_minus"], a["acc_plus"]):
                ax.plot([x + 0.36, x + 0.48], [y, y], color=INK, lw=1.0, zorder=7)
        ax.text(x, acc + se + 0.012, f"{acc:.3f}\nn_unscored={s['n_unscored']}", ha="center", va="bottom",
                fontsize=6.8, color=INK2)
        src = f"summary/{C.marker_filename(TASK)} ({leg['leg']})"
        _num(numbers, f"acc[{kind}/{arm}]", acc, src)
        _num(numbers, f"se[{kind}/{arm}]", se, "sqrt(acc(1-acc)/n_scored) from " + src)
        _num(numbers, f"n_unscored[{kind}/{arm}]", s["n_unscored"], src)
        _num(numbers, f"acc_minus[{kind}/{arm}]", a["acc_minus"], "outcomes record (censored -> wrong)")
        _num(numbers, f"acc_plus[{kind}/{arm}]", a["acc_plus"], "outcomes record (censored -> right)")
        xs.append(x)
        labels.append(_leg_tick(kind, arm, caps))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("GSM8K accuracy (summary acc)")
    _recessive(ax)
    # the legend sits BELOW the tick labels: inside the axes it overlapped the
    # bar value labels (the validation's PNG on the censored run)
    _legend(Patch, mcolors, ax, loc="upper center", anchor=(0.5, -0.20), caps=caps)
    _titles(fig, "GSM8K accuracy, the 2x2 (Qwen3.5-4B, k = 1)",
            "whiskers = ±1 per-item SE; the black bracket beside a bar is the censoring interval\n"
            "(unterminated -> wrong … -> right) where it differs from the bar", linespacing=1.3)
    return fig, numbers


#: the rule's breakpoints on the effect figure and each one's line style
BREAKPOINT_STYLE: Mapping[str, str] = {"0": "-", "OFF CI lo": ":", "OFF CI hi": ":", "+|E_off|": "--", "-|E_off|": "--"}


def breakpoint_labels(e_off: float, lo: float, hi: float) -> dict[float, list[str]]:
    """The breakpoints grouped by their y (one label per distinct line): with
    effect_off = 0 the 0 and the two ±|E_off| lines coincide and read as ONE
    label instead of three printed on top of each other."""
    groups: dict[float, list[str]] = {}
    for name, y in (("0", 0.0), ("OFF CI lo", lo), ("OFF CI hi", hi), ("+|E_off|", abs(e_off)),
                    ("-|E_off|", -abs(e_off))):
        key = next((k for k in groups if math.isclose(k, y, rel_tol=0.0, abs_tol=1e-12)), y)
        groups.setdefault(key, []).append(name)
    return groups


def _fold_e_off(names: Sequence[str]) -> list[str]:
    """``+|E_off|`` and ``-|E_off|`` on the same line read ``±|E_off|``."""
    if "+|E_off|" in names and "-|E_off|" in names:
        return [n for n in names if n not in ("+|E_off|", "-|E_off|")] + ["±|E_off|"]
    return list(names)


def fig_effect_by_arm(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, mcolors, Patch = _mpl()
    numbers: list[dict[str, Any]] = []
    p = report["estimands"][BRANCH_PRIMARY]["stats"]
    q = report["estimands"][BRANCH_SCORE_AS_IS]["stats"]
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.76, bottom=0.30)
    lo, hi = p["effect_off"]["ci_low"], p["effect_off"]["ci_high"]
    e_off = p["effect_off"]["point"]
    # the y-range is set from EVERYTHING drawn (both branches' CIs and bounds,
    # the band, the breakpoints) with headroom for the value labels, so no
    # label or whisker lands outside the frame or on the title block
    ys = [0.0, lo, hi, abs(e_off), -abs(e_off)]
    for c in ("effect_off", "effect_on", "did"):
        for s in (p[c], q[c]):
            ys += [s["ci_low"], s["ci_high"], s["bound_low"], s["bound_high"]]
    y_lo, y_hi = min(ys), max(ys)
    span = (y_hi - y_lo) or 0.02
    ax.set_ylim(y_lo - 0.10 * span, y_hi + 0.24 * span)
    label_dy = 0.04 * span
    ax.axhspan(lo, hi, color=mcolors.to_rgba(CAT["blue"], 0.12), zorder=1, lw=0)
    for y, names in breakpoint_labels(e_off, lo, hi).items():
        for ls in {BREAKPOINT_STYLE[n] for n in names}:
            ax.axhline(y, color=BASELINE, lw=0.7, ls=ls, zorder=2)
        ax.text(2.55, y, " = ".join(_fold_e_off(names)), fontsize=6.2, color=MUTED, va="center", ha="left")
    xs = {"effect_off": 0.0, "effect_on": 1.0, "did": 2.0}
    for c, x in xs.items():
        sp, sq = p[c], q[c]
        # sharp identification bracket behind the marks
        if sp["bound_low"] != sp["bound_high"]:
            ax.add_patch(plt.Rectangle((x - 0.32, sp["bound_low"]), 0.64, sp["bound_high"] - sp["bound_low"],
                                       facecolor=mcolors.to_rgba(INK2, 0.10), edgecolor=INK2, lw=0.8, zorder=2))
        ax.plot([x - 0.08, x - 0.08], [sp["ci_low"], sp["ci_high"]], color=INK, lw=1.0, zorder=5)
        ax.plot([x - 0.08], [sp["point"]], marker="o", ms=6, color=CAT["orange"], mec=INK, mew=0.6, zorder=6)
        ax.plot([x + 0.08, x + 0.08], [sq["ci_low"], sq["ci_high"]], color=INK2, lw=1.0, zorder=5)
        ax.plot([x + 0.08], [sq["point"]], marker="o", ms=6, mfc="none", mec=CAT["orange"], mew=1.2, zorder=6)
        ax.text(x, max(sp["ci_high"], sq["ci_high"]) + label_dy, f"{_pp(sp['point'])}", ha="center",
                va="bottom", fontsize=6.8, color=INK2, clip_on=True)
        for b, s in ((BRANCH_PRIMARY, sp), (BRANCH_SCORE_AS_IS, sq)):
            for f in STAT_FIELDS:
                _num(numbers, f"{c}.{f}[{b}]", s[f], f"pairing.paired_bootstrap_conditional_2x2 ({b} branch)")
    ax.set_xticks(list(xs.values()))
    # every arm named with its cap, and the DiD for what it is: a render+cap contrast
    caps = arm_caps(report["legs"])
    off, on = arm_cap_label(C.THINKING_OFF, caps), arm_cap_label(C.THINKING_ON, caps)
    ax.set_xticklabels([f"effect_off\n(psr − none, {off})", f"effect_on\n(psr − none, {on})",
                        f"DiD, {RENDER_CAP_CONTRAST_LABEL}\n({on} − {off})"])
    ax.set_xlim(-0.6, 3.0)
    ax.set_ylabel("accuracy difference")
    _recessive(ax)
    handles = [plt.Line2D([], [], marker="o", color=CAT["orange"], mec=INK, ls="", label="primary (unterminated → unscored)"),
               plt.Line2D([], [], marker="o", mfc="none", mec=CAT["orange"], ls="", label="score-as-is (unterminated → wrong)"),
               Patch(facecolor=mcolors.to_rgba(INK2, 0.10), edgecolor=INK2, label=BOUND_LABEL),
               Patch(facecolor=mcolors.to_rgba(CAT["blue"], 0.12), edgecolor="none", label="effect_off 95 % CI (tolerance band)")]
    # below the axes (inside, it sat on the marks and their whiskers)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20), frameon=False,
              fontsize=6.8, ncol=2, columnspacing=1.6)
    # the verdict is a sentence, not a title suffix: the title stays short (it
    # was clipped at the right edge), the verdict leads the subtitle, wrapped
    verdict = textwrap.fill(f"verdict: {report['verdict']['verdict']}", width=110)
    _titles(fig, "Steering effect by arm",
            f"{verdict}\nmarks = point ± {round(100 * report['bootstrap']['level'])} % percentile CI "
            f"(B = {report['bootstrap']['B']:,}, seed {report['bootstrap']['seed']}); breakpoints of the rule\n"
            "(0, the band's edges, ±|effect_off|) as faint lines so a bracket crossing one is legible",
            sub_y=0.905, linespacing=1.3)
    _num(numbers, "verdict", report["verdict"]["verdict"], "the rule of plan §6.2 + the two decidability checks")
    return fig, numbers


CAP_GATE_LABEL = "ON-only 2 % gate"


def fig_controls(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, mcolors, Patch = _mpl()
    numbers: list[dict[str, Any]] = []
    panels = [("mean_completion_tokens", "mean completion tokens"), ("p99_completion_tokens", "p99 completion tokens"),
              ("at_cap_rate", "at-cap rate (summary)"), ("n_think_block_present", "think block present (n)"),
              ("n_think_block_unterminated", "think block unterminated (n)"), ("unparseable_rate", "unparseable rate")]
    caps = arm_caps(report["legs"])
    fig, axes = plt.subplots(2, 3, figsize=(8.4, 5.0))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.1, hspace=0.6, wspace=0.3)
    for ax, (field, title) in zip(axes.flat, panels):
        xs, labels = [], []
        for x, kind, arm in _leg_slots(0.4):
            leg = report["legs"][leg_key(KIND_CELL[kind], arm)]
            v = leg["summary"].get(field)
            src = f"summary/{C.marker_filename(TASK)} ({leg['leg']})"
            _num(numbers, f"{field}[{kind}/{arm}]", v, src)
            if v is not None:
                ax.bar([x], [v], width=0.7, **_bar_kw(mcolors, KIND_COLOR[kind], arm == C.THINKING_ON))
            else:
                ax.text(x, 0, "null", ha="center", va="bottom", fontsize=6.5, color=MUTED)
            xs.append(x)
            labels.append(_leg_tick(kind, arm, caps))
        if field == "at_cap_rate":
            # the gate binds the ON legs only: a segment over each ON
            # bar, never a line across the OFF bars it does not judge
            on_xs = [x for x, _kind, arm in _leg_slots(0.4) if arm == C.THINKING_ON]
            for x in on_xs:
                ax.hlines(CAP_GATE_AT_CAP_RATE, x - 0.42, x + 0.42, color=CAT["orange"], lw=0.9,
                          ls="--", zorder=4)
            ax.text(on_xs[0], CAP_GATE_AT_CAP_RATE, CAP_GATE_LABEL, fontsize=6.2, color=CAT["orange"],
                    va="bottom", ha="center")
        top = max([v for v in (ax.get_ylim()[1],)] + [0.0])
        ax.set_ylim(0, top * 1.25 if top > 0 else 1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=6.6)
        ax.set_title(title, fontsize=8.2, color=INK, loc="left")
        _recessive(ax)
    fig.legend(handles=_kind_handles(Patch, mcolors, caps), loc="upper right", bbox_to_anchor=(0.98, 0.9),
               frameon=False, fontsize=7.4, ncol=3, handlelength=1.6, handleheight=0.9, columnspacing=1.8)
    _titles(fig, "Per-arm controls (from the summary payloads)",
            "every value is the summary's own count / rate over the ATTEMPTED draws — never re-derived "
            "from rows", sub_y=0.905)
    return fig, numbers


def fig_transitions(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mcolors, _Patch = _mpl()
    numbers: list[dict[str, Any]] = []
    tr = report["transitions"]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.16, wspace=0.25)
    keys = TRANSITION_KEYS
    # arms named with their caps; the OFF -> ON panel is a render+cap contrast
    caps = arm_caps(report["legs"])
    off, on = arm_cap_label(C.THINKING_OFF, caps), arm_cap_label(C.THINKING_ON, caps)
    groups = [("off_to_on", axes[0], f"{off} → {on}, per cell ({RENDER_CAP_CONTRAST_LABEL})",
               [("none", "none", KIND_COLOR["none"]), ("psr", "psr", KIND_COLOR["psr"])]),
              ("none_to_psr", axes[1], "none → psr, per arm",
               [(C.THINKING_OFF, off, CAT["aqua"]), (C.THINKING_ON, on, CAT["yellow"])])]
    for prefix, ax, title, series in groups:
        bars: list[tuple[float, int]] = []
        for j, (name, shown, color) in enumerate(series):
            t = tr[f"{prefix}__{name}"]
            vals = [t["counts"][k] for k in keys]
            xs = [i + (j - 0.5) * 0.36 for i in range(len(keys))]
            p = (t["mcnemar"] or {}).get("p_value")
            ax.bar(xs, vals, width=0.34, facecolor=color, edgecolor="none", zorder=3,
                   label=f"{shown} — McNemar p = {'n/a' if p is None else f'{p:.3g}'} (complete-case)")
            bars += list(zip(xs, vals))
            for k, v in zip(keys, vals):
                _num(numbers, f"{prefix}__{name}[{k}]", v, "outcomes records (per-item transitions)")
            _num(numbers, f"{prefix}__{name}.mcnemar_p", p, pr.COMPLETE_CASE_LABEL + " McNemar")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.3)
        # the count on every bar: the decisive 5 vs 0 is unreadable beside ~190 unchanged
        dy = 0.012 * ax.get_ylim()[1]
        for x, v in bars:
            ax.text(x, v + dy, str(v), ha="center", va="bottom", fontsize=6.2, color=INK2, zorder=7)
        ax.set_title(title, fontsize=8.6, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=6.8, loc="upper right")
        _recessive(ax)
    _titles(fig, "Per-item transitions (C = correct, I = incorrect, U = unscored)",
            "counts over the common attempted set; McNemar over the items both legs scored — "
            "complete-case, SECONDARY to the conditional estimand", sub_y=0.885)
    return fig, numbers


def fig_parity(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mcolors, _Patch = _mpl()
    numbers: list[dict[str, Any]] = []
    par = report["serving_parity"]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    fig.subplots_adjust(left=0.12, right=0.98, top=0.78, bottom=0.18)
    ax.axhline(0.0, color=BASELINE, lw=0.8, zorder=2)
    # the y-range from EVERYTHING drawn with headroom for the two-line labels,
    # so an annotation never leaves the axes and lands on the subtitle
    ys = [0.0]
    for kind in pr.LEG_CELLS:
        rec = par["cells"].get(KIND_CELL[kind], {})
        if rec.get("delta_fresh_minus_wave") is not None:
            ys += [rec["delta_fresh_minus_wave"], *rec["ci"]]
    y_lo, y_hi = min(ys), max(ys)
    span = (y_hi - y_lo) or 0.02
    ax.set_ylim(y_lo - 0.15 * span, y_hi + 0.6 * span)
    label_dy = 0.04 * span
    for i, kind in enumerate(pr.LEG_CELLS):
        rec = par["cells"].get(KIND_CELL[kind], {})
        d = rec.get("delta_fresh_minus_wave")
        if d is None:
            ax.text(i, 0.0, rec.get("missing") or rec.get("refused") or "n/a", ha="center", fontsize=6.5, color=MUTED)
            continue
        lo, hi = rec["ci"]
        ax.plot([i, i], [lo, hi], color=INK, lw=1.0, zorder=5)
        ax.plot([i], [d], marker="o", ms=7, color=KIND_COLOR[kind], mec=INK, mew=0.6, zorder=6)
        ax.text(i, hi + label_dy, f"{_pp(d)}\n{rec['regime_label'].split(' (')[0]}", ha="center", va="bottom",
                fontsize=6.6, color=INK2, clip_on=True)
        _num(numbers, f"parity_delta[{kind}]", d, f"pairing.compare_sides fresh OFF vs wave {par['wave_sha']} rows")
        _num(numbers, f"parity_ci[{kind}]", [lo, hi], "paired bootstrap over the matched items")
        _num(numbers, f"parity_regime_label[{kind}]", rec["regime_label"], "manifest tinycap.served, both stores")
    ax.set_xticks(range(len(pr.LEG_CELLS)))
    off = arm_cap_label(C.THINKING_OFF, arm_caps(report["legs"]))      # both sides are the OFF render at 512
    ax.set_xticklabels([f"{k}, {off}" for k in pr.LEG_CELLS])
    ax.set_xlim(-0.6, len(pr.LEG_CELLS) - 0.4)
    ax.set_ylabel("fresh − wave accuracy")
    _recessive(ax)
    _titles(fig, f"Serving parity: fresh OFF legs vs the wave's {par['wave_sha']} stores",
            "paired delta with 95 % CI over the same 200 items (complete-case is exact under OFF); "
            "regime label from the served window / S / client width\n"
            f"{report.get('sensitivity', {}).get('parity', {}).get('wording', '')}", sub_y=0.905,
            linespacing=1.3)
    return fig, numbers


# -- the comparator's figures: the final_e1_* house style — blue
#    `none`, orange `psr`, hatch = the ON render; the OFF/8192 configuration is the
#    cell hue at reduced opacity with a solid edge; every arm named with its cap;
#    one figure per finding; the varies label on every contrast axis


def _cfg_labels(report: Mapping[str, Any]) -> tuple[str, str, str]:
    cfgs = report["offcap"]["configurations"]
    return cfgs[CFG_OFF]["label"], cfgs[CFG_OFF_OVERRIDE]["label"], cfgs[CFG_ON]["label"]


def _cfg_bar_kw(mcolors, fc: str, cfg: str) -> dict[str, Any]:
    """The bar treatment per configuration: OFF/512 solid, OFF/8192 the hue at
    0.55 with a solid edge, ON/8192 hatched (the house `_bar_kw` ON style)."""
    if cfg == CFG_ON:
        return _bar_kw(mcolors, fc, True)
    if cfg == CFG_OFF_OVERRIDE:
        return {"facecolor": mcolors.to_rgba(fc, 0.55), "edgecolor": fc, "linewidth": 0.8, "zorder": 3}
    return _bar_kw(mcolors, fc, False)


def _cfg_handles(Patch, mcolors, report: Mapping[str, Any]) -> list:
    """Legend handles: the two cell kinds (hue) and the three configurations."""
    off, off8, on = _cfg_labels(report)
    return [*_hue_handles(Patch),
            Patch(facecolor=INK2, edgecolor="none", label=f"thinking OFF, table cap ({off})"),
            Patch(facecolor=mcolors.to_rgba(INK2, 0.55), edgecolor=INK2, label=f"thinking OFF, raised cap ({off8})"),
            Patch(facecolor=mcolors.to_rgba(INK2, 0.18), edgecolor=INK2, hatch=ON_HATCH, label=f"thinking ON ({on})")]


def _cfg_slots(gap: float):
    """``(x, kind, cfg)`` for the six configurations in plotting order — cells
    side by side, OFF/512, OFF/8192, ON/8192 within a cell."""
    x = 0.0
    for kind in pr.LEG_CELLS:
        for cfg in (CFG_OFF, CFG_OFF_OVERRIDE, CFG_ON):
            yield x, kind, cfg
            x += 1.0
        x += gap


def _cfg_leg(report: Mapping[str, Any], kind: str, cfg: str) -> Mapping[str, Any]:
    cell = KIND_CELL[kind]
    if cfg == CFG_OFF_OVERRIDE:
        return report["offcap"]["legs"][leg_key(cell, C.THINKING_OFF, cap_override=True)]
    return report["legs"][leg_key(cell, C.THINKING_OFF if cfg == CFG_OFF else C.THINKING_ON)]


def _cfg_acc(report: Mapping[str, Any], kind: str, cfg: str, branch: str = BRANCH_PRIMARY) -> Mapping[str, Any]:
    """The configuration's ``acc_by_leg`` record — from MAP_ORIG for the 2x2
    legs, from MAP_CAP's ``on`` slot for the OFF/8192 leg."""
    if cfg == CFG_OFF_OVERRIDE:
        _off, off8, _on = _cfg_labels(report)
        return report["offcap"]["estimands"][MAP_CAP][branch]["acc_by_leg"][f"{kind}/{off8}"]
    arm = C.THINKING_OFF if cfg == CFG_OFF else C.THINKING_ON
    return report["estimands"][branch]["acc_by_leg"][pr.leg_label((kind, arm))]


def fig_shift_decomposition(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """THE headline: per cell the OFF/512 → ON/8192 total
    shift beside its budget term (cap relief, same render) and its render term
    (the matched shift, same cap), each with its own 95 % CI whisker; the two
    terms sum to the total exactly (the identity in the subtitle)."""
    plt, mcolors, Patch = _mpl()
    numbers: list[dict[str, Any]] = []
    off, off8, on = _cfg_labels(report)
    dec = report["offcap"]["decomposition"]
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.79, bottom=0.24)
    # the tick names the cell and the term; the legend carries each term's configurations
    terms = (("total", "total", CFG_OFF), ("budget", "budget", CFG_OFF_OVERRIDE), ("render", "render", CFG_ON))
    ys = [0.0]
    xs, labels = [], []
    x = 0.0
    for kind in pr.LEG_CELLS:
        d = dec[kind][BRANCH_PRIMARY]
        for term, label, cfg in terms:
            s = d[term]
            ax.bar([x], [s["point"]], width=0.7, **_cfg_bar_kw(mcolors, KIND_COLOR[kind], cfg))
            ax.plot([x, x], [s["ci_low"], s["ci_high"]], color=INK, lw=1.0, zorder=6)
            ys += [s["ci_low"], s["ci_high"], s["point"]]
            top = max(s["ci_high"], s["point"], 0.0)
            ax.text(x, top, f"{_pp(s['point'])}", ha="center", va="bottom", fontsize=6.8, color=INK2, zorder=7)
            src = f"offcap.decomposition.{kind}.{BRANCH_PRIMARY}.{term}"
            for f in ("point", "ci_low", "ci_high"):
                _num_at(numbers, report, f"{term}.{f}[{kind}]", f"{src}.{f}", s[f])
            xs.append(x)
            labels.append(f"{kind}\n{label}")
            x += 1.0
        dsrc = f"offcap.decomposition.{kind}.{BRANCH_PRIMARY}"
        for f in ("identity_exact_full_set", "identity_replicates_to_an_ulp", "budget_share_point"):
            _num_at(numbers, report, f"{f}[{kind}]", f"{dsrc}.{f}", d[f])
        x += 0.6
    y_lo, y_hi = min(ys), max(ys)
    span = (y_hi - y_lo) or 0.02
    ax.set_ylim(y_lo - 0.12 * span, y_hi + 0.26 * span)
    ax.axhline(0.0, color=BASELINE, lw=0.8, zorder=2)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylabel("accuracy difference")
    _recessive(ax)
    handles = [*_hue_handles(Patch),
               Patch(facecolor=INK2, edgecolor="none", label=f"total: render+cap ({off} → {on})"),
               Patch(facecolor=mcolors.to_rgba(INK2, 0.55), edgecolor=INK2, label=f"budget only (OFF render, {off} → {off8})"),
               Patch(facecolor=mcolors.to_rgba(INK2, 0.18), edgecolor=INK2, hatch=ON_HATCH,
                     label=f"render only (cap {off8.split('/')[-1]}: {off8} → {on})")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False, fontsize=6.8,
              ncol=3, columnspacing=1.4, handlelength=1.6, handleheight=0.9)
    _titles(fig, f"Observed {off} → {on} shift decomposition",
            textwrap.fill(f"per cell: total = budget ({off} → {off8}, same render) + render ({off8} → {on}, same cap) — a "
                          "decomposition identity that holds exactly on the full set and in every replicate; whiskers = "
                          f"each term's own {round(100 * report['bootstrap']['level'])} % percentile CI "
                          f"(B = {report['bootstrap']['B']:,}, seed {report['bootstrap']['seed']}); the intervals do not add",
                          width=125),
            sub_y=0.905, linespacing=1.3)
    return fig, numbers


def fig_effect_by_configuration(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """``effect_off`` (512), ``effect_off_8192`` and ``effect_on`` with CIs and
    censoring brackets, both branches; the registered OFF/512 band as today,
    the 8192 reading beside it labelled sensitivity; ``did_matched`` annotated."""
    plt, mcolors, Patch = _mpl()
    numbers: list[dict[str, Any]] = []
    off, off8, on = _cfg_labels(report)
    oc = report["offcap"]
    # (name, estimands block, contrast, tick label, the block's report path)
    series = (("effect_off", report["estimands"], "effect_off", f"effect_off\n(psr − none, {off})", "estimands"),
              ("effect_off_8192", oc["estimands"][MAP_CAP], "effect_off_8192",
               f"effect_off_8192\n(psr − none, {off8}) — sensitivity", f"offcap.estimands.{MAP_CAP}"),
              ("effect_on", report["estimands"], "effect_on", f"effect_on\n(psr − none, {on})", "estimands"))
    p_off = report["estimands"][BRANCH_PRIMARY]["stats"]["effect_off"]
    lo, hi, e_off = p_off["ci_low"], p_off["ci_high"], p_off["point"]
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    fig.subplots_adjust(left=0.11, right=0.98, top=0.78, bottom=0.30)
    ys = [0.0, lo, hi, abs(e_off), -abs(e_off)]
    for _name, est, c, _label, _path in series:
        for b in BRANCHES:
            s = est[b]["stats"][c]
            ys += [s["ci_low"], s["ci_high"], s["bound_low"], s["bound_high"]]
    y_lo, y_hi = min(ys), max(ys)
    span = (y_hi - y_lo) or 0.02
    ax.set_ylim(y_lo - 0.10 * span, y_hi + 0.34 * span)          # headroom for the did_matched line
    label_dy = 0.04 * span
    ax.axhspan(lo, hi, color=mcolors.to_rgba(CAT["blue"], 0.12), zorder=1, lw=0)
    for y, names in breakpoint_labels(e_off, lo, hi).items():
        for ls in {BREAKPOINT_STYLE[n] for n in names}:
            ax.axhline(y, color=BASELINE, lw=0.7, ls=ls, zorder=2)
        ax.text(2.55, y, " = ".join(_fold_e_off(names)), fontsize=6.2, color=MUTED, va="center", ha="left")
    for x, (name, est, c, _label, est_path) in enumerate(series):
        sp, sq = est[BRANCH_PRIMARY]["stats"][c], est[BRANCH_SCORE_AS_IS]["stats"][c]
        if sp["bound_low"] != sp["bound_high"]:
            ax.add_patch(plt.Rectangle((x - 0.32, sp["bound_low"]), 0.64, sp["bound_high"] - sp["bound_low"],
                                       facecolor=mcolors.to_rgba(INK2, 0.10), edgecolor=INK2, lw=0.8, zorder=2))
        ax.plot([x - 0.08, x - 0.08], [sp["ci_low"], sp["ci_high"]], color=INK, lw=1.0, zorder=5)
        ax.plot([x - 0.08], [sp["point"]], marker="o", ms=6, color=CAT["orange"], mec=INK, mew=0.6, zorder=6)
        ax.plot([x + 0.08, x + 0.08], [sq["ci_low"], sq["ci_high"]], color=INK2, lw=1.0, zorder=5)
        ax.plot([x + 0.08], [sq["point"]], marker="o", ms=6, mfc="none", mec=CAT["orange"], mew=1.2, zorder=6)
        ax.text(x, max(sp["ci_high"], sq["ci_high"]) + label_dy, f"{_pp(sp['point'])}", ha="center", va="bottom",
                fontsize=6.8, color=INK2, clip_on=True)
        for b, s in ((BRANCH_PRIMARY, sp), (BRANCH_SCORE_AS_IS, sq)):
            for f in STAT_FIELDS:
                _num_at(numbers, report, f"{name}.{f}[{b}]", f"{est_path}.{b}.stats.{c}.{f}", s[f])
    dm = oc["did_matched"]
    ax.text(0.01, 0.985, f"did_matched = {_pp(dm['point'])} [{_pp(dm['ci'][0])}, {_pp(dm['ci'][1])}] "
            f"(effect_on − effect_off_8192; {oc['contrast_varies']['did_matched']})", transform=ax.transAxes,
            ha="left", va="top", fontsize=6.4, color=INK2, zorder=7)
    _num_at(numbers, report, "did_matched.point", "offcap.did_matched.point", dm["point"])
    _num_at(numbers, report, "did_matched.ci", "offcap.did_matched.ci", dm["ci"])
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([lab for _n, _e, _c, lab, _p in series], fontsize=7.2)
    ax.set_xlim(-0.6, 3.0)
    ax.set_ylabel("accuracy difference")
    _recessive(ax)
    handles = [plt.Line2D([], [], marker="o", color=CAT["orange"], mec=INK, ls="", label="primary (unterminated → unscored)"),
               plt.Line2D([], [], marker="o", mfc="none", mec=CAT["orange"], ls="", label="score-as-is (unterminated → wrong)"),
               Patch(facecolor=mcolors.to_rgba(INK2, 0.10), edgecolor=INK2, label=BOUND_LABEL),
               Patch(facecolor=mcolors.to_rgba(CAT["blue"], 0.12), edgecolor="none",
                     label=f"effect_off ({off}) 95 % CI — the registered band")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=6.8, ncol=2,
              columnspacing=1.6)
    v8 = oc["verdict_sensitivity_off_8192"]
    _titles(fig, "Steering effect by configuration",
            textwrap.fill(f"registered verdict (rule keyed on {off}): {report['verdict']['verdict']}; "
                          f"sensitivity (rule keyed on {off8}): {v8['verdict']}", width=118)
            + f"\nmarks = point ± {round(100 * report['bootstrap']['level'])} % percentile CI; the OFF/8192 reading "
              "is a sensitivity, never the registered input", sub_y=0.905, linespacing=1.3)
    _num_at(numbers, report, "verdict", "verdict.verdict", report["verdict"]["verdict"])
    _num_at(numbers, report, "verdict_sensitivity_off_8192", "offcap.verdict_sensitivity_off_8192.verdict", v8["verdict"])
    return fig, numbers


def fig_acc_2x3(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """Six bars, cell × {OFF/512, OFF/8192, ON/8192}: summary acc, per-item SE,
    n_unscored annotated, censoring brackets where the interval differs."""
    plt, mcolors, Patch = _mpl()
    numbers: list[dict[str, Any]] = []
    cfgs = report["offcap"]["configurations"]
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.79, bottom=0.30)
    xs, labels = [], []
    for x, kind, cfg in _cfg_slots(0.5):
        leg = _cfg_leg(report, kind, cfg)
        s = leg["summary"]
        a = _cfg_acc(report, kind, cfg)
        acc, n_scored = s["acc"], s["n_scored"]
        se = math.sqrt(acc * (1 - acc) / n_scored) if n_scored else 0.0
        ax.bar([x], [acc], width=0.7, **_cfg_bar_kw(mcolors, KIND_COLOR[kind], cfg))
        ax.plot([x, x], [acc - se, acc + se], color=INK2, lw=0.9, zorder=6)
        if a["acc_minus"] != a["acc_plus"]:
            ax.plot([x + 0.42, x + 0.42], [a["acc_minus"], a["acc_plus"]], color=INK, lw=1.4, zorder=7)
            for y in (a["acc_minus"], a["acc_plus"]):
                ax.plot([x + 0.36, x + 0.48], [y, y], color=INK, lw=1.0, zorder=7)
        ax.text(x, acc + se + 0.012, f"{acc:.3f}\nn_unscored={s['n_unscored']}", ha="center", va="bottom",
                fontsize=6.6, color=INK2)
        label = cfgs[cfg]["label"]
        lp, ap = _cfg_leg_path(kind, cfg), _cfg_acc_path(report, kind, cfg)
        _num_at(numbers, report, f"acc[{kind}/{label}]", f"{lp}.summary.acc", acc)
        _num_at(numbers, report, f"se[{kind}/{label}]", [f"{lp}.summary.acc", f"{lp}.summary.n_scored"], se,
                transform="sqrt(acc · (1 − acc) / n_scored)")
        _num_at(numbers, report, f"n_unscored[{kind}/{label}]", f"{lp}.summary.n_unscored", s["n_unscored"])
        _num_at(numbers, report, f"acc_minus[{kind}/{label}]", f"{ap}.acc_minus", a["acc_minus"])
        _num_at(numbers, report, f"acc_plus[{kind}/{label}]", f"{ap}.acc_plus", a["acc_plus"])
        xs.append(x)
        labels.append(f"{kind}\n{label}")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.0)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("GSM8K accuracy (summary acc)")
    _recessive(ax)
    ax.legend(handles=_cfg_handles(Patch, mcolors, report), loc="upper center", bbox_to_anchor=(0.5, -0.20),
              frameon=False, fontsize=6.8, ncol=3, columnspacing=1.4, handlelength=1.6, handleheight=0.9)
    _titles(fig, "GSM8K accuracy, the 2x3 (Qwen3.5-4B, k = 1)",
            "cells none / psr under OFF at the table cap, OFF at the raised cap and ON; whiskers = ±1 per-item SE;\n"
            "the black bracket beside a bar is the censoring interval (unterminated -> wrong … -> right) where it differs",
            linespacing=1.3)
    return fig, numbers


def _transition_panel(ax, report: Mapping[str, Any], names: Sequence[tuple[str, str, str]], title: str,
                      numbers: list[dict[str, Any]], *, split: Mapping[str, Mapping[str, Any]] | None = None,
                      split_label: str = "", mcolors=None) -> None:
    tr = report["offcap"]["transitions"]
    keys = TRANSITION_KEYS
    bars: list[tuple[float, int]] = []
    mcnemar_lines: list[str] = []
    for j, (name, shown, color) in enumerate(names):
        t = tr[name]
        vals = [t["counts"][k] for k in keys]
        xs = [i + (j - 0.5) * 0.36 for i in range(len(keys))]
        p = (t["mcnemar"] or {}).get("p_value")
        p_text = "n/a" if p is None else f"{p:.3g}"
        mcnemar_lines.append(f"{shown} p = {p_text}")
        if split is not None:
            # the part of each count carried by items CAPPED under OFF/512, darker, under the rest
            capped = [split[name.split("__")[-1]]["capped_at_512"]["counts"][k] for k in keys]
            rest = [v - c for v, c in zip(vals, capped)]
            ax.bar(xs, capped, width=0.34, facecolor=color, edgecolor="none", zorder=3,
                   label=f"{shown}: capped under {split_label}")
            ax.bar(xs, rest, width=0.34, bottom=capped, facecolor=mcolors.to_rgba(color, 0.35), edgecolor="none", zorder=3,
                   label=f"{shown}: not capped")
            kind = name.split("__")[-1]
            for k, c in zip(keys, capped):
                _num_at(numbers, report, f"{name}[{k}].capped_at_512",
                        f"offcap.off512_to_off8192.cells.{kind}.by_first_draw_stop.capped_at_512.counts.{k}", c)
        else:
            ax.bar(xs, vals, width=0.34, facecolor=color, edgecolor="none", zorder=3, label=shown)
        bars += list(zip(xs, vals))
        for k, v in zip(keys, vals):
            _num_at(numbers, report, f"{name}[{k}]", f"offcap.transitions.{name}.counts.{k}", v)
        _num_at(numbers, report, f"{name}.mcnemar_p",
                f"offcap.transitions.{name}.mcnemar" + (".p_value" if t["mcnemar"] else ""), p)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.55)
    dy = 0.012 * ax.get_ylim()[1]
    for x, v in bars:
        ax.text(x, v + dy, str(v), ha="center", va="bottom", fontsize=6.2, color=INK2, zorder=7)
    ax.set_title(f"{title}\nMcNemar, complete-case, a sensitivity: " + "; ".join(mcnemar_lines),
                 fontsize=7.6, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=6.4, loc="upper center", ncol=2, columnspacing=1.2)
    _recessive(ax)


def fig_transitions_offcap(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """Per cell: OFF/512 → OFF/8192 (the part carried by items capped under
    OFF/512 highlighted) and OFF/8192 → ON/8192 transition counts; McNemar p
    labelled a complete-case sensitivity."""
    plt, mcolors, _Patch = _mpl()
    numbers: list[dict[str, Any]] = []
    off, off8, on = _cfg_labels(report)
    split = report["offcap"]["off512_to_off8192"]["cells"]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.76, bottom=0.14, wspace=0.22)
    _transition_panel(axes[0], report, [("off512_to_off8192__none", "none", KIND_COLOR["none"]),
                                        ("off512_to_off8192__psr", "psr", KIND_COLOR["psr"])],
                      f"{off} → {off8}, per cell (budget only)", numbers,
                      split={k: v["by_first_draw_stop"] for k, v in split.items()}, split_label=off, mcolors=mcolors)
    _transition_panel(axes[1], report, [("off8192_to_on__none", "none", KIND_COLOR["none"]),
                                        ("off8192_to_on__psr", "psr", KIND_COLOR["psr"])],
                      f"{off8} → {on}, per cell (render only)", numbers)
    _titles(fig, "Per-item transitions of the comparator (C = correct, I = incorrect, U = unscored)",
            "counts over the common attempted set; McNemar over the items both legs scored — complete-case, "
            "SECONDARY to the conditional estimand", sub_y=0.885)
    return fig, numbers


def fig_off512_to_off8192(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """The exploratory description: per cell the
    OFF/512 → OFF/8192 transition counts split by the first draw's at-cap
    status — labelled exploratory / post-selected; no delta, no CI, no
    expectation line."""
    plt, mcolors, _Patch = _mpl()
    numbers: list[dict[str, Any]] = []
    off, off8, _on = _cfg_labels(report)
    blk = report["offcap"]["off512_to_off8192"]
    keys = TRANSITION_KEYS
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.0))
    fig.subplots_adjust(left=0.06, right=0.98, top=0.76, bottom=0.14, wspace=0.22)
    for ax, kind in zip(axes, pr.LEG_CELLS):
        cell = blk["cells"][kind]
        bars: list[tuple[float, int]] = []
        for j, (sub, shown, alpha) in enumerate((("capped_at_512", f"stopped at the cap under {off}", 1.0),
                                                 ("not_capped_at_512", f"stopped before the cap under {off}", 0.35))):
            rec = cell["by_first_draw_stop"][sub]
            vals = [rec["counts"][k] for k in keys]
            xs = [i + (j - 0.5) * 0.36 for i in range(len(keys))]
            ax.bar(xs, vals, width=0.34, facecolor=mcolors.to_rgba(KIND_COLOR[kind], alpha),
                   edgecolor=KIND_COLOR[kind] if alpha < 1 else "none", linewidth=0.6, zorder=3,
                   label=f"{shown} (n = {rec['n']})")
            bars += list(zip(xs, vals))
            csrc = f"offcap.off512_to_off8192.cells.{kind}"
            for k, v in zip(keys, vals):
                _num_at(numbers, report, f"{kind}.{sub}[{k}]", f"{csrc}.by_first_draw_stop.{sub}.counts.{k}", v)
            _num_at(numbers, report, f"{kind}.{sub}.n", f"{csrc}.by_first_draw_stop.{sub}.n", rec["n"])
        cs = cell["capped_at_512_second_draw"]
        csrc = f"offcap.off512_to_off8192.cells.{kind}"
        _num_at(numbers, report, f"{kind}.capped_at_512.n_correct_at_8192", f"{csrc}.capped_at_512_second_draw.n_correct_at_8192",
                cs["n_correct_at_8192"])
        _num_at(numbers, report, f"{kind}.capped_at_512.n_at_cap_at_8192", f"{csrc}.capped_at_512_second_draw.n_at_cap_at_8192",
                cs["n_at_cap_at_8192"])
        _num_at(numbers, report, f"{kind}.not_capped_at_512.n_completion_tokens_differ",
                f"{csrc}.not_capped_at_512.n_completion_tokens_differ", cell["not_capped_at_512"]["n_completion_tokens_differ"])
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
        dy = 0.012 * ax.get_ylim()[1]
        for x, v in bars:
            ax.text(x, v + dy, str(v), ha="center", va="bottom", fontsize=6.2, color=INK2, zorder=7)
        ax.set_title(f"{kind}: {off} → {off8}, by the first draw's stop\nsecond draw of the {cs['n']} formerly "
                     f"capped: {cs['n_correct_at_8192']} correct, {cs['n_at_cap_at_8192']} at cap",
                     fontsize=7.4, color=INK, loc="left")
        ax.legend(frameon=False, fontsize=6.4, loc="upper right")
        _recessive(ax)
    _titles(fig, f"{off} → {off8} transitions by the first draw's stop",
            "an exploratory description, post-selected on the first draw's stopping behaviour: counts only — "
            "no delta, no CI, no expectation line", sub_y=0.885)
    return fig, numbers


def fig_controls_offcap(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """Mean / p99 completion tokens and the at-cap rate per leg over the six
    configurations; the 2 % gate segment over the OFF/8192 and ON bars only."""
    plt, mcolors, Patch = _mpl()
    numbers: list[dict[str, Any]] = []
    cfgs = report["offcap"]["configurations"]
    panels = [("mean_completion_tokens", "mean completion tokens"), ("p99_completion_tokens", "p99 completion tokens"),
              ("at_cap_rate", "at-cap rate (summary)")]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.0))
    fig.subplots_adjust(left=0.05, right=0.99, top=0.76, bottom=0.30, wspace=0.22)
    for ax, (field, title) in zip(axes, panels):
        xs, labels = [], []
        for x, kind, cfg in _cfg_slots(0.4):
            leg = _cfg_leg(report, kind, cfg)
            v = leg["summary"].get(field)
            label = cfgs[cfg]["label"]
            _num_at(numbers, report, f"{field}[{kind}/{label}]", f"{_cfg_leg_path(kind, cfg)}.summary.{field}", v)
            if v is not None:
                ax.bar([x], [v], width=0.7, **_cfg_bar_kw(mcolors, KIND_COLOR[kind], cfg))
            else:
                ax.text(x, 0, "null", ha="center", va="bottom", fontsize=6.5, color=MUTED)
            xs.append(x)
            labels.append(f"{kind}\n{label}")
        if field == "at_cap_rate":
            gated = [x for x, _kind, cfg in _cfg_slots(0.4) if cfg in (CFG_OFF_OVERRIDE, CFG_ON)]
            for x in gated:
                ax.hlines(CAP_GATE_AT_CAP_RATE, x - 0.42, x + 0.42, color=CAT["orange"], lw=0.9, ls="--", zorder=4)
            ax.text(0.99, 0.97, f"2 % gate: the {cfgs[CFG_OFF_OVERRIDE]['label']} and {cfgs[CFG_ON]['label']} bars",
                    transform=ax.transAxes, fontsize=6.2, color=CAT["orange"], va="top", ha="right")
        top = max([ax.get_ylim()[1]] + [0.0])
        ax.set_ylim(0, top * 1.25 if top > 0 else 1.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=6.4)
        ax.set_title(title, fontsize=8.2, color=INK, loc="left")
        _recessive(ax)
    fig.legend(handles=_cfg_handles(Patch, mcolors, report), loc="lower center", bbox_to_anchor=(0.5, 0.0),
               frameon=False, fontsize=6.8, ncol=3, handlelength=1.6, handleheight=0.9, columnspacing=1.4)
    _titles(fig, "Per-leg controls over the six configurations (from the summary payloads)",
            "every value is the summary's own count / rate over the ATTEMPTED draws — never re-derived from rows; "
            "the 2 % gate binds the adequate-budget legs only", sub_y=0.895)
    return fig, numbers


#: stem -> renderer: the four figures the estimand supports, then the parity
#: figure (which needs a wave root); every stem is spelled here and nowhere else
FIGURES_ESTIMAND: dict[str, Callable[[Mapping[str, Any]], tuple[Any, list[dict[str, Any]]]]] = {
    "final_e1_gsm8k_acc_2x2": fig_acc_2x2, "final_e1_effect_by_arm_ci": fig_effect_by_arm,
    "final_e1_controls_by_arm": fig_controls, "final_e1_item_transitions": fig_transitions}
FIGURE_PARITY = "final_e1_serving_parity_off"
FIGURES = {**FIGURES_ESTIMAND, FIGURE_PARITY: fig_parity}
FIGURE_STEMS: tuple[str, ...] = tuple(FIGURES)
#: the comparator's six, gated on ``"offcap" in report``; the
#: headline first
FIGURES_OFFCAP: dict[str, Callable[[Mapping[str, Any]], tuple[Any, list[dict[str, Any]]]]] = {
    "final_e1b_shift_decomposition": fig_shift_decomposition,
    "final_e1b_effect_by_configuration_ci": fig_effect_by_configuration,
    "final_e1b_gsm8k_acc_2x3": fig_acc_2x3,
    "final_e1b_item_transitions_offcap": fig_transitions_offcap,
    "final_e1b_off512_to_off8192_by_first_draw_stop": fig_off512_to_off8192,
    "final_e1b_controls_offcap": fig_controls_offcap}


def figure_captions(report: Mapping[str, Any]) -> dict[str, str]:
    """One caption per figure, written into the numbers and tables JSON beside
    the figures: every arm named with its cap (``OFF/512`` / ``ON/8192``),
    every cross-arm contrast called a render+cap contrast."""
    caps = arm_caps(report["legs"])
    off, on = arm_cap_label(C.THINKING_OFF, caps), arm_cap_label(C.THINKING_ON, caps)
    both = render_cap_configurations(caps)
    wave = (report.get("serving_parity") or {}).get("wave_sha", WAVE_SHA_DEFAULT)
    return {
        "final_e1_gsm8k_acc_2x2": (f"GSM8K accuracy per leg, cells none / psr under {both}; whiskers ±1 per-item "
                                   "SE, black brackets the censoring interval where it differs from the bar"),
        "final_e1_effect_by_arm_ci": (f"steering effect per arm — effect_off (psr − none at {off}), effect_on "
                                      f"(psr − none at {on}) and the DiD ({on} − {off}; a {RENDER_CAP_CONTRAST_LABEL}, "
                                      f"{both}) — point, percentile CI and censoring-sensitivity interval, both branches"),
        "final_e1_controls_by_arm": (f"per-leg controls from the summary payloads under {both}: completion length, "
                                     f"at-cap rate (the 2 % gate binds the {on} legs only), think-block counts, "
                                     "unparseable rate"),
        "final_e1_item_transitions": (f"per-item transitions: {off} → {on} per cell (a {RENDER_CAP_CONTRAST_LABEL}) "
                                      f"and none → psr per arm ({off}, {on}), counts on every bar"),
        FIGURE_PARITY: (f"serving parity: the fresh {off} legs against the wave's {wave} stores, paired delta "
                        "with 95 % CI over the same items"),
        **(offcap_figure_captions(report) if "offcap" in report else {}),
    }


def offcap_figure_captions(report: Mapping[str, Any]) -> dict[str, str]:
    """The comparator figures' captions: every configuration
    named with its cap, every contrast by what it varies; the exploratory
    description's caption names the post-selection and nothing more."""
    cfgs = report["offcap"]["configurations"]
    off, on = cfgs[CFG_OFF]["label"], cfgs[CFG_ON]["label"]
    off8 = cfgs.get(CFG_OFF_OVERRIDE, {}).get("label") or "OFF/(cap unknown)"
    return {
        "final_e1b_shift_decomposition": (
            f"per cell, the {off} → {on} total shift beside its budget term (cap relief, {off} → {off8}, same render) "
            f"and its render term (the matched shift, {off8} → {on}, same cap), each with its own 95 % CI; the two "
            "terms sum to the total exactly (a decomposition identity); the intervals do not add"),
        "final_e1b_effect_by_configuration_ci": (
            f"steering effect per configuration — effect_off (psr − none at {off}), effect_off_8192 (psr − none at "
            f"{off8}; a sensitivity, not the registered input) and effect_on (psr − none at {on}) — point, percentile "
            f"CI and censoring-sensitivity interval, both branches; did_matched (effect_on − effect_off_8192, render "
            f"only at cap {off8.split('/')[-1]}) annotated; the registered {off} band as before"),
        "final_e1b_gsm8k_acc_2x3": (
            f"GSM8K accuracy per leg, cells none / psr under {off}, {off8} and {on}; whiskers ±1 per-item SE, black "
            "brackets the censoring interval where it differs from the bar"),
        "final_e1b_item_transitions_offcap": (
            f"per-item transitions: {off} → {off8} per cell (budget only; the part carried by items capped under "
            f"{off} darker) and {off8} → {on} per cell (render only), counts on every bar; McNemar p a complete-case "
            "sensitivity"),
        "final_e1b_off512_to_off8192_by_first_draw_stop": (
            f"per cell, the {off} → {off8} transition counts split by the first draw's at-cap status — an exploratory "
            "description, post-selected on the first draw's stopping behaviour (no delta, no CI, no expectation line)"),
        "final_e1b_controls_offcap": (
            f"per-leg controls from the summary payloads over the six configurations ({off}, {off8}, {on} per cell): "
            f"completion length and at-cap rate; the 2 % gate binds the {off8} and {on} legs only"),
    }


def contrast_labels(report: Mapping[str, Any]) -> dict[str, str]:
    """What each contrast compares, arms named with their caps — for the
    tables JSON, beside the numbers it labels."""
    caps = arm_caps(report["legs"])
    off, on = arm_cap_label(C.THINKING_OFF, caps), arm_cap_label(C.THINKING_ON, caps)
    return {"effect_off": f"acc(psr, {off}) − acc(none, {off})",
            "effect_on": f"acc(psr, {on}) − acc(none, {on})",
            "shift_none": f"acc(none, {on}) − acc(none, {off}) — {RENDER_CAP_CONTRAST_LABEL}",
            "shift_psr": f"acc(psr, {on}) − acc(psr, {off}) — {RENDER_CAP_CONTRAST_LABEL}",
            "did": f"effect_on ({on}) − effect_off ({off}) — {RENDER_CAP_CONTRAST_LABEL}"}


def make_figures(report: Mapping[str, Any], figs_dir: Path) -> dict[str, Any]:
    """Every figure the report can support (:data:`FIGURES`), PNG +
    PDF, and the numbers JSON (every plotted number with its ``source``) beside
    them. ``figs_dir`` is resolved to an ABSOLUTE path before any manifest entry
    is built (a relative ``figs/`` produced entries like
    ``figs/<stem>.png`` that :func:`mirror_figures` then resolved against
    ``figs_dir`` again), so every entry the manifest names is absolute."""
    figs_dir = Path(figs_dir).expanduser().resolve()
    written: dict[str, list[str]] = {}
    numbers: dict[str, list[dict[str, Any]]] = {}
    skipped: dict[str, str] = {}

    def render(stem: str, fn: Callable[[Mapping[str, Any]], tuple[Any, list[dict[str, Any]]]]) -> None:
        fig, nums = fn(report)
        written[stem] = [str(p) for p in _save(fig, figs_dir, stem)]
        numbers[stem] = nums

    if "estimands" in report:
        for stem, fn in FIGURES_ESTIMAND.items():
            render(stem, fn)
    else:
        for stem in FIGURES_ESTIMAND:
            skipped[stem] = "the estimand did not run (a leg is missing or refused)"
    if report.get("serving_parity", {}).get("available"):
        render(FIGURE_PARITY, fig_parity)
    else:
        skipped[FIGURE_PARITY] = report.get("serving_parity", {}).get("reason", "no wave root")
    figs_dir.mkdir(parents=True, exist_ok=True)
    all_captions = figure_captions(report)

    def numbers_pair(stems: Collection[str], **provenance: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        # one figure set's captions (its WRITTEN stems) and its numbers document —
        # the same shape for the e1 pair and the comparator's e1b pair
        caps = {stem: cap for stem, cap in all_captions.items() if stem in written and stem in stems}
        return caps, {"figures": {k: v for k, v in numbers.items() if k in stems}, "captions": caps,
                      "skipped": {k: v for k, v in skipped.items() if k in stems}, "sha": report["sha"],
                      **provenance, "analysis_code_sha": report.get("analysis_code_sha")}

    # the e1 JSON pair is the FOUR-LEG set, exactly as before the comparator
    captions, numbers_doc = numbers_pair(set(FIGURES))
    (figs_dir / NUMBERS_FILENAME).write_text(json.dumps(numbers_doc, indent=1, default=str), encoding="utf-8")
    tables = {"effects": {b: report["estimands"][b]["stats"] for b in BRANCHES} if "estimands" in report else None,
              "contrast_labels": contrast_labels(report), "captions": captions,
              "acc_by_leg": {b: report["estimands"][b]["acc_by_leg"] for b in BRANCHES} if "estimands" in report else None,
              "controls": report["controls"], "verdict": report["verdict"], "gates": report["gates"],
              "complete_case_rows": report.get("complete_case_rows"), "serving_parity": report.get("serving_parity"),
              "sha": report["sha"], "analysis_code_sha": report.get("analysis_code_sha")}
    (figs_dir / TABLES_FILENAME).write_text(json.dumps(tables, indent=1, default=str), encoding="utf-8")
    manifest: dict[str, Any] = {"written": written, "skipped": skipped, "numbers": str(figs_dir / NUMBERS_FILENAME),
                                "tables": str(figs_dir / TABLES_FILENAME)}
    if "offcap" in report:
        # the comparator's set and its OWN JSON pair, beside the e1 pair
        oc = report["offcap"]
        if "estimands" in oc:
            for stem, fn in FIGURES_OFFCAP.items():
                render(stem, fn)
        else:
            for stem in FIGURES_OFFCAP:
                skipped[stem] = "the comparator estimand did not run (a leg is missing or refused)"
        captions_b, numbers_doc_b = numbers_pair(FIGURES_OFFCAP, offcap_sha=report.get("offcap_sha"))
        (figs_dir / NUMBERS_OFFCAP_FILENAME).write_text(json.dumps(numbers_doc_b, indent=1, default=str),
                                                          encoding="utf-8")
        tables_b = {"effects": ({name: {b: oc["estimands"][name][b]["stats"] for b in BRANCHES}
                                 for name in (MAP_CAP, MAP_MATCHED)} if "estimands" in oc else None),
                    "acc_by_leg": ({name: {b: oc["estimands"][name][b]["acc_by_leg"] for b in BRANCHES}
                                    for name in (MAP_CAP, MAP_MATCHED)} if "estimands" in oc else None),
                    "decomposition": oc.get("decomposition"), "did_matched": oc.get("did_matched"),
                    "effect_off_8192": oc.get("effect_off_8192"),
                    "verdict_sensitivity_off_8192": oc.get("verdict_sensitivity_off_8192"),
                    "contrast_varies": oc.get("contrast_varies"), "configurations": oc.get("configurations"),
                    "captions": captions_b, "transitions": oc.get("transitions"),
                    "off512_to_off8192": oc.get("off512_to_off8192"),
                    "complete_case_rows": oc.get("complete_case_rows"), "gates": oc.get("gates"),
                    "cap_gate": oc.get("cap_gate"), "consistency": oc.get("consistency"),
                    "sha": report["sha"], "offcap_sha": report.get("offcap_sha"),
                    "analysis_code_sha": report.get("analysis_code_sha")}
        (figs_dir / TABLES_OFFCAP_FILENAME).write_text(json.dumps(tables_b, indent=1, default=str), encoding="utf-8")
        manifest["numbers_offcap"] = str(figs_dir / NUMBERS_OFFCAP_FILENAME)
        manifest["tables_offcap"] = str(figs_dir / TABLES_OFFCAP_FILENAME)
    return manifest


# --------------------------------------------------------------------------- #
# W&B and the mirror                                                          #
# --------------------------------------------------------------------------- #


def log_wandb(report: Mapping[str, Any], figs: Mapping[str, Any], *, mode: str,
              name: str | None = None, name_suffix: str = "") -> str | None:
    """``wandb.init(name=e1-thinking-transfer-<sha7><suffix>)`` into the W&B project
    (or ``name``, the full run name — the comparator's ``e1b-offcap-comparator-<pin7>``);
    each figure under ``e1/<figure>`` (``e1b/<figure>`` for the comparator's
    set); the effect table (both branches) and the controls as summary fields;
    the figures dir as an artifact. ``disabled`` touches no network and imports
    nothing. ``name_suffix`` (``-v2``) names a re-analysis of the same stores
    apart from the first run. Under the comparator the
    legacy ``controls/<kind>/<arm>/<field>`` keys stay byte-identical for the
    four 2x2 legs — they cannot hold a second OFF cap — and ALL SIX
    configurations go under ``controls_cap/<kind>/<arm>/<cap>/<field>``, with the
    comparator contrasts under ``offcap/<branch>/<contrast>/…`` and the
    decomposition terms beside them."""
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as e:    # pragma: no cover - venv-dependent
        raise TransferAnalysisError("wandb is not importable — pass --wandb-mode disabled or use a venv "
                                    "that carries it") from e
    sha = report["sha"]
    name = name or f"e1-thinking-transfer-{sha}{name_suffix}"
    from sisteer.wandb_entity import wandb_project
    run = wandb.init(project=wandb_project(WANDB_PURPOSE), name=name, mode=mode,
                     config={"sha": sha, "offcap_sha": report.get("offcap_sha"),
                             "analysis_code_sha": report.get("analysis_code_sha"),
                             "model": report["model"], "task": TASK, **report["bootstrap"]})
    for stem, paths in figs.get("written", {}).items():
        png = next((p for p in paths if p.endswith(".png")), None)
        if png:
            prefix = "e1b" if stem.startswith("final_e1b_") else "e1"
            wandb.log({f"{prefix}/{stem}": wandb.Image(png)})
    summary: dict[str, Any] = {"verdict": report["verdict"]["verdict"]}
    if "estimands" in report:
        for b in BRANCHES:
            for c, s in report["estimands"][b]["stats"].items():
                for f in STAT_FIELDS:
                    summary[f"{b}/{c}/{f}"] = s[f]
    for row in report["controls"]:
        if row["leg"] not in report["legs"]:
            # the legacy keys hold ONE cap per arm: the four 2x2 legs only, never an
            # OFF/8192 row overwriting the OFF/512 one
            continue
        for f, v in row["summary"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                summary[f"controls/{row['kind']}/{row['arm']}/{f}"] = v
    oc = report.get("offcap")
    if oc:
        for row in report["controls"]:
            cap = row["summary"].get("max_tokens")
            for f, v in row["summary"].items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    summary[f"controls_cap/{row['kind']}/{row['arm']}/{cap}/{f}"] = v
        if "estimands" in oc:
            for mapping_name in (MAP_CAP, MAP_MATCHED):
                for b in BRANCHES:
                    for c, s in oc["estimands"][mapping_name][b]["stats"].items():
                        if c in OFFCAP_CONTRASTS:
                            for f in STAT_FIELDS:
                                summary[f"offcap/{b}/{c}/{f}"] = s[f]
            for kind, per in oc["decomposition"].items():
                for b, d in per.items():
                    for term in ("total", "budget", "render"):
                        for f in ("point", "ci_low", "ci_high"):
                            summary[f"offcap/decomposition/{kind}/{b}/{term}/{f}"] = d[term][f]
                    summary[f"offcap/decomposition/{kind}/{b}/budget_share_point"] = d["budget_share_point"]
            summary["offcap/effect_off_8192/resolved"] = oc["effect_off_8192"]["resolved"]
            summary["offcap/verdict_sensitivity_off_8192"] = oc["verdict_sensitivity_off_8192"]["verdict"]
    run.summary.update(summary)
    art = wandb.Artifact(f"figures-{name}", type="figures")
    if figs.get("written"):
        art.add_dir(str(Path(next(iter(figs["written"].values()))[0]).parent))
        run.log_artifact(art)
    url = getattr(run, "url", None)
    run.finish()
    return url


def mirror_figures(figs_dir: Path, mirror_dir: Path, *, manifest: Mapping[str, Any]) -> list[str]:
    """Copy EXACTLY the files the generated-file ``manifest``
    (:func:`make_figures`'s return) names — every PNG/PDF per stem, the e1 JSON
    pair and, when present, the e1b pair — into the served figures directory.
    The old ``final_e1_*`` glob is gone (it never
    matched ``final_e1b_*``, and a glob copies what happens to be on disk); a
    manifest entry that is not on disk refuses by name. Both directories are
    resolved to absolute paths; a relative entry (none since
    :func:`make_figures` resolves ``figs_dir`` itself) resolves against
    ``figs_dir``."""
    figs_dir = Path(figs_dir).expanduser().resolve()
    mirror_dir = Path(mirror_dir).expanduser().resolve()
    mirror_dir.mkdir(parents=True, exist_ok=True)
    entries: list[str] = []
    for paths in manifest.get("written", {}).values():
        entries.extend(paths)
    # `combined`: the four-family assembly's source view (gsm8k `--peer-report`), which the numbers sidecar's
    # `resolves_against` names — published beside the sidecars (review of 2026-09-11, P2), never left behind
    for key in ("numbers", "tables", "numbers_offcap", "tables_offcap", "combined"):
        if manifest.get(key):
            entries.append(str(manifest[key]))
    out = []
    for entry in entries:
        p = Path(entry)
        if not p.is_absolute():
            p = figs_dir / p
        if not p.is_file():
            raise TransferAnalysisError(f"mirror: manifest entry {p.name} is not on disk at {p}")
        dest = mirror_dir / p.name
        shutil.copy2(p, dest)
        # the second analysis review's Medium: the mirror once held an earlier render's
        # sidecars beside byte-identical PNGs — every copy is read back and must reproduce
        # its source byte for byte, never assumed
        if not filecmp.cmp(p, dest, shallow=False):
            raise TransferAnalysisError(f"mirror: {p.name} at {dest} is not byte-identical to {p} after the copy")
        out.append(str(dest))
    return out


# --------------------------------------------------------------------------- #
# the printed table (moved in-repo from analyze_oos.py, F4)                   #
# --------------------------------------------------------------------------- #


def _fmt(v: Any, spec: str = "6.3f") -> str:
    if v is None:
        return "    — "
    if isinstance(v, float):
        return format(v, spec)
    return str(v)


def print_table(report: Mapping[str, Any]) -> None:
    legs = report["legs"]
    label = report["model"]
    model = (f"{label['slug']} ({label['hf']} @ {label['revision'][:12]})" if label["slug"]
             else f"model NOT validated ({label['source']})")
    print(f"\n== the 2x2: {model}, {TASK}, k=1 at the card temperature (oos-{report['sha']}) ==")
    code = report.get("analysis_code") or {}
    if code:
        print(f"  producer/store pin {report['sha']}; analysis code {code['sha']} ({code['source']}"
              f"{'; TREE DIRTY — this sha does not identify the code that ran' if code.get('dirty') else ''})")
    head = (f"{'cell':<30} {'arm':<4} {'oos':<5} {'acc':>7} {'n':>5} {'unsc':>5} "
            f"{'unparse':>8} {'at_cap':>7} {'mean_tok':>9} {'p99_tok':>8} {'repeat':>7} {'empty':>7}")
    print(head)
    print("-" * len(head))
    for leg in legs.values():
        if leg.get("missing") or leg.get("refused"):
            print(f"{leg['cell']:<30} {leg['arm']:<4} {'MISSING' if leg.get('missing') else 'REFUSED'} — "
                  f"{leg.get('missing') or leg.get('refused')}")
            continue
        s = leg["summary"]
        t = leg["eval_check"]["text"]
        oos = leg["identity"][C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING]
        print(f"{leg['cell']:<30} {leg['arm']:<4} {('—' if oos is None else str(oos)):<5} "
              f"{_fmt(s.get('acc'))} {_fmt(s.get('n')):>5} {_fmt(s.get('n_unscored')):>5} "
              f"{_fmt(s.get('unparseable_rate'), '8.3f')} {_fmt(s.get('at_cap_rate'), '7.3f')} "
              f"{_fmt(s.get('mean_completion_tokens'), '9.1f')} {_fmt(s.get('p99_completion_tokens')):>8} "
              f"{_fmt(t.get('repetition_rate'), '7.3f')} {_fmt(t.get('empty_rate'), '7.3f')}")
        if t.get("unavailable"):
            print(f"{'':<30} {'':<4} repetition/empty unavailable: {t['unavailable']}")
    if "estimands" in report:
        print("\n== the estimand (conditional accuracy, own denominators) ==")
        print(f"  point; sampling CI (percentile); c-s = {BOUND_LABEL}, the ends' own bootstrap CI in ()")
        for b in BRANCHES:
            print(f"  [{b}]")
            for c, s in report["estimands"][b]["stats"].items():
                print(f"    {c:<11} {_pp(s['point']):>10}  CI {_ci(s)}  "
                      f"c-s {_ci(s, 'bound_low', 'bound_high')} ({_ci(s, 'bound_ci_low', 'bound_ci_high')})")
        cc = report["estimands"][BRANCH_PRIMARY]["complete_case"]
        print(f"  [{cc['label']}] " + "; ".join(
            f"{c}: {_pp(cc[c]['delta']) if cc[c].get('delta') is not None else '—'} (n={cc[c]['n_items']})"
            for c in pr.CONTRASTS))
        v = report["verdict_rule"]
        print(f"\n  words by branch: {v['word_by_branch']}; agree_branches={v['agree_branches']}; "
              f"decidable_bounds={v['decidable_bounds']}")
        print(f"  signs ({SIGN_IDENTIFIED_MEANING}): " + "; ".join(
            f"{c} {sg['reading']}" for c, sg in v["signs"].items()))
        xa = report.get("cross_arm_contrasts")
        if xa:
            print(f"\n== cross-arm contrasts: each a {xa['label']} ({xa['configurations']}) ==")
            for line in xa["lines"].values():
                print(f"  {line}")
        sens = report.get("sensitivity") or {}
        mc = sens.get("mcnemar_exact") or {}
        if mc:
            print(f"\n== sensitivity: {MCNEMAR_LABEL} ==")
            for k, m in mc.items():
                print(f"  {k:<18} {_mcnemar_text(m)}")
                note = first_one_sided_note(m)
                if note:
                    print(f"  {'':<18} {note}")
            print(f"  {replicate_signs_text(report['estimands'])}")
        ct = sens.get("closed_think_capped")
        if ct:
            print(f"\n== sensitivity: closed-think capped items — {ct['policy']} ==")
            print(f"  items: {ct['items'] or 'none'}")
            if ct.get("recoded"):
                for b in BRANCHES:
                    r = ct["recoded"][b]["effect_on"]
                    print(f"  [{b}] effect_on recoded: {_pp(r['point'])}  CI {_ci(r)}  "
                          f"c-s {_ci(r, 'bound_low', 'bound_high')}  "
                          f"(bound {ct['effect_on_bound_widening'][b] * 100:+.2f} pp wider); "
                          f"word {ct['word_by_branch'][b]}")
        d = report.get("selected_subset_diagnostic")
        if d:
            print(f"\n== {d['label']} ==")
            for kind, r in d["cells"].items():
                sub = r["subset_not_capped_off"]
                a_off, a_on = sub[C.THINKING_OFF], sub[C.THINKING_ON]
                print(f"  {kind:<5} OFF at cap {r['n_at_cap_off']} ({r['n_at_cap_off_incorrect']} incorrect); "
                      f"I→C {r['n_i_to_c']} of which at cap under OFF {r['n_i_to_c_at_cap_off']}; "
                      f"not-capped subset n={sub['n']}: OFF {a_off['correct']}/{a_off['scored']} "
                      f"vs ON {a_on['correct']}/{a_on['scored']}")
    par = (report.get("sensitivity") or {}).get("parity")
    if par:
        print(f"\n== serving parity: {par['wording']} ({par['note']}) ==")
    oc = report.get("offcap")
    if oc:
        print(f"\n== the thinking-OFF comparator at cap {OFFCAP_CAP} (offcap-{report.get('offcap_sha')}) ==")
        print(f"  comparator store pin {report.get('offcap_sha')}; 2x2 store pin {report['sha']}; "
              f"analysis code {report.get('analysis_code_sha')}")
        for leg in oc["legs"].values():
            if leg.get("missing") or leg.get("refused"):
                print(f"  {leg['leg']:<40} {'MISSING' if leg.get('missing') else 'REFUSED'} — "
                      f"{leg.get('missing') or leg.get('refused')}")
                continue
            s = leg["summary"]
            print(f"  {leg['leg']:<40} acc {_fmt(s.get('acc'))} n {_fmt(s.get('n'))} unsc {_fmt(s.get('n_unscored'))} "
                  f"at_cap {_fmt(s.get('at_cap_rate'), '7.3f')} mean_tok {_fmt(s.get('mean_completion_tokens'), '9.1f')} "
                  f"p99_tok {_fmt(s.get('p99_completion_tokens'))}")
        if "estimands" in oc:
            for name in (MAP_CAP, MAP_MATCHED):
                print(f"  [{name}] pseudo-2x2 off = {oc['estimands'][name][BRANCH_PRIMARY]['pseudo_2x2'][C.THINKING_OFF]}, "
                      f"on = {oc['estimands'][name][BRANCH_PRIMARY]['pseudo_2x2'][C.THINKING_ON]}")
                for b in BRANCHES:
                    print(f"    [{b}]")
                    for c, st in oc["estimands"][name][b]["stats"].items():
                        print(f"      {c:<20} {_pp(st['point']):>10}  CI {_ci(st)}  "
                              f"c-s {_ci(st, 'bound_low', 'bound_high')}  — {oc['estimands'][name][b]['varies'][c]}")
            print(f"  consistency: {'ok' if oc['consistency']['ok'] else 'FAILED'} over {len(oc['consistency']['checks'])} checks")
            for line in oc["findings"]:
                print(f"  * {line}")
            d = oc["off512_to_off8192"]
            print(f"  {d['label']}")
            for kind, r in d["cells"].items():
                cs = r["capped_at_512_second_draw"]
                print(f"    {kind:<5} all {r['counts']}; capped at 512: {r['by_first_draw_stop']['capped_at_512']['counts']} "
                      f"(n {cs['n']}, correct at {OFFCAP_CAP}: {cs['n_correct_at_8192']}, at cap: {cs['n_at_cap_at_8192']}); "
                      f"not capped: {r['by_first_draw_stop']['not_capped_at_512']['counts']} "
                      f"(length differs on {r['not_capped_at_512']['n_completion_tokens_differ']} of "
                      f"{r['not_capped_at_512']['n_compared']})")
            og = oc["gates"]["split_anomalies"]
            print(f"  comparator anomaly re-estimation: {og['status']} {og.get('detail') or ''}")
        print(f"  comparator cap gate: {oc['cap_gate']['status']} {oc['cap_gate'].get('detail') or ''}")
    g = report["gates"]
    print(f"\n== gates: failed={g['failed']} pending={g['pending']} ==")
    for name in GATE_NAMES:
        print(f"  {name:<16} {g[name]['status']:<8} {g[name].get('detail') or ''}")
    print(f"\n== VERDICT: {report['verdict']['verdict']} ==")
    if report["verdict"].get("statement"):
        print("  " + report["verdict"]["statement"])
    if report["verdict"].get("reason") and not report["verdict"].get("statement"):
        print("  " + str(report["verdict"]["reason"]))
    for item in report.get("top_items", []):
        print(f"  * {item}")
    for lim in report.get("interpretation_limits") or []:
        print(f"  ! limit: {lim}")


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-root", type=Path, default=None, help="a local copy of the volume's /vol/tinycap/gen")
    ap.add_argument("--sha", default=None, help="the run's out-suffix sha (oos-<sha7>)")
    # the two-pass runner's guards (the second analysis review's High): each is a whole invocation
    ap.add_argument("--gate-report", type=Path, default=None, metavar="REPORT",
                    help="read a written report and exit non-zero unless it is a production result (no failed gate, "
                         "no unwaived pending gate, no unusable leg, the estimands present); the runner calls this on "
                         "run E's report before run F")
    ap.add_argument("--diff-reports", type=Path, nargs=2, default=None, metavar=("A", "B"),
                    help="exit non-zero unless the two reports agree on every numeric leaf outside the log / figure "
                         "/ mirror / W&B provenance; the runner calls this on run E's and run F's reports before DONE")
    ap.add_argument("--require-report-match", type=Path, default=None, metavar="REPORT",
                    help="compare THIS run's report with REPORT's numeric leaves right after the analysis, BEFORE any "
                         "figure, mirror or W&B call; a difference writes the diagnostic report and exits non-zero "
                         "(the runner passes run E's report to run F)")
    ap.add_argument("--analysis-sha", default=None,
                    help="the git sha of the analysis code (recorded as analysis_code_sha beside --sha, the "
                         "producer/store pin); default: read at runtime from this script's own git tree")
    ap.add_argument("--wave-root", type=Path, default=None,
                    help="a local copy of the 57-cell wave's gen root (serving-parity cross-check)")
    ap.add_argument("--wave-sha", default=WAVE_SHA_DEFAULT)
    ap.add_argument("--prompt-budget", type=Path, default=None,
                    help="the CPU prompt-budget artifact (oos_prompt_budget.json; contracts."
                         "OOS_PROMPT_BUDGET_FIELDS, the ON arm's) — the third prompt-max source on the ON legs")
    ap.add_argument("--offcap-sha", default=None,
                    help="the thinking-OFF comparator's store pin (its offcap-<sha7> out-suffix): adds the two "
                         "OFF/8192 legs and the offcap block; absent, the 2x2 analysis exactly as today")
    ap.add_argument("--offcap-gen-root", type=Path, default=None,
                    help="a local copy of the gen root holding the comparator's stores (default: --gen-root; the "
                         "comparator pull lives in its own directory so the 2x2 pull stays untouched)")
    ap.add_argument("--prompt-budget-off", type=Path, default=None,
                    help="the comparator preflight's OFF prompt-budget artifact (arm off, cap 8192) — the third "
                         "prompt-max source on the OFF/8192 legs; --prompt-budget stays the ON artifact")
    ap.add_argument("--paired-ids", default=None,
                    help=f"the frozen paired-200 id set: a JSON list file, or '{PAIRED_IDS_FROM_PLAN}' to derive "
                         "it from the task plan (inspect_served.plan_task); without it the population's "
                         "identity is PENDING and the word is withheld")
    ap.add_argument("--bootstrap", type=int, default=pr.DEFAULT_BOOTSTRAP, metavar="B")
    ap.add_argument("--seed", type=int, default=pr.DEFAULT_BOOTSTRAP_SEED)
    ap.add_argument("--ci", type=float, default=pr.DEFAULT_CI)
    ap.add_argument("--allow-pending-gates", action="store_true",
                    help="file the word even when a stamp this tree does not write yet is pending (named)")
    ap.add_argument("--no-eval-logs", action="store_true", help="skip the .eval log cross-check and text signature")
    ap.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    ap.add_argument("--wandb-name-suffix", default="",
                    help="appended to the run and artifact names (a re-analysis of the same stores; a value "
                         "starting with '-' needs the '=' form: --wandb-name-suffix=-v2)")
    ap.add_argument("--wandb-name", default=None,
                    help="the FULL W&B run name (default e1-thinking-transfer-<sha7><suffix>; the comparator "
                         "analysis uses e1b-offcap-comparator-<pin7>)")
    ap.add_argument("--figs-dir", type=Path, default=None, help="where the figures go (default: beside --out, ./figs)")
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--mirror-dir", type=Path, default=Path("~/figs/si-steering-fit-support-transfer"))
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="write the full JSON report here")
    return ap


def _read_budget(path: Path | None) -> dict[str, Any] | None:
    """A prompt-budget artifact path (``--prompt-budget`` / ``--prompt-budget-off``)
    through the contract's reader — a torn, foreign or malformed artifact refuses
    by name; ``None`` when the flag was not given."""
    if path is None:
        return None
    try:
        return C.read_oos_prompt_budget(path)
    except C.OutOfSupportPromptBudgetError as e:
        raise TransferAnalysisError(str(e)) from e


#: exit codes of the production boundary (the second analysis review's High): the
#: analysis ran but its report is NOT a production result; two reports' numeric
#: leaves differ. Distinct from an exception's non-zero (a refusal before any report).
EXIT_NOT_PRODUCTION = 3
EXIT_REPORTS_DIFFER = 4
#: the report paths the two-venv comparison ignores: the log re-derivation (run E
#: only), the text signature it carries, and the figure / mirror / W&B / code-tree
#: provenance of the run that rendered
REPORT_DIFF_SKIP: tuple[str, ...] = ("eval_check", "figures", "mirrored", "wandb_url", "analysis_code", "report_match")


def production_refusals(report: Mapping[str, Any]) -> list[str]:
    """Why ``report`` is NOT a production result (the second analysis review's
    High) — every reason, by name, or ``[]``: a FAILED gate; a PENDING gate not
    waivable by ``--allow-pending-gates``; an unusable leg; the estimand
    withheld (no ``estimands`` block; under the comparator, no
    ``offcap.estimands``); and, when none of those named it, a verdict word
    withheld by the filed verdict (a pending stamp that was not waived). The
    exit code of :func:`main` is decided here, never by the report's mere
    existence: the v2 runner trusted the exit code, and a store the per-item
    log check had REFUSED went on to a canonical report."""
    gates = _mapping(report.get("gates"))
    out: list[str] = []
    if gates.get("failed"):
        out.append(f"gate(s) FAILED: {', '.join(gates['failed'])}")
    if gates.get("pending_unwaivable"):
        out.append("gate(s) PENDING and not waivable: "
                   + "; ".join(f"{it['gate']}: {it['detail']}" for it in gates["pending_unwaivable"]))
    if report.get("unusable_legs"):
        out.append(f"unusable leg(s): {sorted(report['unusable_legs'])}")
    if "estimands" not in report:
        out.append("the estimand did not run (no `estimands` block)")
    if "offcap" in report and "estimands" not in _mapping(report.get("offcap")):
        out.append("the comparator estimand did not run (no `offcap.estimands` block)")
    verdict = _mapping(report.get("verdict"))
    if not out and verdict.get("word") is None and str(verdict.get("verdict", "")).startswith(NOT_MEASURABLE):
        out.append(f"verdict withheld: {verdict.get('reason')}")
    return out


def _report_leaves(o: Any, path: tuple = ()):
    if isinstance(o, Mapping):
        for k, v in o.items():
            yield from _report_leaves(v, path + (k,))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from _report_leaves(v, path + (i,))
    else:
        yield path, o


def _diff_skipped(path: tuple) -> bool:
    return any(x in REPORT_DIFF_SKIP for x in path) or (len(path) > 2 and path[0] == "controls" and path[2] == "text")


def numeric_leaf_diff(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    """The two-venv comparison as a function: every NUMERIC leaf
    outside :data:`REPORT_DIFF_SKIP` (and the per-leg text signature) must be
    present in both reports with the same value — over the UNION of the two
    reports' numeric paths: ``differing`` (both numeric, value moved),
    ``type_changed`` (numeric on one side, present but not numeric — null,
    string, bool, a container — on the other), ``missing`` (numeric in ``a``,
    absent in ``b``) and ``added`` (numeric in ``b``, absent in ``a``) all
    empty for ``equal``. A numeric leaf that becomes null on one side is
    therefore a discrepancy, never a leaf that quietly leaves the comparison;
    a report whose estimand was withheld never equals one whose estimand
    ran."""
    la = {p: v for p, v in _report_leaves(a) if not _diff_skipped(p)}
    lb = {p: v for p, v in _report_leaves(b) if not _diff_skipped(p)}

    def num(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    na = {p: v for p, v in la.items() if num(v)}
    nb = {p: v for p, v in lb.items() if num(v)}
    shared = [p for p in na if p in nb]
    differing = [(p, na[p], nb[p]) for p in shared
                 if na[p] != nb[p] and not (isinstance(na[p], float) and isinstance(nb[p], float)
                                            and math.isnan(na[p]) and math.isnan(nb[p]))]
    # every path numeric on EITHER side must be numeric on both: present-but-not-numeric
    # on the other side is a type change, absent there a missing / added leaf
    type_changed = [(p, na[p], lb[p]) for p in na if p not in nb and p in lb]
    type_changed += [(p, la[p], nb[p]) for p in nb if p not in na and p in la]
    missing = [p for p in na if p not in lb]
    added = [p for p in nb if p not in la]
    return {"shared_numeric": len(shared), "differing": differing, "type_changed": type_changed, "missing": missing,
            "added": added, "equal": not differing and not type_changed and not missing and not added}


def _path_text(p: tuple) -> str:
    return "/".join(str(x) for x in p)


def gate_report(path: Path) -> int:
    """``--gate-report``: the runner's guard between its two passes — read the
    first pass's report and refuse (``EXIT_NOT_PRODUCTION``) unless
    :func:`production_refusals` is empty. A report that cannot be read is a
    refusal (an exception), never a pass."""
    report = _read_json(Path(path).expanduser())
    if not isinstance(report, Mapping):
        raise TransferAnalysisError(f"--gate-report {path}: not a JSON object")
    refusals = production_refusals(report)
    if refusals:
        print(f"[analyze] {path}: NOT a production result — " + "; ".join(refusals), file=sys.stderr)
        return EXIT_NOT_PRODUCTION
    print(f"[analyze] {path}: production result (gates {report['gates']['failed'] or 'none failed'}, "
          f"{len(report.get('unusable_legs') or {})} unusable legs, estimands present)")
    return 0


def diff_reports(path_a: Path, path_b: Path) -> int:
    """``--diff-reports A B``: the runner's second guard — the two passes'
    reports must agree on every numeric leaf outside :data:`REPORT_DIFF_SKIP`
    (``EXIT_REPORTS_DIFFER`` otherwise, the first differences named)."""
    a, b = _read_json(Path(path_a).expanduser()), _read_json(Path(path_b).expanduser())
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        raise TransferAnalysisError(f"--diff-reports: {path_a} / {path_b} must both be JSON objects")
    d = numeric_leaf_diff(a, b)
    if not d["equal"]:
        lines = [f"  differing {_path_text(p)}: {va} != {vb}" for p, va, vb in d["differing"][:10]]
        lines += [f"  type changed {_path_text(p)}: {va!r} vs {vb!r}" for p, va, vb in d["type_changed"][:10]]
        lines += [f"  only in {path_a}: {_path_text(p)}" for p in d["missing"][:10]]
        lines += [f"  only in {path_b}: {_path_text(p)}" for p in d["added"][:10]]
        print(f"[analyze] the reports' numeric leaves DIFFER — {len(d['differing'])} differing, {len(d['type_changed'])} "
              f"type changed, {len(d['missing'])} missing, {len(d['added'])} added over {d['shared_numeric']} shared:\n"
              + "\n".join(lines), file=sys.stderr)
        return EXIT_REPORTS_DIFFER
    print(f"[analyze] numeric leaves equal: {d['shared_numeric']} shared, 0 differing, 0 type changed, 0 missing, 0 added "
          f"(outside {', '.join(REPORT_DIFF_SKIP)} and the controls' text signature)")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.gate_report is not None:
        return gate_report(a.gate_report)
    if a.diff_reports is not None:
        return diff_reports(*a.diff_reports)
    if a.gen_root is None or a.sha is None:
        ap.error("--gen-root and --sha are required for an analysis run")
    budget = _read_budget(a.prompt_budget)
    budget_off = _read_budget(a.prompt_budget_off)
    paired_ids = None
    if a.paired_ids == PAIRED_IDS_FROM_PLAN:
        paired_ids = frozen_paired_ids_from_plan()
    elif a.paired_ids is not None:
        doc = _read_json(Path(a.paired_ids))
        if not isinstance(doc, list) or not all(isinstance(i, str) for i in doc):
            raise TransferAnalysisError(f"--paired-ids {a.paired_ids}: expected a JSON list of item ids")
        paired_ids = doc
    report = analyze(a.gen_root, a.sha, wave_root=a.wave_root, wave_sha=a.wave_sha, prompt_budget=budget,
                     paired_ids=paired_ids, n_resamples=a.bootstrap, seed=a.seed, ci=a.ci,
                     allow_pending_gates=a.allow_pending_gates, read_eval_logs=not a.no_eval_logs,
                     analysis_sha=a.analysis_sha, offcap_sha=a.offcap_sha, offcap_gen_root=a.offcap_gen_root,
                     prompt_budget_off=budget_off)
    print_table(report)
    # one path convention (the code review's A2): --out, --figs-dir and --mirror-dir
    # are ABSOLUTE from here on, so a bare `--out report.json` files the same
    # manifest entries as an absolute one and the mirror finds every one of them
    out = a.out.expanduser().resolve() if a.out else None

    def write_report() -> None:
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
            print(f"\n[analyze] full report -> {out}")

    # The production boundary fails CLOSED, and it does so BEFORE any side effect
    # (the second analysis review's High): the refusals
    # are decided on the analysis alone, the diagnostic report is written, and no
    # figure is drawn, no mirror overwritten and no W&B run started for a report
    # that is not a production result
    refusals = production_refusals(report)
    if refusals:
        write_report()
        print("[analyze] NOT a production result — " + "; ".join(refusals), file=sys.stderr)
        return EXIT_NOT_PRODUCTION
    # ... nor for a run whose numeric leaves differ from the run it must reproduce
    # (the runner hands run F run E's report): the in-memory report is compared
    # here, before publication; the external --diff-reports stays the post-hoc record
    if a.require_report_match is not None:
        match_path = a.require_report_match.expanduser().resolve()
        other = _read_json(match_path)
        if not isinstance(other, Mapping):
            raise TransferAnalysisError(f"--require-report-match {match_path}: not a JSON object")
        d = numeric_leaf_diff(other, report)
        report["report_match"] = {"against": str(match_path), "equal": d["equal"], "shared_numeric": d["shared_numeric"],
                                  "differing": [[_path_text(p), va, vb] for p, va, vb in d["differing"]],
                                  "missing": [_path_text(p) for p in d["missing"]],
                                  "added": [_path_text(p) for p in d["added"]],
                                  "type_changed": [[_path_text(p), va, vb] for p, va, vb in d["type_changed"]]}
        if not d["equal"]:
            write_report()
            lines = [f"  type changed {_path_text(p)}: {va!r} vs {vb!r}" for p, va, vb in d["type_changed"][:10]]
            print(f"[analyze] this run's numeric leaves DIFFER from {match_path} — {len(d['differing'])} differing, "
                  f"{len(d['type_changed'])} type changed, {len(d['missing'])} missing, {len(d['added'])} added over "
                  f"{d['shared_numeric']} shared; nothing published" + ("\n" + "\n".join(lines) if lines else ""),
                  file=sys.stderr)
            return EXIT_REPORTS_DIFFER
    figs: dict[str, Any] = {}
    if not a.no_figs:
        figs_dir = (a.figs_dir or ((out.parent if out else Path.cwd()) / "figs")).expanduser().resolve()
        figs = make_figures(report, figs_dir)
        report["figures"] = figs
        if not a.no_mirror:
            report["mirrored"] = mirror_figures(figs_dir, a.mirror_dir.expanduser().resolve(), manifest=figs)
    url = log_wandb(report, figs, mode=a.wandb_mode, name=a.wandb_name, name_suffix=a.wandb_name_suffix)
    report["wandb_url"] = url
    write_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
