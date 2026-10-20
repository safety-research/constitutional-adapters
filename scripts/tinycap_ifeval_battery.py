#!/usr/bin/env python3
"""Read out the IFEval canonical-battery wave on the paper's objects and file the verdicts.

Forty-two launched cells on four models under ONE instrument — inspect_evals' stock ``ifeval`` task (541 prompts,
``inspect/ifeval``), thinking OFF, k = 2 seeded trials at each model's vendor preset, one pass at 2,048 with the ladder
2,048 -> 4,096 -> 8,192 (a cell whose RAW at-cap share of DRAWS exceeds 1 % re-draws its truncated, non-loop DRAWS at the next
rung into a sibling root; its reading is the ladder-MERGED one with the 2,048-only reading beside it) — plus the four HELD
it-view constitution cells (a hard exclusion). Per (model, view) every cell is contrasted against that view's ``none`` on the
four IFEval metrics (prompt / instruction level x strict / loose) with item-cluster paired percentile bootstrap CIs; every
cell a two-view model carries on BOTH views with no object on either (llama8b: ``none`` and ``prompted__si_constitution``) is a
NOISE-FLOOR PAIR when the two rows are identical by construction (same served checkpoint, SI prefix, geometry entry, sampling
row, cap — the ``noise_floor`` block): it - main per draw and in aggregate on the same judged outcome, never
pooled, with a shared-configuration attestation read from the two stores that labels the pair a measured floor or "measures
that difference"; a promoted cell's ``merged - r2048`` is its truncation cost. At-cap
draws are SCORED AS IFEVAL SCORES THEM (no exclusion, no re-draw before scoring), then loop-classified with the cap ladder's
whitespace 4-gram rule, and the report carries the at-cap / truncated / loop shares per cell separately.

    PYTHONPATH=src python scripts/tinycap_ifeval_battery.py --gen-root <local pull of the wave root> --sha <WAVE_SHA7>
        --matrix configs/sdfbase/ifeval/matrix.json --ids <canonical 541 ids json> [--expected-refusals JSON]
        [--wandb-mode online|offline|disabled] [--figs-dir DIR] [--out report.json] [--mirror-dir DIR | --no-mirror]
        [--require-report-match REPORT] [--figure-exclude-family FAMILY] [--results-md OUT.md]
    # the two-venv match, in a venv without inspect_ai (the ONLY invocation that may leave the .eval logs unread):
    PYTHONPATH=src python scripts/tinycap_ifeval_battery.py ... --compare-only --require-report-match <production report> --no-eval-logs

Every store validator is PARAMETERIZED by the matrix row (``scripts/tinycap_ifeval_validators.py``); the task-agnostic
statistics and plumbing of the MATH-500 / OOS analyses are imported by module and never copied. The production boundary
fails CLOSED before any figure, mirror or W&B call (exit 3); the two-venv match refuses to publish on a differing numeric leaf
(exit 4). A production report READS the retained ``.eval`` logs: I4 (the stock-score cross-check) and I5 (the loop rule
re-derived from the retained texts) are PENDING with the logs unread, and that PENDING no flag waives —
``--no-eval-logs`` is permitted only inside ``--compare-only``, a pass that claims no I4 / I5 pass and no production and
publishes nothing but its I9 match.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sisteer.capeval.stats import item_bootstrap as IB
from sisteer.capeval.tiers import cap_ladder as CL
from sisteer.capeval.tiers import inspect_rung as IR
from sisteer.capeval.tiers import pairing as pr
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import ifeval_contracts as IC

_HERE = Path(__file__).resolve().parent


def _load_module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


V = _load_module("tinycap_ifeval_validators", _HERE / "tinycap_ifeval_validators.py")
M = _load_module("tinycap_math500_thinking", _HERE / "tinycap_math500_thinking.py")
T = M.T
AnalysisError = V.AnalysisError

EXPERIMENT = "ifeval-canonical-battery"
TASK = IC.IFEVAL_TASK
CAPS = IC.IFEVAL_CAP_LADDER
CAP_BASE = IC.IFEVAL_PASS1_CAP
CEILING = IC.IFEVAL_CAP_CEILING
K = IC.IFEVAL_SAMPLES_PER_ITEM
N = IC.IFEVAL_N
N_DRAWS = N * K
READINGS = IC.READINGS
READING_PASS1, READING_MERGED = IC.READING_PASS1, IC.READING_MERGED
METRICS = IC.IFEVAL_METRIC_NAMES
METRIC_FIELD = {v: k for k, v in IC.METRIC_BY_FIELD.items()}
PROMPT_METRICS = ("prompt_strict", "prompt_loose")
CONTRAST_VS_NONE, CONTRAST_NOISE, CONTRAST_TRUNCATION = IC.CONTRAST_VS_NONE, IC.CONTRAST_NOISE, IC.CONTRAST_TRUNCATION
GATE_NAMES: tuple[str, ...] = ("I0_completeness", "I1_ledger", "I2_identity", "I3_certification", "I4_scoring_path", "I5_at_cap",
                               "I6_ladder", "I7_population", "I8_two_reading", "I9_report_match")
#: a PENDING I9 (the two-venv match not requested in this pass) does not refuse production — it is run as its own pass
WAIVABLE_PENDING_GATES: tuple[str, ...] = ("I9_report_match",)
#: the gates whose PASS needs the retained `.eval` logs: I4's stock-score cross-check and I5's text re-derivation.
#: With the logs unread (`--no-eval-logs`) they read PENDING, and that PENDING is NEVER waivable — not by `--allow-pending-gates`,
#: not by any flag: a report that has not read the authoritative logs is not a production report. Before this the two gates
#: folded to PASSED with the logs unread (their per-leg checks were simply absent), so a `--no-eval-logs` pass certified
#: I4 / I5 and `production: true` without reading a log; the r2 second-venv artifact did exactly that.
LOG_DERIVED_GATES: tuple[str, ...] = ("I4_scoring_path", "I5_at_cap")
EVAL_LOGS_PENDING_I4 = ("the .eval logs were not read (--no-eval-logs): the stock-score cross-check did not run on any leg — PENDING, "
                        "never waivable for production")
EVAL_LOGS_PENDING_I5 = ("the retained .eval texts were not read (--no-eval-logs): the loop rule was not re-derived on any leg (the record's "
                        "lists were checked for internal consistency only) — PENDING, never waivable for production")
#: the EVIDENCE each log-derived gate's PASS rests on: I4's ``eval_logs_read`` (the stock-score cross-check ran on
#: every usable leg) and I5's ``texts_reclassified`` (the loop rule was re-derived from every at-cap draw's retained text). At
#: CERTIFICATION the flag is authoritative, not the status: a PASSED whose flag is not exactly True — False or MISSING — is refused,
#: and no flag waives it. Before this `verdict_of` / `production_refusals` / `gate_report` read the two statuses alone, so a report
#: FILE carrying `I4 PASSED, eval_logs_read false` (the preserved revision-2 no-log artifact) certified `production: true` through
#: `--gate-report` although the generation route had been closed — report-only certification was bypassable.
LOG_DERIVED_EVIDENCE: Mapping[str, str] = {"I4_scoring_path": "eval_logs_read", "I5_at_cap": "texts_reclassified"}
#: the statuses a gate record may carry. At CERTIFICATION every :data:`GATE_NAMES` entry must be PRESENT in the
#: report's ``gates`` block, be a record, and carry one of exactly these three — BEFORE any waiver rule is read. An absent gate,
#: a record with no ``status`` or a status outside this set is an UNWAIVABLE refusal naming the gate and the value the file
#: carries; it is not FAILED, not PENDING and not PASSED-without-evidence, so none of those rules would otherwise see it.
#: Before this `gates_failed` / `gates_pending_*` / `gates_passed_without_evidence` matched explicit FAILED / PENDING / PASSED
#: only, and `verdict_of` required nothing of a gate it could not find: three malformed copies of the revision-3 production
#: report (I4 / I5 deleted; their `status` keys deleted; their statuses set to "UNKNOWN" with the evidence flags removed)
#: certified `production: true` through `--gate-report` — report-only certification failed OPEN on a missing gate.
GATE_STATUSES: tuple[str, ...] = (T.GATE_PASS, T.GATE_FAIL, T.GATE_PENDING)
#: the one invocation that may run with the logs unread: a COMPARISON-ONLY pass (`--compare-only --require-report-match REPORT`),
#: whose report claims no I4 / I5 pass and no production, renders no record, makes no figure, mirrors nothing and logs nothing
COMPARISON_ONLY_MODE = "comparison-only"
GATE_PASS, GATE_FAIL, GATE_PENDING = T.GATE_PASS, T.GATE_FAIL, T.GATE_PENDING
EXIT_NOT_PRODUCTION, EXIT_REPORTS_DIFFER = T.EXIT_NOT_PRODUCTION, T.EXIT_REPORTS_DIFFER
WORD_LOWER, WORD_HIGHER, WORD_NOT_RESOLVED = "lower than none", "higher than none", "not resolved at this cap"
#: the W&B purpose: the project is sisteer.wandb_entity.wandb_project(WANDB_PURPOSE) (the site's SISTEER_WANDB_PROJECT_TINYCAP first)
WANDB_PURPOSE = "tinycap"
WANDB_RUN_PREFIX = "ifeval-canonical-"
MIRROR_DIR_DEFAULT = "~/figs/ifeval-canonical"
NUMBERS_FILENAME = "final_ifeval_figures_numbers.json"
TABLES_FILENAME = "final_ifeval_tables.json"
#: the results SKELETON is an immutable template at its own path, never the rendered record's. Before this
#: the template WAS the rendered record's own path — also the documented `--results-md` output — so the first
#: render consumed the placeholders and every later render silently kept the old numbers (a second output byte-identical
#: after the pin changed). The rendered record lands at `RESULTS_RECORD_REL`; `render_results_md` refuses an output that
#: aliases the template and a template with no placeholders left.
RESULTS_TEMPLATE = _HERE.parent / "data" / "tinycap" / "ifeval" / "results.template.md"
RESULTS_RECORD_REL = "out/ifeval_canonical/results.md"   # an OUTPUT path (gitignored), never a tracked record
LEG_STATUS_USABLE, LEG_STATUS_MISSING, LEG_STATUS_REFUSED, LEG_STATUS_HELD = ("usable", "missing", "refused", "held")
#: the follow-up: a DEFERRED design slot is counted by construction, like a held row — never `missing`
LEG_STATUS_DEFERRED = "deferred"
#: the brief's word for a deferred design slot in every table and figure: PENDING, with the follow-up that fills it
PENDING_LABEL = "PENDING (PR #93: the AlphaSteer / Arditi rows, staged and dispatched as an addendum)"
HELD_LABEL = "HELD"
#: the CENTRAL per-cell GPU-hour estimate (a labelled estimate, never a bound) and the Modal standup adder
CENTRAL_GPU_H_PER_CELL = {"gemma4e4b": 0.28, "qwen35_4b": 0.29, "nemotron3nano4b": 0.22, "llama8b": 0.26}
#: the follow-up (F6): a constitution-prompted cell serves ~6.4k-token prompts at ~0.3x the short cells' S (its own
#: capability:ifeval_longsi_<cap> window) — the same 1,082 draws, more prefill and fewer concurrent sequences
CENTRAL_GPU_H_LONGSI_ADDER = 0.15
#: the wave-level CENTRAL estimate under the current matrix (30 pass-1 cells, 5 of them constitution-prompted, two Nemotron
#: certifications, the S1 ladder scaled to 30 cells) — the earlier 15-17 was the 42-cell matrix's
CENTRAL_ESTIMATE_TOTAL_GPU_H = ("12-13 (the follow-up F6: 30 pass-1 cells ~8.1 + the 5 constitution cells' wider windows "
                                "~0.8 + two Nemotron certifications 0.5 + the S1 ladder at 30 cells ~2.8; plan §9.2's 15-17 "
                                "was the 42-cell matrix, CENTRAL, never a bound; scenarios S0-S4 there)")
STANDUP_ADDER_S = 250.0
ESTIMAND_TEXT = ("per (model, view): the paired within-model contrast metric(cell) - metric(none) over the ONE common set of 541 items x 2 "
                 "seeded trials, each of the four IFEval metrics a ratio (prompt level: passing draws / scored draws; instruction level: "
                 "followed instructions / all instructions), item-cluster paired percentile bootstrap (the item is the resampling unit; "
                 "one index draw per replicate applied to every leg), B and seed as recorded; a verdict word iff the merged reading's CI "
                 "excludes zero on BOTH prompt-level metrics with agreeing sign")
LOOP_RULE_TEXT = CL.DEGENERATE_RULE
LOOP_RULE_DISCLOSURE = ("loop = whitespace-token 4-gram repetition ratio > 0.5 on the whole text or the last 200 WHITESPACE tokens "
                        "(the cap ladder's rule as implemented — 'tokens' in the ruling are whitespace tokens here, not model tokens)")


@dataclass
class Context:
    canonical_ids: list[str] | None
    pins: Mapping[str, Any] | None
    expected_refusals: Mapping[str, str]
    n_resamples: int = pr.DEFAULT_BOOTSTRAP
    seed: int = pr.DEFAULT_BOOTSTRAP_SEED
    ci: float = pr.DEFAULT_CI
    read_eval_logs: bool = True
    exclude_families: tuple[str, ...] = ()
    #: per-LEG admitted producer trees (`{"<model>/<view>/<cell>@<cap>": "<40-hex>"}`, V24's rule: admission by per-row pin,
    #: never one tree per wave — the battery ran at 4d389d87 / c629e506 (pass 1), 44395207 / 5c816cf7 (4,096), e43f81f2
    #: (8,192)); None = the legacy one-pin admission by the `sha` prefix
    store_pins: Mapping[str, str] | None = None
    #: the Nemotron (tree, geometry entry) certifications (`{"<tree>": {"<entry>": {"admitted": bool, "attestation_sha256": ...}}}`)
    nemotron_certifications: Mapping[str, Any] | None = None


# --------------------------------------------------------------------------- one leg
def _leg_skeleton(row: Mapping[str, Any], cap: int, root: Path) -> dict[str, Any]:
    return {"key": V.leg_key(row["model_key"], row["view"], row["cell"], cap), "model_key": row["model_key"], "view": row["view"],
            "cell": row["cell"], "cap": int(cap), "out_root": str(root), "status": LEG_STATUS_MISSING, "refused": None,
            "fitted": bool(row["fitted"]), "route_kind": row["route_kind"], "family": row["family"], "slug": row["slug"]}


def collect_leg(gen_root: Path, row: Mapping[str, Any], *, cap: int, ctx: Context, subset: Mapping[str, Any] | None = None,
                source_manifest: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One leg (a pass-1 store or a rung store) read through the validators: manifest / summary / outcomes / rows / ledger /
    request evidence / endpoint / the retained logs; every check filed on the leg, ``status`` usable iff the store is complete
    and every hard reader check holds (the GATES then read the check records)."""
    paths = IC.ifeval_paths(row["model_key"], row["view"], row["cell"], cap)
    root = Path(gen_root) / paths.out_root
    leg = _leg_skeleton(row, cap, root)
    cd, manifest, why = V.read_store(root)
    if manifest is None:
        leg["refused"] = why
        return leg
    leg["cell_dir"] = str(cd)
    leg["manifest_producer_tree_sha"] = (manifest.get(C.MANIFEST_KEY_PRODUCER) or {}).get("tree_sha")
    leg["result"] = V.result_stamp(root)
    # admission by the per-row PIN: the leg's producer tree must be exactly the tree the wave admitted for it
    why = admitted_pin_problem(leg, ctx)
    if why:
        leg["refused"] = why
        return leg
    summary = completion.read_task_summary(cd, TASK)
    if summary is None:
        leg["refused"] = f"no summary/{completion.marker_filename(TASK)} (the store did not finish)"
        return leg
    try:
        outcomes = completion.read_task_outcomes(cd, TASK)
    except completion.CompletionError as e:
        leg["refused"] = f"outcomes record refused: {e}"
        return leg
    if outcomes is None:
        leg["refused"] = "no outcomes record"
        return leg
    try:
        completion.verify_outcomes_against_summary(outcomes, summary)
    except completion.CompletionError as e:
        leg["refused"] = f"outcomes disagree with the summary: {e}"
        return leg
    marker = V.read_json_if_file(completion.done_marker_path(cd, TASK))
    if marker is None:
        leg["refused"] = "no DONE marker (a PARTIAL store is never a leg)"
        return leg
    if subset is not None:
        from sisteer.capeval.tinycap import rung_probe as RP
        why = RP.draw_subset_task_reason(cd, TASK, draws=subset["draws"], cap=int(cap),
                                         source_identity_hash=(source_manifest or {}).get("harness_config_hash"))
        if why:
            leg["refused"] = f"rung store is not exactly its draw subset: {why}"
            return leg
    srow = IC.vendor_preset_row(row["model_key"])
    leg["summary"] = summary
    leg["outcomes_header"] = {k: v for k, v in outcomes.items() if k != "items"}
    leg["items"] = list(outcomes["items"])
    leg["marker"] = marker
    leg["n_draws"] = int(outcomes["n_attempted_samples"])
    leg["n_items"] = int(outcomes["n_attempted_items"])
    leg["identity"] = V.identity_problem(manifest, row=row, cap=cap, srow=srow)
    leg["stamps"] = V.stamps_problem(summary, row=row)
    rows_jsonl = V.iter_jsonl(cd / "rows.jsonl") if (cd / "rows.jsonl").is_file() else []
    leg["rows_check"] = V.rows_identity_check(rows_jsonl, row=row, manifest=manifest, cap=cap)
    ledger_path = root / IC.LEDGER_FILENAME
    # the PRODUCER's records beside the store (the Modal launcher's): `served_geometry.json` = the served geometry + the
    # standup's endpoint record (there is no endpoint.json — the wave shakedown, V20), `evidence.json` = the driver's
    # serve-header evidence (the thinking pin AS SERVED), the recorder's verdict, the RESULT stamp
    served = V.served_record(root)
    evidence = V.evidence_record(root)
    endpoint = served.get("endpoint") if isinstance(served, Mapping) and isinstance(served.get("endpoint"), Mapping) else None
    leg["certification"] = V.certification_check(manifest, served, row=row, cap=cap, evidence=evidence, certifications=ctx.nemotron_certifications)
    # the served object's FIT PROVENANCE: what it was fit on, from the manifest's block — never from the view
    # name; a fitted cell without it, or a `transfer` flag disagreeing with the fit-vs-served identity, fails I3 by name
    leg["fit_provenance"] = V.fit_provenance_record(manifest, row=row)
    if leg["fit_provenance"].get("problem"):
        leg["certification"]["failures"].append(f"fit provenance: {leg['fit_provenance']['problem']}")
        leg["certification"]["passed"] = False
    leg["recorder"] = V.recorder_record(root)
    leg["recorder_problem"] = V.recorder_problem(leg["recorder"])
    # C4: both the ledger and the evidence are scoped to the manifest's ACCEPTED attempt; the evidence index is the join a
    # ledger row without its own attempt stamp is attributed through (built once, shared by both checks)
    evidence_path = root / C.REQUEST_EVIDENCE_FILENAME
    evidence_recs = V.iter_jsonl(evidence_path) if evidence_path.is_file() else []
    evidence_by_id, evidence_problem = V.evidence_index(evidence_recs)
    leg["ledger"] = V.ledger_check(ledger_path, header=outcomes, summary=summary, cap=cap, max_model_len=leg["certification"].get("max_model_len"),
                                   manifest=manifest, evidence_by_id=evidence_by_id)
    ledger_rows = V.iter_jsonl(ledger_path) if ledger_path.is_file() else []
    leg["request_evidence"] = V.request_evidence_check(evidence_path, ledger_rows=ledger_rows, row=row, srow=srow, n_draws=leg["n_draws"],
                                                       manifest=manifest, evidence_by_id=evidence_by_id, evidence_problem=evidence_problem)
    # the noise-floor attestation's per-draw wire facts and the endpoint record (in memory only; the report carries the
    # attestation's counts, never the per-draw maps)
    leg["wire"] = V.wire_digest(evidence_recs, ledger_rows, manifest=manifest, evidence_by_id=evidence_by_id)
    leg["endpoint"] = dict(endpoint) if isinstance(endpoint, Mapping) else None
    leg["scoring_path"] = {"ifeval_task_version": summary.get("ifeval_task_version"), "langdetect_seed": summary.get("langdetect_seed"),
                           "samples_per_item": outcomes.get("samples_per_item"), "thinking": summary.get("thinking"),
                           "failures": ([] if summary.get("ifeval_task_version") == IC.IFEVAL_TASK_VERSION else [f"ifeval_task_version {summary.get('ifeval_task_version')!r}"])
                           + ([] if summary.get("langdetect_seed") == IC.LANGDETECT_SEED else [f"langdetect_seed {summary.get('langdetect_seed')!r}"])
                           + ([] if outcomes.get("samples_per_item") == K else [f"samples_per_item {outcomes.get('samples_per_item')!r}"])
                           + ([] if summary.get("thinking") == IC.ifeval_effective_arm(row["family"])
                              else [f"thinking {summary.get('thinking')!r} != the family's OFF arm {IC.ifeval_effective_arm(row['family'])!r}"])}
    leg["eval_check"] = (V.eval_cross_check(cd, leg["items"], summary=summary, cap=cap) if ctx.read_eval_logs
                         else {"passed": None, "failures": [], "note": "eval logs not read (--no-eval-logs)"})
    leg["ladder_record"] = V.read_json_if_file(root / IC.CAP_LADDER_FILENAME)
    leg["ladder_problem"] = V.ladder_record_problem(leg["ladder_record"], items=leg["items"], cap=cap, source_manifest=manifest)
    # I5's TEXT half: the loop rule re-derived from the retained .eval texts and compared with the record's
    # draw lists exactly, every at-cap text's sha256 verified — read with the .eval logs (the same venv condition as I4)
    leg["at_cap_text"] = (V.at_cap_text_check(root, leg["items"], record=leg["ladder_record"]) if ctx.read_eval_logs
                          else {"passed": None, "failures": [], "note": "retained texts not read (--no-eval-logs): the record's lists were checked "
                                                                       "for internal consistency only, not re-derived"})
    # C1: THE decision (the planner's block over the MERGED reading) is what the chain walk, I6 and every table read; the
    # store's own classification (exit / counts over its own population) is a diagnostic beside it
    leg["ladder_decision"], leg["decision_problem"] = V.ladder_decision_of(leg["ladder_record"])
    leg["population"] = V.population_problem(leg["items"], canonical_ids=ctx.canonical_ids,
                                             subset_draws=(subset["draws"] if subset is not None else None))
    n_at = sum(1 for it in leg["items"] if it.get("at_cap"))
    rec = leg["ladder_record"] if isinstance(leg["ladder_record"], Mapping) else {}
    dec = leg["ladder_decision"] or {}
    leg["shares"] = {"n_draws": leg["n_draws"], "n_at_cap": n_at, "share_at_cap": n_at / leg["n_draws"] if leg["n_draws"] else None,
                     "n_truncated": rec.get("n_truncated"), "n_loop": rec.get("n_loop"), "n_boundary": rec.get("n_boundary"),
                     "share_truncated": rec.get("share_truncated"), "share_loop": rec.get("share_loop"),
                     "exit": dec.get("exit"), "next_cap": dec.get("next_cap"), "store_exit": rec.get("exit"),
                     "trigger_provenance": rec.get("trigger_provenance"),
                     "decision": ({k: dec.get(k) for k in ("reading", "chain", "n_draws", "n_at_cap", "share_at_cap", "n_rung", "exit", "next_cap")}
                                  if dec else None)}
    leg["manifest_hash"] = manifest.get("harness_config_hash")
    leg["manifest"] = manifest
    leg["status"] = LEG_STATUS_USABLE
    return leg


def usable(leg: Mapping[str, Any]) -> bool:
    return leg.get("status") == LEG_STATUS_USABLE


def admitted_pin_problem(leg: Mapping[str, Any], ctx: Context) -> str | None:
    """Why the leg's producer tree is not the tree the wave ADMITTED for it (``ctx.store_pins``; per-row admission):
    no entry for the leg, or another tree. ``None`` when no pins doc was given (the legacy one-pin admission then
    happens in :func:`analyze`) or when the tree is the admitted one. A pin is a 40-hex tree or a >= 7-hex prefix of it."""
    if ctx.store_pins is None:
        return None
    want = ctx.store_pins.get(leg["key"])
    tree = str(leg.get("manifest_producer_tree_sha") or "")
    if want is None:
        return f"no admitted pin for {leg['key']} in the store-pins doc (the leg is not part of the admitted battery)"
    if not tree or not (tree == str(want) or tree.startswith(str(want)) and len(str(want)) >= 7):
        return f"producer tree {tree[:8] or None!r} is not the admitted pin {str(want)[:8]!r} for {leg['key']}"
    return None


def resolve_store_root(gen_roots: Sequence[Path], row: Mapping[str, Any], *, ctx: Context) -> tuple[Path | None, list[dict[str, Any]], str | None]:
    """The ONE wave root a row's pass-1 store is read from, among several pulls (the battery's 30 rows sit under two
    roots — 18 + 12 by pass-1 pin — and the two gemma-it prompted rows have a FAILED first-pin store beside their complete
    re-run, V21 / V30): every root holding the row's ``out_root`` is a candidate with its producer tree and RESULT stamp; with
    a pins doc the admitted root is the one whose tree IS the row's admitted pass-1 pin (the others are SUPERSEDED stores,
    listed, never legs); without one, the single candidate. Returns ``(root, candidates, problem)`` — two admitted stores or
    none is a problem by name."""
    paths = IC.ifeval_paths(row["model_key"], row["view"], row["cell"], CAP_BASE)
    key = V.leg_key(row["model_key"], row["view"], row["cell"], CAP_BASE)
    want = (ctx.store_pins or {}).get(key) if ctx.store_pins is not None else None
    candidates: list[dict[str, Any]] = []
    for g in gen_roots:
        root = Path(g) / paths.out_root
        if not root.is_dir():
            continue
        cd, manifest, why = V.read_store(root)
        tree = ((manifest or {}).get(C.MANIFEST_KEY_PRODUCER) or {}).get("tree_sha") if manifest else None
        stamp = V.result_stamp(root)
        admitted = (want is None) if ctx.store_pins is None else bool(tree and want and (tree == want or (len(str(want)) >= 7 and tree.startswith(str(want)))))
        candidates.append({"gen_root": str(g), "out_root": str(root), "producer_tree_sha": tree, "result": stamp, "admitted": admitted,
                           "manifest_problem": why})
    chosen = [c for c in candidates if c["admitted"]]
    if not candidates:
        return None, candidates, f"no store for {key} under any of {[str(g) for g in gen_roots]}"
    if ctx.store_pins is not None and want is None:
        return None, candidates, f"no admitted pin for {key} in the store-pins doc"
    if len(chosen) != 1:
        found = [f"{c['gen_root']} -> tree {str(c['producer_tree_sha'])[:8]!r} ({(c['result'] or {}).get('result')})" for c in candidates]
        return None, candidates, (f"{len(chosen)} stores for {key} at the admitted pin {str(want)[:8] if want else '<any>'!r} across the roots "
                                  f"(exactly one expected): {found}")
    return Path(chosen[0]["gen_root"]), candidates, None


# --------------------------------------------------------------------------- one cell: the chain and the readings
def _draw_record(it: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    return {"item_id": str(it["item_id"]), "sample_idx": int(it["sample_idx"]), "scored": bool(it.get("scored")), "at_cap": bool(it.get("at_cap")),
            "completion_tokens": it.get("completion_tokens"), "stop_reason": it.get("stop_reason"), "num_instructions": it.get("num_instructions"),
            **{f: it.get(f) for f in IC.IFEVAL_METRIC_FIELDS}, "source": source}


def _gen_roots(gen_root: Path | str | Sequence[Path | str]) -> list[Path]:
    return [Path(gen_root)] if isinstance(gen_root, (str, Path)) else [Path(g) for g in gen_root]


def collect_cell(gen_root: Path | Sequence[Path], row: Mapping[str, Any], *, ctx: Context) -> dict[str, Any]:
    """The pass-1 leg and its ladder chain (pass 1 -> rung 4096 -> rung 8192), each hop validated by its record + subset
    (``source_out_root`` + identity hash), and the two readings: ``r2048`` (pass 1 alone) and ``merged`` (each listed draw
    replaced by the FINAL rung's draw that lists it) — plus the FINAL READOUT of the merged reading at the exit rung: its
    at-cap draws split into loops and truncated by the record of the leg that last drew each (:func:`final_readout`).
    ``gen_root`` may be several wave roots (:func:`resolve_store_root` picks the row's by its admitted pin; the rung stores
    live under the SAME root as their source by the path constructor). HELD rows are never read (a hard exclusion)."""
    key = V.cell_key(row["model_key"], row["view"], row["cell"])
    cell: dict[str, Any] = {"key": key, "model_key": row["model_key"], "view": row["view"], "cell": row["cell"], "fitted": row["fitted"],
                            "route_kind": row["route_kind"], "family": row["family"], "status": row["status"], "legs": {}, "chain": [],
                            "readings": {}, "ladder": {"exits": {}, "promoted": [], "n_promotions": 0, "residual": None, "decision_problems": []},
                            "superseded_stores": [], "gen_root": None, "final_readout": None}
    if row["status"] == IC.STATUS_HELD:
        cell["held_reason"] = row["held_reason"]
        return cell
    if row["status"] == IC.STATUS_DEFERRED:
        # the follow-up: a design slot with no object sha — never read, never a leg
        cell["deferred"] = dict(row.get("deferred") or {})
        return cell
    roots = _gen_roots(gen_root)
    chosen, candidates, problem = resolve_store_root(roots, row, ctx=ctx)
    cell["superseded_stores"] = [c for c in candidates if not c["admitted"]]
    if problem:
        base = _leg_skeleton(row, CAP_BASE, Path(roots[0]) / IC.ifeval_paths(row["model_key"], row["view"], row["cell"], CAP_BASE).out_root)
        base["refused"] = problem
        cell["legs"][str(CAP_BASE)] = base
        return cell
    gen_root = chosen
    cell["gen_root"] = str(gen_root)
    base = collect_leg(gen_root, row, cap=CAP_BASE, ctx=ctx)
    cell["legs"][str(CAP_BASE)] = base
    if not usable(base):
        return cell
    current, current_cap = base, CAP_BASE
    merged = {(d["item_id"], d["sample_idx"]): d for d in (_draw_record(it, source=READING_PASS1) for it in base["items"])}
    while True:
        rec = current.get("ladder_record") or {}
        dec = current.get("ladder_decision") or {}
        cell["ladder"]["exits"][str(current_cap)] = {
            # THE decision (over the merged reading) — what the chain follows (C1)
            "exit": dec.get("exit"), "next_cap": dec.get("next_cap"), "reading": dec.get("reading"), "chain": dec.get("chain"),
            "n_draws": dec.get("n_draws"), "n_at_cap": dec.get("n_at_cap"), "share_at_cap": dec.get("share_at_cap"), "n_rung": dec.get("n_rung"),
            # the store's own classification over its own population — a diagnostic
            "store": {"exit": rec.get("exit"), "next_cap": rec.get("next_cap"), "n_draws": rec.get("n_draws"), "n_at_cap": rec.get("n_at_cap"),
                      "share_at_cap": rec.get("share_at_cap"), "n_truncated": rec.get("n_truncated"), "n_loop": rec.get("n_loop"),
                      "n_boundary": rec.get("n_boundary")},
            "trigger_provenance": rec.get("trigger_provenance")}
        if current.get("decision_problem"):
            cell["ladder"]["decision_problems"].append(f"{current_cap}: {current['decision_problem']}")
            break
        # the decision's merged population against THIS walk's merge (two independent compositions of the same chain)
        n_at_here = sum(1 for d in merged.values() if d["at_cap"])
        if dec.get("n_draws") != len(merged) or dec.get("n_at_cap") != n_at_here:
            cell["ladder"]["decision_problems"].append(
                f"{current_cap}: the record decided over {dec.get('n_at_cap')} at cap of {dec.get('n_draws')} draws but the chain merges to "
                f"{n_at_here} of {len(merged)}")
        if dec.get("exit") != CL.ESCALATE:
            break
        next_cap = int(dec.get("next_cap") or 0)
        if next_cap not in CAPS:
            cell["ladder"]["problem"] = f"the decision at {current_cap} names a rung {next_cap} off the ladder"
            break
        subset_path = Path(gen_root) / IC.ifeval_paths(row["model_key"], row["view"], row["cell"], next_cap).subset
        subset = V.read_json_if_file(subset_path)
        # C2: the subset's recorded source root is compared as a WAVE-RELATIVE layout (the recorded root is absolute and
        # relocates with every pull: the volume's on the volume, the planner's local root in a pull) + the identity hash
        source_layout = IC.ifeval_paths(row["model_key"], row["view"], row["cell"], current_cap).out_root
        prob = (V.rung_subset_problem(subset, record=rec, decision=dec, source_manifest=current["manifest"], source_out_root=source_layout)
                if subset is not None else f"no {subset_path}")
        hop = {"from_cap": current_cap, "to_cap": next_cap, "subset_path": str(subset_path), "subset_problem": prob,
               "n_draws": (subset or {}).get("n_draws"), "n_items": (subset or {}).get("n_items")}
        cell["chain"].append(hop)
        cell["ladder"]["promoted"].append(next_cap)
        if prob:
            cell["ladder"]["problem"] = prob
            break
        rung = collect_leg(gen_root, row, cap=next_cap, ctx=ctx, subset=subset, source_manifest=current["manifest"])
        cell["legs"][str(next_cap)] = rung
        if not usable(rung):
            break
        # the merge, per (item, trial): the rung draw replaces the source draw for exactly the listed draws
        listed = {(str(i), int(s)) for i, s in subset["draws"]}
        rung_by = {(str(it["item_id"]), int(it["sample_idx"])): it for it in rung["items"]}
        redraws = Counter()
        for d in listed:
            new = rung_by[d]
            outcome = (CL.REDRAW_STILL_AT_CAP if new.get("at_cap") else
                       CL.REDRAW_BOUNDARY_LANDING if (new.get("completion_tokens") is not None and int(new["completion_tokens"]) < current_cap)
                       else CL.REDRAW_EXTENDED)
            redraws[outcome] += 1
            merged[d] = {**_draw_record(new, source=f"rung{next_cap}"), "redraw_outcome": outcome}
        hop["redraw_outcomes"] = dict(sorted(redraws.items()))
        current, current_cap = rung, next_cap
    cell["ladder"]["n_promotions"] = len(cell["ladder"]["promoted"])
    merged_draws = list(merged.values())
    n_res = sum(1 for d in merged_draws if d["at_cap"])
    cell["ladder"]["residual"] = {"n_at_cap": n_res, "share_at_cap": n_res / len(merged_draws) if merged_draws else None,
                                  "final_cap": current_cap, "final_exit": (current.get("ladder_decision") or {}).get("exit"),
                                  "final_store_exit": (current.get("ladder_record") or {}).get("exit")}
    cell["readings"][READING_PASS1] = [_draw_record(it, source=READING_PASS1) for it in base["items"]]
    cell["readings"][READING_MERGED] = merged_draws
    cell["shares"] = dict(base["shares"])
    cell["shares"]["loop_rule"] = LOOP_RULE_DISCLOSURE
    cell["final_readout"] = final_readout(cell)
    cell["pins_by_cap"] = {cap: leg.get("manifest_producer_tree_sha") for cap, leg in cell["legs"].items() if usable(leg)}
    cell["recorder"] = {cap: (leg.get("recorder") or {}).get("verdict") for cap, leg in cell["legs"].items() if usable(leg)}
    return cell


def final_readout(cell: Mapping[str, Any]) -> dict[str, Any]:
    """The ladder-aware FINAL readout of one cell (the brief's per-cell contract): the MERGED reading's at-cap draws at the
    exit rung, each classified by the ladder record of the leg that LAST drew it (a pass-1 loop never re-drawn stays a
    pass-1 loop; a rung's still-at-cap draw is a loop or truncated by THAT rung's record) — so the residual at the exit is
    split into loops and genuine truncation; the exit rung and exit word (``below_threshold`` / ``ceiling`` /
    ``triggered_no_eligible`` / ``clean``) beside it. A residual draw no record classifies is counted ``unclassified`` and
    refused by the ladder gate (a loop is never assumed absent)."""
    merged = cell["readings"].get(READING_MERGED) or []
    res = cell["ladder"].get("residual") or {}
    loops_by_cap: dict[int, set[tuple[str, int]]] = {}
    elig_by_cap: dict[int, set[tuple[str, int]]] = {}
    for cap_s, leg in cell["legs"].items():
        rec = leg.get("ladder_record") if usable(leg) else None
        rec = rec if isinstance(rec, Mapping) else {}
        loops_by_cap[int(cap_s)] = {(str(i), int(t)) for i, t in rec.get("loop_draws") or []}
        elig_by_cap[int(cap_s)] = {(str(i), int(t)) for i, t in rec.get("eligible_draws") or []}
    by_source: dict[str, dict[str, int]] = {}
    n_loop = n_trunc = n_unclassified = 0
    n_pass_strict = n_pass_loose = n_loop_pass_strict = 0
    for d in merged:
        if not d.get("at_cap"):
            continue
        src = d["source"]
        cap = CAP_BASE if src == READING_PASS1 else int(str(src).replace("rung", ""))
        k = (str(d["item_id"]), int(d["sample_idx"]))
        kind = "loop" if k in loops_by_cap.get(cap, set()) else "truncated" if k in elig_by_cap.get(cap, set()) else "unclassified"
        by_source.setdefault(src, {"loop": 0, "truncated": 0, "unclassified": 0})[kind] += 1
        n_loop += kind == "loop"
        n_trunc += kind == "truncated"
        n_unclassified += kind == "unclassified"
        # what the checker made of the at-cap responses AS PRODUCED: an at-cap draw is not a failed draw
        n_pass_strict += bool(d.get("prompt_level_strict"))
        n_pass_loose += bool(d.get("prompt_level_loose"))
        n_loop_pass_strict += (kind == "loop") and bool(d.get("prompt_level_strict"))
    n = len(merged)
    n_at = n_loop + n_trunc + n_unclassified
    return {"reading": READING_MERGED, "exit_cap": res.get("final_cap"), "exit": res.get("final_exit"), "store_exit": res.get("final_store_exit"),
            "n_draws": n, "n_at_cap": n_at, "share_at_cap": (n_at / n if n else None), "n_loop": n_loop, "n_truncated": n_trunc,
            "n_unclassified": n_unclassified, "share_loop": (n_loop / n if n else None), "share_truncated": (n_trunc / n if n else None),
            "n_at_cap_pass_prompt_strict": n_pass_strict, "n_at_cap_pass_prompt_loose": n_pass_loose, "n_loop_pass_prompt_strict": n_loop_pass_strict,
            "by_source": by_source, "promoted": list(cell["ladder"]["promoted"]),
            "rule": ("at-cap draws of the MERGED reading, each classified by the ladder record of the leg that last drew it; "
                     "loop = the cap ladder's whitespace 4-gram rule; truncated = at cap and not a loop"),
            "loops_redrawn": False,
            "protocol": ("the ladder re-draws only at-cap NON-loop draws; a loop (degenerate at-cap text) is never re-drawn, so what it "
                         "would have produced past the cap was not measured — the at-cap responses are scored as produced, and the "
                         "observed net change of the ladder is the truncation-cost contrast (merged - r2048) with its CI")}


# --------------------------------------------------------------------------- the estimands
def _boot(legs: Mapping[str, Any], contrasts: Mapping[str, Mapping[str, int]], ctx: Context) -> Mapping[str, Any]:
    """The ONE estimator call (P13): the item-cluster paired percentile bootstrap under the run's B / seed / level."""
    return IB.paired_item_bootstrap_ratios(legs, contrasts, n_resamples=ctx.n_resamples, seed=ctx.seed, ci=ctx.ci)


def _triples(draws: Sequence[Mapping[str, Any]], metric: str) -> dict[str, IB.Triple]:
    return IB.ifeval_leg_triples(draws, metric=METRIC_FIELD[metric])


def _stat(est: Mapping[str, Any], name: str) -> dict[str, Any]:
    s = est["stats"][name]
    return {k: s[k] for k in ("point", "ci_low", "ci_high", "excludes_zero", "point_sign", "bound_low", "bound_high")}


def view_estimands(cells: Mapping[str, Mapping[str, Any]], *, model_key: str, view: str, ctx: Context) -> dict[str, Any]:
    """Per (model, view), per reading, per metric: every usable cell's point + CI and its ``cell_minus_none`` contrast, ONE
    bootstrap over the view's legs (paired on items). A view without a usable ``none`` files every contrast as not run."""
    none_key = V.cell_key(model_key, view, IC.CELL_NONE)
    out: dict[str, Any] = {"model_key": model_key, "view": view, "readings": {}, "not_run": []}
    members = {k: c for k, c in cells.items() if c["model_key"] == model_key and c["view"] == view and c.get("readings")}
    if none_key not in members:
        out["not_run"] = [k for k in cells if cells[k]["model_key"] == model_key and cells[k]["view"] == view and k != none_key]
        out["problem"] = f"no usable none anchor for {model_key}/{view}"
        return out
    for reading in READINGS:
        out["readings"][reading] = {}
        for metric in METRICS:
            legs = {k: _triples(c["readings"][reading], metric) for k, c in members.items()}
            contrasts = {k: {k: 1, none_key: -1} for k in members if k != none_key}
            try:
                est = _boot(legs, contrasts or {"_none": {none_key: 1}}, ctx)
            except IB.ItemBootstrapError as e:
                raise AnalysisError(f"{model_key}/{view} {reading} {metric}: {e}") from None
            out["readings"][reading][metric] = {
                "legs": {k: est["acc_by_leg"][k] for k in legs},
                "contrasts": {k: _stat(est, k) for k in contrasts},
                "B": est["B"], "seed": est["seed"], "level": est["level"], "n_items": est["n_items"],
            }
    return out


def verdict_word(view_est: Mapping[str, Any], cell_key: str) -> dict[str, Any]:
    """A directional word iff the MERGED reading's CI excludes zero on BOTH prompt-level metrics with agreeing sign; else
    ``not resolved`` naming what failed."""
    merged = (view_est.get("readings") or {}).get(READING_MERGED) or {}
    stats = {m: (merged.get(m) or {}).get("contrasts", {}).get(cell_key) for m in PROMPT_METRICS}
    if any(s is None for s in stats.values()):
        return {"word": None, "reason": "contrast not run", "side": 0}
    sides = {m: (1 if s["ci_low"] > 0 else -1 if s["ci_high"] < 0 else 0) for m, s in stats.items()}
    if all(v != 0 for v in sides.values()) and len(set(sides.values())) == 1:
        side = next(iter(sides.values()))
        return {"word": WORD_HIGHER if side > 0 else WORD_LOWER, "reason": "both prompt-level CIs exclude zero on one side (merged)", "side": side}
    failed = [m for m, v in sides.items() if v == 0] or ["the two prompt-level metrics disagree in sign"]
    return {"word": WORD_NOT_RESOLVED, "reason": f"not excluded: {failed}", "side": 0}


# --------------------------------------------------------------------------- the noise floors
#: the primary judged outcome the floor is read on; the other three metrics are reported alongside
NOISE_PRIMARY_METRIC = "prompt_strict"
LABEL_FLOOR, LABEL_MEASURES, LABEL_NOT_IDENTICAL = "noise_floor", "measures_difference", "not_identical_by_construction"
#: the per-draw classes of a pair on one PROMPT-level metric (the same judged outcome, main vs it) …
CLASS_BOTH_PASS, CLASS_BOTH_FAIL, CLASS_MAIN_ONLY, CLASS_IT_ONLY, CLASS_UNSCORED = ("both_pass", "both_fail", "main_only_pass", "it_only_pass",
                                                                                     "unscored_either")
#: … and on an INSTRUCTION-level one (a followed-instruction count per draw)
CLASS_EQUAL, CLASS_MAIN_HIGHER, CLASS_IT_HIGHER = "equal", "main_higher", "it_higher"
PROMPT_CLASSES: tuple[str, ...] = (CLASS_BOTH_PASS, CLASS_BOTH_FAIL, CLASS_MAIN_ONLY, CLASS_IT_ONLY, CLASS_UNSCORED)
INST_CLASSES: tuple[str, ...] = (CLASS_EQUAL, CLASS_MAIN_HIGHER, CLASS_IT_HIGHER, CLASS_UNSCORED)
DISCORDANT_CLASSES: frozenset[str] = frozenset({CLASS_MAIN_ONLY, CLASS_IT_ONLY, CLASS_MAIN_HIGHER, CLASS_IT_HIGHER})
#: the matrix-row fields two views SHARE by construction when the pair is a floor candidate (the design), and the fields the
#: design VARIES between the views (expected to differ; never label-deciding)
DESIGN_SHARED_FIELDS: tuple[str, ...] = ("hf", "rev", "cell", "route_kind", "fitted", "object_sha256", "pin", "harness_cell", "standup_cell",
                                         "standup_harness_cell", "prompt_class", "si_text_sha256", "thinking", "samples_per_item", "sampling",
                                         "sampling_mode", "geometry_eval_by_cap", "expected_stamps", "family", "image", "header_located",
                                         "declared", "caps", "cap_ceiling", "budget_class")
DESIGN_VARIED_FIELDS: tuple[str, ...] = ("view", "slug", "tag", "job_id", "index", "config", "manifest", "out_root", "staged_id")
SCORING_SENTENCE = ("IFEval scoring is deterministic rule checking of the response text (inspect_evals' stock task: prompt- and "
                    "instruction-level, strict and loose); no judge model is consulted, so there is no judge cache whose state could "
                    "differ between the two legs — the one seeded component (langdetect, for the language instructions) runs under the "
                    "same recorded seed on both")
NOISE_RULING = ("Orchestrator 2026-09-12 (Llama D34; this lane's D33): two cells identical by construction are a measured noise floor — the "
                "between-view discrepancy per draw and in aggregate on the same judged outcome, never collapsed and never pooled; the "
                "paper reader takes one view's row for the figure and the pair's spread is a caption number; the caution states what is "
                "SHARED by construction versus what merely matches, and checks serving geometry, batch composition and the scoring path — "
                "else the pair measures that difference. A pair whose two rows are NOT identical by construction (different served "
                "weights, a fitted object on either view) is refused as a floor.")


def pair_design(row_main: Mapping[str, Any], row_it: Mapping[str, Any]) -> dict[str, Any]:
    """What the MATRIX says the two rows of a candidate pair share by construction (:data:`DESIGN_SHARED_FIELDS`, compared
    field by field) and what the design varies between them (:data:`DESIGN_VARIED_FIELDS`, recorded); identical by
    construction iff every shared field is equal AND neither row serves an object (a fitted object on either view is the
    design's own between-view comparison, never a floor — main = base-fit objects, it = instruct-direct fits)."""
    shared = [{"name": f, "main": row_main.get(f), "it": row_it.get(f), "equal": _eq(row_main.get(f), row_it.get(f))} for f in DESIGN_SHARED_FIELDS]
    differences = [s["name"] for s in shared if not s["equal"]]
    objects = [v for v in (row_main.get("object_sha256"), row_it.get("object_sha256")) if v is not None]
    fitted = bool(row_main.get("fitted")) or bool(row_it.get("fitted"))
    identical = not differences and not objects and not fitted
    reason = None
    if objects or fitted:
        reason = "a served object on at least one view (the design's between-view comparison, not a floor)"
    elif differences:
        reason = "the two rows differ by construction in: " + ", ".join(f"{s['name']} ({s['main']!r} vs {s['it']!r})" for s in shared if not s["equal"])
    return {"identical_by_construction": identical, "shared": shared, "differences": differences, "reason": reason,
            "varied": {f: {"main": row_main.get(f), "it": row_it.get(f)} for f in DESIGN_VARIED_FIELDS},
            "served_weights": {"hf": row_main.get("hf"), "rev": row_main.get("rev")} if "hf" not in differences and "rev" not in differences else None,
            "si_text_sha256": row_main.get("si_text_sha256") if "si_text_sha256" not in differences else None}


def _draw_key(d: Mapping[str, Any]) -> str:
    return f"{d['item_id']}/{int(d['sample_idx'])}"


def per_draw_discordance(draws_main: Sequence[Mapping[str, Any]], draws_it: Sequence[Mapping[str, Any]], *, metric: str) -> dict[str, Any]:
    """The pair's per-draw reading on ONE metric: every common (item, trial) draw classed on the same judged outcome — a
    prompt-level metric by pass / fail per leg, an instruction-level one by the followed count — the discordant draws listed
    per direction (never collapsed into the aggregate alone), and the items with at least one discordant draw counted."""
    field, prompt_level = METRIC_FIELD[metric], metric.startswith("prompt")
    dm, di = {_draw_key(d): d for d in draws_main}, {_draw_key(d): d for d in draws_it}
    if set(dm) != set(di):
        raise AnalysisError(f"noise floor {metric}: the two legs' draw sets differ ({len(dm)} vs {len(di)}; {len(set(dm) & set(di))} common) — no pairing")
    keys = sorted(dm)
    classes: dict[str, str] = {}
    abs_diff = 0
    for k in keys:
        a, b = dm[k], di[k]
        if not (a.get("scored") and b.get("scored")):
            classes[k] = CLASS_UNSCORED
            continue
        if prompt_level:
            va, vb = bool(a.get(field)), bool(b.get(field))
            classes[k] = CLASS_BOTH_PASS if va and vb else CLASS_MAIN_ONLY if va else CLASS_IT_ONLY if vb else CLASS_BOTH_FAIL
        else:
            va, vb = int(a.get(field) or 0), int(b.get(field) or 0)
            classes[k] = CLASS_EQUAL if va == vb else CLASS_MAIN_HIGHER if va > vb else CLASS_IT_HIGHER
            abs_diff += abs(va - vb)
    counts = Counter(classes.values())
    names = PROMPT_CLASSES if prompt_level else INST_CLASSES
    directional = [c for c in names if c in DISCORDANT_CLASSES]
    discordant_items = {k.rsplit("/", 1)[0] for k in keys if classes[k] in DISCORDANT_CLASSES}
    return {"metric": metric, "n_draws": len(keys), "n_items": len({k.rsplit("/", 1)[0] for k in keys}),
            "classes": {c: int(counts.get(c, 0)) for c in names},
            "n_discordant_draws": sum(int(counts.get(c, 0)) for c in directional), "n_discordant_items": len(discordant_items),
            "discordant_draws": {c: [k for k in keys if classes[k] == c] for c in directional},
            **({} if prompt_level else {"sum_abs_diff_followed": abs_diff}),
            "n_at_cap_by_leg": {"main": sum(1 for k in keys if dm[k].get("at_cap")), "it": sum(1 for k in keys if di[k].get("at_cap"))},
            "n_at_cap_both": sum(1 for k in keys if dm[k].get("at_cap") and di[k].get("at_cap"))}


def _eq(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def _present(x: Any) -> bool:
    """Whether a recorded value carries any content (a mapping with every leaf null is absent)."""
    if isinstance(x, Mapping):
        return any(_present(v) for v in x.values())
    return x is not None


def _shared(name: str, a: Any, b: Any) -> dict[str, Any]:
    pa, pb = _present(a), _present(b)
    return {"name": name, "main": a, "it": b, "equal": _eq(a, b), "present": pa and pb, "present_any": pa or pb}


def _check(name: str, passed: bool, detail: str, *, attested: bool = True) -> dict[str, Any]:
    """A check that is NOT attestable (the field is recorded on neither leg) is filed with ``passed: None`` and never counts
    as a difference — the label text names it instead; a field present on ONE leg only is a difference."""
    if not attested:
        return {"name": name, "passed": None, "attested": False, "detail": f"not recorded on either leg — cannot be attested: {detail}"}
    return {"name": name, "passed": bool(passed), "attested": True, "detail": detail}


def _field_check(shared: Mapping[str, Any], detail: str) -> dict[str, Any]:
    return _check(shared["name"], shared["equal"], detail, attested=bool(shared["present_any"]))


def store_facts(leg: Mapping[str, Any]) -> dict[str, Any]:
    """What one leg's STORE says about its configuration and its wire traffic — the validated manifest / summary / endpoint /
    certification / ledger / scoring-path records the reader already holds, plus the accepted attempt's per-draw wire digest
    (``V.wire_digest``) and the per-draw seeds off the outcome record. Read from the same documents the gates read; nothing
    is re-derived."""
    m, s = leg["manifest"], leg["summary"]
    cert, led, sp = leg.get("certification") or {}, leg.get("ledger") or {}, leg.get("scoring_path") or {}
    ep, wire = leg.get("endpoint") or {}, leg.get("wire") or {}
    tiny = (m.get("resolved_config") or {}).get(C.MANIFEST_KEY_TINYCAP) or m.get(C.MANIFEST_KEY_TINYCAP) or {}
    route = m.get(C.MANIFEST_KEY_ROUTE) or {}
    geom = m.get(C.MANIFEST_KEY_GEOMETRY) or {}
    return {
        "key": leg["key"], "out_root": leg["out_root"], "n_draws": leg["n_draws"], "n_items": leg["n_items"],
        "model": {"model_id": m.get("model_id"), "model_revision": m.get("model_revision")},
        "template_hash": ep.get(C.ENDPOINT_KEY_CHAT_TEMPLATE_HASH),
        "serving_identity": route.get("serving_identity"), "virtual_model": m.get("virtual_model"),
        "geometry": {"eval": cert.get("geometry_eval"), "max_model_len": cert.get("max_model_len"), "served_seqs": cert.get("served_seqs"),
                     "manifest_max_model_len": geom.get("max_model_len"), "manifest_max_num_seqs": geom.get("max_num_seqs"), "manifest_eval": geom.get("eval")},
        "kv_shrink": cert.get("kv_shrink"),
        "concurrency": m.get(C.MANIFEST_KEY_CONCURRENCY),
        "sampling": {"block": m.get(C.MANIFEST_KEY_SAMPLING), C.SAMPLING_REF_FIELD: s.get(C.SAMPLING_REF_FIELD),
                     "summary": {k: s.get(k) for k in ("temperature", "top_p", "top_k")}},
        "cap": {"max_tokens_by_task": tiny.get(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK), "off_cap_by_task": V.off_cap_statement(m) or None,
                "summary_max_tokens": s.get("max_tokens")},
        "thinking": {"manifest": m.get(C.MANIFEST_KEY_THINKING), "by_task": tiny.get(C.IDENTITY_KEY_THINKING_BY_TASK), "summary": s.get("thinking"),
                     "template_kwargs": cert.get("template_kwargs"), "switch": cert.get("thinking_switch")},
        "seed_rule": tiny.get(C.IDENTITY_KEY_SEED_RULE),
        "cell": {"cell_id": m.get("cell_id"), "harness_cell_id": m.get("harness_cell_id"), "object_artifact_sha256": route.get(C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256)},
        # derived from the hashed configs (the producer's); recorded and compared, but not label-deciding: a differing hash says
        # the hashed config differs at least in a label (the view rides it), not that the serving did
        "identity_hashes": {"intervention_hash": m.get("intervention_hash"), "harness_config_hash": m.get("harness_config_hash")},
        "producer_tree_sha": (m.get(C.MANIFEST_KEY_PRODUCER) or {}).get("tree_sha"),
        "frameworks": tiny.get("frameworks"),
        "scoring": {"ifeval_task_version": sp.get("ifeval_task_version"), "langdetect_seed": sp.get("langdetect_seed"),
                    "samples_per_item": sp.get("samples_per_item"), "thinking": sp.get("thinking"), "failures": list(sp.get("failures") or [])},
        "ledger": {"passed": led.get("passed"), "n_forwarded": led.get("n_forwarded"), "n_draws": led.get("n_draws"),
                   "accepted_attempt": led.get("accepted_attempt"), "rows_by_attempt": led.get("rows_by_attempt")},
        "designed": {"view": route.get("view"), "bank_view": route.get("bank_view")},
        "incidental": {"host": m.get("host"), "proxy_url": m.get("proxy_url"), "ledger_url": m.get("ledger_url"),
                       "generation_span": led.get("generation_span"), "wall_clock_s": s.get("wall_clock_s"),
                       "n_attempts": len(m.get(C.MANIFEST_KEY_RESUME_ATTEMPTS) or [])},
        "wire": wire,
        "seeds_by_draw": {_draw_key(it): it.get("seed") for it in leg["items"]},
    }


def _per_draw_equal(a: Mapping[str, Any], b: Mapping[str, Any]) -> tuple[int, int]:
    """``(n equal, n common)`` over two per-draw maps, a value counted equal only when present on both."""
    common = sorted(set(a) & set(b))
    return sum(1 for k in common if a[k] is not None and a[k] == b[k]), len(common)


def attest_pair(leg_main: Mapping[str, Any], leg_it: Mapping[str, Any], *, design: Mapping[str, Any],
                ladder: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The shared-configuration ATTESTATION of one pair: ``shared`` = what the two legs' stores share BY CONSTRUCTION,
    each field read from both and compared; ``designed_difference`` = the one thing the design varies (the bank view, which on
    a no-object cell routes to the same weights); ``incidental`` = what merely happens to match or differ (hosts, endpoints,
    wall clock, attempts); ``checks`` = the ruling's three (serving geometry, batch composition, the scoring path) plus the
    identity of everything the request carried (model revision, template, sampling row, cap, thinking, cell, producer tree,
    SI prefix, request bytes, seeds, prompt tokens); ``label`` = ``noise_floor`` iff every attested check passed, else
    ``measures_difference`` naming the differing checks. The two views' later promotion chains are RECORDED
    (``ladder_promotions``) and never decide this label — its scope is pass 1; the merged reading's own
    attestation reads the chains."""
    fm, fi = store_facts(leg_main), store_facts(leg_it)
    geo = lambda f: {k: f["geometry"][k] for k in ("eval", "max_model_len", "served_seqs", "manifest_max_model_len", "manifest_max_num_seqs", "manifest_eval")}
    shared = [_shared("model_revision", fm["model"], fi["model"]), _shared("served_template_hash", fm["template_hash"], fi["template_hash"]),
              _shared("serving_identity", fm["serving_identity"], fi["serving_identity"]), _shared("virtual_model", fm["virtual_model"], fi["virtual_model"]),
              _shared("serving_geometry", geo(fm), geo(fi)), _shared("kv_shrink", fm["kv_shrink"], fi["kv_shrink"]),
              _shared("sampling_row", fm["sampling"], fi["sampling"]), _shared("cap", fm["cap"], fi["cap"]),
              _shared("thinking_mode", fm["thinking"], fi["thinking"]), _shared("seed_rule", fm["seed_rule"], fi["seed_rule"]),
              _shared("cell_identity", fm["cell"], fi["cell"]), _shared("producer_tree_sha", fm["producer_tree_sha"], fi["producer_tree_sha"]),
              _shared("frameworks", fm["frameworks"], fi["frameworks"]), _shared("identity_hashes", fm["identity_hashes"], fi["identity_hashes"]),
              _shared("scoring_path", fm["scoring"], fi["scoring"])]
    by_name = {s["name"]: s for s in shared}
    wm, wi = fm["wire"], fi["wire"]
    n_body_eq, n_body = _per_draw_equal(wm.get("body_sha256_by_draw") or {}, wi.get("body_sha256_by_draw") or {})
    n_seed_eq, n_seed = _per_draw_equal(fm["seeds_by_draw"], fi["seeds_by_draw"])
    n_wseed_eq, n_wseed = _per_draw_equal(wm.get("seed_by_draw") or {}, wi.get("seed_by_draw") or {})
    n_pt_eq, n_pt = _per_draw_equal(wm.get("prompt_tokens_by_draw") or {}, wi.get("prompt_tokens_by_draw") or {})
    n_cap_eq, n_cap = _per_draw_equal(wm.get("max_tokens_by_draw") or {}, wi.get("max_tokens_by_draw") or {})
    n_expected = min(fm["n_draws"], fi["n_draws"])
    wire_ok = not wm.get("problem") and not wi.get("problem")
    sm, si = fm["scoring"], fi["scoring"]
    scoring_ok = (not sm["failures"] and not si["failures"] and by_name["scoring_path"]["equal"])
    geo_present = all(_present(f["geometry"].get("max_model_len")) and _present(f["geometry"].get("served_seqs")) for f in (fm, fi))
    prom_m, prom_i = list((ladder or {}).get("main") or []), list((ladder or {}).get("it") or [])
    checks = [
        _check("serving_geometry", by_name["serving_geometry"]["equal"],
               f"the geometry entry, max_model_len and the served S read from both endpoints / manifests: {geo(fm)} vs {geo(fi)}", attested=geo_present),
        _check("batch_composition", fm["concurrency"] is not None and fm["concurrency"] == fi["concurrency"]
               and fm["ledger"]["n_forwarded"] == fi["ledger"]["n_forwarded"] == fm["ledger"]["n_draws"] == fi["ledger"]["n_draws"]
               and wire_ok and n_cap == n_expected > 0 and n_cap_eq == n_cap,
               f"S (the standup's max_num_seqs = the client's in-flight concurrency) {fm['concurrency']} vs {fi['concurrency']}; forwarded ledger "
               f"task rows {fm['ledger']['n_forwarded']} vs {fi['ledger']['n_forwarded']} for {fm['ledger']['n_draws']} / {fi['ledger']['n_draws']} draws; "
               f"max_tokens equal on {n_cap_eq} of {n_cap} common draws. The per-step co-scheduled set is not recorded by either leg and is "
               "exactly what a fixed-seed floor measures",
               attested=fm["concurrency"] is not None or fi["concurrency"] is not None),
        _check("scoring_path", scoring_ok,
               (SCORING_SENTENCE + f" (task version {sm['ifeval_task_version']!r} / {si['ifeval_task_version']!r}; langdetect seed "
                f"{sm['langdetect_seed']!r} / {si['langdetect_seed']!r}; k {sm['samples_per_item']!r} / {si['samples_per_item']!r}; arm "
                f"{sm['thinking']!r} / {si['thinking']!r}; scoring-path failures {sm['failures']} / {si['failures']})")),
        _field_check(by_name["model_revision"], f"resolved model and revision {fm['model']} vs {fi['model']}"),
        _field_check(by_name["served_template_hash"], f"endpoint chat_template_hash {fm['template_hash']!r} vs {fi['template_hash']!r}"),
        _field_check(by_name["sampling_row"], f"{fm['sampling']} vs {fi['sampling']}"),
        _field_check(by_name["cap"], f"{fm['cap']} vs {fi['cap']}"),
        _field_check(by_name["thinking_mode"], f"{fm['thinking']} vs {fi['thinking']}"),
        _field_check(by_name["seed_rule"], f"{fm['seed_rule']} vs {fi['seed_rule']}"),
        _field_check(by_name["cell_identity"], f"cell id, harness cell, object sha {fm['cell']} vs {fi['cell']}"),
        _field_check(by_name["producer_tree_sha"], f"{fm['producer_tree_sha']!r} vs {fi['producer_tree_sha']!r}"),
        _field_check(by_name["frameworks"], f"{fm['frameworks']} vs {fi['frameworks']}"),
        _check("si_prefix", "si_text_sha256" not in design.get("differences", ()),
               f"the matrix row's si_text_sha256 (the SI prefix both views serve as the system turn; null on a bare cell): {design.get('si_text_sha256')!r}"),
        _check("request_bytes", wire_ok and n_expected > 0 and n_body == n_expected and n_body_eq == n_expected,
               f"task request bodies byte-identical (sha256) on {n_body_eq} of {n_body} common draws ({len(wm.get('body_sha256_by_draw') or {})} / "
               f"{len(wi.get('body_sha256_by_draw') or {})} accepted-attempt task records; {n_expected} draws) — the SI prefix, the prompt, the sampler "
               "keys, max_tokens and the seed all ride the body" + (f"; wire problem: {wm.get('problem') or wi.get('problem')}" if not wire_ok else ""),
               attested=bool(wm.get("body_sha256_by_draw") or wi.get("body_sha256_by_draw") or not wire_ok)),
        _check("seeds", n_expected > 0 and n_seed == n_expected and n_seed_eq == n_expected and (n_wseed == 0 or n_wseed_eq == n_wseed),
               f"per-draw seeds equal on {n_seed_eq} of {n_seed} common draws (the outcome records) and on {n_wseed_eq} of {n_wseed} (the request bodies)",
               attested=any(v is not None for f in (fm, fi) for v in f["seeds_by_draw"].values())),
        _check("prompt_tokens", n_pt_eq == n_pt == n_expected > 0,
               f"the server's prompt_tokens_pre equal on {n_pt_eq} of {n_pt} common ledger rows — the rendered prompt (template + SI + question) "
               "is the same length on both legs", attested=n_pt > 0),
    ]
    # the two views' promotion chains are RECORDED beside the pass-1 attestation, never one of its deciding checks:
    # this attestation's scope is the 2,048 request bodies, which no later promotion changes — two cells with byte-identical
    # pass-1 requests are a pass-1 floor whatever each view's ladder did afterwards; a differing chain is what the MERGED
    # reading's attestation (:func:`attest_merged_reading`) reads, counts per draw and labels
    ladder_promotions = {"main": prom_m, "it": prom_i, "equal": prom_m == prom_i, "deciding": False,
                         "note": ("rungs promoted to, per view — recorded beside the pass-1 attestation and NOT one of its checks (the pass-1 "
                                  "request bodies predate every promotion); a differing chain mixes caps across the pair under the MERGED "
                                  "reading, which its own attestation reads and labels")}
    failing = [c["name"] for c in checks if c["attested"] and not c["passed"]]
    not_attestable = [c["name"] for c in checks if not c["attested"]]
    hashes_note = ("; the producer's intervention and harness-config hashes are equal across the two legs" if by_name["identity_hashes"]["equal"]
                   else "; the producer's intervention / harness-config hashes differ across the two legs (derived from the hashed configs, which "
                        "may carry the view label; recorded, not label-deciding)")
    label = LABEL_FLOOR if not failing else LABEL_MEASURES
    return {
        # the SCOPE of this attestation: the two PASS-ONE stores' request bodies (every draw at 2,048); the
        # merged reading's final draws are attested separately by :func:`attest_merged_reading`
        "scope": ATTESTATION_SCOPE_PASS1,
        "shared": shared,
        "designed_difference": {"bank_view": {"main": fm["designed"], "it": fi["designed"]},
                                "note": "the ONE thing the design varies; on a cell with no object the route serves the same weights either way "
                                        "(object_artifact_sha256 null on both legs)"},
        "incidental": {"main": fm["incidental"], "it": fi["incidental"],
                       "note": "hosts, endpoints, wall clock and attempt counts — separate standups; not part of the construction, expected to differ"},
        "wire": {"n_draws": n_expected, "n_request_bodies_identical": n_body_eq, "n_common_requests": n_body, "n_seeds_identical": n_seed_eq,
                 "n_prompt_tokens_identical": n_pt_eq, "n_common_ledger_rows": n_pt, "n_max_tokens_identical": n_cap_eq},
        "checks": checks, "failing": failing, "not_attestable": not_attestable, "label": label,
        # recorded, never deciding — the pass-1 label is decided by `checks` alone
        "ladder_promotions": ladder_promotions,
        "label_text": (("a measured noise floor: the two legs share the served checkpoint and revision, the SI prefix, the geometry entry and "
                        "served S, the sampling row, the cap, the thinking arm, the per-draw seeds and the request bytes by construction; what "
                        "differs is the standup and its per-step batch composition" + hashes_note
                        + (f" (not attestable from these stores: {', '.join(not_attestable)})" if not_attestable else "")) if not failing else
                       f"NOT a noise floor — the pair measures the difference in: {', '.join(failing)}"),
    }


ATTESTATION_SCOPE_PASS1 = f"pass 1 ({CAP_BASE} request bodies on both legs; the r{CAP_BASE} reading)"
#: the MERGED reading's two labels: a pair whose final draws share request bytes AND caps on every draw is still a
#: byte-identical-request floor under the merged reading; a pair whose two views' ladders re-drew differently is the
#: variability of the WHOLE adaptive protocol (pass 1 + each view's own redraw chain), which is what the merged floor then measures
MERGED_LABEL_BYTE_IDENTICAL = "byte-identical-request floor (merged reading)"
MERGED_LABEL_PROTOCOL = "variability of the whole adaptive protocol, not a byte-identical-request floor (merged reading)"


def _final_wire(cell: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Per draw of the MERGED reading: the cap and request-body sha256 of the leg that LAST drew it (pass 1 or a rung), from
    that leg's accepted-attempt wire digest."""
    out: dict[str, dict[str, Any]] = {}
    for d in cell["readings"].get(READING_MERGED) or []:
        src = d["source"]
        cap = CAP_BASE if src == READING_PASS1 else int(str(src).replace("rung", ""))
        wire = (cell["legs"].get(str(cap)) or {}).get("wire") or {}
        k = _draw_key(d)
        out[k] = {"cap": cap, "source": src, "max_tokens": (wire.get("max_tokens_by_draw") or {}).get(k),
                  "body_sha256": (wire.get("body_sha256_by_draw") or {}).get(k)}
    return out


def attest_merged_reading(cell_main: Mapping[str, Any], cell_it: Mapping[str, Any]) -> dict[str, Any]:
    """The MERGED reading's request-byte attestation of a floor pair: for every draw, the leg that last drew it
    on each view — its cap and its request body's sha256 — compared main vs it. ``n_final_cap_differs`` counts the draws whose
    final caps differ (one view re-drew it at a rung the other did not reach: the adaptive ladder ran differently on the two
    views), ``n_final_body_identical`` the draws whose final request bodies are byte-identical. The label is
    :data:`MERGED_LABEL_BYTE_IDENTICAL` only when every final draw is identical in both; else
    :data:`MERGED_LABEL_PROTOCOL`. A pair with no promotion on either view is identical trivially (merged == pass 1) and says so.
    Redraw eligibility is never changed to force equality (the brief)."""
    wm, wi = _final_wire(cell_main), _final_wire(cell_it)
    common = sorted(set(wm) & set(wi))
    n_cap_differs = sum(1 for k in common if wm[k]["cap"] != wi[k]["cap"])
    n_body_present = sum(1 for k in common if wm[k]["body_sha256"] is not None and wi[k]["body_sha256"] is not None)
    n_body_eq = sum(1 for k in common if wm[k]["body_sha256"] is not None and wm[k]["body_sha256"] == wi[k]["body_sha256"])
    n_mt_eq = sum(1 for k in common if wm[k]["max_tokens"] is not None and wm[k]["max_tokens"] == wi[k]["max_tokens"])
    differing = [{"draw": k, "main": {"cap": wm[k]["cap"], "source": wm[k]["source"]}, "it": {"cap": wi[k]["cap"], "source": wi[k]["source"]},
                  "body_identical": wm[k]["body_sha256"] is not None and wm[k]["body_sha256"] == wi[k]["body_sha256"]}
                 for k in common if wm[k]["cap"] != wi[k]["cap"] or wm[k]["body_sha256"] != wi[k]["body_sha256"]]
    prom_m, prom_i = list(cell_main["ladder"]["promoted"]), list(cell_it["ladder"]["promoted"])
    complete = len(common) == len(wm) == len(wi) == N_DRAWS and n_body_present == len(common)
    identical = complete and n_cap_differs == 0 and n_body_eq == len(common) and n_mt_eq == len(common)
    label = MERGED_LABEL_BYTE_IDENTICAL if identical else MERGED_LABEL_PROTOCOL
    trivially = identical and not prom_m and not prom_i
    text = (("the merged reading IS the pass-1 reading on both views (no promotion on either): the pass-1 attestation covers it; "
             f"{n_body_eq} of {len(common)} final request bodies byte-identical, 0 draws differ in cap") if trivially else
            (f"every final draw of the merged reading shares its request bytes and cap across the two views ({n_body_eq} of {len(common)} bodies "
             f"identical; promotions main {prom_m} / it {prom_i})") if identical else
            (f"the two views' ladders re-drew differently: {n_cap_differs} of {len(common)} final draws differ in cap and {len(common) - n_body_eq} "
             f"in request bytes (promotions main {prom_m} / it {prom_i}); the merged discrepancy therefore includes the adaptive ladder's "
             "own differences and is read as the variability of the whole adaptive protocol — the pass-1 attestation and its label are "
             "unchanged and scoped to pass 1"))
    return {"scope": "merged reading (each draw's FINAL request: pass 1 or the rung that last drew it)", "n_draws": len(common),
            "n_common_draws": len(common), "n_final_cap_differs": n_cap_differs, "n_final_body_identical": n_body_eq,
            "n_final_max_tokens_identical": n_mt_eq, "n_final_body_present": n_body_present, "promotions": {"main": prom_m, "it": prom_i},
            "differing_draws": differing[:50], "n_differing_draws": len(differing), "identical": identical, "trivially_identical": trivially,
            "label": label, "label_text": text,
            "rule": ("byte-identical iff every draw's final request body sha256 and cap agree across the two views; otherwise the merged "
                     "floor measures the whole adaptive protocol's variability; redraw eligibility is never changed to force equality")}


def noise_floor(cells: Mapping[str, Mapping[str, Any]], *, rows: Sequence[Mapping[str, Any]], ctx: Context) -> list[dict[str, Any]]:
    """Per model: every cell the matrix carries on BOTH views with no object on either is a floor CANDIDATE
    (``cell(it) - cell(main)``; llama8b: ``none`` and ``prompted__si_constitution``); a candidate whose two rows are identical by
    construction (:func:`pair_design`) and whose legs are usable is filed with, per reading and per metric, the item-cluster
    paired bootstrap spread (the analysis's own estimator; both legs' points beside it, never pooled) and the per-draw
    discordance (:func:`per_draw_discordance`; the primary metric is :data:`NOISE_PRIMARY_METRIC`), plus the
    shared-configuration attestation (:func:`attest_pair`) that LABELS it a floor or "measures that difference". A candidate
    that is not identical by construction is refused as a floor by name; a one-view model has no floor."""
    out: list[dict[str, Any]] = []
    by_model: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(r["model_key"], []).append(r)
    for model, mrows in by_model.items():
        views = list(dict.fromkeys(r["view"] for r in mrows))
        rec: dict[str, Any] = {"model_key": model, "views": views, "status": None, "primary_metric": NOISE_PRIMARY_METRIC, "pairs": []}
        if not ({IC.VIEW_MAIN, IC.VIEW_IT} <= set(views)):
            rec["status"] = "no floor (one view)"
            out.append(rec)
            continue
        launch = {v: {r["cell"]: r for r in mrows if r["view"] == v and r["status"] == IC.STATUS_LAUNCH} for v in (IC.VIEW_MAIN, IC.VIEW_IT)}
        for cell, rm in launch[IC.VIEW_MAIN].items():
            ri = launch[IC.VIEW_IT].get(cell)
            if ri is None or rm.get("object_sha256") is not None or ri.get("object_sha256") is not None or rm.get("fitted") or ri.get("fitted"):
                continue                     # one view only, or a served object: the design's comparison, never a floor candidate
            main_key, it_key = V.cell_key(model, IC.VIEW_MAIN, cell), V.cell_key(model, IC.VIEW_IT, cell)
            design = pair_design(rm, ri)
            pair: dict[str, Any] = {"name": f"{model}: {cell}(it) - {cell}(main)", "kind": "noise_floor", "model_key": model, "cell": cell,
                                    "main_key": main_key, "it_key": it_key, "contrast": CONTRAST_NOISE, "status": None, "reason": None,
                                    "identical_by_construction": design["identical_by_construction"], "design": design, "readings": {},
                                    "per_draw": {}, "attestation": None, "label": None, "label_text": None, "label_scope": None,
                                    "merged_attestation": None, "merged_label": None, "merged_label_text": None}
            if not design["identical_by_construction"]:
                pair.update(status="not identical by construction", reason=design["reason"], label=LABEL_NOT_IDENTICAL,
                            label_text=f"refused as a floor: {design['reason']}")
                rec["pairs"].append(pair)
                continue
            cm, ci = cells.get(main_key), cells.get(it_key)
            if not (cm and ci and cm.get("readings") and ci.get("readings")):
                missing = [k for k, c in ((main_key, cm), (it_key, ci)) if not (c and c.get("readings"))]
                pair.update(status="not run (a leg unusable)", reason=f"unusable pass-1 leg(s): {missing}")
                rec["pairs"].append(pair)
                continue
            for reading in READINGS:
                pair["readings"][reading], pair["per_draw"][reading] = {}, {}
                for metric in METRICS:
                    legs = {"it": _triples(ci["readings"][reading], metric), "main": _triples(cm["readings"][reading], metric)}
                    est = _boot(legs, {CONTRAST_NOISE: {"it": 1, "main": -1}}, ctx)
                    pair["readings"][reading][metric] = {**_stat(est, CONTRAST_NOISE),
                                                         "legs": {k: {f: est["acc_by_leg"][k][f] for f in ("point", "ci_low", "ci_high", "n_scored")} for k in ("main", "it")},
                                                         "B": est["B"], "seed": est["seed"], "level": est["level"], "n_items": est["n_items"]}
                    pair["per_draw"][reading][metric] = per_draw_discordance(cm["readings"][reading], ci["readings"][reading], metric=metric)
            att = attest_pair(cm["legs"][str(CAP_BASE)], ci["legs"][str(CAP_BASE)], design=design,
                              ladder={"main": cm["ladder"]["promoted"], "it": ci["ladder"]["promoted"]})
            # the pass-1 attestation labels the PAIR (scoped to pass 1); the merged reading carries its own
            merged_att = attest_merged_reading(cm, ci)
            pair.update(status="filed", attestation=att, label=att["label"], label_text=att["label_text"], label_scope=att["scope"],
                        merged_attestation=merged_att, merged_label=merged_att["label"], merged_label_text=merged_att["label_text"])
            rec["pairs"].append(pair)
        filed = [p for p in rec["pairs"] if p["status"] == "filed"]
        if filed:
            rec["status"] = "filed"
        elif any(p["status"].startswith("not run") for p in rec["pairs"]):
            rec["status"] = "no floor (a leg unusable)"
        else:
            rec["status"] = "no floor (no identical-by-construction pair)"
        rec["ruling"] = NOISE_RULING
        out.append(rec)
    return out


def noise_floor_pair(report_or_block: Mapping[str, Any] | Sequence[Mapping[str, Any]], model_key: str, cell: str = IC.CELL_NONE
                     ) -> Mapping[str, Any] | None:
    """The FILED pair of ``cell`` on ``model_key`` in a report's ``noise_floor`` block (or the block itself), else None."""
    block = report_or_block["noise_floor"] if isinstance(report_or_block, Mapping) else report_or_block
    for n in block:
        if n["model_key"] == model_key:
            return next((p for p in n.get("pairs") or [] if p["cell"] == cell and p["status"] == "filed"), None)
    return None


def truncation_costs(cells: Mapping[str, Mapping[str, Any]], *, ctx: Context) -> list[dict[str, Any]]:
    """``merged - r2048`` per PROMOTED cell, per metric — the truncation cost the ladder bought back."""
    out = []
    for key, c in cells.items():
        if not c.get("readings") or not c["ladder"]["promoted"]:
            continue
        rec: dict[str, Any] = {"cell_key": key, "model_key": c["model_key"], "view": c["view"], "cell": c["cell"], "promoted": c["ladder"]["promoted"], "metrics": {}}
        for metric in METRICS:
            legs = {READING_MERGED: _triples(c["readings"][READING_MERGED], metric), READING_PASS1: _triples(c["readings"][READING_PASS1], metric)}
            est = _boot(legs, {CONTRAST_TRUNCATION: {READING_MERGED: 1, READING_PASS1: -1}}, ctx)
            rec["metrics"][metric] = _stat(est, CONTRAST_TRUNCATION)
        out.append(rec)
    return out


# --------------------------------------------------------------------------- the gates
_gate = T._gate


def _fold(per: Mapping[str, Any], failures: list[str], pending: list[str] = ()) -> dict[str, Any]:
    status = GATE_FAIL if failures else (GATE_PENDING if pending else GATE_PASS)
    return _gate(status, "; ".join(failures) or ("; ".join(pending) or None), legs=per, failures=list(failures), pending=list(pending))


def _pending_without_logs(gate: dict[str, Any], *, read: bool, detail: str) -> dict[str, Any]:
    """A log-derived gate (:data:`LOG_DERIVED_GATES`) whose logs were NOT read is PENDING, never PASSED; a FAILED
    verdict from the checks that did run stands. The per-leg map is kept; the pending reason is named."""
    if read or gate.get("status") == GATE_FAIL:
        return gate
    gate["status"] = GATE_PENDING
    gate["pending"] = [detail]
    gate["detail"] = detail
    return gate


def _all_legs(cells: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {leg["key"]: leg for c in cells.values() for leg in c.get("legs", {}).values()}


def _usable_legs(cells: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {k: v for k, v in _all_legs(cells).items() if usable(v)}


def gate_completeness(rows: Sequence[Mapping[str, Any]], cells: Mapping[str, Any], refusals: Mapping[str, str]) -> dict[str, Any]:
    """I0: every LAUNCH row's pass-1 leg usable (or a NAMED expected refusal); every promoted cell's rung legs usable; held
    rows counted ``held`` — never ``missing``."""
    per: dict[str, str] = {}
    failures: list[str] = []
    counts = Counter()
    for r in rows:
        key = V.cell_key(r["model_key"], r["view"], r["cell"])
        c = cells.get(key)
        if r["status"] == IC.STATUS_HELD:
            per[key] = LEG_STATUS_HELD
            counts[LEG_STATUS_HELD] += 1
            continue
        if r["status"] == IC.STATUS_DEFERRED:
            per[key] = LEG_STATUS_DEFERRED
            counts[LEG_STATUS_DEFERRED] += 1
            continue
        base = (c or {}).get("legs", {}).get(str(CAP_BASE))
        leg_k = V.leg_key(r["model_key"], r["view"], r["cell"], CAP_BASE)
        if base is not None and usable(base):
            per[leg_k] = LEG_STATUS_USABLE
            counts[LEG_STATUS_USABLE] += 1
        elif leg_k in refusals:
            per[leg_k] = f"{LEG_STATUS_REFUSED}: {refusals[leg_k]}"
            counts[LEG_STATUS_REFUSED] += 1
        else:
            per[leg_k] = f"{LEG_STATUS_MISSING}: {(base or {}).get('refused') or 'no store'}"
            counts[LEG_STATUS_MISSING] += 1
            failures.append(f"{leg_k}: {(base or {}).get('refused') or 'no store'}")
        for cap in (c or {}).get("ladder", {}).get("promoted", []):
            rk = V.leg_key(r["model_key"], r["view"], r["cell"], cap)
            rung = c["legs"].get(str(cap))
            if rung is not None and usable(rung):
                per[rk] = LEG_STATUS_USABLE
            else:
                per[rk] = f"{LEG_STATUS_MISSING}: {(rung or {}).get('refused') or 'no rung store'}"
                failures.append(f"{rk}: promoted at {cap} but {(rung or {}).get('refused') or 'no rung store'}")
        if (c or {}).get("ladder", {}).get("problem"):
            failures.append(f"{key}: {c['ladder']['problem']}")
    return _gate(GATE_FAIL if failures else GATE_PASS, "; ".join(failures) or None, legs=per, counts=dict(counts), failures=failures)


def _per_leg(cells: Mapping[str, Any], pick: Callable[[Mapping[str, Any]], list[str]]) -> dict[str, Any]:
    per, failures = {}, []
    for k, leg in _usable_legs(cells).items():
        f = pick(leg)
        per[k] = GATE_PASS if not f else GATE_FAIL
        failures.extend(f"{k}: {x}" for x in f[:3])
    return _fold(per, failures)


def gate_ledger(cells: Mapping[str, Any]) -> dict[str, Any]:
    g = _per_leg(cells, lambda leg: list(leg["ledger"]["failures"]))
    g["reconciliation"] = {k: {"ledger_at_cap": leg["ledger"].get("n_length"), "recorded_at_cap": leg["ledger"].get("n_at_cap_recorded"),
                               "summary_n_at_cap": leg["ledger"].get("n_at_cap_summary"), "n_forwarded": leg["ledger"].get("n_forwarded"),
                               "n_draws": leg["ledger"].get("n_draws"), "client_timeout_s": leg["ledger"].get("client_timeout_s")}
                          for k, leg in _usable_legs(cells).items()}
    return g


def gate_identity(cells: Mapping[str, Any]) -> dict[str, Any]:
    def pick(leg):
        f = []
        if leg["identity"]:
            f.append(leg["identity"])
        if leg["stamps"]:
            f.append(leg["stamps"])
        f.extend(leg["rows_check"]["failures"][:2])
        f.extend(leg["request_evidence"]["failures"][:2])
        if leg.get("recorder_problem"):
            f.append(leg["recorder_problem"])
        return f
    g = _per_leg(cells, pick)
    g["recorder"] = recorder_summary(cells)
    return g


def recorder_summary(cells: Mapping[str, Any]) -> dict[str, Any]:
    """The resolved-sampler recorder's verdict per usable leg and the tally (the brief: AGREES everywhere — stated)."""
    per = {k: ((leg.get("recorder") or {}).get("verdict") if leg.get("recorder") is not None else None) for k, leg in _usable_legs(cells).items()}
    tally = Counter("not recorded" if v is None else v for v in per.values())
    return {"per_leg": per, "tally": dict(sorted(tally.items())), "n_legs": len(per),
            "all_agree": bool(per) and all(v == V.RECORDER_AGREES for v in per.values())}


def gate_certification(cells: Mapping[str, Any]) -> dict[str, Any]:
    g = _per_leg(cells, lambda leg: list(leg["certification"]["failures"]))
    g["served"] = {k: {"served_seqs": leg["certification"].get("served_seqs"), "kv_shrink": leg["certification"].get("kv_shrink"),
                       "max_model_len": leg["certification"].get("max_model_len"), "geometry_eval": leg["certification"].get("geometry_eval")}
                   for k, leg in _usable_legs(cells).items()}
    return g


def gate_scoring_path(cells: Mapping[str, Any]) -> dict[str, Any]:
    def pick(leg):
        f = list(leg["scoring_path"]["failures"])
        if leg["eval_check"].get("passed") is False:
            f.extend(leg["eval_check"]["failures"][:3])
        return f
    g = _per_leg(cells, pick)
    g["eval_logs_read"] = all(leg["eval_check"].get("passed") is not None for leg in _usable_legs(cells).values())
    # with the logs unread the cross-check did not run: PENDING, never PASSED — a per-leg failure still FAILS
    return _pending_without_logs(g, read=g["eval_logs_read"], detail=EVAL_LOGS_PENDING_I4)


def gate_at_cap(cells: Mapping[str, Any]) -> dict[str, Any]:
    """I5: the ladder record's internal identity (``ladder_record_problem``) AND, with the .eval logs read, the loop rule
    RE-DERIVED from every at-cap draw's retained text compared exactly with the record's draw lists, every text hash verified
    (``at_cap_text_check``). ``texts_reclassified`` says whether the second half ran on every usable leg."""
    def pick(leg):
        f = [leg["ladder_problem"]] if leg["ladder_problem"] else []
        if (leg.get("at_cap_text") or {}).get("passed") is False:
            f.extend(leg["at_cap_text"]["failures"][:3])
        return f
    g = _per_leg(cells, pick)
    legs = _usable_legs(cells)
    g["texts_reclassified"] = bool(legs) and all((leg.get("at_cap_text") or {}).get("passed") is not None for leg in legs.values())
    g["reclassification"] = {k: {f: (leg.get("at_cap_text") or {}).get(f) for f in ("n_at_cap", "n_loop_derived", "n_eligible_derived", "n_loop_recorded",
                                                                                      "n_eligible_recorded", "n_text_sha256_verified", "n_text_sha256_mismatch")}
                             for k, leg in legs.items()}
    g["n_at_cap_reclassified"] = sum(int((leg.get("at_cap_text") or {}).get("n_at_cap") or 0) for leg in legs.values()
                                     if (leg.get("at_cap_text") or {}).get("passed") is not None)
    # the record's internal identity alone is not I5: without the retained texts the gate is PENDING, never PASSED
    return _pending_without_logs(g, read=g["texts_reclassified"], detail=EVAL_LOGS_PENDING_I5)


def gate_ladder(cells: Mapping[str, Any]) -> dict[str, Any]:
    """I6: promoted iff the share exceeded 1 % (THE decision's exit — the planner's block over the merged reading, C1; at
    pass 1 the reading is the store); every rung's decision is over the merged reading of the chain up to it and agrees with
    this analysis's own merge (``decision_problems``); rung legs cover exactly the subset (the leg's population check); the
    merged reading has 1,082 draws; the final decision is terminal and, at 8,192, a ceiling / below-threshold / clean exit
    with the residual reported; the trigger provenance filed."""
    per, failures = {}, []
    for k, c in cells.items():
        if not c.get("readings"):
            continue
        f = []
        base = c["legs"][str(CAP_BASE)]
        share = base["shares"]["share_at_cap"]
        exit0 = base["shares"]["exit"]
        promoted = bool(c["ladder"]["promoted"])
        if (share is not None and share > IC.IFEVAL_TRIGGER_THRESHOLD) != (exit0 not in IR.NOT_TRIGGERED_EXITS):
            f.append(f"share {share} vs exit {exit0}")
        if promoted and exit0 != CL.ESCALATE:
            f.append(f"promoted though pass 1 exited {exit0}")
        if len(c["readings"][READING_MERGED]) != N_DRAWS:
            f.append(f"merged reading has {len(c['readings'][READING_MERGED])} draws, not {N_DRAWS}")
        for cap in c["ladder"]["promoted"]:
            rung = c["legs"].get(str(cap))
            if rung is None or not usable(rung):
                continue
            if rung["population"]:
                f.append(rung["population"])
            rdec = (rung["shares"].get("decision") or {})
            if rdec.get("reading") != READING_MERGED or rdec.get("chain") != [x for x in CAPS if x <= cap]:
                f.append(f"the decision at {cap} is over {rdec.get('reading')!r} of {rdec.get('chain')}, not the merged chain up to {cap}")
        f.extend(c["ladder"].get("decision_problems") or [])
        if base["shares"].get("trigger_provenance") != IC.IFEVAL_TRIGGER_PROVENANCE:
            f.append("no trigger provenance on the record")
        res = c["ladder"]["residual"] or {}
        if res.get("final_exit") == CL.ESCALATE:
            f.append(f"the chain ends at {res.get('final_cap')} on an escalating decision with no rung leg")
        if res.get("final_cap") == CEILING and res.get("final_exit") not in (CL.EXIT_CEILING, CL.EXIT_BELOW_THRESHOLD, CL.EXIT_CLEAN):
            f.append(f"at the ceiling with exit {res.get('final_exit')!r}")
        fr = c.get("final_readout") or {}
        if fr.get("n_unclassified"):
            f.append(f"{fr['n_unclassified']} residual at-cap draw(s) of the merged reading no ladder record classifies (loop / truncated)")
        if fr and fr.get("n_at_cap") != res.get("n_at_cap"):
            f.append(f"the final readout counts {fr.get('n_at_cap')} at cap, the chain's residual {res.get('n_at_cap')}")
        per[k] = GATE_PASS if not f else GATE_FAIL
        failures.extend(f"{k}: {x}" for x in f)
    return _fold(per, failures)


def gate_population(cells: Mapping[str, Any], ctx: Context) -> dict[str, Any]:
    per, failures, pending = {}, [], []
    trees = set()
    for k, c in cells.items():
        for leg in c.get("legs", {}).values():
            if not usable(leg):
                continue
            if leg["population"]:
                per[leg["key"]] = GATE_FAIL
                failures.append(f"{leg['key']}: {leg['population']}")
            else:
                per[leg["key"]] = GATE_PASS
            trees.add(leg.get("manifest_producer_tree_sha"))
        if c.get("readings") and len({d["item_id"] for d in c["readings"][READING_MERGED]}) != N:
            failures.append(f"{k}: the merged reading spans {len({d['item_id'] for d in c['readings'][READING_MERGED]})} items, not {N}")
    if ctx.canonical_ids is None:
        pending.append("no canonical ids given (--ids): the population is not held to the 541 canonical ids")
    if ctx.store_pins is None and len(trees) > 1:
        failures.append(f"{len(trees)} producer trees across the legs — one tree per wave (no store-pins doc admits them per row)")
    g = _fold(per, failures, pending)
    g["producer_trees"] = sorted(t for t in trees if t)
    g["admission"] = ("per-row pins (--store-pins): every usable leg's producer tree is its admitted pin" if ctx.store_pins is not None
                      else "one store pin (--sha prefix) for every leg")
    g["trees_by_cap"] = {cap: sorted({leg.get("manifest_producer_tree_sha") for c in cells.values() for k, leg in c.get("legs", {}).items()
                                      if k == cap and usable(leg) and leg.get("manifest_producer_tree_sha")}) for cap in sorted({k for c in cells.values() for k in c.get("legs", {})}, key=int)}
    return g


def gate_two_reading(contrasts: Sequence[Mapping[str, Any]], resolution: Mapping[str, Any]) -> dict[str, Any]:
    failures = [f"{c['cell_key']}: missing readings {sorted(set(READINGS) - set(c['readings']))}" for c in contrasts
                if c.get("status") == "filed" and set(c["readings"]) != set(READINGS)]
    failures.extend(resolution.get("unresolved") or [])
    return _gate(GATE_FAIL if failures else GATE_PASS, "; ".join(failures) or None, failures=failures, contrasts=list(IC.IFEVAL_CONTRASTS))


def gate_report_match(match: Mapping[str, Any] | None) -> dict[str, Any]:
    if match is None:
        return _gate(GATE_PENDING, "two-venv report match not requested in this pass (--require-report-match)", waivable=True)
    if match.get("equal"):
        return _gate(GATE_PASS, f"{match.get('shared_numeric')} shared numeric leaves equal", **{k: match.get(k) for k in ("shared_numeric",)})
    return _gate(GATE_FAIL, f"differing {len(match.get('differing') or [])}, missing {len(match.get('missing') or [])}, added {len(match.get('added') or [])}",
                 **{k: match.get(k) for k in ("differing", "missing", "added", "type_changed")})


def _gate_entry(gates: Mapping[str, Any], gate: str) -> Mapping[str, Any]:
    """The gate's record as the file carries it, or ``{}`` when the entry is absent or not a record (so every status read below
    is ``None`` for it, and :func:`gates_malformed` — not a status rule — is what names it)."""
    entry = gates.get(gate) if isinstance(gates, Mapping) else None
    return entry if isinstance(entry, Mapping) else {}


def gates_malformed(gates: Mapping[str, Any]) -> list[str]:
    """Every :data:`GATE_NAMES` entry that is ABSENT from the ``gates`` block, is not a record, or carries no status in
    :data:`GATE_STATUSES`. Read FIRST at certification, before any waiver rule: a gate the file does not carry
    with a recognised status has no verdict to waive, and no flag waives its absence."""
    return [g for g in GATE_NAMES if _gate_entry(gates, g).get("status") not in GATE_STATUSES]


def gates_failed(gates: Mapping[str, Any]) -> list[str]:
    return [g for g in GATE_NAMES if _gate_entry(gates, g).get("status") == GATE_FAIL]


def gates_pending_unwaivable(gates: Mapping[str, Any]) -> list[str]:
    return [g for g in GATE_NAMES if _gate_entry(gates, g).get("status") == GATE_PENDING and g not in WAIVABLE_PENDING_GATES]


def refresh_gate_aggregates(gates: dict[str, Any]) -> dict[str, Any]:
    """The tallies of what the file's gate records SAY (a FAILED / PENDING / PASSED count) — a description, never a
    certification: :func:`verdict_of` re-reads the records, and a gate these tallies cannot see (absent, unrecognised) is
    a refusal there (:func:`gates_malformed`), not a lower ``n_passed``."""
    gates["failed"] = gates_failed(gates)
    gates["pending"] = [g for g in GATE_NAMES if _gate_entry(gates, g).get("status") == GATE_PENDING]
    gates["pending_unwaivable"] = gates_pending_unwaivable(gates)
    gates["n_passed"] = sum(1 for g in GATE_NAMES if _gate_entry(gates, g).get("status") == GATE_PASS)
    return gates


# --------------------------------------------------------------------------- the report
def load_ids(source: Path | str | Sequence[str] | None) -> list[str] | None:
    if source is None:
        return None
    ids = json.loads(Path(source).read_text(encoding="utf-8")) if isinstance(source, (str, Path)) else list(source)
    ids = [str(i) for i in ids]
    if len(ids) != N or len(set(ids)) != N:
        raise AnalysisError(f"the canonical ids must be {N} distinct ids, got {len(ids)} ({len(set(ids))} distinct)")
    return ids


def load_expected_refusals(source: Path | str | Mapping[str, str] | None) -> dict[str, str]:
    if source is None:
        return {}
    doc = json.loads(Path(source).read_text(encoding="utf-8")) if isinstance(source, (str, Path)) else dict(source)
    return {str(k): str(v) for k, v in doc.items()}


#: the two bases a leg's actual GPU-hours can rest on, named on every leg record
COST_BASIS_LEDGER_SPAN = "ledger_generation_span"
COST_BASIS_FALLBACK_WALL = "finalisation_wall_clock (FALLBACK: the ledger spans nothing for the accepted attempt)"


def leg_cost(leg: Mapping[str, Any]) -> dict[str, Any]:
    """One usable leg's cost record. ``gpu_h_actual`` = the ACCEPTED generation attempt's ledger span
    (``leg["ledger"]["generation_span"]["elapsed_s"]`` — the wall its forwarded task rows spanned, C5's reading) + the
    Modal standup adder; its ``basis`` names that. The summary's ``wall_clock_s`` is the FINALISATION's wall — a
    reused-log finalisation of a 600 s generation reads ~1 s there, so it understated a leg by ~3x — and is used only
    when the ledger spans nothing, LABELLED as the fallback (``fallback`` True, ``basis`` naming it). Distinct from
    ``gpu_h_billed_all_attempts``: every recorded attempt's own span + one standup adder each (an abandoned attempt paid
    for its standup and its rows), ``None`` when the ledger check recorded no per-attempt spans."""
    ledger = leg.get("ledger") or {}
    span = ledger.get("generation_span") or {}
    elapsed = span.get("elapsed_s")
    wall = float((leg.get("summary") or {}).get("wall_clock_s") or 0.0)
    if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool) and float(elapsed) > 0:
        basis, fallback, seconds = COST_BASIS_LEDGER_SPAN, False, float(elapsed)
    else:
        basis, fallback, seconds = COST_BASIS_FALLBACK_WALL, True, wall
    by_attempt = ledger.get("generation_span_by_attempt")
    billed = None
    if isinstance(by_attempt, Mapping) and by_attempt:
        spans = [float((v or {}).get("elapsed_s") or 0.0) for v in by_attempt.values()]
        billed = round(sum((s + STANDUP_ADDER_S) / 3600.0 for s in spans), 4)
    central = CENTRAL_GPU_H_PER_CELL.get(leg["model_key"])
    if central is not None and IC.ifeval_prompt_class_of_cell(leg["cell"]) == C.IFEVAL_PROMPT_CLASS_LONG_SI:
        central = round(central + CENTRAL_GPU_H_LONGSI_ADDER, 3)
    return {"gpu_h_actual": round((seconds + STANDUP_ADDER_S) / 3600.0, 4), "basis": basis, "fallback": fallback,
            "generation_elapsed_s": (float(elapsed) if not fallback else None), "finalisation_wall_clock_s": wall,
            "gpu_h_billed_all_attempts": billed, "n_attempts_billed": (len(by_attempt) if isinstance(by_attempt, Mapping) else None),
            "accepted_attempt": ledger.get("accepted_attempt"), "gpu_h_central_estimate": central,
            "decode_tok_s_aggregate": ledger.get("decode_tok_s_aggregate"), "decode_tok_s_per_request": ledger.get("decode_tok_s_per_request"),
            "generation_span": ledger.get("generation_span"), "generation_span_by_attempt": by_attempt,
            "completion_tokens": ledger.get("completion_tokens")}


def cost_block(cells: Mapping[str, Any]) -> dict[str, Any]:
    """Per leg the ACTUAL GPU-hours (:func:`leg_cost`: the accepted attempt's ledger generation span + the standup adder;
    the finalisation wall only as a LABELLED fallback), the BILLED GPU-hours over every recorded attempt, and the CENTRAL
    per-cell estimate — an estimate, never a bound."""
    per = {k: leg_cost(leg) for k, leg in _usable_legs(cells).items()}
    total = round(sum(p["gpu_h_actual"] for p in per.values()), 3)
    billed = [p["gpu_h_billed_all_attempts"] for p in per.values()]
    return {"legs": per, "total_gpu_h_actual": total,
            "total_gpu_h_billed_all_attempts": (round(sum(billed), 3) if billed and all(b is not None for b in billed) else None),
            "n_legs_on_fallback_basis": sum(1 for p in per.values() if p["fallback"]),
            "central_estimate_total_gpu_h": CENTRAL_ESTIMATE_TOTAL_GPU_H, "standup_adder_s": STANDUP_ADDER_S,
            "basis": COST_BASIS_LEDGER_SPAN,
            "note": ("actual = the ACCEPTED generation attempt's ledger span + the Modal standup adder per leg (the summary's "
                     "finalisation wall only as a labelled fallback); billed = every recorded attempt's span + one adder each; "
                     "the estimate is labelled CENTRAL, never a bound")}


def load_store_pins(source: Path | str | Mapping[str, str] | None) -> dict[str, str] | None:
    """The per-leg admitted producer trees (``{"<model>/<view>/<cell>@<cap>": "<tree sha>"}``); a JSON file or the mapping.
    Every value must be hex of >= 7 characters (a tree sha or its prefix); anything else refuses by name."""
    if source is None:
        return None
    doc = json.loads(Path(source).read_text(encoding="utf-8")) if isinstance(source, (str, Path)) else dict(source)
    doc = {k: v for k, v in doc.items() if not str(k).startswith("_")}
    out: dict[str, str] = {}
    for k, v in doc.items():
        v = str(v)
        if len(v) < 7 or any(ch not in "0123456789abcdef" for ch in v.lower()):
            raise AnalysisError(f"store pins: {k}: {v!r} is not a tree sha (>= 7 hex)")
        if "@" not in str(k):
            raise AnalysisError(f"store pins: {k!r} is not a leg key <model>/<view>/<cell>@<cap>")
        out[str(k)] = v.lower()
    return out


def load_certifications(source: Path | str | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if source is None:
        return None
    doc = json.loads(Path(source).read_text(encoding="utf-8")) if isinstance(source, (str, Path)) else dict(source)
    return {k: v for k, v in doc.items() if not str(k).startswith("_")}


# --------------------------------------------------------------------------- instrument facts AS SERVED, from the admitted legs
def thinking_pins_by_family(cells: Mapping[str, Any]) -> dict[str, Any]:
    """Per model family: the thinking-OFF pin AS SERVED read off every usable leg's certification (``template_kwargs`` from the
    driver's serve-header evidence, ``thinking_switch`` / ``header_moves`` from the endpoint's probe) — the r1 record's §3.1
    table typed "none composed (the template has no switch)" for gemma while every gemma leg records ``switch true`` and
    ``{"enable_thinking": false}`` composed. A family whose legs disagree is filed ``consistent: false`` with every variant."""
    out: dict[str, Any] = {}
    for leg in _usable_legs(cells).values():
        cert = leg.get("certification") or {}
        fam = leg.get("family")
        rec = out.setdefault(fam, {"n_legs": 0, "models": [], "variants": [], "template_kwargs": None, "switch": None, "header_moves": None, "consistent": True})
        rec["n_legs"] += 1
        if leg["model_key"] not in rec["models"]:
            rec["models"].append(leg["model_key"])
        variant = {"template_kwargs": cert.get("template_kwargs"), "switch": cert.get("thinking_switch"), "header_moves": cert.get("header_moves")}
        if variant not in rec["variants"]:
            rec["variants"].append(variant)
    for fam, rec in out.items():
        rec["consistent"] = len(rec["variants"]) == 1
        if rec["consistent"]:
            rec.update(rec["variants"][0])
        rec["as_served"] = (("no switch: nothing composed (template_kwargs {})" if rec["switch"] is False else
                             f"switch on the template: {json.dumps(rec['template_kwargs'])} composed on every request"
                             + (" (the header moves with the switch)" if rec["header_moves"] else " (the header does not move)" if rec["header_moves"] is False else ""))
                            if rec["consistent"] else f"INCONSISTENT across the family's legs: {rec['variants']}")
    return out


def kv_shrink_legs(cells: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every usable leg the allocator SHRANK (a holding ``kv_shrink`` record): key, entry, configured -> served S — the r1
    prose said "all five Nemotron 4,096 legs shrink 171 -> 168" while the long-SI leg served 74 with no shrink."""
    out = []
    for k, leg in _usable_legs(cells).items():
        cert = leg.get("certification") or {}
        if cert.get("kv_shrink"):
            out.append({"key": k, "geometry_eval": cert.get("geometry_eval"), "configured_seqs": cert.get("configured_seqs"), "served_seqs": cert.get("served_seqs"),
                        "max_model_len": cert.get("max_model_len")})
    return out


def label_note(sha: str, trees_by_cap: Mapping[str, Any] | None, ctx: Context) -> str:
    trees = sorted({t for ts in (trees_by_cap or {}).values() for t in ts})
    if ctx.store_pins is None:
        return f"the record label `{sha}` is the ONE store pin every admitted leg's producer tree carries (legacy one-tree admission)"
    return (f"the record label `{sha}` names the run (the last rung's producer tree); it is NOT the producer tree of every store — "
            f"{len(trees)} producer trees were admitted per row (A.1: {', '.join(t[:8] for t in trees)})")


def analyze(gen_root: Path | str | Sequence[Path | str], sha: str, *, matrix: Path | str | Sequence[Any], canonical_ids: Path | str | Sequence[str] | None,
            pins: Mapping[str, Any] | Path | str | None = None, expected_refusals: Path | str | Mapping[str, str] | None = None,
            n_resamples: int = pr.DEFAULT_BOOTSTRAP, seed: int = pr.DEFAULT_BOOTSTRAP_SEED, ci: float = pr.DEFAULT_CI,
            allow_pending_gates: bool = False, read_eval_logs: bool = True, analysis_sha: str | None = None,
            exclude_families: Sequence[str] = (), store_pins: Path | str | Mapping[str, str] | None = None,
            nemotron_certifications: Path | str | Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The report over one or several wave roots. ``sha`` is the RECORD LABEL (the W&B run name, the sidecars) — with a
    ``store_pins`` doc every leg is admitted by ITS OWN pin (five trees across the battery's row classes) and
    ``sha`` names the last rung's tree; without one it is also the single admission pin (the legacy one-tree wave)."""
    gen_roots = _gen_roots(gen_root)
    mat = V.load_matrix(matrix)
    rows = mat["rows"]
    ctx = Context(canonical_ids=load_ids(canonical_ids), pins=(json.loads(Path(pins).read_text()) if isinstance(pins, (str, Path)) else pins),
                  expected_refusals=load_expected_refusals(expected_refusals), n_resamples=n_resamples, seed=seed, ci=ci,
                  read_eval_logs=read_eval_logs, exclude_families=tuple(exclude_families), store_pins=load_store_pins(store_pins),
                  nemotron_certifications=load_certifications(nemotron_certifications))
    if ctx.pins:
        for r in rows:
            if r["fitted"] and (ctx.pins.get(r["model_key"]) or {}).get(r["staged_id"], {}).get("object_sha256") not in (None, r["object_sha256"]):
                raise AnalysisError(f"matrix row {r['index']}: object_sha256 != the pins file's")
    cells = {V.cell_key(r["model_key"], r["view"], r["cell"]): collect_cell(gen_roots, r, ctx=ctx) for r in rows}
    # the producer pin: without a per-row pins doc, every usable leg's tree starts with the ONE store pin
    if ctx.store_pins is None:
        for leg in _usable_legs(cells).values():
            if not str(leg.get("manifest_producer_tree_sha") or "").startswith(str(sha)):
                leg["status"] = LEG_STATUS_MISSING
                leg["refused"] = f"producer tree {str(leg.get('manifest_producer_tree_sha'))[:8]} is not the store pin {sha[:8]}"
    gen_root = gen_roots[0]
    for c in cells.values():
        if c.get("readings") and not usable(c["legs"][str(CAP_BASE)]):
            c["readings"] = {}
    views = list(dict.fromkeys((r["model_key"], r["view"]) for r in rows))
    view_est = {f"{m}/{v}": view_estimands(cells, model_key=m, view=v, ctx=ctx) for m, v in views}
    contrasts: list[dict[str, Any]] = []
    for r in rows:
        if r["status"] in (IC.STATUS_HELD, IC.STATUS_DEFERRED) or r["cell"] == IC.CELL_NONE:
            continue
        key = V.cell_key(r["model_key"], r["view"], r["cell"])
        ve = view_est[f"{r['model_key']}/{r['view']}"]
        rec: dict[str, Any] = {"name": CONTRAST_VS_NONE, "cell_key": key, "model_key": r["model_key"], "view": r["view"], "cell": r["cell"],
                               "route_kind": r["route_kind"], "family": r["family"], "readings": {}}
        if not cells[key].get("readings") or key in ve.get("not_run", []) or not ve.get("readings"):
            rec["status"] = "not run"
            rec["reason"] = ve.get("problem") or "the cell's pass-1 leg is unusable"
        else:
            rec["status"] = "filed"
            for reading in READINGS:
                rec["readings"][reading] = {m: ve["readings"][reading][m]["contrasts"][key] for m in METRICS}
            rec["verdict"] = verdict_word(ve, key)
        contrasts.append(rec)
    noise = noise_floor(cells, rows=rows, ctx=ctx)
    trunc = truncation_costs(cells, ctx=ctx)
    resolution = {"unresolved": [f"{c['cell_key']}: {c.get('reason')}" for c in contrasts if c["status"] != "filed"]}
    gates: dict[str, Any] = {
        "I0_completeness": gate_completeness(rows, cells, ctx.expected_refusals), "I1_ledger": gate_ledger(cells),
        "I2_identity": gate_identity(cells), "I3_certification": gate_certification(cells), "I4_scoring_path": gate_scoring_path(cells),
        "I5_at_cap": gate_at_cap(cells), "I6_ladder": gate_ladder(cells), "I7_population": gate_population(cells, ctx),
        "I8_two_reading": gate_two_reading(contrasts, resolution), "I9_report_match": gate_report_match(None)}
    refresh_gate_aggregates(gates)
    per_cell = {}
    # the cross-eval guard reads object sha + view + pin per row: every per-cell
    # record names the served object's sha (a stated null on bare / prompted), its bank view and the STORE PIN
    row_by_key = {V.cell_key(r["model_key"], r["view"], r["cell"]): r for r in rows}
    for k, c in cells.items():
        base_leg = (c.get("legs") or {}).get(str(CAP_BASE)) or {}
        # `pin` = the pass-1 store's producer tree (per-row admission, V24); `pins_by_cap` every leg's; `label` the record's
        guard = {"object_sha256": row_by_key[k].get("object_sha256"), "view": c["view"],
                 "pin": (base_leg.get("manifest_producer_tree_sha") if usable(base_leg) else None) or (sha if ctx.store_pins is None else None),
                 "pins_by_cap": c.get("pins_by_cap") or {}, "label": sha}
        if not c.get("readings"):
            per_cell[k] = {"status": c["status"], "held_reason": c.get("held_reason"), "deferred": c.get("deferred"),
                           "usable": False, "model_key": c["model_key"], "cell": c["cell"], "family": c["family"], "route_kind": c["route_kind"],
                           "slot": (HELD_LABEL if c["status"] == IC.STATUS_HELD else PENDING_LABEL if c["status"] == IC.STATUS_DEFERRED else "missing"),
                           "slot_reason": (c.get("held_reason") if c["status"] == IC.STATUS_HELD else (c.get("deferred") or {}).get("reason")
                                           if c["status"] == IC.STATUS_DEFERRED else (base_leg.get("refused") or "no usable pass-1 leg")),
                           "superseded_stores": c.get("superseded_stores") or [], **guard}
            continue
        ve = view_est[f"{c['model_key']}/{c['view']}"]
        base = c["legs"][str(CAP_BASE)]
        per_cell[k] = {
            "status": c["status"], "usable": True, "model_key": c["model_key"], "view": c["view"], "cell": c["cell"], "route_kind": c["route_kind"],
            **guard,
            "family": c["family"], "shares": c["shares"], "ladder": c["ladder"], "chain": c["chain"],
            "metrics": {reading: {m: (ve["readings"].get(reading, {}).get(m, {}).get("legs", {}).get(k)) for m in METRICS} for reading in READINGS},
            "decode_tok_s_aggregate": base["ledger"].get("decode_tok_s_aggregate"), "decode_tok_s_per_request": base["ledger"].get("decode_tok_s_per_request"),
            "completion_tokens": base["ledger"].get("completion_tokens"),
            "served_seqs": base["certification"].get("served_seqs"), "kv_shrink": base["certification"].get("kv_shrink"),
            "n_draws": base["n_draws"], "n_items": base["n_items"],
            "redraw_outcomes": {str(h["to_cap"]): h.get("redraw_outcomes") for h in c["chain"]},
            "final_readout": c.get("final_readout"), "recorder": c.get("recorder"), "gen_root": c.get("gen_root"),
            "superseded_stores": c.get("superseded_stores") or [],
            "served_by_cap": {cap: {f: leg["certification"].get(f) for f in ("geometry_eval", "max_model_len", "served_seqs", "configured_seqs", "kv_shrink")}
                              for cap, leg in c["legs"].items() if usable(leg)},
            "leg_cost_by_cap": {cap: leg_cost(leg)["gpu_h_actual"] for cap, leg in c["legs"].items() if usable(leg)},
            # the served object's FIT PROVENANCE from the pass-1 manifest: fit weights vs served weights, the
            # producer's transfer flag, the chat-template bypass — the condition word is derived from these, never from the view
            "fit_provenance": base.get("fit_provenance"),
        }
    legs_out = {k: {f: leg.get(f) for f in ("key", "model_key", "view", "cell", "cap", "out_root", "status", "refused", "n_draws", "n_items",
                                            "identity", "stamps", "population", "ladder_problem", "shares", "manifest_producer_tree_sha", "manifest_hash",
                                            "recorder", "recorder_problem", "result")}
                | {"ledger": {f: leg.get("ledger", {}).get(f) for f in ("passed", "failures", "n_forwarded", "n_length", "n_at_cap_recorded", "n_at_cap_summary",
                                                                      "decode_tok_s_aggregate", "decode_tok_s_per_request", "generation_span",
                                                                      "completion_tokens", "client_timeout_s", "window",
                                                                      "accepted_attempt", "rows_by_attempt", "n_task_rows_all_attempts")},
                   "request_evidence": {f: leg.get("request_evidence", {}).get(f) for f in ("passed", "failures", "n_task_records", "n_v3",
                                                                                           "n_presence_penalty_present", "presence_penalty_expected",
                                                                                           "accepted_attempt", "n_task_records_all_attempts", "records_by_attempt")},
                   "certification": {f: leg.get("certification", {}).get(f) for f in ("passed", "failures", "served_seqs", "configured_seqs", "kv_shrink", "max_model_len",
                                                                                     "geometry_eval", "window_source", "seqs_source", "manifest_geometry_source",
                                                                                     "template_kwargs", "thinking_switch", "header_moves", "served_template_hash", "arm",
                                                                                     "object_sha256", "nemotron_attestation", "nemotron_certification",
                                                                                     "default_chat_template_kwargs")},
                   "rows_check": leg.get("rows_check"), "scoring_path": leg.get("scoring_path"), "eval_check": leg.get("eval_check"),
                   "fit_provenance": leg.get("fit_provenance"), "at_cap_text": leg.get("at_cap_text")}
                for k, leg in _all_legs(cells).items()}
    words = Counter((c.get("verdict") or {}).get("word") for c in contrasts if c["status"] == "filed")
    report: dict[str, Any] = {
        "experiment": EXPERIMENT, "task": TASK, "sha": sha, "analysis_code_sha": analysis_sha, "gen_root": str(gen_root),
        "gen_roots": [str(g) for g in gen_roots], "store_pins": ctx.store_pins, "admission": ("per-row pins" if ctx.store_pins is not None else "one store pin"),
        "nemotron_certifications": ctx.nemotron_certifications,
        "superseded_stores": {k: c.get("superseded_stores") for k, c in cells.items() if c.get("superseded_stores")},
        "final_readouts": {k: c.get("final_readout") for k, c in cells.items() if c.get("final_readout")},
        "slots": {k: {"status": c["status"], "label": (HELD_LABEL if c["status"] == IC.STATUS_HELD else PENDING_LABEL),
                      "reason": (c.get("held_reason") if c["status"] == IC.STATUS_HELD else (c.get("deferred") or {}).get("reason")),
                      "model_key": c["model_key"], "view": c["view"], "cell": c["cell"]}
                  for k, c in cells.items() if c["status"] in (IC.STATUS_HELD, IC.STATUS_DEFERRED)},
        "estimand": ESTIMAND_TEXT, "bootstrap": {"B": n_resamples, "seed": seed, "level": ci}, "allow_pending_gates": bool(allow_pending_gates),
        "matrix": {"path": str(matrix) if isinstance(matrix, (str, Path)) else None,
                   "sha256": (hashlib.sha256(Path(matrix).read_bytes()).hexdigest() if isinstance(matrix, (str, Path)) else None), "counts": mat["counts"]},
        "design": {"n_launch_rows": mat["counts"]["launched"], "n_held_rows": mat["counts"]["held"],
                   "n_deferred_rows": mat["counts"].get("deferred", 0), "models": list(dict.fromkeys(r["model_key"] for r in rows)),
                   "views": [f"{m}/{v}" for m, v in views], "n_contrasts_designed": mat["counts"]["contrasts_designed"], "k": K, "n_items": N, "n_draws": N_DRAWS,
                   "caps": list(CAPS), "ceiling": CEILING, "trigger_threshold": IC.IFEVAL_TRIGGER_THRESHOLD, "trigger_rule": IC.IFEVAL_TRIGGER_RULE,
                   "trigger_provenance": IC.IFEVAL_TRIGGER_PROVENANCE, "loop_rule": LOOP_RULE_TEXT, "loop_rule_disclosure": LOOP_RULE_DISCLOSURE,
                   "seed_rule": IC.SEED_RULE_IFEVAL, "sampling_mode": IC.SAMPLING_VENDOR_PRESET,
                   "sampling_rows": {k: IC.sampling_row_dict(IC.vendor_preset_row(k)) for k in IC.IFEVAL_MODELS},
                   "thinking": C.THINKING_OFF, "thinking_arm_by_family": {f: IC.ifeval_effective_arm(f) for f in sorted({r["family"] for r in rows})},
                   "scoring": "at-cap draws scored exactly as IFEval scores them (no exclusion, no re-draw before scoring), THEN loop-classified",
                   "exclude_families": list(ctx.exclude_families),
                   # the instrument facts AS SERVED, generated from the admitted legs — never typed by family
                   "thinking_pin_as_served": thinking_pins_by_family(cells), "kv_shrink_legs": kv_shrink_legs(cells)},
        "held": [{"cell_key": k, "reason": c.get("held_reason")} for k, c in cells.items() if c["status"] == IC.STATUS_HELD],
        "deferred": [{"cell_key": k, "record": (c.get("deferred") or {}).get("record")} for k, c in cells.items()
                     if c["status"] == IC.STATUS_DEFERRED],
        "expected_refusals": dict(ctx.expected_refusals), "canonical_ids": {"n": len(ctx.canonical_ids) if ctx.canonical_ids else None},
        "legs": legs_out, "unusable_legs": {k: v["refused"] for k, v in legs_out.items() if v["status"] != LEG_STATUS_USABLE},
        "cells": per_cell, "view_estimands": {k: {"problem": v.get("problem"), "not_run": v.get("not_run"), "readings": v.get("readings")} for k, v in view_est.items()},
        "contrasts": contrasts, "noise_floor": noise, "truncation_cost": trunc,
        "contrast_summary": {"n_designed": len(contrasts), "n_filed": sum(1 for c in contrasts if c["status"] == "filed"), "words": dict(words)},
        "gates": gates, "cost": cost_block(cells),
        "provenance": {"producer_trees": gates["I7_population"].get("producer_trees"), "trees_by_cap": gates["I7_population"].get("trees_by_cap"),
                       "admission": gates["I7_population"].get("admission"), "store_pin": sha, "label": sha,
                       # the LABEL is not "every store's producer tree": with a pins doc it names the last rung's tree
                       # and the admitted trees are the per-row ones in `trees_by_cap`
                       "label_note": label_note(sha, gates["I7_population"].get("trees_by_cap"), ctx),
                       "pins_by_cell": {k: c.get("pins_by_cap") for k, c in cells.items() if c.get("pins_by_cap")},
                       "recorder": gates["I2_identity"].get("recorder"), "instrument": IR.IFEVAL_INSTRUMENT._asdict()},
    }
    report["verdict"] = verdict_of(report)
    # the in-memory report IS the file: a JSON round trip here so that a tuple (the instrument's cap ladder) reads as the
    # list the file carries, and --require-report-match compares like with like (a tuple is a non-numeric LEAF to the
    # leaf walker; the same value re-read from disk is three numeric leaves, which read as "missing" in one direction)
    return json.loads(json.dumps(report, default=str))


def gates_pending_log_derived(gates: Mapping[str, Any]) -> list[str]:
    """The log-derived gates (:data:`LOG_DERIVED_GATES`) left PENDING because the `.eval` logs were not read — the PENDING no
    flag waives."""
    return [g for g in LOG_DERIVED_GATES if _gate_entry(gates, g).get("status") == GATE_PENDING]


def gates_passed_without_evidence(gates: Mapping[str, Any]) -> list[str]:
    """The log-derived gates whose stored status is PASSED while the evidence flag that PASS rests on
    (:data:`LOG_DERIVED_EVIDENCE`) is not exactly True — False or missing. A refusal no flag waives: the status
    is a claim, the flag is what says the `.eval` logs were actually read on every leg. Distinct from
    :func:`gates_pending_log_derived` (the honest no-log state, PENDING): this is a PASSED that the file cannot support."""
    return [g for g, flag in LOG_DERIVED_EVIDENCE.items()
            if _gate_entry(gates, g).get("status") == GATE_PASS and _gate_entry(gates, g).get(flag) is not True]


def eval_logs_validated(gates: Mapping[str, Any]) -> bool:
    """True iff EVERY log-derived gate is present, PASSED, and its evidence flag is exactly True: the positive
    statement that the `.eval` logs were read on every leg. Absent, PENDING, FAILED, unrecognised, or PASSED without the flag
    all read False — the status alone certifies nothing, and a gate the file does not carry certifies less."""
    return all(_gate_entry(gates, g).get("status") == GATE_PASS and _gate_entry(gates, g).get(flag) is True
               for g, flag in LOG_DERIVED_EVIDENCE.items())


def _gates_of(report: Mapping[str, Any]) -> Mapping[str, Any]:
    """The report's ``gates`` block, or ``{}`` when the file has none / it is not a mapping — every gate then reads as
    malformed (absent) rather than as a KeyError, so the refusal names the gates instead of a traceback."""
    gates = report.get("gates") if isinstance(report, Mapping) else None
    return gates if isinstance(gates, Mapping) else {}


def verdict_of(report: Mapping[str, Any]) -> dict[str, Any]:
    gates = _gates_of(report)
    # FIRST, before any waiver rule: every gate present with a recognised status, or no verdict to waive
    malformed = gates_malformed(gates)
    failed = gates_failed(gates)
    pending = gates_pending_unwaivable(gates)
    log_pending = gates_pending_log_derived(gates)
    unevidenced = gates_passed_without_evidence(gates)
    validated = eval_logs_validated(gates)
    # `--allow-pending-gates` waives the other unwaivable PENDINGs (a population gate without the ids file); it never waives a
    # log-derived one, never a malformed gate, a PASSED without its evidence flag is never production, a comparison-only
    # pass never claims production, and production REQUIRES the positive `eval_logs_validated` — I4 / I5
    # present, PASSED, evidence True — rather than merely the absence of a refusal
    production = (not malformed and not failed and not log_pending and not unevidenced and validated
                  and (not pending or bool(report.get("allow_pending_gates"))) and not report.get("comparison_only"))
    return {"production": production, "gates_malformed": malformed, "gates_failed": failed, "gates_pending_unwaivable": pending,
            "gates_pending_log_derived": log_pending, "gates_passed_without_evidence": unevidenced,
            # validated = every log-derived gate present, PASSED AND its evidence flag True; the status alone certifies nothing
            "eval_logs_validated": validated,
            "comparison_only": bool(report.get("comparison_only")),
            "words": report["contrast_summary"]["words"], "n_filed": report["contrast_summary"]["n_filed"],
            "n_promoted_cells": sum(1 for c in report["cells"].values() if c.get("usable") and c["ladder"]["promoted"]),
            "n_usable_cells": sum(1 for c in report["cells"].values() if c.get("usable"))}


def _log_pending_refusals(report: Mapping[str, Any]) -> list[str]:
    gates = _gates_of(report)
    return [f"gate {g} PENDING (never waivable): {_gate_entry(gates, g).get('detail')}" for g in gates_pending_log_derived(gates)]


def _malformed_refusals(report: Mapping[str, Any]) -> list[str]:
    """One refusal per :data:`GATE_NAMES` entry the file does not carry with a recognised status: ABSENT from the
    ``gates`` block, not a record, or a ``status`` outside :data:`GATE_STATUSES` (``None`` when the key is missing). Never
    waivable; read before every other rule."""
    gates = _gates_of(report)
    out = []
    for g in gates_malformed(gates):
        entry = gates.get(g)
        if entry is None and g not in gates:
            out.append(f"gate {g} ABSENT from the report's gates (never waivable): every one of the {len(GATE_NAMES)} gates must be present "
                       f"with a status in {list(GATE_STATUSES)} before any waiver rule is read")
        elif not isinstance(entry, Mapping):
            out.append(f"gate {g} is not a gate record (never waivable): the file carries {type(entry).__name__} {str(entry)[:60]!r}, "
                       f"not a record with a status in {list(GATE_STATUSES)}")
        else:
            out.append(f"gate {g} carries no recognised status (never waivable): `status` is {entry.get('status')!r}, not one of "
                       f"{list(GATE_STATUSES)} — an unrecognised or missing status certifies nothing and no flag waives it")
    return out


def _unevidenced_refusals(report: Mapping[str, Any]) -> list[str]:
    """One refusal per log-derived gate stored PASSED without its evidence flag True: the flag's name and the value the
    file carries (``None`` when the key is missing altogether), never waivable."""
    gates = _gates_of(report)
    return [f"gate {g} PASSED without its evidence (never waivable): `{LOG_DERIVED_EVIDENCE[g]}` is "
            f"{_gate_entry(gates, g).get(LOG_DERIVED_EVIDENCE[g])!r}, not True — the status alone does not certify that the .eval logs were read"
            for g in gates_passed_without_evidence(gates)]


def production_refusals(report: Mapping[str, Any]) -> list[str]:
    gates = _gates_of(report)
    # the malformed gates FIRST: a gate the file does not carry with a recognised status is refused before any
    # FAILED / PENDING / waiver rule is read, and nothing below can waive it
    out = _malformed_refusals(report)
    out.extend(f"gate {g} FAILED: {_gate_entry(gates, g).get('detail')}" for g in gates_failed(gates))
    log_pending = set(gates_pending_log_derived(gates))
    if not report.get("allow_pending_gates"):
        out.extend(f"gate {g} PENDING: {_gate_entry(gates, g).get('detail')}" for g in gates_pending_unwaivable(gates) if g not in log_pending)
    out.extend(_log_pending_refusals(report))
    out.extend(_unevidenced_refusals(report))
    if report.get("comparison_only"):
        out.append(f"a {COMPARISON_ONLY_MODE} report: it claims no production (its purpose is the two-venv numeric-leaf match, I9)")
    if report.get("unusable_legs"):
        out.append(f"{len(report['unusable_legs'])} unusable leg(s): {sorted(report['unusable_legs'])[:3]}")
    not_run = [c["cell_key"] for c in report["contrasts"] if c["status"] != "filed"]
    if not_run:
        out.append(f"{len(not_run)} designed contrast(s) did not run: {not_run[:3]}")
    return out


def refusals_beyond_eval_logs(report: Mapping[str, Any]) -> list[str]:
    """:func:`production_refusals` without the two log-derived PENDINGs and the comparison-only line — a DIAGNOSTIC: what,
    besides reading the `.eval` logs, stands between this report and production. It certifies nothing (``production`` stays
    False until the logs are read); the comparison-only pass refuses on it BEFORE matching, and a venv without inspect_ai
    reads its gates through it. A PASSED without its evidence flag is NOT filtered here: it is not the honest
    no-log state but a claim the file cannot support, and it refuses a comparison-only pass too. Neither is a malformed
    gate: a report missing a gate is not the honest no-log state either."""
    log_lines = set(_log_pending_refusals(report))
    return [r for r in production_refusals(report) if r not in log_lines and not r.startswith(f"a {COMPARISON_ONLY_MODE} report")]


def non_log_gates_clear(report: Mapping[str, Any]) -> bool:
    """True iff nothing but the unread `.eval` logs (or the comparison-only mode) refuses production — every other gate passed,
    every leg usable, every designed contrast filed. A predicate for the comparison-only pass and for tests run without
    inspect_ai; never a production claim."""
    return not refusals_beyond_eval_logs(report)


# --------------------------------------------------------------------------- figures (dataviz: one system, fixed hue per family)
FAMILY_NONE, FAMILY_PROMPTED, FAMILY_CONST, FAMILY_CTL, FAMILY_SI12, FAMILY_DERIVED = ("none", "prompted", "constitution", "control", "safety_si12", "derived")
CAT: Mapping[str, str] = {**T.CAT, "magenta": "#e87ba4", "green": "#008300"}
FAMILY_COLOR: Mapping[str, str] = {FAMILY_NONE: CAT["blue"], FAMILY_PROMPTED: CAT["aqua"], FAMILY_CONST: CAT["orange"], FAMILY_CTL: CAT["yellow"],
                                   FAMILY_SI12: CAT["magenta"], FAMILY_DERIVED: CAT["green"]}
FAMILY_LABEL: Mapping[str, str] = {FAMILY_NONE: "none (bare)", FAMILY_PROMPTED: "prompted (safety_refusal)", FAMILY_CONST: "constitution object",
                                   FAMILY_CTL: "control object (constv2ctl2)", FAMILY_SI12: "safety si12 object (header-located)", FAMILY_DERIVED: "derived residual"}
INK, INK2, MUTED, BASELINE = M.INK, M.INK2, M.MUTED, M.BASELINE
AXES_MIN_FRACTION = M.AXES_MIN_FRACTION
check_title, axes_height_fraction, _title, _layout = M.check_title, M.axes_height_fraction, M._title, M._layout


def cell_family(cell: str) -> str:
    if cell == IC.CELL_NONE:
        return FAMILY_NONE
    if cell.startswith(IC.PROMPTED_PREFIX):
        return FAMILY_PROMPTED
    if "minus_ctl" in cell:
        return FAMILY_DERIVED
    if cell.startswith("safety_refusal__"):
        return FAMILY_SI12
    if cell.startswith("constv2ctl2"):
        return FAMILY_CTL
    return FAMILY_CONST


def _short(cell: str) -> str:
    return {"none": "none", "prompted__safety_refusal": "prompted", "prompted__si_constitution": "prompted (constitution)",
            "constv2-mix-full__psr_ce_all": "const PSR", "constv2-mix-full__lora_r4_ce": "const LoRA",
            "constv2ctl2-full__psr_ce_all": "ctl PSR", "constv2ctl2-full__lora_r4_ce": "ctl LoRA", "safety_refusal__psr_kl_all": "si12 PSR",
            "safety_refusal__lora_r4": "si12 LoRA", "constv2-mix-full__psr_ce_all_minus_ctl_distilled": "diff PSR (distilled)",
            "constv2-mix-full__lora_r4_ce_minus_ctl": "diff LoRA"}.get(cell, cell)


def _visible_cells(report: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    excl = set(report["design"].get("exclude_families") or [])
    return [(k, c) for k, c in report["cells"].items() if c.get("usable") and c["family"] not in excl]


def _slot_cells(report: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Every cell of the matrix a figure shows a SLOT for, in matrix order: the usable cells and the HELD / PENDING rows
    (never as measured — hatched, labelled with their word; the brief: shown as such in every table and figure)."""
    excl = set(report["design"].get("exclude_families") or [])
    return [(k, c) for k, c in report["cells"].items() if c.get("family") not in excl and (c.get("usable") or c.get("slot"))]


def _slot_word(c: Mapping[str, Any]) -> str:
    return HELD_LABEL if c.get("status") == IC.STATUS_HELD else "PENDING" if c.get("status") == IC.STATUS_DEFERRED else "MISSING"


def _title2(fig, title: str, reading: str) -> None:
    """The title rule as the figure builder inherits it: line 1 IS the title (<= :data:`M.TITLE_MAX_WORDS` words),
    line 2 the one-line reading (<= 90 chars); everything else goes to the caption. One bold text, so
    :func:`M.figure_title_problem` can hold every figure to it in :func:`make_figures`."""
    if len(title.split()) > M.TITLE_MAX_WORDS:
        raise AnalysisError(f"figure title {title!r} has {len(title.split())} words (> {M.TITLE_MAX_WORDS})")
    _title(fig, check_title(f"{title}\n{reading}"))


def _not_shown(report: Mapping[str, Any], shown_cell_keys: Sequence[str]) -> dict[str, Any]:
    """What a SUBSET figure leaves out: the HELD / PENDING slots (never measured) and the usable cells the figure's
    own selection rule excludes — counted by name so the exclusion is visible on the figure and in its caption."""
    excl = set(report["design"].get("exclude_families") or [])
    shown = set(shown_cell_keys)
    slots = [k for k, c in report["cells"].items() if c.get("slot") and c.get("family") not in excl]
    usable_out = [k for k, c in report["cells"].items() if c.get("usable") and c.get("family") not in excl and k not in shown]
    held = [k for k in slots if report["cells"][k].get("status") == IC.STATUS_HELD]
    pending = [k for k in slots if report["cells"][k].get("status") == IC.STATUS_DEFERRED]
    return {"held": held, "pending": pending, "usable_not_shown": usable_out}


def _footnote(fig, report: Mapping[str, Any], shown_cell_keys: Sequence[str], *, why: str) -> dict[str, Any]:
    """The exclusion footnote of a subset figure: the slots not measured and the usable cells its rule leaves out, named."""
    ns = _not_shown(report, shown_cell_keys)
    parts = []
    if ns["held"] or ns["pending"]:
        parts.append(f"{len(ns['held'])} HELD + {len(ns['pending'])} PENDING slot(s) not measured")
    if ns["usable_not_shown"]:
        parts.append(f"{len(ns['usable_not_shown'])} measured cell(s) not shown ({why})")
    if parts:
        fig.text(0.01, 0.005, "not shown: " + "; ".join(parts), fontsize=6.2, color=MUTED, ha="left", va="bottom", fontstyle="italic")
    return ns


def _slot_bar(ax, x: float, height: float, *, width: float = 0.7, label: str, vertical: bool = True) -> None:
    """A HELD / PENDING slot: a hatched, edge-only bar with its word written on it — never a measured value."""
    if vertical:
        ax.bar(x, height, width=width, facecolor="none", edgecolor=MUTED, hatch="////", lw=0.8, zorder=2)
        ax.text(x, height * 0.5, label, rotation=90, ha="center", va="center", fontsize=6.4, color=INK2, zorder=5)
    else:
        ax.barh(x, height, height=width, facecolor="none", edgecolor=MUTED, hatch="////", lw=0.8, zorder=2)
        ax.text(height * 0.5, x, label, ha="center", va="center", fontsize=6.4, color=INK2, zorder=5)


def _fig_contrasts_model(report: Mapping[str, Any], model_key: str) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mc, Patch = T._mpl()
    excl = set(report["design"].get("exclude_families") or [])
    cs = [c for c in report["contrasts"] if c["model_key"] == model_key and c["status"] == "filed" and c["family"] not in excl]
    slots = [(k, c) for k, c in report["cells"].items() if c.get("model_key") == model_key and c.get("slot") and c["family"] not in excl]
    metrics = ("prompt_strict", "inst_strict")
    n_rows = len(cs) + len(slots)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, max(3.2, 0.34 * n_rows + 1.6)), sharey=True)
    numbers: list[dict[str, Any]] = []
    # the grey band is the model's none(it) - none(main) pair ONLY when the attestation labels it a floor (a pair that
    # measures a difference is not a floor and is not drawn as one)
    floor = noise_floor_pair(report, model_key, IC.CELL_NONE)
    if floor is not None and floor.get("label") != LABEL_FLOOR:
        floor = None
    labels = [f"{c['view']}: {_short(c['cell'])}" for c in cs] + [f"{c['view']}: {_short(c['cell'])}" for _, c in slots]
    for ax, metric in zip(axes, metrics):
        T._recessive(ax)
        ax.axvline(0, color=BASELINE, lw=0.9, zorder=1)
        if floor:
            band = floor["readings"][READING_MERGED][metric]
            ax.axvspan(band["ci_low"], band["ci_high"], color=MUTED, alpha=0.18, lw=0, zorder=0)
            i_block = next(i for i, n in enumerate(report["noise_floor"]) if n["model_key"] == model_key)
            j_pair = next(j for j, p in enumerate(report["noise_floor"][i_block]["pairs"]) if p is floor)
            for f in ("ci_low", "ci_high"):
                T._num(numbers, f"{model_key}/floor_band/{metric}/{f}", band[f], f"noise_floor.{i_block}.pairs.{j_pair}.readings.{READING_MERGED}.{metric}.{f}")
        for y, c in enumerate(cs):
            color = FAMILY_COLOR[cell_family(c["cell"])]
            m, r = c["readings"][READING_MERGED][metric], c["readings"][READING_PASS1][metric]
            ax.plot([m["ci_low"], m["ci_high"]], [y, y], color=color, lw=1.6, zorder=3)
            ax.plot([m["point"]], [y], marker="o", color=color, ms=5.5, zorder=4)
            ax.plot([r["point"]], [y + 0.22], marker="o", mfc="none", mec=color, ms=4.5, zorder=4)
            ax.plot([r["ci_low"], r["ci_high"]], [y + 0.22, y + 0.22], color=color, lw=0.8, alpha=0.6, zorder=3)
            idx = report["contrasts"].index(c)
            # every plotted endpoint: the merged point + CI and the r2048 point + CI
            for reading, s in ((READING_MERGED, m), (READING_PASS1, r)):
                for f in ("point", "ci_low", "ci_high"):
                    T._num(numbers, f"{model_key}/{c['view']}/{c['cell']}/{metric}/{reading}/{f}", s[f], f"contrasts.{idx}.readings.{reading}.{metric}.{f}")
            if ax is axes[-1]:
                ax.text(1.02, y, (c.get("verdict") or {}).get("word") or "", transform=ax.get_yaxis_transform(), fontsize=7, color=INK2, va="center")
        for j, (k, c) in enumerate(slots):
            y = len(cs) + j
            ax.text(0.5, y, f"{_slot_word(c)} — not measured", transform=ax.get_yaxis_transform(), fontsize=7, color=MUTED, va="center", ha="center",
                    fontstyle="italic")
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(labels, fontsize=7.6)
        ax.set_xlabel(f"{metric.replace('_', ' ')}: cell - none (fraction)", fontsize=8)
        ax.set_ylim(n_rows - 0.4, -0.7)
    _title2(fig, "IFEval contrasts", f"{model_key}: cell - none (merged); grey = none(it) - none(main) floor; word = verdict")
    fig.legend(handles=[Patch(color=FAMILY_COLOR[f], label=FAMILY_LABEL[f]) for f in FAMILY_COLOR if any(cell_family(c["cell"]) == f for c in cs)]
               + [plt.Line2D([], [], marker="o", color=INK, ls="", label="merged reading"), plt.Line2D([], [], marker="o", mfc="none", mec=INK2, ls="", label="r2048 reading")],
               loc="lower center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.03))
    _layout(fig, left=0.24, right=0.84, wspace=0.12)
    return fig, numbers


def fig_metrics_by_cell(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """Per MODEL a panel: the arms of that model (view: cell) on x, grouped bars of the four IFEval metrics (merged reading)
    with item-paired 95 % CI whiskers; HELD / PENDING rows as hatched slots — the brief's bar chart with error bars per
    model x arm."""
    plt, _mc, Patch = T._mpl()
    excl = set(report["design"].get("exclude_families") or [])
    models = [m for m in report["design"]["models"] if C.FAMILY_BY_MODEL_KEY.get(m) not in excl]
    slots_all = _slot_cells(report)
    ncol = 2 if len(models) > 1 else 1
    nrow = max(1, (len(models) + ncol - 1) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.4 * ncol, 3.6 * nrow), squeeze=False)
    numbers: list[dict[str, Any]] = []
    width = 0.2
    shades = (0.95, 0.7, 0.5, 0.3)
    for ax, model in zip(axes.flat, models):
        T._recessive(ax)
        cells = [(k, c) for k, c in slots_all if c["model_key"] == model]
        for x, (k, c) in enumerate(cells):
            if not c.get("usable"):
                _slot_bar(ax, x, 1.0, label=_slot_word(c))
                continue
            color = FAMILY_COLOR[cell_family(c["cell"])]
            for j, metric in enumerate(METRICS):
                leg = (c["metrics"].get(READING_MERGED) or {}).get(metric)
                if not leg:
                    continue
                ax.bar(x + (j - 1.5) * width, leg["point"], width=width, color=color, alpha=shades[j], edgecolor=INK if j % 2 else None, lw=0.4, zorder=3)
                ax.errorbar(x + (j - 1.5) * width, leg["point"], yerr=[[leg["point"] - leg["ci_low"]], [leg["ci_high"] - leg["point"]]], color=INK2, lw=0.7,
                            capsize=1.5, zorder=4)
                for f in ("point", "ci_low", "ci_high"):
                    T._num(numbers, f"{k}/{metric}/merged/{f}", leg[f], f"cells.{k}.metrics.merged.{metric}.{f}")
        ax.set_xticks(range(len(cells)))
        ax.set_xticklabels([f"{c['view']}: {_short(c['cell'])}" for _, c in cells], rotation=30, ha="right", fontsize=6.2)
        ax.set_ylim(0, 1)
        ax.set_ylabel("IFEval metric (merged)", fontsize=7.5)
        ax.text(0.01, 0.985, model, transform=ax.transAxes, ha="left", va="top", fontsize=8.2, color=INK)
    for ax in list(axes.flat)[len(models):]:
        ax.axis("off")
    fig.legend(handles=[Patch(facecolor=INK2, alpha=shades[j], label=m.replace("_", " ")) for j, m in enumerate(METRICS)]
               + [Patch(facecolor="none", edgecolor=MUTED, hatch="////", label="HELD / PENDING slot (not measured)")],
               loc="lower center", ncol=5, fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.09))
    _title2(fig, "IFEval metrics", "merged reading per arm: prompt strict / loose, inst strict / loose; item-paired 95 % CIs")
    _layout(fig, left=0.06, right=0.99, hspace=0.6, wspace=0.18)
    return fig, numbers


def fig_shares(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mc, Patch = T._mpl()
    cells = _slot_cells(report)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.32 * len(cells) + 2), 4.0))
    T._recessive(ax)
    numbers: list[dict[str, Any]] = []
    top = 0.0
    for x, (k, c) in enumerate(cells):
        if not c.get("usable"):
            continue
        s = c["shares"]
        st, sl = float(s.get("share_truncated") or 0.0), float(s.get("share_loop") or 0.0)
        top = max(top, st + sl)
        ax.bar(x, st, color=CAT["orange"], width=0.7, zorder=3)
        ax.bar(x, sl, bottom=st, color=CAT["magenta"], width=0.7, zorder=3)
        if c["ladder"]["promoted"]:
            ax.text(x, st + sl + 0.002, "->" + "->".join(str(cap) for cap in c["ladder"]["promoted"]), ha="center", va="bottom", fontsize=5.2, color=INK2)
        T._num(numbers, f"{k}/share_at_cap", s.get("share_at_cap"), f"cells.{k}.shares.share_at_cap")
        T._num(numbers, f"{k}/share_truncated", s.get("share_truncated"), f"cells.{k}.shares.share_truncated")
        T._num(numbers, f"{k}/share_loop", s.get("share_loop"), f"cells.{k}.shares.share_loop")
    for x, (k, c) in enumerate(cells):
        if not c.get("usable"):
            _slot_bar(ax, x, max(top, 0.02), label=_slot_word(c))
    ax.axhline(IC.IFEVAL_TRIGGER_THRESHOLD, color=INK, lw=0.9, ls="--", zorder=4)
    ax.text(len(cells) - 0.5, IC.IFEVAL_TRIGGER_THRESHOLD, " 1 % trigger", va="bottom", ha="right", fontsize=7, color=INK)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"{c['model_key']}/{c['view']}\n{_short(c['cell'])}" for _, c in cells], rotation=90, fontsize=6.4)
    ax.set_ylabel("share of the 1,082 pass-1 draws at cap 2,048")
    fig.legend(handles=[Patch(color=CAT["orange"], label="truncated (at cap, not a loop; boundary included)"), Patch(color=CAT["magenta"], label="loop (whitespace 4-gram rule)"),
                        Patch(facecolor="none", edgecolor=MUTED, hatch="////", label="HELD / PENDING slot")],
               loc="upper right", fontsize=7, frameon=False, bbox_to_anchor=(0.99, 0.9))
    _title2(fig, "At-cap shares", "pass 1 at 2,048: truncated + loop per cell; dashed = the 1 % trigger; '->cap' = promoted")
    _layout(fig, left=0.07, right=0.99)
    return fig, numbers


def fig_exits(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    """The ladder EXITS: per cell the FINAL readout's residual at-cap share of the merged reading, split into loops and
    genuine truncation by the record of the leg that last drew each draw, with the exit rung and exit word written above;
    HELD / PENDING slots hatched."""
    plt, _mc, Patch = T._mpl()
    cells = _slot_cells(report)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.32 * len(cells) + 2), 4.2))
    T._recessive(ax)
    numbers: list[dict[str, Any]] = []
    exit_short = {CL.EXIT_BELOW_THRESHOLD: "below", CL.EXIT_CEILING: "CEILING", IR.EXIT_TRIGGERED_NO_ELIGIBLE: "no-eligible", CL.EXIT_CLEAN: "clean"}
    top = 0.0
    for x, (k, c) in enumerate(cells):
        fr = c.get("final_readout") if c.get("usable") else None
        if not fr:
            continue
        st, sl = float(fr.get("share_truncated") or 0.0), float(fr.get("share_loop") or 0.0)
        top = max(top, st + sl)
        ax.bar(x, sl, color=CAT["magenta"], width=0.7, zorder=3)
        ax.bar(x, st, bottom=sl, color=CAT["orange"], width=0.7, zorder=3)
        ax.text(x, st + sl + 0.004, f"{fr['exit_cap']}\n{exit_short.get(fr['exit'], fr['exit'])}", ha="center", va="bottom", fontsize=4.8, color=INK2)
        T._num(numbers, f"{k}/final/share_at_cap", fr.get("share_at_cap"), f"final_readouts.{k}.share_at_cap")
        T._num(numbers, f"{k}/final/share_loop", fr.get("share_loop"), f"final_readouts.{k}.share_loop")
        T._num(numbers, f"{k}/final/share_truncated", fr.get("share_truncated"), f"final_readouts.{k}.share_truncated")
        T._num(numbers, f"{k}/final/n_loop", fr.get("n_loop"), f"final_readouts.{k}.n_loop")
        T._num(numbers, f"{k}/final/n_truncated", fr.get("n_truncated"), f"final_readouts.{k}.n_truncated")
    for x, (k, c) in enumerate(cells):
        if not c.get("usable"):
            _slot_bar(ax, x, max(top, 0.02), label=_slot_word(c))
    ax.axhline(IC.IFEVAL_TRIGGER_THRESHOLD, color=INK, lw=0.9, ls="--", zorder=4)
    ax.text(len(cells) - 0.5, IC.IFEVAL_TRIGGER_THRESHOLD, " 1 % trigger", va="bottom", ha="right", fontsize=7, color=INK)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"{c['model_key']}/{c['view']}\n{_short(c['cell'])}" for _, c in cells], rotation=90, fontsize=6.4)
    ax.set_ylabel("residual at-cap share at exit (merged, 1,082 draws)", fontsize=8)
    ax.set_ylim(0, max(top * 1.35, 0.04))
    fig.legend(handles=[Patch(color=CAT["magenta"], label="loop (the cap ladder's whitespace 4-gram rule)"), Patch(color=CAT["orange"], label="truncated (at cap, not a loop)"),
                        Patch(facecolor="none", edgecolor=MUTED, hatch="////", label="HELD / PENDING slot")],
               loc="upper right", fontsize=7, frameon=False, bbox_to_anchor=(0.99, 0.9))
    _title2(fig, "Ladder exits", "per cell: exit rung (2,048 / 4,096 / 8,192 = last) + word; residual = loop + truncated")
    _layout(fig, left=0.07, right=0.99)
    return fig, numbers


def fig_ladder(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mc, Patch = T._mpl()
    promoted = [(k, c) for k, c in _visible_cells(report) if c["ladder"]["promoted"]]
    fig, ax = plt.subplots(figsize=(8.4, max(3.0, 0.4 * len(promoted) + 1.6)))
    T._recessive(ax)
    numbers: list[dict[str, Any]] = []
    colors = {CL.REDRAW_STILL_AT_CAP: CAT["magenta"], CL.REDRAW_EXTENDED: CAT["aqua"], CL.REDRAW_BOUNDARY_LANDING: CAT["yellow"]}
    for y, (k, c) in enumerate(promoted):
        left = 0
        for cap in c["ladder"]["promoted"]:
            outs = c["redraw_outcomes"].get(str(cap)) or {}
            for o, n in outs.items():
                ax.barh(y, n, left=left, color=colors.get(o, MUTED), height=0.6, zorder=3)
                left += n
            T._num(numbers, f"{k}/redraws/{cap}", outs, f"cells.{k}.redraw_outcomes.{cap}")
        res = c["ladder"]["residual"] or {}
        T._num(numbers, f"{k}/residual/n_at_cap", res.get("n_at_cap"), f"cells.{k}.ladder.residual.n_at_cap")
        ax.text(left + 0.5, y, f"residual {res.get('n_at_cap')} at cap (final {res.get('final_cap')}, {res.get('final_exit')})", va="center", fontsize=6.6, color=INK2)
    ax.set_yticks(range(len(promoted)))
    ax.set_yticklabels([f"{c['model_key']}/{c['view']}: {_short(c['cell'])}" for _, c in promoted], fontsize=7)
    ax.set_xlabel("re-drawn draws per rung (still at cap / extended / boundary landing)")
    ax.invert_yaxis()
    if not promoted:
        ax.text(0.5, 0.5, "no cell promoted (every pass-1 at-cap share <= 1 %)", transform=ax.transAxes, ha="center", fontsize=8, color=INK2)
    fig.legend(handles=[Patch(color=colors[o], label=o) for o in colors], loc="lower center", ncol=3, fontsize=7, frameon=False)
    _title2(fig, "Rung outcomes", "per promoted cell: re-drawn draws per rung; 8,192 is the last rung; residual at exit")
    _layout(fig, left=0.3, right=0.98)
    _footnote(fig, report, [k for k, _ in promoted], why="promoted cells only")
    return fig, numbers


def fig_truncation_cost(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mc, Patch = T._mpl()
    excl = set(report["design"].get("exclude_families") or [])
    tc = [t for t in report["truncation_cost"] if report["cells"][t["cell_key"]]["family"] not in excl]
    fig, ax = plt.subplots(figsize=(8.0, max(3.0, 0.4 * len(tc) + 1.6)))
    T._recessive(ax)
    ax.axvline(0, color=BASELINE, lw=0.9)
    numbers: list[dict[str, Any]] = []
    for y, t in enumerate(tc):
        for j, metric in enumerate(("prompt_strict", "inst_strict")):
            s = t["metrics"][metric]
            yy = y + (j - 0.5) * 0.25
            ax.plot([s["ci_low"], s["ci_high"]], [yy, yy], color=CAT["blue"] if j == 0 else CAT["orange"], lw=1.4)
            ax.plot([s["point"]], [yy], marker="o", color=CAT["blue"] if j == 0 else CAT["orange"], ms=5)
            i_t = report["truncation_cost"].index(t)
            T._num(numbers, f"{t['cell_key']}/{metric}/merged_minus_r2048", s["point"], f"truncation_cost.{i_t}.metrics.{metric}.point")
            for f in ("ci_low", "ci_high"):
                T._num(numbers, f"{t['cell_key']}/{metric}/merged_minus_r2048/{f}", s[f], f"truncation_cost.{i_t}.metrics.{metric}.{f}")
    ax.set_yticks(range(len(tc)))
    ax.set_yticklabels([f"{t['model_key']}/{t['view']}: {_short(t['cell'])}" for t in tc], fontsize=7)
    ax.set_xlabel("merged - r2048 (fraction); blue = prompt strict, orange = inst strict")
    ax.invert_yaxis()
    if not tc:
        ax.text(0.5, 0.5, "no promoted cell: no truncation cost to file", transform=ax.transAxes, ha="center", fontsize=8, color=INK2)
    _title2(fig, "Truncation cost", "merged - r2048 per promoted cell, item-paired 95 % CIs; at-cap non-loop draws re-drawn")
    _layout(fig, left=0.3, right=0.98)
    _footnote(fig, report, [t["cell_key"] for t in tc], why="promoted cells only")
    return fig, numbers


def fig_completion_tokens(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mc, Patch = T._mpl()
    cells = _visible_cells(report)
    fig, ax = plt.subplots(figsize=(max(8.0, 0.32 * len(cells) + 2), 4.0))
    T._recessive(ax)
    numbers: list[dict[str, Any]] = []
    for x, (k, c) in enumerate(cells):
        q = c.get("completion_tokens") or {}
        if not q:
            continue
        ax.plot([x, x], [q.get("p50", 0), q.get("p99", 0)], color=FAMILY_COLOR[cell_family(c["cell"])], lw=2.2, alpha=0.6, zorder=3)
        ax.plot([x], [q.get("mean", 0)], marker="o", color=FAMILY_COLOR[cell_family(c["cell"])], ms=4, zorder=4)
        for f in ("mean", "p50", "p99"):
            T._num(numbers, f"{k}/completion_tokens/{f}", q.get(f), f"cells.{k}.completion_tokens.{f}")
    for cap in CAPS:
        ax.axhline(cap, color=BASELINE, lw=0.8, ls=":")
        ax.text(len(cells) - 0.5, cap, f" cap {cap}", va="bottom", ha="right", fontsize=6.6, color=INK2)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"{c['model_key']}/{c['view']}\n{_short(c['cell'])}" for _, c in cells], rotation=90, fontsize=6.4)
    ax.set_ylabel("completion tokens per draw (pass 1): mean dot, p50-p99 band")
    _title2(fig, "Completion lengths", "pass-1 completion tokens per cell vs the three caps (usage ledger, 1,082 draws per cell)")
    _layout(fig, left=0.07, right=0.99)
    _footnote(fig, report, [k for k, _ in cells], why="measured cells only")
    return fig, numbers


def fig_ledger(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mc, Patch = T._mpl()
    legs = [(k, v) for k, v in report["legs"].items() if v["status"] == LEG_STATUS_USABLE and report["cells"].get(V.cell_key(v["model_key"], v["view"], v["cell"]), {}).get("family") not in set(report["design"].get("exclude_families") or [])]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.24 * len(legs) + 2), 3.8))
    T._recessive(ax)
    numbers: list[dict[str, Any]] = []
    for x, (k, v) in enumerate(legs):
        led = v["ledger"]
        ax.bar(x, led.get("n_forwarded") or 0, color=CAT["blue"], width=0.7, zorder=3)
        ax.bar(x, led.get("n_length") or 0, color=CAT["orange"], width=0.7, zorder=4)
        ax.plot([x], [led.get("n_at_cap_recorded") or 0], marker="_", color=INK, ms=8, mew=1.4, zorder=5)
        T._num(numbers, f"{k}/ledger_at_cap", led.get("n_length"), f"legs.{k}.ledger.n_length")
        T._num(numbers, f"{k}/forwarded", led.get("n_forwarded"), f"legs.{k}.ledger.n_forwarded")
        T._num(numbers, f"{k}/recorded_at_cap", led.get("n_at_cap_recorded"), f"legs.{k}.ledger.n_at_cap_recorded")
    ax.set_xticks(range(len(legs)))
    ax.set_xticklabels([f"{v['model_key']}/{v['view']}\n{_short(v['cell'])}@{v['cap']}" for _, v in legs], rotation=90, fontsize=5.8)
    ax.set_ylabel("ledger rows per leg: forwarded (blue), finish_reason == length (orange); tick = recorded at-cap")
    fig.legend(handles=[Patch(color=CAT["blue"], label="forwarded task rows"), Patch(color=CAT["orange"], label="ledger at-cap (finish_reason length)")],
               loc="upper right", fontsize=7, frameon=False, bbox_to_anchor=(0.99, 0.9))
    _title2(fig, "Ledger reconciliation", "per leg: forwarded rows == attempted draws; ledger at-cap (orange) == recorded (tick)")
    _layout(fig, left=0.07, right=0.99)
    _footnote(fig, report, [V.cell_key(v["model_key"], v["view"], v["cell"]) for _, v in legs], why="usable legs only")
    return fig, numbers


def fig_cost(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mc, Patch = T._mpl()
    legs = [(k, v) for k, v in report["cost"]["legs"].items() if report["legs"][k]["cap"] == CAP_BASE]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.3 * len(legs) + 2), 3.8))
    T._recessive(ax)
    numbers: list[dict[str, Any]] = []
    for x, (k, v) in enumerate(legs):
        ax.bar(x - 0.2, v["gpu_h_central_estimate"] or 0, width=0.4, color=MUTED, zorder=3)
        ax.bar(x + 0.2, v["gpu_h_actual"], width=0.4, color=CAT["blue"], zorder=3)
        T._num(numbers, f"{k}/gpu_h_actual", v["gpu_h_actual"], f"cost.legs.{k}.gpu_h_actual")
        T._num(numbers, f"{k}/gpu_h_central_estimate", v["gpu_h_central_estimate"], f"cost.legs.{k}.gpu_h_central_estimate")
    ax.set_xticks(range(len(legs)))
    ax.set_xticklabels([k.replace("@2048", "") for k, _ in legs], rotation=90, fontsize=5.8)
    ax.set_ylabel("GPU-hours per pass-1 cell: central estimate (grey) vs actual (blue)")
    _title2(fig, "GPU-hours", "per pass-1 cell: CENTRAL estimate (never a bound) vs actual = ledger span + 250 s adder")
    _layout(fig, left=0.07, right=0.99)
    _footnote(fig, report, [k.rsplit("@", 1)[0] for k, _ in legs], why="pass-1 legs only")
    return fig, numbers


def _make_contrast_builders(models: Sequence[str]) -> dict[str, Callable[[Mapping[str, Any]], tuple[Any, list[dict[str, Any]]]]]:
    return {f"final_ifeval_contrasts_{m}": (lambda report, _m=m: _fig_contrasts_model(report, _m)) for m in models}


NOISE_FLOOR_STEM = "final_ifeval_noise_floor"
#: the two scope labels every noise-floor panel carries: the hollow marks are the pass-1 attestation's reading, the
#: filled marks the merged estimate under ITS attestation — never one heading for both
PANEL_SCOPE_PASS1 = "pass 1 (r2048, hollow)"
PANEL_SCOPE_MERGED = "merged (filled)"
PANEL_MERGED_PROTOCOL_WORD = "adaptive-protocol variability"


def merged_scope_word(pair: Mapping[str, Any]) -> str:
    """The MERGED reading's scope word for a filed pair's panel, from its merged attestation (never from the pass-1 label):
    the protocol's variability with the count of final draws differing in cap; ``= pass 1`` when no view promoted; the same
    byte-identical floor when every final draw agrees in bytes and cap; ``not attested`` otherwise."""
    ma = pair.get("merged_attestation") or {}
    label = pair.get("merged_label")
    if label == MERGED_LABEL_PROTOCOL:
        # two lines: the panel is ~4.8 in wide and the one-line form clipped at the axes' right edge
        return (f"{PANEL_MERGED_PROTOCOL_WORD} ({ma.get('n_final_cap_differs')} of {ma.get('n_draws')} final draws differ in cap),\n"
                "    not a byte-identical-request floor")
    if label == MERGED_LABEL_BYTE_IDENTICAL:
        return ("= pass 1 (no promotion on either view)" if ma.get("trivially_identical")
                else "byte-identical requests and caps on every final draw: the same measured floor")
    return "not attested"


def _pp(x: float) -> float:
    return 100.0 * float(x)


def fig_noise_floor_model(report: Mapping[str, Any], model_key: str) -> tuple[Any, list[dict[str, Any]]]:
    """One panel per FILED pair of ``model_key``: it - main on the four metrics (merged filled, r2048 hollow; item-paired CIs),
    the per-draw discordance on the primary metric written in the panel, the attestation's label in the panel title — a pair
    that measures a difference is labelled so in place, never drawn as a floor."""
    plt, _mc, Patch = T._mpl()
    i_block, block = next((i, n) for i, n in enumerate(report["noise_floor"]) if n["model_key"] == model_key)
    pairs = [(j, p) for j, p in enumerate(block["pairs"]) if p["status"] == "filed"]
    n = max(len(pairs), 1)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n + 1.2, 4.4), sharey=True, squeeze=False)
    numbers: list[dict[str, Any]] = []
    ys = list(range(len(METRICS)))[::-1]
    for ax, (j, p) in zip(axes[0], pairs):
        T._recessive(ax)
        ax.axvline(0, color=BASELINE, lw=0.9, zorder=1)
        color = FAMILY_COLOR[cell_family(p["cell"])]
        for y, metric in zip(ys, METRICS):
            m, r = p["readings"][READING_MERGED][metric], p["readings"][READING_PASS1][metric]
            ax.plot([_pp(m["ci_low"]), _pp(m["ci_high"])], [y + 0.12, y + 0.12], color=color, lw=1.6, zorder=3)
            ax.plot([_pp(m["point"])], [y + 0.12], marker="o", color=color, ms=5.5, zorder=4)
            ax.plot([_pp(r["ci_low"]), _pp(r["ci_high"])], [y - 0.12, y - 0.12], color=color, lw=0.8, alpha=0.6, zorder=3)
            ax.plot([_pp(r["point"])], [y - 0.12], marker="o", mfc="none", mec=color, ms=4.5, zorder=4)
            ax.text(1.01, y, f"{_pp(m['point']):+.2f} pp [{_pp(m['ci_low']):+.2f}, {_pp(m['ci_high']):+.2f}]", transform=ax.get_yaxis_transform(),
                    fontsize=6.2, color=INK2, va="center", clip_on=False)
            base = f"noise_floor.{i_block}.pairs.{j}.readings"
            for reading, s in ((READING_MERGED, m), (READING_PASS1, r)):
                for f in ("point", "ci_low", "ci_high"):
                    T._num(numbers, f"{p['name']}/{reading}/{metric}/{f}", s[f], f"{base}.{reading}.{metric}.{f}")
        pd = p["per_draw"][READING_MERGED][NOISE_PRIMARY_METRIC]
        cls = pd["classes"]
        # the per-draw reading INSIDE the panel (below the lowest metric), never under the axes where the legend sits
        ax.text(0.01, 0.015, f"{NOISE_PRIMARY_METRIC}, merged: {pd['n_discordant_draws']} of {pd['n_draws']} draws discordant\n"
                             f"(main-only pass {cls[CLASS_MAIN_ONLY]}, it-only pass {cls[CLASS_IT_ONLY]}); {pd['n_discordant_items']} items",
                transform=ax.transAxes, fontsize=6.4, color=INK2, ha="left", va="bottom")
        for c in (CLASS_MAIN_ONLY, CLASS_IT_ONLY):
            T._num(numbers, f"{p['name']}/per_draw/{c}", cls[c], f"noise_floor.{i_block}.pairs.{j}.per_draw.{READING_MERGED}.{NOISE_PRIMARY_METRIC}.classes.{c}")
        T._num(numbers, f"{p['name']}/per_draw/n_discordant_draws", pd["n_discordant_draws"],
               f"noise_floor.{i_block}.pairs.{j}.per_draw.{READING_MERGED}.{NOISE_PRIMARY_METRIC}.n_discordant_draws")
        att = p["attestation"] or {}
        pass1_word = "measured noise floor" if p["label"] == LABEL_FLOOR else f"measures a difference in: {', '.join(att.get('failing') or [])}"
        # BOTH scopes INSIDE the panel: the hollow r2048 marks are the pass-1 attestation's, the filled marks the
        # MERGED estimate whose own attestation may read "adaptive-protocol variability" — the pass-1 word never heads a panel
        # whose filled marker it does not describe; the figure title owns the space above the axes
        ax.text(0.01, 0.985, f"{_short(p['cell'])}(it) - {_short(p['cell'])}(main)\n{PANEL_SCOPE_PASS1}: {pass1_word}\n"
                             f"{PANEL_SCOPE_MERGED}: {merged_scope_word(p)}",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.8, color=INK, linespacing=1.25)
        ax.set_ylim(-1.35, len(METRICS) - 0.3 + 1.75)
        ax.set_yticks(ys)
        ax.set_yticklabels([m.replace("_", " ") for m in METRICS], fontsize=7.4)
        ax.set_xlabel("it - main (percentage points)")
    if not pairs:
        axes[0][0].text(0.5, 0.5, "no identical-by-construction pair filed", transform=axes[0][0].transAxes, ha="center", fontsize=8, color=INK2)
    _title2(fig, "Noise floor", f"{model_key}: it - main on identical cells by construction; pass 1 hollow, merged filled")
    # a two-row legend naming which attestation each marker answers to, placed where nothing else is: with two or more panels the
    # title band's right half (the title text owns its left half); with one panel the bottom-right corner, far right of that
    # panel's x-label (a centred legend under the axes collided with the middle panel's x-label; one inside a panel with its
    # per-draw reading)
    handles = [plt.Line2D([], [], marker="o", color=INK, ls="", label="filled: merged reading, its own attestation"),
               plt.Line2D([], [], marker="o", mfc="none", mec=INK2, ls="", label="hollow: r2048 = pass 1, the pass-1 attestation")]
    if n >= 2:
        fig.legend(handles=handles, loc="upper right", ncol=1, fontsize=6.4, frameon=False, bbox_to_anchor=(0.998, 0.998), handletextpad=0.4, labelspacing=0.25)
    else:
        fig.legend(handles=handles, loc="lower right", ncol=1, fontsize=6.0, frameon=False, bbox_to_anchor=(0.998, -0.012), handletextpad=0.4, labelspacing=0.25)
    _layout(fig, left=0.11, right=0.80 if n == 1 else 0.90, wspace=0.55)
    return fig, numbers


def _make_noise_floor_builders(report: Mapping[str, Any]) -> dict[str, Callable[[Mapping[str, Any]], tuple[Any, list[dict[str, Any]]]]]:
    excl = set(report["design"].get("exclude_families") or [])
    return {f"{NOISE_FLOOR_STEM}_{n['model_key']}": (lambda report, _m=n["model_key"]: fig_noise_floor_model(report, _m))
            for n in report["noise_floor"] if n["status"] == "filed" and C.FAMILY_BY_MODEL_KEY.get(n["model_key"]) not in excl}


FIGURE_BUILDERS_FIXED: Mapping[str, Callable[[Mapping[str, Any]], tuple[Any, list[dict[str, Any]]]]] = {
    "final_ifeval_metrics_by_cell": fig_metrics_by_cell, "final_ifeval_shares": fig_shares, "final_ifeval_exits": fig_exits, "final_ifeval_ladder": fig_ladder,
    "final_ifeval_truncation_cost": fig_truncation_cost, "final_ifeval_ledger": fig_ledger, "final_ifeval_cost": fig_cost,
    # last and alone on its line: the secret scanner's generic rule reads a quoted string after the word "tokens"
    "final_ifeval_completion_tokens": fig_completion_tokens,
}


def figures_for(report: Mapping[str, Any]) -> dict[str, Callable[[Mapping[str, Any]], tuple[Any, list[dict[str, Any]]]]]:
    excl = set(report["design"].get("exclude_families") or [])
    models = [m for m in report["design"]["models"] if C.FAMILY_BY_MODEL_KEY.get(m) not in excl]
    return {**_make_contrast_builders(models), **_make_noise_floor_builders(report), **FIGURE_BUILDERS_FIXED}


def _noise_floor_caption(report: Mapping[str, Any], stem: str, model_key: str) -> str:
    """The caption STATES THE READING of every pair the figure draws (the ruling: the spread is a caption number)."""
    block = next(n for n in report["noise_floor"] if n["model_key"] == model_key)
    parts = []
    for p in block["pairs"]:
        if p["status"] != "filed":
            parts.append(f"{p['cell']}: {p['status']} ({p.get('reason')})")
            continue
        s = p["readings"][READING_MERGED][NOISE_PRIMARY_METRIC]
        pd = p["per_draw"][READING_MERGED][NOISE_PRIMARY_METRIC]
        word = "a MEASURED NOISE FLOOR" if p["label"] == LABEL_FLOOR else f"NOT a floor — measures the difference in {', '.join(p['attestation']['failing'])}"
        ma = p.get("merged_attestation") or {}
        merged_word = ("the merged reading is the pass-1 reading (no promotion on either view)" if ma.get("trivially_identical") else
                       f"the merged reading's final draws are byte-identical across the views too ({ma.get('n_final_body_identical')} of {ma.get('n_draws')})"
                       if ma.get("identical") else
                       f"the MERGED reading is the variability of the whole adaptive protocol, not a byte-identical-request floor: the two views' ladders "
                       f"re-drew differently, {ma.get('n_final_cap_differs')} of {ma.get('n_draws')} final draws differ in cap")
        parts.append(f"{p['cell']}(it) - {p['cell']}(main) is {word} on pass 1 (2,048 request bodies): {NOISE_PRIMARY_METRIC} (merged) {_pp(s['point']):+.2f} pp "
                     f"[{_pp(s['ci_low']):+.2f}, {_pp(s['ci_high']):+.2f}] (main {_pp(s['legs']['main']['point']):.2f} %, it {_pp(s['legs']['it']['point']):.2f} %; "
                     f"never pooled), {pd['n_discordant_draws']} of {pd['n_draws']} draws discordant "
                     f"(main-only {pd['classes'][CLASS_MAIN_ONLY]}, it-only {pd['classes'][CLASS_IT_ONLY]}); {merged_word}")
    return (f"{stem}: {model_key}, it - main on cells the matrix carries on both views with no object, identical by construction (same served "
            f"checkpoint and revision, SI prefix, geometry entry, sampling row, cap, seeds; the attestation checks serving geometry, batch composition "
            f"and the scoring path from the two PASS-ONE stores; the merged reading's final request bodies and caps are attested separately). "
            f"{' | '.join(parts)}. The other three metrics and the r2048 reading are drawn alongside. "
            f"A floor is reported BESIDE the contrasts, never as a threshold. "
            f"IFEval, k = 2 seeded trials, thinking OFF, vendor presets; ladder trigger = {report['design']['trigger_provenance']}; loops by the "
            "whitespace-token 4-gram rule.")


def _slot_sentence(report: Mapping[str, Any]) -> str:
    slots = report.get("slots") or {}
    if not slots:
        return ""
    held = [k for k, v in slots.items() if v["status"] == IC.STATUS_HELD]
    pend = [k for k, v in slots.items() if v["status"] == IC.STATUS_DEFERRED]
    return (f" Hatched slots are NOT measured: {len(held)} HELD ({', '.join(held)}; {next((v['reason'] for v in slots.values() if v['status'] == IC.STATUS_HELD), '')}) "
            f"and {len(pend)} PENDING ({PENDING_LABEL}).")


def _pct(x: Any) -> str:
    return "—" if x is None else f"{100 * float(x):.2f} %"


def _contrast_caption(report: Mapping[str, Any], stem: str, model_key: str) -> str:
    cs = [c for c in report["contrasts"] if c["model_key"] == model_key and c["status"] == "filed"]
    parts = []
    for c in cs:
        s = c["readings"][READING_MERGED]["prompt_strict"]
        parts.append(f"{c['view']}: {_short(c['cell'])} {_pp(s['point']):+.1f} pp [{_pp(s['ci_low']):+.1f}, {_pp(s['ci_high']):+.1f}] -> {c['verdict']['word']}")
    floor = noise_floor_pair(report, model_key, IC.CELL_NONE)
    fl = ""
    if floor is not None and floor.get("label") == LABEL_FLOOR:
        s = floor["readings"][READING_MERGED]["prompt_strict"]
        ma = floor.get("merged_attestation") or {}
        scope = (" its merged reading is its pass-1 reading (no promotion on either view), so the merged-scope relabel of promoted pairs does not touch this band"
                 if ma.get("trivially_identical") else f" merged-reading attestation: {ma.get('label')}")
        fl = (f" Grey band = the none(it) - none(main) noise floor on prompt strict: {_pp(s['point']):+.2f} pp [{_pp(s['ci_low']):+.2f}, {_pp(s['ci_high']):+.2f}] "
              f"(a floor, never a threshold;{scope}).")
    words = Counter(c["verdict"]["word"] for c in cs)
    return (f"{stem}: {model_key}, cell - none per view on the MERGED reading (prompt strict shown; inst strict beside), item-paired 95 % CIs; the "
            f"word is the preregistered rule (both prompt-level CIs exclude zero with one sign, else 'not resolved at this cap'); no pooled word. "
            f"Reading: {'; '.join(parts)}. Words: {dict(words)}.{fl}{_slot_sentence(report)} IFEval, k = 2 seeded trials, thinking OFF, vendor presets; "
            f"ladder trigger = {report['design']['trigger_provenance']}; loops by the whitespace-token 4-gram rule.")


def _metrics_caption(report: Mapping[str, Any], stem: str) -> str:
    parts = []
    for model in report["design"]["models"]:
        cells = [(k, c) for k, c in report["cells"].items() if c.get("usable") and c["model_key"] == model]
        if not cells:
            continue
        pts = {k: c["metrics"][READING_MERGED]["prompt_strict"]["point"] for k, c in cells}
        nones = [k for k in pts if k.endswith("/" + IC.CELL_NONE)]
        parts.append(f"{model}: prompt strict (merged) none = " + ", ".join(f"{k.split('/')[1]} {_pp(pts[k]):.1f} %" for k in nones)
                     + f"; arms span {_pp(min(pts.values())):.1f}-{_pp(max(pts.values())):.1f} % over {len(cells)} cells")
    return (f"{stem}: the four IFEval metrics per model x arm on the MERGED reading (the ladder-aware final readout), bars with item-paired 95 % CI "
            f"whiskers. Reading: {' | '.join(parts)}.{_slot_sentence(report)} IFEval, k = 2 seeded trials, thinking OFF, vendor presets; ladder trigger = "
            f"{report['design']['trigger_provenance']}; loops by the whitespace-token 4-gram rule.")


def _shares_caption(report: Mapping[str, Any], stem: str) -> str:
    over = [(k, c) for k, c in report["cells"].items() if c.get("usable") and (c["shares"].get("share_at_cap") or 0) > IC.IFEVAL_TRIGGER_THRESHOLD]
    parts = [f"{k} {_pct(c['shares']['share_at_cap'])} (loop {c['shares'].get('n_loop')}, truncated {c['shares'].get('n_truncated')}) -> {c['ladder']['promoted']}"
             for k, c in over]
    return (f"{stem}: pass-1 (2,048) at-cap share of the 1,082 draws per cell = truncated + loop; the dashed line is the 1 % trigger of RAW at-cap draws. "
            f"Reading: {len(over)} of {sum(1 for c in report['cells'].values() if c.get('usable'))} cells exceed it and were promoted: {'; '.join(parts) or 'none'}."
            f"{_slot_sentence(report)} IFEval, k = 2 seeded trials, thinking OFF, vendor presets; ladder trigger = {report['design']['trigger_provenance']}; "
            f"loops by the whitespace-token 4-gram rule.")


def _fit_condition_short(report: Mapping[str, Any], cell_key: str) -> str:
    """The cell's fit condition FROM ITS MANIFEST, short: the fit weights vs the served weights and the producer's flag."""
    fp = (report["cells"].get(cell_key) or {}).get("fit_provenance") or {}
    if not fp.get("recorded") or fp.get("model_id") is None:
        return fp.get("condition") or V.FIT_CONDITION_UNRECORDED
    same = fp.get("fit_weights") == "served_checkpoint"
    return (f"document-fit LoRA on {'the served weights themselves' if same else 'OTHER weights'} ({fp.get('model_id')} @ {str(fp.get('revision'))[:8]}), "
            f"served on {fp.get('served_model_id')} @ {str(fp.get('served_revision'))[:8]}; transfer = {fp.get('transfer')}; "
            f"chat_template_bypassed = {fp.get('chat_template_bypassed')}")


def _net_change(report: Mapping[str, Any], cell_key: str, metric: str = "prompt_strict") -> str:
    """The OBSERVED net change of the ladder on a promoted cell (merged - r2048 with its CI), the number the prose may claim."""
    t = next((t for t in report["truncation_cost"] if t["cell_key"] == cell_key), None)
    if t is None:
        return "not promoted"
    s = t["metrics"][metric]
    return f"{_pp(s['point']):+.2f} pp [{_pp(s['ci_low']):+.2f}, {_pp(s['ci_high']):+.2f}]"


def _exclusion_sentence(report: Mapping[str, Any], shown_cell_keys: Sequence[str], *, why: str) -> str:
    ns = _not_shown(report, shown_cell_keys)
    parts = []
    if ns["held"] or ns["pending"]:
        parts.append(f"{len(ns['held'])} HELD + {len(ns['pending'])} PENDING slots are not measured and not drawn")
    if ns["usable_not_shown"]:
        parts.append(f"{len(ns['usable_not_shown'])} measured cells are not drawn ({why})")
    return (" Not shown: " + "; ".join(parts) + ".") if parts else ""


def _exits_caption(report: Mapping[str, Any], stem: str) -> str:
    frs = report.get("final_readouts") or {}
    ceil = [(k, f) for k, f in frs.items() if f["exit"] == CL.EXIT_CEILING]
    noel = [(k, f) for k, f in frs.items() if f["exit"] == IR.EXIT_TRIGGERED_NO_ELIGIBLE]
    parts = [f"{k}: {_pct(f['share_at_cap'])} at cap at {f['exit_cap']} = {f['n_loop']} loops + {f['n_truncated']} truncated; "
             f"{f.get('n_at_cap_pass_prompt_strict')} of {f['n_at_cap']} at-cap responses pass prompt strict as scored; observed net change of the "
             f"ladder (merged - r2048, prompt strict) {_net_change(report, k)}; object: {_fit_condition_short(report, k)}" for k, f in ceil]
    parts2 = [f"{k}: {_pct(f['share_at_cap'])} at {f['exit_cap']}, every remaining at-cap draw a loop (no eligible re-draw)" for k, f in noel]
    tally = Counter(f["exit"] for f in frs.values())
    # the loop finding, stated OPERATIONALLY: loops were never re-drawn, so their continuation was not tested;
    # the object condition comes from each cell's manifest, never from the view name
    return (f"{stem}: per cell the ladder's EXIT rung (2,048 / 4,096 / 8,192 = the last rung) and word, with the MERGED reading's residual at-cap "
            f"share split into loops (the cap ladder's whitespace 4-gram rule) and truncation (at cap, not a loop) by the record of the leg that "
            f"last drew each draw. Exits: {dict(tally)}. Ceiling residuals: {'; '.join(parts) or 'none'}. {'; '.join(parts2)}. Loops are never "
            f"re-drawn by the protocol (only at-cap non-loop draws are), so what a looping draw would have produced past the cap was not measured; "
            f"the at-cap responses are scored as produced.{_slot_sentence(report)} IFEval, k = 2 seeded trials, thinking OFF, vendor presets; "
            f"ladder trigger = {report['design']['trigger_provenance']}.")


def _ladder_caption(report: Mapping[str, Any], stem: str) -> str:
    promoted = [(k, c) for k, c in report["cells"].items() if c.get("usable") and c["ladder"]["promoted"]]
    parts = [f"{k} -> {c['ladder']['promoted']} ({(c['ladder']['residual'] or {}).get('final_exit')}, residual {(c['ladder']['residual'] or {}).get('n_at_cap')})"
             for k, c in promoted]
    return (f"{stem}: per promoted cell the re-drawn draws per rung (still at cap / extended / boundary landing) and the residual at exit. "
            f"Reading: {len(promoted)} cells promoted: {'; '.join(parts) or 'none'}.{_exclusion_sentence(report, [k for k, _ in promoted], why='promoted cells only')} "
            f"IFEval, k = 2 seeded trials, thinking OFF, vendor presets; ladder trigger = {report['design']['trigger_provenance']}; loops by the whitespace-token 4-gram rule.")


def _truncation_caption(report: Mapping[str, Any], stem: str) -> str:
    parts = [f"{t['cell_key']} {_pp(t['metrics']['prompt_strict']['point']):+.2f} pp [{_pp(t['metrics']['prompt_strict']['ci_low']):+.2f}, {_pp(t['metrics']['prompt_strict']['ci_high']):+.2f}]"
             for t in report["truncation_cost"]]
    # the OBSERVED net change of re-drawing — not "what the ladder bought back", not a maximum: loops were never re-drawn
    return (f"{stem}: merged - r2048 per promoted cell (prompt strict and inst strict; item-paired 95 % CIs) — the observed net change from re-drawing the "
            f"at-cap NON-loop draws at the same seeds; loops were never re-drawn, so this is the protocol's observed change, not a bound on what a wider "
            f"ladder could recover. Reading (prompt strict): {'; '.join(parts) or 'no promoted cell'}."
            f"{_exclusion_sentence(report, [t['cell_key'] for t in report['truncation_cost']], why='promoted cells only')} IFEval, k = 2 seeded trials, thinking "
            f"OFF, vendor presets; ladder trigger = {report['design']['trigger_provenance']}; loops by the whitespace-token 4-gram rule.")


def _completion_caption(report: Mapping[str, Any], stem: str) -> str:
    parts = []
    for model in report["design"]["models"]:
        qs = [c["completion_tokens"] for c in report["cells"].values() if c.get("usable") and c["model_key"] == model and c.get("completion_tokens")]
        if qs:
            parts.append(f"{model}: mean {min(q['mean'] for q in qs):.0f}-{max(q['mean'] for q in qs):.0f}, p99 {min(q['p99'] for q in qs)}-{max(q['p99'] for q in qs)} tokens")
    shown = [k for k, c in report["cells"].items() if c.get("usable")]
    return (f"{stem}: pass-1 completion tokens per cell (mean dot, p50-p99 band) against the three caps. Reading: {'; '.join(parts)}."
            f"{_exclusion_sentence(report, shown, why='measured cells only')} IFEval, k = 2 seeded "
            f"trials, thinking OFF, vendor presets; ladder trigger = {report['design']['trigger_provenance']}; loops by the whitespace-token 4-gram rule.")


def _ledger_caption(report: Mapping[str, Any], stem: str) -> str:
    rec = report["gates"]["I1_ledger"].get("reconciliation") or {}
    ok = sum(1 for r in rec.values() if r["ledger_at_cap"] == r["recorded_at_cap"] == r["summary_n_at_cap"])
    shown = [k for k, c in report["cells"].items() if c.get("usable")]
    return (f"{stem}: per leg the forwarded ledger rows (== attempted draws) and finish_reason == length (== the recorded at-cap, the tick). Reading: "
            f"ledger at-cap == recorded at-cap == the summary's on {ok} of {len(rec)} legs; I1 {report['gates']['I1_ledger']['status']}."
            f"{_exclusion_sentence(report, shown, why='usable legs only')} IFEval, k = 2 seeded "
            f"trials, thinking OFF, vendor presets; ladder trigger = {report['design']['trigger_provenance']}; loops by the whitespace-token 4-gram rule.")


def _cost_caption(report: Mapping[str, Any], stem: str) -> str:
    cost = report["cost"]
    shown = [k.rsplit("@", 1)[0] for k in cost["legs"] if k.endswith(f"@{CAP_BASE}")]
    return (f"{stem}: GPU-hours per pass-1 cell, the CENTRAL estimate (grey; never a bound) beside the actual (blue: the accepted attempt's ledger span "
            f"+ the 250 s Modal standup adder). Reading: total actual {cost['total_gpu_h_actual']} GPU-h over {len(cost['legs'])} legs (pass 1 + rungs); "
            f"central estimate {cost['central_estimate_total_gpu_h'].split(' (')[0]} GPU-h.{_exclusion_sentence(report, shown, why='pass-1 legs only')} "
            f"IFEval, k = 2 seeded trials, thinking OFF, vendor presets; ladder trigger = {report['design']['trigger_provenance']}; loops by the whitespace-token 4-gram rule.")


def figure_captions(report: Mapping[str, Any]) -> dict[str, str]:
    """Every caption STATES THE READING of its figure from the report's numbers (the rule: the prose lives in the caption)."""
    fixed = {"final_ifeval_metrics_by_cell": _metrics_caption, "final_ifeval_shares": _shares_caption, "final_ifeval_exits": _exits_caption,
             "final_ifeval_ladder": _ladder_caption, "final_ifeval_truncation_cost": _truncation_caption, "final_ifeval_completion_tokens": _completion_caption,
             "final_ifeval_ledger": _ledger_caption, "final_ifeval_cost": _cost_caption}
    caps = {}
    for stem in figures_for(report):
        if stem.startswith(NOISE_FLOOR_STEM + "_"):
            caps[stem] = _noise_floor_caption(report, stem, stem[len(NOISE_FLOOR_STEM) + 1:])
        elif stem.startswith("final_ifeval_contrasts_"):
            caps[stem] = _contrast_caption(report, stem, stem[len("final_ifeval_contrasts_"):])
        else:
            caps[stem] = fixed[stem](report, stem)
    return caps


def _refuse_comparison_only(report: Mapping[str, Any], what: str) -> None:
    """A comparison-only report is the two-venv match's instrument and nothing else: no figure, no mirror, no W&B
    run, no rendered record may be made from it — refused by name before anything is written."""
    if report.get("comparison_only"):
        raise AnalysisError(f"{what}: a {COMPARISON_ONLY_MODE} report (the .eval logs unread; I4 / I5 PENDING; no production claim) "
                            "publishes nothing — run the production pass with the logs read")


def make_figures(report: Mapping[str, Any], figs_dir: Path) -> dict[str, Any]:
    _refuse_comparison_only(report, "figures")
    figs_dir = Path(figs_dir).expanduser().resolve()
    figs_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, list[str]] = {}
    numbers: dict[str, list[dict[str, Any]]] = {}
    for stem, fn in figures_for(report).items():
        fig, nums = fn(report)
        frac = axes_height_fraction(fig)
        if frac < AXES_MIN_FRACTION:
            raise AnalysisError(f"{stem}: the axes span {frac:.3f} of the figure height (< {AXES_MIN_FRACTION})")
        title_problem = M.figure_title_problem(fig)
        if title_problem:
            raise AnalysisError(f"{stem}: {title_problem}")
        written[stem] = [str(p) for p in T._save(fig, figs_dir, stem)]
        numbers[stem] = nums
    captions = figure_captions(report)
    (figs_dir / NUMBERS_FILENAME).write_text(json.dumps({"figures": numbers, "captions": captions, "sha": report["sha"], "analysis_code_sha": report.get("analysis_code_sha"),
                                                         "trigger_provenance": report["design"]["trigger_provenance"]}, indent=1, default=str), encoding="utf-8")
    tables = {"contrasts": report["contrasts"], "noise_floor": report["noise_floor"], "truncation": report["truncation_cost"], "cells": report["cells"],
              "gates": report["gates"], "verdict": report["verdict"], "held": report["held"], "deferred": report.get("deferred"), "slots": report.get("slots"),
              "final_readouts": report.get("final_readouts"), "cost": report["cost"], "provenance": report["provenance"],
              "captions": captions, "sha": report["sha"], "analysis_code_sha": report.get("analysis_code_sha")}
    (figs_dir / TABLES_FILENAME).write_text(json.dumps(tables, indent=1, default=str), encoding="utf-8")
    # the captions ride the manifest so the W&B images carry them; `T.mirror_figures` reads only the path keys
    return {"written": written, "skipped": {}, "numbers": str(figs_dir / NUMBERS_FILENAME), "tables": str(figs_dir / TABLES_FILENAME), "captions": captions}


def log_wandb(report: Mapping[str, Any], figs: Mapping[str, Any], *, mode: str, name: str | None = None, name_suffix: str = "") -> str | None:
    _refuse_comparison_only(report, "W&B")
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as e:
        raise AnalysisError("wandb is not importable — pass --wandb-mode disabled or use a venv that carries it") from e
    run_name = name or f"{WANDB_RUN_PREFIX}{report['sha']}{name_suffix}"
    from sisteer.wandb_entity import wandb_project
    run = wandb.init(project=wandb_project(WANDB_PURPOSE), name=run_name, mode=mode, config={"experiment": EXPERIMENT, "sha": report["sha"], "task": TASK, **report["bootstrap"],
                                                                              "matrix_sha256": report["matrix"]["sha256"]})
    captions = figs.get("captions") or {}
    for stem, paths in figs.get("written", {}).items():
        png = next((p for p in paths if p.endswith(".png")), None)
        if png:
            # the caption STATES THE READING: the same text the numbers sidecar carries, attached to the image
            wandb.log({f"ifeval/{stem}": wandb.Image(png, caption=captions.get(stem))})
    run.summary["production"] = report["verdict"]["production"]
    run.summary["gates_failed"] = report["verdict"]["gates_failed"]
    for k, c in report["cells"].items():
        if c.get("usable"):
            for reading in READINGS:
                for m in METRICS:
                    leg = (c["metrics"].get(reading) or {}).get(m)
                    if leg:
                        run.summary[f"{k}/{reading}/{m}/point"] = leg["point"]
            run.summary[f"{k}/share_at_cap"] = c["shares"]["share_at_cap"]
    for n in report["noise_floor"]:
        for p in n.get("pairs") or []:
            run.summary[f"noise_floor/{p['name']}/label"] = p["label"]
            if p["status"] == "filed":
                for reading in READINGS:
                    for m in METRICS:
                        run.summary[f"noise_floor/{p['name']}/{reading}/{m}/point"] = p["readings"][reading][m]["point"]
                    run.summary[f"noise_floor/{p['name']}/{reading}/n_discordant_draws"] = p["per_draw"][reading][NOISE_PRIMARY_METRIC]["n_discordant_draws"]
    art = wandb.Artifact(f"figures-{run_name}", type="figures")
    art.add_dir(str(Path(figs["numbers"]).parent))
    run.log_artifact(art)
    url = run.url
    run.finish()
    return url


# --------------------------------------------------------------------------- the results record
def render_results_md(report: Mapping[str, Any], template: Path = RESULTS_TEMPLATE, *, out: Path | str | None = None) -> str:
    """Fill the results record's ``{{name}}`` placeholders from the report; a placeholder the report cannot fill stays as
    ``{…}`` — visibly unfilled, never a guess. ``out`` (the path the caller will write) must not be the template
    itself (a render written over the skeleton consumes every placeholder, and the next render silently keeps the old
    numbers), and the template must still CARRY placeholders — both refused by name, before anything is rendered. A
    comparison-only report renders no record."""
    _refuse_comparison_only(report, "the results record")
    template = Path(template)
    if out is not None and Path(out).expanduser().resolve() == template.expanduser().resolve():
        raise AnalysisError(f"--results-md {out}: the output aliases the results TEMPLATE {template} — a render written over "
                            f"the skeleton consumes its placeholders and every later render keeps the old numbers; write the "
                            f"record elsewhere ({RESULTS_RECORD_REL} is the documented output) and keep the template immutable")
    text = template.read_text(encoding="utf-8")
    if "{{" not in text:
        raise AnalysisError(f"{template}: the results template carries no {{{{name}}}} placeholder — a CONSUMED template (a "
                            "render was written over the skeleton); restore results.template.md from git before rendering")
    v = report["verdict"]
    words = report["contrast_summary"]["words"]
    values = {
        "store_pin": report["sha"], "analysis_code_sha": report.get("analysis_code_sha") or "{…}", "n_launch": report["design"]["n_launch_rows"],
        "n_held": report["design"]["n_held_rows"], "n_usable_cells": v["n_usable_cells"], "n_promoted": v["n_promoted_cells"],
        "n_filed": v["n_filed"], "n_designed": report["contrast_summary"]["n_designed"], "n_lower": words.get(WORD_LOWER, 0),
        "n_higher": words.get(WORD_HIGHER, 0), "n_not_resolved": words.get(WORD_NOT_RESOLVED, 0),
        "gates_passed": report["gates"].get("n_passed"), "gates_total": len(GATE_NAMES), "gates_failed": ", ".join(v["gates_failed"]) or "none",
        "matrix_sha256": report["matrix"].get("sha256") or "{…}", "B": report["bootstrap"]["B"], "seed": report["bootstrap"]["seed"],
        "level": report["bootstrap"]["level"], "trigger_provenance": report["design"]["trigger_provenance"], "loop_rule": report["design"]["loop_rule_disclosure"],
        "seed_rule": report["design"]["seed_rule"], "total_gpu_h_actual": report["cost"]["total_gpu_h_actual"],
        "production": "PRODUCTION RESULT" if v["production"] else "NOT a production result",
        "shares_table": _shares_table(report), "contrasts_table": _contrasts_table(report), "ladder_table": _ladder_table(report),
        "ledger_table": _ledger_table(report), "gates_table": _gates_table(report), "noise_floor_table": _noise_table(report),
        "figures_publication": _figures_publication(report),
    }
    for k, val in values.items():
        text = text.replace("{{" + k + "}}", str(val))
    return text.rstrip("\n") + "\n\n" + supplement_md(report)


def _figures_publication(report: Mapping[str, Any]) -> str:
    """The figures row's account of what THIS run did with its figures, read from the report: the files the mirror received
    (`mirrored`, written only without `--no-mirror`) and the run `log_wandb` returned (`wandb_url`). A destination the run did not
    write to is never named (the template once stated a fixed mirror, serving path and W&B project for every run)."""
    if not report.get("figures"):
        return "no figures made by this run (`--no-figs`): nothing mirrored, nothing logged to W&B"
    mirrored = report.get("mirrored") or []
    dirs = sorted({_home_relative(Path(p).parent) for p in mirrored})
    mirror = (("mirrored byte-identical to " + ", ".join(f"`{d}/`" for d in dirs) + f" ({len(mirrored)} files)") if mirrored
              else "not mirrored (`--no-mirror`)")
    url = report.get("wandb_url")
    wb = (f"logged to W&B as {url}, panel keys `ifeval/<stem>`" if url
          else "no W&B run URL (`--wandb-mode disabled`, or offline until synced)")
    return f"{mirror}; {wb}"


def supplement_md(report: Mapping[str, Any]) -> str:
    """The generated AS-RUN SUPPLEMENT appended to every rendered record (the immutable template carries no placeholder for
    it): the roots and per-row pins the legs were admitted by, the FINAL per-cell readout (exit rung, the four metrics on
    the merged reading, at-cap share and its loop / truncated split, recorder verdict, pins), the HELD / PENDING slots
    with their reasons, the Nemotron certifications, the recorder statement and the cost totals — every number read from
    the report."""
    prov = report.get("provenance") or {}
    lines = ["## Appendix A — as-run supplement (generated by the analysis from the report; nothing typed by hand)", "",
             f"Record label `{report['sha']}`; analysis tip `{report.get('analysis_code_sha') or '{…}'}`; admission: {report.get('admission')} "
             f"({prov.get('admission')}).", "",
             "### A.1 Wave roots and the producer trees admitted per row", "",
             "| what | value |", "|---|---|",
             # the roots are printed home-relative: the record is committed, and an absolute path names the machine
             f"| wave roots read | {', '.join(f'`{_home_relative(g)}`' for g in report.get('gen_roots') or [report.get('gen_root')])} |"]
    for cap, trees in (prov.get("trees_by_cap") or {}).items():
        lines.append(f"| producer trees at {cap} | {', '.join(f'`{t}`' for t in trees)} |")
    sup = report.get("superseded_stores") or {}
    if sup:
        parts = []
        for k, v in sup.items():
            stores = ", ".join(f"`{str(c.get('producer_tree_sha'))[:8]}` ({(c.get('result') or {}).get('result')})" for c in v)
            parts.append(f"{k}: {stores}")
        lines.append("| superseded stores (not legs) | " + "; ".join(parts) + " |")
    lines += ["", f"{prov.get('label_note') or ''}", "", _pins_table(report), "", "### A.2 The FINAL per-cell readout (merged reading at the exit rung)", "",
              "Every score is the ladder-aware MERGED reading over the 1,082 draws (541 items x k = 2; a rung re-draws only the at-cap non-loop draws of the "
              "rung below at their pass-1 seeds); at-cap rows are scored as the checker scores them, then loop-classified; the at-cap share, the loop vs "
              "truncated split and the exit rung stand beside every score. The completion-token and throughput columns are the PASS-1 leg's (the "
              "1,082 draws at 2,048; usage ledger), labelled so; the stock column is inspect_evals' own per-trial `prompt_strict_acc ± stderr` read off "
              "each trial's retained `.eval` log (one trial = one seeded draw per prompt), for comparability with published IFEval tables only — never "
              "deciding (the intervals that decide are the item-paired bootstrap over both trials).", "", _final_table(report), "",
              "### A.3 HELD and PENDING rows (never measured)", "", _slots_table(report), "",
              "### A.4 Recorder", "", _recorder_statement(report), "",
              "### A.5 Nemotron certifications (one per producer tree x geometry entry)", "", _certifications_table(report), "",
              "### A.6 Cost", "", _cost_lines(report), "",
              "### A.7 Fit provenance of the served object per cell (from each pass-1 manifest's `fit_provenance` block)", "",
              V.FIT_CONDITION_RULE + ".", "", _fit_table(report), "",
              "### A.8 Instrument facts as served (generated from the admitted legs' records)", "", _instrument_facts_md(report), ""]
    return "\n".join(lines)


def _fit_table(report: Mapping[str, Any]) -> str:
    lines = ["| cell | object sha | fit model @ rev | served model @ rev | fit kind | transfer (manifest) | chat_template_bypassed | condition (derived) |",
             "|---|---|---|---|---|---|---|---|"]
    for k, c in report["cells"].items():
        if not c.get("usable"):
            lines.append(f"| {k} | — | — | — | — | — | — | **{c.get('slot')}** (not measured) |")
            continue
        fp = c.get("fit_provenance") or {}
        obj = str(c.get("object_sha256") or "")[:8] or "—"
        fit = f"`{fp.get('model_id')}` @ `{str(fp.get('revision'))[:8]}`" if fp.get("model_id") else "—"
        served = f"`{fp.get('served_model_id')}` @ `{str(fp.get('served_revision'))[:8]}`" if fp.get("served_model_id") else "—"
        lines.append(f"| {k} | `{obj}` | {fit} | {served} | {fp.get('fit_kind') or '—'} | {fp.get('transfer')} | {fp.get('chat_template_bypassed')} | "
                     f"{fp.get('condition')}{(' — PROBLEM: ' + fp['problem']) if fp.get('problem') else ''} |")
    return "\n".join(lines)


def _instrument_facts_md(report: Mapping[str, Any]) -> str:
    design = report.get("design") or {}
    pins = design.get("thinking_pin_as_served") or {}
    lines = ["Thinking-OFF pin AS SERVED per model family (from every usable leg's `evidence.serve_header.template_kwargs` judged by the endpoint's "
             "`thinking_header.switch`; the wrapper composes the pin on every family whose template has a switch):", "",
             "| family | models | legs | template has a switch | template_kwargs composed | header moves | as served |", "|---|---|---|---|---|---|---|"]
    for fam, rec in pins.items():
        lines.append(f"| {fam} | {', '.join(rec.get('models') or [])} | {rec.get('n_legs')} | {rec.get('switch')} | `{json.dumps(rec.get('template_kwargs'))}` | "
                     f"{rec.get('header_moves')} | {rec.get('as_served')} |")
    shrink = design.get("kv_shrink_legs") or []
    lines += ["", f"KV-derived served S (the allocator shrank the configured S): {len(shrink)} leg(s)"
              + (" — " + "; ".join(f"`{s['key']}` ({str(s.get('geometry_eval')).replace('capability:', '')}: configured {s.get('configured_seqs')} -> served {s.get('served_seqs')})"
                                  for s in shrink) if shrink else " — none") + "."]
    lines += ["", (report.get("provenance") or {}).get("label_note") or ""]
    return "\n".join(lines)


def _home_relative(path: Any) -> str:
    """A path with the user's home spelled ``~`` (a committed record never names the machine)."""
    home = str(Path.home())
    text = str(path)
    return "~" + text[len(home):] if text.startswith(home) else text


def _ci(s: Mapping[str, Any] | None, *, pct: bool = True) -> str:
    if not s:
        return "—"
    f = (lambda x: f"{100 * x:.1f}") if pct else (lambda x: f"{x:.3f}")
    return f"{f(s['point'])} [{f(s['ci_low'])}, {f(s['ci_high'])}]"


def _stock_cell(report: Mapping[str, Any], cell_key: str) -> str:
    """inspect_evals' own per-trial `prompt_strict_acc ± stderr` of the PASS-1 leg's two logs, or a stated absence."""
    leg = (report.get("legs") or {}).get(f"{cell_key}@{CAP_BASE}") or {}
    stock = ((leg.get("eval_check") or {}).get("stock_metrics_by_trial")) or {}
    parts = []
    for trial in sorted(stock):
        m = stock[trial] or {}
        acc, se = m.get("prompt_strict_acc"), m.get("prompt_strict_stderr")
        parts.append(f"t{trial}: {100 * acc:.1f} ± {100 * se:.1f}" if acc is not None and se is not None else f"t{trial}: not recorded")
    return "; ".join(parts) if parts else "logs not read"


def _final_table(report: Mapping[str, Any]) -> str:
    lines = ["| model / view / arm | exit rung | exit | prompt strict (merged) % [CI] | prompt loose | inst strict | inst loose | at cap at exit | loop / truncated | "
             "at-cap passing prompt strict | mean / p99 completion tokens (pass-1 leg) | decode tok/s aggregate (pass-1 leg) | stock prompt strict ± stderr per trial (pass 1) | recorder | pins (pass 1; rungs) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in report["cells"].items():
        if not c.get("usable"):
            lines.append(f"| {k} | — | **{c.get('slot')}** | not measured | | | | | | | | | | | — |")
            continue
        fr = c.get("final_readout") or {}
        m = c["metrics"].get(READING_MERGED) or {}
        rec = c.get("recorder") or {}
        pins = c.get("pins_by_cap") or {}
        q = c.get("completion_tokens") or {}
        pin_s = "; ".join(f"{cap}: `{str(t)[:8]}`" for cap, t in pins.items())
        tok = f"{q['mean']:.0f} / {q['p99']}" if q.get("mean") is not None and q.get("p99") is not None else "—"
        tps = f"{c['decode_tok_s_aggregate']:.0f}" if c.get("decode_tok_s_aggregate") is not None else "—"
        lines.append(f"| {k} | {fr.get('exit_cap')} | {fr.get('exit')} | {_ci(m.get('prompt_strict'))} | {_ci(m.get('prompt_loose'))} | {_ci(m.get('inst_strict'))} | "
                     f"{_ci(m.get('inst_loose'))} | {fr.get('n_at_cap')} ({_pct(fr.get('share_at_cap'))}) | {fr.get('n_loop')} / {fr.get('n_truncated')} | "
                     f"{fr.get('n_at_cap_pass_prompt_strict')} of {fr.get('n_at_cap')} | {tok} | {tps} | {_stock_cell(report, k)} | "
                     f"{', '.join(sorted(set(str(v) for v in rec.values()))) or '—'} | {pin_s} |")
    return "\n".join(lines)


def _pins_table(report: Mapping[str, Any]) -> str:
    pins = (report.get("provenance") or {}).get("pins_by_cell") or {}
    lines = ["| cell | pass 1 | 4,096 | 8,192 |", "|---|---|---|---|"]
    for k, by in pins.items():
        lines.append(f"| {k} | `{str(by.get('2048'))[:8]}` | {('`' + str(by.get('4096'))[:8] + '`') if by.get('4096') else '—'} | {('`' + str(by.get('8192'))[:8] + '`') if by.get('8192') else '—'} |")
    return "\n".join(lines)


def _slots_table(report: Mapping[str, Any]) -> str:
    slots = report.get("slots") or {}
    if not slots:
        return "_none_"
    lines = ["| row | status | reason |", "|---|---|---|"]
    for k, v in slots.items():
        lines.append(f"| {k} | **{v['label']}** | {v.get('reason')} |")
    return "\n".join(lines)


def _recorder_statement(report: Mapping[str, Any]) -> str:
    rec = (report.get("provenance") or {}).get("recorder") or {}
    tally = rec.get("tally") or {}
    if rec.get("all_agree"):
        return (f"The resolved-sampler recorder's verdict is **{V.RECORDER_AGREES} on every one of the {rec.get('n_legs')} legs** (pass 1 and every rung): the "
                f"client class matched the pinned vendor-preset row and every probe kind agreed; nothing unclassified (tally {tally}).")
    return f"Recorder verdicts per leg: {tally} (n = {rec.get('n_legs')}); not every leg AGREES — see I2."


def _certifications_table(report: Mapping[str, Any]) -> str:
    certs = report.get("nemotron_certifications") or {}
    if not certs:
        return "_no certification doc given_"
    lines = ["| producer tree | geometry entry | admitted | attestation sha256 | source |", "|---|---|---|---|---|"]
    for tree, by in certs.items():
        for entry, rec in (by or {}).items():
            lines.append(f"| `{str(tree)[:8]}` | {entry} | {rec.get('admitted')} | `{str(rec.get('attestation_sha256'))[:12]}` | {rec.get('source')} |")
    return "\n".join(lines)


def _cost_lines(report: Mapping[str, Any]) -> str:
    cost = report["cost"]
    by_cap: dict[str, float] = {}
    for k, v in cost["legs"].items():
        cap = k.rsplit("@", 1)[-1]
        by_cap[cap] = round(by_cap.get(cap, 0.0) + v["gpu_h_actual"], 3)
    return (f"Actual GPU-hours (accepted generation span + the 250 s standup adder per leg): **{cost['total_gpu_h_actual']}** over {len(cost['legs'])} legs — "
            f"by rung {by_cap}; {cost['n_legs_on_fallback_basis']} leg(s) on the fallback basis. Central estimate: {cost['central_estimate_total_gpu_h']}.")


def _fmt(x: Any, pct: bool = True) -> str:
    if x is None:
        return "—"
    return f"{100 * x:.1f}" if pct else f"{x}"


def _shares_table(report: Mapping[str, Any]) -> str:
    lines = ["| cell | n draws | at cap (%) | truncated (%) | loop (%) | boundary | ladder exit | promoted |", "|---|---|---|---|---|---|---|---|"]
    for k, c in report["cells"].items():
        if not c.get("usable"):
            lines.append(f"| {k} | **{c.get('slot')}** — not measured ({str(c.get('slot_reason'))[:90]}) | | | | | | |")
            continue
        s = c["shares"]
        lines.append(f"| {k} | {s['n_draws']} | {_fmt(s['share_at_cap'])} | {_fmt(s['share_truncated'])} | {_fmt(s['share_loop'])} | {s.get('n_boundary')} | {s.get('exit')} | {c['ladder']['promoted'] or '—'} |")
    return "\n".join(lines)


def _contrasts_table(report: Mapping[str, Any]) -> str:
    lines = ["| cell | prompt strict (merged) pp [CI] | prompt loose | inst strict | inst loose | word |", "|---|---|---|---|---|---|"]
    for c in report["contrasts"]:
        if c["status"] != "filed":
            lines.append(f"| {c['cell_key']} | not run: {c.get('reason')} | | | | |")
            continue
        cells = []
        for m in METRICS:
            s = c["readings"][READING_MERGED][m]
            cells.append(f"{100 * s['point']:+.1f} [{100 * s['ci_low']:+.1f}, {100 * s['ci_high']:+.1f}]")
        lines.append(f"| {c['cell_key']} | " + " | ".join(cells) + f" | {c['verdict']['word']} |")
    for k, v in (report.get("slots") or {}).items():
        if v["cell"] != IC.CELL_NONE:
            lines.append(f"| {k} | **{v['label']}** — not measured ({str(v.get('reason'))[:90]}) | | | | |")
    return "\n".join(lines)


def _ladder_table(report: Mapping[str, Any]) -> str:
    lines = ["| cell | pass-1 at cap | exit | rungs | redraw outcomes | residual at exit (decision; the last store's own exit) |", "|---|---|---|---|---|---|"]
    for k, c in report["cells"].items():
        if not c.get("usable") or not c["ladder"]["promoted"]:
            continue
        res = c["ladder"]["residual"] or {}
        lines.append(f"| {k} | {c['shares']['n_at_cap']} ({_fmt(c['shares']['share_at_cap'])} %) | {c['shares']['exit']} | {c['ladder']['promoted']} | {c['redraw_outcomes']} | "
                     f"{res.get('n_at_cap')} at {res.get('final_cap')} ({res.get('final_exit')}; store {res.get('final_store_exit')}) |")
    return "\n".join(lines) if len(lines) > 2 else "_no cell promoted_"


def _ledger_table(report: Mapping[str, Any]) -> str:
    rec = report["gates"]["I1_ledger"].get("reconciliation") or {}
    lines = ["| leg | forwarded | draws | ledger at cap | recorded at cap | client timeout (s) |", "|---|---|---|---|---|---|"]
    for k, r in rec.items():
        lines.append(f"| {k} | {r['n_forwarded']} | {r['n_draws']} | {r['ledger_at_cap']} | {r['recorded_at_cap']} | {r['client_timeout_s']} |")
    return "\n".join(lines)


def _gates_table(report: Mapping[str, Any]) -> str:
    lines = ["| gate | status | detail |", "|---|---|---|"]
    for g in GATE_NAMES:
        rec = report["gates"][g]
        lines.append(f"| {g} | {rec['status']} | {(rec.get('detail') or '')[:160]} |")
    return "\n".join(lines)


def _merged_scope_cell(p: Mapping[str, Any]) -> str:
    """The MERGED reading's attestation of one filed pair, for the table."""
    ma = p.get("merged_attestation") or {}
    if not ma:
        return "—"
    if ma.get("trivially_identical"):
        return f"= pass 1 (no promotion on either view); {ma.get('n_final_body_identical')} of {ma.get('n_draws')} final request bodies identical"
    if ma.get("identical"):
        return f"byte-identical-request floor: every final draw shares bytes and cap ({ma.get('n_final_body_identical')} of {ma.get('n_draws')})"
    return (f"**variability of the whole adaptive protocol, not a byte-identical-request floor**: {ma.get('n_final_cap_differs')} of {ma.get('n_draws')} "
            f"final draws differ in cap ({ma.get('n_draws', 0) - ma.get('n_final_body_identical', 0)} in request bytes); promotions main "
            f"{(ma.get('promotions') or {}).get('main')} / it {(ma.get('promotions') or {}).get('it')}")


def _noise_table(report: Mapping[str, Any]) -> str:
    lines = ["| model | pair | label (pass-1 attestation: 2,048 request bodies) | merged reading | prompt strict (merged) it - main pp [CI] (main / it %) | inst strict | discordant draws (prompt strict, merged) |",
             "|---|---|---|---|---|---|---|"]
    for n in report["noise_floor"]:
        if not n.get("pairs"):
            lines.append(f"| {n['model_key']} | — | {n['status']} | | | | |")
            continue
        for p in n["pairs"]:
            if p["status"] != "filed":
                lines.append(f"| {n['model_key']} | {p['cell']}(it) - {p['cell']}(main) | {p['label']}: {p['status']} ({p.get('reason')}) | | | | |")
                continue
            s, i = p["readings"][READING_MERGED]["prompt_strict"], p["readings"][READING_MERGED]["inst_strict"]
            pd = p["per_draw"][READING_MERGED]["prompt_strict"]
            lines.append(f"| {n['model_key']} | {p['cell']}(it) - {p['cell']}(main) | {p['label']}"
                         + (f" ({', '.join(p['attestation']['failing'])})" if p["label"] == LABEL_MEASURES else "")
                         + f" | {_merged_scope_cell(p)}"
                         + f" | {100 * s['point']:+.2f} [{100 * s['ci_low']:+.2f}, {100 * s['ci_high']:+.2f}] "
                         f"({100 * s['legs']['main']['point']:.2f} / {100 * s['legs']['it']['point']:.2f}) | "
                         f"{100 * i['point']:+.2f} [{100 * i['ci_low']:+.2f}, {100 * i['ci_high']:+.2f}] | "
                         f"{pd['n_discordant_draws']} of {pd['n_draws']} (main-only {pd['classes'][CLASS_MAIN_ONLY]}, it-only {pd['classes'][CLASS_IT_ONLY]}) |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- the CLI
def print_table(report: Mapping[str, Any]) -> None:
    print(f"IFEval canonical battery @ {report['sha']}: {report['verdict']['n_usable_cells']} usable cells, "
          f"{report['verdict']['n_promoted_cells']} promoted, {report['contrast_summary']['n_filed']}/{report['contrast_summary']['n_designed']} contrasts filed, "
          f"words {report['contrast_summary']['words']}; gates {report['gates']['n_passed']}/{len(GATE_NAMES)} passed, failed {report['gates']['failed']}")
    for c in report["contrasts"]:
        if c["status"] == "filed":
            s = c["readings"][READING_MERGED]["prompt_strict"]
            print(f"  {c['cell_key']:70s} prompt_strict {100 * s['point']:+6.1f} pp [{100 * s['ci_low']:+6.1f}, {100 * s['ci_high']:+6.1f}]  {c['verdict']['word']}")


def gate_report(path: Path) -> int:
    """Report-only certification (``--gate-report``): every refusal of :func:`production_refusals` — every gate PRESENT with a
    recognised status, the gates' statuses, AND the evidence flags each log-derived PASS rests on — plus one
    more for each claim of the file's STORED verdict (``production``, ``eval_logs_validated``) that the gates, re-derived here by
    :func:`verdict_of`, do not support. The stored verdict is a claim, never the evidence."""
    report = json.loads(Path(path).read_text(encoding="utf-8"))
    refusals = production_refusals(report)
    stored = report.get("verdict") if isinstance(report, Mapping) else None
    stored = stored if isinstance(stored, Mapping) else {}
    derived = verdict_of(report)
    if bool(stored.get("production")) and not derived["production"]:
        refusals.append("the stored verdict claims production: true, but re-derived from the gates and their evidence flags it is false — "
                        "the file's verdict is not trusted")
    if bool(stored.get("eval_logs_validated")) and not derived["eval_logs_validated"]:
        refusals.append("the stored verdict claims eval_logs_validated: true, but re-derived from the gates and their evidence flags it is "
                        "false — the file's verdict is not trusted")
    for r in refusals:
        print(f"REFUSED: {r}", file=sys.stderr)
    return EXIT_NOT_PRODUCTION if refusals else 0


REPORT_MATCH_META_KEYS: tuple[str, ...] = ("I9_report_match", "n_passed", "pending", "pending_unwaivable")


def comparable_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """``report`` with the leaves that describe a COMPARISON removed, so two reports over the same stores compare equal
    whether or not either of them was itself held to a third: the ``report_match`` block (already in
    :data:`T.REPORT_DIFF_SKIP`), the I9 gate (whose ``shared_numeric`` count is a numeric leaf that exists only after a
    match ran) and the gate tallies that move with I9's status. Every RESULT leaf — shares, metrics, CIs, contrasts, ladder
    counts, costs, the other nine gates — stays in the comparison."""
    out = json.loads(json.dumps(report, default=str))
    out.pop("report_match", None)
    # the comparison-only pass's own description of itself: a mode block, never a result leaf
    out.pop("comparison_only", None)
    gates = out.get("gates")
    if isinstance(gates, dict):
        for k in REPORT_MATCH_META_KEYS:
            gates.pop(k, None)
        # I5's text half and its per-leg counts exist only where the .eval logs were read (the same condition as I4's
        # `eval_check`, which `T.REPORT_DIFF_SKIP` already leaves out): a venv without inspect_ai reads neither
        for k in ("texts_reclassified", "reclassification", "n_at_cap_reclassified"):
            (gates.get("I5_at_cap") or {}).pop(k, None)
    for leg in (out.get("legs") or {}).values():
        if isinstance(leg, dict):
            leg.pop("at_cap_text", None)
    return out


def report_match(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, Any]:
    return T.numeric_leaf_diff(comparable_report(a), comparable_report(b))


def diff_reports(a: Path, b: Path) -> int:
    match = report_match(json.loads(Path(a).read_text()), json.loads(Path(b).read_text()))
    print(json.dumps({k: (v if not isinstance(v, list) else v[:20]) for k, v in match.items()}, indent=1))
    return 0 if match["equal"] else EXIT_REPORTS_DIFFER


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-root", type=Path, action="append", default=None,
                    help="a pulled wave root (repeatable: the battery's rows sit under several pulls; each row's root is resolved by its admitted pin)")
    ap.add_argument("--sha", default=None, help="the record label (the W&B run, the sidecars) — and, without --store-pins, the ONE store pin every producer tree carries")
    ap.add_argument("--store-pins", type=Path, default=None,
                    help="JSON {'<model>/<view>/<cell>@<cap>': '<tree sha>'}: every leg admitted by ITS OWN producer tree (per-row admission)")
    ap.add_argument("--nemotron-certifications", type=Path, default=None,
                    help="JSON {'<tree>': {'<geometry entry>': {'admitted': true, 'attestation_sha256': ...}}}: the Nemotron (tree, entry) certifications the launch admitted")
    ap.add_argument("--matrix", type=Path, default=_HERE.parent / IC.IFEVAL_MATRIX_REL)
    ap.add_argument("--ids", default=None, help="the canonical 541 ids (a JSON list); absent -> the population gate is PENDING")
    ap.add_argument("--pins", type=Path, default=None)
    ap.add_argument("--expected-refusals", type=Path, default=None, help="JSON {leg key: refusal code}")
    ap.add_argument("--gate-report", type=Path, default=None, metavar="REPORT")
    ap.add_argument("--diff-reports", type=Path, nargs=2, default=None, metavar=("A", "B"))
    ap.add_argument("--require-report-match", type=Path, default=None, metavar="REPORT")
    ap.add_argument("--analysis-sha", default=None)
    ap.add_argument("--bootstrap", type=int, default=pr.DEFAULT_BOOTSTRAP, metavar="B")
    ap.add_argument("--seed", type=int, default=pr.DEFAULT_BOOTSTRAP_SEED)
    ap.add_argument("--ci", type=float, default=pr.DEFAULT_CI)
    ap.add_argument("--allow-pending-gates", action="store_true",
                    help="waive a PENDING gate that is not log-derived (e.g. I7 without the ids file); NEVER waives I4 / I5 with the logs unread")
    ap.add_argument("--no-eval-logs", action="store_true",
                    help="do not read the .eval logs — I4 / I5 then read PENDING (never waivable) and the report is not production; permitted "
                         "ONLY with --compare-only (a venv without inspect_ai running the two-venv match)")
    ap.add_argument("--compare-only", action="store_true",
                    help="a COMPARISON-ONLY pass (review r3 S1): requires --require-report-match; claims no I4 / I5 pass and no production; "
                         "renders no record, makes no figure, mirrors nothing, logs nothing to W&B; exit 0 iff the numeric leaves match")
    ap.add_argument("--figure-exclude-family", action="append", default=[], help="a cell family / model family to leave out of the figures (repeatable)")
    ap.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-name-suffix", default="")
    ap.add_argument("--figs-dir", type=Path, default=None)
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--mirror-dir", type=Path, default=Path(MIRROR_DIR_DEFAULT))
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--results-md", type=Path, default=None,
                    help=f"render the immutable template {RESULTS_TEMPLATE.name}'s placeholders from the report into this file "
                         f"(the documented record is {RESULTS_RECORD_REL}; the template path itself is refused)")
    ap.add_argument("--out", type=Path, default=None)
    return ap


def comparison_only_argument_problem(a: argparse.Namespace) -> str | None:
    """The comparison-only contract at the command line: ``--no-eval-logs`` ONLY inside ``--compare-only``;
    ``--compare-only`` ONLY with ``--require-report-match``; and a comparison-only pass names no output it is not allowed to
    make (a record, a figures dir, a W&B run name or suffix, a mirror dir other than the default it will not use)."""
    if a.no_eval_logs and not a.compare_only:
        return ("--no-eval-logs is permitted only with --compare-only: a pass that does not read the .eval logs cannot certify I4 / I5 or "
                "production, so it may only run as the two-venv comparison (--compare-only --require-report-match REPORT)")
    if a.compare_only:
        if a.require_report_match is None:
            return "--compare-only requires --require-report-match REPORT (a comparison-only pass exists to match a production report)"
        named = [f for f, v in (("--results-md", a.results_md), ("--figs-dir", a.figs_dir), ("--wandb-name", a.wandb_name),
                                ("--wandb-name-suffix", a.wandb_name_suffix or None)) if v]
        if a.wandb_mode != "disabled" and a.wandb_mode != build_parser().get_default("wandb_mode"):
            named.append("--wandb-mode")
        if named:
            return f"--compare-only publishes nothing (no record, no figures, no mirror, no W&B run): drop {', '.join(named)}"
    return None


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.gate_report is not None:
        return gate_report(a.gate_report)
    if a.diff_reports is not None:
        return diff_reports(*a.diff_reports)
    if a.gen_root is None or a.sha is None:
        raise SystemExit("--gen-root and --sha are required")
    problem = comparison_only_argument_problem(a)
    if problem:
        ap.error(problem)
    report = analyze(a.gen_root, a.sha, matrix=a.matrix, canonical_ids=a.ids, pins=a.pins, expected_refusals=a.expected_refusals,
                     n_resamples=a.bootstrap, seed=a.seed, ci=a.ci, allow_pending_gates=a.allow_pending_gates, read_eval_logs=not a.no_eval_logs,
                     analysis_sha=a.analysis_sha, exclude_families=a.figure_exclude_family, store_pins=a.store_pins,
                     nemotron_certifications=a.nemotron_certifications)
    if a.compare_only:
        # the pass says what it is BEFORE any verdict is read: no I4 / I5 pass, no production, nothing published
        report["comparison_only"] = {"mode": COMPARISON_ONLY_MODE, "eval_logs_read": not a.no_eval_logs,
                                     "reference_report": str(a.require_report_match),
                                     "claims": ("no I4 / I5 pass with the logs unread (PENDING, never waivable); no production; no record, figure, "
                                                "mirror or W&B run from this report — its one output is the I9 numeric-leaf match against the "
                                                "reference report")}
        report["verdict"] = verdict_of(report)
    print_table(report)
    out = a.out or (Path(a.gen_root[0]) / "ifeval_report.json")

    def write():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")

    # the production boundary: every refusal for a production pass; for a comparison-only pass everything EXCEPT the unread
    # logs and its own mode (those are what it is) — a failed gate, an unusable leg or an unfiled contrast still refuses
    refusals = refusals_beyond_eval_logs(report) if a.compare_only else production_refusals(report)
    if refusals:
        write()
        for r in refusals:
            print(f"REFUSED: {r}", file=sys.stderr)
        return EXIT_NOT_PRODUCTION
    if a.require_report_match is not None:
        other = json.loads(Path(a.require_report_match).read_text(encoding="utf-8"))
        match = report_match(other, report)
        report["report_match"] = match
        report["gates"]["I9_report_match"] = gate_report_match(match)
        refresh_gate_aggregates(report["gates"])
        report["verdict"] = verdict_of(report)
        if a.compare_only:
            report["comparison_only"]["reference_production"] = bool((other.get("verdict") or {}).get("production"))
            report["comparison_only"]["reference_eval_logs_validated"] = bool((other.get("verdict") or {}).get("eval_logs_validated"))
        if not match["equal"]:
            write()
            print(f"REFUSED: the two reports differ on {len(match['differing'])} numeric leaves", file=sys.stderr)
            return EXIT_REPORTS_DIFFER
    if a.compare_only:
        # the match is the whole output: the report is written with I9's verdict, production False, nothing published
        write()
        print(f"{COMPARISON_ONLY_MODE}: I9 {report['gates']['I9_report_match']['status']} — {report['gates']['I9_report_match'].get('detail')}; "
              f"production {report['verdict']['production']} (claims none)", file=sys.stderr)
        return 0
    figs: dict[str, Any] = {}
    if not a.no_figs:
        figs = make_figures(report, a.figs_dir or (out.parent / "figs"))
        report["figures"] = figs
        if not a.no_mirror:
            report["mirrored"] = T.mirror_figures(Path(figs["numbers"]).parent, a.mirror_dir, manifest=figs)
        report["wandb_url"] = log_wandb(report, figs, mode=a.wandb_mode, name=a.wandb_name, name_suffix=a.wandb_name_suffix)
    if a.results_md is not None:
        # the aliasing refusal happens BEFORE the output directory is made or anything written
        rendered = render_results_md(report, out=a.results_md)
        a.results_md.parent.mkdir(parents=True, exist_ok=True)
        a.results_md.write_text(rendered, encoding="utf-8")
    write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
