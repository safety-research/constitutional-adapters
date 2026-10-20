"""Read out the GSM8K paired-200 thinking-ON experiment on the paper's objects and file the verdicts.

Forty-two designed cells on four models under ONE instrument — inspect-ai's stock ``gsm8k`` task
narrowed to the 200 Platinum-paired items (``inspect/gsm8k_paired200``), thinking ON, k = 1, one
pass at an explicit 8,192 cap, each model's thinking-card row, on the served path — 38 launched and
4 HELD, plus the standing ladder: a cell over the trigger at cap (1 %, harmonised 2026-09-12) re-draws its ELIGIBLE truncating items
at 16,384 into a sibling root, and its reading is the ladder-MERGED one with the 8,192-only reading
beside it. Per (model, view) every cell is contrasted against that view's ``none`` with paired
bootstrap CIs and exact McNemar under the think-block branches AND the residual-at-cap axis (stock /
W wrong / X excluded); the word is filed only where W and X agree and the branches agree. The two
``none`` cells of gemma and qwen are the wave's noise-floor pair; the overlap check's record (two
instruments, never pooled) is read and gated.

    PYTHONPATH=src python scripts/tinycap_gsm8k_paired200.py --gen-root <local pull of the wave root>[=PIN] ...
        --sha <WAVE_SHA7> --matrix configs/sdfbase/gsm8k/matrix.json --ids <canonical ids json>
        [--also-sha SHA40 ... --tree-diff-allow GLOB ... | --tree-diff-allow-file REVIEW_MD --tree-set-reason TEXT]
        [--waves WAVE[,WAVE]] [--peer-report REPORT[=PIN] ...]
        [--overlap overlap.json] [--sampling-rows YAML] [--pins JSON] [--expected-refusals JSON]
        [--wandb-mode online|offline|disabled] [--figs-dir DIR] [--out report.json]
        [--figure-exclude-family MODEL_KEY ...] [--figure-exclude-arm CELL ...] [--figure-modes MODE[,MODE]]
        [--mirror-dir DIR | --no-mirror] [--no-eval-logs] [--require-report-match REPORT]
        [--peer-report REPORT ...] [--figure-exclude-family MODEL_KEY ...]

A matrix row's ARM (``thinking_mode``: ``on`` on every
published row, ``not_available`` on a family with no thinking switch) selects the reader's class for every gate — the
ledger's ``thinking`` null, no switch on the wire, the no-switch certification class, the ``none`` split with its
PRIMARY fail-closed stray-tag rule — and every figure / table row of such a leg is labelled "thinking: not available";
such legs are held out of any pooled cross-family mean. ``--peer-report`` (repeatable) assembles the four-family figure
set (``*_four_family`` stems) from this report and the published wave's already-gated report(s), each column at its own
pin, the hold-out footer on every such figure. ``--figure-exclude-family`` (repeatable; a model key of the design — or of a peer's, on an assembly)
drops a family from the figures and the figure-facing tables ONLY — the report's data is untouched, no aggregate is
recomputed, and the exclusion is recorded as ``figures.excluded_families`` (the main line's implementation; GLM is dropped
from the paper figures this way — decisions-impl I23).

The EIGHT-ARM extension: store admission over several producer trees (``--gen-root PATH=PIN`` repeatable through the
MATH-500 module's ``GenRoots``; ``--also-sha`` + the reviewed allowlist through its ``verify_tree_set``), the three
cross-tree compatibility relations replacing the one-sha rule under a DECLARED tree set, the REQUIRED-EVIDENCE set
(``--waves``; ``deferred`` / ``held`` rows are design statuses), the pooled-by-arm estimands over the ``main`` view as
report fields, the eight-arm figures, the validated presentation selectors, the publication gate and the Llama
``--peer-report``. Every new block that carries a NUMBER is written only in eight-arm mode (``--waves``, a matrix
carrying waves / a deferred status, a pinned or repeated gen root, a peer report), so the published 1 % report
reproduces leaf for leaf through these code paths.

Every store validator is PARAMETERIZED by the matrix row (``scripts/tinycap_gsm8k_validators.py``);
the genuinely task-agnostic statistics and plumbing of the MATH-500 analysis
(``scripts/tinycap_math500_thinking.py`` and its parent ``tinycap_oos_transfer.py``) are imported by
module and never copied or edited. Three checks are RETAINED in kind: the canonical population
(every 8,192 leg's attempted ids == the 200 ids derived once, independently of any store), producer
consistency (one tree sha across every comparison's legs) and the authoritative per-item score
re-derivation from the retained ``.eval`` log. The production boundary fails CLOSED before any
figure, mirror or W&B call (exit 3), and the two-venv match refuses to publish on a differing numeric
leaf (exit 4).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from sisteer.capeval.tiers import cap_ladder as CL
from sisteer.capeval.tiers import inspect_rung as IR
from sisteer.capeval.tiers import pairing as pr
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import paper_rows as PRW
from sisteer.capeval.tinycap import refusal_shaped as RFS   # not `RS`: the MATH-500 module this file imports binds RS to its resolved-sampler recorder
from sisteer.capeval.tinycap.completion import CompletionError

_HERE = Path(__file__).resolve().parent


def _load_module(name: str, path: Path) -> Any:
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


V = _load_module("tinycap_gsm8k_validators", _HERE / "tinycap_gsm8k_validators.py")
M = V.M
T = V.T
PG = M.PG
AnalysisError = V.AnalysisError
_mapping, _iter_jsonl, _read_json = T._mapping, T._iter_jsonl, T._read_json

# --------------------------------------------------------------------------- #
# vocabulary                                                                   #
# --------------------------------------------------------------------------- #

EXPERIMENT = V.EXPERIMENT
TASK, TASK_NAME, N_ITEMS = V.TASK, V.TASK_NAME, V.N_ITEMS
CAP_BASE, CAP_RUNG = V.CAP_BASE, V.CAP_RUNG
MODE_PRIMARY = V.MODE_PRIMARY
CELL_NONE = V.CELL_NONE
VIEW_MAIN, VIEW_IT = V.VIEW_MAIN, V.VIEW_IT
STATUS_LAUNCH, STATUS_COMPLETE, STATUS_HELD, STATUS_DEFERRED = V.STATUS_LAUNCH, V.STATUS_COMPLETE, V.STATUS_HELD, V.STATUS_DEFERRED
EVIDENCE_STATUSES, WAVE_PUBLISHED = V.EVIDENCE_STATUSES, V.WAVE_PUBLISHED
#: the eight-arm registry (the MATH-500 module's, by import — one panel order across both instruments): none · prompted
#: safety · prompted constitution · LoRA ctl2 · LoRA mix · LoRA mix−ctl · AlphaSteer · Arditi; every other cell (the four
#: PSR cells, `safety_refusal__lora_r4`) is an APPENDIX arm — in the report and the tables, never pooled, out of the figures
ARMS, ARM_BY_CELL, ARM_BY_KIND, ARM_KINDS = M.ARMS, M.ARM_BY_CELL, M.ARM_BY_KIND, M.ARM_KINDS
ARM_KIND_APPENDIX = "appendix"
#: the pooled panel (the MATH-500 module's constants): equal weights over gemma / qwen / nemotron; llama
#: HELD OUT (thinking: not available); GLM not in the panel (appendix tables)
POOLED_PANEL_MODELS, POOLED_HELD_OUT, POOLED_NOT_IN_PANEL = M.POOLED_PANEL_MODELS, M.POOLED_HELD_OUT, M.POOLED_NOT_IN_PANEL
POOLED_AVAILABLE, POOLED_UNAVAILABLE = M.POOLED_AVAILABLE, M.POOLED_UNAVAILABLE
#: the eight-arm figures' model groups, in the plan's order: the pooled panel, then its hold-out (GLM is not drawn: Q1)
EIGHTARM_MODEL_ORDER: tuple[str, ...] = (*POOLED_PANEL_MODELS, *POOLED_HELD_OUT)
#: the trigger provenance sentence every report states — inspect_rung's, by import, never re-typed (plan P10)
TRIGGER_PROVENANCE: str = V.GSM8K_TRIGGER_PROVENANCE
PEER_MODEL_LLAMA = "llama8b"
PEER_STATUS_PRESENT, PEER_STATUS_ABSENT = "present", "absent"
PEER_PENDING_LABEL = "pending — Llama lane"
TREE_SET_KEY = M.TREE_SET_KEY
READING_R8192, READING_MERGED = "r8192", "merged"
READINGS: tuple[str, ...] = (READING_R8192, READING_MERGED)
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
SOURCE_R8192, SOURCE_RUNG = "r8192", "rung16384"
#: the residual-at-cap axis (plan P1-5): the stock value as filed, W every residual item WRONG, X every residual item EXCLUDED
AXIS_STOCK, AXIS_W, AXIS_X = "stock", "W", "X"
AXES: tuple[str, ...] = (AXIS_STOCK, AXIS_W, AXIS_X)
AXIS_NAMES: Mapping[str, str] = {AXIS_STOCK: "stock score as filed", AXIS_W: "residual at-cap scored wrong",
                                 AXIS_X: "residual at-cap excluded from both legs"}
BRANCH_PRIMARY, BRANCH_SCORE_AS_IS, BRANCHES, BRANCH_NAMES = T.BRANCH_PRIMARY, T.BRANCH_SCORE_AS_IS, T.BRANCHES, T.BRANCH_NAMES
#: the READER's policy for a draw that FINISHED (stop, below the cap) inside an unterminated think block. The store scores such
#: a draw UNSCORED (``C.UNTERMINATED_UNSCORED`` — the producer's policy, held by gate A4 and untouched here). ``unscored``
#: (the default): the store's scoring stands — the published behaviour, byte for byte (no field below is written). ``failure``
#: (the rule of 2026-09-13 on the GSM8K paired-200 artifact: an answer the model never gave is a failure of the
#: object): the draw is re-scored WRONG under BOTH think-block branches, so the conditional and score-as-is readings coincide
#: on it; an AT-CAP unterminated draw (case b) is NOT touched — the ladder and the two branches treat it as before. Named
#: wherever it acts: ``design``, each reading (with its rescored count), the estimand sentence, every paper row of an
#: affected reading (in ``source``), and the flag on each rescored item. The rung redraw records keep the store's
#: classification (a redraw's outcome is the store's fact; no rung of the published wave carries such a draw).
UNTERMINATED_FINISHED_UNSCORED, UNTERMINATED_FINISHED_FAILURE = "unscored", "failure"
UNTERMINATED_FINISHED_POLICIES: tuple[str, ...] = (UNTERMINATED_FINISHED_UNSCORED, UNTERMINATED_FINISHED_FAILURE)
UNTERMINATED_FINISHED_POLICY_KEY = "unterminated_finished_policy"
N_UNTERMINATED_FINISHED_RESCORED_KEY = "n_unterminated_finished_rescored"
ITEM_RESCORED_UNTERMINATED_FINISHED_KEY = "rescored_unterminated_finished"
UNTERMINATED_FINISHED_FAILURE_TEXT = (" Reader policy `failure` for unterminated-but-FINISHED draws (the user's ruling of 2026-09-13): a draw "
                                      "that ended the turn below the cap inside an unterminated think block is scored WRONG under BOTH think-block "
                                      "branches — an answer the model never gave is a failure of the object — so on such draws the primary and "
                                      "score-as-is branches coincide; at-cap unterminated draws keep the two-branch treatment above.")
GATE_PASS, GATE_FAIL, GATE_PENDING = T.GATE_PASS, T.GATE_FAIL, T.GATE_PENDING
GATE_NAMES: tuple[str, ...] = ("A0_completeness", "A1_ledger", "A2_identity", "A3_certification", "A4_scoring_path",
                               "A5_at_cap", "A6_ladder", "A7_population", "A8_two_branch", "A9_report_match", "A10_overlap")
WORD_LOWER, WORD_HIGHER, WORD_NOT_RESOLVED = "lower than none", "higher than none", M.WORD_NOT_RESOLVED
WORDS: tuple[str, ...] = (WORD_LOWER, WORD_HIGHER)
_WORD_MAP: Mapping[str, str] = {M.WORD_LOWER: WORD_LOWER, M.WORD_HIGHER: WORD_HIGHER, M.WORD_NOT_RESOLVED: WORD_NOT_RESOLVED}
STAT_FIELDS = M.STAT_FIELDS
NOT_MEASURABLE = T.NOT_MEASURABLE
#: THE NAMED OUTCOME of a cell whose thinking-ON render the DRIVER refused to certify (predicate (A),
#: ``contracts.render_certified`` through ``served_capability_run._render_certification``): the driver writes
#: ``evidence.json`` with ``render_certified.render_certified: false`` and every problem named, then aborts before any
#: generation, so the store has a manifest and NO summary. Such a leg is an ADMITTED, NAMED refusal (``refused_certified``,
#: counted beside the operator's ``refused_expected``), its contrast not computable exactly like an expected refusal,
#: and the wave's verdict proceeds — the honest outcome for gsm8k-paired200 row 27 (nemotron `constv2-mix-full__psr_ce_all`:
#: every ON draw ended inside the prompt-opened block without `</think>`; in production the draws would be unterminated
#: and unscored, so the cell has no result either way). A leg with no summary and NO such evidence stays ``missing``;
#: so does one whose refusal fails the identity / producer / binding / non-behavioural checks — the
#: report states what the draws did, on which route; the causal reading (object vs bare) is the results document's.
REFUSAL_CERTIFIED_UNSERVABLE = C.REC_THINKING_UNSERVABLE
LEG_STATUS_REFUSED_CERTIFIED = "refused_certified"
LEG_STATUS_REFUSED_EXPECTED = "refused_expected"
#: the driver's render-probe draw class for "finished inside the prompt-opened block without closing it"
#: (``served_capability_run.PROBE_CLASS_UNCLOSED_BLOCK``; pinned equal by test — the analysis is torch-free and does
#: not import the driver)
PROBE_CLASS_UNCLOSED_BLOCK = "unclosed_block"
CONTRAST_VS_NONE = "cell_minus_none"
CONTRAST_NOISE = "none_it_minus_none_main"
CONTRAST_TRUNCATION = "merged_minus_r8192"
#: the W&B purpose: the project is sisteer.wandb_entity.wandb_project(WANDB_PURPOSE) (the site's SISTEER_WANDB_PROJECT_TINYCAP first)
WANDB_PURPOSE = "tinycap"
WANDB_RUN_PREFIX = "gsm8k-paired200-"
MIRROR_DIR_DEFAULT = "~/figs/gsm8k-paired200-paper-objects"
NUMBERS_FILENAME = "final_g8k_figures_numbers.json"
TABLES_FILENAME = "final_g8k_tables.json"
#: ``figures.excluded_families`` — the model families the FIGURES and the rendered tables leave out (``--figure-exclude-family``,
#: repeatable; GLM is out of the paper's figures and headline tables while its legs stay in the report's data). The report's
#: ``cells`` / ``legs`` / ``contrasts`` / gates keep every leg and no numeric leaf moves; the same key is recorded in the
#: numbers sidecar and the tables JSON.
FIGURES_EXCLUDED_KEY = "excluded_families"
#: beside it, only when a family is excluded: the families the figures DO show, and the stem prefix every figure file of this analysis carries
FAMILIES_SHOWN_KEY = "families_shown"
STEM_PREFIX = "final_g8k_"
#: the tables' note on aggregates under an exclusion (review of 2026-09-11: never a silent substitution)
AGGREGATES_NOTE = ("the unsuffixed aggregates (n_cells_read, max_share, over_1pct, triggered, escalated, terminal_no_rung) are the WAVE's, "
                   "unchanged by the figure exclusion and duplicated under *_report; the *_shown variants are over the families shown")
#: the trigger's ONE spelling in prose, derived from the constant (never re-typed): "1 %"; the findings marker top_items_of filters on
TRIGGER_PCT = V.trigger_pct_label()
TRIGGER_FINDING_MARK = f"> the {TRIGGER_PCT} trigger"
EXIT_NOT_PRODUCTION, EXIT_REPORTS_DIFFER = T.EXIT_NOT_PRODUCTION, T.EXIT_REPORTS_DIFFER
#: the report block of the per-row store-pins admission (--store-pins)
STORE_PINS_KEY = "store_pins"
REPORT_DIFF_SKIP = T.REPORT_DIFF_SKIP
#: the by-design leaves (MATH's shapes, reused): the reported loop split under `at_cap` and `truncation.cells[*]`
REPORT_DIFF_SKIP_SPLIT_FIELDS = M.REPORT_DIFF_SKIP_SPLIT_FIELDS
numeric_leaf_diff = M.numeric_leaf_diff
mirror_figures = M.mirror_figures
report_path = T.report_path
#: the loop rule that DECIDES eligibility — Evals' degeneracy rule (cap_ladder); the MATH periodic-block rule is a second column
LOOP_RULE_TEXT = CL.DEGENERATE_RULE
PERIODIC_RULE_TEXT = M.LOOP_RULE_TEXT
SPLIT_SOURCE_EVAL_LOG, SPLIT_SOURCE_STORE_ROWS = M.SPLIT_SOURCE_EVAL_LOG, M.SPLIT_SOURCE_STORE_ROWS

#: the arm labels (the validators' one table; the comparability rule: "thinking: not available", never "OFF")
THINKING_ARM_ON, THINKING_ARM_NOT_AVAILABLE = V.THINKING_ARM_ON, V.THINKING_ARM_NOT_AVAILABLE
thinking_label = V.thinking_label


def _pct(x: float) -> str:
    """``0.03 -> "3 %"``; ``0.01 -> "1 %"`` — the report's spelling of a share threshold."""
    return f"{100 * float(x):g} %"


#: THE TRIGGER PROVENANCE: every report SAYS which trigger produced its ladder. The instrument's sentence is the adapter's
#: (`inspect_rung.GSM8K_TRIGGER_PROVENANCE`: the harmonised 1 % RAW at-cap — one trigger across the battery;
#: `TRIGGER_PROVENANCE`). The published wave's FIRST analysis ran the 3 % exception of 2026-09-10 (the retired pair,
#: `inspect_rung.GSM8K_TRIGGER_THRESHOLD_3PCT_RETIRED`), and a report filed under it says so (`TRIGGER_PROVENANCE_3PCT`). The
#: text is a FUNCTION of the trigger (never re-typed), pinned by test to its exact words for both. It rides the `truncation`
#: block, the at-cap figure's caption / footer and its table header; a four-family assembly whose columns ran different
#: triggers files it PER MODEL.
TRIGGER_PROVENANCE = V.GSM8K_TRIGGER_PROVENANCE
TRIGGER_PROVENANCE_3PCT = (f"escalation trigger {_pct(IR.GSM8K_TRIGGER_THRESHOLD_3PCT_RETIRED)} at cap, per the user-approved GSM8K amendment of 2026-09-10 "
                           f"(battery standing rule {_pct(V.GSM8K_REPORT_LINE)}; the {_pct(V.GSM8K_REPORT_LINE)} line filed as a report field)")


def _close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) < 1e-9


STANDING_RULE_LINE = {"line": V.GSM8K_REPORT_LINE,
                      "text": (f"battery standing rule: escalate the truncating subset of any cell over {_pct(V.GSM8K_REPORT_LINE)} at cap "
                               f"(inspect_rung.GSM8K_REPORT_LINE) — the instrument's trigger since the user decision of 2026-09-12 "
                               f"(inspect_rung.GSM8K_TRIGGER_THRESHOLD); the published wave's first analysis ran the "
                               f"{_pct(IR.GSM8K_TRIGGER_THRESHOLD_3PCT_RETIRED)} exception of 2026-09-10, and a rung planned under that pair is "
                               f"admitted on its evidence (inspect_rung.GSM8K_TRIGGER_RULES_ADMITTED)")}
PROVENANCE_PER_MODEL = "per model — see four_family.trigger_provenance_by_model"
#: the hold-out: a family with no thinking switch is a DIFFERENT INSTRUMENT — labelled, contrasted within itself, and never
#: pooled with the thinking families; the footer every pooled / four-family figure carries
HOLD_OUT_REASON = ("no thinking switch — a different instrument (thinking: not available); within-model contrasts only; "
                   "held out of any pooled cross-family mean")
def _arm_design(arm_by_model: Mapping[str, str]) -> dict[str, Any]:
    """The design block's four ARM keys from one ``{model_key: arm}`` map: the arm per model, its label, the
    no-switch models and the hold-out reason for each (a family with no thinking switch is a different instrument, never
    pooled). Written by :func:`analyze` for one report and by :func:`combine_reports` for the four-family assembly."""
    no_switch = [m for m, a in arm_by_model.items() if a == THINKING_ARM_NOT_AVAILABLE]
    return {"thinking_mode_by_model": dict(arm_by_model),
            "thinking_label_by_model": {m: thinking_label(a) for m, a in arm_by_model.items()},
            "no_switch_models": no_switch,
            "held_out_of_pooled_means": {m: HOLD_OUT_REASON for m in no_switch}}

ESTIMAND_TEXT = ("per (model, view): the paired within-model contrast acc(cell) - acc(none) over the ONE common set of 200 "
                 "GSM8K paired items (k = 1, thinking ON), conditional accuracy with each cell's OWN scored denominator; B "
                 "replicates, one index draw per replicate applied to both legs; percentile CIs; sharp censoring bounds from "
                 "leg-wise adverse imputation; two axes reported side by side and never folded — the think-block policy "
                 "(primary: an unterminated block -> unscored; score_as_is: -> wrong) and the residual-at-cap treatment "
                 "(stock: the value as filed; W: every residual at-cap item wrong; X: every residual at-cap item of either leg "
                 "excluded from both). Two readings per cell: the 8,192-only draws and the ladder-MERGED reading (the 16,384 "
                 "redraw for each eligible truncating item, the 8,192 draw for every other); the merged reading is the headline. "
                 "The word is filed only where W and X agree and both think-block branches agree.")


# --------------------------------------------------------------------------- #
# inputs                                                                       #
# --------------------------------------------------------------------------- #


class Context:
    """Everything a leg's reading is held to: the sampling rows, the render
    facts, the served model pin per TAG, the store pin, the expected refusals."""

    def __init__(self, *, sampling_rows: Mapping[str, Any], render_facts: Mapping[str, Mapping[str, Any]],
                 model_pins: Mapping[str, tuple[str, str]], store_pin: str, expected_refusals: Mapping[str, str],
                 also_trees: Sequence[str] = (), declared: bool | None = None, arm_contracts: bool = False,
                 unterminated_finished_policy: str = UNTERMINATED_FINISHED_UNSCORED,
                 store_pins: Mapping[str, str] | None = None,
                 object_generations: Mapping[str, Mapping[str, Any]] | None = None) -> None:
        problem = M.pin_problem(store_pin)
        if problem:
            raise AnalysisError(problem)
        #: E3 tier-2 item 5: ``{object sha256: {generation, effective_batch, source}}`` — the fit-provenance bindings a fitted row's
        #: ``object_generation`` stamp is READ from (V.object_generation_bindings); empty = no record binds any object (every fitted
        #: cell stays unstamped, the export's ``n/a``)
        self.object_generations: dict[str, dict[str, Any]] = {str(k): dict(v) for k, v in (object_generations or {}).items()}
        if unterminated_finished_policy not in UNTERMINATED_FINISHED_POLICIES:
            raise AnalysisError(f"{UNTERMINATED_FINISHED_POLICY_KEY} {unterminated_finished_policy!r} outside {UNTERMINATED_FINISHED_POLICIES}")
        #: the reader's policy for unterminated-but-finished draws (module constants above); the default writes nothing
        self.unterminated_finished_policy = str(unterminated_finished_policy)
        for sha in also_trees:
            if not isinstance(sha, str) or not _HEX40.match(sha):
                raise AnalysisError(f"declared producer tree {sha!r} is not a 40-hex commit")
        self.store_pin = str(store_pin)
        #: the further producer trees a bare gen root's legs may be from (a pinned root binds its own)
        self.also_trees: tuple[str, ...] = tuple(also_trees)
        #: per-ROW store pins (--store-pins; the IFEval battery's admission): ``{<model>/<view>/<cell>@<cap>: tree}``
        #: — every leg admitted by ITS OWN producer tree; None = the one-pin admission (--sha [+ --also-sha])
        self.store_pins: dict[str, str] | None = dict(store_pins) if store_pins is not None else None
        #: B9: cross-tree comparisons are admitted only under a DECLARED tree set (also trees, pinned gen roots, or per-row pins)
        self.declared = (bool(also_trees) or store_pins is not None) if declared is None else bool(declared)
        #: B5 (i′): the per-row ARM CONTRACT is checked on every leg in eight-arm mode / under a declaration (the published
        #: one-sha run keeps its published validation set, byte for byte)
        self.arm_contracts = bool(arm_contracts)
        self.sampling_rows, self.render_facts = sampling_rows, render_facts
        self.model_pins, self.expected_refusals = dict(model_pins), dict(expected_refusals)

    def row_for(self, model_key: str, mode: str) -> Mapping[str, Any]:
        table = self.sampling_rows.get(mode)
        if table is None or model_key not in table:
            raise AnalysisError(f"sampling mode {mode!r} names no row for {model_key!r}")
        return table[model_key]

    def fact_for(self, family: str) -> Mapping[str, Any]:
        fact = self.render_facts.get(family)
        if fact is None:
            raise AnalysisError(f"family {family!r} has no render fact (families: {sorted(self.render_facts)})")
        return fact


def load_ids(source: Path | str | Sequence[str] | None) -> tuple[list[str] | None, str | None]:
    if source is None:
        return None, None
    if isinstance(source, (str, Path)):
        doc = _read_json(Path(source).expanduser())
        path = str(source)
    else:
        doc, path = list(source), None
    if not isinstance(doc, list) or not all(isinstance(i, str) for i in doc):
        raise AnalysisError("--ids: expected a JSON list of item ids")
    if len(set(doc)) != N_ITEMS:
        raise AnalysisError(f"--ids names {len(set(doc))} unique ids, not {N_ITEMS}")
    return sorted(doc), path


def load_matrix_union(paths: Sequence[Path | str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The rows of SEVERAL matrix documents read as ONE design (a matrix-union read: the it-doc matrix carries no ``none``
    rows, its cells pair against the published matrix's ``it`` anchors). Rule: rows are keyed by ``(model_key, view, cell)``;
    a LAUNCH / COMPLETE row of a LATER document REPLACES a HELD row of the same key from an earlier one (a hold is a design
    status, not evidence — the later matrix is the one that launched the cell); any other duplicate (two evidence rows, two
    holds, a hold after evidence) refuses by name. Returns ``(rows, provenance)`` — the rows as a bare list (an in-memory
    matrix: ``load_matrix`` validates every row and holds no single document schema) and the per-document paths / sha256 /
    counts plus every replacement, filed in the report as ``matrix.union``."""
    rows_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    origin: dict[tuple[str, str, str], str] = {}
    prov: dict[str, Any] = {"documents": [], "replaced": []}
    for path in paths:
        p = Path(path).expanduser()
        if not p.is_file():
            raise AnalysisError(f"matrix {p}: no such file")
        blob = p.read_bytes()
        doc = json.loads(blob.decode("utf-8"))
        rows_in = doc.get("rows") if isinstance(doc, Mapping) else doc
        if not isinstance(rows_in, list):
            raise AnalysisError(f"matrix {p}: expected rows")
        prov["documents"].append({"path": str(p), "sha256": hashlib.sha256(blob).hexdigest(), "n_rows": len(rows_in),
                                  "schema": doc.get("schema") if isinstance(doc, Mapping) else None})
        for r in rows_in:
            r = dict(r)
            k = (str(r.get("model_key")), str(r.get("view")), str(r.get("cell")))
            prev = rows_by_key.get(k)
            if prev is None:
                rows_by_key[k], origin[k] = r, str(p)
                continue
            if prev.get("status") == STATUS_HELD and r.get("status") in EVIDENCE_STATUSES:
                prov["replaced"].append({"cell_key": V.cell_key(*k), "held_in": origin[k], "held_reason": prev.get("held_reason"),
                                         "launched_in": str(p), "status": r.get("status")})
                rows_by_key[k], origin[k] = r, str(p)
                continue
            raise AnalysisError(f"matrix union: {V.cell_key(*k)} is named in {origin[k]} ({prev.get('status')!r}) and again in {p} "
                                f"({r.get('status')!r}) — only a LAUNCH row replacing a HELD row is admitted")
    return list(rows_by_key.values()), prov


def load_expected_refusals(matrix: Mapping[str, Any], extra: Path | str | Mapping[str, str] | None) -> dict[str, str]:
    """``{8,192 leg key -> refusal code}``, resolved ONCE for the whole run: the
    matrix rows' ``expected_refusal`` (lane B writes a boolean flag with the
    reason beside it as ``expected_refusal_reason``; a string is taken as the
    code itself) and an optional ``--expected-refusals`` JSON file / mapping
    keyed by leg. Every consumer — the completeness gate, the contrasts, the
    figures, the publication boundary — reads THIS mapping and never the row."""
    out: dict[str, str] = {}
    for r in matrix["launch"]:
        flag = r.get("expected_refusal")
        if flag:
            code = flag if isinstance(flag, str) else (r.get("expected_refusal_reason") or str(flag))
            out[V.leg_key(r["model_key"], r["view"], r["cell"], CAP_BASE)] = str(code)
    if extra is not None:
        doc = _read_json(Path(extra).expanduser()) if isinstance(extra, (str, Path)) else extra
        if not isinstance(doc, Mapping):
            raise AnalysisError("--expected-refusals: expected a JSON object {leg key: refusal code}")
        launch_legs = {V.leg_key(r["model_key"], r["view"], r["cell"], CAP_BASE) for r in matrix["launch"]}
        unknown = sorted(str(k) for k in doc if str(k) not in launch_legs)
        if unknown:
            raise AnalysisError(f"--expected-refusals names {len(unknown)} leg(s) that are not launch rows of the matrix (first {unknown[:3]})")
        out.update({str(k): str(v) for k, v in doc.items()})
    return out


# --------------------------------------------------------------------------- #
# the at-cap record per leg                                                    #
# --------------------------------------------------------------------------- #


def _degeneracy_split(at_cap_ids: Sequence[str], texts: Mapping[str, str | None], *, source: str) -> dict[str, Any]:
    """Evals' degeneracy rule over the text each at-cap draw's ``texts`` carries
    (absent = unavailable): ``n_at_cap == n_loop + n_truncated_reasoning +
    n_text_unavailable`` by construction; per draw the rule's two ratios."""
    classes: dict[str, Any] = {}
    n_loop = n_trunc = n_missing = 0
    for i in sorted(at_cap_ids):
        text = texts.get(i)
        if text is None:
            classes[i] = {"text_source": None, "text_sha256": None, "degenerate": None, "rep_ratio": None, "tail_rep_ratio": None}
            n_missing += 1
            continue
        deg = CL.degeneracy(text)
        classes[i] = {"text_source": source, "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(), **deg}
        if deg["degenerate"]:
            n_loop += 1
        else:
            n_trunc += 1
    return {"text_source": source, "n_at_cap": len(at_cap_ids), "n_classified": n_loop + n_trunc, "n_loop": n_loop,
            "n_truncated_reasoning": n_trunc, "n_text_unavailable": n_missing, "tail_classes": classes}


def at_cap_record(summary: Mapping[str, Any], items: Mapping[str, Mapping[str, Any]], *, rows_text: Mapping[str, str | None],
                  cap: int, trigger_threshold: float = V.GSM8K_TRIGGER_THRESHOLD) -> dict[str, Any]:
    """The leg's at-cap readout: the share (``n_at_cap`` over
    ``n_stop_reason``, the summary's own definition) against the 1 % REPORT line
    and the TRIGGER (``>`` at ``V.GSM8K_TRIGGER_THRESHOLD``: at 1 %, 2 of 200 does not trigger, 3 does), the two
    cases ((b) the cap inside the think block, unscored under the primary policy;
    (c) inside the answer, scored), and the loop split under Evals' degeneracy
    rule (the DECIDING rule for eligibility) over the store rows' text as the
    split the record REPORTS until :func:`choose_split` swaps in the log's, with
    the MATH-500 periodic-block rule as a second, non-deciding column."""
    at_cap_ids = [i for i, rec in items.items() if rec["at_cap"]]
    n_reason = int(summary["n_stop_reason"])
    n_at_cap = int(summary["n_at_cap"] or 0)
    share = (n_at_cap / n_reason) if n_reason else None
    split = _degeneracy_split(at_cap_ids, rows_text, source=M.TEXT_SOURCE_ROWS)
    periodic = M._tail_classes(at_cap_ids, rows_text, source=M.TEXT_SOURCE_ROWS)
    return {"cap": int(cap), "rate": summary.get("at_cap_rate"), "share": share, "n_at_cap": n_at_cap, "n_with_stop_reason": n_reason,
            "line_1pct": V.GSM8K_REPORT_LINE, "line_1pct_exceeded": None if share is None else share > V.GSM8K_REPORT_LINE,
            "trigger_threshold": trigger_threshold, "trigger_exceeded": None if share is None else share > trigger_threshold,
            "n_case_b_unterminated_at_cap": sum(1 for i in at_cap_ids if items[i]["think_block_unterminated"]),
            "n_case_c_answer_at_cap": sum(1 for i in at_cap_ids if not items[i]["think_block_unterminated"]),
            "split_source": SPLIT_SOURCE_STORE_ROWS, "split_source_reason": "the .eval log was not read; the store rows' split stands",
            "loop_rule": LOOP_RULE_TEXT, **{k: split[k] for k in ("n_loop", "n_truncated_reasoning", "n_text_unavailable")},
            "store_rows_split": split,
            "periodic_block_column": {"rule": PERIODIC_RULE_TEXT, "deciding": False,
                                      **{k: periodic[k] for k in ("n_loop", "n_truncated_reasoning", "n_text_unavailable")}}}


def choose_split(at_cap: dict[str, Any], log_split: Mapping[str, Any] | None) -> None:
    """Swap the .eval log's degeneracy split in when it carried EVERY at-cap draw's text."""
    if not log_split:
        return
    if log_split["text_source"] != M.TEXT_SOURCE_EVAL_LOG or log_split["n_classified"] != log_split["n_at_cap"]:
        at_cap["split_source_reason"] = (f"the .eval log was read but carried no text for {log_split['n_text_unavailable']} of "
                                         f"{log_split['n_at_cap']} at-cap draw(s); the store rows' split stands")
        return
    at_cap.update(split_source=SPLIT_SOURCE_EVAL_LOG,
                  split_source_reason="the .eval log was read and carried every at-cap draw's text (the unscored case-(b) draws included)",
                  **{k: log_split[k] for k in ("n_loop", "n_truncated_reasoning", "n_text_unavailable")})


# --------------------------------------------------------------------------- #
# one leg                                                                      #
# --------------------------------------------------------------------------- #


def _leg_skeleton(row: Mapping[str, Any], cap: int, root: Path) -> dict[str, Any]:
    key, view, cell = str(row["model_key"]), str(row["view"]), str(row["cell"])
    arm = V.row_thinking(row)
    return {"leg": V.leg_key(key, view, cell, cap), "cell_key": V.cell_key(key, view, cell), "job_id": row.get("job_id"),
            "model_key": key, "view": view, "tag": row["tag"], "cell": cell, "family": row["family"], "cap": int(cap),
            "rung": cap == CAP_RUNG, "fitted": bool(row["fitted"]), "declared": bool(row["declared"]), "route_kind": row.get("route_kind"),
            "tuple_name": row.get("tuple_name"), "out_root": str(root), "missing": None, "refused": None,
            "certified_unservable": False,
            # llama-paper-rows: the row's ARM (the matrix's `thinking_mode`; `on` on a schema-1 row) and its figure / table label
            "thinking_mode": arm, "thinking_label": thinking_label(arm), "sampling_mode": str(row["sampling_mode"]),
            # E3: the row's object source and, on an it-doc row, the lane its pin is keyed under and the pair's qualification
            # (verbatim off the row; every field None on any other row) — so every reading of the leg carries them
            "object_source": row.get("object_source"), **V.itdoc_fields(row)}


def certified_unservable_record(cell_dir: Path, *, family: str, cell: str, fitted: bool) -> dict[str, Any] | None:
    """The driver's predicate-(A) refusal read off ``<cell_dir>/evidence.json``, or
    ``None`` when the file is absent, unreadable, or does not say
    ``render_certified: false`` — only that exact statement makes a summary-less
    leg a CANDIDATE certified refusal (the validators then decide whether it is
    admitted: ``V.refused_producer_problem`` / ``V.refusal_binding_problem`` /
    ``V.certified_refusal_problem``); anything else leaves it ``missing``.
    Carries the certification problems, the recorded ON draws (class per draw),
    the certifying draw's class (absent on evidence written before the field),
    the evidence itself (for the validators) and a NEUTRAL observation of what
    the draws did, naming the route (the cell's object, or the BARE route) —
    never a causal reading; that belongs in the results document beside a
    control."""
    path = Path(cell_dir) / "evidence.json"
    if not path.is_file():
        return None
    try:
        evidence = _mapping(_read_json(path))
    except AnalysisError:
        return None
    rc = _mapping(evidence.get("render_certified"))
    if rc.get("render_certified") is not False:
        return None
    problems = [str(x) for x in (rc.get("certification_problems") or [])]
    attempts = [a for a in (_mapping(evidence.get("render_probe")).get("on_attempts") or []) if isinstance(a, Mapping)]
    classes = [str(a.get("class")) for a in attempts]
    n_unclosed = sum(1 for c in classes if c == PROBE_CLASS_UNCLOSED_BLOCK)
    route = f"the route of the cell's object ({cell})" if fitted else "the BARE route (no object)"
    observation = None
    if attempts and n_unclosed == len(attempts):
        observation = (f"all {len(attempts)} ON probe draws in this standup stopped without closing the thought block, on {route}; "
                       "certification refused")
    elif attempts:
        observation = f"the {len(attempts)} ON probe draws in this standup classified {classes}, on {route}; certification refused"
    return {"code": REFUSAL_CERTIFIED_UNSERVABLE, "certification_problems": problems,
            "problems_joined": "; ".join(problems) or "predicate (A) refused the render (no problem text recorded)",
            "certifying_draw_class": rc.get(V.CERTIFYING_DRAW_CLASS_FIELD),
            "on_attempts": [{k: a.get(k) for k in ("attempt", "prompt_id", "seed", "max_tokens", "finish_reason", "reasoning_chars",
                                                   "content_chars", "class")} for a in attempts],
            "attempt_classes": classes, "n_unclosed": n_unclosed, "family": family, "route": route, "fitted": bool(fitted),
            "split_mode": rc.get("split_mode"), "virtual_model": evidence.get("virtual_model"),
            "evidence_written_at": evidence.get("written_at"), "observation": observation, "evidence": dict(evidence)}


def load_store_pins(sources: Sequence[Path | str | Mapping[str, str]] | Path | str | Mapping[str, str] | None) -> dict[str, str] | None:
    """The per-leg admitted producer trees (``{"<model>/<view>/<cell>@<cap>": "<tree sha>"}``; the IFEval battery's
    ``--store-pins`` contract): one or more JSON files or mappings, merged; keys starting with ``_`` are documentation
    (``_meta``, per-row sources) and are skipped; every value is hex of >= 7 characters (a tree sha or its prefix); a leg
    key named twice with DIFFERENT pins refuses by name; anything else off-contract refuses by name."""
    if sources is None:
        return None
    if isinstance(sources, (str, Path, Mapping)):
        sources = [sources]
    out: dict[str, str] = {}
    for src in sources:
        doc = json.loads(Path(src).expanduser().read_text(encoding="utf-8")) if isinstance(src, (str, Path)) else dict(src)
        if not isinstance(doc, Mapping):
            raise AnalysisError(f"store pins {src}: not a JSON object")
        for k, v in doc.items():
            if str(k).startswith("_"):
                continue
            v = str(v).lower()
            if len(v) < 7 or any(ch not in "0123456789abcdef" for ch in v):
                raise AnalysisError(f"store pins: {k}: {v!r} is not a tree sha (>= 7 hex)")
            if "@" not in str(k) or str(k).count("/") != 2:
                raise AnalysisError(f"store pins: {k!r} is not a leg key <model>/<view>/<cell>@<cap>")
            prev = out.get(str(k))
            if prev is not None and not (prev.startswith(v) or v.startswith(prev)):
                raise AnalysisError(f"store pins: {k} is pinned twice with different trees ({prev!r} vs {v!r})")
            out[str(k)] = max((prev or ""), v, key=len)        # two spellings of ONE tree: keep the longer
    return out


def admitted_pin_problem(leg_key: str, tree: Any, pins: Mapping[str, str]) -> tuple[str | None, str | None]:
    """``(pin, problem)`` for one leg under per-row pins: no entry for the leg, or a producer tree that is not its admitted
    pin (a 40-hex tree or a >= 7-hex prefix of it), each refused by name."""
    want = pins.get(leg_key)
    if want is None:
        return None, f"no admitted pin for {leg_key} in the store-pins doc (the leg is not part of the admitted set)"
    tree = str(tree or "")
    if not tree or not (tree == want or (len(want) >= 7 and tree.startswith(want))):
        return want, f"producer tree {tree[:8] or None!r} is not the admitted pin {want[:8]!r} for {leg_key} (--store-pins)"
    return want, None


def _roots_of(gen_root: Path | str | M.GenRoots, ctx: Context) -> M.GenRoots:
    return M.GenRoots.of(gen_root, primary_pin=ctx.store_pin)


def resolve_under_roots(roots: M.GenRoots, rel: str, *, is_dir: bool = True) -> tuple[Path | None, str | None]:
    """The one gen root holding ``rel`` (a directory, or a file): ``(root path, None)``; ``(None, None)`` when none does;
    ``(None, why)`` when two or more do — the same logical object under two roots (a filesystem union, plan P5)."""
    hits = [e for e in roots.entries if ((e["path"] / rel).is_dir() if is_dir else (e["path"] / rel).is_file())]
    if len(hits) > 1:
        return None, (f"{rel!r} exists under {len(hits)} gen roots ({[str(e['path']) for e in hits]}) — the same logical "
                      "object under two roots (a filesystem union); pull each wave into its own root")
    return (hits[0]["path"] if hits else None), None


def collect_leg(gen_root: Path | M.GenRoots, row: Mapping[str, Any], *, cap: int, ctx: Context, read_eval_logs: bool = True,
                subset_ids: Sequence[str] | None = None, source_manifest: Mapping[str, Any] | None = None,
                subset: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """One leg's record (the 8,192 leg, or the 16,384 rung leg over ``subset_ids``
    bound to its ``source_manifest``, its ``cap_rung`` identity and ``item_subset``
    block held to the planner's validated ``subset`` file). ``missing`` names a
    leg the run did not produce; ``refused`` an inconsistency between sources, a
    store that is not this leg's production run, a partial run, a population
    that is not the expected one, or a ``.eval`` re-derivation that disagrees.
    The gate records (ledger, request evidence, stamps, certification,
    sampling_ref, at-cap) are filed on a usable leg and read by the gates."""
    roots = _roots_of(gen_root, ctx)
    out_rel = str(row["rung_out_root"] if cap == CAP_RUNG else row["out_root"])
    # plan P5 / B6: the leg's out root resolves under exactly ONE gen root — a bare root keeps the published one-pin
    # semantics, a PINNED root binds its legs to that admitted tree; two roots holding it is a duplicate logical leg
    entry, dup = roots.resolve(out_rel)
    gen_root_path = entry["path"] if entry is not None else roots.primary
    root = gen_root_path / out_rel
    leg = _leg_skeleton(row, cap, root)
    arm, sampling_mode = str(leg["thinking_mode"]), str(leg["sampling_mode"])
    leg["gen_root"] = str(gen_root_path)
    leg["wave"], leg["status"] = V.row_wave(row), str(row["status"])
    if dup:
        leg["refused"] = dup
        return leg
    if entry is None and len(roots.entries) > 1:
        leg["missing"] = f"no out-root {out_rel!r} under any gen root ({[str(e['path']) for e in roots.entries]})"
        return leg
    root_pin, root_also = (str(entry["pin"]), ()) if entry is not None and entry["pinned"] else (ctx.store_pin, ctx.also_trees)
    admitted_pin: str | None = None
    if ctx.store_pins is not None:
        # per-ROW admission (--store-pins): the leg is held to ITS row's pin, whatever root it sits under; a leg without an entry
        # refuses here by name, before any store is opened
        admitted_pin, why = admitted_pin_problem(str(leg["leg"]), None, ctx.store_pins)
        if admitted_pin is None:
            leg["refused"] = why
            return leg
        root_pin, root_also = admitted_pin, ()
        leg["admitted_pin"] = admitted_pin
    expected_ids = sorted(str(i) for i in subset_ids) if subset_ids is not None else None
    cd, manifest, why = V.read_store(root)
    if cd is None:
        leg["missing"] = why
        return leg
    if manifest is None:
        leg["refused"] = why
        return leg
    leg["cell_dir"], leg["store_key"] = str(cd), cd.name
    # code review C2: the launch attestation — the standup's own record of the composed launch (the API server's startup
    # echo in its log), located through the manifest under THIS leg's gen root and bound to its job; `absent` with its
    # problems when the pull carries no serving/ record (a declared-tree contrast refuses on that, the one-sha rule does not)
    leg["launch"] = M.launch_attestation(manifest, gen_root=gen_root_path, task_subdir=LAUNCH_TASK_SUBDIR)
    # a DECLARED header-located cell's served positions: the authoritative fact lives in the standup record beside the
    # job (served-positions-null-diagnosis.md §2); resolved ONCE here and read by identity_problem / certification_check
    # only where the manifest carries the producer's null (§1) — never where the manifest states the value
    if bool(row["fitted"]) and bool(row["declared"]):
        standup_record = V.declared_served_positions_record(manifest, row=row, gen_root=gen_root_path, fact=ctx.fact_for(str(row["family"])))
    elif bool(row["fitted"]) and (V.native_served_positions_gap(manifest) or V.native_served_positions_stamped(manifest)):
        # the NATIVE record gap (V.RECORD_GAP_CAUSE_NATIVE_LENS_MAP_ONLY): a natively served undeclared fitted cell whose
        # manifest carries the producer's null — resolved from the native standup record beside the job (the engine-neutral
        # served_positions map, else the routes[] position-mode attestation), ONE predicate shared with identity_problem; a
        # manifest that states the value is held to the row as before. The FIXED producer's STAMP (V.native_served_positions_
        # stamped: the driver's neutral-map lookup wrote the engine-neutral value into the lens-named `lens_served_positions`
        # field) is resolved too, so certification_check can hold the stamp to the record's engine-neutral map — never the
        # stamp alone, never the engine's name
        standup_record = V.native_served_positions_record(manifest, row=row, gen_root=gen_root_path)
    else:
        standup_record = None
    if standup_record is not None:
        leg["standup_record"] = {k: standup_record.get(k) for k in ("resolved", "resolved_from", "source", "problem", "standup_dir", "jobdir",
                                                                    "job_id", "standup_sha", "producer_tree_sha")}
    summary = completion.read_task_summary(cd, TASK)
    if summary is None:
        why = f"no {C.SUMMARY_DIR}/{C.marker_filename(TASK)} under {cd} — the leg filed no summary"
        cert = certified_unservable_record(cd, family=str(row["family"]), cell=str(row["cell"]), fitted=bool(row["fitted"]))
        if cert is None:
            leg["missing"] = why
            return leg
        evidence = cert.pop("evidence")
        leg["no_summary"] = why
        # The refusal is admitted ONLY through the same identity and producer validation a
        # summary-bearing leg passes (refusal-specific: the last run incomplete, nothing accepted), BOUND to the run
        # that ended the store (the evidence written after that run started; the wrapper's ladder_state, when beside
        # the store, naming the same run as DRIVER_FAILED at the pin), and ONLY when the non-behavioural predicates
        # hold independently — the served header ids, the regime stamps, the cell's route, and every ON draw in the
        # behavioural set. Anything else is BLOCKING: the leg stays `missing` with the problem named.
        problem: str | None = None
        if cap != CAP_BASE:
            problem = "a rung leg is never admitted as a certified refusal (its base leg produced; the rung's refusal is a defect)"
        if problem is None:
            problem = V.identity_problem(manifest, None, row=row, cap=cap, model_pin=ctx.model_pins[str(row["tag"])],
                                         sampling_row=ctx.row_for(str(row["model_key"]), sampling_mode), fact=ctx.fact_for(str(row["family"])),
                                         standup_record=standup_record, arm=arm)
        if problem is None and ctx.arm_contracts:
            problem = V.arm_contract_problem(manifest, row=row)
        last_attempt = None
        if problem is None:
            problem, last_attempt = V.refused_producer_problem(manifest, pin=root_pin, also=root_also)
        if problem is None:
            ladder_path = root / V.LADDER_STATE_FILENAME
            try:
                ladder_state = _mapping(_read_json(ladder_path)) if ladder_path.is_file() else None
            except AnalysisError as e:
                ladder_state, problem = None, f"{V.LADDER_STATE_FILENAME} beside the store is unreadable: {e}"
            if problem is None:
                problem = V.refusal_binding_problem(evidence, last_attempt=last_attempt, ladder_state=ladder_state, pin=root_pin)
        if problem is None:
            problem = V.certified_refusal_problem(evidence, manifest=manifest, row=row, fact=ctx.fact_for(str(row["family"])))
        cert["admitted"] = problem is None
        cert["not_admitted_because"] = problem
        cert["refused_attempt"] = (last_attempt.get(V.RESUME_ATTEMPT_ID_FIELD) if isinstance(last_attempt, Mapping) else None)
        leg["render_certification"] = cert
        if problem is not None:
            leg["missing"] = (f"{why}; evidence.json names {REFUSAL_CERTIFIED_UNSERVABLE} ({cert['problems_joined']}) but the leg is NOT "
                              f"admitted as a certified refusal — {problem}")
            return leg
        # the driver certified the cell UNSERVABLE before generating and every independent check holds: an admitted,
        # named outcome — refused, never missing (gate_completeness files it as `refused_certified`)
        _file_record_gaps(leg, standup_record)
        leg["certified_unservable"] = True
        leg["refused"] = f"{REFUSAL_CERTIFIED_UNSERVABLE}: {cert['problems_joined']}"
        return leg
    leg["summary"] = summary
    tinycap = _mapping(manifest.get(C.MANIFEST_KEY_TINYCAP))
    served = _mapping(tinycap.get("served"))
    route = _mapping(manifest.get(C.MANIFEST_KEY_ROUTE))
    producer = _mapping(manifest.get(C.MANIFEST_KEY_PRODUCER))
    leg["identity"] = {
        "cell_id": manifest.get("cell_id"), "harness_cell_id": manifest.get("harness_cell_id"), "virtual_model": route.get("virtual_model"),
        "model_id": manifest.get("model_id"), "model_revision": manifest.get("model_revision"),
        "intervention_hash": manifest.get("intervention_hash"), "harness_config_hash": manifest.get("harness_config_hash"),
        "producer_tree_sha": producer.get("tree_sha"), "object_artifact_sha256": _mapping(manifest.get(C.MANIFEST_KEY_PROVENANCE)).get("artifact_sha256"),
        C.IDENTITY_KEY_IDENTITY_VERSION: tinycap.get(C.IDENTITY_KEY_IDENTITY_VERSION),
        C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING: tinycap.get(C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING),
        C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT: tinycap.get(C.IDENTITY_KEY_SERVE_OUT_OF_SUPPORT),
        C.IDENTITY_KEY_MAX_TOKENS_BY_TASK: tinycap.get(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK),
        V.IDENTITY_KEY_CAP_RUNG: tinycap.get(V.IDENTITY_KEY_CAP_RUNG),
        "served": {k: served.get(k) for k in ("max_model_len", "max_num_seqs", "concurrency", "reasoning_parser", C.REDUCTION_MODE_FIELD)},
        C.MANIFEST_KEY_AUTHORISED_EXPERIMENT: manifest.get(C.MANIFEST_KEY_AUTHORISED_EXPERIMENT),
        C.MANIFEST_KEY_RESUME_ATTEMPTS: manifest.get(C.MANIFEST_KEY_RESUME_ATTEMPTS),
        V.MANIFEST_KEY_ACCEPTED_ATTEMPT: manifest.get(V.MANIFEST_KEY_ACCEPTED_ATTEMPT),
        "sampling": _mapping(manifest.get(C.MANIFEST_KEY_SAMPLING)) or None,
        C.MANIFEST_KEY_TRANSFER_PROVENANCE: manifest.get(C.MANIFEST_KEY_TRANSFER_PROVENANCE),
        "proxy_upstream_timeout_s": manifest.get("proxy_upstream_timeout_s"),
        # B5: the shared-field / server-settings projection and the arm contract read these (strings and bools only)
        **M.projection_identity_fields(manifest),
    }
    fact = ctx.fact_for(str(row["family"]))
    tuple_row = dict(row)
    if cap == CAP_RUNG and row.get("tuple_name"):
        tuple_row["tuple_name"] = V.expected_tuple_name(row, cap)          # the contract's rung tuple (contracts.g8k_tuple_name)
    problem = V.identity_problem(manifest, summary, row=tuple_row, cap=cap, model_pin=ctx.model_pins[str(row["tag"])],
                                 sampling_row=ctx.row_for(str(row["model_key"]), sampling_mode), fact=fact,
                                 rung_source_manifest=source_manifest, rung_subset=subset, standup_record=standup_record, arm=arm)
    if problem is None and cap == CAP_RUNG and subset is not None and source_manifest is not None:
        problem = V.item_subset_block_problem(manifest, subset=subset, source_manifest=source_manifest, source_out_root=str(row["out_root"]))
    leg["identity_validated"] = problem is None
    if problem:
        leg["refused"] = problem
        return leg
    # B5 (i′) / plan P5: the row's OWN arm contract (route class, SI prefix digest, the row's geometry) — checked in
    # eight-arm mode and under a declared tree set; the published one-sha run keeps its published validation set
    if ctx.arm_contracts:
        problem = V.arm_contract_problem(manifest, row=row)
        leg["arm_contract"] = {"checked": True, "problem": problem, "route_class": V.route_class_of(str(row["cell"])),
                               # the steered PATH beside the class:
                               # `lens` | `lora` | null (an unsteered arm) — a new analysis-output field
                               "steered_path": _steered_path_of_manifest(manifest)}
        if problem:
            leg["refused"] = problem
            return leg
    else:
        leg["arm_contract"] = {"checked": False, "problem": None, "route_class": V.route_class_of(str(row["cell"])),
                               "steered_path": _steered_path_of_manifest(manifest)}
    _file_record_gaps(leg, standup_record)
    problem = V.producer_problem(manifest, pin=root_pin, also=root_also)
    if problem is None and ctx.store_pins is not None:
        _pin, problem = admitted_pin_problem(str(leg["leg"]), producer.get("tree_sha"), ctx.store_pins)
    attempts = manifest.get(C.MANIFEST_KEY_RESUME_ATTEMPTS)
    leg["producer"] = {"pin": root_pin, "tree_sha": producer.get("tree_sha"), "driver": producer.get("driver"), "outcomes_tree_sha": None,
                       # per-row admission (--store-pins): the pin this leg was held to, from the pins doc (None under the one-pin rule)
                       "admitted_pin": admitted_pin, "admission": ("per-row pins" if ctx.store_pins is not None else "one store pin"),
                       "n_runs": len(attempts) if isinstance(attempts, list) else None, "n_incomplete_runs": M.incomplete_runs(manifest),
                       "accepted_attempt": manifest.get(V.MANIFEST_KEY_ACCEPTED_ATTEMPT), "matches_pin": problem is None,
                       # plan P5: the gen root the leg was read under and whether that root was PINNED to one admitted tree
                       "root": str(gen_root_path), "root_pinned": bool(entry is not None and entry["pinned"])}
    if ctx.also_trees:
        leg["producer"]["declared_trees"] = list(ctx.also_trees)
    if problem:
        leg["refused"] = problem
        return leg
    for field, want in (("thinking", arm), ("task", TASK), ("max_tokens", int(cap)), ("item_set", V.ITEM_SET)):
        if summary.get(field) != want:
            leg["refused"] = f"summary {field} {summary.get(field)!r} != {want!r}"
            return leg
    partial_path, done_path = completion.partial_marker_path(cd, TASK), completion.done_marker_path(cd, TASK)
    if partial_path.is_file():
        partial = _mapping(_read_json(partial_path))
        leg["refused"] = f"PARTIAL marker at {partial_path}: a limited run ({partial.get('n_items')!r} of {partial.get('n_expected_full')!r} items)"
        return leg
    if not done_path.is_file():
        leg["refused"] = f"no done marker at {done_path} — the task did not complete in this store"
        return leg
    try:
        outcomes = completion.read_task_outcomes(cd, TASK)
    except CompletionError as e:
        leg["refused"] = f"outcomes/{C.task_outcomes_filename(TASK)}: {e}"
        return leg
    if outcomes is None:
        leg["missing"] = f"no {C.TASK_OUTCOMES_DIR}/{C.task_outcomes_filename(TASK)} under {cd} — the all-attempted record is required"
        return leg
    try:
        completion.verify_outcomes_against_summary(outcomes, summary)
    except CompletionError as e:
        leg["refused"] = str(e)
        return leg
    header = {k: v for k, v in outcomes.items() if k != "items"}
    leg["outcomes_header"] = header
    leg["producer"]["outcomes_tree_sha"] = header.get("producer_tree_sha")
    problem = V.outcomes_producer_problem(header, manifest_sha=str(leg["producer"]["tree_sha"]))
    if problem:
        leg["refused"] = problem
        return leg
    if int(header["samples_per_item"]) != 1:
        leg["refused"] = f"samples_per_item {header['samples_per_item']}: the estimand is defined at k = 1"
        return leg
    n_expected = len(expected_ids) if expected_ids is not None else N_ITEMS
    if header["n_attempted_items"] != n_expected or header["n_attempted_samples"] != n_expected:
        leg["refused"] = (f"n_attempted_items {header['n_attempted_items']} / n_attempted_samples {header['n_attempted_samples']} != the "
                          f"{n_expected} draws this leg plans (one per item)")
        return leg
    if summary.get("n_scored", 0) + summary.get("n_unscored", 0) != n_expected:
        leg["refused"] = f"n_scored {summary.get('n_scored')} + n_unscored {summary.get('n_unscored')} != {n_expected}"
        return leg
    if summary.get("n_paired") != n_expected:
        leg["refused"] = f"summary n_paired {summary.get('n_paired')!r} != {n_expected} — not every item paired to the retired selection"
        return leg
    items = {str(rec["item_id"]): rec for rec in outcomes["items"]}
    if len(items) != n_expected:
        leg["refused"] = f"{len(items)} unique attempted ids != {n_expected}"
        return leg
    leg["items"] = items
    leg["attempted_ids"] = sorted(items)
    leg["n_expected"] = n_expected
    if expected_ids is not None and sorted(items) != expected_ids:
        leg["refused"] = f"the rung's attempted ids are not its subset ({len(set(items) ^ set(expected_ids))} differ)"
        return leg
    rows = T._rows_for_task(cd, TASK)
    leg["rows_check"] = T.rows_cross_check(rows, items)
    if not leg["rows_check"]["agrees"]:
        leg["refused"] = f"rows.jsonl disagrees with the outcome record: {leg['rows_check']['detail']}"
        return leg
    if len(rows) != summary.get("n_scored"):
        leg["refused"] = f"rows.jsonl holds {len(rows)} item rows != n_scored {summary.get('n_scored')}"
        return leg
    # llama-paper-rows (I2): the per-row ARM every emitted row carries (`extra_json.thinking`, the EXISTING key; vocabulary
    # on | off | not_available) — every row of the leg wears the leg's arm — and the four row-level provenance fields
    # (`contracts.ROW_PROVENANCE_FIELDS`), PRESENT together iff the run was launched with `--bank-view`, held to the matrix
    # row and the manifest when present, legal when absent (a run without the flag emits rows byte-identical to before)
    problem = row_thinking_problem(rows, arm=arm)
    if problem:
        leg["refused"] = problem
        return leg
    leg["row_thinking"] = arm
    leg["row_provenance"], problem = row_provenance_record(rows, row=row, manifest=manifest)
    if problem:
        leg["refused"] = problem
        return leg
    for field, flag in (("n_at_cap", "at_cap"), ("n_think_block_present", "think_block_present"), ("n_stop_reason", None)):
        have = sum(1 for rec in items.values() if (rec["stop_reason"] is not None if flag is None else rec[flag]))
        want = summary.get(field)
        if field == "n_at_cap" and want is None and summary.get("n_stop_reason") == 0:
            want = 0
        if want != have:
            leg["refused"] = f"summary {field} {summary.get(field)!r} != {have} outcome items with {flag or 'a stop reason'}"
            return leg
    leg["sampling_ref"] = M.sampling_ref_check(rows, manifest, items=outcomes["items"])
    ident_cap = int(_mapping(tinycap.get(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK))[TASK])
    if read_eval_logs:
        leg["eval_check"] = V.eval_cross_check(cd, header, outcomes["items"], max_tokens=ident_cap, accepted_attempt=V.accepted_attempt(manifest))
    else:
        leg["eval_check"] = {"available": False, "reason": "log reading disabled", "agrees": None, "log_present": None, "selected_logs": None,
                             "text": T.text_signature(None, "log reading disabled"), "at_cap_tails": None, "channels": None, "inputs": None}
    channels = leg["eval_check"].pop("channels", None)
    leg["eval_check"].pop("inputs", None)
    if leg["eval_check"]["available"] and not leg["eval_check"]["agrees"]:
        leg["refused"] = f"the .eval log re-derivation disagrees with the record: {leg['eval_check']['detail']}"
        return leg
    try:
        evidence = _mapping(_read_json(cd / "evidence.json"))
    except AnalysisError as e:
        leg["refused"] = str(e)
        return leg
    leg["evidence_present"] = bool(evidence)
    leg["render_fact"] = {"split_mode": fact["split_mode"], "server_parser": fact["server_parser"], "evidence_status": fact.get("evidence_status"),
                          "thinking_switch": fact.get("thinking_switch", True)}
    # the stamp CLASS this leg is held to — the matrix row's, or the served-steer class the leg's own steering-bites record
    # selects (`liveness_gate == logprob`; the E3 follow-up brief's item 6): one triple for the stamps (A2) and predicate (B′) (A3)
    stamp_class = V.leg_expected_stamps(row, evidence, arm=arm, route=route)     # engine-aware: a native leg is never served-steer
    leg["stamp_class"] = stamp_class
    if stamp_class.get("problem"):
        leg["refused"] = str(stamp_class["problem"])
        return leg
    leg["certification"] = V.certification_check(evidence, manifest, row=row, fact=fact, standup_record=standup_record, arm=arm,
                                                 expected_stamps=stamp_class["stamps"], stamp_class=str(stamp_class["class"]))
    ledger_path = root / Path(str(manifest.get(C.MANIFEST_KEY_USAGE_LEDGER) or C.LEDGER_FILENAME)).name
    ledger_rows = list(_iter_jsonl(ledger_path)) if ledger_path.is_file() else None
    leg["ledger"] = V.ledger_check(ledger_path, rows=ledger_rows, header=header, summary=summary, attempted_ids=leg["attempted_ids"],
                                   vmodel=str(route.get("virtual_model")), max_model_len=served.get("max_model_len"), manifest=manifest, cap=cap,
                                   arm=arm)
    leg["request_evidence"] = V.request_evidence_check(
        root, manifest=manifest, evidence=evidence, attempted_samples=[C.sample_header_value(it["item_id"], it["sample_idx"]) for it in outcomes["items"]],
        ledger_rows=ledger_rows, route_vmodel=str(route.get("virtual_model")), fitted=bool(row["fitted"]), declared=bool(row["declared"]), cap=cap,
        arm=arm)
    leg["stamps"] = M.stamp_check(summary, stamp_class["stamps"])
    # the engine-resolved sampler attestation beside the store (the launcher's recorder) — a finding, never a gate
    leg["resolved_sampler"] = V.resolved_sampler_record(root, cap=cap, sampling=row["sampling"])
    rows_text = {str(r["item_id"]): M.emitted_text(r["_extra"].get("content"), r["_extra"].get("reasoning")) for r in rows}
    # E3 tier-2 item 8: the REFUSAL-SHAPED class over the scored draws' CONTENT channel — a judge-free pattern count filed
    # BESIDE n_correct / n_unscored (the denominators and every grader value untouched), so a capability cost can be named as
    # over-refusal, wrong arithmetic or degeneration; the flag rides each outcome item into the readings (`_reading_items`)
    try:
        leg["refusal_shaped"] = RFS.readout(rows, items)
    except RFS.RefusalShapedError as e:
        leg["refused"] = str(e)
        return leg
    for iid in leg["refusal_shaped"]["examined_ids"]:
        items[iid][ITEM_REFUSAL_EXAMINED_KEY] = True       # the denominator: the draws the classifier actually examined
    for iid in leg["refusal_shaped"]["ids"]:
        items[iid][ITEM_REFUSAL_SHAPED_KEY] = True
    at_cap = at_cap_record(summary, items, rows_text=rows_text, cap=cap, trigger_threshold=V.row_trigger_threshold(row))
    if not M._same_number(at_cap["rate"], at_cap["share"]):
        leg["refused"] = f"summary at_cap_rate {at_cap['rate']!r} != n_at_cap / n_stop_reason = {at_cap['share']!r}"
        return leg
    log_texts: dict[str, str | None] | None = None
    if channels:
        at_cap_ids = [i for i, rec in items.items() if rec["at_cap"]]
        log_texts = {i: M.emitted_text(*channels[i]) for i in at_cap_ids if i in channels}
        rows_classes = at_cap["store_rows_split"]["tail_classes"]
        log_split = _degeneracy_split(at_cap_ids, log_texts, source=M.TEXT_SOURCE_EVAL_LOG)
        torn = sorted(i for i, rec in log_split["tail_classes"].items()
                      if rec["text_sha256"] is not None and rows_classes[i]["text_sha256"] is not None
                      and rec["text_sha256"] != rows_classes[i]["text_sha256"])
        if torn:
            leg["refused"] = f"the .eval log's text for {len(torn)} scored at-cap draw(s) differs from rows.jsonl's (first {torn[:3]})"
            return leg
        choose_split(at_cap, log_split)
        # the log-derived reading the figure venv cannot make: under eval_check, outside the two-venv match
        leg["eval_check"]["ladder_derivation"] = V.eligible_from_text(at_cap_ids, log_texts)
    leg["at_cap"] = at_cap
    # the readout as the MATH-500 module now takes it: the row's reasoning channel per draw; under the `none` split the
    # provider's lift record per row rides beside it, so a draw whose row says the client-side lift fired is a stray-tag
    # anomaly even where its text was restored — every other mode ignores it (byte-identical)
    rows_lifted = ({str(r["item_id"]): {f: r["_extra"].get(f) for f in C.ROW_LIFT_FIELDS} for r in rows}
                   if fact["split_mode"] == C.THINK_SPLIT_NONE else None)
    leg["thinking"] = M.thinking_record(items, split_mode=fact["split_mode"],
                                        rows_reasoning={str(r["item_id"]): r["_extra"].get("reasoning") for r in rows},
                                        rows_lifted=rows_lifted)
    if rows_lifted is not None:
        leg["rows_lifted"] = rows_lifted
        # The raw row content read on its own — a torn record refuses the leg; a literal tag the flags missed is a
        # stray-tag anomaly for A4 (stray_tag_record reads `rows_raw_tags`)
        raw_tags, problems = no_switch_row_evidence(rows, items)
        leg["rows_raw_tags"] = raw_tags
        if problems:
            leg["refused"] = f"no-switch row evidence torn ({len(problems)} problem(s)): " + "; ".join(problems[:3]) + (f"; and {len(problems) - 3} more" if len(problems) > 3 else "")
            return leg
    leg["hop_timeouts"] = CL.served_hop_timeouts(V.read_json_if_file(root / "endpoint.json") or {"upstream_timeout_s": manifest.get("proxy_upstream_timeout_s")},
                                                 C.PROXY_UPSTREAM_TIMEOUT_S)
    return leg


THINK_TAG_LITERALS: tuple[str, ...] = ("<think", "</think")      # every spelling: <think>, </think>, <thinking>, </thinking>


def no_switch_row_evidence(rows: Sequence[Mapping[str, Any]], items: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, list[str]], list[str]]:
    """The RAW ROW CONTENT of a no-switch leg, inspected INDEPENDENTLY of the provider's lift record and of the outcome
    record's flags (review of 2026-09-11, P2: a scored row whose raw content carried `</think>` while the outcome flag read
    false and the lift record said nothing passed A4). Returns ``(raw_tags, problems)``: ``raw_tags`` names, per scored row,
    the think-tag literals its content carries (:data:`THINK_TAG_LITERALS`; fed into the stray-tag rule as a named anomaly
    even where every flag reads false); ``problems`` are TORN records — a row flag (``stray_open_tag`` / ``stray_close_tag``)
    disagreeing with the outcome record's, a ``raw_content_sha256`` that is not the sha256 of the row's text, or a no-switch
    row carrying no ``raw_content_sha256`` at all — each named; the leg is refused on any."""
    raw_tags: dict[str, list[str]] = {}
    problems: list[str] = []
    for r in rows:
        iid, ex = str(r["item_id"]), r["_extra"]
        content = ex.get("content") or ""
        found = [t for t in THINK_TAG_LITERALS if t in content]
        if found:
            raw_tags[iid] = found
        rec = items.get(iid)
        for f in C.STRAY_TAG_FIELDS:
            if f in ex and rec is not None and bool(ex.get(f)) != bool(rec.get(f)):
                problems.append(f"{iid}: the row's {f}={bool(ex.get(f))} disagrees with the outcome record's {bool(rec.get(f))} — a torn record")
        sha = ex.get("raw_content_sha256")
        if sha is None:
            problems.append(f"{iid}: the row carries no raw_content_sha256 — a no-switch row records the provider's raw content digest")
        else:
            digest = CL.text_sha256(content)
            if sha != digest:
                problems.append(f"{iid}: raw_content_sha256 {str(sha)[:16]} is not the sha256 of the row's text ({digest[:16]}) — the row's text is not the raw content")
    return raw_tags, problems


def row_thinking_problem(rows: Sequence[Mapping[str, Any]], *, arm: str) -> str | None:
    """Why the leg's SCORED rows do not all wear the leg's arm on the EXISTING
    per-row key ``extra_json.thinking`` (``contracts.ROW_KEY_THINKING``; the
    cross-eval fork guard reads rows, not manifests): a row lacking the key,
    or carrying another arm, refuses by name."""
    values = Counter(r["_extra"].get(C.ROW_KEY_THINKING, M._ABSENT) for r in rows)
    if not rows:
        return None
    if set(values) != {arm}:
        got = {("<absent>" if k is M._ABSENT else str(k)): n for k, n in values.items()}
        return (f"rows.jsonl: the per-row {C.ROW_KEY_THINKING} key reads {got} on {len(rows)} scored row(s) — every row of a {arm} leg carries "
                f"{C.ROW_KEY_THINKING}: {arm}")
    return None


def row_provenance_record(rows: Sequence[Mapping[str, Any]], *, row: Mapping[str, Any],
                          manifest: Mapping[str, Any]) -> tuple[dict[str, Any], str | None]:
    """The ROW-LEVEL provenance the driver emits under ``--bank-view``
    (``contracts.ROW_PROVENANCE_FIELDS``: ``object_artifact_sha256``,
    ``route_kind``, ``view``, ``producer_tree_sha``), read off the leg's rows:
    ABSENT on every row (a run without the flag — legal; ``present`` False,
    every field None) or PRESENT on every row with ONE value each, held to the
    matrix row and the manifest — the object digest equal to the row's
    ``object_sha256`` (a stated null on a base / prompted route), the route kind
    the matrix's, the view the row's STAMP RULE gives (``contracts.expected_bank_view_stamp``:
    the row's view under the instrument convention, else its SOURCE bank when it declares one —
    the seed-replicate Arditi rows evaluate on ``s<s>`` from ``cells_view/ar_c0p3`` — else the
    lane view), the producer tree the manifest's (itself held to the pin). A mixed leg (some rows with, some without), a field
    present alone, or a disagreement refuses by name — the record is never
    guessed from the manifest."""
    fields = tuple(C.ROW_PROVENANCE_FIELDS)
    out: dict[str, Any] = {"present": False, **{f: None for f in fields}}
    if not rows:
        return out, None
    with_all = [r for r in rows if all(f in r["_extra"] for f in fields)]
    with_any = [r for r in rows if any(f in r["_extra"] for f in fields)]
    if not with_any:
        return out, None
    if len(with_any) != len(rows) or len(with_all) != len(with_any):
        return out, (f"rows.jsonl: the row provenance {list(fields)} rides {len(with_any)} of {len(rows)} row(s) ({len(with_all)} complete) — the four "
                     "fields are present together on every row of a run launched with --bank-view, or on none")
    values = {f: sorted({json.dumps(r["_extra"].get(f), default=str) for r in rows}) for f in fields}
    mixed = {f: v for f, v in values.items() if len(v) != 1}
    if mixed:
        return out, f"rows.jsonl: the row provenance carries several values on one leg: {mixed}"
    got = {f: rows[0]["_extra"].get(f) for f in fields}
    want_sha = row.get("object_sha256") if bool(row["fitted"]) else None
    tree = _mapping(manifest.get(C.MANIFEST_KEY_PRODUCER)).get("tree_sha")
    want = {"object_artifact_sha256": want_sha, "route_kind": row.get("route_kind"),
            "view": C.expected_bank_view_stamp(view=row["view"], bank_view=row.get("bank_view")), "producer_tree_sha": tree}
    off = {f: (got[f], want[f]) for f in fields if got[f] != want[f]}
    if off:
        return out, ("rows.jsonl: the row provenance disagrees with the matrix row / the manifest — "
                     + "; ".join(f"{f} {g!r} != {w!r}" for f, (g, w) in off.items()))
    if got["route_kind"] not in C.ROUTE_KINDS or C.bank_view_problem(got["view"]) is not None:
        # the view vocabulary is `main` | `it` + the roster's lane views (contracts.bank_views_admitted; the Qwen
        # seed-replicate rows sit on `s1` / `s2` / `s3` themselves, and the equality above already held the stamp to the row)
        return out, (f"rows.jsonl: the row provenance's route_kind {got['route_kind']!r} / view {got['view']!r} are outside "
                     f"{C.ROUTE_KINDS} / {C.bank_views_admitted()}")
    out.update(present=True, **got)
    return out, None


def _steered_path_of_manifest(manifest: Mapping[str, Any]) -> str | None:
    """The steered PATH a store's route names — ``"lens"`` | ``"lora"`` | ``None`` (an unsteered
    passthrough arm, or a manifest with no route identity) — read from ``route.serving_identity``
    through the MATH module's :func:`steered_path_of`, the same reading
    :func:`V.arm_contract_problem` makes. A torn identity (both markers) raises there and
    is left to raise here too: the leg is malformed, and this field must not name a path for it."""
    route = manifest.get(C.MANIFEST_KEY_ROUTE)
    identity = route.get("serving_identity") if isinstance(route, Mapping) else None
    return M.steered_path_of(identity)


def _file_record_gaps(leg: dict[str, Any], standup_record: Mapping[str, Any] | None) -> None:
    """File the producer RECORD GAPS the standup record closed on this leg
    (``V.RECORD_GAPS_KEY``): one entry per manifest served-positions field the
    producer wrote null, naming the field, the resolved value, its source and
    the producer cause. Nothing is filed where the manifest stated the value
    (no gap) or where the record did not resolve (the leg refused upstream)."""
    if isinstance(standup_record, Mapping) and not standup_record.get("problem") and standup_record.get("gaps"):
        leg[V.RECORD_GAPS_KEY] = [dict(g) for g in standup_record["gaps"]]


def record_gaps_block(legs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The report's RECORD-GAP summary: every leg whose manifest carried a
    producer null that the analysis resolved from the authoritative standup
    record, the gaps themselves, and the counts by field and by cause — so the
    artifact names the producer defect instead of hiding a resolved value
    behind a passing gate. The manifests themselves are unchanged."""
    per = {k: [dict(g) for g in leg[V.RECORD_GAPS_KEY]] for k, leg in legs.items() if leg.get(V.RECORD_GAPS_KEY)}
    by_cause = Counter(str(g["cause"]) for gaps in per.values() for g in gaps)
    by_field = Counter(str(g["field"]) for gaps in per.values() for g in gaps)
    return {"n_legs": len(per), "n_gaps": sum(len(v) for v in per.values()), "by_cause": dict(sorted(by_cause.items())),
            "by_field": dict(sorted(by_field.items())), "legs": dict(sorted(per.items())),
            "note": ("a manifest field the producer wrote null (or derived from that null), resolved from the authoritative standup record "
                     f"beside the job ({V.STANDUP_DIR_PARENT}/<jobdir>/{V.STANDUP_TASK_SUBDIR}/{V.STANDUP_ROUTING_TABLE}: on a DECLARED cell the lens "
                     "map under the declared key, cross-checked against the routes entry, the lens registry, the attestation's job id and the "
                     "standup's tree sha, and the manifest's own declared-class facts; on a NATIVELY served undeclared cell the engine-neutral "
                     f"{V.SERVED_POSITIONS_KEY} map, else the routes[] {V.ROUTE_POSITION_MODE_KEY} attestation, cross-checked against the routes "
                     "entry, the serving attestation's kind and job id, the standup's tree sha and engine, the row's expected positions and the "
                     "contract's header-located reading of the resolved fact, with the block's header fields re-derived through the contract); "
                     "the manifest is unchanged and the gate stays authoritative — an unresolvable null refuses")}


def record_gap_text(gaps: Sequence[Mapping[str, Any]]) -> str:
    """One leg's record gaps, in words: the fields the producer wrote NULL, the
    fields it DERIVED from that null (``manifest -> re-derived``), and the value
    resolved from the record with its source — the findings line and the CLI
    summary share it."""
    nulls = [str(g["field"]) for g in gaps if g.get("manifest") is None]
    derived = [f"{g['field']} {g['manifest']!r} -> {g['resolved']!r}" for g in gaps if g.get("manifest") is not None]
    parts = ([f"{', '.join(nulls)} null in the manifest"] if nulls else []) + ([f"{', '.join(derived)} re-derived from the null"] if derived else [])
    return f"{'; '.join(parts)}; resolved {gaps[0]['resolved']} from {gaps[0]['resolved_from']}"


def usable(leg: Mapping[str, Any]) -> bool:
    return not (leg.get("missing") or leg.get("refused"))


#: the DECLARED expected refusal of a CONTRAST (arditi-allpos-cap, 2026-09-17) — distinct from the LEG-level `expected_refusal`
#: above (a standup refusing a cell by name; A0's refused_expected): the matrix row's `expected_contrast_refusal` (g8k_views stamps
#: it from the steering entry; g8k_cells.EXPECTED_CONTRAST_REFUSAL_CODES) says the row's contrast against the view's `none` anchor
#: refuses BY DESIGN with a code. Today ONE code: the O106 cross-engine ruling — the reader's own constant, `M.CROSS_ENGINE_RULING`,
#: never a new string. The ruling in force: no native `none` anchor is run; the arm's numbers are reported as the arm's numbers and
#: the pairing's absence accepted. The code is DERIVED from the contrast record (contrast_refusal_code), never read off a stamp
ROW_EXPECTED_CONTRAST_REFUSAL_KEY = "expected_contrast_refusal"
CONTRAST_REFUSAL_CODES: tuple[str, ...] = (M.CROSS_ENGINE_RULING,)
EXPECTED_CONTRAST_REFUSAL_PROBLEM_KEY = "expected_contrast_refusal_problem"
CONTRAST_REFUSAL_CODE_KEY, CONTRAST_REFUSED_BY_DESIGN_KEY = "refusal_code", "refused_by_design"


def contrast_refusal_code(rec: Mapping[str, Any]) -> str | None:
    """The CODE a refused contrast's refusal derives to, or ``None``: :data:`M.CROSS_ENGINE_RULING` iff the record's compatibility
    says ``cross_engine`` and the contrast's ``reason`` IS the cross-engine refusal (:func:`M.server_settings_difference`'s text, filed
    as ``compatibility.cross_engine_refusal``) — the record's own facts, never a stamp or a substring. A computable contrast, or one
    refused on any other rule (a missing leg, a differing flag, an unattested launch), has no code."""
    if rec.get("computable") or rec.get("reason") is None:
        return None
    compat = _mapping(rec.get("compatibility"))
    if compat.get("cross_engine") is True and rec.get("reason") == compat.get("cross_engine_refusal"):
        return M.CROSS_ENGINE_RULING
    return None


def _not_run(contrasts: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The contrasts that filed no estimand and were NOT refused by name — neither
    an expected refusal (the operator's mapping) nor the driver's certified one,
    nor a refusal the design DECLARED and the record derives EXACTLY
    (:data:`CONTRAST_REFUSED_BY_DESIGN_KEY`: a filed outcome, not a contrast that did not run)."""
    return [c for c in contrasts if not c.get("computable") and not c.get("expected_refusal") and not c.get("certified_refusal")
            and not c.get(CONTRAST_REFUSED_BY_DESIGN_KEY)]


def contrast_declaration_problems(contrasts: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every way a DECLARED expected contrast refusal is not met, by name: a declaration the reader cannot honour (a code it does not
    derive); a declared contrast that COMPUTED ("expected a refusal, got a contrast" — the declaration is stale, never a contrast
    filed under a declaration that says there is none); a declared contrast refused with another code, or with no named code.
    An undeclared contrast contributes nothing here (it is read exactly as before)."""
    out: list[str] = []
    for c in contrasts:
        if c.get(EXPECTED_CONTRAST_REFUSAL_PROBLEM_KEY):
            out.append(f"{c['name']}: {c[EXPECTED_CONTRAST_REFUSAL_PROBLEM_KEY]}")
            continue
        code = c.get(ROW_EXPECTED_CONTRAST_REFUSAL_KEY)
        if code is None:
            continue
        if c.get("computable"):
            out.append(f"{c['name']}: expected a refusal ({code}), got a contrast — the design declares this contrast refuses by name and it "
                       "computed; retire the declaration or name the pairing")
        elif not c.get(CONTRAST_REFUSED_BY_DESIGN_KEY):
            out.append(f"{c['name']}: declared an expected refusal ({code}) but refused with {c.get(CONTRAST_REFUSAL_CODE_KEY) or 'no named code'} — "
                       f"{c.get('reason')}")
    return out


# --------------------------------------------------------------------------- #
# the ladder per cell: the record, the subset, the rung leg, the merge         #
# --------------------------------------------------------------------------- #


def collect_cell(gen_root: Path | M.GenRoots, row: Mapping[str, Any], *, ctx: Context, read_eval_logs: bool) -> dict[str, Any]:
    """One cell: its 8,192 leg, THIS instrument's ladder record beside it, the
    planner's subset, the rung leg when escalated, the redraw records and the
    two readings (the merge is analysis-side — per item the 16,384
    draw where the id is in the subset, else the 8,192 draw)."""
    roots = _roots_of(gen_root, ctx)
    base = collect_leg(roots, row, cap=CAP_BASE, ctx=ctx, read_eval_logs=read_eval_logs)
    cell: dict[str, Any] = {"cell_key": base["cell_key"], "model_key": base["model_key"], "view": base["view"], "cell": base["cell"],
                            "fitted": base["fitted"], "declared": base["declared"], "base_leg": base["leg"], "rung_leg": None,
                            "legs": {base["leg"]: base}, "ladder": None, "readings": None, "redraws": None,
                            "wave": base["wave"], "status": base["status"],
                            # E3: the it-doc carriage rides the cell record too (None on every other row)
                            "object_source": base.get("object_source"), **{f: base.get(f) for f in V.ITDOC_ROW_FIELDS}}
    # E3 tier-2 item 5: the object GENERATION stamp (batch8 | batch64), READ from the object's fit provenance by the row's
    # `object_sha256` (V.object_generation_of_row over the Context's bindings) and held to the row's declared word where it carries
    # one — never inferred from a staged id; a disagreement refuses the analysis by name (a design defect, not evidence). The paper-rows
    # export reads the stamp off THIS record (`_pr_object_generation` -> `_pr_stamp`); an unbound fitted row stays None (`n/a` there).
    gen, gen_problem = V.object_generation_of_row(row, ctx.object_generations)
    if gen_problem:
        raise AnalysisError(f"cells[{cell['cell_key']}]: {gen_problem}")
    cell["object_generation"] = gen
    cell["object_generation_source"] = (ctx.object_generations.get(str(row.get("object_sha256")), {}).get("source") if gen is not None else None)
    if not usable(base):
        return cell
    # the planner's files of THIS cell (its subset, its rung store) resolve under exactly one gen root — the base leg's
    # own root first (a wave-isolated pull carries its own `rung16384/`), else the one root holding them
    base_root = Path(base["gen_root"])
    rung_root_dir, rung_dup = resolve_under_roots(roots, str(row["rung_out_root"]))
    subset_root, subset_dup = resolve_under_roots(roots, str(row["rung_subset"]), is_dir=False)
    if (base_root / str(row["rung_subset"])).is_file():
        subset_root, subset_dup = base_root, None
    at_cap_ids = sorted(i for i, rec in base["items"].items() if rec["at_cap"])
    source_manifest = completion.read_store_manifest(Path(base["cell_dir"])) or {}
    record_path, record, path_problem = V.read_ladder_record(Path(base["out_root"]), Path(base["cell_dir"]))
    rec = record if isinstance(record, Mapping) else {}
    trigger = V.row_trigger_threshold(row)
    eligible = sorted(str(i) for i in rec[V.LADDER_ELIGIBLE_IDS]) if isinstance(rec.get(V.LADDER_ELIGIBLE_IDS), list) else None
    ladder: dict[str, Any] = {"record_path": str(record_path), "record_present": record is not None,
                              "record_problem": path_problem or V.ladder_record_problem(record, n=N_ITEMS, at_cap_ids=at_cap_ids,
                                                                                        source_manifest=source_manifest, trigger_threshold=trigger),
                              "exit": rec.get("exit"), "n_at_cap": len(at_cap_ids),
                              "share_at_cap": len(at_cap_ids) / N_ITEMS,
                              "triggered": len(at_cap_ids) / N_ITEMS > trigger,
                              "over_1pct": len(at_cap_ids) / N_ITEMS > V.GSM8K_REPORT_LINE,
                              "n_non_terminating": rec.get("n_non_terminating"), "n_boundary": rec.get("n_boundary"),
                              "n_eligible": rec.get("n_eligible"), "eligible": eligible,
                              "next_cap": rec.get("next_cap"), "transport": rec.get("transport"),
                              "escalated": isinstance(record, Mapping) and record.get("exit") == CL.ESCALATE,
                              "subset_path": None, "subset_problem": None, "subset_ids": None, "n_subset": None, "subset_sha16": None,
                              "rung_planned_under": None,
                              "rung_store_present": rung_root_dir is not None or rung_dup is not None,
                              "rule": V.GSM8K_TRIGGER_RULE, "threshold": V.GSM8K_TRIGGER_THRESHOLD, "report_line": V.GSM8K_REPORT_LINE}
    derivation = (base.get("eval_check") or {}).get("ladder_derivation")
    if (derivation and derivation["complete"] and isinstance(record, Mapping) and ladder["record_problem"] is None
            and sorted(derivation["eligible"]) != ladder["eligible"]):
        ladder["record_problem"] = (f"the planner's eligible set ({len(ladder['eligible'])}) != this reader's re-derivation under Evals' "
                                        f"degeneracy rule over the .eval log's text ({len(derivation['eligible'])})")
    cell["ladder"] = ladder
    rung: dict[str, Any] | None = None
    if ladder["escalated"]:
        subset_path = (subset_root if subset_root is not None else base_root) / str(row["rung_subset"])
        subset = V.read_json_if_file(subset_path)
        ladder.update(subset_path=str(subset_path),
                      subset_problem=(subset_dup or rung_dup
                                      or V.rung_subset_problem(subset, record=record, source_manifest=source_manifest, source_out_root=str(row["out_root"]))))
        if isinstance(subset, Mapping):
            # the pair the planner wrote this subset under (the current 1 % rule, or the retired 3 % one of the first publication)
            ladder["rung_planned_under"] = V.rung_planned_under(subset.get("rule"))
        if isinstance(subset, Mapping) and isinstance(subset.get("ids"), list):
            ladder["subset_ids"] = sorted(str(i) for i in subset["ids"])
            ladder["n_subset"] = len(ladder["subset_ids"])
            ladder["subset_sha16"] = V.subset_sha16(ladder["subset_ids"])
        if ladder["subset_problem"] is None:
            # the rung's cap_rung identity and item_subset block are held to THIS validated subset file inside collect_leg
            rung = collect_leg(roots, row, cap=CAP_RUNG, ctx=ctx, read_eval_logs=read_eval_logs, subset_ids=ladder["subset_ids"],
                               source_manifest=source_manifest, subset=subset)
            cell["rung_leg"] = rung["leg"]
            cell["legs"][rung["leg"]] = rung
    cell["readings"], cell["redraws"] = readings_of(base, rung if (rung is not None and usable(rung)) else None, ladder,
                                                    policy=ctx.unterminated_finished_policy)
    # the matrix row's DECLARED selection outcome (the steer-admission record's word on an AlphaSteer row: chosen / a4_fallback /
    # ruled_* / chosen_confirmed_* / candidate_*; null on every other row), carried VERBATIM — the paper-rows export reads it
    # from the cell record and never relabels it
    cell["selection_outcome"] = row.get("selection_outcome")
    # arditi-allpos-cap (schema paper_rows/3): the row's DECLARED `selection` block (both readings + the reading's identity) and
    # `fit_provenance` block, VERBATIM off the matrix row (the admission record's, copied by the launcher) — read by the export
    # through `_pr_stamp`, never composed or filled here; null on every row that declares none
    cell["selection"] = row.get("selection")
    cell["fit_provenance"] = row.get("fit_provenance")
    return cell


#: E3 tier-2 item 8: the per-item flag the leg's refusal-shaped readout (sisteer.capeval.tinycap.refusal_shaped) sets on a SCORED
#: outcome item whose content channel matched a pattern; it rides the reading's items so a reading's `n_refusal_shaped` sums from them
ITEM_REFUSAL_SHAPED_KEY = "refusal_shaped"
#: and the DENOMINATOR's flag: the outcome item the classifier actually examined (its content channel on record) — a reading's
#: `n_refusal_shaped_of` sums from these, so the count carries its n on every row
ITEM_REFUSAL_EXAMINED_KEY = "refusal_shaped_examined"


def _reading_items(items: Mapping[str, Mapping[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    return {i: {"scored": bool(rec["scored"]), "value": rec.get("value"), "at_cap": bool(rec["at_cap"]),
                "think_block_unterminated": bool(rec["think_block_unterminated"]), "think_block_present": bool(rec["think_block_present"]),
                "stray_open_tag": bool(rec.get("stray_open_tag")), "stop_reason": rec.get("stop_reason"),
                "completion_tokens": rec.get("completion_tokens"), "source": source,
                ITEM_REFUSAL_SHAPED_KEY: bool(rec.get(ITEM_REFUSAL_SHAPED_KEY)),
                ITEM_REFUSAL_EXAMINED_KEY: bool(rec.get(ITEM_REFUSAL_EXAMINED_KEY))} for i, rec in items.items()}


def unterminated_finished(rec: Mapping[str, Any]) -> bool:
    """A draw that ended the turn BELOW the cap inside an unterminated think block (the store's ``think_block_unterminated``
    without its ``at_cap``) — the class the ``failure`` policy re-scores; an at-cap unterminated draw is case (b), not this."""
    return bool(rec.get("think_block_unterminated")) and not bool(rec.get("at_cap"))


def apply_unterminated_finished_policy(items: dict[str, dict[str, Any]], policy: str) -> tuple[dict[str, dict[str, Any]], int]:
    """One reading's items under the reader's policy for unterminated-but-finished draws: ``unscored`` returns ``items``
    UNCHANGED (the same object — the default adds no field and moves no value); ``failure`` returns a copy in which every
    unscored unterminated-finished draw is scored WRONG (``scored`` True, ``value`` 0.0) and flagged, plus the count."""
    if policy not in UNTERMINATED_FINISHED_POLICIES:
        raise AnalysisError(f"{UNTERMINATED_FINISHED_POLICY_KEY} {policy!r} outside {UNTERMINATED_FINISHED_POLICIES}")
    if policy == UNTERMINATED_FINISHED_UNSCORED:
        return items, 0
    out: dict[str, dict[str, Any]] = {}
    n = 0
    for i, rec in items.items():
        if unterminated_finished(rec) and not rec["scored"]:
            out[i] = {**rec, "scored": True, "value": 0.0, ITEM_RESCORED_UNTERMINATED_FINISHED_KEY: True}
            n += 1
        else:
            out[i] = rec
    return out, n


def _reading_summary(items: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    n = len(items)
    scored = [r for r in items.values() if r["scored"]]
    n_correct = sum(1 for r in scored if r["value"] == 1.0)
    return {"n": n, "n_scored": len(scored), "n_unscored": n - len(scored),
            "n_correct": n_correct,
            "acc_stock": (n_correct / len(scored)) if scored else None,
            "n_at_cap_residual": sum(1 for r in items.values() if r["at_cap"]),
            "n_think_unscored": sum(1 for r in items.values() if r["think_block_unterminated"] and not r["scored"]),
            "n_from_rung": sum(1 for r in items.values() if r["source"] == SOURCE_RUNG),
            # item 8: the refusal-shaped class inside the scored draws (a count beside the accuracy counts, never a denominator) WITH its
            # own denominator — the scored draws the classifier actually examined (content channel on record)
            "n_refusal_shaped": sum(1 for r in scored if r.get(ITEM_REFUSAL_SHAPED_KEY)),
            "n_refusal_shaped_of": sum(1 for r in scored if r.get(ITEM_REFUSAL_EXAMINED_KEY))}


def readings_of(base: Mapping[str, Any], rung: Mapping[str, Any] | None, ladder: Mapping[str, Any], *,
                policy: str = UNTERMINATED_FINISHED_UNSCORED) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The two readings of a cell: ``r8192`` (the base draws) and ``merged`` (the
    rung draw substituted for every subset item), with the per-item redraw
    records re-derived from the two outcome records. ``policy`` is the reader's
    treatment of unterminated-but-finished draws (:func:`apply_unterminated_finished_policy`):
    under the default nothing here changes; under ``failure`` each reading's items
    are re-scored and the reading files the policy and its rescored count. The
    redraw records are the store's outcomes either way."""
    r8192, n_r8192 = apply_unterminated_finished_policy(_reading_items(base["items"], SOURCE_R8192), policy)
    merged = dict(r8192)
    redraws: dict[str, Any] | None = None
    if rung is not None:
        redraws = {}
        rung_items, _n_rung = apply_unterminated_finished_policy(_reading_items(rung["items"], SOURCE_RUNG), policy)
        for i, rec in rung_items.items():
            redraws[i] = V.redraw_outcome(base["items"][i], rung["items"][i])
            merged[i] = rec
    named = ({} if policy == UNTERMINATED_FINISHED_UNSCORED
             else {UNTERMINATED_FINISHED_POLICY_KEY: policy})
    n_merged = sum(1 for r in merged.values() if r.get(ITEM_RESCORED_UNTERMINATED_FINISHED_KEY))
    return ({READING_R8192: {"items": r8192, **_reading_summary(r8192), **named,
                             **({N_UNTERMINATED_FINISHED_RESCORED_KEY: n_r8192} if named else {})},
             READING_MERGED: {"items": merged, **_reading_summary(merged), "escalated": rung is not None,
                              "same_as_r8192": rung is None,
                              "redraw_outcomes": dict(Counter(r["outcome"] for r in redraws.values())) if redraws else {},
                              **named, **({N_UNTERMINATED_FINISHED_RESCORED_KEY: n_merged} if named else {})}},
            redraws)


# --------------------------------------------------------------------------- #
# the estimand under both axes                                                 #
# --------------------------------------------------------------------------- #


def branch_items(items: Mapping[str, Mapping[str, Any]], branch: str, axis: str, *, exclude: Sequence[str] = ()) -> dict[str, tuple[int, float | None]]:
    """One leg's pairing input: the residual-at-cap axis applied first (W: every
    at-cap item ``(1, 0.0)``; X: the caller drops the contrast's excluded ids
    from every leg), then the think-block branch on what remains (PRIMARY masks
    an unscored draw out, SCORE-AS-IS scores it wrong)."""
    if branch not in BRANCHES or axis not in AXES:
        raise AnalysisError(f"branch {branch!r} / axis {axis!r} outside {BRANCHES} / {AXES}")
    drop = set(exclude)
    out: dict[str, tuple[int, float | None]] = {}
    for i, rec in items.items():
        if i in drop:
            continue
        if axis == AXIS_W and rec["at_cap"]:
            out[i] = (1, 0.0)
        elif rec["scored"]:
            out[i] = (1, float(rec["value"]))
        elif branch == BRANCH_PRIMARY:
            out[i] = (0, None)
        else:
            out[i] = (1, 0.0)
    return out


def axis_items(legs: Mapping[str, Mapping[str, Mapping[str, Any]]], axis: str) -> dict[str, dict[str, dict[str, Any]]]:
    """The legs' item records under one axis, for the transition / McNemar
    helpers (W: at-cap items scored wrong; X: the union of at-cap ids dropped)."""
    if axis == AXIS_STOCK:
        return {k: {"items": dict(v)} for k, v in legs.items()}
    if axis == AXIS_W:
        return {k: {"items": {i: ({**r, "scored": True, "value": 0.0} if r["at_cap"] else r) for i, r in v.items()}} for k, v in legs.items()}
    drop = excluded_ids(legs)
    return {k: {"items": {i: r for i, r in v.items() if i not in drop}} for k, v in legs.items()}


def excluded_ids(legs: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> list[str]:
    """X's exclusion: every id at cap in ANY leg of the contrast."""
    return sorted({i for v in legs.values() for i, r in v.items() if r["at_cap"]})


def estimands(legs: Mapping[str, Mapping[str, Mapping[str, Any]]], contrast: Mapping[str, int], name: str, *, n_resamples: int, seed: int,
              ci: float) -> dict[str, Any]:
    """``{axis: {branch: paired_bootstrap_conditional(...)}}`` plus the X exclusion."""
    drop = excluded_ids(legs)
    out: dict[str, Any] = {"n_excluded_X": len(drop), "excluded_X": drop}
    for axis in AXES:
        exclude = drop if axis == AXIS_X else ()
        out[axis] = {b: M.paired_bootstrap_conditional({k: branch_items(v, b, axis, exclude=exclude) for k, v in legs.items()},
                                                       {name: dict(contrast)}, n_resamples=n_resamples, seed=seed, ci=ci) for b in BRANCHES}
    return out


def verdict_for(est: Mapping[str, Any], name: str, *, mcnemar: Mapping[str, Mapping[str, Any]] | None, file_word: bool) -> dict[str, Any]:
    """The MATH-500 (a)-(d) rule on the W axis and on the X axis; the word only
    where both file the same directional word (plan P1-5)."""
    per: dict[str, Any] = {}
    for axis in (AXIS_W, AXIS_X, AXIS_STOCK):
        v = M.verdict_for(est[axis], name, mcnemar=(mcnemar or {}).get(axis), file_word=file_word)
        v["word"] = _WORD_MAP.get(v["word"], v["word"]) if v["word"] is not None else None
        per[axis] = v
    stock = per.pop(AXIS_STOCK)
    w, x = per[AXIS_W]["word"], per[AXIS_X]["word"]
    if not file_word:
        word, reason = None, "no word is filed on this comparison"
    elif w in WORDS and x in WORDS and w == x:
        word, reason = w, None
    elif w in WORDS and x in WORDS:
        word, reason = WORD_NOT_RESOLVED, f"W and X file opposite words ({w} / {x})"
    else:
        parts = [f"{axis}: {per[axis]['reason']}" for axis in (AXIS_W, AXIS_X) if per[axis]["word"] not in WORDS]
        word, reason = WORD_NOT_RESOLVED, "; ".join(parts) or "an axis filed no word"
    return {"word": word, "reason": reason, "W": per[AXIS_W], "X": per[AXIS_X], "stock": stock,
            "agree_W_X": w == x, "point_by_axis_branch": {a: {b: est[a][b]["stats"][name]["point"] for b in BRANCHES} for a in AXES},
            "ci_by_axis_branch": {a: {b: [est[a][b]["stats"][name]["ci_low"], est[a][b]["stats"][name]["ci_high"]] for b in BRANCHES} for a in AXES}}


def run_contrast(legs: Mapping[str, Mapping[str, Mapping[str, Any]]], contrast: Mapping[str, int], name: str, *, plus: str, minus: str,
                 n_resamples: int, seed: int, ci: float, file_word: bool) -> dict[str, Any]:
    est = estimands(legs, contrast, name, n_resamples=n_resamples, seed=seed, ci=ci)
    mc = {axis: T.mcnemar_by_branch(axis_items(legs, axis), minus, plus) for axis in AXES}
    for axis in AXES:
        T.attach_registered_replicate_signs({name: mc[axis]}, {name: (est[axis], name)})
    trans = {axis: T.transitions(axis_items(legs, axis), minus, plus) for axis in AXES}
    return {"estimands": est, "mcnemar": mc, "transitions": trans,
            "verdict": verdict_for(est, name, mcnemar=mc, file_word=file_word)}


# --------------------------------------------------------------------------- #
# the design layer: contrasts per view, the noise-floor pairs, the truncation cost #
# --------------------------------------------------------------------------- #


def _leg_reading(cells: Mapping[str, Any], cell_key: str, reading: str) -> Mapping[str, Mapping[str, Any]] | None:
    c = cells.get(cell_key)
    if c is None or c.get("readings") is None:
        return None
    return c["readings"][reading]["items"]


#: B5 (i) / (i″): the SHARED-FIELD projection every arm of a contrast must agree on and the NORMALISED server settings are
#: the MATH-500 module's — ONE spelling across both instruments (the attempted id set is compared beside them; the NUMERIC
#: sampling projection is `M.ROW_NUMERIC_FIELDS` of the manifest's sampling block; `source` prose never enters; the
#: row-contracted geometry is validated per row by `V.arm_contract_problem`, never compared)
SHARED_PROJECTION_FIELDS = M.SHARED_PROJECTION_FIELDS
COMPAT_RULE_ONE_TREE = "one producer tree per comparison (no declared tree set)"
COMPAT_RULE_DECLARED = ("shared-field projection + per-row arm contract + normalised server settings + rung bound to its own source + "
                        "producer membership")


def shared_projection(leg: Mapping[str, Any]) -> dict[str, Any]:
    """B5 (i): :func:`M.shared_projection` under THIS instrument's task id."""
    return M.shared_projection(leg, task=TASK)


server_projection = M.server_projection
#: this instrument's standup task subdirectory (`serving/<jobdir>/gsm8k/`) — the launch attestation is read from it (C2)
LAUNCH_TASK_SUBDIR = V.STANDUP_TASK_SUBDIR


def _rung_projection_problem(base: Mapping[str, Any], rung: Mapping[str, Any]) -> str | None:
    """B5 (ii): a rung leg's projection equals its OWN source base's except the cap statement (its subset ⊆ the source's
    eligible set and == the planner's file were held at collection: `V.rung_subset_problem` / `cap_rung_problem` /
    `item_subset_block_problem`) — the launch settings included (C2), which needs both legs' launch ATTESTED: an
    unattested pair cannot be bound and refuses by name (the record's ``bound_to_source`` reads ``None`` then)."""
    pb, pr_ = shared_projection(base), shared_projection(rung)
    pb.pop(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK, None)
    pr_.pop(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK, None)
    why = M._first_difference({"base": pb, "rung": pr_})
    if why:
        return f"{rung['leg']}: the rung's shared-field projection differs from its source base's — {why}"
    why = M.launch_unattested({"source base": base, "rung": rung}, {"source base": "source base", "rung": "rung"})
    if why:
        return f"{rung['leg']}: the rung cannot be bound to its source base on launch settings — {why}"
    # the capture-size carve (M.capture_sizes_by_derivation): a rung served at a lower S than its base — the kernel-ceiling
    # lowering at a wider window — echoes a different `cudagraph_capture_sizes` BY CONSTRUCTION (the composer derives the
    # list from S); the difference is admitted iff each side's list is the composer's derivation for its own attested S,
    # every other field walked exactly as before
    projections, capture_note = M.capture_sizes_by_derivation({"base": base, "rung": rung},
                                                              {"base": server_projection(base), "rung": server_projection(rung)})
    why = M._first_difference(projections)
    if why:
        note = f"; {capture_note}" if capture_note is not None and why.startswith(f"launch.{M.LAUNCH_CAPTURE_SIZES_KEY}: ") else ""
        return f"{rung['leg']}: the rung's normalised server settings differ from its source base's — {why}{note}"
    return None


def _launch_attested(*legs: Mapping[str, Any]) -> bool:
    return all(_mapping(leg.get("launch")).get("status") == M.LAUNCH_ATTESTED for leg in legs)


def _contrast_legs(cells: Mapping[str, Any], keys: Sequence[str]) -> tuple[dict[str, Mapping[str, Any]], list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]]:
    """``({cell key: base leg}, [(cell key, base leg, rung leg), ...])`` of a contrast's usable cells."""
    bases = {k: cells[k]["legs"][cells[k]["base_leg"]] for k in keys}
    rungs = [(k, bases[k], cells[k]["legs"][cells[k]["rung_leg"]]) for k in keys if cells[k]["rung_leg"] is not None]
    return bases, rungs


def _comparable(cells: Mapping[str, Any], keys: Sequence[str], *, declared: bool = False) -> str | None:
    """Why a set of cells has no joint estimand: a missing / refused cell, differing ids, or — without a DECLARED tree set —
    mixed producers (the published one-sha rule, verbatim: a 7-char pin admits two trees sharing a prefix). Under a
    declaration (B9 / plan P5) the contrast may span admitted trees when (i) the legs' shared-field projection is equal,
    (i′) every leg satisfied its own row's arm contract, (i″) the normalised server settings are equal, (ii) each rung is
    bound to ITS OWN source base and (iii) every leg's tree is admitted — each refusal naming the field."""
    for k in keys:
        c = cells.get(k)
        if c is None:
            return f"{k}: not in the design"
        base = c["legs"][c["base_leg"]]
        if not usable(base):
            return f"{k}: {base.get('missing') or base.get('refused')}"
        if c["rung_leg"] is not None and not usable(c["legs"][c["rung_leg"]]):
            return f"{k} rung: {c['legs'][c['rung_leg']].get('missing') or c['legs'][c['rung_leg']].get('refused')}"
    ids = {frozenset(cells[k]["legs"][cells[k]["base_leg"]]["attempted_ids"]) for k in keys}
    if len(ids) != 1:
        return "the cells' attempted id sets differ"
    bases, rungs = _contrast_legs(cells, keys)
    if not declared:
        shas = {k: bases[k]["producer"]["tree_sha"] for k in keys}
        for k, _b, rung in rungs:
            shas[k + "@rung"] = rung["producer"]["tree_sha"]
        if len(set(shas.values())) > 1:
            return f"the legs' producer trees differ — a mixed-producer comparison: {shas}"
        return None
    # (iii) producer membership — every leg's tree admitted (its producer record matched its root's pin or a declared tree)
    for k, base in bases.items():
        if not _mapping(base.get("producer")).get("matches_pin"):
            return f"{k}: its producer tree {_mapping(base.get('producer')).get('tree_sha')} is not an admitted tree"
    for k, _b, rung in rungs:
        if not _mapping(rung.get("producer")).get("matches_pin"):
            return f"{k} rung: its producer tree {_mapping(rung.get('producer')).get('tree_sha')} is not an admitted tree"
    # (i) the shared-field projection across the BASE legs
    why = M._first_difference({k: shared_projection(b) for k, b in bases.items()})
    if why:
        return f"the legs' shared-field projection differs — {why}"
    # (i″) the normalised server settings — the launch settings from each leg's ATTESTATION (C2): an unattested leg refuses
    why = M.launch_unattested(bases, {k: k for k in bases})
    if why:
        return f"the legs' launch settings cannot be compared — {why}"
    # the MATH-500 reader's rule, by import — one rule across both instruments: every echoed launch flag compared
    # LITERALLY except the server's default sampling (the --override-generation-config echo and the engine's override
    # WARNING), whose keys are REFUSED UNLESS UNDERSTOOD and then compared by EFFECT over the keys the requests do not
    # send (`effective_decode_problem`); the output limit by provenance. Before this the GSM8K reader compared the two
    # override records literally, so a `none` leg launched without the composed server default and an arm launched with it
    # refused although every request sends the sampling keys and the one omitted key resolves to the same value.
    labels = {k: k for k in bases}
    # launch-attestation-native: legs attested through DIFFERENT engines (a lens anchor beside a lora_native leg) name the
    # engine pair and EVERY differing key (M.server_settings_difference); same-engine legs the first, as before
    why = M.server_settings_difference(bases, labels)
    if why:
        return why
    why = M.effective_decode_problem(bases, labels, same_sampling=True)
    if why:
        return f"the legs' effective decode differs (D48) — {why}"
    # (i′) every leg's own arm contract held (checked at collection under a declaration; a failing one refused the leg)
    for k, base in bases.items():
        ac = _mapping(base.get("arm_contract"))
        if not ac.get("checked") or ac.get("problem"):
            return f"{k}: its row's arm contract was not validated ({ac.get('problem') or 'not checked'})"
    # (ii) each rung bound to its own source base
    for _k, base, rung in rungs:
        why = _rung_projection_problem(base, rung)
        if why:
            return why
    return None


def compatibility_record(cells: Mapping[str, Any], keys: Sequence[str], labels: Mapping[str, str], *, declared: bool) -> dict[str, Any]:
    """The per-contrast record (plan P5): each leg's tree / root / pin, the rungs with their source, whether the
    projection was equal and every arm's contract held, and which rule admitted the contrast. Strings and bools only (B1)."""
    bases, rungs = _contrast_legs(cells, keys)

    def where(leg: Mapping[str, Any]) -> dict[str, Any]:
        prod = _mapping(leg.get("producer"))
        return {"leg": leg["leg"], "tree": prod.get("tree_sha"), "root": prod.get("root"), "pin": prod.get("pin")}

    legs = {labels[k]: where(bases[k]) for k in keys}

    def bound(base: Mapping[str, Any], rung: Mapping[str, Any]) -> tuple[bool | None, str | None]:
        # C2: True / False only when both legs' launch is attested; None (with the reason) when the pair cannot be bound
        why = _rung_projection_problem(base, rung)
        if why is None:
            return True, None
        return (None if not _launch_attested(base, rung) else False), why

    rung_recs = []
    for k, base, rung in rungs:
        ok, why = bound(base, rung)
        rung_recs.append({**where(rung), "label": labels[k], "source": base["leg"], "bound_to_source": ok, "bound_to_source_reason": why})
    trees = {v["tree"] for v in legs.values()} | {r["tree"] for r in rung_recs}
    return {"legs": legs, "rungs": rung_recs, "cross_tree": len(trees) > 1, "declared": bool(declared),
            "projection_equal": M._first_difference({k: shared_projection(b) for k, b in bases.items()}) is None,
            **M.server_settings_verdict(bases, {k: k for k in bases}),
            "arm_contracts_valid": all(_mapping(b.get("arm_contract")).get("checked") and not _mapping(b.get("arm_contract")).get("problem")
                                       for b in bases.values()),
            "rule": COMPAT_RULE_DECLARED if declared else COMPAT_RULE_ONE_TREE}


def _compat_or_none(cells: Mapping[str, Any], keys: Sequence[str], labels: Mapping[str, str], *, declared: bool) -> dict[str, Any] | None:
    """The compatibility record when every named cell's legs are usable (else the contrast's `reason` says why)."""
    for k in keys:
        c = cells.get(k)
        if c is None or not usable(c["legs"][c["base_leg"]]) or (c["rung_leg"] is not None and not usable(c["legs"][c["rung_leg"]])):
            return None
    return compatibility_record(cells, keys, labels, declared=declared)


def design_contrasts(matrix: Mapping[str, Any], cells: Mapping[str, Any], *, refusals: Mapping[str, str], n_resamples: int, seed: int,
                     ci: float, declared: bool = False) -> list[dict[str, Any]]:
    """Every launch row that is not the view's ``none``: ``cell - none(view)``
    under both readings. ``refusals`` is THE ONE resolved expected-refusal
    mapping (:func:`load_expected_refusals`: the matrix rows' field and the
    ``--expected-refusals`` file together, keyed by the 8,192 leg) — the same
    object the completeness gate, the figures and the publication read, so a
    file-only refusal is filed as expected here too, never as "did not run"."""
    out: list[dict[str, Any]] = []
    for r in matrix["launch"]:
        if r["cell"] == CELL_NONE:
            continue
        ck, nk = V.cell_key(r["model_key"], r["view"], r["cell"]), V.cell_key(r["model_key"], r["view"], CELL_NONE)
        rec: dict[str, Any] = {"name": f"{r['model_key']}/{r['view']}: {r['cell']} - none", "kind": "contrast", "model_key": r["model_key"],
                               "view": r["view"], "cell": r["cell"], "cell_key": ck, "none_key": nk, "fitted": bool(r["fitted"]),
                               "declared": bool(r["declared"]), "expected_refusal": refusals.get(V.leg_key(r["model_key"], r["view"], r["cell"], CAP_BASE)),
                               # the driver's certified refusal (collect_leg: evidence.json says render_certified false, no summary):
                               # the contrast is not computable BY NAME, like an expected refusal — never "did not run"
                               "certified_refusal": (REFUSAL_CERTIFIED_UNSERVABLE
                                                     if cells[ck]["legs"][cells[ck]["base_leg"]].get("certified_unservable") else None),
                               "computable": False, "reason": None, "escalated": None, "readings": {},
                               # E3: an it-doc reading carries its lane and the pair's qualification VERBATIM (the row's; None otherwise)
                               "object_source": r.get("object_source"), **V.itdoc_fields(r),
                               # plan P5: the compatibility record (strings / bools): each leg's tree, root and pin, the rungs and
                               # which rule admitted the contrast
                               "compatibility": _compat_or_none(cells, [nk, ck], {nk: "minus", ck: "plus"}, declared=declared)}
        # the DECLARED expected refusal of THIS contrast (the row's expected_contrast_refusal): recorded only when the row declares one,
        # so every undeclared record is byte-identical; a code the reader does not derive is a named problem, never a match
        declared_code = r.get(ROW_EXPECTED_CONTRAST_REFUSAL_KEY)
        if declared_code is not None:
            rec[ROW_EXPECTED_CONTRAST_REFUSAL_KEY] = str(declared_code)
            if str(declared_code) not in CONTRAST_REFUSAL_CODES:
                rec[EXPECTED_CONTRAST_REFUSAL_PROBLEM_KEY] = (f"{ROW_EXPECTED_CONTRAST_REFUSAL_KEY} {declared_code!r} is not a refusal code this "
                                                              f"reader derives ({list(CONTRAST_REFUSAL_CODES)})")
        problem = _comparable(cells, [nk, ck], declared=declared)
        if problem:
            rec["reason"] = problem
            code = contrast_refusal_code(rec)
            if code is not None:
                rec[CONTRAST_REFUSAL_CODE_KEY] = code
            if declared_code is not None and EXPECTED_CONTRAST_REFUSAL_PROBLEM_KEY not in rec and code == str(declared_code):
                # refused BY DESIGN: the refusal derives exactly the declared code — a filed outcome (the contrast stays refused with its
                # text, never computed); the production-result rule and the verdict read this flag, not the text
                rec[CONTRAST_REFUSED_BY_DESIGN_KEY] = True
            out.append(rec)
            continue
        rec["computable"] = True
        rec["escalated"] = bool(cells[ck]["readings"][READING_MERGED]["escalated"] or cells[nk]["readings"][READING_MERGED]["escalated"])
        for reading in READINGS:
            legs = {"none": _leg_reading(cells, nk, reading), "cell": _leg_reading(cells, ck, reading)}
            rec["readings"][reading] = run_contrast(legs, {"cell": +1, "none": -1}, CONTRAST_VS_NONE, plus="cell", minus="none",
                                                    n_resamples=n_resamples, seed=seed, ci=ci, file_word=True)
        out.append(rec)
    return out


def noise_floor_pairs(matrix: Mapping[str, Any], cells: Mapping[str, Any], *, n_resamples: int, seed: int, ci: float,
                      declared: bool = False) -> list[dict[str, Any]]:
    """``none(it) - none(main)`` for every model the matrix runs in both views."""
    out: list[dict[str, Any]] = []
    keys = sorted({r["model_key"] for r in matrix["launch"] if r["view"] == V.VIEW_IT} & {r["model_key"] for r in matrix["launch"] if r["view"] == V.VIEW_MAIN})
    for key in keys:
        ik, mk = V.cell_key(key, V.VIEW_IT, CELL_NONE), V.cell_key(key, V.VIEW_MAIN, CELL_NONE)
        rec: dict[str, Any] = {"name": f"{key}: none(it) - none(main)", "kind": "noise_floor", "model_key": key, "it_key": ik, "main_key": mk,
                               "computable": False, "reason": None, "readings": {},
                               "compatibility": _compat_or_none(cells, [mk, ik], {mk: "minus", ik: "plus"}, declared=declared)}
        problem = _comparable(cells, [mk, ik], declared=declared)
        if problem:
            rec["reason"] = problem
            out.append(rec)
            continue
        rec["computable"] = True
        for reading in READINGS:
            legs = {"it": _leg_reading(cells, ik, reading), "main": _leg_reading(cells, mk, reading)}
            rec["readings"][reading] = run_contrast(legs, {"it": +1, "main": -1}, CONTRAST_NOISE, plus="it", minus="main",
                                                    n_resamples=n_resamples, seed=seed, ci=ci, file_word=True)
        out.append(rec)
    return out


def truncation_costs(cells: Mapping[str, Any], *, n_resamples: int, seed: int, ci: float) -> list[dict[str, Any]]:
    """Per escalated cell: ``merged - r8192`` over the cell's 200 items (the stock
    axis under both think branches) and the residual at the ceiling."""
    out: list[dict[str, Any]] = []
    for ck in sorted(cells):
        c = cells[ck]
        if not c.get("readings") or not c["readings"][READING_MERGED]["escalated"]:
            continue
        legs = {"merged": c["readings"][READING_MERGED]["items"], "r8192": c["readings"][READING_R8192]["items"]}
        est = {b: M.paired_bootstrap_conditional({k: branch_items(v, b, AXIS_STOCK) for k, v in legs.items()},
                                                 {CONTRAST_TRUNCATION: {"merged": +1, "r8192": -1}}, n_resamples=n_resamples, seed=seed, ci=ci)
               for b in BRANCHES}
        redraws = c["redraws"] or {}
        out.append({"name": f"{ck}: merged - r8192", "cell_key": ck, "model_key": c["model_key"], "view": c["view"], "cell": c["cell"],
                    "thinking_mode": c["legs"][c["base_leg"]].get("thinking_mode"),
                    "n_subset": c["ladder"]["n_subset"], "estimands": est,
                    "residual_at_ceiling": sum(1 for r in redraws.values() if r["still_at_cap"]),
                    "n_extended": sum(1 for r in redraws.values() if r["extended"]),
                    "n_boundary_landing": sum(1 for r in redraws.values() if r["boundary_landing"]),
                    "n_flipped_to_correct": sum(1 for r in redraws.values() if r["new_scored"] and r["new_value"] == 1.0 and not (r["source_scored"] and r["source_value"] == 1.0)),
                    "n_flipped_to_wrong": sum(1 for r in redraws.values() if r["source_scored"] and r["source_value"] == 1.0 and not (r["new_scored"] and r["new_value"] == 1.0)),
                    "point_primary": est[BRANCH_PRIMARY]["stats"][CONTRAST_TRUNCATION]["point"],
                    "ci_primary": [est[BRANCH_PRIMARY]["stats"][CONTRAST_TRUNCATION]["ci_low"], est[BRANCH_PRIMARY]["stats"][CONTRAST_TRUNCATION]["ci_high"]]})
    return out


# --------------------------------------------------------------------------- #
# the gates A0-A10                                                             #
# --------------------------------------------------------------------------- #


def _gate(status: str, detail: str | None = None, **more: Any) -> dict[str, Any]:
    return T._gate(status, detail, **more)


def _all_legs(cells: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {lk: leg for c in cells.values() for lk, leg in c["legs"].items()}


RESOLVED_SAMPLER_KEY = "resolved_sampler"
RESOLVED_SAMPLER_NOTE = ("the sampler the ENGINE resolved per request (vLLM's own log under --enable-log-requests, read back by "
                         "scripts/sdfbase/record_resolved_sampler.py into <out root>/resolved_sampler.json after the driver; PR #50, "
                         "adopted 2026-09-12): per usable leg the recorder's verdict against the row's sampling row — a FINDING beside "
                         "the gates, never a gate; a store produced before the recorder rode the launcher is NOT RECORDED, and says so")


def resolved_sampler_block(legs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """The report's attestation block over every usable leg (``V.resolved_sampler_record``): counts by verdict, the legs with
    a binding problem, and the per-leg record."""
    fields = ("present", "verdict", "why", "problem", "n_client_requests", "n_probe_requests", "n_unclassified_requests",
              "reference", "client_caps", "divergences", "path")
    per: dict[str, dict[str, Any]] = {}
    for lk, leg in legs.items():
        rs = leg.get(RESOLVED_SAMPLER_KEY)
        if isinstance(rs, Mapping):
            per[lk] = {k: rs.get(k) for k in fields}
    if not any(r["present"] for r in per.values()):
        # no leg carries the recorder's record (stores produced before the recorder rode the launcher — the published wave's):
        # no block, so a report over those stores keeps its frozen shape; the Llama wave's legs are recorded and get the block
        return None
    return {"note": RESOLVED_SAMPLER_NOTE, "filename": V.RESOLVED_SAMPLER_FILENAME, "n_legs": len(per),
            "n_recorded": sum(1 for r in per.values() if r["present"]),
            "n_not_recorded": sum(1 for r in per.values() if not r["present"]),
            "verdicts": dict(sorted(Counter(str(r["verdict"]) for r in per.values() if r["present"]).items())),
            "n_problems": sum(1 for r in per.values() if r["problem"]),
            "legs_not_agreeing": sorted(lk for lk, r in per.items() if r["present"] and (r["verdict"] != V.RESOLVED_SAMPLER_AGREES or r["problem"])),
            "legs": per}


def _usable_legs(cells: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {k: leg for k, leg in _all_legs(cells).items() if usable(leg)}


def gate_completeness(matrix: Mapping[str, Any], cells: Mapping[str, Any], refusals: Mapping[str, str]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    """A0: every launch row's 8,192 leg present and usable (200 unique attempted
    ids, rows == n_scored, summary == outcomes, the .eval re-derivation
    agreeing — each enforced at collection) unless it is an EXPECTED refusal
    (filed by name with its code); every escalated cell's rung leg usable; the
    held rows never legs. Returns the gate, the refused block, the unexpected
    unusable legs."""
    refused: list[dict[str, Any]] = []
    unusable: dict[str, str] = {}
    findings: list[str] = []
    per: dict[str, Any] = {}
    for r in matrix["launch"]:
        ck = V.cell_key(r["model_key"], r["view"], r["cell"])
        c = cells[ck]
        base = c["legs"][c["base_leg"]]
        code = refusals.get(c["base_leg"])
        if usable(base):
            per[c["base_leg"]] = {"status": "usable", "n_attempted": len(base["attempted_ids"]), "eval_check": (base.get("eval_check") or {}).get("agrees"),
                                  V.RECORD_GAPS_KEY: len(base.get(V.RECORD_GAPS_KEY) or [])}
            if code:
                findings.append(f"{c['base_leg']}: named as an expected refusal ({code}) but a usable store is present — the contract fix landed; read normally")
        elif base.get("certified_unservable"):
            # the DRIVER refused the cell's thinking-ON render by name (predicate (A); evidence.json, no summary): an
            # admitted outcome counted beside the operator's expected refusals; the verdict proceeds without this cell
            cert = _mapping(base.get("render_certification"))
            per[c["base_leg"]] = {"status": LEG_STATUS_REFUSED_CERTIFIED, "code": cert.get("code"), "reason": base["refused"],
                                  "attempt_classes": list(cert.get("attempt_classes") or []), "certifying_draw_class": cert.get("certifying_draw_class"),
                                  "expected_code": code}
            refused.append({"leg": c["base_leg"], "cell_key": ck, "model_key": r["model_key"], "view": r["view"], "cell": r["cell"],
                            "code": cert.get("code"), "status": LEG_STATUS_REFUSED_CERTIFIED,
                            "reason": "refused by the driver's predicate (A) render certification: " + str(cert.get("problems_joined")),
                            "attempt_classes": list(cert.get("attempt_classes") or []), "on_attempts": list(cert.get("on_attempts") or []),
                            "certifying_draw_class": cert.get("certifying_draw_class"), "observation": cert.get("observation"),
                            "expected_code": code, "store": base.get("no_summary")})
        elif base.get("missing") and code:
            per[c["base_leg"]] = {"status": LEG_STATUS_REFUSED_EXPECTED, "code": code, "reason": base["missing"]}
            refused.append({"leg": c["base_leg"], "cell_key": ck, "model_key": r["model_key"], "view": r["view"], "cell": r["cell"], "code": code,
                            "status": LEG_STATUS_REFUSED_EXPECTED,
                            "reason": "refused by name at the standup (plan §3.3: a header-located object on a header-invariant family)",
                            "store": base["missing"]})
        else:
            per[c["base_leg"]] = {"status": "missing" if base.get("missing") else "refused", "reason": base.get("missing") or base.get("refused")}
            unusable[c["base_leg"]] = str(base.get("missing") or base.get("refused"))
        if c["rung_leg"] is not None:
            rung = c["legs"][c["rung_leg"]]
            per[c["rung_leg"]] = {"status": "usable" if usable(rung) else ("missing" if rung.get("missing") else "refused"),
                                  "reason": rung.get("missing") or rung.get("refused"), "n_attempted": len(rung.get("attempted_ids") or [])}
            if not usable(rung):
                unusable[c["rung_leg"]] = str(rung.get("missing") or rung.get("refused"))
    n_launch = len(matrix["launch"])
    detail = ("; ".join(f"{k}: {why}" for k, why in unusable.items()) or None)
    status = GATE_FAIL if unusable else GATE_PASS
    base_keys = {cells[V.cell_key(r["model_key"], r["view"], r["cell"])]["base_leg"] for r in matrix["launch"]}
    return (_gate(status, detail, legs=per, n_launch_rows=n_launch,
                  n_usable_base=sum(1 for k, p in per.items() if k in base_keys and p["status"] == "usable"),
                  n_usable_rung=sum(1 for k, p in per.items() if k not in base_keys and p["status"] == "usable"),
                  n_refused_expected=sum(1 for x in refused if x.get("status") == LEG_STATUS_REFUSED_EXPECTED),
                  n_refused_certified=sum(1 for x in refused if x.get("status") == LEG_STATUS_REFUSED_CERTIFIED),
                  n_unusable=len(unusable), n_held=len(matrix["held"]), findings=findings),
            refused, unusable)


def _per_leg_gate(cells: Mapping[str, Any], key: str) -> dict[str, Any]:
    failures, pending, stamps = [], [], []
    per: dict[str, Any] = {}
    for k, leg in _usable_legs(cells).items():
        rec = leg[key]
        per[k] = {f: v for f, v in rec.items() if f not in ("attempts",)} if key == "ledger" else rec
        failures.extend(f"{k}: {f}" for f in rec.get("failures", []))
        for s in rec.get("pending_stamps", []):
            pending.append(f"{k}: {s} not stamped")
            stamps.append(s)
    return M._fold_gate(per, failures, pending, stamps)


def gate_ledger(cells: Mapping[str, Any]) -> dict[str, Any]:
    """A1: the accepted-attempt ledger criterion of every usable leg."""
    g = _per_leg_gate(cells, "ledger")
    g["attempts_by_leg"] = {k: leg["ledger"].get("attempts") for k, leg in _usable_legs(cells).items()}
    return g


def gate_identity(cells: Mapping[str, Any]) -> dict[str, Any]:
    """A2: the manifest-level identity refuses a leg at collection (-> A0); here the
    request evidence per class and attempt, the stamp triple, the sampling digest."""
    failures, pending, stamps = [], [], []
    per: dict[str, Any] = {}
    for k, leg in _usable_legs(cells).items():
        per[k] = {"request_evidence": {f: v for f, v in (leg.get("request_evidence") or {}).items()}, "stamps": leg.get("stamps"),
                  "sampling_ref": leg.get("sampling_ref"), "identity_validated": leg.get("identity_validated")}
        failures += [f"{k}: request evidence: {f}" for f in (leg.get("request_evidence") or {}).get("failures", [])]
        failures += [f"{k}: stamps: {f}" for f in (leg.get("stamps") or {}).get("failures", [])]
        sr = leg.get("sampling_ref") or {}
        if sr.get("status") == GATE_FAIL:
            failures.append(f"{k}: {sr['detail']}")
        elif sr.get("status") == GATE_PENDING:
            pending.append(f"{k}: {sr['detail']}")
            stamps.extend(sr.get("pending_stamps", []))
    return M._fold_gate(per, failures, pending, stamps)


def gate_certification(cells: Mapping[str, Any]) -> dict[str, Any]:
    return _per_leg_gate(cells, "certification")


def stray_tag_record(leg: Mapping[str, Any]) -> dict[str, Any]:
    """The ``none`` split's STRAY-TAG reading of one leg (the reader's rule
    ``contracts.no_switch_stray_tag_problems``): every draw whose outcome
    record flags a stray tag (``stray_open_tag`` / ``stray_close_tag``) or a
    think block, or whose row records the provider's client-side lift
    (``client_lift_fired`` / a non-empty ``lifted_reasoning``), named.
    PRIMARY: any such draw FAILS the leg closed (a no-thinking-channel family
    emitted reasoning evidence; the scorer saw the raw text, but the
    instrument is not the one the design named). SENSITIVITY: the same leg
    with those draws MASKED out of the scored set, its accuracy beside the
    stock one and the masked count — reported, never the headline."""
    items = leg["items"]
    problems = C.no_switch_stray_tag_problems(items, lifted=leg.get("rows_lifted"))
    # The raw row content, read independently of every flag (collect_leg's no_switch_row_evidence)
    problems += [f"{iid}: raw row content carries think-tag literal(s) {tags} on a no-thinking-channel leg — a stray-tag anomaly the flags did not name"
                 for iid, tags in sorted(_mapping(leg.get("rows_raw_tags")).items()) if tags]
    masked = sorted({p.split(":", 1)[0] for p in problems})
    masked_set = set(masked)
    kept = {i: rec for i, rec in items.items() if i not in masked_set}
    scored = [rec for rec in kept.values() if rec["scored"]]
    n_correct = sum(1 for rec in scored if rec["value"] == 1.0)
    scored_all = [rec for rec in items.values() if rec["scored"]]
    thinking = leg.get("thinking") or {}
    return {"rule": M.NO_REASONING_RULE[C.THINK_SPLIT_NONE], "n_stray": len(masked), "stray_ids": masked[:50], "problems": problems[:50],
            "n_problems": len(problems), "n_stray_tag_draws_filed": thinking.get("n_stray_tag_draws"),
            "primary": {"fail_closed": bool(masked)},
            "sensitivity": {"n_masked": len(masked), "n": len(kept), "n_scored": len(scored), "n_correct": n_correct,
                            "acc_stock_masked": (n_correct / len(scored)) if scored else None,
                            "acc_stock": (sum(1 for rec in scored_all if rec["value"] == 1.0) / len(scored_all)) if scored_all else None}}


def gate_scoring_path(cells: Mapping[str, Any]) -> dict[str, Any]:
    """A4: the primary unscored policy, the family's split mode and parser, no
    model-graded companion; a present .eval log the reader venv could not use is PENDING (unwaivable).

    Under the ``none`` split (llama-paper-rows P19 / P21) the PRIMARY rule fails
    a leg CLOSED on any stray-tag anomaly (:func:`stray_tag_record`: a think-tag
    literal in a completion, a client-lifted channel, the provider's
    ``client_lift_fired``), each draw named; the SENSITIVITY reading (the same
    leg with those draws masked, the masked count) is filed beside it. A
    ``reasoning_parser`` on such a family refuses too (there is no channel to
    parse). The switch arms' legs read exactly as before."""
    failures, pending = [], []
    per: dict[str, Any] = {}
    for k, leg in _usable_legs(cells).items():
        s, fact = leg["summary"], leg["render_fact"]
        ec = leg.get("eval_check") or {}
        per[k] = {"unterminated_policy": s.get("unterminated_policy"), "think_split_mode": s.get("think_split_mode"), "reasoning_parser": s.get("reasoning_parser"),
                  "grader_max_tokens": s.get("grader_max_tokens"), "task_rows_on_other_vmodel": (leg.get("ledger") or {}).get("task_rows_on_other_vmodel"),
                  "eval_check": {f: ec.get(f) for f in ("available", "agrees", "log_present", "reason")}}
        if fact["split_mode"] == C.THINK_SPLIT_NONE:
            rec = stray_tag_record(leg)
            per[k]["thinking_mode"] = leg.get("thinking_mode")
            per[k]["stray_tag"] = rec
            if rec["primary"]["fail_closed"]:
                shown = "; ".join(rec["problems"][:3]) + (f"; and {rec['n_problems'] - 3} more" if rec["n_problems"] > 3 else "")
                failures.append(f"{k}: {rec['n_stray']} stray-tag draw(s) on a no-thinking-channel leg (PRIMARY fails closed; the sensitivity "
                                f"reading masks them: acc {rec['sensitivity']['acc_stock_masked']!r} over {rec['sensitivity']['n_scored']} scored) — {shown}")
            if rec["n_stray_tag_draws_filed"] is not None and rec["n_stray_tag_draws_filed"] != rec["n_stray"]:
                failures.append(f"{k}: the thinking record files {rec['n_stray_tag_draws_filed']} stray-tag draw(s) but the reader's rule names "
                                f"{rec['n_stray']} — two readings of one record")
            if s.get("reasoning_parser") is not None:
                failures.append(f"{k}: reasoning_parser {s.get('reasoning_parser')!r} on a family with no reasoning channel")
        if s.get("unterminated_policy") != C.UNTERMINATED_UNSCORED:
            failures.append(f"{k}: unterminated_policy {s.get('unterminated_policy')!r} != {C.UNTERMINATED_UNSCORED!r}")
        if s.get("think_split_mode") != fact["split_mode"]:
            failures.append(f"{k}: think_split_mode {s.get('think_split_mode')!r} != the {leg['family']!r} fact's {fact['split_mode']!r}")
        if s.get("reasoning_parser") != fact["server_parser"]:
            failures.append(f"{k}: reasoning_parser {s.get('reasoning_parser')!r} != the family's {fact['server_parser']!r}")
        if s.get("grader_max_tokens") is not None:
            failures.append(f"{k}: grader_max_tokens {s.get('grader_max_tokens')!r} — a model-graded companion ran; GSM8K is judge-free")
        other = per[k]["task_rows_on_other_vmodel"] or {}
        if other:
            failures.append(f"{k}: {sum(other.values())} ledger call(s) of {TASK} on another virtual model {dict(other)}")
        if ec.get("available") is False and ec.get("log_present") is True:
            pending.append(f"{k}: independent .eval re-derivation not run: {ec.get('reason')}")
    return M._fold_gate(per, failures, pending, [None] * len(pending))


def gate_at_cap(cells: Mapping[str, Any]) -> dict[str, Any]:
    """A5: THIS instrument's ladder record on every usable 8,192 leg (rule /
    threshold / report line; every at-cap draw classified; the eligible set ==
    the rung subset when escalated; the log-text re-derivation agreeing where read)."""
    failures = []
    per: dict[str, Any] = {}
    for ck in sorted(cells):
        c = cells[ck]
        if not usable(c["legs"][c["base_leg"]]) or c.get("ladder") is None:
            continue
        lad = c["ladder"]
        per[ck] = {f: lad[f] for f in ("record_present", "record_problem", "exit", "n_at_cap", "share_at_cap", "over_1pct", "triggered",
                                       "n_non_terminating", "n_boundary", "n_eligible", "escalated", "subset_problem", "n_subset")}
        if lad["record_problem"]:
            failures.append(f"{ck}: {lad['record_problem']}")
        if lad["escalated"] and lad["subset_problem"]:
            failures.append(f"{ck}: {lad['subset_problem']}")
    return _gate(GATE_FAIL if failures else GATE_PASS, "; ".join(failures) or None, cells=per,
                 n_over_1pct=sum(1 for p in per.values() if p["over_1pct"]), n_triggered=sum(1 for p in per.values() if p["triggered"]),
                 n_escalated=sum(1 for p in per.values() if p["escalated"]))


def gate_ladder(gen_root: Path, cells: Mapping[str, Any]) -> dict[str, Any]:
    """A6: a triggered cell has EITHER a usable rung store over exactly its
    subset (its ``cap_rung`` bound to the source) OR a terminal record
    (``triggered_no_eligible`` / ``transport_refused``); an un-triggered cell has
    no rung store; the merged reading is the source with the redraws substituted."""
    failures = []
    per: dict[str, Any] = {}
    for ck in sorted(cells):
        c = cells[ck]
        base = c["legs"][c["base_leg"]]
        if not usable(base) or c.get("ladder") is None:
            continue
        lad = c["ladder"]
        rec: dict[str, Any] = {"triggered": lad["triggered"], "exit": lad["exit"], "escalated": lad["escalated"], "rung_store_present": lad["rung_store_present"],
                               "rung_leg": c["rung_leg"], "rung_usable": (usable(c["legs"][c["rung_leg"]]) if c["rung_leg"] else None),
                               "n_subset": lad["n_subset"], "merged_n": c["readings"][READING_MERGED]["n"], "merged_from_rung": c["readings"][READING_MERGED]["n_from_rung"]}
        per[ck] = rec
        if lad["triggered"]:
            if lad["exit"] == CL.ESCALATE:
                if c["rung_leg"] is None or not rec["rung_usable"]:
                    failures.append(f"{ck}: escalated but the rung leg is {'absent' if c['rung_leg'] is None else 'not usable'}: "
                                    f"{(c['legs'].get(c['rung_leg']) or {}).get('missing') or (c['legs'].get(c['rung_leg']) or {}).get('refused') or lad['subset_problem']}")
                elif rec["merged_from_rung"] != lad["n_subset"]:
                    failures.append(f"{ck}: the merged reading substitutes {rec['merged_from_rung']} draws != the subset's {lad['n_subset']}")
            elif lad["exit"] not in V.EXITS_TRIGGERED_TERMINAL:
                failures.append(f"{ck}: triggered ({lad['n_at_cap']} at cap) with exit {lad['exit']!r} — neither a rung nor a terminal record")
            elif lad["rung_store_present"]:
                failures.append(f"{ck}: a rung store exists beside a terminal record {lad['exit']!r}")
        else:
            if lad["rung_store_present"]:
                failures.append(f"{ck}: an un-triggered cell ({lad['n_at_cap']} at cap) has a rung store")
            if lad["exit"] not in V.EXITS_NOT_TRIGGERED and lad["exit"] is not None:
                failures.append(f"{ck}: an un-triggered cell carries exit {lad['exit']!r}")
        if rec["merged_n"] != N_ITEMS:
            failures.append(f"{ck}: the merged reading holds {rec['merged_n']} items != {N_ITEMS}")
    return _gate(GATE_FAIL if failures else GATE_PASS, "; ".join(failures) or None, cells=per)


COMPAT_REFUSAL_MARKERS: tuple[str, ...] = ("producer", "admitted tree", "shared-field projection", "normalised server settings", "arm contract")


def gate_population(cells: Mapping[str, Any], contrasts: Sequence[Mapping[str, Any]], noise: Sequence[Mapping[str, Any]],
                    canonical_ids: Sequence[str] | None, *, declared: bool = False) -> dict[str, Any]:
    """A7: every usable 8,192 leg's attempted ids == the canonical 200 (derived
    once, independently; absent -> PENDING, an operator input never waived); a
    rung leg's == its subset ⊆ canonical; one producer tree across each cell's
    legs and each computable comparison's legs — or, under a DECLARED tree set
    (B9), both legs on ADMITTED trees with the rung bound to its own source and
    every contrast admitted by the compatibility relations."""
    failures = []
    per: dict[str, Any] = {}
    if canonical_ids is None:
        return _gate(GATE_PENDING, "no canonical id set supplied (--ids); the population's identity is not verified", canonical="not supplied")
    for ck in sorted(cells):
        c = cells[ck]
        base = c["legs"][c["base_leg"]]
        if not usable(base):
            continue
        problem = V.population_problem(base["attempted_ids"], canonical_ids)
        per[c["base_leg"]] = {"canonical": problem is None, "problem": problem}
        if problem:
            failures.append(f"{c['base_leg']}: {problem}")
        if c["rung_leg"] is not None and usable(c["legs"][c["rung_leg"]]):
            rung = c["legs"][c["rung_leg"]]
            problem = V.population_problem(rung["attempted_ids"], canonical_ids, subset_ids=c["ladder"]["subset_ids"])
            per[c["rung_leg"]] = {"canonical": problem is None, "problem": problem}
            if problem:
                failures.append(f"{c['rung_leg']}: {problem}")
            shas = {base["leg"]: base["producer"]["tree_sha"], rung["leg"]: rung["producer"]["tree_sha"]}
            if not declared and len(set(shas.values())) > 1:
                failures.append(f"{ck}: the 8,192 and rung legs come from different producer trees {shas}")
            elif declared:
                if not (base["producer"].get("matches_pin") and rung["producer"].get("matches_pin")):
                    failures.append(f"{ck}: the 8,192 and rung legs are not both on admitted producer trees {shas}")
                problem = _rung_projection_problem(base, rung)
                if problem:
                    failures.append(f"{ck}: {problem}")
    for rec in [*contrasts, *noise]:
        if rec["reason"] and any(m in rec["reason"] for m in COMPAT_REFUSAL_MARKERS):
            failures.append(f"{rec['name']}: {rec['reason']}")
    return _gate(GATE_FAIL if failures else GATE_PASS, "; ".join(failures) or None, legs=per, canonical="verified" if not failures else "mismatch",
                 n_canonical=len(set(canonical_ids)))


def gate_two_branch(contrasts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A8: every computable contrast carries its W / X x primary / score-as-is
    verdicts under both readings; the word only where all agree. A computable
    contrast without an estimand FAILS; a non-computable one is filed by name."""
    failures = []
    per: dict[str, Any] = {}
    n_words = 0
    for rec in contrasts:
        if not rec["computable"]:
            per[rec["name"]] = {"computable": False, "reason": rec["reason"],
                                **({CONTRAST_REFUSED_BY_DESIGN_KEY: True, ROW_EXPECTED_CONTRAST_REFUSAL_KEY: rec[ROW_EXPECTED_CONTRAST_REFUSAL_KEY]}
                                   if rec.get(CONTRAST_REFUSED_BY_DESIGN_KEY) else {})}
            continue
        entry: dict[str, Any] = {"computable": True}
        for reading in READINGS:
            v = (rec["readings"].get(reading) or {}).get("verdict")
            if v is None:
                failures.append(f"{rec['name']}: no verdict under the {reading} reading")
                continue
            entry[reading] = {"word": v["word"], "agree_W_X": v["agree_W_X"], "W": v["W"]["word"], "X": v["X"]["word"], "stock": v["stock"]["word"]}
            if reading == READING_MERGED and v["word"] in WORDS:
                n_words += 1
        per[rec["name"]] = entry
    return _gate(GATE_FAIL if failures else GATE_PASS, "; ".join(failures) or None, contrasts=per, n_computable=sum(1 for r in contrasts if r["computable"]),
                 n_designed=len(contrasts), n_words_filed_merged=n_words)


def gate_report_match(match: Mapping[str, Any] | None) -> dict[str, Any]:
    """A9: the two-venv match — enforced at the production boundary
    (``--require-report-match`` exits 4 on a differing numeric leaf); the record
    of this pass's comparison when it ran."""
    if match is None:
        return _gate(GATE_PASS, "enforced at the production boundary (--require-report-match exits 4 on a differing numeric leaf); "
                                "not compared in this pass", compared=False)
    # no numeric leaf here: the second pass files this record AFTER its match, and the post-hoc --diff-reports of the two
    # written reports must still find them equal (the counts live under the skipped `report_match` block)
    return _gate(GATE_PASS if match["equal"] else GATE_FAIL, None if match["equal"] else f"{len(match['differing'])} differing numeric leaf(s)",
                 compared=True, against=match["against"], equal=bool(match["equal"]))


def _same_pin(a: str, b: str) -> bool:
    """Two store-pin spellings of ONE wave sha (7-40 lowercase hex each, the
    runner's 7-char form or the full commit): one is a prefix of the other."""
    return a.startswith(b) or b.startswith(a)


OVERLAP_STATUS_NOT_APPLICABLE = "not_applicable"
OVERLAP_SCHEMA = "gsm8k_overlap/1"


def _not_applicable_block_problems(b: Mapping[str, Any], *, key: str, family: str | None, store_pin: str) -> list[str]:
    """The NO-SWITCH family's overlap block (P12; ``gsm8k_overlap_check.py``'s
    ``not_applicable`` record): its schema, its status, ONE sentence saying why,
    this wave's pin, the model key and the family it was written for."""
    out: list[str] = []
    if b.get("schema") != OVERLAP_SCHEMA:
        out.append(f"{key}: not_applicable block schema {b.get('schema')!r} != {OVERLAP_SCHEMA!r}")
    if not isinstance(b.get("why"), str) or not b.get("why").strip():
        out.append(f"{key}: not_applicable block carries no `why`")
    pin = b.get("pin")
    if not isinstance(pin, str) or M.pin_problem(pin) or not _same_pin(pin, str(store_pin)):
        out.append(f"{key}: not_applicable block pin {pin!r} is not this analysis's store pin {store_pin!r}")
    if b.get("model_key") != key:
        out.append(f"{key}: not_applicable block model_key {b.get('model_key')!r} != {key!r}")
    if b.get("family") != family:
        out.append(f"{key}: not_applicable block family {b.get('family')!r} != the model's {family!r}")
    return out


def overlap_pin_under_store_pins(models: Sequence[str], pins: Mapping[str, str]) -> str:
    """The pin the overlap record is bound to under per-row store pins: the overlap check is a per-MODEL fact about the
    ``none`` leg's instrument (the retired thinking-OFF reading vs the stock thinking-OFF leg), so it binds to the ADMITTED pin
    of each model's ``none@8192`` leg (its ``main`` view, else its ``it`` view). Every model's none pin must be ONE tree here —
    models whose none legs come from different trees refuse by name (the record would need a pin per model)."""
    found: dict[str, str] = {}
    for m in models:
        pin = pins.get(V.leg_key(m, VIEW_MAIN, CELL_NONE, CAP_BASE)) or pins.get(V.leg_key(m, VIEW_IT, CELL_NONE, CAP_BASE))
        if pin is None:
            raise AnalysisError(f"--store-pins names no `none@{CAP_BASE}` leg for {m!r} — the overlap record's binding pin is undefined for it")
        found[m] = pin
    distinct = sorted(set(found.values()))
    if len(distinct) != 1:
        raise AnalysisError(f"the models' `none` legs are admitted at different trees {found} — the overlap record binds to ONE pin; "
                            "read the models with one none-tree per run")
    return distinct[0]


def gate_overlap(overlap: Mapping[str, Any] | None, models: Sequence[str], canonical_ids: Sequence[str] | None, *,
                 store_pin: str) -> dict[str, Any]:
    """A10: the overlap record present (schema, the canonical ids' digest),
    BOUND TO THIS WAVE, with a usable, COMPLETE (200 pairs) block and a verdict
    for every model of the design. The binding (review of 337efb7a, HIGH): the
    record's own ``store_pin`` must be this analysis's store pin (the runner
    passes ONE ``--sha`` to the checker and to run E; a record from another
    wave pairs the same 200 ids and would otherwise pass), and every model's
    stock leg must carry a 40-hex ``producer_tree_sha`` under that pin with the
    checker's recorded ``store_pin`` beside it — a record from before those
    provenance fields (the 2eff115b checker) refuses by name rather than
    passing on ids alone. And every stock block's
    ``outcomes_producer_tree_sha`` — the tree that wrote the values the
    comparison consumed — must be a 40-hex sha equal to that block's
    ``producer_tree_sha``; a record from a checker that never recorded it
    (before db8642eb) is refused with the instruction to regenerate it.

    llama-paper-rows (P12): a model whose family is in
    ``contracts.NO_THINKING_CHANNEL_FAMILIES`` has NO overlap to check (the
    retired probe presupposes a thinking-OFF arm this family does not have) and
    its block is the checker's EXPLICIT ``not_applicable`` record
    (:func:`_not_applicable_block_problems`) — the gate PASSES with per-model
    status ``not_applicable`` for exactly those models; a real block on such a
    model, or a ``not_applicable`` block on a switch family, FAILS by name."""
    if overlap is None:
        return _gate(GATE_FAIL, "no overlap record supplied (--overlap; scripts/gsm8k_overlap_check.py writes it)", models={})
    failures = []
    per: dict[str, Any] = {}
    not_applicable: list[str] = []
    if overlap.get("schema") != OVERLAP_SCHEMA:
        failures.append(f"overlap record schema {overlap.get('schema')!r} != {OVERLAP_SCHEMA!r}")
    if canonical_ids is not None:
        want = hashlib.sha256(json.dumps(sorted(set(canonical_ids))).encode("utf-8")).hexdigest()
        if overlap.get("ids_sha256") != want:
            failures.append("the overlap record's ids_sha256 is not the canonical id set's digest — paired against another selection")
    report_pin = overlap.get("store_pin")
    if not isinstance(report_pin, str) or M.pin_problem(report_pin):
        failures.append(f"the overlap record carries no usable store_pin ({report_pin!r}) — a record from before the pin binding; "
                        f"regenerate it (scripts/gsm8k_overlap_check.py --sha {store_pin})")
    elif not _same_pin(report_pin, str(store_pin)):
        failures.append(f"the overlap record's store_pin {report_pin!r} is not this analysis's store pin {store_pin!r} — a record from "
                        "ANOTHER wave (the same 200 ids pair either way; the pin is what tells the waves apart)")
    blocks = _mapping(overlap.get("models"))
    for key in models:
        b = _mapping(blocks.get(key))
        family = C.FAMILY_BY_MODEL_KEY.get(key)
        no_switch = family in C.NO_THINKING_CHANNEL_FAMILIES
        if b.get("status") == OVERLAP_STATUS_NOT_APPLICABLE or (no_switch and b):
            per[key] = {"present": bool(b), "status": b.get("status"), "why": b.get("why"), "pin": b.get("pin"), "family": family,
                        "usable": None, "n_pairs": None, "complete": None, "verdict": None}
            if not no_switch:
                failures.append(f"{key}: a {OVERLAP_STATUS_NOT_APPLICABLE} overlap block on family {family!r}, which HAS a thinking switch — the "
                                "overlap check applies; regenerate the record with its real block")
                continue
            if b.get("status") != OVERLAP_STATUS_NOT_APPLICABLE:
                failures.append(f"{key}: a REAL overlap block on the no-switch family {family!r} — no thinking-OFF arm exists to overlap-check; "
                                f"the record must carry the checker's {OVERLAP_STATUS_NOT_APPLICABLE} block (P12)")
                continue
            problems = _not_applicable_block_problems(b, key=key, family=family, store_pin=store_pin)
            failures.extend(problems)
            if not problems:
                not_applicable.append(key)
            continue
        cmp_ = _mapping(b.get("comparison"))
        stock = _mapping(b.get("stock"))
        tree, stock_pin, outcomes_tree = stock.get("producer_tree_sha"), stock.get("store_pin"), stock.get("outcomes_producer_tree_sha")
        per[key] = {"present": bool(b), "usable": b.get("usable"), "n_pairs": cmp_.get("n_pairs"), "complete": cmp_.get("complete"),
                    "verdict": b.get("verdict"), "delta": cmp_.get("delta"), "ci": [cmp_.get("ci_low"), cmp_.get("ci_high")] if cmp_ else None,
                    "reason": b.get("reason"), "producer_tree_sha": tree, "outcomes_producer_tree_sha": outcomes_tree, "stock_store_pin": stock_pin}
        if not b:
            failures.append(f"{key}: no overlap block" + (f" (the no-switch family {family!r} needs the checker's {OVERLAP_STATUS_NOT_APPLICABLE} block)"
                                                          if no_switch else ""))
            continue
        per[key]["status"] = "checked"
        if not b.get("usable"):
            failures.append(f"{key}: overlap not usable — {b.get('reason')}")
        elif not cmp_.get("complete"):
            failures.append(f"{key}: {cmp_.get('n_pairs')} of {N_ITEMS} items paired — {cmp_.get('reason')}")
        elif b.get("verdict") not in ("continuation", "new_instrument"):
            failures.append(f"{key}: no overlap verdict filed")
        # the stock leg's PRODUCER, held to this analysis's pin here — never trusted from the record's own verdict
        if not isinstance(tree, str) or not _HEX40.match(tree):
            failures.append(f"{key}: the overlap record's stock block carries no 40-hex producer_tree_sha ({tree!r}) — a record from before the "
                            "provenance fields; regenerate it")
        elif not tree.startswith(str(store_pin)):
            failures.append(f"{key}: the overlap record's stock producer tree {tree} does not carry this analysis's store pin {store_pin!r} — "
                            "the stock leg of ANOTHER wave")
        if not isinstance(stock_pin, str) or not _same_pin(stock_pin, str(store_pin)):
            failures.append(f"{key}: the overlap record's stock block was read under store_pin {stock_pin!r}, not this analysis's {store_pin!r}")
        # the OUTCOMES record's own producer — the tree that wrote the values the comparison consumed — must be recorded and
        # be the manifest's tree (a same-pin record generated by a checker that never asked, from a store whose outcomes
        # another tree wrote, reads usable and advertises only the manifest's tree)
        if not isinstance(outcomes_tree, str) or not _HEX40.match(outcomes_tree):
            failures.append(f"{key}: the overlap record's stock block carries no 40-hex outcomes_producer_tree_sha ({outcomes_tree!r}) — the record "
                            "predates the outcomes-producer check (scripts/gsm8k_overlap_check.py); regenerate the overlap record with the checker "
                            "of this tree")
        elif outcomes_tree != tree:
            failures.append(f"{key}: the overlap record's outcomes producer tree {outcomes_tree} != its manifest producer tree {tree} — the values "
                            "the comparison consumed were written by another tree")
    return _gate(GATE_FAIL if failures else GATE_PASS, "; ".join(failures) or None, models=per, store_pin=str(store_pin), report_store_pin=report_pin,
                 not_applicable=not_applicable)


def pending_items(gates: Mapping[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for gate in GATE_NAMES:
        g = gates.get(gate)
        if not isinstance(g, Mapping) or g.get("status") != GATE_PENDING:
            continue
        named = g.get("pending")
        if isinstance(named, list) and named:
            stamps = g.get("pending_stamps") or [None] * len(named)
            for detail, stamp in zip(named, stamps if len(stamps) == len(named) else [None] * len(named)):
                items.append({"gate": gate, "detail": detail, "stamps": [stamp], "waivable": stamp is not None and stamp in M.WAIVABLE_PENDING_STAMPS})
        else:
            items.append({"gate": gate, "detail": str(g.get("detail")), "stamps": [None], "waivable": False})
    return items


def gates_failed(gates: Mapping[str, Any]) -> list[str]:
    """The FAILED gates read off the individual gate RECORDS (A0..A10), never
    off the ``failed`` aggregate: the aggregate is a convenience written from
    these, and a gate re-filed after it (A9 at the production boundary) must
    not be able to fall out of it."""
    return sorted(g for g in GATE_NAMES if _mapping(gates.get(g)).get("status") == GATE_FAIL)


def gates_pending(gates: Mapping[str, Any]) -> list[str]:
    return sorted(g for g in GATE_NAMES if _mapping(gates.get(g)).get("status") == GATE_PENDING)


def refresh_gate_aggregates(gates: dict[str, Any]) -> dict[str, Any]:
    """Re-derive ``failed`` / ``pending`` / ``pending_items`` / ``pending_unwaivable``
    from the gate records — called after every (re)filing of a gate."""
    gates["failed"] = gates_failed(gates)
    gates["pending"] = gates_pending(gates)
    gates["pending_items"] = pending_items(gates)
    gates["pending_unwaivable"] = [it for it in gates["pending_items"] if not it["waivable"]]
    return gates


def verdict_of(gates: Mapping[str, Any], contrasts: Sequence[Mapping[str, Any]], words: Mapping[str, Mapping[str, str | None]], *,
               allow_pending_gates: bool) -> dict[str, Any]:
    """The report's verdict from the gate RECORDS and the contrasts: withheld
    on a contrast that did not run, on any FAILED gate, on a PENDING gate that
    is not waivable (or any PENDING gate unless allowed); else the merged words."""
    not_run = [c["name"] for c in _not_run(contrasts)]
    declaration_problems = contrast_declaration_problems(contrasts)
    failed, pending = gates_failed(gates), gates_pending(gates)
    unwaivable = [it for it in pending_items(gates) if not it["waivable"]]
    if not_run:
        verdict: dict[str, Any] = {"word": None, "verdict": NOT_MEASURABLE, "reason": f"contrast(s) did not run: {not_run}"}
    elif declaration_problems:
        verdict = {"word": None, "verdict": NOT_MEASURABLE, "reason": "expected contrast refusal(s) not met: " + "; ".join(declaration_problems)}
    elif failed:
        verdict = {"word": None, "verdict": NOT_MEASURABLE, "reason": "gate(s) FAILED: " + "; ".join(f"{g}: {gates[g]['detail']}" for g in failed)}
    elif pending and (not allow_pending_gates or unwaivable):
        verdict = {"word": None, "verdict": NOT_MEASURABLE + " (pending)", "reason": "gate(s) PENDING: " + "; ".join(f"{g}: {gates[g].get('detail')}" for g in pending),
                   "pending_unwaivable": unwaivable}
    else:
        verdict = {"word": "filed", "verdict": "; ".join(f"{n}: {w[READING_MERGED]}" for n, w in words.items()), "reason": None}
    verdict["words"] = dict(words)
    return verdict


def top_items_of(report: Mapping[str, Any]) -> list[str]:
    """What to read first: every FAILED gate with its detail (from the gate
    RECORDS), every unwaivable PENDING gate, every contrast that filed no
    estimand, every cell over the trigger, then the merged words filed."""
    gates = _mapping(report.get("gates"))
    findings = list(report.get("findings") or [])
    out = [f"{g}: FAILED — {_mapping(gates.get(g)).get('detail')}" for g in gates_failed(gates)]
    out += [f"{it['gate']}: PENDING (not waivable) — {it['detail']}" for it in pending_items(gates) if not it["waivable"]]
    out += [f for f in findings if "no estimand" in f]
    out += [f for f in findings if f"refused by name ({REFUSAL_CERTIFIED_UNSERVABLE})" in f]
    out += [f for f in findings if ": RECORD GAP — " in f]
    out += [f for f in findings if _FINDING_TRIGGER.search(f)]
    words = _mapping(_mapping(report.get("verdict")).get("words"))
    out += [f"{n}: {w[READING_MERGED]}" for n, w in words.items() if isinstance(w, Mapping) and w.get(READING_MERGED) in WORDS]
    return out


# --------------------------------------------------------------------------- #
# the report                                                                   #
# --------------------------------------------------------------------------- #

_CONTROLS_SUMMARY_FIELDS: tuple[str, ...] = (
    "acc", "n", "n_paired", "n_scored", "n_unscored", "at_cap_rate", "n_at_cap", "n_stop_reason", "mean_completion_tokens", "p99_completion_tokens",
    "n_think_block_present", "n_think_block_unterminated", "n_unterminated_not_at_cap", "n_content_empty_with_reasoning", "think_split_mode",
    "unterminated_policy", "max_tokens", "reasoning_parser", "temperature", "top_p", "top_k", "samples_per_item", "wall_clock_s", "n_requests",
    "n_http_rejected_retried", "render_regime", "salt_regime", "resumed_policy", "item_set")
_LEDGER_CONTROL_FIELDS: tuple[str, ...] = (
    "accepted_attempt", "n_calls", "n_unique_samples", "expected_calls", "n_rejected", "n_task_rows_all_attempts", "n_stray_rows", "error_code",
    "error_class", "n_length", "window", "decode_tok_s_mean", "decode_tok_s_aggregate", "completion_tokens_quantiles", "passed")


def controls_table(cells: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for ck in sorted(cells):
        c = cells[ck]
        for lk in ([c["base_leg"]] + ([c["rung_leg"]] if c["rung_leg"] else [])):
            leg = c["legs"][lk]
            s, led = leg.get("summary") or {}, leg.get("ledger") or {}
            rows.append({"leg": lk, "cell_key": ck, "job_id": leg.get("job_id"), "model_key": leg["model_key"], "view": leg["view"], "cell": leg["cell"],
                         "cap": leg["cap"], "rung": leg["rung"], "fitted": leg["fitted"], "declared": leg["declared"], "missing": leg.get("missing"),
                         "refused": leg.get("refused"), "summary": {f: s.get(f) for f in _CONTROLS_SUMMARY_FIELDS}, "at_cap": leg.get("at_cap"),
                         "thinking": leg.get("thinking"), "ledger": {f: led.get(f) for f in _LEDGER_CONTROL_FIELDS},
                         # llama-paper-rows: the row's ARM and its label on every table row; the per-row provenance when emitted
                         "thinking_mode": leg.get("thinking_mode"), "thinking_label": leg.get("thinking_label"),
                         "row_thinking": leg.get("row_thinking"), "row_provenance": leg.get("row_provenance"),
                         "attempts": {aid: {f: a.get(f) for f in ("kind", "completed", "n_forwarded", "n_rejected", "n_probe_rows", "accepted")}
                                      for aid, a in (led.get("attempts") or {}).items()},
                         "text": (leg.get("eval_check") or {}).get("text"), "provenance": provenance_row(leg),
                         "stamps": (leg.get("stamps") or {}).get("observed"), "render_fact": leg.get("render_fact")})
    return rows


def provenance_row(leg: Mapping[str, Any]) -> dict[str, Any]:
    ident = leg.get("identity") or {}
    return {"job_id": leg.get("job_id"), "model_key": leg["model_key"], "view": leg["view"], "cell": leg["cell"], "cap": leg["cap"],
            "store_key": leg.get("store_key"), "harness_config_hash": ident.get("harness_config_hash"), "intervention_hash": ident.get("intervention_hash"),
            "producer_tree_sha": ident.get("producer_tree_sha"), "model_id": ident.get("model_id"), "model_revision": ident.get("model_revision"),
            "object_artifact_sha256": ident.get("object_artifact_sha256"), "tuple_name": leg.get("tuple_name"),
            C.MANIFEST_KEY_AUTHORISED_EXPERIMENT: ident.get(C.MANIFEST_KEY_AUTHORISED_EXPERIMENT), "resume_attempts": ident.get(C.MANIFEST_KEY_RESUME_ATTEMPTS),
            "accepted_attempt": ident.get(V.MANIFEST_KEY_ACCEPTED_ATTEMPT), "cap_rung": ident.get(V.IDENTITY_KEY_CAP_RUNG), "producer": leg.get("producer"),
            C.IDENTITY_KEY_IDENTITY_VERSION: ident.get(C.IDENTITY_KEY_IDENTITY_VERSION), "route_kind": leg.get("route_kind"),
            "served_positions": ((leg.get("certification") or {}).get("route_lens_served_positions")) if leg.get("fitted") else None,
            "served_positions_source": ((leg.get("certification") or {}).get("route_lens_served_positions_source")) if leg.get("fitted") else None,
            V.RECORD_GAPS_KEY: [dict(g) for g in (leg.get(V.RECORD_GAPS_KEY) or [])],
            # the per-row identity keys (strings) — `pin` is the leg's producer pin, `root` its gen root
            "thinking": (leg.get("summary") or {}).get("thinking") or ident.get("thinking_mode"),
            "sampling_ref": _mapping(leg.get("sampling_ref")).get("expected"),
            "sampling_mode": _mapping(ident.get("sampling")).get("mode"),
            "pin": _mapping(leg.get("producer")).get("pin"), "root": _mapping(leg.get("producer")).get("root"),
            "wave": leg.get("wave"), "status": leg.get("status"), "arm": arm_kind_of(str(leg["cell"])),
            "thinking_mode": leg.get("thinking_mode"), "thinking_label": leg.get("thinking_label"), "row_thinking": leg.get("row_thinking"),
            "row_provenance": dict(leg.get("row_provenance") or {"present": False, **{f: None for f in C.ROW_PROVENANCE_FIELDS}}),
            # E3: the object source and the it-doc carriage (lane, qualification verbatim) on every provenance row (None otherwise)
            "object_source": leg.get("object_source"), **{f: leg.get(f) for f in V.ITDOC_ROW_FIELDS}}


_TRUNCATION_CELL_AT_CAP_FIELDS: tuple[str, ...] = ("share", "n_at_cap", "n_with_stop_reason", "line_1pct_exceeded", "trigger_exceeded",
                                                    "n_case_b_unterminated_at_cap", "n_case_c_answer_at_cap", "split_source", "n_loop",
                                                    "n_truncated_reasoning", "n_text_unavailable")


def truncation_block(cells: Mapping[str, Any], costs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The report's at-cap RESULT block: every usable 8,192 leg's share against
    the 1 % line and the trigger, the loop split, the ladder exit, and the
    truncation cost of every escalated cell."""
    out_cells: list[dict[str, Any]] = []
    for ck in sorted(cells):
        c = cells[ck]
        leg = c["legs"][c["base_leg"]]
        if not usable(leg):
            continue
        ac, lad = leg["at_cap"], c["ladder"] or {}
        out_cells.append({"leg": leg["leg"], "cell_key": ck, "model_key": leg["model_key"], "view": leg["view"], "cell": leg["cell"],
                          "thinking_mode": leg.get("thinking_mode"),
                          **{f: ac[f] for f in _TRUNCATION_CELL_AT_CAP_FIELDS},
                          "periodic_block_n_loop": ac["periodic_block_column"]["n_loop"],
                          "exit": lad.get("exit"), "escalated": lad.get("escalated"), "n_eligible": lad.get("n_eligible"),
                          "n_non_terminating": lad.get("n_non_terminating"), "n_subset": lad.get("n_subset"),
                          "rung_planned_under": lad.get("rung_planned_under"),
                          "residual_at_ceiling": (next((t["residual_at_ceiling"] for t in costs if t["cell_key"] == ck), None))})
    return {"line_1pct": V.GSM8K_REPORT_LINE, "trigger_threshold": V.GSM8K_TRIGGER_THRESHOLD, "cap": CAP_BASE, "ceiling": CAP_RUNG,
            "rule": V.GSM8K_TRIGGER_RULE, "trigger_provenance": V.GSM8K_TRIGGER_PROVENANCE, "loop_rule": LOOP_RULE_TEXT, "periodic_block_rule": PERIODIC_RULE_TEXT,
            "table_header": f"at-cap share per cell at the {CAP_BASE:,} cap — {V.GSM8K_TRIGGER_PROVENANCE}",
            # the pairs a rung identity / subset file may carry, and which pair each escalated cell's rung was planned under
            "rules_admitted_on_rungs": {V.rung_planned_under(r): {"rule": r, "threshold": t} for r, t in V.GSM8K_TRIGGER_RULES_ADMITTED.items()},
            "rungs_planned_under": {lab: [c["leg"] for c in out_cells if c["escalated"] and c["rung_planned_under"] == lab]
                                    for lab in (V.RUNG_PLANNED_UNDER_CURRENT, V.RUNG_PLANNED_UNDER_RETIRED_3PCT)},
            "n_cells_read": len(out_cells), "over_1pct": [c["leg"] for c in out_cells if c["line_1pct_exceeded"]],
            "triggered": [c["leg"] for c in out_cells if c["trigger_exceeded"]], "escalated": [c["leg"] for c in out_cells if c["escalated"]],
            "terminal_no_rung": [c["leg"] for c in out_cells if c["trigger_exceeded"] and not c["escalated"]],
            "max_share": max((c["share"] for c in out_cells if c["share"] is not None), default=None), "cells": out_cells,
            "costs": [{k: v for k, v in t.items() if k != "estimands"} for t in costs]}


def _contrast_summary(rec: Mapping[str, Any]) -> dict[str, Any]:
    out = {k: rec[k] for k in ("name", "kind", "model_key", "computable", "reason")}
    out.update({k: rec.get(k) for k in ("view", "cell", "escalated", "expected_refusal", "certified_refusal", "fitted", "declared")})
    out.update({k: rec[k] for k in (ROW_EXPECTED_CONTRAST_REFUSAL_KEY, EXPECTED_CONTRAST_REFUSAL_PROBLEM_KEY, CONTRAST_REFUSAL_CODE_KEY,
                                    CONTRAST_REFUSED_BY_DESIGN_KEY) if k in rec})
    # E3: the it-doc carriage (object source, lane, the qualification verbatim) — None on every other contrast / noise-floor pair
    out.update({k: rec.get(k) for k in ("object_source", *V.ITDOC_ROW_FIELDS)})
    for reading in READINGS:
        r = (rec.get("readings") or {}).get(reading)
        if r is None:
            continue
        name = next(iter(r["estimands"][AXIS_STOCK][BRANCH_PRIMARY]["stats"]))
        out[reading] = {"word": r["verdict"]["word"], "reason": r["verdict"]["reason"], "agree_W_X": r["verdict"]["agree_W_X"],
                        "point": r["verdict"]["point_by_axis_branch"], "ci": r["verdict"]["ci_by_axis_branch"],
                        "n_excluded_X": r["estimands"]["n_excluded_X"],
                        "acc_by_leg_stock_primary": r["estimands"][AXIS_STOCK][BRANCH_PRIMARY]["acc_by_leg"],
                        "mcnemar_p_stock_primary": r["mcnemar"][AXIS_STOCK][BRANCH_PRIMARY]["p_value"], "contrast": name}
    return out


# --------------------------------------------------------------------------- #
# the eight-arm extension: the mode switch, the design block, the pooled-by-arm  #
# estimands over the main view, the Llama peer report                            #
# --------------------------------------------------------------------------- #


def eight_arm_mode(rows: Sequence[Mapping[str, Any]], waves: Sequence[str] | None, roots: M.GenRoots, *, peers: bool = False) -> bool:
    """B2 (GSM8K reading): ``--waves`` given, a matrix carrying any ``wave`` field, a ``deferred`` status or any status
    outside the published ``launch`` / ``held`` pair, a pinned / repeated gen root, or a peer report. Outside this mode every
    output keeps the published shape (the frozen 1 % report reproduces leaf for leaf)."""
    return (waves is not None or roots.declared or peers
            or any(r.get("wave") is not None or r["status"] not in (STATUS_LAUNCH, STATUS_HELD) for r in rows))


ROW_STATUS_REQUIRED, ROW_STATUS_NOT_ANALYSED, ROW_STATUS_ABSENT = M.ROW_STATUS_REQUIRED, M.ROW_STATUS_NOT_ANALYSED, M.ROW_STATUS_ABSENT


def arm_kind_of(cell: str) -> str:
    """The eight-arm registry's kind of a cell, or ``appendix`` (the PSR cells, the safety LoRA — never pooled, no figure)."""
    return M.CELL_KIND.get(cell, ARM_KIND_APPENDIX)


def design_status_block(rows: Sequence[Mapping[str, Any]], required: Sequence[Mapping[str, Any]], waves: Sequence[str] | None) -> dict[str, Any]:
    """The eight-arm keys of ``report["design"]``: every matrix row's status (``required`` /
    ``not_analysed`` / ``held`` / ``deferred`` with its reason), the analysed waves, the eight fixed arm slots per model
    on the ``main`` view, and the pooled panel's membership. The EXPECTED LEG SET (``required_legs``) is filed by
    :func:`analyze` once the ladder records are read — derived, never a typed count."""
    req = {V.cell_key(str(r["model_key"]), str(r["view"]), str(r["cell"])) for r in required}
    status_by_row: dict[str, dict[str, Any]] = {}
    for r in rows:
        k = V.cell_key(str(r["model_key"]), str(r["view"]), str(r["cell"]))
        if k in req:
            status, reason = ROW_STATUS_REQUIRED, None
        elif r["status"] in EVIDENCE_STATUSES:
            status, reason = ROW_STATUS_NOT_ANALYSED, f"wave {V.row_wave(r)!r} is not among the analysed waves"
        else:
            status = str(r["status"])
            reason = r.get("held_reason") or r.get("deferred_reason") or r.get("reason") or f"{status} row (no reason recorded)"
        status_by_row[k] = {"status": status, "wave": V.row_wave(r), "reason": reason, "model_key": r["model_key"], "view": r["view"],
                            "cell": r["cell"], "arm": arm_kind_of(str(r["cell"])), "object_source": r.get("object_source"),
                            # E3: an it-doc row's lane and the pair's qualification VERBATIM (the row's; None on any other row)
                            **V.itdoc_fields(r)}
    models = list(dict.fromkeys(str(r["model_key"]) for r in rows))
    design_arm_list = M.design_arm_roster({str(r["cell"]) for r in rows})   # the declared roster + the arms these rows carry
    arms_by_model: dict[str, dict[str, str]] = {}
    for m in models:
        arms_by_model[m] = {}
        for arm in design_arm_list:
            rec = status_by_row.get(V.cell_key(m, VIEW_MAIN, arm.cell))
            arms_by_model[m][arm.kind] = rec["status"] if rec else ROW_STATUS_ABSENT
    # E3 (additive): the same eight slots per (model, VIEW) the matrix carries — `arms_by_model` stays the `main` view's (the
    # eight-arm panel's); the `it` view's slots (the it-doc re-fit arms live there) are visible here with their statuses
    views_by_model = {m: list(dict.fromkeys(str(r["view"]) for r in rows if str(r["model_key"]) == m)) for m in models}
    arms_by_model_by_view: dict[str, dict[str, dict[str, str]]] = {}
    for m in models:
        arms_by_model_by_view[m] = {}
        for view in views_by_model[m]:
            arms_by_model_by_view[m][view] = {}
            for arm in design_arm_list:
                rec = status_by_row.get(V.cell_key(m, view, arm.cell))
                arms_by_model_by_view[m][view][arm.kind] = rec["status"] if rec else ROW_STATUS_ABSENT
    counts = Counter(v["status"] for v in status_by_row.values())
    return {"waves": (list(waves) if waves is not None else [WAVE_PUBLISHED]), "waves_in_matrix": sorted({V.row_wave(r) for r in rows}),
            "n_rows": len(rows), "n_required": counts.get(ROW_STATUS_REQUIRED, 0), "n_not_analysed": counts.get(ROW_STATUS_NOT_ANALYSED, 0),
            "n_held": counts.get(STATUS_HELD, 0), "n_deferred": counts.get(STATUS_DEFERRED, 0),
            "status_by_row": status_by_row, "modes": [MODE_PRIMARY], "arms": [a.cell for a in design_arm_list],
            "arm_kinds": [a.kind for a in design_arm_list],
            "arms_by_model": arms_by_model, "eightarm_view": VIEW_MAIN, "arms_by_model_by_view": arms_by_model_by_view,
            "panel": {"models": list(POOLED_PANEL_MODELS), "held_out": dict(POOLED_HELD_OUT), "not_in_panel": list(POOLED_NOT_IN_PANEL),
                      "view": VIEW_MAIN},
            "rule": ("the required set = the evidence rows (launch / complete) of the analysed waves, every one mandatory (a missing "
                     "leg fails the completeness gate); held / deferred rows and evidence rows of other waves are statuses here and "
                     "are never collected; the expected leg set is derived from the required rows, the ladder records (a rung per "
                     "escalated cell) and the certified refusals — never a typed count")}


def expected_leg_set(required: Sequence[Mapping[str, Any]], cells: Mapping[str, Any]) -> list[str]:
    """The EXPECTED LEG SET, derived: every required row's base leg + a rung leg for every cell whose ladder record exits
    ``escalate`` (and whose subset the planner filed) — a certified refusal is its base leg, filed by name."""
    out: list[str] = []
    for r in required:
        c = cells[V.cell_key(r["model_key"], r["view"], r["cell"])]
        out.append(c["base_leg"])
        if c["rung_leg"] is not None:
            out.append(c["rung_leg"])
        elif c.get("ladder") and c["ladder"].get("escalated"):
            out.append(V.leg_key(r["model_key"], r["view"], r["cell"], CAP_RUNG))
    return sorted(out)


POOLED_ESTIMAND_TEXT = ("per arm, over the MAIN view only (the it view never enters): the EQUAL-WEIGHT mean (1/3 each) of the three "
                        "panel models' stock conditional accuracies on that arm (each leg's own scored denominator), the same mean over "
                        "none, and their difference; `_base` reads each cell's 8,192-only draws, `_merged` its ladder-merged reading (the "
                        "16,384 redraw for every eligible truncating item); B replicates, ONE item-resampling index vector per replicate "
                        "applied to every model's leg (the generator re-seeded per call: a pure function of (seed, n, B)); percentile "
                        "CIs; both think-block branches; `unavailable` with the reason whenever any panel model's leg for the arm (or "
                        "its none) is missing, refused, held, deferred, absent or on an unadmitted tree — never a mean over fewer "
                        "models; llama held out (thinking: not available), GLM not in the panel")


def _pooled_leg(cell: Mapping[str, Any], leg_key: str, reading: str) -> dict[str, Any]:
    """The reading's items as the pooled bootstrap consumes a leg: ``{leg, attempted_ids, items}``."""
    items = cell["readings"][reading]["items"]
    return {"leg": leg_key, "attempted_ids": sorted(items), "items": items}


def eightarm_block(cells: Mapping[str, Any], design: Mapping[str, Any], *, n_resamples: int, seed: int, ci: float,
                   declared: bool = False) -> dict[str, Any]:
    """``report["eightarm"]``: ``pooled_by_arm_base`` and ``pooled_by_arm_merged`` per arm kind over the MAIN view of the
    panel models — :func:`M.pooled_arm_bootstrap` when every panel model's leg for the arm (and none) is usable under
    that reading AND each model's arm↔none PAIR passes the same policy every contrast runs through
    (:func:`_comparable` over ``[none, arm]`` — without it a contrast the comparison had refused still pooled as
    ``available``), else ``unavailable`` naming each model's status (or ``incompatible with <model>/main/none — <why>``)
    with no point or CI field — plus the panel membership, the hold-out and the estimand text. The figures DRAW these
    fields and compute nothing. ``declared``: the run's ``ctx.declared``."""
    status_by_row = _mapping(design.get("status_by_row"))

    def state(m: str, cell: str, reading: str) -> tuple[Mapping[str, Any] | None, str | None]:
        ck = V.cell_key(m, VIEW_MAIN, cell)
        c = cells.get(ck)
        if c is None:
            rec = status_by_row.get(ck)
            return None, (rec["status"] if rec else ROW_STATUS_ABSENT)
        base = c["legs"][c["base_leg"]]
        if base.get("missing"):
            return None, f"missing ({base['missing']})"
        if base.get("refused"):
            return None, f"refused ({base['refused']})"
        if not _mapping(base.get("producer")).get("matches_pin"):
            return None, "its producer tree is not an admitted tree"
        if reading == READING_MERGED and c["rung_leg"] is not None and not usable(c["legs"][c["rung_leg"]]):
            rung = c["legs"][c["rung_leg"]]
            return None, f"rung {'missing' if rung.get('missing') else 'refused'} ({rung.get('missing') or rung.get('refused')})"
        if reading == READING_MERGED and c["ladder"] and c["ladder"].get("escalated") and c["rung_leg"] is None:
            return None, f"escalated but no usable rung ({c['ladder'].get('subset_problem')})"
        return _pooled_leg(c, c["base_leg"], reading), None

    out: dict[str, Any] = {"view": VIEW_MAIN, "panel_models": list(POOLED_PANEL_MODELS), "held_out": dict(POOLED_HELD_OUT),
                           "not_in_panel": list(POOLED_NOT_IN_PANEL), "arms": [a.kind for a in M.design_arms(design)],
                           "readings": list(READINGS)}
    for reading, field in ((READING_R8192, "pooled_by_arm_base"), (READING_MERGED, "pooled_by_arm_merged")):
        bare: dict[str, Mapping[str, Any]] = {}
        bare_problems: list[str] = []
        for m in POOLED_PANEL_MODELS:
            leg, why = state(m, CELL_NONE, reading)
            if leg is None:
                bare_problems.append(f"{m}/{VIEW_MAIN}/{CELL_NONE}: {why}")
            else:
                bare[m] = leg
        pooled: dict[str, Any] = {}
        for arm in M.design_arms(design):           # the arms this analysis covers (design.arms), never the live registry
            arm_legs: dict[str, Mapping[str, Any]] = {}
            problems = list(bare_problems)
            pairs: dict[str, dict[str, Any]] = {}
            for m in POOLED_PANEL_MODELS:
                leg, why = state(m, arm.cell, reading)
                if leg is None:
                    problems.append(f"{m}/{VIEW_MAIN}/{arm.cell}: {why}")
                    continue
                if m not in bare:
                    continue                    # the none problem is already named
                # B3: the arm↔none pair under the SAME policy every contrast runs through (none↔none is the identity pair)
                nk, ck = V.cell_key(m, VIEW_MAIN, CELL_NONE), V.cell_key(m, VIEW_MAIN, arm.cell)
                why = _comparable(cells, [nk, ck], declared=declared) if arm.cell != CELL_NONE else None
                base_tree = _mapping(cells[ck]["legs"][cells[ck]["base_leg"]].get("producer")).get("tree_sha")
                none_tree = _mapping(cells[nk]["legs"][cells[nk]["base_leg"]].get("producer")).get("tree_sha")
                pairs[m] = {"compatible": why is None, "reason": why, "cross_tree": base_tree != none_tree}
                if why:
                    problems.append(f"{m}/{VIEW_MAIN}/{arm.cell}: incompatible with {m}/{VIEW_MAIN}/{CELL_NONE} — {why}")
                else:
                    arm_legs[m] = leg
            rec: dict[str, Any] = {"cell": arm.cell, "label": arm.label, "models": list(POOLED_PANEL_MODELS), "reading": reading, "pairs": pairs,
                                   "pair_rule": ("each panel model's arm↔none pair passes the report's contrast policy "
                                                 + (COMPAT_RULE_DECLARED if declared else COMPAT_RULE_ONE_TREE))}
            if problems:
                rec.update(status=POOLED_UNAVAILABLE, reason="; ".join(problems))
            else:
                rec.update(status=POOLED_AVAILABLE, reason=None,
                           **M.pooled_arm_bootstrap(arm_legs, bare, n_resamples=n_resamples, seed=seed, ci=ci))
            pooled[arm.kind] = rec
        out[field] = pooled
        out[f"n_available_{field.rsplit('_', 1)[-1]}"] = sum(1 for r in pooled.values() if r["status"] == POOLED_AVAILABLE)
    out["bootstrap"] = {"B": int(n_resamples), "seed": int(seed), "level": float(ci),
                        "shared_indices": ("one index vector per replicate applied to every model's leg (rng.integers(0, n, size=(m, n)) in "
                                           "pairing's chunk order; a pure function of (seed, n, B); the generator re-seeded per arm, branch "
                                           "and reading)")}
    out["estimand"] = POOLED_ESTIMAND_TEXT
    return out


def eightarm_findings(block: Mapping[str, Any]) -> list[str]:
    out: list[str] = []
    panel = "/".join(block["panel_models"])
    for field, label in (("pooled_by_arm_merged", "merged"), ("pooled_by_arm_base", "8192-only")):
        for kind, rec in _mapping(block.get(field)).items():
            if rec["status"] != POOLED_AVAILABLE:
                out.append(f"pooled ({panel}, {label}) {kind}: unavailable — {rec['reason']}")
                continue
            p = rec[BRANCH_PRIMARY]
            if kind == M.KIND_BARE:
                out.append(f"pooled ({panel}, {label}) none accuracy {100 * p['point']:.1f} % [{100 * p['ci_low']:.1f}, {100 * p['ci_high']:.1f}]")
                continue
            d = p["minus_bare"]
            out.append(f"pooled ({panel}, {label}) {kind} - none = {100 * d['point']:+.1f} pp [{100 * d['ci_low']:+.1f}, {100 * d['ci_high']:+.1f}] "
                       "(primary branch; equal weights; llama held out)")
    return out


#: per model, the primary sampling row's `source` prose — the MATH-500 module's reader (its POOLED_MODE is this instrument's
#: MODE_PRIMARY, `thinking_card`; the Context here carries the same `sampling_rows` table)
sampling_provenance = M.sampling_provenance


def _design_models(doc: Mapping[str, Any]) -> list[str]:
    """A report's ``design.models`` (empty when the block is absent — a peer document read defensively)."""
    models = _mapping(doc.get("design")).get("models")
    return [str(m) for m in models] if isinstance(models, list) else []


def parse_peer_report_spec(spec: str) -> tuple[Path, str | None]:
    """``REPORT`` (PR-A's grammar: the pin is the peer report's own ``sha``) or ``REPORT=PIN`` (the plan's: the pin must
    equal it) — B14: one flag, both spellings."""
    return M.parse_gen_root_spec(str(spec))


#: the peer report's THINKING vocabulary (`V.THINKING_ARMS`): the arm every published row runs, and the no-switch arm the
#: Llama rows carry; its design block files `thinking_mode_by_model` (`_arm_design`) and stamps `thinking_mode` on
#: every leg / controls / provenance row. A report from BEFORE the field reads every model as `on` (`_arm_of_model`).
PEER_THINKING_ON, PEER_THINKING_NOT_AVAILABLE = C.THINKING_ON, "not_available"
PEER_THINKING_ARMS: tuple[str, ...] = (PEER_THINKING_ON, PEER_THINKING_NOT_AVAILABLE)
PEER_THINKING_LABEL: Mapping[str, str] = {PEER_THINKING_ON: "thinking: on", PEER_THINKING_NOT_AVAILABLE: "thinking: not available"}
#: the design keys PR-A's `_arm_design` writes — kept through slimming and assembly (code review C3: `peer_slim` dropped them
#: and the figure label hard-coded "thinking: not available")
PEER_THINKING_DESIGN_KEYS: tuple[str, ...] = ("thinking_mode_by_model", "thinking_label_by_model", "no_switch_models", "held_out_of_pooled_means")
#: the evidence rows whose per-model metadata must agree with the design: every row names a design model; a row carrying a
#: `thinking_mode` carries the design's arm for its model
PEER_EVIDENCE_ROW_KEYS: tuple[str, ...] = ("controls", "contrast_summary", "noise_floor", "truncation_cost", "held", "refused")


def peer_thinking_mode_by_model(doc: Mapping[str, Any]) -> dict[str, str]:
    """``{model: arm}`` as the peer's design files it (PR-A `thinking_mode_by_model`), every design model covered — a model
    the block does not name reads ``on`` (PR-A's rule for a report from before the field)."""
    design = _mapping(doc.get("design"))
    filed = _mapping(design.get("thinking_mode_by_model"))
    return {m: str(filed.get(m, PEER_THINKING_ON)) for m in _design_models(doc)}


def peer_thinking_label(doc: Mapping[str, Any], model_key: str) -> str:
    """The figure / table label of a peer column's arm, DERIVED from the peer's metadata (never hard-coded)."""
    return PEER_THINKING_LABEL[peer_thinking_mode_by_model(doc).get(model_key, PEER_THINKING_ON)]


#: the report blocks a peer's arm walk does NOT read: the design (the reference the rows are held to), and anything another
#: report's (an assembled document is never a peer)
PEER_THINKING_WALK_SKIP: frozenset[str] = frozenset({"design", "peer_assembly", "peer_reports"})
#: the NESTED blocks whose `thinking_mode` is not an arm: a leg's `identity` is the manifest projection
#: (`projection_identity_fields`: `thinking_mode` = the manifest's served render mode, `thinking.mode`), a same-named field
#: the arm walk must not read as PR-A's arm stamp
PEER_THINKING_WALK_SKIP_NESTED: frozenset[str] = frozenset({"identity"})


def peer_thinking_mode_carriers(doc: Mapping[str, Any]) -> list[tuple[str, str | None, Any]]:
    """Every row of a peer report that carries a ``thinking_mode``, as ``(path, model, arm)``: PR-A's report
    (``origin/llama-paper-rows``) stamps the arm on ``legs.<leg>`` (the leg skeleton), on every ``controls`` row AND its
    nested ``provenance`` row, on ``provenance.cells.<leg>``, on the ``contrast_summary`` / ``noise_floor`` /
    ``truncation_cost`` / ``held`` / ``refused`` rows, on the ``cells`` block once assembled, on ``truncation.cells`` and
    ``truncation.costs``, and on a gate's per-leg record (``gates.<gate>.legs.<leg>``, the scoring-path gate) — so the walk
    is RECURSIVE over the whole document outside :data:`PEER_THINKING_WALK_SKIP`, never a list of paths that PR-A can
    outgrow. A row's model is its own ``model_key``; a record without one is resolved through the report's ``legs`` when
    its key is a leg key (``legs.<leg>.model_key``), else ``None`` — the caller refuses it. A leg's ``identity`` block is
    NOT a carrier (:data:`PEER_THINKING_WALK_SKIP_NESTED`): its ``thinking_mode`` is the manifest's served render mode."""
    legs = _mapping(doc.get("legs"))
    out: list[tuple[str, str | None, Any]] = []

    def walk(node: Any, path: str, key: Any) -> None:
        if isinstance(node, Mapping):
            if "thinking_mode" in node and path:
                model = node.get("model_key")
                if model is None and isinstance(key, str) and key in legs:
                    model = _mapping(legs.get(key)).get("model_key")
                out.append((path, str(model) if model is not None else None, node.get("thinking_mode")))
            for k, v in node.items():
                if (not path and k in PEER_THINKING_WALK_SKIP) or (path and k in PEER_THINKING_WALK_SKIP_NESTED):
                    continue
                walk(v, f"{path}.{k}" if path else str(k), k)
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]", None)

    walk(doc, "", None)
    return out


def peer_admission_problem(doc: Mapping[str, Any], *, path: Path, pin: str | None, this: Mapping[str, Any], seen_models: set[str]) -> str | None:
    """Why a ``--peer-report`` document is NOT admissible (admission used to fail OPEN: a report whose gate records were
    simply absent passed `production_refusals`, its sha was unvalidated and its thinking metadata dropped): the
    experiment / task; a VALID store pin of its own (the pin grammar), another wave's, equal to a given ``=PIN``; EVERY
    gate of :data:`GATE_NAMES` on record with an admissible status (PASS, or a waivable PENDING) and `gates.failed` empty;
    :func:`production_refusals` empty; ``design.models`` non-empty and disjoint from this run's / the other peers'; every
    evidence row (:data:`PEER_EVIDENCE_ROW_KEYS`, ``cells``) naming a design model; ``thinking_mode_by_model`` (when
    filed) naming design models only with arms in :data:`PEER_THINKING_ARMS`, and EVERY row of the document that carries a
    ``thinking_mode`` — the legs, the controls rows and their nested provenance rows, ``provenance.cells``, the truncation
    block's cells and costs, a gate's per-leg records, the evidence rows (:func:`peer_thinking_mode_carriers`) — carrying
    the design's arm for its model (the evidence rows alone were checked, so a leg-only ``not_available`` under a design
    ``on`` was admitted and :func:`peer_slim` dropped the contradiction)."""
    if doc.get("experiment") != EXPERIMENT or doc.get("task") != TASK:
        return f"experiment / task {doc.get('experiment')!r} / {doc.get('task')!r} != {EXPERIMENT!r} / {TASK!r}"
    sha = doc.get("sha")
    problem = M.pin_problem(sha)
    if problem:
        return f"its store pin is not valid — {problem}"
    if str(sha) == str(this["sha"]):
        return f"store pin {sha!r} is this run's own — a peer is another wave's report"
    if pin is not None and not _same_pin(str(sha), pin):
        return f"the given pin {pin!r} is not the peer report's own store pin {sha!r}"
    gates = doc.get("gates")
    if not isinstance(gates, Mapping):
        return "carries no `gates` block — not a gated report"
    missing = [g for g in GATE_NAMES if not isinstance(gates.get(g), Mapping) or "status" not in gates[g]]
    if missing:
        return f"gate record(s) absent or without a status: {missing} — the expected gate schema is {list(GATE_NAMES)}"
    bad = {g: _mapping(gates.get(g)).get("status") for g in GATE_NAMES if _mapping(gates.get(g)).get("status") not in (GATE_PASS, GATE_PENDING)}
    if bad:
        return f"gate status(es) not admissible: {bad} (a peer's gates are PASS, or PENDING on a waivable stamp)"
    if gates.get("failed"):
        return f"gates.failed is not empty: {gates.get('failed')}"
    refusals = production_refusals(doc)
    if refusals:
        return "not a production result — " + "; ".join(refusals)
    models = _design_models(doc)
    if not models:
        return "names no model (design.models)"
    clash = sorted(set(models) & seen_models)
    if clash:
        return f"names model(s) {clash} this run or another peer already names — one column per model"
    design = _mapping(doc.get("design"))
    filed = design.get("thinking_mode_by_model")
    if filed is not None:
        if not isinstance(filed, Mapping):
            return "design.thinking_mode_by_model is not a mapping"
        stray = sorted(set(filed) - set(models))
        if stray:
            return f"design.thinking_mode_by_model names model(s) {stray} outside design.models {models}"
        bad_arms = {m: a for m, a in filed.items() if a not in PEER_THINKING_ARMS}
        if bad_arms:
            return f"design.thinking_mode_by_model carries arm(s) outside {list(PEER_THINKING_ARMS)}: {bad_arms}"
    arms = peer_thinking_mode_by_model(doc)
    for key in PEER_EVIDENCE_ROW_KEYS:
        for i, row in enumerate(doc.get(key) or []):
            if not isinstance(row, Mapping):
                return f"{key}[{i}] is not a row"
            m = row.get("model_key")
            if m not in arms:
                return f"{key}[{i}] names model {m!r}, not one of design.models {models}"
            if "thinking_mode" in row and row.get("thinking_mode") != arms[m]:
                return f"{key}[{i}] ({m}) carries thinking_mode {row.get('thinking_mode')!r} while the design files {arms[m]!r} for it"
    for ck, c in _mapping(doc.get("cells")).items():
        m = _mapping(c).get("model_key")
        if m not in arms:
            return f"cells.{ck} names model {m!r}, not one of design.models {models}"
        if "thinking_mode" in _mapping(c) and _mapping(c).get("thinking_mode") != arms[m]:
            return f"cells.{ck} ({m}) carries thinking_mode {_mapping(c).get('thinking_mode')!r} while the design files {arms[m]!r} for it"
    # EVERY carrier of a thinking_mode in the document, wherever it is stamped, agrees with the design's arm
    for where, m, arm in peer_thinking_mode_carriers(doc):
        if m is None:
            return f"{where} carries thinking_mode {arm!r} for no model this report names (no model_key, not a leg key of `legs`)"
        if m not in arms:
            return f"{where} names model {m!r}, not one of design.models {models}"
        if arm != arms[m]:
            return f"{where} ({m}) carries thinking_mode {arm!r} while the design files {arms[m]!r} for it"
    return None


def load_peer_reports(paths: Sequence[Path | str], *, this: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The ``--peer-report`` documents (plan P5; PR-A's ``load_peer_reports`` signature and rules): each a
    PRODUCTION report of this experiment and task, admitted by :func:`peer_admission_problem` — a valid pin of another
    wave (equal to a given ``=PIN``), the complete gate schema with admissible statuses, no production refusal, disjoint
    models, and model / thinking metadata consistent across the design and every evidence row. Anything else refuses by
    name — the eight-arm figures are assembled from already-gated tables, never from a report the reader would not publish."""
    out: list[dict[str, Any]] = []
    seen_models = set(_design_models(this))
    for spec in paths:
        p, pin = parse_peer_report_spec(str(spec))
        doc = _read_json(p)
        if not isinstance(doc, Mapping):
            raise AnalysisError(f"--peer-report {p}: not a JSON object")
        problem = peer_admission_problem(doc, path=p, pin=pin, this=this, seen_models=seen_models)
        if problem:
            raise AnalysisError(f"--peer-report {p}{'=' + pin if pin else ''}: {problem}")
        seen_models |= set(_design_models(doc))
        out.append({**dict(doc), "_path": str(p), "_pin": str(doc.get("sha"))})
    return out




def combine_reports(this: Mapping[str, Any], peers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The assembled view (PR-A's ``combine_reports`` signature): the peers' and this run's already-gated tables side by
    side — ``design.models`` / ``views``, ``cells``, ``controls``, ``contrasts``, ``contrast_summary``, ``noise_floor``,
    ``truncation_cost``, ``held``, ``refused``, ``truncation.cells`` / ``costs``, ``overlap.models`` — every row carrying
    its report's ``pin``; the ``peer_assembly`` block names the column pins and the hold-out. NOTHING is merged at the
    store level: each report's numbers are copied as gated; a no-switch model is labelled and held out, never pooled."""
    reports = [*peers, this]
    pins = {m: str(r["sha"]) for r in reports for m in _design_models(r)}

    def stamped(rows: Sequence[Mapping[str, Any]], sha: str) -> list[dict[str, Any]]:
        return [{**dict(x), "pin": sha} for x in rows]

    combined: dict[str, Any] = {
        "experiment": EXPERIMENT, "task": TASK, "sha": this["sha"], "analysis_code_sha": this.get("analysis_code_sha"), "bootstrap": this.get("bootstrap"),
        "peer_assembly": {"pins": pins, "reports": [{"sha": str(r["sha"]), "path": r.get("_path"), "models": _design_models(r)} for r in reports],
                          "held_out": dict(POOLED_HELD_OUT), "not_in_panel": list(POOLED_NOT_IN_PANEL),
                          "note": "assembled from the reports' already-gated tables; no store-level merge; llama held out of every pooled mean"},
        "design": {**_mapping(this.get("design")), "models": [m for r in reports for m in _design_models(r)],
                   "views": [v for r in reports for v in (_mapping(r.get("design")).get("views") or [])]},
        "cells": {}, "controls": [], "contrasts": [], "contrast_summary": [], "noise_floor": [], "truncation_cost": [], "held": [], "refused": [],
        "truncation": {**{k: v for k, v in _mapping(this.get("truncation")).items() if k not in ("cells", "costs")}, "cells": [], "costs": []},
        "overlap": {"models": {}},
    }
    for r in reports:
        sha = str(r["sha"])
        combined["cells"].update({ck: {**dict(c), "pin": sha} for ck, c in _mapping(r.get("cells")).items()})
        for key in ("controls", "contrasts", "contrast_summary", "noise_floor", "truncation_cost", "held", "refused"):
            combined[key].extend(stamped(r.get(key) or [], sha))
        combined["truncation"]["cells"].extend(stamped(_mapping(r.get("truncation")).get("cells") or [], sha))
        combined["truncation"]["costs"].extend(stamped(_mapping(r.get("truncation")).get("costs") or [], sha))
        combined["overlap"]["models"].update({k: {**dict(b), "pin": sha} for k, b in _mapping(_mapping(r.get("overlap")).get("models")).items()})
    tr = combined["truncation"]
    for k in ("over_1pct", "triggered", "escalated", "terminal_no_rung"):
        tr[k] = [leg for r in reports for leg in (_mapping(r.get("truncation")).get(k) or [])]
    tr["n_cells_read"] = len(tr["cells"])
    tr["max_share"] = max((c["share"] for c in tr["cells"] if c.get("share") is not None), default=None)
    return combined


def peer_slim(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a peer report the eight-arm figures DRAW (its main-view contrast summaries — points, CIs, words — and its
    cells' two readings), filed under ``report["peer_reports"][<pin>]`` so every plotted number resolves in THIS report."""
    arms = peer_thinking_mode_by_model(doc)
    cells = {ck: {"model_key": c.get("model_key"), "view": c.get("view"), "cell": c.get("cell"), "base_leg": c.get("base_leg"),
                  "rung_leg": c.get("rung_leg"),
                  # C3: the row's arm rides the slim (the peer's own stamp, else the design's arm for its model)
                  "thinking_mode": c.get("thinking_mode", arms.get(c.get("model_key"))),
                  "readings": ({r: {k: v for k, v in _mapping(_mapping(c.get("readings")).get(r)).items() if k != "items"} for r in READINGS}
                               if c.get("readings") else None)}
             for ck, c in _mapping(doc.get("cells")).items()}
    design = _mapping(doc.get("design"))
    return {"path": doc.get("_path"), "pin": str(doc.get("sha")), "experiment": doc.get("experiment"), "task": doc.get("task"),
            "analysis_code_sha": doc.get("analysis_code_sha"), "models": _design_models(doc),
            # C3: PR-A's four arm keys are PRESERVED (the label and the hold-out footer derive from them); a report from before
            # the field gets the derived `on` map so every consumer reads one shape
            "design": {**{k: design.get(k) for k in ("models", "views", "n_launch_rows", "n_held_rows")},
                       "thinking_mode_by_model": arms,
                       "thinking_label_by_model": design.get("thinking_label_by_model") or {m: PEER_THINKING_LABEL[a] for m, a in arms.items()},
                       "no_switch_models": design.get("no_switch_models") or [m for m, a in arms.items() if a == PEER_THINKING_NOT_AVAILABLE],
                       "held_out_of_pooled_means": design.get("held_out_of_pooled_means")},
            "cells": cells, "contrast_summary": list(doc.get("contrast_summary") or []),
            "held": list(doc.get("held") or []), "refused": list(doc.get("refused") or []),
            "verdict": {k: _mapping(doc.get("verdict")).get(k) for k in ("word", "verdict")},
            "truncation": {k: _mapping(doc.get("truncation")).get(k) for k in ("trigger_threshold", "trigger_provenance", "rule")}}


def peer_reports_design(peers: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """``design.peer_reports``: one record per peer present, else the EXPLICIT absence of the Llama report."""
    if peers:
        return [{"path": p.get("_path"), "pin": str(p.get("sha")), "models": _design_models(p), "status": PEER_STATUS_PRESENT, "reason": None,
                 # C3: the peer's arm per model rides the design record (the figure labels derive from it)
                 "thinking_mode_by_model": peer_thinking_mode_by_model(p)} for p in peers]
    return [{"path": None, "pin": None, "models": [PEER_MODEL_LLAMA], "status": PEER_STATUS_ABSENT,
             "reason": "no --peer-report supplied: the llama column is the Llama lane's production report (PR-A) at its own pin, pending"}]


def completeness_text(models_present: Sequence[str], peers: Sequence[Mapping[str, Any]]) -> str:
    """``completeness.eight_arm.gsm8k`` — a STRING: how many of the four panel columns (gemma / qwen / nemotron / llama)
    this assembly carries; never "4 of 4" without the Llama peer report."""
    peer_models = {m for p in peers for m in _design_models(p)}
    present = [m for m in EIGHTARM_MODEL_ORDER if m in set(models_present) | peer_models]
    missing = [m for m in EIGHTARM_MODEL_ORDER if m not in present]
    text = f"{len(present)} of {len(EIGHTARM_MODEL_ORDER)} panel models present ({', '.join(present) or 'none'})"
    if missing:
        text += f"; missing: {', '.join(missing)}" + (f" ({PEER_PENDING_LABEL})" if PEER_MODEL_LLAMA in missing else "")
    return text


def analyze(gen_root: Path | str | M.GenRoots | Sequence[str], sha: str, *, matrix: Path | str | Sequence[Any], canonical_ids: Path | str | Sequence[str] | None,
            sampling_rows: Mapping[str, Any] | Path | str | None = None, render_facts: Mapping[str, Any] | Path | str | None = None,
            pins: Mapping[str, Any] | Path | str | None = None, model_pins: Mapping[str, tuple[str, str]] | None = None,
            expected_refusals: Path | str | Mapping[str, str] | None = None, overlap: Mapping[str, Any] | Path | str | None = None,
            n_resamples: int = pr.DEFAULT_BOOTSTRAP, seed: int = pr.DEFAULT_BOOTSTRAP_SEED, ci: float = pr.DEFAULT_CI,
            allow_pending_gates: bool = False, read_eval_logs: bool = True, analysis_sha: str | None = None,
            repo_root: Path | None = None, tree_set: Mapping[str, Any] | None = None, waves: Sequence[str] | None = None,
            store_pins: Path | str | Mapping[str, str] | Sequence[Path | str | Mapping[str, str]] | None = None,
            peer_reports: Sequence[Path | str] = (), unterminated_finished_policy: str = UNTERMINATED_FINISHED_UNSCORED,
            object_generations: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Collect every required cell (its 8,192 leg, ladder record, subset, rung leg,
    readings) -> the contrasts per view under both readings and both axes ->
    the noise-floor pairs -> the truncation costs -> the gates -> the verdict.
    ``gen_root``: a Path (the published call), a ``GenRoots``, or ``PATH[=PIN]``
    specs (the repeatable CLI form). ``tree_set`` is a DECLARATION:
    ``also`` / ``allow`` / ``why`` [/ ``allow_file`` / ``allow_evidence``]
    — verified HERE on ``repo_root`` from those fields alone before any cell is
    loaded (a caller's ``problems: []`` is never trusted). ``waves``: the
    REQUIRED-EVIDENCE set. ``peer_reports``: the Llama production
    report(s) (``REPORT[=PIN]``), validated by :func:`load_peer_reports`.
    ``unterminated_finished_policy``: the reader's treatment of a draw that finished
    inside an unterminated think block (module constants; the default writes nothing)."""
    repo_root = Path(repo_root) if repo_root is not None else _HERE.parent
    decl = _mapping(tree_set)
    pins_by_leg = load_store_pins(store_pins)
    if pins_by_leg is not None and decl.get("also"):
        raise AnalysisError("--store-pins and --also-sha are two admissions of the same legs — give one")
    if decl.get("primary_pin", sha) != sha:
        raise AnalysisError(f"{TREE_SET_KEY}: the declaration's primary pin {decl.get('primary_pin')!r} is not --sha {sha!r}")
    ts: dict[str, Any] | None = None
    if decl.get("also"):
        ts = M.verify_tree_set(sha, [str(x) for x in decl["also"]], [str(g) for g in decl.get("allow") or ()], repo_root=repo_root,
                               reason=decl.get("why"), allow_file=decl.get("allow_file"), allow_evidence=_mapping(decl.get("allow_evidence")) or None)
        if ts["problems"]:
            raise AnalysisError(f"{TREE_SET_KEY}: " + "; ".join(ts["problems"]))
    admitted = tuple(ts["also"]) if ts else ()
    # B6: the gen roots — a Path (the published call), a GenRoots, or `PATH[=PIN]` specs (the repeatable CLI form); under per-row
    # pins a PINNED root may name any admitted tree
    roots = M.GenRoots.of(gen_root, primary_pin=sha, admitted=admitted if pins_by_leg is None else tuple(sorted(set(pins_by_leg.values()))))
    mx = V.load_matrix(matrix)
    required = V.required_rows(mx["launch"], waves)
    eight_arm = eight_arm_mode(mx["rows"], waves, roots, peers=bool(peer_reports))
    if not required:
        raise AnalysisError(f"no evidence row of the analysed waves {list(waves) if waves is not None else '(all)'} — nothing to collect "
                            f"(the matrix's waves: {sorted({V.row_wave(r) for r in mx['rows']})})")
    # the matrix VIEW every consumer of "the launch rows" reads: the REQUIRED set — the published call keeps every row
    mxr: dict[str, Any] = {**mx, "launch": required}
    pins_doc = (_read_json(Path(pins).expanduser()) if isinstance(pins, (str, Path)) else pins) if pins is not None \
        else _read_json(repo_root / "configs" / "sdfbase_full22_pins.json")
    problem = V.matrix_pins_problem(mx["rows"], _mapping(pins_doc))
    if problem:
        raise AnalysisError(f"matrix vs pin file: {problem}")
    served_pins = dict(model_pins or {})
    for r in required:
        if str(r["tag"]) not in served_pins:
            served_pins[str(r["tag"])] = M.model_pin(repo_root, str(r["tag"]))
    ids, ids_path = load_ids(canonical_ids)
    rows_src = sampling_rows if sampling_rows is not None else (repo_root / "configs" / "sdfbase" / "math500" / "sampling_rows.yaml")
    overlap_doc = (_read_json(Path(overlap).expanduser()) if isinstance(overlap, (str, Path)) else overlap) if overlap is not None else None
    # E3 tier-2 item 5: the fit-provenance bindings a fitted row's `object_generation` is READ from — given (a test's mapping), else
    # derived from the tree's two records where present (the it-doc admission record's fit records + the declared batch-8 fit record);
    # a tree with neither binds nothing (every fitted cell unstamped, as before)
    if object_generations is None:
        itdoc_doc = V.read_json_if_file(repo_root / V.ITDOC_ADMISSION_REL)
        fit_batches = V.load_object_fit_batches(repo_root / V.OBJECT_FIT_BATCHES_REL) if (repo_root / V.OBJECT_FIT_BATCHES_REL).is_file() else None
        object_generations = V.object_generation_bindings(itdoc_doc if isinstance(itdoc_doc, Mapping) else None, fit_batches)
    ctx = Context(sampling_rows=M.load_sampling_rows(rows_src), render_facts=M.load_render_facts(render_facts), model_pins=served_pins,
                  store_pin=sha, expected_refusals=load_expected_refusals(mxr, expected_refusals), also_trees=admitted,
                  declared=bool(ts) or roots.declared or pins_by_leg is not None,
                  arm_contracts=eight_arm or bool(ts) or roots.declared or pins_by_leg is not None,
                  unterminated_finished_policy=unterminated_finished_policy, store_pins=pins_by_leg, object_generations=object_generations)
    cells = {V.cell_key(r["model_key"], r["view"], r["cell"]): collect_cell(roots, r, ctx=ctx, read_eval_logs=read_eval_logs)
             for r in required}
    if ts is not None:
        ts["legs"] = {k: _mapping(leg.get("producer")).get("tree_sha") for k, leg in _all_legs(cells).items()}      # The per-leg tree assignment
    contrasts = design_contrasts(mxr, cells, refusals=ctx.expected_refusals, n_resamples=n_resamples, seed=seed, ci=ci, declared=ctx.declared)
    noise = noise_floor_pairs(mxr, cells, n_resamples=n_resamples, seed=seed, ci=ci, declared=ctx.declared)
    costs = truncation_costs(cells, n_resamples=n_resamples, seed=seed, ci=ci)
    code = T.analysis_code_provenance(analysis_sha)
    a0, refused, unusable = gate_completeness(mxr, cells, ctx.expected_refusals)
    models = list(dict.fromkeys(r["model_key"] for r in required))
    arm_by_model = V.thinking_mode_by_model(mx["launch"])
    gates: dict[str, Any] = {"A0_completeness": a0, "A1_ledger": gate_ledger(cells), "A2_identity": gate_identity(cells),
                             "A3_certification": gate_certification(cells), "A4_scoring_path": gate_scoring_path(cells),
                             "A5_at_cap": gate_at_cap(cells), "A6_ladder": gate_ladder(roots.primary, cells),
                             "A7_population": gate_population(cells, contrasts, noise, ids, declared=ctx.declared), "A8_two_branch": gate_two_branch(contrasts),
                             "A9_report_match": gate_report_match(None),
                             "A10_overlap": gate_overlap(overlap_doc, models, ids,
                                                         store_pin=(overlap_pin_under_store_pins(models, pins_by_leg) if pins_by_leg is not None else ctx.store_pin))}
    refresh_gate_aggregates(gates)
    legs_all = _all_legs(cells)
    design: dict[str, Any] = {"n_launch_rows": len(required), "n_held_rows": len(mx["held"]), "models": models,
                              "views": sorted({f"{r['model_key']}/{r['view']}" for r in required}), "n_contrasts_designed": len(contrasts),
                              "n_contrasts_computable": sum(1 for c in contrasts if c["computable"]), "n_noise_floor_pairs": len(noise),
                              "n_escalated_cells": len(costs), "cap": CAP_BASE, "ceiling": CAP_RUNG, "trigger_threshold": V.GSM8K_TRIGGER_THRESHOLD,
                              "report_line": V.GSM8K_REPORT_LINE, "n_expected_refusals": len(ctx.expected_refusals),
                              "n_servable": len(required) - len(ctx.expected_refusals),
                              # the DECLARED expected CONTRAST refusals (the rows' expected_contrast_refusal), by contrast name
                              "expected_contrast_refusals": {c["name"]: c[ROW_EXPECTED_CONTRAST_REFUSAL_KEY] for c in contrasts
                                                             if c.get(ROW_EXPECTED_CONTRAST_REFUSAL_KEY)},
                              "n_expected_contrast_refusals": sum(1 for c in contrasts if c.get(ROW_EXPECTED_CONTRAST_REFUSAL_KEY)),
                              "n_certified_unservable": sum(1 for x in refused if x.get("status") == LEG_STATUS_REFUSED_CERTIFIED),
                                # the ARM per model (`on` | `not_available`), its label, and the models HELD OUT of any
                                # pooled cross-family mean (a family with no thinking switch is a different instrument)
                                **_arm_design(arm_by_model),
                                # the schema-2 facts the Llama matrix carries (None on the published schema-1 document)
                                "matrix_schema": mx.get("schema"), "wave_root_rel": mx.get("wave_root_rel")}
    if ctx.unterminated_finished_policy != UNTERMINATED_FINISHED_UNSCORED:
        # the reader's non-default policy is NAMED in the design (the default writes no field — the published shape)
        design[UNTERMINATED_FINISHED_POLICY_KEY] = ctx.unterminated_finished_policy
    peers: list[dict[str, Any]] = []
    if eight_arm:
        # the immutable panel design apart from the required-evidence set; the expected leg set DERIVED (never typed)
        design.update(design_status_block(mx["rows"], required, waves))
        design["required_legs"] = expected_leg_set(required, cells)
        design["n_required_legs"] = len(design["required_legs"])
    report: dict[str, Any] = {
        "experiment": EXPERIMENT, "task": TASK, "sha": sha, "analysis_code_sha": code["sha"], "analysis_code": code, "gen_root": str(roots.primary),
        "estimand": ESTIMAND_TEXT + (UNTERMINATED_FINISHED_FAILURE_TEXT if ctx.unterminated_finished_policy == UNTERMINATED_FINISHED_FAILURE else ""),
        "bootstrap": {"B": n_resamples, "seed": seed, "level": ci}, "allow_pending_gates": bool(allow_pending_gates),
        "matrix": {"path": mx["path"], "sha256": mx["sha256"], "counts": mx["counts"]},
        "design": design,
        #: THE ONE resolved expected-refusal mapping (rows + --expected-refusals), read by every consumer
        "expected_refusals": dict(sorted(ctx.expected_refusals.items())),
        # E3: a held row carries its wave and object source and, on an it-doc row, its lane and the pair's qualification VERBATIM
        # (None on every other held row; the published four carry None throughout)
        "held": [{"model_key": r["model_key"], "view": r["view"], "cell": r["cell"], "job_id": r["job_id"], "reason": r["held_reason"],
                  "wave": V.row_wave(r), "object_source": r.get("object_source"), **V.itdoc_fields(r)} for r in mx["held"]],
        "refused": refused,
        "canonical_ids": ({"path": ids_path, "sha256": hashlib.sha256(json.dumps(sorted(ids)).encode("utf-8")).hexdigest(), "n": len(ids)}
                          if ids is not None else None),
        "sampling_rows": {MODE_PRIMARY: ctx.sampling_rows.get(MODE_PRIMARY)},
        # the families THIS matrix runs (the published four-family report stays byte-identical in shape when the shared fact table
        # gains a family — llama's, the no-switch fact — that its rows never ran)
        "render_facts": {fam: {k: f.get(k) for k in ("split_mode", "server_parser", "evidence_status", "off_ids", "on_ids")} for fam, f in ctx.render_facts.items()
                         if fam in {str(r["family"]) for r in mx["launch"]}},
        "legs": legs_all, "unusable_legs": unusable,
        "cells": {ck: {k: v for k, v in c.items() if k != "legs"} for ck, c in cells.items()},
        "contrasts": contrasts, "noise_floor": noise, "truncation_cost": costs,
        "contrast_summary": [_contrast_summary(c) for c in contrasts], "noise_floor_summary": [_contrast_summary(c) for c in noise],
        "gates": gates, "overlap": overlap_doc, "truncation": truncation_block(cells, costs),
        #: the producer record gaps the standup records closed (served-positions-null-diagnosis.md §4.2), by leg
        V.RECORD_GAPS_KEY: record_gaps_block(legs_all),
        #: the engine-resolved sampler attestation per leg (a finding, never a gate)
        **({RESOLVED_SAMPLER_KEY: rsb} if (rsb := resolved_sampler_block(legs_all)) is not None else {}),
    }
    if ts is not None:
        report[TREE_SET_KEY] = ts
    if pins_by_leg is not None:
        # per-row admission (--store-pins): the doc as read, and every leg's admitted pin beside its producer tree
        report[STORE_PINS_KEY] = {"admission": "per-row pins (--store-pins): every leg admitted by ITS OWN producer tree",
                                  "pins": dict(sorted(pins_by_leg.items())),
                                  "legs": {k: {"admitted_pin": _mapping(leg.get("producer")).get("admitted_pin") or leg.get("admitted_pin"),
                                               "tree_sha": _mapping(leg.get("producer")).get("tree_sha"), "usable": usable(leg)}
                                           for k, leg in legs_all.items()},
                                  "trees": sorted({_mapping(leg.get("producer")).get("tree_sha") for leg in legs_all.values()
                                                   if usable(leg) and _mapping(leg.get("producer")).get("tree_sha")})}
    if roots.declared or pins_by_leg is not None:
        report["gen_roots"] = roots.record()
    ran = [c for c in contrasts if c["computable"]]
    if ran:
        report["estimands"] = {c["name"]: {reading: c["readings"][reading]["estimands"] for reading in READINGS} for c in ran}
    if eight_arm:
        # the pooled-by-arm estimands as REPORT FIELDS over the main view of the panel models (the figures draw them)
        report["eightarm"] = eightarm_block(cells, design, n_resamples=n_resamples, seed=seed, ci=ci, declared=ctx.declared)
        report["deferred"] = [{"model_key": r["model_key"], "view": r["view"], "cell": r["cell"], "job_id": r.get("job_id"),
                               "reason": r.get("deferred_reason") or r.get("reason") or "deferred row (no reason recorded)",
                               "object_source": r.get("object_source"), "wave": V.row_wave(r)} for r in mx["deferred"]]
        # the Llama production report(s), validated against THIS report; absence is EXPLICIT
        peers = load_peer_reports(list(peer_reports), this=report)
        design["peer_reports"] = peer_reports_design(peers)
        report["peer_reports"] = {p["_pin"]: peer_slim(p) for p in peers}
        report["completeness"] = {"eight_arm": {"gsm8k": completeness_text(models, peers)}}
    report["controls"] = controls_table(cells)
    report["provenance"] = {"store_pin": sha, "analysis_code_sha": code["sha"], "analysis_code": code, "matrix_sha256": mx["sha256"],
                            "canonical_ids_sha256": (report["canonical_ids"] or {}).get("sha256"),
                            "note": "kept apart: the store pin (every cell's wave sha), the analysis code's tree, the matrix's digest, the canonical ids' "
                                    "digest, and each cell's own producer / config / intervention hashes",
                            # the spec's provenance sentences (strings — B1): the trigger this instrument reports under (inspect_rung's
                            # constant, never re-typed) and the sampling source per model, read off the pinned run config
                            "trigger": TRIGGER_PROVENANCE, "sampling": sampling_provenance(ctx, required),
                            **({"waves": design["waves"], "gen_roots": roots.record()} if eight_arm else {}),
                            "cells": {k: provenance_row(leg) for k, leg in legs_all.items()}}
    findings: list[str] = list(a0.get("findings") or [])
    if eight_arm:
        findings.extend(eightarm_findings(report["eightarm"]))
        for k, rec in design["status_by_row"].items():
            if rec["status"] in (STATUS_HELD, STATUS_DEFERRED, ROW_STATUS_NOT_ANALYSED):
                findings.append(f"{k}: {rec['status'].upper()} — {rec['reason']} (a design status; never collected)")
        for p in design["peer_reports"]:
            findings.append(f"peer report ({', '.join(p['models'])}): {p['status']}" + (f" — {p['reason']}" if p.get("reason") else f" at pin {p['pin']}"))
        findings.append(f"eight-arm completeness: {report['completeness']['eight_arm']['gsm8k']}")
    if ts is not None:
        findings.append(M.tree_set_note(ts))
    for rec in refused:
        line = f"{rec['leg']}: refused by name ({rec['code']}) — {rec['reason']}"
        if rec.get("status") == LEG_STATUS_REFUSED_CERTIFIED:
            line += (f"; ON probe draws {rec.get('attempt_classes')}, certifying draw {rec.get('certifying_draw_class')!r}"
                     + (f"; {rec['observation']}" if rec.get("observation") else ""))
        findings.append(line)
    for h in report["held"]:
        findings.append(f"{h['model_key']}/{h['view']}/{h['cell']}: HELD — {h['reason']}"
                        # E3: an it-doc hold carries the pair's qualification beside its reason (verbatim off the row)
                        + (f" — qualification: {V.qualification_text(h)}" if V.qualification_text(h) else ""))
    for lk, gaps in report[V.RECORD_GAPS_KEY]["legs"].items():
        findings.append(f"{lk}: RECORD GAP — {record_gap_text(gaps)} ({gaps[0]['cause']})")
    for lk in _mapping(report.get(RESOLVED_SAMPLER_KEY)).get("legs_not_agreeing") or []:
        rs = report[RESOLVED_SAMPLER_KEY]["legs"][lk]
        findings.append(f"{lk}: RESOLVED SAMPLER {rs['verdict']} — {rs['problem'] or rs['why']}")
    tr = report["truncation"]
    for c in tr["cells"]:
        if c["trigger_exceeded"]:
            findings.append(f"{c['leg']}: at-cap share {100 * c['share']:.1f} % > the {TRIGGER_PCT} trigger ({c['n_at_cap']} of {c['n_with_stop_reason']}; loop {c['n_loop']} / "
                            f"truncated {c['n_truncated_reasoning']} / text unavailable {c['n_text_unavailable']}) -> exit {c['exit']}")
        elif c["line_1pct_exceeded"]:
            findings.append(f"{c['leg']}: at-cap share {100 * c['share']:.1f} % > the 1 % report line (at or below the {TRIGGER_PCT} trigger)")
    for t in costs:
        findings.append(f"{t['cell_key']}: truncation cost merged - r8192 = {100 * t['point_primary']:+.1f} pp (primary), residual at 16,384: {t['residual_at_ceiling']} of {t['n_subset']}")
    words: dict[str, dict[str, str | None]] = {}
    for c in ran:
        words[c["name"]] = {reading: c["readings"][reading]["verdict"]["word"] for reading in READINGS}
        findings.append(f"{c['name']}: merged {words[c['name']][READING_MERGED]}; 8192-only {words[c['name']][READING_R8192]}")
    for c in _not_run(contrasts):
        findings.append(f"{c['name']}: no estimand — {c['reason']}")
    for c in contrasts:
        if c.get(CONTRAST_REFUSED_BY_DESIGN_KEY):
            findings.append(f"{c['name']}: refused by design ({c[ROW_EXPECTED_CONTRAST_REFUSAL_KEY]}) — the arm's numbers are reported as the arm's "
                            f"numbers, the pairing's absence accepted — {c['reason']}")
    for n in noise:
        if n["computable"]:
            st = n["readings"][READING_MERGED]["estimands"][AXIS_STOCK][BRANCH_PRIMARY]["stats"][CONTRAST_NOISE]
            findings.append(f"{n['name']}: noise floor {100 * st['point']:+.1f} pp [{100 * st['ci_low']:+.1f}, {100 * st['ci_high']:+.1f}] (stock, primary, merged)")
    if overlap_doc:
        for key, b in _mapping(overlap_doc.get("models")).items():
            cmp_ = _mapping(b.get("comparison"))
            if cmp_.get("complete"):
                findings.append(f"overlap {key}: stock - retired {100 * cmp_['delta']:+.1f} pp [{100 * cmp_['ci_low']:+.1f}, {100 * cmp_['ci_high']:+.1f}] "
                                f"-> {b.get('verdict')}")
            else:
                findings.append(f"overlap {key}: {b.get('reason')}")
    report["verdict"] = verdict_of(gates, contrasts, words, allow_pending_gates=allow_pending_gates)
    report["findings"] = findings
    report["top_items"] = top_items_of(report)
    return report


# --------------------------------------------------------------------------- #
# figures (dataviz: fixed hue per cell family, never cycled; title <= 2 lines of <= 90 chars; axes >= 80 % of the height) #
# --------------------------------------------------------------------------- #

SURFACE, INK, INK2, MUTED, GRID, BASELINE = M.SURFACE, M.INK, M.INK2, M.MUTED, M.GRID, M.BASELINE
#: the reference palette's first seven slots, in order (dataviz palette.md): slots 5-7 are the MATH-500 module's CAT8 hex
#: values (one spelling per hue); violet is taken here by the prompted-constitution family — a fixed hue per entity, never cycled
CAT: Mapping[str, str] = {**T.CAT, **{k: M.CAT8[k] for k in ("magenta", "green", "violet")}}
FAMILY_NONE, FAMILY_PROMPTED, FAMILY_CONST, FAMILY_CTL, FAMILY_SI12, FAMILY_DERIVED = ("none", "prompted", "constitution", "control", "safety_si12", "derived")
#: the eight-arm extension's prompted CONSTITUTION cell is its own family (plan P5: per-SI labels — "prompted (constitution)")
FAMILY_PROMPTED_CONST = "prompted_constitution"
#: the constitution prompted cell's id — the MATH-500 module's spelling (one across both instruments); g8k_noise_floor reads it here
CELL_PROMPTED_CONST = M.CELL_PROMPTED_CONST
FAMILY_COLOR: Mapping[str, str] = {FAMILY_NONE: CAT["blue"], FAMILY_PROMPTED: CAT["aqua"], FAMILY_CONST: CAT["orange"], FAMILY_CTL: CAT["yellow"],
                                   FAMILY_SI12: CAT["magenta"], FAMILY_DERIVED: CAT["green"], FAMILY_PROMPTED_CONST: CAT["violet"]}
FAMILY_LABEL: Mapping[str, str] = {FAMILY_NONE: "none (bare)", FAMILY_PROMPTED: "prompted (safety_refusal)", FAMILY_CONST: "constitution object",
                                   FAMILY_CTL: "control object (constv2ctl2)", FAMILY_SI12: "safety si12 object (header-located)", FAMILY_DERIVED: "derived residual",
                                   FAMILY_PROMPTED_CONST: "prompted (constitution)"}
#: hue = ARM for the eight-arm figures (the MATH-500 module's table: one fixed hue per arm across both instruments)
KIND_COLOR: Mapping[str, str] = M.KIND_COLOR
TITLE_MAX_LINES, TITLE_MAX_CHARS, AXES_MIN_FRACTION = M.TITLE_MAX_LINES, M.TITLE_MAX_CHARS, M.AXES_MIN_FRACTION
check_title, axes_height_fraction, _title, _layout, _label = M.check_title, M.axes_height_fraction, M._title, M._layout, M._label
_fit_title_line = M._fit_title_line


def cell_family(cell: str) -> str:
    if cell == CELL_NONE:
        return FAMILY_NONE
    if cell == M.CELL_PROMPTED_CONST:
        return FAMILY_PROMPTED_CONST
    if cell.startswith("prompted__"):
        return FAMILY_PROMPTED
    if cell.startswith("safety_refusal__"):
        return FAMILY_SI12
    if cell.endswith(("_minus_ctl", "_distilled")):
        return FAMILY_DERIVED
    if cell.startswith("constv2ctl2"):
        return FAMILY_CTL
    return FAMILY_CONST


_pp = M._pp


def figure_excluded_families(report: Mapping[str, Any]) -> frozenset[str]:
    """The model families the figures and the rendered tables leave out —
    ``figures.excluded_families`` of the report (:func:`make_figures` draws from
    a view that carries it); empty when nothing is excluded."""
    return frozenset(_mapping(report.get("figures")).get(FIGURES_EXCLUDED_KEY) or ())


#: the two further PRESENTATION selectors (plan P5; lane B B11 / B14): ``figures.excluded_arms`` — matrix CELL ids
#: left out of every figure (``--figure-exclude-arm``, validated against the report's cells) — and ``figures.modes`` — the
#: sampling modes DRAWN (``--figure-modes``; None = every mode; GSM8K has the one thinking-card mode). Like the family
#: exclusion they select the rows SHOWN and recompute no aggregate; the tables JSON keeps its family-exclusion behaviour.
#: The keys and their readers are the MATH-500 module's (one spelling across both instruments; B14: PR-A's key).
FIGURES_EXCLUDED_ARMS_KEY, FIGURES_MODES_KEY, FIGURES_SELECTION_KEY = M.FIGURES_EXCLUDED_ARMS_KEY, M.FIGURES_MODES_KEY, M.FIGURES_SELECTION_KEY
figure_excluded_arms, figure_modes, figure_selection_active = M.figure_excluded_arms, M.figure_modes, M.figure_selection_active


def _row_shown(report: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    """A row (a controls / truncation / contrast / matrix row: ``model_key``, ``cell``) passes all three selectors (every
    GSM8K row is the one thinking-card mode)."""
    if row.get("model_key") in figure_excluded_families(report):
        return False
    if row.get("cell") in figure_excluded_arms(report):
        return False
    modes = figure_modes(report)
    return modes is None or MODE_PRIMARY in modes


def _shown_rows(report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """The rows every selector lets through (family, arm, mode)."""
    return [r for r in rows if _row_shown(report, r)]


def _shown(report: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> list[tuple[int, Mapping[str, Any]]]:
    """``(index, row)`` for the shown rows — the index is the row's position in
    the report's OWN list, so every sidecar path still resolves in the full report."""
    return [(i, r) for i, r in enumerate(rows) if _row_shown(report, r)]


def _selection_line(report: Mapping[str, Any]) -> str | None:
    """The title clause every figure carries under an arm / mode selection: the full-report leg count and the count
    shown — never a recomputed aggregate (plan P5). ``None`` when neither selector is active (B11)."""
    if not figure_selection_active(report):
        return None
    rows = _leg_rows(report)
    return f"{len(rows)} legs in report · {len(_shown_rows(report, rows))} shown"


def _with_selection(report: Mapping[str, Any], line: str, short: str | None = None) -> str:
    """A figure's second title line with the selection clause appended (bounded by ``_fit_title_line``); under a selection
    the SHORT form of the line (when given) makes room for both counts inside the title rule."""
    sel = _selection_line(report)
    if sel is None:
        return _fit_title_line(line)
    return _fit_title_line(f"{short if short is not None else line} · {sel}")


def _selection_suffix(report: Mapping[str, Any]) -> str:
    """The selection clause as a figure's OWN second title line (``"\\n<clause>"``), ``""`` when neither selector is active."""
    sel = _selection_line(report)
    return f"\n{sel}" if sel else ""


def _families_present(report: Mapping[str, Any]) -> list[str]:
    """The cell families the report's cells belong to, in palette order — the legend names these (the published report
    carries every published family, so its legend is unchanged)."""
    present = {cell_family(str(c["cell"])) for c in report["cells"].values()}
    return [f for f in FAMILY_COLOR if f in present]


def _shown_models(report: Mapping[str, Any]) -> list[str]:
    excluded = figure_excluded_families(report)
    return [m for m in report["design"]["models"] if m not in excluded]


def _families_text(n: int) -> str:
    return f"{n} famil{'y' if n == 1 else 'ies'} shown"


def _short(cell: str) -> str:
    return cell.replace("constv2-mix-full__", "const:").replace("constv2ctl2-full__", "ctl:").replace("safety_refusal__", "si12:").replace("prompted__", "prompted:")


NA_LABEL = thinking_label(THINKING_ARM_NOT_AVAILABLE)


def _arm_of_model(report: Mapping[str, Any], model_key: str) -> str:
    """A model's arm as the report's design block files it (``thinking_mode_by_model``);
    a report from before the field (the published wave) reads every model as ``on``."""
    return str(_mapping(_mapping(report.get("design")).get("thinking_mode_by_model")).get(model_key) or THINKING_ARM_ON)


def _design_models(report: Mapping[str, Any]) -> list[str]:
    """A report's ``design.models`` (empty when the block is absent — a peer document read defensively)."""
    return list(_mapping(report.get("design")).get("models") or [])


def _na_suffix(arm: object) -> str:
    """The label appended to every figure / table row of a no-switch leg."""
    return f" · {NA_LABEL}" if arm == THINKING_ARM_NOT_AVAILABLE else ""


_FINDING_TRIGGER = re.compile(r"> the \S+ % trigger \(")


def _trigger_words(report: Mapping[str, Any]) -> str:
    """'the 1 % trigger' / 'the 3 % trigger' / 'its trigger (per model)' — the report's, for prose."""
    t = _mapping(report.get("truncation")).get("trigger_threshold")
    return f"the {_pct(t)} trigger" if t is not None else "its trigger (per model: see four_family.trigger_threshold_by_model)"


def _trigger_caption_text(report: Mapping[str, Any]) -> str:
    tr = _mapping(report.get("truncation"))
    t = tr.get("trigger_threshold")
    if t is None:
        by = _mapping(_mapping(report.get("four_family")).get("trigger_threshold_by_model"))
        return "per model: " + ", ".join(f"{m} {_pct(v)}" for m, v in by.items()) if by else "per model"
    return f"dash-dot at {_pct(t)}" if not _close(float(t), V.GSM8K_REPORT_LINE) else f"the dashed {_pct(t)} line itself"


#: the footer sits BELOW the figure canvas, a fixed distance under its bottom edge (in inches, so a short figure's x label —
#: which overflows the 9.5 % bottom margin — is cleared too): `_save`'s tight bounding box takes it into the PNG / PDF, the
#: axes keep their >= 80 % of the canvas height, and nothing collides with the x labels
FOOTER_DROP_IN = 0.34


def _contrast_index(report: Mapping[str, Any]) -> dict[str, int]:
    return {c["name"]: i for i, c in enumerate(report["contrasts"])}


def _fig_contrasts_model(report: Mapping[str, Any], model_key: str) -> tuple[Any, list[dict[str, Any]]]:
    plt, mcolors, Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    idx = _contrast_index(report)
    rows = [c for c in report["contrasts"] if c["model_key"] == model_key and _row_shown(report, c)]
    held = [h for h in report["held"] if h["model_key"] == model_key and _row_shown(report, h)]
    deferred = [d for d in (report.get("deferred") or []) if d["model_key"] == model_key and _row_shown(report, d)]
    n_rows = len(rows) + len(held) + len(deferred)
    fig, ax = plt.subplots(figsize=(9.0, 0.34 * max(n_rows, 1) + 1.5))
    _layout(fig, left=0.36, right=0.84)
    ax.axvline(0.0, color=BASELINE, lw=0.8, zorder=2)
    # the noise-floor band: the merged stock/primary CI of none(it) - none(main), when computable
    nf = next((n for n in report["noise_floor"] if n["model_key"] == model_key and n["computable"]), None)
    if nf is not None:
        st = nf["readings"][READING_MERGED]["estimands"][AXIS_STOCK][BRANCH_PRIMARY]["stats"][CONTRAST_NOISE]
        ax.axvspan(_pp(st["ci_low"]), _pp(st["ci_high"]), color=mcolors.to_rgba(MUTED, 0.12), zorder=1)
        j = report["noise_floor"].index(nf)
        T._num_at(numbers, report, f"noise_floor.ci_low[{model_key}]", f"noise_floor.{j}.readings.{READING_MERGED}.estimands.{AXIS_STOCK}.{BRANCH_PRIMARY}.stats.{CONTRAST_NOISE}.ci_low", st["ci_low"])
        T._num_at(numbers, report, f"noise_floor.ci_high[{model_key}]", f"noise_floor.{j}.readings.{READING_MERGED}.estimands.{AXIS_STOCK}.{BRANCH_PRIMARY}.stats.{CONTRAST_NOISE}.ci_high", st["ci_high"])
    labels: list[str] = []
    y = n_rows - 1
    for c in rows:
        labels.append(f"{c['view']}: {_short(c['cell'])}")
        color = FAMILY_COLOR[cell_family(c["cell"])]
        if not c["computable"]:
            ax.text(0.02, y, (f"refused by name: {c.get('expected_refusal') or c.get('certified_refusal')}"
                              if (c.get("expected_refusal") or c.get("certified_refusal")) else f"no estimand: {str(c['reason'])[:60]}"),
                    transform=ax.get_yaxis_transform(), va="center", fontsize=6.4, color=MUTED)
            y -= 1
            continue
        j = idx[c["name"]]
        base = f"contrasts.{j}.readings"
        pm = c["readings"][READING_MERGED]["estimands"][AXIS_STOCK][BRANCH_PRIMARY]["stats"][CONTRAST_VS_NONE]
        p8 = c["readings"][READING_R8192]["estimands"][AXIS_STOCK][BRANCH_PRIMARY]["stats"][CONTRAST_VS_NONE]
        ax.plot([_pp(pm["ci_low"]), _pp(pm["ci_high"])], [y + 0.14, y + 0.14], color=INK, lw=1.1, zorder=5)
        ax.plot([_pp(pm["point"])], [y + 0.14], marker="o", ms=5.5, color=color, mec=INK, mew=0.5, ls="", zorder=6)
        ax.plot([_pp(p8["ci_low"]), _pp(p8["ci_high"])], [y - 0.14, y - 0.14], color=INK2, lw=0.8, zorder=4)
        ax.plot([_pp(p8["point"])], [y - 0.14], marker="o", ms=5, mfc="none", mec=INK2, mew=1.0, ls="", zorder=6)
        for axis, marker in ((AXIS_W, "v"), (AXIS_X, "^")):
            pa = c["readings"][READING_MERGED]["estimands"][axis][BRANCH_PRIMARY]["stats"][CONTRAST_VS_NONE]
            ax.plot([_pp(pa["point"])], [y + 0.14], marker=marker, ms=3.8, color=INK2, ls="", zorder=7)
            T._num_at(numbers, report, f"{c['name']}.merged.{axis}.point", f"{base}.{READING_MERGED}.estimands.{axis}.{BRANCH_PRIMARY}.stats.{CONTRAST_VS_NONE}.point", pa["point"])
        for f in STAT_FIELDS:
            T._num_at(numbers, report, f"{c['name']}.merged.stock.{f}", f"{base}.{READING_MERGED}.estimands.{AXIS_STOCK}.{BRANCH_PRIMARY}.stats.{CONTRAST_VS_NONE}.{f}", pm[f])
            T._num_at(numbers, report, f"{c['name']}.r8192.stock.{f}", f"{base}.{READING_R8192}.estimands.{AXIS_STOCK}.{BRANCH_PRIMARY}.stats.{CONTRAST_VS_NONE}.{f}", p8[f])
        word = c["readings"][READING_MERGED]["verdict"]["word"]
        ax.text(1.005, y, word, transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.4, color=INK if word in WORDS else MUTED, clip_on=False)
        y -= 1
    for h in held:
        labels.append(f"{h['view']}: {_short(h['cell'])}")
        # the published hold label names the superseded direct-fit hold; an it-doc row (the re-fit itself) is named by ITS reason
        text = f"HELD — {str(h['reason'])[:64]}" if V.is_itdoc_row(h) else "HELD (D8: re-fit at effective batch 64)"
        ax.text(0.02, y, text, transform=ax.get_yaxis_transform(), va="center", fontsize=6.4, color=MUTED)
        y -= 1
    for d in deferred:
        labels.append(f"{d['view']}: {_short(d['cell'])}")
        ax.text(0.02, y, f"DEFERRED — {str(d['reason'])[:64]}", transform=ax.get_yaxis_transform(), va="center", fontsize=6.4, color=MUTED)
        y -= 1
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(list(reversed(labels)), fontsize=6.8)
    ax.set_ylim(-0.7, n_rows - 0.3)
    ax.set_xlabel("accuracy difference vs the view's none (percentage points)")
    T._recessive(ax)
    ax.xaxis.grid(True, zorder=0)
    ax.yaxis.grid(False)
    handles = [plt.Line2D([], [], marker="o", color=INK, mec=INK, ls="", label="merged reading (stock, primary), 95 % CI"),
               plt.Line2D([], [], marker="o", mfc="none", mec=INK2, ls="", label="8,192-only reading (stock, primary), 95 % CI"),
               plt.Line2D([], [], marker="v", color=INK2, ls="", label="W: residual at-cap wrong"), plt.Line2D([], [], marker="^", color=INK2, ls="", label="X: residual at-cap excluded"),
               Patch(facecolor=mcolors.to_rgba(MUTED, 0.12), label="noise-floor band: none(it) - none(main)")]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=6.0)
    _title(fig, f"{model_key}: every cell minus its view's none (paired items, B = {report['bootstrap']['B']:,})\n"
                + _with_selection(report, "merged = headline; 8,192-only, W and X beside it; noise-floor band; refused / held named",
                                  short="merged headline; 8,192-only, W and X beside; noise band"))
    return fig, numbers


def _make_contrast_builders(models: Sequence[str]) -> dict[str, Callable[[Mapping[str, Any]], tuple[Any, list[dict[str, Any]]]]]:
    return {f"final_g8k_contrasts_{m}": (lambda report, m=m: _fig_contrasts_model(report, m)) for m in models}


def fig_accuracy_by_cell(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mcolors, Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    excluded = figure_excluded_families(report)
    views = [vw for vw in report["design"]["views"] if vw.split("/", 1)[0] not in excluded]
    fig, axes = plt.subplots(1, len(views), figsize=(2.4 * len(views) + 1.4, 4.4), sharey=True, squeeze=False)
    _layout(fig, left=0.06, wspace=0.10)
    cells = report["cells"]
    for ax, vw in zip(axes[0], views):
        key, view = vw.split("/", 1)
        mine = [ck for ck in sorted(cells) if cells[ck]["model_key"] == key and cells[ck]["view"] == view and cells[ck].get("readings")
                and _row_shown(report, cells[ck])]
        for x, ck in enumerate(mine):
            rd = cells[ck]["readings"]
            m, r8 = rd[READING_MERGED], rd[READING_R8192]
            color = FAMILY_COLOR[cell_family(cells[ck]["cell"])]
            if m["acc_stock"] is not None:
                ax.bar([x], [m["acc_stock"]], width=0.64, facecolor=color, edgecolor="none", zorder=3)
                T._num_at(numbers, report, f"acc_stock.merged[{ck}]", f"cells.{ck}.readings.{READING_MERGED}.acc_stock", m["acc_stock"])
            if r8["acc_stock"] is not None:
                ax.plot([x], [r8["acc_stock"]], marker="o", ms=4.5, mfc="none", mec=INK, mew=1.0, ls="", zorder=6)
                T._num_at(numbers, report, f"acc_stock.r8192[{ck}]", f"cells.{ck}.readings.{READING_R8192}.acc_stock", r8["acc_stock"])
        ax.set_xticks(range(len(mine)))
        ax.set_xticklabels([_short(cells[ck]["cell"]) for ck in mine], fontsize=5.6, rotation=60, ha="right")
        ax.set_ylim(0, 1.06)
        _label(ax, vw + _na_suffix(_arm_of_model(report, key)))
        T._recessive(ax)
    axes[0][0].set_ylabel("GSM8K accuracy (stock scorer, own scored denominator)")
    handles = [Patch(facecolor=FAMILY_COLOR[f], edgecolor="none", label=FAMILY_LABEL[f]) for f in _families_present(report)]
    handles.append(plt.Line2D([], [], marker="o", mfc="none", mec=INK, ls="", label="8,192-only reading"))
    axes[0][-1].legend(handles=handles, loc="lower right", frameon=False, fontsize=5.6)
    _title(fig, "GSM8K accuracy per cell and view, thinking ON: ladder-merged bars, 8,192-only markers" + _selection_suffix(report))
    return fig, numbers


_leg_rows = M._leg_rows


def _leg_short(row: Mapping[str, Any]) -> str:
    return f"{row['model_key']}/{row['view']} {_short(row['cell'])}" + (" @16k" if row["rung"] else "") + _na_suffix(row.get("thinking_mode"))


_ctl_index = M._ctl_index


_recessive_xgrid = M._recessive_xgrid


def fig_atcap_share(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mcolors, Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    tr = report["truncation"]
    shown = _shown(report, tr["cells"])
    cells = [c for _i, c in shown]
    hidden_legs = {c["leg"] for c in tr["cells"] if not _row_shown(report, c)}
    fig, ax = plt.subplots(figsize=(9.0, 0.22 * max(len(cells), 1) + 1.6))
    _layout(fig, left=0.30)
    ys = list(range(len(cells)))[::-1]
    for y, (i, c) in zip(ys, shown):
        share = c["share"] or 0.0
        ax.barh([y], [_pp(share)], height=0.62, color=FAMILY_COLOR[cell_family(c["cell"])], edgecolor="none", zorder=3)
        if c["share"] is not None:
            T._num_at(numbers, report, f"at_cap.share[{c['leg']}]", f"truncation.cells.{i}.share", c["share"])
        if c["escalated"]:
            ax.text(_pp(share) + 0.15, y, f"escalated: {c['n_eligible']} redrawn", va="center", fontsize=5.6, color=INK2)
        elif c["trigger_exceeded"]:
            ax.text(_pp(share) + 0.15, y, f"{c['exit']}", va="center", fontsize=5.6, color=INK2)
    lines_coincide = V.GSM8K_TRIGGER_THRESHOLD == V.GSM8K_REPORT_LINE
    if not lines_coincide:
        ax.axvline(_pp(V.GSM8K_REPORT_LINE), color=INK2, lw=0.9, ls="--", zorder=4)
    ax.axvline(_pp(V.GSM8K_TRIGGER_THRESHOLD), color=CAT["orange"], lw=0.9, ls="-.", zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{c['model_key']}/{c['view']} {_short(c['cell'])}" for c in cells], fontsize=6.0)
    top = max([_pp(c["share"] or 0.0) for c in cells] + [4.0])
    ax.set_xlim(0, top * 1.3)
    ax.set_xlabel("at-cap draws at 8,192, % of draws with a stop reason", fontsize=7.4)
    _recessive_xgrid(ax)
    trigger_label = (f"1 % report line = {TRIGGER_PCT} trigger (rung at 16,384)" if lines_coincide else f"{TRIGGER_PCT} trigger (rung at 16,384)")
    ax.legend(handles=([] if lines_coincide else [plt.Line2D([], [], color=INK2, ls="--", label="1 % report line")])
              + [plt.Line2D([], [], color=CAT["orange"], ls="-.", label=trigger_label)]
              + [Patch(color=FAMILY_COLOR[f], label=FAMILY_LABEL[f]) for f in _families_present(report)], loc="lower right", frameon=False, fontsize=5.6)
    n_trig_report, n_over_report = len(tr["triggered"]), len(tr["over_1pct"])
    if figure_excluded_families(report) or figure_selection_active(report):
        # an aggregate over the cells shown is LABELLED as such, with the wave's value beside it (title and sidecar) — never a silent substitution
        n_trig, n_over = (len([leg for leg in tr[k] if leg not in hidden_legs]) for k in ("triggered", "over_1pct"))
        fams = _shown_models(report)
        what = (f"count of the legs in the list whose family is among the {len(fams)} shown ({', '.join(fams)})" if not figure_selection_active(report)
                else "count of the legs in the list the figure shows (family / arm / mode selection)")
        for name, shown_v, report_v in (("triggered", n_trig, n_trig_report), ("over_1pct", n_over, n_over_report)):
            T._num_at(numbers, report, f"at_cap.n_{name}_shown", f"truncation.{name}", shown_v, transform=what)
            T._num_at(numbers, report, f"at_cap.n_{name}_report", f"truncation.{name}", report_v, transform="length of the list (the wave's value)")
        if figure_excluded_families(report) and not figure_selection_active(report):
            second = f"over the {_families_text(len(fams))}: {n_trig} triggered, {n_over} over the 1 % line (wave: {n_trig_report}, {n_over_report})"
        else:
            second = f"shown: {n_trig} triggered, {n_over} over the 1 % line (wave: {n_trig_report}, {n_over_report})"
    else:
        second = f"{n_trig_report} cell(s) triggered, {n_over_report} over the 1 % line; escalated cells annotated"
    _title(fig, f"At-cap share per cell at the 8,192 cap against the 1 % line and the {TRIGGER_PCT} trigger\n" + _with_selection(report, second))
    return fig, numbers


def fig_atcap_classes(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, mcolors, Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    rows = [r for r in _shown_rows(report, _leg_rows(report)) if r.get("at_cap")]
    idx = _ctl_index(report)
    fig, ax = plt.subplots(figsize=(9.0, 0.22 * max(len(rows), 1) + 1.6))
    _layout(fig, left=0.30)
    ys = list(range(len(rows)))[::-1]
    widest = 0
    for y, row in zip(ys, rows):
        i, ac = idx[row["leg"]], row["at_cap"]
        left = 0
        for f, color, hatch in (("n_loop", CAT["orange"], None), ("n_truncated_reasoning", CAT["blue"], None), ("n_text_unavailable", mcolors.to_rgba(MUTED, 0.55), "////")):
            ax.barh([y], [ac[f]], left=[left], height=0.62, color=color, edgecolor="none" if hatch is None else MUTED, hatch=hatch, linewidth=0, zorder=3)
            left += ac[f]
            T._num_at(numbers, report, f"at_cap.{f}[{row['leg']}]", f"controls.{i}.at_cap.{f}", ac[f])
        widest = max(widest, left)
    ax.set_yticks(ys)
    ax.set_yticklabels([_leg_short(r) for r in rows], fontsize=6.0)
    ax.set_xlim(0, max(1, widest) * 1.2)
    ax.set_xlabel("at-cap draws: degenerate loop (Evals' rule, decides eligibility) | truncated reasoning (eligible) | text unavailable", fontsize=6.8)
    _recessive_xgrid(ax)
    sources = {r["at_cap"]["split_source"] for r in rows}
    src = "classified from the eval logs (every draw)" if sources == {SPLIT_SOURCE_EVAL_LOG} else ("store rows only (unscored draws' text not carried)" if sources <= {SPLIT_SOURCE_STORE_ROWS} else "mixed sources per cell")
    ax.legend(handles=[Patch(color=CAT["orange"], label="degenerate loop (non-terminating)"), Patch(color=CAT["blue"], label="truncated reasoning (eligible)"),
                       Patch(facecolor=mcolors.to_rgba(MUTED, 0.55), edgecolor=MUTED, hatch="////", label="text unavailable to the split")],
              loc="lower right", frameon=False, fontsize=6.0, title=src, title_fontsize=6.0)
    _title(fig, "At-cap draws per leg: degenerate loop vs truncated reasoning under Evals' degeneracy rule\n"
                + _with_selection(report, "the periodic-block rule is a second, non-deciding column in the report",
                                  short="periodic-block rule: a non-deciding column"))
    return fig, numbers


def fig_truncation_cost(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mcolors, _Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    shown = _shown(report, report["truncation_cost"])
    costs = [t for _j, t in shown]
    fig, ax = plt.subplots(figsize=(8.6, 0.40 * max(len(costs), 1) + 1.6))
    _layout(fig, left=0.34, right=0.86)
    ax.axvline(0.0, color=BASELINE, lw=0.8, zorder=2)
    ys = list(range(len(costs)))[::-1]
    for y, (j, t) in zip(ys, shown):
        st = t["estimands"][BRANCH_PRIMARY]["stats"][CONTRAST_TRUNCATION]
        ax.plot([_pp(st["ci_low"]), _pp(st["ci_high"])], [y, y], color=INK, lw=1.1, zorder=5)
        ax.plot([_pp(st["point"])], [y], marker="o", ms=5.5, color=FAMILY_COLOR[cell_family(t["cell"])], mec=INK, mew=0.5, ls="", zorder=6)
        ax.text(1.005, y, f"redrawn {t['n_subset']}; residual at 16k {t['residual_at_ceiling']}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.2, color=INK2, clip_on=False)
        for f in STAT_FIELDS:
            T._num_at(numbers, report, f"truncation_cost.{t['cell_key']}.{f}", f"truncation_cost.{j}.estimands.{BRANCH_PRIMARY}.stats.{CONTRAST_TRUNCATION}.{f}", st[f])
        T._num_at(numbers, report, f"truncation_cost.{t['cell_key']}.residual", f"truncation_cost.{j}.residual_at_ceiling", t["residual_at_ceiling"])
    n_report = len(report["truncation_cost"])
    if not costs:
        # A filter never rewrites a conclusion — an empty figure under an exclusion says the wave DID escalate
        excluded = sorted(figure_excluded_families(report))
        arms = figure_excluded_arms(report)
        by_fam = [t for t in report["truncation_cost"] if t["model_key"] in excluded]
        by_arm = [t for t in report["truncation_cost"] if t["model_key"] not in excluded and not _row_shown(report, t)]
        if n_report and (by_fam or by_arm):
            why = " / ".join(([f"in the excluded famil{'y' if len(excluded) == 1 else 'ies'} ({', '.join(excluded)})"] if by_fam else [])
                             + ([f"carry an excluded arm ({', '.join(arms)})"] if by_arm else []))
            msg = f"no escalated cell shown: the wave's {n_report} escalated cell(s) are {why}"
        else:
            msg = f"no cell escalated: every cell at or below {_trigger_words(report)}"
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center", va="center", fontsize=8, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{t['model_key']}/{t['view']} {_short(t['cell'])}" + _na_suffix(t.get("thinking_mode")) for t in costs], fontsize=6.6)
    ax.set_ylim(-0.7, max(len(costs), 1) - 0.3)
    ax.set_xlabel("truncation cost: merged - 8,192-only accuracy (percentage points; stock scorer, primary branch)")
    _recessive_xgrid(ax)
    n_report = len(report["truncation_cost"])
    if figure_excluded_families(report):
        n_fam = len(_shown_models(report))
        T._num_at(numbers, report, "truncation_cost.n_escalated_shown", "truncation_cost", len(costs), transform=f"count of the escalated cells whose family is among the {n_fam} shown")
        T._num_at(numbers, report, "truncation_cost.n_escalated_report", "truncation_cost", n_report, transform="length of the list (the wave's value)")
        second = f"over the {_families_text(n_fam)}: {len(costs)} of the wave's {n_report} escalated cells; residual at 16,384 annotated"
    else:
        second = f"{n_report} escalated cell(s); the residual at the 16,384 ceiling annotated"
    _title(fig, "Truncation cost per escalated cell: ladder-merged minus 8,192-only accuracy\n"
                + _with_selection(report, second, short=f"shown: {len(costs)} of the wave's {n_report} escalated cell(s)"))
    return fig, numbers


def fig_overlap(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, mcolors, Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    ov = report.get("overlap") or {}
    models = [m for m in _shown_models(report) if m in _mapping(ov.get("models"))]
    fig, ax = plt.subplots(figsize=(8.0, 0.5 * max(len(models), 1) + 1.8))
    _layout(fig, left=0.22, right=0.82)
    ax.axvspan(-5.0, 5.0, color=mcolors.to_rgba(CAT["aqua"], 0.10), zorder=1)
    ax.axvline(0.0, color=BASELINE, lw=0.8, zorder=2)
    ys = list(range(len(models)))[::-1]
    for y, key in zip(ys, models):
        b = ov["models"][key]
        cmp_ = b.get("comparison") or {}
        if b.get("status") == OVERLAP_STATUS_NOT_APPLICABLE:
            # P12: the no-switch family's N/A card — no thinking-OFF arm exists to overlap-check
            ax.text(0.02, y, f"not applicable ({NA_LABEL}): {str(b.get('why'))[:80]}", transform=ax.get_yaxis_transform(), va="center", fontsize=6.4, color=MUTED)
            continue
        if not cmp_ or cmp_.get("delta") is None:
            ax.text(0.02, y, f"not usable: {str(b.get('reason'))[:70]}", transform=ax.get_yaxis_transform(), va="center", fontsize=6.4, color=MUTED)
            continue
        ax.plot([_pp(cmp_["ci_low"]), _pp(cmp_["ci_high"])], [y, y], color=INK, lw=1.1, zorder=5)
        ax.plot([_pp(cmp_["delta"])], [y], marker="o", ms=6, color=CAT["blue"], mec=INK, mew=0.5, ls="", zorder=6)
        ax.text(1.005, y, f"{b['verdict']} (n = {cmp_['n_pairs']}, b+c = {cmp_['disagreement_b_plus_c']})", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=6.4, color=INK if b["verdict"] == "continuation" else INK2, clip_on=False)
        for f in ("delta", "ci_low", "ci_high", "acc_stock", "acc_retired", "n_pairs", "disagreement_b_plus_c"):
            T._num_at(numbers, report, f"overlap.{key}.{f}", f"overlap.models.{key}.comparison.{f}", cmp_[f])
    if not models:
        ax.text(0.5, 0.5, "no overlap record", transform=ax.transAxes, ha="center", va="center", fontsize=8, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([m + _na_suffix(_arm_of_model(report, m)) for m in models], fontsize=7.0)
    ax.set_ylim(-0.7, max(len(models), 1) - 0.3)
    ax.set_xlabel("stock (OFF, greedy, 512) - retired s2parity (OFF, greedy, 512) accuracy, percentage points")
    _recessive_xgrid(ax)
    ax.legend(handles=[Patch(facecolor=mcolors.to_rgba(CAT["aqua"], 0.10), label="±5 pp tolerance (continuation iff the CI lies inside)"),
                       plt.Line2D([], [], marker="o", color=CAT["blue"], mec=INK, ls="", label="paired difference, 95 % bootstrap CI")], loc="lower left", frameon=False, fontsize=6.0)
    _title(fig, "Overlap per model: stock vs retired instrument, same 200 items (thinking OFF, greedy)\n"
                + _with_selection(report, "two instruments, paired on item_id, never pooled"))
    return fig, numbers


def fig_ledger(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mcolors, Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    rows = _shown_rows(report, _leg_rows(report))
    idx = _ctl_index(report)
    fig, ax = plt.subplots(figsize=(9.0, 0.22 * max(len(rows), 1) + 1.6))
    _layout(fig, left=0.30)
    ys = list(range(len(rows)))[::-1]
    for y, row in zip(ys, rows):
        i, led = idx[row["leg"]], row["ledger"]
        n_calls = led.get("n_calls") or 0
        others = sum((a.get("n_forwarded") or 0) + (a.get("n_rejected") or 0) for a in (row.get("attempts") or {}).values() if not a.get("accepted"))
        ax.barh([y], [n_calls], height=0.62, color=CAT["aqua"], edgecolor="none", zorder=3)
        ax.barh([y], [others], left=[n_calls], height=0.62, color=CAT["yellow"], edgecolor="none", zorder=3)
        ax.barh([y], [led.get("n_rejected") or 0], left=[n_calls + others], height=0.62, color=CAT["orange"], edgecolor="none", zorder=3)
        ax.text(n_calls + others + (led.get("n_rejected") or 0) + 3, y, f"{n_calls} accepted, {others} attributed, {led.get('n_rejected') or 0} rejected, "
                f"{sum((led.get('error_code') or {}).values())} coded", va="center", fontsize=5.6, color=INK2)
        T._num_at(numbers, report, f"ledger.n_calls[{row['leg']}]", f"controls.{i}.ledger.n_calls", led.get("n_calls"))
        T._num_at(numbers, report, f"ledger.n_rejected[{row['leg']}]", f"controls.{i}.ledger.n_rejected", led.get("n_rejected"))
    ax.set_yticks(ys)
    ax.set_yticklabels([_leg_short(r) for r in rows], fontsize=6.0)
    ax.set_xlim(0, N_ITEMS * 1.9)
    ax.set_xlabel("usage-ledger calls of the task per leg", fontsize=7.4)
    _recessive_xgrid(ax)
    ax.legend(handles=[Patch(color=CAT["aqua"], label="accepted attempt, forwarded"), Patch(color=CAT["yellow"], label="other recorded attempts (attributed)"),
                       Patch(color=CAT["orange"], label="rejected in the accepted attempt")], loc="lower right", frameon=False, fontsize=6.0)
    _title(fig, "Ledger per leg: accepted attempt's calls, other attempts' rows (attributed), rejections" + _selection_suffix(report))
    return fig, numbers


def fig_completion_tokens(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, mcolors, _Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    rows = _shown_rows(report, _leg_rows(report))
    idx = _ctl_index(report)
    fig, ax = plt.subplots(figsize=(9.0, 0.22 * max(len(rows), 1) + 1.6))
    _layout(fig, left=0.30)
    ys = list(range(len(rows)))[::-1]
    for y, row in zip(ys, rows):
        i, s, led = idx[row["leg"]], row["summary"], row["ledger"]
        qs = led.get("completion_tokens_quantiles") or {}
        if qs:
            ax.plot([qs["p50"], qs["p99"]], [y, y], color=mcolors.to_rgba(CAT["blue"], 0.55), lw=3.0, solid_capstyle="butt", zorder=3)
            for k in ("p50", "p99"):
                T._num_at(numbers, report, f"ledger.completion_tokens.{k}[{row['leg']}]", f"controls.{i}.ledger.completion_tokens_quantiles.{k}", qs[k])
        if s.get("mean_completion_tokens") is not None:
            ax.plot([s["mean_completion_tokens"]], [y], marker="o", ms=4.5, color=INK, ls="", zorder=6)
            T._num_at(numbers, report, f"mean_completion_tokens[{row['leg']}]", f"controls.{i}.summary.mean_completion_tokens", s["mean_completion_tokens"])
    ax.axvline(CAP_BASE, color=CAT["orange"], lw=0.9, ls="--", zorder=4)
    ax.axvline(CAP_RUNG, color=CAT["magenta"], lw=0.9, ls="-.", zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([_leg_short(r) for r in rows], fontsize=6.0)
    ax.set_xlabel("completion tokens: summary mean (dot), ledger p50-p99 (band); dashed 8,192 cap, dash-dot 16,384 ceiling", fontsize=7.0)
    _recessive_xgrid(ax)
    _title(fig, "Completion lengths per leg against the 8,192 cap and the 16,384 ceiling" + _selection_suffix(report))
    return fig, numbers


# --------------------------------------------------------------------------- #
# the eight-arm figures for the MAIN view: eight FIXED arm slots per model in   #
# panel order (gemma / qwen / nemotron / llama), the ladder-MERGED reading as   #
# the headline bar with the 8,192-only reading beside it, a deferred / held /   #
# absent slot drawn EMPTY and lettered (never dropped), the llama group from    #
# the Llama peer report (else "pending — Llama lane"), the pooled group from    #
# report["eightarm"] — nothing is recomputed here                             #
# --------------------------------------------------------------------------- #

EIGHTARM_STEM_ACCURACY, EIGHTARM_STEM_CONTRASTS = "final_g8k_eightarm_accuracy", "final_g8k_eightarm_contrasts"
EIGHTARM_STEMS: tuple[str, ...] = (EIGHTARM_STEM_ACCURACY, EIGHTARM_STEM_CONTRASTS)
POOLED_GROUP_LABEL, SLOT_HATCH = M.POOLED_GROUP_LABEL, M.SLOT_HATCH
SLOT_STATUS_PENDING_PEER = "pending_peer"
#: the letter an EMPTY slot wears, per design status: the MATH-500 module's table plus this instrument's three further
#: states (a pending peer column; a missing / refused REQUIRED leg)
SLOT_LETTER: Mapping[str, str] = {**M.SLOT_LETTER, SLOT_STATUS_PENDING_PEER: "p", "missing": "m", "refused": "r"}
SLOT_LEGEND = ("empty slot — D deferred (dose unmeasured, D2/D3) · H held · n not analysed · – absent · m missing · r refused · "
               "u pooled unavailable · p pending (Llama lane)")


def _eightarm_group_label(report: Mapping[str, Any], model_key: str) -> str:
    """The x label of a model's group: a peer column's label DERIVED from the peer's thinking metadata (C3 — it used to
    hard-code "thinking: not available"), the pending label for the Llama column without a peer, the bare key otherwise."""
    _pin, peer = _peer_of(report, model_key)
    if peer is not None:
        return f"{model_key}\n{peer_thinking_label(peer, model_key)}"
    if model_key in POOLED_HELD_OUT:
        return f"{model_key}\n{PEER_PENDING_LABEL}"
    return model_key


def _slot_status(report: Mapping[str, Any], model_key: str, kind: str) -> str:
    design = report["design"]
    status = str(_mapping(_mapping(design.get("arms_by_model")).get(model_key)).get(kind, ROW_STATUS_ABSENT))
    if status == ROW_STATUS_REQUIRED:
        ck = V.cell_key(model_key, VIEW_MAIN, ARM_BY_KIND[kind].cell)
        c = _mapping(report["cells"].get(ck))
        base = _mapping(_mapping(report.get("legs")).get(str(c.get("base_leg"))))
        return "missing" if base.get("missing") else ("refused" if base.get("refused") else "no estimand")
    return status


def _empty_slot(ax, x: float, width: float, status: str, *, top: float = 0.05) -> None:
    """:func:`M._empty_slot` with this instrument's letter table (:data:`SLOT_LETTER`)."""
    M._empty_slot(ax, x, width, status, top=top, letters=SLOT_LETTER)


def _peer_of(report: Mapping[str, Any], model_key: str) -> tuple[str | None, Mapping[str, Any] | None]:
    """``(pin, slim peer report)`` naming ``model_key`` among ``report["peer_reports"]``, else ``(None, None)``."""
    for pin, doc in _mapping(report.get("peer_reports")).items():
        if model_key in (doc.get("models") or []):
            return str(pin), doc
    return None, None


def _acc_readings(doc: Mapping[str, Any], prefix: str, model_key: str, cell: str) -> dict[str, dict[str, Any]]:
    """Per reading, the cell's stock / primary accuracy with its CI where a computable contrast of the view carries it
    (``contrast_summary[j].<reading>.acc_by_leg_stock_primary.{cell|none}``), else the cell's reading point alone
    (``cells.<ck>.readings.<reading>.acc_stock``); every value with its resolvable path under ``prefix``."""
    out: dict[str, dict[str, Any]] = {}
    ck = V.cell_key(model_key, VIEW_MAIN, cell)
    summaries = list(doc.get("contrast_summary") or [])
    for reading in READINGS:
        rec: dict[str, Any] | None = None
        for j, cs in enumerate(summaries):
            if cs.get("model_key") != model_key or cs.get("view") != VIEW_MAIN or not cs.get("computable") or reading not in cs:
                continue
            leg_label = "none" if cell == CELL_NONE else ("cell" if cs.get("cell") == cell else None)
            if leg_label is None:
                continue
            a = _mapping(_mapping(cs[reading]).get("acc_by_leg_stock_primary")).get(leg_label)
            if a is None:
                continue
            base = f"{prefix}contrast_summary.{j}.{reading}.acc_by_leg_stock_primary.{leg_label}"
            rec = {"point": a["point"], "ci_low": a.get("ci_low"), "ci_high": a.get("ci_high"), "paths": {f: f"{base}.{f}" for f in ("point", "ci_low", "ci_high")}}
            break
        if rec is None:
            c = _mapping(_mapping(doc.get("cells")).get(ck))
            r = _mapping(_mapping(c.get("readings")).get(reading))
            if r.get("acc_stock") is not None:
                rec = {"point": r["acc_stock"], "ci_low": None, "ci_high": None, "paths": {"point": f"{prefix}cells.{ck}.readings.{reading}.acc_stock"}}
        if rec is not None:
            out[reading] = rec
    return out


def fig_eightarm_accuracy(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mcolors, Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    ea = report["eightarm"]
    excluded_arms = figure_excluded_arms(report)
    excluded_families = figure_excluded_families(report)
    models = [m for m in EIGHTARM_MODEL_ORDER if m not in excluded_families]
    slots = M.panel_slots(report)                   # the geometry derives from the arms this report lays out, never len(ARMS)
    arms = [a for a in slots if a.cell not in excluded_arms]
    groups = [*models, POOLED_GROUP_LABEL]
    pitch = 0.105
    width = pitch * 0.86
    fig, ax = plt.subplots(figsize=(1.95 * len(groups) + 2.0, 4.4))
    _layout(fig, left=0.08)
    kinds_drawn: set[str] = set()
    for g, m in enumerate(models):
        pin, peer = _peer_of(report, m)
        doc, prefix = (peer, f"peer_reports.{pin}.") if peer is not None else (report, "")
        in_design = m in (report["design"].get("models") or []) or peer is not None
        for s_, arm in enumerate(slots):
            if arm.cell in excluded_arms:
                continue
            x = g + (s_ - (len(slots) - 1) / 2) * pitch
            if not in_design:
                _empty_slot(ax, x, width, SLOT_STATUS_PENDING_PEER if m in POOLED_HELD_OUT else ROW_STATUS_ABSENT)
                continue
            acc = _acc_readings(doc, prefix, m, arm.cell)
            merged, r8 = acc.get(READING_MERGED), acc.get(READING_R8192)
            if merged is None:
                _empty_slot(ax, x, width, _slot_status(report, m, arm.kind) if peer is None else ROW_STATUS_ABSENT)
                continue
            kinds_drawn.add(arm.kind)
            ax.bar([x], [merged["point"]], width=width, facecolor=KIND_COLOR[arm.kind], edgecolor="none", zorder=3)
            T._num_at(numbers, report, f"eightarm.acc.merged.point[{m}/{arm.kind}]", merged["paths"]["point"], merged["point"])
            if merged.get("ci_low") is not None:
                ax.plot([x, x], [merged["ci_low"], merged["ci_high"]], color=INK, lw=0.9, zorder=5)
                for f in ("ci_low", "ci_high"):
                    T._num_at(numbers, report, f"eightarm.acc.merged.{f}[{m}/{arm.kind}]", merged["paths"][f], merged[f])
            if r8 is not None:
                ax.plot([x], [r8["point"]], marker="o", ms=3.6, mfc="none", mec=INK, mew=0.9, ls="", zorder=6)
                T._num_at(numbers, report, f"eightarm.acc.r8192.point[{m}/{arm.kind}]", r8["paths"]["point"], r8["point"])
    g = len(models)
    for s_, arm in enumerate(slots):
        if arm.cell in excluded_arms:
            continue
        x = g + (s_ - (len(slots) - 1) / 2) * pitch
        merged = _mapping(_mapping(ea.get("pooled_by_arm_merged")).get(arm.kind))
        base = _mapping(_mapping(ea.get("pooled_by_arm_base")).get(arm.kind))
        if merged.get("status") != POOLED_AVAILABLE:
            _empty_slot(ax, x, width, POOLED_UNAVAILABLE)
            continue
        pm = merged[BRANCH_PRIMARY]
        ax.bar([x], [pm["point"]], width=width, facecolor=KIND_COLOR[arm.kind], edgecolor=INK, linewidth=0.4, zorder=3)
        ax.plot([x, x], [pm["ci_low"], pm["ci_high"]], color=INK, lw=0.9, zorder=5)
        for f in ("point", "ci_low", "ci_high"):
            T._num_at(numbers, report, f"eightarm.pooled.merged.{f}[{arm.kind}]", f"eightarm.pooled_by_arm_merged.{arm.kind}.{BRANCH_PRIMARY}.{f}", pm[f])
        if base.get("status") == POOLED_AVAILABLE:
            pb = base[BRANCH_PRIMARY]
            ax.plot([x], [pb["point"]], marker="o", ms=3.6, mfc="none", mec=INK, mew=0.9, ls="", zorder=6)
            T._num_at(numbers, report, f"eightarm.pooled.r8192.point[{arm.kind}]", f"eightarm.pooled_by_arm_base.{arm.kind}.{BRANCH_PRIMARY}.point", pb["point"])
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([_eightarm_group_label(report, m) for m in models]
                       + [f"{POOLED_GROUP_LABEL}\n({'/'.join(ea['panel_models'])})"], fontsize=6.8)
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("GSM8K accuracy (stock scorer, own denominator; main view)", fontsize=7.2)
    T._recessive(ax)
    handles = [Patch(facecolor=KIND_COLOR[a.kind], edgecolor="none", label=a.label) for a in arms]
    handles += [Patch(facecolor="none", edgecolor=MUTED, hatch=SLOT_HATCH, label=SLOT_LEGEND),
                plt.Line2D([], [], color=INK, lw=0.9, label="95 % bootstrap CI (merged); pooled bars outlined"),
                plt.Line2D([], [], marker="o", mfc="none", mec=INK, ls="", label="8,192-only reading")]
    ax.legend(handles=handles, loc="lower left", ncol=2, frameon=False, fontsize=5.2)
    ax.text(0.995, 0.012, f"pooled = equal-weight mean over {'/'.join(ea['panel_models'])}; held out: {', '.join(ea['held_out'])} "
                          f"(thinking: not available); not in the panel: {', '.join(ea['not_in_panel'])}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.4, color=INK2)
    _title(fig, "Eight arms\n" + _with_selection(report, f"main view; merged bar, 8,192-only marker; {len(kinds_drawn)} of {len(slots)} arms present",
                                                  short=f"main view; merged bar + 8,192 marker; {len(kinds_drawn)}/{len(slots)} arms"))
    return fig, numbers


def _contrast_rows(doc: Mapping[str, Any], prefix: str, model_key: str) -> dict[str, dict[str, Any]]:
    """Per arm cell, the model's main-view contrast summary under both readings with its paths under ``prefix``."""
    out: dict[str, dict[str, Any]] = {}
    for j, cs in enumerate(list(doc.get("contrast_summary") or [])):
        if cs.get("model_key") != model_key or cs.get("view") != VIEW_MAIN or not cs.get("computable"):
            continue
        rec: dict[str, Any] = {}
        for reading in READINGS:
            r = _mapping(cs.get(reading))
            if not r:
                continue
            base = f"{prefix}contrast_summary.{j}.{reading}"
            rec[reading] = {"point": r["point"][AXIS_STOCK][BRANCH_PRIMARY], "ci_low": r["ci"][AXIS_STOCK][BRANCH_PRIMARY][0],
                            "ci_high": r["ci"][AXIS_STOCK][BRANCH_PRIMARY][1], "word": r.get("word"),
                            "paths": {"point": f"{base}.point.{AXIS_STOCK}.{BRANCH_PRIMARY}", "ci_low": f"{base}.ci.{AXIS_STOCK}.{BRANCH_PRIMARY}.0",
                                      "ci_high": f"{base}.ci.{AXIS_STOCK}.{BRANCH_PRIMARY}.1"}}
        out[str(cs.get("cell"))] = rec
    return out


def fig_eightarm_contrasts(report: Mapping[str, Any]) -> tuple[Any, list[dict[str, Any]]]:
    plt, _mcolors, _Patch = T._mpl()
    numbers: list[dict[str, Any]] = []
    ea = report["eightarm"]
    excluded_arms = figure_excluded_arms(report)
    excluded_families = figure_excluded_families(report)
    models = [m for m in EIGHTARM_MODEL_ORDER if m not in excluded_families]
    slots = M.panel_slots(report)                   # one forest row per arm the REPORT registers (a status row where no estimand)
    rows: list[dict[str, Any]] = []
    for m in models:
        pin, peer = _peer_of(report, m)
        doc, prefix = (peer, f"peer_reports.{pin}.") if peer is not None else (report, "")
        in_design = m in (report["design"].get("models") or []) or peer is not None
        by_cell = _contrast_rows(doc, prefix, m) if in_design else {}
        for arm in slots:
            if arm.kind == M.KIND_BARE or arm.cell in excluded_arms:
                continue
            # C3: a peer column's label derives from the peer's metadata; the Llama column without a peer reads pending
            suffix = (f" ({peer_thinking_label(peer, m)})" if peer is not None else (f" ({PEER_PENDING_LABEL})" if m in POOLED_HELD_OUT else ""))
            label = f"{m}{suffix}: {arm.kind}"
            rec = by_cell.get(arm.cell)
            if not in_design:
                rows.append({"label": label, "kind": arm.kind, "rec": None, "status": SLOT_STATUS_PENDING_PEER if m in POOLED_HELD_OUT else ROW_STATUS_ABSENT})
            elif rec is None or READING_MERGED not in rec:
                rows.append({"label": label, "kind": arm.kind, "rec": None, "status": _slot_status(report, m, arm.kind) if peer is None else ROW_STATUS_ABSENT})
            else:
                rows.append({"label": label, "kind": arm.kind, "rec": rec, "status": None, "name": f"{m}/{arm.kind}"})
    for arm in slots:
        if arm.kind == M.KIND_BARE or arm.cell in excluded_arms:
            continue
        label = f"{POOLED_GROUP_LABEL} ({'/'.join(ea['panel_models'])}): {arm.kind}"
        merged = _mapping(_mapping(ea.get("pooled_by_arm_merged")).get(arm.kind))
        base = _mapping(_mapping(ea.get("pooled_by_arm_base")).get(arm.kind))
        if merged.get("status") != POOLED_AVAILABLE:
            rows.append({"label": label, "kind": arm.kind, "rec": None, "status": POOLED_UNAVAILABLE})
            continue
        rec = {READING_MERGED: {**{f: merged[BRANCH_PRIMARY]["minus_bare"][f] for f in ("point", "ci_low", "ci_high")}, "word": None,
                                "paths": {f: f"eightarm.pooled_by_arm_merged.{arm.kind}.{BRANCH_PRIMARY}.minus_bare.{f}" for f in ("point", "ci_low", "ci_high")}}}
        if base.get("status") == POOLED_AVAILABLE:
            rec[READING_R8192] = {**{f: base[BRANCH_PRIMARY]["minus_bare"][f] for f in ("point", "ci_low", "ci_high")}, "word": None,
                                  "paths": {f: f"eightarm.pooled_by_arm_base.{arm.kind}.{BRANCH_PRIMARY}.minus_bare.{f}" for f in ("point", "ci_low", "ci_high")}}
        rows.append({"label": label, "kind": arm.kind, "rec": rec, "status": None, "name": f"pooled/{arm.kind}"})
    fig, ax = plt.subplots(figsize=(8.8, 0.30 * max(len(rows), 1) + 1.4))
    _layout(fig, left=0.32, right=0.85)
    ax.axvline(0.0, color=BASELINE, lw=0.8, zorder=2)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        if r["rec"] is None:
            word = (M.DEFERRED_SLOT_LABEL if r["status"] == STATUS_DEFERRED else (PEER_PENDING_LABEL if r["status"] == SLOT_STATUS_PENDING_PEER
                                                                                  else f"{r['status']}: no estimand"))
            ax.text(0.02, y, word, transform=ax.get_yaxis_transform(), va="center", fontsize=6.2, color=MUTED)
            continue
        pm = r["rec"][READING_MERGED]
        ax.plot([_pp(pm["ci_low"]), _pp(pm["ci_high"])], [y + 0.12, y + 0.12], color=INK, lw=1.1, zorder=5)
        ax.plot([_pp(pm["point"])], [y + 0.12], marker="o", ms=5, color=KIND_COLOR[r["kind"]], mec=INK, mew=0.5, ls="", zorder=6)
        for f in ("point", "ci_low", "ci_high"):
            T._num_at(numbers, report, f"eightarm.contrast.merged.{f}[{r['name']}]", pm["paths"][f], pm[f])
        p8 = r["rec"].get(READING_R8192)
        if p8 is not None:
            ax.plot([_pp(p8["ci_low"]), _pp(p8["ci_high"])], [y - 0.14, y - 0.14], color=INK2, lw=0.9, zorder=5)
            ax.plot([_pp(p8["point"])], [y - 0.14], marker="o", ms=5, mfc="none", mec=INK2, mew=1.0, ls="", zorder=6)
            for f in ("point", "ci_low", "ci_high"):
                T._num_at(numbers, report, f"eightarm.contrast.r8192.{f}[{r['name']}]", p8["paths"][f], p8[f])
        if pm.get("word") is not None:
            ax.text(1.005, y, pm["word"], transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.2,
                    color=INK if pm["word"] in WORDS else MUTED, clip_on=False)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["label"] for r in reversed(rows)], fontsize=6.4)
    ax.set_ylim(-0.7, max(len(rows), 1) - 0.3)
    ax.set_xlabel("accuracy difference vs the main view's none (percentage points)")
    T._recessive(ax)
    ax.xaxis.grid(True, zorder=0)
    ax.yaxis.grid(False)
    handles = [plt.Line2D([], [], marker="o", color=INK2, mec=INK, ls="", label="merged reading (stock, primary), 95 % bootstrap CI"),
               plt.Line2D([], [], marker="o", mfc="none", mec=INK2, ls="", label="8,192-only reading, 95 % bootstrap CI")]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=6.2)
    _title(fig, "Arm contrasts\n" + _with_selection(report, "arm − none per model and pooled (main view; merged headline, 8,192-only beside); llama held out",
                                                     short="arm − none per model and pooled; llama held out"))
    return fig, numbers


EIGHTARM_FIGURES: Mapping[str, Callable[[Mapping[str, Any]], tuple[Any, list[dict[str, Any]]]]] = {
    EIGHTARM_STEM_ACCURACY: fig_eightarm_accuracy, EIGHTARM_STEM_CONTRASTS: fig_eightarm_contrasts}


FIGURE_BUILDERS_FIXED: Mapping[str, Callable[[Mapping[str, Any]], tuple[Any, list[dict[str, Any]]]]] = {
    "final_g8k_accuracy_by_cell": fig_accuracy_by_cell, "final_g8k_atcap_share": fig_atcap_share, "final_g8k_atcap_classes": fig_atcap_classes,
    "final_g8k_truncation_cost": fig_truncation_cost, "final_g8k_overlap": fig_overlap, "final_g8k_ledger": fig_ledger,
    # last and alone on its line: the secret scanner's generic rule reads a quoted string after the word "tokens"
    "final_g8k_completion_tokens": fig_completion_tokens,
}


def figures_for(report: Mapping[str, Any]) -> dict[str, Callable[[Mapping[str, Any]], tuple[Any, list[dict[str, Any]]]]]:
    """The published eleven (a contrasts forest per model shown + the seven fixed), plus the two eight-arm figures when
    the report carries the eight-arm blocks (B2) — so a published-shape run writes exactly the published figure set."""
    out = {**_make_contrast_builders(_shown_models(report)), **FIGURE_BUILDERS_FIXED}
    if "eightarm" in report and "status_by_row" in _mapping(report.get("design")):
        out.update(EIGHTARM_FIGURES)
    return out


def figure_captions(report: Mapping[str, Any]) -> dict[str, str]:
    b = report["bootstrap"]
    models = _shown_models(report)
    caps = {f"final_g8k_contrasts_{m}": (f"{m}: every cell minus its view's none (paired items, B = {b['B']:,}, seed {b['seed']}). Filled = the ladder-merged "
                                          "reading (the headline; stock scorer, primary branch), hollow = the 8,192-only reading; small triangles = the W (residual "
                                          "at-cap wrong) and X (excluded) axes' merged points; the grey band = the noise-floor pair none(it) - none(main); refused "
                                          "and held cells named in place. The word at the right is the merged verdict (filed only where W, X and both branches agree)."
                                          + (f" {NA_LABEL.capitalize()}: {HOLD_OUT_REASON}." if _arm_of_model(report, m) == THINKING_ARM_NOT_AVAILABLE else ""))
            for m in models}
    caps.update({
        "final_g8k_accuracy_by_cell": "Per view, each cell's stock accuracy under the ladder-merged reading (bars) and the 8,192-only reading (hollow markers); hue = cell family.",
        "final_g8k_atcap_share": (f"At-cap share per cell at the 8,192 cap (n_at_cap over draws with a stop reason) against the 1 % report line (dashed) and the {TRIGGER_PCT} "
                                  f"trigger (dash-dot; > {TRIGGER_PCT} re-draws the eligible items at 16,384; {V.GSM8K_TRIGGER_PROVENANCE}); escalated cells and terminal exits annotated."),
        "final_g8k_atcap_classes": ("Each leg's at-cap draws under Evals' degeneracy rule (the rule that decides eligibility): orange = degenerate loop "
                                    "(non-terminating, never redrawn), blue = truncated reasoning (eligible), hatched = text the split's source did not carry. "
                                    "The MATH-500 periodic-block rule rides in the report as a non-deciding column."),
        "final_g8k_truncation_cost": "Per escalated cell, the ladder-merged minus the 8,192-only accuracy (paired, primary branch) with the residual at the ceiling.",
        "final_g8k_overlap": ("Per model, the stock task's OFF greedy 512 leg minus the retired s2parity reading over the 200 paired items with its 95 % CI; "
                              "the band is the ±5 pp tolerance. Verdict continuation iff the CI lies inside it, else new instrument (power-limited at n = 200)."),
        "final_g8k_ledger": "Usage-ledger confirmation per leg: the accepted attempt's forwarded calls, other recorded attempts' rows (attributed), rejections.",
        "final_g8k_completion_tokens": "Completion lengths per leg: the summary mean and the ledger's p50-p99 band against the 8,192 cap and the 16,384 ceiling.",
    })
    na = [m for m in models if _arm_of_model(report, m) == THINKING_ARM_NOT_AVAILABLE]
    if na:
        note = " " + " ".join(f"{m}: {NA_LABEL} — {HOLD_OUT_REASON}." for m in na)
        for stem in ("final_g8k_accuracy_by_cell", "final_g8k_atcap_share", "final_g8k_atcap_classes", "final_g8k_truncation_cost", "final_g8k_ledger",
                     "final_g8k_completion_tokens", "final_g8k_overlap"):
            caps[stem] += note
    # E3: every model whose SHOWN cells include an it-doc leg carries the pair's qualification (verbatim off the row) on its
    # contrasts caption; the eight-arm captions carry every such model's (the it view's arms are not the panel's, but the reader of
    # those figures meets the same models)
    itdoc_notes = {m: note for m in models if (note := _itdoc_caption_note(report, m))}
    for m, note in itdoc_notes.items():
        caps[f"final_g8k_contrasts_{m}"] += " " + note
    if "eightarm" in report:
        ea = report["eightarm"]
        # the slot count is the PANEL's own (the arms this report registers), never len(ARMS): a report without a
        # later-registered arm keeps its published caption word for word
        caps[EIGHTARM_STEM_ACCURACY] = (f"{M.panel_numeral(len(M.panel_slots(report))).capitalize()} fixed arm slots per model on the main view "
                                        f"(gemma / qwen / nemotron / llama): the ladder-merged stock "
                                        f"accuracy as the headline bar with its 95 % bootstrap CI and the 8,192-only reading as a hollow marker; an empty "
                                        f"hatched slot is a deferred / held / absent arm (lettered), never dropped; the llama group is the Llama lane's "
                                        f"production report (thinking: not available; held out of the pooled group) or reads '{PEER_PENDING_LABEL}'; the "
                                        f"pooled group is the equal-weight mean over {'/'.join(ea['panel_models'])} from report.eightarm.")
        caps[EIGHTARM_STEM_CONTRASTS] = ("Per model (main view) and pooled, each arm minus none: the merged reading (filled, headline) and the 8,192-only "
                                         "reading (hollow) with 95 % bootstrap CIs; the word at the right is the merged verdict as filed; a deferred / held / "
                                         "absent arm is named in place; the pooled rows are report.eightarm's shared-draw estimands, llama held out.")
        for note in itdoc_notes.values():
            caps[EIGHTARM_STEM_ACCURACY] += " " + note
            caps[EIGHTARM_STEM_CONTRASTS] += " " + note
    return caps


def _arm_short_label(cell: str) -> str:
    """The short arm name a caption uses: the eight-arm registry's label head (``LoRA ctl2``, ``LoRA mix``, ``object``), else
    the cell's short spelling."""
    arm = ARM_BY_CELL.get(cell)
    return arm.label.split(" (")[0] if arm is not None else _short(cell)


def _itdoc_caption_note(report: Mapping[str, Any], model_key: str) -> str | None:
    """The it-doc qualification sentence(s) for one model's captions (E3): over the model's SHOWN cells that are it-doc LEGS (a
    held row is a design status, not a reading), grouped by their qualification text — ``it-doc re-fit (<model> <view>: <arm>,
    <arm>): <qualification>`` per distinct text, joined by a space; ``None`` when no shown cell of the model is an it-doc leg or
    none carries a qualification. The text is the ROW's, verbatim, never retyped here."""
    by_text: dict[str, list[tuple[str, str]]] = {}
    for c in report["cells"].values():
        if c.get("model_key") != model_key or not V.is_itdoc_row(c) or not _row_shown(report, c):
            continue
        text = V.qualification_text(c)
        if text:
            by_text.setdefault(text, []).append((str(c.get("view")), str(c["cell"])))
    if not by_text:
        return None
    notes = []
    for text, pairs in by_text.items():
        views = sorted({v for v, _ in pairs})
        arms = ", ".join(_arm_short_label(cell) for _, cell in pairs)
        notes.append(f"it-doc re-fit ({model_key} {'/'.join(views)}: {arms}): {text}")
    return " ".join(notes)


def validated_exclusions(report: Mapping[str, Any], families: Sequence[str], *, peer_models: Sequence[str] = ()) -> list[str]:
    """``--figure-exclude-family`` checked against the design: every name a model
    key of the report — or, on a ``--peer-report`` assembly, of a peer's design
    (``peer_models``; the four-family figures drop the family from every column)
    — (a typo would otherwise exclude nothing, silently) and at least one model
    left to draw; sorted, deduplicated."""
    known = list(report["design"]["models"])
    known_all = known + [m for m in peer_models if m not in known]
    excluded = sorted(set(families))
    unknown = [f for f in excluded if f not in known_all]
    if unknown:
        raise AnalysisError(f"--figure-exclude-family {unknown}: not a model key of this report (the design's models are {known}"
                            + (f"; the peers' {[m for m in known_all if m not in known]}" if len(known_all) > len(known) else "") + ")")
    if not [m for m in known if m not in excluded]:
        raise AnalysisError(f"--figure-exclude-family {excluded} leaves no model to draw (the design's models are {known})")
    return excluded


def report_cells(report: Mapping[str, Any]) -> list[str]:
    """Every CELL id the report's matrix names (the collected cells, the held rows, the design's status rows), in order."""
    out: list[str] = []
    for c in report["cells"].values():
        out.append(str(c["cell"]))
    for h in report.get("held") or []:
        out.append(str(h["cell"]))
    for rec in _mapping(_mapping(report.get("design")).get("status_by_row")).values():
        out.append(str(rec["cell"]))
    return list(dict.fromkeys(out))


def validated_selection(report: Mapping[str, Any], families: Sequence[str] = (), arms: Sequence[str] = (),
                        modes: Sequence[str] | None = None) -> dict[str, Any]:
    """The three PRESENTATION selectors validated against THIS report (plan P5): the family exclusion
    (:func:`validated_exclusions`), ``--figure-exclude-arm`` (every name a CELL of the report's matrix — an arm absent
    from the report is an ERROR; ``none`` can never be excluded, every contrast is cell − none; at least one other cell
    must stay) and ``--figure-modes`` (every name a sampling mode of the matrix — GSM8K has the one ``thinking_card``
    mode; at least one). Returns ``{excluded_families, excluded_arms, modes}`` (``modes`` None = every mode)."""
    excluded = validated_exclusions(report, families) if families else []
    cells = report_cells(report)
    given = sorted({str(a) for a in arms})
    unknown = [a for a in given if a not in cells]
    if unknown:
        raise AnalysisError(f"--figure-exclude-arm {unknown}: not a cell of this report's matrix (the cells are {cells})")
    if CELL_NONE in given:
        raise AnalysisError(f"--figure-exclude-arm {CELL_NONE!r}: the bare cell is every contrast's reference and cannot be left out")
    # every PSR cell of the report is out of EVERY figure, selector or no selector — recorded under `excluded_arms` like
    # a given one, so the sidecar states it and the gate reads it
    arms_x = sorted(set(given) | set(M.psr_cells_of(cells)))
    if arms_x and len(arms_x) >= len(cells) - 1:
        raise AnalysisError(f"the figure exclusion {arms_x} leaves no cell to contrast with none (the cells are {cells}; PSR cells "
                            f"{M.psr_cells_of(cells)} are out of every figure by ruling) — nothing to draw")
    modes_x: list[str] | None = None
    if modes is not None:
        present = list(_mapping(report.get("design")).get("modes") or [MODE_PRIMARY])
        modes_x = list(dict.fromkeys(str(m) for m in modes))
        unknown = [m for m in modes_x if m not in present]
        if unknown:
            raise AnalysisError(f"--figure-modes {unknown}: not a sampling mode of this report's matrix (the modes are {present})")
        if not modes_x:
            raise AnalysisError("--figure-modes names no mode")
    return {FIGURES_EXCLUDED_KEY: excluded, FIGURES_EXCLUDED_ARMS_KEY: arms_x, FIGURES_MODES_KEY: modes_x}


def _truncation_shown(tr: Mapping[str, Any], excluded: frozenset[str], shown_models: Sequence[str]) -> dict[str, Any]:
    """The report's truncation block for the tables JSON under an exclusion: the
    per-leg ``cells`` and ``costs`` drop the excluded families; every AGGREGATE
    keeps the WAVE's value under its own key (unchanged by the exclusion) and is
    duplicated as ``*_report`` with an explicit ``*_shown`` counterpart over the
    families shown — a reader never mistakes one for the other."""
    cells = [c for c in tr["cells"] if c["model_key"] not in excluded]
    hidden = {c["leg"] for c in tr["cells"] if c["model_key"] in excluded}
    out = dict(tr)
    out.update({"cells": cells, "costs": [t for t in tr["costs"] if t["model_key"] not in excluded], FAMILIES_SHOWN_KEY: list(shown_models),
                "aggregates_note": AGGREGATES_NOTE, "n_cells_read_report": tr["n_cells_read"], "n_cells_read_shown": len(cells),
                "max_share_report": tr["max_share"], "max_share_shown": max((c["share"] for c in cells if c["share"] is not None), default=None)})
    for k in ("over_1pct", "triggered", "escalated", "terminal_no_rung"):
        out[f"{k}_report"], out[f"{k}_shown"] = list(tr[k]), [leg for leg in tr[k] if leg not in hidden]
    return out


def refuse_mixed_render(figs_dir: Path, excluded: Sequence[str], stems: Sequence[str], *, arms: Sequence[str] = (),
                        modes: Sequence[str] | None = None) -> None:
    """:func:`M.refuse_mixed_render` over THIS instrument's rows sidecar and figure-stem prefix: a figures directory (or
    mirror) holds ONE selection; another selection's sidecar or a stale figure file refuses by name — an isolated directory
    is the remedy (review of 2026-09-11: a directory re-used across a no-flag render and an exclusion render kept the
    excluded family's forest on disk, and a whole-directory upload would have shipped it)."""
    M.refuse_mixed_render(figs_dir, excluded, stems, arms=arms, modes=modes, numbers_filename=NUMBERS_FILENAME, stem_prefix=STEM_PREFIX)


def make_figures(report: Mapping[str, Any], figs_dir: Path, *, exclude_families: Sequence[str] = (), exclude_arms: Sequence[str] = (),
                 modes: Sequence[str] | None = None) -> dict[str, Any]:
    """Every figure, PNG + PDF, the numbers sidecar and the tables JSON.
    ``peer_reports`` (``--peer-report``, :func:`load_peer_reports`) adds the
    ``*_four_family`` stems (:data:`FOUR_FAMILY_STEMS`) drawn from
    :func:`combine_reports`'s view, written beside the figures
    (:data:`COMBINED_FILENAME`) so their sidecar records resolve against a file;
    the exclusion list then applies to every column of the assembly.
    ``exclude_families`` (``--figure-exclude-family``, repeatable) leaves the
    named model families out of every figure and of the per-model / per-leg rows
    of the tables JSON: the figures are drawn from a shallow VIEW of the report
    whose ``figures.excluded_families`` names them, so the report itself is
    untouched and every sidecar path still resolves in it; the gates, the
    verdict and the provenance are the wave's; the exclusion is recorded in the
    returned manifest (-> ``report["figures"]``), the numbers sidecar and the
    tables — all of it ONLY when a family is excluded, so with the flag absent
    every output (figures, sidecars, manifest) is byte-identical to the pre-flag
    shape. ``exclude_arms`` (``--figure-exclude-arm``) and ``modes``
    (``--figure-modes``) are the two further PRESENTATION selectors: validated
    against the report (:func:`validated_selection`), they pick the rows SHOWN,
    recompute no aggregate (pooled means, verdict words and the tables JSON are
    the full report's) and put both counts in every title; recorded under
    ``figures.selection`` (the numbers sidecar and the returned manifest) ONLY
    when one is in play; the tables JSON keeps its family-exclusion behaviour
    and is untouched by them. The directory must hold one selection
    (:func:`refuse_mixed_render`)."""
    figs_dir = Path(figs_dir).expanduser().resolve()
    sel = validated_selection(report, exclude_families, exclude_arms, modes)
    excluded, arms, modes_x = sel[FIGURES_EXCLUDED_KEY], sel[FIGURES_EXCLUDED_ARMS_KEY], sel[FIGURES_MODES_KEY]
    hidden = frozenset(excluded)
    figures_block: dict[str, Any] = {**_mapping(report.get("figures")), FIGURES_EXCLUDED_KEY: list(excluded)}
    if arms:
        figures_block[FIGURES_EXCLUDED_ARMS_KEY] = list(arms)
    if modes_x is not None:
        figures_block[FIGURES_MODES_KEY] = list(modes_x)
    view: dict[str, Any] = {**report, "figures": figures_block}
    builders = figures_for(view)
    refuse_mixed_render(figs_dir, excluded, list(builders), arms=arms, modes=modes_x)
    figs_dir.mkdir(parents=True, exist_ok=True)
    # recorded ONLY under an exclusion: with the flag absent every JSON output is byte-identical to the pre-flag shape
    family_extra: dict[str, Any] = {FIGURES_EXCLUDED_KEY: list(excluded), FAMILIES_SHOWN_KEY: _shown_models(view)} if excluded else {}
    extra: dict[str, Any] = dict(family_extra)
    if arms:
        extra[FIGURES_EXCLUDED_ARMS_KEY] = list(arms)
    if modes_x is not None:
        extra[FIGURES_MODES_KEY] = list(modes_x)
    if arms or modes_x is not None:
        # recorded ONLY under an arm / mode selection: the counts a title carries, never a recomputed aggregate (B11)
        rows_all = _leg_rows(report)
        extra[FIGURES_SELECTION_KEY] = {FIGURES_EXCLUDED_KEY: list(excluded), FIGURES_EXCLUDED_ARMS_KEY: list(arms), FIGURES_MODES_KEY: modes_x,
                                        "n_legs_report": len(rows_all), "n_legs_shown": len(_shown_rows(view, rows_all)),
                                        "rule": ("presentation only: the report, the tables JSON and every aggregate are the full report's; the "
                                                 "selectors pick the rows SHOWN and every title carries both counts")}
    written: dict[str, list[str]] = {}
    numbers: dict[str, list[dict[str, Any]]] = {}
    for stem, fn in builders.items():
        fig, nums = fn(view)
        frac = axes_height_fraction(fig)
        if frac < AXES_MIN_FRACTION:
            raise AnalysisError(f"{stem}: the axes span {frac:.3f} of the figure height (< {AXES_MIN_FRACTION})")
        written[stem] = [str(p) for p in T._save(fig, figs_dir, stem)]
        numbers[stem] = nums
    captions = figure_captions(view)
    (figs_dir / NUMBERS_FILENAME).write_text(json.dumps({"figures": numbers, "captions": captions, "sha": report["sha"],
                                                         "analysis_code_sha": report.get("analysis_code_sha"), **extra}, indent=1, default=str), encoding="utf-8")

    def rows(block: Sequence[Mapping[str, Any]]) -> Sequence[Mapping[str, Any]]:
        return [r for r in block if r["model_key"] not in hidden] if hidden else block

    prov = report["provenance"]
    if hidden and isinstance(prov.get("cells"), Mapping):
        hidden_legs = {row["leg"] for row in report["controls"] if row["model_key"] in hidden}
        prov = {**prov, "cells": {k: v for k, v in prov["cells"].items() if k not in hidden_legs and _mapping(v).get("model_key") not in hidden}}
    tables = {"contrasts": rows(report["contrast_summary"]), "noise_floor": rows(report["noise_floor_summary"]),
              "truncation": _truncation_shown(report["truncation"], hidden, _shown_models(view)) if hidden else report["truncation"],
              "controls": rows(report["controls"]), "gates": report["gates"], "verdict": report["verdict"], "held": rows(report["held"]), "refused": rows(report["refused"]),
              "overlap": {k: {f: v for f, v in b.items() if f != "retired_id_match"} for k, b in _mapping((report.get("overlap") or {}).get("models")).items()
                          if k not in hidden},
              "provenance": prov, "captions": captions, "sha": report["sha"], "analysis_code_sha": report.get("analysis_code_sha"), **family_extra}
    (figs_dir / TABLES_FILENAME).write_text(json.dumps(tables, indent=1, default=str), encoding="utf-8")
    return {"written": written, "skipped": {}, "numbers": str(figs_dir / NUMBERS_FILENAME), "tables": str(figs_dir / TABLES_FILENAME), **extra}


PUBLICATION_GATE_KEY = M.PUBLICATION_GATE_KEY


def publication_gate_record(report: Mapping[str, Any], figs: Mapping[str, Any], figs_dir: Path) -> dict[str, Any]:
    """:func:`M.publication_gate_record` over THIS instrument's rows sidecar and the sampling modes its matrix carries
    (``design.modes`` — the one thinking-card mode): plan P5's gate before any mirror or W&B call (exit 6). It runs on
    EVERY render (code review C1 of 2026-09-12: a published-matrix render with a bare ``--gen-root`` and no selector used
    to skip it and proceed to the mirror and W&B with sixteen PSR control rows in the figure builders' view; the PSR cells
    are now out of every figure by :func:`validated_selection`, and the published numeric reproduction runs ``--no-figs``)."""
    return M.publication_gate_record(report, figs, figs_dir, sidecar=NUMBERS_FILENAME,
                                     present_modes=list(_mapping(report.get("design")).get("modes") or [MODE_PRIMARY]))


# --------------------------------------------------------------------------- #
# W&B, the production boundary, the CLI                                        #
# --------------------------------------------------------------------------- #


def log_wandb(report: Mapping[str, Any], figs: Mapping[str, Any], *, mode: str, name: str | None = None, name_suffix: str = "") -> str | None:
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as e:    # pragma: no cover - venv-dependent
        raise AnalysisError("wandb is not importable — pass --wandb-mode disabled or use a venv that carries it") from e
    sha = report["sha"]
    name = name or f"{WANDB_RUN_PREFIX}{sha}{name_suffix}"
    from sisteer.wandb_entity import wandb_project
    run = wandb.init(project=wandb_project(WANDB_PURPOSE), name=name, mode=mode,
                     config={"experiment": EXPERIMENT, "sha": sha, "analysis_code_sha": report.get("analysis_code_sha"), "task": TASK, **report["bootstrap"],
                             "matrix_sha256": report["matrix"]["sha256"], f"figures_{FIGURES_EXCLUDED_KEY}": list(figs.get(FIGURES_EXCLUDED_KEY) or [])})
    for stem, paths in figs.get("written", {}).items():
        png = next((p for p in paths if p.endswith(".png")), None)
        if png:
            wandb.log({f"g8k/{stem}": wandb.Image(png)})
    summary: dict[str, Any] = {"verdict": report["verdict"]["verdict"], "gates_failed": len(report["gates"]["failed"])}
    for c in report["contrast_summary"]:
        if not c["computable"]:
            continue
        for reading in READINGS:
            for axis in AXES:
                for b in BRANCHES:
                    summary[f"{c['model_key']}/{c['view']}/{c['cell']}/{reading}/{axis}/{b}/point"] = c[reading]["point"][axis][b]
                    lo, hi = c[reading]["ci"][axis][b]
                    summary[f"{c['model_key']}/{c['view']}/{c['cell']}/{reading}/{axis}/{b}/ci_low"] = lo
                    summary[f"{c['model_key']}/{c['view']}/{c['cell']}/{reading}/{axis}/{b}/ci_high"] = hi
    for row in report["controls"]:
        for f, v in row["summary"].items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                summary[f"controls/{row['leg']}/{f}"] = v
        for f, v in (row.get("at_cap") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                summary[f"controls/{row['leg']}/at_cap_{f}"] = v
    for key, per in _mapping(report["gates"]["A10_overlap"].get("models")).items():
        if isinstance(per.get("delta"), (int, float)):
            summary[f"overlap/{key}/delta"] = per["delta"]
    run.summary.update(summary)
    # the artifact is EXACTLY the manifest's files (the figures written by this render and its two sidecars), never the directory:
    # a re-used directory can hold a previous render's files, and those must not travel under this run's config
    files = [p for paths in figs.get("written", {}).values() for p in paths] + [str(figs[k]) for k in ("numbers", "tables", "combined") if figs.get(k)]
    if files:
        art = wandb.Artifact(f"figures-{name}", type="figures")
        for p in files:
            art.add_file(str(p), name=Path(p).name)
        run.log_artifact(art)
    url = getattr(run, "url", None)
    run.finish()
    return url


def production_refusals(report: Mapping[str, Any]) -> list[str]:
    """Why ``report`` is NOT a production result — every reason by name, or ``[]``:
    a FAILED gate; a PENDING gate not waivable; an unusable leg (an expected,
    named refusal is not one); a computable contrast that did not run; the
    estimand absent; the verdict withheld."""
    gates = _mapping(report.get("gates"))
    out: list[str] = []
    problems = _mapping(report.get(TREE_SET_KEY)).get("problems")
    if problems:
        out.append(f"{TREE_SET_KEY}: " + "; ".join(str(x) for x in problems))          # A report written at the startup boundary
    failed = gates_failed(gates)                       # from the RECORDS: a saved A9 FAILED is read even if the aggregate missed it
    if failed:
        out.append(f"gate(s) FAILED: {', '.join(failed)}")
    if sorted(gates.get("failed") or []) != failed:
        out.append(f"the report's gates.failed aggregate {sorted(gates.get('failed') or [])} != the gate records' {failed} — a torn report")
    unwaivable = [it for it in pending_items(gates) if not it["waivable"]]
    if unwaivable:
        out.append("gate(s) PENDING and not waivable: " + "; ".join(f"{it['gate']}: {it['detail']}" for it in unwaivable))
    if report.get("unusable_legs"):
        out.append(f"unusable leg(s): {sorted(report['unusable_legs'])}")
    not_run = [c["name"] for c in _not_run(report.get("contrasts", []))]
    if not_run:
        out.append(f"contrast(s) did not run: {not_run}")
    declaration_problems = contrast_declaration_problems(report.get("contrasts", []))
    if declaration_problems:
        out.append("expected contrast refusal(s) not met: " + "; ".join(declaration_problems))
    if "estimands" not in report:
        out.append("the estimand did not run (no `estimands` block)")
    verdict = _mapping(report.get("verdict"))
    if not out and verdict.get("word") is None:
        out.append(f"verdict withheld: {verdict.get('reason')}")
    return out


# ---------------------------------------------------------------------------- #
# the paper rows: the flat per-row export beside the report                     #
# ---------------------------------------------------------------------------- #
#
# Every per-cell `point` above is the CONDITIONAL estimand (n_correct / n_scored: the items whose
# draw terminated). The paper's capability panel draws SCORE-AS-IS (n_correct / n_paired: every
# unscored draw wrong). `paper_rows_of_report` reads BOTH from the report already written — the
# readings' counts, the estimands' bootstrap CIs (primary AND score_as_is branches), the legs'
# summaries, the redraw records — and files them flat, one row per (cell, reading), with the
# four count fields beside them. It computes no estimator: a rung leg the report bootstraps
# nowhere gets the exact binomial CI, labelled. The report itself is never rewritten.

#: the ladder exits and the rung state each names (`escalate` splits on whether the rung store was read)
PAPER_RUNG_STATE_BY_EXIT: dict[str, str] = {
    CL.EXIT_CLEAN: PRW.RUNG_NOT_APPLICABLE, CL.EXIT_BELOW_THRESHOLD: PRW.RUNG_NOT_APPLICABLE,
    IR.EXIT_TRIGGERED_NO_ELIGIBLE: PRW.RUNG_NOT_APPLICABLE, CL.EXIT_CEILING_AT_OUTSET: PRW.RUNG_NOT_APPLICABLE,
    CL.EXIT_CEILING: PRW.RUNG_COMPLETE, CL.EXIT_TRANSPORT_REFUSED: PRW.RUNG_REFUSED,
    CL.EXIT_NO_RUNG: PRW.RUNG_NEVER_RUN, CL.EXIT_LEDGER_CENSORED: PRW.RUNG_NEVER_RUN,
}


def _pr_cap_of_leg_key(leg_key: str) -> int:
    return int(str(leg_key).rsplit("@", 1)[1])


def _pr_reading_counts(items: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """A reading's per-item flags, summed — the check that the reading's headline counts are its own items'."""
    caps: set[int] = set()
    for it in items.values():
        src = str(it.get("source") or "")
        if src.startswith("r") and src[1:].isdigit():
            caps.add(int(src[1:]))
    return {"n": len(items),
            "n_scored": sum(1 for it in items.values() if it.get("scored")),
            "n_correct": sum(1 for it in items.values() if it.get("scored") and float(it.get("value") or 0.0) == 1.0),
            "n_at_cap": sum(1 for it in items.values() if it.get("at_cap")),
            "n_think_unscored": sum(1 for it in items.values() if it.get("think_block_unterminated") and not it.get("scored")),
            "n_unterminated_not_at_cap": sum(1 for it in items.values() if it.get("think_block_unterminated") and not it.get("at_cap")),
            "n_refusal_shaped": sum(1 for it in items.values() if it.get("scored") and it.get(ITEM_REFUSAL_SHAPED_KEY)),
            "n_refusal_shaped_of": sum(1 for it in items.values() if it.get("scored") and it.get(ITEM_REFUSAL_EXAMINED_KEY)),
            "caps": sorted(caps)}


def _pr_rung_refusal_shaped(rung: Mapping[str, Any], rung_key: str) -> tuple[int | None, int | None, str]:
    """``(count, denominator, note)``: the rung leg's refusal-shaped count and the draws the classifier examined for it, READ from its
    filed readout (the leg's rows.jsonl content channel over its scored redraws) and held to the readout's own contract; a leg written
    before the class existed carries no block -> both None (absence, never zero), said in the note. A present, off-contract block refuses."""
    block = rung.get("refusal_shaped")
    if block is None:
        return None, None, "n_refusal_shaped / n_refusal_shaped_of: this rung leg carries no refusal-shaped readout (written before the class existed) -> null together"
    problems = RFS.readout_problems(block, n_scored=int((rung.get("summary") or {}).get("n_scored") or 0))
    if problems:
        raise AnalysisError(f"paper rows: legs[{rung_key}].refusal_shaped: " + "; ".join(problems))
    return (int(block["n_refusal_shaped"]), int(block["n_examined"]),
            f"legs[{rung_key}].refusal_shaped: n_refusal_shaped / n_examined (-> n_refusal_shaped_of) ({RFS.RULE})")


def _pr_ci_index(report: Mapping[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """``(cell_key, reading) -> {"primary": acc entry, "score_as_is": acc entry, "contrast", "role"}`` from the report's
    estimands, joined to the cells through the contrast RECORDS (`cell_key` / `none_key`), never by parsing a name. The
    `none` leg of a view sits in every contrast of that view; its entries must agree everywhere (one seeded draw)."""
    by_name = {c["name"]: c for c in report.get("contrasts", []) if isinstance(c, Mapping) and "name" in c}
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for name, per_reading in (report.get("estimands") or {}).items():
        contrast = by_name.get(name)
        if contrast is None:
            raise AnalysisError(f"paper rows: estimands[{name!r}] has no contrast record")
        for reading, block in per_reading.items():
            stock = (block or {}).get("stock") or {}
            if "primary" not in stock or "score_as_is" not in stock:
                raise AnalysisError(f"paper rows: estimands[{name!r}].{reading}.stock lacks a primary / score_as_is branch")
            for role, ck in (("cell", contrast["cell_key"]), ("none", contrast["none_key"])):
                entry = {br: stock[br]["acc_by_leg"][role] for br in ("primary", "score_as_is")}
                prev = index.get((ck, reading))
                if prev is not None:
                    if any(prev[br] != entry[br] for br in ("primary", "score_as_is")):
                        raise AnalysisError(f"paper rows: the {role} leg {ck}@{reading} reads differently in {prev['contrast']!r} and {name!r}")
                    continue
                index[(ck, reading)] = {**entry, "contrast": name, "role": role}
    return index


def _pr_stamp(cell: Mapping[str, Any], leg: Mapping[str, Any], key: str) -> Any:
    """A stamp READ from the record: the cell, the leg, its identity or its summary — the first that carries ``key``."""
    for holder in (cell, leg, leg.get("identity"), leg.get("summary")):
        if isinstance(holder, Mapping) and holder.get(key) is not None:
            return holder[key]
    return None


def _pr_object_generation(cell: Mapping[str, Any], leg: Mapping[str, Any]) -> tuple[str, str | None]:
    if not cell.get("fitted"):
        return PRW.GEN_NONE, None
    stamp = _pr_stamp(cell, leg, "object_generation")
    if stamp is None:
        return PRW.GEN_NA, "object_generation: this record carries no generation stamp (no fit provenance binds the object, or the record predates the stamp) -> n/a"
    if stamp not in PRW.OBJECT_GENERATION_STAMPS:
        raise AnalysisError(f"paper rows: object_generation stamp {stamp!r} not in {PRW.OBJECT_GENERATION_STAMPS}")
    source = cell.get("object_generation_source") if isinstance(cell, Mapping) else None
    return str(stamp), (f"object_generation: read from the cell record's fit-provenance binding ({source})" if source else None)


def _pr_selection_outcome(cell: Mapping[str, Any], leg: Mapping[str, Any]) -> tuple[str | None, str | None]:
    stamp = _pr_stamp(cell, leg, "selection_outcome")
    if stamp is None:
        return None, None
    if stamp not in PRW.SELECTION_OUTCOMES:
        raise AnalysisError(f"paper rows: selection_outcome {stamp!r} not in {PRW.SELECTION_OUTCOMES}")
    return str(stamp), None


def _pr_selection(cell: Mapping[str, Any], leg: Mapping[str, Any], *, selection_outcome: str | None) -> tuple[Mapping[str, Any] | None, str | None]:
    """The cell's DECLARED ``selection`` block (schema /3), read like ``selection_outcome`` — never composed here. A present block
    that is not a reading refuses by name (:func:`PRW.selection_problems`: no ``ladder_cap``, an outcome that disagrees with the
    row's word, the override word beside a within-budget reading or a ``rule_returns`` naming the selected dose, an alpha-shaped
    dose on the raw-coefficient arm); absent -> ``None`` (the verifier requires it under the override word)."""
    block = _pr_stamp(cell, leg, "selection")
    if block is None:
        return None, None
    problems = PRW.selection_problems(block, tag=f"cells[{cell.get('cell_key')}]", selection_outcome=selection_outcome, cell=cell.get("cell"))
    if problems:
        raise AnalysisError("paper rows: " + "; ".join(problems))
    return dict(block), "selection: the admission record's block, carried verbatim off the matrix row (both readings, the budget, the ladder cap)"


def _pr_fit_provenance(cell: Mapping[str, Any], leg: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str | None]:
    """The cell's DECLARED ``fit_provenance`` block (schema /3), verbatim off the record; a present block missing any
    :data:`PRW.FIT_PROVENANCE_FIELDS` key refuses by name; absent -> ``None`` (required on the all-positions Arditi arm by the
    verifier — the export never fills it)."""
    block = _pr_stamp(cell, leg, "fit_provenance")
    if block is None:
        return None, None
    problems = PRW.fit_provenance_problems(block, tag=f"cells[{cell.get('cell_key')}]")
    if problems:
        raise AnalysisError("paper rows: " + "; ".join(problems))
    return dict(block), "fit_provenance: the admission record's block, carried verbatim off the matrix row"


def _pr_serving_commit(cell: Mapping[str, Any], leg: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """The SERVING STACK's commit (schema /3), READ from its authoritative source: ``standup_jobs.jsonl[0].sha`` of the leg's
    own standup job — the tree the STANDUP ran from, i.e. the checkout that built and served the adapter. It is NOT
    ``producer.tree_sha``, which is the DRIVER's tree; for these launchers the standup and the driver run from one eval-repo
    checkout, so the two MUST agree, and the standup-record resolution already refuses by name when they do not (two stacks
    under one row is what this field exists to catch). A producer that stamps ``serving_commit`` explicitly wins over the
    derivation and is validated the same way; with neither present this reads ``None`` and the export refuses a served row of
    an arm that requires it, rather than filing the driver's tree under a serving label."""
    stamp = _pr_stamp(cell, leg, "serving_commit")
    if stamp is None:
        rec = _mapping(leg.get("standup_record"))
        sha, tree = rec.get("standup_sha"), rec.get("producer_tree_sha")
        if sha and tree and str(sha) != str(tree):
            raise AnalysisError(f"paper rows: cells[{cell.get('cell_key')}]: the standup ran from {str(sha)[:12]}… but the driver's "
                                f"producer tree is {str(tree)[:12]}… — two stacks under one row; serving_commit is not derivable "
                                "from a record whose standup and driver disagree")
        if sha:
            return str(sha), ("serving_commit: standup_jobs.jsonl[0].sha (the tree the standup served from), cross-checked equal "
                              "to the manifest's producer.tree_sha")
        return None, None
    problems = PRW.serving_commit_problems(stamp, tag=f"cells[{cell.get('cell_key')}]", cell=cell.get("cell"), has_reading=True)
    if problems:
        raise AnalysisError("paper rows: " + "; ".join(problems))
    return str(stamp), "serving_commit: the serving stack's commit as the store record stamps it"


def _pr_rung_state(cell: Mapping[str, Any], legs: Mapping[str, Any]) -> tuple[str, str]:
    """The cell's rung state and the reason, from the base leg's outcome and the cap-ladder record."""
    base = legs.get(cell.get("base_leg") or "") or {}
    if base.get("refused") or base.get("certified_unservable"):
        return PRW.RUNG_REFUSED, f"rung_state: the cell is a named refusal ({str(base.get('refused'))[:60]})"
    if base.get("missing"):
        return PRW.RUNG_NEVER_RUN, f"rung_state: the cell's store is missing ({str(base.get('missing'))[:60]})"
    ladder = cell.get("ladder") or {}
    if not ladder.get("record_present"):
        return PRW.RUNG_NEVER_RUN, f"rung_state: no cap_ladder record ({ladder.get('record_problem')})"
    exit_ = ladder.get("exit")
    if exit_ == CL.ESCALATE:
        rung_key = cell.get("rung_leg")
        rung = legs.get(rung_key or "") or {}
        if rung.get("refused"):
            return PRW.RUNG_REFUSED, f"rung_state: ladder exit {exit_}; the rung leg {rung_key} is a named refusal"
        if ladder.get("rung_store_present") and rung_key and rung.get("summary") and not rung.get("missing"):
            return PRW.RUNG_COMPLETE, f"rung_state: ladder exit {exit_}; rung store read ({rung_key})"
        return PRW.RUNG_OWED, f"rung_state: ladder exit {exit_}; no rung store yet (n_eligible {ladder.get('n_eligible')})"
    if exit_ in PAPER_RUNG_STATE_BY_EXIT:
        return PAPER_RUNG_STATE_BY_EXIT[exit_], f"rung_state: ladder exit {exit_}"
    raise AnalysisError(f"paper rows: cap_ladder exit {exit_!r} of {cell.get('cell_key')} names no rung state")


def _pr_null_row(*, model_key: str, view: str, cell: str, arm: str, cap: int, leg: str, rung_state: str, gen: str,
                 sel: str | None, source: str, prov3: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return PRW.make_row(instrument=PRW.INSTRUMENT_GSM8K, model_key=model_key, view=view, cell=cell, arm=arm, cap=cap, leg=leg,
                        **{f: None for f in PRW.COUNT_FIELDS}, rung_state=rung_state, object_generation=gen, selection_outcome=sel,
                        source=source, **dict(prov3 or {}))


def paper_rows_of_report(report: Mapping[str, Any], *, report_path: Path | str | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The paper rows of a written report: one per (cell, reading) — the base-cap reading, the merged reading, and the
    rung leg over its redrawn subset where the cell escalated — plus a null-count row for a cell with no reading (a named
    refusal). Returns ``(rows, not_exported)``; every number is READ (the reading's counts are re-summed from its own
    per-item flags and must agree; the CIs are the estimands' bootstrap where the report has one, else exact binomial)."""
    legs = report.get("legs") or {}
    cells = report.get("cells") or {}
    pin, acs = report.get("sha"), report.get("analysis_code_sha")
    ci_index = _pr_ci_index(report)
    # the contrast each cell sits in (by the record's cell_key, never by parsing a name): a contrast refused BY DESIGN (the declared
    # expected refusal matched exactly) leaves the cell no bootstrap — its rows carry the ARM's own numbers and name the refusal
    by_design = {c["cell_key"]: c for c in report.get("contrasts", [])
                 if isinstance(c, Mapping) and c.get("kind") == "contrast" and c.get(CONTRAST_REFUSED_BY_DESIGN_KEY)}
    rows: list[dict[str, Any]] = []

    def src(reads: Sequence[str]) -> str:
        return PRW.source_text(report_path, store_pin=pin, analysis_code_sha=acs, reads=reads)

    for ck, cell in cells.items():
        mk, view, cid = cell["model_key"], cell["view"], cell["cell"]
        arm = PRW.PAPER_ARM_BY_CELL.get(cid)
        notes: list[str] = []
        if arm is None:
            arm = cid
            notes.append("arm: the cell id (no declared paper arm for this cell)")
        if ck in by_design:
            c = by_design[ck]
            notes.append(f"contrasts[{c['name']}]: refused by design ({c[ROW_EXPECTED_CONTRAST_REFUSAL_KEY]}) — the arm's numbers are reported "
                         f"as the arm's numbers, the pairing's absence accepted; no estimand read — {str(c.get('reason'))[:200]}")
        base_key = cell["base_leg"]
        base = legs.get(base_key) or {}
        base_cap = int(base.get("cap") or _pr_cap_of_leg_key(base_key))
        rung_state, rung_note = _pr_rung_state(cell, legs)
        gen, gen_note = _pr_object_generation(cell, base)
        sel, sel_note = _pr_selection_outcome(cell, base)
        selection, selection_note = _pr_selection(cell, base, selection_outcome=sel)
        fit_prov, fit_prov_note = _pr_fit_provenance(cell, base)
        serving_commit, serving_commit_note = _pr_serving_commit(cell, base)
        notes += [x for x in (rung_note, gen_note, sel_note, selection_note, fit_prov_note, serving_commit_note) if x]
        prov3 = {"selection": selection, "serving_commit": serving_commit, "fit_provenance": fit_prov}
        readings = cell.get("readings")
        if not isinstance(readings, Mapping):
            why = base.get("refused") or base.get("missing") or "no reading"
            rows.append(_pr_null_row(model_key=mk, view=view, cell=cid, arm=arm, cap=base_cap, leg=f"r{base_cap}", rung_state=rung_state,
                                     gen=gen, sel=sel, source=src([f"cells[{ck}].readings = null", f"legs[{base_key}]: {str(why)[:160]}", *notes]),
                                     prov3=prov3))
            continue
        summary = base.get("summary") or {}
        for rk, rd in readings.items():
            counts = _pr_reading_counts(rd["items"])
            for a, b in (("n", "n"), ("n_scored", "n_scored"), ("n_correct", "n_correct"), ("n_at_cap_residual", "n_at_cap"), ("n_think_unscored", "n_think_unscored")):
                if rd.get(a) != counts[b]:
                    raise AnalysisError(f"paper rows: cells[{ck}].readings.{rk}.{a} = {rd.get(a)} but its items sum to {counts[b]}")
            n, ns, nu, nc, nac = int(rd["n"]), int(rd["n_scored"]), int(rd["n_unscored"]), int(rd["n_correct"]), int(rd["n_at_cap_residual"])
            if nu != n - ns:
                raise AnalysisError(f"paper rows: cells[{ck}].readings.{rk}: n_unscored {nu} != n {n} - n_scored {ns}")
            reads = [f"cells[{ck}].readings.{rk}: n (-> n_paired), n_scored, n_unscored, n_correct, n_at_cap_residual (-> n_at_cap), n_think_unscored (-> unscored_reason)"]
            # schema /4 (item 8): the refusal-shaped count where the reading COUNTED it (its items' flags re-summed and held to the headline);
            # a reading written before the class existed carries no key -> null (absence, never zero), said in `source`
            nrs: int | None = None
            nrs_of: int | None = None
            if "n_refusal_shaped" in rd or "n_refusal_shaped_of" in rd:
                # the count and its denominator travel together: a reading that counted the class filed both, each re-summed from its items
                for f in ("n_refusal_shaped", "n_refusal_shaped_of"):
                    if f not in rd:
                        raise AnalysisError(f"paper rows: cells[{ck}].readings.{rk} carries a refusal-shaped count without its pair ({f} missing)")
                    if rd[f] != counts[f]:
                        raise AnalysisError(f"paper rows: cells[{ck}].readings.{rk}.{f} = {rd[f]} but its items sum to {counts[f]}")
                nrs, nrs_of = int(rd["n_refusal_shaped"]), int(rd["n_refusal_shaped_of"])
                reads.append(f"readings.{rk}.n_refusal_shaped / n_refusal_shaped_of = the count over the scored draws the classifier examined ({RFS.RULE})")
            else:
                reads.append("n_refusal_shaped / n_refusal_shaped_of: this reading carries no refusal-shaped count (written before the class existed) -> null together")
            if rk == f"r{base_cap}":
                cap = base_cap
                rate = summary.get("at_cap_rate")
                if rate is None or abs(float(rate) - nac / n) > 1e-6:
                    raise AnalysisError(f"paper rows: legs[{base_key}].summary.at_cap_rate {rate} != n_at_cap / n = {nac / n}")
                untnac = summary.get("n_unterminated_not_at_cap")
                if untnac != counts["n_unterminated_not_at_cap"]:
                    raise AnalysisError(f"paper rows: legs[{base_key}].summary.n_unterminated_not_at_cap {untnac} != the reading's items {counts['n_unterminated_not_at_cap']}")
                n_rescored = int(rd.get(N_UNTERMINATED_FINISHED_RESCORED_KEY) or 0)
                if summary.get("n_think_block_unterminated") != rd["n_think_unscored"] + n_rescored:
                    raise AnalysisError(f"paper rows: legs[{base_key}].summary.n_think_block_unterminated {summary.get('n_think_block_unterminated')} != "
                                        f"readings.{rk}.n_think_unscored {rd['n_think_unscored']}"
                                        + (f" + {N_UNTERMINATED_FINISHED_RESCORED_KEY} {n_rescored}" if n_rescored else ""))
                reads.append(f"legs[{base_key}].summary: at_cap_rate, n_unterminated_not_at_cap (n_think_block_unterminated cross-checked)")
            else:
                cap = max(counts["caps"]) if counts["caps"] else base_cap
                rate = nac / n
                untnac = counts["n_unterminated_not_at_cap"]
                reads.append("at_cap_rate = n_at_cap_residual / n (the merged reading files no rate)")
                reads.append("n_unterminated_not_at_cap summed from the reading's per-item flags (think_block_unterminated and not at_cap)")
                reads.append(f"cap = the highest cap among the merged items' sources {counts['caps']}")
            ur, ur_note = PRW.unscored_reason(nu, unterminated_think_block=int(rd["n_think_unscored"]))
            if rd.get(UNTERMINATED_FINISHED_POLICY_KEY):
                # the reader's non-default policy rides every row written under it (the schema is untouched): the count where it acted
                n_resc = int(rd.get(N_UNTERMINATED_FINISHED_RESCORED_KEY) or 0)
                reads.append(f"{UNTERMINATED_FINISHED_POLICY_KEY}={rd[UNTERMINATED_FINISHED_POLICY_KEY]}: "
                             + (f"{n_resc} draw(s) that finished below the cap inside an unterminated think block are scored WRONG under both "
                                f"branches and counted in n_scored (the store left them unscored)" if n_resc
                                else "no unterminated-but-finished draw in this reading (nothing re-scored)"))
            ci = ci_index.get((ck, rk))
            if ci is not None:
                p, s = ci["primary"], ci["score_as_is"]
                if (p["n_correct"], p["n_scored"], s["n_correct"], s["n_scored"]) != (nc, ns, nc, n):
                    raise AnalysisError(f"paper rows: estimands[{ci['contrast']!r}].{rk} acc_by_leg.{ci['role']} counts "
                                        f"({p['n_correct']}/{p['n_scored']}, {s['n_correct']}/{s['n_scored']}) != the reading's ({nc}/{ns}, {nc}/{n})")
                pc, cl, ch, ps, sl, sh, method = p["point"], p["ci_low"], p["ci_high"], s["point"], s["ci_low"], s["ci_high"], PRW.CI_BOOTSTRAP
                reads.append(f"estimands[{ci['contrast']}].{rk}.stock.{{primary,score_as_is}}.acc_by_leg.{ci['role']}: point, ci_low, ci_high")
            else:
                pc, ps = PRW.ratio(nc, ns), PRW.ratio(nc, n)
                cl, ch = PRW.exact_binomial_ci(nc, ns)
                sl, sh = PRW.exact_binomial_ci(nc, n)
                method = PRW.CI_EXACT
                reads.append("no bootstrap for this reading in the report (the cell's contrast was refused by design, so it filed no estimand) "
                             "-> exact binomial (Clopper-Pearson) CI" if ck in by_design else
                             "no bootstrap for this reading in the report (the cell sits in no contrast) -> exact binomial (Clopper-Pearson) CI")
            rows.append(PRW.make_row(instrument=PRW.INSTRUMENT_GSM8K, model_key=mk, view=view, cell=cid, arm=arm, cap=cap, leg=rk,
                                     n_paired=n, n_scored=ns, n_unscored=nu, n_at_cap=nac, at_cap_rate=float(rate), n_unterminated_not_at_cap=int(untnac),
                                     n_correct=nc, point_conditional=pc, ci_conditional_low=cl, ci_conditional_high=ch,
                                     point_score_as_is=ps, ci_score_as_is_low=sl, ci_score_as_is_high=sh, ci_method=method,
                                     rung_state=rung_state, object_generation=gen, selection_outcome=sel, unscored_reason=ur,
                                     source=src([*reads, *([ur_note] if ur_note else []), *notes]), n_refusal_shaped=nrs, n_refusal_shaped_of=nrs_of, **prov3))
        rung_key = cell.get("rung_leg")
        rung = legs.get(rung_key or "") or {}
        rs = rung.get("summary") or {}
        if rung_key and rs:
            rcap = int(rung.get("cap") or _pr_cap_of_leg_key(rung_key))
            n, ns, nu = int(rs["n_paired"]), int(rs["n_scored"]), int(rs["n_unscored"])
            if int(rs["n"]) != n or nu != n - ns:
                raise AnalysisError(f"paper rows: legs[{rung_key}].summary: n {rs['n']} / n_paired {n} / n_scored {ns} / n_unscored {nu} disagree")
            redraws = cell.get("redraws") or {}
            nc = sum(1 for x in redraws.values() if x.get("new_scored") and float(x.get("new_value") or 0.0) == 1.0)
            if len(redraws) != n or sum(1 for x in redraws.values() if x.get("new_scored")) != ns:
                raise AnalysisError(f"paper rows: cells[{ck}].redraws ({len(redraws)} items, {sum(1 for x in redraws.values() if x.get('new_scored'))} scored) "
                                    f"!= legs[{rung_key}].summary (n_paired {n}, n_scored {ns})")
            if ns and abs(float(rs["acc"]) - nc / ns) > 1e-6:
                raise AnalysisError(f"paper rows: legs[{rung_key}].summary.acc {rs['acc']} != redraws' n_correct / n_scored = {nc}/{ns}")
            nac, rate, untnac = int(rs["n_at_cap"]), float(rs["at_cap_rate"]), int(rs["n_unterminated_not_at_cap"])
            ur, ur_note = PRW.unscored_reason(nu, unterminated_think_block=int(rs["n_think_block_unterminated"]))
            cl, ch = PRW.exact_binomial_ci(nc, ns)
            sl, sh = PRW.exact_binomial_ci(nc, n)
            reads = [f"legs[{rung_key}].summary: n_paired, n_scored, n_unscored, n_at_cap, at_cap_rate, n_unterminated_not_at_cap, n_think_block_unterminated (the rung store's gen_stats.json as the report read it)",
                     f"n_correct summed from cells[{ck}].redraws (new_scored and new_value == 1.0); summary.acc cross-checked",
                     "the report bootstraps no rung leg on its own -> exact binomial (Clopper-Pearson) CI",
                     f"n_paired = the {n} item(s) the rung re-drew (the ladder's eligible set), not the 200 posed"]
            nrs_rung, nrs_rung_of, nrs_note = _pr_rung_refusal_shaped(rung, rung_key)
            reads.append(nrs_note)
            rows.append(PRW.make_row(instrument=PRW.INSTRUMENT_GSM8K, model_key=mk, view=view, cell=cid, arm=arm, cap=rcap, leg=f"r{rcap}",
                                     n_paired=n, n_scored=ns, n_unscored=nu, n_at_cap=nac, at_cap_rate=rate, n_unterminated_not_at_cap=untnac,
                                     n_correct=nc, point_conditional=PRW.ratio(nc, ns), ci_conditional_low=cl, ci_conditional_high=ch,
                                     point_score_as_is=PRW.ratio(nc, n), ci_score_as_is_low=sl, ci_score_as_is_high=sh, ci_method=PRW.CI_EXACT,
                                     rung_state=rung_state, object_generation=gen, selection_outcome=sel, unscored_reason=ur,
                                     source=src([*reads, *([ur_note] if ur_note else []), *notes]), n_refusal_shaped=nrs_rung, n_refusal_shaped_of=nrs_rung_of, **prov3))
    return rows, []


def write_paper_rows_beside(report: Mapping[str, Any], report_path: Path | str) -> dict[str, Any]:
    """``paper_rows.json`` + ``paper_rows.csv`` beside ``report_path`` (nothing else there is touched)."""
    rows, skipped = paper_rows_of_report(report, report_path=report_path)
    return PRW.write_paper_rows(rows, Path(report_path).expanduser().parent, instrument=PRW.INSTRUMENT_GSM8K, report_path=report_path,
                                not_exported=skipped)


def paper_rows_only(path: Path) -> int:
    """``--paper-rows-only REPORT``: regenerate the export from an EXISTING production report — no store opened, no
    analysis re-run, the report not rewritten. A report that is not a production result gets no paper rows (exit 3)."""
    path = Path(path).expanduser().resolve()
    report = _read_json(path)
    if not isinstance(report, Mapping):
        raise AnalysisError(f"--paper-rows-only {path}: not a JSON object")
    refusals = production_refusals(report)
    if refusals:
        print(f"[g8k] {path}: NOT a production result — no paper rows written — " + "; ".join(refusals), file=sys.stderr)
        return EXIT_NOT_PRODUCTION
    written = write_paper_rows_beside(report, path)
    print(f"[g8k] paper rows -> {written['json']} ({written['n_rows']} rows) + {written['csv']}")
    return 0


def gate_report(path: Path) -> int:
    report = _read_json(Path(path).expanduser())
    if not isinstance(report, Mapping):
        raise AnalysisError(f"--gate-report {path}: not a JSON object")
    refusals = production_refusals(report)
    if refusals:
        print(f"[g8k] {path}: NOT a production result — " + "; ".join(refusals), file=sys.stderr)
        return EXIT_NOT_PRODUCTION
    gates = _mapping(report.get("gates"))
    passed = [g for g in GATE_NAMES if _mapping(gates.get(g)).get("status") == GATE_PASS]
    print(f"[g8k] {path}: production result ({report['design']['n_contrasts_computable']} of {report['design']['n_contrasts_designed']} contrasts computable; "
          f"{len(passed)} of {len(GATE_NAMES)} gates PASSED" + (f", waivable pending: {gates_pending(gates)}" if gates_pending(gates) else "") + ")")
    return 0


def diff_reports(path_a: Path, path_b: Path) -> int:
    a, b = _read_json(Path(path_a).expanduser()), _read_json(Path(path_b).expanduser())
    if not isinstance(a, Mapping) or not isinstance(b, Mapping):
        raise AnalysisError(f"--diff-reports: {path_a} / {path_b} must both be JSON objects")
    d = numeric_leaf_diff(a, b)
    if not d["equal"]:
        lines = [f"  differing {T._path_text(p)}: {va} != {vb}" for p, va, vb in d["differing"][:10]]
        lines += [f"  type changed {T._path_text(p)}: {va!r} vs {vb!r}" for p, va, vb in d["type_changed"][:10]]
        lines += [f"  only in {path_a}: {T._path_text(p)}" for p in d["missing"][:10]]
        lines += [f"  only in {path_b}: {T._path_text(p)}" for p in d["added"][:10]]
        print(f"[g8k] the reports' numeric leaves DIFFER — {len(d['differing'])} differing, {len(d['type_changed'])} type changed, {len(d['missing'])} missing, "
              f"{len(d['added'])} added over {d['shared_numeric']} shared:\n" + "\n".join(lines), file=sys.stderr)
        return EXIT_REPORTS_DIFFER
    print(f"[g8k] numeric leaves equal: {d['shared_numeric']} shared (outside {', '.join(REPORT_DIFF_SKIP)} and the by-design split leaves)")
    return 0


def print_table(report: Mapping[str, Any]) -> None:
    print(f"{EXPERIMENT} — store pin {report['sha']}, analysis code {report['analysis_code_sha'][:12]}, matrix {report['matrix']['sha256'][:12]}")
    for row in report["controls"]:
        s = row["summary"]
        state = row["missing"] or row["refused"] or "ok"
        acc = "  —  " if s.get("acc") is None else f"{s['acc']:.3f}"
        label = f"  [{row['thinking_label']}]" if row.get("thinking_mode") == THINKING_ARM_NOT_AVAILABLE else ""
        print(f"  {row['leg']:<72} acc {acc} n_unscored {s.get('n_unscored')!s:>4} at_cap {s.get('n_at_cap')!s:>3}  {state[:60]}{label}")
    for c in report["contrast_summary"]:
        if not c["computable"]:
            print(f"  {c['name']}: no estimand — {c['reason']}")
            continue
        m = c[READING_MERGED]
        p = m["point"][AXIS_STOCK][BRANCH_PRIMARY]
        lo, hi = m["ci"][AXIS_STOCK][BRANCH_PRIMARY]
        print(f"  {c['name']}: merged {100 * p:+.1f} pp [{100 * lo:+.1f}, {100 * hi:+.1f}] -> {m['word']}")
    for rec in report["refused"]:
        extra = (f" — ON draws {rec.get('attempt_classes')}; {rec.get('observation') or ''}"
                 if rec.get("status") == LEG_STATUS_REFUSED_CERTIFIED else "")
        print(f"  REFUSED by name: {rec['leg']} ({rec['code']}){extra}")
    for h in report["held"]:
        print(f"  HELD: {h['model_key']}/{h['view']}/{h['cell']}")
    for lk, gaps in _mapping(report.get(V.RECORD_GAPS_KEY)).get("legs", {}).items():
        print(f"  RECORD GAP: {lk} — {record_gap_text(gaps)}")
    g = report["gates"]
    print(f"  gates failed: {g['failed'] or 'none'}; pending: {g['pending'] or 'none'}")
    print(f"  verdict: {report['verdict']['verdict']}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gen-root", action="append", default=[], metavar="PATH[=PIN]",
                    help="a local copy of a wave root (controls/<TAG>/<cell slug>/thinking_card/..., overlap/...; repeatable). Bare PATH: bound to "
                         "--sha, a leg under it may be from --sha or any --also-sha tree (D131). PATH=PIN (plan P5 / B6): a wave-isolated pull "
                         "bound to that ONE admitted tree (--sha or an --also-sha); a row's out root under two roots refuses (a filesystem union)")
    ap.add_argument("--sha", default=None, help="the store pin: the wave sha every cell's producer tree carries")
    ap.add_argument("--also-sha", action="append", default=[], metavar="SHA40",
                    help="D131 / plan P5: a further producer tree the wave's cells may come from (repeatable; a full 40-hex sha), verified on "
                         "--repo-root to differ from --sha in --tree-diff-allow paths only; needs --tree-set-reason")
    ap.add_argument("--tree-diff-allow", action="append", default=[], metavar="GLOB",
                    help="a path glob the declared trees may differ from the primary in (repeatable; fnmatch, case-sensitive)")
    ap.add_argument("--tree-diff-allow-file", type=Path, default=None, metavar="REVIEW_MD",
                    help="the REVIEWED allowlist file (plan P5 / B7): every line `- `<glob>` — <evidence>` contributes one glob to --tree-diff-allow; "
                         "the path and the per-glob evidence are filed as tree_set.allow_file / tree_set.allow_evidence")
    ap.add_argument("--tree-set-reason", default=None, metavar="TEXT",
                    help="why the cells come from more than one tree — filed verbatim as tree_set.why (required with --also-sha)")
    ap.add_argument("--waves", default=None, metavar="WAVE[,WAVE]",
                    help="the REQUIRED-EVIDENCE set (plan P5 / review 3): the evidence rows (status launch; `complete` tolerated as a synonym) of these waves are mandatory — "
                         "a missing leg fails the completeness gate; held / deferred rows and rows of other waves land in design.status_by_row "
                         "and are never collected. Absent: every evidence row of the matrix (the published behaviour)")
    ap.add_argument("--peer-report", action="append", default=[], metavar="REPORT[=PIN]",
                    help="the Llama lane's PRODUCTION report of this experiment at ITS pin (plan P5, finding 8; repeatable): its llama column is "
                         "assembled into the eight-arm figures and held out of the pooled panel; REPORT=PIN must equal the report's own sha. "
                         "Absent: design.peer_reports records status absent and the llama column reads 'pending — Llama lane'")
    ap.add_argument("--matrix", type=Path, action="append", default=None,
                    help="lane B's frozen matrix (default: configs/sdfbase/gsm8k/matrix.json of this tree); REPEATABLE — several matrices read as one "
                         "design (a matrix-union read: a later matrix's LAUNCH row replaces an earlier HELD row of the same cell; any other duplicate refuses)")
    ap.add_argument("--store-pins", type=Path, action="append", default=None, metavar="JSON",
                    help="JSON {'<model>/<view>/<cell>@<cap>': '<tree sha>'} (repeatable, merged): every leg admitted by ITS OWN producer tree — the "
                         "per-row admission (the IFEval battery's); --sha is then the record label and the published anchors' pin; not with --also-sha")
    ap.add_argument("--ids", default=None, help="the canonical 200 paired ids (a JSON list); absent -> the population gate is PENDING")
    ap.add_argument("--overlap", type=Path, default=None, help="the overlap record scripts/gsm8k_overlap_check.py wrote (gate A10)")
    ap.add_argument("--sampling-rows", default=None, help="the pinned sampling rows (default: configs/sdfbase/math500/sampling_rows.yaml)")
    ap.add_argument("--render-facts", default=None)
    ap.add_argument("--pins", type=Path, default=None, help="the object pin file (default: configs/sdfbase_full22_pins.json)")
    ap.add_argument("--expected-refusals", type=Path, default=None, help="JSON {leg key: refusal code} beside the matrix rows' expected_refusal")
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--gate-report", type=Path, default=None, metavar="REPORT")
    ap.add_argument("--diff-reports", type=Path, nargs=2, default=None, metavar=("A", "B"))
    ap.add_argument("--paper-rows-only", type=Path, default=None, metavar="REPORT",
                    help="regenerate paper_rows.json + paper_rows.csv beside an EXISTING production report and exit (D103 / D104): "
                         "no store opened, no analysis re-run, the report not rewritten")
    ap.add_argument("--require-report-match", type=Path, default=None, metavar="REPORT")
    ap.add_argument("--analysis-sha", default=None)
    ap.add_argument("--bootstrap", type=int, default=pr.DEFAULT_BOOTSTRAP, metavar="B")
    ap.add_argument("--seed", type=int, default=pr.DEFAULT_BOOTSTRAP_SEED)
    ap.add_argument("--ci", type=float, default=pr.DEFAULT_CI)
    ap.add_argument("--allow-pending-gates", action="store_true")
    ap.add_argument("--unterminated-finished-policy", choices=UNTERMINATED_FINISHED_POLICIES, default=UNTERMINATED_FINISHED_UNSCORED,
                    help="the READER's treatment of a draw that finished below the cap inside an unterminated think block: `unscored` (default; "
                         "the store's scoring stands, nothing is written) or `failure` (the user's ruling of 2026-09-13: scored wrong under both "
                         "branches; named in design, every affected reading, the estimand sentence and the paper rows' source)")
    ap.add_argument("--no-eval-logs", action="store_true")
    ap.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    ap.add_argument("--wandb-name", default=None)
    ap.add_argument("--wandb-name-suffix", default="")
    ap.add_argument("--figs-dir", type=Path, default=None)
    ap.add_argument("--no-figs", action="store_true")
    ap.add_argument("--figure-exclude-family", action="append", default=[], metavar="MODEL_KEY",
                    help="leave this model family out of every FIGURE and of the per-model / per-leg rows of the tables JSON (repeatable; "
                         "must be a model key of the design). The report's cells, legs, contrasts and gates keep every leg and no numeric "
                         f"leaf moves; the exclusion is recorded as figures.{FIGURES_EXCLUDED_KEY} and in both sidecars")
    ap.add_argument("--figure-exclude-arm", action="append", default=[], metavar="CELL",
                    help="leave this matrix CELL out of every FIGURE (repeatable; plan P5, finding 7 — the paper view passes the four PSR cells and "
                         f"safety_refusal__lora_r4). Validated against the report's cells (unknown = error; {CELL_NONE!r} cannot be excluded); "
                         f"recorded as figures.{FIGURES_EXCLUDED_ARMS_KEY} / figures.{FIGURES_SELECTION_KEY}; no aggregate is recomputed, the "
                         "tables JSON is untouched, every title carries 'N legs in report · k shown'")
    ap.add_argument("--figure-modes", default=None, metavar="MODE[,MODE]",
                    help=f"the sampling modes the figures DRAW (GSM8K has the one {MODE_PRIMARY!r} mode; validated against the matrix); recorded "
                         f"as figures.{FIGURES_MODES_KEY} only when given")
    ap.add_argument("--mirror-dir", type=Path, default=Path(MIRROR_DIR_DEFAULT))
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if a.gate_report is not None:
        return gate_report(a.gate_report)
    if a.diff_reports is not None:
        return diff_reports(*a.diff_reports)
    if a.paper_rows_only is not None:
        return paper_rows_only(a.paper_rows_only)
    if not a.gen_root or a.sha is None:
        ap.error("--gen-root and --sha are required for an analysis run")
    if a.also_sha and not (a.tree_set_reason or "").strip():
        ap.error("--also-sha requires --tree-set-reason <text>: why the wave's cells come from more than one tree")
    repo_root = (a.repo_root or _HERE.parent).expanduser().resolve()
    matrix_paths = list(a.matrix) if a.matrix else [repo_root / "configs" / "sdfbase" / "gsm8k" / "matrix.json"]
    matrix_union: dict[str, Any] | None = None
    matrix: Any = matrix_paths[0]
    if len(matrix_paths) > 1:
        matrix, matrix_union = load_matrix_union(matrix_paths)
    if a.store_pins and a.also_sha:
        ap.error("--store-pins and --also-sha are two admissions of the same legs — give one")
    out = a.out.expanduser().resolve() if a.out else None
    waves = [w.strip() for w in str(a.waves).split(",") if w.strip()] if a.waves is not None else None
    if a.waves is not None and not waves:
        ap.error("--waves names no wave")
    figure_modes_arg = [m.strip() for m in str(a.figure_modes).split(",") if m.strip()] if a.figure_modes is not None else None
    if a.figure_modes is not None and not figure_modes_arg:
        ap.error("--figure-modes names no mode")
    # B7: the reviewed allowlist FILE joins the --tree-diff-allow globs; its path and per-glob evidence ride the tree_set block
    allow = list(a.tree_diff_allow)
    allow_file: str | None = None
    allow_evidence: dict[str, str] | None = None
    if a.tree_diff_allow_file is not None:
        globs, allow_evidence = M.load_tree_diff_allow_file(a.tree_diff_allow_file)
        allow += [g for g in globs if g not in allow]
        allow_file = str(Path(a.tree_diff_allow_file).expanduser())
    # the declared producer trees are verified on the local repository BEFORE any leg is read; a refusal here is a
    # production refusal by name (exit 3) — no store opened, no figure, mirror or W&B call. analyze() verifies the same
    # declaration again from its fields (it never trusts a verified block), so this pass only buys the early boundary.
    tree_set = M.verify_tree_set(a.sha, a.also_sha, allow, repo_root=repo_root, reason=a.tree_set_reason, allow_file=allow_file,
                                 allow_evidence=allow_evidence)
    if tree_set["problems"]:
        stub = {"experiment": EXPERIMENT, "task": TASK, "sha": a.sha, TREE_SET_KEY: tree_set,
                "verdict": {"word": None, "verdict": NOT_MEASURABLE, "reason": f"{TREE_SET_KEY} refused before any leg was read"}}
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(stub, indent=1, default=str), encoding="utf-8")
        print("[g8k] NOT a production result — " + "; ".join(production_refusals(stub)), file=sys.stderr)
        return EXIT_NOT_PRODUCTION
    report = analyze(list(a.gen_root), a.sha, matrix=matrix, canonical_ids=a.ids, sampling_rows=a.sampling_rows, render_facts=a.render_facts, pins=a.pins,
                     expected_refusals=a.expected_refusals, overlap=a.overlap, n_resamples=a.bootstrap, seed=a.seed, ci=a.ci,
                     allow_pending_gates=a.allow_pending_gates, read_eval_logs=not a.no_eval_logs, analysis_sha=a.analysis_sha, repo_root=repo_root,
                     tree_set=tree_set if a.also_sha else None, waves=waves, peer_reports=list(a.peer_report),
                     unterminated_finished_policy=a.unterminated_finished_policy, store_pins=list(a.store_pins) if a.store_pins else None)
    if matrix_union is not None:
        report["matrix"]["union"] = matrix_union
    print_table(report)

    def write_report() -> None:
        if out:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
            print(f"\n[g8k] full report -> {out}")

    refusals = production_refusals(report)
    if refusals:
        write_report()
        print("[g8k] NOT a production result — " + "; ".join(refusals), file=sys.stderr)
        return EXIT_NOT_PRODUCTION
    if a.require_report_match is not None:
        match_path = a.require_report_match.expanduser().resolve()
        other = _read_json(match_path)
        if not isinstance(other, Mapping):
            raise AnalysisError(f"--require-report-match {match_path}: not a JSON object")
        d = numeric_leaf_diff(other, report)
        report["report_match"] = {"against": str(match_path), "equal": d["equal"], "shared_numeric": d["shared_numeric"],
                                  "differing": [[T._path_text(p), va, vb] for p, va, vb in d["differing"]],
                                  "missing": [T._path_text(p) for p in d["missing"]], "added": [T._path_text(p) for p in d["added"]],
                                  "type_changed": [[T._path_text(p), va, vb] for p, va, vb in d["type_changed"]]}
        report["gates"]["A9_report_match"] = gate_report_match(report["report_match"])
        # A9 is filed AFTER analyze(): the aggregates, the verdict and the top items are re-derived from the records so the
        # saved report of a failed match reads as what it is (--gate-report refuses it; nothing downstream reads "all passed")
        refresh_gate_aggregates(report["gates"])
        report["verdict"] = verdict_of(report["gates"], report["contrasts"], report["verdict"]["words"], allow_pending_gates=a.allow_pending_gates)
        report["top_items"] = top_items_of(report)
        if not d["equal"]:
            write_report()
            print(f"[g8k] this run's numeric leaves DIFFER from {match_path} — {len(d['differing'])} differing, {len(d['type_changed'])} type changed, "
                  f"{len(d['missing'])} missing, {len(d['added'])} added over {d['shared_numeric']} shared; nothing published", file=sys.stderr)
            return EXIT_REPORTS_DIFFER
    figs: dict[str, Any] = {}
    if not a.no_figs:
        figs_dir = (a.figs_dir or ((out.parent if out else Path.cwd()) / "figs")).expanduser().resolve()
        figs = make_figures(report, figs_dir, exclude_families=a.figure_exclude_family, exclude_arms=a.figure_exclude_arm, modes=figure_modes_arg)
        report["figures"] = figs
        # plan P5: the PUBLICATION GATE — on EVERY render (C1), before the W&B / mirror pass: every figure's rows sidecar is read
        # and publication is refused if any figure presents a PSR arm id, or a cell / family / mode this render declared out (exit 6)
        try:
            report[PUBLICATION_GATE_KEY] = publication_gate_record(report, figs, figs_dir)
        except PG.PublicationRefused as e:
            report[PUBLICATION_GATE_KEY] = {"status": PG.GATE_REFUSED, "violations": list(e.violations), "rule": PG.RULE}
            write_report()
            print(f"[g8k] PUBLICATION REFUSED — {len(e.violations)} figure record(s) present what must not be published: "
                  + "; ".join(e.violations), file=sys.stderr)
            return PG.EXIT_PUBLICATION_REFUSED
        if not a.no_mirror:
            refuse_mixed_render(a.mirror_dir, figs.get(FIGURES_EXCLUDED_KEY) or [], list(figs["written"]),
                                arms=figs.get(FIGURES_EXCLUDED_ARMS_KEY) or [], modes=figs.get(FIGURES_MODES_KEY))
            report["mirrored"] = mirror_figures(figs_dir, a.mirror_dir.expanduser().resolve(), manifest=figs)
    url = log_wandb(report, figs, mode=a.wandb_mode, name=a.wandb_name, name_suffix=a.wandb_name_suffix)
    report["wandb_url"] = url
    write_report()
    if out:
        # the flat per-row export beside the report — read from the report just written, never an estimator
        written = write_paper_rows_beside(report, out)
        print(f"[g8k] paper rows -> {written['json']} ({written['n_rows']} rows) + {written['csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
