#!/usr/bin/env python
"""bcb-bfcl-200 analysis, generalised to ROSTER-DRIVEN, PER-FAMILY WAVES.

A WAVE is one model row served at one tree: one unsteered cell (`none`) and
K steered cells (prompted or fitted), each filing the runnable legs
(`inspect/bigcodebench_200`, `inspect/bfcl_200`) under the benchmarks' own
decoding protocol. Every contrast in the lane is a steered cell against ITS
wave's `none` — never steered-vs-steered, so a batch-8 object is never paired
with a batch-64 one. The roster (`configs/capeval/bcb_bfcl_200_widen_roster.yaml`,
Slice B) names the waves, their cells, legs, objects and generations; the
analysis reads it and REFUSES a store for a cell the roster does not list as
runnable, and exports the queued / blocked entries as rows with a `status`.

Pipeline (each stage is a flag; they chain in this order when combined)::

    --roster R --wave W       the wave being analysed (required for --pull / --analyze / --paper-rows)
    --pull <suffix>           fetch  /vol/tinycap/gen/<row>/<cell>/thinking_off/<suffix>/  for every runnable
                              cell (a `--serving-engine lora_native` run's under /vol/tinycap/gen/native/…, the
                              namespace the native launcher writes into; the lens default byte-identical),
                              the per-cell ladder certificates + the wave record under _ladder/<suffix>/,
                              and every fitted object's SIDECAR + its staged_from.json (objects_dir/<CELL ID>/…;
                              the roster's staged_id is COMPARED against staged_from.json, never used as a path) into
                              <pull root>/<suffix>/{<cell>/, _ladder/, _objects/<cell>/}
    --analyze <rung-1 dir>    identity (ONE experiment across the wave; per leg the same instrument, projected
                              through `tinycap_ladder.project_identity`), the generation guard, the paired
                              analysis per steered cell vs `none`, breakdowns, diagnostics, the protocol record,
                              ladder readings (ONLY from `ladder merge`), cost (meter ledger by wave + sandbox
                              lifetimes) and wall; writes <dir>/analysis.json, figures under
                              <figs root>/<suffix>/, logs to W&B (project `capeval`)
    --paper-rows PATH         append this wave's rows (measured + queued/blocked) to the paper parquet
    --summary A.json ...      the cross-wave summary: figures + summary.json (+ --write-results renders the doc)
    --write-results PATH      render the results doc at PATH from analysis.json files (--skeleton: pending)

What each number means, and the labels it must carry:

- **pass rate** — per leg, per cell: k passes over the 200 pinned items with the Wilson 95 %
  interval (`capeval_analysis_common.wilson_interval`). Every pinned item must carry exactly one `acc`
  row in every cell; a missing item REFUSES.
- **paired difference** — steered − none over the 200 identical items: the percentile paired
  bootstrap (B = 10,000, seed 0), the exact two-sided McNemar p on the discordant pairs and the
  discordant counts. k = 1: the intervals are wide by design and say so.
- **identity** — asserted BEFORE pairing. (1) ONE experiment across every (cell, leg) store of the
  wave: the roster's served model / revision, the tree, the engine, the hashed `tinycap` scalars,
  the `served` geometry (minus the tool-call facts, which follow the legs), the `sampling` /
  `thinking` blocks in their scalars and per-task maps on the wave's filed tasks — all through
  `tinycap_ladder.project_identity`, the ONE grouped projection rule set —
  and SAME-WAVE PROVENANCE: one launcher run id (the attempt pointers), one suffix, one tree.
  (2) Per leg, every steered cell against `none`: the instrument MUST agree, the intervention MAY
  differ and is reported. (3) The generation guard: a FITTED store's generation label is read from the
  launch-time `lane_roster.json` sidecar (digest recomputed; object sha bound to the manifest's),
  never from the current roster; object-free baselines are generation-neutral; two fitted cells
  of different generations refuse pairing (defensive — none is computed by design).
- **protocol** — the manifest's `sampling.by_task` record: what the benchmark DECLARES and what
  the stack EXECUTED per leg (BigCodeBench T 0; BFCL T 0.001 executed literally under the pinned
  clamp override), read literally; the `decode` string of every paper
  row is built from that record, never typed.
- **fit layers** — per FITTED cell from the object's own sidecar (`derived_meta.json` `layers` or
  `train_budget_meta.json` `target_layers`), never a lane constant (two objects under one
  arm label can differ by layer SET); cross-checked against the roster's declaration.
- **at-cap / unparseable / code_source / sandbox_status / model_length** — reported apart, as the
  first lane did.
- **ladder** — read ONLY from `ladder merge`'s output, bound to the rung-1 stores; per-cell
  certificates (`rung_probe.<cell>.json`) are validated against the merge's common rung.
- **cost** — every dollar line is a RESOURCE-RATE ESTIMATE, not billing: H200-hours from the
  meter's ledger (this wave's reservations and the lane's Σ against the 25 h envelope; actual
  where recorded, else forecast) at $5/h; sandbox dollars from `sandboxes.jsonl` lifetimes ending
  at the FIRST observed termination; stated as a PARTIAL SUBTOTAL naming what it excludes.

Grading-host use (repo venv; figures need matplotlib, W&B needs ``wandb`` + ``WANDB_API_KEY`` from
the secret-manager CLI — never printed here)::

    PYTHONPATH=src python scripts/bcb_bfcl_200_analysis.py --roster configs/capeval/bcb_bfcl_200_widen_roster.yaml \\
        --wave gemma4e4b_it --pull bcbw-gemma4e4b_it-<sha7> --analyze \\
        --paper-rows out/paper_rows_bcb_bfcl_200_widen.parquet
    PYTHONPATH=src python scripts/bcb_bfcl_200_analysis.py --roster ... --summary <dir1>/analysis.json <dir2>/analysis.json \\
        --write-results out/bcb_bfcl_200_widen.md
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import datetime as _dt
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sisteer.capeval.tiers import pairing as P   # noqa: E402
from sisteer.capeval.tinycap import contracts as C   # noqa: E402
from sisteer.capeval import manifest as cm   # noqa: E402
from sisteer.capeval.tinycap import rung_probe as RP   # noqa: E402
from sisteer.capeval.tinycap import widen_roster as W   # noqa: E402
from sisteer.eval import train_budget as TB   # noqa: E402


def _load_sibling(name: str):
    """A sibling script under ``scripts/`` as a module (the tests load scripts
    the same way); the public helpers of ``capeval_analysis_common`` are
    IMPORTED; the private ones below were lifted by copy from the retired
    Terminal-Bench pilot analysis."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


A = _load_sibling("capeval_analysis_common")
L = _load_sibling("tinycap_ladder")
M = _load_sibling("bcb_bfcl_200_meter")

AnalysisError = A.AnalysisError
read_json, read_jsonl, pass_rate, mcnemar_exact_p = A.read_json, A.read_jsonl, A.pass_rate, A.mcnemar_exact_p
wilson_interval = A.wilson_interval
wrap_title, finish_figure, save_figure, modal_volume_ls = A.wrap_title, A.finish_figure, A.save_figure, A.modal_volume_ls
SURFACE, INK, INK2, MUTED, GRID, BASELINE = A.SURFACE, A.INK, A.INK2, A.MUTED, A.GRID, A.BASELINE
CAT, SEQ, STATUS, DPI = A.CAT, A.SEQ, A.STATUS, A.DPI

# --------------------------------------------------------------------------- #
# constants                                                                    #
# --------------------------------------------------------------------------- #

ANALYSIS_SCHEMA_VERSION = 2
ANALYSIS_FILENAME = "analysis.json"
SUMMARY_FILENAME = "summary.json"
LANE = "bcb-bfcl-200-widen"
DEFAULT_ROSTER_REL = "configs/capeval/bcb_bfcl_200_widen_roster.yaml"
DEFAULT_PULL_ROOT = "$GRADE/tinycap/bcb-bfcl-200-widen"   # the grade root's placeholder form; pull_suffix expands it through sisteer.site
DEFAULT_FIGS_ROOT = "~/figs/bcb-bfcl-200-widen"
FIGS_URL_BASE_ENV = "SISTEER_FIGS_URL_BASE"


def figs_url_base_for(figs_root: Union[str, Path]) -> str:
    """Where the figures are linked from (analysis.json ``figures.url_base``, the results doc's figure lines):
    ``$SISTEER_FIGS_URL_BASE/<lane>`` when the figure mirror has a public base URL, else the figure root itself as an
    absolute path — analysis.json has no document to be relative to and the results doc is written wherever
    ``--write-results`` says, so only an absolute path resolves from both. ``--figs-url-base`` overrides either."""
    base = os.environ.get(FIGS_URL_BASE_ENV, "").strip()
    if base:
        return base.rstrip("/") + "/" + LANE
    return str(Path(figs_root).expanduser().resolve())


#: the default under the default figure root (a function of the environment at import; main() re-derives it from --figs-root)
DEFAULT_FIGS_URL_BASE = figs_url_base_for(DEFAULT_FIGS_ROOT)
DEFAULT_MODAL_PYTHON = "~/venvs/modal/bin/python"
DEFAULT_VOLUME = C.MODAL_VOLUME
DEFAULT_MODE = "off"
BASE_CELL = C.BASE_CELL
DEFAULT_PAPER_ROWS = "$WORK/bcb-bfcl-200-widen/analysis/paper_rows_bcb_bfcl_200_widen.parquet"   # expanded where it is written
WANDB_PROJECT = "capeval"
WANDB_JOB_TYPE = "bcb-bfcl-200-widen-analysis"
LIB_FAMILIES_REL = C.BCB_LIB_FAMILIES_PATH
BCB_TASK, BFCL_TASK, FULL_N = L.BCB_TASK, L.BFCL_TASK, L.FULL_N
TASKS: Tuple[str, ...] = (BCB_TASK, BFCL_TASK)
TASK_LABEL = {BCB_TASK: "BigCodeBench-200 (non-hard, sandboxed verify)", BFCL_TASK: "BFCL-200 (function calling)"}
TASK_SHORT = {BCB_TASK: "BCB-200", BFCL_TASK: "BFCL-200"}
#: the roster's leg atoms (`legs: {bigcodebench_200: …, bfcl_200: …}`) → row tasks
TASK_OF_ATOM: Mapping[str, str] = {C.CONTROL_BIGCODEBENCH_200: BCB_TASK, C.CONTROL_BFCL_200: BFCL_TASK}
ATOM_OF_TASK: Mapping[str, str] = {v: k for k, v in TASK_OF_ATOM.items()}
BOOTSTRAP_N, BOOTSTRAP_SEED = 10_000, 0
#: Modal's published Sandbox rates (plan facts table) — a resource-rate estimate, not billing.
SANDBOX_USD_PER_CORE_H = 0.14191
SANDBOX_USD_PER_GIB_H = 0.024012
H200_USD_PER_H = M.H200_USD_PER_H
#: The launch-time roster snapshot the launcher writes into every store root (`widen_roster`
#: is the ONE writer/reader of its name, key set, vocabulary and digest — aliased, never restated).
LANE_ROSTER_SIDECAR = W.SNAPSHOT_FILENAME
LANE_ROSTER_FIELDS: Tuple[str, ...] = W.SNAPSHOT_FIELDS
GENERATION_FINAL = W.GENERATION_FINAL
GENERATION_BATCH8 = W.GENERATION_BATCH8
GENERATION_BATCH64 = W.GENERATION_BATCH64          # the batch-64 re-fit lineage; rows name the mix through it
OBJECT_GENERATIONS: Tuple[str, ...] = W.GENERATIONS
ROSTER_STATUSES: Tuple[str, ...] = W.STATUSES
#: where --pull lays the object sidecars: <rung dir>/_objects/<cell id>/<sidecar file>
OBJECTS_SUBDIR = "_objects"
#: the ladder's own layout (`rung_probe`): `_ladder/<suffix>/rung_probe.<cell>.json` + `rung_admission.json`
LADDER_SUBDIR = RP.LADDER_SUBDIR
RUNG_ADMISSION_FILENAME = RP.ADMISSION_FILENAME
#: a per-cell certificate's name, and nothing else: `rung_probe.<cell>.json` (a cell id carries no '.' or '/');
#: `rung_probe.<cell>.<ts>.superseded.json` is a RECORDED supersession (`rung_probe.superseded_certificate_path`), listed, never read
CERTIFICATE_NAME_RE = re.compile(r"rung_probe\.([^./]+)\.json")
#: the sidecar keys that name a fit's layer set: derived objects `layers`, direct fits `target_layers`
SIDECAR_LAYER_KEYS: Tuple[str, ...] = ("layers", "target_layers")
#: the object sidecars by fit kind — `train_budget`'s own names (a derived object carries `derived_meta.json`)
DEFAULT_SIDECAR_OF_KIND: Mapping[str, str] = {"lora_subtraction": TB.DERIVED_META_NAME, "psr_subtraction": TB.DERIVED_META_NAME}
DEFAULT_SIDECAR = TB.BUDGET_META_NAME
#: the lane's store suffix grammar (`bcb_bfcl_200_widen_launch.out_suffix`): `bcbw-<wave>-<sha7>[-<tag>]` — the DURABLE wave
#: identity: one suffix, several launcher invocations (attempts) allowed, every one bound to it
SUFFIX_RE = re.compile(r"^bcbw-(?P<wave>[^-]+)-(?P<sha7>[0-9a-f]{7})(?:-(?P<tag>.+))?$")
#: the launcher's run id: `<sha7>-<hex8>` (`tinycap_served.mint_run_id`), minted once per invocation
RUN_ID_RE = re.compile(r"^(?P<sha7>[0-9a-f]{7})-(?P<hex>[0-9a-f]{8})$")
#: the external watcher's sweep certificates (`tinycap_sandbox_watch.STATE_DIR_DEFAULT`): `<root>/<run_id>/sweep_<fcid>.json`
SANDBOX_STATE_ROOT = "~/.local/state/tinycap-sandboxes"
#: the FINAL-reading rule's sources (`final_readings`)
FINAL_SOURCE_MERGED, FINAL_SOURCE_RUNG1, FINAL_SOURCE_PENDING = "merged", "rung1", "pending"
#: a leg the CURRENT roster marks blocked / queued AFTER its wave was measured is WITHDRAWN as a capability reading (the
#: llama BFCL instrument: the pinned template rendered no tool definitions, so 56/200 = 56/200 measured a model that never saw a
#: tool). The gate files it under `<view>[WITHDRAWN_LEGS_KEY]` and `final[leg]` becomes source `withdrawn`, final False, with the
#: numbers KEPT VERBATIM under `final[leg]["withdrawn"]` for the record — never read as a reading by the headline, the summary,
#: the figures, W&B or the paper values (the paper rows carry `status: withdrawn: <reason>` with null measurements)
FINAL_SOURCE_WITHDRAWN = "withdrawn"
WITHDRAWN_LEGS_KEY = "withdrawn_legs"
#: the reading rule, stated wherever a reading is; the figures' emphasis follows THIS rule (not the bootstrap interval)
READING_RULE_SENTENCE = ("cost detected iff the exact two-sided McNemar p < 0.05 AND the paired Δ < 0; the percentile paired-bootstrap "
                         "95 % interval is an estimation interval, not the test")
#: every per-comparison test and interval is ONE nominal comparison among the lane's contrasts (11 at present) — said once
#: per wave section and once in the summary header; no adjusted rule is adopted (the pre-registered rule stands)
UNADJUSTED_NOTE = "nominal, unadjusted; one draw per item on the pinned 200"
#: no degeneration classifier is declared in this lane — `loop_share` / `degenerate_share` stay null and the doc says so
DEGENERATION_NOTE = "degeneration: not measured (no declared classifier in this lane; `loop_share` / `degenerate_share` are null)"
QUOTE_CHARS = 1_200
N_QUOTED = 10
#: The fields that MAY differ across cells (the intervention) — reported, never a refusal.
IDENTITY_MAY_DIFFER: Tuple[Tuple[str, ...], ...] = (
    ("intervention_hash",), ("harness_config_hash",), (C.MANIFEST_KEY_ROUTE,), (C.MANIFEST_KEY_GEOMETRY,),
    ("arm_name",), ("kind",), (C.MANIFEST_KEY_SANDBOX,))
#: The run-level scalar paths every store of the wave must share beyond the projected `tinycap` block.
RUN_SCALAR_PATHS: Tuple[Tuple[str, ...], ...] = (("model_id",), ("model_revision",), ("engine",), (C.MANIFEST_KEY_PRODUCER, "tree_sha"))
SERVING_IDENTITY_PATH: Tuple[str, ...] = (C.MANIFEST_KEY_ROUTE, "serving_identity")
PER_CELL_IDENTITY_PATHS: Tuple[Tuple[str, ...], ...] = (
    ("intervention_hash",), ("arm_name",), ("kind",), (C.MANIFEST_KEY_ROUTE, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256),
    (C.MANIFEST_KEY_ROUTE, C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL), (C.MANIFEST_KEY_PROVENANCE, "artifact_sha256"))
#: manifest `sampling` / `thinking` blocks — projected like the identity (task maps on the retained tasks)
PROJECTED_BLOCKS: Tuple[str, ...] = (C.MANIFEST_KEY_SAMPLING, C.MANIFEST_KEY_THINKING)
SAMPLING_MODE_PROTOCOL = "protocol"
#: BFCL executes 0.001 literally; BigCodeBench greedy. Read from the manifest's record; these are the
#: LITERAL protocol temperatures, held against the record (never substituted for it).
PROTOCOL_EXPECTED_TEMPERATURE: Mapping[str, float] = {BCB_TASK: 0.0, BFCL_TASK: 0.001}
FULL22_ANCHORS_NOTE = ("full22 export (MMLU / GSM8K, n = 200, served, thinking OFF) — qualitative only: other instruments, other item "
                       "sets; anchors per (family, arm) from the spec's survey")
#: Capability anchors (full22 export): (family, arm key) -> {MMLU, GSM8K}; `none` per family.
FULL22_ANCHORS: Mapping[Tuple[str, str], Mapping[str, float]] = {
    ("gemma4e4b", "none"): {"MMLU": 0.715, "GSM8K": 0.880}, ("gemma4e4b", "si_safety"): {"MMLU": 0.715, "GSM8K": 0.915},
    ("gemma4e4b", "const_psr"): {"MMLU": 0.660, "GSM8K": 0.890}, ("gemma4e4b", "const_lora"): {"MMLU": 0.695, "GSM8K": 0.895},
    ("qwen35_4b", "none"): {"MMLU": 0.720, "GSM8K": 0.900}, ("qwen35_4b", "diff_lora"): {"MMLU": 0.720, "GSM8K": 0.880},
    ("nemotron3nano4b", "none"): {"MMLU": 0.590, "GSM8K": 0.910}, ("nemotron3nano4b", "diff_lora"): {"MMLU": 0.570, "GSM8K": 0.925}}

# --------------------------------------------------------------------------- #
# lifted private helpers (copied from the retired tb_pilot_analysis)          #
# --------------------------------------------------------------------------- #


def _pct(x: Optional[float], digits: int = 1) -> str:
    return "n/a" if x is None else f"{100.0 * x:.{digits}f} %"


def _pp(x: Optional[float], digits: int = 1) -> str:
    return "n/a" if x is None else f"{100.0 * x:+.{digits}f} pp"


def _p_value(p: Any) -> str:
    """A p-value for print: ``< 0.001`` below 0.001, else three significant figures — never ``0.000``
    (8.7e-6 and 3.9e-4 would otherwise both print as 0.000)."""
    if p is None:
        return "n/a"
    p = float(p)
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3g}" if p < 0.1 else f"{p:.3f}"


def _p(p: Any) -> str:
    """``p < 0.001`` or ``p = 0.125`` — the same value as :func:`_p_value`, with its relation."""
    v = _p_value(p)
    return "p n/a" if v == "n/a" else ("p " + v if v.startswith("<") else f"p = {v}")


def thinking_mode_of(value: Any) -> str:
    """The thinking MODE string the paper's `thinking` column carries: the producer records
    ``{"mode": "off", "render": "native"}`` and the column carried that dict's repr — a filter on
    ``thinking == "off"`` returned 0 rows. The render detail is not in the 48-column schema."""
    if isinstance(value, Mapping):
        return str(value.get("mode") or C.THINKING_OFF)
    return str(value or C.THINKING_OFF)


def _num(x: Any, digits: int = 0) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int) or (isinstance(x, float) and digits == 0):
        return f"{int(round(x)):,}"
    return f"{x:,.{digits}f}"


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def _interval(pr: Mapping[str, Any]) -> str:
    if pr.get("rate") is None:
        return "n/a"
    return f"{_pct(pr['rate'])} (Wilson 95 % {_pct(pr['wilson_low'])} – {_pct(pr['wilson_high'])}), {pr['k']}/{pr['n']}"


def _parse_dt(value: Any) -> Optional[_dt.datetime]:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _dt.datetime.fromtimestamp(float(value), tz=_dt.timezone.utc)
    if not isinstance(value, str) or not value:
        return None
    try:
        d = _dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return d if d.tzinfo else d.replace(tzinfo=_dt.timezone.utc)


def _iso(d: Optional[_dt.datetime]) -> Optional[str]:
    return d.isoformat() if d is not None else None


def _mpl():
    try:
        import matplotlib
    except ImportError as e:    # pragma: no cover - venv-dependent
        raise AnalysisError("matplotlib is not importable in this venv — pass --no-figs or use the repo venv") from e
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "text.color": INK, "text.parse_math": False,
        "axes.labelcolor": INK2, "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
        "grid.color": GRID, "grid.linewidth": 0.7, "grid.linestyle": "-", "hatch.linewidth": 0.7,
        "svg.fonttype": "none", "pdf.fonttype": 42})
    return plt, Patch


def _recessive(ax, *, grid_axis: str = "x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(axis="both", length=0, pad=4)
    ax.grid(False)
    if grid_axis == "x":
        ax.xaxis.grid(True, zorder=0)
    elif grid_axis == "y":
        ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)


def _fit_header(fig, title: str, subtitle: Optional[str], *, bottom_in: float, left: float, right: float,
                hspace: Optional[float] = None, wspace: Optional[float] = None, extra_top_in: float = 0.0) -> None:
    """Size the axes box under the header the way ``finish_figure`` draws it
    (lifted from the retired tb_pilot_analysis: reserve the wrapped title + subtitle inches,
    grow the figure until the axes keep ≥ 80 % of the height)."""
    fig_w, fig_h = (float(v) for v in fig.get_size_inches())
    n_t = wrap_title(title, width=min(A.TITLE_WIDTH, int(fig_w * 10.5))).count("\n") + 1
    n_s = (wrap_title(subtitle, width=int(fig_w * 17), max_lines=A.SUBTITLE_MAX_LINES, what="subtitle").count("\n") + 1
           if subtitle else 0)
    header_in = A.HEADER_TOP_PAD_IN + n_t * A.TITLE_LINE_IN + n_s * A.SUBTITLE_LINE_IN + A.HEADER_GAP_IN + float(extra_top_in)
    need_h = (header_in + bottom_in) / (1.0 - A.AXES_MIN_FRACTION - 0.01)
    if fig_h < need_h:
        fig.set_size_inches(fig_w, need_h)
        fig_h = need_h
    kw = {"left": left, "right": right, "top": 1.0 - header_in / fig_h, "bottom": bottom_in / fig_h}
    if hspace is not None:
        kw["hspace"] = hspace
    if wspace is not None:
        kw["wspace"] = wspace
    fig.subplots_adjust(**kw)


def _bar_h(ax, y, w, color, height: float = 0.62):
    return ax.barh(y, w, height=height, color=color, edgecolor="none", linewidth=0, zorder=3)


# --------------------------------------------------------------------------- #
# the roster (Slice B's `configs/capeval/bcb_bfcl_200_widen_roster.yaml`)     #
# --------------------------------------------------------------------------- #


def load_roster(path: Union[str, Path]) -> Dict[str, Any]:
    """The roster yaml as a dict (Slice B's reader, `widen_roster.load_roster`)."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise AnalysisError(f"roster {p} does not exist (Slice B writes configs/capeval/bcb_bfcl_200_widen_roster.yaml)")
    doc = W.load_roster(p)
    if not isinstance(doc, Mapping) or not isinstance(doc.get("waves"), Mapping) or not doc["waves"]:
        raise AnalysisError(f"{p}: not a roster (no `waves` map)")
    return dict(doc)


def roster_sha256(path: Union[str, Path]) -> str:
    """sha256 of the roster FILE bytes — `widen_roster.roster_sha256`, what a launch-time snapshot records."""
    return W.roster_sha256(Path(path).expanduser())


_ROSTER_AT_COMMIT: Dict[Tuple[str, str, str], Tuple[Dict[str, Any], str]] = {}


def _git(args: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True)
    except OSError as e:        # pragma: no cover - git missing
        raise AnalysisError(f"git is not runnable ({e}) — the roster at the store's producer commit cannot be read") from e


def roster_at_producer_commit(roster_path: Union[str, Path], tree_sha: str, want_sha: str) -> Tuple[Dict[str, Any], str]:
    """The roster DOCUMENT the STORE'S PRODUCER COMMIT carried at ``roster_path``
    (``git show <producer.tree_sha>:<roster path>`` in the roster's repository — the
    launched tree IS a commit of it), required to hash to ``want_sha``, the sidecar's
    `roster_sha256`. This binds the label to the roster that PRODUCED the store: the
    working file, or any other revision of the file (a foreign branch carrying a
    relabelled roster), proves only that such a roster exists and is never consulted.
    Refuses when the roster is not in a git repository, the producer commit is unknown
    there, the commit carries no such file, or the blob's sha differs from the
    sidecar's (a sidecar regenerated from another roster). Returns ``(doc, source)``
    with ``source`` = ``git <tree_sha>:<rel>``."""
    p = Path(roster_path).expanduser().resolve()
    tree_sha, want_sha = str(tree_sha), str(want_sha)
    key = (str(p), tree_sha, want_sha)
    if key in _ROSTER_AT_COMMIT:
        return _ROSTER_AT_COMMIT[key]
    if not re.fullmatch(r"[0-9a-f]{40}", tree_sha):
        raise AnalysisError(f"producer tree sha {tree_sha!r} is not a 40-hex commit — the roster that produced the store cannot be located")
    top = _git(["rev-parse", "--show-toplevel"], cwd=p.parent)
    if top.returncode != 0:
        raise AnalysisError(f"{p.parent} is not in a git repository — the roster at the store's producer commit {tree_sha[:12]} cannot be "
                            "read (the lane keeps the roster in git; a sidecar's label is verified only against that commit's roster)")
    root = Path(top.stdout.decode().strip()).resolve()
    try:
        rel = p.relative_to(root).as_posix()
    except ValueError as e:
        raise AnalysisError(f"{p} is not under the git repository {root}") from e
    known = _git(["cat-file", "-e", f"{tree_sha}^{{commit}}"], cwd=root)
    if known.returncode != 0:
        raise AnalysisError(f"the store's producer commit {tree_sha[:12]}… is unknown to the roster's repository {root} (not fetched, or this "
                            "clone does not carry the development history that commit belongs to) — fetch it, or the store's generation "
                            "label cannot be re-derived")
    blob = _git(["show", f"{tree_sha}:{rel}"], cwd=root)
    if blob.returncode != 0:
        raise AnalysisError(f"the producer commit {tree_sha[:12]}… carries no {rel}: {blob.stderr.decode(errors='replace').strip()[-200:]}")
    have = hashlib.sha256(blob.stdout).hexdigest()
    if have != want_sha:
        raise AnalysisError(f"the roster at the store's producer commit {tree_sha[:12]}… ({rel}) hashes to {have[:12]}…, the sidecar names "
                            f"{want_sha[:12]}… — the sidecar was regenerated from another roster (an uncommitted edit, another branch); the "
                            "store's generation label cannot be re-derived")
    doc = yaml.safe_load(blob.stdout.decode("utf-8"))
    if not isinstance(doc, Mapping):
        raise AnalysisError(f"{rel} at {tree_sha[:12]}… is not a roster mapping")
    out = (dict(doc), f"git {tree_sha[:12]}:{rel}")
    _ROSTER_AT_COMMIT[key] = out
    return out


@dataclass
class Entry:
    entry_id: str
    cell_id: Optional[str]
    status: str
    paper_arm: Optional[str]
    kind: str                                   # base | prompted | fitted | matrix (queued, unkeyed)
    object: Optional[Dict[str, Any]]            # {staged_id, sha256, kind, sidecar, fit, object_generation, superseded_by_key}
    reason: Optional[str]
    unblock: Optional[str]
    note: Optional[str]
    rerun_trigger: Optional[str]
    pooling: Optional[Dict[str, Any]] = None    # {held_out, ruled_by, reason}: the roster's publication-pooling declaration (W.POOLING_KEY)

    @property
    def runnable(self) -> bool:
        return self.status == "runnable"

    @property
    def fitted(self) -> bool:
        return self.object is not None

    @property
    def object_generation(self) -> Optional[str]:
        """The CURRENT roster's label — used only to cross-check a store's sidecar, never as the store's label."""
        return (self.object or {}).get("object_generation")


@dataclass
class WaveSpec:
    name: str
    family: str
    lane_kind: str
    view: Optional[str]
    model_row: Optional[str]
    served: Dict[str, Any]                      # {hf, rev}
    objects_dir: Optional[str]
    legs: Dict[str, Dict[str, Any]]             # row task -> {status, reason, unblock}
    entries: List[Entry]                        # roster order, `none` first among the runnable
    prerequisites: List[str] = field(default_factory=list)      # the UNMET ones (`widen_roster.prerequisites`)
    sampling: Optional[str] = None
    sampling_clamp_override: Optional[Dict[str, Any]] = None
    object_generations: Tuple[str, ...] = OBJECT_GENERATIONS
    thinking: Optional[str] = None                              # the roster's declared thinking mode ("off")

    @property
    def runnable_entries(self) -> List[Entry]:
        run = [e for e in self.entries if e.runnable]
        return sorted(run, key=lambda e: (e.cell_id != BASE_CELL, self.entries.index(e)))

    @property
    def cells(self) -> List[str]:
        return [e.cell_id for e in self.runnable_entries]                     # type: ignore[misc]

    @property
    def steered_cells(self) -> List[str]:
        return [c for c in self.cells if c != BASE_CELL]

    @property
    def fitted_cells(self) -> List[str]:
        return [e.cell_id for e in self.runnable_entries if e.fitted]        # type: ignore[misc]

    @property
    def tasks(self) -> List[str]:
        return [t for t in TASKS if self.legs.get(t, {}).get("status") == "runnable"]

    @property
    def blocked_legs(self) -> Dict[str, Dict[str, Any]]:
        return {t: v for t, v in self.legs.items() if v.get("status") != "runnable"}

    @property
    def queued_or_blocked(self) -> List[Entry]:
        return [e for e in self.entries if not e.runnable]

    def entry_of_cell(self, cell: str) -> Entry:
        hits = [e for e in self.entries if e.cell_id == cell]
        if len(hits) != 1:
            raise AnalysisError(f"wave {self.name}: {len(hits)} roster entries name cell {cell!r}")
        return hits[0]


def _leg_block(status: str, detail: Mapping[str, Any], *, where: str) -> Dict[str, Any]:
    """`widen_roster.legs`' ``(status, detail)`` as the analysis's leg block."""
    if status not in ROSTER_STATUSES:
        raise AnalysisError(f"{where}: leg status {status!r} not in {list(ROSTER_STATUSES)} (a leg is a status string or a "
                            "{status, reason, unblock} map)")
    return {"status": status, "reason": detail.get("reason"), "unblock": detail.get("unblock")}


def wave_spec(roster: Mapping[str, Any], wave: str) -> WaveSpec:
    """The wave's block of the roster through Slice B's readers (`widen_roster.wave`
    / `entries` / `legs` / `paper_lane` / `object_generation` / `pin_of` /
    `prerequisites`), checked for what the analysis consumes: statuses, a runnable
    entry's non-null `cell_id`, every non-runnable entry's `reason` + `unblock`, an
    object entry's sha and generation vocabulary, a runnable steered entry implying
    a runnable `none`; the runnable order is `widen_roster.runnable_cells`' (none first)."""
    try:
        w = W.wave(roster, wave)
        lane = W.paper_lane(roster, wave)
        raw_entries = W.entries(roster, wave)
        raw_legs = W.legs(roster, wave)
        unmet = W.prerequisites(roster, wave)
    except W.RosterError as e:
        raise AnalysisError(f"wave {wave!r} is not in the roster: {e}") from e
    where = f"roster wave {wave}"
    for key in ("family", "lane_kind"):
        if not isinstance(lane.get(key), str) or not lane[key]:
            raise AnalysisError(f"{where}: `{key}` is required")
    excluded = [str(x) for x in (roster.get("excluded_families") or [])]
    if lane["family"] in excluded:
        raise AnalysisError(f"{where}: family {lane['family']!r} is excluded from the lane ({excluded})")
    gens = tuple(str(g) for g in (roster.get("object_generations") or OBJECT_GENERATIONS))
    legs = {TASK_OF_ATOM.get(atom, atom): _leg_block(status, detail, where=f"{where} legs.{atom}") for atom, (status, detail) in raw_legs.items()}
    for t in legs:
        if t not in TASKS:
            raise AnalysisError(f"{where}: leg {t!r} is not one of this lane's legs {list(ATOM_OF_TASK.values())}")
    entries: List[Entry] = []
    for eid, e in raw_entries.items():
        ew = f"{where} entries.{eid}"
        status = e.get("status")
        if status not in ROSTER_STATUSES:
            raise AnalysisError(f"{ew}: status {status!r} not in {list(ROSTER_STATUSES)}")
        cell = e.get("cell_id")
        if status == "runnable" and not isinstance(cell, str):
            raise AnalysisError(f"{ew}: a runnable entry needs a cell_id")
        if status != "runnable" and not (e.get("reason") and e.get("unblock")):
            raise AnalysisError(f"{ew}: a {status} entry needs a non-empty `reason` AND `unblock`")
        obj = dict(e["object"]) if e.get("object") else None
        if obj is not None:
            gen, sup = W.object_generation(roster, wave, eid)
            pin = W.pin_of(roster, wave, eid)
            if not re.fullmatch(r"[0-9a-f]{64}", str(pin or "")):
                raise AnalysisError(f"{ew}: object.sha256 must be a 64-hex sha")
            if gen not in gens:
                raise AnalysisError(f"{ew}: object_generation {gen!r} not in {list(gens)}")
            if gen == GENERATION_BATCH8 and not sup:
                raise AnalysisError(f"{ew}: a {GENERATION_BATCH8} object names its superseding key")
        kind = str(e.get("kind") or ("base" if cell == BASE_CELL else "fitted" if obj is not None else
                                     "prompted" if isinstance(cell, str) and cell.startswith("prompted__") else "unkeyed"))
        entries.append(Entry(entry_id=str(eid), cell_id=cell, status=status, paper_arm=e.get("paper_arm"), kind=kind, object=obj,
                             reason=e.get("reason"), unblock=e.get("unblock"), note=e.get("note"), rerun_trigger=e.get("rerun_trigger"), pooling=e.get(W.POOLING_KEY)))
    # the paper's lane vocabulary spells the transfer lane `base→instruct` (the first lane's rows); the yaml may write `->`
    lane_kind = str(lane["lane_kind"]).replace("->", "→")
    spec = WaveSpec(name=wave, family=lane["family"], lane_kind=lane_kind, view=lane.get("view"), model_row=w.get("model_row"),
                    served=dict(w.get("served") or {}), objects_dir=w.get("objects_dir"), legs=legs, entries=entries,
                    prerequisites=list(unmet), sampling=roster.get("sampling"),
                    sampling_clamp_override=(dict(roster["sampling_clamp_override"]) if roster.get("sampling_clamp_override") else None),
                    object_generations=gens, thinking=roster.get("thinking"))
    if spec.steered_cells and BASE_CELL not in spec.cells:
        # a steered-only wave names its baseline wave (`anchor_wave`); THIS reader pairs within one wave's
        # stores and does not resolve a cross-wave anchor yet — it refuses by name rather than mis-pair (the paired readout of such a
        # wave is the registry adapters' / the lane's L7 work, not this analysis)
        anchor = w.get("anchor_wave")
        if anchor:
            raise AnalysisError(f"{where}: runnable steered cells {spec.steered_cells} with no runnable `none` of their own — their baseline is "
                                f"wave {anchor!r} (anchor_wave); this analysis pairs within ONE wave's stores and does not read a cross-wave "
                                "anchor (D1859): not analysable here, by design")
        raise AnalysisError(f"{where}: runnable steered cells {spec.steered_cells} without a runnable `none`")
    if len(set(spec.cells)) != len(spec.cells):
        raise AnalysisError(f"{where}: a cell id appears twice among the runnable entries: {spec.cells}")
    if spec.cells != W.runnable_cells(roster, wave):
        raise AnalysisError(f"{where}: the analysis's runnable order {spec.cells} != widen_roster.runnable_cells {W.runnable_cells(roster, wave)}")
    return spec


def validate_roster(roster: Mapping[str, Any], roster_path: Union[str, Path]) -> Dict[str, Any]:
    """Slice B's `widen_roster.validate(roster, pins, models)` — a refusal there is a
    refusal here (the wave-level checks of :func:`wave_spec` run besides)."""
    pins_rel = str(roster.get("pins") or "configs/sdfbase_full22_pins.json")
    pins_path = Path(pins_rel).expanduser()
    if not pins_path.is_absolute():
        pins_path = REPO / pins_rel
    pins = read_json(pins_path) if pins_path.is_file() else {}
    try:
        problems = W.validate(roster, pins, C.MODELS)
    except Exception as e:      # noqa: BLE001 — B's refusal, named
        raise AnalysisError(f"roster {roster_path} fails widen_roster.validate: {type(e).__name__}: {e}") from e
    if problems:
        raise AnalysisError(f"roster {roster_path} fails widen_roster.validate ({len(problems)} problem(s)): " + "; ".join(str(p) for p in problems[:6]))
    return {"roster_validated_by": "sisteer.capeval.tinycap.widen_roster.validate", "note": None}


# --------------------------------------------------------------------------- #
# --pull                                                                       #
# --------------------------------------------------------------------------- #


def _volume_relative(full: str) -> str:
    """A container path under the volume mount as the path the ``modal volume``
    CLI takes (the mount stripped); a path outside the mount refuses."""
    mount = C.VOL_TINYCAP_ROOT.rsplit("/", 1)[0]
    if not full.startswith(mount + "/"):
        raise AnalysisError(f"{full!r} is not under the volume mount {mount!r}")
    return full[len(mount) + 1:]


#: the two serving engines a wave is generated on (`rung_probe.SERVING_ENGINE_*`); the pull's `--serving-engine` choices
SERVING_ENGINES: Tuple[str, ...] = (RP.SERVING_ENGINE_LENS, RP.SERVING_ENGINE_NATIVE)


def gen_root_for_engine(serving_engine: Optional[str]) -> str:
    """The volume's generation root a run on ``serving_engine`` wrote under — `contracts.gen_root_for_engine`, THE ONE
    derivation the launcher's out roots, the sandbox watcher and now the pull share: ``tinycap/gen`` for the lens (and
    ``None``, the lens by era — every historical address byte-identical), ``tinycap/gen/native`` for `lora_native`
    (without it the pull addressed the LENS namespace for every run, so a native `--pull` found no run or, when a lens
    run shared the suffix, pulled the LENS stores and certificates under the operator's native-run suffix). An unknown
    engine refuses by name."""
    try:
        return C.gen_root_for_engine(C.VOL_GEN_ROOT, serving_engine)
    except ValueError as e:
        raise AnalysisError(str(e)) from e


def remote_cell_path(suffix: str, cell: str, *, model_slug: str, mode: str = DEFAULT_MODE, serving_engine: Optional[str] = None) -> str:
    """``tinycap/gen[/native]/<model>/<cell>/thinking_<mode>/<suffix>`` — the store root INSIDE the volume, under the
    ENGINE's generation root (:func:`gen_root_for_engine`; the lens default byte-identical)."""
    return _volume_relative(f"{C.cell_out_dir(gen_root_for_engine(serving_engine), cell, mode, model_slug=model_slug)}/{suffix.strip('/')}")


def remote_ladder_dir(suffix: str, *, model_slug: str, serving_engine: Optional[str] = None) -> str:
    """``tinycap/gen[/native]/<model>/_ladder/<suffix>`` (`rung_probe.ladder_dir` under the ENGINE's generation root) — the
    per-cell certificates `rung_probe.<cell>.json`, the wave record `rung_admission.json` and the legacy single
    `rung_probe.json`; a native ladder files its records in the native namespace (`tinycap_served.ladder_gen_root`)."""
    try:
        return _volume_relative(RP.ladder_dir(gen_root_for_engine(serving_engine), model_slug, suffix))
    except RP.LadderError as e:
        raise AnalysisError(str(e)) from e


#: beside every staged object on the volume: the staging script's record (`stage_full22.sh` / `80_stage_wave.sh`) — the
#: `staged_id` (the cluster A / the grading host staging name the roster's `object.staged_id` carries), `cell_id`, `object_sha256`,
#: `staged_shas` (sha256 per staged file), `source`, `staged_by`
STAGED_FROM_FILENAME = "staged_from.json"


def remote_sidecar_path(objects_dir: str, cell_id: str, filename: str) -> str:
    """``objects/sdfbase/full22/<model>/<view>/<CELL ID>/<filename>`` — the volume's view layout keys each staged object by
    its BARE cell id (`constv2-mix-full__lora_r4_ce_minus_ctl`), never by the roster's `staged_id` (`derived__…`,
    `derived_itdoc__…`, `itdoc__…`), which is the staging name recorded INSIDE the dir's staged_from.json (round 6:
    composing the path from the staged_id refused every pull of a fitted cell as INCOMPLETE)."""
    return _volume_relative(f"{objects_dir.rstrip('/')}/{cell_id}/{filename}")


def staged_from_problems(doc: Any, entry: Entry, *, sidecar: str, sidecar_sha256: Optional[str]) -> List[str]:
    """What the volume dir's staged_from.json must agree with: the roster's `staged_id` (exact), the entry's cell id,
    the roster's object sha256, and — when the record lists the sidecar's sha256 under `staged_shas` — the bytes of the
    sidecar that landed. Every disagreement is named; [] means the dir IS the object the roster stages."""
    obj = entry.object or {}
    where = f"{entry.cell_id}: {STAGED_FROM_FILENAME}"
    if not isinstance(doc, Mapping):
        return [f"{where} is not a JSON object"]
    problems: List[str] = []
    staged_id = doc.get("staged_id")
    if not staged_id:
        problems.append(f"{where} names no staged_id (keys {sorted(doc)[:8]})")
    elif str(staged_id) != str(obj.get("staged_id")):
        problems.append(f"{where} staged_id {staged_id!r} != the roster's {obj.get('staged_id')!r} — the volume dir holds another staging")
    if doc.get("cell_id") is not None and str(doc["cell_id"]) != str(entry.cell_id):
        problems.append(f"{where} cell_id {doc['cell_id']!r} != the entry's {entry.cell_id!r}")
    if doc.get("object_sha256") is not None and str(doc["object_sha256"]) != str(obj.get("sha256")):
        problems.append(f"{where} object_sha256 {str(doc['object_sha256'])[:12]}… != the roster's {str(obj.get('sha256'))[:12]}…")
    want = (doc.get("staged_shas") or {}).get(sidecar) if isinstance(doc.get("staged_shas"), Mapping) else None
    if want is not None and sidecar_sha256 is not None and str(want) != sidecar_sha256:
        problems.append(f"{where} staged_shas[{sidecar}] {str(want)[:12]}… != the sha256 of the {sidecar} that landed {sidecar_sha256[:12]}…")
    return problems


def sidecar_filename(entry: Entry) -> str:
    obj = entry.object or {}
    return str(obj.get("sidecar") or DEFAULT_SIDECAR_OF_KIND.get(str(obj.get("kind")), DEFAULT_SIDECAR))


def _modal_get(modal_python: str, volume: str, remote: str, dest: Path) -> subprocess.CompletedProcess:
    argv = [modal_python, "-m", "modal", "volume", "get", "--force", volume, remote, str(dest)]
    return subprocess.run(argv, capture_output=True, text=True)


def pull_suffix(suffix: str, wave: WaveSpec, *, dest_root: Union[str, Path] = DEFAULT_PULL_ROOT, volume: str = DEFAULT_VOLUME,
                modal_python: str = DEFAULT_MODAL_PYTHON, mode: str = DEFAULT_MODE, log: Callable[[str], None] = print,
                serving_engine: Optional[str] = None) -> Path:
    """Every runnable cell's store root for ``suffix`` → ``<dest_root>/<suffix>/<cell>/``
    (verified against the volume listing at the root and under ``cells/``), the
    ladder dir → ``_ladder/`` when it exists, and every fitted cell's object
    SIDECAR → ``_objects/<cell>/<sidecar>`` (the fit layers are read from it).
    The Modal token reaches the CLI through the environment.

    ``serving_engine``: the engine the run was generated on — the store roots and the
    ladder dir are addressed under THAT engine's generation root
    (:func:`gen_root_for_engine`): ``lora_native`` reads ``tinycap/gen/native/…``; the lens (the
    default, ``None``) the historical addresses, byte-identical. The object sidecars live under
    the roster's ``objects_dir`` on either engine."""
    if wave.model_row is None:
        raise AnalysisError(f"wave {wave.name} has no model_row — nothing to pull")
    gen_root_for_engine(serving_engine)                  # an unknown engine refuses before anything is fetched
    modal_python = str(Path(modal_python).expanduser())
    if not Path(modal_python).is_file():
        raise AnalysisError(f"--modal-python {modal_python} does not exist (the pull shells out to the modal venv's CLI)")
    from sisteer import site as _site
    dest = _site.expand_path(dest_root, key="--pull-root") / suffix
    dest.mkdir(parents=True, exist_ok=True)
    problems: List[str] = []
    n_files = 0
    if serving_engine == RP.SERVING_ENGINE_NATIVE:
        log(f"[{LANE}] pull {suffix}: the {serving_engine} engine's namespace — store roots and the ladder dir under "
            f"{_volume_relative(gen_root_for_engine(serving_engine))}/")

    def _pull_cell(cell: str) -> Tuple[List[str], int]:
        """One cell's store root: fetched into its own scratch dir, moved into place, verified against the listing."""
        remote = remote_cell_path(suffix, cell, model_slug=wave.model_row, mode=mode, serving_engine=serving_engine)
        scratch = dest / ".pull" / cell
        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir(parents=True)
        log(f"[{LANE}] pull {cell}: modal volume get --force {volume} {remote} {scratch}")
        proc = _modal_get(modal_python, volume, remote, scratch)
        if proc.returncode != 0:
            raise AnalysisError(f"modal volume get failed for {cell} (rc {proc.returncode}): {proc.stderr.strip()[-800:]}")
        landed = scratch / Path(remote).name
        if not landed.is_dir():
            raise AnalysisError(f"pull of {cell} finished but {landed} does not exist (check the remote path {remote!r})")
        target = dest / cell
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(landed), str(target))
        shutil.rmtree(scratch, ignore_errors=True)
        found: List[str] = []
        for name in modal_volume_ls(modal_python, volume, remote):
            if not (target / name).exists():
                found.append(f"{cell}: root entry {name!r} listed on the volume but absent locally")
        if (target / "cells").is_dir():
            for key in modal_volume_ls(modal_python, volume, f"{remote}/cells"):
                if not (target / "cells" / key).exists():
                    found.append(f"{cell}: store {key!r} listed on the volume but absent locally")
        return found, sum(1 for p in target.rglob("*") if p.is_file())

    # the cells' pulls are independent (own scratch dir, own target): fanned out over a thread pool, gathered in roster order
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(8, len(wave.cells)))) as pool:
        for found, n in pool.map(_pull_cell, list(wave.cells)):
            problems.extend(found)
            n_files += n
    shutil.rmtree(dest / ".pull", ignore_errors=True)
    lad_remote = remote_ladder_dir(suffix, model_slug=wave.model_row, serving_engine=serving_engine)
    scratch = dest / ".pull_ladder"
    scratch.mkdir(exist_ok=True)
    proc = _modal_get(modal_python, volume, lad_remote, scratch)
    landed = scratch / Path(lad_remote).name
    if proc.returncode == 0 and landed.is_dir():
        target = dest / LADDER_SUBDIR
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(landed), str(target))
        log(f"[{LANE}] ladder dir → {target} ({sorted(p.name for p in target.iterdir())})")
    else:
        (dest / LADDER_SUBDIR).mkdir(exist_ok=True)
        log(f"[{LANE}] no ladder dir at {lad_remote} (a rung-1 suffix has none)")
    shutil.rmtree(scratch, ignore_errors=True)
    for entry in wave.runnable_entries:
        if not entry.fitted or wave.objects_dir is None:
            continue
        # the object's dir on the volume is keyed by the BARE cell id; the roster's staged_id is checked against the dir's own record
        side = sidecar_filename(entry)
        target = dest / OBJECTS_SUBDIR / str(entry.cell_id)
        target.mkdir(parents=True, exist_ok=True)
        landed_all = True
        for filename in (side, STAGED_FROM_FILENAME):
            remote = remote_sidecar_path(wave.objects_dir, str(entry.cell_id), filename)
            proc = _modal_get(modal_python, volume, remote, target)
            if proc.returncode != 0 or not (target / filename).is_file():
                landed_all = False
                problems.append(f"{entry.cell_id}: {'object sidecar' if filename == side else 'staging record'} {remote} did not land "
                                f"(rc {proc.returncode}): {proc.stderr.strip()[-300:]}")
        if not landed_all:
            continue
        try:
            record = read_json(target / STAGED_FROM_FILENAME)
        except Exception as e:                                   # noqa: BLE001 - the record's unreadability is the finding
            problems.append(f"{entry.cell_id}: {STAGED_FROM_FILENAME} is unreadable ({type(e).__name__}: {e})")
            continue
        found = staged_from_problems(record, entry, sidecar=side, sidecar_sha256=hashlib.sha256((target / side).read_bytes()).hexdigest())
        if found:
            problems.extend(found)
        else:
            log(f"[{LANE}] sidecar {entry.cell_id} → {target / side} (staged_id {entry.object['staged_id']} = the volume's {STAGED_FROM_FILENAME}; "
                f"object sha256 {str(entry.object['sha256'])[:12]}… agrees)")
    if problems:
        raise AnalysisError(f"pull of {suffix} is INCOMPLETE ({n_files} files landed): " + "; ".join(problems))
    log(f"[{LANE}] pulled {n_files} files to {dest} (verified against the volume listing)")
    return dest


# --------------------------------------------------------------------------- #
# --analyze: loading                                                           #
# --------------------------------------------------------------------------- #


def load_lib_families(repo_root: Union[str, Path]) -> Dict[str, str]:
    """`configs/capeval/tasks/bcb_lib_families.json` — a TOTAL ``lib -> family`` map, through the
    contracts' ONE reader (`contracts.bcb_lib_families`: schema, task, a non-empty map of strings)."""
    try:
        return C.bcb_lib_families(repo_root)
    except C.SubsetPinError as e:
        raise AnalysisError(f"{Path(repo_root) / LIB_FAMILIES_REL}: {e}") from e


_MISSING = object()
_dig = L._dig


def _dig_present(d: Mapping[str, Any], path: Sequence[str]) -> Any:
    """Like :func:`_dig`, but an ABSENT key (at any level) is :data:`_MISSING`,
    distinct from a key PRESENT with a null value."""
    cur: Any = d
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return _MISSING
        cur = cur[key]
    return cur


def _canon(v: Any) -> str:
    return json.dumps(v, sort_keys=True, default=str)


def task_identity_paths(task: str) -> Tuple[Tuple[str, ...], ...]:
    """The hashed `tinycap` keys a run that files ``task`` records BEYOND the
    unconditional set (`contracts.tinycap_identity_fields` minus `TINYCAP_IDENTITY_FIELDS`)."""
    fields = C.tinycap_identity_fields(files_bigcodebench_200=(task == BCB_TASK), files_bfcl_200=(task == BFCL_TASK))
    return tuple(("tinycap", f) for f in fields if f not in C.TINYCAP_IDENTITY_FIELDS)


def intended_instrument(task: str, pin: Mapping[str, Any]) -> Dict[str, Any]:
    """What THIS lane's stores must record under the leg-specific hashed keys."""
    if task == BCB_TASK:
        return {C.IDENTITY_KEY_BCB_ITEM_SET: C.item_set_identity(pin), C.IDENTITY_KEY_SANDBOX: C.bcb_sandbox_identity()}
    if task == BFCL_TASK:
        return {C.IDENTITY_KEY_BFCL_ITEM_SET: C.item_set_identity(pin)}
    raise AnalysisError(f"{task!r} is not a leg of this lane")


def sampling_mode_of(manifest: Mapping[str, Any]) -> str:
    mode = _dig(manifest, (C.MANIFEST_KEY_SAMPLING, "mode"))
    if not isinstance(mode, str) or not mode:
        raise AnalysisError("manifest carries no `sampling.mode` — a store that cannot say how its rows were drawn")
    return mode


#: the projection KIND of each block `tinycap_ladder.project_identity` knows (`PROJECTION_KINDS`)
PROJECTION_KIND_OF_BLOCK: Mapping[str, str] = {C.MANIFEST_KEY_TINYCAP: "identity", C.MANIFEST_KEY_SAMPLING: C.MANIFEST_KEY_SAMPLING,
                                                C.MANIFEST_KEY_THINKING: C.MANIFEST_KEY_THINKING}


def project(block: Mapping[str, Any], retained_tasks: Sequence[str], *, mode: str, where: str, kind: str = "identity") -> Any:
    """`tinycap_ladder.project_identity(block, retained_tasks, kind=, mode=)` — Slice B's ONE
    grouped projection rule set (scalars stripped of every task-map / provenance
    field; task maps on the retained tasks; per-task provenance; optional keys
    conditional on the mode). The analysis compares PROJECTIONS, never blocks,
    and has no local re-implementation (a rename of the ladder's function fails here at call time, loudly)."""
    fn = L.project_identity          # resolved outside the refusal wrapper: a missing function is a broken tree, not a projection refusal
    try:
        return fn(dict(block or {}), list(retained_tasks), kind=kind, mode=mode)
    except Exception as e:      # noqa: BLE001 — a projection refusal (a retained task missing, a protocol/card mix) is named
        raise AnalysisError(f"{where}: identity projection refused: {type(e).__name__}: {e}") from e


def _projection_value(p: Any, name: str) -> Any:
    """The value a `tinycap_ladder.projection_differences` key names inside a projection: a
    scalar by key, a task map entry (``key[task]``), a per-task provenance (``key[task].provenance``)
    or the optional-key set (``<optional keys>``) — so a refusal message carries the differing values."""
    if not isinstance(p, Mapping):
        return None
    if name == "<optional keys>":
        return p.get("optional")
    m = re.fullmatch(r"(?P<key>[^\[\]]+)\[(?P<task>[^\[\]]+)\](?P<prov>\.provenance)?", name)
    if m:
        group = "provenance" if m.group("prov") else "maps"
        return ((p.get(group) or {}).get(m.group("key")) or {}).get(m.group("task"))
    if name in (p.get("scalars") or {}):
        return p["scalars"][name]
    return (p.get("maps") or {}).get(name)


def _projection_diff(pa: Any, pb: Any) -> List[str]:
    """Named differences between two projections: `tinycap_ladder.projection_differences`
    (scalars key by key, the optional-key set, each map per retained task, each
    provenance per task); a recursive key diff for anything that is not a projection."""
    if isinstance(pa, Mapping) and isinstance(pb, Mapping) and "kind" in pa and "kind" in pb:
        try:
            return sorted(L.projection_differences(pa, pb))
        except Exception as e:      # noqa: BLE001 — incomparable projections are a named difference
            return [f"<incomparable: {type(e).__name__}: {e}>"]
    if isinstance(pa, Mapping) and isinstance(pb, Mapping):
        out = []
        for k in sorted(set(pa) | set(pb)):
            va, vb = pa.get(k, _MISSING), pb.get(k, _MISSING)
            if va is _MISSING or vb is _MISSING:
                out.append(str(k))
            elif isinstance(va, Mapping) and isinstance(vb, Mapping):
                out.extend(f"{k}.{sub}" for sub in _projection_diff(va, vb))
            elif _canon(va) != _canon(vb):
                out.append(str(k))
        return out
    return [] if _canon(pa) == _canon(pb) else ["value"]


def store_object_sha256(manifest: Mapping[str, Any], *, cell: str) -> Tuple[Optional[str], Optional[str]]:
    """``(sha, source)`` of the object the store SERVED — the ladder's ONE reader of
    where a producer records it (`tinycap_ladder.served_object_sha256`, the sources of
    `tinycap_ladder.OBJECT_SHA_SOURCES` in order; both null → (None, None), an
    object-free cell), its disagreement refusal re-raised as the analysis's."""
    try:
        return L.served_object_sha256(manifest, cell=cell)
    except L.LadderError as e:
        raise AnalysisError(str(e)) from e


def snapshot_digest(doc: Mapping[str, Any]) -> str:
    """sha256 of the canonical JSON of every field but `snapshot_sha256` — Slice B's
    `widen_roster.snapshot_digest` (the launcher's writer), so writer and reader share ONE rule."""
    return W.snapshot_digest(dict(doc))


def load_lane_roster_sidecar(cell_root: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """`<store root>/lane_roster.json` (the launch-time roster snapshot) when present,
    held to the writer's own rules (`widen_roster.snapshot_problems`: exact key set,
    a recomputing `snapshot_sha256`, the declared vocabulary, a batch-8 label naming
    its superseding key, generation and object sha both null or both set). None when
    absent (the caller decides whether absence is a refusal: it is for a fitted cell)."""
    p = Path(cell_root) / LANE_ROSTER_SIDECAR
    if not p.is_file():
        return None
    doc = read_json(p)
    problems = W.snapshot_problems(doc)
    if problems:
        raise AnalysisError(f"{p}: not a valid launch-time snapshot — " + "; ".join(problems)
                            + "; the store's generation label cannot be trusted")
    return dict(doc)


def rederive_snapshot(side: Mapping[str, Any], *, roster_path: Union[str, Path], wave: str, tree_sha: str) -> Tuple[Dict[str, Any], str]:
    """The sidecar's block RE-DERIVED from the roster the STORE'S PRODUCER COMMIT
    carried (the label is never trusted, and a roster that merely exists somewhere
    proves nothing): `widen_roster.snapshot` over
    :func:`roster_at_producer_commit` (``git show <producer.tree_sha>:<roster>``,
    hashed to the sidecar's `roster_sha256`), for the sidecar's wave / entry / launch
    time. Every content field must come out equal — a relabelled sidecar whose
    digest was recomputed still disagrees with the roster that produced the store.
    Returns ``(rebuilt block, roster source)``; refuses when the entry was not
    runnable in that roster (nothing could have launched it)."""
    at, source = roster_at_producer_commit(roster_path, tree_sha, str(side["roster_sha256"]))
    try:
        rebuilt = W.snapshot(at, wave, str(side["entry_id"]), roster_sha256=str(side["roster_sha256"]), launched_at=str(side["launched_at"]))
    except W.RosterError as e:
        raise AnalysisError(f"the sidecar names entry {side.get('entry_id')!r} of wave {wave!r}, which the roster at "
                            f"{str(side['roster_sha256'])[:12]}… ({source}) could not have launched: {e}") from e
    bad = {k: (side.get(k), rebuilt.get(k)) for k in LANE_ROSTER_FIELDS if side.get(k) != rebuilt.get(k)}
    if bad:
        raise AnalysisError(f"the sidecar's label disagrees with the roster at the producer commit ({source}, sha {str(side['roster_sha256'])[:12]}…) — "
                            + "; ".join(f"{k}: sidecar {a!r} vs roster {b!r}" for k, (a, b) in bad.items())
                            + " — the sidecar was rewritten after the launch (a recomputed digest does not make a label true)")
    return rebuilt, source


def generation_record(cell_root: Union[str, Path], *, cell: str, entry: Entry, wave: WaveSpec, manifest: Mapping[str, Any],
                      roster_sha: Optional[str], roster_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """The generation record for one store: for a FITTED cell the sidecar is REQUIRED
    and bound — wave, cell, entry, the served object sha (the manifest's) — and its
    generation label is THE label ONLY after it is re-derived from the roster at the
    sidecar's own roster sha (:func:`rederive_snapshot`; never the current roster,
    which is only cross-checked and its drift reported). Object-free baselines
    (`none`, prompted) are generation-NEUTRAL (`object_generation` None); a sidecar
    on them, when written, is validated and re-derived the same way. A root whose
    worker filed a sampling-validation FAILURE is refused before anything else."""
    # bcb-bfcl-200-widen code review: the ONE root-admissibility rule the ladder's store
    # loader and `merge` share (`tinycap_ladder.root_admissibility_reason`) — a root whose worker filed a sampling-
    # validation failure is NEVER protocol evidence, whatever its DONE markers say
    reason = L.root_admissibility_reason(cell_root)
    if reason:
        raise AnalysisError(reason)
    tree_sha = _dig(manifest, (C.MANIFEST_KEY_PRODUCER, "tree_sha"))
    if not isinstance(tree_sha, str) or not re.fullmatch(r"[0-9a-f]{40}", tree_sha):
        raise AnalysisError(f"{cell_root}: the manifest's producer.tree_sha {tree_sha!r} is not a 40-hex commit — the roster that "
                            "launched this store cannot be located")
    side = load_lane_roster_sidecar(cell_root)
    served_sha, sha_source = store_object_sha256(manifest, cell=cell)
    out: Dict[str, Any] = {"cell": cell, "entry_id": entry.entry_id, "fitted": entry.fitted, "object_sha256": served_sha,
                           "object_sha256_source": sha_source, "sidecar": (str(Path(cell_root) / LANE_ROSTER_SIDECAR) if side else None),
                           "object_generation": None, "superseded_by_key": None, "provisional": False, "roster_object_drift": False,
                           "roster_sha256_at_launch": None, "launched_at": None, "roster_at_launch_source": None}
    if entry.fitted and served_sha is None:
        raise AnalysisError(f"{cell_root}: a fitted cell whose manifest records no served object sha (route digests and "
                            "provenance.artifact_sha256 all null)")
    if not entry.fitted and served_sha is not None:
        raise AnalysisError(f"{cell_root}: the roster lists {cell!r} as object-free but the store served object {served_sha[:12]}…")
    if side is None:
        if entry.fitted:
            raise AnalysisError(f"{cell_root}: fitted cell {cell!r} has no {LANE_ROSTER_SIDECAR} — the launch-time generation "
                                "label is missing; the analysis never labels a store from the current roster")
        return out
    for key, want in (("wave", wave.name), ("cell_id", cell)):
        if side[key] != want:
            raise AnalysisError(f"{cell_root}/{LANE_ROSTER_SIDECAR}: {key} {side[key]!r} != {want!r} — a snapshot from another launch")
    if side["entry_id"] != entry.entry_id:
        raise AnalysisError(f"{cell_root}/{LANE_ROSTER_SIDECAR}: entry_id {side['entry_id']!r} != the roster's {entry.entry_id!r} for {cell!r}")
    if side["object_sha256"] != served_sha:
        raise AnalysisError(f"{cell_root}/{LANE_ROSTER_SIDECAR}: object_sha256 {str(side['object_sha256'])[:12]}… != the object the store "
                            f"served {str(served_sha)[:12]}… ({sha_source}) — the snapshot does not describe this store's object")
    if entry.fitted and side["object_generation"] is None:
        raise AnalysisError(f"{cell_root}/{LANE_ROSTER_SIDECAR}: a fitted cell's snapshot carries no object_generation")
    if roster_path is None:
        raise AnalysisError(f"{cell_root}/{LANE_ROSTER_SIDECAR}: the generation label is re-derived from the roster at the sidecar's "
                            "roster_sha256 — the analysis needs the roster path (a label is never taken on the sidecar's word)")
    try:
        _, at_source = rederive_snapshot(side, roster_path=roster_path, wave=wave.name, tree_sha=tree_sha)
    except AnalysisError as e:
        raise AnalysisError(f"{cell_root}/{LANE_ROSTER_SIDECAR}: {e}") from e
    roster_sha_now = (entry.object or {}).get("sha256")
    out.update(object_generation=side["object_generation"], superseded_by_key=side.get("superseded_by_key"),
               provisional=(side["object_generation"] == GENERATION_BATCH8), roster_sha256_at_launch=side["roster_sha256"],
               launched_at=side["launched_at"], roster_at_launch_source=at_source,
               roster_object_drift=bool(entry.fitted and roster_sha_now is not None and roster_sha_now != served_sha),
               roster_moved_since_launch=bool(roster_sha is not None and side["roster_sha256"] != roster_sha))
    return out


def check_pairing_generations(base_rec: Mapping[str, Any], steered_rec: Mapping[str, Any]) -> None:
    """The generation rule at the write moment of a paired contrast: an object-free side
    is neutral; two FITTED sides must share a generation (never computed by
    design — steered is paired with `none` only — the guard is for a future
    cross-object mode and for a roster that mislabels `none`)."""
    ga, gb = base_rec.get("object_generation"), steered_rec.get("object_generation")
    if base_rec.get("fitted") and steered_rec.get("fitted") and ga != gb:
        raise AnalysisError(f"pairing {base_rec['cell']!r} ({ga}) with {steered_rec['cell']!r} ({gb}): two fitted cells of different "
                            "object generations are never contrasted (D7: a batch-8 vs batch-64 contrast is a learning-rate contrast)")


def _layers_text(layers: Sequence[int]) -> str:
    ls = sorted(int(x) for x in layers)
    if ls and ls == list(range(ls[0], ls[-1] + 1)) and len(ls) > 2:
        return f"{ls[0]}-{ls[-1]}"
    return ",".join(str(x) for x in ls)


def _declared_layers(fit: Mapping[str, Any]) -> Optional[List[int]]:
    v = fit.get("layers")
    if v is None:
        return None
    if isinstance(v, str):
        m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", v)
        if m:
            return list(range(int(m.group(1)), int(m.group(2)) + 1))
        return [int(x) for x in v.split(",") if x.strip()]
    return [int(x) for x in v]


def fit_layers_of(rung1: Union[str, Path], entry: Entry, *, served_sha: Optional[str]) -> Dict[str, Any]:
    """The fitted object's layer SET from its OWN sidecar under
    ``<rung dir>/_objects/<cell>/<sidecar>`` (`derived_meta.json` `layers` or
    `train_budget_meta.json` `target_layers`), cross-checked against the
    sidecar's own `object_sha256` when it carries one and against the roster's
    `fit.layers` declaration when given. A fitted cell without its sidecar refuses."""
    side = sidecar_filename(entry)
    p = Path(rung1) / OBJECTS_SUBDIR / str(entry.cell_id) / side
    if not p.is_file():
        raise AnalysisError(f"{p} is missing — the fit layers are read per cell from the object's sidecar (D4); --pull fetches "
                            f"<objects_dir>/{entry.cell_id}/{side} (the volume's cell-id dir; the roster's staged_id "
                            f"{(entry.object or {}).get('staged_id')} is checked against its {STAGED_FROM_FILENAME})")
    doc = read_json(p)
    if not isinstance(doc, Mapping):
        raise AnalysisError(f"{p}: not an object sidecar")
    # the staging record pulled beside the sidecar (when present): the roster's staged_id must be the one the volume dir records
    staged_from, staged_id_volume = p.with_name(STAGED_FROM_FILENAME), None
    if staged_from.is_file():
        rec = read_json(staged_from)
        found = staged_from_problems(rec, entry, sidecar=side, sidecar_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        if found:
            raise AnalysisError(f"{staged_from}: " + "; ".join(found))
        staged_id_volume = str(rec.get("staged_id"))
    key = next((k for k in SIDECAR_LAYER_KEYS if isinstance(doc.get(k), list) and doc[k]), None)
    if key is None:
        raise AnalysisError(f"{p}: no layer set under {list(SIDECAR_LAYER_KEYS)}")
    layers = sorted(int(x) for x in doc[key])
    side_sha = doc.get("object_sha256")
    if isinstance(side_sha, str) and served_sha and side_sha != served_sha:
        raise AnalysisError(f"{p}: the sidecar's object_sha256 {side_sha[:12]}… != the object the store served {served_sha[:12]}… — "
                            "the sidecar is another object's")
    declared = _declared_layers((entry.object or {}).get("fit") or {})
    if declared is not None and sorted(declared) != layers:
        raise AnalysisError(f"{p}: layers {_layers_text(layers)} != the roster's declaration {_layers_text(declared)} for {entry.cell_id}")
    return {"fit_layers": _layers_text(layers), "fit_layer_count": len(layers), "layers": layers, "sidecar": str(p), "layer_key": key,
            "fit_model_id": doc.get("model_id"), "fit_model_key": doc.get("model_key"), "positions": doc.get("positions_at_fit", doc.get("positions")),
            "staged_id": (entry.object or {}).get("staged_id"), "staged_id_volume": staged_id_volume,
            "staged_from": (str(staged_from) if staged_id_volume else None)}


# --------------------------------------------------------------------------- #
# --analyze: identity                                                          #
# --------------------------------------------------------------------------- #


def identity_check(manifests: Mapping[str, Mapping[str, Any]], *, task: str, pin: Mapping[str, Any], mode: str,
                   base: str = BASE_CELL) -> Dict[str, Any]:
    """Per leg, every steered cell against ``base`` (two instruments do not
    pair): the hashed `tinycap` block PROJECTED onto this leg (scalars stripped of
    every task-map / provenance field, task maps on this task, optional keys
    conditional on the mode — `tinycap_ladder.project_identity`), the run-level
    scalars (model, revision, engine, tree), the `sampling` / `thinking` blocks
    projected the same way; the leg-specific hashed keys (derived from the
    contracts) are part of the projected scalars. The MAY-DIFFER intervention is
    reported. Every cell's leg-specific values are then held to the lane's
    intended instrument (the pin the analysis joins on, the contracts' sandbox facts)."""
    if base not in manifests:
        raise AnalysisError(f"identity: base cell {base!r} not among {sorted(manifests)}")
    for cell, m in manifests.items():
        if not isinstance(m.get(C.MANIFEST_KEY_TINYCAP), Mapping):
            raise AnalysisError(f"identity: cell {cell!r} manifest carries no `{C.MANIFEST_KEY_TINYCAP}` block")
    other_leg_fields = {f for t, fields in L.LEG_IDENTITY_FIELDS.items() if t != task for f in fields}

    def _tinycap_for_leg(m: Mapping[str, Any]) -> Dict[str, Any]:
        """The hashed block as THIS leg sees it: the other legs' hashed keys and `tasks` follow what else a store
        files (a two-leg store beside a one-leg store is the same instrument on the shared leg); the served
        tool-call facts are BFCL's instrument and are compared on that leg only."""
        tc = m[C.MANIFEST_KEY_TINYCAP]
        out = {k: v for k, v in tc.items() if k not in other_leg_fields and k != "tasks"}
        if task != BFCL_TASK:
            out["served"] = {k: v for k, v in (tc.get("served") or {}).items() if k not in L.LEG_DERIVED_SERVED_KEYS}
        return out

    mb = manifests[base]
    pb_tc = project(_tinycap_for_leg(mb), [task], mode=mode, where=f"{base}/{task} tinycap")
    pb_blocks = {blk: project(mb.get(blk) or {}, [task], mode=mode, where=f"{base}/{task} {blk}", kind=PROJECTION_KIND_OF_BLOCK[blk])
                 for blk in PROJECTED_BLOCKS}
    mismatched: Dict[str, Dict[str, Any]] = {}
    differs: Dict[str, Dict[str, Any]] = {}
    same_optional: List[str] = []
    equal_scalars: List[str] = []
    for cell, m in manifests.items():
        if cell == base:
            continue
        try:
            pc = project(_tinycap_for_leg(m), [task], mode=mode, where=f"{cell}/{task} tinycap")
            for name in _projection_diff(pb_tc, pc):
                mismatched[f"{cell}:tinycap.{name}"] = {"base": _projection_value(pb_tc, name), "cell": _projection_value(pc, name)}
            for blk, pb in pb_blocks.items():
                pcb = project(m.get(blk) or {}, [task], mode=mode, where=f"{cell}/{task} {blk}", kind=PROJECTION_KIND_OF_BLOCK[blk])
                for name in _projection_diff(pb, pcb):
                    mismatched[f"{cell}:{blk}.{name}"] = {"base": _projection_value(pb, name), "cell": _projection_value(pcb, name)}
        except AnalysisError as e:
            mismatched[f"{cell}:<projection>"] = {"base": None, "cell": str(e)}
        for path in RUN_SCALAR_PATHS:
            va, vb = _dig_present(mb, path), _dig_present(m, path)
            name = ".".join(path)
            if va is _MISSING or vb is _MISSING or va != vb:
                mismatched[f"{cell}:{name}"] = {"base": (None if va is _MISSING else va), "cell": (None if vb is _MISSING else vb),
                                                "absent": [side for side, v in (("base", va), ("cell", vb)) if v is _MISSING]}
            else:
                equal_scalars.append(name)
        for path in IDENTITY_MAY_DIFFER:
            va, vb = _dig(mb, path), _dig(m, path)
            name = ".".join(path)
            if _canon(va) == _canon(vb):
                same_optional.append(name)
            else:
                differs[f"{cell}:{name}"] = {"base": va, "cell": vb}
    intended = intended_instrument(task, pin)
    off_instrument: Dict[str, Dict[str, Any]] = {}
    for cell, m in manifests.items():
        for key, want in intended.items():
            have = _dig_present(m, (C.MANIFEST_KEY_TINYCAP, key))
            if have is _MISSING or have != want:
                off_instrument[f"{cell}:tinycap.{key}"] = {"store": (None if have is _MISSING else have), "intended": want}
    return {"task": task, "mode": mode, "mismatched_required": mismatched, "differs": differs,
            "required_equal": sorted(set(equal_scalars) | {"tinycap (projected)", *(f"{b} (projected)" for b in PROJECTED_BLOCKS)}),
            "optional_equal": sorted(set(same_optional)), "same_instrument": not mismatched,
            "task_identity_paths": [".".join(p) for p in task_identity_paths(task)],
            "intended_instrument": intended, "off_instrument": off_instrument, "base": base, "cells": sorted(manifests)}


#: what the analysis reads off a launcher attempt pointer (a subset of `tinycap_served.POINTER_FIELDS`)
POINTER_FIELDS_READ: Tuple[str, ...] = ("run_id", "cell", "function_call_id", "mode", "out_dir", "controls", "tree_sha", "started_at")


def attempt_pointers_of_cell(cell_root: Path) -> List[Dict[str, Any]]:
    """Every `watch/attempt_<fcid>.json` pointer under the store root — one per launcher
    ATTEMPT (a FunctionCall) — with the fields the provenance check reads; a pointer
    missing one refuses (the record's shape moved)."""
    out: List[Dict[str, Any]] = []
    for f in sorted(cell_root.rglob("attempt_*.json")):
        doc = read_json(f)
        if not isinstance(doc, Mapping):
            raise AnalysisError(f"{f}: not an attempt pointer (not an object)")
        missing = [k for k in POINTER_FIELDS_READ if k not in doc]
        if missing:
            raise AnalysisError(f"{f}: attempt pointer without {missing} (keys {sorted(doc)}) — the pointer's shape moved")
        out.append({"path": str(f), **{k: doc[k] for k in POINTER_FIELDS_READ}})
    return out


def parse_suffix(suffix: str) -> Dict[str, Optional[str]]:
    """``bcbw-<wave>-<sha7>[-<tag>]`` → its parts; another grammar refuses (the pulled dir IS the durable wave identity)."""
    m = SUFFIX_RE.match(str(suffix))
    if not m:
        raise AnalysisError(f"suffix {suffix!r} is not the lane's `bcbw-<wave>-<sha7>[-<tag>]` (the pulled dir's name is the store "
                            "suffix, the wave's durable identity) — analyse the pulled dir under its suffix")
    return {"wave": m.group("wave"), "sha7": m.group("sha7"), "tag": m.group("tag")}


def sweep_certificate_of(cell_root: Path, *, run_id: str, fcid: str, state_root: Union[str, Path] = SANDBOX_STATE_ROOT) -> Optional[Dict[str, Any]]:
    """The attempt's sandbox sweep certificate: the container's `watch/sweep_<fcid>.json`, else the
    external watcher's `<state root>/<run_id>/sweep_<fcid>.json`; None when neither exists."""
    for p in (cell_root / "watch" / f"sweep_{fcid}.json", Path(state_root).expanduser() / run_id / f"sweep_{fcid}.json"):
        if p.is_file():
            doc = read_json(p)
            if not isinstance(doc, Mapping):
                raise AnalysisError(f"{p}: not a sweep certificate (not an object)")
            return {"path": str(p), **doc}
    return None


def sweep_never_listed(cert: Mapping[str, Any]) -> bool:
    """The sweep certificate's authoritative "the driver never created a sandbox": verified empty with
    NOTHING listed in its first round and nothing terminated, left, unterminatable or still live."""
    return (cert.get("verified_empty") is True and int(cert.get("n_listed_first_round") or 0) == 0
            and not cert.get("ids_terminated") and not cert.get("left") and not cert.get("could_not_terminate") and not cert.get("still_live"))


def sweep_certificate_reason(cert: Optional[Mapping[str, Any]], *, run_id: str, cell: str, fcid: str) -> Optional[str]:
    """Why a sweep certificate does NOT cover the attempt (the watcher's `certificate_admissible`
    rule: the owner tuple exact, `verified_empty` true) — None when it does."""
    if cert is None:
        return "no sweep certificate (neither watch/sweep_<fcid>.json nor the watcher's state dir)"
    got = cert.get("owner")
    want = {"run_id": run_id, "cell": cell, "function_call_id": fcid}
    if not isinstance(got, Mapping) or {k: str(v) for k, v in got.items()} != want:
        return f"sweep certificate owner {got!r} != this attempt's {want}"
    if cert.get("verified_empty") is not True:
        return f"sweep certificate did not verify empty (still_live {cert.get('still_live')}, could_not_terminate {cert.get('could_not_terminate')})"
    return None


def same_wave_provenance(stores: Mapping[Tuple[str, str], Any], *, rung1: Path, cells: Sequence[str], wave: WaveSpec,
                         mode: str = DEFAULT_MODE, sandbox_state_root: Union[str, Path] = SANDBOX_STATE_ROOT) -> Dict[str, Any]:
    """The pairing rule is SAME-WAVE PROVENANCE, and the wave's
    DURABLE identity is its store SUFFIX (``bcbw-<wave>-<sha7>[-tag]``, the pulled
    dir's name) — not a launcher invocation: a resumed cell legitimately carries
    several attempt pointers with several run ids. Every attempt is bound to the
    suffix and kept: its pointer's `out_dir` is this suffix's store root for this
    cell / row / mode, its run id and `tree_sha` carry the suffix's sha7 (= every
    store's producer tree), its cell is the cell; every attempt of a sandboxed leg
    has a sweep certificate covering it (the container's or the watcher's) and its
    per-attempt event record — EMPTY for a standup that failed before the driver
    started (the launcher creates the record before publishing the pointer); the
    legacy shape of such an attempt, no record at all, is admitted ONLY when its
    sweep is verified empty with zero sandboxes listed or terminated
    (:func:`sweep_never_listed`). A cell without a pointer, a pointer from another
    suffix / tree / cell, an attempt without its sweep, a missing record without
    that evidence, or an empty record beside a sweep that found sandboxes refuses.
    Nothing is selected silently: every attempt is listed with whether its driver
    started; `run_id` is set only when the whole wave is one invocation."""
    suffix = rung1.name
    parts = parse_suffix(suffix)
    problems: List[str] = []
    if parts["wave"] != wave.name:
        problems.append(f"suffix {suffix!r} names wave {parts['wave']!r}, not the analysed {wave.name!r}")
    trees = {_dig(st.manifest, (C.MANIFEST_KEY_PRODUCER, "tree_sha")) for st in stores.values()}
    tree = sorted(t for t in trees if isinstance(t, str))[0] if len(trees) == 1 and all(isinstance(t, str) for t in trees) else None
    if tree is None:
        problems.append(f"the stores' producer tree is not one sha: {sorted(str(t) for t in trees)}")
    elif tree[:7] != parts["sha7"]:
        problems.append(f"suffix {suffix!r} carries sha7 {parts['sha7']} but the stores were produced at tree {tree}")
    # the pull's sibling in the analyze path: a pointer's `out_dir` is this suffix's store
    # root under the ENGINE's generation root — a native container writes under `<gen root>/native/…` (the launcher's
    # `gen_root_for_engine`), so the lens root refused every native attempt as "a pointer from another suffix". The engine
    # is the STORES' own (`contracts.read_store_serving_engine`, THE ONE reader; nothing stated on a served store reads as
    # the lens by era only on its positive evidence), never a flag: a torn manifest or two engines across the wave refuse.
    engines_by_store: Dict[str, Optional[str]] = {}
    for (cell, task), st in sorted(stores.items()):
        se = C.read_store_serving_engine(st.manifest, where=f"{cell}/{task} manifest")
        if se.refusal:
            problems.append(f"{cell}/{task}: the store's serving engine cannot be read — {se.refusal}")
        engines_by_store[f"{cell}/{task}"] = se.engine
    engines = sorted({e for e in engines_by_store.values() if e is not None})
    if len(engines) > 1:
        problems.append(f"the stores were generated on several serving engines {engines} — not one wave: "
                        + ", ".join(f"{k}={v}" for k, v in sorted(engines_by_store.items())))
    serving_engine = engines[0] if len(engines) == 1 else None
    gen_root = C.gen_root_for_engine(C.VOL_GEN_ROOT, serving_engine)
    attempts_by_cell: Dict[str, List[Dict[str, Any]]] = {}
    run_ids: set = set()
    for cell in cells:
        root = rung1 / cell
        pointers = attempt_pointers_of_cell(root)
        if not pointers:
            problems.append(f"{cell}: no watch/attempt_*.json pointer carries the launcher's run_id")
        recs = []
        for ptr in pointers:
            where = f"{cell}: pointer {Path(ptr['path']).name}"
            rid = str(ptr["run_id"])
            m = RUN_ID_RE.match(rid)
            if not m:
                problems.append(f"{where}: run_id {rid!r} is not the launcher's <sha7>-<hex8>")
            elif m.group("sha7") != parts["sha7"]:
                problems.append(f"{where}: run_id {rid!r} was minted at sha {m.group('sha7')}, not this suffix's {parts['sha7']} — another launch")
            if str(ptr["cell"]) != cell:
                problems.append(f"{where}: names cell {ptr['cell']!r}, not {cell!r}")
            if tree is not None and ptr["tree_sha"] != tree:
                problems.append(f"{where}: tree_sha {str(ptr['tree_sha'])[:12]}… != the stores' producer tree {tree[:12]}…")
            want_out = f"{C.cell_out_dir(gen_root, cell, str(ptr['mode']), model_slug=str(wave.model_row))}/{suffix}"
            if str(ptr["out_dir"]).rstrip("/") != want_out:
                problems.append(f"{where}: out_dir {ptr['out_dir']!r} is not this suffix's store root {want_out!r}"
                                + (f" (the stores' {serving_engine} engine's namespace)" if serving_engine == RP.SERVING_ENGINE_NATIVE else "")
                                + " — a pointer from another suffix")
            fcid = str(ptr["function_call_id"])
            sandboxed = any(str(a) == C.CONTROL_BIGCODEBENCH_200 for a in (ptr["controls"] or []))
            sweep = sweep_certificate_of(root, run_id=rid, fcid=fcid, state_root=sandbox_state_root) if sandboxed else None
            reason = sweep_certificate_reason(sweep, run_id=rid, cell=cell, fcid=fcid) if sandboxed else None
            record = root / "watch" / f"sandboxes_{fcid}.jsonl"
            n_events: Optional[int] = None
            legacy_no_record = False
            if sandboxed:
                # the per-attempt event record. A standup that FAILED before the driver started leaves the pointer (published
                # before the standup) with an EMPTY record — the launcher creates it before publishing the pointer — and the
                # watcher's sweep verified empty with nothing ever listed. The legacy shape of that attempt (no record file
                # at all) is admitted ONLY on that same authoritative evidence: a sweep certificate covering the attempt,
                # verified empty, zero sandboxes listed or terminated — "the driver never started".
                if record.is_file():
                    n_events = len(read_jsonl(record))
                    if n_events == 0 and sweep is not None and not sweep_never_listed(sweep):
                        problems.append(f"{where}: attempt {fcid} has an EMPTY watch/sandboxes_{fcid}.jsonl record but its sweep found sandboxes "
                                        f"(listed {sweep.get('n_listed_first_round')}, terminated {sweep.get('ids_terminated')}) — a torn record")
                elif sweep is not None and reason is None and sweep_never_listed(sweep):
                    legacy_no_record, n_events = True, 0
                else:
                    problems.append(f"{where}: attempt {fcid} drove a sandboxed leg but has no watch/sandboxes_{fcid}.jsonl record"
                                    + ("" if sweep is None else " (and its sweep certificate does not prove the driver never started: "
                                       f"listed {sweep.get('n_listed_first_round')}, terminated {sweep.get('ids_terminated')}, "
                                       f"verified_empty {sweep.get('verified_empty')})"))
            if reason:
                problems.append(f"{where}: attempt {fcid} is not covered — {reason}")
            run_ids.add(rid)
            recs.append({"run_id": rid, "function_call_id": fcid, "started_at": ptr["started_at"], "sandboxed": sandboxed,
                         "sweep_certificate": (sweep or {}).get("path"), "pointer": ptr["path"],
                         "events_record": (str(record) if record.is_file() else None), "n_events": n_events,
                         "driver_started": (None if not sandboxed else bool(n_events)), "legacy_no_record": legacy_no_record})
        attempts_by_cell[cell] = recs
    if problems:
        raise AnalysisError("same-wave provenance: " + "; ".join(problems))
    return {"suffix": suffix, "suffix_wave": parts["wave"], "sha7": parts["sha7"], "tag": parts["tag"], "tree_sha": tree,
            "run_ids": sorted(run_ids), "run_id": (sorted(run_ids)[0] if len(run_ids) == 1 else None),
            "n_attempts": sum(len(v) for v in attempts_by_cell.values()), "attempts_by_cell": attempts_by_cell, "one_wave": True,
            "rule": "durable wave identity = the store suffix; every attempt (launcher invocation) bound to it and covered by a sweep certificate"}


def common_identity_check(stores: Mapping[Tuple[str, str], Any], *, cells: Sequence[str], tasks: Sequence[str],
                          wave: WaveSpec, mode: str) -> Dict[str, Any]:
    """ONE experiment across every (cell, task) store of the wave: every store's
    `model_id` / `model_revision` equals the roster's served pair (and the
    registry row's when the slug is registered), every run scalar is one per
    run, the hashed `tinycap` block and the `sampling` / `thinking` blocks agree
    in their PROJECTION onto the wave's filed tasks (scalars stripped of all
    task-map / provenance fields at once; maps per retained task), the
    `served` geometry agrees minus the tool-call facts (which follow the legs
    and are then held equal per task across cells), the serving identity's
    backend + mode is one per run (the PATH is per cell), and the per-cell
    intervention facts agree across a cell's stores. Under `protocol` every
    store must carry the served clamp-override record. The caller refuses on any finding."""
    hf, rev = wave.served.get("hf"), wave.served.get("rev")
    if not hf or not rev:
        raise AnalysisError(f"wave {wave.name}: the roster's served {{hf, rev}} is incomplete: {wave.served}")
    row_rec: Dict[str, Any] = {"slug": wave.model_row, "hf": hf, "rev": rev, "registered": False}
    if wave.model_row and wave.model_row in C.MODELS:
        row = C.model_row(wave.model_row)
        row_rec["registered"] = True
        if (row.hf, row.rev) != (hf, rev):
            raise AnalysisError(f"wave {wave.name}: the registry row {wave.model_row} serves {row.hf} @ {row.rev} but the roster says "
                                f"{hf} @ {rev} — the roster and the registry disagree")
    off_row: Dict[str, Dict[str, Any]] = {}
    for (cell, task), st in stores.items():
        m = st.manifest
        if m.get("model_id") != hf or m.get("model_revision") != rev:
            off_row[f"{cell}/{task}"] = {"model_id": m.get("model_id"), "model_revision": m.get("model_revision"), "row": {"hf": hf, "rev": rev}}
    keys = sorted(stores)
    ref_key = keys[0]
    ref = stores[ref_key].manifest
    mismatched: Dict[str, Any] = {}
    modes = {f"{c}/{t}": sampling_mode_of(st.manifest) for (c, t), st in stores.items()}
    if len(set(modes.values())) != 1:
        mismatched["sampling.mode"] = modes
    for path in RUN_SCALAR_PATHS:
        want = _dig_present(ref, path)
        for key in keys[1:]:
            have = _dig_present(stores[key].manifest, path)
            if want is _MISSING or have is _MISSING or want != have:
                mismatched[".".join(path)] = {f"{ref_key[0]}/{ref_key[1]}": (None if want is _MISSING else want),
                                              f"{key[0]}/{key[1]}": (None if have is _MISSING else have)}
    retained = list(tasks)
    leg_fields = {f for fields in L.LEG_IDENTITY_FIELDS.values() for f in fields}

    def _tinycap_run_level(m: Mapping[str, Any]) -> Dict[str, Any]:
        """The hashed block minus what follows the LEGS a store files rather than the run: the leg-specific keys (the
        per-leg check's), `tasks` (held to the roster's runnable legs above) and the served tool-call facts (compared
        per leg below)."""
        tc = m[C.MANIFEST_KEY_TINYCAP]
        out = {k: v for k, v in tc.items() if k not in leg_fields and k != "tasks"}
        out["served"] = {k: v for k, v in (tc.get("served") or {}).items() if k not in L.LEG_DERIVED_SERVED_KEYS}
        return out

    if "sampling.mode" not in mismatched:
        def _filed(st) -> List[str]:
            try:
                return L.row_tasks_of(st.identity, where=f"{st.cell}/{st.task}")
            except L.LadderError as e:
                raise AnalysisError(str(e)) from e
        for key in keys[1:]:
            # a pair is compared on the tasks BOTH stores file (the normal launch files both legs in one store; a
            # single-leg store — a rerun, a per-leg launch — is projected onto the legs it shares with the reference)
            both = [t for t in retained if t in _filed(stores[ref_key]) and t in _filed(stores[key])]
            m = stores[key].manifest
            try:
                p_ref = project(_tinycap_run_level(ref), both, mode=mode, where=f"{ref_key[0]}/{ref_key[1]} tinycap")
                for name in _projection_diff(p_ref, project(_tinycap_run_level(m), both, mode=mode, where=f"{key[0]}/{key[1]} tinycap")):
                    mismatched.setdefault(f"tinycap.{name}", {})[f"{key[0]}/{key[1]}"] = "differs from " + f"{ref_key[0]}/{ref_key[1]}"
                for blk in PROJECTED_BLOCKS:
                    pb = project(ref.get(blk) or {}, both, mode=mode, where=f"{ref_key[0]}/{ref_key[1]} {blk}", kind=PROJECTION_KIND_OF_BLOCK[blk])
                    for name in _projection_diff(pb, project(m.get(blk) or {}, both, mode=mode, where=f"{key[0]}/{key[1]} {blk}",
                                                              kind=PROJECTION_KIND_OF_BLOCK[blk])):
                        mismatched.setdefault(f"{blk}.{name}", {})[f"{key[0]}/{key[1]}"] = "differs from " + f"{ref_key[0]}/{ref_key[1]}"
            except AnalysisError as e:
                mismatched[f"<projection>[{key[0]}/{key[1]}]"] = str(e)
    # the serving identity, component-wise: backend + mode one per run, the path one per cell
    ident: Dict[Tuple[str, str], Any] = {}
    for key in keys:
        raw = _dig_present(stores[key].manifest, SERVING_IDENTITY_PATH)
        try:
            ident[key] = cm.parse_serving_identity(None if raw is _MISSING else raw)
        except Exception as e:  # noqa: BLE001 — a missing or malformed identity is a refusal, named
            mismatched[f"{'.'.join(SERVING_IDENTITY_PATH)}[{key[0]}/{key[1]}]"] = {"value": (None if raw is _MISSING else raw),
                                                                                  "error": f"{type(e).__name__}: {e}"}
    if len(ident) == len(keys):
        inst = {key: (si.backend, si.mode) for key, si in ident.items()}
        if len(set(inst.values())) != 1:
            mismatched[".".join(SERVING_IDENTITY_PATH) + ".backend+mode"] = {f"{k[0]}/{k[1]}": "+".join(v) for k, v in inst.items()}
    serving_path_by_cell: Dict[str, Any] = {}
    for cell in cells:
        paths = {task: (ident[(cell, task)].path if (cell, task) in ident else None) for task in tasks}
        serving_path_by_cell[cell] = paths[tasks[0]] if len(set(paths.values())) == 1 else paths
    # `served` minus the tool-call facts, which follow the legs a configuration files
    served_ref = {k: v for k, v in (ref[C.MANIFEST_KEY_TINYCAP].get("served") or {}).items() if k not in L.LEG_DERIVED_SERVED_KEYS}
    for key in keys[1:]:
        served = {k: v for k, v in (stores[key].manifest[C.MANIFEST_KEY_TINYCAP].get("served") or {}).items()
                  if k not in L.LEG_DERIVED_SERVED_KEYS}
        if served != served_ref:
            mismatched["tinycap.served"] = {f"{ref_key[0]}/{ref_key[1]}": served_ref, f"{key[0]}/{key[1]}": served}
    for task in tasks:
        if task != BFCL_TASK:                           # the tool-call facts are BFCL's instrument; a BCB-only store serves without them
            continue
        for k in L.LEG_DERIVED_SERVED_KEYS:            # the same leg served the same way in every cell
            vals = {cell: (stores[(cell, task)].identity.get("served") or {}).get(k) for cell in cells}
            if len({_canon(v) for v in vals.values()}) != 1:
                mismatched[f"tinycap.served.{k}[{task}]"] = vals
    if mode == SAMPLING_MODE_PROTOCOL:
        for key in keys:
            if not isinstance((stores[key].identity.get("served") or {}).get("sampling_clamp_override"), Mapping):
                mismatched[f"tinycap.served.sampling_clamp_override[{key[0]}/{key[1]}]"] = "absent under protocol (D13: the override is hashed into every protocol store)"
    per_cell: Dict[str, Dict[str, Any]] = {}
    for cell in cells:
        for path in PER_CELL_IDENTITY_PATHS:
            vals = {task: _dig(stores[(cell, task)].manifest, path) for task in tasks}
            if len({_canon(v) for v in vals.values()}) != 1:
                per_cell[f"{cell}:{'.'.join(path)}"] = vals
        if isinstance(serving_path_by_cell.get(cell), dict):       # the path differs across THIS cell's stores
            per_cell[f"{cell}:{'.'.join(SERVING_IDENTITY_PATH)}.path"] = serving_path_by_cell[cell]
    return {"model_row": row_rec, "off_row": off_row, "mismatched": mismatched, "per_cell_mismatched": per_cell,
            "one_experiment": not (off_row or mismatched or per_cell), "serving_path_by_cell": serving_path_by_cell,
            "sampling_mode": (sorted(set(modes.values()))[0] if len(set(modes.values())) == 1 else None),
            "paths": [".".join(p) for p in RUN_SCALAR_PATHS] + ["tinycap (projected on the wave's tasks)", *(f"{b} (projected)" for b in PROJECTED_BLOCKS),
                                                                "tinycap.served (minus the leg-derived keys)", ".".join(SERVING_IDENTITY_PATH) + ".backend+mode"]}


def protocol_section(stores: Mapping[Tuple[str, str], Any], *, cells: Sequence[str], tasks: Sequence[str], wave: WaveSpec,
                     mode: str) -> Dict[str, Any]:
    """The decoding record, read LITERALLY: under
    `protocol`, per leg the manifest's `sampling.by_task[task]` — `executed`
    {temperature, top_p, top_k}, `protocol_temperature`, `executed_reason`,
    `effective` (the in-image resolution) — and the served block's
    `sampling_clamp_override`; BigCodeBench must execute T 0 and BFCL T 0.001 ==
    its protocol temperature, `effective.temperature` equal to the executed one
    when recorded. Under `card` (a legacy store) the run-level scalars are the
    record and the `decode` string says so. The roster's declared mode must match."""
    ref = stores[(cells[0], tasks[0])].manifest
    samp = ref.get(C.MANIFEST_KEY_SAMPLING) or {}
    out: Dict[str, Any] = {"mode": mode, "roster_mode": wave.sampling, "by_task": {}, "clamp_override": None, "decode_by_task": {},
                           "check_4_literal": {}, "problems": []}
    if wave.sampling and wave.sampling != mode:
        raise AnalysisError(f"wave {wave.name}: the stores were drawn under sampling mode {mode!r} but the roster declares {wave.sampling!r}")
    if mode != SAMPLING_MODE_PROTOCOL:
        # a model-card / table-mode store: named as such and NEVER carrying the word "protocol" (the paper notebook selects
        # protocol rows by that word and excludes model-card rows by name)
        card = f"model card decoding (mode {mode}): sampled T={samp.get('temperature')} top_p={samp.get('top_p')} top_k={samp.get('top_k')}, thinking OFF, k=1"
        for task in tasks:
            out["decode_by_task"][task] = card
            out["by_task"][task] = {"executed": {k: samp.get(k) for k in ("temperature", "top_p", "top_k")}, "protocol_temperature": None,
                                    "executed_reason": f"{mode} mode: the run-level row"}
        out["note"] = "a non-protocol store: the run-level sampling scalars are the record; acceptance check 4's literal reading does not apply"
        return out
    by_task = samp.get("by_task")
    if not isinstance(by_task, Mapping):
        raise AnalysisError(f"protocol store without a manifest sampling.by_task record ({cells[0]}/{tasks[0]})")
    override = (ref[C.MANIFEST_KEY_TINYCAP].get("served") or {}).get("sampling_clamp_override")
    out["clamp_override"] = dict(override) if isinstance(override, Mapping) else None
    if wave.sampling_clamp_override and out["clamp_override"] is not None:
        # compare on the numbers (`why` prose may be worded differently); an ABSENT override is one problem, filed below
        for k in ("from", "to"):
            if out["clamp_override"].get(k) != wave.sampling_clamp_override.get(k):
                out["problems"].append(f"served.sampling_clamp_override.{k} {out['clamp_override'].get(k)!r} != the roster's "
                                       f"{wave.sampling_clamp_override.get(k)!r}")
    for task in tasks:
        rec = (((stores[(cells[0], task)].manifest.get(C.MANIFEST_KEY_SAMPLING) or {}).get("by_task")) or {}).get(task)
        if not isinstance(rec, Mapping) or not isinstance(rec.get("executed"), Mapping):
            raise AnalysisError(f"protocol store: sampling.by_task has no executed record for {task} ({cells[0]})")
        ex = dict(rec["executed"])
        eff = rec.get("effective") if isinstance(rec.get("effective"), Mapping) else None
        want_t = PROTOCOL_EXPECTED_TEMPERATURE[task]
        checks = {"executed_temperature": (ex.get("temperature") == want_t),
                  "protocol_equals_executed": (rec.get("protocol_temperature") == ex.get("temperature")),
                  "effective_equals_executed": (eff is None or eff.get("temperature") == ex.get("temperature")),
                  "top_p_1_top_k_0": (ex.get("top_p") in (1.0, 1) and ex.get("top_k") in (0, None))}
        for name, ok in checks.items():
            if not ok:
                out["problems"].append(f"{TASK_SHORT[task]}: {name} FAILS (executed {ex}, protocol {rec.get('protocol_temperature')}, effective {eff})")
        for cell in cells:
            other = ((stores[(cell, task)].manifest.get(C.MANIFEST_KEY_SAMPLING) or {}).get("by_task") or {}).get(task)
            if not isinstance(other, Mapping) or _canon(other.get("executed")) != _canon(ex):
                out["problems"].append(f"{cell}/{TASK_SHORT[task]}: executed record differs from {cells[0]}'s")
        out["by_task"][task] = {"executed": ex, "protocol_temperature": rec.get("protocol_temperature"), "executed_reason": rec.get("executed_reason"),
                                "effective": (dict(eff) if eff else None), "source": rec.get("source")}
        out["check_4_literal"][task] = checks
        ov = out["clamp_override"] or {}
        out["decode_by_task"][task] = protocol_decode_string(
            task, temperature=ex.get("temperature"), top_p=ex.get("top_p"), top_k=ex.get("top_k"),
            declared=rec.get("protocol_temperature"), reason=rec.get("executed_reason"), clamp=(ov.get("from"), ov.get("to")))
    if out["clamp_override"] is None:
        out["problems"].append("served.sampling_clamp_override absent on a protocol store (D13)")
    if out["problems"]:
        raise AnalysisError("protocol record (acceptance check 4, literal): " + "; ".join(out["problems"]))
    return out


def protocol_decode_string(task: str, *, temperature: Any, top_p: Any, top_k: Any, declared: Any = None, reason: Optional[str] = None,
                           clamp: Optional[Tuple[Any, Any]] = None, thinking: str = C.THINKING_OFF, not_run: bool = False) -> str:
    """The paper row's `decode` string under the benchmarks' protocol — it begins with the literal words
    ``benchmark protocol`` (the paper notebook SELECTS protocol rows by the word "protocol"; a model-card
    row never carries it): the leg, the executed T (with the benchmark's declared T and the executed
    reason when recorded), top_p / top_k, thinking, k = 1, the served vLLM clamp override. A row that
    did NOT run carries the roster's DECLARED protocol, labelled so."""
    head = "benchmark protocol (declared by the roster; NOT RUN)" if not_run else "benchmark protocol"
    detail = ""
    if declared is not None or reason:
        detail = " (" + "; ".join(x for x in ((f"declared {declared}" if declared is not None else None), reason) if x) + ")"
    tail = f"; vLLM clamp override {clamp[0]}->{clamp[1]} (D13)" if clamp and clamp[0] is not None else ""
    return f"{head}: {TASK_SHORT[task]} T={temperature}{detail}, top_p {top_p}, top_k {top_k}, thinking {str(thinking).upper()}, k=1{tail}"


def declared_decode_string(task: str, wave: WaveSpec) -> Optional[str]:
    """The decode string of a row that did not run: the roster's declared mode — `protocol` →
    the contracts' per-task protocol row (`PROTOCOL_SAMPLING_BY_TASK`), labelled NOT RUN; another
    declared mode → None (nothing to state)."""
    if wave.sampling != SAMPLING_MODE_PROTOCOL or task not in C.PROTOCOL_SAMPLING_BY_TASK:
        return None
    row = C.PROTOCOL_SAMPLING_BY_TASK[task].executed
    ov = wave.sampling_clamp_override or {}
    return protocol_decode_string(task, temperature=row.temperature, top_p=row.top_p, top_k=row.top_k, thinking=str(wave.thinking or C.THINKING_OFF),
                                  clamp=((ov.get("from"), ov.get("to")) if ov else None), not_run=True)


def load_run(rung1_dir: Union[str, Path], wave: WaveSpec, *, repo_root: Union[str, Path] = REPO,
             roster_sha: Optional[str] = None, roster_path: Optional[Union[str, Path]] = None,
             sandbox_state_root: Union[str, Path] = SANDBOX_STATE_ROOT) -> Dict[str, Any]:
    """Stores, pins and every identity gate of the wave, refused before any pairing:
    the stores present are EXACTLY the roster's runnable cells × runnable legs (a
    store for a cell the roster does not list as runnable refuses; a missing one
    refuses), ONE experiment across all of them (:func:`common_identity_check`),
    same-wave provenance, per leg the same instrument (:func:`identity_check`),
    the generation record per cell (:func:`generation_record`), the fit layers
    per fitted cell (:func:`fit_layers_of`) and the protocol record."""
    rung1 = Path(rung1_dir).expanduser().resolve()
    cells, tasks = wave.cells, wave.tasks
    if not tasks:
        raise AnalysisError(f"wave {wave.name}: no runnable leg in the roster ({wave.legs})")
    if not wave.steered_cells:
        raise AnalysisError(f"wave {wave.name}: no runnable steered cell — nothing to contrast against `none` (a baseline-only wave, D1859: "
                            "its steered contrasts are read where its anchoring waves are paired, not here)")
    present = L.cells_of(rung1)
    extra = sorted(set(present) - set(cells))
    if extra:
        raise AnalysisError(f"{rung1}: stores for cells the roster does not list as runnable in wave {wave.name}: {extra} "
                            "(a store is never analysed under a label the roster did not launch)")
    missing = sorted(set(cells) - set(present))
    if missing:
        raise AnalysisError(f"{rung1}: the roster's runnable cells {missing} have no store here — pull them or mark them in the roster")
    try:
        # the ladder's loader admits each store's ROOT first (`tinycap_ladder.root_admissibility_reason`: a root carrying the worker's
        # sampling-validation failure refuses before a byte is read) — its refusal is the analysis's refusal
        stores = {(cell, task): L.load_store(rung1, cell, task) for task in tasks for cell in cells}
    except L.LadderError as e:
        raise AnalysisError(str(e)) from e
    for cell in cells:
        filed = set(L.row_tasks_of(stores[(cell, tasks[0])].identity, where=f"{cell} identity"))
        blocked_filed = [t for t in wave.blocked_legs if t in filed]
        if blocked_filed:
            raise AnalysisError(f"{cell}: the store files {blocked_filed}, legs the roster marks blocked / not runnable for wave {wave.name}")
    pins = {task: L.load_pin(repo_root, task) for task in tasks}
    families = load_lib_families(repo_root) if BCB_TASK in tasks else {}
    modes = {sampling_mode_of(st.manifest) for st in stores.values()}
    if len(modes) != 1:
        raise AnalysisError(f"identity: the stores were drawn under several sampling modes {sorted(modes)} — not one experiment")
    mode = modes.pop()
    common = common_identity_check(stores, cells=cells, tasks=tasks, wave=wave, mode=mode)
    if common["off_row"]:
        raise AnalysisError(f"identity: stores are not the wave's served model {common['model_row']}: "
                            + "; ".join(f"{k} {v['model_id']} @ {v['model_revision']}" for k, v in common["off_row"].items()))
    if common["mismatched"] or common["per_cell_mismatched"]:
        raise AnalysisError("identity: the wave's stores are not ONE experiment — "
                            + "; ".join(f"{k} differs across stores: {v}" for k, v in common["mismatched"].items())
                            + ("; " if common["mismatched"] and common["per_cell_mismatched"] else "")
                            + "; ".join(f"{k} differs across a cell's stores: {v}" for k, v in common["per_cell_mismatched"].items()))
    provenance = same_wave_provenance(stores, rung1=rung1, cells=cells, wave=wave, sandbox_state_root=sandbox_state_root)
    identity = {task: identity_check({cell: stores[(cell, task)].manifest for cell in cells}, task=task, pin=pins[task], mode=mode)
                for task in tasks}
    for task, idc in identity.items():
        if not idc["same_instrument"]:
            raise AnalysisError(f"identity: the cells are not the same instrument on {task}: "
                                + "; ".join(f"{k} base={v.get('base')!r} cell={v.get('cell')!r}" for k, v in idc["mismatched_required"].items()))
        if idc["off_instrument"]:
            raise AnalysisError(f"identity: the stores of {task} are not this lane's intended instrument: "
                                + "; ".join(f"{k} store={v['store']!r} intended={v['intended']!r}" for k, v in idc["off_instrument"].items()))
    generations: Dict[str, Dict[str, Any]] = {}
    fit_layers: Dict[str, Optional[Dict[str, Any]]] = {}
    for cell in cells:
        entry = wave.entry_of_cell(cell)
        generations[cell] = generation_record(rung1 / cell, cell=cell, entry=entry, wave=wave,
                                              manifest=stores[(cell, tasks[0])].manifest, roster_sha=roster_sha, roster_path=roster_path)
        fit_layers[cell] = fit_layers_of(rung1, entry, served_sha=generations[cell]["object_sha256"]) if entry.fitted else None
    for steered in wave.steered_cells:
        check_pairing_generations(generations[BASE_CELL], generations[steered])
    protocol = protocol_section(stores, cells=cells, tasks=tasks, wave=wave, mode=mode)
    return {"rung1_dir": str(rung1), "suffix": rung1.name, "cells": list(cells), "steered": list(wave.steered_cells), "tasks": list(tasks),
            "stores": stores, "pins": pins, "families": families, "identity": identity, "identity_common": common, "mode": mode,
            "provenance": provenance, "generations": generations, "fit_layers": fit_layers, "protocol": protocol}


# --------------------------------------------------------------------------- #
# --analyze: the paired analysis per leg (K steered cells vs `none`)           #
# --------------------------------------------------------------------------- #


def _values(rows: Mapping[str, Mapping[str, Any]], *, where: str) -> Dict[str, float]:
    out = {}
    for item, r in rows.items():
        v = r.get("value")
        if v not in (0, 1, 0.0, 1.0) or isinstance(v, bool):
            raise AnalysisError(f"{where}: item {item!r} acc value {v!r} is not 0/1 — the paired design reads a pass/fail")
        out[item] = float(v)
    return out


def paired_section(a: Mapping[str, float], b: Mapping[str, float]) -> Dict[str, Any]:
    """steered (b) − none (a): the paired bootstrap, McNemar and the discordant counts."""
    if set(a) != set(b):
        raise AnalysisError("paired: the two cells' item sets differ")
    boot = P.paired_bootstrap(a, b, n_resamples=BOOTSTRAP_N, seed=BOOTSTRAP_SEED)
    b_pf = sum(1 for i in a if a[i] == 1.0 and b[i] == 0.0)
    c_fp = sum(1 for i in a if a[i] == 0.0 and b[i] == 1.0)
    out = {"n_items": len(a), "delta": boot["delta"], "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
           "B": boot["B"], "seed": boot["seed"], "level": boot["level"], "excludes_zero": boot["excludes_zero"],
           "b_pass_to_fail": b_pf, "c_fail_to_pass": c_fp, "n_discordant": b_pf + c_fp,
           "mcnemar_exact_p": mcnemar_exact_p(b_pf, c_fp)}
    out["cost_detected"] = cost_detected(out)           # the ONE bit every emphasis follows, recorded beside the interval
    return out


def cost_detected(paired: Mapping[str, Any]) -> bool:
    """The reading rule's decision bit (pre-registered in the plan): the exact two-sided McNemar test
    rejects at 0.05 AND the paired point estimate is negative. The figures highlight on THIS, never on the bootstrap
    interval, which is an estimation interval."""
    return float(paired["mcnemar_exact_p"]) < 0.05 and float(paired["delta"]) < 0


def interval_disagrees(paired: Mapping[str, Any]) -> bool:
    """True when the bootstrap interval's exclusion of zero and the exact test's decision disagree on a NEGATIVE estimate
    — the gemma-it si_safety BFCL row at the review: −2.0 [−4.0, −0.5], b 4 / c 0, p 0.125. With b + c = 4 discordant
    pairs all one way, the percentile bootstrap has ≈ 1.8 % mass at zero and its interval excludes 0 while the exact
    test does not reject; sparse discordance, not evidence. (A positive estimate is never a cost either way.)"""
    return float(paired["delta"]) < 0 and bool(paired["excludes_zero"]) != cost_detected(paired)


def zero_mass_share(paired: Mapping[str, Any]) -> float:
    """The share of percentile-bootstrap resamples whose Δ is EXACTLY 0 WHEN EVERY DISCORDANT ITEM POINTS THE SAME WAY
    (b = 0 or c = 0): then a resample's Δ is 0 iff it draws none of the b + c discordant items, ((n − (b + c)) / n)^n —
    1.76 % for b + c = 4 of 200 (computed, never hard-coded). With discordants in BOTH directions resampled Δ can also
    reach 0 by cancellation (n 200, b 3, c 1 → ≈ 13 %), so callers use this only under one-way discordance."""
    n, d = int(paired["n_items"]), int(paired["n_discordant"])
    return ((n - d) / n) ** n if n else 0.0


def sparse_discordance_sentence(paired: Mapping[str, Any]) -> str:
    """The one sentence the doc / figures add where the interval and the exact test disagree, saying exactly why: when
    the interval excludes 0 but the exact test does not reject, the resamples with Δ exactly 0 (all discordant items
    missed) are FEWER than the 2.5 % a percentile bound needs to reach 0, so the bound stays strictly negative — the
    gemma-it si_safety BFCL row: b 4 / c 0, (196/200)^200 = 1.76 % < 2.5 %, interval [−4.0, −0.5], exact p 0.125."""
    n, d, p = int(paired["n_items"]), int(paired["n_discordant"]), paired["mcnemar_exact_p"]
    b, c = int(paired["b_pass_to_fail"]), int(paired["c_fail_to_pass"])
    if paired["excludes_zero"] and not cost_detected(paired):
        if min(b, c) == 0:      # every discordant item points the same way: the exact-0 resamples are those drawing none of them
            share = 100.0 * zero_mass_share(paired)
            return (f"the bootstrap interval excludes 0 while the exact test does not reject ({_p(p)}): all b + c = {d} discordant items among {n} "
                    f"point the same way (b = {b}, c = {c}), so a resample's Δ is 0 only when it draws none of them — probability "
                    f"(({n} − {d})/{n})^{n} = {share:.2f} %, fewer than the 2.5 % a percentile bound needs to reach 0, which is why the interval "
                    "excludes 0 — sparse discordance, not evidence; the exact test is the rule")
        return (f"the bootstrap interval excludes 0 while the exact test does not reject ({_p(p)}) on b = {b} / c = {c} discordant items among {n} — "
                "with few discordant items the percentile bound is set by a handful of resample counts and can exclude 0 while the exact "
                "conditional test does not reject; the exact test is the rule")
    return (f"the bootstrap interval includes 0 while the exact test rejects ({_p(p)}) on b + c = {d} discordant pairs — the percentile interval "
            "and the exact conditional test answer different questions; the exact test is the rule")


def reading_rule(paired: Mapping[str, Any]) -> str:
    """The first lane's reading rule, stated the same way for every cell: a
    cost is detected only when the exact test rejects AND the
    point estimate is negative; a positive estimate is never an improvement
    unless the exact test rejects — and even then it is "no cost detected; a
    gain is not established" here (equivalence not tested). Where the bootstrap
    interval disagrees with the test, the sparse-discordance sentence rides the
    reading; p prints `< 0.001` or three significant figures."""
    p, d = float(paired["mcnemar_exact_p"]), float(paired["delta"])
    tail = ("; " + sparse_discordance_sentence(paired)) if interval_disagrees(paired) else ""
    if cost_detected(paired):
        return (f"cost detected: {_pp(d)} [{_pp(paired['ci_low'])}, {_pp(paired['ci_high'])}], exact McNemar {_p(p)} "
                f"(b = {paired['b_pass_to_fail']} pass→fail, c = {paired['c_fail_to_pass']} fail→pass); equivalence not tested" + tail)
    if p < 0.05 and d > 0:
        return (f"no cost detected; the exact test rejects in the POSITIVE direction ({_pp(d)}, {_p(p)}) — not reported as an "
                "improvement (one draw, k = 1; equivalence not tested)")
    return (f"no cost detected ({_pp(d)} [{_pp(paired['ci_low'])}, {_pp(paired['ci_high'])}], {_p(p)}); no gain established; "
            "equivalence not tested" + tail)


def _counts(values: Iterable[Any]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in values:
        k = "null" if v is None else str(v)
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def cell_section(task: str, st: Any) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], Dict[str, float]]:
    """One (cell, leg) store's reading: the pass rate, the diagnostics, the rows."""
    rows = st.acc_rows()
    vals = _values(rows, where=str(st.cell_dir))
    at_cap = L.at_cap_reading(st)
    extras = [r["extra"] for r in rows.values()]
    unp = [e.get("unparseable") for e in extras]
    n_answered = sum(1 for u in unp if u is not None)
    n_unparseable = sum(1 for u in unp if u is True)
    s = st.summary
    if s.get("n_unparseable") != (n_unparseable if n_answered else None) or s.get("n_answered") != n_answered:
        raise AnalysisError(f"{st.cell_dir}: {task} unparseable disagreement — rows {n_unparseable}/{n_answered}, "
                            f"summary {s.get('n_unparseable')}/{s.get('n_answered')}")
    out: Dict[str, Any] = {
        **pass_rate(int(sum(vals.values())), len(vals)),
        "at_cap": at_cap,
        "unparseable": {"n_unparseable": n_unparseable, "n_answered": n_answered, "rate": (n_unparseable / n_answered) if n_answered else None},
        "n_unscored": sum(1 for e in extras if e.get("unscored") is True),
        "stop_reasons": _counts(e.get("stop_reason") for e in extras),
        "wall_clock_s": s.get("wall_clock_s"), "n_requests": s.get("n_requests"), "cap": s.get("max_tokens"),
        "store_key": st.key, "marker": st.marker_kind,
        "completion_tokens": A.summary_stats([e.get("completion_tokens") for e in extras]),
        "prompt_tokens": A.summary_stats([e.get("prompt_tokens") for e in extras])}
    if task == BCB_TASK:
        for key in C.BCB_ROW_FACT_FIELDS:
            if any(key not in e for e in extras):
                raise AnalysisError(f"{st.cell_dir}: a BCB row carries no `{key}` — the leg's row facts (plan §S2) are missing")
        out["code_source"] = _counts(e["code_source"] for e in extras)
        out["sandbox_status"] = _counts(e["sandbox_status"] for e in extras)
        out["n_verify_timeout"] = out["sandbox_status"].get("verify_timeout", 0)
        want = {"n_sandbox_ok": out["sandbox_status"].get("ok", 0), "n_verify_timeout": out["n_verify_timeout"],
                **{f"n_code_source_{k}": out["code_source"].get(k, 0) for k in C.BCB_CODE_SOURCES}}
        missing_f = [f for f in C.BCB_SUMMARY_COUNT_FIELDS if f not in s]
        if missing_f:
            raise AnalysisError(f"{st.cell_dir}: the BCB summary carries no {missing_f} — the producer's counts are missing")
        bad = {f: (s[f], want[f]) for f in C.BCB_SUMMARY_COUNT_FIELDS if s[f] != want[f]}
        if bad:
            raise AnalysisError(f"{st.cell_dir}: BCB summary counts disagree with the rows: {bad} (summary, rows)")
    return out, rows, vals


def leg_section(task: str, stores: Mapping[str, Any], pin: Mapping[str, Any], *, generations: Mapping[str, Mapping[str, Any]],
                base: str = BASE_CELL) -> Dict[str, Any]:
    """One leg of the wave: every cell's reading on exactly the 200 pinned
    items, and for EVERY steered cell the paired contrast against the wave's
    `none` — the only contrast the lane computes; the generation guard is
    checked at the write moment of each pair."""
    pin_log = [L.to_log_id(task, i) for i in pin["ids"]]
    rows_by_cell: Dict[str, Dict[str, Dict[str, Any]]] = {}
    vals_by_cell: Dict[str, Dict[str, float]] = {}
    per_cell: Dict[str, Any] = {}
    for cell, st in stores.items():
        cell_out, rows, vals = cell_section(task, st)
        missing = [i for i in pin_log if i not in rows]
        extra = sorted(set(rows) - set(pin_log))
        if missing or extra:
            raise AnalysisError(f"{st.cell_dir}: {task} rows are not exactly the 200 pinned items "
                                f"(missing {len(missing)}, e.g. {missing[:5]}; extra {len(extra)}, e.g. {extra[:5]})")
        rows_by_cell[cell], vals_by_cell[cell], per_cell[cell] = rows, vals, cell_out
    if base not in stores:
        raise AnalysisError(f"{task}: the wave's `{base}` store is missing")
    steered = [c for c in stores if c != base]
    if not steered:
        raise AnalysisError(f"{task}: the paired design needs `{base}` + at least one steered cell, got {sorted(stores)}")
    paired: Dict[str, Any] = {}
    for cell in steered:
        check_pairing_generations(generations[base], generations[cell])
        paired[cell] = paired_section(vals_by_cell[base], vals_by_cell[cell])
        paired[cell]["reading"] = reading_rule(paired[cell])
        paired[cell]["provisional"] = bool(generations[cell].get("provisional"))
    readings = {c: per_cell[c]["at_cap"] for c in stores}
    return {"task": task, "label": TASK_LABEL[task], "n": FULL_N, "base": base, "steered": steered,
            "cells": per_cell, "paired": paired,
            "fired": L.leg_fires(readings), "fire_threshold": L.FIRE_THRESHOLD, "_rows": rows_by_cell}


RowsByCell = Mapping[str, Mapping[str, Mapping[str, Any]]]     # cell -> item id -> acc row (with `extra` parsed)


def _breakdown_rows(rows_by_cell: RowsByCell, groups: Mapping[str, List[str]], *, base: str, steered: Sequence[str]) -> List[Dict[str, Any]]:
    """Per group: every cell's pass rate over the group's items and, per steered cell, the paired
    delta with its discordant counts — over WHATEVER per-item rows the caller selected (the leg's
    FINAL rows: merged after an applied ladder, else rung 1)."""
    rows_a = rows_by_cell[base]
    out = []
    for name, items in groups.items():
        n = len(items)
        cells = {c: pass_rate(sum(1 for i in items if rows_by_cell[c][i]["value"] == 1.0), n) for c in rows_by_cell}
        deltas = {}
        for c in steered:
            rows_b = rows_by_cell[c]
            deltas[c] = {"delta": (cells[c]["k"] - cells[base]["k"]) / n,
                         "b_pass_to_fail": sum(1 for i in items if rows_a[i]["value"] == 1.0 and rows_b[i]["value"] == 0.0),
                         "c_fail_to_pass": sum(1 for i in items if rows_a[i]["value"] == 0.0 and rows_b[i]["value"] == 1.0)}
        out.append({"group": name, "n": n, "cells": cells, "deltas": deltas})
    return out


def bfcl_by_category(rows_by_cell: RowsByCell, pin: Mapping[str, Any], *, base: str, steered: Sequence[str]) -> Dict[str, Any]:
    cat_of: Dict[str, str] = {}
    for cat, ids in pin["population"].items():
        for i in ids:
            cat_of[i] = cat
    groups: Dict[str, List[str]] = {}
    for item in rows_by_cell[base]:
        cat = cat_of.get(item)
        if cat is None:
            raise AnalysisError(f"BFCL item {item!r} is in no category of the pin's population map")
        groups.setdefault(cat, []).append(item)
    rows = _breakdown_rows(rows_by_cell, groups, base=base, steered=steered)
    for r in rows:
        r["category"] = r["group"]
        r["population_n"] = len(pin["population"][r["group"]])
        r["wide_interval"] = r["n"] <= 5
    rows.sort(key=lambda r: (-r["n"], r["category"]))
    return {"rows": rows, "n_categories": len(rows), "note": "categories with n ≤ 5 carry wide intervals by design (Hamilton floor 1)"}


def bcb_by_family(rows_by_cell: RowsByCell, pin: Mapping[str, Any], families: Mapping[str, str], *, base: str, steered: Sequence[str]) -> Dict[str, Any]:
    libs_by_id = pin["libs_by_id"]
    groups: Dict[str, List[str]] = {}
    unmapped: List[str] = []
    for item in rows_by_cell[base]:
        pid = L.to_pinned_id(BCB_TASK, item)
        libs = libs_by_id.get(pid)
        if libs is None:
            raise AnalysisError(f"BCB item {item!r} ({pid}) has no libs_by_id entry in the pin")
        fams = set()
        for lib in libs:
            fam = families.get(lib)
            if fam is None:
                unmapped.append(lib)
                continue
            fams.add(fam)
        for fam in fams:
            groups.setdefault(fam, []).append(item)
    if unmapped:
        raise AnalysisError(f"{len(set(unmapped))} libraries have no family in {LIB_FAMILIES_REL}: {sorted(set(unmapped))[:8]} (the map must be total)")
    rows = _breakdown_rows(rows_by_cell, groups, base=base, steered=steered)
    for r in rows:
        r["family"] = r["group"]
    rows.sort(key=lambda r: (-r["n"], r["family"]))
    pin_pf = pin.get("per_family")
    if isinstance(pin_pf, Mapping) and pin_pf:
        mine = {r["family"]: r["n"] for r in rows}
        if {k: int(v) for k, v in pin_pf.items()} != mine:
            raise AnalysisError(f"BCB per-family counts differ from the pin's per_family: analysis {mine} vs pin {dict(pin_pf)} — "
                                "the family map or the join moved")
    return {"rows": rows, "n_families": len(rows), "multi_label": True, "matches_pin_per_family": isinstance(pin_pf, Mapping),
            "note": "an item counts under every family of its libraries (multi-label), so the rows do not sum to 200"}


def _quote(text: Any) -> Dict[str, Any]:
    if not isinstance(text, str):
        return {"text": None, "chars": 0, "truncated": False}
    return {"text": text[:QUOTE_CHARS], "chars": len(text), "truncated": len(text) > QUOTE_CHARS}


def bcb_failures(rows_by_cell: RowsByCell, *, n: int = N_QUOTED) -> Dict[str, Any]:
    """Per cell, up to ``n`` failing items by `code_source × verdict × sandbox_status`,
    with the verifier's explanation and the RAW completion (the doc quotes the
    first QUOTE_CHARS characters; analysis.json carries the full text)."""
    out: Dict[str, Any] = {}
    for cell, rows in rows_by_cell.items():
        fails = [r for r in rows.values() if r["value"] == 0.0]
        fails.sort(key=lambda r: int(r["item_id"]) if r["item_id"].isdigit() else r["item_id"])
        grid: Dict[str, int] = {}
        for r in fails:
            e = r["extra"]
            key = f"{e.get('code_source')} × fail × {e.get('sandbox_status')}"
            grid[key] = grid.get(key, 0) + 1
        out[cell] = {"n_fail": len(fails), "grid": dict(sorted(grid.items())),
                     "items": [{"item_id": r["item_id"], "pinned_id": L.to_pinned_id(BCB_TASK, r["item_id"]),
                                "code_source": r["extra"].get("code_source"), "sandbox_status": r["extra"].get("sandbox_status"),
                                "stop_reason": r["extra"].get("stop_reason"), "completion_tokens": r["extra"].get("completion_tokens"),
                                "verifier_explanation": _quote(r["extra"].get("verifier_explanation")),
                                "content": _quote(r["extra"].get("content")), "content_full": r["extra"].get("content")}
                               for r in fails[:n]]}
    return out


def bfcl_discordant(rows_by_cell: RowsByCell, pin: Mapping[str, Any], *, base: str, steered: Sequence[str], n: int = N_QUOTED) -> Dict[str, Any]:
    """Per steered cell: up to ``n`` items whose verdict differs from `none`'s, quoted raw."""
    rows_a = rows_by_cell[base]
    cat_of = {i: cat for cat, ids in pin["population"].items() for i in ids}
    out: Dict[str, Any] = {}
    for cell in steered:
        rows_b = rows_by_cell[cell]
        disc = [i for i in rows_a if rows_a[i]["value"] != rows_b[i]["value"]]
        disc.sort(key=lambda i: (rows_a[i]["value"] < rows_b[i]["value"], i))
        out[cell] = {"n_discordant": len(disc),
                     "items": [{"item_id": i, "category": cat_of.get(i), "direction": ("pass→fail" if rows_a[i]["value"] == 1.0 else "fail→pass"),
                                "cells": {c: {"value": rows[i]["value"], "answer": _quote(rows[i]["extra"].get("answer")),
                                              "content": _quote(rows[i]["extra"].get("content")), "content_full": rows[i]["extra"].get("content"),
                                              "stop_reason": rows[i]["extra"].get("stop_reason")}
                                          for c, rows in ((base, rows_a), (cell, rows_b))}}
                               for i in disc[:n]]}
    return out


# --------------------------------------------------------------------------- #
# --analyze: ladder, sandboxes, cost, wall                                     #
# --------------------------------------------------------------------------- #


def find_certificates(rung_dirs: Sequence[Union[str, Path]], *, explicit: Sequence[Union[str, Path]] = ()) -> Tuple[List[Path], List[Path]]:
    """``(certificates, superseded)``: every per-cell certificate named EXACTLY
    `rung_probe.<cell>.json` (`rung_probe.certificate_filename(cell)`) or the legacy
    single `rung_probe.json` (`rung_probe.CERTIFICATE_FILENAME`) under the supplied
    rung dirs' `_ladder/`, plus explicit paths; a `rung_probe.<cell>.<ts>.superseded.json`
    (a recorded re-probe, `--pull` fetches the whole dir) is LISTED apart, never read
    as a certificate."""
    out: List[Path] = [Path(p).expanduser() for p in explicit]
    superseded: List[Path] = []
    for rd in rung_dirs:
        lad = Path(rd).expanduser() / LADDER_SUBDIR
        if not lad.is_dir():
            continue
        for p in sorted(lad.iterdir()):
            if not p.is_file():
                continue
            if p.name == RP.CERTIFICATE_FILENAME or CERTIFICATE_NAME_RE.fullmatch(p.name):
                out.append(p)
            elif p.name.startswith("rung_probe.") and p.name.endswith(".superseded.json"):
                superseded.append(p)
    seen, uniq = set(), []
    for p in out:
        if str(p) not in seen:
            seen.add(str(p))
            uniq.append(p)
    return uniq, superseded


def ladder_spec_for_certificates(model_key: str, certificates: Mapping[str, Mapping[str, Any]]) -> Tuple["RP.LadderSpec", str]:
    """The ladder spec the wave record is judged against (``rung_probe.admission_is_valid`` reads its window and rung list):
    the SHARED table's (`RP.ladder_spec`) when the table sizes this model, else the ladder the certificates RECORDED (their
    `ladder` block: window, seqs, identity_S — one and the same across the certificates, else refuse). The shipped geometry
    rows make llama8b's row provisional, so `ladder_spec` refuses — but the llama ladders ran under a measured row and every
    certificate carries the ladder it walked, so the analysis judges the run by the ladder the run used rather than refusing
    to re-read a finished wave. Returns ``(spec, source)``; ``source`` names which."""
    # the wave's certificates name ONE engine in their `ladder` blocks (`tinycap_ladder.certificate_serving_engine`:
    # the native keys' `serving_engine`, else lens by era); a native wave is judged by the ENGINE-SERVED ladder
    # (`rung_probe.native_ladder_spec` of the table's), never by the lens rungs
    engines = sorted({L.certificate_serving_engine(doc) for doc in certificates.values()})
    if len(engines) > 1:
        raise AnalysisError(f"the certificates name several serving engines {engines} — not one ladder")
    engine = engines[0] if engines else RP.SERVING_ENGINE_LENS
    try:
        spec = RP.ladder_spec(model_key)
    except RP.LadderError as table_refusal:
        blocks = {json.dumps(dict(doc.get("ladder") or {}), sort_keys=True) for doc in certificates.values()}
        blocks.discard("{}")
        if len(blocks) != 1:
            raise AnalysisError(f"the shared geometry table refuses the ladder of {model_key!r} ({table_refusal}) and the certificates "
                                f"record {len(blocks)} distinct `ladder` blocks — one recorded ladder is needed to judge the wave record") from table_refusal
        lad = json.loads(next(iter(blocks)))
        if lad.get("model_key") != model_key:
            raise AnalysisError(f"the certificates' ladder names model {lad.get('model_key')!r}, not {model_key!r}") from table_refusal
        from sisteer.capeval.serving import geometry as _sg
        try:
            # the recorded block through the ONE reader (a native block re-derived from its requested rungs; the lens
            # block's four keys restated) — the same fallback `tinycap_ladder.spec_for_certificate` makes
            spec = RP.spec_from_block(lad, _sg.BUDGET_BY_MODEL.get(model_key))
        except RP.LadderError as e:
            raise AnalysisError(f"the shared geometry table refuses the ladder of {model_key!r} ({table_refusal}) and the certificates' "
                                f"recorded ladder block cannot be read: {e}") from table_refusal
        return spec, f"certificates (the shared table refuses: {table_refusal})"
    if engine == RP.SERVING_ENGINE_NATIVE:
        try:
            spec = RP.native_ladder_spec(spec)
        except RP.LadderError as e:
            raise AnalysisError(f"the {RP.SERVING_ENGINE_NATIVE} certificates of {model_key!r} cannot be judged: {e}") from None
    return spec, "table"


def ladder_section(legs: Mapping[str, Mapping[str, Any]], *, stores: Mapping[Tuple[str, str], Any], pins: Mapping[str, Mapping[str, Any]],
                   cells: Sequence[str], fitted_cells: Sequence[str], merge_dir: Optional[Union[str, Path]],
                   selection_path: Optional[Union[str, Path]], certificate_paths: Sequence[Union[str, Path]],
                   ledger_entries: Sequence[Mapping[str, Any]], base: str = BASE_CELL,
                   admission_path: Optional[Union[str, Path]] = None, superseded_certificates: Sequence[Union[str, Path]] = (),
                   rung2_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Both readings ONLY from `ladder merge`'s output, bound to the rung-1
    stores being analysed (record keys + file hashes, exactly these cells,
    exactly the 200 pinned items, every count RECOMPUTED). The merged reading is
    FINAL (``applied``) only with the COMPLETE evidence: a valid per-cell
    certificate for EVERY fitted cell (`probed_cell` = itself,
    validated by the launcher's gate, all at the merge's common rung), the wave
    record `rung_admission.json` binding exactly those certificates
    (`rung_probe.admission_is_valid`), and EVERY contributing rung-2 root RE-ADMITTED
    here (the pulled rung-2 dir's store for each (cell, task) loaded
    through `tinycap_ladder.load_store` — whose root admission refuses a
    `sampling_validation_failed.json` — and held to the merge's recorded store key
    and file hashes; a poisoned root REFUSES, a merge analysed without its rung-2
    dir is NOT final). A fitted cell without a certificate, or no record, leaves
    the merged tables recorded but NOT final (``applied`` False, ``final_reason``
    says why, check 4 NOT MET); a record that disagrees with the certificates
    refuses. A fired leg without a merge is check 4 NOT MET — stated, never hidden.
    The merged per-item rows ride ``tasks[task]["_rows"]`` for the final breakdowns
    (the caller pops them)."""
    fired = {task: bool(leg["fired"]) for task, leg in legs.items()}
    decisions = [{"at": e.get("at"), "ceiling_h": e.get("ceiling_h"), "note": e.get("note"), "fired_legs": e.get("fired_legs")}
                 for e in ledger_entries if e.get("kind") == "decision"]
    out: Dict[str, Any] = {"fired": fired, "fired_any": any(fired.values()), "threshold": L.FIRE_THRESHOLD, "applied": False,
                           "budget_decisions": decisions, "operator_rung": False,
                           "at_cap_counts": {task: {c: leg["cells"][c]["at_cap"] for c in leg["cells"]} for task, leg in legs.items()},
                           "certificates": {}, "admission_record": None, "admission_path": None, "final_reason": None,
                           "superseded_certificates": [str(Path(p).expanduser()) for p in superseded_certificates],
                           # `merge --admit-tree`'s record (rung 2 produced at another tree by name); None otherwise
                           "tree_admission": None}
    if admission_path and Path(admission_path).expanduser().is_file():
        out["admission_path"] = str(Path(admission_path).expanduser())
        out["admission_record"] = read_json(admission_path)
    if merge_dir is None:
        out["reason"] = ("no leg fired: the rung-1 reading stands alone" if not out["fired_any"] else
                         "a leg FIRED but no `ladder merge` output was given — the ladder readings are absent and "
                         "acceptance check 4 is NOT MET until the capped items are re-run at 16,384 and merged")
        out["acceptance_check_4"] = "met (no leg fired)" if not out["fired_any"] else "NOT MET (ladder pending)"
        return out
    merged = read_json(Path(merge_dir).expanduser() / L.MERGED_SUMMARY_FILENAME)
    if not isinstance(merged, Mapping) or merged.get("schema") != L.LADDER_SCHEMA:
        raise AnalysisError(f"{merge_dir}: not a `ladder merge` summary (schema {L.LADDER_SCHEMA})")
    if selection_path is None:
        raise AnalysisError("a merge is analysed only with its ladder_selection.json (--selection): the selected item sets are held to it exactly")
    sel = L.load_selection(selection_path)
    if sel["selection_sha256"] != merged["selection_sha256"]:
        raise AnalysisError("the selection file's digest differs from the merged summary's — not the same ladder")
    rec = merged["certificate"]
    accepted = dict(rec["accepted"])
    # the merge's tree admission is carried as recorded; the certificate is issued at the rung-2 tree
    # (`tinycap_ladder.bind_certificate_to_store`), so an admission naming another rung-2 sha is not this merge's
    tree_admission = merged.get("tree_admission")
    if tree_admission is not None:
        if set(tree_admission) != set(L.TREE_ADMISSION_FIELDS):
            raise AnalysisError(f"{merge_dir}: tree_admission carries {sorted(tree_admission)}, not {sorted(L.TREE_ADMISSION_FIELDS)}")
        if tree_admission["rung2_tree_sha"] != rec["tree_sha"]:
            raise AnalysisError(f"{merge_dir}: tree_admission.rung2_tree_sha {str(tree_admission['rung2_tree_sha'])[:12]}… != the "
                                f"certificate's tree {str(rec['tree_sha'])[:12]}… — not the admission of this merge")
    out["tree_admission"] = tree_admission
    certs: Dict[str, Any] = {}
    for path in certificate_paths:
        cert = L.load_certificate(path, selection=sel["selection"])
        cell = str(cert["probed_cell"])
        if cell not in fitted_cells:
            raise AnalysisError(f"{path}: probed_cell {cell!r} is not a fitted cell of this wave ({list(fitted_cells)})")
        if cell in certs:
            raise AnalysisError(f"{path}: a second certificate for {cell!r} ({certs[cell]['path']})")
        for key in ("tree_sha", "rev", "hf", "cfg_sha256"):
            if key not in rec or cert[key] != rec[key]:
                raise AnalysisError(f"{path}: {key} {cert[key]!r} != what the merge recorded {rec.get(key)!r} — not a certificate of this merge")
        if cert["selection_sha256"] != merged["selection_sha256"]:
            raise AnalysisError(f"{path}: the certificate's selection digest differs from the merged summary's")
        if dict(cert["accepted"]) != accepted:
            raise AnalysisError(f"{path}: accepted {cert['accepted']} != the merge's common rung {accepted} — every final per-cell certificate "
                                "carries the wave's common rung (plan §S6: a cell whose walk accepted another S is re-probed at the common footprint)")
        certs[cell] = {"path": str(Path(path).expanduser()), "accepted": dict(cert["accepted"]), "n_walked": len(cert["walked"]),
                       "walked": cert["walked"], "reprobed_from": cert.get("reprobed_from"), "standup_s": cert.get("standup_s"),
                       "suffix": cert.get("suffix"), "model_slug": cert.get("model_slug"), "sha256": RP.certificate_sha256(cert), "_doc": dict(cert)}
    missing_certs = [c for c in fitted_cells if c not in certs]
    not_final: List[str] = []
    if missing_certs:
        not_final.append(f"fitted cell(s) without a certificate: {missing_certs} (one `rung_probe.<cell>.json` per fitted cell under _ladder/)")
    # the wave record binds EXACTLY the final certificates (rung_probe.admission_is_valid): required for a final merged reading
    adm = out["admission_record"]
    if adm is None:
        not_final.append(f"no wave record {RUNG_ADMISSION_FILENAME} under _ladder/ (the launcher writes it before any scoring container; "
                         "--pull fetches the ladder dir)")
    elif certs:
        suffixes = sorted({str(v["suffix"]) for v in certs.values()})
        slugs = sorted({str(v["model_slug"]) for v in certs.values()})
        if len(suffixes) != 1 or len(slugs) != 1:
            raise AnalysisError(f"the certificates name several suffixes {suffixes} / model rows {slugs} — not one ladder")
        try:
            model_key = C.model_row(slugs[0]).model_key
        except C.UnknownModel as e:
            raise AnalysisError(f"certificate model_slug {slugs[0]!r}: {e}") from e
        spec, out["ladder_spec_source"] = ladder_spec_for_certificates(model_key, {c: v["_doc"] for c, v in certs.items()})
        # the engine the certificates were walked on rides the ladder section (the served
        # rungs and the requested -> served mapping with it under native), so a reader of the analysis knows which
        # engine's footprint the rung-2 reading was served at; a lens wave records `lens` and nothing more
        out["serving_engine"] = spec.serving_engine or RP.SERVING_ENGINE_LENS
        out["ladder_spec"] = spec.block()
        reason = RP.admission_is_valid(adm, suffix=suffixes[0], spec=spec, certificates={c: v["_doc"] for c, v in certs.items()})
        if reason and not missing_certs:
            raise AnalysisError(f"{out['admission_path']}: the wave record does not admit these certificates — {reason}")
        if reason:
            not_final.append(f"the wave record does not admit the certificates present: {reason}")
    for v in certs.values():
        v.pop("_doc", None)
    rung_source = accepted["rung_source"]
    merged_cells = merged["per_cell"]
    if set(merged.get("cells") or []) != set(cells) or set(merged_cells) != set(cells):
        raise AnalysisError(f"the merge covers cells {sorted(merged_cells)} / {sorted(merged.get('cells') or [])}, not the analysed {sorted(cells)}")
    if rung2_dir is None:
        not_final.append("the rung-2 stores were not re-admitted: pass --rung2 <the pulled rung-2 dir> (every contributing rung-2 root must "
                         f"be admissible — no {C.SAMPLING_VALIDATION_FAILED_FILENAME} — before the merged reading is final)")
    per_task: Dict[str, Any] = {}
    for task in merged["tasks"]:
        if not fired.get(task):
            raise AnalysisError(f"the merge covers {task}, which did not fire — a merge is for FIRED legs only")
        pin_log = {L.to_log_id(task, i) for i in pins[task]["ids"]}
        want_selected = set(C.ladder_log_ids(task, sel["selection"][task]["ids"]))
        selected_by_cell: Dict[str, set] = {}
        per_cell = {}
        merged_vals: Dict[str, Dict[str, float]] = {}
        merged_rows_by_cell: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for cell in cells:
            r = merged_cells[cell].get(task)
            if r is None:
                raise AnalysisError(f"the merge records no {task} for cell {cell!r}")
            st = stores[(cell, task)]
            if r["rung1_store_key"] != st.key or dict(r["rung1_shas"]) != dict(st.shas):
                raise AnalysisError(f"{cell}/{task}: the merge was recorded against rung-1 store {r['rung1_store_key']!r}, not the store being "
                                    f"analysed ({st.key!r}) — a stale or foreign merge cannot certify check 4")
            rung2_root = None
            if rung2_dir is not None:
                # the contributing rung-2 store, RE-ADMITTED now (its root's sampling-validation marker refuses through the loader) and
                # held to be the very store the merge read
                try:
                    st2 = L.load_store(rung2_dir, cell, task)
                except L.LadderError as e:
                    raise AnalysisError(f"{cell}/{task}: the contributing rung-2 store is not admissible — {e}") from e
                if st2.key != r["rung2_store_key"] or dict(st2.shas) != dict(r.get("rung2_shas") or {}):
                    raise AnalysisError(f"{cell}/{task}: the rung-2 store under {rung2_dir} ({st2.key!r}) is not the one the merge read "
                                        f"({r['rung2_store_key']!r}) — the merge cannot certify check 4 against it")
                rung2_root = str(st2.root)
            path = Path(r["merged_rows"])
            if not path.is_file():
                raise AnalysisError(f"{cell}/{task}: merged rows {path} are missing")
            sha = L.sha256_file(path)
            if r.get("merged_rows_sha256") != sha:
                raise AnalysisError(f"{cell}/{task}: merged rows {path} hash {sha[:12]}… != the merge's recorded "
                                    f"{str(r.get('merged_rows_sha256'))[:12]}… — the rows were edited since the merge")
            rows = L.acc_rows_of(L.read_jsonl(path), task, where=str(path))
            if set(rows) != pin_log:
                raise AnalysisError(f"{cell}/{task}: merged rows are not exactly the {FULL_N} pinned items")
            vals = _values(rows, where=str(path))
            merged_vals[cell] = vals
            merged_rows_by_cell[cell] = rows
            selected = {i for i, x in rows.items() if x["extra"].get("cap_rung") == 2}
            rung1_rows = legs[task]["_rows"][cell]          # parsed once by `leg_section`, never re-read
            recomputed = {"k_merged": int(sum(vals.values())), "k_rung1": sum(1 for x in rung1_rows.values() if float(x["value"]) == 1.0),
                          "n_at_cap_rung1": int(st.summary["n_at_cap"]), "n_selected": len(selected),
                          "n_at_cap_residual": sum(1 for i in selected if rows[i]["extra"].get("stop_reason") == L.STOP_REASON_AT_CAP), "n": len(rows)}
            bad = {k: (r.get(k), v) for k, v in recomputed.items() if r.get(k) != v}
            if bad:
                raise AnalysisError(f"{cell}/{task}: the merge's summary disagrees with its rows / the current rung-1 store: {bad} (recorded, recomputed)")
            if selected != want_selected:
                raise AnalysisError(f"{cell}/{task}: the merge's rung-2 rows are not exactly the selection's {len(want_selected)} items")
            selected_by_cell[cell] = selected
            n_sel, residual = recomputed["n_selected"], recomputed["n_at_cap_residual"]
            # a capped output KEEPS its benchmark score — the residual capped items that the scorer still passed are
            # counted from the merged rows (six of the twelve residual capped Llama-LoRA BFCL outputs), so the doc
            # can say so with the store's number rather than a hand-typed one
            residual_pass = sum(1 for i in selected if rows[i]["extra"].get("stop_reason") == L.STOP_REASON_AT_CAP and vals[i] == 1.0)
            per_cell[cell] = {"rung1": pass_rate(recomputed["k_rung1"], FULL_N), "merged": pass_rate(recomputed["k_merged"], FULL_N),
                              "n_selected": n_sel, "n_at_cap_rung1": recomputed["n_at_cap_rung1"], "n_at_cap_residual": residual,
                              "n_at_cap_residual_pass": residual_pass,
                              "residual_share": residual / FULL_N, "residual_share_of_selected": (residual / n_sel) if n_sel else None,
                              "n_model_length_rung2": sum(1 for i in selected if rows[i]["extra"].get("stop_reason") == L.STOP_REASON_MODEL_LENGTH),
                              "cap_from": r["cap_from"], "cap_to": r["cap_to"], "rung1_store_key": st.key, "rung2_store_key": r["rung2_store_key"],
                              "rung2_root": rung2_root, "rung2_root_readmitted": rung2_root is not None,
                              "merged_rows": str(path), "merged_rows_sha256": sha}
        if len({frozenset(v) for v in selected_by_cell.values()}) != 1:
            raise AnalysisError(f"{task}: the rung-2 item sets differ across cells")
        if base not in merged_vals:
            raise AnalysisError(f"merged summary cells {sorted(merged_cells)} carry no `{base}`")
        paired_merged = {}
        for cell in cells:
            if cell == base:
                continue
            paired_merged[cell] = paired_section(merged_vals[base], merged_vals[cell])
            paired_merged[cell]["reading"] = reading_rule(paired_merged[cell])
        per_task[task] = {"cells": per_cell, "paired_merged": paired_merged, "paired_rung1": legs[task]["paired"],
                          "n_selected": sel["selection"][task]["n_selected"], "max_prompt_tokens_rung1": sel["selection"][task]["max_prompt_tokens_rung1"],
                          "_rows": merged_rows_by_cell}
    missing = [t for t, f in fired.items() if f and t not in per_task]
    if missing:
        not_final.append(f"{', '.join(TASK_SHORT[t] for t in missing)} fired but not merged")
    out.update({"applied": not not_final, "merge_dir": str(Path(merge_dir).expanduser()), "selection_sha256": merged["selection_sha256"],
                "accepted": accepted, "rung_source": rung_source, "operator_rung": rung_source == "operator",
                "n_walked": rec.get("n_walked"), "certificates": certs, "fitted_without_certificate": missing_certs, "tasks": per_task,
                "fired_without_merge": missing, "final_reason": ("; ".join(not_final) or None),
                "acceptance_check_4": ("met (fired legs re-run at 16,384 and merged; every fitted cell certified under the wave record)"
                                       if not not_final else "NOT MET (" + "; ".join(not_final) + ")")})
    return out


def _find_files(root: Path, name: str) -> List[Path]:
    return sorted(p for p in root.rglob(name) if p.is_file())


def sandbox_section(rung_dirs: Sequence[Union[str, Path]], *, cells: Sequence[str]) -> Dict[str, Any]:
    """Per cell: the durable sandbox record (`watch/sandboxes_<fcid>.jsonl`),
    lifetimes ending at the FIRST observed termination (a sandbox's exit is
    observed twice on the 200-item legs — the sample's cleanup poll and the
    task-level re-list — and the later one is the task's end, not the
    sandbox's), the sweep certificates, `sandbox_failures.json`, and the
    resource-rate dollar estimate from lifetimes × the pinned compose's cpu / memory."""
    out: Dict[str, Any] = {}
    cpu = float(C.BCB_SANDBOX_RESOURCES["cpu"])
    mem_gib = float(C.BCB_SANDBOX_RESOURCES["memory_mib"]) / 1024.0
    for cell in cells:
        events: List[Dict[str, Any]] = []
        files: List[str] = []
        sweeps: List[Dict[str, Any]] = []
        failures: List[Any] = []
        for rd in rung_dirs:
            root = Path(rd).expanduser() / cell
            if not root.is_dir():
                continue
            for f in _find_files(root, "sandboxes*.jsonl"):
                files.append(str(f))
                for e in read_jsonl(f):
                    for key in ("event", "sandbox_id"):
                        if key not in e:
                            raise AnalysisError(f"{f}: a sandbox event without `{key}` (keys {sorted(e)}) — the record's shape moved")
                    if "ts" not in e and "at" not in e:
                        raise AnalysisError(f"{f}: a sandbox event without `ts` (keys {sorted(e)}) — the record's shape moved")
                    events.append(e)
            for f in _find_files(root / "watch" if (root / "watch").is_dir() else root, "sweep_*.json"):
                sweeps.append({"file": str(f), **{k: v for k, v in read_json(f).items() if k != "owner"}})
            for f in _find_files(root, "sandbox_failures.json"):
                doc = read_json(f)
                failures.extend(doc if isinstance(doc, list) else [doc])
        by_event = _counts(e["event"] for e in events)
        created: Dict[str, _dt.datetime] = {}
        ended: Dict[str, _dt.datetime] = {}
        verified_by: Dict[str, int] = {}
        n_duplicate_terminated = 0
        for e in events:
            stamp = e.get("ts", e.get("at"))
            t = _parse_dt(stamp)
            if t is None:
                raise AnalysisError(f"sandbox event {e['sandbox_id']!r} carries an unparseable timestamp {stamp!r}")
            ev = str(e["event"])
            if ev == "created":
                created[e["sandbox_id"]] = t
            elif ev.startswith("terminated"):
                sid = e["sandbox_id"]
                if sid in ended:
                    n_duplicate_terminated += 1
                    ended[sid] = min(ended[sid], t)
                else:
                    ended[sid] = t
                vb = str(e.get("verified_by") or "unstated")
                verified_by[vb] = verified_by.get(vb, 0) + 1
        lifetimes = [(ended[s] - created[s]).total_seconds() for s in created if s in ended]
        unaccounted = sorted(s for s in created if s not in ended)
        core_h = sum(lifetimes) / 3600.0 * cpu
        gib_h = sum(lifetimes) / 3600.0 * mem_gib
        out[cell] = {"files": files, "n_events": len(events), "by_event": by_event, "n_created": len(created), "n_terminated": len(ended),
                     "terminated_verified_by": verified_by, "n_duplicate_terminated_events": n_duplicate_terminated,
                     "n_unaccounted": len(unaccounted), "unaccounted_ids": unaccounted[:20],
                     "n_terminate_unverified": by_event.get("terminate_unverified", 0),
                     "n_cleanup_skipped_interrupted": by_event.get("cleanup_skipped_interrupted", 0),
                     "lifetime_s": A.summary_stats(lifetimes), "cpu": cpu, "memory_gib": mem_gib, "core_hours": core_h, "gib_hours": gib_h,
                     "usd_estimate": core_h * SANDBOX_USD_PER_CORE_H + gib_h * SANDBOX_USD_PER_GIB_H,
                     "sweeps": sweeps, "n_failures": len(failures), "failures": failures[:20],
                     "basis": f"lifetimes created→FIRST observed termination × ({cpu} core × ${SANDBOX_USD_PER_CORE_H}/core-h + "
                              f"{mem_gib:.2f} GiB × ${SANDBOX_USD_PER_GIB_H}/GiB-h) — a resource-rate estimate, not billing"}
    return out


def wall_section(rung_dirs: Sequence[Union[str, Path]], *, cells: Sequence[str], stores: Mapping[Tuple[str, str], Any],
                 tasks: Sequence[str]) -> Dict[str, Any]:
    """Per cell: the earliest attempt pointer → the latest marker `finished_at` of every supplied rung."""
    out: Dict[str, Any] = {}
    for cell in cells:
        starts: List[_dt.datetime] = []
        pointers: List[str] = []
        standup_s: List[float] = []
        for rd in rung_dirs:
            root = Path(rd).expanduser() / cell
            if not root.is_dir():
                continue
            for f in _find_files(root, "attempt_*.json"):
                doc = read_json(f)
                pointers.append(str(f))
                for key in ("at", "started_at", "created_at"):
                    t = _parse_dt(doc.get(key)) if isinstance(doc, Mapping) else None
                    if t is not None:
                        starts.append(t)
                        break
                if isinstance(doc, Mapping) and isinstance(doc.get("standup_s"), (int, float)):
                    standup_s.append(float(doc["standup_s"]))
            sg = root / "served_geometry.json"
            if sg.is_file():
                doc = read_json(sg)
                if isinstance(doc, Mapping) and isinstance(doc.get("standup_s"), (int, float)):
                    standup_s.append(float(doc["standup_s"]))
        ends = []
        per_task = {}
        for task in tasks:
            st = stores.get((cell, task))
            if st is None:
                continue
            fa = _parse_dt(st.marker.get("finished_at"))
            if fa is not None:
                ends.append(fa)
            per_task[task] = {"wall_clock_s": st.summary.get("wall_clock_s"), "marker_wall_clock_s": st.marker.get("wall_clock_s"), "finished_at": _iso(fa)}
        later_rungs = {}
        for rd in list(rung_dirs)[1:]:
            for task in tasks:
                try:
                    st2 = L.load_store(rd, cell, task)
                except L.LadderError:
                    continue
                fa = _parse_dt(st2.marker.get("finished_at"))
                if fa is not None:
                    ends.append(fa)
                later_rungs[f"{Path(rd).expanduser().name}/{task}"] = {"marker": st2.marker_kind, "finished_at": _iso(fa), "wall_clock_s": st2.summary.get("wall_clock_s")}
        start = min(starts) if starts else None
        end = max(ends) if ends else None
        out[cell] = {"start": _iso(start), "end": _iso(end), "hours": ((end - start).total_seconds() / 3600.0) if start and end else None,
                     "tasks_wall_s": sum(float(v["wall_clock_s"] or 0.0) for v in per_task.values()) + sum(float(v["wall_clock_s"] or 0.0) for v in later_rungs.values()),
                     "per_task": per_task, "later_rungs": later_rungs, "standup_s": standup_s, "attempt_pointers": pointers,
                     "note": ("wall from the first attempt pointer to the last marker of every supplied rung" if start else
                              "no attempt pointer under watch/ — only the tasks' own wall_clock_s is known")}
    return out


def gpu_cost_section(ledger_path: Union[str, Path], *, wave: Optional[str]) -> Dict[str, Any]:
    """H200-hours from the meter's ledger: THIS wave's reservations (by `wave`),
    the lane's Σ against the ceiling in force (25 h or the latest recorded
    decision), every row per attempt — actual where recorded, else forecast —
    at $5/h; a partial subtotal by construction (smokes and probes reserved under
    other waves or none are in the lane total, not the wave's)."""
    p = Path(ledger_path).expanduser()
    if not p.is_file():
        return {"ledger": str(p), "present": False, "stages": {}, "rows": [], "committed_h": None, "wave_h": None, "ceiling_h": None,
                "usd_estimate": None, "decisions": [], "by_wave": {}, "note": f"no meter ledger at {p} — GPU hours unknown"}
    entries = M.read_ledger(p)
    table = M.stage_table(entries)
    committed = M.committed_hours(table)
    ceil = M.ceiling_hours(entries)
    rows = []
    for stage, row in table.items():
        for n, a in sorted(row["attempts"].items()):
            superseded = a["actual"] is None and a.get("superseded_by") is not None
            rows.append({"stage": stage, "wave": row.get("wave"), "suffix": row.get("suffix"), "unit": row.get("unit") or stage,
                         "stage_kind": row.get("stage_kind"), "attempt": n, "label": (stage if len(row["attempts"]) == 1 else f"{stage} #{n}"),
                         "forecast": a["forecast"], "actual": a["actual"], "superseded_by": a.get("superseded_by"),
                         "released": bool(a.get("released")),
                         "counted": (0.0 if superseded else float(a["actual"] if a["actual"] is not None else (a["forecast"] or 0.0)))})
    wave_h = M.committed_hours(table, wave=wave) if wave else None
    return {"ledger": str(p), "present": True, "stages": table, "rows": rows, "committed_h": committed, "wave": wave, "wave_h": wave_h,
            "by_wave": M.hours_by_wave(table), "ceiling_h": ceil["ceiling_h"], "ceiling_source": ceil["source"],
            "remaining_h": ceil["ceiling_h"] - committed, "within_ceiling": committed <= ceil["ceiling_h"] + 1e-9,
            "usd_estimate": committed * H200_USD_PER_H, "wave_usd_estimate": (wave_h * H200_USD_PER_H if wave_h is not None else None),
            "usd_per_h": H200_USD_PER_H, "n_actual": sum(1 for r in rows if r["actual"] is not None),
            "n_forecast_only": sum(1 for r in rows if r["actual"] is None and r["superseded_by"] is None),
            "n_superseded": sum(1 for r in rows if r["superseded_by"] is not None and r["actual"] is None),
            "decisions": [e for e in entries if e["kind"] == "decision"],
            "note": f"H200-hours from the meter's ledger (actual where recorded, else forecast) × ${H200_USD_PER_H:.0f}/h — a resource-rate "
                    "estimate, not billing; a PARTIAL subtotal (the lane's other waves' reservations are in the lane Σ, not this wave's)"}


# --------------------------------------------------------------------------- #
# --analyze: the record                                                        #
# --------------------------------------------------------------------------- #


def analyze(rung1_dir: Union[str, Path], *, roster_path: Union[str, Path], wave_name: str, repo_root: Union[str, Path] = REPO,
            ladder_merge: Optional[Union[str, Path]] = None, selection: Optional[Union[str, Path]] = None,
            certificates: Sequence[Union[str, Path]] = (), ledger: Union[str, Path] = M.LEDGER_DEFAULT,
            rung2_dir: Optional[Union[str, Path]] = None, sandbox_state_root: Union[str, Path] = SANDBOX_STATE_ROOT,
            launch_roster_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    roster = load_roster(roster_path)
    validation = validate_roster(roster, roster_path)
    wave = wave_spec(roster, wave_name)                     # the CURRENT roster: publication and gating, as for every consumer
    # the stores are admitted under the roster that LAUNCHED them (`--launch-roster`; a store is still never analysed
    # under a label the roster did not launch) while the CURRENT roster gates what is published — so ONE canonical record carries the
    # withdrawal (`withdrawn_legs`, `final[leg].withdrawn`, the gated figures, the gated W&B scalars) instead of a render-time view over a
    # saved file that still read the leg as final. Without a launch roster (or with the same file) nothing changes.
    launch_path = Path(launch_roster_path).expanduser() if launch_roster_path else None
    launch_sha = roster_sha256(launch_path) if launch_path else None
    store_roster_path, store_roster_sha = roster_path, roster_sha256(roster_path)
    if launch_path is not None and launch_sha != store_roster_sha:
        launch_roster = load_roster(launch_path)
        validate_roster(launch_roster, launch_path)
        launch_wave = wave_spec(launch_roster, wave_name)
        differ = [k for k in ("family", "lane_kind", "view", "model_row", "served", "objects_dir") if getattr(launch_wave, k) != getattr(wave, k)]
        if differ:
            raise AnalysisError(f"--launch-roster {launch_path} and --roster {Path(roster_path).expanduser()} disagree on wave {wave_name!r}'s "
                                f"identity ({differ}) — not the same wave; nothing is analysed")
        wave, store_roster_path, store_roster_sha = launch_wave, launch_path, launch_sha
    run = load_run(rung1_dir, wave, repo_root=repo_root, roster_sha=store_roster_sha, roster_path=store_roster_path,
                   sandbox_state_root=sandbox_state_root)
    stores, pins, cells, tasks = run["stores"], run["pins"], run["cells"], run["tasks"]
    legs = {task: leg_section(task, {cell: stores[(cell, task)] for cell in cells}, pins[task], generations=run["generations"]) for task in tasks}
    ledger_entries = M.read_ledger(ledger) if Path(ledger).expanduser().is_file() else []
    rung_dirs = [rung1_dir] + ([rung2_dir] if rung2_dir else [])
    cert_paths, superseded_certs = find_certificates(rung_dirs if ladder_merge else [], explicit=certificates)
    admission = next((Path(rd).expanduser() / LADDER_SUBDIR / RUNG_ADMISSION_FILENAME for rd in rung_dirs
                      if (Path(rd).expanduser() / LADDER_SUBDIR / RUNG_ADMISSION_FILENAME).is_file()), None)
    ladder = ladder_section(legs, stores=stores, pins=pins, cells=cells, fitted_cells=wave.fitted_cells, merge_dir=ladder_merge,
                            selection_path=selection, certificate_paths=cert_paths, ledger_entries=ledger_entries, admission_path=admission,
                            superseded_certificates=superseded_certs, rung2_dir=rung2_dir)
    # the breakdowns and the quoted items are computed from each leg's FINAL per-item rows (the same rule as `final_readings`:
    # merged after an applied ladder, else rung 1) and carry their source; a pending leg's rung-1 breakdown is labelled NOT final
    finals = final_readings({"legs": legs, "ladder": ladder})
    rows_by_task: Dict[str, Tuple[Dict[str, Any], str, bool]] = {}
    for task in tasks:
        fin = finals[task]
        merged_rows = ((ladder.get("tasks") or {}).get(task) or {}).pop("_rows", None)
        if fin["source"] == FINAL_SOURCE_MERGED:
            if merged_rows is None:
                raise AnalysisError(f"{task}: the final reading is the merge but the merged rows were not retained")
            rows_by_task[task] = (merged_rows, FINAL_SOURCE_MERGED, True)
        else:
            rows_by_task[task] = (legs[task]["_rows"], FINAL_SOURCE_RUNG1, fin["final"])
    breakdowns: Dict[str, Any] = {}
    quoted: Dict[str, Any] = {}
    steered = list(wave.steered_cells)
    if BFCL_TASK in legs:
        rows, source, final = rows_by_task[BFCL_TASK]
        label = {"source": source, "final": final, "label": breakdown_label(source, final)}
        breakdowns["bfcl_by_category"] = {**bfcl_by_category(rows, pins[BFCL_TASK], base=BASE_CELL, steered=steered), **label}
        quoted["bfcl_discordant"] = {**label, "cells": bfcl_discordant(rows, pins[BFCL_TASK], base=BASE_CELL, steered=steered)}
    if BCB_TASK in legs:
        rows, source, final = rows_by_task[BCB_TASK]
        label = {"source": source, "final": final, "label": breakdown_label(source, final)}
        breakdowns["bcb_by_family"] = {**bcb_by_family(rows, pins[BCB_TASK], run["families"], base=BASE_CELL, steered=steered), **label}
        quoted["bcb_failures"] = {**label, "cells": bcb_failures(rows)}
    sandboxes = sandbox_section(rung_dirs, cells=cells) if BCB_TASK in tasks else {}
    wall = wall_section(rung_dirs, cells=cells, stores=stores, tasks=tasks)
    gpu = gpu_cost_section(ledger, wave=wave.name)
    sandbox_usd = sum(v["usd_estimate"] for v in sandboxes.values())
    cost = {"gpu": gpu, "sandboxes": sandboxes, "sandbox_usd_estimate": sandbox_usd,
            "wave_usd_estimate": ((gpu["wave_usd_estimate"] or 0.0) + sandbox_usd) if gpu["present"] else None,
            "lane_gpu_usd_estimate": gpu["usd_estimate"],
            "envelope": {"gpu_h_ceiling": gpu.get("ceiling_h"), "note": "spec: ≤ 25 H200-h for smokes + every runnable cell; + ≤ 1.5 H200-h per leg "
                                                                        "ONLY if the truncation rule fires (a recorded meter decision); sandbox / CPU-suite dollars separate"},
            "basis": "every dollar line is a resource-rate estimate, not billing; every subtotal is PARTIAL and names what it excludes"}
    manifests = {cell: stores[(cell, tasks[0])].manifest for cell in cells}
    base_m = manifests[BASE_CELL]
    identity_facts = {"model_id": base_m.get("model_id"), "model_revision": base_m.get("model_revision"), "engine": base_m.get("engine"),
                      "tree_sha": _dig(base_m, (C.MANIFEST_KEY_PRODUCER, "tree_sha")), "served": _dig(base_m, (C.MANIFEST_KEY_TINYCAP, "served")),
                      "sampling": base_m.get(C.MANIFEST_KEY_SAMPLING), "thinking": _dig(base_m, (C.MANIFEST_KEY_TINYCAP, "thinking")),
                      "caps": {task: legs[task]["cells"][BASE_CELL]["cap"] for task in legs},
                      "routes": {cell: _dig(m, (C.MANIFEST_KEY_ROUTE, "virtual_model")) for cell, m in manifests.items()},
                      "object_sha256": {cell: run["generations"][cell]["object_sha256"] for cell in cells},
                      "object_sha256_source": {cell: run["generations"][cell]["object_sha256_source"] for cell in cells},
                      "intervention_hash": {cell: m.get("intervention_hash") for cell, m in manifests.items()},
                      "checks": run["identity"], "common": run["identity_common"], "provenance": run["provenance"], "mode": run["mode"]}
    for leg in legs.values():
        leg.pop("_rows", None)
    entries_rec = [{"entry_id": e.entry_id, "cell_id": e.cell_id, "status": e.status, "kind": e.kind, "paper_arm": e.paper_arm,
                    "object_generation_roster": e.object_generation, "reason": e.reason, "unblock": e.unblock, "note": e.note,
                    "rerun_trigger": e.rerun_trigger, W.POOLING_KEY: e.pooling} for e in wave.entries]
    out = {"schema": ANALYSIS_SCHEMA_VERSION, "generator": "scripts/bcb_bfcl_200_analysis.py", "lane": LANE,
           "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(), "run_name": run["suffix"], "rung1_dir": run["rung1_dir"],
           "rung2_dir": (str(Path(rung2_dir).expanduser()) if rung2_dir else None),
           "wave": {"name": wave.name, "family": wave.family, "lane_kind": wave.lane_kind, "view": wave.view, "model_row": wave.model_row,
                    "served": wave.served, "objects_dir": wave.objects_dir, "legs": wave.legs, "entries": entries_rec, "sampling": wave.sampling,
                    "sampling_clamp_override": wave.sampling_clamp_override, "thinking": wave.thinking,
                    "roster": str(Path(roster_path).expanduser()), "roster_sha256": roster_sha256(roster_path),
                    # the roster the stores were admitted under (None = the current one); the record's entries / legs are ITS, with
                    # every status refreshed to the current roster by the gate (`status_at_analysis` kept)
                    "launch_roster": (str(launch_path) if launch_path else None), "launch_roster_sha256": launch_sha, **validation},
           "cells": list(cells), "base": BASE_CELL, "steered": list(run["steered"]), "fitted": list(wave.fitted_cells), "tasks": list(tasks),
           "blocked_legs": wave.blocked_legs, "identity": identity_facts, "generations": run["generations"], "fit_layers": run["fit_layers"],
           "protocol": run["protocol"], "legs": legs, "breakdowns": breakdowns, "quoted": quoted, "ladder": ladder, "cost": cost, "wall": wall,
           "anchors": {"note": FULL22_ANCHORS_NOTE, "rows": {f"{fam}/{arm}": v for (fam, arm), v in FULL22_ANCHORS.items() if fam == wave.family}},
           "design": {"k": 1, "sampling": run["protocol"]["decode_by_task"], "thinking": identity_facts["thinking"], "paired": True,
                      "bootstrap": {"B": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED},
                      "note": "200 items at k = 1 give ≈ 3–4 pp standard errors per cell; the paired contrast (each steered cell vs its wave's "
                              "`none`) is the readout; no cross-object contrast is computed (D7); k = 4 is the recorded follow-up"}}
    out["final"] = final_readings(out)          # the ONE final-reading rule, recorded (every consumer re-derives it the same way)
    # stamped `publication`: gated against the CURRENT roster — with a launch roster that differs, a leg it launched and the current one
    # blocks is WITHDRAWN here, in the saved file; a later `--summary` / `--analysis` re-gates the SAVED record against the roster of that day
    view = gate_saved_analysis(out, roster, roster_path=roster_path)
    view[PUBLICATION_KEY].update({"launch_roster": (str(launch_path) if launch_path else None), "launch_roster_sha256": launch_sha})
    return view


def write_analysis(analysis: Mapping[str, Any], run_dir: Union[str, Path]) -> Path:
    out = Path(run_dir).expanduser() / ANALYSIS_FILENAME
    out.write_text(json.dumps(analysis, indent=1, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
# the FINAL reading of a leg — ONE selector for summaries, headlines, W&B,      #
# figures and paper rows                                                      #
# --------------------------------------------------------------------------- #

_PASS_RATE_KEYS: Tuple[str, ...] = ("k", "n", "rate", "wilson_low", "wilson_high")


def _refresh_paired(paired: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Copies of the paired blocks with `reading` and `cost_detected` RE-DERIVED from the numbers: the reading
    string is a pure function of the block (`reading_rule`), so a saved analysis renders the current rule — the p format,
    the sparse-discordance sentence, the decision bit — without its file being rewritten; a fresh analysis is unchanged."""
    out = {}
    for c, p in paired.items():
        q = dict(p)
        q["cost_detected"] = cost_detected(q)
        q["reading"] = reading_rule(q)
        out[c] = q
    return out


def final_readings(analysis: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Per leg, WHAT THE WAVE'S FINAL READING IS — the one rule every consumer
    reads (summaries, the headline, W&B, figures, paper rows): the MERGED reading
    when the truncation ladder was applied to the leg (fired, re-run at 16,384,
    merged, every fitted cell certified under the wave record); the RUNG-1 reading
    when the leg did not fire (it stands alone); NO final reading (``final`` False,
    null measurements) when the leg fired and the ladder is not applied — pending.
    Rung-1 numbers appear anywhere else ONLY under an explicit "rung 1" label."""
    lad = analysis.get("ladder") or {}
    applied_tasks = (lad.get("tasks") or {}) if lad.get("applied") else {}
    withdrawn = analysis.get(WITHDRAWN_LEGS_KEY) or {}
    out: Dict[str, Dict[str, Any]] = {}
    for task, leg in analysis["legs"].items():
        fired = bool(leg["fired"])
        if task in applied_tasks:
            t = applied_tasks[task]
            rec = {"source": FINAL_SOURCE_MERGED, "final": True,
                   "cells": {c: {k: r["merged"][k] for k in _PASS_RATE_KEYS} for c, r in t["cells"].items()},
                   "paired": _refresh_paired(t["paired_merged"]),
                   "cap_by_cell": {c: int(r["cap_to"]) for c, r in t["cells"].items()},
                   "at_cap_share_by_cell": {c: r["n_at_cap_residual"] / r["merged"]["n"] for c, r in t["cells"].items()},
                   "why": "the leg fired; its capped items were re-run at 16,384 and merged per item, certified per fitted cell at the "
                          "wave's common rung under the wave record — the merged reading is final"}
        elif not fired:
            rec = {"source": FINAL_SOURCE_RUNG1, "final": True,
                   "cells": {c: {k: r[k] for k in _PASS_RATE_KEYS} for c, r in leg["cells"].items()},
                   "paired": _refresh_paired(leg["paired"]),
                   "cap_by_cell": {c: int(r["cap"]) for c, r in leg["cells"].items()},
                   "at_cap_share_by_cell": {c: r["at_cap"]["n_at_cap"] / r["at_cap"]["n_stop_reason"] for c, r in leg["cells"].items()},
                   "why": "the truncation rule did not fire: the rung-1 reading stands alone and is final"}
        else:
            rec = {"source": FINAL_SOURCE_PENDING, "final": False, "cells": None, "paired": None, "cap_by_cell": None,
                   "at_cap_share_by_cell": None,
                   "why": ("the leg FIRED and the ladder is not applied — no final reading; " +
                           str(lad.get("final_reason") or lad.get("reason") or lad.get("acceptance_check_4") or "ladder pending"))}
        if task in withdrawn:
            # the CURRENT roster blocks a leg this wave measured — WITHDRAWN. The numbers that would have been the reading
            # (the merged / rung-1 ones; a pending leg's rung-1 ones, labelled so) are kept verbatim under `withdrawn` for the record
            # and nowhere a consumer reads a reading from (`cells` / `paired` are null, `final` False)
            wd = withdrawn[task]
            kept_cells = rec["cells"] if rec["final"] else {c: {k: r[k] for k in _PASS_RATE_KEYS} for c, r in leg["cells"].items()}
            kept_paired = rec["paired"] if rec["final"] else _refresh_paired(leg["paired"])
            rec = {"source": FINAL_SOURCE_WITHDRAWN, "final": False, "cells": None, "paired": None, "cap_by_cell": None,
                   "at_cap_share_by_cell": None,
                   "withdrawn": {"status": wd.get("status"), "reason": wd.get("reason"), "unblock": wd.get("unblock"),
                                 "roster_sha256": wd.get("roster_sha256"),
                                 "cells": kept_cells, "paired": kept_paired,
                                 "kept_from": (rec["source"] if rec["final"] else f"{FINAL_SOURCE_RUNG1} (the leg fired; no final reading was reached)"),
                                 "cap_by_cell": rec["cap_by_cell"] if rec["final"] else {c: int(r["cap"]) for c, r in leg["cells"].items()}},
                   "why": (f"WITHDRAWN as a capability reading (D79): the current roster marks this leg {wd.get('status')} — {wd.get('reason')}; "
                           "the store's numbers are kept under `withdrawn` for the record, never as a reading")}
        out[task] = rec
    return out


def residual_capped_passes(ladder_cell: Mapping[str, Any], *, task: str) -> Optional[int]:
    """How many of a merged cell's RESIDUAL capped items (still at the cap at 16,384) the scorer passed: the
    analysis-time field when the record carries it, else read from the merged rows the record names — the merged store
    is the authoritative source and its sha is held to the record's — else None (stated as not recorded, never guessed)."""
    if ladder_cell.get("n_at_cap_residual_pass") is not None:
        return int(ladder_cell["n_at_cap_residual_pass"])
    path = Path(str(ladder_cell.get("merged_rows") or ""))
    if not ladder_cell.get("merged_rows") or not path.is_file() or L.sha256_file(path) != ladder_cell.get("merged_rows_sha256"):
        return None
    rows = L.acc_rows_of(L.read_jsonl(path), task, where=str(path))
    return sum(1 for r in rows.values() if r["extra"].get("cap_rung") == 2 and r["extra"].get("stop_reason") == L.STOP_REASON_AT_CAP
               and float(r["value"]) == 1.0)


def final_reading(analysis: Mapping[str, Any], task: str) -> Dict[str, Any]:
    return final_readings(analysis)[task]


def breakdown_label(source: str, final: bool) -> str:
    """How a breakdown / quoted section names the rows it was computed from: ``final: merged``,
    ``final: rung 1``, or ``rung 1 — NOT final (ladder pending)``; images and tables carry it."""
    if final:
        return "final: merged" if source == FINAL_SOURCE_MERGED else "final: rung 1"
    return "rung 1 — NOT final (ladder pending)"


# --------------------------------------------------------------------------- #
# publication eligibility — a SAVED analysis is gated by the CURRENT roster:  #
# a measured cell whose roster entry is now blocked, queued or gone is not a  #
# result. Its measurements are suppressed in the summary, the doc and the        #
# figures — named "out" with the status and reason, never silently dropped —   #
# and the paper export REFUSES. The saved file is never rewritten: the gate     #
# returns a publication VIEW (a deep copy).                                   #
# --------------------------------------------------------------------------- #

PUBLICATION_KEY = "publication"
#: `publication.held_out` = {cell: pooling block}: the measured cells the CURRENT roster holds out of pooled category means (W.POOLING_KEY)
HELD_OUT_KEY = "held_out"
OUT_SOURCE = "out"
#: the branches of an analysis that carry per-cell MEASUREMENTS (pruned for an out cell); identity, generations, fit
#: layers, cost, wall and the wave record are facts about what ran and stay (the wave record's entry carries the CURRENT status)
MEASUREMENT_BRANCHES: Tuple[str, ...] = ("legs", "breakdowns", "quoted", "ladder")
ABSENT_STATUS = "absent"
ABSENT_LABEL = "out (not in the current roster)"


def publication_eligibility(analysis: Mapping[str, Any], roster: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """``{cell: {entry_id, status, reason, unblock, label}}`` for every MEASURED cell of the analysis that the CURRENT
    roster does not list as runnable in its wave — blocked / queued, or no entry names the cell (`absent`). The wave
    itself missing from the roster, or its `none` anchor not runnable, refuses: nothing of the analysis has a home."""
    wname = analysis["wave"]["name"]
    if wname not in (roster.get("waves") or {}):
        raise AnalysisError(f"the current roster has no wave {wname!r}: the saved analysis {analysis.get('run_name')!r} cannot be published against it")
    by_cell = {e.get("cell_id"): (eid, e) for eid, e in W.entries(roster, wname).items() if e.get("cell_id")}
    measured = list((analysis.get(PUBLICATION_KEY) or {}).get("cells_at_analysis") or analysis["cells"])
    out: Dict[str, Dict[str, Any]] = {}
    for cell in measured:
        hit = by_cell.get(cell)
        if hit is None:
            out[cell] = {"entry_id": None, "status": ABSENT_STATUS, "reason": "no entry of the current roster names this cell", "unblock": None,
                         "label": ABSENT_LABEL}
        elif hit[1].get("status") != "runnable":
            eid, e = hit
            out[cell] = {"entry_id": eid, "status": e.get("status"), "reason": e.get("reason"), "unblock": e.get("unblock"),
                         "label": roster_status_label(e.get("status"), e.get("reason"))}
    base = analysis.get("base", BASE_CELL)
    if base in out:
        raise AnalysisError(f"wave {wname!r}: the current roster does not run its `{base}` anchor ({out[base]['status']}: {out[base]['reason']}) — "
                            f"no contrast of the saved analysis {analysis.get('run_name')!r} can be published")
    return out


def withdrawn_legs(analysis: Mapping[str, Any], roster: Mapping[str, Any], *, roster_sha256: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """``{task: {status, reason, unblock, roster_sha256}}`` for every leg the analysis MEASURED (a `legs` entry — stores and
    readings exist) that the CURRENT roster marks not runnable in its wave: the leg is WITHDRAWN as a capability
    reading. Keyed on the roster's leg status alone — the wave's `legs` block — never inferred from the numbers."""
    wname = analysis["wave"]["name"]
    out: Dict[str, Dict[str, Any]] = {}
    for atom, (status, detail) in W.legs(roster, wname).items():
        task = TASK_OF_ATOM.get(atom, atom)
        if status != W.STATUS_RUNNABLE and task in (analysis.get("legs") or {}):
            out[task] = {"status": status, "reason": detail.get("reason"), "unblock": detail.get("unblock"), "roster_sha256": roster_sha256}
    return out


def _prune_cells(node: Any, cells: Sequence[str]) -> Any:
    """Every dict key and every list string naming one of ``cells`` removed, recursively (the measurement branches only)."""
    drop = set(cells)
    if isinstance(node, dict):
        return {k: _prune_cells(v, cells) for k, v in node.items() if k not in drop}
    if isinstance(node, list):
        return [_prune_cells(v, cells) for v in node if not (isinstance(v, str) and v in drop)]
    return node


def gate_saved_analysis(analysis: Mapping[str, Any], roster: Mapping[str, Any], *, roster_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """The PUBLICATION VIEW of an analysis under the CURRENT roster — a deep copy; the record handed in (and the saved
    file) is untouched. ``view["publication"]`` = {roster, roster_sha256, gated_at, out: {cell: {entry_id, status,
    reason, unblock, label}}, cells_at_analysis, steered_at_analysis, fitted_at_analysis, unmeasured_runnable}. Every
    out cell is removed from `cells` / `steered` / `fitted` and from the measurement branches (legs, breakdowns,
    quoted, ladder), `final` is re-derived, and the wave record's entry for it carries the CURRENT status (its status
    at analysis kept as `status_at_analysis`). Idempotent: a view whose stamp already names the same out set is
    returned as a copy; a stamp from another roster (the record's own, made when the cell was runnable) is redone from
    `cells_at_analysis`; a view pruned by an earlier gate cannot be widened again — reload the saved file."""
    stamp = dict(analysis.get(PUBLICATION_KEY) or {})
    out = publication_eligibility(analysis, roster)
    sha = roster_sha256(roster_path) if roster_path else stamp.get("roster_sha256")
    wname = analysis["wave"]["name"]
    # a measured LEG the current roster no longer runs is withdrawn (the numbers kept for the record under `final`)
    withdrawn = withdrawn_legs(analysis, roster, roster_sha256=sha)
    view: Dict[str, Any] = copy.deepcopy(dict(analysis))
    cells_at = list(stamp.get("cells_at_analysis") or analysis["cells"])
    steered_at = list(stamp.get("steered_at_analysis") or analysis["steered"])
    fitted_at = list(stamp.get("fitted_at_analysis") or analysis["fitted"])
    unmeasured = [c for c in W.runnable_cells(roster, wname) if c not in cells_at]
    # EVERY entry of the wave record carries its CURRENT roster status (status at analysis kept beside it), not only the out
    # cells — the roster figure and the doc's roster table state eligibility under the current roster, and a saved record's
    # not-run entries were showing their analysis-time status (three gemma-it PSR entries read `queued` after being blocked)
    current_entries = W.entries(roster, wname)
    # the CURRENT roster's pooling declaration: a measured, eligible cell whose entry holds it out of pooled category means (reported, stamped)
    held = {c: blk for c, blk in W.held_out_cells(roster, wname).items() if c in cells_at and c not in out}
    for e in view["wave"]["entries"]:
        cur = current_entries.get(str(e.get("entry_id")))
        if cur is None:
            continue
        e[W.POOLING_KEY] = cur.get(W.POOLING_KEY)         # a publication decision: the current roster's, never the launch-time record's
        if (e.get("status"), e.get("reason"), e.get("unblock")) != (cur.get("status"), cur.get("reason"), cur.get("unblock")):
            e.setdefault("status_at_analysis", e.get("status"))
            e.update(status=cur.get("status"), reason=cur.get("reason"), unblock=cur.get("unblock"))
    if stamp and stamp.get("out") == out and (stamp.get(WITHDRAWN_LEGS_KEY) or {}) == withdrawn:
        view[WITHDRAWN_LEGS_KEY] = withdrawn                  # present on every view, empty or not (an older record lacks it)
        view["final"] = final_readings(view)                  # the readings re-derived under the current rule
        view[PUBLICATION_KEY] = {**stamp, "roster": (str(Path(roster_path).expanduser()) if roster_path else stamp.get("roster")),
                                 "roster_sha256": sha, "unmeasured_runnable": unmeasured, WITHDRAWN_LEGS_KEY: withdrawn, HELD_OUT_KEY: held}
        return view
    view["cells"] = [c for c in cells_at if c not in out]
    view["steered"] = [c for c in steered_at if c not in out]
    view["fitted"] = [c for c in fitted_at if c not in out]
    for task, leg in view["legs"].items():
        missing = [c for c in view["cells"] if c not in (leg.get("cells") or {})]
        if missing:
            raise AnalysisError(f"{analysis.get('run_name')!r} {task}: cells {missing} are eligible under the current roster but were pruned by an "
                                "earlier gate of this view — reload the saved analysis.json and gate it once")
    figures_at_analysis = stamp.get("figures_at_analysis")
    if out:
        for branch in MEASUREMENT_BRANCHES:
            if view.get(branch) is not None:
                view[branch] = _prune_cells(view[branch], list(out))
        for e in view["wave"]["entries"]:
            rec = out.get(e.get("cell_id"))
            if rec is not None:
                e.setdefault("status_at_analysis", e.get("status"))
                e.update(status=rec["status"], reason=rec["reason"], unblock=rec["unblock"])
    # the wave record's leg block carries the CURRENT status of a withdrawn leg (its status at analysis kept beside it);
    # the leg's measurements stay in `legs` / `ladder` (diagnostics and the record) — `final_readings` files them under `withdrawn`
    view[WITHDRAWN_LEGS_KEY] = withdrawn
    for task, wd in withdrawn.items():
        blk = (view["wave"].get("legs") or {}).get(task)
        if isinstance(blk, dict):
            blk.setdefault("status_at_analysis", blk.get("status"))
            blk.update(status=wd["status"], reason=wd["reason"], unblock=wd["unblock"])
    if out or withdrawn:
        # the analysis-time figures were drawn WITH the out cells / the withdrawn leg as a reading: they are provenance, never linked
        # under this roster. The view carries a WITHHELD marker (no url_base, so the renderer prints no fallback URL) until
        # `regenerate_gated_figures` draws the gated view under its own namespace
        figs = view.get("figures")
        if figs is not None:                                  # a fresh record has none yet: `--analyze` draws them FROM the gated record
            if not figs.get("withheld") and not figs.get("gated"):
                figures_at_analysis = figs
            view["figures"] = withheld_figures(out, sha, withdrawn=withdrawn)
    view["final"] = final_readings(view)
    view[PUBLICATION_KEY] = {"roster": (str(Path(roster_path).expanduser()) if roster_path else stamp.get("roster")), "roster_sha256": sha,
                             "gated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(), "out": out, WITHDRAWN_LEGS_KEY: withdrawn,
                             "cells_at_analysis": cells_at, "steered_at_analysis": steered_at, "fitted_at_analysis": fitted_at,
                             "unmeasured_runnable": unmeasured, "figures_at_analysis": figures_at_analysis, HELD_OUT_KEY: held,
                             # the roster the stores were admitted under rides every re-gate of the record (None = the roster of the analysis)
                             "launch_roster": stamp.get("launch_roster"), "launch_roster_sha256": stamp.get("launch_roster_sha256")}
    return view


def _withdrawn_reason_text(reason: Any) -> str:
    """``instrument failure: <reason>`` — unless the roster's reason already names the category (otherwise the cross-wave rows
    read "instrument failure: instrument failure: …")."""
    r = str(reason or "")
    return r if r.lower().startswith("instrument failure") else f"instrument failure: {r}"


def withdrawn_record_location(view: Mapping[str, Any]) -> str:
    """Where `final.<leg>.withdrawn` actually lives for this view: the SAVED analysis.json when the record was analysed with a
    launch roster under the current roster (the canonical record), else this publication view only — the saved file predates the
    withdrawal and still reads the leg as final; naming that is the honest note, never a pointer to a record that does not exist."""
    pub = view.get(PUBLICATION_KEY) or {}
    saved = f"{view.get('rung1_dir')}/{ANALYSIS_FILENAME}"
    if pub.get("launch_roster_sha256"):
        return (f"`{saved}` (analysed with `--launch-roster` {str(pub['launch_roster_sha256'])[:8]} under the current roster "
                f"{str(pub.get('roster_sha256') or 'unknown')[:8]})")
    return (f"this publication view only — the saved `{saved}` predates the withdrawal and still reads the leg as final; re-run `--analyze "
            "--launch-roster <the roster that launched the stores> --roster <the current roster>` to persist it")


def withheld_figures(out: Mapping[str, Any], roster_sha256: Optional[str], *, withdrawn: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """The `figures` block of a gated view whose analysis-time figures may not be linked: drawn while ``out`` was runnable
    and / or while a now-``withdrawn`` leg was read as a reading."""
    drawn_while = [f"{', '.join(f'`{c}`' for c in out)} was runnable"] if out else []
    if withdrawn:
        drawn_while.append(f"leg(s) {', '.join(TASK_SHORT.get(t, t) for t in withdrawn)} were read as readings (now withdrawn)")
    return {"withheld": True, "out": list(out), WITHDRAWN_LEGS_KEY: list(withdrawn or {}),
            "why": (f"this analysis's figures were drawn while {' and '.join(drawn_while)}; they are provenance and are not "
                    f"linked under the current roster (sha `{str(roster_sha256 or 'unknown')[:8]}`). Regenerated from the gated view by "
                    "`--analysis` / `--summary` with figures enabled (under `<figs-root>/<suffix>/gated-<roster sha8>/`), or on the next `--analyze`")}


def gated_figs_dir(view: Mapping[str, Any], figs_root: Union[str, Path]) -> Path:
    """``<figs-root>/<suffix>/gated-<roster sha8>/`` — the publication namespace of a gated view's figures, never the
    analysis-time `<figs-root>/<suffix>/` (which stays untouched as provenance)."""
    sha = str((view.get(PUBLICATION_KEY) or {}).get("roster_sha256") or "unknown")[:8]
    return Path(figs_root).expanduser() / str(view["run_name"]) / f"gated-{sha}"


def regenerate_gated_figures(view: Dict[str, Any], *, figs_root: Union[str, Path], figs_url_base: str = DEFAULT_FIGS_URL_BASE) -> Optional[Dict[str, Any]]:
    """When the gate withheld a view's figures, draw the wave's figures FROM THE GATED VIEW into :func:`gated_figs_dir` and point
    ``view["figures"]`` at them (``gated`` names the roster sha and the out cells); a view whose figures are not withheld is left
    alone (None). The analysis-time assets and the saved analysis.json are not touched."""
    figs = view.get("figures") or {}
    if not figs.get("withheld"):
        return None
    figs_dir = gated_figs_dir(view, figs_root)
    files = make_figures(view, figs_dir)
    sha = (view.get(PUBLICATION_KEY) or {}).get("roster_sha256")
    view["figures"] = {"dir": str(figs_dir), "files": files, "url_base": f"{figs_url_base.rstrip('/')}/{view['run_name']}/{figs_dir.name}",
                       "gated": {"roster_sha256": sha, "out": list((view.get(PUBLICATION_KEY) or {}).get("out") or {}),
                                 WITHDRAWN_LEGS_KEY: list((view.get(PUBLICATION_KEY) or {}).get(WITHDRAWN_LEGS_KEY) or {})}}
    return view["figures"]


def require_publication_gate(analysis: Mapping[str, Any], *, what: str) -> None:
    """``what`` (the paper export) publishes ONLY a gated view with nothing out: an ungated record refuses (gate it
    against the current roster first), and a view with an out cell refuses NAMING the cell, its status and the reason."""
    stamp = analysis.get(PUBLICATION_KEY)
    if stamp is None:
        raise AnalysisError(f"{what}: analysis {analysis.get('run_name')!r} is not gated against the current roster (no `{PUBLICATION_KEY}` "
                            "record) — gate_saved_analysis(analysis, roster) first; only eligible cells are published")
    if stamp.get("out"):
        raise AnalysisError(f"{what} refused for {analysis.get('run_name')!r} (wave {analysis['wave']['name']!r}): "
                            + "; ".join(f"cell {c!r} is {r['status']} in the current roster ({r['reason']}) yet measured in this analysis"
                                        for c, r in stamp["out"].items())
                            + " — a blocked cell's measurement is not a paper row; re-run --analyze under the current roster, or reinstate the entry")


# --------------------------------------------------------------------------- #
# --paper-rows: the paper notebook's long schema                               #
# --------------------------------------------------------------------------- #

PAPER_COLUMNS: Tuple[str, ...] = (
    "source", "model", "lane_kind", "view", "dose", "cell_id", "arm", "family", "object_type",
    "component", "eval_id", "eval", "metric", "threshold", "bon_n",
    "value", "k", "n", "ci_lo", "ci_hi",
    "delta", "delta_lo", "delta_hi", "delta_p", "delta_method", "effect_size", "effect_kind",
    "at_cap_share", "loop_share", "degenerate_share",
    "saturated", "near_floor", "suggestive", "retention",
    "decode_cap_tokens", "decode", "judge", "status", "note", "provenance",
    "fit_layers", "fit_layer_count",
    # OBJECT IDENTITY beyond the layer set: the 64-hex sha of the served object.pt — null on the object-free `none` /
    # prompted cells and on rows that did not run — so the notebook's cross-eval one-object-per-arm guard separates the
    # batch-8 provisional constitution objects from the eb64 re-fits without reading `note`; and the thinking mode the
    # rows were drawn under ("off")
    "object_artifact_sha256", "thinking",
    "coherence", "delta_coherence", "degenerate_win",
    # HELD OUT of pooled category means: true on a GRADED row whose roster entry carries a `pooling: {held_out: true, ...}`
    # block — the row is reported with its numbers (never withdrawn, never averaged); the ruling and the reason lead its
    # `note`. False on every other row, including not-run and withdrawn ones. Declared in the roster, never inferred.
    "held_out")
assert len(PAPER_COLUMNS) == 48
PAPER_SOURCE = "bcb_bfcl_200_widen"
#: `paperlib.config.ARMS` by campaign cell id: (arm key, family, object_type). Read 2026-09-11 from the paper worktree.
PAPER_ARM_OF_CELL: Mapping[str, Tuple[str, str, str]] = {
    BASE_CELL: ("none", "none", "none"),
    "prompted__safety_refusal": ("si_safety", "prompted", "prompt"),
    # the paper's constitution-SI arm: this lane's launcher cell id is `prompted__si_constitution` (harness id
    # prompted__constitution — paperlib maps both spellings to the `si_constitution` arm).
    # `prompted__si_constitution_full` is a DIFFERENT arm (`si_constitution_full`: the entire constitution, ~34k
    # tokens, Llama fullconst view only; paperlib 098a3a81, 2026-09-11) — one label over the two prompts is the fork the
    # paper's guards refuse, so the two ids never share an arm key here
    "prompted__si_constitution": ("si_constitution", "prompted", "prompt"),
    "prompted__si_constitution_full": ("si_constitution_full", "prompted", "prompt"),
    "constv2-mix-full__psr_ce_all": ("const_psr", "constitution", "vector"),
    "constv2-mix-full__lora_r4_ce": ("const_lora", "constitution", "lora"),
    "constv2ctl2-full__psr_ce_all": ("ctl_psr", "control", "vector"),
    "constv2ctl2-full__lora_r4_ce": ("ctl_lora", "control", "lora"),
    "constv2-mix-full__psr_ce_all_minus_ctl_distilled": ("diff_psr", "difference", "vector"),
    "constv2-mix-full__lora_r4_ce_minus_ctl": ("diff_lora", "difference", "lora"),
    "safety_refusal__psr_kl_all": ("safety_psr", "safety-distilled", "vector"),
    "safety_refusal__lora_r4": ("safety_lora", "safety-distilled", "lora")}
#: arm keys the roster may name for an UNKEYED entry (cell_id null): (family, object_type)
PAPER_ARM_UNKEYED: Mapping[str, Tuple[str, str]] = {"alphasteer_as_fit": ("baseline", "matrix"), "alphasteer_rebake": ("baseline", "matrix"),
                                                    "arditi": ("baseline", "vector")}
PAPER_EVAL_ID: Mapping[str, str] = {BCB_TASK: C.CONTROL_BIGCODEBENCH_200, BFCL_TASK: C.CONTROL_BFCL_200}
PAPER_EVAL_LABEL: Mapping[str, str] = {BCB_TASK: "BigCodeBench-200", BFCL_TASK: "BFCL-200"}
PAPER_JUDGE: Mapping[str, str] = {BCB_TASK: "bigcodebench verify (sandboxed unit tests, inspect_evals 0.18.0)", BFCL_TASK: "bfcl AST scorer (inspect_evals 0.18.0)"}
PAPER_COMPONENT, PAPER_METRIC, PAPER_DOSE = "capability", "acc", 1.0
PAPER_KEY_COLUMNS: Tuple[str, ...] = ("source", "model", "lane_kind", "view", "arm", "eval_id")


def paper_arm(entry_rec: Mapping[str, Any]) -> Tuple[str, str, str]:
    """(arm key, family, object_type) for a roster entry: the paper's table by
    cell id; an unkeyed entry (cell_id null) by its `paper_arm`; a roster
    `paper_arm` that disagrees with the table refuses (one vocabulary)."""
    cell, want = entry_rec.get("cell_id"), entry_rec.get("paper_arm")
    if cell in PAPER_ARM_OF_CELL:
        arm, fam, kind = PAPER_ARM_OF_CELL[cell]
        if want and want != arm:
            raise AnalysisError(f"roster paper_arm {want!r} for cell {cell!r} != the paper's arm key {arm!r}")
        return arm, fam, kind
    if want in PAPER_ARM_UNKEYED:
        fam, kind = PAPER_ARM_UNKEYED[want]
        return want, fam, kind
    raise AnalysisError(f"no paper arm for entry {entry_rec.get('entry_id')!r} (cell {cell!r}, paper_arm {want!r})")


def provisional_mark(gen: Mapping[str, Any]) -> Optional[str]:
    if not gen.get("provisional"):
        return None
    return (f"object_generation={gen['object_generation']}; superseded_by_key={gen.get('superseded_by_key')}; re-run on the batch-64 re-fit "
            "when staged; do not pool with the batch-64 rows")


def _empty_row(**kw: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {c: None for c in PAPER_COLUMNS}
    row.update(source=PAPER_SOURCE, dose=PAPER_DOSE, component=PAPER_COMPONENT, metric=PAPER_METRIC, saturated=False, near_floor=False, suggestive=False,
               held_out=False)
    row.update(kw)
    return row


def _wave_spec_of_record(analysis: Mapping[str, Any]) -> WaveSpec:
    """A WaveSpec carrying what the not-run rows' declared decode needs, from the analysis record."""
    w = analysis["wave"]
    return WaveSpec(name=w["name"], family=w["family"], lane_kind=w["lane_kind"], view=w.get("view"), model_row=w.get("model_row"),
                    served=dict(w.get("served") or {}), objects_dir=w.get("objects_dir"), legs=dict(w.get("legs") or {}), entries=[],
                    sampling=w.get("sampling"), sampling_clamp_override=w.get("sampling_clamp_override"), thinking=w.get("thinking"))


def paper_rows(analysis: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """ONE row per (cell, leg) of the wave — measured rows for the runnable cells
    from the leg's FINAL reading (:func:`final_readings`: `status graded`; `pending
    ladder` with null measurements when a fired leg has no final reading), and
    QUEUED / BLOCKED rows with null measurements and an explicit `status` for every
    non-runnable entry × leg and every runnable cell × blocked leg (reported
    by name, never dropped, never filled). Every row's `decode` carries
    the literal words "benchmark protocol" (a not-run row: the roster's DECLARED
    protocol, labelled NOT RUN); `object_artifact_sha256` is the served object's
    (null on object-free and not-run rows); `thinking` the drawn / declared mode."""
    require_publication_gate(analysis, what="the paper export")
    w = analysis["wave"]
    base, lad = analysis["base"], analysis.get("ladder") or {}
    gens, layers = analysis["generations"], analysis["fit_layers"]
    prov = analysis["identity"]["provenance"]
    run_ids = prov.get("run_ids") or ([prov["run_id"]] if prov.get("run_id") else [])
    prov_base = (f"{analysis['rung1_dir']}; wave={w['name']}; suffix={prov.get('suffix')}; run_ids={','.join(run_ids)}; "
                 f"tree={analysis['identity']['tree_sha']}")
    # the MODE string ("off"), never the producer's {mode, render} dict — the render detail is not in the schema
    thinking = thinking_mode_of(analysis["identity"].get("thinking") or w.get("thinking"))
    common = dict(model=w["family"], lane_kind=w["lane_kind"], view=w["view"], thinking=thinking)
    entries = {e["cell_id"]: e for e in w["entries"] if e.get("cell_id")}
    finals = final_readings(analysis)
    held = (analysis.get(PUBLICATION_KEY) or {}).get(HELD_OUT_KEY) or {}     # the gate's stamp: the CURRENT roster's pooling declaration
    rows: List[Dict[str, Any]] = []
    for task, leg in analysis["legs"].items():
        fin = finals[task]
        decode_base = analysis["protocol"]["decode_by_task"][task]
        if fin["source"] == FINAL_SOURCE_WITHDRAWN:
            # a leg the current roster blocks after it was measured — the row STAYS (reported by name, never dropped) with
            # `status: withdrawn: <reason>`, every measurement null, the note naming the withdrawal; identity columns (the served
            # object's sha, its fit layers) are facts about what ran and stay
            wd = fin["withdrawn"]
            for cell in analysis["cells"]:
                arm, family, object_type = paper_arm(entries[cell])
                gen, fl = gens[cell], (layers.get(cell) or {})
                rows.append(_empty_row(**common, cell_id=cell, arm=arm, family=family, object_type=object_type, eval_id=PAPER_EVAL_ID[task],
                                       eval=PAPER_EVAL_LABEL[task], judge=PAPER_JUDGE[task], status=f"withdrawn: {wd.get('reason')}",
                                       decode=f"{decode_base}; WITHDRAWN as a capability reading (D79) — not a reading",
                                       note=(f"WITHDRAWN as a capability reading (D79): the current roster (sha {str(wd.get('roster_sha256') or 'unknown')[:12]}) "
                                             f"marks leg {ATOM_OF_TASK[task]} {wd.get('status')} for wave {w['name']} after the wave was measured; the store's "
                                             f"numbers ({leg['cells'][cell]['store_key']}) are kept under final[{task}].withdrawn in "
                                             f"{withdrawn_record_location(analysis)} for the record, not exported here; unblock: {wd.get('unblock')}"),
                                       provenance=prov_base + (f"; object_sha256={gen['object_sha256']}" if gen.get("object_sha256") else "")
                                       + f"; roster wave {w['name']} leg {ATOM_OF_TASK[task]} {wd.get('status')}",
                                       fit_layers=fl.get("fit_layers"), fit_layer_count=fl.get("fit_layer_count"),
                                       object_artifact_sha256=gen.get("object_sha256")))
            continue
        applied = fin["source"] == FINAL_SOURCE_MERGED
        pending = not fin["final"]
        for cell in analysis["cells"]:
            arm, family, object_type = paper_arm(entries[cell])
            r = leg["cells"][cell]
            pr = fin["cells"][cell] if fin["final"] else None
            paired = (fin["paired"] or {}).get(cell) if fin["final"] else None
            gen = gens[cell]
            mark = provisional_mark(gen)
            note = (f"tinycap served leg {task} on {analysis['run_name']} (wave {w['name']}); store {r['store_key']}; OFF cap {r['cap']}; "
                    f"rung-1 at-cap {r['at_cap']['n_at_cap']}/{r['at_cap']['n_stop_reason']}; final reading: {fin['source']}")
            cap_tokens, at_cap_share, decode = int(r["cap"]), r["at_cap"]["n_at_cap"] / r["at_cap"]["n_stop_reason"], decode_base
            if applied:
                lc = lad["tasks"][task]["cells"][cell]
                cap_tokens, at_cap_share = int(fin["cap_by_cell"][cell]), fin["at_cap_share_by_cell"][cell]
                decode = f"{decode_base}; adaptive truncation rerun: items at the {r['cap']} cap re-run at {lc['cap_to']} (the lane's > 1 % rule), merged per item"
                note += (f"; truncation ladder APPLIED: merged reading ({lc['n_selected']} selected items re-run at {lc['cap_to']}, residual at-cap "
                         f"{lc['n_at_cap_residual']}/{lc['merged']['n']}); rung 1 (diagnostic, not final) read {r['k']}/{r['n']} = "
                         f"{100 * r['rate']:.1f} % at cap {r['cap']}")
            elif pending:
                note += (f"; truncation rule FIRED, no final reading ({fin['why']}): NO reading exported (rung 1, diagnostic only, read "
                         f"{r['k']}/{r['n']} = {100 * r['rate']:.1f} % at cap {r['cap']}); acceptance check 4 NOT MET")
            if mark:
                note += "; " + mark
            entry_note = entries[cell].get("note")
            if entry_note and cell != base:
                note += "; roster note: " + str(entry_note)
            pool = held.get(cell)
            if pool:
                # REPORTED with its numbers, HELD OUT of pooled category means — the ruling and the reason
                # (the behaviour, a verbatim opening, the recorded mechanism numbers) lead the note; `held_out` is the notebook's flag
                note = f"HELD OUT of pooled category means ({pool.get('ruled_by')}): {pool.get('reason')} | " + note
            provenance = (prov_base + (f"; object_sha256={gen['object_sha256']}" if gen.get("object_sha256") else "") + (f"; {mark}" if mark else "")
                          + ("; held_out=true" if pool else ""))
            fl = layers.get(cell) or {}
            row = _empty_row(**common, cell_id=cell, arm=arm, family=family, object_type=object_type, eval_id=PAPER_EVAL_ID[task], held_out=bool(pool),
                             eval=PAPER_EVAL_LABEL[task], value=(None if pending else float(pr["rate"])), k=(None if pending else int(pr["k"])),
                             n=int(r["n"]), ci_lo=(None if pending else float(pr["wilson_low"])), ci_hi=(None if pending else float(pr["wilson_high"])),
                             at_cap_share=at_cap_share, decode_cap_tokens=cap_tokens, decode=decode, judge=PAPER_JUDGE[task],
                             status=("pending ladder" if pending else "graded"), note=note, provenance=provenance,
                             fit_layers=fl.get("fit_layers"), fit_layer_count=fl.get("fit_layer_count"),
                             object_artifact_sha256=gen.get("object_sha256"))
            if cell != base and not pending and paired is not None:
                row.update(delta=100.0 * float(paired["delta"]), delta_lo=100.0 * float(paired["ci_low"]), delta_hi=100.0 * float(paired["ci_high"]),
                           delta_p=float(paired["mcnemar_exact_p"]),
                           delta_method=(f"paired on {paired['n_items']} items (per-item acc) vs this wave's unsteered cell: percentile paired "
                                         f"bootstrap B={paired['B']:,} seed {paired['seed']}; exact two-sided McNemar p (b={paired['b_pass_to_fail']}, "
                                         f"c={paired['c_fail_to_pass']}); final reading ({fin['source']}): {paired['reading']}"))
            rows.append(row)
    # queued / blocked: every non-runnable entry × every leg of the lane; every runnable cell × every blocked leg — the decode is the
    # roster's DECLARED protocol, labelled NOT RUN, so the notebook's protocol selection keeps these rows (with their status) in view
    spec = _wave_spec_of_record(analysis)
    for e in w["entries"]:
        if e["status"] == "runnable":
            continue
        arm, family, object_type = paper_arm(e)
        for task in TASKS:
            rows.append(_empty_row(**common, cell_id=e.get("cell_id"), arm=arm, family=family, object_type=object_type, eval_id=PAPER_EVAL_ID[task],
                                   eval=PAPER_EVAL_LABEL[task], judge=PAPER_JUDGE[task], status=f"{e['status']}: {e.get('reason')}",
                                   note=f"unblock: {e.get('unblock')}", provenance=f"roster wave {w['name']} entry {e['entry_id']}",
                                   decode=declared_decode_string(task, spec)))
    for task, blk in (analysis.get("blocked_legs") or {}).items():
        for cell in analysis["cells"]:
            arm, family, object_type = paper_arm(entries[cell])
            rows.append(_empty_row(**common, cell_id=cell, arm=arm, family=family, object_type=object_type, eval_id=PAPER_EVAL_ID[task],
                                   eval=PAPER_EVAL_LABEL[task], judge=PAPER_JUDGE[task], status=f"{blk.get('status')}: {blk.get('reason')}",
                                   note=f"unblock: {blk.get('unblock')}", provenance=f"roster wave {w['name']} leg {ATOM_OF_TASK[task]}",
                                   decode=declared_decode_string(task, spec)))
    return rows


def _typed_frame(rows: Sequence[Mapping[str, Any]]):
    import pandas as pd
    df = pd.DataFrame(list(rows)).reindex(columns=list(PAPER_COLUMNS))
    for c in ("k", "n", "bon_n", "decode_cap_tokens", "fit_layer_count"):
        df[c] = df[c].astype("Int64")
    for c in ("dose", "threshold", "value", "ci_lo", "ci_hi", "delta", "delta_lo", "delta_hi", "delta_p", "effect_size",
              "at_cap_share", "loop_share", "degenerate_share", "retention", "coherence", "delta_coherence"):
        df[c] = df[c].astype("float64")
    for c in ("saturated", "near_floor", "suggestive", "held_out"):
        df[c] = df[c].fillna(False).astype(bool)
    return df


def _row_keys(df) -> List[Tuple[Any, ...]]:
    keys = df[list(PAPER_KEY_COLUMNS)].astype(object)
    return [tuple(k) for k in keys.where(keys.notna(), None).values.tolist()]


@dataclass
class PaperRowsWrite:
    path: str
    n_rows_written: int
    n_added: int
    n_replaced: int
    n_kept: int

    def __getitem__(self, key: str) -> Any:          # the record's dict face (the CLI prints it, the tests index it)
        return getattr(self, key)


def write_paper_rows_all(analyses: Sequence[Mapping[str, Any]], out: Union[str, Path], *, replace: bool = False) -> List[PaperRowsWrite]:
    """`--paper-rows`: the waves' rows into the lane's parquet (or CSV) in ONE
    locked read–merge–write: under ``fcntl.flock`` on ``<out>.lock``
    the existing file is read once, every wave's rows are assembled, an existing
    row with the same (source, model, lane_kind, view, arm, eval_id) key is
    REPLACED by the newest wave's (a queued row becomes measured when the wave
    lands; a re-analysis refreshes in place), every other row is kept — or, with
    ``replace``, the file is REPLACED ONCE by all the waves' rows together (never
    wave by wave, which kept only the last) — and the result is published through a
    temp file + ``os.replace``, so a reader never sees a partial parquet and two
    exports never lose each other's waves. One record per wave."""
    import pandas as pd
    p = Path(out).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    lock = p.with_name(p.name + ".lock")
    frames = [(a, _typed_frame(paper_rows(a))) for a in analyses]
    with open(lock, "a+") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            if not replace and p.is_file():
                old = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
                current = _typed_frame(old.reindex(columns=list(PAPER_COLUMNS)).to_dict("records")) if len(old) else None
            else:
                current = None
            records: List[PaperRowsWrite] = []
            for a, new in frames:
                kept, replaced = None, 0
                if current is not None and len(current):
                    keys_new = set(_row_keys(new))
                    mask = [k in keys_new for k in _row_keys(current)]
                    replaced = int(sum(mask))
                    kept = current[[not m for m in mask]]
                    current = pd.concat([_typed_frame(kept.to_dict("records")) if len(kept) else new.iloc[0:0], new], ignore_index=True)
                else:
                    current = new
                records.append(PaperRowsWrite(path=str(p), n_rows_written=int(len(current)), n_added=int(len(new)), n_replaced=replaced,
                                              n_kept=(int(len(kept)) if kept is not None else 0)))
            tmp = p.with_name(f".{p.name}.tmp-{os.getpid()}")
            try:
                if p.suffix == ".parquet":
                    current.to_parquet(tmp, index=False)
                else:
                    current.to_csv(tmp, index=False)
                os.replace(tmp, p)                  # atomic publish: a reader sees the old file or the new one, never a torn one
            finally:
                if tmp.exists():
                    tmp.unlink()
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return records


def write_paper_rows(analysis: Mapping[str, Any], out: Union[str, Path], *, append: bool = True) -> PaperRowsWrite:
    """One wave's rows: `write_paper_rows_all([analysis], out, replace=not append)`."""
    return write_paper_rows_all([analysis], out, replace=not append)[0]


# --------------------------------------------------------------------------- #
# figures (dataviz: one claim each; titles ≤ 2 lines, subtitle ≤ 3, axes ≥ 80 %)  #
# --------------------------------------------------------------------------- #

#: the base cell muted; steered cells in a fixed categorical order (≤ 4 steered per wave)
STEERED_COLORS: Tuple[str, ...] = (CAT["blue"], CAT["orange"], CAT["aqua"], CAT["yellow"])
FIGURE_STEMS: Tuple[str, ...] = ("pass_rate_by_cell", "paired_delta", "bfcl_by_category", "bcb_by_family", "diagnostics", "cost_wall",
                                 "ladder", "gpu_budget", "roster_status")
SUMMARY_FIGURE_STEMS: Tuple[str, ...] = ("summary_paired_delta", "summary_pass_rate", "summary_roster", "summary_gpu_budget")


def short_cell(cell: str, analysis: Optional[Mapping[str, Any]] = None) -> str:
    """The figure's name for a cell: the paper's arm key (+ ' (provisional)' for a batch-8 object)."""
    if cell == BASE_CELL:
        return "none"
    arm = PAPER_ARM_OF_CELL.get(cell, (cell, "", ""))[0]
    if analysis is not None and (analysis.get("generations") or {}).get(cell, {}).get("provisional"):
        arm += " (prov.)"
    return arm


def _cell_series(analysis: Mapping[str, Any]) -> List[Tuple[str, str]]:
    out = [(analysis["base"], MUTED)]
    for k, cell in enumerate(analysis["steered"]):
        out.append((cell, STEERED_COLORS[k % len(STEERED_COLORS)]))
    return out


def _cell_legend(target, Patch, analysis: Mapping[str, Any], **kw):
    return target.legend(handles=[Patch(color=col, label=short_cell(cell, analysis)) for cell, col in _cell_series(analysis)], frameon=False, **kw)


def _wave_tag(analysis: Mapping[str, Any]) -> str:
    w = analysis["wave"]
    return f"{w['family']} · {w['lane_kind']}"


def _out_title_tag(analysis: Mapping[str, Any]) -> str:
    """`` · N out`` on a gated view's figure titles: N measured cells the current roster does not run are not drawn;
    `` · <leg> withdrawn`` names a measured leg the current roster blocks — its numbers are not drawn either."""
    out = (analysis.get(PUBLICATION_KEY) or {}).get("out") or {}
    tag = f" · {len(out)} out" if out else ""
    withdrawn = analysis.get(WITHDRAWN_LEGS_KEY) or {}
    if withdrawn:
        tag += " · " + ", ".join(TASK_SHORT[t] for t in withdrawn) + " withdrawn"
    return tag


def _shown_tasks(analysis: Mapping[str, Any]) -> List[str]:
    """The legs a per-wave READING figure draws: every measured leg but the withdrawn ones."""
    withdrawn = analysis.get(WITHDRAWN_LEGS_KEY) or {}
    return [t for t in analysis["legs"] if t not in withdrawn]


def _withdrawn_placeholder(analysis: Mapping[str, Any], title: str):
    """A reading figure with nothing left to draw: every measured leg is withdrawn under the current roster."""
    wd = analysis.get(WITHDRAWN_LEGS_KEY) or {}
    return _placeholder(title + " — WITHDRAWN: " + "; ".join(f"{TASK_SHORT[t]} ({str(w.get('reason') or '')[:90]})" for t, w in wd.items()),
                        "instrument failure (D79): the current roster blocks the leg; the store's numbers are kept in analysis.json under "
                        "`final.<leg>.withdrawn` for the record and are not drawn as a reading")


def _final_or_rung1(analysis: Mapping[str, Any], task: str) -> Tuple[Mapping[str, Any], Mapping[str, Any], str]:
    """``(cells, paired, label)`` a per-wave figure draws for a leg: the FINAL reading when there
    is one (label names its source), else the rung-1 reading under an explicit NOT-final label."""
    fin = final_readings(analysis)[task]
    leg = analysis["legs"][task]
    if fin["final"]:
        return fin["cells"], fin["paired"], ("final: merged" if fin["source"] == FINAL_SOURCE_MERGED else "final: rung 1")
    return leg["cells"], leg["paired"], "rung 1 — NOT final (ladder pending)"


def fig_pass_rate_by_cell(analysis: Mapping[str, Any]):
    plt, Patch = _mpl()
    legs, series = analysis["legs"], _cell_series(analysis)
    tasks = _shown_tasks(analysis)                    # a withdrawn leg is not drawn as a reading
    if not tasks:
        return _withdrawn_placeholder(analysis, f"Pass rate per leg and cell — {_wave_tag(analysis)}")
    shown = {t: _final_or_rung1(analysis, t) for t in tasks}
    title = f"Pass rate per leg and cell — {_wave_tag(analysis)}: " + "; ".join(
        f"{TASK_SHORT[t]} none {_pct(shown[t][0][analysis['base']]['rate'], 0)} ({shown[t][2]})" for t in tasks) + _out_title_tag(analysis)
    subtitle = ("n = 200 pinned items per leg, k = 1 under the benchmarks' protocol decoding, thinking OFF, served path; whiskers = Wilson 95 % "
                "intervals — wide by design at k = 1; each leg's FINAL reading (merged after a fired ladder, else rung 1; a pending leg shows "
                "rung 1 as NOT final); the paired contrast (next figure) is the readout; '(prov.)' = batch-8 object, re-run pending")
    fig, ax = plt.subplots(figsize=(max(8.4, 2.6 * len(series) + 2.0), 5.2))
    _fit_header(fig, title, subtitle, bottom_in=0.7, left=0.09, right=0.98)
    width = 0.8 / len(series)
    for k, t in enumerate(tasks):
        for j, (cell, col) in enumerate(series):
            pr = shown[t][0][cell]
            x = k + (j - (len(series) - 1) / 2) * width
            r = 100 * (pr["rate"] or 0.0)
            ax.bar(x, r, width=width * 0.92, color=col, edgecolor="none", zorder=3)
            ax.errorbar(x, r, yerr=[[r - 100 * pr["wilson_low"]], [100 * pr["wilson_high"] - r]], fmt="none", ecolor=INK2, elinewidth=1.0, capsize=3, zorder=4)
            ax.text(x, 100 * pr["wilson_high"] + 1.2, f"{pr['k']}", ha="center", va="bottom", fontsize=7.0, color=INK2)
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([f"{TASK_LABEL[t]}\n({shown[t][2]})" for t in tasks], fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("pass rate (%) over the 200 pinned items")
    _cell_legend(ax, Patch, analysis, loc="upper right", fontsize=7.6, ncol=min(3, len(series)))
    _recessive(ax, grid_axis="y")
    finish_figure(fig, title, subtitle)
    return fig


def _forest(ax, rows: Sequence[Tuple[str, Mapping[str, Any], str]], *, label_p: bool = True):
    """rows = (label, paired block, colour); y from top; returns (los, his)."""
    y = list(range(len(rows)))[::-1]
    los, his = [], []
    for yi, (label, p, col) in zip(y, rows):
        d, lo, hi = 100 * p["delta"], 100 * p["ci_low"], 100 * p["ci_high"]
        los.append(lo); his.append(hi)
        # the emphasis follows the READING RULE (exact test AND Δ < 0), never the bootstrap interval — they disagreed on
        # one row (b 4 / c 0: the interval excluded 0, the exact test did not reject)
        c = col if cost_detected(p) else MUTED
        ax.errorbar([d], [yi], xerr=[[d - lo], [hi - d]], fmt="o", color=c, ecolor=c, elinewidth=1.3, capsize=3.5, markersize=5.5, zorder=4)
        if label_p:
            note = f"b {p['b_pass_to_fail']} / c {p['c_fail_to_pass']}; {_p(p['mcnemar_exact_p'])}"
            if interval_disagrees(p):
                note += f" — interval {'excludes' if p['excludes_zero'] else 'includes'} 0 (sparse discordance; the test is the rule)"
            ax.text(d, yi + 0.2, note, va="bottom", ha="center", fontsize=6.8, color=INK2)
    ax.axvline(0, color=BASELINE, linewidth=1.0, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_ylim(-0.6, len(rows) - 0.4 + 0.3)
    span = (max(his) - min(los)) if rows else 1.0
    pad = max(1.0, 0.12 * span)
    ax.set_xlim(min(0.0, min(los, default=0.0)) - pad, max(0.0, max(his, default=0.0)) + pad)
    ax.set_xlabel("difference in pass rate (pp), steered − none")
    _recessive(ax, grid_axis="x")


def fig_paired_delta(analysis: Mapping[str, Any]):
    plt, _ = _mpl()
    series = _cell_series(analysis)[1:]
    tasks = _shown_tasks(analysis)                    # a withdrawn leg is not drawn as a reading
    if not tasks:
        return _withdrawn_placeholder(analysis, f"Paired per-item difference vs none — {_wave_tag(analysis)}")
    rows = []
    for t in tasks:
        cells, paired, label = _final_or_rung1(analysis, t)
        final = final_readings(analysis)[t]["final"]
        for cell, col in series:
            rows.append((f"{TASK_SHORT[t]} · {short_cell(cell, analysis)}" + ("" if final else " · rung 1, NOT final"), paired[cell],
                         (col if final else MUTED)))
    title = f"Paired per-item difference vs none — {_wave_tag(analysis)}, {len(series)} steered cell(s) × {len(tasks)} leg(s)" + _out_title_tag(analysis)
    subtitle = ("point = mean difference over the same 200 items; whisker = percentile paired bootstrap 95 % (B = 10,000, seed 0), an estimation "
                "interval; COLOURED when the reading rule detects a cost (exact McNemar p < 0.05 and Δ < 0; nominal, unadjusted; one draw per item); "
                "label = discordants b (pass→fail) / c (fail→pass) + exact p; each leg's FINAL reading (merged after a fired ladder, else rung 1); "
                "a pending leg's rung-1 rows are grey, NOT final")
    fig, ax = plt.subplots(figsize=(8.8, max(3.4, 0.62 * len(rows) + 1.6)))
    _fit_header(fig, title, subtitle, bottom_in=0.65, left=0.3, right=0.97)
    _forest(ax, rows)
    finish_figure(fig, title, subtitle)
    return fig


def _grouped_breakdown(analysis: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, key: str, title: str, subtitle: str):
    plt, Patch = _mpl()
    series = _cell_series(analysis)
    n = len(rows)
    fig, ax = plt.subplots(figsize=(9.2, max(4.6, 0.16 * len(series) * n + 1.9)))
    _fit_header(fig, title, subtitle, bottom_in=0.62, left=0.27, right=0.97)
    y = list(range(n))[::-1]
    h = 0.8 / len(series)
    for yi, r in zip(y, rows):
        for j, (cell, col) in enumerate(series):
            pr = r["cells"][cell]
            yy = yi + ((len(series) - 1) / 2 - j) * h
            rate = 100 * (pr["rate"] or 0.0)
            _bar_h(ax, yy, rate, col, height=h * 0.9)
        ax.text(101.5, yi, f"n = {r['n']}  Δ " + " / ".join(_pp(r["deltas"][c]["delta"], 0) for c in analysis["steered"]), va="center", ha="left",
                fontsize=6.8, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r[key]}" for r in rows], fontsize=7.6)
    ax.set_xlim(0, 132)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("pass rate (%) — bar = k/n; Δ = steered − none in pp, one per steered cell")
    _cell_legend(ax, Patch, analysis, loc="lower right", fontsize=7.4)
    _recessive(ax, grid_axis="x")
    finish_figure(fig, title, subtitle)
    return fig


def fig_bfcl_by_category(analysis: Mapping[str, Any]):
    bd = analysis["breakdowns"].get("bfcl_by_category")
    if BFCL_TASK in (analysis.get(WITHDRAWN_LEGS_KEY) or {}):
        return _withdrawn_placeholder(analysis, f"BFCL-200 by category — {_wave_tag(analysis)}")
    if bd is None:
        return _placeholder(f"BFCL-200 by category — {_wave_tag(analysis)}: the BFCL leg is not runnable in this wave",
                            "the roster marks the leg blocked (parser unverified) or not filed; nothing to break down")
    return _grouped_breakdown(analysis, bd["rows"], key="category",
                              title=f"BFCL-200 pass rate by category — {_wave_tag(analysis)}, {bd['n_categories']} categories ({bd.get('label', 'final: rung 1')})",
                              subtitle="Hamilton allocation with floor 1 over the 18 single-turn categories; categories with n ≤ 5 carry wide intervals "
                                       "by design; Δ = steered − none in pp per steered cell; rows = the leg's FINAL per-item rows (merged after an "
                                       "applied ladder, else rung 1; a pending leg's rung-1 rows are marked NOT final)")


def fig_bcb_by_family(analysis: Mapping[str, Any]):
    bd = analysis["breakdowns"].get("bcb_by_family")
    if BCB_TASK in (analysis.get(WITHDRAWN_LEGS_KEY) or {}):
        return _withdrawn_placeholder(analysis, f"BigCodeBench-200 by library family — {_wave_tag(analysis)}")
    if bd is None:
        return _placeholder(f"BigCodeBench-200 by library family — {_wave_tag(analysis)}: the BCB leg is not runnable in this wave", "nothing to break down")
    return _grouped_breakdown(analysis, bd["rows"], key="family",
                              title=f"BigCodeBench-200 pass rate by library family — {_wave_tag(analysis)}, {bd['n_families']} families ({bd.get('label', 'final: rung 1')})",
                              subtitle="multi-label: an item counts under every family of its libraries (the pin's libs_by_id × bcb_lib_families.json), "
                                       "so n does not sum to 200; Δ = steered − none in pp per steered cell; rows = the leg's FINAL per-item rows (merged "
                                       "after an applied ladder, else rung 1; a pending leg's rung-1 rows are marked NOT final)")


def _placeholder(title: str, subtitle: str):
    plt, _ = _mpl()
    fig, ax = plt.subplots(figsize=(8.0, 3.6))
    _fit_header(fig, title, subtitle, bottom_in=0.4, left=0.05, right=0.97)
    ax.text(0.5, 0.5, "not applicable in this wave", transform=ax.transAxes, ha="center", va="center", color=INK2)
    ax.set_xticks([]); ax.set_yticks([])
    _recessive(ax, grid_axis="none")
    finish_figure(fig, title, subtitle)
    return fig


def rung1_cap_label(analysis: Mapping[str, Any]) -> str:
    """``rung 1 (cap 1,024 / 4,096)`` — the rung and the OFF caps the diagnostics were read at, so rung-1 at-cap
    counts are never shown unlabelled beside the final merged counts at 16,384."""
    caps = sorted({int(leg["cells"][analysis["base"]]["cap"]) for leg in analysis["legs"].values()})
    return "rung 1 (cap " + " / ".join(f"{c:,}" for c in caps) + ")"


def merged_legs(analysis: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    """The fired legs whose ladder merge is recorded (``ladder.tasks``), applied or not — their final merged at-cap counts."""
    return dict((analysis.get("ladder") or {}).get("tasks") or {})


def fig_diagnostics(analysis: Mapping[str, Any]):
    plt, Patch = _mpl()
    legs, series = analysis["legs"], _cell_series(analysis)
    tasks, cells = list(legs), [c for c, _ in _cell_series(analysis)]
    withdrawn, merged = analysis.get(WITHDRAWN_LEGS_KEY) or {}, merged_legs(analysis)
    title = f"Diagnostics per leg and cell — {_wave_tag(analysis)}, {rung1_cap_label(analysis)}: at-cap vs the 1 % rule, unparseable, BCB code source, sandbox status"
    subtitle = ("at-cap = stop_reason == max_tokens at the rung-1 OFF cap (fires above 2/200; model_length = context overflow, counted apart)"
                + (f"; hatched = FINAL residual at-cap after the 16,384 re-run, merged per item ({', '.join(TASK_SHORT[t] for t in merged)})" if merged else "")
                + "; unparseable = the scorer's empty extraction; code_source = the fence the verifier took (a format diagnostic); degeneration not "
                "measured (no declared classifier)" + ("; a withdrawn leg's diagnostics are instrument facts, not a reading" if withdrawn else ""))
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0))
    _fit_header(fig, title, subtitle, bottom_in=0.95, left=0.17, right=0.98, hspace=0.6, wspace=0.34, extra_top_in=0.3)
    width = 0.8 / len(series)

    def leg_tick(t: str) -> str:
        return TASK_SHORT[t] + (" (withdrawn)" if t in withdrawn else "")

    ax = axes[0][0]
    for k, t in enumerate(tasks):
        for j, (cell, col) in enumerate(series):
            v = legs[t]["cells"][cell]["at_cap"]["n_at_cap"]
            x = k + (j - (len(series) - 1) / 2) * width
            if t in merged:
                # the rung-1 count beside the FINAL merged residual (cap 16,384), hatched, same colour
                r2 = merged[t]["cells"][cell]["n_at_cap_residual"]
                ax.bar(x - width * 0.23, v, width=width * 0.44, color=col, zorder=3)
                ax.bar(x + width * 0.23, r2, width=width * 0.44, color=col, hatch="////", edgecolor=SURFACE, linewidth=0.6, zorder=3)
                ax.text(x - width * 0.23, v + 0.15, str(v), ha="center", va="bottom", fontsize=6.4, color=INK2)
                ax.text(x + width * 0.23, r2 + 0.15, f"{r2}", ha="center", va="bottom", fontsize=6.4, color=INK2)
            else:
                ax.bar(x, v, width=width * 0.9, color=col, zorder=3)
                ax.text(x, v + 0.15, str(v), ha="center", va="bottom", fontsize=6.8, color=INK2)
    ax.axhline(2, color=STATUS["warning"], linewidth=1.0, linestyle="--", zorder=2)
    ax.text(len(tasks) - 0.55, 2.15, "1 % = 2/200 (fires above)", fontsize=7.0, color=STATUS["warning"], ha="right")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([leg_tick(t) + (f"\nrung 1 cap {legs[t]['cells'][analysis['base']]['cap']:,}" + (" → hatched: merged, cap 16,384" if t in merged else ""))
                        for t in tasks], fontsize=7.4)
    ax.set_ylabel("items at the cap (of 200)")
    ax.set_ylim(0, max(4.0, max(legs[t]["cells"][c]["at_cap"]["n_at_cap"] for t in tasks for c in cells) * 1.4))
    _recessive(ax, grid_axis="y"); ax.set_title("at-cap per leg — rung 1" + ("; hatched = merged (16,384)" if merged else ""), fontsize=8.5, color=INK2, loc="left")
    ax = axes[0][1]
    for k, t in enumerate(tasks):
        for j, (cell, col) in enumerate(series):
            u = legs[t]["cells"][cell]["unparseable"]
            x = k + (j - (len(series) - 1) / 2) * width
            ax.bar(x, u["n_unparseable"], width=width * 0.9, color=col, zorder=3)
            ax.text(x, u["n_unparseable"] + 0.15, f"{u['n_unparseable']}", ha="center", va="bottom", fontsize=6.8, color=INK2)
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels([leg_tick(t) for t in tasks]); ax.set_ylabel("unparseable (of answered)")
    ax.set_ylim(0, max(4.0, max(legs[t]["cells"][c]["unparseable"]["n_unparseable"] for t in tasks for c in cells) * 1.4))
    _recessive(ax, grid_axis="y"); ax.set_title("unparseable per leg", fontsize=8.5, color=INK2, loc="left")
    bcb = legs.get(BCB_TASK)
    ax = axes[1][0]
    if bcb:
        keys = sorted({k for c in cells for k in bcb["cells"][c]["code_source"]})
        shades = [SEQ[600], SEQ[450], SEQ[250], SEQ[100]]
        for j, cell in enumerate(cells):
            left = 0
            for ki, key in enumerate(keys):
                v = bcb["cells"][cell]["code_source"].get(key, 0)
                ax.barh(j, v, left=left, color=shades[ki % len(shades)], height=0.6, zorder=3)
                if v:
                    ax.text(left + v / 2, j, f"{key} {v}", ha="center", va="center", fontsize=6.4, color=SURFACE if ki < 2 else INK)
                left += v
        ax.set_yticks(range(len(cells))); ax.set_yticklabels([short_cell(c, analysis) for c in cells], fontsize=7.4)
    else:
        ax.text(0.5, 0.5, "no BCB leg", transform=ax.transAxes, ha="center", va="center", color=INK2); ax.set_yticks([])
    ax.set_xlabel("BCB items by code_source"); ax.set_title("BCB code source", fontsize=8.5, color=INK2, loc="left"); _recessive(ax, grid_axis="x")
    ax = axes[1][1]
    if bcb:
        keys = sorted({k for c in cells for k in bcb["cells"][c]["sandbox_status"]})
        for ki, key in enumerate(keys):
            for j, (cell, col) in enumerate(series):
                v = bcb["cells"][cell]["sandbox_status"].get(key, 0)
                x = ki + (j - (len(series) - 1) / 2) * width
                ax.bar(x, v, width=width * 0.9, color=col, zorder=3)
                ax.text(x, v + 1, str(v), ha="center", va="bottom", fontsize=6.6, color=INK2)
        ax.set_xticks(range(len(keys))); ax.set_xticklabels(keys)
    else:
        ax.text(0.5, 0.5, "no BCB leg", transform=ax.transAxes, ha="center", va="center", color=INK2); ax.set_xticks([])
    ax.set_ylabel("BCB items"); ax.set_ylim(0, 230); ax.set_title("BCB sandbox status", fontsize=8.5, color=INK2, loc="left"); _recessive(ax, grid_axis="y")
    _cell_legend(fig, Patch, analysis, loc="lower center", ncol=len(series), fontsize=7.4, bbox_to_anchor=(0.5, 0.005))
    finish_figure(fig, title, subtitle)
    return fig


def fig_cost_wall(analysis: Mapping[str, Any]):
    plt, Patch = _mpl()
    gpu, wall = analysis["cost"]["gpu"], analysis["wall"]
    title = (f"Cost and wall clock — {_wave_tag(analysis)}: wave GPU {_num(gpu['wave_h'], 2)} H200-h (lane Σ {_num(gpu['committed_h'], 2)} of "
             f"{_num(gpu['ceiling_h'], 1)} h), sandboxes ≈ ${analysis['cost']['sandbox_usd_estimate']:.2f}"
             if gpu["present"] else f"Cost and wall clock — {_wave_tag(analysis)}: no meter ledger found; sandbox dollars only")
    subtitle = ("left: H200-hours per reservation of THIS wave from the meter's ledger (actual where recorded, else forecast); right: wall clock per "
                "cell (first attempt pointer → last marker) — every dollar is a resource-rate estimate, not billing, and a PARTIAL subtotal")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.9), gridspec_kw={"width_ratios": [1.6, 1.0]})
    _fit_header(fig, title, subtitle, bottom_in=0.75, left=0.18, right=0.98, wspace=0.4)
    rows = [r for r in gpu["rows"] if r.get("wave") == analysis["wave"]["name"]] if gpu["present"] else []
    if rows:
        y = list(range(len(rows)))[::-1]
        for yi, row in zip(y, rows):
            if row["forecast"] is not None:
                _bar_h(ax1, yi + 0.18, row["forecast"], SEQ[250] if row["superseded_by"] is None else GRID, height=0.34)
            if row["actual"] is not None:
                _bar_h(ax1, yi - 0.18, row["actual"], CAT["blue"], height=0.34)
        ax1.set_yticks(y)
        ax1.set_yticklabels([f"{r['unit']} ({r['stage_kind'] or '—'})" + (f" #{r['attempt']}" if r["attempt"] > 1 else "") for r in rows], fontsize=7.2)
        ax1.legend(handles=[Patch(color=SEQ[250], label="forecast"), Patch(color=CAT["blue"], label="actual")], loc="lower right", frameon=False, fontsize=7.4)
    else:
        ax1.text(0.5, 0.5, "no reservations for this wave", transform=ax1.transAxes, ha="center", va="center", color=INK2); ax1.set_yticks([])
    ax1.set_xlabel("H200-hours per reservation"); _recessive(ax1, grid_axis="x")
    cells = list(wall)
    colors = dict(_cell_series(analysis))
    hours = [wall[c]["hours"] if wall[c]["hours"] is not None else wall[c]["tasks_wall_s"] / 3600.0 for c in cells]
    ax2.bar(range(len(cells)), hours, color=[colors.get(c, MUTED) for c in cells], width=0.55, zorder=3)
    for k, (c, h) in enumerate(zip(cells, hours)):
        ax2.text(k, h + 0.01, f"{h:.2f} h" + ("" if wall[c]["hours"] is not None else " (tasks only)"), ha="center", va="bottom", fontsize=6.8, color=INK2)
    ax2.set_xticks(range(len(cells))); ax2.set_xticklabels([short_cell(c, analysis) for c in cells], fontsize=7.2, rotation=15, ha="right")
    ax2.set_ylabel("wall clock (h) per cell container"); ax2.set_ylim(0, max(0.5, max(hours or [0.0]) * 1.35)); _recessive(ax2, grid_axis="y")
    finish_figure(fig, title, subtitle)
    return fig


def fig_ladder(analysis: Mapping[str, Any]):
    plt, Patch = _mpl()
    lad, series = analysis["ladder"], _cell_series(analysis)
    width = 0.8 / len(series)
    if not lad["applied"]:
        title = (f"Truncation ladder — {_wave_tag(analysis)}: " + ("no leg fired (every cell ≤ 2/200 at the cap): the rung-1 reading stands alone"
                 if not lad["fired_any"] else "a leg FIRED but the 16,384-cap re-run is not merged yet: acceptance check 4 NOT MET"))
        subtitle = ("the rule (2026-09-10): a leg fires when more than 1 % of a cell's responses stop at the cap; bars = at-cap counts per leg and cell "
                    "against the 2/200 line; the rerun is certified per fitted cell at the wave's common rung")
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        _fit_header(fig, title, subtitle, bottom_in=0.62, left=0.1, right=0.97)
        tasks = list(lad["at_cap_counts"])
        for k, t in enumerate(tasks):
            for j, (cell, col) in enumerate(series):
                v = lad["at_cap_counts"][t][cell]["n_at_cap"]
                x = k + (j - (len(series) - 1) / 2) * width
                ax.bar(x, v, width=width * 0.9, color=col, zorder=3)
                ax.text(x, v + 0.1, str(v), ha="center", va="bottom", fontsize=6.8, color=INK2)
        ax.axhline(2, color=STATUS["warning"], linestyle="--", linewidth=1.0)
        ax.set_xticks(range(len(tasks))); ax.set_xticklabels([TASK_LABEL[t] for t in tasks], fontsize=8); ax.set_ylabel("items at the cap (of 200)")
        ax.set_ylim(0, max(4.0, max(lad["at_cap_counts"][t][c]["n_at_cap"] for t in tasks for c, _ in series) * 1.4))
        _cell_legend(ax, Patch, analysis, loc="upper right", fontsize=7.4, ncol=min(3, len(series)))
        _recessive(ax, grid_axis="y")
        finish_figure(fig, title, subtitle)
        return fig
    acc, tasks = lad["accepted"], list(lad["tasks"])
    withdrawn = analysis.get(WITHDRAWN_LEGS_KEY) or {}
    title = (f"Truncation ladder — {_wave_tag(analysis)}: rung 1 (OFF cap) vs merged (capped items re-run at 16,384), {len(tasks)} fired leg(s)"
             + (" — " + ", ".join(TASK_SHORT[t] for t in tasks if t in withdrawn) + " WITHDRAWN (D79)" if any(t in withdrawn for t in tasks) else ""))
    subtitle = (f"rung 2 served at L = {acc['max_model_len']:,} × S = {acc['max_num_seqs']} (seqs_source {acc['seqs_source']}, rung_source "
                f"{acc['rung_source']}{' — OPERATOR OVERRIDE' if lad['operator_rung'] else ''}); certificates for {len(lad['certificates'])} fitted cell(s); "
                + "; ".join(f"{TASK_SHORT[t]} n_selected = {lad['tasks'][t]['n_selected']}" for t in tasks) + "; whiskers = Wilson 95 %")
    fig, ax = plt.subplots(figsize=(max(9.0, 1.4 * len(series) * len(tasks) + 2), 5.0))
    _fit_header(fig, title, subtitle, bottom_in=0.85, left=0.09, right=0.98)
    labels, x = [], 0
    for t in tasks:
        for cell, _ in series:
            r = lad["tasks"][t]["cells"][cell]
            for j, (which, col) in enumerate((("rung1", SEQ[250]), ("merged", CAT["blue"]))):
                pr = r[which]
                rate = 100 * pr["rate"]
                xx = x + (j - 0.5) * 0.36
                ax.bar(xx, rate, width=0.34, color=col, zorder=3)
                ax.errorbar(xx, rate, yerr=[[rate - 100 * pr["wilson_low"]], [100 * pr["wilson_high"] - rate]], fmt="none", ecolor=INK2, elinewidth=1.0, capsize=3, zorder=4)
            ax.text(x, 3, f"at-cap {r['n_at_cap_rung1']} → {r['n_at_cap_residual']}", ha="center", va="bottom", fontsize=6.4, color=SURFACE)
            labels.append(f"{TASK_SHORT[t]}{' (WITHDRAWN)' if t in withdrawn else ''}\n{short_cell(cell, analysis)}")
            x += 1
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=7.0); ax.set_ylim(0, 100); ax.set_ylabel("pass rate (%) over the 200 items")
    ax.legend(handles=[Patch(color=SEQ[250], label="rung 1 (OFF cap)"), Patch(color=CAT["blue"], label="merged (rung 2 at 16,384 for the selected)")],
              loc="upper right", frameon=False, fontsize=7.4)
    _recessive(ax, grid_axis="y")
    finish_figure(fig, title, subtitle)
    return fig


def fig_gpu_budget(analysis: Mapping[str, Any]):
    plt, _ = _mpl()
    gpu = analysis["cost"]["gpu"]
    title = (f"GPU envelope — lane Σ {_num(gpu['committed_h'], 2)} of {_num(gpu['ceiling_h'], 1)} H200-h ({gpu.get('ceiling_source', '')}); "
             f"this wave {_num(gpu['wave_h'], 2)} h" if gpu["present"] else "GPU envelope — no meter ledger found; nothing to meter")
    subtitle = ("cumulative H200-hours by ledger reservation across the lane (actual where recorded, else forecast) against the 25 h spec envelope; "
                "a recorded 'ladder fired' decision moves the line by ≤ 1.5 h per leg and is named here; this wave's reservations highlighted")
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    _fit_header(fig, title, subtitle, bottom_in=1.0, left=0.1, right=0.97)
    if gpu["present"] and gpu["rows"]:
        rows = list(gpu["rows"])
        cum, tot = [], 0.0
        for row in rows:
            tot += row["counted"]; cum.append(tot)
        ax.step(range(len(rows)), cum, where="mid", color=CAT["blue"], linewidth=1.6, zorder=3)
        mine = [k for k, r in enumerate(rows) if r.get("wave") == analysis["wave"]["name"]]
        ax.scatter(range(len(rows)), cum, color=[CAT["orange"] if k in mine else CAT["blue"] for k in range(len(rows))], s=20, zorder=4)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([f"{r['wave'] or ''}/{r['unit']}"[:28] for r in rows], fontsize=6.6, rotation=35, ha="right")
        ax.set_ylim(0, max(gpu["ceiling_h"], tot) * 1.15)
        ax.axhline(gpu["ceiling_h"], color=STATUS["critical"] if gpu.get("ceiling_source") == "decision" else STATUS["warning"], linestyle="--", linewidth=1.0)
        ax.text(0.0, gpu["ceiling_h"] + 0.2, f"{gpu['ceiling_h']:.1f} h — {gpu.get('ceiling_source')}", fontsize=7.0, color=STATUS["warning"])
    else:
        ax.text(0.5, 0.5, "no ledger", transform=ax.transAxes, ha="center", va="center", color=INK2); ax.set_xticks([]); ax.set_ylim(0, 26)
        ax.axhline(M.CEILING_H, color=STATUS["warning"], linestyle="--", linewidth=1.0)
    ax.set_ylabel("cumulative H200-hours")
    _recessive(ax, grid_axis="y")
    finish_figure(fig, title, subtitle)
    return fig


def _status_color(status: str) -> str:
    return {"runnable": STATUS["good"], "queued": STATUS["warning"], "blocked": STATUS["serious"]}.get(status.split(":")[0], MUTED)


#: the roster figures carry TWO encodings — ELIGIBILITY (the roster's status per entry × leg: the cell's FILL) and RESULT
#: status (what the analysis holds for that cell × leg: a MARKER); conflating them loses the provisional mark, draws
#: pending as measured and drops blocked legs. Result vocabulary:
RESULT_MEASURED, RESULT_PROVISIONAL, RESULT_PENDING, RESULT_WITHDRAWN, RESULT_OUT, RESULT_NOT_RUN = (
    "measured", "provisional", "pending", "withdrawn", "out", "not run")
RESULT_STATUSES: Tuple[str, ...] = (RESULT_MEASURED, RESULT_PROVISIONAL, RESULT_PENDING, RESULT_WITHDRAWN, RESULT_OUT, RESULT_NOT_RUN)
#: marker, filled?
RESULT_MARKER: Mapping[str, Tuple[str, bool]] = {RESULT_MEASURED: ("o", True), RESULT_PROVISIONAL: ("o", False), RESULT_PENDING: ("D", False),
                                                 RESULT_WITHDRAWN: ("X", True), RESULT_OUT: ("_", True), RESULT_NOT_RUN: ("", False)}


def roster_grid_items(view: Mapping[str, Any]) -> List[Tuple[str, str, Dict[str, Tuple[str, str]]]]:
    """Per entry of an analysed wave's (gated) view: ``(row label, right-hand note, {task: (eligibility, result)})`` over the lane's
    two legs. Eligibility = the entry's roster status, else the leg's CURRENT status (a runnable cell on a blocked / withdrawn leg is
    `blocked` there). Result = `out` (measured, gated out by the current roster), `withdrawn`, `measured` / `provisional`
    (a final reading; the batch-8 generation keeps its mark), `pending` (fired, ladder not applied), else `not run`."""
    w = view["wave"]
    out_cells = (view.get(PUBLICATION_KEY) or {}).get("out") or {}
    withdrawn = view.get(WITHDRAWN_LEGS_KEY) or {}
    finals = final_readings(view)
    legs_now = w.get("legs") or {}
    items = []
    for e in w["entries"]:
        cell = e.get("cell_id")
        gen = (view["generations"].get(cell) or {}) if cell else {}
        note = roster_status_label(e["status"], e.get("reason")) + (" · " + gen["object_generation"] if gen.get("object_generation") else "")
        if e["status"] != "runnable":
            note += f" — {str(e.get('reason') or '')[:70]}"
        per_leg: Dict[str, Tuple[str, str]] = {}
        for task in TASKS:
            leg_status = str((legs_now.get(task) or {}).get("status") or "blocked")
            elig = e["status"] if e["status"] != "runnable" else leg_status
            if cell in out_cells and task in view["legs"]:
                res = RESULT_OUT
            elif task in withdrawn and cell in view["cells"]:
                res = RESULT_WITHDRAWN
            elif cell in view["cells"] and task in view["legs"]:
                fin = finals[task]
                res = (RESULT_PROVISIONAL if gen.get("provisional") else RESULT_MEASURED) if fin["final"] else RESULT_PENDING
            else:
                res = RESULT_NOT_RUN
            per_leg[task] = (elig, res)
        items.append((f"{e['entry_id']} ({cell or 'no cell id'})", note, per_leg))
    return items


def _roster_grid(ax, items: Sequence[Tuple[str, str, Mapping[str, Tuple[str, str]]]], *, Patch, note_x: float = 2.15):
    """Draw the entry × leg grid: fill = eligibility, marker = result; the note to the right; a legend for BOTH encodings."""
    from matplotlib.lines import Line2D
    y = list(range(len(items)))[::-1]
    for yi, (_, note, per_leg) in zip(y, items):
        for j, task in enumerate(TASKS):
            elig, res = per_leg.get(task, ("blocked", RESULT_NOT_RUN))
            ax.barh(yi, 0.92, left=j, color=_status_color(elig), height=0.62, zorder=3,
                    hatch=("////" if res == RESULT_OUT else None), edgecolor=(SURFACE if res == RESULT_OUT else "none"), linewidth=0.5)
            marker, filled = RESULT_MARKER[res]
            if marker:
                # the unfilled `_` marker (out) draws with its facecolor; an edgecolor there is ignored with a matplotlib warning
                ax.scatter([j + 0.46], [yi], marker=marker, s=(42 if marker != "_" else 90), facecolors=(INK if filled else SURFACE),
                           edgecolors=(None if marker == "_" else INK), linewidths=1.2, zorder=5)
        ax.text(note_x, yi, note, va="center", ha="left", fontsize=6.6, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([label for label, _, _ in items], fontsize=6.9)
    ax.set_xticks([j + 0.46 for j in range(len(TASKS))]); ax.set_xticklabels([TASK_SHORT[t] for t in TASKS], fontsize=7.4)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.05, 6.4)
    ax.set_ylim(-3.0, len(items) - 0.4)                  # a band below the last row for the legend, so it never covers a note
    handles = [Patch(color=_status_color(s), label=f"eligibility: {s}") for s in ROSTER_STATUSES]
    for res in RESULT_STATUSES:
        marker, filled = RESULT_MARKER[res]
        if marker:
            handles.append(Line2D([], [], marker=marker, linestyle="", color=INK, markerfacecolor=(INK if filled else SURFACE), markeredgecolor=INK,
                                  markersize=6.5, label=f"result: {res}"))
        else:
            handles.append(Patch(facecolor="none", edgecolor="none", label=f"result: {res} (no marker)"))
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=6.6, ncol=2)
    _recessive(ax, grid_axis="none")


def _roster_counts(items: Sequence[Tuple[str, str, Mapping[str, Tuple[str, str]]]]) -> str:
    """``N runnable, N queued, N blocked entries; N measured, N provisional, … cell × legs`` for a roster figure's title."""
    entry_status = [max(per_leg.values(), key=lambda v: (v[0] == "runnable"))[0] for _, _, per_leg in items]      # runnable if any leg is
    ent = ", ".join(f"{sum(s == st for s in entry_status)} {st}" for st in ROSTER_STATUSES)
    results = [r for _, _, per_leg in items for _, r in per_leg.values()]
    res = ", ".join(f"{results.count(r)} {r}" for r in RESULT_STATUSES if results.count(r))
    return f"{ent} entries; cell × leg results: {res}"


def fig_roster_status(analysis: Mapping[str, Any]):
    plt, Patch = _mpl()
    items = roster_grid_items(analysis)
    title = f"Roster — {_wave_tag(analysis)}: {_roster_counts(items)}"
    subtitle = ("fill = ELIGIBILITY under the current roster (entry status, or the leg's where the entry is runnable); marker = RESULT of this analysis "
                "for that cell × leg (measured ●, provisional ○ = batch-8 object re-run pending, pending ◇ = fired ladder not applied, withdrawn ✕ = the "
                "roster blocks a measured leg (D79), out = measured cell gated out); what ran and what did not, by name (spec criterion 5): a queued / "
                "blocked entry carries its reason and unblocking trigger in the doc, never dropped and never filled")
    fig, ax = plt.subplots(figsize=(11.0, max(4.4, 0.42 * len(items) + 3.0)))
    _fit_header(fig, title, subtitle, bottom_in=0.55, left=0.24, right=0.98)
    _roster_grid(ax, items, Patch=Patch)
    finish_figure(fig, title, subtitle)
    return fig


FIGURES: Dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "pass_rate_by_cell": fig_pass_rate_by_cell, "paired_delta": fig_paired_delta, "bfcl_by_category": fig_bfcl_by_category,
    "bcb_by_family": fig_bcb_by_family, "diagnostics": fig_diagnostics, "cost_wall": fig_cost_wall, "ladder": fig_ladder,
    "gpu_budget": fig_gpu_budget, "roster_status": fig_roster_status}
assert tuple(FIGURES) == FIGURE_STEMS


def make_figures(analysis: Mapping[str, Any], figs_dir: Union[str, Path], *, figures: Mapping[str, Callable] = FIGURES) -> Dict[str, List[str]]:
    figs_dir = Path(figs_dir).expanduser()
    out: Dict[str, List[str]] = {}
    for stem, fn in figures.items():
        fig = fn(analysis)
        out[stem] = save_figure(fig, figs_dir, f"final_{stem}")
    return out


# --------------------------------------------------------------------------- #
# --summary figures (cross-family / cross-object)                              #
# --------------------------------------------------------------------------- #


def _summary_rows(summary: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Every (wave, steered cell, leg) contrast of the summary from the leg's FINAL reading
    (:func:`final_readings`); a leg without one (fired, ladder pending) yields a row with
    ``paired`` None and ``source`` "pending" — listed, never drawn as a reading."""
    rows = []
    for wv in summary["waves"]:
        finals = final_readings(wv)
        for task, leg in wv["legs"].items():
            fin = finals[task]
            for cell in leg["paired"]:
                p = (fin["paired"] or {}).get(cell) if fin["final"] else None
                rows.append({"wave": wv["wave"]["name"], "family": wv["wave"]["family"], "lane_kind": wv["wave"]["lane_kind"], "cell": cell,
                             "arm": short_cell(cell, wv), "task": task, "paired": p, "source": fin["source"], "final": fin["final"],
                             "provisional": bool(leg["paired"][cell].get("provisional")), "why": fin["why"]})
    return rows


def fig_summary_paired_delta(summary: Mapping[str, Any]):
    plt, _ = _mpl()
    rows = _summary_rows(summary)
    drawn = [r for r in rows if r["final"]]
    withdrawn = [r for r in rows if r["source"] == FINAL_SOURCE_WITHDRAWN]           # excluded, named apart from pending
    pending = [r for r in rows if not r["final"] and r["source"] != FINAL_SOURCE_WITHDRAWN]
    fam_colors = {}
    for r in drawn:
        fam_colors.setdefault(r["family"], STEERED_COLORS[len(fam_colors) % len(STEERED_COLORS)])
    forest = [(f"{r['family']} · {r['lane_kind']} · {r['arm']} · {TASK_SHORT[r['task']]} ({r['source']})", r["paired"], fam_colors[r["family"]]) for r in drawn]
    n_out = sum(len((wv.get(PUBLICATION_KEY) or {}).get("out") or {}) for wv in summary["waves"])
    title = (f"Paired difference vs each family's own none — {len(summary['waves'])} wave(s), {len(drawn)} final (object, leg) contrast(s)"
             + (f", {len(pending)} pending" if pending else "") + (f", {len(withdrawn)} withdrawn" if withdrawn else "")
             + (f", {n_out} cell(s) out under the current roster" if n_out else ""))
    # the pending legs are named per (wave, leg) with a CELL COUNT, never cell by cell: the subtitle is capped at
    # A.SUBTITLE_MAX_LINES and a per-cell list grows with the wave (a four-steered-cell wave pushed it to a 4th line)
    pending_by_leg: Dict[Tuple[str, str], int] = {}
    for r in pending:
        pending_by_leg[(r["wave"], r["task"])] = pending_by_leg.get((r["wave"], r["task"]), 0) + 1
    withdrawn_legs_named = sorted({f"{r['wave']} {TASK_SHORT[r['task']]}" for r in withdrawn})
    subtitle = ("every point is a leg's FINAL reading (merged after a fired ladder, else rung 1) of steered − none within ONE wave (same tree, suffix, "
                "protocol decoding); whisker = paired bootstrap 95 % (estimation interval); COLOURED when the reading rule detects a cost (exact McNemar "
                "p < 0.05 and Δ < 0; nominal, unadjusted); no cross-object or cross-family contrast; '(prov.)' = batch-8 object"
                + ("; not drawn (ladder pending): " + ", ".join(f"{wave} {TASK_SHORT[task]} × {n}" for (wave, task), n in pending_by_leg.items())
                   if pending else "")
                + ("; WITHDRAWN (D79; not drawn): " + ", ".join(withdrawn_legs_named) if withdrawn else ""))
    fig, ax = plt.subplots(figsize=(10.0, max(3.6, 0.5 * max(1, len(forest)) + 1.8)))
    _fit_header(fig, title, subtitle, bottom_in=0.65, left=0.42, right=0.97)
    if forest:
        _forest(ax, forest, label_p=False)
    else:
        ax.text(0.5, 0.5, "no final contrast yet (every fired leg is pending its ladder)", transform=ax.transAxes, ha="center", va="center", color=INK2)
        ax.set_yticks([]); ax.set_xticks([])
        _recessive(ax, grid_axis="none")
    finish_figure(fig, title, subtitle)
    return fig


def fig_summary_pass_rate(summary: Mapping[str, Any]):
    plt, Patch = _mpl()
    rows = []
    for wv in summary["waves"]:
        finals = final_readings(wv)
        for task, leg in wv["legs"].items():
            fin = finals[task]
            if not fin["final"]:
                continue                                    # a pending leg has no final pass rate to draw
            for cell, pr in fin["cells"].items():
                rows.append((f"{wv['wave']['family']} · {wv['wave']['lane_kind']} · {short_cell(cell, wv)}", task, pr, cell == BASE_CELL))
    withdrawn_named = sorted({f"{wv['wave']['name']} {TASK_SHORT[t]}" for wv in summary["waves"] for t in (wv.get(WITHDRAWN_LEGS_KEY) or {})})
    title = (f"Pass rate per (wave, cell, leg) — {len(summary['waves'])} wave(s) under the benchmarks' protocol decoding (final readings)"
             + (f"; {len(withdrawn_named)} leg(s) withdrawn" if withdrawn_named else ""))
    subtitle = ("bar = k/200, whisker = Wilson 95 %; grey = the wave's unsteered cell; per leg side by side; a leg pending its ladder is not drawn"
                + ("; WITHDRAWN (instrument failure, D79; not drawn): " + ", ".join(withdrawn_named) if withdrawn_named else ""))
    labels = sorted({r[0] for r in rows}, key=lambda s: [r[0] for r in rows].index(s))
    tasks = [t for t in TASKS if any(r[1] == t for r in rows)]
    fig, ax = plt.subplots(figsize=(10.0, max(3.6, 0.5 * max(1, len(labels)) + 1.8)))
    _fit_header(fig, title, subtitle, bottom_in=0.6, left=0.4, right=0.97)
    y = list(range(len(labels)))[::-1]
    h = 0.8 / max(1, len(tasks))
    shades = {BCB_TASK: SEQ[450], BFCL_TASK: SEQ[250]}
    for label, task, pr, is_base in rows:
        yi = y[labels.index(label)] + ((len(tasks) - 1) / 2 - tasks.index(task)) * h
        rate = 100 * (pr["rate"] or 0.0)
        _bar_h(ax, yi, rate, MUTED if is_base else shades[task], height=h * 0.9)
        ax.errorbar([rate], [yi], xerr=[[rate - 100 * pr["wilson_low"]], [100 * pr["wilson_high"] - rate]], fmt="none", ecolor=INK2, elinewidth=0.8, capsize=2, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.4); ax.set_xlim(0, 100); ax.set_xlabel("pass rate (%)")
    if not rows:
        ax.text(0.5, 0.5, "no final pass rate yet (every leg is pending its ladder)", transform=ax.transAxes, ha="center", va="center", color=INK2)
    ax.legend(handles=[Patch(color=shades[t], label=TASK_SHORT[t]) for t in tasks] + [Patch(color=MUTED, label="none (unsteered)")], loc="lower right", frameon=False, fontsize=7.2)
    _recessive(ax, grid_axis="x")
    finish_figure(fig, title, subtitle)
    return fig


def fig_summary_roster(summary: Mapping[str, Any]):
    plt, Patch = _mpl()
    items: List[Tuple[str, str, Dict[str, Tuple[str, str]]]] = []
    for wv in summary["waves"]:
        items += [(f"{wv['wave']['name']} · {label}", note, per_leg) for label, note, per_leg in roster_grid_items(wv)]
    # the roster's waves WITHOUT an analysis: eligibility from the roster (entry status, else the wave's leg status); nothing measured
    unanalysed_legs = summary.get("unanalysed_wave_legs") or {}
    for wname, e in summary.get("unanalysed_entries", []):
        legs_now = unanalysed_legs.get(wname) or {}
        per_leg = {t: ((e["status"] if e["status"] != "runnable" else str((legs_now.get(t) or {}).get("status") or "blocked")), RESULT_NOT_RUN)
                   for t in TASKS}
        note = roster_status_label(e["status"], e.get("reason")) + (f" — {str(e.get('reason') or '')[:60]}" if e["status"] != "runnable" else "")
        items.append((f"{wname} · {e['entry_id']} ({e.get('cell_id') or 'no cell id'})", note, per_leg))
    n_waves = len(summary["waves"]) + len({w for w, _ in summary.get("unanalysed_entries", [])})
    title = f"Roster across the lane — {n_waves} wave(s): {_roster_counts(items)}"
    subtitle = ("fill = ELIGIBILITY under the current roster (entry status, or the leg's where the entry is runnable); marker = RESULT for that cell × leg "
                "(measured ●, provisional ○ = batch-8 object re-run pending, pending ◇ = fired ladder not applied, withdrawn ✕ = the roster blocks a "
                "measured leg (D79), out = measured cell gated out; no marker = not run); a queued / blocked entry carries its reason in the doc and "
                "its unblocking trigger; nothing is filled")
    fig, ax = plt.subplots(figsize=(11.6, max(4.6, 0.36 * len(items) + 3.0)))
    _fit_header(fig, title, subtitle, bottom_in=0.55, left=0.36, right=0.98)
    _roster_grid(ax, items, Patch=Patch)
    finish_figure(fig, title, subtitle)
    return fig


def fig_summary_gpu_budget(summary: Mapping[str, Any]):
    plt, _ = _mpl()
    gpu = summary["cost"]["gpu"]
    title = (f"GPU envelope across the lane — Σ {_num(gpu['committed_h'], 2)} of {_num(gpu['ceiling_h'], 1)} H200-h ({gpu.get('ceiling_source', '')})"
             if gpu["present"] else "GPU envelope across the lane — no meter ledger found")
    subtitle = "H200-hours by wave from the meter's ledger (actual where recorded, else forecast) against the 25 h envelope; a recorded decision names its fired leg"
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    _fit_header(fig, title, subtitle, bottom_in=0.8, left=0.1, right=0.97)
    if gpu["present"]:
        waves = list(gpu["by_wave"])
        ax.bar(range(len(waves)), [gpu["by_wave"][w] for w in waves], color=CAT["blue"], width=0.6, zorder=3)
        for k, w in enumerate(waves):
            ax.text(k, gpu["by_wave"][w] + 0.05, f"{gpu['by_wave'][w]:.2f}", ha="center", va="bottom", fontsize=7.0, color=INK2)
        ax.set_xticks(range(len(waves))); ax.set_xticklabels(waves, fontsize=7.4, rotation=20, ha="right")
        ax.set_ylim(0, max(1.0, max(gpu["by_wave"].values(), default=0.0) * 1.3))
    else:
        ax.text(0.5, 0.5, "no ledger", transform=ax.transAxes, ha="center", va="center", color=INK2); ax.set_xticks([])
    ax.set_ylabel("H200-hours (by wave)")
    _recessive(ax, grid_axis="y")
    finish_figure(fig, title, subtitle)
    return fig


SUMMARY_FIGURES: Dict[str, Callable[[Mapping[str, Any]], Any]] = {
    "summary_paired_delta": fig_summary_paired_delta, "summary_pass_rate": fig_summary_pass_rate, "summary_roster": fig_summary_roster,
    "summary_gpu_budget": fig_summary_gpu_budget}
assert tuple(SUMMARY_FIGURES) == SUMMARY_FIGURE_STEMS


# --------------------------------------------------------------------------- #
# W&B                                                                          #
# --------------------------------------------------------------------------- #


def wandb_scalars(analysis: Mapping[str, Any]) -> Dict[str, Any]:
    """The FINAL reading per leg under the plain keys (None — dropped by `log_wandb` — while a
    fired leg is pending); the rung-1 numbers ONLY under the explicit `<leg>/rung1/…` keys."""
    out: Dict[str, Any] = {}
    finals = final_readings(analysis)
    for task, leg in analysis["legs"].items():
        short = TASK_SHORT[task].lower().replace("-", "")
        fin = finals[task]
        out[f"{short}/final_source"] = fin["source"]
        out[f"{short}/final"] = int(fin["final"])
        if fin["source"] == FINAL_SOURCE_WITHDRAWN:
            # a withdrawn leg logs its status and nothing that reads as a reading — not even under the rung-1 keys
            out[f"{short}/withdrawn"] = 1
            out[f"{short}/fired"] = int(leg["fired"])
            continue
        for cell, r in leg["cells"].items():
            tag = short_cell(cell).replace(" (prov.)", "")
            fr = (fin["cells"] or {}).get(cell) if fin["final"] else None
            out[f"{short}/{tag}/pass_rate"] = fr["rate"] if fr else None
            out[f"{short}/{tag}/wilson_low"] = fr["wilson_low"] if fr else None
            out[f"{short}/{tag}/wilson_high"] = fr["wilson_high"] if fr else None
            out[f"{short}/rung1/{tag}/pass_rate"] = r["rate"]
            out[f"{short}/rung1/{tag}/n_at_cap"] = r["at_cap"]["n_at_cap"]
            out[f"{short}/rung1/{tag}/unparseable_rate"] = r["unparseable"]["rate"]
        for cell, p in leg["paired"].items():
            tag = short_cell(cell).replace(" (prov.)", "")
            fp = (fin["paired"] or {}).get(cell) if fin["final"] else None
            out[f"{short}/paired/{tag}/delta"] = fp["delta"] if fp else None
            out[f"{short}/paired/{tag}/ci_low"] = fp["ci_low"] if fp else None
            out[f"{short}/paired/{tag}/ci_high"] = fp["ci_high"] if fp else None
            out[f"{short}/paired/{tag}/mcnemar_p"] = fp["mcnemar_exact_p"] if fp else None
            out[f"{short}/paired/{tag}/n_discordant"] = fp["n_discordant"] if fp else None
            out[f"{short}/paired/{tag}/provisional"] = int(bool(p.get("provisional")))
            out[f"{short}/rung1/paired/{tag}/delta"] = p["delta"]
            out[f"{short}/rung1/paired/{tag}/mcnemar_p"] = p["mcnemar_exact_p"]
        out[f"{short}/fired"] = int(leg["fired"])
    gpu = analysis["cost"]["gpu"]
    out["cost/lane_gpu_h"] = gpu["committed_h"]
    out["cost/wave_gpu_h"] = gpu["wave_h"]
    out["cost/sandbox_usd"] = analysis["cost"]["sandbox_usd_estimate"]
    out["cost/wave_usd"] = analysis["cost"]["wave_usd_estimate"]
    out["ladder/applied"] = int(analysis["ladder"]["applied"])
    return out


def log_wandb(analysis: Mapping[str, Any], figures: Mapping[str, Sequence[str]], figs_dir: Union[str, Path], *, mode: Optional[str],
              project: str = WANDB_PROJECT, name: Optional[str] = None, scalars: Optional[Mapping[str, Any]] = None,
              image_prefix: str = "bcb_bfcl_200_widen") -> Optional[Dict[str, Any]]:
    """``wandb.init(project="capeval", job_type="bcb-bfcl-200-widen-analysis")``: scalars as
    summary, every figure as an Image, the figure dir as an artifact. ``mode``
    "disabled" returns None without importing wandb (the tests' path). The API
    key is read by wandb from the environment; nothing here touches it."""
    if mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as e:    # pragma: no cover - venv-dependent
        raise AnalysisError("wandb is not importable in this venv — pass --no-wandb") from e
    ident = analysis.get("identity") or {}
    wb_name = name or analysis["run_name"]
    kwargs: Dict[str, Any] = {"project": project, "name": wb_name, "job_type": WANDB_JOB_TYPE,
                              "config": {"run_name": analysis["run_name"], "lane": LANE, "wave": (analysis.get("wave") or {}).get("name"),
                                         "cells": analysis.get("cells"), "tasks": analysis.get("tasks"), "model_id": ident.get("model_id"),
                                         "model_revision": ident.get("model_revision"), "tree_sha": ident.get("tree_sha"), "served": ident.get("served"),
                                         "caps": ident.get("caps"), "object_sha256": ident.get("object_sha256"),
                                         "generations": {c: g.get("object_generation") for c, g in (analysis.get("generations") or {}).items()},
                                         "generator": analysis["generator"]}}
    if mode:
        kwargs["mode"] = mode
    run = wandb.init(**kwargs)
    run.summary.update({k: v for k, v in (scalars if scalars is not None else wandb_scalars(analysis)).items() if v is not None})
    images = {f"{image_prefix}/{stem}": wandb.Image(next(p for p in paths if p.endswith(".png")))
              for stem, paths in figures.items() if any(p.endswith(".png") for p in paths)}
    if images:
        wandb.log(images)
    art = wandb.Artifact(f"figures-{wb_name}", type="figures")
    art.add_dir(str(Path(figs_dir).expanduser()))
    run.log_artifact(art)
    info = {"project": project, "name": wb_name, "id": getattr(run, "id", None), "url": getattr(run, "url", None),
            "mode": mode or os.environ.get("WANDB_MODE") or "online"}
    run.finish()
    return info


# --------------------------------------------------------------------------- #
# --summary                                                                    #
# --------------------------------------------------------------------------- #


def summarize(analysis_paths: Sequence[Union[str, Path]], *, roster_path: Optional[Union[str, Path]], ledger: Union[str, Path]) -> Dict[str, Any]:
    """The cross-wave record: every wave's analysis.json (schema 2, one per wave;
    two for one wave refuse), the roster's waves that have no analysis yet
    (their entries exported as not run), the lane's GPU ledger across waves."""
    if not roster_path:
        raise AnalysisError("--summary needs the roster: every saved analysis is gated by the CURRENT roster's publication eligibility "
                            "(gate_saved_analysis) before its readings are summarised, drawn or exported")
    roster = load_roster(roster_path)
    waves: List[Dict[str, Any]] = []
    seen: Dict[str, str] = {}
    for p in analysis_paths:
        a = read_json(Path(p).expanduser())
        if not isinstance(a, Mapping) or a.get("schema") != ANALYSIS_SCHEMA_VERSION or a.get("lane") != LANE:
            raise AnalysisError(f"{p}: not a {LANE} analysis.json (schema {ANALYSIS_SCHEMA_VERSION})")
        wname = a["wave"]["name"]
        if wname in seen:
            raise AnalysisError(f"two analyses for wave {wname!r}: {seen[wname]} and {p} — one per wave")
        seen[wname] = str(p)
        waves.append(gate_saved_analysis(a, roster, roster_path=roster_path))      # the saved file is read, never rewritten
    unanalysed: List[Tuple[str, Dict[str, Any]]] = []
    unanalysed_legs: Dict[str, Dict[str, Any]] = {}
    roster_waves: List[str] = []
    for wname in roster["waves"]:
        roster_waves.append(wname)
        if wname in seen:
            continue
        spec = wave_spec(roster, wname)
        unanalysed_legs[wname] = dict(spec.legs)          # the roster figure shows blocked LEGS of unanalysed waves too
        for e in spec.entries:
            unanalysed.append((wname, {"entry_id": e.entry_id, "cell_id": e.cell_id, "status": e.status, "reason": e.reason, "unblock": e.unblock,
                                       "paper_arm": e.paper_arm, "kind": e.kind}))
    gpu = gpu_cost_section(ledger, wave=None)
    readings = []
    for wv in waves:
        finals = final_readings(wv)
        held_cells = wv[PUBLICATION_KEY].get(HELD_OUT_KEY) or {}
        for task, leg in wv["legs"].items():
            fin = finals[task]
            for cell in leg["paired"]:
                if fin["source"] == FINAL_SOURCE_WITHDRAWN:
                    # one WITHDRAWN row per (wave, steered cell, leg) — the status and the reason, no delta, no p — never a silent drop
                    wd = fin["withdrawn"]
                    readings.append({"wave": wv["wave"]["name"], "family": wv["wave"]["family"], "lane_kind": wv["wave"]["lane_kind"], "cell": cell,
                                     "arm": short_cell(cell), "task": task, "source": FINAL_SOURCE_WITHDRAWN, "final": False,
                                     "delta_pp": None, "ci_low_pp": None, "ci_high_pp": None, "mcnemar_p": None,
                                     "reading": f"withdrawn — {_withdrawn_reason_text(wd.get('reason'))} (unblock: {wd.get('unblock')})",
                                     "provisional": bool(leg["paired"][cell].get("provisional")), "fired": bool(leg["fired"]),
                                     "check_4": wv["ladder"]["acceptance_check_4"], "out": False, "withdrawn": True, "held_out": False,
                                     "withdrawn_status": wd.get("status"), "withdrawn_reason": wd.get("reason")})
                    continue
                p = (fin["paired"] or {}).get(cell) if fin["final"] else None
                readings.append({"wave": wv["wave"]["name"], "family": wv["wave"]["family"], "lane_kind": wv["wave"]["lane_kind"], "cell": cell,
                                 "arm": short_cell(cell), "task": task, "source": fin["source"], "final": fin["final"],
                                 "delta_pp": (100 * p["delta"] if p else None), "ci_low_pp": (100 * p["ci_low"] if p else None),
                                 "ci_high_pp": (100 * p["ci_high"] if p else None), "mcnemar_p": (p["mcnemar_exact_p"] if p else None),
                                 "reading": (p["reading"] if p else f"pending — {fin['why']}"),
                                 "provisional": bool(leg["paired"][cell].get("provisional")), "fired": bool(leg["fired"]),
                                 "check_4": wv["ladder"]["acceptance_check_4"], "out": False, "withdrawn": False, "held_out": cell in held_cells})
        # a measured cell the CURRENT roster does not run: one OUT reading per leg — the status and the reason, no delta — never a silent drop
        for cell, rec in (wv[PUBLICATION_KEY].get("out") or {}).items():
            for task, leg in wv["legs"].items():
                readings.append({"wave": wv["wave"]["name"], "family": wv["wave"]["family"], "lane_kind": wv["wave"]["lane_kind"], "cell": cell,
                                 "arm": short_cell(cell, wv), "task": task, "source": OUT_SOURCE, "final": False,
                                 "delta_pp": None, "ci_low_pp": None, "ci_high_pp": None, "mcnemar_p": None,
                                 "reading": f"{rec['label']} — {rec['status']}: {rec['reason']}", "provisional": False, "fired": bool(leg["fired"]),
                                 "check_4": wv["ladder"]["acceptance_check_4"], "out": True, "out_status": rec["status"], "out_reason": rec["reason"],
                                 "held_out": False})
    return {"schema": ANALYSIS_SCHEMA_VERSION, "generator": "scripts/bcb_bfcl_200_analysis.py --summary", "lane": LANE,
            "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(), "waves": waves, "roster_waves": roster_waves,
            "out": {wv["wave"]["name"]: wv[PUBLICATION_KEY].get("out") or {} for wv in waves},
            WITHDRAWN_LEGS_KEY: {wv["wave"]["name"]: wv.get(WITHDRAWN_LEGS_KEY) or {} for wv in waves},
            "analysed_waves": sorted(seen), "unanalysed_waves": [w for w in roster_waves if w not in seen], "unanalysed_entries": unanalysed,
            "unanalysed_wave_legs": unanalysed_legs,
            "readings": readings, "cost": {"gpu": gpu, "sandbox_usd_estimate": sum(wv["cost"]["sandbox_usd_estimate"] for wv in waves),
                                           "basis": "PARTIAL subtotal: the analysed waves' production sandboxes + the ledger's reservations; smoke / canary "
                                                    "sandboxes, CPU-suite runs and staging containers are outside it; resource-rate estimates, not billing"},
            "budget_decisions": gpu.get("decisions") or []}


def summary_scalars(summary: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for r in summary["readings"]:
        key = f"{r['wave']}/{TASK_SHORT[r['task']].lower().replace('-', '')}/{r['arm'].replace(' (prov.)', '')}"
        out[f"{key}/delta_pp"] = r["delta_pp"]                  # None (dropped) while the leg is pending its ladder
        out[f"{key}/mcnemar_p"] = r["mcnemar_p"]
        out[f"{key}/final_source"] = r["source"]
        out[f"{key}/held_out"] = bool(r.get("held_out"))          # reported, but out of every pooled category mean (ruling (a))
    gpu = summary["cost"]["gpu"]
    out["cost/lane_gpu_h"] = gpu["committed_h"]
    out["cost/ceiling_h"] = gpu["ceiling_h"]
    return out


# --------------------------------------------------------------------------- #
# results doc                                                                  #
# --------------------------------------------------------------------------- #

PENDING = "_pending — this wave has not landed_"
#: the waves the plan names (§S7), for the skeleton when no roster is at hand; the roster is the source once it lands
PLANNED_WAVES: Tuple[Tuple[str, str], ...] = (("gemma4e4b_it", "gemma4e4b · instruct-direct"),
                                              ("gemma4e4b_main", "gemma4e4b · base→instruct (LoRA subtraction; the distilled difference PSR is out by default — user ruling 2026-09-11, D48)"),
                                              ("qwen35_4b_main", "qwen35_4b · base→instruct"), ("nemotron3nano4b_main", "nemotron3nano4b · chat-ckpt"),
                                              ("qwen35_4b_it", "qwen35_4b · instruct-direct (queued)"),
                                              # the queued `llama8b` wave became the two runnable llama8b waves
                                              ("llama8b_it", "llama8b · instruct-direct"), ("llama8b_main", "llama8b · base→instruct"))


#: the reason prefix that takes PSR out of the paper by default: the roster BLOCKS every PSR cell with a reason
#: that begins with this prefix; the doc names such a row "out"
PSR_OUT_RULING_PREFIX = "ruling of 2026-09-11"
PSR_OUT_LABEL = "out (ruling of 2026-09-11: PSR out by default)"


def roster_status_label(status: Optional[str], reason: Optional[str]) -> str:
    """How the doc names a roster entry's status: the roster's own vocabulary (`runnable` / `queued` /
    `blocked`), except a cell BLOCKED by the PSR-out rule, which reads as out of the paper."""
    r = str(reason or "").lower()
    if status == "blocked" and r.startswith(PSR_OUT_RULING_PREFIX) and "psr is out" in r:
        return PSR_OUT_LABEL
    return str(status)


def _fence(text: Optional[str], *, truncated: bool = False, chars: int = 0) -> str:
    body = text if text is not None else "(no text recorded)"
    tail = f"\n... (truncated: {chars:,} chars; the full text is in analysis.json)" if truncated else ""
    return "````text\n" + body + tail + "\n````"


def _prov_mark(a: Mapping[str, Any], cell: str) -> str:
    g = (a.get("generations") or {}).get(cell) or {}
    return f" — **provisional** (batch-8 fit; superseded by `{g.get('superseded_by_key')}` when staged)" if g.get("provisional") else ""


def render_wave_doc(a: Mapping[str, Any], *, figs_url_base: str = DEFAULT_FIGS_URL_BASE, wandb_url: Optional[str] = None, level: int = 2) -> str:
    """One wave's section of the results doc, every number traceable to its analysis.json."""
    out: List[str] = []
    w = out.append
    h = "#" * level
    wv, base = a["wave"], a["base"]

    def tag(cell: str) -> str:
        return short_cell(cell, a)

    prov = a["identity"]["provenance"]
    run_ids = prov.get("run_ids") or ([prov["run_id"]] if prov.get("run_id") else [])
    pub = a.get(PUBLICATION_KEY) or {}
    out_cells: Dict[str, Dict[str, Any]] = dict(pub.get("out") or {})
    withdrawn: Dict[str, Dict[str, Any]] = dict(a.get(WITHDRAWN_LEGS_KEY) or {})
    w(f"{h} Wave `{wv['name']}` — {wv['family']}, {wv['lane_kind']} (view `{wv['view']}`, row `{wv['model_row']}`)\n")
    w(f"_generated from `{a['run_name']}/analysis.json` on {a['generated_at']} · cells " + ", ".join(f"`{c}`" for c in a["cells"])
      + (" · out: " + ", ".join(f"`{c}`" for c in out_cells) if out_cells else "")
      + f" · legs {', '.join(TASK_SHORT[t] + (' (WITHDRAWN)' if t in withdrawn else '') for t in a['tasks'])} · suffix `{prov.get('suffix')}` · run id(s) "
      + ", ".join(f"`{r}`" for r in run_ids) + f" ({prov.get('n_attempts', len(run_ids))} attempt(s)) · tree `{a['identity']['tree_sha']}`_\n")
    if out_cells:
        w("**Not published under the current roster**" + (f" (`{pub['roster']}`, sha256 `{str(pub.get('roster_sha256'))[:12]}`)" if pub.get("roster") else "")
          + ": " + "; ".join(f"`{c}` — **{r['label']}** ({r['status']}: {r['reason']}; unblock: {r['unblock']})" for c, r in out_cells.items())
          + ". These cells were measured in this analysis (runnable when it was made); their readings are suppressed here, in the figures and in the "
            "summary, and the paper export refuses them — the saved analysis.json keeps them, unrewritten.\n")
    finals = final_readings(a)
    for task, wd in withdrawn.items():
        # the WITHDRAWN block — clearly labelled, quoting the kept numbers WITH the reason, so a reader sees why 56/200 = 56/200 was not a result
        fin = finals[task]
        kept = fin.get("withdrawn") or {}
        w(f"**WITHDRAWN — instrument failure: {TASK_LABEL[task]}** (the current roster"
          + (f" `{pub['roster']}`, sha256 `{str(kept.get('roster_sha256') or pub.get('roster_sha256'))[:12]}`" if pub.get("roster") else "")
          + f" marks leg `{ATOM_OF_TASK[task]}` **{wd.get('status')}** for this wave after it was measured). Reason: {wd.get('reason')}. "
          f"Unblock: {wd.get('unblock')}. The leg's stores were analysed (kept from the {kept.get('kept_from')} reading) and their numbers are kept "
          f"here and, under `final.<leg>.withdrawn`, in {withdrawn_record_location(a)} FOR THE RECORD — they are not a capability reading and appear "
          "in no headline, summary, figure or paper value (the paper rows carry `status: withdrawn`):")
        for cell, pr in (kept.get("cells") or {}).items():
            line = f"  - {tag(cell)} (`{cell}`): {_interval(pr)} at cap {(kept.get('cap_by_cell') or {}).get(cell, '?')}"
            p = (kept.get("paired") or {}).get(cell)
            if p:
                line += (f"; vs none Δ {_pp(p['delta'])} [{_pp(p['ci_low'])}, {_pp(p['ci_high'])}], b = {p['b_pass_to_fail']}, c = {p['c_fail_to_pass']}, "
                         f"exact McNemar {_p(p['mcnemar_exact_p'])} — NOT a reading (withdrawn)")
            w(line + ".")
        w("")
    if pub.get("unmeasured_runnable"):
        w(f"_Runnable in the current roster but not measured here (the analysis predates them): {', '.join(f'`{c}`' for c in pub['unmeasured_runnable'])}._\n")
    if pub.get(HELD_OUT_KEY):
        w("**Reported, HELD OUT of pooled category means** (the current roster's `pooling` declaration — the readings below stand, the figures draw them, "
          "the paper rows carry `held_out: true` so the notebook averages nothing over them): "
          + " ".join(f"`{c}` — {blk.get('reason')} _({blk.get('ruled_by')})_." for c, blk in pub[HELD_OUT_KEY].items()) + "\n")
    w("Every contrast below is a steered cell against THIS wave's `none` (same suffix, same tree, same decoding); no cross-object contrast is "
      "computed, so no batch-8 object is ever paired with a batch-64 one (D7). Each leg's **final reading** is the merged one after a fired "
      "ladder, else rung 1; a fired leg whose ladder is not applied has NO final reading (its rung-1 numbers appear only as diagnostics); a leg the "
      "current roster blocks after it was measured is WITHDRAWN (its numbers kept for the record, never as a reading).")
    w(f"\n{h}# Headline (final readings)\n")
    # the reading rule and the unadjusted caveat, once per wave section, above the rows
    w(f"_Reading rule: {READING_RULE_SENTENCE}. Every test and interval below is {UNADJUSTED_NOTE}._\n")
    for task, leg in a["legs"].items():
        fin = finals[task]
        out_lines = [f"  - {tag(c)} (`{c}`): **{r['label']}** — not published ({r['status']}: {r['reason']})." for c, r in out_cells.items()]
        if fin["source"] == FINAL_SOURCE_WITHDRAWN:
            w(f"- **{TASK_LABEL[task]}** — **WITHDRAWN**: {_withdrawn_reason_text(fin['withdrawn'].get('reason'))}. No reading; the kept numbers are in the "
              "WITHDRAWN block above.")
            continue
        if not fin["final"]:
            w(f"- **{TASK_LABEL[task]}** — **no final reading**: {fin['why']}. Rung 1 (diagnostic, NOT final): `none` {_interval(leg['cells'][base])}; "
              + "; ".join(f"{tag(c)} {_interval(leg['cells'][c])}, Δ {_pp(p['delta'])}" for c, p in leg["paired"].items()) + ".")
            for line in out_lines:
                w(line)
            continue
        w(f"- **{TASK_LABEL[task]}** (final: {fin['source']}) — `none` {_interval(fin['cells'][base])}.")
        for cell, p in fin["paired"].items():
            w(f"  - {tag(cell)} (`{cell}`){_prov_mark(a, cell)}: {_interval(fin['cells'][cell])}; paired difference **{_pp(p['delta'])}** "
              f"[{_pp(p['ci_low'])}, {_pp(p['ci_high'])}] (B = {p['B']:,}, seed {p['seed']}); b = {p['b_pass_to_fail']}, c = {p['c_fail_to_pass']}; "
              f"exact McNemar {_p(p['mcnemar_exact_p'])}. **Reading: {p['reading']}.**")
        for line in out_lines:
            w(line)
    w(f"- Design: {a['design']['note']}.")
    w("- Truncation rule: " + "; ".join(f"{TASK_SHORT[t]} {'FIRED' if leg['fired'] else 'did not fire'} ("
                                        + ", ".join(f"{tag(c)} {leg['cells'][c]['at_cap']['n_at_cap']}/200" for c in leg["cells"]) + ")"
                                        for t, leg in a["legs"].items()) + f" — acceptance check 4: {a['ladder']['acceptance_check_4']}.")
    for task, blk in (a.get("blocked_legs") or {}).items():
        w(f"- **{TASK_SHORT[task]} not run in this wave** — {blk.get('status')}: {blk.get('reason')}; unblock: {blk.get('unblock')}.")
    w(f"\n{h}# Decoding protocol (acceptance check 4, read literally)\n")
    pr = a["protocol"]
    w(f"Sampling mode `{pr['mode']}`" + (f"; served clamp override `{json.dumps(pr['clamp_override'], sort_keys=True)}` (D13)" if pr.get("clamp_override") else "") + ".")
    w(_md_table(["leg", "executed", "protocol T", "reason", "effective (in-image)", "decode string"],
                [[TASK_SHORT[t], json.dumps(r["executed"], sort_keys=True), r.get("protocol_temperature"), r.get("executed_reason"),
                  json.dumps(r.get("effective"), sort_keys=True) if r.get("effective") else "—", pr["decode_by_task"][t]] for t, r in pr["by_task"].items()]))
    w(f"\n{h}# Roster and generations\n")
    w(_md_table(["entry", "cell", "status", "kind", "generation (launch-time sidecar)", "object sha256 (served)", "fit layers", "reason / unblock", "note"],
                [[e["entry_id"], e["cell_id"] or "—", roster_status_label(e["status"], e.get("reason")), e["kind"],
                  ((a["generations"].get(e["cell_id"]) or {}).get("object_generation") or ("neutral (no object)" if e["status"] == "runnable" else "—")),
                  (str((a["generations"].get(e["cell_id"]) or {}).get("object_sha256") or "—")[:16]),
                  ((a["fit_layers"].get(e["cell_id"]) or {}).get("fit_layers") or "—"),
                  ("" if e["status"] == "runnable" else f"{e.get('reason')} / {e.get('unblock')}"),
                  (str(e.get("note") or "").replace("|", "/") or "")] for e in wv["entries"]]))
    sources = sorted({str(g.get("roster_at_launch_source")) for g in a["generations"].values() if g.get("roster_at_launch_source")})
    if sources:
        w(f"\nGeneration labels are the launch-time sidecars', each RE-DERIVED from the roster at the sidecar's own roster sha ({'; '.join(sources)}).")
    drift = [c for c, g in a["generations"].items() if g.get("roster_object_drift")]
    if drift:
        w(f"\nThe CURRENT roster stages a different object under {drift} than these stores served — the labels above are the launch-time sidecar's.")
    w(f"\n{h}# Identity (asserted before pairing)\n")
    idf = a["identity"]
    w(_md_table(["fact", "value"], [
        ["model", f"`{idf['model_id']}` @ `{idf['model_revision']}`, engine {idf['engine']}, thinking {idf['thinking']}"],
        ["served", f"`{json.dumps(idf['served'], sort_keys=True)}`"],
        ["caps (OFF)", ", ".join(f"{TASK_SHORT[t]} {c}" for t, c in idf["caps"].items())],
        ["provenance", f"suffix `{idf['provenance'].get('suffix')}` (the durable wave identity), run id(s) " + ", ".join(f"`{r}`" for r in run_ids)
                       + f", {idf['provenance'].get('n_attempts', len(run_ids))} attempt(s) all bound to the suffix and swept, tree `{idf['tree_sha']}`"],
        ["routes", ", ".join(f"{tag(c)} → `{v}`" for c, v in idf["routes"].items())],
        ["object sha256", ", ".join(f"{tag(c)} `{str(v)[:16] if v else None}` ({idf['object_sha256_source'].get(c)})" for c, v in idf["object_sha256"].items())],
        ["intervention hash", ", ".join(f"{tag(c)} `{v}`" for c, v in idf["intervention_hash"].items())]]))
    for task, chk in idf["checks"].items():
        w(f"\n{TASK_SHORT[task]}: MUST-EQUAL (projected through `tinycap_ladder.project_identity`, mode `{chk['mode']}`): {', '.join(chk['required_equal'])}; "
          f"MAY-DIFFER fields that differ: {', '.join(sorted(chk['differs'])) or 'none'}; same instrument: **{chk['same_instrument']}**.")
    for task in withdrawn:
        # no per-category / per-family table and no quoted items for a withdrawn leg — the numbers are the record, not a reading
        w(f"\n_{TASK_SHORT[task]} breakdown and quoted items: not rendered — the leg is WITHDRAWN (see the block above); analysis.json keeps them._")
    if "bfcl_by_category" in a["breakdowns"] and BFCL_TASK not in withdrawn:
        bd = a["breakdowns"]["bfcl_by_category"]
        w(f"\n{h}# BFCL-200 by category ({bd.get('label', 'final: rung 1')})\n")
        w(_md_table(["category", "n (pinned / population)", "none"] + [tag(c) for c in a["steered"]] + ["Δ pp (per steered)"],
                    [[r["category"], f"{r['n']} / {r['population_n']}", _interval(r["cells"][base])] + [_interval(r["cells"][c]) for c in a["steered"]]
                     + [" / ".join(_pp(r["deltas"][c]["delta"]) for c in a["steered"])] for r in bd["rows"]]))
        w(f"\n{bd['note']}." + ("" if bd.get("final", True) else " **These rows are rung 1 and NOT final: the leg fired and its ladder is pending.**"))
    if "bcb_by_family" in a["breakdowns"] and BCB_TASK not in withdrawn:
        bd = a["breakdowns"]["bcb_by_family"]
        w(f"\n{h}# BigCodeBench-200 by library family ({bd.get('label', 'final: rung 1')})\n")
        w(_md_table(["family", "n", "none"] + [tag(c) for c in a["steered"]] + ["Δ pp (per steered)"],
                    [[r["family"], r["n"], _interval(r["cells"][base])] + [_interval(r["cells"][c]) for c in a["steered"]]
                     + [" / ".join(_pp(r["deltas"][c]["delta"]) for c in a["steered"])] for r in bd["rows"]]))
        w(f"\n{bd['note']}." + ("" if bd.get("final", True) else " **These rows are rung 1 and NOT final: the leg fired and its ladder is pending.**"))
    # the rung the diagnostics were read at is labelled; a merged leg ALSO shows its final at-cap counts (cap 16,384); degeneration is
    # stated as not measured; a withdrawn leg's row is labelled (instrument facts, not a reading)
    w(f"\n{h}# Diagnostics — {rung1_cap_label(a)}: unparseable, code source, sandbox status, at-cap\n")
    w(f"_{DEGENERATION_NOTE}._\n")
    rows = []
    for task, leg in a["legs"].items():
        for cell, r in leg["cells"].items():
            rows.append([TASK_SHORT[task] + (" (WITHDRAWN)" if task in withdrawn else ""), tag(cell),
                         f"{r['unparseable']['n_unparseable']}/{r['unparseable']['n_answered']} ({_pct(r['unparseable']['rate'])})",
                         (json.dumps(r["code_source"]) if "code_source" in r else "—"), (json.dumps(r["sandbox_status"]) if "sandbox_status" in r else "—"),
                         f"{r['at_cap']['n_at_cap']}/{r['at_cap']['n_stop_reason']} ({_pct(r['at_cap']['at_cap_rate'])}) at cap {r['cap']}",
                         r["at_cap"]["n_model_length"], r["n_unscored"],
                         f"{_num(r['completion_tokens']['median'])} / {_num(r['completion_tokens']['p90'])} / {_num(r['completion_tokens']['max'])}"])
    w(_md_table(["leg", "cell", "unparseable", "code_source", "sandbox_status", "at-cap (stop_reason = max_tokens) — rung 1", "model_length", "unscored",
                 "completion tokens median / p90 / max"], rows))
    merged = merged_legs(a)
    if merged:
        rows = []
        for task, t in merged.items():
            for cell, r in t["cells"].items():
                n_pass = residual_capped_passes(r, task=task)
                rows.append([TASK_SHORT[task] + (" (WITHDRAWN)" if task in withdrawn else ""), tag(cell), f"{r['n_at_cap_rung1']}/200 at cap {r['cap_from']:,}",
                             f"{r['n_at_cap_residual']}/200 at cap {r['cap_to']:,} ({_pct(r['residual_share'])})",
                             ("not recorded (analysis predates D79; the merged rows are not at hand)" if n_pass is None else f"{n_pass} of {r['n_at_cap_residual']}"),
                             r["n_model_length_rung2"]])
        w(f"\nFinal merged at-cap counts (cap 16,384) for the merged leg(s) — the rung-1 counts above are NOT the final diagnostics of these legs:\n")
        w(_md_table(["leg", "cell", "at-cap rung 1", "at-cap FINAL (merged residual)", "residual capped items scored PASS", "model_length at rung 2"], rows))
    w(f"\n{h}# Truncation ladder\n")
    lad = a["ladder"]
    w("Rule (the user, 2026-09-10): a leg fires when more than 1 % of a cell's responses stop at the cap; the capped items (the union across the "
      "wave's cells) are re-run at 16,384 into a new store root, certified PER FITTED CELL at the wave's common rung, and BOTH readings are reported. "
      "Fired: " + ", ".join(f"{TASK_SHORT[t]} {'yes' if f else 'no'}" for t, f in lad["fired"].items()) + ".")
    if lad.get("superseded_certificates"):
        w(f"\nSuperseded certificates under `_ladder/` (recorded re-probes, not read): {', '.join(Path(p).name for p in lad['superseded_certificates'])}.")
    if not lad["applied"] and not lad.get("tasks"):
        w(f"\n**{lad['reason']}.** Acceptance check 4: {lad['acceptance_check_4']}.")
    else:
        if not lad["applied"]:
            w(f"\n**The merged tables below are NOT the final reading** — {lad.get('final_reason')}. Acceptance check 4: {lad['acceptance_check_4']}.")
        acc = lad["accepted"]
        # a native wave's rung 2 names its engine (the lens line is byte-identical)
        engine_note = (f", served by the `{lad['serving_engine']}` engine" if lad.get("serving_engine") == RP.SERVING_ENGINE_NATIVE else "")
        w(f"\nRung 2 served at **L = {acc['max_model_len']:,} × S = {acc['max_num_seqs']}** (`seqs_source` {acc['seqs_source']}, `rung_source` **{acc['rung_source']}**{engine_note}"
          f"{'; OPERATOR OVERRIDE' if lad['operator_rung'] else ''}); certificates: " + (", ".join(f"`{c}` ({v['n_walked']} rung(s) walked"
          + (f", re-probed from {v['reprobed_from']}" if v.get("reprobed_from") else "") + ")" for c, v in lad["certificates"].items()) or "none supplied")
          + (f"; fitted cells WITHOUT a certificate: {lad['fitted_without_certificate']}" if lad.get("fitted_without_certificate") else "") + ".")
        if lad.get("tree_admission"):
            ta = lad["tree_admission"]
            w(f"\nrung 2 produced at `{ta['rung2_tree_sha'][:8]}` — {ta['why']} (rung 1 at `{ta['rung1_tree_sha'][:8]}`; "
              f"admitted by name with `{ta['flag']}` at the merge, D73).")
        rows = []
        for task, t in lad["tasks"].items():
            for cell, r in t["cells"].items():
                rows.append([TASK_SHORT[task], tag(cell), r["n_selected"], _interval(r["rung1"]), _interval(r["merged"]),
                             f"{r['n_at_cap_rung1']} → {r['n_at_cap_residual']} ({_pct(r['residual_share'])} of 200)"])
        w("\n" + _md_table(["leg", "cell", "n_selected", "rung 1 (OFF cap)", "merged (selected at 16,384)", "at-cap rung 1 → residual"], rows))
        # a capped output keeps its benchmark score — said with the merged stores' own count of residual capped items the scorer passed
        # (six of the twelve residual capped Llama-LoRA BFCL outputs), never a hand-typed number
        passes = []
        for task, t in lad["tasks"].items():
            for cell, r in t["cells"].items():
                if r["n_at_cap_residual"]:
                    n_pass = residual_capped_passes(r, task=task)
                    passes.append(f"{'not recorded' if n_pass is None else n_pass} of the {r['n_at_cap_residual']} residual capped {tag(cell)} {TASK_SHORT[task]} outputs"
                                  + (" are scored passes" if n_pass is not None else " — pass count not recorded (analysis predates D79)"))
        w("\nCapped outputs keep their benchmark score (" + ("; ".join(passes) if passes else "no residual capped item in the merged legs")
          + "); readings are qualified by the tested caps (rung 1 at the benchmark's OFF cap, the selected items at 16,384).")
        for task, t in lad["tasks"].items():
            for cell, p2 in t["paired_merged"].items():
                p1 = t["paired_rung1"][cell]
                w(f"\n{TASK_SHORT[task]}{' (WITHDRAWN — record, not a reading)' if task in withdrawn else ''} · {tag(cell)} paired difference: rung 1 (diagnostic) "
                  f"{_pp(p1['delta'])} [{_pp(p1['ci_low'])}, {_pp(p1['ci_high'])}], {_p(p1['mcnemar_exact_p'])} → merged {_pp(p2['delta'])} "
                  f"[{_pp(p2['ci_low'])}, {_pp(p2['ci_high'])}], {_p(p2['mcnemar_exact_p'])}. "
                  + (f"WITHDRAWN (D79): not a reading." if task in withdrawn else f"Reading (merged{'' if lad['applied'] else ', NOT final'}): {reading_rule(p2)}."))
        w(f"\nAcceptance check 4: {lad['acceptance_check_4']}.")
    if lad["budget_decisions"]:
        w("\nBudget decision(s) recorded in the meter's ledger: " + "; ".join(f"{d['at']}: ceiling {d['ceiling_h']} H200-h — {d['note']}"
                                                                          + (f" (fired legs {d['fired_legs']})" if d.get("fired_legs") else "") for d in lad["budget_decisions"]) + ".")
    else:
        w("\nNo budget decision recorded in the meter's ledger (the 25 h spec envelope stood).")
    if a["cost"]["sandboxes"]:
        w(f"\n{h}# Sandbox health\n")
        w(_md_table(["cell", "created", "terminated (observed)", "unaccounted", "terminate_unverified", "cleanup skipped", "failures", "sweeps",
                     "lifetime s median / p90 / max", "$ estimate"],
                    [[tag(c), s["n_created"], s["n_terminated"], s["n_unaccounted"], s["n_terminate_unverified"], s["n_cleanup_skipped_interrupted"], s["n_failures"],
                      len(s["sweeps"]), f"{_num(s['lifetime_s']['median'], 1)} / {_num(s['lifetime_s']['p90'], 1)} / {_num(s['lifetime_s']['max'], 1)}",
                      f"${s['usd_estimate']:.3f}"] for c, s in a["cost"]["sandboxes"].items()]))
        w("\nLifetimes end at the FIRST observed termination (the exit is observed twice on the 200-item legs). Dollars = lifetimes × the compose's cpu / memory "
          "at Modal's published rates — the PRODUCTION run's sandboxes only.")
    w(f"\n{h}# Cost and wall clock\n")
    gpu, wall = a["cost"]["gpu"], a["wall"]
    if gpu["present"]:
        mine = [r for r in gpu["rows"] if r.get("wave") == wv["name"]]
        w(_md_table(["unit", "kind", "attempt", "forecast h", "actual h", "counted h"],
                    [[r["unit"], r["stage_kind"] or "—", r["attempt"], _num(r["forecast"], 2), _num(r["actual"], 2),
                      (f"superseded by {r['superseded_by']['stage']} #{r['superseded_by']['attempt']}" if r["superseded_by"] and r["actual"] is None
                       else "released (never ran)" if r.get("released") else _num(r["counted"], 2))]
                     for r in mine]) if mine else "_no reservation of this wave in the ledger_")
        w(f"\nGPU: this wave **{_num(gpu['wave_h'], 2)} H200-h** (≈ ${_num(gpu['wave_usd_estimate'], 0)}); the lane Σ {gpu['committed_h']:.2f} of the "
          f"{gpu['ceiling_h']:.1f} h ceiling ({gpu['ceiling_source']}; within: {gpu['within_ceiling']}) ≈ ${gpu['usd_estimate']:.0f} at ${gpu['usd_per_h']:.0f}/h; "
          f"this wave's production sandboxes ≈ ${a['cost']['sandbox_usd_estimate']:.2f}; wave GPU + sandboxes ≈ **${_num(a['cost']['wave_usd_estimate'], 0)}** — a PARTIAL "
          "subtotal: the other waves' reservations, smoke and canary sandboxes, the Modal CPU-suite runs (≈ $1 each) and the staging / preflight containers are not "
          f"in it; {a['cost']['basis']}.")
    else:
        w(f"{gpu['note']}; sandboxes ≈ ${a['cost']['sandbox_usd_estimate']:.2f}.")
    w("\n" + _md_table(["cell", "start", "end", "wall h", "Σ task wall_clock_s", "note"],
                       [[tag(c), v["start"] or "—", v["end"] or "—", _num(v["hours"], 2), _num(v["tasks_wall_s"]), v["note"]] for c, v in wall.items()]))
    anc = a.get("anchors") or {}
    if anc.get("rows"):
        w(f"\n{h}# Comparison to full22 anchors (qualitative)\n")
        w(_md_table(["family / arm", "MMLU", "GSM8K"], [[k, v["MMLU"], v["GSM8K"]] for k, v in anc["rows"].items()]) + f"\n\n{anc['note']}.")
    w(f"\n{h}# Figures\n")
    figs = a.get("figures") or {}
    if figs.get("withheld"):
        # no URL at all — not the analysis-time ones, not the renderer's default `<figs url base>/<suffix>/` fallback
        w(f"_Figures withheld: {figs['why']}._")
    else:
        if figs.get("gated"):
            g = figs["gated"]
            wd_named = ", ".join(TASK_SHORT.get(t, t) for t in g.get(WITHDRAWN_LEGS_KEY) or [])
            if g.get("at_root"):
                # the canonical record's own figures, drawn from the gated record at `--analyze` (a launch roster admitted the stores)
                w(f"_Figures drawn from the GATED record under the current roster (sha `{str(g.get('roster_sha256') or 'unknown')[:8]}`"
                  + (f"; withdrawn: {wd_named}" if wd_named else "") + ("; out: " + ", ".join(f"`{c}`" for c in g["out"]) if g.get("out") else "")
                  + ") — no figure draws a withdrawn leg or an out cell as a reading._\n")
            else:
                w(f"_Figures regenerated from the GATED view under the current roster (sha `{str(g.get('roster_sha256') or 'unknown')[:8]}`; out: "
                  + ", ".join(f"`{c}`" for c in g.get("out") or []) + (f"; withdrawn: {wd_named}" if wd_named else "")
                  + ") — the analysis-time figures are provenance and are not linked._\n")
        base_url = figs.get("url_base") or f"{figs_url_base.rstrip('/')}/{a['run_name']}"
        for stem in FIGURE_STEMS:
            if stem in (figs.get("files") or {}) or not figs:
                w(f"- `final_{stem}` — {base_url}/final_{stem}.png")
    if wandb_url:
        w(f"\nW&B: {wandb_url} (project `{WANDB_PROJECT}`, job type `{WANDB_JOB_TYPE}`).")
    if "bcb_failures" in a["quoted"] and BCB_TASK not in withdrawn:
        q = a["quoted"]["bcb_failures"]
        w(f"\n{h}# BigCodeBench failures, quoted raw (10 per cell; {q.get('label', 'final: rung 1')})\n")
        for cell, blk in q["cells"].items():
            w(f"\n{h}## {tag(cell)} — {blk['n_fail']} failures; grid {json.dumps(blk['grid'])}\n")
            for it in blk["items"]:
                w(f"**{it['pinned_id']}** — code_source `{it['code_source']}`, sandbox_status `{it['sandbox_status']}`, stop_reason `{it['stop_reason']}`, "
                  f"{_num(it['completion_tokens'])} completion tokens. Verifier: "
                  + (f"`{it['verifier_explanation']['text'][:300]}`" if it["verifier_explanation"]["text"] else "(the submission was incorrect; no stderr recorded)"))
                w(_fence(it["content"]["text"], truncated=it["content"]["truncated"], chars=it["content"]["chars"]))
    if "bfcl_discordant" in a["quoted"] and BFCL_TASK not in withdrawn:
        q = a["quoted"]["bfcl_discordant"]
        w(f"\n{h}# BFCL discordant items, quoted raw (up to 10 per steered cell; {q.get('label', 'final: rung 1')})\n")
        for cell, d in q["cells"].items():
            w(f"\n{h}## {tag(cell)} vs none — {d['n_discordant']} discordant items\n")
            for it in d["items"]:
                w(f"**{it['item_id']}** ({it['category']}) — {it['direction']}")
                for c, r in it["cells"].items():
                    w(f"\n_{tag(c)}_ (value {r['value']}, stop_reason `{r['stop_reason']}`):")
                    w(_fence(r["content"]["text"], truncated=r["content"]["truncated"], chars=r["content"]["chars"]))
    return "\n".join(out) + "\n"


def render_results_doc(analyses: Sequence[Mapping[str, Any]], *, roster: Optional[Mapping[str, Any]] = None, summary: Optional[Mapping[str, Any]] = None,
                       figs_url_base: str = DEFAULT_FIGS_URL_BASE, wandb_urls: Optional[Mapping[str, str]] = None, skeleton: bool = False) -> str:
    """The lane's results doc: a header, the cross-wave summary, one section per
    analysed wave, and a PENDING section for every roster wave without an
    analysis; ``skeleton`` renders every wave as pending."""
    out: List[str] = []
    w = out.append
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    w("# bcb-bfcl-200-widen — every Gemma instruct-direct object, and LoRA-subtraction vs none on qwen35_4b / nemotron3nano4b, "
      "under each benchmark's own decoding protocol")
    w(f"\n_{'skeleton — every wave pending; ' if skeleton else ''}generated by `scripts/bcb_bfcl_200_analysis.py --write-results` on {now}_")
    w("\nPaired design on identical pinned items, k = 1 under "
      "the benchmarks' protocol decoding (BigCodeBench greedy T 0; BFCL T 0.001 executed literally under the pinned vLLM clamp override, D13), "
      "thinking OFF, served path only. Every contrast is a steered cell vs ITS wave's `none` — no cross-object contrast, hence no batch-8-vs-batch-64 "
      "pairing (D7); the gemma instruct-direct constitution object(s) staged at batch 8 are **provisional** — every reading on them is re-run on the "
      "batch-64 re-fit when staged. PSR is out of the paper by default (the user's ruling of 2026-09-11, D48): every PSR cell is blocked in the roster "
      "— not run, not plotted — and appears below only as an `out` row; the six arms are none, safety SI, constitution SI, control LoRA, constitution "
      "LoRA, difference LoRA. Every number is traceable to a wave's `analysis.json`; every dollar line is a resource-rate estimate, not billing, and a "
      "partial subtotal naming its exclusions.")
    w("\n**Final-reading rule** (one selector for the headline, the summary, W&B, the figures and the paper rows): a leg's final reading is the "
      "MERGED one when the truncation rule fired and the ladder was applied (capped items re-run at 16,384 and merged per item, every fitted cell "
      "certified under the wave record), the RUNG-1 one when the leg did not fire, and NONE while a fired leg's ladder is pending — its rung-1 "
      "numbers appear only under an explicit \"rung 1\" label and its paper rows carry null measurements with `status: pending ladder`. A fitted "
      "store's generation label is re-derived from the roster at its launch-time sidecar's roster sha, never taken on the sidecar's word; the wave's "
      "durable identity is its store suffix (`bcbw-<wave>-<sha7>[-tag]`), every launcher attempt bound to it and covered by a sandbox sweep. "
      "A measured leg the CURRENT roster marks blocked is **WITHDRAWN** (D79): its numbers are kept in `analysis.json` under `final.<leg>.withdrawn` "
      "for the record and appear in no headline, summary, figure or paper value (its paper rows read `status: withdrawn: <reason>`).")
    w(f"\n**Reading rule** (pre-registered; every reading below states it): {READING_RULE_SENTENCE}. Where the interval and the test disagree the row "
      "says so — when every discordant item points the same way (b = 0 or c = 0) the share of bootstrap resamples whose Δ is exactly 0 is "
      "((n − (b + c))/n)^n — 1.76 % for b = 4, c = 0 of 200 — fewer than the 2.5 % a percentile bound needs to reach 0, so the interval can exclude 0 "
      "while the exact test does not reject (sparse discordance; with discordants in both directions cancellation adds zero mass and no such "
      f"formula is claimed); the exact test is the rule. Every per-comparison test and interval is {UNADJUSTED_NOTE}; no adjusted rule is adopted.")
    w(f"\n**Paper rows** (`--paper-rows`, source `{PAPER_SOURCE}`): the notebook's long schema in {len(PAPER_COLUMNS)} columns — the first lane's 45 plus "
      "`object_artifact_sha256` (the 64-hex sha of the served object.pt; null on `none` / prompted cells and on rows that did not run), `thinking` "
      "(`off`) and `held_out` (true on a GRADED row whose roster entry carries a `pooling: {held_out: true, ruled_by, reason}` block — reported "
      "with its numbers, held out of every pooled category mean, never withdrawn; the ruling and the reason lead its `note`); every row's `decode` string begins with the literal words \"benchmark protocol\" (a not-run row carries the roster's DECLARED protocol, "
      "labelled NOT RUN; a model-card row never carries the word); `fit_layers` / `fit_layer_count` per fitted cell from the object's own sidecar; "
      "rows keyed (source, model, lane_kind, view, arm, eval_id) are replaced in place under a file lock, published atomically.")
    if roster is not None and not skeleton:
        analyses = [gate_saved_analysis(a, roster) for a in analyses]          # the CURRENT roster's eligibility; idempotent on a gated view
    by_name = {a["wave"]["name"]: a for a in analyses}
    roster_waves = list((roster or {}).get("waves") or {}) if roster else [name for name, _ in PLANNED_WAVES]
    w("\n## Waves\n")
    w(_md_table(["wave", "family · lane", "status", "cells (runnable)", "legs"],
                [[name, (f"{by_name[name]['wave']['family']} · {by_name[name]['wave']['lane_kind']}" if name in by_name else
                         (f"{wave_spec(roster, name).family} · {wave_spec(roster, name).lane_kind}" if roster else dict(PLANNED_WAVES).get(name, ""))),
                  ("analysed" if name in by_name else "pending"),
                  (", ".join(by_name[name]["cells"]) if name in by_name else "—"),
                  (", ".join(TASK_SHORT[t] + (" (WITHDRAWN)" if t in (by_name[name].get(WITHDRAWN_LEGS_KEY) or {}) else "") for t in by_name[name]["tasks"])
                   if name in by_name else "—")] for name in roster_waves]))
    if summary is not None and not skeleton:
        w("\n## Cross-wave summary\n")
        # the rule and the caveat once, in the header — not per row
        w(f"_Reading rule: {READING_RULE_SENTENCE}. Every test and interval in this table is {UNADJUSTED_NOTE}. A `withdrawn` row is a measured leg the "
          "current roster blocks (instrument failure, D79): no Δ, no p; its numbers are kept in the wave's `analysis.json` for the record._\n")
        w(_md_table(["wave", "family · lane", "cell", "leg", "final reading", "Δ pp [95 %]", "McNemar p", "reading", "check 4"],
                    [[r["wave"], f"{r['family']} · {r['lane_kind']}", r["arm"] + (" (provisional)" if r["provisional"] else "") + (" (held out)" if r.get("held_out") else ""),
                      TASK_SHORT[r["task"]],
                      r.get("source", "—"),
                      (f"{r['delta_pp']:+.1f} [{r['ci_low_pp']:+.1f}, {r['ci_high_pp']:+.1f}]" if r.get("delta_pp") is not None
                       else ("out" if r.get("out") else "withdrawn" if r.get("withdrawn") else "pending")),
                      (_p_value(r["mcnemar_p"]) if r.get("mcnemar_p") is not None else "—"), r["reading"], r["check_4"]]
                     for r in summary["readings"]]))
        held_rows = [(wv["wave"]["name"], c, blk) for wv in summary["waves"] for c, blk in ((wv.get(PUBLICATION_KEY) or {}).get(HELD_OUT_KEY) or {}).items()]
        if held_rows:
            w("\n**Reported, HELD OUT of pooled category means** (the roster's `pooling` declaration: the readings above stand, the figures draw them, "
              "the paper rows carry `held_out: true` so the notebook averages nothing over them): "
              + " ".join(f"`{wn}` / `{c}` — {blk.get('reason')} _({blk.get('ruled_by')})_." for wn, c, blk in held_rows))
        gpu = summary["cost"]["gpu"]
        if gpu["present"]:
            w(f"\nGPU across the lane: **{gpu['committed_h']:.2f} of {gpu['ceiling_h']:.1f} H200-h** ({gpu['ceiling_source']}) — by wave "
              + "; ".join(f"{k} {v:.2f} h" for k, v in gpu["by_wave"].items()) + f"; ≈ ${gpu['usd_estimate']:.0f}; analysed waves' production sandboxes ≈ "
              f"${summary['cost']['sandbox_usd_estimate']:.2f}. {summary['cost']['basis']}.")
        if summary.get("budget_decisions"):
            w("\nBudget decisions: " + "; ".join(f"{d.get('at')}: ceiling {d.get('ceiling_h')} h — {d.get('note')}" for d in summary["budget_decisions"]) + ".")
        if summary.get("unanalysed_entries"):
            w("\nNot run (roster entries of waves without an analysis): " + "; ".join(f"{wn}/{e['entry_id']} ({e['status']}: {e.get('reason')})"
                                                                                    for wn, e in summary["unanalysed_entries"]) + ".")
        figs = summary.get("figures") or {}
        if figs.get("files"):
            w("\nFigures: " + "; ".join(f"`final_{s}` — {figs['url_base']}/final_{s}.png" for s in SUMMARY_FIGURE_STEMS if s in figs["files"]) + ".")
    for name in roster_waves:
        w("")
        if name in by_name and not skeleton:
            w(render_wave_doc(by_name[name], figs_url_base=figs_url_base, wandb_url=(wandb_urls or {}).get(name)))
        else:
            w(f"## Wave `{name}`\n\n{PENDING}\n")
            if roster and name in roster.get("waves", {}):
                spec = wave_spec(roster, name)
                w(_md_table(["entry", "cell", "status", "reason / unblock"],
                            [[e.entry_id, e.cell_id or "—", roster_status_label(e.status, e.reason), ("" if e.runnable else f"{e.reason} / {e.unblock}")]
                             for e in spec.entries]))
            w("\n" + "\n".join(f"- `final_{s}` — {PENDING}" for s in FIGURE_STEMS))
    w("\n## Acceptance checks (spec §Acceptance)\n")
    if skeleton or not analyses:
        w(PENDING)
    else:
        for a in analyses:
            w(f"- Wave `{a['wave']['name']}` — check 4 (200/200 items with a verdict per cell × leg; BCB T 0 / BFCL T 0.001 recorded; at-cap share; the rule "
              f"applied; identity asserted): every leg carries exactly 200 items per cell (asserted); protocol record checked literally (mode `{a['protocol']['mode']}`); "
              f"ladder: **{a['ladder']['acceptance_check_4']}**.")
        w("- Check 5 (analysis + figures + results doc + paper rows + artifact; cost vs the envelope): this doc, the figures above, the cost sections; "
          "the review artifact is built by `/review`.")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--roster", default=str(REPO / DEFAULT_ROSTER_REL), help="the lane roster yaml")
    p.add_argument("--wave", default=None, help="the roster wave being pulled / analysed")
    p.add_argument("--pull", metavar="SUFFIX", help="fetch every runnable cell's store root, the ladder dir and the object sidecars for SUFFIX")
    p.add_argument("--pull-root", default=DEFAULT_PULL_ROOT)
    p.add_argument("--volume", default=DEFAULT_VOLUME)
    p.add_argument("--modal-python", default=DEFAULT_MODAL_PYTHON, help="the venv python that has `modal` (never imported here)")
    p.add_argument("--mode", default=DEFAULT_MODE)
    p.add_argument("--serving-engine", default=None, choices=list(SERVING_ENGINES),
                   help=f"with --pull: the engine the run was generated on (default {RP.SERVING_ENGINE_LENS}) — a {RP.SERVING_ENGINE_NATIVE} run's "
                        f"store roots and ladder dir live under the native namespace ({C.OUT_SUBDIR_NATIVE}/) of the generation root")
    p.add_argument("--analyze", metavar="RUNG1_DIR", nargs="?", const="", default=None, help="analyze a pulled rung-1 dir (bare --analyze after --pull uses the pulled dir)")
    p.add_argument("--launch-roster", metavar="ROSTER_YAML", default=None,
                   help="with --analyze / --pull: the roster the stores were LAUNCHED under — used ONLY to admit the stores (a store is never "
                        "analysed under a label the roster did not launch); --roster (the current one) gates what is published, so a leg the "
                        "current roster blocks is WITHDRAWN in the saved analysis.json, the root figures and W&B (D86)")
    p.add_argument("--repo-root", default=str(REPO), help="where the pins and the library-family map live")
    p.add_argument("--ladder-merge", metavar="DIR", default=None, help="`tinycap_ladder.py merge`'s output dir (the ONLY source of ladder readings)")
    p.add_argument("--selection", default=None, help="ladder_selection.json")
    p.add_argument("--certificate", action="append", default=None, help="a per-cell rung_probe.<cell>.json (repeat; default: every one under _ladder/)")
    p.add_argument("--rung2", metavar="DIR", default=None, help="the pulled rung-2 dir, for its sandboxes / wall / certificates")
    p.add_argument("--ledger", default=M.LEDGER_DEFAULT, help="the meter's ledger")
    p.add_argument("--figs-root", default=DEFAULT_FIGS_ROOT, help="figures land at <figs-root>/<suffix>/ (summary: <figs-root>/summary/)")
    p.add_argument("--figs-url-base", default=None,
                   help="the URL the figures are linked under (default: $SISTEER_FIGS_URL_BASE/<lane>, else the absolute path of --figs-root)")
    p.add_argument("--no-figs", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--wandb-mode", default=None, choices=("online", "offline", "disabled"))
    p.add_argument("--wandb-project", default=WANDB_PROJECT)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--summary", metavar="ANALYSIS_JSON", nargs="*", default=None, help="the cross-wave summary over these analysis.json files")
    p.add_argument("--summary-out", default=None, help="where summary.json goes (default: <figs-root>/summary/summary.json)")
    p.add_argument("--write-results", metavar="PATH", default=None,
                   help="render the results doc at PATH from --analyze / --summary / --analysis")
    p.add_argument("--analysis", metavar="ANALYSIS_JSON", action="append", default=None, help="analysis.json file(s) to render / export without --analyze")
    p.add_argument("--skeleton", action="store_true", help="with --write-results: render every wave as pending")
    p.add_argument("--paper-rows", metavar="PATH", nargs="?", const=DEFAULT_PAPER_ROWS, default=None,
                   help=f"append this wave's rows to the paper parquet (default {DEFAULT_PAPER_ROWS})")
    p.add_argument("--paper-rows-replace", action="store_true", help="with --paper-rows: overwrite the file instead of appending")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.figs_url_base is None:
        args.figs_url_base = figs_url_base_for(args.figs_root)
    if not (args.pull or args.analyze is not None or args.write_results or args.paper_rows or args.summary is not None):
        build_parser().print_help()
        return 2
    roster_path = Path(args.roster).expanduser()
    roster: Optional[Dict[str, Any]] = load_roster(roster_path) if roster_path.is_file() else None
    run_dir: Optional[Path] = None
    analysis: Optional[Dict[str, Any]] = None
    analyses: List[Dict[str, Any]] = []
    if args.pull or args.analyze is not None:
        if roster is None:
            raise AnalysisError(f"--pull / --analyze need the roster ({roster_path} does not exist)")
        if not args.wave:
            raise AnalysisError("--pull / --analyze need --wave <roster wave>")
        if args.launch_roster and not Path(args.launch_roster).expanduser().is_file():
            raise AnalysisError(f"--launch-roster {args.launch_roster} does not exist")
        # `--pull` fetches what the LAUNCH roster ran (its runnable cells / legs); the current roster gates at analysis
        wave = wave_spec(load_roster(Path(args.launch_roster).expanduser()) if args.launch_roster else roster, args.wave)
    if args.pull:
        validate_roster(roster, roster_path)            # the roster passes its own rules before anything is fetched under its names
        run_dir = pull_suffix(args.pull, wave, dest_root=args.pull_root, volume=args.volume, modal_python=args.modal_python, mode=args.mode,
                              serving_engine=args.serving_engine)
    if args.analyze is not None:
        if args.analyze:
            run_dir = Path(args.analyze).expanduser()
        if run_dir is None:
            raise AnalysisError("--analyze needs a rung-1 dir (or a --pull in the same invocation)")
        analysis = analyze(run_dir, roster_path=roster_path, wave_name=args.wave, repo_root=args.repo_root, ladder_merge=args.ladder_merge,
                           selection=args.selection, certificates=args.certificate or (), ledger=args.ledger, rung2_dir=args.rung2,
                           launch_roster_path=args.launch_roster)
        figs_dir = Path(args.figs_root).expanduser() / str(analysis["run_name"])
        if not args.no_figs:
            # the ROOT figures are drawn from the GATED record — a withdrawn leg is not drawn (placeholder / title-labelled), an out cell
            # not drawn — and say so; the launch-time numbers live under `final.<leg>.withdrawn`, not in any figure
            files = make_figures(analysis, figs_dir)
            analysis["figures"] = {"dir": str(figs_dir), "files": files, "url_base": f"{args.figs_url_base.rstrip('/')}/{analysis['run_name']}"}
            pub = analysis.get(PUBLICATION_KEY) or {}
            if pub.get("out") or analysis.get(WITHDRAWN_LEGS_KEY):
                analysis["figures"]["gated"] = {"roster_sha256": pub.get("roster_sha256"), "out": list(pub.get("out") or {}),
                                                WITHDRAWN_LEGS_KEY: list(analysis.get(WITHDRAWN_LEGS_KEY) or {}), "at_root": True}
            print(f"[{LANE}] {len(files)} figures → {figs_dir}"
                  + (f" (drawn from the gated record; withdrawn: {', '.join(TASK_SHORT[t] for t in analysis[WITHDRAWN_LEGS_KEY])})"
                     if analysis.get(WITHDRAWN_LEGS_KEY) else ""))
        mode = "disabled" if args.no_wandb else args.wandb_mode
        analysis["wandb"] = log_wandb(analysis, (analysis.get("figures") or {}).get("files") or {}, figs_dir, mode=mode, project=args.wandb_project,
                                      name=args.wandb_name)
        if analysis["wandb"]:
            print(f"[{LANE}] W&B: {analysis['wandb']}")
        out = write_analysis(analysis, run_dir)
        for task, leg in analysis["legs"].items():
            fin = analysis["final"][task]
            if not fin["final"]:
                print(f"[{LANE}] {args.wave} {TASK_SHORT[task]}: NO final reading — {fin['why']} (rung 1 is diagnostic only)")
                continue
            for cell, p in fin["paired"].items():
                print(f"[{LANE}] {args.wave} {TASK_SHORT[task]} {short_cell(cell, analysis)} (final: {fin['source']}): none "
                      f"{_interval(fin['cells'][analysis['base']])}; steered {_interval(fin['cells'][cell])}; Δ {_pp(p['delta'])} "
                      f"[{_pp(p['ci_low'])}, {_pp(p['ci_high'])}], McNemar {_p(p['mcnemar_exact_p'])}; {p['reading']}; fired {leg['fired']}")
        print(f"[{LANE}] ladder: {analysis['ladder']['acceptance_check_4']} → {out}")
        analyses.append(analysis)
    if args.analysis and roster is None:
        raise AnalysisError(f"--analysis needs the roster ({roster_path} does not exist): a saved analysis is published only through the CURRENT "
                            "roster's eligibility gate")
    for src in (args.analysis or []):
        view = gate_saved_analysis(read_json(Path(src).expanduser()), roster, roster_path=roster_path)    # the file is read, never rewritten
        if not args.no_figs and regenerate_gated_figures(view, figs_root=args.figs_root, figs_url_base=args.figs_url_base):
            print(f"[{LANE}] {view['wave']['name']}: figures regenerated from the gated view → {view['figures']['dir']} "
                  f"(out: {', '.join(view['figures']['gated']['out'])})")
        analyses.append(view)
    summary: Optional[Dict[str, Any]] = None
    if args.summary is not None:
        paths = list(args.summary) or [a["rung1_dir"] + "/" + ANALYSIS_FILENAME for a in analyses]
        summary = summarize(paths, roster_path=(roster_path if roster else None), ledger=args.ledger)
        figs_dir = Path(args.figs_root).expanduser() / "summary"
        if not args.no_figs:
            for view in summary["waves"]:                    # a gated wave's per-wave figures: regenerated from the view, never the analysis-time ones
                if regenerate_gated_figures(view, figs_root=args.figs_root, figs_url_base=args.figs_url_base):
                    print(f"[{LANE}] {view['wave']['name']}: figures regenerated from the gated view → {view['figures']['dir']} "
                          f"(out: {', '.join(view['figures']['gated']['out'])})")
            files = make_figures(summary, figs_dir, figures=SUMMARY_FIGURES)
            summary["figures"] = {"dir": str(figs_dir), "files": files, "url_base": f"{args.figs_url_base.rstrip('/')}/summary"}
            print(f"[{LANE}] {len(files)} summary figures → {figs_dir}")
        mode = "disabled" if args.no_wandb else args.wandb_mode
        summary["wandb"] = log_wandb({**summary, "run_name": "summary", "identity": {}, "generator": summary["generator"]},
                                     (summary.get("figures") or {}).get("files") or {}, figs_dir, mode=mode, project=args.wandb_project,
                                     name=(args.wandb_name or f"{LANE}-summary"), scalars=summary_scalars(summary), image_prefix="bcb_bfcl_200_widen/summary")
        out_path = Path(args.summary_out).expanduser() if args.summary_out else figs_dir / SUMMARY_FILENAME
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=1, sort_keys=True, allow_nan=False, default=str) + "\n", encoding="utf-8")
        print(f"[{LANE}] summary over {len(summary['waves'])} wave(s) → {out_path}")
        analyses = list(summary["waves"])
    if args.write_results:
        if args.skeleton:
            doc = render_results_doc([], roster=roster, figs_url_base=args.figs_url_base, skeleton=True)
        else:
            if not analyses:
                raise AnalysisError("--write-results needs --analyze, --summary, --analysis <analysis.json>, or --skeleton")
            doc = render_results_doc(analyses, roster=roster, summary=summary, figs_url_base=args.figs_url_base,
                                     wandb_urls={a["wave"]["name"]: (a.get("wandb") or {}).get("url") for a in analyses if a.get("wandb")})
        out_path = Path(args.write_results).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(doc, encoding="utf-8")
        print(f"[{LANE}] results doc → {out_path} ({'skeleton' if args.skeleton else f'{len(analyses)} wave(s) rendered'})")
    if args.paper_rows:
        from sisteer import site as _site
        args.paper_rows = str(_site.expand_path(args.paper_rows, key="--paper-rows"))   # the default sits under $WORK
        if not analyses:
            raise AnalysisError("--paper-rows needs --analyze, --summary or --analysis <analysis.json>")
        # every wave assembled, ONE locked read–merge–write (with --paper-rows-replace the file is replaced once by all of them)
        for a, rec in zip(analyses, write_paper_rows_all(analyses, args.paper_rows, replace=args.paper_rows_replace)):
            print(f"[{LANE}] paper rows ({a['wave']['name']}) → {rec['path']}: {rec['n_added']} rows added, {rec['n_replaced']} replaced, "
                  f"{rec['n_kept']} kept ({rec['n_rows_written']} total; source {PAPER_SOURCE}; eval ids {sorted(set(PAPER_EVAL_ID.values()))})")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (AnalysisError, L.LadderError, M.MeterError, P.PairingError) as e:
        print(f"[{LANE}] REFUSED: {e}", file=sys.stderr)
        sys.exit(3)
