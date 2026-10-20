#!/usr/bin/env python3
"""The paper's ALL-MODEL APPENDIX GSM8K rows: the paired-write difference ReLU
(`constv2-mix-full__psr_ce_all_minus_ctl`, kind psr_paired_write — the composed arm the Qwen paired views
s<s>p measured) served on gemma4e4b `it`, llama8b `it` and nemotron3nano4b `main`, plus nemotron3nano4b × the roster's constitution PSR
object (`constv2-mix-full__psr_ce_all`, whose lens-era default-matrix row never completed) — FOUR rows, one RECORD document beside the
matrices, read by the Slurm launcher BY NAME (`paired_allmodels`, g8k_cells.MATRICES).

WHY A RECORD. The composed objects are ANOTHER LANE's (the relu_paired_mm bake, staged read-only into exp/objects/sdfbase/relu_paired/
<model>/ on the primary cluster with a one-entry bank view per model), so the rows are EXTERNAL rows — and the matrix tooling admits an
external row only for the rung's steer cells on their declared arm view (g8k_cells.EXTERNAL_ROW_VIEW_BY_CELL). This record's rows carry
`role: paired_allmodels_value`, which `external_row_problem` dispatches on: the composed cell on the row's DECLARED PANEL VIEW (`panel_view`
== `view`, `it` or `main`), the default Inspect seed rule (no seed policy — the appendix column's instrument, one trial per model), no
companion (a single-object standup: the bank view holds exactly the one cell). The nemotron constitution row is a roster-served row
(no object source; the frozen pin file's pin) re-keyed into this wave with its own overlay pair.

THE DESIGN: the composed objects are fitted on the BASE checkpoint and served on the instruct one for gemma and llama (as for
Qwen) — that transfer IS the design, not a defect; nemotron is fitted and served on the same BF16 checkpoint only because no base
Nemotron exists. Nothing here is provisional on a re-fit. The read-out carries a STRUCTURAL OFF-TASK CHECK beside accuracy (an
open finding: under steering llama / gemma may stop tracking the conversation — first-person restatement, `## Step 1:` templates
to the cap); degeneration is scored separately, as everywhere in the lane.

  python scripts/sdfbase/gsm8k/g8k_paired_allmodels.py            # write configs/sdfbase/gsm8k/paired_allmodels_rows.json
  python scripts/sdfbase/gsm8k/g8k_paired_allmodels.py --check    # exit 1 on drift

Read by: gsm8k_thinking.sbatch / ladder_plan.sbatch (`... <sha> <rung> paired_allmodels`), g8k_cells.py plan --matrix paired_allmodels,
make_gsm8k_overlays.py --matrix paired_allmodels (the four own-overlay pairs), tests/test_gsm8k_paired_allmodels.py, and the registry
reader (source string `lane_source` below — the coexist rule r2v4-coexist-gsm8k-paired-allmodels-beside-in-frame-gsm8k-rows binds
on it).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import g8k_cells as M  # noqa: E402

REPO = HERE.parents[2]
SCHEMA = "g8k_paired_allmodels_rows/1"
OUT = REPO / "configs" / "sdfbase" / "gsm8k" / "paired_allmodels_rows.json"
DEFAULT_MATRIX = M.MATRICES[M.MATRIX_DEFAULT_NAME]
LLAMA_MATRIX = M.MATRICES[M.MATRIX_LLAMA_NAME]
ROLE = M.ROLE_PAIRED_ALLMODELS
WAVE = M.MATRIX_PAIRED_ALLMODELS_NAME                  # the wave IS the document: `paired_allmodels`
WAVE_ROOT_REL = "exp/gsm8k_paired200_paper_objects_paired_allmodels"
#: the registry reader's source string: the coexist rule matches on it — spelled here, nowhere else
LANE_SOURCE = "gsm8k_paired_allmodels"
CELL_COMPOSED = M.CELL_PAIRED_DIFFERENCE
CELL_CONSTITUTION = "constv2-mix-full__psr_ce_all"
#: the fitted rows' --time class per lane (g8k_cells' class tables — the plan's `-J` tag is looked up per model scope, so a Llama row must
#: carry a Llama-lane class): the full22 families' TIME_FITTED (200 items thinking-ON ≈ 5–12 min native incl. standup), the Llama lane's
#: TIME_FITTED_LLAMA (no thinking channel, ~200–300-token completions)
TIME_LIMIT_BY_LANE = {"llama8b": M.TIME_FITTED_LLAMA}
LANE_ROOT = "exp/objects/sdfbase/relu_paired"            # the read-only staged bank on the primary cluster (per model: cells/, cells_view/)

#: THE STAGED OBJECTS (the relu_paired_mm compose-v outputs, staged AS-IS — dose 1.0 IS the composed object — from $STAGE on the grading
#: host into the primary cluster's bank 2026-09-22 04:24Z, sha256 re-verified after the copy; the VIEW_MANIFEST.json of each one-entry
#: view names the same sha). `fit` is the checkpoint the parents were fitted on, `served` the one the row serves on (the design).
#: All three were composed by scripts/sdfbase/psr_residual_distill.py compose-v at 0b68b71a80fa2891 (derived_meta.json builder_commit).
COMPOSE_BUILDER_COMMIT = "0b68b71a80fa28914f077b387b7f076f50eaa534"
OBJECTS = {
    "gemma4e4b": {
        "staged_id": "paired_eb8__constv2-mix-full__psr_ce_all_minus_ctl",
        "object_sha256": "87863240eb52646b247315ce07b53ad601b7aff5a0207e528545f2c0564bba40",
        "lane": "paired_eb8", "parents_effective_batch": 8,
        "parents": {"pos": ("constv2-mix-full__psr_ce_all", "a75233d855538e52a692491ea0e92a0600197fcdb61c83c46d19c4d03f2ff83d"),
                    "neg": ("constv2ctl2-full__psr_ce_all", "a8a79d731773b730a394c32d11b67175e2034e5ed015c9beb5f93a51bd83c5fd")},
        "fit": {"model_key": "gemma4e4b_base", "model_id": "google/gemma-4-E4B", "revision": "411aa17b749aa952df1359d2dcea73917a544d9a"},
        "served": {"model_id": "google/gemma-4-E4B-it", "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2"},
    },
    "llama8b": {
        "staged_id": "paired_eb64__constv2-mix-full__psr_ce_all_minus_ctl",
        "object_sha256": "5066333f8c10d57cd9992a33d8c26c7350a3bff6c1d401596f3a8fadb856d4e3",
        "lane": "paired_eb64", "parents_effective_batch": 64,
        "parents": {"pos": ("constv2-mix-full__psr_ce_all", "4df62c74415737e2e910d4b3c7a55d169012b6cff538934d4a053ec7318db799"),
                    "neg": ("constv2ctl2-full__psr_ce_all", "c30ec630a5e32eff916eda6142108e179e0374d307c8f9373641ed600f546da5")},
        "fit": {"model_key": "llama8b_base", "model_id": "NousResearch/Meta-Llama-3.1-8B", "revision": "1f47e50cdbe801ad8a5174156ec3a0655108fb9f"},
        "served": {"model_id": "NousResearch/Meta-Llama-3.1-8B-Instruct", "revision": "d10aef7999a2b5ba950ab3974312feeedbfe0b77"},
    },
    "nemotron3nano4b": {
        "staged_id": "paired_eb8__constv2-mix-full__psr_ce_all_minus_ctl",
        "object_sha256": "ad3e19228ed41ea861d4c4d6af3c322eb2a56e3c2dc1c3e197721fa89e96fd9f",
        "lane": "paired_eb8", "parents_effective_batch": 8,
        "parents": {"pos": ("constv2-mix-full__psr_ce_all", "81321803c479f3afc9e4738955e881d277a530f2bd5cfcbdcc9ad60e8a44f42f"),
                    "neg": ("constv2ctl2-full__psr_ce_all", "326faf845b2681a37fe6adb7c38d063592dcb7953b47b3a8ca6028f12ad2c704")},
        "fit": {"model_key": "nemotron3nano4b_doc", "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "revision": "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"},
        "served": {"model_id": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", "revision": "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"},
    },
}
#: the nemotron × constitution PSR row's filed refusal: the reason the launcher logs and the analysis reads (`expected_refusal_reason`)
CONSTITUTION_REFUSAL_REASON = ("THINKING_UNSERVABLE (predicate A): under the constitution PSR steering nemotron3nano4b ends its turn inside the "
                               "prompt-opened think block on 3/3 render-probe draws (class unclosed_block; evidence.json render_probe.on_attempts), so "
                               "a thinking-ON GSM8K would score no item; the driver refuses. Reproduced on the lens engine 2026-09-11 (jobs "
                               "483252_27 / 483711_27, the published matrix's row 27) and the native engine 2026-09-22 (596419_1, tree a85369ce0). "
                               "Filed as a refusal state; no thinking-ON GSM8K number exists for this pair")
#: (model_key, panel view, anchor matrix) — the row order is the launch order: nemotron's two cells first
PANEL = (("nemotron3nano4b", M.VIEW_MAIN), ("gemma4e4b", M.VIEW_IT), ("llama8b", M.VIEW_IT))
#: the object's one-entry bank view per model (built 2026-09-22; `paired_d1`: the dose-1.0 composed point, seed slot 1 — the Qwen views' spelling)
BANK_VIEW = "paired_d1_{key}_s1"

#: contracts' spellings, mirrored (g8k_cells is stdlib-only): the composed arm's OOS tuple pair (G8K_NATIVE_PAIRED_MODELS: qwen35_4b,
#: gemma4e4b, nemotron3nano4b) and the appendix constitution row's (G8K_NATIVE_APPENDIX_MODELS: nemotron3nano4b); tests pin them equal
def native_paired_tuple_name(key: str, cap: int) -> str:
    return f"gsm8k_paired200_{key}_native_paired_{cap}"


def native_tuple_name(key: str, cap: int) -> str:
    return f"gsm8k_paired200_{key}_native_{cap}"


#: the structural off-task check the read-out carries beside accuracy (an open finding; degeneration is scored separately).
#: Computed post hoc over the stores' completions by the wave root's g8k_three_numbers tool; the record NAMES the
#: rule so the share cannot be read without its definition.
OFF_TASK_CHECK = {
    "what": "share of items whose completion restates the problem in the first person or continues a `## Step 1:`-style template without "
            "reaching an answer — the conversation-tracking failure ReLU saw under steering on llama / gemma (inversion_probe.md §5 A1)",
    "reported": "beside correct/scored and correct/200 as a third number, per row; never folded into accuracy, never a null",
    "rule": "first_person_restatement: the completion's first 400 characters contain a first-person restatement of the question stem "
            "(`I have`, `I need to`, `My`) AND no `####`/boxed answer; template_continuation: >= 3 `## Step N:` headers with no final answer "
            "— either marks the item off_task; loops (the ladder classifier's non_terminating_ids) are counted separately and first",
}


def anchor_row(key: str, view: str) -> dict:
    """The model's in-frame unsteered row (`none` on the panel view) — the row whose sampling / family / parser / geometry facts the
    appendix rows inherit, and the run the panel POOLS as the contrast (the reader's `pooled_in_frame_anchors`)."""
    doc = M.load_matrix(LLAMA_MATRIX if key == "llama8b" else DEFAULT_MATRIX)
    rows = [r for r in doc["rows"] if (r["model_key"], r["view"], r["cell"]) == (key, view, M.CELL_NONE)]
    if len(rows) != 1:
        raise SystemExit(f"expected exactly one {key}/{view}/none anchor row, found {len(rows)}")
    if rows[0].get("status") != M.STATUS_LAUNCH:
        raise SystemExit(f"the {key}/{view}/none anchor is not a launch row ({rows[0].get('status')!r})")
    return rows[0]


def constitution_source_row() -> dict:
    """The default matrix's nemotron3nano4b/main constitution row — the roster-pinned PSR object this record re-keys (its pin is the
    frozen pin file's, the same sha the composed nemotron object's positive parent records)."""
    doc = M.load_matrix(DEFAULT_MATRIX)
    rows = [r for r in doc["rows"] if (r["model_key"], r["view"], r["cell"]) == ("nemotron3nano4b", M.VIEW_MAIN, CELL_CONSTITUTION)]
    if len(rows) != 1 or rows[0].get("status") != M.STATUS_LAUNCH or rows[0].get("object_source") is not None:
        raise SystemExit(f"expected one launched roster-served nemotron3nano4b/main/{CELL_CONSTITUTION} row in the default matrix")
    if rows[0]["object_sha256"] != OBJECTS["nemotron3nano4b"]["parents"]["pos"][1]:
        raise SystemExit("the roster's nemotron constitution pin != the composed nemotron object's positive parent")
    return rows[0]


def _common(row: dict, anchor: dict, key: str, view: str, cell: str, index: int) -> None:
    """The fields every row of this record takes from its anchor + the wave (in the anchor's field order; new fields appended)."""
    tag = M.tag_for(key, view)
    row.update({
        "index": index, "job_id": f"g8k-{key}-{view}-{M.cell_short(cell)}", "model_key": key, "view": view, "tag": tag,
        "cell": cell, "cell_slug": M.cell_slug(cell), "harness_cell": _harness(cell), "standup_cell": cell, "standup_harness_cell": _harness(cell),
        "fitted": True, "route_kind": "lens", "served_positions_expected": "all", "declared": False,
        "thinking_mode": M.THINKING_MODE_BY_FAMILY[anchor["family"]], "trigger_threshold": M.TRIGGER_THRESHOLD,
        "sampling_mode": anchor["sampling_mode"], "sampling": copy.deepcopy(anchor["sampling"]),
        "expected_stamps": ["cell-stable", "always", anchor["expected_stamps"][2]],
        "out_root": f"{M.out_root_prefix_for(WAVE)}/{tag}/{M.cell_slug(cell)}/{anchor['sampling_mode']}",
        "rung_out_root": f"{M.out_root_prefix_for(WAVE)}/{tag}/{M.cell_slug(cell)}/{anchor['sampling_mode']}__rung{M.RUNG_CAP}",
        "time_limit": TIME_LIMIT_BY_LANE.get(key, M.TIME_FITTED), "family": anchor["family"], "split_mode_candidate": anchor["split_mode_candidate"],
        "reasoning_parser": anchor["reasoning_parser"], "gpu_mem_util": anchor["gpu_mem_util"], "budget_class": "none",
        # the objects are staged on the PRIMARY cluster only (bank_view_rel.overflow null), and nemotron's hybrid ceilings are measured there
        "cluster_constraint": "primary",
        "status": M.STATUS_LAUNCH, "held_reason": None, "expected_refusal": False, "expected_refusal_reason": None,
        "wave": WAVE, "extension": True, "own_overlay": True,
    })


def _harness(cell: str) -> str:
    """The harness spelling of a constitution-family cell (registry_adapter.harness_cell_id mirrored for the two cells this record names)."""
    si, _, arm = cell.partition("__")
    return f"si_{si.replace('-', '_')}__{arm}"


def composed_row(key: str, view: str, index: int) -> dict:
    anchor = anchor_row(key, view)
    obj = OBJECTS[key]
    row: dict = {}
    _common(row, anchor, key, view, CELL_COMPOSED, index)
    thinking_on = row["thinking_mode"] == M.THINKING_ON
    row.update({
        "staged_id": obj["staged_id"], "object_sha256": obj["object_sha256"],
        # llama's no-switch arm is IN SUPPORT (not_available carries no tuple: the pin rides PIN as identity); gemma / nemotron thinking-ON
        # is out of support and matches the composed arm's own tuple pair by content (contracts G8K_NATIVE_PAIRED_MODELS)
        "tuple_name": native_paired_tuple_name(key, M.CAP) if thinking_on else None,
        "tuple_names": {str(c): native_paired_tuple_name(key, c) for c in M.RUNGS} if thinking_on else None,
        "object_source": M.OBJECT_SOURCE_EXTERNAL,
        "lane_root": LANE_ROOT,
        "bank_view": BANK_VIEW.format(key=key),
        "bank_view_rel": {"primary": f"{LANE_ROOT}/{key}/cells_view/{BANK_VIEW.format(key=key)}", "overflow": None},
        "bank_cell_dir": CELL_COMPOSED,
        "selection_label": (f"dose 1.0 = the composed object itself (no scaling; ReLU's relu_paired_mm compose-v of the two {obj['lane']} parents, "
                            f"effective batch {obj['parents_effective_batch']}): the appendix serves the paired-write difference at the dose the "
                            f"Qwen paired views measured; no selection rule, no ladder"),
        "selection_outcome": None, "grid_limited": False,
        "role": ROLE, "panel_view": view,
        "object_kind": "psr_paired_write",
        "object_parents": {"pos": {"cell": obj["parents"]["pos"][0], "object_sha256": obj["parents"]["pos"][1]},
                           "neg": {"cell": obj["parents"]["neg"][0], "object_sha256": obj["parents"]["neg"][1]},
                           "effective_batch": obj["parents_effective_batch"], "lane": obj["lane"]},
        "fit_checkpoint": dict(obj["fit"]), "served_checkpoint": dict(obj["served"]), "compose_builder_commit": COMPOSE_BUILDER_COMMIT,
        "anchor_out_root": anchor["out_root"],           # the in-frame unsteered run the panel pools as this row's contrast (informational)
    })
    return row


def constitution_row(index: int) -> dict:
    src = constitution_source_row()
    anchor = anchor_row("nemotron3nano4b", M.VIEW_MAIN)
    row = copy.deepcopy(src)
    _common(row, anchor, "nemotron3nano4b", M.VIEW_MAIN, CELL_CONSTITUTION, index)
    row.update({
        "tuple_name": native_tuple_name("nemotron3nano4b", M.CAP),
        "tuple_names": {str(c): native_tuple_name("nemotron3nano4b", c) for c in M.RUNGS},
        "role": ROLE, "panel_view": M.VIEW_MAIN,
        "object_kind": "psr",
        # FILED REFUSAL: the driver's predicate (A) refuses this pair — under the constitution PSR steering nemotron ends its
        # turn INSIDE the prompt-opened think block (3/3 probe draws, class unclosed_block), so a thinking-ON GSM8K would leave every item
        # unscored; REC_THINKING_UNSERVABLE, DRIVER_FAILED. Reproduced across engines and trees: lens 2026-09-11 (jobs 483252_27 / 483711_27,
        # the published matrix's row 27) and native 2026-09-22 (596419_1 at a85369ce0). The row stays `launch` (a relaunch reproduces the
        # refusal, which IS the record) and is counted as an expected refusal: no contrast is computable from it.
        "expected_refusal": True,
        "expected_refusal_reason": CONSTITUTION_REFUSAL_REASON,
        "source_row": {"matrix": str(DEFAULT_MATRIX.relative_to(REPO)), "index": src["index"], "job_id": src["job_id"], "out_root": src["out_root"],
                       "note": "the lens-era row of the published matrix never completed; this row serves the same roster-pinned object natively "
                               "under the appendix wave's own root and the appendix tuple (contracts G8K_NATIVE_APPENDIX_MODELS)"},
        "anchor_out_root": anchor["out_root"],
    })
    return row


def derive_rows() -> list[dict]:
    rows = [composed_row("nemotron3nano4b", M.VIEW_MAIN, 0), constitution_row(1), composed_row("gemma4e4b", M.VIEW_IT, 2), composed_row("llama8b", M.VIEW_IT, 3)]
    return rows


def validate(rows: list[dict], doc_stub: dict) -> None:
    for r in rows:
        if r.get("object_source") == M.OBJECT_SOURCE_EXTERNAL:
            problem = M.external_row_problem(r)
            if problem:
                raise SystemExit(f"{r['job_id']}: {problem}")
        for fn in (M.inspect_seed_row_problem, M.companion_row_problem, M.geometry_class_row_problem):
            problem = fn(r, doc_stub)
            if problem:
                raise SystemExit(f"{r['job_id']}: {problem}")
        for rung in M.RUNGS:
            ev, L = M.geometry_for_cell(r["cell"], rung, model_key=r["model_key"])      # the prompt-class rule: short cells
            if ev != M.GEOMETRY_EVAL_BY_RUNG[rung] or L != M.GEOMETRY_L_BY_RUNG[rung]:
                raise SystemExit(f"{r['job_id']}: geometry {ev} x {L} at {rung} is not the short class's")
        if M.row_companions(r) != [] or any(k in r for k in M.STANDUP_COMPANION_FIELDS):
            raise SystemExit(f"{r['job_id']}: a single-object standup registers no companion")
        if r["thinking_mode"] != M.THINKING_MODE_BY_FAMILY[r["family"]] or r["sampling_mode"] != M.SAMPLING_MODE_BY_MODEL[r["model_key"]]:
            raise SystemExit(f"{r['job_id']}: arm / sampling mode are the model's")
        if (r["tuple_names"] is None) != (r["thinking_mode"] == M.THINKING_NOT_AVAILABLE):
            raise SystemExit(f"{r['job_id']}: a thinking-ON fitted row carries its OOS tuple pair; a not_available row none")
        M.class_tag_for(r["time_limit"], [r])                    # the row's --time class is one of ITS lane's (the plan refuses otherwise)
    if len({r["out_root"] for r in rows}) != len(rows):
        raise SystemExit("the rows need distinct out roots")
    if [r["index"] for r in rows] != list(range(len(rows))):
        raise SystemExit("rows are indexed in order")
    others = set()
    for name, path in M.MATRICES.items():
        if name != M.MATRIX_PAIRED_ALLMODELS_NAME and path.is_file():
            others |= {r["out_root"] for r in json.loads(path.read_text(encoding="utf-8"))["rows"]}
    if others & {r["out_root"] for r in rows}:
        raise SystemExit("an appendix row's out root collides with another document's")


def build() -> dict:
    default_doc = M.load_matrix(DEFAULT_MATRIX)
    llama_doc = M.load_matrix(LLAMA_MATRIX)
    rows = derive_rows()
    stub = {"schema": SCHEMA, "wave_root_rel": WAVE_ROOT_REL, "trigger_threshold": M.TRIGGER_THRESHOLD}
    validate(rows, stub)
    return {
        "schema": SCHEMA,
        "generated_by": "scripts/sdfbase/gsm8k/g8k_paired_allmodels.py",
        "what": ("the paper's all-model appendix GSM8K rows: ReLU's paired-write difference ReLU (the composed arm) on gemma4e4b it / llama8b it / "
                 "nemotron3nano4b main, plus nemotron3nano4b × the roster's constitution PSR object — one trial per model under the default "
                 "Inspect seed rule (the appendix column's instrument), thinking ON where the model has it (llama: not_available / card), "
                 "8,192 base + the 16,384 rung, the primary cluster"),
        "wave": WAVE, "wave_root_rel": WAVE_ROOT_REL, "trigger_threshold": M.TRIGGER_THRESHOLD,
        "lane_source": LANE_SOURCE, "role": ROLE,
        # the FROZEN INPUTS the record binds to: the published (default) matrix's — its `source` names the frozen roster + pins copies the
        # frozen-inputs test holds every registered document to (VIEWS_BY_MATRIX maps this record to the default views document)
        "source": dict(default_doc["source"]),
        "llama_source": dict(llama_doc["source"]),        # the llama anchor row's frozen inputs (informational: its matrix has its own freeze)
        "planned_rows_total": len(rows),
        "design": {
            "ruling": "2026-09-22: the composed objects are fitted on the BASE checkpoint and served "
                      "on the instruct checkpoint for gemma / llama exactly as for Qwen — the transfer IS the design; nemotron is fitted and served on "
                      "the same BF16 checkpoint because no base Nemotron exists. No re-fit is pending; these rows are canonical as configured.",
            "instrument": "evalcheck/gsm8k_paired200 as the appendix column runs it: default Inspect seed rule, one trial per model; vendor "
                          "thinking-card decode ON (gemma / nemotron), card decode under the no-switch arm (llama); cap 8,192 with the 16,384 rung "
                          "at the 1 % at-cap trigger; item-wise final state; the three numbers (correct/scored, correct/200, at-cap share split "
                          "loops / eligible / still-at-ceiling)",
            "contrast": "no `none` row in this document (the reader refuses an unsteered cell by name here): each row's contrast is the model's "
                        "in-frame unsteered GSM8K run the panel pools (`anchor_out_root` on the row; the data lane's pooled_in_frame_anchors)",
            "off_task_check": OFF_TASK_CHECK,
            "cross_model_note": "a cross-model comparison is matched on the training MIX (constv2-mix-full vs constv2ctl2-full) but NOT on effective "
                                "batch: llama's parents are batch-64, gemma's and nemotron's batch-8 (ReLU's staged_from lineage_note); Qwen's "
                                "composed rows (paired views s1p/s2p/s3p, batch-64 seed replicates) are the reference for the transfer effect",
        },
        "bank": {"lane_root": LANE_ROOT, "staged": "2026-09-22 04:24Z from the grading host staging directory (the registry's recorded, sha-verified location); "
                 "read-only; one-entry views cells_view/paired_d1_<model>_s1 with VIEW_MANIFEST.json; sha256 re-verified after the copy",
                 "objects": {k: {"staged_id": v["staged_id"], "object_sha256": v["object_sha256"], "lane": v["lane"],
                                 "parents_effective_batch": v["parents_effective_batch"]} for k, v in OBJECTS.items()}},
        "counts": {"rows": len(rows), "models": [r["model_key"] for r in rows], "views": [r["view"] for r in rows], "cells": [r["cell"] for r in rows],
                   "external": sum(1 for r in rows if r.get("object_source") == M.OBJECT_SOURCE_EXTERNAL),
                   "thinking_on": sum(1 for r in rows if r["thinking_mode"] == M.THINKING_ON),
                   "object_sha256": {r["job_id"]: r["object_sha256"] for r in rows}},
        "rows": rows,
    }


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1, sort_keys=False) + "\n"


def load_record(path: Path = OUT) -> dict:
    if not path.is_file():
        raise SystemExit(f"no appendix rows record at {path} — generate it: python scripts/sdfbase/gsm8k/g8k_paired_allmodels.py")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA or not isinstance(doc.get("rows"), list) or len(doc["rows"]) != 4:
        raise SystemExit(f"{path}: not a {SCHEMA} record with four rows")
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="exit 1 if the committed record drifts from the generator")
    a = ap.parse_args(argv)
    text = render(build())
    if a.check:
        if not OUT.is_file() or OUT.read_text(encoding="utf-8") != text:
            print(f"DRIFT {OUT}")
            return 1
        print(f"OK {OUT}: 4 appendix rows byte-identical to the generator")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}: 4 appendix rows ({', '.join(r['job_id'] for r in json.loads(text)['rows'])})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
