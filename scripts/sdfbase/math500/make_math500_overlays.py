#!/usr/bin/env python3
"""Generate the math500-thinking-on-full22 config + manifest OVERLAYS from the campaign originals. The
campaign files are UNTOUCHED (every wave test pinning them stays green); the overlays are the same files
with ONE change each:

  configs/sdfbase/math500/sdfbase_full22_<key>.math500.yaml
      the campaign config configs/sdfbase_full22_<key>.yaml with `explicit_cells` replaced by EXACTLY the cells of
      THIS MODEL's matrix rows, in matrix order (cells.py: the published three — none, prompted__safety_refusal,
      constv2-mix-full__lora_r4_ce_minus_ctl — then, for the extension models, prompted__si_constitution,
      constv2ctl2-full__lora_r4_ce, constv2-mix-full__lora_r4_ce; glm keeps the published three), and an overlay
      header prepended. Every target gets the same shape so a prompted job exports its OWN route, which the
      driver's `_served_route_for` / `_pick_cell` require;
  configs/capeval/manifests/petri_full22_<key>.math500.yaml
      the campaign manifest with `cells:` replaced by those cells' harness ids (the standup refuses a config /
      manifest cell mismatch), content otherwise identical — the SAME chat_template_hash, serving_launch_flags,
      tool_parser, prompted_prefix_hashes (which already carry the constitution SI's hex16, so appending the
      constitution cell changes no manifest hash; F2 / R14: store identity excludes explicit_cells, so no
      published store re-keys).

The two extra LoRA cells are served from the SAME `main` bank view as the published object cell; the generator
refuses a fitted extension cell the roster does not stage on that view (the view would lack its object.pt).
`prompted__si_constitution_full` is refused by name before anything is rendered.

The campaign originals may list MORE cells than the matrix's — every other served
object, and any prompted baseline the roster declares (`prompted__si_constitution` on the gemma / qwen / nemotron
views) — and the overlays serve EXACTLY the matrix's cells regardless: a roster-declared prompted cell
rides into an overlay only where the matrix schedules it (the constitution SI as rows 24-26 on the three extension models,
never on glm), leaving no other trace beyond the original's own header comments riding through. Regenerate when the
originals move; `--check` is the gate.

The QWEN SEED-REPLICATE wave (cells.SEEDREP_ROWS, indices 54-86) gets ONE pair PER ROW,
cut from the ROW'S VIEW's campaign config (`configs/sdfbase_full22_qwen35_4b_<view>.yaml`, views s1..s3 / s1p..s3p) —
`explicit_cells = [none, <cell>] + the row's standup companions` (the `none` row: `[none, <companion>]`; the companion's own
row: `[none, <companion>]`; the composed row: `[none, <composed>, <companion>]`; the Arditi seed rows: `[none, arditi_allpos]`,
a single-object standup) — so the standup's select_cells admits the row's cell AND the r=4 companion every standup registers
(the fitted --max-lora-rank rule of the reader's launch-flag compare), and the driver's `_served_route_for` finds the row's own
route. 30 pairs `configs/sdfbase/math500/sdfbase_full22_qwen35_4b_<view>.math500_<cellslug>.yaml` + manifests.

Never hand-edit the outputs: tests/test_sdfbase_math500_sbatch.py requires byte-identity (`--check`).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "math500"))
from cells import (ALLPOS_ROWS, CELL_NONE, CELL_OBJECT, FAMILY_BY_MODEL, FITTED_CELLS, HARNESS_BY_CELL, ITDOC_ROWS,  # noqa: E402
                   LLAMA_MODELS, M5T_CELLS, MODEL_KEYS, OBJECT_SOURCE_EXTERNAL, OBJECT_SOURCE_ITDOC, SEEDREP_ROWS, STEER_ROWS,
                   THINKING_MODE_BY_FAMILY, THINKING_NOT_AVAILABLE, VIEW_MAIN, cell_slug, key_view, prompted_si, row_companions,
                   standup_cells)

CONFIG_DIR = REPO / "configs" / "sdfbase" / "math500"
MANIFEST_DIR = REPO / "configs" / "capeval" / "manifests"
ROSTER = REPO / "configs" / "sdfbase_full22_roster.yaml"


def explicit_cells(key: str) -> list[str]:
    """This model's BASE-WAVE + EXTENSION matrix cells in matrix order, each once. A steering row is NOT here: its object
    lives in another lane's bank view, which holds that one cell and none of these, so it gets its own overlay pair
    (:func:`steer_overlay_config`) — mixing them would make the explicit-cell registry validate five cells the bank
    does not carry."""
    cells = list(dict.fromkeys(r["cell"] for r in M5T_CELLS if r["model_key"] == key
                               and r.get("object_source") not in ("external", OBJECT_SOURCE_ITDOC)))
    for c in cells:
        prompted_si(c)                # refuses the `_full` spelling and any unknown prompted id BY NAME
    return cells


def harness_cells(key: str) -> list[str]:
    return [HARNESS_BY_CELL[c] for c in explicit_cells(key)]


def campaign_config(key: str) -> Path:
    return REPO / "configs" / f"sdfbase_full22_{key}.yaml"


def campaign_manifest(key: str) -> Path:
    return MANIFEST_DIR / f"petri_full22_{key}.yaml"


def overlay_config(key: str) -> Path:
    return CONFIG_DIR / f"sdfbase_full22_{key}.math500.yaml"


def overlay_manifest(key: str) -> Path:
    return MANIFEST_DIR / f"petri_full22_{key}.math500.yaml"


_CELLS_BLOCK = re.compile(r"^explicit_cells:\n(?:- .*\n)+", re.M)
_MANIFEST_CELLS = re.compile(r"^  cells:\n(?:    - .*\n)+", re.M)


def _spliced_config(src_path: Path, cells: list[str]) -> str:
    """The campaign config at `src_path` with its `explicit_cells` block replaced by `cells` — every other byte rides through. The ONE
    splice every overlay config is cut with; the header in front of it is the renderer's."""
    src = src_path.read_text(encoding="utf-8")
    m = _CELLS_BLOCK.search(src)
    if not m:
        raise SystemExit(f"{src_path}: no explicit_cells block")
    return src[:m.start()] + "explicit_cells:\n" + "".join(f"- {c}\n" for c in cells) + src[m.end():]


def _spliced_manifest(src_path: Path, ids: list[str]) -> str:
    """The campaign manifest at `src_path` with its `cells` block replaced by the harness ids `ids` — every other byte rides through."""
    src = src_path.read_text(encoding="utf-8")
    m = _MANIFEST_CELLS.search(src)
    if not m:
        raise SystemExit(f"{src_path}: no cells block")
    return src[:m.start()] + "  cells:\n" + "".join(f"    - {c}\n" for c in ids) + src[m.end():]


def roster_main_cells(key: str) -> dict:
    """The roster's `main` view cells for `key` (cell id -> staged id): the objects the view stages."""
    import yaml
    doc = yaml.safe_load(ROSTER.read_text(encoding="utf-8"))
    return dict((doc.get("targets") or doc)[key]["views"][VIEW_MAIN]["cells"])


def check_fitted_cells_staged(key: str) -> None:
    """Every FITTED cell of this model's rows is staged on the roster's `main` view (the view the jobs read)."""
    staged = roster_main_cells(key)
    for c in explicit_cells(key):
        if c in FITTED_CELLS and not staged.get(c):
            raise SystemExit(f"{key}: fitted cell {c!r} is not staged on the roster's {VIEW_MAIN} view "
                             f"(configs/sdfbase_full22_roster.yaml) — its object.pt would be absent from the bank view")


def render_config(key: str) -> str:
    src = campaign_config(key).read_text(encoding="utf-8")
    m = _CELLS_BLOCK.search(src)
    if not m:
        raise SystemExit(f"{campaign_config(key)}: no explicit_cells block")
    listed = [l[2:].strip() for l in m.group(0).splitlines()[1:]]
    if CELL_OBJECT not in listed:
        raise SystemExit(f"{campaign_config(key)}: the campaign config does not list {CELL_OBJECT}")
    check_fitted_cells_staged(key)
    cells = explicit_cells(key)
    no_switch = THINKING_MODE_BY_FAMILY.get(FAMILY_BY_MODEL[key]) == THINKING_NOT_AVAILABLE
    arm_line = ("thinking: NOT AVAILABLE (the no-switch arm — no chat-template switch is sent; the fitted rows are IN SUPPORT, no tuple; E2 / P7)."
                if no_switch else "thinking ON on every request.")
    objects_line = ("the object cells (positions=all; served under the ONE render of a family with no thinking switch — in support)"
                    if no_switch else "the object cells (positions=all; served\n# thinking-ON UNDECLARED under their authorised tuples)")
    header = f"""# sdfbase_full22_{key}.math500.yaml — the math500-thinking-on-full22 OVERLAY of configs/sdfbase_full22_{key}.yaml
# GENERATED by scripts/sdfbase/math500/make_math500_overlays.py —
# never hand-edited (tests/test_sdfbase_math500_sbatch.py requires byte-identity). The campaign config's body follows
# BYTE FOR BYTE except `explicit_cells`, which is EXACTLY this model's matrix cells ({len(cells)}): the same-wave bare
# anchor, the prompted SI baselines (a prompted job exports its OWN route), and {objects_line}. Served with mt_serving_up.py --cell <cell> per
# job at the cell's prompt class's geometry entry (capability:gen_long_math, L 17,664; the constitution SI at
# capability:gen_long_math_longsi, L 24,064), --upstream-timeout-s 7200, {arm_line}
#
"""
    body = src[:m.start()] + "explicit_cells:\n" + "".join(f"- {c}\n" for c in cells) + src[m.end():]
    return header + body


def render_manifest(key: str) -> str:
    header = (f"# petri_full22_{key}.math500.yaml — the math500-thinking-on-full22 OVERLAY of petri_full22_{key}.yaml: the\n"
              f"# campaign manifest BYTE FOR BYTE except `cells`, which names EXACTLY the harness ids of\n"
              f"# configs/sdfbase/math500/sdfbase_full22_{key}.math500.yaml's explicit_cells (mt_serving_up.py refuses a\n"
              f"# config / manifest cell mismatch). GENERATED by scripts/sdfbase/math500/make_math500_overlays.py.\n")
    return header + _spliced_manifest(campaign_manifest(key), harness_cells(key))


# --------------------------------------------------------------------------- the steering rows' own overlays
def steer_overlay_config(key: str, cell: str) -> Path:
    return CONFIG_DIR / f"sdfbase_full22_{key}.math500_{cell_slug(cell)}.yaml"


def steer_overlay_manifest(key: str, cell: str) -> Path:
    return MANIFEST_DIR / f"petri_full22_{key}.math500_{cell_slug(cell)}.yaml"


def _steer_cells(cell: str) -> list[str]:
    """The steering overlay's explicit cells: the same-wave bare anchor and the ONE external object the row's bank
    view carries (`none` is served by the bare model and needs no bank entry)."""
    return [CELL_NONE, cell]


def render_steer_config(key: str, cell: str) -> str:
    """The campaign config of `key` with `explicit_cells` = `[none, <cell>]` — the ONE cell the external bank view
    stages, plus the bare anchor. Everything else rides through byte for byte, as in the base overlay."""
    cells = _steer_cells(cell)
    header = f"""# sdfbase_full22_{key}.math500_{cell_slug(cell)}.yaml — the math500-thinking-on-full22 W4 STEERING overlay of
# configs/sdfbase_full22_{key}.yaml for the external cell `{cell}` (math-eight-arms-extension plan P8).
# GENERATED by scripts/sdfbase/math500/make_math500_overlays.py — never hand-edited (the test requires byte-identity).
# The campaign config's body follows BYTE FOR BYTE except `explicit_cells`, which is exactly `{cells}`: the object this
# row serves comes from ANOTHER LANE's bank view (the row's `bank_view_rel`), which stages that one cell — so the base
# overlay's six cells would make the explicit-cell registry refuse on five objects the bank does not carry. Served with
# mt_serving_up.py --cell {cell} --expect-object-sha256 {cell}=<the admission record's object sha256>, at
# capability:gen_long_math (L 17,664) x the model's budget, --upstream-timeout-s 7200.
#
"""
    return header + _spliced_config(campaign_config(key), cells)


def render_steer_manifest(key: str, cell: str) -> str:
    header = (f"# petri_full22_{key}.math500_{cell_slug(cell)}.yaml — the W4 STEERING overlay of petri_full22_{key}.yaml:\n"
              f"# the campaign manifest BYTE FOR BYTE except `cells`, which names EXACTLY the harness ids of\n"
              f"# configs/sdfbase/math500/sdfbase_full22_{key}.math500_{cell_slug(cell)}.yaml's explicit_cells\n"
              f"# (mt_serving_up.py refuses a config / manifest cell mismatch). GENERATED by\n"
              f"# scripts/sdfbase/math500/make_math500_overlays.py.\n")
    ids = [HARNESS_BY_CELL[c] for c in _steer_cells(cell)]
    return header + _spliced_manifest(campaign_manifest(key), ids)


# --------------------------------------------------------------------------- the E3 it-doc rows' own overlays
def itdoc_campaign_config(key: str, view: str) -> Path:
    return REPO / "configs" / f"sdfbase_full22_{key}_{view}.yaml"


def itdoc_campaign_manifest(key: str, view: str) -> Path:
    return MANIFEST_DIR / f"petri_full22_{key}_{view}.yaml"


def itdoc_overlay_config(key: str, view: str, cell: str) -> Path:
    return CONFIG_DIR / f"sdfbase_full22_{key}_{view}.math500_{cell_slug(cell)}.yaml"


def itdoc_overlay_manifest(key: str, view: str, cell: str) -> Path:
    return MANIFEST_DIR / f"petri_full22_{key}_{view}.math500_{cell_slug(cell)}.yaml"


def render_itdoc_config(row: dict) -> str:
    """The `it` view's campaign config (`configs/sdfbase_full22_<key>_it.yaml`) with `explicit_cells` = `[none, <cell>]` — the ONE
    re-fit object the row's bank entry stages, plus the bare anchor (math-eight-arms-extension E3). Never the `main` overlay: the
    same cell id there is the BASE wave's object; never the `it` roster cell: the superseded batch-8 object."""
    key, view, cell = str(row["model_key"]), str(row["view"]), str(row["cell"])
    src_path = itdoc_campaign_config(key, view)
    cells = [CELL_NONE, cell]
    header = f"""# sdfbase_full22_{key}_{view}.math500_{cell_slug(cell)}.yaml — the math500-thinking-on-full22 IT-DOC overlay of
# configs/sdfbase_full22_{key}_{view}.yaml for the instruct-direct batch-64 re-fit cell `{cell}` (math-eight-arms-extension E3:
# the MATH-500 it-view row family is a SCOPE ADDITION — the published legs serve `main`-view objects on the instruct target).
# GENERATED by scripts/sdfbase/math500/make_math500_overlays.py — never hand-edited (the test requires byte-identity).
# The campaign config's body follows BYTE FOR BYTE except `explicit_cells`, which is exactly `{cells}`: the object this row serves
# is the re-fit the it-doc ADMISSION RECORD pins by (lane {row['lane']}, target {key}, view {view}, cell, sha) under its ruled staged id
# `{row['staged_id']}` — never the `{view}` view's roster cell of the same id (the superseded batch-8 object), never the `main` view's
# (the base lane's). Served with mt_serving_up.py --cell {cell} --expect-object-sha256 {cell}=<the record's object sha256>, at
# capability:gen_long_math (L 17,664) x the model's budget, --upstream-timeout-s 7200. The pair's qualification rides the row.
#
"""
    return header + _spliced_config(src_path, cells)


def render_itdoc_manifest(row: dict) -> str:
    key, view, cell = str(row["model_key"]), str(row["view"]), str(row["cell"])
    src_path = itdoc_campaign_manifest(key, view)
    header = (f"# petri_full22_{key}_{view}.math500_{cell_slug(cell)}.yaml — the IT-DOC overlay of petri_full22_{key}_{view}.yaml:\n"
              f"# the campaign manifest BYTE FOR BYTE except `cells`, which names EXACTLY the harness ids of\n"
              f"# configs/sdfbase/math500/sdfbase_full22_{key}_{view}.math500_{cell_slug(cell)}.yaml's explicit_cells\n"
              f"# (mt_serving_up.py refuses a config / manifest cell mismatch). GENERATED by\n"
              f"# scripts/sdfbase/math500/make_math500_overlays.py.\n")
    ids = [HARNESS_BY_CELL[c] for c in (CELL_NONE, cell)]
    return header + _spliced_manifest(src_path, ids)


# --------------------------------------------------------------------------- the ALL-POSITIONS Arditi row's own overlay
# The naming rule is the it-doc rows' — `<key>_<panel view>.math500_<cell slug>` over the PANEL view's campaign config —
# shared on purpose: ONE rule for "a row on a non-main panel view gets its own single-cell pair".
def allpos_overlay_config(key: str, view: str, cell: str) -> Path:
    return itdoc_overlay_config(key, view, cell)


def allpos_overlay_manifest(key: str, view: str, cell: str) -> Path:
    return itdoc_overlay_manifest(key, view, cell)


def render_allpos_config(row: dict) -> str:
    """The PANEL view's campaign config (`configs/sdfbase_full22_<key>_it.yaml`) with `explicit_cells` = `[none,
    arditi_allpos]` — the ONE external object this row serves, plus the bare anchor (arditi-qwen-all-positions). A
    SINGLE-OBJECT standup by construction: the object lives in ANOTHER LANE's bank view, which stages that one cell, so a
    second object here would both make the explicit-cell registry validate a cell the bank does not carry AND put two
    different interventions in one standup. The `it` TAG, never the `main` one: this direction was fit against THIS
    target's `it` wave view on the SERVED checkpoint, and the row sits on the `it` panel view."""
    key, view, cell = str(row["model_key"]), str(row["view"]), str(row["cell"])
    src_path = itdoc_campaign_config(key, view)
    cells = [CELL_NONE, cell]
    header = f"""# sdfbase_full22_{key}_{view}.math500_{cell_slug(cell)}.yaml — the math500-thinking-on-full22 ALL-POSITIONS ARDITI
# overlay of configs/sdfbase_full22_{key}_{view}.yaml for the external cell `{cell}` (arditi-qwen-all-positions).
# GENERATED by scripts/sdfbase/math500/make_math500_overlays.py — never hand-edited (the test requires byte-identity).
# The campaign config's body follows BYTE FOR BYTE except `explicit_cells`, which is exactly `{cells}`: a SINGLE-OBJECT
# standup. The `{view}` TAG's config, never the `main` one — the direction was fit against THIS target's `{view}` wave view on
# the served checkpoint and the row sits on the `{view}` PANEL view; the `{view}` and `main` campaign bodies are otherwise the
# same data, differing only in `explicit_cells`, `defenses_dir` and the `out_dir` / `private_run` namespace (a test pins
# that, so a future divergence between them is not absorbed silently here).
# The object comes from ANOTHER LANE's bank view (the row's `bank_view_rel`, staged under
# `{row['bank_cell_dir']}`) and is pinned by the admission record's object sha256.
# `{cell}` is NOT `arditi`: a different intervention (every position in BOTH phases; a raw-coefficient dose on a
# `vector_all` object, not a residual-norm-fraction alpha on a `static_r`), never served beside it or relabelled into its
# slot. Served with mt_serving_up.py --cell {cell} --expect-object-sha256 {cell}=<the record's object sha256>
# on the NATIVE residual slot (serving_path lora_native, in EVERY leg — a lens leg would measure something else under this
# arm's name), at capability:gen_long_math (L 17,664) x the model's budget, --upstream-timeout-s 7200.
#
"""
    return header + _spliced_config(src_path, cells)


def render_allpos_manifest(row: dict) -> str:
    key, view, cell = str(row["model_key"]), str(row["view"]), str(row["cell"])
    src_path = itdoc_campaign_manifest(key, view)
    header = (f"# petri_full22_{key}_{view}.math500_{cell_slug(cell)}.yaml — the ALL-POSITIONS ARDITI overlay of\n"
              f"# petri_full22_{key}_{view}.yaml: the campaign manifest BYTE FOR BYTE except `cells`, which names EXACTLY\n"
              f"# the harness ids of configs/sdfbase/math500/sdfbase_full22_{key}_{view}.math500_{cell_slug(cell)}.yaml's\n"
              f"# explicit_cells (mt_serving_up.py refuses a config / manifest cell mismatch). GENERATED by\n"
              f"# scripts/sdfbase/math500/make_math500_overlays.py.\n")
    ids = [HARNESS_BY_CELL[c] for c in (CELL_NONE, cell)]
    return header + _spliced_manifest(src_path, ids)


# --------------------------------------------------------------------------- a row's OWN overlay pair (the it-doc, all-positions and seed rows)
# ONE rule for "a row on a non-main panel view gets its own pair": `<key>_<key view>.math500_<cell slug>` over the KEY view's campaign
# config (cells.key_view — the it-doc rows' `it`, the all-positions row's `it`, a seed row's lane view; the campaign configs exist for
# every roster view). The it-doc spelling is the path rule's home; the wave-named callables below are its aliases.
def row_overlay_config(row: dict) -> Path:
    return itdoc_overlay_config(str(row["model_key"]), key_view(row), str(row["cell"]))


def row_overlay_manifest(row: dict) -> Path:
    return itdoc_overlay_manifest(str(row["model_key"]), key_view(row), str(row["cell"]))


seedrep_overlay_config, seedrep_overlay_manifest = row_overlay_config, row_overlay_manifest    # the seed wave's spelling (test-pinned)


# --------------------------------------------------------------------------- the QWEN SEED-REPLICATE wave's own overlays


def seedrep_cells(row: dict) -> list[str]:
    """The seed row's explicit cells — exactly the cells its standup registers (`--cell` per cell), in the standup's order: the same-wave
    bare anchor, the row's own cell (unless it IS `none`) and its standup companions once each (the view's r=4 LoRA on every campaign
    row but the companion's own; none on the single-object Arditi rows). cells.standup_cells is the ONE rule; the sbatch's
    F22_LAUNCH_CELLS is the same list in harness spelling (cells.launch_cells_for)."""
    return standup_cells(row)


def _seedrep_flags(row: dict) -> str:
    """The standup flags the header documents: the row's `--cell` (+ its pin iff fitted), then each companion's `--cell` + pin."""
    parts = [f"--cell {row['cell']}"]
    if row["fitted"]:
        parts.append(f"--expect-object-sha256 {row['cell']}=<the row's object sha256>")
    for c, _h, _sid, pin in row_companions(row):
        parts.append(f"--cell {c} --expect-object-sha256 {c}={pin}")
    return " ".join(parts)


def render_seedrep_config(row: dict) -> str:
    """The KEY view's campaign config (`configs/sdfbase_full22_qwen35_4b_<view>.yaml`) with `explicit_cells` = :func:`seedrep_cells`
    — the bare anchor, the row's cell and its standup companion — the campaign body byte for byte below."""
    key, view, cell = str(row["model_key"]), key_view(row), str(row["cell"])
    src_path = itdoc_campaign_config(key, view)
    cells = seedrep_cells(row)
    external = row.get("object_source") == OBJECT_SOURCE_EXTERNAL
    where = ("ANOTHER LANE's bank view (the row's `bank_view_rel`, the all-positions Arditi ladder's `ar_c0p3`; a SINGLE-OBJECT standup, "
             "no companion)" if external else
             f"the seed view's roster bank (`cells_view/{view}`: the batch-64 objects staged `base64s<s>__…` / `derived_base64s<s>__…` / "
             "`paired_eb64s<s>__…`, pinned in configs/sdfbase_full22_pins.json)")
    header = f"""# sdfbase_full22_{key}_{view}.math500_{cell_slug(cell)}.yaml — the math500-thinking-on-full22 QWEN SEED-REPLICATE overlay of
# configs/sdfbase_full22_{key}_{view}.yaml for the wave `qwen_seedrep` row `{cell}` on view `{view}` (b64-capability-agentic-queue).
# GENERATED by scripts/sdfbase/math500/make_math500_overlays.py — never hand-edited (the test
# requires byte-identity). The campaign config's body follows BYTE FOR BYTE except `explicit_cells`, which is exactly `{cells}`:
# the same-wave bare anchor, the row's own cell and its STANDUP COMPANION (the view's r=4 constitution LoRA, registered UNMEASURED
# on every standup of the wave so every standup composes --max-lora-rank 8 and the reader's launch-flag compare admits every
# contrast against the view's `none`; empty on the companion's own row and on the Arditi rows). The object comes from {where}.
# Served with mt_serving_up.py {_seedrep_flags(row)} on the NATIVE engine
# (serving_path lora_native, selected by kind over the declared launch cells), at the row's prompt class's geometry entry
# (capability:gen_long_math L 17,664; the constitution SI at capability:gen_long_math_longsi L 24,064), --upstream-timeout-s 7200,
# thinking ON on every request, under the OPT-IN Inspect REPLICATE seed rule at this view's base (--inspect-seed-policy replicate
# --inspect-seed-base {row['inspect_seed_base']}) and with --bank-view {row['bank_view_stamp']} stamped on every emitted row.
#
"""
    return header + _spliced_config(src_path, cells)


def render_seedrep_manifest(row: dict) -> str:
    key, view, cell = str(row["model_key"]), key_view(row), str(row["cell"])
    src_path = itdoc_campaign_manifest(key, view)
    header = (f"# petri_full22_{key}_{view}.math500_{cell_slug(cell)}.yaml — the QWEN SEED-REPLICATE overlay of petri_full22_{key}_{view}.yaml\n"
              f"# (wave qwen_seedrep, b64-capability-agentic-queue W1): the campaign manifest BYTE FOR BYTE except `cells`, which names\n"
              f"# EXACTLY the harness ids of configs/sdfbase/math500/sdfbase_full22_{key}_{view}.math500_{cell_slug(cell)}.yaml's\n"
              f"# explicit_cells (mt_serving_up.py refuses a config / manifest cell mismatch). GENERATED by\n"
              f"# scripts/sdfbase/math500/make_math500_overlays.py.\n")
    ids = [HARNESS_BY_CELL[c] for c in seedrep_cells(row)]
    return header + _spliced_manifest(src_path, ids)


def outputs():
    # the base overlay pair per model with CAMPAIGN-BANK rows: the four wave models and, since E2, the no-switch family
    # (its six llama-wave rows are served from the shared roster block's `main` view through this pair)
    for key in MODEL_KEYS + LLAMA_MODELS:
        yield overlay_config(key), render_config(key)
        yield overlay_manifest(key), render_manifest(key)
    # one pair per steering row (its model may be outside MODEL_KEYS: the rung runs on the spec's four models)
    for r in STEER_ROWS:
        key, cell = str(r["model_key"]), str(r["cell"])
        yield steer_overlay_config(key, cell), render_steer_config(key, cell)
        yield steer_overlay_manifest(key, cell), render_steer_manifest(key, cell)
    # one pair per E3 it-doc row (the `it` view's campaign config, explicit_cells [none, <cell>])
    for r in ITDOC_ROWS:
        yield row_overlay_config(r), render_itdoc_config(r)
        yield row_overlay_manifest(r), render_itdoc_manifest(r)
    # the ALL-POSITIONS Arditi row's pair (the panel view's campaign config, explicit_cells [none, arditi_allpos])
    for r in ALLPOS_ROWS:
        yield row_overlay_config(r), render_allpos_config(r)
        yield row_overlay_manifest(r), render_allpos_manifest(r)
    # one pair per QWEN SEED-REPLICATE row: the seed / paired view's campaign config, explicit_cells [none, <cell>] + companions
    for r in SEEDREP_ROWS:
        yield row_overlay_config(r), render_seedrep_config(r)
        yield row_overlay_manifest(r), render_seedrep_manifest(r)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--check", action="store_true", help="exit 1 if any committed overlay drifts from the generator")
    a = ap.parse_args(argv)
    drift = 0
    for path, text in outputs():
        if a.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                print(f"DRIFT {path.relative_to(REPO)}")
                drift += 1
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            print(f"wrote {path.relative_to(REPO)}")
    return 1 if drift else 0


if __name__ == "__main__":
    raise SystemExit(main())
