#!/usr/bin/env python3
"""THE FALLBACK: the three Arditi all-positions GSM8K rows of the `qwen_seedrep` wave,
FILED BY THE LANE'S LAUNCHER from the `arditi_allpos` matrix's row shape at seed bases 1..3 — a lane RECORD beside the matrix, never a
row of it.

WHY A RECORD AND NOT THREE MATRIX ROWS. The plan's default was an EXTERNAL row per seed view (`arditi_allpos` on `s<s>` with
`inspect_seed_base` s). The matrix tooling refuses that BY NAME, twice, and both refusals are lane invariants worth keeping:
  * g8k_views._steering_entries: `g8k_cells.EXTERNAL_ROW_VIEW_BY_CELL` declares the all-positions arm's view PER ARM (`it`), and an
    entry naming another view refuses so two arms can never be mixed into one column by a views-file edit.
  * g8k_views.select_rows: a matrix that declares `inspect_seed_policy` / `standup_companions` schedules no steering row ("their row
    builders carry no seed policy" / "no companion") — and the companion rule is the deeper reason: every LAUNCH row of the seedrep
    matrix must register the view's r=4 parent beside its own cell (O12a / O13), while the Arditi engine is a SINGLE-OBJECT standup
    (a second object beside `arditi_allpos` would be skipped by name). An Arditi row cannot satisfy the matrix's
    companion invariant without breaking its own standup.
So the three rows are DERIVED here from the `arditi_allpos` matrix's ONE row — the same object, pin, tuple names, sampling row,
admission facts and bank binding — re-keyed onto the seed views (`s<s>`, TAG `qwen35_4b_s<s>`), under the seedrep wave root, with
the wave's replicate seed policy at base s, the wave's common geometry class, NO companion, and their VALUE filed unpaired (the
reader pairs an Arditi row against the view's native `none` only if S2's `_comparable` verdict admits it — the single-object engine
attests `max_lora_rank` 1 against the view engines' 8). The record also names the config the Arditi engine is stood up from: the
`arditi_allpos_qwen35_4b_ar_c0p3.yaml` standup config + Petri manifest at a source AT OR AFTER 0f3576f3, copied into the lane
namespace by `render_arditi_seedrep_configs.py`, with the blob sha256s read at that sha so a copy that drifts from the
source is a refusal, never a guess.

  python scripts/sdfbase/gsm8k/g8k_seedrep_arditi.py            # write configs/sdfbase/gsm8k/arditi_rows_qwen_seedrep.json
  python scripts/sdfbase/gsm8k/g8k_seedrep_arditi.py --check    # exit 1 on drift

The record is READ by: the lane's Modal launcher (the three Arditi GSM8K legs' argv — `--inspect-seed-policy replicate
--inspect-seed-base <s>`, the row's out roots, the common geometry entry), the coverage / at-cap tables (27 planned rows = the
matrix's 24 + these 3), and tests/test_gsm8k_seedrep_common_geometry.py (S0 proof-set gate 1: every planned row through the reader).
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
import g8k_views as V  # noqa: E402

REPO = HERE.parents[2]
SCHEMA = "g8k_seedrep_arditi_rows/1"
OUT = REPO / "configs" / "sdfbase" / "gsm8k" / "arditi_rows_qwen_seedrep.json"
SEEDREP_MATRIX = M.MATRICES[M.MATRIX_QWEN_SEEDREP_NAME]
ALLPOS_MATRIX = M.MATRICES[M.MATRIX_ALLPOS_NAME]
ALLPOS_PINS = REPO / "configs" / "arditi_allpos_pins.json"
MODEL_KEY = "qwen35_4b"
CELL = M.CELL_ARDITI_ALLPOS
SEED_VIEWS = ("s1", "s2", "s3")
#: how the reader treats an Arditi row's contrast (the default: the VALUE is filed; the pairing
#: against the view's native `none` happens only where S2's `_comparable` verdict admits the two standups' launch flags)
ROLE = "arditi_unpaired_value"
#: THE CONFIG SOURCE: the Arditi engine is stood up from the LANE COPY of the standup config + Petri manifest,
#: taken from a source at or after 0f3576f3 (the OR-Bench ladder fix; branch `allpos-orbench`, whose tip IS 0f3576f3). Neither file is
#: on the wave tree itself (W = 5fa280ef carries only the arm's pins, roster, matrix and own-overlay pair); the blob sha256s below were
#: read with `git show 0f3576f3:<path> | sha256sum` and are byte-identical to the copies on the `arditi-allpos-reseed-c0p3` branch
#: (925895e4). The lane copies are rendered into configs/seedrep/; a copy whose sha256 is not one of these is a drifted source.
ARDITI_SOURCE = {
    "sha": "0f3576f38ef739b986fe9a3ea379ae86e3081beb",
    "branch": "allpos-orbench",
    "note": "The lane copy of the source Arditi config must come from a source at or after 0f3576f3; this sha IS 0f3576f3 "
            "(the source's tip). 0f3576f3 is an ancestor of neither W (5fa280ef) nor origin/main (ab6f7c2e) at the time of writing, so "
            "the paths below do not exist on the wave tree — they are read from the source commit and copied by WS1-D'.",
    "standup_config": {"path": "configs/arditi_allpos_qwen35_4b_ar_c0p3.yaml",
                       "sha256": "1c5d8b33ef355bdf" + "0" * 0,     # sha256_16 prefix as read; the full digest is asserted by the test where the commit is present
                       "sha256_prefix_len": 16},
    "petri_manifest": {"path": "configs/capeval/manifests/arditi_allpos_qwen35_4b_ar_c0p3.yaml", "sha256": "6898b5bae7f6d7db", "sha256_prefix_len": 16},
    "pins": {"path": "configs/arditi_allpos_pins.json", "sha256": "8607fd979ae7e059", "sha256_prefix_len": 16,
             "note": "byte-identical to the wave tree's configs/arditi_allpos_pins.json (blob acf25f01)"},
    "lane_copy": {"standup_config": "configs/seedrep/arditi_allpos_qwen35_4b_ar_c0p3_seedrep.yaml",
                  "petri_manifest": "configs/seedrep/arditi_allpos_qwen35_4b_ar_c0p3_seedrep_manifest.yaml",
                  "rendered_by": "scripts/sdfbase/seedrep/render_arditi_seedrep_configs.py (WS1-D')"},
}


def matrix_refusals() -> dict:
    """The two refusals the matrix tooling gives an Arditi row on a seed view, captured from the LIVE functions (never retyped): the
    views loader's per-arm view rule, and select_rows' seed-policy / companion rule for a steering row on this matrix."""
    out = {}
    spec = V.load_views_yaml(M.VIEWS_BY_MATRIX[M.MATRIX_QWEN_SEEDREP_NAME])
    entry = {"key": MODEL_KEY, "view": "s1", "arms": [CELL], "reason": "probe: an Arditi row on a seed view", "since": "2026-09-17"}
    try:
        V._steering_entries({"steering": [entry]}, {MODEL_KEY: ["s1", "s2", "s3"]}, path=Path("views_qwen_seedrep.yaml"))
        out["views_loader"] = None
    except SystemExit as e:
        out["views_loader"] = str(e)
    frozen = V.frozen_input_paths(spec, REPO)
    roster, pins = V.load_roster(frozen[0]), V.load_pins(frozen[1])
    try:
        V.select_rows(roster, pins, spec.views, holds=spec.holds, exclude=spec.exclude, schema=spec.schema, trigger_threshold=spec.trigger_threshold,
                      where="g8k_seedrep_arditi probe", extensions=spec.extensions, steering=[{**entry, "view": M.VIEW_IT}],
                      admission=V.load_admission(REPO), row_overlays=spec.row_overlays, wave=spec.wave,
                      inspect_seed_policy=spec.inspect_seed_policy, inspect_seed_base_by_view=spec.inspect_seed_base_by_view,
                      standup_companions=spec.standup_companions)
        out["select_rows"] = None
    except SystemExit as e:
        out["select_rows"] = str(e)
    return out


def derive_rows(seedrep_doc: dict, allpos_doc: dict) -> list[dict]:
    """The three Arditi rows: the `arditi_allpos` matrix's ONE row re-keyed onto the seed views under the seedrep wave's facts."""
    src = [r for r in allpos_doc["rows"] if r["cell"] == CELL and r["model_key"] == MODEL_KEY]
    if len(src) != 1:
        raise SystemExit(f"{ALLPOS_MATRIX}: expected exactly one {MODEL_KEY}/{CELL} row, found {len(src)}")
    base = src[0]
    if base.get("object_source") != M.OBJECT_SOURCE_EXTERNAL or base.get("status") != M.STATUS_LAUNCH:
        raise SystemExit(f"{ALLPOS_MATRIX}: the {CELL} row is not a launched EXTERNAL row ({base.get('object_source')!r}, {base.get('status')!r})")
    pins = json.loads(ALLPOS_PINS.read_text(encoding="utf-8"))
    pin = pins[MODEL_KEY][base["staged_id"]]["object_sha256"]
    if pin != base["object_sha256"]:
        raise SystemExit(f"{ALLPOS_PINS.name} {MODEL_KEY}/{base['staged_id']} pin {pin} != the matrix row's object_sha256 {base['object_sha256']}")
    policy = seedrep_doc.get("inspect_seed_policy")
    bases = seedrep_doc.get("inspect_seed_base_by_view") or {}
    gclass = seedrep_doc.get("geometry_class")
    if policy != M.INSPECT_SEED_POLICY_REPLICATE or not set(SEED_VIEWS) <= set(bases) or gclass not in M.GEOMETRY_CLASSES:
        raise SystemExit(f"{SEEDREP_MATRIX}: the seedrep document must declare inspect_seed_policy replicate, a base per seed view and a known "
                         f"geometry_class (got {policy!r}, {sorted(bases)}, {gclass!r})")
    # the document may ALSO carry the paired views s<s>p (the composed difference_relu arm beside the seed's companion), each at
    # ITS seed's base — no other extra view, and no paired view whose base differs from its seed view's. The Arditi rows stay on the
    # seed views alone (the Arditi value is unpaired; the paired views carry no Arditi cell).
    extra = sorted(set(bases) - set(SEED_VIEWS))
    bad = [v for v in extra if not (v.endswith("p") and v[:-1] in SEED_VIEWS and bases[v] == bases[v[:-1]])]
    if bad:
        raise SystemExit(f"{SEEDREP_MATRIX}: inspect_seed_base_by_view carries views that are neither seed views nor their paired views at the "
                         f"same base: {bad} (bases {dict(sorted(bases.items()))})")
    prefix = M.out_root_prefix_for(seedrep_doc["wave"])
    rows = []
    for i, view in enumerate(SEED_VIEWS):
        tag = M.tag_for(MODEL_KEY, view)
        row = copy.deepcopy(base)
        # the fields the re-keying MOVES (every other field is the source row, byte for byte)
        row.update({
            "index": i, "job_id": f"g8k-{MODEL_KEY}-{view}-{M.cell_short(CELL)}", "view": view, "tag": tag,
            "trigger_threshold": float(seedrep_doc["trigger_threshold"]),
            "out_root": f"{prefix}/{tag}/{M.cell_slug(CELL)}/{base['sampling_mode']}",
            "rung_out_root": f"{prefix}/{tag}/{M.cell_slug(CELL)}/{base['sampling_mode']}__rung{M.RUNG_CAP}",
            "wave": seedrep_doc["wave"],
        })
        # `own_overlay` STAYS True (2026-09-21, the Slurm path): gsm8k_thinking.sbatch resolves an own-overlay row's pair
        # sdfbase_full22_<TAG>.gsm8k_arditi_allpos.yaml — explicit_cells [none, arditi_allpos] cut from the seed view's campaign config
        # (make_gsm8k_overlays --matrix arditi_seedrep) — while the bank is the EXTERNAL lane's view (bank_view_rel, checked to the pin).
        # The Modal launcher stands the engine up from the lane's config copy (standup_config below) and ignores the overlay.
        row["own_overlay"] = True
        # APPENDED (the seedrep wave's facts; the field order of the source row never moves)
        row["inspect_seed_policy"] = policy
        row["inspect_seed_base"] = int(bases[view])
        row["geometry_class"] = gclass
        # NO companion fields on a record row (2026-09-21): a single-object engine registers none, and g8k_cells' companion check reads
        # "carries the companion fields but the document declares no standup_companions" as a collision — the record declares none,
        # so its rows carry none (row_companions() reads [] off absent fields; validate() below still asserts it)
        for k in ("standup_companion_cells", "standup_companion_harness_cells", "standup_companion_staged_ids", "standup_companion_pins"):
            row.pop(k, None)
        row["role"] = ROLE
        row["standup_config"] = ARDITI_SOURCE["lane_copy"]["standup_config"]
        row["standup_manifest"] = ARDITI_SOURCE["lane_copy"]["petri_manifest"]
        row["standup_config_source_sha"] = ARDITI_SOURCE["sha"]
        rows.append(row)
    return rows


def validate(rows: list[dict], seedrep_doc: dict) -> None:
    """The record's own checks: the seed fields agree with the seedrep document, the geometry resolves to the common class at both
    rungs, the pin is the arm's, no companion, distinct out roots disjoint from the matrix's, and the rows are one object at three bases."""
    for r in rows:
        problem = M.inspect_seed_row_problem(r, seedrep_doc)
        if problem:
            raise SystemExit(f"{r['job_id']}: {problem}")
        for rung in M.RUNGS:
            ev, L = M.geometry_for_cell(r["cell"], rung, model_key=r["model_key"], geometry_class=r["geometry_class"])
            want_ev, want_l = M.GEOMETRY_CLASSES[r["geometry_class"]][0][rung], M.GEOMETRY_CLASSES[r["geometry_class"]][1][rung]
            if (ev, L) != (want_ev, want_l):
                raise SystemExit(f"{r['job_id']}: geometry {ev} x {L} at {rung} != the class's {want_ev} x {want_l}")
        if M.row_companions(r) != []:
            raise SystemExit(f"{r['job_id']}: an Arditi row registers NO companion (a single-object engine, plan §2.3)")
        if r["object_sha256"] != rows[0]["object_sha256"] or r["staged_id"] != rows[0]["staged_id"]:
            raise SystemExit(f"{r['job_id']}: the three rows serve ONE object")
    if len({r["out_root"] for r in rows}) != 3 or len({r["inspect_seed_base"] for r in rows}) != 3:
        raise SystemExit("the three Arditi rows need three distinct out roots and three distinct seed bases")
    matrix_roots = {mr["out_root"] for mr in seedrep_doc["rows"]}
    if matrix_roots & {r["out_root"] for r in rows}:
        raise SystemExit("an Arditi row's out root collides with a matrix row's")


def build() -> dict:
    seedrep_doc = M.load_matrix(SEEDREP_MATRIX)
    allpos_doc = M.load_matrix(ALLPOS_MATRIX)
    rows = derive_rows(seedrep_doc, allpos_doc)
    validate(rows, seedrep_doc)
    return {
        "schema": SCHEMA,
        "generated_by": "scripts/sdfbase/gsm8k/g8k_seedrep_arditi.py",
        "what": ("the three Arditi all-positions GSM8K rows of the qwen_seedrep wave, FILED BY THE LANE'S LAUNCHER from the arditi_allpos "
                 "matrix's row shape at seed bases 1..3 (the fallback for a matrix that refuses steering rows) — not rows of "
                 "matrix_qwen_seedrep.json, whose tooling refuses a steering row on this matrix by name (matrix_refusals)"),
        "wave": seedrep_doc["wave"], "wave_root_rel": seedrep_doc["wave_root_rel"],
        "planned_rows_total": len(seedrep_doc["rows"]) + len(rows),
        "trigger_threshold": float(seedrep_doc["trigger_threshold"]),      # load_matrix reads the record with schema-2 semantics (the Slurm launch)
        # the FROZEN INPUTS the record binds to (the frozen-inputs test walks every registered document and holds `source.roster_sha256` /
        # `source.pins_sha256` to the views file's frozen copies): the record regenerates from the seedrep DOCUMENT, whose own `source`
        # names exactly those copies — copied verbatim, so a registry move can never move this record either (`arditi_source` below is
        # the OTHER provenance: the Arditi config copy the engine is stood up from)
        "source": dict(seedrep_doc["source"]),
        # the document-level facts g8k_cells' schema-2 row checks read (inspect_seed_row_problem / geometry_class_row_problem / the
        # companion check) — copied from the seedrep document the rows are derived from, so the record loads as a launchable document
        "inspect_seed_policy": seedrep_doc["inspect_seed_policy"],
        "inspect_seed_base_by_view": {v: int(b) for v, b in (seedrep_doc.get("inspect_seed_base_by_view") or {}).items() if v in SEED_VIEWS},
        M.GEOMETRY_CLASS_KEY: seedrep_doc[M.GEOMETRY_CLASS_KEY],
        # NO standup_companions on the record document: the Arditi engine is single-object; its rows carry no companion fields
        "matrix": {"path": str(SEEDREP_MATRIX.relative_to(REPO)), "rows": len(seedrep_doc["rows"]), "roster_sha256": seedrep_doc["source"]["roster_sha256"]},
        "allpos_matrix": {"path": str(ALLPOS_MATRIX.relative_to(REPO)), "roster_sha256": allpos_doc["source"]["roster_sha256"]},
        "matrix_refusals": matrix_refusals(),
        "arditi_source": ARDITI_SOURCE,
        "role": ROLE,
        "counts": {"rows": len(rows), "views": [r["view"] for r in rows], "seed_bases": [r["inspect_seed_base"] for r in rows],
                   "geometry_class": rows[0]["geometry_class"], "object_sha256": rows[0]["object_sha256"]},
        "rows": rows,
    }


def render(doc: dict) -> str:
    return json.dumps(doc, indent=1, sort_keys=False) + "\n"


def bank_view_stamp(row: dict) -> str:
    """The ``--bank-view`` an Arditi GSM8K child of ``row`` stamps: the row's SOURCE bank
    (``bank_view`` = ``ar_c0p3``, where its object lives), not its evaluation view (``view`` = ``s<s>``) — the ONE
    rule ``contracts.expected_bank_view_stamp`` gives every row, so the reader holds the children's rows to the
    same value the launcher composed. A seed-replicate MATRIX row (no source bank) stamps its own view."""
    from sisteer.capeval.tinycap import contracts as C
    return C.expected_bank_view_stamp(view=row["view"], bank_view=row.get("bank_view"))


def load_record(path: Path = OUT) -> dict:
    """The committed record; refuses a missing or foreign file by name."""
    if not path.is_file():
        raise SystemExit(f"no Arditi rows record at {path} — generate it: python scripts/sdfbase/gsm8k/g8k_seedrep_arditi.py")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != SCHEMA or not isinstance(doc.get("rows"), list) or len(doc["rows"]) != 3:
        raise SystemExit(f"{path}: not a {SCHEMA} record with three rows")
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
        print(f"OK {OUT}: 3 Arditi rows byte-identical to the generator")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT}: 3 Arditi rows (views s1..s3, seed bases 1..3, geometry class {json.loads(text)['counts']['geometry_class']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
