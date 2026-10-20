#!/usr/bin/env python3
"""ifeval-canonical-battery — THE FROZEN MATRIX'S READER. Stdlib + json only, so the Modal launcher,
the dispatcher and the analysis read it with no yaml / torch.

The 40-row matrix (30 launch + 2 held + 8 deferred) is GENERATED —
``make_ifeval_matrix.py`` -> ``configs/sdfbase/ifeval/matrix.json`` — through ``if_views.select_rows`` and FROZEN
(``--check`` pins byte-identity and resolves every ``IFEVAL_CONTRASTS`` leg). This file READS it. HELD rows (P41, a HARD
EXCLUSION) and DEFERRED rows (AlphaSteer / Arditi design slots whose object sha is filled later from a per-model
record) are IN the matrix (indices, TAGs and overlays stable) and excluded by every production consumer: ``launch_rows``
never yields one, ``refuse_held`` refuses an index or a cell naming either by name, and a dry run prints ``HELD:`` /
``DEFERRED:`` for them. Un-holding is one ``views.yaml`` edit + regenerate; filling a slot is its record + regenerate.

  if_matrix.py list [--json]              every row, status included
  if_matrix.py row --key K --view V --cell C [--rung R]   one row as JSON (+ its paths at the rung)
  if_matrix.py counts                     the acceptance counts, derived from the matrix
  ... every sub-command takes ``--matrix <path>`` (default: the frozen 40-row matrix); the sibling seed document is
  ``configs/sdfbase/ifeval/matrix_qwen_seedrep.json`` (MATRIX_QWEN_SEEDREP_PATH)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MATRIX_PATH = REPO / "configs" / "sdfbase" / "ifeval" / "matrix.json"
VIEWS_PATH = REPO / "configs" / "sdfbase" / "ifeval" / "views.yaml"
#: the sibling document of the batch-64 seed lane views (30 rows, native serving)
MATRIX_QWEN_SEEDREP_PATH = REPO / "configs" / "sdfbase" / "ifeval" / "matrix_qwen_seedrep.json"
#: every registered IFEval matrix document (mirrors contracts.IFEVAL_MATRIX_DOCUMENTS; pinned equal by test)
MATRIX_PATHS = (MATRIX_PATH, MATRIX_QWEN_SEEDREP_PATH)
MATRIX_SCHEMA = "ifeval_matrix/1"
STATUS_LAUNCH, STATUS_HELD, STATUS_DEFERRED = "launch", "held", "deferred"
STATUSES = (STATUS_LAUNCH, STATUS_HELD, STATUS_DEFERRED)
CELL_NONE = "none"
CELL_PROMPTED = "prompted__safety_refusal"
CELL_PROMPTED_CONSTITUTION = "prompted__si_constitution"
VIEW_MAIN, VIEW_IT = "main", "it"
#: the ladder, mirrored from ifeval_contracts (pinned equal by test — the mirror is checked, never trusted)
CAP_LADDER = (2048, 4096, 8192)
CAP = CAP_LADDER[0]


def load_matrix(path: Path = MATRIX_PATH) -> dict:
    """The frozen matrix document; refuses a missing or foreign file, an out-of-order index, an unknown status."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"no frozen matrix at {path} — generate it: python scripts/sdfbase/ifeval/make_ifeval_matrix.py")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != MATRIX_SCHEMA or not isinstance(doc.get("rows"), list):
        raise SystemExit(f"{path}: not a {MATRIX_SCHEMA} matrix (schema {doc.get('schema')!r})")
    for i, r in enumerate(doc["rows"]):
        if r.get("index") != i:
            raise SystemExit(f"{path}: row {i} carries index {r.get('index')!r} — the matrix is not in index order")
        if r.get("status") not in STATUSES:
            raise SystemExit(f"{path}: row {i} has status {r.get('status')!r}, not launch | held | deferred")
    return doc


_ROWS: list[dict] | None = None


def matrix_rows(path: Path = MATRIX_PATH) -> list[dict]:
    """Every row of the frozen matrix, held rows INCLUDED (the production consumers filter on `status`). ``path`` names
    another document (the seed sibling): read fresh, never cached under the frozen matrix's name."""
    global _ROWS
    if Path(path).resolve() != MATRIX_PATH.resolve():
        return load_matrix(path)["rows"]
    if _ROWS is None:
        _ROWS = load_matrix(path)["rows"]
    return _ROWS


def launch_rows(rows: list[dict] | None = None) -> list[dict]:
    rows = matrix_rows() if rows is None else rows
    return [r for r in rows if r["status"] == STATUS_LAUNCH]


def held_rows(rows: list[dict] | None = None) -> list[dict]:
    rows = matrix_rows() if rows is None else rows
    return [r for r in rows if r["status"] == STATUS_HELD]


def deferred_rows(rows: list[dict] | None = None) -> list[dict]:
    """The DEFERRED design slots (status ``deferred``): an object sha to be filled from a per-model record; never launched."""
    rows = matrix_rows() if rows is None else rows
    return [r for r in rows if r["status"] == STATUS_DEFERRED]


def row_for(model_key: str, view: str, cell: str, rows: list[dict] | None = None) -> dict:
    rows = matrix_rows() if rows is None else rows
    for r in rows:
        if (r["model_key"], r["view"], r["cell"]) == (model_key, view, cell):
            return r
    raise SystemExit(f"not a matrix row: ({model_key!r}, {view!r}, {cell!r}) — the frozen matrix's (key, view, cell) "
                     f"triples are {[(r['model_key'], r['view'], r['cell']) for r in rows]}")


def rows_of_slug(slug: str, rows: list[dict] | None = None) -> list[dict]:
    """The rows one launcher slug (a (model, view) bank) serves, held rows INCLUDED (the launcher refuses them by name)."""
    rows = matrix_rows() if rows is None else rows
    return [r for r in rows if r["slug"] == slug]


def refuse_held(rows: list[dict], *, flag: str) -> None:
    """A production consumer naming a HELD row refuses BY NAME (P41: no configuration, no store, no estimate line) — and
    a DEFERRED one the same way (a design slot with no object sha is not launchable until its record is filled)."""
    held = [r for r in rows if r["status"] == STATUS_HELD]
    if held:
        raise SystemExit(f"{flag}: HELD row(s) named — {[(r['index'], r['job_id'], r['held_reason']) for r in held]}; "
                         "a held cell is a hard exclusion (P41): un-hold it in views.yaml and regenerate the matrix")
    deferred = [r for r in rows if r["status"] == STATUS_DEFERRED]
    if deferred:
        raise SystemExit(f"{flag}: DEFERRED row(s) named — "
                         f"{[(r['index'], r['job_id'], (r.get('deferred') or {}).get('record')) for r in deferred]}; "
                         "a deferred slot has no object sha yet: fill its record (object_sha256, view, rung, selection_rule) "
                         "and regenerate the matrix before launching it")


refuse_unlaunchable = refuse_held


def counts(rows: list[dict] | None = None) -> dict:
    """The acceptance counts, DERIVED from the matrix: rows 40 / held 2 / launched 30 / deferred 8 / fitted 14 (launch +
    held) / fitted_launched 12 / in_support 18 / prompted_constitution 5 / header_located 0 / tags 7 / models 4 for THIS
    experiment's views.yaml."""
    rows = matrix_rows() if rows is None else rows
    launched = [r for r in rows if r["status"] == STATUS_LAUNCH]
    held = [r for r in rows if r["status"] == STATUS_HELD]
    deferred = [r for r in rows if r["status"] == STATUS_DEFERRED]
    fitted_l = [r for r in launched if r["fitted"]]
    contrasts = [r for r in launched if r["cell"] != CELL_NONE]
    tags = list(dict.fromkeys(r["tag"] for r in rows))
    slugs = list(dict.fromkeys(r["slug"] for r in rows))
    paths = [p for r in rows for p in (r["out_root"], *r["rung_out_roots"].values(), *r["subset_paths"].values(),
                                       *r["budget_paths"].values(), *r["cap_ladder_paths"].values(),
                                       *r["ladder_state_paths"].values(), *r["dispatch_paths"].values())]
    return {
        "rows": len(rows), "held": len(held), "launched": len(launched), "deferred": len(deferred),
        "fitted": sum(r["fitted"] for r in rows if r["status"] != STATUS_DEFERRED), "fitted_launched": len(fitted_l),
        "in_support": sum(not r["fitted"] for r in launched),
        "prompted_constitution": sum(r["cell"] == CELL_PROMPTED_CONSTITUTION for r in launched),
        "prompted_safety": sum(r["cell"] == CELL_PROMPTED for r in launched),
        "header_located": sum(r["header_located"] for r in launched),
        "declared": sum(r["declared"] for r in rows),
        "contrasts_designed": len(contrasts),
        "tags": tags, "n_tags": len(tags), "slugs": slugs, "n_slugs": len(slugs),
        "by_tag": {t: {"rows": sum(r["tag"] == t for r in rows), "held": sum(r["tag"] == t for r in held),
                       "launched": sum(r["tag"] == t for r in launched), "deferred": sum(r["tag"] == t for r in deferred)}
                   for t in tags},
        "by_model": {m: {"launched": sum(r["model_key"] == m for r in launched), "held": sum(r["model_key"] == m for r in held),
                         "deferred": sum(r["model_key"] == m for r in deferred)}
                     for m in dict.fromkeys(r["model_key"] for r in rows)},
        "by_model_view": {f"{m}/{v}": {"launched": sum((r["model_key"], r["view"]) == (m, v) for r in launched),
                                       "held": sum((r["model_key"], r["view"]) == (m, v) for r in held),
                                       "deferred": sum((r["model_key"], r["view"]) == (m, v) for r in deferred)}
                          for m, v in dict.fromkeys((r["model_key"], r["view"]) for r in rows)},
        "held_rows": [{"index": r["index"], "job_id": r["job_id"], "tag": r["tag"], "cell": r["cell"]} for r in held],
        "deferred_rows": [{"index": r["index"], "job_id": r["job_id"], "tag": r["tag"], "cell": r["cell"],
                           "record": (r.get("deferred") or {}).get("record")} for r in deferred],
        "distinct_paths": len(set(paths)), "n_paths": len(paths),
        "distinct_out_roots": len({r["out_root"] for r in rows}),
        "pins": {k: sum(1 for r in rows if r["object_sha256"] and r["model_key"] == k)
                 for k in dict.fromkeys(r["model_key"] for r in rows)},
        "unpinned_deferred": sum(1 for r in deferred if not r["object_sha256"]),
        "families": sorted({r["family"] for r in rows}),
        "images": sorted({r["image"] for r in launched}),
        "models": list(dict.fromkeys(r["model_key"] for r in rows)),
        "two_view_models": sorted({r["model_key"] for r in rows if any(x["model_key"] == r["model_key"] and x["view"] == VIEW_IT for x in rows)
                                   and any(x["model_key"] == r["model_key"] and x["view"] == VIEW_MAIN for x in rows)}),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--matrix", type=Path, default=MATRIX_PATH, help="the matrix document (default: the frozen 40-row matrix)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("--json", action="store_true")
    p = sub.add_parser("row"); p.add_argument("--key", required=True); p.add_argument("--view", required=True)
    p.add_argument("--cell", required=True); p.add_argument("--rung", type=int, default=CAP)
    sub.add_parser("counts")
    a = ap.parse_args(argv)
    rows = matrix_rows(a.matrix)
    if a.cmd == "list":
        if a.json:
            print(json.dumps(rows, indent=1))
        else:
            for r in rows:
                print(f"{r['index']:2d} {r['status']:6s} {r['job_id']:40s} {r['tag']:20s} {r['cell']}")
        return 0
    if a.cmd == "row":
        r = row_for(a.key, a.view, a.cell, rows)
        if a.rung not in CAP_LADDER:
            raise SystemExit(f"--rung must be one of {CAP_LADDER}")
        print(json.dumps({**r, "rung": a.rung,
                          "out_root_at_rung": r["out_root"] if a.rung == CAP else r["rung_out_roots"][str(a.rung)]}, indent=1))
        return 0
    print(json.dumps(counts(rows), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
