#!/usr/bin/env python3
"""The wave shell's serving-path selection — the ONE call site of
``registry_adapter.select_serving_path`` (the kind-aware default; the rule of
2026-09-15): ``common.sh f22_init`` on the cluster and ``petri_drive_full22.sh`` on the
grading host both run THIS over the launch's cell list when the caller set no
``F22_SERVING_PATH``, and eval the ``--sh`` lines. Never a shell predicate beside it.

  serving_path_select.py --manifest configs/capeval/manifests/petri_full22_<tag>.yaml [--cells-source LABEL] --sh
  serving_path_select.py --cells <harness_cell>... [--cells-source LABEL] --sh
  serving_path_select.py [...] --explicit lens|lora_native --sh      # the caller's path, recorded as such
  serving_path_select.py [...] --json                                # the record + the log line

THE CELL LIST is the launch's HARNESS cells: the (KEY, VIEW) Petri manifest's ``cells``
(``manifest.load_petri_spec`` — the config's ``explicit_cells`` in harness spelling;
``mt_serving_up.assert_manifest_matches_config`` holds the two equal at every standup), or
``--cells`` in the harness grammar (``none``, ``prompted__<si>``, ``si_<si>__<arm>``, a
Class B id such as ``alphasteer`` / ``arditi`` — the latter and ``none`` spell the same as
their TrainTest ids). Each is classified BY KIND (``registry_adapter.kind_only_cell_source``
-> ``defense_cells.native_serving_disposition`` on a ``None`` spec): this runs at
``f22_init``, before any venv or bank view exists, on the node's stdlib python + PyYAML —
the TrainTest registry (torch) is not importable there, so the objects' resolved specs are
not read and a position-masked LoRA fit reads as its natively servable kind; the standup's
own partition then skips it by name, exactly as before. Zero cells (no ``--manifest``
/ ``--cells``: a job with no (KEY, VIEW) and no launcher list) is a legal input and records
the entry-point default with that reason.

``--sh`` prints shell assignments (``shlex``-quoted) for ``eval``:
``F22_SERVING_PATH_SELECTED``, ``F22_SERVING_PATH_MODE``, ``F22_SERVING_PATH_LOG`` (the ONE
log line) and ``F22_SERVING_PATH_SELECTION_JSON`` (the compact record). A refusal (a
REFUSE-class cell, a malformed / duplicate / unmapped id, a bad ``--explicit``, an
unreadable manifest) prints ``[serving-path] REFUSED: ...`` on stderr and exits 2 — the
shell turns it into a fatal — inside the allocation, before the server launch (seconds in, not after a model
load).
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.petri import defense_cells as dc  # noqa: E402
from sisteer.capeval.petri import registry_adapter as ra  # noqa: E402

#: the shell names ``--sh`` assigns (read by common.sh f22_select_serving_path and petri_drive_full22.sh)
SH_SELECTED, SH_MODE, SH_LOG, SH_JSON = ("F22_SERVING_PATH_SELECTED", "F22_SERVING_PATH_MODE",
                                         "F22_SERVING_PATH_LOG", "F22_SERVING_PATH_SELECTION_JSON")


def select(harness_cells: list, *, explicit: str | None, cells_source: str | None) -> tuple:
    """``(selection, record)``: ``selection.record()`` plus the cells as requested (harness ids) and, for the
    operator who knows them by the config's spelling, their TrainTest ids (``registry_adapter.traintest_cell_id``)."""
    cells = [str(c) for c in harness_cells]
    sel = ra.select_serving_path(cells, ra.kind_only_cell_source(cells), explicit=explicit,
                                 cells_source=cells_source, spec_source=ra.SPEC_SOURCE_KIND_ONLY)
    record = sel.record()
    record["cells_requested"] = cells
    record["cells_requested_traintest"] = [ra.traintest_cell_id(c) for c in cells]
    return sel, record


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--manifest", default=None, help="the (KEY, VIEW) Petri manifest: its cells are the launch's, in harness spelling")
    src.add_argument("--cells", nargs="*", default=None, metavar="HARNESS_CELL", help="the launch's cells, harness grammar")
    ap.add_argument("--explicit", choices=list(ra.SERVING_PATH_POLICIES), default=None,
                    help="the caller's serving path: recorded as an EXPLICIT selection, nothing classified")
    ap.add_argument("--cells-source", default=None, help="where the cell list came from (rides the record)")
    fmt = ap.add_mutually_exclusive_group()
    fmt.add_argument("--sh", action="store_true", help="shell assignments for eval")
    fmt.add_argument("--json", action="store_true", help="the record + log_line as JSON")
    args = ap.parse_args(argv)
    try:
        if args.manifest is not None:
            cells = list(cm.load_petri_spec(Path(args.manifest)).cells)
            cells_source = args.cells_source or f"manifest:{args.manifest}"
        else:
            cells = list(args.cells or [])
            cells_source = args.cells_source or ("--cells" if cells else "none")
        sel, record = select(cells, explicit=args.explicit, cells_source=cells_source)
    except (ra.RegistryAdapterError, dc.DefenseCellError, cm.ManifestError, OSError, ValueError) as e:
        print(f"{ra.SERVING_PATH_LOG_PREFIX} REFUSED: {type(e).__name__}: {e}", file=sys.stderr)
        return 2
    compact = json.dumps(record, separators=(",", ":"), sort_keys=True)
    if args.json:
        print(json.dumps({**record, "log_line": sel.log_line}, indent=1, sort_keys=True))
    else:
        for name, value in ((SH_SELECTED, sel.selected), (SH_MODE, sel.mode), (SH_LOG, sel.log_line),
                            (SH_JSON, compact)):
            print(f"{name}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
