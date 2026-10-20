#!/usr/bin/env python
"""Re-derive a tinycap store's served EvalPlus numbers OFF-VOLUME under the
current ``pass@1_plus`` definition (PW12 / E19, NEW High).

``pass@1_plus`` had two definitions under one metric identity: evalplus
0.3.1's own headline — a sample passes the plus set only when it passes the
BASE tests AND the plus tests (``evaluate.py`` ``new_correct``; every
converter of this tree since PW10) — and the earlier ``plus_status`` alone,
which the retained 618f9dd stores were converted under (their EvalPlus
summaries name no ``metric_def``, so ``cell_controls.load_controls_summaries``
stamps them ``plus_only``). The two differ by the base-fail / plus-pass items
(two on every MBPP+ run: 0.5 pp), enough to flip a close gate.

This script recomputes the rows and the summary payload from the store's
RETAINED ``evalplus/<dataset>[_open]/eval_results.json`` + ``gen_meta.json``
through the SAME functions the conversion uses today
(``evalplus_runner.eval_results_to_rows`` / ``_summary_payload``), and writes
the result as a SIDECAR under ``--out`` — never into the store, never into
its out_root, never a ``done/`` marker or a ``controls_summary.json``: the
canonical summary keeps the retained (``plus_only``) numbers, explicitly
marked, and the sidecar is the current-definition reading for the results
operator to tabulate against. A re-conversion of the stores on the volume
would regenerate rows under the same ``harness_config_hash`` (PW10-8), which
is why this exists instead.

Usage (from a checkout; the stores may be the local mirror copies or a
mounted volume path — reading only)::

  PYTHONPATH=src python scripts/tinycap_rederive_evalplus.py \\
      <out_root>/cells/<cell_key> [<cell_dir> ...] --out $GRADE/tinycap/rederived

Output per store: ``<out>/<cell_key>/rows.jsonl`` (the re-derived EvalPlus
rows, ``metric_def`` in ``extra_json``), ``<out>/<cell_key>/summary/<task
marker>.json`` (the re-derived payload, ``metric_def: base_and_plus``) and
``<out>/<cell_key>/rederivation.json`` (provenance: source store, producer
sha, identity hash, the eval results' sha256, retained vs re-derived headline,
the flipped items). Tasks are selected by the store's DONE ``evalplus/*``
markers (``--tasks`` narrows); a task whose results are missing refuses,
named.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE.parent / "src") not in sys.path:
    sys.path.insert(0, str(HERE.parent / "src"))

from sisteer.capeval.tiers import evalplus_runner as er
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C


def store_evalplus_tasks(cell_dir: Path) -> list[tuple[str, bool, str]]:
    """``[(dataset, open_variant, task)]`` for every DONE EvalPlus task of the store."""
    out = []
    for dataset in C.EVALPLUS_DATASETS:
        for open_variant in (False, True):
            task = C.evalplus_task(dataset, open_variant=open_variant)
            if completion.is_done(cell_dir, task):
                out.append((dataset, open_variant, task))
    return out


def rederive_store(cell_dir: Path, out: Path, *, tasks: set[str] | None = None) -> list[dict]:
    """Re-derive every selected EvalPlus task of ``cell_dir`` into ``out``;
    returns the per-task records (without rows) for the report."""
    selected = [(d, o, t) for d, o, t in store_evalplus_tasks(cell_dir) if not tasks or t in tasks]
    if tasks:
        missing = sorted(tasks - {t for _d, _o, t in selected})
        if missing:
            raise SystemExit(f"[rederive] {cell_dir}: no DONE marker for {missing}")
    if not selected:
        raise SystemExit(f"[rederive] {cell_dir}: no DONE evalplus/* task to re-derive")
    records = []
    for dataset, open_variant, _task in selected:
        derived = er.rederive_evalplus(cell_dir, dataset=dataset, open_variant=open_variant)
        er.write_rederivation(out, derived)
        records.append({k: v for k, v in derived.items() if k != "rows"})
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cell_dirs", nargs="+", help="<out_root>/cells/<cell_key> store directories (read only)")
    ap.add_argument("--out", required=True,
                    help="sidecar root OFF the volume; refuses a path inside any source store or its out_root")
    ap.add_argument("--tasks", default="",
                    help="comma-separated evalplus/* row tasks to re-derive (default: every DONE one)")
    a = ap.parse_args(argv)
    tasks = {t for t in a.tasks.split(",") if t} or None
    for t in tasks or ():
        if t not in C.EVALPLUS_TASKS:
            raise SystemExit(f"[rederive] {t!r} is not an evalplus row task: {list(C.EVALPLUS_TASKS)}")
    out = Path(a.out)
    print(f"[rederive] metric_def={er.PLUS_METRIC_DEF} (base AND plus); sidecars under {out}")
    for d in a.cell_dirs:
        cell_dir = Path(d)
        if not (cell_dir / "manifest.json").is_file():
            raise SystemExit(f"[rederive] {cell_dir}: no manifest.json — not a cell store")
        try:
            records = rederive_store(cell_dir, out, tasks=tasks)
        except er.EvalPlusSeamError as e:      # a store/seam refusal is the operator's error, not a traceback
            raise SystemExit(f"[rederive] {e}") from None
        for rec in records:
            retained = rec["retained_summary"] or {}
            print(f"[rederive] {rec['cell_key']} {rec['task']}: retained pass@1_plus="
                  f"{retained.get('pass_at_1_plus')} ({rec['retained_metric_def']}) -> "
                  f"{rec['summary']['pass_at_1_plus']:.6f} ({er.PLUS_METRIC_DEF}); "
                  f"delta={rec['delta_pass_at_1_plus']}; flipped={rec['flipped_items']}; "
                  f"producer={str(rec['producer_tree_sha'])[:7]} eval_results sha256="
                  f"{rec['eval_results_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
