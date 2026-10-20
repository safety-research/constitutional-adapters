#!/usr/bin/env python
"""Merge a truncation-ladder rung into a headline judged tree (user rule 2026-09-10: rerun truncated
responses >1% at the next cap into a NEW root beside the original; the pair is the measurement).

  src   = the first rung's judged root   ($SHARED/exp/misalign/<target>,     agentic at 3000)
  rerun = the ladder rung's judged root  ($SHARED/exp/misalign_16k/<target>, only the selected pairs at 16384)
  out   = a NEW root: for every laddered cell, judged.jsonl = src rows whose response completed or is a
          repetition loop (degenerate; a longer cap only yields a longer loop) + the rerun rows for the
          selected pairs; each row carries `ladder_rung` (the cap it was generated under). Families that
          were not laddered (em8q, the paper's 600-token protocol) are copied verbatim, as are the root's
          cells.json / endpoint.snapshot.json / serving/, so rows.py + analyze.py run on `out` unchanged.

Refuses when a selected pair has no judged rerun row (the rung is incomplete) unless --allow-missing, in
which case the src row stands and is counted as `unrerun`. Refuses, before anything is written, when --src-root
or --rerun-root is not a directory, when --src-root holds no laddered family cell (nothing to merge), and when
--rerun-root holds none unless --allow-missing (an empty rung lets every src row stand). Prints the residual per
cell x behavior: the share still at cap after the rung, the degenerate (loop) count, and the unrerun count — the
rule hides nothing. --rows-analyze then files rows and analyses `out` as `--target`.

Usage ($GRADE is the site's grade directory, SISTEER_GRADE_DIR):
  python scripts/sdfbase/full22/misalign/misalign_ladder_merge.py \\
      --src-root $GRADE/misalign/glm46vflash --rerun-root $GRADE/misalign_16k/glm46vflash \\
      --out-root $GRADE/misalign_ladder/glm46vflash --target glm46vflash --rows-analyze
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

HERE = Path(__file__).resolve()
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "src"))

from sisteer.capeval.misalign.runner import RERUN_MIN_ZLIB_RATIO, zlib_ratio  # noqa: E402

LADDERED_FAMILIES = ("agentic",)
ROOT_FILES = ("cells.json", "cells_full.json", "endpoint.snapshot.json")


class LadderMergeError(SystemExit):
    pass


def read_jsonl(p: Path) -> List[dict]:
    with p.open() as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p: Path, rows: List[dict]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def key(r: dict) -> Tuple[str, int]:
    return (str(r["item_id"]), int(r["sample_idx"]))


def merge_cell(src_dir: Path, rerun_dir: Path | None, out_dir: Path, *, min_zlib: float,
               allow_missing: bool) -> Dict[str, Dict[str, int]]:
    """Returns per-behavior counts: n, at_cap_src, degenerate, selected, rerun_present, unrerun,
    at_cap_after (rerun rows still at cap), at_cap_after_all (all rows at cap in the merged tree)."""
    src_j = src_dir / "judged.jsonl"
    if not src_j.is_file():
        raise LadderMergeError(f"{src_dir}: no judged.jsonl — judge the first rung before merging")
    src_rows = read_jsonl(src_j)
    src_meta = json.loads((src_dir / "meta.json").read_text())
    src_cap = int(src_meta["params"]["max_tokens"])
    selected: Set[Tuple[str, int]] = set()
    degenerate: Set[Tuple[str, int]] = set()
    for r in src_rows:
        if r.get("finish_reason") == "length":
            (degenerate if zlib_ratio(r.get("response") or "") < min_zlib else selected).add(key(r))
    rerun_rows: Dict[Tuple[str, int], dict] = {}
    rerun_cap = None
    if rerun_dir is not None and (rerun_dir / "judged.jsonl").is_file():
        rerun_meta = json.loads((rerun_dir / "meta.json").read_text())
        rerun_cap = int(rerun_meta["params"]["max_tokens"])
        for r in read_jsonl(rerun_dir / "judged.jsonl"):
            k = key(r)
            if k not in selected:
                raise LadderMergeError(f"{rerun_dir}: rerun row {k} was not a selected (truncated, non-loop) "
                                       "pair of the source — the rung does not match this source")
            rerun_rows[k] = r
    elif rerun_dir is not None and (rerun_dir / "responses.jsonl").is_file():
        raise LadderMergeError(f"{rerun_dir}: responses present but no judged.jsonl — judge the rung first")
    missing = sorted(selected - set(rerun_rows))
    if missing and not allow_missing:
        raise LadderMergeError(f"{src_dir.name}: {len(missing)} selected pair(s) have no judged rerun row "
                               f"(first {missing[:3]}); rerun the rung or pass --allow-missing")
    merged: List[dict] = []
    counts: Dict[str, Counter] = defaultdict(Counter)
    for r in src_rows:
        beh = r.get("behavior") or "betley"
        k = key(r)
        c = counts[beh]
        c["n"] += 1
        if r.get("finish_reason") == "length":
            c["at_cap_src"] += 1
        if k in degenerate:
            c["degenerate"] += 1
        if k in selected:
            c["selected"] += 1
            if k in rerun_rows:
                rr = {**rerun_rows[k], "ladder_rung": rerun_cap}
                merged.append(rr)
                c["rerun_present"] += 1
                if rr.get("finish_reason") == "length":
                    c["at_cap_after"] += 1
                continue
            c["unrerun"] += 1
        merged.append({**r, "ladder_rung": src_cap})
    for beh, c in counts.items():
        c["at_cap_after_all"] = sum(1 for r in merged if (r.get("behavior") or "betley") == beh
                                    and r.get("finish_reason") == "length")
    write_jsonl(out_dir / "judged.jsonl", merged)
    write_jsonl(out_dir / "responses.jsonl", [{k: v for k, v in r.items()
                                               if k in ("family", "cell_id", "virtual_model", "item_id", "sample_idx",
                                                        "behavior", "prompt_sha", "response", "finish_reason", "usage",
                                                        "model_reported", "ladder_rung")} for r in merged])
    meta = {**src_meta, "ladder": {"src_dir": str(src_dir), "rerun_dir": str(rerun_dir) if rerun_dir else None,
                                   "src_cap": src_cap, "rerun_cap": rerun_cap, "min_zlib": min_zlib,
                                   "selected": len(selected), "degenerate": len(degenerate),
                                   "rerun_present": len(rerun_rows), "unrerun": len(missing)}}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    return {b: dict(c) for b, c in counts.items()}


def laddered_cells(root: Path) -> List[Path]:
    """The cell directories of the laddered families under a judged root (``<family>/<cell>/``)."""
    return [c for fam in LADDERED_FAMILIES if (root / fam).is_dir() for c in sorted((root / fam).iterdir()) if c.is_dir()]


def check_roots(src_root: Path, rerun_root: Path, *, allow_missing: bool) -> None:
    """Refuse, naming the root, before anything is written: a missing or non-directory root, a src root with no laddered
    cell (the merge would write an empty report), and a rung with none unless --allow-missing."""
    for flag, root, what in (("--src-root", src_root, "the first rung's judged root"),
                             ("--rerun-root", rerun_root, "the ladder rung's judged root")):
        if not root.is_dir():
            raise LadderMergeError(f"{flag} {root}: not a directory ({what}); nothing was written")
    fams = "/".join(LADDERED_FAMILIES)
    if not laddered_cells(src_root):
        raise LadderMergeError(f"--src-root {src_root}: no laddered family cell ({fams}/<cell>/) to merge; "
                               "nothing was written")
    if not laddered_cells(rerun_root) and not allow_missing:
        raise LadderMergeError(f"--rerun-root {rerun_root}: no laddered family cell ({fams}/<cell>/): the rung is empty; "
                               "rerun it, or pass --allow-missing to let every src row stand; nothing was written")


def merge_root(src_root: Path, rerun_root: Path, out_root: Path, *, min_zlib: float,
               allow_missing: bool) -> Dict[str, Dict[str, Dict[str, int]]]:
    if out_root.resolve() in (src_root.resolve(), rerun_root.resolve()):
        raise LadderMergeError("--out-root must be a NEW root, never the source or the rung")
    check_roots(src_root, rerun_root, allow_missing=allow_missing)
    out_root.mkdir(parents=True, exist_ok=True)
    for name in ROOT_FILES:
        if (src_root / name).exists():
            shutil.copy2(src_root / name, out_root / name)
    if (src_root / "serving").is_dir() and not (out_root / "serving").exists():
        shutil.copytree(src_root / "serving", out_root / "serving")
    report: Dict[str, Dict[str, Dict[str, int]]] = {}
    for fam_dir in sorted(p for p in src_root.iterdir() if p.is_dir() and p.name not in ("serving", "store", "analysis")):
        fam = fam_dir.name
        if fam in LADDERED_FAMILIES:
            for cell_dir in sorted(p for p in fam_dir.iterdir() if p.is_dir()):
                rerun_dir = rerun_root / fam / cell_dir.name
                report[f"{fam}/{cell_dir.name}"] = merge_cell(
                    cell_dir, rerun_dir if rerun_dir.is_dir() else None, out_root / fam / cell_dir.name,
                    min_zlib=min_zlib, allow_missing=allow_missing)
            for extra in fam_dir.glob("*.json"):
                shutil.copy2(extra, out_root / fam / extra.name)
        else:
            if (out_root / fam).exists():
                shutil.rmtree(out_root / fam)
            shutil.copytree(fam_dir, out_root / fam)
    return report


def print_report(report: Dict[str, Dict[str, Dict[str, int]]]) -> None:
    print(f"{'cell/behavior':75s} {'n':>5s} {'cap@src':>7s} {'loop':>5s} {'sel':>5s} {'rerun':>5s} "
          f"{'unrerun':>7s} {'cap@rung':>8s} {'residual%':>9s}")
    for cell, behs in report.items():
        for beh, c in sorted(behs.items()):
            n = c.get("n", 0)
            print(f"{cell + ' ' + beh:75s} {n:5d} {c.get('at_cap_src', 0):7d} {c.get('degenerate', 0):5d} "
                  f"{c.get('selected', 0):5d} {c.get('rerun_present', 0):5d} {c.get('unrerun', 0):7d} "
                  f"{c.get('at_cap_after', 0):8d} {100 * c.get('at_cap_after_all', 0) / max(1, n):9.1f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--src-root", type=Path, required=True)
    ap.add_argument("--rerun-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--target", default=None)
    ap.add_argument("--min-zlib", type=float, default=RERUN_MIN_ZLIB_RATIO)
    ap.add_argument("--allow-missing", action="store_true")
    ap.add_argument("--rows-analyze", action="store_true", help="file rows and analyse the merged root")
    args = ap.parse_args(argv)
    report = merge_root(args.src_root, args.rerun_root, args.out_root, min_zlib=args.min_zlib,
                        allow_missing=args.allow_missing)
    print_report(report)
    (args.out_root / "ladder_report.json").write_text(json.dumps(report, indent=2))
    if args.rows_analyze:
        target = args.target or args.src_root.name
        from sisteer.capeval.misalign import analyze as A, rows as Rw
        rc = Rw.main(["--judged-root", str(args.out_root), "--target", target])
        if rc:
            return rc
        return A.main(["--judged", str(args.out_root), "--target", target,
                       "--out", str(args.out_root / "analysis")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
