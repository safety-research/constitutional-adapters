#!/usr/bin/env python3
"""The StrongREJECT re-judge's goal table, ``goals.json``: every goal's (dataset, revision, text) and the goal sets, built at run time from
the pinned benchmark revisions (the goal texts are harmful requests and are never committed to the repository).

    PYTHONPATH=src:scripts python scripts/rejudge/strongreject/goals.py --out SCOPE/goals.json \\
        [--harmbench-parquet FILE] [--jbb-csv FILE]

The goals are the attack harness's confirmatory set (``scripts/jailbreak_run.py`` ``_eval_goals(cfg, confirmatory=True)``, 259 goals),
read with the same loaders and settings every attack config carries:

  HarmBench   ``load_harmbench(split="test", revision=4aa1f46a..., functional="standard", exclude_copyright=True)``: 159 goals
  JBB         ``stratify_jbb(load_jbb("harmful", revision=886acc35...), n=100, seed=0)``: 100 goals

Each goal is ``{dataset, revision, text}`` keyed by its id; ``goal_sets`` names which goals each attack category judges: ``static``
(every goal: the four single-turn attacks), ``mt`` (every goal: the three multi-turn attacks) and ``bon`` (the 100 JBB goals: best-of-N).
The request key (``request.cache_key``) and the judge prompt use this ORIGINAL goal text, never the attack prompt. The file is written as
``json.dump(..., indent=0, sort_keys=True)``, the bytes the paper's scope used, so its sha256 is comparable (printed).

The texts come from the Hugging Face cache (``huggingface_hub.hf_hub_download`` at the pinned revisions; ``HF_HUB_OFFLINE=1`` reads the
cache only). ``--harmbench-parquet`` / ``--jbb-csv`` read local copies of the same files instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, Optional

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
for _p in (os.path.join(_REPO, "src"), os.path.join(_REPO, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sisteer.eval import jailbreak_data as JD  # noqa: E402

#: the attack configs' goal settings (``datasets.harmbench`` / ``datasets.jbb`` / ``prereg.confirmatory_goals`` of every
#: ``configs/*.yaml`` that names them: one value each)
HARMBENCH = {"split": "test", "revision": JD.HARMBENCH_REVISION, "functional": "standard", "exclude_copyright": True}
JBB = {"kind": "harmful", "revision": JD.JBB_REVISION, "n": 100, "seed": 0}
EXPECTED = {"harmbench": 159, "jbb": 100}


def build(harmbench_path: Optional[str] = None, jbb_path: Optional[str] = None) -> Dict[str, Any]:
    """``{goals, goal_sets}`` (module docstring). Refuses (SystemExit) a goal count other than the harness's, a duplicate id, or a
    revision other than the pinned one."""
    hb = JD.load_harmbench(split=HARMBENCH["split"], revision=HARMBENCH["revision"], path=harmbench_path,
                           functional=HARMBENCH["functional"], exclude_copyright=HARMBENCH["exclude_copyright"])
    jbb = JD.stratify_jbb(JD.load_jbb(JBB["kind"], revision=JBB["revision"], path=jbb_path), n=JBB["n"], seed=JBB["seed"])
    if (len(hb), len(jbb)) != (EXPECTED["harmbench"], EXPECTED["jbb"]):
        raise SystemExit(f"REFUSED: {len(hb)} HarmBench and {len(jbb)} JBB goals, not the harness's {EXPECTED['harmbench']} and "
                         f"{EXPECTED['jbb']} (another file than the pinned revision's?)")
    goals: Dict[str, Dict[str, str]] = {}
    for g in hb + jbb:
        if g.goal_id in goals:
            raise SystemExit(f"REFUSED: goal id {g.goal_id} appears twice")
        if g.revision not in (HARMBENCH["revision"], JBB["revision"]):
            raise SystemExit(f"REFUSED: goal {g.goal_id} carries revision {g.revision}")
        goals[g.goal_id] = {"dataset": g.dataset, "revision": g.revision, "text": g.text}
    return {"goals": goals, "goal_sets": {"static": sorted(goals), "mt": sorted(goals), "bon": sorted(g.goal_id for g in jbb)}}


def write(table: Dict[str, Any], out: str) -> str:
    """Write ``goals.json`` (``indent=0, sort_keys=True``, no trailing newline) and return its sha256."""
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(table, fh, indent=0, sort_keys=True)
    with open(out, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the goals.json to write")
    ap.add_argument("--harmbench-parquet", help="a local copy of the pinned HarmBench test split (default: the HF cache)")
    ap.add_argument("--jbb-csv", help="a local copy of the pinned JBB harmful behaviours (default: the HF cache)")
    args = ap.parse_args(argv)
    table = build(args.harmbench_parquet, args.jbb_csv)
    sha = write(table, args.out)
    print(json.dumps({"out": os.path.abspath(args.out), "sha256": sha, "goals": len(table["goals"]),
                      "goal_sets": {k: len(v) for k, v in table["goal_sets"].items()}}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
