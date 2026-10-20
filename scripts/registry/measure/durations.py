#!/usr/bin/env python3
"""durations.py JUNIT_XML [--top N] [--json] — per-test wall time from a pytest JUnit report: the ``time`` attribute of every
``<testcase>``, longest first, with the total, per-file subtotals and each test's outcome. S7's record of the landing chain's
live-DB step: land.sh runs ``pytest -m live_db … --junitxml=$STATE/land/<branch>.junit.xml`` and renders this into
``$STATE/land/<branch>.durations.txt`` (the homework found no per-test evidence at all: ``-q`` on ``addopts="-q"`` and
``-p no:cacheprovider`` left every pytest.log ending at the warnings summary). Stdlib only; safe on a partial report.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def _node_of(tc: ET.Element) -> str:
    """``tests.test_x.TestK`` + ``test_s`` -> ``tests/test_x.py::TestK::test_s``: the module is the classname up to its last ``test_*``
    component (pytest names test modules so), the rest is the class chain."""
    name = tc.get("name") or ""
    parts = (tc.get("classname") or "").split(".") if tc.get("classname") else []
    k = max((i for i, p in enumerate(parts) if p.startswith("test_")), default=len(parts) - 1) + 1
    file = "/".join(parts[:k]) + ".py" if parts else ""
    return "::".join(p for p in [file, *parts[k:], name] if p)


def _outcome(tc: ET.Element) -> str:
    for tag, label in (("failure", "failed"), ("error", "error"), ("skipped", "skipped")):
        if tc.find(tag) is not None:
            return label
    return "passed"


def parse(path: str | Path) -> dict:
    tree = ET.parse(str(path))
    root = tree.getroot()
    cases = []
    for tc in root.iter("testcase"):
        try:
            t = float(tc.get("time") or 0.0)
        except ValueError:
            t = 0.0
        node = _node_of(tc)
        cases.append({"node": node, "file": node.split("::")[0], "time_s": round(t, 3), "outcome": _outcome(tc)})
    cases.sort(key=lambda c: -c["time_s"])
    files: dict = {}
    for c in cases:
        f = files.setdefault(c["file"], {"time_s": 0.0, "n": 0, "outcomes": {}})
        f["time_s"] = round(f["time_s"] + c["time_s"], 3)
        f["n"] += 1
        f["outcomes"][c["outcome"]] = f["outcomes"].get(c["outcome"], 0) + 1
    counts = dict(Counter(c["outcome"] for c in cases))
    suite_time = None
    ts = root if root.tag == "testsuite" else root.find("testsuite")
    if ts is not None and ts.get("time"):
        try:
            suite_time = float(ts.get("time"))
        except ValueError:
            suite_time = None
    return {"path": str(path), "n": len(cases), "total_s": round(sum(c["time_s"] for c in cases), 3), "suite_time_s": suite_time,
            "counts": counts, "cases": cases, "files": files, "rendered_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def render(rep: dict, top: int | None = None) -> str:
    lines = [f"# per-test durations from {rep['path']}: {rep['n']} tests, total {rep['total_s']:.1f} s"
             + (f" (suite wall {rep['suite_time_s']:.1f} s)" if rep.get("suite_time_s") is not None else "")
             + "; " + ", ".join(f"{k} {v}" for k, v in sorted(rep["counts"].items())) + f"; rendered {rep['rendered_utc']}"]
    for c in rep["cases"][: top or None]:
        lines.append(f"{c['time_s']:10.2f} s  {c['node']}  [{c['outcome']}]")
    if top and rep["n"] > top:
        lines.append(f"           ... {rep['n'] - top} more")
    lines.append("# per file")
    for f, d in sorted(rep["files"].items(), key=lambda kv: -kv[1]["time_s"]):
        lines.append(f"{d['time_s']:10.2f} s  {f} ({d['n']} tests: " + ", ".join(f"{k} {v}" for k, v in sorted(d["outcomes"].items())) + ")")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("junit")
    ap.add_argument("--top", type=int, default=None, help="list only the N longest tests (default: all)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if not Path(a.junit).is_file():
        print(f"no JUnit report at {a.junit}", file=sys.stderr)
        return 2
    rep = parse(a.junit)
    if a.json:
        print(json.dumps(rep, indent=1))
    else:
        sys.stdout.write(render(rep, a.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
