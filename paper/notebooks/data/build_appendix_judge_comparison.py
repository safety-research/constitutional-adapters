#!/usr/bin/env python
"""Build `appendix_judge_comparison.json`, the frozen readings behind the judge-robustness figure of appendix D (`fig:app_judges`,
figA_judge_comparison_a and _b) and the numbers the paragraph beside it quotes.

WHAT IT HOLDS. Readings only, no item rows, no paths:
  marks           every mark the figure draws for Qwen3.5-4B, under the Claude judges (`original`, REGISTRY_NB_JUDGES=primary) and the
                  non-Claude judges (`new`, REGISTRY_NB_JUDGES=rejudge): per judge, category (static, multi-query, multi-turn,
                  over-refusal, Petri), eval, object and seed, the change from the unsteered model oriented as the main figures orient
                  it (up is better) with its 95% interval, whether its anchor came from outside the paper frame (`open`), and per mean
                  mark whether it averages fewer evals than the category names (`partial`); the eval's judge model
  static_anchors  per judge and static attack, the undefended model's attack success the static panel's rows are differenced against
  mean_columns    per judge, model, judged category and object, the mean mark of the all-model figures (figA_jailbreaks_models,
                  figA_misalignment_models, figA_capabilities_models), from which the paragraph's rank correlations are computed

WHERE THE NUMBERS COME FROM. The notebook's own load chain, run against a frozen copy of the registry once per judging policy, each in a
subprocess of this script: the generator's header and load cells (`583e3f04`, `ec6aacd8`) as `make_registry_notebook.py` writes them at
the pinned code commit (`CODE_COMMIT`, exported with `git archive`), then the statements of Figure 3's and the all-model figures' cells
that build the deltas and the category means (`FRAME_STATEMENTS`). Under `rejudge` the strength lanes' XSTest readings come from the
commit's committed projection of their reads sidecars, held to the two sidecars the delivered figure's render read
(`STRENGTH_READS_SHA256`). The reductions are the figure's: a per-eval mark is `registry_nb.pool_draws` over the frame's rows of the
cell, its value and interval the pooled `d_y` / `d_lo` / `d_hi` times the category's sign; a mean mark is `registry_nb.pool_seeds`
over the category means (`cm` for Qwen, `cm_all` for the all-model columns); a static anchor is the mean `anchor_value` of the static
rows of one attack. The registry is opened read-only and nothing is written to it.

PINNED TO HEAD 537565 (`AS_OF`), the head the paper's final figures were drawn at. The frozen copy may hold records ingested after it;
the build reads them and refuses unless every one is either bookkeeping the notebook never reads (`BOOKKEEPING_TABLES`) or a row of a
run (a judgment, its item rows, the run row it updated) that the paper's selection excludes and that no reading draws on, so the
readings are those at the pinned head. The copy's sha256, its head, the rows after the pinned head and the runs they belong to are
recorded.

Recorded in `provenance` besides: the builder and its sha256, the code commit and the sha256 of the `registry_nb.py` the frames were
built with, each policy's judge ids per category, the selection (how many runs and anchor runs the readings draw on per judge, and the
sha256 of their sorted ids), the strength-read projection and the sidecars it projects, by sha256, the numbers the paragraph quotes as
this file yields them, and, when `--compare-values` names the value files written beside the figure as delivered, the sha256 of each
and how many of its values the marks equal exactly. `payload_sha256` covers the three blocks; the notebook's reader refuses a file
whose blocks do not hash to it.

usage, from the repository root (the notebook environment; the registry copy is read-only):
  PYTHONPATH=$PWD/src:$PWD/scripts .venv/bin/python paper/notebooks/data/build_appendix_judge_comparison.py --registry DIR \
      [--compare-values A.json B.json] [--check] [--out FILE] [--keep DIR]
--registry is the directory holding the frozen copy as registry.sqlite (with its config.toml); --check rebuilds and compares with the
committed file, bookkeeping aside; --keep keeps each policy's reduced readings in DIR. A build takes about half an hour (the two
policies' load chains run at once).

A REBUILD NEEDS what the published repository does not carry: the development history that holds CODE_COMMIT (`git archive` reads
it) and the frozen registry copy. Without them the committed file is verified by what it records: the reader refuses readings that
do not hash to `payload_sha256`, and the tests pin that hash, the copy's sha256 and the delivered value files' sha256s, redraw both
parts against the delivered PNGs' pixels and recompute the quoted numbers.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import re
import sqlite3
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
NOTEBOOKS = HERE.parent
OUT = HERE / "appendix_judge_comparison.json"
SCRIPT = "paper/notebooks/data/build_appendix_judge_comparison.py"
SCHEMA = "appendix_judge_comparison/1"
AS_OF = 537565
#: the code whose load chain builds the frames: the commit this figure was added on, the notebook's default re-judge policy with the
#: strength-lane reads committed as a projection. At the pinned head its frames give, value for value, every reading of the delivered
#: figure, which was drawn from two renders of the code the paper's final figures were rendered with (the judge switch on the
#: decode-only and table-quantity change), which --compare-values checks against the value files written beside it. The commit is in
#: the development history only (the published repository is one squashed commit), and a pin tag there keeps it reachable
CODE_COMMIT = "d347b7439ad77d5c68ea2101c53ee2e35aafe32b"
POLICIES = {"original": "primary", "new": "rejudge"}
#: the strength-lane XSTest reads sidecars the delivered figure's re-judge render read; at CODE_COMMIT the rejudge policy reads their
#: committed projection (data/strength_reads/), whose manifests must vouch for exactly these
STRENGTH_READS_SHA256 = ("d234d671a399f2f59abeddfec54ee1ca33ac36bf0540d370cae5bd8638ff3cf2",
                         "3ddf1eeedf67ce367f78f9835a4e887af28f42043851d8c36d2f4a7134460a96")
CELLS = ("583e3f04", "ec6aacd8")
FRAME_STATEMENTS = ("d = R.anchored(op, pool=POOL_Q)",
                    "CAT_Q = R.categories_for(R.MODEL_ID)",
                    "cm = R.category_means(d, categories=CAT_Q)",
                    "d_all = R.anchored(op_all, pool=POOL_Q, setup='any')",
                    "cm_all = R.category_means(d_all, categories=R.APPENDIX_CATEGORIES)")
JUDGED = ("static", "multi-query", "multi-turn", "over-refusal", "Petri")
#: tables whose rows after the pinned head cannot reach a reading: the ledger of record files and its bookkeeping
BOOKKEEPING_TABLES = ("record_ingest", "file_arrival", "mirror_ack", "artifact", "record_delivery", "record_run_index")
MARK_COLUMNS = ["judge", "category", "eval", "object", "seed", "y", "lo", "hi", "open", "partial", "n", "k", "judge_model"]
ANCHOR_COLUMNS = ["judge", "eval", "asr"]
MEAN_COLUMNS = ["judge", "model", "category", "object", "y"]
VOLATILE = ("generated_utc", "git_commit", "git_dirty", "script_sha256", "seconds")


def refuse(msg: str):
    raise SystemExit(f"build_appendix_judge_comparison: REFUSED: {msg}")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha256_lines(items) -> str:
    return hashlib.sha256("\n".join(sorted(items)).encode()).hexdigest()


def git(*args, cwd=REPO) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


# ---- one policy's readings (runs inside the exported code tree) ------------------------------------------------------------------
def _num(x):
    import math
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def frames_main(policy: str, out: str) -> int:
    """The subprocess: execute the generator's header and load cells and FRAME_STATEMENTS under `policy` in the current directory (the
    exported tree's paper/notebooks), reduce the frames to this file's readings and write them, with the selection, to `out` (JSON)."""
    t0 = time.time()
    sys.path.insert(0, os.getcwd())
    src = Path("make_registry_notebook.py").read_text(encoding="utf-8")
    ns = {"__name__": "__judge_comparison_frames__"}
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        for cid in CELLS:
            m = re.search(r'^(?:md|code)\("%s", r("""|\'\'\')(.*?)\1\)' % cid, src, flags=re.M | re.S)
            if not m:
                refuse(f"the generator at the code commit has no cell {cid}")
            exec(m.group(2).strip("\n"), ns)
        exec("\n".join(FRAME_STATEMENTS), ns)
    R = ns["R"]
    import numpy as np
    import pandas as pd
    if R.JUDGES != policy or R.SOURCE != "live":
        refuse(f"the frames read policy {R.JUDGES!r} from {R.SOURCE!r}, expected {policy!r} from the registry copy")
    CAT = R.categories_for(R.MODEL_ID)
    objects = list(R.OBJECTS)
    d, cm, d_all, cm_all = (ns[k] for k in ("d", "cm", "d_all", "cm_all"))
    marks, used, judge_ids = [], [], {}
    for cat in JUDGED:
        evals, sign, _ = CAT[cat]
        f = d[(d.model_id == R.MODEL_ID) & d.eval_id.isin(evals) & d.d_y.notna()]
        used.append(f)
        judge_ids[cat] = sorted({str(x) for x in f.judge_id.dropna()})
        p = R.pool_draws(f) if len(f) else f
        p = p[p.object_label.isin(objects)]
        for _, r in p.iterrows():
            lo, hi = sorted((sign * float(r.d_lo), sign * float(r.d_hi))) if pd.notna(r.d_lo) and pd.notna(r.d_hi) else (None, None)
            marks.append([None, cat, str(r.eval_id), str(r.object_label), "" if r.object_label in R.DECODE_ONLY else str(r.seed),
                          sign * float(r.d_y), lo, hi, not bool(r.anchor_in_frame), False, int(r.get("n_draws", 1)), None, str(r.judge_model)])
        mp = R.pool_seeds(cm[(cm.model_id == R.MODEL_ID) & (cm.category == cat)])
        mp = mp[mp.object_label.isin(objects)]
        jm = ", ".join(sorted({str(x) for x in f.judge_model.dropna()}))
        for r in mp.itertuples():
            marks.append([None, cat, R.MEAN_COLUMN, str(r.object_label), str(r.seeds or ""), float(r.y), _num(r.lo), _num(r.hi), bool(r.open),
                          bool(r.partial), int(r.n_seeds), int(r.k), jm])
    st = d[d.eval_id.isin(CAT["static"][0]) & d.anchor_value.notna() & (d.model_id == R.MODEL_ID)]
    anchors = [[None, str(e), float(v)] for e, v in st.groupby("eval_id").anchor_value.mean().items()]
    columns, used_all = [], []
    for cat in JUDGED:
        x = cm_all[cm_all.category.eq(cat) & cm_all.object_label.ne("none")]
        for r in R.pool_seeds(x).itertuples():
            columns.append([None, str(r.model_id), cat, str(r.object_label), float(r.y)])
        evals = R.APPENDIX_CATEGORIES[cat][0] if cat in R.APPENDIX_CATEGORIES else CAT[cat][0]
        used_all.append(d_all[d_all.eval_id.isin(list(evals) + ([R.POOLED_EVAL[cat]] if cat in R.POOLED_EVAL else []))
                              & d_all.d_y.notna() & d_all.object_label.ne("none")])

    def anchor_ids(frame) -> set:
        out = set()
        for v in (frame["anchor_run"].dropna().astype(str) if "anchor_run" in frame.columns else []):
            out |= {x for x in re.split(r"[;,|]\s*", v) if x}                  # a pooled anchor names its runs comma-joined
        return out

    # the slices are never concatenated: the frames carry DataFrames in .attrs, which pd.concat cannot compare
    run_ids = sorted(set().union(*({str(x) for x in f.run_id} for f in used + used_all)))
    anchor = sorted(set().union(*(anchor_ids(f) for f in used + used_all)))
    rep = R.JUDGE_REPORT.get("strength_reads_files") if isinstance(R.JUDGE_REPORT, dict) else None
    # each strength-read file the frames read, by its manifest's table digest
    reads = sorted({str(f.get("table_sha256") or f.get("sha256")) for f in rep} if isinstance(rep, list) else set())
    res = dict(policy=policy, head=int(ns["HEAD"]), registry_nb_sha256=sha256(Path(R.__file__)), marks=marks, static_anchors=anchors,
               mean_columns=columns, judge_ids=judge_ids, run_ids=run_ids, anchor_run_ids=anchor, strength_reads_sha256=reads,
               seconds=round(time.time() - t0))
    Path(out).write_text(json.dumps(res, allow_nan=False))
    return 0


# ---- the registry copy ---------------------------------------------------------------------------------------------------------------
def registry_facts(db: Path) -> dict:
    """The copy's head and every row ingested after AS_OF, by table, with the runs they belong to (read-only)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        head = int(con.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0])
        if not con.execute("SELECT 1 FROM record_ingest WHERE ingest_seq = ?", (AS_OF,)).fetchone():
            refuse(f"the registry copy holds no record file at the pinned head {AS_OF}")
        tables = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")]
        after, scoped, runs = {}, [], set()
        for t in tables:
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({t})")]
            if "ingest_seq" not in cols:
                continue
            if "run_id" in cols and t not in BOOKKEEPING_TABLES:          # one pass: the count and the runs together
                by_run = dict(con.execute(f"SELECT run_id, COUNT(*) FROM {t} WHERE ingest_seq > ? GROUP BY run_id", (AS_OF,)).fetchall())
                n = sum(by_run.values())
            else:
                by_run, n = {}, con.execute(f"SELECT COUNT(*) FROM {t} WHERE ingest_seq > ?", (AS_OF,)).fetchone()[0]
            if not n:
                continue
            after[t] = int(n)
            if t in BOOKKEEPING_TABLES:
                continue
            if "run_id" not in cols:
                refuse(f"{n} rows of {t} were ingested after the pinned head {AS_OF}; only bookkeeping and rows of runs outside the "
                       f"readings are admitted")
            scoped.append(t)                                           # a judgment, its items, the run row it updated
            runs |= set(by_run)
        # the selection state of every such run, in ONE read of the view (it is costly to evaluate, so never once per run)
        states = {}
        if runs:
            q = f"SELECT run_id, selection_state FROM v_paper_current WHERE run_id IN ({','.join('?' * len(runs))})"
            for rid, st in con.execute(q, sorted(runs)).fetchall():
                states.setdefault(rid, set()).add(st)
        bad = sorted(rid for rid in runs if states.get(rid) != {"excluded"})
        if bad:
            refuse(f"runs with rows after the pinned head that the paper's selection does not exclude: {bad[:5]}")
    finally:
        con.close()
    return dict(head=head, after=after, run_scoped=sorted(scoped), after_runs=sorted(runs))


def export_code(commit: str, dest: Path) -> Path:
    """`git archive` of the code commit's notebooks, src and scripts into dest; returns its paper/notebooks."""
    dest.mkdir(parents=True, exist_ok=True)
    arch = subprocess.run(["git", "archive", commit, "paper/notebooks", "src", "scripts"], cwd=REPO, capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=arch.stdout, check=True)
    return dest / "paper" / "notebooks"


def strength_projection(nb: Path) -> list:
    """The exported tree's committed strength-read projections: [(its table sha256, the sidecar it projects)], refused unless they
    project exactly the sidecars the delivered figure's re-judge render read."""
    out = []
    for m in sorted((nb / "data" / "strength_reads").glob("*/manifest.json")):
        j = json.loads(m.read_text())
        out.append((str(j.get("table_sha256")), str((j.get("source") or {}).get("table_sha256"))))
    if sorted(src for _, src in out) != sorted(STRENGTH_READS_SHA256):
        refuse(f"the code commit's strength-read projections vouch for {[src[:12] for _, src in out]}, not the render's "
               f"{[x[:12] for x in STRENGTH_READS_SHA256]}")
    return out


def run_policy(side: str, nb: Path, code_root: Path, regdir: Path, work: Path) -> dict:
    out = work / f"{side}.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("REGISTRY_NB_", "NB_", "ANTHROPIC_"))}
    env.update(REGISTRY_NB_SOURCE="live", REGISTRY_ROOT=str(regdir), NB_CACHE="0", MPLBACKEND="Agg", REGISTRY_NB_JUDGES=POLICIES[side],
               REGISTRY_NB_FIGS=str(work / f"figures-{side}"), PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([str(code_root / "src"), str(code_root / "scripts")]))
    p = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--frames", POLICIES[side], str(out)], cwd=nb, env=env,
                       capture_output=True, text=True)
    (work / f"{side}.log").write_text(p.stdout + p.stderr)
    if p.returncode != 0:
        refuse(f"the {POLICIES[side]} frames failed (exit {p.returncode}): {(p.stdout + p.stderr).strip().splitlines()[-1:]}")
    return json.loads(out.read_text())


# ---- the file ------------------------------------------------------------------------------------------------------------------------
def _module():
    os.environ.setdefault("REGISTRY_NB_SOURCE", "snapshot")
    os.environ.setdefault("NB_CACHE", "0")
    os.environ.setdefault("MPLBACKEND", "Agg")
    if str(NOTEBOOKS) not in sys.path:
        sys.path.insert(0, str(NOTEBOOKS))
    import appendix_judge_comparison as JC
    return JC


def block(columns, rows) -> dict:
    return {"columns": list(columns), "rows": rows}


def compare_values(marks: list, files: list) -> dict:
    """The delivered value files against the marks: each value is (judge, category, eval, object, seed) -> (delta, lo, hi), exact."""
    mine = {(r[0], r[1], r[2], r[3], r[4] or None): (r[5], r[6], r[7]) for r in marks}
    out, equal, total = [], 0, 0
    for f in files:
        v = json.loads(Path(f).read_text())["values"]
        eq = sum(mine.get((x["judge"], x["category"], x["eval"], x["object"], x["seed"] or None)) == (x["delta"], x["lo"], x["hi"]) for x in v)
        out.append(dict(sha256=sha256(Path(f)), n_values=len(v), equal=int(eq)))
        equal, total = equal + eq, total + len(v)
    if equal != total or total != len(marks):
        refuse(f"the marks equal {equal} of the {total} delivered values ({len(marks)} marks)")
    return dict(files=out, equal=int(equal))


def build(regdir: Path, compare: list, keep: Path | None) -> dict:
    t0 = time.time()
    db = regdir / "registry.sqlite"
    if not db.is_file():
        refuse(f"--registry names no directory holding registry.sqlite")
    with tempfile.TemporaryDirectory(prefix="judge-comparison-") as tmp:
        work = Path(keep) if keep else Path(tmp)
        work.mkdir(parents=True, exist_ok=True)
        code_root = Path(tmp) / "code"
        nb = export_code(CODE_COMMIT, code_root)
        projection = strength_projection(nb)
        # the two policies, the scan for rows after the pinned head and the copy's hash are independent: all four at once
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {side: ex.submit(run_policy, side, nb, code_root, regdir, work) for side in POLICIES}
            f_facts, f_sha = ex.submit(registry_facts, db), ex.submit(sha256, db)
            res = {side: f.result() for side, f in futs.items()}
            facts, copy_sha = f_facts.result(), f_sha.result()
    for side, r in res.items():
        if r["policy"] != POLICIES[side] or r["head"] != facts["head"]:
            refuse(f"{side}: policy {r['policy']}, head {r['head']}; expected {POLICIES[side]} at the copy's head {facts['head']}")
    drawn = set().union(*(set(r["run_ids"]) | set(r["anchor_run_ids"]) for r in res.values()))
    if drawn & set(facts["after_runs"]):
        refuse(f"runs judged after the pinned head enter the readings: {sorted(drawn & set(facts['after_runs']))[:5]}")
    if sorted(res["new"]["strength_reads_sha256"]) != sorted(t for t, _ in projection):
        refuse(f"the rejudge frames read the strength-lane reads {res['new']['strength_reads_sha256']}, not the committed projection")
    rows = {k: [] for k in ("marks", "static_anchors", "mean_columns")}
    for side in POLICIES:
        for k in rows:
            rows[k] += [[side] + r[1:] for r in res[side][k]]
    order = {c: i for i, c in enumerate(JUDGED)}
    rows["marks"].sort(key=lambda r: (order[r[1]], r[2] == "mean", r[2], r[3], r[4], r[0]))
    rows["static_anchors"].sort(key=lambda r: (r[1], r[0]))
    rows["mean_columns"].sort(key=lambda r: (r[1], order[r[2]], r[3], r[0]))
    JC = _module()
    rnb = {side: res[side]["registry_nb_sha256"] for side in POLICIES}
    out = {
        "schema": SCHEMA,
        "about": ("The judge-robustness comparison of appendix D for Qwen3.5-4B: every mark of figA_judge_comparison_a and _b under the "
                  "Claude judges the registry filed first (claude-haiku-4-5 for StrongREJECT, claude-sonnet-5 for over-refusal, claude-opus-5 "
                  "for Petri) and under the non-Claude judges the paper reads (minimax-m3; glm-5.3 for Petri), as changes from the unsteered "
                  "model oriented up-is-better with 95% intervals; the undefended model's static attack success under each; and the all-model "
                  "figures' mean marks under each, from which the rank correlations are computed. Frozen at registry head 537565."),
        "judges": {side: {"policy": POLICIES[side], "judge_ids": res[side]["judge_ids"]} for side in POLICIES},
        "provenance": {
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "script": SCRIPT, "script_sha256": sha256(Path(__file__).resolve()),
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--", SCRIPT)),
            "registry": {"opened": "read-only", "copy_sha256": copy_sha, "head_in_copy": facts["head"], "as_of": AS_OF,
                         "head_read_by_frames": {side: res[side]["head"] for side in POLICIES},
                         "rows_after_as_of": facts["after"], "run_scoped_tables": facts["run_scoped"],
                         "runs_after_as_of": {"n": len(facts["after_runs"]), "selection_state": "excluded",
                                              "run_ids_sha256": sha256_lines(facts["after_runs"]), "in_readings": 0}},
            "frames": {"code_commit": CODE_COMMIT, "registry_nb_sha256": rnb["original"] if rnb["original"] == rnb["new"] else rnb,
                       "cells": list(CELLS), "statements": list(FRAME_STATEMENTS), "policies": dict(POLICIES),
                       "strength_reads": {"projection_sha256": sorted(t for t, _ in projection),
                                          "projects_sidecars_sha256": sorted(STRENGTH_READS_SHA256)},
                       "seconds": {side: res[side]["seconds"] for side in POLICIES}},
            "selection": {side: {"runs": len(res[side]["run_ids"]), "run_ids_sha256": sha256_lines(res[side]["run_ids"]),
                                 "anchor_runs": len(res[side]["anchor_run_ids"]),
                                 "anchor_run_ids_sha256": sha256_lines(res[side]["anchor_run_ids"])} for side in POLICIES},
            "checks": {},
        },
        "marks": block(MARK_COLUMNS, rows["marks"]),
        "static_anchors": block(ANCHOR_COLUMNS, rows["static_anchors"]),
        "mean_columns": block(MEAN_COLUMNS, rows["mean_columns"]),
    }
    out["provenance"]["payload_sha256"] = JC.payload_sha256(out)
    if compare:
        out["provenance"]["checks"]["delivered_values"] = compare_values(rows["marks"], compare)
    t = JC.text_numbers(out)
    out["provenance"]["checks"]["text_numbers"] = {k: (list(v) if isinstance(v, tuple) else v) for k, v in t.items()}
    out["provenance"]["checks"]["text_numbers"]["static_unsteered_asr_pct"] = {j: list(v) for j, v in t["static_unsteered_asr_pct"].items()}
    out["provenance"]["seconds"] = round(time.time() - t0)
    bad = JC.problems(out)
    if bad:
        refuse("; ".join(bad))
    return out


def dumps(obj: dict) -> str:
    """Top-level keys indented, each block row on one line."""
    parts = []
    for k, v in obj.items():
        if isinstance(v, dict) and "rows" in v and "columns" in v:
            body = ("{\n  \"columns\": " + json.dumps(v["columns"]) + ",\n  \"rows\": [\n"
                    + ",\n".join("   " + json.dumps(r, allow_nan=False) for r in v["rows"]) + "\n  ]\n }")
        else:
            body = json.dumps(v, indent=1, allow_nan=False).replace("\n", "\n ")
        parts.append(f" {json.dumps(k)}: {body}")
    return "{\n" + ",\n".join(parts) + "\n}\n"


def _strip(d):
    if isinstance(d, dict):
        return {k: _strip(v) for k, v in d.items() if k not in VOLATILE}
    if isinstance(d, list):
        return [_strip(v) for v in d]
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--registry", help="the directory holding the frozen registry copy (registry.sqlite)")
    ap.add_argument("--compare-values", nargs="*", default=[], help="the value files written beside the delivered figure")
    ap.add_argument("--check", action="store_true", help="rebuild and compare with the committed file")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--keep", default=None, help="keep each policy's reduced readings and log in this directory")
    ap.add_argument("--frames", nargs=2, metavar=("POLICY", "OUT"), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.frames:
        return frames_main(*a.frames)
    if not a.registry:
        refuse("no --registry (the directory holding the frozen registry copy)")
    out = build(Path(os.path.expanduser(a.registry)), a.compare_values, Path(a.keep) if a.keep else None)
    text = dumps(out)
    if json.loads(text) != out:
        refuse("the written text does not parse back to the readings")
    if a.check:
        old = json.loads(Path(a.out).read_text())
        if not a.compare_values:                                  # a check without the delivered value files compares the rest
            old["provenance"]["checks"].pop("delivered_values", None)
        if _strip(old) != _strip(out):
            diff = sorted(k for k in set(old) | set(out) if _strip(old.get(k)) != _strip(out.get(k)))
            print(f"DIFFERS from {Path(a.out).name}: {diff}")
            return 1
        print(f"{Path(a.out).name}: matches a fresh build ({len(out['marks']['rows'])} marks)")
        return 0
    Path(a.out).write_text(text)
    pv = out["provenance"]
    print(f"wrote {Path(a.out).name} ({len(text) / 1e3:.0f} kB): {len(out['marks']['rows'])} marks, {len(out['static_anchors']['rows'])} "
          f"static anchors, {len(out['mean_columns']['rows'])} mean marks; registry copy head {pv['registry']['head_in_copy']} pinned to "
          f"{AS_OF}; payload {pv['payload_sha256'][:12]}; {pv['seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
