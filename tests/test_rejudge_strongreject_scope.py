"""The StrongREJECT scope (scripts/rejudge/strongreject/scope.py): ``--runs FILE`` selects exactly the listed runs (a supplement of
contrast anchors that are not paper rows) and builds each one's job by the same rules as the paper-row scope, so a paper run listed in
the file gets the job the paper-row scope gives it, byte for byte. Synthetic registry, maps and goals in tmp_path."""
from __future__ import annotations

import json
import pathlib
import sqlite3
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rejudge.strongreject import scope as SC  # noqa: E402

V1 = SC.HAIKU_V1
GOALS = {"g1": {"dataset": "harmbench", "revision": "r", "text": "goal one"}}


def _registry(path: pathlib.Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE run (run_id TEXT PRIMARY KEY, campaign TEXT, eval_id TEXT, instrument_id TEXT, judge_id TEXT, model_id TEXT, "
        "object_id TEXT, lane_source_legacy TEXT, pass_label TEXT, trial_index INTEGER, token_cap_tokens INTEGER);"
        "CREATE TABLE run_flags (run_id TEXT, in_current_paper INTEGER, selection_state TEXT);"
        "CREATE VIEW v_run_current AS SELECT run_id, in_current_paper, selection_state FROM run_flags;"
        "CREATE TABLE judgment (judgment_id TEXT, run_id TEXT, judge_id TEXT, status TEXT, finished_utc TEXT, n_items_judged INTEGER);")
    runs = [("c:codeattack:o1:paper", 1), ("c:codeattack:o0:anchor", 0), ("c:pasttense:o0:anchor", 0), ("c:mmlu:o0:other", 0)]
    for rid, paper in runs:
        camp, ev, obj, _ = rid.split(":")
        con.execute("INSERT INTO run VALUES (?,?,?,?,?,?,?,?,?,?,?)", (rid, camp, ev, "i", V1, "m", obj, camp, None, 0, None))
        con.execute("INSERT INTO run_flags VALUES (?,?,?)", (rid, paper, "current"))
        con.execute("INSERT INTO judgment VALUES (?,?,?,?,?,?)", (f"j-{rid}", rid, V1, "complete", "2026-11-01T00:00:00Z", 1))
    con.commit()
    con.close()


def _inputs(tmp_path: pathlib.Path):
    reg = tmp_path / "registry.sqlite"
    _registry(reg)
    static_map = [{"judgment_id": f"j-{rid}", "units": [f"{tmp_path}/stores/{rid.replace(':', '_')}"], "unresolved": []}
                  for rid in ("c:codeattack:o1:paper", "c:codeattack:o0:anchor", "c:pasttense:o0:anchor")]
    (tmp_path / "static_map.json").write_text(json.dumps(static_map))
    (tmp_path / "mtbon_map.json").write_text(json.dumps({}))
    (tmp_path / "out_x").mkdir()
    (tmp_path / "out_x" / "manifest.jsonl").write_text("")
    (tmp_path / "parts.json").write_text(json.dumps({"goals": GOALS, "goal_sets": {"mt": ["g1"], "bon": ["g1"]}}))
    (tmp_path / "goals259.json").write_text(json.dumps([dict(goal_id="g1", **GOALS["g1"])]))
    return ["--registry", str(reg), "--static-map", str(tmp_path / "static_map.json"), "--mtbon-map", str(tmp_path / "mtbon_map.json"),
            "--mtbon-manifests", str(tmp_path / "out_x" / "manifest.jsonl"), "--goals", str(tmp_path / "parts.json"),
            "--static-goals", str(tmp_path / "goals259.json")]


def _jobs(out: pathlib.Path):
    return [json.loads(l) for l in (out / "jobs.jsonl").read_text().splitlines()]


def test_runs_file_selects_exactly_the_listed_runs_by_the_same_rules(tmp_path):
    base = _inputs(tmp_path)
    assert SC.main(base + ["--out", str(tmp_path / "paper")]) == 0
    paper = _jobs(tmp_path / "paper")
    assert [j["run_id"] for j in paper] == ["c:codeattack:o1:paper"]          # in_current_paper = 1 only
    runs = tmp_path / "runs.txt"
    runs.write_text("c:pasttense:o0:anchor\nc:codeattack:o1:paper\n\nc:codeattack:o0:anchor\n")
    assert SC.main(base + ["--out", str(tmp_path / "supp"), "--runs", str(runs)]) == 0
    supp = _jobs(tmp_path / "supp")
    assert [j["run_id"] for j in supp] == ["c:codeattack:o0:anchor", "c:codeattack:o1:paper", "c:pasttense:o0:anchor"]
    # the paper run's job is the paper-row scope's, line for line; the goal table is the same file
    assert (tmp_path / "paper" / "jobs.jsonl").read_text().splitlines()[0] == (tmp_path / "supp" / "jobs.jsonl").read_text().splitlines()[1]
    assert (tmp_path / "paper" / "goals.json").read_bytes() == (tmp_path / "supp" / "goals.json").read_bytes()
    anchor = supp[0]
    assert anchor["category"] == "static" and anchor["kind"] == "sr" and anchor["haiku_judgment_id"] == "j-c:codeattack:o0:anchor"
    summary = json.loads((tmp_path / "supp" / "scope_summary.json").read_text())
    assert summary["jobs"] == 3 and summary["problems"] == [] and "--runs" in summary["selection"]


def test_runs_file_refuses_an_unknown_or_off_scope_run(tmp_path):
    base = _inputs(tmp_path)
    runs = tmp_path / "runs.txt"
    runs.write_text("c:codeattack:o0:anchor\nc:mmlu:o0:other\n")            # a run, but not on a jailbreak eval
    with pytest.raises(SystemExit, match="not runs on the eight jailbreak evals"):
        SC.main(base + ["--out", str(tmp_path / "s1"), "--runs", str(runs)])
    runs.write_text("c:codeattack:o0:anchor\nnot:a:run:id\n")
    with pytest.raises(SystemExit, match="1 run ids"):
        SC.main(base + ["--out", str(tmp_path / "s2"), "--runs", str(runs)])
