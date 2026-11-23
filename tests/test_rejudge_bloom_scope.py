"""The Petri re-judge scope (scripts/rejudge/petri/scope.py): the default scope is the paper rows; ``--runs`` scopes exactly the listed
runs of the Petri evals whatever their ``in_current_paper`` (the contrast-anchor supplement), with the same job and unit rows a run gets
in the paper scope, and refuses an id that names no Petri run. Offline: an invented registry (the views as plain tables) and an invented
transcript mapping in temp directories."""
from __future__ import annotations

import gzip
import json
import pathlib
import sqlite3
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from rejudge.petri import scope as SC  # noqa: E402

PAPER = "camp:petri_defer_to_users:aaaaaaaaaaaa:1111111111111111"
POOLED = "camp:petri_pooled:bbbbbbbbbbbb:1111111111111111"
ANCHOR = "camp:petri_defer_to_users:aaaaaaaaaaaa:2222222222222222"
OTHER_EVAL = "camp:strongreject:cccccccccccc:2222222222222222"


def _registry(path: pathlib.Path) -> str:
    db = str(path / "registry.sqlite")
    con = sqlite3.connect(db)
    con.executescript("""
        CREATE TABLE run (run_id TEXT, campaign TEXT, eval_id TEXT, model_id TEXT, object_id TEXT, instrument_id TEXT, judge_id TEXT,
                          trial_index INTEGER, store_uri TEXT, extra_json TEXT);
        CREATE TABLE v_run_current (run_id TEXT, selection_state TEXT, in_current_paper INTEGER);
        CREATE TABLE judgment (judgment_id TEXT, run_id TEXT, judge_id TEXT, status TEXT, n_items_judged INTEGER, n_judge_refusals INTEGER,
                               finished_utc TEXT);
        CREATE TABLE v_item_result_current (run_id TEXT, judgment_id TEXT, item_id TEXT, metric_id TEXT, trial_index INTEGER,
                                            sample_index INTEGER, turn_index INTEGER, value REAL, label TEXT, source_revision_id TEXT);
    """)
    runs = [(PAPER, "petri_defer_to_users", "si:obj", 1, "selected"), (POOLED, "petri_pooled", "si:obj", 1, "selected"),
            (ANCHOR, "petri_defer_to_users", "none", 0, "superseded"), (OTHER_EVAL, "strongreject", "none", 0, "selected")]
    for rid, ev, obj, paper, sel in runs:
        con.execute("INSERT INTO run VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (rid, "camp", ev, "model_x", obj, rid.split(":")[2] * 2, SC.PETRI_JUDGE, 0, None, json.dumps({"petri_run_key": "k" + rid[-4:]})))
        con.execute("INSERT INTO v_run_current VALUES (?,?,?)", (rid, sel, paper))
        jid = "j-" + rid[-20:]
        con.execute("INSERT INTO judgment VALUES (?,?,?,?,?,?,?)", (jid, rid, SC.PETRI_JUDGE, "complete", 2, 0, "2026-11-01T00:00:00.000Z"))
        for scen, score in (("scen_a", 3.0), ("scen_b", 7.0)):
            item = f"defer_to_users/{scen}"
            for metric, v in (("petri_score", score), ("eval_awareness", 2.0), ("scenario_realism", 8.0)):
                con.execute("INSERT INTO v_item_result_current VALUES (?,?,?,?,?,?,?,?,?,?)", (rid, jid, item, metric, 0, 0, -1, v, None, ""))
    con.commit()
    con.close()
    return db


def _matched(path: pathlib.Path) -> str:
    items = []
    for rid, root in ((PAPER, "petri_native/audit_s1"), (POOLED, "petri_native/audit_s1"), (ANCHOR, "petri_native/audit_s2")):
        for scen in ("scen_a", "scen_b"):
            items.append({"run_id": rid, "item_id": f"defer_to_users/{scen}", "trial_index": 0, "sample_index": 0,
                          "phys": f"{root}|key|defer_to_users|{scen}|1", "path": f"/logs/{root}/x.eval", "scenario": scen, "epoch": 1,
                          "behavior": "defer_to_users"})
    m = str(path / "items_matched.json")
    json.dump({"items": items, "unmatched": [], "run_root": {PAPER: {"root": "petri_native/audit_s1"}, POOLED: {"root": "petri_native/audit_s1"},
                                                             ANCHOR: {"root": "petri_native/audit_s2"}}}, open(m, "w"))
    return m


def _read(out: pathlib.Path):
    jobs = {j["run_id"]: j for j in (json.loads(l) for l in open(out / "jobs.jsonl"))}
    with gzip.open(out / "units.tsv.gz", "rt") as fh:
        units = fh.read().splitlines()
    return jobs, units


def test_default_scope_is_the_paper_rows(tmp_path):
    db, m = _registry(tmp_path), _matched(tmp_path)
    assert SC.main(["--registry", db, "--matched", m, "--out", str(tmp_path / "paper")]) == 0
    jobs, units = _read(tmp_path / "paper")
    assert sorted(jobs) == sorted([PAPER, POOLED])
    assert len(units) == 1 + 4                                     # header + two items per paper run
    assert json.load(open(tmp_path / "paper" / "report.json"))["runs_file"] is None


def test_runs_scopes_exactly_the_listed_runs_paper_rows_or_not(tmp_path):
    db, m = _registry(tmp_path), _matched(tmp_path)
    runs = tmp_path / "runs.txt"
    runs.write_text(f"# anchors\n{ANCHOR}\n{PAPER}\n")
    assert SC.main(["--registry", db, "--matched", m, "--out", str(tmp_path / "anchors"), "--runs", str(runs)]) == 0
    jobs, units = _read(tmp_path / "anchors")
    assert sorted(jobs) == sorted([ANCHOR, PAPER])
    assert jobs[ANCHOR]["selection_state"] == "superseded" and jobs[ANCHOR]["n_units_mapped"] == 2
    assert [u for u in units[1:] if u.startswith(ANCHOR)] and all("audit_s2|" in u for u in units[1:] if u.startswith(ANCHOR))
    # the paper run listed with --runs gets the SAME job and unit rows as in the paper scope
    SC.main(["--registry", db, "--matched", m, "--out", str(tmp_path / "paper")])
    pjobs, punits = _read(tmp_path / "paper")
    assert jobs[PAPER] == pjobs[PAPER]
    assert units[0] == punits[0]
    assert [u for u in units[1:] if u.startswith(PAPER + "\t")] == [u for u in punits[1:] if u.startswith(PAPER + "\t")]
    assert json.load(open(tmp_path / "anchors" / "report.json"))["runs_file"] == str(runs)


def test_runs_refuses_an_id_that_names_no_petri_run(tmp_path):
    db, m = _registry(tmp_path), _matched(tmp_path)
    for bad in (OTHER_EVAL, "camp:petri_defer_to_users:aaaaaaaaaaaa:9999999999999999"):
        runs = tmp_path / "runs.txt"
        runs.write_text(f"{ANCHOR}\n{bad}\n")
        with pytest.raises(SystemExit, match="name no run of a Petri eval"):
            SC.main(["--registry", db, "--matched", m, "--out", str(tmp_path / "bad"), "--runs", str(runs)])
    empty = tmp_path / "empty.txt"
    empty.write_text("# nothing\n")
    with pytest.raises(SystemExit, match="lists no run id"):
        SC.main(["--registry", db, "--matched", m, "--out", str(tmp_path / "bad2"), "--runs", str(empty)])
