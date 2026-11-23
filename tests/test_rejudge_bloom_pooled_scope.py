"""The pooled-run scope (scripts/rejudge/petri/pooled_scope.py): a ``petri_pooled`` run whose Opus judgment filed no item rows is
composed from its six behaviour runs' units in the paper-row scope; a pooled run whose Opus items exist is rebuilt to exactly the unit
lines ``scope.py`` writes for it; every inconsistency is refused by name; and ``file_records.py`` files the composed run with the
behaviour runs' verdicts, naming the behaviour runs it was composed from. Offline: an invented registry (the views as plain tables), an
invented transcript mapping, the real harvester over the fake Batch API."""
from __future__ import annotations

import gzip
import json
import os
import pathlib
import sqlite3
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from rejudge.petri import pooled_scope as PS  # noqa: E402
from rejudge.petri import scope as SC  # noqa: E402

CAMP = "testcamp"
OWN = {"cyber_bio_chem_capabilities": "cyber_bio_chem_operational_uplift", "defer_to_users": "defer_to_users",
       "delusion_sycophancy": "delusion_sycophancy", "instruction_evasion": "instruction_evasion",
       "reasoning_unfaithfulness": "reasoning_unfaithfulness", "self_preservation": "self_preservation"}
SCENS = ("scen_a", "scen_b")
A, B = "1111111111111111", "2222222222222222"            # two standups: A's pooled run has no Opus items, B's has them
POOLED_A = f"{CAMP}:petri_pooled:aaaaaaaaaaaa:{A}"
POOLED_B = f"{CAMP}:petri_pooled:aaaaaaaaaaaa:{B}"


def beh_run(beh: str, key: str) -> str:
    return f"{CAMP}:petri_{beh}:{beh[:4] * 3}:{key}"


def phys(key: str, beh: str, scen: str) -> str:
    return f"petri/TESTHOST_audit|rk{key[:4]}|{beh}|{scen}|1"


def _world(tmp_path: pathlib.Path, pooled_a_items: bool = False, extra_runs=()) -> tuple:
    """The invented registry and transcript mapping; returns (registry path, matched path)."""
    db = str(tmp_path / "registry.sqlite")
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
    items = []

    def add_run(rid, ev, key, n_items, with_items, rkey=None, model="model_x", paper=1, sel="selected"):
        con.execute("INSERT INTO run VALUES (?,?,?,?,?,?,?,?,?,?)", (rid, CAMP, ev, model, "si:obj" + key[:2], rid.split(":")[2] * 5,
                                                                     SC.PETRI_JUDGE, 0, None, json.dumps({"petri_run_key": rkey or "rk" + key[:4]})))
        con.execute("INSERT INTO v_run_current VALUES (?,?,?)", (rid, sel, paper))
        jid = "opus-" + rid[-24:]
        con.execute("INSERT INTO judgment VALUES (?,?,?,?,?,?,?)", (jid, rid, SC.PETRI_JUDGE, "complete", n_items, 0, "2026-11-01T00:00:00.000Z"))
        return jid

    def add_items(rid, jid, key, behs):
        for beh in behs:
            for i, scen in enumerate(SCENS):
                con.execute("INSERT INTO v_item_result_current VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (rid, jid, f"{beh}/{scen}", "petri_score", 0, 0, -1, float(3 + i + len(beh) % 4), None, "rev1"))
                items.append({"run_id": rid, "item_id": f"{beh}/{scen}", "trial_index": 0, "sample_index": 0, "phys": phys(key, beh, scen),
                              "path": f"/logs/TESTHOST_audit/{key[:4]}.eval", "scenario": scen, "epoch": 1, "behavior": beh})

    for key in (A, B):
        for beh in OWN:
            rid = beh_run(beh, key)
            add_items(rid, add_run(rid, "petri_" + beh, key, len(SCENS), True), key, [beh])
    ja = add_run(POOLED_A, "petri_pooled", A, len(OWN) * len(SCENS), pooled_a_items)
    if pooled_a_items:
        add_items(POOLED_A, ja, A, list(OWN))
    add_items(POOLED_B, add_run(POOLED_B, "petri_pooled", B, len(OWN) * len(SCENS), True), B, list(OWN))
    for rid, ev, key, kw in extra_runs:
        jid = add_run(rid, ev, key, kw.pop("n_items", len(SCENS)), True, **kw)
        if ev != "petri_pooled":
            add_items(rid, jid, key, [ev[len("petri_"):]])
    con.commit()
    con.close()
    m = str(tmp_path / "items_matched.json")
    json.dump({"items": items, "unmatched": [], "run_root": {}}, open(m, "w"))
    return db, m


def _scope(tmp_path, db, m, name="main"):
    out = tmp_path / name
    assert SC.main(["--registry", db, "--matched", m, "--out", str(out)]) == 0
    return str(out)


def _runs(tmp_path, *ids, name="runs.txt"):
    p = tmp_path / name
    p.write_text("# pooled runs\n" + "\n".join(ids) + "\n")
    return str(p)


def _read(out):
    jobs = {j["run_id"]: j for j in (json.loads(l) for l in open(os.path.join(out, "jobs.jsonl")))}
    with gzip.open(os.path.join(out, "units.tsv.gz"), "rt") as fh:
        lines = fh.read().splitlines()
    return jobs, lines


def test_a_pooled_run_without_items_is_its_six_behaviour_runs_and_one_with_items_is_rebuilt_exactly(tmp_path):
    db, m = _world(tmp_path)
    main = _scope(tmp_path, db, m)
    mjobs, mlines = _read(main)
    assert mjobs[POOLED_A]["n_units_mapped"] == 0 and not [l for l in mlines if l.startswith(POOLED_A + "\t")]   # what scope.py leaves
    out = str(tmp_path / "pooled")
    assert PS.main(["--registry", db, "--scope", main, "--runs", _runs(tmp_path, POOLED_A, POOLED_B), "--out", out]) == 0
    jobs, lines = _read(out)
    assert lines[0] == mlines[0]
    # A: the six behaviour runs' unit lines under the pooled run id, nothing else changed
    want = [POOLED_A + l[l.index("\t"):] for ev in PS.BEHAVIOUR_EVALS for l in mlines[1:] if l.startswith(beh_run(ev[6:], A) + "\t")]
    assert [l for l in lines[1:] if l.startswith(POOLED_A + "\t")] == want and len(want) == 12
    ja = jobs[POOLED_A]
    assert ja["opus_judgment_id"] == mjobs[POOLED_A]["opus_judgment_id"] and ja["n_items"] == ja["n_units_mapped"] == 12
    assert ja["units_composed_from"] == {ev: beh_run(ev[6:], A) for ev in PS.BEHAVIOUR_EVALS} and ja["opus_item_keys"] == 0
    # B: its Opus items exist, so scope.py mapped it; the composition reproduces scope.py's lines and job exactly
    assert [l for l in lines[1:] if l.startswith(POOLED_B + "\t")] == [l for l in mlines[1:] if l.startswith(POOLED_B + "\t")]
    jb = dict(jobs[POOLED_B])
    assert jb.pop("units_composed_from") and jb.pop("opus_item_keys") == 12 and jb == mjobs[POOLED_B]


@pytest.mark.parametrize("case", ["missing", "duplicate", "run_key", "model", "unmapped", "n_items", "opus_keys", "not_pooled", "unknown"])
def test_an_inconsistent_pooled_run_is_refused_by_name(tmp_path, case):
    extra, runs, match = [], [POOLED_A], None
    if case == "duplicate":                     # a second defer_to_users run of standup A in the paper scope
        extra = [(f"{CAMP}:petri_defer_to_users:dddddddddddd:{A}", "petri_defer_to_users", A, {})]
        match = "duplicated"
    elif case == "run_key":                     # a behaviour run of another Petri run under the same object key
        runs, match = [f"{CAMP}:petri_pooled:bbbbbbbbbbbb:{A}"], "differs from the pooled run"
        extra = [(runs[0], "petri_pooled", A, {"rkey": "other", "n_items": 12})]
    elif case == "model":
        runs, match = [f"{CAMP}:petri_pooled:cccccccccccc:{A}"], "differs from the pooled run"
        extra = [(runs[0], "petri_pooled", A, {"model": "model_y", "n_items": 12})]
    elif case == "n_items":                     # the Opus header names 13 items; 12 compose
        runs, match = [f"{CAMP}:petri_pooled:eeeeeeeeeeee:{A}"], "judged 13 items"
        extra = [(runs[0], "petri_pooled", A, {"n_items": 13})]
    db, m = _world(tmp_path, pooled_a_items=(case == "opus_keys"), extra_runs=extra)
    main = _scope(tmp_path, db, m)
    if case == "missing":                       # standup A's self_preservation run is not in the paper scope
        jobs = [l for l in open(os.path.join(main, "jobs.jsonl")) if beh_run("self_preservation", A) not in l]
        open(os.path.join(main, "jobs.jsonl"), "w").writelines(jobs)
        match = "missing for"
    elif case == "unmapped":                    # one of its items did not map to a transcript
        with gzip.open(os.path.join(main, "units.tsv.gz"), "rt") as fh:
            lines = fh.read().splitlines()
        drop = next(i for i, l in enumerate(lines) if l.startswith(beh_run("defer_to_users", A) + "\t"))
        with gzip.open(os.path.join(main, "units.tsv.gz"), "wt") as fh:
            fh.write("\n".join(lines[:drop] + lines[drop + 1:]) + "\n")
        match = "every item must map"
    elif case == "opus_keys":                   # filed Opus items that the composition does not reproduce
        con = sqlite3.connect(db)
        con.execute("DELETE FROM v_item_result_current WHERE run_id = ? AND item_id = 'defer_to_users/scen_a'", (POOLED_A,))
        con.commit()
        con.close()
        match = "item keys differ"
    elif case == "not_pooled":
        runs, match = [beh_run("defer_to_users", A)], "not petri_pooled"
    elif case == "unknown":
        runs, match = [f"{CAMP}:petri_pooled:aaaaaaaaaaaa:9999999999999999"], "name no run"
    with pytest.raises(SystemExit, match=match):
        PS.main(["--registry", db, "--scope", main, "--runs", _runs(tmp_path, *runs), "--out", str(tmp_path / "out")])
    assert not os.path.exists(tmp_path / "out" / "jobs.jsonl")


def test_the_filing_files_a_composed_pooled_run_with_its_behaviour_runs_verdicts(tmp_path):
    pytest.importorskip("inspect_petri")
    from rejudge.petri import file_records as FR
    from rejudge.petri import request as R
    from test_rejudge_bloom_request import synthetic_prompt, synthetic_tool
    from test_rejudge_bloom_units import _harvest, _run_dir, verdict_entry
    from sisteer.registry import records as RR
    db, m = _world(tmp_path)
    main = _scope(tmp_path, db, m)
    pooled = str(tmp_path / "pooled")
    assert PS.main(["--registry", db, "--scope", main, "--runs", _runs(tmp_path, POOLED_A), "--out", pooled]) == 0
    # standup A's twelve transcripts: the enumeration and a ledger the real harvester filled over the fake Batch API
    rows, tools = [], {}
    for beh, own in OWN.items():
        tool = R.canonical_tool(synthetic_tool(own))
        tools[R.tool_sha(tool)] = tool
        for scen in SCENS:
            prompt = synthetic_prompt(f"[M1] USER:\n{beh} {scen}\n\n[M2] ASSISTANT:\nreply")
            head, rest = R.split_prompt(prompt)
            rows.append({"phys": phys(A, beh, scen), "key": R.judge_input_key(prompt, tool), "prefix_sha": R.prefix_sha(prompt, tool),
                         "tool_sha": R.tool_sha(tool), "behavior": beh, "own_dimension": own, "prompt": prompt,
                         "tokens": {"head": len(head) // 4, "rest": len(rest) // 4, "tool": 2000}})
    enum = tmp_path / "enum"
    enum.mkdir()
    with gzip.open(enum / "inputs.jsonl.gz", "wt") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    (enum / "tools.json").write_text(json.dumps(tools))
    rd = _run_dir(tmp_path, "ledger", rows, tools)
    h, _ = _harvest(rd, lambda req, a: verdict_entry(req), cycles=6)
    assert h.ledger.outcome_counts() == {"verdict": 12}
    h.ledger.close()

    def stage(scope, name, runs):
        assert FR.main(["--scope", scope, "--enum", str(enum), "--run-dir", rd, "--staging", str(tmp_path / name), "--runs",
                        _runs(tmp_path, *runs, name=name + ".txt"), "--code-sha", "0" * 39 + "1", "--dry-run"]) == 0
        recs = [rec for p in sorted(pathlib.Path(tmp_path, name).rglob("*.jsonl*")) for _, rec, _ in RR.read_record_file(str(p))]
        js = {r["payload"]["run_id"]: r["payload"] for r in recs if r["record_type"] == "judgment"}
        items = [r["payload"] for r in recs if r["record_type"] == "item_result"]
        return js, items, json.load(open(tmp_path / name / "filing_report.json"))

    pj, pitems, prep = stage(pooled, "s_pooled", [POOLED_A])
    bj, bitems, _ = stage(main, "s_beh", [beh_run(ev[6:], A) for ev in PS.BEHAVIOUR_EVALS])
    assert prep["validation"]["refusals"] == [] and prep["runs"][POOLED_A] == {"status": "complete", "rows": 36, "verdict": 12}
    opus = json.loads(open(os.path.join(pooled, "jobs.jsonl")).readline())
    j = pj[POOLED_A]
    assert j["status"] == "complete" and j["n_items_judged"] == 12 and j["extra"]["re_measures_judgment_id"] == opus["opus_judgment_id"]
    assert j["extra"]["units_composed_from"] == opus["units_composed_from"]
    assert all("units_composed_from" not in x["extra"] for x in bj.values())          # a scope.py job files no such key
    # every pooled row is the behaviour run's row of the same key: same value, request key, batch, transcript
    strip = lambda r: (r["item_id"], r["metric_id"], r["trial_index"], r["sample_index"], r["turn_index"], r["value"], r["label"],
                       r["judged_utc"], {k: v for k, v in r["extra"].items() if k != "re_measures_judgment_id"})
    assert sorted(map(strip, pitems), key=str) == sorted(map(strip, bitems), key=str) and len(pitems) == 36
    assert {r["extra"]["re_measures_judgment_id"] for r in pitems} == {opus["opus_judgment_id"]}
