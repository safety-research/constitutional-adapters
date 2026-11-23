"""The Petri item matcher (scripts/rejudge/petri/match_items.py): every CURRENT item of the Opus 5 Petri judge is mapped to the stored
inspect sample whose stored judge scores are the registry's, the run's audit root chosen by the most matching scores (a tie goes to the
root the run's store_uri names), ``petri_score`` compared with the behaviour's own dimension, and an item with no candidate listed as
unmatched (exit 2). Offline: a synthetic log index and a synthetic registry file; no inspect logs are read."""
from __future__ import annotations

import json
import pathlib
import sqlite3
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
for p in (ROOT / "scripts", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from rejudge.petri import match_items as MI  # noqa: E402
from rejudge.petri.scope import PETRI_JUDGE  # noqa: E402

STATE = "/data/TESTHOST/.local/state"


def _log(root, name, run_key, behavior, samples, mtime=1.0):
    """One index entry (``index_file``'s shape) for a log of ``behavior`` under the audit root ``root``."""
    return {"path": f"{STATE}/{root}/inspect_logs/{name}.eval", "size": 1, "mtime": mtime, "status": "success", "task": "petri",
            "md": {"petri_run_key": run_key, "behavior_dir_key": f"{behavior}/bundle"}, "roles": {},
            "samples": [{"id": sid, "epoch": ep, "completed": True, "error": False, "judge_usage": {"input_tokens": 10}, "scores": sc,
                         "n_msgs": 4, "role_models": []} for sid, ep, sc in samples]}


def _registry(path, runs, items):
    """A registry file with the tables ``current_items`` reads: ``runs`` = [(run_id, judge_id, status, store_uri, run_key)], ``items`` =
    [(run_id, item_id, trial, metric, value, judgment_id or None for the run's)]."""
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE eval (eval_id TEXT PRIMARY KEY, paper_status TEXT);
        CREATE TABLE run (run_id TEXT PRIMARY KEY, campaign TEXT, eval_id TEXT, model_id TEXT, object_id TEXT, store_uri TEXT,
                          extra_json TEXT, instrument_id TEXT);
        CREATE TABLE judgment (judgment_id TEXT PRIMARY KEY, judge_id TEXT, status TEXT, n_items_judged INTEGER, run_id TEXT);
        CREATE TABLE v_item_result_current (run_id TEXT, item_id TEXT, metric_id TEXT, trial_index INTEGER, sample_index INTEGER,
                                            turn_index INTEGER, judgment_id TEXT, value REAL);
        INSERT INTO eval VALUES ('petri_bloom', 'paper');
    """)
    for rid, judge, status, store_uri, key in runs:
        con.execute("INSERT INTO run VALUES (?, 'c1', 'petri_bloom', 'm1', 'o1', ?, ?, 'i1')",
                    (rid, store_uri, json.dumps({"petri_run_key": key} if key else {})))
        con.execute("INSERT INTO judgment VALUES (?, ?, ?, 3, ?)", (f"j-{rid}", judge, status, rid))
    for rid, item, trial, metric, value, jid in items:
        con.execute("INSERT INTO v_item_result_current VALUES (?, ?, ?, ?, 0, 0, ?, ?)", (rid, item, metric, trial, jid or f"j-{rid}", value))
    con.commit()
    con.close()
    return str(path)


def _scores(own_key, own, aware, real):
    return {own_key: own, "eval_awareness": aware, "scenario_realism": real}


def _world(tmp_path):
    runs = [("R1", PETRI_JUDGE, "complete", "/data/cells/audit-b/k1", "k1"),
            ("R2", PETRI_JUDGE, "partial", "/data/cells/audit-a/k2", "k2"),
            ("R3", "0" * 64, "complete", "/data/cells/audit-a/k3", "k3"),                  # another judge: not read
            ("R4", PETRI_JUDGE, "failed", "/data/cells/audit-a/k4", "k4")]                 # a failed judgment: not read
    items = []
    for item, trial, (own, aware, real) in (("self_preservation/s1", 0, (7, 2, 8)), ("self_preservation/s2", 1, (3, 1, 9)),
                                            ("cyber/s9", 0, (5, 4, 6))):
        for m, v in (("petri_score", own), ("eval_awareness", aware), ("scenario_realism", real)):
            items.append(("R1", item, trial, m, v, None))
    items.append(("R1", "self_preservation/s1", 0, "petri_score", 1.0, "j-stale"))           # another judgment's row: ignored
    items += [("R2", "self_preservation/s5", 0, "petri_score", 4, None), ("R3", "self_preservation/s1", 0, "petri_score", 7, None),
              ("R4", "self_preservation/s1", 0, "petri_score", 7, None)]
    reg = _registry(tmp_path / "registry.sqlite", runs, items)
    sp = "self_preservation"
    index = [
        # two audit roots carry the same scores for R1's key: the tie goes to audit-b, which R1's store_uri names
        _log("petri/audit-a", "a1", "k1", sp, [("s1", 1, _scores(sp, 7, 2, 8)), ("s2", 2, _scores(sp, 3, 1, 9))]),
        _log("petri/audit-b", "b1", "k1", sp, [("s1", 1, _scores(sp, 7, 2, 8)), ("s2", 2, _scores(sp, 3, 1, 9))], mtime=5.0),
        # in the chosen root, a copy under before-rejudge with the same scores loses to the one outside it, though newer
        _log("petri/audit-b", "before-rejudge/b0", "k1", sp, [("s1", 1, _scores(sp, 7, 2, 8))], mtime=9.0),
        # cyber is scored under its own dimension's name; a wrong-score candidate in the same root loses
        _log("petri/audit-b", "b2", "k1", "cyber", [("s9", 1, _scores("cyber_bio_chem_operational_uplift", 5, 4, 6))]),
        _log("petri/audit-b", "b3", "k1", "cyber", [("s9", 1, _scores("cyber_bio_chem_operational_uplift", 2, 4, 6))], mtime=7.0),
        # an unreadable log is listed, never fatal; an uncompleted sample is no candidate
        {"path": f"{STATE}/petri/audit-a/inspect_logs/bad.eval", "err": "OSError()"},
        _log("petri/audit-a", "a9", "k2", sp, [("s5", 2, _scores(sp, 4, 1, 1))]),        # R2's s5 is trial 0 = epoch 1: no candidate
    ]
    return reg, index


def test_current_items_reads_only_the_opus_judges_complete_and_partial_judgments(tmp_path):
    reg, _ = _world(tmp_path)
    with sqlite3.connect(reg) as con:
        doc = MI.current_items(con)
    assert sorted(doc["runs"]) == ["R1", "R2"] and doc["runs"]["R1"]["petri_run_key"] == "k1"
    r1 = {(i["item_id"], i["trial_index"]): i["metrics"] for i in doc["items"] if i["run_id"] == "R1"}
    assert r1[("self_preservation/s1", 0)] == {"petri_score": 7, "eval_awareness": 2, "scenario_realism": 8}   # not the stale row
    with sqlite3.connect(reg) as con:
        assert sorted(MI.current_items(con, ["R2"])["runs"]) == ["R2"]


def test_items_map_to_the_matching_sample_of_the_chosen_root(tmp_path):
    reg, index = _world(tmp_path)
    with sqlite3.connect(reg) as con:
        m = MI.match(index, MI.current_items(con))
    rr = m["run_root"]["R1"]
    assert rr["root"] == "petri/audit-b" and not rr["ambiguous"] and rr["n_roots"] == 2 and rr["matches"] == rr["total"]
    by = {o["item_id"]: o for o in m["items"]}
    assert by["self_preservation/s1"]["path"].endswith("/audit-b/inspect_logs/b1.eval")
    assert by["self_preservation/s1"]["n_candidates_same_root"] == 2
    assert by["self_preservation/s2"]["epoch"] == 2 and by["self_preservation/s2"]["phys"] == "petri/audit-b|k1|self_preservation|s2|2"
    assert by["cyber/s9"]["path"].endswith("/b2.eval") and (by["cyber/s9"]["score_match"], by["cyber/s9"]["score_tot"]) == (3, 3)
    assert m["unmatched"] == [("R2", "self_preservation/s5")]
    s = MI.summary(m)
    assert s["matched"] == 3 and s["unmatched"] == 1 and s["items_with_score_mismatch"] == 0 and s["runs_without_root"] == 1


def test_root_of_names_the_audit_root_and_the_target_seed():
    assert MI.root_of(f"{STATE}/petri_native/sweep-1/cells/x/inspect_logs/a.eval") == "petri_native/sweep-1/cells/x"
    assert MI.root_of(f"{STATE}/petri/sweep-2/target-seed-3/inspect_logs/a.eval") == "petri/sweep-2/target-seed-3#seed3"
    assert MI.root_of("/elsewhere/a.eval") == "/elsewhere/a.eval"
    assert MI.root_of("/x/runs/r1/logs/a.eval", regex=r".*?/runs/(.+?)/logs/") == "r1"
    assert MI.run_key({"petri_run_key": None, "store_uri": "/data/cells/audit-a/k7/"}) == "k7"


def test_the_cli_reads_the_index_and_the_registry_read_only_and_exits_2_on_an_unmatched_item(tmp_path, capsys):
    reg, index = _world(tmp_path)
    idx = tmp_path / "index.json"
    idx.write_text(json.dumps(index))
    before = pathlib.Path(reg).read_bytes()
    out = tmp_path / "items_matched.json"
    assert MI.main(["--out", str(out), "--index", str(idx), "--registry", reg, "--workers", "1"]) == 2
    assert pathlib.Path(reg).read_bytes() == before
    s = json.loads(capsys.readouterr().out)
    assert s["matched"] == 3 and s["unmatched"] == 1 and s["unreadable_logs"] == 1 and s["logs_indexed"] == len(index)
    runs = tmp_path / "runs.txt"
    runs.write_text("# only R1\nR1\n")
    assert MI.main(["--out", str(out), "--index", str(idx), "--registry", reg, "--runs", str(runs)]) == 0
    assert {o["run_id"] for o in json.loads(out.read_text())["items"]} == {"R1"}
