"""31b-fix-4: the placement's ledger event is reconciled idempotently from the record, under the ledger's lock — a mark-placed killed between
the remote PLACED commit and the local ledger append is repaired by the retry with exactly one event."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import job_ids, set_job_state  # noqa: E402
from test_pool_member_lifecycle_funnel import GROUP, World, _members, _target  # noqa: E402


def test_a_crash_between_the_remote_commit_and_the_ledger_append_is_repaired_by_the_retry_with_exactly_one_event(tmp_path):
    """PLACED is committed on the cluster under the lock and released before the local ledger append; an
    invocation dying in that window left a live placement the append-only ledger never recorded, and every retry read `unchanged`. The
    event is now reconciled from the record itself on every result, check-and-append under the ledger's lock: the retry appends exactly
    one event; a further repeat appends none; the same for a supersession."""
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    j1, j2 = _members(w)
    lw1 = _target(w, "crescendo/g012", "0,1,2"); lw2 = _target(w, "crescendo/g34", "3,4"); lw3 = _target(w, "pair/g012", "0,1,2")
    pool = w.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}/attackers"
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}", SISTEER_TWIN_CRASH_AT="placement_after_remote")
    assert r.returncode == 97 and "CRASH INJECTED at placement_after_remote" in r.stderr, r.stdout + r.stderr
    assert json.loads((pool / "PLACED").read_text())["version"] == 1                                       # the record is live …
    assert not [e for e in w.ledger() if e["event"] in ("placement_complete", "placement_superseded")]     # … and the ledger never heard of it
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}")                                 # the retry: unchanged, and REPAIRED
    assert r.returncode == 0 and "the ledger lacked its event; appended it" in (r.stdout + r.stderr), r.stdout + r.stderr
    ev = [e for e in w.ledger() if e["event"] == "placement_complete"]
    assert len(ev) == 1 and ev[0]["group"] == GROUP and ev[0]["version"] == 1 and [t["assigned_member_job_id"] for t in ev[0]["targets"]] == [j1, j2] and ev[0]["members"] == [j1, j2]
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}")                                 # a further repeat: nothing appended
    assert r.returncode == 0 and "nothing written" in (r.stdout + r.stderr) and len([e for e in w.ledger() if e["event"] == "placement_complete"]) == 1
    # the same window on a SUPERSESSION: the record moved to version 2, the ledger lacks placement_superseded; the retry (same roster, as a
    # plain mark-placed) reconciles exactly one supersede event from the record's own version, reason and changes
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2},{lw3}", "--supersede", "--reason", "pair/g012 on the second pass", SISTEER_TWIN_CRASH_AT="placement_after_remote")
    assert r.returncode == 97, r.stdout + r.stderr
    doc = json.loads((pool / "PLACED").read_text()); assert doc["version"] == 2 and not [e for e in w.ledger() if e["event"] == "placement_superseded"]
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2},{lw3}")
    assert r.returncode == 0 and "appended it" in (r.stdout + r.stderr), r.stdout + r.stderr
    sup = [e for e in w.ledger() if e["event"] == "placement_superseded"]
    assert len(sup) == 1 and sup[0]["from_version"] == 1 and sup[0]["to_version"] == 2 and sup[0]["reason"] == "pair/g012 on the second pass" and sup[0]["changes"]["kept"] == 2
    assert sup[0]["targets"] == doc["targets"] and sup[0]["members"] == doc["members"]
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2},{lw3}")
    assert r.returncode == 0 and len([e for e in w.ledger() if e["event"] == "placement_superseded"]) == 1 and len([e for e in w.ledger() if e["event"] == "placement_complete"]) == 1
    assert (w.state / "attempts.jsonl.lock").exists()                                                       # the ledger lock the check-and-append holds


def test_an_earlier_versions_missing_event_is_reconciled_before_a_refusal_and_before_a_supersession(tmp_path):
    """31b-fix-5: v1 commits, the process dies before its ledger append, and the target roster changes before
    the next walk — `refused` used to repair nothing and `--supersede` appended only v2, so the ledger lacked the supersession's baseline for
    good. Now the transaction returns every archived version and mark-placed reconciles each one's event (and, on a refusal, the live record's)
    before acting."""
    # the refused case: v1's placement_complete is appended before the refusal
    (tmp_path / "refused").mkdir(); w = World(tmp_path / "refused"); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    _members(w)
    lw1 = _target(w, "crescendo/g012", "0,1,2"); lw2 = _target(w, "crescendo/g34", "3,4"); lw3 = _target(w, "pair/g012", "0,1,2")
    pool = w.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}/attackers"
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}", SISTEER_TWIN_CRASH_AT="placement_after_remote")
    assert r.returncode == 97 and json.loads((pool / "PLACED").read_text())["version"] == 1 and not [e for e in w.ledger() if e["event"].startswith("placement_")], r.stdout + r.stderr
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2},{lw3}")                                                  # the roster changed: refused …
    assert r.returncode == 3 and "a placement is immutable" in r.stderr and "reconciled 1 missing ledger event(s) for earlier placement version(s) [1]" in (r.stdout + r.stderr), r.stdout + r.stderr
    ev = [e for e in w.ledger() if e["event"].startswith("placement_")]                                                              # … but v1's event is there now
    assert [(e["event"], e["version"]) for e in ev] == [("placement_complete", 1)] and [t["logical_work_id"] for t in ev[0]["targets"]] == [lw1, lw2]
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2},{lw3}")
    assert r.returncode == 3 and "reconciled" not in (r.stdout + r.stderr) and len([e for e in w.ledger() if e["event"].startswith("placement_")]) == 1   # idempotent
    # the supersession case: v1 then v2, in order; then v2's own window, reconciled by the v3 supersession
    (tmp_path / "sup").mkdir(); w = World(tmp_path / "sup"); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    _members(w)
    lw1 = _target(w, "crescendo/g012", "0,1,2"); lw2 = _target(w, "crescendo/g34", "3,4"); lw3 = _target(w, "pair/g012", "0,1,2")
    pool = w.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}/attackers"
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}", SISTEER_TWIN_CRASH_AT="placement_after_remote"); assert r.returncode == 97
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2},{lw3}", "--supersede", "--reason", "pair/g012 on the second pass")
    assert r.returncode == 0 and "reconciled 1 missing ledger event(s) for earlier placement version(s) [1]" in (r.stdout + r.stderr) and "SUPERSEDED version 1 -> 2" in (r.stdout + r.stderr), r.stdout + r.stderr
    ev = [e for e in w.ledger() if e["event"].startswith("placement_")]
    assert [(e["event"], e.get("version") or e.get("to_version")) for e in ev] == [("placement_complete", 1), ("placement_superseded", 2)]
    assert ev[1]["from_version"] == 1 and ev[1]["reason"] == "pair/g012 on the second pass" and (pool / "PLACED.v1").exists()
    lw4 = _target(w, "pair/g34", "3,4")
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2},{lw3},{lw4}", "--supersede", "--reason", "pair/g34 on the third pass", SISTEER_TWIN_CRASH_AT="placement_after_remote")
    assert r.returncode == 97 and json.loads((pool / "PLACED").read_text())["version"] == 3, r.stdout + r.stderr
    assert len([e for e in w.ledger() if e["event"].startswith("placement_")]) == 2                                              # v3's event missing
    lw5 = _target(w, "pair/g56", "5,6")
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2},{lw3},{lw4},{lw5}", "--supersede", "--reason", "pair/g56 on the fourth pass")
    assert r.returncode == 0 and "reconciled 1 missing ledger event(s) for earlier placement version(s) [1, 2, 3]" in (r.stdout + r.stderr), r.stdout + r.stderr
    ev = [e for e in w.ledger() if e["event"].startswith("placement_")]
    assert [(e["event"], e.get("version") or e.get("to_version")) for e in ev] == [("placement_complete", 1), ("placement_superseded", 2), ("placement_superseded", 3), ("placement_superseded", 4)]
    assert ev[2]["from_version"] == 2 and ev[2]["reason"] == "pair/g34 on the third pass" and ev[3]["from_version"] == 3
