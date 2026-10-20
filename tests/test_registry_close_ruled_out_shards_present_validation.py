"""A run whose remaining shards are ruled out is cancelled ONLY after every PRESENT shard
passed the full multi-turn inspection — its own goal assignment (jailbreak_run deals goals[shard::nshards]), graded outcomes, a response per goal,
no torn lines, no goal in two shards. The landed closer-fix cancelled as soon as the missing names were ruled out, counting only parseable rows
with responses; a torn response or an unfinished / refused goal in shard00 beside ruled-out shards 05-08 would have been cancelled with a
plausible partial count and never retried. Reuses the landed ruled-out test's world (the real closer over the heavy multi-turn fixtures)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_registry_close_ruled_out_shards import RULED, _close, _multiturn_store, _records, _runs, _world  # noqa: E402

PRESENT = [0, 1, 2, 3, 4]


def _not_cancelled(spool, rid):
    return not [p for t, p, *_ in _records(spool, rid) if t == "run_state" and p.get("state") == "cancelled"]


def test_an_incomplete_present_shard_beside_ruled_out_missing_shards_keeps_the_run_running(tmp_path, monkeypatch):
    root, spool, wave, pending, prun, real_ids, IDS = _world(tmp_path, monkeypatch)
    rid = _runs(pending)[("crescendo", "none")]["run_id"]
    # (1) an error outcome the grader would not grade in shard00
    _multiturn_store(wave, prun, "crescendo", "none", nshards=9, goals=259, shards=PRESENT, goal_ids=real_ids, error_goal_in_shard=0)
    r = _close(pending, spool, wave, *IDS, *RULED); out = r.stdout + r.stderr
    assert "cancelled" not in out and "progress row(s) the grader would not grade" in out and "the run stays running" in out, out
    assert _not_cancelled(spool, rid)
    # (2) a goal without a response row in shard00
    _multiturn_store(wave, prun, "crescendo", "none", nshards=9, goals=259, shards=PRESENT, goal_ids=real_ids, drop_response_goal_in_shard=0)
    r = _close(pending, spool, wave, *IDS, *RULED); out = r.stdout + r.stderr
    assert "cancelled" not in out and "carry no response row" in out, out
    # (3) a goal of shard01 duplicated into shard00: not shard00's assignment
    _multiturn_store(wave, prun, "crescendo", "none", nshards=9, goals=259, shards=PRESENT, goal_ids=real_ids, duplicate_goal_into_shard=0)
    r = _close(pending, spool, wave, *IDS, *RULED); out = r.stdout + r.stderr
    assert "cancelled" not in out and ("does not hold its assignment goals[0::9]" in out or "appear in two shards" in out), out
    # (4) a goal of a RULED-OUT shard (shard05 = goals[5::9]) written into shard00: outside shard00's assignment
    store = _multiturn_store(wave, prun, "crescendo", "none", nshards=9, goals=259, shards=PRESENT, goal_ids=real_ids)
    stray = real_ids[5]
    with open(store / "shard00" / "progress.jsonl", "a") as fh:
        fh.write(json.dumps({"goal_id": stray, "category": "x", "n_target_forwards": 3, "outcome": "AttackOutcome.FAILURE", "turns": 3, "capped": False, "served_window": 49152}) + "\n")
    with open(store / "shard00" / "responses.jsonl", "a") as fh:
        fh.write(json.dumps({"goal_id": stray, "order": 0, "response": "x"}) + "\n")
    r = _close(pending, spool, wave, *IDS, *RULED); out = r.stdout + r.stderr
    assert "cancelled" not in out and "shard00 does not hold its assignment goals[0::9]" in out and f"1 outside it: {stray}" in out, out
    # (5) a torn progress line in shard00
    store = _multiturn_store(wave, prun, "crescendo", "none", nshards=9, goals=259, shards=PRESENT, goal_ids=real_ids)
    with open(store / "shard00" / "progress.jsonl", "a") as fh:
        fh.write('{"goal_id": "torn", "outc')
    r = _close(pending, spool, wave, *IDS, *RULED); out = r.stdout + r.stderr
    assert "cancelled" not in out and "unparsable line(s)" in out, out
    assert _not_cancelled(spool, rid)
    # (6) the same world, clean: cancels with the present shards' assignment (145 goals) — the inspection's own counts
    _multiturn_store(wave, prun, "crescendo", "none", nshards=9, goals=259, shards=PRESENT, goal_ids=real_ids)
    r = _close(pending, spool, wave, *IDS, *RULED); out = r.stdout + r.stderr
    assert "1 cancelled (shards ruled out; D1168)" in out and "n_items=145/259" in out, out
    st = [p for t, p, *_ in _records(spool, rid) if t == "run_state" and p.get("state") == "cancelled"]
    assert len(st) == 1 and '"shards_closed": ["shard00", "shard01", "shard02", "shard03", "shard04"]' in st[0]["note"] and '"goals_per_shard": [29, 29, 29, 29, 29]' in st[0]["note"], st[0]["note"]


def test_a_partial_close_without_the_registered_goal_set_refuses_by_name(tmp_path, monkeypatch):
    root, spool, wave, pending, prun, real_ids, IDS = _world(tmp_path, monkeypatch)
    _multiturn_store(wave, prun, "crescendo", "none", nshards=9, goals=259, shards=PRESENT, goal_ids=real_ids)
    # inspect_multiturn directly: tolerated missing shards but no expected ids -> a refusal naming the need, never a partial pass
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "sdfbase" / "full22"))
    import registry_close_stores as RC
    store = str(wave / "transcripts_results" / prun / "crescendo" / "none")
    why, facts, art = RC.inspect_multiturn(store, planned=259, nshards=9, expected_ids=None, tolerated_missing=["shard05", "shard06", "shard07", "shard08"])
    assert why and "needs the registered goal set" in why and "partial" not in facts, why
    why, facts, art = RC.inspect_multiturn(store, planned=259, nshards=9, expected_ids=real_ids, tolerated_missing=["shard05", "shard06", "shard07", "shard08"])
    assert why is None and facts["partial"] == {"shards_missing_ruled_out": ["shard05", "shard06", "shard07", "shard08"], "shards_closed": ["shard00", "shard01", "shard02", "shard03", "shard04"]}
    assert facts["goals_distinct"] == 145 and facts["goals_expected"] == 145 and facts["goals_per_shard"] == [29, 29, 29, 29, 29] and art.endswith("shard00/progress.jsonl"), facts
    # shards merely not yet run (shard02 absent, not tolerated) still refuse as before
    why, facts, art = RC.inspect_multiturn(store, planned=259, nshards=9, expected_ids=real_ids, tolerated_missing=["shard06"])
    assert why and "expected shard(s) missing or incomplete: shard05, shard06, shard07, shard08" in why, why
