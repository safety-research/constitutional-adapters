"""A multi-turn response row counts for a goal only in the goal's OWN shard. jailbreak_run's drivers write a
shard's progress.jsonl and responses.jsonl side by side and the grader reads each shard's pair; the closer tallied responses across the shards,
so a row filed under another shard stood in for a missing reply and a row for a goal no shard holds passed unnoticed. Now: a goal's own shard must
hold its response row(s) (`carry no response row`, the existing wording) and a response row whose shard's progress holds no row for its goal is an
orphan refused by name; the facts carry the per-shard response counts. Synthetic store (tests/test_registry_close_stores._multiturn_store)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402
from test_registry_close_stores import _multiturn_store  # noqa: E402

sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
import registry_close_stores as RC  # noqa: E402

GOALS = 259
IDS = [f"goal_{g:03d}" for g in range(GOALS)]


def _move_rows(store: Path, gid: str, src: str, dst: str) -> int:
    s = store / src / "responses.jsonl"; d = store / dst / "responses.jsonl"
    rows = s.read_text().splitlines(); mine = [r for r in rows if json.loads(r)["goal_id"] == gid]
    s.write_text("".join(r + "\n" for r in rows if json.loads(r)["goal_id"] != gid)); d.write_text(d.read_text() + "".join(r + "\n" for r in mine))
    return len(mine)


def test_a_response_row_counts_only_in_its_goals_own_shard(tmp_path):
    store = _multiturn_store(tmp_path / "wave", "prun", "crescendo", "none", nshards=9, goals=GOALS)
    why, facts, art = RC.inspect_multiturn(str(store), planned=GOALS, nshards=9, expected_ids=IDS)
    assert why is None and art.endswith("shard00/progress.jsonl"), why
    assert facts["responses"] == GOALS * 3 and facts["responses_per_shard"] == [29 * 3] * 7 + [28 * 3] * 2 and facts["response_rows_outside_their_shards_progress"] == 0
    assert facts["goals_without_response"] == 0
    # goal_007's three rows (shard07) filed under shard08: the global tally is unchanged (259 x 3) — the old reading saw nothing wrong
    g = "goal_007"; moved = _move_rows(store, g, "shard07", "shard08"); assert moved == 3
    why, facts, _a = RC.inspect_multiturn(str(store), planned=GOALS, nshards=9, expected_ids=IDS)
    assert why and "3 response row(s) for 1 goal(s) sit in a shard whose progress holds no row for their goal" in why, why
    assert f"shard08:{g} (progress in shard07)" in why and "a response counts only in its goal's own shard" in why, why
    assert facts["responses"] == GOALS * 3 and facts["response_rows_outside_their_shards_progress"] == 3 and facts["goals_without_response"] == 1
    assert facts["responses_per_shard"][7] == 27 * 3 and facts["responses_per_shard"][8] == 29 * 3
    # moved back: complete again
    assert _move_rows(store, g, "shard08", "shard07") == 3
    why, facts, _a = RC.inspect_multiturn(str(store), planned=GOALS, nshards=9, expected_ids=IDS)
    assert why is None and facts["response_rows_outside_their_shards_progress"] == 0, why
    # a response row for a goal NO shard holds in its progress: an orphan, named as such
    with open(store / "shard00" / "responses.jsonl", "a") as fh:
        fh.write(json.dumps({"goal_id": "goal_999", "order": 0, "response": "stray", "serving_engine": "lora_native"}) + "\n")
    why, facts, _a = RC.inspect_multiturn(str(store), planned=GOALS, nshards=9, expected_ids=IDS)
    assert why and "1 response row(s) for 1 goal(s) sit in a shard" in why and "shard00:goal_999 (no progress row in any shard)" in why, why
    assert facts["response_rows_outside_their_shards_progress"] == 1 and facts["goals_without_response"] == 0 and facts["responses"] == GOALS * 3 + 1


def test_a_goal_whose_rows_were_dropped_keeps_the_existing_wording_and_orphans_are_named_first(tmp_path):
    store = _multiturn_store(tmp_path / "wave", "prun", "pair", "none", nshards=9, goals=GOALS, drop_response_goal_in_shard=7)
    g7 = [gid for g, gid in enumerate(IDS) if g % 9 == 7][0]
    why, facts, _a = RC.inspect_multiturn(str(store), planned=GOALS, nshards=9, expected_ids=IDS)
    assert why and f"1 goal(s) carry no response row (e.g. {g7})" in why and facts["goals_without_response"] == 1 and facts["response_rows_outside_their_shards_progress"] == 0, why
    # both at once: the orphan refusal comes first (it explains the missing coverage), the facts carry both
    store2 = _multiturn_store(tmp_path / "wave2", "prun", "pair", "none", nshards=9, goals=GOALS)
    _move_rows(store2, "goal_003", "shard03", "shard04")
    why, facts, _a = RC.inspect_multiturn(str(store2), planned=GOALS, nshards=9, expected_ids=IDS)
    assert why and "sit in a shard whose progress holds no row" in why and "carry no response row" not in why, why
    assert facts["goals_without_response"] == 1 and facts["response_rows_outside_their_shards_progress"] == 3
