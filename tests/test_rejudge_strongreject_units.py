"""The StrongREJECT store reader applies the graders' rules (scripts/rejudge/strongreject/enumerate_units.py): static variants keyed by
(goal, trial, sample_idx), multi-turn replies keyed by their position within the goal (the Sonnet twin's convention) with order and
attack stream kept, errored / refused goals dropped, a goal re-run in a later shard taken from the last shard, the best-of-N climb fold
replacing the climbed slots. Synthetic stores in tmp_path; synthetic goals."""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from rejudge.strongreject import enumerate_units as EU  # noqa: E402

GOALS = {"g1": {"dataset": "harmbench", "revision": "r", "text": "goal one"}, "g2": {"dataset": "harmbench", "revision": "r", "text": "goal two"},
         "g3": {"dataset": "jbb", "revision": "s", "text": "goal three"}}


def _write(path: pathlib.Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _setup(tmp_path, goal_sets):
    gp = tmp_path / "goals.json"
    gp.write_text(json.dumps({"goals": GOALS, "goal_sets": goal_sets}))
    EU._init(str(gp))
    return gp


def test_static_rows_keep_every_variant_and_trial(tmp_path):
    _setup(tmp_path, {"static": ["g1", "g2"], "mt": [], "bon": []})
    cell = tmp_path / "t0" / "codeattack" / "none"
    _write(cell / "shard00" / "responses.jsonl", [{"goal_id": "g1", "trial": 0, "sample_idx": i, "response": f"r{i}"} for i in range(3)]
           + [{"goal_id": "unknown", "trial": 0, "sample_idx": 0, "response": "x"}])
    cell1 = tmp_path / "t1" / "codeattack" / "none"
    _write(cell1 / "shard00" / "responses.jsonl", [{"goal_id": "g1", "trial": 1, "sample_idx": i, "response": f"r{i}"} for i in range(3)])
    job = {"run_id": "c:codeattack:i:x", "category": "static", "kind": "sr", "units": [{"type": "static", "path": str(cell)}, {"type": "static", "path": str(cell1)}]}
    rep, units, texts = EU.process(job)
    assert [(u[1], u[2], u[3], u[4]) for u in units] == [("g1", 0, i, -1) for i in range(3)] + [("g1", 1, i, -1) for i in range(3)]
    assert rep["unknown_goal"] == 1 and rep["collisions"] == 0
    assert len(texts) == 3          # the same three texts in both trials: three requests, six units
    assert units[0][5] == EU.cache_key("sr", "harmbench", "r", "goal one", "r0")


def test_multi_turn_rules(tmp_path):
    _setup(tmp_path, {"static": [], "mt": ["g1", "g2", "g3"], "bon": []})
    cell = tmp_path / "pr" / "actorattack" / "obj"
    # shard00: g1 with TWO streams restarting order at 0; g2 first attempt (re-run in shard01); g3 errored (dropped)
    _write(cell / "shard00" / "responses.jsonl",
           [{"goal_id": "g1", "order": o, "response": f"a{o}"} for o in range(3)] + [{"goal_id": "g1", "order": o, "response": f"b{o}"} for o in range(2)]
           + [{"goal_id": "g2", "order": o, "response": f"old{o}"} for o in range(2)] + [{"goal_id": "g3", "order": 0, "response": "e"}])
    _write(cell / "shard01" / "responses.jsonl", [{"goal_id": "g2", "order": o, "response": f"new{o}"} for o in range(2)])
    files = sorted(str(p) for p in cell.glob("shard*/responses.jsonl"))
    job = {"run_id": "c:actorattack:i:x", "category": "mt", "kind": "sr", "units": [{"type": "mt", "path": str(cell), "files": files, "drop_goals": ["g3"]}]}
    rep, units, texts = EU.process(job)
    g1 = [u for u in units if u[1] == "g1"]
    g2 = [u for u in units if u[1] == "g2"]
    assert not [u for u in units if u[1] == "g3"] and rep["dropped_goal_rows"] == 1
    # every g1 reply judged, keyed by its position within the goal; order and stream kept in the note
    assert [(u[3], u[4], u[8]) for u in g1] == [(0, -1, "order=0;stream=0"), (1, -1, "order=1;stream=0"), (2, -1, "order=2;stream=0"),
                                                (3, -1, "order=0;stream=1"), (4, -1, "order=1;stream=1")]
    # g2 re-run in shard01: only the last shard's replies (the grader's dict.update merge)
    assert [texts[u[5]][4] for u in g2] == ["new0", "new1"]
    assert rep["collisions"] == 0


def test_bon_climb_fold_replaces_climbed_slots(tmp_path):
    _setup(tmp_path, {"static": [], "mt": [], "bon": ["g3"]})
    store = tmp_path / "bon_responses.jsonl"
    _write(store, [{"goal_id": "g3", "trial": 0, "sample_idx": i, "response": f"d{i}"} for i in range(4)])
    climb = tmp_path / "out" / "cell" / "shard00.jsonl"
    _write(climb, [{"goal_id": "g3", "trial": 0, "sample_idx": 2, "response": "redraw2"}])
    job = {"run_id": "c:strongreject_bon:i:x", "category": "bon", "kind": "sr2", "units": [],
           "fold": {"store": [{"type": "bon", "path": str(store), "files": [str(store)]}], "climb": [{"type": "climb", "path": str(climb.parent), "files": [str(climb)]}]}}
    rep, units, texts = EU.process(job)
    by_sample = {u[3]: texts[u[5]][4] for u in units}
    assert by_sample == {0: "d0", 1: "d1", 2: "redraw2", 3: "d3"} and rep["climb_replaced"] == 1
    assert all(u[6] == "sr2" for u in units)
