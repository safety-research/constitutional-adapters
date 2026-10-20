"""scripts/sdfbase/full22/mt_degeneracy_union.py — the canonical 4,096 multi-turn degeneracy pass under the union classifier.
The union itself lives in the degeneracy-union worktree (not on main); these tests pin the driver's PURE helpers: the lane view
-> roster target rule, the grades-root -> transcripts-root rule, the merged-shard winner rule (last progress row per goal wins and
that goal's responses come from the winning shard only, target_timing joined by position), and the goal-level readings."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "scripts/sdfbase/full22/mt_degeneracy_union.py"
KEEP = {"base_target_of", "transcripts_root_of", "transcripts_root_candidates", "resolve_transcripts_root", "shard_order", "winning_shards", "load_store_rows", "_share", "goal_level", "Refuse"}


def _load():
    tree = ast.parse(SRC.read_text()); body = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in KEEP:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("_SHARD_RE", "CAP", "SCHEMA", "ATTACKS", "VIEW_SUFFIXES") for t in node.targets):
            body.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            body.append(node)
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
              and ast.unparse(node.value.func.value) == "sys.path"):
            body.append(node)     # the script's own import path (scripts/, for site_paths): the file passes alone, in any order
    mod = ast.Module(body=body, type_ignores=[]); ast.fix_missing_locations(mod)
    ns: dict = {"__name__": "mt_degeneracy_union_under_test", "__file__": str(SRC)}; exec(compile(mod, str(SRC), "exec"), ns); return ns


@pytest.fixture(scope="module")
def m():
    return _load()


def test_view_to_target_and_grades_to_transcripts_roots(m):
    assert m["base_target_of"]("gemma4e4b_it") == "gemma4e4b" and m["base_target_of"]("llama8b_shortconst") == "llama8b" and m["base_target_of"]("qwen35_4b") == "qwen35_4b"
    assert m["transcripts_root_of"]("grades_cap4096_qwen35_4b") == "transcripts_cap4096_qwen35_4b"
    assert m["transcripts_root_of"]("relu_native/grades_relu_native_bu_gemma4e4b") == "relu_native/transcripts_relu_native_bu_gemma4e4b"
    with pytest.raises(SystemExit):
        m["transcripts_root_of"]("transcripts_cap4096_qwen35_4b")          # not a grades root


def test_shard_order_puts_base_shards_before_reruns(m):
    names = ["shard_r3_00", "shard01", "shard_r2_07", "shard00", "shard_r2_00"]
    assert sorted(names, key=m["shard_order"]) == ["shard00", "shard01", "shard_r2_00", "shard_r2_07", "shard_r3_00"]


def _store(tmp_path, shards):
    """shards: {name: (progress rows, response rows, timing rows or None)}"""
    st = tmp_path / "store"
    for name, (prog, resp, timing) in shards.items():
        d = st / name; d.mkdir(parents=True)
        (d / "progress.jsonl").write_text("\n".join(json.dumps(r) for r in prog) + "\n")
        (d / "responses.jsonl").write_text("\n".join(json.dumps(r) for r in resp) + "\n")
        if timing is not None:
            (d / "target_timing.jsonl").write_text("\n".join(json.dumps(r) for r in timing) + "\n")
    return st


def test_merged_root_last_progress_row_wins_and_responses_follow_the_winner(m, tmp_path):
    st = _store(tmp_path, {
        "shard00": ([{"goal_id": "g1", "capped": False}, {"goal_id": "g2", "capped": True}],
                    [{"goal_id": "g1", "order": 0, "response": "a1"}, {"goal_id": "g2", "order": 0, "response": "b1 old"}],
                    [{"finish_reason": "stop", "completion_tokens": 3}, {"finish_reason": "length", "completion_tokens": 4096}]),
        "shard_r2_00": ([{"goal_id": "g2", "capped": False}],
                        [{"goal_id": "g2", "order": 0, "response": "b1 new"}, {"goal_id": "g2", "order": 1, "response": "b2 new"}],
                        None),                                              # no timing file: finish_reason undefined here
    })
    win = m["winning_shards"](st)
    assert win["g1"].name == "shard00" and win["g2"].name == "shard_r2_00"
    rows, progress, facts = m["load_store_rows"](st)
    texts = sorted(r["response"] for _, r in rows)
    assert texts == ["a1", "b1 new", "b2 new"]                             # g2's old shard00 response is NOT read
    assert progress["g2"] == {"capped": False, "outcome": "None", "turns": None, "n_target_forwards": None}   # the LAST progress row
    g1 = next(r for _, r in rows if r["goal_id"] == "g1")
    assert g1["finish_reason"] == "stop" and g1["at_cap"] is False and g1["completion_tokens"] == 3 and g1["sample_idx"] == 0
    assert all("finish_reason" not in r for _, r in rows if r["goal_id"] == "g2")     # unaligned shard: estimand undefined
    assert facts["shard00"]["aligned"] is True and facts["shard_r2_00"]["aligned"] is False


def test_timing_join_requires_equal_row_counts(m, tmp_path):
    st = _store(tmp_path, {"shard00": ([{"goal_id": "g1"}], [{"goal_id": "g1", "order": 0, "response": "x"}, {"goal_id": "g1", "order": 1, "response": "y"}],
                                       [{"finish_reason": "length"}])})       # 2 responses, 1 timing row -> not aligned
    rows, _, facts = m["load_store_rows"](st)
    assert len(rows) == 2 and all("finish_reason" not in r for _, r in rows) and facts["shard00"]["aligned"] is False


def test_goal_level_any_turn_and_defended_split(m):
    per_row = [
        {"goal_id": "g1", "degenerate": False, "at_cap_retokenized": False, "at_cap_finish_reason": False, "rep_ratio_loop": False},
        {"goal_id": "g1", "degenerate": True, "at_cap_retokenized": True, "at_cap_finish_reason": True, "rep_ratio_loop": False},
        {"goal_id": "g2", "degenerate": True, "at_cap_retokenized": False, "at_cap_finish_reason": False, "rep_ratio_loop": True},
        {"goal_id": "g3", "degenerate": False, "at_cap_retokenized": False, "at_cap_finish_reason": False, "rep_ratio_loop": False},
    ]
    progress = {"g1": {"capped": True}, "g2": {"capped": False}, "g3": {"capped": False}}
    gl = m["goal_level"](per_row, progress, {"g1": False, "g2": True, "g3": False})
    assert gl["n_goals"] == 3 and gl["canonical_at_cap"] == "at_cap_retokenized_any_turn"
    assert gl["degenerate_any_turn"]["k"] == 2 and gl["at_cap_retokenized_any_turn"]["k"] == 1 and gl["loop_any_turn"]["k"] == 1
    assert gl["at_cap_finish_reason_any_turn"]["k"] == 1 and gl["producer_capped_flag"]["k"] == 1
    j = gl["judged"]; assert j["jailbroken"]["k"] == 1 and j["defended"]["k"] == 2
    assert j["defended_any_turn_degenerate"] == {"k": 1, "n": 2, "p": 0.5} and j["jailbroken_any_turn_degenerate"]["k"] == 1
    # a goal with no finish_reason on some turn makes that estimand undefined at goal level
    per_row2 = per_row + [{"goal_id": "g4", "degenerate": False, "at_cap_retokenized": False, "at_cap_finish_reason": None, "rep_ratio_loop": False}]
    assert m["goal_level"](per_row2, progress, {})["at_cap_finish_reason_any_turn"] == {"defined": False}


def test_transcripts_root_resolves_without_the_lanes_view_suffix(m, tmp_path):
    """The grades roots carry the view suffix (grades_obj4096_gemma4e4b_it); their transcripts roots do not
    (transcripts_obj4096_gemma4e4b) — and the convention is not uniform (grades_anchor4096r2_qwen35_4b matches directly), so
    the resolver tries the direct name first and derives the stripped one (2026-09-17: 19 of 20 'pending' cells)."""
    assert m["transcripts_root_candidates"]("grades_obj4096_gemma4e4b_it") == ["transcripts_obj4096_gemma4e4b_it", "transcripts_obj4096_gemma4e4b"]
    assert m["transcripts_root_candidates"]("grades_mt4kscr2_ch_llama8b_shortconst") == ["transcripts_mt4kscr2_ch_llama8b_shortconst", "transcripts_mt4kscr2_ch_llama8b"]
    assert m["transcripts_root_candidates"]("grades_anchor4096r2_qwen35_4b") == ["transcripts_anchor4096r2_qwen35_4b"]
    assert m["transcripts_root_candidates"]("relu_native/grades_relu_native_bu_gemma4e4b") == ["relu_native/transcripts_relu_native_bu_gemma4e4b"]
    rs = tmp_path; (rs / "transcripts_obj4096_gemma4e4b").mkdir(); (rs / "transcripts_mt4kscr2_ch_llama8b_shortconst").mkdir()
    root, tried = m["resolve_transcripts_root"](rs, "grades_obj4096_gemma4e4b_it")
    assert root == rs / "transcripts_obj4096_gemma4e4b" and tried[0] == "transcripts_obj4096_gemma4e4b_it"     # direct tried first, stripped won
    root2, _ = m["resolve_transcripts_root"](rs, "grades_mt4kscr2_ch_llama8b_shortconst")
    assert root2 == rs / "transcripts_mt4kscr2_ch_llama8b_shortconst"                                            # direct exists: never stripped
    assert m["resolve_transcripts_root"](rs, "grades_anchor4096_qwen35_4b_it") == (None, ["transcripts_anchor4096_qwen35_4b_it", "transcripts_anchor4096_qwen35_4b"])
