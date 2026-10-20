"""scripts/fst_exp2_goals.py — experiment 2's fixed, stratified 10-goal subset of
the 87 goals. Synthetic goal ids and a synthetic
aggregate reproducing the 41 / 14 / 30 / 2 strata; the ONE read-only look at
the real aggregate (sha + histogram + counts) is skipped when it is absent.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import fst_exp2_goals as fg  # noqa: E402
import jailbreak_run  # noqa: E402

S = fg.STRATUM_ORDER


def _full_ids():
    return [f"g{i:03d}" for i in range(259)]


def _world(counts=(41, 14, 30, 2)):
    """Synthetic best scores + groups over the §15 slice reproducing the given
    stratum counts (in STRATUM_ORDER)."""
    full = _full_ids()
    s15 = fg.s15_subset(full)
    assert sum(counts) == len(s15)
    best, group = {}, {}
    i = 0
    for stratum, n in zip(S, counts):
        grp, b = stratum.split("|")
        for _ in range(n):
            gid = s15[i]; i += 1
            group[gid] = grp
            best[gid] = (1.0 if i % 2 else 0.875) if b == fg.BIN_GT else 0.75
    # every goal outside the slice has a group too (the real group_of covers all 259)
    for gid in full:
        group.setdefault(gid, "held_in")
    return full, s15, best, group


def test_s15_subset_is_the_wave_slice():
    full = _full_ids()
    s15 = fg.s15_subset(full)
    assert len(s15) == 87 and s15 == [g for i, g in enumerate(full) if i % 9 in (0, 1, 2)]
    assert set(s15) <= set(full)
    with pytest.raises(fg.GoalSelectionError):
        fg.s15_subset(full[:-1])


def test_allocation_is_floor_first_then_largest_remainder():
    assert fg.allocate({S[0]: 41, S[1]: 14, S[2]: 30, S[3]: 2}) == \
        {S[0]: 4, S[1]: 2, S[2]: 3, S[3]: 1}
    # plain largest-remainder would give 5/2/3/0 — the floor is what puts the
    # seat on held_out<=0.75 and takes it from held_in>0.75
    with pytest.raises(fg.GoalSelectionError, match="empty stratum"):
        fg.allocate({S[0]: 43, S[1]: 14, S[2]: 30, S[3]: 0})
    with pytest.raises(fg.GoalSelectionError, match="floor"):
        fg.allocate({S[0]: 1, S[1]: 1, S[2]: 1, S[3]: 1}, k=3)
    # a stratum can never receive more seats than goals
    with pytest.raises(fg.GoalSelectionError, match="exceeds"):
        fg.allocate({S[0]: 1, S[1]: 1, S[2]: 1, S[3]: 1}, k=10)
    # ties go to the earlier stratum in STRATUM_ORDER (exact fractions)
    assert fg.allocate({s: 10 for s in S}) == {S[0]: 3, S[1]: 3, S[2]: 2, S[3]: 2}


def test_select_goals_is_deterministic_and_stratified():
    full, s15, best, group = _world()
    a = fg.select_goals(full_goal_ids=full, best_score=best, group_of=group)
    b = fg.select_goals(full_goal_ids=full, best_score=best, group_of=group)
    assert a == b
    assert len(a["goal_ids"]) == 10 == len(set(a["goal_ids"]))
    assert set(a["goal_ids"]) <= set(s15) <= set(full)
    assert a["stratum_counts"] == fg.EXPECTED_STRATUM_COUNTS
    assert a["allocation"] == fg.EXPECTED_ALLOCATION == {S[0]: 4, S[1]: 2, S[2]: 3, S[3]: 1}
    for s in S:
        assert len(a["by_stratum"][s]) == a["allocation"][s]
        for gid in a["by_stratum"][s]:
            assert fg.stratum_key(group[gid], best[gid]) == s
            assert a["stratum_of"][gid] == s
    # file order = stratum order
    assert a["goal_ids"] == [g for s in S for g in a["by_stratum"][s]]
    # within a stratum: sorted, shuffled with a fresh Random(0), first k
    import random
    ids = sorted(g for g in s15 if fg.stratum_key(group[g], best[g]) == S[0])
    random.Random(0).shuffle(ids)
    assert a["by_stratum"][S[0]] == ids[:4]
    # a different seed picks differently but the same strata
    c = fg.select_goals(full_goal_ids=full, best_score=best, group_of=group, seed=1)
    assert c["goal_ids"] != a["goal_ids"] and c["allocation"] == a["allocation"]


def test_select_goals_refuses_a_changed_baseline():
    full, s15, best, group = _world()
    # one goal crosses the predicate: 41/14 -> 40/15
    moved = next(g for g in s15 if group[g] == "held_in" and best[g] > 0.75)
    best2 = {**best, moved: 0.75}
    with pytest.raises(fg.GoalSelectionError, match="observed stratum counts"):
        fg.select_goals(full_goal_ids=full, best_score=best2, group_of=group)
    # an empty stratum (every held_out goal above the predicate)
    best3 = {g: (1.0 if group[g] == "held_out" else best[g]) for g in best}
    with pytest.raises(fg.GoalSelectionError, match="observed stratum counts"):
        fg.select_goals(full_goal_ids=full, best_score=best3, group_of=group)
    with pytest.raises(fg.GoalSelectionError, match="empty stratum"):
        fg.select_goals(full_goal_ids=full, best_score=best3, group_of=group,
                        expected_counts=None, expected_allocation=None)
    # the aggregate's per-goal set is not the slice
    with pytest.raises(fg.GoalSelectionError, match="not the §15 slice"):
        fg.select_goals(full_goal_ids=full, best_score={**best, "g003": 1.0}, group_of=group)
    with pytest.raises(fg.GoalSelectionError, match="not the §15 slice"):
        fg.select_goals(full_goal_ids=full, best_score={k: v for k, v in best.items()
                                                        if k != s15[0]}, group_of=group)
    # the predicate is STRICT >: 0.75 is the low bin
    assert fg.stratum_key("held_in", 0.75) == S[1] and fg.stratum_key("held_in", 0.875) == S[0]
    with pytest.raises(fg.GoalSelectionError):
        fg.stratum_key("elsewhere", 1.0)


def test_sha_pin_and_histogram_pin(tmp_path):
    doc = {"defenses": {"none": {"per_goal": {
        f"g{i}": {"best_score": v} for i, v in enumerate([1.0] * 53 + [0.875] * 18 + [0.75] * 16)}}}}
    p = tmp_path / "agg.json"
    p.write_text(json.dumps(doc))
    sha = hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(fg.GoalSelectionError, match="sha256"):
        fg.load_aggregate(p)                       # not the pinned baseline
    assert fg.load_aggregate(p, expect_sha256=sha) == doc
    scores = fg.none_best_scores(doc)
    assert len(scores) == 87
    bad = {"defenses": {"none": {"per_goal": {"g0": {"best_score": 0.5}}}}}
    with pytest.raises(fg.GoalSelectionError, match="histogram"):
        fg.none_best_scores(bad)
    with pytest.raises(fg.GoalSelectionError, match="per_goal"):
        fg.none_best_scores({"defenses": {}})


def test_ids_file_replays_in_file_order(tmp_path):
    full, s15, best, group = _world()
    rec = fg.select_goals(full_goal_ids=full, best_score=best, group_of=group)
    text = fg.ids_file_text(rec, aggregate_sha256=fg.AGGREGATE_SHA256, config="cfg.yaml")
    f = tmp_path / fg.IDS_FILE
    f.write_text(text)
    assert fg.read_ids_file(f) == rec["goal_ids"]
    # exactly what the driver's --goal-ids-file reader sees, in this order
    goals = [SimpleNamespace(goal_id=g, text="t", category="c") for g in full]
    sel = jailbreak_run._select_goal_ids(goals, f)
    assert [g.goal_id for g in sel] == rec["goal_ids"]
    # 5 subshards -> 2 goals each
    assert [len(sel[k::5]) for k in range(5)] == [2] * 5
    # the header names the rule, the predicate, the counts, the allocation, the sha
    head = [l for l in text.splitlines() if l.startswith("#")]
    joined = "\n".join(head)
    assert fg.AGGREGATE_SHA256 in joined and fg.PREDICATE in joined
    assert "held_in|>0.75=41" in joined and "held_out|<=0.75=1" in joined
    # ids and numbers only: no free text beyond the header's fixed prose
    for line in text.splitlines():
        if not line.startswith("#"):
            assert " " not in line


def test_categories_from_progress_and_holdout_check(tmp_path):
    root = tmp_path / "none"
    for k, rows in enumerate(([{"goal_id": "g000", "category": "A"}],
                              [{"goal_id": "g001", "category": "B"}])):
        d = root / f"shard{k:02d}"
        d.mkdir(parents=True)
        (d / "progress.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows)
                                          + '{"goal_id": "g9", "categ')       # torn tail
    assert fg.categories_from_progress(root) == {"g000": "A", "g001": "B"}
    with pytest.raises(fg.GoalSelectionError, match="no shard"):
        fg.categories_from_progress(tmp_path / "empty")
    full, s15, best, group = _world()
    held_in = [f"in{i}" for i in range(11)]
    held_out = [f"out{i}" for i in range(5)]
    agg = {"category_holdout": {"held_in": held_in, "held_out": held_out, "seed": 0, "frac": 0.33}}
    info = fg.check_holdout(agg, held_in, held_out, group, s15)
    assert info["held_in_goals"] == 55 and info["held_out_goals"] == 32 and info["seed"] == 0
    with pytest.raises(fg.GoalSelectionError, match="differs from the aggregate"):
        fg.check_holdout(agg, held_in[:-1], held_out + [held_in[-1]], group, s15)


@pytest.mark.skipif(
    not (Path.home() / "heldout_gated_grade/wave_a328caf/crescendo_aggregate.json").is_file(),
    reason="the §15 aggregate is not on this machine (read-only input)")
def test_real_aggregate_matches_the_pins():
    """READ-ONLY: the pinned sha, the none best_score histogram and the 2x2
    counts on the real §15 aggregate + the wave's own categories."""
    root = Path.home() / "heldout_gated_grade/wave_a328caf"
    agg = fg.load_aggregate(root / "crescendo_aggregate.json")
    best = fg.none_best_scores(agg)
    cats = fg.categories_from_progress(root / "two_si_v1/crescendo/none")
    held_out = set(agg["category_holdout"]["held_out"])
    group = {g: ("held_out" if cats[g] in held_out else "held_in") for g in best}
    strata = fg.stratify(sorted(best), best, group)
    assert {s: len(v) for s, v in strata.items()} == fg.EXPECTED_STRATUM_COUNTS
    assert fg.allocate({s: len(v) for s, v in strata.items()}) == fg.EXPECTED_ALLOCATION


# --------------------------------------------------------------------------- #
# the TRACKED pair under configs/fst/ (the run identity points at it)          #
# --------------------------------------------------------------------------- #

TRACKED_IDS = REPO / fg.TRACKED_DIR / fg.IDS_FILE
TRACKED_SEL = REPO / fg.TRACKED_DIR / fg.SELECTION_FILE


def test_tracked_goal_files_are_consistent():
    """Offline the selector cannot be re-run (it needs the HF cache and the
    read-only aggregate), so pin the tracked pair's INTERNAL consistency: the
    ids file has exactly 10 non-comment lines, unique, equal to the selection
    JSON's ids in order; the JSON records the file's sha256, the pinned
    aggregate sha, the plan's counts and allocation, and a stratum per id."""
    assert TRACKED_IDS.is_file() and TRACKED_SEL.is_file()
    ids = fg.read_ids_file(TRACKED_IDS)
    assert len(ids) == fg.N_GOALS == 10 and len(set(ids)) == 10
    sel = json.loads(TRACKED_SEL.read_text())
    assert sel["schema"] == "fst_exp2_goal_selection/1"
    assert sel["goal_ids"] == ids
    assert sel["ids_file_sha256"] == hashlib.sha256(TRACKED_IDS.read_bytes()).hexdigest()
    assert sel["aggregate"]["sha256"] == fg.AGGREGATE_SHA256
    assert fg.AGGREGATE_SHA256 in TRACKED_IDS.read_text()
    assert sel["stratum_counts"] == fg.EXPECTED_STRATUM_COUNTS
    assert sel["allocation"] == fg.EXPECTED_ALLOCATION
    assert sel["k"] == 10 and sel["n_s15"] == 87 and sel["n_confirmatory"] == 259
    assert sel["predicate"] == fg.PREDICATE and sel["shuffle_seed"] == fg.SHUFFLE_SEED
    assert [g for s in fg.STRATUM_ORDER for g in sel["by_stratum"][s]] == ids
    for s in fg.STRATUM_ORDER:
        assert len(sel["by_stratum"][s]) == fg.EXPECTED_ALLOCATION[s]
        for gid in sel["by_stratum"][s]:
            assert sel["stratum_of"][gid] == s
    assert set(sel["stratum_of"]) == set(ids)
    # ids and numbers only: an id is a single token; no free text in the JSON's string values
    for gid in ids:
        assert " " not in gid and "\t" not in gid
    for v in sel["stratum_of"].values():
        assert v in fg.STRATUM_ORDER
    # the driver's reader sees the same order (file order = stratum order)
    goals = [SimpleNamespace(goal_id=g, text="t", category="c") for g in ids]
    assert [g.goal_id for g in jailbreak_run._select_goal_ids(goals, TRACKED_IDS)] == ids
