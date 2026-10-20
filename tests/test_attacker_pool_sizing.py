"""The multi-turn attacker pool is sized from the targets that will be CONCURRENTLY live on its cluster — one member per three targets,
minimum one, `SISTEER_MT_ATTACKER_GPUS` a CAP — and a composition whose pool:target ratio exceeds 1:3 refuses by name.

Measured on the u0 wave: four identical replicas of the 27B attacker per cluster served 1–4 targets at a time (typically 1–2), each member
under one request in flight on average and idle ~62 % of the time, a 14-minute window with no consumer at all — four GPUs per cluster held
for a service one member could carry. The plan constant was four regardless of the target count."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import stage as ST  # noqa: E402

KEY = "qwen35_4b_demo"
SHA = "a" * 40
AB = ["cluster_a", "cluster_b"]


def _pool(rows, label):
    return sorted(r["unit"] for r in rows if r["kind"] == "attacker" and r["cluster"] == label)


def _targets(rows, label):
    return [r for r in rows if r["kind"] == "target" and r["family"] == "multiturn" and r["cluster"] == label]


def test_the_pool_is_one_member_per_three_concurrent_targets_minimum_one():
    rows = ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB)
    for lab in AB:
        n_t = len(_targets(rows, lab))
        assert n_t == 6 and _pool(rows, lab) == ["attacker1", "attacker2"], (lab, n_t, _pool(rows, lab))       # the live u0 shape: 6 targets -> 2 members
    one = ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=["cluster_a"])                     # 12 targets on one label -> 4 members
    assert _pool(one, "cluster_a") == ["attacker1", "attacker2", "attacker3", "attacker4"]
    few = ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB, nshards=3, shard_groups_spec="0,1,2")   # 1 group x 3 attacks -> 2 + 1 targets
    assert _pool(few, "cluster_a") == ["attacker1"] and _pool(few, "cluster_b") == ["attacker1"]
    assert ST.attacker_pool_size(6) == 2 and ST.attacker_pool_size(7) == 3 and ST.attacker_pool_size(1) == 1 and ST.attacker_pool_size(0) == 1


def test_a_forced_pool_larger_than_the_ratio_allows_refuses_by_name_and_the_cap_only_lowers():
    with pytest.raises(ValueError, match=r"attacker pool of 4 for 2 target\(s\) on cluster_a exceeds one member per three"):
        ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB, nshards=3, shard_groups_spec="0,1,2", mt_attacker_gpus=4)
    capped = ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB, mt_attacker_gpus=1)        # the cap lowers 2 -> 1, never raises
    assert _pool(capped, "cluster_a") == ["attacker1"] and _pool(capped, "cluster_b") == ["attacker1"]
    assert ST.attacker_pool_size(6, cap=1) == 1 and ST.attacker_pool_size(6, cap=4) == 2
