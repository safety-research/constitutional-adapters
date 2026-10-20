"""The four multi-turn shard groups: union = 0..8, pairwise disjoint, nonempty — as the site example
and the shell default spell them, and as the ONE unit plan (lib/stage.py) carries them into every target unit's
positionals (the stage script no longer spells the loop itself)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "reproduce/lib"))
import stage as ST  # noqa: E402


def _groups(text: str):
    m = re.search(r'SISTEER_SHARD_GROUPS="([^"]+)"', text)
    assert m, "no SISTEER_SHARD_GROUPS"
    return [tuple(int(x) for x in g.split(",")) for g in m.group(1).split()]


def _check(groups):
    assert len(groups) == 4
    flat = [s for g in groups for s in g]
    assert all(groups), groups
    assert sorted(flat) == list(range(9)), flat
    for i in range(4):
        for j in range(i + 1, 4):
            assert not (set(groups[i]) & set(groups[j])), (groups[i], groups[j])


def test_site_example_groups():
    _check(_groups((REPO / "reproduce/site.example/config.env").read_text()))


def test_common_sh_default_groups():
    text = (REPO / "reproduce/lib/common.sh").read_text()
    m = re.search(r'SISTEER_SHARD_GROUPS:-([0-9, ]+)\}', text)
    assert m
    groups = [tuple(int(x) for x in g.split(",")) for g in m.group(1).split()]
    _check(groups)
    assert groups == [tuple(int(x) for x in g.split(",")) for g in ST.DEFAULT_SHARD_GROUPS]      # the shell default and the plan's agree


def test_groups_are_disjoint_per_attack_in_the_unit_plan():
    rows = ST.unit_plan("qwen35_4b_demo", "battery", table=str(REPO / "reproduce/models.yaml"), sha="c" * 40, clusters=["cluster_a"])
    targets = [r for r in rows if r["family"] == "multiturn" and r["kind"] == "target"]
    for atk in ("crescendo", "pair", "actorattack"):
        mine = [r for r in targets if r["args"][2] == atk]
        groups = [tuple(int(x) for x in r["args"][5].split(",")) for r in mine]      # the 6th positional of target_attacks.sbatch
        _check(groups)
        assert all(r["args"][4] == "9" for r in mine)                                # the 5th: nshards
        for r, g in zip(mine, groups):
            assert r["expected_shards"][atk] == [f"shard{s:02d}" for s in g]         # the unit covers ITS group, nothing else
