"""Pins tests/full22_wave_configs.py, the ONE enumerator of the full22 WAVE configs.

The prefix ``configs/sdfbase_full22_*.yaml`` is shared by the wave configs, three tables (the two rosters
and the arms table) and the generated sampled configs. The enumerator drops the tables BY VALUE and the
sampled files by their ``_sampled_`` mark, and nothing else: no shape filter, so a wave config missing a
key still fails loudly in the test that reads it. Three tests walk the waves through it
(test_sampled_controls, test_bon_climb_default, test_review_p2_6_xstest_safe); the arms table's name
reddened the first two when the arms table arrived, and this file is what keeps the next table from doing it again.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from full22_wave_configs import CONFIGS_GLOB, EXCLUDED_TABLES, SAMPLED_MARK, wave_configs

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
ROSTERS = ("sdfbase_full22_roster.yaml", "sdfbase_full22_roster_psrcert.yaml")
WALKERS = ("test_sampled_controls.py", "test_bon_climb_default.py", "test_review_p2_6_xstest_safe.py")


def _roster_view_configs() -> set[str]:
    """The file names the two rosters' views point at: the AUTHORITATIVE list of wave configs, since
    make_full22_configs renders every wave config from a roster view."""
    names: set[str] = set()
    for rn in ROSTERS:
        roster = yaml.safe_load((CONFIGS / rn).read_text())
        names |= {Path(v["config"]).name for t in roster["targets"].values() for v in t["views"].values()}
    return names


def test_every_prefixed_yaml_except_the_named_tables_and_the_sampled_files():
    waves = wave_configs(REPO)
    assert waves, "no wave configs found"
    assert waves == sorted(waves) and all(p.is_file() and p.parent == CONFIGS for p in waves)
    names = {p.name for p in waves}
    everything = {p.name for p in CONFIGS.glob(CONFIGS_GLOB)}
    assert names == {n for n in everything if n not in EXCLUDED_TABLES and SAMPLED_MARK not in n}
    assert not (names & set(EXCLUDED_TABLES)) and not any(SAMPLED_MARK in n for n in names)
    # the exclusion list is by VALUE and must not rot: every entry names a live file and carries its reason
    assert set(EXCLUDED_TABLES) <= everything, sorted(set(EXCLUDED_TABLES) - everything)
    assert all(isinstance(why, str) and why.strip() for why in EXCLUDED_TABLES.values())
    assert "sdfbase_full22_arms.yaml" in EXCLUDED_TABLES        # the table whose NAME reddened two tests when it arrived
    # the default repo is this one
    assert wave_configs() == waves


def test_the_exclusions_hide_no_wave_config_and_no_table_slips_through():
    names = {p.name for p in wave_configs(REPO)}
    # every roster view's config is enumerated: the by-value list never drops a REAL wave config
    roster_views = _roster_view_configs()
    assert roster_views and roster_views <= names, sorted(roster_views - names)
    # nothing excluded is a wave config
    for table in EXCLUDED_TABLES:
        cfg = yaml.safe_load((CONFIGS / table).read_text())
        assert "attacks" not in cfg and "bon" not in cfg, table
    # and anything enumerated that NO roster names must still be wave-shaped: a new table sneaking in under the
    # prefix is named HERE, with the remedy, instead of dying as a bare KeyError in whichever walker runs first
    strays = [n for n in sorted(names - roster_views)
              if not {"attacks", "bon"} <= set(yaml.safe_load((CONFIGS / n).read_text()))]
    assert not strays, f"neither a wave config nor a named table: {strays} -- add to EXCLUDED_TABLES with a reason"


def test_the_wave_walking_tests_use_the_enumerator_not_the_bare_glob():
    bare = re.compile(r"""glob\(\s*["'](?:configs/)?sdfbase_full22_\*\.yaml["']""")
    for f in WALKERS:
        src = (REPO / "tests" / f).read_text()
        assert "wave_configs(REPO)" in src and not bare.search(src), f
    # the globs that DELIBERATELY match generated files stay as they are (they are not wave-config walks)
    assert 'glob("sdfbase_full22_llama8b*_sampled_t*.yaml")' in (REPO / "tests/test_review_p2_6_xstest_safe.py").read_text()


def test_on_a_synthetic_tree_only_the_named_tables_and_the_sampled_files_are_dropped(tmp_path):
    cfgs = tmp_path / "configs"
    cfgs.mkdir()
    for n in (*EXCLUDED_TABLES, "sdfbase_full22_x.yaml", "sdfbase_full22_x_d1.yaml", "sdfbase_full22_x_sampled_t0.yaml",
              "sdfbase_full22_x_d1_sampled_t1_cap8192.yaml", "sdfbase_full22_newtable.yaml", "sdfbase_other.yaml",
              "sdfbase_full22_x.json"):
        (cfgs / n).write_text("{}\n")
    # the two wave configs stay -- and so does the UNNAMED table: there is no shape filter, so the walker that
    # reads it fails on its NAME and the pin above names the remedy
    assert [p.name for p in wave_configs(tmp_path)] == [
        "sdfbase_full22_newtable.yaml", "sdfbase_full22_x.yaml", "sdfbase_full22_x_d1.yaml"]
