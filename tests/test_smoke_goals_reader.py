"""The downstream reader of the smoke goal-shard knob (`sisteer.eval.smoke_goals`; demo-import M10): ONE grammar with the planner's
(`reproduce/lib/stage.py parse_smoke_goals` — every valid spec and every refusal agree), ONE dealing rule with the planner's
(`stage.shard_goal_ids`: goals[s::N] per shard), read from the registry's run-overrides STAMP (`extra.run_overrides.{id, sha256, keys}` as
`emit.run_overrides_stamp` builds it from the document and the run row carries it): no stamp or no key -> None (the paper's rows), an id
without `smoke` or a malformed spec -> refused by name. The registration's producer key must spell the same shards (`sh<shards>of<N>` for a
multi-turn target, `p<pad>-sh<lo>_<hi>_<N>` for a padding level); the store's shard directories must be exactly lo..hi.

Review round 1 on the lane (the Major): the planner narrowed generation and the frozen coverage, but neither the job-side closer nor the
two production graders consumed the subset, so a smoke store could never close `generated` nor be graded. Every consumer now reads this
one module; this test pins the module itself."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO  # noqa: E402

sys.path.insert(0, str(REPRO / "lib"))
import stage as ST  # noqa: E402

from sisteer.eval import smoke_goals as SG  # noqa: E402
from sisteer.registry import emit  # noqa: E402

GOALS = [f"goal-{i:03d}" for i in range(259)]


def _stamp(spec="0-0/99", ident="demo_smoke", **more):
    keys = {"OVERRIDE_ID": ident, "OVERRIDE_MODEL": "qwen35_4b_demo", **more}
    if spec is not None:
        keys[SG.KEY] = spec
    return {"id": ident, "path": "demo_smoke.env", "sha256": "f" * 64, "keys": keys}


# --------------------------------------------------------------------------------------------------- one grammar, one dealing rule

@pytest.mark.parametrize("spec", ["0-0/99", "0-2/16", " 4-7/12 ", "3-3/4", "0-98/99"])
def test_every_valid_spec_parses_as_the_planner_parses_it(spec):
    assert SG.parse_spec(spec) == ST.parse_smoke_goals(spec)
    sm = SG.from_stamp(_stamp(spec))
    lo, hi, n = ST.parse_smoke_goals(spec)
    assert (sm.lo, sm.hi, sm.nshards, sm.spec) == (lo, hi, n, f"{lo}-{hi}/{n}")
    assert sm.shards == [f"shard{s:02d}" for s in range(lo, hi + 1)]


@pytest.mark.parametrize("spec", ["", "99", "0/99", "0-0", "a-b/9", "0-0/0", "0-0/100", "0-0/259", "2-1/9", "0-9/9", "9-9/9", "0-0/99 # a comment"])
def test_every_malformed_spec_refuses_here_exactly_where_the_planner_refuses(spec):
    with pytest.raises(ValueError):
        ST.parse_smoke_goals(spec)
    with pytest.raises(SG.SmokeGoalsRefused):
        SG.parse_spec(spec)
    if spec:
        with pytest.raises(SG.SmokeGoalsRefused):
            SG.from_stamp(_stamp(spec))


@pytest.mark.parametrize("spec", ["0-0/99", "0-2/16", "4-7/12", "0-8/9", "3-3/4"])
def test_the_dealing_is_the_planners_shard_goal_ids_byte_for_byte(spec):
    sm = SG.from_stamp(_stamp(spec))
    assert sm.deal(GOALS) == ST.shard_goal_ids(GOALS, sm.shards, sm.nshards)
    dealt = ST.shard_goal_ids(GOALS, sm.shards, sm.nshards)
    assert sm.goal_ids(GOALS) == [g for sh in sm.shards for g in dealt[sh]]
    assert sm.n_goals(len(GOALS)) == len(sm.goal_ids(GOALS)) == sum(len(v) for v in dealt.values())


def test_the_shipped_recipe_keeps_three_of_259_goals_in_shard00():
    sm = SG.from_stamp(_stamp("0-0/99"))
    assert sm.shards == ["shard00"] and sm.goal_ids(GOALS) == [GOALS[0], GOALS[99], GOALS[198]] and sm.n_goals(259) == 3


# --------------------------------------------------------------------------------------------------- the stamp

def test_no_stamp_or_no_key_is_none_the_papers_rows():
    assert SG.from_stamp(None) is None
    assert SG.from_stamp({}) is None
    assert SG.from_stamp({"id": "paper_defaults", "keys": {"OVERRIDE_HPO_TRIALS": "16"}}) is None
    assert SG.from_stamp(_stamp(None)) is None
    assert SG.from_stamp(_stamp("")) is None


def test_a_key_under_an_id_without_smoke_refuses_by_name():
    with pytest.raises(SG.SmokeGoalsRefused, match="names no smoke"):
        SG.from_stamp(_stamp("0-0/99", ident="paper_defaults"))
    with pytest.raises(SG.SmokeGoalsRefused, match="names no smoke"):
        SG.from_stamp({"keys": {SG.KEY: "0-0/99"}})          # no id anywhere
    assert SG.from_stamp(_stamp("0-0/99", ident="My_SMOKE_v2")).override_id == "My_SMOKE_v2"     # case-insensitive substring


def test_the_record_names_the_document_and_the_range():
    r = SG.from_stamp(_stamp("0-2/16")).record()
    assert r == {"source": SG.STAMP_SOURCE, "key": SG.KEY, "spec": "0-2/16", "lo": 0, "hi": 2, "nshards": 16, "shards": ["shard00", "shard01", "shard02"],
                 "override_id": "demo_smoke", "overrides_sha256": "f" * 64, "overrides_path": "demo_smoke.env"}


def test_from_env_reads_the_document_through_the_registrations_own_reader(tmp_path, monkeypatch):
    """The closer reads the knob inside the job from $SISTEER_RUN_OVERRIDES through emit.run_overrides_stamp — the SAME reader that stamped the
    run at registration, so what the closer honours is what the run row carries. The shipped recipe reads as 0-0/99."""
    monkeypatch.delenv(emit.RUN_OVERRIDES_ENV, raising=False)
    assert SG.from_env() is None
    doc = tmp_path / "t_smoke.env"
    doc.write_text("# synthetic\nOVERRIDE_ID=t_smoke\nOVERRIDE_MODEL=qwen35_4b_demo\nOVERRIDE_SMOKE_GOALS=0-1/8\n")
    monkeypatch.setenv(emit.RUN_OVERRIDES_ENV, str(doc))
    sm = SG.from_env()
    assert (sm.spec, sm.override_id, sm.overrides_path) == ("0-1/8", "t_smoke", "t_smoke.env") and len(sm.overrides_sha256) == 64
    assert SG.from_env() == SG.from_stamp(emit.run_overrides_stamp())
    shipped = SG.from_env(str(REPRO / "overrides" / "demo_smoke.env"))
    assert (shipped.spec, shipped.override_id, shipped.shards) == ("0-0/99", "demo_smoke", ["shard00"])
    monkeypatch.setenv(emit.RUN_OVERRIDES_ENV, str(tmp_path / "absent.env"))
    with pytest.raises(SG.SmokeGoalsRefused, match="cannot be read"):
        SG.from_env()


def test_from_run_row_reads_extra_json_and_tolerates_a_row_without_a_stamp():
    import sqlite3
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE run (run_id TEXT PRIMARY KEY, extra_json TEXT)")
    conn.execute("INSERT INTO run VALUES (?, ?)", ("r-stamped", json.dumps({"run_overrides": _stamp("0-0/99")})))
    conn.execute("INSERT INTO run VALUES (?, ?)", ("r-paper", json.dumps({"run_overrides": {"id": "paper_defaults", "keys": {"OVERRIDE_HPO_TRIALS": "16"}}})))
    conn.execute("INSERT INTO run VALUES (?, ?)", ("r-bare", None))
    conn.execute("INSERT INTO run VALUES (?, ?)", ("r-bad", json.dumps({"run_overrides": _stamp("0-0/259")})))
    assert SG.from_run_row(conn, "r-stamped").spec == "0-0/99"
    assert SG.from_run_row(conn, "r-paper") is None and SG.from_run_row(conn, "r-bare") is None and SG.from_run_row(conn, "r-missing") is None
    with pytest.raises(SG.SmokeGoalsRefused, match="> 99"):
        SG.from_run_row(conn, "r-bad")


# --------------------------------------------------------------------------------------------------- the key and the layout

def test_the_producer_key_must_spell_the_stamped_shards():
    sm = SG.from_stamp(_stamp("0-0/99"))
    assert SG.key_shards("sdfbase_full22-att-crescendo-qwen35_4b_demo_battery-sh0of99", "multiturn") == ([0], 99)
    assert SG.key_shards("sdfbase_full22-att-crescendo-qwen35_4b_s1-sh0_1of9", "multiturn") == ([0, 1], 9)
    assert SG.key_shards("sdfbase_full22-padobj-qwen35_4b_demo_battery-p0-sh0_0_99", "padding") == ([0], 99)
    assert SG.key_shards("sdfbase_full22-padobj-qwen35_4b_s1-p0-sh3", "padding") == ([0, 1, 2], 3)
    assert SG.key_shards("sdfbase_full22-padobj-qwen35_4b-p8192-sh4_7_16", "padding") == ([4, 5, 6, 7], 16)
    assert SG.key_shards("sdfbase_full22-att-crescendo-x", "multiturn") is None and SG.key_shards("whatever", "static") is None
    assert SG.key_disagreement("sdfbase_full22-att-crescendo-qwen35_4b_demo_battery-sh0of99", sm, "multiturn") is None
    assert SG.key_disagreement("sdfbase_full22-padobj-qwen35_4b_demo_battery-p0-sh0_0_99", sm, "padding") is None
    assert "did not run the document's plan" in SG.key_disagreement("sdfbase_full22-att-crescendo-qwen35_4b_s1-sh0_1of9", sm, "multiturn")
    assert "did not run the document's plan" in SG.key_disagreement("sdfbase_full22-att-crescendo-qwen35_4b_s1-sh0of9", sm, "multiturn")
    assert "did not run the document's plan" in SG.key_disagreement("sdfbase_full22-padobj-qwen35_4b_s1-p0-sh3", sm, "padding")
    assert "carries no shard segment" in SG.key_disagreement("sdfbase_full22-att-crescendo-x", sm, "multiturn")
    assert SG.nshards_disagreement(99, sm) is None and SG.nshards_disagreement(None, sm) is None
    assert "one store, one layout" in SG.nshards_disagreement(9, sm)


def test_the_store_must_hold_exactly_the_stamped_shards():
    sm = SG.from_stamp(_stamp("1-2/8"))
    assert SG.layout_disagreement(["shard01", "shard02"], sm) is None
    assert SG.layout_disagreement([Path("shard02"), Path("shard01")], sm) is None
    assert "1 missing ['shard02']" in SG.layout_disagreement(["shard01"], sm)
    assert "1 outside the range ['shard00']" in SG.layout_disagreement(["shard00", "shard01", "shard02"], sm)
    assert SG.NARROWED_FAMILIES == ("multiturn", "padding")
