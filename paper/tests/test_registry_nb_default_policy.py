"""The paper's judging policy is the notebook's default (the owner, 2026-11-14: the paper's numbers are the re-judges', filed as second
judgments).

  * with REGISTRY_NB_JUDGES unset the policy is `rejudge` (PAPER_JUDGES), the render is unstamped, and the judges it resolves to in the
    committed paper frame are exactly paper/notebooks/data/paper_judges.json's five, every one among the frame's item judges; the file
    lists exactly the policy's families, and neither names an agentic-misalignment judge (the paper draws no agentic reading)
  * a spelling that resolves to the paper's judges is the paper's render; anything else prints the line, naming the judges it draws
  * a lane registered with judge ids the policy already draws adds no stamp; under `primary` it does; a free-text registration stamps
    under every policy
  * the source guards: PolicyJudgesAbsent when the source holds no judge the policy names (any source), SnapshotLacksJudgeItems when
    the frame carries no item rows for one, SnapshotLacksItemKey when an item read asks a frame for a key its projection drops; under
    `primary` nothing is checked
"""
import json
import os
import pathlib
import sqlite3
import subprocess
import sys

import pandas as pd
import pytest

NOTEBOOKS = pathlib.Path(__file__).resolve().parents[1] / "notebooks"
REPO = NOTEBOOKS.parents[1]
if str(NOTEBOOKS) not in sys.path:
    sys.path.insert(0, str(NOTEBOOKS))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import registry_nb as R  # noqa: E402

PJ = json.loads((NOTEBOOKS / "data" / "paper_judges.json").read_text(encoding="utf-8"))
GLM = PJ["petri_alias"]["glm-5.3"]
SWITCHES = ("REGISTRY_NB_JUDGES", "REGISTRY_NB_PETRI_JUDGE", "REGISTRY_NB_PETRI_EXCLUDE", "REGISTRY_NB_SR_THRESHOLD", "REGISTRY_NB_FIGS",
            "REGISTRY_NB_STRENGTH_READS")


@pytest.fixture
def paper_switches(monkeypatch):
    """This module's R as a render with no switch set: the default policy, no Petri override or exclusion, the paper's threshold."""
    monkeypatch.setattr(R, "JUDGES", R.PAPER_JUDGES)
    monkeypatch.setattr(R, "PETRI_JUDGE", None)
    monkeypatch.setattr(R, "PETRI_EXCLUDE", "")
    monkeypatch.setattr(R, "SR_THRESHOLD", R.SR_PAPER_THRESHOLD)


# ---- the default -------------------------------------------------------------------------------------------------------------------
def test_with_no_switch_set_the_render_is_the_papers_and_unstamped(tmp_path):
    env = {k: v for k, v in os.environ.items() if k not in SWITCHES}
    env.update(REGISTRY_NB_SOURCE="snapshot", MPLBACKEND="Agg")
    out = subprocess.run([sys.executable, "-c", "import registry_nb as R; print('JUDGES=' + R.JUDGES + '|' + R.PAPER_JUDGES); "
                          "print('POLICY=' + R.RENDER_POLICY); print('STAMP=' + R.render_stamp())"],
                         cwd=NOTEBOOKS, env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    lines = out.stdout.splitlines()
    assert "JUDGES=rejudge|rejudge" in lines and "POLICY=" in lines and "STAMP=" in lines
    assert "RENDER POLICY" not in out.stdout, "the paper's render prints no policy line (the control-pool lane's judges are the policy's)"


def test_the_default_policy_resolves_to_exactly_the_paper_judges_in_the_committed_frame():
    """paper_judges.json is the one definition: each family's model is the policy's, each (model, rubric, version) is ONE judge of the
    committed frame's judge table with that id, the policy's specs reach exactly those ids, and the frame carries their item rows."""
    assert PJ["policy"] == R.PAPER_JUDGES == "rejudge"
    pol = R.judge_policy(R.PAPER_JUDGES)
    sid = (REPO / "paper" / "snapshot" / "CURRENT").read_text(encoding="utf-8").strip()
    frame = REPO / "paper" / "snapshot" / "snapshots" / sid
    judges = pd.read_parquet(frame / "judge.parquet", columns=["judge_id", "judge_model", "rubric_name", "rubric_version"])
    for j in PJ["judges"]:
        assert pol["families"][j["family"]] == j["judge_model"], j
        hit = judges[judges.judge_model.eq(j["judge_model"]) & judges.rubric_name.eq(j["rubric_name"])
                     & judges.rubric_version.astype(str).eq(j["rubric_version"])]
        assert hit.judge_id.tolist() == [j["judge_id"]], j
    reached = set()
    for spec, rub in R.policy_judge_specs(pol):
        reached |= set(judges[judges.rubric_name.eq(rub) & (judges.judge_id.eq(spec) | judges.judge_model.eq(spec))].judge_id)
    drawn = {j["judge_id"] for j in PJ["judges"]}
    assert reached == drawn
    # the frame carries exactly the paper judges' item rows: the cut also took the agentic judge's (the policy named it then), and the
    # release pass left those out, which the MANIFEST records
    manifest = json.loads((frame / "MANIFEST.json").read_text(encoding="utf-8"))
    assert drawn == {j["judge_id"] for j in manifest["item_judges"]}
    assert {j["judge_id"] for j in manifest.get("release_omitted_item_judges") or []}.isdisjoint(drawn)


def test_the_paper_judges_file_lists_exactly_the_policys_families_and_no_agentic_judge():
    """The agentic-misalignment evals are omitted from the paper (OMITTED_EVALS), so the paper's policy names no judge for their family
    and paper_judges.json lists none: the file's families are the policy's, one entry per (judge model, rubric, version)."""
    pol = R.judge_policy(R.PAPER_JUDGES)
    assert {j["family"] for j in PJ["judges"]} == set(R.JUDGE_POLICIES[R.PAPER_JUDGES]) == set(pol["families"])
    assert "agentic" not in pol["families"] and R.row_judge_spec(pol, "agentic_misalignment_classifier", "agentic_leaking") is None
    assert not [j for j in PJ["judges"] if R.RUBRIC_FAMILY.get(j["rubric_name"]) not in pol["families"]]
    assert len({(j["judge_model"], j["rubric_name"], j["rubric_version"]) for j in PJ["judges"]}) == len(PJ["judges"]) == 5
    assert all(e in R.OMITTED_EVALS for e, rub in R.EVAL_RUBRIC.items() if rub == "agentic_misalignment_classifier")


# ---- what counts as the paper's render ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("judges, petri", [
    ("rejudge", None),
    ("rejudge,petri=glm-5.3", None),
    (f"rejudge,petri_bloom_judge={GLM}", None),                               # by the paper's judge id
    ("primary,strongreject=minimax-m3,over_refusal=minimax-m3,petri=glm-5.3", None),
    ("primary,strongreject=minimax-m3,over_refusal=minimax-m3,agentic=minimax-m3,petri=glm-5.3", None),   # the agentic evals are not drawn
    ("rejudge,agentic=minimax-m3", None),                                     # an agentic judge on top of the paper's: still not drawn
    ("primary,strongreject=minimax-m3,over_refusal=minimax-m3", "glm-5.3"),   # REGISTRY_NB_PETRI_JUDGE=glm-5.3
    ("rejudge", "glm-5.3"),
])
def test_a_spelling_that_resolves_to_the_paper_judges_is_the_papers_render(paper_switches, judges, petri):
    assert R.policy_is_papers(judges, petri or "")
    assert R.render_policy(judges=judges, petri_judge=petri or "", petri_exclude="", threshold=0.5) == ""


@pytest.mark.parametrize("judges, petri, threshold, exclude, says", [
    ("primary", None, 0.5, "", "REGISTRY_NB_JUDGES=primary (the filed Claude judges: claude-haiku-4-5 StrongREJECT, claude-sonnet-5 C/P/I, "
                               "claude-opus-5 Petri); StrongREJECT threshold 0.5 (the paper's: minimax-m3, glm-5.3 on Petri, at 0.5)"),
    ("rejudge,pair=primary", None, 0.5, "", "(StrongREJECT minimax-m3, C/P/I minimax-m3, Petri glm-5.3; pair filed)"),
    ("rejudge", "primary", 0.5, "", "REGISTRY_NB_PETRI_JUDGE=primary (StrongREJECT minimax-m3, C/P/I minimax-m3, Petri filed)"),
    ("rejudge,petri=primary", None, 0.5, "", "Petri filed"),
    ("rejudge", None, 0.75, "", "StrongREJECT threshold 0.75"),
    ("rejudge", None, 0.5, "x.tsv", "REGISTRY_NB_PETRI_EXCLUDE="),
    ("primary,strongreject=minimax-m3", None, 0.5, "", "(StrongREJECT minimax-m3, C/P/I filed, Petri filed)"),
])
def test_anything_else_says_so_and_names_the_judges_it_draws(paper_switches, judges, petri, threshold, exclude, says):
    line = R.render_policy(judges=judges, petri_judge=petri or "", petri_exclude=exclude, threshold=threshold)
    assert line.startswith(f"judges: REGISTRY_NB_JUDGES={judges}") and says in line, line


# ---- a judge id passes for the paper's judge only where it draws every row the paper's judge would ---------------------------------
SR_V1 = next(j["judge_id"] for j in PJ["judges"] if (j["rubric_name"], j["rubric_version"]) == ("strongreject", "v1"))
SR_V2 = next(j["judge_id"] for j in PJ["judges"] if (j["rubric_name"], j["rubric_version"]) == ("strongreject", "v2"))


@pytest.fixture
def filed_source(tmp_path, monkeypatch):
    """A source shaped as the committed frame is: every StrongREJECT eval filed under claude-haiku-4-5 v1, best-of-N under v1 AND v2
    (its runs mix the two rubric versions), each re-judged by the paper judge of the same version."""
    p = tmp_path / "filed.sqlite"
    c = sqlite3.connect(p)
    c.executescript("""
        CREATE TABLE judge (judge_id TEXT, judge_model TEXT, rubric_name TEXT, rubric_version TEXT, rubric_sha256 TEXT);
        CREATE TABLE instrument (instrument_id TEXT, judge_id TEXT);
        CREATE TABLE run (run_id TEXT, eval_id TEXT, instrument_id TEXT);
        CREATE TABLE judgment (judgment_id TEXT, run_id TEXT, judge_id TEXT, status TEXT);
    """)
    c.executemany("INSERT INTO judge VALUES (?,?,?,?,?)", [(jid, m, rub, ver, f"sha-{rub}-{ver}") for jid, m, rub, ver in PAPER]
                  + [("hk1", "claude-haiku-4-5", "strongreject", "v1", "sha-strongreject-v1"),
                     ("hk2", "claude-haiku-4-5", "strongreject", "v2", "sha-strongreject-v2")])
    c.executemany("INSERT INTO instrument VALUES (?,?)", [("ins1", "hk1"), ("ins2", "hk2")])
    runs = [(f"{e}:r", e, "ins1") for e in R._drawn_rubric_evals("strongreject")] + [("strongreject_bon:r2", "strongreject_bon", "ins2")]
    c.executemany("INSERT INTO run VALUES (?,?,?)", runs)
    c.executemany("INSERT INTO judgment VALUES (?,?,?,'complete')", [(f"g:{rid}", rid, SR_V2 if ins == "ins2" else SR_V1) for rid, _e, ins in runs])
    c.commit()
    c.close()
    monkeypatch.setattr(R, "connect", lambda: sqlite3.connect(p))
    R.resolve_judge.cache_clear()
    R._filed_judges.cache_clear()
    yield
    R.resolve_judge.cache_clear()
    R._filed_judges.cache_clear()


@pytest.mark.parametrize("judges", [f"rejudge,strongreject={SR_V1}", f"rejudge,strongreject={SR_V2}", f"rejudge,strongreject_bon={SR_V1}"])
def test_a_strongreject_judge_id_that_misses_a_rubric_version_is_stamped_and_refused_the_papers_figures(paper_switches, filed_source,
                                                                                                        monkeypatch, judges):
    """One StrongREJECT id draws every row from one version: best-of-N rows filed under v2 lose their reading under the v1 id, and every
    v1-filed row under the v2 id. Such a spelling is not the paper's render: it is stamped, the control-pool lane is stamped with it, and
    it may not write the paper's figures directory."""
    assert not R.policy_is_papers(judges, "")
    line = R.render_policy(judges=judges, petri_judge="", petri_exclude="", threshold=0.5)
    assert line.startswith(f"judges: REGISTRY_NB_JUDGES={judges}"), line
    assert not R.lane_drawn_by_policy(R.CTLSTOP_JUDGES, judges)
    monkeypatch.setattr(R, "RENDER_POLICY", line)
    monkeypatch.setattr(R, "FIGS", os.path.join(R.HERE, "figures"))
    with pytest.raises(RuntimeError, match="does not write the paper's figures directory"):
        R.announce_render_policy()


def test_a_judge_id_that_draws_every_row_the_papers_judge_would_is_the_papers_render(paper_switches, filed_source):
    """Crescendo is filed under v1 alone, so its v1 id draws what the paper's model draws there; the same holds per eval for the
    control-pool lane, and an id of a judge outside the paper's set is never the paper's."""
    for judges in (f"rejudge,crescendo={SR_V1}", f"rejudge,grok420={SR_V1},pair={SR_V1}"):
        assert R.policy_is_papers(judges, ""), judges
        assert R.render_policy(judges=judges, petri_judge="", petri_exclude="", threshold=0.5) == ""
        assert R.lane_drawn_by_policy(R.CTLSTOP_JUDGES, judges), judges
    assert not R.policy_is_papers(f"rejudge,crescendo={SR_V2}", "")                 # every crescendo row is filed under v1
    assert not R.policy_is_papers("rejudge,crescendo=hk1", "")                       # the filed judge's own id is another model
    assert R._same_judge(SR_V1, "minimax-m3", "strongreject", "crescendo")
    assert not R._same_judge(SR_V1, "minimax-m3", "strongreject", "strongreject_bon")


def test_a_judge_id_the_source_cannot_settle_is_not_the_papers(paper_switches, monkeypatch):
    def nowhere():
        raise FileNotFoundError("no source")
    monkeypatch.setattr(R, "connect", nowhere)
    R._filed_judges.cache_clear()
    try:
        assert not R.policy_is_papers(f"rejudge,crescendo={SR_V1}", "")
        assert R.policy_is_papers("rejudge", "")                                       # equal spellings need no source
    finally:
        R._filed_judges.cache_clear()


# ---- the lane-fixed stamp ----------------------------------------------------------------------------------------------------------
def test_a_lane_named_by_judge_id_is_stamped_only_under_a_policy_that_draws_other_judges(monkeypatch, paper_switches, capsys):
    monkeypatch.setattr(R, "RENDER_POLICY", "")
    monkeypatch.setattr(R, "LANE_FIXED", {})
    monkeypatch.setattr(R, "LANE_FIXED_IDS", {})
    R.register_lane_fixed("lane figures", "minimax-m3 judges", judge_ids=R.CTLSTOP_JUDGES)
    assert R.render_stamp() == "" and "RENDER POLICY" not in capsys.readouterr().out          # the paper's policy draws the lane's judges
    assert R.lane_drawn_by_policy(R.CTLSTOP_JUDGES, "rejudge,petri=primary")                # Petri is not the lane's rubric
    for other in ("primary", "rejudge,pair=primary", "rejudge,over_refusal=primary", "primary,strongreject=minimax-m3"):
        assert not R.lane_drawn_by_policy(R.CTLSTOP_JUDGES, other), other
    monkeypatch.setattr(R, "JUDGES", "primary")
    assert R.render_stamp() == "lane figures lane-fixed: minimax-m3 judges"
    # a free-text registration stamps under every policy; re-registering without ids drops the ids
    R.register_lane_fixed("lane figures", "minimax-m3 judges")
    assert "lane figures" not in R.LANE_FIXED_IDS
    monkeypatch.setattr(R, "JUDGES", R.PAPER_JUDGES)
    assert R.render_stamp() == "lane figures lane-fixed: minimax-m3 judges"
    # a lane on judges the paper's policy does not draw stamps under it
    R.register_lane_fixed("haiku figures", "claude-haiku-4-5", judge_ids={("claude-haiku-4-5", "strongreject", "v1"): "h" * 64})
    assert "haiku figures lane-fixed: claude-haiku-4-5" in R.render_stamp()


def test_the_control_pool_lane_is_registered_by_its_judge_ids():
    assert R.LANE_FIXED_IDS.get("control-pool figures") == R.CTLSTOP_JUDGES
    assert R.lane_drawn_by_policy(R.CTLSTOP_JUDGES, R.PAPER_JUDGES) and not R.lane_drawn_by_policy(R.CTLSTOP_JUDGES, "primary")


def test_a_primary_render_may_not_write_the_papers_figures(monkeypatch):
    monkeypatch.setattr(R, "RENDER_POLICY", R.render_policy(judges="primary", petri_judge="", petri_exclude="", threshold=0.5))
    monkeypatch.setattr(R, "FIGS", os.path.join(R.HERE, "figures"))
    with pytest.raises(RuntimeError, match="figures-primary"):
        R.announce_render_policy()
    monkeypatch.setattr(R, "FIGS", R.FIGS_PRIMARY)
    R.announce_render_policy()


# ---- the source guards -------------------------------------------------------------------------------------------------------------
def _registry(tmp_path, judges):
    p = tmp_path / "reg.sqlite"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE judge (judge_id TEXT, judge_model TEXT, rubric_name TEXT, rubric_version TEXT)")
    c.executemany("INSERT INTO judge VALUES (?,?,?,?)", judges)
    c.commit()
    c.close()
    return str(p)


CLAUDE = [("hk", "claude-haiku-4-5", "strongreject", "v1"), ("sn-x", "claude-sonnet-5", "xstest_cpi", "v1"),
          ("sn-o", "claude-sonnet-5", "or_bench_cpi", "v1"), ("op", "claude-opus-5", "petri_bloom_judge", "x")]
PAPER = [(j["judge_id"], j["judge_model"], j["rubric_name"], j["rubric_version"]) for j in PJ["judges"]]


def test_a_source_without_the_policy_judges_refuses_by_name(tmp_path, monkeypatch):
    """The front door's case: a registry graded by the pipeline's own stage 08 holds the Claude judges only."""
    db = _registry(tmp_path, CLAUDE)
    monkeypatch.setattr(R, "connect", lambda: sqlite3.connect(db))
    monkeypatch.setattr(R, "SOURCE", "live")
    monkeypatch.setattr(R, "PETRI_JUDGE", None)
    with pytest.raises(R.PolicyJudgesAbsent) as e:
        R.check_policy_judges(R.judge_policy("rejudge"))
    msg = str(e.value)
    for want in ("minimax-m3 on strongreject", "minimax-m3 on xstest_cpi", "minimax-m3 on or_bench_cpi", GLM[:12],
                 "REGISTRY_NB_JUDGES=primary", "--judges primary", "--source snapshot"):
        assert want in msg, (want, msg)
    assert R.check_policy_judges(R.judge_policy("primary")) == {}                  # nothing substituted, nothing checked


def test_a_source_with_the_policy_judges_names_them(tmp_path, monkeypatch):
    db = _registry(tmp_path, CLAUDE + PAPER)
    monkeypatch.setattr(R, "connect", lambda: sqlite3.connect(db))
    monkeypatch.setattr(R, "SOURCE", "live")
    monkeypatch.setattr(R, "PETRI_JUDGE", None)
    got = R.check_policy_judges(R.judge_policy("rejudge"))
    sr = sorted(j["judge_id"] for j in PJ["judges"] if j["rubric_name"] == "strongreject")
    assert got[("minimax-m3", "strongreject")] == sr and got[(GLM, "petri_bloom_judge")] == [GLM]
    assert not any(rub == "agentic_misalignment_classifier" for _s, rub in got)    # the omitted evals' judge is never required


def test_a_frame_without_the_policy_judges_items_refuses_by_name(tmp_path, monkeypatch):
    db = _registry(tmp_path, CLAUDE + PAPER)
    monkeypatch.setattr(R, "connect", lambda: sqlite3.connect(db))
    monkeypatch.setattr(R, "SOURCE", "snapshot")
    monkeypatch.setattr(R, "PETRI_JUDGE", None)
    have = [dict(judge_id=j["judge_id"]) for j in PJ["judges"] if j["family"] != "petri"]
    monkeypatch.setattr(R, "_frame_manifest", lambda root: dict(item_judges=have))
    with pytest.raises(R.SnapshotLacksJudgeItems, match="snapshot_paper_frame.sh"):
        R.check_policy_judges(R.judge_policy("rejudge"))
    assert R.check_policy_judges(R.judge_policy("rejudge,petri=primary"))          # without Petri every judge it draws has items
    monkeypatch.setattr(R, "_frame_manifest", lambda root: {})                     # a frame cut with no item judges
    with pytest.raises(R.SnapshotLacksJudgeItems):
        R.check_policy_judges(R.judge_policy("rejudge"))
    assert R.check_policy_judges(R.judge_policy("primary")) == {}


def test_an_item_read_asks_a_frame_only_for_the_keys_its_projection_keeps(monkeypatch):
    monkeypatch.setattr(R, "SOURCE", "snapshot")
    monkeypatch.setattr(R, "_frame_manifest", lambda root: dict(item_projection=dict(name="reader", extra_json_keys=["subset"])))
    assert "json_extract(ir.extra_json, '$.subset')" in R.item_rows_sql("1 = 1", ("subset",))
    assert "json_extract" not in R.item_rows_sql("1 = 1", ())
    with pytest.raises(R.SnapshotLacksItemKey, match="at_cap"):
        R.item_rows_sql("1 = 1", ("subset", "at_cap"))
    monkeypatch.setattr(R, "connect", lambda: sqlite3.connect(":memory:"))       # the public reader goes through the same check
    with pytest.raises(R.SnapshotLacksItemKey):
        R.judge_items(["j1"], fields=("at_cap",))
    monkeypatch.setattr(R, "_frame_manifest", lambda root: {})                     # a frame with no projection (verbatim items)
    assert "'$.at_cap'" in R.item_rows_sql("1 = 1", ("at_cap",))
    monkeypatch.setattr(R, "SOURCE", "live")
    monkeypatch.setattr(R, "_frame_manifest", lambda root: dict(item_projection=dict(name="reader", extra_json_keys=["subset"])))
    assert "'$.at_cap'" in R.item_rows_sql("1 = 1", ("at_cap",))                    # the live registry keeps every key
