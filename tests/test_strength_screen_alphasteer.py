"""screen_verdict.py's AlphaSteer extension: the AlphaSteer strength roster (configs/alphasteer_f22_strength_roster.yaml) spells its
dose `strength:` and its screens are named `asscr-<TAG>-ctl<T>` (scripts/alphasteer_f22/screen/asscreen_controls.sbatch);
the tool reads both from the roster and changes nothing else — the 58 tests in tests/test_strength_screen.py are the
pin that the full22 roster's `dose:` views and `f22s-` names still read exactly as before.

Fixtures: a SYNTHETIC strength roster in the real file's shape (wave, one target, five strength views, one `alphasteer`
cell each) plus the real roster committed on this tree; the store / positive-control builders are the sibling test's.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sdfbase" / "full22" / "strength" / "screen_verdict.py"
REAL_ROSTER = REPO / "configs" / "alphasteer_f22_strength_roster.yaml"
sys.path.insert(0, str(REPO / "src"))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sv = _load("screen_verdict", SCRIPT)
sib = _load("test_strength_screen", REPO / "tests" / "test_strength_screen.py")   # its Sampled builder + templates

STRENGTH_ROSTER_YAML = """\
wave: alphasteer_f22_strength
bank_root: exp/objects/alphasteer_f22
bank_subdir: strength
targets:
  nemotron3nano4b:
    model: nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
    campaign_view: main
    views:
      as_dm0p5:
        strength: -0.5
        config: configs/alphasteer_f22_serve_nemotron3nano4b_as_dm0p5.yaml
        manifest: configs/capeval/manifests/alphasteer_f22_nemotron3nano4b_as_dm0p5.yaml
        cells:
          alphasteer: as_dm0p5__alphasteer
      as_d0p1:
        strength: 0.1
        config: configs/alphasteer_f22_serve_nemotron3nano4b_as_d0p1.yaml
        cells:
          alphasteer: as_d0p1__alphasteer
      as_d0p3:
        strength: 0.3
        config: configs/alphasteer_f22_serve_nemotron3nano4b_as_d0p3.yaml
        cells:
          alphasteer: as_d0p3__alphasteer
      as_d0p5:
        strength: 0.5
        config: configs/alphasteer_f22_serve_nemotron3nano4b_as_d0p5.yaml
        cells:
          alphasteer: as_d0p5__alphasteer
      as_d1:
        strength: 1.0
        config: configs/alphasteer_f22_serve_nemotron3nano4b_as_d1.yaml
        cells:
          alphasteer: as_d1__alphasteer
"""


@pytest.fixture
def strength_roster(tmp_path):
    p = tmp_path / "strength_roster.yaml"
    p.write_text(STRENGTH_ROSTER_YAML)
    return p


@pytest.fixture
def sampled(tmp_path):
    return sib.Sampled(tmp_path / "sampled")


def test_strength_key_is_the_dose_and_the_job_prefix_follows_the_wave(strength_roster):
    ro = sv.load_dose_views(strength_roster)
    views = ro["views"]
    assert [v["view"] for v in views] == ["as_dm0p5", "as_d0p1", "as_d0p3", "as_d0p5", "as_d1"]
    assert [v["dose"] for v in views] == [-0.5, 0.1, 0.3, 0.5, 1.0]
    assert all(v["dose_key"] == "strength" and v["job_prefix"] == "asscr" for v in views)
    assert [v["direction"] for v in views] == ["-", "+", "+", "+", "+"]
    assert [v["anchor_exempt"] for v in views] == [False, False, False, False, True]      # as_d1 IS the parent: the 1.0 anchor
    assert all(v["cells"] == ["alphasteer"] for v in views)
    assert views[2]["tag"] == "nemotron3nano4b_as_d0p3" and views[2]["objects"] == {"alphasteer": "as_d0p3__alphasteer"}
    assert sv.job_name(views[2]["tag"], 0, views[2]["job_prefix"]) == "asscr-nemotron3nano4b_as_d0p3-ctl0"
    n, err = sv._count_dose_views(strength_roster)
    assert (n, err) == (5, None)
    p, how = sv.resolve_roster(str(strength_roster))
    assert p == strength_roster and how["how"] == "--roster flag" and how["examined"][0]["dose_views"] == 5


def test_the_full22_spelling_and_default_prefix_are_unchanged(tmp_path):
    p = tmp_path / "roster.yaml"
    p.write_text(sib.ROSTER_YAML)
    views = sv.load_dose_views(p)["views"]
    assert all(v["dose_key"] == "dose" and v["job_prefix"] == "f22s" for v in views)
    assert sv.job_name("nemotron3nano4b_dm1", 0) == "f22s-nemotron3nano4b_dm1-ctl0"          # the 2-arg call, as the 58 tests use it
    assert sv.DEFAULT_JOB_PREFIX == "f22s" and sv.JOB_PREFIX_BY_WAVE == {"full22": "f22s", "alphasteer_f22_strength": "asscr"}


def test_a_view_spelling_dose_and_strength_differently_is_refused_and_agreeing_is_read_once(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(STRENGTH_ROSTER_YAML.replace("        strength: 0.3\n", "        strength: 0.3\n        dose: 0.5\n"))
    with pytest.raises(sv.Refusal, match="disagree"):
        sv.load_dose_views(bad)
    ok = tmp_path / "ok.yaml"
    ok.write_text(STRENGTH_ROSTER_YAML.replace("        strength: 0.3\n", "        strength: 0.3\n        dose: 0.3\n"))
    v = [x for x in sv.load_dose_views(ok)["views"] if x["view"] == "as_d0p3"][0]
    assert v["dose"] == 0.3 and v["dose_key"] == "dose"          # `dose:` wins the spelling when both agree
    zero = tmp_path / "zero.yaml"
    zero.write_text(STRENGTH_ROSTER_YAML.replace("        strength: 0.3\n", "        strength: 0\n"))
    with pytest.raises(sv.Refusal, match="strength 0 is not a rung"):
        sv.load_dose_views(zero)
    text = tmp_path / "text.yaml"
    text.write_text(STRENGTH_ROSTER_YAML.replace("        strength: 0.3\n", "        strength: high\n"))
    with pytest.raises(sv.Refusal, match="not numeric"):
        sv.load_dose_views(text)


def test_the_committed_strength_roster_enumerates_the_lanes_twenty_views():
    views = sv.load_dose_views(REAL_ROSTER)["views"]
    assert len(views) == 20 and {v["target"] for v in views} == {"gemma4e4b", "qwen35_4b", "nemotron3nano4b", "glm46vflash"}
    assert all(v["job_prefix"] == "asscr" and v["cells"] == ["alphasteer"] for v in views)
    assert sorted({v["dose"] for v in views}) == [-0.5, 0.1, 0.3, 0.5, 1.0]
    sel = sv.select_views(views, "nemotron3nano4b_as_d0p3")
    assert len(sel) == 1 and sel[0]["objects"] == {"alphasteer": "as_d0p3__alphasteer"}
    assert sv.METHOD_LABEL["alphasteer"] == "AlphaSteer (as-fit layers)"


def test_evaluate_binds_the_asscr_screen_job_and_reads_a_strength_store(strength_roster, sampled):
    """A complete as_d0p3 view in the wave layout (the same store shape the wrapper writes): the cell record names the
    asscr- job, the method label is AlphaSteer's, the store is complete, the positive control is MEASURED on the cell."""
    tag = "nemotron3nano4b_as_d0p3"
    sampled.complete_view(tag, cells=("alphasteer",), anchor=(56, 118),
                          cell_counts={"alphasteer": {"xstest": 60, "or_bench_hard_1k": 130}}, verdict="PASS")
    # the PC record names the alphasteer cell (one object cell: measured, never asserted)
    sd = sampled.positive_control(tag, "PASS", cell="alphasteer")
    assert (sd / "positive_control_controls_512.json.verdict").read_text().strip() == "PASS"
    views = sv.select_views(sv.load_dose_views(strength_roster)["views"], tag)
    req = sv.build_request(views, str(sampled.root), str(sampled.root.parent), 0, None, None, [], probe_jobs=False)
    gathered = sv.gather(req)
    res = sv.evaluate(views, gathered, 0)
    (cell,) = res["cells"]
    assert cell["job_name"] == "asscr-nemotron3nano4b_as_d0p3-ctl0" and cell["job_prefix"] == "asscr"
    assert cell["method"] == "alphasteer" and cell["method_label"] == "AlphaSteer (as-fit layers)" and cell["dose"] == 0.3
    assert cell["store"]["complete"] and cell["anchor"]["complete"]
    assert cell["positive_control"]["status"] == "pass" and cell["positive_control"]["liveness"] == "measured"
    assert cell["clauses"]["c"]["status"] == "clear" and cell["verdict"] in ("ok", "flagged", "incomplete")
    # the direction record carries the asscr screen job too
    (d,) = res["directions"]
    assert d["rungs"][0]["screen_job"] == "asscr-nemotron3nano4b_as_d0p3-ctl0" and d["method_label"] == "AlphaSteer (as-fit layers)"


def test_dry_run_names_the_asscr_job_and_the_strength_store_paths(strength_roster, tmp_path, capsys):
    rc = sv.main(["--local-root", str(tmp_path / "sampled"), "--roster", str(strength_roster),
                  "--tags", "nemotron3nano4b_as_d0p3", "--dry-run", "--no-jobs"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "screen job=asscr-nemotron3nano4b_as_d0p3-ctl0" in out and "dose=+0.3" in out
    assert "controls_cap512/nemotron3nano4b_as_d0p3_t0/cells/<store>/gen_stats.json" in out
    assert "serving/controls_t0_nemotron3nano4b_as_d0p3_<jobid>/positive_control_controls_512.json.verdict" in out
    # the SCREEN job is ours; the expensive-leg patterns printed beside it keep the sampled submitter's f22s- names on
    # purpose (leg_job_patterns is untouched: no AlphaSteer expensive leg exists yet)
    assert "screen job=f22s-" not in out and "f22s-nemotron3nano4b_as_d0p3-ctl0" not in out
