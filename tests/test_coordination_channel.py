"""The coordination channel is SITE config (``SISTEER_COORDINATION_CHANNEL``): a site that coordinates its shared API
pools in a chat channel names it, and only then does a run print its ready-to-post line; unset or empty prints none.

* ``sisteer.coordination`` (the harness) and ``corpus_generator.coordination`` (its twin; the corpus generator does not
  depend on the harness) behave identically, held here.
* The print sites reached without a model or a key: the probe-set builder's post and the grading daemon's line.
Synthetic channel names only.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "corpus_generator" / "src", REPO / "scripts" / "sdfbase"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from corpus_generator import coordination as cg_coordination  # noqa: E402
from sisteer import coordination  # noqa: E402

CH = "#example-api-coordination"


@pytest.fixture
def no_channel(monkeypatch):
    monkeypatch.delenv(coordination.ENV, raising=False)


def test_channel_is_read_from_the_site_variable_and_blank_means_none(monkeypatch):
    assert coordination.ENV == "SISTEER_COORDINATION_CHANNEL"
    for env in ({}, {coordination.ENV: ""}, {coordination.ENV: "  \t"}):
        assert coordination.channel(env) == "" and coordination.header(env=env) == ""
    assert coordination.channel({coordination.ENV: f"  {CH} "}) == CH
    monkeypatch.setenv(coordination.ENV, CH)
    assert coordination.channel() == CH                                       # the process environment by default
    assert coordination.header() == f"--- ready-to-post for {CH} ---"
    assert coordination.header("at launch") == f"--- ready-to-post for {CH} (at launch) ---"


def test_post_prints_nothing_without_a_channel_and_the_three_lines_with_one():
    out = []
    assert coordination.post("200 sonnet lowprio ~1h", out.append, env={}) is False and out == []
    assert coordination.post("200 sonnet lowprio ~1h", out.append, env={coordination.ENV: CH}) is True
    assert out == [f"--- ready-to-post for {CH} ---", "200 sonnet lowprio ~1h", coordination.PROCEEDING]
    out.clear()
    coordination.post("5 haiku batch ~2h", out.append, note="at launch", proceeding=False, env={coordination.ENV: CH})
    assert out == [f"--- ready-to-post for {CH} (at launch) ---", "5 haiku batch ~2h"]


@pytest.mark.parametrize("env", [{}, {"SISTEER_COORDINATION_CHANNEL": ""}, {"SISTEER_COORDINATION_CHANNEL": CH},
                                 {"SISTEER_COORDINATION_CHANNEL": f" {CH}\n"}])
@pytest.mark.parametrize("note,proceeding", [("", True), ("at launch", False)])
def test_the_corpus_generator_twin_matches_the_harness(env, note, proceeding):
    assert cg_coordination.ENV == coordination.ENV and cg_coordination.PROCEEDING == coordination.PROCEEDING
    assert cg_coordination.channel(env) == coordination.channel(env)
    assert cg_coordination.header(note, env) == coordination.header(note, env)
    a, b = [], []
    assert (cg_coordination.post("1 m p d", a.append, note=note, proceeding=proceeding, env=env)
            == coordination.post("1 m p d", b.append, note=note, proceeding=proceeding, env=env))
    assert a == b


def test_probe_set_post_follows_the_site_channel(monkeypatch, capsys, no_channel):
    import build_probe_set as bps
    bps._coordination_post("200 sonnet lowprio ~30m — probe set")
    assert capsys.readouterr().out == ""
    monkeypatch.setenv(coordination.ENV, CH)
    bps._coordination_post("200 sonnet lowprio ~30m — probe set")
    out = capsys.readouterr().out
    assert f"paste into {CH} and PROCEED" in out and "  200 sonnet lowprio ~30m — probe set" in out


def test_grade_daemon_line_follows_the_site_channel(monkeypatch, no_channel):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_grade_daemon import _load_daemon_module
    gd = _load_daemon_module()
    args = argparse.Namespace(concurrency=200, expected_hours=3, model_id=["synthmodel_4b_s1"], once=True, interval=60)
    assert gd.coordination_line(args) == ""
    monkeypatch.setenv(coordination.ENV, CH)
    line = gd.coordination_line(args)
    assert line.startswith(f"coordination (post to {CH}; the daemon does not wait for it): 200 haiku+sonnet lowprio 3h")
