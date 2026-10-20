"""The sampled generator KNOWS the front door's phase views (`run_identity` in grade_static_sampled refused a front-door
model's TAGs because the scope named the paper's models only):

* `make_sampled_configs.FRONT_DOOR_VIEWS` == every models-table key's phase views (the seedrep twin's / misalign generator's grammar),
  read through the table's ONE reader; `SAMPLED_VIEWS` (the paper's production list) is byte-identical; an absent table derives
  nothing; a `view_legacy_prefix` that differs from the key refuses; a key SAMPLED_VIEWS already lists refuses;
* TODAY (the production roster does not carry the front-door model) the scope is UNCHANGED: `sampled_tags()` is the paper's 69, the shell
  gate answers 3 for a front-door view, `outputs()` is the 405, `check_pins` passes, and `main` names every pending TAG;
* the moment a production roster names a front-door (key, view) the TAG is in scope — `in_scope`, `split_tag`, `sampled_tags()`,
  `outputs()` (two trials × base + two rungs, the campaign's count and seed base) and `render()` over its u0 wave config carry the
  KEY's preset (the shipped generation_config of the served checkpoint), and `check_pins` holds the preset to the roster's pins;
  the graders' `KEYS = tuple(sampled_tags())` therefore admits the model exactly when its runs can exist;
* a lane-private roster keeps its own semantics (the private roster's views are the scope for every key)."""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SAMPLED = REPO / "scripts" / "sdfbase" / "full22" / "sampled"
TABLE = REPO / "reproduce" / "models.yaml"
READER = REPO / "reproduce" / "lib" / "models.py"
for p in (SAMPLED, REPO / "scripts" / "sdfbase" / "full22", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import make_sampled_configs as msc  # noqa: E402

ENV = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin", "HOME": str(REPO)}
KEY = "qwen35_4b_demo"


def _reader():
    spec = importlib.util.spec_from_file_location("_fd_models_reader_sampled", READER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fresh(table: Path | None = None):
    """A fresh module object (its import-time derivation re-run), under SISTEER_MODELS_TABLE when given."""
    if table is not None:
        os.environ["SISTEER_MODELS_TABLE"] = str(table)
    else:
        os.environ.pop("SISTEER_MODELS_TABLE", None)
    try:
        spec = importlib.util.spec_from_file_location("_msc_fd_fresh", SAMPLED / "make_sampled_configs.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.environ.pop("SISTEER_MODELS_TABLE", None)
    return mod


# --------------------------------------------------------------------------- the derivation
def test_front_door_views_derive_from_the_models_table_and_sampled_views_is_untouched(tmp_path):
    reader = _reader()
    table = reader.load_models(str(TABLE))
    # the reader's ONE classification: the grammar's views plus a row's declared wave (`battery`; demo-import M3)
    expect = {k: tuple(reader.phase_views_of(m)) for k, m in table.items()}
    expect = {k: v for k, v in expect.items() if v}
    assert all(msc.PHASE_VIEW_RE.match(v) or v == table[k].get("wave") for k, vs in expect.items() for v in vs)
    assert expect and msc.FRONT_DOOR_VIEWS == expect and KEY in expect
    for k, m in table.items():
        if k not in expect:
            assert set(m.get("views") or {}) <= {"main", "it"}, k                   # the paper's four contribute nothing
        else:
            assert m.get("view_legacy_prefix") == k                                  # one spelling per wave
    assert not (set(msc.FRONT_DOOR_VIEWS) & set(msc.SAMPLED_VIEWS))
    assert msc.scope_keys() == (*msc.SAMPLED_VIEWS, *msc.FRONT_DOOR_VIEWS)
    assert msc.front_door_views(tmp_path / "absent.yaml") == {}
    bad = tmp_path / "shadow.yaml"
    bad.write_text(yaml.safe_dump({"schema": 1, "models": {"qwen35_4b": {"view_legacy_prefix": "qwen35_4b", "views": {"s1": {}}}}}))
    with pytest.raises(SystemExit, match="one key, one scope list"):
        msc.front_door_views(bad)
    bad2 = tmp_path / "prefix.yaml"
    bad2.write_text(yaml.safe_dump({"schema": 1, "models": {"mk": {"view_legacy_prefix": "other", "views": {"s1": {}}}}}))
    with pytest.raises(SystemExit, match="one spelling per wave"):
        msc.front_door_views(bad2)
    dup = tmp_path / "dup.yaml"
    dup.write_text("schema: 1\nmodels:\n  a: {view_legacy_prefix: a, views: {u0: {}}}\n  a: {view_legacy_prefix: a, views: {u0: {}}}\n")
    with pytest.raises(Exception, match="duplicate key"):
        msc.front_door_views(dup)


def test_the_scope_is_the_papers_plus_exactly_the_front_door_views_the_roster_carries():
    """ROSTER-CONDITIONAL (the fix-1 owner follow-up 10): before stage 05 lands the test model's block, the scope is the paper's 69 and
    every front-door TAG is pending BY NAME; once the production roster carries a front-door view, that TAG is in scope AFTER the paper's
    69, in the table's order, and is no longer pending — so stage 05 on the real site does not turn this suite red."""
    roster = msc.load_roster()
    assert not msc._private()
    tags = msc.sampled_tags()
    carried = [(k, v) for k, vs in msc.FRONT_DOOR_VIEWS.items() for v in vs if v in ((roster["targets"].get(k) or {}).get("views") or {})]
    paper = [msc.tag_of(k, v) for k in msc.SAMPLED_VIEWS for v in msc.SAMPLED_VIEWS[k]]           # SAMPLED_VIEWS' order, untouched
    assert len(paper) == 69 and tags == paper + [msc.tag_of(k, v) for k, v in carried]
    for k, views in msc.FRONT_DOOR_VIEWS.items():
        assert msc.scope_views(k) == tuple(v for v in views if (k, v) in carried) and all(msc.in_scope(k, v) == ((k, v) in carried) for v in views)
    assert set(msc.front_door_pending()) == {(k, v) for k, vs in msc.FRONT_DOOR_VIEWS.items() for v in vs} - set(carried)
    if (KEY, "battery") in carried:
        assert msc.split_tag(f"{KEY}_battery") == (KEY, "battery")
    else:
        with pytest.raises(SystemExit, match="not a \\(target, view\\) in the sampled scope"):
            msc.split_tag(f"{KEY}_battery")
    presets = msc.load_presets()
    msc.check_pins(presets, roster)                                                # no refusal: pending is not an error; carried views are pinned
    for k in msc.FRONT_DOOR_VIEWS:
        assert k in presets["presets"], k                                          # the KEY's row exists (C-E seeded it)
    assert len(list(msc.outputs(presets))) == 405 + sum(msc.n_trials_of_tag(presets, msc.tag_of(k, v)) * 3 for k, v in carried)
    # the shell gate: 3 (not in scope) with the scope on stderr, never a query failure — 0 once the roster carries the view; --tags unchanged
    r = subprocess.run([sys.executable, str(SAMPLED / "make_sampled_configs.py"), "--is-in-scope", KEY, "battery"], capture_output=True, text=True, cwd=REPO, env=ENV)
    if (KEY, "battery") in carried:
        assert r.returncode == 0, r.stderr
    else:
        assert r.returncode == 3 and f"{KEY}/battery is not in the sampled scope" in r.stderr
    r = subprocess.run([sys.executable, str(SAMPLED / "make_sampled_configs.py"), "--tags"], capture_output=True, text=True, cwd=REPO, env=ENV)
    assert r.returncode == 0 and r.stdout.split() == tags


# --------------------------------------------------------------------------- once the roster names the views
def _front_door_roster(msc_mod, key: str, views: tuple) -> dict:
    """The production roster + the test model's block as 05 --step roster prints it (the served model / revision from the preset row)."""
    roster = yaml.safe_load(msc_mod.ROSTER.read_text(encoding="utf-8"))
    p = msc_mod.load_presets()["presets"][key]
    roster["targets"][key] = {"model": p["model"], "model_revision": p["revision"], "bank": "full22",
                              "views": {v: {"config": f"configs/sdfbase_full22_{key}_{v}.yaml",
                                            "manifest": f"configs/capeval/manifests/petri_full22_{key}_{v}.yaml", "cells": {}} for v in views}}
    return roster


def test_a_front_door_view_the_roster_names_enters_the_scope_and_renders_under_the_keys_preset(tmp_path, monkeypatch):
    mod = _fresh()
    views = mod.FRONT_DOOR_VIEWS[KEY]
    roster = _front_door_roster(mod, KEY, views)
    rpath = tmp_path / "roster.yaml"; rpath.write_text(yaml.safe_dump(roster))
    monkeypatch.setattr(mod, "ROSTER", rpath); monkeypatch.setattr(mod, "_ROSTER_PATH", rpath)      # a PRODUCTION roster that carries the block
    assert not mod._private()
    assert mod.scope_views(KEY) == views and mod.front_door_pending() == [(k, v) for k, vs in mod.FRONT_DOOR_VIEWS.items() if k != KEY for v in vs]
    tags = mod.sampled_tags()
    paper = [msc.tag_of(k, v) for k in msc.SAMPLED_VIEWS for v in msc.SAMPLED_VIEWS[k]]           # the paper's 69 (never the live scope, which may already carry the front door)
    assert len(paper) == 69 and tags[:69] == paper and tags[69:] == [f"{KEY}_{v}" for v in views]  # the paper's 69 first, byte for byte
    for v in views:
        assert mod.in_scope(KEY, v) and mod.split_tag(f"{KEY}_{v}") == (KEY, v) and mod.key_of(f"{KEY}_{v}") == KEY
    presets = mod.load_presets()
    mod.check_pins(presets, roster)                                                # the preset's model / revision == the roster's
    bad = {**roster, "targets": {**roster["targets"], KEY: {**roster["targets"][KEY], "model_revision": "0" * 40}}}
    with pytest.raises(SystemExit, match="was read from"):
        mod.check_pins(presets, bad)
    outs = list(mod.outputs(presets))
    names = [p.name for p, *_ in outs]
    assert len(outs) == 405 + len(views) * 2 * 3                                    # two trials x base + two rungs per new TAG
    for v in views:
        for t in (0, 1):
            for s in ("", "_cap8192", "_cap16384"):
                assert f"sdfbase_full22_{KEY}_{v}_sampled_t{t}{s}.yaml" in names
    # render the battery TAG (the single-phase recipe's one wave) over a synthetic wave config (the real qwen35_4b wave config renamed): the KEY's
    # preset rides the block — the paper's Qwen3.5-4B card row (the demo serves the same checkpoint, which ships no generation_config.json)
    tree = tmp_path / "repo"; (tree / "configs").mkdir(parents=True)
    src = (REPO / "configs" / "sdfbase_full22_qwen35_4b.yaml").read_text(encoding="utf-8")
    src = re.sub(r"^out_dir: out/sdfbase_full22_qwen35_4b$", f"out_dir: out/sdfbase_full22_{KEY}_battery", src, flags=re.M)
    src = re.sub(r"^private_run: sdfbase_full22_qwen35_4b$", f"private_run: sdfbase_full22_{KEY}_battery", src, flags=re.M)
    (tree / "configs" / f"sdfbase_full22_{KEY}_battery.yaml").write_text(src, encoding="utf-8")
    monkeypatch.setattr(mod, "REPO", tree)
    for esc in (False, 8192, 16384):
        text = mod.render(presets, f"{KEY}_battery", 0, esc)
        cfg = yaml.safe_load(text)
        blk = cfg["sampled_decode"]
        assert blk["trial"] == 0 and blk["n_trials"] == 2 and blk["seed_base"] == 20260909 and blk["arm"] == "vendor_recommended"
        assert {k: blk[k] for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")} == presets["presets"][KEY]["preset"] \
            == presets["presets"]["qwen35_4b"]["preset"] == {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0}
        assert blk["source_kind"] == "model_card" and "Qwen/Qwen3.5-4B" in blk["source"]
        # the GSM8K readout runs thinking ON under the vendor's THINKING row, which the data file carries under the demo key as under
        # qwen35_4b (the same checkpoint; 2026-10-08) — not the single-preset fallback
        assert blk["readouts"]["gsm8k"]["thinking"] == "on" and blk["readouts"]["gsm8k"]["preset"] == presets["readouts"]["gsm8k"]["presets"][KEY]["preset"] \
            == presets["readouts"]["gsm8k"]["presets"]["qwen35_4b"]["preset"] and blk["readouts"]["gsm8k"]["preset"]["temperature"] == 1.0
        assert cfg["private_run"] == mod.sampled_name(f"{KEY}_battery", 0, esc) and blk["static_cap"] == cfg["attacks"]["static_max_new_tokens"]
        assert "Qwen/Qwen3.5-4B @ 851bf6e8" in text.split("\n", 12)[8] or "Qwen/Qwen3.5-4B" in text[:2000]
    assert mod.expected_block(presets, f"{KEY}_battery", 1)["trial"] == 1


def test_a_lane_private_roster_keeps_its_own_scope_and_names_no_front_door_view(tmp_path):
    mod = _fresh()
    private = tmp_path / "private_roster.yaml"
    private.write_text(yaml.safe_dump({"targets": {"qwen35_4b": {"views": {"dm1_ctllora64s1": {}}}}}))
    mod._ROSTER_PATH = private
    assert mod._private() and mod.scope_views("qwen35_4b") == ("dm1_ctllora64s1",)
    assert all(mod.scope_views(k) == () for k in mod.FRONT_DOOR_VIEWS) and mod.front_door_pending() == []
    assert mod.sampled_tags() == ["qwen35_4b_dm1_ctllora64s1"]
