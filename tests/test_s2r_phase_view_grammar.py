"""s2r-grammar: the view `s2r` — the ReLU-parent subset of phase 2 (control_relu +
constitution_relu at alpha 1) — sat outside the committed phase-view grammar `s[0-9]+p?`, so the sampled, misalign and GSM8K
generators rendered nothing for it while the planner still planned static / controls / agentic / gsm8k units: a static or controls job
would have refused the view as out of sampled scope, agentic could not resolve its target, GSM8K had no row. Now: every copy of
PHASE_VIEW_RE admits `s<N>r` (and still refuses `s1x`, `main`, `it`), the copies are pinned equal, the sampled and misalign generators
resolve an `s<N>r` view of a front-door key, and padding stays EXCLUDED for such a view by an explicit, tested rule (models.yaml
`exclude_families`) — the padding generator omits and refuses the view by name, the planner refuses `--family padding` for it, 06's
`--family all` drops it.
The multi-turn / BoN grader `scripts/sdfbase/seedrep/grade_full22_native.py` carried a SEVENTH
copy this pin missed (`s[0-9]+p?`, no `r?`), so the front door's `<key>_s2r` / `_s3r` tags refused by name under the grade daemon's multi-turn
dispatch; it is byte-equal now and in COPIES, so the next divergence fails here.
No model of the tree declares an `s<N>r` view now, so the rule's mechanism and the generators' scope are exercised on a SYNTHETIC copy of the
models table (under tmp) that adds one to the front-door model; the committed tree is read for the grammar copies."""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, REPRO, VENV_PY  # noqa: E402

sys.path.insert(0, str(LIB))
import models as M  # noqa: E402
import stage as ST  # noqa: E402

KEY = "qwen35_4b_demo"                 # the front-door model of reproduce/models.yaml (configured, not run)
VIEW = "s9r"                       # an `s<N>r` view declared on a SYNTHETIC copy of the table (no model of the tree declares one)
EXCLUDE_REASON = "USER RULING D889 (paper-companion-repo): the padding ladder never runs on the ReLU-parent subset's objects"
GRAMMAR = r"^(u0|s[0-9]+r?p?|s[0-9]+rl|d[0-9]+(p[0-9]+)?(_relul?)?|dm[0-9]+(p[0-9]+)?(_relul?)?)$"   # S2RL: + `s<N>rl`, the LeCun-init refit of phase N's ReLU line; D0P75RELUL: + `d<dose>_relul`, its dose lane
COPIES = ("reproduce/lib/models.py", "scripts/sdfbase/make_full22_configs.py", "scripts/sdfbase/full22/sampled/make_sampled_configs.py",
          "scripts/sdfbase/full22/misalign/make_misalign_manifest.py", "scripts/sdfbase/gsm8k/g8k_cells.py",
          "scripts/sdfbase/full22/static_padding/make_padding_configs.py",
          "scripts/sdfbase/seedrep/grade_full22_native.py")     # the seventh copy: the multi-turn / BoN grader's own, which this pin missed


def _module(rel: str):
    spec = importlib.util.spec_from_file_location("_s2r_" + Path(rel).stem, REPO / rel)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _table_with_view(tmp_path: Path, name: str, **view_fields) -> str:
    """A copy of the models table under tmp with VIEW (the two ReLU parents) added to KEY's views plus `view_fields` — the ruling's shape;
    the ONE reader loads it (duplicate keys refused)."""
    t = yaml.safe_load((REPRO / "models.yaml").read_text())
    t["models"][KEY]["views"][VIEW] = {"cells": ["control_relu", "constitution_relu"], **view_fields}
    d = tmp_path / name; d.mkdir(exist_ok=True); p = d / "models.yaml"; p.write_text(yaml.safe_dump(t, sort_keys=False))
    return str(p)


def _ruled_table(tmp_path: Path) -> str:
    return _table_with_view(tmp_path, "ruled", exclude_families=["padding"], exclude_reason=EXCLUDE_REASON)


def test_every_copy_of_the_phase_view_grammar_is_the_same_and_admits_the_relu_parent_subset():
    pats = {}
    for rel in COPIES:
        src = (REPO / rel).read_text()
        m = re.search(r'^PHASE_VIEW_RE = re\.compile\(r"(.*)"\)$', src, re.M)
        assert m, f"{rel}: no PHASE_VIEW_RE definition"
        pats[rel] = m.group(1)
    assert set(pats.values()) == {GRAMMAR}, pats                                                   # one grammar, seven copies, byte-equal
    rx = re.compile(GRAMMAR)
    for v in ("s1", "s1", "s2", "s12", "s1p", "s2r", "s12r", "s2rp", "s2rl", "s12rl", "d0p75", "d0p75_relu", "d0p75_relul", "dm1", "dm1_relu", "dm1_relul", "d1", "d4"):
        assert rx.match(v), v
    for v in ("main", "it", "it_d1", "halfdose", "fullconst", "shortconst", "s1x", "sr2", "s2pr", "s2rr", "relu", "s2_relu", "s2rlp", "s2l", "s2lr", "s2rll", "d0p75_relull", "d0p75_relu_l", "d0p75_lrelu", "d0p75l_relu"):
        assert not rx.match(v), v
    # the live readers agree with the text
    assert M.PHASE_VIEW_RE.pattern == GRAMMAR and M.PHASE_VIEW_RE.match(VIEW) and M.PHASE_VIEW_RE.match("s2r")
    assert KEY in M.front_door_keys() and all(v in M.phase_views(KEY) for v in M.views(KEY))     # the grammar's views + the row's DECLARED wave (battery; demo-import M3)
    assert M.declared_wave(KEY) == "battery" and not rx.match("battery") and all(M.PHASE_VIEW_RE.match(v) for v in M.views(KEY) if v != "battery")


def test_the_table_excludes_padding_from_an_s_r_view_by_ruling_and_the_reader_and_cli_say_so(tmp_path):
    table = _ruled_table(tmp_path)
    ex = M.excluded_families(KEY, VIEW, table)
    assert list(ex) == ["padding"] and "D889" in ex["padding"] and "padding ladder" in ex["padding"]
    wave = M.baseline_view(KEY)                                                                   # the demo's one declared wave (battery), never a literal
    assert M.excluded_families(KEY, wave, table) == {} and M.excluded_families(KEY, wave) == {}
    with pytest.raises(KeyError, match="no view 'nope'"):
        M.excluded_families(KEY, "nope")
    r = subprocess.run([sys.executable, str(LIB / "models.py"), "--table", table, "excluded-families", KEY, VIEW], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.startswith("padding\tUSER RULING D889"), r.stdout + r.stderr
    r = subprocess.run([sys.executable, str(LIB / "models.py"), "--table", table, "excluded-families", KEY, wave], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout == "", r.stdout + r.stderr
    # a list without its reason refuses by name (a ruling names its reason)
    bare = _table_with_view(tmp_path, "bare", exclude_families=["padding"])
    with pytest.raises(ValueError, match="exclude_families \\['padding'\\] without an exclude_reason"):
        M.excluded_families(KEY, VIEW, bare)


def test_the_padding_generator_omits_and_refuses_the_excluded_view_by_name_while_the_grammar_admits_it(tmp_path, monkeypatch):
    monkeypatch.setenv("SISTEER_MODELS_TABLE", _ruled_table(tmp_path))     # the generator reads the table at import, through its ONE reader
    mp = _module("scripts/sdfbase/full22/static_padding/make_padding_configs.py")
    fams = mp.front_door_view_families(KEY)
    wave = M.baseline_view(KEY)                                                                   # the demo's one declared wave (battery), never a literal
    # omitted — the other phase views stand: the grammar's dose views and the row's DECLARED wave (outside the grammar; demo-import M3)
    assert VIEW not in fams and "u0" not in fams and all(mp.PHASE_VIEW_RE.match(f) or f == wave for f in fams), fams
    assert fams == tuple(M.views(KEY, str(REPRO / "models.yaml"))), fams       # the single-phase demo: every declared view (the wave included) pads per view; no u0
    # the grammar admits the NAME; the ruling excludes the FAMILY — the name-only predicate reads the TABLE-built family names (demo-import M3),
    # which keep the excluded view's NAME (every non-u0 phase view of a front-door key), so a config name, a private run or a resolve call
    # spelling it parses and is refused by the RULING below, never as an unknown family; it renders as no family (fams omits it)
    assert mp.PHASE_VIEW_RE.match(VIEW) and mp.is_front_door_family(VIEW) is True and VIEW in mp.FRONT_DOOR_FAMILY_NAMES and VIEW in mp.ALL_FAMILY_NAMES
    assert VIEW not in fams and VIEW not in mp.FRONT_DOOR_VIEW_FAMILIES[KEY]
    assert mp.is_front_door_family(wave) is True and all(mp.is_front_door_family(f) for f in fams)
    assert mp.front_door_view_excluded(KEY, VIEW) and "D889" in mp.front_door_view_excluded(KEY, VIEW)
    assert mp.front_door_view_excluded(KEY, wave) is None and mp.front_door_view_excluded("no_such_key", VIEW) is None
    with pytest.raises(SystemExit, match=f"{VIEW}: the padding family is EXCLUDED for this view by ruling .*D889.* not rendered, not planned"):
        mp._require_front_door_family(KEY, VIEW)
    assert VIEW not in mp.families_of(KEY)
    assert not list((REPO / "configs" / "sdfbase" / "static_padding").glob(f"*{KEY}*{VIEW}*")), "a padding config exists for the excluded view"


def test_the_planner_refuses_an_explicit_padding_family_for_the_excluded_view_and_plans_the_rest(tmp_path):
    table = _ruled_table(tmp_path); sha = "a" * 40
    with pytest.raises(ValueError, match=f"{KEY}/{VIEW}: the padding family is EXCLUDED for this view by ruling .*D889.* not planned"):
        ST.unit_plan(KEY, VIEW, table=table, families=["padding"], sha=sha, clusters=["cluster_a", "cluster_b"])
    with pytest.raises(ValueError, match="EXCLUDED for this view by ruling"):
        ST.unit_plan(KEY, VIEW, table=table, families=["static", "padding"], sha=sha, clusters=["cluster_a", "cluster_b"])
    rows = ST.unit_plan(KEY, VIEW, table=table, families=["static", "controls", "agentic", "gsm8k", "mmlu"], sha=sha, clusters=["cluster_a", "cluster_b"])
    fams = sorted({r["family"] for r in rows})
    assert fams == ["agentic", "bon", "controls", "gsm8k", "mmlu", "static"], fams                           # bon = the 512 store placed by --family static
    assert sorted(r["unit"] for r in rows if r["family"] == "gsm8k") == ["constitution_relu", "control_relu"]
    # the baseline wave (the row's declared `battery`; no exclusion) still plans padding
    assert any(r["family"] == "padding" for r in ST.unit_plan(KEY, M.baseline_view(KEY, table), table=table, families=["padding"], sha=sha, clusters=["cluster_a", "cluster_b"]))
    # the CLI refuses the same way
    r = subprocess.run([sys.executable, str(LIB / "stage.py"), "plan", "--model", KEY, "--view", VIEW, "--families", "padding", "--sha", sha,
                        "--clusters", "cluster_a,cluster_b", "--models-table", table], capture_output=True, text=True)
    assert r.returncode == 2 and "EXCLUDED for this view by ruling" in r.stderr, r.stdout + r.stderr


def test_the_sampled_scope_admits_an_s_r_view_of_the_front_door_once_a_roster_names_it(tmp_path):
    gen = REPO / "scripts/sdfbase/full22/sampled/make_sampled_configs.py"
    table = _ruled_table(tmp_path)
    # a lane-private roster naming the view (what 05 --step roster prints; the production roster carries no front-door block)
    roster = tmp_path / "roster.yaml"
    roster.write_text(yaml.safe_dump({"targets": {KEY: {"views": {VIEW: {"config": f"configs/sdfbase_full22_{KEY}_{VIEW}.yaml",
                                                                            "cells": {"constv2ctl2-full__psr_ce_all": "base__constv2ctl2-full__psr_ce_all",
                                                                                      "constv2-mix-full__psr_ce_all": "base__constv2-mix-full__psr_ce_all"}}}}}}))
    env = {**os.environ, "SISTEER_MODELS_TABLE": table, "F22_ROSTER": str(roster)}
    r = subprocess.run([VENV_PY or sys.executable, str(gen), "--is-in-scope", KEY, VIEW], capture_output=True, text=True, cwd=REPO, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run([VENV_PY or sys.executable, str(gen), "--is-in-scope", KEY, VIEW + "x"], capture_output=True, text=True, cwd=REPO, env=env)
    assert r.returncode == 3, r.stdout + r.stderr
    # the scope's view list is the table's phase views: the grammar admits the name before any roster names it (pending, never refused)
    r = subprocess.run([VENV_PY or sys.executable, "-c", "import sys, importlib.util; spec = importlib.util.spec_from_file_location('ms', sys.argv[1]); "
                        "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); print(' '.join(m.FRONT_DOOR_VIEWS[sys.argv[2]])); "
                        "print(' '.join(f'{k}/{v}' for k, v in m.front_door_pending()))", str(gen), KEY],
                       capture_output=True, text=True, cwd=REPO, env={**os.environ, "SISTEER_MODELS_TABLE": table})
    assert r.returncode == 0, r.stdout + r.stderr
    views, pending = r.stdout.splitlines()[:2]
    assert VIEW in views.split() and f"{KEY}/{VIEW}" in pending.split(), r.stdout


def test_the_misalign_generator_resolves_an_s_r_view_of_the_front_door_as_a_front_door_target(tmp_path):
    gen = REPO / "scripts/sdfbase/full22/misalign/make_misalign_manifest.py"
    env = {**os.environ, "SISTEER_MODELS_TABLE": _ruled_table(tmp_path), "PYTHONPATH": str(REPO / "src")}
    r = subprocess.run([VENV_PY or sys.executable, str(gen), "--resolve", f"{KEY}_{VIEW}"], capture_output=True, text=True, cwd=REPO, env=env)
    assert r.returncode == 0 and r.stdout.strip() == f"{KEY} {VIEW} front_door", r.stdout + r.stderr
    wave = M.baseline_view(KEY)                                                                   # the demo's one declared wave (battery), never a literal
    r = subprocess.run([VENV_PY or sys.executable, str(gen), "--resolve", f"{KEY}_{wave}"], capture_output=True, text=True, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert r.returncode == 0 and r.stdout.strip() == f"{KEY} {wave} front_door", r.stdout + r.stderr    # the table's own declared wave, no synthetic row needed
