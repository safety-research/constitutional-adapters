"""The misalign generator KNOWS the front door's phase views — DERIVED from the models table, never a hand list:

* `make_misalign_manifest.FRONT_DOOR` == every `<view_legacy_prefix>_<view>` of `reproduce/models.yaml` whose view matches the phase-view
  grammar (the seedrep grader twin's PHASE_VIEW_RE), read through the table's ONE reader; the paper's four models (`main` / `it`) derive
  nothing; an absent table derives nothing; a prefix + view two models share refuses; a derived key that shadows a hand-listed one
  refuses at import; the hand-listed targets are byte-identical and come first in TARGETS;
* a front-door view renders from its OWN wave pair under the FRONT-DOOR POLICY: served == measured == EXACTLY the view's Petri cells
  (`none` first, the ReLU arms KEPT, the prompted arms iff the roster lists them on that view, nothing appended, no alphasteer, no
  `misalign_policy` block), both files validating as the standup would; a manifest listing `alphasteer` or not starting with `none`
  refuses by name;
* every pre-existing target's committed files are byte-identical (`--check`) and the front-door targets are SKIPPED, by name, until
  their wave pair lands on the tree (05 --step configs renders it once the roster carries the views);
* `--resolve <target>` prints `<roster target> <view> <kind>` for any key and refuses an unknown one; misalign_proxy.sbatch derives a
  front-door target's (KEY, VIEW) from it instead of a second copy of the table, keeping every hand-listed `case` line.

Everything synthetic lives under tmp_path (the real qwen35_4b wave pair with its cells swapped; the test model's tag spellings come
from the table, never typed twice)."""
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
MIS = REPO / "scripts" / "sdfbase" / "full22" / "misalign"
GEN = MIS / "make_misalign_manifest.py"
TABLE = REPO / "reproduce" / "models.yaml"
READER = REPO / "reproduce" / "lib" / "models.py"
for p in (REPO / "src",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sisteer.capeval.petri import registry_adapter as ra  # noqa: E402

ENV = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin", "HOME": str(REPO)}


def _load(name: str = "_mm_front_door", *, table: Path | None = None):
    if table is not None:
        os.environ["SISTEER_MODELS_TABLE"] = str(table)
    else:
        os.environ.pop("SISTEER_MODELS_TABLE", None)
    try:
        spec = importlib.util.spec_from_file_location(name, GEN)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.environ.pop("SISTEER_MODELS_TABLE", None)
    return mod


def _reader():
    spec = importlib.util.spec_from_file_location("_fd_models_reader", READER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _expected_front_door(mm) -> dict:
    """The derivation re-done from the table by hand: {tag: (key, view)} over the phase views — the reader's ONE classification, the grammar's
    views plus a row's declared wave (never a typed list)."""
    reader = _reader()
    table = reader.load_models(str(TABLE))
    out = {f"{m['view_legacy_prefix']}_{v}": (k, v) for k, m in table.items() for v in (m.get("views") or {}) if v in reader.phase_views_of(m)}
    assert all(mm.PHASE_VIEW_RE.match(v) or v == table[k].get("wave") for k, v in out.values())
    return out


# --------------------------------------------------------------------------- the derivation
def test_front_door_targets_derive_from_the_models_table_and_the_hand_listed_targets_are_untouched(tmp_path):
    mm = _load()
    expect = _expected_front_door(mm)
    assert expect and mm.FRONT_DOOR == expect
    keys = {k for k, _v in expect.values()}
    assert {"qwen35_4b_demo"} <= keys                                        # the front-door model; the paper's four carry main / it only
    table = _reader().load_models(str(TABLE))
    for k, m in table.items():
        if k not in keys:
            assert set(m.get("views") or {}) <= {"main", "it"}, (k, m.get("views"))
    for tag, (key, view) in expect.items():
        assert tag in mm.TARGETS and mm.is_front_door(tag) and mm.wave_of(tag) == (key, view)
        assert mm.target_kind(tag) == mm.FRONT_DOOR_STATUS and mm.source_stem(tag) == f"{key}_{view}"
        assert mm.alphasteer_status(tag) == (mm.FRONT_DOOR_STATUS, {}) and mm.lane_policy(tag) is None
    # the hand-listed targets are the prefix of TARGETS, byte for byte, and no front-door key is among them
    assert mm.TARGETS[:len(mm.HAND_LISTED_TARGETS)] == mm.HAND_LISTED_TARGETS and not (set(mm.HAND_LISTED_TARGETS) & set(expect))
    assert mm.TARGETS == (*mm.HAND_LISTED_TARGETS, *expect)
    for t in ("qwen35_4b_s1", "qwen35_4b_s1p", "gemma4e4b_bl", "llama8b_org", "qwen35_4b"):
        assert t in mm.HAND_LISTED_TARGETS and not mm.is_front_door(t)
    # the grammar admits the plan's phase views and nothing the paper's roster calls a view
    for v in ("s1", "s1", "s12", "s1p", "d0p75", "d0p75_relu", "d0p75_relul", "dm1", "dm1_relu", "d1", "d4"):
        assert mm.PHASE_VIEW_RE.match(v), v
    for v in ("main", "it", "it_d1", "halfdose", "s1x", "relu"):
        assert not mm.PHASE_VIEW_RE.match(v), v
    # an absent table contributes nothing (the paper's tree before the front door); the reader is the one reader
    assert mm.front_door_targets(tmp_path / "no_such_table.yaml") == {}
    # two models sharing a prefix + view refuse (one key, one wave pair)
    t2 = tmp_path / "two.yaml"
    t2.write_text(yaml.safe_dump({"schema": 1, "models": {"a": {"view_legacy_prefix": "zz", "views": {"s1": {}}},
                                                          "b": {"view_legacy_prefix": "zz", "views": {"s1": {}}}}}))
    with pytest.raises(SystemExit, match="both derive the misalign target 'zz_s1'"):
        mm.front_door_targets(t2)
    # a duplicate mapping key in the table refuses through the reader, never read another way
    t3 = tmp_path / "dup.yaml"
    t3.write_text("schema: 1\nmodels:\n  a: {view_legacy_prefix: aa, views: {u0: {}}}\n  a: {view_legacy_prefix: ab, views: {u0: {}}}\n")
    with pytest.raises(Exception, match="duplicate key"):
        mm.front_door_targets(t3)


def test_a_derived_key_that_shadows_a_hand_listed_target_refuses_at_import(tmp_path):
    t = tmp_path / "shadow.yaml"
    t.write_text(yaml.safe_dump({"schema": 1, "models": {"m": {"view_legacy_prefix": "qwen35_4b", "views": {"s1": {}}}}}))
    r = subprocess.run([sys.executable, str(GEN), "--resolve", "qwen35_4b_s1"], capture_output=True, text=True, cwd=REPO,
                       env={**ENV, "SISTEER_MODELS_TABLE": str(t)})
    assert r.returncode != 0 and "collide with hand-listed" in r.stdout + r.stderr and "qwen35_4b_s1" in r.stdout + r.stderr
    # SISTEER_MODELS_TABLE is honoured (the stage scripts' override): a table with a new model derives its views
    t2 = tmp_path / "new.yaml"
    t2.write_text(yaml.safe_dump({"schema": 1, "models": {"newmodel": {"view_legacy_prefix": "newmodel", "views": {"s1": {}, "d2_relu": {}, "main": {}}}}}))
    r = subprocess.run([sys.executable, str(GEN), "--resolve", "newmodel_d2_relu"], capture_output=True, text=True, cwd=REPO,
                       env={**ENV, "SISTEER_MODELS_TABLE": str(t2)})
    assert r.returncode == 0 and r.stdout.strip() == "newmodel d2_relu front_door", r.stdout + r.stderr
    r = subprocess.run([sys.executable, str(GEN), "--resolve", "newmodel_main"], capture_output=True, text=True, cwd=REPO,
                       env={**ENV, "SISTEER_MODELS_TABLE": str(t2)})
    assert r.returncode == 2 and "unknown misalign target 'newmodel_main'" in r.stderr


# --------------------------------------------------------------------------- --resolve
def test_resolve_prints_roster_target_view_and_kind_for_every_key_and_refuses_an_unknown_one():
    mm = _load()
    fd = sorted(mm.FRONT_DOOR)
    cases = {fd[0]: f"{mm.FRONT_DOOR[fd[0]][0]} {mm.FRONT_DOOR[fd[0]][1]} front_door", "qwen35_4b_s1": "qwen35_4b s1 lane:reseed",
             "qwen35_4b_s2p": "qwen35_4b s2p lane:paired", "gemma4e4b_bl": "gemma4e4b main baselines_panel",
             "llama8b_org": "llama8b main organisms_panel", "qwen35_4b_psr": "qwen35_4b main psr_panel",
             "qwen35_4b_allpos": "qwen35_4b it allpos_panel", "llama8b_it": "llama8b it campaign", "nemotron3nano4b": "nemotron3nano4b main campaign"}
    for target, want in cases.items():
        r = subprocess.run([sys.executable, str(GEN), "--resolve", target], capture_output=True, text=True, cwd=REPO, env=ENV)
        assert r.returncode == 0 and r.stdout.strip() == want, (target, r.stdout, r.stderr)
        assert mm.resolve(target) == tuple(want.split())
    r = subprocess.run([sys.executable, str(GEN), "--resolve", "no_such_target"], capture_output=True, text=True, cwd=REPO, env=ENV)
    assert r.returncode == 2 and "unknown misalign target 'no_such_target'" in r.stderr and "qwen35_4b_s1" in r.stderr and fd[0] in r.stderr
    with pytest.raises(SystemExit, match="unknown misalign target"):
        mm.resolve("no_such_target")


# --------------------------------------------------------------------------- rendering a front-door view
PROMPTED_H = ["prompted__safety", "prompted__constitution"]
RESIDUAL_H = ["arditi_allpos", "si_constv2ctl2_full__psr_ce_all", "si_constv2_mix_full__psr_ce_all", "si_constv2_mix_full__psr_ce_all_minus_ctl_distilled"]
LORA_H = ["si_constv2ctl2_full__lora_r4_ce", "si_constv2_mix_full__lora_r4_ce", "si_constv2_mix_full__lora_r4_ce_minus_ctl"]
#: the phase views the synthetic tree renders, as (view, harness cells) — the shape of models.yaml's battery / d0p75 / d0p75_relu views under the
#: SINGLE-PHASE recipe: every view carries its own `none` anchor; battery (the row's declared wave) is the ONE wave (the prompted legs + every arm)
VIEW_CELLS = {"battery": ["none", *PROMPTED_H, RESIDUAL_H[0], *LORA_H, *RESIDUAL_H[1:]], "d0p75": ["none", LORA_H[2]],
              "d0p75_relu": ["none", "si_constv2_mix_full__psr_ce_all_minus_ctl_distilled"]}


def _front_door_tree(tmp_path, mm, key: str, views: dict):
    """A REPO-shaped tmp tree holding one wave pair per view of ``key``, cut from the real qwen35_4b pair with the cells swapped."""
    root = tmp_path / "repo"
    (root / "configs" / "capeval" / "manifests").mkdir(parents=True)
    src_m = (REPO / "configs/capeval/manifests/petri_full22_qwen35_4b.yaml").read_text()
    src_c = (REPO / "configs/sdfbase_full22_qwen35_4b.yaml").read_text()
    marker = "# -- the explicit cell enumeration (ordered; the registry is EXACTLY this) ---\nexplicit_cells:\n"
    for view, cells in views.items():
        before, _cells, after = mm._split_cells_block(src_m)
        before = re.sub(r"^  audit: petri_full22_\S+$", f"  audit: petri_full22_{key}_{view}", before, flags=re.M)
        (root / f"configs/capeval/manifests/petri_full22_{key}_{view}.yaml").write_text(
            before + "\n  cells:\n" + "\n".join(f"    - {c}" for c in cells) + "\n" + after)
        head, tail = src_c.split(marker)
        lines = tail.split("\n")
        k = 0
        while lines[k].startswith("- "):
            k += 1
        tt = [ra.traintest_cell_id(c) for c in cells]
        new_tail = "\n".join([f"- {c}" for c in tt] + lines[k:])
        new_tail = re.sub(r"^defenses_dir: .*$", f"defenses_dir: out/full22/{key}/cells_view/{view}", new_tail, flags=re.M)
        new_tail = re.sub(r"^private_run: .*$", f"private_run: sdfbase_full22_{key}_{view}", new_tail, flags=re.M)
        (root / f"configs/sdfbase_full22_{key}_{view}.yaml").write_text(head + marker + new_tail)
    for attr, rel in (("REPO", ""), ("MANIFEST_DIR", "configs/capeval/manifests"), ("ROSTER", "configs/sdfbase_full22_roster.yaml"),
                      ("PINS", "configs/sdfbase_full22_pins.json"), ("ALPHASTEER_PINS", "configs/alphasteer_f22_pins.json"),
                      ("ALPHASTEER_ROSTER", "configs/alphasteer_f22_roster.yaml"), ("BASELINE_PINS", "configs/misalign_baseline_pins.json")):
        setattr(mm, attr, root / rel if rel else root)
    return root


def test_a_front_door_view_renders_exactly_its_cells_relu_kept_prompted_as_listed_no_alphasteer_no_policy_block(tmp_path):
    mm = _load("_mm_fd_render")
    key = next(k for k, _v in mm.FRONT_DOOR.values() if k == "qwen35_4b_demo")          # the front-door model's key, from the table
    views = {v: c for v, c in VIEW_CELLS.items() if f"{key}_{v}" in mm.FRONT_DOOR}
    assert set(views) == set(VIEW_CELLS), (sorted(views), "the table no longer carries one of the synthetic views — update VIEW_CELLS")
    _front_door_tree(tmp_path, mm, key, views)
    for view, petri in views.items():
        target = f"{key}_{view}"
        assert mm.wave_sources_present(target) and mm.lane_cells_pending(target) is None and target in mm.present_targets()
        m_text, cells = mm.render_manifest(target)
        c_text, tt = mm.render_config(target, cells)
        (tmp_path / f"v_{view}").mkdir()
        mm.validate(target, m_text, c_text, tmp_path / f"v_{view}")
        assert cells == petri, (view, cells)                                   # EXACTLY the view's Petri cells: nothing appended, nothing dropped
        assert tt == [ra.traintest_cell_id(c) for c in petri]
        cfg = yaml.safe_load(c_text)
        steered = False                                                         # the single-phase demo: none of its views is a steered wave (each measures its own none)
        assert not mm.is_steered_wave_target(target) and cells[0] == "none"
        if steered:
            assert "none" not in cells and cfg["explicit_cells"] == tt and cfg["misalign_policy"]["measure_none"] is False and cfg["misalign_policy"]["measured_cells"] == tt
            assert "STEERED-WAVE" in c_text and "measure_none: false" in c_text
        else:
            assert cfg["explicit_cells"] == tt and "misalign_policy" not in cfg   # every served leg measured: a wave that carries its own unsteered legs
        assert cfg["private_run"] == f"misalign_full22_{target}" and cfg["out_dir"] == f"out/misalign_full22_{target}"
        assert cfg["defenses_dir"] == f"out/misalign/{target}/cells_view"
        assert "alphasteer" not in cells and all("psr" in c for c in petri if "psr" in c)   # the ReLU arms are KEPT (FD 5)
        if view in ("d0p75", "d0p75_relu"):
            assert not any(c.startswith("prompted__") for c in cells)         # no prompted arm appended where the view lists none
            assert "no prompted cell on this phase view" in m_text
        else:
            assert [c for c in cells if c.startswith("prompted__")] == PROMPTED_H
            assert "verified per request (H8) for the prompted cell(s) prompted__safety, prompted__constitution" in m_text
        _b, man_cells, _a = mm._split_cells_block(m_text)
        assert man_cells == petri and f"  audit: misalign_full22_{target}" in m_text
        assert "FRONT DOOR" in m_text and "FRONT-DOOR POLICY" in c_text and "nothing appended" in c_text and "nothing dropped" in c_text
        if steered:
            assert "STEERED-WAVE" in m_text and "STEERED-WAVE" in c_text           # the manifest and the config name the steered-wave policy
        else:
            assert "FRONT-DOOR POLICY" in m_text and "no `misalign_policy` block" in c_text
        assert "no alphasteer cell: the front door serves exactly the view's cells" in c_text
        pending = re.findall(r"^#\s+(\S+)\s+<- \(staging pending: the roster's " + re.escape(key) + "/" + re.escape(view), c_text, re.M)
        assert pending == [c for c in tt if c != "none" and not c.startswith("prompted__")], (view, pending)   # every object cell awaits the roster + pins
    # the generator's summary names the policy
    outs = mm.outputs()
    names = sorted(str(p.relative_to(mm.REPO)) for p, _t in outs)
    assert names == sorted([f"configs/capeval/manifests/misalign_full22_{key}_{v}.yaml" for v in views] + [f"configs/misalign_full22_{key}_{v}.yaml" for v in views])


def test_a_steered_wave_front_door_view_renders_exactly_its_cells_with_no_none_and_a_measure_none_false_policy(tmp_path):
    """s3r-fix: a steered-wave phase view (s<N>r) lists NO `none` in its Petri manifest; its misalign pair serves exactly
    its cells and carries `misalign_policy.measure_none: false` (the standup's implicit base route is never measured); a steered-wave
    manifest that lists `none` refuses by name. No model of the table names an s<N>r view, so the view rides a SYNTHETIC table (an invented
    key that RUNS IN PHASES — it declares u0; the generator derives its targets from whichever table SISTEER_MODELS_TABLE names). The tree's
    demo runs the single-phase recipe (no u0), so none of its own views is a steered wave."""
    t = tmp_path / "steered.yaml"
    t.write_text(yaml.safe_dump({"schema": 1, "models": {"synth_fd": {"view_legacy_prefix": "synth_fd", "views": {"u0": {}, "s1": {}, "s2": {}, "s2r": {}}}}}))
    mm = _load("_mm_fd_steered", table=t)
    key = "synth_fd"
    assert f"{key}_s2r" in mm.FRONT_DOOR and mm.is_steered_wave_target(f"{key}_s2r") and mm.is_steered_wave_target(f"{key}_s2") and mm.is_steered_wave_target(f"{key}_s1")   # Every front-door view but u0
    assert not mm.is_steered_wave_target(f"{key}_u0") and key not in mm.SINGLE_PHASE_KEYS         # a declared u0: the phased recipe
    assert "qwen35_4b_demo" in _load("_mm_fd_steered_default").SINGLE_PHASE_KEYS                    # the tree's demo: single-phase (no u0)
    assert mm.STEERED_WAVE_VIEW_RE.pattern == r"^s[0-9]+rl?$"   # S2RL
    steered = [RESIDUAL_H[1], RESIDUAL_H[2]]
    _front_door_tree(tmp_path, mm, key, {"s2r": steered})
    target = f"{key}_s2r"
    m_text, cells = mm.render_manifest(target)
    c_text, tt = mm.render_config(target, cells)
    (tmp_path / "v").mkdir(); mm.validate(target, m_text, c_text, tmp_path / "v")
    assert cells == steered and "none" not in cells and tt == [ra.traintest_cell_id(c) for c in steered]
    cfg = yaml.safe_load(c_text)
    assert cfg["explicit_cells"] == tt and cfg["misalign_policy"]["measure_none"] is False and cfg["misalign_policy"]["measured_cells"] == tt
    assert cfg["misalign_policy"]["lane"] == "front_door_steered_wave" and cfg["misalign_policy"]["pairing_anchor"] == f"misalign_full22_{key}_u0/agentic/none"
    assert "STEERED-WAVE" in c_text and "measure_none: false" in c_text
    with pytest.raises(SystemExit, match="lists an unsteered leg"):
        mm.render_config(target, ["none", *steered])                           # a steered-wave misalign config never lists none
    _front_door_tree(tmp_path / "bad", mm, key, {"s2r": ["none", *steered]})
    with pytest.raises(SystemExit, match="lists an unsteered leg"):
        mm.render_manifest(target)


def test_a_front_door_manifest_listing_alphasteer_or_not_starting_with_none_refuses(tmp_path):
    mm = _load("_mm_fd_refuse")
    key = "qwen35_4b_demo"
    assert f"{key}_battery" in mm.FRONT_DOOR and f"{key}_s1" not in mm.FRONT_DOOR       # the declared wave, never its former grammar spelling
    _front_door_tree(tmp_path, mm, key, {"battery": ["none", *LORA_H, "alphasteer"], "d0p75": [LORA_H[2], "none"]})   # the demo's views carry their own none, first
    with pytest.raises(SystemExit, match="lists `alphasteer` — not one of the front door's arms"):
        mm.render_manifest(f"{key}_battery")
    with pytest.raises(SystemExit, match="must list `none` first"):
        mm.render_manifest(f"{key}_d0p75")
    # the config side: a misalign cell list that is not the wave's refuses (served == the wave's view, exactly)
    _front_door_tree(tmp_path / "b", mm, key, {"battery": ["none", *LORA_H]})
    with pytest.raises(SystemExit, match="serves EXACTLY the wave's battery view cells"):
        mm.render_config(f"{key}_battery", ["none", *LORA_H[:2]])


# --------------------------------------------------------------------------- the real tree
def test_every_pre_existing_target_is_byte_identical_and_the_front_door_targets_are_skipped_until_their_wave_lands():
    """ROSTER-CONDITIONAL (the fix-1 owner follow-up 10): a front-door tag whose wave pair is NOT on this tree is skipped by name; one whose
    pair 05 --step configs rendered (the roster carries the view) checks `ok:` like every hand-listed target."""
    mm = _load("_mm_fd_check")
    r = subprocess.run([sys.executable, str(GEN), "--check"], capture_output=True, text=True, cwd=REPO, env=ENV)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "DRIFT" not in r.stdout
    for tag in mm.FRONT_DOOR:
        if mm.wave_sources_present(tag):
            assert f"ok: configs/misalign_full22_{tag}.yaml" in r.stdout, (tag, r.stdout[-3000:])
            continue
        assert f"skip {tag}: its wave's configs are not on this tree" in r.stdout, (tag, r.stdout[-3000:])
        assert not (REPO / f"configs/misalign_full22_{tag}.yaml").exists() and not (REPO / f"configs/capeval/manifests/misalign_full22_{tag}.yaml").exists()
    for t in mm.HAND_LISTED_TARGETS:
        if mm.wave_sources_present(t) and mm.lane_cells_pending(t) is None:
            assert f"ok: configs/misalign_full22_{t}.yaml" in r.stdout, t


# --------------------------------------------------------------------------- the cluster script
def test_misalign_proxy_derives_a_front_door_target_from_the_generator_and_keeps_every_hand_listed_case():
    px = (MIS / "misalign_proxy.sbatch").read_text()
    assert subprocess.run(["bash", "-n", str(MIS / "misalign_proxy.sbatch")]).returncode == 0
    case = px[px.index('case "$TARGET" in'):]
    case = case[:case.index("\nesac")]
    assert '*) f22_fatal "unknown target $TARGET";;' not in px                   # the closed default is gone
    assert 'make_misalign_manifest.py --resolve "$TARGET"' in case and 'read -r KEY VIEW KIND' in case
    assert case.index("f22_driver_venv") < case.index('"$MPY" scripts/sdfbase/full22/misalign/make_misalign_manifest.py --resolve')   # the resolver's venv first
    assert '[ "$KIND" = front_door ] || f22_fatal' in case                       # only a front-door key may derive; every other key is named
    for ln in ('qwen35_4b_s1|qwen35_4b_s2|qwen35_4b_s3) KEY=qwen35_4b; VIEW="${TARGET#qwen35_4b_}";;',
               'qwen35_4b_s1p|qwen35_4b_s2p|qwen35_4b_s3p) KEY=qwen35_4b; VIEW="${TARGET#qwen35_4b_}";;',
               'gemma4e4b_bl|qwen35_4b_bl|nemotron3nano4b_bl|llama8b_bl) KEY="${TARGET%_bl}"; VIEW=main; BASELINES=1;;',
               "llama8b_org) KEY=llama8b; VIEW=main;;", "qwen35_4b_allpos) KEY=qwen35_4b; VIEW=it;;"):
        assert ln in case, ln
    assert "<models.yaml key>_<phase view>" in px                              # the usage names the derived grammar
    live = [ln for ln in px.splitlines() if not ln.lstrip().startswith("#")]          # a comment saying "never --export=" is the rule, not a use
    assert not any("--export=" in ln for ln in live) and "--mem=0" not in px
    assert "CUDA_VISIBLE_DEVICES" not in case and "--export=" not in case               # the derived branch adds none of the hard-rule forms
    # the launch-manifest declaration still precedes f22_init (the wave test's ordering pin) and the pair binding precedes registration
    assert px.index('F22_LAUNCH_MANIFEST="configs/capeval/manifests/misalign_full22_${TARGET}.yaml"') < px.index('f22_init "$WAVE_SHA"') \
        < px.index('CFG="configs/misalign_full22_${TARGET}.yaml"') < px.index("f22_registry_register || true")


def test_misalign_bank_view_derives_a_front_door_target_from_the_generator_and_keeps_every_hand_listed_case():
    """The bank-view script's closed `case "$TARGET"` (the fix-1 owner follow-up 2) asks make_misalign_manifest.py --resolve for anything
    not named explicitly: a front-door phase view resolves to (roster target, view) under LANE=front_door — PSR cells kept, alphasteer
    excluded by policy, beside reseed / paired — and an unknown target still dies by name."""
    bv = (MIS / "misalign_bank_view.sh").read_text()
    assert subprocess.run(["bash", "-n", str(MIS / "misalign_bank_view.sh")]).returncode == 0
    case = bv[bv.index('case "$TARGET" in'):]
    case = case[:case.index("\nesac")]
    assert '*) fatal "unknown target $TARGET";;' not in bv and 'make_misalign_manifest.py" --resolve "$TARGET"' in case and "LANE=front_door" in case
    assert '[ "$KIND" = front_door ] || fatal' in case and "<models.yaml key>_<phase view>" in bv
    assert 'reseed|paired|front_door) log "lane policy $LANE: PSR cells KEPT' in bv and 'reseed|paired|front_door) AS_STATUS=lane_excluded' in bv
    harness = f'set -u; PY={sys.executable}; HERE={MIS}; fatal(){{ echo "FATAL: $*"; exit 2; }}; log(){{ echo "$*"; }}; LANE=""; BASELINES=""; ORG=""; '
    block = case + "\nesac"
    for target, want in (("qwen35_4b_demo_battery", "qwen35_4b_demo battery front_door"), ("qwen35_4b_demo_d0p75_relu", "qwen35_4b_demo d0p75_relu front_door"),
                         ("qwen35_4b_s1", "qwen35_4b s1 reseed"), ("qwen35_4b_s2p", "qwen35_4b s2p paired"), ("qwen35_4b_it", "qwen35_4b it "),
                         ("llama8b_bl", "llama8b main ")):
        r = subprocess.run(["bash", "-c", harness + f"TARGET={target}\n" + block + '\necho "$RTARGET $WVIEW $LANE"'], capture_output=True, text=True,
                           env={**os.environ, "PYTHONPATH": str(REPO / "src")})
        assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == want.strip(), (target, r.stdout, r.stderr)
    r = subprocess.run(["bash", "-c", harness + "TARGET=nope\n" + block], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert r.returncode == 2 and "unknown target nope" in r.stdout and "front-door phase view" in r.stdout, r.stdout + r.stderr
    live = [ln for ln in bv.splitlines() if not ln.lstrip().startswith("#")]
    assert not any("--export=" in ln for ln in live) and "--mem=0" not in bv and "CUDA_VISIBLE_DEVICES" not in case
