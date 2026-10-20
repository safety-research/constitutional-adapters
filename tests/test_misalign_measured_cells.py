"""The misalign LANE POLICY across its three builders:

* make_misalign_manifest.py renders `qwen35_4b_s<s>` from a SYNTHETIC s1 wave pair (the real qwen35_4b Petri manifest /
  wave config with the cells swapped for `none` + the view's ADMITTED objects + the two prompted arms): SERVED cells == none +
  the five (PSR INCLUDED — decisions O20, 2026-09-17: the distilled PSR difference failed admission for every seed and is on no
  view) + the TWO PROMPTED SI arms (the Qwen seed-replicate battery, 2026-09-17: `prompted: True`; a view whose manifest lacks
  them refuses by name), no alphasteer; the config's `misalign_policy` block says `measure_none: false` with the seven MEASURED cells, both validate as
  the standup would; a policy cell outside the view refuses; every pre-existing target's committed files are byte-identical
  (`--check`) and the three live lane pairs render their five + two;
* misalign_cells.py (the proxy's cells.json builder) over a synthetic cells_served.json: under the policy the runner's
  cells.json holds EXACTLY the measured set (no `none`, no `none__lora0`), the default policy still yields today's list, a
  synthetic runner output tree carries no `none` dir, a declared measured set that the served legs do not match refuses;
* misalign_bank_view.sh / misalign_proxy.sbatch admit the tags (RTARGET qwen35_4b, WVIEW s<s>) with the lane's PSR and
  alphasteer exceptions; the header lists each measured cell's staged id + object sha256 from the roster + pins.
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
MIS = REPO / "scripts" / "sdfbase" / "full22" / "misalign"
for p in (REPO / "src",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sisteer.capeval.petri import registry_adapter as ra  # noqa: E402

SIX_H = ["si_constv2_mix_full__psr_ce_all", "si_constv2_mix_full__lora_r4_ce", "si_constv2ctl2_full__psr_ce_all",
         "si_constv2ctl2_full__lora_r4_ce", "si_constv2_mix_full__lora_r4_ce_minus_ctl",
         "si_constv2_mix_full__psr_ce_all_minus_ctl_distilled"]
SIX_TT = [ra.traintest_cell_id(c) for c in SIX_H]
#: decisions O20 (2026-09-17): the distilled PSR difference is NOT ADMITTED for any seed and is on no view — the lane policy
#: serves + measures the FIVE admitted objects; SIX_H stays as the declared grammar (the generator's SEEDREP_SIX_HARNESS)
DISTILLED_H = "si_constv2_mix_full__psr_ce_all_minus_ctl_distilled"
FIVE_H = [c for c in SIX_H if c != DISTILLED_H]
FIVE_TT = [ra.traintest_cell_id(c) for c in FIVE_H]
STAGED = {tt: (("derived_base64s1__" if "minus_ctl" in tt else "base64s1__") + tt) for tt in FIVE_TT}
#: the two prompted SI arms every seed view serves + measures under `prompted: True`
#: (the generator's PROMPTED_CELLS, harness ids; TrainTest prompted__safety_refusal / prompted__si_constitution)
PROMPTED_H = ["prompted__safety", "prompted__constitution"]
PROMPTED_TT = [ra.traintest_cell_id(c) for c in PROMPTED_H]
SEVEN_H, SEVEN_TT = FIVE_H + PROMPTED_H, FIVE_TT + PROMPTED_TT
TARGET = "qwen35_4b_s1"


def _load(name="_mm_lane"):
    spec = importlib.util.spec_from_file_location(name, MIS / "make_misalign_manifest.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_mc():
    spec = importlib.util.spec_from_file_location("_misalign_cells", MIS / "misalign_cells.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _lane_tree(tmp_path, mm, *, extra_manifest_cells=(), drop_none=False, roster=None, pins=None, lane_cells=None, prompted=True):
    """A REPO-shaped tmp tree holding only the s1 wave pair, derived from the real qwen35_4b pair with the cells swapped
    (`lane_cells`: the view's staged objects in place of the five admitted — `[]` = every object still pending_fit, as they land;
    `prompted`: the view lists its two prompted SI arms after the objects, as every seed view's roster does — False =
    the earlier shape, the known-bad a `prompted: True` policy refuses)."""
    six = FIVE_H if lane_cells is None else list(lane_cells)
    extra_manifest_cells = list(PROMPTED_H if prompted else []) + [c for c in extra_manifest_cells if not (prompted and c in PROMPTED_H)]
    root = tmp_path / "repo"
    (root / "configs" / "capeval" / "manifests").mkdir(parents=True)
    src_m = (REPO / "configs/capeval/manifests/petri_full22_qwen35_4b.yaml").read_text()
    before, _cells, after = mm._split_cells_block(src_m)
    cells = ([] if drop_none else ["none"]) + six + list(extra_manifest_cells)
    (root / "configs/capeval/manifests/petri_full22_qwen35_4b_s1.yaml").write_text(
        before + "\n  cells:\n" + "\n".join(f"    - {c}" for c in cells) + "\n" + after)
    src_c = (REPO / "configs/sdfbase_full22_qwen35_4b.yaml").read_text()
    marker = "# -- the explicit cell enumeration (ordered; the registry is EXACTLY this) ---\nexplicit_cells:\n"
    head, tail = src_c.split(marker)
    lines = tail.split("\n")
    k = 0
    while lines[k].startswith("- "):
        k += 1
    tt_cells = ([] if drop_none else ["none"]) + [ra.traintest_cell_id(c) for c in six] + [ra.traintest_cell_id(c) for c in extra_manifest_cells]
    new_tail = "\n".join([f"- {c}" for c in tt_cells] + lines[k:])
    new_tail = re.sub(r"^defenses_dir: .*$", "defenses_dir: out/full22/qwen35_4b/cells_view/s1", new_tail, flags=re.M)
    (root / "configs/sdfbase_full22_qwen35_4b_s1.yaml").write_text(head + marker + new_tail)
    if roster is not None:
        (root / "configs/sdfbase_full22_roster.yaml").write_text(yaml.safe_dump(roster))
    if pins is not None:
        (root / "configs/sdfbase_full22_pins.json").write_text(json.dumps(pins))
    for attr, rel in (("REPO", ""), ("MANIFEST_DIR", "configs/capeval/manifests"), ("ROSTER", "configs/sdfbase_full22_roster.yaml"),
                      ("PINS", "configs/sdfbase_full22_pins.json"), ("ALPHASTEER_PINS", "configs/alphasteer_f22_pins.json"),
                      ("ALPHASTEER_ROSTER", "configs/alphasteer_f22_roster.yaml"), ("BASELINE_PINS", "configs/misalign_baseline_pins.json")):
        setattr(mm, attr, root / rel if rel else root)
    return root


# --------------------------------------------------------------------------- the generator

def test_the_lane_renders_none_plus_the_five_admitted_with_psr_plus_the_two_prompted_arms_and_no_alphasteer(tmp_path):
    mm = _load()
    assert mm.SEEDREP_ADMITTED_HARNESS == tuple(FIVE_H) and mm.SEEDREP_DROPPED_HARNESS == (DISTILLED_H,)
    assert all(mm.LANE_POLICY[f"qwen35_4b_{v}"]["cells"] == tuple(FIVE_H) for v in ("s1", "s2", "s3"))
    assert all(mm.LANE_POLICY[f"qwen35_4b_{v}"]["prompted"] is True and mm.LANE_POLICY[f"qwen35_4b_{v}"]["alphasteer"] is False for v in ("s1", "s2", "s3"))
    assert list(mm.PROMPTED_CELLS) == PROMPTED_H
    root = _lane_tree(tmp_path, mm)
    m_text, cells = mm.render_manifest(TARGET)
    assert cells == ["none", *FIVE_H, *PROMPTED_H] and DISTILLED_H not in cells       # objects first, the two prompted arms LAST
    assert sum("psr" in c for c in cells) == 2
    assert [c for c in cells if c.startswith("prompted__")] == PROMPTED_H and "alphasteer" not in cells
    assert f"  audit: misalign_full22_{TARGET}" in m_text and "LANE POLICY" in m_text and "SERVED != MEASURED" in m_text
    assert "verified per request (H8) for the prompted cell(s) prompted__safety, prompted__constitution" in m_text
    assert "NO `prompted__*` arm" not in m_text and "two prompted SI arms" in m_text
    c_text, tt = mm.render_config(TARGET, cells)
    assert tt == ["none", *SEVEN_TT]
    cfg = yaml.safe_load(c_text)
    assert cfg["explicit_cells"] == ["none", *SEVEN_TT]
    assert cfg["misalign_policy"] == {"lane": "reseed", "view": "s1", "measure_none": False, "measured_cells": SEVEN_TT,
                                      "psr": "include", "prompted": True, "alphasteer": False,
                                      "pairing_anchor": "exp/misalign/qwen35_4b/agentic/none"}
    assert "no prompted arm" not in c_text and "the two prompted arms" in c_text
    assert cfg["defenses_dir"] == f"out/misalign/{TARGET}/cells_view" and cfg["private_run"] == f"misalign_full22_{TARGET}"
    (tmp_path / "v").mkdir(exist_ok=True); mm.validate(TARGET, m_text, c_text, tmp_path / "v")
    outs = dict(mm.outputs())
    assert set(p.name for p in outs) == {f"misalign_full22_{TARGET}.yaml"} and len(outs) == 2
    assert outs[root / "configs/capeval/manifests" / f"misalign_full22_{TARGET}.yaml"] == m_text
    # the source stem is the view's own wave pair, and the view is what wave_of says
    assert mm.source_stem(TARGET) == "qwen35_4b_s1" and mm.wave_of(TARGET) == ("qwen35_4b", "s1")
    assert mm.source_stem("qwen35_4b_it") == "qwen35_4b_it" and mm.source_stem("llama8b_org") == "llama8b"
    assert mm.alphasteer_status(TARGET) == (mm.ALPHASTEER_LANE_EXCLUDED, {})


def test_a_view_whose_manifest_lacks_a_prompted_arm_refuses_by_name_under_prompted_true(tmp_path, monkeypatch):
    """The known-bad: the policy admits the two prompted arms, so a view whose Petri manifest lists neither (an earlier
    roster) or only one of them refuses BY NAME — never a silent one-arm or no-arm panel; and under a `prompted: False` policy
    (every earlier lane target) a prompted cell on the view is still DROPPED by policy, not refused as a lost wave cell."""
    mm = _load()
    _lane_tree(tmp_path, mm, prompted=False)
    with pytest.raises(SystemExit, match=r"admits the prompted arms \['prompted__safety', 'prompted__constitution'\] but the view's Petri manifest does not list \['prompted__safety', 'prompted__constitution'\]"):
        mm.render_manifest(TARGET)
    _lane_tree(tmp_path / "one", mm, prompted=False, extra_manifest_cells=["prompted__constitution"])
    with pytest.raises(SystemExit, match=r"does not list \['prompted__safety'\]"):
        mm.render_manifest(TARGET)
    # a `prompted: False` policy over a view that lists a prompted cell: dropped by policy (the earlier behaviour, byte for byte)
    _lane_tree(tmp_path / "off", mm, prompted=False, extra_manifest_cells=["prompted__constitution"])
    monkeypatch.setitem(mm.LANE_POLICY, TARGET, {**mm.LANE_POLICY[TARGET], "prompted": False})
    m_text, cells = mm.render_manifest(TARGET)
    assert cells == ["none", *FIVE_H]
    c_text, tt = mm.render_config(TARGET, cells)
    assert "prompted__si_constitution" not in yaml.safe_load(c_text)["explicit_cells"]
    assert yaml.safe_load(c_text)["misalign_policy"]["prompted"] is False and "no prompted arm" in c_text
    (tmp_path / "v").mkdir(exist_ok=True); mm.validate(TARGET, m_text, c_text, tmp_path / "v")
    # an alphasteer-admitting policy still refuses (the generator renders objects + prompted arms only)
    monkeypatch.setitem(mm.LANE_POLICY, TARGET, {**mm.LANE_POLICY[TARGET], "prompted": True, "alphasteer": True})
    _lane_tree(tmp_path / "as", mm)
    with pytest.raises(SystemExit, match="admits an alphasteer arm"):
        mm.render_manifest(TARGET)


def test_a_policy_cell_outside_the_view_refuses(tmp_path, monkeypatch):
    mm = _load()
    _lane_tree(tmp_path, mm)
    monkeypatch.setitem(mm.LANE_POLICY, TARGET, {**mm.LANE_POLICY[TARGET], "cells": (*FIVE_H, "si_safety__lora_r4")})
    with pytest.raises(SystemExit, match="does not list \\['si_safety__lora_r4'\\]"):
        mm.render_manifest(TARGET)
    # the O20 shape exactly: a policy that still named the DROPPED distilled cell against a view that (like every live view)
    # no longer lists it refuses BY NAME — the drop could never have rendered as a silent five
    monkeypatch.setitem(mm.LANE_POLICY, TARGET, {**mm.LANE_POLICY[TARGET], "cells": tuple(SIX_H)})
    with pytest.raises(SystemExit, match="does not list \\['si_constv2_mix_full__psr_ce_all_minus_ctl_distilled'\\]"):
        mm.render_manifest(TARGET)


def test_a_view_without_none_refuses(tmp_path):
    mm = _load()
    _lane_tree(tmp_path, mm, drop_none=True)
    with pytest.raises(SystemExit, match="no `none` cell"):
        mm.render_manifest(TARGET)


def test_every_pre_existing_target_is_byte_identical_and_the_lane_targets_render_their_five_admitted_cells_plus_two_prompted():
    # On this tree the three seed-replicate views are STAGED (2026-09-17): the four parents + the LoRA difference admitted and
    # served, the distilled PSR difference NOT admitted for any seed and dropped from every view, so the lane
    # targets RENDER (no skip, no refusal) beside every pre-existing target, byte-identical. (Before staging the lane targets were
    # skipped as PENDING — "its wave's Petri manifest lists none of the lane policy's cells yet"; earlier still, "its wave's configs
    # are not on this tree".)
    r = subprocess.run([sys.executable, str(MIS / "make_misalign_manifest.py"), "--check"], capture_output=True, text=True,
                       cwd=REPO, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin", "HOME": str(REPO)})
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "DRIFT" not in r.stdout and "skip qwen35_4b_s" not in r.stdout, r.stdout[-3000:]
    for s in ("s1", "s2", "s3"):
        assert f"ok: configs/capeval/manifests/misalign_full22_qwen35_4b_{s}.yaml" in r.stdout, r.stdout[-3000:]
        assert f"ok: configs/misalign_full22_qwen35_4b_{s}.yaml" in r.stdout, r.stdout[-3000:]
        assert f"# qwen35_4b_{s} (roster qwen35_4b/{s}): alphasteer lane_excluded -> no alphasteer cell (LANE POLICY reseed: 5 admitted objects + 2 prompted arms served + measured" in r.stdout
        cfg = yaml.safe_load((REPO / f"configs/misalign_full22_qwen35_4b_{s}.yaml").read_text(encoding="utf-8"))
        assert cfg["explicit_cells"] == ["none", *SEVEN_TT] and cfg["misalign_policy"]["measured_cells"] == SEVEN_TT
        assert cfg["misalign_policy"]["measure_none"] is False and cfg["misalign_policy"]["prompted"] is True
        assert ra.traintest_cell_id(DISTILLED_H) not in cfg["explicit_cells"]
        text = (REPO / f"configs/misalign_full22_qwen35_4b_{s}.yaml").read_text(encoding="utf-8")
        n = s[1:]
        pins = re.findall(rf"^#\s+(\S+)\s+<- ((?:derived_)?base64s{n}__\S+) \(object_sha256 [0-9a-f]{{64}}\)$", text, re.M)
        assert [tt for tt, _sid in pins] == FIVE_TT and all(sid == (("derived_base64s" if "minus_ctl" in tt else "base64s") + f"{n}__{tt}") for tt, sid in pins), pins
        prompted_lines = re.findall(r"^#\s+(prompted__\S+)\s+<- \(prompted baseline: the SI as the system prompt; no object, no pin", text, re.M)
        assert prompted_lines == PROMPTED_TT, prompted_lines
        assert "pending" not in "\n".join(ln for ln in text.splitlines() if "<-" in ln)
        _b, man_cells, _a = _load("_mm_split")._split_cells_block(
            (REPO / f"configs/capeval/manifests/misalign_full22_qwen35_4b_{s}.yaml").read_text(encoding="utf-8"))
        assert man_cells == ["none", *SEVEN_H], (s, man_cells)
    # + the three PAIRED views: lane `paired`, three admitted cells, no prompted arms
    assert sorted(mm_t for mm_t in _load("_mm_check").TARGETS if mm_t.startswith("qwen35_4b_s")) == [
        "qwen35_4b_s1", "qwen35_4b_s1p", "qwen35_4b_s2", "qwen35_4b_s2p", "qwen35_4b_s3", "qwen35_4b_s3p"]
    mm = _load("_mm_paired")
    for s in ("s1p", "s2p", "s3p"):
        pol = mm.LANE_POLICY[f"qwen35_4b_{s}"]
        assert pol["lane"] == "paired" and pol["prompted"] is False and pol["measure_none"] is False
        assert pol["cells"] == mm.PAIRED_HARNESS == ("si_constv2_mix_full__psr_ce_all_minus_ctl", "si_constv2_mix_full__lora_r4_ce",
                                                    "si_constv2_mix_full__lora_r4_ce_minus_ctl")
        assert pol["pairing_anchor"] == "exp/misalign/qwen35_4b/agentic/none" and mm.WAVE_OF[f"qwen35_4b_{s}"] == ("qwen35_4b", s)


def test_a_view_with_every_policy_cell_pending_is_skipped_and_a_partial_view_still_refuses(tmp_path, capsys):
    """The merge of the pending-view case (views landed wholly `pending_fit`, Petri cells == [none]) with the lane policy: a view
    listing NONE of the policy's cells is a pending state — outputs() renders nothing for it and says why — while a view listing SOME
    of them is not pending and refuses on the missing ones (never a guess). Absent wave sources still read as before."""
    mm = _load("_mm_pending")
    _lane_tree(tmp_path, mm, lane_cells=[], prompted=False)
    note = mm.lane_cells_pending(TARGET)
    assert note and "lists none of the lane policy's cells yet" in note and "['none']" in note
    assert mm.outputs() == []
    out = capsys.readouterr().out
    assert f"skip {TARGET}: its wave's Petri manifest lists none of the lane policy's cells yet" in out
    assert "skip qwen35_4b_s2: its wave's configs are not on this tree" in out
    assert TARGET not in mm.present_targets()
    # a partial view: three of the five staged -> not pending, and the render refuses the two missing (before it ever asks about
    # the prompted arms: the objects are checked first)
    _lane_tree(tmp_path / "b", mm, lane_cells=FIVE_H[:3], prompted=False)
    assert mm.lane_cells_pending(TARGET) is None
    with pytest.raises(SystemExit, match="LANE_POLICY names cells the view's Petri manifest does not list"):
        mm.outputs()
    # the full view renders (the pending note is None, the target is present)
    _lane_tree(tmp_path / "c", mm)
    assert mm.lane_cells_pending(TARGET) is None and TARGET in mm.present_targets()
    assert [p.name for p, _ in mm.outputs()] == [f"misalign_full22_{TARGET}.yaml"] * 2
    # a non-lane target is never pending
    assert mm.lane_cells_pending("qwen35_4b") is None


def test_the_header_lists_each_measured_cells_staged_id_and_pin_sha_or_says_pending(tmp_path):
    mm = _load()
    roster = {"targets": {"qwen35_4b": {"views": {"s1": {"cells": STAGED}}}}}
    pins = {"qwen35_4b": {sid: {"object_sha256": (f"{i:02x}" * 32)} for i, sid in enumerate(STAGED.values(), start=1)}}
    _lane_tree(tmp_path, mm, roster=roster, pins=pins)
    c_text, _ = mm.render_config(TARGET, ["none", *FIVE_H])
    for tt, sid in STAGED.items():
        sha = pins["qwen35_4b"][sid]["object_sha256"]
        assert re.search(rf"^#\s+{re.escape(tt)}\s+<- {re.escape(sid)} \(object_sha256 {sha}\)$", c_text, re.M), tt
    # no pins yet: pending, named; no roster view yet: pending, named
    _lane_tree(tmp_path / "b", mm, roster=roster)
    c_text, _ = mm.render_config(TARGET, ["none", *FIVE_H])
    assert c_text.count("(pin pending: not in sdfbase_full22_pins.json yet — W5)") == 5
    _lane_tree(tmp_path / "c", mm)
    c_text, _ = mm.render_config(TARGET, ["none", *FIVE_H])
    assert c_text.count("(staging pending: the roster's qwen35_4b/s1 view does not list it yet — W5)") == 5
    # a prompted arm is never "pending": it has no staged id and no pin BY DESIGN (the roster's `prompted__<si>: null`)
    c_text, _ = mm.render_config(TARGET, ["none", *FIVE_H, *PROMPTED_H])
    assert c_text.count("(staging pending: the roster's qwen35_4b/s1 view does not list it yet — W5)") == 5
    assert c_text.count("(prompted baseline: the SI as the system prompt; no object, no pin — H8-verified per request)") == 2
    assert "pending" not in "\n".join(ln for ln in c_text.splitlines() if "prompted__" in ln and "<-" in ln)
    # a malformed pin refuses
    bad = {"qwen35_4b": {sid: {"object_sha256": "nope"} for sid in STAGED.values()}}
    _lane_tree(tmp_path / "d", mm, roster=roster, pins=bad)
    with pytest.raises(SystemExit, match="is not 64 hex"):
        mm.render_config(TARGET, ["none", *FIVE_H])


# --------------------------------------------------------------------------- the proxy's cells.json builder

def _leg(cell_id, cell_class, kind, i):
    return {"cell_id": cell_id, "cell_class": cell_class, "kind": kind, "virtual_model_name": f"petri__{cell_id}",
            "serving_identity": f"sid-{i}", "intervention_hash": f"ih-{i}", "si_id": None, "petri_run_key": f"rk-{i}"}


def _served(extra_canonical=()):
    canonical = [_leg("none", "base", "none", 0)] + [_leg(c, "class_a", "psr" if "psr" in c else "lora", i + 1) for i, c in enumerate(FIVE_H)] \
        + list(extra_canonical)
    controls = [_leg("none__lora0", "control", "lora", 9)]
    cs = {"canonical": canonical, "controls": controls, "manifest": "m.yaml", "config": "c.yaml", "defenses_dir": "d",
          "reduction_mode": "not_applicable", "chat_template": {"override": None}, "server_default_policy": "engine_defaults"}
    ep = {"served_models": [c["virtual_model_name"] for c in canonical + controls], "attested_max_num_seqs": 44,
          "attested_max_model_len": 16384, "max_num_seqs": 44, "max_model_len": 16384, "node": "TESTHOST",
          "server_default_policy": "engine_defaults", "override_generation_config": None, "model": "Qwen/Qwen3.5-4B"}
    return cs, ep


def test_under_the_policy_the_runner_measures_the_five_only_and_by_default_everything(tmp_path):
    mm, mc = _load(), _load_mc()
    _lane_tree(tmp_path, mm)
    c_text, _ = mm.render_config(TARGET, ["none", *FIVE_H])
    cfg_path = tmp_path / "misalign_full22_qwen35_4b_s1.yaml"
    cfg_path.write_text(c_text)
    policy = mc.read_policy(cfg_path)
    assert policy["measure_none"] is False and policy["measured_cells"] == FIVE_TT and policy["lane"] == "reseed"
    # the LIVE lane pair's measured set is the five + the two prompted arms — a prompted leg is MEASURED under measure_none
    # false (only the `base` / `control` classes are left unmeasured), and the runner's set-equality check holds on the seven
    c7, _ = mm.render_config(TARGET, ["none", *FIVE_H, *PROMPTED_H])
    (tmp_path / "seven.yaml").write_text(c7)
    pol7 = mc.read_policy(tmp_path / "seven.yaml")
    assert pol7["measured_cells"] == SEVEN_TT
    cs7, ep7 = _served(extra_canonical=[_leg(c, "prompted", "prompted", 7 + i) for i, c in enumerate(PROMPTED_H)])
    cells7, full7 = mc.build_cells(cs7, ep7, target=TARGET, wave_sha="a" * 40, model_key="qwen35_4b", wave_view="s1", policy=pol7,
                                   served_cluster="cluster_a", slurm_job_id="1")
    assert [r["cell_id"] for r in cells7["cells"]] == SEVEN_H and cells7["serving"]["measured_policy"]["unmeasured_served"] == ["none", "none__lora0"]
    assert [l["measured"] for l in full7["legs"]] == [False] + [True] * 7 + [False]
    with pytest.raises(SystemExit, match="silent subset or superset"):      # the five-only policy against the seven served legs
        mc.build_cells(cs7, ep7, target=TARGET, wave_sha="a" * 40, model_key="qwen35_4b", wave_view="s1", policy=policy)
    cs, ep = _served()
    cells, full = mc.build_cells(cs, ep, target=TARGET, wave_sha="a" * 40, model_key="qwen35_4b", wave_view="s1", policy=policy,
                                 served_cluster="cluster_a", slurm_job_id="1")
    assert [r["cell_id"] for r in cells["cells"]] == FIVE_H
    assert cells["serving"]["measured_policy"]["unmeasured_served"] == ["none", "none__lora0"]
    assert cells["serving"]["measured_policy"]["measure_none"] is False
    assert cells["serving"]["server_default_policy"] == "engine_defaults"
    assert [l["cell_id"] for l in full["legs"]] == ["none", *FIVE_H, "none__lora0"]
    assert [l["measured"] for l in full["legs"]] == [False] + [True] * 5 + [False]
    # a synthetic runner output tree, one dir per measured leg: no `none` dir, no controls dir
    out = tmp_path / "agentic"
    for r in cells["cells"]:
        (out / r["cell_id"]).mkdir(parents=True)
        (out / r["cell_id"] / "responses.jsonl").write_text('{"cell_id": "%s"}\n' % r["cell_id"])
    assert sorted(p.name for p in out.iterdir()) == sorted(FIVE_H)
    assert not (out / "none").exists() and not (out / "none__lora0").exists()
    # the DEFAULT policy (every other panel): today's list, `none` and the controls twin measured
    cells_d, full_d = mc.build_cells(cs, ep, target="qwen35_4b", wave_sha="a" * 40, model_key="qwen35_4b", wave_view="main",
                                     policy=mc.read_policy(None))
    assert [r["cell_id"] for r in cells_d["cells"]] == ["none", *FIVE_H, "none__lora0"]
    assert cells_d["serving"]["measured_policy"]["unmeasured_served"] == []
    # the runner's contract fields are intact on every row
    for r in cells["cells"] + cells_d["cells"]:
        assert set(r) == {"cell_id", "virtual_model", "serving_identity", "intervention_hash", "si_id", "kind"}


def test_a_declared_measured_set_the_served_legs_do_not_match_refuses(tmp_path):
    mc = _load_mc()
    cs, ep = _served()
    pol = {"measure_none": False, "measured_cells": FIVE_TT[:-1], "lane": "reseed", "source": "test"}
    with pytest.raises(SystemExit, match="silent subset or superset"):
        mc.build_cells(cs, ep, target=TARGET, wave_sha="a", model_key="qwen35_4b", wave_view="s1", policy=pol)
    # a served leg missing from the proxy's served_models refuses (as before)
    ep2 = {**ep, "served_models": ep["served_models"][:-1]}
    with pytest.raises(SystemExit, match="is not served"):
        mc.build_cells(cs, ep2, target=TARGET, wave_sha="a", model_key="qwen35_4b", wave_view="s1", policy=pol)
    # a config without the block reads the default; a malformed block refuses
    p = tmp_path / "cfg.yaml"
    p.write_text("explicit_cells: [none]\n")
    assert mc.read_policy(p)["measure_none"] is True
    p.write_text("misalign_policy: {measure_none: maybe}\n")
    with pytest.raises(SystemExit, match="boolean measure_none"):
        mc.read_policy(p)


def test_the_builder_cli_writes_both_records(tmp_path):
    cs, ep = _served()
    (tmp_path / "cs.json").write_text(json.dumps(cs)); (tmp_path / "ep.json").write_text(json.dumps(ep))
    (tmp_path / "cfg.yaml").write_text(yaml.safe_dump({"explicit_cells": ["none", *FIVE_TT],
                                                        "misalign_policy": {"measure_none": False, "measured_cells": FIVE_TT, "lane": "reseed"}}))
    r = subprocess.run([sys.executable, str(MIS / "misalign_cells.py"), "--cells-served", str(tmp_path / "cs.json"), "--endpoint", str(tmp_path / "ep.json"),
                        "--out", str(tmp_path / "cells.json"), "--full", str(tmp_path / "cells_full.json"), "--target", TARGET, "--wave-sha", "a" * 40,
                        "--model-key", "qwen35_4b", "--wave-view", "s1", "--config", str(tmp_path / "cfg.yaml")],
                       capture_output=True, text=True, env={"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert "SERVED, NOT MEASURED" in r.stdout and "['none', 'none__lora0']" in r.stdout
    cells = json.loads((tmp_path / "cells.json").read_text())
    assert [c["cell_id"] for c in cells["cells"]] == FIVE_H
    from sisteer.capeval.misalign.runner import load_cells_file
    rows, extra = load_cells_file(tmp_path / "cells.json")   # the runner's own reader accepts the record
    assert len(rows) == 5 and "serving" in extra



def _no_export_flag(text: str) -> bool:
    """No `--export=` in a live line (a comment saying "never --export=" is the rule, not a use)."""
    return not any("--export=" in ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))

# --------------------------------------------------------------------------- the shell admits the tags

def test_bank_view_and_proxy_admit_the_tags_with_the_lanes_exceptions():
    bv = (MIS / "misalign_bank_view.sh").read_text()
    px = (MIS / "misalign_proxy.sbatch").read_text()
    for f in ("misalign_bank_view.sh", "misalign_proxy.sbatch"):
        assert subprocess.run(["bash", "-n", str(MIS / f)]).returncode == 0, f
    assert 'qwen35_4b_s1|qwen35_4b_s2|qwen35_4b_s3) RTARGET=qwen35_4b; WVIEW="${TARGET#qwen35_4b_}"; LANE=reseed;;' in bv
    # the lane policy branch is a `case` since the paired lane joined (2026-09-20): reseed AND paired keep the PSR cells (the composed
    # difference_relu arm IS a psr_paired_write) and exclude alphasteer; every other lane drops 'psr' cells as before
    # ... and the front door's phase views: LANE=front_door keeps the PSR cells and
    # excludes alphasteer by policy, beside reseed / paired; the default arm asks make_misalign_manifest.py --resolve instead of refusing
    assert 'case "$LANE" in\n  reseed|paired|front_door) log "lane policy $LANE: PSR cells KEPT' in bv and 'if [ "$LANE" != reseed ]; then' not in bv
    assert 'case "$LANE" in reseed|paired|front_door) AS_STATUS=lane_excluded; AS_SHA=-; AS_SIDECAR=-;; esac' in bv and "|lane_excluded) ;;" in bv
    assert '*) fatal "unknown target $TARGET";;' not in bv and 'make_misalign_manifest.py" --resolve "$TARGET"' in bv and "LANE=front_door" in bv
    assert 'qwen35_4b_s1p|qwen35_4b_s2p|qwen35_4b_s3p) RTARGET=qwen35_4b; WVIEW="${TARGET#qwen35_4b_}"; LANE=paired;;' in bv
    assert 'qwen35_4b_s1p|qwen35_4b_s2p|qwen35_4b_s3p) KEY=qwen35_4b; VIEW="${TARGET#qwen35_4b_}";;' in px
    assert 'qwen35_4b_s1|qwen35_4b_s2|qwen35_4b_s3) KEY=qwen35_4b; VIEW="${TARGET#qwen35_4b_}";;' in px
    # the proxy builds cells.json through the module, from the config's policy, and says what it measures
    assert 'scripts/sdfbase/full22/misalign/misalign_cells.py --cells-served "$OUT/cells_served.json" --endpoint "$EP_SNAP"' in px
    assert '--config "$CFG" || exit 1' in px and 'MEASURE_NONE=$(' in px
    assert "<<'PY' || exit 1" not in px.split("cells.json: the runner's contract")[1].split("cp \"$CELLS_FILE\"")[0]
    # the kind-aware launch-manifest declaration still precedes f22_init (the wave test's ordering pin)
    assert px.index('F22_LAUNCH_MANIFEST="configs/capeval/manifests/misalign_full22_${TARGET}.yaml"') < px.index('f22_init "$WAVE_SHA"')
    assert _no_export_flag(px) and "--mem=0" not in px
