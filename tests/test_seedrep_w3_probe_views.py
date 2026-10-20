"""Launch F-B: the S5 multi-turn window probe's two cells are pinned under LANE-OWNED views, and their configs exist on the tree.

Before this change `s5_probe_rows` was empty (`s5_multiturn_probe.status` UNREGISTERED): nothing pinned the LoRA-difference children 86fa8e9c /
0522a255 under any view, and the probe rows' argv would have named `configs/sdfbase_full22_qwen35_4b_<view>.yaml` + a Petri manifest that
were not on the tree. The lane's own peer-shaped roster (`lp.PROBE_ROSTER_REL`, STRING cells `{parent: <dose>_eb64s1__<parent>}`) + pins
(`lp.PROBE_PINS_REL`, `object_sha256` = the bank shas) now register them under `d1p5_lora` / `dm0p5_lora`; the two configs and two Petri
manifests are `make_full22_configs.py --roster <that roster>` output (held byte-identical here through the generator's own `--check`);
and the plan's `main()` resolves to the pair when no `--peer-roster / --peer-pins` is given. Fails before the fix (no files, no rows).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts" / "modal") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "modal"))
import seedrep_launch_plan as lp  # noqa: E402

W = "2" * 40
WAVE = "22222222-3333-4444-8444-555555555555"
LORA = lp.LORA_DIFF_CELL
EXPECTED = {"d1p5_lora": f"d1p5_eb64s1__{LORA}", "dm0p5_lora": f"dm0p5_eb64s1__{LORA}"}


@pytest.fixture(scope="module")
def inputs():
    return lp.load_inputs(REPO, peer_roster=REPO / lp.PROBE_ROSTER_REL, peer_pins=REPO / lp.PROBE_PINS_REL)


@pytest.fixture(scope="module")
def plan(inputs):
    return lp.build_plan(W=W, wave_uuid=WAVE, reqs_sha256=inputs["reqs_sha256"], roster=inputs["roster"], pins=inputs["pins"], arditi_pins=inputs["arditi_pins"],
                         arms=inputs["arms"], materialisation="draft", generated_at="2026-09-18T00:00:00Z", peer_roster=inputs["peer_roster"],
                         peer_pins=inputs["peer_pins"], config_private_runs=inputs["config_private_runs"])


def test_the_lane_roster_and_pins_are_on_the_tree_in_the_peer_shape(inputs):
    """`_peer`'s shape (tests/test_seedrep_review_fixes.py TestDoseStandups): string cells keyed by the PARENT cell, pins keyed by staged id."""
    ro, pins = inputs["peer_roster"], inputs["peer_pins"]
    views = lp.peer_dose_views(ro)
    assert set(views) == set(EXPECTED) == set(lp.PROBE_VIEWS)
    for view, sid in EXPECTED.items():
        b = views[view]
        assert b["cells"] == {LORA: sid} and isinstance(b["cells"][LORA], str)
        assert b["config"] == lp.VIEW_CONFIG_REL.format(view=view) and b["manifest"] == lp.VIEW_MANIFEST_REL.format(view=view)
        assert (REPO / b["config"]).is_file() and (REPO / b["manifest"]).is_file()
        assert lp.dose_staged_id_problem(sid) is None and lp.DOSE_S1_STAGED_RE.match(sid).group("parent") == LORA
        assert pins[lp.MODEL_KEY][sid]["object_sha256"] == lp.DOSE_CHILD_SHA[sid] == {c["staged_id"]: c["object_sha256"] for c in lp.S5_PROBE_CELLS}[sid]
        assert len(pins[lp.MODEL_KEY][sid]["object_sha256"]) == 64
        assert b["dose"] == float(sid.split("_eb64s1__")[0].replace("dm", "-").replace("d", "").replace("p", ".")) if sid.startswith("dm") else True
    # the shared surfaces are untouched: no probe staged id in the lane pins, no probe view in the production roster
    shared_pins = json.loads((REPO / lp.PINS_REL).read_text())
    assert not set(EXPECTED.values()) & set(shared_pins[lp.MODEL_KEY])
    assert not set(EXPECTED) & set(inputs["roster"]["targets"][lp.MODEL_KEY]["views"])


def test_the_draft_plan_registers_both_probe_cells_on_both_probe_legs(plan):
    """Two views x the two S5 probe legs -> four smoke-only rows; each cell registered on crescendo AND actorattack under its bank sha."""
    rows = plan["s5_probe_rows"]
    assert len(rows) == 2 * len(lp.S5_PROBE_LEGS) == 4
    assert {(r["view"], r["leg"]) for r in rows} == {(v, leg) for v in EXPECTED for leg in lp.S5_PROBE_LEGS}
    staged = {o["staged_id"] for r in rows for o in r["objects"].values()}
    assert staged == set(EXPECTED.values())
    for r in rows:
        assert r["kind"] == lp.S5_PROBE_KIND and r["s5_probe"] and r["smoke_only"] and r["modal"]["max_containers"] == 1 and r["seed_bases"] == [1]
        assert r["route_set"] == ["none", LORA] and r["bank_view"] if "bank_view" in r else True
        a = r["argv"]
        cfg = a[a.index("--config") + 1]
        assert cfg == lp.VIEW_CONFIG_REL.format(view=r["view"]) and (REPO / cfg).is_file()
        assert a[a.index("--defenses-dir") + 1].endswith(f"/cells_view/{r['view']}")
        o = r["objects"][LORA]
        assert o["staged_id"] == EXPECTED[r["view"]] and o["object_sha256"] == lp.DOSE_CHILD_SHA[o["staged_id"]] and o["s5_probe_cell"] and o["dose_view"] == r["view"]
        assert [k["arm"] for k in r["children"]] == [f"{r['view']}__none", EXPECTED[r["view"]]] and all(k["dose_view"] == r["view"] for k in r["children"])
    probe = plan["s5_multiturn_probe"]
    assert probe["status"].startswith("registered")
    for sid, reg in probe["registration"].items():
        assert reg["complete"] and set(reg["registered_on"]) == set(lp.S5_PROBE_LEGS) and reg["sha_mismatch"] == {}
    assert lp.s5_probe_registration_problems(plan) == [] and lp.validate_plan(plan) == []
    # no dose standups, the ledger unchanged, the probe rows outside standups / rungs
    assert plan["dose_standups"]["status"] == lp.DOSE_STATUS_NOT_IN_WAVE and plan["dose_standups"]["peer_files_given"] is True
    assert plan["capacity_ledger"]["unstaged_peak"] == {"containers": 134, "gpus": 134} and len(plan["standups"]) == 74     # One H200 per pool
    assert lp.bank_views_for_plan(plan) == {v: {LORA: sid} for v, sid in EXPECTED.items()}


def test_the_rendered_configs_are_the_generators_output():
    """`make_full22_configs.py --check --roster <lane roster>` holds the two configs + two manifests byte-identical (no hand edit), and the
    production roster's own --check is unaffected by the new files."""
    env = {**os.environ, "PYTHONPATH": f"{REPO}/src"}
    r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/make_full22_configs.py"), "--check", "--roster", str(REPO / lp.PROBE_ROSTER_REL)],
                       cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    oks = sorted(ln.split("ok: ", 1)[1] for ln in r.stdout.splitlines() if ln.startswith("ok: "))
    assert oks == sorted(str(REPO / rel) for v in EXPECTED for rel in (lp.VIEW_CONFIG_REL.format(view=v), lp.VIEW_MANIFEST_REL.format(view=v)))
    for v, sid in EXPECTED.items():
        cfg = (REPO / lp.VIEW_CONFIG_REL.format(view=v)).read_text()
        assert f"cells_view/{v}" in cfg and sid in cfg and lp.PROBE_ROSTER_REL in cfg
        man = yaml.safe_load((REPO / lp.VIEW_MANIFEST_REL.format(view=v)).read_text())
        assert man["target_decode"]["mode"] == "vendor" if "target_decode" in man else True


def test_main_defaults_to_the_lane_probe_files_when_no_peer_pair_is_given(tmp_path):
    """`20_launch_plan.sh` passes --peer-* only for Strength's default names; without them the generator now resolves to the lane's pair."""
    out, ledger, wave = tmp_path / "plan.json", tmp_path / "ledger.json", tmp_path / "wave.json"
    rc = lp.main(["--repo", str(REPO), "--w", W, "--wave-json", str(wave), "--mint-wave", "--materialisation", "draft", "--out", str(out), "--ledger", str(ledger)])
    assert rc == 0
    plan = json.loads(out.read_text())
    assert len(plan["s5_probe_rows"]) == 4 and plan["s5_multiturn_probe"]["status"].startswith("registered")
    assert {o["staged_id"] for r in plan["s5_probe_rows"] for o in r["objects"].values()} == set(EXPECTED.values())
    # an explicit pair still wins: pointing at a roster with NO probe view leaves the probe UNREGISTERED (nothing is silently merged)
    ro = yaml.safe_load((REPO / lp.PROBE_ROSTER_REL).read_text())
    ro["targets"][lp.MODEL_KEY]["views"] = {"d0p5_lora": {**ro["targets"][lp.MODEL_KEY]["views"]["d1p5_lora"], "dose": 0.5, "cells": {LORA: f"d0p5_eb64s1__{LORA}"}}}
    pins = json.loads((REPO / lp.PROBE_PINS_REL).read_text())
    pins[lp.MODEL_KEY] = {f"d0p5_eb64s1__{LORA}": {"object_sha256": lp.DOSE_CHILD_SHA[f"d0p5_eb64s1__{LORA}"]}}
    (tmp_path / "ro.yaml").write_text(yaml.safe_dump(ro)); (tmp_path / "pins.json").write_text(json.dumps(pins))
    rc = lp.main(["--repo", str(REPO), "--w", W, "--wave-json", str(wave), "--materialisation", "draft", "--out", str(tmp_path / "p2.json"),
                  "--ledger", str(tmp_path / "l2.json"), "--peer-roster", str(tmp_path / "ro.yaml"), "--peer-pins", str(tmp_path / "pins.json")])
    assert rc == 0
    p2 = json.loads((tmp_path / "p2.json").read_text())
    assert p2["s5_probe_rows"] == [] and p2["s5_multiturn_probe"]["status"].startswith("UNREGISTERED")


def test_a_final_at_the_s4_windows_holds_to_its_draft_with_the_probe_registered(inputs, plan):
    """The FINAL (S4-decided windows) == the DRAFT outside the S4 fields WITH the probe rows present (the dormant dose path off, so the
    probe rides its own smoke-only rows): a raw probe row made every final differ on geometry / readiness / argv windows. A real invariant
    change inside a probe row (a pin, a route) still surfaces under the row's standup id."""
    import copy
    final = lp.build_plan(W=W, wave_uuid=WAVE, reqs_sha256=inputs["reqs_sha256"], roster=inputs["roster"], pins=inputs["pins"], arditi_pins=inputs["arditi_pins"],
                          arms=inputs["arms"], materialisation="final", s4={"multiturn_window": {"crescendo": 16384, "actorattack": 16384}},
                          generated_at="2026-09-18T00:00:01Z", peer_roster=inputs["peer_roster"], peer_pins=inputs["peer_pins"],
                          config_private_runs=inputs["config_private_runs"])
    assert len(final["s5_probe_rows"]) == 4 and {r["geometry"]["L"] for r in final["s5_probe_rows"]} == {16384} and {r["geometry"]["L"] for r in plan["s5_probe_rows"]} == {49152}
    assert lp.compare_draft_final(plan, final) == []
    proj = lp.scientific_projection(plan)
    assert "s5_probe_rows" not in proj and all(proj["rows"][r["standup_id"]]["s5_probe"] is True for r in plan["s5_probe_rows"])
    bad = copy.deepcopy(final)
    # the two leg rows of one view share ONE `objects` dict (s5_probe_rows builds it per view); un-alias before mutating so the diff is one row's
    bad["s5_probe_rows"][0]["objects"] = json.loads(json.dumps(bad["s5_probe_rows"][0]["objects"]))
    bad["s5_probe_rows"][0]["objects"][LORA]["object_sha256"] = "0" * 64
    diffs = lp.compare_draft_final(plan, bad)
    assert diffs and all(d.startswith(f"/rows/{bad['s5_probe_rows'][0]['standup_id']}/objects") for d in diffs)
    bad = copy.deepcopy(final)
    bad["s5_probe_rows"][1]["route_set"] = ["none"]
    assert any(d.startswith(f"/rows/{bad['s5_probe_rows'][1]['standup_id']}/route_set") for d in lp.compare_draft_final(plan, bad))
