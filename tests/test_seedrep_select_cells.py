"""Explicit six-object selection on every producer.

`select_cells.py <view>` prints exactly the roster view's admitted cells — its steered objects, then the prompted SI baselines the
roster DECLARES on the view (`prompted__<si>: null`; commit 978a7a97 declares the two on every seed view, served beside the
objects — admitted exactly as declared) — never `none`, never a prompted cell with a staged id, never a
zero-adapter control, never a foreign object; the launch plan's MEASURED cells stay the objects; a view short of
the six is allowed (a derived object the admission refused is simply not on its roster view — on 2026-09-17 the
distilled PSR difference failed admission for every seed, so every live view carries FIVE); --require-complete refuses only a
view that still declares a `pending_fit` object (neither staged into `cells` nor ruled out of the view); the
Petri indices come from the frozen enumeration's CANONICAL legs only; the launch plan (RS/reseed/launch_plan.json) names the
six as the MEASURED cells of every producer and no measurement request targets the anchor, a prompted baseline or a control.
Everything synthetic: an invented roster, placeholder staged ids, a synthetic cells.json; nothing is launched."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "scripts" / "sdfbase" / "full22", REPO / "scripts" / "sdfbase" / "seedrep"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import eval_seed as es  # noqa: E402
import select_cells as sc  # noqa: E402
from f22_roster import load_roster, pending_fit_of  # noqa: E402

SCRIPT = REPO / "scripts/sdfbase/seedrep/select_cells.py"
SIX = list(sc.SIX_CELLS)
DISTILLED = "constv2-mix-full__psr_ce_all_minus_ctl_distilled"
FIVE = [c for c in SIX if c != DISTILLED]          # the admitted cells of every live view (decisions O20)
SHA = "ab" * 20


def _roster(views: dict) -> dict:
    return {"targets": {"qwen35_4b": {"model": "Qwen/Qwen3.5-4B", "model_revision": "0" * 40, "views": views}}}


def _view(s: int, cells=None, pending_fit=None, **override) -> dict:
    cells = dict(cells) if cells is not None else {c: sc.staged_id_for(f"s{s}", c) for c in SIX}
    cells.update(override)
    v = {"config": f"configs/sdfbase_full22_qwen35_4b_s{s}.yaml", "manifest": f"configs/capeval/manifests/petri_full22_qwen35_4b_s{s}.yaml",
         "cells": cells}
    if pending_fit:
        v["pending_fit"] = dict(pending_fit)
    return v


def _write(tmp_path, roster: dict) -> Path:
    p = tmp_path / "roster.yaml"
    p.write_text(yaml.safe_dump(roster, sort_keys=False))
    return p


def test_the_six_cells_in_roster_order_and_the_staged_id_grammar():
    assert sc.SIX_CELLS == ("constv2-mix-full__psr_ce_all", "constv2-mix-full__lora_r4_ce", "constv2ctl2-full__psr_ce_all",
                            "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl", "constv2-mix-full__psr_ce_all_minus_ctl_distilled")
    assert sc.staged_id_for("s2", "constv2-mix-full__psr_ce_all") == "base64s2__constv2-mix-full__psr_ce_all"
    assert sc.staged_id_for("s3", "constv2-mix-full__lora_r4_ce_minus_ctl") == "derived_base64s3__constv2-mix-full__lora_r4_ce_minus_ctl"
    roster = _roster({"s1": _view(1), "s2": _view(2)})
    assert list(sc.select(roster, "s1")) == SIX and sc.select(roster, "s1") == {c: sc.staged_id_for("s1", c) for c in SIX}
    assert list(sc.select(roster, "s2", require_complete=True)) == SIX
    # roster order is the printed order (a roster that lists the derived pair first prints it first)
    reordered = {c: sc.staged_id_for("s1", c) for c in reversed(SIX)}
    assert list(sc.select(_roster({"s1": _view(1, reordered)}), "s1")) == list(reversed(SIX))


def test_a_ruled_view_short_of_the_six_is_complete_but_a_pending_fit_object_refuses_complete_and_a_foreign_object_refuses():
    five = {c: sc.staged_id_for("s2", c) for c in SIX if c != DISTILLED}
    roster = _roster({"s2": _view(2, five)})
    assert list(sc.select(roster, "s2")) == SIX[:-1]
    # RULED: no pending_fit left — five admitted cells ARE complete (decisions O20: "fewer cells if an admission failed")
    assert list(sc.select(roster, "s2", require_complete=True)) == SIX[:-1]
    # an object still under pending_fit is neither staged nor ruled out: --require-complete refuses BY NAME, the plain call lists the five
    pending = _roster({"s2": _view(2, five, pending_fit={DISTILLED: sc.staged_id_for("s2", DISTILLED)})})
    assert list(sc.select(pending, "s2")) == SIX[:-1]
    with pytest.raises(SystemExit, match="still declares pending_fit .*psr_ce_all_minus_ctl_distilled.*5 of the six declared cells admitted.*--require-complete refused"):
        sc.select(pending, "s2", require_complete=True)
    # the batch-8 object, another replicate's object, a pin mapping: never this replicate's cell
    for bad, needle in (("base__constv2-mix-full__psr_ce_all", "not this replicate's object"),
                        ("base64s2__constv2-mix-full__psr_ce_all", "not this replicate's object"),
                        ({"pins": "p.json", "pin": "qwen35_4b.x"}, "never a pin mapping")):
        with pytest.raises(SystemExit, match=needle):
            sc.select(_roster({"s1": _view(1, **{"constv2-mix-full__psr_ce_all": bad})}), "s1")


def test_none_prompted_controls_and_unknown_cells_are_refused_as_measurement_targets():
    assert sc.is_measurement_forbidden("none") and sc.is_measurement_forbidden("prompted__si_constitution") and sc.is_measurement_forbidden("none__lens0")
    assert sc.is_measurement_forbidden("constv2-mix-full__psr_ce_all__lora0") and sc.is_measurement_forbidden("safety_refusal__psr_kl_all")
    assert sc.is_measurement_forbidden("constv2-mix-full__psr_ce_all") is None
    # a DECLARED prompted baseline (`prompted__<si>: null`, the roster's prompted grammar) is ADMITTED as declared — served beside the
    # objects, `None` staged id, listed where the roster lists it (commit 978a7a97 puts the two on every seed view) — while it stays a
    # forbidden MEASUREMENT target of the launch plan (is_measurement_forbidden above)
    admitted = sc.select(_roster({"s1": _view(1, **{"prompted__si_constitution": None})}), "s1")
    assert list(admitted) == SIX + ["prompted__si_constitution"] and admitted["prompted__si_constitution"] is None
    assert admitted == {**{c: sc.staged_id_for("s1", c) for c in SIX}, "prompted__si_constitution": None}
    cases = (({"none": "base64s1__none"}, "unsteered anchor"),          # (a null id is f22_roster's own refusal first — below)
             ({"prompted__safety_refusal": "si12__safety_refusal__psr_kl_all"}, "prompted cell .* maps to a staged id"),   # f22_roster's own refusal, surfaced
             ({"none__lens0": "base64s1__x"}, "zero-adapter control"),
             ({"safety_refusal__psr_kl_all": "si12__safety_refusal__psr_kl_all"}, "campaign's main-view controls"),
             ({"constv2-mix-1m__psr_ce_all": "base64s1__constv2-mix-1m__psr_ce_all"}, "not one of the lane's six cells"))
    for extra, needle in cases:
        with pytest.raises(SystemExit, match=needle):
            sc.select(_roster({"s1": _view(1, **extra)}), "s1")
    with pytest.raises(SystemExit, match="maps to null"):
        sc.select(_roster({"s1": _view(1, **{"constv2-mix-full__lora_r4_ce": None})}), "s1")
    with pytest.raises(SystemExit, match="cell none maps to null"):                                   # f22_roster's grammar, surfaced
        sc.select(_roster({"s1": _view(1, **{"none": None})}), "s1")
    # only the seed-replicate views; a roster without the view refuses by name
    for view, needle in (("main", "not a seed-replicate view"), ("it", "not a seed-replicate view"), ("s4", "not a seed-replicate view"),
                         ("s3", "roster names no view qwen35_4b/s3")):
        with pytest.raises(SystemExit, match=needle):
            sc.select(_roster({"s1": _view(1)}), view)
    with pytest.raises(SystemExit, match="lists no steered cell"):
        sc.select(_roster({"s1": _view(1, {})}), "s1")
    with pytest.raises(SystemExit, match="lists no steered cell"):                                     # prompted baselines alone are no object
        sc.select(_roster({"s1": _view(1, {"prompted__safety_refusal": None, "prompted__si_constitution": None})}), "s1")


def _cells_json(tmp_path, canonical_ids, control_ids):
    from sisteer.capeval.petri import registry_adapter as ra
    rec = {"serving_path": "lora_native",
           "canonical": [{"cell_id": (ra.harness_cell_id(c) if c not in ("none",) else c), "petri_run_key": f"k{i}", "virtual_model_name": f"vm{i}"}
                         for i, c in enumerate(canonical_ids)],
           "controls": [{"cell_id": (ra.harness_cell_id(c) if not c.startswith("none") else c), "petri_run_key": f"c{i}", "virtual_model_name": f"cvm{i}"}
                        for i, c in enumerate(control_ids)]}
    p = tmp_path / "cells.json"
    p.write_text(json.dumps(rec))
    return p


def test_petri_indices_are_the_six_canonical_legs_never_a_control_and_never_the_anchor(tmp_path):
    p = _cells_json(tmp_path, ["none", *SIX], ["none__lens0"])
    assert sc.petri_indices(p, SIX) == [1, 2, 3, 4, 5, 6]                       # never 0 (none), never 7 (the controls twin)
    assert sc.petri_indices(p, SIX[:2]) == [1, 2]
    # a different frozen order maps by id, not by position
    (tmp_path / "b").mkdir()
    p2 = _cells_json(tmp_path / "b", [SIX[5], "none", *SIX[:5]], ["none__lens0"])
    assert sc.petri_indices(p2, SIX) == [2, 3, 4, 5, 6, 0]
    # a cell frozen as a `controls` leg, or absent, refuses
    (tmp_path / "c").mkdir()
    p3 = _cells_json(tmp_path / "c", ["none", *SIX[:5]], [SIX[5]])
    with pytest.raises(SystemExit, match="controls` leg"):
        sc.petri_indices(p3, SIX)
    (tmp_path / "d").mkdir()
    p4 = _cells_json(tmp_path / "d", ["none", *SIX[:4]], [])
    with pytest.raises(SystemExit, match="no leg named"):
        sc.petri_indices(p4, SIX)


def test_the_launch_plan_names_the_six_on_every_producer_and_never_a_forbidden_target(tmp_path):
    roster = _roster({"s2": _view(2)})
    cells = sc.select(roster, "s2")
    cj = _cells_json(tmp_path, ["none", *SIX], ["none__lens0"])
    plan = sc.launch_plan("s2", cells, wave_sha=SHA, petri_cells_json=cj)
    assert plan["schema"] == "seedrep_launch_plan/1" and plan["view"] == "s2" and plan["tag"] == "qwen35_4b_s2" and plan["seed"] == 2
    assert plan["seed_base"] == es.seed_base(2) and plan["cells"] == SIX and plan["staged_ids"] == cells
    legs = [p["leg"] for p in plan["producers"]]
    assert legs == ["static_sampled", "over_refusal", "static_sampled", "over_refusal", "mmlu", "bon", "gsm8k", "multiturn", "multiturn", "multiturn", "petri", "misalign"]
    covered = sorted(e for p in plan["producers"] for e in p["eval_ids"])
    assert sorted(set(covered)) == sorted(es.EVAL_IDS)                                # every leg of the coverage table has a producer
    forbidden = {"none", "none__lens0", "prompted__si_constitution", "prompted__safety_refusal", "safety_refusal__psr_kl_all"}
    csv = ",".join(SIX)
    for p in plan["producers"]:
        assert p["measured_cells"] == SIX and not (set(p["measured_cells"]) & forbidden)
        assert not any(a == "all" or a == "" and p["leg"] in ("static_sampled", "over_refusal", "mmlu", "bon") for a in p["argv"])
        for e in p["eval_ids"]:
            assert p["seed_rule"][e] == es.leg_seed_rule(e).seed_rule and p["seed_base"][e] == es.leg_seed_base(e, 2)
        assert "--export=" not in " ".join(p["argv"])
    by_leg = {}
    for p in plan["producers"]:
        by_leg.setdefault(p["leg"], []).append(p)
    # the sampled / static jobs carry the literal comma list in the documented positional
    for p in by_leg["static_sampled"] + by_leg["over_refusal"]:
        assert p["argv"][-2] == csv and p["argv"][-1] == "s2" and p["argv"][-3] == "9" and p["argv"][-4] == SHA
    for p in by_leg["mmlu"] + by_leg["bon"]:
        assert p["argv"][-1] == csv and p["argv"][-2] == "9" and p["argv"][-3] == SHA and p["argv"][-5:-3] == ["s2", "controls"] or p["argv"][-5:-3] == ["s2", "bon"]
    bon = by_leg["bon"][0]
    assert bon["env"] == {"F22_BON_DECODE_PRESET": "anchor_match", "F22_BON_DECODE_SEED_POLICY": "replicate_v1", "F22_BON_DECODE_SEED_BASE": str(es.seed_base(2))}
    assert by_leg["gsm8k"][0]["argv"][-4:] == ["ARRAY", SHA, "8192", "qwen_seedrep"]
    # decisions O12a: the GSM8K paired-200 array is the ONE producer that also measures an unsteered anchor of its own — the matrix's
    # natively served `none` row per view — recorded apart from the six MEASURED cells; every other producer records none
    g8k = by_leg["gsm8k"][0]
    assert g8k["anchor_measured"] == ["none"] and g8k["measured_cells"] == SIX and "O12a" in g8k["notes"] and "O106" in g8k["notes"]
    assert all(p["anchor_measured"] == [] for p in plan["producers"] if p["leg"] != "gsm8k")
    assert "EXCEPT the GSM8K" in plan["anchor_policy"] and "anchor_measured" in plan["anchor_policy"]
    assert all(p["argv"][2] in ("crescendo", "pair", "actorattack") or p["argv"][-11] in ("crescendo", "pair", "actorattack") for p in by_leg["multiturn"])
    assert all(p["env"] == {"F22_LOG_REQUESTS": "1"} and "temperature=1.0,top_p=1.0,top_k=-1" in p["argv"] for p in by_leg["multiturn"])
    petri = by_leg["petri"][0]
    assert petri["petri_indices"] == [1, 2, 3, 4, 5, 6] and petri["env"] == {"F22_SERVER_DEFAULT_POLICY": "engine_defaults"} and "never ALL" not in petri["notes"]
    mis = by_leg["misalign"][0]
    assert mis["served_only"] == ["none"] and mis["measured_cells"] == SIX and mis["argv"][-3:] == ["qwen35_4b_s2", SHA, "full"]
    assert mis["env"] == {"F22_SERVER_DEFAULT_POLICY": "engine_defaults"}
    # without a frozen enumeration the plan records the indices as pending — never `ALL`
    plan2 = sc.launch_plan("s2", cells, wave_sha=SHA)
    petri2 = [p for p in plan2["producers"] if p["leg"] == "petri"][0]
    assert petri2["petri_indices"] is None and "never ALL" in petri2["notes"]
    # a plan for a forbidden cell can never be built
    with pytest.raises(SystemExit, match="unsteered anchor"):
        sc.launch_plan("s2", {"none": None, **cells}, wave_sha=SHA)
    # the roster's DECLARED prompted baselines ride the plan as `prompted_cells` — never a MEASURED cell of any producer
    with_prompted = sc.select(_roster({"s2": _view(2, **{"prompted__safety_refusal": None, "prompted__si_constitution": None})}), "s2")
    plan3 = sc.launch_plan("s2", with_prompted, wave_sha=SHA, petri_cells_json=cj)
    assert plan3["cells"] == SIX and plan3["staged_ids"] == cells and plan3["prompted_cells"] == ["prompted__safety_refusal", "prompted__si_constitution"]
    assert all(p["measured_cells"] == SIX for p in plan3["producers"]) and plan["prompted_cells"] == []
    assert [p for p in plan3["producers"] if p["leg"] == "petri"][0]["petri_indices"] == [1, 2, 3, 4, 5, 6]


def test_the_cli_prints_the_list_writes_the_plan_and_refuses_on_stderr(tmp_path):
    good = _write(tmp_path, _roster({"s1": _view(1), "s3": _view(3, {c: sc.staged_id_for("s3", c) for c in SIX[:4]},
                                                            pending_fit={c: sc.staged_id_for("s3", c) for c in SIX[4:]})}))
    py = sys.executable
    env = {"PYTHONPATH": str(REPO / "src")}
    r = subprocess.run([py, str(SCRIPT), "s1", "--roster", str(good)], capture_output=True, text=True, env=env, check=False)
    assert r.returncode == 0 and r.stdout.strip() == ",".join(SIX) and r.stderr == ""
    r = subprocess.run([py, str(SCRIPT), "s3", "--roster", str(good)], capture_output=True, text=True, env=env, check=False)
    assert r.returncode == 0 and r.stdout.strip() == ",".join(SIX[:4])
    r = subprocess.run([py, str(SCRIPT), "s3", "--roster", str(good), "--require-complete"], capture_output=True, text=True, env=env, check=False)
    assert r.returncode == 2 and r.stdout == "" and "still declares pending_fit" in r.stderr and "--require-complete refused" in r.stderr
    r = subprocess.run([py, str(SCRIPT), "s2", "--roster", str(good)], capture_output=True, text=True, env=env, check=False)
    assert r.returncode == 2 and "roster names no view qwen35_4b/s2" in r.stderr
    cj = _cells_json(tmp_path, ["none", *SIX], ["none__lens0"])
    r = subprocess.run([py, str(SCRIPT), "s1", "--roster", str(good), "--petri-indices", str(cj)], capture_output=True, text=True, env=env, check=False)
    assert r.returncode == 0 and r.stdout.strip() == "1 2 3 4 5 6"
    out = tmp_path / "RS" / "launch_plan.json"
    r = subprocess.run([py, str(SCRIPT), "s1", "--roster", str(good), "--launch-plan", str(out), "--wave-sha", SHA, "--petri-cells-json", str(cj)],
                       capture_output=True, text=True, env=env, check=False)
    assert r.returncode == 0 and r.stdout.strip() == str(out), r.stderr
    plan = json.loads(out.read_text())
    assert plan["cells"] == SIX and all(p["measured_cells"] == SIX for p in plan["producers"]) and plan["seed_base"] == es.seed_base(1)
    r = subprocess.run([py, str(SCRIPT), "s1", "--roster", str(good), "--launch-plan", str(out)], capture_output=True, text=True, env=env, check=False)
    assert r.returncode == 2 and "needs --wave-sha" in r.stderr
    # the LIVE roster today (the derived pass staged 2026-09-17: the LoRA difference admitted for every seed and moved
    # pending_fit -> cells; the distilled PSR difference NOT admitted for any seed and DROPPED from every view, no pending_fit left;
    # then commit 978a7a97: the two prompted SI baselines DECLARED on every seed view, LAST, `null`): every view
    # lists its FIVE admitted objects, then the two prompted baselines, in roster order, and --require-complete PASSES — the view is
    # RULED. The expectation is DERIVED from the roster, never typed: objects (staged ids) first, prompted (`null`) last.
    live = load_roster(Path(sc.DEFAULT_ROSTER))
    for v in ("s1", "s2", "s3"):
        assert pending_fit_of(live, "qwen35_4b", v) == {}, v
        declared = live["targets"]["qwen35_4b"]["views"][v]["cells"]
        assert DISTILLED not in declared, v
        objects = [c for c, sid in declared.items() if sid is not None]
        prompted = [c for c, sid in declared.items() if sid is None]
        assert objects == FIVE and prompted == ["prompted__safety_refusal", "prompted__si_constitution"], (v, declared)
        assert list(declared) == objects + prompted, "objects before the prompted cells, which stay LAST"
        r = subprocess.run([py, str(SCRIPT), v], capture_output=True, text=True, env=env, check=False)
        assert r.returncode == 0 and r.stdout.strip() == ",".join(list(declared)) and r.stderr == "", (v, r.stderr)
        r = subprocess.run([py, str(SCRIPT), v, "--require-complete"], capture_output=True, text=True, env=env, check=False)
        assert r.returncode == 0 and r.stdout.strip() == ",".join(list(declared)) and r.stderr == "", (v, r.stderr)
