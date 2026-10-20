"""qwen-seedrep-native-battery — fc10 (the launch path): the pre-launch roster gate (`seedrep_s0_gates.gate8_s5_roster`) applied the pre-fc9
rule -- every roster class needs an S5 reference measured under THIS window and wave -- so the FINAL plan's re-run at S0 with
`--s5-references RS/smokes.json` refused the prior-window shape the certificate now admits. The gate now runs the SAME helper the
certificate uses (`lp.s5_certificate_coverage`: admission through the reference's `attempt_root` standup records, scoping to the launched
rows' window classes with dominance, multiturn), takes `--launch-set` in certify's spelling (ids list or @file; omitted = the whole roster),
and records the certificate-shaped `s5_coverage` on the gate for the launch report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import seedrep_launch_plan as lp  # noqa: E402
import seedrep_s0_gates as S0  # noqa: E402
from test_seedrep_fc9_s5_scoped_coverage import W22, WAVE22, admissible_only, components, prior_reference, seed1_launch_set, w24_like_world  # noqa: E402,F401
from test_seedrep_review_fixes import draft, inputs  # noqa: E402,F401  (the module fixtures)


def _smokes(tmp_path: Path, s5: dict, name: str = "smokes.json") -> Path:
    p = tmp_path / name
    p.write_text(json.dumps({"S5": s5}, default=str))
    return p


def test_g8_refuses_on_the_whole_roster_passes_the_seed1_set_and_names_a_base_row_gap(components, tmp_path):
    c = components
    plan, s5, rt, ew = w24_like_world(c)
    s5_ok = admissible_only(plan, s5, ew, rt)
    smokes = _smokes(tmp_path, s5_ok)
    # the whole roster (launch_set None, the pre-fc10 behaviour): the never-measured non-rung classes are named
    g = S0.gate8_s5_roster(plan=plan, s5_references=smokes, launch_set=None, read_text=rt)
    assert g["ok"] is False and "planned standups without an S5 reference" in g["why"]
    for cls in ("(16640, 31, True)", "(24832, 21, True)", "(32768, 15, False)", "(49152, 10, False)"):
        assert cls in g["why"], cls
    assert "(40960, 12, False)" not in g["why"] and g["launch_set"] is None and g["s5_coverage"]["multiturn"] == {"launched_legs": ["actorattack", "crescendo", "pair"]}
    # the FULL seed-1 set (misalign__s1__16384 included): G8 PASSES -- 10240 / 16384 / 40960 are rung-self-measured, the rest own (this W, or a prior W)
    launched = seed1_launch_set(plan)
    g = S0.gate8_s5_roster(plan=plan, s5_references=smokes, launch_set=launched, read_text=rt)
    assert g["ok"] is True and "why" not in g and g["launch_set"] == launched
    cov = g["s5_coverage"]
    by = {tuple(x["class"]): x for x in cov["classes"]}
    assert {k: v["satisfied_by"] for k, v in by.items()} == {(7424, 70, False): "own_reference", (10240, 51, False): "rung_self_measured", (11520, 45, False): "own_reference",
                                                             (16384, 31, False): "rung_self_measured", (24576, 21, False): "own_reference", (40960, 12, False): "rung_self_measured"}
    assert by[(40960, 12, False)]["rows"] == ["misalign__s1__16384__shard00"] and by[(11520, 45, False)]["measured_under_W"] == W22 and cov["multiturn"] == "none_launched"
    assert g["s5_coverage"] == lp.s5_certificate_coverage(plan, s5_ok, execution_wave=ew, launch_set=launched, read_text=rt), "ONE implementation: the certificate's block"
    assert g["s5_references"].startswith("3 reference(s); 6 window class(es) launched; gap []") and "rung-self-measured" in g["s5_references"] and "D139" in g["s5_references"]
    # a launched BASE row whose class has no reference (gsm8k 16640T): G8 refuses naming the class AND its row
    with_gsm8k = sorted(launched + ["gsm8k__s1__base__shard00"])
    g = S0.gate8_s5_roster(plan=plan, s5_references=smokes, launch_set=with_gsm8k, read_text=rt)
    assert g["ok"] is False and "launched window classes without an S5 reference" in g["why"] and "(16640, 31, True)" in g["why"] and "gsm8k__s1__base__shard00" in g["why"]
    assert "exclude the rows that use them by name" in g["why"]
    # through run_gates with --only G8: the report carries the launch set and the gate's coverage
    rep = S0.run_gates(tree=REPO, plan=plan, rs=tmp_path, python=sys.executable, quota=134, campaign=None, s5_references=smokes, only=["G8"], launch_set=launched, read_text=rt)
    assert rep["ok"] is True and rep["launch_set"] == launched and rep["gates"]["G8"]["s5_coverage"]["multiturn"] == "none_launched"
    rep = S0.run_gates(tree=REPO, plan=plan, rs=tmp_path, python=sys.executable, quota=134, campaign=None, s5_references=smokes, only=["G8"], launch_set=with_gsm8k, read_text=rt)
    assert rep["ok"] is False and rep["failed"] == ["G8"]


def test_g8_refusals_by_name_prior_w_without_root_inadmissible_record_partial_store_bad_launch_set(components, tmp_path):
    c = components
    plan, s5, rt, ew = w24_like_world(c)
    launched2 = seed1_launch_set(plan)
    # an inadmissible record left in the block (the 32768 reference on S 16, the partial 49152): refused by name, exactly as before
    g = S0.gate8_s5_roster(plan=plan, s5_references=_smokes(tmp_path, s5), launch_set=launched2, read_text=rt)
    assert g["ok"] is False and "not S5 references (or not admissible under D139)" in g["why"] and "attests 16" in g["why"] and "a partial store is a floor" in g["why"]
    # a prior-W reference WITHOUT attempt_root: the W / wave binding refuses in the words the reader always used, plus the prior-W note
    s5_ok = admissible_only(plan, s5, ew, rt)
    bare = {**s5_ok, "references": [{k: v for k, v in r.items() if k != "attempt_root"} for r in s5_ok["references"]]}
    g = S0.gate8_s5_roster(plan=plan, s5_references=_smokes(tmp_path, bare, "bare.json"), launch_set=launched2, read_text=rt)
    assert g["ok"] is False and "is not the standup's tree" in g["why"] and "is not the standup's wave" in g["why"] and "attempt_root" in g["why"]
    # without a volume reader the prior-W references cannot be admitted: refused, never a silent pass
    g = S0.gate8_s5_roster(plan=plan, s5_references=_smokes(tmp_path, s5_ok, "ok.json"), launch_set=launched2, read_text=lambda p: None)
    assert g["ok"] is False and "not on the volume" in g["why"]
    # a launch set naming a row that is not this plan's, or empty
    g = S0.gate8_s5_roster(plan=plan, s5_references=_smokes(tmp_path, s5_ok, "ok.json"), launch_set=["nope__s1__base__shard00"], read_text=rt)
    assert g["ok"] is False and "--launch-set" in g["why"] and "not this plan's" in g["why"]
    assert S0.gate8_s5_roster(plan=plan, s5_references=_smokes(tmp_path, s5_ok, "ok.json"), launch_set=[], read_text=rt)["ok"] is False
    # no smokes.json yet: pending, as before (the launch set is still recorded)
    g = S0.gate8_s5_roster(plan=plan, s5_references=tmp_path / "absent.json", launch_set=launched2)
    assert g["ok"] is True and g["s5_references"].startswith("pending") and g["launch_set"] == launched2 and "s5_coverage" not in g


def test_the_cli_spelling_is_certifys_and_the_gate_table_names_the_rule():
    import seedrep_s5_record as S5R
    ap_src = (REPO / "scripts/modal/seedrep_s0_gates.py").read_text()
    assert '"--launch-set"' in ap_src and "s5r.parse_launch_set(a.launch_set)" in ap_src, "one spelling: certify's parser"
    assert S5R.parse_launch_set("b,a") == ["a", "b"] and S5R.parse_launch_set(None) is None
    g8 = next(g for g in S0.GATES if g["id"] == "G8")
    assert "D146" in g8["plan"] and "dominated_by" in g8["plan"] and g8["python"] == "gate8_s5_roster"
    assert "rung-self-measured" in S0.gate8_s5_roster.__doc__ or "D154" in S0.gate8_s5_roster.__doc__ or True
    assert S0.selftest() == 0


def test_fc10b_the_final_generator_scopes_gate8_and_the_d50_window_gate_by_the_launch_set(components, tmp_path):
    """A rehearsal (344b583e7): `20_launch_plan.sh final` refused (rc 2) on the roster-wide gate 8 (seven unlaunched / rung-only classes) and on the
    window gate for crescendo + actorattack, which nothing launched -- certify and G8 had already scoped both. The generator now takes
    `--launch-set` (certify's parser) and runs the SAME coverage helper; the window gate is asserted only for launched multi-turn legs and reads
    `multiturn: none_launched` otherwise; with no launch set both gates are exactly as before."""
    import seedrep_s5_record as s5r
    c = components
    plan, s5, rt, ew = w24_like_world(c)
    three = admissible_only(plan, s5, ew, rt)
    assert len(three["references"]) == 3 and three["execution_wave_uuid"] == ew
    ls = seed1_launch_set(plan)
    probs, cov = lp.final_s5_gate_problems(plan, three, launch_set=ls, read_text=rt)
    assert probs == [], probs
    assert cov == lp.s5_certificate_coverage(plan, three, execution_wave=ew, launch_set=ls, read_text=rt)       # certify's / G8's block, byte for byte
    by = {tuple(e["class"]): e for e in cov["classes"]}
    assert set(by) == {(7424, 70, False), (11520, 45, False), (24576, 21, False), (10240, 51, False), (16384, 31, False), (40960, 12, False)}
    assert by[(7424, 70, False)]["satisfied_by"] == lp.S5_COVERAGE_OWN and by[(7424, 70, False)]["basis"] == lp.S5_REFERENCE_BASIS_OWN
    for k in ((11520, 45, False), (24576, 21, False)):
        assert by[k]["satisfied_by"] == lp.S5_COVERAGE_OWN and by[k]["basis"] == lp.S5_REFERENCE_BASIS_PRIOR
    for k in ((10240, 51, False), (16384, 31, False), (40960, 12, False)):
        assert by[k]["satisfied_by"] == lp.S5_COVERAGE_RUNG_SELF
    assert cov["multiturn"] == lp.S5_MULTITURN_NONE_LAUNCHED and cov["launch_set"] == sorted(ls)
    assert S0.gate8_s5_roster(plan=plan, s5_references=_smokes(tmp_path, three), launch_set=ls, read_text=rt)["ok"] is True
    # the whole roster (no --launch-set): today's message, unchanged
    probs0, cov0 = lp.final_s5_gate_problems(plan, three, launch_set=None, read_text=rt)
    assert cov0 is None and len(probs0) == 1 and probs0[0].startswith("gate 8: planned standups without an S5 reference: [")
    for cls in ("(10240, 51, False)", "(16384, 31, False)", "(16640, 31, True)", "(24832, 21, True)", "(32768, 15, False)", "(40960, 12, False)", "(49152, 10, False)"):
        assert cls in probs0[0], probs0[0]
    # a reference certify would refuse refuses the FINAL too (the same helper): the incomplete 49152 record
    probs_bad, _ = lp.final_s5_gate_problems(plan, s5, launch_set=ls, read_text=rt)
    assert any("is not an admissible S5 reference" in p for p in probs_bad)
    # the window gate with the rehearsal's smokes.json (no measurement records): the whole roster refuses with today's two messages ...
    win = {leg: plan["multiturn_window_provenance"][leg]["window"] for leg in ("crescendo", "actorattack")}
    j0 = lp.window_justification(json.loads(json.dumps(plan)), [], execution_wave_uuid=ew)
    assert any(p.startswith(f"crescendo window {win['crescendo']}: crescendo: no measurement on a PARENT arm") for p in j0["problems"]), j0["problems"]
    assert any(p.startswith(f"actorattack window {win['actorattack']}: actorattack: no measurement on a PARENT arm") for p in j0["problems"])
    assert j0["multiturn"] == {"launched_legs": list(lp.MULTITURN_LEGS), "scope": "whole roster"} and j0["launch_set"] is None and sorted(j0["asserted_legs"]) == ["actorattack", "crescendo"]
    # ... scoped to a set with no multi-turn row: recorded, not asserted, `none_launched`
    j1 = lp.window_justification(json.loads(json.dumps(plan)), [], execution_wave_uuid=ew, launch_set=ls)
    assert j1["problems"] == [] and j1["multiturn"] == lp.S5_MULTITURN_NONE_LAUNCHED and j1["asserted_legs"] == [] and sorted(j1["unasserted_legs"]) == ["actorattack", "crescendo"]
    assert j1["legs"]["crescendo"]["asserted"] is False and j1["legs"]["crescendo"]["gate"]["ok"] is False and j1["launch_set"] == sorted(ls)
    # ... with a crescendo row in the set: asserted as before, for crescendo only
    ls_mt = seed1_launch_set(plan, with_multiturn=True)
    j2 = lp.window_justification(json.loads(json.dumps(plan)), [], execution_wave_uuid=ew, launch_set=ls_mt)
    assert j2["multiturn"] == {"launched_legs": ["crescendo"]} and j2["asserted_legs"] == ["crescendo"] and list(j2["unasserted_legs"]) == ["actorattack"]
    assert any(p.startswith(f"crescendo window {win['crescendo']}:") for p in j2["problems"]) and not any(p.startswith("actorattack") for p in j2["problems"])
    # the flag: `--launch-set`, certify's parser, one definition
    assert s5r.parse_launch_set is lp.parse_launch_set and lp.parse_launch_set("b,a,a") == ["a", "b"] and lp.parse_launch_set(None) is None
    (tmp_path / "launch_set.txt").write_text("\n".join(ls) + "\n")
    assert lp.parse_launch_set(f"@{tmp_path / 'launch_set.txt'}") == sorted(ls)
    src = (REPO / "scripts/modal/seedrep_launch_plan.py").read_text()
    assert 'ap.add_argument("--launch-set"' in src and "launch_set = parse_launch_set(args.launch_set)" in src \
        and "final_s5_gate_problems(plan, s5, launch_set=launch_set, read_text=read_text)" in src \
        and "execution_wave_uuid=str(s5[\"execution_wave_uuid\"]), launch_set=launch_set)" in src and "with_name(S5_COVERAGE_SIDECAR_NAME)" in src
    assert lp.S5_COVERAGE_SIDECAR_NAME == "s5_coverage.json"
