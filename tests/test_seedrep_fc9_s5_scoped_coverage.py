"""qwen-seedrep-native-battery: `certificate_problems` required a reference for EVERY roster class measured under THIS plan's
tree and wave, plus a non-empty multi-turn reference list -- under the previous plan only 7424 was admissible, the earlier
references refused on tree / wave, 49152 is a partial store, and the rung classes cannot be measured before launch. Now: a
reference measured under a PRIOR tree / wave is admitted when its `attempt_root`'s ATTEMPT.json is the record's and its
standup's engine identity + geometry (endpoint.json: engine, L, S, thinking, graph mode, server default policy, model +
revision, served objects in this plan's adapter bank) match the plan's standup for that class, with `measured_under_W` /
`measured_under_wave` + the qualification recorded; a partial store stays a floor. The reference gap is scoped to the window
classes of the LAUNCHED rows (`launch_set`), a class satisfied by its own complete reference or by a complete reference of a
LARGER window at the same thinking mode (`dominated_by`); a class with neither is a named gap. The multi-turn list may be
empty iff no multi-turn row is launched (`multiturn: none_launched`). The certificate records `launch_set` and the derived
reference coverage, re-derived at `--go`, and a `--go` row outside the certified set is refused by name.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "scripts" / "sdfbase" / "seedrep", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import seedrep_launch_plan as lp  # noqa: E402
from test_seedrep_review_fixes import _row, _uuid4, draft, inputs  # noqa: E402,F401  (the module fixtures)
from test_seedrep_review_fixes import TestSolR3_2_launch_certificate as _CertComponents  # noqa: E402  (aliased: not collected here)

W22 = "5" * 40
W19 = "6" * 40
WAVE22, WAVE19 = _uuid4(22), _uuid4(19)


@pytest.fixture(scope="module")
def components(inputs, draft, tmp_path_factory):
    return _CertComponents()._components(inputs, draft, tmp_path_factory.mktemp("fc9"))


def _class_row(plan, L, S, th):
    return next(r for r in lp._plan_rows_the_app_may_execute(plan) if (r["geometry"]["L"], r["geometry"]["S"], r["geometry"]["thinking"]) == (L, S, th))


def prior_root(W, leg):
    return f"{lp.LANE}/{W[:7]}/exp/{lp.RESULTS_ROOT_NAME}{lp.SMOKE_RESULTS_SUFFIX}/S5/{leg}/s1/shard00/attempts/ta-{leg}-{W[:4]}"


def prior_reference(plan, *, L, S, th, W, wave, leg, complete=True, endpoint_over=None, tok_s=1500.0):
    """A reference measured under a PRIOR tree / wave + the standup records its root would carry on the volume (ATTEMPT.json + endpoint.json)."""
    root = prior_root(W, leg)
    aid = root.rsplit("/", 1)[1]
    rec = lp.s5_reference_record(L=L, S=S, thinking=th, tok_s_aggregate=tok_s, duration_s=1962.0, store_complete=complete, sampling_coverage=1.0,
                                 n_eligible_intervals=197, W=W, wave_uuid=wave, attempt_id=aid, attempt_root=root) if complete else \
        {**lp.s5_reference_record(L=L, S=S, thinking=th, tok_s_aggregate=tok_s, duration_s=1962.0, store_complete=True, sampling_coverage=1.0, n_eligible_intervals=197,
                                  W=W, wave_uuid=wave, attempt_id=aid, attempt_root=root), "store_complete": False}
    row = _class_row(plan, L, S, th)
    rd = row["readiness"]
    bank = [a["object_sha256"] for a in plan["adapter_hash_set"] if a.get("object_sha256")]
    # the REAL attestation shape (endpoint.json): vLLM 0.25.1 attests FULL_AND_PIECEWISE with the FULL capture max == S, while the
    # plan row requests `FULL` -- the admission holds the reference to the readiness predicate, never to string equality on the mode
    ep = {"engine": rd["engine"], "attested_max_model_len": L, "attested_max_num_seqs": S, "thinking_mode": rd["thinking_mode"],
          "attested_cudagraph_mode": "FULL_AND_PIECEWISE", "attested_cudagraph_full_capture_max": S,
          "server_default_policy": rd["server_default_policy"], "model": plan["model"], "model_revision": plan["model_revision"],
          "lora_config": {"max_lora_rank": rd["max_lora_rank"], "max_loras": rd["max_loras"]}, "object_artifact_sha256_by_cell": {"cellA": bank[0], "cellB": bank[1]},
          "vmodels": {"none": "petri__base"}}
    ep.update(endpoint_over or {})
    texts = {f"{root}/ATTEMPT.json": json.dumps({"schema": "seedrep_attempt/1", "attempt_id": aid, "W": W, "wave_uuid": wave, "launch_plan_row": f"smoke-S5__{leg}__s1__base__shard00"}),
             f"{root}/{lp.SERVE_DIR_REL}/endpoint.json": json.dumps(ep)}
    return rec, texts


def w24_like_world(c):
    """The fixture shape over the components' plan: 7424 own (this tree + wave, complete); 11520 (pasttense) + 24576 (misalign) measured under an
    earlier wave on a matching engine; 32768 (petri) under an older wave on a DIFFERENT engine geometry (S 16, not 15); 49152 (crescendo) a partial
    store; the thinking classes and the rung classes never measured."""
    plan, ew = c["final"], c["waves"]["S5"]
    texts = dict(c["texts"])
    refs = []
    own = next(r for r in c["smokes"]["S5"]["references"] if (r["L"], r["S"]) == (7424, 70))
    refs.append(dict(own))
    for L, S, th, W, wave, leg, kw in ((11520, 45, False, W22, WAVE22, "pasttense", {}), (24576, 21, False, W22, WAVE22, "misalign", {}),
                                       (32768, 15, False, W19, WAVE19, "petri", {"endpoint_over": {"attested_max_num_seqs": 16}}),
                                       (49152, 10, False, W22, WAVE22, "crescendo", {"complete": False})):
        rec, t = prior_reference(plan, L=L, S=S, th=th, W=W, wave=wave, leg=leg, **kw)
        refs.append(rec)
        texts.update(t)
    s5 = {k: v for k, v in c["smokes"]["S5"].items() if k not in ("multiturn", "window_derivation")}
    s5["references"] = refs
    return plan, s5, lp.volume_only(lambda path: texts.get(str(path))), ew


def seed1_launch_set(plan, *, with_misalign_rung=True, with_multiturn=False):
    ids = []
    for r in lp._plan_rows_the_app_may_execute(plan):
        if r["engine"] != "s1" or r["shard"] != 0 or r.get("dose_standup") or r.get("s5_probe"):
            continue
        if r["leg"] in ("pasttense", "codeattack", "renellm", "grok420") and r["stage"] in ("base", "8192", "16384"):
            ids.append(r["standup_id"])
        elif r["leg"] == "controls" and r["stage"] in ("base", "2048", "8192", "16384"):
            ids.append(r["standup_id"])
        elif r["leg"] == "misalign" and (r["stage"] == "base" or (with_misalign_rung and r["stage"] == "16384")):
            ids.append(r["standup_id"])
        elif with_multiturn and r["leg"] == "crescendo" and r["stage"] == "base":
            ids.append(r["standup_id"])
    return sorted(ids)


def admissible_only(plan, s5, ew, rt):
    """The S5 block with ONLY the admissible references (what the record-time admission lets into RS/smokes.json): an inadmissible record left in
    `S5.references[]` (the volume moved since it was recorded) still refuses the whole certificate by name, as before."""
    cov = lp.s5_certificate_coverage(plan, s5, execution_wave=ew, launch_set=None, read_text=rt)
    return {**s5, "references": [r for i, r in enumerate(s5["references"]) if not cov["references"][str(i)]["problems"]]}


def _cert(c, s5, rt, launch_set):
    smokes = {**c["smokes"], "S5": s5}
    return lp.build_certificate(plan=c["final"], plan_path=c["plan_p"], draft=c["draft"], s0_record=c["s0"], smokes_doc=smokes, w_gate=c["w_gate"], decode_proof=c["proof"],
                                rs=c["rs"], registry_sha256=c["proof"]["registry_sha256"], read_text=rt, accepted_reads=c["reads"], launch_set=launch_set)


def own_ref(plan, ew, *, L, S, th, tok_s=1500.0, complete=True, aid="ta-own"):
    rec = lp.s5_reference_record(L=L, S=S, thinking=th, tok_s_aggregate=tok_s, duration_s=1962.0, store_complete=True, sampling_coverage=1.0, n_eligible_intervals=197,
                                 W=plan["W"], wave_uuid=ew, attempt_id=f"{aid}-{L}x{S}")
    if not complete:
        rec["store_complete"] = False
    return rec


def test_d152_dominance_is_two_dimensional():
    # the two earlier "successes" are NOT dominance (a larger window at a SMALLER width proves nothing about the wider class)
    assert lp.s5_dominates((11520, 45, False), (10240, 51, False)) is False
    assert lp.s5_dominates((24576, 21, False), (16384, 31, False)) is False
    assert lp.s5_dominates((49152, 10, False), (32768, 15, False)) is False
    # equal-L / larger-S, larger-L / equal-S, larger both: dominance; the same thinking flag is required; an exact class match is not dominance
    assert lp.s5_dominates((32768, 20, False), (32768, 15, False)) and lp.s5_dominates((40960, 15, False), (32768, 15, False)) and lp.s5_dominates((49152, 16, False), (32768, 15, False))
    assert lp.s5_dominates((49152, 16, True), (32768, 15, False)) is False and lp.s5_dominates((32768, 15, False), (32768, 15, False)) is False
    assert lp.s5_dominates((24832, 31, True), (16640, 31, True)) and lp.s5_dominates((24832, 21, True), (16640, 31, True)) is False


def test_the_seed1_launch_set_certifies_with_own_prior_and_rung_self_measured_classes(components):
    c = components
    plan, s5, rt, ew = w24_like_world(c)
    launched = seed1_launch_set(plan)                      # the FULL seed-1 set, misalign__s1__16384 included
    classes = {(x["L"], x["S"], x["thinking"]) for x in lp.s5_launched_classes(plan, launched)}
    assert classes == {(7424, 70, False), (10240, 51, False), (11520, 45, False), (16384, 31, False), (24576, 21, False), (40960, 12, False)}
    cov = lp.s5_certificate_coverage(plan, s5, execution_wave=ew, launch_set=launched, read_text=rt)
    by = {tuple(x["class"]): x for x in cov["classes"]}
    assert by[(7424, 70, False)]["satisfied_by"] == "own_reference" and by[(7424, 70, False)]["basis"] == "own" and by[(7424, 70, False)]["measured_under_W"] == plan["W"]
    assert by[(11520, 45, False)]["satisfied_by"] == "own_reference" and by[(11520, 45, False)]["basis"] == "prior_W_or_wave_D139"
    assert by[(11520, 45, False)]["measured_under_W"] == W22 and by[(11520, 45, False)]["measured_under_wave"] == WAVE22 and "D139" in by[(11520, 45, False)]["qualification"]
    assert by[(24576, 21, False)]["satisfied_by"] == "own_reference" and by[(24576, 21, False)]["measured_under_W"] == W22
    # (10240, 51) is NOT dominated by (11520, 45) nor (16384, 31) by (24576, 21); both are used only by ladder-consumer rungs -> self-measured,
    # each naming its BASE row's class reference (the estimate's band) and the rows it covers
    # (10240: the controls 2048 rung -> base 7424; 16384: four statics' 8192 rungs -> base 11520 AND the controls 8192 rung -> base 7424; 40960: misalign 16384 -> base 24576)
    for cls, bases, n_rows in (((10240, 51, False), {(7424, 70, False)}, 1), ((16384, 31, False), {(11520, 45, False), (7424, 70, False)}, 5), ((40960, 12, False), {(24576, 21, False)}, 1)):
        e = by[cls]
        assert e["satisfied_by"] == "rung_self_measured" and e["basis"] == "D154" and e["dominated_by"] is None and e["reference"] is None and len(e["rows"]) == n_rows, e
        assert all(lp.plan_row_by_id(plan, sid)["rung_mode"] == lp.RUNG_MODE_CONSUME for sid in e["rows"])
        est = e["estimator_reference"]
        assert {tuple(v["base_class"]) for v in est.values()} == bases and set(est) == {f"{b[0]}x{b[1]}" for b in bases}
        for k, v in est.items():
            base = tuple(v["base_class"])
            assert v["reference_index"] == by[base]["reference_index"] and v["satisfied_by"] == "own_reference" and sorted(sum((x["rows"] for x in est.values()), [])) == e["rows"]
            assert f"index {by[base]['reference_index']}" in e["qualification"]
        assert "D154" in e["qualification"] and "D148" in e["qualification"]
    assert by[(40960, 12, False)]["rows"] == ["misalign__s1__16384__shard00"] and cov["multiturn"] == "none_launched"
    # the 32768 reference on a DIFFERENT engine geometry is refused by name; the partial 49152 is refused as a reference (a floor)
    adm = {tuple(a["class"]) if a["class"] else i: a for i, a in cov["references"].items()}
    assert any("attests 16" in p and "states 15" in p for p in adm[(32768, 15, False)]["problems"])
    assert any("a partial store is a floor" in p for p in adm[(49152, 10, False)]["problems"])
    # an inadmissible record left in the block refuses the certificate by name; over the admissible ones the FULL seed-1 set certifies (fc9c)
    with pytest.raises(ValueError, match=r"S5.references\[3\] \(\[32768, 15, False\]\) is not an admissible S5 reference"):
        _cert(c, s5, rt, launched)
    s5_ok = admissible_only(plan, s5, ew, rt)
    assert {(r["L"], r["S"]) for r in s5_ok["references"]} == {(7424, 70), (11520, 45), (24576, 21)}
    cert = _cert(c, s5_ok, rt, launched)
    assert cert["launch_set"] == launched and cert["s5_coverage"]["launch_set"] == launched and cert["s5_coverage"]["multiturn"] == "none_launched"
    assert {tuple(x["class"]): x["satisfied_by"] for x in cert["s5_coverage"]["classes"]} == {
        (7424, 70, False): "own_reference", (10240, 51, False): "rung_self_measured", (11520, 45, False): "own_reference", (16384, 31, False): "rung_self_measured",
        (24576, 21, False): "own_reference", (40960, 12, False): "rung_self_measured"}
    assert not cert["s5"].get("multiturn"), "no multi-turn row launched: the block carries no D50 windows and certifies (D145)"
    assert lp.go_preconditions(plan, certificate=cert, registry_sha256=c["proof"]["registry_sha256"], read_text=rt, launched_ids=launched) == []
    assert lp.go_preconditions(plan, certificate=json.loads(json.dumps(cert)), registry_sha256=c["proof"]["registry_sha256"], read_text=rt, launched_ids=launched[:3]) == []
    # a --go row OUTSIDE the certified set is refused by name (the gsm8k base row was never in the set)
    probs = lp.go_preconditions(plan, certificate=cert, registry_sha256=c["proof"]["registry_sha256"], read_text=rt, launched_ids=["gsm8k__s1__base__shard00"])
    assert any("row gsm8k__s1__base__shard00 is not in the certified launch set" in p for p in probs), probs
    # a launched class that is neither referenced, dominated nor rung-only: the gsm8k base row (16640T) -> a NAMED gap with its rows; the certificate refuses
    with_gsm8k = sorted(launched + ["gsm8k__s1__base__shard00"])
    cov2 = lp.s5_certificate_coverage(plan, s5_ok, execution_wave=ew, launch_set=with_gsm8k, read_text=rt)
    gap = next(x for x in cov2["classes"] if x.get("gap"))
    assert tuple(gap["class"]) == (16640, 31, True) and gap["rows"] == ["gsm8k__s1__base__shard00"] and "D152" in gap["why"]
    with pytest.raises(ValueError, match=r"launched window classes without an S5 reference .*\(16640, 31, True\), \['gsm8k__s1__base__shard00'\].*exclude the rows"):
        _cert(c, s5_ok, rt, with_gsm8k)
    # the recorded coverage is a DERIVED statement: an edited block, or a moved launch set, refuses
    bad = copy.deepcopy(cert)
    bad["s5_coverage"]["classes"][0]["satisfied_by"] = "dominated"
    bad["certificate_sha256"] = lp.certificate_sha256(bad)
    assert any("s5_coverage is not what the launched rows and the references derive today" in p for p in lp.go_preconditions(plan, certificate=bad, registry_sha256=c["proof"]["registry_sha256"], read_text=rt))
    bad = {**cert, "launch_set": with_gsm8k}
    bad["certificate_sha256"] = lp.certificate_sha256(bad)
    probs = lp.go_preconditions(plan, certificate=bad, registry_sha256=c["proof"]["registry_sha256"], read_text=rt)
    assert any("(16640, 31, True)" in p for p in probs) and any("s5_coverage is not what" in p for p in probs)
    # without the volume reader the prior-W references cannot be re-admitted: refused by name (never a silent pass on the recorded block)
    probs = lp.go_preconditions(plan, certificate=cert, registry_sha256=c["proof"]["registry_sha256"])
    assert any("D139 admission needs the volume reader" in p for p in probs), probs


def test_a_class_shared_by_a_rung_and_a_base_row_is_never_self_measured_and_two_d_dominance_governs_it(components):
    c = components
    plan, s5, rt, ew = w24_like_world(c)
    s5_ok = admissible_only(plan, s5, ew, rt)
    # 24576 is used by the misalign BASE row and the statics' 16384 rungs: own reference, never rung-self-measured
    launched = seed1_launch_set(plan)
    e = {tuple(x["class"]): x for x in lp.s5_certificate_coverage(plan, s5_ok, execution_wave=ew, launch_set=launched, read_text=rt)["classes"]}[(24576, 21, False)]
    assert e["satisfied_by"] == "own_reference" and sorted(lp.plan_row_by_id(plan, sid)["stage"] for sid in e["rows"]) == ["16384"] * 5 + ["base"]     # 4 statics + controls rungs, misalign base
    # the same class with its own reference REMOVED: a base row uses it, so rung self-measurement does not apply -> a named gap (nothing dominates it)
    no24 = {**s5_ok, "references": [r for r in s5_ok["references"] if (r["L"], r["S"]) != (24576, 21)]}
    e = {tuple(x["class"]): x for x in lp.s5_certificate_coverage(plan, no24, execution_wave=ew, launch_set=launched, read_text=rt)["classes"]}[(24576, 21, False)]
    assert e.get("gap") is True and "misalign__s1__base__shard00" in e["rows"] and "D152" in e["why"]
    # the 40960 rung's base class is now unreferenced too: the rung has no estimator band -> a named gap, not a silent self-measurement
    e40 = {tuple(x["class"]): x for x in lp.s5_certificate_coverage(plan, no24, execution_wave=ew, launch_set=launched, read_text=rt)["classes"]}[(40960, 12, False)]
    assert e40.get("gap") is True and "base class (24576, 21, False) has no complete admissible reference" in e40["why"]
    # two-dimensional dominance on a BASE-row class: petri (32768, 15) is dominated by a complete own-W reference with L >= and S >= (three shapes), never by a
    # larger window at a smaller width, never by a partial store, never across the thinking flag
    petri = [r["standup_id"] for r in lp._plan_rows_the_app_may_execute(plan) if r["leg"] == "petri" and r["stage"] == "base" and r["engine"] == "s1"]
    for ref_cls, dominated in (((49152, 16, False), True), ((32768, 20, False), True), ((40960, 15, False), True), ((49152, 10, False), False), ((49152, 16, True), False)):
        block = {**s5_ok, "references": list(s5_ok["references"]) + [own_ref(plan, ew, L=ref_cls[0], S=ref_cls[1], th=ref_cls[2])]}
        e = {tuple(x["class"]): x for x in lp.s5_certificate_coverage(plan, block, execution_wave=ew, launch_set=petri, read_text=rt)["classes"]}[(32768, 15, False)]
        assert (e.get("satisfied_by") == "dominated") is dominated and (e.get("dominated_by") == list(ref_cls)) is dominated, (ref_cls, e)
        if dominated:
            assert "D152" in e["qualification"] and e["rows"] == petri
    block = {**s5_ok, "references": list(s5_ok["references"]) + [own_ref(plan, ew, L=49152, S=16, th=False, complete=False)]}
    cov = lp.s5_certificate_coverage(plan, block, execution_wave=ew, launch_set=petri, read_text=rt)
    assert next(x for x in cov["classes"] if tuple(x["class"]) == (32768, 15, False)).get("gap") is True, "a partial store never dominates"


def test_the_fanout_intent_and_every_spawn_boundary_hold_the_certified_launch_set(components):
    """A correctly certificate-bound intent carrying an excluded but valid plan row is refused by name at fan-out; the same guard runs at the
    launcher's selection, each fan-out spawn (fresh and recovered) and the re-attempt policy's spawn."""
    import importlib.util
    import types
    c = components
    plan, s5, rt, ew = w24_like_world(c)
    s5_ok = admissible_only(plan, s5, ew, rt)
    launched = seed1_launch_set(plan)
    cert = _cert(c, s5_ok, rt, launched)
    rows = [lp.plan_row_by_id(plan, sid) for sid in launched]
    intent = lp.production_intent_new(plan=plan, certificate=cert, rows=rows, selection={"go": "", "legs": "seed1"})
    assert lp.production_intent_problems(intent, plan=plan, certificate=cert) == []
    extra = lp.plan_row_by_id(plan, "gsm8k__s1__base__shard00")                          # a VALID plan row the certificate never covered
    bad = json.loads(json.dumps(intent))
    bad["rows"][extra["standup_id"]] = json.loads(json.dumps(intent["rows"][launched[0]]))
    bad["rows"][extra["standup_id"]].update(leg=extra["leg"], engine=extra["engine"], stage=extra["stage"], shard=extra["shard"], kind=extra["kind"],
                                            control_key=extra["control_key"], accepted_key=extra["accepted_key"])
    probs = lp.production_intent_problems(bad, plan=plan, certificate=cert)
    assert any("row gsm8k__s1__base__shard00 is not in the certified launch set" in p for p in probs), probs
    assert lp.launched_ids_problems(cert, launched) == [] and lp.launched_ids_problems({**cert, "launch_set": None}, launched + ["gsm8k__s1__base__shard00"]) == []
    # the app: one guard, called at every spawn boundary
    spec = importlib.util.spec_from_file_location("seedrep_served_fc9b", REPO / "scripts/modal/seedrep_served.py")
    srv = importlib.util.module_from_spec(spec)
    from test_seedrep_review_fixes import _stub_modal
    prior = sys.modules.get("modal")
    sys.modules["modal"] = _stub_modal()
    try:
        spec.loader.exec_module(srv)
    finally:
        if prior is None:
            sys.modules.pop("modal", None)
        else:
            sys.modules["modal"] = prior
    with pytest.raises(srv.LaunchRefused, match=r"fan-out REFUSED \(D146; the certified launch set\): row gsm8k__s1__base__shard00 is not in the certified launch set"):
        srv.certified_launch_guard(cert, list(bad["rows"]), where="fan-out")
    srv.certified_launch_guard(cert, launched, where="fan-out")
    src = (REPO / "scripts/modal/seedrep_served.py").read_text()
    fan = src.split("def run_fanout(", 1)[1].split("\ndef ", 1)[0]
    assert fan.count("certified_launch_guard(cert, ") == 3 and fan.index("certified_launch_guard(cert, list(intent[\"rows\"])") < fan.index("due = lp.production_intent_unspawned")
    for site in ("re-spawn", "spawn"):
        i = fan.index(f'where="fan-out {site}"')
        assert fan.index("spawn(row, plan_d, app_name, rs, pools, fanout_claim=", i) > i, site
    lp_body = src.split("def launch_production(", 1)[1].split("\ndef ", 1)[0]
    assert 'certified_launch_guard(cert, [r["standup_id"] for r in picked], where="production launch")' in lp_body
    rp = src.split("def reattempt_pass(", 1)[1].split("\ndef ", 1)[0]
    assert 'certified_launch_guard(certificate, [row["standup_id"]], where="re-attempt spawn")' in rp and "spawn=guarded_spawn" in rp
    assert "reattempt_pass(plan_d, rs_p, run_name, pools, certificate=cert)" in src


def test_the_whole_roster_names_every_gap_and_the_old_full_coverage_world_still_certifies(components):
    c = components
    plan, s5, rt, ew = w24_like_world(c)
    cov = lp.s5_certificate_coverage(plan, s5, execution_wave=ew, launch_set=None, read_text=rt)
    gaps = {tuple(x["class"]) for x in cov["classes"] if x.get("gap")}
    # 10240 / 16384 / 40960 are ladder-consumer rung classes (self-measured); 24832T is gsm8k's inspect-subset rung (not a consumer) and 49152 is
    # shared with the multi-turn base rows -> gaps; nothing is `dominated` on this shape
    assert gaps == {(16640, 31, True), (24832, 21, True), (32768, 15, False), (49152, 10, False)}, gaps
    assert {tuple(x["class"]) for x in cov["classes"] if x.get("satisfied_by") == "rung_self_measured"} == {(10240, 51, False), (16384, 31, False), (40960, 12, False)}
    assert not any(x.get("satisfied_by") == "dominated" for x in cov["classes"])
    with pytest.raises(ValueError, match="launched window classes without an S5 reference"):
        _cert(c, admissible_only(plan, s5, ew, rt), rt, None)
    assert lp.s5_gap(plan, [r for i, r in enumerate(s5["references"]) if not cov["references"][str(i)]["problems"]]) and len(lp.s5_gap(plan, [])) == len(plan["s5_roster"])
    # the components' own world (every class measured under THIS W + wave, the windows present): launch_set None certifies exactly as before, every class `own`
    cert = _cert(c, c["smokes"]["S5"], c["read_text"], None)
    assert cert["launch_set"] is None and all(x["satisfied_by"] == "own_reference" and x["basis"] == "own" for x in cert["s5_coverage"]["classes"])
    assert len(cert["s5_coverage"]["classes"]) == len(plan["s5_roster"]) and isinstance(cert["s5_coverage"]["multiturn"], dict)
    assert lp.go_preconditions(plan, certificate=cert, registry_sha256=c["proof"]["registry_sha256"]) == []


def test_multiturn_windows_may_be_empty_only_when_no_multiturn_row_is_launched(components):
    c = components
    plan, s5, rt, ew = w24_like_world(c)
    # a crescendo row in the launch set: the windows are required again, and the 49152 class (partial only) is a gap
    launched = seed1_launch_set(plan, with_multiturn=True)
    s5_ok = admissible_only(plan, s5, ew, rt)
    cov = lp.s5_certificate_coverage(plan, s5_ok, execution_wave=ew, launch_set=launched, read_text=rt)
    assert cov["multiturn"] == {"launched_legs": ["crescendo"]} and any(tuple(x["class"]) == (49152, 10, False) and x.get("gap") for x in cov["classes"])
    with pytest.raises(ValueError) as ei:
        _cert(c, s5_ok, rt, launched)
    msg = str(ei.value)
    assert "no D50 multi-turn window measurements (S5.multiturn[]) while multi-turn row(s) of ['crescendo'] are in the launch set" in msg and "(49152, 10, False)" in msg
    # the components' windows present + a complete 49152 reference under a prior wave -> the crescendo row certifies; the probe-leg window rule applies to the launched leg only
    rec, texts = prior_reference(plan, L=49152, S=10, th=False, W=W22, wave=WAVE22, leg="crescendo")
    s5b = dict(s5_ok)
    s5b["references"] = list(s5_ok["references"]) + [rec]
    s5b["multiturn"] = c["smokes"]["S5"]["multiturn"]
    s5b["window_derivation"] = c["smokes"]["S5"]["window_derivation"]
    rt2 = lp.volume_only(lambda path: texts.get(str(path)) or rt(path))
    cert = _cert(c, s5b, rt2, launched)
    assert cert["s5_coverage"]["multiturn"] == {"launched_legs": ["crescendo"]}
    assert next(x for x in cert["s5_coverage"]["classes"] if tuple(x["class"]) == (49152, 10, False))["basis"] == "prior_W_or_wave_D139"
    assert lp.go_preconditions(plan, certificate=cert, registry_sha256=c["proof"]["registry_sha256"], read_text=rt2, launched_ids=launched) == []


def test_reference_admission_refusals_by_name(components):
    c = components
    plan, ew = c["final"], c["waves"]["S5"]
    rec, texts = prior_reference(plan, L=11520, S=45, th=False, W=W22, wave=WAVE22, leg="pasttense")
    rt = lp.volume_only(lambda path: texts.get(str(path)))
    ok = lp.s5_reference_admission(rec, plan=plan, execution_wave=ew, read_text=rt)
    assert ok["problems"] == [] and ok["basis"] == "prior_W_or_wave_D139" and ok["identity"]["objects_in_plan_adapter_bank"]["foreign"] == []
    # no attempt_root -> cannot be located; no reader -> cannot be re-read; a root whose ATTEMPT.json is another attempt's; a foreign object; a different model revision
    bare = {k: v for k, v in rec.items() if k != "attempt_root"}
    probs = lp.s5_reference_admission(bare, plan=plan, execution_wave=ew, read_text=rt)["problems"]
    assert any("is not the standup's tree" in p for p in probs) and any("is not the standup's wave" in p for p in probs) and any("D139" in p and "attempt_root" in p for p in probs), probs
    assert any("needs the volume reader" in p for p in lp.s5_reference_admission(rec, plan=plan, execution_wave=ew, read_text=None)["problems"])
    root = rec["attempt_root"]
    t2 = dict(texts); t2[f"{root}/ATTEMPT.json"] = json.dumps({**json.loads(texts[f"{root}/ATTEMPT.json"]), "attempt_id": "ta-other"})
    assert any("ATTEMPT.json attempt_id 'ta-other' is not the record's" in p for p in lp.s5_reference_admission(rec, plan=plan, execution_wave=ew, read_text=lp.volume_only(lambda p_: t2.get(str(p_))))["problems"])
    ep = json.loads(texts[f"{root}/{lp.SERVE_DIR_REL}/endpoint.json"])
    t3 = dict(texts); t3[f"{root}/{lp.SERVE_DIR_REL}/endpoint.json"] = json.dumps({**ep, "object_artifact_sha256_by_cell": {"cellZ": "f" * 64}})
    assert any("served object(s) ['cellZ'] whose bytes are not in this plan's adapter bank" in p for p in lp.s5_reference_admission(rec, plan=plan, execution_wave=ew, read_text=lp.volume_only(lambda p_: t3.get(str(p_))))["problems"])
    t4 = dict(texts); t4[f"{root}/{lp.SERVE_DIR_REL}/endpoint.json"] = json.dumps({**ep, "model_revision": "0" * 40})
    assert any("model_revision: the reference's standup attests" in p for p in lp.s5_reference_admission(rec, plan=plan, execution_wave=ew, read_text=lp.volume_only(lambda p_: t4.get(str(p_))))["problems"])
    t5 = dict(texts); t5[f"{root}/{lp.SERVE_DIR_REL}/endpoint.json"] = json.dumps({**ep, "thinking_mode": "on"})
    assert any("thinking_mode:" in p for p in lp.s5_reference_admission(rec, plan=plan, execution_wave=ew, read_text=lp.volume_only(lambda p_: t5.get(str(p_))))["problems"])
    # the root gone from the volume
    assert any("not on the volume" in p for p in lp.s5_reference_admission(rec, plan=plan, execution_wave=ew, read_text=lp.volume_only(lambda p_: None))["problems"])
    # an attempt_root that does not end in the attempt id is not a reference at all (schema)
    assert any("not a `/vol/...` root ending in the record's attempt id" in p for p in lp.s5_reference_problems({**rec, "attempt_root": "/vol/x/other"}))
    # record time: the CLI's merge admits the prior reference through the same rule, and refuses it without the plan (the pre-fc9 rule)
    doc = {"S5": {"W": plan["W"], "execution_wave_uuid": ew, "production_wave_uuid": plan["wave_uuid"], "launch_plan_sha": plan["launch_plan_sha"]}}
    lp.s5_smokes_add_reference(doc, rec, W=plan["W"], wave_uuid=ew, plan=plan, read_text=rt)
    assert doc["S5"]["references"] == [rec]
    with pytest.raises(ValueError, match="is not the standup's tree"):
        lp.s5_smokes_add_reference({"S5": {}}, rec, W=plan["W"], wave_uuid=ew)
    with pytest.raises(ValueError, match="attests 16"):
        bad, tb = prior_reference(plan, L=32768, S=15, th=False, W=W19, wave=WAVE19, leg="petri", endpoint_over={"attested_max_num_seqs": 16})
        lp.s5_smokes_add_reference({"S5": {}}, bad, W=plan["W"], wave_uuid=ew, plan=plan, read_text=lp.volume_only(lambda p_: tb.get(str(p_))))
    # the launch set is validated: unknown rows, duplicates, an empty list
    assert lp.launch_set_problems(plan, ["nope__s1__base__shard00"]) and lp.launch_set_problems(plan, []) and lp.launch_set_problems(plan, None) == []
    sid = _row(plan, "pasttense")["standup_id"]
    assert lp.launch_set_problems(plan, [sid, sid]) == ["launch_set names a row twice"] and lp.launch_set_problems(plan, [sid]) == []
    import seedrep_s5_record as S5R
    assert S5R.parse_launch_set(None) is None and S5R.parse_launch_set(" b , a,,") == ["a", "b"]
    ns = S5R.build_parser().parse_args(["certify", "--rs", "r", "--plan", "p", "--draft", "d", "--tree", "t", "--launch-set", "b,a"])
    assert ns.launch_set == "b,a" and S5R.parse_launch_set(ns.launch_set) == ["a", "b"]
    f = Path(str(components["rs"])) / "launch_set.txt"
    f.write_text("x__s1__base__shard00\ny__s1__base__shard00\n")
    assert S5R.parse_launch_set(f"@{f}") == ["x__s1__base__shard00", "y__s1__base__shard00"]


def test_a_rung_row_with_its_kind_flipped_is_refused_at_plan_validation_and_never_self_measured(components):
    """On the generated plan, flipping pasttense__s1__8192__shard00.kind from "rung" to "base" (digest recomputed) used to pass the
    PLAN gate and read `rung_self_measured` at G8 / certify while the container's `kind` dispatch ran the row's children as ordinary generation at the
    higher cap. Now ONE rung definition is shared: `validate_plan` refuses the contradictory row by name (the PLAN gate with it), the coverage's
    ladder-consumer predicate excludes it (the class becomes a named gap, never self-measured), and the container dispatches on the same predicate."""
    import seedrep_s0_gates as S0
    c = components
    plan, s5, rt, ew = w24_like_world(c)
    s5_ok = admissible_only(plan, s5, ew, rt)
    assert not any("a rungs row" in p or "a standups row" in p for p in lp.validate_plan(plan)), "the generated plan's rows all agree with their collection"
    assert all(lp.row_shape_problems(r, "standups") == [] for r in plan["standups"]) and all(lp.row_shape_problems(r, "rungs") == [] for r in plan["rungs"])
    for sid in ("pasttense__s1__8192__shard00", "codeattack__s1__8192__shard00", "misalign__s1__16384__shard00"):
        row = lp.plan_row_by_id(plan, sid)
        assert lp.is_rung_row(row) and lp.is_ladder_consumer_row(row) and lp.row_shape_problems(row, "rungs") == []
        flipped = json.loads(json.dumps(plan))
        r = next(x for x in flipped["rungs"] if x["standup_id"] == sid)
        r["kind"] = "base"
        flipped["launch_plan_sha"] = lp.plan_sha(flipped)
        assert lp.plan_sha(flipped) == flipped["launch_plan_sha"] and lp.plan_row_by_id(flipped, sid)["kind"] == "base"
        # the shared predicate: not a rung, not a ladder consumer -- the container would not dispatch it as one either
        assert lp.is_rung_row(r) is False and lp.is_ladder_consumer_row(r) is False
        probs = lp.row_shape_problems(r, "rungs")
        assert any(f"{sid}: a rungs row with kind 'base'" in p and "ordinary generation" in p for p in probs), probs
        # PLAN validation refuses by name; the S0 PLAN gate with it
        vp = lp.validate_plan(flipped)
        assert any(sid in p and "kind 'base'" in p for p in vp), vp
        g = S0.gate_plan_validates(plan=flipped, tree=REPO)
        assert g["ok"] is False and "validate_plan problem" in g["why"]
        # the coverage never self-measures a class through a row that is not a ladder consumer: the flipped row's class is a NAMED gap
        cov = lp.s5_certificate_coverage(flipped, s5_ok, execution_wave=ew, launch_set=seed1_launch_set(flipped), read_text=rt)
        cls = tuple(lp.s5_window_class({"L": r["geometry"]["L"], "S": r["geometry"]["S"], "thinking": r["geometry"]["thinking"]}))
        e = next(x for x in cov["classes"] if tuple(x["class"]) == cls)
        assert e.get("gap") is True and sid in e["rows"] and "D152" in e["why"] and f"not ladder consumers: ['{sid}']" in e["why"], e
    # the other shape contradictions are named too: a consume rung without its consumer binding; a base row carrying one / a rung_mode / a non-base stage
    r = json.loads(json.dumps(lp.plan_row_by_id(plan, "pasttense__s1__8192__shard00")))
    r["rung_consumer"] = None
    assert any("without a consumer binding" in p for p in lp.row_shape_problems(r, "rungs"))
    r["rung_mode"] = "whatever"
    assert any("rung_mode 'whatever'" in p for p in lp.row_shape_problems(r, "rungs")) and lp.is_rung_row(r) is False
    b = json.loads(json.dumps(lp.plan_row_by_id(plan, "pasttense__s1__base__shard00")))
    assert lp.row_shape_problems(b, "standups") == [] and lp.is_rung_row(b) is False
    b2 = {**b, "rung_mode": lp.RUNG_MODE_CONSUME, "rung_consumer": list(lp.plan_row_by_id(plan, "pasttense__s1__8192__shard00")["rung_consumer"])}
    probs = lp.row_shape_problems(b2, "standups")
    assert any("rung_mode 'consume'" in p for p in probs) and any("consumer binding" in p for p in probs)
    assert lp.row_shape_problems({**b, "stage": "8192"}, "standups") and lp.row_shape_problems(b, "rungs")
    # the gsm8k subset rung and the petri rerun are rungs (dispatched as such) but NOT ladder consumers -- never self-measured
    for leg in ("gsm8k", "petri"):
        rr = next(x for x in plan["rungs"] if x["leg"] == leg)
        assert lp.is_rung_row(rr) and not lp.is_ladder_consumer_row(rr) and lp.row_shape_problems(rr, "rungs") == []
    # the container dispatches on the SAME definition
    src = (REPO / "scripts/modal/seedrep_served.py").read_text()
    assert "is_rung = lp.is_rung_row(row) and not is_petri" in src and 'is_rung = row["kind"] == "rung"' not in src
    assert "if not lp.is_ladder_consumer_row(row):" in src and "RUNG_ROW_SHAPE_REFUSED" in src
    # a consume rung whose consumer argv is present but kind flipped is ALSO not a rung to the container -- it never reaches rung_consumer_child
    assert lp.is_rung_row({**lp.plan_row_by_id(plan, "pasttense__s1__8192__shard00"), "kind": "static"}) is False


def test_fc9e_reference_admission_holds_the_graph_mode_to_the_readiness_predicate_not_string_equality(components):
    """At a3528d4b1, every S5 reference (the 7424, 11520 and 24576 classes) was refused because the plan row's readiness states `FULL` while
    every engine attests `FULL_AND_PIECEWISE` (FULL capture max == S), which the readiness contract already accepts. The admission now compares with
    the ONE predicate (`lp.cudagraph_full_capture_problems`, the list `lp.CUDAGRAPH_MODES_WITH_FULL_CAPTURE` re-exported by seedrep_served), records
    both values, and refuses only a mode without a FULL surface, an absent capture max, or a capture max below S -- by name."""
    import importlib.util
    import seedrep_s0_gates as S0
    c = components
    plan, ew = c["final"], c["waves"]["S5"]
    spec = importlib.util.spec_from_file_location("seedrep_served_fc9e", REPO / "scripts/modal/seedrep_served.py")
    srv = importlib.util.module_from_spec(spec)
    from test_seedrep_review_fixes import _stub_modal
    prior = sys.modules.get("modal")
    sys.modules["modal"] = _stub_modal()
    try:
        spec.loader.exec_module(srv)
    finally:
        if prior is not None:
            sys.modules["modal"] = prior
        else:
            sys.modules.pop("modal", None)
    assert srv.CUDAGRAPH_MODES_WITH_FULL_CAPTURE is lp.CUDAGRAPH_MODES_WITH_FULL_CAPTURE and lp.CUDAGRAPH_MODES_WITH_FULL_CAPTURE == ("FULL", "FULL_AND_PIECEWISE")
    assert "cudagraph_mode" not in lp.S5_IDENTITY_ENDPOINT_OF_READINESS            # no longer a string-equality field
    row = _class_row(plan, 7424, 70, False)
    assert row["readiness"]["cudagraph_mode"] == "FULL" and int(row["geometry"]["S"]) == 70
    cases = (({"attested_cudagraph_mode": "FULL_AND_PIECEWISE", "attested_cudagraph_full_capture_max": 70}, None),          # the controls attestation
             ({"attested_cudagraph_mode": "FULL", "attested_cudagraph_full_capture_max": 70}, None),
             ({"attested_cudagraph_mode": "PIECEWISE", "attested_cudagraph_full_capture_max": 70}, "attested_cudagraph_mode 'PIECEWISE' has no FULL capture surface"),
             ({"attested_cudagraph_mode": "FULL_AND_PIECEWISE", "attested_cudagraph_full_capture_max": 60}, "cudagraph FULL capture max 60 < S 70"),
             ({"attested_cudagraph_mode": "FULL_AND_PIECEWISE", "attested_cudagraph_full_capture_max": None}, "attested_cudagraph_full_capture_max ABSENT"))
    for over, needle in cases:
        rec, texts = prior_reference(plan, L=7424, S=70, th=False, W=W22, wave=WAVE22, leg="controls", endpoint_over=over)
        adm = lp.s5_reference_admission(rec, plan=plan, execution_wave=ew, read_text=texts.get)
        cg = adm["identity"]["cudagraph_mode"]
        assert cg["plan"] == "FULL" and cg["reference"] == over["attested_cudagraph_mode"] and cg["reference_full_capture_max"] == over["attested_cudagraph_full_capture_max"] and cg["S"] == 70
        if needle is None:
            assert adm["problems"] == [] and adm["basis"] == lp.S5_REFERENCE_BASIS_PRIOR and cg["problems"] == [], adm
            # certify-time admission is the SAME helper (s5_certificate_coverage -> s5_reference_admission)
            cov = lp.s5_certificate_coverage(plan, {"references": [rec]}, execution_wave=ew, launch_set=None, read_text=texts.get)
            assert cov["references"]["0"]["problems"] == []
        else:
            assert len(adm["problems"]) == 1 and f"cudagraph_mode: {needle}" in adm["problems"][0] and "D139" in adm["problems"][0], adm["problems"]
            assert any(needle in x for x in cg["problems"])
    # the readiness contract's own strings are the predicate's (one definition, shared)
    assert lp.cudagraph_full_capture_problems(attested_mode="FULL_AND_PIECEWISE", capture_max=70, S=70) == []
    assert lp.cudagraph_full_capture_problems(attested_mode=None, capture_max=None, S=70) == [
        "attested_cudagraph_mode None has no FULL capture surface (accepted: ['FULL', 'FULL_AND_PIECEWISE'])",
        "attested_cudagraph_full_capture_max ABSENT (a mandated attestation: the FULL capture surface must cover S 70)"]
    src = (REPO / "scripts/modal/seedrep_served.py").read_text()
    assert "lp.cudagraph_full_capture_problems(attested_mode=endpoint.get" in src and src.count('("FULL", "FULL_AND_PIECEWISE")') == 0
    # the served-object clause (fc9f / fc9g): the bank is the UNION of the rows' object pins (objects + adapters, agreeing per cell),
    # never read off the summary alone; no plan-shape carve-out
    bank_objs = [a["object_sha256"] for a in plan["adapter_hash_set"] if a.get("object_sha256")]
    info = lp.plan_adapter_bank(plan)
    assert len(bank_objs) >= 2 and info["bank"] == set(bank_objs) and info["path"] == "summary" and info["problems"] == [] and info["n_objects"] == len(bank_objs)
    assert info["sources"]["objects"] > 0 and info["sources"]["adapters"] > 0 and info["n_rows"] == len(plan["standups"]) + len(plan["rungs"])
    foreign_rec, foreign_texts = prior_reference(plan, L=7424, S=70, th=False, W=W22, wave=WAVE22, leg="controls",
                                                 endpoint_over={"object_artifact_sha256_by_cell": {"cellZ": "f" * 64}})
    clean_rec, clean_texts = prior_reference(plan, L=7424, S=70, th=False, W=W22, wave=WAVE22, leg="controls")
    adm = lp.s5_reference_admission(foreign_rec, plan=plan, execution_wave=ew, read_text=foreign_texts.get)
    assert adm["problems"] and "served object(s) ['cellZ'] whose bytes are not in this plan's adapter bank" in adm["problems"][0]
    assert adm["identity"]["objects_in_plan_adapter_bank"] == {"served": ["cellZ"], "foreign": ["cellZ"], "bank_size": len(bank_objs), "path": "summary",
                                                               "sources": info["sources"], "problems": []}
    adm = lp.s5_reference_admission(clean_rec, plan=plan, execution_wave=ew, read_text=clean_texts.get)
    assert adm["problems"] == [] and adm["identity"]["objects_in_plan_adapter_bank"]["path"] == "summary" and "bank summary" in adm["qualification"]
    # the summary cleared / removed (digest recomputed) -- foreign bytes are STILL refused, through the rows-derived bank
    for nobank in ({**plan, "adapter_hash_set": []}, {k: v for k, v in plan.items() if k != "adapter_hash_set"}):
        nobank = {**nobank, "launch_plan_sha": lp.plan_sha(nobank)}
        adm = lp.s5_reference_admission(foreign_rec, plan=nobank, execution_wave=ew, read_text=foreign_texts.get)
        assert adm["problems"] and "served object(s) ['cellZ'] whose bytes are not in this plan's adapter bank" in adm["problems"][0], adm
        b = adm["identity"]["objects_in_plan_adapter_bank"]
        assert b["foreign"] == ["cellZ"] and b["bank_size"] == len(bank_objs) and b["path"] == "derived_from_rows" and b["problems"] == [], b
        adm = lp.s5_reference_admission(clean_rec, plan=nobank, execution_wave=ew, read_text=clean_texts.get)
        assert adm["problems"] == [] and adm["identity"]["objects_in_plan_adapter_bank"]["path"] == "derived_from_rows" and "bank derived_from_rows" in adm["qualification"]
    # a summary inconsistent with the rows is refused by object: one omitted; one no row pins (a smuggled entry does not widen the bank either)
    short = {**plan, "adapter_hash_set": [a for a in plan["adapter_hash_set"] if a.get("object_sha256") != bank_objs[0]]}
    adm = lp.s5_reference_admission(clean_rec, plan=short, execution_wave=ew, read_text=clean_texts.get)
    assert len(adm["problems"]) == 1 and f"adapter_hash_set omits 1 object(s) its rows pin (['{bank_objs[0][:12]}']) -- the summary is inconsistent with the rows" in adm["problems"][0], adm
    wide = {**plan, "adapter_hash_set": [*plan["adapter_hash_set"], {"staged_id": "smuggled", "object_sha256": "f" * 64, "intervention_hash": None, "cells": ["cellZ"], "standups": 0}]}
    adm = lp.s5_reference_admission(foreign_rec, plan=wide, execution_wave=ew, read_text=foreign_texts.get)
    assert any("adapter_hash_set names 1 object(s) no planned row pins (['ffffffffffff'])" in x for x in adm["problems"]) \
        and "served object(s) ['cellZ'] whose bytes are not in this plan's adapter bank" in adm["problems"][0], adm
    # the object-bearing adapters entries stripped from every row, the summary regenerated, objects / serving argv intact --
    # the two representations DISAGREE by cell: refused at admission and by validate_plan (the PLAN gate); the earlier check read this as a base-only plan
    stripped = json.loads(json.dumps(plan))
    for r in [*stripped["standups"], *stripped["rungs"]]:
        r["adapters"] = {cell: a for cell, a in r["adapters"].items() if not a.get("object_sha256")}
    stripped["adapter_hash_set"] = lp.adapter_hash_set([*stripped["standups"], *stripped["rungs"]])
    stripped["launch_plan_sha"] = lp.plan_sha(stripped)
    ctl = lp.plan_row_by_id(stripped, "controls__s1__base__shard00")
    pinned_cells = sorted(c for c, o in ctl["objects"].items() if o.get("object_sha256"))
    assert pinned_cells and any("--expect-object-sha256" in x for x in ctl["argv"])
    info = lp.plan_adapter_bank(stripped)
    assert info["problems"] and info["bank"] == set(bank_objs) and info["sources"]["adapters"] == 0 and info["sources"]["objects"] > 0
    assert any(f"controls__s1__base__shard00: cell {pinned_cells[0]!r} object pin disagrees between objects (" in x and "adapters (absent)" in x for x in info["problems"]), info["problems"][:3]
    assert any("adapter_hash_set omits" in x for x in info["problems"])
    adm = lp.s5_reference_admission(foreign_rec, plan=stripped, execution_wave=ew, read_text=foreign_texts.get)
    assert adm["problems"] and "object pin disagrees between objects" in adm["problems"][0] and "served object(s) ['cellZ'] whose bytes are not in this plan's adapter bank" in adm["problems"][0]
    assert adm["identity"]["objects_in_plan_adapter_bank"]["problems"] == info["problems"] and "exempt" not in adm["identity"]["objects_in_plan_adapter_bank"]
    adm = lp.s5_reference_admission(clean_rec, plan=stripped, execution_wave=ew, read_text=clean_texts.get)
    assert adm["problems"] and "object pin disagrees" in adm["problems"][0]          # even a reference serving only banked bytes: the plan itself is inconsistent
    vp = lp.validate_plan(stripped)
    assert any("controls__s1__base__shard00: cell" in x and "object pin disagrees" in x for x in vp), vp[:3]
    assert S0.gate_plan_validates(plan=stripped, tree=REPO)["ok"] is False
    assert not any("object pin disagrees" in x or "adapter_hash_set" in x for x in lp.validate_plan(plan))
    # the mirror image: an objects pin changed under an agreeing adapters entry is the same disagreement, by cell and hash
    flipped = json.loads(json.dumps(plan))
    r = lp.plan_row_by_id(flipped, "controls__s1__base__shard00")
    cell = pinned_cells[0]
    r["objects"][cell]["object_sha256"] = "e" * 64
    assert any(f"cell {cell!r} object pin disagrees between objects (eeeeeeeeeeee) and adapters ({r['adapters'][cell]['object_sha256'][:12]})" in x
               for x in lp.plan_adapter_bank(flipped)["problems"])
    # rows that truly pin no object in EITHER representation: bank_size 0 and no carve-out text -- a reference serving nothing is admitted; one
    # serving any object is refused as foreign
    bare = json.loads(json.dumps(plan))
    for r in [*bare["standups"], *bare["rungs"]]:
        r["adapters"] = {cell: a for cell, a in r["adapters"].items() if not a.get("object_sha256")}
        r["objects"] = {cell: o for cell, o in r["objects"].items() if not o.get("object_sha256")}
    bare["adapter_hash_set"] = lp.adapter_hash_set([*bare["standups"], *bare["rungs"]])
    info = lp.plan_adapter_bank(bare)
    assert info["problems"] == [] and info["n_objects"] == 0 and info["bank"] == set() and info["sources"] == {"objects": 0, "adapters": 0}
    none_rec, none_texts = prior_reference(plan, L=7424, S=70, th=False, W=W22, wave=WAVE22, leg="controls", endpoint_over={"object_artifact_sha256_by_cell": {}})
    adm = lp.s5_reference_admission(none_rec, plan=bare, execution_wave=ew, read_text=none_texts.get)
    assert adm["problems"] == [], adm
    b = adm["identity"]["objects_in_plan_adapter_bank"]
    assert b["bank_size"] == 0 and b["served"] == [] and b["foreign"] == [] and b["problems"] == [] and "exempt" not in b and "exempt" not in adm["qualification"]
    adm = lp.s5_reference_admission(foreign_rec, plan=bare, execution_wave=ew, read_text=foreign_texts.get)
    assert len(adm["problems"]) == 1 and "served object(s) ['cellZ'] whose bytes are not in this plan's adapter bank (0 object(s) pinned by its rows)" in adm["problems"][0]


def _without_flag(argv, flag):
    out, skip = [], False
    for x in argv:
        if skip:
            skip = False
            continue
        if x == flag:
            skip = True
            continue
        out.append(x)
    return out


def test_fc9e_the_rung_mode_is_a_function_of_the_leg_and_a_consume_rung_names_a_known_kind(components):
    """Two holes: (a) gsm8k__s1__16384__shard00 relabelled `consume` with a REAL consumer argv read as a ladder consumer (the class would have
    self-measured); (b) the Petri rerun row relabelled `consume` with an argv carrying no --kind passed the argv check (`None == None`, since
    LADDER_KIND_OF_LEG has no petri). Now LADDER_MODE_OF_LEG is the one map: (leg, mode) pairs outside it are refused by name in row_shape_problems /
    is_rung_row, and a ladder consumer needs a KNOWN kind named by --kind."""
    import seedrep_s0_gates as S0
    c = components
    plan, s5, rt, ew = w24_like_world(c)
    s5_ok = admissible_only(plan, s5, ew, rt)
    assert lp.LADDER_MODE_OF_LEG["gsm8k"] == lp.RUNG_MODE_GSM8K_SUBSET and lp.LADDER_MODE_OF_LEG["petri"] == lp.RUNG_MODE_PETRI_RERUN
    assert all(lp.LADDER_MODE_OF_LEG[leg] == lp.RUNG_MODE_CONSUME for leg in (*lp.STATIC_LEGS, "controls", "misalign"))
    assert all(lp.is_rung_row(r) for r in plan["rungs"]) and all(lp.row_shape_problems(r, "rungs") == [] for r in plan["rungs"])
    # (a) the gsm8k rung relabelled consume with a REAL consumer argv (rung_consumer_argv for leg gsm8k)
    g = lp.plan_row_by_id(plan, "gsm8k__s1__16384__shard00")
    assert g["rung_mode"] == lp.RUNG_MODE_GSM8K_SUBSET and g["rung_consumer"] is None
    argv = lp.rung_consumer_argv(leg="gsm8k", engine=g["engine"], stage=int(g["stage"]), shard=g["shard"], feeder_shards=[g["shard"]], rung_shards=[g["shard"]],
                                 next_cap=None, W=plan["W"], W7=plan["W7"], wave_uuid=plan["wave_uuid"], timeout_s=int(g["modal"]["timeout_s"]), width=int(g["geometry"]["S"]))
    assert lp._flag_value(argv, "--kind") == "gsm8k" and argv[2:3] == ["consume"]
    flipped = json.loads(json.dumps(plan))
    r = next(x for x in flipped["rungs"] if x["standup_id"] == g["standup_id"])
    r["rung_mode"], r["rung_consumer"] = lp.RUNG_MODE_CONSUME, list(argv)
    flipped["launch_plan_sha"] = lp.plan_sha(flipped)
    assert lp.is_rung_row(r) is False and lp.is_ladder_consumer_row(r) is False
    probs = lp.row_shape_problems(r, "rungs")
    assert any("leg 'gsm8k' with rung_mode 'consume'" in p and "'gsm8k_inspect_subset' (LADDER_MODE_OF_LEG)" in p for p in probs), probs
    vp = lp.validate_plan(flipped)
    assert any(g["standup_id"] in p and "leg 'gsm8k' with rung_mode 'consume'" in p for p in vp), vp
    assert S0.gate_plan_validates(plan=flipped, tree=REPO)["ok"] is False
    cov = lp.s5_certificate_coverage(flipped, s5_ok, execution_wave=ew, launch_set=[*seed1_launch_set(flipped), g["standup_id"]], read_text=rt)
    cls = (int(g["geometry"]["L"]), int(g["geometry"]["S"]), bool(g["geometry"]["thinking"]))
    e = next(x for x in cov["classes"] if tuple(x["class"]) == cls)
    assert e.get("satisfied_by") != lp.S5_COVERAGE_RUNG_SELF and e.get("gap") is True and f"not ladder consumers: ['{g['standup_id']}']" in e["why"], e
    # (b) the Petri rerun row relabelled consume with a consumer argv carrying NO --kind
    pr = next(x for x in plan["rungs"] if x["leg"] == "petri")
    assert pr["rung_mode"] == lp.RUNG_MODE_PETRI_RERUN and lp.LADDER_KIND_OF_LEG.get("petri") is None
    base_argv = list(lp.plan_row_by_id(plan, "pasttense__s1__8192__shard00")["rung_consumer"])
    nokind = _without_flag(base_argv, "--kind")
    assert "--kind" in base_argv and "--kind" not in nokind and lp._flag_value(nokind, "--kind") is None
    flipped2 = json.loads(json.dumps(plan))
    r2 = next(x for x in flipped2["rungs"] if x["standup_id"] == pr["standup_id"])
    r2["rung_mode"], r2["rung_consumer"] = lp.RUNG_MODE_CONSUME, nokind
    flipped2["launch_plan_sha"] = lp.plan_sha(flipped2)
    assert lp.is_rung_row(r2) is False and lp.is_ladder_consumer_row(r2) is False
    probs = lp.row_shape_problems(r2, "rungs")
    assert any("leg 'petri' with rung_mode 'consume'" in p for p in probs) and any("KNOWN ladder kind by --kind (None per LADDER_KIND_OF_LEG)" in p for p in probs), probs
    vp = lp.validate_plan(flipped2)
    assert any(pr["standup_id"] in p and "KNOWN ladder kind" in p for p in vp) and any(pr["standup_id"] in p and "rung consumer argv is not WS2-E's `consume`" in p for p in vp), vp
    assert S0.gate_plan_validates(plan=flipped2, tree=REPO)["ok"] is False
    # a consume rung whose --kind names ANOTHER known kind is refused too; the generated pasttense rung is admitted
    pt = lp.plan_row_by_id(plan, "pasttense__s1__8192__shard00")
    assert lp.is_ladder_consumer_row(pt) is True
    wrong = {**pt, "rung_consumer": [("controls" if x == "static" else x) for x in pt["rung_consumer"]]}
    assert lp.is_ladder_consumer_row(wrong) is False and any("--kind ('static' per LADDER_KIND_OF_LEG); the consumer argv carries 'controls'" in p for p in lp.row_shape_problems(wrong, "rungs"))
