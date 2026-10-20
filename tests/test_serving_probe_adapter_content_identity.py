"""The serving probe's reference identity guard is keyed on adapter CONTENT, not on the adapter DIRECTORY (probe 979954).

`build_adapter_dirs` exports each slot ONCE into `adapters/_bank/<bank key>` (kind, modules, rank, r_e, seed, scale — never the manifest
or the rank cap) and links it into every manifest that serves it, so the paper-pair probe's rank-cap ladder (S1cc at caps 8-128) carries
the SAME control / constitution bytes under five directories. `identical_reference_outputs` grouped by directory and read five copies of
one adapter agreeing with themselves as "a reference that does not move between different adapters" — 10 of 11 adapters filed
`reference_missing`, the pair unreferenced, the verdict UNCERTIFIED; and `one_ulp_sensitivity` was seeded by the adapter's place in the
phase's loop, so identical bytes drew different noise and read v1 bounds of 0.03 on one dir and 0.12-0.15 on the others. This file pins:
* `adapter_content_id` = sha256 of the dir's adapter_model.safetensors (None when unreadable); the phase records it as `content_sha256`
  on EVERY per-adapter outcome (judged, missing, error) and seeds the one-ulp draws from it (v1 index 0, v2 prompt p index p + 1);
* five twin dirs of one content with identical HF output are ONE subject: no group, no `reference_missing`, each stays judged;
* different contents with identical HF output are still flagged — the reason names the contents (sha prefixes) and the dirs of each;
* a record without `content_sha256` (every record before the key) behaves as today: its directory stands for its content;
* `regate_adapter_record` and `launch_reference_check` carry `content_sha256` (a recorded fact, never re-derived), and never invent one;
* the shape 979954 leaves after the fix: twins collapse, and control's v2 FAIL stays a judged `fail` — a `reference_mismatch` for the
  launch, never `reference_missing` (the guard was fixed; adapter parity is not thereby verified).
CPU only; synthetic ids; no cluster path, no host name."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
import serving_probe as sp  # noqa: E402

SRC = (REPO / "scripts" / "sdfbase" / "serving_probe.py").read_text(encoding="utf-8")
CONTROL = "c" * 64
CONSTITUTION = "d" * 64
DIFFERENCE = "e" * 64
LADDER = ("wo_r4_S1cc", "wo_r4_S1cc_cap16", "wo_r4_S1cc_cap32", "wo_r4_S1cc_cap64", "wo_r4_S1cc_cap128")


def _hf(top1, lp, *rest):
    top5 = [[top1, lp]] + [list(r) for r in rest]
    while len(top5) < 5:
        top5.append([top5[-1][0] + 1, top5[-1][1] - 0.5])
    return {"top1_id": top1, "top1_logprob": lp, "top5": top5}


def _judged(content, hf, *, status="pass", vllm=None, delta=0.006, tol=0.12):
    return {"status": status, "hf": hf, "hf_base": _hf(99, -1.0), "vllm": vllm or {"id": hf["top1_id"], "logprob": hf["top1_logprob"] - delta},
            "moved": {"first_token_changed": True, "max_abs_delta": 2.0, "common_tokens": 3, "top_set_changed": True},
            "sensitivity": {"draws": 3, "rel": 2 ** -8, "shifts": [0.01, 0.02, 0.015], "flips": 0, "max_shift": 0.02, "n_lora_B": 8},
            "top1_equal": status == "pass", "logprob_abs_delta": delta, "tolerance": tol,
            "tolerance_terms": {"tol": 0.02, "floor_term": None, "sensitivity_term": 0.04}, sp.ADAPTER_CONTENT_KEY: content}


def _ladder(content, hf, slot, **kw):
    return {f"/rec/adapters/{m}/{slot}": _judged(content, dict(hf), **kw) for m in LADDER}


# ------------------------------------------------------------------------------------------ the content identity and the seed

def test_adapter_content_id_is_the_sha256_of_the_weights_file_and_none_when_unreadable(tmp_path):
    d = tmp_path / "control__lora"; d.mkdir()
    (d / sp.sa.ADAPTER_WEIGHTS_FILE).write_bytes(b"\x00" * 8 + b"{}" + b"\x01\x02\x03")
    want = hashlib.sha256((d / sp.sa.ADAPTER_WEIGHTS_FILE).read_bytes()).hexdigest()
    assert sp.adapter_content_id(d) == want and len(want) == 64
    twin = tmp_path / "wo_r4_S1cc_cap16" / "control__lora"; twin.mkdir(parents=True)
    (twin / sp.sa.ADAPTER_WEIGHTS_FILE).write_bytes((d / sp.sa.ADAPTER_WEIGHTS_FILE).read_bytes())
    assert sp.adapter_content_id(twin) == want                     # the same bytes under another name: the same identity
    assert sp.adapter_content_id(tmp_path / "absent") is None     # unreadable: None, never a guess


def test_the_one_ulp_seed_is_a_function_of_the_content_and_the_prompt_never_of_the_loop_position():
    s0, s1, s8 = (sp.sensitivity_seed(CONTROL, i) for i in (0, 1, 8))
    assert s0 == sp.sensitivity_seed(CONTROL, 0) == sp.sensitivity_seed(CONTROL, 0, fallback=4)   # the same bytes, the same draw, whatever the loop position
    assert len({s0, s1, s8}) == 3 and all(0 <= s < 2 ** 31 for s in (s0, s1, s8))               # v1 (0) and every v2 prompt (p + 1) draw differently
    assert sp.sensitivity_seed(CONSTITUTION, 0) != s0                                             # another content, another draw
    assert sp.sensitivity_seed(None, 0, fallback=3) == 303 and sp.sensitivity_seed(None, 2, fallback=3) == 305   # unknown content: the caller's position


def test_the_reference_phase_records_the_content_on_every_outcome_and_seeds_by_it():
    """The GPU phase cannot run here; its text is held to the rule: no `seed=len(per)` survives, both sensitivity calls seed by content,
    and the content key is written on every per-adapter outcome (prefix missing, attachment missing, identical to base, error, judged)."""
    body = SRC[SRC.index("def _production_reference_phase("):SRC.index("def finish_reference_record(")]
    assert "seed=len(per)" not in body and "seed=len(per) * 101 + p_i" not in body
    assert "sensitivity_seed(content, 0, fallback=len(per))" in body and "sensitivity_seed(content, p_i + 1, fallback=len(per) * 101 + p_i)" in body
    assert "content = adapter_content_id(d)" in body
    assert body.count("ADAPTER_CONTENT_KEY: content") == 4 and "per[str(d)][ADAPTER_CONTENT_KEY] = content" in body
    outcomes = re.findall(r"per\[str\(d\)\] = \{", body)
    assert len(outcomes) == 4, outcomes            # every `per[str(d)] = {...}` literal carries the key (the judged record sets it on the next line)


# ------------------------------------------------------------------------------------------ the guard

def test_five_twin_dirs_of_one_content_with_identical_hf_output_are_one_subject_and_stay_judged():
    same = _hf(34724, -5.19, [152621, -5.57])
    per = _ladder(CONTROL, same, "control__lora")
    assert sp.identical_reference_outputs(per) == [] and sp.identical_reference_groups(per) == []
    layouts = {"wo": {"status": "written", "compare": {"pass": True, "n_ours": 2, "n_peft": 2, "only_ours": [], "only_peft": []}}}
    rec = sp.finish_reference_record(per, layouts, {d: "wo" for d in per})
    assert all(v["status"] == "pass" and "reason" not in v for v in rec["per_adapter"].values())
    assert rec["counts"] == {"pass": 5, "fail": 0, sp.REF_STATUS_UNDECIDABLE: 0, sp.REF_STATUS_MISSING: 0, "error": 0}
    assert all(v[sp.ADAPTER_CONTENT_KEY] == CONTROL for v in rec["per_adapter"].values())      # the fact travels into the record


def test_different_contents_with_identical_hf_output_are_still_flagged_and_the_reason_names_contents_and_dirs():
    same = _hf(34724, -5.19, [152621, -5.57])
    per = _ladder(CONTROL, same, "control__lora")
    per["/rec/adapters/wo_r4_S1cc/constitution__lora"] = _judged(CONSTITUTION, dict(same))     # another adapter, bit-identical output: not applied
    groups = sp.identical_reference_groups(per)
    assert len(groups) == 1 and len(groups[0]["dirs"]) == 6 and set(groups[0]["contents"]) == {CONTROL, CONSTITUTION}
    assert sp.identical_reference_outputs(per) == [sorted(per)]
    rec = sp.finish_reference_record(per, {"wo": {"status": "written", "compare": {"pass": True}}}, {d: "wo" for d in per})
    got = rec["per_adapter"]
    assert all(v["status"] == sp.REF_STATUS_MISSING for v in got.values())
    reason = got["/rec/adapters/wo_r4_S1cc/constitution__lora"]["reason"]
    assert "IDENTICAL across 2 distinct adapters" in reason and "6 adapter dir(s) by content" in reason
    assert f"content {CONTROL[:12]}: wo_r4_S1cc/control__lora" in reason and f"content {CONSTITUTION[:12]}: wo_r4_S1cc/constitution__lora" in reason
    assert "never applied them" in reason
    # two contents in two dirs (no twins at all): flagged exactly as before the key existed
    two = {"/r/a/x": _judged(CONTROL, dict(same)), "/r/a/y": _judged(CONSTITUTION, dict(same))}
    assert sp.identical_reference_outputs(two) == [["/r/a/x", "/r/a/y"]]


def test_a_record_without_the_content_key_behaves_as_today_by_directory():
    same = _hf(29108, -0.12, [760, -2.25])
    old = {"/r/a/constitution__lora": {"status": "fail", "hf": same}, "/r/a/control__lora": {"status": "fail", "hf": dict(same)},
           "/r/b/difference__lora": {"status": "pass", "hf": _hf(75297, -4.2)}, "/r/c/x": {"status": "reference_missing"}}
    assert sp.identical_reference_outputs(old) == [["/r/a/constitution__lora", "/r/a/control__lora"]]
    g = sp.identical_reference_groups(old)[0]
    assert set(g["contents"]) == {sp._CONTENT_FALLBACK_PREFIX + "/r/a/constitution__lora", sp._CONTENT_FALLBACK_PREFIX + "/r/a/control__lora"}
    reason = sp._identical_group_reason(g)
    assert "IDENTICAL across 2 distinct adapters" in reason and reason.count("content unrecorded") == 2 and "a/control__lora" in reason
    # a lone adapter, and twins whose content is UNKNOWN, read as before too (an unrecorded content never licenses merging two dirs)
    assert sp.identical_reference_outputs({"/r/a": {"status": "pass", "hf": same}}) == []
    unknown_twins = {f"/rec/adapters/{m}/control__lora": {"status": "pass", "hf": dict(same)} for m in LADDER}
    assert len(sp.identical_reference_outputs(unknown_twins)) == 1


def test_regate_and_launch_reference_check_carry_the_content_and_never_invent_one():
    v = _judged(CONTROL, _hf(34724, -5.19, [152621, -5.57]))
    out = sp.regate_adapter_record(v, floor=0.0001)
    assert out[sp.ADAPTER_CONTENT_KEY] == CONTROL and out["status"] == "pass"
    without = {k: val for k, val in v.items() if k != sp.ADAPTER_CONTENT_KEY}
    assert sp.ADAPTER_CONTENT_KEY not in sp.regate_adapter_record(without, floor=0.0001)
    assert sp.regate_adapter_record({"status": sp.REF_STATUS_MISSING, "reason": "x", sp.ADAPTER_CONTENT_KEY: CONTROL}, floor=None)[sp.ADAPTER_CONTENT_KEY] == CONTROL
    rc = sp.launch_reference_check([{"name": "control__lora", "dir": "/rec/adapters/wo_r4_S1cc/control__lora"}], {"/rec/adapters/wo_r4_S1cc/control__lora": v})
    assert rc["status"] == "pass" and rc["per_adapter"]["control__lora"][sp.ADAPTER_CONTENT_KEY] == CONTROL


# ------------------------------------------------------------------------------------------ the 979954 shape after the fix

def _v2(status, launch_id, *, delta):
    return {"version": 2, "status": status, "gate": None, "reason": None, "launch_id": launch_id, "max_abs_delta": delta, "tolerance": 0.9,
            "prompts_sha": "p" * 64, "per_prompt": [], "base_complete": True, "top1_all_equal": status == "pass"}


def test_the_declared_pair_ladder_collapses_to_three_subjects_and_controls_v2_fail_stays_a_mismatch():
    """979954's shape: five S1cc dirs of control (one content) and of constitution (one content) on the ladder, difference on S1d; every HF
    output identical within a content. Under the content-keyed guard nothing is `reference_missing`; the per-launch read then composes
    control's stored v2 FAIL — a measured fail at the scale-100 synthetic scale (a top-1 flip within the base discrepancy), not a LoRA
    parity defect — into a `fail` for every S1cc launch: a `reference_mismatch` for the pair on this record (the parity pass at the
    benign scale is where assertion 4 is certified), never `reference_missing`, and never a pass because the guard was fixed."""
    ctl, con = _hf(34724, -5.19, [152621, -5.57]), _hf(127165, -4.58, [238870, -5.02])
    per = _ladder(CONTROL, ctl, "control__lora")
    per.update(_ladder(CONSTITUTION, con, "constitution__lora", delta=0.056, tol=0.126))
    per["/rec/adapters/wo_r4_S1d/difference__lora"] = _judged(DIFFERENCE, _hf(862, -3.15, [5572, -3.46]), delta=0.078, tol=0.153)
    launches = {m: f"{m}_L4096" if "cap" in m else "wo_r4_S1cc_cap8_L4096" for m in LADDER}
    for m, lid in launches.items():
        per[f"/rec/adapters/{m}/control__lora"]["v2_by_launch"] = {lid: _v2("fail", lid, delta=0.73)}
        per[f"/rec/adapters/{m}/constitution__lora"]["v2_by_launch"] = {lid: _v2("pass", lid, delta=0.09)}
    per["/rec/adapters/wo_r4_S1d/difference__lora"]["v2_by_launch"] = {"wo_r4_S1d_cap8_L4096": _v2("pass", "wo_r4_S1d_cap8_L4096", delta=0.078)}
    layouts = {"wo": {"status": "written", "compare": {"pass": True, "n_ours": 2, "n_peft": 2, "only_ours": [], "only_peft": []}}}
    rec = sp.finish_reference_record(per, layouts, {d: "wo" for d in per})
    got = rec["per_adapter"]
    assert rec["counts"][sp.REF_STATUS_MISSING] == 0 and rec["counts"]["pass"] == 11
    assert len({v[sp.ADAPTER_CONTENT_KEY] for v in got.values()}) == 3          # three subjects under eleven names
    for m, lid in launches.items():
        pending = [{"name": "control__lora", "dir": f"/rec/adapters/{m}/control__lora", "vllm_first_tokens_v2": [{}] * 8},
                   {"name": "constitution__lora", "dir": f"/rec/adapters/{m}/constitution__lora", "vllm_first_tokens_v2": [{}] * 8}]
        rc = sp.launch_reference_check(pending, got, launch_id=lid)
        assert rc["status"] == sp.REF_STATUS_FAIL and rc["basis"] == sp.BASIS_HF_PEFT, (m, rc["status"], rc.get("reason"))
        assert rc["per_adapter"]["control__lora"]["status"] == "fail" and rc["per_adapter"]["control__lora"]["v2"]["status"] == "fail"
        assert rc["per_adapter"]["control__lora"]["v1"]["status"] == "pass"       # v1 passes at every name (0.006 against 0.12); v2 rules
        assert rc["per_adapter"]["constitution__lora"]["status"] == "pass"
        assert rc["per_adapter"]["control__lora"][sp.ADAPTER_CONTENT_KEY] == CONTROL
    s1d = sp.launch_reference_check([{"name": "difference__lora", "dir": "/rec/adapters/wo_r4_S1d/difference__lora", "vllm_first_tokens_v2": [{}] * 8}],
                                    got, launch_id="wo_r4_S1d_cap8_L4096")
    assert s1d["status"] == "pass"
    # the pair's reading: a judged FAIL on a certifying launch is a parity reason (reference_mismatch), not missing evidence
    assert sp.PARITY_REASON_PREFIX.startswith("reference_mismatch") and json.dumps(rec)      # the record serialises (no stray objects)
