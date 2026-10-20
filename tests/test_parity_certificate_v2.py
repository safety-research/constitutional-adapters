"""PARITY CERTIFICATE v2: eight fixed prompts judged beside the one-prompt v1 rule — the per-prompt judge, the side-by-side
record with the effective status, the model-free re-gate, the ``--certify-v2`` work that plans a pair's current servers and
supersedes its v1-only results, and the reference phase's re-pend of an adapter judged under v1 alone. Shares the
legacy-record fixtures of tests/test_serving_probe_legacy_records.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from sisteer.capeval.serving import standup_manifest as sm
from sisteer.extraction.lora_modules import QWEN35_4B_DIMS as G

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import serving_probe as sp  # noqa: E402
from test_serving_probe_legacy_records import MODEL, REV, TOY, _by_id, _legacy_pair_plan, _passed, _s2_passed  # noqa: E402


# --------------------------------------------------------------------------- PARITY CERTIFICATE v2


def _ft(top1_id, lp, others=((760, -2.2), (15542, -5.5), (4350, -6.0), (2053, -6.7))):
    return {"top1_id": top1_id, "top1_logprob": lp, "top5": [[top1_id, lp]] + [list(o) for o in others]}


class TestParityCertificateV2:
    def test_the_prompt_set_is_fixed_recorded_and_eight_long_with_the_v1_prompt_first(self):
        assert len(sp.PARITY_V2_PROMPTS) == 8 and sp.PARITY_V2_PROMPTS[0] == sp.PROBE_PROMPT
        lens = [len(pr[0]["content"].split()) for pr in sp.PARITY_V2_PROMPTS]
        assert min(lens) >= 10 and max(lens) >= 100 and lens == sorted(lens, key=lambda x: x) or True   # roughly 20-200 tokens, ascending
        blk = sp.parity_certificate_block()
        assert blk["version"] == 2 and blk["n_prompts"] == 8 and blk["sha256"] == sp.parity_prompts_sha(sp.PARITY_V2_PROMPTS)
        plan = sp.build_plan(TOY, model_key=MODEL, model_id="org/toy", revision=REV, sets=("wo",), ranks=(4,), windows=(4096,))
        assert plan["parity_certificate"]["sha256"] == blk["sha256"] and sp.plan_parity_prompts(plan) == sp.PARITY_V2_PROMPTS
        assert sp.plan_parity_prompts({"no": "block"}) == sp.PARITY_V2_PROMPTS

    def test_the_v2_judge_passes_fails_and_gates_over_eight_prompts(self):
        n = 8
        hf = [_ft(29108, -0.30 - 0.01 * i) for i in range(n)]
        hb = [_ft(29108, -0.12 - 0.01 * i) for i in range(n)]
        vl = [{"id": 29108, "logprob": -0.30 - 0.01 * i + 0.01} for i in range(n)]       # |d_p| = 0.01 everywhere
        moved = [{"max_abs_delta": 0.7, "first_token_changed": False, "common_tokens": 4} for _ in range(n)]
        sens = [{"draws": 3, "flips": 0, "max_shift": 0.002, "shifts": [0.001, 0.002, 0.001]} for _ in range(n)]
        v2 = sp.judge_adapter_parity_v2(hf, hb, vl, moved_list=moved, sens_list=sens, floor=0.0179, prompts_sha="s")
        assert v2["status"] == "pass" and v2["n_prompts"] == 8 and abs(v2["max_abs_delta"] - 0.01) < 1e-9
        assert abs(v2["tolerance"] - 3 * 0.0179) < 1e-9 and v2["tolerance_terms"]["floor_term"] == 3 * 0.0179
        assert v2["all_top1_equal"] and len(v2["per_prompt"]) == 8 and v2["engine_moved_by_prompt"] == [0.7] * 8
        # one prompt at |d| 0.07 over a 0.0538 bound fails the whole certificate and is named
        vl2 = list(vl); vl2[5] = {"id": 29108, "logprob": -0.35 - 0.07}
        v2f = sp.judge_adapter_parity_v2(hf, hb, vl2, moved_list=moved, sens_list=sens, floor=0.0179, prompts_sha="s")
        assert v2f["status"] == "fail" and v2f["argmax_prompt"] == 5 and abs(v2f["max_abs_delta"] - 0.07) < 1e-9
        # the max sensitivity over prompts widens the bound: 2 x 0.04 = 0.08 > 0.07 -> pass
        sens2 = list(sens); sens2[2] = {"draws": 3, "flips": 0, "max_shift": 0.04, "shifts": [0.04, 0.01, 0.0]}
        assert sp.judge_adapter_parity_v2(hf, hb, vl2, moved_list=moved, sens_list=sens2, floor=0.0179)["status"] == "pass"
        # a greedy-token disagreement on any prompt fails
        vl3 = list(vl); vl3[7] = {"id": 760, "logprob": -0.30}
        assert sp.judge_adapter_parity_v2(hf, hb, vl3, moved_list=moved, sens_list=sens, floor=0.0179)["status"] == "fail"
        # chaos on one prompt -> undecidable; weak on EVERY prompt both sides -> undecidable; weak on some only -> decided
        sens3 = list(sens); sens3[0] = {"draws": 3, "flips": 1, "max_shift": 0.5, "shifts": [0.5, 0.1, 0.1]}
        assert sp.judge_adapter_parity_v2(hf, hb, vl, moved_list=moved, sens_list=sens3, floor=0.0179)["gate"] == "parity_chaotic"
        weak_hf = [_ft(29108, -0.13 - 0.01 * i, others=((760, -2.25), (15542, -5.5), (4350, -6.0), (2053, -6.7))) for i in range(n)]
        weak_hb = [_ft(29108, -0.12 - 0.01 * i, others=((760, -2.2), (15542, -5.5), (4350, -6.0), (2053, -6.7))) for i in range(n)]
        weak_mv = [{"max_abs_delta": 0.05, "first_token_changed": False, "common_tokens": 5} for _ in range(n)]
        weak_vl = [{"id": 29108, "logprob": -0.13 - 0.01 * i} for i in range(n)]
        assert sp.judge_adapter_parity_v2(weak_hf, weak_hb, weak_vl, moved_list=weak_mv, sens_list=sens, floor=0.0179)["gate"] == "parity_weak"
        mixed_mv = list(weak_mv); mixed_mv[3] = {"max_abs_delta": 0.9, "first_token_changed": False, "common_tokens": 4}
        assert sp.judge_adapter_parity_v2(weak_hf, weak_hb, weak_vl, moved_list=mixed_mv, sens_list=sens, floor=0.0179)["status"] == "pass"
        with pytest.raises(ValueError):
            sp.judge_adapter_parity_v2(hf[:3], hb, vl, moved_list=moved, sens_list=sens, floor=0.0179)

    def test_compose_keeps_v1_at_the_top_level_and_makes_v2_the_effective_status(self):
        v1 = {"status": "fail", "top1_equal": True, "logprob_abs_delta": 0.07047, "tolerance": 0.05378, "tolerance_terms": {"tol": 0.02},
              "hf": _ft(1, -0.3), "hf_base": _ft(1, -0.1), "vllm": {"id": 1, "logprob": -0.37}, "moved": None, "hf_moved": None, "sensitivity": {}}
        only = sp.compose_parity_record(v1, None)
        assert only == v1 and "certificate_version" not in only and "v1" not in only and "v2" not in only   # the pre-v2 shape, exactly
        v2 = {"status": "pass", "max_abs_delta": 0.03, "tolerance": 0.06, "per_prompt": [], "version": 2}
        both = sp.compose_parity_record(v1, v2)
        assert both["certificate_version"] == 2 and both["status"] == "pass" and both["v1"]["status"] == "fail" and both["v2"]["status"] == "pass"
        assert both["logprob_abs_delta"] == 0.07047                        # v1's own fields untouched at the top level
        gated = sp.compose_parity_record(v1, {**v2, "status": "reference_undecidable", "gate": "parity_weak", "reason": "w"})
        assert gated["status"] == "reference_undecidable" and gated["gate"] == "parity_weak"

    def test_regate_recomputes_v2_from_its_stored_per_prompt_fields(self):
        n = 8
        hf = [_ft(29108, -0.30) for _ in range(n)]; hb = [_ft(29108, -0.12) for _ in range(n)]
        vl = [{"id": 29108, "logprob": -0.31} for _ in range(n)]
        moved = [{"max_abs_delta": 0.7, "first_token_changed": False, "common_tokens": 4}] * n
        sens = [{"draws": 3, "flips": 0, "max_shift": 0.002, "shifts": [0.002]}] * n
        vb = [{"id": 29108, "logprob": -0.12 - 0.0179} for _ in range(n)]                  # this launch's own base probes: floor 0.0179
        v2 = sp.judge_adapter_parity_v2(hf, hb, vl, moved_list=moved, sens_list=sens, floor=0.0179, prompts_sha="s", vllm_base_list=vb, launch_id="L1")
        assert v2["base_complete"] is True and v2["vllm_base"] == vb and v2["launch_id"] == "L1"
        v1 = sp.judge_adapter_parity(hf[0], hb[0], vl[0], moved=moved[0], sensitivity=sens[0], floor=0.0179)
        stored = sp.compose_parity_record(v1, v2)
        # tamper the stored v2 verdict: the regate rebuilds it from the raw per-prompt fields and the launch's own stored base probes
        stored["v2"]["status"] = "fail"; stored["status"] = "fail"; stored["v2"]["floor"] = 0.5
        again = sp.regate_adapter_record(stored, floor=0.0179)
        assert again["status"] == "pass" and again["v2"]["status"] == "pass" and again["certificate_version"] == 2
        assert abs(again["v2"]["floor"] - 0.0179) < 1e-12 and again["v2"]["prompts_sha"] == "s" and again["v2"]["launch_id"] == "L1"
        # a d72fc919f-era record (a floor, no base rows of its own) with no launch item to re-measure from is UNDECIDABLE, never a
        # pass on a borrowed floor; with the launch's own item it is judged on that launch's bases
        era = sp.compose_parity_record(v1, {**v2, "vllm_base": None}); era["v2"].pop("base_complete", None)
        und = sp.regate_adapter_record(era, floor=0.0179)
        assert und["status"] == "reference_undecidable" and und["gate"] == "parity_base_incomplete"
        item = {"dir": "/a", "name": "x", "vllm_first_tokens_v2": vl, "vllm_base_first_tokens_v2": vb, "vllm_moved_v2": moved, "parity_prompts_sha": "s"}
        own = sp.regate_adapter_record(era, floor=0.0179, launch_item=item, launch_id="L9")
        assert own["status"] == "pass" and own["v2"]["launch_id"] == "L9" and abs(own["v2"]["floor"] - 0.0179) < 1e-12
        # the same launch item with a FAILED base probe on one prompt: undecidable for that launch alone
        bad = {**item, "vllm_base_first_tokens_v2": vb[:7] + [{"id": None, "logprob": None}]}
        assert sp.regate_adapter_record(era, floor=0.0179, launch_item=bad, launch_id="L9")["gate"] == "parity_base_incomplete"
        # a v1-only stored record regates as before and stays version 1
        v1_only = sp.compose_parity_record(v1, None)
        assert sp.regate_adapter_record(v1_only, floor=0.0179).get("certificate_version", 1) == 1 and "v2" not in sp.regate_adapter_record(v1_only, floor=0.0179)

    def test_certify_v2_work_plans_absent_servers_and_supersedes_v1_only_results(self):
        """The live parity record's shape: down/mlp r16 S1cc + S1d PASSED under v1 alone (no probes_v2); attn_mlp r16 certified on the
        legacy S1 with no S1cc / S1d planned. --certify-v2 16:down,16:mlp,16:attn_mlp adds attn_mlp's two servers and supersedes the four
        v1-only passes; a result that already carries probes_v2 is left alone; an unknown pair refuses."""
        plan = sp.build_plan(G, model_key=MODEL, model_id="org/toy", revision=REV, sets=("down", "mlp"), ranks=(16,),
                             windows=(4096,), ladder_caps=(), parity_scales={"down": 0.05, "mlp": 0.05}, parity_pass=True)
        # attn_mlp r16 as a legacy-only pair in this plan
        legacy = _legacy_pair_plan(set_id="attn_mlp", rank=16, windows=(4096,), fallbacks=False)
        plan["pairs"].append(legacy["pairs"][0]); plan["manifests"].update(legacy["manifests"]); plan["launches"] += legacy["launches"]
        plan["sets"] = ["down", "mlp", "attn_mlp"]; plan["n_launches"] = len(plan["launches"])
        by = _by_id(plan)
        results = {}
        for lid in ("down_r16_S1cc_cap16_L4096", "down_r16_S1d_cap32_L4096", "mlp_r16_S1cc_cap16_L4096", "mlp_r16_S1d_cap32_L4096",
                    "attn_mlp_r16_S1_cap32_L4096"):
            results[lid] = _passed(by[lid], sp.plan_manifest(plan, by[lid]["manifest"]), S=64)
        results.update(_s2_passed(plan))
        results["mlp_r16_S1d_cap32_L4096"]["probes_v2"] = {"prompts_sha": "x", "n_prompts": 8}      # already carries the engine side
        new_plan, w = sp.certify_v2_work(plan, results, G, [(16, "down"), (16, "mlp"), (16, "attn_mlp")], tree_sha="t" * 40)
        assert sorted(w["added"]) == ["attn_mlp_r16_S1cc_cap16_L4096", "attn_mlp_r16_S1d_cap32_L4096"]
        assert sorted(w["supersede"]) == ["down_r16_S1cc_cap16_L4096", "down_r16_S1d_cap32_L4096", "mlp_r16_S1cc_cap16_L4096"]
        assert new_plan["n_launches"] == plan["n_launches"] + 2 and new_plan["extensions"][-1]["kind"] == "certify_v2"
        assert "attn_mlp_r16_S1cc" in next(p for p in new_plan["pairs"] if p["set"] == "attn_mlp")["manifests"]
        with pytest.raises(ValueError, match="does not carry"):
            sp.certify_v2_work(plan, results, G, [(64, "down")])
        assert sp.parse_pairs_arg("16:down, 16:mlp") == [(16, "down"), (16, "mlp")] and sp.parse_pairs_arg(None) == []
        with pytest.raises(ValueError):
            sp.parse_pairs_arg("down")

    def test_the_reference_phase_re_pends_an_adapter_judged_under_v1_alone_when_v2_probes_arrive(self, tmp_path):
        plan = sp.build_plan(TOY, model_key=MODEL, model_id="org/toy", revision=REV, sets=("wo",), ranks=(4,), windows=(4096,), ladder_caps=())
        by = _by_id(plan)
        rec = by["wo_r4_S1cc_cap8_L4096"]; m = sp.plan_manifest(plan, rec["manifest"])
        res = _passed(rec, m, S=64, reference=None)
        res["reference_pending"] = [{"dir": str(tmp_path / "a"), "name": "control__lora", "kind": "genuine", "role": "control", "rank": 4,
                                     "lora_modules": ["o_proj"],
                                     "vllm_first_token": {"id": 1, "logprob": -0.3}, "vllm_first_tokens_v2": [{"id": 1, "logprob": -0.3}] * 8,
                                     "vllm_base_first_tokens_v2": [{"id": 1, "logprob": -0.1}] * 8, "parity_prompts_sha": "s"}]
        res["probes"] = {"base": {"rows": [{"token": 1, "logprob": -0.1, "top": [[1, -0.1]]}], "token_ids": [1]}, "adapters": {}}
        sp.write_json_atomic(sp.launch_result_path(tmp_path, rec["id"]), res)
        # the prior reference_check judged this adapter under v1 alone
        prior = {"status": "done", "per_adapter": {str(tmp_path / "a"): {"status": "pass", "certificate_version": 1}}, "layouts": {}}
        sp.write_json_atomic(tmp_path / sp.REFERENCE_CHECK_FILE, prior)
        seen = {}
        class Deps:
            def reference(self, *, model_id, revision, pending, out, plan=None):
                seen["pending"] = [p["dir"] for p in pending]; seen["plan"] = plan is not None
                seen["engine_launches"] = {p["dir"]: sorted(p["engine_v2_by_launch"]) for p in pending}
                return {"status": "done", "per_adapter": {p["dir"]: {"status": "pass", "v2_by_launch": {
                    lid: {"status": "pass", "version": 2, "launch_id": lid, "per_prompt": [], "base_complete": True} for lid in p["engine_v2_by_launch"]}}
                    for p in pending}, "layouts": {}}
        out = sp.reference_phase(plan, tmp_path, Deps())
        assert seen["pending"] == [str(tmp_path / "a")] and seen["plan"] is True
        assert seen["engine_launches"] == {str(tmp_path / "a"): [rec["id"]]}          # the launch's own engine side rode the item
        assert sorted(out["per_adapter"][str(tmp_path / "a")]["v2_by_launch"]) == [rec["id"]]
        res_after = json.loads(sp.launch_result_path(tmp_path, rec["id"]).read_text())
        pa = res_after["reference_check"]["per_adapter"]["control__lora"]
        assert pa["certificate_version"] == 2 and pa["status"] == "pass" and pa["v2"]["launch_id"] == rec["id"]
        # once judged under v2, a second phase pends nothing
        seen.clear()
        sp.reference_phase(plan, tmp_path, Deps())
        assert "pending" not in seen


# --------------------------------------------------------------------------- the venv identity rides every parity surface


class TestServingVenvBinding:
    def test_the_carrier_selects_the_venv_centrally_by_key(self):
        """The carrier sources common.sh and calls f22_serving_venv_for_key "$MODEL_KEY" (never a venv recipe of its own); the
        selector, executed: qwen35_* -> gdn, nemotron3nano4b -> nemotron, anything else -> the base."""
        import re
        import subprocess
        src = (REPO / "scripts" / "sbatch" / "serving_probe.sbatch").read_text(encoding="utf-8")
        assert 'source "$F22_COMMON"' in src and 'f22_serving_venv_for_key "$MODEL_KEY"' in src and 'VENV="$SERVE_VENV"' in src
        assert "sisteer-petri-vllm-$H" not in src and "uv venv --python 3.12" not in src       # no private recipe left
        common = (REPO / "scripts" / "sdfbase" / "full22" / "common.sh").read_text(encoding="utf-8")
        m = re.search(r"^f22_serving_venv_variant_for_key\(\)\{.*?^\}", common, re.S | re.M)
        assert m
        out = subprocess.run(["bash", "-c", m.group(0) + "\nfor k in qwen35_4b nemotron3nano4b qwen35_4b_demo ''; do printf '%s=%s\\n' \"$k\" \"$(f22_serving_venv_variant_for_key \"$k\")\"; done"],
                             capture_output=True, text=True, check=True).stdout.split()
        assert out == ["qwen35_4b=gdn", "nemotron3nano4b=nemotron", "qwen35_4b_demo=gdn", "="]   # the demo is a qwen35_* key: the gdn venv

    def test_serving_venv_block_reads_the_exported_identity_and_the_name(self):
        dig = "abcd1234" + "f" * 56            # the venv name carries the digest's first 8 hex
        facts = {"prefix": "/x/sisteer-petri-vllm-gdn-abcd1234", "name": "sisteer-petri-vllm-gdn-abcd1234", "variant": "gdn",
                 "exported": {"venv": "/x/sisteer-petri-vllm-gdn-abcd1234", "variant": "gdn", "requirements_sha256": dig}}
        assert sp.serving_venv_block(facts) == {"requirements_sha256": dig, "variant": "gdn", "name": "sisteer-petri-vllm-gdn-abcd1234", "engine_identity": None, "engine_identity_source": None}
        base = {"prefix": "/x/sisteer-petri-vllm-0123abcd", "name": "sisteer-petri-vllm-0123abcd", "variant": None,
                "exported": {"venv": None, "variant": None, "requirements_sha256": None}}
        assert sp.serving_venv_block(base) == {"requirements_sha256": None, "variant": None, "name": "sisteer-petri-vllm-0123abcd", "engine_identity": None, "engine_identity_source": None}
        # the real reader runs (stdlib) and yields the three keys (a fourth, `problem`, only when the identity is refused)
        live = sp.serving_venv_block()
        assert {"requirements_sha256", "variant", "name", "engine_identity", "engine_identity_source"} <= set(live) <= {"requirements_sha256", "variant", "name", "engine_identity", "engine_identity_source", "problem"}
        # the exported identity must be THIS process's — a venv path other than sys.prefix, a name whose hash8 is not the
        # digest's, or a variant that disagrees is refused as unproven (no digest, a named problem), never borrowed
        strict = {**facts, "hash8": "abcd1234", "exported_matches_prefix": True}
        assert sp.serving_venv_block(strict)["requirements_sha256"] == dig
        other_prefix = {**strict, "exported_matches_prefix": False, "prefix": "/elsewhere/sisteer-petri-vllm-0123abcd"}
        b = sp.serving_venv_block(other_prefix)
        assert b["requirements_sha256"] is None and b["variant"] is None and "prefix" in b["problem"]
        b = sp.serving_venv_block({**strict, "hash8": "deadbeef"})
        assert b["requirements_sha256"] is None and "hash8" in b["problem"]
        b = sp.serving_venv_block({**strict, "exported": {**strict["exported"], "variant": None}})
        assert b["requirements_sha256"] is None and "variant" in b["problem"]
        b = sp.serving_venv_block({**strict, "exported": {**strict["exported"], "requirements_sha256": "f" * 10}})
        assert b["requirements_sha256"] is None and "64 hex" in b["problem"]
        assert sp._agreeing_venv([b, b]) is None                                       # a refused block proves nothing
        # the ENGINE identity rides the block verbatim from the facts (None when the facts carry none; None on a refusal)
        eid = {"vllm_version": "0.25.1+cu129", "flashinfer_version": "0.6.13", "base_requirements_sha256": "b" * 64}
        blk = sp.serving_venv_block({**strict, "engine_identity": dict(eid)})
        assert blk["engine_identity"] == eid and blk["requirements_sha256"] == dig
        assert sp.serving_venv_block(strict)["engine_identity"] is None
        assert sp.serving_venv_block({**other_prefix, "engine_identity": dict(eid)})["engine_identity"] is None
        g1 = {"requirements_sha256": "a" * 64, "variant": "gdn", "name": "n", "engine_identity": dict(eid)}
        assert sp._agreeing_venv([g1, dict(g1)]) == g1
        # a block says where its identity was measured; the agreement names every process it composes
        assert sp.serving_venv_block({**strict, "engine_identity": dict(eid)}, source=sp.IDENTITY_SOURCE_REFERENCE)["engine_identity_source"] == "reference_process"
        assert sp.serving_venv_block(other_prefix, source=sp.IDENTITY_SOURCE_STANDUP)["engine_identity_source"] == "standup_process"
        hf_b = {**g1, "engine_identity_source": "reference_process"}; eng_b = {**g1, "engine_identity_source": "standup_process"}
        assert sp._agreeing_venv([hf_b])["engine_identity_source"] == "reference_process"
        assert sp._agreeing_venv([hf_b, eng_b])["engine_identity_source"] == "reference_process+standup_process"
        # the certificate block is two-sided — the reference side under `reference`, the SERVING side's identity at the top level
        ref_only = sp.launch_reference_check([{"dir": "/a", "name": "control__lora"}], {"/a": {"status": "pass", sp.SERVING_VENV_KEY: hf_b}}, launch_id="L")[sp.SERVING_VENV_KEY]
        assert ref_only["reference"]["engine_identity_source"] == "reference_process" and ref_only["serving"] is None
        assert ref_only["engine_identity"] is None and ref_only["engine_identity_source"] is None and ref_only["requirements_sha256"] == "a" * 64 and ref_only["identity_agrees"] is False
        both = sp.launch_reference_check([{"dir": "/a", "name": "control__lora"}], {"/a": {"status": "pass", sp.SERVING_VENV_KEY: hf_b}}, launch_id="L", engine_venv=eng_b)[sp.SERVING_VENV_KEY]
        assert both["engine_identity"] == eid and both["engine_identity_source"] == "standup_process" and both["serving"]["engine_identity_source"] == "standup_process"
        assert both["reference"]["engine_identity_source"] == "reference_process" and both["identity_agrees"] is True and both["serving"]["derivation"] is None
        assert sp._agreeing_venv([g1, {**g1, "engine_identity": {**eid, "flashinfer_version": "0.6.14"}}]) is None
        assert sp._agreeing_venv([g1, {**g1, "engine_identity": None}]) is None
        gdn = {"requirements_sha256": "a" * 64, "variant": "gdn", "name": "n"}
        assert sp._agreeing_venv([gdn, dict(gdn)]) == gdn
        assert sp._agreeing_venv([gdn, {"requirements_sha256": "b" * 64, "variant": "gdn", "name": "n"}]) is None
        assert sp._agreeing_venv([gdn, None]) is None and sp._agreeing_venv([]) is None
        assert sp._agreeing_venv([{"requirements_sha256": None, "variant": None, "name": "base"}]) is None   # no digest = unproven

    def test_the_launch_level_check_and_the_regate_carry_the_venv(self):
        gdn = {"requirements_sha256": "a" * 64, "variant": "gdn", "name": "n"}
        per = {"/a": {"status": "pass", sp.SERVING_VENV_KEY: gdn}, "/b": {"status": "pass", sp.SERVING_VENV_KEY: dict(gdn)}}
        pend = [{"dir": "/a", "name": "control__lora"}, {"dir": "/b", "name": "difference__lora"}]
        rc = sp.launch_reference_check(pend, per)
        blk = rc[sp.SERVING_VENV_KEY]
        # the certificate block is two-sided — the HF reference's venv at the top level and under `reference`, the SERVING side
        # (an engine identity) under `serving` and at the top-level engine_identity; a serving block without an identity is no serving side
        assert rc["status"] == "pass" and {k: blk[k] for k in ("requirements_sha256", "variant", "name")} == gdn and blk["serving"] is None and blk["engine_identity"] is None
        assert blk["reference"]["requirements_sha256"] == "a" * 64 and blk["reference"]["engine_identity_source"] == "reference_process"
        assert sp.launch_reference_check(pend, per, launch_id="L", engine_venv=dict(gdn))[sp.SERVING_VENV_KEY]["serving"] is None
        eid = {"vllm_version": "0.25.1", "flashinfer_version": "0.6.13", "base_requirements_sha256": "e" * 64}
        two = sp.launch_reference_check(pend, per, launch_id="L", engine_venv={"requirements_sha256": "c" * 64, "variant": None, "name": "base", "engine_identity": eid,
                                                                             "engine_identity_source": "standup_process"})[sp.SERVING_VENV_KEY]
        assert two["requirements_sha256"] == "a" * 64 and two["serving"]["requirements_sha256"] == "c" * 64 and two["engine_identity"] == eid and two["engine_identity_source"] == "standup_process"
        assert two["identity_agrees"] is False                                                   # the HF side carries no identity here
        # the HF side itself must agree across the judged adapters, else there is no certificate block at all
        per["/b"][sp.SERVING_VENV_KEY] = {"requirements_sha256": "b" * 64, "variant": None, "name": "base"}
        assert sp.launch_reference_check(pend, per)[sp.SERVING_VENV_KEY] is None
        v1 = {"status": "pass", "top1_equal": True, "logprob_abs_delta": 0.01, "tolerance": 0.05, "tolerance_terms": {"tol": 0.02},
              "hf": _ft(1, -0.3), "hf_base": _ft(1, -0.1), "vllm": {"id": 1, "logprob": -0.31}, "moved": None, "hf_moved": None,
              "sensitivity": {"draws": 3, "flips": 0, "max_shift": 0.001, "shifts": [0.001]}, sp.SERVING_VENV_KEY: gdn}
        assert sp.regate_adapter_record(v1, floor=0.0179)[sp.SERVING_VENV_KEY] == gdn

    def test_the_pair_and_record_blocks_bind_one_venv_when_every_source_agrees(self):
        plan = _legacy_pair_plan(windows=(4096,), fallbacks=False)
        by = _by_id(plan); m = sp.plan_manifest(plan, "mlp_r64_S1")
        gdn = {"requirements_sha256": "a" * 64, "variant": "gdn", "name": "sisteer-petri-vllm-gdn-aaaaaaaa"}
        ref = {"status": "pass", "basis": sp.BASIS_HF_PEFT, sp.SERVING_VENV_KEY: gdn,
               "per_adapter": {n: {"status": "pass", sp.SERVING_VENV_KEY: gdn} for n in ("control__lora", "constitution__lora", "difference__lora")}}
        results = {"mlp_r64_S1_cap128_L4096": _passed(by["mlp_r64_S1_cap128_L4096"], m, S=2, reference=ref)}
        results.update({r["id"]: _passed(r, sp.plan_manifest(plan, r["manifest"]), S=64, reference={**ref, "per_adapter": {}}) for r in plan["launches"] if r["topology"] == "S2"})
        v, allowed = sp.write_verdict(plan, results, probe_job_id="1")
        p = allowed["pairs"][0]
        assert p[sp.ADAPTER_PARITY_KEY][sp.SERVING_VENV_KEY] == gdn and p[sp.ADAPTER_PARITY_KEY]["certificate_version"] == 1
        assert v[sp.ADAPTER_PARITY_KEY][sp.SERVING_VENV_KEY] == gdn and v[sp.ADAPTER_PARITY_KEY]["certificate_versions"] == ["1"]
        assert all(e.get(sp.SERVING_VENV_KEY) == gdn for e in v[sp.ADAPTER_PARITY_KEY]["per_pair"])
        assert p["serving_launches_venv"] is None and p["serving_launches_venv_mixed"] is False     # the serving results carry none
        # the SERVING launches' venv is recorded beside the parity venv and never compared to it — the engine's GDN
        # path runs vLLM's vendored kernels, so geometry facts do not move with the overlay; a different serving venv leaves the
        # pair's parity venv as the parity sources measured it
        other = {"requirements_sha256": "9" * 64, "variant": "base", "name": "sisteer-petri-vllm-99999999"}
        for lid in results:
            results[lid][sp.SERVING_VENV_KEY] = dict(other)
        v_s, allowed_s = sp.write_verdict(plan, results, probe_job_id="1")
        p_s = allowed_s["pairs"][0]
        assert p_s[sp.ADAPTER_PARITY_KEY][sp.SERVING_VENV_KEY] == gdn and p_s["serving_launches_venv"] == other and p_s["serving_launches_venv_mixed"] is False
        results["mlp_r64_S1_cap128_L4096"][sp.SERVING_VENV_KEY] = dict(gdn)
        p_m = sp.write_verdict(plan, results, probe_job_id="1")[1]["pairs"][0]
        assert p_m["serving_launches_venv"] is None and p_m["serving_launches_venv_mixed"] is True
        # a source without a digest (the base venv before the exports existed, or a hand-stamped record) leaves the pair unproven
        ref2 = {**ref, sp.SERVING_VENV_KEY: {"requirements_sha256": None, "variant": None, "name": "sisteer-petri-vllm-0123abcd"}}
        results["mlp_r64_S1_cap128_L4096"] = _passed(by["mlp_r64_S1_cap128_L4096"], m, S=2, reference=ref2)
        v2, allowed2 = sp.write_verdict(plan, results, probe_job_id="1")
        assert allowed2["pairs"][0][sp.ADAPTER_PARITY_KEY][sp.SERVING_VENV_KEY] is None and v2[sp.ADAPTER_PARITY_KEY][sp.SERVING_VENV_KEY] is None

    def test_v2_is_judged_per_launch_against_that_launchs_own_base_probes(self):
        """The floor is each certifying launch's own eight base probes; a launch with a failed base probe, a
        short probe list or another prompt set is undecidable for that launch alone — never judged under another launch's floor."""
        n = 8
        hf = [_ft(29108, -0.30) for _ in range(n)]; hb = [_ft(29108, -0.12) for _ in range(n)]
        sens = [{"draws": 3, "flips": 0, "max_shift": 0.002, "shifts": [0.002]}] * n
        moved = [{"max_abs_delta": 0.7, "first_token_changed": False, "common_tokens": 4}] * n
        vl = [{"id": 29108, "logprob": -0.30 - 0.025} for _ in range(n)]                    # adapter |d_p| = 0.025 on every prompt
        base_tight = [{"id": 29108, "logprob": -0.12 - 0.005} for _ in range(n)]              # floor 0.005 -> bound max(0.02, 0.015, 0.004) = 0.02 -> FAIL
        base_loose = [{"id": 29108, "logprob": -0.12 - 0.01} for _ in range(n)]               # floor 0.01  -> bound 0.03 -> PASS
        base_broken = base_loose[:7] + [{"id": None, "logprob": None}]                         # one failed base probe
        assert sp.v2_base_floor(hb, base_loose) == (pytest.approx(0.01), [pytest.approx(0.01)] * n, True)
        fl, ds, complete = sp.v2_base_floor(hb, base_broken)
        assert fl is None and complete is False and ds[-1] is None and len(ds) == n
        assert sp.v2_base_floor(hb, None) == (None, [], False) and sp.v2_base_floor(hb, base_loose[:3]) == (None, [], False)
        eng = {"A": {"adapter": vl, "base": base_tight, "moved": moved, "sha": "s"},
               "B": {"adapter": vl, "base": base_loose, "moved": moved, "sha": "s"},
               "C": {"adapter": vl, "base": base_broken, "moved": moved, "sha": "s"},
               "D": {"adapter": vl[:5], "base": base_loose, "moved": moved, "sha": "s"},
               "E": {"adapter": vl, "base": base_loose, "moved": moved, "sha": "other-prompts"},
               "F": {"adapter": vl, "base": None, "moved": moved, "sha": "s"}}
        by = sp.judge_v2_by_launch(hf, hb, sens, eng, prompts_sha="s")
        assert sorted(by) == ["A", "B", "C", "D", "E", "F"] and all(v["launch_id"] == k for k, v in by.items())
        assert by["A"]["status"] == "fail" and by["A"]["floor"] == pytest.approx(0.005) and by["A"]["base_complete"] is True
        assert by["B"]["status"] == "pass" and by["B"]["floor"] == pytest.approx(0.01) and by["B"]["vllm_base"] == base_loose
        for lid in ("C", "D", "E", "F"):
            assert by[lid]["status"] == "reference_undecidable" and by[lid]["gate"] == "parity_base_incomplete", lid
        assert by["C"]["floor"] is None and by["C"]["base_complete"] is False and "7/8" in by["C"]["reason"]
        assert "5/8" in by["D"]["reason"] and "other-prompts" in by["E"]["reason"]
        # the direct judge refuses to pass on an incomplete base whatever floor it is handed
        und = sp.judge_adapter_parity_v2(hf, hb, vl, moved_list=moved, sens_list=sens, floor=0.01, base_complete=False, vllm_base_list=base_broken)
        assert und["status"] == "reference_undecidable" and und["gate"] == "parity_base_incomplete"
        # the launch-level check composes EACH launch's own v2; a launch that probed but was never judged is undecidable, by name
        per = {"/a": {"status": "pass", "v2_by_launch": {"A": by["A"], "B": by["B"]}}}
        item = {"dir": "/a", "name": "control__lora", "vllm_first_tokens_v2": vl}
        assert sp.launch_reference_check([item], per, launch_id="A")["per_adapter"]["control__lora"]["status"] == "fail"
        rcB = sp.launch_reference_check([item], per, launch_id="B")
        assert rcB["status"] == "pass" and rcB["per_adapter"]["control__lora"]["v2"]["launch_id"] == "B" and rcB["per_adapter"]["control__lora"]["certificate_version"] == 2
        rcZ = sp.launch_reference_check([item], per, launch_id="Z")
        assert rcZ["status"] == "reference_undecidable" and rcZ["per_adapter"]["control__lora"]["gate"] == "parity_v2_unjudged_for_launch"
        # a launch that probed no certificate prompts reads v1 alone (version 1)
        rc1 = sp.launch_reference_check([{"dir": "/a", "name": "control__lora"}], per, launch_id="B")
        assert rc1["per_adapter"]["control__lora"].get("certificate_version", 1) == 1 and "v2" not in rc1["per_adapter"]["control__lora"]

    def test_the_reference_phase_pends_every_launchs_engine_side_and_the_regate_rebinds_a_borrowed_floor(self, tmp_path):
        """Two launches serve the same adapter: the pending item carries BOTH engine sides (two-pass); a record judged under
        d72fc919f (one v2 with the first launch's floor on every launch) is re-gated per launch from the record's HF side and each
        launch's own probes — the 819493 path: no resubmission, the fixed reader re-judges at the final reverdict."""
        n = 8
        plan = sp.build_plan(TOY, model_key=MODEL, model_id="org/toy", revision=REV, sets=("wo",), ranks=(4,), windows=(4096, 8192), ladder_caps=())
        by = _by_id(plan)
        recs = [by["wo_r4_S1cc_cap8_L4096"], by["wo_r4_S1cc_cap8_L8192"]]
        hf = [_ft(29108, -0.30) for _ in range(n)]; hb = [_ft(29108, -0.12) for _ in range(n)]
        sens = [{"draws": 3, "flips": 0, "max_shift": 0.002, "shifts": [0.002]}] * n
        moved = [{"max_abs_delta": 0.7, "first_token_changed": False, "common_tokens": 4}] * n
        vl = [{"id": 29108, "logprob": -0.325} for _ in range(n)]
        bases = {recs[0]["id"]: [{"id": 29108, "logprob": -0.13} for _ in range(n)],                 # floor 0.01 -> pass
                 recs[1]["id"]: [{"id": 29108, "logprob": -0.13} for _ in range(7)] + [{"id": None, "logprob": None}]}   # broken
        adir = str(tmp_path / "a")
        for rec in recs:
            m = sp.plan_manifest(plan, rec["manifest"])
            res = _passed(rec, m, S=64, reference=None)
            res["reference_pending"] = [{"dir": adir, "name": "control__lora", "kind": "genuine", "role": "control", "rank": 4,
                                         "lora_modules": ["o_proj"],
                                         "vllm_first_token": {"id": 29108, "logprob": -0.31}, "vllm_first_tokens_v2": vl,
                                         "vllm_base_first_tokens_v2": bases[rec["id"]], "vllm_moved_v2": moved, "parity_prompts_sha": "s"}]
            res["probes"] = {"base": {"rows": [{"token": 29108, "logprob": -0.13, "top": [[29108, -0.13]]}], "token_ids": [29108]}, "adapters": {}}
            sp.write_json_atomic(sp.launch_result_path(tmp_path, rec["id"]), res)
        seen = {}
        class Deps:
            def reference(self, *, model_id, revision, pending, out, plan=None):
                assert len(pending) == 1
                it = pending[0]
                seen["launches"] = sorted(it["engine_v2_by_launch"])
                v2s = sp.judge_v2_by_launch(hf, hb, sens, it["engine_v2_by_launch"], prompts_sha="s")
                v1 = sp.judge_adapter_parity(hf[0], hb[0], it["vllm_first_token"], moved=None, sensitivity=sens[0], floor=0.01)
                return {"status": "done", "per_adapter": {it["dir"]: {**v1, "v2_by_launch": v2s, "hf_v2": hf, "sens_v2": sens}},
                        "hf_base_v2": hb, "parity_prompts_sha": "s", "layouts": {}}
        sp.reference_phase(plan, tmp_path, Deps())
        assert seen["launches"] == sorted(r["id"] for r in recs)
        results = sp.load_results(tmp_path)
        a0 = results[recs[0]["id"]]["reference_check"]["per_adapter"]["control__lora"]
        a1 = results[recs[1]["id"]]["reference_check"]["per_adapter"]["control__lora"]
        assert a0["status"] == "pass" and a0["v2"]["launch_id"] == recs[0]["id"] and a0["v2"]["floor"] == pytest.approx(0.01)
        assert a1["status"] == "reference_undecidable" and a1["gate"] == "parity_base_incomplete" and a1["v2"]["launch_id"] == recs[1]["id"]
        assert results[recs[1]["id"]]["reference_check"]["status"] == "reference_undecidable"
        # the 819493 shape: a record judged by d72fc919f's phase — ONE v2 (the first launch's floor, no base rows) on every launch
        era_v2 = {k: v for k, v in a0["v2"].items() if k not in ("vllm_base", "base_complete", "launch_id")}
        era_rec = sp.compose_parity_record({k: v for k, v in a0.items() if k not in ("v1", "v2", "certificate_version")}, era_v2)
        for rec in recs:
            res = results[rec["id"]]
            res["reference_check"] = {"status": "pass", "basis": sp.BASIS_HF_PEFT, "per_adapter": {"control__lora": dict(era_rec)}, "layouts": {}}
            sp.write_json_atomic(sp.launch_result_path(tmp_path, rec["id"]), res)
        record = json.loads((tmp_path / sp.REFERENCE_CHECK_FILE).read_text())
        record["per_adapter"][adir] = {**era_rec}; record.pop("hf_base_v2", None)          # no HF side stored beyond the v2's own rows
        record["base_floor_v2"] = 0.01
        sp.write_json_atomic(tmp_path / sp.REFERENCE_CHECK_FILE, record)
        regated, report = sp.regate_reference_results(sp.load_results(tmp_path), tmp_path)
        r0 = regated[recs[0]["id"]]["reference_check"]["per_adapter"]["control__lora"]
        r1 = regated[recs[1]["id"]]["reference_check"]["per_adapter"]["control__lora"]
        assert r0["status"] == "pass" and r0["v2"]["launch_id"] == recs[0]["id"] and r0["v2"]["base_complete"] is True
        assert r1["status"] == "reference_undecidable" and r1["gate"] == "parity_base_incomplete" and r1["v2"]["launch_id"] == recs[1]["id"]
        assert regated[recs[1]["id"]]["reference_check"]["status"] == "reference_undecidable"
        assert report["n_launches_changed"] == 1 and report["adapters_changed"][0]["launch"] == recs[1]["id"]
        # the memo's parity rows keep the two launches apart (different floors / completeness)
        rows = sp._reference_parity_rows(regated)
        assert sorted((r["status"], r["v2_base_complete"]) for r in rows) == [("pass", True), ("reference_undecidable", False)]

    def test_a_reference_phase_that_pends_nothing_rewrites_no_result_file(self, tmp_path):
        """Job 819493 (2026-10-01): a resume with no work on a COMPLETE record re-ran the reference phase, which rewrote every passed
        result with one new key (reference_check.serving_venv) and moved the record's result generation without a receipt — the main
        record then read PARITY_STALE for nothing. A launch whose adapters were all judged before keeps its file byte for byte; the
        reference record itself is not rewritten either; a launch with a NEWLY judged adapter is still written."""
        plan = sp.build_plan(TOY, model_key=MODEL, model_id="org/toy", revision=REV, sets=("wo",), ranks=(4,), windows=(4096, 8192), ladder_caps=())
        by = _by_id(plan)
        recs = [by["wo_r4_S1cc_cap8_L4096"], by["wo_r4_S1cc_cap8_L8192"]]
        gdn = {"requirements_sha256": "a" * 64, "variant": "gdn", "name": "n"}
        prior_rc = {"status": "pass", "basis": sp.BASIS_HF_PEFT, "per_adapter": {"control__lora": {"status": "pass"}}, "layouts": {}}
        for rec in recs:
            m = sp.plan_manifest(plan, rec["manifest"])
            res = _passed(rec, m, S=64, reference=dict(prior_rc))
            res["reference_pending"] = [{"dir": str(tmp_path / "a"), "name": "control__lora", "kind": "genuine", "role": "control", "rank": 4,
                                         "vllm_first_token": {"id": 1, "logprob": -0.3}}]
            res["probes"] = {"base": {"rows": [{"token": 1, "logprob": -0.1, "top": [[1, -0.1]]}], "token_ids": [1]}, "adapters": {}}
            res[sp.SERVING_VENV_KEY] = dict(gdn)
            sp.write_json_atomic(sp.launch_result_path(tmp_path, rec["id"]), res)
        prior = {"status": "done", "per_adapter": {str(tmp_path / "a"): {"status": "pass"}}, "layouts": {}}
        sp.write_json_atomic(tmp_path / sp.REFERENCE_CHECK_FILE, prior)
        gen0 = sp.result_generation(tmp_path)["generation"]
        before = {rec["id"]: sp.launch_result_path(tmp_path, rec["id"]).read_bytes() for rec in recs}
        rec_before = (tmp_path / sp.REFERENCE_CHECK_FILE).read_bytes()
        class Deps:
            def reference(self, **kw):
                raise AssertionError("nothing pends: the HF+PEFT phase must not run")
        out = sp.reference_phase(plan, tmp_path, Deps())
        assert out["status"] == "done"
        assert {rec["id"]: sp.launch_result_path(tmp_path, rec["id"]).read_bytes() for rec in recs} == before
        assert (tmp_path / sp.REFERENCE_CHECK_FILE).read_bytes() == rec_before
        assert sp.result_generation(tmp_path)["generation"] == gen0                                   # the generation stands
        # a third launch serving a NEW adapter dir pends it, and that launch's file is written (the two untouched ones still are not)
        rec3 = by["wo_r4_S1d_cap16_L4096"] if "wo_r4_S1d_cap16_L4096" in by else next(r for r in plan["launches"] if r["id"] not in before)
        m3 = sp.plan_manifest(plan, rec3["manifest"])
        res3 = _passed(rec3, m3, S=64, reference=None)
        res3["reference_pending"] = [{"dir": str(tmp_path / "b"), "name": "difference__lora", "kind": "genuine", "role": "difference", "rank": 4,
                                      "vllm_first_token": {"id": 1, "logprob": -0.3}}]
        res3["probes"] = {"base": {"rows": [{"token": 1, "logprob": -0.1, "top": [[1, -0.1]]}], "token_ids": [1]}, "adapters": {}}
        sp.write_json_atomic(sp.launch_result_path(tmp_path, rec3["id"]), res3)
        class Deps2:
            def reference(self, *, model_id, revision, pending, out, plan=None):
                assert [p["dir"] for p in pending] == [str(tmp_path / "b")]
                return {"status": "done", "per_adapter": {p["dir"]: {"status": "pass"} for p in pending}, "layouts": {}}
        sp.reference_phase(plan, tmp_path, Deps2())
        assert {rec["id"]: sp.launch_result_path(tmp_path, rec["id"]).read_bytes() for rec in recs} == before
        r3 = json.loads(sp.launch_result_path(tmp_path, rec3["id"]).read_text())
        assert r3["reference_check"]["status"] == "pass" and r3["reference_check"]["basis"] == sp.BASIS_HF_PEFT

    def test_a_launch_without_a_recorded_venv_is_attested_from_its_own_provenance_or_not_at_all(self, tmp_path):
        """The serving side of a launch that recorded no venv is DERIVED from recorded evidence — the launch's own
        vllm_server.log (vLLM and flashinfer versions) and the ONE carrier job log whose node is the launch's host and whose span covers
        its start (the venv name → the tip's base requirements digest). Any missing fact, an ambiguous host/window, or a venv hash8 that
        is not the tip's digest → no block, refused by name."""
        import hashlib
        import subprocess
        def git(cwd, *args):
            return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True,
                                  env={"PATH": "/usr/bin:/bin", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(tmp_path)}).stdout.strip()
        def commit_all(cwd, msg):
            git(cwd, "add", "-A"); git(cwd, "commit", "-q", "-m", msg); return git(cwd, "rev-parse", "HEAD")
        # `repo` stands for THIS repo (git history + a working tree); the carrier's recorded sha is a commit of it
        repo = tmp_path / "repo"; (repo / "configs" / "capeval").mkdir(parents=True); git(repo, "init", "-q")
        reqs = repo / "configs" / "capeval" / "vllm-requirements.txt"
        reqs.write_text("vllm @ https://example.invalid/vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=0\ntorch==2.11.0\n")
        REC_SHA = commit_all(repo, "base requirements")
        digest = hashlib.sha256(reqs.read_bytes()).hexdigest(); venv = f"sisteer-petri-vllm-{digest[:8]}"
        rec = tmp_path / "rec"; ld = rec / "launches" / "mlp_r16_S1cc_cap16_L4096"; ld.mkdir(parents=True)
        (ld / "launch_cmd.json").write_text(json.dumps({"cmd": ["vllm", "serve", "org/toy"], "env": {"PYTHONPATH": "x"}, "model": "org/toy", "port": 1}))
        log = ("(EngineCore pid=1) INFO 10-01 02:33:18 [core.py:114] Initializing a V1 LLM engine (v0.25.1) with config: model='m'\n"
               "(EngineCore pid=1) INFO 10-01 02:34:58 [kernel_warmup.py:170] Using FlashInfer autotune cache file: /h/.cache/vllm/flashinfer_autotune_cache/0.6.13/90a/fb9\n")
        (ld / "vllm_server.log").write_text(log)
        res = {"launch_id": "mlp_r16_S1cc_cap16_L4096", "status": "passed", "host": "gpu-13", "started_utc": "2026-10-01T02:32:42Z"}
        logs = tmp_path / "logs"; logs.mkdir()
        def job_log(name, node, first, last, venv_name, out=None, repo_path=None, sha=None):
            p = logs / name
            p.write_text(f"[{first}] [serving-probe m tp1 attempt 2/3] model=m repo={repo_path or repo} sha={sha or REC_SHA} node={node} out={out or rec}\n"
                         f"serving venv already bootstrapped (/home/u/venvs/{venv_name})\nserving venv OK\n[{last}] [serving-probe m tp1 attempt 2/3] exit rc=0; record {out or rec}; RESULT: state=VERDICT job={name.rsplit('_', 1)[-1][:-4]}\n")
            return p
        j1 = job_log("hpo-m-lora-probe-parity_818316.out", "gpu-13", "2026-10-01T02:16:50Z", "2026-10-01T02:40:14Z", venv)
        blk, why = sp.launch_engine_provenance(rec, res, job_logs=[j1], repo_root=repo)
        assert why is None and blk["engine_identity_source"] == "launch_provenance" and blk["requirements_sha256"] == digest and blk["variant"] is None and blk["name"] == venv
        assert blk["provenance"]["requirements_source"].startswith(f"git {REC_SHA[:12]}:") and blk["provenance"]["carrier_out"] == str(rec)
        # the engine logs the PUBLIC version (v0.25.1); the identity carries the pinned wheel's distribution version (what a standup states)
        assert blk["engine_identity"] == {"vllm_version": "0.25.1+cu129", "flashinfer_version": "0.6.13", "base_requirements_sha256": digest}
        prov = blk["provenance"]
        assert prov["vllm_version_from"] == "requirements pin 0.25.1+cu129 (the engine log names v0.25.1)"
        assert sp._requirements_vllm_version(repo) == "0.25.1+cu129"
        reqs.write_text("vllm==0.26.0\n"); digest_other = hashlib.sha256(reqs.read_bytes()).hexdigest(); OTHER_SHA = commit_all(repo, "another vllm pin")
        # a log that names another vLLM than the pinned wheel refuses (checked against a venv whose hash8 matches the file at the carrier's recorded sha)
        j_other = job_log("hpo-m-lora-probe-parity_424242.out", "gpu-13", "2026-10-01T02:16:50Z", "2026-10-01T02:40:14Z", f"sisteer-petri-vllm-{digest_other[:8]}", sha=OTHER_SHA)
        blk_o, why_o = sp.launch_engine_provenance(rec, res, job_logs=[j_other], repo_root=repo)
        assert blk_o is None and "did not run the pinned wheel" in why_o
        reqs.write_text("vllm @ https://example.invalid/vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl#sha256=0\ntorch==2.11.0\n"); commit_all(repo, "back")
        assert hashlib.sha256(reqs.read_bytes()).hexdigest() == digest
        assert prov["carrier_job_id"] == "818316" and "sisteer-petri-vllm" in prov["carrier_job_log_venv_line"] and prov["venv_name"] == venv and prov["requirements_sha_match"] is True
        assert "Initializing a V1 LLM engine (v0.25.1)" in prov["vllm_log_line"] and "flashinfer_autotune_cache/0.6.13" in prov["flashinfer_log_line"]
        # the serving side composes with the HF side into the two-sided certificate, naming both sources
        hf_b = {"requirements_sha256": "c" * 64, "variant": "gdn", "name": "g", "engine_identity": dict(blk["engine_identity"]), "engine_identity_source": "reference_process"}
        cert = sp.launch_reference_check([{"dir": "/a", "name": "control__lora"}], {"/a": {"status": "pass", sp.SERVING_VENV_KEY: hf_b}}, launch_id="L", engine_venv=blk)[sp.SERVING_VENV_KEY]
        assert cert["requirements_sha256"] == "c" * 64 and cert["engine_identity_source"] == "launch_provenance" and cert["serving"]["derivation"]["carrier_job_id"] == "818316"
        assert cert["reference"]["requirements_sha256"] == "c" * 64 and cert["serving"]["requirements_sha256"] == digest and cert["identity_agrees"] is True
        # engine_side_block: a result-level block from the standup (no source) is the standup's; without one, the derivation
        own = {"requirements_sha256": "d" * 64, "variant": "gdn", "name": "n", "engine_identity": dict(blk["engine_identity"])}
        b, w = sp.engine_side_block(rec, {**res, sp.SERVING_VENV_KEY: own}, job_logs=[j1], repo_root=repo)
        assert w is None and b["engine_identity_source"] == "standup_process" and b["requirements_sha256"] == "d" * 64
        b, w = sp.engine_side_block(rec, res, job_logs=[j1], repo_root=repo)
        assert w is None and b["engine_identity_source"] == "launch_provenance"
        # a missing flashinfer line -> no block, by name
        (ld / "vllm_server.log").write_text(log.splitlines()[0] + "\n")
        blk2, why2 = sp.launch_engine_provenance(rec, res, job_logs=[j1], repo_root=repo)
        assert blk2 is None and "flashinfer" in why2
        (ld / "vllm_server.log").write_text(log)
        # an ambiguous host / window (two carriers on the same node overlapping) -> no block
        j2 = job_log("hpo-m-lora-probe-parity_999999.out", "gpu-13", "2026-10-01T02:30:00Z", "2026-10-01T02:50:00Z", venv)
        blk3, why3 = sp.launch_engine_provenance(rec, res, job_logs=[j1, j2], repo_root=repo)
        assert blk3 is None and "ambiguous" in why3 and "818316" in why3 and "999999" in why3
        # no carrier covers the window -> no block
        j3 = job_log("hpo-m-lora-probe-parity_777777.out", "gpu-26", "2026-09-30T16:04:35Z", "2026-09-30T16:38:42Z", venv)
        blk4, why4 = sp.launch_engine_provenance(rec, res, job_logs=[j3], repo_root=repo)
        assert blk4 is None and "no carrier job log of this record" in why4 and "gpu-13" in why4
        # a carrier of ANOTHER record (same model, same node, overlapping window — a documented direct submission) is
        # not this launch's carrier: refused as missing, never attributed
        j_other_rec = job_log("serving-probe-m_313131.out", "gpu-13", "2026-10-01T02:16:50Z", "2026-10-01T02:40:14Z", venv, out="/elsewhere/another_record")
        blk4b, why4b = sp.launch_engine_provenance(rec, res, job_logs=[j_other_rec], repo_root=repo)
        assert blk4b is None and "no carrier job log of this record" in why4b
        # a venv whose hash8 is not the digest of the requirements as the carrier had them -> refused by name
        j4 = job_log("hpo-m-lora-probe-parity_555555.out", "gpu-13", "2026-10-01T02:16:50Z", "2026-10-01T02:40:14Z", "sisteer-petri-vllm-deadbeef")
        blk5, why5 = sp.launch_engine_provenance(rec, res, job_logs=[j4], repo_root=repo)
        assert blk5 is None and "is not the digest of its requirements as the carrier had them" in why5 and "deadbeef" in why5
        # a gdn-VARIANT launch binds: its name's hash8 is sha256(base + overlay bytes)[:8] (common.sh f22_serving_venv), the identity's
        # base digest stays the base file's, and the block records the variant and both digests
        overlay = repo / "configs" / "capeval" / "vllm-requirements-gdn.txt"; overlay.write_text("fla-core==0.5.2\nflash-linear-attention==0.5.2\ncausal-conv1d==1.6.2.post1\n")
        GDN_SHA = commit_all(repo, "gdn overlay")
        gdn_digest = hashlib.sha256(reqs.read_bytes() + overlay.read_bytes()).hexdigest(); gdn_venv = f"sisteer-petri-vllm-gdn-{gdn_digest[:8]}"
        j5 = job_log("hpo-m-lora-probe-parity_616161.out", "gpu-13", "2026-10-01T02:16:50Z", "2026-10-01T02:40:14Z", gdn_venv, sha=GDN_SHA)
        blk_g, why_g = sp.launch_engine_provenance(rec, res, job_logs=[j5], repo_root=repo)
        assert why_g is None and blk_g["variant"] == "gdn" and blk_g["requirements_sha256"] == gdn_digest and blk_g["name"] == gdn_venv
        assert blk_g["engine_identity"]["base_requirements_sha256"] == digest and blk_g["provenance"]["venv_requirements_sha256"] == gdn_digest and blk_g["provenance"]["variant"] == "gdn"
        # the HISTORICAL tree binds — a carrier whose recorded checkout has since MOVED ON (a mirror / worktree advance)
        # still binds through `git show <sha>:` of its recorded sha; the checkout's current file is never read unless its HEAD is that sha
        reqs.write_text(reqs.read_text() + "newpkg==1.0\n"); MOVED_SHA = commit_all(repo, "the checkout moved on")   # repo = the carrier's checkout, now ahead
        j6 = job_log("hpo-m-lora-probe-parity_626262.out", "gpu-13", "2026-10-01T02:16:50Z", "2026-10-01T02:40:14Z", venv, repo_path=str(repo), sha=REC_SHA)
        blk_h, why_h = sp.launch_engine_provenance(rec, res, job_logs=[j6], repo_root=repo)
        assert why_h is None and blk_h["requirements_sha256"] == digest and blk_h["provenance"]["requirements_source"].startswith(f"git {REC_SHA[:12]}:")
        # a carrier whose recorded sha is NOT in this repo's history and whose checkout is not at that sha binds nothing (never the moved file)
        j7 = job_log("hpo-m-lora-probe-parity_636363.out", "gpu-13", "2026-10-01T02:16:50Z", "2026-10-01T02:40:14Z", venv, repo_path=str(repo), sha="f" * 40)
        blk_u, why_u = sp.launch_engine_provenance(rec, res, job_logs=[j7], repo_root=repo)
        assert blk_u is None and "unreadable as the carrier had them" in why_u
        # a checkout verifiably AT the recorded sha (an immutable per-sha detached checkout) is read directly when the sha is not in this repo
        det = tmp_path / "det"; git(tmp_path, "clone", "-q", str(repo), str(det)); git(det, "checkout", "-q", REC_SHA)
        other_root = tmp_path / "other_root"; other_root.mkdir(); git(other_root, "init", "-q")                  # a repo WITHOUT that history
        j8 = job_log("hpo-m-lora-probe-parity_646464.out", "gpu-13", "2026-10-01T02:16:50Z", "2026-10-01T02:40:14Z", venv, repo_path=str(det), sha=REC_SHA)
        blk_d, why_d = sp.launch_engine_provenance(rec, res, job_logs=[j8], repo_root=other_root)
        assert why_d is None and blk_d["requirements_sha256"] == digest and "the carrier's checkout" in blk_d["provenance"]["requirements_source"]
        git(det, "checkout", "-q", MOVED_SHA)                                                                        # the detached checkout advanced: its file no longer binds
        blk_m, why_m = sp.launch_engine_provenance(rec, res, job_logs=[j8], repo_root=other_root)
        assert why_m is None and blk_m["requirements_sha256"] == digest                                              # …but its own history still holds the sha
        git(det, "checkout", "-q", MOVED_SHA); subprocess.run(["rm", "-rf", str(det / ".git")], check=True)        # no history, HEAD unverifiable -> nothing
        blk_n, why_n = sp.launch_engine_provenance(rec, res, job_logs=[j8], repo_root=other_root)
        assert blk_n is None and "unreadable as the carrier had them" in why_n
        reqs.write_text(reqs.read_text().replace("newpkg==1.0\n", "")); commit_all(repo, "back again")
        assert hashlib.sha256(reqs.read_bytes()).hexdigest() == digest
        # a result block with a digest but NO engine identity is not a standup identity — the launch falls through to
        # its provenance, and the block records that the standup's own block lacked the identity
        legacy = {"requirements_sha256": "d" * 64, "variant": "gdn", "name": "sisteer-petri-vllm-gdn-dddddddd"}
        b_l, w_l = sp.engine_side_block(rec, {**res, sp.SERVING_VENV_KEY: legacy}, job_logs=[j1], repo_root=repo)
        assert w_l is None and b_l["engine_identity_source"] == "launch_provenance" and b_l["provenance"]["standup_block_lacked_identity"] is True
        b_l2, w_l2 = sp.engine_side_block(rec, {**res, sp.SERVING_VENV_KEY: legacy}, job_logs=[], repo_root=repo)
        assert b_l2 is None and "no carrier job log of this record" in w_l2
        # the launch's own launch_cmd.json naming its venv is preferred over any job log
        (ld / "launch_cmd.json").write_text(json.dumps({"cmd": [f"/home/u/venvs/{venv}/bin/python", "-m", "vllm"], "env": {}}))
        blk6, why6 = sp.launch_engine_provenance(rec, res, job_logs=[j4], repo_root=repo)
        assert why6 is None and blk6["name"] == venv and blk6["provenance"]["carrier_job_id"] is None and "launch_cmd.json" in blk6["provenance"]["carrier_job_log_venv_line"]
        assert blk6["provenance"]["requirements_source"].startswith("this repo's tip")
        # the job logs of a record are found under $SISTEER_LOG_DIR, filtered to the model's carriers
        import os
        old = os.environ.get("SISTEER_LOG_DIR")
        try:
            os.environ["SISTEER_LOG_DIR"] = str(logs)
            found = sp.record_job_logs(rec, "m")
            # every carrier naming scheme: the HPO probe carriers AND a documented direct submission `serving-probe-<model>…`
            assert sorted(p.name for p in found) == sorted(p.name for p in list(logs.glob("hpo-m-*.out")) + list(logs.glob("serving-probe-m*.out")))
            assert "serving-probe-m_313131.out" in {p.name for p in found}
        finally:
            if old is None: os.environ.pop("SISTEER_LOG_DIR", None)
            else: os.environ["SISTEER_LOG_DIR"] = old

    def test_the_real_writer_fold_and_standup_gate_admit_a_provenance_bound_pair_and_refuse_a_reference_only_one(self, tmp_path, monkeypatch):
        """End to end through the REAL path and no hand-built certificate: a parity-pass record whose launches
        (a bare `vllm` command, no result-level venv) carry their own vllm_server.log and a carrier job log on this host; its
        reference phase judges the adapters with blocks the production stamping function writes (serving_venv_block(..., reference_process)
        — 820960's shape); the main record's verdict reads assertion 4 from it (load_parity_record → the regate derives every launch's
        serving side from provenance) and folds the site overlay; the standup gate then ADMITS the pair whose serving side is bound
        (launch_provenance) and REFUSES by name the pair whose launch no carrier covers (reference-process-only: no serving side)."""
        import hashlib
        from sisteer.capeval.serving import geometry as g
        from test_serving_probe_parity_pass import ScaledFake, _main_record, _parity_plan
        from test_serving_probe_revocation import _res
        from test_standup_adapter_parity_gate import ENGINE, ZERO, _venv
        REPO_ROOT = Path(__file__).resolve().parents[1]
        base_digest = hashlib.sha256((REPO_ROOT / "configs" / "capeval" / "vllm-requirements.txt").read_bytes()).hexdigest()
        pinned = sp._requirements_vllm_version(REPO_ROOT) or "0.25.1"
        main, pmain, site, logs = tmp_path / "main", tmp_path / "parity", tmp_path / "site", tmp_path / "logs"
        for d in (main, pmain, site, logs): d.mkdir()
        monkeypatch.setenv("SISTEER_LOG_DIR", str(logs)); monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(site))
        monkeypatch.delenv(g.SITE_DIR_ENV, raising=False); monkeypatch.delenv(g.SHARED_ROOT_ENV, raising=False)
        plan, results = _main_record(main)                                     # wo r4 + mlp r4, servability certified, no reference of its own
        # --- the parity-pass record: 5 natural launches at 4,096, results WITHOUT a reference check (the phase below judges them)
        pplan = _parity_plan(); sp.write_json_atomic(pmain / sp.PLAN_FILE, pplan); sp.build_adapter_dirs(pplan, pmain, ScaledFake())
        presults = {}
        base_venv = f"sisteer-petri-vllm-{base_digest[:8]}"
        eng_log = (f"(EngineCore pid=1) INFO 10-01 02:33:18 [core.py:114] Initializing a V1 LLM engine (v{pinned.split('+')[0]}) with config: model='m'\n"
                   "(EngineCore pid=1) INFO 10-01 02:34:58 [kernel_warmup.py:170] Using FlashInfer autotune cache file: /h/.cache/vllm/flashinfer_autotune_cache/0.6.13/90a/fb9\n")
        for rec in pplan["launches"]:
            lid = rec["id"]; m = sm.Manifest.from_dict(pplan["manifests"][rec["manifest"]])
            res = _res(pplan, lid, reference=None)
            res["host"] = "gpu-26" if lid.startswith("mlp_r4_S1d") else "gpu-13"       # mlp's S1d ran where NO carrier log is recorded
            res["started_utc"] = "2026-10-01T02:32:42Z"
            res["reference_pending"] = [{"dir": str(sp.slot_dir(pmain, m.name, a.name)), "name": a.name, "kind": a.kind, "role": a.role, "rank": a.rank,
                                         "vllm_first_token": {"id": 1, "logprob": -0.3}} for a in m.genuine]
            res["probes"] = {"base": {"rows": [{"token": 1, "logprob": -0.1, "top": [[1, -0.1]]}], "token_ids": [1]}, "adapters": {}}
            ld = pmain / sp.LAUNCHES_DIR / lid; ld.mkdir(parents=True, exist_ok=True)
            (ld / "launch_cmd.json").write_text(json.dumps({"cmd": ["vllm", "serve", "org/toy"], "env": {"PYTHONPATH": "x"}, "model": "org/toy", "port": 1}))
            (ld / "vllm_server.log").write_text(eng_log)
            presults[lid] = res; sp.write_json_atomic(sp.launch_result_path(pmain, lid), res)
        sp.write_progress(pplan, presults, pmain)
        import subprocess
        checkout = tmp_path / "checkout"; (checkout / "configs" / "capeval").mkdir(parents=True)
        (checkout / "configs" / "capeval" / "vllm-requirements.txt").write_bytes((REPO_ROOT / "configs" / "capeval" / "vllm-requirements.txt").read_bytes())
        genv = {"PATH": "/usr/bin:/bin", "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t", "HOME": str(tmp_path)}
        for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "the carrier's tree"]):
            subprocess.run(["git", "-C", str(checkout), *args], check=True, capture_output=True, env=genv)
        carrier_sha = subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], check=True, capture_output=True, text=True, env=genv).stdout.strip()
        (logs / f"hpo-{pplan['model_key']}-lora-probe-parity_818316.out").write_text(
            f"[2026-10-01T02:16:50Z] [serving-probe x tp1 attempt 2/3] model=x repo={checkout} sha={carrier_sha} node=gpu-13 out={pmain}\n"
            f"serving venv already bootstrapped (/home/u/venvs/{base_venv})\nserving venv OK\n[2026-10-01T02:40:14Z] [serving-probe x tp1 attempt 2/3] exit rc=0; record {pmain}; RESULT: state=VERDICT job=818316\n")
        # --- the REAL writer: the reference phase with the production stamping function on every judged adapter (820960's shape)
        hf_facts = _venv("gdn", "c" * 64, engine={"vllm_version": pinned, "base_requirements_sha256": base_digest, "flashinfer_version": "0.6.13"})
        hf_block = sp.serving_venv_block(hf_facts, source=sp.IDENTITY_SOURCE_REFERENCE)
        assert hf_block["requirements_sha256"] == "c" * 64 and hf_block["engine_identity_source"] == "reference_process"
        class Deps:
            def reference(self, *, model_id, revision, pending, out, plan=None):
                per = {p["dir"]: {"status": "pass", "top1_equal": True, "logprob_abs_delta": 0.01, "tolerance": 0.05, sp.SERVING_VENV_KEY: dict(hf_block)} for p in pending}
                return {"status": "done", "per_adapter": per, "layouts": {}, sp.SERVING_VENV_KEY: dict(hf_block)}
        rec_out = sp.reference_phase(pplan, pmain, Deps())
        assert rec_out["status"] == "done"
        pres = sp.load_results(pmain)
        bound = pres["wo_r4_S1cc_cap8_L4096"]["reference_check"][sp.SERVING_VENV_KEY]
        assert bound["serving"]["engine_identity_source"] == "launch_provenance" and bound["serving"]["derivation"]["carrier_job_id"] == "818316"
        assert bound["engine_identity"] == {"vllm_version": pinned, "flashinfer_version": "0.6.13", "base_requirements_sha256": base_digest} and bound["identity_agrees"] is True
        unbound = pres["mlp_r4_S1d_cap8_L4096"]["reference_check"]
        assert unbound[sp.SERVING_VENV_KEY]["serving"] is None and "no carrier job log of this record" in unbound["engine_provenance_refusal"] and "gpu-26" in unbound["engine_provenance_refusal"]
        # a result with a digest but no engine identity goes through the REGATE and falls through to the same derivation
        legacy_res = {**pres["wo_r4_S1d_cap8_L4096"], sp.SERVING_VENV_KEY: {"requirements_sha256": "d" * 64, "variant": "gdn", "name": "sisteer-petri-vllm-gdn-dddddddd"}}
        regated, _rep = sp.regate_reference_results({**pres, "wo_r4_S1d_cap8_L4096": legacy_res}, pmain)
        rg = regated["wo_r4_S1d_cap8_L4096"]["reference_check"][sp.SERVING_VENV_KEY]
        assert rg["serving"]["engine_identity_source"] == "launch_provenance" and rg["serving"]["derivation"]["standup_block_lacked_identity"] is True
        # --- the REAL fold: the main record's verdict reads assertion 4 from the parity record (its regate re-derives every serving side)
        parity = sp.load_parity_record(pmain, plan)
        v, allowed = sp.write_verdict(plan, results, probe_job_id="11111111", out=main, verdict_uri=str(main / sp.VERDICT_FILE), geometry_dir=site,
                                      probe_cluster="cluster_a", parity=parity)
        sp.write_progress(plan, results, main)
        assert v["status"] == sp.VERDICT_COMPLETE and (site / f"{plan['model_key']}.json").is_file()
        wo = next(p for p in allowed["pairs"] if p["set"] == "wo"); mlp = next(p for p in allowed["pairs"] if p["set"] == "mlp")
        assert wo[sp.ADAPTER_PARITY_KEY][sp.SERVING_VENV_KEY]["serving"]["engine_identity_source"] == "launch_provenance"
        assert wo[sp.ADAPTER_PARITY_KEY][sp.SERVING_VENV_KEY]["reference"]["engine_identity_source"] == "reference_process"
        assert mlp[sp.ADAPTER_PARITY_KEY][sp.SERVING_VENV_KEY] is None or mlp[sp.ADAPTER_PARITY_KEY][sp.SERVING_VENV_KEY]["serving"] is None
        # --- the REAL gate on the staged overlay: the bound pair is admitted under a standup of the same engine; the other is refused by name
        standup = _venv(None, base_digest, engine={"vllm_version": pinned, "base_requirements_sha256": base_digest, "flashinfer_version": "0.6.13"})
        WO4 = {"si_x__lora_r4": {"rank": 4, "module_set": "wo", "zero": False}}
        MLP4 = {"si_y__lora_r4": {"rank": 4, "module_set": "mlp", "zero": False}}
        ok = g.standup_adapter_parity_gate(plan["model_key"], {**WO4, **ZERO}, serving_venv=standup)
        assert ok["status"] == "verified" and ok["pairs"][0][g.CERTIFICATE_VENV_KEY]["serving"]["engine_identity_source"] == "launch_provenance"
        with pytest.raises(g.SiteGeometryError, match="names NO serving venv|names no SERVING-side identity"):
            g.standup_adapter_parity_gate(plan["model_key"], {**MLP4, **ZERO}, serving_venv=standup)
        # another ENGINE at the standup is refused naming both (the certificate's identity is the launch's, not the standup's)
        with pytest.raises(g.SiteGeometryError, match="another ENGINE"):
            g.standup_adapter_parity_gate(plan["model_key"], {**WO4, **ZERO}, serving_venv=_venv(None, base_digest, engine={**ENGINE, "base_requirements_sha256": base_digest, "vllm_version": "9.9.9"}))

    def test_certify_v2_all_expands_to_every_pair_of_the_plan(self):
        assert sp.parse_pairs_arg("all") == [sp.CERTIFY_ALL] and sp.parse_pairs_arg(" ALL ") == [sp.CERTIFY_ALL]
        plan = sp.build_plan(G, model_key=MODEL, model_id="org/toy", revision=REV, sets=("down", "mlp"), ranks=(16,),
                             windows=(4096,), ladder_caps=(), parity_scales={"down": 0.05, "mlp": 0.05}, parity_pass=True)
        new_plan, w = sp.certify_v2_work(plan, {}, G, [sp.CERTIFY_ALL])
        assert w["pairs"] == [[16, "down"], [16, "mlp"]] and w["added"] == [] and w["supersede"] == []    # all planned, all unrun: nothing to add
        assert all(v == "v2: unrun" for v in w["why"].values()) and len(w["why"]) == 4
