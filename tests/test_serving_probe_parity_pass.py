"""The PARITY PASS: assertion 4 at per-set benign synthetic scales on a separate record, read by the main record's verdict.

Probe 805161's scale-100 synthetic adapters flattened its subject's first-token distribution (top-1 p ~ 1 %) and, on the
MLP-bearing sets (every MLP projection of every block adapted), made it chaotic under one bf16 ulp — no rule can decide parity
there. Measured (cluster A 807473 @ scale 1, 807594 @ scale 0.05): wo r4 passes cleanly at scale 1, mlp r4 at scale 0.05,
so the pass exports each set's adapters at ITS benign scale and stands up only the natural launches at the ladder window
(31 on the real plan under the split topology: an S1cc + an S1d per (set, rank), plus the one shared S2). This file pins:
* `--parity-scales` / `--parity-pass` -> plan.synthetic_scale_by_set + provenance, the ladder window alone, no rank ladder;
* `build_adapter_dirs` hands each GENUINE slot its set's scale; bank keys carry the scale (no collision with a scale-100 bank);
* rule (a): |d| <= max(tol, K x base floor, M x one-ulp sensitivity); `parity_weak` / `parity_chaotic` -> `reference_undecidable`,
  a JUDGED status under the hf_peft basis — a DECIDED exclusion of the pair, never `reference_missing`;
* `write_verdict(parity=...)`: each pair's assertion 4 comes from the parity record's natural launch of the same MANIFEST
  name (or, for a manifest the parity record never planned, its GENUINE adapters read BY NAME off a judged parity launch of the
  same (set, rank) carrying every one of them); a parity launch not run, with no such donor, -> `reference_missing` for that
  pair alone; the stamp (`parity_record`) rides verdict + menu;
* a parity-pass record folds no geometry and marks its own menu as a parity summary;
* `reverdict --parity-record` passes through; the carrier's phase `parity` composes the plan flags (executed under bash).
CPU only; fake engines; synthetic ids; no cluster path, no host name."""
from __future__ import annotations

import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pytest

from sisteer.capeval.serving import geometry as g
from sisteer.capeval.serving import standup_manifest as sm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
sys.path.insert(0, str(REPO / "tests"))
import serving_probe as sp  # noqa: E402
from test_serving_probe_revocation import _mint_receipt, _plan, _res  # noqa: E402
from test_serving_probe_smoke import REF_OK, REV, TOY, FakeExporter  # noqa: E402

MODEL = "qwen35_4b_demo"       # the front door's worked example: the probe subject (a key of lora_modules.ARCH_DIMS_BY_MODEL_KEY; a hybrid)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for v in (g.SITE_GEOMETRY_DIR_ENV, g.SITE_DIR_ENV, g.SHARED_ROOT_ENV, "SISTEER_MODELS_TABLE", sp.VISIBLE_DEVICES_ENV, "SLURM_JOB_ID"):
        monkeypatch.delenv(v, raising=False)


# ------------------------------------------------------------------------------------------------ the scale table

def test_parse_parity_scales_default_explicit_and_refusals():
    assert sp.parse_parity_scales(None) == sp.PARITY_SCALES_BY_SET == sp.parse_parity_scales("default") == sp.parse_parity_scales("  ")
    assert sp.parse_parity_scales("wo=1.0, mlp=0.05,qkvo=0.3") == {"wo": 1.0, "mlp": 0.05, "qkvo": 0.3}
    for bad, why in (("wo", "not set=scale"), ("gate=1", "names set"), ("wo=x", "not a number"), ("wo=0", "must be > 0"), ("wo=-1", "must be > 0")):
        with pytest.raises(ValueError, match=why):
            sp.parse_parity_scales(bad)
    assert set(sp.PARITY_SCALES_BY_SET) == set(sm.MODULE_SETS) if hasattr(sm, "MODULE_SETS") else True
    assert sp.PARITY_SCALES_BY_SET == {"wo": 1.0, "qkvo": 0.3, "attn_mlp": 0.05, "mlp": 0.05, "down": 0.05}
    assert "807473" in sp.PARITY_SCALE_PROVENANCE and "807594" in sp.PARITY_SCALE_PROVENANCE and "INTERPOLATED" in sp.PARITY_SCALE_PROVENANCE


def _parity_plan(**kw):
    return sp.build_plan(TOY, model_key=MODEL, model_id="org/toy", revision=REV, sets=("wo", "mlp"), ranks=(4,), windows=(4096,),
                         ladder_caps=(), parity_scales=sp.PARITY_SCALES_BY_SET, parity_pass=True, **kw)


def test_build_plan_parity_pass_carries_the_scale_table_and_provenance_and_refuses_a_set_without_a_scale():
    plan = _parity_plan()
    assert plan["parity_pass"] is True and plan["synthetic_scale_by_set"] == {"wo": 1.0, "mlp": 0.05}
    assert plan["parity_scale_provenance"] == sp.PARITY_SCALE_PROVENANCE
    assert plan["n_launches"] == 5 and all(int(l["L"]) == 4096 and l["kind"] == "natural" for l in plan["launches"])
    assert sorted(l["id"] for l in plan["launches"]) == ["mlp_r4_S1cc_cap8_L4096", "mlp_r4_S1d_cap8_L4096", "mlp_r4_S2_cap8_L4096",
                                                          "wo_r4_S1cc_cap8_L4096", "wo_r4_S1d_cap8_L4096"]
    assert "PARITY PASS" in sp.plan_summary_lines(plan)[0]
    plain = sp.build_plan(TOY, model_key=MODEL, model_id="org/toy", revision=REV, sets=("wo",), ranks=(4,), windows=(4096,), ladder_caps=())
    assert plain["parity_pass"] is False and plain["synthetic_scale_by_set"] is None and plain["parity_scale_provenance"] is None
    with pytest.raises(ValueError, match="no scale for set"):
        sp.build_plan(TOY, model_key=MODEL, model_id="org/toy", revision=REV, sets=("wo", "mlp"), ranks=(4,), windows=(4096,),
                      ladder_caps=(), parity_scales={"wo": 1.0}, parity_pass=True)


def test_plan_cli_parity_pass_forces_the_ladder_window_alone_and_ladder_caps_none_is_empty(tmp_path):
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = sp.main(["plan", "--model-key", MODEL, "--model-id", "org/toy", "--revision", REV, "--out", str(tmp_path / "p"),
                      "--sets", "wo,mlp", "--ranks", "4", "--parity-pass", "--dry-run"])     # default windows + ladder ARE overridden
    assert rc == 0 and "PARITY PASS" in buf.getvalue() and "5 launches" in buf.getvalue()
    rc = sp.main(["plan", "--model-key", MODEL, "--model-id", "org/toy", "--revision", REV, "--out", str(tmp_path / "q"),
                  "--sets", "wo", "--ranks", "4", "--windows", "4096", "--ladder-caps", "none"])
    plan = sp.load_plan(tmp_path / "q")
    assert rc == 0 and plan["ladder_caps"] == [] and plan["n_launches"] == 3 and plan["parity_pass"] is False      # wo: S1cc + S1d + the union
    rc = sp.main(["plan", "--model-key", MODEL, "--model-id", "org/toy", "--revision", REV, "--out", str(tmp_path / "r"),
                  "--sets", "wo,mlp", "--ranks", "4", "--parity-pass", "--parity-scales", "wo=2,mlp=0.1"])
    plan = sp.load_plan(tmp_path / "r")
    assert rc == 0 and plan["synthetic_scale_by_set"] == {"wo": 2.0, "mlp": 0.1} and plan["windows"] == [4096] and plan["ladder_caps"] == []
    with pytest.raises(SystemExit, match="names set"):
        sp.main(["plan", "--model-key", MODEL, "--model-id", "org/toy", "--revision", REV, "--out", str(tmp_path / "s"),
                 "--sets", "wo", "--ranks", "4", "--parity-pass", "--parity-scales", "bogus=1", "--dry-run"])


class ScaledFake(FakeExporter):
    def export_synthetic(self, out_dir, *, modules, rank, per_expert_rank, seed, scale=None):
        self.calls.append(("genuine", tuple(modules), rank, per_expert_rank, seed, scale))
        self._write(out_dir, modules, rank, per_expert_rank)


def test_build_adapter_dirs_exports_each_genuine_slot_at_its_sets_scale_and_bank_keys_never_collide_with_the_scale_100_bank(tmp_path):
    plan = _parity_plan()
    ex = ScaledFake()
    layout = sp.build_adapter_dirs(plan, tmp_path, ex)
    genuine = [c for c in ex.calls if c[0] == "genuine"]
    assert genuine and {c[-1] for c in genuine if "o_proj" in c[1] and len(c[1]) == 1} == {1.0}
    assert {c[-1] for c in genuine if "down_proj" in c[1]} == {0.05}
    assert layout["synthetic_scale_by_set"] == {"wo": 1.0, "mlp": 0.05}
    for name in ("mlp_r4_S1cc", "mlp_r4_S1d"):                                   # the set's genuine slots, split across its two servers
        slots = layout["manifests"][name]["slots"]
        assert {v["synthetic_scale"] for v in slots.values() if v["kind"] == sm.KIND_GENUINE} == {0.05}
        assert all(v["synthetic_scale"] is None for v in slots.values() if v["kind"] != sm.KIND_GENUINE)
    assert all(v["synthetic_scale"] is None for v in layout["manifests"]["mlp_r4_S2"]["slots"].values())   # S2 carries no genuine slot
    # the bank key carries the scale: a parity bank never shares an entry with the scale-100 bank of the main record
    slot = next(a for a in sm.Manifest.from_dict(plan["manifests"]["wo_r4_S1cc"]).adapters if a.kind == sm.KIND_GENUINE)
    assert sp.adapter_bank_key(slot, seed=2, scale=100.0) != sp.adapter_bank_key(slot, seed=2, scale=1.0)
    # a plan without the table still drives the ORIGINAL exporter signature (no scale kwarg) — the main record's path
    plain = sp.build_plan(TOY, model_key=MODEL, model_id="org/toy", revision=REV, sets=("wo",), ranks=(4,), windows=(4096,), ladder_caps=())
    ex0 = FakeExporter()
    lay0 = sp.build_adapter_dirs(plain, tmp_path / "plain", ex0)
    assert ex0.calls and lay0["synthetic_scale_by_set"] is None and lay0["refused"] == {}


# ------------------------------------------------------------------------------------------------ rule (a) + gates

def _hf(top1, lp):
    return {"top1_id": top1, "top1_logprob": lp, "top5": [[top1, lp], [7, lp - 1], [8, lp - 2], [9, lp - 3], [10, lp - 4]]}


def test_first_token_parity_rule_a_takes_the_widest_of_tol_floor_and_sensitivity_terms():
    hf = _hf(29108, -0.1107)
    bare = sp.first_token_parity(hf, {"id": 29108, "logprob": -0.1534})                       # wo @ 0.05 constitution (807594)
    assert bare["status"] == "fail" and bare["tolerance"] == sp.REFERENCE_LOGPROB_TOL and bare["logprob_abs_delta"] == pytest.approx(0.0427)
    floored = sp.first_token_parity(hf, {"id": 29108, "logprob": -0.1534}, floor=0.0179)      # 3 x the measured base floor
    assert floored["status"] == "pass" and floored["tolerance"] == pytest.approx(3 * 0.0179) and floored["tolerance_terms"]["floor_term"] == pytest.approx(0.0537)
    sens = sp.first_token_parity(_hf(31618, -4.6705), {"id": 31618, "logprob": -4.4305}, floor=0.0179, sensitivity=0.139)   # scale-100 wo control
    assert sens["status"] == "pass" and sens["tolerance"] == pytest.approx(0.278) and sens["tolerance_terms"]["sensitivity_term"] == pytest.approx(0.278)
    assert sp.first_token_parity(_hf(1, -1.0), {"id": 2, "logprob": -1.0}, floor=1.0, sensitivity=1.0)["status"] == "fail"   # the greedy token must agree


def test_parity_gate_weak_chaotic_and_decidable():
    ok = {"draws": 3, "flips": 0, "max_shift": 0.006, "shifts": [0.006, 0.001, 0.002]}
    hf_weak = {"first_token_changed": False, "max_abs_delta": 0.05, "common_tokens": 5, "top_set_changed": False}    # HF barely moved off base
    hf_strong = {"first_token_changed": True, "max_abs_delta": 2.3, "common_tokens": 3, "top_set_changed": True}     # HF changed its greedy token
    assert sp.parity_gate({"first_token_changed": False, "max_abs_delta": 1.81}, ok, hf_moved=hf_weak) is None       # mlp @ 0.05: the engine moved
    assert sp.parity_gate({"first_token_changed": True, "max_abs_delta": None}, ok, hf_moved=hf_strong) is None      # wo @ 1: the greedy token moved
    assert sp.parity_gate(None, ok, hf_moved=hf_weak) is None                                                        # an old result without the delta: not weak
    weak = sp.parity_gate({"first_token_changed": False, "max_abs_delta": 0.12}, ok, hf_moved=hf_weak)               # wo @ 0.05 control: BOTH sides weak
    assert weak is not None and weak[0] == "parity_weak" and "0.120 nats" in weak[1] and "0.050 nats" in weak[1]
    # the engine side alone never suppresses — a silently non-applying engine moves ZERO while the HF
    # reference plainly changes its greedy token; that is decidable (and rule (a) then reads it as a mismatch)
    assert sp.parity_gate({"first_token_changed": False, "max_abs_delta": 0.0}, ok, hf_moved=hf_strong) is None
    assert sp.parity_gate({"first_token_changed": False, "max_abs_delta": 0.0}, ok) is None                          # HF unmeasured: not weak
    assert sp.parity_gate({"first_token_changed": False, "max_abs_delta": 0.12}, ok,
                          hf_moved={"first_token_changed": False, "max_abs_delta": 0.9}) is None                     # HF moved 0.9 nats: decidable
    chaos = sp.parity_gate({"first_token_changed": True, "max_abs_delta": None}, {"draws": 3, "flips": 3, "max_shift": 0.609}, hf_moved=hf_strong)   # mlp @ 1
    assert chaos is not None and chaos[0] == "parity_chaotic" and "3/3 draws" in chaos[1]
    assert sp.parity_gate({"first_token_changed": False, "max_abs_delta": 0.12}, {"draws": 3, "flips": 1, "max_shift": 1.0}, hf_moved=hf_weak)[0] == "parity_chaotic"   # chaos wins


SENS_OK = {"draws": 3, "rel": 2 ** -8, "shifts": [0.004, 0.001, 0.002], "flips": 0, "max_shift": 0.004, "n_lora_B": 4}
BASE = {"top1_id": 5, "top1_logprob": -0.4, "top5": [[5, -0.4], [7, -1.4], [8, -2.4], [9, -3.4], [10, -4.4]]}


def test_hf_first_token_moved_reads_the_two_stored_summaries():
    same = sp.hf_first_token_moved(BASE, BASE)
    assert same == {"max_abs_delta": 0.0, "common_tokens": 5, "top_set_changed": False, "first_token_changed": False}
    shifted = {"top1_id": 9, "top1_logprob": -0.2, "top5": [[9, -0.2], [5, -1.9], [7, -2.4], [11, -3.0], [8, -3.1]]}
    mv = sp.hf_first_token_moved(BASE, shifted)
    assert mv["first_token_changed"] is True and mv["max_abs_delta"] == pytest.approx(3.2) and mv["common_tokens"] == 4 and mv["top_set_changed"] is True
    assert sp.hf_first_token_moved(None, shifted) is None and sp.hf_first_token_moved({"top1_id": 5}, shifted) is None


def test_a_silently_non_applying_engine_is_a_mismatch_never_buried_as_weak():
    """The engine returns the BASE's first token (moved 0.000 nats) while HF+PEFT changes its greedy token and is
    stable across the three one-ulp forwards — a wrongly / never applied adapter on the engine. The old gate filed it `parity_weak`
    (reference_undecidable); the rule now reads both sides and rule (a) files the token disagreement as a fail."""
    hf = {"top1_id": 9, "top1_logprob": -0.2, "top5": [[9, -0.2], [5, -1.9], [7, -2.4], [8, -3.4], [10, -4.4]]}
    engine_noop = {"id": 5, "logprob": -0.4}
    moved_zero = {"max_abs_delta": 0.0, "common_tokens": 5, "top_set_changed": False, "first_token_changed": False}
    v = sp.judge_adapter_parity(hf, BASE, engine_noop, moved=moved_zero, sensitivity=SENS_OK, floor=0.0179, attachment={"n_file_tensors": 4, "n_attached": 4})
    assert v["status"] == "fail" and v["top1_equal"] is False and "gate" not in v
    assert v["hf_moved"]["first_token_changed"] is True and v["moved"] == moved_zero and v["hf_base"] == BASE and v["vllm"] == engine_noop
    assert v["tolerance"] == pytest.approx(3 * 0.0179) and v["attachment"] == {"n_file_tensors": 4, "n_attached": 4}
    # both sides weak (wo @ 0.05): still undecidable, with BOTH movements in the reason
    hf_weak = {"top1_id": 5, "top1_logprob": -0.38, "top5": [[5, -0.38], [7, -1.42], [8, -2.4], [9, -3.4], [10, -4.4]]}
    w = sp.judge_adapter_parity(hf_weak, BASE, {"id": 5, "logprob": -0.39}, moved={"max_abs_delta": 0.01, "first_token_changed": False}, sensitivity=SENS_OK, floor=0.0179)
    assert w["status"] == sp.REF_STATUS_UNDECIDABLE and w["gate"] == "parity_weak" and "0.010 nats" in w["reason"] and "0.020 nats" in w["reason"]
    # an engine that applied it (same token, logprob inside the bound) passes; a stored record re-judged gives the same answer (idempotent)
    ok = sp.judge_adapter_parity(hf, BASE, {"id": 9, "logprob": -0.21}, moved={"max_abs_delta": 1.5, "first_token_changed": True}, sensitivity=SENS_OK, floor=0.0179)
    assert ok["status"] == "pass" and sp.regate_adapter_record(ok, floor=0.0179) == ok and sp.regate_adapter_record(v, floor=0.0179) == v
    # a record WITHOUT the raw fields (a fixture, an older record) and a harness `reference_missing` keep their recorded status
    assert sp.regate_adapter_record({"status": "pass"}, floor=None) == {"status": "pass"}
    assert sp.regate_adapter_record({"status": sp.REF_STATUS_MISSING, "hf": hf, "hf_base": BASE, "vllm": engine_noop, "sensitivity": SENS_OK, "reason": "x"}, floor=None)["status"] == sp.REF_STATUS_MISSING


def test_launch_reference_check_a_judged_fail_outranks_undecidable_and_missing_outranks_both():
    items = [{"name": "control__lora", "dir": "/r/c"}, {"name": "constitution__lora", "dir": "/r/k"}, {"name": "difference__lora", "dir": "/r/d"}]
    per = {"/r/c": {"status": "pass"}, "/r/k": {"status": sp.REF_STATUS_UNDECIDABLE, "gate": "parity_chaotic", "reason": "flips 3/3"}, "/r/d": {"status": "fail"}}
    rc = sp.launch_reference_check(items, per)
    assert rc["status"] == "fail" and rc["basis"] == sp.BASIS_HF_PEFT and rc["undecidable"] == {"constitution__lora": "parity_chaotic: flips 3/3"}
    per["/r/d"] = {"status": "pass"}
    assert sp.launch_reference_check(items, per)["status"] == sp.REF_STATUS_UNDECIDABLE
    per["/r/d"] = {"status": sp.REF_STATUS_MISSING, "reason": "PEFT attached 0/20"}
    per["/r/c"] = {"status": "fail"}
    assert sp.launch_reference_check(items, per)["status"] == sp.REF_STATUS_MISSING                     # no evidence beats a fail beats undecidable
    assert sp.launch_reference_check(items, {"/r/c": {"status": "pass"}, "/r/k": {"status": "pass"}, "/r/d": {"status": "pass"}})["status"] == "pass"


def _buried_parity_launch(pmain: Path, plan, results, *, manifest="wo_r4_S1cc"):
    """A parity launch as b497680d2's reference phase WROTE it for a silently non-applying engine: per-adapter `reference_undecidable`
    / `parity_weak` (the engine-only gate), with the RAW comparison fields beside it (hf, hf_base, vllm, moved, sensitivity,
    attachment) and the record-level base floor — enough to re-judge without a model. Returns the launch id."""
    lid = next(r["id"] for r in plan["launches"] if r["manifest"] == manifest)
    hf = {"top1_id": 9, "top1_logprob": -0.2, "top5": [[9, -0.2], [5, -1.9], [7, -2.4], [8, -3.4], [10, -4.4]]}
    per = {"status": sp.REF_STATUS_UNDECIDABLE, "gate": "parity_weak",
           "reason": "the engine moved the first-token top-5 by only 0.000 nats (< 0.3) and left the greedy token unchanged — too weak an effect",
           "hf": hf, "hf_base": BASE, "vllm": {"id": 5, "logprob": -0.4}, "moved": {"max_abs_delta": 0.0, "common_tokens": 5, "top_set_changed": False, "first_token_changed": False},
           "sensitivity": SENS_OK, "attachment": {"n_file_tensors": 4, "n_attached": 4}}
    res = dict(results[lid])
    res["reference_pending"] = [{"name": "control__lora", "dir": str(pmain / "adapters" / "control__lora"), "kind": "genuine"}]
    res["reference_check"] = {"status": sp.REF_STATUS_UNDECIDABLE, "basis": sp.BASIS_HF_PEFT, "undecidable": {"control__lora": f"parity_weak: {per['reason']}"},
                              "per_adapter": {"control__lora": per}, "layouts": {}, "reference_class": "ToyForCausalLM"}
    results[lid] = res
    sp.write_json_atomic(sp.launch_result_path(pmain, lid), res)
    sp.write_json_atomic(pmain / sp.REFERENCE_CHECK_FILE, {"status": "done", "base_floor": 0.0179, "per_adapter": {res["reference_pending"][0]["dir"]: per},
                                                          "parity_rule": {"moved_min": 0.3}})
    sp.write_progress(plan, results, pmain)
    return lid


def test_reverdict_regates_a_stored_parity_record_from_its_raw_fields_without_rerunning_it(tmp_path):
    """The parity record LINE 1 wrote at b497680d2 carries every raw field; `reverdict --parity-record` at this sha re-judges them under
    the current rule IN MEMORY (the record's files are the evidence and stay as written), files the buried mismatch as
    `reference_mismatch:control__lora`, and stamps what changed into the verdict + menu (`parity_record.regated`)."""
    main, pmain = tmp_path / "main", tmp_path / "parity"; main.mkdir(); pmain.mkdir(); site = tmp_path / "site"; site.mkdir()
    plan, results = _main_record(main)
    sp.write_verdict(plan, results, probe_job_id="800", out=main, verdict_uri=str(main / sp.VERDICT_FILE), geometry_dir=site)
    sp.write_progress(plan, results, main)
    pplan, presults = _parity_record(pmain)
    lid = _buried_parity_launch(pmain, pplan, presults)
    stored = json.loads(sp.launch_result_path(pmain, lid).read_text())
    # the pure re-gate: one launch changes, undecidable/parity_weak -> fail, and it is idempotent
    regated, report = sp.regate_reference_results(presults, pmain)
    assert report["n_launches_changed"] == 1 and report["base_floor"] == 0.0179
    assert report["adapters_changed"] == [{"launch": lid, "adapter": "control__lora", "from": sp.REF_STATUS_UNDECIDABLE, "from_gate": "parity_weak", "to": "fail", "to_gate": None}]
    rc = regated[lid]["reference_check"]
    assert rc["status"] == "fail" and rc["regated"]["from_status"] == sp.REF_STATUS_UNDECIDABLE and rc["per_adapter"]["control__lora"]["top1_equal"] is False
    assert rc["per_adapter"]["control__lora"]["hf_moved"]["first_token_changed"] is True
    again, report2 = sp.regate_reference_results(regated, pmain)
    assert report2["n_launches_changed"] == 0 and again[lid]["reference_check"]["status"] == "fail"
    # through the CLI: reverdict --parity-record on the main record (a reopen: the receipt), then the pair is excluded as a mismatch
    gen = sp.result_generation(main)["generation"]
    nonce = _mint_receipt(main, gen)
    assert sp.main(["reverdict", "--record", str(main), "--parity-record", str(pmain), "--reopen-token", nonce, "--geometry-dir", str(site)]) == 0
    v = json.loads((main / sp.VERDICT_FILE).read_text()); menu = json.loads((main / sp.ALLOWED_PAIRS_FILE).read_text())
    wo = next(e for e in v["excluded"] if e["set"] == "wo")
    assert "reference_mismatch:control__lora" in wo["reasons"] and not any(r.startswith("reference_undecidable") for r in wo["reasons"])
    assert [p["set"] for p in menu["pairs"]] == ["mlp"]                                            # mlp's parity launch passed as recorded
    stamp = v[sp.PARITY_RECORD_KEY]
    assert stamp["regated"]["n_launches_changed"] == 1 and stamp["regated"]["adapters_changed"][0]["adapter"] == "control__lora"
    assert menu[sp.PARITY_RECORD_KEY]["regated"] == stamp["regated"] and stamp[sp.RESULT_GENERATION_KEY] == sp.result_generation(pmain)["generation"]
    assert json.loads(sp.launch_result_path(pmain, lid).read_text()) == stored                  # the evidence is untouched
    assert v[sp.ADAPTER_PARITY_KEY]["status"] == "mismatch"
    rows = {r["adapter"]: r for r in v["memo_inputs"]["reference_parity"]}                        # the memo's terms: the parity record's, re-gated
    assert v["memo_inputs"]["reference_parity_source"] == "parity_record" and rows["control__lora"]["status"] == "fail" and rows["control__lora"]["top1_equal"] is False


def test_launch_reference_check_undecidable_is_a_judged_status_under_the_hf_peft_basis_and_missing_still_wins():
    items = [{"name": "control__lora", "dir": "/r/c"}, {"name": "constitution__lora", "dir": "/r/k"}]
    per = {"/r/c": {"status": "pass"}, "/r/k": {"status": sp.REF_STATUS_UNDECIDABLE, "gate": "parity_chaotic", "reason": "flips 3/3"}}
    rc = sp.launch_reference_check(items, per)
    assert rc["status"] == sp.REF_STATUS_UNDECIDABLE and rc["basis"] == sp.BASIS_HF_PEFT
    assert rc["undecidable"] == {"constitution__lora": "parity_chaotic: flips 3/3"}
    per["/r/c"] = {"status": sp.REF_STATUS_MISSING, "reason": "PEFT attached 0/20"}
    rc = sp.launch_reference_check(items, per)
    assert rc["status"] == sp.REF_STATUS_MISSING and rc["basis"] is None                           # no evidence beats undecidable
    rec = sp.finish_reference_record({"/r/k": {"status": sp.REF_STATUS_UNDECIDABLE, "hf": _hf(1, -1.0)}, "/r/p": {"status": "pass", "hf": _hf(2, -1.0)}}, {}, {})
    assert rec["counts"][sp.REF_STATUS_UNDECIDABLE] == 1 and rec["counts"]["pass"] == 1


# ---------------------------------------------------------------------------------- the records: main + parity

def _main_record(out: Path):
    """The main record: wo r4 + mlp r4 at 4,096 / 49,152 (10 launches: S1cc + S1d per set, mlp's S2), every launch passed, NO reference of its own (the
    refusal case) — servability certified, parity to be read from the parity pass."""
    plan = _plan()
    sp.write_json_atomic(out / sp.PLAN_FILE, plan)
    sp.build_adapter_dirs(plan, out, FakeExporter())
    results = {lid: _res(plan, lid, reference=None) for lid in (r["id"] for r in plan["launches"])}
    for lid, res in results.items():
        sp.write_json_atomic(sp.launch_result_path(out, lid), res)
    sp.write_progress(plan, results, out)
    return plan, results


def _parity_record(out: Path, *, reference_by_launch=None, skip=()):
    """The parity record: the 5 natural launches at 4,096 of the same toy (wo S1cc + S1d, mlp S1cc + S1d + S2) at the per-set scales; each
    run launch carries the reference_check `reference_by_launch[lid]` (default REF_OK); launches in `skip` never ran."""
    plan = _parity_plan()
    sp.write_json_atomic(out / sp.PLAN_FILE, plan)
    sp.build_adapter_dirs(plan, out, ScaledFake())
    results = {}
    for rec in plan["launches"]:
        if rec["id"] in skip:
            continue
        m = sm.Manifest.from_dict(plan["manifests"][rec["manifest"]])
        ref = (reference_by_launch or {}).get(rec["id"], REF_OK)
        from test_serving_probe_smoke import _passed
        res = _passed(rec, m, reference=ref)
        results[rec["id"]] = res
        sp.write_json_atomic(sp.launch_result_path(out, rec["id"]), res)
    sp.write_progress(plan, results, out)
    return plan, results


UNDECIDABLE = {"status": sp.REF_STATUS_UNDECIDABLE, "basis": sp.BASIS_HF_PEFT,
               "undecidable": {"constitution__lora": "parity_chaotic: one bf16 ulp flips the greedy token in 3/3 draws"},
               "per_adapter": {"constitution__lora": {"status": sp.REF_STATUS_UNDECIDABLE}, "control__lora": {"status": "pass"}}}


def test_the_verdict_reads_assertion_4_from_the_parity_record_and_stamps_it(tmp_path):
    main, pmain = tmp_path / "main", tmp_path / "parity"; main.mkdir(); pmain.mkdir(); site = tmp_path / "site"; site.mkdir()
    plan, results = _main_record(main)
    # without a parity record: no reference anywhere -> UNCERTIFIED, both pairs unreferenced
    v0, a0 = sp.write_verdict(plan, results, probe_job_id="t", out=main, verdict_uri=str(main / sp.VERDICT_FILE), geometry_dir=site)
    assert v0["status"] == sp.VERDICT_UNCERTIFIED and sorted(map(tuple, v0["unreferenced_pairs"])) == [(4, "mlp"), (4, "wo")] and v0[sp.PARITY_RECORD_KEY] is None
    _parity_record(pmain)
    parity = sp.load_parity_record(pmain, plan)
    v, allowed = sp.write_verdict(plan, results, probe_job_id="t", out=main, verdict_uri=str(main / sp.VERDICT_FILE), geometry_dir=site, parity=parity)
    assert v["status"] == sp.VERDICT_COMPLETE and sorted((p["rank"], p["set"]) for p in allowed["pairs"]) == [(4, "mlp"), (4, "wo")]
    assert v["reference_check_basis"] == sp.BASIS_HF_PEFT and all(p["reference_check_basis"] == sp.BASIS_HF_PEFT for p in v["pairs"])
    assert v[sp.ADAPTER_PARITY_KEY]["status"] == g.PARITY_VERIFIED
    stamp = v[sp.PARITY_RECORD_KEY]
    assert stamp["dir"] == str(pmain) and stamp[sp.RESULT_GENERATION_KEY] == sp.result_generation(pmain)["generation"] and stamp["n_results"] == 5
    assert stamp["synthetic_scale_by_set"] == {"wo": 1.0, "mlp": 0.05} and stamp["scale_provenance"] == sp.PARITY_SCALE_PROVENANCE
    assert stamp["windows"] == [4096] and sorted(stamp["judged_launches"]) == sorted(r["id"] for r in parity["plan"]["launches"])
    assert allowed[sp.PARITY_RECORD_KEY] == stamp and allowed["parity_pass"] is False and v["parity_pass"] is False
    # the MAIN record's own geometry folded (serving-certified pairs, parity aside), its overlay written from ITS launches
    assert v["geometry_fold"]["status"] == "written" and (site / f"{MODEL}.json").is_file()
    assert (main / sp.ALLOWED_PAIRS_FILE).is_file()


def test_a_parity_launch_not_run_is_reference_missing_for_that_pair_alone(tmp_path):
    main, pmain = tmp_path / "main", tmp_path / "parity"; main.mkdir(); pmain.mkdir(); site = tmp_path / "site"; site.mkdir()
    plan, results = _main_record(main)
    # mlp's S1cc launch never ran. Its two adapters could still be read BY NAME off another judged parity launch of (mlp, 4)
    # (`sp._parity_by_adapter_name`), so the pair's other launches judge only what THEY serve: S1d its difference adapter, S2 nothing
    _parity_record(pmain, skip=("mlp_r4_S1cc_cap8_L4096",),
                   reference_by_launch={"mlp_r4_S1d_cap8_L4096": {"status": "pass", "basis": sp.BASIS_HF_PEFT, "per_adapter": {"difference__lora": {"status": "pass"}}},
                                        "mlp_r4_S2_cap8_L4096": {"status": "pass", "basis": sp.BASIS_HF_PEFT, "per_adapter": {}}})
    v, allowed = sp.write_verdict(plan, results, probe_job_id="t", out=main, verdict_uri=str(main / sp.VERDICT_FILE), geometry_dir=site,
                                  parity=sp.load_parity_record(pmain, plan))
    assert v["unreferenced_pairs"] == [[4, "mlp"]] and v["status"] == sp.VERDICT_UNCERTIFIED          # precedence: no menu until every pair has evidence
    ex = next(e for e in allowed["excluded"] if e["set"] == "mlp")
    assert ex["reason"].startswith("reference_missing:") and "parity record: mlp_r4_S1cc_cap8_L4096 not_run" in ex["reason"] and ex["decided"] is True
    assert "mlp_r4_S1d_cap8_L4096" not in ex["reason"] and "mlp_r4_S2_cap8_L4096" not in ex["reason"]   # the launches that ran are not the gap
    wo = next(p for p in v["pairs"] if p["set"] == "wo")
    assert wo[sp.ADAPTER_PARITY_KEY]["status"] == g.PARITY_VERIFIED                                    # wo read its parity fine
    assert [4, "mlp"] in v["serving_certified_pairs"] and [4, "wo"] in v["serving_certified_pairs"]     # serving is untouched by parity
    assert not (main / sp.ALLOWED_PAIRS_FILE).exists()


def _legacy_mlp_r16_manifest():
    from test_serving_probe_legacy_records import _legacy_pair_plan
    plan = _legacy_pair_plan(set_id="mlp", rank=16, windows=(4096,), fallbacks=False)
    return sp.plan_manifest(plan, "mlp_r16_S1"), plan


def _split_parity(*, s1cc_per, s1d_per, s1cc_venv=None, s1d_venv=None, s1cc_und=None, legacy="refused"):
    """The parity record of job 820268's shape for (16, mlp): the legacy S1 launch refused at the ladder window (no reference_check), the
    S1cc launch judged control + constitution, the S1d launch judged difference."""
    p_plan = {"launches": [{"id": "mlp_r16_S1_cap32_L4096", "set": "mlp", "rank": 16, "kind": "natural", "manifest": "mlp_r16_S1"},
                           {"id": "mlp_r16_S1cc_cap16_L4096", "set": "mlp", "rank": 16, "kind": "natural", "manifest": "mlp_r16_S1cc"},
                           {"id": "mlp_r16_S1d_cap32_L4096", "set": "mlp", "rank": 16, "kind": "natural", "manifest": "mlp_r16_S1d"},
                           {"id": "wo_r16_S1cc_cap16_L4096", "set": "wo", "rank": 16, "kind": "natural", "manifest": "wo_r16_S1cc"}]}
    def rc(per, venv, und=None):
        d = {"status": ("fail" if any(v.get("status") == "fail" for v in per.values()) else "pass"), "basis": sp.BASIS_HF_PEFT, "per_adapter": per,
             "undecidable": und or {}, "layouts": {}}
        if venv is not None:
            d[sp.SERVING_VENV_KEY] = venv
        return d
    p_results = {"mlp_r16_S1_cap32_L4096": {"status": legacy},
                 "mlp_r16_S1cc_cap16_L4096": {"status": "passed", "reference_check": rc(s1cc_per, s1cc_venv, s1cc_und)},
                 "mlp_r16_S1d_cap32_L4096": {"status": "passed", "reference_check": rc(s1d_per, s1d_venv)},
                 "wo_r16_S1cc_cap16_L4096": {"status": "passed", "reference_check": rc({"control__lora": {"status": "pass"}, "constitution__lora": {"status": "pass"},
                                                                                        "difference__lora": {"status": "pass"}}, None)}}
    return p_plan, p_results


def test_the_parity_donor_reads_a_legacy_manifests_adapters_across_the_split_servers(tmp_path):
    """Job 820268 (2026-10-01): a legacy S1 manifest certifies (16, mlp) at the long windows; its parity launch refused at the
    ladder window, and its three adapters' judgements live on TWO parity launches (S1cc: control + constitution; S1d: difference). The
    by-name donor reads each adapter from any judged launch of the same (set, rank): certified when all three pass; reference_missing
    (None) when one adapter has NO judged donor; fail when a donor failed anywhere; undecidable when one is; and donors that name
    different venv digests are refused as undecidable by name, never composed into a pass."""
    m, _plan = _legacy_mlp_r16_manifest()
    assert sorted(a.name for a in m.genuine) == ["constitution__lora", "control__lora", "difference__lora"]
    ok = {"control__lora": {"status": "pass"}, "constitution__lora": {"status": "pass"}}
    p_plan, p_results = _split_parity(s1cc_per=ok, s1d_per={"difference__lora": {"status": "pass"}})
    got = sp._parity_by_adapter_name(m, p_plan, p_results)
    assert got is not None
    label, rc = got
    assert rc["status"] == "pass" and sorted(rc["per_adapter"]) == ["constitution__lora", "control__lora", "difference__lora"] and rc["undecidable"] == {}
    assert rc["donor_launches"] == {"control__lora": "mlp_r16_S1cc_cap16_L4096", "constitution__lora": "mlp_r16_S1cc_cap16_L4096", "difference__lora": "mlp_r16_S1d_cap32_L4096"}
    assert rc["restricted_to"] == [a.name for a in m.genuine] and "across 2 launch(es)" in label and "mlp_r16_S1cc_cap16_L4096, mlp_r16_S1d_cap32_L4096" in label
    assert "wo_r16" not in label                                                                   # another set is never a donor
    # the whole source path: parity_reference_sources for a pair certified by the legacy manifest reads the union, not "refused"
    plan = dict(_plan); plan["manifests"] = {**plan["manifests"]}
    srcs = sp.parity_reference_sources(plan, ["mlp_r16_S1_cap32_L4096"], {"plan": p_plan, "results": p_results})
    assert len(srcs) == 1 and srcs[0][1] is not None and srcs[0][1]["status"] == "pass" and "read by name across 2" in srcs[0][0]
    # difference absent from every judged donor -> None -> reference_missing for the pair (the earlier outcome stands when there is a real gap)
    p_plan2, p_results2 = _split_parity(s1cc_per=ok, s1d_per={})
    assert sp._parity_by_adapter_name(m, p_plan2, p_results2) is None
    srcs2 = sp.parity_reference_sources(plan, ["mlp_r16_S1_cap32_L4096"], {"plan": p_plan2, "results": p_results2})
    assert srcs2[0][1] is None and "mlp_r16_S1_cap32_L4096 refused" in srcs2[0][0]
    # a donor fail anywhere -> fail
    p_plan3, p_results3 = _split_parity(s1cc_per=ok, s1d_per={"difference__lora": {"status": "fail"}})
    assert sp._parity_by_adapter_name(m, p_plan3, p_results3)[1]["status"] == "fail"
    # an undecidable adapter on one donor -> undecidable, named
    p_plan4, p_results4 = _split_parity(s1cc_per={**ok, "constitution__lora": {"status": sp.REF_STATUS_UNDECIDABLE}}, s1d_per={"difference__lora": {"status": "pass"}},
                                        s1cc_und={"constitution__lora": "parity_chaotic: x"})
    rc4 = sp._parity_by_adapter_name(m, p_plan4, p_results4)[1]
    assert rc4["status"] == sp.REF_STATUS_UNDECIDABLE and rc4["undecidable"] == {"constitution__lora": "parity_chaotic: x"}
    # donors measured in DIFFERENT venvs: refused as undecidable by name (never a composed pass); the same venv on both composes and binds it
    gdn = {"requirements_sha256": "a" * 64, "variant": "gdn", "name": "n"}; base = {"requirements_sha256": "b" * 64, "variant": None, "name": "m"}
    p_plan5, p_results5 = _split_parity(s1cc_per=ok, s1d_per={"difference__lora": {"status": "pass"}}, s1cc_venv=gdn, s1d_venv=base)
    rc5 = sp._parity_by_adapter_name(m, p_plan5, p_results5)[1]
    assert rc5["status"] == sp.REF_STATUS_UNDECIDABLE and all(v.startswith("parity_venv_disagreement:") for v in rc5["undecidable"].values()) and rc5[sp.SERVING_VENV_KEY] is None
    p_plan6, p_results6 = _split_parity(s1cc_per=ok, s1d_per={"difference__lora": {"status": "pass"}}, s1cc_venv=gdn, s1d_venv=dict(gdn))
    rc6 = sp._parity_by_adapter_name(m, p_plan6, p_results6)[1]
    assert rc6["status"] == "pass" and rc6[sp.SERVING_VENV_KEY] == gdn
    # a single launch carrying everything still reads as before (one donor, no "across")
    p_plan7, p_results7 = _split_parity(s1cc_per={**ok, "difference__lora": {"status": "pass"}}, s1d_per={})
    label7, rc7 = sp._parity_by_adapter_name(m, p_plan7, p_results7)
    assert rc7["status"] == "pass" and "across" not in label7 and rc7["donor_launch"] == "mlp_r16_S1cc_cap16_L4096"


def test_an_undecidable_parity_adapter_excludes_the_pair_as_decided_and_keeps_its_serving_certification(tmp_path):
    main, pmain = tmp_path / "main", tmp_path / "parity"; main.mkdir(); pmain.mkdir(); site = tmp_path / "site"; site.mkdir()
    plan, results = _main_record(main)
    _parity_record(pmain, reference_by_launch={"mlp_r4_S1cc_cap8_L4096": UNDECIDABLE})
    v, allowed = sp.write_verdict(plan, results, probe_job_id="t", out=main, verdict_uri=str(main / sp.VERDICT_FILE), geometry_dir=site,
                                  parity=sp.load_parity_record(pmain, plan))
    assert v["status"] == sp.VERDICT_COMPLETE and [(p["rank"], p["set"]) for p in allowed["pairs"]] == [(4, "wo")]
    ex = next(e for e in allowed["excluded"] if e["set"] == "mlp")
    assert ex["reason"] == "reference_undecidable:constitution__lora" and ex["decided"] is True and v["unreferenced_pairs"] == []
    mlp = next(p for p in v[sp.ADAPTER_PARITY_KEY]["per_pair"] if p["set"] == "mlp")
    assert mlp["status"] == g.PARITY_UNVERIFIED and mlp["mismatched_adapters"] == []
    assert [4, "mlp"] in v["serving_certified_pairs"]                                                  # a parity reason never touches serving
    assert v["geometry_fold"]["status"] == "written" and (main / sp.ALLOWED_PAIRS_FILE).is_file()
    assert "mlp" in {s for s in (v["geometry_fold"].get("sets") or {})}                                # the geometry still covers mlp


def test_a_parity_pass_record_folds_no_geometry_and_marks_its_menu_as_a_parity_summary(tmp_path):
    pmain = tmp_path / "parity"; pmain.mkdir(); site = tmp_path / "site"; site.mkdir()
    plan, results = _parity_record(pmain)
    v, allowed = sp.write_verdict(plan, results, probe_job_id="t", out=pmain, verdict_uri=str(pmain / sp.VERDICT_FILE), geometry_dir=site)
    assert v["status"] == sp.VERDICT_COMPLETE and v["parity_pass"] is True and allowed["parity_pass"] is True
    assert v["geometry_fold"]["status"] == "skipped" and "main record" in v["geometry_fold"]["reason"]
    assert not (site / f"{MODEL}.json").exists() and not (pmain / sp.GEOMETRY_RECORD_FILE).exists()
    memo = sp.render_memo(v)
    assert "This record IS a parity pass" in memo


def test_load_parity_record_refuses_a_non_parity_record_and_another_checkpoint(tmp_path):
    main, pmain, other = tmp_path / "main", tmp_path / "parity", tmp_path / "other"; main.mkdir(); pmain.mkdir(); other.mkdir()
    plan, _ = _main_record(main)
    with pytest.raises(ValueError, match="not written with --parity-pass"):
        sp.load_parity_record(main, plan)                                  # the main record itself is not a parity pass
    _parity_record(pmain)
    assert sp.load_parity_record(pmain, plan)["plan"]["parity_pass"] is True
    with pytest.raises(ValueError, match="revision"):
        sp.load_parity_record(pmain, {**plan, "revision": "f" * 40})
    with pytest.raises(ValueError, match="not a probe record"):
        sp.load_parity_record(other, plan)


def test_reverdict_reads_the_parity_record_and_writes_the_menu_with_the_stamp(tmp_path, monkeypatch):
    main, pmain = tmp_path / "main", tmp_path / "parity"; main.mkdir(); pmain.mkdir(); site = tmp_path / "site"; site.mkdir()
    plan, results = _main_record(main)
    sp.write_verdict(plan, results, probe_job_id="800", out=main, verdict_uri=str(main / sp.VERDICT_FILE), geometry_dir=site)   # UNCERTIFIED
    sp.write_progress(plan, results, main)
    _parity_record(pmain)
    gen = sp.result_generation(main)["generation"]
    with pytest.raises(SystemExit, match="revocation receipt"):                    # the main record is complete: a reverdict is a reopen
        sp.main(["reverdict", "--record", str(main), "--parity-record", str(pmain), "--geometry-dir", str(site)])
    nonce = _mint_receipt(main, gen)
    rc = sp.main(["reverdict", "--record", str(main), "--parity-record", str(pmain), "--reopen-token", nonce, "--geometry-dir", str(site)])
    assert rc == 0
    v = json.loads((main / sp.VERDICT_FILE).read_text()); menu = json.loads((main / sp.ALLOWED_PAIRS_FILE).read_text())
    assert v["status"] == sp.VERDICT_COMPLETE and menu[sp.PARITY_RECORD_KEY]["dir"] == str(pmain) and len(menu["pairs"]) == 2
    memo = (main / sp.MEMO_FILE).read_text()
    assert "8b. Adapter parity basis" in memo and "wo` | 1.0" in memo and "807594" in memo
    assert sp.result_generation(main)["generation"] == gen                          # a reverdict rewrites artifacts, never results
    with pytest.raises(SystemExit, match="not written with --parity-pass"):        # the main record is no parity pass (verdict needs no receipt)
        sp.main(["verdict", "--model-key", MODEL, "--out", str(main), "--parity-record", str(main), "--geometry-dir", str(site)])


# ------------------------------------------------------------------------------------------------------ the carrier

def test_the_carrier_composes_the_parity_plan_flags_for_phase_parity(tmp_path):
    text = (REPO / "scripts/sbatch/serving_probe.sbatch").read_text()
    assert "full|reference|parity)" in text
    i = text.index('PLAN_ARGS="--sets $SETS --ranks $RANKS --tp $TP"')
    j = text.index("srun --ntasks=1", i)
    block = text[i:j]

    def run(phase, certify=None):
        script = ("set -u\nlog(){ echo \"$@\"; }\nSETS=wo,mlp; RANKS=4; TP=1; OUT=/r\n" + f"PHASE={phase}\n"
                  + (f"SERVING_PROBE_CERTIFY_V2={certify}\n" if certify else "unset SERVING_PROBE_CERTIFY_V2\n") + block
                  + '\nprintf "PLAN=%s\\n" "$PLAN_ARGS"\nprintf "EXTRA=%s\\n" "${RUN_EXTRA[*]:-}"\n')
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    r = run("parity")
    assert r.returncode == 0 and "PLAN=--sets wo,mlp --ranks 4 --tp 1 --parity-pass --parity-scales default" in r.stdout and "EXTRA=--no-throughput\n" in r.stdout, r
    r = run("full")
    assert r.returncode == 0 and "PLAN=--sets wo,mlp --ranks 4 --tp 1\n" in r.stdout and "EXTRA=\n" in r.stdout, r
    # job 819493: the certificate request RIDES the parity phase's extras — the phase block appends --no-throughput and never
    # overwrites the --certify-v2 queued from SERVING_PROBE_CERTIFY_V2 (the overwrite is what made attempt 3/3 measure nothing)
    r = run("parity", certify="all")
    assert r.returncode == 0 and "EXTRA=--certify-v2 all --no-throughput\n" in r.stdout and "certify-v2 pairs: all" in r.stdout, r
    r = run("full", certify="16:down,16:mlp")
    assert r.returncode == 0 and "EXTRA=--certify-v2 16:down,16:mlp\n" in r.stdout, r
    run_line = text[text.index("srun --ntasks=1"):text.index("WORK=$!")]
    assert '--plan-args "$PLAN_ARGS"' in run_line and '"${RUN_EXTRA[@]}"' in run_line and '"${RESUME[@]}"' in run_line


def test_the_carriers_time_budget_reads_zero_padded_limits_in_base_ten():
    """Job 820268 (2026-10-01, attempt 5/5 at --time 08:00:00): bash read the `08` field as octal, `limit_to_s` printed nothing and the
    carrier fell back to the 4-hour budget inside an 8-hour allocation. Every field is base 10 now; malformed input still yields 0."""
    text = (REPO / "scripts/sbatch/serving_probe.sbatch").read_text()
    i = text.index("limit_to_s(){"); j = text.index("\n", text.index("case $# in", i)) + 1
    fn = text[i:j]
    cases = {"08:00:00": 28800, "04:00:00": 14400, "1-08:30:00": 86400 + 8 * 3600 + 30 * 60, "1-02:03:04": 86400 + 7384, "09:05": 9 * 60 + 5, "00:00:07": 7,
             "2-00:00:00": 172800, "8h": 0, "": 0, "aa:bb:cc": 0}
    script = fn + "\n" + "\n".join(f'printf "%s=%s\\n" {t!r} "$(limit_to_s {t!r})"' for t in cases)
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert r.returncode == 0 and "value too great" not in r.stderr and r.stderr == "", r
    got = dict(line.split("=", 1) for line in r.stdout.splitlines())
    assert got == {k: str(v) for k, v in cases.items()}, got
    # the budget line that consumes it: an 8-hour limit yields the 7h15 budget, never the 14400 fallback
    k = text.index('LIMIT_S="$(limit_to_s')
    budget = fn + "\nTIMELIMIT=08:00:00\n" + text[k:text.index("\n", text.index("BUDGET_S=$((", k))] + '\necho "$LIMIT_S $BUDGET_S"'
    r = subprocess.run(["bash", "-c", budget], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "28800 26100" and r.stderr == "", r


def test_the_attempt_record_carries_an_operator_cap_widen(tmp_path):
    """SERVING_PROBE_CAP_WIDEN (the front door's --probe-max-attempts note) is recorded on probe_attempt.json as `cap_widen`, null
    without it; quotes and backslashes in the note never break the JSON."""
    text = (REPO / "scripts/sbatch/serving_probe.sbatch").read_text()
    fn = text[text.index("attempt_record_write(){"):text.index("\n}\n", text.index("attempt_record_write(){")) + 3]
    base = ("set -u\nlog(){ echo \"$@\"; }\nFINAL_RC=\nOUT=%s\nATTEMPT=6; MAX_ATTEMPTS=6; NGPU=4; PARALLEL=auto; MODE=tp1; TP=1; SETS=wo; RANKS=4; MODEL_KEY=m; SERVED_ID=org/toy; "
            "REVISION=0; HEAD_SHA=abc; COMMENT=sha:abc,nonce:0123456789ab; TIMELIMIT=08:00:00\n" % tmp_path) + fn
    env0 = {"PATH": "/usr/bin:/usr/local/bin:/bin"}
    # the record's previous attempt (cap 5, no history) is what the first widen reads its `from` off
    (tmp_path / "probe_attempt.json").write_text(json.dumps({"version": 1, "attempt": 5, "max_attempts": 5}))
    r = subprocess.run(["bash", "-c", base.replace("ATTEMPT=6; MAX_ATTEMPTS=6", "ATTEMPT=5; MAX_ATTEMPTS=5") + "\nattempt_record_write"], capture_output=True, text=True, env=env0)
    assert r.returncode == 0, r
    rec0 = json.loads((tmp_path / "probe_attempt.json").read_text())
    assert rec0["cap_widen"] is None and rec0["cap_history"] == []                                  # no widen: nothing appended, history kept
    note = 'cap widened 5->6 by the operator (03_hpo.sh --probe-max-attempts 6): D549 "quoted" back\\slash'
    r = subprocess.run(["bash", "-c", base + "\nattempt_record_write"], capture_output=True, text=True, env={**env0, "SERVING_PROBE_CAP_WIDEN": note})
    assert r.returncode == 0, r
    rec = json.loads((tmp_path / "probe_attempt.json").read_text())
    assert rec["cap_widen"] == "cap widened 5->6 by the operator (03_hpo.sh --probe-max-attempts 6): D549 quoted backslash" and rec["max_attempts"] == 6
    # the widen is DURABLE — cap_history carries {from, to, attempt, utc, host, note}, and a later attempt without a widen
    # rewrites the record with cap_widen null but the history intact (the JSON-escaped note keeps its quotes and backslash there)
    assert len(rec["cap_history"]) == 1 and rec["cap_history"][0]["from"] == 5 and rec["cap_history"][0]["to"] == 6 and rec["cap_history"][0]["attempt"] == 6
    assert rec["cap_history"][0]["note"] == note and rec["cap_history"][0]["utc"].endswith("Z") and rec["cap_history"][0]["host"]
    r = subprocess.run(["bash", "-c", base.replace("ATTEMPT=6; MAX_ATTEMPTS=6", "ATTEMPT=7; MAX_ATTEMPTS=6") + "\nattempt_record_write"], capture_output=True, text=True, env=env0)
    assert r.returncode == 0, r
    rec2 = json.loads((tmp_path / "probe_attempt.json").read_text())
    assert rec2["cap_widen"] is None and rec2["max_attempts"] == 6 and rec2["cap_history"] == rec["cap_history"]
    # a second widen appends (from the record's cap 6 to 7)
    r = subprocess.run(["bash", "-c", base.replace("ATTEMPT=6; MAX_ATTEMPTS=6", "ATTEMPT=7; MAX_ATTEMPTS=7") + "\nattempt_record_write"], capture_output=True, text=True,
                       env={**env0, "SERVING_PROBE_CAP_WIDEN": "cap widened 6->7 by the operator: second widen"})
    rec3 = json.loads((tmp_path / "probe_attempt.json").read_text())
    assert [ (h["from"], h["to"], h["attempt"]) for h in rec3["cap_history"] ] == [(5, 6, 6), (6, 7, 7)]
    # a self-resubmitted successor inherits SERVING_PROBE_CAP_WIDEN through Slurm's implicit export — the cap did NOT
    # change (7 -> 7), so nothing is appended, cap_widen is null, the history is carried unchanged; twice
    for att_n in ("8", "9"):
        r = subprocess.run(["bash", "-c", base.replace("ATTEMPT=6; MAX_ATTEMPTS=6", f"ATTEMPT={att_n}; MAX_ATTEMPTS=7") + "\nattempt_record_write"], capture_output=True, text=True,
                           env={**env0, "SERVING_PROBE_CAP_WIDEN": "cap widened 6->7 by the operator: second widen"})
        assert r.returncode == 0, r
        succ = json.loads((tmp_path / "probe_attempt.json").read_text())
        assert succ["cap_widen"] is None and succ["max_attempts"] == 7 and succ["attempt"] == int(att_n) and succ["cap_history"] == rec3["cap_history"], succ
    # a note whose transition is not this record's (6->8 claimed, the record moves 7->8) refuses with the file untouched
    before = (tmp_path / "probe_attempt.json").read_bytes()
    r = subprocess.run(["bash", "-c", base.replace("ATTEMPT=6; MAX_ATTEMPTS=6", "ATTEMPT=10; MAX_ATTEMPTS=8") + "\nattempt_record_write"], capture_output=True, text=True,
                       env={**env0, "SERVING_PROBE_CAP_WIDEN": "cap widened 6->8 by the operator: stale note"})
    assert r.returncode == 2 and "not this record's" in r.stdout + r.stderr and (tmp_path / "probe_attempt.json").read_bytes() == before, r
    # an EXISTING file that is damaged — not JSON, not an object, a cap_history of the wrong shape, a max_attempts
    # that is not a positive integer — refuses with the file byte-identical, with and without a widen; only a MISSING file is fresh
    for damaged in ('{"version": 1, "attempt": 6, "max_attempts": 6, "cap_history": [', "[1, 2, 3]",
                    '{"version": 1, "attempt": 6, "max_attempts": 6, "cap_history": "5->6"}', '{"version": 1, "attempt": 6, "max_attempts": "six"}',
                    '{"version": 1, "attempt": 6}'):
        (tmp_path / "probe_attempt.json").write_text(damaged)
        for extra in ({}, {"SERVING_PROBE_CAP_WIDEN": "cap widened 6->7 by the operator: x"}):
            r = subprocess.run(["bash", "-c", base.replace("ATTEMPT=6; MAX_ATTEMPTS=6", "ATTEMPT=7; MAX_ATTEMPTS=7") + "\nattempt_record_write"], capture_output=True, text=True, env={**env0, **extra})
            assert r.returncode == 2 and "refusing to rewrite a damaged attempt record" in r.stdout + r.stderr, (damaged, extra, r)
            assert (tmp_path / "probe_attempt.json").read_text() == damaged, damaged
    # a pre-cap_history record (no key at all) is carried: an empty history, then the widen appends from its cap
    (tmp_path / "probe_attempt.json").write_text(json.dumps({"version": 1, "attempt": 6, "max_attempts": 6}))
    r = subprocess.run(["bash", "-c", base.replace("ATTEMPT=6; MAX_ATTEMPTS=6", "ATTEMPT=7; MAX_ATTEMPTS=8") + "\nattempt_record_write"], capture_output=True, text=True,
                       env={**env0, "SERVING_PROBE_CAP_WIDEN": "cap widened 6->8 by the operator: y"})
    assert r.returncode == 0 and [(h["from"], h["to"]) for h in json.loads((tmp_path / "probe_attempt.json").read_text())["cap_history"]] == [(6, 8)], r
    (tmp_path / "probe_attempt.json").write_text(json.dumps(rec3))                                      # restore for the python-failure cases below
    # a widen whose history cannot be recorded (python3 broken) refuses the attempt by name; a plain attempt still writes an empty history
    nopy = tmp_path / "nopy"; nopy.mkdir(); (nopy / "python3").write_text("#!/bin/sh\nexit 127\n"); (nopy / "python3").chmod(0o755)
    env_nopy = {"PATH": f"{nopy}:/usr/bin:/bin"}
    r = subprocess.run(["bash", "-c", base + "\nattempt_record_write"], capture_output=True, text=True, env={**env_nopy, "SERVING_PROBE_CAP_WIDEN": "x"})
    assert r.returncode == 2 and "cannot read/record the cap history" in r.stdout + r.stderr, r
    assert json.loads((tmp_path / "probe_attempt.json").read_text()) == rec3                           # the refused attempt wrote nothing
    # a PLAIN attempt that cannot read the existing record's history refuses too (a rewrite that drops cap_history is the hole); only a
    # fresh record with no attempt file starts an empty history without python
    r = subprocess.run(["bash", "-c", base.replace("ATTEMPT=6; MAX_ATTEMPTS=6", "ATTEMPT=7; MAX_ATTEMPTS=7") + "\nattempt_record_write"], capture_output=True, text=True, env=env_nopy)
    assert r.returncode == 2 and "cannot read/record the cap history" in r.stdout + r.stderr and json.loads((tmp_path / "probe_attempt.json").read_text()) == rec3, r
    fresh = tmp_path / "fresh"; fresh.mkdir()
    r = subprocess.run(["bash", "-c", base.replace(f"OUT={tmp_path}", f"OUT={fresh}") + "\nattempt_record_write"], capture_output=True, text=True, env=env_nopy)
    assert r.returncode == 0 and json.loads((fresh / "probe_attempt.json").read_text())["cap_history"] == [], r


def test_a_parity_pass_carriers_self_resubmission_hands_the_phase_on_as_an_env_prefix(tmp_path):
    """The successor of a parity-pass attempt inherits the phase through SERVING_PROBE_PHASE (an env prefix beside SERVING_PROBE_OUT,
    Slurm's implicit export — never --export=, never a new positional: positional 10 stays unhanded, the successor line stays as the
    smoke suite pins it), and the carrier reads positional 11 first, the env second, `full` last."""
    import os
    text = (REPO / "scripts/sbatch/serving_probe.sbatch").read_text()
    assert 'PHASE="${11:-${SERVING_PROBE_PHASE:-full}}"' in text
    frag = text[text.index("escalate(){"):text.index("WORK=\"\"\nhandled=0")]
    bin_dir = tmp_path / "bin"; bin_dir.mkdir()
    calls = tmp_path / "calls"
    (bin_dir / "sbatch").write_text(f'#!/bin/bash\nprintf "PHASE_ENV=%s OUT_ENV=%s NARGS=%s LAST=%s\\n" "${{SERVING_PROBE_PHASE:-unset}}" "${{SERVING_PROBE_OUT:-unset}}" "$#" "${{!#}}" > "{calls}"; echo "999;cluster"\n')
    (bin_dir / "sbatch").chmod(0o755)

    def run(phase):
        script = (f'set -u; PHASE={phase}; OUT="{tmp_path}"; SLURM_JOB_NAME=x; ATTEMPT=1; MAX_ATTEMPTS=3; SCRIPT_PATH="{tmp_path / "s.sh"}"; '
                  'ESCALATE_QOS=high; TIMELIMIT=02:00:00; NGPU=2; PARALLEL=2; STDOUT_PATH=; HEAD_SHA=abc; COMMENT=sha:abc,nonce:0123456789ab; MODEL_KEY=k; SERVED_ID=o/t; REVISION=r; '
                  'SETS=wo; RANKS=4; MODE=tp1; SUCCESSOR=; log(){ echo "[log] $*"; }; stop_requested(){ return 1; }\n'
                  + frag + '\nescalate "why"; echo "SUCCESSOR=$SUCCESSOR"')
        (tmp_path / "s.sh").write_text("#!/bin/bash\n")
        r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60,
                           env={"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp_path)})
        assert r.returncode == 0 and "SUCCESSOR=999" in r.stdout, r.stdout + r.stderr
        return calls.read_text().strip()
    assert run("parity") == f"PHASE_ENV=parity OUT_ENV={tmp_path} NARGS=16 LAST=2"      # …the same 9 positionals + flags; the phase in the env
    assert run("full") == f"PHASE_ENV=full OUT_ENV={tmp_path} NARGS=16 LAST=2"
    # the carrier's own read: positional 11 wins, then the env, then full
    head = text[text.index('PHASE="${11:-'):text.index("esac", text.index('PHASE="${11:-')) + 4]
    for argv, env, want in ((["a"] * 10 + ["parity"], {}, "parity"), (["a"] * 9, {"SERVING_PROBE_PHASE": "parity"}, "parity"),
                            (["a"] * 10 + ["reference"], {"SERVING_PROBE_PHASE": "parity"}, "reference"), (["a"] * 9, {}, "full")):
        r = subprocess.run(["bash", "-c", "set -u\n" + head + '\necho "PHASE=$PHASE"', "x", *argv], capture_output=True, text=True,
                           env={"PATH": os.environ["PATH"], **env})
        assert r.returncode == 0 and f"PHASE={want}" in r.stdout, (argv, env, r.stdout, r.stderr)
