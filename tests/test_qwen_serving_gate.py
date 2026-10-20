"""The qwen gate driver's own gating rules — the strictness that was
missing when seven scale-1 psr conditions passed while validating
nothing. Pure-logic tests: no GPU, no engine."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location(
    "_qwen_serving_gate", ROOT / "scripts" / "qwen_serving_gate.py")
qg = importlib.util.module_from_spec(_spec)
sys.modules["_qwen_serving_gate"] = qg
_spec.loader.exec_module(qg)


def _arms():
    return [{"cell_id": "safety_refusal__psr_kl_all"},
            {"cell_id": "safety_refusal__lora_r4"}]


def _good_result():
    bites = {
        "safety_refusal__psr_kl_all":
            {"n": 8, "n_differing": 5, "scale0_identical_to_unsteered": 8},
        "safety_refusal__lora_r4":
            {"n": 8, "n_differing": 2, "scale0_identical_to_unsteered": 8},
    }
    return {
        "stage": "completed",
        "install": {"oproj_layers": [3, 7, 11, 15, 19, 23, 27, 31]},
        "response_bites": bites,
        "response_errors": {},
        "response_served_arm_ids": list(bites),
        "geometry": {"kv_cache": {"covers_full_budget": True,
                                  "kv_cache_tokens": 1357531,
                                  "full_length_concurrency": 82.86}},
    }


OPROJ = (3, 7, 11, 15, 19, 23, 27, 31)


def test_verdict_passes_on_full_evidence():
    v = qg.verdict_from_worker(_good_result(), _arms(), OPROJ)
    assert v["problems"] == []


def test_verdict_refuses_a_biteless_arm():
    r = _good_result()
    r["response_bites"]["safety_refusal__lora_r4"]["n_differing"] = 0
    v = qg.verdict_from_worker(r, _arms(), OPROJ)
    assert any("does not BITE" in p for p in v["problems"])


def test_verdict_refuses_a_leaking_scale0_twin():
    r = _good_result()
    r["response_bites"]["safety_refusal__psr_kl_all"][
        "scale0_identical_to_unsteered"] = 7
    v = qg.verdict_from_worker(r, _arms(), OPROJ)
    assert any("NOT a no-op" in p for p in v["problems"])


def test_verdict_refuses_missing_evidence_not_a_pass():
    """The campaign rule: a missing probe is a failure. An arm with no
    hardware evidence at all must fail the gate, never slide by."""
    r = _good_result()
    del r["response_bites"]["safety_refusal__lora_r4"]
    r["response_served_arm_ids"] = ["safety_refusal__psr_kl_all"]
    v = qg.verdict_from_worker(r, _arms(), OPROJ)
    assert any("no hardware evidence" in p for p in v["problems"])


def test_verdict_refuses_an_errored_arm_despite_exit0():
    """The response phase is 'advisory' in the llama smoke — it records
    an arm's exception and still exits 0. Here an errored arm FAILS."""
    r = _good_result()
    r["response_errors"] = {"safety_refusal__psr_kl_all":
                            {"error": "LensWorkerError: header mismatch"}}
    v = qg.verdict_from_worker(r, _arms(), OPROJ)
    assert any("RAISED on hardware" in p for p in v["problems"])


def test_verdict_refuses_oproj_drift():
    """The worker asserts only the BLOCK-site install half (the llama
    smoke had no o_proj arm); the driver owns the o_proj half."""
    r = _good_result()
    r["install"]["oproj_layers"] = [3, 7]
    v = qg.verdict_from_worker(r, _arms(), OPROJ)
    assert any("oproj_layers" in p for p in v["problems"])


def test_verdict_refuses_uncovered_kv_and_wrong_stage():
    r = _good_result()
    r["geometry"]["kv_cache"]["covers_full_budget"] = False
    r["stage"] = "engine_init"
    v = qg.verdict_from_worker(r, _arms(), OPROJ)
    assert any("ceiling, not a grant" in p for p in v["problems"])
    assert any("stage" in p for p in v["problems"])


def test_bank_cells_refuses_record_only_arm(tmp_path):
    """A record-banked arm (lora_r16) has no registered object kind and
    must refuse loudly, never be gated as if matrix-wired."""
    d = tmp_path / "safety_refusal__lora_r16"
    d.mkdir()
    (d / "object.pt").write_bytes(b"x")
    (d / "provenance.json").write_text(json.dumps(
        {"model": "TESTORG/model", "model_revision": "ab" * 20}))
    with pytest.raises(qg.QwenGateError, match="not matrix-wired"):
        qg.bank_cells(tmp_path, ["safety_refusal__lora_r16"])


def test_bank_cells_refuses_missing_object(tmp_path):
    with pytest.raises(qg.QwenGateError, match="lacks object.pt"):
        qg.bank_cells(tmp_path, ["safety_refusal__psr_kl_all"])


def test_bank_cells_refuses_model_disagreement(tmp_path):
    for cid, rev in (("safety_refusal__psr_kl_all", "aa" * 20),
                     ("si_constitution__psr_kl_all", "bb" * 20)):
        d = tmp_path / cid
        d.mkdir()
        (d / "object.pt").write_bytes(b"x")
        (d / "provenance.json").write_text(json.dumps(
            {"model": "TESTORG/model", "model_revision": rev}))
    with pytest.raises(qg.QwenGateError, match="disagree"):
        qg.bank_cells(tmp_path, ["safety_refusal__psr_kl_all",
                                 "si_constitution__psr_kl_all"])
