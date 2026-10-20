"""The M-RoPE dispatch attestation lifted into ``sisteer.capeval.serving.mrope_attest``: the cluster
standup (``scripts/sdfbase/mt_serving_up.py``) imports the four names back and behaves byte-for-byte;
the tinycap standup asks the same question; the launcher's reader
(``endpoint_mrope_problems``) and the STOP-time record (``stop_report``: ``routed_native_calls`` must
have MOVED for a non-neox M-RoPE model) are pinned here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sisteer.capeval.serving import mrope_attest as mra
from sisteer.capeval.serving import vllm_server as vs

REPO = Path(__file__).resolve().parents[1]
GOOD = {"patch_applied": True, "has_rotary": True, "rotary_layer_index": 0, "rotary_attr": "self_attn",
        "rotary_cls": "MRotaryEmbedding", "is_neox_style": False, "mrope_section": [8, 12, 12],
        "rotary_dim": 64, "dispatch": "forward_cuda", "kernel_reachable": True, "routed_native_calls": 0,
        "needs_patch": True}
NEMO_NO_ROTARY = {"patch_applied": True, "n_layers": 42, "n_layers_with_rotary": 0, "has_rotary": False,
                  "rotary_layer_index": None, "rotary_attr": None, "rotary_cls": None, "needs_patch": False}


def _mt():
    sys.path.insert(0, str(REPO / "scripts/sdfbase"))
    sys.path.insert(0, str(REPO / "scripts"))
    import mt_serving_up as m
    return m


def test_the_cluster_standup_imports_the_shared_attestation_back():
    m = _mt()
    assert m.mrope_attest is mra.mrope_attest
    assert m.mrope_attestation_problems is mra.mrope_attestation_problems
    assert m.MROPE_RPC_METHOD == mra.MROPE_RPC_METHOD == "sisteer_mrope_report"
    assert m.NON_NEOX_MROPE_KEYS == mra.NON_NEOX_MROPE_KEYS == ("glm46vflash",)
    assert mra.NON_NEOX_MROPE_FAMILIES == ("glm4v",)
    # the ONE definition of the judge: the cluster script no longer carries its own body
    src = (REPO / "scripts/sdfbase/mt_serving_up.py").read_text(encoding="utf-8")
    assert "def mrope_attestation_problems" not in src and "def mrope_attest(" not in src
    assert "from sisteer.capeval.serving.mrope_attest import" in src
    # the worker's RPC and the in-process attestor still speak the same name
    lw = (REPO / "src/sisteer/capeval/serving/lens_worker.py").read_text(encoding="utf-8")
    vp = (REPO / "src/sisteer/capeval/serving/vllm_patches.py").read_text(encoding="utf-8")
    assert "def sisteer_mrope_report(self)" in lw and 'llm.collective_rpc("sisteer_mrope_report")' in vp


def test_the_judge_keeps_its_rules():
    assert mra.mrope_attestation_problems([GOOD], "glm46vflash") == []
    assert mra.mrope_attestation_problems([GOOD, dict(GOOD)], "glm46vflash") == []
    bad = {**GOOD, "patch_applied": False}
    assert any("neox-only" in p for p in mra.mrope_attestation_problems([bad], "glm46vflash"))
    assert any("neox-only" in p for p in mra.mrope_attestation_problems([bad], "someothermodel"))
    compiled = {**GOOD, "dispatch": "forward_native", "kernel_reachable": False, "needs_patch": False}
    assert any("not reachable" in p for p in mra.mrope_attestation_problems([compiled], "glm46vflash"))
    assert mra.mrope_attestation_problems([compiled], "qwen35_4b") == []
    neox = {**GOOD, "is_neox_style": True, "needs_patch": False}
    assert mra.mrope_attestation_problems([neox], "qwen35_4b") == []
    assert any("is_neox_style=True" in p for p in mra.mrope_attestation_problems([neox], "glm46vflash"))
    assert any("not MRotaryEmbedding" in p
               for p in mra.mrope_attestation_problems([{**GOOD, "rotary_cls": "RotaryEmbedding"}], "glm46vflash"))
    assert any("errored" in p for p in mra.mrope_attestation_problems([{"patch_applied": True, "error": "x"}], "glm46vflash"))
    assert mra.mrope_attestation_problems([], "glm46vflash") == ["no sisteer_mrope_report report from any worker"]
    assert any("not a mapping" in p for p in mra.mrope_attestation_problems(["oops"], "glm46vflash"))
    # Nemotron-H: attention layers without a rotary — a report, not an error, and no problem for its key
    assert mra.mrope_attestation_problems([NEMO_NO_ROTARY], "nemotron3nano4b") == []
    assert mra.mrope_attestation_problems([NEMO_NO_ROTARY], None) == []


def test_attest_asks_the_rpc_route_and_a_failure_is_a_problem(monkeypatch):
    calls = []

    def fake_rpc(launch, method, args, *, timeout_s, what):
        calls.append((method, args, timeout_s, what))
        return {"results": [GOOD]}
    monkeypatch.setattr(vs, "_post_collective_rpc", fake_rpc)
    att = mra.mrope_attest(object(), "glm46vflash")
    assert att["ok"] is True and att["reports"] == [GOOD] and att["model_key"] == "glm46vflash"
    assert calls == [("sisteer_mrope_report", [], 120.0, "M-RoPE dispatch attestation")]

    def boom(*a, **k):
        raise vs.VllmServerError("HTTP 401 — token refused")
    monkeypatch.setattr(vs, "_post_collective_rpc", boom)
    att = mra.mrope_attest(object(), "glm46vflash", timeout_s=7.0)
    assert att["ok"] is False and att["reports"] == [] and "RPC failed" in att["problems"][0]


def test_stop_report_requires_the_native_route_to_have_moved_for_glm_only():
    ready = {"ok": True, "problems": [], "reports": [GOOD], "rpc_method": "sisteer_mrope_report"}
    moved = {**ready, "reports": [{**GOOD, "routed_native_calls": 4130}]}
    rec = mra.stop_report("glm46vflash", ready, moved)
    assert rec["routed_native_moved"] is True
    assert rec["routed_native_calls_at_ready"] == [0] and rec["routed_native_calls_at_stop"] == [4130]
    still = mra.stop_report("glm46vflash", ready, ready)
    assert still["routed_native_moved"] is False          # traffic never reached the patched route
    # two workers: BOTH must move
    two_ready = {**ready, "reports": [GOOD, GOOD]}
    one_moved = {**ready, "reports": [{**GOOD, "routed_native_calls": 9}, GOOD]}
    assert mra.stop_report("glm46vflash", two_ready, one_moved)["routed_native_moved"] is False
    # a worker count change at STOP is not a proof
    assert mra.stop_report("glm46vflash", two_ready, moved)["routed_native_moved"] is False
    # any other key: the counter is not the proof -> null, recorded
    nemo = {**ready, "reports": [NEMO_NO_ROTARY]}
    assert mra.stop_report("nemotron3nano4b", nemo, nemo)["routed_native_moved"] is None
    assert mra.stop_report("qwen35_4b", ready, ready)["routed_native_moved"] is None
    assert mra.routed_native_calls([GOOD, "oops", NEMO_NO_ROTARY]) == [0, None, None]


def test_endpoint_reader_is_the_launchers_mrope_unattested_belt():
    ok = {"ok": True, "problems": [], "reports": [GOOD], "rpc_method": "sisteer_mrope_report", "model_key": "glm46vflash"}
    assert mra.endpoint_mrope_problems({"mrope_attestation": ok}, "glm46vflash") == []
    absent = mra.endpoint_mrope_problems({"mrope_attestation": None, "mrope_attestation_error": "not asked"}, "glm46vflash")
    assert len(absent) == 1 and "not asked" in absent[0]
    assert any("this launch serves" in p for p in mra.endpoint_mrope_problems({"mrope_attestation": ok}, "qwen35_4b"))
    # a block marked ok whose reports do not satisfy the key's shape is STILL refused (re-judged, never trusted)
    liar = {**ok, "reports": [{**GOOD, "patch_applied": False}]}
    assert any("neox-only" in p for p in mra.endpoint_mrope_problems({"mrope_attestation": liar}, "glm46vflash"))
    assert any("not ok" in p for p in mra.endpoint_mrope_problems({"mrope_attestation": {**ok, "ok": False}}, "glm46vflash"))
    assert any("rpc_method" in p for p in mra.endpoint_mrope_problems({"mrope_attestation": {**ok, "rpc_method": "x"}}, "glm46vflash"))


@pytest.mark.parametrize("key", ["glm46vflash", "qwen35_4b", None])
def test_utc_stamp_and_shape(key, monkeypatch):
    monkeypatch.setattr(vs, "_post_collective_rpc", lambda *a, **k: {"results": [GOOD]})
    att = mra.mrope_attest(object(), key)
    assert set(att) == {"ok", "problems", "reports", "rpc_method", "model_key", "asked_at"}
    assert att["asked_at"].endswith("+00:00")
