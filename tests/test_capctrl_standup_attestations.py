"""The tinycap standup's three full22-modal additions, on the REAL
``bring_up`` over the pinned Nemotron config with the GPU launch stubbed (the
``test_serve_declarations_attestation`` harness): the text-only multimodal pin composed into the
launch and recorded; the M-RoPE attestation asked at readiness and REFUSING (server torn down,
nothing served) on a failing report; the Nemotron backend attestation MANDATORY for a nemotron_h
target — refused on a short / deviating live report, recorded as unattested with the reason when no
contract is handed in; every existing library caller byte-identical (all three OFF by default);
``main`` wiring pinned by source.
"""
from __future__ import annotations

import inspect
import json
import types
from pathlib import Path

import pytest
from test_serve_declarations_attestation import (
    NEMOTRON,
    _att,
    _cached_tok,
    _load,
    _stubbed_bring_up,
)

from sisteer.capeval.serving import mamba_backend as mb
from sisteer.capeval.serving import vllm_server as vs

REPO = Path(__file__).resolve().parents[1]
PATTERN = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"
OPS = "vllm.model_executor.layers.mamba.ops"
NEMO_MROPE_REPORT = {"patch_applied": True, "n_layers": 42, "n_layers_with_rotary": 0, "has_rotary": False,
                     "rotary_cls": None, "needs_patch": False}


@pytest.fixture
def capctrl():
    return _load("_attestations_capctrl_serving_up", "scripts/capctrl_serving_up.py")


@pytest.fixture(scope="module")
def contract():
    return mb.load_contract(REPO / mb.CONTRACT_RELPATH)


def _conformant_live_report(contract, n_mamba=21):
    """A worker report of the shape ``live_backend_report`` produces for a conformant engine."""
    roles = {r: {"present": True, "module": ep.rsplit(".", 1)[0], "qualname": ep.rsplit(".", 1)[1]}
             for r, ep in mb.role_entry_points(contract).items()}
    layers = [{"layer_index": i, "attr": "mixer", "mixer_class": contract["mixer_class"], "roles": roles}
              for i, ch in enumerate(PATTERN) if ch == "M"][:n_mamba]
    return {"rpc_method": mb.RPC_METHOD, "n_layers": 42, "n_mamba_layers": len(layers), "layers": layers,
            "mixer_module": contract["mixer_module"], "vllm_version": contract["vllm_version"],
            "dispatch": {"backend": "triton", "backend_class": f"{OPS}.ssu_dispatch.TritonSSUBackend",
                         "kernel": {"module": f"{OPS}.mamba_ssm", "qualname": "selective_state_update"}}}


def _rpc(monkeypatch, answers):
    """Route the standup's collective_rpc asks to canned worker reports, by method name."""
    calls = []

    def fake(launch, method, args, *, timeout_s, what):
        calls.append(method)
        if method not in answers:
            raise vs.VllmServerError(f"no such RPC {method}")
        return {"results": list(answers[method])}
    monkeypatch.setattr(vs, "_post_collective_rpc", fake)
    return calls


def _capture_launch(monkeypatch):
    """Wrap the (already stubbed) launch_server so the composed argv is readable after bring_up."""
    seen = {}
    orig = vs.launch_server

    def wrapper(launch, **kw):
        seen["launch"] = launch
        return orig(launch, **kw)
    monkeypatch.setattr(vs, "launch_server", wrapper)
    return seen


def _shutdowns(monkeypatch):
    calls = []
    monkeypatch.setattr(vs, "_shutdown_child", lambda server: calls.append(server))
    return calls


# --------------------------------------------------------------------------------------------------
# defaults: a library caller sees exactly what it saw before
# --------------------------------------------------------------------------------------------------

def test_library_defaults_compose_nothing_new_and_record_why(capctrl, tmp_path, monkeypatch):
    _cached_tok(*NEMOTRON)
    st, _ = _stubbed_bring_up(capctrl, tmp_path, monkeypatch, attest=_att({}, {}))
    seen = st.launch
    assert vs.text_only_mm_in_cmd(seen.cmd) is None
    ep = st.endpoint()
    assert ep["text_only_mm"] is False
    assert ep["text_only_mm_record"]["mode"] == "off" and ep["text_only_mm_record"]["multimodal"] is False
    assert ep["mrope_attestation"] is None and "not asked" in ep["mrope_attestation_error"]
    # a nemotron_h target WITHOUT a contract serves UNATTESTED, with the reason recorded (the
    # launcher's belt refuses this for a Nemotron row)
    assert ep["mamba_backend_attestation"] is None
    assert "no backend contract" in ep["mamba_backend_attestation_error"]
    assert mb.endpoint_attestation_problems(ep)
    # every new bring_up parameter is off by default
    params = inspect.signature(capctrl.bring_up).parameters
    assert params["text_only_mm"].default == "off"
    assert params["attest_mrope"].default is False and params["model_key"].default is None
    assert params["mamba_backend_contract"].default is None


# --------------------------------------------------------------------------------------------------
# §4.1 the text-only multimodal pin
# --------------------------------------------------------------------------------------------------

def test_text_only_pin_is_composed_on_demand_and_never_for_a_non_multimodal_target_under_auto(
        capctrl, tmp_path, monkeypatch):
    _cached_tok(*NEMOTRON)
    st, _ = _stubbed_bring_up(capctrl, tmp_path / "auto", monkeypatch, attest=_att({}, {}), text_only_mm="auto")
    assert vs.text_only_mm_in_cmd(st.launch.cmd) is None
    rec = st.endpoint()["text_only_mm_record"]
    assert rec["text_only_mm"] is False and rec["multimodal"] is False and "not multimodal" in rec["reason"]
    st, _ = _stubbed_bring_up(capctrl, tmp_path / "on", monkeypatch, attest=_att({}, {}), text_only_mm="on")
    assert vs.text_only_mm_in_cmd(st.launch.cmd) == '{"image":0,"video":0}'
    assert st.launch.cmd[-2:] == vs.TEXT_ONLY_MM_FLAG_TOKENS
    ep = st.endpoint()
    assert ep["text_only_mm"] is True and ep["text_only_mm_record"]["source"] == "standup"
    with pytest.raises(ValueError, match="--text-only-mm must be one of"):
        _stubbed_bring_up(capctrl, tmp_path / "bad", monkeypatch, attest=_att({}, {}), text_only_mm="maybe")


def test_text_only_decision_on_the_pinned_configs_and_the_manifest_source():
    # glm4v is OUT of the verified token table on main (zero GLM footprint): the multimodal
    # target composes NOTHING under `auto` and says so; a GLM lane passes `--text-only-mm on` (the generic tokens)
    glm = vs.text_only_mm_decision("auto", model_id="zai-org/GLM-4.6V-Flash",
                                   revision="411bb4d77144a3f03accbf4b780f5acb8b7cde4e", family="glm4v", cmd=())
    assert glm["multimodal"] is True and glm["text_only_mm"] is False and "no verified text-only tokens" in glm["reason"]
    glm_on = vs.text_only_mm_decision("on", model_id="zai-org/GLM-4.6V-Flash",
                                      revision="411bb4d77144a3f03accbf4b780f5acb8b7cde4e", family="glm4v", cmd=())
    assert glm_on["text_only_mm"] is True and glm_on["tokens"] == list(vs.TEXT_ONLY_MM_FLAG_TOKENS) and glm_on["source"] == "standup"
    # Qwen3.5-4B's pinned config IS multimodal (Qwen3_5ForConditionalGeneration, vision_config +
    # image/video token ids — verified with AutoConfig 2026-09-11): the cluster manifest's pin on qwen
    # was needed, and auto composes it
    qwen = vs.text_only_mm_decision("auto", model_id="Qwen/Qwen3.5-4B",
                                    revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", family="qwen3_5", cmd=())
    assert qwen["multimodal"] is True and qwen["text_only_mm"] is True
    # gemma-4 is multimodal too but has no verified tokens: auto composes NOTHING and says why; on forces it
    gem = vs.text_only_mm_decision("auto", model_id="google/gemma-4-E4B-it", revision=None, family="gemma4", cmd=())
    assert gem["multimodal"] is True and gem["text_only_mm"] is False and "no verified text-only tokens" in gem["reason"]
    forced = vs.text_only_mm_decision("on", model_id="google/gemma-4-E4B-it", revision=None, family="gemma4", cmd=())
    assert forced["text_only_mm"] is True and "generic tokens" in forced["reason"]
    # a manifest already carrying the flag is the recorded source — never duplicated
    man = vs.text_only_mm_decision("auto", model_id="zai-org/GLM-4.6V-Flash", revision=None, family="glm4v",
                                   cmd=("vllm", "serve", "--limit-mm-per-prompt", '{"image":0,"video":0}'))
    assert man["source"] == "manifest" and man["text_only_mm"] is True
    other = vs.text_only_mm_decision("auto", model_id="zai-org/GLM-4.6V-Flash", revision=None, family="glm4v",
                                     cmd=("--limit-mm-per-prompt={\"image\":2}",))
    assert other["source"] == "manifest" and other["text_only_mm"] is False and "not the zero-image" in other["reason"]
    assert vs.text_only_mm_in_cmd(("--limit-mm-per-prompt", "X")) == "X"
    assert vs.text_only_mm_in_cmd(("--limit-mm-per-prompt=Y",)) == "Y"
    off = vs.text_only_mm_decision("off", model_id="zai-org/GLM-4.6V-Flash", revision=None, family="glm4v", cmd=())
    assert off["text_only_mm"] is False and off["tokens"] is None
    # the name fallback for synthetic ids (CPU tests): the three multimodal families, nothing else
    assert vs.multimodal_target("TESTORG/glm-4.6v-thing", "0" * 40) is True
    assert vs.multimodal_target("TESTORG/qwen3.5-thing", "0" * 40) is True
    assert vs.multimodal_target("meta-llama/Llama-3.1-8B-Instruct", "0" * 40) is False
    assert vs.TEXT_ONLY_MM_MODES == ("auto", "on", "off")
    # glm4v out of the verified table on main; qwen3_5_moe — the family's routed-experts member (an HF model_type,
    # keyed RAW the way the serving path reads it) — sits beside qwen3_5 (tests/test_vllm_server_qwen3_5_moe_family_tables.py)
    assert set(vs.TEXT_ONLY_MM_TOKENS_BY_FAMILY) == {"qwen3_5", "qwen3_5_moe"}


# --------------------------------------------------------------------------------------------------
# §4.2 the M-RoPE attestation at readiness
# --------------------------------------------------------------------------------------------------

def test_mrope_attestation_is_asked_at_readiness_recorded_and_refuses_a_bad_report(capctrl, tmp_path, monkeypatch):
    _cached_tok(*NEMOTRON)
    calls = _rpc(monkeypatch, {"sisteer_mrope_report": [NEMO_MROPE_REPORT]})
    st, _ = _stubbed_bring_up(capctrl, tmp_path / "ok", monkeypatch, attest=_att({}, {}),
                              attest_mrope=True, model_key="nemotron3nano4b")
    assert calls == ["sisteer_mrope_report"]
    ep = st.endpoint()
    assert ep["mrope_attestation"]["ok"] is True and ep["mrope_attestation"]["model_key"] == "nemotron3nano4b"
    assert ep["mrope_attestation"]["reports"] == [NEMO_MROPE_REPORT] and ep["mrope_attestation_error"] is None
    # a report that does not prove the rotary for the asked key -> REFUSED before the proxy exists:
    # the server is shut down and no Standup (hence no endpoint.json) comes back
    _rpc(monkeypatch, {"sisteer_mrope_report": [{**NEMO_MROPE_REPORT, "error": "AttributeError: layers"}]})
    down = _shutdowns(monkeypatch)
    with pytest.raises(RuntimeError, match="M-RoPE dispatch attestation FAILED"):
        _stubbed_bring_up(capctrl, tmp_path / "bad", monkeypatch, attest=_att({}, {}),
                          attest_mrope=True, model_key="nemotron3nano4b")
    assert len(down) == 1
    # the GLM shape: a glm46vflash key on a report without the patch refuses by name
    _rpc(monkeypatch, {"sisteer_mrope_report": [{"patch_applied": False, "rotary_cls": "MRotaryEmbedding",
                                                 "is_neox_style": False, "kernel_reachable": True, "needs_patch": True}]})
    with pytest.raises(RuntimeError, match="neox-only"):
        _stubbed_bring_up(capctrl, tmp_path / "glm", monkeypatch, attest=_att({}, {}),
                          attest_mrope=True, model_key="glm46vflash")
    # an unreachable RPC route is a refusal too (never a pass)
    _rpc(monkeypatch, {})
    with pytest.raises(RuntimeError, match="RPC failed"):
        _stubbed_bring_up(capctrl, tmp_path / "norpc", monkeypatch, attest=_att({}, {}),
                          attest_mrope=True, model_key="nemotron3nano4b")


# --------------------------------------------------------------------------------------------------
# §5.3 the Nemotron backend attestation, mandatory at every nemotron_h standup
# --------------------------------------------------------------------------------------------------

def test_nemotron_backend_attestation_admits_a_conformant_report_and_refuses_a_short_one(
        capctrl, tmp_path, monkeypatch, contract):
    _cached_tok(*NEMOTRON)
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda dist: contract["vllm_version"] if dist == "vllm" else "0")
    good = _conformant_live_report(contract)
    calls = _rpc(monkeypatch, {mb.RPC_METHOD: [good]})
    st, _ = _stubbed_bring_up(capctrl, tmp_path / "ok", monkeypatch, attest=_att({}, {}),
                              mamba_backend_contract=contract)
    assert calls == [mb.RPC_METHOD]
    ep = st.endpoint()
    att = ep["mamba_backend_attestation"]
    assert att["ok"] is True and att["expected_n_mamba_layers"] == 21 and att["reports"] == [good]
    assert att["contract_sha256"] == mb.contract_sha256(contract) and ep["mamba_backend_attestation_error"] is None
    assert mb.endpoint_attestation_problems(ep, contract_sha256_expected=mb.contract_sha256(contract)) == []
    # (vi) 20 of 21 layers -> refused, server torn down, nothing served
    _rpc(monkeypatch, {mb.RPC_METHOD: [_conformant_live_report(contract, n_mamba=20)]})
    down = _shutdowns(monkeypatch)
    with pytest.raises(RuntimeError, match="never runs on the pure-torch fallback"):
        _stubbed_bring_up(capctrl, tmp_path / "short", monkeypatch, attest=_att({}, {}),
                          mamba_backend_contract=contract)
    assert len(down) == 1
    # a worker whose mixer binds a same-named callable from elsewhere -> refused (substituted)
    sub = json.loads(json.dumps(good))
    for layer in sub["layers"]:
        layer["roles"]["prefill_scan"]["module"] = "mamba_ssm.ops.torch_ref"
    _rpc(monkeypatch, {mb.RPC_METHOD: [sub]})
    with pytest.raises(RuntimeError, match="SUBSTITUTED"):
        _stubbed_bring_up(capctrl, tmp_path / "sub", monkeypatch, attest=_att({}, {}),
                          mamba_backend_contract=contract)
    # (v) the worker does not answer the RPC -> refused
    _rpc(monkeypatch, {})
    with pytest.raises(RuntimeError, match="RPC failed"):
        _stubbed_bring_up(capctrl, tmp_path / "norpc", monkeypatch, attest=_att({}, {}),
                          mamba_backend_contract=contract)
    # (ix) an unreviewed contract is not a denominator: refused before the RPC is asked
    calls = _rpc(monkeypatch, {mb.RPC_METHOD: [good]})
    with pytest.raises(RuntimeError, match="UNREVIEWED"):
        _stubbed_bring_up(capctrl, tmp_path / "unrev", monkeypatch, attest=_att({}, {}),
                          mamba_backend_contract={**contract, "review_ref": ""})
    assert calls == []


def test_the_contract_is_ignored_for_a_non_nemotron_target(contract):
    """The family gate is the config's model_type: only nemotron_h attests (the reason is recorded for
    every other family). Pinned on the decision code rather than a second heavy bring_up."""
    src = (REPO / "scripts/capctrl_serving_up.py").read_text(encoding="utf-8")
    assert "if family == mb.NEMOTRON_FAMILY:" in src
    assert 'mamba_err = (f"not a {mb.NEMOTRON_FAMILY} target (model_type={family!r}): "' in src
    assert mb.NEMOTRON_FAMILY == "nemotron_h"


# --------------------------------------------------------------------------------------------------
# main: the CLI turns all three on and re-asks the rotary at STOP
# --------------------------------------------------------------------------------------------------

def test_main_wires_the_flags_and_the_stop_time_reask(capctrl):
    src = inspect.getsource(capctrl.main)
    assert "text_only_mm=args.text_only_mm" in src
    assert "attest_mrope=bool(args.mrope_attest), model_key=args.model_key" in src
    assert "mamba_backend_contract=mamba_contract" in src
    assert "upstream_timeout_s=args.upstream_timeout_s" in src        # the existing pin, untouched
    assert "mra.mrope_attest(su.launch, args.model_key, timeout_s=60.0)" in src
    assert "MROPE_STOP_REPORT_FILENAME" in src and "mb.LIVE_REPORT_FILENAME" in src
    # the STOP re-ask happens BEFORE tear_down, inside the finally
    assert src.index("mra.mrope_attest(su.launch") < src.index("su.tear_down()")
    base = ["--repo-dir", "r", "--config", "c", "--cell", "none", "--defenses-dir", "d", "--out", "o", "--stop-file", "s"]
    ns = capctrl.build_parser().parse_args(base)
    assert ns.text_only_mm == "auto" and ns.mrope_attest is True
    assert ns.mamba_backend_contract == capctrl.MAMBA_CONTRACT_DEFAULT == mb.CONTRACT_RELPATH
    ns = capctrl.build_parser().parse_args(base + ["--text-only-mm", "off", "--no-mrope-attest",
                                                  "--mamba-backend-contract", "/x/contract.json"])
    assert ns.text_only_mm == "off" and ns.mrope_attest is False and ns.mamba_backend_contract == "/x/contract.json"
    assert capctrl.TEXT_ONLY_MM_MODES == vs.TEXT_ONLY_MM_MODES
    assert capctrl.MROPE_STOP_REPORT_FILENAME == "mrope_attestation.json"
    assert mb.LIVE_REPORT_FILENAME == "mamba_backend_live_report.json"


def test_the_flags_the_launcher_composes_are_spelled_once(capctrl):
    assert capctrl.TEXT_ONLY_MM_FLAG == "--text-only-mm"
    assert capctrl.MROPE_ATTEST_FLAG == "--mrope-attest"
    assert capctrl.MAMBA_CONTRACT_FLAG == "--mamba-backend-contract"
    assert capctrl.KV_DERIVED_SEQS_FLAG == "--kv-derived-seqs"          # The launcher's STANDUP_KV_DERIVED_SEQS_FLAG
    # the standup's endpoint carries the six new keys, all readable by the launcher's two readers
    fake = types.SimpleNamespace(text_only_mm={"text_only_mm": True}, mrope_attestation=None,
                                 mrope_attestation_error="e", mamba_backend_attestation=None,
                                 mamba_backend_attestation_error="f")
    for k in ("text_only_mm", "text_only_mm_record", "mrope_attestation", "mrope_attestation_error",
              "mamba_backend_attestation", "mamba_backend_attestation_error"):
        assert f'"{k}"' in inspect.getsource(capctrl.Standup.endpoint), k
    del fake
