"""The Nemotron fused-kernel backend attestation: runtime discovery compared
against the REVIEWED reference contract, fail-closed.

Cases, by the plan's numbering: (i) a role's binding REMOVED from the mixer namespace before discovery
-> ``missing``; (ii) a same-named callable from another module -> ``substituted``; (iii) an
implementation outside ``approved_impls`` -> ``unrecognised``; (iv) ``nemotron_h_uses_mixer`` False;
(v) live report absent; (vi) 20 of 21 layers; (vii) peer incomparable / zero; (viii) a cached file for
another key; (ix) an unreviewed contract or a wheel-hash mismatch; (x) the positive control (the
recorded duty numbers + a complete, conformant live report) admits; plus the false-admission case (one
fused-named symbol present, three absent -> REFUSE), the cache-key composition, the contract author's
readers on synthetic sources (and the real wheel when it is on disk), and the lens_worker RPC pin.

Every vLLM-shaped input is a FAKE: the tests never import vLLM.
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

from sisteer.capeval.serving import mamba_backend as mb

REPO = Path(__file__).resolve().parents[1]
CONTRACT_PATH = REPO / mb.CONTRACT_RELPATH
LAUNCHER_SRC = (REPO / "scripts/modal/tinycap_served.py").read_text(encoding="utf-8")
#: Nemotron-3-Nano-4B's pinned hybrid_override_pattern (tests/test_nemotron_serving_registration.py)
PATTERN = "M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-"
N_MAMBA = 21
MIXER_MOD = "vllm.model_executor.layers.mamba.mamba_mixer2"
OPS = "vllm.model_executor.layers.mamba.ops"
#: the cluster's same-GPU measurement: Nemotron vs qwen35_4b on one GPU at S 44
D114C_NEMO = {"sm_duty_mean": 0.31, "sm_duty_min": 0.05, "tok_s": 1553.0, "n_seqs": 64, "max_tokens": 512,
              "gpu_index": "0", "burst_s": 21.1, "completion_tokens_total": 32768,
              "served": {"max_num_seqs": 44}}
D114C_PEER = {"sm_duty_mean": 0.40, "sm_duty_min": 0.10, "tok_s": 1442.0, "n_seqs": 64, "max_tokens": 512,
              "gpu_index": "0", "burst_s": 22.7, "completion_tokens_total": 32768,
              "served": {"max_num_seqs": 44}}


def _load_script(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def contract():
    return mb.load_contract(CONTRACT_PATH)


@pytest.fixture(scope="module")
def author():
    return _load_script("_mamba_contract_author_under_test", "scripts/mamba_backend_contract_author.py")


def launcher_wheel_sha() -> str:
    m = re.search(r'VLLM_WHEEL = \(\s*"[^"]+"\s*"[^"]+"\s*"#sha256=([0-9a-f]{64})"', LAUNCHER_SRC)
    assert m, "the launcher's VLLM_WHEEL pin (URL + #sha256=) must be readable from its source"
    return m.group(1)


# --------------------------------------------------------------------------------------------------
# the contract: reviewed, keyed to the image wheel, the wheel's own entry points
# --------------------------------------------------------------------------------------------------

def test_contract_is_reviewed_and_keyed_to_the_image_wheel(contract):
    """(ix) An unreviewed contract or one for another wheel is refused BEFORE anything runs; the
    checked-in contract is neither."""
    sha = launcher_wheel_sha()
    assert mb.contract_problems(contract, wheel_sha256=sha, installed_vllm_version="0.25.1+cu129") == []
    assert contract["vllm_wheel_sha256"] == sha
    assert contract["vllm_version"] == "0.25.1+cu129"
    for f in ("reviewed_by", "reviewed_on", "review_ref"):
        assert str(contract[f]).strip()
    unreviewed = {**contract, "reviewed_by": ""}
    probs = mb.contract_problems(unreviewed, wheel_sha256=sha)
    assert any("UNREVIEWED" in p and "reviewed_by" in p for p in probs)
    other_wheel = mb.contract_problems(contract, wheel_sha256="ab" * 32)
    assert any("re-draft and re-review" in p for p in other_wheel)
    other_vllm = mb.contract_problems(contract, installed_vllm_version="0.26.0+cu129")
    assert any("installed vLLM" in p for p in other_vllm)
    # an incomplete roles table refuses too
    broken = {**contract, "roles": {k: v for k, v in contract["roles"].items() if k != "decode_scan"}}
    assert any("lacks role 'decode_scan'" in p for p in mb.contract_problems(broken))


def test_required_entry_points_are_the_wheels(contract):
    """The four exact paths the pinned mixer binds (read from the wheel at sha 9e206f37…: mamba_mixer2.py
    imports at lines 30-38, call sites 810 / 636+848 / 991 / 1030), the worker's binding names, and the
    author's roles table are ONE vocabulary."""
    assert mb.role_entry_points(contract) == {
        "prefill_conv": f"{OPS}.causal_conv1d.causal_conv1d_fn",
        "prefill_scan": f"{OPS}.ssd_combined.mamba_chunk_scan_combined_varlen",
        "decode_conv": f"{OPS}.causal_conv1d.causal_conv1d_update",
        "decode_scan": f"{OPS}.ssu_dispatch.selective_state_update",
    }
    assert mb.role_binding_names(contract) == dict(mb.DEFAULT_ROLE_NAMES)
    assert contract["roles"]["prefill_conv"]["call_sites"] == [810]
    assert contract["roles"]["prefill_scan"]["call_sites"] == [636, 848]
    assert contract["roles"]["decode_conv"]["call_sites"] == [991]
    assert contract["roles"]["decode_scan"]["call_sites"] == [1030]
    # the dispatch role: the Triton kernel approved, FlashInfer recorded but NOT approved
    disp = contract["roles"]["decode_scan"]["dispatch"]["backends"]
    assert disp["TRITON"]["approved"] is True and disp["TRITON"]["kernel"] == f"{OPS}.mamba_ssm.selective_state_update"
    assert disp["FLASHINFER"]["approved"] is False and disp["FLASHINFER"]["kernel"] == "flashinfer.mamba.selective_state_update"
    assert contract["roles"]["decode_scan"]["approved_impls"] == [f"{OPS}.mamba_ssm"]
    assert contract["mixer_class"] == f"{MIXER_MOD}.MambaMixer2"
    assert contract["nemotron_h_uses_mixer"] is True
    assert mb.expected_n_mamba_layers(PATTERN) == N_MAMBA
    assert mb.expected_n_mamba_layers({"hybrid_override_pattern": PATTERN}) == N_MAMBA
    assert mb.expected_n_mamba_layers(types.SimpleNamespace(hybrid_override_pattern=PATTERN)) == N_MAMBA
    with pytest.raises(TypeError):
        mb.expected_n_mamba_layers(types.SimpleNamespace())


# --------------------------------------------------------------------------------------------------
# the author: synthetic sources, then the real wheel when it is on disk
# --------------------------------------------------------------------------------------------------

MIXER_SRC = '''
import torch
from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from vllm.model_executor.layers.mamba.ops.ssd_combined import (
    mamba_chunk_scan_combined_varlen,
)
from vllm.model_executor.layers.mamba.ops.ssu_dispatch import selective_state_update

class MambaMixer2:
    def forward(self, x):
        y = causal_conv1d_fn(x)
        y = mamba_chunk_scan_combined_varlen(y)
        z = causal_conv1d_update(x)
        selective_state_update(z)
        return y
'''
DISPATCH_SRC = '''
from abc import ABC
class MambaSSUBackend(ABC):
    pass
class TritonSSUBackend(MambaSSUBackend):
    def __init__(self, cfg):
        from vllm.model_executor.layers.mamba.ops.mamba_ssm import (
            selective_state_update as _triton_selective_state_update,
        )
        self._kernel = _triton_selective_state_update
class FlashInferSSUBackend(MambaSSUBackend):
    def __init__(self, cfg):
        from flashinfer.mamba import selective_state_update as _fi_ssu
        self._kernel = _fi_ssu
_BACKEND_REGISTRY: dict[MambaBackendEnum, type[MambaSSUBackend]] = {
    MambaBackendEnum.TRITON: TritonSSUBackend,
    MambaBackendEnum.FLASHINFER: FlashInferSSUBackend,
}
def selective_state_update(*a, **k):
    pass
'''
MODEL_SRC = "from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2\nclass NemotronHModel: pass\n"


def test_author_reads_bindings_call_sites_and_the_annotated_registry(author):
    b = author.import_bindings(MIXER_SRC)
    assert b["causal_conv1d_fn"] == f"{OPS}.causal_conv1d.causal_conv1d_fn"
    assert b["selective_state_update"] == f"{OPS}.ssu_dispatch.selective_state_update"
    assert author.call_sites(MIXER_SRC, "causal_conv1d_fn") == [14]
    assert author.call_sites(MIXER_SRC, "mamba_chunk_scan_combined_varlen") == [15]
    reg = author.dispatch_registry(DISPATCH_SRC)
    assert reg == {"TRITON": {"class": f"{OPS}.ssu_dispatch.TritonSSUBackend",
                              "kernel": f"{OPS}.mamba_ssm.selective_state_update"},
                   "FLASHINFER": {"class": f"{OPS}.ssu_dispatch.FlashInferSSUBackend",
                                  "kernel": "flashinfer.mamba.selective_state_update"}}
    assert author.model_binds_mixer(MODEL_SRC) is True
    assert author.model_binds_mixer("import torch\n") is False
    draft = author.draft_from_sources(mixer_src=MIXER_SRC, dispatch_src=DISPATCH_SRC, model_src=MODEL_SRC,
                                      wheel_sha256="ab" * 32, vllm_version="0.25.1+cu129")
    assert draft["roles"]["decode_scan"]["approved_impls"] == [f"{OPS}.mamba_ssm"]
    assert draft["roles"]["decode_scan"]["dispatch"]["backends"]["FLASHINFER"]["approved"] is False
    assert draft["reviewed_by"] == "" and draft["nemotron_h_uses_mixer"] is True
    # the draft is a valid (if unreviewed) contract shape
    probs = mb.contract_problems(draft)
    assert probs and all("UNREVIEWED" in p for p in probs)
    # a stale roles table (the mixer no longer binds a role's name) refuses to draft
    with pytest.raises(ValueError, match="does not import 'selective_state_update'"):
        author.draft_from_sources(mixer_src=MIXER_SRC.replace("selective_state_update", "ssu_other"),
                                  dispatch_src=DISPATCH_SRC, model_src=MODEL_SRC,
                                  wheel_sha256="ab" * 32, vllm_version="x")
    # --check: a committed contract that drifted on a mechanical field is named
    committed = {**draft, "roles": {**draft["roles"],
                                    "prefill_conv": {**draft["roles"]["prefill_conv"], "call_sites": [1]}}}
    drift = author.mechanical_drift(draft, committed)
    assert len(drift) == 1 and drift[0].startswith("roles:")
    assert author.mechanical_drift(draft, dict(draft)) == []
    assert author.parse_wheel_spec("https://x/y.whl#sha256=" + "ab" * 32) == ("https://x/y.whl", "ab" * 32)
    with pytest.raises(ValueError):
        author.parse_wheel_spec("https://x/y.whl#md5=abc")


def _wheel_on_disk():
    import os
    for cand in (os.environ.get("SISTEER_VLLM_WHEEL", ""), "/tmp/vllm-wheel/vllm-0.25.1+cu129.whl"):
        if cand and Path(cand).is_file():
            return Path(cand)
    return None


def test_committed_contract_matches_the_pinned_wheel_bytes(author, contract):
    """The reviewer's ``--check``, run here when the pinned wheel is on disk (SISTEER_VLLM_WHEEL or the
    drafting session's /tmp copy): the committed mechanical fields are the wheel's."""
    whl = _wheel_on_disk()
    if whl is None:
        pytest.skip("the pinned vLLM wheel is not on disk (set SISTEER_VLLM_WHEEL to run the --check here)")
    draft = author.draft_from_wheel(f"{whl}#sha256={launcher_wheel_sha()}")
    assert author.mechanical_drift(draft, contract) == []
    assert author.main(["--wheel", f"{whl}#sha256={launcher_wheel_sha()}", "--check", str(CONTRACT_PATH)]) == 0


# --------------------------------------------------------------------------------------------------
# static discovery vs the contract — fakes for the vLLM modules
# --------------------------------------------------------------------------------------------------

def _fn(module: str, name: str):
    f = lambda *a, **k: None
    f.__module__, f.__qualname__, f.__name__ = module, name, name
    return f


def _fake_modules(*, drop=(), substitute=None, unrecognised=None, model_binds=True, registry=None):
    """The mixer / model / dispatch modules as a ``name -> module`` map for ``import_module``."""
    mixer = types.ModuleType(MIXER_MOD)
    mixer.__file__ = "/fake/mamba_mixer2.py"
    cls = type("MambaMixer2", (), {})
    cls.__module__, cls.__qualname__ = MIXER_MOD, "MambaMixer2"
    mixer.MambaMixer2 = cls
    bind = {
        "causal_conv1d_fn": _fn(f"{OPS}.causal_conv1d", "causal_conv1d_fn"),
        "causal_conv1d_update": _fn(f"{OPS}.causal_conv1d", "causal_conv1d_update"),
        "mamba_chunk_scan_combined_varlen": _fn(f"{OPS}.ssd_combined", "mamba_chunk_scan_combined_varlen"),
        "selective_state_update": _fn(f"{OPS}.ssu_dispatch", "selective_state_update"),
    }
    for name in drop:
        bind.pop(name)
    if substitute:
        for name, module in substitute.items():
            bind[name] = _fn(module, name)
    if unrecognised:
        for name, module in unrecognised.items():
            bind[name] = _fn(module, name)
    for name, f in bind.items():
        setattr(mixer, name, f)
    model = types.ModuleType("vllm.model_executor.models.nemotron_h")
    model.__file__ = "/fake/nemotron_h.py"
    if model_binds:
        model.MambaMixer2 = cls
    disp = types.ModuleType(f"{OPS}.ssu_dispatch")
    tri = type("TritonSSUBackend", (), {})
    tri.__module__, tri.__qualname__ = f"{OPS}.ssu_dispatch", "TritonSSUBackend"
    fi = type("FlashInferSSUBackend", (), {})
    fi.__module__, fi.__qualname__ = f"{OPS}.ssu_dispatch", "FlashInferSSUBackend"
    enum = {k: type("MambaBackendEnum", (), {"name": k})() for k in ("TRITON", "FLASHINFER")}   # hashable keys
    disp._BACKEND_REGISTRY = registry if registry is not None else {enum["TRITON"]: tri, enum["FLASHINFER"]: fi}
    disp._classes = {"tri": tri, "fi": fi}
    return {MIXER_MOD: mixer, "vllm.model_executor.models.nemotron_h": model, f"{OPS}.ssu_dispatch": disp}


def _importer(mods):
    def import_module(name):
        if name not in mods:
            raise ImportError(name)
        return mods[name]
    return import_module


def _static(contract, mods, monkeypatch):
    # the installed-vLLM read: pinned to the contract's version for these fakes
    monkeypatch.setattr(mb, "static_discovery", mb.static_discovery)
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda dist: "0.25.1+cu129" if dist == "vllm" else "0")
    return mb.static_attest(contract, import_module=_importer(mods))


def test_static_discovery_conforms_on_the_wheels_shape(contract, monkeypatch):
    att = _static(contract, _fake_modules(), monkeypatch)
    assert att["ok"] is True and att["problems"] == []
    assert att["discovery"]["nemotron_h_uses_mixer"] is True
    assert att["discovery"]["mixer_class"] == contract["mixer_class"]
    assert att["discovery"]["dispatch_registry"] == {
        "TRITON": f"{OPS}.ssu_dispatch.TritonSSUBackend", "FLASHINFER": f"{OPS}.ssu_dispatch.FlashInferSSUBackend"}
    assert att["contract_sha256"] == mb.contract_sha256(contract)


def test_removing_a_binding_before_discovery_is_rejected_as_missing(contract, monkeypatch):
    """(i) the P1-D case: a binding that vanished from the mixer module drops out of nothing — the
    contract still requires it, so discovery refuses it BY NAME as missing."""
    att = _static(contract, _fake_modules(drop=("mamba_chunk_scan_combined_varlen",)), monkeypatch)
    assert att["ok"] is False
    assert any("prefill_scan" in p and "MISSING" in p for p in att["problems"])
    assert not any("prefill_conv" in p for p in att["problems"])


def test_one_fused_symbol_present_three_absent_must_refuse(contract, monkeypatch):
    """The review's false-admission case: the cluster probe's ``all(...)`` over PRESENT symbols admitted a
    mixer with one fused-named symbol and three absent; against the contract that is three ``missing``."""
    att = _static(contract, _fake_modules(drop=("causal_conv1d_update", "mamba_chunk_scan_combined_varlen",
                                                "selective_state_update")), monkeypatch)
    assert att["ok"] is False
    missing = [p for p in att["problems"] if "MISSING" in p]
    assert len(missing) == 3
    assert {r for r in mb.ROLES if any(r in p for p in missing)} == {"decode_conv", "prefill_scan", "decode_scan"}


def test_substituted_and_unrecognised_implementations_are_rejected(contract, monkeypatch):
    """(ii) a same-named callable from another module is ``substituted``; (iii) an entry point whose
    implementation module is not approved is ``unrecognised``; (iv) a model module that does not bind
    the mixer class is rejected; a dispatch registry with an extra backend is rejected."""
    sub = _static(contract, _fake_modules(substitute={"causal_conv1d_fn": "torch_ref.causal_conv1d"}), monkeypatch)
    assert sub["ok"] is False and any("prefill_conv" in p and "SUBSTITUTED" in p for p in sub["problems"])
    # unrecognised at the static level: the binding IS the contract's entry point but the contract's
    # approved set does not contain that module (a contract edited to approve something else)
    narrowed = json.loads(json.dumps(contract))
    narrowed["roles"]["prefill_conv"]["approved_impls"] = ["some.other.module"]
    unr = _static(narrowed, _fake_modules(), monkeypatch)
    assert unr["ok"] is False and any("prefill_conv" in p and "UNRECOGNISED" in p for p in unr["problems"])
    nomix = _static(contract, _fake_modules(model_binds=False), monkeypatch)
    assert nomix["ok"] is False and any("nemotron_h_uses_mixer" in p for p in nomix["problems"])
    mods = _fake_modules()
    extra = type("CudaSSUBackend", (), {})
    extra.__module__, extra.__qualname__ = f"{OPS}.ssu_dispatch", "CudaSSUBackend"
    mods[f"{OPS}.ssu_dispatch"]._BACKEND_REGISTRY[type("MambaBackendEnum", (), {"name": "CUDA"})()] = extra
    reg = _static(contract, mods, monkeypatch)
    assert reg["ok"] is False and any("dispatch registry" in p for p in reg["problems"])
    # an import failure is a problem, never a pass
    gone = _static(contract, {k: v for k, v in _fake_modules().items() if k != MIXER_MOD}, monkeypatch)
    assert gone["ok"] is False and any("cannot import" in p for p in gone["problems"])


# --------------------------------------------------------------------------------------------------
# the live worker report and its judge
# --------------------------------------------------------------------------------------------------

def _live_model(n_mamba=N_MAMBA, *, mods=None, pattern=PATTERN):
    """A NemotronH-shaped model tree: 42 layers, each wrapping ONE ``mixer``; the Mamba ones are
    instances of the fake mixer class whose module globals bind the four roles."""
    mods = mods if mods is not None else _fake_modules()
    cls = mods[MIXER_MOD].MambaMixer2
    layers = []
    placed = 0
    for ch in pattern:
        if ch == "M" and placed < n_mamba:
            layers.append(types.SimpleNamespace(mixer=cls()))
            placed += 1
        else:
            layers.append(types.SimpleNamespace(mixer=types.SimpleNamespace()))
    return types.SimpleNamespace(model=types.SimpleNamespace(layers=layers)), mods


def _dispatch_live(mods, *, backend="tri", kernel_module=f"{OPS}.mamba_ssm", initialised=True):
    disp = mods[f"{OPS}.ssu_dispatch"]
    cls = disp._classes[backend]
    inst = cls()
    inst.name = "triton" if backend == "tri" else "flashinfer"
    inst._kernel = _fn(kernel_module, "selective_state_update")
    if initialised:
        disp.get_mamba_ssu_backend = lambda: inst
    else:
        def raise_():
            raise RuntimeError("Mamba SSU backend has not been initialized")
        disp.get_mamba_ssu_backend = raise_
    return mods


def _report(contract, model, mods):
    import importlib.metadata as md
    saved = md.version
    md.version = lambda dist: "0.25.1+cu129" if dist == "vllm" else saved(dist)
    try:
        return mb.live_backend_report(model, mixer_class=contract["mixer_class"], modules=mods)
    finally:
        md.version = saved


def test_live_report_walks_every_mamba_layer_and_the_judge_admits_a_conformant_one(contract):
    """(x) the positive control's live half: 21 mixer layers, every role bound to the contract's entry
    point, the Triton dispatch initialised -> no problems."""
    model, mods = _live_model()
    _dispatch_live(mods)
    rep = _report(contract, model, mods)
    assert rep["n_layers"] == 42 and rep["n_mamba_layers"] == N_MAMBA and len(rep["layers"]) == N_MAMBA
    assert rep["layers"][0]["roles"]["decode_scan"] == {"present": True, "module": f"{OPS}.ssu_dispatch",
                                                        "qualname": "selective_state_update"}
    assert rep["dispatch"]["backend"] == "triton" and rep["dispatch"]["kernel"]["module"] == f"{OPS}.mamba_ssm"
    assert rep["rpc_method"] == mb.RPC_METHOD
    assert mb.live_problems([rep], contract, expected_n_mamba_layers=N_MAMBA) == []
    # two workers (TP) both conformant -> still clean
    assert mb.live_problems([rep, json.loads(json.dumps(rep))], contract, expected_n_mamba_layers=N_MAMBA) == []


def test_live_report_absent_or_short_is_not_admitted(contract):
    """(v) no report; (vi) 20 of 21 layers; an errored report; a non-mapping."""
    assert mb.live_problems([], contract, expected_n_mamba_layers=N_MAMBA) == [f"no {mb.RPC_METHOD} report from any worker"]
    model, mods = _live_model(n_mamba=20)
    _dispatch_live(mods)
    rep = _report(contract, model, mods)
    probs = mb.live_problems([rep], contract, expected_n_mamba_layers=N_MAMBA)
    assert any("20 Mamba mixer layers reported, the pinned config has 21" in p for p in probs)
    assert any("errored" in p for p in mb.live_problems([{**rep, "error": "AttributeError: x"}], contract,
                                                        expected_n_mamba_layers=N_MAMBA))
    assert any("not a mapping" in p for p in mb.live_problems(["oops"], contract, expected_n_mamba_layers=N_MAMBA))
    # a model with no decoder layers reports an error rather than a clean zero
    bare = _report(contract, types.SimpleNamespace(), mods)
    assert bare.get("error") and mb.live_problems([bare], contract, expected_n_mamba_layers=N_MAMBA)


def test_live_deviations_missing_substituted_unrecognised_and_uninitialised_dispatch(contract):
    # missing: the binding removed from the mixer module BEFORE the walk
    model, mods = _live_model(mods=_fake_modules(drop=("causal_conv1d_fn",)))
    _dispatch_live(mods)
    probs = mb.live_problems([_report(contract, model, mods)], contract, expected_n_mamba_layers=N_MAMBA)
    assert sum("prefill_conv" in p and "MISSING" in p for p in probs) == N_MAMBA
    # substituted: a same-named callable from another module
    model, mods = _live_model(mods=_fake_modules(substitute={"causal_conv1d_update": "mamba_ssm.ops.torch_ref"}))
    _dispatch_live(mods)
    probs = mb.live_problems([_report(contract, model, mods)], contract, expected_n_mamba_layers=N_MAMBA)
    assert any("decode_conv" in p and "SUBSTITUTED" in p for p in probs)
    # unrecognised: the dispatch wrapper forwards to a kernel module outside approved_impls (FlashInfer)
    model, mods = _live_model()
    _dispatch_live(mods, backend="fi", kernel_module="flashinfer.mamba")
    probs = mb.live_problems([_report(contract, model, mods)], contract, expected_n_mamba_layers=N_MAMBA)
    assert any("decode_scan" in p and "UNRECOGNISED" in p for p in probs)
    assert any("not an approved backend" in p for p in probs)
    # the dispatch not initialised -> refused (never admitted on the wrapper alone)
    model, mods = _live_model()
    _dispatch_live(mods, initialised=False)
    probs = mb.live_problems([_report(contract, model, mods)], contract, expected_n_mamba_layers=N_MAMBA)
    assert any("dispatch backend not attested" in p for p in probs)
    assert any("decode_scan" in p and "UNRECOGNISED" in p for p in probs)
    # a mixer of another class is not counted as a Mamba layer -> layer count refuses
    model, mods = _live_model()
    other = type("MambaMixer2", (), {})
    other.__module__, other.__qualname__ = "vendor.mixers", "MambaMixer2"
    model.model.layers[0].mixer = other()
    _dispatch_live(mods)
    rep = _report(contract, model, mods)
    assert rep["n_mamba_layers"] == N_MAMBA - 1
    assert mb.live_problems([rep], contract, expected_n_mamba_layers=N_MAMBA)


def test_live_attest_asks_the_rpc_route_and_a_transport_failure_refuses(contract, monkeypatch):
    from sisteer.capeval.serving import vllm_server as vs
    model, mods = _live_model()
    _dispatch_live(mods)
    rep = _report(contract, model, mods)
    calls = []

    def fake_rpc(launch, method, args, *, timeout_s, what):
        calls.append((method, args, what))
        return {"results": [rep]}
    monkeypatch.setattr(vs, "_post_collective_rpc", fake_rpc)
    att = mb.live_attest(object(), contract, expected_n_mamba_layers=N_MAMBA)
    assert att["ok"] is True and att["reports"] == [rep] and calls[0][0] == mb.RPC_METHOD
    assert att["contract_sha256"] == mb.contract_sha256(contract) and att["expected_n_mamba_layers"] == N_MAMBA

    def boom(*a, **k):
        raise vs.VllmServerError("HTTP 404 — route absent")
    monkeypatch.setattr(vs, "_post_collective_rpc", boom)
    att = mb.live_attest(object(), contract, expected_n_mamba_layers=N_MAMBA)
    assert att["ok"] is False and any("RPC failed" in p for p in att["problems"]) and att["reports"] == []


def test_endpoint_reader_refuses_absent_failing_or_foreign_contract_attestations(contract):
    ok_att = {"ok": True, "problems": [], "reports": [{"n_mamba_layers": N_MAMBA}], "rpc_method": mb.RPC_METHOD,
              "contract_sha256": mb.contract_sha256(contract)}
    assert mb.endpoint_attestation_problems({"mamba_backend_attestation": ok_att},
                                            contract_sha256_expected=mb.contract_sha256(contract)) == []
    absent = mb.endpoint_attestation_problems({"mamba_backend_attestation": None,
                                               "mamba_backend_attestation_error": "no contract"})
    assert absent and "no contract" in absent[0]
    bad = mb.endpoint_attestation_problems({"mamba_backend_attestation": {**ok_att, "ok": False, "problems": ["x"]}})
    assert any("not ok" in p for p in bad)
    foreign = mb.endpoint_attestation_problems({"mamba_backend_attestation": ok_att}, contract_sha256_expected="ff" * 32)
    assert any("this launch's contract" in p for p in foreign)
    assert any("no worker report" in p for p in mb.endpoint_attestation_problems(
        {"mamba_backend_attestation": {**ok_att, "reports": []}}))


# --------------------------------------------------------------------------------------------------
# the peer duty rule, the folded document, its re-derivation, the cache key
# --------------------------------------------------------------------------------------------------

def test_duty_admission_d114c_numbers_admit_and_incomparable_or_zero_never_does():
    adm = mb.duty_admission(D114C_NEMO, D114C_PEER, ratio=0.5)
    assert adm["duty_comparable"] and adm["duty_ok"] and adm["tok_s_ok"]
    assert adm["duty_floor"] == 0.2 and adm["tok_s_floor"] == 721.0
    # (vii) a zero / non-numeric peer, a different burst, another GPU, another served S -> incomparable
    for peer, why in (({**D114C_PEER, "tok_s": 0}, "not positive"),
                      ({**D114C_PEER, "sm_duty_mean": None}, "non-numeric"),
                      ({**D114C_PEER, "n_seqs": 32}, "bursts differ"),
                      ({**D114C_PEER, "gpu_index": "1"}, "different GPUs"),
                      ({**D114C_PEER, "served": {"max_num_seqs": 22}}, "served S differs")):
        a = mb.duty_admission(D114C_NEMO, peer, ratio=0.5)
        assert a["duty_comparable"] is False and not a["duty_ok"] and not a["tok_s_ok"]
        assert any(why in r for r in a["incomparable_reasons"]), (why, a["incomparable_reasons"])
    # a served S recorded on one side only is not a mismatch (the cluster probe's rule)
    assert mb.duty_admission({**D114C_NEMO, "served": {}}, D114C_PEER, ratio=0.5)["duty_comparable"] is True
    # below the ratio: comparable but not admitted
    slow = mb.duty_admission({**D114C_NEMO, "sm_duty_mean": 0.05, "tok_s": 150.0}, D114C_PEER, ratio=0.5)
    assert slow["duty_comparable"] and not slow["duty_ok"] and not slow["tok_s_ok"]


def _positive_doc(contract, *, key="k" * 64):
    model, mods = _live_model()
    _dispatch_live(mods)
    rep = _report(contract, model, mods)
    live = {"ok": True, "problems": [], "reports": [rep], "rpc_method": mb.RPC_METHOD,
            "contract_sha256": mb.contract_sha256(contract), "expected_n_mamba_layers": N_MAMBA}
    disc = {"model_module": "/x", "mixer_module": "/y", "mixer_class": contract["mixer_class"],
            "nemotron_h_uses_mixer": True, "vllm_version": "0.25.1+cu129", "errors": [],
            "bindings": {r: {"present": True, "module": ep.rsplit(".", 1)[0], "qualname": ep.rsplit(".", 1)[1]}
                         for r, ep in mb.role_entry_points(contract).items()},
            "dispatch_registry": {"TRITON": f"{OPS}.ssu_dispatch.TritonSSUBackend",
                                  "FLASHINFER": f"{OPS}.ssu_dispatch.FlashInferSSUBackend"}}
    static = {"ok": True, "problems": [], "discovery": disc}
    inputs = dict(image_specs=["vllm==0.25.1", "mamba-ssm==2.3.2.post1"], wheel_sha256=contract["vllm_wheel_sha256"],
                  contract_sha256=mb.contract_sha256(contract), tree_sha="a" * 40,
                  model_id="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", model_revision="dfaf35de", serving_config_sha256="c" * 64,
                  max_model_len=16384, max_num_seqs=48, reduction_mode="stacked")
    return mb.attestation_document(
        packages={"ok": True, "mamba_ssm_version": "2.3.2.post1", "causal_conv1d_version": "1.6.2.post1"},
        static=static, live=live, duty=D114C_NEMO, peer_duty=D114C_PEER, peer_key="qwen35_4b",
        peer_image="image_nemotron", ratio=0.5, cache_key=key, key_inputs=mb.cache_key_inputs(**inputs),
        geometry="wave", standups=[{"model": "nemotron3nano4b", "L": 16384, "S": 48},
                                   {"model": "qwen35_4b", "L": 16384, "S": 48}])


def test_positive_control_admits_and_readmission_rederives_from_the_numbers(contract, tmp_path):
    """(x) the recorded numbers + a complete conformant live report -> admitted; the readers re-derive (never
    trust ``admitted``): a flipped verdict on failing numbers is refused, a foreign key is refused."""
    doc = _positive_doc(contract)
    assert doc["admitted"] is True and doc["fused_backend_attested"] is True and doc["contract_clean"] is True
    assert doc["duty_reference"]["model_key"] == "qwen35_4b" and doc["duty_reference"]["image"] == "image_nemotron"
    assert [s["model"] for s in doc["standups"]] == ["nemotron3nano4b", "qwen35_4b"]
    assert mb.readmission_problems(doc, contract, ratio=0.5, cache_key="k" * 64) == []
    # (viii) the file for ANOTHER key refuses at both doors
    p = tmp_path / mb.ATTESTATION_FILENAME
    p.write_text(json.dumps(doc))
    other = mb.read_attestation(p, contract, ratio=0.5, cache_key="z" * 64)
    assert other["ok"] is False and any("another image / tree" in q for q in other["problems"])
    same = mb.read_attestation(p, contract, ratio=0.5, cache_key="k" * 64)
    assert same["ok"] is True
    missing = mb.read_attestation(tmp_path / "nope" / mb.ATTESTATION_FILENAME, contract, ratio=0.5, cache_key="k" * 64)
    assert missing["ok"] is False and "run the backend probe" in missing["problems"][0]
    # a doc that SAYS admitted but whose numbers fail the ratio is refused (re-derivation)
    lying = json.loads(json.dumps(doc))
    lying["duty_record"]["tok_s"] = 100.0
    assert any("tok/s 100.0 <" in q for q in mb.readmission_problems(lying, contract, ratio=0.5, cache_key="k" * 64))
    # a stricter ratio at read time refuses what the probe admitted at 0.5
    assert mb.readmission_problems(doc, contract, ratio=0.9, cache_key="k" * 64)
    # a doc whose recorded live report lost a layer is refused on re-judgement
    short = json.loads(json.dumps(doc))
    short["live"]["reports"][0]["layers"].pop()
    short["live"]["reports"][0]["n_mamba_layers"] = N_MAMBA - 1
    assert any("recorded live report" in q for q in mb.readmission_problems(short, contract, ratio=0.5, cache_key="k" * 64))
    # a doc without the static discovery cannot be re-derived
    nostatic = {**doc, "static": {"ok": True}}
    assert any("no static discovery" in q for q in mb.readmission_problems(nostatic, contract, ratio=0.5))


def test_a_failing_half_never_admits(contract):
    doc = _positive_doc(contract)
    base = dict(packages=doc["packages"], static=doc["static"], live=doc["live"], duty=D114C_NEMO, peer_duty=D114C_PEER,
                peer_key="qwen35_4b", peer_image="image_nemotron", ratio=0.5, cache_key="k" * 64, key_inputs=doc["key_inputs"])
    assert mb.attestation_document(**{**base, "packages": {"ok": False, "error": "no mamba_ssm"}})["admitted"] is False
    assert mb.attestation_document(**{**base, "live": {**doc["live"], "ok": False, "problems": ["x"]}})["admitted"] is False
    assert mb.attestation_document(**{**base, "static": {**doc["static"], "ok": False}})["admitted"] is False
    assert mb.attestation_document(**{**base, "peer_duty": {**D114C_PEER, "tok_s": 0}})["admitted"] is False
    assert mb.attestation_document(**{**base, "duty": {**D114C_NEMO, "sm_duty_mean": 0.01}})["admitted"] is False


def test_cache_key_composition_is_per_image_tree_model_config_geometry_and_mode(contract):
    base = dict(image_specs=["vllm==0.25.1", "mamba-ssm==2.3.2.post1"], wheel_sha256=contract["vllm_wheel_sha256"],
                contract_sha256=mb.contract_sha256(contract), tree_sha="a" * 40, model_id="nvidia/x",
                model_revision="dfaf35de", serving_config_sha256="c" * 64, max_model_len=16384, max_num_seqs=48,
                reduction_mode="stacked")
    k = mb.attestation_cache_key(**base)
    assert re.fullmatch(r"[0-9a-f]{64}", k) and k == mb.attestation_cache_key(**base)
    # image specs are order-insensitive; every other input moves the key
    assert mb.attestation_cache_key(**{**base, "image_specs": list(reversed(base["image_specs"]))}) == k
    for change in ({"image_specs": ["vllm==0.25.1"]}, {"wheel_sha256": "ff" * 32}, {"contract_sha256": "ee" * 32},
                   {"tree_sha": "b" * 40}, {"model_id": "nvidia/y"}, {"model_revision": "00000000"},
                   {"serving_config_sha256": "d" * 64}, {"max_model_len": 17664}, {"max_num_seqs": 44},
                   {"reduction_mode": "per_row"},
                   # native-serving-default §1.7: the ENGINE is a key input (with its own mode value)
                   {"reduction_mode": "not_applicable", "serving_engine": "lora_native"}):
        assert mb.attestation_cache_key(**{**base, **change}) != k, change
    # (P1-E) the ladder geometry is its own key
    assert mb.attestation_cache_key(**{**base, "max_model_len": 17664, "max_num_seqs": 44}) != k
    for bad in ({"tree_sha": ""}, {"max_num_seqs": 0}, {"reduction_mode": None}, {"max_model_len": True}):
        with pytest.raises(ValueError):
            mb.attestation_cache_key(**{**base, **bad})
    assert mb.attestation_path("/vol/tinycap/nemo_backend", k) == Path(f"/vol/tinycap/nemo_backend/{k}/{mb.ATTESTATION_FILENAME}")


#: TODAY's lens key over the composition test's base inputs, computed on the tree BEFORE the engine-aware key API
#: landed (the contract's sha e3b561b4… and wheel sha 9e206f37… are inputs, so a
#: contract edit moves this literal deliberately): every historical certificate on the volume keeps this address
LENS_KEY_BEFORE_ENGINE_API = "f695dfff5df153cd267f8b3c746334c3f57f5887f5731687b9eeeed791be7bfd"


def _key_base(contract):
    return dict(image_specs=["vllm==0.25.1", "mamba-ssm==2.3.2.post1"], wheel_sha256=contract["vllm_wheel_sha256"],
                contract_sha256=mb.contract_sha256(contract), tree_sha="a" * 40, model_id="nvidia/x",
                model_revision="dfaf35de", serving_config_sha256="c" * 64, max_model_len=16384, max_num_seqs=48,
                reduction_mode="stacked")


def test_cache_key_is_engine_aware_and_the_lens_key_is_byte_identical(contract):
    """``cache_key_inputs(serving_engine=)`` — absent or ``lens`` is
    TODAY's dict and TODAY's 64-hex address (pinned as a literal computed on the pre-landing tree); ``lora_native``
    gains the engine key and requires ``reduction_mode == not_applicable``; the two malformed pairings refuse with
    the same ``ValueError`` shape as the other input checks; ``attestation_cache_key`` forwards unchanged."""
    base = _key_base(contract)
    k = mb.attestation_cache_key(**base)
    assert k == LENS_KEY_BEFORE_ENGINE_API
    assert mb.attestation_cache_key(**base, serving_engine=None) == k
    assert mb.attestation_cache_key(**base, serving_engine="lens") == k
    lens_doc = mb.cache_key_inputs(**base)
    assert "serving_engine" not in lens_doc and lens_doc == mb.cache_key_inputs(**base, serving_engine="lens")
    # the native key: another address, the engine in the dict, the mode the driver's explicit value
    native = mb.cache_key_inputs(**{**base, "reduction_mode": mb.REDUCTION_MODE_NOT_APPLICABLE},
                                 serving_engine=mb.SERVING_ENGINE_NATIVE)
    assert native["serving_engine"] == "lora_native" and native["reduction_mode"] == "not_applicable"
    assert {k_ for k_ in native} - {k_ for k_ in lens_doc} == {"serving_engine"}
    k_native = mb.attestation_cache_key(**{**base, "reduction_mode": "not_applicable"}, serving_engine="lora_native")
    assert re.fullmatch(r"[0-9a-f]{64}", k_native) and k_native != k
    # the two malformed pairings — a native key with a lens mode, the native mode without the engine — refuse
    with pytest.raises(ValueError, match="not_applicable"):
        mb.cache_key_inputs(**base, serving_engine="lora_native")
    with pytest.raises(ValueError, match="lora_native"):
        mb.cache_key_inputs(**{**base, "reduction_mode": "not_applicable"})
    with pytest.raises(ValueError, match="lora_native"):
        mb.cache_key_inputs(**{**base, "reduction_mode": "not_applicable"}, serving_engine="lens")
    with pytest.raises(ValueError, match="serving_engine"):
        mb.cache_key_inputs(**base, serving_engine="native")
    assert mb.SERVING_ENGINES == ("lens", "lora_native")


def test_native_worker_answers_the_ported_mamba_rpc_with_the_lens_workers_report(contract, monkeypatch):
    """``NativeWorkerExtension.sisteer_mamba_backend_report`` — the FIFTH
    ``sisteer_`` method on the plugin-free worker — is the lens worker's report over the native runner's model:
    the same pure ``live_backend_report`` (here over the fake NemotronH tree with the role bindings in
    ``sys.modules``), judged clean by ``live_problems`` against the reviewed contract; a short tree (20 of 21
    Mamba layers) is a REPORT the judge refuses, never an exception out of the worker."""
    import importlib.metadata as md

    from sisteer.capeval.serving import native_worker as nw

    monkeypatch.setattr(md, "version", lambda dist: "0.25.1+cu129" if dist == "vllm" else "0")
    assert mb.RPC_METHOD == "sisteer_mamba_backend_report"
    for n, want_clean in ((N_MAMBA, True), (N_MAMBA - 1, False)):
        model, mods = _live_model(n)
        _dispatch_live(mods)
        for name, mod in mods.items():
            monkeypatch.setitem(sys.modules, name, mod)
        worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(model=model))
        rep = nw.NativeWorkerExtension.sisteer_mamba_backend_report(worker)
        assert rep["rpc_method"] == mb.RPC_METHOD and rep["n_mamba_layers"] == n and "error" not in rep
        # ONE report shape on both workers: the lens worker's method body is the same call
        assert rep == mb.live_backend_report(model, modules=sys.modules) or rep["layers"] == \
            mb.live_backend_report(model, modules=sys.modules)["layers"]
        problems = mb.live_problems([rep], contract, expected_n_mamba_layers=N_MAMBA)
        assert (problems == []) is want_clean, problems


def test_duty_ratio_env_and_dmon_parsing(monkeypatch):
    monkeypatch.delenv(mb.DUTY_RATIO_ENV, raising=False)
    assert mb.duty_ratio() == 0.5
    monkeypatch.setenv(mb.DUTY_RATIO_ENV, "0.7")
    assert mb.duty_ratio() == 0.7
    monkeypatch.setenv(mb.DUTY_RATIO_ENV, "1.5")
    with pytest.raises(ValueError):
        mb.duty_ratio()
    assert mb.parse_dmon_line("# gpu   sm   mem   enc   dec") is None
    assert mb.parse_dmon_line("    0    31    12     0     0") == 0.31
    assert mb.parse_dmon_line("") is None
    rec = mb.fold_duty_samples([0.3, 0.4, 0.2], 3000, 2.0)
    assert rec == {"completion_tokens_total": 3000, "burst_s": 2.0, "tok_s": 1500.0, "sm_duty_mean": 0.3,
                   "sm_duty_min": 0.2, "n_samples": 3}
    assert mb.fold_duty_samples([], 0, 0.0)["tok_s"] is None


def test_packages_report_records_absence_as_not_ok():
    def importer(name):
        raise ImportError(name)
    rec = mb.packages_report(import_module=importer)
    assert rec["ok"] is False and "causal_conv1d" in rec["error"]
    fake = {"causal_conv1d": types.SimpleNamespace(__version__="1.6.2.post1", __file__="/c", causal_conv1d_fn=lambda: 0,
                                                    causal_conv1d_update=lambda: 0),
            "mamba_ssm": types.SimpleNamespace(__version__="2.3.2.post1", __file__="/m"),
            "mamba_ssm.ops.triton.ssd_combined": types.SimpleNamespace(mamba_chunk_scan_combined=lambda: 0),
            "mamba_ssm.ops.triton.selective_state_update": types.SimpleNamespace(selective_state_update=lambda: 0)}
    rec = mb.packages_report(import_module=lambda n: fake[n])
    assert rec["ok"] is True and rec["mamba_ssm_version"] == "2.3.2.post1"


# --------------------------------------------------------------------------------------------------
# the worker RPC is wired (source pin; GPU-only body) and the standup asks it
# --------------------------------------------------------------------------------------------------

def test_lens_worker_answers_the_backend_rpc_by_delegating_to_the_walker():
    lw = (REPO / "src/sisteer/capeval/serving/lens_worker.py").read_text(encoding="utf-8")
    assert "def sisteer_mamba_backend_report(self)" in lw
    assert "return _mb.live_backend_report(self.model_runner.model)" in lw
    assert "from sisteer.capeval.serving import mamba_backend as _mb" in lw
    su = (REPO / "scripts/capctrl_serving_up.py").read_text(encoding="utf-8")
    assert "mb.live_attest(launch, mamba_backend_contract" in su
    assert 'family == mb.NEMOTRON_FAMILY' in su
    assert "Nemotron fused-kernel backend attestation FAILED" in su
    assert 'MAMBA_CONTRACT_DEFAULT = "configs/capeval/mamba_backend_contract.json"' in su
    assert mb.CONTRACT_RELPATH == "configs/capeval/mamba_backend_contract.json"


def test_the_contract_has_one_identity_whatever_the_files_serialisation(tmp_path):
    """Round 6: `contract_sha256(Path)` hashed the FILE BYTES while the standup attested the
    LOADED mapping's canonical JSON, so the launcher's expected sha (`full22_endpoint_refusal` over
    `Path(tree) / CONTRACT_RELPATH`) never equalled the attestation and every Nemotron standup read
    NEMO_BACKEND_UNATTESTED with an `ok: true` attestation. ONE identity now: a path loads then
    canonicalises; a re-serialised file (whitespace, key order) hashes the same; the raw bytes are
    `contract_file_sha256`, never compared to an attestation."""
    import json as _json
    contract = mb.load_contract(CONTRACT_PATH)
    ident = mb.contract_sha256(contract)
    assert mb.contract_sha256(CONTRACT_PATH) == mb.contract_sha256(str(CONTRACT_PATH)) == ident
    reserialised = tmp_path / "contract_reserialised.json"
    keys = list(contract)
    reserialised.write_text(_json.dumps({k: contract[k] for k in reversed(keys)}, indent=4, sort_keys=False) + "\n\n")
    assert reserialised.read_bytes() != CONTRACT_PATH.read_bytes()
    assert mb.contract_sha256(reserialised) == ident                              # the same contract, one identity
    assert mb.contract_file_sha256(reserialised) != mb.contract_file_sha256(CONTRACT_PATH)   # the bytes differ, and are never compared
    assert mb.contract_file_sha256(CONTRACT_PATH) != ident                       # the old path spelling's number, retired
    # the launcher's expected sha for the tree's file (`full22_endpoint_refusal`: contract_sha256(Path)) == what the
    # standup's attestation carries for the LOADED contract (live_attest / static_attest: contract_sha256(mapping))
    att = {"ok": True, "problems": [], "reports": [{"worker": 0}], "rpc_method": mb.RPC_METHOD,
           "contract_sha256": mb.contract_sha256(contract), "expected_n_mamba_layers": 1}
    assert mb.endpoint_attestation_problems({"mamba_backend_attestation": att},
                                            contract_sha256_expected=mb.contract_sha256(CONTRACT_PATH)) == []
    assert mb.endpoint_attestation_problems({"mamba_backend_attestation": att},
                                            contract_sha256_expected=mb.contract_sha256(reserialised)) == []
    # the old path spelling (file bytes) is exactly the mismatch seen live
    assert mb.endpoint_attestation_problems({"mamba_backend_attestation": att},
                                            contract_sha256_expected=mb.contract_file_sha256(CONTRACT_PATH)) != []
