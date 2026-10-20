"""The plugin-free worker reports: ``worker_reports`` +
``native_worker`` + the per-name binding checks in
``serving_attestation``.

Load-bearing here:

- IMPORT ISOLATION: ``native_worker`` (and ``worker_reports``) import
  with NO vllm and NO vllm-lens present, and never pull ``lens_worker``
  in — a lora-native server process is exactly that environment;
- the MOVED functions are the same objects under both modules
  (``lw.X is wr.X``), so every existing ``lw.`` reference holds;
- ``engine_mode_report`` / ``lora_config_report`` on fake configs, every
  missing surface a refusal;
- THE FINGERPRINT ON REAL bf16 TENSORS (never fp32 stand-ins — the very
  torch build where ``.numpy()`` on bf16 raises): the int16 byte view,
  invariance to orientation / module order / storage / device, CHANGE on a
  layer swap and a site change, the packed and non-unit-scaling REFUSALS
  (worker AND launcher side), the ``{"o_proj"}`` unique-set acceptance of
  the multi-layer zero adapter's real config, the launcher<->worker
  equality (held under repeated identical casts; broken by ONE flipped
  mantissa bit through the int16 view);
- ``loaded_adapters_report`` over a fake manager in the pinned 0.25.1
  shape, refusing packed / scaled modules and surface drift by name;
- THE BINDING (``validate_native_lora_reports``): two same-shaped
  adapters PERMUTED between ids (count and fingerprint multiset
  unchanged) refuse naming both names; ids ``{1, 3}``, a per-name rank
  or module-key drift, a non-bf16 dtype refuse.

Everything is synthetic: invented weights, a from-config tiny llama for
the zero adapter's o_proj shapes, no hub access.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sisteer.capeval.serving import lens_worker as lw
from sisteer.capeval.serving import lora_export as lx
from sisteer.capeval.serving import native_worker as nw
from sisteer.capeval.serving import serving_attestation as sa
from sisteer.capeval.serving import vllm_server as vs
from sisteer.capeval.serving import worker_reports as wr

sys.path.insert(0, str(Path(__file__).parent))
from test_petri_serving_gate import _tiny_peft_adapter  # noqa: E402

SRC = str(Path(wr.__file__).resolve().parents[3])


# --------------------------------------------------------------------------
# import isolation + the moved functions' identity
# --------------------------------------------------------------------------


def test_native_worker_imports_without_vllm_or_vllm_lens():
    """In a FRESH interpreter with ``vllm`` and ``vllm_lens`` blocked (an
    ``import`` of either raises), ``native_worker`` imports, exposes the
    extension class with exactly its seven ``sisteer_`` RPCs as non-dunder
    attributes (the three routing-recorder RPCs; the Nemotron backend
    report PORTED from the lens worker — the fifth, so a native Nemotron
    standup attests its fused kernels; the residual slot report and the
    per-request failure claim), and never imports ``lens_worker`` (whose
    extension subclasses vllm-lens's)."""
    code = r"""
import sys
sys.modules["vllm"] = None
sys.modules["vllm_lens"] = None
sys.modules["vllm_lens._worker_ext"] = None
sys.modules["vllm_lens._activations_plugin"] = None
from sisteer.capeval.serving import native_worker as nw
from sisteer.capeval.serving import worker_reports as wr
assert "sisteer.capeval.serving.lens_worker" not in sys.modules
attrs = [a for a in dir(nw.NativeWorkerExtension) if not a.startswith("__")]
assert attrs == ["sisteer_arm_lora_routing_recorder", "sisteer_disarm_lora_routing_recorder",
                 "sisteer_lora_routing_observations", "sisteer_mamba_backend_report",
                 "sisteer_residual_slot_report", "sisteer_scheduler_config",
                 "sisteer_take_request_failure"], attrs
assert nw.WORKER_EXTENSION_CLS == \
    "sisteer.capeval.serving.native_worker.NativeWorkerExtension"
# the fingerprint machinery works in the plugin-free process too
import torch
t = torch.randn(3, 5, generator=torch.Generator().manual_seed(101)).to(torch.bfloat16)
assert len(wr.bf16_bytes(t)) == 30
print("OK")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "OK"


def test_moved_report_functions_are_the_same_objects():
    assert lw.LensWorkerError is wr.LensWorkerError
    assert lw.SCHEDULER_CONFIG_FIELDS is wr.SCHEDULER_CONFIG_FIELDS
    assert lw.scheduler_config_report is wr.scheduler_config_report
    assert lw.kv_group_report is wr.kv_group_report
    assert lw.batch_invariant_active is wr.batch_invariant_active
    assert lw.batch_invariant_report is wr.batch_invariant_report
    assert lw.BATCH_INVARIANT_MODE_ATTR_VERSIONS is \
        wr.BATCH_INVARIANT_MODE_ATTR_VERSIONS
    assert issubclass(wr.FingerprintError, wr.LensWorkerError)
    assert vs.NATIVE_WORKER_EXTENSION_CLS == nw.WORKER_EXTENSION_CLS


def test_native_worker_extension_defines_only_its_rpcs():
    """vLLM's injection (worker_base.py @0.25.1) asserts NO non-dunder
    attribute of the extension exists on the Worker; the class defines
    only its ``sisteer_``-prefixed RPCs (the readiness report, the three
    LoRA routing recorder RPCs, the Nemotron backend report ported
    from the lens worker, and the residual slot's report + failure claim:
    seven deliberately; ``test_native_lora_constants_agree`` pins the same seven)
    and reads worker state through them — the recorder itself is INSTANCE
    state set at arm time, never a class attribute."""
    attrs = [a for a in dir(nw.NativeWorkerExtension) if not a.startswith("__")]
    assert attrs == ["sisteer_arm_lora_routing_recorder", "sisteer_disarm_lora_routing_recorder",
                     "sisteer_lora_routing_observations", "sisteer_mamba_backend_report",
                     "sisteer_residual_slot_report", "sisteer_scheduler_config",
                     "sisteer_take_request_failure"]


# --------------------------------------------------------------------------
# engine_mode_report / lora_config_report on fake configs
# --------------------------------------------------------------------------


class _Mode:
    def __init__(self, name):
        self.name = name


class _Comp:
    def __init__(self, mode="FULL_AND_PIECEWISE", specialize=True,
                 sizes=(1, 2, 4, 8, 16), max_size=16):
        if mode is not None:
            self.cudagraph_mode = _Mode(mode)
        self.cudagraph_specialize_lora = specialize
        # the RESOLVED capture surface (CompilationConfig @0.25.1); vLLM sets
        # [] / 0 under eager, None before resolution
        self.cudagraph_capture_sizes = (list(sizes) if sizes is not None else None)
        self.max_cudagraph_capture_size = max_size


class _Model:
    def __init__(self, eager=False, mml=16384):
        self.enforce_eager = eager
        self.max_model_len = mml


class _Lora:
    def __init__(self, max_loras=2, rank=8, cpu=2, dtype=torch.bfloat16,
                 sharded=False, target_modules=None):
        self.max_loras = max_loras
        self.max_lora_rank = rank
        self.max_cpu_loras = cpu
        self.lora_dtype = dtype
        self.fully_sharded_loras = sharded
        self.target_modules = target_modules


class _Cfg:
    def __init__(self, model=None, comp=None, lora="default"):
        if model is not None:
            self.model_config = model
        if comp is not None:
            self.compilation_config = comp
        if lora != "absent":
            self.lora_config = _Lora() if lora == "default" else lora


def test_engine_mode_report_graphs_on_and_eager():
    rep = wr.engine_mode_report(_Cfg(_Model(False), _Comp()), sys_modules={})
    assert rep == {"enforce_eager": False,
                   "cudagraph_mode": "FULL_AND_PIECEWISE",
                   "cudagraph_mode_is_none": False,
                   "cudagraph_specialize_lora": True,
                   "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
                   "max_cudagraph_capture_size": 16,
                   "lens_plugin_imported": False}
    # eager: vLLM sets sizes [] and max 0 (config/vllm.py:1325 @0.25.1)
    rep = wr.engine_mode_report(_Cfg(_Model(True), _Comp("NONE", False, sizes=(), max_size=0)),
                                sys_modules={})
    assert rep["enforce_eager"] is True
    assert rep["cudagraph_mode"] == "NONE"
    assert rep["cudagraph_mode_is_none"] is True
    assert rep["cudagraph_specialize_lora"] is False
    assert rep["cudagraph_capture_sizes"] == [] and rep["max_cudagraph_capture_size"] == 0


def test_engine_mode_report_records_the_capture_surface_and_refuses_drift():
    """The RESOLVED capture surface rides on the report (sorted sizes; an
    unresolved None passes through as None — the launcher refuses it), and
    every surface drift refuses by name."""
    rep = wr.engine_mode_report(_Cfg(_Model(), _Comp(sizes=(8, 1, 4, 2), max_size=8)),
                                sys_modules={})
    assert rep["cudagraph_capture_sizes"] == [1, 2, 4, 8]
    assert rep["max_cudagraph_capture_size"] == 8
    rep = wr.engine_mode_report(_Cfg(_Model(), _Comp(sizes=None, max_size=None)),
                                sys_modules={})
    assert rep["cudagraph_capture_sizes"] is None and rep["max_cudagraph_capture_size"] is None

    class _NoSizes:
        cudagraph_mode = _Mode("FULL")
        cudagraph_specialize_lora = True
        max_cudagraph_capture_size = 16
    with pytest.raises(wr.LensWorkerError, match="no 'cudagraph_capture_sizes'"):
        wr.engine_mode_report(_Cfg(_Model(), _NoSizes()), sys_modules={})

    class _NoMax:
        cudagraph_mode = _Mode("FULL")
        cudagraph_specialize_lora = True
        cudagraph_capture_sizes = [1, 2]
    with pytest.raises(wr.LensWorkerError, match="no 'max_cudagraph_capture_size'"):
        wr.engine_mode_report(_Cfg(_Model(), _NoMax()), sys_modules={})
    scalar = _Comp()
    scalar.cudagraph_capture_sizes = "x"
    with pytest.raises(wr.LensWorkerError, match="cudagraph_capture_sizes is 'x', not a list"):
        wr.engine_mode_report(_Cfg(_Model(), scalar), sys_modules={})
    with pytest.raises(wr.LensWorkerError, match="not a list of ints"):
        wr.engine_mode_report(_Cfg(_Model(), _Comp(sizes=(1, True))), sys_modules={})
    with pytest.raises(wr.LensWorkerError, match="max_cudagraph_capture_size is '16'"):
        wr.engine_mode_report(_Cfg(_Model(), _Comp(max_size="16")), sys_modules={})


def test_engine_mode_report_sees_the_plugin_when_imported():
    mods = {"torch": object(), "vllm": object()}
    assert wr.engine_mode_report(_Cfg(_Model(), _Comp()),
                                 sys_modules=mods)["lens_plugin_imported"] is False
    mods["vllm_lens._activations_plugin"] = object()
    assert wr.engine_mode_report(_Cfg(_Model(), _Comp()),
                                 sys_modules=mods)["lens_plugin_imported"] is True
    # the bare package counts too: nothing in a native process should
    # import any vllm_lens module
    assert wr.engine_mode_report(_Cfg(_Model(), _Comp()),
                                 sys_modules={"vllm_lens": object()}
                                 )["lens_plugin_imported"] is True
    # ... but a name that merely CONTAINS the string does not
    assert wr.engine_mode_report(_Cfg(_Model(), _Comp()),
                                 sys_modules={"my_vllm_lens_tools": object()}
                                 )["lens_plugin_imported"] is False


def test_engine_mode_report_refuses_surface_drift():
    with pytest.raises(wr.LensWorkerError, match="enforce_eager"):
        wr.engine_mode_report(_Cfg(None, _Comp()), sys_modules={})
    with pytest.raises(wr.LensWorkerError, match="cudagraph_mode"):
        wr.engine_mode_report(_Cfg(_Model(), _Comp(mode=None)), sys_modules={})
    with pytest.raises(wr.LensWorkerError, match="cudagraph_mode"):
        wr.engine_mode_report(_Cfg(_Model(), None), sys_modules={})

    class _StrMode(_Comp):
        def __init__(self):
            self.cudagraph_mode = "FULL"   # not an enum member: no .name
            self.cudagraph_specialize_lora = True
    with pytest.raises(wr.LensWorkerError, match="CUDAGraphMode enum member"):
        wr.engine_mode_report(_Cfg(_Model(), _StrMode()), sys_modules={})

    class _NoSpec:
        cudagraph_mode = _Mode("FULL")
    with pytest.raises(wr.LensWorkerError, match="cudagraph_specialize_lora"):
        wr.engine_mode_report(_Cfg(_Model(), _NoSpec()), sys_modules={})
    with pytest.raises(wr.LensWorkerError, match="not a bool"):
        wr.engine_mode_report(_Cfg(_Model(eager="no"), _Comp()), sys_modules={})


def test_lora_config_report_reads_and_refuses():
    rep = wr.lora_config_report(_Cfg(lora=_Lora(3, 16, 3)))
    assert rep == {"lora_max_loras": 3, "lora_max_lora_rank": 16,
                   "lora_max_cpu_loras": 3, "lora_dtype": "torch.bfloat16",
                   "lora_fully_sharded": False, "lora_target_modules": None}
    # a deployment-time restriction is REPORTED as its sorted list (the
    # launcher asserts None: --lora-target-modules is wrapper-owned)
    assert wr.lora_config_report(_Cfg(lora=_Lora(target_modules=["qkv_proj", "o_proj"]))
                                 )["lora_target_modules"] == ["o_proj", "qkv_proj"]
    with pytest.raises(wr.LensWorkerError, match="no LoRA config"):
        wr.lora_config_report(_Cfg(lora=None))
    with pytest.raises(wr.LensWorkerError, match="no lora_config attribute"):
        wr.lora_config_report(_Cfg(lora="absent"))

    class _Partial:
        max_loras = 2
    with pytest.raises(wr.LensWorkerError, match="max_lora_rank"):
        wr.lora_config_report(_Cfg(lora=_Partial()))
    with pytest.raises(wr.LensWorkerError, match="not an int"):
        wr.lora_config_report(_Cfg(lora=_Lora(max_loras="2")))

    class _NoTargets(_Lora):
        def __init__(self):
            super().__init__()
            del self.target_modules
    with pytest.raises(wr.LensWorkerError, match="no 'target_modules'"):
        wr.lora_config_report(_Cfg(lora=_NoTargets()))
    with pytest.raises(wr.LensWorkerError, match="not None or a list of str"):
        wr.lora_config_report(_Cfg(lora=_Lora(target_modules="o_proj")))


# --------------------------------------------------------------------------
# the fingerprint, on REAL bf16 tensors
# --------------------------------------------------------------------------


def _bf16(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g).to(torch.bfloat16)


def _mw(a, b, r=4, alpha=None, packed=False):
    return wr.ModuleWeights(lora_a=a, lora_b=b, rank=r,
                            lora_alpha=(r if alpha is None else alpha),
                            is_packed=packed)


HF = "base_model.model.model.layers.{}.self_attn.o_proj"
GEMMA_VLLM = "language_model.model.layers.{}.self_attn.o_proj"
NEMOTRON_VLLM = "model.layers.{}.mixer.o_proj"


def test_bf16_bytes_is_the_int16_view_and_refuses_other_dtypes():
    t = _bf16(4, 6)
    with pytest.raises(TypeError):     # the reason for the int16 view
        t.numpy()
    raw = wr.bf16_bytes(t)
    assert len(raw) == 2 * t.numel() == 48
    assert raw == t.view(torch.int16).numpy().tobytes()
    # a non-contiguous view hashes to the same bytes as its contiguous copy
    assert wr.bf16_bytes(t.t().t()) == raw
    assert wr.bf16_bytes(t.t()) == t.t().contiguous().view(torch.int16).numpy().tobytes()
    with pytest.raises(wr.FingerprintError, match="defined on bf16, got torch.float32"):
        wr.bf16_bytes(t.to(torch.float32))
    with pytest.raises(wr.FingerprintError, match="defined on bf16, got torch.float16"):
        wr.bf16_bytes(t.to(torch.float16))
    with pytest.raises(wr.FingerprintError, match="not a tensor"):
        wr.bf16_bytes([1, 2])


def test_canonical_module_key_unifies_the_three_spellings():
    assert wr.canonical_module_key(HF.format(7)) == "L7.self_attn.o_proj"
    assert wr.canonical_module_key(GEMMA_VLLM.format(7)) == "L7.self_attn.o_proj"
    assert wr.canonical_module_key("model.layers.7.self_attn.o_proj") == \
        "L7.self_attn.o_proj"
    assert wr.canonical_module_key(NEMOTRON_VLLM.format(24)) == "L24.mixer.o_proj"
    assert wr.canonical_module_key("layers.0.self_attn.o_proj") == \
        "L0.self_attn.o_proj"
    # every adapted module of the module sets canonicalises now
    assert wr.canonical_module_key("model.layers.3.self_attn.q_proj") == "L3.self_attn.q_proj"
    assert wr.canonical_module_key("model.layers.3.mlp.down_proj") == "L3.mlp.down_proj"
    for bad in ("model.layers.3.self_attn.foo_proj",        # not an adapted projection
                "model.layers.3.mlp.gate",                  # the router: never a key
                "model.layers.self_attn.o_proj",            # no layer index
                "model.layers.3.self_attn.o_proj.lora_A.weight",  # a tensor key
                "model.xlayers.3.self_attn.o_proj",         # not a component
                "o_proj", ""):
        with pytest.raises(wr.FingerprintError):
            wr.canonical_module_key(bad)


def test_adapter_fingerprint_is_invariant_to_orientation_order_and_storage():
    a0, b0 = _bf16(4, 24, seed=1), _bf16(32, 4, seed=2)
    a1, b1 = _bf16(4, 24, seed=3), _bf16(32, 4, seed=4)
    base = wr.adapter_fingerprint_bf16({HF.format(0): _mw(a0, b0),
                                        HF.format(1): _mw(a1, b1)})
    fp, keys = base
    assert keys == ("L0.self_attn.o_proj", "L1.self_attn.o_proj")
    assert len(fp) == 64 and int(fp, 16) >= 0
    # module ORDER
    assert wr.adapter_fingerprint_bf16({HF.format(1): _mw(a1, b1),
                                        HF.format(0): _mw(a0, b0)}) == base
    # stored ORIENTATION (transposed A and/or B, as a worker might hold them)
    assert wr.adapter_fingerprint_bf16({HF.format(0): _mw(a0.t(), b0),
                                        HF.format(1): _mw(a1, b1.t())}) == base
    # non-contiguous STORAGE
    assert wr.adapter_fingerprint_bf16({HF.format(0): _mw(a0.t().contiguous().t(), b0),
                                        HF.format(1): _mw(a1, b1)}) == base
    # NAMING spelling (the worker's post-mapper names vs the file's)
    assert wr.adapter_fingerprint_bf16({GEMMA_VLLM.format(0): _mw(a0, b0),
                                        GEMMA_VLLM.format(1): _mw(a1, b1)}) == base
    # a bf16 -> fp32 -> bf16 round trip is byte-identical: same fingerprint
    assert wr.adapter_fingerprint_bf16({
        HF.format(0): _mw(a0.float().to(torch.bfloat16), b0),
        HF.format(1): _mw(a1, b1.float().to(torch.bfloat16))}) == base


@pytest.mark.skipif(not torch.cuda.is_available(), reason="device invariance needs a GPU")
def test_adapter_fingerprint_is_device_invariant():  # pragma: no cover — GPU box only
    a0, b0 = _bf16(4, 24, seed=1), _bf16(32, 4, seed=2)
    cpu = wr.adapter_fingerprint_bf16({HF.format(0): _mw(a0, b0)})
    gpu = wr.adapter_fingerprint_bf16({HF.format(0): _mw(a0.cuda(), b0.cuda())})
    assert cpu == gpu


def test_adapter_fingerprint_changes_on_layer_swap_and_site_change():
    a0, b0 = _bf16(4, 24, seed=1), _bf16(32, 4, seed=2)
    a1, b1 = _bf16(4, 24, seed=3), _bf16(32, 4, seed=4)
    base, _ = wr.adapter_fingerprint_bf16({HF.format(0): _mw(a0, b0),
                                           HF.format(1): _mw(a1, b1)})
    # the same two same-shaped modules with their tensors SWAPPED between
    # layers: identical bytes, different layer identity -> different hash
    swapped, _ = wr.adapter_fingerprint_bf16({HF.format(0): _mw(a1, b1),
                                              HF.format(1): _mw(a0, b0)})
    assert swapped != base
    # one module's canonical key changes SITE (self_attn -> mixer)
    site, keys = wr.adapter_fingerprint_bf16({
        HF.format(0): _mw(a0, b0),
        "base_model.model.model.layers.1.mixer.o_proj": _mw(a1, b1)})
    assert site != base and keys == ("L0.self_attn.o_proj", "L1.mixer.o_proj")
    # ... and a different layer INDEX under the same bytes
    idx, _ = wr.adapter_fingerprint_bf16({HF.format(0): _mw(a0, b0),
                                          HF.format(2): _mw(a1, b1)})
    assert idx != base
    # a single flipped mantissa bit in ONE element (a different
    # representable bf16 value) changes it
    a0f = a0.clone()
    a0f.view(torch.int16)[0, 0] ^= 1
    assert not torch.equal(a0f, a0)
    flipped, _ = wr.adapter_fingerprint_bf16({HF.format(0): _mw(a0f, b0),
                                              HF.format(1): _mw(a1, b1)})
    assert flipped != base


def test_adapter_fingerprint_refuses_packed_scaled_ambiguous_and_wrong_rank():
    a, b = _bf16(4, 24, seed=1), _bf16(32, 4, seed=2)
    ok = {HF.format(0): _mw(a, b)}
    assert wr.adapter_fingerprint_bf16(ok)
    # PACKED: the flag, or list-valued factors (PackedLoRALayerWeights)
    with pytest.raises(wr.FingerprintError, match="packed module .*PackedLoRALayerWeights"):
        wr.adapter_fingerprint_bf16({HF.format(0): _mw(a, b, packed=True)})
    with pytest.raises(wr.FingerprintError, match="packed module"):
        wr.adapter_fingerprint_bf16({HF.format(0): _mw([a, None, a], [b, None, b])})
    # SCALED: alpha != r (optimize() folds alpha/r into lora_b at load)
    with pytest.raises(wr.FingerprintError, match="non-unit LoRA scaling alpha=8 r=4"):
        wr.adapter_fingerprint_bf16({HF.format(0): _mw(a, b, alpha=8)})
    with pytest.raises(wr.FingerprintError, match="non-unit LoRA scaling alpha=1.0 r=4"):
        wr.adapter_fingerprint_bf16({HF.format(0): _mw(a, b, alpha=1.0)})
    # AMBIGUOUS orientation: [r, r]
    with pytest.raises(wr.FingerprintError, match="BOTH axes equal to r=4"):
        wr.adapter_fingerprint_bf16({HF.format(0): _mw(_bf16(4, 4), b)})
    # neither axis is r
    with pytest.raises(wr.FingerprintError, match="neither axis equals r=4"):
        wr.adapter_fingerprint_bf16({HF.format(0): _mw(_bf16(8, 24), b)})
    # an unrecognised module refuses by NAME (q_proj is a recognised adapted module;
    # the router gate never is)
    with pytest.raises(wr.FingerprintError, match="gate.*not a recognised adapted module"):
        wr.adapter_fingerprint_bf16({"model.layers.0.mlp.gate": _mw(a, b)})
    # two names canonicalising to one key
    with pytest.raises(wr.FingerprintError, match="two modules canonicalise"):
        wr.adapter_fingerprint_bf16({HF.format(0): _mw(a, b),
                                     GEMMA_VLLM.format(0): _mw(a, b)})
    # fp32 tensors refuse (the fingerprint is defined on bf16)
    with pytest.raises(wr.FingerprintError, match="defined on bf16"):
        wr.adapter_fingerprint_bf16({HF.format(0): _mw(a.float(), b)})
    with pytest.raises(wr.FingerprintError, match="no LoRA modules"):
        wr.adapter_fingerprint_bf16({})


# --------------------------------------------------------------------------
# the launcher side: expected_adapter_fingerprint_bf16 over a directory
# --------------------------------------------------------------------------


def _worker_side_modules(adapter_dir, *, spelling=None):
    """What the worker holds after vLLM's ONE cast: the file's fp32
    factors cast to bf16, keyed by the (optionally re-spelled) module."""
    from safetensors.torch import load_file

    cfg = json.loads((Path(adapter_dir) / lx.ADAPTER_CONFIG).read_text())
    tensors = load_file(str(Path(adapter_dir) / lx.ADAPTER_WEIGHTS))
    grouped = {}
    for key, t in tensors.items():
        module, _, which = key.rpartition(".lora_")
        which = which.split(".")[0]
        if spelling is not None:
            module = spelling(module)
        grouped.setdefault(module, {})[which] = t.to(torch.bfloat16)
    return {m: wr.ModuleWeights(ab["A"], ab["B"], cfg["r"], cfg["lora_alpha"], False)
            for m, ab in grouped.items()}


def test_launcher_expectation_equals_worker_fingerprint(tmp_path):
    d = _tiny_peft_adapter(tmp_path / "a", r=4, layers=(0, 1, 3))
    expected = wr.expected_adapter_fingerprint_bf16(d)
    assert expected[1] == ("L0.self_attn.o_proj", "L1.self_attn.o_proj",
                           "L3.self_attn.o_proj")
    # the worker's view under the file's own naming ...
    assert wr.adapter_fingerprint_bf16(_worker_side_modules(d)) == expected
    # ... and under vLLM's post-mapper naming (base_model.model. stripped,
    # gemma's language_model.model. prefix)
    assert wr.adapter_fingerprint_bf16(_worker_side_modules(
        d, spelling=lambda m: m.replace("base_model.model.model.",
                                        "language_model.model."))) == expected
    # REPEATED IDENTICAL CASTS ARE EQUAL — a content fingerprint cannot and
    # must not count casts (bf16->bf16 is the identity; bf16->fp32->bf16 is
    # byte-identical), so the contract HOLDS however many times the worker
    # re-casts
    twice = {m: wr.ModuleWeights(w.lora_a.to(torch.bfloat16),
                                 w.lora_b.float().to(torch.bfloat16),
                                 w.rank, w.lora_alpha, False)
             for m, w in _worker_side_modules(d).items()}
    assert wr.adapter_fingerprint_bf16(twice) == expected
    # THE NEGATIVE IS CONTENT: flip the lowest mantissa bit of ONE element
    # through the int16 view -> a different representable value -> changed
    perturbed = _worker_side_modules(d)
    first = next(iter(perturbed))
    a = perturbed[first].lora_a.clone()
    a.view(torch.int16)[0, 0] ^= 1
    perturbed[first] = perturbed[first]._replace(lora_a=a)
    assert wr.adapter_fingerprint_bf16(perturbed) != expected
    # two different adapters of identical shape and rank differ
    other = _tiny_peft_adapter(tmp_path / "b", r=4, layers=(0, 1, 3), seed=9)
    assert wr.expected_adapter_fingerprint_bf16(other) != expected
    # determinism
    assert wr.expected_adapter_fingerprint_bf16(d) == expected


def test_launcher_expectation_refuses_config_deviations(tmp_path):
    with pytest.raises(wr.FingerprintError, match="non-unit LoRA scaling lora_alpha=8.0 r=4"):
        wr.expected_adapter_fingerprint_bf16(
            _tiny_peft_adapter(tmp_path / "scaled", r=4, lora_alpha=8))
    with pytest.raises(wr.FingerprintError, match="target_modules=\\['o_proj', 'q_proj'\\]"):
        wr.expected_adapter_fingerprint_bf16(
            _tiny_peft_adapter(tmp_path / "qo", target_modules=("o_proj", "q_proj")))
    with pytest.raises(wr.FingerprintError, match="target_modules=\\[\\]"):
        wr.expected_adapter_fingerprint_bf16(
            _tiny_peft_adapter(tmp_path / "empty", target_modules=()))
    with pytest.raises(wr.FingerprintError, match="target_modules='o_proj'"):
        wr.expected_adapter_fingerprint_bf16(
            _tiny_peft_adapter(tmp_path / "str", extra_config={"target_modules": "o_proj"}))
    with pytest.raises(wr.FingerprintError, match="target_modules=\\['q_proj'\\]"):
        wr.expected_adapter_fingerprint_bf16(
            _tiny_peft_adapter(tmp_path / "q", target_modules=("q_proj",)))
    # the per-TENSOR guard: a config saying o_proj over tensors adapting q_proj (the file's
    # module names must equal target_modules, generalising the o_proj-only rule)
    with pytest.raises(wr.FingerprintError, match=r"target_modules=\['o_proj'\].*\['q_proj'\]"):
        wr.expected_adapter_fingerprint_bf16(_qproj_tensors(tmp_path / "lie"))
    with pytest.raises(wr.FingerprintError, match="no adapter_config.json"):
        wr.expected_adapter_fingerprint_bf16(tmp_path / "nowhere")
    # a non-fp32 file: the contract is exactly one fp32 -> bf16 cast
    with pytest.raises(wr.FingerprintError, match="is torch.bfloat16; the admitted exports write float32"):
        wr.expected_adapter_fingerprint_bf16(
            _tiny_peft_adapter(tmp_path / "bf16file", dtype=torch.bfloat16))
    # a missing lora_B
    d = _tiny_peft_adapter(tmp_path / "half", layers=(0,))
    from safetensors.torch import load_file, save_file
    t = load_file(str(d / lx.ADAPTER_WEIGHTS))
    save_file({k: v for k, v in t.items() if "lora_A" in k}, str(d / lx.ADAPTER_WEIGHTS))
    with pytest.raises(wr.FingerprintError, match="needs both lora_A and lora_B"):
        wr.expected_adapter_fingerprint_bf16(d)
    # r > declared shapes: neither axis equals r
    with pytest.raises(wr.FingerprintError, match="neither axis equals r=8"):
        wr.expected_adapter_fingerprint_bf16(
            _tiny_peft_adapter(tmp_path / "r8", r=4, extra_config={"r": 8, "lora_alpha": 8.0}))


def _qproj_tensors(d):
    """A config that claims o_proj over tensors that adapt q_proj."""
    from safetensors.torch import load_file, save_file

    d = _tiny_peft_adapter(d, r=4, layers=(0,), target_modules=("o_proj",))
    t = load_file(str(d / lx.ADAPTER_WEIGHTS))
    save_file({k.replace(".o_proj.", ".q_proj."): v for k, v in t.items()},
              str(d / lx.ADAPTER_WEIGHTS))
    return d


def test_zero_adapter_real_config_is_accepted(tmp_path):
    """The EXISTING export_zero_adapter writes one un-deduplicated suffix
    per adapted module — ``["o_proj", "o_proj"]`` on a two-layer target
 — so the preflight admits the UNIQUE set
    ``{"o_proj"}``, and the zero export over a real tiny llama's o_proj
    shapes fingerprints with module keys == the model's layers, equal to
    the worker-side fingerprint over the same zero bf16 modules."""
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, vocab_size=97,
                      max_position_embeddings=64, tie_word_embeddings=False)
    model = LlamaForCausalLM(cfg)
    # what A's oproj_target_modules will return: every layer's o_proj with
    # (in_features, out_features) off the module tree
    targets = {f"model.layers.{i}.self_attn.o_proj":
               (layer.self_attn.o_proj.in_features,
                layer.self_attn.o_proj.out_features)
               for i, layer in enumerate(model.model.layers)}
    assert len(targets) == 3
    zero = lx.export_zero_adapter(tmp_path / "zero", base_model="synthetic/tiny",
                                  rank=1, target_modules=targets)
    written = json.loads((zero / lx.ADAPTER_CONFIG).read_text())
    assert written["target_modules"] == ["o_proj", "o_proj", "o_proj"]  # as today
    assert written["lora_alpha"] == 1.0 == written["r"]
    fp, keys = wr.expected_adapter_fingerprint_bf16(zero)
    assert keys == ("L0.self_attn.o_proj", "L1.self_attn.o_proj", "L2.self_attn.o_proj")
    assert wr.adapter_fingerprint_bf16(_worker_side_modules(zero)) == (fp, keys)
    # the zero fingerprint is a function of shapes + zeros: a second zero
    # export for the same target equals it, a genuine adapter never does
    zero2 = lx.export_zero_adapter(tmp_path / "zero2", base_model="synthetic/tiny",
                                   rank=1, target_modules=targets)
    assert wr.expected_adapter_fingerprint_bf16(zero2) == (fp, keys)
    assert wr.expected_adapter_fingerprint_bf16(
        _tiny_peft_adapter(tmp_path / "g", r=1, layers=(0, 1, 2), d_in=32, d_out=32)
    )[0] != fp


# --------------------------------------------------------------------------
# loaded_adapters_report over a fake manager in the 0.25.1 shape
# --------------------------------------------------------------------------


class _W:
    """LoRALayerWeights stand-in: lora_a, lora_b, rank, lora_alpha, is_packed."""

    def __init__(self, a, b, rank=4, alpha=None, packed=False):
        self.lora_a, self.lora_b, self.rank = a, b, rank
        self.lora_alpha = rank if alpha is None else alpha
        self.is_packed = packed


class _LoRAModel:
    def __init__(self, lora_id, rank, loras):
        self.id, self.rank, self.loras = lora_id, rank, loras


class _Wrap:
    """BaseLinearLayerWithLoRA stand-in (base_linear.py @0.25.1): the base
    layer's ``input_size`` and the per-slice ``output_slices`` — one slice
    for a plain projection, the packed widths for gate_up / qkv."""

    def __init__(self, input_size=24, output_slices=(32,)):
        self.input_size = input_size
        self.output_slices = tuple(output_slices)


def _wrap(input_size=24, output_slices=(32,)):
    return _Wrap(input_size, output_slices)


class _AdapterManager:
    """LoRAModelManager stand-in: the registered adapters (``get_adapter``)
    and the EXECUTION registry ``modules`` ({module_name: BaseLayerWithLoRA},
    filled by ``_create_lora_modules`` from ``_match_target_modules``
    @0.25.1) — by default every module any registered adapter names, each
    wrapped at the fake factors' own (24, 32) dims, i.e. a healthy engine
    restricted to the adapted o_proj class."""

    def __init__(self, models, modules="all", lookup=True):
        self._models = {m.id: m for m in models}
        if modules == "all":
            modules = {name: _wrap() for m in models for name in m.loras}
        self.modules = modules
        if not lookup:
            # surface drift: no lookup method on this manager
            self._get_lora_layer_weights = None

    def get_adapter(self, lora_id):
        return self._models.get(lora_id)

    def _get_lora_layer_weights(self, lora_model, module_name):
        """``LoRAModelManager._get_lora_layer_weights`` @0.25.1 for a
        non-pooling model: ``lora_model.get_lora(module_name)`` ==
        ``loras.get(module_name)`` — the EXACT dict lookup."""
        return lora_model.loras.get(module_name)


class _WorkerLoRAManager:
    def __init__(self, models, modules="all", lookup=True):
        self._adapter_manager = _AdapterManager(models, modules, lookup)

    def list_adapters(self):
        return set(self._adapter_manager._models)


class _Desc:
    """A ``BatchDescriptor`` stand-in: the dispatcher keys on it, the report reads
    ``num_tokens`` (``uniform`` rides along as the real key does)."""
    def __init__(self, num_tokens, uniform=True):
        self.num_tokens = num_tokens
        self.uniform = uniform


class _Dispatcher:
    """A ``CudagraphDispatcher`` stand-in (v1/cudagraph_dispatcher.py @0.25.1):
    ``cudagraph_keys`` = {PIECEWISE: set, FULL: set} keyed by CUDAGraphMode
    members, the resolved ``cudagraph_mode``, ``uniform_decode_query_len``,
    ``keys_initialized``. Defaults model the fake geometry S=8 under
    FULL_AND_PIECEWISE: PIECEWISE keys at every capture size, FULL keys at the
    sizes <= 8; ``full`` may repeat a size (the real set holds one key per
    (size, LoRA case))."""
    def __init__(self, full=(1, 2, 4, 8), piecewise=(1, 2, 4, 8, 16),
                 mode="FULL_AND_PIECEWISE", qlen=1, initialized=True):
        self.piecewise_mode, self.full_mode = _Mode("PIECEWISE"), _Mode("FULL")
        self.cudagraph_keys = {self.piecewise_mode: {_Desc(n, False) for n in piecewise},
                               self.full_mode: {_Desc(n) for n in full}}
        self.cudagraph_mode = _Mode(mode)
        self.uniform_decode_query_len = qlen
        self.keys_initialized = initialized


class _Desc2:
    """A V2 ``BatchExecutionDescriptor`` stand-in (v1/worker/gpu/cudagraph_utils.py @0.25.1): the mode the graph
    was captured under rides ON the key (``cg_mode``), with the token / request counts and the LoRA case."""
    def __init__(self, cg_mode, num_tokens, num_reqs=None, uniform_token_count=None, num_active_loras=0):
        self.cg_mode = _Mode(cg_mode) if isinstance(cg_mode, str) else cg_mode
        self.num_tokens = num_tokens
        self.num_reqs = num_reqs
        self.uniform_token_count = uniform_token_count
        self.num_active_loras = num_active_loras


class _Manager:
    """A V2 ``ModelCudaGraphManager`` stand-in: ``graphs`` = the CAPTURED graphs keyed by descriptor — as the real
    manager stores them, the FULL decode graphs only (one per (size <= max_num_reqs, LoRA case); the PIECEWISE graphs
    live in the compiled model, not in ``graphs``), ``_graphs_captured``, the resolved ``cudagraph_mode``,
    ``decode_query_len``, ``lora_capture_cases``, ``max_num_reqs``. Defaults model the fake geometry S=8 under
    FULL_AND_PIECEWISE with LoRA cases [0, 1]; ``mixed=True`` ALSO stores a PIECEWISE entry per (size, LoRA case) —
    one explicit case pins that the report filters on ``cg_mode`` rather than counting every key."""
    def __init__(self, sizes=(1, 2, 4, 8, 16), max_num_reqs=8, mode="FULL_AND_PIECEWISE", qlen=1,
                 lora_cases=(0, 1), captured=True, full=None, mixed=False):
        self.cudagraph_mode = _Mode(mode)
        self.decode_query_len = qlen
        self.lora_capture_cases = list(lora_cases)
        self.max_num_reqs = max_num_reqs
        self._graphs_captured = captured
        self.graphs = {}
        q = qlen if isinstance(qlen, int) and not isinstance(qlen, bool) and qlen > 0 else 1   # inert on a bad qlen
        full_sizes = [s for s in sizes if s <= max_num_reqs * q] if full is None else list(full)
        if mixed:
            for n in sizes:
                for c in self.lora_capture_cases:
                    self.graphs[_Desc2("PIECEWISE", n, None, None, c)] = object()
        for n in full_sizes:
            for c in self.lora_capture_cases:
                self.graphs[_Desc2("FULL", n, n // q, qlen, c)] = object()


class _Runner:
    """The fake worker's model runner: V1-shaped (a ``cudagraph_dispatcher``) by default, V2-shaped when a
    ``manager`` is given (``dispatcher="absent"``), neither when both are absent."""
    def __init__(self, models=None, modules="all", lookup=True, dispatcher="default", manager=None):
        if models is not None:
            self.lora_manager = _WorkerLoRAManager(models, modules, lookup)
        if dispatcher != "absent":
            self.cudagraph_dispatcher = _Dispatcher() if dispatcher == "default" else dispatcher
        if manager is not None:
            self.cudagraph_manager = None if manager == "none" else manager


def _fake_model(lora_id, *, seed, layers=(0, 1), rank=4, spelling=GEMMA_VLLM,
                **wkw):
    loras = {}
    for i, layer in enumerate(layers):
        a = _bf16(rank, 24, seed=seed * 10 + i)
        b = _bf16(32, rank, seed=seed * 10 + 5 + i)
        loras[spelling.format(layer)] = _W(a, b, rank=rank, **wkw)
    return _LoRAModel(lora_id, rank, loras)


def test_loaded_adapters_report_shape_and_fingerprints():
    m1, m2 = _fake_model(1, seed=1), _fake_model(2, seed=2)
    rep = wr.loaded_adapters_report(_Runner([m2, m1]))   # order: sorted ids
    assert rep["lora_loaded_count"] == 2
    assert [a["id"] for a in rep["lora_adapters"]] == [1, 2]
    a1 = rep["lora_adapters"][0]
    assert a1["rank"] == 4 and a1["n_modules"] == 2
    assert a1["module_keys"] == ["L0.self_attn.o_proj", "L1.self_attn.o_proj"]
    assert a1["raw_module_names"] == sorted(m1.loras)
    assert a1["raw_shapes"][GEMMA_VLLM.format(0)] == {"lora_a": [4, 24], "lora_b": [32, 4]}
    assert a1["tensor_dtype"] == "torch.bfloat16"
    want, _ = wr.adapter_fingerprint_bf16(
        {n: wr.ModuleWeights(w.lora_a, w.lora_b, w.rank, w.lora_alpha, False)
         for n, w in m1.loras.items()})
    assert a1["fingerprint_bf16"] == want
    assert rep["lora_adapters"][1]["fingerprint_bf16"] != want
    # the EXACT-NAME bindings: every execution o_proj
    # module selects the factor of its own exact name
    assert a1["execution_bindings"] == {GEMMA_VLLM.format(0): GEMMA_VLLM.format(0),
                                        GEMMA_VLLM.format(1): GEMMA_VLLM.format(1)}
    # the report shape, pinned (native-psr-serving plan: the residual identity beside the
    # carrier's — None for a plain LoRA, the payload's sha256 for a residual entry)
    assert set(a1) == {"id", "rank", "n_modules", "module_keys", "raw_module_names",
                       "raw_shapes", "tensor_dtype", "fingerprint_bf16", "residual_fingerprint",
                       "execution_bindings"}
    assert a1["residual_fingerprint"] is None
    assert set(sa.WORKER_ADAPTER_KEYS) <= set(a1)


def test_loaded_adapters_report_binds_by_exact_name_not_canonical_key():
    """The prefix-mismatch failure, worker side. The
    factors of one adapter carry an extra ``model.`` wrapper prefix
    (``language_model.model.model.layers.N...``) while the execution
    registry names the wrapper ``language_model.model.layers.N...``: both
    canonicalise to ``L<N>.self_attn.o_proj``, so fingerprint and module
    keys are identical to the well-named adapter's — but vLLM's lookup
    (``LoRAModel.get_lora`` == ``loras.get``) finds nothing, and the report
    says so: every execution module binds ``None`` for that adapter."""
    prefixed = "language_model.model.model.layers.{}.self_attn.o_proj"
    good = _fake_model(1, seed=1)
    bad = _fake_model(2, seed=1, spelling=prefixed)          # same bytes, other prefix
    runner = _Runner([good, bad], modules={GEMMA_VLLM.format(0): _wrap(),
                                           GEMMA_VLLM.format(1): _wrap()})
    rep = wr.loaded_adapters_report(runner)
    a_good, a_bad = rep["lora_adapters"]
    assert a_good["fingerprint_bf16"] == a_bad["fingerprint_bf16"]      # canonical: identical
    assert a_good["module_keys"] == a_bad["module_keys"] == ["L0.self_attn.o_proj",
                                                             "L1.self_attn.o_proj"]
    assert a_good["execution_bindings"] == {GEMMA_VLLM.format(i): GEMMA_VLLM.format(i)
                                            for i in (0, 1)}
    assert a_bad["execution_bindings"] == {GEMMA_VLLM.format(i): None for i in (0, 1)}
    assert a_bad["raw_module_names"] == [prefixed.format(0), prefixed.format(1)]
    # the bindings ride the EXECUTION registry over every RECOGNISED wrapper: the
    # packed qkv wrapper is accounted for and binds None for an adapter that
    # carries no q/k/v factor
    qkv = "language_model.model.layers.0.self_attn.qkv_proj"
    wide = _Runner([good], modules={GEMMA_VLLM.format(0): _wrap(), qkv: _wrap(24, (8, 8, 8))})
    assert wr.loaded_adapters_report(wide)["lora_adapters"][0]["execution_bindings"] == {
        GEMMA_VLLM.format(0): GEMMA_VLLM.format(0), qkv: None}
    assert wr.lora_execution_modules_report(wide)["lora_execution_module_names"] == [
        GEMMA_VLLM.format(0), qkv]                  # the packed wrapper is a recognised module
    # a manager without the pinned lookup method refuses by name (never a
    # canonical guess)
    with pytest.raises(wr.LensWorkerError, match="no callable _get_lora_layer_weights"):
        wr.loaded_adapters_report(_Runner([good], lookup=False))
    # a lookup that hands back weights not among the adapter's factors refuses
    class _Foreign(_AdapterManager):
        def _get_lora_layer_weights(self, lora_model, module_name):
            return _W(_bf16(4, 24), _bf16(32, 4))
    runner = _Runner([good])
    runner.lora_manager._adapter_manager = _Foreign([good])
    with pytest.raises(wr.LensWorkerError, match="not among this adapter's 2 loaded factors"):
        wr.loaded_adapters_report(runner)


def test_loaded_adapters_report_refuses_by_name():
    with pytest.raises(wr.LensWorkerError, match="no lora_manager"):
        wr.loaded_adapters_report(_Runner())
    with pytest.raises(wr.LensWorkerError, match="no model_runner"):
        wr.loaded_adapters_report(None)
    # packed module (fused q/k/v): refused naming it
    packed = _fake_model(1, seed=1, packed=True)
    with pytest.raises(wr.FingerprintError, match="packed module 'language_model.model.layers.0"):
        wr.loaded_adapters_report(_Runner([packed]))
    # scaled module: alpha != r (optimize() folded it)
    scaled = _fake_model(1, seed=1, alpha=16)
    with pytest.raises(wr.FingerprintError, match="non-unit LoRA scaling alpha=16 r=4"):
        wr.loaded_adapters_report(_Runner([scaled]))
    # a plain tensor pair under the packed qkv_proj site (vLLM packs q/k/v into
    # list-valued PackedLoRALayerWeights there)
    odd = _fake_model(1, seed=1, spelling="language_model.model.layers.{}.self_attn.qkv_proj")
    with pytest.raises(wr.FingerprintError, match="qkv_proj.*packed wrapper site"):
        wr.loaded_adapters_report(_Runner([odd]))
    # surface drift: a LoRALayerWeights without is_packed
    drift = _fake_model(1, seed=1)
    for w in drift.loras.values():
        del w.is_packed
    with pytest.raises(wr.LensWorkerError, match="has no 'is_packed'"):
        wr.loaded_adapters_report(_Runner([drift]))
    # a manager whose _adapter_manager is gone
    runner = _Runner([_fake_model(1, seed=1)])
    del runner.lora_manager._adapter_manager
    with pytest.raises(wr.LensWorkerError, match="_adapter_manager"):
        wr.loaded_adapters_report(runner)
    # fp16 tensors: the fingerprint refuses the dtype by name
    half = _fake_model(1, seed=1)
    for w in half.loras.values():
        w.lora_a = w.lora_a.to(torch.float16)
    with pytest.raises(wr.FingerprintError, match="defined on bf16, got torch.float16"):
        wr.loaded_adapters_report(_Runner([half]))


def test_cudagraph_capture_coverage_report_reads_the_dispatchers_full_keys():
    """The FULL surface comes off the dispatcher's OWN keys, never the config
    list (2026-09-16): sorted distinct ``num_tokens`` of the FULL keys, their
    max (None when none), the tokens per decode sequence and the resolved
    dispatch mode. KNOWN-BAD reproduced: S=51 on vLLM's default list — the
    config reads max 96, the FULL keys stop at 48."""
    rep = wr.cudagraph_capture_coverage_report(_Runner())
    assert rep == {"cudagraph_full_capture_sizes": [1, 2, 4, 8],
                   "cudagraph_full_capture_max": 8,
                   "uniform_decode_query_len": 1,
                   "cudagraph_dispatcher_mode": "FULL_AND_PIECEWISE",
                   "model_runner_surface": "v1",
                   "cudagraph_lora_capture_cases": None}
    sizes = [1, 2, 4] + list(range(8, 97, 8))                   # vLLM's default for S=51
    full = [x for x in sizes if x <= 51]                          # initialize_cudagraph_keys
    disp = _Dispatcher(full=full * 2, piecewise=sizes)            # one key per (size, LoRA case)
    rep = wr.cudagraph_capture_coverage_report(_Runner(dispatcher=disp))
    assert rep["cudagraph_full_capture_max"] == 48
    assert rep["cudagraph_full_capture_sizes"] == [1, 2, 4, 8, 16, 24, 32, 40, 48]
    assert 51 not in rep["cudagraph_full_capture_sizes"] and 96 == max(sizes)
    # the composed list (51 a member) gives a FULL key at 51
    fixed = sorted(set(sizes) | {51})
    rep = wr.cudagraph_capture_coverage_report(
        _Runner(dispatcher=_Dispatcher(full=[x for x in fixed if x <= 51], piecewise=fixed)))
    assert rep["cudagraph_full_capture_max"] == 51
    # PIECEWISE-only: no FULL key -> [] / None (the launcher refuses that by name)
    rep = wr.cudagraph_capture_coverage_report(
        _Runner(dispatcher=_Dispatcher(full=(), mode="PIECEWISE")))
    assert rep["cudagraph_full_capture_sizes"] == [] and rep["cudagraph_full_capture_max"] is None
    assert rep["cudagraph_dispatcher_mode"] == "PIECEWISE"
    # speculative decode: two tokens per decode sequence ride along
    assert wr.cudagraph_capture_coverage_report(
        _Runner(dispatcher=_Dispatcher(qlen=2)))["uniform_decode_query_len"] == 2


def test_cudagraph_capture_coverage_report_reads_the_v2_managers_captured_graphs():
    """The V2 runner (v1/worker/gpu/model_runner.py — gemma4e4b, llama8b: VllmConfig.use_v2_model_runner picks it for
    a generate-type non-hybrid dense model) has no dispatcher; its graphs live in ``cudagraph_manager.graphs`` keyed
    by ``BatchExecutionDescriptor`` with the mode ON the key. KNOWN-BAD first: Slurm 521717 (gemma4e4b, S=107) — the
    V1-only reader refused every request while vLLM had captured FULL 34/34; the same shape now reads FULL max 107."""
    composed = list(vs.cudagraph_capture_sizes_for(107))                      # 30 entries, 107 among them, max 208
    assert len(composed) == 30 and 107 in composed and composed[-1] == 208
    mgr = _Manager(sizes=composed, max_num_reqs=107, lora_cases=(0, 1))
    assert len(mgr.graphs) == 34 and all(d.cg_mode.name == "FULL" for d in mgr.graphs)   # 17 sizes <= 107 x 2 LoRA cases
    rep = wr.cudagraph_capture_coverage_report(_Runner(dispatcher="absent", manager=mgr))
    assert rep == {"cudagraph_full_capture_sizes": [x for x in composed if x <= 107],
                   "cudagraph_full_capture_max": 107,
                   "uniform_decode_query_len": 1,
                   "cudagraph_dispatcher_mode": "FULL_AND_PIECEWISE",
                   "model_runner_surface": "v2",
                   "cudagraph_lora_capture_cases": [0, 1]}
    assert len(rep["cudagraph_full_capture_sizes"]) == 17
    # the report filters on cg_mode, never counts keys: a manager that ALSO held a PIECEWISE entry per (size, LoRA
    # case) — 60 more keys — reads identically (the LoRA cases collapse into distinct token counts either way)
    mixed = _Manager(sizes=composed, max_num_reqs=107, lora_cases=(0, 1), mixed=True)
    assert len(mixed.graphs) == 94 and sum(1 for d in mixed.graphs if d.cg_mode.name == "PIECEWISE") == 60
    assert wr.cudagraph_capture_coverage_report(_Runner(dispatcher="absent", manager=mixed)) == rep
    # the incident's surface on V2: the bare default list for S=51 — FULL stops at 48 (the launcher refuses that)
    default_51 = [1, 2, 4] + list(range(8, 97, 8))
    rep = wr.cudagraph_capture_coverage_report(
        _Runner(dispatcher="absent", manager=_Manager(sizes=default_51, max_num_reqs=51)))
    assert rep["cudagraph_full_capture_max"] == 48 and rep["model_runner_surface"] == "v2"
    # PIECEWISE-only on V2: no FULL key -> [] / None; speculative decode: two tokens per sequence
    rep = wr.cudagraph_capture_coverage_report(
        _Runner(dispatcher="absent", manager=_Manager(mode="PIECEWISE", full=())))
    assert rep["cudagraph_full_capture_sizes"] == [] and rep["cudagraph_full_capture_max"] is None
    assert wr.cudagraph_capture_coverage_report(
        _Runner(dispatcher="absent", manager=_Manager(qlen=2, full=(2, 4, 8, 16))))["uniform_decode_query_len"] == 2
    # every unreadable V2 surface refuses by name
    with pytest.raises(wr.LensWorkerError, match="cudagraph_manager is None"):
        wr.cudagraph_capture_coverage_report(_Runner(dispatcher="absent", manager="none"))
    with pytest.raises(wr.LensWorkerError, match="_graphs_captured is not True"):
        wr.cudagraph_capture_coverage_report(_Runner(dispatcher="absent", manager=_Manager(captured=False)))
    m = _Manager(); m.graphs = [1]
    with pytest.raises(wr.LensWorkerError, match="graphs is list, not a mapping"):
        wr.cudagraph_capture_coverage_report(_Runner(dispatcher="absent", manager=m))
    m = _Manager(); m.graphs = {_Desc2("FULL", "8", 8, 1, 0): object()}
    with pytest.raises(wr.LensWorkerError, match="num_tokens='8', not an int"):
        wr.cudagraph_capture_coverage_report(_Runner(dispatcher="absent", manager=m))
    m = _Manager(); m.graphs = {_Desc2(None, 8, 8, 1, 0): object()}
    with pytest.raises(wr.LensWorkerError, match="carries no cg_mode member"):
        wr.cudagraph_capture_coverage_report(_Runner(dispatcher="absent", manager=m))
    with pytest.raises(wr.LensWorkerError, match="lora_capture_cases is \\[\\], not a non-empty list"):
        wr.cudagraph_capture_coverage_report(_Runner(dispatcher="absent", manager=_Manager(lora_cases=())))
    for q in (0, True, "1", None):
        with pytest.raises(wr.LensWorkerError, match="cudagraph_manager.decode_query_len is"):
            wr.cudagraph_capture_coverage_report(
                _Runner(dispatcher="absent", manager=_Manager(qlen=q, full=(1, 2, 4, 8))))
    m = _Manager(); m.cudagraph_mode = "FULL"
    with pytest.raises(wr.LensWorkerError, match="cudagraph_manager.cudagraph_mode is not a CUDAGraphMode member"):
        wr.cudagraph_capture_coverage_report(_Runner(dispatcher="absent", manager=m))


def test_cudagraph_capture_coverage_report_refuses_every_unreadable_surface():
    with pytest.raises(wr.LensWorkerError, match="no model_runner"):
        wr.cudagraph_capture_coverage_report(None)
    # NEITHER surface (the known-bad the V1-only reader turned every V2 runner into) and BOTH refuse by name
    with pytest.raises(wr.LensWorkerError, match="has neither 'cudagraph_dispatcher' .* nor 'cudagraph_manager'"):
        wr.cudagraph_capture_coverage_report(_Runner(dispatcher="absent"))
    with pytest.raises(wr.LensWorkerError, match="carries BOTH"):
        wr.cudagraph_capture_coverage_report(_Runner(manager=_Manager()))
    r = _Runner(dispatcher="absent"); r.cudagraph_dispatcher = None
    with pytest.raises(wr.LensWorkerError, match="cudagraph_dispatcher is None"):
        wr.cudagraph_capture_coverage_report(r)
    with pytest.raises(wr.LensWorkerError, match="keys_initialized is not True"):
        wr.cudagraph_capture_coverage_report(_Runner(dispatcher=_Dispatcher(initialized=False)))
    d = _Dispatcher()
    d.cudagraph_keys = [1, 2]
    with pytest.raises(wr.LensWorkerError, match="not a mapping"):
        wr.cudagraph_capture_coverage_report(_Runner(dispatcher=d))
    d = _Dispatcher()
    d.cudagraph_keys = {d.piecewise_mode: set()}
    with pytest.raises(wr.LensWorkerError, match="carries no FULL entry"):
        wr.cudagraph_capture_coverage_report(_Runner(dispatcher=d))
    d = _Dispatcher()
    d.cudagraph_keys[d.full_mode] = 7
    with pytest.raises(wr.LensWorkerError, match="not iterable"):
        wr.cudagraph_capture_coverage_report(_Runner(dispatcher=d))
    d = _Dispatcher()
    d.cudagraph_keys[d.full_mode] = {_Desc("8")}
    with pytest.raises(wr.LensWorkerError, match="num_tokens='8', not an int"):
        wr.cudagraph_capture_coverage_report(_Runner(dispatcher=d))
    d = _Dispatcher()
    d.cudagraph_keys[d.full_mode] = {_Desc(True)}
    with pytest.raises(wr.LensWorkerError, match="not an int"):
        wr.cudagraph_capture_coverage_report(_Runner(dispatcher=d))
    for q in (0, True, "1", None):
        with pytest.raises(wr.LensWorkerError, match="cudagraph_dispatcher.uniform_decode_query_len is"):
            wr.cudagraph_capture_coverage_report(_Runner(dispatcher=_Dispatcher(qlen=q)))
    d = _Dispatcher()
    d.cudagraph_mode = "FULL_AND_PIECEWISE"                       # a str, not a member
    with pytest.raises(wr.LensWorkerError, match="cudagraph_dispatcher.cudagraph_mode is not a CUDAGraphMode member"):
        wr.cudagraph_capture_coverage_report(_Runner(dispatcher=d))
    assert wr.CUDAGRAPH_FULL_MODE_NAME == "FULL"


def test_native_worker_rpc_composes_every_report(monkeypatch):
    """The extension's one RPC is the six reports merged, on a fake worker
    (kv_group_report and batch_invariant_report are stubbed: the real
    ones import vLLM, absent on this box — their own tests cover them)."""
    class _Sched:
        max_num_seqs = 8
        max_num_batched_tokens = 8 * 16384
        long_prefill_token_threshold = 0
        max_num_partial_prefills = 1
        max_long_partial_prefills = 1
        max_num_scheduled_tokens = None
        scheduler_cls = None
        async_scheduling = False
        enable_chunked_prefill = True

    class _Cache:
        enable_prefix_caching = True
        mamba_ssm_cache_dtype = "auto"
        num_gpu_blocks = 8192
        block_size = 16

    class _VC:
        scheduler_config = _Sched()
        model_config = _Model(False)
        speculative_config = None
        cache_config = _Cache()
        compilation_config = _Comp()
        lora_config = _Lora(target_modules=["o_proj"])

    class _Self(nw.NativeWorkerExtension):
        vllm_config = _VC()
        model_runner = _Runner([_fake_model(1, seed=1), _fake_model(2, seed=2)])

    monkeypatch.setattr(wr, "kv_group_report",
                        lambda cfg, runner: {"kv_num_cache_groups": 1})
    monkeypatch.setattr(wr, "batch_invariant_report",
                        lambda: {"batch_invariant_env": False})
    out = _Self().sisteer_scheduler_config()
    assert out["max_num_batched_tokens"] == 8 * 16384          # scheduler
    assert out["kv_num_cache_groups"] == 1                     # kv groups
    assert out["batch_invariant_env"] is False                 # batch invariance
    assert out["enforce_eager"] is False and out["cudagraph_mode"] == "FULL_AND_PIECEWISE"
    assert out["lora_max_loras"] == 2 and out["lora_dtype"] == "torch.bfloat16"
    assert out["lora_loaded_count"] == 2
    assert [a["id"] for a in out["lora_adapters"]] == [1, 2]
    # the EXECUTION set (the seventh report): the modules the fake wraps,
    # canonical keys and exact names both, and every adapter's bindings
    assert out["lora_execution_module_keys"] == ["L0.self_attn.o_proj", "L1.self_attn.o_proj"]
    assert out["lora_execution_module_names"] == [GEMMA_VLLM.format(0), GEMMA_VLLM.format(1)]
    assert out["lora_execution_modules_n"] == 2 and out["lora_execution_modules_other"] == []
    assert all(set(a["execution_bindings"]) == set(out["lora_execution_module_names"])
               for a in out["lora_adapters"])
    assert set(out["lora_execution_module_strides"]) == set(out["lora_execution_module_names"])
    assert out["lora_execution_module_stride_max"] == 32
    # every key the launcher asserts or holds under lora_native is present
    for key in ("enforce_eager", "cudagraph_mode_is_none", "lens_plugin_imported",
                "lora_max_loras", "lora_max_lora_rank", "lora_max_cpu_loras",
                "lora_loaded_count", "enable_prefix_caching", "scheduler_cls",
                "lora_target_modules", "cudagraph_capture_sizes",
                "max_cudagraph_capture_size", "lora_execution_module_keys",
                "lora_execution_module_names", "lora_execution_module_strides",
                "lora_execution_module_stride_max"):
        assert key in out
    assert out["lora_target_modules"] == ["o_proj"]
    assert out["max_cudagraph_capture_size"] == 16 and out["cudagraph_capture_sizes"] == [1, 2, 4, 8, 16]
    # the FULL surface (2026-09-16) off the runner's dispatcher, beside the config's list
    assert out["cudagraph_full_capture_sizes"] == [1, 2, 4, 8] and out["cudagraph_full_capture_max"] == 8
    assert out["uniform_decode_query_len"] == 1 and out["cudagraph_dispatcher_mode"] == "FULL_AND_PIECEWISE"
    assert out["model_runner_surface"] == "v1" and out["cudagraph_lora_capture_cases"] is None
    # the eighth report: WHICH model runner — the fake is neither
    # pinned class, so the kind reads unknown here (the launcher refuses that;
    # test_lora_routing_recorder pins the two real class names)
    assert out["model_runner_class"].endswith("._Runner") and out["model_runner_kind"] == "unknown"


# --------------------------------------------------------------------------
# the per-NAME binding: validate_native_lora_reports
# --------------------------------------------------------------------------


def _two_adapters(tmp_path):
    a = _tiny_peft_adapter(tmp_path / "bank" / "cellA", r=4, seed=1)
    b = _tiny_peft_adapter(tmp_path / "bank" / "cellB", r=4, seed=2)
    names = ("cellA", "cellB")
    dirs = {"cellA": str(a.resolve()), "cellB": str(b.resolve())}
    return names, dirs, sa.expected_native_lora_state_for(names, dirs)


def _report(expected, **over):
    from test_petri_serving_gate import _native_sched_report
    return _native_sched_report(expected, **over)


def test_expected_native_lora_state_ids_follow_the_sorted_order(tmp_path):
    names, dirs, exp = _two_adapters(tmp_path)
    assert exp["n_adapters"] == 2
    assert exp["adapters"]["cellA"]["lora_int_id"] == 1
    assert exp["adapters"]["cellB"]["lora_int_id"] == 2
    assert exp["adapters"]["cellA"]["r"] == 4
    assert exp["adapters"]["cellA"]["module_keys"] == ["L0.self_attn.o_proj",
                                                      "L1.self_attn.o_proj"]
    assert exp["adapters"]["cellA"]["fingerprint_bf16"] != \
        exp["adapters"]["cellB"]["fingerprint_bf16"]
    # the order is the NAMES order the composer emitted, not alphabetical
    # by accident: reversing the names reverses the ids
    rev = sa.expected_native_lora_state_for(("cellB", "cellA"), dirs)
    assert rev["adapters"]["cellB"]["lora_int_id"] == 1
    with pytest.raises(sa.ServingAttestationError, match="disagree"):
        sa.expected_native_lora_state_for(("cellA",), dirs)
    with pytest.raises(sa.ServingAttestationError, match="no adapter names"):
        sa.expected_native_lora_state_for((), {})


def test_binding_passes_on_an_agreeing_report(tmp_path):
    _, _, exp = _two_adapters(tmp_path)
    rep = _report(exp)
    assert sa.native_lora_report_problems([rep, rep], exp) == []
    assert vs.validate_native_lora_reports([rep], exp) == [rep]


def test_binding_refuses_permuted_fingerprints_naming_both(tmp_path):
    """Count and fingerprint MULTISET unchanged, ids swapped: the old
    multiset comparison passed this; the per-name assertion refuses,
    naming both names and both ids."""
    _, _, exp = _two_adapters(tmp_path)
    rep = _report(exp)
    a, b = rep["lora_adapters"]
    a["fingerprint_bf16"], b["fingerprint_bf16"] = b["fingerprint_bf16"], a["fingerprint_bf16"]
    problems = sa.native_lora_report_problems([rep], exp)
    assert len(problems) == 2
    assert any("'cellA' (lora_int_id 1)" in p and "expected for adapter 'cellB' (lora_int_id 2)" in p
               for p in problems)
    assert any("'cellB' (lora_int_id 2)" in p and "expected for adapter 'cellA' (lora_int_id 1)" in p
               for p in problems)
    with pytest.raises(vs.VllmServerError, match="do not bind to the composed names"):
        vs.validate_native_lora_reports([rep], exp)


def test_binding_refuses_ids_count_rank_keys_and_dtype(tmp_path):
    _, _, exp = _two_adapters(tmp_path)
    # ids {1, 3}
    rep = _report(exp)
    rep["lora_adapters"][1]["id"] = 3
    problems = sa.native_lora_report_problems([rep], exp)
    assert any("ids [1, 3] != the expected [1, 2]" in p for p in problems)
    assert any("'cellB' (lora_int_id 2) has no report at that id" in p for p in problems)
    # count
    rep = _report(exp, lora_loaded_count=1)
    assert any("lora_loaded_count = 1, expected 2" in p
               for p in sa.native_lora_report_problems([rep], exp))
    # rank drift on one name
    rep = _report(exp)
    rep["lora_adapters"][0]["rank"] = 8
    assert any("'cellA' (lora_int_id 1) reports rank 8, expected 4" in p
               for p in sa.native_lora_report_problems([rep], exp))
    # module_keys drift (a layer missing)
    rep = _report(exp)
    rep["lora_adapters"][1]["module_keys"] = ["L0.self_attn.o_proj"]
    assert any("'cellB' (lora_int_id 2) reports module_keys ['L0.self_attn.o_proj']" in p
               for p in sa.native_lora_report_problems([rep], exp))
    # a swapped-layer fingerprint (same multiset of module bytes) refuses
    rep = _report(exp)
    dirs = {n: i["dir"] for n, i in exp["adapters"].items()}
    mods = _worker_side_modules(dirs["cellA"])
    (k0, w0), (k1, w1) = list(mods.items())
    swapped, _ = wr.adapter_fingerprint_bf16({k0: w1, k1: w0})
    assert swapped != exp["adapters"]["cellA"]["fingerprint_bf16"]
    rep["lora_adapters"][0]["fingerprint_bf16"] = swapped
    problems = sa.native_lora_report_problems([rep], exp)
    assert len(problems) == 1 and "'cellA' (lora_int_id 1) reports fingerprint_bf16" in problems[0]
    assert "expected for adapter" not in problems[0]   # not another name's
    # non-bf16 dtype
    rep = _report(exp)
    rep["lora_adapters"][0]["tensor_dtype"] = "torch.float16"
    assert any("tensor_dtype 'torch.float16', expected 'torch.bfloat16'" in p
               for p in sa.native_lora_report_problems([rep], exp))
    # a missing adapters block / no results at all
    assert sa.native_lora_report_problems([{"lora_loaded_count": 2}], exp)
    assert sa.native_lora_report_problems([], exp)
    assert sa.native_lora_report_problems(None, exp)
    with pytest.raises(vs.VllmServerError):
        vs.validate_native_lora_reports([], exp)


def test_binding_refuses_an_execution_set_that_excludes_a_fingerprinted_layer(tmp_path):
    """THE FINDING (code review 2026-09-11, P1): `--lora-target-modules`
    filters which model modules get a LoRA execution wrapper but NOT the
    CPU-side `loras` factors the fingerprint hashes. A report whose CPU
    adapter carries every fitted layer (fingerprint, rank and module keys
    all agree with the launcher) while the EXECUTION set excludes one layer
    is refused, naming the adapter and the layer that would never be
    applied — before it, readiness attested the full intervention over a
    server applying a subset of it."""
    _, _, exp = _two_adapters(tmp_path)
    healthy = _report(exp)
    assert sa.native_lora_report_problems([healthy], exp) == []
    # the execution set excludes L1 for EVERY adapter (the CPU side is untouched)
    rep = _report(exp, lora_execution_module_keys=["L0.self_attn.o_proj"])
    assert rep["lora_adapters"][0]["module_keys"] == ["L0.self_attn.o_proj", "L1.self_attn.o_proj"]
    problems = sa.native_lora_report_problems([rep], exp)
    assert len(problems) == 2
    for name, lid in (("cellA", 1), ("cellB", 2)):
        assert any(f"adapter '{name}' (lora_int_id {lid}): 1 of 2 fingerprinted module(s) "
                   "['L1.self_attn.o_proj'] have NO LoRA execution layer" in p
                   for p in problems), problems
    with pytest.raises(vs.VllmServerError, match="have NO LoRA execution layer"):
        vs.validate_native_lora_reports([rep], exp)
    # an execution set that is a SUPERSET (every projection wrapped, the
    # healthy no-restriction default) is fine — coverage, not equality
    wide = _report(exp, lora_execution_module_keys=["L0.self_attn.o_proj", "L1.self_attn.o_proj",
                                                    "L2.self_attn.o_proj"])
    assert sa.native_lora_report_problems([wide], exp) == []
    # an execution set that cannot be read refuses (never a silent pass)
    for bad in ({"lora_execution_module_keys": None}, {"lora_execution_module_keys": "L0"},
                {"lora_execution_module_keys": [1, 2]}):
        rep = _report(exp, **bad)
        assert any("lora_execution_module_keys MISSING" in p
                   for p in sa.native_lora_report_problems([rep], exp)), bad
    unread = _report(exp)
    del unread["lora_execution_module_keys"]
    assert any("lora_execution_module_keys MISSING" in p
               for p in sa.native_lora_report_problems([unread], exp))


def _prefixed_report(exp, prefix="language_model.model.model.layers.{}.self_attn.o_proj",
                     exec_spelling=GEMMA_VLLM):
    """A report whose factors carry ``prefix`` while the execution registry
    names the wrappers ``exec_spelling``: canonical keys, fingerprints and
    the canonical execution set all agree with the launcher; the exact
    lookup binds nothing — exactly what the worker reports for a
    mis-prefixed export."""
    rep = _report(exp)
    keys = rep["lora_execution_module_keys"]
    names = [exec_spelling.format(k[1:].split(".")[0]) for k in keys]
    rep["lora_execution_module_names"] = names
    for a in rep["lora_adapters"]:
        a["raw_module_names"] = [prefix.format(k[1:].split(".")[0]) for k in a["module_keys"]]
        a["execution_bindings"] = {m: None for m in names}
    return rep


def test_binding_refuses_matching_canonical_keys_with_different_raw_prefixes(tmp_path):
    """The prefix-mismatch failure, launcher side.
    vLLM 0.25.1 applies adapters by EXACT post-mapper module name
    (``activate_adapter`` -> ``_get_lora_layer_weights`` ->
    ``LoRAModel.get_lora`` == ``loras.get``; ``reset_lora`` on a miss) while
    the canonical keys strip every prefix before ``layers``. A report whose
    factors read ``language_model.model.model.layers.N...`` against wrappers
    at ``language_model.model.layers.N...`` passes the fingerprint, the
    module keys and the canonical coverage check — before this, readiness
    attested the full intervention over an engine applying none of it.
    Refused naming the adapter, the canonical key, the factor and the
    execution module it should have bound."""
    _, _, exp = _two_adapters(tmp_path)
    assert sa.native_lora_report_problems([_report(exp)], exp) == []      # exact names: passes
    rep = _prefixed_report(exp)
    assert rep["lora_execution_module_keys"] == ["L0.self_attn.o_proj", "L1.self_attn.o_proj"]
    problems = sa.native_lora_report_problems([rep], exp)
    assert len(problems) == 2, problems                                    # one per adapter
    for name, lid in (("cellA", 1), ("cellB", 2)):
        mine = [p for p in problems if p.startswith(f"worker 0: adapter {name!r} (lora_int_id {lid})")]
        assert len(mine) == 1, problems
        msg = mine[0]
        assert ("2 of 2 fingerprinted module(s) have a LoRA execution layer at the same "
                "canonical key but NO exact-name binding on it") in msg
        assert ("L0.self_attn.o_proj: factor "
                "['language_model.model.model.layers.0.self_attn.o_proj'] vs execution "
                "['language_model.model.layers.0.self_attn.o_proj']") in msg
        assert "LoRAModel.get_lora" in msg and "reset_lora()" in msg and msg.endswith("; refusing")
    with pytest.raises(vs.VllmServerError, match="NO exact-name binding"):
        vs.validate_native_lora_reports([rep], exp)
    # ONE mis-prefixed layer in ONE adapter is named alone; the other adapter passes
    rep = _report(exp)
    a = rep["lora_adapters"][0]
    a["raw_module_names"][1] = "language_model.model.model.layers.1.self_attn.o_proj"
    a["execution_bindings"][rep["lora_execution_module_names"][1]] = None
    problems = sa.native_lora_report_problems([rep], exp)
    assert len(problems) == 1 and "adapter 'cellA' (lora_int_id 1): 1 of 2" in problems[0]
    assert "L1.self_attn.o_proj: factor ['language_model.model.model.layers.1.self_attn.o_proj']" \
        in problems[0]
    # a layer with NO execution module at all stays the coverage refusal,
    # not a second exact-name one (each root cause named once)
    rep = _report(exp, lora_execution_module_keys=["L0.self_attn.o_proj"])
    problems = sa.native_lora_report_problems([rep], exp)
    assert len(problems) == 2 and all("have NO LoRA execution layer" in p for p in problems)
    # an engine wrapping MORE o_proj layers than the adapter adapts is the
    # healthy unrestricted default (a genuine export covers its fit layers
    # only): unbound execution modules are not a problem
    wide = _report(exp)
    extra = "model.layers.7.self_attn.o_proj"
    wide["lora_execution_module_keys"].append("L7.self_attn.o_proj")
    wide["lora_execution_module_names"].append(extra)
    for a in wide["lora_adapters"]:
        a["execution_bindings"][extra] = None
    assert sa.native_lora_report_problems([wide], exp) == []


def test_binding_refuses_unreadable_or_inconsistent_exact_names(tmp_path):
    """The exact-name evidence is REQUIRED (never a silent pass) and held
    consistent: missing ``lora_execution_module_names`` / ``execution_bindings``
    / ``raw_module_names``; names that disagree with the canonical set;
    bindings not resolved over the registry; a selection that is not this
    adapter's factor; one factor selected twice; a bound module outside the
    fingerprinted set (both directions)."""
    _, _, exp = _two_adapters(tmp_path)
    names = _report(exp)["lora_execution_module_names"]
    for bad in ({"lora_execution_module_names": None}, {"lora_execution_module_names": "x"},
                {"lora_execution_module_names": [1]}):
        assert any("lora_execution_module_names MISSING" in p
                   for p in sa.native_lora_report_problems([_report(exp, **bad)], exp)), bad
    unread = _report(exp)
    del unread["lora_execution_module_names"]
    assert any("lora_execution_module_names MISSING" in p
               for p in sa.native_lora_report_problems([unread], exp))
    rep = _report(exp)
    rep["lora_execution_module_names"] = ["model.layers.0.self_attn.o_proj"]      # canonical set is L0, L1
    assert any("exact and canonical spellings of the execution set disagree" in p
               for p in sa.native_lora_report_problems([rep], exp))
    rep = _report(exp)
    rep["lora_execution_module_names"] = ["model.layers.0.self_attn.q_proj",
                                          "model.layers.1.self_attn.o_proj"]
    assert any("exact and canonical spellings of the execution set disagree" in p
               for p in sa.native_lora_report_problems([rep], exp))
    rep = _report(exp)
    rep["lora_execution_module_names"] = ["model.layers.0.mlp.gate",            # the router: no key
                                          "model.layers.1.self_attn.o_proj"]
    assert any("not a recognised adapted module" in p for p in sa.native_lora_report_problems([rep], exp))
    for over, match in (({"execution_bindings": None}, "execution_bindings MISSING"),
                        ({"execution_bindings": {1: "x"}}, "execution_bindings MISSING"),
                        ({"execution_bindings": {names[0]: names[0]}},
                         "bindings were not resolved over the execution registry"),
                        ({"raw_module_names": None}, "raw_module_names MISSING")):
        rep = _report(exp)
        rep["lora_adapters"][1].update(over)
        problems = sa.native_lora_report_problems([rep], exp)
        assert any(p.startswith("worker 0: adapter 'cellB' (lora_int_id 2)") and match in p
                   for p in problems), (over, problems)
    unread = _report(exp)
    del unread["lora_adapters"][0]["execution_bindings"]
    assert any("adapter 'cellA' (lora_int_id 1): execution_bindings MISSING" in p
               for p in sa.native_lora_report_problems([unread], exp))
    # a selection that is not among the adapter's factors
    rep = _report(exp)
    rep["lora_adapters"][0]["execution_bindings"][names[0]] = "model.layers.0.self_attn.q_proj"
    assert any("select ['model.layers.0.self_attn.q_proj'], which are not among this adapter's 2 "
               "loaded factors" in p for p in sa.native_lora_report_problems([rep], exp))
    # one factor selected by two execution modules
    rep = _report(exp)
    rep["lora_adapters"][0]["execution_bindings"][names[1]] = names[0]
    problems = sa.native_lora_report_problems([rep], exp)
    assert any(f"factor(s) ['{names[0]}'] are selected by more than one execution module" in p
               for p in problems), problems
    # a bound module whose canonical key lies outside the fingerprinted set
    rep = _report(exp)
    extra = "model.layers.7.self_attn.o_proj"
    rep["lora_execution_module_keys"].append("L7.self_attn.o_proj")
    rep["lora_execution_module_names"].append(extra)
    for a in rep["lora_adapters"]:
        a["execution_bindings"][extra] = None
    rep["lora_adapters"][0]["execution_bindings"][extra] = names[0]      # L7 applies L0's factor
    problems = sa.native_lora_report_problems([rep], exp)
    assert any("execution module(s) at ['L7.self_attn.o_proj'] bind a factor of this adapter" in p
               and f"{extra} -> {names[0]}" in p for p in problems), problems
    assert any("selected by more than one execution module" in p for p in problems)


def test_execution_strides_are_held_to_the_adapted_class_and_the_ceilings_D(tmp_path):
    """ASTRA ROUND 3 (P1, verified in the pinned source): with
    ``lora_target_modules=None`` vLLM 0.25.1 wraps EVERY linear projection
    and dispatches the LoRA kernels on each whenever the batch has LoRA
    requests, so the int32 index ceiling must be sized from the WIDEST
    executing wrapper — gemma-4-E4B's packed gate_up_proj output stride is
    20,480 (2 x 10,240), not the adapted o_proj's 4,096. The launcher
    restricts the wrappers to the adapted class and readiness holds every
    executing stride to D: an o_proj-only registry within D passes; a
    gate_up wrapper refuses TWICE by name (outside the class, and wider than
    D with the real ceiling spelled out); an o_proj wrapper wider than D
    refuses; missing / inconsistent stride evidence refuses."""
    _, _, exp = _two_adapters(tmp_path)
    D = 32                                      # the tiny adapters' widest dim
    rep = _report(exp)
    assert rep["lora_target_modules"] == ["o_proj"]
    assert rep["lora_execution_module_strides"] == {
        "model.layers.0.self_attn.o_proj": {"in": 24, "out": 32},
        "model.layers.1.self_attn.o_proj": {"in": 24, "out": 32}}
    assert rep["lora_execution_module_stride_max"] == 32
    assert sa.execution_stride_problems([rep, rep], target_modules=("o_proj",), max_module_dim=D) == []
    assert sa.execution_stride_problems([rep], target_modules=("o_proj",), max_module_dim=4096) == []
    # the finding: a gate_up wrapper at gemma's packed stride, on an engine
    # whose ceiling D was sized for the adapted o_proj class
    gate_up = "model.layers.0.mlp.gate_up_proj"
    wide = _report(exp, lora_execution_modules_other=[gate_up])
    wide["lora_execution_module_strides"][gate_up] = {"in": 2560, "out": 20480}
    wide["lora_execution_module_stride_max"] = 20480
    problems = sa.execution_stride_problems([wide], target_modules=("o_proj",), max_module_dim=4096)
    assert len(problems) == 2, problems
    assert ("1 execution module(s) carry a LoRA wrapper whose suffix is NOT in the composed "
            "--lora-target-modules ['o_proj']: ['model.layers.0.mlp.gate_up_proj']") in problems[0]
    assert ("execution module 'model.layers.0.mlp.gate_up_proj' carries a LoRA wrapper with stride "
            "20,480 (in 2,560 x out 20,480) > the composed lora_max_module_dim 4,096") in problems[1]
    assert "(2^31-1) // 20,480 = 104,857 batched tokens, not the composed 524,287" in problems[1]
    assert "lora_expand_op: ram * output_d0_stride" in problems[1] and "job 488932" in problems[1]
    # a wrapper INSIDE the class but wider than D refuses on the stride alone
    tall = _report(exp)
    tall["lora_execution_module_strides"]["model.layers.1.self_attn.o_proj"] = {"in": 4096, "out": 32}
    tall["lora_execution_module_stride_max"] = 4096
    problems = sa.execution_stride_problems([tall], target_modules=("o_proj",), max_module_dim=D)
    assert len(problems) == 1 and "stride 4,096 (in 4,096 x out 32) > the composed lora_max_module_dim 32" \
        in problems[0]
    # the same registry under a D that covers it passes: the check is against D, not a constant
    assert sa.execution_stride_problems([tall], target_modules=("o_proj",), max_module_dim=4096) == []
    # evidence that cannot be read, or disagrees with itself, refuses
    for over, match in (({"lora_execution_module_strides": None}, "lora_execution_module_strides MISSING"),
                        ({"lora_execution_module_strides": {"x": {"in": 1}}},
                         "lora_execution_module_strides MISSING"),
                        ({"lora_execution_module_strides": {"model.layers.0.self_attn.o_proj": {"in": 24, "out": 32}}},
                         "were not read over the execution registry"),
                        ({"lora_execution_module_stride_max": 31}, "lora_execution_module_stride_max 31 != the widest"),
                        ({"lora_execution_module_names": None}, "MISSING from the report"),
                        ({"lora_execution_modules_other": "gate_up_proj"}, "MISSING from the report")):
        bad = _report(exp)
        bad.update(over)
        assert any(match in p for p in sa.execution_stride_problems(
            [bad], target_modules=("o_proj",), max_module_dim=D)), (over, match)
    unread = _report(exp)
    del unread["lora_execution_module_strides"]
    assert any("lora_execution_module_strides MISSING" in p for p in sa.execution_stride_problems(
        [unread], target_modules=("o_proj",), max_module_dim=D))
    # a full-name target matches too (vLLM's is_in_target_modules admits the exact module name)
    assert sa.execution_stride_problems(
        [rep], target_modules=("model.layers.0.self_attn.o_proj", "model.layers.1.self_attn.o_proj"),
        max_module_dim=D) == []
    # no class / no D / no reports: never a silent pass
    assert sa.execution_stride_problems([rep], target_modules=(), max_module_dim=D)
    assert sa.execution_stride_problems([rep], target_modules=("o_proj",), max_module_dim=0)
    assert sa.execution_stride_problems([], target_modules=("o_proj",), max_module_dim=D)
    # ... and the launcher's wrapper raises, or refuses a launch without the class / D
    launch = vs.ServerLaunch(cmd=("x",), env={}, host="127.0.0.1", port=1, model="m",
                             max_num_seqs=8, max_num_batched_tokens=8 * 16384, max_model_len=16384,
                             prefix_caching=True, lora_native=True, lora_modules=("cellA", "cellB"),
                             max_loras=2, max_lora_rank=8, max_cpu_loras=2,
                             lora_target_modules=("o_proj",), lora_max_module_dim=4096,
                             lora_kernel_index_ceiling=sa.lora_kernel_index_ceiling(4096))
    assert vs.validate_execution_strides([rep], launch) == [rep]
    with pytest.raises(vs.VllmServerError, match="wider stride than the geometry was held to"):
        vs.validate_execution_strides([wide], launch)
    bare = vs.ServerLaunch(cmd=("x",), env={}, host="127.0.0.1", port=1, model="m", lora_native=True)
    with pytest.raises(vs.VllmServerError, match="no composed lora_max_module_dim / lora_target_modules"):
        vs.validate_execution_strides([rep], bare)


def test_lora_execution_modules_report_reads_the_registry_and_refuses_drift():
    """The worker side of the same finding: the report reads
    ``_adapter_manager.modules`` (the registry ``_create_lora_modules`` fills
    with exactly the modules that passed ``_match_target_modules``), keys the
    o_proj wrappers canonically, and carries other families by raw name."""
    m1, m2 = _fake_model(1, seed=1), _fake_model(2, seed=2)
    rep = wr.lora_execution_modules_report(_Runner([m1, m2]))
    assert rep == {"lora_execution_module_keys": ["L0.self_attn.o_proj", "L1.self_attn.o_proj"],
                   "lora_execution_module_names": [GEMMA_VLLM.format(0), GEMMA_VLLM.format(1)],
                   "lora_execution_modules_n": 2, "lora_execution_modules_other": [],
                   # each wrapper's (in, out) stride and the widest
                   "lora_execution_module_strides": {GEMMA_VLLM.format(0): {"in": 24, "out": 32},
                                                     GEMMA_VLLM.format(1): {"in": 24, "out": 32}},
                   "lora_execution_module_stride_max": 32,
                   # the LoRA buffer / MoE surfaces: the bare _Wrap fixture exposes no
                   # stacked buffers, so the total covers nothing and the unread list
                   # names both wrappers (never empty on a real engine)
                   "lora_execution_moe_wrappers": {},
                   "lora_execution_module_lora_buffer_bytes": 0,
                   "lora_execution_module_lora_buffer_bytes_by_module": {},
                   "lora_execution_module_lora_buffer_unread": [GEMMA_VLLM.format(0),
                                                                GEMMA_VLLM.format(1)]}
    # a restricted engine (target_modules filtered L1 out) reports the subset —
    # the LAUNCHER refuses it against the adapters' keys, the worker just reports
    restricted = _Runner([m1, m2], modules={GEMMA_VLLM.format(0): _wrap()})
    assert wr.lora_execution_modules_report(restricted)["lora_execution_module_keys"] == \
        ["L0.self_attn.o_proj"]
    # the unrestricted default wraps EVERY supported projection: other families ride
    # along by raw name and never enter the o_proj key set
    wide = _Runner([m1], modules={GEMMA_VLLM.format(0): _wrap(), GEMMA_VLLM.format(1): _wrap(),
                                  "language_model.model.layers.0.self_attn.qkv_proj": _wrap(24, (8, 8, 8)),
                                  "language_model.model.layers.0.mlp.gate_up_proj": _wrap(24, (10240, 10240))})
    rep = wr.lora_execution_modules_report(wide)
    # the packed gate_up / qkv wrappers are RECOGNISED execution modules (their
    # packed keys), so nothing rides in `other` here
    assert rep["lora_execution_module_keys"] == ["L0.mlp.gate_up_proj", "L0.self_attn.o_proj",
                                                 "L0.self_attn.qkv_proj", "L1.self_attn.o_proj"]
    assert rep["lora_execution_modules_n"] == 4
    assert rep["lora_execution_modules_other"] == []
    # the packed gate_up wrapper's output stride is the SUM of its slices —
    # the expand kernel's output_d0_stride — and it is the widest executing
    # stride on this (unrestricted) engine; the launcher refuses it against D
    assert rep["lora_execution_module_strides"]["language_model.model.layers.0.mlp.gate_up_proj"] == \
        {"in": 24, "out": 20480}
    assert rep["lora_execution_module_strides"]["language_model.model.layers.0.self_attn.qkv_proj"] == \
        {"in": 24, "out": 24}
    assert rep["lora_execution_module_stride_max"] == 20480
    # the nemotron spelling canonicalises too, and its exact name rides along
    nemo = _Runner([m1], modules={NEMOTRON_VLLM.format(3): _wrap()})
    assert wr.lora_execution_modules_report(nemo)["lora_execution_module_keys"] == \
        ["L3.mixer.o_proj"]
    assert wr.lora_execution_modules_report(nemo)["lora_execution_module_names"] == \
        [NEMOTRON_VLLM.format(3)]
    # a wrapper without the pinned stride surface refuses by name (never a
    # guessed stride)
    class _NoIn:
        output_slices = (32,)

    class _NoSlices:
        input_size = 24

    with pytest.raises(wr.LensWorkerError, match="no positive int input_size"):
        wr.lora_execution_modules_report(_Runner([m1], modules={GEMMA_VLLM.format(0): _NoIn()}))
    with pytest.raises(wr.LensWorkerError, match="no non-empty tuple of positive int output_slices"):
        wr.lora_execution_modules_report(_Runner([m1], modules={GEMMA_VLLM.format(0): _NoSlices()}))
    with pytest.raises(wr.LensWorkerError, match="output_slices"):
        wr.lora_execution_modules_report(_Runner([m1], modules={GEMMA_VLLM.format(0): _wrap(24, ())}))
    # drift and emptiness refuse by name
    with pytest.raises(wr.LensWorkerError, match="modules is EMPTY"):
        wr.lora_execution_modules_report(_Runner([m1], modules={}))
    with pytest.raises(wr.LensWorkerError, match="not the pinned LoRAModelManager module registry"):
        wr.lora_execution_modules_report(_Runner([m1], modules=["a", "b"]))
    with pytest.raises(wr.LensWorkerError, match="no lora_manager"):
        wr.lora_execution_modules_report(_Runner())
    with pytest.raises(wr.LensWorkerError, match="no model_runner"):
        wr.lora_execution_modules_report(None)
    runner = _Runner([m1])
    del runner.lora_manager._adapter_manager.modules
    with pytest.raises(wr.LensWorkerError, match="not the pinned LoRAModelManager module registry"):
        wr.lora_execution_modules_report(runner)


# --------------------------------------------------------------------------
# the CUDA-graph capture surface covers the decode concurrency
# --------------------------------------------------------------------------


def test_cudagraph_coverage_holds_the_capture_surface_to_max_num_seqs(tmp_path):
    """THE FINDING (code review 2026-09-11, P2): `--max-cudagraph-capture-size 1`
    keeps enforce_eager False and a non-NONE mode while the dispatcher
    returns CUDAGraphMode.NONE for every batch above the max (dispatch()
    @0.25.1) — production decode eager under a graphs-on attestation. The
    resolved capture max and the largest captured size must both reach the
    composed max_num_seqs."""
    _, _, exp = _two_adapters(tmp_path)
    S = 8
    ok = _report(exp)          # vLLM's default for S = 8: sizes [1..16], max 16
    assert ok["cudagraph_capture_sizes"] == [1, 2, 4, 8, 16] and ok["max_cudagraph_capture_size"] == 16
    assert sa.cudagraph_coverage_problems([ok, ok], max_num_seqs=S) == []
    # exactly covering is fine
    tight = _report(exp, cudagraph_capture_sizes=[1, 2, 4, 8], max_cudagraph_capture_size=8)
    assert sa.cudagraph_coverage_problems([tight], max_num_seqs=S) == []
    # the finding's shape: max 1 under a non-NONE mode
    one = _report(exp, cudagraph_capture_sizes=[1], max_cudagraph_capture_size=1)
    problems = sa.cudagraph_coverage_problems([one], max_num_seqs=S)
    assert any("max_cudagraph_capture_size 1 < the composed max_num_seqs 8" in p for p in problems)
    assert any("largest cudagraph_capture_size 1 < the composed max_num_seqs 8" in p
               for p in problems)
    # a capture list that stops short of the max (surface drift as well as a gap)
    short = _report(exp, cudagraph_capture_sizes=[1, 2, 4], max_cudagraph_capture_size=16)
    problems = sa.cudagraph_coverage_problems([short], max_num_seqs=S)
    assert any("largest cudagraph_capture_size 4 < the composed max_num_seqs 8" in p
               for p in problems)
    assert any("max(cudagraph_capture_sizes) 4 != max_cudagraph_capture_size 16" in p
               for p in problems)
    # a width above vLLM's default cap (512) is refused by name, never
    # silently served with eager full-width decode
    wide = _report(exp, cudagraph_capture_sizes=[1, 2, 4, 8, 512], max_cudagraph_capture_size=512)
    assert any("max_cudagraph_capture_size 512 < the composed max_num_seqs 1024" in p
               for p in sa.cudagraph_coverage_problems([wide], max_num_seqs=1024))
    # unresolved / eager shapes refuse
    for over, match in (({"max_cudagraph_capture_size": None}, "not an int"),
                        ({"cudagraph_capture_sizes": []}, "expected a non-empty list"),
                        ({"cudagraph_capture_sizes": None}, "expected a non-empty list")):
        assert any(match in p for p in sa.cudagraph_coverage_problems(
            [_report(exp, **over)], max_num_seqs=S)), over
    missing = _report(exp)
    del missing["max_cudagraph_capture_size"]
    assert any("max_cudagraph_capture_size MISSING" in p
               for p in sa.cudagraph_coverage_problems([missing], max_num_seqs=S))
    assert sa.cudagraph_coverage_problems([], max_num_seqs=S)
    assert sa.cudagraph_coverage_problems(None, max_num_seqs=S)
    assert sa.cudagraph_coverage_problems(["x"], max_num_seqs=S)
    # the composer wrapper raises naming the consequence, and refuses a
    # launch with no composed width
    launch = vs.ServerLaunch(cmd=("x",), env={}, host="127.0.0.1", port=1, model="m",
                             max_num_seqs=S, max_num_batched_tokens=S * 16384,
                             max_model_len=16384, prefix_caching=True, lora_native=True)
    assert vs.validate_cudagraph_coverage([ok], launch) == [ok]
    with pytest.raises(vs.VllmServerError, match="would run production decode EAGER"):
        vs.validate_cudagraph_coverage([one], launch)
    bare = vs.ServerLaunch(cmd=("x",), env={}, host="127.0.0.1", port=1, model="m",
                           lora_native=True)
    with pytest.raises(vs.VllmServerError, match="no composed max_num_seqs"):
        vs.validate_cudagraph_coverage([ok], bare)


# --------------------------------------------------------------------------
# the /v1/models per-name root binding
# --------------------------------------------------------------------------


def test_native_models_problems(tmp_path):
    _, dirs, _ = _two_adapters(tmp_path)
    model = "m"
    ok = {"data": [{"id": model}, {"id": "cellA", "root": dirs["cellA"]},
                   {"id": "cellB", "root": dirs["cellB"]}]}
    assert sa.native_models_problems(ok, model=model, adapter_dirs=dirs) == []
    assert sa.models_roots(ok, ["cellA"]) == {"cellA": dirs["cellA"]}
    # a symlink or relative spelling of the same dir resolves equal
    rel = {"data": [{"id": model}, {"id": "cellA", "root": dirs["cellA"] + "/."},
                    {"id": "cellB", "root": dirs["cellB"]}]}
    assert sa.native_models_problems(rel, model=model, adapter_dirs=dirs) == []
    swapped = {"data": [{"id": model}, {"id": "cellA", "root": dirs["cellB"]},
                        {"id": "cellB", "root": dirs["cellA"]}]}
    problems = sa.native_models_problems(swapped, model=model, adapter_dirs=dirs)
    assert len(problems) == 2 and all("binding does not match" in p for p in problems)
    missing = {"data": [{"id": model}, {"id": "cellA", "root": dirs["cellA"]}]}
    assert ["adapter 'cellB' listed 0 times" in p
            for p in sa.native_models_problems(missing, model=model, adapter_dirs=dirs)] == [True]
    dup = {"data": ok["data"] + [{"id": "cellB", "root": dirs["cellB"]}]}
    assert any("adapter 'cellB' listed 2 times" in p
               for p in sa.native_models_problems(dup, model=model, adapter_dirs=dirs))
    noroot = {"data": [{"id": model}, {"id": "cellA"}, {"id": "cellB", "root": dirs["cellB"]}]}
    assert any("carries no root" in p
               for p in sa.native_models_problems(noroot, model=model, adapter_dirs=dirs))
    nobase = {"data": ok["data"][1:]}
    assert any("base model 'm' listed 0 times" in p
               for p in sa.native_models_problems(nobase, model=model, adapter_dirs=dirs))
    assert sa.native_models_problems("nope", model=model, adapter_dirs=dirs)


# --------------------------------------------------------------------------
# the compile-cache env factor: the worker never sees VLLM_API_KEY
# (tinycap-shared-standup warm_cache_miss_diagnosis 2026-09-29, fix F1)
# --------------------------------------------------------------------------

#: a synthetic per-launch token (assembled, so no scanner reads it as a credential)
_PROBE = "-".join(["probe", "value", "1234"])


def _import_in_child(extra_env: dict, probe: str) -> str:
    """Import ``native_worker`` in a FRESH interpreter (vllm / vllm-lens blocked, as the import-isolation test
    does) with ``extra_env`` in its environment, then print ``probe``'s value (or ABSENT)."""
    code = r"""
import os, sys
sys.modules["vllm"] = None
sys.modules["vllm_lens"] = None
from sisteer.capeval.serving import native_worker as nw
print(os.environ.get(%r, "ABSENT"))
""" % probe
    env = {"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin", **extra_env}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_importing_native_worker_drops_the_api_key_from_the_compiling_process_env():
    """vLLM 0.25.1 hashes the VALUE of every known env var — ``VLLM_API_KEY`` included, it is not in
    ``envs.compile_factors()``'s ignore list at the tag — into the compile cache's ``env_hash``; the launcher
    mints a fresh random token per launch and the engine/worker child inherits it, so every launch got a new
    cache directory and no restored tarball could ever hit (probe ``485efd7135`` vs row ``71450296d1`` with
    every config factor equal). The worker extension's module is imported in every worker before
    ``init_device`` / ``load_model`` (``worker_base.py`` 261–263 @0.25.1) — the one hook the harness has in
    the compiling process before the first compile — so its import pops the variable there. Every other
    variable is left alone (the cache root the env hook set must survive)."""
    assert _import_in_child({"VLLM_API_KEY": _PROBE, "VLLM_CACHE_ROOT": "/tmp/cc/vllm"},
                            "VLLM_API_KEY") == "ABSENT"
    assert _import_in_child({"VLLM_API_KEY": _PROBE, "VLLM_CACHE_ROOT": "/tmp/cc/vllm"},
                            "VLLM_CACHE_ROOT") == "/tmp/cc/vllm"
    # no token at all: the import is a no-op, never an error
    assert _import_in_child({}, "VLLM_API_KEY") == "ABSENT"


def test_an_in_process_engine_keeps_the_token_because_the_worker_shares_the_api_servers_interpreter():
    """Under ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` (the lens gate's / smoke's in-process worker) the engine core
    — and so this module's import — runs INSIDE the API-server process, and ``api_server.py`` @0.25.1 builds
    the engine (``run_server_worker`` 712) BEFORE ``build_app`` reads ``envs.VLLM_API_KEY`` for the stock
    guard (252) and constructs the ``--middleware`` guard (285): a pop there would silently drop the stock
    guard and make ``SisteerTokenAuthMiddleware`` refuse to start. So the pop is gated on the engine running
    OUT of process (the default, and every lora-native tinycap launch)."""
    assert _import_in_child({"VLLM_API_KEY": _PROBE, "VLLM_ENABLE_V1_MULTIPROCESSING": "0"},
                            "VLLM_API_KEY") == _PROBE
    assert _import_in_child({"VLLM_API_KEY": _PROBE, "VLLM_ENABLE_V1_MULTIPROCESSING": "1"},
                            "VLLM_API_KEY") == "ABSENT"


def test_drop_api_key_from_compile_env_names_what_it_did():
    """The pure half, on a plain mapping: ``dropped`` (the token was there and is gone), ``absent`` (nothing
    to drop), ``kept_in_process_engine`` (the in-process guard above; the mapping is untouched). The variable
    name is ``vllm_auth.TOKEN_ENV`` — the SAME one the launcher injects and both guards read."""
    from sisteer.capeval.serving import vllm_auth
    assert nw.TOKEN_ENV == vllm_auth.TOKEN_ENV == "VLLM_API_KEY"
    assert nw.ENGINE_IN_PROCESS_ENV == "VLLM_ENABLE_V1_MULTIPROCESSING"
    env = {"VLLM_API_KEY": "t", "OTHER": "x"}
    assert nw.drop_api_key_from_compile_env(env) == "dropped" and env == {"OTHER": "x"}
    assert nw.drop_api_key_from_compile_env(env) == "absent" and env == {"OTHER": "x"}
    env = {"VLLM_API_KEY": "t", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}
    assert nw.drop_api_key_from_compile_env(env) == "kept_in_process_engine"
    assert env == {"VLLM_API_KEY": "t", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}
    # the import-time call recorded its own verdict (this test process carries no token)
    assert nw.COMPILE_ENV_SCRUB in ("dropped", "absent", "kept_in_process_engine")


def test_the_api_server_side_modules_never_import_native_worker():
    """F1's premise: the API-server process keeps ``VLLM_API_KEY`` because nothing it imports pops it. The
    modules that run in that process — ``vllm_entry`` (the ``python -m`` entry the launcher composes),
    ``vllm_auth`` (the ``--middleware`` guard) and the clamp shim's ``sitecustomize`` — plus the launcher
    module that names the extension as an engine-args STRING (``vllm_server``) import ``native_worker``
    nowhere (by AST over their import statements, not by grep: a docstring naming the module is fine)."""
    import ast
    serving = Path(nw.__file__).resolve().parent
    for rel in ("vllm_entry.py", "vllm_auth.py", "vllm_server.py", "vllm_shim/sitecustomize.py"):
        tree = ast.parse((serving / rel).read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [f"{node.module or ''}.{a.name}" for a in node.names]
            assert not any("native_worker" in n for n in names), (rel, names)
