"""THE ROUTED-EXPERT LoRA BUFFER RANK and the split default topology (the
2026-10-01 ruling) on a routed-experts model: CPU tests over the worker patch
(``native_moe_lora``, on a fake of vLLM 0.25.1's ``FusedMoE3DWithLoRA`` surface with
real CPU tensors), the composer's reader + env knob + readiness validator
(``vllm_server``), the manifest (``standup_manifest``: S1cc / S1d, ``moe_lora_rank``,
the difference's ``2·r_e``, the expert-adapter bytes law, the config-derived
recurrent state), and the probe's stand-up cross-check (``serving_probe``). No vllm
here: the fake carries exactly the attributes the pinned class reads in the patched
methods. Every dims object is a SYNTHETIC tiny MoE (the fused-3-D-experts +
shared-expert shape); none of the paper models routes experts, so this path is an
OPTION of the serving library and these are its pins.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import torch

from sisteer.capeval.serving import native_moe_lora as nml
from sisteer.capeval.serving import standup_manifest as sm
from sisteer.capeval.serving import vllm_server as vs
from sisteer.extraction.lora_modules import ArchDims

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts", REPO / "scripts" / "sdfbase"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import serving_probe as sp  # noqa: E402

#: a 2-MoE-block toy: hidden 64, one attention block (q_out 128, kv_out 16, o_in 64), 4 routed experts (2 active), I = 32
TOY = ArchDims(hidden=64, attention_layers=1, q_out=128, kv_out=16, o_in=64, moe_layers=2,
               num_experts=4, active_experts=2, moe_intermediate=32, shared_intermediate=32)
#: the same toy with EIGHT active experts of sixteen: FD 6's rule at r = 4 / 16 / 64 reads 1 / 2 / 8 on it
TOY8 = ArchDims(hidden=64, attention_layers=1, q_out=128, kv_out=16, o_in=64, moe_layers=2,
                num_experts=16, active_experts=8, moe_intermediate=32, shared_intermediate=32)
REV = "0123456789abcdef0123456789abcdef01234567"
#: transformers' tiny Qwen3_5Moe TEXT config (tests/test_lora_modules.py's fixture shape): the Gated-DeltaNet fields the
#: recurrent-state derivation reads, as a plain mapping (no transformers needed to pin the arithmetic)
MOE_TEXT = dict(hidden_size=32, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                intermediate_size=64, moe_intermediate_size=16, shared_expert_intermediate_size=16, num_experts=4,
                num_experts_per_tok=2, vocab_size=128, full_attention_interval=4, linear_num_key_heads=2,
                linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4)


# --------------------------------------------------------------------------- a fake of the pinned wrapper surface


class FakeLoRAConfig:
    def __init__(self, max_lora_rank: int, max_loras: int, fully_sharded: bool = False) -> None:
        self.max_lora_rank = max_lora_rank
        self.max_loras = max_loras
        self.lora_dtype = torch.float32
        self.fully_sharded_loras = fully_sharded


class FakeFusedMoEWithLoRA:
    """``fused_moe.py`` @0.25.1's allocation + ``set_lora``, spelled over the attributes
    the real methods read (``local_num_experts``, ``hidden_size``,
    ``intermediate_size_per_partition``, ``fully_sharded``, ``tp_size``, ``device``)."""

    _w13_slices = 2

    def __init__(self, E: int, d: int, I: int, tp_size: int = 1) -> None:
        self.local_num_experts, self.hidden_size, self.intermediate_size_per_partition = E, d, I
        self.tp_size, self.device, self.fully_sharded = tp_size, "cpu", False

    def _create_lora_a_weights(self, max_loras, lora_config):
        R = lora_config.max_lora_rank if not self.fully_sharded else lora_config.max_lora_rank // self.tp_size
        self.w13_lora_a_stacked = tuple(torch.zeros((max_loras, self.local_num_experts, R, self.hidden_size),
                                                    dtype=lora_config.lora_dtype) for _ in range(self._w13_slices))
        self.w2_lora_a_stacked = (torch.zeros((max_loras, self.local_num_experts, lora_config.max_lora_rank,
                                               self.intermediate_size_per_partition), dtype=lora_config.lora_dtype),)

    def _create_lora_b_weights(self, max_loras, lora_config):
        self.w13_lora_b_stacked = tuple(torch.zeros((max_loras, self.local_num_experts, self.intermediate_size_per_partition,
                                                     lora_config.max_lora_rank), dtype=lora_config.lora_dtype)
                                        for _ in range(self._w13_slices))
        self.w2_lora_b_stacked = (torch.zeros((max_loras, self.local_num_experts, self.hidden_size, lora_config.max_lora_rank),
                                              dtype=lora_config.lora_dtype),)

    def create_lora_weights(self, max_loras, lora_config, model_config=None):
        self.fully_sharded = lora_config.fully_sharded_loras
        self._create_lora_a_weights(max_loras, lora_config)
        self._create_lora_b_weights(max_loras, lora_config)

    def set_lora(self, index, lora_a, lora_b):
        # the pinned method copies the leading r rows / columns into the slot (fused_moe.py:385-427); the fake records
        # what reached it — the patch's refusal happens BEFORE this body runs
        self.applied = (index, [tuple(t.shape) for t in lora_a])


class Fake3D(FakeFusedMoEWithLoRA):
    _w13_slices = 1

    def _create_lora_b_weights(self, max_loras, lora_config):
        self.w13_lora_b_stacked = (torch.zeros((max_loras, self.local_num_experts, 2 * self.intermediate_size_per_partition,
                                                lora_config.max_lora_rank), dtype=lora_config.lora_dtype),)
        self.w2_lora_b_stacked = (torch.zeros((max_loras, self.local_num_experts, self.hidden_size, lora_config.max_lora_rank),
                                              dtype=lora_config.lora_dtype),)

    def set_lora(self, index, lora_a, lora_b):
        self.applied = (index, [tuple(t.shape) for t in lora_a])


class FakeManager:
    def __init__(self, lora_config):
        self.lora_config = lora_config

    def get_dummy_lora_warmup_rank(self, default_rank):
        return default_rank


def _own(cls):
    return {k: v for k, v in cls.__dict__.items() if k not in ("__dict__", "__weakref__")}


def _fresh_classes():
    """Fresh class objects per test (the patch mutates the class): the 3-D fake INHERITS the fresh 2-D fake, as
    ``FusedMoE3DWithLoRA`` inherits ``FusedMoEWithLoRA`` (its ``_create_lora_a_weights`` comes from the parent)."""
    c2d = type("FakeFusedMoEWithLoRA", (object,), _own(FakeFusedMoEWithLoRA))
    c3d = type("Fake3D", (c2d,), _own(Fake3D))
    mgr = type("FakeManager", (object,), _own(FakeManager))
    return c2d, c3d, mgr


def _factors(E, r, d, I):
    a13, a2 = torch.randn(E, r, d), torch.randn(E, r, I)
    b13, b2 = torch.randn(E, 2 * I, r), torch.randn(E, d, r)
    return [a13, a2], [b13, b2]


@pytest.fixture(autouse=True)
def _reset():
    nml.reset_for_tests()
    yield
    nml.reset_for_tests()


class TestWorkerPatch:
    def test_buffers_are_allocated_at_the_routed_rank_not_the_cap(self):
        c2d, c3d, mgr = _fresh_classes()
        rep = nml.install_patches(nml.PatchTargets(moe_classes=(c2d, c3d), manager_cls=mgr), routed_rank=16)
        assert rep.armed and rep.routed_rank == 16 and rep.manager_patched
        assert rep.patched_methods[nml._qual(c2d)] == ["_create_lora_a_weights", "_create_lora_b_weights", "set_lora"]
        assert rep.patched_methods[nml._qual(c3d)] == ["_create_lora_b_weights", "set_lora"]   # the 3-D class inherits the A allocation
        cfg = FakeLoRAConfig(max_lora_rank=128, max_loras=4)
        E, d, I = TOY.num_experts, TOY.hidden, TOY.moe_intermediate                               # the toy's routed FFN
        w = c3d(E=E, d=d, I=I)
        w.create_lora_weights(4, cfg)
        assert tuple(w.w13_lora_a_stacked[0].shape) == (4, E, 16, d)
        assert tuple(w.w13_lora_b_stacked[0].shape) == (4, E, 2 * I, 16)
        assert tuple(w.w2_lora_a_stacked[0].shape) == (4, E, 16, I)
        assert tuple(w.w2_lora_b_stacked[0].shape) == (4, E, d, 16)
        nbytes = sum(t.numel() * t.element_size() for t in (w.w13_lora_a_stacked[0], w.w13_lora_b_stacked[0],
                                                             w.w2_lora_a_stacked[0], w.w2_lora_b_stacked[0]))
        # the bytes law at R = 16 in fp32 here (the fake's dtype): 4 slots x E x 16 x (d + 2I + I + d) x 4 B — the same
        # E·R·(2d + 3I) term standup_manifest prices one MoE block at (in bf16)
        assert nbytes == 4 * E * 16 * (d + 2 * I + I + d) * 4
        assert nbytes // 4 // 4 * 2 == sm._stacked_bytes_per_slot(TOY, sm.target_class_for(TOY, "mlp"), 16, 16)["routed_experts"] // TOY.moe_layers
        # unpatched, the same wrapper allocated at the cap: 8x more
        u2d, u3d, _ = _fresh_classes()
        u = u3d(E=E, d=d, I=I)
        u.create_lora_weights(4, cfg)
        assert tuple(u.w13_lora_a_stacked[0].shape) == (4, E, 128, d)
        # the config view delegates everything else and is read-only
        view = nml._RankView(cfg, 16)
        assert view.max_lora_rank == 16 and view.max_loras == 4 and view.lora_dtype is torch.float32
        with pytest.raises(AttributeError):
            view.max_loras = 9

    def test_an_adapter_above_the_routed_rank_is_refused_by_name_never_truncated(self):
        c2d, c3d, mgr = _fresh_classes()
        nml.install_patches(nml.PatchTargets(moe_classes=(c2d, c3d), manager_cls=mgr), routed_rank=4)
        w = c3d(E=4, d=8, I=4)
        w.create_lora_weights(2, FakeLoRAConfig(max_lora_rank=32, max_loras=2))
        a, b = _factors(E=4, r=4, d=8, I=4)
        w.set_lora(0, a, b)
        assert w.applied[0] == 0 and w.applied[1][0] == (4, 4, 8)
        a8, b8 = _factors(E=4, r=8, d=8, I=4)
        with pytest.raises(nml.MoeLoraRankError) as ei:
            w.set_lora(1, a8, b8)
        msg = str(ei.value)
        assert nml.REFUSAL_PHRASE.format(incoming=8, routed=4) in msg
        assert sp.MOE_RANK_REFUSAL_RE.search(msg), msg
        assert sp.classify_failure("", RuntimeError(msg)) == (sp.STATUS_REFUSED, sp.MOE_RANK_REFUSAL_RE.search(msg).group(0))
        assert not hasattr(w, "applied") or w.applied[0] == 0            # slot 1 untouched
        # a lora_b above the rank refuses too (the B side is read on its own axis)
        a4, b4 = _factors(E=4, r=4, d=8, I=4)
        _a6, b6 = _factors(E=4, r=6, d=8, I=4)
        with pytest.raises(nml.MoeLoraRankError):
            w.set_lora(1, a4, b6)

    def test_the_two_d_class_set_lora_is_checked_over_every_slice(self):
        c2d, c3d, mgr = _fresh_classes()
        nml.install_patches(nml.PatchTargets(moe_classes=(c2d, c3d), manager_cls=mgr), routed_rank=2)
        w = c2d(E=2, d=4, I=2)
        w.create_lora_weights(1, FakeLoRAConfig(max_lora_rank=8, max_loras=1))
        ok = [torch.zeros(2, 2, 4), torch.zeros(2, 2, 2), torch.zeros(2, 2, 4)], [torch.zeros(2, 2, 2), torch.zeros(2, 4, 2), torch.zeros(2, 2, 2)]
        w.set_lora(0, *ok)
        bad = [torch.zeros(2, 2, 4), torch.zeros(2, 2, 2), torch.zeros(2, 3, 4)], ok[1]
        with pytest.raises(nml.MoeLoraRankError):
            w.set_lora(0, *bad)

    def test_the_dummy_warmup_rank_is_clamped_to_the_routed_rank(self):
        c2d, c3d, mgr = _fresh_classes()
        nml.install_patches(nml.PatchTargets(moe_classes=(c2d, c3d), manager_cls=mgr), routed_rank=2)
        m = mgr(FakeLoRAConfig(max_lora_rank=32, max_loras=4))
        assert m.get_dummy_lora_warmup_rank(8) == 2 and m.get_dummy_lora_warmup_rank(1) == 1

    def test_fully_sharded_tp_that_cannot_divide_the_rank_refuses_before_vllms_divide(self):
        c2d, c3d, mgr = _fresh_classes()
        nml.install_patches(nml.PatchTargets(moe_classes=(c2d, c3d), manager_cls=mgr), routed_rank=1)
        w = c3d(E=2, d=4, I=2, tp_size=2)
        with pytest.raises(nml.MoeLoraRankError, match="not divisible by tp_size=2"):
            w.create_lora_weights(1, FakeLoRAConfig(max_lora_rank=8, max_loras=1, fully_sharded=True))

    def test_install_is_idempotent_for_one_rank_and_refuses_a_second(self):
        c2d, c3d, mgr = _fresh_classes()
        t = nml.PatchTargets(moe_classes=(c2d, c3d), manager_cls=mgr)
        nml.install_patches(t, 8)
        nml.install_patches(t, 8)
        with pytest.raises(nml.MoeLoraRankError, match="already patched"):
            nml.install_patches(t, 16)

    def test_env_parsing_and_the_report(self, monkeypatch):
        assert nml.routed_rank_from_env({}) is None and nml.routed_rank_from_env({nml.ENV_FLAG: " "}) is None
        assert nml.routed_rank_from_env({nml.ENV_FLAG: "16"}) == 16
        for bad in ("x", "0", "-4", "256"):
            with pytest.raises(nml.MoeLoraRankError):
                nml.routed_rank_from_env({nml.ENV_FLAG: bad})
        assert nml.report() == {nml.REPORT_ARMED_KEY: False, nml.REPORT_RANK_KEY: None, nml.REPORT_CLASSES_KEY: []}
        c2d, c3d, mgr = _fresh_classes()
        nml.install(4, targets=nml.PatchTargets(moe_classes=(c2d, c3d), manager_cls=mgr))
        rep = nml.report()
        assert rep[nml.REPORT_ARMED_KEY] is True and rep[nml.REPORT_RANK_KEY] == 4 and len(rep[nml.REPORT_CLASSES_KEY]) == 2
        # arming from the env without vllm RAISES (a worker asked to size its buffers never comes up padded and quiet)
        monkeypatch.setenv(nml.ENV_FLAG, "8")
        with pytest.raises(nml.MoeLoraRankError, match="vllm is not importable"):
            nml._arm_from_env()

    def test_the_env_knob_is_one_string_across_the_composer_the_worker_and_the_probe(self):
        from sisteer.capeval.serving import native_worker as nw
        assert vs.MOE_LORA_RANK_ENV == nml.ENV_FLAG == "SISTEER_MOE_LORA_RANK"
        assert nml.ENV_FLAG in sp.MOE_RANK_REFUSAL_RE.pattern
        # native_worker imports the module (the import IS the pre-load hook) and folds its report into the RPC payload
        assert nw._native_moe_lora is nml
        import inspect
        assert "_native_moe_lora.report()" in inspect.getsource(nw.NativeWorkerExtension.sisteer_scheduler_config)
        # the composer's experts suffix is the attestation vocabulary's word (never a second spelling)
        from sisteer.capeval.serving import serving_attestation as sa
        assert vs.MOE_EXPERTS_SUFFIX == sa.EXPERTS_LEAF == sm.EXPERTS_TARGET == "experts"


# --------------------------------------------------------------------------- the composer


def _adapter_dir(tmp_path: Path, name: str, *, r: int, rank_pattern: Dict[str, int] | None, expert_rows: int | None,
                 target_modules=("down_proj", "gate_proj", "up_proj")) -> Path:
    from safetensors.torch import save_file
    d = tmp_path / name
    d.mkdir(parents=True)
    cfg: Dict[str, Any] = {"r": r, "lora_alpha": float(r), "target_modules": list(target_modules), "peft_type": "LORA",
                           "base_model_name_or_path": "org/toy"}
    if rank_pattern is not None:
        cfg["rank_pattern"] = dict(rank_pattern)
    (d / "adapter_config.json").write_text(json.dumps(cfg))
    tensors = {"base_model.model.model.layers.0.mlp.shared_expert.down_proj.lora_A.weight": torch.zeros(r, 8),
               "base_model.model.model.layers.0.mlp.shared_expert.down_proj.lora_B.weight": torch.zeros(16, r)}
    if expert_rows is not None:
        tensors["base_model.model.model.layers.0.mlp.experts.lora_A.weight"] = torch.zeros(expert_rows, 8)
        tensors["base_model.model.model.layers.0.mlp.experts.lora_B.weight"] = torch.zeros(16, expert_rows)
        tensors["base_model.model.model.layers.0.mlp.experts.base_layer.lora_A.weight"] = torch.zeros(expert_rows, 8)
        tensors["base_model.model.model.layers.0.mlp.experts.base_layer.lora_B.weight"] = torch.zeros(32, expert_rows)
    save_file(tensors, str(d / "adapter_model.safetensors"))
    return d


class TestComposer:
    def test_fitted_moe_lora_rank_is_the_largest_declared_routed_rank_cross_checked_against_the_file(self, tmp_path):
        rx = r"model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)"
        c = _adapter_dir(tmp_path, "control", r=64, rank_pattern={rx: 8}, expert_rows=4 * 8)
        d = _adapter_dir(tmp_path, "difference", r=128, rank_pattern={rx: 16}, expert_rows=4 * 16)
        z = _adapter_dir(tmp_path, "zero", r=128, rank_pattern={rx: 16}, expert_rows=4 * 16)
        assert vs.fitted_moe_lora_rank([c, d, z]) == 16
        assert vs.fitted_moe_lora_rank([c]) == 8
        # a routed-expert adapter without a rank_pattern cannot be sized: refused by name
        bad = _adapter_dir(tmp_path, "nopattern", r=16, rank_pattern=None, expert_rows=4 * 2)
        with pytest.raises(vs.VllmServerError, match="declares no rank_pattern"):
            vs.fitted_moe_lora_rank([c, bad])
        # a declaration the file contradicts (rows not divisible by r_e) refuses
        lie = _adapter_dir(tmp_path, "lie", r=16, rank_pattern={rx: 3}, expert_rows=4 * 2)
        with pytest.raises(vs.VllmServerError, match="does not divide"):
            vs.fitted_moe_lora_rank([lie])
        # dense-only adapters everywhere: nothing to size (the caller only asks for an experts class)
        dense = _adapter_dir(tmp_path, "dense", r=4, rank_pattern=None, expert_rows=None, target_modules=("o_proj",))
        with pytest.raises(vs.VllmServerError, match="none of the registered adapters declares a routed-expert rank"):
            vs.fitted_moe_lora_rank([dense])
        with pytest.raises(vs.VllmServerError):
            vs.fitted_moe_lora_rank([])

    def test_validate_moe_lora_rank_holds_every_wrapper_to_the_composed_value(self):
        mk = lambda **kw: vs.ServerLaunch(cmd=("vllm",), env={}, host="h", port=1, model="m", lora_native=True, **kw)  # noqa: E731
        good = [{"moe_lora_rank_armed": True, "moe_lora_rank_composed": 16,
                 "lora_execution_moe_wrappers": {"model.layers.0.mlp.experts": {"max_lora_rank": 16},
                                                 "model.layers.1.mlp.experts": {"max_lora_rank": 16}}}]
        frag = vs.validate_moe_lora_rank(good, mk(moe_lora_rank=16))
        assert frag["composed"] == 16 and frag["env"] == vs.MOE_LORA_RANK_ENV and frag["wrappers_per_worker"][0] == {
            "model.layers.0.mlp.experts": 16, "model.layers.1.mlp.experts": 16}
        with pytest.raises(vs.VllmServerError, match="allocated at R != 16"):
            vs.validate_moe_lora_rank([{**good[0], "lora_execution_moe_wrappers": {"x": {"max_lora_rank": 128}}}], mk(moe_lora_rank=16))
        with pytest.raises(vs.VllmServerError, match="armed=False"):
            vs.validate_moe_lora_rank([{**good[0], "moe_lora_rank_armed": False}], mk(moe_lora_rank=16))
        with pytest.raises(vs.VllmServerError, match="reports no fused-MoE LoRA wrapper"):
            vs.validate_moe_lora_rank([{**good[0], "lora_execution_moe_wrappers": {}}], mk(moe_lora_rank=16))
        # a launch that composed no routed rank must see neither an armed patch nor an expert wrapper
        none = [{"moe_lora_rank_armed": False, "moe_lora_rank_composed": None, "lora_execution_moe_wrappers": {}}]
        assert vs.validate_moe_lora_rank(none, mk())["composed"] is None
        with pytest.raises(vs.VllmServerError, match="composed no routed rank"):
            vs.validate_moe_lora_rank(good, mk())
        # a dense launch's worker report carries the unarmed keys (native_worker folds report() unconditionally): clean
        assert vs.validate_moe_lora_rank([{**nml.report(), "lora_execution_moe_wrappers": {}}], mk())["armed_per_worker"] == [
            {"armed": False, "composed": None}]

    def test_the_server_launch_field_defaults_to_none(self):
        launch = vs.ServerLaunch(cmd=("vllm",), env={}, host="h", port=1, model="m")
        assert launch.moe_lora_rank is None

    def test_the_readiness_fold_names_the_moe_fragment(self):
        """``_verify_native_lora_state`` runs the validator between the stride check and the model-runner check and
        carries its fragment as ``moe_lora_buffers`` (source pin: the fold is one function, no live RPC here)."""
        import inspect
        src = inspect.getsource(vs._verify_native_lora_state)
        assert "moe_lora_buffers = validate_moe_lora_rank(reports, launch)" in src and '"moe_lora_buffers": moe_lora_buffers' in src
        assert src.index("validate_execution_strides(reports, launch)") < src.index("validate_moe_lora_rank(reports, launch)")


# --------------------------------------------------------------------------- the manifest


class TestManifest:
    def test_split_default_topology_and_the_routed_rank_per_server(self):
        for set_id in ("qkvo", "attn_mlp", "mlp", "down"):
            ms = sm.build_manifests(TOY8, set_id=set_id, rank=64)
            assert [m.topology for m in ms] == ["S1cc", "S1d", "S2"] and all(m.fallback_of is None for m in ms)
            cc, d, s2 = ms
            assert cc.genuine_roles == ("control", "constitution") and d.genuine_roles == ("difference",)
            assert cc.max_lora_rank == 64 and d.max_lora_rank == 128 and cc.max_loras == 3 and d.max_loras == 2
            assert cc.zero.rank == 64 and d.zero.rank == 128
            if set_id == "qkvo":
                assert cc.moe_lora_rank is None and d.moe_lora_rank is None and not cc.wraps_experts
            else:
                assert cc.moe_lora_rank == 8 and d.moe_lora_rank == 16 and cc.wraps_experts and d.wraps_experts
                assert "|moe8|" in cc.launch_key and "|moe16|" in d.launch_key
                assert cc.target_class == tuple(sorted(sm.MODULE_SETS[set_id] + ("experts",)))
            assert s2.moe_lora_rank is None and not s2.wraps_experts
        cc, d = sm.build_manifests(TOY8, set_id="wo", rank=16)
        assert cc.topology == "S1cc" and d.topology == "S1d" and cc.max_loras == 7 and d.max_loras == 2
        assert sorted(a.kind for a in cc.adapters) == ["carrier"] * 4 + ["genuine"] * 2 + ["zero"]
        assert cc.max_lora_rank == 16 and d.max_lora_rank == 32 and cc.moe_lora_rank is None
        assert sm.topology_label(sm.build_manifests(TOY8, set_id="mlp", rank=4)) == "S1ccS1dS2"
        assert sm.topology_label(sm.build_manifests(TOY8, set_id="wo", rank=4)) == "S1ccS1d"
        # the 2-active toy: r = 4 -> r_e 2 on the single-rank adapters, 4 on the difference AND the zero (8 // 2)
        cc, d, _ = sm.build_manifests(TOY, set_id="mlp", rank=4)
        assert [a.per_expert_rank for a in cc.adapters] == [2, 2, 4] and [a.per_expert_rank for a in d.adapters] == [4, 4]
        assert (cc.moe_lora_rank, d.moe_lora_rank) == (4, 4)

    def test_the_difference_routed_rank_is_twice_the_parents_not_r_e_of_2r(self):
        # 8 active experts: r=4 -> r_e 1, difference 2 (the old rule gave per_expert_rank(8) = 1)
        assert sm.difference_per_expert_rank(TOY8, sm.MODULE_SETS["mlp"], 4) == 2
        assert sm.difference_per_expert_rank(TOY8, sm.MODULE_SETS["mlp"], 16) == 4
        assert sm.difference_per_expert_rank(TOY8, sm.MODULE_SETS["mlp"], 64) == 16
        assert sm.difference_per_expert_rank(TOY8, sm.MODULE_SETS["wo"], 4) is None
        cc, d, _ = sm.build_manifests(TOY8, set_id="mlp", rank=4)
        assert next(a for a in d.adapters if a.role == "difference").per_expert_rank == 2
        assert cc.moe_lora_rank == 1 and d.moe_lora_rank == 2
        assert sm.pair_per_expert_rank(TOY8, "mlp", 64) == {"shared_expert": 64, "routed_expert": 8}
        assert sm.pair_per_expert_rank(TOY8, "wo", 4) is None and sm.pair_per_expert_rank(TOY, "down", 4) == {"shared_expert": 4, "routed_expert": 2}

    def test_moe_lora_rank_is_validated_and_round_trips_and_legacy_records_derive_it(self):
        import dataclasses
        cc, d, _ = sm.build_manifests(TOY8, set_id="mlp", rank=64)
        with pytest.raises(sm.ManifestError, match="never truncated"):
            dataclasses.replace(d, moe_lora_rank=8)
        with pytest.raises(sm.ManifestError, match="> cap"):
            dataclasses.replace(cc, moe_lora_rank=128)
        wo = sm.build_manifests(TOY8, set_id="wo", rank=4)[0]
        with pytest.raises(sm.ManifestError, match="wraps no expert"):
            dataclasses.replace(wo, moe_lora_rank=1)
        # an experts class whose slots state no routed rank cannot size its buffers
        with pytest.raises(sm.ManifestError, match="no adapter states a per_expert_rank"):
            dataclasses.replace(cc, adapters=tuple(dataclasses.replace(a, per_expert_rank=None) for a in cc.adapters), moe_lora_rank=None)
        # a record written before the key existed carries none: the adapters' need is derived
        legacy = d.as_dict()
        legacy.pop("moe_lora_rank")
        assert sm.Manifest.from_dict(legacy).moe_lora_rank == 16
        assert sm.Manifest.from_dict(json.loads(json.dumps(d.as_dict()))) == d
        # a DENSE record (no per_expert_rank key on its slots at all) reads back with None everywhere
        slot = {"name": "control__lora", "kind": "genuine", "lora_modules": ["o_proj"], "rank": 4, "role": "control"}
        assert sm.AdapterSlot.from_dict(slot).per_expert_rank is None

    def test_the_bytes_law_prices_the_experts_at_the_routed_rank(self):
        cc, d, _ = sm.build_manifests(TOY8, set_id="mlp", rank=64)
        E, dd, I, I_s, n = TOY8.num_experts, TOY8.hidden, TOY8.moe_intermediate, TOY8.shared_intermediate, TOY8.moe_layers
        e = sm.expected_lora_bytes(cc, 64, TOY8)
        assert e["moe_lora_rank"] == 8 and e["rank_by_family"] == {"shared_expert": 64, "routed_experts": 8}
        assert e["per_slot"]["routed_experts"] == n * E * 8 * (dd + 2 * I + I + dd) * 2
        assert e["per_slot"]["shared_expert"] == n * ((2 * 64 * dd + 2 * I_s * 64) + (64 * I_s + dd * 64)) * 2
        assert e["total"] == e["per_slot_total"] * 3
        assert abs(e["per_slot_mib_per_R"]["routed_experts"] - n * E * (2 * dd + 3 * I) * 2 / 2**20) < 1e-12   # per unit of R_e
        e2 = sm.expected_lora_bytes(d, 128, TOY8)
        assert e2["per_slot"]["routed_experts"] == n * E * 16 * (2 * dd + 3 * I) * 2 and e2["max_loras"] == 2
        # the whole pair's LoRA buffers against the old four-slot server at cap 128 with the experts padded to the cap
        old = sm.expected_lora_bytes(cc, 128, TOY8, moe_lora_rank=128)["per_slot_total"] * 4
        assert sm.expected_lora_bytes(cc, 128, TOY8, moe_lora_rank=128)["per_slot"]["routed_experts"] == 16 * e["per_slot"]["routed_experts"]
        assert (e["total"] + e2["total"]) / old < 0.2
        # the executing strides and the warmup activation term carry the MoE wrappers
        strides = sm.executing_strides(TOY8, cc.target_class)
        assert {"mlp.experts.w13", "mlp.experts.w2", "mlp.shared_expert.gate_up_proj", "mlp.shared_expert.down_proj"} == set(strides)
        assert strides["mlp.experts.w13"] == {"in": dd, "out": 2 * I} and strides["mlp.experts.w2"] == {"in": I, "out": dd}
        mie = sm.mlp_intermediate_elements(TOY8)
        assert mie["moe_block"] == 8 * I + I_s == mie["elements"] and mie["dense_block"] is None

    def test_the_recurrent_state_is_derived_from_the_text_config_never_a_typed_constant(self):
        """``gated_deltanet_state_per_seq_bytes`` on the tiny Qwen3_5Moe text config: 3 linear blocks (4 blocks, full attention
        every 4th) x (SSM 4 x 8 x 8 + conv (4 - 1) x (2 x 2 x 8 + 4 x 8)) x 2 B = 2,688; 0 for a dense config; a partial
        GDN block refuses by name; a top-level multimodal config's text_config is read; layer_types wins over the interval."""
        assert sm.gated_deltanet_state_per_seq_bytes(MOE_TEXT) == 3 * (4 * 8 * 8 + 3 * (2 * 2 * 8 + 4 * 8)) * 2 == 2_688
        assert sm.gated_deltanet_state_per_seq_bytes(MOE_TEXT, dtype_bytes=4) == 2 * 2_688
        assert sm.gated_deltanet_state_per_seq_bytes({"hidden_size": 32, "num_hidden_layers": 2}) == 0
        assert sm.gated_deltanet_state_per_seq_bytes({"text_config": dict(MOE_TEXT)}) == 2_688
        with_types = {**MOE_TEXT, "layer_types": ["linear_attention", "full_attention", "linear_attention", "full_attention"]}
        assert sm.gated_deltanet_state_per_seq_bytes(with_types) == 2 * (256 + 192) * 2
        with pytest.raises(sm.ManifestError, match="states .* but not"):
            sm.gated_deltanet_state_per_seq_bytes({k: v for k, v in MOE_TEXT.items() if k != "linear_conv_kernel_dim"})
        with pytest.raises(sm.ManifestError, match="not a positive int"):
            sm.gated_deltanet_state_per_seq_bytes({**MOE_TEXT, "linear_num_value_heads": 0})
        assert sm.kv_bytes_per_seq(TOY, 4096, recurrent_state_bytes=2_688) == sm.kv_bytes_per_token(TOY) * 4096 + 2_688
        transformers = pytest.importorskip("transformers")
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig
        cfg = Qwen3_5MoeConfig(text_config=dict(MOE_TEXT))
        assert cfg.model_type == "qwen3_5_moe" and sm.gated_deltanet_state_per_seq_bytes(cfg) == 2_688
        assert sm.gated_deltanet_state_per_seq_bytes(cfg.text_config) == 2_688
        # the probe's plan-time derivation: a declared value wins, an unreadable config prices 0 and says why
        assert sp.recurrent_state_for("org/toy", REV, declared_mib=1.5) == (int(1.5 * 2**20), "declared")
        got, why = sp.recurrent_state_for("org/toy", REV)
        assert got == 0 and why.startswith("config_unreadable:")
        plan = sp.build_plan(TOY, model_key="qwen35_4b_demo", model_id="org/toy", revision=REV, sets=("mlp",), ranks=(4,),
                             windows=(4096,), ladder_caps=(), recurrent_state_per_seq_bytes=2_688)
        assert plan["budget"]["recurrent_state_per_seq_bytes"] == 2_688 and plan["is_moe"] is True
        assert all(r["admission"]["kv_per_seq_bytes"] == sm.kv_bytes_per_token(TOY) * 4096 + 2_688 for r in plan["launches"])
        assert plan["pairs"][0]["per_expert_rank"] == {"shared_expert": 4, "routed_expert": 2}


# --------------------------------------------------------------------------- the probe's stand-up cross-check


class TestStandUpCrossCheck:
    def test_a_composed_routed_rank_that_disagrees_with_the_manifest_refuses(self, tmp_path, monkeypatch):
        """The probe's stand-up holds the composer's SISTEER_MOE_LORA_RANK to the manifest's moe_lora_rank (both None for a
        class without experts)."""
        import dataclasses
        from sisteer.capeval.serving import native_standup as ns
        cc = sm.build_manifests(TOY8, set_id="mlp", rank=64)[0]
        calls = {}

        class FakeLaunch:
            def __init__(self, moe):
                self.lora_target_modules = tuple(cc.target_class); self.max_lora_rank = cc.max_lora_rank; self.max_loras = cc.max_loras
                self.moe_lora_rank = moe; self.cmd = ("vllm", "serve", "--moe-backend", "triton"); self.env = {}; self.port = 1; self.model = "m"
        monkeypatch.setattr(ns, "write_chat_template", lambda out, b: (out / "t.jinja", "h"))
        monkeypatch.setattr(sp, "_chat_template_bytes", lambda m, r: b"x")
        monkeypatch.setattr(sp, "_tree_sha", lambda: None)
        monkeypatch.setattr(vs, "launch_only_spec", lambda *a, **k: object())
        monkeypatch.setattr(vs, "build_server_launch", lambda *a, **k: FakeLaunch(calls["moe"]))
        monkeypatch.setattr(vs, "multimodal_target", lambda m, r: False)
        monkeypatch.setattr(vs, "launch_server", lambda launch, **k: "proc")
        monkeypatch.setattr(dataclasses, "replace", lambda obj, **kw: obj)
        calls["moe"] = 7
        with pytest.raises(sp.LaunchRefused, match="composer_moe_rank_mismatch"):
            sp._production_stand_up(model_id="m", revision=REV, adapter_dirs={}, manifest=cc, L=4096, S=1, out_dir=tmp_path,
                                    gpu_mem_util=0.9, ready_timeout_s=1, tp=1)
        calls["moe"] = None
        with pytest.raises(sp.LaunchRefused, match="composer_moe_rank_mismatch"):
            sp._production_stand_up(model_id="m", revision=REV, adapter_dirs={}, manifest=cc, L=4096, S=1, out_dir=tmp_path,
                                    gpu_mem_util=0.9, ready_timeout_s=1, tp=1)
        calls["moe"] = 8
        launch, proc = sp._production_stand_up(model_id="m", revision=REV, adapter_dirs={}, manifest=cc, L=4096, S=1, out_dir=tmp_path,
                                               gpu_mem_util=0.9, ready_timeout_s=1, tp=1)
        assert proc == "proc" and launch.moe_lora_rank == 8
