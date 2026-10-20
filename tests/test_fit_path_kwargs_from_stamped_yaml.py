"""The selected architecture must reach the trainer through the real production chain —
scripts/sbatch/sdfbase_fit_cell.sbatch -> scripts/sdfbase/slurm_fit_cell.py (the trainer argv) -> scripts/train_two_si_objects.py (the
per-arm kwargs, fit_selection popped) -> sisteer.methods._train_lora_ce / _train_psr_ce -> extraction.object_factory.build_object. Driven here
from the Qwen3.5-4B's committed configs/sdfbase_qwen35_4b_base_fits.yaml with a SYNTHETIC architecture block in the shape a fold writes (the
committed config carries none; the block states the paper's pick for that model, read off reproduce/models.yaml: rank 4 on o_proj, no
per-expert map, zero init) with the real functions at every hop (the model
is a tiny Llama snapshot; the kwargs, the argv and the object construction are the production code), not from a stub: before this fix the
trainer received no lora_modules / per_expert_rank / init_scheme and built its defaults under a sidecar claiming another architecture."""
from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import train_two_si_objects as t2  # noqa: E402

YAML = REPO / "configs" / "sdfbase_qwen35_4b_base_fits.yaml"
MODELS = REPO / "reproduce" / "models.yaml"
#: the paper's pick for the Qwen3.5-4B (reproduce/models.yaml `lora_architecture`: rank 4, the o_proj set, no per-expert map) with the
#: paper's zero init — what the stamped block below states and what the built object must embody
PICK = {"lora_rank": 4, "lora_modules": ["o_proj"], "per_expert_rank": None, "init_scheme": "zero"}
#: a selection NO dense object can embody: the MLP set at rank 16 with a routed-expert map (the parity refusal's probe)
ROUTED_CLAIM = {"lora_rank": 16, "lora_modules": ["down_proj", "gate_proj", "up_proj"],
                "per_expert_rank": {"shared_expert": 16, "routed_expert": 2}, "init_scheme": "random_lecun_1e-3"}


def _load(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path); mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cfg() -> dict:
    """The 4B's committed fits config plus a SYNTHETIC architecture block in the shape a fold writes for its pick: the promoted top-level keys
    (lora_modules / per_expert_rank / init_scheme, the way make_sdfbase_configs writes them) and the `fit_selection` sidecar block
    stating the same architecture — sourced from reproduce/models.yaml so the two tables cannot drift apart silently."""
    base = yaml.safe_load(YAML.read_text())
    arch = yaml.safe_load(MODELS.read_text())["models"]["qwen35_4b"]["lora_architecture"]
    assert (arch["rank"], list(arch["modules"]), arch["per_expert_rank"]) == (PICK["lora_rank"], PICK["lora_modules"], None), arch
    lora = dict(base["train_kwargs"]["lora_r4_ce"])
    assert lora["lora_rank"] == PICK["lora_rank"] and "fit_selection" not in lora, lora
    lora.update(lora_modules=list(PICK["lora_modules"]), per_expert_rank=None, init_scheme=PICK["init_scheme"])
    lora["fit_selection"] = {"fit_search_id": "paper-pick", "fit_lr_selected_by": "search", "architecture_selection_ruling_id": None, **PICK}
    psr = dict(base["train_kwargs"]["psr_ce_all"])
    psr.update(init_scheme="zero", fit_selection={"fit_search_id": "paper-pick", "init_scheme": "zero"})
    return {**base, "train_kwargs": {**base["train_kwargs"], "lora_r4_ce": lora, "psr_ce_all": psr}}


def test_the_real_trainer_argv_names_the_stamped_config_and_the_lora_arm(tmp_path):
    sfc = _load("slurm_fit_cell", REPO / "scripts" / "sdfbase" / "slurm_fit_cell.py")
    cmd = sfc.trainer_command(sys.executable, REPO, config=str(YAML), si="constv2-mix-full", arm="lora_r4_ce", pairs="/vol/pool/x/pairs.jsonl",
                              cells_dir=tmp_path / "cells", expect_sha="0" * 16, checkpoint_dir=tmp_path / "ck", checkpoint_every=100,
                              overwrite=False, fit_width=1, torchrun_log_dir=tmp_path / "torchrun")
    assert str(REPO / "scripts" / "train_two_si_objects.py") in cmd and "train" in cmd
    assert cmd[cmd.index("--config") + 1] == str(YAML) and cmd[cmd.index("--arms") + 1] == "lora_r4_ce"
    assert "--expected-width" not in cmd and "torchrun" not in " ".join(cmd)      # the 4B fits single-process (no top-level fit_width)
    wide = sfc.trainer_command(sys.executable, REPO, config=str(YAML), si="constv2-mix-full", arm="lora_r4_ce", pairs="/vol/pool/x/pairs.jsonl",
                               cells_dir=tmp_path / "cells", expect_sha="0" * 16, checkpoint_dir=tmp_path / "ck", checkpoint_every=100,
                               overwrite=False, fit_width=2, torchrun_log_dir=tmp_path / "torchrun")
    assert wide[wide.index("--expected-width") + 1] == "2" and "--nproc_per_node=2" in wide      # A wide fit is gated on its width


def test_the_kwargs_that_reach_the_trainer_carry_the_picked_architecture(cfg):
    """The fitter's ONE construction of the per-arm trainer kwargs (arm_trainer_kwargs — what the solo path protects and passes) from the
    stamped yaml: the picked rank / modules / per-expert map / init, equal to the sidecar block, and none of the driver-only keys."""
    kw = t2.arm_trainer_kwargs(cfg["train_kwargs"], "lora_r4_ce")
    assert {k: kw[k] for k in PICK} == PICK, kw
    # the 4B's folded HPO winners travel untouched beside the architecture keys
    assert kw["lr"] == pytest.approx(0.0012600020062978761) and kw["warmup_frac"] == 0.15 and kw["max_grad_norm"] == pytest.approx(0.3364459977095236)
    assert not (set(kw) & set(t2.DRIVER_ONLY_ARM_KEYS)), kw
    sel = cfg["train_kwargs"]["lora_r4_ce"]["fit_selection"]
    assert {k: sel[k] for k in PICK} == {k: kw[k] for k in PICK}, "the sidecar and the trainer read the same architecture"
    psr = t2.arm_trainer_kwargs(cfg["train_kwargs"], "psr_ce_all")
    assert psr["init_scheme"] == "zero" == cfg["train_kwargs"]["psr_ce_all"]["fit_selection"]["init_scheme"]
    assert "fit_selection" not in psr and "gate_reg" in psr


def test_the_trainers_name_every_promoted_key_explicitly():
    """Protected kwargs that no trainer names would TypeError the arm; keys only **kw swallows would be silently ignored — both trainers
    declare the promoted keys as named parameters, so the config value is the one the object is built from."""
    from sisteer import methods
    lora = inspect.signature(methods._train_lora_ce).parameters
    assert {"lora_rank", "lora_modules", "per_expert_rank", "init_scheme"} <= set(lora)
    assert "init_scheme" in inspect.signature(methods._train_psr_ce).parameters


def _tiny_dense():
    """A dense tiny model (every block carries self_attn.o_proj and a dense MLP), the shape the 4B's pick is built on."""
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    return LlamaForCausalLM(LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                                        num_key_value_heads=2, vocab_size=96, max_position_embeddings=64))


def _build(model, kw: dict):
    from sisteer.extraction.object_factory import build_object
    return build_object(model, object_kind="lora", lora_rank=int(kw["lora_rank"]), lora_modules=kw["lora_modules"],
                        per_expert_rank=kw["per_expert_rank"], init_scheme=kw["init_scheme"], seed=0)


def test_the_object_built_from_the_stamped_kwargs_embodies_the_pick_and_passes_the_publish_parity(cfg, tmp_path):
    """The trainer-side construction, for real: object_factory.build_object with the stamped kwargs yields rank 4 on o_proj with no
    per-expert map and the zero init; its blob passes the publish parity under the stamped selection, and a selection claiming
    another rank over the same blob is refused naming the key — the sidecar may publish ONLY what the saved object embodies."""
    kw = t2.arm_trainer_kwargs(cfg["train_kwargs"], "lora_r4_ce")
    obj = _build(_tiny_dense(), kw)
    d = obj.describe()
    assert (d["lora_rank"], sorted(d["lora_modules"]), d["per_expert_rank"], d["init_scheme"]) == (4, PICK["lora_modules"], None, "zero"), d
    blob = obj.to_blob()
    assert t2.artefact_architecture(blob) == PICK
    cell = tmp_path / "cell"; cell.mkdir(); torch.save(blob, cell / "object.pt")
    assert t2.write_fit_selection(cell, cfg, "constv2-mix-full", "lora_r4_ce") == cell / t2.FIT_SELECTION_NAME
    sel = cfg["train_kwargs"]["lora_r4_ce"]["fit_selection"]
    wrong_rank = {**cfg, "train_kwargs": {**cfg["train_kwargs"], "lora_r4_ce": {**cfg["train_kwargs"]["lora_r4_ce"], "fit_selection": {**sel, "lora_rank": 16}}}}
    with pytest.raises(t2.CellAuditError, match=r"lora_rank: selection 16 vs artefact 4"):
        t2.write_fit_selection(cell, wrong_rank, "constv2-mix-full", "lora_r4_ce")


def test_a_dense_object_cannot_embody_a_routed_map(cfg, tmp_path):
    """The dense path pinned: no routed experts -> the construction's per_expert_rank is None whatever set is named, a dense selection
    (null, no rule) passes parity, and a selection claiming a routed-expert map over a dense object is refused naming per_expert_rank."""
    kw = t2.arm_trainer_kwargs(cfg["train_kwargs"], "lora_r4_ce")
    obj = _build(_tiny_dense(), {**kw, **ROUTED_CLAIM, "per_expert_rank": None})
    assert obj.describe()["per_expert_rank"] is None and sorted(obj.describe()["lora_modules"]) == ROUTED_CLAIM["lora_modules"]
    cell = tmp_path / "dense"; cell.mkdir(); torch.save(obj.to_blob(), cell / "object.pt")
    sel = cfg["train_kwargs"]["lora_r4_ce"]["fit_selection"]
    dense_sel = {**sel, **ROUTED_CLAIM, "per_expert_rank": None}
    dense = {**cfg, "train_kwargs": {**cfg["train_kwargs"], "lora_r4_ce": {**cfg["train_kwargs"]["lora_r4_ce"], **ROUTED_CLAIM, "per_expert_rank": None,
                                                                            "fit_selection": dense_sel}}}
    assert t2.write_fit_selection(cell, dense, "constv2-mix-full", "lora_r4_ce") == cell / t2.FIT_SELECTION_NAME
    routed = {**cfg, "train_kwargs": {**cfg["train_kwargs"], "lora_r4_ce": {**cfg["train_kwargs"]["lora_r4_ce"], **ROUTED_CLAIM,
                                                                             "fit_selection": {**sel, **ROUTED_CLAIM}}}}
    with pytest.raises(t2.CellAuditError, match=r"per_expert_rank: selection \{'shared_expert': 16, 'routed_expert': 2\} vs artefact None"):
        t2.write_fit_selection(cell, routed, "constv2-mix-full", "lora_r4_ce")


def _tiny_qwen_moe(active: int, n_experts: int = 8):
    """A tiny Qwen3.5-MoE-shaped model (transformers' Qwen3_5Moe tiny config: routed + shared experts, Gated-DeltaNet blocks, `active`
    experts per token) — the fused-3-D-experts + shared-expert shape the routed-expert serving option is written for, the fixture
    tests/test_lora_modules.py and tests/test_zero_adapter_per_class.py build on. None of the paper models routes experts."""
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration
    torch.manual_seed(0)
    cfg = Qwen3_5MoeConfig(text_config=dict(hidden_size=32, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                            intermediate_size=64, moe_intermediate_size=16, shared_expert_intermediate_size=16,
                                            num_experts=n_experts, num_experts_per_tok=active, vocab_size=128, full_attention_interval=4,
                                            linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=8, linear_value_head_dim=8,
                                            linear_conv_kernel_dim=4))
    return Qwen3_5MoeForConditionalGeneration(cfg)


def test_a_routed_map_that_contradicts_fd6s_rule_is_refused_at_construction_on_a_tiny_moe(cfg, tmp_path):
    """On the tiny Qwen3_5Moe fixture (the routed-expert path kept as an option): a stamped
    selection whose per_expert_rank {shared 16, routed 2} IS FD 6's rule for 8 active experts builds on an 8-active MoE (rank 16, the
    MLP set, the LeCun init) and its blob passes the publish parity under that selection with the rule and the active count
    recorded; the SAME stamped map over a model with ANOTHER active-expert count (2 of 4 — the rule says routed 8) is refused at
    CONSTRUCTION naming the rule, before any training; and the pre-fix sidecar (per_expert_rank null, no rule) over the MoE blob
    is refused at publish naming per_expert_rank."""
    kw = {**t2.arm_trainer_kwargs(cfg["train_kwargs"], "lora_r4_ce"), **ROUTED_CLAIM}
    obj = _build(_tiny_qwen_moe(active=8), kw)
    d = obj.describe()
    assert (d["lora_rank"], sorted(d["lora_modules"]), d["per_expert_rank"], d["init_scheme"]) == (
        16, ROUTED_CLAIM["lora_modules"], ROUTED_CLAIM["per_expert_rank"], "random_lecun_1e-3"), d
    blob = obj.to_blob()
    assert t2.artefact_architecture(blob) == ROUTED_CLAIM
    cell = tmp_path / "moe"; cell.mkdir(); torch.save(blob, cell / "object.pt")
    sel = cfg["train_kwargs"]["lora_r4_ce"]["fit_selection"]
    routed_sel = {**sel, **ROUTED_CLAIM, "per_expert_rank_rule": "max(1,r//active_experts)", "moe_active_experts": 8}
    routed = {**cfg, "train_kwargs": {**cfg["train_kwargs"], "lora_r4_ce": {**cfg["train_kwargs"]["lora_r4_ce"], **ROUTED_CLAIM,
                                                                             "fit_selection": routed_sel}}}
    assert t2.write_fit_selection(cell, routed, "constv2-mix-full", "lora_r4_ce") == cell / t2.FIT_SELECTION_NAME
    # the pre-fix world: the selection said null while the construction derived the map -> refused by name at publish
    pre_sel = {k: v for k, v in routed_sel.items() if k not in ("per_expert_rank_rule", "moe_active_experts")} | {"per_expert_rank": None}
    pre = {**cfg, "train_kwargs": {**cfg["train_kwargs"], "lora_r4_ce": {**cfg["train_kwargs"]["lora_r4_ce"], **ROUTED_CLAIM,
                                                                          "per_expert_rank": None, "fit_selection": pre_sel}}}
    with pytest.raises(t2.CellAuditError, match=r"per_expert_rank: selection None vs artefact \{'shared_expert': 16, 'routed_expert': 2\}"):
        t2.write_fit_selection(cell, pre, "constv2-mix-full", "lora_r4_ce")
    # a model whose active-expert count disagrees with the stamped map refuses at CONSTRUCTION, before any training
    with pytest.raises(ValueError, match="contradicts FD 6's rule"):
        _build(_tiny_qwen_moe(active=2, n_experts=4), kw)
