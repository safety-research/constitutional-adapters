"""Checkpointed == uncheckpointed training, per hook arm: activation
checkpointing is a memory transform, and after the
backward-inside-apply() fix (lockstep._masked_student_step /
psr._psr_step_loss call site) it must not change what a fit LEARNS. torch
itself enforces recompute determinism (the CheckpointError this campaign
hit was that check working); these tests pin the other half — that the
gradients, and therefore the trained objects, match the uncheckpointed
path to numerical tolerance. A fix that silences the error but changes
gradients would fail here.

Runs the PRODUCTION driver end-to-end (train_si_objects) on the tiny CPU
model, fresh model + object per run, identical seeds: the only difference
between the two runs is the checkpointing flag. On CPU fp32 the recompute
replays identical ops in identical order, so parity is near-bitwise; the
assertion uses a tolerance a real divergence (wrong hooks in recompute,
stale mask state) would exceed by orders of magnitude.

token_faithful is not here: the tiny tokenizer has no single-token
sentinel for it, and its checkpointed fit was proven directly at
production scale (the Qwen H200 canary + memprobe — it has no layer
hooks, so it never had the bug).
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "_t2_parity", REPO / "scripts/train_two_si_objects.py")
t2 = importlib.util.module_from_spec(spec)
sys.modules["_t2_parity"] = t2
spec.loader.exec_module(t2)

SI = "safety_refusal"


def _pairs():
    from sisteer.data.pairs import Pair

    data = [("Say something nice.", "Bonjour le monde", "Hello world"),
            ("Describe a tree.", "C'est un grand arbre", "It is a tall tree"),
            ("Greet the reader.", "Salut a tous", "Hi everyone")]
    return [Pair(prompt=p, pos=pos, neg=neg, instruction_id=SI)
            for p, pos, neg in data]


def _cfg(arm: str, ckpt: bool) -> dict:
    cfg = {
        "layer": 1, "completion_tokens": 16, "max_len": 256,
        "max_len_by_si": {SI: 256},
        "teacher_prefix_cache": False,
        "batch_size_by_arm": {arm: 2},
        "train_kwargs": {
            "default": {"epochs": 1, "seed": 0, "holdout_frac": 0.0,
                        "floor_ratio_max": None, "probe_n": 2,
                        "monitors": False, "positions": "response",
                        "response_header_len": 4},
            arm: ({"gradient_checkpointing": True} if ckpt else {}),
        },
        "si_ids": [SI],
    }
    return t2._cfg_with_positions(cfg, "response")


def _fit(arm: str, ckpt: bool):
    import torch

    from sisteer.models import load_model_and_tokenizer

    # Fresh model per run: the tiny-random checkpoint's weights are fixed
    # bytes from the hub, so both runs start identical.
    model, tok = load_model_and_tokenizer(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        device="cpu", dtype=torch.float32)
    cfg = _cfg(arm, ckpt)
    if ckpt:
        assert t2.maybe_enable_gradient_checkpointing(
            model, cfg, arms=[arm]) is True
        assert model.is_gradient_checkpointing and model.training
    outcome = t2.train_si_objects(model, tok, SI, _pairs(), [arm],
                                  cfg=cfg, save_dir=None)
    assert outcome.ok, {a: (r.error_type, r.error)
                        for a, r in outcome.failed.items()}
    obj = outcome.objects[arm]
    return [p.detach().clone() for p in obj.trainable]


@pytest.mark.parametrize("arm", ["psr_mse_all", "psr_kl_all",
                                 "lora_r1_all"])
def test_ckpt_matches_unckpt_trained_object(arm):
    import torch

    plain = _fit(arm, ckpt=False)
    ckpt = _fit(arm, ckpt=True)
    assert len(plain) == len(ckpt) and len(plain) > 0
    moved = 0.0
    for i, (a, b) in enumerate(zip(plain, ckpt)):
        diff = float((a - b).abs().max())
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5), (
            f"{arm}: trainable[{i}] diverges between checkpointed and "
            f"uncheckpointed fits (max abs diff {diff:.3e}) — the "
            "checkpointing path is changing gradients")
        moved += float(a.abs().sum())
    # parity of two no-op fits would be vacuous — something must have
    # actually trained
    assert moved > 0.0


# -- the document-CE arms (sdf-base-steering §5.1 contract (iii)) -------------
# train_object_ce runs forward + loss assembly + BACKWARD inside
# obj.apply(): the qwen fits train under non-reentrant activation
# checkpointing, whose backward RECOMPUTES the forward — the hooks must be
# live then, and the recompute must not change what the fit LEARNS. Same
# contract, same driver path (train_si_objects), doc-shaped config: the
# only difference between the two runs is the checkpointing flag.

CE_SI = "constv2-mix-1k"


def _ce_docs():
    from sisteer.data.pairs import Pair

    rows = [("Sys scaffold\n\n", "A doc body about trees and rivers."),
            ("", "Story body with no masked prefix at all here."),
            ("Second sys\n\n", "Another advice body follows the turn."),
            ("", "One more tiny story body.")]
    return [Pair(prompt=p, pos=pos, neg="", instruction_id=CE_SI)
            for p, pos in rows]


def _ce_cfg(arm: str, ckpt: bool, heldout_path) -> dict:
    arm_kw = {"lr": 1e-3, "floor_ratio_max": None, "accum_steps": 2}
    if arm == "psr_ce_all":
        arm_kw["gate_reg"] = 0.1
    if ckpt:
        arm_kw["gradient_checkpointing"] = True
    return {
        "layer": 1, "max_len": 256,
        "data_mode": "documents",
        "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "teacher_prefix_cache": False,
        "batch_size_by_arm": {arm: 2},
        "train_kwargs": {
            "default": {"epochs": 2, "seed": 0, "holdout_frac": 0.0,
                        "heldout_pool": str(heldout_path), "probe_n": 2,
                        "monitors": False},
            arm: arm_kw,
        },
        "si_ids": [CE_SI],
    }


def _ce_fit(arm: str, ckpt: bool, heldout_path):
    import torch

    from sisteer.models import load_model_and_tokenizer

    model, tok = load_model_and_tokenizer(
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        device="cpu", dtype=torch.float32)
    cfg = _ce_cfg(arm, ckpt, heldout_path)
    if ckpt:
        assert t2.maybe_enable_gradient_checkpointing(
            model, cfg, arms=[arm]) is True
        assert model.is_gradient_checkpointing and model.training
    outcome = t2.train_si_objects(model, tok, CE_SI, _ce_docs(), [arm],
                                  cfg=cfg, save_dir=None)
    assert outcome.ok, {a: (r.error_type, r.error)
                        for a, r in outcome.failed.items()}
    return [p.detach().clone() for p in outcome.objects[arm].trainable]


@pytest.mark.parametrize("arm", ["psr_ce_all", "lora_r4_ce"])
def test_ce_arm_ckpt_matches_unckpt_trained_object(arm, tmp_path):
    import torch

    from sisteer.data.pairs import Pair, save_pairs

    hp = tmp_path / "pairs_constv2-heldout.jsonl"
    save_pairs([Pair(prompt="Held sys\n\n", pos="Held-out doc body one.",
                     neg="", instruction_id=CE_SI),
                Pair(prompt="", pos="Held-out story body two, longer.",
                     neg="", instruction_id=CE_SI)], hp)
    plain = _ce_fit(arm, ckpt=False, heldout_path=hp)
    ckpt = _ce_fit(arm, ckpt=True, heldout_path=hp)
    assert len(plain) == len(ckpt) and len(plain) > 0
    moved = 0.0
    for i, (a, b) in enumerate(zip(plain, ckpt)):
        diff = float((a - b).abs().max())
        assert torch.allclose(a, b, atol=1e-6, rtol=1e-5), (
            f"{arm}: trainable[{i}] diverges between checkpointed and "
            f"uncheckpointed CE fits (max abs diff {diff:.3e}) — the "
            "recompute is not seeing the hooks the loss was assembled "
            "under (backward left the apply() context?)"
        )
        moved += float(a.abs().sum())
    assert moved > 0.0
