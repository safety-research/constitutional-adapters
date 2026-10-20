"""``train_budget_meta.json`` records the seed a fit ran under (2026-09-15).

The production driver's per-arm ``meta`` dict (train_two_si_objects.py ->
``_save_object(meta_extra=)`` -> the sidecar's top level) now carries
``seed`` = ``train_kwargs.default.seed`` (default 0): before this every
cell's sidecar — the four batch-8 Qwen cells and the eight Llama cells
included — read the seed back as absent, so a replicate's identity had to
be inferred from its config. Pinned through the real fit path on the tiny
model (both CE arms): a declared seed lands as that int on both arms; an
undeclared one lands as 0; the record still passes the shared document
budget verdict.

CPU-only: tiny model (the shared ``tiny_llama`` fixture), synthetic
doc-shaped pairs — never staged pools.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import train_two_si_objects as t2  # noqa: E402

from sisteer.data.pairs import Pair, save_pairs  # noqa: E402
from sisteer.eval.train_budget import document_budget_problems  # noqa: E402

SI = "constv2-mix-1k"
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"
ARMS = ("psr_ce_all", "lora_r4_ce")


def _docs():
    rows = [("Sys scaffold\n\n", "A doc about trees and rivers."),
            ("", "Story body with no masked prefix at all here."),
            ("Second sys\n\n", "Another advice body follows the user turn."),
            ("", "Yet another tiny story body.")]
    return [Pair(prompt=p, pos=pos, neg="", instruction_id=SI)
            for p, pos in rows]


def _held():
    return [Pair(prompt="Held sys\n\n", pos="Held-out document body one.",
                 neg="", instruction_id=SI),
            Pair(prompt="", pos="Held-out story body two, somewhat longer.",
                 neg="", instruction_id=SI)]


def _cfg(tmp_path, arms=ARMS, seed=None):
    """The campaign's CE config shape on the tiny model; ``seed`` declared
    in train_kwargs.default when given, ABSENT otherwise (the frozen
    configs' shape)."""
    hp = tmp_path / "pairs_constv2-heldout.jsonl"
    if not hp.exists():
        save_pairs(_held(), hp)
    default = {"epochs": 2, "holdout_frac": 0.0, "heldout_pool": str(hp),
               "probe_n": 2, "early_stop_min_epochs": 1,
               "early_stop_patience": 2}
    if seed is not None:
        default["seed"] = seed
    return {
        "layer": 1, "max_len": 256,
        "data_mode": "documents",
        "training_budget_kind": "document_tokens",
        "completion_tokens": None,
        "model": TINY,
        "batch_size_by_arm": {a: 2 for a in arms},
        "train_kwargs": {
            "default": default,
            "psr_ce_all": {"lr": 1e-3, "gate_reg": 0.1,
                           "floor_ratio_max": None,
                           "ce_floor_ratio_max": 1.1},
            "lora_r4_ce": {"lr": 1e-3, "lora_rank": 4,
                           "floor_ratio_max": None,
                           "ce_floor_ratio_max": 1.1},
        },
        "si_ids": [SI],
    }


def _pool_with_manifest(tmp_path):
    pool = tmp_path / f"pairs_{SI}.jsonl"
    save_pairs(_docs(), pool)
    Path(str(pool) + ".manifest.json").write_text(json.dumps({
        "cell": SI, "budget": 1000, "seed_salts": ["synthetic"],
        "pool_sha256": hashlib.sha256(pool.read_bytes()).hexdigest()}))
    return pool


def _fit(tiny_llama, tmp_path, cfg, arms):
    model, tok = tiny_llama
    pool = _pool_with_manifest(tmp_path)
    out = tmp_path / "defenses"
    outcome = t2.train_si_objects(model, tok, SI, _docs(), list(arms),
                                  cfg=cfg, save_dir=out, pairs_path=str(pool))
    assert outcome.ok, {a: (r.error_type, r.error)
                        for a, r in outcome.failed.items()}
    metas = {}
    for arm in arms:
        cell = out / f"{SI}__{arm}"
        assert (cell / "COMPLETE").exists()
        metas[arm] = json.loads((cell / "train_budget_meta.json").read_text())
    return metas


def test_sidecar_records_the_declared_seed_on_both_arms(tiny_llama, tmp_path):
    metas = _fit(tiny_llama, tmp_path, _cfg(tmp_path, seed=2), ARMS)
    for arm, meta in metas.items():
        assert meta["seed"] == 2, arm
        assert isinstance(meta["seed"], int) and not isinstance(meta["seed"], bool)
        # beside the record's other explicit fit facts, at the top level
        assert {"batch_size", "effective_batch", "accum_steps", "lr",
                "seed"} <= set(meta)
        assert meta["data_mode"] == "documents"
        assert document_budget_problems(meta) == []


def test_sidecar_records_seed_0_when_the_config_declares_none(tiny_llama,
                                                             tmp_path):
    """The frozen configs' shape (no seed key): the sidecar says 0 — the
    trainer's default, which is what those fits ran under — never absent."""
    cfg = _cfg(tmp_path, arms=("lora_r4_ce",))
    assert "seed" not in cfg["train_kwargs"]["default"]
    metas = _fit(tiny_llama, tmp_path, cfg, ("lora_r4_ce",))
    assert metas["lora_r4_ce"]["seed"] == 0
    assert document_budget_problems(metas["lora_r4_ce"]) == []


def test_meta_builder_source_reads_candidate_kw_seed():
    """The one spelling: the driver's meta dict takes the seed from the
    kwargs the trainer really received (candidate_kw), defaulting to 0."""
    src = (REPO / "scripts/train_two_si_objects.py").read_text()
    assert '"seed": int(candidate_kw.get("seed", 0) or 0),' in src
