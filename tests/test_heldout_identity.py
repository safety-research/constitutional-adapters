"""The shared held-out pool's CONTENT is part of every cell's fit identity
(sdf-base-steering fix item 1) + the sanctioned heldout tokcache path (item 8).

The campaign's whole verdict for a CE cell is its per-epoch CE on the ONE
shared held-out pool (train_kwargs.default.heldout_pool). Before this,
mutating the pool's bytes at the same path left every COMPLETE marker,
checkpoint and budget sidecar validating — a resumed cell could carry an
early-stopping decision made against data that no longer exists. Pinned
here:

- the pool's sha256 joins ``cell_fit_fingerprint`` (via the ONE shared
  derivation ``heldout_pool_identity``), so mutated held-out bytes refuse
  the COMPLETE skip AND the checkpoint load — both key on the fingerprint;
- legacy (non-heldout_pool) fingerprints are byte-identical to the
  pre-change payload (no key is added when no pool is declared);
- a declared-but-missing pool refuses loudly BEFORE any resume decision;
- on resume, ``check_budget_sidecars`` compares the current hash against
  the sidecar's recorded ``heldout_pool_sha`` and hard-refuses a mismatch;
- ``precompute-tokens``/``build_token_sidecar`` accept the (deliberately
  unregistered) heldout pool id ``constv2-heldout`` in doc mode, producing
  a sidecar identical to a corpus cell's modulo the ``si_id`` stamp.

CPU-only: tiny model, synthetic doc-shaped pairs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_two_si_objects as t2

from sisteer.data.pairs import Pair, save_pairs
from sisteer.extraction import fitckpt
from sisteer.extraction.fit_fingerprint import fit_fingerprint

SI = "constv2-mix-1k"
ARM = "lora_r4_ce"
TINY = "hf-internal-testing/tiny-random-LlamaForCausalLM"


def _docs():
    rows = [("Sys scaffold\n\n", "A doc about trees and rivers."),
            ("", "Story body with no masked prefix at all here."),
            ("Second sys\n\n", "Another advice body follows the user turn."),
            ("", "Yet another tiny story body.")]
    return [Pair(prompt=p, pos=pos, neg="", instruction_id=SI)
            for p, pos in rows]


def _held(extra: int = 0):
    rows = [("Held sys\n\n", "Held-out document body one."),
            ("", "Held-out story body two, somewhat longer.")]
    rows += [("", f"Mutated extra held-out body number {i}.")
             for i in range(extra)]
    return [Pair(prompt=p, pos=pos, neg="", instruction_id=SI)
            for p, pos in rows]


def _cfg(tmp_path, *, heldout: bool = True) -> dict:
    hp = tmp_path / "pairs_constv2-heldout.jsonl"
    if heldout and not hp.exists():
        save_pairs(_held(), hp)
    default = {"epochs": 2, "seed": 0, "holdout_frac": 0.0, "probe_n": 2,
               "early_stop_min_epochs": 1, "early_stop_patience": 2}
    if heldout:
        default["heldout_pool"] = str(hp)
    return {
        "layer": 1, "max_len": 256,
        "data_mode": "documents",
        "training_budget_kind": "document_tokens",
        "completion_tokens": None,
        "model": TINY,
        "batch_size_by_arm": {ARM: 2},
        "train_kwargs": {
            "default": default,
            ARM: {"lr": 1e-3, "lora_rank": 4, "floor_ratio_max": None,
                  "ce_floor_ratio_max": 1.1},
        },
        "si_ids": [SI],
    }


def _fp(cfg) -> str:
    return t2.cell_fit_fingerprint(
        cfg, SI, ARM, positions="all",
        pool_sha256=t2.pool_content_sha256(_docs()),
        config_sha=t2._cfg_dict_sha(cfg))


def _mutate_heldout(cfg) -> None:
    """Different BYTES at the SAME path (one extra doc appended)."""
    hp = Path(cfg["train_kwargs"]["default"]["heldout_pool"])
    save_pairs(_held(extra=1), hp)


# --------------------------------------------------------------------------- #
# fingerprint identity                                                          #
# --------------------------------------------------------------------------- #


def test_heldout_sha_joins_the_fit_identity(tmp_path):
    """The fingerprint binds the pool's CONTENT hash (in the optimizer
    extensible slot, the teacher-leg convention) and reproduces the shared
    utility byte-for-byte — mutated held-out bytes at the same path are a
    different fit identity."""
    from sisteer.extraction.lockstep import lockstep_compatible

    cfg = _cfg(tmp_path)
    base = _fp(cfg)
    assert base == _fp(cfg)  # deterministic

    hp_path, hp_sha = t2.heldout_pool_identity(cfg, SI)
    assert hp_path == cfg["train_kwargs"]["default"]["heldout_pool"]
    assert hp_sha == t2._sha256_file(Path(hp_path))
    eff = t2.effective_optimizer_settings(cfg, ARM,
                                          lockstep=lockstep_compatible(ARM))
    manual = fit_fingerprint({
        "model": TINY, "model_revision": None, "si_id": SI, "arm": ARM,
        "positions": "all",
        "pool_sha256": t2.pool_content_sha256(_docs()),
        "optimizer": {**{k: v for k, (v, _s) in eff.items()},
                      "teacher": "online", "heldout_pool_sha": hp_sha},
        "config_sha": t2._cfg_dict_sha(cfg)})
    assert manual == base

    _mutate_heldout(cfg)
    assert _fp(cfg) != base


def test_legacy_fingerprints_stay_byte_identical(tmp_path):
    """No declared heldout_pool -> NO key added: the fingerprint equals
    the pre-change payload exactly, so every existing COMPLETE marker and
    checkpoint of the legacy campaigns still validates."""
    from sisteer.extraction.lockstep import lockstep_compatible

    cfg = _cfg(tmp_path, heldout=False)
    assert t2.heldout_pool_identity(cfg, SI) == (None, None)
    eff = t2.effective_optimizer_settings(cfg, ARM,
                                          lockstep=lockstep_compatible(ARM))
    legacy_manual = fit_fingerprint({
        "model": TINY, "model_revision": None, "si_id": SI, "arm": ARM,
        "positions": "all",
        "pool_sha256": t2.pool_content_sha256(_docs()),
        "optimizer": {**{k: v for k, (v, _s) in eff.items()},
                      "teacher": "online"},
        "config_sha": t2._cfg_dict_sha(cfg)})
    assert _fp(cfg) == legacy_manual


def test_declared_but_missing_heldout_pool_refuses_before_any_skip(tmp_path):
    """The sha is part of the identity, so a fingerprint computed against
    a missing pool file must refuse loudly — never guess, never let a
    COMPLETE-skip or checkpoint decision run on an unresolvable identity."""
    cfg = _cfg(tmp_path, heldout=False)
    cfg["train_kwargs"]["default"]["heldout_pool"] = str(
        tmp_path / "nowhere.jsonl")
    with pytest.raises(SystemExit, match="heldout_pool"):
        _fp(cfg)


# --------------------------------------------------------------------------- #
# COMPLETE skip + budget-sidecar resume + checkpoint load                       #
# --------------------------------------------------------------------------- #


def test_mutated_heldout_refuses_complete_skip_and_resume(tiny_llama,
                                                          tmp_path):
    """End to end through the production driver: a published cell resumes
    (skipped_existing) under the SAME held-out bytes; after mutating them
    at the same path, (a) the COMPLETE marker no longer stands in for the
    requested fit (validated_complete -> None: skip refused), and (b) the
    budget-sidecar resume gate HARD-refuses the recorded-vs-current
    heldout_pool_sha mismatch — never a silent resume across a changed
    held-out world."""
    model, tok = tiny_llama
    cfg = _cfg(tmp_path)
    out = tmp_path / "defenses"
    cell_id = f"{SI}__{ARM}"

    outcome = t2.train_si_objects(model, tok, SI, _docs(), [ARM], cfg=cfg,
                                  save_dir=out)
    assert outcome.ok, {a: (r.error_type, r.error)
                        for a, r in outcome.failed.items()}
    fp_orig = _fp(cfg)
    assert t2.validated_complete(out, cell_id, fp_orig) is not None
    meta = json.loads((out / cell_id / "train_budget_meta.json").read_text())
    assert meta["heldout_pool_sha"] == t2.heldout_pool_identity(cfg, SI)[1]

    # Same bytes -> ordinary resume skip.
    again = t2.train_si_objects(model, tok, SI, _docs(), [ARM], cfg=cfg,
                                save_dir=out)
    assert again.arms[ARM].status == "skipped_existing"

    _mutate_heldout(cfg)
    fp_new = _fp(cfg)
    assert fp_new != fp_orig
    # (a) COMPLETE skip refused: the marker no longer matches the request.
    assert t2.validated_complete(out, cell_id, fp_new) is None
    assert t2.validated_complete(out, cell_id, fp_orig) is not None
    # (b) the resume gate refuses the mismatch loudly.
    with pytest.raises(SystemExit, match="heldout_pool_sha"):
        t2.train_si_objects(model, tok, SI, _docs(), [ARM], cfg=cfg,
                            save_dir=out)


def test_budget_sidecar_gate_details(tmp_path):
    """check_budget_sidecars directly: a recorded heldout_pool_sha that
    matches passes; a mismatch is HARD; a sidecar WITHOUT the key (every
    legacy SI-pair fit) never compares."""
    cell_dir = tmp_path / f"{SI}__{ARM}"
    cell_dir.mkdir(parents=True)
    obj = cell_dir / "object.pt"
    obj.write_bytes(b"synthetic")
    from sisteer.eval.defense_cell import sha256_file

    meta = {"cell_id": f"{SI}__{ARM}", "max_len": 256,
            "object_sha256": sha256_file(obj),
            "training_budget_kind": "document_tokens",
            "sequence_max_len": 256, "completion_tokens": None,
            "sample_manifest_sha": None, "heldout_pool_sha": "a" * 64}
    (cell_dir / "train_budget_meta.json").write_text(json.dumps(meta))

    kw = dict(max_len=256, completion_tokens=None,
              budget_kind="document_tokens")
    assert t2.check_budget_sidecars(tmp_path, SI, [ARM],
                                    heldout_pool_sha="a" * 64, **kw) == []
    with pytest.raises(SystemExit, match="heldout_pool_sha"):
        t2.check_budget_sidecars(tmp_path, SI, [ARM],
                                 heldout_pool_sha="b" * 64, **kw)
    # a recorded sha vs a CURRENT absence is a mismatch too
    with pytest.raises(SystemExit, match="heldout_pool_sha"):
        t2.check_budget_sidecars(tmp_path, SI, [ARM],
                                 heldout_pool_sha=None, **kw)
    # legacy sidecar (no key): never compared
    meta2 = {"cell_id": f"{SI}__{ARM}", "max_len": 256,
             "completion_tokens": 512, "object_sha256": sha256_file(obj)}
    (cell_dir / "train_budget_meta.json").write_text(json.dumps(meta2))
    assert t2.check_budget_sidecars(
        tmp_path, SI, [ARM], max_len=256, completion_tokens=512,
        budget_kind=None, heldout_pool_sha="b" * 64) == []


def test_mutated_heldout_refuses_checkpoint_load(tmp_path):
    """The checkpoint key IS the fingerprint (the driver's _ckpt_kwargs
    joins _cell_fp), so a checkpoint written under the original held-out
    bytes is IGNORED — loudly, fresh start — once they change."""
    cfg = _cfg(tmp_path)
    fp_old = _fp(cfg)
    ckdir = tmp_path / "ckpt" / f"{SI}__{ARM}"
    ckdir.mkdir(parents=True)
    torch.save({"checkpoint_key": fp_old, "step": 7},
               str(ckdir / fitckpt.LATEST_NAME))
    got = fitckpt.load_checkpoint(ckdir, fp_old)
    assert got is not None and got["step"] == 7

    _mutate_heldout(cfg)
    fp_new = _fp(cfg)
    assert fp_new != fp_old
    assert fitckpt.load_checkpoint(ckdir, fp_new) is None  # load refused


# --------------------------------------------------------------------------- #
# item 8 — the sanctioned heldout tokcache path                                 #
# --------------------------------------------------------------------------- #


def test_precompute_tokens_accepts_the_heldout_pool_id(tiny_llama, tmp_path):
    """build_token_sidecar (the precompute-tokens entrypoint) accepts
    ``constv2-heldout`` in doc mode — the id is deliberately unregistered
    (an eval-only pool, never a training cell) — and produces the
    IDENTICAL sidecar a registered corpus-cell build would, modulo the
    si_id stamp."""
    _, tok = tiny_llama
    pool = tmp_path / "pairs_constv2-heldout.jsonl"
    save_pairs(_held(), pool)
    cfg = _cfg(tmp_path)

    out_h = tmp_path / "heldout.tokcache.npz"
    out_c = tmp_path / "corpus.tokcache.npz"
    assert t2.HELDOUT_POOL_SI == "constv2-heldout"
    t2.build_token_sidecar(cfg, t2.HELDOUT_POOL_SI, str(pool),
                           out=str(out_h), tok=tok)
    t2.build_token_sidecar(cfg, SI, str(pool), out=str(out_c), tok=tok)

    with np.load(out_h, allow_pickle=False) as zh, \
            np.load(out_c, allow_pickle=False) as zc:
        hh = json.loads(str(zh["meta"][0]))
        hc = json.loads(str(zc["meta"][0]))
        assert hh["si_id"] == t2.HELDOUT_POOL_SI
        assert hc["si_id"] == SI
        # identical modulo the stamp
        assert {k: v for k, v in hh.items() if k != "si_id"} == \
            {k: v for k, v in hc.items() if k != "si_id"}
        for key in ("hashes", "flags", "offsets", "ids"):
            assert (zh[key] == zc[key]).all(), key

    # and the standard reader warms it (the fits' --token-cache path)
    from sisteer.extraction import tokcache

    cache = tokcache.warm_from_sidecar(out_h, tok,
                                       expect_model_id=cfg["model"],
                                       expect_pool_path=str(pool))
    assert len(cache) > 0


def test_precompute_tokens_still_refuses_other_unregistered_ids(
        tiny_llama, tmp_path):
    """The special case is EXACTLY the heldout pool id: any other
    unregistered id still fails in milliseconds (typo'd cell ids must
    never reach fit time)."""
    _, tok = tiny_llama
    pool = tmp_path / "pairs_bogus.jsonl"
    save_pairs(_held(), pool)
    cfg = _cfg(tmp_path)
    with pytest.raises(KeyError):
        t2.build_token_sidecar(cfg, "constv2-bogus-1k", str(pool),
                               out=str(tmp_path / "x.npz"), tok=tok)


def test_precompute_tokens_accepts_the_control_heldout_pool_id(tiny_llama,
                                                                tmp_path):
    """The sanctioned path is a SET (HELDOUT_POOL_SIS): the control
    corpus's held-out pool ``constv2ctl2-heldout`` — the sdfbase-llama8b
    lane scores both held-out pools per fit and stages a
    sidecar for each — takes the same held-out route as ``constv2-heldout``
    (no resolve_instruction, which refuses the unregistered id), and the
    sidecar header stamps ``si_id: constv2ctl2-heldout``. The historical
    constant keeps its spelling, and the build is identical to the
    constitution held-out's modulo the stamp."""
    _, tok = tiny_llama
    assert t2.HELDOUT_POOL_SI == "constv2-heldout"
    assert t2.HELDOUT_POOL_SIS == frozenset({"constv2-heldout",
                                             "constv2ctl2-heldout"})
    pool = tmp_path / "pairs_constv2ctl2-heldout.jsonl"
    save_pairs(_held(), pool)
    cfg = _cfg(tmp_path)

    out_ctl = tmp_path / "ctl_heldout.tokcache.npz"
    out_h = tmp_path / "heldout.tokcache.npz"
    t2.build_token_sidecar(cfg, "constv2ctl2-heldout", str(pool),
                           out=str(out_ctl), tok=tok)
    t2.build_token_sidecar(cfg, t2.HELDOUT_POOL_SI, str(pool),
                           out=str(out_h), tok=tok)
    with np.load(out_ctl, allow_pickle=False) as zc, \
            np.load(out_h, allow_pickle=False) as zh:
        hc = json.loads(str(zc["meta"][0]))
        hh = json.loads(str(zh["meta"][0]))
        assert hc["si_id"] == "constv2ctl2-heldout"
        assert hh["si_id"] == "constv2-heldout"
        assert {k: v for k, v in hc.items() if k != "si_id"} == \
            {k: v for k, v in hh.items() if k != "si_id"}
        for key in ("hashes", "flags", "offsets", "ids"):
            assert (zc[key] == zh[key]).all(), key

    # the CLI entrypoint (`precompute-tokens --si constv2ctl2-heldout
    # --pairs ... --out ...`) routes through the same build
    from types import SimpleNamespace

    out_cli = tmp_path / "ctl_cli.tokcache.npz"
    t2.cmd_precompute_tokens(
        SimpleNamespace(si="constv2ctl2-heldout", pairs=str(pool),
                        out=str(out_cli)), cfg)
    with np.load(out_cli, allow_pickle=False) as z:
        assert json.loads(str(z["meta"][0]))["si_id"] == "constv2ctl2-heldout"

    # the standard reader warms it
    from sisteer.extraction import tokcache

    cache = tokcache.warm_from_sidecar(out_ctl, tok,
                                       expect_model_id=cfg["model"],
                                       expect_pool_path=str(pool))
    assert len(cache) > 0

    # the set is EXACTLY the two held-out ids: any other unregistered id
    # still refuses in milliseconds
    with pytest.raises(KeyError):
        t2.build_token_sidecar(cfg, "constv2ctl3-heldout", str(pool),
                               out=str(tmp_path / "x.npz"), tok=tok)
