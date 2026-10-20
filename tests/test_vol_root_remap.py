"""SISTEER_VOL_ROOT (2026-09-06): the campaign configs name their shared
held-out pool by its Modal-volume path (/vol/pool/<key>/...), and the config
FILE's bytes are part of every cell's fit identity, so the path cannot be
rewritten per host. On a cluster node (no /vol, / not writable) the env var
names the directory that plays the volume's root. Pinned here:

- unset: every path is returned unchanged (today's behaviour byte for byte);
- set: /vol/... resolves under the root, anything else is untouched;
- heldout_pool_identity returns the DECLARED path and the CONTENT sha read
  through the remap — so the same config bytes fingerprint identically
  wherever the held-out bytes sit, and differently when the bytes differ
  (content, never path, is the identity);
- a declared-but-missing pool refuses loudly, naming the resolved path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for _sub in ("scripts", "scripts/sdfbase"):
    _p = str(REPO / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import train_two_si_objects as t2  # noqa: E402
from sisteer.data.pairs import Pair, save_pairs  # noqa: E402

SI = "constv2-mix-1k"
ARM = "lora_r4_ce"


def _pairs(tag: str):
    return [Pair(prompt="", pos=f"Document body {tag} one.", neg="",
                 instruction_id=SI),
            Pair(prompt="Sys\n\n", pos=f"Document body {tag} two, longer.",
                 neg="", instruction_id=SI)]


def _cfg(heldout_path: str) -> dict:
    return {
        "layer": 1, "max_len": 256, "data_mode": "documents",
        "training_budget_kind": "document_tokens", "completion_tokens": None,
        "model": "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "model_revision": "0" * 40,
        "batch_size_by_arm": {ARM: 1},
        "train_kwargs": {
            "default": {"seed": 0, "holdout_frac": 0.0,
                        "heldout_pool": heldout_path, "probe_n": 2,
                        "early_stop_min_epochs": 1,
                        "early_stop_patience": 2},
            ARM: {"lr": 1e-3, "lora_rank": 4, "lr_schedule": "constant",
                  "floor_ratio_max": None, "ce_floor_ratio_max": 1.1},
        },
        "train_kwargs_by_si": {SI: {"epochs": 2}},
        "si_ids": [SI],
    }


def _write_cfg(path: Path, heldout: str) -> Path:
    path.write_text(yaml.safe_dump(_cfg(heldout), sort_keys=False))
    return path


def _fp(cfg_path: Path, pairs: Path) -> str:
    cfg = t2._cfg_with_positions(t2._load_cfg(str(cfg_path)), "all")
    pool_sha, config_sha = t2._cli_fit_identity(
        SimpleNamespace(pairs=str(pairs), config=str(cfg_path)), cfg)
    return t2.cell_fit_fingerprint(cfg, SI, ARM, positions="all",
                                   pool_sha256=pool_sha,
                                   config_sha=config_sha, teacher="online")


def test_unset_is_identity(monkeypatch):
    monkeypatch.delenv(t2.VOL_ROOT_ENV, raising=False)
    for p in ("/vol/pool/k/x.jsonl", "/vol", "relative/x", "/other/vol/x"):
        assert t2.resolve_vol_path(p) == p
    monkeypatch.setenv(t2.VOL_ROOT_ENV, "   ")   # whitespace = unset
    assert t2.resolve_vol_path("/vol/pool/k/x.jsonl") == "/vol/pool/k/x.jsonl"


def test_set_remaps_only_vol_paths(monkeypatch, tmp_path):
    monkeypatch.setenv(t2.VOL_ROOT_ENV, str(tmp_path / "root"))
    assert t2.resolve_vol_path("/vol/pool/k/x.jsonl") == \
        str(tmp_path / "root" / "pool" / "k" / "x.jsonl")
    assert t2.resolve_vol_path("/vol") == str(tmp_path / "root")
    for p in ("/volume/x", "/other/vol/x", "relative/vol/x", "/data/x"):
        assert t2.resolve_vol_path(p) == p
    assert t2.resolve_vol_path(Path("/vol/a")) == str(tmp_path / "root" / "a")


def test_identity_returns_declared_path_and_content_sha(monkeypatch,
                                                         tmp_path):
    root = tmp_path / "root"
    hp = root / "pool" / "k" / "pairs_constv2-heldout.jsonl"
    hp.parent.mkdir(parents=True)
    save_pairs(_pairs("held"), hp)
    monkeypatch.setenv(t2.VOL_ROOT_ENV, str(root))
    cfg = _cfg("/vol/pool/k/pairs_constv2-heldout.jsonl")
    path, sha = t2.heldout_pool_identity(cfg, SI)
    assert path == "/vol/pool/k/pairs_constv2-heldout.jsonl"   # DECLARED
    assert sha == t2._sha256_file(hp)                            # CONTENT


def test_fingerprint_follows_content_never_location(monkeypatch, tmp_path):
    """Same config bytes (declaring /vol/...), the held-out bytes staged
    under two different roots -> identical fingerprints; different held-out
    bytes under a third root -> a different fingerprint."""
    pairs = tmp_path / "pairs_constv2-mix-1k.jsonl"
    save_pairs(_pairs("train"), pairs)
    cfg_path = _write_cfg(tmp_path / "cfg.yaml",
                          "/vol/pool/k/pairs_constv2-heldout.jsonl")
    fps = {}
    for tag, held in (("A", "held"), ("B", "held"), ("C", "OTHER")):
        root = tmp_path / f"root{tag}"
        hp = root / "pool" / "k" / "pairs_constv2-heldout.jsonl"
        hp.parent.mkdir(parents=True)
        save_pairs(_pairs(held), hp)
        monkeypatch.setenv(t2.VOL_ROOT_ENV, str(root))
        fps[tag] = _fp(cfg_path, pairs)
    assert fps["A"] == fps["B"]
    assert fps["A"] != fps["C"]
    # and the fit_cell_identity module (the Slurm driver's gate) derives the
    # same value through the same functions
    from fit_cell_identity import expected_fit_fingerprint
    monkeypatch.setenv(t2.VOL_ROOT_ENV, str(tmp_path / "rootA"))
    assert expected_fit_fingerprint(cfg_path, pairs, SI, ARM) == fps["A"]


def test_missing_pool_refuses_naming_the_resolved_path(monkeypatch,
                                                        tmp_path):
    monkeypatch.setenv(t2.VOL_ROOT_ENV, str(tmp_path / "empty"))
    cfg = _cfg("/vol/pool/k/pairs_constv2-heldout.jsonl")
    with pytest.raises(SystemExit) as ei:
        t2.heldout_pool_identity(cfg, SI)
    msg = str(ei.value)
    assert "/vol/pool/k/pairs_constv2-heldout.jsonl" in msg
    assert f"resolved through {t2.VOL_ROOT_ENV} to" in msg
    assert str(tmp_path / "empty" / "pool" / "k") in msg
    # unset: the message names the declared path only (no remap happened)
    monkeypatch.delenv(t2.VOL_ROOT_ENV, raising=False)
    cfg = _cfg(str(tmp_path / "nowhere.jsonl"))
    with pytest.raises(SystemExit) as ei:
        t2.heldout_pool_identity(cfg, SI)
    assert "resolved through" not in str(ei.value)
