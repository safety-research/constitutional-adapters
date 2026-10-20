"""Partial Llama winner folds land safely (companion to test_fit_configs).

Each Llama fit arm's HPO lands on its own clock, so the fold machinery
accepts winners for a SUBSET of arms — folded arms get real values, the
rest keep the fail-loud sentinel — and the launch gate
(scripts/check_fit_config_folded.py, called per-leg by
twosi_fits_llama.sbatch with that leg's $ARMS) refuses exactly the legs
whose own arms are still unfolded. (The example arm here was
token_faithful until the re-tune dropped it, 2026-08-27; the mechanics are
arm-generic, so psr_kl_all carries the same tests now.)

2026-08-27: every current Llama fit arm's winner LANDED in
LLAMA_LANDED_WINNERS (and the lora arm is lora_r4 since the 2026-08-28
rename), so the shipped generator no longer emits any sentinel. The
machinery stays load-bearing for future arms/waves; the pending-world
tests below monkeypatch LLAMA_LANDED_WINNERS to {} to re-create a
partial world (the same convention as test_fit_configs'
test_llama_waits_on_its_hpo_loudly). The subprocess CLI test asserts the
LANDED world instead — the CLI cannot be monkeypatched.
"""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

spec = importlib.util.spec_from_file_location(
    "make_fit_configs_pf", REPO / "scripts/make_fit_configs.py")
mfc = importlib.util.module_from_spec(spec)
sys.modules["make_fit_configs_pf"] = mfc
spec.loader.exec_module(mfc)

spec_ck = importlib.util.spec_from_file_location(
    "check_fit_config_folded", REPO / "scripts/check_fit_config_folded.py")
ckf = importlib.util.module_from_spec(spec_ck)
sys.modules["check_fit_config_folded"] = ckf
spec_ck.loader.exec_module(ckf)

KL_WINNER = {"lr": 5e-4, "warmup_frac": 0.15, "max_grad_norm": 0.1}
#: A re-tune@32 winner shape: the expanded axes + the realized accumulation
#: (tune_arm.py analyze's winner dict) — every key must fold.
KL_WINNER_RETUNE = {**KL_WINNER, "temperature": 2.0, "weight_decay": 1e-3,
                    "accum_steps": 16, "batch_size": 2}
LLAMA_FILES = ("two_si_llama_fits.yaml", "two_si_llama_fits_dp3.yaml")


def test_partial_fold_keeps_other_sentinels(monkeypatch):
    monkeypatch.setattr(mfc, "LLAMA_LANDED_WINNERS", {})  # pending world
    out = mfc.build_all({"psr_kl_all": KL_WINNER})
    for name in LLAMA_FILES:
        tk = out[name][1]["train_kwargs"]
        assert tk["psr_kl_all"]["lr"] == KL_WINNER["lr"]
        assert (tk["psr_kl_all"]["max_grad_norm"]
                == KL_WINNER["max_grad_norm"])
        for arm in ("psr_mse_all", "lora_r4"):
            assert tk[arm]["lr"] == mfc.LLAMA_PENDING, (name, arm)
    # the other models' folds are untouched by a Llama partial fold
    q = out["two_si_qwen38_27b_fits.yaml"][1]["train_kwargs"]
    assert q["psr_kl_all"]["lr"] \
        == mfc.QWEN_INTERIM_WINNERS["psr_kl_all"]["lr"]


def test_retune_winner_keys_fold_through(monkeypatch):
    """The re-tune@32 harvest's expanded winner shape lands verbatim in the
    arm's train_kwargs block: temperature/weight_decay/accum_steps (and
    lora_rank for the lora arm) next to the classic keys; batch_size stays
    OUT of train_kwargs (it is the driver's batch_size_by_arm world) but is
    VALIDATED against the measured table."""
    monkeypatch.setattr(mfc, "LLAMA_LANDED_WINNERS", {})  # pending world
    lora_w = {"lr": 3e-4, "warmup_frac": 0.05, "max_grad_norm": 1.2,
              "temperature": 1.0, "weight_decay": 0.0, "lora_rank": 4,
              "accum_steps": 16, "batch_size": 2}
    out = mfc.build_all({"psr_kl_all": KL_WINNER_RETUNE,
                         "lora_r4": lora_w})
    for name in LLAMA_FILES:
        tk = out[name][1]["train_kwargs"]
        kl = tk["psr_kl_all"]
        assert kl["temperature"] == 2.0
        assert kl["weight_decay"] == 1e-3
        assert kl["accum_steps"] == 16
        assert "batch_size" not in kl and "effective_batch" not in kl
        lo = tk["lora_r4"]
        assert lo["lora_rank"] == 4
        assert lo["weight_decay"] == 0.0  # wd=0 folds as a real value
        assert tk["psr_mse_all"]["lr"] == mfc.LLAMA_PENDING
        # every fit cell is decode-only now (the all->response flip)
        assert tk["default"]["positions"] == "response"


def test_retune_winner_batch_size_mismatch_refuses():
    """accum_steps only reproduces the tuned effective batch at the batch
    size it was tuned with — a winner tuned at bs4 against the measured
    bs2 table must refuse, never silently refit at effective 64."""
    bad = {**KL_WINNER_RETUNE, "batch_size": 4}
    with pytest.raises(SystemExit, match="DIFFERENT effective batch"):
        mfc.build_all({"psr_kl_all": bad})


def test_landed_constant_reaches_build_none(monkeypatch):
    """Winners pinned into LLAMA_LANDED_WINNERS must flow through
    build_all(None) — the byte-pin test's path — so a landed fold stays
    generator-reproducible without a winners.json on disk."""
    monkeypatch.setattr(mfc, "LLAMA_LANDED_WINNERS",
                        {"psr_kl_all": KL_WINNER})
    out = mfc.build_all(None)
    for name in LLAMA_FILES:
        tk = out[name][1]["train_kwargs"]
        assert tk["psr_kl_all"]["lr"] == KL_WINNER["lr"]
        assert tk["psr_mse_all"]["lr"] == mfc.LLAMA_PENDING


def test_cli_winners_override_landed(monkeypatch):
    monkeypatch.setattr(mfc, "LLAMA_LANDED_WINNERS",
                        {"psr_kl_all": KL_WINNER})
    cli = {"psr_kl_all": {"lr": 9e-3, "warmup_frac": 0.05,
                          "max_grad_norm": 0.3}}
    out = mfc.build_all(cli)
    for name in LLAMA_FILES:
        assert (out[name][1]["train_kwargs"]["psr_kl_all"]["lr"]
                == pytest.approx(9e-3))


def _write_partial_cfg(tmp_path: Path) -> Path:
    # callers monkeypatch LLAMA_LANDED_WINNERS to {} first, so ONLY the
    # CLI-folded arm gets real values here
    header, cfg = mfc.build_all({"psr_kl_all": KL_WINNER})[
        "two_si_llama_fits.yaml"]
    p = tmp_path / "partial.yaml"
    p.write_text(header + yaml.safe_dump(cfg, sort_keys=False))
    return p


def test_gate_passes_folded_leg_refuses_pending_leg(tmp_path, capsys,
                                                    monkeypatch):
    monkeypatch.setattr(mfc, "LLAMA_LANDED_WINNERS", {})  # pending world
    p = _write_partial_cfg(tmp_path)
    # the folded arm's leg is launchable
    assert ckf.main(["ckf", str(p), "psr_kl_all"]) == 0
    # a leg with pending arms is refused, naming every unfolded arm
    rc = ckf.main(["ckf", str(p), "psr_mse_all,lora_r4"])
    assert rc == 1
    err = capsys.readouterr().err
    for arm in ("psr_mse_all", "lora_r4"):
        assert arm in err
    # empty arm list is a caller bug, never a pass
    assert ckf.main(["ckf", str(p), ""]) == 2


def test_gate_refuses_fully_sentinel_config(tmp_path, monkeypatch):
    monkeypatch.setattr(mfc, "LLAMA_LANDED_WINNERS", {})  # pending world
    header, cfg = mfc.build_all(None)["two_si_llama_fits.yaml"]
    p = tmp_path / "sentinel.yaml"
    p.write_text(header + yaml.safe_dump(cfg, sort_keys=False))
    assert ckf.main(["ckf", str(p), "psr_kl_all"]) == 1


def test_cli_fold_end_to_end(tmp_path):
    """make_fit_configs.py --llama-winners end to end, in the LANDED world
    (a subprocess cannot monkeypatch LLAMA_LANDED_WINNERS): a winners.json
    naming ONE arm emits with no refusal and no pending NOTE — the landed
    constants cover the rest, so nothing is partial any more — the CLI
    value wins over the landed one for its arm, and no sentinel survives
    anywhere. Unknown arms still refuse — token_faithful is now unknown
    (dropped), so a stale pre-drop winners file refuses loudly instead of
    silently folding a dead arm."""
    wj = tmp_path / "winners.json"
    wj.write_text(json.dumps({"psr_kl_all": KL_WINNER}))
    out_dir = tmp_path / "configs"
    out_dir.mkdir()
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts/make_fit_configs.py"),
         "--llama-winners", str(wj), "--out-dir", str(out_dir)],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "NOTE: partial fold" not in r.stdout  # nothing pends any more
    cfg = yaml.safe_load((out_dir / "two_si_llama_fits.yaml").read_text())
    tk = cfg["train_kwargs"]
    # CLI over landed for the named arm; landed values for the rest
    assert tk["psr_kl_all"]["lr"] == pytest.approx(KL_WINNER["lr"])
    assert tk["psr_mse_all"]["lr"] == pytest.approx(
        mfc.LLAMA_LANDED_WINNERS["psr_mse_all"]["lr"])
    assert tk["lora_r4"]["lr"] == pytest.approx(
        mfc.LLAMA_LANDED_WINNERS["lora_r4"]["lr"])
    # no sentinel VALUE survives (the emitted header comment narrates the
    # sentinel mechanism by name, so this is a value check, not a grep —
    # the same distinction the per-leg sbatch gate exists for)
    assert not any(v == mfc.LLAMA_PENDING
                   for block in tk.values() if isinstance(block, dict)
                   for v in block.values())

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"token_faithful": KL_WINNER}))
    r2 = subprocess.run(
        [sys.executable, str(REPO / "scripts/make_fit_configs.py"),
         "--llama-winners", str(bad), "--out-dir", str(out_dir)],
        capture_output=True, text=True)
    assert r2.returncode != 0
    assert "unknown arms" in r2.stderr


def test_sbatch_gate_is_per_leg():
    """The sbatch must consult the per-leg checker with $ARMS — a
    whole-file sentinel grep would re-block one leg's wave on another
    leg's pending sentinels."""
    sb = (REPO / "scripts/sbatch/twosi_fits_llama.sbatch").read_text()
    assert 'check_fit_config_folded.py" "$CFG" "$ARMS"' in sb
    assert 'grep -q PENDING_LLAMA_HPO_WINNERS "$CFG"' not in sb
