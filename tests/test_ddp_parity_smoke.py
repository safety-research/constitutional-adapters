"""The parity smoke's VERDICT (scripts/sdfbase/ddp_parity_smoke.py compare).

A 1 % loss band over 20 steps cannot see a gradient-scaling or reduction error inside a base-dominated CE (and Adam's scale
invariance hides a constant gradient factor from the parameters too), so the verdict gates on ALL of: the loss trajectory, the
final adapter parameters (bf16-relative), EQUAL per-step token histories (the row union) and LR histories, the step-0 grad
norm (the direct reading of the reduction's scale before any optimizer step diverges the replicas) and the median per-step
grad-norm difference, the step count and the speed-up. Pinned here on synthetic result pairs (equal; each gate broken alone)
and on the RECORDED job 804970 (tests/fixtures/ddp_parity_smoke/804970 — the real two legs, identifiers scrubbed), which
must PASS the tightened verdict.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
for _sub in ("scripts/sdfbase", "scripts", "src"):
    _p = str(REPO / _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ddp_parity_smoke as dps  # noqa: E402

FIX = REPO / "tests" / "fixtures" / "ddp_parity_smoke" / "804970"


def _result(world: int, *, steps: int = 6, seed: int = 0, loss_bump: float = 0.0, tokens_bump: int = 0, lr_bump: float = 0.0,
            gn_scale: float = 1.0, gn_scale_from: int = 0, s_per_step: float = 4.0) -> dict:
    g = torch.Generator().manual_seed(seed)
    losses = [2.0 + 0.01 * i for i in range(steps)]
    per = []
    for i in range(steps):
        per.append({"step": i + 1, "ce_loss": losses[i] + loss_bump, "ce": losses[i] + loss_bump,
                    "lr": 1e-5 * (i + 1) / steps + (lr_bump if i == steps - 1 else 0.0),
                    "loss_tokens": 20000 + i + (tokens_bump if i == 2 else 0), "real_tokens": 20400 + i,
                    "padded_tokens": 20400 + i, "sec_per_step": s_per_step,
                    "grad_norm": 0.03 * (1 + 0.01 * i) * (gn_scale if i >= gn_scale_from else 1.0)})
    return {"schema": 1, "kind": "ddp_parity_smoke", "arm": "lora_r4_ce", "world_size": world, "fit_width": world,
            "micro_batch": 2, "accum_steps": 4, "per_rank_accum": 4 // world, "steps_requested": steps, "steps_done": steps,
            "rows": 8 * steps, "seq_len": 8192, "lr": 1e-5, "seed": 0, "init_scheme": "zero",
            "history": [l + loss_bump for l in losses], "per_step": per, "model_load_s": 100.0, "train_s": s_per_step * steps,
            "s_per_step_mean": s_per_step, "s_per_step_steady": s_per_step, "tokens_real_total": sum(p["real_tokens"] for p in per),
            "tokens_per_s_wall": 1000.0 / s_per_step, "peak_alloc_bytes": 10 ** 11, "params_sha256": f"sha-{world}-{seed}",
            "trainable_params": 64}


def _write(dirpath: Path, res: dict, params: list) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "result.json").write_text(json.dumps(res))
    torch.save({"params": params}, dirpath / "params.pt")
    return dirpath


def _params(seed: int = 0, delta: float = 0.0):
    g = torch.Generator().manual_seed(seed)
    base = [torch.randn(8, 4, generator=g) * 0.05, torch.randn(8, 4, generator=g) * 0.05]
    if delta:
        base[1] = base[1].clone()
        base[1][0, 0] += delta
    return base


def _compare(tmp_path, ref, wide, pref=None, pwide=None, **over):
    r = _write(tmp_path / "ref", ref, pref if pref is not None else _params())
    w = _write(tmp_path / "wide", wide, pwide if pwide is not None else _params())
    argv = ["compare", str(r), str(w), "--out", str(tmp_path / "out"), "--min-speedup", "1.6"]
    for k, v in over.items():
        argv += [f"--{k.replace('_', '-')}", str(v)]
    rc = dps.main(argv)
    doc = json.loads((tmp_path / "out" / "parity.json").read_text())
    return rc, doc


def test_identical_runs_pass_every_gate(tmp_path):
    rc, doc = _compare(tmp_path, _result(1), _result(2, s_per_step=2.0))
    assert rc == 0 and doc["verdict"] == "PASS", doc
    assert all(doc["gates"].values()), doc["gates"]
    assert doc["params_rel_diff"] == 0.0 and doc["token_histories_equal"] and doc["lr_history_equal"]
    assert doc["gradnorm_step0_rel_diff"] == 0.0 and doc["gradnorm_median_rel_diff"] == 0.0
    md = (tmp_path / "out" / "parity.md").read_text()
    assert "**PASS**" in md and "grad norm" in md and "token histories" in md


def test_different_adapter_weights_fail_even_when_the_loss_agrees(tmp_path):
    """The finding: a 1 % loss band certified different weights. max |p| = ~0.15 here, so a 2e-3 bump is ~1.3 % relative."""
    rc, doc = _compare(tmp_path, _result(1), _result(2, s_per_step=2.0), pwide=_params(delta=2e-3))
    assert rc == 1 and doc["verdict"] == "FAIL" and doc["gates"]["params"] is False
    assert doc["gates"]["loss"] is True                         # the loss gate alone would have passed
    assert doc["params_rel_diff"] > doc["params_rtol"]


def test_a_reduction_scaled_twice_fails_on_the_step0_grad_norm(tmp_path):
    """A summed-twice (or averaged) all-reduce reads 2x (0.5x) on the first step's accumulated grad norm — Adam hides it
    from the loss and the params, the gate does not."""
    rc, doc = _compare(tmp_path, _result(1), _result(2, s_per_step=2.0, gn_scale=2.0))
    assert rc == 1 and doc["gates"]["gradnorm_step0"] is False and doc["gates"]["gradnorm_median"] is False
    assert doc["gradnorm_step0_rel_diff"] == pytest.approx(1.0)


def test_one_early_grad_norm_outlier_passes_the_median_gate(tmp_path):
    """The zero-init LoRA / Adam sign-flip amplifier: one large early outlier (step 1) is tolerated by the MEDIAN gate while
    step 0 and the rest agree — the recorded 804970 shape."""
    wide = _result(2, s_per_step=2.0)
    wide["per_step"][1]["grad_norm"] *= 1.12
    rc, doc = _compare(tmp_path, _result(1), wide)
    assert rc == 0 and doc["gates"]["gradnorm_step0"] and doc["gates"]["gradnorm_median"]


def test_token_and_lr_histories_must_be_equal(tmp_path):
    rc, doc = _compare(tmp_path, _result(1), _result(2, s_per_step=2.0, tokens_bump=1))
    assert rc == 1 and doc["gates"]["tokens"] is False and doc["histories"]["loss_tokens"]["first_mismatch_step"] == 2
    rc, doc = _compare(tmp_path, _result(1), _result(2, s_per_step=2.0, lr_bump=1e-9))
    assert rc == 1 and doc["gates"]["lr"] is False


def test_loss_drift_and_a_slow_wide_leg_still_fail(tmp_path):
    rc, doc = _compare(tmp_path, _result(1), _result(2, s_per_step=2.0, loss_bump=0.05))
    assert rc == 1 and doc["gates"]["loss"] is False
    rc, doc = _compare(tmp_path, _result(1), _result(2, s_per_step=3.5))
    assert rc == 1 and doc["gates"]["speed"] is False and doc["gates"]["loss"] is True


def test_a_different_run_shape_is_a_problem_not_a_pass(tmp_path):
    wide = _result(2, s_per_step=2.0)
    wide["accum_steps"] = 8
    rc, doc = _compare(tmp_path, _result(1), wide)
    assert rc == 1 and doc["gates"]["same_run_shape"] is False and any("accum_steps" in p for p in doc["problems"])


def test_the_recorded_job_804970_passes_the_tightened_verdict(tmp_path):
    """The REAL two legs (cluster A 804970, 2026-09-30): loss within 8.5e-4 relative, adapter params within 1.6e-3 of the
    largest weight (bf16 reduction-order noise), token and lr histories equal, step-0 grad norm within 1e-3, median grad-norm
    difference ~1 %, step time 4.53 -> 1.96 s. A tightened verdict that failed this would be certifying nothing real."""
    rc = dps.main(["compare", str(FIX / "w1"), str(FIX / "w2"), "--out", str(tmp_path)])
    doc = json.loads((tmp_path / "parity.json").read_text())
    assert rc == 0 and doc["verdict"] == "PASS", doc["gates"]
    assert doc["widths"] == {"reference": 1, "wide": 2} and doc["steps_compared"] == 20
    assert doc["loss_max_rel_diff"] < 1e-3 and doc["params_rel_diff"] < 2e-3
    assert doc["token_histories_equal"] and doc["lr_history_equal"]
    assert doc["gradnorm_step0_rel_diff"] < 1e-3 and doc["gradnorm_median_rel_diff"] < 2e-2
    assert doc["step_speedup"] > 2.0
    # the recorded shape the median gate exists for: step 1's grad norm differs by ~12 %, every other step by < 3 %
    rels = [abs(x - y) / x for x, y in zip(doc["gradnorm_reference"], doc["gradnorm_wide"])]
    assert rels[1] > 0.1 and sorted(rels)[-2] < 0.03
