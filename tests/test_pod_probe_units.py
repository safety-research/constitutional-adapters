"""T11 — pod acceptance probe pure-function units (twosi-model-sweeps §8).

The probe itself needs a GPU, nvidia-smi and the real trainer; these tests
cover the ported PURE logic only — ``decide()`` verdict priority,
``pairs_needed`` arithmetic and ``derive_probe_config`` overrides — plus the
small measurement helpers they lean on. CPU-only, no subprocesses.
"""

from __future__ import annotations

import math
import sys
from copy import deepcopy
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import pod_acceptance_probe as pap  # noqa: E402


# --------------------------------------------------------------------------- #
# decide(): verdict priority (crash > no_steps > throughput > gpu_util)        #
# --------------------------------------------------------------------------- #

HEALTHY = dict(steps_total=50, steps_per_sec=3.0, gpu_util=95.0,
               crashed=False, min_steps_per_sec=1.0, min_gpu_util=40.0)


def test_decide_accepts_healthy():
    assert pap.decide(**HEALTHY) == (pap.ACCEPT, None)


def test_decide_crash_beats_everything():
    # Even with perfectly healthy measurements, a trainer crash rejects.
    assert pap.decide(**{**HEALTHY, "crashed": True}) == (
        pap.REJECT, pap.REASON_CRASH)


def test_decide_no_steps_on_thin_measurement():
    got = pap.decide(**{**HEALTHY, "steps_total": 5})
    assert got == (pap.REJECT, pap.REASON_NO_STEPS)


def test_decide_no_steps_on_missing_rate():
    got = pap.decide(**{**HEALTHY, "steps_per_sec": None})
    assert got == (pap.REJECT, pap.REASON_NO_STEPS)


def test_decide_no_steps_beats_throughput_and_util():
    # A stepless pod must read no_steps, never throughput/gpu_util — the
    # latter two would suggest a measured-but-slow pod.
    got = pap.decide(**{**HEALTHY, "steps_total": 0, "steps_per_sec": None,
                        "gpu_util": 0.0})
    assert got == (pap.REJECT, pap.REASON_NO_STEPS)


def test_decide_throughput_reject():
    got = pap.decide(**{**HEALTHY, "steps_per_sec": 0.5})
    assert got == (pap.REJECT, pap.REASON_THROUGHPUT)


def test_decide_throughput_beats_gpu_util():
    got = pap.decide(**{**HEALTHY, "steps_per_sec": 0.5, "gpu_util": 1.0})
    assert got == (pap.REJECT, pap.REASON_THROUGHPUT)


def test_decide_gpu_util_reject():
    got = pap.decide(**{**HEALTHY, "gpu_util": 10.0})
    assert got == (pap.REJECT, pap.REASON_GPU_UTIL)


def test_decide_thresholds_inclusive():
    # Exactly AT the threshold passes: the thresholds sit between the healthy
    # and pathological bands, so the boundary is healthy territory.
    got = pap.decide(**{**HEALTHY, "steps_per_sec": 1.0, "gpu_util": 40.0})
    assert got == (pap.ACCEPT, None)


def test_decide_unmeasured_util_after_throughput_pass_is_probe_error():
    # gpu_util=None once throughput passed is a MEASUREMENT failure, not pod
    # evidence — it must raise, never guess a verdict.
    with pytest.raises(ValueError):
        pap.decide(**{**HEALTHY, "gpu_util": None})


def test_decide_min_measured_steps_clamp():
    # A deliberately tiny probe stays judgeable when the caller clamps
    # min_measured_steps to --steps.
    got = pap.decide(**{**HEALTHY, "steps_total": 5},
                     min_measured_steps=5)
    assert got == (pap.ACCEPT, None)


# --------------------------------------------------------------------------- #
# pairs_needed(): one epoch ≈ steps optimizer steps                            #
# --------------------------------------------------------------------------- #


def test_pairs_needed_reference_arithmetic():
    # steps×bs fit rows, grossed up for the holdout carve and the hash
    # split's variance margin: 50×8 / 0.9 × 1.15 = 511.1… → 512.
    assert pap.pairs_needed(50, 8, 0.1) == 512
    assert pap.pairs_needed(50, 8, 0.1) == math.ceil(50 * 8 / 0.9 * 1.15)


def test_pairs_needed_zero_holdout():
    assert pap.pairs_needed(50, 8, 0.0) == math.ceil(400 * 1.15)


def test_pairs_needed_margin_override():
    assert pap.pairs_needed(10, 2, 0.0, margin=1.0) == 20


def test_pairs_needed_floor_of_one():
    assert pap.pairs_needed(1, 1, 0.0, margin=0.5) == 1


@pytest.mark.parametrize("steps,bs,holdout", [
    (0, 8, 0.1),      # no steps
    (50, 0, 0.1),     # no batch
    (-1, 8, 0.1),
    (50, 8, 1.0),     # holdout eats the whole pool
    (50, 8, -0.1),
])
def test_pairs_needed_invalid_inputs_raise(steps, bs, holdout):
    with pytest.raises(ValueError):
        pap.pairs_needed(steps, bs, holdout)


# --------------------------------------------------------------------------- #
# derive_probe_config(): exactly four overrides, everything else preserved     #
# --------------------------------------------------------------------------- #


def _wave_cfg() -> dict:
    return {
        "model": "org/model",
        "model_revision": "abc123",
        "dtype": "bfloat16",
        "max_len": 6400,
        "max_len_by_si": {"si_constitution": 6400},
        "batch_size_by_arm": {"lora_r1_all": 8, "psr_kl_all": 4},
        "data_parallel": {"gpus": 8, "arm_gpus": {"diffmean": 8}},
        "train_kwargs": {
            "default": {"epochs": 3, "holdout_frac": 0.1,
                        "floor_ratio_max": 3.0, "batch_size": 8},
            "bidpo": {"floor_ratio_max": 2.0, "lr": 1e-4},
            "psr_kl_all": {"lr": 3e-3},
        },
        "wandb_fits": True,
    }


def test_derive_probe_config_overrides():
    out = pap.derive_probe_config(_wave_cfg())
    assert out["data_parallel"] == {"gpus": 1}
    assert out["train_kwargs"]["default"]["epochs"] == 1
    assert out["train_kwargs"]["default"]["floor_ratio_max"] is None
    assert out["wandb_fits"] is False


def test_derive_probe_config_nulls_per_arm_floor_bars_too():
    # arm_floor_ratio_max lets a per-arm bar BEAT the default, so nulling
    # only the default would leave e.g. the DPO arms' own bar judging fit
    # quality on the probe's truncated fit.
    out = pap.derive_probe_config(_wave_cfg())
    assert out["train_kwargs"]["bidpo"]["floor_ratio_max"] is None
    # ... while the rest of the arm's kwargs are preserved.
    assert out["train_kwargs"]["bidpo"]["lr"] == 1e-4
    # An arm block WITHOUT a floor bar is untouched.
    assert out["train_kwargs"]["psr_kl_all"] == {"lr": 3e-3}


def test_derive_probe_config_preserves_everything_else():
    out = pap.derive_probe_config(_wave_cfg())
    for key in ("model", "model_revision", "dtype", "max_len",
                "max_len_by_si", "batch_size_by_arm"):
        assert out[key] == _wave_cfg()[key], key
    assert out["train_kwargs"]["default"]["holdout_frac"] == 0.1
    assert out["train_kwargs"]["default"]["batch_size"] == 8


def test_derive_probe_config_never_mutates_input():
    cfg = _wave_cfg()
    before = deepcopy(cfg)
    pap.derive_probe_config(cfg)
    assert cfg == before


def test_derive_probe_config_tolerates_missing_train_kwargs():
    out = pap.derive_probe_config({"model": "m"})
    assert out["data_parallel"] == {"gpus": 1}
    assert out["train_kwargs"]["default"]["epochs"] == 1
    assert out["train_kwargs"]["default"]["floor_ratio_max"] is None
    assert out["wandb_fits"] is False


# --------------------------------------------------------------------------- #
# Measurement helpers (also pure)                                              #
# --------------------------------------------------------------------------- #


def test_is_step_line_accepts_per_step_losses():
    assert pap.is_step_line({"step": 3, "lockstep/lora_r1/kl_loss": 0.5})
    assert pap.is_step_line({"step": 0, "distill/mse_loss": 1.0})


def test_is_step_line_rejects_summaries_and_non_steps():
    # heldout_loss lines carry a non-null step too — the suffix filter is
    # load-bearing.
    assert not pap.is_step_line({"step": 3, "heldout_loss": 0.5})
    assert not pap.is_step_line({"step": 3, "arm_step0_loss": 0.5})
    assert not pap.is_step_line({"step": 3, "final_probe_loss": 0.5})
    assert not pap.is_step_line({"step": None, "distill/mse_loss": 1.0})
    assert not pap.is_step_line({"step": 3, "some_loss": True})  # bool ≠ number
    assert not pap.is_step_line("not a dict")


def test_steps_per_sec_tail_excludes_encode_phase():
    # Count sits at 0 during encode; the window starts at the LAST all-zero
    # sample, so the encode seconds never dilute the rate.
    samples = [(0.0, 0), (10.0, 0), (12.0, 10), (14.0, 20), (16.0, 30)]
    tail = pap.steps_per_sec_tail(samples, tail_steps=30)
    assert tail is not None
    assert tail.steps == 30
    assert tail.seconds == pytest.approx(6.0)
    assert tail.rate == pytest.approx(5.0)


def test_steps_per_sec_tail_none_when_no_steps():
    assert pap.steps_per_sec_tail([], 30) is None
    assert pap.steps_per_sec_tail([(0.0, 0), (5.0, 0)], 30) is None


def test_mean_gpu_util_stepping_window_only():
    # Pre-step (count 0) and post-final samples are excluded; None reads
    # are skipped inside the window.
    samples = [(0.0, 0, 5.0), (2.0, 10, 90.0), (4.0, 20, None),
               (6.0, 30, 80.0), (8.0, 30, 0.0)]
    assert pap.mean_gpu_util(samples) == pytest.approx(85.0)


def test_mean_gpu_util_none_without_steps():
    assert pap.mean_gpu_util([(0.0, 0, 50.0)]) is None


def test_format_final_line_nan_never_zero():
    line = pap.format_final_line(pap.REJECT, pap.REASON_NO_STEPS, None, None)
    assert line == "REJECT reason=no_steps steps_per_sec=nan gpu_util=nan"
    assert math.isnan(float(line.split("steps_per_sec=")[1].split()[0]))
    ok = pap.format_final_line(pap.ACCEPT, None, 2.345, 91.2)
    assert ok == "ACCEPT steps_per_sec=2.345 gpu_util=91.2"


def test_truncate_pool_head_truncation(tmp_path):
    src = tmp_path / "pool.jsonl"
    src.write_text('{"a": 1}\n\n{"a": 2}\n{"a": 3}')  # blank line + no EOL
    dst = tmp_path / "trunc.jsonl"
    assert pap.truncate_pool(src, dst, 2) == 2
    assert dst.read_text() == '{"a": 1}\n{"a": 2}\n'
    # Fewer rows than requested: writes what exists, returns the true count.
    assert pap.truncate_pool(src, dst, 10) == 3


def test_truncate_pool_output_is_owner_only(tmp_path):
    # Review fix F13: the truncated copy is raw attack data on a mount other
    # users can read, and a REWRITE keeps a pre-existing file's mode — so
    # 0600 is enforced explicitly, never inherited from the umask.
    src = tmp_path / "pool.jsonl"
    src.write_text('{"a": 1}\n')
    dst = tmp_path / "trunc.jsonl"
    dst.write_text("stale")
    dst.chmod(0o644)
    pap.truncate_pool(src, dst, 1)
    assert dst.stat().st_mode & 0o777 == 0o600


# --------------------------------------------------------------------------- #
# Pilot sustained-duty helpers (review fix F11; consumed by pod_pilot.sh)      #
# --------------------------------------------------------------------------- #


def test_parse_dmon_sm_max_takes_per_sample_max_across_gpus():
    text = ("# gpu    sm   mem   enc   dec\n"
            "# Idx     %     %     %     %\n"
            "    0    97    40     0     0\n"
            "    1     5     1     0     0\n"
            "    0    99    41     0     0\n"
            "    1     3     0     0     0\n")
    assert pap.parse_dmon_sm_max(text) == [97.0, 99.0]


def test_parse_dmon_sm_max_skips_unsupported_and_garbage_rows():
    text = "0 - - - -\n0 88 1 0 0\nnot a dmon row\n"
    assert pap.parse_dmon_sm_max(text) == [88.0]


def test_parse_dmon_sm_max_truncated_output_keeps_complete_samples():
    # gpu 1's second row never arrived (dmon cut mid-sample): only the
    # complete cross-GPU samples survive.
    text = "0 90 0 0 0\n1 10 0 0 0\n0 95 0 0 0\n"
    assert pap.parse_dmon_sm_max(text) == [90.0]


def test_parse_dmon_sm_max_empty():
    assert pap.parse_dmon_sm_max("") == []


DUTY_OK = dict(sample_max=[97.0] * 25, min_duty=90.0, want_samples=25,
               train_alive_through_window=True, steps_before=3,
               steps_after=40)


def test_duty_verdict_go_on_a_healthy_window():
    rec = pap.duty_verdict(**DUTY_OK)
    assert (rec["verdict"], rec["no_go"]) == (pap.DUTY_GO, None)
    assert rec["mean_sm_pct"] == 97.0
    assert rec["n_samples"] == 25
    assert rec["steps_before"] == 3 and rec["steps_after"] == 40


def test_duty_verdict_train_exit_beats_perfect_samples():
    # A 97%-SM window is worthless if the trainer stopped running inside
    # it: the remaining samples were idle GPU.
    rec = pap.duty_verdict(**{**DUTY_OK,
                              "train_alive_through_window": False})
    assert (rec["verdict"], rec["no_go"]) == (pap.DUTY_NO_GO,
                                              pap.DUTY_TRAIN_EXITED)


def test_duty_verdict_requires_optimizer_advance():
    # THE F11 false-GO: the trainer runs step0_loss + an initial held-out
    # eval at high SM% BEFORE the optimizer loop — a hot window with zero
    # optimizer steps completed measured the wrong phase and must not
    # authorize the 16-GPU fan-out.
    rec = pap.duty_verdict(**{**DUTY_OK, "steps_before": 7,
                              "steps_after": 7})
    assert (rec["verdict"], rec["no_go"]) == (pap.DUTY_NO_GO,
                                              pap.DUTY_NO_STEP_ADVANCE)


def test_duty_verdict_thin_window():
    rec = pap.duty_verdict(**{**DUTY_OK, "sample_max": [99.0] * 10})
    assert (rec["verdict"], rec["no_go"]) == (pap.DUTY_NO_GO,
                                              pap.DUTY_THIN_WINDOW)


def test_duty_verdict_want_samples_slack_of_five():
    # want=25 tolerates down to max(20, 25-5)=20 usable samples.
    assert pap.duty_verdict(
        **{**DUTY_OK, "sample_max": [95.0] * 20})["verdict"] == pap.DUTY_GO
    assert pap.duty_verdict(
        **{**DUTY_OK, "sample_max": [95.0] * 19})["no_go"] == \
        pap.DUTY_THIN_WINDOW


def test_duty_verdict_low_duty_and_inclusive_threshold():
    low = pap.duty_verdict(**{**DUTY_OK, "sample_max": [38.0] * 25})
    assert (low["verdict"], low["no_go"]) == (pap.DUTY_NO_GO, pap.DUTY_LOW)
    # Exactly AT the threshold passes (same convention as decide()).
    at = pap.duty_verdict(**{**DUTY_OK, "sample_max": [90.0] * 25})
    assert at["verdict"] == pap.DUTY_GO


def test_duty_verdict_reason_priority():
    # train_exited > thin_window > no_step_advance > low_duty.
    rec = pap.duty_verdict(**{**DUTY_OK, "sample_max": [],
                              "train_alive_through_window": False,
                              "steps_after": 3})
    assert rec["no_go"] == pap.DUTY_TRAIN_EXITED
    assert rec["mean_sm_pct"] is None
    rec = pap.duty_verdict(**{**DUTY_OK, "sample_max": [99.0] * 3,
                              "steps_after": 3})
    assert rec["no_go"] == pap.DUTY_THIN_WINDOW
    rec = pap.duty_verdict(**{**DUTY_OK, "sample_max": [30.0] * 25,
                              "steps_after": 3})
    assert rec["no_go"] == pap.DUTY_NO_STEP_ADVANCE
