"""Pod acceptance probe (scripts/pod_acceptance_probe.py) — the gate that
keeps a GPU wave off a slow pod (the 2026-08-22 wave lost most of a 37-GPU
budget to hosts running identical H100s at ~1/6 speed).

Everything except the real trainer subprocess + nvidia-smi is exercised
here (the grading host has no GPU): the decision logic exhaustively (accept,
every reject reason, boundary values), the metrics-jsonl step counting and
tail-rate computation against synthetic files/samples, the stepping-window
GPU-util mean, config derivation, pool truncation and sizing, the monitor
loop against a fake process/clock, and the whole ``run_probe`` flow
end-to-end with a fake trainer subprocess standing in for the real one.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import pod_acceptance_probe as pap  # noqa: E402


# --------------------------------------------------------------------------- #
# decide(): thresholds -> verdict, exhaustively                                #
# --------------------------------------------------------------------------- #

THRESH = dict(min_steps_per_sec=1.0, min_gpu_util=40.0)


def test_decide_accept():
    assert pap.decide(steps_total=50, steps_per_sec=3.2, gpu_util=92.0,
                      crashed=False, **THRESH) == (pap.ACCEPT, None)


def test_decide_reject_throughput():
    """The wave's actual failure mode: 0.3-0.8 steps/s on a slow host."""
    assert pap.decide(steps_total=50, steps_per_sec=0.5, gpu_util=95.0,
                      crashed=False, **THRESH) == (pap.REJECT, "throughput")


def test_decide_reject_gpu_util():
    assert pap.decide(steps_total=50, steps_per_sec=2.0, gpu_util=5.0,
                      crashed=False, **THRESH) == (pap.REJECT, "gpu_util")


def test_decide_reject_no_steps_zero():
    assert pap.decide(steps_total=0, steps_per_sec=None, gpu_util=None,
                      crashed=False, **THRESH) == (pap.REJECT, "no_steps")


def test_decide_reject_no_steps_below_min_measured():
    """A handful of steps is not evidence; below MIN_MEASURED_STEPS the
    tail-rate estimate is too thin to certify a pod either way."""
    assert pap.decide(steps_total=pap.MIN_MEASURED_STEPS - 1,
                      steps_per_sec=5.0, gpu_util=95.0,
                      crashed=False, **THRESH) == (pap.REJECT, "no_steps")


def test_decide_no_steps_when_rate_uncomputable():
    """Steps observed but no computable window (e.g. all before the first
    sample) must never silently accept."""
    assert pap.decide(steps_total=50, steps_per_sec=None, gpu_util=95.0,
                      crashed=False, **THRESH) == (pap.REJECT, "no_steps")


def test_decide_reject_crash():
    assert pap.decide(steps_total=0, steps_per_sec=None, gpu_util=None,
                      crashed=True, **THRESH) == (pap.REJECT, "crash")


def test_decide_crash_wins_over_good_numbers():
    """A trainer that crashes on a pod rejects it even if the steps it did
    take were fast — a wave cannot run on a host where the trainer dies."""
    assert pap.decide(steps_total=50, steps_per_sec=5.0, gpu_util=95.0,
                      crashed=True, **THRESH) == (pap.REJECT, "crash")


def test_decide_boundary_values_pass_inclusive():
    """Exactly AT a threshold is a pass — thresholds sit between the
    healthy and pathological bands, so the boundary is healthy."""
    assert pap.decide(steps_total=50, steps_per_sec=1.0, gpu_util=40.0,
                      crashed=False, **THRESH) == (pap.ACCEPT, None)


def test_decide_boundary_just_below_fails():
    assert pap.decide(steps_total=50, steps_per_sec=0.999, gpu_util=95.0,
                      crashed=False, **THRESH) == (pap.REJECT, "throughput")
    assert pap.decide(steps_total=50, steps_per_sec=1.0, gpu_util=39.9,
                      crashed=False, **THRESH) == (pap.REJECT, "gpu_util")


def test_decide_min_measured_steps_boundary():
    ok = pap.decide(steps_total=10, steps_per_sec=2.0, gpu_util=90.0,
                    crashed=False, min_measured_steps=10, **THRESH)
    assert ok == (pap.ACCEPT, None)
    bad = pap.decide(steps_total=9, steps_per_sec=2.0, gpu_util=90.0,
                     crashed=False, min_measured_steps=10, **THRESH)
    assert bad == (pap.REJECT, "no_steps")


def test_decide_throughput_reject_stands_without_gpu_util():
    """A below-threshold rate is real pod evidence even when the util read
    broke — the reject stands on what WAS measured."""
    assert pap.decide(steps_total=50, steps_per_sec=0.4, gpu_util=None,
                      crashed=False, **THRESH) == (pap.REJECT, "throughput")


def test_decide_raises_on_unmeasured_gpu_util_after_throughput_passes():
    """Fast steps + no util measurement: accepting would be a guess and
    rejecting would blame the pod for the probe's failure — raise."""
    with pytest.raises(ValueError):
        pap.decide(steps_total=50, steps_per_sec=3.0, gpu_util=None,
                   crashed=False, **THRESH)


# --------------------------------------------------------------------------- #
# Step-line predicate + counting against a synthetic metrics jsonl             #
# --------------------------------------------------------------------------- #

STEP_LINE = {"step": 7, "lockstep/lora_r1/soft_loss": 1.23, "epoch": 0}


def test_is_step_line_accepts_lockstep_and_distill_shapes():
    assert pap.is_step_line(STEP_LINE)
    assert pap.is_step_line({"step": 3, "distill/soft_loss": 0.5,
                             "distill/vec_norm": 1.0, "epoch": 0})


def test_is_step_line_rejects_summaries_and_garbage():
    # heldout lines carry a non-null step too — the suffix filter is what
    # keeps them out.
    assert not pap.is_step_line(
        {"step": 12, "lockstep/lora_r1/heldout_loss": 0.9, "epoch": 0})
    # startup_loss_check logs step0_loss WITH step=0.
    assert not pap.is_step_line(
        {"step": 0, "lockstep/lora_r1/step0_loss": 4.0,
         "lockstep/lora_r1/noop_floor": 0.1})
    assert not pap.is_step_line(
        {"step": None, "lockstep/lora_r1/final_probe_loss": 0.2,
         "lockstep/lora_r1/floor_ratio": 2.0})
    assert not pap.is_step_line(
        {"step": 50, "lockstep/lora_r1/stopped_epoch": 1,
         "lockstep/lora_r1/best_epoch": 0})
    assert not pap.is_step_line({"no_step": True})
    assert not pap.is_step_line("not a dict")
    assert not pap.is_step_line({"step": 1, "some_loss": True})  # bool != rate


def test_count_step_lines_synthetic_file(tmp_path):
    p = tmp_path / "m.jsonl"
    lines = [
        {"step": 0, "lockstep/lora_r1/step0_loss": 4.0,
         "lockstep/lora_r1/noop_floor": 0.1},          # startup — no
        {"step": 0, "lockstep/lora_r1/heldout_loss": 3.0,
         "epoch": -1},                                  # baseline — no
    ]
    lines += [{"step": i + 1, "lockstep/lora_r1/soft_loss": 2.0 - i * 0.01,
               "epoch": 0} for i in range(50)]          # 50 real steps
    lines += [
        {"step": 50, "lockstep/lora_r1/heldout_loss": 1.0, "epoch": 0},
        {"step": None, "lockstep/lora_r1/final_probe_loss": 0.2,
         "lockstep/lora_r1/noop_floor": 0.1,
         "lockstep/lora_r1/floor_ratio": 2.0},
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    assert pap.count_step_lines(p) == 50


def test_count_step_lines_tolerates_torn_tail_and_blanks(tmp_path):
    """The monitor reads while the trainer appends: a torn (mid-write) last
    line must be skipped, never crash the poll."""
    p = tmp_path / "m.jsonl"
    body = "\n".join(json.dumps(
        {"step": i + 1, "lockstep/lora_r1/soft_loss": 1.0, "epoch": 0})
        for i in range(7))
    p.write_text(body + "\n\n" + '{"step": 8, "lockstep/lora_r1/sof')
    assert pap.count_step_lines(p) == 7


def test_count_step_lines_missing_file(tmp_path):
    assert pap.count_step_lines(tmp_path / "absent.jsonl") == 0


# --------------------------------------------------------------------------- #
# Tail-rate computation                                                        #
# --------------------------------------------------------------------------- #


def test_tail_rate_last_k_only():
    """First 20 steps slow (warmup), last 30 fast — the tail window must
    report the fast phase only."""
    samples = [(0.0, 0), (10.0, 5), (20.0, 10), (30.0, 15), (40.0, 20)]
    samples += [(41.0, 26), (42.0, 32), (43.0, 38), (44.0, 44), (45.0, 50)]
    tr = pap.steps_per_sec_tail(samples, tail_steps=30)
    # target = 50 - 30 = 20 -> start at (40.0, 20); end first c==50.
    assert tr.steps == 30 and tr.seconds == 5.0 and tr.rate == 6.0


def test_tail_rate_excludes_encode_phase():
    """A long zero prefix (model load + tokenization) must not count."""
    samples = [(0.0, 0), (100.0, 0), (200.0, 0), (210.0, 20), (220.0, 40)]
    tr = pap.steps_per_sec_tail(samples, tail_steps=100)
    # N=40 <= K -> start at the LAST zero sample (200.0), not t=0.
    assert tr.steps == 40 and tr.seconds == 20.0 and tr.rate == 2.0


def test_tail_rate_excludes_post_training_plateau():
    """Held-out pass + save after the last step must not dilute the rate:
    the window ends at the FIRST observation of the final count."""
    samples = [(0.0, 0), (10.0, 20), (20.0, 40), (80.0, 40), (140.0, 40)]
    tr = pap.steps_per_sec_tail(samples, tail_steps=40)
    assert tr.seconds == 20.0 and tr.rate == 2.0


def test_tail_rate_none_cases():
    assert pap.steps_per_sec_tail([], 30) is None
    assert pap.steps_per_sec_tail([(0.0, 0), (10.0, 0)], 30) is None
    # every step predates the first sample — no measurable window
    assert pap.steps_per_sec_tail([(0.0, 50)], 30) is None
    # zero elapsed
    assert pap.steps_per_sec_tail([(5.0, 0), (5.0, 10)], 30) is None


def test_tail_rate_from_synthetic_metrics_file(tmp_path):
    """The full rate path against a real file on disk: append per-step
    lines the way the trainer does, sample counts at known wall-times,
    compute the tail rate from those samples."""
    p = tmp_path / "m.jsonl"
    p.write_text("")
    samples = []
    t = 0.0
    for batch in ([0] * 3 + [10] * 5):  # 3 encode polls, then 10 steps/poll
        with open(p, "a") as fh:
            for _ in range(batch):
                n = pap.count_step_lines(p)  # not used; realism only
                del n
                fh.write(json.dumps(
                    {"step": 1, "lockstep/lora_r1/soft_loss": 1.0,
                     "epoch": 0}) + "\n")
        samples.append((t, pap.count_step_lines(p)))
        t += 2.0
    assert samples[-1][1] == 50
    tr = pap.steps_per_sec_tail(samples, tail_steps=30)
    assert tr.steps == 30 and tr.rate == pytest.approx(5.0)


# --------------------------------------------------------------------------- #
# GPU-util mean over the stepping window                                       #
# --------------------------------------------------------------------------- #


def test_mean_gpu_util_stepping_window_only():
    samples = [
        (0.0, 0, 0.0),     # model load — excluded
        (2.0, 0, 3.0),     # encode — excluded
        (4.0, 10, 90.0),   # stepping
        (6.0, 20, 94.0),   # stepping
        (8.0, 30, 92.0),   # first observation of final count — included
        (10.0, 30, 1.0),   # post-training save — excluded
    ]
    assert pap.mean_gpu_util(samples) == pytest.approx(92.0)


def test_mean_gpu_util_skips_failed_reads():
    samples = [(0.0, 0, 5.0), (2.0, 10, None), (4.0, 20, 80.0)]
    assert pap.mean_gpu_util(samples) == pytest.approx(80.0)


def test_mean_gpu_util_none_when_no_usable_sample():
    assert pap.mean_gpu_util([]) is None
    assert pap.mean_gpu_util([(0.0, 0, 50.0)]) is None       # never stepped
    assert pap.mean_gpu_util([(0.0, 0, 5.0), (2.0, 10, None)]) is None


# --------------------------------------------------------------------------- #
# Sizing + pool truncation + config derivation                                 #
# --------------------------------------------------------------------------- #


def test_pairs_needed_arithmetic():
    # 50 steps x bs 8 = 400 fit rows; /0.9 holdout; x1.15 margin = 511.2
    assert pap.pairs_needed(50, 8, 0.1) == 512
    assert pap.pairs_needed(50, 8, 0.0, margin=1.0) == 400
    assert pap.pairs_needed(1, 1, 0.0, margin=1.0) == 1


def test_pairs_needed_rejects_bad_inputs():
    with pytest.raises(ValueError):
        pap.pairs_needed(0, 8, 0.1)
    with pytest.raises(ValueError):
        pap.pairs_needed(50, 0, 0.1)
    with pytest.raises(ValueError):
        pap.pairs_needed(50, 8, 1.0)


def test_truncate_pool_head_bytes_preserved(tmp_path):
    src, dst = tmp_path / "pool.jsonl", tmp_path / "trunc.jsonl"
    rows = [json.dumps({"prompt": f"p{i}", "pos": "a", "neg": "b"})
            for i in range(10)]
    src.write_text(rows[0] + "\n\n" + "\n".join(rows[1:]) + "\n")  # blank line
    assert pap.truncate_pool(src, dst, 4) == 4
    assert dst.read_text() == "\n".join(rows[:4]) + "\n"


def test_truncate_pool_small_pool_takes_all(tmp_path):
    src, dst = tmp_path / "pool.jsonl", tmp_path / "trunc.jsonl"
    src.write_text('{"prompt": "p", "pos": "a", "neg": "b"}\n')
    assert pap.truncate_pool(src, dst, 100) == 1


def test_derive_probe_config_overrides_and_preserves():
    cfg = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())
    probe = pap.derive_probe_config(cfg)
    # the four overrides
    assert probe["data_parallel"] == {"gpus": 1}
    assert probe["train_kwargs"]["default"]["epochs"] == 1
    assert probe["train_kwargs"]["default"]["floor_ratio_max"] is None
    assert probe["wandb_fits"] is False
    # everything load-bearing is preserved from the wave config
    assert probe["model"] == cfg["model"]
    assert probe["model_revision"] == cfg["model_revision"]
    assert probe["dtype"] == cfg["dtype"]
    assert probe["max_len_by_si"] == cfg["max_len_by_si"]
    assert probe["batch_size_by_arm"] == cfg["batch_size_by_arm"]
    assert (probe["train_kwargs"]["default"]["holdout_frac"]
            == cfg["train_kwargs"]["default"]["holdout_frac"])
    # per-arm kwargs pass through (the schedule opt-in survives; the old
    # alphakl_soft beta pin is GONE from the config on purpose — beta is
    # derived from the eps budget since 2026-08-24)...
    assert (probe["train_kwargs"]["psr_kl_all"]
            == cfg["train_kwargs"]["psr_kl_all"])
    # ...EXCEPT per-arm floor bars, neutralised like the default —
    # arm_floor_ratio_max lets them BEAT the default-None, which would
    # judge fit quality on the probe's 1-epoch truncated fit.
    for arm in ("bidpo", "bidpo_lora", "diffmean"):
        assert cfg["train_kwargs"][arm]["floor_ratio_max"] is not None
        assert probe["train_kwargs"][arm]["floor_ratio_max"] is None
    # and the input was never mutated
    assert cfg["data_parallel"]["gpus"] > 1
    assert "epochs" not in cfg["train_kwargs"]["default"]


def test_derive_probe_config_tolerates_bare_config():
    probe = pap.derive_probe_config({"model": "m"})
    assert probe["data_parallel"] == {"gpus": 1}
    assert probe["train_kwargs"]["default"]["epochs"] == 1


def test_default_probe_arm_is_bs8_on_the_wave_config():
    """The docstring's 'lora_r1, bs=8' claim, asserted against the real
    config through the trainer's own authoritative resolver."""
    from sisteer.eval.train_budget import resolve_batch_size

    cfg = yaml.safe_load((REPO / "configs" / "two_si.yaml").read_text())
    assert resolve_batch_size(cfg, "safety_refusal", "lora_r1") == 8


# --------------------------------------------------------------------------- #
# Final-line format                                                            #
# --------------------------------------------------------------------------- #


def test_format_final_line_accept():
    assert (pap.format_final_line(pap.ACCEPT, None, 3.14159, 92.35)
            == "ACCEPT steps_per_sec=3.142 gpu_util=92.3")


def test_format_final_line_reject_reasons():
    assert (pap.format_final_line(pap.REJECT, "throughput", 0.5, 95.0)
            == "REJECT reason=throughput steps_per_sec=0.500 gpu_util=95.0")
    assert (pap.format_final_line(pap.REJECT, "no_steps", None, None)
            == "REJECT reason=no_steps steps_per_sec=nan gpu_util=nan")
    # nan is deliberate: parseable by float(), never a fake 0
    assert str(float("nan")) == "nan"


# --------------------------------------------------------------------------- #
# Monitor loop against a fake process + clock                                  #
# --------------------------------------------------------------------------- #


class _FakeProc:
    """poll() returns None for the first ``live_polls`` calls, then rc."""

    def __init__(self, live_polls: int, rc: int = 0):
        self._live = live_polls
        self.rc = rc
        self.returncode = None
        self.pid = 424242

    def poll(self):
        if self._live > 0:
            self._live -= 1
            return None
        self.returncode = self.rc
        return self.rc


def test_monitor_collects_samples_then_final_count():
    clock = iter(float(i) for i in range(1000))
    counts = iter([0, 5, 12, 20, 25])
    proc = _FakeProc(live_polls=4)
    res = pap.monitor_trainer(
        proc, count_steps=lambda: next(counts), sample_util=lambda: 91.0,
        deadline=1e9, poll_s=0.0, now=lambda: next(clock),
        sleep=lambda s: None,
        kill=lambda p: pytest.fail("must not kill a live run"))
    assert not res.timed_out and res.returncode == 0
    assert len(res.samples) == 5
    assert [c for _, c, _ in res.samples] == [0, 5, 12, 20, 25]
    assert all(u == 91.0 for _, _, u in res.samples[:-1])
    assert res.samples[-1][2] is None  # post-exit sample: util excluded


def test_monitor_timeout_kills_and_reports():
    killed = []
    proc = _FakeProc(live_polls=10 ** 6)
    res = pap.monitor_trainer(
        proc, count_steps=lambda: 0, sample_util=lambda: 0.0,
        deadline=0.0, poll_s=0.0, now=lambda: 1.0, sleep=lambda s: None,
        kill=lambda p: killed.append(p))
    assert res.timed_out and killed == [proc]
    assert res.samples[-1][1] == 0


# --------------------------------------------------------------------------- #
# nvidia-smi edges (subprocess.run stubbed — deterministic on any host)        #
# --------------------------------------------------------------------------- #


def test_gpu_preflight_raises_probe_error_without_nvidia_smi(monkeypatch):
    def boom(*a, **kw):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(pap.subprocess, "run", boom)
    with pytest.raises(pap.ProbeError):
        pap._gpu_preflight()


def test_sample_gpu_util_max_across_gpus(monkeypatch):
    class R:
        returncode = 0
        stdout = "3\n97\n12\n"
        stderr = ""

    monkeypatch.setattr(pap.subprocess, "run", lambda *a, **kw: R())
    assert pap._sample_gpu_util() == 97.0


def test_sample_gpu_util_none_on_failure(monkeypatch):
    def boom(*a, **kw):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(pap.subprocess, "run", boom)
    assert pap._sample_gpu_util() is None


# --------------------------------------------------------------------------- #
# run_probe end-to-end with a fake trainer (no GPU needed)                     #
# --------------------------------------------------------------------------- #

_NOOP_CMD = [sys.executable, "-c", "pass"]

_FAKE_TRAIN = textwrap.dedent("""\
    import json, pathlib, sys, time
    p = pathlib.Path(sys.argv[1]); p.parent.mkdir(parents=True, exist_ok=True)
    n, dt = int(sys.argv[2]), float(sys.argv[3])
    with open(p, "a") as fh:
        fh.write(json.dumps({"step": 0,
                             "lockstep/lora_r1/step0_loss": 4.0,
                             "lockstep/lora_r1/noop_floor": 0.1}) + "\\n")
        fh.flush()
        for i in range(n):
            fh.write(json.dumps({"step": i + 1,
                                 "lockstep/lora_r1/soft_loss": 1.0,
                                 "epoch": 0}) + "\\n")
            fh.flush()
            time.sleep(dt)
    """)


def _probe_env(tmp_path, monkeypatch, *, train_cmd, util=95.0,
               precompute_cmd=None):
    """Common fixture: mini config + pool on disk, fake GPU sampling, fake
    trainer subprocess commands."""
    cfg = {"batch_size_by_arm": {"lora_r1": 4},
           "train_kwargs": {"default": {"holdout_frac": 0.1}}}
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    pool = tmp_path / "pool.jsonl"
    pool.write_text("".join(
        json.dumps({"prompt": f"p{i}", "pos": "a", "neg": "b"}) + "\n"
        for i in range(200)))
    monkeypatch.setattr(pap, "_gpu_preflight", lambda: ["FAKE GPU 80GB"])
    monkeypatch.setattr(pap, "_sample_gpu_util", lambda: util)
    monkeypatch.setattr(pap, "_build_precompute_cmd",
                        lambda *a: precompute_cmd or _NOOP_CMD)
    monkeypatch.setattr(pap, "_build_train_cmd", train_cmd)
    return cfg_path, pool, tmp_path / "verdict.json"


def _base_args(cfg_path, pool, out):
    return ["--config", str(cfg_path), "--si", "safety_refusal",
            "--pairs", str(pool), "--out", str(out),
            "--steps", "12", "--tail-steps", "5", "--poll", "0.03",
            "--timeout", "30"]


def test_run_probe_accepts_fast_pod(tmp_path, monkeypatch, capsys):
    def train_cmd(cfg_path, si, arm, pairs, out_dir):
        metrics = Path(out_dir) / f"{si}__metrics.jsonl"
        return [sys.executable, "-c", _FAKE_TRAIN, str(metrics), "25", "0.01"]

    cfg_path, pool, out = _probe_env(tmp_path, monkeypatch,
                                     train_cmd=train_cmd)
    rc = pap.run_probe(_base_args(cfg_path, pool, out)
                       + ["--scratch", str(tmp_path / "scratch"),
                          "--min-steps-per-sec", "1.0",
                          "--min-gpu-util", "40"])
    assert rc == 0
    line = capsys.readouterr().out.strip().splitlines()[-1]
    assert line.startswith("ACCEPT steps_per_sec=")
    rec = json.loads(out.read_text())
    assert rec["verdict"] == "ACCEPT" and rec["reason"] is None
    assert rec["steps_total"] == 25
    assert rec["gpu_util_mean"] == pytest.approx(95.0)
    assert rec["steps_per_sec"] > 1.0
    assert rec["tail_window"]["steps"] >= 1
    # sizing: 12 steps x bs 4 / 0.9 x 1.15 margin = ceil(61.3) = 62 pairs
    assert rec["pairs_requested"] == 62 and rec["pairs_used"] == 62
    assert rec["batch_size"] == 4
    assert rec["samples"], "the sample trace must land in the verdict json"
    # the truncated pool really is a head slice of the real pool
    trunc = Path(rec["scratch"]) / "probe_pairs.jsonl"
    assert trunc.read_text().splitlines()[:3] == \
        pool.read_text().splitlines()[:3]
    # the derived config landed beside it with the probe overrides
    probe_cfg = yaml.safe_load(
        (Path(rec["scratch"]) / "probe_config.yaml").read_text())
    assert probe_cfg["data_parallel"] == {"gpus": 1}
    assert probe_cfg["train_kwargs"]["default"]["epochs"] == 1


def test_run_probe_rejects_low_throughput(tmp_path, monkeypatch, capsys):
    def train_cmd(cfg_path, si, arm, pairs, out_dir):
        metrics = Path(out_dir) / f"{si}__metrics.jsonl"
        return [sys.executable, "-c", _FAKE_TRAIN, str(metrics), "25", "0.01"]

    cfg_path, pool, out = _probe_env(tmp_path, monkeypatch,
                                     train_cmd=train_cmd)
    rc = pap.run_probe(_base_args(cfg_path, pool, out)
                       + ["--min-steps-per-sec", "100000"])
    assert rc == 2
    assert "REJECT reason=throughput" in capsys.readouterr().out
    assert json.loads(out.read_text())["reason"] == "throughput"


def test_run_probe_rejects_low_gpu_util(tmp_path, monkeypatch, capsys):
    def train_cmd(cfg_path, si, arm, pairs, out_dir):
        metrics = Path(out_dir) / f"{si}__metrics.jsonl"
        return [sys.executable, "-c", _FAKE_TRAIN, str(metrics), "25", "0.01"]

    cfg_path, pool, out = _probe_env(tmp_path, monkeypatch,
                                     train_cmd=train_cmd, util=4.0)
    rc = pap.run_probe(_base_args(cfg_path, pool, out)
                       + ["--min-steps-per-sec", "1.0",
                          "--min-gpu-util", "40"])
    assert rc == 2
    assert "REJECT reason=gpu_util" in capsys.readouterr().out
    rec = json.loads(out.read_text())
    assert rec["reason"] == "gpu_util"
    assert rec["gpu_util_mean"] == pytest.approx(4.0)


def test_run_probe_rejects_no_steps_on_timeout_never_hangs(
        tmp_path, monkeypatch, capsys):
    """A stepless trainer is killed at --timeout and rejected no_steps —
    the probe returns instead of hanging."""
    def train_cmd(cfg_path, si, arm, pairs, out_dir):
        return [sys.executable, "-c", "import time; time.sleep(600)"]

    cfg_path, pool, out = _probe_env(tmp_path, monkeypatch,
                                     train_cmd=train_cmd)
    args = _base_args(cfg_path, pool, out)
    args[args.index("--timeout") + 1] = "1.5"
    rc = pap.run_probe(args)
    assert rc == 2
    assert "REJECT reason=no_steps" in capsys.readouterr().out
    rec = json.loads(out.read_text())
    assert rec["reason"] == "no_steps" and rec["timed_out"] is True
    assert rec["steps_total"] == 0


def test_run_probe_rejects_crash_with_log_tail(tmp_path, monkeypatch, capsys):
    def train_cmd(cfg_path, si, arm, pairs, out_dir):
        return [sys.executable, "-c",
                "import sys; print('CUDA OOM at step 3'); sys.exit(3)"]

    cfg_path, pool, out = _probe_env(tmp_path, monkeypatch,
                                     train_cmd=train_cmd)
    rc = pap.run_probe(_base_args(cfg_path, pool, out))
    assert rc == 2
    assert "REJECT reason=crash" in capsys.readouterr().out
    rec = json.loads(out.read_text())
    assert rec["reason"] == "crash" and rec["crash_phase"] == "train"
    assert rec["train_returncode"] == 3
    assert "CUDA OOM at step 3" in rec["log_tail"]


def test_run_probe_rejects_crash_in_precompute(tmp_path, monkeypatch, capsys):
    """The sidecar build is part of the real path — a pod where it dies is
    rejected with the phase recorded."""
    cfg_path, pool, out = _probe_env(
        tmp_path, monkeypatch,
        train_cmd=lambda *a: pytest.fail("train must not start"),
        precompute_cmd=[sys.executable, "-c", "import sys; sys.exit(2)"])
    rc = pap.run_probe(_base_args(cfg_path, pool, out))
    assert rc == 2
    assert "REJECT reason=crash" in capsys.readouterr().out
    rec = json.loads(out.read_text())
    assert rec["reason"] == "crash"
    assert rec["crash_phase"] == "precompute-tokens"
    assert rec["precompute_returncode"] == 2


def test_run_probe_refuses_dpo_arms_as_probe_error(tmp_path, monkeypatch,
                                                   capsys):
    """DPO preference arms train on their own sha-pinned pool files — a
    truncated pool cannot satisfy them, so the probe errors (exit 1, a
    misinvocation) rather than issuing a pod verdict."""
    cfg_path, pool, out = _probe_env(
        tmp_path, monkeypatch,
        train_cmd=lambda *a: pytest.fail("train must not start"))
    for arm in ("bidpo", "bidpo_lora"):
        rc = pap.run_probe(_base_args(cfg_path, pool, out) + ["--arm", arm])
        assert rc == 1
        assert "PROBE-ERROR" in capsys.readouterr().err
        assert json.loads(out.read_text())["verdict"] == "error"


def test_run_probe_error_when_pairs_missing(tmp_path, monkeypatch, capsys):
    cfg_path, pool, out = _probe_env(
        tmp_path, monkeypatch,
        train_cmd=lambda *a: pytest.fail("train must not start"))
    rc = pap.run_probe(["--config", str(cfg_path), "--si", "safety_refusal",
                        "--pairs", str(tmp_path / "nope.jsonl"),
                        "--out", str(out)])
    assert rc == 1
    assert "PROBE-ERROR" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# The real trainer command lines (paths + subcommands, no execution)           #
# --------------------------------------------------------------------------- #


def test_build_cmds_point_at_the_real_trainer(tmp_path):
    pre = pap._build_precompute_cmd(tmp_path / "c.yaml", "safety_refusal",
                                    tmp_path / "p.jsonl")
    assert pre[0] == sys.executable
    assert pre[1].endswith("scripts/train_two_si_objects.py")
    assert pre[2] == "precompute-tokens"
    tr = pap._build_train_cmd(tmp_path / "c.yaml", "safety_refusal",
                              "lora_r1", tmp_path / "p.jsonl",
                              tmp_path / "objects")
    assert tr[1] == pre[1] and tr[2] == "train"
    assert tr[tr.index("--arms") + 1] == "lora_r1"
    assert "--overwrite" in tr
    # single-arm invocation writes <si>__metrics.jsonl under --out — the
    # file the monitor counts (cmd_train's make_metrics_log_fn path)
    assert tr[tr.index("--out") + 1] == str(tmp_path / "objects")
