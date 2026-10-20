"""tune_arm.py bench — the reconnaissance additions: per-run --lr /
--max-grad-norm / --warmup-frac overrides recorded on the output, the
pre-clip grad-norm series reused from the trainer's own ``<arm>/grad_norm``
record, and an explicit ok | diverged | failed outcome that still writes
the record (partial series) and exits non-zero.

CPU-only: the model load and the trainer are replaced by a fake CE adapter
that emits the trainer's per-step payload shape through ``log_fn`` (the
very seam bench records from) and can raise the trainer's divergence abort
mid-run. The real Heartbeat, the real pool sha, the real arg parser.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from test_tune_arm import tune_arm  # noqa: E402  — ONE tune_arm module instance

from sisteer.extraction.fitmon import TrainingDiverged  # noqa: E402

ARM = "psr_ce_all"


class FakeCE(tune_arm.CESoloAdapter):
    """A CESoloAdapter whose train() replays a scripted per-step record
    through log_fn — the payload keys train_object_ce writes (ce_loss, ce,
    lr, the token counts, sec_per_step, epoch, grad_norm) — then optionally
    raises the trainer's own abort or an arbitrary exception."""

    def __init__(self, norms, *, diverge_at=None, fail_at=None):
        super().__init__(ARM)
        self.norms = list(norms)
        self.diverge_at = diverge_at
        self.fail_at = fail_at
        self.seen_hp = None
        self._n_fit = 64

    def route_pairs(self, setup):
        return ["pair"] * 64

    def prepare_data(self, setup, pool, *, length_sort=True, take_longest=None):
        self._fit_ex = ["ex"] * 64

    def make_object(self, setup, sentinel=None, lora_rank=None):
        self._obj = object()

    def optimizer_steps(self, batch_size, epochs):
        return 1000

    def accum_for(self, batch_size):
        return 64 // int(batch_size)

    def train(self, setup, hp, *, epochs, batch_size, seed, log_fn, probe_cb,
              probe_every, diagnostics):
        self.seen_hp = hp
        self._resolved = {"warmup_steps": 7, "total_steps": len(self.norms),
                          "accum_steps": self.accum_for(batch_size),
                          "effective_batch": 64, "early_stop_patience": 2}
        for i, g in enumerate(self.norms, start=1):
            if self.diverge_at == i:
                raise TrainingDiverged(f"{ARM}: non-finite training loss "
                                       f"(nan) at optimizer step {i}")
            if self.fail_at == i:
                raise RuntimeError("CUDA error: an illegal memory access")
            log_fn({f"{ARM}/ce_loss": 2.5 - 0.01 * i, f"{ARM}/ce": 2.5 - 0.01 * i,
                    f"{ARM}/lr": hp.lr * min(1.0, i / 7),
                    f"{ARM}/loss_tokens": 1000, f"{ARM}/real_tokens": 1000,
                    f"{ARM}/padded_tokens": 1100,
                    f"{ARM}/sec_per_step": 0.01, "epoch": 0,
                    f"{ARM}/grad_norm": g}, i)
        if diagnostics is not None:
            diagnostics.setdefault(ARM, {})["stopped_by_max_steps"] = True


def _args(tmp_path, **over):
    pool = tmp_path / "pairs.jsonl"
    if not pool.exists():
        pool.write_text('{"prompt": "p", "pos": "x", "neg": ""}\n')
    ns = dict(config=str(tmp_path / "cfg.yaml"), pairs=str(pool),
              si="constv2-mix-full", batch_size=1, allow_tokenize=False,
              bidpo_harmful_pairs=None, bidpo_benign_pairs=None,
              teacher_topk="off", arm=ARM, steps=20, accum=0, seed=0,
              out=str(tmp_path / "bench.json"), check_grads=False,
              loss_impl="config", chunk_tokens=0,
              gradient_checkpointing="config",
              lr=None, max_grad_norm=None, warmup_frac=None)
    ns.update(over)
    return argparse.Namespace(**ns)


def _world(monkeypatch, fake, cfg=None):
    setup = SimpleNamespace(cfg=cfg or {"model": "m", "model_revision": "r"},
                            max_len=64, model=None, tok=None)
    monkeypatch.setattr(tune_arm, "make_adapter", lambda arm: fake)
    monkeypatch.setattr(tune_arm, "_setup", lambda *a, **k: setup)
    return setup


NORMS = [0.9, 0.8, 0.7, 0.6, 0.5,            # warm-up steps 1-5 (dropped)
         0.30, 0.32, 0.28, 0.35, 0.31, 0.29, 0.33, 0.30, 2.40, 0.31,
         0.30, 0.34, 0.29, 0.31, 0.30]       # steps 6-20; one spike at 14


def test_overrides_reach_the_trainer_and_the_record(tmp_path, monkeypatch):
    fake = FakeCE(NORMS)
    _world(monkeypatch, fake)
    tune_arm.bench(_args(tmp_path, lr=4.5e-4, max_grad_norm=1e6,
                         warmup_frac=0.15))
    assert (fake.seen_hp.lr, fake.seen_hp.max_grad_norm,
            fake.seen_hp.warmup_frac) == (4.5e-4, 1e6, 0.15)
    rec = json.loads((tmp_path / "bench.json").read_text())
    assert rec["status"] == "ok" and rec["error"] == ""
    assert rec["exception_type"] is None
    assert (rec["lr"], rec["max_grad_norm"], rec["warmup_frac"]) == \
        (4.5e-4, 1e6, 0.15)
    assert rec["hp_source"] == {"lr": "flag", "max_grad_norm": "flag",
                                "warmup_frac": "flag"}
    assert rec["warmup_steps"] == 7 and rec["total_steps_planned"] == 20
    assert rec["steps_requested"] == 20 and rec["steps_run"] == 20
    assert rec["steps_completed"] == 20 and rec["steps_measured"] == 15
    assert rec["effective_batch"] == 64 and rec["accum_steps"] == 64
    # the grad-norm record IS the trainer's <arm>/grad_norm series
    gn = rec["grad_norm_preclip"]
    assert gn["key"] == f"{ARM}/grad_norm"
    assert gn["steps"] == list(range(1, 21)) and gn["values"] == NORMS
    assert gn["n_steps"] == 20 and gn["warmup_steps_dropped"] == 5
    assert gn["quantiles"]["max"] == 2.4 and gn["quantiles"]["n"] == 20
    post = NORMS[5:]
    assert gn["quantiles_after_warmup"]["n"] == 15
    assert gn["quantiles_after_warmup"]["p50"] == \
        tune_arm.recon_clip.quantile(post, 0.5)
    assert gn["quantiles_after_warmup"]["p95"] == \
        tune_arm.recon_clip.quantile(post, 0.95)
    assert rec["loss"]["key"] == f"{ARM}/ce_loss"
    assert rec["loss"]["steps"] == list(range(1, 21))
    assert len(rec["loss"]["values"]) == 20
    # the throughput fields still compute over the steady window
    assert rec["tokens_real"] == 15 * 1000 and rec["sec_per_step_median"] == 0.01
    assert rec["stopped_by_max_steps"] is True
    # the record feeds the clip reader directly — and a 20-step smoke run
    # is INCOMPLETE for clip-setting (short of the reconnaissance budget,
    # whatever it requested): the reader refuses to set a clip from it
    v = tune_arm.recon_clip.classify_run(rec)
    assert v["class"] == "incomplete" and not v["eligible"]
    assert "reconnaissance budget" in v["reasons"][0]
    with pytest.raises(ValueError, match="no STABLE recon run"):
        tune_arm.recon_clip.clip_from_records({"r1": rec}, arm=ARM)


def test_config_block_then_defaults_when_no_flags(tmp_path, monkeypatch):
    fake = FakeCE(NORMS)
    _world(monkeypatch, fake, cfg={
        "model": "m", "train_kwargs": {ARM: {"lr": 3e-4, "warmup_frac": 0.15,
                                             "max_grad_norm": 0.7}}})
    tune_arm.bench(_args(tmp_path))
    rec = json.loads((tmp_path / "bench.json").read_text())
    assert (rec["lr"], rec["warmup_frac"], rec["max_grad_norm"]) == \
        (3e-4, 0.15, 0.7)
    assert rec["hp_source"] == {"lr": "config", "max_grad_norm": "config",
                                "warmup_frac": "config"}
    # a sentinel (non-numeric) block falls to the historical defaults
    fake2 = FakeCE(NORMS)
    _world(monkeypatch, fake2, cfg={
        "model": "m", "train_kwargs": {ARM: {"lr": "PENDING_HPO",
                                             "max_grad_norm": "PENDING_HPO"}}})
    tune_arm.bench(_args(tmp_path))
    rec = json.loads((tmp_path / "bench.json").read_text())
    assert (rec["lr"], rec["warmup_frac"], rec["max_grad_norm"]) == \
        (1e-5, 0.05, 1.0)
    assert rec["hp_source"] == {"lr": "default", "max_grad_norm": "default",
                                "warmup_frac": "default"}
    # a flag beats a numeric block
    fake3 = FakeCE(NORMS)
    _world(monkeypatch, fake3, cfg={
        "model": "m", "train_kwargs": {ARM: {"lr": 3e-4}}})
    tune_arm.bench(_args(tmp_path, lr=9e-4))
    assert fake3.seen_hp.lr == 9e-4


def test_non_finite_clip_is_refused_before_training(tmp_path, monkeypatch):
    fake = FakeCE(NORMS)
    _world(monkeypatch, fake)
    with pytest.raises(SystemExit, match="finite positive"):
        tune_arm.bench(_args(tmp_path, max_grad_norm=float("inf")))
    assert fake.seen_hp is None
    assert not (tmp_path / "bench.json").exists()
    with pytest.raises(SystemExit, match="lr="):
        tune_arm.bench(_args(tmp_path, lr=0.0))
    with pytest.raises(SystemExit, match="warmup_frac"):
        tune_arm.bench(_args(tmp_path, warmup_frac=1.0))


def test_divergence_writes_a_diverged_record_with_the_partial_series(
        tmp_path, monkeypatch, capsys):
    fake = FakeCE(NORMS, diverge_at=12)
    _world(monkeypatch, fake)
    with pytest.raises(SystemExit) as ei:
        tune_arm.bench(_args(tmp_path, lr=1e-2, max_grad_norm=1e6))
    assert ei.value.code == tune_arm.BENCH_EXIT_DIVERGED == 3
    rec = json.loads((tmp_path / "bench.json").read_text())
    assert rec["status"] == "diverged"
    assert rec["exception_type"] == "TrainingDiverged"
    assert "non-finite training loss" in rec["error"]
    assert rec["steps_completed"] == 11 and rec["steps_run"] == 11
    assert rec["steps_requested"] == 20
    assert rec["grad_norm_preclip"]["values"] == NORMS[:11]
    assert rec["grad_norm_preclip"]["steps"] == list(range(1, 12))
    assert len(rec["loss"]["values"]) == 11
    assert rec["lr"] == 1e-2 and rec["max_grad_norm"] == 1e6
    # the throughput block is None-safe on a partial run (6 steady steps)
    assert rec["steps_measured"] == 6 and rec["tokens_real"] == 6000
    out = capsys.readouterr().out
    assert "[bench] DIVERGED after 11 optimizer step(s)" in out
    assert "[bench] RESULT " in out
    # the recon reader classifies it as DIVERGED — diagnostic only
    v = tune_arm.recon_clip.classify_run(rec)
    assert v["class"] == "diverged" and not v["eligible"]


def test_other_exceptions_write_a_failed_record(tmp_path, monkeypatch):
    fake = FakeCE(NORMS, fail_at=3)
    _world(monkeypatch, fake)
    with pytest.raises(SystemExit) as ei:
        tune_arm.bench(_args(tmp_path))
    assert ei.value.code == tune_arm.BENCH_EXIT_FAILED == 4
    rec = json.loads((tmp_path / "bench.json").read_text())
    assert rec["status"] == "failed"
    assert rec["exception_type"] == "RuntimeError"
    assert "illegal memory access" in rec["error"]
    assert rec["steps_completed"] == 2
    assert rec["grad_norm_preclip"]["values"] == NORMS[:2]
    # no steady window: every throughput field is None, none is fabricated
    assert rec["steps_measured"] == 0
    for k in ("tok_per_s_loss", "sec_per_step_median", "pad_fraction",
              "steady_train_s"):
        assert rec[k] is None
    assert tune_arm.recon_clip.classify_run(rec)["class"] == "failed"


def test_too_few_steps_is_an_explicit_failed_outcome_not_a_bare_crash(
        tmp_path, monkeypatch):
    fake = FakeCE(NORMS[:4])
    _world(monkeypatch, fake)
    with pytest.raises(SystemExit) as ei:
        tune_arm.bench(_args(tmp_path, steps=4))
    assert ei.value.code == tune_arm.BENCH_EXIT_FAILED
    rec = json.loads((tmp_path / "bench.json").read_text())
    assert rec["status"] == "failed" and "steady window" in rec["error"]
    assert rec["steps_completed"] == 4


def test_no_out_still_prints_the_result_line(tmp_path, monkeypatch, capsys):
    fake = FakeCE(NORMS, diverge_at=9)
    _world(monkeypatch, fake)
    with pytest.raises(SystemExit):
        tune_arm.bench(_args(tmp_path, out=""))
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("[bench] RESULT "))
    rec = json.loads(line[len("[bench] RESULT "):])
    assert rec["status"] == "diverged" and rec["steps_completed"] == 8
    assert "no --out" in out


def test_cli_round_trip_of_the_three_flags(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(tune_arm, "bench", lambda a: seen.update(vars(a)))
    monkeypatch.setattr(sys, "argv", [
        "tune_arm.py", "bench", "--pairs", "p.jsonl", "--arm", ARM,
        "--batch-size", "1", "--accum", "64", "--steps", "150",
        "--lr", "4.5e-4", "--max-grad-norm", "1e6", "--warmup-frac", "0.05"])
    tune_arm.main()
    assert (seen["lr"], seen["max_grad_norm"], seen["warmup_frac"]) == \
        (4.5e-4, 1e6, 0.05)
    assert seen["accum"] == 64 and seen["steps"] == 150
    seen.clear()
    monkeypatch.setattr(sys, "argv", ["tune_arm.py", "bench", "--pairs",
                                      "p.jsonl", "--arm", ARM])
    tune_arm.main()
    assert seen["lr"] is None and seen["max_grad_norm"] is None
    assert seen["warmup_frac"] is None


def test_bench_warmup_constant_is_the_clip_rules():
    assert tune_arm.BENCH_WARMUP_STEPS == tune_arm.recon_clip.BENCH_WARMUP_STEPS == 5
    assert math.isfinite(tune_arm.BENCH_EXIT_DIVERGED)
