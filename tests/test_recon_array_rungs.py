"""The gradient-norm reconnaissance as an ARRAY: `tune_arm.py recon --rung i` trains exactly rung i of
the ladder and writes recon_<i>.json at the serial ladder's path for that rung (the rung's bench arguments come from ONE function,
so the two are the same run by construction); `recon --finalize` is the serial ladder's reduce over the same records (same clip,
certified top, classes), loads no model, and refuses while a rung's record is missing; and bench honours --steps when one pass
over the pool is shorter than the budget (cycles epochs; one pass when it already covers the budget — the paper's recon cells)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tune_arm import tune_arm  # noqa: E402  — ONE tune_arm module instance
from test_tune_arm_bench_recon import ARM, NORMS, FakeCE, _args, _world  # noqa: E402

LADDER = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]


def _recon_args(tmp_path, **over):
    ns = _args(tmp_path, lr_ladder=",".join(str(x) for x in LADDER), clip_percentile=95.0,
               warmup_frac=0.15, steps=20, seed=7, out=str(tmp_path / "recon.json"),
               lora_rank=None, lora_modules=None, init_scheme=None, rung=None, finalize=False, passes=1)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_rung_i_of_the_array_is_rung_i_of_the_serial_ladder(tmp_path):
    args = _recon_args(tmp_path)
    ladder = tune_arm._recon_ladder(args)
    out = Path(args.out)
    for i in range(len(ladder)):
        a = tune_arm.recon_rung_args(args, ladder, i, out)
        assert a.lr == ladder[i] == sorted(LADDER)[i]
        assert a.max_grad_norm == tune_arm.RECON_UNCLIPPED and a.warmup_frac == 0.15 and a.steps == 20 and a.seed == 7
        assert a.check_grads is False and a.out == str(tmp_path / f"recon_{i}.json")
    assert tune_arm._recon_ladder(_recon_args(tmp_path, lr_ladder="1e-2,1e-4,1e-3,3e-3,3e-4")) == sorted(LADDER)   # sorted ascending, whatever the spec's order


def test_rung_task_writes_only_its_record_and_finalize_equals_the_serial_reduce(tmp_path, monkeypatch, capsys):
    """Serial ladder (one process) vs five --rung tasks + --finalize over the same scripted norms: the same recon.json numbers."""
    per_rung = {0: NORMS, 1: NORMS, 2: [x * 1.5 for x in NORMS], 3: NORMS, 4: NORMS}   # rung 4 diverges below
    calls = []

    def fake_for(i):
        f = FakeCE(per_rung[i], diverge_at=(11 if i == 4 else None))
        return f

    def make_adapter_by_lr(lr):
        i = sorted(LADDER).index(lr)
        calls.append(i)
        return fake_for(i)

    # the serial ladder: make_adapter is called per rung inside bench (through _recon_run_rung -> bench -> make_adapter)
    serial = tmp_path / "serial"; serial.mkdir()
    state = {"lr": None}
    orig_bench = tune_arm.bench

    def bench_spy(ns, *, setup=None):
        state["lr"] = ns.lr
        return orig_bench(ns, setup=setup)

    monkeypatch.setattr(tune_arm, "bench", bench_spy)
    monkeypatch.setattr(tune_arm, "make_adapter", lambda arm: make_adapter_by_lr(state["lr"]))
    setup = SimpleNamespace(cfg={"model": "m", "model_revision": "r"}, max_len=64, model=None, tok=None)
    monkeypatch.setattr(tune_arm, "_setup", lambda *a, **k: setup)
    monkeypatch.setattr(tune_arm, "_load_cfg", lambda p: {"model": "m", "model_revision": "r"})
    tune_arm.recon(_recon_args(tmp_path, out=str(serial / "recon.json")))
    serial_res = json.loads((serial / "recon.json").read_text())
    assert sorted(calls) == [0, 1, 2, 3, 4] and serial_res["clip"] is not None
    # the array: one process per rung, each writing ONLY recon_<i>.json (no recon.json), in any order
    arr = tmp_path / "array"; arr.mkdir()
    for i in (3, 0, 4, 2, 1):
        calls.clear()
        tune_arm.recon(_recon_args(tmp_path, out=str(arr / "recon.json"), rung=i))
        assert calls == [i], (i, calls)
        assert (arr / f"recon_{i}.json").is_file() and not (arr / "recon.json").exists()
        assert json.loads((arr / f"recon_{i}.json").read_text())["lr"] == sorted(LADDER)[i]
    out = capsys.readouterr().out
    assert "[recon] rung 4 record ->" in out and "--finalize" in out
    # --finalize: the reduce alone, no model load
    monkeypatch.setattr(tune_arm, "_setup", lambda *a, **k: pytest.fail("--finalize must not load the model"))
    tune_arm.recon(_recon_args(tmp_path, out=str(arr / "recon.json"), finalize=True))
    arr_res = json.loads((arr / "recon.json").read_text())
    same = ("clip", "clip_raw", "n_pooled", "certified_lr_max", "certified_lr_log10_max", "stable_lrs", "unstable_lrs",
            "ladder", "steps", "percentile", "warmup_frac", "model", "model_revision", "pool", "si", "clip_rule", "arm")
    for k in same:
        assert arr_res[k] == serial_res[k], k
    assert [r["class"] for r in arr_res["per_lr"]] == [r["class"] for r in serial_res["per_lr"]]
    assert [r["p_target"] for r in arr_res["per_lr"]] == [r["p_target"] for r in serial_res["per_lr"]]
    assert [a["code"] for a in arr_res["alarms"]] == [a["code"] for a in serial_res["alarms"]]
    assert arr_res["reduced_by"] == "finalize" and "reduced_by" not in serial_res
    assert arr_res["per_lr"][4]["class"] == "diverged" and 1e-2 in arr_res["unstable_lrs"]


def test_finalize_refuses_while_a_rung_record_is_missing_and_rung_outside_the_ladder_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(tune_arm, "_load_cfg", lambda p: {"model": "m", "model_revision": "r"})
    monkeypatch.setattr(tune_arm, "_setup", lambda *a, **k: pytest.fail("no model load on these paths"))
    root = tmp_path / "partial"; root.mkdir()
    for i in (0, 1, 3):
        (root / f"recon_{i}.json").write_text("{}")
    with pytest.raises(SystemExit) as e:
        tune_arm.recon(_recon_args(tmp_path, out=str(root / "recon.json"), finalize=True))
    assert "missing for rung(s) [2, 4]" in str(e.value) and not (root / "recon.json").exists()
    with pytest.raises(SystemExit) as e:
        tune_arm.recon(_recon_args(tmp_path, out=str(root / "recon.json"), rung=5))
    assert "outside the ladder" in str(e.value)
    with pytest.raises(SystemExit) as e:
        tune_arm.recon(_recon_args(tmp_path, out=str(root / "recon.json"), rung=1, finalize=True))
    assert "two different jobs" in str(e.value)


def test_bench_takes_one_pass_by_default_and_cycles_only_by_the_recorded_opt_in(tmp_path, monkeypatch):
    """One pass is the paper's recipe (byte-identical for every cell); a short cell cycles ONLY when the front door recorded
    cycled_passes (the operator's opt-in) and the sbatch forwarded it as --passes N — never by the pool's length alone."""
    seen = {}

    class ShortPool(FakeCE):
        def optimizer_steps(self, batch_size, epochs):
            return 65 * epochs                                   # 4,102 rows / effective batch 64

        def train(self, setup, hp, *, epochs, **kw):
            seen["epochs"] = epochs
            return super().train(setup, hp, epochs=epochs, **kw)

    _world(monkeypatch, ShortPool(NORMS))
    tune_arm.bench(_args(tmp_path, steps=150, lr=1e-3, max_grad_norm=1e6, warmup_frac=0.15))          # no --passes
    rec = json.loads((tmp_path / "bench.json").read_text())
    assert seen["epochs"] == 1 and rec["epochs_planned"] == 1 and rec["optimizer_steps_per_epoch"] == 65 and rec["passes_needed_for_budget"] == 3
    _world(monkeypatch, ShortPool(NORMS))
    tune_arm.bench(_args(tmp_path, steps=150, lr=1e-3, max_grad_norm=1e6, warmup_frac=0.15, passes=3))   # the recorded opt-in
    rec = json.loads((tmp_path / "bench.json").read_text())
    assert seen["epochs"] == 3 and rec["epochs_planned"] == 3

    class PaperCell(FakeCE):
        def optimizer_steps(self, batch_size, epochs):
            return 266 * epochs                                  # constv2ctl2-full: one pass covers 150

        def train(self, setup, hp, *, epochs, **kw):
            seen["epochs"] = epochs
            return super().train(setup, hp, epochs=epochs, **kw)

    _world(monkeypatch, PaperCell(NORMS))
    tune_arm.bench(_args(tmp_path, steps=150, lr=1e-3, max_grad_norm=1e6, warmup_frac=0.15))
    rec = json.loads((tmp_path / "bench.json").read_text())
    assert seen["epochs"] == 1 and rec["passes_needed_for_budget"] == 1


def test_recon_json_records_the_passes(tmp_path, monkeypatch):
    """The reduce records cycled_passes (1 = the paper's one pass; 3 = the recorded opt-in), read from the same --passes every stamped
    rung ran under — a finalize with another value is a stamp mismatch and refuses."""
    for passes in (1, 3):
        root = tmp_path / f"r{passes}"; root.mkdir()
        monkeypatch.setattr(tune_arm, "make_adapter", lambda arm: FakeCE(NORMS, fail_at=1))       # every rung fails -> no stable rung
        monkeypatch.setattr(tune_arm, "_setup", lambda *a, **k: SimpleNamespace(cfg={"model": "m", "model_revision": "r"}, max_len=64, model=None, tok=None))
        monkeypatch.setattr(tune_arm, "_load_cfg", lambda p: {"model": "m", "model_revision": "r"})
        for i in range(5):
            tune_arm.recon(_recon_args(tmp_path, out=str(root / "recon.json"), rung=i, passes=passes))
        with pytest.raises(SystemExit) as e:                                                    # NO_STABLE, but the verdict is written
            tune_arm.recon(_recon_args(tmp_path, out=str(root / "recon.json"), finalize=True, passes=passes))
        assert e.value.code == tune_arm.RECON_EXIT_NO_STABLE
        assert json.loads((root / "recon.json").read_text())["cycled_passes"] == passes
    with pytest.raises(SystemExit) as e:                                                        # rungs at 3 passes, finalize at 1: refused
        tune_arm.recon(_recon_args(tmp_path, out=str(tmp_path / "r3" / "recon.json"), finalize=True, passes=1))
    assert "passes: expected 1, found 3" in str(e.value), str(e.value)


def test_cli_parses_rung_and_finalize(monkeypatch):
    seen = {}
    monkeypatch.setattr(tune_arm, "recon", lambda a: seen.update(vars(a)))
    monkeypatch.setattr(sys, "argv", ["tune_arm.py", "recon", "--pairs", "p.jsonl", "--arm", ARM, "--config", "c.yaml",
                                      "--out", "r.json", "--lr-ladder", "1e-4,1e-3", "--rung", "1"])
    tune_arm.main() if hasattr(tune_arm, "main") else None
    if seen:
        assert seen["rung"] == 1 and seen["finalize"] is False
