"""The batch-forward microbenchmark's DECISION RULE (the actionable part; the
timing loop itself is CPU-validated ad-hoc). A harness bug here would misread the
anomaly, so the flat/linear verdict is pinned against synthetic rows."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _bench():
    p = Path(__file__).resolve().parents[1] / "scripts" / "bench_batch_forward.py"
    spec = importlib.util.spec_from_file_location("bench_batch_forward", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _rows(t1, t8):
    return [{"tag": "no_ctx", "seq_len": 300, "batch": 1, "per_forward_ms": t1,
             "per_seq_ms": t1},
            {"tag": "no_ctx", "seq_len": 300, "batch": 8, "per_forward_ms": t8,
             "per_seq_ms": t8 / 8}]


def test_flat_is_memory_bound():
    v = _bench()._verdict(_rows(100.0, 130.0), 300)   # ratio 1.3 -> flat
    assert v["ratio_B8_over_B1"] == 1.3
    assert v["verdict"].startswith("FLAT_memory_bound")


def test_linear_is_compute_bound():
    v = _bench()._verdict(_rows(100.0, 800.0), 300)   # ratio 8 -> linear
    assert v["verdict"].startswith("LINEAR_compute_bound")


def test_partial_between():
    v = _bench()._verdict(_rows(100.0, 350.0), 300)   # ratio 3.5 -> partial
    assert v["verdict"].startswith("PARTIAL")


def test_insufficient_data_when_missing_a_batch():
    rows = [{"tag": "no_ctx", "seq_len": 300, "batch": 1, "per_forward_ms": 100.0,
             "per_seq_ms": 100.0}]
    assert _bench()._verdict(rows, 300)["verdict"] == "insufficient_data"
