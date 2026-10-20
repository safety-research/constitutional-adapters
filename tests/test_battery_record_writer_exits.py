"""battery_record.py settles a ladder where the WRITER says the pair left it.

The ladder writer (ladder_decide.py / write_ladder.py) records `left_ladder_at: {status: left, rung, exit}` and lists every
campaign rung — including the 16,384 extend it has not run — under `rungs`, with `reached_zero_at` phrased
"<rung> + residual <n> (<exit>)". The record's chain walked on past a pair the writer had exited by exit_le_1pct and held it
for a rung nobody climbed; `ladder_final` rejected the phrase for its suffix. Both now read the writer's verdict.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sdfbase" / "full22" / "strength" / "battery_record.py"
sys.path.insert(0, str(REPO / "src"))


def _load():
    spec = importlib.util.spec_from_file_location("battery_record_writer_exits", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PAIR_LEFT = {"n_at_cap": {"512": 154, "2048": 13}, "reached_zero_at": "2048 + residual 13 (exit_le_1pct)",
             "rungs_attempted": [512, 2048],
             "left_ladder_at": {"status": "left", "rung": 2048, "exit": "exit_le_1pct", "n_truncated": 0}}
PAIR_ESCALATING = {"n_at_cap": {"512": 154, "2048": 13}, "reached_zero_at": "incomplete", "rungs_attempted": [512, 2048],
                   "left_ladder_at": {"status": "escalating"}}
REC = {"rungs": [512, 2048, 8192, 16384], "cells": {"none": {"xstest": PAIR_LEFT, "or_bench_hard_1k": PAIR_ESCALATING}}}


def test_ladder_final_reads_the_writers_left_ladder_at_before_the_count_phrase():
    br = _load()
    rung, status = br.ladder_final(REC, "none", "xstest")
    assert rung == 2048 and "exit_le_1pct" in status
    rung2, status2 = br.ladder_final(REC, "none", "or_bench_hard_1k")
    assert rung2 is None and status2 == "incomplete"                        # still escalating: not a final rung
    # the ceiling phrase with the writer's exit word appended still yields the rung (the count is the first token)
    rec = {"cells": {"c": {"x": {"reached_zero_at": "8192 + residual 37 (exit_ceiling)"}}}}
    assert br.ladder_final(rec, "c", "x") == (8192, "8192 + residual 37 (exit_ceiling)")
    assert br.ladder_final(rec, "c", "missing") == (None, "ladder record has no (c, missing) entry")


def test_left_rung_is_the_writers_exit_rung_or_none():
    br = _load()
    assert br.left_rung(REC, "none", "xstest") == 2048
    assert br.left_rung(REC, "none", "or_bench_hard_1k") is None
    assert br.left_rung(None, "none", "xstest") is None and br.left_rung(REC, "nope", "xstest") is None


def _present(counts):
    return {R: (({"n_at_cap": n}, None) if n is not None else (None, f"no store at rung {R}")) for R, n in counts.items()}


def test_rung_chain_settles_at_the_writers_exit_rung_with_its_residual():
    br = _load()
    ladder = (512, 2048, 8192, 16384)
    present = _present({512: 154, 2048: 13, 8192: None, 16384: None})
    walked = br.rung_chain(present, ladder, lambda b: b["n_at_cap"])              # no writer verdict: the old rule holds
    assert walked["settled"] is False and walked["deepest_complete"] == 2048 and "rung 8192 is not a complete store" in walked["why"]
    left = br.rung_chain(present, ladder, lambda b: b["n_at_cap"], left_at=2048)   # the writer exited it at 2048
    assert left["settled"] is True and left["final"] == 2048 and left["chain"][-1] == {"rung": 2048, "complete": True, "n_at_cap": 13}
    assert left["at_ceiling_with_residual"] is False                               # a residual, but not the ceiling
    zero = br.rung_chain(_present({512: 154, 2048: 0}), ladder, lambda b: b["n_at_cap"], left_at=8192)
    assert zero["final"] == 2048                                                   # at-cap 0 still settles first
