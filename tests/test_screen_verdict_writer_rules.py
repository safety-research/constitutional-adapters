"""screen_verdict.py reads the WRITERS' rules, not its own re-derivation.

Two reader gaps read every base->instruct controls cell as `pending`:

* the XSTest pin was 450 for every target, while the roster declares `xstest_subset: safe` for llama8b (its over-refusal
  control is the 250 SAFE prompts) — the pin is now per target, from the roster;
* the ladder walk accepted only at-cap 0 or the ceiling, while the writer's gate exits a pair by `exit_le_1pct`
  (truncated <= 1%; loop / repetitive / boundary items stay by policy) — `ladder.json` says so itself ("`reached_zero_at` ...
  never the governing verdict — `left_ladder_at` is"), and the scanner had dropped both `left_ladder_at` and `decisions`.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sdfbase" / "full22" / "strength" / "screen_verdict.py"
sys.path.insert(0, str(REPO / "src"))


def _load():
    spec = importlib.util.spec_from_file_location("screen_verdict_writer_rules", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_pinned_count_is_per_target_and_the_longest_key_prefix_wins():
    sv = _load()
    sv.ROSTER_KEYS[:] = ["llama8b", "qwen35_4b", "qwen35", "gemma4e4b"]
    sv.SUBSET_PINS.clear()
    sv.SUBSET_PINS["llama8b"] = {"xstest": sv.XSTEST_SAFE}
    sv.SUBSET_PINS["qwen35"] = {"xstest": 7}                      # a decoy shorter key: never chosen for qwen35_4b tags
    assert sv._key_of_tag("qwen35_4b_d0p25") == "qwen35_4b"
    assert sv._key_of_tag("llama8b_dm1") == "llama8b" and sv._key_of_tag(None) is None and sv._key_of_tag("nemo") is None
    assert sv._pinned("xstest", "llama8b_dm1") == 250
    assert sv._pinned("or_bench_hard_1k", "llama8b_dm1") == 1319
    assert sv._pinned("xstest", "qwen35_4b_d0p25") == 450 and sv._pinned("xstest", "gemma4e4b_d1") == 450
    assert sv._pinned("xstest", None) == 450                      # no TAG: the screen's default, never a KeyError


def test_roster_load_fills_the_subset_pins_from_xstest_subset(tmp_path):
    sv = _load()
    roster = {"wave": "sdfbase_full22", "targets": {
        "llama8b": {"xstest_subset": "safe",
                    "views": {"dm1": {"dose": -1.0, "cells": {"none": None, "constv2-mix-full__lora_r4_ce": "dm1__constv2-mix-full__lora_r4_ce"}}}},
        "qwen35_4b": {"views": {"dm1": {"dose": -1.0, "cells": {"none": None, "constv2-mix-full__lora_r4_ce": "dm1__constv2-mix-full__lora_r4_ce"}}}}}}
    p = tmp_path / "roster.yaml"
    p.write_text(yaml.safe_dump(roster))
    sv.load_dose_views(p)
    assert sv.SUBSET_PINS == {"llama8b": {"xstest": 250}}
    assert sorted(sv.ROSTER_KEYS) == ["llama8b", "qwen35_4b"]


def test_scan_ladder_keeps_the_writers_governing_verdict(tmp_path):
    sv = _load()
    d = tmp_path / "ladders" / "llama8b_dm1_t0"
    d.mkdir(parents=True)
    pair = {"n_at_cap": {"512": 154, "2048": 13}, "reached_zero_at": "2048 + residual 13 (exit_le_1pct)", "residual": 13,
            "rungs_attempted": [512, 2048],
            "left_ladder_at": {"status": "left", "rung": 2048, "exit": "exit_le_1pct", "n_truncated": 0, "n_loop": 3,
                               "n_repetitive": 10, "n_boundary": 0, "unresolved_truncation": False},
            "decisions": {"512": {"decision": "escalate", "n_truncated": 141, "n_loop": 3, "n_repetitive": 10, "n_boundary": 0,
                                  "n_at_cap": 154, "next_cap": 2048, "gate": "/somewhere/gate.json"},
                          "2048": {"decision": "exit_le_1pct", "n_truncated": 0, "n_loop": 3, "n_repetitive": 10,
                                   "n_boundary": 0, "n_at_cap": 13, "next_cap": None}},
            "roots": {"512": "/x", "2048": "/y"}}
    (d / "ladder.json").write_text(json.dumps({"rungs": [512, 2048, 8192, 16384], "exit_rule": "…", "extend_kind": "controls",
                                              "cells": {"none": {"or_bench_hard_1k": pair}}}))
    rec = sv._scan_ladder(str(tmp_path), "llama8b_dm1", 0)["record"]["cells"]["none"]["or_bench_hard_1k"]
    assert rec["left_ladder_at"]["exit"] == "exit_le_1pct" and rec["left_ladder_at"]["rung"] == 2048
    assert rec["decisions"]["2048"]["decision"] == "exit_le_1pct" and rec["decisions"]["2048"]["n_at_cap"] == 13
    assert "gate" not in rec["decisions"]["512"] and "roots" not in rec            # slimmed: the decision and its counts only
    assert rec["n_at_cap"] == {"512": 154, "2048": 13} and rec["rungs_attempted"] == [512, 2048]


def _rec(pair_record):
    rungs = {str(c): {"cap": c, "root": f"/r/{c}", "root_exists": c in (512, 2048), "stores": [], "helper": {}, "tag": "llama8b_dm1"}
             for c in (512, 2048, 8192, 16384)}
    return {"tag": "llama8b_dm1", "rungs": rungs, "root_exists": True, "stores": [], "helper": {},
            "ladder": {"record": {"rungs": [512, 2048, 8192, 16384], "cells": {"none": {"or_bench_hard_1k": pair_record}}}}}


def _fake_reader(counts):
    def read_store_control(rr, cell, ctl, cap):
        if cap not in counts:
            return {"state": "absent_root", "n_at_cap": None, "n_total": None, "share": None, "dir": None,
                    "helper_check": "unavailable", "reasons": [f"rung {cap}: never written (no controls_cap{cap} root)"]}
        n = counts[cap]
        return {"state": "complete", "n_at_cap": n, "n_total": 1319, "share": n / 1319, "dir": f"/r/{cap}/none",
                "helper_check": "unavailable", "reasons": []}
    return read_store_control


def test_walk_marks_a_pair_complete_when_the_writer_exited_it_by_exit_le_1pct(monkeypatch):
    sv = _load()
    monkeypatch.setattr(sv, "read_store_control", _fake_reader({512: 154, 2048: 13}))
    pair = {"n_at_cap": {"512": 154, "2048": 13}, "rungs_attempted": [512, 2048],
            "left_ladder_at": {"status": "left", "rung": 2048, "exit": "exit_le_1pct"},
            "decisions": {"2048": {"decision": "exit_le_1pct", "n_truncated": 0, "n_loop": 3, "n_repetitive": 10, "n_boundary": 0,
                                   "n_at_cap": 13}}}
    w = sv.ladder_walk(_rec(pair), "none", "or_bench_hard_1k")
    assert w["complete"] is True and w["final_rung"] == 2048 and w["n_at_cap_final"] == 13 and w["truncated_at"] is None
    assert "exit_le_1pct" in w["exit"] and "residual 13" in w["exit"] and w["ladder_json_check"] == "agree"


def test_walk_still_reads_a_pair_that_is_escalating_as_truncated(monkeypatch):
    sv = _load()
    monkeypatch.setattr(sv, "read_store_control", _fake_reader({512: 154, 2048: 13}))
    pair = {"n_at_cap": {"512": 154, "2048": 13}, "rungs_attempted": [512, 2048],
            "left_ladder_at": {"status": "escalating"},
            "decisions": {"2048": {"decision": "escalate", "n_truncated": 13, "n_at_cap": 13, "next_cap": 8192}}}
    w = sv.ladder_walk(_rec(pair), "none", "or_bench_hard_1k")
    assert w["complete"] is False and w["truncated_at"] == 8192 and w["final_rung"] is None
    assert "never written" in w["why"]
