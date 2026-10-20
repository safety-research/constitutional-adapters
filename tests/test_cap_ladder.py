"""The standing truncation rule (tiers/cap_ladder), synthetic and offline:

* the 1% trigger at, just above and just below the threshold;
* the 16,384 ceiling reached with a non-zero residual, reported — and told
  apart from a cap that was AT the ceiling from the outset;
* a degenerate (decode-loop) cell that does NOT escalate and is reported as
  non-terminating, never as unresolved truncation;
* a boundary-draw-only cell that does NOT escalate (the other draw in hand
  terminated below the cap), while such items are still redrawn if the cell
  climbs;
* a share that falls below the threshold mid-ladder exits early, with its
  residual named among the threshold exits;
* the sizing function picking a rung from a measured (censored) p99, the npy
  round trip, and the drivers' --cap-sizing gate;
* the loop rule (the peer lane's rep_ratio, plus the tail window), the redraw
  record, the sidecar, the carry-over bundle's four-way partition and the
  served side's re-checks;
* (the served-driver end-to-end cases — the bundle carried through
  served_capability_run, its cap_ladder.json + manifest block, and the
  --cap-sizing gate — live with the driver integration on the fst-pair
  lineage, not in this ladder-only module.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import served_capability_run as srun  # noqa: E402  (the served driver the cap-ladder integration tests drive)

import cap_ladder as cli  # noqa: E402  (scripts/cap_ladder.py)

from sisteer.capeval.tiers import cap_ladder as cl  # noqa: E402
from sisteer.capeval.tiers import carry_over as cco  # noqa: E402
from sisteer.capeval.tiers import judged  # noqa: E402

from test_served_capability import (  # noqa: E402,F401  (the fixture is used by name)
    ORB_ROWS, REV, TT_CELL, XST_ROWS, _cell_dir, world)


class CountTok:
    """A whitespace tokenizer: one id per word; decode joins with single
    spaces (so single-spaced text round-trips and anything else does not).
    (Inlined from the evals lane's test_capctrl_cap2048.py, which did not
    travel to this tree: its carry-over bundle planner is not this tree's rung
    mechanism.)"""

    def __call__(self, text, add_special_tokens=False):
        if isinstance(text, list):
            return {"input_ids": [[1] * len(t.split()) for t in text]}
        return {"input_ids": [1] * len(text.split())}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join("w" for _ in ids)


def _header(rows, cap=64, system_sha=None, task="judged/xstest_v2"):
    return {"task": task, "task_version": judged.TASK_VERSION,
            "intervention_hash": "ab" * 8, "model_revision": REV,
            "gen_cfg": {"max_new_tokens": cap, "do_sample": False, "batch_size": 8},
            "system_sha": system_sha, "items_sha": judged.rows_content_sha(rows)}


def _spec(rows, cap=2048, system=None, task="judged/xstest_v2"):
    return judged.JudgedSpec(task=task, template={}, domain_of={}, rows=[dict(r) for r in rows],
                             order_seed=0, system=system, judge_model="claude-sonnet-5",
                             gen={"max_new_tokens": cap, "do_sample": False, "batch_size": 8})

TOK = CountTok()


def W(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


#: a decode loop: 7 words repeated -> 280 whitespace tokens, rep_ratio ~ 0.97
LOOP = " ".join(["a b c d e f g"] * 40)


def cell(n: int, *, n_trunc: int = 0, n_degen: int = 0, cap: int = 64,
         below: int = 10) -> dict:
    """A synthetic complete spool: ``n_trunc`` coherent texts past the cap,
    ``n_degen`` loops past the cap, the rest short terminated texts."""
    assert len(LOOP.split()) >= cap
    entries = {f"i{i:04d}": W(below) for i in range(n - n_trunc - n_degen)}
    entries.update({f"t{i:04d}": W(cap + 5) for i in range(n_trunc)})
    entries.update({f"d{i:04d}": LOOP for i in range(n_degen)})
    return entries


# --------------------------------------------------------------------------- #
# the trigger                                                                  #
# --------------------------------------------------------------------------- #


def test_trigger_at_just_above_and_just_below_the_threshold():
    n = 1000
    at, _ = cl.measure_rung(cell(n, n_trunc=10), TOK, 64, hop_timeouts_s={})          # exactly 1.00%
    above, _ = cl.measure_rung(cell(n, n_trunc=11), TOK, 64, hop_timeouts_s={})       # 1.10%
    below, _ = cl.measure_rung(cell(n, n_trunc=9), TOK, 64, hop_timeouts_s={})        # 0.90%
    assert at.share_truncated == pytest.approx(0.01)
    assert not at.escalate and at.exit == cl.EXIT_BELOW_THRESHOLD and at.n_truncated == 10
    assert at.next_cap is None
    assert above.escalate and above.exit == cl.ESCALATE and above.next_cap == 512
    assert not below.escalate and below.exit == cl.EXIT_BELOW_THRESHOLD and below.n_truncated == 9
    clean, _ = cl.measure_rung(cell(n), TOK, 64, hop_timeouts_s={})
    assert clean.exit == cl.EXIT_CLEAN and clean.n_at_cap == 0
    # the rung's rules travel with it
    assert at.rules["at_cap"] == "carry_over.AT_CAP_RULE"
    assert at.rules["degenerate"] == cl.DEGENERATE_RULE


def test_next_rung_doubles_to_the_ceiling_and_stops_there():
    assert cl.next_rung(512) == 1024 and cl.next_rung(2048) == 4096 and cl.next_rung(8192) == 16384
    assert cl.next_rung(16384) is None and cl.next_rung(20000) is None
    assert cl.next_rung(64) == 512            # off-ladder caps climb to the next rung above
    assert cl.next_rung(1000) == 1024
    assert cl.next_rung(4096, ceiling=4096) is None


# --------------------------------------------------------------------------- #
# the ceiling, two ways                                                        #
# --------------------------------------------------------------------------- #


def test_ceiling_reached_with_a_non_zero_residual_is_reported_and_told_from_the_outset_case():
    n = 100
    entries = {f"i{i:04d}": W(10) for i in range(n - 5)}
    entries.update({f"t{i:04d}": W(cl.CAP_CEILING + 3) for i in range(5)})   # 5% at the ceiling
    climbed, _ = cl.measure_rung(entries, TOK, cl.CAP_CEILING, climbed=True, hop_timeouts_s={})
    assert climbed.at_ceiling and not climbed.escalate and climbed.next_cap is None
    assert climbed.exit == cl.EXIT_CEILING and climbed.n_truncated == 5
    outset, _ = cl.measure_rung(entries, TOK, cl.CAP_CEILING, climbed=False, hop_timeouts_s={})
    assert outset.exit == cl.EXIT_CEILING_AT_OUTSET and outset.n_truncated == 5
    assert not outset.escalate and outset.next_cap is None
    # a configured cap ABOVE the ceiling is the same first-class case
    over, _ = cl.measure_rung({"a": W(40000), **{f"i{i}": W(3) for i in range(60)}}, TOK, 32768, hop_timeouts_s={})
    assert over.exit == cl.EXIT_CEILING_AT_OUTSET and over.at_ceiling
    # the report keeps the two apart
    rep = cl.ladder_report([
        {"cell_id": "c1", "control": "xstest", "cap": 4096,
         "status": cl.measure_rung(cell(100, n_trunc=20), TOK, 64, hop_timeouts_s={})[0].as_dict() | {"cap": 4096}},
        {"cell_id": "c1", "control": "xstest", "cap": cl.CAP_CEILING, "status": climbed.as_dict()},
        {"cell_id": "c2", "control": "inspect/math", "cap": cl.CAP_CEILING, "status": outset.as_dict()},
    ])
    s = rep["summary"]
    assert s["exits"] == {cl.EXIT_CEILING: 1, cl.EXIT_CEILING_AT_OUTSET: 1}
    assert [e["cell_id"] for e in s["ceiling_exits"]] == ["c1"]
    assert [e["cell_id"] for e in s["ceiling_at_outset"]] == ["c2"]
    assert s["ceiling_exits"][0]["residual_n"] == 5 and s["ceiling_at_outset"][0]["residual_n"] == 5
    assert rep["cells"]["c1|xstest"]["rungs_attempted"] == [4096, cl.CAP_CEILING]
    assert rep["cells"]["c2|inspect/math"]["rungs_attempted"] == [cl.CAP_CEILING]
    md = cl.render_report_md(rep)
    assert "climbed and ran out" in md and "from the outset" in md and "NOT an exhausted ladder" in md


# --------------------------------------------------------------------------- #
# degenerate items never climb                                                 #
# --------------------------------------------------------------------------- #


def test_rep_ratio_is_the_peer_lanes_loop_rule_and_the_tail_window_catches_late_onset_loops():
    assert cl.rep_ratio("one two three") == 0.0               # < 8 words -> 0
    assert cl.rep_ratio(W(300)) == 0.0                        # all distinct 4-grams
    assert cl.rep_ratio(LOOP) > 0.9
    assert cl.is_degenerate(LOOP) and not cl.is_degenerate(W(300))
    # a coherent body with a long loop tail: the whole-text ratio stays under
    # the threshold, the tail window trips it
    late = W(300) + " " + " ".join(["a b c d e f g"] * 30)
    d = cl.degeneracy(late)
    assert d["rep_ratio"] < cl.REP_RATIO_THRESHOLD < d["tail_rep_ratio"] and d["degenerate"]
    # a text shorter than the window: the two readings coincide
    assert cl.tail_rep_ratio(W(50)) == cl.rep_ratio(W(50))
    # the loop rule's threshold and n-gram order are the peer's
    assert cl.REP_RATIO_THRESHOLD == 0.5 and cl.REP_NGRAM == 4 and cl.REP_MIN_WORDS == 8


def test_degenerate_cell_does_not_escalate_and_is_reported_as_non_terminating():
    st, recs = cl.measure_rung(cell(100, n_degen=20), TOK, 64, hop_timeouts_s={})     # 20% at cap, all loops
    assert st.n_at_cap == 20 and st.share_at_cap == pytest.approx(0.20)
    assert st.n_non_terminating == 20 and st.n_truncated == 0 and st.share_truncated == 0
    assert not st.escalate and st.exit == cl.EXIT_CLEAN and st.next_cap is None
    assert all(recs[i].kind == cl.KIND_NON_TERMINATING and recs[i].at_cap and recs[i].degenerate
               for i in st.non_terminating_ids)
    # mixed: 20 loops + 5 truncated of 1000 -> the 5 alone decide (0.5%: no climb)
    mixed, _ = cl.measure_rung(cell(1000, n_trunc=5, n_degen=20), TOK, 64, hop_timeouts_s={})
    assert mixed.n_at_cap == 25 and mixed.n_truncated == 5 and not mixed.escalate
    # ... and 20 loops + 11 truncated -> the 11 decide (climb), loops stay out of the set
    climb, _ = cl.measure_rung(cell(1000, n_trunc=11, n_degen=20), TOK, 64, hop_timeouts_s={})
    assert climb.escalate and set(climb.truncated_ids) == {f"t{i:04d}" for i in range(11)}
    assert not (set(climb.truncated_ids) & set(climb.non_terminating_ids))
    rep = cl.ladder_report([{"cell_id": "c", "control": "xstest", "cap": 64, "status": st.as_dict()}])
    assert rep["summary"]["non_terminating_total"] == 20
    assert rep["summary"]["threshold_exits_with_residual"] == [] and rep["summary"]["exits"] == {cl.EXIT_CLEAN: 1}


def test_bundle_partition_carries_loops_as_non_terminating_and_redraws_only_truncated_items():
    entries = cell(100, n_trunc=20, n_degen=3)
    block = cco.build_task_block(control="xstest", task="judged/xstest_v2",
                                 header=_header(XST_ROWS, cap=64), entries=entries, tok=TOK,
                                 cap=64, parity_n=2, parity_seed=0, hop_timeouts_s={})
    assert block["rung"]["escalate"] and block["rung"]["exit"] == cl.ESCALATE
    assert set(block["regenerate"]) == {f"t{i:04d}" for i in range(20)}
    assert set(block["non_terminating"]) == {f"d{i:04d}" for i in range(3)}
    assert all(t == LOOP for t in block["non_terminating"].values())
    assert block["residual"] == {} and block["n_residual"] == 0 and not block["forced"]
    assert set(block["carried"]) | set(block["regenerate"]) | set(block["non_terminating"]) == set(entries)
    assert set(block["source_at_cap"]) == set(block["regenerate"])
    assert block["source_caps"] == {iid: 64 for iid in entries}
    assert set(block["parity_sample"]) <= set(block["carried"]) and len(block["parity_sample"]) == 2


# --------------------------------------------------------------------------- #
# boundary draws are out of the trigger                                        #
# --------------------------------------------------------------------------- #


def test_boundary_draw_only_cell_does_not_escalate():
    entries = cell(100, n_trunc=5)                                  # 5% at cap
    other = {f"t{i:04d}": [(64, W(20))] for i in range(5)}          # a second draw at the same cap, terminated
    st, recs = cl.measure_rung(entries, TOK, 64, other_draws=other, hop_timeouts_s={})
    assert st.n_at_cap == 5 and st.n_boundary == 5 and st.n_truncated == 0
    assert not st.escalate and st.exit == cl.EXIT_CLEAN
    assert all(recs[i].kind == cl.KIND_BOUNDARY and recs[i].boundary_from == 64 for i in st.boundary_ids)
    # without the other draws the same cell climbs
    plain, _ = cl.measure_rung(entries, TOK, 64, hop_timeouts_s={})
    assert plain.escalate and plain.n_truncated == 5
    # a terminated draw at a HIGHER cap that finished below the source cap counts ...
    hi, _ = cl.measure_rung(entries, TOK, 64, other_draws={"t0000": [(128, W(30))]}, hop_timeouts_s={})
    assert hi.n_boundary == 1 and hi.n_truncated == 4
    # ... one that finished at or past the source cap does not (real length) ...
    ext, _ = cl.measure_rung(entries, TOK, 64, other_draws={"t0000": [(128, W(70))]}, hop_timeouts_s={})
    assert ext.n_boundary == 0 and ext.n_truncated == 5
    # ... and neither does another AT-CAP draw (censored, not a termination)
    cap2, _ = cl.measure_rung(entries, TOK, 64, other_draws={"t0000": [(64, W(90))]}, hop_timeouts_s={})
    assert cap2.n_boundary == 0 and cap2.n_truncated == 5


def test_boundary_items_are_still_redrawn_when_the_cell_climbs_for_other_reasons():
    entries = cell(100, n_trunc=25)
    other = {f"t{i:04d}": [(64, W(20))] for i in range(5)}          # 5 boundary, 20 truncated -> climbs
    block = cco.build_task_block(control="xstest", task="judged/xstest_v2",
                                 header=_header(XST_ROWS, cap=64), entries=entries, tok=TOK,
                                 cap=64, parity_n=0, parity_seed=0, other_draws=other, hop_timeouts_s={})
    assert block["rung"]["n_boundary"] == 5 and block["rung"]["n_truncated"] == 20
    assert block["rung"]["share_truncated"] == pytest.approx(0.20) and block["rung"]["escalate"]
    assert set(block["regenerate"]) == {f"t{i:04d}" for i in range(25)}   # boundary ids included
    # when only the boundary items are at cap nothing climbs and they are the residual
    only = cell(100, n_trunc=5)
    b2 = cco.build_task_block(control="xstest", task="judged/xstest_v2",
                              header=_header(XST_ROWS, cap=64), entries=only, tok=TOK,
                              cap=64, parity_n=0, parity_seed=0, other_draws=other, hop_timeouts_s={})
    assert b2["regenerate"] == [] and set(b2["residual"]) == {f"t{i:04d}" for i in range(5)}
    assert b2["rung"]["exit"] == cl.EXIT_CLEAN


# --------------------------------------------------------------------------- #
# the ladder exits early when the share drops                                  #
# --------------------------------------------------------------------------- #


def test_share_falling_below_threshold_mid_ladder_exits_early_with_its_residual_named():
    # rung 1 (cap 64): 200 items, 40 truncated (20%) -> climb to 512
    r1_entries = cell(200, n_trunc=40)
    r1, _ = cl.measure_rung(r1_entries, TOK, 64, hop_timeouts_s={})
    assert r1.escalate and r1.next_cap == 512
    regen = sorted(r1.truncated_ids)
    # rung 2 (cap 512): the 40 redrawn — 37 terminate past the old cap, 2 hit 512
    # again, 1 lands below the old cap (boundary noise on the first draw)
    r2_entries = {iid: t for iid, t in r1_entries.items() if iid not in regen}
    r2_entries.update({iid: W(100) for iid in regen[:37]})
    r2_entries.update({iid: W(600) for iid in regen[37:39]})
    r2_entries[regen[39]] = W(30)
    r2, recs2 = cl.measure_rung(
        r2_entries, TOK, 512,
        source_caps={iid: 64 for iid in r2_entries if iid not in regen},
        regenerated=regen, source_texts={iid: r1_entries[iid] for iid in regen},
        source_caps_of_regenerated={iid: 64 for iid in regen}, climbed=True, hop_timeouts_s={})
    assert r2.n_truncated == 2 and r2.share_truncated == pytest.approx(0.01)
    assert not r2.escalate and r2.exit == cl.EXIT_BELOW_THRESHOLD and r2.next_cap is None
    assert r2.n_regenerated == 40
    assert r2.redraw_outcomes == {cl.REDRAW_BOUNDARY_LANDING: 1, cl.REDRAW_EXTENDED: 37,
                                  cl.REDRAW_STILL_AT_CAP: 2}
    assert r2.source_caps_seen == {"64": 160, "512": 40}
    # the redraw record per item: identity / prefix / fork RECORDED, never asserted
    still = recs2[regen[37]].redraw
    assert still["outcome"] == cl.REDRAW_STILL_AT_CAP and still["still_at_cap"] and still["prefix_extension"]
    landed = recs2[regen[39]].redraw
    assert landed["outcome"] == cl.REDRAW_BOUNDARY_LANDING and landed["boundary_landing"]
    assert landed["source_cap"] == 64 and landed["new_cap"] == 512 and landed["source_completion_tokens"] == 64
    assert tuple(still) == cl.REDRAW_KEYS
    # the report: two rungs, an early exit at 512 with residual 2, listed
    rep = cl.ladder_report([
        {"cell_id": "c", "control": "or_bench_hard_1k", "cap": 64, "status": r1.as_dict()},
        {"cell_id": "c", "control": "or_bench_hard_1k", "cap": 512, "status": r2.as_dict()},
    ])
    e = rep["cells"]["c|or_bench_hard_1k"]
    assert e["rungs_attempted"] == [64, 512] and e["exit"] == cl.EXIT_BELOW_THRESHOLD
    assert e["exit_cap"] == 512 and e["residual_n"] == 2 and e["climbed"]
    assert rep["summary"]["threshold_exits_with_residual"] == [
        {"cell_id": "c", "control": "or_bench_hard_1k", "exit_cap": 512, "residual_n": 2,
         "residual_share": 0.01, "non_terminating_n": 0}]
    assert rep["summary"]["incomplete"] == [] and rep["summary"]["ceiling_exits"] == []
    # a ladder whose top rung still says escalate is INCOMPLETE, not an exit
    rep1 = cl.ladder_report([{"cell_id": "c", "control": "or_bench_hard_1k", "cap": 64, "status": r1.as_dict()}])
    assert rep1["summary"]["incomplete"][0]["next_cap"] == 512 and rep1["summary"]["exits"] == {"incomplete": 1}
    with pytest.raises(cl.CapLadderError, match="two rung records"):
        cl.ladder_report([{"cell_id": "c", "control": "x", "cap": 64, "status": r1.as_dict()}] * 2)


def test_redraw_resolution_records_prefix_fork_and_divergence():
    ext = cl.resolve_redraw(W(64), W(80), TOK, source_cap=64, new_cap=128)
    assert ext["outcome"] == cl.REDRAW_EXTENDED and ext["prefix_extension"] and not ext["identical_text"]
    assert ext["first_divergence_char"] is None and ext["first_divergence_token"] is None
    fork = cl.resolve_redraw(W(64), "x " + W(40), TOK, source_cap=64, new_cap=128)
    assert fork["outcome"] == cl.REDRAW_BOUNDARY_LANDING and not fork["prefix_extension"]
    assert fork["first_divergence_char"] == 0 and fork["boundary_landing"]
    same = cl.resolve_redraw(W(70), W(70), TOK, source_cap=64, new_cap=128)
    assert same["identical_text"] and same["prefix_extension"] and same["outcome"] == cl.REDRAW_EXTENDED


# --------------------------------------------------------------------------- #
# sizing at the outset                                                         #
# --------------------------------------------------------------------------- #


def test_size_cap_picks_the_smallest_rung_under_the_threshold_from_a_measured_p99():
    comps = [100] * 985 + [700] * 10 + [1500] * 5              # measured at 4096: resolvable
    s = cl.size_cap(comps, 4096)
    assert not s.censored and s.p99_completion == 700
    assert s.shares_by_rung["512"] == pytest.approx(0.015) and s.shares_by_rung["1024"] == pytest.approx(0.005)
    assert s.recommended_cap == 1024 and not s.at_ceiling
    assert s.shares_by_rung["8192"] is None                      # above the measured cap: unreadable
    # everything short: the smallest rung
    small = cl.size_cap([50] * 400 + [300] * 4, 512)
    assert small.recommended_cap == 512 and not small.censored and small.p99_completion == 50
    # censored at 512 (30% at the cap): the next rung, flagged; no p99
    cen = cl.size_cap([100] * 700 + [512] * 300, 512)
    assert cen.censored and cen.p99_completion is None and cen.recommended_cap == 1024
    assert cen.share_at_measured_cap == pytest.approx(0.30) and "censored" in cen.reason
    # censored AT the ceiling: the ceiling is the recommendation, flagged at_ceiling
    top = cl.size_cap([10] * 90 + [cl.CAP_CEILING] * 10, cl.CAP_CEILING)
    assert top.censored and top.recommended_cap == cl.CAP_CEILING and top.at_ceiling
    with pytest.raises(cl.CapLadderError, match="not clamped"):
        cl.size_cap([600], 512)
    with pytest.raises(cl.CapLadderError, match="no completions"):
        cl.size_cap([], 512)


def test_size_cap_from_measurement_reads_the_npy_and_its_clamp_cap(tmp_path):
    arr = np.stack([np.full(1000, 40, dtype=np.int64),
                    np.array([100] * 700 + [512] * 300, dtype=np.int64)], axis=1)
    np.save(tmp_path / "cap__testfam__xstest_v2.npy", arr)
    (tmp_path / "cap__testfam__xstest_v2.meta.json").write_text(json.dumps(
        {"family": "testfam", "task": "xstest_v2", "note": "gens re-tokenized (+1 EOS, clamped 512)"}))
    s = cl.size_cap_from_measurement("testfam", "xstest_v2", raw_dir=tmp_path)
    assert s["measured_cap"] == 512 and s["censored"] and s["recommended_cap"] == 1024 and s["n"] == 1000
    assert s["family"] == "testfam" and s["task"] == "xstest_v2"
    # an explicit measured cap wins over the note
    s2 = cl.size_cap_from_measurement("testfam", "xstest_v2", raw_dir=tmp_path, measured_cap=1024)
    assert s2["measured_cap"] == 1024 and not s2["censored"] and s2["recommended_cap"] == 1024
    # a completion column that is a bound, not a distribution, is refused
    np.save(tmp_path / "cap__testfam__gsm8k_0shot.npy", arr)
    (tmp_path / "cap__testfam__gsm8k_0shot.meta.json").write_text(json.dumps(
        {"note": "completions NOT stored; ct set to the 512 cap (upper bound)"}))
    with pytest.raises(cl.CapLadderError, match="stored no completions"):
        cl.size_cap_from_measurement("testfam", "gsm8k_0shot", raw_dir=tmp_path)
    # no note and no explicit cap: refused, never guessed
    np.save(tmp_path / "cap__testfam__alpaca_pair.npy", arr)
    with pytest.raises(cl.CapLadderError, match="does not name the clamp cap"):
        cl.size_cap_from_measurement("testfam", "alpaca_pair", raw_dir=tmp_path)
    with pytest.raises(cl.CapLadderError, match="no measurement"):
        cl.size_cap_from_measurement("nofam", "xstest_v2", raw_dir=tmp_path)


def _sizing(rec_xst: int, rec_orb: int, *, censored: bool = True) -> dict:
    def entry(rec):
        return {"recommended_cap": rec, "censored": censored, "share_at_measured_cap": 0.3,
                "measured_cap": 512, "reason": f"synthetic: start at {rec}"}
    return {"schema": cl.SIZING_SCHEMA, "family": "testfam", "source": "/synthetic/raw",
            "threshold": 0.01, "tasks": {"xstest_v2": entry(rec_xst), "or_bench_hard_1k": entry(rec_orb)}}


def test_check_cap_against_sizing_refuses_an_undersized_cap_and_records_a_passing_one():
    sizing = _sizing(2048, 1024)
    with pytest.raises(cl.CapLadderError, match="cap 512 < recommended 2048"):
        cl.check_cap_against_sizing(512, sizing, controls=["xstest", "or_bench_hard_1k"])
    with pytest.raises(cl.CapLadderError, match="cap 1024 < recommended 2048"):
        cl.check_cap_against_sizing(1024, sizing, controls=["xstest", "or_bench_hard_1k"])
    ok = cl.check_cap_against_sizing(2048, sizing, controls=["xstest", "or_bench_hard_1k", "mmlu"])
    assert ok["cap"] == 2048 and set(ok["controls"]) == {"xstest", "or_bench_hard_1k"}
    assert ok["controls"]["xstest"] == {"task": "xstest_v2", "recommended_cap": 2048, "censored": True,
                                        "share_at_measured_cap": 0.3, "measured_cap": 512, "ok": True}
    # a judged control the sizing did not size is a refusal, not a pass; a
    # retired name (alpaca_eval, 2026-09-11) is no judged control and is skipped
    partial = {**sizing, "tasks": {"xstest_v2": sizing["tasks"]["xstest_v2"]}}
    with pytest.raises(cl.CapLadderError, match="no entry for 'or_bench_hard_1k'"):
        cl.check_cap_against_sizing(4096, partial, controls=["or_bench_hard_1k"])
    assert cl.check_cap_against_sizing(4096, sizing, controls=["alpaca_eval"])["controls"] == {}
    with pytest.raises(cl.CapLadderError, match="schema"):
        cl.check_cap_against_sizing(4096, {"tasks": {}}, controls=["xstest"])


def test_size_cli_reads_the_configured_cap_from_the_harness_not_a_copy(tmp_path, capsys):
    arr = np.stack([np.full(500, 40, dtype=np.int64),
                    np.array([100] * 400 + [512] * 100, dtype=np.int64)], axis=1)
    for task in ("xstest_v2", "or_bench_hard_1k"):
        np.save(tmp_path / f"cap__testfam__{task}.npy", arr)
        (tmp_path / f"cap__testfam__{task}.meta.json").write_text(json.dumps(
            {"note": "gens re-tokenized (+1 EOS, clamped 512)"}))
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("model: m\nattacks:\n  target_max_new_tokens: 512\ncompletion_tokens: 512\n")
    out = tmp_path / "sizing.json"
    cli.main(["size", "--family", "testfam", "--tasks", "xstest_v2,or_bench_hard_1k",
              "--raw-dir", str(tmp_path), "--config", str(cfg), "--out", str(out)])
    s = json.loads(out.read_text())
    assert s["schema"] == cl.SIZING_SCHEMA and set(s["tasks"]) == {"xstest_v2", "or_bench_hard_1k"}
    x = s["tasks"]["xstest_v2"]
    assert x["censored"] and x["recommended_cap"] == 1024
    assert x["configured"] == {"owner": "cell_controls.controls_gen_config(config)", "cap": 512}
    assert x["configured_ok"] is False and x["configured_at_ceiling"] is False
    assert "UNDER" in capsys.readouterr().out
    # an Inspect task's configured cap is read off inspect_served's table (which
    # reads the contracts constant) — the same value, never a literal here
    isv = pytest.importorskip("sisteer.capeval.tiers.inspect_served",
                              reason="no Inspect legs on this tree (tiny-capability lineage only)")
    tc = pytest.importorskip("sisteer.capeval.tinycap.contracts")
    conf = cli.configured_cap_for("inspect/math", cfg=None)
    assert conf["cap"] == isv.MAX_TOKENS_OFF["math"] == tc.MATH500_MAX_TOKENS_DEFAULT
    assert "inspect_served.MAX_TOKENS_OFF" in conf["owner"]
    assert cli.configured_cap_for("no_such_task", cfg=None) is None


# --------------------------------------------------------------------------- #
# the Inspect legs: measure / classify / report, at the ceiling from the outset #
# --------------------------------------------------------------------------- #


def _inspect_row(item, *, stop, content, cap, attempt=1, metric="acc", sample_idx=0):
    return {"task": "inspect/math", "item_id": item, "metric": metric, "value": 1.0,
            "extra_json": json.dumps({"stop_reason": stop, "content": content,
                                      "max_tokens_used": cap, "attempt": attempt,
                                      "completion_tokens": len(content.split()),
                                      "sample_idx": sample_idx})}


def test_inspect_rung_reads_the_rows_own_caps_and_files_ceiling_at_outset():
    rows = [_inspect_row(f"m{i}", stop="stop", content=W(20), cap=cl.CAP_CEILING) for i in range(96)]
    rows += [_inspect_row(f"t{i}", stop="max_tokens", content=W(500), cap=cl.CAP_CEILING) for i in range(3)]
    rows += [_inspect_row("d0", stop="max_tokens", content=LOOP, cap=cl.CAP_CEILING)]
    rows += [{"task": "inspect/math", "item_id": "__aggregate__", "metric": "acc_aggregate", "value": 0.9,
              "extra_json": json.dumps({"n": 100})}]
    st = cl.inspect_rung_from_rows(rows, task="inspect/math")
    assert st.cap == cl.CAP_CEILING and st.n == 100 and st.n_at_cap == 4
    assert st.n_truncated == 3 and st.n_non_terminating == 1
    assert st.exit == cl.EXIT_CEILING_AT_OUTSET and not st.escalate and not st.ladder_available
    assert not st.climbed and st.next_cap is None
    # the same rows at a cap below the ceiling: over threshold, but no rung
    # mechanism exists on this leg -> no_rung, with what the ladder would pick
    low = cl.inspect_rung_from_rows(
        [dict(r, extra_json=json.dumps({**json.loads(r["extra_json"]), "max_tokens_used": 4096}))
         if r["item_id"] != "__aggregate__" else r for r in rows], task="inspect/math")
    assert low.exit == cl.EXIT_NO_RUNG and low.next_cap == 8192 and not low.escalate
    # a retried sample reads at its retry cap; source caps record the mix
    mix = cl.inspect_rung_from_rows(
        rows[:96] + [_inspect_row("r0", stop="stop", content=W(30), cap=8192, attempt=2)],
        task="inspect/math")
    assert mix.source_caps_seen == {"8192": 1, "16384": 96} and mix.n_regenerated == 1
    # a row that says it stopped but not at what cap is refused, not guessed
    bad = [{"task": "inspect/math", "item_id": "x", "metric": "acc", "value": 1.0,
            "extra_json": json.dumps({"stop_reason": "stop", "content": "hi"})}]
    with pytest.raises(cl.CapLadderError, match="no max_tokens_used"):
        cl.inspect_rung_from_rows(bad, task="inspect/math")
    rep = cl.ladder_report([{"cell_id": "c", "control": "inspect/math", "cap": low.cap, "status": low.as_dict()}])
    assert rep["summary"]["no_rung"] == [{"cell_id": "c", "control": "inspect/math", "exit_cap": 4096,
                                          "residual_n": 3, "residual_share": 0.03, "non_terminating_n": 1,
                                          "would_climb_to": 8192}]
    assert "no rung mechanism" in cl.render_report_md(rep)


# --------------------------------------------------------------------------- #
# the sidecar                                                                  #
# --------------------------------------------------------------------------- #


def test_sidecar_round_trips_binds_by_text_sha_and_yields_source_caps(tmp_path):
    entries = cell(20, n_trunc=2, n_degen=1)
    st, recs = cl.measure_rung(entries, TOK, 64, hop_timeouts_s={})
    spool = tmp_path / "private" / "gens_xstest_v2.jsonl"
    side = cl.sidecar_path(spool)
    assert side.name == "gens_xstest_v2.caps.jsonl"
    cl.write_sidecar(side, recs, header={"cell_id": "c", "control": "xstest", "cap": 64})
    header, rows = cl.read_sidecar(side)
    assert header["schema"] == cl.RECORD_SCHEMA and header["row_keys"] == list(cl.ROW_KEYS)
    assert set(rows) == set(entries) and all(set(r) == set(cl.ROW_KEYS) for r in rows.values())
    assert not any("text" in r for r in rows.values())          # ids and numbers only
    caps, src = cl.source_caps_for(entries, side, 999)
    assert src == "sidecar" and caps == {iid: 64 for iid in entries}
    caps2, src2 = cl.source_caps_for(entries, tmp_path / "absent.caps.jsonl", 512)
    assert src2 == "header_cap" and set(caps2.values()) == {512}
    # a spooled text the sidecar does not describe refuses (stale / another draw)
    changed = dict(entries, i0000="something else entirely")
    with pytest.raises(cl.CapLadderError, match="no sidecar row matching"):
        cl.source_caps_for(changed, side, 64)
    with pytest.raises(cl.CapLadderError, match="not a .jsonl"):
        cl.sidecar_path(tmp_path / "gens.txt")


# --------------------------------------------------------------------------- #
# the bundle's served-side re-checks                                           #
# --------------------------------------------------------------------------- #


def _spool(tmp_path, cap=2048):
    return judged.GenSpool(tmp_path / "gens_xstest_v2.jsonl", _header(XST_ROWS, cap=cap))


def _block_for(entries, **kw):
    return cco.build_task_block(control="xstest", task="judged/xstest_v2",
                                header=_header(XST_ROWS, cap=64), entries=entries, tok=TOK,
                                cap=64, parity_n=0, parity_seed=0, **kw, hop_timeouts_s={})


def test_seed_spool_seeds_non_terminating_and_residual_texts_and_re_checks_them(tmp_path):
    ids = [r["item_id"] for r in XST_ROWS]
    assert len(ids) == 2
    # a loop at cap + a truncated item at cap: the loop is carried as-is, the
    # truncated item climbs (50% share)
    entries = {ids[0]: LOOP, ids[1]: W(70)}
    block = _block_for(entries)
    assert block["regenerate"] == [ids[1]] and set(block["non_terminating"]) == {ids[0]}
    spec = _spec(XST_ROWS)
    counts = cco.seed_spool(_spool(tmp_path), spec, block, tok=TOK, cap=64, new_cap=2048)
    assert counts == {"n_carried": 0, "n_seeded_now": 1, "n_already_present": 0,
                      "n_regenerate": 1, "n_non_terminating": 1, "n_residual": 0}
    assert _spool(tmp_path).entries() == {ids[0]: LOOP}
    # a 'non_terminating' text that is not a loop is refused
    fake = dict(block, non_terminating={ids[0]: W(70)}, source_caps=block["source_caps"])
    with pytest.raises(cco.CarryOverError, match="NOT degenerate"):
        cco.seed_spool(_spool(tmp_path / "b"), spec, fake, tok=TOK, cap=64, new_cap=2048)
    # a residual is carried only when the readout does not climb
    with_resid = dict(block, residual={ids[1]: W(70)}, regenerate=[])
    with pytest.raises(cco.CarryOverError, match="residual .* but also climbs"):
        cco.seed_spool(_spool(tmp_path / "c"), spec, with_resid, tok=TOK, cap=64, new_cap=2048)
    # a residual with a no-climb verdict seeds (the report names it)
    empty = dict(block, regenerate=[], residual={ids[1]: W(70)},
                 rung=dict(block["rung"], escalate=False))
    ok = cco.seed_spool(_spool(tmp_path / "d"), spec, empty, tok=TOK, cap=64, new_cap=2048)
    assert ok["n_residual"] == 1 and ok["n_seeded_now"] == 2
    # an escalate verdict with nothing to regenerate is an inconsistent plan
    loops_only = _block_for({ids[0]: LOOP, ids[1]: W(3)})
    assert loops_only["regenerate"] == [] and not loops_only["rung"]["escalate"]
    incons = dict(loops_only, rung=dict(loops_only["rung"], escalate=True))
    with pytest.raises(cco.CarryOverError, match="regenerates nothing"):
        cco.seed_spool(_spool(tmp_path / "e"), spec, incons, tok=TOK, cap=64, new_cap=2048)
    # the partition must be disjoint and complete
    overlap = dict(block, residual={ids[1]: W(70)}, rung=dict(block["rung"], escalate=False))
    with pytest.raises(cco.CarryOverError, match="overlaps"):
        cco.seed_spool(_spool(tmp_path / "f"), spec, overlap, tok=TOK, cap=64, new_cap=2048)


def test_block_carries_the_residual_when_the_readout_does_not_climb_and_per_item_source_caps():
    ids = [r["item_id"] for r in XST_ROWS]
    # 2 items, 1 at cap -> 50% share: would climb; force the no-climb path via a huge threshold
    entries = {ids[0]: W(3), ids[1]: W(70)}
    block = _block_for(entries, threshold=0.9)
    assert block["regenerate"] == [] and block["residual"] == {ids[1]: W(70)}
    assert block["rung"]["exit"] == cl.EXIT_BELOW_THRESHOLD and not block["rung"]["escalate"]
    # and --force redraws it anyway, recorded
    forced = _block_for(entries, threshold=0.9, force=True)
    assert forced["regenerate"] == [ids[1]] and forced["forced"] and forced["residual"] == {}
    # per-item source caps: a carried text keeps the rung that produced it
    block2 = _block_for({ids[0]: W(3), ids[1]: W(70)}, source_caps={ids[0]: 512, ids[1]: 64})
    assert block2["source_caps"] == {ids[0]: 512, ids[1]: 64}
    assert block2["rung"]["source_caps_seen"] == {"64": 1, "512": 1}
    # a 'climbed' source makes a ceiling verdict read as the ladder running out
    at_top = cco.build_task_block(control="xstest", task="judged/xstest_v2",
                                  header=_header(XST_ROWS, cap=64), entries=entries, tok=TOK,
                                  cap=64, parity_n=0, parity_seed=0, ceiling=64, climbed=True, hop_timeouts_s={})
    assert at_top["rung"]["exit"] == cl.EXIT_CEILING and at_top["residual"] == {ids[1]: W(70)}
    fresh = cco.build_task_block(control="xstest", task="judged/xstest_v2",
                                 header=_header(XST_ROWS, cap=64), entries=entries, tok=TOK,
                                 cap=64, parity_n=0, parity_seed=0, ceiling=64, hop_timeouts_s={})
    assert fresh["rung"]["exit"] == cl.EXIT_CEILING_AT_OUTSET


# --------------------------------------------------------------------------- #
# the served driver end to end                                                 #
# --------------------------------------------------------------------------- #


def test_served_generate_without_a_bundle_measures_every_judged_pass(world, tmp_path):
    srun.main(world.argv)
    cell_dir = _cell_dir(world.out_root)
    rec = json.loads((cell_dir / cl.RECORD_NAME).read_text())
    assert rec["climbed"] is False and rec["carry_over_bundle_sha256"] is None
    assert set(rec["controls"]) == {"xstest", "or_bench_hard_1k"}
    for c, st in rec["controls"].items():
        assert st["n_at_cap"] == 0 and st["exit"] == cl.EXIT_CLEAN and not st["climbed"]
        assert st["n_regenerated"] == 0 and st["source_caps_seen"] == {str(st["cap"]): st["n"]}
    assert (cell_dir / "private" / "gens_xstest_v2.caps.jsonl").is_file()
    assert (cell_dir / "private" / "gens_or_bench_hard_1k.caps.jsonl").is_file()
    manifest = json.loads((cell_dir / "manifest.json").read_text())
    assert manifest[cl.MANIFEST_KEY]["controls"] == rec["controls"]
    assert cl.MANIFEST_SIZING_KEY not in manifest          # no --cap-sizing given


def test_served_cap_sizing_gate_refuses_before_any_store_and_records_a_pass(world, tmp_path):
    sizing = tmp_path / "sizing.json"
    sizing.write_text(json.dumps(_sizing(2048, 1024)))
    with pytest.raises(SystemExit, match="under the measured recommendation.*xstest: cap 64 < recommended 2048"):
        srun.main(world.argv + ["--cap-sizing", str(sizing)])
    assert not (world.out_root / "cells").exists()
    # this tree's controls cap is the config's (the wave's rungs are the sampled run's
    # --controls-cap re-draws, not a --max-new-tokens override): a recommendation at
    # the world's cap passes and is recorded
    sizing.write_text(json.dumps(_sizing(64, 32)))
    srun.main(world.argv + ["--cap-sizing", str(sizing)])
    manifest = json.loads((_cell_dir(world.out_root) / "manifest.json").read_text())
    block = manifest[cl.MANIFEST_SIZING_KEY]
    assert block["cap"] == 64 and block["path"] == str(sizing)
    assert block["controls"]["xstest"]["ok"] and block["controls"]["or_bench_hard_1k"]["ok"]
    assert block["controls"]["xstest"]["recommended_cap"] == 64


def test_report_cli_walks_stores_and_writes_the_exit_report(world, tmp_path, capsys):
    srun.main(world.argv)
    out = tmp_path / "report.json"
    md = tmp_path / "report.md"
    cli.main(["report", str(world.out_root), "--out", str(out), "--md", str(md)])
    rep = json.loads(out.read_text())
    assert rep["n_records"] == 2 and rep["summary"]["n_pairs"] == 2
    assert rep["summary"]["exits"] == {cl.EXIT_CLEAN: 2}
    assert set(rep["cells"]) == {f"{TT_CELL}|xstest", f"{TT_CELL}|or_bench_hard_1k"}
    assert "threshold exits with a non-zero residual" in md.read_text()
    with pytest.raises(SystemExit, match="no cap_ladder.json"):
        cli.main(["report", str(tmp_path / "nowhere")])


def test_cell_controls_driver_exposes_the_shared_helpers_the_served_driver_reuses():
    import cell_controls_run as ccr

    for name in ("measure_cap_ladder", "write_cap_ladder", "cap_sizing_gate", "_judged_cap"):
        assert callable(getattr(ccr, name)), name
    ns = ccr.main.__globals__  # the module namespace, for the argparse seam
    assert "cap_sizing_gate" in ns
    p = srun.build_parser().parse_args(["generate", "--proxy-url", "u", "--routing-table", "t",
                                        "--cell", "c", "--cap-sizing", "/x.json"])
    assert p.cap_sizing == "/x.json"


def _ladder_bundle(tmp_path, *, xst_entries, orb_entries, cap=64, new_cap=2048) -> Path:
    tasks = {
        "xstest": cco.build_task_block(control="xstest", task="judged/xstest_v2",
                                       header=_header(XST_ROWS, cap=cap), entries=xst_entries,
                                       tok=TOK, cap=cap, parity_n=0, parity_seed=0, hop_timeouts_s={}),
        "or_bench_hard_1k": cco.build_task_block(control="or_bench_hard_1k",
                                                 task="judged/or_bench_hard_1k",
                                                 header=_header(ORB_ROWS, cap=cap,
                                                                task="judged/or_bench_hard_1k"),
                                                 entries=orb_entries, tok=TOK, cap=cap,
                                                 parity_n=0, parity_seed=0, hop_timeouts_s={}),
    }
    b = {"schema": cco.BUNDLE_SCHEMA, "family": "testfam", "cell_id": TT_CELL,
         "model_id": "meta-llama/Llama-3.1-8B-Instruct", "model_revision": REV,
         "cap": cap, "new_cap": new_cap, "at_cap_rule": cco.AT_CAP_RULE,
         "parity": {"n": 0, "seed": 0},
         "source": {"store_dir": "/synthetic/hf/store", "engine": "hf",
                    "intervention_hash": "ab" * 8, "harness_config_hash": "cd" * 8,
                    "artifact_sha256": None, "judge_model": "claude-sonnet-5",
                    "judge_backend": "anthropic", "limit": None,
                    "gen_cfg": {"max_new_tokens": cap, "do_sample": False}},
         "tasks": tasks}
    p = tmp_path / "bundle.json"
    cco.write_bundle(p, b)
    return p


# --------------------------------------------------------------------------- #
# sizing counts truncation, not presence at the cap                            #
# --------------------------------------------------------------------------- #


def test_size_cap_sets_aside_at_cap_rows_that_are_not_truncation():
    # GLM-shaped: 18% of the rows sit at 8192, every one a decode loop, no truncation
    vals = [100] * 820 + [8192] * 180
    s = cl.size_cap(vals, 8192, n_at_cap_not_truncated=180)
    assert not s.censored and s.recommended_cap == 512 and s.p99_completion == 100
    assert s.n_at_cap == 180 and s.n_at_cap_not_truncated == 180 and s.degeneracy_classified
    assert s.share_at_measured_cap == pytest.approx(0.18)
    assert s.share_truncated_at_measured_cap == 0.0
    # the loops leave the numerator at EVERY rung, not just the measured cap
    assert s.shares_by_rung["512"] == 0.0 and s.shares_by_rung["4096"] == 0.0 and s.shares_by_rung["8192"] == 0.0
    assert "180 of the 180 at-cap rows" in s.reason
    # the same counts, unclassified: every at-cap row counts, the answer is an upper bound and says so
    u = cl.size_cap(vals, 8192)
    assert u.censored and u.recommended_cap == 16384 and not u.degeneracy_classified
    assert u.n_at_cap_not_truncated is None and u.share_truncated_at_measured_cap == pytest.approx(0.18)
    assert "UPPER BOUND" in u.reason
    # exactly at the threshold is not censored; one more truncated row is
    at = cl.size_cap(vals, 8192, n_at_cap_not_truncated=170)      # 10 / 1000 truncated
    assert not at.censored and at.recommended_cap == 512 and at.shares_by_rung["512"] == pytest.approx(0.01)
    over = cl.size_cap(vals, 8192, n_at_cap_not_truncated=169)    # 11 / 1000 truncated
    assert over.censored and over.recommended_cap == 16384 and not over.at_ceiling
    assert over.share_truncated_at_measured_cap == pytest.approx(0.011)
    # the exclusion is bounded by the at-cap count, both ways
    with pytest.raises(cl.CapLadderError, match=r"must lie in \[0, 180\]"):
        cl.size_cap(vals, 8192, n_at_cap_not_truncated=181)
    with pytest.raises(cl.CapLadderError, match=r"must lie in \[0, 180\]"):
        cl.size_cap(vals, 8192, n_at_cap_not_truncated=-1)
    # classified with nothing to exclude reads exactly like the old rule
    z = cl.size_cap([100] * 700 + [512] * 300, 512, n_at_cap_not_truncated=0)
    assert z.censored and z.recommended_cap == 1024 and z.degeneracy_classified


def _length_quantiles(tmp_path, lane="lanefam", readout="xstest"):
    """A synthetic length_quantiles/1 report + items file in the producer's
    shape: one uncensored cell, one whose residual is all loops, one whose
    residual is undetermined, one with a pending trial, one torn."""
    rows, entries = [], {}
    cap = 8192

    def trial(n, at_cap, *, status, n_truncated=0, final_cap=cap):
        rc = None
        if status != "uncensored":
            rc = {"n_at_cap": at_cap, "n_boundary": 0, "n_loop": at_cap - n_truncated, "n_repetitive": 0,
                  "n_truncated": n_truncated, "share_at_cap": at_cap / n, "share_truncated": n_truncated / n}
        return {"final_cap": final_cap, "n": n, "n_censored_at_final_cap": at_cap,
                "censoring": {"status": status, "decision": "exit_le_1pct" if status != "uncensored" else None,
                              "decision_path": "/synthetic/decision.json", "residual_composition": rc},
                "quantiles": {"n": n}, "by_source_cap": {str(final_cap): n}, "rungs_present": [final_cap]}

    def emit(cell, values, at_cap_flags, t, final_cap=cap):
        for j, (v, a) in enumerate(zip(values, at_cap_flags)):
            rows.append({"lane": lane, "readout": readout, "cell": cell, "kind": "controls", "trial": t,
                         "item": f"i{j}", "completion_tokens": v, "at_cap": a, "source_cap": final_cap,
                         "final_cap": final_cap})

    def entry(cell, trials, status="ok", incomplete=()):
        entries[f"{lane}|{readout}|{cell}"] = {
            "cell": cell, "kind": "controls", "lane": lane, "readout": readout, "status": status,
            "incomplete_trials": list(incomplete), "trials": trials,
            "n_censored_pooled": sum(t["n_censored_at_final_cap"] for t in trials.values()),
            "pooled": {"n": sum(t["n"] for t in trials.values())}}

    # none: 440 short + 10 at 700, both trials, nothing at cap -> 2.2% >= 512, 0% >= 1024
    for t in (0, 1):
        emit("none", [100] * 440 + [700] * 10, [False] * 450, t)
    entry("none", {"0": trial(450, 0, status="uncensored"), "1": trial(450, 0, status="uncensored")})
    # loopy: 370 short + 80 at the cap, all loops -> sizes to 512 once the loops are set aside
    for t in (0, 1):
        emit("loopy", [100] * 370 + [cap] * 80, [False] * 370 + [True] * 80, t)
    entry("loopy", {"0": trial(450, 80, status="residual_threshold_exit"),
                    "1": trial(450, 80, status="residual_threshold_exit")})
    # undet: at cap, no reclassify decision on record
    emit("undet", [100] * 440 + [cap] * 10, [False] * 440 + [True] * 10, 0)
    entry("undet", {"0": trial(450, 10, status="residual_undetermined")})
    # pend: one trial's store pending
    emit("pend", [100] * 450, [False] * 450, 0)
    entry("pend", {"0": trial(450, 0, status="uncensored")}, status="incomplete", incomplete=["1"])
    # torn: the report says 5 at cap, the items file has 6
    emit("torn", [100] * 444 + [cap] * 6, [False] * 444 + [True] * 6, 0)
    entry("torn", {"0": trial(450, 5, status="residual_threshold_exit")})
    # mixed: the trials climbed to different final caps (trial 1 needs 1024, trial 0 needs 512)
    emit("mixed", [100] * 450, [False] * 450, 0, final_cap=2048)
    emit("mixed", [100] * 440 + [700] * 10, [False] * 450, 1, final_cap=8192)
    entry("mixed", {"0": trial(450, 0, status="uncensored", final_cap=2048),
                    "1": trial(450, 0, status="uncensored", final_cap=8192)})

    items = tmp_path / "length_quantiles.items.jsonl"
    items.write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = tmp_path / "length_quantiles.json"
    report.write_text(json.dumps({
        "name": "length_quantiles", "ceiling": 16384, "items_file": str(items),
        "schema": {"version": 1, "censoring.status": {"residual_undetermined": "no reclassify decision"}},
        "tool": {"name": "analysis_sampled.py", "git_sha": "deadbeef", "sha256": "00", "written_at": "2026-09-10"},
        "entries": entries}))
    return report


def test_size_cap_from_length_quantiles_reads_the_residual_the_way_the_ladder_classified_it(tmp_path):
    rep = _length_quantiles(tmp_path)
    assert cl.cells_in_length_quantiles(rep, "lanefam", "xstest") == ["loopy", "mixed", "none", "pend", "torn", "undet"]
    none = cl.size_cap_from_length_quantiles(rep, "lanefam", "xstest", "none")
    assert none["n"] == 900 and none["measured_cap"] == 8192 and not none["censored"]
    assert none["recommended_cap"] == 1024 and none["trials"] == ["0", "1"] and none["degeneracy_classified"]
    assert none["n_at_cap"] == 0 and none["n_at_cap_not_truncated"] == 0
    assert none["producer"]["git_sha"] == "deadbeef" and none["censoring"]["0"]["status"] == "uncensored"
    # the loops leave the count: 80 + 80 at cap, none truncated -> 512, not the ceiling
    loopy = cl.size_cap_from_length_quantiles(rep, "lanefam", "xstest", "loopy")
    assert loopy["n_at_cap"] == 160 and loopy["n_at_cap_not_truncated"] == 160
    assert not loopy["censored"] and loopy["recommended_cap"] == 512
    assert loopy["share_at_measured_cap"] == pytest.approx(160 / 900, abs=1e-6)
    assert loopy["share_truncated_at_measured_cap"] == 0.0
    assert loopy["censoring"]["1"]["decision"] == "exit_le_1pct"
    # the same rows, fed as counts only, would have sized to the ceiling
    raw = cl.size_cap([100] * 740 + [8192] * 160, 8192)
    assert raw.censored and raw.recommended_cap == 16384
    # one trial
    one = cl.size_cap_from_length_quantiles(rep, "lanefam", "xstest", "loopy", trial="0")
    assert one["n"] == 450 and one["trials"] == ["0"] and one["recommended_cap"] == 512
    # refusals: unknown cell, a residual without a decision, a pending trial, a torn snapshot, mixed caps
    with pytest.raises(cl.CapLadderError, match="no entry"):
        cl.size_cap_from_length_quantiles(rep, "lanefam", "xstest", "nope")
    with pytest.raises(cl.CapLadderError, match="residual_undetermined.*not sizeable"):
        cl.size_cap_from_length_quantiles(rep, "lanefam", "xstest", "undet")
    with pytest.raises(cl.CapLadderError, match="incomplete"):
        cl.size_cap_from_length_quantiles(rep, "lanefam", "xstest", "pend")
    with pytest.raises(cl.CapLadderError, match="torn snapshot"):
        cl.size_cap_from_length_quantiles(rep, "lanefam", "xstest", "torn")
    # trials at different final caps: each sized at its own cap, the cell takes the worst
    mixed = cl.size_cap_from_length_quantiles(rep, "lanefam", "xstest", "mixed")
    assert mixed["pooled"] is False and mixed["sized_on_trial"] == "1" and mixed["trials"] == ["0", "1"]
    assert mixed["measured_cap"] == 8192 and mixed["recommended_cap"] == 1024 and mixed["n"] == 450
    assert mixed["per_trial"]["0"]["recommended_cap"] == 512 and mixed["per_trial"]["0"]["measured_cap"] == 2048
    assert mixed["per_trial"]["1"]["recommended_cap"] == 1024
    assert set(mixed["censoring"]) == {"0", "1"}
    m0 = cl.size_cap_from_length_quantiles(rep, "lanefam", "xstest", "mixed", trial="0")
    assert m0["measured_cap"] == 2048 and m0["recommended_cap"] == 512 and m0["pooled"] is False
    assert none["pooled"] is True and none["sized_on_trial"] is None and none["per_trial"] is None
    with pytest.raises(cl.CapLadderError, match="has trials"):
        cl.size_cap_from_length_quantiles(rep, "lanefam", "xstest", "none", trial="7")
    # not a report
    (tmp_path / "other.json").write_text(json.dumps({"name": "x"}))
    with pytest.raises(cl.CapLadderError, match="not a length_quantiles/1"):
        cl.size_cap_from_length_quantiles(tmp_path / "other.json", "lanefam", "xstest", "none")
    with pytest.raises(cl.CapLadderError, match="does not exist"):
        cl.size_cap_from_length_quantiles(tmp_path / "missing.json", "lanefam", "xstest", "none")


def test_size_cli_from_length_quantiles_gives_the_task_its_worst_cell(tmp_path, capsys):
    rep = _length_quantiles(tmp_path)
    out = tmp_path / "sizing.json"
    cli.main(["size", "--family", "lanefam", "--tasks", "xstest_v2", "--from-length-quantiles", str(rep),
              "--cells", "none,loopy", "--out", str(out)])
    s = json.loads(out.read_text())
    assert s["schema"] == cl.SIZING_SCHEMA and s["source_kind"] == "length_quantiles" and s["lane"] == "lanefam"
    x = s["tasks"]["xstest_v2"]
    assert x["task"] == "xstest_v2" and x["readout"] == "xstest"
    assert x["sized_on_cell"] == "none" and x["recommended_cap"] == 1024 and not x["censored"]
    assert set(x["per_cell"]) == {"none", "loopy"}
    assert x["per_cell"]["loopy"]["recommended_cap"] == 512 and x["per_cell"]["loopy"]["n_at_cap_not_truncated"] == 160
    # the sizing file still passes the drivers' gate
    ok = cl.check_cap_against_sizing(1024, s, controls=["xstest"])
    assert ok["controls"]["xstest"]["recommended_cap"] == 1024
    with pytest.raises(cl.CapLadderError, match="cap 512 < recommended 1024"):
        cl.check_cap_against_sizing(512, s, controls=["xstest"])
    printed = capsys.readouterr().out
    assert "worst cell none of 2" in printed and "truncated 0.00%" in printed
    # a task the report has no cells for is refused, not skipped; so is a task that is no judged
    # control's measured task, and a cell that is not there
    with pytest.raises(SystemExit, match="no cells for lanefam|or_bench_hard_1k"):
        cli.main(["size", "--family", "lanefam", "--tasks", "or_bench_hard_1k", "--from-length-quantiles", str(rep)])
    with pytest.raises(SystemExit, match="not a judged control"):
        cli.main(["size", "--family", "lanefam", "--tasks", "inspect/math", "--from-length-quantiles", str(rep)])
    with pytest.raises(SystemExit, match="no entry"):
        cli.main(["size", "--family", "lanefam", "--tasks", "xstest_v2", "--from-length-quantiles", str(rep),
                  "--cells", "ghost"])
    with pytest.raises(SystemExit, match="not sizeable"):
        cli.main(["size", "--family", "lanefam", "--tasks", "xstest_v2", "--from-length-quantiles", str(rep),
                  "--cells", "undet"])
    # every cell present is the default, and one unsizeable cell refuses the task rather than dropping out
    with pytest.raises(SystemExit, match="not sizeable|incomplete|torn"):
        cli.main(["size", "--family", "lanefam", "--tasks", "xstest_v2", "--from-length-quantiles", str(rep)])
    assert x["per_cell"]["none"]["pooled"] is True and x["per_cell"]["none"]["sized_on_trial"] is None


# --------------------------------------------------------------------------- #
# the producer's ledger: a client timeout cannot hide truncation                #
# --------------------------------------------------------------------------- #


def _ledger_rows(vmodel, cap, *, served_terminated, served_at_cap, rejected_transport=0, task=None,
                 other_cap=0):
    rows = []
    for _ in range(served_terminated):
        rows.append({"vmodel": vmodel, "max_tokens_requested": cap, "rejected": False,
                     "finish_reason": "stop", "usage_completion_tokens": 40, "task": task})
    for _ in range(served_at_cap):
        rows.append({"vmodel": vmodel, "max_tokens_requested": cap, "rejected": False,
                     "finish_reason": "length", "usage_completion_tokens": cap, "task": task})
    for _ in range(rejected_transport):
        rows.append({"vmodel": vmodel, "max_tokens_requested": cap, "rejected": True,
                     "finish_reason": None, "error_class": "transport", "task": task})
    for _ in range(other_cap):
        rows.append({"vmodel": vmodel, "max_tokens_requested": cap * 2, "rejected": False,
                     "finish_reason": "length", "task": task})
    return rows


def test_ledger_reconciliation_turns_a_clean_reading_into_ledger_censored(tmp_path):
    """The peer lane's incident: a client timeout below the server's completion
    time abandoned the slowest draws while the server finished them; the scored
    spool held ZERO at-cap draws where the ledger held hundreds. Read from the
    spool alone the 1 % trigger never fires; held to the ledger the rung exits
    ledger_censored — unresolved, never below_threshold."""
    cap = 64
    spool = {f"i{k}": W(10) for k in range(300)}                       # every scored draw terminated
    counts = cl.ledger_at_cap_counts(
        _ledger_rows("vm-cell", cap, served_terminated=300, served_at_cap=20, rejected_transport=20),
        vmodel="vm-cell", cap=cap)
    assert counts == {"selector": {"vmodel": "vm-cell", "cap": cap, "task": None}, "n_rows": 340,
                      "n_served": 320, "n_at_cap": 20, "n_rejected": 20, "n_transport": 20}
    st, _ = cl.measure_rung(spool, TOK, cap, hop_timeouts_s={}, ledger_counts=counts)
    assert st.exit == cl.EXIT_LEDGER_CENSORED and not st.escalate
    assert st.exit in cl.UNRESOLVED_EXITS and st.exit in cl.EXITS
    lg = st.ledger
    assert lg["censored"] is True and lg["censored_at_cap"] == 20 and lg["spool_at_cap"] == 0
    assert lg["extra_completions"] == 20 and lg["n_transport"] == 20 and lg["spool_n"] == 300
    assert st.rules["ledger"] == cl.LEDGER_RULE
    # the same spool read WITHOUT a ledger: clean, and marked unreconciled
    bare, _ = cl.measure_rung(spool, TOK, cap, hop_timeouts_s={})
    assert bare.exit == cl.EXIT_CLEAN and bare.ledger is None
    # the report names the censored pair and counts the unreconciled ones
    rep = cl.ladder_report([
        {"cell_id": "c", "control": "xstest", "cap": cap, "status": st.as_dict()},
        {"cell_id": "d", "control": "xstest", "cap": cap, "status": bare.as_dict()},
    ])
    s = rep["summary"]
    assert s["exits"] == {cl.EXIT_LEDGER_CENSORED: 1, cl.EXIT_CLEAN: 1} and s["n_unreconciled"] == 1
    assert s["ledger_censored"] == [{"cell_id": "c", "control": "xstest", "exit_cap": cap, "residual_n": 0,
                                     "residual_share": 0.0, "non_terminating_n": 0, "ledger_at_cap": 20,
                                     "spool_at_cap": 0, "censored_at_cap": 20, "extra_completions": 20,
                                     "n_transport": 20}]
    md = cl.render_report_md(rep)
    assert "censored by the transport" in md and "ledger at cap 20 vs spool at cap 0" in md
    assert "UNRECONCILED: 1 of 2" in md


def test_ledger_reconciliation_that_agrees_keeps_the_exit_and_records_the_evidence():
    cap = 64
    spool = {**{f"t{k}": W(10) for k in range(90)}, **{f"a{k}": W(80) for k in range(10)}}   # 10 % at cap, truncated
    counts = cl.ledger_at_cap_counts(_ledger_rows("vm", cap, served_terminated=90, served_at_cap=10),
                                     vmodel="vm", cap=cap)
    st, _ = cl.measure_rung(spool, TOK, cap, hop_timeouts_s={}, ledger_counts=counts)
    assert st.exit == cl.ESCALATE and st.escalate and st.next_cap == 512   # the first rung above 64
    assert st.ledger["censored"] is False and st.ledger["censored_at_cap"] == 0 and st.ledger["n_at_cap"] == 10
    # fewer at-cap rows in the ledger than the spool (a spool assembled from two ledgers, say) is
    # not censoring: nothing the producer finished is missing from the scored set
    fewer = cl.ledger_at_cap_counts(_ledger_rows("vm", cap, served_terminated=95, served_at_cap=5),
                                    vmodel="vm", cap=cap)
    st2 = cl.reconcile_with_ledger(st, fewer)
    assert st2.exit == cl.ESCALATE and st2.ledger["censored"] is False


def test_ledger_selector_reads_only_this_route_at_this_cap_and_refuses_an_empty_match(tmp_path):
    rows = (_ledger_rows("vm-a", 64, served_terminated=5, served_at_cap=1, other_cap=3, task="judged/x")
            + _ledger_rows("vm-b", 64, served_terminated=2, served_at_cap=2))
    c = cl.ledger_at_cap_counts(rows, vmodel="vm-a", cap=64)
    assert (c["n_rows"], c["n_served"], c["n_at_cap"]) == (6, 6, 1)     # the cap-128 rows are not this rung's
    assert cl.ledger_at_cap_counts(rows, vmodel="vm-a", cap=64, task="judged/x")["n_at_cap"] == 1
    assert cl.ledger_at_cap_counts(rows, vmodel="vm-b", cap=64)["n_at_cap"] == 2
    with pytest.raises(cl.CapLadderError, match="no row for vmodel 'vm-a' at max_tokens_requested 512"):
        cl.ledger_at_cap_counts(rows, vmodel="vm-a", cap=512)
    with pytest.raises(cl.CapLadderError, match="with task 'judged/y'"):
        cl.ledger_at_cap_counts(rows, vmodel="vm-a", cap=64, task="judged/y")
    # the file reader: JSONL, refusing a row without the columns the rule reads
    p = tmp_path / "usage_ledger.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert len(cl.read_usage_ledger(p)) == len(rows)
    p.write_text(json.dumps({"vmodel": "vm-a", "rejected": False}) + "\n")
    with pytest.raises(cl.CapLadderError, match="lacks"):
        cl.read_usage_ledger(p)
    with pytest.raises(cl.CapLadderError, match="does not exist"):
        cl.read_usage_ledger(tmp_path / "nope.jsonl")
