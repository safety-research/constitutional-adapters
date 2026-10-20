"""Offline tests for scripts/sdfbase/full22/sampled/ladder_decide.py — the per-(cell, readout)
escalation decision of the SAMPLED cap ladder (LADDER_RULES.md). Everything SYNTHETIC: invented
texts, goal ids g0.., item ids it0.., placeholder cell names; no GPU, no network, no API.

Covered:
  * the classifier: loop by rep_ratio (>= 0.5, incl. the EXACT tie at 0.5 the artifact's strict
    rule would not call), loop by the degeneracy detector alone, the repetitive band
    (0.2 <= rr < 0.5), cap_coherent (rr < 0.2, not degenerate);
  * the boundary rule against a sibling row (terminated at <= 0.5 x cap => boundary; above it /
    at cap / unreadable / no sibling => not);
  * the trigger at EXACTLY 1% (<= 1% exits; above escalates), the ceiling exit with its residual;
  * end to end on a static store: the decision record, the disposition file (one row per at-cap
    item; escalate iff truncated), a sibling present / absent / incomplete (no boundary calls, a
    valid decision, never a refusal), a rung store at the ceiling, the refusals (greedy store,
    torn store, wrong --cap, off-ladder cap, missing rep_ratio source) -> exit 2;
  * end to end on a controls store (at_cap_controls' own reader) with and without a sibling;
  * the disposition file reader's schema refusals;
  * --share on at_cap_static.py / at_cap_controls.py (the bare integer stays the default).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SAMPLED_DIR = REPO / "scripts" / "sdfbase" / "full22" / "sampled"
sys.path.insert(0, str(SAMPLED_DIR))

import at_cap_controls as acc  # noqa: E402
import at_cap_static as acs  # noqa: E402
import ladder_decide as ld  # noqa: E402

REP_SRC = ('def rep_ratio(t):\n    w = t.split()\n    if len(w) < 8:\n        return 0.0\n'
           '    g = [tuple(w[i:i + 4]) for i in range(len(w) - 3)]\n    return 1 - len(set(g)) / len(g)\n')

# texts by class (rep_ratio worked out by hand; every one is < 200 chars and one line, so the
# degeneracy detector stays quiet unless a test wants it)
LOOP_RR = "a b c d " * 6                     # 24 words, 21 4-grams, 4 unique -> 0.81
LOOP_TIE = "a b c d a b c d a b c d e"       # 13 words, 10 4-grams, 5 unique -> EXACTLY 0.5
LOOP_DEG = "x1\nx2\nx3\nsame\nsame\nx6\nsame\nx8"  # 8 lines, one line 3/8 -> line_loop; rr 0.0
REPETITIVE = "a b c d e f g h a b c d e f g h"  # 16 words, 13 4-grams, 8 unique -> 5/13 = 0.385
COHERENT = "one two three four five six seven eight nine ten eleven twelve"  # 0.0
SHORT = "just a few words"                  # < 8 words -> 0.0

PRESET = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
ARM = "vendor_recommended"
SEED_BASE = 20260909


@pytest.fixture()
def rep_src(tmp_path):
    p = tmp_path / "join_scores.py"
    p.write_text("import json\n\n\n" + REP_SRC + "\n\nX = 1\n")
    return p


@pytest.fixture()
def classifier(rep_src):
    rep_ratio, info = ld.extract_rep_ratio(rep_src)
    degeneracy, dinfo = ld.load_degeneracy()
    return rep_ratio, degeneracy, info, dinfo


# --------------------------------------------------------------------------- #
# the classifier + the boundary rule + the trigger                             #
# --------------------------------------------------------------------------- #


def test_rep_ratio_is_extracted_by_ast_with_provenance(rep_src, classifier):
    rep_ratio, _, info, dinfo = classifier
    assert info["rep_ratio_source"] == REP_SRC.rstrip("\n")
    assert info["source_sha256"] == hashlib.sha256(rep_src.read_bytes()).hexdigest()
    assert info["rep_ratio_source_sha256"] == hashlib.sha256(REP_SRC.rstrip("\n").encode()).hexdigest()
    assert rep_ratio(LOOP_RR) == pytest.approx(1 - 4 / 21)
    assert rep_ratio(LOOP_TIE) == 0.5
    assert rep_ratio(REPETITIVE) == pytest.approx(5 / 13)
    assert rep_ratio(COHERENT) == 0.0 and rep_ratio(SHORT) == 0.0
    assert dinfo["source"].endswith("degeneracy.py") and dinfo["rules"]["line"]["min_lines"] == 8
    with pytest.raises(ld.Refusal, match="absent"):
        ld.extract_rep_ratio(rep_src.with_name("nope.py"))
    two = rep_src.with_name("two.py")
    two.write_text(REP_SRC + "\n" + REP_SRC)
    with pytest.raises(ld.Refusal, match="exactly one"):
        ld.extract_rep_ratio(two)


def test_classify_text_four_ways(classifier):
    rep_ratio, degeneracy, _, _ = classifier
    c = lambda t: ld.classify_text(t, rep_ratio, degeneracy)  # noqa: E731
    rr = c(LOOP_RR)
    assert rr["class"] == "loop" and rr["loop_by"] == "rep_ratio" and rr["degenerate"] is False
    assert rr["loop_window"] == "both"  # shorter than the window: the two readings coincide
    assert rr["rep_ratio"] == rr["tail_rep_ratio"] == rr["rep_ratio_max"]
    # the STRICT operator (the peer's cap_ladder + the artifact): EXACTLY 0.5 is NOT a loop
    tie = c(LOOP_TIE)
    assert tie["class"] == "repetitive" and tie["loop_by"] is None and tie["rep_ratio_at_tie"] is True
    assert tie["rep_ratio_max"] == 0.5 and tie["loop_window"] is None
    deg = c(LOOP_DEG)
    assert deg["class"] == "loop" and deg["loop_by"] == "degeneracy" and deg["loop_window"] is None
    assert deg["degenerate_reason"] == "line_loop" and deg["rep_ratio"] == 0.0
    both = c("\n".join([LOOP_RR.strip()] * 8) + "\n" + LOOP_RR * 4)
    assert both["class"] == "loop" and both["loop_by"] == "both"
    rep = c(REPETITIVE)
    assert rep["class"] == "repetitive" and rep["loop_by"] is None and rep["degenerate"] is False
    assert 0.2 <= rep["rep_ratio_max"] <= 0.5
    coh = c(COHERENT)
    assert coh["class"] == "cap_coherent" and coh["rep_ratio_max"] == 0.0 and coh["loop_by"] is None
    assert c(SHORT)["class"] == "cap_coherent"
    assert "STRICT" in ld.LOOP_RULE_NOTE and ld.ARTIFACT_LOOP_RULE == "rep_ratio(text) > 0.5"
    assert ld.CLASSIFIER["source_commit"] == "146d11acd56218e4ac05d70ea684cc3c9d302f1b"
    assert ld.CLASSIFIER["window"] == "whole_or_last_200_words" and ld.CLASSIFIER["loop_operator"] == ">"
    assert ld.CLASSIFIER["version"] == ld.RULE == "item_filtered_v1"


def test_tail_window_catches_a_late_onset_loop_the_whole_text_dilutes(classifier):
    """The peer's wide window: 300 distinct words then a loop that dominates the last 200
    words — the whole-text ratio stays under 0.5, the tail reading crosses it. The loop's
    inter-cycle spacing grows strictly (1, 2, 3, ... spaces), so no 60-char window of the text
    repeats and degeneracy's exact-substring tail rule does NOT fire (a periodic spacing of
    period 13 did: three repeats of a 3-cycle window fit in 38 cycles): the tail rep_ratio alone
    makes the call — whitespace runs are invisible to the word statistic."""
    rep_ratio, degeneracy, _, _ = classifier
    head = " ".join(f"w{i}" for i in range(300))
    loop = "".join("a b c d" + " " * (1 + i) for i in range(38))
    text = head + " " + loop
    assert ld.tail_rep_ratio(text, rep_ratio) > 0.5 > rep_ratio(text)
    c = ld.classify_text(text, rep_ratio, degeneracy)
    assert c["class"] == "loop" and c["loop_by"] == "rep_ratio" and c["loop_window"] == "tail"
    assert c["degenerate"] is False and c["rep_ratio"] < 0.5 < c["tail_rep_ratio"] == c["rep_ratio_max"]
    # a text shorter than the window reads identically under both windows
    assert ld.tail_rep_ratio(LOOP_RR, rep_ratio) == rep_ratio(LOOP_RR)
    assert ld.tail_rep_ratio(head, rep_ratio, words=50) == rep_ratio(" ".join(f"w{i}" for i in range(250, 300)))
    # the same text under a wider window than its tail loop is only repetitive
    wide = ld.classify_text(text, rep_ratio, degeneracy, tail_words=10_000)
    assert wide["class"] == "repetitive" and wide["loop_window"] is None


def test_boundary_call_against_the_sibling_row():
    cap = 4096
    b = ld.boundary_call({"finish_reason": "stop", "completion_tokens": 2048, "source_cap": 4096}, cap=cap)
    assert b["boundary"] is True and b["sibling"]["threshold_tokens"] == 2048.0
    assert ld.boundary_call({"finish_reason": "stop", "completion_tokens": 2049}, cap=cap)["boundary"] is False
    assert ld.boundary_call({"finish_reason": "length", "completion_tokens": 100}, cap=cap)["boundary"] is False
    unread = ld.boundary_call({"finish_reason": "stop", "completion_tokens": None}, cap=cap)
    assert unread["boundary"] is False and unread["sibling"]["completion_tokens"] is None
    assert ld.boundary_call(None, cap=cap) == {"boundary": False, "sibling": None}
    # the parameter is a parameter
    assert ld.boundary_call({"finish_reason": "stop", "completion_tokens": 1000}, cap=cap,
                            fraction=0.2)["boundary"] is False
    assert ld.boundary_call({"finish_reason": "stop", "completion_tokens": 800}, cap=cap,
                            fraction=0.2)["boundary"] is True


def _counts(n_resp, n_trunc):
    return {"n_responses": n_resp, "n_truncated": n_trunc, "share_truncated": n_trunc / n_resp}


def test_trigger_at_exactly_one_percent_and_the_ceiling():
    ladder = ld.LADDERS["controls"]
    assert ld.decide(_counts(100, 1), cap=512, ladder=ladder)["decision"] == "exit_le_1pct"   # == 1% exits
    d = ld.decide(_counts(100, 2), cap=512, ladder=ladder)
    assert d == {"decision": "escalate", "next_cap": 2048, "at_ceiling": False}
    assert ld.decide(_counts(200, 2), cap=2048, ladder=ladder)["decision"] == "exit_le_1pct"
    assert ld.decide(_counts(200, 3), cap=8192, ladder=ladder)["next_cap"] == 16384
    top = ld.decide(_counts(100, 5), cap=16384, ladder=ladder)
    assert top == {"decision": "exit_ceiling", "next_cap": None, "at_ceiling": True}
    assert ld.decide(_counts(100, 1), cap=16384, ladder=ladder)["decision"] == "exit_le_1pct"
    assert ld.decide(_counts(100, 0), cap=4096, ladder=ld.LADDERS["static"])["decision"] == "exit_le_1pct"
    assert ld.LADDERS == {"static": (4096, 8192, 16384), "controls": (512, 2048, 8192, 16384)}
    assert ld.CEILING == 16384 and ld.TRIGGER_SHARE == 0.01 and ld.BOUNDARY_FRACTION == 0.5
    assert ld.LOOP_REP_RATIO_THRESHOLD == 0.5 and ld.REPETITIVE_REP_RATIO_MIN == 0.2 and ld.TAIL_WORDS == 200


# --------------------------------------------------------------------------- #
# static stores                                                                #
# --------------------------------------------------------------------------- #

N_GOALS = 100  # 2 samples each -> 200 responses


def _srow(gid, idx, text, fr, *, trial=0, cap=4096, ct=None, **over):
    block = {"do_sample": True, **PRESET, "cap": cap, "arm": ARM, "seed_base": SEED_BASE, "trial": trial}
    r = {"goal_id": gid, "sample_idx": idx, "trial": trial, "variant_key": f"{idx}:{trial}",
         "family": "pasttense", "attack_prompt": f"{gid}/{idx}", "response": text, "decode": block,
         "sampling_seed": 7, "finish_reason": fr,
         "completion_tokens": (cap if fr == "length" else 10) if ct is None else ct,
         "prompt_tokens": 5, "max_tokens": cap, "at_cap": fr == "length"}
    r.update(over)
    return r


def _write_static(root, run, rows_by_shard, *, escalation=None):
    for i, rows in enumerate(rows_by_shard):
        d = root / run / "pasttense" / "none" / f"shard{i:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "responses.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        goals = {}
        for r in rows:
            goals[r["goal_id"]] = goals.get(r["goal_id"], 0) + 1
        (d / "progress.jsonl").write_text("".join(
            json.dumps({"goal_id": g, "family": "pasttense", "n_variants": n, "outcome": "done"}) + "\n"
            for g, n in goals.items()))
        if escalation is not None:
            (d / "escalation.json").write_text(json.dumps(escalation))


#: the at-cap items of the main store, by class
AT_CAP = {("g0", 0): COHERENT, ("g1", 0): COHERENT, ("g2", 0): COHERENT,   # truncated (coherent)
          ("g3", 0): COHERENT,                                             # boundary via the sibling
          ("g4", 0): LOOP_RR, ("g5", 0): LOOP_DEG, ("g6", 0): REPETITIVE}


def _base_rows(trial=0, cap=4096):
    rows = []
    for g in range(N_GOALS):
        for idx in (0, 1):
            key = (f"g{g}", idx)
            if key in AT_CAP:
                rows.append(_srow(*key, AT_CAP[key], "length", trial=trial, cap=cap))
            else:
                rows.append(_srow(*key, f"fine answer {g} {idx}", "stop", trial=trial, cap=cap))
    return [rows[:100], rows[100:]]


def _sibling_rows(cap=4096):
    """Trial 1: g3/0 terminated at 100 tokens (a boundary call for our g3/0); g4/0 is at cap
    there too; every OTHER item at cap in trial 0 terminated LONG in trial 1 (3000 > 0.5 x 4096:
    not boundary); the items terminated in trial 0 terminated short here (irrelevant: the rule
    only looks at OUR at-cap items)."""
    rows = []
    for g in range(N_GOALS):
        for idx in (0, 1):
            key = (f"g{g}", idx)
            if key == ("g3", 0):
                rows.append(_srow("g3", 0, "sibling short", "stop", trial=1, cap=cap, ct=100))
            elif key == ("g4", 0):
                rows.append(_srow("g4", 0, LOOP_RR, "length", trial=1, cap=cap))
            elif key in AT_CAP:
                rows.append(_srow(*key, "sibling long", "stop", trial=1, cap=cap, ct=3000))
            else:
                rows.append(_srow(*key, f"fine {g} {idx}", "stop", trial=1, cap=cap, ct=50))
    return [rows]


def _static_argv(root, rep_src, out, *extra, run="run_t0", cap=4096):
    return ["static", "--private-root", str(root), "--private-run", run, "--attack", "pasttense",
            "--defense", "none", "--cap", str(cap), "--rep-ratio-source", str(rep_src),
            "--out", str(out), *extra]


def _items(path):
    lines = [json.loads(l) for l in Path(path).read_text().splitlines()]
    return lines[0][ld.ITEMS_HEADER_KEY], lines[1:]


def test_static_decision_with_sibling_escalates_only_truncated(tmp_path, rep_src, capsys):
    _write_static(tmp_path, "run_t0", _base_rows())
    _write_static(tmp_path, "run_t1", _sibling_rows())
    out, items = tmp_path / "d.json", tmp_path / "items.jsonl"
    rc = ld.main(_static_argv(tmp_path, rep_src, out, "--sibling-private-run", "run_t1",
                              "--items-out", str(items)))
    assert rc == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    rec = json.loads(out.read_text())
    assert line["decision"] == rec["decision"] == "escalate" and line["n_truncated"] == 3
    assert "rep_ratio_source" not in line["rep_ratio"] and "rep_ratio_source" in rec["rep_ratio"]
    c = rec["counts"]
    assert c["n_responses"] == 200 and c["n_at_cap"] == 7
    assert c["n_loop"] == 2 and c["n_loop_by_rule"] == {"rep_ratio": 1, "degeneracy": 1, "both": 0}
    assert c["n_repetitive"] == 1 and c["n_boundary"] == 1 and c["n_truncated"] == 3
    assert c["n_boundary_any"] == 1 and c["share_truncated"] == pytest.approx(0.015)
    assert c["n_loop"] + c["n_repetitive"] + c["n_boundary"] + c["n_truncated"] == c["n_at_cap"]
    assert rec["next_cap"] == 8192 and rec["at_ceiling"] is False and rec["residual"] is None
    assert rec["sibling"]["status"] == "used" and rec["sibling"]["private_run"] == "run_t1"
    assert rec["sibling"]["trial"] == 1
    assert rec["rules"]["boundary_fraction"] == 0.5 and rec["rules"]["trigger_share"] == 0.01
    assert rec["rep_ratio"]["source_sha256"] == hashlib.sha256(rep_src.read_bytes()).hexdigest()
    assert rec["degeneracy"]["source_sha256"]
    assert rec["tool"]["sha256"] == hashlib.sha256((SAMPLED_DIR / "ladder_decide.py").read_bytes()).hexdigest()
    assert rec["identity"]["trial"] == 0 and rec["identity"]["is_rung"] is False
    assert rec["items_out_sha256"] == hashlib.sha256(items.read_bytes()).hexdigest()
    assert rec["n_items_escalate"] == 3
    # the rule + classifier stamp (an extend pass asserts them; one classifier covered the run)
    assert rec["rule"] == "item_filtered_v1" and rec["classifier"] == ld.CLASSIFIER
    assert rec["rules"]["loop_rep_ratio_threshold"] == 0.5 and rec["rules"]["tail_words"] == 200
    assert c["n_loop_by_window"] == {"whole": 0, "tail": 0, "both": 1} and c["n_rep_ratio_at_tie"] == 0
    # the disposition file: one row per AT-CAP item, escalate iff truncated
    header, rows = _items(items)
    assert header["rule"] == "item_filtered_v1" and header["classifier"] == ld.CLASSIFIER
    assert header["schema"] == ld.ITEMS_SCHEMA == 2
    assert all({"tail_rep_ratio", "rep_ratio_max", "loop_window"} <= set(r) for r in rows)
    assert header["kind"] == "static" and header["cap"] == 4096 and header["private_run"] == "run_t0"
    assert header["attack"] == "pasttense" and header["defense"] == "none" and header["trial"] == 0
    assert header["n_at_cap"] == 7 and header["n_escalate"] == 3 and header["decision"] == "escalate"
    by = {(r["goal_id"], r["sample_idx"]): r for r in rows}
    assert set(by) == set(AT_CAP) and all(r["trial"] == 0 for r in rows)
    assert {k for k, r in by.items() if r["escalate"]} == {("g0", 0), ("g1", 0), ("g2", 0)}
    assert by[("g3", 0)]["excluded_as"] == "boundary" and by[("g3", 0)]["class"] == "cap_coherent"
    assert by[("g3", 0)]["sibling"] == {"finish_reason": "stop", "completion_tokens": 100,
                                        "source_cap": 4096, "threshold_tokens": 2048.0}
    assert by[("g0", 0)]["boundary"] is False and by[("g0", 0)]["sibling"]["completion_tokens"] == 3000
    assert by[("g4", 0)]["excluded_as"] == "loop" and by[("g4", 0)]["loop_by"] == "rep_ratio"
    assert by[("g5", 0)]["excluded_as"] == "loop" and by[("g5", 0)]["loop_by"] == "degeneracy"
    assert by[("g6", 0)]["excluded_as"] == "repetitive"
    assert all(r["excluded_as"] is None for r in rows if r["escalate"])
    # the reader round-trips it
    h2, r2 = ld.read_items_file(items)
    assert h2 == header and r2 == rows


def test_static_decision_without_or_with_unavailable_sibling_calls_no_boundary(tmp_path, rep_src, capsys):
    _write_static(tmp_path, "run_t0", _base_rows())
    out = tmp_path / "d.json"
    # (a) no sibling named
    assert ld.main(_static_argv(tmp_path, rep_src, out)) == 0
    rec = json.loads(out.read_text())
    assert rec["sibling"] == {"status": "not_given"}
    assert rec["counts"]["n_boundary"] == 0 and rec["counts"]["n_truncated"] == 4
    assert rec["counts"]["share_truncated"] == pytest.approx(0.02) and rec["decision"] == "escalate"
    # (b) a sibling named but ABSENT: still a valid decision, no boundary, the reason recorded
    assert ld.main(_static_argv(tmp_path, rep_src, out, "--sibling-private-run", "run_t9")) == 0
    rec = json.loads(out.read_text())
    assert rec["sibling"]["status"] == "unavailable" and "absent or incomplete" in rec["sibling"]["reason"]
    assert rec["counts"]["n_boundary"] == 0 and rec["counts"]["n_truncated"] == 4
    # (c) a sibling named but INCOMPLETE (a torn final line): unavailable, not a refusal
    _write_static(tmp_path, "run_t1", _sibling_rows())
    p = tmp_path / "run_t1" / "pasttense" / "none" / "shard00" / "responses.jsonl"
    p.write_bytes(p.read_bytes() + b'{"goal_id": "torn')
    assert ld.main(_static_argv(tmp_path, rep_src, out, "--sibling-private-run", "run_t1")) == 0
    rec = json.loads(out.read_text())
    assert rec["sibling"]["status"] == "unavailable" and "torn" in rec["sibling"]["reason"]
    assert rec["counts"]["n_boundary"] == 0 and rec["counts"]["n_truncated"] == 4
    # (d) a sibling with a different item set: unavailable
    _write_static(tmp_path, "run_t2", [_sibling_rows()[0][:-2]])
    assert ld.main(_static_argv(tmp_path, rep_src, out, "--sibling-private-run", "run_t2")) == 0
    assert "item set differs" in json.loads(out.read_text())["sibling"]["reason"]
    capsys.readouterr()


def test_static_exit_at_one_percent_and_at_the_ceiling(tmp_path, rep_src, capsys):
    # exactly 1%: 2 truncated of 200 -> exit_le_1pct with the residual composition
    rows = _base_rows()
    flat = rows[0] + rows[1]
    for r in flat:
        if (r["goal_id"], r["sample_idx"]) in {("g1", 0), ("g2", 0)}:
            r.update(response="fine", finish_reason="stop", completion_tokens=10, at_cap=False)
    _write_static(tmp_path, "run_t0", [flat[:100], flat[100:]])
    out = tmp_path / "d.json"
    assert ld.main(_static_argv(tmp_path, rep_src, out)) == 0
    rec = json.loads(out.read_text())
    assert rec["counts"]["n_truncated"] == 2 and rec["counts"]["share_truncated"] == pytest.approx(0.01)
    assert rec["decision"] == "exit_le_1pct" and rec["next_cap"] is None
    assert rec["residual"] == {"n_at_cap": 5, "n_loop": 2, "n_loop_by_rule": {"rep_ratio": 1, "degeneracy": 1, "both": 0},
                               "n_loop_by_window": {"whole": 0, "tail": 0, "both": 1},
                               "n_repetitive": 1, "n_boundary": 0, "n_truncated": 2,
                               "share_truncated": pytest.approx(0.01), "share_at_cap": pytest.approx(0.025)}
    # the ceiling: a rung store at 16384 (escalation.json cap_to) with 3 truncated -> exit_ceiling
    esc = {"cap_from": 8192, "cap_to": 16384, "n_rows": 200}
    rows16 = _base_rows(cap=16384)
    _write_static(tmp_path, "run_t0_cap16384", [rows16[0] + rows16[1]], escalation=esc)
    assert ld.main(_static_argv(tmp_path, rep_src, out, run="run_t0_cap16384", cap=16384)) == 0
    rec = json.loads(out.read_text())
    assert rec["decision"] == "exit_ceiling" and rec["at_ceiling"] is True and rec["next_cap"] is None
    assert rec["identity"]["is_rung"] is True and rec["residual"]["n_truncated"] == 4
    # a rung store whose rows were ALL copied forward (decode.cap 8192) still reads cap_to
    rows8 = _base_rows(cap=8192)
    _write_static(tmp_path, "run_t0_cap16384b", [rows8[0] + rows8[1]], escalation=esc)
    assert ld.main(_static_argv(tmp_path, rep_src, out, run="run_t0_cap16384b", cap=16384)) == 0
    capsys.readouterr()


def test_static_refusals_exit_2(tmp_path, rep_src, capsys):
    out = tmp_path / "d.json"
    # a greedy store (no finish_reason / decode)
    greedy = [[{k: v for k, v in r.items() if k not in ("finish_reason", "decode", "at_cap")}
               for r in _base_rows()[0]]]
    _write_static(tmp_path, "greedy", greedy)
    with pytest.raises(SystemExit) as ei:
        ld.main(_static_argv(tmp_path, rep_src, out, run="greedy"))
    assert ei.value.code == 2 and "GREEDY" in capsys.readouterr().err
    # a torn store
    _write_static(tmp_path, "torn", _base_rows())
    p = tmp_path / "torn" / "pasttense" / "none" / "shard01" / "responses.jsonl"
    p.write_bytes(p.read_bytes() + b'{"goal_id": "x"')
    with pytest.raises(SystemExit) as ei:
        ld.main(_static_argv(tmp_path, rep_src, out, run="torn"))
    assert ei.value.code == 2 and "torn" in capsys.readouterr().err
    # a missing store
    with pytest.raises(SystemExit) as ei:
        ld.main(_static_argv(tmp_path, rep_src, out, run="absent"))
    assert ei.value.code == 2
    capsys.readouterr()
    # --cap not the store's cap; --cap off the ladder
    _write_static(tmp_path, "run_t0", _base_rows())
    with pytest.raises(SystemExit) as ei:
        ld.main(_static_argv(tmp_path, rep_src, out, cap=8192))
    assert ei.value.code == 2 and "not --cap 8192" in capsys.readouterr().err
    with pytest.raises(SystemExit) as ei:
        ld.main(_static_argv(tmp_path, rep_src, out, cap=5000))
    assert ei.value.code == 2 and "not a rung" in capsys.readouterr().err
    # an absent rep_ratio source
    with pytest.raises(SystemExit) as ei:
        ld.main(_static_argv(tmp_path, rep_src.with_name("gone.py"), out))
    assert ei.value.code == 2 and "rep_ratio source absent" in capsys.readouterr().err
    # a short shard (rows vs progress)
    short = _base_rows()
    short[0] = short[0][:-1]
    _write_static(tmp_path, "short", short)
    d = tmp_path / "short" / "pasttense" / "none" / "shard00"
    prog = [json.loads(l) for l in (d / "progress.jsonl").read_text().splitlines()]
    prog[-1]["n_variants"] = 2
    (d / "progress.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prog))
    with pytest.raises(SystemExit) as ei:
        ld.main(_static_argv(tmp_path, rep_src, out, run="short"))
    assert ei.value.code == 2 and "short shard" in capsys.readouterr().err
    assert not out.exists()


# --------------------------------------------------------------------------- #
# controls stores                                                              #
# --------------------------------------------------------------------------- #

CTL = "xstest"
SPOOL = "gens_xstest_v2.jsonl"


def _write_controls(root, cell, items, *, cap=512, arm=ARM):
    """A complete sampled served-controls store for CELL under ROOT: items = [(item_id, text,
    finish_reason, completion_tokens)]."""
    d = root / "cells" / f"{cell}__controls__abc"
    (d / "private").mkdir(parents=True, exist_ok=True)
    gen = {"max_new_tokens": cap, "do_sample": True, **PRESET, "arm": arm, "seed_base": SEED_BASE,
           "trial": 0, "batch_size": 8}
    (d / "manifest.json").write_text(json.dumps({"cell_id": cell, "intervention_hash": "ii" * 8,
                                                 "resolved_config": {"driver": "served_capability_run_v2",
                                                                     "gen": gen}}))
    with (d / "private" / SPOOL).open("w") as f:
        f.write(json.dumps({"__spool_header__": {"gen_cfg": gen}}) + "\n")
        for iid, text, _, _ in items:
            f.write(json.dumps({"item_id": iid, "text": text}) + "\n")
    with (d / "private" / (SPOOL[:-6] + ".meta.jsonl")).open("w") as f:
        for iid, text, fr, ct in items:
            f.write(json.dumps({"item_id": iid, "sampling_seed": 1, "max_tokens": cap, "finish_reason": fr,
                                "completion_tokens": ct, "prompt_tokens": 9, "at_cap": fr == "length",
                                "regenerated": False, "floor_sample": False, "source_cap": cap,
                                "text_sha256": hashlib.sha256(text.encode()).hexdigest()}) + "\n")
    at_cap = sorted(i for i, _, fr, _ in items if fr == "length")
    (d / "gen_stats.json").write_text(json.dumps({CTL: {
        "control": CTL, "n_total": len(items), "spool": str(d / "private" / SPOOL), "cap": cap,
        "decode": {"do_sample": True, **PRESET, "cap": cap, "arm": arm, "seed_base": SEED_BASE, "trial": 0},
        "n_at_cap": len(at_cap), "at_cap_item_ids": at_cap}}))
    return d


def _ctl_items(n=100):
    out = []
    for i in range(n):
        iid = f"it{i:04d}"
        if i == 0:
            out.append((iid, LOOP_RR, "length", 512))
        elif i == 1:
            out.append((iid, REPETITIVE, "length", 512))
        elif i in (2, 3, 4):
            out.append((iid, COHERENT, "length", 512))
        else:
            out.append((iid, f"terminated reply {i}", "stop", 40))
    return out


def _ctl_argv(root, rep_src, out, *extra, cap=512):
    return ["controls", "--out-root", str(root), "--cell", "cellA", "--control", CTL, "--cap", str(cap),
            "--rep-ratio-source", str(rep_src), "--out", str(out), *extra]


def test_controls_decision_with_and_without_sibling(tmp_path, rep_src, capsys):
    root, sib = tmp_path / "cap512_t0", tmp_path / "cap512_t1"
    _write_controls(root, "cellA", _ctl_items())
    sib_items = [(iid, "sib", "stop", 100) if iid == "it0002" else (iid, t, fr, ct)
                 for iid, t, fr, ct in _ctl_items()]
    _write_controls(sib, "cellA", sib_items)
    out, items = tmp_path / "d.json", tmp_path / "items.jsonl"
    assert ld.main(_ctl_argv(root, rep_src, out, "--sibling-out-root", str(sib), "--items-out", str(items))) == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    rec = json.loads(out.read_text())
    c = rec["counts"]
    assert c["n_responses"] == 100 and c["n_at_cap"] == 5 and c["n_loop"] == 1 and c["n_repetitive"] == 1
    assert c["n_boundary"] == 1 and c["n_truncated"] == 2 and c["share_truncated"] == pytest.approx(0.02)
    assert rec["decision"] == line["decision"] == "escalate" and rec["next_cap"] == 2048
    assert rec["sibling"]["status"] == "used" and rec["sibling"]["out_root"] == str(sib)
    assert rec["identity"]["cell"] == "cellA" and rec["identity"]["control"] == CTL
    header, rows = _items(items)
    assert header["kind"] == "controls" and header["cap"] == 512 and header["cell"] == "cellA"
    assert header["control"] == CTL and header["out_root"] == str(root)
    by = {r["item_id"]: r for r in rows}
    assert set(by) == {"it0000", "it0001", "it0002", "it0003", "it0004"}
    assert all(r["control"] == CTL for r in rows)
    assert by["it0000"]["excluded_as"] == "loop" and by["it0001"]["excluded_as"] == "repetitive"
    assert by["it0002"]["excluded_as"] == "boundary" and by["it0002"]["sibling"]["completion_tokens"] == 100
    assert by["it0003"]["escalate"] is True and by["it0004"]["escalate"] is True
    # no sibling: it0002 is truncated too -> 3/100
    assert ld.main(_ctl_argv(root, rep_src, out)) == 0
    rec = json.loads(out.read_text())
    assert rec["sibling"] == {"status": "not_given"} and rec["counts"]["n_truncated"] == 3
    # a sibling root with no store for the cell: unavailable, not a refusal
    (tmp_path / "empty" / "cells").mkdir(parents=True)
    assert ld.main(_ctl_argv(root, rep_src, out, "--sibling-out-root", str(tmp_path / "empty"))) == 0
    rec = json.loads(out.read_text())
    assert rec["sibling"]["status"] == "unavailable" and rec["counts"]["n_truncated"] == 3
    # an INCOMPLETE sibling (a spooled item without its meta row): unavailable
    meta = sib / "cells" / "cellA__controls__abc" / "private" / (SPOOL[:-6] + ".meta.jsonl")
    lines = meta.read_text().splitlines()
    meta.write_text("".join(l + "\n" for l in lines[:-1]))
    assert ld.main(_ctl_argv(root, rep_src, out, "--sibling-out-root", str(sib))) == 0
    rec = json.loads(out.read_text())
    assert rec["sibling"]["status"] == "unavailable" and "no bound meta" in rec["sibling"]["reason"]
    capsys.readouterr()


def test_controls_refusals_and_share_flags(tmp_path, rep_src, capsys):
    root = tmp_path / "cap512_t0"
    d = _write_controls(root, "cellA", _ctl_items())
    out = tmp_path / "d.json"
    # --cap not the store's
    with pytest.raises(SystemExit) as ei:
        ld.main(_ctl_argv(root, rep_src, out, cap=2048))
    assert ei.value.code == 2 and "not --cap 2048" in capsys.readouterr().err
    # a greedy store (no decode / meta) refuses
    gs = json.loads((d / "gen_stats.json").read_text())
    del gs[CTL]["decode"]
    saved = (d / "gen_stats.json").read_text()
    (d / "gen_stats.json").write_text(json.dumps(gs))
    with pytest.raises(SystemExit) as ei:
        ld.main(_ctl_argv(root, rep_src, out))
    assert ei.value.code == 2 and "GREEDY" in capsys.readouterr().err
    (d / "gen_stats.json").write_text(saved)
    # --share on both at-cap tools; the bare integer stays the default
    assert acc.main(["--out-root", str(root), "--cell", "cellA", "--control", CTL]) == 0
    assert capsys.readouterr().out.strip() == "5"
    assert acc.main(["--out-root", str(root), "--cell", "cellA", "--control", CTL, "--share"]) == 0
    assert json.loads(capsys.readouterr().out.strip()) == {"control": CTL, "n_at_cap": 5, "n": 100,
                                                            "cap": 512, "share": 0.05}
    _write_static(tmp_path, "run_t0", _base_rows())
    argv = ["--private-root", str(tmp_path), "--private-run", "run_t0", "--attack", "pasttense",
            "--defense", "none"]
    assert acs.main(argv) == 0 and capsys.readouterr().out.strip() == "7"
    assert acs.main(argv + ["--share"]) == 0
    assert json.loads(capsys.readouterr().out.strip()) == {"n_at_cap": 7, "n_rows": 200, "share": 0.035}


def test_items_file_reader_refusals(tmp_path):
    p = tmp_path / "items.jsonl"
    hdr = {ld.ITEMS_HEADER_KEY: {"schema": ld.ITEMS_SCHEMA, "kind": "static", "cap": 4096, "rule": ld.RULE}}
    ok = {"goal_id": "g0", "sample_idx": 0, "trial": 0, "class": "cap_coherent", "escalate": True,
          "excluded_as": None}
    p.write_text(json.dumps(hdr) + "\n" + json.dumps(ok) + "\n")
    header, rows = ld.read_items_file(p)
    assert header["cap"] == 4096 and rows == [ok]
    p.write_text(json.dumps(ok) + "\n")
    with pytest.raises(ld.Refusal, match="header"):
        ld.read_items_file(p)
    # a decision made under another rule (or none) is refused: the extend pass never acts on it
    other = {ld.ITEMS_HEADER_KEY: {**hdr[ld.ITEMS_HEADER_KEY], "rule": "at_cap_v0"}}
    p.write_text(json.dumps(other) + "\n" + json.dumps(ok) + "\n")
    with pytest.raises(ld.Refusal, match="not 'item_filtered_v1'"):
        ld.read_items_file(p)
    no_rule = {ld.ITEMS_HEADER_KEY: {k: v for k, v in hdr[ld.ITEMS_HEADER_KEY].items() if k != "rule"}}
    p.write_text(json.dumps(no_rule) + "\n" + json.dumps(ok) + "\n")
    with pytest.raises(ld.Refusal, match="decided under rule None"):
        ld.read_items_file(p)
    p.write_text(json.dumps(hdr) + "\n" + json.dumps({**ok, "escalate": False}) + "\n")
    with pytest.raises(ld.Refusal, match="escalate must be true iff"):
        ld.read_items_file(p)
    p.write_text(json.dumps(hdr) + "\n" + json.dumps({**ok, "class": "weird"}) + "\n")
    with pytest.raises(ld.Refusal, match="disposition row"):
        ld.read_items_file(p)
    p.write_text(json.dumps(hdr) + "\n" + '{"goal_id": ')
    with pytest.raises(ld.Refusal, match="torn"):
        ld.read_items_file(p)
    with pytest.raises(ld.Refusal, match="absent"):
        ld.read_items_file(tmp_path / "none.jsonl")
    bad_schema = {ld.ITEMS_HEADER_KEY: {"schema": 99}}
    p.write_text(json.dumps(bad_schema) + "\n")
    with pytest.raises(ld.Refusal, match="schema"):
        ld.read_items_file(p)


# --------------------------------------------------------------------------- #
# reclassify: every rung already generated                                     #
# --------------------------------------------------------------------------- #


def test_reclassify_static_walks_every_rung_present(tmp_path, rep_src, capsys):
    root = tmp_path / "private"
    _write_static(root, "run_t0", _base_rows())                                   # rung 4096
    rows8 = _base_rows(cap=8192)
    _write_static(root, "run_t0_cap8192", [rows8[0] + rows8[1]],
                  escalation={"cap_from": 4096, "cap_to": 8192, "n_rows": 200})  # rung 8192; 16384 absent
    _write_static(root, "run_t1", _sibling_rows())                                # the sibling, base only
    out_dir = tmp_path / "reclass"
    rc = ld.main(["reclassify", "--kind", "static", "--root", str(root), "--run", "run_t0",
                  "--sibling-run", "run_t1", "--out-dir", str(out_dir), "--rep-ratio-source", str(rep_src)])
    assert rc == 0
    line = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    summary = json.loads((out_dir / "reclassify.json").read_text())
    assert "entries" not in line and line["complete"] is True
    assert summary["rungs_present"] == [4096, 8192] and summary["rungs_absent"] == [16384]
    assert summary["rule"] == "item_filtered_v1" and summary["classifier"] == ld.CLASSIFIER
    assert summary["n_entries"] == summary["n_decided"] == 2 and summary["n_refused"] == 0
    assert summary["n_by_decision"] == {"escalate": 2, "exit_le_1pct": 0, "exit_ceiling": 0}
    by_cap = {e["cap"]: e for e in summary["entries"]}
    assert by_cap[4096]["private_run"] == "run_t0" and by_cap[4096]["sibling_status"] == "used"
    assert by_cap[4096]["n_truncated"] == 3 and by_cap[4096]["next_cap"] == 8192
    assert by_cap[8192]["private_run"] == "run_t0_cap8192" and by_cap[8192]["sibling_private_run"] == "run_t1_cap8192"
    assert by_cap[8192]["sibling_status"] == "unavailable" and by_cap[8192]["n_truncated"] == 4
    for e in summary["entries"]:
        rec = json.loads(Path(e["decision_path"]).read_text())
        assert rec["rule"] == "item_filtered_v1" and rec["decision"] == e["decision"]
        header, rows = _items(Path(e["items_path"]))
        assert header["rule"] == "item_filtered_v1" and header["private_run"] == e["private_run"]
        assert len(rows) == 7
    assert (out_dir / "static" / "run_t0" / "pasttense" / "none" / "decision.json").exists()
    # a torn store at one rung: listed as refused, the rest decided, exit 2, complete false
    p = root / "run_t0_cap8192" / "pasttense" / "none" / "shard00" / "responses.jsonl"
    p.write_bytes(p.read_bytes() + b'{"goal_id": ')
    rc = ld.main(["reclassify", "--kind", "static", "--root", str(root), "--run", "run_t0",
                  "--out-dir", str(tmp_path / "reclass2"), "--rep-ratio-source", str(rep_src)])
    assert rc == 2  # the pass is written in full but never reads as complete
    summary = json.loads((tmp_path / "reclass2" / "reclassify.json").read_text())
    assert summary["complete"] is False and summary["n_refused"] == 1 and summary["n_decided"] == 1
    refused = [e for e in summary["entries"] if e["status"] == "refused"][0]
    assert refused["cap"] == 8192 and "torn" in refused["reason"]
    assert "REFUSED" in capsys.readouterr().err
    # --run is required
    with pytest.raises(SystemExit) as ei:
        ld.main(["reclassify", "--kind", "static", "--root", str(root), "--out-dir", str(tmp_path / "x"),
                 "--rep-ratio-source", str(rep_src)])
    assert ei.value.code == 2


def test_reclassify_controls_walks_every_rung_root(tmp_path, rep_src, capsys):
    root = tmp_path / "sampled"
    _write_controls(root / "controls_cap512" / "TAG_t0", "cellA", _ctl_items())
    _write_controls(root / "controls_cap512" / "TAG_t1", "cellA",
                    [(iid, "sib", "stop", 100) if iid == "it0002" else (iid, t, fr, ct)
                     for iid, t, fr, ct in _ctl_items()])
    _write_controls(root / "controls_cap2048" / "TAG_t0", "cellA", _ctl_items(), cap=2048)  # no sibling at 2048
    out_dir = tmp_path / "reclass"
    rc = ld.main(["reclassify", "--kind", "controls", "--root", str(root), "--tag", "TAG", "--trial", "0",
                  "--sibling-trial", "1", "--out-dir", str(out_dir), "--rep-ratio-source", str(rep_src)])
    assert rc == 0
    capsys.readouterr()
    summary = json.loads((out_dir / "reclassify.json").read_text())
    assert summary["rungs_present"] == [512, 2048] and summary["rungs_absent"] == [8192, 16384]
    assert summary["n_decided"] == 2 and summary["n_refused"] == 0
    by_cap = {e["cap"]: e for e in summary["entries"]}
    assert by_cap[512]["cell"] == "cellA" and by_cap[512]["control"] == CTL
    assert by_cap[512]["sibling_status"] == "used" and by_cap[512]["n_truncated"] == 2
    assert by_cap[2048]["sibling_status"] == "unavailable" and by_cap[2048]["n_truncated"] == 3
    assert by_cap[512]["sibling_out_root"] == str(root / "controls_cap512" / "TAG_t1")
    assert (out_dir / "controls" / "cap512" / "cellA" / CTL / "items.jsonl").exists()
    rec = json.loads(Path(by_cap[2048]["decision_path"]).read_text())
    assert rec["cap"] == 2048 and rec["rule"] == "item_filtered_v1" and rec["next_cap"] == 8192
    # --controls filters; --tag/--trial are required
    rc = ld.main(["reclassify", "--kind", "controls", "--root", str(root), "--tag", "TAG", "--trial", "0",
                  "--controls", "or_bench_hard_1k", "--out-dir", str(tmp_path / "r2"),
                  "--rep-ratio-source", str(rep_src)])
    assert rc == 0
    assert json.loads((tmp_path / "r2" / "reclassify.json").read_text())["n_entries"] == 0
    with pytest.raises(SystemExit) as ei:
        ld.main(["reclassify", "--kind", "controls", "--root", str(root), "--out-dir", str(tmp_path / "r3"),
                 "--rep-ratio-source", str(rep_src)])
    assert ei.value.code == 2
    capsys.readouterr()
