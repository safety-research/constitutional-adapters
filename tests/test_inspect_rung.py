"""The Inspect rung adapter for the GSM8K paired-200 paper objects (the rung
SIZING, planner idempotence, the two-half subdivision and the one-writer walk
over a wave root are ``ladder_plan.py``'s and land beside these).

What is pinned: at-cap is the FILED ``stop_reason`` (a cap-hit draw whose text
re-tokenizes below the cap is at-cap); trigger (the AT-CAP share > 1 %, harmonised
2026-09-12 from the first publication's 3 %) vs
ELIGIBILITY (at-cap minus loops minus boundary) — a 4 % all-loop cell
triggers with no eligible item and no subset, a mixed 1.0 % cell does not
trigger (the 1 % flag coincides with the trigger), a mixed 3.5 % cell ESCALATES on the recorded
7,200 s hops and is TRANSPORT-REFUSED on a recorded 600 s upstream, with
``cap_ladder.verdict_for`` never called on either path; 2 of 200 does not
trigger, 3 does (and 6 does, the retired boundary); the serialized rule / threshold / report line are this
instrument's; redraw outcomes come from the rung's filed outcomes (never a
tokenizer); the boundary class is empty and recorded; the periodic-block
column is present and non-deciding; the merge preserves ids; the module writes
nothing. Everything synthetic: invented item ids, placeholder hashes, no
recorded data.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from sisteer.capeval.tiers import cap_ladder as L
from sisteer.capeval.tiers import inspect_rung as R
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C

TASK = C.GSM8K_STANDARD_TASK
N = C.GSM8K_PAIRED_N
CAP = C.G8K_CAP
RUNG = C.G8K_RUNG_CAP
SRC_HASH = "0123456789abcdef"
RUNG_HASH = "fedcba9876543210"
HOPS_OK = {L.HOP_DRIVER: 7200.0, L.HOP_PROXY_UPSTREAM: 7200.0}

#: a loop by Evals' rule (whitespace 4-grams repeat), a non-loop truncation
#: (every 4-gram distinct), and a SHORT cap-hit text (re-tokenizes far below any cap)
LOOP_TEXT = "the answer is the answer is the answer is " * 120
TRUNC_TEXT = " ".join(f"step{i} compute value{i} then carry{i}" for i in range(300))
SHORT_TEXT = "a few words only, cut here"


class CountTok:
    """A whitespace tokenizer — what a RE-tokenizing rule would read."""

    def __call__(self, text, add_special_tokens=None, **kw):
        return {"input_ids": list(range(len((text or "").split())))}


def _ids(n: int = N) -> list[str]:
    return [f"gsm8k_{i:08x}" for i in range(n)]


def _outcomes(ids, *, cap: int, at_cap: dict, unscored=()) -> dict:
    """An all-attempted outcome record (``contracts.TASK_OUTCOMES_HEADER_FIELDS``)
    for ``ids`` at ``cap``; ``at_cap`` = ``{item_id: completion_tokens}`` of the
    draws the provider stopped at the cap (``stop_reason == max_tokens``)."""
    items = []
    for i, iid in enumerate(ids):
        at = iid in at_cap
        scored = iid not in unscored
        items.append({"item_id": iid, "sample_idx": 0, "attempt": 1, "scored": scored,
                      "value": (1.0 if scored else None), "unparseable": None,
                      "stop_reason": (R.STOP_REASON_AT_CAP if at else "stop"), "at_cap": at,
                      "think_block_present": True, "think_block_unterminated": (iid in unscored),
                      "stray_open_tag": False, "content_empty_with_reasoning": False,
                      "prompt_tokens": 2000, "completion_tokens": int(at_cap.get(iid, 100 + i)), "seed": i})
    n, n_uns = len(ids), sum(1 for i in ids if i in unscored)
    return {"task": TASK, "item_set": C.GSM8K_ITEM_SET_PAIRED, "thinking": C.THINKING_ON,
            "think_split_mode": "implicit_open", "unterminated_policy": "unscored", "max_tokens": cap,
            "samples_per_item": 1, "n_attempted_items": n, "n_attempted_samples": n,
            "n_scored": n - n_uns, "n_unscored": n_uns, "n_scored_samples": n - n_uns, "n_unscored_samples": n_uns,
            "n_stop_reason": n, "n_at_cap": len(at_cap), "n_think_block_present": n,
            "n_think_block_unterminated": n_uns, "n_unterminated_not_at_cap": 0, "n_completion_tokens": n,
            "producer_tree_sha": None, "items": items}


def _store(root: Path, *, cap: int, at_cap: dict, ids=None, upstream=7200.0, identity_hash: str = SRC_HASH,
           cap_rung=None, item_subset=None, k: int = 1, tamper=None) -> Path:
    """A synthetic served store root: one cell dir with the version-2 manifest
    facts the adapter reads and the outcome record."""
    cell = root / "cells" / "cell__controls__t0__a0__synthetic"
    cell.mkdir(parents=True, exist_ok=True)
    tinycap = {C.IDENTITY_KEY_IDENTITY_VERSION: 2, C.IDENTITY_KEY_MAX_TOKENS_BY_TASK: {TASK: cap}}
    if cap_rung is not None:
        tinycap[C.IDENTITY_KEY_CAP_RUNG] = cap_rung
    manifest = {"harness_config_hash": identity_hash, "resolved_config": {C.MANIFEST_KEY_TINYCAP: tinycap},
                C.ENDPOINT_KEY_PROXY_UPSTREAM_TIMEOUT_S: upstream}
    if item_subset is not None:
        manifest[C.MANIFEST_KEY_ITEM_SUBSET] = item_subset
    (cell / "manifest.json").write_text(json.dumps(manifest))
    outcomes = _outcomes(ids if ids is not None else _ids(), cap=cap, at_cap=at_cap)
    if k != 1:
        outcomes["samples_per_item"] = k
    if tamper is not None:
        tamper(outcomes)
    completion.write_task_outcomes(cell, TASK, outcomes)
    return root


def _texts(ids, *, loops=(), short=()) -> dict:
    return {(i, 0): (LOOP_TEXT if i in loops else SHORT_TEXT if i in short else TRUNC_TEXT) for i in ids}


def _forbid_peer_exit_rule(monkeypatch):
    """The spy: this instrument decides its exit ITSELF."""
    for name in ("verdict_for", "rung_status", "measure_rung", "inspect_rung_from_rows", "classify_items",
                 "resolve_redraw"):
        def boom(*a, _n=name, **k):
            raise AssertionError(f"cap_ladder.{_n} must never be called by the Inspect rung adapter")
        monkeypatch.setattr(L, name, boom)


# --------------------------------------------------------------------------- #
# at-cap is the filed stop reason                                              #
# --------------------------------------------------------------------------- #


def test_at_cap_is_the_filed_stop_reason_never_a_retokenization(tmp_path, monkeypatch):
    """P1-2: a cap-hit outcome whose emitted text re-tokenizes far BELOW the cap
    (a served draw the provider stopped at ``max_tokens``) is at-cap here; the
    re-tokenizing rule would have read it as terminated."""
    _forbid_peer_exit_rule(monkeypatch)
    ids = _ids()
    hit = ids[:7]
    root = _store(tmp_path / "s", cap=CAP, at_cap={i: CAP for i in hit})
    record, subset = R.classify_store(root, texts=_texts(ids, short=hit))
    assert not L.is_at_cap(CountTok(), SHORT_TEXT, CAP)          # the re-tokenizing reading says "below"
    facts = {f["item_id"]: f for f in record["item_facts"]}
    assert all(facts[i]["at_cap"] is True and facts[i]["stop_reason"] == "max_tokens" for i in hit)
    assert record["n_at_cap"] == 7 and record["at_cap_rule"] == R.AT_CAP_RULE_INSPECT
    rows = {r["item_id"]: r for r in record["items"]}
    assert all(rows[i]["at_cap"] is True and rows[i]["ntok"] == rows[i]["completion_tokens"] == CAP for i in hit)
    assert set(rows[hit[0]]) == set(L.ROW_KEYS)
    assert record["exit"] == L.ESCALATE and subset is not None and sorted(subset["ids"]) == sorted(hit)


# --------------------------------------------------------------------------- #
# trigger vs eligibility                                                       #
# --------------------------------------------------------------------------- #


def test_four_percent_all_loops_triggers_with_no_eligible_item_and_no_subset(tmp_path, monkeypatch):
    _forbid_peer_exit_rule(monkeypatch)
    ids = _ids()
    loops = ids[:8]                                                   # 4.0 % at cap, every one a loop
    root = _store(tmp_path / "s", cap=CAP, at_cap={i: CAP for i in loops})
    record, subset = R.classify_store(root, texts=_texts(ids, loops=loops))
    assert record["exit"] == R.EXIT_TRIGGERED_NO_ELIGIBLE and record["escalate"] is False and subset is None
    assert record["n_at_cap"] == 8 and record["share_at_cap"] == 0.04 and record["share_at_cap_over_1pct"] is True
    assert record["n_non_terminating"] == 8 and record["n_eligible"] == 0 and record["eligible_ids"] == []
    assert sorted(record["non_terminating_ids"]) == sorted(loops)
    assert record["next_cap"] is None and record["transport"] is None
    status = record["controls"][TASK]
    assert status["exit"] == R.EXIT_TRIGGERED_NO_ELIGIBLE and status["n_truncated"] == 0 and status["n_at_cap"] == 8
    assert all(f["kind"] == L.KIND_NON_TERMINATING and f["degenerate"] is True and f["eligible"] is False
               for f in record["item_facts"] if f["at_cap"])


def test_a_mixed_cell_at_exactly_one_percent_does_not_trigger_and_the_one_percent_flag_is_off(tmp_path, monkeypatch):
    """Since the harmonisation of 2026-09-12 the trigger IS the 1 % line (both strict ``>``): 2 of 200 sits AT the
    line, so it neither triggers nor files the flag; a cell that files the flag necessarily triggers."""
    _forbid_peer_exit_rule(monkeypatch)
    ids = _ids()
    at = ids[:2]                                                       # 1.0 %: 1 loop + 1 truncated
    root = _store(tmp_path / "s", cap=CAP, at_cap={i: CAP for i in at})
    record, subset = R.classify_store(root, texts=_texts(ids, loops=at[:1]))
    assert record["exit"] == L.EXIT_BELOW_THRESHOLD and subset is None and record["escalate"] is False
    assert record["share_at_cap"] == 0.01 and record["share_at_cap_over_1pct"] is False
    assert record["n_eligible"] == 1 and record["share_eligible"] == 0.005 and record["share_eligible_over_1pct"] is False
    assert record["n_non_terminating"] == 1 and record["transport"] is None
    assert R.GSM8K_TRIGGER_THRESHOLD == R.GSM8K_REPORT_LINE == 0.01


def test_a_mixed_cell_at_three_and_a_half_percent_escalates_on_the_recorded_hops_and_is_refused_at_600(
        tmp_path, monkeypatch):
    """7 at-cap of 200 incl. 2 loops — the instrument's trigger is TRUE
    (3.5 %), the eligible share 2.5 % (the peer's ``verdict_for`` reads the TRUNCATED
    share; the mixed cell below shows the two straddling the threshold). The adapter
    escalates on the RECORDED hops (driver 7,200 s, upstream 7,200 s) with the
    subset = the 5 eligible ids; the SAME cell on a recorded 600 s upstream is
    ``transport_refused`` with no subset; ``verdict_for`` is never called."""
    _forbid_peer_exit_rule(monkeypatch)
    ids = _ids()
    at = ids[:7]
    loops, eligible = at[:2], at[2:]
    root_ok = _store(tmp_path / "ok", cap=CAP, at_cap={i: CAP for i in at}, upstream=7200.0)
    record, subset = R.classify_store(root_ok, texts=_texts(ids, loops=loops))
    assert record["exit"] == L.ESCALATE and record["escalate"] is True and record["next_cap"] == RUNG
    assert record["share_at_cap"] == 0.035 and record["share_eligible"] == 0.025
    assert sorted(record["eligible_ids"]) == sorted(eligible) and sorted(record["non_terminating_ids"]) == sorted(loops)
    assert record["hops"] == HOPS_OK and record["transport"]["ok"] is True
    assert record["transport"]["binding_s"] == 7200.0 and record["transport"]["required_s"] == 6144.0
    assert subset is not None
    assert set(subset) == set(C.ITEM_SUBSET_FIELDS) and subset["schema"] == C.ITEM_SUBSET_SCHEMA
    assert sorted(subset["ids"]) == sorted(eligible) and subset["cap"] == RUNG and subset["source_cap"] == CAP
    assert subset["source_identity_hash"] == SRC_HASH and subset["source_out_root"] == str(root_ok)
    assert (subset["n_at_cap"], subset["n_non_terminating"], subset["n_boundary"]) == (7, 2, 0)
    assert subset["rule"] == R.GSM8K_TRIGGER_RULE and subset["threshold"] == R.GSM8K_TRIGGER_THRESHOLD == 0.01
    assert subset["transport_check"]["ok"] is True
    C.validate_item_subset(subset)
    # the record names BOTH readings: the at-cap share that decided, the eligible share the peer would read
    status = record["controls"][TASK]
    assert status["n_at_cap"] == 7 and status["share_at_cap"] == 0.035
    assert status["n_truncated"] == 5 and status["share_truncated"] == 0.025 and sorted(status["truncated_ids"]) == sorted(eligible)
    assert status["escalate"] is True and status["exit"] == L.ESCALATE and status["ladder_available"] is True

    root_600 = _store(tmp_path / "slow", cap=CAP, at_cap={i: CAP for i in at}, upstream=600.0)
    record2, subset2 = R.classify_store(root_600, texts=_texts(ids, loops=loops))
    assert record2["exit"] == L.EXIT_TRANSPORT_REFUSED and record2["escalate"] is False and subset2 is None
    assert record2["next_cap"] == RUNG and record2["transport"]["ok"] is False
    assert record2["transport"]["binding_hop"] == L.HOP_PROXY_UPSTREAM and record2["transport"]["binding_s"] == 600.0
    assert "REFUSED" in record2["transport"]["reason"]
    assert record2["controls"][TASK]["exit"] == L.EXIT_TRANSPORT_REFUSED


def test_the_peers_verdict_on_the_mixed_cell_is_below_threshold_which_is_why_it_is_not_consulted():
    """The reason for the spy: the peer's rule reads the TRUNCATED share, this
    instrument the AT-CAP share. A mixed cell with 5 at cap of which 3 are loops
    (2.5 % at cap, 1.0 % truncated) TRIGGERS here (2.5 % > 1 %) while the peer
    reads 1.0 % <= 1 % -> ``below_threshold`` with NO transport verdict. Consuming
    it would suppress the required rung."""
    verdict, escalate, nxt, _at_ceiling, tv = L.verdict_for(
        share_truncated=2 / 200, n_truncated=2, cap=CAP, threshold=R.GSM8K_TRIGGER_THRESHOLD,
        hop_timeouts_s=HOPS_OK)
    assert verdict == L.EXIT_BELOW_THRESHOLD and escalate is False and nxt is None and tv is None


def test_two_at_cap_of_200_does_not_trigger_and_three_does(tmp_path, monkeypatch):
    _forbid_peer_exit_rule(monkeypatch)
    ids = _ids()
    two = _store(tmp_path / "two", cap=CAP, at_cap={i: CAP for i in ids[:2]})
    rec2, sub2 = R.classify_store(two, texts=_texts(ids))
    assert rec2["exit"] == L.EXIT_BELOW_THRESHOLD and sub2 is None and rec2["share_at_cap"] == 0.01
    three = _store(tmp_path / "three", cap=CAP, at_cap={i: CAP for i in ids[:3]})
    rec3, sub3 = R.classify_store(three, texts=_texts(ids))
    assert rec3["exit"] == L.ESCALATE and sub3 is not None and len(sub3["ids"]) == 3
    # the retired 3 % boundary (6 vs 7 of 200) no longer decides: 6 of 200 escalates too
    six = _store(tmp_path / "six", cap=CAP, at_cap={i: CAP for i in ids[:6]})
    rec6, sub6 = R.classify_store(six, texts=_texts(ids))
    assert rec6["exit"] == L.ESCALATE and sub6 is not None and rec6["share_at_cap"] == 0.03
    # and the boundary of the rule itself, on the pure decision
    assert R.decide(n=200, n_at_cap=2, eligible_ids=["x"], cap=CAP, hops=HOPS_OK)[0] == L.EXIT_BELOW_THRESHOLD
    assert R.decide(n=200, n_at_cap=3, eligible_ids=["x"], cap=CAP, hops=HOPS_OK)[0] == L.ESCALATE
    assert R.decide(n=200, n_at_cap=0, eligible_ids=[], cap=CAP, hops=HOPS_OK)[0] == L.EXIT_CLEAN


def test_a_clean_cell_exits_clean_with_the_one_percent_flag_off(tmp_path, monkeypatch):
    _forbid_peer_exit_rule(monkeypatch)
    root = _store(tmp_path / "s", cap=CAP, at_cap={})
    record, subset = R.classify_store(root, texts={})                 # below-cap items need no text
    assert record["exit"] == L.EXIT_CLEAN and subset is None
    assert record["n_at_cap"] == 0 and record["share_at_cap_over_1pct"] is False and record["n_eligible"] == 0


# --------------------------------------------------------------------------- #
# the serialized rule                                                          #
# --------------------------------------------------------------------------- #


def test_the_serialized_rule_threshold_and_report_line_are_this_instruments(tmp_path, monkeypatch):
    _forbid_peer_exit_rule(monkeypatch)
    ids = _ids()
    root = _store(tmp_path / "s", cap=CAP, at_cap={i: CAP for i in ids[:7]})
    record, _ = R.classify_store(root, texts=_texts(ids))
    assert record["schema"] == L.RECORD_SCHEMA == "cap_ladder/1"
    assert record["instrument"] == R.INSTRUMENT
    assert record["rule"] == R.GSM8K_TRIGGER_RULE == record["rules"]["trigger"]
    assert "n_at_cap / n > 0.01" in record["rule"] and "16,384" in record["rule"] and "no rung above" in record["rule"]
    assert record["threshold"] == R.GSM8K_TRIGGER_THRESHOLD == 0.01
    # a FRESH record never carries the retired 3 % pair (that pair is admitted on rung evidence only)
    assert R.GSM8K_TRIGGER_RULE_3PCT_RETIRED not in json.dumps(record) and record["threshold"] != R.GSM8K_TRIGGER_THRESHOLD_3PCT_RETIRED
    assert record["report_line"] == R.GSM8K_REPORT_LINE == L.AT_CAP_SHARE_THRESHOLD == 0.01
    assert record["ceiling"] == R.GSM8K_CEILING == L.CAP_CEILING == C.ON_CAP_CEILING == 16384
    assert record["ladder"] == [CAP, RUNG]
    assert record["rules"]["at_cap"] == R.AT_CAP_RULE_INSPECT and "never re-tokenized" in record["rules"]["at_cap"]
    assert record["rules"]["degenerate"] == L.DEGENERATE_RULE and record["rules"]["transport"] == L.TRANSPORT_RULE
    assert record["rules"]["boundary"] == R.BOUNDARY_RULE_K1
    # Evals' 1 %-truncated-share TRIGGER string is NOT what this instrument serializes
    assert L.TRIGGER_RULE not in json.dumps(record)
    assert record["controls"][TASK]["threshold"] == 0.01 and record["controls"][TASK]["rules"]["trigger"] == R.GSM8K_TRIGGER_RULE
    assert all(f in record for f in R.RECORD_INSTRUMENT_FIELDS)


# --------------------------------------------------------------------------- #
# the redraw record (a rung store against its source)                          #
# --------------------------------------------------------------------------- #


def _rung_world(tmp_path):
    """A source at 8,192 with 7 at-cap (2 loops) and its rung at 16,384 over the
    5 eligible ids: one still at cap, one landing below the SOURCE cap (boundary
    noise), three extended past it."""
    ids = _ids()
    at = ids[:7]
    loops, eligible = at[:2], sorted(at[2:])
    src = _store(tmp_path / "src", cap=CAP, at_cap={i: CAP for i in at})
    src_texts = _texts(ids, loops=loops)
    subset = {"schema": C.ITEM_SUBSET_SCHEMA, "task": TASK, "source_out_root": str(src),
              "source_identity_hash": SRC_HASH, "source_cap": CAP, "cap": RUNG, "rule": R.GSM8K_TRIGGER_RULE,
              "threshold": R.GSM8K_TRIGGER_THRESHOLD, "ids": eligible, "n_at_cap": 7, "n_non_terminating": 2, "n_boundary": 0,
              "transport_check": L.transport_check(RUNG, HOPS_OK).as_dict()}
    C.validate_item_subset(subset)
    still, landing, *extended = eligible
    rung_at = {still: RUNG}
    rung_ct = {landing: 3000, extended[0]: 9000, extended[1]: 12000, extended[2]: 16000}
    rung = _store(tmp_path / "rung", cap=RUNG, ids=eligible, at_cap=rung_at, identity_hash=RUNG_HASH,
                  cap_rung=C.cap_rung_identity(subset), item_subset=C.item_subset_block(subset, path="rung_subset.json"),
                  tamper=lambda o: [it.update(completion_tokens=rung_ct[it["item_id"]]) for it in o["items"]
                                    if it["item_id"] in rung_ct])
    rung_texts = {(i, 0): (TRUNC_TEXT + " more words after the cut" if i != still else LOOP_TEXT[:50] + TRUNC_TEXT)
                  for i in eligible}
    return {"ids": ids, "eligible": eligible, "still": still, "landing": landing, "extended": extended,
            "src": src, "rung": rung, "src_texts": src_texts, "rung_texts": rung_texts, "subset": subset}


def test_redraw_outcomes_come_from_the_rung_outcomes_never_a_tokenizer(tmp_path, monkeypatch):
    _forbid_peer_exit_rule(monkeypatch)          # resolve_redraw (re-tokenizing) included
    w = _rung_world(tmp_path)
    record, subset = R.classify_store(w["rung"], texts=w["rung_texts"], source_texts=w["src_texts"])
    assert subset is None                                                   # no rung above
    assert record["source"]["is_rung"] is True and record["source"]["source_identity_hash"] == SRC_HASH
    assert record["source"]["n_redrawn"] == 5 and record["cap"] == RUNG and record["ladder"] == [RUNG]
    rows = {r["item_id"]: r for r in record["items"]}
    assert all(rows[i]["regenerated"] is True and set(rows[i]["redraw"]) == set(L.REDRAW_KEYS) for i in w["eligible"])
    assert rows[w["still"]]["redraw"]["outcome"] == L.REDRAW_STILL_AT_CAP and rows[w["still"]]["redraw"]["still_at_cap"] is True
    assert rows[w["landing"]]["redraw"]["outcome"] == L.REDRAW_BOUNDARY_LANDING
    assert rows[w["landing"]]["redraw"]["new_completion_tokens"] == 3000 < CAP
    for i in w["extended"]:
        assert rows[i]["redraw"]["outcome"] == L.REDRAW_EXTENDED and rows[i]["redraw"]["extended"] is True
        assert rows[i]["redraw"]["prefix_extension"] is True and rows[i]["redraw"]["identical_text"] is False
    # the text relationship is recorded from the two emitted texts; NOTHING re-tokenizes
    assert all(rows[i]["redraw"]["first_divergence_token"] is None for i in w["eligible"])
    assert rows[w["still"]]["redraw"]["prefix_extension"] is False and isinstance(rows[w["still"]]["redraw"]["first_divergence_char"], int)
    assert all(rows[i]["redraw"]["source_cap"] == CAP and rows[i]["redraw"]["new_cap"] == RUNG for i in w["eligible"])
    assert all(rows[i]["redraw"]["source_completion_tokens"] == CAP for i in w["eligible"])
    status = record["controls"][TASK]
    assert status["climbed"] is True and status["n_regenerated"] == 5
    assert status["redraw_outcomes"] == {L.REDRAW_BOUNDARY_LANDING: 1, L.REDRAW_EXTENDED: 3, L.REDRAW_STILL_AT_CAP: 1}
    assert status["n_prefix_extension"] == 4 and status["n_fork"] == 1
    # the residual at the ceiling: 1 of 5 still at cap (20 % of the rung's population) -> `ceiling`, no rung above
    assert record["exit"] == L.EXIT_CEILING and record["n_at_cap"] == 1 and record["next_cap"] is None


def test_a_rung_bound_to_another_source_or_over_another_population_refuses(tmp_path, monkeypatch):
    _forbid_peer_exit_rule(monkeypatch)
    w = _rung_world(tmp_path)
    # another source hash
    other = _store(tmp_path / "other", cap=CAP, at_cap={i: CAP for i in w["ids"][:7]}, identity_hash="1111111111111111")
    bad_subset = {**w["subset"], "source_out_root": str(other)}
    rung2 = _store(tmp_path / "rung2", cap=RUNG, ids=w["eligible"], at_cap={}, identity_hash=RUNG_HASH,
                   cap_rung=C.cap_rung_identity(w["subset"]), item_subset=C.item_subset_block(bad_subset, path="x.json"))
    with pytest.raises(R.InspectRungError, match="bound to source identity"):
        R.classify_store(rung2, texts=w["rung_texts"], source_texts=w["src_texts"])
    # a rung that drew an id outside its recorded subset
    rung3 = _store(tmp_path / "rung3", cap=RUNG, ids=w["eligible"] + [w["ids"][100]], at_cap={}, identity_hash=RUNG_HASH,
                   cap_rung=C.cap_rung_identity(w["subset"]), item_subset=C.item_subset_block(w["subset"], path="x.json"))
    with pytest.raises(R.InspectRungError, match="not the recorded subset"):
        R.classify_store(rung3, texts={**w["rung_texts"], (w["ids"][100], 0): TRUNC_TEXT}, source_texts=w["src_texts"])
    # a rung item the source did NOT truncate
    src_ok = _store(tmp_path / "src_ok", cap=CAP, at_cap={i: CAP for i in w["ids"][2:7]})   # first two below cap
    sub = {**w["subset"], "source_out_root": str(src_ok), "ids": sorted(w["ids"][:5]), "n_at_cap": 5, "n_non_terminating": 0}
    C.validate_item_subset(sub)
    rung4 = _store(tmp_path / "rung4", cap=RUNG, ids=sorted(w["ids"][:5]), at_cap={}, identity_hash=RUNG_HASH,
                   cap_rung=C.cap_rung_identity(sub), item_subset=C.item_subset_block(sub, path="x.json"))
    with pytest.raises(R.InspectRungError, match="NOT at cap in the source"):
        R.classify_store(rung4, texts={(i, 0): TRUNC_TEXT for i in w["ids"][:5]}, source_texts=_texts(w["ids"]))


# --------------------------------------------------------------------------- #
# boundary, the periodic-block column, transport                               #
# --------------------------------------------------------------------------- #


def test_boundary_is_empty_by_construction_at_k_1_and_recorded(tmp_path, monkeypatch):
    _forbid_peer_exit_rule(monkeypatch)
    ids = _ids()
    root = _store(tmp_path / "s", cap=CAP, at_cap={i: CAP for i in ids[:7]})
    record, subset = R.classify_store(root, texts=_texts(ids))
    assert record["n_boundary"] == 0 and record["boundary_ids"] == [] and record["controls"][TASK]["n_boundary"] == 0
    assert "k = 1" in record["boundary_rule"] and "EMPTY by construction" in record["boundary_rule"]
    assert all(r["boundary_from"] is None for r in record["items"])
    assert subset is not None and subset["n_boundary"] == 0
    # a k > 1 store is not this instrument's — the boundary reading rests on k = 1
    with pytest.raises(R.InspectRungError, match="k = 1"):
        R.load_store(_store(tmp_path / "k2", cap=CAP, at_cap={}, k=2))


def test_the_periodic_block_column_is_present_and_non_deciding(tmp_path, monkeypatch):
    """A cap-hit text that is a pure 8-char repeat with NO whitespace: the periodic-
    block rule fires (period 8, hundreds of repeats) but Evals' whitespace
    4-gram rule reads one "word" (rep_ratio 0.0) — the item stays ELIGIBLE. The
    column is recorded, named non-deciding, and changes no exit."""
    _forbid_peer_exit_rule(monkeypatch)
    ids = _ids()
    at = ids[:7]
    periodic = "abcdefgh" * 400
    assert R.periodic_block(periodic)["loop"] is True and R.periodic_block(periodic)["period"] == 8
    assert L.is_degenerate(periodic) is False
    assert R.periodic_block(TRUNC_TEXT)["loop"] is False and R.periodic_block(None)["loop"] is False
    root = _store(tmp_path / "s", cap=CAP, at_cap={i: CAP for i in at})
    texts = _texts(ids)
    texts[(at[0], 0)] = periodic
    record, subset = R.classify_store(root, texts=texts)
    facts = {f["item_id"]: f for f in record["item_facts"]}
    assert facts[at[0]]["periodic_block"]["loop"] is True and facts[at[0]]["eligible"] is True
    assert facts[at[0]]["kind"] == L.KIND_TRUNCATED and facts[at[0]]["degenerate"] is False
    assert record["periodic_block"] == {"rule": R.PERIODIC_BLOCK_RULE, "n_loop": 1, "ids": [at[0]], "deciding": False}
    assert record["exit"] == L.ESCALATE and subset is not None and at[0] in subset["ids"]


def test_the_transport_guard_refuses_600_admits_7200_and_refuses_an_unrecorded_upstream(tmp_path, monkeypatch):
    _forbid_peer_exit_rule(monkeypatch)
    ok = R.decide(n=200, n_at_cap=7, eligible_ids=["a"], cap=CAP, hops=HOPS_OK)
    assert ok[0] == L.ESCALATE and ok[1] == RUNG and ok[2].ok is True
    slow = R.decide(n=200, n_at_cap=7, eligible_ids=["a"], cap=CAP, hops={**HOPS_OK, L.HOP_PROXY_UPSTREAM: 600.0})
    assert slow[0] == L.EXIT_TRANSPORT_REFUSED and slow[2].ok is False and slow[2].binding_hop == L.HOP_PROXY_UPSTREAM
    # a store whose manifest recorded NO upstream timeout is undeterminable — refused, never defaulted
    ids = _ids()
    root = _store(tmp_path / "s", cap=CAP, at_cap={i: CAP for i in ids[:7]}, upstream=None)
    record, subset = R.classify_store(root, texts=_texts(ids))
    assert record["exit"] == L.EXIT_TRANSPORT_REFUSED and subset is None
    assert record["hops"] == {L.HOP_DRIVER: 7200.0, L.HOP_PROXY_UPSTREAM: None}
    assert "undeterminable" in record["transport"]["reason"]
    assert R.DRIVER_HOP_TIMEOUT_S == C.PROXY_UPSTREAM_TIMEOUT_S == 7200.0


def test_at_the_ceiling_the_exit_is_ceiling_and_the_residual_is_filed(tmp_path, monkeypatch):
    _forbid_peer_exit_rule(monkeypatch)
    ids = _ids(20)
    root = _store(tmp_path / "s", cap=RUNG, ids=ids, at_cap={i: RUNG for i in ids[:3]})
    record, subset = R.classify_store(root, texts=_texts(ids))
    assert record["exit"] == L.EXIT_CEILING and subset is None and record["n_at_cap"] == 3
    assert record["controls"][TASK]["at_ceiling"] is True and record["ladder"] == [RUNG]


# --------------------------------------------------------------------------- #
# the merge, the text source, refusals, one writer                             #
# --------------------------------------------------------------------------- #


def test_the_merge_preserves_ids_and_takes_the_rung_draw_for_the_subset_only():
    ids = _ids(10)
    src = [{"item_id": i, "value": 0.0, "at_cap": i in ids[:3]} for i in ids]
    rung = [{"item_id": i, "value": 1.0, "at_cap": False} for i in ids[:3]]
    merged = R.merge_items(src, rung, ids[:3])
    assert [m["item_id"] for m in merged] == ids
    assert [m["draw"] for m in merged] == ["rung"] * 3 + ["source"] * 7
    assert [m["value"] for m in merged] == [1.0] * 3 + [0.0] * 7
    with pytest.raises(R.InspectRungError, match="outside its subset"):
        R.merge_items(src, rung + [{"item_id": ids[5], "value": 1.0}], ids[:3])
    with pytest.raises(R.InspectRungError, match="missed"):
        R.merge_items(src, rung[:2], ids[:3])
    with pytest.raises(R.InspectRungError, match="not source items"):
        R.merge_items(src, [{"item_id": "gsm8k_zzzz"}], ["gsm8k_zzzz"])


def test_classify_store_reads_texts_from_the_retained_log_by_default_and_refuses_a_textless_at_cap_item(
        tmp_path, monkeypatch):
    _forbid_peer_exit_rule(monkeypatch)
    ids = _ids()
    root = _store(tmp_path / "s", cap=CAP, at_cap={i: CAP for i in ids[:7]})
    seen = []

    def fake_texts(facts):
        seen.append(facts.cell_dir)
        return _texts(ids)

    monkeypatch.setattr(R, "store_texts", fake_texts)
    record, _ = R.classify_store(root)
    assert seen and record["exit"] == L.ESCALATE
    # an at-cap item with no emitted text on record refuses — a loop is never assumed absent
    texts = _texts(ids)
    texts.pop((ids[0], 0))
    with pytest.raises(R.InspectRungError, match="no emitted text on record"):
        R.classify_store(root, texts=texts)


def test_store_texts_reads_through_the_producers_own_log_reader(tmp_path, monkeypatch):
    """The text source is the retained ``.eval`` through ``inspect_served.
    reusable_log`` + ``results_from_log`` at the store's cap over its ids; a
    log whose stop reason disagrees with the outcome record refuses."""
    pytest.importorskip("inspect_ai", reason="the client venv (hashed lock)")
    from types import SimpleNamespace

    from sisteer.capeval.tiers import inspect_served as isv

    assert R.STOP_REASON_AT_CAP == isv.STOP_REASON_AT_CAP
    ids = _ids(4)
    root = _store(tmp_path / "s", cap=CAP, ids=ids, at_cap={ids[0]: CAP})
    facts = R.load_store(root)
    calls = {}

    def fake_reusable(log_dir, *, expected_n, expected_ids=None, expected_max_tokens=None):
        calls.update(log_dir=Path(log_dir), n=expected_n, ids=list(expected_ids), cap=expected_max_tokens)
        return object()

    def fake_results(log, *, sample_idx, primary_scorer, paired_ids, attempt, max_tokens_used, target_model=None):
        return [SimpleNamespace(item_id=i, sample_idx=0, content=f"content {i}", reasoning="thoughts ",
                                stop_reason=(R.STOP_REASON_AT_CAP if i == ids[0] else "stop")) for i in ids]

    monkeypatch.setattr(isv, "reusable_log", fake_reusable)
    monkeypatch.setattr(isv, "results_from_log", fake_results)
    texts = R.store_texts(facts)
    assert texts == {(i, 0): f"thoughts content {i}" for i in ids}
    assert calls["log_dir"] == C.inspect_log_dir(facts.cell_dir, TASK, 0) and calls["n"] == 4 and calls["cap"] == CAP
    assert calls["ids"] == sorted(ids)
    monkeypatch.setattr(isv, "results_from_log", lambda *a, **k: [
        SimpleNamespace(item_id=ids[0], sample_idx=0, content="x", reasoning=None, stop_reason="stop")])
    with pytest.raises(R.InspectRungError, match="torn store"):
        R.store_texts(facts)
    monkeypatch.setattr(isv, "reusable_log", lambda *a, **k: None)
    with pytest.raises(R.InspectRungError, match="no complete retained .eval"):
        R.store_texts(facts)


def test_load_store_refuses_torn_v1_and_headerless_stores(tmp_path):
    ids = _ids(5)
    # a torn record: at_cap disagrees with stop_reason
    root = _store(tmp_path / "torn", cap=CAP, ids=ids, at_cap={ids[0]: CAP},
                  tamper=lambda o: o["items"][0].update(at_cap=False))
    with pytest.raises(R.InspectRungError, match="disagree"):
        R.load_store(root)
    # a v1 store: no hashed cap map
    v1 = tmp_path / "v1"
    _store(v1, cap=CAP, ids=ids, at_cap={})
    cell = R.cell_dir_of(v1)
    m = json.loads((cell / "manifest.json").read_text())
    m["resolved_config"][C.MANIFEST_KEY_TINYCAP].pop(C.IDENTITY_KEY_MAX_TOKENS_BY_TASK)
    (cell / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(R.InspectRungError, match="identity-version-2"):
        R.load_store(v1)
    # the outcome record's cap is not the identity's
    root2 = _store(tmp_path / "cap", cap=CAP, ids=ids, at_cap={}, tamper=lambda o: o.update(max_tokens=4096))
    with pytest.raises(R.InspectRungError, match="not the identity's executed cap"):
        R.load_store(root2)
    # no outcome record at all
    bare = tmp_path / "bare" / "cells" / "c"
    bare.mkdir(parents=True)
    (bare / "manifest.json").write_text(json.dumps({"harness_config_hash": SRC_HASH, "resolved_config": {
        C.MANIFEST_KEY_TINYCAP: {C.IDENTITY_KEY_MAX_TOKENS_BY_TASK: {TASK: CAP}}}}))
    with pytest.raises(R.InspectRungError, match="no outcomes/"):
        R.load_store(tmp_path / "bare")
    with pytest.raises(R.InspectRungError, match="not exactly one"):
        R.cell_dir_of(tmp_path / "nowhere")


def test_the_adapter_writes_nothing_and_never_names_the_peers_exit_machinery():
    src = Path(R.__file__).read_text()
    for forbidden in ("write_cell_record", "RECORD_NAME", "L.verdict_for(", "L.rung_status(", "L.measure_rung(",
                      "L.classify_items(", "L.inspect_rung_from_rows(", "L.resolve_redraw(", ".write_text(", "open("):
        assert forbidden not in src, forbidden
    # the exits this instrument files, and the borrowed vocabulary they come from
    assert set(R.EXITS) == {L.EXIT_CLEAN, L.EXIT_BELOW_THRESHOLD, L.EXIT_CEILING, R.EXIT_TRIGGERED_NO_ELIGIBLE,
                            L.EXIT_TRANSPORT_REFUSED, L.ESCALATE}
    assert set(R.TERMINAL_EXITS) == set(R.EXITS) - {L.ESCALATE}
    assert re.fullmatch(r"[a-z_]+", R.EXIT_TRIGGERED_NO_ELIGIBLE) and R.EXIT_TRIGGERED_NO_ELIGIBLE not in L.EXITS


def test_the_producer_contract_is_exported_by_name_and_spelled_in_ascii(tmp_path, monkeypatch):
    """Integration review A-P1.1 / B-P1.2 / B-P1.3: the planner and the reader
    IMPORT the producer's spellings — never re-type them. A re-typed copy drifted
    once (a U+2212 minus for this module's ASCII hyphen-minus, `eligible` for
    `eligible_ids`, the unprefixed task name for the ROW task, a newline-joined
    digest for the contract's canonical-JSON one), so every literal is pinned
    here BY NAME: a later drift in the producer fails this test, and a copy
    elsewhere fails its own pin against these names."""
    # the rule string, VERBATIM, ASCII throughout (U+002D, never U+2212)
    assert R.GSM8K_TRIGGER_RULE == (
        "trigger iff n_at_cap / n > 0.01 (at-cap = stop_reason == max_tokens at the request's cap); "
        "redraw the eligible set = at_cap - degenerate - boundary; ceiling 16,384; no rung above")
    assert R.GSM8K_TRIGGER_RULE.isascii() and "\u2212" not in R.GSM8K_TRIGGER_RULE
    assert R.GSM8K_TRIGGER_RULE != R.GSM8K_TRIGGER_RULE.replace(" - ", " \u2212 ")      # the drifted copy is NOT equal
    assert R.GSM8K_TRIGGER_THRESHOLD == 0.01 and R.GSM8K_REPORT_LINE == 0.01 == L.AT_CAP_SHARE_THRESHOLD
    # the RETIRED first-publication pair, verbatim (the four published rung stores hash this string into their identity),
    # the admitted map a reader consults on rung evidence, and the provenance sentence every report states
    assert R.GSM8K_TRIGGER_RULE_3PCT_RETIRED == (
        "trigger iff n_at_cap / n > 0.03 (at-cap = stop_reason == max_tokens at the request's cap); "
        "redraw the eligible set = at_cap - degenerate - boundary; ceiling 16,384; no rung above")
    assert R.GSM8K_TRIGGER_THRESHOLD_3PCT_RETIRED == 0.03 and R.GSM8K_TRIGGER_RULE_3PCT_RETIRED != R.GSM8K_TRIGGER_RULE
    assert R.GSM8K_TRIGGER_RULES_ADMITTED == {R.GSM8K_TRIGGER_RULE: 0.01, R.GSM8K_TRIGGER_RULE_3PCT_RETIRED: 0.03}
    assert R.GSM8K_TRIGGER_RULE.replace("0.01", "0.03") == R.GSM8K_TRIGGER_RULE_3PCT_RETIRED
    for piece in ("1 % RAW at-cap", "2026-09-12", "3 % exception", "2026-09-10"):
        assert piece in R.GSM8K_TRIGGER_PROVENANCE, piece
    # the ROW task: the one comparison spelling; the unprefixed NAME is the CLI cap flags' and plan_task's only
    assert R.GSM8K_TASK == C.GSM8K_STANDARD_TASK == "inspect/gsm8k_paired200"
    assert C.GSM8K_PAIRED_TASK_NAME == "gsm8k_paired200" != R.GSM8K_TASK
    assert C.inspect_task(C.GSM8K_PAIRED_TASK_NAME) == R.GSM8K_TASK
    # the exit vocabulary and its two reader classes
    assert R.EXIT_TRIGGERED_NO_ELIGIBLE == "triggered_no_eligible" and R.EXIT_ESCALATE == L.ESCALATE == "escalate"
    assert R.NOT_TRIGGERED_EXITS == ("clean", "below_threshold") == (L.EXIT_CLEAN, L.EXIT_BELOW_THRESHOLD)
    assert R.TRIGGERED_TERMINAL_EXITS == ("triggered_no_eligible", "transport_refused")
    assert set(R.EXITS) == {*R.NOT_TRIGGERED_EXITS, *R.TRIGGERED_TERMINAL_EXITS, R.EXIT_ESCALATE, L.EXIT_CEILING}
    assert set(R.TERMINAL_EXITS) == set(R.EXITS) - {R.EXIT_ESCALATE}
    # the record's three id lists and the per-draw classes
    assert R.RECORD_FIELD_ELIGIBLE_IDS == "eligible_ids"
    assert R.RECORD_FIELD_NON_TERMINATING_IDS == "non_terminating_ids" and R.RECORD_FIELD_BOUNDARY_IDS == "boundary_ids"
    assert R.RECORD_ID_LIST_FIELDS == ("eligible_ids", "non_terminating_ids", "boundary_ids")
    assert all(f in R.RECORD_INSTRUMENT_FIELDS for f in R.RECORD_ID_LIST_FIELDS)
    assert "eligible" not in R.RECORD_INSTRUMENT_FIELDS and "boundary" not in R.RECORD_INSTRUMENT_FIELDS
    assert R.ITEM_KINDS == ("truncated", "non_terminating") == (L.KIND_TRUNCATED, L.KIND_NON_TERMINATING)
    assert R.ITEM_FACT_FIELDS[:2] == ("item_id", "sample_idx")
    assert {"kind", "eligible", "at_cap", "degenerate", "completion_tokens", "stop_reason"} <= set(R.ITEM_FACT_FIELDS)
    # a REAL record and subset carry them under exactly those names
    _forbid_peer_exit_rule(monkeypatch)
    ids = _ids()
    at = ids[:7]
    root = _store(tmp_path / "s", cap=CAP, at_cap={i: CAP for i in at})
    record, subset = R.classify_store(root, texts=_texts(ids, loops=at[:2]))
    assert record["exit"] == R.EXIT_ESCALATE and subset is not None
    assert record[R.RECORD_FIELD_ELIGIBLE_IDS] == sorted(at[2:])
    assert record[R.RECORD_FIELD_NON_TERMINATING_IDS] == sorted(at[:2]) and record[R.RECORD_FIELD_BOUNDARY_IDS] == []
    assert record["rule"] == R.GSM8K_TRIGGER_RULE and record["source"]["task"] == R.GSM8K_TASK
    assert all(tuple(f) == R.ITEM_FACT_FIELDS for f in record["item_facts"])
    assert {f["kind"] for f in record["item_facts"]} == {None, *R.ITEM_KINDS}
    assert tuple(subset) == C.ITEM_SUBSET_FIELDS and subset["schema"] == C.ITEM_SUBSET_SCHEMA == "rung_subset/1"
    assert subset["task"] == C.GSM8K_STANDARD_TASK and subset["rule"] == R.GSM8K_TRIGGER_RULE
    assert C.validate_item_subset(subset) == subset
    # ONE digest of the ids — the contract's (sorted unique ids' canonical JSON, 16 hex); the planner's newline-joined
    # sha is NOT it, and a subset file carrying a digest field of its own is refused (the digest rides cap_rung)
    assert C.cap_rung_identity(subset)["subset_sha16"] == C.item_subset_sha16(subset["ids"])
    assert C.item_subset_sha16(subset["ids"]) != hashlib.sha256("\n".join(subset["ids"]).encode("utf-8")).hexdigest()[:16]
    with pytest.raises(C.ItemSubsetError, match="not an Inspect row task"):
        C.validate_item_subset({**subset, "task": C.GSM8K_PAIRED_TASK_NAME})
    with pytest.raises(C.ItemSubsetError, match="schema"):
        C.validate_item_subset({**subset, "schema": "gsm8k_rung_subset/1"})
    with pytest.raises(C.ItemSubsetError, match="unknown \\['subset_sha16'\\]"):
        C.validate_item_subset({**subset, "subset_sha16": C.item_subset_sha16(subset["ids"])})
    with pytest.raises(C.ItemSubsetError, match="rule must"):
        C.validate_item_subset({**subset, "rule": ""})
    # the module names its exports where a reader looks first
    doc = R.__doc__ or ""
    for name in ("GSM8K_TRIGGER_RULE", "GSM8K_TRIGGER_RULE_3PCT_RETIRED", "GSM8K_TRIGGER_RULES_ADMITTED", "GSM8K_TRIGGER_PROVENANCE",
                 "NOT_TRIGGERED_EXITS", "TRIGGERED_TERMINAL_EXITS", "RECORD_FIELD_ELIGIBLE_IDS",
                 "ITEM_KINDS", "ITEM_FACT_FIELDS", "item_subset_sha16", "GSM8K_STANDARD_TASK", "U+2212"):
        assert name in doc, name


# --------------------------------------------------------------------------- #
# the trigger as a PARAMETER (1 % RAW at-cap everywhere)                       #
# --------------------------------------------------------------------------- #


def test_the_trigger_is_a_parameter_defaulting_to_the_harmonised_one_percent_and_the_rule_string_is_one_template(tmp_path, monkeypatch):
    """The ladder trigger is 1 % RAW at-cap for everything. The adapter takes the trigger as a PARAMETER
    (a schema-2 matrix declares it per row; the planner passes the row's) whose DEFAULT is the harmonised 1 %; the rule string is
    ONE template, so the retired 3 % pair's string is the same template at 0.03 (`gsm8k_trigger_rule(0.03) ==
    GSM8K_TRIGGER_RULE_3PCT_RETIRED`, byte for byte — the first-publication rungs hash it); the record, its rules, its per-control
    status and the subset carry the trigger they were decided under; the trigger is strict (>) at either value; a share outside
    (0, 1) is refused by name."""
    _forbid_peer_exit_rule(monkeypatch)
    assert R.GSM8K_TRIGGER_THRESHOLD == 0.01 == R.GSM8K_REPORT_LINE and R.GSM8K_TRIGGER_THRESHOLD_3PCT_RETIRED == 0.03
    assert R.gsm8k_trigger_rule(0.01) == R.GSM8K_TRIGGER_RULE and R.gsm8k_trigger_rule(0.03) == R.GSM8K_TRIGGER_RULE_3PCT_RETIRED
    three = R.gsm8k_trigger_rule(R.GSM8K_TRIGGER_THRESHOLD_3PCT_RETIRED)
    assert three.startswith("trigger iff n_at_cap / n > 0.03 (at-cap = stop_reason == max_tokens") and three.isascii() and three != R.GSM8K_TRIGGER_RULE
    assert three.replace("0.03", "0.01") == R.GSM8K_TRIGGER_RULE                   # ONE template, the threshold the only difference
    ids = _ids()
    at = ids[:3]                                                                     # 1.5 %: over the 1 % rule, under the retired 3 %
    root = _store(tmp_path / "s", cap=CAP, at_cap={i: CAP for i in at})
    rec1, sub1 = R.classify_store(root, texts=_texts(ids))                          # the default: the instrument's 1 %
    assert rec1["exit"] == L.ESCALATE and rec1["escalate"] is True and rec1["next_cap"] == RUNG
    assert rec1["threshold"] == 0.01 and rec1["rule"] == rec1["rules"]["trigger"] == R.GSM8K_TRIGGER_RULE and rec1["report_line"] == 0.01
    assert rec1["controls"][TASK]["threshold"] == 0.01 and rec1["controls"][TASK]["rules"]["trigger"] == R.GSM8K_TRIGGER_RULE and rec1["controls"][TASK]["escalate"] is True
    assert sub1 is not None and sub1["threshold"] == 0.01 and sub1["rule"] == R.GSM8K_TRIGGER_RULE and sorted(sub1["ids"]) == sorted(at) and sub1["cap"] == RUNG
    assert rec1["share_at_cap"] == 0.015 and rec1["share_at_cap_over_1pct"] is True and rec1["n_at_cap"] == 3
    rec3, sub3 = R.classify_store(root, texts=_texts(ids), threshold=R.GSM8K_TRIGGER_THRESHOLD_3PCT_RETIRED)   # a matrix declaring 3 %
    assert rec3["exit"] == L.EXIT_BELOW_THRESHOLD and sub3 is None and rec3["threshold"] == 0.03 and rec3["rule"] == three
    assert rec3["controls"][TASK]["threshold"] == 0.03 and rec3["controls"][TASK]["rules"]["trigger"] == three
    # everything but the trigger-bound fields is the same decision over the same store
    same = ("n", "n_at_cap", "share_at_cap", "n_eligible", "eligible_ids", "non_terminating_ids", "item_facts", "hops")
    assert {k: rec1[k] for k in same} == {k: rec3[k] for k in same}
    # strict `>`: 2 of 200 (1.0 %) does not trigger at 1 %; 3 does (above)
    two = _store(tmp_path / "two", cap=CAP, at_cap={i: CAP for i in ids[:2]})
    rec2, sub2 = R.classify_store(two, texts=_texts(ids), threshold=0.01)
    assert rec2["exit"] == L.EXIT_BELOW_THRESHOLD and sub2 is None and rec2["share_at_cap"] == 0.01 and rec2["share_at_cap_over_1pct"] is False
    # decide() alone, at either trigger
    hops = {L.HOP_DRIVER: 7200.0, L.HOP_PROXY_UPSTREAM: 7200.0}
    assert R.decide(n=200, n_at_cap=3, eligible_ids=at, cap=CAP, hops=hops)[0] == L.ESCALATE
    assert R.decide(n=200, n_at_cap=3, eligible_ids=at, cap=CAP, hops=hops, threshold=0.03)[0] == L.EXIT_BELOW_THRESHOLD
    # a trigger outside (0, 1) is refused by name, before any store is read
    for bad in (0.0, 1.0, -0.01, 3.0):
        with pytest.raises(R.InspectRungError, match="trigger_threshold"):
            R.classify_store(root, texts=_texts(ids), threshold=bad)
