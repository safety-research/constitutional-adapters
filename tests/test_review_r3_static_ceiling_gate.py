"""Review round 3, P2-a — the completed 16,384 static store is gated at the ceiling before the final fold.

Before: `static_esc16_t<T>` read each pair's 16,384 escalation record and folded the ladder; no gate ran over the
finished ceiling store, so a pair that escalated at 8,192 stayed `{"status": "escalating", "from": 8192, "to": 16384}`
in `static_ladder_t<T>.json` after the ceiling rung finished — never `exit_le_1pct` / `exit_ceiling` — and the exit
summary counted it as escalating. Now, after each SUCCESSFUL 16,384 escalation record, the phase runs
`f22s_static_gate` over `PRIVATE_RUN_ESC16` at cap 16,384 (`ladder_gate.py static`: the same bound decision the
8,192 rung was gated with; at the ceiling it is `exit_ceiling` while truncation remains above 1 %, `exit_le_1pct`
otherwise), appends its gate event, and only then folds. A record problem or a gate refusal is a counted failure
(TOTAL_FAIL -> rc 2) that files a rung-keyed `static_undecided` event before its `continue`, so the fold reports the pair
UNDECIDED at 16,384 with the cause and the reason — never a silent success and never an exit before the fold. (Review round 4,
P2-α: until then no event was appended, so `write_ladder.py` kept the 8,192 ESCALATE and the durable ladder said
`{"status": "escalating", "from": 8192, "to": 16384}` for a pair whose 16,384 store exists and whose ceiling decision is
undeterminable; the first version of this file asserted that wrong state.)

The phase-generated event is tested THROUGH THE REAL SHELL PATH: the sbatch's own post-driver block (extracted from
target_sampled.sbatch, run under bash with common_sampled.sh sourced, the real ladder_gate.py / write_ladder.py) over a
real 16,384 store — not a synthetic ceiling event handed to write_ladder.fold.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMPLED = REPO / "scripts" / "sdfbase" / "full22" / "sampled"
sys.path.insert(0, str(REPO / "tests"))

import test_review_r2_static_16384_rung as r2  # the sampled static store builders (after the tests/ path insert)

PY = sys.executable
RUN16 = "sdfbase_full22_llama8b_sampled_t0_cap16384"


def _esc16_body() -> str:
    src = (SAMPLED / "target_sampled.sbatch").read_text()
    i0 = src.index('elif [ "$PHASE_KIND" = static_esc16 ]; then')   # one dispatch chain (test_sampled_extend): an elif arm
    i1 = src.index('elif [ "$PHASE_KIND" = controls ]; then')
    return src[i0:i1]


def _post_driver_block() -> str:
    """The sbatch's lines from 'drivers done' to the standup stop: the record read, (the fix's gate), the fold."""
    body = _esc16_body()
    a = body.index('f22_log "phase STATIC_ESC16 drivers done')
    b = body.index("f22_standup_stop", a)
    return body[a:b]


def _escalation(n_after: int) -> dict:
    return {"attack": "pasttense", "defense": "none", "cap_from": 8192, "cap_to": 16384, "n_rows": 200,
            "n_at_cap_before": 5, "n_regenerated": 3, "n_at_cap_after": n_after, "n_copied": 197,
            "private_run": RUN16, "base_private_run": "sdfbase_full22_llama8b_sampled_t0_cap8192"}


def _earlier_events() -> list:
    """What the base and 8,192 phases filed for this pair: the 4,096 count, the 8,192 escalation record, and the
    8,192 gate's ESCALATE -> 16,384 (the state 'escalating from 8192 to 16384')."""
    return [
        {"kind": "static", "attack": "pasttense", "cell": "none", "rung": 4096, "n_at_cap": 12, "at": "2026-09-10T00:00:00+00:00"},
        {"kind": "static_escalation", "attack": "pasttense", "cell": "none", "problem": None, "at": "2026-09-10T00:00:01+00:00",
         "escalation": {"attack": "pasttense", "defense": "none", "cap_from": 4096, "cap_to": 8192, "n_rows": 200,
                        "n_at_cap_before": 12, "n_regenerated": 12, "n_at_cap_after": 5}},
        {"kind": "static", "attack": "pasttense", "cell": "none", "rung": 8192, "n_at_cap": 5, "root": "/r8192",
         "decision": "escalate", "next_cap": 16384, "n_truncated": 3, "n_loop": 1, "n_repetitive": 1, "n_boundary": 0,
         "share_truncated": 0.015, "transport_ok": True, "n_items_escalate": 3, "gate": "/g8192.gate.json",
         "at": "2026-09-10T00:00:02+00:00"},
    ]


def _run_phase_tail(tmp_path, rows, *, escalation, progress_tamper=None):
    root = tmp_path / "private"
    shard = r2._write_static(root, RUN16, rows, escalation=escalation)
    if progress_tamper is not None:
        progress_tamper(shard)
    serve = tmp_path / "serve"
    (serve / "static_esc16").mkdir(parents=True)
    (serve / "static_esc16" / "endpoint.json").write_text(json.dumps({"proxy_url": "http://127.0.0.1:1/v1", "upstream_timeout_s": 7200}))
    ladders = tmp_path / "ladders"
    ladders.mkdir()
    (ladders / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in _earlier_events()))
    script = (f'source "{SAMPLED}/common_sampled.sh"\nset -uo pipefail\n'
              'f22_log(){ echo "LOG $*" >&2; }\n'
              'FAMS=(pasttense); CELLS=(none); NCELLS=1; NFAIL=0; TOTAL_FAIL=0\n'
              'declare -A ITEMS16=(["pasttense|none"]="/x/items.jsonl")\n'
              f"{_post_driver_block()}\n"
              'echo "TOTAL_FAIL=$TOTAL_FAIL"\n')
    env = {**os.environ, "MPY": PY, "REPO_DIR": str(REPO), "SERVE_DIR": str(serve), "LADDER_DIR": str(ladders),
           "WAVE_ROOT": str(tmp_path), "PRIVATE_ROOT": str(root), "TAG": "llama8b", "TRIAL": "0", "KEY": "llama8b",
           "PRIVATE_RUN_BASE": "sdfbase_full22_llama8b_sampled_t0", "PRIVATE_RUN_ESC": "sdfbase_full22_llama8b_sampled_t0_cap8192",
           "PRIVATE_RUN_ESC16": RUN16}
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, cwd=REPO, env=env, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    total_fail = int(r.stdout.strip().splitlines()[-1].split("=")[1])
    ladder = json.loads((ladders / "static_ladder_t0.json").read_text())
    events = [json.loads(l) for l in (ladders / "events.jsonl").read_text().splitlines() if l.strip()]
    return total_fail, ladder, events, r.stderr, ladders, serve


def test_a_pair_escalating_at_8192_leaves_the_ladder_at_the_ceiling_with_the_gates_exit_ceiling(tmp_path):
    rows = r2._rows(cap=16384)      # 1 loop + 1 repetitive + 3 coherent at cap of 200: 1.5 % truncated at the ceiling
    total_fail, lad, events, log, ladders, serve = _run_phase_tail(tmp_path, rows, escalation=_escalation(5))
    assert total_fail == 0, log
    rec = lad["cells"]["none"]["pasttense"]
    assert rec["left_ladder_at"] == {"status": "left", "rung": 16384, "exit": "exit_ceiling", "n_truncated": 3, "n_loop": 1,
                                     "n_repetitive": 1, "n_boundary": 0, "unresolved_truncation": True}, rec["left_ladder_at"]
    assert rec["n_at_cap"] == {"4096": 12, "8192": 5, "16384": 5} and rec["reached_zero_at"] == "16384 + residual 5"
    assert set(rec["decisions"]) == {"8192", "16384"} and rec["decisions"]["16384"]["decision"] == "exit_ceiling"
    assert rec["decisions"]["16384"]["next_cap"] is None and rec["decisions"]["16384"]["n_items_escalate"] == 0
    assert rec["escalations"]["16384"]["cap_to"] == 16384 and rec["problem"] is None
    assert lad["summary"]["exits"] == {"exit_le_1pct": 0, "exit_ceiling": 1, "transport_refused": 0, "escalating": 0, "undecided": 0}
    # the gate event came from the REAL gate over the ceiling store: its record is under gates/static_cap16384 and names the run
    gate_ev = [e for e in events if e.get("kind") == "static" and e.get("rung") == 16384]
    assert len(gate_ev) == 1 and gate_ev[0]["decision"] == "exit_ceiling" and gate_ev[0]["n_at_cap"] == 5
    gate_path = Path(gate_ev[0]["gate"])
    assert gate_path == ladders / "gates" / "static_cap16384" / "pasttense__none.gate.json" and gate_path.is_file()
    g = json.loads(gate_path.read_text())
    assert g["kind"] == "static" and g["cap"] == 16384 and g["private_run"] == RUN16 and g["decision"] == "exit_ceiling"
    assert g["decision_tool"]["at_ceiling"] is True and g["endpoint"] == str(serve / "static_esc16" / "endpoint.json")
    # order: the escalation record, then the gate's event, then the fold (the fold is the last thing the block does)
    i_rec = next(i for i, e in enumerate(events) if e.get("kind") == "static_escalation" and (e.get("escalation") or {}).get("cap_to") == 16384)
    assert events.index(gate_ev[0]) > i_rec
    assert (serve / "static_ladder_t0.json").read_text() == (ladders / "static_ladder_t0.json").read_text()
    assert "EXIT exit_ceiling" in log and "UNRESOLVED at the ceiling: 3 truncated items" in log


def test_a_pair_resolved_at_the_ceiling_leaves_with_exit_le_1pct(tmp_path):
    rows = r2._rows(cap=16384, loops=0, repetitive=0, coherent=0)     # every item terminated: nothing at cap
    total_fail, lad, _, log, *_ = _run_phase_tail(tmp_path, rows, escalation=_escalation(0))
    assert total_fail == 0, log
    rec = lad["cells"]["none"]["pasttense"]
    assert rec["left_ladder_at"]["status"] == "left" and rec["left_ladder_at"]["rung"] == 16384
    assert rec["left_ladder_at"]["exit"] == "exit_le_1pct" and rec["left_ladder_at"]["unresolved_truncation"] is False
    assert rec["n_at_cap"]["16384"] == 0 and rec["reached_zero_at"] == "16384"
    assert lad["summary"]["exits"]["exit_le_1pct"] == 1 and lad["summary"]["exits"]["escalating"] == 0


def _the_one_undecided_event(events):
    und = [e for e in events if e.get("kind") == "static_undecided"]
    assert len(und) == 1, und
    assert und[0]["attack"] == "pasttense" and und[0]["cell"] == "none" and und[0]["rung"] == 16384 and und[0]["at"]
    return und[0]


def test_a_record_problem_or_a_gate_refusal_at_the_ceiling_is_a_counted_failure_folded_as_undecided(tmp_path):
    # no escalation.json: the record is the problem; the gate never runs over a store that did not finish. The pair is
    # UNDECIDED at 16,384 (the rung ran and left no record), never "escalating from 8192 to 16384" — that would claim an
    # active escalation to a rung that has already run (P2-α)
    rows = r2._rows(cap=16384)
    total_fail, lad, events, log, ladders, _ = _run_phase_tail(tmp_path / "norec", rows, escalation=None)
    assert total_fail == 1 and "escalation record PROBLEM" in log
    rec = lad["cells"]["none"]["pasttense"]
    und = _the_one_undecided_event(events)
    assert und["cause"] == "record_problem" and und["reason"].startswith("escalation.json not found under ")
    assert rec["left_ladder_at"] == {"status": "undecided", "rung": 16384, "cause": "record_problem", "reason": und["reason"]}
    assert rec["undecided"] == {"16384": {"cause": "record_problem", "reason": und["reason"], "at": und["at"]}}
    assert rec["reached_zero_at"] == "incomplete" and rec["problem"] == und["reason"]
    assert set(rec["decisions"]) == {"8192"}      # the 8,192 ESCALATE stays as history; it no longer governs
    assert not (ladders / "gates" / "static_cap16384").exists()
    assert not any(e.get("kind") == "static" and e.get("rung") == 16384 for e in events)
    assert lad["summary"]["exits"] == {"exit_le_1pct": 0, "exit_ceiling": 0, "transport_refused": 0, "escalating": 0, "undecided": 1}
    # the event is filed AFTER the escalation record (the fold reads both) and the fold is last
    i_rec = next(i for i, e in enumerate(events) if e.get("kind") == "static_escalation")
    assert events.index(und) > i_rec
    # a short store (progress claims a variant the rows lack): the gate REFUSES; counted, folded as UNDECIDED at the
    # ceiling with the gate's own refusal as the reason — the store exists, its exit is undeterminable
    def tamper(shard):
        recs = [json.loads(l) for l in (shard / "progress.jsonl").read_text().splitlines()]
        recs[0]["n_variants"] += 1
        (shard / "progress.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    total_fail, lad, events, log, ladders, _ = _run_phase_tail(tmp_path / "refused", rows, escalation=_escalation(5), progress_tamper=tamper)
    assert total_fail == 1 and "REFUSED" in log and "undeterminable" in log
    rec = lad["cells"]["none"]["pasttense"]
    und = _the_one_undecided_event(events)
    assert und["cause"] == "gate_refused" and und["reason"].startswith("ladder_gate.py static REFUSED")
    assert "[ladder-gate] REFUSED" in und["reason"] and "n_variants" in und["reason"]   # the real gate's own words
    assert rec["left_ladder_at"] == {"status": "undecided", "rung": 16384, "cause": "gate_refused", "reason": und["reason"]}
    assert rec["undecided"] == {"16384": {"cause": "gate_refused", "reason": und["reason"], "at": und["at"]}}
    assert rec["n_at_cap"]["16384"] == 5 and "16384" not in (rec.get("decisions") or {}) and rec["problem"] is None
    assert rec["escalations"]["16384"]["cap_to"] == 16384    # the record was read; only the gate refused
    assert lad["summary"]["exits"] == {"exit_le_1pct": 0, "exit_ceiling": 0, "transport_refused": 0, "escalating": 0, "undecided": 1}


def test_a_refusal_below_a_later_decision_is_superseded_and_a_decision_at_the_same_rung_governs():
    """The fold's rule for the undecided marks (write_ladder.left_ladder_at): a decision at a rung governs it (a resumed
    job that re-ran the gate successfully supersedes an earlier refusal there); a mark below the highest decided rung
    is stale history; a mark ABOVE every decided rung, with no decision there, makes the pair undecided at that rung."""
    import write_ladder as wl  # after the tests/ path insert; r2 already put SAMPLED on sys.path
    rungs = wl.RUNGS["static"]
    esc = {"decision": "escalate", "next_cap": 16384}
    mark = {"cause": "gate_refused", "reason": "r", "at": "t"}
    assert wl.left_ladder_at({}, rungs) == {"status": "undecided"}
    assert wl.left_ladder_at({}, rungs, None) == {"status": "undecided"}
    assert wl.left_ladder_at({"8192": esc}, rungs, {"16384": mark}) == {"status": "undecided", "rung": 16384, "cause": "gate_refused", "reason": "r"}
    assert wl.left_ladder_at({}, rungs, {"8192": mark}) == {"status": "undecided", "rung": 8192, "cause": "gate_refused", "reason": "r"}
    # the same rung refused once and decided later: the decision governs
    assert wl.left_ladder_at({"8192": esc, "16384": {"decision": "exit_ceiling", "n_truncated": 3}}, rungs, {"16384": mark})["status"] == "left"
    # a stale mark below a later decision: superseded
    assert wl.left_ladder_at({"8192": esc, "16384": {"decision": "exit_le_1pct"}}, rungs, {"8192": mark})["exit"] == "exit_le_1pct"
    assert wl.left_ladder_at({"8192": esc}, rungs, {"4096": mark}) == {"status": "escalating", "from": 8192, "to": 16384}


def test_the_phase_gates_the_ceiling_store_after_its_record_and_before_the_final_fold():
    body = _esc16_body()
    block = _post_driver_block()
    i_rec = block.index('f22s_escalation_event "$fam" "$cell" "$PRIVATE_RUN_ESC16"')
    i_gate = block.index('f22s_static_gate "$PRIVATE_RUN_ESC16" "$fam" "$cell" "$F22S_STATIC_CAP_ESC16" "$SERVE_DIR/static_esc16/endpoint.json"')
    i_ev = block.index('f22s_event "$(f22s_gate_event "$gate")"')
    i_fold = block.index('f22s_write_ladder static "$LADDER_DIR/static_ladder_t${TRIAL}.json"')
    assert i_rec < i_gate < i_ev < i_fold
    assert 'gate="$LADDER_DIR/gates/static_cap${F22S_STATIC_CAP_ESC16}/${fam}__${cell}.gate.json"' in block
    # a record problem skips the gate (continue), a refusal is counted — neither exits before the fold
    assert block.count("TOTAL_FAIL=$((TOTAL_FAIL + 1))") >= 2 and "exit 1" not in block
    # (round 4, P2-α) both counted failures file the rung-keyed undecided event BEFORE their `continue`, so the fold sees them:
    # the record problem's right after the record read, the gate's right after the refused gate call
    und_rec = 'f22s_undecided_event "$fam" "$cell" "$F22S_STATIC_CAP_ESC16" record_problem'
    und_gate = 'f22s_undecided_event "$fam" "$cell" "$F22S_STATIC_CAP_ESC16" gate_refused'
    assert block.count(und_rec) == 1 and block.count(und_gate) == 1
    assert i_rec < block.index(und_rec) < i_gate < block.index(und_gate) < i_ev < i_fold
    assert block.count("continue") == 3     # the ITEMS16 skip, the record problem, the gate refusal — nothing else short-circuits
    # the gate's items loop is unchanged: the driver launch and the record read (the round-2 pin holds)
    assert body.count('[ -n "${ITEMS16[$fam|$cell]:-}" ] || continue') == 2
    assert body.count('f22s_write_ladder static "$LADDER_DIR/static_ladder_t${TRIAL}.json"') == 2
