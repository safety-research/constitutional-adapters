"""Review round 1 (2026-09-10), P1-5 — the cap-ladder verdict GOVERNS the controls escalation loop.

Before: target_sampled.sbatch escalated a (cell, control) on n_at_cap > 0 (loops included), knew only the 2048 / 8192
rungs, filtered no items, consulted no transport verdict, and labelled an 8192 residual "genuine non-termination".
Now: ladder_gate.py issues ONE bound decision per (cell, control) at the rung just measured — ladder_decide.py's
item-filtered 1 % TRUNCATED verdict (loops / repetitive / boundary excluded, 16,384 ceiling) bound to the transport
approval for the rung above — and the loop climbs ONLY on its ESCALATE, with exactly its items (--items), re-checking
the transport against the new standup's record before any driver; write_ladder.py folds the decisions and reports
where each pair LEFT the ladder, with unresolved truncation at the ceiling distinct from loops.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SAMPLED = REPO / "scripts" / "sdfbase" / "full22" / "sampled"
sys.path.insert(0, str(SAMPLED))

import ladder_gate as lg  # noqa: E402
import write_ladder as wl  # noqa: E402

# the classifier's text classes (tests/test_ladder_decide.py's constants; rep_ratio = 4-gram repetition)
LOOP_RR = "a b c d " * 6                                    # 0.81 > 0.5 -> loop
REPETITIVE = "a b c d e f g h a b c d e f g h"             # 0.385 -> repetitive (never escalates)
COHERENT = "one two three four five six seven eight nine ten eleven twelve"   # 0.0 -> cap-coherent
CTL = "xstest"
SPOOL = "gens_xstest_v2.jsonl"
PRESET = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}


def _write_controls(root, cell, items, *, cap=512):
    """A complete sampled served-controls store for CELL under ROOT: items = [(item_id, text, finish_reason,
    completion_tokens)] (the shape at_cap_controls / ladder_decide read)."""
    d = root / "cells" / f"{cell}__controls__abc"
    (d / "private").mkdir(parents=True, exist_ok=True)
    gen = {"max_new_tokens": cap, "do_sample": True, **PRESET, "arm": "vendor_recommended", "seed_base": 1,
           "trial": 0, "batch_size": 8}
    (d / "manifest.json").write_text(json.dumps({"cell_id": cell, "intervention_hash": "ii" * 8,
                                                 "resolved_config": {"driver": "served_capability_run_v2", "gen": gen}}))
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
        "decode": {"do_sample": True, **PRESET, "cap": cap, "arm": "vendor_recommended", "seed_base": 1, "trial": 0},
        "n_at_cap": len(at_cap), "at_cap_item_ids": at_cap}}))
    return d


def _items(n=100, *, loops=1, repetitive=1, coherent=3, cap=512):
    out = []
    for i in range(n):
        iid = f"it{i:04d}"
        if i < loops:
            out.append((iid, LOOP_RR, "length", cap))
        elif i < loops + repetitive:
            out.append((iid, REPETITIVE, "length", cap))
        elif i < loops + repetitive + coherent:
            out.append((iid, COHERENT, "length", cap))
        else:
            out.append((iid, f"terminated reply {i}", "stop", 40))
    return out


def _endpoint(tmp_path, **rec):
    p = tmp_path / "endpoint.json"
    p.write_text(json.dumps({"proxy_url": "http://127.0.0.1:1/v1", **rec}))
    return p


def _argv(root, cap, tmp_path, endpoint, *extra):
    return ["decide", "--out-root", str(root), "--cell", "cellA", "--control", CTL, "--cap", str(cap),
            "--endpoint", str(endpoint), "--out", str(tmp_path / "gate.json"),
            "--decision-out", str(tmp_path / "decision.json"), "--items-out", str(tmp_path / "items.jsonl"), *extra]


def _run(argv):
    r = subprocess.run([sys.executable, str(SAMPLED / "ladder_gate.py"), *argv], capture_output=True, text=True, cwd=REPO)
    return r.returncode, r.stdout.strip(), r.stderr


# --------------------------------------------------------------------------- the bound decision

def test_the_gate_escalates_exactly_the_truncated_items_with_the_transports_approval(tmp_path):
    root = tmp_path / "cap512_t0"
    _write_controls(root, "cellA", _items())            # 1 loop + 1 repetitive + 3 coherent at cap of 100 -> 3 % truncated
    ep = _endpoint(tmp_path, upstream_timeout_s=7200)
    rc, line, err = _run(_argv(root, 512, tmp_path, ep))
    assert rc == 0, err
    assert line == f"ESCALATE 2048 {tmp_path / 'items.jsonl'}"
    g = json.loads((tmp_path / "gate.json").read_text())
    assert g["schema"] == "ladder_gate/1" and g["decision"] == "escalate" and g["next_cap"] == 2048
    assert g["counts"] == {"n_responses": 100, "n_at_cap": 5, "n_loop": 1, "n_repetitive": 1, "n_boundary": 0,
                           "n_boundary_any": 0, "n_truncated": 3, "share_truncated": pytest.approx(0.03)}
    assert g["n_items_escalate"] == 3 and g["items"] == str(tmp_path / "items.jsonl")
    assert g["transport"]["ok"] is True and g["transport"]["next_cap"] == 2048
    assert g["hop_timeouts_s"] == {"driver": 7200.0, "proxy_upstream": 7200.0}
    assert g["decision_tool"]["rule"] == "item_filtered_v1" and g["decision_tool"]["decision"] == "escalate"
    assert g["decision_tool"]["record_sha256"] == hashlib.sha256((tmp_path / "decision.json").read_bytes()).hexdigest()
    assert g["driver_record"]["present"] is False            # the synthetic store carries no cap_ladder.json; recorded, not required
    # the items file is the driver's --items: loops and repetitive rows excluded, the three truncated rows escalate
    rows = [json.loads(l) for l in (tmp_path / "items.jsonl").read_text().splitlines()]
    header = rows[0]["__ladder_items__"]
    assert header["kind"] == "controls" and header["cap"] == 512 and header["control"] == CTL and header["n_escalate"] == 3
    by = {r["item_id"]: r for r in rows[1:]}
    assert by["it0000"]["excluded_as"] == "loop" and by["it0001"]["excluded_as"] == "repetitive"
    assert all(by[f"it{i:04d}"]["escalate"] for i in (2, 3, 4))


def test_the_gate_refuses_the_climb_when_the_transport_is_undeterminable_or_too_small(tmp_path):
    root = tmp_path / "cap8192_t0"
    _write_controls(root, "cellA", _items(cap=8192), cap=8192)      # 3 % truncated at 8192: the verdict says climb to 16384
    # (a) the standup recorded NO upstream timeout: undeterminable -> never a climb on absent evidence
    rc, line, _ = _run(_argv(root, 8192, tmp_path, _endpoint(tmp_path)))
    assert rc == 0 and line == "EXIT transport_refused"
    g = json.loads((tmp_path / "gate.json").read_text())
    assert g["decision"] == "transport_refused" and g["decision_tool"]["decision"] == "escalate"
    assert g["next_cap"] is None and g["items"] is None and g["n_items_escalate"] == 0
    assert g["transport"]["ok"] is False and "undeterminable" in g["transport"]["reason"]
    assert g["hop_timeouts_s"] == {"driver": 7200.0, "proxy_upstream": None}
    # (b) the OLD proxy default (600 s) cannot carry a 16,384-token draw at the 4 tok/s floor x 1.5 (6,144 s)
    rc, line, _ = _run(_argv(root, 8192, tmp_path, _endpoint(tmp_path, upstream_timeout_s=600)))
    assert rc == 0 and line == "EXIT transport_refused"
    g = json.loads((tmp_path / "gate.json").read_text())
    assert g["transport"]["binding_hop"] == "proxy_upstream" and g["transport"]["required_s"] == pytest.approx(6144.0)
    assert "REFUSED" in g["transport"]["reason"] and "--upstream-timeout-s" in g["transport"]["reason"]
    # (c) no endpoint record at all
    rc, line, _ = _run(_argv(root, 8192, tmp_path, tmp_path / "missing.json"))
    assert rc == 0 and line == "EXIT transport_refused"
    assert json.loads((tmp_path / "gate.json").read_text())["endpoint_present"] is False
    # (d) the wave's standup (7,200 s) carries it
    rc, line, _ = _run(_argv(root, 8192, tmp_path, _endpoint(tmp_path, upstream_timeout_s=7200)))
    assert rc == 0 and line.startswith("ESCALATE 16384 ")


def test_loops_alone_never_escalate_and_stay_a_distinct_class(tmp_path):
    root = tmp_path / "cap512_t0"
    _write_controls(root, "cellA", _items(loops=2, repetitive=1, coherent=0))   # 3 at cap, ZERO truncated
    rc, line, _ = _run(_argv(root, 512, tmp_path, _endpoint(tmp_path, upstream_timeout_s=7200)))
    assert rc == 0 and line == "EXIT exit_le_1pct"
    g = json.loads((tmp_path / "gate.json").read_text())
    assert g["counts"]["n_at_cap"] == 3 and g["counts"]["n_loop"] == 2 and g["counts"]["n_repetitive"] == 1
    assert g["counts"]["n_truncated"] == 0 and g["items"] is None and g["transport"] is None
    assert "never escalated" in g["classes"]["loop"]


def test_at_the_ceiling_a_truncated_residual_is_unresolved_never_non_termination(tmp_path):
    root = tmp_path / "cap16384_t0"
    _write_controls(root, "cellA", _items(cap=16384), cap=16384)
    rc, line, _ = _run(_argv(root, 16384, tmp_path, _endpoint(tmp_path, upstream_timeout_s=7200)))
    assert rc == 0 and line == "EXIT exit_ceiling"
    g = json.loads((tmp_path / "gate.json").read_text())
    assert g["next_cap"] is None and g["decision_tool"]["at_ceiling"] is True and g["counts"]["n_truncated"] == 3
    assert "unresolved" in g["classes"]["truncated"] and "non-termination" not in json.dumps(g)


def test_a_mismatched_cap_or_a_non_rung_is_a_refusal_never_a_decision(tmp_path):
    root = tmp_path / "cap512_t0"
    _write_controls(root, "cellA", _items())
    ep = _endpoint(tmp_path, upstream_timeout_s=7200)
    rc, line, err = _run(_argv(root, 2048, tmp_path, ep))          # the store is at 512
    assert rc == 2 and line == "" and "REFUSED" in err and "at cap 512, not --cap 2048" in err
    assert not (tmp_path / "gate.json").exists()
    rc, line, err = _run(_argv(root, 1000, tmp_path, ep))          # not a rung
    assert rc == 2 and line == "" and "not a rung" in err


def test_the_transport_recheck_against_the_new_standups_record(tmp_path):
    ok = _endpoint(tmp_path, upstream_timeout_s=7200)
    rc, line, _ = _run(["transport", "--endpoint", str(ok), "--next-cap", "16384"])
    assert rc == 0 and line.startswith("OK ")
    small = tmp_path / "small.json"; small.write_text(json.dumps({"upstream_timeout_s": 600}))
    rc, line, _ = _run(["transport", "--endpoint", str(small), "--next-cap", "16384"])
    assert rc == 3 and line.startswith("REFUSED ")
    rc, line, _ = _run(["transport", "--endpoint", str(tmp_path / "none.json"), "--next-cap", "2048"])
    assert rc == 3 and "undeterminable" in line


def test_the_driver_timeout_is_read_from_the_driver_not_a_second_copy():
    # the driver binds its client timeout to the proxy engine's constant (main: `AIOHTTP_TOTAL_TIMEOUT_S = pe.AIOHTTP_TOTAL_TIMEOUT_S`);
    # the gate reads THAT value, never a literal of its own
    src = (REPO / "scripts/served_capability_run.py").read_text()
    line = [l for l in src.splitlines() if l.startswith("AIOHTTP_TOTAL_TIMEOUT_S")][0]
    assert line.split("=", 1)[1].strip().split()[0].endswith("AIOHTTP_TOTAL_TIMEOUT_S")
    from sisteer.capeval.serving import proxy_engine as pe
    assert lg.driver_timeout_default() == float(pe.AIOHTTP_TOTAL_TIMEOUT_S) == 7200.0


# --------------------------------------------------------------------------- the ladder record

def test_write_ladder_folds_the_gate_decisions_and_reports_where_each_pair_left(tmp_path):
    assert wl.RUNGS["controls"] == [512, 2048, 8192, 16384]
    ev = tmp_path / "events.jsonl"; out = tmp_path / "ladder.json"
    dec = lambda d, **k: {"decision": d, "next_cap": k.get("next_cap"), "n_truncated": k.get("nt", 0), "n_loop": k.get("nl", 0),
                          "n_repetitive": 0, "n_boundary": 0, "share_truncated": k.get("nt", 0) / 100, "transport_ok": k.get("tr"),
                          "n_items_escalate": k.get("nt", 0) if d == "escalate" else 0, "gate": "/g"}
    rows = [
        {"kind": "controls", "cell": "c1", "control": "xstest", "rung": 512, "n_at_cap": 5, "root": "/r512", **dec("escalate", next_cap=2048, nt=3, nl=2, tr=True)},
        {"kind": "controls", "cell": "c1", "control": "xstest", "rung": 2048, "n_at_cap": 2, "root": "/r2048", **dec("exit_le_1pct", nt=0, nl=2)},
        {"kind": "controls", "cell": "c1", "control": "or_bench_hard_1k", "rung": 512, "n_at_cap": 9, "root": "/r512", **dec("escalate", next_cap=2048, nt=9, tr=True)},
        {"kind": "controls", "cell": "c1", "control": "or_bench_hard_1k", "rung": 2048, "n_at_cap": 6, "root": "/r2048", **dec("escalate", next_cap=8192, nt=6, tr=True)},
        {"kind": "controls", "cell": "c1", "control": "or_bench_hard_1k", "rung": 8192, "n_at_cap": 4, "root": "/r8192", **dec("escalate", next_cap=16384, nt=4, tr=True)},
        {"kind": "controls", "cell": "c1", "control": "or_bench_hard_1k", "rung": 16384, "n_at_cap": 2, "root": "/r16384", **dec("exit_ceiling", nt=2)},
        {"kind": "controls", "cell": "c2", "control": "xstest", "rung": 512, "n_at_cap": 3, "root": "/r512", **dec("transport_refused", nt=3, tr=False)},
        {"kind": "controls", "cell": "c2", "control": "or_bench_hard_1k", "rung": 512, "n_at_cap": 0, "root": "/r512"},   # pre-gate event: no decision yet
    ]
    ev.write_text("".join(json.dumps(r) + "\n" for r in rows))
    r = subprocess.run([sys.executable, str(SAMPLED / "write_ladder.py"), "--events", str(ev), "--out", str(out), "--kind", "controls",
                        "--tag", "llama8b", "--trial", "0"], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lad = json.loads(out.read_text())
    assert lad["rungs"] == [512, 2048, 8192, 16384] and "left_ladder_at" in lad["exit_rule"] and "16,384 ceiling" in lad["exit_rule"]
    c = lad["cells"]
    x = c["c1"]["xstest"]
    assert x["decisions"]["512"]["decision"] == "escalate" and x["decisions"]["512"]["next_cap"] == 2048
    assert x["left_ladder_at"] == {"status": "left", "rung": 2048, "exit": "exit_le_1pct", "n_truncated": 0, "n_loop": 2,
                                   "n_repetitive": 0, "n_boundary": 0, "unresolved_truncation": False}
    # the RAW count never reached zero (2 loops at 2048); the count history closes on the gate's EXIT decision at the rung
    # (formatted "<rung> + residual n (<decision>)"), so it names where the pair left and agrees with left_ladder_at
    assert x["reached_zero_at"] == "2048 + residual 2 (exit_le_1pct)"
    o = c["c1"]["or_bench_hard_1k"]
    assert o["left_ladder_at"]["exit"] == "exit_ceiling" and o["left_ladder_at"]["unresolved_truncation"] is True
    assert o["reached_zero_at"] == "16384 + residual 2" and o["rungs_attempted"] == [512, 2048, 8192, 16384]
    assert c["c2"]["xstest"]["left_ladder_at"]["exit"] == "transport_refused"
    assert c["c2"]["or_bench_hard_1k"]["left_ladder_at"] == {"status": "undecided"} and c["c2"]["or_bench_hard_1k"]["reached_zero_at"] == "512"
    assert lad["summary"]["exits"] == {"exit_le_1pct": 1, "exit_ceiling": 1, "transport_refused": 1, "escalating": 0, "undecided": 1}
    assert lad["summary"]["left_at"] == {"512": 1, "2048": 0, "8192": 0, "16384": 0} and lad["summary"]["residual_at_top"] == 1
    # an unknown decision word is refused, never folded
    ev.write_text(json.dumps({"kind": "controls", "cell": "c", "control": "xstest", "rung": 512, "n_at_cap": 1, "decision": "maybe"}) + "\n")
    r = subprocess.run([sys.executable, str(SAMPLED / "write_ladder.py"), "--events", str(ev), "--out", str(tmp_path / "x.json"),
                        "--kind", "controls", "--tag", "t", "--trial", "0"], capture_output=True, text=True)
    assert r.returncode != 0 and "unknown ladder decision" in r.stderr


def test_left_ladder_at_reports_an_escalating_pair_between_rungs():
    assert wl.left_ladder_at({"512": {"decision": "escalate", "next_cap": 2048}}, wl.RUNGS["controls"]) == \
        {"status": "escalating", "from": 512, "to": 2048}


# --------------------------------------------------------------------------- the loop

def test_the_sbatch_controls_loop_climbs_only_on_the_gates_escalate_with_its_items():
    src = (SAMPLED / "target_sampled.sbatch").read_text()
    common = (SAMPLED / "common_sampled.sh").read_text()
    assert "F22S_CONTROLS_RUNGS=(512 2048 8192 16384)" in common
    assert 'for RUNG in "${F22S_CONTROLS_RUNGS[@]:1}"; do' in src and "for RUNG in 2048 8192; do" not in src
    # the decision is the gate's, never n_at_cap > 0
    # 2026-09-13: the previous rung's server is STOPPED when this gate runs and mt_serving_up.py unlinks its live endpoint.json at
    # teardown, so the gate reads EP_PREV — the 512 record resolved before the loop, else the rung's READY copy (f22s_standup_endpoint);
    # an absent record still refuses (sampled-llama-grade; every llama pair had left `transport_refused` on the unlinked file)
    assert 'f22s_gate_pairs "$PREV_RUNG" "$PREV_ROOT" "$EP_PREV" "$RUNG"' in src
    assert 'if [ "$PREV_RUNG" = 512 ]; then EP_PREV="$EP_512"; else EP_PREV="$(f22s_standup_endpoint "$SERVE_DIR/controls_${PREV_RUNG}")"; fi' in src
    # ... in the CONTROLS arm (the ladder this test is about); lane G2's ladder_extend arm below it reads the at-cap
    # COUNT at the 16,384 ceiling as a record ("residual at the CEILING: recorded, never escalated further"), which is
    # not a climb decision
    controls_arm = src[src.index('elif [ "$PHASE_KIND" = controls ]; then'):src.index('elif [ "$PHASE_KIND" = ladder_extend ]; then')]
    assert '[ "$n" -gt 0 ] && keep=' not in controls_arm and "f22s_at_cap_controls" not in controls_arm
    assert "genuine non-termination" not in src and "genuine non-termination" not in common
    assert 'ESCALATE)' in src and 'EXIT)' in src and 'ITEMS_OF["$cell/$ctl"]="$arg2"' in src
    assert 'REFUSING: the gate escalates $cell/$ctl to $arg1 but the loop\'s next rung is' in src
    # the driver regenerates exactly the gate's items
    i_drv = src.index('f22s_launch_cell "controls${RUNG}__${cell}"')
    assert 'ITEMS_ARGS+=(--items "${ITEMS_OF["$cell/$ctl"]}")' in src[:i_drv]
    assert '--controls-cap "$RUNG" --escalate-from-root "$PREV_ROOT" --floor-sample "$F22S_FLOOR_SAMPLE_K" "${ITEMS_ARGS[@]}"' in src
    # the transport is re-checked against the NEW rung's record after the standup and before the positive control / drivers
    i_up = src.index('F22_LOAD_CHECK_CAP="$RUNG" f22_standup attacks "$SERVE_DIR/controls_${RUNG}"')
    i_tr = src.index('f22s_ladder_transport "$SERVE_DIR/controls_${RUNG}/endpoint.json" "$RUNG"')
    i_pc = src.index('f22s_positive_control "controls_${RUNG}"')
    assert i_up < i_tr < i_pc < i_drv
    # the ceiling: the pairs still on the ladder after the top rung get their final (exit) decision there — through the top rung's
    # SURVIVING endpoint record (the live endpoint.json is unlinked at teardown; 2026-09-14), and under a `:<rung>` ceiling below
    # 16,384 the gate runs toward the real next rung (ESCALATE recorded, nothing climbed), never as if that rung were the ceiling
    assert 'if [ "$LAST_RAN" = "${F22S_CONTROLS_RUNGS[-1]}" ] && [ "$NRUNG" -gt 0 ]; then' in src
    assert 'EP_LAST="$(f22s_standup_endpoint "$SERVE_DIR/controls_${LAST_RAN}")"' in src
    assert 'f22s_gate_pairs "$LAST_RAN" "$(ROOT_OF "$LAST_RAN")" "$EP_LAST" ""' in src
    assert 'f22s_gate_pairs "$LAST_RAN" "$(ROOT_OF "$LAST_RAN")" "$EP_LAST" "$F22S_CONTROLS_CEILING_ABOVE"' in src
    assert 'f22s_gate_pairs "$LAST_RAN" "$(ROOT_OF "$LAST_RAN")" "$SERVE_DIR/controls_${LAST_RAN}/endpoint.json" ""' not in src
    # the helpers: the gate line grammar is enforced in the shell too, and a refusal is never a number
    for fn in ("f22s_ladder_gate(){", "f22s_gate_event(){", "f22s_gate_summary(){", "f22s_ladder_transport(){"):
        assert fn in common, fn
    assert 'ESCALATE\\ [0-9]*\\ /*|EXIT\\ exit_le_1pct|EXIT\\ exit_ceiling|EXIT\\ transport_refused) ;;' in common
    assert "ladder_gate.py\" decide --out-root" in common and "--sibling-out-root \"$sib\"" in common
    assert 'sib="$WAVE_ROOT/sampled/controls_cap$4/${TAG}_t$((1 - TRIAL))"' in common


def test_the_shell_gate_helper_accepts_only_a_bound_line(tmp_path):
    """f22s_ladder_gate echoes the gate's line verbatim and returns 2 on anything else (a refusal, a stray number)."""
    stub = tmp_path / "mpy"
    stub.write_text("#!/bin/bash\ncat \"$STUB_OUT\"\nexit \"${STUB_RC:-0}\"\n"); stub.chmod(0o755)
    (tmp_path / "serve").mkdir(); (tmp_path / "ladders").mkdir()
    env = {**os.environ, "MPY": str(stub), "REPO_DIR": str(REPO), "F22S_DIR": str(SAMPLED), "SERVE_DIR": str(tmp_path / "serve"),
           "LADDER_DIR": str(tmp_path / "ladders"), "WAVE_ROOT": str(tmp_path), "TAG": "llama8b", "TRIAL": "0"}
    def run(text, rc="0"):
        (tmp_path / "out.txt").write_text(text)
        return subprocess.run(["bash", "-c", f'source "{SAMPLED}/common_sampled.sh"; f22_log(){{ echo "LOG $*" >&2; }}; '
                               f'f22s_ladder_gate /root cellA xstest 512 /ep.json; echo "rc=$?"'],
                              capture_output=True, text=True, cwd=REPO,
                              env={**env, "STUB_OUT": str(tmp_path / "out.txt"), "STUB_RC": rc})
    r = run("ESCALATE 2048 /x/items.jsonl\n")
    assert r.stdout.strip().splitlines() == ["ESCALATE 2048 /x/items.jsonl", "rc=0"], r.stdout + r.stderr
    r = run("EXIT exit_ceiling\n")
    assert r.stdout.strip().splitlines() == ["EXIT exit_ceiling", "rc=0"]
    r = run("42\n")
    assert r.stdout.strip() == "rc=2" and "something other than a bound decision" in r.stderr
    r = run("", rc="2")
    assert r.stdout.strip() == "rc=2" and "REFUSED rc=2" in r.stderr
    # the sibling root is the OTHER trial's store at the same rung
    assert (tmp_path / "ladders" / "gates" / "cap512").is_dir()
