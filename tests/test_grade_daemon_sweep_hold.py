"""The once-per-advance contrast re-emission sweep (grade_daemon.contrasts_sweep_for) is held and released by a FILE,
`<state-dir>/contrasts_sweep_hold`, read ONCE per tick -- never a unit restart. Earlier, `--no-contrasts-sweep` lived only in argv
and the daemon's re-exec on code drift keeps sys.argv, so releasing the sweep on 2026-10-04 took a unit restart. Now: the file
present -> the sweep is held this tick (no registry read, no mark written, the tick event says so with the file's reason, one
`contrasts_sweep_held` event per tick); absent -> the sweep runs under its once-per-advance rule. The argv flag only SEEDS the
file, ONCE per state dir and persisted (`contrasts_sweep_hold.seeded`): the first build re-seeded the file on every code-drift
re-exec and unit restart with the flag, undoing a release at the next advance. The file was also read twice per tick, so a flip
during the per-tick pass made the tick report the opposite of what the sweep did: now one snapshot before the contrasts block is
handed to the sweep and the tick reports the snapshot AND the sweep's actual status. The schema-held early return carries the hold
status too. The per-tick contrast pass and the operator hold (`dispatch.hold`) are untouched. Synthetic registry; the live state
dir is never touched."""
from __future__ import annotations

import json
import os
import sys
import threading

import test_grade_daemon as TGD
from test_grade_daemon import _events, _load_daemon_module, _mirror_args, _root, _run_daemon
from test_grade_daemon_schema_hold import _grader_env, _tick_args

REPO = TGD.REPO
sys.path.insert(0, str(REPO / "scripts" / "registry"))

REASON = "ops: paused while the pins landing settles"


def _hold_path(tmp_path):
    return tmp_path / "state" / "contrasts_sweep_hold"


def _seeded_path(tmp_path):
    return tmp_path / "state" / "contrasts_sweep_hold.seeded"


def _mark_path(tmp_path):
    return tmp_path / "state" / "contrasts_sweep.json"


def _last_tick(events):
    return [e for e in events if e["event"] == "tick"][-1]


def _started(events):
    return [e for e in events if e["event"] == "daemon_started"][-1]


def test_the_hold_file_holds_the_sweep_without_a_mark_and_removing_it_releases_at_the_next_tick_without_a_restart(tmp_path, monkeypatch):
    gd = _load_daemon_module()
    assert gd.CONTRASTS_SWEEP_HOLD == "contrasts_sweep_hold" and gd.CONTRASTS_SWEEP_HOLD_SEEDED == "contrasts_sweep_hold.seeded"
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, _rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    assert args.contrasts_sweep is True                                                  # the flag is absent: the file alone holds
    (tmp_path / "state").mkdir(exist_ok=True)
    _hold_path(tmp_path).write_text(f"# why\n{REASON}\n")
    # tick 1: held -- no sweep event, no mark, the tick event names the reason and what the sweep did, the block says held_by_file, one held event
    s1 = gd.tick(args, *rest)
    ev1 = _events(tmp_path)
    assert s1["contrasts_sweep_hold"] == {"held": True, "reason": REASON, "sweep": gd.SWEEP_STATUS_HELD_FILE} and s1["contrasts_sweep"] == {"held_by_file": REASON}
    assert not [e for e in ev1 if e["event"] == "sweep_noncurrent"] and not _mark_path(tmp_path).exists()
    held = [e for e in ev1 if e["event"] == "contrasts_sweep_held"]
    assert len(held) == 1 and held[0]["reason"] == REASON and held[0]["file"] == str(_hold_path(tmp_path))
    assert _last_tick(ev1)["contrasts_sweep_hold"] == {"held": True, "reason": REASON, "sweep": "held_by_file"}
    # tick 2: still held (the file is read every tick): a second held event, still no mark
    s2 = gd.tick(args, *rest)
    ev2 = _events(tmp_path)[len(ev1):]
    assert s2["contrasts_sweep_hold"]["held"] is True and len([e for e in ev2 if e["event"] == "contrasts_sweep_held"]) == 1 and not _mark_path(tmp_path).exists()
    # the file is removed: tick 3 (same process, same args) sweeps once and writes the mark; the tick event says not held and that the sweep ran
    _hold_path(tmp_path).unlink()
    s3 = gd.tick(args, *rest)
    ev3 = _events(tmp_path)[len(ev1) + len(ev2):]
    assert s3["contrasts_sweep_hold"] == {"held": False, "reason": None, "sweep": gd.SWEEP_STATUS_RAN}
    (sweep,) = [e for e in ev3 if e["event"] == "sweep_noncurrent"]
    assert sweep["daemon_sha"] == gd.IMPORTED_CODE_SHA and "held_by_file" not in s3["contrasts_sweep"] and s3["contrasts_sweep"]["daemon_sha"] == gd.IMPORTED_CODE_SHA
    assert json.loads(_mark_path(tmp_path).read_text())["daemon_sha"] == gd.IMPORTED_CODE_SHA
    assert not [e for e in ev3 if e["event"] == "contrasts_sweep_held"]
    # tick 4: same sha, not held -- the once-per-advance rule stands (no second sweep) and the tick says so
    s4 = gd.tick(args, *rest)
    ev4 = _events(tmp_path)[len(ev1) + len(ev2) + len(ev3):]
    assert s4["contrasts_sweep_hold"] == {"held": False, "reason": None, "sweep": gd.SWEEP_STATUS_SKIPPED} and "contrasts_sweep" not in s4
    assert not [e for e in ev4 if e["event"] == "sweep_noncurrent"]
    # the file re-created with no reason holds again with "(no reason given)"
    _hold_path(tmp_path).write_text("")
    s5 = gd.tick(args, *rest)
    assert s5["contrasts_sweep_hold"] == {"held": True, "reason": "(no reason given)", "sweep": "held_by_file"}
    assert not _seeded_path(tmp_path).exists()                                           # no flag was ever given: no seeding record


def test_the_argv_flag_seeds_the_hold_file_once_per_state_dir_and_a_released_hold_survives_re_execs_and_restarts_with_the_flag(tmp_path, monkeypatch):
    """The first build re-seeded the hold on every code-drift re-exec (reexec_self keeps sys.argv) and every unit restart with the
    flag, so a release was undone at the next advance. Now the seeding is one-time and PERSISTED in
    <state>/contrasts_sweep_hold.seeded: the first flagged start seeds exactly once; a release (rm) then survives a simulated
    re-exec (the start-up seeding called again with the same args and state) and a unit restart with the flag (a subprocess start
    over the same state dir)."""
    gd = _load_daemon_module()
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, _rids = _root(tmp_path)
    # the first start with --no-contrasts-sweep and no hold file: the file AND the seeding record are written, the tick is held, no mark
    r = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep"); assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)
    assert _hold_path(tmp_path).read_text().splitlines()[0] == gd.CONTRASTS_SWEEP_HOLD_ARGV_REASON == "argv --no-contrasts-sweep (initial state)"
    rec = json.loads(_seeded_path(tmp_path).read_text())
    assert rec["wrote_file"] is True and rec["reason"] == gd.CONTRASTS_SWEEP_HOLD_ARGV_REASON and rec["daemon_sha"] == _started(ev)["daemon_sha"] and rec["utc"] and rec["pid"]
    st = _started(ev)
    assert st["contrasts_sweep_hold_seeded"] == str(_hold_path(tmp_path)) and st["contrasts_sweep_hold_seeding"]["seeded_now"] == str(_hold_path(tmp_path))
    assert st["contrasts_sweep_hold_seeding"]["seeded_before"] is None and not [e for e in ev if e["event"].startswith("contrasts_sweep_hold_")]
    assert st["contrasts_sweep_hold_seeding"]["evidence"] == [] and st["contrasts_sweep_hold_seeding"]["skipped"] is None     # fix-2: a truly fresh dir seeds
    assert _last_tick(ev)["contrasts_sweep_hold"] == {"held": True, "reason": gd.CONTRASTS_SWEEP_HOLD_ARGV_REASON, "sweep": "held_by_file"}
    assert _last_tick(ev)["contrasts_sweep"] == {"held_by_file": gd.CONTRASTS_SWEEP_HOLD_ARGV_REASON} and not _mark_path(tmp_path).exists()
    assert len([e for e in ev if e["event"] == "contrasts_sweep_held"]) == 1
    # a second start with the flag over an operator's own file: nothing rewritten, the record stands, the file's reason stands
    _hold_path(tmp_path).write_text(f"{REASON}\n")
    n = len(ev)
    r2 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep"); assert r2.returncode == 0, r2.stderr + r2.stdout
    ev2 = _events(tmp_path)[n:]
    assert _hold_path(tmp_path).read_text() == f"{REASON}\n" and _started(ev2)["contrasts_sweep_hold_seeded"] is None
    assert _started(ev2)["contrasts_sweep_hold_seeding"]["seeded_before"] == rec and json.loads(_seeded_path(tmp_path).read_text()) == rec
    assert _last_tick(ev2)["contrasts_sweep_hold"] == {"held": True, "reason": REASON, "sweep": "held_by_file"}
    # the RELEASE: the operator removes the file. A simulated code-drift re-exec -- the start-up seeding called again with the same flagged args
    # and state -- writes NOTHING: the hold stays released
    _hold_path(tmp_path).unlink()
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep"); _grader_env(monkeypatch, env)
    assert args.contrasts_sweep is False
    again = gd.contrasts_sweep_hold_initial(args)
    assert again["seeded_now"] is None and again["seeded_before"] == rec and again["hold_present"] is False and not _hold_path(tmp_path).exists()
    # in-process ticks under those args sweep: the flag is never re-consulted, the file alone decides
    n = len(_events(tmp_path))
    s = gd.tick(args, *rest)
    ev3 = _events(tmp_path)[n:]
    assert s["contrasts_sweep_hold"] == {"held": False, "reason": None, "sweep": "ran"} and [e["event"] for e in ev3 if e["event"] == "sweep_noncurrent"] == ["sweep_noncurrent"]
    assert json.loads(_mark_path(tmp_path).read_text())["daemon_sha"] == gd.IMPORTED_CODE_SHA and not _hold_path(tmp_path).exists()
    # a UNIT RESTART with the flag (a subprocess start over the same state dir): the hold stays released, the start says the seeding already happened
    n = len(_events(tmp_path))
    r3 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep"); assert r3.returncode == 0, r3.stderr + r3.stdout
    ev4 = _events(tmp_path)[n:]
    assert not _hold_path(tmp_path).exists() and _started(ev4)["contrasts_sweep_hold_seeded"] is None and _started(ev4)["contrasts_sweep_hold_seeding"]["seeded_before"] == rec
    assert _last_tick(ev4)["contrasts_sweep_hold"] == {"held": False, "reason": None, "sweep": "skipped_once_per_advance"}      # same sha: the once-per-advance rule
    assert not [e for e in ev4 if e["event"] in ("contrasts_sweep_held", "sweep_noncurrent")]
    # a --dry-run start with the flag writes NO file and NO record (an audit must not leave a hold or a seeding behind) and holds for itself
    _seeded_path(tmp_path).unlink()
    n = len(_events(tmp_path))
    r4 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep", dry_run=True); assert r4.returncode == 0, r4.stderr + r4.stdout
    ev5 = _events(tmp_path)[n:]
    assert not _hold_path(tmp_path).exists() and not _seeded_path(tmp_path).exists() and _started(ev5)["contrasts_sweep_hold_seeding"]["skipped"] == "dry_run"
    assert _last_tick(ev5)["contrasts_sweep"] == {"held_by_file": gd.CONTRASTS_SWEEP_HOLD_ARGV_REASON + " (dry run: no file written)"}
    assert _last_tick(ev5)["contrasts_sweep_hold"] == {"held": False, "reason": None, "sweep": "held_by_flag_dry_run"} and not [e for e in ev5 if e["event"] == "sweep_noncurrent"]
    # the seeding is inert without the flag and without a state dir
    assert gd.contrasts_sweep_hold_initial(type("A", (), {"contrasts_sweep": True, "state_dir": str(tmp_path / "state")})())["skipped"] == "no_flag"
    assert gd.contrasts_sweep_hold_initial(type("A", (), {"contrasts_sweep": False, "state_dir": None})())["skipped"] == "no_state_dir"
    assert not _hold_path(tmp_path).exists() and not _seeded_path(tmp_path).exists()
    # fix-2: THIS state dir now carries history (the flagged starts above said their seeding on daemon_started and held ticks) and no record: a
    # flagged start over an operator's file writes NOTHING -- not even the record -- and names the evidence; the operator decides
    _hold_path(tmp_path).write_text(f"{REASON}\n")
    amb = gd.contrasts_sweep_hold_initial(args)
    assert amb["skipped"] == gd.SWEEPHOLD_SKIP_AMBIGUOUS == "ambiguous_legacy_state" and amb["seeded_now"] is None and amb["seeded_before"] is None and amb["hold_present"] is True
    assert any("daemon_started" in e for e in amb["evidence"]) and any("tick event(s)" in e for e in amb["evidence"]) and "acknowledge" in amb["operator_decision"]
    assert _hold_path(tmp_path).read_text() == f"{REASON}\n" and not _seeded_path(tmp_path).exists()
    # a FRESH state dir with an operator's hold file alone (no history): the hold stands and the record is written -- fix-1's behaviour
    fresh = tmp_path / "fresh"; fresh.mkdir(); (fresh / gd.CONTRASTS_SWEEP_HOLD).write_text(f"{REASON}\n")
    first = gd.contrasts_sweep_hold_initial(type("A", (), {"contrasts_sweep": False, "state_dir": str(fresh), "dry_run": False})())
    assert first["seeded_now"] is None and first["seeded_before"] is None and first["hold_present"] is True and first.get("seeding_recorded") is True and first["evidence"] == []
    assert (fresh / gd.CONTRASTS_SWEEP_HOLD).read_text() == f"{REASON}\n" and json.loads((fresh / gd.CONTRASTS_SWEEP_HOLD_SEEDED).read_text())["wrote_file"] is False
    assert gd.contrasts_sweep_hold(None) is None and gd.contrasts_sweep_hold_seeding(None) is None and gd.sweephold_evidence(None) == []


def test_the_hold_is_read_once_per_tick_and_a_flip_during_the_per_tick_pass_cannot_contradict_what_the_sweep_did(tmp_path, monkeypatch):
    """The file was read for the tick event and again inside contrasts_sweep_for after the per-tick pass. Now one snapshot before
    the contrasts block is handed to the sweep; the tick reports the snapshot and the sweep's actual status, and they agree
    whatever the per-tick pass does to the file in between."""
    gd = _load_daemon_module()
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, _rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    (tmp_path / "state").mkdir(exist_ok=True)
    real_pass = gd.emit_contrasts_for
    flips = {"do": None}

    def pass_then_flip(*a, **k):                                                            # the per-tick pass, with a file flip inside it
        out = real_pass(*a, **k)
        if flips["do"] == "remove":
            _hold_path(tmp_path).unlink()
        elif flips["do"] == "create":
            _hold_path(tmp_path).write_text(f"{REASON}\n")
        return out

    monkeypatch.setattr(gd, "emit_contrasts_for", pass_then_flip)
    # tick A: the file is present at the snapshot and REMOVED during the pass -> the sweep is held (the snapshot's decision), no mark; the report agrees
    _hold_path(tmp_path).write_text(f"{REASON}\n"); flips["do"] = "remove"
    sA = gd.tick(args, *rest)
    evA = _events(tmp_path)
    assert sA["contrasts_sweep_hold"] == {"held": True, "reason": REASON, "sweep": "held_by_file"} and sA["contrasts_sweep"] == {"held_by_file": REASON}
    assert not _mark_path(tmp_path).exists() and not [e for e in evA if e["event"] == "sweep_noncurrent"] and len([e for e in evA if e["event"] == "contrasts_sweep_held"]) == 1
    assert not _hold_path(tmp_path).exists()                                               # the flip happened (the next tick will see it)
    # tick B: the file is absent at the snapshot and CREATED during the pass -> the sweep RAN (mark written); the report says not held / ran
    flips["do"] = "create"
    sB = gd.tick(args, *rest)
    evB = _events(tmp_path)[len(evA):]
    assert sB["contrasts_sweep_hold"] == {"held": False, "reason": None, "sweep": "ran"} and sB["contrasts_sweep"]["daemon_sha"] == gd.IMPORTED_CODE_SHA
    assert json.loads(_mark_path(tmp_path).read_text())["daemon_sha"] == gd.IMPORTED_CODE_SHA and [e["event"] for e in evB if e["event"] == "sweep_noncurrent"] == ["sweep_noncurrent"]
    assert not [e for e in evB if e["event"] == "contrasts_sweep_held"] and _hold_path(tmp_path).exists()
    # tick C: the file the pass created stands at the snapshot -> held, said once; the pass flips nothing more
    flips["do"] = None
    sC = gd.tick(args, *rest)
    evC = _events(tmp_path)[len(evA) + len(evB):]
    assert sC["contrasts_sweep_hold"] == {"held": True, "reason": REASON, "sweep": "held_by_file"} and len([e for e in evC if e["event"] == "contrasts_sweep_held"]) == 1
    # tick D: released -> same sha: skipped once-per-advance, and the report says so
    _hold_path(tmp_path).unlink()
    sD = gd.tick(args, *rest)
    assert sD["contrasts_sweep_hold"] == {"held": False, "reason": None, "sweep": "skipped_once_per_advance"} and "contrasts_sweep" not in sD


def test_the_operator_hold_still_skips_the_whole_contrasts_block_and_the_sweep_hold_is_reported_beside_it(tmp_path, monkeypatch):
    """(4) no other behaviour change: under `dispatch.hold` the contrasts block (pass + sweep) is skipped as today; the sweep-hold status is
    still read and reported on the tick (sweep: operator_held), and said once while held."""
    gd = _load_daemon_module()
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, _rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / gd.DISPATCH_HOLD).write_text("operator: paused\n")
    s1 = gd.tick(args, *rest)
    ev1 = _events(tmp_path)
    assert s1["held"] is True and s1["contrasts_sweep_hold"] == {"held": False, "reason": None, "sweep": "operator_held"} and "contrasts_sweep" not in s1
    assert not [e for e in ev1 if e["event"] in ("contrasts", "sweep_noncurrent", "contrasts_sweep_held")] and not _mark_path(tmp_path).exists()
    _hold_path(tmp_path).write_text(f"{REASON}\n")
    s2 = gd.tick(args, *rest)
    ev2 = _events(tmp_path)[len(ev1):]
    assert s2["held"] is True and s2["contrasts_sweep_hold"] == {"held": True, "reason": REASON, "sweep": "operator_held"} and "contrasts_sweep" not in s2
    assert [e["reason"] for e in ev2 if e["event"] == "contrasts_sweep_held"] == [REASON] and not [e for e in ev2 if e["event"] in ("contrasts", "sweep_noncurrent")]
    assert not _mark_path(tmp_path).exists()


def test_a_schema_held_tick_carries_the_sweep_hold_status_too(tmp_path, monkeypatch):
    """The schema-hold early return (no work at all) carries contrasts_sweep_hold as every other tick does, with the sweep status
    `schema_held`; the start-up event sequence the schema-hold tests pin is untouched (no new event)."""
    gd = _load_daemon_module()
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, _rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    (tmp_path / "state").mkdir(exist_ok=True)
    gap = {"kind": "version", "have": 1, "want": 2, "n_drift": 0, "since_utc": "2026-01-01T00:00:00Z"}
    monkeypatch.setattr(gd, "SCHEMA_GATE", type("G", (), {"check": lambda self, root_, state_dir: gap, "held": True})())
    s1 = gd.tick(args, *rest)
    assert s1["held"] is True and s1["hold"] == "schema" and s1["contrasts_sweep_hold"] == {"held": False, "reason": None, "sweep": "schema_held"}
    _hold_path(tmp_path).write_text(f"{REASON}\n")
    s2 = gd.tick(args, *rest)
    assert s2["hold"] == "schema" and s2["contrasts_sweep_hold"] == {"held": True, "reason": REASON, "sweep": "schema_held"} and "contrasts_sweep" not in s2
    assert not _mark_path(tmp_path).exists() and not [e for e in _events(tmp_path) if e["event"] in ("sweep_noncurrent", "contrasts")]


# ------------------------------------------------------------------------------------------- sweephold-fix-2: the upgrade path
# The absence of `.seeded` is not permission to seed: a daemon on the parent code (f03e6ee76) wrote the hold file but no record, so a release (rm)
# under it would have been undone by the first flagged re-exec into fix-1. Evidence of an earlier sweephold-era daemon -- the state dir's event
# log (a daemon_started carrying contrasts_sweep_hold_seeded / _seeding, a contrasts_sweep_held event, a tick carrying contrasts_sweep_hold) or
# a mark carrying sweephold_era -- makes a flagged start write nothing and leave the decision to the operator.


def _line(ev):
    return json.dumps(ev, sort_keys=True) + "\n"


def _parent_style_history(tmp_path, hold_path):
    """What the PARENT version (f03e6ee76) leaves in a state dir it ran flagged in: a daemon_started saying it seeded the hold (its
    contrasts_sweep_hold_seeded, no _seeding dict, no record), held ticks and contrasts_sweep_held events -- then the operator's release (rm)."""
    (tmp_path / "state").mkdir(exist_ok=True)
    reason = "argv --no-contrasts-sweep (initial state)"
    lines = [_line({"event": "daemon_started", "daemon_sha": "f03e6ee76c8fe71538ffeaa46155e32d1eb6f78c", "pid": 4242, "once": False, "contrasts_sweep_hold_seeded": str(hold_path),
                    "utc": "2026-10-04T13:00:00.000Z"}),
             _line({"event": "contrasts_sweep_held", "reason": reason, "file": str(hold_path), "utc": "2026-10-04T13:01:00.000Z"}),
             _line({"event": "tick", "contrasts_sweep_hold": {"held": True, "reason": reason}, "contrasts_sweep": {"held_by_file": reason}, "daemon_sha": "f03e6ee76c8fe71538ffeaa46155e32d1eb6f78c",
                    "utc": "2026-10-04T13:01:00.500Z"}),
             _line({"event": "contrasts_sweep_held", "reason": reason, "file": str(hold_path), "utc": "2026-10-04T13:02:00.000Z"}),
             _line({"event": "tick", "contrasts_sweep_hold": {"held": True, "reason": reason}, "contrasts_sweep": {"held_by_file": reason}, "daemon_sha": "f03e6ee76c8fe71538ffeaa46155e32d1eb6f78c",
                    "utc": "2026-10-04T13:02:00.500Z"})]
    (tmp_path / "state" / "grade_daemon.jsonl").write_text("".join(lines))
    return len(lines)


def test_a_predecessor_state_dir_with_a_recorded_seeding_and_the_hold_released_is_not_re_held_by_a_flagged_start(tmp_path, monkeypatch):
    """The parent seeded the hold (said on daemon_started) and never wrote .seeded; the operator released it (rm); the new code
    re-execs with the same flagged argv. Earlier, both files absent read as a first-ever start and the hold was recreated. Now the
    state dir's history is evidence: nothing is written, the release survives, daemon_started reports ambiguous_legacy_state with
    the evidence named, the sweep runs; the operator's two acknowledgements each end the ambiguity."""
    gd = _load_daemon_module()
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, _rids = _root(tmp_path)
    n0 = _parent_style_history(tmp_path, _hold_path(tmp_path))                         # the hold file itself is gone: released
    assert not _hold_path(tmp_path).exists() and not _seeded_path(tmp_path).exists()
    ev_names = gd.sweephold_evidence(str(tmp_path / "state"))
    assert len(ev_names) == 3 and ev_names[0].startswith("event log: 1 daemon_started event(s)") and "first at 2026-10-04T13:00:00.000Z" in ev_names[0]
    assert ev_names[1] == "event log: 2 contrasts_sweep_held event(s)" and ev_names[2] == "event log: 2 tick event(s) carrying contrasts_sweep_hold"
    # the flagged start on the new code: NOTHING written, the release intact, the decision reported, the sweep runs (its mark now carries sweephold_era)
    r = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep"); assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)[n0:]
    st = _started(ev)
    assert st["contrasts_sweep_hold_seeded"] is None and st["contrasts_sweep_hold_seeding"]["skipped"] == "ambiguous_legacy_state"
    assert st["contrasts_sweep_hold_seeding"]["evidence"] == ev_names and "touch" in st["contrasts_sweep_hold_seeding"]["operator_decision"]
    assert not _hold_path(tmp_path).exists() and not _seeded_path(tmp_path).exists()
    assert _last_tick(ev)["contrasts_sweep_hold"] == {"held": False, "reason": None, "sweep": "ran"} and [e["event"] for e in ev if e["event"] == "sweep_noncurrent"] == ["sweep_noncurrent"]
    mark = json.loads(_mark_path(tmp_path).read_text())
    assert mark["sweephold_era"] is True and mark["daemon_sha"] == st["daemon_sha"]
    # a second flagged start: still ambiguous (now the mark is evidence too), still nothing written, the once-per-advance rule stands
    n = len(_events(tmp_path))
    r2 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep"); assert r2.returncode == 0, r2.stderr + r2.stdout
    ev2 = _events(tmp_path)[n:]
    assert _started(ev2)["contrasts_sweep_hold_seeding"]["skipped"] == "ambiguous_legacy_state" and any(e.startswith("mark: contrasts_sweep.json carries sweephold_era") for e in _started(ev2)["contrasts_sweep_hold_seeding"]["evidence"])
    assert not _hold_path(tmp_path).exists() and not _seeded_path(tmp_path).exists() and _last_tick(ev2)["contrasts_sweep_hold"]["sweep"] == "skipped_once_per_advance"
    # the operator acknowledges the released state: any JSON object as the record -> seeded_before, nothing written, no ambiguity
    _seeded_path(tmp_path).write_text(json.dumps({"acknowledged_by": "operator", "utc": "2026-10-04T14:00:00Z"}) + "\n")
    n = len(_events(tmp_path))
    r3 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep"); assert r3.returncode == 0, r3.stderr + r3.stdout
    ev3 = _events(tmp_path)[n:]
    assert _started(ev3)["contrasts_sweep_hold_seeding"]["seeded_before"] == {"acknowledged_by": "operator", "utc": "2026-10-04T14:00:00Z"} and _started(ev3)["contrasts_sweep_hold_seeding"]["skipped"] is None
    assert not _hold_path(tmp_path).exists() and _last_tick(ev3)["contrasts_sweep_hold"]["held"] is False
    # ...or the operator holds: a touch of the hold file -> held at the next tick (the file alone decides), still nothing seeded by the daemon
    _seeded_path(tmp_path).unlink(); _hold_path(tmp_path).write_text("ops: hold it\n")
    n = len(_events(tmp_path))
    r4 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep"); assert r4.returncode == 0, r4.stderr + r4.stdout
    ev4 = _events(tmp_path)[n:]
    assert _started(ev4)["contrasts_sweep_hold_seeding"]["skipped"] == "ambiguous_legacy_state" and _started(ev4)["contrasts_sweep_hold_seeding"]["hold_present"] is True
    assert _last_tick(ev4)["contrasts_sweep_hold"] == {"held": True, "reason": "ops: hold it", "sweep": "held_by_file"} and not _seeded_path(tmp_path).exists()


def test_the_evidence_is_exactly_the_event_log_and_the_mark_and_a_legacy_mark_or_an_empty_dir_is_none(tmp_path):
    gd = _load_daemon_module()
    st = tmp_path / "state"; st.mkdir()
    assert gd.sweephold_evidence(str(st)) == []                                                          # a truly fresh dir
    (st / "contrasts_sweep.json").write_text(json.dumps({"daemon_sha": "0" * 40, "utc": "2026-10-01T00:00:00Z", "counts": {}}) + "\n")
    assert gd.sweephold_evidence(str(st)) == []                                                          # a pre-sweephold mark says nothing about the hold
    (st / "grade_daemon.jsonl").write_text(_line({"event": "daemon_started", "daemon_sha": "0" * 40, "pid": 1, "once": True, "utc": "2026-10-01T00:00:00Z"})
                                           + _line({"event": "tick", "dispatched": 0, "daemon_sha": "0" * 40, "utc": "2026-10-01T00:01:00Z"}))
    assert gd.sweephold_evidence(str(st)) == []                                                          # pre-sweephold events say nothing either
    (st / "contrasts_sweep.json").write_text(json.dumps({"daemon_sha": "1" * 40, "utc": "2026-10-04T00:00:00Z", "counts": {}, "sweephold_era": True}) + "\n")
    assert gd.sweephold_evidence(str(st)) == ["mark: contrasts_sweep.json carries sweephold_era (daemon_sha " + "1" * 40 + ", 2026-10-04T00:00:00Z)"]
    (st / "contrasts_sweep.json").unlink()
    # fix-3: a fix-1 / fix-2 DRY-RUN start-up (no stamp yet; its seeding dict says skipped: dry_run) is noise, not evidence; a REAL seeding start is
    (st / "grade_daemon.jsonl").write_text(_line({"event": "daemon_started", "daemon_sha": "2" * 40, "pid": 2, "once": False, "contrasts_sweep_hold_seeded": None,
                                                 "contrasts_sweep_hold_seeding": {"seeded_now": None, "skipped": "dry_run"}, "utc": "2026-10-04T01:00:00Z"}))
    assert gd.sweephold_evidence(str(st)) == [] and gd.sweephold_history(str(st)) == {"evidence": [], "dry_run_events_skipped": 1}
    (st / "grade_daemon.jsonl").write_text(_line({"event": "daemon_started", "daemon_sha": "2" * 40, "pid": 2, "once": False, "contrasts_sweep_hold_seeded": None,
                                                 "contrasts_sweep_hold_seeding": {"seeded_now": None, "skipped": None, "seeded_before": {"utc": "x"}}, "utc": "2026-10-04T01:00:00Z"}))
    assert gd.sweephold_evidence(str(st)) == ["event log: 1 daemon_started event(s) carrying contrasts_sweep_hold_seeded / contrasts_sweep_hold_seeding (first at 2026-10-04T01:00:00Z)"]
    # with evidence and no record a flagged seeding writes nothing; a hold file alone on this dir is left as it is, unrecorded (ambiguous too)
    a = gd.contrasts_sweep_hold_initial(type("A", (), {"contrasts_sweep": False, "state_dir": str(st), "dry_run": False})())
    assert a["skipped"] == "ambiguous_legacy_state" and not (st / gd.CONTRASTS_SWEEP_HOLD).exists() and not (st / gd.CONTRASTS_SWEEP_HOLD_SEEDED).exists()
    (st / gd.CONTRASTS_SWEEP_HOLD).write_text("ops\n")
    b = gd.contrasts_sweep_hold_initial(type("A", (), {"contrasts_sweep": False, "state_dir": str(st), "dry_run": False})())
    assert b["skipped"] == "ambiguous_legacy_state" and b["hold_present"] is True and not (st / gd.CONTRASTS_SWEEP_HOLD_SEEDED).exists()
    # an operator's record alone ends it: seeded_before, nothing written
    (st / gd.CONTRASTS_SWEEP_HOLD_SEEDED).write_text(json.dumps({"acknowledged_by": "operator"}) + "\n")
    c = gd.contrasts_sweep_hold_initial(type("A", (), {"contrasts_sweep": False, "state_dir": str(st), "dry_run": False})())
    assert c["seeded_before"] == {"acknowledged_by": "operator"} and c["skipped"] is None and (st / gd.CONTRASTS_SWEEP_HOLD).read_text() == "ops\n"


# ---------------------------------------------------------------------------------------------- sweephold-fix-3: dry runs are not evidence
# A `--dry-run --no-contrasts-sweep` audit writes no hold and no record but logs its start-up and ticks; fix-2 counted them, so the first REAL
# flagged start after an audit read ambiguous_legacy_state and the sweep ran against the requested hold. Now a dry-run daemon stamps every
# event `dry_run: true` and the evidence excludes them (counted as dry_run_events_skipped); a real predecessor still makes a flagged start ambiguous.


def test_a_dry_run_flagged_audit_on_a_fresh_dir_leaves_the_first_real_flagged_start_free_to_seed(tmp_path, monkeypatch):
    gd = _load_daemon_module()
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, _rids = _root(tmp_path)
    # the audit: a dry-run flagged start on a fresh state dir -- nothing written, but its start-up and tick are logged, stamped dry_run
    r = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep", dry_run=True); assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)
    assert not _hold_path(tmp_path).exists() and not _seeded_path(tmp_path).exists() and not _mark_path(tmp_path).exists()
    assert _started(ev)["dry_run"] is True and _started(ev)["contrasts_sweep_hold_seeding"]["skipped"] == "dry_run" and _last_tick(ev)["dry_run"] is True
    assert "contrasts_sweep_hold" in _last_tick(ev) and _last_tick(ev)["contrasts_sweep_hold"]["sweep"] == "held_by_flag_dry_run"
    hist = gd.sweephold_history(str(tmp_path / "state"))
    assert hist["evidence"] == [] and hist["dry_run_events_skipped"] >= 2 and gd.sweephold_evidence(str(tmp_path / "state")) == []
    # the first REAL flagged start: seeds the hold (file + record, seeded_now), reports the dry-run noise it skipped, holds the sweep (no mark)
    n = len(ev)
    r2 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep"); assert r2.returncode == 0, r2.stderr + r2.stdout
    ev2 = _events(tmp_path)[n:]
    st2 = _started(ev2)
    assert st2["dry_run"] is False and st2["contrasts_sweep_hold_seeded"] == str(_hold_path(tmp_path)) and st2["contrasts_sweep_hold_seeding"]["seeded_now"] == str(_hold_path(tmp_path))
    assert st2["contrasts_sweep_hold_seeding"]["skipped"] is None and st2["contrasts_sweep_hold_seeding"]["evidence"] == [] and st2["contrasts_sweep_hold_seeding"]["dry_run_events_skipped"] >= 2
    assert _hold_path(tmp_path).read_text().splitlines()[0] == gd.CONTRASTS_SWEEP_HOLD_ARGV_REASON and json.loads(_seeded_path(tmp_path).read_text())["wrote_file"] is True
    assert _last_tick(ev2)["dry_run"] is False and _last_tick(ev2)["contrasts_sweep_hold"] == {"held": True, "reason": gd.CONTRASTS_SWEEP_HOLD_ARGV_REASON, "sweep": "held_by_file"}
    assert not _mark_path(tmp_path).exists() and [e["dry_run"] for e in ev2 if e["event"] == "contrasts_sweep_held"] == [False]
    # and a later dry-run audit over that state dir still writes nothing and is still not evidence; the real start's events are
    n = len(_events(tmp_path))
    r3 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep", dry_run=True); assert r3.returncode == 0, r3.stderr + r3.stdout
    ev3 = _events(tmp_path)[n:]
    assert _started(ev3)["contrasts_sweep_hold_seeding"]["skipped"] == "dry_run" and [e["dry_run"] for e in ev3 if e["event"] == "contrasts_sweep_held"] == [True]
    hist2 = gd.sweephold_history(str(tmp_path / "state"))
    assert any(e.startswith("event log: 1 daemon_started event(s)") for e in hist2["evidence"]) and hist2["dry_run_events_skipped"] >= 5


def test_the_evidence_excludes_stamped_dry_run_events_and_keeps_the_real_predecessors(tmp_path):
    gd = _load_daemon_module()
    st = tmp_path / "state"; st.mkdir()
    hold = str(st / gd.CONTRASTS_SWEEP_HOLD); reason = gd.CONTRASTS_SWEEP_HOLD_ARGV_REASON
    dry = [_line({"event": "daemon_started", "daemon_sha": "3" * 40, "pid": 3, "once": True, "dry_run": True, "contrasts_sweep_hold_seeded": None,
                  "contrasts_sweep_hold_seeding": {"seeded_now": None, "skipped": "dry_run", "evidence": []}, "utc": "2026-10-04T02:00:00Z"}),
           _line({"event": "contrasts_sweep_held", "dry_run": True, "reason": reason, "file": hold, "utc": "2026-10-04T02:00:01Z"}),
           _line({"event": "tick", "dry_run": True, "contrasts_sweep_hold": {"held": False, "reason": None, "sweep": "held_by_flag_dry_run"}, "daemon_sha": "3" * 40, "utc": "2026-10-04T02:00:02Z"})]
    (st / "grade_daemon.jsonl").write_text("".join(dry))
    assert gd.sweephold_history(str(st)) == {"evidence": [], "dry_run_events_skipped": 3} and gd.sweephold_evidence(str(st)) == []     # dry-run-only history: none
    a = gd.contrasts_sweep_hold_initial(type("A", (), {"contrasts_sweep": False, "state_dir": str(st), "dry_run": False})())
    assert a["seeded_now"] == hold and a["skipped"] is None and a["dry_run_events_skipped"] == 3 and (st / gd.CONTRASTS_SWEEP_HOLD_SEEDED).exists()    # a real start seeds
    (st / gd.CONTRASTS_SWEEP_HOLD).unlink(); (st / gd.CONTRASTS_SWEEP_HOLD_SEEDED).unlink()
    real = [_line({"event": "daemon_started", "daemon_sha": "f03e6ee76c8fe71538ffeaa46155e32d1eb6f78c", "pid": 4, "once": False, "contrasts_sweep_hold_seeded": hold, "utc": "2026-10-04T03:00:00Z"}),
            _line({"event": "tick", "contrasts_sweep_hold": {"held": True, "reason": reason}, "daemon_sha": "f03e6ee76c8fe71538ffeaa46155e32d1eb6f78c", "utc": "2026-10-04T03:00:01Z"})]
    (st / "grade_daemon.jsonl").write_text("".join(dry + real + dry))
    hist = gd.sweephold_history(str(st))
    assert hist["evidence"] == ["event log: 1 daemon_started event(s) carrying contrasts_sweep_hold_seeded / contrasts_sweep_hold_seeding (first at 2026-10-04T03:00:00Z)",
                                "event log: 1 tick event(s) carrying contrasts_sweep_hold"] and hist["dry_run_events_skipped"] == 6      # mixed: the real ones only
    b = gd.contrasts_sweep_hold_initial(type("A", (), {"contrasts_sweep": False, "state_dir": str(st), "dry_run": False})())
    assert b["skipped"] == "ambiguous_legacy_state" and b["evidence"] == hist["evidence"] and b["dry_run_events_skipped"] == 6 and not (st / gd.CONTRASTS_SWEEP_HOLD).exists()


# ------------------------------------------------------------------------------------------------ run-context classification
# The stamping covers NEW dry-run start-ups and ticks, but an earlier dry-run audit left a start-up (its seeding dict
# `skipped: dry_run`, no stamp) FOLLOWED by ticks with no stamp at all, and those ticks counted as a real hold era. Now every era
# event is classified from its run context: (a) its own stamp / held_by_flag_dry_run, (b) a dry-run start-up's window, (c) a real
# start-up's window, (d) no window -> reported through the operator decision, never counted, never silently seeded over. The log
# shape below is the live event log's (read-only; synthetic daemon_sha / pid / utc, TESTHOST, no run ids): daemon_started {event,
# daemon_sha, pid, once, utc, contrasts_sweep_hold_seeded, contrasts_sweep_hold_seeding}, tick {event, ...,
# contrasts_sweep_hold: {held, reason, sweep}, daemon_sha, utc}.


def _old_style_log(st, hold, reason):
    """An OLD-STYLE history: an orphan tick before any start-up (its start-up outside the scanned text); a fix-1-era DRY-RUN start-up (seeding
    `skipped: dry_run`, no stamp) with three unstamped ticks and a held event after it -- one tick's sweep says held_by_flag_dry_run, two say
    `ran` / `skipped_once_per_advance` (a --dry-run without the flag), none stamped; a REAL seeding start-up with two held ticks and a held
    event; a daemon_exit closing that window and one orphan tick after it. Returns the lines' classes in order for the assertion."""
    dry_sha, real_sha = "4" * 40, "5" * 40
    lines = [
        _line({"event": "tick", "dispatched": 0, "contrasts_sweep_hold": {"held": True, "reason": reason, "sweep": "held_by_file"}, "daemon_sha": dry_sha, "utc": "2026-10-04T05:59:00.000Z"}),
        _line({"event": "daemon_started", "daemon_sha": dry_sha, "pid": 4041, "once": False, "contrasts_sweep_hold_seeded": None,
               "contrasts_sweep_hold_seeding": {"seeded_now": None, "seeded_before": None, "hold_present": False, "skipped": "dry_run", "evidence": []}, "utc": "2026-10-04T06:00:00.000Z"}),
        _line({"event": "tick", "dispatched": 0, "contrasts_sweep_hold": {"held": False, "reason": None, "sweep": "held_by_flag_dry_run"}, "daemon_sha": dry_sha, "utc": "2026-10-04T06:01:00.000Z"}),
        _line({"event": "tick", "dispatched": 0, "contrasts_sweep_hold": {"held": False, "reason": None, "sweep": "ran"}, "daemon_sha": dry_sha, "utc": "2026-10-04T06:02:00.000Z"}),
        _line({"event": "contrasts_sweep_held", "reason": reason, "file": hold, "utc": "2026-10-04T06:03:00.000Z"}),
        _line({"event": "tick", "dispatched": 0, "contrasts_sweep_hold": {"held": True, "reason": reason, "sweep": "skipped_once_per_advance"}, "daemon_sha": dry_sha, "utc": "2026-10-04T06:03:00.500Z"}),
        _line({"event": "daemon_started", "daemon_sha": real_sha, "pid": 4042, "once": False, "contrasts_sweep_hold_seeded": hold,
               "contrasts_sweep_hold_seeding": {"seeded_now": hold, "seeded_before": None, "hold_present": True, "skipped": None, "evidence": [], "seeding_recorded": True}, "utc": "2026-10-04T07:00:00.000Z"}),
        _line({"event": "contrasts_sweep_held", "reason": reason, "file": hold, "utc": "2026-10-04T07:01:00.000Z"}),
        _line({"event": "tick", "dispatched": 0, "contrasts_sweep_hold": {"held": True, "reason": reason, "sweep": "held_by_file"}, "daemon_sha": real_sha, "utc": "2026-10-04T07:01:00.500Z"}),
        _line({"event": "tick", "dispatched": 0, "contrasts_sweep_hold": {"held": True, "reason": reason, "sweep": "held_by_file"}, "daemon_sha": real_sha, "utc": "2026-10-04T07:02:00.500Z"}),
        _line({"event": "daemon_exit", "daemon_sha": real_sha, "pid": 4042, "utc": "2026-10-04T07:03:00.000Z"}),
        _line({"event": "tick", "dispatched": 0, "contrasts_sweep_hold": {"held": True, "reason": reason, "sweep": "held_by_file"}, "daemon_sha": real_sha, "utc": "2026-10-04T07:04:00.000Z"}),
    ]
    (st / "grade_daemon.jsonl").write_text("".join(lines))
    return [("tick", "unclassified"), ("daemon_started", "dry_run"), ("tick", "dry_run"), ("tick", "dry_run"), ("contrasts_sweep_held", "dry_run"), ("tick", "dry_run"),
            ("daemon_started", "real"), ("contrasts_sweep_held", "real"), ("tick", "real"), ("tick", "real"), ("tick", "unclassified")]


def test_old_style_history_is_classified_by_run_context_dry_run_windows_are_noise_real_windows_are_evidence_and_orphan_ticks_are_reported_not_counted(tmp_path):
    gd = _load_daemon_module()
    st = tmp_path / "state"; st.mkdir()
    hold = str(st / gd.CONTRASTS_SWEEP_HOLD); reason = gd.CONTRASTS_SWEEP_HOLD_ARGV_REASON
    expect = _old_style_log(st, hold, reason)
    classes = gd.sweephold_classify(str(st))
    assert [(c["event"], c["class"]) for c in classes] == expect, classes
    assert classes[0]["window"] is None and classes[2]["window"] == "2026-10-04T06:00:00.000Z" and classes[8]["window"] == "2026-10-04T07:00:00.000Z" and classes[-1]["window"] is None
    hist = gd.sweephold_history(str(st))
    assert hist["evidence"] == ["event log: 1 daemon_started event(s) carrying contrasts_sweep_hold_seeded / contrasts_sweep_hold_seeding (first at 2026-10-04T07:00:00.000Z)",
                                "event log: 1 contrasts_sweep_held event(s)", "event log: 2 tick event(s) carrying contrasts_sweep_hold"]   # the REAL window only
    assert hist["dry_run_events_skipped"] == 5                                                           # the dry-run start-up + its 3 unstamped ticks + its held event
    assert hist["unclassified"] == {"starts": 0, "ticks": 2, "held_events": 0, "first_utc": "2026-10-04T05:59:00.000Z", "last_utc": "2026-10-04T07:04:00.000Z", "note": hist["unclassified"]["note"]}   # fix-5: starts counted too
    assert "not counted as evidence" in hist["unclassified"]["note"]
    # the flagged start: ambiguous on the real evidence (as before), the unclassified history reported beside it, nothing written
    a = gd.contrasts_sweep_hold_initial(type("A", (), {"contrasts_sweep": False, "state_dir": str(st), "dry_run": False})())
    assert a["skipped"] == "ambiguous_legacy_state" and a["evidence"] == hist["evidence"] and a["unclassified"]["ticks"] == 2 and a["dry_run_events_skipped"] == 5
    assert "touch" in a["operator_decision"] and not (st / gd.CONTRASTS_SWEEP_HOLD).exists() and not (st / gd.CONTRASTS_SWEEP_HOLD_SEEDED).exists()
    # the fix-3 reading of the same log, for the record: it would have counted the dry-run window's three unstamped ticks and the two orphans as evidence
    assert sum(1 for c in classes if c["event"] == "tick" and c["class"] != "real") == 5


def test_a_dry_run_window_alone_leaves_the_first_real_flagged_start_free_to_seed_and_an_orphan_tick_alone_is_the_operators_call(tmp_path):
    gd = _load_daemon_module()
    st = tmp_path / "state"; st.mkdir()
    hold = str(st / gd.CONTRASTS_SWEEP_HOLD); reason = gd.CONTRASTS_SWEEP_HOLD_ARGV_REASON
    dry_sha = "6" * 40
    # (b) a fix-1-era dry-run start-up followed by UNSTAMPED ticks whose sweep does not say held_by_flag_dry_run: noise by the window alone
    (st / "grade_daemon.jsonl").write_text("".join([
        _line({"event": "daemon_started", "daemon_sha": dry_sha, "pid": 4043, "once": False, "contrasts_sweep_hold_seeded": None,
               "contrasts_sweep_hold_seeding": {"seeded_now": None, "skipped": "dry_run"}, "utc": "2026-10-04T08:00:00.000Z"}),
        _line({"event": "tick", "dispatched": 0, "contrasts_sweep_hold": {"held": False, "reason": None, "sweep": "ran"}, "daemon_sha": dry_sha, "utc": "2026-10-04T08:01:00.000Z"}),
        _line({"event": "contrasts_sweep_held", "reason": reason, "file": hold, "utc": "2026-10-04T08:02:00.000Z"}),
        _line({"event": "tick", "dispatched": 0, "contrasts_sweep_hold": {"held": True, "reason": reason, "sweep": "skipped_once_per_advance"}, "daemon_sha": dry_sha, "utc": "2026-10-04T08:02:00.500Z"})]))
    assert [c["class"] for c in gd.sweephold_classify(str(st))] == ["dry_run"] * 4
    assert gd.sweephold_history(str(st)) == {"evidence": [], "dry_run_events_skipped": 4} and gd.sweephold_evidence(str(st)) == []
    a = gd.contrasts_sweep_hold_initial(type("A", (), {"contrasts_sweep": False, "state_dir": str(st), "dry_run": False})())
    assert a["seeded_now"] == hold and a["skipped"] is None and a["dry_run_events_skipped"] == 4 and "unclassified" not in a                 # a real start seeds
    (st / gd.CONTRASTS_SWEEP_HOLD).unlink(); (st / gd.CONTRASTS_SWEEP_HOLD_SEEDED).unlink()
    # (d) orphan ticks ONLY (their start-up outside the scanned text): no evidence, reported, and the flag is NOT applied -- the operator decides
    (st / "grade_daemon.jsonl").write_text("".join([
        _line({"event": "tick", "dispatched": 0, "contrasts_sweep_hold": {"held": True, "reason": reason, "sweep": "held_by_file"}, "daemon_sha": dry_sha, "utc": "2026-10-04T09:00:00.000Z"}),
        _line({"event": "contrasts_sweep_held", "reason": reason, "file": hold, "utc": "2026-10-04T09:01:00.000Z"}),
        _line({"event": "tick", "dispatched": 0, "contrasts_sweep_hold": {"held": True, "reason": reason, "sweep": "held_by_file"}, "daemon_sha": dry_sha, "utc": "2026-10-04T09:01:00.500Z"})]))
    assert [c["class"] for c in gd.sweephold_classify(str(st))] == ["unclassified"] * 3
    hist = gd.sweephold_history(str(st))
    assert hist["evidence"] == [] and hist["dry_run_events_skipped"] == 0 and hist["unclassified"]["ticks"] == 2 and hist["unclassified"]["held_events"] == 1
    assert (hist["unclassified"]["first_utc"], hist["unclassified"]["last_utc"]) == ("2026-10-04T09:00:00.000Z", "2026-10-04T09:01:00.500Z")
    b = gd.contrasts_sweep_hold_initial(type("A", (), {"contrasts_sweep": False, "state_dir": str(st), "dry_run": False})())
    assert b["skipped"] == gd.SWEEPHOLD_SKIP_UNCLASSIFIED == "unclassifiable_legacy_history" and b["evidence"] == [] and b["unclassified"] == hist["unclassified"]
    assert "2 tick(s) and 1 contrasts_sweep_held event(s)" in b["operator_decision"] and "2026-10-04T09:00:00.000Z" in b["operator_decision"] and "touch" in b["operator_decision"]
    assert not (st / gd.CONTRASTS_SWEEP_HOLD).exists() and not (st / gd.CONTRASTS_SWEEP_HOLD_SEEDED).exists()                           # nothing written: the flag did not turn on
    # the operator's record ends it, as for ambiguous_legacy_state
    (st / gd.CONTRASTS_SWEEP_HOLD_SEEDED).write_text(json.dumps({"acknowledged_by": "operator"}) + "\n")
    c = gd.contrasts_sweep_hold_initial(type("A", (), {"contrasts_sweep": False, "state_dir": str(st), "dry_run": False})())
    assert c["seeded_before"] == {"acknowledged_by": "operator"} and c["skipped"] is None
    # a daemon_exit of ANOTHER pid does not close a window whose events carry no pid; the same pid does
    (st / gd.CONTRASTS_SWEEP_HOLD_SEEDED).unlink()
    (st / "grade_daemon.jsonl").write_text("".join([
        _line({"event": "daemon_started", "daemon_sha": "7" * 40, "pid": 7001, "once": False, "contrasts_sweep_hold_seeded": hold, "contrasts_sweep_hold_seeding": {"seeded_now": hold, "skipped": None}, "utc": "2026-10-04T10:00:00.000Z"}),
        _line({"event": "daemon_exit", "daemon_sha": "7" * 40, "pid": 7999, "utc": "2026-10-04T10:00:30.000Z"}),
        _line({"event": "tick", "dispatched": 0, "contrasts_sweep_hold": {"held": True, "reason": reason, "sweep": "held_by_file"}, "daemon_sha": "7" * 40, "utc": "2026-10-04T10:01:00.000Z"}),
        _line({"event": "daemon_exit", "daemon_sha": "7" * 40, "pid": 7001, "utc": "2026-10-04T10:02:00.000Z"}),
        _line({"event": "tick", "dispatched": 0, "contrasts_sweep_hold": {"held": True, "reason": reason, "sweep": "held_by_file"}, "daemon_sha": "7" * 40, "utc": "2026-10-04T10:03:00.000Z"})]))
    assert [(c["event"], c["class"]) for c in gd.sweephold_classify(str(st))] == [("daemon_started", "real"), ("tick", "real"), ("tick", "unclassified")]


# ---------------------------------------------------------------------------------------------- sweephold-fix-5: no real-by-default
# fix-4 defaulted every unstamped daemon_started that did not say `skipped: dry_run` to a REAL window. Two actual predecessor shapes defeat
# that (shapes derived from those commits' CODE -- synthetic sha / pid / utc, no run ids; fix-6 corrected the attribution: the live dir's log carries
# neither as such -- its unreadable start is the fix-1-era `no_flag` shape, (B) without `evidence`, with 213 unstamped ticks, beside a stamped fix-3+
# start, the two shapes the fix-6 fixtures below reproduce key-for-key): (A) f03e6ee76, the first sweephold -- a `--dry-run
# --no-contrasts-sweep` start wrote `contrasts_sweep_hold_seeded: null` and NO seeding dict, then unstamped ticks carrying contrasts_sweep_hold
# with no `sweep` status; (B) 00cf8b96c, fix-1 -- a `--dry-run` WITHOUT the flag wrote `seeding.skipped: no_flag` (the no-flag return precedes
# the dry-run check), then unstamped ticks. Either, followed by the first genuine `--no-contrasts-sweep` start, read as real evidence ->
# ambiguous_legacy_state -> nothing written -> the once-per-advance sweep RAN on a flagged start. Now an old start without an AFFIRMATIVE
# discriminator (a dry_run stamp, the fix-2 `skipped: dry_run`, or a seeding that wrote a file) is unclassified, and so are its ticks; the
# decision is the operator's; and the flagged start HOLDS the sweep (`held_by_flag_pending_decision`) until the operator decides.

SHA_F03 = "f03e6ee76c8fe71538ffeaa46155e32d1eb6f78c"; SHA_00C = "00cf8b96c" + "a" * 31


def _shape_a_lines():
    """(A) f03e6ee76: `--dry-run --no-contrasts-sweep` -- daemon_started with contrasts_sweep_hold_seeded null and no seeding dict (that commit logged
    `contrasts_sweep_hold_seeded` alone, verified from its code); ticks with contrasts_sweep_hold {held, reason} only (no `sweep` key yet, no stamp).
    fix-6: `once: false` -- a `--once` daemon runs ONE tick, so the two ticks below could never follow `once: true`; and this is a code-derived shape,
    not one the live dir's log carries (its legacy start is shape (B) with 213 ticks: the fix-6 fixtures)."""
    return [_line({"event": "daemon_started", "daemon_sha": SHA_F03, "pid": 4301, "once": False, "contrasts_sweep_hold_seeded": None, "utc": "2026-10-04T04:00:00.000Z"}),
            _line({"event": "tick", "dispatched": 0, "contrasts_sweep_hold": {"held": False, "reason": None}, "daemon_sha": SHA_F03, "utc": "2026-10-04T04:00:01.000Z"}),
            _line({"event": "tick", "dispatched": 0, "contrasts_sweep_hold": {"held": False, "reason": None}, "daemon_sha": SHA_F03, "utc": "2026-10-04T04:02:01.000Z"})]


def _shape_b_lines():
    """(B) 00cf8b96c: fix-1 -- an unstamped daemon_started whose seeding dict says `skipped: no_flag` (the no-flag return preceded the dry-run check,
    so a dry run and a real run wrote the same line), then unstamped ticks whose sweep status reads ran / skipped_once_per_advance. fix-6: the
    seeding dict is fix-1's FOUR keys {hold_present, seeded_before, seeded_now, skipped} -- `evidence` arrived with fix-2 and fix-1 never wrote it --
    which is exactly the live dir's legacy start (a1f0341e0-era code, fix-1's shape; the fix-6 fixtures carry it with its 213 ticks)."""
    seeding = {"seeded_now": None, "seeded_before": None, "hold_present": False, "skipped": "no_flag"}
    return [_line({"event": "daemon_started", "daemon_sha": SHA_00C, "pid": 4302, "once": False, "contrasts_sweep_hold_seeded": None, "contrasts_sweep_hold_seeding": seeding,
                   "utc": "2026-10-04T04:30:00.000Z"}),
            _line({"event": "tick", "dispatched": 0, "contrasts_sweep_hold": {"held": False, "reason": None, "sweep": "ran"}, "daemon_sha": SHA_00C, "utc": "2026-10-04T04:30:01.000Z"}),
            _line({"event": "tick", "dispatched": 0, "contrasts_sweep_hold": {"held": False, "reason": None, "sweep": "skipped_once_per_advance"}, "daemon_sha": SHA_00C,
                   "utc": "2026-10-04T04:32:01.000Z"})]


def _flagged_start_over(gd, tmp_path, monkeypatch, lines, *, start_utc, first_utc, last_utc):
    """The predecessor's log, then the first GENUINE `--no-contrasts-sweep` start on that state dir -- main()'s seeding decision and its first tick,
    in process. Asserts: every old event unclassified (the ticks placed in the start's window, yet unclassified with it), no real evidence, the
    operator path (unclassifiable_legacy_history, nothing written), and the sweep HELD on the flagged start -- never `ran`, no mark, no
    sweep_noncurrent event. Returns (st, args, rest, decision, tick summary)."""
    st = tmp_path / "state"; st.mkdir(exist_ok=True)
    (st / "grade_daemon.jsonl").write_text("".join(lines))
    classes = gd.sweephold_classify(str(st))
    assert [c["class"] for c in classes] == ["unclassified"] * len(lines) and [c["event"] for c in classes] == ["daemon_started"] + ["tick"] * (len(lines) - 1), classes
    assert all(c["window"] == start_utc for c in classes)                                      # placed in the start's window -- and unclassified WITH it
    hist = gd.sweephold_history(str(st))
    assert hist["evidence"] == [] and hist["dry_run_events_skipped"] == 0 and gd.sweephold_evidence(str(st)) == []
    assert hist["unclassified"] == {"starts": 1, "ticks": len(lines) - 1, "held_events": 0, "first_utc": first_utc, "last_utc": last_utc, "note": hist["unclassified"]["note"]}
    assert "affirmative discriminator" in hist["unclassified"]["note"]
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, _rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep"); _grader_env(monkeypatch, env)
    assert args.contrasts_sweep is False and args.dry_run is False
    dec = gd.contrasts_sweep_hold_initial(args)                                                    # what main() decides at the genuine flagged start
    assert dec["skipped"] == gd.SWEEPHOLD_SKIP_UNCLASSIFIED and dec["evidence"] == [] and dec["unclassified"] == hist["unclassified"] and dec["seeded_now"] is None
    assert f"1 daemon_started event(s), {len(lines) - 1} tick(s) and 0 contrasts_sweep_held event(s)" in dec["operator_decision"] and "held in this process" in dec["operator_decision"]
    assert not _hold_path(tmp_path).exists() and not _seeded_path(tmp_path).exists()               # nothing written: the flag never turns on from unclassifiable history
    n = len(_events(tmp_path))
    s = gd.tick(args, *rest)
    ev = _events(tmp_path)[n:]
    assert s["contrasts_sweep_hold"] == {"held": False, "reason": None, "sweep": gd.SWEEP_STATUS_HELD_FLAG_PENDING} and gd.SWEEP_STATUS_HELD_FLAG_PENDING == "held_by_flag_pending_decision"
    assert s["contrasts_sweep_hold"]["sweep"] != "ran" and s["dry_run"] is False
    (why,) = s["contrasts_sweep"].values(); assert list(s["contrasts_sweep"]) == ["held_by_flag_pending_decision"]
    assert "nobody can classify" in why and "1 daemon_started" in why and "touch" in why and str(_seeded_path(tmp_path)) in why
    assert not [e for e in ev if e["event"] == "sweep_noncurrent"] and not _mark_path(tmp_path).exists()      # the sweep did not run: no event, no mark
    assert not _hold_path(tmp_path).exists() and not _seeded_path(tmp_path).exists()               # and the tick wrote neither file either
    return st, args, rest, dec, s


def test_the_first_sweephold_dry_audit_shape_a_is_unclassified_never_real_and_the_first_flagged_start_holds_the_sweep_until_the_operator_decides(tmp_path, monkeypatch):
    gd = _load_daemon_module()
    st, args, rest, dec, s = _flagged_start_over(gd, tmp_path, monkeypatch, _shape_a_lines(), start_utc="2026-10-04T04:00:00.000Z",
                                                 first_utc="2026-10-04T04:00:00.000Z", last_utc="2026-10-04T04:02:01.000Z")
    # a GENUINE real predecessor beside the unreadable one (a fix-3+ start, stamped dry_run: false, no flag -- the live log's newest start shape):
    # real evidence, so the decision is ambiguous_legacy_state WITH the unclassified block reported -- and the sweep is still held
    real = [_line({"event": "daemon_started", "daemon_sha": "9" * 40, "pid": 9001, "once": False, "dry_run": False, "contrasts_sweep_hold_seeded": None,
                   "contrasts_sweep_hold_seeding": {"evidence": [], "seeded_now": None, "seeded_before": None, "hold_present": False, "skipped": "no_flag"},   # fix-6: a fix-3+ start writes evidence: [] (31aa8f880's live shape)
                   "utc": "2026-10-04T05:00:00.000Z"}),
            _line({"event": "tick", "dry_run": False, "dispatched": 0, "contrasts_sweep_hold": {"held": False, "reason": None, "sweep": "ran"}, "daemon_sha": "9" * 40, "utc": "2026-10-04T05:00:01.000Z"})]
    with open(st / "grade_daemon.jsonl", "a") as fh:
        fh.write("".join(real))
    classes = gd.sweephold_classify(str(st))
    assert [c["class"] for c in classes if c["utc"].startswith("2026-10-04T05:00")] == ["real", "real"]          # the stamped start and its tick: real by the stamp
    assert [c["class"] for c in classes if c["utc"].startswith("2026-10-04T04:0")] == ["unclassified"] * 3        # the old ones stay unclassified
    live = [c for c in classes if not c["utc"].startswith("2026-10-04T0")]                                        # the genuine flagged start's own tick above, stamped dry_run: false
    assert [(c["event"], c["class"]) for c in live] == [("tick", "real")]
    hist = gd.sweephold_history(str(st))
    assert hist["evidence"] == ["event log: 1 daemon_started event(s) carrying contrasts_sweep_hold_seeded / contrasts_sweep_hold_seeding (first at 2026-10-04T05:00:00.000Z)",
                                "event log: 2 tick event(s) carrying contrasts_sweep_hold"] and hist["unclassified"]["starts"] == 1 and hist["unclassified"]["ticks"] == 2
    dec2 = gd.contrasts_sweep_hold_initial(args)
    assert dec2["skipped"] == "ambiguous_legacy_state" and dec2["unclassified"] == hist["unclassified"] and not _hold_path(tmp_path).exists() and not _seeded_path(tmp_path).exists()
    s2 = gd.tick(args, *rest)
    assert s2["contrasts_sweep_hold"]["sweep"] == gd.SWEEP_STATUS_HELD_FLAG_PENDING and not _mark_path(tmp_path).exists()
    # the operator acknowledges the released state (the record): the hold is spent, the sweep runs once per advance, the mark appears
    _seeded_path(tmp_path).write_text(json.dumps({"acknowledged_by": "operator"}) + "\n")
    n = len(_events(tmp_path))
    s3 = gd.tick(args, *rest)
    assert s3["contrasts_sweep_hold"] == {"held": False, "reason": None, "sweep": "ran"} and [e["event"] for e in _events(tmp_path)[n:] if e["event"] == "sweep_noncurrent"] == ["sweep_noncurrent"]
    assert json.loads(_mark_path(tmp_path).read_text())["daemon_sha"] == gd.IMPORTED_CODE_SHA
    assert gd.contrasts_sweep_hold_initial(args)["seeded_before"] == {"acknowledged_by": "operator"}


def test_a_fix1_dry_run_without_the_flag_shape_b_is_unclassified_never_real_and_the_first_flagged_start_holds_the_sweep_until_the_operator_decides(tmp_path, monkeypatch):
    gd = _load_daemon_module()
    st, args, rest, dec, s = _flagged_start_over(gd, tmp_path, monkeypatch, _shape_b_lines(), start_utc="2026-10-04T04:30:00.000Z",
                                                 first_utc="2026-10-04T04:30:00.000Z", last_utc="2026-10-04T04:32:01.000Z")
    s2 = gd.tick(args, *rest)                                                                      # held again: the decision is still pending
    assert s2["contrasts_sweep_hold"]["sweep"] == gd.SWEEP_STATUS_HELD_FLAG_PENDING and not _mark_path(tmp_path).exists()
    # the operator holds: a touch of the hold file -> held by the file at the next tick (the file alone decides), still nothing seeded by the daemon
    _hold_path(tmp_path).write_text("ops: hold it\n")
    s3 = gd.tick(args, *rest)
    assert s3["contrasts_sweep_hold"] == {"held": True, "reason": "ops: hold it", "sweep": "held_by_file"} and not _seeded_path(tmp_path).exists()
    # ...then releases and acknowledges: the sweep runs
    _hold_path(tmp_path).unlink(); _seeded_path(tmp_path).write_text(json.dumps({"acknowledged_by": "operator"}) + "\n")
    s4 = gd.tick(args, *rest)
    assert s4["contrasts_sweep_hold"] == {"held": False, "reason": None, "sweep": "ran"} and _mark_path(tmp_path).exists()
    # a dry-run flagged start over the same unreadable history holds by its OWN status (fix-3), not the pending one; without the flag nothing is held
    _seeded_path(tmp_path).unlink(); _mark_path(tmp_path).unlink()
    args_dry, rest_dry, env = _tick_args(gd, _root.__wrapped__(tmp_path) if hasattr(_root, "__wrapped__") else str(tmp_path / "reg"), tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep", dry_run=True)
    assert gd.tick(args_dry, *rest_dry)["contrasts_sweep_hold"]["sweep"] == "held_by_flag_dry_run"
    args_noflag, rest_noflag, env = _tick_args(gd, str(tmp_path / "reg"), tmp_path, *_mirror_args(tmp_path))
    assert args_noflag.contrasts_sweep is True and gd.contrasts_sweep_flag_pending(args_noflag) is None and gd.tick(args_noflag, *rest_noflag)["contrasts_sweep_hold"]["sweep"] == "ran"
    # the discriminator itself, by shape
    assert gd._sweephold_event_class({"event": "daemon_started", "dry_run": False, "contrasts_sweep_hold_seeded": None}) == "real"                 # the fix-3+ stamp
    assert gd._sweephold_event_class({"event": "daemon_started", "contrasts_sweep_hold_seeded": "/x/contrasts_sweep_hold"}) == "real"               # f03e6ee76, flagged for real: it seeded
    assert gd._sweephold_event_class({"event": "daemon_started", "contrasts_sweep_hold_seeding": {"seeded_now": "/x/h", "skipped": None}}) == "real"
    assert gd._sweephold_event_class({"event": "daemon_started", "contrasts_sweep_hold_seeding": {"seeded_now": None, "seeding_recorded": True}}) == "real"
    assert gd._sweephold_event_class({"event": "daemon_started", "dry_run": True}) == "dry_run" and gd._sweephold_event_class({"event": "daemon_started", "contrasts_sweep_hold_seeding": {"skipped": "dry_run"}}) == "dry_run"
    assert gd._sweephold_event_class({"event": "tick", "contrasts_sweep_hold": {"held": False, "reason": None, "sweep": "held_by_flag_dry_run"}}) == "dry_run"
    assert gd._sweephold_event_class({"event": "daemon_started", "contrasts_sweep_hold_seeded": None}) is None                                       # (A): says nothing
    assert gd._sweephold_event_class({"event": "daemon_started", "contrasts_sweep_hold_seeded": None, "contrasts_sweep_hold_seeding": {"skipped": "no_flag"}}) is None   # (B): says nothing
    assert gd._sweephold_event_class({"event": "daemon_started", "contrasts_sweep_hold_seeding": {"seeded_before": {"utc": "x"}, "skipped": None}}) == "real"   # past the dry-run return at fix-1 / fix-2: real by construction
    assert gd._sweephold_event_class({"event": "daemon_started", "contrasts_sweep_hold_seeding": {"seeded_now": None, "skipped": "ambiguous_legacy_state"}}) == "real"   # likewise fix-2's ambiguous return
    assert gd._sweephold_event_class({"event": "daemon_started", "contrasts_sweep_hold_seeding": {"seeded_now": None, "skipped": "no_state_dir"}}) is None          # before the dry-run check: says nothing
    assert gd._sweephold_event_class({"event": "tick", "contrasts_sweep_hold": {"held": True, "reason": "r"}}) is None and gd._sweephold_event_class({"event": "tick", "dry_run": False, "contrasts_sweep_hold": {}}) == "real"


# ---------------------------------------------------------------------------------------------- sweephold-fix-6: the pending decision persists
# fix-5 re-derived its only guard, `contrasts_sweep_flag_pending`, EVERY tick from sweephold_history -- the bounded scan of the log's first and
# last SWEEPHOLD_EVIDENCE_SCAN_BYTES. On the live dir the 214 unclassified events (one unstamped fix-1-era `no_flag` start and its 213 unstamped
# ticks) sit mid-log, none in the head, with no hold file and no .seeded record: a flagged start held correctly, but ~4 MB more of stamped ticks
# and the tail window passes them, pending derives False with no operator decision ever made, and the sweep runs on a flagged start; a re-exec
# re-derives the same non-durable state. Now the FIRST derivation (the start-up seeding path or the tick) persists
# `<state-dir>/contrasts_sweep_pending_decision` (tmp + os.replace; never in a dry run); the record holds a flagged non-dry tick whatever the scan
# sees and a flagged start seeds nothing while it stands; only the operator clears it -- the .seeded record (honoured: removed, the sweep runs
# once per advance) -- and the hold file supersedes it while present (held_by_file; the record stays). Fixtures: the live log's two start shapes,
# surveyed READ-ONLY and reproduced key-for-key with synthetic values (invented shas, pids, utcs; no run ids) -- (L) the fix-1-era start
# a1f0341e0's code wrote: UNSTAMPED daemon_started {contrasts_sweep_hold_seeded: null, contrasts_sweep_hold_seeding: {hold_present, seeded_before,
# seeded_now, skipped: no_flag} (no `evidence`: fix-2 added it), once: false}, then 213 unstamped ticks carrying contrasts_sweep_hold {held,
# reason, sweep} (1 ran / 7 skipped_once_per_advance / 2 operator_held / 203 schema_held, the live counts); (S) the fix-3+ start 31aa8f880's
# code wrote: STAMPED dry_run: false, seeding {evidence: [], hold_present, seeded_before, seeded_now, skipped: no_flag}, once: false, then
# stamped ticks.

SHA_LEGACY = "a1f0341e0" + "b" * 31; SHA_STAMPED = "31aa8f880" + "c" * 31          # invented 40-hex shas whose prefixes name the era of the shape
LIVE_START_KEYS_LEGACY = frozenset({"contrasts_sweep_hold_seeded", "contrasts_sweep_hold_seeding", "daemon_sha", "event", "once", "pid", "utc"})
LIVE_START_KEYS_STAMPED = LIVE_START_KEYS_LEGACY | {"dry_run"}
LIVE_SEEDING_KEYS_LEGACY = frozenset({"hold_present", "seeded_before", "seeded_now", "skipped"}); LIVE_SEEDING_KEYS_STAMPED = LIVE_SEEDING_KEYS_LEGACY | {"evidence"}
_TICK_KEYS_COMMON = frozenset({"claim_retry_later", "confirm_only", "contrasts_sweep_hold", "daemon_sha", "dispatched", "event", "failed", "held", "identity_changed",
                               "identity_held", "not_folded", "pending", "problems", "refused", "retryable", "rows_cleared", "rows_exhausted", "rows_failed",
                               "rows_pending", "rows_retried", "schema_held", "skipped_identity", "skipped_locked", "skipped_low_disk", "supersessions_reconciled",
                               "supersessions_refused", "unknown", "utc"})
LIVE_TICK_KEYS_SCHEMA_HELD = _TICK_KEYS_COMMON | {"held_reason", "hold", "schema", "schema_hold_since_utc"}                     # the schema-preflight early return (32 keys)
LIVE_TICK_KEYS_FULL = _TICK_KEYS_COMMON | {"fold_mode", "held_families", "held_family_runs", "mirror_claims_deferred", "mirror_claims_reconciled", "mirror_du_bytes",
                                           "mirror_free_bytes", "mirror_prune_pending", "mirror_pruned_bytes", "mirror_pruned_dirs", "mirror_retired_bytes",
                                           "mirror_retired_dirs", "quarantined", "reclaimed", "spooled_hold", "spooled_rebuilt"}     # a full tick (44 keys)


def _live_tick(sha, utc, sweep, *, schema_held=False, operator_held=False, dry_run=None):
    """A tick in the live log's shape -- the key set reproduced exactly (LIVE_TICK_KEYS_*), the values synthetic: the schema-preflight early return
    (most of the legacy window's 213) or a full tick (operator_held adds hold / held_reason to it); unstamped (fix-1 era) or stamped `dry_run` (fix-3+)."""
    ev = {"event": "tick", "claim_retry_later": {}, "confirm_only": 0, "contrasts_sweep_hold": {"held": False, "reason": None, "sweep": sweep}, "daemon_sha": sha,
          "dispatched": 0, "failed": 0, "held": False, "identity_changed": 0, "identity_held": 0, "not_folded": 0, "pending": 0, "problems": [], "refused": 0,
          "retryable": 0, "rows_cleared": 0, "rows_exhausted": 0, "rows_failed": 0, "rows_pending": 0, "rows_retried": 0, "schema_held": 0, "skipped_identity": 0,
          "skipped_locked": 0, "skipped_low_disk": 0, "supersessions_reconciled": 0, "supersessions_refused": 0, "unknown": 0, "utc": utc}
    if schema_held:
        ev.update(held=True, hold="schema", held_reason="registry schema 23 under code 24 (synthetic)", schema={"kind": "version", "have": 23, "want": 24, "n_drift": 0},
                  schema_hold_since_utc=utc, problems=["schema_hold: registry schema 23 under code 24 (synthetic)"])
        assert set(ev) == LIVE_TICK_KEYS_SCHEMA_HELD
    else:
        ev.update(fold_mode="registry", held_families=[], held_family_runs={}, mirror_claims_deferred=0, mirror_claims_reconciled=0, mirror_du_bytes={},
                  mirror_free_bytes={}, mirror_prune_pending=0, mirror_pruned_bytes=0, mirror_pruned_dirs=0, mirror_retired_bytes=0, mirror_retired_dirs=0,
                  quarantined=0, reclaimed=0, spooled_hold=0, spooled_rebuilt=0)
        assert set(ev) == LIVE_TICK_KEYS_FULL
        if operator_held:
            ev.update(held=True, hold="operator", held_reason="operator: paused (synthetic)")
    if dry_run is not None:
        ev["dry_run"] = dry_run
    return ev


def _legacy_window_lines(n_ticks=213):
    """(L) the fix-1-era start and its unstamped ticks as the live log carries them (1 ran -- with its `contrasts_sweep` counts block -- then 7
    skipped_once_per_advance, 2 operator_held, the rest schema_held). Returns (lines, first_utc, last_utc)."""
    start = {"event": "daemon_started", "daemon_sha": SHA_LEGACY, "pid": 4601, "once": False, "contrasts_sweep_hold_seeded": None,
             "contrasts_sweep_hold_seeding": {"hold_present": False, "seeded_before": None, "seeded_now": None, "skipped": "no_flag"}, "utc": "2026-10-04T14:00:00.000Z"}
    assert set(start) == LIVE_START_KEYS_LEGACY and set(start["contrasts_sweep_hold_seeding"]) == LIVE_SEEDING_KEYS_LEGACY
    ticks = []
    for i in range(n_ticks):
        h, m = divmod(2 * i, 60); utc = f"2026-10-04T{14 + h:02d}:{m:02d}:01.000Z"
        if i == 0:
            t = _live_tick(SHA_LEGACY, utc, "ran")
            t["contrasts_sweep"] = {"cap_mismatch": 0, "daemon_sha": SHA_LEGACY, "deferred": 0, "dry_run": False, "due": 0, "emitted": 0, "first": 0, "no_anchor": 0,
                                    "rejudged": 0, "retired": 0, "was_daemon_sha": None}
        elif i <= 7:
            t = _live_tick(SHA_LEGACY, utc, "skipped_once_per_advance")
        elif i <= 9:
            t = _live_tick(SHA_LEGACY, utc, "operator_held", operator_held=True)
        else:
            t = _live_tick(SHA_LEGACY, utc, "schema_held", schema_held=True)
        ticks.append(t)
    return [_line(start)] + [_line(t) for t in ticks], start["utc"], ticks[-1]["utc"]


def _stamped_start_line(utc="2026-10-04T21:53:00.000Z"):
    """(S) the fix-3+ start, stamped dry_run: false, its seeding dict carrying evidence: [] -- the live log's newest start shape."""
    start = {"event": "daemon_started", "daemon_sha": SHA_STAMPED, "pid": 4602, "once": False, "dry_run": False, "contrasts_sweep_hold_seeded": None,
             "contrasts_sweep_hold_seeding": {"evidence": [], "hold_present": False, "seeded_before": None, "seeded_now": None, "skipped": "no_flag"}, "utc": utc}
    assert set(start) == LIVE_START_KEYS_STAMPED and set(start["contrasts_sweep_hold_seeding"]) == LIVE_SEEDING_KEYS_STAMPED
    return _line(start)


def _stamped_tick_line(i, sweep="skipped_once_per_advance"):
    """A stamped (dry_run: false) real tick of the fix-3+ daemon, the i-th after its start."""
    d, rem = divmod(i, 24 * 30); h, m = divmod(rem, 30)
    return _line(_live_tick(SHA_STAMPED, f"2026-10-{5 + d:02d}T{h:02d}:{2 * m:02d}:03.000Z", sweep, dry_run=False))


def _pad_lines(n_bytes):
    """Pre-sweephold volume AHEAD of the legacy window (the live log's head is ~5 MB of pre-sweephold events): ticks without contrasts_sweep_hold,
    which the scan prefilters by substring; ~4 KiB each so the whole-log readers stay quick. Enough lines to exceed ``n_bytes``."""
    pad = _line({"event": "tick", "dispatched": 0, "pending": 0, "daemon_sha": "0" * 40, "utc": "2026-10-03T00:00:00.000Z", "pad": "." * 4000})
    return [pad] * (n_bytes // len(pad) + 1)


def _pending_path(tmp_path, gd):
    return tmp_path / "state" / gd.CONTRASTS_SWEEP_PENDING_DECISION


def test_a_flagged_start_over_the_live_shapes_stays_held_across_eviction_from_the_bounded_scan_and_a_re_exec_until_the_operator_decides(tmp_path, monkeypatch):
    """The finding: the live dir's history -- pre-sweephold volume, then the unreadable fix-1-era window (one `no_flag` start, 213 unstamped
    ticks) beyond the head window, then the stamped fix-3+ start -- under a flagged start. The pending decision is derived once and PERSISTED; stamped
    ticks then push the unreadable window out of the tail window (the scan sees no unclassified event any more -- fix-5's guard would release here);
    a re-exec (in process, and a real subprocess restart with the flag) still holds, seeds nothing, sweeps nothing, writes no mark. The hold file
    supersedes the record without removing it; the operator's .seeded record removes it (said once) and the sweep runs once per advance."""
    gd = _load_daemon_module()
    assert gd.CONTRASTS_SWEEP_PENDING_DECISION == "contrasts_sweep_pending_decision"
    st = tmp_path / "state"; st.mkdir(); log = st / "grade_daemon.jsonl"; rec_path = _pending_path(tmp_path, gd); scan = gd.SWEEPHOLD_EVIDENCE_SCAN_BYTES
    legacy, first_utc, last_utc = _legacy_window_lines()
    with open(log, "w") as fh:
        fh.write("".join(_pad_lines(scan + 64 * 1024))); head_end = fh.tell()
        fh.write("".join(legacy)); legacy_end = fh.tell()
        fh.write(_stamped_start_line() + "".join(_stamped_tick_line(i, "ran" if i == 0 else "skipped_once_per_advance") for i in range(3)))
    assert head_end > scan                                                                   # the unreadable window lies beyond the head window, as on the live dir
    hist = gd.sweephold_history(str(st))
    assert hist["unclassified"] == {"starts": 1, "ticks": 213, "held_events": 0, "first_utc": first_utc, "last_utc": last_utc, "note": hist["unclassified"]["note"]}
    assert hist["evidence"] == ["event log: 1 daemon_started event(s) carrying contrasts_sweep_hold_seeded / contrasts_sweep_hold_seeding (first at 2026-10-04T21:53:00.000Z)",
                                "event log: 3 tick event(s) carrying contrasts_sweep_hold"] and hist["dry_run_events_skipped"] == 0
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, _rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep"); _grader_env(monkeypatch, env)
    assert args.contrasts_sweep is False and args.dry_run is False
    # the flagged start (main()'s seeding decision): ambiguous on the stamped evidence with the unclassified block beside it -- and the pending decision PERSISTED
    n0 = len(_events(tmp_path))
    dec = gd.contrasts_sweep_hold_initial(args)
    assert dec["skipped"] == "ambiguous_legacy_state" and dec["unclassified"] == hist["unclassified"] and dec["seeded_now"] is None and dec["evidence"] == hist["evidence"]
    assert not _hold_path(tmp_path).exists() and not _seeded_path(tmp_path).exists() and rec_path.exists()
    rec = json.loads(rec_path.read_text()); raw = rec_path.read_bytes()
    assert dec["pending_decision"] == {"path": str(rec_path), "record": rec, "written_now": True} and str(rec_path) in dec["operator_decision"] and "written now" in dec["operator_decision"]
    assert "neither file was written" in dec["operator_decision"] and "nothing was written" not in dec["operator_decision"]
    assert set(rec) == {"utc", "daemon_sha", "pid", "unclassified", "evidence", "hold_file", "seeded_record", "operator_decision", "note"}
    assert rec["unclassified"] == {"starts": 1, "ticks": 213, "held_events": 0, "first_utc": first_utc, "last_utc": last_utc}       # as sweephold_history reported them
    assert rec["daemon_sha"] == gd.IMPORTED_CODE_SHA and rec["pid"] == os.getpid() and rec["utc"] and rec["evidence"] == hist["evidence"]
    assert rec["hold_file"] == str(_hold_path(tmp_path)) and rec["seeded_record"] == str(_seeded_path(tmp_path))
    assert "touch " + str(_hold_path(tmp_path)) in rec["operator_decision"] and "write " + str(_seeded_path(tmp_path)) in rec["operator_decision"] and "acknowledge" in rec["operator_decision"]
    recorded = [e for e in _events(tmp_path)[n0:] if e["event"] == "sweep_pending_decision_recorded"]
    assert len(recorded) == 1 and recorded[0]["path"] == str(rec_path) and recorded[0]["record"] == rec and recorded[0]["dry_run"] is False
    # tick 1: held pending; the block names the record's path and contents; the tick carries the record; the record is not rewritten
    n = len(_events(tmp_path)); s1 = gd.tick(args, *rest); ev1 = _events(tmp_path)[n:]
    assert s1["contrasts_sweep_hold"] == {"held": False, "reason": None, "sweep": "held_by_flag_pending_decision"} and list(s1["contrasts_sweep"]) == ["held_by_flag_pending_decision"]
    why = s1["contrasts_sweep"]["held_by_flag_pending_decision"]
    assert str(rec_path) in why and "1 daemon_started, 213 tick" in why and f"({first_utc} .. {last_utc})" in why and rec["utc"] in why and gd.IMPORTED_CODE_SHA in why
    assert "standing" in why and "written now" not in why and "touch" in why and str(_seeded_path(tmp_path)) in why
    assert s1["contrasts_sweep_pending_decision"] == {"path": str(rec_path), "record": rec, "written_now": False} and s1["dry_run"] is False
    assert _last_tick(ev1)["contrasts_sweep_pending_decision"]["record"] == rec
    assert not [e for e in ev1 if e["event"] in ("sweep_noncurrent", "sweep_pending_decision_recorded", "contrasts_sweep_held")] and not _mark_path(tmp_path).exists()
    assert rec_path.read_bytes() == raw and not _hold_path(tmp_path).exists() and not _seeded_path(tmp_path).exists()
    # EVICTION: stamped real ticks appended until the tail window passes the unreadable window -- the scan now sees NO unclassified event (fix-5's
    # guard would derive False here); the record is untouched
    with open(log, "a") as fh:
        i = 3
        while fh.tell() < legacy_end + scan + 64 * 1024:
            fh.write(_stamped_tick_line(i)); i += 1
    assert os.path.getsize(log) - scan > legacy_end
    hist2 = gd.sweephold_history(str(st))
    assert "unclassified" not in hist2 and hist2["dry_run_events_skipped"] == 0 and len(hist2["evidence"]) == 1     # the stamped start left the window with the legacy one; its ticks remain
    assert hist2["evidence"][0].startswith("event log: ") and hist2["evidence"][0].endswith(" tick event(s) carrying contrasts_sweep_hold")
    assert gd.sweephold_classify(str(st)) and all(c["class"] == "real" for c in gd.sweephold_classify(str(st)))
    assert gd.contrasts_sweep_pending_decision(str(st)) == rec and rec_path.read_bytes() == raw
    # the RE-EXEC, in process: a new start-up's seeding decision over the evicted scan and its tick -- the record holds, nothing seeded, no sweep, no mark
    dec2 = gd.contrasts_sweep_hold_initial(args)
    assert dec2["skipped"] == "ambiguous_legacy_state" and "unclassified" not in dec2 and dec2["seeded_now"] is None and "seeding_recorded" not in dec2
    assert dec2["pending_decision"] == {"path": str(rec_path), "record": rec, "written_now": False} and str(rec_path) in dec2["operator_decision"]
    assert "standing: derived " + rec["utc"] in dec2["operator_decision"] and not _hold_path(tmp_path).exists() and not _seeded_path(tmp_path).exists()
    n = len(_events(tmp_path)); s2 = gd.tick(args, *rest); ev2 = _events(tmp_path)[n:]
    assert s2["contrasts_sweep_hold"] == {"held": False, "reason": None, "sweep": "held_by_flag_pending_decision"} and s2["contrasts_sweep_pending_decision"]["record"] == rec
    assert not [e for e in ev2 if e["event"] in ("sweep_noncurrent", "sweep_pending_decision_recorded")] and not _mark_path(tmp_path).exists() and rec_path.read_bytes() == raw
    # ...and a real one: a unit restart with the flag (a subprocess over the same state dir) -- daemon_started names the record, its tick is held, no mark
    n = len(_events(tmp_path))
    r = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep"); assert r.returncode == 0, r.stderr + r.stdout
    ev3 = _events(tmp_path)[n:]; st3 = _started(ev3)
    assert st3["dry_run"] is False and st3["contrasts_sweep_hold_seeded"] is None and st3["contrasts_sweep_hold_seeding"]["skipped"] == "ambiguous_legacy_state"
    assert st3["contrasts_sweep_hold_seeding"]["pending_decision"] == {"path": str(rec_path), "record": rec, "written_now": False} and "unclassified" not in st3["contrasts_sweep_hold_seeding"]
    assert str(rec_path) in st3["contrasts_sweep_hold_seeding"]["operator_decision"]
    assert _last_tick(ev3)["contrasts_sweep_hold"]["sweep"] == "held_by_flag_pending_decision" and _last_tick(ev3)["contrasts_sweep_pending_decision"]["record"] == rec
    assert not [e for e in ev3 if e["event"] in ("sweep_noncurrent", "sweep_pending_decision_recorded", "sweep_pending_decision_cleared")] and not _mark_path(tmp_path).exists()
    assert not _hold_path(tmp_path).exists() and not _seeded_path(tmp_path).exists() and rec_path.read_bytes() == raw
    # the operator HOLDS: the hold file supersedes the record (held_by_file; the record stays); its removal without a .seeded record holds pending again
    _hold_path(tmp_path).write_text("ops: hold it\n")
    s4 = gd.tick(args, *rest)
    assert s4["contrasts_sweep_hold"] == {"held": True, "reason": "ops: hold it", "sweep": "held_by_file"} and "contrasts_sweep_pending_decision" not in s4 and rec_path.read_bytes() == raw
    dec4 = gd.contrasts_sweep_hold_initial(args)
    assert dec4["hold_present"] is True and dec4["pending_decision"]["record"] == rec and not _seeded_path(tmp_path).exists() and rec_path.read_bytes() == raw
    _hold_path(tmp_path).unlink()
    s5 = gd.tick(args, *rest)
    assert s5["contrasts_sweep_hold"]["sweep"] == "held_by_flag_pending_decision" and rec_path.read_bytes() == raw and not _mark_path(tmp_path).exists()
    # the operator ACKNOWLEDGES: the .seeded record -> the pending record is removed (one cleared event, the tick says so), the sweep runs once per advance
    _seeded_path(tmp_path).write_text(json.dumps({"acknowledged_by": "operator"}) + "\n")
    n = len(_events(tmp_path)); s6 = gd.tick(args, *rest); ev6 = _events(tmp_path)[n:]
    assert s6["contrasts_sweep_hold"] == {"held": False, "reason": None, "sweep": "ran"} and s6["contrasts_sweep"]["daemon_sha"] == gd.IMPORTED_CODE_SHA
    assert s6["contrasts_sweep_pending_decision"] == {"path": str(rec_path), "cleared": True, "by": "contrasts_sweep_hold.seeded"} and not rec_path.exists()
    cleared = [e for e in ev6 if e["event"] == "sweep_pending_decision_cleared"]
    assert len(cleared) == 1 and cleared[0]["path"] == str(rec_path) and cleared[0]["by"] == gd.CONTRASTS_SWEEP_HOLD_SEEDED and cleared[0]["dry_run"] is False
    assert [e["event"] for e in ev6 if e["event"] == "sweep_noncurrent"] == ["sweep_noncurrent"] and json.loads(_mark_path(tmp_path).read_text())["daemon_sha"] == gd.IMPORTED_CODE_SHA
    # a later flagged start: seeded_before, nothing pending, nothing to clear (said once), the once-per-advance rule
    dec7 = gd.contrasts_sweep_hold_initial(args)
    assert dec7["seeded_before"] == {"acknowledged_by": "operator"} and "pending_decision" not in dec7 and "pending_decision_cleared" not in dec7
    s7 = gd.tick(args, *rest)
    assert s7["contrasts_sweep_hold"] == {"held": False, "reason": None, "sweep": "skipped_once_per_advance"} and "contrasts_sweep_pending_decision" not in s7 and not rec_path.exists()


def test_a_dry_run_derives_no_pending_record_an_unflagged_daemon_is_not_bound_by_it_and_tick_and_start_up_each_persist_and_clear_it(tmp_path, monkeypatch):
    gd = _load_daemon_module()
    st = tmp_path / "state"; st.mkdir(); rec_path = _pending_path(tmp_path, gd)
    legacy, first_utc, last_utc = _legacy_window_lines()
    (st / "grade_daemon.jsonl").write_text("".join(legacy))                                   # the unreadable window alone: no real evidence
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, _rids = _root(tmp_path)
    # a --dry-run flagged audit: start-up and tick derive nothing and write nothing -- not the record either (and log no pending event)
    args_dry, rest_dry, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep", dry_run=True); _grader_env(monkeypatch, env)
    assert gd.contrasts_sweep_hold_initial(args_dry)["skipped"] == "dry_run" and not rec_path.exists()
    assert gd.contrasts_sweep_flag_pending(args_dry) is None and gd.tick(args_dry, *rest_dry)["contrasts_sweep_hold"]["sweep"] == "held_by_flag_dry_run" and not rec_path.exists()
    assert not [e for e in _events(tmp_path) if e["event"].startswith("sweep_pending_decision")]
    # the real flagged TICK (no start-up call first: the tick path alone) derives pending and persists it -- written_now on this tick, one recorded event
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep"); _grader_env(monkeypatch, env)
    n = len(_events(tmp_path)); s1 = gd.tick(args, *rest); ev1 = _events(tmp_path)[n:]
    rec = json.loads(rec_path.read_text()); raw = rec_path.read_bytes()
    assert s1["contrasts_sweep_hold"]["sweep"] == "held_by_flag_pending_decision" and s1["contrasts_sweep_pending_decision"] == {"path": str(rec_path), "record": rec, "written_now": True}
    assert "written now" in s1["contrasts_sweep"]["held_by_flag_pending_decision"] and rec["evidence"] == [] and rec["daemon_sha"] == gd.IMPORTED_CODE_SHA
    assert rec["unclassified"] == {"starts": 1, "ticks": 213, "held_events": 0, "first_utc": first_utc, "last_utc": last_utc}
    assert [e["event"] for e in ev1 if e["event"].startswith("sweep_pending_decision")] == ["sweep_pending_decision_recorded"] and not _mark_path(tmp_path).exists()
    # a dry run over the dir WITH the record: neither rewrites nor clears it, holds by its own status
    assert gd.tick(args_dry, *rest_dry)["contrasts_sweep_hold"]["sweep"] == "held_by_flag_dry_run" and rec_path.read_bytes() == raw
    assert gd.contrasts_sweep_hold_initial(args_dry)["skipped"] == "dry_run" and rec_path.read_bytes() == raw
    # an UNFLAGGED daemon is not bound by the record (it binds the flag's intent) and does not clear it: the sweep runs, the record stands
    args_nf, rest_nf, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    assert args_nf.contrasts_sweep is True and gd.contrasts_sweep_flag_pending(args_nf) is None and gd.contrasts_sweep_hold_initial(args_nf)["skipped"] == "no_flag"
    s_nf = gd.tick(args_nf, *rest_nf)
    assert s_nf["contrasts_sweep_hold"]["sweep"] == "ran" and "contrasts_sweep_pending_decision" not in s_nf and rec_path.read_bytes() == raw and _mark_path(tmp_path).exists()
    # the flagged daemon again: held by the record (the scan still sees the events here; the record is the authority either way)
    s2 = gd.tick(args, *rest)
    assert s2["contrasts_sweep_hold"]["sweep"] == "held_by_flag_pending_decision" and s2["contrasts_sweep_pending_decision"] == {"path": str(rec_path), "record": rec, "written_now": False}
    # the START-UP path clears it on the operator's .seeded record (seeded_before), reporting the path removed; the tick then follows the once-per-advance rule
    _seeded_path(tmp_path).write_text(json.dumps({"acknowledged_by": "operator"}) + "\n")
    n = len(_events(tmp_path)); dec = gd.contrasts_sweep_hold_initial(args); ev = _events(tmp_path)[n:]
    assert dec["seeded_before"] == {"acknowledged_by": "operator"} and dec["pending_decision_cleared"] == str(rec_path) and "pending_decision" not in dec and not rec_path.exists()
    assert [e["event"] for e in ev] == ["sweep_pending_decision_cleared"] and ev[0]["path"] == str(rec_path) and ev[0]["by"] == "contrasts_sweep_hold.seeded"
    s3 = gd.tick(args, *rest)
    assert s3["contrasts_sweep_hold"]["sweep"] == "skipped_once_per_advance" and "contrasts_sweep_pending_decision" not in s3
    assert "pending_decision_cleared" not in gd.contrasts_sweep_hold_initial(args)                   # nothing left to clear: said once
    # the START-UP path persists it too (the daemon's first flagged start over such a dir), on the unclassifiable path, naming the record
    st2 = tmp_path / "state2"; st2.mkdir(); (st2 / "grade_daemon.jsonl").write_text("".join(legacy)); rec2 = st2 / gd.CONTRASTS_SWEEP_PENDING_DECISION
    a = type("A", (), {"contrasts_sweep": False, "state_dir": str(st2), "dry_run": False})()
    d = gd.contrasts_sweep_hold_initial(a)
    assert d["skipped"] == "unclassifiable_legacy_history" and d["pending_decision"] == {"path": str(rec2), "record": json.loads(rec2.read_text()), "written_now": True}
    assert "held in this process" in d["operator_decision"] and str(rec2) in d["operator_decision"] and "neither file was written" in d["operator_decision"]
    assert f"1 daemon_started event(s), 213 tick(s) and 0 contrasts_sweep_held event(s) carrying contrasts_sweep_hold between {first_utc} and {last_utc}" in d["operator_decision"]
    assert not (st2 / gd.CONTRASTS_SWEEP_HOLD).exists() and not (st2 / gd.CONTRASTS_SWEEP_HOLD_SEEDED).exists()
    # the same start-up over the record with the scan EMPTIED (the log truncated to nothing: the extreme of eviction) still goes the operator path and seeds nothing
    (st2 / "grade_daemon.jsonl").write_text("")
    d2 = gd.contrasts_sweep_hold_initial(a)
    assert d2["skipped"] == "unclassifiable_legacy_history" and "unclassified" not in d2 and d2["pending_decision"]["written_now"] is False and d2["seeded_now"] is None
    assert "the bounded scan no longer sees them" in d2["operator_decision"] and "1 daemon_started event(s), 213 tick(s)" in d2["operator_decision"]
    assert not (st2 / gd.CONTRASTS_SWEEP_HOLD).exists() and not (st2 / gd.CONTRASTS_SWEEP_HOLD_SEEDED).exists() and gd.contrasts_sweep_flag_pending(a) is not None
    # a corrupt record still holds (fail toward the hold) and says so; no record / no state dir is None
    rec2.write_text("{not json")
    assert "unreadable" in gd.contrasts_sweep_pending_decision(str(st2)) and "cannot be read" in gd.contrasts_sweep_flag_pending(a)
    assert gd.contrasts_sweep_pending_decision(None) is None and gd.contrasts_sweep_pending_decision(str(tmp_path / "nowhere")) is None


# ---------------------------------------------------------------------------------------------- sweephold-fix-7: the publish is race-free
# fix-6 wrote the record through ONE fixed `.tmp` path and an unconditional os.replace, so two real flagged starts on one state dir raced: the
# loser died at start-up with FileNotFoundError (its tmp renamed away by the winner) BEFORE daemon_started, or both published and the dir ended
# with a double record and two "one-time" recorded events. Now: a unique tmp (tempfile.mkstemp in the state dir) and an EXCLUSIVE publish
# (os.link, EEXIST when a winner's record exists); the loser reads the winner's record and proceeds held on it -- one record, one recorded event
# in total, the loser reporting written_now: false with the winner's record and saying no event of its own -- and its tmp is removed either way.
# The .seeded-driven clearing already tolerated the other process removing first (FileNotFoundError -> None, no event).


def _pending_tmps(st):
    return [f for f in os.listdir(st) if "contrasts_sweep_pending_decision" in f and f.endswith(".tmp")]


def test_two_simultaneous_real_flagged_starts_on_one_state_dir_publish_one_record_say_one_event_and_both_hold(tmp_path, monkeypatch):
    """(a) the live shape: two `--once` daemons with the flag started together over one state dir (two processes). Both reach daemon_started (neither
    dies on its tmp), exactly one record and one recorded event, both ticks held pending on the same record, no tmp left. (b) a third flagged start
    over the standing record rewrites nothing and says no second event."""
    gd = _load_daemon_module()
    st = tmp_path / "state"; st.mkdir(); rec_path = _pending_path(tmp_path, gd)
    legacy, first_utc, last_utc = _legacy_window_lines()
    (st / "grade_daemon.jsonl").write_text("".join(legacy))
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, _rids = _root(tmp_path)
    n0 = len(_events(tmp_path))                                                                     # the fixture's own lines (its daemon_started among them) sit before the two starts
    p1 = TGD._start_daemon(root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep")
    p2 = TGD._start_daemon(root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep")
    o1, e1 = p1.communicate(timeout=600); o2, e2 = p2.communicate(timeout=600)
    assert p1.returncode == 0, e1 + o1
    assert p2.returncode == 0, e2 + o2
    ev = _events(tmp_path)[n0:]
    starts = [e for e in ev if e["event"] == "daemon_started"]
    assert len(starts) == 2 and {s["pid"] for s in starts} == {p1.pid, p2.pid} and all(s["dry_run"] is False for s in starts)
    recorded = [e for e in ev if e["event"] == "sweep_pending_decision_recorded"]
    assert len(recorded) == 1 and rec_path.exists() and not _hold_path(tmp_path).exists() and not _seeded_path(tmp_path).exists()
    rec = json.loads(rec_path.read_text()); raw = rec_path.read_bytes()
    assert recorded[0]["record"] == rec and rec["pid"] in (p1.pid, p2.pid) and rec["unclassified"] == {"starts": 1, "ticks": 213, "held_events": 0, "first_utc": first_utc, "last_utc": last_utc}
    pds = {s["pid"]: s["contrasts_sweep_hold_seeding"]["pending_decision"] for s in starts}
    assert all(pd["record"] == rec and pd["path"] == str(rec_path) for pd in pds.values())
    assert sorted(pd["written_now"] for pd in pds.values()) == [False, True]                      # exactly one published; the other read the winner's
    assert pds[rec["pid"]]["written_now"] is True                                                  # and the record names the one that did
    # the later start may already see the earlier one's stamped tick as real evidence (ambiguous_legacy_state) or not (unclassifiable_legacy_history):
    # both are the operator path, neither seeds, and the record binds either way
    assert all(s["contrasts_sweep_hold_seeding"]["skipped"] in ("unclassifiable_legacy_history", "ambiguous_legacy_state") and s["contrasts_sweep_hold_seeded"] is None for s in starts)
    ticks = [e for e in ev if e["event"] == "tick"]
    assert len(ticks) == 2 and all(t["contrasts_sweep_hold"]["sweep"] == "held_by_flag_pending_decision" for t in ticks)
    assert all(t["contrasts_sweep_pending_decision"] == {"path": str(rec_path), "record": rec, "written_now": False} for t in ticks)
    assert not [e for e in ev if e["event"] in ("sweep_noncurrent", "sweep_pending_decision_cleared")] and not _mark_path(tmp_path).exists()
    assert _pending_tmps(st) == []                                                                   # the winner's tmp unlinked after the publish, the loser's removed
    assert (rec_path.stat().st_mode & 0o777) == 0o644 and rec_path.stat().st_nlink == 1
    # (b) a third flagged start over the standing record: nothing rewritten, no second event, held on the same record
    n = n0 + len(ev)
    r3 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path), "--no-contrasts-sweep"); assert r3.returncode == 0, r3.stderr + r3.stdout
    ev3 = _events(tmp_path)[n:]
    assert _started(ev3)["contrasts_sweep_hold_seeding"]["pending_decision"] == {"path": str(rec_path), "record": rec, "written_now": False}
    assert _last_tick(ev3)["contrasts_sweep_hold"]["sweep"] == "held_by_flag_pending_decision" and rec_path.read_bytes() == raw
    assert not [e for e in ev3 if e["event"].startswith("sweep_pending_decision")] and _pending_tmps(st) == []
    assert len([e for e in _events(tmp_path) if e["event"] == "sweep_pending_decision_recorded"]) == 1


def test_the_race_forced_at_the_publish_point_yields_one_record_the_loser_reads_the_winner_and_a_vanished_winner_never_raises(tmp_path, monkeypatch):
    """The race made certain: two in-process start-ups (then two tick guards) that have BOTH derived the decision -- a barrier after the history
    scan -- publish at once. Exactly one wins the exclusive link; the loser returns the winner's record with written_now: false and says no event;
    no tmp is left. The clearing raced the same way says its event once. And the degenerate loser whose winner VANISHED by the read-back never
    raises: nothing persisted, nothing pending reported (or nothing pending at all when the .seeded record is what removed it)."""
    gd = _load_daemon_module()
    st = tmp_path / "state"; st.mkdir(); rec_path = _pending_path(tmp_path, gd)
    legacy, first_utc, last_utc = _legacy_window_lines()
    (st / "grade_daemon.jsonl").write_text("".join(legacy))
    a = type("A", (), {"contrasts_sweep": False, "state_dir": str(st), "dry_run": False})()
    barrier = threading.Barrier(2, timeout=60)
    real_history = gd.sweephold_history

    def history_then_meet(state_dir):                                                              # both derive, then race the publish
        h = real_history(state_dir); barrier.wait(); return h

    def race(fn):
        outs = [None, None]; errs = [None, None]

        def run(i):
            try:
                outs[i] = fn()
            except BaseException as exc:  # noqa: BLE001 -- the test reports it
                errs[i] = exc
        ts = [threading.Thread(target=run, args=(i,)) for i in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(120)
        assert errs == [None, None], errs
        return outs

    monkeypatch.setattr(gd, "sweephold_history", history_then_meet)
    # the START-UP path, raced
    outs = race(lambda: gd.contrasts_sweep_hold_initial(a))
    rec = json.loads(rec_path.read_text())
    assert sorted(o["pending_decision"]["written_now"] for o in outs) == [False, True] and rec["pid"] == os.getpid()
    assert all(o["pending_decision"]["record"] == rec and o["pending_decision"]["path"] == str(rec_path) for o in outs)
    assert all(o["skipped"] == "unclassifiable_legacy_history" and str(rec_path) in o["operator_decision"] and o["seeded_now"] is None for o in outs)
    assert [e["event"] for e in _events(tmp_path) if e["event"].startswith("sweep_pending_decision")] == ["sweep_pending_decision_recorded"]
    assert _pending_tmps(st) == [] and not (st / gd.CONTRASTS_SWEEP_HOLD).exists() and not (st / gd.CONTRASTS_SWEEP_HOLD_SEEDED).exists()
    # the TICK's guard, raced (a fresh dir state: the record gone, the log the legacy window again)
    rec_path.unlink(); (st / "grade_daemon.jsonl").write_text("".join(legacy)); barrier.reset()
    outs2 = race(lambda: gd._contrasts_sweep_pending(a))
    rec2 = json.loads(rec_path.read_text())
    assert sorted(o["written_now"] for o in outs2) == [False, True] and all(o["record"] == rec2 and o["path"] == str(rec_path) for o in outs2)
    assert all("held until the operator decides" in o["reason"] and str(rec_path) in o["reason"] for o in outs2)
    assert [e["event"] for e in _events(tmp_path) if e["event"].startswith("sweep_pending_decision")] == ["sweep_pending_decision_recorded"] and _pending_tmps(st) == []
    # the publish itself, raced directly, with the history both derived
    rec_path.unlink(); (st / "grade_daemon.jsonl").write_text("".join(legacy)); barrier.reset()
    hist = real_history(str(st))
    outs3 = race(lambda: (barrier.wait(), gd._sweephold_pending_write(str(st), hist))[1])
    assert sorted(w for _r, w in outs3) == [False, True] and all(r == json.loads(rec_path.read_text()) for r, _w in outs3) and _pending_tmps(st) == []
    # the CLEARING, raced: the operator's .seeded record honoured by two at once -- one removes and says the event, the other says nothing
    (st / gd.CONTRASTS_SWEEP_HOLD_SEEDED).write_text(json.dumps({"acknowledged_by": "operator"}) + "\n")
    n = len(_events(tmp_path))
    outs4 = race(lambda: (barrier.wait(), gd._sweephold_pending_clear(str(st)))[1])
    assert set(outs4) == {None, str(rec_path)} and not rec_path.exists()                             # one removed it, the other found it gone
    assert [e["event"] for e in _events(tmp_path)[n:]] == ["sweep_pending_decision_cleared"]
    assert gd._sweephold_pending_clear(str(st)) is None                                            # nothing left: None, no event
    (st / gd.CONTRASTS_SWEEP_HOLD_SEEDED).unlink()
    # the degenerate loser: the record it collided with has VANISHED by the read-back -- (None, False), no raise, no event, no tmp; with the .seeded
    # record standing (what removed it) the same, at once
    monkeypatch.setattr(gd, "sweephold_history", real_history)
    rec_path.write_text("{}\n")                                                                     # a winner's record for the link to collide with
    monkeypatch.setattr(gd, "contrasts_sweep_pending_decision", lambda state_dir: None)              # ...that the read-back no longer sees
    n = len(_events(tmp_path))
    assert gd._sweephold_pending_write(str(st), hist) == (None, False) and _pending_tmps(st) == [] and len(_events(tmp_path)) == n
    (st / gd.CONTRASTS_SWEEP_HOLD_SEEDED).write_text(json.dumps({"acknowledged_by": "operator"}) + "\n")
    assert gd._sweephold_pending_write(str(st), hist) == (None, False) and _pending_tmps(st) == [] and len(_events(tmp_path)) == n
    assert rec_path.read_text() == "{}\n"                                                            # the colliding record was never overwritten
    # and the tick's guard over that degenerate publish holds nothing and raises nothing (the next tick derives again)
    (st / gd.CONTRASTS_SWEEP_HOLD_SEEDED).unlink()
    assert gd._contrasts_sweep_pending(a) is None and len(_events(tmp_path)) == n
    # with the normal read-back restored, the standing (empty-object) record binds as any record does: written_now false, no event
    monkeypatch.undo()
    gd2 = _load_daemon_module()
    out = gd2.contrasts_sweep_hold_initial(a)
    assert out["pending_decision"] == {"path": str(rec_path), "record": {}, "written_now": False} and len(_events(tmp_path)) == n
