"""The grade daemon's PER-FAMILY operator hold. The daemon had one hold only -- the global `dispatch.hold` -- and the
quarantine releases every held run on ANY daemon-sha change, so advancing the daemon's sha for one family's sake would re-dispatch the 7
quarantined s3r Petri runs at once (each a Petri re-dispatch costing 14 harvester cycles under the write lock until pending.claim/release's
fold is fixed). `<state-dir>/dispatch.hold.<family>` (first non-comment line = the reason) makes every tick SKIP that family's dispatch
groups -- not mirrored, not dispatched, not retried (rows_retry_pass), not counted as failed -- while the other families dispatch in the
same tick; one deduped `held_family` event per (family, reason) naming the file, the family and the runs held, re-said when the reason
changes; the tick summary carries `held_families` and `held_family_runs`; a quarantine release of a held family's run re-dispatches
nothing and re-quarantines nothing; removing the file releases at the next tick (`held_family_released`); a dry run reports it the same
way. It does NOT touch the Petri DRIVE thread (the global hold covers that). Synthetic registry (test_grade_daemon._family_registry: one
pending run per family for two synthetic models) -- never the live one."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from sisteer.registry import db as DB
from sisteer.registry import emit, synth
from sisteer.registry import pending as P

import test_grade_daemon as TGD                                   # the per-family synthetic registry, the fake driver and rsync, the events reader
from test_grade_daemon import _events, _family_registry, _load_daemon_module

REPO, DAEMON = TGD.REPO, TGD.DAEMON
MODEL, MODEL_KEY = "qwen35_4b_it", "qwen35_4b"                    # the paper model of _family_registry: view == tag == key


def _world(tmp_path: Path):
    """The registry with one pending run per family for MODEL (and the other model, never selected), the fake driver and rsync, the mirror."""
    root = str(tmp_path / "registry"); shared = str(tmp_path / "shared")
    rids = _family_registry(root, shared)
    fake = tmp_path / "fake_driver.py"; fake.write_text(TGD.FAKE_DRIVER)
    rsync = tmp_path / "rsync"; rsync.write_text(TGD.FAKE_RSYNC); rsync.chmod(0o755)
    return root, shared, {fam: rid for (mid, fam), (rid, _store) in rids.items() if mid == MODEL}


def _argv_env(tmp_path: Path, root: str, shared: str, *extra: str, families: str = "static,bon", dry_run: bool = False):
    record = tmp_path / "driver_calls.jsonl"
    template = f"{{python}} {tmp_path / 'fake_driver.py'} --worklist registry --registry-root {{registry_root}} {{run_id_flags}} {{store_map_flags}} --record {record}"
    env = dict(os.environ, SISTEER_SRC=str(REPO / "src"), PYTHONPATH=str(REPO / "src"), REGISTRY_ROOT=root, FAKE_RSYNC_LOG=str(tmp_path / "rsync_calls.log"))
    env.pop(emit.DISABLED_ENV, None)
    argv = [sys.executable, str(DAEMON), "--registry-root", root, "--model-id", MODEL, "--campaign", synth.CAMPAIGN, "--once", "--python", sys.executable,
            "--repo", str(REPO), "--state-dir", str(tmp_path / "state"), "--grader-session", "fake-driver", "--mirror-min-free-bytes", str(1024 * 1024),
            "--store-root-map", f"cluster_a={shared}={tmp_path / 'mirror'}", "--mirror", "cluster_a=cluster-a-login", "--rsync-bin", str(tmp_path / "rsync"),
            "--driver", f"static={template}", "--driver", f"bon={template}", "--driver", f"padding={template}",      # padding: the third family
            "--families", families, "--no-contrasts", "--no-contrasts-sweep", *extra]
    if dry_run:
        argv.append("--dry-run")
    return argv, env


def _run(tmp_path: Path, root: str, shared: str, *extra: str, **kw):
    import subprocess
    argv, env = _argv_env(tmp_path, root, shared, *extra, **kw)
    r = subprocess.run(argv, capture_output=True, text=True, env=env, timeout=600, cwd=str(REPO))
    assert r.returncode == 0, r.stderr + r.stdout
    return r


def _tick_args(gd, tmp_path: Path, root: str, shared: str, *extra: str, **kw):
    """The parsed args + the tick's other inputs exactly as main() derives them, for in-process ticks (the dedupe is per process)."""
    argv, _env = _argv_env(tmp_path, root, shared, *extra, **kw)
    args = gd.build_parser().parse_args(argv[2:])
    args.families = [f.strip() for f in args.families.split(",") if f.strip()] if args.families else None
    templates = dict(gd.DEFAULT_TEMPLATES); templates.update(gd._kv(args.driver, "--driver"))
    return args, (templates, gd._kv(args.extra, "--extra"), gd._kv(args.mirror, "--mirror"), P.parse_store_root_map(args.store_root_map) or None,
                  gd._kv(args.eval_family, "--eval-family"))


def _state(conn, rid: str) -> str:
    return conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0]


def _rsync_lines(tmp_path: Path) -> int:
    p = tmp_path / "rsync_calls.log"
    return len(p.read_text().splitlines()) if p.exists() else 0


def test_a_held_familys_groups_are_skipped_while_another_family_dispatches_in_the_same_tick(tmp_path):
    """`dispatch.hold.bon` present: the tick mirrors and dispatches the static group (two runs, graded) and SKIPS the bon group -- no mirror, no
    dispatch, no claim, not failed; one `held_family` event names the file, the family and the one run held; the summary says so."""
    root, shared, rids = _world(tmp_path)
    state = tmp_path / "state"; state.mkdir()
    hold = state / "dispatch.hold.bon"; hold.write_text("# D1531\nPetri/BoN re-dispatch costs 14 harvester cycles until the fold fix lands\n")
    _run(tmp_path, root, shared)
    ev = _events(tmp_path)
    (h,) = [e for e in ev if e["event"] == "held_family"]
    assert h["family"] == "bon" and h["reason"] == "Petri/BoN re-dispatch costs 14 harvester cycles until the fold fix lands" and h["file"] == str(hold) and h["n_runs"] == 1
    assert "Petri DRIVE" in h["note"] and "dispatch.hold" in h["note"]                     # the distinction from the drive thread's hold is documented
    dispatched = [e for e in ev if e["event"] == "dispatch"]
    assert [e["family"] for e in dispatched] == ["static"] and sorted(dispatched[0]["run_ids"]) == sorted([rids["static"], rids["static2"]])
    assert not [e for e in ev if e["event"] in ("dispatch_failed", "dispatch_refused", "held", "problem")], [e["event"] for e in ev]
    tick = [e for e in ev if e["event"] == "tick"][-1]
    assert tick["held_families"] == ["bon"] and tick["held_family_runs"] == 1 and tick["pending"] == 3 and tick["dispatched"] == 2 and tick["failed"] == 0
    assert tick["held"] is False                                                             # the global hold is a different thing
    assert _rsync_lines(tmp_path) == 2                                                       # the static group's two stores; nothing of bon mirrored
    with DB.reading(DB.db_path(root)) as conn:
        assert _state(conn, rids["static"]) == "judged" and _state(conn, rids["static2"]) == "judged" and _state(conn, rids["bon"]) == "generated"
        assert conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (rids["bon"],)).fetchone()[0] == 0
    # the file gone: the next tick dispatches bon (nothing to release: nothing was quarantined or claimed)
    hold.unlink()
    _run(tmp_path, root, shared)
    ev2 = _events(tmp_path)[len(ev):]
    tick2 = [e for e in ev2 if e["event"] == "tick"][-1]
    assert tick2["held_families"] == [] and tick2["held_family_runs"] == 0 and tick2["dispatched"] == 1 and [e["family"] for e in ev2 if e["event"] == "dispatch"] == ["bon"]
    with DB.reading(DB.db_path(root)) as conn:
        assert _state(conn, rids["bon"]) == "judged"


def test_the_hold_is_said_once_re_said_on_a_reason_change_released_once_and_a_dry_run_reports_it(tmp_path, monkeypatch):
    """In-process dry-run ticks (the dedupe is per process): tick 1 says `held_family` once; tick 2 (same reason) says nothing more; a new
    reason is said again; the file gone is one `held_family_released` and the dry run then reports every family's dispatch."""
    gd = _load_daemon_module()
    root, shared, rids = _world(tmp_path)
    args, rest = _tick_args(gd, tmp_path, root, shared, dry_run=True)
    state = tmp_path / "state"; state.mkdir()
    hold = state / "dispatch.hold.bon"; hold.write_text("first reason\n")
    s1 = gd.tick(args, *rest)
    assert s1["held_families"] == ["bon"] and s1["held_family_runs"] == 1 and s1["dispatched"] == 2 and s1["pending"] == 3, s1
    assert [e["family"] for e in _events(tmp_path) if e["event"] == "dispatch"] == ["static"]
    assert [(e["family"], e["reason"], e["n_runs"]) for e in _events(tmp_path) if e["event"] == "held_family"] == [("bon", "first reason", 1)]
    s2 = gd.tick(args, *rest)
    assert s2["held_families"] == ["bon"] and len([e for e in _events(tmp_path) if e["event"] == "held_family"]) == 1        # the same (family, reason): silent
    hold.write_text("# changed\nsecond reason\n")
    gd.tick(args, *rest)
    assert [e["reason"] for e in _events(tmp_path) if e["event"] == "held_family"] == ["first reason", "second reason"]
    hold.unlink()
    s4 = gd.tick(args, *rest)
    rel = [e for e in _events(tmp_path) if e["event"] == "held_family_released"]
    assert len(rel) == 1 and rel[0]["family"] == "bon" and rel[0]["was_reason"] == "second reason" and rel[0]["file"] == str(hold)
    assert s4["held_families"] == [] and s4["held_family_runs"] == 0 and s4["dispatched"] == 3
    gd.tick(args, *rest)
    assert len([e for e in _events(tmp_path) if e["event"] == "held_family_released"]) == 1
    assert not (tmp_path / "driver_calls.jsonl").exists()                                   # a dry run dispatched nothing throughout
    # two families held at once: both named, both counted; the bare global file is not a family
    (state / "dispatch.hold.static").write_text("s\n"); (state / "dispatch.hold.bon").write_text("b\n"); (state / "dispatch.hold").write_text("global\n")
    assert gd.family_holds(str(state)) == {"bon": "b", "static": "s"}
    s6 = gd.tick(args, *rest)
    assert s6["held_families"] == ["bon", "static"] and s6["held_family_runs"] == 3 and s6["held"] is True and s6["dispatched"] == 0


def test_a_quarantine_release_of_a_held_familys_runs_does_not_dispatch_them_and_does_not_re_quarantine(tmp_path, monkeypatch):
    """The reason the hold exists: a daemon-sha advance releases every quarantined run. Tick 1 (bon alone, the driver failing) quarantines the
    bon run; the hold goes on and the ledger's sha is stale (as after an advance); tick 2 releases the bon run (`quarantine_released`) and
    dispatches static -- bon is skipped, still pending, not re-quarantined, not failed; tick 3 without the file dispatches it."""
    root, shared, rids = _world(tmp_path)
    fail = tmp_path / "driver_fail"; fail.write_text(""); monkeypatch.setenv("FAKE_DRIVER_FAIL_FILE", str(fail))
    state = tmp_path / "state"; ledger = state / "dispatch_quarantine.json"
    _run(tmp_path, root, shared, families="bon")
    ev = _events(tmp_path)
    (df,) = [e for e in ev if e["event"] == "dispatch_failed"]; assert df["run_ids"] == [rids["bon"]]
    assert sorted(json.loads(ledger.read_text())["runs"]) == [rids["bon"]]
    fail.unlink()
    (state / "dispatch.hold.bon").write_text("hold bon across the sha advance\n")
    doc = json.loads(ledger.read_text()); doc["runs"][rids["bon"]]["daemon_sha"] = "0" * 40; ledger.write_text(json.dumps(doc))
    _run(tmp_path, root, shared)
    ev2 = _events(tmp_path)[len(ev):]
    (rel,) = [e for e in ev2 if e["event"] == "quarantine_released"]; assert rel["run_id"] == rids["bon"] and rel["reason"] == "daemon_sha"
    assert [e["family"] for e in ev2 if e["event"] == "dispatch"] == ["static"]
    assert not [e for e in ev2 if e["event"] in ("dispatch_failed", "problem")]
    tick2 = [e for e in ev2 if e["event"] == "tick"][-1]
    assert tick2["held_families"] == ["bon"] and tick2["held_family_runs"] == 1 and tick2["quarantined"] == 0 and tick2["failed"] == 0 and tick2["dispatched"] == 2
    assert json.loads(ledger.read_text())["runs"] == {}                                     # released, and NOT re-quarantined
    # the only grader call ever RECORDED is tick 2's static one -- the fake driver's fail path exits before its record line, so tick 1's
    # failing bon grader left none -- and the released bon run reached no grader
    calls = [json.loads(ln) for ln in (tmp_path / "driver_calls.jsonl").read_text().splitlines()]
    assert len(calls) == 1 and sorted(calls[0]["run_ids"]) == sorted([rids["static"], rids["static2"]]) and rids["bon"] not in json.dumps(calls)
    with DB.reading(DB.db_path(root)) as conn:
        assert _state(conn, rids["bon"]) == "generated" and _state(conn, rids["static"]) == "judged"
    (state / "dispatch.hold.bon").unlink()
    _run(tmp_path, root, shared)
    ev3 = _events(tmp_path)[len(ev) + len(ev2):]
    assert [e["family"] for e in ev3 if e["event"] == "dispatch"] == ["bon"] and [e for e in ev3 if e["event"] == "tick"][-1]["dispatched"] == 1
    with DB.reading(DB.db_path(root)) as conn:
        assert _state(conn, rids["bon"]) == "judged"


def test_a_held_familys_rows_items_are_not_retried_and_are_retried_once_the_file_goes(tmp_path):
    """rows_retry_pass under the family hold: a pending derived-rows item of the held family is named in a `rows_retry_held` event of kind
    `family` and left as it stands; without the file the next tick retries it (its command exits 0 -> `rows_cleared`)."""
    gd = _load_daemon_module()
    root, shared, rids = _world(tmp_path)
    state = tmp_path / "state"; state.mkdir()
    gd.rows_save(str(state), {rids["bon"]: {"run_id": rids["bon"], "family": "bon", "model_key": MODEL_KEY, "campaign": synth.CAMPAIGN, "store_root": "/x",
                                            "snapshot_dir": None, "cmd": ["true"], "dirs": [], "judgment_ids": ["j1"], "attempts": 1, "state": gd.ROWS_STATE_PENDING}})
    (state / "dispatch.hold.bon").write_text("hold\n")
    _run(tmp_path, root, shared, families="bon")
    ev = _events(tmp_path)
    (rh,) = [e for e in ev if e["event"] == "rows_retry_held"]
    assert rh["hold"] == "family" and rh["family"] == "bon" and rh["run_ids"] == [rids["bon"]] and rh["reason"] == "hold"
    tick = [e for e in ev if e["event"] == "tick"][-1]
    assert tick["rows_retried"] == 0 and tick["rows_pending"] == 1 and tick["held_families"] == ["bon"] and tick["dispatched"] == 0
    assert not [e for e in ev if e["event"] in ("rows_retry", "rows_cleared", "dispatch")]
    # a dry run reports the same hold over the item
    _run(tmp_path, root, shared, families="bon", dry_run=True)
    ev2 = _events(tmp_path)[len(ev):]
    (rh2,) = [e for e in ev2 if e["event"] == "rows_retry_held"]; assert rh2["hold"] == "family" and rh2["dry_run"] is True and rh2["run_ids"] == [rids["bon"]]
    assert not [e for e in ev2 if e["event"] == "rows_retry"]
    (state / "dispatch.hold.bon").unlink()
    _run(tmp_path, root, shared, families="bon")
    ev3 = _events(tmp_path)[len(ev) + len(ev2):]
    assert [e for e in ev3 if e["event"] == "rows_cleared"][0]["run_ids"] == [rids["bon"]] and [e for e in ev3 if e["event"] == "tick"][-1]["rows_retried"] == 1
    assert gd.rows_load(str(state)) == {}


# ----------------------------------------------------------------------------------------------------
# the fake rsync that writes a FAMILY hold file during the real copy (after the size probe): the operator's `dispatch.hold.<family>` lands while
# mirror_stores copies the group's stores -- after the pre-mirror re-read of the holds, before the grader
FAKE_RSYNC_WRITES_FAMILY_HOLD = TGD.FAKE_RSYNC.replace(
    'echo "$*" >> "${FAKE_RSYNC_LOG:?}"',
    'echo "$*" >> "${FAKE_RSYNC_LOG:?}"; if [ -n "${FAKE_RSYNC_WRITE_FAMILY_HOLD:-}" ]; then printf "# written while this group mirrored\\nheld mid-mirror\\n" > "$FAKE_RSYNC_WRITE_FAMILY_HOLD"; fi')
assert FAKE_RSYNC_WRITES_FAMILY_HOLD != TGD.FAKE_RSYNC


def _grader_env(monkeypatch, tmp_path: Path, root: str) -> None:
    """An in-process tick spawns its graders and rsyncs with os.environ: give them what the subprocess runs give."""
    _argv, env = _argv_env(tmp_path, root, str(tmp_path / "shared"))
    for k in ("SISTEER_SRC", "PYTHONPATH", "REGISTRY_ROOT", "FAKE_RSYNC_LOG"):
        monkeypatch.setenv(k, env[k])
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)


def test_a_family_hold_written_while_the_mirror_copies_stops_the_group_before_its_grader_and_releases_the_claim(tmp_path, monkeypatch):
    """The pre-mirror re-read passed, then `dispatch.hold.bon` lands while the bon group's stores copy -- the post-mirror /
    pre-grader gate re-reads the FAMILY holds too: the mirror is settled as not dispatched (the copy retired), the dispatch claim released, no
    grader spawned, the run still pending and unquarantined, one `held_family` (mid_tick, where) -- and the static group of the same tick
    dispatches and is graded."""
    root, shared, rids = _world(tmp_path)
    (tmp_path / "rsync").write_text(FAKE_RSYNC_WRITES_FAMILY_HOLD)
    state = tmp_path / "state"; state.mkdir()
    monkeypatch.setenv("FAKE_RSYNC_WRITE_FAMILY_HOLD", str(state / "dispatch.hold.bon"))
    _run(tmp_path, root, shared)
    ev = _events(tmp_path)
    assert [e["family"] for e in ev if e["event"] == "dispatch"] == ["bon", "static"]              # bon was announced and claimed (its re-read had passed) ...
    assert [e["family"] for e in ev if e["event"] == "dispatched"] == ["static"]                   # ... but only static's grader ran
    calls = [json.loads(ln) for ln in (tmp_path / "driver_calls.jsonl").read_text().splitlines()]
    assert len(calls) == 1 and rids["bon"] not in json.dumps(calls)
    (h,) = [e for e in ev if e["event"] == "held_family"]
    assert h["family"] == "bon" and h["mid_tick"] is True and h["where"] == "after this group's mirror and claim, before its grader"
    assert h["reason"] == "held mid-mirror" and h["n_runs"] == 1 and h["file"] == str(state / "dispatch.hold.bon")
    assert json.loads((state / "mirror_dispatch_claims.json").read_text())["claims"] == []         # the claim is released
    (ret,) = [e for e in ev if e["event"] == "mirror_retired_incomplete"]                          # settled as not dispatched: the copy retired
    assert ret["run_id"] == rids["bon"] and "before the grader started" in ret["note"] and not os.path.isdir(ret["dir"])
    assert not [e for e in ev if e["event"] in ("dispatch_failed", "problem", "rows_opened", "held", "held_family_released")], [e["event"] for e in ev]
    tick = [e for e in ev if e["event"] == "tick"][-1]
    assert tick["held_families"] == ["bon"] and tick["held_family_runs"] == 1 and tick["dispatched"] == 2 and tick["failed"] == 0 and tick["held"] is False
    assert not (state / "dispatch_quarantine.json").exists() and (state / "dispatch.hold.bon").exists()   # unquarantined; the operator's file is never consumed
    with DB.reading(DB.db_path(root)) as conn:
        assert _state(conn, rids["bon"]) == "generated" and conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (rids["bon"],)).fetchone()[0] == 0
        assert _state(conn, rids["static"]) == "judged" and _state(conn, rids["static2"]) == "judged"


def test_a_family_hold_written_during_an_earlier_rows_retry_stops_the_later_retries_of_that_family(tmp_path):
    """The rows path: two pending derived-rows items of the bon family in two groups; the first retry's command writes
    `dispatch.hold.bon`; the second group is skipped on the CURRENT holds (`rows_retry_held`, kind family, mid_tick), not on the start-of-tick
    snapshot. The static family is held throughout, so nothing dispatches."""
    gd = _load_daemon_module()
    root, shared, rids = _world(tmp_path)
    state = tmp_path / "state"; state.mkdir(); (state / "dispatch.hold.static").write_text("static held throughout\n")
    hold = state / "dispatch.hold.bon"; second = rids["bon"] + ":second-rows-item"

    def item(rid, store_root, cmd):
        return {"run_id": rid, "family": "bon", "model_key": MODEL_KEY, "campaign": synth.CAMPAIGN, "store_root": store_root, "snapshot_dir": None,
                "cmd": cmd, "dirs": [], "judgment_ids": ["j1"], "attempts": 1, "state": gd.ROWS_STATE_PENDING}
    gd.rows_save(str(state), {rids["bon"]: item(rids["bon"], "/a-first", ["bash", "-c", f"printf 'written by the first retry\\n' > {hold}"]),
                              second: item(second, "/b-second", ["true"])})
    _run(tmp_path, root, shared, families="static")
    ev = _events(tmp_path)
    (cl,) = [e for e in ev if e["event"] == "rows_cleared"]; assert cl["run_ids"] == [rids["bon"]] and hold.exists()
    (rh,) = [e for e in ev if e["event"] == "rows_retry_held"]
    assert rh["hold"] == "family" and rh["family"] == "bon" and rh["run_ids"] == [second] and rh["mid_tick"] is True and rh["reason"] == "written by the first retry"
    assert [e["run_ids"] for e in ev if e["event"] == "rows_retry"] == [[rids["bon"]]]            # one retry ran, the second never started
    tick = [e for e in ev if e["event"] == "tick"][-1]
    assert tick["rows_retried"] == 1 and tick["rows_cleared"] == 1 and tick["rows_pending"] == 1 and tick["dispatched"] == 0
    assert tick["held_families"] == ["static"] and tick["held_family_runs"] == 2                   # static's two pending runs, held at the top
    assert gd.rows_load(str(state))[second]["state"] == gd.ROWS_STATE_PENDING and rids["bon"] not in gd.rows_load(str(state))


def test_two_families_held_at_once_one_added_mid_tick_release_only_when_their_own_file_goes(tmp_path, monkeypatch):
    """Static held at tick start; `dispatch.hold.padding` lands while the bon group's stores copy; the per-group re-read
    reports the COMPLETE map, so static is NOT announced released while its file stands. In-process ticks (the dedupe is per process): removing
    static's file releases exactly static; padding stays held until its own file goes."""
    gd = _load_daemon_module()
    root, shared, rids = _world(tmp_path)
    (tmp_path / "rsync").write_text(FAKE_RSYNC_WRITES_FAMILY_HOLD)
    state = tmp_path / "state"; state.mkdir(); (state / "dispatch.hold.static").write_text("static held at the top\n")
    args, rest = _tick_args(gd, tmp_path, root, shared, families="static,bon,padding")
    _grader_env(monkeypatch, tmp_path, root)
    monkeypatch.setenv("FAKE_RSYNC_WRITE_FAMILY_HOLD", str(state / "dispatch.hold.padding"))
    s1 = gd.tick(args, *rest)
    ev = _events(tmp_path)
    assert [e["family"] for e in ev if e["event"] == "dispatch"] == ["bon"] and [e["family"] for e in ev if e["event"] == "dispatched"] == ["bon"]
    hf = [e for e in ev if e["event"] == "held_family"]
    assert [(e["family"], e["mid_tick"], e["where"]) for e in hf] == [("static", False, None), ("padding", True, "before this group's mirror and claim")]
    assert not [e for e in ev if e["event"] == "held_family_released"]                            # static's file still stands: no spurious release
    assert s1["held_families"] == ["padding", "static"] and s1["held_family_runs"] == 3 and s1["dispatched"] == 1, s1
    monkeypatch.delenv("FAKE_RSYNC_WRITE_FAMILY_HOLD")
    (state / "dispatch.hold.static").unlink()
    s2 = gd.tick(args, *rest)
    ev2 = _events(tmp_path)[len(ev):]
    assert [e["family"] for e in ev2 if e["event"] == "held_family_released"] == ["static"] and [e["family"] for e in ev2 if e["event"] == "dispatch"] == ["static"]
    assert s2["held_families"] == ["padding"] and s2["held_family_runs"] == 1 and s2["dispatched"] == 2, s2
    (state / "dispatch.hold.padding").unlink()
    s3 = gd.tick(args, *rest)
    ev3 = _events(tmp_path)[len(ev) + len(ev2):]
    assert [e["family"] for e in ev3 if e["event"] == "held_family_released"] == ["padding"] and [e["family"] for e in ev3 if e["event"] == "dispatch"] == ["padding"]
    assert s3["held_families"] == [] and s3["held_family_runs"] == 0 and s3["dispatched"] == 1, s3
    assert len([e for e in _events(tmp_path) if e["event"] == "held_family_released"]) == 2
    with DB.reading(DB.db_path(root)) as conn:
        assert all(_state(conn, rids[f]) == "judged" for f in ("bon", "static", "static2", "padding"))


# ----------------------------------------------------------------------------------------------------
def test_a_refused_run_is_said_once_with_its_config_path_raises_a_problem_each_tick_and_is_cleared_when_it_dispatches_or_leaves(tmp_path, monkeypatch):
    """The live daemon refused all 24 s1 static runs every tick for an hour (`config_missing`) and the only trace was the tick line's
    `refused: 24` counter. Now (in-process dry-run ticks; the dedupe is per process) the static and bon groups, whose configs are missing from
    an EMPTY configs root under the drivers' own templates: tick 1 says one `dispatch_refused` PER GROUP naming the config path, the run ids
    and the daemon sha, and raises a `problem` (kind dispatch_refused) + the tick line's `problems` entry; tick 2 says no refusal again
    (same runs, same reason) but the problem again; bon's config appearing dispatches bon -> `dispatch_refusal_cleared` (dispatched); the
    static runs judged by another route leave the pending set -> `dispatch_refusal_cleared` (left the pending set), nothing refused, no
    problem. The summary's `refused` counter is kept throughout."""
    import subprocess
    gd = _load_daemon_module()
    root, shared, rids = _world(tmp_path)
    cfgs = tmp_path / "configs"; cfgs.mkdir()                                                  # this tree lacks the view's configs
    args, rest = _tick_args(gd, tmp_path, root, shared, "--driver", f"static={gd.DEFAULT_TEMPLATES['static']}", "--driver", f"bon={gd.DEFAULT_TEMPLATES['bon']}",
                            "--grades-root", str(tmp_path / "grades"), "--configs-root", str(cfgs), families="static,bon", dry_run=True)
    static_cfg = cfgs / "sdfbase_full22_qwen35_4b_sampled_t0.yaml"; bon_cfg = cfgs / "sdfbase_full22_qwen35_4b.yaml"
    s1 = gd.tick(args, *rest)
    ev = _events(tmp_path)
    ref = {e["family"]: e for e in ev if e["event"] == "dispatch_refused"}
    assert set(ref) == {"static", "bon"} and s1["refused"] == 3 and s1["dispatched"] == 0, s1
    assert ref["static"]["reason"] == "config_missing" and ref["static"]["config_path"] == str(static_cfg) and str(static_cfg) in ref["static"]["detail"]
    assert sorted(ref["static"]["run_ids"]) == sorted(ref["static"]["new_run_ids"]) == sorted([rids["static"], rids["static2"]]) and ref["static"]["daemon_sha"] == gd.IMPORTED_CODE_SHA
    assert ref["bon"]["config_path"] == str(bon_cfg) and ref["bon"]["run_ids"] == ref["bon"]["new_run_ids"] == [rids["bon"]] and ref["bon"]["first_seen_utc"]
    (pr,) = [e for e in ev if e["event"] == "problem"]
    assert pr["kind"] == "dispatch_refused" and pr["n"] == 3 and pr["reasons"] == ["config_missing"] and sorted(pr["run_ids"]) == sorted(rids[f] for f in ("static", "static2", "bon"))
    assert s1["problems"] == [f"dispatch_refused: 3 run(s) refused: config_missing: {ref['bon']['detail']}"], s1["problems"]   # bon sorts first
    s2 = gd.tick(args, *rest)
    ev2 = _events(tmp_path)[len(ev):]
    assert not [e for e in ev2 if e["event"] == "dispatch_refused"] and [e["kind"] for e in ev2 if e["event"] == "problem"] == ["dispatch_refused"]
    assert s2["refused"] == 3 and s2["problems"] == s1["problems"]
    # bon's config lands: bon dispatches (dry run) and its refusal is cleared; static stays refused and is not re-said
    bon_cfg.write_text("private_run: sdfbase_full22_qwen35_4b\n")
    s3 = gd.tick(args, *rest)
    ev3 = _events(tmp_path)[len(ev) + len(ev2):]
    assert [e["family"] for e in ev3 if e["event"] == "dispatch"] == ["bon"] and not [e for e in ev3 if e["event"] == "dispatch_refused"]
    (cl,) = [e for e in ev3 if e["event"] == "dispatch_refusal_cleared"]
    assert cl["run_id"] == rids["bon"] and cl["family"] == "bon" and cl["why"] == "dispatched" and cl["was_reason"] == "config_missing" and cl["was_config_path"] == str(bon_cfg)
    assert s3["refused"] == 2 and s3["dispatched"] == 1 and s3["problems"] == [f"dispatch_refused: 2 run(s) refused: config_missing: {ref['static']['detail']}"]
    # the static runs are judged by ANOTHER route (a grader of another session, out of band): they leave the pending set -> cleared, nothing refused
    argv, env = _argv_env(tmp_path, root, shared)
    r = subprocess.run([sys.executable, str(tmp_path / "fake_driver.py"), "--worklist", "registry", "--registry-root", root, "--run-id", rids["static"], "--run-id", rids["static2"],
                        "--session", "someone-else@TESTHOST", "--record", str(tmp_path / "oob_calls.jsonl")], capture_output=True, text=True, env=env, cwd=str(REPO), timeout=600)
    assert r.returncode == 0, r.stderr + r.stdout
    s4 = gd.tick(args, *rest)
    ev4 = _events(tmp_path)[len(ev) + len(ev2) + len(ev3):]
    cls = sorted((e["run_id"], e["why"]) for e in ev4 if e["event"] == "dispatch_refusal_cleared")
    assert cls == sorted((rids[f], "left the pending set") for f in ("static", "static2")), cls
    assert s4["refused"] == 0 and s4["problems"] == [] and not [e for e in ev4 if e["event"] in ("dispatch_refused", "problem")]
    with DB.reading(DB.db_path(root)) as conn:
        assert _state(conn, rids["static"]) == "judged" and _state(conn, rids["bon"]) == "generated"
