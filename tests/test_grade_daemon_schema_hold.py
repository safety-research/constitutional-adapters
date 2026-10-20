"""The grade daemon holds ALL work while the code and the live SQLite registry disagree on the schema, and the operator's hold covers the
Petri thread. The defect: (a) the migration of 2026-10-02 20:53-21:30Z landed under a running daemon -- the tick ran retries and
reconciliations, read the pending rows (`DB.reading`: read-only, no version check), mirrored stores and dispatched graders whose
`DB.open_or_create` claim raised SchemaDrift(have=21, want=22); `dispatch_failure_kind` read a PERMANENT failure and `quarantine_add` parked
the runs until a code sha / store generation / manual release, none of which a COMPLETED migration changes, so legitimate pending grading
stayed stuck until a sha change; (b) the Petri drive thread (its own cadence, started before the tick loop) ignored `dispatch.hold`, so an
API-consuming drive (and its registry hand-off) could start under an operator hold; (c) the documented migration window mitigated only with
the unit stopped. Now: (1) a schema preflight -- `db.require_shape` over a read-only connection -- at the top of every tick and in main()
before either loop holds everything: one `schema_hold` event per distinct gap (have, want, the drift lines), the tick line says
`held: true, hold: schema`, and it releases by itself (`schema_hold_released`) when the live schema is the code's; no exception escapes, the
daemon keeps ticking; (2) the Petri start is forbidden under `dispatch.hold` and under the schema hold alike (`petri_drive_held`, once per
tag per hold; settling and probing go on); (3) a grader whose claim refuses the schema is `schema_drift`, held transient
(`dispatch_schema_held`: pending, copy kept, never quarantined, never counted toward the retryable cap) and the tick stops on the gap;
(4) `--once`, `--dry-run`, `--no-contrasts`, `--no-contrasts-sweep` and the re-exec on code drift are untouched. Synthetic registries only
(test_grade_daemon._root, db.open_fresh under tmp), never the live one."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from sisteer.registry import db as DB
from sisteer.registry import emit
from sisteer.registry import migrations as RM
from sisteer.registry import pending as P

import test_grade_daemon as TGD                                                        # the daemon fixtures: the synthetic registry, argv/env, fake driver, events
from test_grade_daemon import _events, _load_daemon_module, _mirror_args, _root, _run_daemon
from test_grade_daemon_petri_drive import TAG, World                                   # the Petri drive world: fakes, repo, endpoint, proxy job

REPO = TGD.REPO
sys.path.insert(0, str(REPO / "scripts" / "registry"))
import grade_daemon as GD  # noqa: E402  -- the module the Petri world monkeypatches (the in-process tick tests load their own copy)


def _db(root: str) -> str:
    return DB.db_path(root)


def _downgrade(root: str) -> int:
    """The live incident's signature: the database's SHAPE is the code's (every migration applied) but the newest migration's version row is
    gone -- `db_version` reads code_version() - 1, `open_or_create` raises SchemaDrift(have=N-1, want=N, lines=[]). Returns N."""
    want = RM.code_version()
    conn = sqlite3.connect(_db(root))
    try:
        conn.execute("DELETE FROM schema_migration WHERE version = ?", (want,)); conn.commit()
        assert RM.db_version(conn) == want - 1
    finally:
        conn.close()
    return want


def _migrate(root: str) -> list:
    """The migration COMPLETES: `_downgrade` left the database's SHAPE the code's and removed only the newest version row (the live incident's
    signature), so completing it is recording that version row the way the runner records one (`migrations._apply_one`'s INSERT, its stamp
    format) -- after asserting the shape IS the code's (`db.schema_drift` empty) and the version is exactly one behind. Not
    `migrations.apply_migrations`: that re-runs the newest migration's BODY, which re-applied cleanly while 0022 (DROP VIEW IF EXISTS + CREATE
    VIEW) was the newest and cannot since 0023 (CREATE TABLE record_refusal_fact, ALTER TABLE ... ADD COLUMN: `already exists` on a shape that
    already has them; since 2026-10-04). Returns the runner's `applied` shape, ``[N]``."""
    want = RM.code_version()
    conn = DB.connect(_db(root))
    try:
        have, drift = RM.db_version(conn), DB.schema_drift(conn)
        assert have == want - 1 and drift == [], (have, want, drift[:3])
        slug = next(s for v, s, _p in RM.migration_files() if v == want)
        conn.execute("INSERT INTO schema_migration(version, applied_utc, description) VALUES (?, ?, ?)", (int(want), RM._now(), slug))
        conn.commit()
        assert RM.db_version(conn) == want
        return [want]
    finally:
        conn.close()


def _tick_args(gd, root: str, tmp_path: Path, *extra: str, dry_run: bool = False):
    """The daemon's parsed args + the tick's other inputs exactly as main() derives them, for an in-process tick; and the grader's env."""
    argv, env = TGD._daemon_argv_env(root, tmp_path, *extra, dry_run=dry_run)
    args = gd.build_parser().parse_args(argv[2:])
    args.families = None
    templates = dict(gd.DEFAULT_TEMPLATES); templates.update(gd._kv(args.driver, "--driver"))
    rest = (templates, gd._kv(args.extra, "--extra"), gd._kv(args.mirror, "--mirror"), P.parse_store_root_map(args.store_root_map) or None,
            gd._kv(args.eval_family, "--eval-family"))
    return args, rest, env


def _grader_env(monkeypatch, env: dict) -> None:
    """An in-process tick spawns its graders with os.environ: give them what the subprocess runs give (the fake driver's imports and registry)."""
    for k in ("SISTEER_SRC", "PYTHONPATH", "REGISTRY_ROOT"):
        monkeypatch.setenv(k, env[k])
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)


# ------------------------------------------------------------------------------------------------------- (1) the schema preflight, the tick
def test_a_registry_behind_the_code_holds_every_tick_with_one_schema_hold_event_until_the_migration_completes(tmp_path, monkeypatch):
    """Finding (a) inverted: version 21 under code 22 is a visible NO-WORK hold, not a quarantine. Tick 1 says `schema_hold` once (have, want,
    drift count + lines) and does nothing else -- no retry, reconcile, pending read, mirror, dispatch, contrast or prune event, nothing claimed;
    tick 2 is held on the same gap with no second event; the migration completing (for real) releases tick 3 by itself, which grades."""
    gd = _load_daemon_module()
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    both = sorted(rids[k] for k in ("job7@none", "job7@steer"))
    want = _downgrade(root)
    s1 = gd.tick(args, *rest)
    ev = _events(tmp_path)
    assert s1["held"] is True and s1["hold"] == "schema" and s1["schema"] == {"kind": "version", "have": want - 1, "want": want, "n_drift": 0}, s1
    assert s1["pending"] == 0 and s1["dispatched"] == 0 and s1["schema_held"] == 0 and "registry migrate" in s1["held_reason"]
    assert s1["problems"] == [f"schema_hold: {s1['held_reason']}"] and s1["schema_hold_since_utc"]
    assert [e["event"] for e in ev] == ["schema_hold", "tick"], [e["event"] for e in ev]
    sh = ev[0]
    assert sh["kind"] == "version" and sh["have"] == want - 1 and sh["want"] == want and sh["n_drift"] == 0 and sh["drift_lines"] == []
    assert "run `registry migrate`" in sh["message"] and sh["reason"] == s1["held_reason"] and sh["since_utc"] == s1["schema_hold_since_utc"]
    assert not (tmp_path / "rsync_calls.log").exists() and not (tmp_path / "driver_calls.jsonl").exists()      # no mirror, no grader
    # tick 2: the SAME gap -- held again, said no second time
    s2 = gd.tick(args, *rest)
    ev2 = _events(tmp_path)
    assert s2["held"] is True and s2["hold"] == "schema" and s2["schema_hold_since_utc"] == s1["schema_hold_since_utc"]
    assert [e["event"] for e in ev2] == ["schema_hold", "tick", "tick"]
    with DB.reading(_db(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (r,)).fetchone()[0] == "generated" for r in both)
    # the migration completes: tick 3 releases by itself (one event, naming what was held and since when) and grades both runs
    assert _migrate(root) == [want]
    s3 = gd.tick(args, *rest)
    ev3 = _events(tmp_path)[len(ev2):]
    assert s3["held"] is False and "hold" not in s3 and s3["pending"] == 2 and s3["dispatched"] == 2, s3
    rel = ev3[0]
    assert rel["event"] == "schema_hold_released" and rel["was_have"] == want - 1 and rel["was_kind"] == "version" and rel["have"] == rel["want"] == want
    assert rel["held_since_utc"] == s1["schema_hold_since_utc"]
    kinds3 = [e["event"] for e in ev3]
    assert kinds3.count("schema_hold_released") == 1 and "schema_hold" not in kinds3 and "dispatch" in kinds3 and "dispatched" in kinds3
    assert [e for e in ev3 if e["event"] == "dispatched"][-1]["rc"] == 0
    with DB.reading(_db(root)) as conn:
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (r,)).fetchone()[0] == "judged" for r in both)
    # a fourth tick: no transition event at all
    gd.tick(args, *rest)
    assert not [e for e in _events(tmp_path)[len(ev2) + len(ev3):] if e["event"] in ("schema_hold", "schema_hold_released")]


def test_a_shape_drift_at_the_codes_version_holds_too_and_names_the_differing_objects(tmp_path, monkeypatch):
    """The version guard alone cannot see an edited migration or a DDL change without one (db.schema_drift's own history): the SHAPE is compared
    too. An index the code declares, dropped: held, kind `shape`, have == want, the one drift line verbatim; a second differing object is a new
    gap and is said again; the shape restored (the code's own DDL -- what a landed migration would do) releases."""
    gd = _load_daemon_module()
    root, rids = _root(tmp_path)
    args, rest, _env = _tick_args(gd, root, tmp_path, dry_run=True)
    want = RM.code_version()
    indexes = sorted(n for (t, n) in DB.code_ddl_master() if t == "index")
    name, name2 = indexes[0], indexes[1]
    sql, sql2 = DB.code_ddl_master()[("index", name)], DB.code_ddl_master()[("index", name2)]
    conn = sqlite3.connect(_db(root)); conn.execute(f"DROP INDEX {name}"); conn.commit(); conn.close()
    s1 = gd.tick(args, *rest)
    assert s1["held"] is True and s1["hold"] == "schema" and s1["schema"] == {"kind": "shape", "have": want, "want": want, "n_drift": 1}, s1
    (sh,) = [e for e in _events(tmp_path) if e["event"] == "schema_hold"]
    assert sh["drift_lines"] == [f"index {name} is absent"] and sh["n_drift"] == 1 and "registry migrate --plan --sql" in sh["message"]
    assert f"index {name} is absent" in s1["held_reason"] and "migration is missing" in s1["held_reason"]
    gd.tick(args, *rest)
    assert len([e for e in _events(tmp_path) if e["event"] == "schema_hold"]) == 1                     # the same gap: silent
    conn = sqlite3.connect(_db(root)); conn.execute(f"DROP INDEX {name2}"); conn.commit(); conn.close()
    s3 = gd.tick(args, *rest)
    assert s3["schema"]["n_drift"] == 2 and len([e for e in _events(tmp_path) if e["event"] == "schema_hold"]) == 2   # a grown gap: said again
    assert not [e for e in _events(tmp_path) if e["event"] in ("dispatch", "mirror", "contrasts", "reclaim_own_session")]
    conn = sqlite3.connect(_db(root)); conn.execute(sql); conn.execute(sql2); conn.commit(); conn.close()
    s4 = gd.tick(args, *rest)
    assert s4["held"] is False and s4["dispatched"] == 2 and len([e for e in _events(tmp_path) if e["event"] == "schema_hold_released"]) == 1


def test_an_unreadable_registry_is_a_hold_not_an_exception_and_the_daemon_keeps_ticking(tmp_path, monkeypatch):
    """Fail closed: a registry the preflight cannot read (the file gone mid-swap, as a `registry rebuild` swap leaves it for a moment) is a hold of
    kind `check_failed` -- no exception escapes the tick, nothing is worked against the unverified schema -- and the file back releases."""
    gd = _load_daemon_module()
    root, _rids = _root(tmp_path)
    args, rest, _env = _tick_args(gd, root, tmp_path, dry_run=True)
    path = Path(_db(root)); away = path.with_name(path.name + ".away")
    conn = sqlite3.connect(str(path)); conn.execute("PRAGMA wal_checkpoint(TRUNCATE)"); conn.close()
    for suffix in ("-wal", "-shm"):
        Path(str(path) + suffix).unlink(missing_ok=True)
    os.rename(path, away)
    s1 = gd.tick(args, *rest)
    assert s1["held"] is True and s1["hold"] == "schema" and s1["schema"]["kind"] == "check_failed" and s1["schema"]["have"] is None
    assert s1["schema"]["want"] == RM.code_version() and "could not be read" in s1["held_reason"]
    (sh,) = [e for e in _events(tmp_path) if e["event"] == "schema_hold"]
    assert sh["kind"] == "check_failed" and "OperationalError" in sh["message"]
    os.rename(away, path)
    s2 = gd.tick(args, *rest)
    assert s2["held"] is False and s2["dispatched"] == 2
    (rel,) = [e for e in _events(tmp_path) if e["event"] == "schema_hold_released"]
    assert rel["was_kind"] == "check_failed" and rel["was_have"] is None


def test_once_dry_run_under_a_behind_registry_exits_0_with_the_hold_said_before_the_tick_and_no_dispatch(tmp_path):
    """The daemon's own CLI: `--once --dry-run --no-contrasts --no-contrasts-sweep` against a registry one version behind exits 0, main()'s
    preflight says `schema_hold` BEFORE the first tick (the tick re-checks and does not re-say), the tick line is held and dispatches nothing;
    the same run after the migration is a normal dry-run tick with its dispatch."""
    root, _rids = _root(tmp_path)
    want = _downgrade(root)
    r = _run_daemon(root, tmp_path, "--no-contrasts", "--no-contrasts-sweep", dry_run=True,
                    env_extra={"SISTEER_COORDINATION_CHANNEL": "#example-api-coordination"})
    assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)
    assert [e["event"] for e in ev] == ["daemon_started", "schema_hold", "tick"], [e["event"] for e in ev]
    assert ev[1]["have"] == want - 1 and ev[1]["want"] == want and '"event": "schema_hold"' in r.stdout
    assert ev[2]["held"] is True and ev[2]["hold"] == "schema" and ev[2]["dispatched"] == 0 and ev[2]["pending"] == 0
    assert "coordination (post to #example-api-coordination;" in r.stdout
    _migrate(root)
    r2 = _run_daemon(root, tmp_path, dry_run=True, env_extra={"SISTEER_COORDINATION_CHANNEL": ""})
    assert r2.returncode == 0, r2.stderr + r2.stdout
    assert "coordination (post to" not in r2.stdout                                  # no channel named: no ready-to-post line
    ev2 = _events(tmp_path)[len(ev):]
    tick2 = [e for e in ev2 if e["event"] == "tick"][-1]
    assert tick2["held"] is False and tick2["dispatched"] == 2 and "schema_hold" not in [e["event"] for e in ev2]
    assert [e for e in ev2 if e["event"] == "dispatch"][0]["dry_run"] is True
    assert not (tmp_path / "driver_calls.jsonl").exists()                                             # a dry run dispatches nothing


# ------------------------------------------------------------------------------------------ (3) SchemaDrift at a grader's claim: transient
# the fake driver with the migration window moving UNDER it: right before its claim, the newest version row goes (what the live database looked
# like to a grader whose daemon had preflighted a moment earlier), so the REAL `pending.claim` -> `db.open_or_create` raises the REAL SchemaDrift
FAKE_DRIVER_SCHEMA_BEHIND = TGD.FAKE_DRIVER.replace("wl = W.RegistryWorklist(args.registry_root,", """if os.environ.get("FAKE_DRIVER_SCHEMA_BEHIND_FILE") and os.path.exists(os.environ["FAKE_DRIVER_SCHEMA_BEHIND_FILE"]):   # g35: the database moves under this grader
    os.unlink(os.environ["FAKE_DRIVER_SCHEMA_BEHIND_FILE"])
    import sqlite3 as _sq
    _c = _sq.connect(os.path.join(args.registry_root, "registry.sqlite"))
    _c.execute("DELETE FROM schema_migration WHERE version = (SELECT MAX(version) FROM schema_migration)"); _c.commit(); _c.close()
wl = W.RegistryWorklist(args.registry_root,""", 1)
assert FAKE_DRIVER_SCHEMA_BEHIND != TGD.FAKE_DRIVER


def test_a_grader_whose_claim_refuses_the_schema_is_held_transient_never_quarantined_and_the_tick_stops_on_the_gap(tmp_path, monkeypatch):
    """Finding (a) at the claim: the preflight passes, the migration window moves under the grader, its claim raises the real SchemaDrift. The
    dispatch is `schema_drift` -- one `dispatch_schema_held` naming the gap (have, want), the runs still pending, the copy KEPT, no quarantine
    ledger, nothing counted toward the retryable cap -- and the gate re-read right after holds the rest of the tick (`held`, kind schema). The
    next tick's preflight holds; the completed migration lets the following tick dispatch the same runs and grade them."""
    monkeypatch.setattr(TGD, "FAKE_DRIVER", FAKE_DRIVER_SCHEMA_BEHIND)
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    flag = tmp_path / "schema_behind"; flag.write_text(""); monkeypatch.setenv("FAKE_DRIVER_SCHEMA_BEHIND_FILE", str(flag))
    root, rids = _root(tmp_path)
    both = sorted(rids[k] for k in ("job7@none", "job7@steer")); want = RM.code_version()
    mirror = tmp_path / "mirror"; dirs = [mirror / "stores" / "job7_none", mirror / "stores" / "job7_steer"]
    state = tmp_path / "state"
    r = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r.returncode == 0, r.stderr + r.stdout
    ev = _events(tmp_path)
    kinds = [e["event"] for e in ev]
    (d,) = [e for e in ev if e["event"] == "dispatched"]
    assert d["rc"] == 1 and d["outcome"] == "schema_drift" and d["terminal_exception"].startswith("sisteer.registry.db.SchemaDrift: "), d
    assert f"is at schema version {want - 1}, code expects {want}" in d["terminal_exception"]
    (h,) = [e for e in ev if e["event"] == "dispatch_schema_held"]
    assert sorted(h["run_ids"]) == both and h["n"] == 2 and h["have"] == want - 1 and h["want"] == want and h["shape_only"] is False and h["rc"] == 1
    assert h["terminal_exception"] == d["terminal_exception"] and os.path.isfile(h["dispatch_log"])
    assert not {"dispatch_failed", "dispatch_retryable", "problem", "quarantine_released", "contrasts", "sweep_noncurrent", "mirror_retired_incomplete"} & set(kinds), kinds
    assert not (state / "dispatch_quarantine.json").exists()
    assert not (state / "dispatch_retries.json").exists() or json.loads((state / "dispatch_retries.json").read_text())["runs"] == {}
    assert all(d_.is_dir() for d_ in dirs)                                                           # the copy is KEPT, as for the lock contention
    (sh,) = [e for e in ev if e["event"] == "schema_hold"]; assert sh["have"] == want - 1 and sh["want"] == want and sh["kind"] == "version"
    (held,) = [e for e in ev if e["event"] == "held"]; assert held["mid_tick"] is True and held["hold"] == "schema" and held["stopped_groups"] == []
    assert kinds.index("dispatch_schema_held") < kinds.index("schema_hold") < kinds.index("held") < kinds.index("tick")   # the preflight had passed; the gap is read after the refusal
    tick = ev[-1]
    assert tick["event"] == "tick" and tick["schema_held"] == 2 and tick["held"] is True and tick["hold"] == "schema" and tick["schema"]["have"] == want - 1
    assert tick["failed"] == 0 and tick["retryable"] == 0 and tick["quarantined"] == 0 and tick["dispatched"] == 0 and tick["pending"] == 2
    assert tick["mirror_retired_dirs"] == 0 and tick["mirror_prune_pending"] == 1 and tick["mirror_pruned_dirs"] == 0
    # tick 2: the preflight holds -- no second grader, nothing claimed (the claim WAS the refusal)
    r2 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r2.returncode == 0, r2.stderr + r2.stdout
    ev2 = _events(tmp_path)[len(ev):]
    assert [e["event"] for e in ev2] == ["daemon_started", "schema_hold", "tick"] and ev2[-1]["held"] is True and ev2[-1]["hold"] == "schema"
    assert len((tmp_path / "driver_calls.jsonl").read_text().splitlines()) == 1
    with DB.reading(_db(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (r_,)).fetchone()[0] == "generated" for r_ in both)
    # the migration completes: tick 3 dispatches the still-pending runs (nothing to release -- nothing was quarantined) and grades them
    assert _migrate(root) == [want]
    r3 = _run_daemon(root, tmp_path, *_mirror_args(tmp_path)); assert r3.returncode == 0, r3.stderr + r3.stdout
    ev3 = _events(tmp_path)[len(ev) + len(ev2):]
    tick3 = [e for e in ev3 if e["event"] == "tick"][-1]
    assert tick3["held"] is False and tick3["dispatched"] == 2 and tick3["schema_held"] == 0
    assert not [e for e in ev3 if e["event"] in ("schema_hold", "quarantine_released", "dispatch_failed")]
    assert len((tmp_path / "driver_calls.jsonl").read_text().splitlines()) == 2
    with DB.reading(_db(root)) as conn:
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (r_,)).fetchone()[0] == "judged" for r_ in both)


# ------------------------------------------------------------------------------------------------ (2) the Petri thread under either hold
@pytest.fixture
def world(tmp_path, monkeypatch):
    w = World(tmp_path, monkeypatch)
    monkeypatch.setattr(GD, "SCHEMA_GATE", GD.SchemaGate())                 # a fresh gate per test (the world resets the Petri said-sets the same way)
    yield w
    w.hold.unlink(missing_ok=True)                                           # release any held fake drive so nothing outlives the test


def test_the_operator_hold_forbids_a_petri_start_but_not_settling_and_is_said_again_after_it_lifts_and_returns(world):
    """Finding (b): the proxy is up and RUNNING, the endpoint has no drive -- under `dispatch.hold` the pass probes and settles but STARTS nothing:
    one `petri_drive_held` per tag naming the file and its reason, nothing claimed or recorded, no drive call; the same pass again says nothing
    more; the file gone, the drive starts; a hold back while the drive runs lets the pass settle it (completed) and starts nothing; a new
    endpoint under a returned hold is said again (once per hold episode)."""
    world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING")
    world.state.mkdir(parents=True, exist_ok=True)
    hold = world.state / GD.DISPATCH_HOLD
    hold.write_text("# D1255: static correction op in flight\nre-filing judgments under rung-keyed run ids\n")
    s = world.run_pass()
    assert s["held"] == 1 and s["started"] == 0 and s["running"] == 0, s
    (h,) = world.events("petri_drive_held")
    assert h["tag"] == TAG and h["hold"] == "operator" and h["file"] == str(hold) and h["reason"] == "re-filing judgments under rung-keyed run ids"
    assert h["identity"]["job_id"] == "11111" and h["wave_sha"] == world.sha
    assert world.events("petri_drive_started") == [] and world.drive_calls() == [] and not (world.state / f"petri_drive.{TAG}.json").exists()
    s = world.run_pass()
    assert s["held"] == 1 and len(world.events("petri_drive_held")) == 1                           # said once per hold
    hold.unlink()
    s = world.run_pass()
    assert s["started"] == 1 and s["held"] == 0 and len(world.events("petri_drive_started")) == 1
    att = world.record()["attempts"][0]
    hold.write_text("back\n")
    world.wait_exit(att)
    s = world.run_pass()                                                                             # settling under the hold is fine
    assert s["completed"] == 1 and s["started"] == 0 and world.record()["attempts"][0]["state"] == "completed", s
    assert len(world.events("petri_drive_held")) == 1 and world.wait_calls(1) and len(world.drive_calls()) == 1
    world.endpoint_up("22222", 1_700_000_100); world.job("22222", "RUNNING")                        # a replaced proxy: a new identity, still held
    s = world.run_pass()
    assert s["held"] == 1 and s["started"] == 0 and len(world.events("petri_drive_held")) == 2 and world.events("petri_drive_held")[-1]["identity"]["job_id"] == "22222"
    assert len(world.drive_calls()) == 1


def test_the_schema_hold_forbids_a_petri_start_the_same_way_and_lifts_with_the_migration(world):
    """The schema gap holds the Petri start exactly as the operator's file does: `petri_drive_held` of kind `schema` (no file), the `schema_hold`
    event said once across passes, no drive; the migration completing releases (`schema_hold_released`) and the next pass starts the drive. A
    pass with no registry named (the direct callers) checks no schema."""
    reg = world.tmp / "reg"; reg.mkdir(); DB.open_fresh(str(reg / DB.DB_NAME)).close()
    want = _downgrade(str(reg))
    world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING")
    s = world.run_pass(registry_root=str(reg))
    assert s["held"] == 1 and s["started"] == 0, s
    (h,) = world.events("petri_drive_held")
    assert h["hold"] == "schema" and h["file"] is None and "registry migrate" in h["reason"] and h["tag"] == TAG
    (sh,) = world.events("schema_hold"); assert sh["have"] == want - 1 and sh["want"] == want and sh["kind"] == "version"
    assert world.drive_calls() == [] and world.events("petri_drive_started") == []
    s = world.run_pass(registry_root=str(reg))
    assert s["held"] == 1 and len(world.events("petri_drive_held")) == 1 and len(world.events("schema_hold")) == 1
    assert _migrate(str(reg)) == [want]
    s = world.run_pass(registry_root=str(reg))
    assert s["started"] == 1 and s["held"] == 0 and len(world.events("schema_hold_released")) == 1 and len(world.events("petri_drive_started")) == 1
    assert len(world.wait_calls(1)) == 1


def test_main_preflights_the_schema_before_the_petri_thread_and_the_tick_loop_start(world, monkeypatch):
    """A RUNNING daemon over a registry that is not the code's. main() checks the schema before either loop: the `schema_hold` is on
    the log before the first Petri pass and the first tick, the gate is held when the tick runs, the Petri thread (its own cadence) starts no
    drive (`petri_drive_held`, kind schema, once), and the loop keeps ticking to its end."""
    monkeypatch.setattr(GD, "TICK_INTERVAL_FLOOR_S", 0.2); monkeypatch.setattr(GD, "PETRI_INTERVAL_FLOOR_S", 0.05)
    world.write_wave(); world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING")
    argv = world.main_argv("--max-ticks", "2", "--interval", "0.2")
    reg = argv[argv.index("--registry-root") + 1]
    want = _downgrade(reg)
    seen = []

    def fake_tick(args, *a, **k):
        seen.append(GD.SCHEMA_GATE.held)
        end = time.time() + 1.0                                               # give the Petri thread its passes
        while time.time() < end and not world.events("petri_drive_held"):
            time.sleep(0.05)
        return {}
    monkeypatch.setattr(GD, "tick", fake_tick)
    assert GD.main(argv) == 0
    kinds = [e["event"] for e in world.events()]
    assert kinds[:2] == ["daemon_started", "schema_hold"] and kinds.count("schema_hold") == 1 and "petri_drive_started" not in kinds, kinds
    assert kinds.index("schema_hold") < kinds.index("petri_drive_held") and kinds.count("petri_drive_held") == 1
    assert len(seen) == 2 and all(g is not None and g["have"] == want - 1 for g in seen)
    assert world.drive_calls() == []


def test_once_runs_the_inline_petri_pass_under_the_operator_hold_without_a_start(world, monkeypatch):
    """`--once` runs the single Petri pass inline after its tick: against a registry at the code's version with `dispatch.hold` present, the pass
    is held (operator) and starts nothing; the file gone, the same `--once` starts the drive."""
    world.write_wave(); world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING")
    world.state.mkdir(parents=True, exist_ok=True); hold = world.state / GD.DISPATCH_HOLD; hold.write_text("hold\n")
    monkeypatch.setattr(GD, "tick", lambda *a, **k: {})
    assert GD.main(world.main_argv("--once")) == 0
    (h,) = world.events("petri_drive_held"); assert h["hold"] == "operator" and h["file"] == str(hold)
    assert world.events("petri_drive_started") == [] and world.drive_calls() == [] and world.events("schema_hold") == []
    hold.unlink()
    assert GD.main(world.main_argv("--once")) == 0
    assert len(world.events("petri_drive_started")) == 1 and len(world.wait_calls(1)) == 1


# ----------------------------------------------------------------------------------------------------------------- (4) the untouched surfaces
def test_the_flags_and_the_code_drift_re_exec_are_untouched():
    """`--once`, `--dry-run`, `--no-contrasts`, `--no-contrasts-sweep` parse as before, and the re-exec on code drift (check_code_drift /
    reexec_self in main's loop) still precedes the tick -- the preflight is INSIDE tick() and once in main() before the loops, never between."""
    ns = GD.build_parser().parse_args(["--model-id", "m", "--once", "--dry-run", "--no-contrasts", "--no-contrasts-sweep"])
    assert ns.once is True and ns.dry_run is True and ns.contrasts is False and ns.contrasts_sweep is False
    src = Path(GD.__file__).read_text(encoding="utf-8")
    main_body = src[src.index("def main("):]
    assert main_body.index("SCHEMA_GATE.check(args.registry_root, args.state_dir)") < main_body.index("petri_thread.start()") < main_body.index("check_code_drift(args)") \
        < main_body.index("reexec_self()") < main_body.index("tick(args, templates, extra, mirror_ssh, store_root_map, extra_families)")
    assert "petri_start_hold" in src[src.index("def petri_drive_pass("):src.index("def petri_pass_args(")]


# ------------------------------------------------------------------------------------------------------------------ ClaimNotFolded
# the fake driver whose claim is REFUSED as not folded (pending.claim raises pending.ClaimNotFolded when the run's or the judge's row has
# not folded yet; both graders leave it uncaught, so the terminal line is the exception's qualified name); the flag is NOT consumed,
# so the refusal repeats every tick until the flag goes
FAKE_DRIVER_NOT_FOLDED = TGD.FAKE_DRIVER.replace('if os.environ.get("FAKE_DRIVER_ENV_RECORD"):', r"""if os.environ.get("FAKE_DRIVER_NOT_FOLDED_FILE") and os.path.exists(os.environ["FAKE_DRIVER_NOT_FOLDED_FILE"]):   # D1645: the claim refused, the run row not folded
    print("Traceback (most recent call last):\n  File \"pending.py\", line 171, in _require_parents", file=sys.stderr)
    print("sisteer.registry.pending.ClaimNotFolded: run r is not folded yet; retry after the harvester folds it (claim: no run row yet; nothing written)", file=sys.stderr); sys.exit(1)
if os.environ.get("FAKE_DRIVER_ENV_RECORD"):""", 1)
assert FAKE_DRIVER_NOT_FOLDED != TGD.FAKE_DRIVER


def test_a_claim_refused_as_not_folded_is_transient_held_never_quarantined_and_becomes_a_problem_when_it_persists(tmp_path, monkeypatch):
    """`pending.claim` refuses by name -- `pending.ClaimNotFolded` -- a claim whose run or judge row has not folded yet (nothing
    written; retry after the harvester folds it). A grader ending on that terminal line is the `not_folded` kind: the runs stay pending, the
    copy is kept, nothing is quarantined, nothing counts toward the retryable cap, one `dispatch_not_folded` names the group; a run still not
    folded on the next tick is a `problem` (kind dispatch_not_folded, uncapped, every tick it stands); the runs judged by another route leave
    the pending set and are cleared. The classifier keys on the terminal line's PREFIX, never on the text buried mid-tail."""
    import subprocess
    gd = _load_daemon_module()
    monkeypatch.setattr(TGD, "FAKE_DRIVER", FAKE_DRIVER_NOT_FOLDED)
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    flag = tmp_path / "not_folded"; flag.write_text(""); monkeypatch.setenv("FAKE_DRIVER_NOT_FOLDED_FILE", str(flag))
    root, rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    both = sorted(rids[k] for k in ("job7@none", "job7@steer"))
    mirror = tmp_path / "mirror"; dirs = [mirror / "stores" / "job7_none", mirror / "stores" / "job7_steer"]
    state = tmp_path / "state"
    line = "sisteer.registry.pending.ClaimNotFolded: run x is not folded yet; retry after the harvester folds it (claim: no run row yet; nothing written)"
    assert gd.dispatch_failure_kind(1, "Traceback (most recent call last):\n  File pending.py\n" + line + "\n") == "not_folded"
    assert gd.dispatch_failure_kind(1, line + "\n[judge] retrying...\nKeyError: 'x'\n") == "failed"                 # the same text buried mid-tail
    assert gd.dispatch_failure_kind(1, "a note mentioning sisteer.registry.pending.ClaimNotFolded: in passing\nKeyError: 'x'\n") == "failed"
    assert gd.dispatch_failure_kind(0, line) == "ok" and gd.NOT_FOLDED_TERMINAL == "sisteer.registry.pending.ClaimNotFolded:"
    # tick 1: the grader's claim is refused as not folded -> transient held: pending, copy kept, nothing quarantined, nothing counted, no problem yet
    s1 = gd.tick(args, *rest)
    ev = _events(tmp_path)
    (d,) = [e for e in ev if e["event"] == "dispatched"]
    assert d["rc"] == 1 and d["outcome"] == "not_folded" and d["terminal_exception"].startswith("sisteer.registry.pending.ClaimNotFolded:"), d
    (nf,) = [e for e in ev if e["event"] == "dispatch_not_folded"]
    assert sorted(nf["run_ids"]) == both and nf["family"] == "static" and nf["model_key"] == "synthmodel_4b_s1" and nf["message"] == d["terminal_exception"]
    assert nf["ticks"] == 1 and nf["since_utc"] and os.path.isfile(nf["dispatch_log"])
    assert s1["not_folded"] == 2 and s1["failed"] == 0 and s1["retryable"] == 0 and s1["quarantined"] == 0 and s1["dispatched"] == 0 and s1["problems"] == [], s1
    assert not {"dispatch_failed", "dispatch_retryable", "problem", "dispatch_schema_held", "mirror_retired_incomplete"} & {e["event"] for e in ev}, [e["event"] for e in ev]
    assert not (state / "dispatch_quarantine.json").exists() and all(d_.is_dir() for d_ in dirs)
    assert not (state / "dispatch_retries.json").exists() or json.loads((state / "dispatch_retries.json").read_text())["runs"] == {}
    # tick 2: the same outcome -> still pending and dispatched again (never quarantined), and the standing refusal is now a problem
    s2 = gd.tick(args, *rest)
    ev2 = _events(tmp_path)[len(ev):]
    assert [e for e in ev2 if e["event"] == "dispatched"][0]["outcome"] == "not_folded" and [e for e in ev2 if e["event"] == "dispatch_not_folded"][0]["ticks"] == 2
    (pr,) = [e for e in ev2 if e["event"] == "problem"]
    assert pr["kind"] == "dispatch_not_folded" and sorted(pr["run_ids"]) == both and pr["ticks"] == 2 and pr["since_utc"] == nf["since_utc"] and pr["message"] == d["terminal_exception"]
    assert s2["not_folded"] == 2 and s2["quarantined"] == 0 and s2["failed"] == 0
    assert s2["problems"] == [f"dispatch_not_folded: 2 run(s) not folded for 2 tick(s): {d['terminal_exception']}"], s2["problems"]
    with DB.reading(_db(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0                        # the claim WAS the refusal: nothing written
    # the runs judged OUT OF BAND (the rows folded; another session's grader graded them): they leave the pending set -> cleared, no problem
    flag.unlink()
    r = subprocess.run([sys.executable, str(tmp_path / "fake_driver.py"), "--worklist", "registry", "--registry-root", root, "--run-id", both[0], "--run-id", both[1],
                        "--session", "someone-else@TESTHOST", "--record", str(tmp_path / "oob_calls.jsonl")], capture_output=True, text=True, env=env, cwd=str(REPO), timeout=600)
    assert r.returncode == 0, r.stderr + r.stdout
    s3 = gd.tick(args, *rest)
    ev3 = _events(tmp_path)[len(ev) + len(ev2):]
    assert sorted((e["run_id"], e["why"]) for e in ev3 if e["event"] == "dispatch_not_folded_cleared") == [(r_, "left the pending set") for r_ in both]
    assert s3["not_folded"] == 0 and s3["pending"] == 0 and s3["problems"] == [] and not [e for e in ev3 if e["event"] in ("problem", "dispatch_not_folded", "dispatched")]
    with DB.reading(_db(root)) as conn:
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (r_,)).fetchone()[0] == "judged" for r_ in both)



# ----------------------------------------------------------------------------- the hold re-read before the claim and the spawn
def test_an_operator_hold_arriving_during_the_petri_preparation_stops_the_start_before_the_claim(world, monkeypatch):
    """The start decision passed with no hold; `dispatch.hold` lands while the preparation (the bank view's import / build / verify, the
    expected-set reads) runs. The hold is re-read right before the durable claim: `petri_drive_held` (operator, where = before the durable
    claim), no attempt written, no spawn; the endpoint stays startable and the next pass without the file starts it."""
    world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING")
    world.state.mkdir(parents=True, exist_ok=True); hold = world.state / GD.DISPATCH_HOLD
    real = GD._petri_prepare_spec

    def preparing(spec, **prep):
        hold.write_text("# landed mid-preparation\nhold during the bank view build\n")
        return real(spec, **prep)
    monkeypatch.setattr(GD, "_petri_prepare_spec", preparing)
    s = world.run_pass()
    assert s["held"] == 1 and s["started"] == 0, s
    (h,) = world.events("petri_drive_held")
    assert h["hold"] == "operator" and h["where"] == "before the durable claim" and h["reason"] == "hold during the bank view build" and h["file"] == str(hold)
    assert world.events("petri_drive_started") == [] and world.drive_calls() == [] and world.record()["attempts"] == []     # nothing claimed, nothing spawned
    assert len(world.events("petri_drive_prepared")) == 1                                                                 # the preparation itself ran
    s = world.run_pass()
    assert s["held"] == 1 and len(world.events("petri_drive_held")) == 1                                                  # still held: said once
    monkeypatch.setattr(GD, "_petri_prepare_spec", real); hold.unlink()
    s = world.run_pass()
    assert s["started"] == 1 and s["held"] == 0 and len(world.events("petri_drive_started")) == 1 and len(world.wait_calls(1)) == 1


def test_a_schema_drift_arriving_during_the_petri_preparation_stops_the_start_before_the_claim(world, monkeypatch):
    """The same re-read catches the schema hold: the migration window opens while the preparation runs -> `petri_drive_held` (schema, before the
    durable claim) and the `schema_hold`, nothing claimed or spawned; the migration completing releases and the next pass starts."""
    reg = world.tmp / "reg"; reg.mkdir(); DB.open_fresh(str(reg / DB.DB_NAME)).close()
    world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING")
    real = GD._petri_prepare_spec
    want = RM.code_version()

    def preparing(spec, **prep):
        _downgrade(str(reg))                                                       # the migration window opens while the bank view builds
        return real(spec, **prep)
    monkeypatch.setattr(GD, "_petri_prepare_spec", preparing)
    s = world.run_pass(registry_root=str(reg))
    assert s["held"] == 1 and s["started"] == 0, s
    (h,) = world.events("petri_drive_held")
    assert h["hold"] == "schema" and h["where"] == "before the durable claim" and "registry migrate" in h["reason"] and h["file"] is None
    (sh,) = world.events("schema_hold"); assert sh["have"] == want - 1 and sh["want"] == want
    assert world.drive_calls() == [] and world.record()["attempts"] == [] and world.events("petri_drive_started") == []
    monkeypatch.setattr(GD, "_petri_prepare_spec", real); assert _migrate(str(reg)) == [want]
    s = world.run_pass(registry_root=str(reg))
    assert s["started"] == 1 and s["held"] == 0 and len(world.events("schema_hold_released")) == 1 and len(world.wait_calls(1)) == 1


def test_a_hold_arriving_between_the_claim_and_the_spawn_withdraws_the_claim_and_spawns_nothing(world, monkeypatch):
    """The last re-read, right before the spawn: a hold landing after the durable claim was written withdraws that claim (the attempt removed,
    the record saved as before it: nothing for the reconciler to settle as failed, nothing toward the cap) and spawns nothing; the next pass
    without the hold starts attempt 1."""
    world.endpoint_up("11111", 1_700_000_000); world.job("11111", "RUNNING")
    world.state.mkdir(parents=True, exist_ok=True); hold = world.state / GD.DISPATCH_HOLD
    real_save = GD._petri_save

    def saving(rec_path, rec):
        real_save(rec_path, rec)
        if rec.get("attempts") and rec["attempts"][-1].get("state") == "claimed":   # the durable claim just landed: the operator's hold lands right after it
            hold.write_text("hold after the claim\n")
    monkeypatch.setattr(GD, "_petri_save", saving)
    s = world.run_pass()
    assert s["held"] == 1 and s["started"] == 0, s
    (h,) = world.events("petri_drive_held"); assert h["hold"] == "operator" and h["where"] == "before the spawn" and h["reason"] == "hold after the claim"
    assert world.record()["attempts"] == [] and world.drive_calls() == [] and world.events("petri_drive_started") == []     # the claim withdrawn, no spawn
    monkeypatch.setattr(GD, "_petri_save", real_save); hold.unlink()
    s = world.run_pass()
    assert s["started"] == 1 and s["held"] == 0 and world.record()["attempts"][0]["n"] == 1 and len(world.wait_calls(1)) == 1   # no numbering gap


# ------------------------------------------------------------------------------------------------------ ClaimPendingCorrection
# the fake driver whose claim is REFUSED by a retry-later class named in the environment, on the registry's message shape
# (ClaimPendingCorrection: an identity correction of the run is pending outside its own record files; refused before anything is
# written; retry after the next harvester cycle) -- the terminal line is the exception's qualified name, exactly as pending.claim's
# uncaught refusal leaves it; or, when the environment names the CLI shape, the CLI's `claim refused: <message>` line with no
# traceback (exit 1)
FAKE_DRIVER_RETRY_LATER = TGD.FAKE_DRIVER.replace(
    'if os.environ.get("FAKE_DRIVER_ENV_RECORD"):',
    'if os.environ.get("FAKE_DRIVER_RETRY_LATER_CLASS"):   # D1986 / D1990\n'
    '    _msg = "run r: an identity correction of this run is pending outside its own record files -- adapter revision; retry after the next harvester cycle (nothing written)"\n'
    '    if os.environ["FAKE_DRIVER_RETRY_LATER_CLASS"] == "CLI":\n'
    '        print("claim refused: " + _msg, file=sys.stderr); sys.exit(1)\n'
    '    print("Traceback (most recent call last):\\n  File \\"pending.py\\", line 171, in claim", file=sys.stderr)\n'
    '    print("sisteer.registry.pending." + os.environ["FAKE_DRIVER_RETRY_LATER_CLASS"] + ": " + _msg, file=sys.stderr); sys.exit(1)\n'
    'if os.environ.get("FAKE_DRIVER_ENV_RECORD"):', 1)
assert FAKE_DRIVER_RETRY_LATER != TGD.FAKE_DRIVER


def test_the_owners_claim_pending_correction_is_held_transient_like_not_folded_by_type_and_counted_by_class(tmp_path, monkeypatch):
    """`pending.claim` refuses with `ClaimPendingCorrection` (a ClaimNotFolded SUBCLASS) while an identity correction of the run is
    pending. It is the `not_folded` kind exactly like its parent -- held transient: pending, copy kept, no quarantine, no retryable
    count, no family hold -- routed by TYPE: the terminal line's class resolves on the imported pending module to a ClaimNotFolded
    subclass (the registry's REAL class, and a second synthetic subclass), or is the known retry-later class name (accepted by its
    prefix), or is the CLI's `claim refused: ` line; a name that resolves to ClaimHeld / ClaimPrecheckRefused or to nothing is
    `failed` / identity_changed as before. The ledger entry and event name the class and the tick counts the runs per class
    (`claim_retry_later`)."""
    gd = _load_daemon_module()
    owner = gd.RETRY_LATER_CLAIM_REFUSAL
    assert owner == "ClaimPendingCorrection" and gd.NOT_FOLDED_TERMINAL == "sisteer.registry.pending.ClaimNotFolded:"
    msg = "run x: an identity correction of this run is pending outside its own record files -- run_patch; retry after the next harvester cycle (nothing written)"
    tb = "Traceback (most recent call last):\n  File pending.py\n"
    # by type: the registry's REAL class is on this tree (a ClaimNotFolded subclass with a (run_id, what) signature; its message IS
    # the terminal line fed below); a second, synthetic subclass proves the route is by import on the class, never by name
    assert issubclass(getattr(P, owner), P.ClaimNotFolded) and getattr(P, owner) is not P.ClaimNotFolded
    assert str(getattr(P, owner)("x", "run_patch")) == msg
    synthetic = type("ClaimSyntheticLater", (P.ClaimNotFolded,), {})
    monkeypatch.setattr(P, "ClaimSyntheticLater", synthetic, raising=False)
    for name in ("ClaimNotFolded", owner, "ClaimSyntheticLater"):
        line = f"sisteer.registry.pending.{name}: {msg}"
        assert gd.retry_later_refusal_class(line) == name and gd.dispatch_failure_kind(1, tb + line + "\n") == "not_folded"
        assert gd.dispatch_failure_kind(1, line + "\n[judge] retrying...\nKeyError: 'x'\n") == "failed"          # the same text buried mid-tail
    assert gd.retry_later_refusal_class(f"claim refused: {msg}") == gd.CLAIM_REFUSED_CLI_CLASS and gd.dispatch_failure_kind(1, f"claim refused: {msg}\n") == "not_folded"
    assert gd.retry_later_refusal_class("sisteer.registry.pending.ClaimHeld: run x is held by s-B") is None                 # ClaimHeld is not retry-later
    assert gd.retry_later_refusal_class("sisteer.registry.pending.ClaimPrecheckRefused: claim refused: run x now designates judge j") is None
    assert gd.dispatch_failure_kind(1, "sisteer.registry.pending.ClaimPrecheckRefused: claim refused: run x now designates judge j\n") == "identity_changed"
    assert gd.retry_later_refusal_class("sisteer.registry.pending.ClaimSomethingElse: run x\n") is None and gd.dispatch_failure_kind(1, "sisteer.registry.pending.ClaimSomethingElse: run x\n") == "failed"
    assert gd.retry_later_refusal_class("KeyError: 'x'") is None and gd.retry_later_refusal_class("") is None
    monkeypatch.delattr(P, "ClaimSyntheticLater")
    # the daemon under a grader refused by the owner's class (its exact terminal line), two ticks, then the refusal lifts
    monkeypatch.setattr(TGD, "FAKE_DRIVER", FAKE_DRIVER_RETRY_LATER)
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log")); monkeypatch.setenv("FAKE_DRIVER_RETRY_LATER_CLASS", owner)
    root, rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    both = sorted(rids[k] for k in ("job7@none", "job7@steer"))
    s1 = gd.tick(args, *rest)
    ev = _events(tmp_path)
    (d,) = [e for e in ev if e["event"] == "dispatched"]
    assert d["rc"] == 1 and d["outcome"] == "not_folded" and d["terminal_exception"].startswith(f"sisteer.registry.pending.{owner}: run r: an identity correction")
    (nf,) = [e for e in ev if e["event"] == "dispatch_not_folded"]
    assert sorted(nf["run_ids"]) == both and nf["refusal_class"] == owner and nf["ticks"] == 1 and nf["message"] == d["terminal_exception"]
    assert s1["not_folded"] == 2 and s1["claim_retry_later"] == {owner: 2} and s1["failed"] == 0 and s1["retryable"] == 0 and s1["quarantined"] == 0 and s1["dispatched"] == 0
    assert s1["problems"] == [] and not {"dispatch_failed", "dispatch_retryable", "problem", "mirror_retired_incomplete"} & {e["event"] for e in ev}
    assert not (tmp_path / "state" / "dispatch_quarantine.json").exists()
    # tick 2: re-selected (still pending), refused again, counted again; two ticks standing -> a problem naming the class's message, never a quarantine
    s2 = gd.tick(args, *rest)
    ev2 = _events(tmp_path)[len(ev):]
    assert [e["outcome"] for e in ev2 if e["event"] == "dispatched"] == ["not_folded"] and s2["claim_retry_later"] == {owner: 2} and s2["quarantined"] == 0
    (pr,) = [e for e in ev2 if e["event"] == "problem"]
    assert pr["kind"] == "dispatch_not_folded" and sorted(pr["run_ids"]) == both and owner in pr["message"]
    assert not (tmp_path / "state" / "dispatch_quarantine.json").exists()
    # tick 3: the CLI shape of the same refusal (`claim refused: ...`, no traceback) is the same kind, counted under its own key
    monkeypatch.setenv("FAKE_DRIVER_RETRY_LATER_CLASS", "CLI")
    s3 = gd.tick(args, *rest)
    ev3 = _events(tmp_path)[len(ev) + len(ev2):]
    assert [e["outcome"] for e in ev3 if e["event"] == "dispatched"] == ["not_folded"] and s3["claim_retry_later"] == {gd.CLAIM_REFUSED_CLI_CLASS: 2} and s3["quarantined"] == 0
    assert [e["refusal_class"] for e in ev3 if e["event"] == "dispatch_not_folded"] == [gd.CLAIM_REFUSED_CLI_CLASS]
    # the refusal lifts (the correction landed): the grader grades, the standing refusal is cleared by the other outcome, nothing counted
    monkeypatch.delenv("FAKE_DRIVER_RETRY_LATER_CLASS")
    s4 = gd.tick(args, *rest)
    ev4 = _events(tmp_path)[len(ev) + len(ev2) + len(ev3):]
    assert s4["dispatched"] == 2 and s4["claim_retry_later"] == {} and s4["not_folded"] == 0 and s4["problems"] == []
    assert sorted(e["run_id"] for e in ev4 if e["event"] == "dispatch_not_folded_cleared") == both


# ----------------------------------------------------------------------------- the CLI-wrapped refusal is classed by its TYPE
# The CLI's real shape is `claim refused: {type(exc).__name__}: {exc}` (src/sisteer/registry/cli.py: the class tells a held claim
# from a precheck refusal). The earlier CLI branch returned CLAIM_REFUSED_CLI_CLASS for EVERY line under the prefix before any type
# test, so a CLI-wrapped ClaimPrecheckRefused (a ClaimHeld subclass) read retry-later / not_folded -- against
# retry_later_refusal_class's own docstring and the pin in tests/test_grade_daemon_identity_changed.py (`failed`, the CLI's own
# line is not a grader's terminal). Now the wrapped `<Name>:` resolves exactly as on the module-prefixed branch. A fake grader
# printing the CLI's real shape:
FAKE_DRIVER_CLI_NAMED = TGD.FAKE_DRIVER.replace(
    'if os.environ.get("FAKE_DRIVER_ENV_RECORD"):',
    'if os.environ.get("FAKE_DRIVER_CLI_NAMED_CLASS"):   # RETRYCLASSFIX: the CLI\'s real shape, `claim refused: <Name>: <message>`\n'
    '    _n = os.environ["FAKE_DRIVER_CLI_NAMED_CLASS"]\n'
    '    _m = "claim refused (precheck): run r: synthetic" if _n == "ClaimPrecheckRefused" else "run r: an identity correction of this run is pending outside its own record files -- run_patch; retry after the next harvester cycle (nothing written)"\n'
    '    print("claim refused: " + _n + ": " + _m, file=sys.stderr); sys.exit(1)\n'
    'if os.environ.get("FAKE_DRIVER_ENV_RECORD"):', 1)
assert FAKE_DRIVER_CLI_NAMED != TGD.FAKE_DRIVER


def test_a_cli_wrapped_refusal_is_classed_by_its_named_type_so_a_precheck_or_held_claim_is_never_retry_later(tmp_path, monkeypatch):
    """On the CLI's `claim refused: <Name>: ...` line the class decides, as on the module-prefixed branch -- ClaimPendingCorrection /
    ClaimNotFolded / any ClaimNotFolded subclass by type -> CLAIM_REFUSED_CLI_CLASS and `not_folded` (the tick's `claim_retry_later`
    key); ClaimPrecheckRefused / ClaimHeld / an unknown name -> None and `failed` (the pinned line, green with no change to that
    test). A CLI line with NO class name is not the CLI's shape; the message-only pins (the test above) keep its reading. The
    module-prefixed branch is unchanged. End to end: a grader printing the CLI's real shape with the retry-later class is held
    transient and counted under the CLI key; with ClaimPrecheckRefused it is a FAILED dispatch."""
    gd = _load_daemon_module()
    owner = gd.RETRY_LATER_CLAIM_REFUSAL; cli = gd.CLAIM_REFUSED_CLI_PREFIX
    msg_corr = str(getattr(P, owner)("x", "run_patch")); msg_nf = str(P.ClaimNotFolded("x", "run"))
    msg_pre = str(P.ClaimPrecheckRefused("x", "synthetic")); msg_held = str(P.ClaimHeld("x", "s-B", 12))
    assert msg_pre.startswith("claim refused (precheck): run x: ") and msg_held.startswith("run x is claimed by session") and not msg_corr.split(":", 1)[0].isidentifier()
    # the CLI's real shape: a retry-later class -> the CLI key, not_folded
    for name, msg in ((owner, msg_corr), ("ClaimNotFolded", msg_nf)):
        line = f"{cli}{name}: {msg}"
        assert gd.retry_later_refusal_class(line) == gd.CLAIM_REFUSED_CLI_CLASS and gd.dispatch_failure_kind(1, line + "\n") == "not_folded", line
    synthetic = type("ClaimSyntheticLater", (P.ClaimNotFolded,), {})
    monkeypatch.setattr(P, "ClaimSyntheticLater", synthetic, raising=False)
    assert gd.retry_later_refusal_class(f"{cli}ClaimSyntheticLater: run x") == gd.CLAIM_REFUSED_CLI_CLASS                    # by TYPE, through the import
    monkeypatch.delattr(P, "ClaimSyntheticLater")
    assert gd.retry_later_refusal_class(f"{cli}ClaimSyntheticLater: run x") is None                                            # ... and nothing once the class is gone
    # a class that is NOT retry-later, or no class at all under that name -> None -> `failed` by the daemon's other rules
    for name, msg in (("ClaimPrecheckRefused", msg_pre), ("ClaimHeld", msg_held), ("ClaimSomethingElse", "run x"),
                      ("StaleRequiredJudge", "claim refused: run x now designates judge j (instrument i), the worklist row carried k; re-select")):
        line = f"{cli}{name}: {msg}"
        assert gd.retry_later_refusal_class(line) is None and gd.dispatch_failure_kind(1, line + "\n") == "failed", line
    assert gd.dispatch_failure_kind(1, "claim refused: ClaimPrecheckRefused: claim refused (precheck): run x: synthetic\n") == "failed"   # the pinned line
    # no class name on the line (the message-only shape, pinned above): the same kind by the prefix
    assert gd.retry_later_refusal_class(f"{cli}{msg_corr}") == gd.CLAIM_REFUSED_CLI_CLASS and gd.dispatch_failure_kind(1, f"{cli}{msg_corr}\n") == "not_folded"
    assert gd.retry_later_refusal_class(f"{cli}{msg_nf}") == gd.CLAIM_REFUSED_CLI_CLASS and gd.retry_later_refusal_class(cli) == gd.CLAIM_REFUSED_CLI_CLASS
    # the module-prefixed branch, unchanged
    for name, msg in ((owner, msg_corr), ("ClaimNotFolded", msg_nf)):
        assert gd.retry_later_refusal_class(f"sisteer.registry.pending.{name}: {msg}") == name
    assert gd.retry_later_refusal_class(f"sisteer.registry.pending.ClaimPrecheckRefused: {msg_pre}") is None
    assert gd.retry_later_refusal_class(f"sisteer.registry.pending.ClaimHeld: {msg_held}") is None
    assert gd.retry_later_refusal_class("sisteer.registry.pending.ClaimSomethingElse: run x") is None and gd.retry_later_refusal_class("sisteer.registry.pending.: x") is None
    assert (gd._retry_later_class_name("ClaimNotFolded"), gd._retry_later_class_name(owner), gd._retry_later_class_name("ClaimHeld"), gd._retry_later_class_name(""),
            gd._retry_later_class_name("not an identifier")) == ("ClaimNotFolded", owner, None, None, None)
    # end to end, the CLI's real shape: the retry-later class -> held transient under the CLI key (as the message-only tick); ClaimPrecheckRefused -> a FAILED dispatch
    monkeypatch.setattr(TGD, "FAKE_DRIVER", FAKE_DRIVER_CLI_NAMED)
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log")); monkeypatch.setenv("FAKE_DRIVER_CLI_NAMED_CLASS", owner)
    root, rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    both = sorted(rids[k] for k in ("job7@none", "job7@steer"))
    s1 = gd.tick(args, *rest)
    ev = _events(tmp_path)
    (d,) = [e for e in ev if e["event"] == "dispatched"]
    assert d["rc"] == 1 and d["outcome"] == "not_folded" and d["terminal_exception"].startswith(f"claim refused: {owner}: run r: an identity correction")
    (nf,) = [e for e in ev if e["event"] == "dispatch_not_folded"]
    assert sorted(nf["run_ids"]) == both and nf["refusal_class"] == gd.CLAIM_REFUSED_CLI_CLASS
    assert s1["not_folded"] == 2 and s1["claim_retry_later"] == {gd.CLAIM_REFUSED_CLI_CLASS: 2} and s1["failed"] == 0 and s1["quarantined"] == 0 and s1["dispatched"] == 0
    assert not (tmp_path / "state" / "dispatch_quarantine.json").exists()
    monkeypatch.setenv("FAKE_DRIVER_CLI_NAMED_CLASS", "ClaimPrecheckRefused")
    s2 = gd.tick(args, *rest)
    ev2 = _events(tmp_path)[len(ev):]
    (d2,) = [e for e in ev2 if e["event"] == "dispatched"]
    assert d2["rc"] == 1 and d2["outcome"] == "failed" and d2["terminal_exception"].startswith("claim refused: ClaimPrecheckRefused: claim refused (precheck): run r")
    assert not [e for e in ev2 if e["event"] == "dispatch_not_folded"] and s2["not_folded"] == 0 and s2["claim_retry_later"] == {}
    assert [e["event"] for e in ev2 if e["event"] in ("dispatch_failed", "dispatch_not_folded", "dispatch_identity_changed")] == ["dispatch_failed"]
    (df,) = [e for e in ev2 if e["event"] == "dispatch_failed"]
    assert df["rc"] == 1 and df["n"] == 2 and sorted(df["run_ids"]) == both and s2["failed"] == 2 and s2["quarantined"] == 0       # `quarantined` counts the runs HELD at a tick's start
    assert sorted(gd.quarantine_load(str(tmp_path / "state"))) == both                                                               # a failed dispatch: the quarantine path, both runs
    s3 = gd.tick(args, *rest)                                                                                                        # next tick: held by the quarantine, nothing dispatched
    assert s3["quarantined"] == 2 and s3["dispatched"] == 0 and s3["not_folded"] == 0 and s3["claim_retry_later"] == {}


# ------------------------------------------------------------------- the wrapped claim refusal is matched on the message text
# Every multi-run grader's claim handler is widened to `except (RW.P.ClaimHeld, RW.P.ClaimNotFolded) as e:` and each re-raises its
# OWN wrapper of `--worklist registry: refusing — {e}`, so a ClaimNotFolded-class refusal reaches the daemon as a wrapped terminal
# line that drops the class name -- read `failed` (the quarantine path) under the by-type routing. The match is on the three
# MESSAGE texts of pending.py (cited by line), never on the class name or the wrapper alone. The wrapper lines below are derived
# from each driver's handler (driver and lines stated); the run / judge ids are synthetic. An identity-change message under ANY of
# these wrappers -- grade_full22_native's `[<target>] ` form included -- is `identity_changed`, under the group_tags rule for the
# tagged forms.
RUN_W = "synthcampaign:syntheval:aaaaaaaaaaaa:0000000000000001"; JUDGE_W = "1" * 64
PRIVATE_RUN_W = "sdfbase_full22_synthtarget_sampled_t1"             # a REAL-shaped private run (the live ones: sdfbase_full22_<tag>[_sampled_tN]): the static graders' REFUSING tag
TAG_W = GD.target_for("bon", PRIVATE_RUN_W, "mk")                     # the TARGET the native grader is dispatched with and tags its wrapper by (RETRYCLASSFIX3): the suffix, not the private run
assert TAG_W == "synthtarget" and TAG_W != PRIVATE_RUN_W
WRAPPED_MSG_NOT_FOLDED = f"run {RUN_W} is not folded yet; retry after the harvester folds it (claim: no run row yet; nothing written)"               # pending.py:62
WRAPPED_MSG_PENDING_CORRECTION = (f"run {RUN_W}: an identity correction of this run is pending outside its own record files -- run_patch; "
                                  "retry after the next harvester cycle (nothing written)")                                                       # pending.py:78-79
WRAPPED_MSG_NOT_PENDING = f"run {RUN_W}: not pending for judge {JUDGE_W} any more -- covered by judgment j-x; re-select (nothing written)"           # pending.py:97 (main)
WRAPPED_MSG_HELD = f"run {RUN_W} is claimed by session 's-B' (12 s ago)"                                                                           # pending.py:51 ClaimHeld
WRAPPED_MESSAGES = ((WRAPPED_MSG_NOT_FOLDED, "wrapped_not_folded"), (WRAPPED_MSG_PENDING_CORRECTION, "wrapped_pending_correction"), (WRAPPED_MSG_NOT_PENDING, "wrapped_not_pending"))
#: driver (origin/main handler lines) -> (the terminal line it emits for message {msg}, its exit code)
WRAPPER_FORMS = (
    ("scripts/petri_run.py:2166-2167 -- raise DriverError(f\"--worklist registry: refusing — {e}\"), uncaught: the traceback's terminal line as a script", "DriverError: --worklist registry: refusing — {msg}", 1),
    ("scripts/petri_run.py:2166-2167 -- the same DriverError, module-qualified when petri_run is imported", "petri_run.DriverError: --worklist registry: refusing — {msg}", 1),
    ("scripts/sdfbase/full22/misalign/misalign_judge.py:444-445 -- raise SystemExit(f\"--worklist registry: refusing — {e}\")", "--worklist registry: refusing — {msg}", 1),
    ("scripts/served_capability_judge.py:920-921 -- raise SystemExit(f\"--worklist registry: refusing — {e}\")", "--worklist registry: refusing — {msg}", 1),
    ("scripts/sdfbase/seedrep/grade_full22_native.py:1303-1304 (GRADER_DIFF.patch:834 the same) -- raise SystemExit(f\"[{args.target}] --worklist registry: refusing — {e}\")", "[" + TAG_W + "] --worklist registry: refusing — {msg}", 1),
    ("scripts/sdfbase/full22/sampled/grade_static_sampled.py:604-605 -- raise Refuse(f\"[{run}] --worklist registry: refusing — {e}\"): Refuse prints REFUSING: {msg}, exit 2; its tag is the PRIVATE RUN", "REFUSING: [" + PRIVATE_RUN_W + "] --worklist registry: refusing — {msg}", 2),
    ("scripts/sdfbase/full22/static_padding/grade_padded_static.py:838-839 -- raise Refuse(f\"[{ident['run']}] --worklist registry: refusing — {e}\") (Refuse = sg.Refuse); its tag is the padded PRIVATE RUN", "REFUSING: [" + PRIVATE_RUN_W + "] --worklist registry: refusing — {msg}", 2),
    ("scripts/grokmt_grade.py:269-270 -- raise SystemExit(f\"--worklist registry: refusing — {e}\")", "--worklist registry: refusing — {msg}", 1),
    ("scripts/grade_bon_live.py:801-802 -- raise SystemExit(f\"--worklist registry: refusing — {e}\")", "--worklist registry: refusing — {msg}", 1),
)
IDENTITY_CHANGE_MSG = "anchor identity changed between the binding and the filing: self_preservation: x: is_current: 1 -> 0 (why: superseded); refusing"


def test_every_graders_wrapped_claim_refusal_carrying_a_claim_not_folded_class_message_is_retry_later_by_phrase_never_by_the_wrapper(tmp_path):
    """Each of the eight graders' wrapper forms, for each of the three ClaimNotFolded-class messages -> retry class by message and
    `not_folded`; the tagged wrappers obey the tag rule; ClaimHeld's wrapped message keeps `failed` in every form; an identity-change
    message under every wrapper -- grade_full22_native's `[<target>] ` form included -- is `identity_changed` (a foreign tag:
    `failed`); a plain REFUSING, a wrapper with any other message, and the phrase outside a wrapper stay `failed`; the
    module-prefixed traceback line and the CLI line keep their by-type routing. The registry's texts on THIS tree match the pinned
    phrases (a drift guard), ClaimNotPending's when present."""
    gd = _load_daemon_module()
    assert gd.WRAPPED_NOT_FOLDED_PHRASE in str(P.ClaimNotFolded("x", "w")) and gd.WRAPPED_PENDING_CORRECTION_PHRASE in str(P.ClaimPendingCorrection("x", "w"))
    cnp = getattr(P, "ClaimNotPending", None)
    if cnp is not None:                                                                        # main's class: its text must carry the pinned phrase
        assert gd.WRAPPED_NOT_PENDING_RE.search(str(cnp("x", "j", "why"))) is not None
    assert gd.WRAPPED_NOT_PENDING_RE.search(WRAPPED_MSG_NOT_PENDING) and gd.WRAPPED_NOT_FOLDED_PHRASE in WRAPPED_MSG_NOT_FOLDED and gd.WRAPPED_PENDING_CORRECTION_PHRASE in WRAPPED_MSG_PENDING_CORRECTION
    tags = frozenset({PRIVATE_RUN_W, TAG_W, RUN_W}); foreign = frozenset({"other-run"})   # what the dispatch composes (RETRYCLASSFIX3): the private run, its derived target, the run ids
    for driver, fmt, rc in WRAPPER_FORMS:
        tagged = "[" + TAG_W + "]" in fmt or "[" + PRIVATE_RUN_W + "]" in fmt
        for msg, cls in WRAPPED_MESSAGES:
            line = fmt.format(msg=msg)
            assert gd.wrapped_retry_later_class(line) == cls and gd.retry_later_refusal_class(line) == cls, (driver, line)
            assert gd.dispatch_failure_kind(rc, "[judge] binding 2 run(s)...\n" + line + "\n") == "not_folded", (driver, line)
            assert gd.dispatch_failure_kind(rc, line + "\n", group_tags=tags) == "not_folded", (driver, line)                # the group's own tag (or no tag)
            if tagged:                                                                                                          # a FOREIGN tag is not this dispatch's refusal
                assert gd.retry_later_refusal_class(line, foreign) is None and gd.dispatch_failure_kind(rc, line + "\n", group_tags=foreign) == "failed", (driver, line)
            assert gd.dispatch_failure_kind(rc, line + "\n[judge] retrying...\nKeyError: 'x'\n") == "failed", (driver, line)     # the same text buried mid-tail
        held = fmt.format(msg=WRAPPED_MSG_HELD)                                                                                 # ClaimHeld: not retry-later-by-fold, as today
        assert gd.wrapped_retry_later_class(held) is None and gd.retry_later_refusal_class(held) is None and gd.dispatch_failure_kind(rc, held + "\n") == "failed", (driver, held)
        ident = fmt.format(msg=IDENTITY_CHANGE_MSG)                                                                             # the wrapper alone decides nothing: the message's kind
        assert gd.wrapped_retry_later_class(ident) is None and gd.identity_changed_terminal(ident, tags) is True, (driver, ident)
        assert gd.dispatch_failure_kind(rc, ident + "\n", group_tags=tags) == "identity_changed", (driver, ident)                  # The [<target>] form too
        if tagged:
            assert gd.identity_changed_terminal(ident, foreign) is False and gd.dispatch_failure_kind(rc, ident + "\n", group_tags=foreign) == "failed", (driver, ident)
        other = fmt.format(msg="store missing: /x/stores/job7_none/store.bin")
        assert gd.wrapped_retry_later_class(other) is None and gd.dispatch_failure_kind(rc, other + "\n", group_tags=tags) == "failed", (driver, other)
    # IDENTITY_CHANGED_TERMINAL_RE admits grade_full22_native's `[<target>] ` form beside every form it admitted before, and nothing else new
    ic = IDENTITY_CHANGE_MSG
    assert gd.IDENTITY_CHANGED_TERMINAL_RE.match("[" + TAG_W + "] --worklist registry: refusing — " + ic) is not None
    for admitted in ("DriverError: --worklist registry: refusing — ", "petri_run.DriverError: --worklist registry: refusing — ", "DriverError: ", "--worklist registry: refusing — ",
                     "REFUSING: [" + TAG_W + "] --worklist registry: refusing — ", "sisteer.registry.worklist.StaleRequiredJudge: ", "ClaimPrecheckRefused: "):
        assert gd.IDENTITY_CHANGED_TERMINAL_RE.match(admitted + ic) is not None, admitted
    for refused in ("REFUSING: --worklist registry: refusing — ", "[" + TAG_W + "] ", "REFUSING: [" + TAG_W + "] ", "[judge] note: ", "RuntimeError: "):
        assert gd.IDENTITY_CHANGED_TERMINAL_RE.match(refused + ic) is None, refused
    assert gd._REFUSING_TAG_RE.match("[" + TAG_W + "] --worklist registry: refusing — ").group(1) == TAG_W and gd._REFUSING_TAG_RE.match("REFUSING: [" + TAG_W + "] x").group(1) == TAG_W
    assert gd._REFUSING_TAG_RE.match("DriverError: --worklist registry: refusing — ") is None
    # negatives beyond the wrappers: a plain REFUSING, the phrase outside a wrapper, "refusing" in any other sentence
    assert gd.dispatch_failure_kind(2, "REFUSING: [" + TAG_W + "] store missing\n", group_tags=tags) == "failed"
    assert gd.dispatch_failure_kind(1, "[judge] note: " + WRAPPED_MSG_NOT_FOLDED + "\n") == "failed" and gd.wrapped_retry_later_class("[judge] note: " + WRAPPED_MSG_NOT_FOLDED) is None
    assert gd.dispatch_failure_kind(1, "RuntimeError: refusing to grade: " + WRAPPED_MSG_NOT_PENDING + "\n") == "failed"
    assert gd.dispatch_failure_kind(1, "DriverError: refusing — " + WRAPPED_MSG_NOT_PENDING + "\n") == "failed"                        # not the worklist's wrapper text
    # the by-type routes are untouched: the module-prefixed traceback line and the CLI's line
    assert gd.retry_later_refusal_class("sisteer.registry.pending.ClaimNotFolded: " + WRAPPED_MSG_NOT_FOLDED) == "ClaimNotFolded"
    assert gd.retry_later_refusal_class("sisteer.registry.pending.ClaimPendingCorrection: " + WRAPPED_MSG_PENDING_CORRECTION) == "ClaimPendingCorrection"
    assert gd.retry_later_refusal_class("claim refused: ClaimPendingCorrection: " + WRAPPED_MSG_PENDING_CORRECTION) == gd.CLAIM_REFUSED_CLI_CLASS
    assert gd.retry_later_refusal_class("claim refused: ClaimPrecheckRefused: claim refused (precheck): run x: synthetic") is None
    assert gd.dispatch_failure_kind(1, "claim refused: ClaimPrecheckRefused: claim refused (precheck): run x: synthetic\n") == "failed"         # the pin stands


# the fake grader that prints ONE wrapped terminal line (the exact text in the environment) and exits with the named code -- what a widened grader does
FAKE_DRIVER_WRAPPED = TGD.FAKE_DRIVER.replace(
    'if os.environ.get("FAKE_DRIVER_ENV_RECORD"):',
    'if os.environ.get("FAKE_DRIVER_WRAPPED_LINE"):   # RETRYCLASSFIX2: the grader\'s own wrapper of a claim refusal, f0b2ed0c8\n'
    '    print("[judge] binding 2 run(s) under the registry worklist...", file=sys.stderr)\n'
    '    print(os.environ["FAKE_DRIVER_WRAPPED_LINE"], file=sys.stderr); sys.exit(int(os.environ.get("FAKE_DRIVER_WRAPPED_RC", "1")))\n'
    'if os.environ.get("FAKE_DRIVER_ENV_RECORD"):', 1)
assert FAKE_DRIVER_WRAPPED != TGD.FAKE_DRIVER


def test_a_grader_ending_on_a_wrapped_not_pending_line_is_held_transient_and_counted_while_a_wrapped_genuine_failure_is_quarantined(tmp_path, monkeypatch):
    """RETRYCLASSFIX2, end to end: the fake grader prints misalign_judge's bare wrapper of ClaimNotPending's message and exits 1 -> the tick holds
    the group transient (`not_folded`), `dispatch_not_folded` names the class `wrapped_not_pending`, `claim_retry_later` counts both runs under it,
    nothing quarantined; the same wrapper carrying a genuine failure message -> `failed`, `dispatch_failed`, both runs quarantined."""
    gd = _load_daemon_module()
    monkeypatch.setattr(TGD, "FAKE_DRIVER", FAKE_DRIVER_WRAPPED)
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    monkeypatch.setenv("FAKE_DRIVER_WRAPPED_LINE", "--worklist registry: refusing — " + WRAPPED_MSG_NOT_PENDING); monkeypatch.setenv("FAKE_DRIVER_WRAPPED_RC", "1")
    root, rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    both = sorted(rids[k] for k in ("job7@none", "job7@steer"))
    s1 = gd.tick(args, *rest)
    ev = _events(tmp_path)
    (d,) = [e for e in ev if e["event"] == "dispatched"]
    assert d["rc"] == 1 and d["outcome"] == "not_folded" and d["terminal_exception"] == "--worklist registry: refusing — " + WRAPPED_MSG_NOT_PENDING
    (nf,) = [e for e in ev if e["event"] == "dispatch_not_folded"]
    assert sorted(nf["run_ids"]) == both and nf["refusal_class"] == "wrapped_not_pending" and nf["ticks"] == 1
    assert s1["not_folded"] == 2 and s1["claim_retry_later"] == {"wrapped_not_pending": 2} and s1["failed"] == 0 and s1["quarantined"] == 0 and s1["dispatched"] == 0
    assert not {"dispatch_failed", "dispatch_retryable", "problem"} & {e["event"] for e in ev} and not (tmp_path / "state" / "dispatch_quarantine.json").exists()
    # the REFUSING form of the same message (grade_static_sampled / grade_padded_static, exit 2) under the group's own tag: the same kind, its own class
    monkeypatch.setenv("FAKE_DRIVER_WRAPPED_LINE", "REFUSING: [" + both[0] + "] --worklist registry: refusing — " + WRAPPED_MSG_NOT_FOLDED); monkeypatch.setenv("FAKE_DRIVER_WRAPPED_RC", "2")   # tagged with one of the group's run ids
    s2 = gd.tick(args, *rest)
    ev2 = _events(tmp_path)[len(ev):]
    (d2,) = [e for e in ev2 if e["event"] == "dispatched"]
    assert d2["rc"] == 2 and d2["outcome"] == "not_folded" and s2["claim_retry_later"] == {"wrapped_not_folded": 2} and s2["quarantined"] == 0
    # a wrapped GENUINE failure: `failed`, the quarantine path (both runs), as for any failed dispatch
    monkeypatch.setenv("FAKE_DRIVER_WRAPPED_LINE", "--worklist registry: refusing — store missing: /x/stores/job7_none/store.bin"); monkeypatch.setenv("FAKE_DRIVER_WRAPPED_RC", "1")
    s3 = gd.tick(args, *rest)
    ev3 = _events(tmp_path)[len(ev) + len(ev2):]
    (d3,) = [e for e in ev3 if e["event"] == "dispatched"]
    assert d3["rc"] == 1 and d3["outcome"] == "failed" and s3["not_folded"] == 0 and s3["claim_retry_later"] == {} and s3["failed"] == 2 and s3["quarantined"] == 0
    (df,) = [e for e in ev3 if e["event"] == "dispatch_failed"]
    assert df["rc"] == 1 and sorted(df["run_ids"]) == both and sorted(gd.quarantine_load(str(tmp_path / "state"))) == both
    s4 = gd.tick(args, *rest)
    assert s4["quarantined"] == 2 and s4["dispatched"] == 0 and s4["not_folded"] == 0


# -------------------------------------------------------- the derived target joins the dispatch's group tags
# (1) The tagged-wrapper admission accepted a `[<tag>] ` wrapper only when the tag was in the dispatch's group_tags, and the tick
# composed those from the private run and the run ids alone -- but `target_for` passes the `sdfbase_full22_<TAG>` SUFFIX as
# `--target` to the native grader (grade_full22_native), which tags its wrapper `[{args.target}]`: neither the private run nor a
# run id, so every ClaimNotPending / ClaimNotFolded / identity-change refusal from a native-grader group read as FOREIGN -> failed
# -> the quarantine. Now the group's derived target joins its tags where the dispatch composes them. Driven through tick() with a
# REAL-shaped private run whose emitted `[<target>]` differs from the private run and from every run id. (2) The two-run sibling
# scenario end to end.
PRIVATE_RUN_T = "sdfbase_full22_synthqwen_u0_sampled_t1"          # a REAL-shaped private run (the live ones: sdfbase_full22_<tag>[_sampled_tN]); its target is the <tag>


def _root_full22(tmp_path):
    """TGD._root's world with the runs' stores under a REAL-shaped private run (`<root>/<private_run>/<attack>/<cell>`, the static graders' layout),
    so the dispatch group's private_run is PRIVATE_RUN_T and the target its grader is dispatched with is the `<tag>` suffix."""
    root = str(tmp_path / "reg")
    TGD.H.ensure_layout(root)
    TGD.R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), TGD._dims_with_other_model() + [TGD._none_baseline_include_ruling()])
    rids = {}
    specs = (("job7@none", "none", "none", TGD.synth.MODEL_ID), ("job7@steer", "lora@abcd", TGD.synth.OBJECT_SHA, TGD.synth.MODEL_ID), ("job8@other", "none", "none", TGD.OTHER_MODEL))
    for k, (key, okey, oid, model) in enumerate(specs):
        rid, recs = TGD.synth.run_records(key, 100 + 20 * k, object_key=okey, object_id=oid, through="generated", model_id=model,
                                          view_legacy="synthmodel_4b_s1", cluster_job_id="7" if key.startswith("job7") else "8")
        recs[0]["payload"]["store_uri"] = f"/cluster_a/user/exp/{PRIVATE_RUN_T}/attack_a/{key.replace('@', '_')}"
        TGD.R.write_records(os.path.join(root, "records", "c", f"00000{k + 1}-run.jsonl"), recs)
        rids[key] = rid
    rep = TGD.H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    return root, rids


def test_a_native_graders_target_tagged_wrapper_is_this_dispatchs_own_refusal_never_a_foreign_one(tmp_path, monkeypatch):
    """Through tick(): the group's private run is REAL-shaped and the native grader's `[<target>]` tag is the derived suffix --
    neither the private run nor a run id. (a) `[<target>] --worklist registry: refusing — <ClaimNotPending message>` ->
    not_folded, wrapped_not_pending, nothing quarantined, no failed; (b) the same wrapper with an identity-change message ->
    identity_changed through the existing path, nothing quarantined; the negative: a genuinely foreign tag (not the private run,
    not a run id, not the derived target) -> failed, as today."""
    gd = _load_daemon_module()
    root, rids = _root_full22(tmp_path)
    both = sorted(rids[k] for k in ("job7@none", "job7@steer"))
    target = gd.target_for("static", PRIVATE_RUN_T, "synthmodel_4b_s1")
    assert target == "synthqwen_u0" and target != PRIVATE_RUN_T and target not in both                       # the tag the native grader emits is neither the private run nor a run id
    monkeypatch.setattr(TGD, "FAKE_DRIVER", FAKE_DRIVER_WRAPPED)
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    msg_np = f"run {both[1]}: not pending for judge {TGD.synth.JUDGE_ID} any more -- covered by judgment j-peer; re-select (nothing written)"   # pending.py:102's text
    monkeypatch.setenv("FAKE_DRIVER_WRAPPED_LINE", f"[{target}] --worklist registry: refusing — {msg_np}"); monkeypatch.setenv("FAKE_DRIVER_WRAPPED_RC", "1")
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    s1 = gd.tick(args, *rest)
    ev = _events(tmp_path)
    (d,) = [e for e in ev if e["event"] == "dispatch"]
    assert d["private_run"] == PRIVATE_RUN_T and sorted(d["run_ids"]) == both                                   # the group as the live daemon composes it
    (dd,) = [e for e in ev if e["event"] == "dispatched"]
    assert dd["rc"] == 1 and dd["outcome"] == "not_folded", dd
    (nf,) = [e for e in ev if e["event"] == "dispatch_not_folded"]
    assert sorted(nf["run_ids"]) == both and nf["refusal_class"] == "wrapped_not_pending"
    assert s1["not_folded"] == 2 and s1["claim_retry_later"] == {"wrapped_not_pending": 2} and s1["failed"] == 0 and s1["quarantined"] == 0 and s1["dispatched"] == 0
    assert not [e for e in ev if e["event"] in ("dispatch_failed", "problem")] and not (tmp_path / "state" / "dispatch_quarantine.json").exists()
    # (b) an identity-change message under the same `[<target>]` wrapper: identity_changed through the existing path, nothing quarantined
    monkeypatch.setenv("FAKE_DRIVER_WRAPPED_LINE", f"[{target}] --worklist registry: refusing — {IDENTITY_CHANGE_MSG}")
    s2 = gd.tick(args, *rest)
    ev2 = _events(tmp_path)[len(ev):]
    (dd2,) = [e for e in ev2 if e["event"] == "dispatched"]
    assert dd2["outcome"] == "identity_changed" and [e["event"] for e in ev2 if e["event"] in ("dispatch_identity_changed", "dispatch_failed", "dispatch_not_folded")] == ["dispatch_identity_changed"]
    assert s2["identity_changed"] == 2 and s2["failed"] == 0 and s2["quarantined"] == 0 and not (tmp_path / "state" / "dispatch_quarantine.json").exists()
    # the classifier with EXACTLY the tags the dispatch composed: the derived target admits both forms; a foreign tag admits neither
    composed = frozenset([PRIVATE_RUN_T, target] + both)
    assert gd.dispatch_failure_kind(1, f"[{target}] --worklist registry: refusing — {msg_np}\n", group_tags=composed) == "not_folded"
    assert gd.dispatch_failure_kind(1, f"[{target}] --worklist registry: refusing — {IDENTITY_CHANGE_MSG}\n", group_tags=composed) == "identity_changed"
    assert gd.dispatch_failure_kind(1, f"[othertag_u0] --worklist registry: refusing — {msg_np}\n", group_tags=composed) == "failed"
    assert gd.dispatch_failure_kind(1, f"[othertag_u0] --worklist registry: refusing — {IDENTITY_CHANGE_MSG}\n", group_tags=composed) == "failed"
    # the negative through tick(): a genuinely FOREIGN tag -- not the private run, not a run id, not the derived target -- is not this dispatch's refusal: failed, quarantined, as today
    monkeypatch.setenv("FAKE_DRIVER_WRAPPED_LINE", f"[othertag_u0] --worklist registry: refusing — {msg_np}")
    s3 = gd.tick(args, *rest)
    ev3 = _events(tmp_path)[len(ev) + len(ev2):]
    (dd3,) = [e for e in ev3 if e["event"] == "dispatched"]
    assert dd3["outcome"] == "failed" and [e["event"] for e in ev3 if e["event"] == "dispatch_failed"] == ["dispatch_failed"] and s3["failed"] == 2 and s3["not_folded"] == 0
    assert sorted(gd.quarantine_load(str(tmp_path / "state"))) == both


# the fake grader of the sibling scenario: it reads its rows (the daemon's selection), then a PEER session's completion of run 2 lands in run 2's
# spool, then the REAL claim_all runs -- its scoped fold-first folds the peer's completion and (run 2, judge) is not pending any more:
# pending.claim raises ClaimNotPending, claim_all releases what it took, and the grader ends with the widened handler's wrapper in the named shape
FAKE_DRIVER_PEER = TGD.FAKE_DRIVER.replace('if os.environ.get("FAKE_DRIVER_NO_FOLD"):', """if os.environ.get("FAKE_DRIVER_PEER_COMPLETES_RUN"):   # RETRYCLASSFIX3 (D2236 F1): a peer completes run 2 between this grader's selection (rows) and its claim_all
    from sisteer.registry import emit as _E, identity as _I, harvest as _H
    _r2 = os.environ["FAKE_DRIVER_PEER_COMPLETES_RUN"]; (_row2,) = [r for r in rows if str(r["run_id"]) == _r2]
    _j = str(_row2["required_judge_id"]); _t0 = "2026-10-05T00:00:00.000Z"
    _w = _E.RecordWriter(_H.records_dir(args.registry_root), str(_row2["campaign"]), _r2, _E.producer_block("tests/peer_grader.py", session="s-peer"), disabled=False)
    _w.emit_judgment({"judgment_id": _I.judgment_id(_r2, _j, "s-peer", _t0), "run_id": _r2, "judge_id": _j, "status": "complete", "started_utc": _t0,
                      "finished_utc": "2026-10-05T00:00:01.000Z", "n_items_judged": 200, "api_pool": "lowprio", "store_uri": str(_row2.get("store_uri") or ""),
                      "claimed_by_session": "s-peer", "claimed_utc": _t0})
    try:
        wl.claim_all(rows)                                      # the REAL claim: the scoped fold-first folds the peer's completion -> ClaimNotPending for run 2
    except (W.P.ClaimHeld, W.P.ClaimNotFolded) as _e:            # f0b2ed0c8's widened handler; claim_all released what it took
        _m = f"--worklist registry: refusing — {_e}"
        if os.environ.get("FAKE_DRIVER_PEER_WRAPPER") == "driver":   # petri_run: the uncaught DriverError's traceback terminal line
            print("Traceback (most recent call last):\\n  File petri_run.py, line 2167, in main", file=sys.stderr); print("DriverError: " + _m, file=sys.stderr)
        else:                                                         # misalign_judge: the bare SystemExit line
            print(_m, file=sys.stderr)
        sys.exit(1)
    print("peer scenario: claim_all did not refuse", file=sys.stderr); sys.exit(3)
if os.environ.get("FAKE_DRIVER_NO_FOLD"):""", 1)
assert FAKE_DRIVER_PEER != TGD.FAKE_DRIVER


@pytest.mark.parametrize("shape", ["driver", "bare"])
def test_the_two_run_sibling_scenario_a_peer_completes_run_2_between_selection_and_claim_and_the_daemon_re_selects_only_run_1(tmp_path, monkeypatch, shape):
    """End to end: the daemon selects {run 1, run 2}; after the selection and before the grader's claim_all a peer session's
    complete, identity-current judgment of run 2 lands (the 0025 view drops run 2); the REAL claim_all refuses with ClaimNotPending and the grader ends
    with the wrapper carrying the ACTUAL message, in petri_run's DriverError shape and misalign_judge's bare shape. Tick 1: not_folded,
    wrapped_not_pending, nothing quarantined, no failed for either run. Tick 2: the dispatch set is exactly {run 1}, once; run 2 is never
    re-selected. Run 1 is graded and folded; tick 3 re-dispatches nothing, pending 0, problems []."""
    gd = _load_daemon_module()
    monkeypatch.setattr(TGD, "FAKE_DRIVER", FAKE_DRIVER_PEER)
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    root, rids = _root(tmp_path)
    r1, r2 = sorted(rids[k] for k in ("job7@none", "job7@steer"))                                           # claim_all claims r1 first, then r2 refuses: r1's claim is released
    monkeypatch.setenv("FAKE_DRIVER_PEER_COMPLETES_RUN", r2); monkeypatch.setenv("FAKE_DRIVER_PEER_WRAPPER", shape)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    # tick 1: the group {r1, r2} selected and dispatched; inside the grader the peer's completion lands and the REAL claim_all refuses r2
    s1 = gd.tick(args, *rest)
    ev = _events(tmp_path)
    (d,) = [e for e in ev if e["event"] == "dispatch"]
    assert sorted(d["run_ids"]) == [r1, r2]
    (dd,) = [e for e in ev if e["event"] == "dispatched"]
    prefix = "DriverError: --worklist registry: refusing — " if shape == "driver" else "--worklist registry: refusing — "
    assert dd["rc"] == 1 and dd["outcome"] == "not_folded", dd
    assert dd["terminal_exception"].startswith(prefix + f"run {r2}: not pending for judge {TGD.synth.JUDGE_ID} any more -- ") and dd["terminal_exception"].endswith("; re-select (nothing written)")
    (nf,) = [e for e in ev if e["event"] == "dispatch_not_folded"]
    assert sorted(nf["run_ids"]) == [r1, r2] and nf["refusal_class"] == "wrapped_not_pending" and nf["message"] == dd["terminal_exception"]
    assert s1["not_folded"] == 2 and s1["claim_retry_later"] == {"wrapped_not_pending": 2} and s1["failed"] == 0 and s1["quarantined"] == 0 and s1["dispatched"] == 0
    assert not [e for e in ev if e["event"] in ("dispatch_failed", "dispatch_retryable", "problem")] and not (tmp_path / "state" / "dispatch_quarantine.json").exists()
    with DB.reading(DB.db_path(root)) as conn:
        assert [tuple(r) for r in conn.execute("SELECT judge_id, status, claimed_by_session FROM judgment WHERE run_id = ?", (r2,))] == [(TGD.synth.JUDGE_ID, "complete", "s-peer")]   # folded by the claim's scoped fold-first
        assert conn.execute("SELECT COUNT(*) FROM judgment WHERE status = 'in_progress'").fetchone()[0] == 0                                   # claim_all released r1's claim
        assert [r["run_id"] for r in P.pending(conn, None, TGD.synth.CAMPAIGN, run_id=(r1, r2))] == [r1]                                      # r2 is not pending any more
    # tick 2: the dispatch set is EXACTLY {r1}, once; r2 is never re-selected; the grader grades r1 on its normal path and folds
    monkeypatch.delenv("FAKE_DRIVER_PEER_COMPLETES_RUN"); monkeypatch.delenv("FAKE_DRIVER_PEER_WRAPPER")
    s2 = gd.tick(args, *rest)
    ev2 = _events(tmp_path)[len(ev):]
    assert [sorted(e["run_ids"]) for e in ev2 if e["event"] == "dispatch"] == [[r1]] and s2["dispatched"] == 1 and s2["pending"] == 1 and s2["failed"] == 0 and s2["quarantined"] == 0
    (dd2,) = [e for e in ev2 if e["event"] == "dispatched"]
    assert dd2["rc"] == 0 and dd2["outcome"] == "ok"
    assert sorted(e["run_id"] for e in ev2 if e["event"] == "dispatch_not_folded_cleared") == [r1, r2]                                   # r2 left the pending set; r1's dispatch had another outcome
    # tick 3: r1 is judged; nothing re-dispatched, pending 0, problems []
    s3 = gd.tick(args, *rest)
    ev3 = _events(tmp_path)[len(ev) + len(ev2):]
    assert not [e for e in ev3 if e["event"] == "dispatch"] and s3["dispatched"] == 0 and s3["pending"] == 0 and s3["problems"] == [] and s3["not_folded"] == 0 and s3["failed"] == 0 and s3["quarantined"] == 0
    with DB.reading(DB.db_path(root)) as conn:
        assert P.pending(conn, None, TGD.synth.CAMPAIGN, run_id=(r1, r2)) == []
    assert not (tmp_path / "state" / "dispatch_quarantine.json").exists()
