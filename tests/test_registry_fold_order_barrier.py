"""Migration 0030 `fold_order_barrier`: the deployment barrier of the fold-order version (registry-fold-order-version, 2026-10-09), in two
layers one file lands together.

The stamp (`schema_migration` row 0, `reducer.FOLD_ORDER_VERSION`) fences the writers that READ it; the writers of the incident never did.
Layer 1, the VERSION: a pre-guard checkout -- commit 880be4e1c: code version 27, the pre-tie-break reducer order, `db.open_or_create`
checking only the migration version and the DDL shape -- accepts a stamped database whose version it knows and refolds a same-stamp pair
back to `judged`, leaving row 0 as it was. The one check such a writer already enforces is its own `db_version != code_version` ->
`SchemaDrift`, so 0030 advances the version and at 30 every such writer refuses by name. Layer 2, the FENCE: one shipped writer checks
nothing at all -- commit 5ea3fa411's `registry replica refold-apply --db <file> --keys <keys>` opens the file with a bare `db.connect` and
rewrites run rows under its own order -- so the database enforces the rule itself: every `db.connect` of this code registers the SQL
function `registry_fold_order()` (`db.FOLD_ORDER_FUNCTION`), and the two triggers 0030 creates on `run` (`schema_def.fold_order_fence_triggers`)
refuse a write whose connection's order is not the recorded one; a connection without the function cannot write a `run` row at all
(`no such function`, as SQLite compiles the write). These tests pin the file (its body is the render against the 0001..0029 shape) and the
plan's reading of it, the fence as a unit, and run the ACTUAL historical code (exported with `git archive`) as subprocesses: 880be4e1c's
open / tick / fold refused at 30 (rows byte-identical) and folding at its own version 27 as the control; 5ea3fa411's `refold-apply` refused
at the row (rows byte-identical) while this tree's refold succeeds on the same file, and rewriting the pair back to `judged` against a copy
with the fence dropped as the control. Synthetic registries under tmp only."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import textwrap
from pathlib import Path

import pytest

from sisteer.registry import cli as CLI
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import migrations as M
from sisteer.registry import records as R
from sisteer.registry import reducer as RD
from sisteer.registry import replica as RP
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from seedrep_history_fixtures import pinned_commit_or_skip
from test_registry_migrations import NTPL_TABLES, _sans_ntpl_drift, _sans_ntpl_render   # migration 0031's two rebuilds, set aside in the plan test

REPO = Path(__file__).resolve().parents[1]
BARRIER = "0030_fold_order_barrier"
#: The two fold-order fence triggers on `run` -- migration 0030's whole shape change.
FENCE = ("trg_run_fold_order_fence_ins", "trg_run_fold_order_fence_upd")
FENCE_DRIFT = [f"trigger {t} is absent" for t in FENCE]
#: The pre-guard writer: follow-ups-4 of 2026-10-05 -- migrations 0001..0027, `_load_events` ordered by `written_utc, record_sha256` (no
#: tie_rank), `db.open_or_create` checking the migration version and the DDL shape and nothing else. Reachable from main.
PREGUARD = "880be4e1c"
#: The direct-connect writer: the cluster replica's first commit (2026-10-04, before the tie-break) -- migrations 0001..0022, the old order,
#: and `replica refold-apply --db <file> --keys <keys>`, which opens the file with `DB.connect` and rewrites run rows with no check of the
#: version, the shape or the stamp (the review's round-3 MEDIUM). Reachable from main.
REFOLD = "5ea3fa411"
HISTORY_BRANCH = "main"


# ---- the file and the plan -----------------------------------------------------------------------------------------------------------
def test_0030_carries_the_two_fence_triggers_and_plans_as_pending_with_exactly_that_shape_change(tmp_path, capsys):
    version, slug, path = next(f for f in M.migration_files() if f[1] == "fold_order_barrier")
    assert (version, slug) == (30, "fold_order_barrier") and M.code_version() == 32, "0031 (non_template_positions: a rebuild of instrument and run) and 0032 (site_label_rename: the five tables whose CHECK names a site label, rebuilt) follow it on this tree; 0031's lines are set aside below"
    text = Path(path).read_text(encoding="utf-8")
    assert M.reserved_statement(text) is None
    assert "880be4e1c" in text and "5ea3fa411" in text and "WINDOW_EXPECT_MIGRATION=0030_fold_order_barrier" in text and "row 0" in text
    # the body is the render against the 0001..0029 shape: two triggers created, nothing else
    body = text[text.index("-- drift: "):]
    assert body.count("CREATE TRIGGER ") == 2 and body.count("DROP TRIGGER IF EXISTS ") == 2
    for absent in ("CREATE TABLE", "ALTER TABLE", "INSERT INTO", "CREATE VIEW", "DROP VIEW", "CREATE INDEX", "DROP TABLE", "__new"):
        assert absent not in body, absent
    for t in FENCE:
        assert f"-- drift: trigger {t} is absent\n" in body and f"DROP TRIGGER IF EXISTS {t};\nCREATE TRIGGER {t} BEFORE " in body
    assert [s.split("\n", 1)[0] for s in S.fold_order_fence_triggers()] == [f"CREATE TRIGGER {FENCE[0]} BEFORE INSERT ON run",
                                                                            f"CREATE TRIGGER {FENCE[1]} BEFORE UPDATE ON run"]
    # a database one version below: the frozen 0001..0029 applied -> the code's shape but for the two triggers
    db_path = str(tmp_path / "v29.sqlite")
    conn = DB.connect(db_path)
    for v, s, p in M.migration_files():
        if v <= 29:
            M._apply_one(conn, v, s, Path(p).read_text(encoding="utf-8"))
    assert M.db_version(conn) == 29 and _sans_ntpl_drift(DB.schema_drift(conn)) == FENCE_DRIFT          # 0031's two `lacks` lines aside
    plan = M.shape_plan(conn)
    # 0031 rebuilds `instrument` and `run` on this shape too, and a rebuild recreates its table's triggers itself -- so the plan lists the two
    # rebuilds and NO recreate of the fence triggers; the render's two rebuild blocks are set aside (`_sans_ntpl_render`), the rest is 0030's body
    assert _sans_ntpl_drift(plan["drift"]) == FENCE_DRIFT and [r["table"] for r in plan["rebuild"]] == list(NTPL_TABLES) and plan["add_columns"] == [] and plan["create"] == [] and plan["undeclared_tables"] == []
    assert plan["objects"] == {"index": {"recreate": [], "drop": []}, "trigger": {"recreate": [], "drop": []}, "view": {"recreate": [], "drop": []}}
    rendered = M.render_shape_migration(conn, plan)
    # 0031's run rebuild SUBSUMES the fence on this shape: a rebuild recreates its table's triggers inside its own block, so today's render
    # carries no DROP / CREATE of the two at the top level -- without 0031's part it is the two drift lines alone, and the committed body's
    # two CREATE TRIGGER statements are byte for byte the ones inside the run rebuild block
    assert _sans_ntpl_render(rendered).strip() == "\n".join(f"-- drift: {d}" for d in FENCE_DRIFT)
    for t in FENCE:
        create = body[body.index(f"CREATE TRIGGER {t} "):]
        create = create[:create.index("END;") + len("END;")]
        assert rendered.count(f"CREATE TRIGGER {t} ") == 1 and create in rendered, t
        assert rendered.index(f"CREATE TRIGGER {t} ") > rendered.index("-- rebuild run:") and f"DROP TRIGGER IF EXISTS {t};" not in rendered
    assert [(v, s) for v, s, _ in M.pending_migrations(conn)] == [(30, "fold_order_barrier"), (31, "non_template_positions"), (32, "site_label_rename")]
    conn.close()
    # the plan as the window scripts read it: the pending line names exactly the barrier, the shape differs by the two triggers, the one
    # recreate line names them, no rebuild / create / add / drop line (window.sh keys on the pending line and `rebuild table` lines alone)
    assert CLI.main(["migrate", "--plan", "--db", db_path]) == 1                        # pending: not yet current
    out = capsys.readouterr().out.splitlines()
    assert out[0] == f"schema version 29; code 32; pending: {BARRIER}, 0031_non_template_positions, 0032_site_label_rename"
    assert out[1].startswith("apply the pending migrations first") and out[2] == "shape differs from the code's schema:"
    assert [ln for ln in out[3:7] if ln.strip() in FENCE_DRIFT] == [f"  {d}" for d in FENCE_DRIFT]      # 0031's two `lacks` lines beside them
    assert [ln for ln in out if ln.startswith("recreate ")] == []                                       # the fence triggers ride 0031's run rebuild
    assert [ln for ln in out if ln.startswith("rebuild table ")] == [f"rebuild table {t}: 0 rows to copy (run-scale)" for t in NTPL_TABLES]
    assert not [ln for ln in out if ln.startswith(("create table ", "add column", "drop undeclared "))], out
    # the symmetric fence: THIS code's writers' open refuses the database at 29 by the same version check a pre-guard writer runs at 30
    with pytest.raises(DB.SchemaDrift) as ei:
        DB.open_or_create(db_path)
    assert ei.value.have == 29 and ei.value.want == M.code_version() and f"schema version 29, code expects {M.code_version()}" in str(ei.value) and not isinstance(ei.value, DB.FoldOrderDrift)
    assert DB.fold_order_version(DB.connect(db_path, readonly=True)) is None, "a refusal stamps nothing"
    # `registry migrate` applies the barrier (a schema writer: no stamp), then the first writer's open stamps row 0 on the same database
    assert CLI.main(["migrate", "--db", db_path]) == 0
    out = capsys.readouterr().out
    assert "applied [30, 31, 32]" in out and "schema version 32" in out                  # 0031 and 0032 follow the barrier on this tree
    with DB.reading(db_path) as ro:
        rows = [tuple(r) for r in ro.execute("SELECT version, description FROM schema_migration ORDER BY version")]
        assert rows[-3:] == [(30, "fold_order_barrier"), (31, "non_template_positions"), (32, "site_label_rename")] and [v for v, _ in rows] == list(range(1, 33)) and DB.schema_drift(ro) == []
        assert sorted(r[0] for r in ro.execute("SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'trg_run_fold_order_fence_%'")) == sorted(FENCE)
        assert DB.fold_order_version(ro) is None
    assert CLI.main(["migrate", "--plan", "--db", db_path]) == 0
    out = capsys.readouterr().out
    assert "pending: none" in out and "shape: matches the code's schema" in out
    DB.open_or_create(db_path).close()
    with DB.reading(db_path) as ro:
        assert DB.fold_order_version(ro) == RD.FOLD_ORDER_VERSION
        assert [v for (v,) in ro.execute("SELECT version FROM schema_migration ORDER BY version")] == list(range(0, 33))


# ---- the fixtures: historical code as it ran, a registry built by THIS code ----------------------------------------------------------
def _export(commit: str, dst: Path) -> Path:
    """``git archive <commit> src`` unpacked under ``dst``: the historical code as it ran, no patch."""
    dst.mkdir(parents=True, exist_ok=True)
    tar_path = dst / "src.tar"
    with open(tar_path, "wb") as fh:
        subprocess.run(["git", "-C", str(REPO), "archive", commit, "src"], check=True, stdout=fh)
    with tarfile.open(tar_path) as tf:
        tf.extractall(dst)
    src = dst / "src"
    assert (src / "sisteer" / "registry" / "db.py").is_file()
    return src


def _old_env(old_src: Path, tmp: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHONPATH") and k not in ("REGISTRY_ROOT", "REGISTRY_REPLICA_ROOT", "REGISTRY_LIVE_DB")}
    env["PYTHONPATH"] = str(old_src)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["REGISTRY_ROOT"] = str(tmp / "unused-root")          # the historical `main` reads it only to spell a default; nothing there is opened
    return env


def _old(old_src: Path, code: str, timeout_s: int = 300) -> subprocess.CompletedProcess:
    """Run ``code`` under the historical code -- a fresh interpreter with PYTHONPATH on the exported src -- the way its harvester tick ran."""
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=_old_env(old_src, old_src.parent), capture_output=True, text=True,
                          timeout=timeout_s, cwd=str(old_src.parent))


def _old_replica(old_src: Path, args, timeout_s: int = 300) -> subprocess.CompletedProcess:
    """``python -m sisteer.registry.replica <args>`` under the historical code: the shipped command, as an operator ran it."""
    return subprocess.run([sys.executable, "-m", "sisteer.registry.replica", *args], env=_old_env(old_src, old_src.parent), capture_output=True,
                          text=True, timeout=timeout_s, cwd=str(old_src.parent))


def _terminal_line(stderr: str) -> str:
    lines = [ln for ln in (stderr or "").strip().splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _rows(path: str):
    """Every row of the tables a refold touches, as bytes a comparison can read: the run row, its history, its state events, the deliveries."""
    with DB.reading(path) as ro:
        out = {}
        for table, order in (("run", "run_id"), ("row_history", "table_name, natural_key_json, written_utc, record_sha256"),
                             ("run_state_event", "run_id, state, event_utc, evidence_uri"), ("judgment", "judgment_id"),
                             ("record_delivery", "file_uri"), ("schema_migration", "version")):
            out[table] = [tuple(r) for r in ro.execute(f"SELECT * FROM {table} ORDER BY {order}").fetchall()]
    return out


#: a substrate both this code and the exported pre-guard writer admit: that writer predates the neutral site labels, so the records the
#: control below rebuilds with it carry a value both vocabularies hold
BOTH_CODES_SUBSTRATE = "runpod"


def _pair(rid: str, stamp: str):
    prod = synth.producer(substrate=BOTH_CODES_SUBSTRATE)          # the final bytes: the hash order below is the one every writer reads
    judged = synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": stamp, "evidence_kind": "completion_artifact"}, stamp,
                       prod=prod)
    for n in range(512):
        sup = synth.rec("run_state", {"run_id": rid, "state": "superseded", "event_utc": stamp, "evidence_kind": "ruling",
                                      "note": f"superseded by a later run of the same key ({n})"}, stamp, prod=prod)
        if R.record_sha(sup) < R.record_sha(judged):
            return judged, sup
    raise AssertionError("no adversarial note in 512 tries")


def _either_code(recs):
    """The records with every substrate (the producer's and a payload's) set to :data:`BOTH_CODES_SUBSTRATE`, each record id derived
    again from its payload, so the exported pre-guard writer folds exactly what this code folds."""
    out = []
    for r in recs:
        payload = dict(r["payload"])
        if payload.get("substrate") is not None:
            payload["substrate"] = BOTH_CODES_SUBSTRATE
        prod = dict(r["producer"], substrate=BOTH_CODES_SUBSTRATE)
        rid = I.adapter_record_id(payload, r["record_type"]) if payload != r["payload"] else r["record_id"]
        out.append(R.make_record(r["record_type"], payload, prod, rid, r["written_utc"]))
    return out


def _either_code_tree(records_dir) -> None:
    """Every record file under ``records_dir`` rewritten through :func:`_either_code`, the harvester's own fact records included (this
    code's harvester stamps them with its own substrate)."""
    for dirpath, _, files in os.walk(records_dir):
        for fn in files:
            if R.RECORD_FILE_RE.match(fn) or fn.endswith(".jsonl") or fn.endswith(".jsonl.gz"):
                path = os.path.join(dirpath, fn)
                recs = [rec for _, rec, _ in R.read_record_file(path, compute_sha=False)]
                os.remove(path)
                R.write_records(path, _either_code(recs))


def _registry_with_the_pair(root: str):
    """A registry built by THIS code: a run, its judgment and a same-stamp judged/superseded pair folded `superseded`; the code's version (32
    on this tree: 0031 and 0032 follow the barrier), row 0 stamped."""
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), _either_code(synth.seed_dimensions()))
    rid, recs = synth.run_records("k", 100, through="generated")
    R.write_records(os.path.join(root, "records", "c", "000001-run.jsonl"), _either_code(recs))
    stamp = synth.t(300)
    judged, sup = _pair(rid, stamp)
    jid = I.judgment_id(rid, synth.JUDGE_ID)
    judgment = synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "started_utc": stamp, "finished_utc": stamp,
                                      "n_items_judged": 200, "status": "complete", "api_pool": "lowprio"}, stamp)
    R.write_records(os.path.join(root, "records", "c", "000002-pair.jsonl"), _either_code([judgment, sup, judged]))
    rep = H.ingest_once(root, workers=0, min_free=0, derive_facts=False)
    assert rep.refused == 0, rep.refusals
    live = DB.db_path(root)
    with DB.reading(live) as ro:
        assert ro.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "superseded"
        assert M.db_version(ro) == M.code_version() and DB.fold_order_version(ro) == RD.FOLD_ORDER_VERSION
        assert ro.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    return rid


def _patch_file(root: str, rid: str) -> str:
    """A new record for the run -- what makes a writer refold the key's whole history under ITS order."""
    return R.write_records(os.path.join(root, "records", "c", "000003-patch.jsonl"),
                           _either_code([synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_planned": 7}}, synth.t(500))]))


def _copy(src: str, dst: str) -> None:
    """A consistent copy of a live database (WAL included) through SQLite's backup API."""
    a = DB.connect(src, readonly=True)
    b = sqlite3.connect(dst)
    try:
        a.backup(b)
    finally:
        b.close()
        a.close()


# ---- the fence as a unit ---------------------------------------------------------------------------------------------------------------
def test_the_fence_refuses_a_run_write_by_a_connection_without_the_function_or_with_another_order_and_admits_this_code(tmp_path):
    root = str(tmp_path / "reg")
    rid = _registry_with_the_pair(root)
    live = DB.db_path(root)
    before = _rows(live)
    # the shape: both triggers in the code's DDL, on `run`, recreated by a rebuild of the table; present in a database this code built
    assert all(("trigger", t) in DB.code_ddl_master() for t in FENCE)
    assert [sql.split("\n", 1)[0].split()[2] for sql in S.triggers_on("run") if "fold_order_fence" in sql] == list(FENCE)
    assert all(f"CREATE TRIGGER {t} " in M.render_table_rebuild("run", ["run_id"]) for t in FENCE)
    assert S.FOLD_ORDER_FENCE_TABLES == ("run",)
    with DB.reading(live) as ro:
        assert sorted(r[0] for r in ro.execute("SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'trg_run_fold_order_fence_%'")) == sorted(FENCE)
        assert ro.execute(f"SELECT {DB.FOLD_ORDER_FUNCTION}()").fetchone()[0] == RD.FOLD_ORDER_VERSION      # a reader registers it too
    assert DB.connect_memory().execute(f"SELECT {DB.FOLD_ORDER_FUNCTION}()").fetchone()[0] == RD.FOLD_ORDER_VERSION
    cols = [r[1] for r in sqlite3.connect(live).execute("PRAGMA table_info(run)")]
    insert_copy = "INSERT INTO run SELECT " + ", ".join("run_id || '-copy'" if c == "run_id" else c for c in cols) + " FROM run WHERE run_id = ?"
    writes = (("UPDATE", "UPDATE run SET notes = 'by hand' WHERE run_id = ?"), ("INSERT", insert_copy))
    # a bare connection -- a sqlite3 shell, a process of code from before the function: refused as the write COMPILES, UPDATE and INSERT alike
    bare = sqlite3.connect(live)
    for what, sql in writes:
        with pytest.raises(sqlite3.OperationalError) as ei:
            bare.execute(sql, (rid,))
        assert str(ei.value) == f"no such function: {DB.FOLD_ORDER_FUNCTION}", what
    bare.close()
    assert _rows(live) == before
    # a connection of code that folds under ANOTHER order: the trigger refuses by its own message
    other = sqlite3.connect(live)
    other.create_function(DB.FOLD_ORDER_FUNCTION, 0, lambda: RD.FOLD_ORDERS[1])
    assert other.execute(f"SELECT {DB.FOLD_ORDER_FUNCTION}()").fetchone()[0] == "fold_order/1: written_utc, record_sha256"
    for what, sql in writes:
        with pytest.raises(sqlite3.IntegrityError) as ei:
            other.execute(sql, (rid,))
        assert str(ei.value).startswith("run: fold-order fence -- ") and "schema_migration row 0" in str(ei.value), (what, str(ei.value))
    other.close()
    assert _rows(live) == before
    # this code's connection writes; a database recording NO order is not fenced against it (the bootstrap: require_fold_order stamps it)
    # and stays closed to a bare connection
    conn = DB.connect(live)
    conn.execute("UPDATE run SET notes = 'this code' WHERE run_id = ?", (rid,))
    assert conn.execute("SELECT notes FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "this code"
    conn.execute("UPDATE run SET notes = NULL WHERE run_id = ?", (rid,))
    conn.execute("DELETE FROM schema_migration WHERE version = ?", (DB.FOLD_ORDER_ROW_VERSION,))
    assert DB.fold_order_version(conn) is None
    conn.execute("UPDATE run SET notes = 'unstamped' WHERE run_id = ?", (rid,))
    conn.execute("UPDATE run SET notes = NULL WHERE run_id = ?", (rid,))
    bare = sqlite3.connect(live)
    with pytest.raises(sqlite3.OperationalError):
        bare.execute("UPDATE run SET notes = 'by hand' WHERE run_id = ?", (rid,))
    bare.close()
    assert DB.require_fold_order(conn) == RD.FOLD_ORDER_VERSION                       # the first writer stamps it again
    conn.close()
    assert {k: v for k, v in _rows(live).items() if k != "schema_migration"} == {k: v for k, v in before.items() if k != "schema_migration"}
    # a database WITHOUT the triggers reads as shape drift: this code's writers refuse it by name
    conn = DB.connect(live)
    for t in FENCE:
        conn.execute(f"DROP TRIGGER {t}")
    conn.close()
    with pytest.raises(DB.SchemaDrift) as ei:
        DB.open_or_create(live)
    assert ei.value.lines == FENCE_DRIFT and ei.value.have == ei.value.want == M.code_version()


# ---- the ACTUAL pre-guard writer (layer 1: the version) --------------------------------------------------------------------------------
def test_the_pre_guard_writer_refuses_a_database_at_the_barrier_by_its_own_version_check_and_folds_at_its_own_version(tmp_path):
    pinned_commit_or_skip(PREGUARD, HISTORY_BRANCH, "the pre-guard writer regression of migration 0030", repo=REPO, anchor=Path(__file__))
    old_src = _export(PREGUARD, tmp_path / "preguard")
    # what the exported code is: version 27, the pre-tie-break order, no fold-order guard -- and no patch of ours
    r = _old(old_src, """
        import sisteer, json
        from sisteer.registry import db, migrations, reducer, harvest
        print(json.dumps({"file": sisteer.__file__, "code_version": migrations.code_version(), "tie_rank": hasattr(reducer, "tie_rank"),
                          "fold_order_drift": hasattr(db, "FoldOrderDrift"), "require_fold_order": hasattr(db, "require_fold_order"),
                          "fold_only_files": hasattr(harvest, "fold_only_files")}))
    """)
    assert r.returncode == 0, r.stderr
    facts = json.loads(r.stdout.strip().splitlines()[-1])
    assert facts["file"].startswith(str(old_src)) and facts["code_version"] == 27, facts
    assert facts["tie_rank"] is False and facts["fold_order_drift"] is False and facts["require_fold_order"] is False and facts["fold_only_files"] is True, facts

    # THE BARRIER: a database this code built -- version 30, stamped, the pair folded `superseded`
    root = str(tmp_path / "reg")
    rid = _registry_with_the_pair(root)
    live = DB.db_path(root)
    control_records = tmp_path / "control-records"
    shutil.copytree(os.path.join(root, "records"), control_records)          # the log before the patch, for the control below
    _either_code_tree(control_records)                                       # the harvester's own records too: the pre-guard writer folds them
    patch = _patch_file(root, rid)
    before = _rows(live)
    # the pre-guard writer's open, and its whole harvester tick: refused by ITS OWN SchemaDrift (the version it does not have), nothing written
    for what, code in (("open_or_create", f"""
            from sisteer.registry import db
            db.open_or_create({live!r})
        """), ("ingest_once", f"""
            from sisteer.registry import harvest
            harvest.ingest_once({root!r}, workers=0, min_free=0, derive_facts=False)
        """), ("fold_only_files", f"""
            from sisteer.registry import db, harvest
            conn = db.open_or_create({live!r})
            harvest.fold_only_files({root!r}, conn, [{patch!r}], harvest.IngestReport(), "2026-01-01T00:10:00.000Z")
        """)):
        r = _old(old_src, code)
        assert r.returncode != 0, (what, r.stdout)
        term = _terminal_line(r.stderr)
        assert term.startswith("sisteer.registry.db.SchemaDrift: "), (what, term)
        assert f"registry database at {live} is at schema version {M.code_version()}, code expects 27" in term, (what, term)
        assert _rows(live) == before, f"{what}: the refusal wrote something"
    with DB.reading(live) as ro:
        assert ro.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "superseded"
        assert DB.fold_order_version(ro) == RD.FOLD_ORDER_VERSION and M.db_version(ro) == M.code_version()
        assert ro.execute("SELECT COUNT(*) FROM record_delivery WHERE file_uri LIKE '%000003-patch%'").fetchone()[0] == 0, "the patch never folded"

    # THE CONTROL (the hole): the same log at the pre-guard writer's OWN version. Its own `rebuild` makes a version-27 database (the shape it
    # expects, which has no fence triggers; the pair folds `judged` under its order); THIS code's refold writer then folds the pair
    # `superseded` under the current order and stamps row 0 -- the state the live database was in -- and the UNPATCHED pre-guard writer's tick
    # folds the patch and refolds the key back to `judged`, row 0 untouched: nothing fenced it, and the new code reads agreement. The barrier
    # above is what closes this.
    ctrl = str(tmp_path / "ctrl")
    H.ensure_layout(ctrl)
    shutil.rmtree(os.path.join(ctrl, "records"))
    shutil.copytree(control_records, os.path.join(ctrl, "records"))
    ctrl_db = DB.db_path(ctrl)
    r = _old(old_src, f"""
        import json
        from sisteer.registry import harvest, db, migrations
        rep = harvest.rebuild({ctrl!r}, {ctrl_db!r}, workers=0)
        c = db.connect({ctrl_db!r}, readonly=True)
        print(json.dumps({{"refused": rep.refused, "version": migrations.db_version(c),
                           "state": c.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", ({rid!r},)).fetchone()[0],
                           "run_columns": [r[1] for r in c.execute("PRAGMA table_info(run)")],
                           "fence": [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE '%fold_order_fence%'")]}}))
    """)
    assert r.returncode == 0, r.stderr
    built = json.loads(r.stdout.strip().splitlines()[-1])
    assert built["refused"] == 0 and built["version"] == 27 and built["state"] == "judged" and built["fence"] == [], built      # the old order's draw
    with DB.reading(live) as ro:
        assert built["run_columns"] == [r_[1] for r_ in ro.execute("PRAGMA table_info(run)")], "the run table has the same columns at 27 and at the code's version: the version, the triggers and a CHECK list differ, no column"
    key = I.canonical_json({"run_id": rid})
    conn = DB.connect(ctrl_db)
    try:
        assert RP.refold_apply(conn, [key]) == 1                                  # this code's order written into the row; row 0 stamped
    finally:
        conn.close()
    with DB.reading(ctrl_db) as ro:
        assert ro.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "superseded"
        assert DB.fold_order_version(ro) == RD.FOLD_ORDER_VERSION and M.db_version(ro) == 27
    ctrl_patch = _patch_file(ctrl, rid)
    r = _old(old_src, f"""
        from sisteer.registry import harvest
        rep = harvest.ingest_once({ctrl!r}, workers=0, min_free=0, derive_facts=False)
        print("refused", rep.refused, "ok", rep.ok)
    """)
    assert r.returncode == 0, (r.stdout, r.stderr)
    with DB.reading(ctrl_db) as ro:
        assert ro.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "judged", "the hole: the pre-guard writer refolded the pair under its order"
        assert ro.execute("SELECT n_items_planned FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == 7, "its tick folded the patch"
        assert DB.fold_order_version(ro) == RD.FOLD_ORDER_VERSION, "and row 0 still says the code's order: the stamp alone fences nobody who does not read it"
        assert ro.execute("SELECT COUNT(*) FROM record_delivery WHERE file_uri LIKE '%000003-patch%'").fetchone()[0] == 1
    # and THIS code refuses that database, by the same version check: at 27 it is below the barrier
    with pytest.raises(DB.SchemaDrift) as ei:
        DB.open_or_create(ctrl_db)
    assert ei.value.have == 27 and ei.value.want == M.code_version()


# ---- the ACTUAL direct-connect writer (layer 2: the fence) -----------------------------------------------------------------------------
def test_the_historical_refold_apply_is_refused_at_the_row_while_this_tree_refolds_and_rewrites_the_pair_once_the_fence_is_dropped(tmp_path):
    pinned_commit_or_skip(REFOLD, HISTORY_BRANCH, "the direct-connect refold-apply regression of migration 0030", repo=REPO, anchor=Path(__file__))
    old_src = _export(REFOLD, tmp_path / "refold")
    # what the exported code is: version 22, the pre-tie-break order, no fold-order guard, and a `refold-apply` that opens the file it is
    # given with a bare DB.connect -- the shipped command, not a patch of ours
    r = _old(old_src, """
        import sisteer, json
        from sisteer.registry import db, migrations, reducer, replica
        src = open(replica.__file__, encoding="utf-8").read()
        print(json.dumps({"file": sisteer.__file__, "code_version": migrations.code_version(), "tie_rank": hasattr(reducer, "tie_rank"),
                          "fold_order_drift": hasattr(db, "FoldOrderDrift"), "fold_order_function": hasattr(db, "FOLD_ORDER_FUNCTION"),
                          "refold_apply": hasattr(replica, "refold_apply"), "action": "refold-apply" in replica.ACTIONS,
                          "bare_connect": "conn = DB.connect(args.db)" in src, "open_or_create_in_refold": "open_or_create" in src[src.index('a == "refold-apply"'):src.index('a == "refold-compare"')]}))
    """)
    assert r.returncode == 0, r.stderr
    facts = json.loads(r.stdout.strip().splitlines()[-1])
    assert facts["file"].startswith(str(old_src)) and facts["code_version"] == 22, facts
    assert facts["tie_rank"] is False and facts["fold_order_drift"] is False and facts["fold_order_function"] is False, facts
    assert facts["refold_apply"] is True and facts["action"] is True and facts["bare_connect"] is True and facts["open_or_create_in_refold"] is False, facts

    # THE FENCE: a database this code built -- version 30, stamped, the pair folded `superseded` -- and the shipped command aimed at it
    root = str(tmp_path / "reg")
    rid = _registry_with_the_pair(root)
    live = DB.db_path(root)
    key = I.canonical_json({"run_id": rid})
    keys_path = tmp_path / "keys.json"
    keys_path.write_text(json.dumps([key]), encoding="utf-8")
    before = _rows(live)
    r = _old_replica(old_src, ["refold-apply", "--db", live, "--keys", str(keys_path)])
    assert r.returncode != 0, (r.stdout, r.stderr)
    assert "rewritten" not in r.stdout
    assert _terminal_line(r.stderr) == f"sqlite3.OperationalError: no such function: {DB.FOLD_ORDER_FUNCTION}", r.stderr[-800:]
    assert _rows(live) == before, "the refusal wrote something"
    with DB.reading(live) as ro:
        assert ro.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "superseded"
        assert DB.fold_order_version(ro) == RD.FOLD_ORDER_VERSION and M.db_version(ro) == M.code_version()
    # this tree's refold of the same key on the same file succeeds: the row is already this order's, so it is written back as it stands
    conn = DB.connect(live)
    try:
        assert RP.refold_apply(conn, [key]) == 1
    finally:
        conn.close()
    after = _rows(live)
    assert after["run"] == before["run"] and after["row_history"] == before["row_history"] and after["schema_migration"] == before["schema_migration"]

    # THE CONTROL (the hole): the same database with the two fence triggers dropped -- the file as it stood before 0030 -- and the same
    # command: it rewrites the pair back to `judged` (the old order's draw), row 0 still reading fold_order/2. The fence above is what closes it.
    ctrl = str(tmp_path / "ctrl.sqlite")
    _copy(live, ctrl)
    c = DB.connect(ctrl)
    for t in FENCE:
        c.execute(f"DROP TRIGGER {t}")
    c.close()
    r = _old_replica(old_src, ["refold-apply", "--db", ctrl, "--keys", str(keys_path)])
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "1 run row(s) rewritten from their refold" in r.stdout
    with DB.reading(ctrl) as ro:
        assert ro.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "judged", "the hole: the direct-connect writer refolded the pair under its order"
        assert DB.fold_order_version(ro) == RD.FOLD_ORDER_VERSION and M.db_version(ro) == M.code_version(), "row 0 and the version as they were: neither fenced a writer that reads neither"
    # and THIS code refuses that copy: a database without the fence is shape drift, whatever its version says
    with pytest.raises(DB.SchemaDrift) as ei:
        DB.open_or_create(ctrl)
    assert ei.value.lines == FENCE_DRIFT and ei.value.have == ei.value.want == M.code_version()
