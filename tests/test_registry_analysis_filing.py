"""Analyses filed over ingested runs and objects (migration 0029 `analysis_filing`), on synthetic corpora:

1. the three kinds are LEDGER FACTS: in `RECORD_TYPES`, mapped to ledger-scale tables, in `UNLEDGERED_FACT_TYPES`, so the two ledgers'
   `record_type` CHECK lists never name them; the required fields and the declared FKs;
2. the migration on the shape one version below it: the drift is its nine lines, the plan five creates (four indexes) and NOTHING else --
   no rebuild, no column, no recreate -- the committed file IS the render, its sha is frozen, the runner applies it, the ledgers' CREATE text
   is byte-identical before and after;
3. the records validate against their tables and the two pairing rules refuse by name before the spool;
4. the fold is INSERT OR IGNORE with no `row_history` event, a re-filing is a duplicate, a rebuild reproduces the rows;
5. a record naming a parent the registry does not hold is REFUSED by name (never queued: a ledger-fact kind cannot wait in
   `blocked_records`), and the cycle completes;
6. the CLI's dry run prints the plan and writes nothing; a missing input, a remote report without its sha and a malformed spec refuse by name;
7. the CLI's real run writes the four record files in order through one writer, the harvester folds them, the artifacts are registered (kind by
   suffix; a remote report under its declared sha), `analysis list` and the status page read them back, and a re-filing is a duplicate;
8. the frame export lists the two small tables as a set of their own (the analyses wholly inside the frame; the exporter's own
   tests cover the rule) and keeps `artifact` (and so `analysis_artifact`) out;
9. a digest the registry already holds is LINKED, never re-registered from what the filing alone sees: a remote declaration of a verified
   file keeps the verification and the local location and adds its uri; a local read of a remote-only row fills the verification and keeps the
   primary; a filing that adds nothing writes no artifact record; the rows rebuild equal (the third review round);
10. every artifact record names the ONE location its writer sees and the fold unions `artifact.locations_json` on every record: two filings
    of one known digest from two cluster paths before a fold keep every location, a fresh digest filed remote and local in one window keeps
    the verification in either order, a notebook's re-registration keeps a filing's cluster uri, the validator admits the (now redundant)
    marker `extra.union_columns` on `artifact` alone, and the rows rebuild equal with and without the appended-fold shortcut (the fourth
    review round);
11. an ordered-fact refusal -- conflict, displaced, re-judged -- names the record's smallest uri whatever copy arrives last, so the refusal
    table rebuilds equal (the fourth review round);
12. a local copy verified after a remote declaration of the same bytes PUBLISHES: `publish.registered_sha` reads a path from the row's
    primary or its locations (the standing primary stands), the bytes still checked against the sha (the closing pass);
13. the same bytes under several entries of one filing are ONE link listing every further entry in `extra.same_bytes`, every further
    distinct path writes its own artifact record (the first path the row's primary), and a page whose files are all another page's bytes
    is registered, read back by its url, and publishable (the closing pass);
14. a report re-filed at a path an earlier version holds as its row's primary publishes the bytes the path now holds:
    `publish.registered_shas` holds every row naming the path, newest first, the bytes pick the registration, the earlier row stands as
    history, and bytes matching none are refused naming them all (the second closing pass);
15. the union is the FOLD's rule for `artifact.locations_json`, not a writer's directive: an adapter revision re-registering an unchanged
    input (a record naming the adapter's own locations and carrying no marker) keeps the location an analysis filed for the same bytes,
    the filed copy still publishes, a null carried over a standing list adds nothing, and the rows rebuild equal with and without the
    appended-fold shortcut (the fourth closing pass);
16. a link's `extra.same_bytes` is checked for its shape by the validator (a list of {uri, role, ...} objects; anything else refused by
    name) and the listing and the status page skip an accepted entry that is not an object instead of failing whole (the fourth closing
    pass);
17. a child record is BOUND to the filing that wrote it (`filing_sha256`, the header's record sha, required) and folds only while that
    header is the kept filing or a duplicate of it: a re-filing with another report has its header refused and its input and link refused
    WITH it in either arrival order -- when the earlier filing arrives last its rival's children are withdrawn and refused, when the
    earlier filing's children arrive before their header they wait as dependency refusals and fold the cycle it lands -- a duplicate
    filing's children are the kept analysis's, a child naming another analysis's header is refused by name, and the rows, the index and
    the refusals rebuild equal (the fifth closing pass);
18. the dependency retry moves each PART's acceptances once, so a file spanning several parts with a refusal still standing never reads
    `ok` in `record_delivery` (the fifth closing pass);
19. a re-filing with its input ids in another order is the SAME filing: the normalized spec sorts the input lists, so `spec_sha256` and
    `analysis_id` agree, the header is a duplicate (never a conflicting re-filing) with its inputs and links, and the rows rebuild equal;
    the artifacts and the cluster job ids keep the spec's order (the sixth closing pass);
20. a budget spent before or inside the late phase stops the cycle by name and counts the filing files it left under
    `plain_files_deferred_budget` (the cycle state and the metrics line with it), and the next cycle folds them (the sixth closing pass);
21. a filing is PUBLISHED WHOLE or not at all: its files are staged under `staging/analyses/<token>/` and renamed into
    `records/analyses/<day>/<filing>/` by one rename, so a filing interrupted after its header or after its inputs publishes nothing (the
    harvester folds no header with no inputs or no links), refuses by name naming the staging directory, and the CLI names every
    abandoned staging on every later run; a completed filing publishes whole, its children carrying the header's sha, and rebuilds equal
    from `records/` alone; a destination that already exists refuses by name with the staging left in place (the seventh closing pass);
22. an `artifact` record's `locations_json` in any shape but a JSON list -- an object, a scalar, a string encoding either -- is refused by
    name by the validator, marker or none, so no location is lost to the union's empty reading of it; a list, an empty list and a null
    are admitted, the refusal is a fact, and a well-formed record beside it keeps its uri live and on rebuild (the seventh closing pass);
23. the delivery audit (`record_delivery`, the status page's recent-ingests list) FOLLOWS a record judged again after its file's fold: a
    filing displaced by an earlier-stamped rival has its header's, its input's and its link's files moved to `refused` as a rebuild of the
    same records writes them; a re-filing or a child whose refusal is withdrawn when the kept filing moves is moved back to the accepted
    side; the dependency retry moves each accepted record once, by the fold's own status rule (a wholly refused file with one record now
    accepted reads `partial`); `n_refused`, the accepted total and the status's reading of them equal a shuffled rebuild's, the `ok` /
    `duplicate` split being the fold's arrival order on either path (the eighth closing pass).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil

import pytest

from sisteer.registry import analysis as AN
from sisteer.registry import cli
from sisteer.registry.adapters import base as AB
from sisteer.registry import db as DB
from sisteer.registry import emit as EM
from sisteer.registry import harvest as H
from sisteer.registry import migrations as M
from sisteer.registry import publish as PUB
from sisteer.registry import records as R
from sisteer.registry import reducer as RED
from sisteer.registry import schema_def as S
from sisteer.registry import snapshot as SN
from sisteer.registry import status
from sisteer.registry import synth
from sisteer.registry import validate as VAL
from sisteer.registry.identity import canonical_json
from sisteer.registry import vocab as V

from test_registry_migrations import AF_DRIFT, AF_SLUG, AF_TABLES, FENCE_TRIGGERS, NTPL_TABLES, _sans_fence_drift, _sans_fence_render
from test_registry_schema import FROZEN_MIGRATIONS

KINDS = ("analysis", "analysis_input", "analysis_artifact")
SHA_A = "a" * 64
FSHA = "d" * 64                                            # a stand-in header sha for records that are validated, never folded
AID = "11111111-2222-5333-8444-555555555555"


def _root(tmp_path, name="reg"):
    root = str(tmp_path / name)
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=2)
    H.ingest_once(root, workers=0, min_free=0)
    return root, ids


def _q(root, sql, *params):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _count(root, table):
    return _q(root, f"SELECT COUNT(*) AS n FROM {table}")[0]["n"]


def _page_dir(tmp_path, name="page"):
    d = tmp_path / name
    (d / "figures").mkdir(parents=True)
    (d / "index.html").write_text("<html>synthetic page</html>", encoding="utf-8")
    (d / "numbers.json").write_text('{"n": 1}', encoding="utf-8")
    (d / "figures" / "a.png").write_bytes(b"\x89PNG synthetic a")
    (d / "figures" / "a.svg").write_bytes(b"<svg/>")
    (d / "notes.txt").write_text("not publishable", encoding="utf-8")
    return str(d)


def _spec(ids, page_dir=None, report=None, **over):
    spec = {"name": "study.capture", "code_sha": "c" * 40, "code_ref": "branch study @ ccccccccc", "cluster_job_ids": ["123456", "123457"],
            "substrate": "cluster_a", "report": report, "inputs": {"runs": list(ids["run_ids"]), "objects": [synth.OBJECT_SHA, "none"]},
            "artifacts": ([{"dir": page_dir, "role": "page", "url": "https://example.invalid/figs/study/"}] if page_dir else []),
            "filed_by": "the user", "provenance": {"kind": "relayed", "relayed_by": "TESTSESSION", "message_utc": "2026-10-08T21:07:32.152Z"},
            "notes": "synthetic", "extra": {"study": "synthetic"}}
    spec.update(over)
    return spec


def _write_spec(tmp_path, spec, name="spec.json"):
    p = tmp_path / name
    p.write_text(json.dumps(spec), encoding="utf-8")
    return str(p)


def _analysis_payload(aid=AID, **over):
    p = {"analysis_id": aid, "name": "study.capture", "code_sha": "c" * 40, "code_ref": "branch study", "cluster_job_ids_json": ["1"],
         "substrate": "cluster_a", "report_uri": None, "report_sha256": None, "filed_utc": synth.t(500), "filed_by": "the user", "notes": None,
         "extra": {"provenance": {"kind": "relayed", "relayed_by": "TESTSESSION", "message_utc": "2026-10-08T21:07:32.152Z"}, "decision": "D357"}}
    p.update(over)
    return p


def _rebuild_copy(root, tmp_path, shuffle_seed=3):
    """A rebuild of the root's records from a shuffled copy of them; returns the rebuilt database's path."""
    copy = str(tmp_path / f"copy{shuffle_seed}")
    files = list(R.iter_record_files(os.path.join(root, "records")))
    random.Random(shuffle_seed).shuffle(files)
    for f in files:
        dest = os.path.join(copy, "records", os.path.relpath(f, os.path.join(root, "records")))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(f, dest)
    out = str(tmp_path / f"rebuilt{shuffle_seed}.sqlite")
    H.rebuild(copy, out, workers=0)
    return out


def _rebuild_equal(root, tmp_path, shuffle_seed=3):
    out = _rebuild_copy(root, tmp_path, shuffle_seed)
    a = DB.connect(DB.db_path(root), readonly=True)
    b = DB.connect(out, readonly=True)
    try:
        diff = {k: v for k, v in H.verify_equal(a, b).items() if v}
        assert not diff, diff
        return {t: b.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in KINDS}
    finally:
        a.close()
        b.close()


# --------------------------------------------------------------------------------------------
# 1: the kinds, the tables, the ledgers
# --------------------------------------------------------------------------------------------


def test_the_kinds_are_ledger_facts_and_the_two_ledgers_check_lists_never_name_them():
    assert V.ANALYSIS_RECORD_TYPES == KINDS
    for k in KINDS:
        assert k in R.RECORD_TYPES and R.RECORD_TYPE_TABLE[k] == k and k in V.UNLEDGERED_FACT_TYPES and k not in V.LEDGERED_RECORD_TYPES
        assert k in S.LEDGER_TABLES and k not in S.RUN_SCALE_TABLES and k not in S.ITEM_SCALE_TABLES
        assert S.TABLE_BY_NAME[k].scale == "ledger"
    assert V.UNLEDGERED_FACT_TYPES == ("record_refusal",) + KINDS
    for t in ("blocked_records", "row_history"):
        ddl = S.render_create_table(S.TABLE_BY_NAME[t])
        assert S.TABLE_BY_NAME[t].column("record_type").vocab == "LEDGERED_RECORD_TYPES" and "'analysis" not in ddl, t
    assert R.REQUIRED_FIELDS["analysis"] == ("analysis_id", "name", "code_sha", "substrate", "filed_utc", "filed_by")
    assert R.REQUIRED_FIELDS["analysis_input"] == ("analysis_id", "filing_sha256", "input_kind", "input_id")
    assert R.REQUIRED_FIELDS["analysis_artifact"] == ("analysis_id", "filing_sha256", "artifact_id", "role")
    # the binding to the filing (the fifth closing pass): a required sha column on both child tables, no FK, an EVENT column of the ordering rule
    assert R.LEDGER_FILING_CHILDREN == {"analysis_input": ("analysis", "filing_sha256"), "analysis_artifact": ("analysis", "filing_sha256")}
    for t in ("analysis_input", "analysis_artifact"):
        col = S.TABLE_BY_NAME[t].column("filing_sha256")
        assert col.nn and col.pattern == "SHA256" and col.fk is None and not col.derived
        assert R.LEDGER_ORDERED_EVENT_COLUMNS[t] == ("filing_sha256",)
    assert "filing_sha256" not in S.payload_columns("analysis") and H.FILING_PARENT == "analysis_filing"
    assert S.foreign_keys("analysis_input") == [("analysis_id", "analysis", "analysis_id"), ("run_id", "run", "run_id"), ("object_id", "object", "object_id")]
    assert S.foreign_keys("analysis_artifact") == [("analysis_id", "analysis", "analysis_id"), ("artifact_id", "artifact", "artifact_id")]
    assert S.foreign_keys("analysis") == []
    assert V.VOCABULARIES["ANALYSIS_INPUT_KINDS"] == ("run", "object") and V.VOCABULARIES["ANALYSIS_ARTIFACT_ROLES"] == ("report", "page", "figure")
    assert R.NATURAL_KEYS["analysis_input"] == ("analysis_id", "input_kind", "input_id") and R.NATURAL_KEYS["analysis_artifact"] == ("analysis_id", "artifact_id")
    # the frame export: the two small tables for the analyses wholly inside the frame (their own set, never the dimensions exported in
    # full: tests/test_snapshot_paper_frame.py), the artifact link out with its parent (review C7)
    assert SN.PAPER_FRAME_ANALYSIS_TABLES == ("analysis", "analysis_input") and set(SN.PAPER_FRAME_ANALYSIS_TABLES) <= set(SN.PAPER_FRAME_TABLES)
    assert not set(SN.PAPER_FRAME_ANALYSIS_TABLES) & set(SN.PAPER_FRAME_DIMENSION_TABLES)
    assert "analysis_artifact" not in SN.PAPER_FRAME_TABLES and "artifact" not in SN.PAPER_FRAME_TABLES


# --------------------------------------------------------------------------------------------
# 2: the migration on the shape below it
# --------------------------------------------------------------------------------------------


def _af_migration():
    return next(f for f in M.migration_files() if f[1] == AF_SLUG)


def _shape_below(path: str) -> None:
    version = _af_migration()[0]
    conn = DB.connect(path)
    for v, slug, mpath in M.migration_files():
        if v < version:
            M._apply_one(conn, v, slug, open(mpath, encoding="utf-8").read())
    conn.commit()
    conn.close()


def test_the_migration_on_the_shape_below_it_is_five_creates_and_no_rebuild_and_the_committed_body_is_the_render(tmp_path, capsys):
    version, slug, mpath = _af_migration()
    assert version == M.code_version() - 3, "the newest file but three on this tree: 0030 (fold_order_barrier, the fold-order version's deployment barrier: two fence triggers on run), 0031 (non_template_positions: a rebuild of instrument and run) and 0032 (site_label_rename: the five tables whose CHECK names a site label, rebuilt) follow it; their lines are set aside below (0032 adds none on a shape built from these files)"
    assert os.path.basename(mpath) == f"{version:04d}_{AF_SLUG}.sql"
    committed = open(mpath, encoding="utf-8").read()
    assert FROZEN_MIGRATIONS[os.path.basename(mpath)] == hashlib.sha256(committed.encode("utf-8")).hexdigest()
    path = str(tmp_path / "below.sqlite")
    _shape_below(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == version - 1
    assert _sans_fence_drift(DB.schema_drift(conn)) == AF_DRIFT                              # exactly this file's lines (0030's two triggers aside): nothing else moved on the shape below
    have, want = DB.ddl_master(conn), DB.code_ddl_master()
    ledgers_before = {t: conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (t,)).fetchone()[0] for t in ("blocked_records", "row_history")}
    for t in ("blocked_records", "row_history"):
        assert have[("table", t)] == want[("table", t)], t                                 # the CHECK lists are byte-identical: no ledger rebuild
    plan = M.shape_plan(conn)
    assert plan["create"] == list(AF_TABLES) and [r["table"] for r in plan["rebuild"]] == list(NTPL_TABLES) and plan["add_columns"] == [] and plan["undeclared_tables"] == []   # 0031's two rebuilds ride along against the code's shape
    assert plan["objects"] == {"index": {"recreate": [], "drop": []}, "trigger": {"recreate": [], "drop": []}, "view": {"recreate": [], "drop": []}}   # 0030's two fence triggers ride 0031's run rebuild
    rendered = M.render_shape_migration(conn, plan)
    header = committed[:committed.index("-- drift: ")]
    assert committed.startswith(f"-- {os.path.basename(mpath)}\n") and "a registry decision of 2026-10-08" in header and "NO rebuild" in header
    body = committed[committed.index("-- drift: "):]
    assert body.strip() == _sans_fence_render(rendered).strip()                             # today's render carries 0030's two triggers too
    assert body.count("\nCREATE TABLE ") == 5 and body.count("\nCREATE INDEX ") == 4                # the three analysis tables, the record index, the carrier audit
    for t in AF_TABLES:
        assert f"-- create {t}\nCREATE TABLE {t} (" in body, t
    for absent in ("__new", "INSERT INTO", "ALTER TABLE", "DROP VIEW", "CREATE VIEW", "DROP TABLE", "CREATE TRIGGER", "rebuild"):
        assert absent not in body, absent
    assert AF_SLUG not in M.POST_APPLY                                                       # no Python post-step: the tables start empty
    conn.close()
    # the CLI's plan on a COPY: the pending file alone, five `create table` lines, no rebuild / add-column / recreate line, exit 1
    root = str(tmp_path / "copyroot")
    os.makedirs(root)
    shutil.copy(path, DB.db_path(root))
    assert cli.main(["--root", root, "migrate", "--plan"]) == 1
    out = capsys.readouterr().out
    assert f"schema version {version - 1}; code {M.code_version()}; pending: {version:04d}_{AF_SLUG}, {version + 1:04d}_fold_order_barrier, {version + 2:04d}_non_template_positions, {version + 3:04d}_site_label_rename\n" in out   # 0030 and 0031 follow it in the pending list
    assert [ln for ln in out.splitlines() if ln.startswith("create table ")] == [f"create table {t}" for t in AF_TABLES]
    for absent in ("add column", "drop undeclared"):
        assert absent not in out, out
    assert [ln for ln in out.splitlines() if ln.startswith("rebuild table ")] == [f"rebuild table {t}: 0 rows to copy (run-scale)" for t in NTPL_TABLES]   # 0031's two, pending behind it; none of this file's
    assert [ln for ln in out.splitlines() if ln.startswith("recreate ")] == []   # 0030's two fence triggers ride 0031's run rebuild
    for d in AF_DRIFT:
        assert f"  {d}" in out
    # the runner: the version row, the five tables, zero drift, the ledgers' stored CREATE text untouched, the writers' open accepts it
    conn = DB.connect(path)
    notes = []
    assert M.apply_migrations(conn, notes) == [version, version + 1, version + 2, version + 3] and notes == [] and M.db_version(conn) == M.code_version() and DB.schema_drift(conn) == []   # 0030 and 0031 follow
    assert conn.execute("SELECT description FROM schema_migration WHERE version = ?", (version,)).fetchone()[0] == AF_SLUG
    for t, sql in ledgers_before.items():
        assert conn.execute("SELECT sql FROM sqlite_master WHERE name = ?", (t,)).fetchone()[0] == sql, t
    for t in AF_TABLES:
        assert conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0
    assert M.foreign_key_violations(conn, list(AF_TABLES)) == []
    conn.close()
    DB.open_or_create(path).close()


# --------------------------------------------------------------------------------------------
# 3: validation
# --------------------------------------------------------------------------------------------


def test_the_records_validate_against_their_tables_and_the_pairing_rules_refuse_by_name():
    rec = synth.rec("analysis", _analysis_payload(), synth.t(500))
    v = VAL.validate_record(rec, "<t>", R.record_sha(rec))
    assert isinstance(v, VAL.Validated) and v.table == "analysis" and v.payload["cluster_job_ids_json"] == '["1"]'
    assert v.extra["provenance"]["relayed_by"] == "TESTSESSION" and VAL.parent_references(v) == []
    bad = synth.rec("analysis", _analysis_payload(report_uri="/x/report.json"), synth.t(500))
    out = VAL.validate_record(bad, "<t>", R.record_sha(bad))
    assert isinstance(out, VAL.Malformed) and out.field == "report_sha256" and "together" in out.reason
    bad = synth.rec("analysis", _analysis_payload(report_sha256=SHA_A), synth.t(500))
    out = VAL.validate_record(bad, "<t>", R.record_sha(bad))
    assert isinstance(out, VAL.Malformed) and out.field == "report_uri"
    bad = synth.rec("analysis", {k: v for k, v in _analysis_payload().items() if k != "filed_by"}, synth.t(500))
    out = VAL.validate_record(bad, "<t>", R.record_sha(bad))
    assert isinstance(out, VAL.Malformed) and out.field == "filed_by"
    bad = synth.rec("analysis", _analysis_payload(code_sha="63cecaa7e"), synth.t(500))
    out = VAL.validate_record(bad, "<t>", R.record_sha(bad))
    assert isinstance(out, VAL.Malformed) and out.field == "code_sha"
    # inputs: the kind's FK column AND input_id name the input; the parent references read the FK; `filing_sha256` names the header (no FK)
    rid = "camp:ev:abcdefabcdef:0123456789abcdef"
    rec = synth.rec("analysis_input", {"analysis_id": AID, "filing_sha256": FSHA, "input_kind": "run", "input_id": rid, "run_id": rid, "object_id": None}, synth.t(501))
    v = VAL.validate_record(rec, "<t>", R.record_sha(rec))
    assert isinstance(v, VAL.Validated) and VAL.parent_references(v) == [("analysis", "analysis_id", AID), ("run", "run_id", rid)]
    rec = synth.rec("analysis_input", {"analysis_id": AID, "filing_sha256": FSHA, "input_kind": "object", "input_id": SHA_A, "object_id": SHA_A}, synth.t(501))
    v = VAL.validate_record(rec, "<t>", R.record_sha(rec))
    assert isinstance(v, VAL.Validated) and VAL.parent_references(v) == [("analysis", "analysis_id", AID), ("object", "object_id", SHA_A)]
    for payload, field_name in (({"analysis_id": AID, "filing_sha256": FSHA, "input_kind": "run", "input_id": rid}, "run_id"),
                                ({"analysis_id": AID, "filing_sha256": FSHA, "input_kind": "run", "input_id": rid, "run_id": rid, "object_id": SHA_A}, "run_id"),
                                ({"analysis_id": AID, "filing_sha256": FSHA, "input_kind": "object", "input_id": SHA_A, "run_id": rid}, "object_id"),
                                ({"analysis_id": AID, "filing_sha256": FSHA, "input_kind": "transcript", "input_id": rid}, "input_kind"),
                                # the binding is required, and a sha: a child naming no filing (or a short one) is refused by name before the fold
                                ({"analysis_id": AID, "input_kind": "run", "input_id": rid, "run_id": rid}, "filing_sha256"),
                                ({"analysis_id": AID, "filing_sha256": "abc123", "input_kind": "run", "input_id": rid, "run_id": rid}, "filing_sha256")):
        bad = synth.rec("analysis_input", payload, synth.t(501))
        out = VAL.validate_record(bad, "<t>", R.record_sha(bad))
        assert isinstance(out, VAL.Malformed) and out.field == field_name, (payload, out)
    rec = synth.rec("analysis_artifact", {"analysis_id": AID, "filing_sha256": FSHA, "artifact_id": SHA_A, "role": "page", "extra": {"page_url": "https://example.invalid/p/"}}, synth.t(502))
    v = VAL.validate_record(rec, "<t>", R.record_sha(rec))
    assert isinstance(v, VAL.Validated) and VAL.parent_references(v) == [("analysis", "analysis_id", AID), ("artifact", "artifact_id", SHA_A)]
    bad = synth.rec("analysis_artifact", {"analysis_id": AID, "filing_sha256": FSHA, "artifact_id": SHA_A, "role": "deck"}, synth.t(502))
    out = VAL.validate_record(bad, "<t>", R.record_sha(bad))
    assert isinstance(out, VAL.Malformed) and out.field == "role"
    bad = synth.rec("analysis_artifact", {"analysis_id": AID, "artifact_id": SHA_A, "role": "page"}, synth.t(502))
    out = VAL.validate_record(bad, "<t>", R.record_sha(bad))
    assert isinstance(out, VAL.Malformed) and out.field == "filing_sha256" and "required" in out.reason


# --------------------------------------------------------------------------------------------
# 4-5: the fold
# --------------------------------------------------------------------------------------------


def _header_sha_of(path):
    """The record sha of the one `analysis` record in a header file: what the filing's children carry in `filing_sha256`."""
    return next(sha for _line, _rec, sha in R.read_record_file(path))


def _hand_filing(root, ids, day="2026-01-02", seq0=1, aid=AID, run_id=None, artifact_sha=None, with_artifact_record=True, header_over=None):
    """A filing written by hand under records/analyses/<day>/ in the CLI's order: the artifact record, the analysis, one run input, one
    object input, the artifact link -- one stamp per file, strictly increasing; every child bound to the header's record sha
    (`filing_sha256`), as the CLI writes it. ``header_over`` overrides fields of the header payload."""
    rid = run_id or ids["run_ids"][0]
    sha = artifact_sha or hashlib.sha256(b"synthetic report").hexdigest()
    d = os.path.join(root, "records", "analyses", day)
    paths = []
    if with_artifact_record:
        art = {"artifact_id": sha, "kind": "report", "uri_primary": f"/synthetic/{aid[:8]}/report.json", "locations_json": [], "bytes": 16,
               "created_utc": synth.t(600), "verified_utc": synth.t(600), "extra": {"analysis": "study.capture"}}
        paths.append(R.write_records(os.path.join(d, f"{seq0:06d}-artifact.jsonl"), [synth.rec("artifact", art, synth.t(600))]))
    header = synth.rec("analysis", _analysis_payload(aid, filed_utc=synth.t(601), **(header_over or {})), synth.t(601))
    hsha = R.record_sha(header)
    paths.append(R.write_records(os.path.join(d, f"{seq0 + 1:06d}-analysis.jsonl"), [header]))
    inputs = [{"analysis_id": aid, "filing_sha256": hsha, "input_kind": "run", "input_id": rid, "run_id": rid, "object_id": None},
              {"analysis_id": aid, "filing_sha256": hsha, "input_kind": "object", "input_id": synth.OBJECT_SHA, "run_id": None, "object_id": synth.OBJECT_SHA}]
    paths.append(R.write_records(os.path.join(d, f"{seq0 + 2:06d}-analysis_input.jsonl"), [synth.rec("analysis_input", p, synth.t(602)) for p in inputs]))
    link = {"analysis_id": aid, "filing_sha256": hsha, "artifact_id": sha, "role": "report", "extra": {}}
    paths.append(R.write_records(os.path.join(d, f"{seq0 + 3:06d}-analysis_artifact.jsonl"), [synth.rec("analysis_artifact", link, synth.t(603))]))
    return paths, sha


def test_the_fold_is_insert_or_ignore_with_no_row_history_a_refiling_is_a_duplicate_and_a_rebuild_reproduces_the_rows(tmp_path):
    root, ids = _root(tmp_path)
    _hand_filing(root, ids)
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(700))
    assert rep.refused == 0 and rep.blocked == 0 and rep.analysis_records == 4           # the analysis, two inputs, one link; the artifact is `ok`
    assert _count(root, "analysis") == 1 and _count(root, "analysis_input") == 2 and _count(root, "analysis_artifact") == 1
    row = _q(root, "SELECT * FROM analysis")[0]
    assert row["analysis_id"] == AID and row["cluster_job_ids_json"] == '["1"]' and json.loads(row["extra_json"])["provenance"]["kind"] == "relayed"
    assert "field_versions_json" not in row and "ingest_seq" not in row                     # a ledger row: no stamps, no pointer, no rank
    assert _q(root, "SELECT input_kind, input_id, run_id, object_id FROM analysis_input ORDER BY input_kind") == [
        {"input_kind": "object", "input_id": synth.OBJECT_SHA, "run_id": None, "object_id": synth.OBJECT_SHA},
        {"input_kind": "run", "input_id": ids["run_ids"][0], "run_id": ids["run_ids"][0], "object_id": None}]
    assert _q(root, "SELECT COUNT(*) AS n FROM row_history WHERE table_name IN ('analysis', 'analysis_input', 'analysis_artifact')")[0]["n"] == 0
    assert _q(root, "SELECT COUNT(*) AS n FROM row_history WHERE table_name = 'artifact'")[0]["n"] >= 1   # the artifact folded through the run-scale path
    assert _count(root, "blocked_records") == 0 and _count(root, "record_refusal") == 0
    # a re-filing of the same analysis (new record ids, the same payloads): every row is a duplicate, nothing moves
    _hand_filing(root, ids, day="2026-01-03", seq0=1)
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(701))
    assert rep.analysis_records == 0 and rep.duplicates >= 4 and rep.refused == 0
    assert _count(root, "analysis") == 1 and _count(root, "analysis_input") == 2 and _count(root, "analysis_artifact") == 1
    assert _rebuild_equal(root, tmp_path) == {"analysis": 1, "analysis_input": 2, "analysis_artifact": 1}


def test_a_record_naming_a_parent_the_registry_does_not_hold_is_refused_by_name_never_queued_and_the_cycle_completes(tmp_path):
    root, ids = _root(tmp_path)
    ghost_run = "camp:ev:abcdefabcdef:0000000000000000"
    ghost_art = "f" * 64
    _hand_filing(root, ids, run_id=ghost_run, artifact_sha=ghost_art, with_artifact_record=False)
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(700))
    # the analysis and the object input fold; the run input and the artifact link are refused, naming the FK column, the parent and the id
    assert rep.refused == 2 and rep.blocked == 0 and rep.analysis_records == 2
    assert _count(root, "analysis") == 1 and _count(root, "analysis_input") == 1 and _count(root, "analysis_artifact") == 0
    assert _count(root, "blocked_records") == 0                                             # never queued: the queue's CHECK could not hold the kind
    refused = {r["record_type"]: r for r in _q(root, "SELECT record_type, field, reason FROM record_refusal")}
    assert set(refused) == {"analysis_input", "analysis_artifact"}
    assert refused["analysis_input"]["field"] == "run_id" and f"run {ghost_run} is not in this registry" in refused["analysis_input"]["reason"]
    assert refused["analysis_artifact"]["field"] == "artifact_id" and f"artifact {ghost_art} is not in this registry" in refused["analysis_artifact"]["reason"]
    assert H.dependency_refusal_parent(refused["analysis_input"]["reason"]) == ("run", ghost_run)
    assert "cannot wait in the blocked queue" in refused["analysis_input"]["reason"]
    # the refusal is a fact like any other
    assert _q(root, "SELECT COUNT(*) AS n FROM record_refusal_fact WHERE record_type = 'analysis_input'")[0]["n"] == 1
    # a child ahead of its parent in the SAME cycle is not a refusal: the records sort by stamp, so the analysis folds first (above). A child
    # whose parent analysis is absent altogether refuses on `analysis_id`
    other = "22222222-3333-5444-8555-666666666666"
    R.write_records(os.path.join(root, "records", "analyses", "2026-01-04", "000001-analysis_input.jsonl"),
                    [synth.rec("analysis_input", {"analysis_id": other, "filing_sha256": FSHA, "input_kind": "run", "input_id": ids["run_ids"][1], "run_id": ids["run_ids"][1]}, synth.t(610))])
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(701))
    assert rep.refused == 1 and rep.blocked == 0
    row = _q(root, "SELECT field, reason FROM record_refusal WHERE record_type = 'analysis_input' AND field = 'analysis_id'")
    assert len(row) == 1 and f"analysis {other} is not in this registry" in row[0]["reason"]
    assert _rebuild_equal(root, tmp_path) == {"analysis": 1, "analysis_input": 1, "analysis_artifact": 0}


# --------------------------------------------------------------------------------------------
# 6-7: the CLI
# --------------------------------------------------------------------------------------------


def _no_filing_written(root):
    assert not os.path.isdir(os.path.join(root, "records", "analyses"))
    assert not AN.abandoned_stagings(root)                                                  # and nothing left half-staged either
    assert _count(root, "analysis") == 0 and _count(root, "analysis_input") == 0 and _count(root, "analysis_artifact") == 0


def test_the_dry_run_prints_the_plan_and_writes_nothing_and_the_refusals_name_their_field(tmp_path, capsys):
    root, ids = _root(tmp_path)
    page = _page_dir(tmp_path)
    report = tmp_path / "report.json"
    report.write_text('{"delta": 0.1}', encoding="utf-8")
    spec = _spec(ids, page, report={"uri": str(report), "sha256": None})
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec), "--dry-run"]) == 0
    out = capsys.readouterr().out
    aid = AN.analysis_id(AN.normalize_spec(spec))
    assert f"analysis study.capture -> analysis_id {aid}" in out and "dry run: nothing written" in out
    assert "inputs: 2 run(s), 2 object(s)" in out and "MISSING" not in out and "artifacts: 5 file(s) to register" in out
    assert "report " + str(report) in out and PUB.sha256_of(str(report))[:12] in out
    assert "1 artifact file (5 record(s)), 1 analysis, 4 analysis_input, 5 analysis_artifact" in out
    _no_filing_written(root)
    assert not os.path.exists(os.path.join(root, "state", "last_run", "analysis-file.json"))   # a read-only command stamps nothing
    # a missing input: exit 1, the run named, nothing written (the real run refuses the same way before any file)
    ghost = "camp:ev:abcdefabcdef:0000000000000000"
    bad = _spec(ids, page, inputs={"runs": list(ids["run_ids"]) + [ghost], "objects": [synth.OBJECT_SHA]})
    for extra_args in (["--dry-run"], []):
        assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, bad, "bad.json")] + extra_args) == 1
        err = capsys.readouterr().err
        assert f"run {ghost}  MISSING" in err and "REFUSED: 1 input(s) not in this registry" in err and "nothing written" in err
        _no_filing_written(root)
    # a remote report with no declared sha: refused by name, nothing written
    remote = _spec(ids, page, report={"uri": "/scratch/example/someone/exp/study/report.json", "sha256": None, "substrate": "cluster_a"})
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, remote, "remote.json")]) == 2
    err = capsys.readouterr().err
    assert "refused by name: report.sha256:" in err and "declares no sha256" in err
    _no_filing_written(root)
    # a malformed spec: the field named
    for over, field_name in (({"code_sha": "63cecaa7e"}, "code_sha"), ({"substrate": "laptop"}, "substrate"), ({"inputs": {"runs": [], "objects": []}}, "inputs"),
                             ({"cluster_job_id": ["1"]}, "cluster_job_id"),
                             ({"inputs": {"runs": list(ids["run_ids"]), "transcripts": ["x"]}}, "inputs.transcripts"),
                             ({"provenance": {"kind": "relayed"}}, "provenance"), ({"artifacts": [{"dir": page, "path": "x"}]}, "artifacts[0]"),
                             ({"artifacts": [{"dir": str(tmp_path / "nowhere")}]}, "artifacts[0].dir"), ({"artifacts": [{"path": "/nowhere/x.png"}]}, "artifacts[0].sha256")):
        assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, _spec(ids, page, **over), "m.json"), "--dry-run"]) == 2
        err = capsys.readouterr().err
        assert f"refused by name: {field_name}:" in err, (over, err)
    _no_filing_written(root)
    # the dry run on an empty root (no database yet): refused by name, nothing written
    empty = str(tmp_path / "empty")
    H.ensure_layout(empty)
    assert cli.main(["--root", empty, "analysis", "file", "--spec", _write_spec(tmp_path, spec), "--dry-run"]) == 1
    assert "cannot read the database" in capsys.readouterr().err


def test_the_filing_writes_four_record_files_in_order_the_harvester_folds_them_and_list_and_the_status_page_read_them_back(tmp_path, capsys):
    root, ids = _root(tmp_path)
    page = _page_dir(tmp_path)
    report = tmp_path / "report.json"
    report.write_text('{"delta": 0.1}', encoding="utf-8")
    spec = _spec(ids, page, report={"uri": str(report), "sha256": PUB.sha256_of(str(report))})
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    aid = AN.analysis_id(AN.normalize_spec(spec))
    assert out["analysis_id"] == aid and len(out["written"]) == 4 and len(out["artifacts"]) == 5
    day = R.utc_now()[:10]
    d = os.path.join(root, "records", "analyses", day)
    assert [os.path.basename(p) for p in out["written"]] == ["000001-artifact.jsonl", "000002-analysis.jsonl", "000003-analysis_input.jsonl", "000004-analysis_artifact.jsonl"]
    # the filing's own directory under the day, named by its header's sha12, published whole by one rename (the seventh closing pass):
    # nothing staged is left behind
    filing_dir = os.path.dirname(out["written"][0])
    assert os.path.dirname(filing_dir) == d and os.path.basename(filing_dir) == AN.filing_dir_name(out["filing_sha256"]) == out["filing_sha256"][:12]
    assert all(os.path.dirname(p) == filing_dir for p in out["written"]) and out["filing_dir"] == filing_dir
    assert AN.filing_sha_of(out["written"][1]) == out["filing_sha256"] and not AN.abandoned_stagings(root)
    # the database is untouched until the harvester folds the records
    assert _count(root, "analysis") == 0 and _count(root, "analysis_artifact") == 0
    recs = {os.path.basename(p): [r for _, r, _ in R.read_record_file(p)] for p in out["written"]}
    stamps = [recs[f][0]["written_utc"] for f in sorted(recs)]
    assert stamps == sorted(stamps) and len(set(stamps)) == 4                               # strictly increasing across the four files
    for f, rs in recs.items():
        for r in rs:
            assert r["producer"]["script"] == AN.PRODUCER_SCRIPT and r["producer"]["code_sha"] and r["producer"]["host"]
            assert "run_overrides" not in r["payload"].get("extra", {}) and "cluster_label" not in r["payload"].get("extra", {})
    ana = recs["000002-analysis.jsonl"][0]["payload"]
    assert ana["analysis_id"] == aid and ana["extra"]["provenance"] == spec["provenance"] and ana["extra"]["decision"] == "D357"
    assert ana["extra"]["study"] == "synthetic" and len(ana["extra"]["spec_sha256"]) == 64 and ana["report_sha256"] == PUB.sha256_of(str(report))
    assert {(r["payload"]["input_kind"], r["payload"]["input_id"]) for r in recs["000003-analysis_input.jsonl"]} == {("run", ids["run_ids"][0]), ("run", ids["run_ids"][1]), ("object", synth.OBJECT_SHA), ("object", "none")}
    arts = {r["payload"]["uri_primary"]: r["payload"] for r in recs["000001-artifact.jsonl"]}
    assert arts[str(report)]["kind"] == "report" and arts[os.path.join(page, "index.html")]["kind"] == "page"
    assert arts[os.path.join(page, "figures", "a.png")]["kind"] == "figure" and arts[os.path.join(page, "numbers.json")]["kind"] == "report"
    assert os.path.join(page, "notes.txt") not in arts                                      # not a publishable suffix
    assert arts[os.path.join(page, "figures", "a.svg")]["extra"] == {"analysis": "study.capture", "page_dir": page, "relpath": "figures/a.svg", "page_url": "https://example.invalid/figs/study/"}
    assert all("union_columns" not in a["extra"] for a in arts.values())                      # no fold directive: the union is the fold's rule
    links = {r["payload"]["artifact_id"]: r["payload"] for r in recs["000004-analysis_artifact.jsonl"]}
    assert links[PUB.sha256_of(str(report))]["role"] == "report" and links[PUB.sha256_of(os.path.join(page, "index.html"))]["role"] == "page"
    assert links[PUB.sha256_of(os.path.join(page, "figures", "a.png"))]["extra"]["page_url"] == "https://example.invalid/figs/study/"
    # the harvester folds them: one cycle, every parent ahead of its child
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0 and rep.analysis_records == 1 + 4 + 5
    assert _count(root, "analysis") == 1 and _count(root, "analysis_input") == 4 and _count(root, "analysis_artifact") == 5
    assert _q(root, "SELECT kind, COUNT(*) AS n FROM artifact WHERE artifact_id IN (SELECT artifact_id FROM analysis_artifact) GROUP BY kind ORDER BY kind") == [
        {"kind": "figure", "n": 2}, {"kind": "page", "n": 1}, {"kind": "report", "n": 2}]
    row = _q(root, "SELECT * FROM analysis")[0]
    assert row["name"] == "study.capture" and row["report_uri"] == str(report) and row["report_sha256"] == PUB.sha256_of(str(report)) and row["filed_by"] == "the user"
    assert json.loads(row["cluster_job_ids_json"]) == ["123456", "123457"] and json.loads(row["extra_json"])["provenance"]["relayed_by"] == "TESTSESSION"
    # read back: the listing, its json, the status page's section with the page link
    assert cli.main(["--root", root, "analysis", "list"]) == 0
    text = capsys.readouterr().out
    assert "study.capture" in text and "relayed by TESTSESSION" in text and "inputs 2 run(s) / 2 object(s); 5 artifact(s)" in text and "https://example.invalid/figs/study/" in text
    assert cli.main(["--root", root, "analysis", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert len(listed) == 1 and listed[0]["analysis_id"] == aid and len(listed[0]["inputs"]) == 4 and len(listed[0]["artifacts"]) == 5
    assert {a["role"] for a in listed[0]["artifacts"]} == {"report", "page"} and listed[0]["page_urls"] == "https://example.invalid/figs/study/"
    assert not os.path.exists(os.path.join(root, "state", "last_run", "analysis-list.json"))    # list is read-only: no stamp
    assert os.path.exists(os.path.join(root, "state", "last_run", "analysis-file.json"))        # the filing wrote under the root: stamped
    ctx = status.status_json(root)
    assert "analyses" in status.QUERIES and ctx["analyses"][0]["name"] == "study.capture" and ctx["analyses"][0]["n_runs"] == 2 and ctx["analyses"][0]["n_artifacts"] == 5
    assert ctx["analyses"][0]["provenance"] == "relayed" and ctx["analyses"][0]["page_urls"] == "https://example.invalid/figs/study/"
    out_html = str(tmp_path / "index.html")
    status.render(root, out_path=out_html, register=False)
    html = open(out_html, encoding="utf-8").read()
    assert "Analyses (filed over ingested runs and objects)" in html and '<a href="https://example.invalid/figs/study/">' in html and "relayed (by TESTSESSION)" in html
    # a re-filing of the same spec: four more files in a second filing directory of the day, the same analysis id, every row a duplicate
    # at the fold
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec)]) == 0
    text = capsys.readouterr().out
    assert f"analysis_id {aid}" in text and "written 4 record file(s) (filing " in text and "published whole" in text
    dirs = sorted(os.listdir(d))
    assert len(dirs) == 2 and os.path.basename(filing_dir) in dirs and all(len(os.listdir(os.path.join(d, x))) == 4 for x in dirs)
    assert all(os.path.join(d, x, "000001-artifact.jsonl") in text or x == os.path.basename(filing_dir) for x in dirs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.analysis_records == 0 and rep.refused == 0 and rep.duplicates >= 10
    assert _count(root, "analysis") == 1 and _count(root, "analysis_input") == 4 and _count(root, "analysis_artifact") == 5
    assert _rebuild_equal(root, tmp_path) == {"analysis": 1, "analysis_input": 4, "analysis_artifact": 5}


def test_a_remote_report_is_registered_under_its_declared_sha_with_its_location_unverified(tmp_path, capsys):
    root, ids = _root(tmp_path)
    declared = "d" * 64
    spec = _spec(ids, None, report={"uri": "/scratch/example/someone/exp/study/report.json", "sha256": declared, "substrate": "cluster_a"},
                 provenance={"kind": "user_direct"})
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec)]) == 0
    text = capsys.readouterr().out
    assert "(remote: declared sha, unverified)" in text and "000001-artifact.jsonl" in text
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0 and _count(root, "analysis_artifact") == 1
    art = _q(root, "SELECT * FROM artifact WHERE artifact_id = ?", declared)[0]
    assert art["kind"] == "report" and art["uri_primary"] == "/scratch/example/someone/exp/study/report.json" and art["bytes"] is None and art["verified_utc"] is None
    assert json.loads(art["locations_json"]) == [{"uri": "/scratch/example/someone/exp/study/report.json", "verified_utc": None, "bytes": None, "substrate": "cluster_a"}]
    assert json.loads(art["extra_json"]) == {"analysis": "study.capture", "declared_sha256": declared, "role": "report"}      # no `remote` flag, no directive: the null columns say it
    row = _q(root, "SELECT report_uri, report_sha256, extra_json FROM analysis")[0]
    assert row["report_sha256"] == declared and json.loads(row["extra_json"])["provenance"] == {"kind": "user_direct"}
    assert _q(root, "SELECT role, extra_json FROM analysis_artifact") == [{"role": "report", "extra_json": json.dumps({"declared_sha256": declared}, separators=(",", ":"))}]
    ctx = status.status_json(root)
    assert ctx["analyses"][0]["provenance"] == "user_direct" and ctx["analyses"][0]["page_urls"] is None


def test_the_identity_is_name_code_sha_and_inputs_in_any_order_and_nothing_else():
    base = {"name": "n", "code_sha": "c" * 40, "inputs": {"runs": ["r2", "r1"], "objects": ["o1"]}}
    aid = AN.analysis_id(base)
    assert aid == AN.analysis_id(dict(base, inputs={"runs": ["r1", "r2"], "objects": ["o1"]}, notes="x", cluster_job_ids=["9"], artifacts=[{"dir": "/p"}]))
    assert aid != AN.analysis_id(dict(base, code_sha="d" * 40)) and aid != AN.analysis_id(dict(base, inputs={"runs": ["r1"], "objects": ["o1"]}))
    assert aid != AN.analysis_id(dict(base, name="m"))
    assert re.match(r"^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$", aid)


# --------------------------------------------------------------------------------------------
# The dependency-refusal retry, first-arrival by the log's order, hash once
# --------------------------------------------------------------------------------------------


def _new_run(root, key, i0):
    """A fresh judged run of the synthetic campaign, written as the emitter writes it; returns its run_id."""
    rid, recs = synth.run_records(key, i0, "none", "none", 2)
    synth.write_run_dir(root, synth.CAMPAIGN, rid, [(str(r["record_type"]), [r]) for r in recs], gz_items=False)
    return rid


def test_a_dependency_refusal_is_retried_every_cycle_and_accepted_once_the_parent_folds_the_fact_kept(tmp_path):
    root, ids = _root(tmp_path)
    rid_present = ids["run_ids"][0]
    rid_later, _ = synth.run_records("synthstore_late", 900, "none", "none", 2)       # the id of a run NOT yet in the registry
    d = os.path.join(root, "records", "analyses", "2026-01-05")
    header = synth.rec("analysis", _analysis_payload(filed_utc=synth.t(901)), synth.t(901))
    R.write_records(os.path.join(d, "000001-analysis.jsonl"), [header])
    # a MIXED file: one input the registry holds, one it does not
    inputs = [{"analysis_id": AID, "filing_sha256": R.record_sha(header), "input_kind": "run", "input_id": rid_present, "run_id": rid_present, "object_id": None},
              {"analysis_id": AID, "filing_sha256": R.record_sha(header), "input_kind": "run", "input_id": rid_later, "run_id": rid_later, "object_id": None}]
    R.write_records(os.path.join(d, "000002-analysis_input.jsonl"), [synth.rec("analysis_input", p, synth.t(902)) for p in inputs])
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(950))
    assert rep.refused == 1 and rep.analysis_records == 2 and _count(root, "analysis_input") == 1
    refused = _q(root, "SELECT record_sha256, field, reason FROM record_refusal WHERE record_type = 'analysis_input'")
    assert len(refused) == 1 and refused[0]["field"] == "run_id" and H.dependency_refusal_parent(refused[0]["reason"]) == ("run", rid_later)
    sha_refused = refused[0]["record_sha256"]
    assert _q(root, "SELECT COUNT(*) AS n FROM record_refusal_fact WHERE record_sha256 = ?", sha_refused)[0]["n"] == 1
    # cycles while the parent is still absent: re-selected, still refused, the judgment unchanged, no second fact, no churn
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(951))
    assert rep.dependency_refusals_retried == 1 and rep.dependency_refusals_resolved == 0 and rep.refused == 0
    assert _q(root, "SELECT COUNT(*) AS n FROM record_refusal_fact WHERE record_sha256 = ?", sha_refused)[0]["n"] == 1
    # the parent folds (an ordinary producer run directory): the SAME cycle re-selects the refused child and accepts it
    assert _new_run(root, "synthstore_late", 900) == rid_later
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(960))
    assert rep.dependency_refusals_retried == 1 and rep.dependency_refusals_resolved == 1
    assert _count(root, "analysis_input") == 2 and _q(root, "SELECT COUNT(*) AS n FROM record_refusal WHERE record_sha256 = ?", sha_refused)[0]["n"] == 0
    # the historical fact stands; no file was edited (the sha the ledger ranked is the file's)
    assert _q(root, "SELECT COUNT(*) AS n FROM record_refusal_fact WHERE record_sha256 = ?", sha_refused)[0]["n"] == 1
    uri = "records/analyses/2026-01-05/000002-analysis_input.jsonl"
    assert _q(root, "SELECT file_sha256 FROM record_ingest WHERE file_uri = ?", uri)[0]["file_sha256"] == R.file_logical_sha(os.path.join(root, uri))
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(961))
    assert rep.dependency_refusals_retried == 0                                               # nothing left to retry
    assert _rebuild_equal(root, tmp_path) == {"analysis": 1, "analysis_input": 2, "analysis_artifact": 0}


def _filing_pair(root, day, seq0, stamp_i, notes):
    """One `analysis` record of AID under records/analyses/<day>/ with the given stamp and notes."""
    d = os.path.join(root, "records", "analyses", day)
    rec = synth.rec("analysis", _analysis_payload(filed_utc=synth.t(stamp_i), notes=notes), synth.t(stamp_i))
    return R.write_records(os.path.join(d, f"{seq0:06d}-analysis.jsonl"), [rec]), R.record_sha(rec)


def test_the_analysis_header_kept_is_the_smallest_stamp_and_sha_whatever_the_arrival_order_and_a_conflict_is_a_named_refusal(tmp_path):
    root, ids = _root(tmp_path)
    # late delivery: the LATER-stamped filing arrives (and folds) first; the earlier-stamped one a cycle later
    _, sha_late = _filing_pair(root, "2026-01-06", 1, 820, "second filing (later stamp), arrived first")
    H.ingest_once(root, workers=0, min_free=0, now=synth.t(900))
    row = _q(root, "SELECT notes, record_written_utc, record_sha256 FROM analysis")[0]
    assert row["notes"].startswith("second filing") and row["record_written_utc"] == synth.t(820) and row["record_sha256"] == sha_late
    _, sha_early = _filing_pair(root, "2026-01-07", 1, 810, "first filing (earlier stamp), arrived second")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(901))
    # the earlier-stamped filing REPLACES the header (the rebuild's order); the displaced filing is a named refusal fact, not a silent drop
    row = _q(root, "SELECT notes, record_written_utc, record_sha256 FROM analysis")[0]
    assert row["notes"].startswith("first filing") and row["record_written_utc"] == synth.t(810) and row["record_sha256"] == sha_early
    assert rep.analysis_records == 1 and rep.refused == 1 and _count(root, "analysis") == 1
    ref = _q(root, "SELECT record_sha256, field, reason FROM record_refusal WHERE record_type = 'analysis'")
    assert len(ref) == 1 and ref[0]["record_sha256"] == sha_late and ref[0]["field"] == "analysis_id"
    assert f"conflicting re-filing of analysis {AID}" in ref[0]["reason"] and "notes" in ref[0]["reason"]
    # rebuild equality of header VALUES (the rebuild folds the two files in arrival order and keeps the smallest key too)
    assert _rebuild_equal(root, tmp_path, shuffle_seed=4) == {"analysis": 1, "analysis_input": 0, "analysis_artifact": 0}
    # a later-stamped CONFLICTING re-filing after the kept one: refused by name, the header unchanged
    _, sha_conflict = _filing_pair(root, "2026-01-08", 1, 830, "third filing (conflicting notes)")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(902))
    assert rep.refused == 1 and rep.analysis_records == 0
    assert _q(root, "SELECT notes FROM analysis")[0]["notes"].startswith("first filing")
    ref = _q(root, "SELECT reason FROM record_refusal WHERE record_sha256 = ?", sha_conflict)
    assert len(ref) == 1 and f"conflicting re-filing of analysis {AID}" in ref[0]["reason"] and "notes" in ref[0]["reason"]
    # an IDENTICAL re-filing (same payload, a later stamp): a duplicate, nothing moves, no refusal
    d = os.path.join(root, "records", "analyses", "2026-01-09")
    same = synth.rec("analysis", _analysis_payload(filed_utc=synth.t(810), notes="first filing (earlier stamp), arrived second"), synth.t(840))
    R.write_records(os.path.join(d, "000001-analysis.jsonl"), [same])
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(903))
    assert rep.refused == 0 and rep.analysis_records == 0 and rep.duplicates >= 1
    row = _q(root, "SELECT record_written_utc, record_sha256 FROM analysis")[0]
    assert row["record_written_utc"] == synth.t(810) and row["record_sha256"] == sha_early
    assert _rebuild_equal(root, tmp_path) == {"analysis": 1, "analysis_input": 0, "analysis_artifact": 0}
    # an identical-substance filing under an EARLIER stamp (another filed_utc, the filing's own clock, never a conflict): the row becomes
    # that filing's whole -- its key columns and its filed_utc -- as a rebuild meeting it first would write it; no refusal
    d = os.path.join(root, "records", "analyses", "2026-01-10")
    earlier = synth.rec("analysis", _analysis_payload(filed_utc=synth.t(805), notes="first filing (earlier stamp), arrived second"), synth.t(805))
    R.write_records(os.path.join(d, "000001-analysis.jsonl"), [earlier])
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(904))
    assert rep.refused == 0 and rep.analysis_records == 0
    row = _q(root, "SELECT record_written_utc, record_sha256, filed_utc FROM analysis")[0]
    assert row["record_written_utc"] == synth.t(805) and row["record_sha256"] == R.record_sha(earlier) and row["filed_utc"] == synth.t(805)
    assert _rebuild_equal(root, tmp_path, shuffle_seed=5) == {"analysis": 1, "analysis_input": 0, "analysis_artifact": 0}
    # the ordering columns are derived: never required of a record, never a payload column, and the record_refusal fact keeps its semantics
    assert R.LEDGER_ORDERED_TABLES == {t: ("record_written_utc", "record_sha256", "record_uri") for t in KINDS}      # every analysis table (round two)
    for t in KINDS:
        assert "record_written_utc" not in R.REQUIRED_FIELDS[t] and "record_written_utc" not in S.payload_columns(t) and "record_uri" not in S.payload_columns(t)
        assert [c.name for c in S.TABLE_BY_NAME[t].columns if c.derived] == ["record_written_utc", "record_sha256", "record_uri"], t
    assert "record_refusal_fact" not in R.LEDGER_ORDERED_TABLES
    # the record index: a ledger table of the harvester's (fed by no record kind), keyed by table, key and sha; every filing above is in it
    idx = R.LEDGER_ORDERED_RECORD_INDEX
    assert idx == "ordered_fact_record" and idx in S.LEDGER_TABLES and idx not in R.RECORD_TYPE_TABLE.values() and idx not in R.RECORD_TYPES
    assert R.NATURAL_KEYS[idx] == ("table_name", "natural_key_json", "record_sha256") and S.TABLE_BY_NAME[idx].column("table_name").vocab is None
    indexed = _q(root, "SELECT record_sha256, written_utc FROM ordered_fact_record WHERE table_name = 'analysis' AND natural_key_json = ? ORDER BY written_utc", canonical_json({"analysis_id": AID}))
    assert [r["written_utc"] for r in indexed] == [synth.t(805), synth.t(810), synth.t(820), synth.t(830), synth.t(840)]
    assert indexed[0]["record_sha256"] == R.record_sha(earlier) and indexed[1]["record_sha256"] == sha_early and indexed[2]["record_sha256"] == sha_late
    assert indexed[3]["record_sha256"] == sha_conflict and indexed[4]["record_sha256"] == R.record_sha(same)


def _child_filing(root, day, seq0, stamp_i, record_type, payload, filing=None):
    """One record of an analysis child kind (`analysis_input` / `analysis_artifact`) of AID under records/analyses/<day>/ with the given stamp,
    bound to the header ``filing`` (its record sha) when given."""
    d = os.path.join(root, "records", "analyses", day)
    rec = synth.rec(record_type, dict(payload, filing_sha256=filing) if filing else payload, synth.t(stamp_i))
    return R.write_records(os.path.join(d, f"{seq0:06d}-{record_type}.jsonl"), [rec]), R.record_sha(rec)


def _refusals(root, record_type):
    return _q(root, "SELECT record_sha256, field, reason FROM record_refusal WHERE record_type = ? ORDER BY record_sha256", record_type)


def test_a_link_filed_twice_with_another_role_or_page_is_kept_by_the_logs_order_whatever_the_arrival_order_and_the_other_is_refused(tmp_path):
    """`analysis_artifact` (and `analysis_input`) once folded by INSERT OR IGNORE, so two filings of one link
    with another role or page url kept whichever ARRIVED first while a rebuild kept the other -- a wrong role or page url in one database
    and no refusal for the dropped link. Now every analysis table is ordered: the record with the smallest (written_utc, record_sha256) is
    kept, the displaced one refused by name, and the rows, the record index and the refusals are the same sets live and on rebuild."""
    root, ids = _root(tmp_path)
    paths, art_sha = _hand_filing(root, ids)                                             # the artifact, the header, two inputs, the link (stamp 603, role report)
    hsha = _header_sha_of(paths[1])                                                      # every further child here is the kept filing's (bound to its header)
    H.ingest_once(root, workers=0, min_free=0, now=synth.t(700))
    assert _q(root, "SELECT role, extra_json, record_written_utc FROM analysis_artifact") == [{"role": "report", "extra_json": "{}", "record_written_utc": synth.t(603)}]
    # a LATER-stamped link of the same (analysis, artifact) with another role and a page url: refused by name, the kept link unchanged
    late = {"analysis_id": AID, "artifact_id": art_sha, "role": "page", "extra": {"page_url": "https://example.invalid/figs/late/"}}
    _, sha_late = _child_filing(root, "2026-01-06", 1, 650, "analysis_artifact", late, filing=hsha)
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(701))
    assert rep.refused == 1 and rep.analysis_records == 0 and rep.refilings_rejudged == 0
    row = _q(root, "SELECT role, extra_json, record_written_utc, record_sha256 FROM analysis_artifact")[0]
    assert row["role"] == "report" and row["record_written_utc"] == synth.t(603)
    ref = _refusals(root, "analysis_artifact")
    assert len(ref) == 1 and ref[0]["record_sha256"] == sha_late and ref[0]["field"] == "analysis_id"
    assert ref[0]["reason"].startswith(f"conflicting re-filing of analysis_artifact {AID}, {art_sha}: role, extra_json differ from the kept filing {row['record_sha256'][:12]}")
    assert "a file's role or page" in ref[0]["reason"]
    # an EARLIER-stamped link with another role arrives last: it takes the row (the order a rebuild folds in), the displaced link is refused by
    # name, and the standing refusal of the late link is re-worded to name the new kept record
    early = {"analysis_id": AID, "artifact_id": art_sha, "role": "figure", "extra": {"page_url": "https://example.invalid/figs/early/"}}
    _, sha_early = _child_filing(root, "2026-01-07", 1, 590, "analysis_artifact", early, filing=hsha)
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(702))
    assert rep.analysis_records == 1 and rep.refused == 1 and rep.refilings_rejudged == 2                # the displaced link and the late one, judged again
    row = _q(root, "SELECT role, extra_json, record_written_utc, record_sha256, record_uri FROM analysis_artifact")[0]
    assert row["role"] == "figure" and json.loads(row["extra_json"])["page_url"].endswith("/early/") and row["record_written_utc"] == synth.t(590)
    assert row["record_sha256"] == sha_early and row["record_uri"] == "records/analyses/2026-01-07/000001-analysis_artifact.jsonl#1"
    sha_displaced = next(sha for _line, _rec, sha in R.read_record_file(paths[3]))                  # the hand filing's link (stamp 603)
    ref = {r["record_sha256"]: r["reason"] for r in _refusals(root, "analysis_artifact")}
    assert set(ref) == {sha_late, sha_displaced}
    for reason in ref.values():
        assert f"differ from the kept filing {sha_early[:12]}" in reason
    assert _count(root, "analysis_artifact") == 1 and _count(root, "analysis") == 1
    # the record index holds all three links of the key; the rows, the index and the refusals are the same sets after a shuffled rebuild
    key = canonical_json({"analysis_id": AID, "artifact_id": art_sha})
    assert [r["written_utc"] for r in _q(root, "SELECT written_utc FROM ordered_fact_record WHERE table_name = 'analysis_artifact' AND natural_key_json = ? ORDER BY written_utc", key)] == [synth.t(590), synth.t(603), synth.t(650)]
    assert _rebuild_equal(root, tmp_path, shuffle_seed=7) == {"analysis": 1, "analysis_input": 2, "analysis_artifact": 1}
    # the same for an INPUT whose `extra_json` (its role to the analysis) differs: substance, not an event column
    rid = ids["run_ids"][0]
    inp_late = {"analysis_id": AID, "input_kind": "run", "input_id": rid, "run_id": rid, "object_id": None, "extra": {"role": "steered"}}
    _, in_sha_late = _child_filing(root, "2026-01-08", 1, 660, "analysis_input", inp_late, filing=hsha)
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(703))
    assert rep.refused == 1 and _refusals(root, "analysis_input")[0]["record_sha256"] == in_sha_late
    assert f"conflicting re-filing of analysis_input {AID}, run, {rid}: extra_json differ" in _refusals(root, "analysis_input")[0]["reason"]
    inp_early = {"analysis_id": AID, "input_kind": "run", "input_id": rid, "run_id": rid, "object_id": None, "extra": {"role": "unsteered"}}
    _, in_sha_early = _child_filing(root, "2026-01-09", 1, 580, "analysis_input", inp_early, filing=hsha)
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(704))
    assert rep.analysis_records == 1 and rep.refused == 1 and rep.refilings_rejudged == 2
    row = _q(root, "SELECT extra_json, record_written_utc, record_sha256 FROM analysis_input WHERE input_kind = 'run'")[0]
    assert json.loads(row["extra_json"]) == {"role": "unsteered"} and row["record_written_utc"] == synth.t(580) and row["record_sha256"] == in_sha_early
    assert len(_refusals(root, "analysis_input")) == 2 and _count(root, "analysis_input") == 2
    assert _rebuild_equal(root, tmp_path, shuffle_seed=8) == {"analysis": 1, "analysis_input": 2, "analysis_artifact": 1}
    # an identical later link is a duplicate (indexed, never refused); the same record under a SECOND uri keeps the smaller as its carrier,
    # in the row and in the index, whichever arrived first
    _, sha_dup = _child_filing(root, "2026-01-10", 1, 670, "analysis_artifact", early, filing=hsha)
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(705))
    assert rep.refused == 0 and rep.analysis_records == 0 and rep.duplicates >= 1 and len(_refusals(root, "analysis_artifact")) == 2
    twin_dir = os.path.join(root, "records", "analyses", "2026-01-05")
    os.makedirs(twin_dir, exist_ok=True)
    shutil.copy2(os.path.join(root, "records", "analyses", "2026-01-07", "000001-analysis_artifact.jsonl"), os.path.join(twin_dir, "000001-analysis_artifact.jsonl"))
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(706))
    assert rep.refused == 0 and rep.analysis_records == 0
    assert _q(root, "SELECT record_uri FROM analysis_artifact")[0]["record_uri"] == "records/analyses/2026-01-05/000001-analysis_artifact.jsonl#1"
    assert _q(root, "SELECT record_uri FROM ordered_fact_record WHERE record_sha256 = ?", sha_early)[0]["record_uri"] == "records/analyses/2026-01-05/000001-analysis_artifact.jsonl#1"
    assert _q(root, "SELECT COUNT(*) AS n FROM ordered_fact_record WHERE table_name = 'analysis_artifact'")[0]["n"] == 4
    assert _rebuild_equal(root, tmp_path, shuffle_seed=9) == {"analysis": 1, "analysis_input": 2, "analysis_artifact": 1}


def test_a_duplicate_of_a_displaced_filing_is_refused_when_an_earlier_filing_arrives_as_a_rebuild_refuses_it(tmp_path):
    """B folds, C (a later stamp, B's substance) is a duplicate of it and leaves no refusal row; then A, an
    earlier-stamped filing with another header, arrives and replaces B. The first rendering judged only the STANDING conflict refusals
    again, so B was refused and C stayed accepted, while a rebuild -- A first -- refused both: one record log, two refusal tables. Now every
    record of the key the fold applied (the `ordered_fact_record` index) is judged again against A: C is refused by name, with the reason a
    rebuild writes for it."""
    root, ids = _root(tmp_path)
    _, sha_b = _filing_pair(root, "2026-01-06", 1, 820, "B: the first filing to arrive")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(900))
    assert rep.analysis_records == 1 and rep.refused == 0
    d = os.path.join(root, "records", "analyses", "2026-01-07")
    c = synth.rec("analysis", _analysis_payload(filed_utc=synth.t(830), notes="B: the first filing to arrive"), synth.t(830))      # B's substance, a later stamp
    R.write_records(os.path.join(d, "000001-analysis.jsonl"), [c])
    sha_c = R.record_sha(c)
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(901))
    assert rep.refused == 0 and rep.analysis_records == 0 and rep.duplicates >= 1 and _count(root, "record_refusal") == 0
    assert _q(root, "SELECT COUNT(*) AS n FROM ordered_fact_record WHERE table_name = 'analysis'")[0]["n"] == 2                  # B and C, the duplicate too
    # A: an earlier stamp, another header. It takes the row; B is displaced and refused; C -- a duplicate of B, not of A -- is refused too
    _, sha_a = _filing_pair(root, "2026-01-08", 1, 810, "A: the earliest filing, arrived last")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(902))
    assert rep.analysis_records == 1 and rep.refused == 2 and rep.refilings_rejudged == 2
    row = _q(root, "SELECT notes, record_written_utc, record_sha256 FROM analysis")[0]
    assert row["notes"].startswith("A:") and row["record_written_utc"] == synth.t(810) and row["record_sha256"] == sha_a
    ref = {r["record_sha256"]: r for r in _refusals(root, "analysis")}
    assert set(ref) == {sha_b, sha_c}
    for sha in (sha_b, sha_c):
        assert ref[sha]["field"] == "analysis_id" and ref[sha]["reason"].startswith(f"conflicting re-filing of analysis {AID}: notes differ from the kept filing {sha_a[:12]}")
    assert ref[sha_c]["reason"] == ref[sha_b]["reason"]
    assert _q(root, "SELECT COUNT(*) AS n FROM record_refusal_fact WHERE record_sha256 IN (?, ?)", sha_b, sha_c)[0]["n"] == 2       # each judgment a fact
    assert _q(root, "SELECT COUNT(*) AS n FROM ordered_fact_record WHERE table_name = 'analysis'")[0]["n"] == 3
    # the refusal table (reason, field, type, uri alike), the header and the index equal a shuffled rebuild's -- which meets A first
    assert _rebuild_equal(root, tmp_path, shuffle_seed=11) == {"analysis": 1, "analysis_input": 0, "analysis_artifact": 0}
    # a cycle with nothing new judges nothing again and refuses nothing
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(903))
    assert rep.refused == 0 and rep.refilings_rejudged == 0 and set(r["record_sha256"] for r in _refusals(root, "analysis")) == {sha_b, sha_c}
    # a filing of A's substance under a stamp between A and B: a duplicate of the kept filing, indexed, never refused; the refusals stand
    d = os.path.join(root, "records", "analyses", "2026-01-09")
    a2 = synth.rec("analysis", _analysis_payload(filed_utc=synth.t(815), notes="A: the earliest filing, arrived last"), synth.t(815))
    R.write_records(os.path.join(d, "000001-analysis.jsonl"), [a2])
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(904))
    assert rep.refused == 0 and rep.duplicates >= 1 and _q(root, "SELECT COUNT(*) AS n FROM ordered_fact_record WHERE table_name = 'analysis'")[0]["n"] == 4
    assert _rebuild_equal(root, tmp_path, shuffle_seed=12) == {"analysis": 1, "analysis_input": 0, "analysis_artifact": 0}


def test_a_local_artifact_is_hashed_once_and_a_file_that_changes_between_capture_and_write_refuses_the_filing(tmp_path, monkeypatch):
    root, ids = _root(tmp_path)
    page = _page_dir(tmp_path)
    report = tmp_path / "report.json"
    report.write_text('{"delta": 0.1}', encoding="utf-8")
    spec = _spec(ids, page, report={"uri": str(report), "sha256": None})
    now = R.utc_now()
    # one read per file: the digest the artifact record carries IS the digest the link carries
    reads = []
    real = PUB.sha256_of
    monkeypatch.setattr(PUB, "sha256_of", lambda p: (reads.append(p), real(p))[1])
    with DB.reading(DB.db_path(root)) as conn:
        plan = AN.build_plan(spec, now, conn)
    assert sorted(reads) == sorted(set(reads)) and len(reads) == 5                            # five files, each hashed exactly once
    for p in plan.artifacts:
        assert p.payload["artifact_id"] == p.sha256 and p.payload["bytes"] == p.size
    assert {l["artifact_id"] for l in plan.link_payloads} == {p.payload["artifact_id"] for p in plan.artifacts}
    # the file changes between the plan (capture) and the write: refused by name before anything is emitted
    (tmp_path / "page" / "numbers.json").write_text('{"n": 2}', encoding="utf-8")
    with pytest.raises(AN.AnalysisRefusal) as ei:
        AN.write_filing(root, plan, AN.producer_block(), now)
    assert "numbers.json" in str(ei.value) and "changed" in str(ei.value)
    _no_filing_written(root)
    # a file that changes WHILE being hashed (size or mtime moved around the read): refused at capture, by name
    monkeypatch.setattr(PUB, "sha256_of", lambda p: (open(p, "ab").write(b" grew"), real(p))[1])
    with DB.reading(DB.db_path(root)) as conn:
        with pytest.raises(AN.AnalysisRefusal) as ei:
            AN.build_plan(spec, now, conn)
    assert "changed while" in str(ei.value)
    _no_filing_written(root)
    # the publish builder takes a precomputed digest and size (defaults unchanged for every other caller)
    monkeypatch.setattr(PUB, "sha256_of", real)
    sha = real(str(report))
    assert PUB.artifact_payload(str(report), None, now, sha=sha, size=14)["artifact_id"] == sha and PUB.artifact_payload(str(report), None, now)["artifact_id"] == sha


# --------------------------------------------------------------------------------------------
# An input whose run rides an adapter revision directory survives a rebuild
# --------------------------------------------------------------------------------------------


def _revision_run(root, key, i0, rev):
    """A run that rides an ADAPTER REVISION DIRECTORY (records/adapters/<adapter>/<rev16>/): its run record and the accepted marker, the
    marker last -- the shape the harvester's revision phase applies whole, AFTER the plain files on every path."""
    rid, recs = synth.run_records(key, i0, "none", "none", 2)
    run_rec = next(r for r in recs if r["record_type"] == "run")
    d = os.path.join(root, "records", "adapters", "synthetic_ad", rev)
    R.write_records(os.path.join(d, "000001-run.jsonl"), [run_rec])
    marker = synth.rec("source_revision", {"revision_id": rev, "source_id": "synthetic_source", "adapter": "synthetic_ad", "observed_utc": synth.t(i0),
                                           "inputs_json": {"synthetic:///revision/a": "b" * 64}, "accepted": 1, "priority": 0}, synth.t(i0 + 9))
    R.write_records(os.path.join(d, "000002-source_revision.jsonl"), [marker])
    return rid


def test_an_input_whose_run_rides_an_adapter_revision_directory_folds_on_rebuild_exactly_as_live(tmp_path):
    root, ids = _root(tmp_path)
    rev = "a1b2c3d4e5f60718"
    rid = _revision_run(root, "synthstore_rev", 700, rev)
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(800))
    assert rep.refused == 0 and _q(root, "SELECT COUNT(*) AS n FROM run WHERE run_id = ?", rid)[0]["n"] == 1
    # a filing naming the adapter-backed run (the CLI's shape, written by hand), folded live: no refusal, the link present
    _hand_filing(root, ids, day="2026-01-10", run_id=rid)
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(801))
    assert rep.refused == 0 and rep.analysis_records == 4 and _count(root, "analysis_input") == 2 and _count(root, "record_refusal") == 0
    # a full rebuild from the shuffled log equals the live database: rows AND refusals -- the analyses files fold after the revision phase
    assert _rebuild_equal(root, tmp_path) == {"analysis": 1, "analysis_input": 2, "analysis_artifact": 1}
    assert _rebuild_equal(root, tmp_path, shuffle_seed=11) == {"analysis": 1, "analysis_input": 2, "analysis_artifact": 1}


# --------------------------------------------------------------------------------------------
# a known digest is linked, its standing row merged, never replaced
# --------------------------------------------------------------------------------------------


def _artifact_row(root, sha):
    row = _q(root, "SELECT * FROM artifact WHERE artifact_id = ?", sha)[0]
    row["locations"] = json.loads(row["locations_json"])
    row["extra"] = json.loads(row["extra_json"])
    return row


def test_the_merged_artifact_record_keeps_the_standing_identity_names_this_filings_one_location_and_nothing_else():
    now = synth.t(900)
    existing = {"kind": "report", "uri_primary": "/a/r.json", "locations_json": json.dumps([{"uri": "/a/r.json", "verified_utc": synth.t(800), "bytes": 3}]),
                "bytes": 3, "created_utc": synth.t(800), "verified_utc": synth.t(800), "extra_json": json.dumps({"notebook_commit": "n" * 40})}
    remote = PUB.remote_artifact_payload("/c/r.json", SHA_A, now, kind="report", substrate="cluster_a", extra={"analysis": "s", "role": "report"})
    # a remote record states no verification column at all (omitted, never null), no `remote` flag and no fold directive (the union is
    # the fold's rule for every artifact record, the fourth closing pass)
    assert not ({"bytes", "verified_utc", "created_utc"} & set(remote)) and remote["extra"] == {"declared_sha256": SHA_A, "analysis": "s", "role": "report"}
    m = PUB.merged_artifact_payload(existing, remote)
    # the standing kind and primary, THIS filing's one location (the fold adds it to the standing list); NO verification column
    assert m == {"artifact_id": SHA_A, "kind": "report", "uri_primary": "/a/r.json",
                 "locations_json": [{"uri": "/c/r.json", "verified_utc": None, "bytes": None, "substrate": "cluster_a"}],
                 "extra": {"declared_sha256": SHA_A, "analysis": "s", "role": "report"}}
    # the same remote uri again over a row that holds it: nothing to add, no record
    existing2 = dict(existing, locations_json=json.dumps(json.loads(existing["locations_json"]) + m["locations_json"]))
    assert PUB.merged_artifact_payload(existing2, remote) is None
    # a local read of the standing uri: verified again -- its entry restated (the fold replaces the standing entry of the uri), bytes and
    # verified_utc stated, created_utc never
    local = PUB.artifact_payload("/a/r.json", None, now, kind="report", sha=SHA_A, size=3)
    assert local["extra"] == {}
    local["extra"] = {**local["extra"], "analysis": "s"}
    m2 = PUB.merged_artifact_payload(existing2, local)
    assert m2["locations_json"] == [{"uri": "/a/r.json", "verified_utc": now, "bytes": 3}]
    assert m2["bytes"] == 3 and m2["verified_utc"] == now and "created_utc" not in m2 and m2["extra"] == {"analysis": "s"} and m2["uri_primary"] == "/a/r.json"
    # a standing UNVERIFIED row: a second remote uri adds its location and states nothing else; a local read of another uri fills the
    # verification; the primary stands either way; no `remote` flag is ever restated from the standing row
    unv = {"kind": "report", "uri_primary": "/c/r.json", "locations_json": json.dumps([{"uri": "/c/r.json", "verified_utc": None, "bytes": None}]),
           "bytes": None, "created_utc": None, "verified_utc": None, "extra_json": json.dumps({"declared_sha256": SHA_A})}
    m3 = PUB.merged_artifact_payload(unv, PUB.remote_artifact_payload("/d/r.json", SHA_A, now, kind="report"))
    assert m3["extra"] == {"declared_sha256": SHA_A} and m3["uri_primary"] == "/c/r.json" and not ({"bytes", "verified_utc", "created_utc"} & set(m3))
    assert m3["locations_json"] == [{"uri": "/d/r.json", "verified_utc": None, "bytes": None}]
    m4 = PUB.merged_artifact_payload(unv, local)
    assert m4["extra"] == {"analysis": "s"} and m4["uri_primary"] == "/c/r.json" and m4["verified_utc"] == now and m4["bytes"] == 3
    assert m4["locations_json"] == [{"uri": "/a/r.json", "verified_utc": now, "bytes": 3}]
    # an unverified entry for a uri the row holds VERIFIED adds nothing: no record
    assert PUB.merged_artifact_payload(existing, PUB.remote_artifact_payload("/a/r.json", SHA_A, now, kind="report")) is None


def test_a_known_digest_filed_as_a_remote_report_keeps_the_standing_rows_verification_and_locations_and_adds_its_uri(tmp_path, capsys):
    root, ids = _root(tmp_path)
    report = tmp_path / "report.json"
    report.write_text('{"delta": 0.1}', encoding="utf-8")
    sha = PUB.sha256_of(str(report))
    # a verified local registration stands (the publish path, as any notebook's figure or report)
    PUB.register_files(root, [str(report)], notebook_commit="n" * 40, now=synth.t(700), ingest=False)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0
    before = _artifact_row(root, sha)
    assert before["verified_utc"] == synth.t(700) and before["bytes"] == 14 and before["created_utc"] == synth.t(700) and before["uri_primary"] == str(report)
    assert before["locations"] == [{"uri": str(report), "verified_utc": synth.t(700), "bytes": 14}] and before["extra"] == {"notebook_commit": "n" * 40}
    # an analysis files the same bytes as a REMOTE report (its cluster copy), declaring the sha: the plan marks it known
    remote_uri = "/scratch/example/someone/exp/study/report.json"
    spec = _spec(ids, None, report={"uri": remote_uri, "sha256": sha, "substrate": "cluster_a"}, provenance={"kind": "user_direct"})
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "artifacts: 1 file(s) to register (1 already registered: linked, the standing row merged, never replaced)" in out
    assert "(remote: declared sha, unverified) [already registered: linked; this uri added to its locations, its verification kept]" in out
    assert "records: 1 artifact file (1 record(s); 1 already registered), 1 analysis, 4 analysis_input, 1 analysis_artifact" in out
    _no_filing_written(root)
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec), "--json"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert js["artifacts"] == [{"role": "report", "kind": "report", "artifact_id": sha, "uri": remote_uri, "remote": True, "known": True, "record_written": True, "same_bytes_as": None}]
    assert len(js["written"]) == 4 and os.path.basename(js["written"][0]) == "000001-artifact.jsonl"
    (_, art_full, _), = R.read_record_file(js["written"][0])
    art_rec = art_full["payload"]
    # the record keeps the standing kind and primary and names THIS filing's one location under the union directive; no verification
    # column stated (the fold keeps them); the fold adds the location to the standing list
    remote_loc = {"uri": remote_uri, "verified_utc": None, "bytes": None, "substrate": "cluster_a"}
    assert art_rec["kind"] == "report" and art_rec["uri_primary"] == str(report)
    assert art_rec["locations_json"] == [remote_loc]
    assert not ({"bytes", "verified_utc", "created_utc"} & set(art_rec))
    assert art_rec["extra"] == {"analysis": "study.capture", "role": "report", "declared_sha256": sha}
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0 and _count(root, "analysis_artifact") == 1 and rep.analysis_records == 1 + 4 + 1
    after = _artifact_row(root, sha)
    # the verification, its stamp, the created time, the kind and the primary stand; the locations gained the remote uri; extra accumulated
    for col in ("verified_utc", "bytes", "created_utc", "uri_primary", "kind"):
        assert after[col] == before[col], col
    assert after["locations"] == [{"uri": str(report), "verified_utc": synth.t(700), "bytes": 14}, remote_loc]
    assert after["extra"] == {"notebook_commit": "n" * 40, "analysis": "study.capture", "role": "report", "declared_sha256": sha}   # the directive never lands
    fv = json.loads(after["field_versions_json"])
    assert fv["verified_utc"] == json.loads(before["field_versions_json"])["verified_utc"] and fv["locations_json"] == [art_full["written_utc"], fv["locations_json"][1]]
    assert fv["locations_json"] != fv["verified_utc"]
    assert _q(root, "SELECT role, extra_json FROM analysis_artifact") == [{"role": "report", "extra_json": json.dumps({"declared_sha256": sha}, separators=(",", ":"))}]
    assert _q(root, "SELECT report_uri, report_sha256 FROM analysis") == [{"report_uri": remote_uri, "report_sha256": sha}]
    # the same filing again: the uri is a location already and nothing is verified -- NO artifact record; the link names the standing row
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "[already registered at this uri: linked, no record]" in out and "records: 0 artifact file (0 record(s); 1 already registered), 1 analysis" in out
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec), "--json"]) == 0
    js2 = json.loads(capsys.readouterr().out)
    assert js2["artifacts"][0]["known"] is True and js2["artifacts"][0]["record_written"] is False
    assert [os.path.basename(p).split("-", 1)[1] for p in js2["written"]] == ["analysis.jsonl", "analysis_input.jsonl", "analysis_artifact.jsonl"]
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.analysis_records == 0 and rep.duplicates >= 6
    assert _artifact_row(root, sha) == after
    assert _rebuild_equal(root, tmp_path) == {"analysis": 1, "analysis_input": 4, "analysis_artifact": 1}


def test_a_known_remote_digest_filed_from_a_local_copy_gains_the_verification_keeps_its_primary_and_is_verified_again_on_a_refiling(tmp_path, capsys):
    root, ids = _root(tmp_path)
    report = tmp_path / "report.json"
    report.write_text('{"delta": 0.1}', encoding="utf-8")
    sha = PUB.sha256_of(str(report))
    remote_uri = "/scratch/example/someone/exp/study/report.json"
    spec = _spec(ids, None, report={"uri": remote_uri, "sha256": sha, "substrate": "cluster_a"}, provenance={"kind": "user_direct"})
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec)]) == 0
    capsys.readouterr()
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    before = _artifact_row(root, sha)
    assert before["verified_utc"] is None and before["bytes"] is None and before["created_utc"] is None and "remote" not in before["extra"]
    # another analysis (another name: another id) files the LOCAL copy of the same bytes beside a page directory of new files
    page = _page_dir(tmp_path)
    spec2 = _spec(ids, page, report={"uri": str(report), "sha256": None}, name="study.local")
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec2, "s2.json"), "--json"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert [a for a in js["artifacts"] if a["known"]] == [{"role": "report", "kind": "report", "artifact_id": sha, "uri": str(report), "remote": False, "known": True, "record_written": True, "same_bytes_as": None}]
    assert sum(1 for a in js["artifacts"] if not a["known"] and a["record_written"]) == 4           # the page's four files are new registrations
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0 and rep.analysis_records == 1 + 4 + 5
    after = _artifact_row(root, sha)
    assert after["uri_primary"] == remote_uri and after["kind"] == "report"                        # a filing never moves another registration's primary
    assert after["bytes"] == 14 and after["verified_utc"] is not None and after["created_utc"] is None   # verified by the local read; created_utc the standing null
    assert after["locations"] == [{"uri": remote_uri, "verified_utc": None, "bytes": None, "substrate": "cluster_a"}, {"uri": str(report), "verified_utc": after["verified_utc"], "bytes": 14}]
    assert "remote" not in after["extra"] and "union_columns" not in after["extra"] and after["extra"]["analysis"] == "study.local" and after["extra"]["declared_sha256"] == sha
    assert _count(root, "analysis") == 2 and _count(root, "analysis_artifact") == 1 + 5
    assert _q(root, "SELECT COUNT(*) AS n FROM analysis_artifact WHERE artifact_id = ?", sha)[0]["n"] == 2       # both analyses link the one row
    # a re-filing of the local analysis reads every file again: each known entry is verified anew (replaced in place), nothing else moves
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec2, "s2.json"), "--json"]) == 0
    js2 = json.loads(capsys.readouterr().out)
    assert all(a["known"] and a["record_written"] for a in js2["artifacts"]) and len(js2["artifacts"]) == 5
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.analysis_records == 0
    again = _artifact_row(root, sha)
    assert again["verified_utc"] >= after["verified_utc"] and again["uri_primary"] == remote_uri and again["created_utc"] is None and again["bytes"] == 14
    assert [l["uri"] for l in again["locations"]] == [remote_uri, str(report)] and again["locations"][1]["verified_utc"] == again["verified_utc"]
    assert again["locations"][0] == before["locations"][0] and again["extra"] == after["extra"]
    assert _rebuild_equal(root, tmp_path) == {"analysis": 2, "analysis_input": 8, "analysis_artifact": 6}


# --------------------------------------------------------------------------------------------
# 10: the union directive (review round four)
# --------------------------------------------------------------------------------------------


def _validate(payload, stamp=700):
    rec = synth.rec("artifact", payload, synth.t(stamp))
    return VAL.validate_record(rec, "<memory>", R.record_sha(rec))


def test_the_union_directive_adds_a_records_locations_to_the_standing_list_and_the_validator_admits_it_on_artifact_alone():
    # one spelling of the marker and the tables with a union column, in the reducer and in the validator (which the reducer imports); the
    # publish writers carry no marker since the fourth closing pass (the union is the fold's rule for every artifact record)
    assert RED.UNION_COLUMNS_KEY == VAL.UNION_COLUMNS_KEY == "union_columns" and RED.UNION_COLUMNS == VAL.UNION_COLUMNS == {"artifact": ("locations_json",)}
    assert not hasattr(PUB, "LOCATIONS_UNION_EXTRA") and "union_columns" not in PUB.artifact_payload("/c/r.json", None, synth.t(1), sha=SHA_A, size=3)["extra"]
    assert RED.UNION_COLUMNS_KEY not in S.payload_columns("artifact")                 # a directive, never a column
    v_a, u_a = {"uri": "/a", "verified_utc": synth.t(1), "bytes": 3}, {"uri": "/a", "verified_utc": None, "bytes": None}
    v_a2, u_b, v_c = {"uri": "/a", "verified_utc": synth.t(2), "bytes": 3}, {"uri": "/b", "verified_utc": None, "bytes": None, "substrate": "cluster_a"}, {"uri": "/c", "verified_utc": synth.t(3), "bytes": 3}
    # a new key is appended in the incoming order; an unverified entry never replaces a standing one (verified or not); a verified one does
    assert RED.union_entries([v_a], [u_b, v_c]) == [v_a, u_b, v_c]
    assert RED.union_entries([v_a], [u_a]) == [v_a] and RED.union_entries([u_b], [{"uri": "/b", "verified_utc": None, "bytes": 9}]) == [u_b]
    assert RED.union_entries([v_a, u_b], [v_a2, {"uri": "/b", "verified_utc": synth.t(4), "bytes": 5}]) == [v_a2, {"uri": "/b", "verified_utc": synth.t(4), "bytes": 5}]
    assert RED.union_entries([v_a, u_b], [v_a, u_b]) == [v_a, u_b] and RED.union_entries([], [u_b]) == [u_b]                  # idempotent; from nothing
    # a bare path (the hooks' artifact records) is its own key; a bare path and a location of that uri are one key
    assert RED.union_entries(["/x/results.jsonl"], ["/x/results.jsonl", "/y"]) == ["/x/results.jsonl", "/y"]
    assert RED.union_entries(["/x"], [{"uri": "/x", "verified_utc": synth.t(1), "bytes": 1}]) == [{"uri": "/x", "verified_utc": synth.t(1), "bytes": 1}]
    assert RED.location_key({"note": "n"}) == canonical_json({"note": "n"})
    # the column form: canonical JSON text in, canonical JSON text out, byte-equal to the standing text when nothing is added
    standing = canonical_json([v_a, u_b])
    assert RED.union_json_list(standing, canonical_json([u_a])) == standing and RED.union_json_list(None, canonical_json([u_b])) == canonical_json([u_b])
    assert RED.union_json_list("[]", canonical_json([v_c])) == canonical_json([v_c]) and RED.union_json_list("not json", canonical_json([v_c])) == canonical_json([v_c])
    # the payload form (fold() and the appended-fold shortcut share it): every union column the payload carries, marker or none, folds over
    # the standing value; a null over a standing list leaves the payload (no statement), over no standing value stays null; another table's
    # payload is untouched
    assert RED.union_payload("artifact", {"locations_json": canonical_json([u_a]), "bytes": 3}, {"locations_json": standing}) == {"locations_json": standing, "bytes": 3}
    assert RED.union_payload("artifact", {"locations_json": canonical_json([v_c])}, {}) == {"locations_json": canonical_json([v_c])}
    assert RED.union_payload("artifact", {"locations_json": None, "bytes": 3}, {"locations_json": standing}) == {"bytes": 3}
    assert RED.union_payload("artifact", {"locations_json": None}, {}) == {"locations_json": None} and RED.union_payload("artifact", {"bytes": 3}, {"locations_json": standing}) == {"bytes": 3}
    assert RED.union_payload("run", {"locations_json": canonical_json([u_a])}, {"locations_json": standing}) == {"locations_json": canonical_json([u_a])}
    # the validator: a well-formed marker on an artifact record passes; every malformation is refused naming `extra.union_columns`
    base = {"artifact_id": SHA_A, "kind": "report", "uri_primary": "/c/r.json", "locations_json": [u_b], "extra": {"union_columns": ["locations_json"]}}
    v = _validate(base)
    assert isinstance(v, VAL.Validated) and v.extra == {"union_columns": ["locations_json"]} and json.loads(v.payload["locations_json"]) == [u_b]
    cases = [
        (dict(base, extra={"union_columns": ["bytes"]}), "is not a list column of artifact the fold unions"),
        (dict(base, extra={"union_columns": []}), "non-empty list"),
        (dict(base, extra={"union_columns": "locations_json"}), "non-empty list"),
        (dict(base, extra={"union_columns": ["locations_json", "locations_json"]}), "repeats a column"),
        ({k: val for k, val in base.items() if k != "locations_json"}, "is not carried in the payload"),
    ]
    for payload, reason in cases:
        m = _validate(payload)
        assert isinstance(m, VAL.Malformed) and m.field == "extra.union_columns" and reason in m.reason, (payload, getattr(m, "reason", m))
    # the column's own shape is the COLUMN's refusal, marker or none (the seventh closing pass): an object where the list belongs names
    # `locations_json`, not the marker
    m = _validate(dict(base, locations_json={"uri": "/c/r.json"}))
    assert isinstance(m, VAL.Malformed) and m.field == "locations_json" and "must carry a JSON list" in m.reason
    # on another table the marker is refused by name (the fold honors it on `artifact` alone)
    rec = synth.rec("analysis", _analysis_payload(extra={"union_columns": ["cluster_job_ids_json"], "provenance": {"kind": "user_direct"}, "decision": "D357"}), synth.t(700))
    m = VAL.validate_record(rec, "<memory>", R.record_sha(rec))
    assert isinstance(m, VAL.Malformed) and m.field == "extra.union_columns" and "admitted on artifact records only" in m.reason


def _remote_spec(ids, name, uri, sha):
    return _spec(ids, None, report={"uri": uri, "sha256": sha, "substrate": "cluster_a"}, name=name, provenance={"kind": "user_direct"})


def test_two_filings_of_one_known_digest_before_a_fold_keep_every_location_and_the_row_rebuilds_equal_on_both_fold_paths(tmp_path, capsys, monkeypatch):
    """Two analyses filed the same known digest from two cluster paths before the harvester ran; each record
    was merged by the filer against the same database row ("the standing locations plus mine"), so the second to fold replaced the first's
    path, and a rebuild replayed the same payloads to the same loss. Now a record names its one location under the fold's union directive
    and the fold adds it to the standing list, so every location folds in whatever the order."""
    root, ids = _root(tmp_path)
    report = tmp_path / "report.json"
    report.write_text('{"delta": 0.1}', encoding="utf-8")
    sha = PUB.sha256_of(str(report))
    PUB.register_files(root, [str(report)], notebook_commit="n" * 40, now=synth.t(700), ingest=False)
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    before = _artifact_row(root, sha)
    assert before["extra"] == {"notebook_commit": "n" * 40}                                   # the directive on the record never lands in the row
    uri1, uri2 = "/scratch/example/someone/exp/study/report-1.json", "/scratch/example/someone/exp/study/report-2.json"
    written = []
    for name, uri in (("study.one", uri1), ("study.two", uri2)):
        assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, _remote_spec(ids, name, uri, sha), f"{name}.json"), "--json"]) == 0
        js = json.loads(capsys.readouterr().out)
        assert js["artifacts"][0]["known"] is True and js["artifacts"][0]["record_written"] is True
        written.append(js["written"][0])
    # each record names ITS location alone, under the directive, and restates the standing kind and primary; neither knows of the other
    for path, uri in zip(written, (uri1, uri2)):
        (_, full, _), = R.read_record_file(path)
        rec = full["payload"]
        assert rec["locations_json"] == [{"uri": uri, "verified_utc": None, "bytes": None, "substrate": "cluster_a"}] and rec["uri_primary"] == str(report)
        assert "union_columns" not in rec["extra"] and "remote" not in rec["extra"] and not ({"bytes", "verified_utc", "created_utc"} & set(rec))
    # ONE fold over both: the row holds the local location and both cluster paths; the verification, the primary and the kind stand
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0 and rep.analysis_records == 2 * (1 + 4 + 1)
    after = _artifact_row(root, sha)
    assert [l["uri"] for l in after["locations"]] == [str(report), uri1, uri2]
    assert after["locations"][0] == before["locations"][0]
    for col in ("verified_utc", "bytes", "created_utc", "uri_primary", "kind"):
        assert after[col] == before[col], col
    assert after["extra"] == {"notebook_commit": "n" * 40, "analysis": "study.two", "role": "report", "declared_sha256": sha}
    assert _q(root, "SELECT COUNT(*) AS n FROM analysis_artifact WHERE artifact_id = ?", sha)[0]["n"] == 2 and _count(root, "analysis") == 2
    assert {r["report_uri"] for r in _q(root, "SELECT report_uri FROM analysis")} == {uri1, uri2}
    # the history holds three events of the key, each snapshot the union so far
    hist = H.history(root, DB.connect(DB.db_path(root), readonly=True), "artifact", {"artifact_id": sha})
    assert [[l["uri"] for l in json.loads(h["state"]["locations_json"])] for h in hist] == [[str(report)], [str(report), uri1], [str(report), uri1, uri2]]
    assert all("union_columns" not in json.loads(h["state"].get("extra_json", "{}")) for h in hist)
    # equal after shuffled rebuilds, with the appended-fold shortcut and without it
    assert _rebuild_equal(root, tmp_path, shuffle_seed=21) == {"analysis": 2, "analysis_input": 8, "analysis_artifact": 2}
    monkeypatch.setattr(RED, "INCREMENTAL_FOLD", False)
    assert _rebuild_equal(root, tmp_path, shuffle_seed=22) == {"analysis": 2, "analysis_input": 8, "analysis_artifact": 2}
    monkeypatch.setattr(RED, "INCREMENTAL_FOLD", True)
    # a third filing of a path the row holds adds nothing: no artifact record, the row byte-identical
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, _remote_spec(ids, "study.three", uri1, sha), "three.json"), "--json"]) == 0
    js = json.loads(capsys.readouterr().out)
    assert js["artifacts"][0]["record_written"] is False and len(js["written"]) == 3
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0 and _artifact_row(root, sha) == after


def test_a_fresh_digest_filed_remote_and_local_in_one_window_keeps_the_verification_in_either_order_and_a_notebook_rereads_keep_the_cluster_uri(tmp_path, capsys):
    """The same window over a digest NOBODY holds yet: a remote declaration and a local read of one file, both full records before a fold.
    The remote record omits the verification columns (never states them null) and both carry the union directive, so the verification and
    both locations fold in whichever order the two were written; a later notebook re-registration (`publish.register_files`) of the local
    file keeps the cluster uri too."""
    for order in ("remote_first", "local_first"):
        root, ids = _root(tmp_path, f"reg_{order}")
        local = tmp_path / f"{order}.json"
        local.write_text(json.dumps({"order": order}), encoding="utf-8")
        sha = PUB.sha256_of(str(local))
        remote_uri = f"/scratch/example/someone/exp/study/{order}.json"
        specs = [("study.remote", _remote_spec(ids, "study.remote", remote_uri, sha)),
                 ("study.local", _spec(ids, None, report={"uri": str(local), "sha256": None}, name="study.local", provenance={"kind": "user_direct"}))]
        if order == "local_first":
            specs.reverse()
        for name, spec in specs:
            assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec, f"{order}-{name}.json"), "--json"]) == 0
            js = json.loads(capsys.readouterr().out)
            assert js["artifacts"][0]["known"] is False and js["artifacts"][0]["record_written"] is True      # neither sees the other: both fresh
        rep = H.ingest_once(root, workers=0, min_free=0)
        assert rep.refused == 0 and rep.blocked == 0 and rep.analysis_records == 2 * (1 + 4 + 1)
        row = _artifact_row(root, sha)
        # both locations, the local one verified; bytes / verified_utc / created_utc from the local read whichever record folded last
        assert sorted(l["uri"] for l in row["locations"]) == sorted([remote_uri, str(local)])
        local_loc = next(l for l in row["locations"] if l["uri"] == str(local))
        assert row["bytes"] == local_loc["bytes"] == len(local.read_bytes()) and row["verified_utc"] == local_loc["verified_utc"] is not None
        assert row["created_utc"] is not None and row["kind"] == "report" and row["uri_primary"] in (remote_uri, str(local))
        assert "remote" not in row["extra"] and "union_columns" not in row["extra"] and row["extra"]["declared_sha256"] == sha
        assert _q(root, "SELECT COUNT(*) AS n FROM analysis_artifact WHERE artifact_id = ?", sha)[0]["n"] == 2
        assert _rebuild_equal(root, tmp_path, shuffle_seed=31 if order == "remote_first" else 32) == {"analysis": 2, "analysis_input": 8, "analysis_artifact": 2}
        # a notebook registers the local file again (the publish path): its record carries the directive too, so the cluster uri stays and
        # the local entry is verified anew
        later = R.utc_now()                                                                          # after the filings in LOG order
        PUB.register_files(root, [str(local)], notebook_commit="m" * 40, now=later, ingest=False)
        assert H.ingest_once(root, workers=0, min_free=0).refused == 0
        again = _artifact_row(root, sha)
        assert sorted(l["uri"] for l in again["locations"]) == sorted([remote_uri, str(local)])
        assert next(l for l in again["locations"] if l["uri"] == str(local))["verified_utc"] == later == again["verified_utc"]
        assert again["extra"]["notebook_commit"] == "m" * 40 and "union_columns" not in again["extra"]
        assert _rebuild_equal(root, tmp_path, shuffle_seed=33 if order == "remote_first" else 34) == {"analysis": 2, "analysis_input": 8, "analysis_artifact": 2}


# --------------------------------------------------------------------------------------------
# 11: the canonical carrier of an ordered-fact refusal (review round four)
# --------------------------------------------------------------------------------------------


def _copy_filing(root, src_day, dst_day, name="000001-analysis.jsonl"):
    """The immutable record file of one day copied under another day: the same record under a second logical uri."""
    dst = os.path.join(root, "records", "analyses", dst_day)
    os.makedirs(dst, exist_ok=True)
    shutil.copy2(os.path.join(root, "records", "analyses", src_day, name), os.path.join(dst, name))
    return f"records/analyses/{dst_day}/{name}#1"


def _refusal_uri(root, sha):
    return _q(root, "SELECT record_uri FROM record_refusal WHERE record_sha256 = ?", sha)[0]["record_uri"]


def _index_uri(root, sha):
    return _q(root, "SELECT record_uri FROM ordered_fact_record WHERE record_sha256 = ?", sha)[0]["record_uri"]


def test_an_ordered_fact_refusal_names_the_records_smallest_uri_whatever_copy_arrives_last_so_the_refusal_table_rebuilds_equal(tmp_path):
    """`record_refusal` is INSERT OR REPLACE by sha, and the conflict path wrote the ARRIVAL's uri, so a
    conflicting filing copied to a smaller uri and ingested later left the live row at the copy that arrived last while a rebuild, meeting
    the smaller copy first, wrote the smaller one -- the same records, two refusal tables (verify_equal: record_refusal 2). Now every
    ordered-fact refusal -- the conflict, the displaced record, the re-judged one -- names the index's canonical (smallest) carrier."""
    # 1. the finding: A kept, B refused for conflict at the larger uri, then B's copy at a SMALLER uri arrives
    root, _ = _root(tmp_path, "reg1")
    _filing_pair(root, "2026-01-06", 1, 810, "A: kept")
    _, sha_b = _filing_pair(root, "2026-01-07", 1, 820, "B: conflicting")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(900))
    assert rep.refused == 1 and _refusal_uri(root, sha_b) == "records/analyses/2026-01-07/000001-analysis.jsonl#1"
    small = _copy_filing(root, "2026-01-07", "2026-01-05")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(901))
    assert rep.refused == 1 and _refusal_uri(root, sha_b) == small == _index_uri(root, sha_b)
    assert _count(root, "record_refusal") == 1
    assert _rebuild_equal(root, tmp_path, shuffle_seed=41) == {"analysis": 1, "analysis_input": 0, "analysis_artifact": 0}
    # a copy at a LARGER uri afterwards moves nothing
    _copy_filing(root, "2026-01-07", "2026-01-09")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(902))
    assert rep.refused == 1 and _refusal_uri(root, sha_b) == small == _index_uri(root, sha_b)
    assert _rebuild_equal(root, tmp_path, shuffle_seed=42) == {"analysis": 1, "analysis_input": 0, "analysis_artifact": 0}
    # 1b. the copies in the opposite order: the smaller copy first, the original later -- the smaller stands
    root, _ = _root(tmp_path, "reg1b")
    _filing_pair(root, "2026-01-06", 1, 810, "A: kept")
    _, sha_b = _filing_pair(root, "2026-01-07", 1, 820, "B: conflicting")
    small = _copy_filing(root, "2026-01-07", "2026-01-05")
    os.replace(os.path.join(root, "records", "analyses", "2026-01-07", "000001-analysis.jsonl"), str(tmp_path / "b-held.jsonl"))
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(900))
    assert rep.refused == 1 and _refusal_uri(root, sha_b) == small
    shutil.copy2(str(tmp_path / "b-held.jsonl"), os.path.join(root, "records", "analyses", "2026-01-07", "000001-analysis.jsonl"))
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(901))
    assert rep.refused == 1 and _refusal_uri(root, sha_b) == small == _index_uri(root, sha_b)
    assert _rebuild_equal(root, tmp_path, shuffle_seed=43) == {"analysis": 1, "analysis_input": 0, "analysis_artifact": 0}
    # 2. the DISPLACED record: B kept, its copy at a smaller uri a duplicate (the row's carrier moves), then A -- earlier -- displaces B
    root, _ = _root(tmp_path, "reg2")
    _, sha_b = _filing_pair(root, "2026-01-07", 1, 820, "B: kept until A arrives")
    H.ingest_once(root, workers=0, min_free=0, now=synth.t(900))
    small = _copy_filing(root, "2026-01-07", "2026-01-05")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(901))
    assert rep.refused == 0 and rep.duplicates >= 1 and _q(root, "SELECT record_uri FROM analysis")[0]["record_uri"] == small
    _filing_pair(root, "2026-01-08", 1, 810, "A: the earliest, arrived last")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(902))
    assert rep.refused == 1 and rep.analysis_records == 1 and _refusal_uri(root, sha_b) == small == _index_uri(root, sha_b)
    assert _rebuild_equal(root, tmp_path, shuffle_seed=44) == {"analysis": 1, "analysis_input": 0, "analysis_artifact": 0}
    # 3. the RE-JUDGED record: B kept, C (B's substance, a later stamp) a duplicate, C's copy at a smaller uri, then A displaces B and C is
    # judged again against A -- refused at its smallest uri
    root, _ = _root(tmp_path, "reg3")
    _, sha_b = _filing_pair(root, "2026-01-07", 1, 820, "B: the first to arrive")
    H.ingest_once(root, workers=0, min_free=0, now=synth.t(900))
    d = os.path.join(root, "records", "analyses", "2026-01-08")
    c = synth.rec("analysis", _analysis_payload(filed_utc=synth.t(830), notes="B: the first to arrive"), synth.t(830))
    R.write_records(os.path.join(d, "000001-analysis.jsonl"), [c])
    sha_c = R.record_sha(c)
    H.ingest_once(root, workers=0, min_free=0, now=synth.t(901))
    small_c = _copy_filing(root, "2026-01-08", "2026-01-06")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(902))
    assert rep.refused == 0 and _index_uri(root, sha_c) == small_c and _count(root, "record_refusal") == 0
    _filing_pair(root, "2026-01-09", 1, 810, "A: the earliest, arrived last")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(903))
    assert rep.refused == 2 and rep.refilings_rejudged == 2
    assert _refusal_uri(root, sha_c) == small_c and _refusal_uri(root, sha_b) == "records/analyses/2026-01-07/000001-analysis.jsonl#1"
    assert _rebuild_equal(root, tmp_path, shuffle_seed=45) == {"analysis": 1, "analysis_input": 0, "analysis_artifact": 0}
    # C's original (larger) uri arrives again in a later cycle as a conflict: the row keeps the smaller carrier
    shutil.copy2(os.path.join(root, "records", "analyses", "2026-01-08", "000001-analysis.jsonl"), os.path.join(root, "records", "analyses", "2026-01-08", "000002-analysis.jsonl"))
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(904))
    assert rep.refused == 1 and _refusal_uri(root, sha_c) == small_c
    assert _rebuild_equal(root, tmp_path, shuffle_seed=46) == {"analysis": 1, "analysis_input": 0, "analysis_artifact": 0}


# --------------------------------------------------------------------------------------------
# 12 and 13: the closing pass -- a verified local copy publishes from the row's locations; the same bytes under several entries
# --------------------------------------------------------------------------------------------


def test_a_local_copy_filed_after_a_remote_declaration_publishes_from_the_rows_locations_while_the_primary_stands(tmp_path, capsys):
    """A filing of a known digest never moves the standing primary, so the local copy
    of a report first declared from the cluster was registered in the row's locations alone -- and `publish.registered_sha` read the
    primary only, so `publish_dir` refused the verified local file as unregistered. Now a path is registered when a row names it as its
    primary OR among its locations, the bytes still checked against the sha before anything is published."""
    root, ids = _root(tmp_path)
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    local = local_dir / "report.json"
    local.write_text('{"delta": 0.1}', encoding="utf-8")
    sha = PUB.sha256_of(str(local))
    remote_uri = "/scratch/example/someone/exp/study/report.json"
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, _remote_spec(ids, "study.remote", remote_uri, sha), "remote.json")]) == 0
    capsys.readouterr()
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        # before the local filing the remote primary is the only path the registry knows: the local copy is unregistered, and refused
        assert PUB.registered_sha(conn, remote_uri) == sha and PUB.registered_sha(conn, str(local)) is None
        with pytest.raises(PUB.Unregistered) as ei:
            PUB.publish_dir(conn, str(local_dir), str(tmp_path / "dest0"))
        assert "no artifact row" in str(ei.value) and str(local) in str(ei.value)
    finally:
        conn.close()
    # another analysis files the local copy: verified among the row's locations, the primary standing (round three's ruling)
    spec = _spec(ids, None, report={"uri": str(local), "sha256": None}, name="study.local", provenance={"kind": "user_direct"})
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec, "local.json"), "--json"]) == 0
    js = json.loads(capsys.readouterr().out)
    (art,) = js["artifacts"]
    assert art["uri"] == str(local) and art["artifact_id"] == sha and art["known"] is True and art["record_written"] is True and art["remote"] is False
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    row = _artifact_row(root, sha)
    assert row["uri_primary"] == remote_uri and [l["uri"] for l in row["locations"]] == [remote_uri, str(local)]
    assert row["locations"][1]["verified_utc"] == row["verified_utc"] and row["verified_utc"] is not None and row["bytes"] == 14
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        assert PUB.registered_sha(conn, str(local)) == sha == PUB.registered_sha(conn, remote_uri)       # registered through the row's locations
        assert PUB.registered_sha(conn, str(local_dir / "nowhere.json")) is None
        assert PUB.unregistered_in_dir(conn, str(local_dir)) == []
        rep = PUB.publish_dir(conn, str(local_dir), str(tmp_path / "dest"))
        assert rep.published == [str(local)] and rep.refused == []
        assert (tmp_path / "dest" / "report.json").read_bytes() == local.read_bytes()
        assert PUB.publish_dir(conn, str(local_dir), str(tmp_path / "dest")).published == [str(local)]    # again, over the destination's copy
        # the bytes are still checked: a drifted local copy is refused by name although its path is a registered location
        local.write_text('{"delta": 0.2}', encoding="utf-8")
        with pytest.raises(PUB.Unregistered) as ei:
            PUB.publish_dir(conn, str(local_dir), str(tmp_path / "dest2"))
        assert "bytes differ from the registered sha" in str(ei.value) and str(local) in str(ei.value)
        assert PUB.unregistered_in_dir(conn, str(local_dir)) == [str(local)]
    finally:
        conn.close()
    assert _rebuild_equal(root, tmp_path, shuffle_seed=51) == {"analysis": 2, "analysis_input": 8, "analysis_artifact": 2}


def test_the_same_bytes_under_several_entries_are_one_link_listing_every_entry_and_every_distinct_path_a_location_of_the_row(tmp_path, capsys, monkeypatch):
    """A digest seen once was dropped silently on its next path, so the second of two page directories with
    identical files was never registered and its url reached no link. Now the FIRST entry of a digest carries the one link ((analysis_id,
    artifact_id) is the key), every further entry is listed in the link's `extra.same_bytes`, and every further DISTINCT path writes its
    own artifact record whose location the union keeps, the first path the row's primary -- so a page whose every file is another page's
    bytes is registered, read back by its url, and publishable."""
    root, ids = _root(tmp_path)
    page_a, page_b = _page_dir(tmp_path, "page_a"), _page_dir(tmp_path, "page_b")        # the same four publishable files, byte for byte
    url_a, url_b = "https://example.invalid/figs/a/", "https://example.invalid/figs/b/"
    a = {f: os.path.join(page_a, f) for f in ("numbers.json", "figures/a.png", "figures/a.svg", "index.html")}
    b = {f: os.path.join(page_b, f) for f in a}
    report = a["numbers.json"]                                                            # the report is ALSO a file of its own page directory
    spec = _spec(ids, None, report={"uri": report, "sha256": None},
                 artifacts=[{"dir": page_a, "role": "page", "url": url_a}, {"dir": page_b, "role": "page", "url": url_b}])
    sha = {f: PUB.sha256_of(p) for f, p in a.items()}
    # the dry run: eight paths to register (A's four, B's four), four links; each same-bytes path marked with the path that carries its link
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "artifacts: 8 file(s) to register" in out and "1 artifact file (8 record(s)), 1 analysis, 4 analysis_input, 4 analysis_artifact" in out
    assert f"{b['numbers.json']} [same bytes as {report}: one link, this path a location of the row]" in out
    assert f"{b['index.html']} [same bytes as {a['index.html']}: one link, this path a location of the row]" in out
    _no_filing_written(root)
    # a directory listed twice adds nothing; the same file under another role is listed in the link too, with no second record
    plan = AN.build_plan(dict(spec, artifacts=spec["artifacts"][:1] + spec["artifacts"] + [{"path": b["figures/a.png"], "role": "figure"}]), R.utc_now(), None)
    assert len(plan.artifacts) == 8 and len(plan.link_payloads) == 4
    png_link = next(l for l in plan.link_payloads if l["artifact_id"] == sha["figures/a.png"])
    assert png_link["extra"]["same_bytes"] == [{"uri": b["figures/a.png"], "role": "page", "page_dir": page_b, "relpath": "figures/a.png", "page_url": url_b},
                                               {"uri": b["figures/a.png"], "role": "figure"}]
    # the filing
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec), "--json"]) == 0
    js = json.loads(capsys.readouterr().out)
    by_uri = {x["uri"]: x for x in js["artifacts"]}
    assert len(by_uri) == 8 and all(x["record_written"] and not x["known"] for x in by_uri.values())
    assert by_uri[report]["same_bytes_as"] is None and by_uri[report]["role"] == "report"
    assert {f: by_uri[p]["same_bytes_as"] for f, p in b.items()} == {"numbers.json": report, "figures/a.png": a["figures/a.png"], "figures/a.svg": a["figures/a.svg"], "index.html": a["index.html"]}
    assert all(by_uri[p]["same_bytes_as"] is None for p in a.values())
    # the records: eight artifact records, each naming ITS one location; a same-bytes record restates the first path as the primary and the
    # first entry's kind and states no page keys (the row's are the carrier's); four links, each listing its further entries
    recs = {os.path.basename(p): [r for _, r, _ in R.read_record_file(p)] for p in js["written"]}
    arts = {r["payload"]["locations_json"][0]["uri"]: r["payload"] for r in recs["000001-artifact.jsonl"]}
    assert list(arts) == [report, a["figures/a.png"], a["figures/a.svg"], a["index.html"], b["figures/a.png"], b["figures/a.svg"], b["index.html"], b["numbers.json"]]
    assert arts[b["numbers.json"]]["artifact_id"] == sha["numbers.json"] and arts[b["numbers.json"]]["uri_primary"] == report and arts[b["numbers.json"]]["kind"] == "report"
    assert arts[b["numbers.json"]]["extra"] == {"analysis": "study.capture"} and arts[b["numbers.json"]]["bytes"] == 8
    assert arts[b["index.html"]]["uri_primary"] == a["index.html"] and arts[b["index.html"]]["kind"] == "page" and arts[a["index.html"]]["uri_primary"] == a["index.html"]
    assert arts[report]["extra"] == {"analysis": "study.capture", "role": "report"}
    links = {r["payload"]["artifact_id"]: r["payload"] for r in recs["000004-analysis_artifact.jsonl"]}
    assert set(links) == set(sha.values()) and len(recs["000004-analysis_artifact.jsonl"]) == 4
    assert links[sha["numbers.json"]]["role"] == "report" and links[sha["numbers.json"]]["extra"] == {"same_bytes": [
        {"uri": report, "role": "page", "page_dir": page_a, "relpath": "numbers.json", "page_url": url_a},
        {"uri": b["numbers.json"], "role": "page", "page_dir": page_b, "relpath": "numbers.json", "page_url": url_b}]}
    assert links[sha["index.html"]]["role"] == "page" and links[sha["index.html"]]["extra"] == {
        "page_dir": page_a, "relpath": "index.html", "page_url": url_a,
        "same_bytes": [{"uri": b["index.html"], "role": "page", "page_dir": page_b, "relpath": "index.html", "page_url": url_b}]}
    # the fold: four rows, each with both paths among its locations (both verified by the one read), the first path its primary and the
    # carrier's extra, whatever order the fold met the two same-stamp records in
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0 and rep.analysis_records == 1 + 4 + 4
    assert _count(root, "analysis_artifact") == 4 and _count(root, "analysis") == 1
    for f in a:
        row = _artifact_row(root, sha[f])
        assert row["uri_primary"] == a[f] and sorted(l["uri"] for l in row["locations"]) == sorted([a[f], b[f]]), f
        assert all(l["verified_utc"] == row["verified_utc"] and l["bytes"] == row["bytes"] for l in row["locations"]) and row["verified_utc"] is not None
        assert "union_columns" not in row["extra"] and row["extra"].get("page_url", url_a) == url_a and row["extra"].get("page_dir", page_a) == page_a, f
    assert _artifact_row(root, sha["numbers.json"])["extra"] == {"analysis": "study.capture", "role": "report"}
    assert _artifact_row(root, sha["index.html"])["extra"] == {"analysis": "study.capture", "page_dir": page_a, "relpath": "index.html", "page_url": url_a}
    # read back: B's url -- named by the same-bytes entries alone -- in the listing, its json and the status page beside A's
    assert cli.main(["--root", root, "analysis", "list"]) == 0
    text = capsys.readouterr().out
    assert "inputs 2 run(s) / 2 object(s); 4 artifact(s)" in text and f"pages {url_a} {url_b}" in text
    assert cli.main(["--root", root, "analysis", "list", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["page_urls"] == f"{url_a} {url_b}" and len(listed[0]["artifacts"]) == 4
    assert json.loads(next(x for x in listed[0]["artifacts"] if x["artifact_id"] == sha["index.html"])["extra_json"])["same_bytes"][0]["page_url"] == url_b
    ctx = status.status_json(root)
    assert ctx["analyses"][0]["page_urls"] == f"{url_a} {url_b}" and ctx["analyses"][0]["n_artifacts"] == 4
    # page B publishes: its files are registered at their own paths, among the rows' locations
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        assert PUB.registered_sha(conn, b["index.html"]) == sha["index.html"] and PUB.unregistered_in_dir(conn, page_b) == []
        assert sorted(os.path.basename(p) for p in PUB.publish_dir(conn, page_b, str(tmp_path / "dest_b")).published) == ["index.html", "numbers.json"]
    finally:
        conn.close()
    # a re-filing: the analysis, its inputs and its four links are duplicates; the eight paths are known and verified anew in place (each a
    # merged record: the standing primary, its own location), so the rows keep their primary and both locations; equal after shuffled
    # rebuilds on both fold paths
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec), "--json"]) == 0
    js2 = json.loads(capsys.readouterr().out)
    assert len(js2["artifacts"]) == 8 and all(x["known"] and x["record_written"] for x in js2["artifacts"])
    assert {x["uri"]: x["same_bytes_as"] for x in js2["artifacts"]} == {x["uri"]: x["same_bytes_as"] for x in js["artifacts"]}
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.analysis_records == 0 and rep.duplicates >= 1 + 4 + 4
    assert _count(root, "analysis_artifact") == 4
    for f in a:
        row = _artifact_row(root, sha[f])
        assert row["uri_primary"] == a[f] and sorted(l["uri"] for l in row["locations"]) == sorted([a[f], b[f]]), f
        assert all(l["verified_utc"] == row["verified_utc"] for l in row["locations"]) and row["verified_utc"] is not None
    assert _rebuild_equal(root, tmp_path, shuffle_seed=61) == {"analysis": 1, "analysis_input": 4, "analysis_artifact": 4}
    monkeypatch.setattr(RED, "INCREMENTAL_FOLD", False)
    assert _rebuild_equal(root, tmp_path, shuffle_seed=62) == {"analysis": 1, "analysis_input": 4, "analysis_artifact": 4}
    monkeypatch.setattr(RED, "INCREMENTAL_FOLD", True)


# --------------------------------------------------------------------------------------------
# 14: a report re-filed at a path an earlier version holds as its primary (the second closing pass)
# --------------------------------------------------------------------------------------------


def test_a_report_re_filed_at_a_path_an_earlier_version_holds_as_its_primary_publishes_the_bytes_the_path_now_holds(tmp_path, capsys):
    """End to end: a notebook registered version A of a report at local path P
    (`publish.register_files`: P its row's primary); an analysis declared version B from the cluster; P was replaced with B's bytes and a
    second analysis filed the local copy -- B's row names P as a verified location, its remote primary standing.
    The newest PRIMARY read first was A's, so `publish_dir` refused the correctly registered bytes as drifted while `unregistered_in_dir`
    reported nothing. The bytes at the path now pick the registration among every row naming it, A's row stands as history, drifted
    bytes are refused naming both registrations, and the rows rebuild equal."""
    root, ids = _root(tmp_path)
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    local = local_dir / "report.json"
    local.write_text('{"delta": 0.1}', encoding="utf-8")
    sha_a = PUB.sha256_of(str(local))
    PUB.register_files(root, [str(local)], notebook_commit="c" * 40, now=synth.t(9000))          # A at P, P its row's primary
    b_text = '{"delta": 0.2}'
    sha_b = hashlib.sha256(b_text.encode("utf-8")).hexdigest()
    remote_uri = "/scratch/example/someone/exp/study/report.json"
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, _remote_spec(ids, "study.remote", remote_uri, sha_b), "remote.json")]) == 0
    capsys.readouterr()
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    local.write_text(b_text, encoding="utf-8")                                                   # P now holds B
    spec = _spec(ids, None, report={"uri": str(local), "sha256": None}, name="study.local", provenance={"kind": "user_direct"})
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec, "local.json"), "--json"]) == 0
    js = json.loads(capsys.readouterr().out)
    (art,) = js["artifacts"]
    assert art["artifact_id"] == sha_b and art["known"] is True and art["record_written"] is True
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    row_b = _artifact_row(root, sha_b)
    assert row_b["uri_primary"] == remote_uri and [l["uri"] for l in row_b["locations"]] == [remote_uri, str(local)]
    assert _artifact_row(root, sha_a)["uri_primary"] == str(local)                               # A's row stands as history
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        # before the fix: refused as "bytes differ from the registered sha <A>" (the newest primary read first) while unregistered_in_dir saw nothing
        assert PUB.unregistered_in_dir(conn, str(local_dir)) == []
        rep = PUB.publish_dir(conn, str(local_dir), str(tmp_path / "dest"))
        assert rep.published == [str(local)] and (tmp_path / "dest" / "report.json").read_text(encoding="utf-8") == b_text
        assert PUB.registered_shas(conn, str(local)) == [sha_b, sha_a] and PUB.registered_sha(conn, str(local)) == sha_b
        assert PUB.registered_shas(conn, remote_uri) == [sha_b]
        # bytes matching neither registration are refused, naming both newest first; the two readers agree
        local.write_text('{"delta": 0.3}', encoding="utf-8")
        with pytest.raises(PUB.Unregistered) as ei:
            PUB.publish_dir(conn, str(local_dir), str(tmp_path / "dest2"))
        msg = str(ei.value)
        assert str(local) in msg and "bytes differ from every sha registered at the path" in msg and msg.index(sha_b[:12]) < msg.index(sha_a[:12])
        assert PUB.unregistered_in_dir(conn, str(local_dir)) == [str(local)]
    finally:
        conn.close()
    assert _rebuild_equal(root, tmp_path, shuffle_seed=52) == {"analysis": 2, "analysis_input": 8, "analysis_artifact": 2}


# --------------------------------------------------------------------------------------------
# 15: the union is the fold's rule for `artifact.locations_json`, from every producer (the fourth closing pass)
# --------------------------------------------------------------------------------------------


def _artifact_event(i, payload, extra=None):
    body = dict(payload)
    body["_kind"] = "full"
    if extra is not None:
        body["_extra"] = extra
    return RED.HistoryEvent(synth.t(i), f"{i:064x}", "artifact", f"records/x/{i:06d}-artifact.jsonl", body)


def test_every_artifact_record_adds_its_locations_to_the_standing_list_marker_or_none_and_a_null_adds_nothing():
    """The union was a per-record directive (`extra.union_columns`) the publish writers alone carried, so a
    record without it -- an adapter's, a hook's, the status page's, one written by hand -- still REPLACED the list and erased a filed
    location. Now the fold unions `artifact.locations_json` on every full record of the table (`reducer.UNION_COLUMNS`); the marker is
    admitted, names what the fold does anyway, and never lands in the extra."""
    v_a = {"uri": "/a", "verified_utc": synth.t(1), "bytes": 3}
    v_c = {"uri": "/c", "verified_utc": synth.t(2), "bytes": 3}
    base = {"artifact_id": SHA_A, "kind": "report", "uri_primary": "/a"}
    events = [_artifact_event(1, {**base, "locations_json": canonical_json([v_a])}),                                   # a publish record
              _artifact_event(2, {**base, "locations_json": canonical_json([v_c]), "bytes": 3}),                       # a filing's copy: no marker
              _artifact_event(3, {**base, "locations_json": canonical_json([v_a]), "bytes": 3, "verified_utc": synth.t(1)}),    # the adapter's re-registration
              _artifact_event(4, {**base, "locations_json": canonical_json(["/x/results.jsonl"])}),                   # a hook's bare path
              _artifact_event(5, {**base, "locations_json": canonical_json([v_a])}, extra={"union_columns": ["locations_json"], "note": "n"}),
              _artifact_event(6, {**base, "locations_json": None})]                                                     # a null: no statement
    folded = RED.fold("artifact", events)
    lists = [json.loads(s["locations_json"]) for s, _ in folded]
    assert lists == [[v_a], [v_a, v_c], [v_a, v_c], [v_a, v_c, "/x/results.jsonl"], [v_a, v_c, "/x/results.jsonl"], [v_a, v_c, "/x/results.jsonl"]]
    stamps = [fv["locations_json"][0] for _, fv in folded]
    assert stamps[:5] == [synth.t(i) for i in range(1, 6)] and stamps[5] == synth.t(5)      # the null moved neither the list nor its stamp
    assert folded[4][0]["extra_json"] == canonical_json({"note": "n"}) and all("union_columns" not in (s.get("extra_json") or "") for s, _ in folded)
    # a first record stating null folds null, as it always did
    assert RED.fold("artifact", [_artifact_event(7, {**base, "locations_json": None})])[0][0]["locations_json"] is None


def _revision_artifact(root, rev, payload, stamp):
    """An adapter's `artifact` record in an ADAPTER REVISION DIRECTORY with its accepted marker, the marker last: the shape an adapter revision
    writes for an input it read (`adapters.base.Input.artifact_payload`), naming the adapter's own locations and carrying no fold directive."""
    d = os.path.join(root, "records", "adapters", "synthetic_ad", rev)
    R.write_records(os.path.join(d, "000001-artifact.jsonl"), [synth.rec("artifact", payload, stamp)])
    marker = synth.rec("source_revision", {"revision_id": rev, "source_id": "synthetic_source", "adapter": "synthetic_ad", "observed_utc": stamp,
                                           "inputs_json": {str(payload["uri_primary"]): str(payload["artifact_id"])}, "accepted": 1, "priority": 0},
                       R.epoch_ms_to_utc(R.utc_to_epoch_ms(stamp) + 9000))
    R.write_records(os.path.join(d, "000002-source_revision.jsonl"), [marker])


def test_an_adapter_revision_re_registering_an_unchanged_input_keeps_the_location_an_analysis_filed_for_its_bytes(tmp_path, capsys, monkeypatch):
    """End to end: an adapter registers an input file; an analysis files a local COPY of the same bytes (a
    known digest: the copy its one location); a fresh adapter revision registers the unchanged input again. The adapter's record names its
    own location and carries no directive, so before this it replaced the list and the filed copy -- still linked -- refused to publish as
    unregistered. Now both locations stand, the copy publishes, and the row rebuilds equal with and without the appended-fold shortcut."""
    root, ids = _root(tmp_path)
    store = tmp_path / "store"
    store.mkdir()
    inp_path = store / "results.jsonl"
    inp_path.write_text('{"delta": 0.1}\n', encoding="utf-8")
    sha = PUB.sha256_of(str(inp_path))
    inp = AB.Input(uri=str(inp_path), path=str(inp_path), sha256=sha, size=inp_path.stat().st_size, mtime_utc=synth.t(650), role="report")
    payload = inp.artifact_payload()
    assert "union_columns" not in payload.get("extra", {}) and payload["locations_json"] == [{"uri": str(inp_path), "verified_utc": synth.t(650), "bytes": 15}]
    _revision_artifact(root, "a1b2c3d4e5f60001", payload, synth.t(700))
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(710))
    assert rep.refused == 0 and rep.blocked == 0
    before = _artifact_row(root, sha)
    assert [l["uri"] for l in before["locations"]] == [str(inp_path)] and before["uri_primary"] == str(inp_path) and before["kind"] == "report"
    assert before["verified_utc"] == synth.t(650) and before["bytes"] == 15 and before["extra"] == {}
    # an analysis files a local copy of the bytes: the digest is known, the copy the record's one location
    copy_dir = tmp_path / "copy"
    copy_dir.mkdir()
    copy = copy_dir / "report.json"                                   # a publishable suffix (publish.FIGURE_SUFFIXES)
    copy.write_bytes(inp_path.read_bytes())
    spec = _spec(ids, None, report={"uri": str(copy), "sha256": None}, name="study.copy", provenance={"kind": "user_direct"})
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec, "copy.json"), "--json"]) == 0
    js = json.loads(capsys.readouterr().out)
    (art,) = js["artifacts"]
    assert art["artifact_id"] == sha and art["known"] is True and art["record_written"] is True and art["remote"] is False
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    mid = _artifact_row(root, sha)
    assert [l["uri"] for l in mid["locations"]] == [str(inp_path), str(copy)] and mid["uri_primary"] == str(inp_path)
    assert mid["locations"][1]["verified_utc"] == mid["verified_utc"] is not None and _count(root, "analysis_artifact") == 1
    # a fresh adapter revision registers the unchanged input again, after the filing in log order: the same record, no directive
    later = R.utc_now()
    _revision_artifact(root, "a1b2c3d4e5f60002", payload, later)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0
    after = _artifact_row(root, sha)
    assert [l["uri"] for l in after["locations"]] == [str(inp_path), str(copy)], after["locations"]
    assert after["locations"][0] == before["locations"][0] and after["locations"][1] == mid["locations"][1]
    assert after["uri_primary"] == str(inp_path) and after["kind"] == "report" and after["verified_utc"] == synth.t(650) and after["bytes"] == 15
    assert after["extra"] == mid["extra"] and "union_columns" not in after["extra"]
    assert _q(root, "SELECT COUNT(*) AS n FROM analysis_artifact WHERE artifact_id = ?", sha)[0]["n"] == 1
    # the history: three events of the key, each snapshot the union so far; the last folded through the appended-fold shortcut
    hist = H.history(root, DB.connect(DB.db_path(root), readonly=True), "artifact", {"artifact_id": sha})
    assert [[l["uri"] for l in json.loads(h["state"]["locations_json"])] for h in hist] == [[str(inp_path)], [str(inp_path), str(copy)], [str(inp_path), str(copy)]]
    # the filed copy publishes from the row's locations
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        assert PUB.registered_sha(conn, str(copy)) == sha == PUB.registered_sha(conn, str(inp_path))
        prep = PUB.publish_dir(conn, str(copy_dir), str(tmp_path / "dest"))
        assert prep.published == [str(copy)] and prep.refused == []
        assert (tmp_path / "dest" / "report.json").read_bytes() == inp_path.read_bytes()
    finally:
        conn.close()
    # equal after shuffled rebuilds (the revision directories replay before the analyses files), with the shortcut and without it
    assert _rebuild_equal(root, tmp_path, shuffle_seed=61) == {"analysis": 1, "analysis_input": 4, "analysis_artifact": 1}
    monkeypatch.setattr(RED, "INCREMENTAL_FOLD", False)
    assert _rebuild_equal(root, tmp_path, shuffle_seed=62) == {"analysis": 1, "analysis_input": 4, "analysis_artifact": 1}
    monkeypatch.setattr(RED, "INCREMENTAL_FOLD", True)


# --------------------------------------------------------------------------------------------
# 16: a link's `extra.same_bytes` has a shape, and the readers never fail on one that does not (the fourth closing pass)
# --------------------------------------------------------------------------------------------


def _link_record(extra, stamp=700):
    rec = synth.rec("analysis_artifact", {"analysis_id": AID, "filing_sha256": FSHA, "artifact_id": SHA_A, "role": "report", "extra": extra}, synth.t(stamp))
    return VAL.validate_record(rec, "<memory>", R.record_sha(rec))


def test_a_links_same_bytes_is_checked_for_its_shape_and_the_listing_and_the_status_section_skip_an_entry_that_is_not_an_object(tmp_path, capsys):
    """`extra.same_bytes` was free-form, and the page-url query handed every entry to `json_extract`,
    which raises `malformed JSON` on a bare string -- one accepted link with path strings for entries failed `registry analysis list` whole
    and showed the status page's Analyses section as a query error. Now the validator refuses a NEW link carrying the key in any shape but a
    list of {uri, role, ...} objects, by name, and the readers skip an entry that is not an object."""
    assert AN.SAME_BYTES_KEY == VAL.SAME_BYTES_KEY == "same_bytes" and VAL.SAME_BYTES_TABLE == "analysis_artifact"
    good = [{"uri": "/p/index.html", "role": "page", "page_dir": "/p", "relpath": "index.html", "page_url": "https://example.invalid/p/"},
            {"uri": "/q/r.json", "role": "report"}]
    v = _link_record({"same_bytes": good})
    assert isinstance(v, VAL.Validated) and v.extra == {"same_bytes": good}
    assert isinstance(_link_record({}), VAL.Validated) and isinstance(_link_record({"same_bytes": []}), VAL.Validated)
    cases = [(["/p/index.html", "/q/r.json"], "same_bytes[0] must be an object"),
             ("/p/index.html", "same_bytes must be a JSON list"),
             ({"uri": "/p", "role": "page"}, "same_bytes must be a JSON list"),
             ([{"role": "page"}], "same_bytes[0].uri must be a non-empty string"),
             ([good[0], {"uri": "  ", "role": "page"}], "same_bytes[1].uri must be a non-empty string"),
             ([{"uri": "/p", "role": "figurine"}], "same_bytes[0].role must be one of report|page|figure"),
             ([{"uri": "/p"}], "same_bytes[0].role must be one of"),
             ([{"uri": "/p", "role": "page", "page_url": 7}], "same_bytes[0].page_url must be a string or null")]
    for entries, reason in cases:
        m = _link_record({"same_bytes": entries, "page_url": "https://example.invalid/x/"})
        assert isinstance(m, VAL.Malformed) and m.field == "extra.same_bytes" and reason in m.reason, (entries, getattr(m, "reason", m))
    # on another table the key is free-form metadata: an artifact record carrying it passes
    assert isinstance(_validate({"artifact_id": SHA_A, "kind": "report", "uri_primary": "/c/r.json", "locations_json": [], "extra": {"same_bytes": ["/x"]}}), VAL.Validated)
    # end to end: an ordinary filing with a page (four links, each with the page's url), then two of its links' extras set to the shapes an
    # earlier validator accepted -- path strings for entries, and a bare string for the key -- beside the two well-formed links
    root, ids = _root(tmp_path)
    page = _page_dir(tmp_path)
    url = "https://example.invalid/figs/study/"
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, _spec(ids, page))]) == 0
    capsys.readouterr()
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0 and _count(root, "analysis_artifact") == 4
    aid = _q(root, "SELECT analysis_id FROM analysis")[0]["analysis_id"]
    links = _q(root, "SELECT artifact_id FROM analysis_artifact WHERE analysis_id = ? ORDER BY artifact_id", aid)
    conn = DB.connect(DB.db_path(root))
    try:
        conn.execute("UPDATE analysis_artifact SET extra_json = ? WHERE analysis_id = ? AND artifact_id = ?",
                     (canonical_json({"same_bytes": ["/legacy/a.png", "/legacy/b.png"], "page_url": "https://example.invalid/legacy/"}), aid, links[0]["artifact_id"]))
        conn.execute("UPDATE analysis_artifact SET extra_json = ? WHERE analysis_id = ? AND artifact_id = ?",
                     (canonical_json({"same_bytes": "/legacy/c.png"}), aid, links[1]["artifact_id"]))
    finally:
        conn.close()
    # the entries' type guards the extraction: a bare string yields no url and raises nothing; the links' own urls still read
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        rows = AN.list_analyses(conn)
        assert len(rows) == 1 and rows[0]["n_artifacts"] == 4 and rows[0]["page_urls"] == f"{url} https://example.invalid/legacy/"
        assert any(f"pages {url} https://example.invalid/legacy/" in line for line in AN.list_lines(rows))
        st = [dict(r) for r in conn.execute(status.QUERIES["analyses"]).fetchall()]
        assert len(st) == 1 and st[0]["n_artifacts"] == 4 and st[0]["page_urls"] == f"{url} https://example.invalid/legacy/"
        # a well-formed entry beside the malformed ones is still read: its url joins the list
        conn.close()
        conn = DB.connect(DB.db_path(root))
        conn.execute("UPDATE analysis_artifact SET extra_json = ? WHERE analysis_id = ? AND artifact_id = ?",
                     (canonical_json({"same_bytes": [{"uri": "/other/index.html", "role": "page", "page_url": "https://example.invalid/other/"}]}), aid, links[2]["artifact_id"]))
        conn.close()
        conn = DB.connect(DB.db_path(root), readonly=True)
        assert AN.list_analyses(conn)[0]["page_urls"] == f"{url} https://example.invalid/legacy/ https://example.invalid/other/"
    finally:
        conn.close()
    assert cli.main(["--root", root, "analysis", "list"]) == 0
    assert f"pages {url} https://example.invalid/legacy/ https://example.invalid/other/" in capsys.readouterr().out
    ctx = status.status_json(root)
    assert ctx["analyses"][0]["n_artifacts"] == 4 and ctx["analyses"][0]["page_urls"] == f"{url} https://example.invalid/legacy/ https://example.invalid/other/"


# --------------------------------------------------------------------------------------------
# 17: a child is bound to its filing; a refused re-filing's report never attaches to the kept analysis
# --------------------------------------------------------------------------------------------


def _report_filing(root, ids, day, seq0, stamp_i, report_tag, run_id=None, aid=AID, header_over=None):
    """A complete filing of ``aid`` by hand, as the CLI writes one: the report's artifact record, the header (its report named by
    ``report_tag``, stamp ``stamp_i``), one run input, one link to the report -- every child bound to the header's record sha. Returns
    ``(header sha, report sha, [artifact path, header path, input path, link path])``."""
    rid = run_id or ids["run_ids"][0]
    rsha = hashlib.sha256(f"report {report_tag}".encode("utf-8")).hexdigest()
    d = os.path.join(root, "records", "analyses", day)
    art = {"artifact_id": rsha, "kind": "report", "uri_primary": f"/synthetic/{report_tag}/report.json", "locations_json": [], "bytes": 16,
           "created_utc": synth.t(stamp_i - 1), "verified_utc": synth.t(stamp_i - 1), "extra": {"analysis": "study.capture"}}
    paths = [R.write_records(os.path.join(d, f"{seq0:06d}-artifact.jsonl"), [synth.rec("artifact", art, synth.t(stamp_i - 1))])]
    over = {"filed_utc": synth.t(stamp_i), "report_uri": f"/synthetic/{report_tag}/report.json", "report_sha256": rsha}
    over.update(header_over or {})
    header = synth.rec("analysis", _analysis_payload(aid, **over), synth.t(stamp_i))
    hsha = R.record_sha(header)
    paths.append(R.write_records(os.path.join(d, f"{seq0 + 1:06d}-analysis.jsonl"), [header]))
    inp = {"analysis_id": aid, "filing_sha256": hsha, "input_kind": "run", "input_id": rid, "run_id": rid, "object_id": None}
    paths.append(R.write_records(os.path.join(d, f"{seq0 + 2:06d}-analysis_input.jsonl"), [synth.rec("analysis_input", inp, synth.t(stamp_i + 1))]))
    link = {"analysis_id": aid, "filing_sha256": hsha, "artifact_id": rsha, "role": "report", "extra": {}}
    paths.append(R.write_records(os.path.join(d, f"{seq0 + 3:06d}-analysis_artifact.jsonl"), [synth.rec("analysis_artifact", link, synth.t(stamp_i + 2))]))
    return hsha, rsha, paths


def _links(root):
    return _q(root, "SELECT artifact_id, filing_sha256 FROM analysis_artifact ORDER BY artifact_id")


def _assert_child_refusals_name_the_refused_filing(root, hsha_refused, hsha_kept):
    """B's input and B's link stand refused by name, on `filing_sha256`, naming B as refused and the kept filing."""
    ref = {r["record_type"]: r for r in _q(root, "SELECT record_type, field, reason FROM record_refusal WHERE record_type != 'analysis'")}
    assert set(ref) == {"analysis_input", "analysis_artifact"}, ref
    for rt in ("analysis_input", "analysis_artifact"):
        assert ref[rt]["field"] == "filing_sha256"
        assert ref[rt]["reason"].startswith(f"filing of analysis {AID} not kept: this {rt} names the filing {hsha_refused[:12]}, which stands refused"), ref[rt]
        assert f"kept filing's ({hsha_kept[:12]})" in ref[rt]["reason"]
        assert H.dependency_refusal_parent(ref[rt]["reason"]) is None                     # final, not re-selected by the retry pass


def test_a_refused_re_filings_report_is_never_attached_to_the_kept_analysis_in_either_arrival_order(tmp_path):
    """A re-filing of the same name, code sha and inputs with ANOTHER report had its header refused as a
    conflicting re-filing while its `analysis_artifact` link folded on its own -- the kept analysis_id being a row -- so the kept analysis
    listed a report its accepted filing never named. Now every child names the filing that wrote it (`filing_sha256`) and folds only
    while that filing is the kept one or a duplicate of it, in every arrival order, live and on rebuild alike."""
    # 1. A (the earlier filing) folds first; B, the same analysis with another report, arrives later: B's header is a conflicting re-filing
    #    and B's input and link are refused WITH it, by name -- the kept analysis links A's report alone
    root, ids = _root(tmp_path, "reg1")
    hsha_a, rsha_a, _ = _report_filing(root, ids, "2026-01-06", 1, 810, "a")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(900))
    assert rep.refused == 0 and rep.analysis_records == 3 and _links(root) == [{"artifact_id": rsha_a, "filing_sha256": hsha_a}]
    hsha_b, rsha_b, _ = _report_filing(root, ids, "2026-01-07", 1, 820, "b")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(901))
    assert rep.refused == 3 and rep.analysis_records == 0 and rep.filing_children_rejudged == 0
    assert _links(root) == [{"artifact_id": rsha_a, "filing_sha256": hsha_a}]
    assert _q(root, "SELECT filing_sha256 FROM analysis_input") == [{"filing_sha256": hsha_a}]
    assert _q(root, "SELECT report_sha256, record_sha256 FROM analysis")[0] == {"report_sha256": rsha_a, "record_sha256": hsha_a}
    assert _q(root, "SELECT COUNT(*) AS n FROM artifact WHERE artifact_id IN (?, ?)", rsha_a, rsha_b)[0]["n"] == 2   # both files are registered; one is LINKED
    hdr = _refusals(root, "analysis")
    assert len(hdr) == 1 and hdr[0]["record_sha256"] == hsha_b
    assert hdr[0]["reason"].startswith(f"conflicting re-filing of analysis {AID}: report_uri, report_sha256 differ from the kept filing {hsha_a[:12]}")
    _assert_child_refusals_name_the_refused_filing(root, hsha_b, hsha_a)
    for sha in _q(root, "SELECT record_sha256 FROM record_refusal"):
        assert _q(root, "SELECT COUNT(*) AS n FROM record_refusal_fact WHERE record_sha256 = ?", sha["record_sha256"])[0]["n"] == 1   # each judgment a fact
    with DB.reading(DB.db_path(root)) as conn:
        assert [a["artifact_id"] for a in AN.artifacts_of(conn, AID)] == [rsha_a]
    assert _rebuild_equal(root, tmp_path, shuffle_seed=51) == {"analysis": 1, "analysis_input": 1, "analysis_artifact": 1}
    # a quiet cycle: the children's refusals are final verdicts, not dependency refusals -- nothing re-selected, nothing re-judged
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(902))
    assert rep.refused == 0 and rep.dependency_refusals_retried == 0 and rep.filing_children_rejudged == 0 and _count(root, "record_refusal") == 3
    # 2. the opposite order: B (later-stamped) folds first and is kept, its link accepted; then A (earlier-stamped) arrives LAST. A takes
    #    the header, B is displaced and refused, and B's input and link -- accepted while B was kept -- are WITHDRAWN and refused by name;
    #    A's own children, after its header in the log's order, fold
    root, ids = _root(tmp_path, "reg2")
    hsha_b, rsha_b, _ = _report_filing(root, ids, "2026-01-07", 1, 820, "b")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(900))
    assert rep.refused == 0 and _links(root) == [{"artifact_id": rsha_b, "filing_sha256": hsha_b}]
    hsha_a, rsha_a, _ = _report_filing(root, ids, "2026-01-08", 1, 810, "a")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(901))
    assert rep.analysis_records == 3 and rep.refused == 3 and rep.filing_children_rejudged == 2
    assert _links(root) == [{"artifact_id": rsha_a, "filing_sha256": hsha_a}]
    assert _q(root, "SELECT filing_sha256, record_written_utc FROM analysis_input") == [{"filing_sha256": hsha_a, "record_written_utc": synth.t(811)}]
    assert _q(root, "SELECT report_sha256, record_sha256 FROM analysis")[0] == {"report_sha256": rsha_a, "record_sha256": hsha_a}
    _assert_child_refusals_name_the_refused_filing(root, hsha_b, hsha_a)
    assert _refusals(root, "analysis")[0]["record_sha256"] == hsha_b
    # the record index holds the kept filing's children alone: a withdrawn child is out of the index as a rebuild never indexes it
    for t in ("analysis_input", "analysis_artifact"):
        idx = _q(root, "SELECT record_sha256, record_uri FROM ordered_fact_record WHERE table_name = ?", t)
        assert len(idx) == 1 and idx[0]["record_uri"].startswith("records/analyses/2026-01-08/"), (t, idx)
    assert _q(root, "SELECT COUNT(*) AS n FROM ordered_fact_record WHERE table_name = 'analysis'")[0]["n"] == 2       # both headers
    assert _rebuild_equal(root, tmp_path, shuffle_seed=52) == {"analysis": 1, "analysis_input": 1, "analysis_artifact": 1}
    assert _rebuild_equal(root, tmp_path, shuffle_seed=53) == {"analysis": 1, "analysis_input": 1, "analysis_artifact": 1}
    # 3. A's CHILDREN arrive before A's header while B is kept: they name a filing the registry has not folded -- dependency refusals on
    #    the filing, re-selected every cycle and untouched while it is absent -- and nothing is attached; the cycle A's header lands,
    #    A displaces B, B's children are withdrawn, and the SAME cycle's retry accepts A's waiting children
    root, ids = _root(tmp_path, "reg3")
    hsha_b, rsha_b, _ = _report_filing(root, ids, "2026-01-07", 1, 820, "b")
    H.ingest_once(root, workers=0, min_free=0, now=synth.t(900))
    hsha_a, rsha_a, paths_a = _report_filing(root, ids, "2026-01-08", 1, 810, "a")
    held = str(tmp_path / "a-header-held.jsonl")
    os.replace(paths_a[1], held)                                                           # A's header is not delivered yet
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(901))
    assert rep.refused == 2 and rep.analysis_records == 0 and _links(root) == [{"artifact_id": rsha_b, "filing_sha256": hsha_b}]
    waiting = _q(root, "SELECT record_type, field, reason FROM record_refusal ORDER BY record_type")
    assert [r["record_type"] for r in waiting] == ["analysis_artifact", "analysis_input"]
    for r in waiting:
        assert r["field"] == "filing_sha256" and H.dependency_refusal_parent(r["reason"]) == (H.FILING_PARENT, hsha_a) and "has not folded" in r["reason"]
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(902))
    assert rep.dependency_refusals_retried == 2 and rep.dependency_refusals_resolved == 0 and rep.refused == 0 and rep.filing_children_rejudged == 0
    shutil.copy2(held, paths_a[1])
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(903))
    assert rep.analysis_records == 3 and rep.refused == 3 and rep.filing_children_rejudged == 2
    assert rep.dependency_refusals_retried == 2 and rep.dependency_refusals_resolved == 2
    assert _links(root) == [{"artifact_id": rsha_a, "filing_sha256": hsha_a}]
    assert _q(root, "SELECT filing_sha256 FROM analysis_input") == [{"filing_sha256": hsha_a}]
    _assert_child_refusals_name_the_refused_filing(root, hsha_b, hsha_a)
    assert set(r["record_sha256"] for r in _refusals(root, "analysis")) == {hsha_b} and _count(root, "record_refusal") == 3
    assert _rebuild_equal(root, tmp_path, shuffle_seed=54) == {"analysis": 1, "analysis_input": 1, "analysis_artifact": 1}
    assert _rebuild_equal(root, tmp_path, shuffle_seed=55) == {"analysis": 1, "analysis_input": 1, "analysis_artifact": 1}
    # 4. a filing of A's SUBSTANCE under a later stamp (a re-run of the command: another filed_utc, another header sha) is a duplicate of
    #    the kept filing, and its children are the kept analysis's: its link to a new figure folds, its input is a duplicate of A's
    hsha_a2, _rsha, paths_a2 = _report_filing(root, ids, "2026-01-09", 1, 830, "a", header_over={"filed_utc": synth.t(830)})
    fig = hashlib.sha256(b"figure z").hexdigest()
    d = os.path.join(root, "records", "analyses", "2026-01-09")
    art = {"artifact_id": fig, "kind": "figure", "uri_primary": "/synthetic/a/z.png", "locations_json": [], "bytes": 3,
           "created_utc": synth.t(829), "verified_utc": synth.t(829), "extra": {"analysis": "study.capture"}}
    R.write_records(os.path.join(d, "000005-artifact.jsonl"), [synth.rec("artifact", art, synth.t(829))])
    link = {"analysis_id": AID, "filing_sha256": hsha_a2, "artifact_id": fig, "role": "figure", "extra": {}}
    R.write_records(os.path.join(d, "000006-analysis_artifact.jsonl"), [synth.rec("analysis_artifact", link, synth.t(833))])
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(904))
    assert rep.refused == 0 and rep.analysis_records == 1 and rep.duplicates >= 3                 # the figure link; the header, input and report link duplicates
    assert _links(root) == sorted([{"artifact_id": rsha_a, "filing_sha256": hsha_a}, {"artifact_id": fig, "filing_sha256": hsha_a2}], key=lambda r: r["artifact_id"])
    assert _q(root, "SELECT record_sha256 FROM analysis")[0]["record_sha256"] == hsha_a and _count(root, "record_refusal") == 3
    assert _rebuild_equal(root, tmp_path, shuffle_seed=56) == {"analysis": 1, "analysis_input": 1, "analysis_artifact": 2}
    # 5. a child naming the header of ANOTHER analysis is refused by name -- nothing to wait for, never re-selected
    other = "22222222-3333-5444-8555-666666666666"
    hsha_o, _r, _p = _report_filing(root, ids, "2026-01-10", 1, 840, "o", run_id=ids["run_ids"][1], aid=other)
    stray = {"analysis_id": AID, "filing_sha256": hsha_o, "artifact_id": fig, "role": "page", "extra": {}}
    _, sha_stray = _child_filing(root, "2026-01-11", 1, 850, "analysis_artifact", stray)
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(905))
    assert rep.refused == 1 and _count(root, "analysis") == 2 and _count(root, "analysis_artifact") == 3
    row = _q(root, "SELECT field, reason FROM record_refusal WHERE record_sha256 = ?", sha_stray)[0]
    assert row["field"] == "filing_sha256" and f"a filing of analysis {other}, not of {AID}" in row["reason"] and H.dependency_refusal_parent(row["reason"]) is None
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(906))
    assert rep.dependency_refusals_retried == 0 and rep.refused == 0
    assert _rebuild_equal(root, tmp_path, shuffle_seed=57) == {"analysis": 2, "analysis_input": 2, "analysis_artifact": 3}
    # the CLI binds its own children: every input and link it writes carries the header's record sha, read back from the header file
    root, ids = _root(tmp_path, "reg4")
    report = tmp_path / "report4.json"
    report.write_text('{"delta": 0.4}', encoding="utf-8")
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, _spec(ids, report={"uri": str(report), "sha256": None}), "spec4.json")]) == 0
    day = R.utc_now()[:10]
    (filing_name,) = os.listdir(os.path.join(root, "records", "analyses", day))          # the filing's own directory (the seventh closing pass)
    d = os.path.join(root, "records", "analyses", day, filing_name)
    hsha = _header_sha_of(os.path.join(d, "000002-analysis.jsonl"))
    assert hsha == AN.filing_sha_of(os.path.join(d, "000002-analysis.jsonl")) and filing_name == AN.filing_dir_name(hsha)
    for fn in ("000003-analysis_input.jsonl", "000004-analysis_artifact.jsonl"):
        for _line, rec, _sha in R.read_record_file(os.path.join(d, fn)):
            assert rec["payload"]["filing_sha256"] == hsha, fn
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and _count(root, "analysis_input") == 4 and _count(root, "analysis_artifact") == 1
    assert {r["filing_sha256"] for r in _q(root, "SELECT filing_sha256 FROM analysis_input UNION ALL SELECT filing_sha256 FROM analysis_artifact")} == {hsha}
    assert _rebuild_equal(root, tmp_path, shuffle_seed=58) == {"analysis": 1, "analysis_input": 4, "analysis_artifact": 1}


# --------------------------------------------------------------------------------------------
# 18: the dependency retry moves each part's acceptances once
# --------------------------------------------------------------------------------------------


def test_the_dependency_retry_moves_each_parts_acceptances_once_so_a_file_with_a_standing_refusal_never_reads_ok(tmp_path, monkeypatch):
    """`_retry_refused_file` subtracted its CUMULATIVE accepted count from `record_delivery.n_refused` after every part, so a file
    read in two or more parts with one acceptance in each had the second part subtract the first acceptance again -- `n_refused` 0
    and status `ok` while a refusal of the file still stood. Now each part moves its own acceptances alone."""
    root, ids = _root(tmp_path)
    monkeypatch.setattr(H, "ITEM_ROWS_PER_TRANSACTION", 2)              # a record file is read in parts of two records: five inputs are three parts
    present = ids["run_ids"]
    missing = [synth.run_records(f"synthstore_late{i}", 900 + 10 * i, "none", "none", 2)[0] for i in range(3)]   # runs NOT yet in the registry
    d = os.path.join(root, "records", "analyses", "2026-01-05")
    header = synth.rec("analysis", _analysis_payload(filed_utc=synth.t(901)), synth.t(901))
    R.write_records(os.path.join(d, "000001-analysis.jsonl"), [header])
    hsha = R.record_sha(header)

    def inp(rid):
        return {"analysis_id": AID, "filing_sha256": hsha, "input_kind": "run", "input_id": rid, "run_id": rid, "object_id": None}
    # ONE file of five inputs, in parts of two: [missing 0, present 0], [missing 1, present 1], [missing 2]
    order = [missing[0], present[0], missing[1], present[1], missing[2]]
    R.write_records(os.path.join(d, "000002-analysis_input.jsonl"), [synth.rec("analysis_input", inp(r), synth.t(902)) for r in order])
    uri = "records/analyses/2026-01-05/000002-analysis_input.jsonl"

    def delivery():
        return _q(root, "SELECT n_ok, n_refused, n_blocked, n_duplicate, status FROM record_delivery WHERE file_uri = ?", uri)[0]
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(950))
    assert rep.refused == 3 and _count(root, "analysis_input") == 2
    assert delivery() == {"n_ok": 2, "n_refused": 3, "n_blocked": 0, "n_duplicate": 0, "status": "partial"}
    # two of the three parents fold -- one per part -- and the third stays absent: the retry accepts one record in each of two parts
    assert _new_run(root, "synthstore_late0", 900) == missing[0] and _new_run(root, "synthstore_late1", 910) == missing[1]
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(960))
    assert rep.dependency_refusals_retried == 3 and rep.dependency_refusals_resolved == 2 and _count(root, "analysis_input") == 4
    standing = _q(root, "SELECT record_sha256, reason FROM record_refusal WHERE record_type = 'analysis_input'")
    assert len(standing) == 1 and H.dependency_refusal_parent(standing[0]["reason"]) == ("run", missing[2])
    # the delivery audit says what the file's records say: four ok, one refused, status partial -- never `ok` with a refusal standing
    assert delivery() == {"n_ok": 4, "n_refused": 1, "n_blocked": 0, "n_duplicate": 0, "status": "partial"}
    assert _rebuild_equal(root, tmp_path, shuffle_seed=61) == {"analysis": 1, "analysis_input": 4, "analysis_artifact": 0}
    # the last parent folds: the file reads `ok` exactly when nothing of it is refused
    assert _new_run(root, "synthstore_late2", 920) == missing[2]
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(970))
    assert rep.dependency_refusals_retried == 1 and rep.dependency_refusals_resolved == 1 and _count(root, "analysis_input") == 5
    assert delivery() == {"n_ok": 5, "n_refused": 0, "n_blocked": 0, "n_duplicate": 0, "status": "ok"}
    assert _q(root, "SELECT COUNT(*) AS n FROM record_refusal WHERE record_type = 'analysis_input'")[0]["n"] == 0
    assert _rebuild_equal(root, tmp_path, shuffle_seed=62) == {"analysis": 1, "analysis_input": 5, "analysis_artifact": 0}


# --------------------------------------------------------------------------------------------
# the sixth closing pass (2026-10-09): one input set is one hash; a budget spent in the late phase is the cycle's stop
# --------------------------------------------------------------------------------------------


def test_a_re_filing_with_its_input_ids_in_another_order_is_a_duplicate_of_the_filing_never_a_conflicting_re_filing(tmp_path, capsys):
    """`analysis_id` sorted the input lists and `spec_sha256` hashed them as the spec listed them, so a re-filing of one analysis with
    two run ids swapped had the kept id and another `extra.spec_sha256` -- a conflicting re-filing, its header refused by name and
    every input and link bound to it with it. Now the normalized spec's input lists are sorted: one input set is one hash whatever
    order a spec lists the ids in, and the re-filing is a duplicate. The artifacts (the first entry of a digest carries the link) and
    the cluster job ids keep the spec's order."""
    root, ids = _root(tmp_path)
    page = _page_dir(tmp_path)
    spec = _spec(ids, page)
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec)]) == 0
    capsys.readouterr()
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(900))
    assert rep.refused == 0 and rep.analysis_records == 1 + 4 + 4
    head = _q(root, "SELECT record_sha256, json_extract(extra_json, '$.spec_sha256') AS spec_sha FROM analysis")[0]
    # the same spec with the run ids and the object ids in the other order: one normalized spec, one hash, one id
    swapped = dict(spec, inputs={"runs": list(reversed(spec["inputs"]["runs"])), "objects": list(reversed(spec["inputs"]["objects"]))})
    assert swapped["inputs"] != spec["inputs"]
    norm, norm_swapped = AN.normalize_spec(spec), AN.normalize_spec(swapped)
    assert norm == norm_swapped and norm["inputs"] == {"runs": sorted(spec["inputs"]["runs"]), "objects": sorted(spec["inputs"]["objects"])}
    assert AN.spec_sha256(norm) == AN.spec_sha256(norm_swapped) == head["spec_sha"] and AN.analysis_id(norm) == AN.analysis_id(norm_swapped)
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, swapped, "swapped.json"), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["analysis_id"] == AN.analysis_id(norm) and [i["id"] for i in out["inputs"]] == sorted(spec["inputs"]["runs"]) + sorted(spec["inputs"]["objects"])
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(910))
    # the header a duplicate of the kept filing, its inputs and links duplicates of the kept filing's children: nothing refused, nothing moved
    assert rep.refused == 0 and rep.analysis_records == 0 and rep.duplicates >= 1 + 4 + 4 and rep.refilings_rejudged == 0, rep.as_dict()
    assert _count(root, "analysis") == 1 and _count(root, "analysis_input") == 4 and _count(root, "analysis_artifact") == 4
    row = _q(root, "SELECT record_sha256, json_extract(extra_json, '$.spec_sha256') AS spec_sha FROM analysis")[0]
    assert row == head
    assert _q(root, "SELECT COUNT(*) AS n FROM record_refusal WHERE record_type IN ('analysis', 'analysis_input', 'analysis_artifact')")[0]["n"] == 0
    assert _q(root, "SELECT COUNT(*) AS n FROM ordered_fact_record WHERE table_name = 'analysis'")[0]["n"] == 2        # both filings indexed
    assert _rebuild_equal(root, tmp_path, shuffle_seed=71) == {"analysis": 1, "analysis_input": 4, "analysis_artifact": 4}
    # the order of the artifacts and of the cluster job ids has meaning and stands as the spec gives it
    two = _spec(ids, page, cluster_job_ids=["7", "3"], artifacts=[{"path": os.path.join(page, "numbers.json"), "role": "report"},
                                                                  {"dir": page, "role": "page", "url": "https://example.invalid/figs/study/"}])
    n2 = AN.normalize_spec(two)
    assert n2["cluster_job_ids"] == ["7", "3"] and [("path" in a, a["role"]) for a in n2["artifacts"]] == [(True, "report"), (False, "page")]
    assert AN.spec_sha256(n2) != AN.spec_sha256(AN.normalize_spec(dict(two, cluster_job_ids=["3", "7"])))     # another job-id order: another hash


def _cycle_line(root, now):
    with open(os.path.join(root, "state", "metrics.jsonl"), encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    (cycle,) = [r for r in rows if r.get("kind") == "cycle" and r.get("utc") == now]
    return cycle


def _cycle_state(root):
    with open(os.path.join(root, "state", "harvest_cycle.json"), encoding="utf-8") as fh:
        return json.load(fh)


def test_a_budget_spent_before_or_inside_the_late_phase_stops_the_cycle_by_name_and_counts_the_filing_files_it_left(tmp_path, monkeypatch):
    """The late phase's `ingest_parsed` result -- (the files taken, whether the budget stopped it) -- was discarded, so a cycle whose
    budget ran out before or inside the analysis filings reported no stop and nothing deferred while the filings stayed unread: under
    sustained budget pressure the harvester's health said no deferral. Now a stop inside the late phase is the cycle's stop
    (`stopped: budget`, the cycle state and the metrics line with it) and the files not taken count under
    `plain_files_deferred_budget`; the next cycle folds them."""
    root, ids = _root(tmp_path)
    page = _page_dir(tmp_path)
    spec = _spec(ids, page)
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec)]) == 0
    index_sha = PUB.sha256_of(os.path.join(page, "index.html"))
    # (a) the budget spent BEFORE the late phase, the filing's four files the only pending work: every one of them left, by name
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(900), budget_s=0.0)
    assert rep.stopped == "budget" and rep.plain_files_deferred_budget == 4 and rep.revisions_deferred_budget == 0, rep.as_dict()
    assert rep.analysis_records == 0 and rep.refused == 0 and _count(root, "analysis") == 0
    assert _q(root, "SELECT COUNT(*) AS n FROM artifact WHERE artifact_id = ?", index_sha)[0]["n"] == 0
    assert _cycle_state(root)["stopped"] == "budget" and _cycle_state(root)["plain_files_deferred_budget"] == 4
    line = _cycle_line(root, synth.t(900))
    assert line["stopped"] == "budget" and line["plain_files_deferred"] == 4 and line["bundles_deferred"] == 0
    # (b) the budget spent INSIDE the late phase: one file per chunk, ten seconds each, a five-second budget -- the first file (the
    # artifact records) is taken and committed, the header and its children are left and counted, the cycle names its stop
    clock = {"t": 0.0}
    inner = H.Ingestor.ingest_chunk

    def chunk_then_ten_seconds(self, chunk, now):
        inner(self, chunk, now)
        clock["t"] += 10.0
    monkeypatch.setattr(H.Ingestor, "ingest_chunk", chunk_then_ten_seconds)
    monkeypatch.setattr(H, "FILES_PER_TRANSACTION", 1)
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(910), budget_s=5.0, clock=lambda: clock["t"])
    monkeypatch.undo()
    assert rep.stopped == "budget" and rep.plain_files_deferred_budget == 3 and rep.revisions_deferred_budget == 0, rep.as_dict()
    assert rep.refused == 0 and _count(root, "analysis") == 0 and _count(root, "analysis_input") == 0
    assert _q(root, "SELECT COUNT(*) AS n FROM artifact WHERE artifact_id = ?", index_sha)[0]["n"] == 1          # the chunk taken stays committed
    assert _cycle_state(root)["stopped"] == "budget" and _cycle_state(root)["plain_files_deferred_budget"] == 3
    assert _cycle_line(root, synth.t(910))["plain_files_deferred"] == 3
    # (c) unbudgeted: the three files fold over the artifact rows the stopped cycle committed; nothing refused, nothing left
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(920))
    assert rep.stopped is None and rep.plain_files_deferred_budget == 0 and rep.refused == 0 and rep.analysis_records == 1 + 4 + 4, rep.as_dict()
    assert _count(root, "analysis") == 1 and _count(root, "analysis_input") == 4 and _count(root, "analysis_artifact") == 4
    assert _cycle_state(root)["stopped"] is None and _cycle_state(root)["plain_files_deferred_budget"] == 0
    assert _rebuild_equal(root, tmp_path, shuffle_seed=72) == {"analysis": 1, "analysis_input": 4, "analysis_artifact": 4}


# --------------------------------------------------------------------------------------------
# the seventh closing pass: a filing is published whole or not at all; a union column's shape is the validator's
# --------------------------------------------------------------------------------------------


def _staging_dirs(root):
    return [d for d, _ in AN.abandoned_stagings(root)]


def test_a_filing_interrupted_after_its_header_or_its_inputs_publishes_nothing_is_named_and_a_completed_filing_publishes_whole(tmp_path, capsys, monkeypatch):
    """The four record files were created one by one under `records/analyses/<day>/`, the tree the harvester scans, so a process
    interrupted -- or a write that failed -- after the header and before the inputs or the links left a header the next cycle accepted
    as a complete analysis with no inputs or no links, its `report_uri` and spec describing evidence the log did not hold, and no
    refusal named it. Now the files are staged under `staging/analyses/<token>/` (outside the scanned trees) and the directory is
    renamed into `records/analyses/<day>/<filing sha12>/` in ONE rename once every file is written and synced: an interrupted filing
    publishes nothing, refuses by name naming its staging directory, and every later run of the command names the abandoned staging;
    a completed filing publishes whole and rebuilds equal from `records/` alone."""
    root, ids = _root(tmp_path)
    page = _page_dir(tmp_path)
    report = tmp_path / "report.json"
    report.write_text('{"delta": 0.1}', encoding="utf-8")
    spec = _spec(ids, page, report={"uri": str(report), "sha256": None})
    now = R.utc_now()
    day = now[:10]
    with DB.reading(DB.db_path(root)) as conn:
        plan = AN.build_plan(spec, now, conn)
    n_art = _count(root, "artifact")
    real_write_many = EM.RecordWriter.write_many

    def dying_at(record_type):
        def write_many(self, rt, payloads, gz=False, written_utc=None):
            if rt == record_type:
                raise OSError(28, "No space left on device")          # the process dies here: the files written so far are on disk
            return real_write_many(self, rt, payloads, gz=gz, written_utc=written_utc)
        return write_many

    # (a) interrupted right after the header (before the inputs), then (b) after the inputs (before the links): on the base tip the header
    # (and in (b) its inputs) stood under records/analyses/ and folded as a complete analysis; now nothing of either reaches records/
    expected = {"analysis_input": ["000001-artifact.jsonl", "000002-analysis.jsonl"],
                "analysis_artifact": ["000001-artifact.jsonl", "000002-analysis.jsonl", "000003-analysis_input.jsonl"]}
    for n, (dies_at, staged_files) in enumerate(expected.items(), start=1):
        monkeypatch.setattr(EM.RecordWriter, "write_many", dying_at(dies_at))
        with pytest.raises(AN.AnalysisRefusal) as ei:
            AN.write_filing(root, plan, AN.producer_block(), now)
        monkeypatch.undo()
        msg = str(ei.value)
        stagings = AN.abandoned_stagings(root)
        assert len(stagings) == n and msg.startswith("analysis: the filing was interrupted while staged under ")
        assert "nothing published" in msg and "No space left on device" in msg and "remove the directory and re-run the filing" in msg
        newest = next(d for d, _ in stagings if d in msg)
        assert dict(stagings)[newest] == staged_files and newest.startswith(AN.staging_root(root) + os.sep)
        # the header in the staging holds the whole spec's description; none of it is in the scanned tree, so the fold sees no analysis
        (_, header, _), = R.read_record_file(os.path.join(newest, "000002-analysis.jsonl"))
        assert header["payload"]["analysis_id"] == plan.analysis_id and header["payload"]["report_uri"] == str(report)
        assert not os.path.isdir(os.path.join(root, "records", "analyses"))
        rep = H.ingest_once(root, workers=0, min_free=0)
        assert rep.refused == 0 and rep.analysis_records == 0, rep.as_dict()
        assert _count(root, "analysis") == 0 and _count(root, "analysis_input") == 0 and _count(root, "analysis_artifact") == 0
        assert _count(root, "artifact") == n_art                                             # the staged artifact records folded nowhere
    # (c) the CLI names both abandoned stagings on a dry run (stderr; stdout stays the plan), refuses nothing, and writes nothing
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec), "--dry-run"]) == 0
    captured = capsys.readouterr()
    assert "dry run: nothing written" in captured.out and "NOTE" not in captured.out
    assert "NOTE: 2 staged filing(s) never published" in captured.err and all(d in captured.err for d in _staging_dirs(root))
    assert "2 file(s) 000001-artifact.jsonl, 000002-analysis.jsonl" in captured.err and "3 file(s) 000001-artifact.jsonl, 000002-analysis.jsonl, 000003-analysis_input.jsonl" in captured.err
    assert "remove the directory by hand and re-run that filing" in captured.err
    assert not os.path.isdir(os.path.join(root, "records", "analyses")) and len(_staging_dirs(root)) == 2
    # (d) the completed filing: four files in the filing's own directory, named by the header's sha12; its own staging is gone, the two
    # abandoned ones still named; the fold takes the whole filing, the children bound to the header
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec), "--json"]) == 0
    captured = capsys.readouterr()
    js = json.loads(captured.out)
    filing_dir = os.path.join(root, "records", "analyses", day, js["filing_sha256"][:12])
    names = ["000001-artifact.jsonl", "000002-analysis.jsonl", "000003-analysis_input.jsonl", "000004-analysis_artifact.jsonl"]
    assert js["filing_dir"] == filing_dir and js["written"] == [os.path.join(filing_dir, f) for f in names] and sorted(os.listdir(filing_dir)) == names
    assert AN.filing_sha_of(js["written"][1]) == js["filing_sha256"] and os.listdir(os.path.join(root, "records", "analyses", day)) == [js["filing_sha256"][:12]]
    assert "NOTE: 2 staged filing(s) never published" in captured.err and len(_staging_dirs(root)) == 2
    for p in js["written"][2:]:
        assert all(r["payload"]["filing_sha256"] == js["filing_sha256"] for _, r, _ in R.read_record_file(p))
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.analysis_records == 1 + 4 + 5, rep.as_dict()
    assert _count(root, "analysis") == 1 and _count(root, "analysis_input") == 4 and _count(root, "analysis_artifact") == 5
    assert _q(root, "SELECT record_sha256, record_uri FROM analysis")[0] == {
        "record_sha256": js["filing_sha256"], "record_uri": f"records/analyses/{day}/{js['filing_sha256'][:12]}/000002-analysis.jsonl#1"}
    assert _rebuild_equal(root, tmp_path, shuffle_seed=73) == {"analysis": 1, "analysis_input": 4, "analysis_artifact": 5}   # records/ alone
    # the operator removes the abandoned directories by hand: the notice goes
    for d in _staging_dirs(root):
        shutil.rmtree(d)
    assert cli.main(["--root", root, "analysis", "file", "--spec", _write_spec(tmp_path, spec), "--dry-run"]) == 0
    assert "NOTE" not in capsys.readouterr().err and AN.abandoned_lines([]) == []
    # (e) the publish itself: a destination that already exists refuses by name and leaves the staging in place; otherwise one rename, the
    # published paths in sequence order, the staging gone
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "000002-analysis.jsonl").write_text("x\n", encoding="utf-8")
    (staged / "000001-artifact.jsonl").write_text("w\n", encoding="utf-8")
    final = tmp_path / "final"
    final.mkdir()
    (final / "000001-artifact.jsonl").write_text("y\n", encoding="utf-8")
    with pytest.raises(AN.AnalysisRefusal) as ei:
        AN.publish_staged(str(staged), str(final))
    assert "already exists" in str(ei.value) and str(staged) in str(ei.value) and (final / "000001-artifact.jsonl").read_text(encoding="utf-8") == "y\n"
    assert sorted(os.listdir(str(staged))) == ["000001-artifact.jsonl", "000002-analysis.jsonl"]
    final2 = tmp_path / "day" / "abcdefabcdef"
    assert AN.publish_staged(str(staged), str(final2)) == [str(final2 / "000001-artifact.jsonl"), str(final2 / "000002-analysis.jsonl")]
    assert not os.path.exists(str(staged)) and (final2 / "000002-analysis.jsonl").read_text(encoding="utf-8") == "x\n"


SHA_B = "b" * 64


def test_an_artifact_records_locations_json_in_any_shape_but_a_list_is_refused_by_name_marker_or_none_so_no_location_is_lost_to_the_union(tmp_path):
    """A hand-written `artifact` record carrying `locations_json` as one JSON location OBJECT (not a list) passed the validator
    whenever it carried no `extra.union_columns` marker (the marker's own check was the only shape check), and the fold's union read
    the non-list as NO entries -- the artifact accepted, its one location gone, no refusal. Now the validator checks the shape of
    every union column on every record of its table, marker or none: a non-null `locations_json` that is not a JSON list -- an object,
    a scalar, a string encoding either -- is refused by name, naming the column; a list, an empty list and a null are admitted.
    Whole-log audit at the rule's landing (read-only): every live artifact record carries a list, so replay is unchanged."""
    uri = "/cluster/study/report.json"
    loc = {"uri": uri, "verified_utc": None, "bytes": None, "substrate": "cluster_a"}
    base = {"artifact_id": SHA_A, "kind": "report", "uri_primary": uri}
    # the fold's reading the rule guards: a non-list unions as nothing (so the row would have stood with no location)
    assert RED.union_json_list(None, canonical_json(loc)) == "[]" and RED.union_json_list(None, "3") == "[]"
    # admitted: a list (as a list or as JSON text), an empty list, a null, an omitted column -- marker or none
    for ok in ({**base, "locations_json": [loc]}, {**base, "locations_json": canonical_json([loc])}, {**base, "locations_json": []},
               {**base, "locations_json": None}, dict(base), {**base, "locations_json": [loc], "extra": {"union_columns": ["locations_json"]}}):
        v = _validate(ok)
        assert isinstance(v, VAL.Validated), (ok, v)
    # refused by name, the COLUMN named: an object, an object as JSON text, a number, a JSON string literal, a bare path that is not JSON;
    # with the marker too (the column's shape before the marker's)
    cases = [({**base, "locations_json": loc}, "got an object"), ({**base, "locations_json": canonical_json(loc)}, "got an object"),
             ({**base, "locations_json": 3}, "got a int"), ({**base, "locations_json": json.dumps(uri)}, "got a str"),
             ({**base, "locations_json": uri}, "is not valid JSON"),
             ({**base, "locations_json": loc, "extra": {"union_columns": ["locations_json"]}}, "got an object")]
    for payload, tail in cases:
        m = _validate(payload)
        assert isinstance(m, VAL.Malformed) and m.field == "locations_json" and tail in m.reason, (payload, getattr(m, "reason", m))
        if "JSON list" in m.reason:
            assert "fold unions the column on every artifact record" in m.reason and "location had vanished" in m.reason
    # end to end: the finding's record beside a well-formed one in a hand-written publish file -- the malformed one refused by name (a fact),
    # the well-formed one's uri kept, the rows the same live and on rebuild
    root, _ = _root(tmp_path)
    good = {"artifact_id": SHA_B, "kind": "report", "uri_primary": "/cluster/study/other.json",
            "locations_json": [{"uri": "/cluster/study/other.json", "verified_utc": None, "bytes": None, "substrate": "cluster_a"}]}
    R.write_records(os.path.join(root, "records", "publish", "2026-01-05", "000001-artifact.jsonl"),
                    [synth.rec("artifact", {**base, "locations_json": loc}, synth.t(700)), synth.rec("artifact", good, synth.t(701))])
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(900))
    assert rep.refused == 1, rep.as_dict()
    refused = _q(root, "SELECT record_type, field, reason FROM record_refusal WHERE record_type = 'artifact'")
    assert len(refused) == 1 and refused[0]["field"] == "locations_json" and "must carry a JSON list" in refused[0]["reason"] and "got an object" in refused[0]["reason"]
    assert _q(root, "SELECT COUNT(*) AS n FROM artifact WHERE artifact_id = ?", SHA_A)[0]["n"] == 0
    assert json.loads(_q(root, "SELECT locations_json FROM artifact WHERE artifact_id = ?", SHA_B)[0]["locations_json"]) == good["locations_json"]
    assert _q(root, "SELECT COUNT(*) AS n FROM record_refusal_fact WHERE record_type = 'artifact' AND field = 'locations_json'")[0]["n"] == 1
    assert _rebuild_equal(root, tmp_path, shuffle_seed=74) == {"analysis": 0, "analysis_input": 0, "analysis_artifact": 0}


# --------------------------------------------------------------------------------------------
# 23: the eighth closing pass (2026-10-09) -- the delivery audit follows a record judged again after its fold
# --------------------------------------------------------------------------------------------

_DELIVERY_COLS = "status, n_records, n_ok, n_refused, n_blocked, n_duplicate"
OK1 = ("ok", 1, 1, 0, 0, 0)                        # a one-record file folded `ok`
REFUSED1 = ("refused", 1, 0, 1, 0, 0)              # ... whose record stands refused
DUPLICATE1 = ("duplicate", 1, 0, 0, 0, 1)          # ... whose record is a duplicate of the kept one


def _deliveries(db_path, prefix="records/analyses/"):
    """{file_uri: (status, n_records, n_ok, n_refused, n_blocked, n_duplicate)} of every filing file's delivery row in ``db_path``."""
    conn = DB.connect(db_path, readonly=True)
    try:
        return {str(r[0]): tuple(r[1:]) for r in conn.execute(
            f"SELECT file_uri, {_DELIVERY_COLS} FROM record_delivery WHERE file_uri LIKE ? ORDER BY file_uri", (prefix + "%",)).fetchall()}
    finally:
        conn.close()


def _refused_side(rows):
    """The arrival-order-INDEPENDENT reading of delivery rows: the status with `ok` and `duplicate` folded into `accepted`, the record
    total, the accepted total (`n_ok + n_duplicate`), the refused and the blocked counts. Which accepted record a file counted `ok` and
    which `duplicate` is the fold's arrival order -- on live ingest and on a rebuild alike -- and is not compared."""
    return {uri: ("accepted" if st in ("ok", "duplicate") else st, n, ok + dup, ref, blk) for uri, (st, n, ok, ref, blk, dup) in rows.items()}


def _filing_uris(day):
    """The four file uris `_report_filing` writes under ``day``: the artifact, the header, the input, the link."""
    return tuple(f"records/analyses/{day}/{n}" for n in ("000001-artifact.jsonl", "000002-analysis.jsonl", "000003-analysis_input.jsonl",
                                                          "000004-analysis_artifact.jsonl"))


def test_the_delivery_audit_of_a_displaced_filing_reads_refused_as_a_rebuilds_does_and_follows_every_verdict_judged_again(tmp_path):
    """The eighth closing pass (MEDIUM): filing B, later-stamped, folded first and its three files read `ok`; then A, earlier-stamped and
    conflicting, displaced it -- B's header refused by name, B's input and link withdrawn and refused -- while neither the displacement nor
    the children's re-derivation touched `record_delivery`, so B's three files kept `ok` with `n_ok` 1 and `n_refused` 0, the status page
    listed the rejected filing as ingested, and a rebuild of the same records wrote `refused` (the audit is outside rebuild equality, so
    nothing caught it). Now every verdict judged again after a file's fold moves that file's audit (`Ingestor._move_delivery`): a record
    refused leaves the accepted side, a refusal withdrawn returns to it as the apply judged the record (`ok` or `duplicate`), and the
    status is derived again by the fold's own rule (`harvest.delivery_status`)."""
    # 1. the finding: B (820) folds and is kept; A (810, another report) arrives a cycle later and displaces it. B's header, input and link
    #    files move to `refused` -- the status page says so -- and equal a shuffled rebuild's rows exactly (every record of them refused on
    #    either path); B's artifact file stands (the report is registered; the LINK to it is what was refused)
    root, ids = _root(tmp_path, "reg1")
    art_b, hdr_b, inp_b, lnk_b = _filing_uris("2026-01-07")
    _report_filing(root, ids, "2026-01-07", 1, 820, "b")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(900))
    assert rep.refused == 0 and rep.deliveries_rejudged == 0
    assert _deliveries(DB.db_path(root)) == {art_b: OK1, hdr_b: OK1, inp_b: OK1, lnk_b: OK1}
    _report_filing(root, ids, "2026-01-08", 1, 810, "a")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(901))
    assert rep.refused == 3 and rep.filing_children_rejudged == 2 and rep.deliveries_rejudged == 3
    expected = {art_b: OK1, hdr_b: REFUSED1, inp_b: REFUSED1, lnk_b: REFUSED1, **{u: OK1 for u in _filing_uris("2026-01-08")}}
    assert _deliveries(DB.db_path(root)) == expected
    recent = {r["file_uri"]: r for r in status.status_json(root)["recent_ingests"]}
    for uri in (hdr_b, inp_b, lnk_b):
        assert (recent[uri]["status"], recent[uri]["n_ok"], recent[uri]["n_refused"]) == ("refused", 0, 1), recent[uri]
    assert (recent[art_b]["status"], recent[art_b]["n_ok"]) == ("ok", 1)
    assert _deliveries(_rebuild_copy(root, tmp_path, shuffle_seed=81)) == expected
    assert _rebuild_equal(root, tmp_path, shuffle_seed=82) == {"analysis": 1, "analysis_input": 1, "analysis_artifact": 1}
    # a quiet cycle moves nothing
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(902))
    assert rep.deliveries_rejudged == 0 and _deliveries(DB.db_path(root)) == expected
    # 2. the other direction, on a re-filing: C (830, A's substance) is refused as a conflicting re-filing of the kept B (820); when A (810)
    #    arrives, B is displaced (`ok` -> `refused`) and C, judged again, is a duplicate of A -- its refusal withdrawn and its file moved
    #    `refused` -> `duplicate`, the row a rebuild writes meeting A first
    root, ids = _root(tmp_path, "reg2")
    hdr_b2 = "records/analyses/2026-01-06/000001-analysis.jsonl"
    hdr_c = "records/analyses/2026-01-07/000001-analysis.jsonl"
    hdr_a2 = "records/analyses/2026-01-08/000001-analysis.jsonl"
    _filing_pair(root, "2026-01-06", 1, 820, "B: the first filing to arrive")
    H.ingest_once(root, workers=0, min_free=0, now=synth.t(900))
    _, sha_c = _filing_pair(root, "2026-01-07", 1, 830, "A: the earliest filing, arrived last")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(901))
    assert rep.refused == 1 and rep.deliveries_rejudged == 0 and _deliveries(DB.db_path(root)) == {hdr_b2: OK1, hdr_c: REFUSED1}
    _filing_pair(root, "2026-01-08", 1, 810, "A: the earliest filing, arrived last")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(902))
    assert rep.refused == 1 and rep.refilings_rejudged == 2 and rep.deliveries_rejudged == 2         # B and C judged again; B's and C's files moved
    expected = {hdr_b2: REFUSED1, hdr_c: DUPLICATE1, hdr_a2: OK1}
    assert _deliveries(DB.db_path(root)) == expected and _count(root, "record_refusal") == 1
    assert _q(root, "SELECT COUNT(*) AS n FROM record_refusal WHERE record_sha256 = ?", sha_c)[0]["n"] == 0
    assert _deliveries(_rebuild_copy(root, tmp_path, shuffle_seed=83)) == expected
    assert _rebuild_equal(root, tmp_path, shuffle_seed=84) == {"analysis": 1, "analysis_input": 0, "analysis_artifact": 0}
    # 3. the other direction, on a filing's CHILDREN: A (810) is kept and B (820, another report) refused with its input and link; then Z
    #    (805, B's substance) arrives and takes the header -- A displaced, A's children withdrawn (`ok` -> `refused`), B a duplicate of Z
    #    and B's children authorized again (`refused` -> accepted). Live, B's children are inserted (Z's own children fold after its
    #    header, as duplicates of them); a rebuild in the log's order inserts Z's and finds B's duplicates -- the same accepted total,
    #    the same refusals, the `ok` / `duplicate` split the arrival order on either path
    root, ids = _root(tmp_path, "reg3")
    _report_filing(root, ids, "2026-01-06", 1, 810, "a")
    H.ingest_once(root, workers=0, min_free=0, now=synth.t(900))
    _report_filing(root, ids, "2026-01-07", 1, 820, "b")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(901))
    art_a, hdr_a, inp_a, lnk_a = _filing_uris("2026-01-06")
    art_b, hdr_b, inp_b, lnk_b = _filing_uris("2026-01-07")
    assert rep.refused == 3 and rep.deliveries_rejudged == 0
    assert _deliveries(DB.db_path(root)) == {art_a: OK1, hdr_a: OK1, inp_a: OK1, lnk_a: OK1, art_b: OK1, hdr_b: REFUSED1, inp_b: REFUSED1, lnk_b: REFUSED1}
    _report_filing(root, ids, "2026-01-08", 1, 805, "b")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(902))
    art_z, hdr_z, inp_z, lnk_z = _filing_uris("2026-01-08")
    # judged again: A and B against Z's header, then B's input and B's link against Z's (each a duplicate under an earlier key); moved:
    # A's header (displaced), B's header (withdrawn), A's two children (withdrawn), B's two children (accepted) -- six rows, one each
    assert rep.refused == 3 and rep.refilings_rejudged == 4 and rep.filing_children_rejudged == 4 and rep.deliveries_rejudged == 6
    live = _deliveries(DB.db_path(root))
    assert {u: live[u] for u in (hdr_a, inp_a, lnk_a)} == {hdr_a: REFUSED1, inp_a: REFUSED1, lnk_a: REFUSED1}
    assert {u: live[u] for u in (hdr_b, inp_b, lnk_b)} == {hdr_b: DUPLICATE1, inp_b: OK1, lnk_b: OK1}
    assert {u: live[u] for u in (hdr_z, inp_z, lnk_z)} == {hdr_z: OK1, inp_z: DUPLICATE1, lnk_z: DUPLICATE1}
    standing = {r["record_type"]: r["record_uri"] for r in _q(root, "SELECT record_type, record_uri FROM record_refusal")}
    assert standing == {"analysis": f"{hdr_a}#1", "analysis_input": f"{inp_a}#1", "analysis_artifact": f"{lnk_a}#1"}      # A's three, nothing of B's
    rebuilt = _deliveries(_rebuild_copy(root, tmp_path, shuffle_seed=85))
    assert _refused_side(rebuilt) == _refused_side(live)
    assert {u: rebuilt[u] for u in (hdr_a, inp_a, lnk_a)} == {hdr_a: REFUSED1, inp_a: REFUSED1, lnk_a: REFUSED1}
    assert _rebuild_equal(root, tmp_path, shuffle_seed=86) == {"analysis": 1, "analysis_input": 1, "analysis_artifact": 1}
    # 4. the dependency retry goes through the same mover: a file whose two inputs both name runs the registry does not hold reads
    #    `refused`; one parent folds and the file reads `partial` (the per-part UPDATE it replaced left `refused` standing beside `n_ok` 1,
    #    where a rebuild -- the parent folded first -- writes `partial`); the other folds and it reads `ok`
    root, ids = _root(tmp_path, "reg4")
    missing = [synth.run_records(f"synthstore_late{i}", 900 + 10 * i, "none", "none", 2)[0] for i in range(2)]
    d = os.path.join(root, "records", "analyses", "2026-01-05")
    header = synth.rec("analysis", _analysis_payload(filed_utc=synth.t(901)), synth.t(901))
    R.write_records(os.path.join(d, "000001-analysis.jsonl"), [header])
    hsha = R.record_sha(header)
    inputs = [synth.rec("analysis_input", {"analysis_id": AID, "filing_sha256": hsha, "input_kind": "run", "input_id": rid, "run_id": rid, "object_id": None},
                        synth.t(902)) for rid in missing]
    R.write_records(os.path.join(d, "000002-analysis_input.jsonl"), inputs)
    uri = "records/analyses/2026-01-05/000002-analysis_input.jsonl"
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(950))
    assert rep.refused == 2 and _deliveries(DB.db_path(root))[uri] == ("refused", 2, 0, 2, 0, 0)
    assert _new_run(root, "synthstore_late0", 900) == missing[0]
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(960))
    assert rep.dependency_refusals_retried == 2 and rep.dependency_refusals_resolved == 1 and rep.deliveries_rejudged == 1
    assert _deliveries(DB.db_path(root))[uri] == ("partial", 2, 1, 1, 0, 0)
    assert _deliveries(_rebuild_copy(root, tmp_path, shuffle_seed=87))[uri] == ("partial", 2, 1, 1, 0, 0)
    assert _new_run(root, "synthstore_late1", 910) == missing[1]
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(970))
    assert rep.dependency_refusals_resolved == 1 and rep.deliveries_rejudged == 1
    assert _deliveries(DB.db_path(root))[uri] == ("ok", 2, 2, 0, 0, 0) and _count(root, "record_refusal") == 0
    assert _rebuild_equal(root, tmp_path, shuffle_seed=88) == {"analysis": 1, "analysis_input": 2, "analysis_artifact": 0}
    # the status rule is one function, the fold's and the mover's
    assert H.delivery_status({"ok": 0, "refused": 0, "blocked": 0, "duplicate": 0}, 0) == "ok"
    assert H.delivery_status({"ok": 1, "refused": 1, "blocked": 0, "duplicate": 0}, 2) == "partial"
    assert H.delivery_status({"ok": 0, "refused": 1, "blocked": 0, "duplicate": 1}, 2) == "partial"
    assert H.delivery_status({"ok": 0, "refused": 2, "blocked": 0, "duplicate": 0}, 2) == "refused"
    assert H.delivery_status({"ok": 1, "refused": 0, "blocked": 1, "duplicate": 0}, 2) == "blocked"
    assert H.delivery_status({"ok": 0, "refused": 0, "blocked": 0, "duplicate": 2}, 2) == "duplicate"
    assert set(V.DELIVERY_STATUSES) >= {"ok", "partial", "refused", "duplicate", "blocked"}


def _carriers(root, sha):
    """{record_uri: delivery side} of every arrival of the record ``sha`` the carrier audit (`ordered_fact_carrier`) holds."""
    return {r["record_uri"]: r["delivery"] for r in _q(root, "SELECT record_uri, delivery FROM ordered_fact_carrier WHERE record_sha256 = ?", sha)}


def test_a_record_copied_to_a_second_uri_moves_every_file_that_carried_it_when_judged_again_so_no_copy_of_a_rejected_filing_reads_ok(tmp_path):
    """The mover followed `ordered_fact_record`'s ONE canonical carrier (the smallest uri of a record), so filing B, folded `ok`, then
    copied to a smaller uri (a duplicate; the index's carrier moved to the copy) and then displaced by the earlier-stamped A had the
    copy's file moved to `refused` while B's original kept `ok` with `n_ok` 1 and `n_refused` 0 -- the status page listed one rejected
    filing file as ingested -- where a rebuild of the same records wrote `refused` for both. Now every arrival of a record of the three
    tables is entered in `ordered_fact_carrier` with the side of its file's counts it stands on, and a record judged again moves EVERY
    file that carried it (`Ingestor._move_record_delivery`): the displaced header, the re-judged re-filing or child, the dependency
    refusal accepted on retry alike; a second copy of an accepted record is counted `duplicate`. The carrier audit is a record's
    standing verdict per file, so it is rebuild-equal and `verify_equal` compares it."""
    # 1. the finding: B (820) folds `ok`; B's header copied to a SMALLER uri folds as a duplicate and takes the index's carrier; A (810)
    #    displaces B -- BOTH of B's files read `refused`, the status page says so, and a shuffled rebuild writes the same rows
    root, _ = _root(tmp_path, "reg1")
    hdr_b = "records/analyses/2026-01-07/000001-analysis.jsonl"
    hdr_a = "records/analyses/2026-01-08/000001-analysis.jsonl"
    _, sha_b = _filing_pair(root, "2026-01-07", 1, 820, "B: kept until A arrives")
    H.ingest_once(root, workers=0, min_free=0, now=synth.t(900))
    small = _copy_filing(root, "2026-01-07", "2026-01-05")
    hdr_small = small.rpartition("#")[0]
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(901))
    assert rep.refused == 0 and _index_uri(root, sha_b) == small and _deliveries(DB.db_path(root)) == {hdr_small: DUPLICATE1, hdr_b: OK1}
    assert _carriers(root, sha_b) == {small: "accepted", f"{hdr_b}#1": "accepted"}
    _filing_pair(root, "2026-01-08", 1, 810, "A: the earliest, arrived last")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(902))
    assert rep.refused == 1 and rep.deliveries_rejudged == 2                      # one record, two files
    expected = {hdr_small: REFUSED1, hdr_b: REFUSED1, hdr_a: OK1}
    assert _deliveries(DB.db_path(root)) == expected and _refusal_uri(root, sha_b) == small
    assert _carriers(root, sha_b) == {small: "refused", f"{hdr_b}#1": "refused"}
    recent = {r["file_uri"]: r for r in status.status_json(root)["recent_ingests"]}
    for uri in (hdr_small, hdr_b):
        assert (recent[uri]["status"], recent[uri]["n_ok"], recent[uri]["n_refused"]) == ("refused", 0, 1), recent[uri]
    assert _deliveries(_rebuild_copy(root, tmp_path, shuffle_seed=91)) == expected
    assert _rebuild_equal(root, tmp_path, shuffle_seed=92) == {"analysis": 1, "analysis_input": 0, "analysis_artifact": 0}
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(903))            # a quiet cycle moves nothing
    assert rep.deliveries_rejudged == 0 and _deliveries(DB.db_path(root)) == expected
    # 1b. the copies in the other order: the smaller copy arrives first (`ok`) and the original later (`duplicate`); A displaces B: both refused
    root, _ = _root(tmp_path, "reg1b")
    _, sha_b = _filing_pair(root, "2026-01-07", 1, 820, "B: kept until A arrives")
    _copy_filing(root, "2026-01-07", "2026-01-05")
    os.replace(os.path.join(root, "records", "analyses", "2026-01-07", "000001-analysis.jsonl"), str(tmp_path / "b-held9.jsonl"))
    H.ingest_once(root, workers=0, min_free=0, now=synth.t(900))
    shutil.copy2(str(tmp_path / "b-held9.jsonl"), os.path.join(root, "records", "analyses", "2026-01-07", "000001-analysis.jsonl"))
    H.ingest_once(root, workers=0, min_free=0, now=synth.t(901))
    assert _deliveries(DB.db_path(root)) == {hdr_small: OK1, hdr_b: DUPLICATE1}
    _filing_pair(root, "2026-01-08", 1, 810, "A: the earliest, arrived last")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(902))
    assert rep.deliveries_rejudged == 2 and _deliveries(DB.db_path(root)) == expected
    assert _carriers(root, sha_b) == {small: "refused", f"{hdr_b}#1": "refused"}
    assert _rebuild_equal(root, tmp_path, shuffle_seed=93) == {"analysis": 1, "analysis_input": 0, "analysis_artifact": 0}
    # 2. the other direction, through the dependency retry: an input naming a run the registry does not hold is refused in its file and
    #    again in a copy at a smaller uri (two refused carriers, one refusal row naming the smaller); the run folds and the retry accepts the
    #    record from the refusal row's carrier -- the copy's file reads `ok`, the original's `duplicate` (the record applied once; which copy
    #    a rebuild counts `ok` is its arrival order), and nothing of either file stands refused
    root, ids = _root(tmp_path, "reg2")
    missing_rid = synth.run_records("synthstore_late", 900, "none", "none", 2)[0]
    d = os.path.join(root, "records", "analyses", "2026-01-07")
    header = synth.rec("analysis", _analysis_payload(filed_utc=synth.t(901)), synth.t(901))
    R.write_records(os.path.join(d, "000001-analysis.jsonl"), [header])
    inp = synth.rec("analysis_input", {"analysis_id": AID, "filing_sha256": R.record_sha(header), "input_kind": "run", "input_id": missing_rid,
                                       "run_id": missing_rid, "object_id": None}, synth.t(902))
    R.write_records(os.path.join(d, "000002-analysis_input.jsonl"), [inp])
    sha_i = R.record_sha(inp)
    inp_uri = "records/analyses/2026-01-07/000002-analysis_input.jsonl"
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(950))
    assert rep.refused == 1 and _deliveries(DB.db_path(root))[inp_uri] == REFUSED1
    small_i = _copy_filing(root, "2026-01-07", "2026-01-05", name="000002-analysis_input.jsonl")
    small_i_file = small_i.rpartition("#")[0]
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(951))
    assert rep.refused == 1 and rep.dependency_refusals_resolved == 0 and _refusal_uri(root, sha_i) == small_i
    assert _deliveries(DB.db_path(root))[small_i_file] == REFUSED1 and _carriers(root, sha_i) == {small_i: "refused", f"{inp_uri}#1": "refused"}
    assert _new_run(root, "synthstore_late", 900) == missing_rid
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(960))
    assert rep.dependency_refusals_resolved == 1 and rep.deliveries_rejudged == 2 and _count(root, "record_refusal") == 0
    live = _deliveries(DB.db_path(root))
    assert live[small_i_file] == OK1 and live[inp_uri] == DUPLICATE1
    assert _carriers(root, sha_i) == {small_i: "accepted", f"{inp_uri}#1": "accepted"}
    rebuilt = _deliveries(_rebuild_copy(root, tmp_path, shuffle_seed=94))
    assert _refused_side(rebuilt) == _refused_side(live)
    assert _rebuild_equal(root, tmp_path, shuffle_seed=95) == {"analysis": 1, "analysis_input": 1, "analysis_artifact": 0}
    # 3. a filing's CHILD refused by name, with a copy at a smaller uri, then authorized again: A (810) kept, B (820) refused with its input and
    #    link; B's input copied to a smaller uri is refused by name too (two refused carriers); Z (805, B's substance) arrives -- A displaced,
    #    B a duplicate of Z, B's children derived again from the refusal row's carrier (the copy) and accepted: the copy's file reads `ok`,
    #    B's input file `duplicate`, A's children refused -- seven rows moved, every carrier on the side the record log says
    root, ids = _root(tmp_path, "reg3")
    _report_filing(root, ids, "2026-01-06", 1, 810, "a")
    H.ingest_once(root, workers=0, min_free=0, now=synth.t(900))
    _report_filing(root, ids, "2026-01-07", 1, 820, "b")
    H.ingest_once(root, workers=0, min_free=0, now=synth.t(901))
    _art_a, hdr_a3, inp_a, lnk_a = _filing_uris("2026-01-06")
    _art_b, hdr_b3, inp_b, lnk_b = _filing_uris("2026-01-07")
    small_inp = _copy_filing(root, "2026-01-07", "2026-01-04", name="000003-analysis_input.jsonl")
    small_inp_file = small_inp.rpartition("#")[0]
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(902))
    sha_inp_b = _q(root, "SELECT record_sha256 FROM record_refusal WHERE record_type = 'analysis_input'")[0]["record_sha256"]
    assert rep.refused == 1 and _refusal_uri(root, sha_inp_b) == small_inp
    live = _deliveries(DB.db_path(root))
    assert live[small_inp_file] == REFUSED1 and live[inp_b] == REFUSED1 and _carriers(root, sha_inp_b) == {small_inp: "refused", f"{inp_b}#1": "refused"}
    _report_filing(root, ids, "2026-01-08", 1, 805, "b")
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(903))
    live = _deliveries(DB.db_path(root))
    assert rep.deliveries_rejudged == 7          # A's header, B's header, A's input and link, B's input in BOTH its files, B's link
    assert {u: live[u] for u in (hdr_a3, inp_a, lnk_a)} == {hdr_a3: REFUSED1, inp_a: REFUSED1, lnk_a: REFUSED1}
    assert (live[hdr_b3], live[small_inp_file], live[inp_b], live[lnk_b]) == (DUPLICATE1, OK1, DUPLICATE1, OK1)
    assert _carriers(root, sha_inp_b) == {small_inp: "accepted", f"{inp_b}#1": "accepted"}
    rebuilt = _deliveries(_rebuild_copy(root, tmp_path, shuffle_seed=96))
    assert _refused_side(rebuilt) == _refused_side(live)
    assert _rebuild_equal(root, tmp_path, shuffle_seed=97) == {"analysis": 1, "analysis_input": 1, "analysis_artifact": 1}
    # 4. the carrier audit is the migration's fifth table, a ledger, compared by verify_equal; its two sides are a controlled vocabulary
    assert R.LEDGER_ORDERED_CARRIERS == "ordered_fact_carrier" and "ordered_fact_carrier" in AF_TABLES and "ordered_fact_carrier" in S.LEDGER_TABLES
    assert "ordered_fact_carrier" not in H.VERIFY_EXCLUDE_TABLES and V.DELIVERY_SIDES == ("accepted", "refused")
    assert S.TABLE_BY_NAME["ordered_fact_carrier"].column("delivery").vocab == "DELIVERY_SIDES"
