"""Fold observations (observations.py; the `fold_observation` table; `registry observations summary | list`; the fold-observation migration;
2026-10-04): a record that restates `extra.serving_jobs` as a STRICT subset of the set the run's folded extra_json already held is
OBSERVED at the fold -- never refused, the fold replacing the key as it always does (replace-at-key is the ruled, universal rule) -- so
the producer whose cumulative set is the only belt (job-start registration on a same-identity rerun) learns it slipped. Pinned here, on
synthetic corpora:

1. a `run_patch` restating a strict subset is observed (the folded set, the restated set, the dropped members, the restating record, the
   record whose statement the folded set came from) and the record folds as before; a second cycle changes nothing; the CLI's two actions,
   read-only (no `state/last_run` stamp); rebuild equality with the table compared whole;
2. equal sets in any order, duplicates, a superset, a first statement and a value that is not a list on either side are not observed; a
   `run` re-registration (the Pipeline case) restating a smaller set is, as `record_type` run;
3. a run none of whose events states the key costs no statement on the table; `run` always refolds, which the detector relies on;
4. a late-arriving earlier-stamped record re-derives the key's observations (one gains members and a new source; one disappears);
5. the same record under two file names: the observation's uri follows the canonical carrier;
6. the migration: the plan on the shape one version below it is one create (with its one index) and nothing else -- `registry migrate
   --plan` on a copy prints `create table fold_observation` and no `rebuild table` line; the committed body IS the render; the two ledgers'
   CHECK lists are untouched (no record kind); the Python post-step derives the rows of the runs already folded exactly as the live path
   wrote them, idempotently;
7. the vocabulary of the module, the help and the docs section; the table's place among the ledgers; `members`.
"""
from __future__ import annotations

import json
import os
import random
import shutil

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import migrations as M
from sisteer.registry import observations as OBS
from sisteer.registry import records as R
from sisteer.registry import reducer as RED
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import vocab as V
from sisteer.registry.identity import canonical_json
from test_registry_migrations import AF_TABLES, FU4_TABLES, NTPL_TABLES, TF_VIEWS, _sans_af_drift, _sans_pes, _sans_pes_block, _strip_af_shape  # noqa: E402  -- migration 0025's, 0026's, 0029's and 0031's lines, set aside here

JOBS = ["job_a", "job_b", "job_c"]
KIND = "serving_jobs_strict_subset"
FO_DRIFT = ["table fold_observation is absent", "index idx_fold_observation_written is absent"]


def _root(tmp_path, name="reg"):
    root = str(tmp_path / name)
    H.ensure_layout(root)
    return root


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _ingest(root, now):
    rep = H.ingest_once(root, workers=0, min_free=0, now=now)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    return rep


def _q(root, sql, *params):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _rows(root):
    return _q(root, "SELECT * FROM fold_observation ORDER BY written_utc, record_sha256")


def _live_jobs(root, rid):
    (row,) = _q(root, "SELECT extra_json FROM run WHERE run_id = ?", rid)
    return json.loads(row["extra_json"]).get("serving_jobs")


def _seed(root, jobs, key="k", i0=100):
    """Dimensions plus one run through `running` in its own directory, its `run` record carrying extra.serving_jobs = jobs (none when None).
    The dims file is written once per root: a second seed on the same root must not append to a file the harvester has folded (an
    ingested record file is immutable -- the append is refused by name)."""
    dims = os.path.join(root, "records", "dims", "000001-dims.jsonl")
    if not os.path.exists(dims) and not os.path.exists(dims + ".gz"):
        _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions())
    over = {"extra": {"serving_jobs": list(jobs)}} if jobs is not None else {}
    rid, recs = synth.run_records(key, i0, through="running", **over)
    sub = os.path.join(synth.CAMPAIGN, I.run_dir_name(rid))
    _write(root, sub, "000001-run.jsonl", recs)
    return rid, sub, recs


def _patch(rid, jobs, t):
    """A `run_patch` restating the key the routed way (fields.extra_json)."""
    return synth.rec("run_patch", {"run_id": rid, "fields": {"extra_json": {"serving_jobs": jobs}}}, synth.t(t))


def _rerun(jobs, t, key="k", i0=100):
    """A `run` record of the SAME identity (the job-start registration of a rerun), its extra restating the key."""
    return synth.rec("run", synth.run_payload(key, i=i0, extra={"serving_jobs": jobs}), synth.t(t))


def _rebuild_equal(root, tmp_path, shuffle_seed=3):
    """Replay a shuffled copy of the log; every compared table equal, the observation table among them, compared whole."""
    assert "fold_observation" not in H.VERIFY_EXCLUDE_TABLES and "fold_observation" not in H.VERIFY_EXCLUDE_COLUMNS
    copy = str(tmp_path / f"copy{shuffle_seed}")
    files = list(R.iter_record_files(os.path.join(root, "records")))
    random.Random(shuffle_seed).shuffle(files)
    for f in files:
        dest = os.path.join(copy, "records", os.path.relpath(f, os.path.join(root, "records")))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(f, dest)
    out = str(tmp_path / f"rebuilt{shuffle_seed}.sqlite")
    H.rebuild(copy, out, workers=0)
    a = DB.connect(DB.db_path(root), readonly=True)
    b = DB.connect(out, readonly=True)
    try:
        diffs = H.verify_equal(a, b)
        assert "fold_observation" in diffs
        diff = {k: v for k, v in diffs.items() if v}
        assert not diff, diff
        return b.execute("SELECT COUNT(*) FROM fold_observation").fetchone()[0]
    finally:
        a.close()
        b.close()


# --------------------------------------------------------------------------------------------
# 1: the observation, the unchanged fold, the CLI, rebuild equality
# --------------------------------------------------------------------------------------------


def test_a_restated_strict_subset_is_observed_and_the_record_folds_as_it_always_did(tmp_path, capsys):
    root = _root(tmp_path)
    rid, sub, recs = _seed(root, JOBS)
    _ingest(root, synth.t(1000))
    assert _rows(root) == [] and _live_jobs(root, rid) == JOBS                              # a first statement is not an observation
    p = _patch(rid, ["job_b", "job_a"], 150)
    _write(root, sub, "000002-run_patch.jsonl", [p])
    rep = _ingest(root, synth.t(2000))
    assert rep.ok >= 1 and rep.refused == 0
    assert _live_jobs(root, rid) == ["job_b", "job_a"]                                       # replace-at-key, the record's own order: the fold is untouched
    rows = _rows(root)
    assert len(rows) == 1
    r = rows[0]
    assert (r["observation_kind"], r["run_id"], r["record_sha256"], r["record_type"]) == (KIND, rid, R.record_sha(p), "run_patch")
    assert r["record_uri"] == f"records/{sub}/000002-run_patch.jsonl" and r["written_utc"] == synth.t(150)
    assert r["folded_record_sha256"] == R.record_sha(recs[0])                                # the run record stated the folded set
    assert json.loads(r["folded_json"]) == sorted(JOBS) and json.loads(r["restated_json"]) == ["job_a", "job_b"]
    assert json.loads(r["dropped_json"]) == ["job_c"] and r["n_dropped"] == 1
    # the uri is the event's carrier in row_history
    (ev,) = _q(root, "SELECT record_uri FROM row_history WHERE table_name = 'run' AND record_sha256 = ?", R.record_sha(p))
    assert ev["record_uri"] == r["record_uri"]
    # a second cycle derives the same rows
    _ingest(root, synth.t(3000))
    assert _rows(root) == rows
    # the CLI: list / summary, text and JSON, read-only; the cycle that folded the record is its file's arrival, joined at read time
    (ri,) = _q(root, "SELECT arrival_utc, ingest_seq FROM record_ingest WHERE file_uri = ?", r["record_uri"])
    assert cli.main(["--root", root, "observations", "list"]) == 0
    out = capsys.readouterr().out
    assert "1 fold observation(s)" in out and rid in out and 'dropped 1: ["job_c"]' in out and f"arrived {ri['arrival_utc']}" in out and "folded 3 -> restated 2" in out
    assert cli.main(["--root", root, "observations", "list", "--run", rid, "--kind", KIND, "--json"]) == 0
    lst = json.loads(capsys.readouterr().out)
    assert len(lst) == 1 and lst[0]["dropped"] == ["job_c"] and lst[0]["folded"] == sorted(JOBS) and lst[0]["restated"] == ["job_a", "job_b"]
    assert lst[0]["arrival_utc"] == ri["arrival_utc"] and lst[0]["file_ingest_seq"] == ri["ingest_seq"]
    assert cli.main(["--root", root, "observations", "list", "--run", "no_such_run"]) == 0
    assert capsys.readouterr().out.startswith("0 fold observation(s)")
    assert cli.main(["--root", root, "observations", "summary", "--json"]) == 0
    summ = json.loads(capsys.readouterr().out)
    assert summ["observations"] == 1 and summ["runs"] == 1 and summ["runs_with_serving_jobs"] == 1 and summ["dropped_members"] == 1
    assert summ["by_kind"] == {KIND: 1} and summ["by_record_type"] == {"run_patch": 1}
    assert summ["latest_written_utc"] == synth.t(150) and summ["latest_arrival_utc"] == ri["arrival_utc"]
    assert summ["top_runs"] == [{"run_id": rid, "observations": 1, "dropped_members": 1, "latest_written_utc": synth.t(150)}]
    assert cli.main(["--root", root, "observations", "summary"]) == 0
    out = capsys.readouterr().out
    assert "fold observations 1 on 1 run(s), of 1 run(s) carrying extra_json.serving_jobs" in out and f"by kind: {KIND}=1" in out and f"{rid} x1" in out
    assert not os.path.exists(os.path.join(root, "state", "last_run"))
    assert cli.main(["--root", str(tmp_path / "nowhere"), "observations", "summary"]) == 1
    assert "observations summary:" in capsys.readouterr().err
    assert _rebuild_equal(root, tmp_path) == 1


# --------------------------------------------------------------------------------------------
# 2: what is NOT observed; the `run` re-registration that is
# --------------------------------------------------------------------------------------------


def test_equal_in_any_order_duplicates_a_superset_a_first_statement_and_a_non_list_are_not_observed_and_a_rerun_is(tmp_path):
    root = _root(tmp_path)
    rid, sub, recs = _seed(root, JOBS)
    patches = {110: ["job_c", "job_b", "job_a"],            # equal, another order
               111: ["job_a", "job_a", "job_b", "job_c"],   # equal, a duplicate
               112: ["job_a", "job_b", "job_c", "job_d"],   # a superset
               113: ["job_b", "job_b"],                     # OBSERVED: {b} < {a, b, c, d}; the duplicate is not a fact
               114: "x",                                    # not a list: not observed, and the folded value is no longer a list
               115: ["job_b"],                              # the prior value is not a list: nothing to compare
               116: []}                                     # OBSERVED: {} < {b}
    recs_by_t = {t: _patch(rid, jobs, t) for t, jobs in patches.items()}
    for t, rec in recs_by_t.items():
        _write(root, sub, f"{t:06d}-run_patch.jsonl", [rec])
    reruns = {117: _rerun(["job_a"], 117),                  # {a} is not a subset of {}
              118: _rerun([], 118)}                         # OBSERVED, record_type run: {} < {a}
    for t, rec in reruns.items():
        _write(root, sub, f"{t:06d}-run.jsonl", [rec])
    _ingest(root, synth.t(1000))
    assert _live_jobs(root, rid) == []                                                     # the last statement stands: the fold is the ruled one
    rows = _rows(root)
    assert [(r["written_utc"], r["record_type"]) for r in rows] == [(synth.t(113), "run_patch"), (synth.t(116), "run_patch"), (synth.t(118), "run")]
    a, b, c = rows
    assert (a["record_sha256"], a["folded_record_sha256"]) == (R.record_sha(recs_by_t[113]), R.record_sha(recs_by_t[112]))
    assert (json.loads(a["folded_json"]), json.loads(a["restated_json"]), json.loads(a["dropped_json"]), a["n_dropped"]) == \
        (["job_a", "job_b", "job_c", "job_d"], ["job_b"], ["job_a", "job_c", "job_d"], 3)
    assert (b["record_sha256"], b["folded_record_sha256"]) == (R.record_sha(recs_by_t[116]), R.record_sha(recs_by_t[115]))
    assert (json.loads(b["folded_json"]), json.loads(b["restated_json"]), json.loads(b["dropped_json"]), b["n_dropped"]) == (["job_b"], [], ["job_b"], 1)
    assert (c["record_sha256"], c["folded_record_sha256"], c["record_uri"]) == (R.record_sha(reruns[118]), R.record_sha(reruns[117]), f"records/{sub}/000118-run.jsonl")
    assert (json.loads(c["folded_json"]), json.loads(c["dropped_json"]), c["n_dropped"]) == (["job_a"], ["job_a"], 1)
    # the pure function over the key's events gives the same rows
    conn = DB.connect(DB.db_path(root), readonly=True)
    events = RED.Reducer(conn)._load_events("run", canonical_json({"run_id": rid}))
    conn.close()
    assert OBS.derive(rid, events, RED.fold("run", events)) == [{k: v for k, v in r.items()} for r in rows]
    assert _rebuild_equal(root, tmp_path) == 3


# --------------------------------------------------------------------------------------------
# 3: no statement for a key that never states the key; the refold the detector relies on
# --------------------------------------------------------------------------------------------


def test_a_run_that_never_states_the_key_costs_no_statement_and_run_always_refolds(tmp_path):
    assert OBS.OBSERVED_TABLES == ("run",) and all(t in RED._REFOLD_ALWAYS for t in OBS.OBSERVED_TABLES)
    root = _root(tmp_path)
    rid, sub, _ = _seed(root, None)
    _write(root, sub, "000002-run_patch.jsonl", [synth.rec("run_patch", {"run_id": rid, "fields": {"job_name": "plain"}}, synth.t(150))])
    conn = DB.open_or_create(DB.db_path(root))
    touched = []
    conn.set_trace_callback(lambda s: touched.append(s) if "fold_observation" in s and s.lstrip().upper().startswith(("DELETE", "INSERT", "UPDATE")) else None)
    try:
        rep = H.ingest_once(root, conn=conn, workers=0, min_free=0, now=synth.t(1000))
    finally:
        conn.set_trace_callback(None)
        conn.close()
    assert rep.refused == 0 and rep.blocked == 0 and rep.ok >= 5, rep.refusals
    assert touched == [] and _rows(root) == []
    # the same cycle shape for a run that does state it: one DELETE and one INSERT per refold of the key
    rid2, sub2, _ = _seed(root, JOBS, key="k2", i0=200)
    _write(root, sub2, "000002-run_patch.jsonl", [_patch(rid2, ["job_a"], 250)])
    conn = DB.open_or_create(DB.db_path(root))
    conn.set_trace_callback(lambda s: touched.append(s) if "fold_observation" in s and s.lstrip().upper().startswith(("DELETE", "INSERT", "UPDATE")) else None)
    try:
        rep = H.ingest_once(root, conn=conn, workers=0, min_free=0, now=synth.t(2000))
    finally:
        conn.set_trace_callback(None)
        conn.close()
    assert rep.refused == 0
    kinds = [s.lstrip().split()[0].upper() for s in touched]
    # every refold of a key that states the key replaces its rows (one DELETE); an INSERT follows only when a row was derived
    assert kinds and set(kinds) <= {"DELETE", "INSERT"} and kinds.count("INSERT") >= 1 and kinds.count("DELETE") >= kinds.count("INSERT")
    assert [r["run_id"] for r in _rows(root)] == [rid2]


# --------------------------------------------------------------------------------------------
# 4: out-of-order arrival re-derives the key's rows
# --------------------------------------------------------------------------------------------


def test_a_late_arriving_earlier_stamped_record_re_derives_the_keys_observations(tmp_path):
    # variant A: a later-arriving SUPERSET stamped between the two: the observation gains members and a new source
    root = _root(tmp_path, "a")
    rid, sub, recs = _seed(root, JOBS)
    late_sub = _patch(rid, ["job_a", "job_b"], 160)
    _write(root, sub, "000002-run_patch.jsonl", [late_sub])
    _ingest(root, synth.t(1000))
    (r,) = _rows(root)
    assert json.loads(r["dropped_json"]) == ["job_c"] and r["folded_record_sha256"] == R.record_sha(recs[0])
    sup = _patch(rid, ["job_a", "job_b", "job_c", "job_d"], 130)
    _write(root, sub, "000003-run_patch.jsonl", [sup])
    _ingest(root, synth.t(2000))
    (r,) = _rows(root)
    assert r["record_sha256"] == R.record_sha(late_sub) and r["folded_record_sha256"] == R.record_sha(sup)
    assert json.loads(r["folded_json"]) == ["job_a", "job_b", "job_c", "job_d"] and json.loads(r["dropped_json"]) == ["job_c", "job_d"] and r["n_dropped"] == 2
    assert _live_jobs(root, rid) == ["job_a", "job_b"]                                       # the fold's order: the t160 statement is the latest
    assert _rebuild_equal(root, tmp_path, shuffle_seed=1) == 1
    # variant B: a later-arriving SMALLER set stamped between the two: it is the observation now, and the t160 one is no longer a subset
    root = _root(tmp_path, "b")
    rid, sub, recs = _seed(root, JOBS)
    late_sub = _patch(rid, ["job_a", "job_b"], 160)
    _write(root, sub, "000002-run_patch.jsonl", [late_sub])
    _ingest(root, synth.t(1000))
    assert len(_rows(root)) == 1
    small = _patch(rid, ["job_a"], 120)
    _write(root, sub, "000003-run_patch.jsonl", [small])
    _ingest(root, synth.t(2000))
    (r,) = _rows(root)
    assert r["record_sha256"] == R.record_sha(small) and r["written_utc"] == synth.t(120) and r["folded_record_sha256"] == R.record_sha(recs[0])
    assert json.loads(r["dropped_json"]) == ["job_b", "job_c"]
    assert _rebuild_equal(root, tmp_path, shuffle_seed=2) == 1


# --------------------------------------------------------------------------------------------
# 5: the canonical carrier
# --------------------------------------------------------------------------------------------


def test_the_same_record_under_two_file_names_follows_the_canonical_carrier(tmp_path):
    root = _root(tmp_path)
    rid, sub, _ = _seed(root, JOBS)
    p = _patch(rid, ["job_a"], 150)
    _write(root, sub, "000003-dup-b.jsonl", [p])
    _ingest(root, synth.t(1000))
    (r,) = _rows(root)
    assert r["record_uri"] == f"records/{sub}/000003-dup-b.jsonl"
    _write(root, sub, "000002-dup-a.jsonl", [p])                                            # the same bytes under a smaller name
    rep = _ingest(root, synth.t(2000))
    assert rep.duplicates == 1
    (r,) = _rows(root)
    (ev,) = _q(root, "SELECT record_uri FROM row_history WHERE table_name = 'run' AND record_sha256 = ?", R.record_sha(p))
    assert r["record_uri"] == ev["record_uri"] == f"records/{sub}/000002-dup-a.jsonl"
    assert _rebuild_equal(root, tmp_path) == 1


# --------------------------------------------------------------------------------------------
# 6: the migration
# --------------------------------------------------------------------------------------------


def _fo_migration():
    return next(f for f in M.migration_files() if f[1] == "fold_observation")


def _shape_below_fo(path: str) -> None:
    """A database at the shape one version below the fold-observation migration: every earlier file applied by the runner itself."""
    version = _fo_migration()[0]
    conn = DB.connect(path)
    for v, slug, mpath in M.migration_files():
        if v < version:
            M._apply_one(conn, v, slug, open(mpath, encoding="utf-8").read())
    conn.commit()
    conn.close()


def test_the_migration_plan_on_the_shape_below_it_is_one_create_and_the_committed_body_is_the_render(tmp_path, capsys):
    version, slug, mpath = _fo_migration()
    # 0025 (pending_excludes_superseded) is the newer file on this tree: its one recreate of v_pending_judgment is set aside below and
    # pinned by its own test (tests/test_registry_pending_excludes_superseded.py)
    pes_version = next(f for f in M.migration_files() if f[1] == "pending_excludes_superseded")[0]
    assert pes_version > version and pes_version == M.code_version() - 7                   # 0026 (the twin fold), 0027 (registry follow-ups-4), 0028 (reference_model_never_selected), 0029 (analysis filing), 0030 (fold_order_barrier) and 0031 (non_template_positions) follow it on this tree
    path = str(tmp_path / "below.sqlite")
    _shape_below_fo(path)
    conn = DB.connect(path)
    assert M.db_version(conn) == version - 1
    assert _sans_pes(DB.schema_drift(conn)) == FO_DRIFT
    have, want = DB.ddl_master(conn), DB.code_ddl_master()
    for t in ("blocked_records", "row_history"):
        assert have[("table", t)] == want[("table", t)], t                                 # no record kind was added: the ledgers' CHECK lists are byte-identical
    plan = M.shape_plan(conn)
    assert [r["table"] for r in plan["rebuild"] if r["table"] not in FU4_TABLES + NTPL_TABLES] == [] and plan["add_columns"] == [] and plan["undeclared_tables"] == [] and [c for c in plan["create"] if c not in AF_TABLES] == ["fold_observation"]   # 0027's run_contrast rebuild, 0031's two and 0029's five creates aside
    assert plan["objects"]["index"] == {"recreate": [], "drop": []} and plan["objects"]["trigger"] == {"recreate": [], "drop": []}   # 0030's two fence triggers ride 0031's run rebuild
    assert [v for v in plan["objects"]["view"]["recreate"] if v not in TF_VIEWS] == ["v_pending_judgment"] and plan["objects"]["view"]["drop"] == []   # the newer files' recreates: 0025's one, 0026's three
    rendered = M.render_shape_migration(conn, plan)
    committed = open(mpath, encoding="utf-8").read()
    assert committed.startswith(f"-- {os.path.basename(mpath)}\n")
    body = committed[committed.index("-- drift: "):]
    assert body.strip() == _sans_pes_block(rendered).strip()                               # today's render carries the newer file's recreate too
    assert body.count("\nCREATE TABLE ") == 1 and body.count("CREATE INDEX idx_fold_observation_written") == 1
    for absent in ("__new", "INSERT INTO", "ALTER TABLE", "DROP VIEW", "CREATE VIEW", "DROP TABLE", "CREATE TRIGGER"):
        assert absent not in body, absent
    assert M.POST_APPLY["fold_observation"] is M.derive_fold_observations
    conn.close()
    # the CLI's plan on a COPY: one `create table` line, no `rebuild table` line, nothing else, exit 1 while the file is pending
    root = str(tmp_path / "copyroot")
    os.makedirs(root)
    shutil.copy(path, DB.db_path(root))
    assert cli.main(["--root", root, "migrate", "--plan"]) == 1
    out = capsys.readouterr().out
    assert f"pending: {version:04d}_fold_observation, {pes_version:04d}_pending_excludes_superseded" in out and "create table fold_observation" in out
    for absent in ("add column", "drop undeclared", "recreate index"):
        assert absent not in out, out
    assert [ln for ln in out.splitlines() if ln.startswith("recreate trigger")] == [], out   # 0030's two fence triggers ride 0031's run rebuild
    assert [ln for ln in out.splitlines() if ln.startswith("rebuild table ") and "run_contrast" not in ln and ln.split()[2].rstrip(":") not in NTPL_TABLES] == [], out   # 0027's and 0031's rebuild lines aside
    recreate_line = next(ln for ln in out.splitlines() if ln.startswith("recreate view(s): "))
    assert "v_pending_judgment" in recreate_line                                             # the newer files' recreates on one line: 0025's one, 0026's three
    assert sum(1 for ln in out.splitlines() if ln.startswith("create table") and ln.split()[-1] not in AF_TABLES) == 1   # 0029's five aside
    # the runner on the empty shape: the version rows, the table, the post-step's line with nothing to derive (0025 has no post-step)
    conn = DB.connect(path)
    notes = []
    assert M.apply_migrations(conn, notes) == list(range(version, M.code_version() + 1))     # 0024, 0025 and 0026 (the twin fold) on this tree
    assert notes == [f"{version:04d}_fold_observation: derived fold observations: 0 observation(s) on 0 run(s) (0 run(s) carrying extra_json.serving_jobs checked)"]
    assert DB.schema_drift(conn) == [] and conn.execute("SELECT description FROM schema_migration WHERE version = ?", (version,)).fetchone()[0] == "fold_observation"
    conn.close()
    DB.open_or_create(path).close()


def test_the_post_step_derives_the_rows_of_the_runs_already_folded_exactly_as_the_live_path_wrote_them(tmp_path):
    """The live path's rows on a corpus with three observations on one run and a run without the key; the table and its version row
    taken away (the shape one below, with every row the old code would have left); the runner puts them back through the post-step."""
    version = _fo_migration()[0]
    root = _root(tmp_path)
    rid, sub, _ = _seed(root, JOBS)
    # t112 a superset (not observed); t113 {b, c} < {a, b, c, d}; t116 {b} < {b, c}; t118 a `run` re-registration stating {} < {b}
    for t, jobs in ((112, ["job_a", "job_b", "job_c", "job_d"]), (113, ["job_b", "job_c"]), (116, ["job_b"])):
        _write(root, sub, f"{t:06d}-run_patch.jsonl", [_patch(rid, jobs, t)])
    _write(root, sub, "000118-run.jsonl", [_rerun([], 118)])
    rid2, sub2, _ = _seed(root, None, key="k2", i0=200)
    _ingest(root, synth.t(1000))
    live = _rows(root)
    assert len(live) == 3 and {r["run_id"] for r in live} == {rid}
    assert [(r["record_type"], json.loads(r["dropped_json"])) for r in live] == [("run_patch", ["job_a", "job_d"]), ("run_patch", ["job_c"]), ("run", ["job_b"])]
    conn = DB.connect(DB.db_path(root))
    conn.execute("DROP INDEX idx_fold_observation_written")
    conn.execute("DROP TABLE fold_observation")
    _strip_af_shape(conn)                   # 0029's five tables too: its file re-applies below and a CREATE TABLE needs them absent
    # this file's version row and every newer one (0025: its object is the code's already, so it re-applies as a no-op recreate)
    conn.execute("DELETE FROM schema_migration WHERE version >= ?", (version,))
    conn.commit()
    pes_version = next(f for f in M.migration_files() if f[1] == "pending_excludes_superseded")[0]
    assert M.db_version(conn) == version - 1 and _sans_af_drift(DB.schema_drift(conn)) == FO_DRIFT
    assert [v for v, _, _ in M.pending_migrations(conn)] == list(range(version, M.code_version() + 1))   # 0024, 0025, 0026
    notes = []
    assert M.apply_migrations(conn, notes) == list(range(version, M.code_version() + 1))     # 0024, 0025 and 0026 (the twin fold) on this tree
    assert notes == [f"{version:04d}_fold_observation: derived fold observations: 3 observation(s) on 1 run(s) (1 run(s) carrying extra_json.serving_jobs checked)"]
    assert DB.schema_drift(conn) == []
    after = [dict(r) for r in conn.execute("SELECT * FROM fold_observation ORDER BY written_utc, record_sha256").fetchall()]
    assert after == live
    # idempotent: a second derivation replaces the rows with the same rows
    assert M.derive_fold_observations(conn) == "derived fold observations: 3 observation(s) on 1 run(s) (1 run(s) carrying extra_json.serving_jobs checked)"
    assert [dict(r) for r in conn.execute("SELECT * FROM fold_observation ORDER BY written_utc, record_sha256").fetchall()] == live
    assert M.foreign_key_violations(conn, ["fold_observation"]) == []
    conn.close()
    DB.open_or_create(DB.db_path(root)).close()
    assert _rebuild_equal(root, tmp_path) == 3


# --------------------------------------------------------------------------------------------
# 7: the vocabulary, the table's place, `members`
# --------------------------------------------------------------------------------------------


def test_the_vocabulary_the_tables_place_and_the_member_rule():
    from test_registry_vocabulary import hits
    assert not hits(open(OBS.__file__, encoding="utf-8").read(), allow_prose_view=True)
    assert V.FOLD_OBSERVATION_KINDS == (KIND,) and V.VOCABULARIES["FOLD_OBSERVATION_KINDS"] == (KIND,) and KIND in V.EVIDENCE["FOLD_OBSERVATION_KINDS"]
    assert "fold_observation" in S.LEDGER_TABLES and "fold_observation" not in S.RUN_SCALE_TABLES and "fold_observation" not in S.ITEM_SCALE_TABLES
    assert "fold_observation" not in R.RECORD_TYPE_TABLE.values() and V.UNLEDGERED_FACT_TYPES == ("record_refusal",) + V.ANALYSIS_RECORD_TYPES   # no record kind of its own: the ledgers stand (0028's three kinds are unledgered too)
    assert S.TABLE_BY_NAME["fold_observation"].column("observation_kind").vocab == "FOLD_OBSERVATION_KINDS"
    assert tuple(S.TABLE_BY_NAME["fold_observation"].pk) == ("observation_kind", "run_id", "record_sha256")
    assert OBS.members(["job_b", "job_a", "job_a"]) == ["job_a", "job_b"] and OBS.members([]) == []
    assert OBS.members(None) is None and OBS.members("job_a") is None and OBS.members({"job_a": 1}) is None
    assert OBS.members([{"job": 1}, "job_a"]) == sorted([canonical_json({"job": 1}), "job_a"])
    assert OBS.derive("r", [], []) == []
