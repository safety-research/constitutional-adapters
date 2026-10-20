"""Site aliases (``sisteer.registry.site_aliases``): the private map from a site's own machine names to the registry's
neutral labels, and the three places that read it -- the validator (before its vocabulary check), the fold (before it
writes a live row) and the migration runner (``temp.site_alias`` and the slug-keyed pre-check).

Every legacy spelling here is invented (``oldsite_a``, ``oldsite_b``, ``oldhost``): the tree never spells a real one.
"""
from __future__ import annotations

import hashlib
import os

import pytest

from sisteer.registry import config as CFG
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import migrations as M
from sisteer.registry import records as R
from sisteer.registry import site_aliases as SA
from sisteer.registry import synth
from sisteer.registry import validate as VAL
from sisteer.registry import vocab as V

ALIASES = {"oldsite_a": "cluster_a", "oldsite_b": "cluster_b", "oldhost": "grading_host"}
ALIAS_TOML = '[site_aliases]\noldsite_a = "cluster_a"\noldsite_b = "cluster_b"\noldhost = "grading_host"\n'


@pytest.fixture(autouse=True)
def _empty_active_table():
    """The active table is process state: every test starts and ends on the empty table, so no other module sees ours."""
    SA.reset()
    yield
    SA.reset()


def _root(tmp_path, aliases: str = ALIAS_TOML) -> str:
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    if aliases is not None:
        with open(os.path.join(root, SA.FILE_NAME), "w", encoding="utf-8") as fh:
            fh.write(aliases)
    return root


# --------------------------------------------------------------------------------------------
# the file and the table
# --------------------------------------------------------------------------------------------


def test_the_labels_are_the_neutral_vocabulary():
    assert {"cluster_a", "cluster_b", "grading_host"} <= set(V.SUBSTRATES)
    assert set(V.MIRROR_DESTINATIONS) <= set(V.SUBSTRATES)
    assert [(t, c) for t, c, _ in SA.substrate_columns()] == [
        ("fit_search", "substrate"), ("object", "fit_substrate"), ("run", "substrate"), ("mirror_ack", "destination"), ("analysis", "substrate")]


def test_an_absent_file_is_the_empty_table_with_a_fixed_digest(tmp_path):
    root = _root(tmp_path, aliases=None)
    assert SA.load_root(root) == {}
    assert SA.digest({}) == SA.EMPTY_DIGEST == hashlib.sha256(b"{}").hexdigest()


def test_load_root_reads_the_table_and_digests_it_canonically(tmp_path):
    root = _root(tmp_path)
    table = SA.load_root(root)
    assert table == ALIASES and list(table) == sorted(ALIASES)
    shuffled = dict(reversed(list(ALIASES.items())))
    assert SA.digest(shuffled) == SA.digest(table) == hashlib.sha256(SA.canonical_json(ALIASES).encode()).hexdigest()
    assert SA.canonical_json(table) == '{"oldhost":"grading_host","oldsite_a":"cluster_a","oldsite_b":"cluster_b"}'
    assert SA.digest(table) != SA.EMPTY_DIGEST


@pytest.mark.parametrize("body, needle", [
    ('[site_aliases]\noldsite_a = "cluster_a"\n[extra]\nx = 1\n', "carries only the [site_aliases] table; found extra"),
    ('[site_alias]\noldsite_a = "cluster_a"\n', "found site_alias"),
    ('# nothing\n', "no [site_aliases] table"),
    ('[site_aliases\n', "not valid TOML"),
])
def test_load_table_refuses_a_file_that_is_not_exactly_one_alias_table(tmp_path, body, needle):
    p = tmp_path / "aliases.toml"
    p.write_text(body)
    with pytest.raises(SA.SiteAliasError) as exc:
        SA.load_table(str(p))
    assert needle in str(exc.value) and str(p) in str(exc.value)


def test_an_empty_table_header_is_the_empty_table(tmp_path):
    p = tmp_path / "aliases.toml"
    p.write_text("[site_aliases]\n")
    assert SA.load_table(str(p)) == {}


def test_validate_names_every_problem_at_once():
    bad = {"Old-Site": "cluster_a", "cluster_b": "cluster_a", "oldsite_c": "nowhere", "oldsite_d": 3, "oldsite_e": "oldsite_f",
           "oldsite_f": "cluster_a", "unverified": "cluster_a"}
    with pytest.raises(SA.SiteAliasError) as exc:
        SA.validate(bad, "probe")
    msg = str(exc.value)
    assert msg.startswith("probe: the [site_aliases] table is refused")
    assert "'Old-Site' does not match" in msg
    assert "spelling 'cluster_b' is itself a label" in msg
    assert "spelling 'unverified' is itself a label" in msg
    assert "oldsite_c = 'nowhere': 'nowhere' is not a label" in msg
    assert "oldsite_d = 3: the label must be a string" in msg
    assert "oldsite_e = 'oldsite_f': 'oldsite_f' is not a label" in msg and "aliases do not chain" in msg
    with pytest.raises(SA.SiteAliasError):
        SA.validate(["oldsite_a"])
    assert SA.validate(None) == {}


def test_a_site_aliases_table_in_config_toml_is_refused_by_name(tmp_path):
    root = _root(tmp_path, aliases=None)
    cfg = os.path.join(root, CFG.CONFIG_NAME)
    with open(cfg, "w", encoding="utf-8") as fh:
        fh.write('[registry]\nroot = "x"\n\n[site_aliases]\noldsite_a = "cluster_a"\n')
    with pytest.raises(SA.SiteAliasError) as exc:
        SA.load_root(root)
    assert cfg in str(exc.value) and "site_aliases.toml" in str(exc.value)
    with pytest.raises(CFG.ConfigError) as exc2:          # the loader every registry command runs refuses it too
        CFG.load(root)
    assert "[site_aliases]" in str(exc2.value)
    # and the refusal stands even when the alias file itself is present and valid: one source only
    with open(os.path.join(root, SA.FILE_NAME), "w", encoding="utf-8") as fh:
        fh.write(ALIAS_TOML)
    with pytest.raises(SA.SiteAliasError):
        SA.load_root(root)


def test_install_loads_a_root_once_and_rereads_a_changed_file(tmp_path, monkeypatch):
    root = _root(tmp_path)
    reads = []
    real = SA.load_root
    monkeypatch.setattr(SA, "load_root", lambda r: reads.append(r) or real(r))
    assert SA.install(root) == ALIASES
    assert SA.install({"registry": {"root": root}}) == ALIASES      # a configuration names its root
    assert len(reads) == 1
    table, dig, source = SA.active()
    assert table == ALIASES and dig == SA.digest(ALIASES) and source == os.path.join(root, SA.FILE_NAME)
    with open(os.path.join(root, SA.FILE_NAME), "w", encoding="utf-8") as fh:
        fh.write('[site_aliases]\noldsite_a = "cluster_b"\n\n')
    assert SA.install(root) == {"oldsite_a": "cluster_b"} and len(reads) == 2
    other = _root(tmp_path / "x", aliases=None)
    assert SA.install(other) == {} and SA.active()[1:] == (SA.EMPTY_DIGEST, None)


def test_install_takes_a_loaded_table_or_none_as_well_as_a_root():
    assert SA.install(dict(ALIASES)) == ALIASES and SA.installed() == ALIASES
    assert SA.active()[1:] == (SA.digest(ALIASES), "<explicit>")
    with pytest.raises(SA.SiteAliasError):
        SA.install({"oldsite_a": "nowhere"})
    assert SA.installed() == ALIASES                               # a refused table leaves the active one as it was
    assert SA.install(None) == {} and SA.installed() == {} and SA.active()[1] == SA.EMPTY_DIGEST


def test_canonical_maps_only_through_the_active_table():
    assert SA.canonical("oldsite_a") == "oldsite_a"                  # the empty table maps nothing
    SA.install_table(ALIASES, "probe")
    assert [SA.canonical(v) for v in ("oldsite_a", "oldhost", "cluster_a", "modal", "nowhere", None, 3)] == [
        "cluster_a", "grading_host", "cluster_a", "modal", "nowhere", None, 3]


def test_legacy_keys_and_get_compat_read_a_persisted_key_under_either_spelling():
    SA.install_table(ALIASES)
    assert list(SA.legacy_keys("verified_cluster_a")) == ["verified_cluster_a", "verified_oldsite_a"]
    assert list(SA.legacy_keys("cluster_a_path")) == ["cluster_a_path", "oldsite_a_path"]
    assert list(SA.legacy_keys("grading_host_mirror_present")) == ["grading_host_mirror_present", "oldhost_mirror_present"]
    assert list(SA.legacy_keys("cluster_a")) == ["cluster_a", "oldsite_a"]
    assert list(SA.legacy_keys("cluster_ab")) == ["cluster_ab"]                      # a label only as a whole token
    assert set(SA.legacy_keys("cluster_a_to_cluster_b")) == {
        "cluster_a_to_cluster_b", "oldsite_a_to_cluster_b", "cluster_a_to_oldsite_b", "oldsite_a_to_oldsite_b"}
    persisted = {"oldsite_a_path": "/x", "oldsite_a": {"pointer": 7}, "cluster_b": {"pointer": 9}}
    assert SA.get_compat(persisted, "cluster_a_path") == "/x"
    assert SA.get_compat(persisted, "cluster_a") == {"pointer": 7}
    assert SA.get_compat(persisted, "cluster_b") == {"pointer": 9}
    assert SA.get_compat(persisted, "grading_host_drive", "absent") == "absent"
    persisted["cluster_a"] = {"pointer": 8}                                           # the label's own spelling wins
    assert SA.get_compat(persisted, "cluster_a") == {"pointer": 8}
    SA.reset()
    assert SA.get_compat({"oldsite_a_path": "/x"}, "cluster_a_path") is None         # no table, no legacy reading


def test_canonical_row_maps_the_label_columns_only():
    row = {"run_id": "r1", "substrate": "oldsite_a", "notes": "oldsite_a"}
    assert SA.canonical_row("run", row) is row                                       # nothing to map with the empty table
    SA.install_table(ALIASES)
    out = SA.canonical_row("run", row)
    assert out == {"run_id": "r1", "substrate": "cluster_a", "notes": "oldsite_a"} and row["substrate"] == "oldsite_a"
    assert SA.canonical_row("mirror_ack", {"destination": "oldsite_b"}) == {"destination": "cluster_b"}
    assert SA.canonical_row("object", {"fit_substrate": None}) == {"fit_substrate": None}
    assert SA.canonical_row("judgment", {"x": "oldsite_a"}) == {"x": "oldsite_a"}


# --------------------------------------------------------------------------------------------
# the migration runner's alias table
# --------------------------------------------------------------------------------------------


def _with_probe_migration(monkeypatch, tmp_path, slug: str, body: str) -> int:
    real = M.migration_files()
    n = len(real) + 1
    path = tmp_path / f"{n:04d}_{slug}.sql"
    path.write_text(body)
    monkeypatch.setattr(M, "migration_files", lambda: real + [(n, slug, str(path))])
    monkeypatch.setattr(M, "code_version", lambda: n)
    return n


def test_the_runner_fills_temp_site_alias_before_a_pending_file_and_drops_it_after(tmp_path, monkeypatch):
    conn = DB.connect_memory()
    M.apply_migrations(conn)
    n = _with_probe_migration(monkeypatch, tmp_path, "alias_probe", "CREATE TABLE alias_probe AS SELECT old, new FROM temp.site_alias;\n")
    assert M.apply_migrations(conn, site_aliases=ALIASES) == [n]
    assert sorted(tuple(r) for r in conn.execute("SELECT old, new FROM alias_probe").fetchall()) == SA.pairs(ALIASES)
    assert conn.execute("SELECT COUNT(*) FROM sqlite_temp_master WHERE name = 'site_alias'").fetchone()[0] == 0
    assert "site_alias" not in {k[1] for k in DB.ddl_master(conn)}


def test_the_runner_defaults_to_the_empty_table_and_skips_it_with_nothing_pending(tmp_path, monkeypatch):
    conn = DB.connect_memory()
    M.apply_migrations(conn)
    assert M.apply_migrations(conn, site_aliases={"Bad Key": "x"}) == []            # nothing pending: nothing read
    n = _with_probe_migration(monkeypatch, tmp_path, "alias_probe", "CREATE TABLE alias_probe AS SELECT old, new FROM temp.site_alias;\n")
    with pytest.raises(SA.SiteAliasError):
        M.apply_migrations(conn, site_aliases={"Bad Key": "x"})                      # refused before any file
    assert M.db_version(conn) == n - 1
    assert M.apply_migrations(conn) == [n]
    assert conn.execute("SELECT COUNT(*) FROM alias_probe").fetchone()[0] == 0


def test_a_pre_check_runs_before_its_file_and_refuses_by_name(tmp_path, monkeypatch):
    conn = DB.connect_memory()
    M.apply_migrations(conn)
    seen = []

    def check(c, table):
        seen.append(dict(table))
        raise M.MigrationFailed("probe refused")
    n = _with_probe_migration(monkeypatch, tmp_path, "alias_probe", "CREATE TABLE alias_probe (x);\n")
    monkeypatch.setitem(M.PRE_CHECK, "alias_probe", check)
    with pytest.raises(M.MigrationFailed, match="probe refused"):
        M.apply_migrations(conn, site_aliases=ALIASES)
    assert seen == [ALIASES] and M.db_version(conn) == n - 1
    assert conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name = 'alias_probe'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM sqlite_temp_master WHERE name = 'site_alias'").fetchone()[0] == 0
    assert "site_label_rename" in M.PRE_CHECK


# --------------------------------------------------------------------------------------------
# the validator and the fold
# --------------------------------------------------------------------------------------------


def _legacy_run(run_key: str, i0: int, through: str = "queued", substrate: str = "oldsite_a"):
    """A run as code from before the neutral labels wrote it: the legacy spelling in the run and in the producer block."""
    return synth.run_records(run_key, i0, through=through, prod=synth.producer(substrate=substrate), substrate=substrate)


def test_the_validator_checks_the_label_and_keeps_the_spelling():
    _, recs = _legacy_run("vprobe", 10)
    v = VAL.validate_record(recs[0], "probe")
    assert isinstance(v, VAL.Malformed) and v.field == "producer.substrate" and "'oldsite_a' not in SUBSTRATES" in v.reason
    SA.install_table(ALIASES)
    v = VAL.validate_record(recs[0], "probe")
    assert isinstance(v, VAL.Validated)
    assert v.payload["substrate"] == "oldsite_a" and v.producer["substrate"] == "oldsite_a"      # the record as written
    unknown = synth.run_records("vprobe2", 11, through="queued", substrate="oldsite_z")[1][0]
    bad = VAL.validate_record(unknown, "probe")
    assert isinstance(bad, VAL.Malformed) and bad.field == "substrate" and "'oldsite_z' is not in SUBSTRATES" in bad.reason
    assert VAL.check_column_value("mirror_ack", "destination", "oldsite_b", {}) is None
    assert "is not in MIRROR_DESTINATIONS" in VAL.check_column_value("mirror_ack", "destination", "oldhost", {})   # a host is no mirror


def test_without_the_alias_file_a_legacy_spelling_is_refused_by_name(tmp_path):
    root = _root(tmp_path, aliases=None)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    rid, recs = _legacy_run("refused", 100)
    synth.write_run_dir(root, synth.CAMPAIGN, rid, [("run", recs[:1])])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 1
    with DB.reading(DB.db_path(root)) as conn:
        field, reason = conn.execute("SELECT field, reason FROM record_refusal WHERE record_type = 'run'").fetchone()
        assert conn.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 0
    assert field == "producer.substrate" and "oldsite_a" in reason


def _live(root, sql, *args):
    with DB.reading(DB.db_path(root)) as conn:
        return [tuple(r) for r in conn.execute(sql, args).fetchall()]


def test_an_old_record_then_a_new_record_of_one_run_fold_to_the_label_and_a_rebuild_equals_live(tmp_path):
    """A run queued by code that wrote the legacy spelling and finished by code that writes the label: both fold, the live row
    carries the label, history keeps each record as written, a later lifecycle event that refolds the whole history (whose
    only full record is the legacy one) still writes the label, and a rebuild of the log equals the live database."""
    root = _root(tmp_path)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    rid, old = _legacy_run("onerun", 100, through="queued")
    synth.write_run_dir(root, synth.CAMPAIGN, rid, [(str(r["record_type"]), [r]) for r in old])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0
    assert _live(root, "SELECT substrate FROM run WHERE run_id = ?", rid) == [("cluster_a",)]
    # the refold path: a lifecycle event on a run whose only full record carries the legacy spelling
    running = synth.rec("run_state", {"run_id": rid, "state": "running", "event_utc": synth.t(101), "evidence_kind": "job_log",
                                      "scheduler_state": "RUNNING"}, synth.t(101), synth.producer(substrate="oldsite_a"))
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, "late", "000001-run_state.jsonl"), [running])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    assert _live(root, "SELECT substrate, lifecycle_state FROM run WHERE run_id = ?", rid) == [("cluster_a", "running")]
    # the same run re-stated by new code (the label), then generated
    restated = synth.rec("run", dict(old[0]["payload"], substrate="cluster_a"), synth.t(102))
    generated = synth.rec("run_state", {"run_id": rid, "state": "generated", "event_utc": synth.t(103), "evidence_kind": "completion_artifact",
                                        "evidence_uri": "synthetic:///store/onerun/results.json"}, synth.t(103))
    patch = synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_generated": 200}}, synth.t(104))
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, "late", "000002-new.jsonl"), [restated, generated, patch])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, _live(root, "SELECT field, reason FROM record_refusal")
    assert _live(root, "SELECT substrate, lifecycle_state FROM run WHERE run_id = ?", rid) == [("cluster_a", "generated")]
    spellings = sorted({p for (p,) in _live(root, "SELECT json_extract(payload_json, '$.substrate') FROM row_history "
                                                  "WHERE table_name = 'run' AND json_extract(payload_json, '$.substrate') IS NOT NULL")})
    assert spellings == ["cluster_a", "oldsite_a"]                                     # history keeps each record as written
    into = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, into, workers=0)
    a = DB.connect(DB.db_path(root), readonly=True)
    b = DB.connect(into, readonly=True)
    try:
        diff = H.verify_equal(a, b)
    finally:
        a.close()
        b.close()
    assert not {t: n for t, n in diff.items() if n}, diff


def test_a_legacy_mirror_destination_folds_as_its_label(tmp_path):
    root = _root(tmp_path)
    ack = synth.rec("mirror_ack", {"destination": "oldsite_b", "acked_through_seq": 0, "manifest_name": "CHECKSUMS-probe.sha256",
                                   "file_count": 0, "bytes": 0, "checksums_sha256": synth.SHA256_A, "acked_utc": synth.t(5),
                                   "last_error": None}, synth.t(5))
    R.write_records(os.path.join(root, "records", "harvester", "probe", "000001-mirror_ack.jsonl"), [ack])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, _live(root, "SELECT field, reason FROM record_refusal")
    assert _live(root, "SELECT destination FROM mirror_ack") == [("cluster_b",)]


def test_the_gzip_pass_compares_a_mirror_destination_as_its_label(tmp_path):
    """A mirror the configuration still names by a legacy spelling, acked by a record that spelled it the same way (folded under the
    label): the acked file is compressed. Comparing the spellings as written would wait for an ack that never comes."""
    import json
    root = _root(tmp_path)
    with open(os.path.join(root, "config.json"), "w", encoding="utf-8") as fh:
        json.dump({"mirrors": [{"name": "oldsite_a", "kind": "local", "path": str(tmp_path / "m1")}]}, fh)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    H.ingest_once(root, workers=0, min_free=0, gzip_min_age_s=0)
    plain = os.path.join(root, "records", "dims", "000001-dims.jsonl")
    assert os.path.exists(plain)                                              # not acked yet: not compressed
    conn = DB.connect(DB.db_path(root), readonly=True)
    seq = conn.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0]
    conn.close()
    ack = synth.rec("mirror_ack", {"destination": "oldsite_a", "acked_through_seq": seq, "manifest_name": "CHECKSUMS-x", "file_count": 1,
                                   "bytes": 1, "checksums_sha256": "0" * 64, "acked_utc": synth.t(999)}, synth.t(999))
    R.write_records(os.path.join(root, "records", "harvester", "ack.jsonl"), [ack])
    rep = H.ingest_once(root, workers=0, min_free=0, gzip_min_age_s=0)
    assert rep.refused == 0
    with DB.reading(DB.db_path(root)) as c:
        assert [tuple(r) for r in c.execute("SELECT destination FROM mirror_ack").fetchall()] == [("cluster_a",)]
    assert not os.path.exists(plain) and os.path.exists(plain + ".gz")


# --------------------------------------------------------------------------------------------
# parsing workers started fresh (spawn, forkserver) fold through the parent's table
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["spawn", "forkserver"])
def test_parsing_workers_started_fresh_fold_a_legacy_spelling_through_the_roots_table(tmp_path, monkeypatch, method):
    """A worker that starts from a fresh interpreter (Linux's default start method from Python 3.14 on, and spawn everywhere) imports
    the module with the empty table. The cycle's pool hands each worker the table the fold installed from its root, so the legacy
    spelling is validated as its label in a worker exactly as in-process: nothing refused, every run lands under the label."""
    import multiprocessing
    if method not in multiprocessing.get_all_start_methods():
        pytest.skip(f"no {method} start method here")
    ctx = multiprocessing.get_context(method)
    pool = H.ProcessPoolExecutor
    monkeypatch.setattr(H, "ProcessPoolExecutor", lambda *a, **k: pool(*a, mp_context=ctx, **k))
    root = _root(tmp_path)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    rids = []
    for i in range(5):                                       # at least four small files: the cycle parses them in the pool
        rid, recs = _legacy_run(f"fresh{i}", 100 + 10 * i)
        synth.write_run_dir(root, synth.CAMPAIGN, rid, [("run", recs[:1])])
        rids.append(rid)
    rep = H.ingest_once(root, workers=2, min_free=0)
    assert rep.refused == 0, _live(root, "SELECT field, reason FROM record_refusal")
    assert sorted(_live(root, "SELECT run_id, substrate FROM run")) == sorted((rid, "cluster_a") for rid in rids)


# --------------------------------------------------------------------------------------------
# a mirror's acknowledgments under either spelling share one history
# --------------------------------------------------------------------------------------------


def _ack(destination: str, seq: int, i: int):
    return synth.rec("mirror_ack", {"destination": destination, "acked_through_seq": seq, "manifest_name": f"CHECKSUMS-{seq}.sha256",
                                    "file_count": seq, "bytes": seq, "checksums_sha256": synth.SHA256_A, "acked_utc": synth.t(i),
                                    "last_error": None}, synth.t(i))


def test_acknowledgments_of_one_mirror_under_either_spelling_fold_as_one_history_and_a_rebuild_equals_live(tmp_path):
    """The mirror's natural key is its destination, so its history is filed under the label whichever spelling a record carries: a
    delayed acknowledgment written before the standing one never regresses the watermark, and a rebuild of the log (which orders the
    records by their stamps) gives the same row and the same history as the live folds in arrival order."""
    root = _root(tmp_path)
    acks = os.path.join(root, "records", "harvester", "acks")
    steps = [("000004-mirror_ack.jsonl", _ack("cluster_a", 20, 20), 20),     # the label first
             ("000001-mirror_ack.jsonl", _ack("oldsite_a", 10, 10), 20),     # a legacy acknowledgment written earlier, delivered later
             ("000003-mirror_ack.jsonl", _ack("oldsite_a", 30, 30), 30),     # a later legacy one advances it
             ("000002-mirror_ack.jsonl", _ack("cluster_a", 25, 25), 30)]     # a delayed label one does not move it back
    for name, ack, want in steps:
        R.write_records(os.path.join(acks, name), [ack])
        rep = H.ingest_once(root, workers=0, min_free=0)
        assert rep.refused == 0, _live(root, "SELECT field, reason FROM record_refusal")
        assert _live(root, "SELECT destination, acked_through_seq FROM mirror_ack") == [("cluster_a", want)], name
    assert _live(root, "SELECT natural_key_json, COUNT(*) FROM row_history WHERE table_name = 'mirror_ack' GROUP BY natural_key_json") == \
        [('{"destination":"cluster_a"}', 4)]
    into = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, into, workers=0)
    a = DB.connect(DB.db_path(root), readonly=True)
    b = DB.connect(into, readonly=True)
    try:
        diff = H.verify_equal(a, b)
    finally:
        a.close()
        b.close()
    assert not {t: n for t, n in diff.items() if n}, diff
