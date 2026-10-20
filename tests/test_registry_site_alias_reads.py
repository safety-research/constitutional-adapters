"""Persisted data written before a site renamed its clusters and grading host reads under the new labels.

The site alias table (``<root>/site_aliases.toml``, ``[site_aliases] old = "label"``) is the only place the older names live; code
spells the labels. Every read of persisted state goes through ``site_aliases.get_compat`` / ``canonical`` / ``legacy_keys``. The
names here are INVENTED (``oldsite_a``, ``oldsite_b``, ``oldhost``): the real ones never appear in the tree. With an empty table
the older spellings are not read at all: nothing is guessed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from sisteer.registry import config as CFG
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import mirrors as M
from sisteer.registry import records as R
from sisteer.registry import site_aliases as SA
from sisteer.registry import snapshot as SN
from sisteer.registry import status as STATUS
from sisteer.registry import stores as ST
from sisteer.registry import synth
from sisteer.registry.adapters import transcript_mirrors as TM
from sisteer.registry.adapters.base import AdapterRefusal

REPO = Path(__file__).resolve().parents[1]
ALIASES = '[site_aliases]\noldsite_a = "cluster_a"\noldsite_b = "cluster_b"\noldhost = "grading_host"\n'


def _install(root: str) -> None:
    SA.install(CFG.load(root))


@pytest.fixture
def empty_aliases(tmp_path):
    """No table installed, before and after the test (a test that installs one leaves the process as it found it)."""
    empty = tmp_path / "no-aliases"
    empty.mkdir()
    _install(str(empty))
    yield str(empty)
    _install(str(empty))


@pytest.fixture
def aliases_root(tmp_path, empty_aliases):
    """A registry root carrying the invented table, installed."""
    root = tmp_path / "reg"
    root.mkdir()
    (root / SA.ALIASES_NAME).write_text(ALIASES, encoding="utf-8")
    _install(str(root))
    return str(root)


def test_the_table_maps_values_and_keys(aliases_root):
    assert SA.canonical("oldsite_a") == "cluster_a" and SA.canonical("cluster_a") == "cluster_a" and SA.canonical("other") == "other"
    assert list(SA.legacy_keys("grading_host_mirror_present")) == ["grading_host_mirror_present", "oldhost_mirror_present"]
    assert SA.get_compat({"oldhost_mirror_present": True}, "grading_host_mirror_present") is True
    assert SA.get_compat({"oldhost_mirror_present": True, "grading_host_mirror_present": False}, "grading_host_mirror_present") is False


# ---- the store authority state (stores.authority_rows, the status page's section) -------------------------------------------

def _authority_state(root: str, key: str) -> None:
    os.makedirs(os.path.join(root, "state"), exist_ok=True)
    row = {"authority": "cluster_a", "state": "ok", "problems": [], "notes": [], "current": "abcd1234", "exists": True, "verified": True,
           "sealed": True, "published_record": True, key: True, "carry_overdue": False}
    with open(os.path.join(root, "state", ST.AUTHORITY_STATE_FILE), "w", encoding="utf-8") as fh:
        json.dump({"schema": ST.AUTHORITY_STATE_SCHEMA, "checked_utc": "2026-10-01T00:00:00Z", "destination": "cluster_a", "ok": True,
                   "problems": [], "components": {"slug_x": row}}, fh)


def test_authority_rows_and_the_status_section_read_a_key_written_under_the_older_name(aliases_root, tmp_path):
    _authority_state(aliases_root, "oldhost_mirror_present")
    _checked, rows = ST.authority_rows(aliases_root)
    assert rows["slug_x"]["grading_host_mirror_present"] is True
    sect = STATUS._authority_section(aliases_root, 1.8e9)
    assert [r["grading_host_mirror_present"] for r in sect["rows"]] == [True]
    # the same file under a root with no table: the older key is not read
    bare = tmp_path / "bare"
    bare.mkdir()
    _authority_state(str(bare), "oldhost_mirror_present")
    _checked, rows = ST.authority_rows(str(bare))
    assert rows["slug_x"]["grading_host_mirror_present"] is None
    assert [r["grading_host_mirror_present"] for r in STATUS._authority_section(str(bare), 1.8e9)["rows"]] == [None]


def test_a_refused_alias_file_is_named_by_the_status_section_and_its_rows_still_render(empty_aliases, tmp_path):
    root = tmp_path / "reg-refused"
    root.mkdir()
    _authority_state(str(root), "grading_host_mirror_present")
    (root / SA.ALIASES_NAME).write_text('[site_aliases]\nOld-Site = "cluster_a"\n', encoding="utf-8")
    sect = STATUS._authority_section(str(root), 1.8e9)
    assert [r["slug"] for r in sect["rows"]] == ["slug_x"] and sect["rows"][0]["grading_host_mirror_present"] is True
    assert sect["level"] == "red" and SA.ALIASES_NAME in sect["error"] and "Old-Site" in sect["error"], sect["error"]


def test_an_authority_written_under_the_older_names_reads_as_its_current_value(aliases_root, tmp_path):
    """The store authorities embed a label (``stores.AUTHORITIES``), so a state file an older writer wrote spells them under the older
    names: each reads back as the authority it is an older spelling of; a value that is none stays as read."""
    older = {a: [k for k in SA.legacy_keys(a) if k != a] for a in ST.AUTHORITIES}
    assert all(older.values()), older
    for a, spellings in older.items():
        assert all(ST.canonical_authority(k) == a for k in spellings) and ST.canonical_authority(a) == a
    assert ST.canonical_authority("unknown_kind") == "unknown_kind" and ST.canonical_authority(None) is None
    first = next(iter(ST.AUTHORITIES))
    _authority_state(aliases_root, "grading_host_mirror_present")
    path = os.path.join(aliases_root, "state", ST.AUTHORITY_STATE_FILE)
    doc = json.load(open(path, encoding="utf-8"))
    doc["components"]["slug_x"]["authority"] = older[first][0]
    json.dump(doc, open(path, "w", encoding="utf-8"))
    assert [r["authority"] for r in STATUS._authority_section(aliases_root, 1.8e9)["rows"]] == [first]
    # no table: the older spelling is shown as written, never guessed
    bare = tmp_path / "bare-authority"
    bare.mkdir()
    _install(str(bare))
    assert ST.canonical_authority(older[first][0]) == older[first][0]


def test_a_refused_alias_file_refuses_the_stores_command_by_name(empty_aliases, tmp_path, capsys):
    root = tmp_path / "reg-bad"
    root.mkdir()
    (root / SA.ALIASES_NAME).write_text('[site_aliases]\nOld-Site = "cluster_a"\n', encoding="utf-8")
    import argparse
    args = argparse.Namespace(action="status", config=str(REPO / "configs" / "registry" / "cluster_reads.yaml"))
    assert ST.cmd_stores(args, str(root)) == 2
    err = capsys.readouterr().err
    assert err.startswith("stores: ") and "Old-Site" in err, err


# ---- the record-log mirror's destination-keyed state ------------------------------------------------------------------------

def _corpus(root: str) -> None:
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=1, n_items=10)
    H.ingest_once(root, workers=0, min_free=0)


def _seq(root: str) -> int:
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return int(conn.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0])
    finally:
        conn.close()


def _more(root: str, i: int) -> None:
    R.write_records(os.path.join(root, "records", "late", f"{i:06d}-late.jsonl"),
                    [synth.rec("si_text", {"si_id": f"late_{i}", "si_text_sha256": ("%064x" % (0xf0 + i)), "si_kind": "irrelevant_control",
                                           "source_uri": "synthetic://x"}, synth.t(6000 + i))])
    H.ingest_once(root, workers=0, min_free=0)


def _names(n: int):
    return [f"CHECKSUMS-{i:012d}.sha256" for i in range(1, n + 1)]


def test_a_pointer_recorded_under_the_older_destination_name_is_the_destination_s_and_moves_to_its_name(aliases_root):
    a, b, c = _names(3)
    os.makedirs(os.path.join(aliases_root, "state"), exist_ok=True)
    with open(M.pointers_path(aliases_root), "w", encoding="utf-8") as fh:
        json.dump({"utc": "2026-10-01T00:00:01.000Z", "destinations": {
            "oldsite_a": {"current": b, "observed_utc": "2026-10-01T00:00:01.000Z", "source": "sync"},
            "oldsite_b": {"current": a, "observed_utc": "2026-10-01T00:00:01.000Z", "source": "sync"},
            "cluster_b": {"current": c, "observed_utc": "2026-10-02T00:00:00.000Z", "source": "sync"}}}, fh)
    recs = M.read_pointers(aliases_root)
    assert {k: v["current"] for k, v in recs.items()} == {"cluster_a": b, "cluster_b": c}, "the current name's entry wins over the older one"
    assert M.last_known_currents(aliases_root) == {"cluster_a": b, "cluster_b": c}
    # an observation OLDER than the one recorded under the older name is stale for the destination it now is
    assert M.record_pointer(aliases_root, "cluster_a", a, "listing", now="2026-09-30T00:00:00.000Z") is False
    # a fresher one merges, and the record is rewritten under the destination's current name only
    assert M.record_pointer(aliases_root, "cluster_a", c, "sync", now="2026-10-03T00:00:00.000Z") is True
    with open(M.pointers_path(aliases_root), encoding="utf-8") as fh:
        assert sorted(json.load(fh)["destinations"]) == ["cluster_a", "cluster_b"]


def test_state_rows_and_held_observations_under_the_older_name_count_for_the_destination(aliases_root):
    a, b = _names(2)
    M.write_state(aliases_root, [{"destination": "oldsite_a", "dest_current": a, "switched": True}])
    assert M.last_known_currents(aliases_root) == {"cluster_a": a}
    held = M.hold_unrecorded(aliases_root, "oldsite_a", b, "sync", now="2026-10-01T00:00:00.000Z", error="lock timeout")
    assert set(M.unrecorded_pointers(aliases_root)) == {"cluster_a"}
    assert M.record_pointer(aliases_root, "cluster_a", b, "sync", now="2026-10-02T00:00:00.000Z") is True
    assert not os.path.exists(held), "a merge for the destination releases the observation held under its older name"
    assert M.unrecorded_pointers(aliases_root) == {}


def _ssh_dest(name: str) -> M.Destination:
    return M.Destination(name, "ssh", "/srv/shared/u/registry/log", "login-host", remote_src="/srv/shared/u/repos/checkout/src")


def test_retention_keeps_the_pointer_recorded_under_the_older_name_and_the_next_sync_stays_incremental(aliases_root, tmp_path):
    root = aliases_root
    _corpus(root)
    base = M.write_manifest(root, _seq(root))
    M.record_pointer(root, "oldsite_a", base, "sync", now="2026-10-01T00:00:00.000Z")
    later = []
    for i in range(1, 5):
        _more(root, i)
        later.append(M.write_manifest(root, _seq(root)))
    summary = M.prune_manifests(root, [], keep=1, configured=["cluster_a"])
    side = summary["sides"][0]
    assert side["error"] is None and base in summary["references"], summary
    kept = sorted(os.listdir(os.path.join(root, M.MANIFEST_DIR)))
    assert base in kept and later[-1] in kept and not (set(later[:-1]) & set(kept)), kept
    # the destination's pointer is still the delta's base: the copy is incremental since it
    _more(root, 9)
    mw = M.write_manifest_report(root, _seq(root))
    n = len(M.log_files(root))

    def respond(argv):
        joined = " ".join(argv)
        if argv[0] == "ssh" and "CHECKSUMS.CURRENT" in joined and "test -f" in joined:
            return base
        if argv[0] == "ssh" and "sisteer.registry.mirrors" in joined and "verify" in argv:
            return json.dumps({"manifest": "x", "ok": n, "missing": [], "corrupt": [], "bytes": 1, "mode": "incremental", "since": base,
                               "hashed": 3, "presence_only": n - 3, "sampled": 2, "sample_seed": 7, "seconds": 1.5})
        if argv[0] == "rsync" and "--files-from=-" in argv:
            return "\nNumber of regular files transferred: 1\n"
        return ""

    calls = []

    def runner(argv, timeout_s, input=None, check=True):
        calls.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 0, stdout=respond(list(argv)), stderr="")

    res = M.sync(root, _ssh_dest("cluster_a"), _seq(root), mw.name, runner=runner, manifest=mw)
    assert res.error is None and res.copy_mode == "incremental" and res.copy_since == base


def test_without_the_table_the_older_record_names_no_configured_destination_and_retention_holds(empty_aliases, tmp_path):
    root = str(tmp_path / "reg-bare")
    _corpus(root)
    base = M.write_manifest(root, _seq(root))
    M.record_pointer(root, "oldsite_a", base, "sync", now="2026-10-01T00:00:00.000Z")
    for i in range(1, 4):
        _more(root, i)
        M.write_manifest(root, _seq(root))
    before = sorted(os.listdir(os.path.join(root, M.MANIFEST_DIR)))
    summary = M.prune_manifests(root, [], keep=1, configured=["cluster_a"])
    assert summary["sides"][0]["error"] and "cluster_a" in summary["sides"][0]["error"]
    assert sorted(os.listdir(os.path.join(root, M.MANIFEST_DIR))) == before, "an unknown pointer deletes nothing on the grading host"


# ---- the transcript mirror index (a pointer file outside the repo, written before the rename) -------------------------------

def _legacy_index(rs: str, tsv_name: str) -> None:
    d = os.path.join(rs, "transcripts_mirrors")
    os.makedirs(d)
    entry = {"oldsite_a_path": "/srv/a/exp/tstore", "oldsite_b_path": "/srv/b/exp/tstore", "origin": "oldsite_b", "mirror": "oldsite_a",
             "content_manifest_sha256": "e" * 64, "former_oldhost_dir": "transcripts_x"}
    with open(os.path.join(d, "index.json"), "w", encoding="utf-8") as fh:
        json.dump({"_meta": {"built_at": "2026-09-17T05:23:53Z"}, "stores": {"tstore": entry}}, fh)
    with open(os.path.join(d, tsv_name), "w", encoding="utf-8") as fh:
        for rel in ("run/pair/cell/shard00/progress.jsonl", "run/pair/cell/shard01/progress.jsonl"):
            fh.write(f"transcripts_x\t{rel}\t{hashlib.sha256(rel.encode()).hexdigest()}\n")


def test_the_transcript_index_and_reclaim_table_written_under_the_older_names_locate_a_cell(aliases_root, tmp_path):
    rs = str(tmp_path / "rs")
    _legacy_index(rs, "oldhost_file_sha256.tsv")
    tm = TM.TranscriptMirrors(rs)
    assert os.path.basename(tm.tsv_path) == "oldhost_file_sha256.tsv"
    ptr, why = tm.locate_run_dir("transcripts_x", "run/pair/cell")
    assert why is None and ptr.stores == ["tstore"] and ptr.per_file_sha
    assert ptr.uri_primary == "cluster_a:/srv/a/exp/tstore/run/pair/cell"
    assert [(loc["cluster"], loc["role"]) for loc in ptr.locations] == [("cluster_a", "mirror"), ("cluster_b", "origin")]
    assert set(ptr.files) == {"tstore:run/pair/cell/shard00/progress.jsonl", "tstore:run/pair/cell/shard01/progress.jsonl"}


def test_without_the_table_the_older_index_names_nothing_and_the_cell_is_refused(empty_aliases, tmp_path):
    """No table: the reclaim table written under the older name is not looked for, so the read is refused under the name a reader expects;
    with a reclaim table under the current name, the index's older keys still name no directory and the cell is refused."""
    rs = str(tmp_path / "rs")
    _legacy_index(rs, "oldhost_file_sha256.tsv")
    tm = TM.TranscriptMirrors(rs)
    assert os.path.basename(tm.tsv_path) == TM.TSV_NAME
    with pytest.raises(AdapterRefusal, match=re.escape(TM.TSV_NAME) + r": input missing"):
        tm.locate_run_dir("transcripts_x", "run/pair/cell")
    os.rename(os.path.join(rs, "transcripts_mirrors", "oldhost_file_sha256.tsv"), os.path.join(rs, "transcripts_mirrors", TM.TSV_NAME))
    ptr, why = TM.TranscriptMirrors(rs).locate_run_dir("transcripts_x", "run/pair/cell")
    assert ptr is None and "has no entry" in why


# ---- a launch plan written before the rename (the seed-replicate carry-over) ------------------------------------------------

def _carry_over():
    for p in (REPO / "scripts" / "modal",):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import seedrep_carry_over as co  # noqa: PLC0415
    return co


def test_a_petri_child_s_drive_block_under_the_older_name_names_its_leg(aliases_root, monkeypatch):
    co = _carry_over()
    row = {"children": [{"arm": "a0", "oldhost_drive": {"cell_index": 3}}, {"arm": "a1", "oldhost_drive": {"cell_index": 1}}]}
    monkeypatch.setenv("REGISTRY_ROOT", aliases_root)
    monkeypatch.setattr(co, "_SITE_ALIASES", [])
    assert co.row_petri_legs(row) == [1, 3]
    current = {"children": [{"arm": "a0", "grading_host_drive": {"cell_index": 2}}]}
    assert co.row_petri_legs(current) == [2]


def test_without_the_table_a_child_s_older_drive_block_is_not_read(empty_aliases, monkeypatch):
    co = _carry_over()
    row = {"children": [{"arm": "a0", "oldhost_drive": {"cell_index": 3}}, {"arm": "a1", "oldhost_drive": {"cell_index": 1}}]}
    monkeypatch.setenv("REGISTRY_ROOT", empty_aliases)
    monkeypatch.setattr(co, "_SITE_ALIASES", [])
    assert co.row_petri_legs(row) == [0, 1], "the child ordinals: nothing is read under a name the site never declared"


# ---- the export scrub rewrites the older names to their labels ---------------------------------------------------------------

def _rows():
    return [{"run_id": "r1", "substrate": "cluster_a", "notes": "served on OldSite_A, pulled from oldsite_b; oldsite_abc stays",
             "extra_json": json.dumps({"oldsite_a_path": "exp/x", "cluster": "oldsite_b", "verified_oldsite_a": True, "note": "the oldhost copy"})}]


def test_the_export_scrub_relabels_the_older_names_in_text_and_json(aliases_root):
    rows, scrubbed = SN.scrub_frame("run", _rows(), tokens=(), repo_roots=())
    r = rows[0]
    assert r["notes"] == "served on cluster A, pulled from cluster_b; oldsite_abc stays"
    assert json.loads(r["extra_json"]) == {"cluster_a_path": "exp/x", "cluster": "cluster_b", "verified_cluster_a": True, "note": "the grading_host copy"}
    assert r["substrate"] == "cluster_a" and scrubbed == ["extra_json", "notes"]


def test_without_the_table_the_export_scrub_changes_nothing_unless_aliases_are_passed(empty_aliases):
    rows, scrubbed = SN.scrub_frame("run", _rows(), tokens=(), repo_roots=())
    assert rows == _rows() and scrubbed == []
    rows, scrubbed = SN.scrub_frame("run", _rows(), tokens=(), repo_roots=(), aliases=[("oldsite_b", "cluster_b")])
    assert rows[0]["notes"].endswith("pulled from cluster_b; oldsite_abc stays") and scrubbed == ["extra_json", "notes"]
