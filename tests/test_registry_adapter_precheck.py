"""The stat-level input precheck (S2; plan Phase 1 step 2): `base.check_input` takes an input's sha from the pass's
`InputHashCache` when the file's full-resolution stat identity (size, mtime_ns, inode, ctime_ns) equals the recorded one, so a
pass over unchanged inputs reads no store byte; a byte change with a new mtime is hashed; a same-size edit that restores the
mtime is hashed (ctime or inode moved); `--full` re-hashes everything; an `expected_sha256` mismatch still refuses when the sha
came from the cache; a file whose identity moves while it is being hashed is hashed once more, then refused by name; the
sidecar `state/input_hash_cache.json` survives a process restart and is never seeded from watermarks.
"""
from __future__ import annotations

import contextlib
import json
import os
import pstats
import subprocess
import sys
import textwrap

import pytest

import sisteer
from sisteer.registry import db as DB
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import base as AB

from test_registry_adapter_pass import SyntheticStoreAdapter, _cli, _pass_and_ingest, _setup, _store  # noqa: E402

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(sisteer.__file__)))


@contextlib.contextmanager
def _counting_sha256():
    """``R.file_sha256`` counted for the block: yields the list of paths hashed, in order (the fold's own record hashing lands
    there too — callers filter to the store root, the S2 question being whether any STORE byte was read)."""
    calls = []
    real = R.file_sha256
    R.file_sha256 = lambda path: (calls.append(path), real(path))[1]
    try:
        yield calls
    finally:
        R.file_sha256 = real


def _pass(root, ad, now, **kw):
    """One pass and a fold; returns ``(report, ingest report, store paths hashed)``."""
    with _counting_sha256() as calls:
        rep, ing = _pass_and_ingest(root, ad, now, **kw)
    return rep, ing, sorted(c for c in calls if c.startswith(str(ad.roots[0])))


def _three_stores(tmp_path):
    root, store_root = _setup(tmp_path)
    _store(store_root, "s2", "legacy_store_2", "none", [0.1, 0.9])
    _store(store_root, "s3", "legacy_store_3", "constitution_lora@aaaa", [0.7, 0.7, 0.2])
    paths = sorted(os.path.realpath(os.path.join(store_root, d, "store.json")) for d in ("s1", "s2", "s3"))
    return root, store_root, paths


def _cache_entries(root):
    with open(AB.InputHashCache.path(root), encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["version"] == AB.INPUT_HASH_CACHE_VERSION and data["fields"] == list(AB.INPUT_STAT_FIELDS)
    return data["entries"]


def test_second_pass_reads_no_store_byte_and_the_sidecar_records_the_full_identity(tmp_path):
    root, store_root, paths = _three_stores(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    rep1, ing1, hashed1 = _pass(root, ad, synth.t(2000))
    assert rep1.groups_refreshed == 3 and ing1.refused == 0
    assert hashed1 == paths and rep1.inputs_hashed == 3 and rep1.inputs_hash_reused == 0     # the first pass hashes everything once
    entries = _cache_entries(root)
    assert set(entries) == set(paths)                                                         # keyed by the REAL path
    for p in paths:
        st = os.stat(p)
        assert entries[p] == {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "ino": st.st_ino, "ctime_ns": st.st_ctime_ns,
                              "sha256": R.file_sha256(p)}
    rep2, ing2, hashed2 = _pass(root, ad, synth.t(2500))
    assert hashed2 == [] and rep2.inputs_hashed == 0 and rep2.inputs_hash_reused == 3          # no store file opened
    assert rep2.inputs_changed == 0 and rep2.groups_refreshed == 0 and rep2.records_written == 0 and rep2.results == []
    assert ing2.ok == 0 and not ing2.changed


def test_a_byte_change_with_a_new_mtime_is_hashed_and_refreshed(tmp_path):
    root, store_root, paths = _three_stores(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    p = os.path.realpath(_store(store_root, "s1", "legacy_store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4, 0.9, 0.1]))
    rep, ing, hashed = _pass(root, ad, synth.t(3000))
    assert hashed == [p] and rep.inputs_hashed == 1 and rep.inputs_hash_reused == 2
    assert rep.inputs_changed == 1 and rep.groups_refreshed == 1 and ing.refused == 0
    assert _cache_entries(root)[p]["sha256"] == R.file_sha256(p)


def _same_size_edit(p):
    """Rewrite ``p`` with the same number of bytes and put its mtime back: only ctime (and, with a replacement, the inode) moves."""
    st = os.stat(p)
    with open(p, "r", encoding="utf-8") as fh:
        text = fh.read()
    new = text.replace("0.2", "0.3", 1)
    assert new != text and len(new.encode()) == len(text.encode())
    return st, new


def test_a_same_size_edit_that_restores_the_mtime_is_still_hashed(tmp_path):
    """The case a watermark or the harvester's three-field rule would miss: same size, same mtime — a ``cp -p`` over the file. The
    fourth field (ctime_ns) moves on every write and every utime, so the cache misses and the bytes are hashed."""
    root, store_root, paths = _three_stores(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    p = paths[0]
    st, new = _same_size_edit(p)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(new)
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns))
    after = os.stat(p)
    assert (after.st_size, after.st_mtime_ns, after.st_ino) == (st.st_size, st.st_mtime_ns, st.st_ino), "the edit must keep size, mtime and inode"
    assert after.st_ctime_ns != st.st_ctime_ns, "a write followed by utime moves ctime; the test needs that to hold on this filesystem"
    rep, ing, hashed = _pass(root, ad, synth.t(3000))
    assert hashed == [p] and rep.inputs_hashed == 1 and rep.groups_refreshed == 1 and ing.refused == 0
    assert _cache_entries(root)[p]["ctime_ns"] == after.st_ctime_ns


def test_a_same_size_replacement_with_a_new_inode_and_the_old_mtime_is_hashed(tmp_path):
    root, store_root, paths = _three_stores(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    p = paths[0]
    st, new = _same_size_edit(p)
    tmp = p + ".new"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(new)
    os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.replace(tmp, p)
    after = os.stat(p)
    assert (after.st_size, after.st_mtime_ns) == (st.st_size, st.st_mtime_ns) and after.st_ino != st.st_ino
    rep, ing, hashed = _pass(root, ad, synth.t(3000))
    assert hashed == [p] and rep.groups_refreshed == 1 and ing.refused == 0
    assert _cache_entries(root)[p]["ino"] == after.st_ino


def test_full_bypasses_the_cache_and_hashes_every_byte(tmp_path):
    root, store_root, paths = _three_stores(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    rep, _, hashed = _pass(root, ad, synth.t(2500))
    assert hashed == []
    rep_full, _, hashed_full = _pass(root, ad, synth.t(3000), full=True)
    assert hashed_full == paths and rep_full.inputs_hashed == 3 and rep_full.inputs_hash_reused == 0
    assert set(_cache_entries(root)) == set(paths)                # a --full pass still records what it hashed
    rep_after, _, hashed_after = _pass(root, ad, synth.t(3500))
    assert hashed_after == [] and rep_after.inputs_hash_reused == 3


def test_expected_sha_mismatch_still_refuses_when_the_sha_came_from_the_cache(tmp_path):
    store_root = str(tmp_path / "stores")
    os.makedirs(store_root)
    p = _store(store_root, "s1", "k", "none", [0.1])
    with _counting_sha256() as calls:
        cache = AB.InputHashCache()
        with AB.using_hash_cache(cache):
            inp = AB.check_input(p)
            assert calls == [os.path.realpath(p)] and cache.hashed == 1
            calls.clear()
            with pytest.raises(AB.InputRefusal) as ei:
                AB.check_input(p, expected_sha256="0000")
            assert "no longer matches" in str(ei.value) and calls == []                  # judged on the cached sha, no read
            assert AB.check_input(p, expected_sha256=inp.sha256[:8]).sha256 == inp.sha256 and calls == []
            assert cache.reused == 2
        # outside a pass (no cache installed) every call hashes, as before
        AB.check_input(p)
        assert calls == [os.path.realpath(p)]


def test_a_file_that_changes_while_being_hashed_is_hashed_again_once_then_refused_by_name(tmp_path):
    """The mutation lands DETERMINISTICALLY between the two reads: ``R.file_sha256`` is wrapped so each hash is followed by a rewrite
    when the phase asks for one -- never a thread or a sleep racing the hash. Every rewrite GROWS the file, one item more than any
    bytes it has held before (counted across the whole test), so the stat identity moves by SIZE, which no clock granularity can hide.
    The first shape wrote ``calls + 3`` items with ``calls`` reset per phase: the refusal phase's first rewrite then had the same length
    as the previous phase's final bytes, and on a fast node both writes landed inside one coarse-clock tick, so size, mtime_ns, ino and
    ctime_ns all matched, the precheck saw no move, hashed once and refused nothing -- the gate failed one landing in three with
    DID NOT RAISE while the precheck's behaviour was right."""
    store_root = str(tmp_path / "stores")
    os.makedirs(store_root)
    p = os.path.realpath(_store(store_root, "s1", "k", "none", [0.1]))
    real = R.file_sha256
    state = {"calls": 0, "mutate_first": 0, "writes": 0}

    def racing(path):
        """Hash the file, then (while the phase asks) rewrite it with strictly more bytes than it has ever held."""
        sha = real(path)
        state["calls"] += 1
        if state["calls"] <= state["mutate_first"]:
            before = os.stat(path).st_size
            state["writes"] += 1
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"key": "k", "object": "none", "items": [{"id": f"q{i}", "score": 0.5} for i in range(state["writes"] + 3)]}, fh)
            assert os.stat(path).st_size > before, "the rewrite must move the size: a timestamp alone is not a deterministic move"
        return sha
    R.file_sha256 = racing
    try:
        # no move between the reads: hashed once, bound to the identity the stat saw, recorded; nothing refused
        state.update(calls=0, mutate_first=0)
        cache = AB.InputHashCache()
        with AB.using_hash_cache(cache):
            inp = AB.check_input(p)
        assert state["calls"] == 1 and inp.sha256 == real(p) and cache.hashed == 1 and cache.entries[p]["sha256"] == inp.sha256
        # one move: hashed again against the new identity, bound to the FINAL bytes, recorded once
        state.update(calls=0, mutate_first=1)
        cache = AB.InputHashCache()
        with AB.using_hash_cache(cache):
            inp = AB.check_input(p)
        assert state["calls"] == 2 and inp.sha256 == real(p) and inp.size == os.stat(p).st_size
        assert cache.entries[p] == dict(AB.stat_identity(os.stat(p)), sha256=inp.sha256) and cache.hashed == 1
        # two moves: refused by name, nothing recorded
        state.update(calls=0, mutate_first=2)
        cache = AB.InputHashCache()
        with AB.using_hash_cache(cache):
            with pytest.raises(AB.InputRefusal) as ei:
                AB.check_input(p)
        assert "changed while being hashed" in str(ei.value) and p in str(ei.value) and state["calls"] == 2
        assert cache.entries == {} and cache.hashed == 0
    finally:
        R.file_sha256 = real


def test_the_sidecar_survives_a_process_restart(tmp_path):
    root, store_root, paths = _three_stores(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    assert os.path.exists(AB.InputHashCache.path(root))
    script = textwrap.dedent(f"""
        from sisteer.registry import db as DB
        from sisteer.registry import records as R
        from sisteer.registry.adapters import runner as RUN
        from test_registry_adapter_pass import SyntheticStoreAdapter
        calls = []
        real = R.file_sha256
        def counted(path):
            calls.append(path)
            return real(path)
        R.file_sha256 = counted
        with DB.reading(DB.db_path({root!r})) as conn:
            rep = RUN.run_pass({root!r}, [SyntheticStoreAdapter({store_root!r})], conn=conn, now={synth.t(2600)!r})
        print(len([c for c in calls if c.startswith({store_root!r})]), rep.inputs_hash_reused, rep.inputs_changed, rep.groups_refreshed)
    """)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([SRC_DIR, TESTS_DIR, os.environ.get("PYTHONPATH", "")]).rstrip(os.pathsep))
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["0", "3", "0", "0"], out.stdout                          # a fresh process: nothing hashed, three reused


def test_the_sidecar_is_never_seeded_from_watermarks(tmp_path):
    """A registry whose watermarks already know every input (a pass before the cache existed) still hashes everything once: the
    ms-truncated ``mtime_utc`` and the missing inode / ctime are no identity."""
    root, store_root, paths = _three_stores(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    os.remove(AB.InputHashCache.path(root))                                               # the state before the cache landed
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM source_watermark WHERE last_error IS NULL").fetchone()[0] == 3
    rep, ing, hashed = _pass(root, ad, synth.t(2500))
    assert hashed == paths and rep.inputs_hashed == 3 and rep.inputs_hash_reused == 0
    assert rep.inputs_changed == 0 and rep.groups_refreshed == 0                           # the watermark compare still says unchanged
    assert set(_cache_entries(root)) == set(paths)


def test_a_torn_or_foreign_sidecar_costs_one_hashing_pass_never_a_wrong_sha(tmp_path):
    root, store_root, paths = _three_stores(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    p = AB.InputHashCache.path(root)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write('{"version": 1, "entries": {"trunc')
    assert AB.InputHashCache.load(root).entries == {}
    rep, _, hashed = _pass(root, ad, synth.t(2500))
    assert hashed == paths and set(_cache_entries(root)) == set(paths)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"version": 99, "entries": {paths[0]: {"size": 1, "mtime_ns": 1, "ino": 1, "ctime_ns": 1, "sha256": "00"}}}, fh)
    assert AB.InputHashCache.load(root).entries == {}                                     # another version is dropped, not reinterpreted
    # an entry with a malformed shape is dropped at load; a well-formed one whose identity moved is a miss, never a wrong sha
    bad = AB.InputHashCache({paths[0]: {"size": "x", "sha256": "00"}, paths[1]: {"size": 1, "mtime_ns": 1, "ino": 1, "ctime_ns": 1, "sha256": "00"}})
    assert set(bad.entries) == {paths[1]}
    assert bad.lookup(paths[1], os.stat(paths[1])) is None and bad.reused == 0


def test_prune_keeps_unseen_entries_whose_files_exist_and_drops_the_gone(tmp_path):
    """A pass over ONE adapter (the way a landing window runs them) must not evict the other adapters' entries."""
    root, store_root, paths = _three_stores(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    cache = AB.InputHashCache.load(root)
    assert set(cache.entries) == set(paths)
    other = str(tmp_path / "elsewhere.json")
    with open(other, "w", encoding="utf-8") as fh:
        fh.write("{}")
    cache.entries[other] = dict(AB.stat_identity(os.stat(other)), sha256=R.file_sha256(other))
    cache.entries[str(tmp_path / "vanished.json")] = {"size": 1, "mtime_ns": 1, "ino": 1, "ctime_ns": 1, "sha256": "00"}
    cache.seen = {paths[0]}                                                                 # this pass consulted one file only
    assert cache.prune() == 1
    assert set(cache.entries) == set(paths) | {other} and cache.dirty
    cache.save(root)
    assert set(_cache_entries(root)) == set(paths) | {other}


def test_cli_full_reaches_the_cache_bypass(tmp_path, monkeypatch, capsys):
    root, store_root, paths = _three_stores(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    with _counting_sha256() as calls:
        rc, summary, _, _, _ = _cli(root, ad, monkeypatch, capsys, "--no-ingest")
    assert rc == 0 and [c for c in calls if c.startswith(store_root)] == [] and summary["inputs_hash_reused"] == 3 and summary["inputs_hashed"] == 0
    with _counting_sha256() as calls:
        rc, summary, _, _, _ = _cli(root, ad, monkeypatch, capsys, "--no-ingest", "--full")
    assert rc == 0 and sorted(c for c in calls if c.startswith(store_root)) == paths and summary["inputs_hashed"] == 3 and summary["inputs_hash_reused"] == 0


def test_profile_writes_pstats_of_the_pass(tmp_path, monkeypatch, capsys):
    root, store_root, paths = _three_stores(tmp_path)
    ad = SyntheticStoreAdapter(store_root)
    prof = str(tmp_path / "profiles" / "pass.prof")                                        # a directory that does not exist yet
    rc, summary, _, _, _ = _cli(root, ad, monkeypatch, capsys, "--no-ingest", "--profile", prof)
    assert rc == 0 and summary["profile"] == prof and summary["groups_refreshed"] == 3
    stats = pstats.Stats(prof)
    names = {func for (_, _, func) in stats.stats}
    assert {"run_pass", "_run_adapter", "plan_groups", "check_input", "hash_input"} <= names, sorted(names)[:20]
