"""The cluster mirrors' INCREMENTAL COPY (2026-09-28, evening). The whole-tree ``rsync -rt --ignore-existing`` of ``records/`` and
``rulings/`` that opened every 5-min cycle stat'ed every destination entry — 461,769 files on cluster A's VAST through a login node
at load 40 — and from ~22:00Z on did not finish inside its 3,600 s bound, so ``registry-mirrors.service`` hit its 2 h
``TimeoutStartSec`` every run and the cluster A copy (the durability copy the 09-28 ruling names) stopped ~6,000 records behind.
Now: the ONE ssh that reads the destination's ``CHECKSUMS.CURRENT`` fixes the copy's base; exactly the on-disk paths behind the
manifest entries the base does not carry (``added_since``) ride ``rsync --files-from=-`` (``-rt --partial-dir --ignore-existing``
kept: never a rewrite), in manifest order, in chunks of ``COPY_CHUNK_FILES`` with a journal line each and the run's budget checked
between chunks; the whole tree only with no acked manifest (or its file gone from the grading host) and under ``--full``; the incremental
verify follows unchanged and its ``missing`` entries are re-copied with the corrupt ones; the row names the copy mode, its base, the
listed count and rsync's transferred count; two destinations in ONE invocation each take their delta from THEIR OWN pointer;
rsync's partial-transfer exits are a journal line, any other exit the destination's error.
"""
from __future__ import annotations

import json
import os
import subprocess

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import mirrors as M
from sisteer.registry import records as R
from sisteer.registry import synth


def _setup(tmp_path, n_runs=2):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=n_runs, n_items=10)
    H.ingest_once(root, workers=0, min_free=0)
    return root


def _seq(root):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return int(conn.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0])
    finally:
        conn.close()


def _more(root, i):
    R.write_records(os.path.join(root, "records", "late", f"{i:06d}-late.jsonl"),
                    [synth.rec("si_text", {"si_id": f"late_{i}", "si_text_sha256": ("%064x" % (0xf0 + i)), "si_kind": "irrelevant_control",
                                           "source_uri": "synthetic://x"}, synth.t(6000 + i))])
    H.ingest_once(root, workers=0, min_free=0)


def _ssh_dest(name="cluster_a", host="login-0", shared="/shared/u", **kw):
    return M.Destination(name, "ssh", f"{shared}/registry/log", host, remote_src=f"{shared}/repos/constitutional-adapters/src", **kw)


class _Recorder:
    """The runner shape ``(argv, timeout_s, input=, check=)``: records argv, the timeout and STDIN; ``respond(argv)`` is the stdout,
    or a ``(returncode, stdout, stderr)`` triple for a call that exits nonzero."""

    def __init__(self, respond):
        self.calls = []
        self.respond = respond

    def __call__(self, argv, timeout_s, input=None, check=True):
        self.calls.append((list(argv), timeout_s, input))
        out = self.respond(list(argv))
        if isinstance(out, tuple):
            rc, stdout, stderr = out
            return subprocess.CompletedProcess(list(argv), rc, stdout=stdout, stderr=stderr)
        return subprocess.CompletedProcess(list(argv), 0, stdout=out, stderr="")

    @property
    def argv(self):
        return [c[0] for c in self.calls]


def _is_probe(argv):
    return argv[0] == "ssh" and "CHECKSUMS.CURRENT" in " ".join(argv) and "test -f" in " ".join(argv)


def _is_verify(argv):
    return argv[0] == "ssh" and "sisteer.registry.mirrors" in " ".join(argv) and "verify" in argv


def _kind(argv):
    joined = " ".join(argv)
    if argv[0] == "ssh":
        if "mkdir -p" in joined:
            return "mkdir"
        if _is_probe(argv):
            return "probe"
        if _is_verify(argv):
            return "verify"
        if "CHECKSUMS.CURRENT.tmp" in joined and "mv -f" in joined:
            return "switch"
        return "ssh?"
    if argv[0] == "rsync":
        if "--files-from=-" in argv:
            return "repair" if "--checksum" in argv else "copy"
        if argv[-1].endswith("/manifests/"):
            return "manifest"
        if argv[-1].endswith(("/records/", "/rulings/")):
            return "tree"
    return "?"


def _stats(n):
    return f"\nNumber of files: 12 (reg: 10, dir: 2)\nNumber of regular files transferred: {n:,}\nTotal file size: 1,000 bytes\n"


def _clean_json(n, since):
    return json.dumps({"manifest": "x", "ok": n, "missing": [], "corrupt": [], "bytes": 1, "mode": "incremental", "since": since,
                       "hashed": 3, "presence_only": n - 3, "sampled": 2, "sample_seed": 7, "seconds": 1.5})


def _full_json(n):
    return json.dumps({"manifest": "x", "ok": n, "missing": [], "corrupt": [], "bytes": 99, "mode": "full", "since": None,
                       "hashed": n, "presence_only": 0, "sampled": None, "sample_seed": None, "seconds": 10.0})


def _delta(root, name, base):
    return M._on_disk_relpaths(root, M.added_since(os.path.join(root, M.MANIFEST_DIR, name), os.path.join(root, M.MANIFEST_DIR, base)))


def _two_manifests(tmp_path):
    """A base manifest the grading host wrote (the destination's pointer), two more files, the run's manifest."""
    root = _setup(tmp_path)
    base = M.write_manifest(root, _seq(root))
    _more(root, 1)
    _more(root, 2)
    mw = M.write_manifest_report(root, _seq(root))
    return root, base, mw


# ---- (a) the incremental copy: --files-from=-, exactly the added paths on stdin, in manifest order --------------------------

def test_incremental_copy_lists_exactly_the_delta_in_manifest_order_and_the_pointer_is_read_first(tmp_path, capsys):
    root, base, mw = _two_manifests(tmp_path)
    n = len(M.log_files(root))
    dest = _ssh_dest()
    rec = _Recorder(lambda argv: base if _is_probe(argv) else _clean_json(n, base) if _is_verify(argv)
                    else _stats(4) if _kind(argv) == "copy" else "")
    res = M.sync(root, dest, _seq(root), mw.name, runner=rec, manifest=mw)
    assert res.switched and res.error is None and res.verify.clean
    logical = M.added_since(os.path.join(root, M.MANIFEST_DIR, mw.name), os.path.join(root, M.MANIFEST_DIR, base))
    assert logical and logical == sorted(logical), "manifest order: sorted logical relpaths"
    base_rels = {rel for _, rel in M.read_manifest(os.path.join(root, M.MANIFEST_DIR, base))}
    assert not (set(logical) & base_rels), "nothing the base carries is listed"
    expected = M._on_disk_relpaths(root, logical)
    assert all(os.path.exists(os.path.join(root, p)) for p in expected) and len(expected) >= len(logical)
    copies = [(a, t, i) for a, t, i in rec.calls if _kind(a) == "copy"]
    assert len(copies) == 1
    argv, timeout_s, stdin = copies[0]
    assert argv == ["rsync", "-rt", f"--partial-dir={M.PARTIAL_DIR}", "--ignore-existing", "--timeout=120", "-e", M.RSYNC_SSH, "--stats",
                    "--files-from=-", root + "/", "login-0:/shared/u/registry/log/"]
    assert stdin == "\n".join(expected) + "\n" and timeout_s == M.COPY_TIMEOUT_S
    assert [_kind(a) for a in rec.argv] == ["mkdir", "probe", "copy", "manifest", "verify", "switch"], "the pointer is the copy's base"
    assert res.copy_mode == "incremental" and res.copy_since == base and res.copy_files == len(expected)
    assert res.copied == 4, "rsync --stats' transferred count"
    assert res.chunks_done == res.chunks_total == 1 and res.stopped is None and res.note is None
    assert res.mode == "incremental" and res.since == base, "the verify runs against the same base"
    out = capsys.readouterr().out
    assert f"[cluster_a] copy chunk 1/1: {len(expected)} files listed, 4 transferred" in out
    assert f"[cluster_a] copy incremental since {base}: {len(expected)} files listed in 1 chunk(s), 1 chunk(s) rsync'd, 4 transferred" in out


def test_added_since_is_the_verify_delta_and_names_a_changed_sha_too(tmp_path):
    root, base, mw = _two_manifests(tmp_path)
    mp, bp = os.path.join(root, M.MANIFEST_DIR, mw.name), os.path.join(root, M.MANIFEST_DIR, base)
    delta = M.added_since(mp, bp)
    entries = M.read_manifest(mp)
    b = {rel: sha for sha, rel in M.read_manifest(bp)}
    assert delta == [rel for sha, rel in entries if b.get(rel) != sha] and 0 < len(delta) < len(entries)
    assert any(rel.startswith("records/late/") for rel in delta)
    assert M.added_since(mp, mp) == [], "a manifest against itself: nothing"
    # an entry whose sha differs from the base's is in the delta (a rewrite never happens; if it did, the verify would hash it)
    sha0, rel0 = entries[0]
    with open(bp, "r", encoding="utf-8") as fh:
        lines = fh.read().splitlines()
    with open(bp, "w", encoding="utf-8") as fh:
        fh.write("\n".join(("0" * 64 + "  " + rel0) if line.endswith("  " + rel0) else line for line in lines) + "\n")
    assert rel0 in M.added_since(mp, bp)


# ---- (b) no acked manifest -> the whole tree; (c) --full -> the whole tree; a base gone from the grading host -> the whole tree + note --

def test_no_acked_manifest_or_full_rsyncs_the_whole_tree(tmp_path, capsys):
    root = _setup(tmp_path)
    mw = M.write_manifest_report(root, _seq(root))
    n = len(M.log_files(root))
    dest = _ssh_dest()
    rec = _Recorder(lambda argv: "" if _is_probe(argv) else _full_json(n) if _is_verify(argv) else _stats(3) if _kind(argv) == "tree" else "")
    res = M.sync(root, dest, _seq(root), mw.name, runner=rec, manifest=mw)
    assert res.switched and res.mode == "full"
    trees = [a for a in rec.argv if _kind(a) == "tree"]
    assert [a[-1] for a in trees] == ["login-0:/shared/u/registry/log/records/", "login-0:/shared/u/registry/log/rulings/"]
    for a in trees:
        assert a[:7] == ["rsync", "-rt", f"--partial-dir={M.PARTIAL_DIR}", "--ignore-existing", "--timeout=120", "-e", M.RSYNC_SSH]
        assert "--files-from=-" not in a and a[-2] in (root + "/records/", root + "/rulings/")
    assert [_kind(a) for a in rec.argv] == ["mkdir", "probe", "tree", "tree", "manifest", "verify", "switch"]
    assert res.copy_mode == "full" and res.copy_since is None and res.copy_files is None and res.copied == 6 and res.chunks_total == 0
    assert "[cluster_a] copy full (no acked manifest on the destination)" in capsys.readouterr().out
    # --full with a pointer on the destination: the whole tree, no pointer probe, a full verify
    rec2 = _Recorder(lambda argv: mw.name if _is_probe(argv) else _full_json(n) if _is_verify(argv) else "")
    res2 = M.sync(root, dest, _seq(root), mw.name, runner=rec2, manifest=mw, full=True)
    assert res2.switched and [_kind(a) for a in rec2.argv] == ["mkdir", "tree", "tree", "manifest", "verify", "switch"]
    assert res2.copy_mode == "full" and res2.copy_since is None and res2.copied == -1, "-1 when rsync printed no stats"
    assert "[cluster_a] copy full (--full)" in capsys.readouterr().out
    # the pointer names a manifest whose file the grading host no longer holds: the whole tree, the verify still incremental, the row says why
    gone = "CHECKSUMS-000000000001.sha256"
    rec3 = _Recorder(lambda argv: gone if _is_probe(argv) else _clean_json(n, gone) if _is_verify(argv) else "")
    res3 = M.sync(root, dest, _seq(root), mw.name, runner=rec3, manifest=mw)
    assert res3.switched and [_kind(a) for a in rec3.argv] == ["mkdir", "probe", "tree", "tree", "manifest", "verify", "switch"]
    assert res3.copy_mode == "full" and res3.copy_since is None and res3.mode == "incremental" and res3.since == gone
    assert res3.note and f"acked manifest {gone} is not under" in res3.note and "whole tree" in res3.note


# ---- (d) the verify follows the copy; a `missing` entry is re-copied and the destination verified again --------------------

def test_the_verify_follows_the_copy_and_recopies_what_it_finds_missing(tmp_path, capsys):
    root, base, mw = _two_manifests(tmp_path)
    n = len(M.log_files(root))
    dest = _ssh_dest()
    entries = M.read_manifest(os.path.join(root, M.MANIFEST_DIR, mw.name))
    carried = [rel for _, rel in entries if rel in {r for _, r in M.read_manifest(os.path.join(root, M.MANIFEST_DIR, base))}]
    lost, damaged = carried[0], carried[1]          # one the destination lost, one truncated there
    reports = iter([json.dumps({"manifest": mw.name, "ok": n - 2, "missing": [lost], "corrupt": [damaged], "bytes": 1, "mode": "incremental",
                                "since": base, "hashed": 4, "presence_only": n - 4, "sampled": 2, "sample_seed": 1, "seconds": 1.0}),
                    _clean_json(n, base)])
    rec = _Recorder(lambda argv: base if _is_probe(argv) else next(reports) if _is_verify(argv) else "")
    res = M.sync(root, dest, _seq(root), mw.name, runner=rec, manifest=mw)
    assert res.switched and res.verify.clean and res.error is None
    assert [_kind(a) for a in rec.argv] == ["mkdir", "probe", "copy", "manifest", "verify", "repair", "verify", "switch"]
    argv, timeout_s, stdin = [c for c in rec.calls if _kind(c[0]) == "repair"][0]
    assert "--ignore-existing" not in argv and "--checksum" in argv and argv[-2:] == [root + "/", "login-0:/shared/u/registry/log/"]
    assert stdin == "\n".join(M._on_disk_relpaths(root, [damaged, lost])) + "\n", "corrupt and missing, one list, the files as on disk"
    assert "[cluster_a] repair chunk 1/1: 2 files listed" in capsys.readouterr().out
    # a missing entry the grading host no longer holds either: nothing to re-copy, no second verify, no switch
    gone_rel = carried[2]
    for cand in M._resolve(root, gone_rel):
        os.remove(cand)
    bad = json.dumps({"manifest": mw.name, "ok": n - 1, "missing": [gone_rel], "corrupt": [], "bytes": 1, "mode": "incremental", "since": base,
                      "hashed": 3, "presence_only": n - 4, "sampled": 2, "sample_seed": 1, "seconds": 1.0})
    rec2 = _Recorder(lambda argv: base if _is_probe(argv) else bad if _is_verify(argv) else "")
    res2 = M.sync(root, dest, _seq(root), mw.name, runner=rec2, manifest=mw)
    assert not res2.switched and res2.verify.missing == [gone_rel] and res2.error is None
    assert [_kind(a) for a in rec2.argv] == ["mkdir", "probe", "copy", "manifest", "verify"]


# ---- (e) the result JSON names the copy mode and counts; the state file holds them ---------------------------------------

def test_the_row_names_the_copy_mode_base_and_counts(tmp_path, capsys):
    root = _setup(tmp_path)
    base = M.write_manifest(root, _seq(root))
    _more(root, 1)
    n = len(M.log_files(root))
    dest = _ssh_dest()
    rec = _Recorder(lambda argv: base if _is_probe(argv) else _clean_json(n, base) if _is_verify(argv)
                    else _stats(2) if _kind(argv) == "copy" else "")
    rows = M.sync_all(root, _seq(root), [dest], runner=rec, now=synth.t(9))
    row = rows[0]
    expected = _delta(root, row["manifest"], base)
    assert row["copy_mode"] == "incremental" and row["copy_since"] == base and row["copy_files"] == len(expected) > 0 and row["copied"] == 2
    assert row["chunks_done"] == 1 and row["chunks_total"] == 1 and row["stopped"] is None and row["switched"] and row["error"] is None
    assert row["mode"] == "incremental" and row["since"] == base, "the verify's own fields stay"
    assert os.path.exists(row["ack_record"])
    st = M.read_state(root)["destinations"][0]
    assert (st["copy_mode"], st["copy_since"], st["copy_files"], st["copied"]) == ("incremental", base, len(expected), 2)
    out = capsys.readouterr().out
    assert f"[cluster_a] verify {row['manifest']}: copy incremental {len(expected)} listed / 2 transferred, mode incremental" in out
    # the whole-tree row: copy_files null, copy_since null
    rec2 = _Recorder(lambda argv: "" if _is_probe(argv) else _full_json(n) if _is_verify(argv) else "")
    row2 = M.sync_all(root, _seq(root), [dest], runner=rec2, now=synth.t(10))[0]
    assert row2["copy_mode"] == "full" and row2["copy_since"] is None and row2["copy_files"] is None and row2["copied"] == -1
    assert f"copy full - listed / -1 transferred" in capsys.readouterr().out
    # a local destination's row says full (its copy is the in-process pass)
    local = M.Destination("m", "local", str(tmp_path / "mirror"))
    row3 = M.sync_all(root, _seq(root), [local], now=synth.t(11))[0]
    assert row3["copy_mode"] == "full" and row3["copy_since"] is None and row3["copy_files"] is None and row3["copied"] > 0


# ---- (f) two destinations, ONE invocation: each delta from ITS OWN pointer --------------------------------------------------

def test_two_destinations_in_one_invocation_take_their_deltas_from_their_own_pointers(tmp_path):
    root = _setup(tmp_path)
    base_b = M.write_manifest(root, _seq(root))          # cluster A acked here …
    _more(root, 1)
    base_c = M.write_manifest(root, _seq(root))          # … cluster B one cycle later
    _more(root, 2)
    n = len(M.log_files(root))
    bulb, char = _ssh_dest(), _ssh_dest("cluster_b", "cluster_b", "/workspace/u")

    def respond(argv):
        base = base_b if "login-0" in argv else base_c
        if _is_probe(argv):
            return base
        if _is_verify(argv):
            return _clean_json(n, base)
        return _stats(1) if _kind(argv) == "copy" else ""

    rec = _Recorder(respond)
    rows = M.sync_all(root, _seq(root), [bulb, char], runner=rec, now=synth.t(5))
    assert [r["destination"] for r in rows] == ["cluster_a", "cluster_b"] and rows[0]["manifest"] == rows[1]["manifest"], "one manifest"
    name = rows[0]["manifest"]
    delta_b, delta_c = _delta(root, name, base_b), _delta(root, name, base_c)
    assert len(delta_b) > len(delta_c) > 0 and set(delta_c) < set(delta_b)
    copies = [(a, i) for a, _, i in rec.calls if _kind(a) == "copy"]
    assert len(copies) == 2
    assert copies[0][0][-1] == "login-0:/shared/u/registry/log/" and copies[0][1] == "\n".join(delta_b) + "\n"
    assert copies[1][0][-1] == "cluster_b:/workspace/u/registry/log/" and copies[1][1] == "\n".join(delta_c) + "\n"
    assert [r["copy_since"] for r in rows] == [base_b, base_c] and [r["copy_files"] for r in rows] == [len(delta_b), len(delta_c)]
    assert all(r["switched"] and r["copy_mode"] == "incremental" and r["copied"] == 1 for r in rows)
    probes = [a for a in rec.argv if _is_probe(a)]
    assert len(probes) == 2 and [a[5] for a in probes] == ["login-0", "cluster_b"], "one pointer read per destination, its own host"
    assert [r["since"] for r in rows] == [base_b, base_c], "each verify against its own base too"


# ---- chunking: > COPY_CHUNK_FILES paths -> several --files-from calls, consecutive slices, a journal line each ------------

def test_a_large_delta_rides_in_chunks_with_a_journal_line_each(tmp_path, monkeypatch, capsys):
    root, base, mw = _two_manifests(tmp_path)
    n = len(M.log_files(root))
    expected = _delta(root, mw.name, base)
    assert len(expected) >= 3
    monkeypatch.setattr(M, "COPY_CHUNK_FILES", 2)
    dest = _ssh_dest()
    rec = _Recorder(lambda argv: base if _is_probe(argv) else _clean_json(n, base) if _is_verify(argv)
                    else _stats(1) if _kind(argv) == "copy" else "")
    res = M.sync(root, dest, _seq(root), mw.name, runner=rec, manifest=mw)
    copies = [i for a, _, i in rec.calls if _kind(a) == "copy"]
    want = ["\n".join(expected[i:i + 2]) + "\n" for i in range(0, len(expected), 2)]
    assert copies == want and len(copies) == (len(expected) + 1) // 2
    assert res.switched and res.chunks_done == res.chunks_total == len(want) and res.copied == len(want) and res.copy_files == len(expected)
    out = capsys.readouterr().out
    for k in range(1, len(want) + 1):
        assert f"[cluster_a] copy chunk {k}/{len(want)}: {len(want[k - 1].split())} files listed, 1 transferred" in out
    assert M.COPY_CHUNK_FILES == 2 and M.__dict__["COPY_CHUNK_FILES"] == 2
    monkeypatch.undo()
    assert M.COPY_CHUNK_FILES == 20000, "the shipped bound"


def test_rsync_list_chunks_in_order_and_stops_on_the_budget(monkeypatch, capsys):
    calls = []

    def run(argv, timeout_s, input=None, check=True):
        calls.append((list(argv), timeout_s, input))
        return subprocess.CompletedProcess(list(argv), 0, stdout=_stats(2), stderr="")

    rels = [f"records/x/{i:06d}-a.jsonl" for i in range(5)]
    done, total, transferred, stopped = M._rsync_list(run, ["rsync", "--files-from=-", "s/", "h:d/"], rels, 77, "[t] copy", chunk_files=2)
    assert (done, total, transferred, stopped) == (3, 3, 6, None)
    assert [i for _, _, i in calls] == ["records/x/000000-a.jsonl\nrecords/x/000001-a.jsonl\n", "records/x/000002-a.jsonl\nrecords/x/000003-a.jsonl\n",
                                        "records/x/000004-a.jsonl\n"]
    assert all(t == 77 for _, t, _ in calls)
    assert M._rsync_list(run, ["rsync"], [], 1, "[t] copy") == (0, 0, 0, None), "an empty delta: no call"
    # a clock that advances 50 s per reading: chunk 1 runs (400 s left), chunk 2 runs (200 s left >= 1.5 x 50), chunk 3 would start with
    # 50 s left < 75: stopped, the two chunks done stand, the third rides the next run

    class Clock:
        t = 0.0

        def monotonic(self):
            self.t += 50.0
            return self.t

    clock = Clock()
    monkeypatch.setattr(M, "time", clock)
    calls.clear()
    done, total, transferred, stopped = M._rsync_list(run, ["rsync"], rels, 1, "[t] copy", deadline=400.0, chunk_files=2)
    assert (done, total, transferred, stopped) == (2, 3, 4, "budget") and len(calls) == 2
    assert "[t] copy: budget: stopped after 2/3 chunks (50 s left, last chunk 50 s); the rest rides the next run" in capsys.readouterr().out
    # a deadline already past: nothing starts
    calls.clear()
    assert M._rsync_list(run, ["rsync"], rels, 1, "[t] copy", deadline=0.0, chunk_files=2) == (0, 3, 0, "budget") and not calls


def test_a_budget_stop_returns_a_row_without_a_verify_that_the_cli_counts_as_progress(tmp_path, capsys):
    root, base, mw = _two_manifests(tmp_path)
    n = len(M.log_files(root))
    dest = _ssh_dest()
    rec = _Recorder(lambda argv: base if _is_probe(argv) else _clean_json(n, base) if _is_verify(argv) else "")
    res = M.sync(root, dest, _seq(root), mw.name, runner=rec, manifest=mw, deadline=0.0)
    assert res.stopped == "budget" and not res.switched and res.error is None and res.chunks_done == 0 and res.chunks_total == 1
    assert res.copy_mode == "incremental" and res.copy_since == base and res.copy_files == len(_delta(root, mw.name, base))
    assert [_kind(a) for a in rec.argv] == ["mkdir", "probe"], "no rsync, no verify, no switch: the next run lists the same delta"
    rows = M.sync_all(root, _seq(root), [dest], runner=rec, now=synth.t(3), deadline=0.0)
    r = rows[0]
    assert r["stopped"] == "budget" and not r["switched"] and r["error"] is None and "ack_record" not in r
    assert r["switched"] or (r.get("stopped") == "budget" and not r.get("error")), "cmd_mirrors' exit-0 condition"
    assert "STOPPED budget" in capsys.readouterr().out
    # no deadline (a direct call): never stops
    rec2 = _Recorder(lambda argv: base if _is_probe(argv) else _clean_json(n, base) if _is_verify(argv) else "")
    assert M.sync(root, dest, _seq(root), mw.name, runner=rec2, manifest=mw).switched


# ---- rsync exits: 23/24 on the copy are a journal line, the verify decides; anything else is the destination's error ---------

def test_partial_transfer_exits_are_tolerated_on_the_copy_and_other_exits_are_the_error(tmp_path, capsys):
    root, base, mw = _two_manifests(tmp_path)
    n = len(M.log_files(root))
    dest = _ssh_dest()
    rec = _Recorder(lambda argv: base if _is_probe(argv) else _clean_json(n, base) if _is_verify(argv)
                    else (23, _stats(3), "rsync: link_stat failed: No such file or directory (2)\nrsync error: some files could not be transferred (code 23)")
                    if _kind(argv) == "copy" else "")
    res = M.sync(root, dest, _seq(root), mw.name, runner=rec, manifest=mw)
    assert res.switched and res.error is None and res.copied == 3
    assert [_kind(a) for a in rec.argv] == ["mkdir", "probe", "copy", "manifest", "verify", "switch"]
    assert "[cluster_a] copy chunk 1/1: rsync exit 23 (partial transfer; the verify decides what arrived): rsync: link_stat" in capsys.readouterr().out
    rec24 = _Recorder(lambda argv: "" if _is_probe(argv) else _full_json(n) if _is_verify(argv) else (24, "", "vanished") if _kind(argv) == "tree" else "")
    assert M.sync(root, dest, _seq(root), mw.name, runner=rec24, manifest=mw).switched, "24 on the whole tree too"
    # exit 12 (protocol / transport): the destination's error, no verify, no switch
    rec12 = _Recorder(lambda argv: base if _is_probe(argv) else _clean_json(n, base) if _is_verify(argv)
                      else (12, "", "rsync: connection unexpectedly closed") if _kind(argv) == "copy" else "")
    res12 = M.sync(root, dest, _seq(root), mw.name, runner=rec12, manifest=mw)
    assert not res12.switched and res12.error and "12" in res12.error and "connection unexpectedly closed" in res12.error
    assert [_kind(a) for a in rec12.argv] == ["mkdir", "probe", "copy"]
    assert res12.copy_mode == "incremental" and res12.copy_since == base, "a failed row still says which copy ran"
    assert M.RSYNC_PARTIAL_EXITS == (23, 24)


def test_rsync_transferred_reads_both_stats_spellings_and_thousands_separators():
    assert M._rsync_transferred("Number of regular files transferred: 1,234\n") == 1234
    assert M._rsync_transferred("Number of files transferred: 7\n") == 7, "rsync < 3.1"
    assert M._rsync_transferred(_stats(461769)) == 461769
    assert M._rsync_transferred("") is None and M._rsync_transferred("sending incremental file list\n") is None
    cmd = M._rsync_files_cmd("/r/", "h:/d/")
    assert cmd[:7] == M._rsync_cmd("/r/", "h:/d/")[:7] and "--stats" in cmd and "--ignore-existing" in cmd and "--checksum" not in cmd
    assert cmd[-3:] == ["--files-from=-", "/r/", "h:/d/"]
    assert "--stats" in M._rsync_cmd("/r/", "h:/d/") and M._rsync_cmd("/r/", "h:/d/")[-2:] == ["/r/", "h:/d/"]
