"""Manifest retention for the record-log mirror (2026-10-08). Every ``mirrors sync`` pass wrote a NEW ~80 MB
``manifests/CHECKSUMS-<seq>.sha256`` and copied it to each cluster mirror, and nothing pruned them: the grading host held 1,248
(72 GB on a root disk at 96 %; a full disk pauses the registry) and each cluster ~1,110 copies. ONE rule
(``mirrors.plan_manifest_retention``) now applies to the grading host's ``manifests/`` and to each destination's, run by the pass
(``sync_all``, after every destination's sync) and by ``registry mirrors prune-manifests`` through the same function
(``mirrors.prune_manifests``): keep the side's own ``CHECKSUMS.CURRENT``, the newest ``MANIFEST_KEEP`` by ingest_seq, and every
manifest a live pointer names -- a destination's pointer is the ``--since`` base of its next incremental copy and verify, which the
grading host must still hold for the delta -- and delete the rest. A dry run deletes nothing and reports the plan; a destination's
listing or deletion failing is that side's error, never the pass's; a pass whose grading host plan deletes nothing lists no destination.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

import pytest

from _env import permission_denial_enforced
from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import mirrors as M
from sisteer.registry import records as R
from sisteer.registry import synth


def _setup(tmp_path, n_runs=1):
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


def _is_probe(argv):
    return argv[0] == "ssh" and "CHECKSUMS.CURRENT" in " ".join(argv) and "test -f" in " ".join(argv)


def _is_verify(argv):
    return argv[0] == "ssh" and "sisteer.registry.mirrors" in " ".join(argv) and "verify" in argv


def _is_listing(argv):
    return argv[0] == "ssh" and "find " in " ".join(argv) and "-maxdepth 1" in " ".join(argv)


def _is_delete(argv):
    return argv[0] == "ssh" and "xargs -r rm -f --" in " ".join(argv)


def _is_manifest_copy(argv):
    return argv[0] == "rsync" and argv[-1].endswith("/manifests/")


def _clean_json(n, since):
    return json.dumps({"manifest": "x", "ok": n, "missing": [], "corrupt": [], "bytes": 1, "mode": "incremental", "since": since,
                       "hashed": 3, "presence_only": n - 3, "sampled": 2, "sample_seed": 7, "seconds": 1.5})


def _listing_text(current, names, size=10):
    """What the fail-closed listing shell prints: the pointer line, `---`, `<bytes> <name>` per manifest, `FIND-OK`."""
    return (f"POINTER {current}" if current else "POINTER-ABSENT") + "\n---\n" + "".join(f"{size} {x}\n" for x in sorted(names)) + "FIND-OK\n"


def _full_json(n):
    return json.dumps({"manifest": "x", "ok": n, "missing": [], "corrupt": [], "bytes": 1, "mode": "full", "hashed": n, "presence_only": 0,
                       "seconds": 1.0})


def _is_cat(argv):
    j = " ".join(argv)
    return argv[0] == "ssh" and j.strip().endswith("|| true") and "cat " in j and "CHECKSUMS.CURRENT" in j and "find" not in j


def _sh(argv, timeout_s, input=None, check=True):
    """A runner that RUNS the remote command locally through `sh -c`, stdin included: the generated listing and deletion shells
    themselves (`cat`/`find` with their markers, `xargs rm -f` over the names on stdin), not a mock of their exit."""
    return subprocess.run(["sh", "-c", argv[-1]], capture_output=True, text=True, input=input)


def _manifests_on_both(root, dest_path, seqs):
    """One manifest per seq on the grading host, each copied to the destination's ``manifests/`` (what a pass's copy does)."""
    names = []
    os.makedirs(os.path.join(dest_path, M.MANIFEST_DIR), exist_ok=True)
    for q in seqs:
        n = M.write_manifest(root, q)
        names.append(n)
        shutil.copy2(os.path.join(root, M.MANIFEST_DIR, n), os.path.join(dest_path, M.MANIFEST_DIR, n))
    return names


def _grading_host_names(root):
    return sorted(os.listdir(os.path.join(root, M.MANIFEST_DIR)))


def _dest_names(path):
    return sorted(n for n in os.listdir(os.path.join(path, M.MANIFEST_DIR)) if n != M.CURRENT_NAME)


REASON = "'s CHECKSUMS.CURRENT: the --since base of its next incremental copy and verify"


# ------------------------------------------------------------------------------------------------------------ the rule, pure
def test_the_rule_keeps_current_the_newest_k_and_the_referenced_and_deletes_the_rest():
    names = [f"CHECKSUMS-{i:012d}.sha256" for i in range(10, 110, 10)]       # ten manifests, seq 10 .. 100
    listing = {n: 1000 + i for i, n in enumerate(names)}
    listing["CHECKSUMS-000000000050.sha256.tmp"] = 5                           # a manifest mid-write: not a manifest
    listing[M.CURRENT_NAME] = 29                                               # the pointer file: never
    listing["notes.txt"] = 3
    current = names[1]                                                         # the side's own pointer, an old one
    refs = {names[3]: ["cluster_a" + REASON], "CHECKSUMS-000000009999.sha256": ["a reference the listing lacks"]}
    plan = M.plan_manifest_retention(listing, current, refs, keep=3, side="grading_host", location="/x/manifests")
    assert plan.kept == [names[1], names[3], names[7], names[8], names[9]]
    assert plan.deleted == [names[0], names[2], names[4], names[5], names[6]]
    assert plan.kept_bytes == sum(listing[n] for n in plan.kept) and plan.deleted_bytes == sum(listing[n] for n in plan.deleted)
    assert plan.oldest_kept == names[1] and plan.newest_kept == names[9] and plan.current == current and plan.keep == 3
    assert plan.referenced == {names[3]: refs[names[3]]}, "an absent reference has nothing to keep"
    d = plan.as_dict()
    assert (d["manifests"], d["kept"], d["deleted"], d["deleted_names"], d["removed"], d["dry_run"], d["error"]) == (10, 5, 5, plan.deleted, 0, False, None)
    assert d["referenced"] == plan.referenced and d["side"] == "grading_host" and d["location"] == "/x/manifests"
    # keep covering the listing: nothing deleted; a pointer outside the listing: nothing to keep; keep < 1 refused
    assert M.plan_manifest_retention(listing, None, {}, keep=10).deleted == []
    assert M.plan_manifest_retention(listing, "CHECKSUMS-000000000777.sha256", {}, keep=1).kept == [names[9]]
    with pytest.raises(ValueError):
        M.plan_manifest_retention(listing, None, {}, keep=0)
    assert M.is_manifest_name(names[0]) and not M.is_manifest_name("CHECKSUMS-1.sha256") and not M.is_manifest_name(names[0] + ".tmp")
    assert not M.is_manifest_name(M.CURRENT_NAME) and not M.is_manifest_name("")
    empty = M.plan_manifest_retention({}, None, {}, keep=2)
    assert empty.kept == [] and empty.deleted == [] and empty.oldest_kept is None and empty.newest_kept is None


# ----------------------------------------------------------------------------------- the grading host and a local destination
def test_prune_keeps_current_newest_k_and_referenced_on_the_grading_host_and_a_local_destination(tmp_path):
    root = _setup(tmp_path)
    dest = M.Destination("cluster_a", "local", str(tmp_path / "mirror"))
    first = M.sync(root, dest, _seq(root))                     # the destination's pointer: this manifest, soon an old one
    assert first.switched and first.dest_current == first.manifest == M.current_manifest(dest.path)
    later = _manifests_on_both(root, dest.path, [_seq(root) + 10 * i for i in range(1, 8)])   # seven more passes
    all_names = sorted([first.manifest] + later)
    assert _grading_host_names(root) == all_names == _dest_names(dest.path)
    assert M.list_manifests(os.path.join(root, M.MANIFEST_DIR)) == {n: os.path.getsize(os.path.join(root, M.MANIFEST_DIR, n)) for n in all_names}
    assert M.list_manifests(str(tmp_path / "absent")) == {}
    total_bytes = sum(M.list_manifests(os.path.join(dest.path, M.MANIFEST_DIR)).values())
    summary = M.prune_manifests(root, [dest], keep=2)
    grading_host, side = summary["sides"]
    expect_keep = [first.manifest] + all_names[-2:]
    # the destination's pointer, an OLD manifest, is a live reference: kept on BOTH sides beyond the newest 2, with its reason
    assert summary["keep"] == 2 and not summary["dry_run"] and summary["references"] == {first.manifest: ["cluster_a" + REASON]}
    assert grading_host["side"] == "grading_host" and grading_host["location"] == os.path.join(root, M.MANIFEST_DIR) and grading_host["current"] is None
    assert (grading_host["manifests"], grading_host["kept"], grading_host["deleted"], grading_host["removed"], grading_host["error"]) == (8, 3, 5, 5, None)
    assert grading_host["oldest_kept"] == first.manifest and grading_host["newest_kept"] == all_names[-1]
    assert grading_host["referenced"] == summary["references"] and grading_host["deleted_names"] == all_names[1:6]
    assert grading_host["kept_bytes"] + grading_host["deleted_bytes"] == side["kept_bytes"] + side["deleted_bytes"] == total_bytes > 0
    assert side["side"] == "cluster_a" and side["location"] == os.path.join(dest.path, M.MANIFEST_DIR) and side["current"] == first.manifest
    assert (side["manifests"], side["kept"], side["deleted"], side["removed"], side["error"]) == (8, 3, 5, 5, None)
    assert _grading_host_names(root) == expect_keep and _dest_names(dest.path) == expect_keep
    assert M.current_manifest(dest.path) == first.manifest, "the pointer file itself is never touched"
    # idempotent: a second run has nothing left to delete; a bigger keep deletes nothing either
    again = M.prune_manifests(root, [dest], keep=2)
    assert [s["deleted"] for s in again["sides"]] == [0, 0] and [s["kept"] for s in again["sides"]] == [3, 3]
    assert [s["deleted"] for s in M.prune_manifests(root, [dest], keep=50)["sides"]] == [0, 0]
    with pytest.raises(ValueError):
        M.prune_manifests(root, [dest], keep=0)


# --------------------------------------------------------------------------------------------- the dry run and the CLI
def test_a_dry_run_deletes_nothing_and_reports_the_plan_and_the_cli_runs_the_same_rule(tmp_path, monkeypatch, capsys):
    root = _setup(tmp_path)
    dest_path = str(tmp_path / "mirror")
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump({"mirrors": [{"name": "cluster_a", "kind": "local", "path": dest_path}]}, fh)
    dest = M.destinations(root)[0]
    first = M.sync(root, dest, _seq(root))
    later = _manifests_on_both(root, dest_path, [_seq(root) + 10 * i for i in range(1, 6)])     # six in all
    all_names = sorted([first.manifest] + later)
    expect_delete, expect_keep = all_names[1:4], [first.manifest] + all_names[-2:]
    summary = M.prune_manifests(root, [dest], keep=2, dry_run=True)
    assert summary["dry_run"] and all(s["dry_run"] and s["removed"] == 0 and s["error"] is None for s in summary["sides"])
    assert [s["deleted_names"] for s in summary["sides"]] == [expect_delete, expect_delete]
    assert _grading_host_names(root) == all_names and _dest_names(dest_path) == all_names, "a dry run deletes nothing"
    text = M.format_retention(summary)
    assert text.splitlines()[0].startswith("manifest retention: keep the newest 2 by ingest_seq") and "DRY RUN: nothing deleted" in text
    assert f"  {first.manifest}: cluster_a{REASON}" in text
    assert f"[retention] grading_host {os.path.join(root, M.MANIFEST_DIR)}: 6 manifest(s) (" in text
    assert "keep 3 (" in text and "delete 3 (" in text and text.count(": dry run") == 2
    assert f"oldest {first.manifest}, newest {all_names[-1]}; 1 by reference" in text
    assert f"[retention] cluster_a {os.path.join(dest_path, M.MANIFEST_DIR)}: 6 manifest(s)" in text
    assert len(M.format_retention(summary, header=False).splitlines()) == 2
    # the CLI's dry run: exit 0, nothing deleted, NO state/last_run stamp (a dry run writes nothing under the root); --json after the text
    monkeypatch.setenv("REGISTRY_ROOT", root)
    stamp = os.path.join(root, "state", "last_run", "mirrors-prune-manifests.json")
    assert cli.main(["mirrors", "prune-manifests", "--dry-run", "--keep", "2", "--json"]) == 0
    out = capsys.readouterr().out
    assert "DRY RUN: nothing deleted" in out and json.loads(out.strip().splitlines()[-1])["sides"][0]["deleted_names"] == expect_delete
    assert _grading_host_names(root) == all_names and _dest_names(dest_path) == all_names and not os.path.exists(stamp)
    # the real run: deletes on both sides, the pointer survives, exit 0, the stamp is this command's own file
    assert cli.main(["mirrors", "prune-manifests", "--keep", "2"]) == 0
    out = capsys.readouterr().out
    assert out.count(": removed 3") == 2 and "DRY RUN" not in out and "applied" in out.splitlines()[0]
    assert _grading_host_names(root) == expect_keep and _dest_names(dest_path) == expect_keep and M.current_manifest(dest_path) == first.manifest
    assert json.load(open(stamp))["rc"] == 0 and not os.path.exists(os.path.join(root, "state", "last_run", "mirrors-sync.json"))
    # refused shapes: --keep below 1; --dry-run / --keep with another action; the default keep is the module constant
    assert cli.main(["mirrors", "prune-manifests", "--keep", "0"]) == 2
    assert cli.main(["mirrors", "sync", "--dry-run"]) == 2 and cli.main(["mirrors", "verify", "--keep", "3"]) == 2
    assert M.MANIFEST_KEEP == 96
    assert cli.main(["mirrors", "prune-manifests", "--dry-run"]) == 0
    assert "keep the newest 96 by ingest_seq" in capsys.readouterr().out


# ------------------------------------------------------------------------------------------ the pass, an ssh destination
def test_the_pass_prunes_after_writing_and_keeps_the_destination_base_when_it_did_not_switch(tmp_path, capsys):
    root = _setup(tmp_path)
    dest = _ssh_dest()
    base = M.write_manifest(root, _seq(root))                                     # the destination's acked manifest, its pointer
    extra = [M.write_manifest(root, _seq(root) + 10 * i) for i in range(1, 5)]    # four more passes' manifests
    _more(root, 1)
    n = len(M.log_files(root))
    remote = {x: 1 for x in [base] + extra}                                       # the destination's copies, the pass's new one on arrival

    def respond(argv):
        if _is_probe(argv):
            return base
        if _is_verify(argv):
            return json.dumps({"ok": n - 1, "missing": ["records/x"], "corrupt": [], "mode": "incremental", "since": base,
                               "hashed": 1, "presence_only": n - 2})
        if _is_manifest_copy(argv):
            remote[os.path.basename(argv[-2])] = 1
            return ""
        if _is_listing(argv):
            return _listing_text(base, remote, size=77)
        return ""

    rec = _Recorder(respond)
    rows = M.sync_all(root, _seq(root) + 100, [dest], runner=rec, now=synth.t(3), keep=2)
    row = rows[0]
    new = row["manifest"]
    assert not row["switched"] and row["error"] is None and row["dest_current"] == base, "no switch: the pointer stays at the base"
    # the grading host after the pass: the newest 2 (the last extra and the pass's own) plus the destination's base; the three between deleted
    assert _grading_host_names(root) == sorted([base, extra[-1], new])
    st = M.read_state(root)
    assert st["destinations"][0]["dest_current"] == base
    ret = st["manifest_retention"]
    assert ret["keep"] == 2 and not ret["dry_run"] and ret["references"] == {base: ["cluster_a" + REASON]}
    dev, side = ret["sides"]
    assert dev["side"] == "grading_host" and (dev["manifests"], dev["kept"], dev["deleted"], dev["removed"]) == (6, 3, 3, 3)
    assert dev["referenced"] == ret["references"] and dev["deleted_names"] == extra[:3] and dev["oldest_kept"] == base and dev["newest_kept"] == new
    # the destination: ONE listing ssh after the verify, ONE deletion ssh with the same names on stdin, bounded; the pointer probe
    # is not mistaken for the listing (the other tests count probes)
    listings = [c for c in rec.calls if _is_listing(c[0])]
    deletes = [c for c in rec.calls if _is_delete(c[0])]
    assert len(listings) == 1 and len(deletes) == 1 and sum(1 for c in rec.calls if _is_probe(c[0])) == 1
    assert rec.calls.index(listings[0]) > max(i for i, c in enumerate(rec.calls) if _is_verify(c[0]))
    assert rec.calls.index(deletes[0]) > rec.calls.index(listings[0])
    assert listings[0][1] == M.PRUNE_LIST_TIMEOUT_S and listings[0][0][:5] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    assert deletes[0][1] == M.PRUNE_DELETE_TIMEOUT_S and deletes[0][2] == "\n".join(extra[:3]) + "\n"
    assert deletes[0][0][-1] == f"cd {dest.path}/manifests && xargs -r rm -f --"
    assert side["side"] == "cluster_a" and side["location"] == f"login-0:{dest.path}/manifests" and side["current"] == base
    assert (side["manifests"], side["kept"], side["deleted"], side["removed"], side["error"]) == (6, 3, 3, 3, None)
    assert side["kept_bytes"] == 3 * 77 and side["deleted_bytes"] == 3 * 77
    out = capsys.readouterr().out
    assert f"[retention] grading_host {os.path.join(root, M.MANIFEST_DIR)}: 6 manifest(s)" in out and ": removed 3" in out
    assert f"[retention] cluster_a login-0:{dest.path}/manifests: 6 manifest(s)" in out
    # the next pass: the destination's pointer is now a pass-switched manifest; the state's last-known pointer protects it on the
    # grading host even when the sync never reads it (an error before the probe), and an errored destination is not listed
    def respond2(argv):
        if argv[0] == "ssh" and "mkdir -p" in " ".join(argv):
            raise subprocess.SubprocessError("ssh: connect to host login-0: Connection timed out")
        return ""

    rec2 = _Recorder(respond2)
    rows2 = M.sync_all(root, _seq(root) + 200, [dest], runner=rec2, now=synth.t(4), keep=1)
    assert rows2[0]["error"] and rows2[0]["dest_current"] is None and not rows2[0]["switched"]
    assert not any(_is_listing(c[0]) or _is_delete(c[0]) for c in rec2.calls), "a destination the sync did not reach is not listed"
    ret2 = M.read_state(root)["manifest_retention"]
    assert [s["side"] for s in ret2["sides"]] == ["grading_host"] and ret2["references"] == {base: ["cluster_a" + REASON]}
    assert _grading_host_names(root) == sorted([base, rows2[0]["manifest"]]), "newest 1 + the destination's base from the state"


def test_a_remote_prune_failure_is_that_sides_error_and_never_the_passes(tmp_path):
    root = _setup(tmp_path)
    dest = _ssh_dest()
    base = M.write_manifest(root, _seq(root))
    extra = [M.write_manifest(root, _seq(root) + 10 * i) for i in range(1, 4)]
    _more(root, 1)
    n = len(M.log_files(root))
    remote = {"pointer": base, "names": {x: 1 for x in [base] + extra}}       # the destination: its pointer follows the switch

    def respond(argv):
        joined = " ".join(argv)
        if _is_probe(argv):
            return base
        if _is_verify(argv):
            return _clean_json(n, base)
        if _is_manifest_copy(argv):
            remote["names"][os.path.basename(argv[-2])] = 1
            return ""
        if "CHECKSUMS.CURRENT.tmp" in joined and "mv -f" in joined:
            remote["pointer"] = joined.split("printf %s '", 1)[1].split("'", 1)[0]
            return ""
        if _is_listing(argv):
            return _listing_text(remote["pointer"], remote["names"])
        if _is_delete(argv):
            return (1, "", "rm: cannot remove 'CHECKSUMS-000000000001.sha256': Read-only file system")
        return ""

    rec = _Recorder(respond)
    rows = M.sync_all(root, _seq(root) + 100, [dest], runner=rec, now=synth.t(4), keep=1)
    row = rows[0]
    assert row["switched"] and row["error"] is None and row["dest_current"] == row["manifest"] and os.path.exists(row["ack_record"])
    assert remote["pointer"] == row["manifest"]
    ret = M.read_state(root)["manifest_retention"]
    dev, side = ret["sides"]
    assert (dev["deleted"], dev["removed"], dev["error"]) == (4, 4, None), "the grading host pruned regardless"
    assert _grading_host_names(root) == [row["manifest"]]
    assert side["current"] == row["manifest"] and side["deleted"] == 4 and side["removed"] == 0
    assert side["error"].startswith("deletion:") and "exited 1" in side["error"] and "Read-only file system" in side["error"]
    deletes = [c for c in rec.calls if _is_delete(c[0])]
    assert len(deletes) == 1 and deletes[0][2] == "\n".join([base] + extra) + "\n", "the same names the grading host deleted"
    # the listing failing is that side's error too; the deletion is not attempted; the pass is still a success
    cur = row["manifest"]

    def respond2(argv):
        if _is_probe(argv):
            return cur
        if _is_verify(argv):
            return _clean_json(n, cur)
        if _is_listing(argv):
            return (255, "", "ssh: connect to host login-0 port 22: Connection timed out")
        return ""

    rec2 = _Recorder(respond2)
    rows2 = M.sync_all(root, _seq(root) + 200, [dest], runner=rec2, now=synth.t(5), keep=1)
    assert rows2[0]["switched"] and rows2[0]["error"] is None
    side2 = M.read_state(root)["manifest_retention"]["sides"][1]
    assert side2["error"].startswith("listing:") and "exited 255" in side2["error"] and side2["deleted"] == 0 and side2["removed"] == 0
    assert not any(_is_delete(c[0]) for c in rec2.calls)
    assert _grading_host_names(root) == [rows2[0]["manifest"]]
    # an unparsable listing is the same class of error
    rec3 = _Recorder(lambda argv: rows2[0]["manifest"] if _is_probe(argv) else _clean_json(n, rows2[0]["manifest"]) if _is_verify(argv)
                     else "bash: find: command not found" if _is_listing(argv) else "")
    M.sync_all(root, _seq(root) + 300, [dest], runner=rec3, now=synth.t(6), keep=1)
    side3 = M.read_state(root)["manifest_retention"]["sides"][1]
    assert side3["error"].startswith("listing:") and "pointer line" in side3["error"]


def test_a_pass_whose_grading_host_plan_deletes_nothing_lists_no_destination_and_prune_false_skips_the_rule(tmp_path, capsys):
    root = _setup(tmp_path)
    dest = _ssh_dest()
    base = M.write_manifest(root, _seq(root))
    _more(root, 1)
    n = len(M.log_files(root))
    rec = _Recorder(lambda argv: base if _is_probe(argv) else _clean_json(n, base) if _is_verify(argv) else "")
    rows = M.sync_all(root, _seq(root), [dest], runner=rec, now=synth.t(2))            # the default keep: two manifests, nothing to delete
    assert rows[0]["switched"] and rows[0]["dest_current"] == rows[0]["manifest"]
    assert not any(_is_listing(c[0]) or _is_delete(c[0]) for c in rec.calls), "no extra ssh on a young registry"
    assert "[retention]" not in capsys.readouterr().out, "an idle side is silent (a local-only pass printed nothing before the rule)"
    ret = M.read_state(root)["manifest_retention"]
    assert ret["keep"] == M.MANIFEST_KEEP == 96 and [s["side"] for s in ret["sides"]] == ["grading_host"]
    assert ret["sides"][0]["deleted"] == 0 and ret["sides"][0]["kept"] == 2 and ret["references"] == {rows[0]["manifest"]: ["cluster_a" + REASON]}
    # the CLI lists the destination regardless (a one-off catch-up reaches names the grading host no longer holds)
    rec_cli = _Recorder(lambda argv: _listing_text(rows[0]["manifest"], [rows[0]["manifest"], "CHECKSUMS-000000000001.sha256"]) if _is_listing(argv) else "")
    summary = M.prune_manifests(root, [dest], keep=96, runner=rec_cli)
    assert [s["side"] for s in summary["sides"]] == ["grading_host", "cluster_a"] and summary["sides"][1]["deleted"] == 0
    assert sum(1 for c in rec_cli.calls if _is_listing(c[0])) == 1
    # prune=False: no rule, no section
    rec2 = _Recorder(lambda argv: rows[0]["manifest"] if _is_probe(argv) else _clean_json(n, rows[0]["manifest"]) if _is_verify(argv) else "")
    M.sync_all(root, _seq(root) + 1, [dest], runner=rec2, now=synth.t(3), prune=False)
    assert "manifest_retention" not in M.read_state(root)


# --------------------------------------------------------------------------------- the remote listing and the state's pointers
def test_remote_listing_parse_commands_and_last_known_pointers_from_the_state(tmp_path):
    a, b = "CHECKSUMS-000000000001.sha256", "CHECKSUMS-000000000002.sha256"
    ok = f"POINTER {b}\n---\n10 {a}\n20 {b}\n7 {M.CURRENT_NAME}\n5 {a}.tmp\n\nFIND-OK\n"
    assert M.parse_remote_listing(ok) == (b, {a: 10, b: 20}), "names that are not a manifest's shape are not ours"
    assert M.parse_remote_listing(f"POINTER-ABSENT\n---\n10 {a}\nFIND-OK\n") == (None, {a: 10})
    assert M.parse_remote_listing("POINTER \n---\nFIND-OK\n") == (None, {}), "an empty pointer file is no pointer"
    assert M.parse_remote_listing("DIR-ABSENT\n") == (None, {}), "no manifests directory yet: an empty side, not an error"
    for bad, why in ((f"POINTER-ERROR\n---\n10 {a}\nFIND-OK\n", "could not be read"),
                     (f"POINTER {b}\n---\n10 {a}\nFIND-ERROR\n", "could not be listed"),
                     (f"POINTER {b}\n---\n10 {a}\n", "completion marker"),
                     (f"POINTER {b}\n10 {a}\nFIND-OK\n", "separator"),
                     ("POINTER nonsense\n---\nFIND-OK\n", "not a manifest"),
                     (f"POINTER {b}\n---\njunk {a}\nFIND-OK\n", "malformed"),
                     ("", "empty"), ("\n\n", "empty"), ("bash: find: command not found\n", "pointer line"),
                     (f"{b}\n---\n10 {a}\n", "pointer line")):
        with pytest.raises(ValueError) as ex:
            M.parse_remote_listing(bad)
        assert why in str(ex.value), (bad, str(ex.value))
    dest = _ssh_dest()
    cmd = M._remote_listing_cmd(dest)
    assert cmd[:5] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"] and cmd[5] == "login-0" and len(cmd) == 7
    remote = cmd[6]
    assert remote.startswith("m='/shared/u/registry/log/manifests'; if [ ! -d \"$m\" ]; then echo DIR-ABSENT; exit 0; fi; ")
    assert ("if [ -e \"$m/CHECKSUMS.CURRENT\" ]; then p=$(cat \"$m/CHECKSUMS.CURRENT\") && printf 'POINTER %s\\n' \"$p\" || echo POINTER-ERROR; "
            "else echo POINTER-ABSENT; fi; echo ---; ") in remote
    assert remote.endswith("find \"$m\" -maxdepth 1 -type f -name 'CHECKSUMS-*.sha256' -printf '%s %f\\n' && echo FIND-OK || echo FIND-ERROR")
    assert "/dev/null" not in remote and "; true" not in remote, "nothing suppressed, no unconditional success"
    assert "test -f" not in remote and "mkdir" not in remote and ".tmp" not in remote and "mv -f" not in remote
    assert M._remote_delete_cmd(dest)[-1] == "cd /shared/u/registry/log/manifests && xargs -r rm -f --"
    assert M._checked_names([a, b]) == [a, b]
    for bad in ([a, M.CURRENT_NAME], [a + ".tmp"], ["../records/x.jsonl"], ["CHECKSUMS-1.sha256"]):
        with pytest.raises(ValueError):
            M._checked_names(bad)
    # read_remote_listing: ssh's own failure, and a failed listing, both carry the stderr
    rec = _Recorder(lambda argv: (255, "", "ssh: connect to host login-0 port 22: Connection timed out"))
    with pytest.raises(subprocess.SubprocessError) as ex2:
        M.read_remote_listing(dest, rec)
    assert "exited 255" in str(ex2.value) and "timed out" in str(ex2.value)
    rec2 = _Recorder(lambda argv: (0, f"POINTER-ERROR\n---\nFIND-OK\n", "cat: CHECKSUMS.CURRENT: Permission denied"))
    with pytest.raises(ValueError) as ex3:
        M.read_remote_listing(dest, rec2)
    assert "could not be read" in str(ex3.value) and "Permission denied" in str(ex3.value)
    # the state file's rows (a registry before the pointer record): `dest_current` first, else the switched manifest, else the base
    # read, else a verify row's `remote_current`; never a non-manifest name
    root = str(tmp_path / "reg")
    os.makedirs(root)
    assert M.last_known_currents(root) == {} and M.read_pointers(root) == {}
    M.write_state(root, [{"destination": "x", "dest_current": a, "manifest": b, "switched": True},
                         {"destination": "y", "manifest": b, "switched": True, "since": a},
                         {"destination": "z", "manifest": b, "switched": False, "since": a},
                         {"destination": "w", "manifest": b, "switched": False, "since": None, "copy_since": None},
                         {"destination": "v", "dest_current": "nonsense"}, {"kind": "ssh"},
                         {"destination": "u", "kind": "ssh", "latest_manifest": b, "remote_current": a, "lag_files": 0, "error": None}])
    # a row that names nothing (w: an outage before the probe; v: not a manifest) ESTABLISHES nothing: absent from the dict, so the
    # retention holds the grading host's deletions for it -- a None value is a pointer observed absent (below)
    assert M.last_known_currents(root) == {"x": a, "y": b, "z": a, "u": a}
    # the pointer RECORD wins over the rows, an unknown observation never erases it, a non-manifest name is never recorded
    assert M.record_pointer(root, "w", b, "test", now=synth.t(1)) and M.last_known_currents(root)["w"] == b
    assert not M.record_pointer(root, "w", None, "sync") and not M.record_pointer(root, "w", "nonsense", "sync")
    assert M.read_pointers(root) == {"w": {"current": b, "observed_utc": synth.t(1), "source": "test"}}
    # a DEFINITE absence (the fail-closed listing's POINTER-ABSENT / DIR-ABSENT, a local destination without a pointer file) is recorded
    # as null for a destination without an entry -- established, nothing to keep -- and never over a recorded name
    assert not M.record_pointer(root, "v", None, "listing") and "v" not in M.read_pointers(root), "a plain None is unknown"
    assert M.record_pointer(root, "v", None, "listing", absent=True, now=synth.t(2))
    assert M.read_pointers(root)["v"] == {"current": None, "observed_utc": synth.t(2), "source": "listing"}
    assert M.last_known_currents(root) == {"v": None, "w": b, "x": a, "y": b, "z": a, "u": a}
    assert not M.record_pointer(root, "w", None, "listing", absent=True) and M.read_pointers(root)["w"]["current"] == b
    assert M.record_pointer(root, "v", a, "sync") and M.read_pointers(root)["v"]["current"] == a, "a name replaces an absence"
    assert M.read_pointers(root)["v"]["source"] == "sync" and os.path.exists(M.pointers_lock_path(root))
    assert M.record_pointer(root, "x", b, "verify") and M.last_known_currents(root)["x"] == b, "a newer observation replaces"
    assert json.load(open(M.pointers_path(root)))["destinations"]["x"]["source"] == "verify"
    assert M.manifest_references(root, {"x": a, "y": b, "w": None, "v": "nonsense"}) == {a: ["x" + REASON], b: ["y" + REASON]}
    # a grading host root with its own pointer (the local-destination layout) names it too
    M.switch_current(root, b)
    assert M.manifest_references(root, {"x": a}) == {a: ["x" + REASON], b: ["the grading host CHECKSUMS.CURRENT"]}
    assert M.read_state(root)["utc"] and "manifest_retention" not in M.read_state(root)
    M.write_state(root, [], extra={"manifest_retention": {"keep": 3}})
    assert M.read_state(root)["manifest_retention"] == {"keep": 3}
    assert M._fmt_bytes(76870873612) == "76.9 GB" and M._fmt_bytes(81313223) == "81.3 MB" and M._fmt_bytes(12) == "12 B"


def test_the_state_protects_a_destinations_pointer_a_filtered_run_does_not_reach(tmp_path, monkeypatch, capsys):
    root = _setup(tmp_path)
    a_path, b_path = str(tmp_path / "ma"), str(tmp_path / "mb")
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump({"mirrors": [{"name": "a", "kind": "local", "path": a_path}, {"name": "b", "kind": "local", "path": b_path}]}, fh)
    rows = M.sync_all(root, _seq(root))                                   # both pointers at the first manifest; the state records them
    first = rows[0]["manifest"]
    assert [r["dest_current"] for r in rows] == [first, first] and M.last_known_currents(root) == {"a": first, "b": first}
    later = [M.write_manifest(root, _seq(root) + 10 * i) for i in range(1, 5)]       # four more on the grading host only
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["mirrors", "prune-manifests", "--keep", "1", "--dest", "a"]) == 0
    out = capsys.readouterr().out
    # b was never listed, yet its pointer (the state's last pass) keeps `first` on the grading host; a's own listing says the same
    assert _grading_host_names(root) == [first, later[-1]]
    assert f"  {first}: a{REASON}; b{REASON}" in out and "[retention] a " in out and "[retention] b " not in out
    assert _dest_names(a_path) == [first] and _dest_names(b_path) == [first]


# ------------------------------------------------------------------ the pointer record survives outages, filters and verify-only cycles
def test_consecutive_failed_passes_past_the_retention_boundary_keep_the_destinations_base(tmp_path):
    """A sync that fails before its pointer probe reports dest_current None; the state's row then said None, so the pass
    after the next lost the destination's base. The pointer record outlives the failed rows."""
    root = _setup(tmp_path)
    dest = _ssh_dest()
    n = len(M.log_files(root))
    rows = M.sync_all(root, 100, [dest], runner=_Recorder(lambda argv: _full_json(n) if _is_verify(argv) else ""), now=synth.t(1))
    base = rows[0]["manifest"]
    assert rows[0]["switched"] and rows[0]["dest_current"] == base
    assert M.last_known_currents(root) == {"cluster_a": base} and M.read_pointers(root)["cluster_a"]["current"] == base

    def outage(argv):
        raise subprocess.SubprocessError("ssh: connect to host login-0 port 22: Connection timed out")

    # three failed passes, each writing a manifest newer than keep=1 allows: the base stays on the grading host throughout
    for seq in (101, 102, 103):
        rows = M.sync_all(root, seq, [dest], runner=_Recorder(outage), now=synth.t(seq), keep=1)
        assert rows[0]["error"] and rows[0]["dest_current"] is None and not rows[0]["switched"]
        assert base in _grading_host_names(root), f"pass {seq}: the destination still names {base}"
        assert M.last_known_currents(root)["cluster_a"] == base
    assert _grading_host_names(root) == sorted([base, rows[0]["manifest"]])
    st = M.read_state(root)
    assert st["destinations"][0]["dest_current"] is None and st["manifest_retention"]["references"] == {base: ["cluster_a" + REASON]}
    assert M.read_pointers(root)["cluster_a"] == {"current": base, "observed_utc": synth.t(1), "source": "sync"}


def test_repeated_filtered_syncs_keep_the_omitted_destinations_pointer(tmp_path):
    """`mirrors sync --dest a` rewrote the state with a's row alone; the second filtered pass no longer knew b's pointer."""
    root = _setup(tmp_path)
    a, b = _ssh_dest("a", "ha"), _ssh_dest("b", "hb", shared="/w/u")
    n = len(M.log_files(root))
    clean = lambda argv: _full_json(n) if _is_verify(argv) else ""
    rows = M.sync_all(root, 100, [a, b], runner=_Recorder(clean), now=synth.t(1))
    first = rows[0]["manifest"]
    assert [r["dest_current"] for r in rows] == [first, first] and M.last_known_currents(root) == {"a": first, "b": first}
    for seq in (101, 102, 103):
        rows = M.sync_all(root, seq, [a], runner=_Recorder(clean), now=synth.t(seq), keep=1)
        assert rows[0]["switched"] and rows[0]["dest_current"] == rows[0]["manifest"]
        assert first in _grading_host_names(root), f"pass {seq}: b still names {first}"
        assert M.last_known_currents(root) == {"a": rows[0]["manifest"], "b": first}
        assert [r["destination"] for r in M.read_state(root)["destinations"]] == ["a"]
    assert _grading_host_names(root) == sorted([first, rows[0]["manifest"]])
    assert M.read_state(root)["manifest_retention"]["references"][first] == ["b" + REASON]


def test_a_pointer_read_by_verify_survives_an_outage(tmp_path):
    """`mirrors verify` wrote rows with `remote_current`, which the retention never read; a verify-then-outage lost it."""
    root = _setup(tmp_path)
    dest = _ssh_dest()
    x = M.write_manifest(root, 50)
    n = len(M.read_manifest(os.path.join(root, M.MANIFEST_DIR, x)))
    rec = _Recorder(lambda argv: (x + "\n") if _is_cat(argv) else json.dumps({"ok": n, "missing": [], "corrupt": []}) if _is_verify(argv) else "")
    row = M.verify_destination(root, dest, runner=rec)
    assert row["remote_current"] == x and row["error"] is None and row["lag_files"] == 0
    M.write_state(root, [row])                                   # what `registry mirrors verify` writes
    assert M.read_pointers(root)["cluster_a"]["source"] == "verify" and M.last_known_currents(root) == {"cluster_a": x}
    # the row alone is read as the fallback too (a registry whose record is gone)
    os.rename(M.pointers_path(root), M.pointers_path(root) + ".aside")
    assert M.last_known_currents(root) == {"cluster_a": x}
    os.rename(M.pointers_path(root) + ".aside", M.pointers_path(root))

    def outage(argv):
        raise subprocess.SubprocessError("timed out")

    for seq in (60, 70):
        rows = M.sync_all(root, seq, [dest], runner=_Recorder(outage), now=synth.t(seq), keep=1)
        assert rows[0]["error"] and x in _grading_host_names(root), f"pass {seq}"
    assert _grading_host_names(root) == sorted([x, rows[0]["manifest"]])
    # a local destination's verify records its pointer too
    local = M.Destination("m", "local", str(tmp_path / "mirror"))
    first = M.sync(root, local, 80)
    assert M.verify_destination(root, local)["remote_current"] == first.manifest and M.read_pointers(root)["m"]["current"] == first.manifest


# ------------------------------------------------------------------------------- the listing shell itself, run for real
@pytest.mark.parametrize("refusal", [
    "stub",
    pytest.param("kernel", marks=pytest.mark.skipif(
        not permission_denial_enforced(),
        reason="the kernel does not refuse this process a read of a file it holds no read bit on (root or CAP_DAC_OVERRIDE), and "
               "that refusal of `cat` on the destination's pointer is what this parametrization asserts")),
])
def test_the_listing_shell_fails_closed_and_a_side_it_cannot_establish_is_refused(tmp_path, refusal):
    """The shell suppressed stderr and ended `; true`, so a failed `find` read as an empty listing and a failed `cat` of an
    EXISTING pointer as an absent one -- with no cached pointer the plan scheduled the destination's own CURRENT for deletion with
    error None. The shell is run here for real through `sh -c`. The unreadable pointer comes in two forms (``refusal``): `cat`
    shadowed by a function that refuses, which runs under any uid, and the pointer file itself `chmod 0` under the plain runner where
    the kernel enforces modes, so the real refusal path is exercised and not only its shape."""
    m = tmp_path / "log" / "manifests"
    m.mkdir(parents=True)
    a, b, c = (f"CHECKSUMS-{i:012d}.sha256" for i in (1, 2, 3))
    (m / a).write_text("x" * 10)
    (m / b).write_text("x" * 20)
    (m / "CHECKSUMS-000000000009.sha256.tmp").write_text("mid-write")
    dest = M.Destination("cluster_b", "ssh", str(tmp_path / "log"), "cluster_b")
    assert M.read_remote_listing(dest, _sh) == (None, {a: 10, b: 20}), "no pointer file: absent, the manifests listed"
    (m / M.CURRENT_NAME).write_text(b + "\n")
    assert M.read_remote_listing(dest, _sh) == (b, {a: 10, b: 20})
    (m / M.CURRENT_NAME).write_text(b)
    assert M.read_remote_listing(dest, _sh) == (b, {a: 10, b: 20}), "the remote switch writes the pointer without a newline"
    (m / M.CURRENT_NAME).write_text("")
    assert M.read_remote_listing(dest, _sh) == (None, {a: 10, b: 20}), "an empty pointer file is no pointer (records.read_pointer)"
    assert M.read_remote_listing(M.Destination("x", "ssh", str(tmp_path / "nowhere"), "h"), _sh) == (None, {}), "no manifests/ yet"
    # a pointer that EXISTS but cannot be read, with a working find: refused, the shell's stderr in the error. The stub form is a
    # shell-function substitute for `cat` (what a pointer file the destination's user holds no read bit on looks like: the `[ -e ]`
    # probe passes, the read is refused on stderr), which a `chmod 0` cannot be where the kernel does not enforce modes (root,
    # CAP_DAC_OVERRIDE); the kernel form IS that `chmod 0`, read through the plain `_sh`, so the refusal is the kernel's own
    def _sh_badcat(argv, timeout_s, input=None, check=True):
        return subprocess.run(["sh", "-c", 'cat() { echo "cat: $1: Permission denied" >&2; return 1; }; ' + argv[-1]],
                              capture_output=True, text=True, input=input)

    kernel = refusal == "kernel"
    unreadable = _sh if kernel else _sh_badcat               # the runner under which the pointer cannot be read

    def _deny():                                             # kernel form: drop every mode bit; stub form: the runner already refuses
        if kernel:
            (m / M.CURRENT_NAME).chmod(0)

    def _restore():                                          # kernel form: 0o644 back before the pointer is next written or read for real
        if kernel:
            (m / M.CURRENT_NAME).chmod(0o644)

    (m / M.CURRENT_NAME).write_text(b)
    _deny()
    with pytest.raises(ValueError) as ex:
        M.read_remote_listing(dest, unreadable)
    assert "exists but could not be read" in str(ex.value) and "Permission denied" in str(ex.value)
    _restore()

    # a failing find (a shell-function substitute: what a find without -printf, or a vanished directory, looks like): refused
    def _sh_badfind(argv, timeout_s, input=None, check=True):
        return subprocess.run(["sh", "-c", "find() { echo 'find: boom' >&2; return 1; }; " + argv[-1]], capture_output=True, text=True)

    with pytest.raises(ValueError) as ex2:
        M.read_remote_listing(dest, _sh_badfind)
    assert "could not be listed" in str(ex2.value) and "find: boom" in str(ex2.value)

    # through prune_manifests, with NO pointer on record: the side is refused and so is the grading host (its error names the destination)
    root = str(tmp_path / "reg")
    os.makedirs(os.path.join(root, M.MANIFEST_DIR))
    for n in (a, b, c):
        with open(os.path.join(root, M.MANIFEST_DIR, n), "w") as fh:
            fh.write("x")
    (m / M.CURRENT_NAME).write_text(b)
    _deny()
    summary = M.prune_manifests(root, [dest], keep=1, runner=unreadable)
    dev, side = summary["sides"]
    assert side["error"].startswith("listing:") and "could not be read" in side["error"] and (side["deleted"], side["removed"], side["manifests"]) == (0, 0, 0)
    assert dev["error"].startswith("refused:") and "cluster_b" in dev["error"] and dev["removed"] == 0 and dev["deleted"] == 2
    assert _grading_host_names(root) == [a, b, c], "nothing deleted anywhere"
    assert M.read_pointers(root) == {}
    text = M.format_retention(summary)
    assert "ERROR refused:" in text and "ERROR listing:" in text
    # with the pointer on record (a past sync, verify or listing), the grading host prunes -- keeping b -- while the unreadable side stays refused
    M.record_pointer(root, "cluster_b", b, "test")
    summary2 = M.prune_manifests(root, [dest], keep=1, runner=unreadable)
    assert summary2["sides"][0]["error"] is None and summary2["sides"][0]["removed"] == 1 and _grading_host_names(root) == [b, c]
    assert summary2["sides"][1]["error"].startswith("listing:") and summary2["sides"][1]["removed"] == 0
    _restore()
    # readable again (the real `cat`, the mode back): the listing establishes the pointer, records it, and the side is pruned by the
    # rule (its own pointer b kept)
    summary3 = M.prune_manifests(root, [dest], keep=1, runner=_sh)
    assert summary3["sides"][1]["error"] is None and summary3["sides"][1]["current"] == b and summary3["sides"][1]["removed"] == 1
    assert sorted(p.name for p in m.iterdir() if p.name != M.CURRENT_NAME and not p.name.endswith(".tmp")) == [b]
    assert M.read_pointers(root)["cluster_b"]["source"] == "listing"
    # a dry run records nothing
    os.remove(M.pointers_path(root))
    M.prune_manifests(root, [dest], keep=1, runner=_sh, dry_run=True)
    assert not os.path.exists(M.pointers_path(root))


# ------------------------------------------------------- every destination is accounted for before the grading host loses a manifest
def test_an_unreached_destination_whose_pointer_is_unknown_from_every_source_holds_the_grading_host_deletions(tmp_path, capsys):
    """Round-1 review (HIGH): on rollout `state/mirror_pointers.json` does not exist yet; when the state's latest row for a
    destination is an outage row (dest_current None) and the destination is still unreachable before its probe, the pass left it
    out of `reached`, prune_manifests checked unknown pointers only for the destinations it listed, and the grading host deleted that
    destination's base once it fell outside the newest K -- its next sync a whole-tree rsync over ~500k files. Every destination
    of the pass is accounted for now: the unreached one with no pointer from any source refuses the grading host's deletions (nothing
    deleted there), the reached one is still pruned by the rule, and a pass that reaches it again releases the hold."""
    root = _setup(tmp_path)
    a, b = _ssh_dest("a", "ha"), _ssh_dest("b", "hb", shared="/w/u")
    n = len(M.log_files(root))
    base = M.write_manifest(root, _seq(root))                                      # a's acked manifest, its pointer: the oldest
    extra = [M.write_manifest(root, _seq(root) + 10 * i) for i in range(1, 4)]     # three later passes' manifests
    # rollout: no pointer record; the state's latest row for a is an outage row, b's names its pointer
    M.write_state(root, [{"destination": "a", "kind": "ssh", "manifest": extra[-1], "switched": False, "dest_current": None,
                          "error": "ssh: connect to host ha port 22: Connection timed out"},
                         {"destination": "b", "kind": "ssh", "manifest": extra[-1], "switched": True, "dest_current": extra[-1]}])
    assert not os.path.exists(M.pointers_path(root)) and M.last_known_currents(root).get("b") == extra[-1]
    assert M.last_known_currents(root).get("a") is None, "nothing names a's pointer"
    copies = {"ha": {base: 1}, "hb": {x: 1 for x in extra}}
    a_down = [True]

    def respond(argv):
        host = argv[5] if argv[0] == "ssh" else argv[-1].split(":", 1)[0]
        if host == "ha" and a_down[0]:
            raise subprocess.SubprocessError("ssh: connect to host ha port 22: Connection timed out")
        if _is_probe(argv):
            return base if host == "ha" else max(copies["hb"])
        if _is_verify(argv):
            return _clean_json(n, base if host == "ha" else max(copies["hb"]))
        if _is_manifest_copy(argv):
            copies[host][os.path.basename(argv[-2])] = 1
            return ""
        if _is_listing(argv):
            return _listing_text(max(copies[host]), copies[host])            # the pointer after the switch: the newest copy
        return ""

    rec = _Recorder(respond)
    rows = M.sync_all(root, _seq(root) + 100, [a, b], runner=rec, now=synth.t(5), keep=1)
    new = rows[0]["manifest"]
    assert rows[0]["error"] and rows[0]["dest_current"] is None and rows[1]["switched"] and rows[1]["dest_current"] == new
    # the grading host: NOTHING deleted -- a's pointer is unknown from every source (no record, an outage row) and a was not reached
    assert _grading_host_names(root) == sorted([base] + extra + [new])
    ret = M.read_state(root)["manifest_retention"]
    dev, side_b = ret["sides"]
    assert [s["side"] for s in ret["sides"]] == ["grading_host", "b"], "the unreached destination is not listed"
    assert dev["error"].startswith("refused: the CHECKSUMS.CURRENT of a (this pass did not reach it) is unknown from every source")
    assert dev["removed"] == 0 and dev["deleted"] == 4 and ret["references"] == {new: ["b" + REASON]}, "the base among the four"
    # b, reached and listed, is still pruned by the rule: its own pointer (the pass's manifest) kept, the three older copies deleted
    assert side_b["error"] is None and side_b["current"] == new and side_b["removed"] == 3 and side_b["deleted_names"] == extra
    assert [c[2] for c in rec.calls if _is_delete(c[0])] == ["\n".join(extra) + "\n"]
    assert not any(_is_listing(c[0]) and c[0][5] == "ha" for c in rec.calls)
    out = capsys.readouterr().out
    assert "[retention] grading_host " in out and "ERROR refused: the CHECKSUMS.CURRENT of a (this pass did not reach it)" in out
    recs = M.read_pointers(root)
    assert set(recs) == {"b"} and recs["b"]["current"] == new and recs["b"]["source"] == "listing", "a's outage recorded nothing; b's sync, then its listing, did"
    # the same pass again (a still down): still refused, still nothing deleted -- the hold is not a one-pass grace
    rows2 = M.sync_all(root, _seq(root) + 200, [a, b], runner=_Recorder(respond), now=synth.t(6), keep=1)
    new2 = rows2[0]["manifest"]
    assert M.read_state(root)["manifest_retention"]["sides"][0]["error"].startswith("refused:")
    assert _grading_host_names(root) == sorted([base] + extra + [new, new2])
    # a reachable again: its probe reads the base (still on the grading host, so the copy is the delta), the clean verify switches it,
    # the record now names it, and the grading host is pruned by the rule -- no refusal
    a_down[0] = False
    rec3 = _Recorder(respond)
    rows3 = M.sync_all(root, _seq(root) + 300, [a, b], runner=rec3, now=synth.t(7), keep=1)
    new3 = rows3[0]["manifest"]
    assert rows3[0]["switched"] and rows3[0]["copy_mode"] == "incremental" and rows3[0]["copy_since"] == base
    ret3 = M.read_state(root)["manifest_retention"]
    assert [s["side"] for s in ret3["sides"]] == ["grading_host", "a", "b"] and all(s["error"] is None for s in ret3["sides"])
    assert ret3["references"] == {new3: ["a" + REASON, "b" + REASON]} and _grading_host_names(root) == [new3]
    assert ret3["sides"][1]["removed"] == 1 and ret3["sides"][1]["deleted_names"] == [base]
    assert M.read_pointers(root)["a"]["current"] == new3 and M.read_pointers(root)["a"]["source"] == "listing", "recorded by the sync, re-observed by its listing"


def test_a_filtered_run_accounts_for_the_destinations_it_omits_and_an_observed_absence_is_established(tmp_path, monkeypatch, capsys):
    """Round-1 review (HIGH), the filtered face: `prune-manifests --dest a` and `sync --dest a` must account for every configured
    destination, the omitted ones included -- one nothing ever observed holds the grading host; a listing that finds no pointer is a
    definite absence, recorded as null, which establishes the destination (nothing to keep for it) and never releases a recorded
    name."""
    root = _setup(tmp_path)
    a_path, b_path, c_path = (str(tmp_path / x) for x in ("ma", "mb", "mc"))
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump({"mirrors": [{"name": x, "kind": "local", "path": px} for x, px in (("a", a_path), ("b", b_path), ("c", c_path))]}, fh)
    a, b, c = M.destinations(root)
    rows = M.sync_all(root, _seq(root), [a, b])                       # a and b synced once; c is configured and never observed
    first = rows[0]["manifest"]
    assert M.last_known_currents(root) == {"a": first, "b": first} and "c" not in M.read_pointers(root)
    later = [M.write_manifest(root, _seq(root) + 10 * i) for i in range(1, 5)]
    monkeypatch.setenv("REGISTRY_ROOT", root)
    # filtered to a: c, configured and unknown from every source, holds the grading host (exit 1: a side errored); a's own side is pruned
    assert cli.main(["mirrors", "prune-manifests", "--keep", "1", "--dest", "a"]) == 1
    out = capsys.readouterr().out
    assert "ERROR refused: the CHECKSUMS.CURRENT of c (this run did not list it) is unknown from every source" in out
    assert _grading_host_names(root) == sorted([first] + later), "nothing deleted on the grading host"
    assert "[retention] a " in out and "[retention] c " not in out and "[retention] b " not in out
    assert "c" not in M.read_pointers(root), "a destination the run did not list is not recorded"
    # a dry run reports the same refusal, writes nothing, exits 0
    assert cli.main(["mirrors", "prune-manifests", "--keep", "1", "--dest", "a", "--dry-run"]) == 0
    assert "ERROR refused:" in capsys.readouterr().out and "c" not in M.read_pointers(root)
    # the pass, filtered to a (the CLI's `mirrors sync --dest a`): the same hold, logged, never the pass's failure
    rows = M.sync_all(root, _seq(root) + 100, [a], keep=1)
    assert rows[0]["switched"] and M.read_state(root)["manifest_retention"]["sides"][0]["error"].startswith("refused:")
    assert "ERROR refused: the CHECKSUMS.CURRENT of c (this run did not list it)" in capsys.readouterr().out
    assert first in _grading_host_names(root) and later[0] in _grading_host_names(root)
    # listing c establishes it: no manifests directory, no pointer file -- a definite absence, recorded as null -- and the grading host is
    # pruned by the rule (first stays: b's pointer, on record; a's is the pass's manifest)
    assert cli.main(["mirrors", "prune-manifests", "--keep", "1", "--dest", "c"]) == 0
    out = capsys.readouterr().out
    assert "refused" not in out and M.read_pointers(root)["c"]["current"] is None and M.read_pointers(root)["c"]["source"] == "listing"
    assert M.last_known_currents(root) == {"a": rows[0]["manifest"], "b": first, "c": None}
    assert _grading_host_names(root) == sorted([first, rows[0]["manifest"]])
    assert f"  {first}: b{REASON}" in out and f"  {rows[0]['manifest']}: a{REASON}" in out and ("c" + REASON) not in out
    # established-absent, c no longer holds a filtered run; and a listing of b WITHOUT its pointer file (the file gone, a mount away)
    # does not release b's recorded name: first stays on the grading host
    os.remove(os.path.join(b_path, M.MANIFEST_DIR, M.CURRENT_NAME))
    more = M.write_manifest(root, _seq(root) + 200)
    assert cli.main(["mirrors", "prune-manifests", "--keep", "1", "--dest", "b"]) == 0
    out = capsys.readouterr().out
    assert "refused" not in out and M.read_pointers(root)["b"]["current"] == first
    assert _grading_host_names(root) == sorted([first, rows[0]["manifest"], more]), "b's recorded first, a's pointer, the newest"
    assert f"  {first}: b{REASON}" in out and "[retention] b " in out


# ------------------------------------------------------------------------------ the pointer record's merge is serialized
def test_the_pointer_record_is_merged_under_a_lock_so_overlapping_observers_keep_both_entries(tmp_path, monkeypatch):
    """Round-1 review (MEDIUM): a scheduled pass and an on-demand prune-manifests can overlap; both read the whole pointer record,
    updated different destinations and replaced it, and the later replacement dropped the other's entry -- at worst an unreachable
    destination's base, which the grading host then deleted. Every merge now holds state/mirror_pointers.lock (an flock, so it serializes
    across processes): a second observer's READ waits for the first's write, and both entries survive."""
    import sys
    import threading
    import time as _time
    root = str(tmp_path / "reg")
    os.makedirs(root)
    one, two = M.write_manifest(root, 1), M.write_manifest(root, 2)
    lock_path = M.pointers_lock_path(root)
    os.makedirs(os.path.dirname(lock_path))
    # (1) another PROCESS holds the lock: record_pointer waits for it and lands once it is released; the lock itself times out
    holder = subprocess.Popen([sys.executable, "-c", "import fcntl, sys; fh = open(sys.argv[1], 'a+'); fcntl.flock(fh.fileno(), fcntl.LOCK_EX); "
                               "print('held', flush=True); sys.stdin.readline()", lock_path],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(TimeoutError) as ex:
            with M.pointers_lock(root, timeout_s=0.2):
                pass
        assert "held by another observer" in str(ex.value) and isinstance(ex.value, OSError)
        landed = []
        t = threading.Thread(target=lambda: landed.append(M.record_pointer(root, "a", one, "sync", now=synth.t(1))))
        t.start()
        t.join(0.4)
        assert t.is_alive() and not os.path.exists(M.pointers_path(root)), "blocked behind the holder, nothing written"
        holder.stdin.write("\n")
        holder.stdin.flush()
        holder.wait(timeout=10)
        t.join(10)
    finally:
        if holder.poll() is None:
            holder.kill()
    assert landed == [True] and M.read_pointers(root)["a"]["current"] == one
    # (2) two observers whose read-modify-write would interleave: the second's read waits until the first has written
    reads, go = [], threading.Event()
    real_read = M.read_pointers

    def slow_read(r):
        recs = real_read(r)
        reads.append(threading.current_thread().name)
        go.wait(5)
        return recs

    monkeypatch.setattr(M, "read_pointers", slow_read)
    t1 = threading.Thread(target=lambda: M.record_pointer(root, "b", two, "listing", now=synth.t(2)), name="t1")
    t2 = threading.Thread(target=lambda: M.record_pointer(root, "c", one, "verify", now=synth.t(3)), name="t2")
    t1.start()
    deadline = _time.monotonic() + 5
    while reads != ["t1"] and _time.monotonic() < deadline:
        _time.sleep(0.01)
    assert reads == ["t1"]
    t2.start()
    t2.join(0.4)
    assert reads == ["t1"] and t2.is_alive(), "the second observer has not read: it waits for the first's write"
    go.set()
    t1.join(10)
    t2.join(10)
    monkeypatch.undo()
    assert reads == ["t1", "t2"] and not t1.is_alive() and not t2.is_alive()
    assert {k: v["current"] for k, v in M.read_pointers(root).items()} == {"a": one, "b": two, "c": one}, "both entries survive"
    # (3) a filtered state write (a's row alone) and a retention plan with b unreachable: b's record keeps its base, no refusal
    later = [M.write_manifest(root, s) for s in (3, 4, 5)]
    M.write_state(root, [{"destination": "a", "kind": "ssh", "manifest": later[-1], "switched": True, "dest_current": later[-1]}])
    summary = M.prune_manifests(root, [], keep=1, configured=["a", "b", "c"])
    assert summary["sides"][0]["error"] is None and summary["sides"][0]["removed"] == 2
    assert summary["references"] == {one: ["a" + REASON, "c" + REASON], two: ["b" + REASON]}
    assert _grading_host_names(root) == [one, two, later[-1]]


def test_a_pointer_record_failure_is_logged_and_never_the_passes_or_the_verifys(tmp_path, monkeypatch, capsys):
    """The record is housekeeping: a merge that cannot take its lock (TimeoutError, an OSError) is logged by the pass, carried by
    the verify's row, and costs neither its state, acks or deletions."""
    root = _setup(tmp_path)
    local = M.Destination("m", "local", str(tmp_path / "mirror"))

    def boom(*args, **kwargs):
        raise TimeoutError("the pointer record's lock is held by another observer")

    monkeypatch.setattr(M, "record_pointer", boom)
    rows = M.sync_all(root, _seq(root), [local])
    assert rows[0]["switched"] and rows[0]["ack_record"] and M.read_state(root)["destinations"][0]["switched"]
    assert "[pointers] ERROR recording m's pointer: the pointer record's lock is held" in capsys.readouterr().out
    row = M.verify_destination(root, local)
    assert row["error"] is None and row["lag_files"] == 0 and row["pointer_record_error"].startswith("the pointer record's lock")
    later = [M.write_manifest(root, _seq(root) + 10 * i) for i in range(1, 4)]
    summary = M.prune_manifests(root, [local], keep=1)
    assert summary["sides"][0]["error"] is None and summary["sides"][0]["removed"] == 2, "the listing established m; the record failed"
    assert "[pointers] ERROR recording m's pointer" in capsys.readouterr().out and not os.path.exists(M.pointers_path(root))
    monkeypatch.undo()
    assert "pointer_record_error" not in M.verify_destination(root, local) and M.read_pointers(root)["m"]["source"] == "verify"


# ------------------------------------------------- an observation whose merge failed is held, never pruned against its predecessor
def test_a_pointer_observation_whose_merge_failed_is_held_and_kept_until_a_merge_records_the_destination(tmp_path, monkeypatch, capsys):
    """Round-2 review (MEDIUM): the record names A; a clean sync switches the destination to B but record_pointer's lock times
    out -- logged, the sync and its state write go on; the destination then drops out before its probe and the next failed pass
    replaces the state row that named B; last_known_currents preferred the surviving record A, so the destination read as
    accounted for, and once B fell outside the newest K the grading host deleted it -- the destination's actual base -- with no refusal,
    its next sync a whole-tree rsync. A name whose merge failed is HELD in its own file now (state/mirror_pointers.unrecorded/),
    kept on every side as a reference and reported each pass, until a merge records that destination's pointer."""
    import contextlib
    root = _setup(tmp_path)
    dest = _ssh_dest()
    n = len(M.log_files(root))
    remote = {"pointer": None, "names": {}}
    down = [False]

    def respond(argv):
        joined = " ".join(argv)
        if down[0]:
            raise subprocess.SubprocessError("ssh: connect to host login-0 port 22: Connection timed out")
        if _is_probe(argv):
            return remote["pointer"] or ""
        if _is_verify(argv):
            return _clean_json(n, remote["pointer"]) if remote["pointer"] else _full_json(n)
        if _is_manifest_copy(argv):
            remote["names"][os.path.basename(argv[-2])] = 1
            return ""
        if "CHECKSUMS.CURRENT.tmp" in joined and "mv -f" in joined:
            remote["pointer"] = joined.split("printf %s '", 1)[1].split("'", 1)[0]
            return ""
        if _is_listing(argv):
            return _listing_text(remote["pointer"], remote["names"])
        return ""

    rows = M.sync_all(root, 100, [dest], runner=_Recorder(respond), now=synth.t(1))
    a = rows[0]["manifest"]
    assert rows[0]["switched"] and remote["pointer"] == a and M.read_pointers(root)["cluster_a"]["current"] == a
    assert M.unrecorded_pointers(root) == {} and not os.path.isdir(M.unrecorded_dir(root))
    capsys.readouterr()
    # the pass that switches the destination to B cannot take the record's lock: the sync, its ack and its state row stand, the
    # record still names A, and the observation is HELD -- the ERROR line names the hold, the retention reports and keeps it
    real_lock = M.pointers_lock

    @contextlib.contextmanager
    def stuck_lock(root_, timeout_s=M.POINTERS_LOCK_TIMEOUT_S):
        raise TimeoutError(f"the pointer record's lock {M.pointers_lock_path(root_)} was held by another observer for over 60 s; "
                           f"this observation was not recorded")
        yield  # pragma: no cover

    monkeypatch.setattr(M, "pointers_lock", stuck_lock)
    rows = M.sync_all(root, 101, [dest], runner=_Recorder(respond), now=synth.t(2), keep=1)
    b = rows[0]["manifest"]
    assert rows[0]["switched"] and rows[0]["dest_current"] == b and remote["pointer"] == b and os.path.exists(rows[0]["ack_record"])
    out = capsys.readouterr().out
    hold = M.unrecorded_path(root, "cluster_a", b, synth.t(2))
    assert out.count("[pointers] ERROR recording cluster_a's pointer: the pointer record's lock") == 2, "the sync's merge and the listing's"
    assert f"the observation (cluster_a: {b}, sync) is held in {hold} and kept by the retention until a merge records cluster_a's pointer" in out
    assert f"the observation (cluster_a: {b}, listing) is held in {hold}" in out
    assert M.read_pointers(root)["cluster_a"] == {"current": a, "observed_utc": synth.t(1), "source": "sync"}, "the merge never landed"
    assert M.read_state(root)["destinations"][0]["dest_current"] == b
    held = M.unrecorded_pointers(root)
    assert set(held) == {"cluster_a"} and len(held["cluster_a"]) == 1, "the sync's and the listing's hold of B at t(2) are ONE observation, one file"
    hb = held["cluster_a"][0]
    assert hb["current"] == b and hb["path"] == hold
    assert hb["source"] == "listing" and hb["error"].startswith("the pointer record's lock"), "the later observer's words"
    on_disk = json.load(open(hold))
    assert on_disk["destination"] == "cluster_a" and on_disk["current"] == b and on_disk["held_utc"] and on_disk["observed_utc"] == hb["observed_utc"]
    held_reason = (f"cluster_a's CHECKSUMS.CURRENT as listing observed it at {hb['observed_utc']}, not yet recorded: kept until "
                   f"a merge records cluster_a's pointer")
    ret = M.read_state(root)["manifest_retention"]
    assert ret["unrecorded"] == {"cluster_a": [hb]} and ret["sides"][0]["error"] is None
    assert ret["references"] == {b: ["cluster_a" + REASON, held_reason]} and _grading_host_names(root) == [b], "A is no one's base now"
    assert f"[pointers] HELD cluster_a's CHECKSUMS.CURRENT {b} (observed by listing at {hb['observed_utc']}, not recorded: the pointer record's lock" in out
    assert "kept on every side until a sync, verify or prune listing records cluster_a's pointer" in out
    # the lock works again but the destination is down before its probe: pass after pass past the retention boundary, each failed
    # row replacing the one that named B, the record still A -- B stays on the grading host, known from the hold (no refusal), reported
    monkeypatch.setattr(M, "pointers_lock", real_lock)
    down[0] = True
    for seq in (102, 103, 104):
        rows = M.sync_all(root, seq, [dest], runner=_Recorder(respond), now=synth.t(seq), keep=1)
        assert rows[0]["error"] and rows[0]["dest_current"] is None and not rows[0]["switched"]
        assert b in _grading_host_names(root), f"pass {seq}: the destination's actual base {b} stays"
        st = M.read_state(root)
        assert st["destinations"][0]["dest_current"] is None and st["destinations"][0]["manifest"] == rows[0]["manifest"]
        ret = st["manifest_retention"]
        # pass 102 adds its own manifest to a grading host holding B alone: nothing to delete yet; 103 and 104 each delete the one before
        assert ret["sides"][0]["error"] is None and ret["sides"][0]["removed"] == (0 if seq == 102 else 1), f"pass {seq}: by the rule, not refused"
        # the record's A (gone from the grading host: nothing to keep for it) and the held B are BOTH references
        assert [h["current"] for h in ret["unrecorded"]["cluster_a"]] == [b] and ret["references"] == {a: ["cluster_a" + REASON], b: [held_reason]}
        out = capsys.readouterr().out
        assert f"[pointers] HELD cluster_a's CHECKSUMS.CURRENT {b} (" in out and "[pointers] ERROR" not in out
    assert _grading_host_names(root) == sorted([b, rows[0]["manifest"]]) and M.read_pointers(root)["cluster_a"]["current"] == a
    # a recorded ABSENCE releases nothing (the record keeps A, the hold keeps B)
    assert not M.record_pointer(root, "cluster_a", None, "listing", absent=True) and os.path.exists(hold)
    # reachable again: its probe reads B, which the grading host still holds, so the copy is the delta since B; the clean verify switches to
    # C and the merge lands, which releases the hold -- nothing held, nothing reported, the grading host pruned to the newest
    down[0] = False
    rows = M.sync_all(root, 105, [dest], runner=_Recorder(respond), now=synth.t(105), keep=1)
    c = rows[0]["manifest"]
    assert rows[0]["switched"] and rows[0]["copy_mode"] == "incremental" and rows[0]["copy_since"] == b and rows[0]["dest_current"] == c
    assert rows[0]["note"] is None, "no whole-tree rsync"
    assert M.read_pointers(root)["cluster_a"]["current"] == c and not os.path.exists(hold) and M.unrecorded_pointers(root) == {}
    ret = M.read_state(root)["manifest_retention"]
    assert ret["unrecorded"] == {} and ret["references"] == {c: ["cluster_a" + REASON]} and _grading_host_names(root) == [c]
    assert all(s["error"] is None for s in ret["sides"]) and ret["sides"][1]["current"] == c
    out = capsys.readouterr().out
    assert "HELD" not in out and "[pointers] ERROR" not in out and "refused" not in out
    assert len(M.format_retention({"sides": ret["sides"], "unrecorded": {}}, header=False).splitlines()) == 2

    # a destination known from a hold ALONE (no record, no row) is established, not refused; the hold file's name is safe for any
    # destination name while the file carries the real one; another destination's merge releases nothing; a name does
    root2 = str(tmp_path / "bare")
    os.makedirs(root2)
    one, two, three = (M.write_manifest(root2, s) for s in (1, 2, 3))
    odd = "cluster/eu west"
    p = M.hold_unrecorded(root2, odd, one, "sync", now=synth.t(9), error="the pointer record's lock x was held")
    assert p == M.unrecorded_path(root2, odd, one, synth.t(9)) and os.path.dirname(p) == M.unrecorded_dir(root2)
    # the name: the destination percent-encoded (exact, decodable), a digest of the exact name, the stamp in plain characters, a digest
    # of the observation -- the lossy plain form "cluster_eu_west" appears nowhere, so a destination of THAT name shares no segment
    lead = f"cluster%2Feu%20west~{M._dest_digest(odd)}~2026-01-01T00_00_09.000Z~"
    assert os.path.basename(p).startswith(lead) and os.path.basename(p).endswith(".json") and len(os.path.basename(p)) == len(lead) + 8 + 5
    assert M._hold_name_destination(os.path.basename(p)) == odd
    assert os.path.basename(M.unrecorded_path(root2, "plain-name.1", one, synth.t(9))).startswith(f"plain-name.1~{M._dest_digest('plain-name.1')}~2026-01-01T00_00_09.000Z~")
    assert M.unrecorded_pointers(root2) == {odd: [{"current": one, "observed_utc": synth.t(9), "source": "sync",
                                                  "error": "the pointer record's lock x was held", "path": p}]}
    with pytest.raises(ValueError):
        M.hold_unrecorded(root2, odd, "nonsense", "sync")
    stray = os.path.join(M.unrecorded_dir(root2), "torn.json")
    with open(stray, "w") as fh:
        fh.write("{not json")
    with open(os.path.join(M.unrecorded_dir(root2), "x.json.123.tmp"), "w") as fh:
        fh.write(json.dumps({"destination": "x", "current": two}))
    assert set(M.unrecorded_pointers(root2)) == {odd}, "a torn file and a mid-write tmp are not holds"
    # the torn file is not ignored: it REFUSES the grading host's deletions by name (re-check of 360d61f4b) -- and its name says no
    # destination, so no merge ever releases it: an operator's to read and remove; the mid-write tmp is neither a hold nor a problem
    assert M.unreadable_holds(root2) == [{"path": stray, "destination": None, "problem": "not JSON"}]
    summary = M.prune_manifests(root2, [], keep=1, configured=[odd])
    assert summary["sides"][0]["error"] == (
        f"refused: 1 hold file(s) under {M.unrecorded_dir(root2)} cannot be read or validated: torn.json (not JSON; its name does not decode "
        "to one exact destination, so no merge releases it: an operator reads it and removes it by hand); a hold is a destination's pointer whose "
        "merge into the record failed, and one that cannot be read may name the base the grading host would otherwise delete, so nothing is "
        "deleted on the grading host while any stands")
    assert summary["sides"][0]["removed"] == 0 and summary["sides"][0]["deleted_names"] == [two] and _grading_host_names(root2) == [one, two, three]
    assert summary["unreadable"] == [{"path": stray, "destination": None, "problem": "not JSON"}]
    assert (f"[pointers] UNREADABLE hold {stray}: not JSON; its name does not decode to one exact destination, so no merge releases it: "
            "the grading host deletes nothing until an operator reads it and removes it by hand") in M.format_retention(summary)
    os.remove(stray)
    assert M.unreadable_holds(root2) == []
    summary = M.prune_manifests(root2, [], keep=1, configured=[odd])
    odd_reason = f"{odd}'s CHECKSUMS.CURRENT as sync observed it at {synth.t(9)}, not yet recorded: kept until a merge records {odd}'s pointer"
    assert summary["sides"][0]["error"] is None and summary["sides"][0]["removed"] == 1 and _grading_host_names(root2) == [one, three]
    assert summary["unreadable"] == []
    assert summary["references"] == {one: [odd_reason]} and [h["current"] for h in summary["unrecorded"][odd]] == [one]
    text = M.format_retention(summary)
    assert f"  {one}: {odd_reason}" in text and f"[pointers] HELD {odd}'s CHECKSUMS.CURRENT {one} (observed by sync at {synth.t(9)}, not recorded: the pointer record's lock x was held)" in text
    assert text.splitlines().index(f"[pointers] HELD {odd}'s CHECKSUMS.CURRENT {one} (observed by sync at {synth.t(9)}, not recorded: the pointer record's lock x was held); kept on every side until a sync, verify or prune listing records {odd}'s pointer") == 3
    assert M.record_pointer(root2, "other", two, "verify") and set(M.unrecorded_pointers(root2)) == {odd}, "another destination's merge"
    assert M.record_pointer(root2, odd, three, "sync") and M.unrecorded_pointers(root2) == {} and not os.path.exists(p)
    assert M.read_pointers(root2)[odd]["current"] == three and os.path.isdir(M.unrecorded_dir(root2))
    # when the hold itself cannot be written, the error names both failures; an absence whose merge fails is not held at all
    root3 = str(tmp_path / "nohold")
    os.makedirs(os.path.join(root3, "state"))
    with open(M.unrecorded_dir(root3), "w") as fh:
        fh.write("in the way")
    monkeypatch.setattr(M, "pointers_lock", stuck_lock)
    with pytest.raises(OSError) as ex:
        M.record_pointer(root3, "m", one, "sync")
    assert "held by another observer" in str(ex.value) and f"the observation (m: {one}, sync) could NOT be held either:" in str(ex.value)
    with pytest.raises(TimeoutError) as ex2:
        M.record_pointer(root3, "m", None, "listing", absent=True)
    assert "could NOT be held" not in str(ex2.value) and "is held in" not in str(ex2.value)
    assert M.unrecorded_pointers(root3) == {} and not os.path.exists(M.pointers_path(root3))


# ------------------------------------------ a stale observation never replaces a fresher pointer, nor releases a fresher hold
def test_a_listing_that_overlapped_a_switch_cannot_put_the_old_pointer_back_nor_release_a_fresher_hold(tmp_path, capsys):
    """Round-3 review (MEDIUM): an on-demand prune-manifests listed a destination's pointer A while the scheduled pass was finishing;
    the pass switched the destination to B and recorded it; the prune then took the lock and wrote its EARLIER observation A over
    B -- the lock serialized the writes, not the observations behind them -- and the same stale merge would have released a hold
    of B. The destination then dropped out, its failed rows replaced the one naming B, last_known_currents preferred the record's
    A, and once B fell outside the newest K the grading host deleted it, the destination's actual base, with no refusal (its next sync
    the whole-tree rsync). Every observation is stamped when it is MADE now -- a read as it begins, a switch as it completes --
    and a merge older than the recorded observation is refused; a hold is released only by a merge of its name or of one
    observed after it."""
    import time as _time
    root = _setup(tmp_path)
    dest = _ssh_dest()
    n = len(M.log_files(root))
    remote = {"pointer": None, "names": {}}
    down = [False]

    def respond(argv):
        joined = " ".join(argv)
        if down[0]:
            raise subprocess.SubprocessError("ssh: connect to host login-0 port 22: Connection timed out")
        if _is_probe(argv):
            return remote["pointer"] or ""
        if _is_verify(argv):
            return _clean_json(n, remote["pointer"]) if remote["pointer"] else _full_json(n)
        if _is_manifest_copy(argv):
            remote["names"][os.path.basename(argv[-2])] = 1
            return ""
        if "CHECKSUMS.CURRENT.tmp" in joined and "mv -f" in joined:
            remote["pointer"] = joined.split("printf %s '", 1)[1].split("'", 1)[0]
            return ""
        if _is_listing(argv):
            return _listing_text(remote["pointer"], remote["names"])
        return ""

    # pass 100, on the clock (no fixed `now`): the destination's pointer is A, recorded by the sync as it switched
    rows = M.sync_all(root, 100, [dest], runner=_Recorder(respond))
    a = rows[0]["manifest"]
    assert rows[0]["switched"] and remote["pointer"] == a and M.read_pointers(root)["cluster_a"]["current"] == a
    rec_a = M.read_pointers(root)["cluster_a"]
    capsys.readouterr()
    # an on-demand prune lists the destination -- its listing reads A -- while the scheduled pass 101 finishes: the pass switches the
    # destination to B and records it before the prune merges what it read
    passes = {}

    def overlap(argv):
        if _is_listing(argv):
            text = _listing_text(remote["pointer"], remote["names"])      # what the listing read: A, the pointer before the switch
            _time.sleep(0.003)                                             # the clock stamps in milliseconds: at least one apart
            passes["b"] = M.sync_all(root, 101, [dest], runner=_Recorder(respond))[0]
            return text
        return respond(argv)

    summary = M.prune_manifests(root, [dest], keep=1, runner=_Recorder(overlap))
    b = passes["b"]["manifest"]
    assert passes["b"]["switched"] and passes["b"]["dest_current"] == b and remote["pointer"] == b
    # the record names B, by the sync: the prune's merge of A -- observed before the switch -- was refused; nothing held, no error
    rec_b = M.read_pointers(root)["cluster_a"]
    assert rec_b["current"] == b and rec_b["source"] == "sync" and rec_b["observed_utc"] > rec_a["observed_utc"]
    assert M.unrecorded_pointers(root) == {} and M.last_known_currents(root) == {"cluster_a": b}
    assert "[pointers] ERROR" not in capsys.readouterr().out
    # the prune reports the listing the record moved past, the record's name standing; its references are the record's B, not A
    st = summary["stale"]["cluster_a"]
    assert st["listed"] == a and st["recorded"] == b and st["source"] == "sync"
    assert st["observed_utc"] < st["recorded_utc"] == rec_b["observed_utc"]
    assert summary["references"] == {b: ["cluster_a" + REASON]} and summary["sides"][0]["error"] is None
    assert summary["sides"][1]["current"] == a and summary["sides"][1]["error"] is None, "the destination's side as its listing read it"
    stale_line = (f"[pointers] STALE cluster_a's CHECKSUMS.CURRENT read {a} by a listing begun at {st['observed_utc']}; sync recorded {b} "
                  f"observed at {st['recorded_utc']} meanwhile, which stands")
    text = M.format_retention(summary)
    assert stale_line in text and "ERROR" not in text and text.splitlines().index(stale_line) == 3
    assert M.format_retention({"sides": [], "stale": summary["stale"]}, header=False) == stale_line, "what the pass would log"
    assert _grading_host_names(root) == [a, b], "the prune's own the grading host listing predates B: nothing to delete yet"
    # the destination drops out before its probe for three passes past the keep window, each failed row replacing the one that named
    # B: the record's B keeps B on the grading host every pass, with no refusal -- before this, the record said A and B went at pass 102
    down[0] = True
    for seq in (102, 103, 104):
        rows = M.sync_all(root, seq, [dest], runner=_Recorder(respond), keep=1)
        assert rows[0]["error"] and rows[0]["dest_current"] is None and not rows[0]["switched"]
        assert b in _grading_host_names(root), f"pass {seq}: the destination's actual base {b} stays"
        st_file = M.read_state(root)
        assert st_file["destinations"][0]["dest_current"] is None
        ret = st_file["manifest_retention"]
        assert ret["sides"][0]["error"] is None and ret["references"] == {b: ["cluster_a" + REASON]} and ret["stale"] == {}
    assert _grading_host_names(root) == sorted([b, rows[0]["manifest"]]), "A, nobody's base, went at pass 102; B stayed"
    assert M.read_pointers(root)["cluster_a"]["current"] == b
    # reachable again: its probe reads B, still on the grading host, so the copy is the delta since B; the clean verify switches to C
    down[0] = False
    rows = M.sync_all(root, 105, [dest], runner=_Recorder(respond), keep=1)
    c = rows[0]["manifest"]
    assert rows[0]["switched"] and rows[0]["copy_mode"] == "incremental" and rows[0]["copy_since"] == b and rows[0]["note"] is None
    assert M.read_pointers(root)["cluster_a"]["current"] == c and _grading_host_names(root) == [c]

    # the hold face, on a bare root: the record names ONE (t1); a switch to THREE whose merge failed is held (t5); a listing begun
    # BEFORE that switch (t3) merges TWO -- fresher than the record, older than the hold -- WITHOUT releasing the hold: both kept
    root2 = str(tmp_path / "bare")
    os.makedirs(root2)
    one, two, three, four = (M.write_manifest(root2, s) for s in (1, 2, 3, 4))
    assert M.record_pointer(root2, "d", one, "sync", now=synth.t(1))
    hold = M.hold_unrecorded(root2, "d", three, "sync", now=synth.t(5), error="the pointer record's lock x was held")
    assert M.record_pointer(root2, "d", two, "listing", now=synth.t(3)) and M.read_pointers(root2)["d"]["current"] == two
    assert os.path.exists(hold) and [h["current"] for h in M.unrecorded_pointers(root2)["d"]] == [three], "a merge older than the hold releases nothing"
    held_reason = f"d's CHECKSUMS.CURRENT as sync observed it at {synth.t(5)}, not yet recorded: kept until a merge records d's pointer"
    assert M.manifest_references(root2, M.last_known_currents(root2), M.unrecorded_pointers(root2)) == {two: ["d" + REASON], three: [held_reason]}
    summary2 = M.prune_manifests(root2, [], keep=1, configured=["d"])
    assert _grading_host_names(root2) == [two, three, four] and summary2["sides"][0]["removed"] == 1 and summary2["sides"][0]["error"] is None
    # a merge OLDER than the record is refused outright: nothing written, the hold untouched
    assert not M.record_pointer(root2, "d", one, "verify", now=synth.t(2))
    assert M.read_pointers(root2)["d"] == {"current": two, "observed_utc": synth.t(3), "source": "listing"} and os.path.exists(hold)
    # the hold's own millisecond, another name: merged (fresher than the record) but the hold stands (not observed strictly after it)
    assert M.record_pointer(root2, "d", two, "verify", now=synth.t(5)) and os.path.exists(hold)
    assert M.read_pointers(root2)["d"] == {"current": two, "observed_utc": synth.t(5), "source": "verify"}
    # a merge observed AFTER the held observation releases it, whatever name it records
    assert M.record_pointer(root2, "d", four, "sync", now=synth.t(6)) and not os.path.exists(hold) and M.unrecorded_pointers(root2) == {}
    # a merge of the HELD name releases its hold even when observed before it: the hold is answered, its name is on record
    hold2 = M.hold_unrecorded(root2, "e", three, "sync", now=synth.t(9), error="x")
    assert M.record_pointer(root2, "e", three, "listing", now=synth.t(7)) and not os.path.exists(hold2)
    assert M.read_pointers(root2)["e"] == {"current": three, "observed_utc": synth.t(7), "source": "listing"}


def test_observations_are_stamped_when_made_a_same_millisecond_name_is_held_and_the_last_look_before_a_deletion_reads_the_record_again(tmp_path, monkeypatch):
    """The stamps behind the freshness rule, and the two guards around it: a sync's observation is the switch as it completed, else
    the base read as it BEGAN; a verify's and a listing's the read as it began (a pointer switched during a read is the fresher
    observation); two names in the same millisecond are both kept, the later-merged one held until a fresher merge decides; and
    right before a deletion the record and the holds are read once more, so a pointer merged since the plan still protects what
    it names."""
    root = _setup(tmp_path)
    dest = _ssh_dest()
    n = len(M.log_files(root))
    base = M.write_manifest(root, _seq(root))
    stamps = {"clean": True}

    def respond(argv):
        joined = " ".join(argv)
        if _is_probe(argv):
            stamps["probe"] = R.utc_now()
            return base
        if _is_verify(argv):
            if stamps["clean"]:
                return _clean_json(n, base)
            return json.dumps({"ok": n - 1, "missing": ["records/x"], "corrupt": [], "mode": "incremental", "since": base, "hashed": 1,
                               "presence_only": n - 2})
        if "CHECKSUMS.CURRENT.tmp" in joined and "mv -f" in joined:
            stamps["switch"] = R.utc_now()
            return ""
        if _is_cat(argv):
            stamps["cat"] = R.utc_now()
            return base + "\n"
        if _is_listing(argv):
            stamps["listing"] = R.utc_now()
            return _listing_text(base, [base])
        return ""

    # a sync that switches: observed as the switch completed (at or after the switch's ssh)
    before = R.utc_now()
    res = M.sync(root, dest, _seq(root) + 100, runner=_Recorder(respond))
    assert res.switched and res.dest_current == res.manifest and before <= stamps["switch"] <= res.observed_utc <= R.utc_now()
    # a sync that does not switch: its dest_current is the base, observed as the base read BEGAN (before the probe answered)
    stamps["clean"] = False
    before = R.utc_now()
    res2 = M.sync(root, dest, _seq(root) + 200, runner=_Recorder(respond))
    assert not res2.switched and res2.dest_current == base and before <= res2.observed_utc <= stamps["probe"]
    stamps["clean"] = True
    assert M.sync(root, dest, _seq(root) + 300, runner=_Recorder(respond), full=True).observed_utc is not None, "a full sync still switches"
    # a verify's read and a listing's read: each stamped as it began; the pass records the sync's own stamp when no time is fixed
    before = R.utc_now()
    row = M.verify_destination(root, dest, runner=_Recorder(respond))
    rec = M.read_pointers(root)["cluster_a"]
    assert row["remote_current"] == base and rec["source"] == "verify" and before <= rec["observed_utc"] <= stamps["cat"]
    before = R.utc_now()
    M.prune_manifests(root, [dest], keep=96, runner=_Recorder(respond))
    rec = M.read_pointers(root)["cluster_a"]
    assert rec["source"] == "listing" and before <= rec["observed_utc"] <= stamps["listing"]
    rows = M.sync_all(root, _seq(root) + 400, [dest], runner=_Recorder(respond))
    rec = M.read_pointers(root)["cluster_a"]
    assert rows[0]["switched"] and rec == {"current": rows[0]["manifest"], "observed_utc": rec["observed_utc"], "source": "sync"}
    assert stamps["switch"] <= rec["observed_utc"] <= R.utc_now()
    assert M.sync_all(root, _seq(root) + 500, [dest], runner=_Recorder(respond), now=synth.t(1))[0]["switched"]
    assert M.read_pointers(root)["cluster_a"]["current"] == rows[0]["manifest"], "a pass whose fixed time is older than the record: refused"

    # two names observed in the same millisecond cannot be ordered: the second merge is refused and HELD besides the recorded name;
    # the same name again in that millisecond merges and leaves the hold; a merge observed after it decides
    root2 = str(tmp_path / "bare")
    os.makedirs(root2)
    one, two, three = (M.write_manifest(root2, s) for s in (1, 2, 3))
    assert M.record_pointer(root2, "d", one, "sync", now=synth.t(7))
    assert not M.record_pointer(root2, "d", two, "listing", now=synth.t(7))
    assert M.read_pointers(root2)["d"] == {"current": one, "observed_utc": synth.t(7), "source": "sync"}
    assert len(M.unrecorded_pointers(root2)["d"]) == 1
    held = M.unrecorded_pointers(root2)["d"][0]
    assert held["current"] == two and held["source"] == "listing" and held["observed_utc"] == synth.t(7)
    assert held["error"] == f"observed in the same millisecond as the recorded {one}: the order is unknown"
    tie_reason = f"d's CHECKSUMS.CURRENT as listing observed it at {synth.t(7)}, not yet recorded: kept until a merge records d's pointer"
    assert M.manifest_references(root2, M.last_known_currents(root2), M.unrecorded_pointers(root2)) == {one: ["d" + REASON], two: [tie_reason]}
    summary = M.prune_manifests(root2, [], keep=1, configured=["d"])
    assert _grading_host_names(root2) == [one, two, three] and summary["sides"][0]["removed"] == 0 and summary["unrecorded"]["d"] == [held]
    assert M.record_pointer(root2, "d", one, "verify", now=synth.t(7)) and os.path.exists(M.unrecorded_path(root2, "d", two, synth.t(7)))
    assert M.read_pointers(root2)["d"]["source"] == "verify"
    assert M.record_pointer(root2, "d", two, "sync", now=synth.t(8)) and M.unrecorded_pointers(root2) == {}
    assert M.read_pointers(root2)["d"]["current"] == two
    # a torn hold file is no hold -- and not ignored either (re-check of 360d61f4b, MEDIUM): it REFUSES the grading host's deletions by
    # name, the destination its name's segment says, until a merge records THAT destination's pointer, which removes it
    torn = M.unrecorded_path(root2, "d", two, synth.t(8))
    with open(torn, "w") as fh:
        fh.write("{torn")
    assert M.unrecorded_pointers(root2) == {}
    assert M.unreadable_holds(root2) == [{"path": torn, "destination": "d", "problem": "not JSON"}]
    summary = M.prune_manifests(root2, [], keep=1, configured=["d"])
    assert summary["sides"][0]["error"] == (
        f"refused: 1 hold file(s) under {M.unrecorded_dir(root2)} cannot be read or validated: {os.path.basename(torn)} (not JSON; named "
        "for d by its file name, released once a sync, verify or prune listing records d's pointer); a hold is a destination's pointer "
        "whose merge into the record failed, and one that cannot be read may name the base the grading host would otherwise delete, so "
        "nothing is deleted on the grading host while any stands")
    assert summary["sides"][0]["removed"] == 0 and summary["sides"][0]["deleted_names"] == [one] and _grading_host_names(root2) == [one, two, three]
    assert summary["unreadable"] == [{"path": torn, "destination": "d", "problem": "not JSON"}] and summary["unrecorded"] == {}
    assert (f"[pointers] UNREADABLE hold {torn}: not JSON; named for d by its file name: the grading host deletes nothing until a sync, verify "
            "or prune listing records d's pointer, which releases it") in M.format_retention(summary)
    # a whole document that is not a hold refuses the same way; another destination's merge and a recorded absence release neither
    not_a_hold = M.unrecorded_path(root2, "d", three, synth.t(8))
    with open(not_a_hold, "w") as fh:
        json.dump({"destination": "d", "current": "nonsense"}, fh)
    assert sorted((u["path"], u["destination"], u["problem"]) for u in M.unreadable_holds(root2)) == sorted(
        [(torn, "d", "not JSON"), (not_a_hold, "d", "not a hold: 'nonsense' is not a manifest name")])
    assert M.record_pointer(root2, "e", three, "verify", now=synth.t(9)) and os.path.exists(torn) and os.path.exists(not_a_hold)
    assert not M.record_pointer(root2, "d", None, "listing", now=synth.t(9), absent=True) and os.path.exists(torn)
    summary = M.prune_manifests(root2, [], keep=1, configured=["d"])
    assert summary["sides"][0]["error"].startswith("refused: 2 hold file(s) under ") and summary["sides"][0]["deleted_names"] == [one]
    assert [u["path"] for u in summary["unreadable"]] == sorted([torn, not_a_hold]) and _grading_host_names(root2) == [one, two, three]
    assert M.record_pointer(root2, "d", three, "sync", now=synth.t(9)) and not os.path.exists(torn) and not os.path.exists(not_a_hold)
    assert M.unreadable_holds(root2) == []
    summary = M.prune_manifests(root2, [], keep=1, configured=["d"])
    assert summary["sides"][0]["error"] is None and summary["unreadable"] == [] and _grading_host_names(root2) == [three], "THREE by d and e; ONE and TWO go"
    # stamps that cannot be read leave the order undecided: the merge lands (the record is never stuck), a hold stays
    with open(M.pointers_path(root2), "w") as fh:
        json.dump({"utc": "x", "destinations": {"f": {"current": one, "observed_utc": "yesterday", "source": "hand"}}}, fh)
    hold_f = M.hold_unrecorded(root2, "f", two, "sync", now="yesterday", error="x")
    assert M.record_pointer(root2, "f", three, "sync", now=synth.t(1)) and M.read_pointers(root2)["f"]["current"] == three
    assert os.path.exists(hold_f), "a hold whose stamp cannot be read stays until a merge of its name"
    assert M.record_pointer(root2, "f", two, "listing", now=synth.t(2)) and not os.path.exists(hold_f)

    # the last look before a deletion: a pointer merged between the plan and the deletion (another process's sync; injected here
    # where prune_manifests reads the record again) is added to the references and its manifest kept, on the grading host and the side
    root3 = str(tmp_path / "late")
    os.makedirs(root3)
    names = [M.write_manifest(root3, s) for s in (1, 2, 3, 4)]
    local = M.Destination("m", "local", str(tmp_path / "late-mirror"))
    os.makedirs(os.path.join(local.path, M.MANIFEST_DIR))
    for nm in names:
        shutil.copy2(os.path.join(root3, M.MANIFEST_DIR, nm), os.path.join(local.path, M.MANIFEST_DIR, nm))
    M.switch_current(local.path, names[3])
    real_late = M._late_references
    calls = []

    def late_with_a_concurrent_merge(root_, live):
        calls.append(dict(live))
        M.record_pointer(root_, "other", names[1], "sync", now=synth.t(20))     # another process recorded a pointer meanwhile
        return real_late(root_, live)

    monkeypatch.setattr(M, "_late_references", late_with_a_concurrent_merge)
    summary = M.prune_manifests(root3, [local], keep=1, configured=["m"])
    assert calls == [{}] and summary["references"] == {names[3]: ["m" + REASON], names[1]: ["other" + REASON]}
    assert _grading_host_names(root3) == [names[1], names[3]] and _dest_names(local.path) == [names[1], names[3]]
    dev, side = summary["sides"]
    assert (dev["removed"], dev["deleted_names"], dev["error"]) == (2, [names[0], names[2]], None) and dev["referenced"] == summary["references"]
    assert (side["removed"], side["current"], side["deleted_names"]) == (2, names[3], [names[0], names[2]]) and names[1] in side["referenced"]
    monkeypatch.undo()
    # a dry run takes no last look (nothing goes) and the pass passes its own results as the live pointers
    summary = M.prune_manifests(root3, [local], keep=1, dry_run=True)
    assert summary["references"] == {names[3]: ["m" + REASON], names[1]: ["other" + REASON]} and summary["stale"] == {}


# ------------------------------------------------- a hold is one file per observation: two observers of one destination both stand


def test_concurrent_holds_of_one_destination_each_stand_so_an_older_observation_held_later_cannot_lose_the_live_pointer(tmp_path, monkeypatch, capsys):
    """Closing review of 8eef8bbb2 (MEDIUM): the pass switched the destination from A to B and its merge timed out, so B was held;
    an on-demand prune that had read A BEFORE the switch (its listing stamped earlier) timed out too and held A -- in the same
    per-destination file, over B, by an unconditional replace -- so the record and the hold both named A, and once B fell outside
    the newest K the grading host deleted it, the destination's actual base, with no refusal (a whole-tree copy on recovery). A hold is
    one file per OBSERVATION now, never overwritten by another observation of the destination: both stand, B is kept on every
    side, and a merge releases only the holds it covers."""
    import contextlib
    root = _setup(tmp_path)
    dest = _ssh_dest()
    n = len(M.log_files(root))
    remote = {"pointer": None, "names": {}}
    down = [False]
    listing_reads = []                     # a pointer the listing shell answers instead of the live one: a read that predates a switch

    def respond(argv):
        joined = " ".join(argv)
        if down[0]:
            raise subprocess.SubprocessError("ssh: connect to host login-0 port 22: Connection timed out")
        if _is_probe(argv):
            return remote["pointer"] or ""
        if _is_verify(argv):
            return _clean_json(n, remote["pointer"]) if remote["pointer"] else _full_json(n)
        if _is_manifest_copy(argv):
            remote["names"][os.path.basename(argv[-2])] = 1
            return ""
        if "CHECKSUMS.CURRENT.tmp" in joined and "mv -f" in joined:
            remote["pointer"] = joined.split("printf %s '", 1)[1].split("'", 1)[0]
            return ""
        if _is_listing(argv):
            return _listing_text(listing_reads.pop() if listing_reads else remote["pointer"], remote["names"])
        return ""

    real_lock = M.pointers_lock

    @contextlib.contextmanager
    def stuck_lock(root_, timeout_s=M.POINTERS_LOCK_TIMEOUT_S):
        raise TimeoutError(f"the pointer record's lock {M.pointers_lock_path(root_)} was held by another observer for over 60 s; "
                           f"this observation was not recorded")
        yield  # pragma: no cover

    rows = M.sync_all(root, 100, [dest], runner=_Recorder(respond), now=synth.t(1))
    a = rows[0]["manifest"]
    assert rows[0]["switched"] and remote["pointer"] == a and M.read_pointers(root)["cluster_a"] == {"current": a, "observed_utc": synth.t(1), "source": "sync"}
    # pass 101 switches the destination to B at t(2); its merges (the sync's, then the listing's) cannot take the lock: B is held
    monkeypatch.setattr(M, "pointers_lock", stuck_lock)
    rows = M.sync_all(root, 101, [dest], runner=_Recorder(respond), now=synth.t(2), keep=1)
    b = rows[0]["manifest"]
    assert rows[0]["switched"] and remote["pointer"] == b and M.read_pointers(root)["cluster_a"]["current"] == a
    assert _grading_host_names(root) == [b], "the pass's own result names B, so A is no one's base"
    capsys.readouterr()
    # the on-demand prune had read A before the switch -- its listing is stamped between the record's A and the switch -- and its
    # merge cannot take the lock either: its hold of A is written AFTER the hold of B
    between = R.epoch_ms_to_utc(R.utc_to_epoch_ms(synth.t(1)) + 500)
    listing_reads.append(a)
    summary = M.prune_manifests(root, [dest], keep=1, runner=_Recorder(respond), now=between)
    out = capsys.readouterr().out
    assert "[pointers] ERROR recording cluster_a's pointer: the pointer record's lock" in out
    assert f"the observation (cluster_a: {a}, listing) is held in " in out
    assert summary["sides"][0]["error"] is None and summary["sides"][0]["removed"] == 0 and _grading_host_names(root) == [b]
    # the lock works again but the destination is down before its probe: each failed pass replaces the row that named B, the record
    # still names A -- and B, the destination's actual base, stays on the grading host past the retention boundary, held: no refusal
    monkeypatch.setattr(M, "pointers_lock", real_lock)
    down[0] = True
    outs = []
    for seq in (102, 103, 104):
        rows = M.sync_all(root, seq, [dest], runner=_Recorder(respond), now=synth.t(seq), keep=1)
        assert b in _grading_host_names(root), f"pass {seq}: the held live pointer {b} survives the older observation held after it"
        assert rows[0]["error"] and rows[0]["dest_current"] is None and not rows[0]["switched"]
        ret = M.read_state(root)["manifest_retention"]
        assert ret["sides"][0]["error"] is None and ret["sides"][0]["removed"] == (0 if seq == 102 else 1), f"pass {seq}: by the rule, not refused"
        outs.append(capsys.readouterr().out)
    assert M.read_pointers(root)["cluster_a"]["current"] == a and _grading_host_names(root) == sorted([b, rows[0]["manifest"]])
    # both observations are held, freshest first: B as pass 101's listing observed it at t(2) (the sync's hold of the same
    # observation -- the same name at the same instant -- is the same file, the later observer's), then A as the prune read it
    held = M.unrecorded_pointers(root)["cluster_a"]
    hold_b, hold_a = M.unrecorded_path(root, "cluster_a", b, synth.t(2)), M.unrecorded_path(root, "cluster_a", a, between)
    assert [(h["current"], h["observed_utc"], h["source"], h["path"]) for h in held] == [(b, synth.t(2), "listing", hold_b), (a, between, "listing", hold_a)]
    assert all(h["error"].startswith("the pointer record's lock") for h in held) and hold_a != hold_b
    assert f"is held in {hold_a} and kept by the retention until a merge records cluster_a's pointer" in out
    assert sorted(f for f in os.listdir(M.unrecorded_dir(root)) if f.endswith(".json")) == sorted(os.path.basename(p) for p in (hold_a, hold_b))
    held_b = f"cluster_a's CHECKSUMS.CURRENT as listing observed it at {synth.t(2)}, not yet recorded: kept until a merge records cluster_a's pointer"
    held_a = f"cluster_a's CHECKSUMS.CURRENT as listing observed it at {between}, not yet recorded: kept until a merge records cluster_a's pointer"
    assert ret["unrecorded"] == {"cluster_a": [dict(h) for h in held]} and ret["references"] == {a: ["cluster_a" + REASON, held_a], b: [held_b]}
    for text in outs:
        lines = text.splitlines()
        assert [l for l in lines if l.startswith("[pointers] HELD")] == [
            f"[pointers] HELD cluster_a's CHECKSUMS.CURRENT {b} (observed by listing at {synth.t(2)}, not recorded: {held[0]['error']}); "
            f"kept on every side until a sync, verify or prune listing records cluster_a's pointer",
            f"[pointers] HELD cluster_a's CHECKSUMS.CURRENT {a} (observed by listing at {between}, not recorded: {held[1]['error']}); "
            f"kept on every side until a sync, verify or prune listing records cluster_a's pointer"], "one line per held observation, freshest first"
        assert "[pointers] ERROR" not in text and "refused" not in text
    # reachable again: the probe reads B, which the grading host still holds, so the copy is the delta since B; the clean verify switches
    # to C and the merge lands -- observed after both held observations, it releases both; nothing held, nothing reported
    down[0] = False
    rows = M.sync_all(root, 105, [dest], runner=_Recorder(respond), now=synth.t(105), keep=1)
    c = rows[0]["manifest"]
    assert rows[0]["switched"] and rows[0]["copy_mode"] == "incremental" and rows[0]["copy_since"] == b and rows[0]["note"] is None
    assert M.read_pointers(root)["cluster_a"]["current"] == c and M.unrecorded_pointers(root) == {}
    assert not os.path.exists(hold_a) and not os.path.exists(hold_b) and [f for f in os.listdir(M.unrecorded_dir(root)) if f.endswith(".json")] == []
    ret = M.read_state(root)["manifest_retention"]
    assert ret["unrecorded"] == {} and ret["references"] == {c: ["cluster_a" + REASON]} and _grading_host_names(root) == [c]
    assert "HELD" not in capsys.readouterr().out

    # on a bare root, both orders of writing: the fresher observation first and the older after it, and the older first and the
    # fresher after -- both stand either way, freshest first, every held name a reference, and a plan keeps the fresher name
    root2 = str(tmp_path / "bare")
    os.makedirs(root2)
    one, two, three = (M.write_manifest(root2, s) for s in (1, 2, 3))
    d_two = M.hold_unrecorded(root2, "d", two, "sync", now=synth.t(2), error="lock")
    d_one = M.hold_unrecorded(root2, "d", one, "listing", now=synth.t(1), error="lock")
    e_one = M.hold_unrecorded(root2, "e", one, "listing", now=synth.t(1), error="lock")
    e_two = M.hold_unrecorded(root2, "e", two, "sync", now=synth.t(2), error="lock")
    assert len({d_one, d_two, e_one, e_two}) == 4 and all(os.path.exists(p) for p in (d_one, d_two, e_one, e_two))
    held = M.unrecorded_pointers(root2)
    assert [(h["current"], h["source"], h["observed_utc"], h["path"]) for h in held["d"]] == [(two, "sync", synth.t(2), d_two), (one, "listing", synth.t(1), d_one)]
    assert [(h["current"], h["source"], h["observed_utc"], h["path"]) for h in held["e"]] == [(two, "sync", synth.t(2), e_two), (one, "listing", synth.t(1), e_one)]
    summary = M.prune_manifests(root2, [], keep=1, configured=["d", "e"])
    assert summary["sides"][0]["error"] is None and summary["sides"][0]["removed"] == 0 and _grading_host_names(root2) == [one, two, three]
    assert summary["unrecorded"] == {"d": [dict(h) for h in held["d"]], "e": [dict(h) for h in held["e"]]}
    assert set(summary["references"]) == {one, two} and len(summary["references"][two]) == 2 and len(summary["references"][one]) == 2
    text = M.format_retention(summary).splitlines()
    assert [l.split(" (observed", 1)[0] for l in text if l.startswith("[pointers] HELD")] == [
        f"[pointers] HELD d's CHECKSUMS.CURRENT {two}", f"[pointers] HELD d's CHECKSUMS.CURRENT {one}",
        f"[pointers] HELD e's CHECKSUMS.CURRENT {two}", f"[pointers] HELD e's CHECKSUMS.CURRENT {one}"]
    # the same observation held again (another observer of the same pointer at the same instant) is the same file, the later
    # observer's words; a name held at several instants is reported once, at its freshest, every file standing
    assert M.hold_unrecorded(root2, "d", two, "listing", now=synth.t(2), error="lock again") == d_two
    assert M.unrecorded_pointers(root2)["d"][0] == {"current": two, "observed_utc": synth.t(2), "source": "listing", "error": "lock again", "path": d_two}
    d_two_later = M.hold_unrecorded(root2, "d", two, "listing", now=synth.t(4), error="lock")
    assert d_two_later != d_two and [(h["current"], h["observed_utc"]) for h in M.unrecorded_pointers(root2)["d"]] == [(two, synth.t(4)), (one, synth.t(1))]
    assert len([f for f in os.listdir(M.unrecorded_dir(root2)) if f.endswith(".json")]) == 5
    # a merge releases what it covers and no more: TWO at t(3) for d -- TWO's holds by name (t(2) and t(4) alike), ONE's as older;
    # ONE at t(1) for e -- ONE's hold by name, TWO's (fresher) stays
    assert M.record_pointer(root2, "d", two, "verify", now=synth.t(3)) and "d" not in M.unrecorded_pointers(root2)
    assert not any(os.path.exists(p) for p in (d_one, d_two, d_two_later))
    assert M.record_pointer(root2, "e", one, "verify", now=synth.t(1))
    assert [(h["current"], h["path"]) for h in M.unrecorded_pointers(root2)["e"]] == [(two, e_two)] and not os.path.exists(e_one)
    # the file: the destination and the stamp in plain characters, then a digest of the real three -- distinct files for two
    # destinations whose plain forms coincide, and for one name at two instants
    p1, p2 = M.unrecorded_path(root2, "cluster/eu west", one, synth.t(1)), M.unrecorded_path(root2, "cluster_eu_west", one, synth.t(1))
    assert os.path.dirname(p1) == M.unrecorded_dir(root2) and os.path.basename(p1).startswith("cluster%2Feu%20west~") and os.path.basename(p2).startswith("cluster_eu_west~")
    assert p1 != p2 and os.path.basename(p1).endswith(".json") and M._hold_name_destination(os.path.basename(p1)) == "cluster/eu west"
    assert M._hold_name_destination(os.path.basename(p2)) == "cluster_eu_west", "each name decodes to its own exact destination"
    assert M.unrecorded_path(root2, "d", one, synth.t(1)) != M.unrecorded_path(root2, "d", one, synth.t(2))


def test_a_hold_written_while_a_merge_releases_stays_unless_the_merge_covers_it(tmp_path, monkeypatch):
    """Closing review of 8eef8bbb2 (MEDIUM, its second face): a successful merge read the destination's hold and then removed the
    destination's hold FILE, so a hold written in between -- a fresher observation, which the merge did not cover -- went with it.
    A release judges every hold file by its own content against the merge's name and stamp: a hold written after the release
    listed the directory is not seen and stays; one it listed goes only when the merge covers it (its name, or an observation
    older than the merge's), whenever it was written."""
    root = str(tmp_path / "bare")
    os.makedirs(root)
    one, two, three, four, five = (M.write_manifest(root, s) for s in (1, 2, 3, 4, 5))
    assert M.record_pointer(root, "d", one, "sync", now=synth.t(1))
    real_files = M._hold_files
    # a fresher hold written AFTER the release listed the directory: not seen, it stays; the older one listed is covered and goes
    older = M.hold_unrecorded(root, "d", two, "listing", now=synth.t(2), error="lock")

    def list_then_hold(root_):
        files = real_files(root_)
        M.hold_unrecorded(root_, "d", four, "sync", now=synth.t(5), error="lock")
        return files

    monkeypatch.setattr(M, "_hold_files", list_then_hold)
    assert M.record_pointer(root, "d", three, "verify", now=synth.t(3))
    monkeypatch.setattr(M, "_hold_files", real_files)
    assert M.read_pointers(root)["d"]["current"] == three and not os.path.exists(older)
    assert [(h["current"], h["observed_utc"]) for h in M.unrecorded_pointers(root)["d"]] == [(four, synth.t(5))]
    assert set(M.manifest_references(root, M.last_known_currents(root), M.unrecorded_pointers(root))) == {three, four}
    summary = M.prune_manifests(root, [], keep=1, configured=["d"])
    assert summary["sides"][0]["error"] is None and _grading_host_names(root) == [three, four, five], "FOUR, held, is kept; ONE and TWO go"
    # a fresher hold written BEFORE the listing: listed, judged on its stamp, it stays; the older holds listed with it go
    M.hold_unrecorded(root, "d", two, "listing", now=synth.t(4), error="lock")

    def hold_then_list(root_):
        M.hold_unrecorded(root_, "d", five, "sync", now=synth.t(7), error="lock")
        return real_files(root_)

    monkeypatch.setattr(M, "_hold_files", hold_then_list)
    assert M.record_pointer(root, "d", three, "verify", now=synth.t(6)), "the same name again, fresher: merged"
    monkeypatch.setattr(M, "_hold_files", real_files)
    assert [(h["current"], h["observed_utc"]) for h in M.unrecorded_pointers(root)["d"]] == [(five, synth.t(7))], "FOUR at t(5) and TWO at t(4) were covered"
    assert [f for f in os.listdir(M.unrecorded_dir(root)) if f.endswith(".json")] == [os.path.basename(M.unrecorded_path(root, "d", five, synth.t(7)))]
    summary = M.prune_manifests(root, [], keep=1, configured=["d"])
    assert summary["sides"][0]["removed"] == 1 and _grading_host_names(root) == [three, five], "FIVE as the newest and held, THREE by the record"
    # a release racing another: a file gone between the listing and the removal is nobody's error
    gone = M.hold_unrecorded(root, "d", two, "listing", now=synth.t(2), error="lock")

    def list_then_remove(root_):
        files = real_files(root_)
        os.remove(gone)
        return files

    monkeypatch.setattr(M, "_hold_files", list_then_remove)
    assert M.record_pointer(root, "d", three, "verify", now=synth.t(8))
    monkeypatch.setattr(M, "_hold_files", real_files)
    assert not os.path.exists(gone) and M.unrecorded_pointers(root) == {}, "the merge at t(8) covered FIVE at t(7) as well"


# ------------------------------------------------------- a hold the retention cannot read refuses the grading host, never reads as no hold


def test_an_unreadable_hold_refuses_the_grading_host_deletions_until_a_merge_records_that_destination(tmp_path, monkeypatch, capsys):
    """Re-check of 360d61f4b (MEDIUM): the record names A; pass 101 switches the destination to B with the record's lock stuck, so B
    is HELD; the hold then becomes unreadable (a storage error, or a crash before an un-fsynced hold was durable). _hold_files read
    the torn file as no hold, unrecorded_pointers dropped it, and the retention trusted the still-valid record naming A -- so once B
    left the newest K the grading host deleted it, the destination's actual base, and its recovery was a whole-tree copy. Now a hold file
    that cannot be read or validated REFUSES the grading host's deletions by name (the error names the destination its file name says and
    the file), reported in the summary (``unreadable``), the pass's log and the CLI's output; the destinations whose own listing
    succeeded are still pruned; only a merge that records THAT destination's pointer removes the file, after which pruning resumes."""
    import contextlib
    root = _setup(tmp_path)
    dest = _ssh_dest()
    with open(os.path.join(root, "config.json"), "w") as fh:           # a second, local destination: it prunes by the rule throughout
        json.dump({"mirrors": [{"name": "m", "kind": "local", "path": str(tmp_path / "mirror-m")}]}, fh)
    other = M.destinations(root)[0]
    n = len(M.log_files(root))
    remote = {"pointer": None, "names": {}}
    down = [False]

    def respond(argv):
        joined = " ".join(argv)
        if down[0]:
            raise subprocess.SubprocessError("ssh: connect to host login-0 port 22: Connection timed out")
        if _is_probe(argv):
            return remote["pointer"] or ""
        if _is_verify(argv):
            return _clean_json(n, remote["pointer"]) if remote["pointer"] else _full_json(n)
        if _is_manifest_copy(argv):
            remote["names"][os.path.basename(argv[-2])] = 1
            return ""
        if "CHECKSUMS.CURRENT.tmp" in joined and "mv -f" in joined:
            remote["pointer"] = joined.split("printf %s '", 1)[1].split("'", 1)[0]
            return ""
        if _is_listing(argv):
            return _listing_text(remote["pointer"], remote["names"])
        return ""

    # pass 100: A on record for both destinations
    rows = M.sync_all(root, 100, [dest, other], runner=_Recorder(respond), now=synth.t(1))
    a = rows[0]["manifest"]
    assert rows[0]["switched"] and rows[1]["switched"] and remote["pointer"] == a
    assert {d: r["current"] for d, r in M.read_pointers(root).items()} == {"cluster_a": a, "m": a}
    # pass 101 switches cluster_a to B; the record's lock is stuck, so B is held in its own file (the local destination sits this one out)
    @contextlib.contextmanager
    def stuck_lock(root_, timeout_s=M.POINTERS_LOCK_TIMEOUT_S):
        raise TimeoutError(f"the pointer record's lock {M.pointers_lock_path(root_)} was held by another observer for over 60 s; "
                           f"this observation was not recorded")
        yield  # pragma: no cover

    real_lock = M.pointers_lock
    monkeypatch.setattr(M, "pointers_lock", stuck_lock)
    rows = M.sync_all(root, 101, [dest], runner=_Recorder(respond), now=synth.t(2), keep=1)
    b = rows[0]["manifest"]
    hold = M.unrecorded_path(root, "cluster_a", b, synth.t(2))
    assert rows[0]["switched"] and remote["pointer"] == b and M.read_pointers(root)["cluster_a"]["current"] == a
    assert [h["current"] for h in M.unrecorded_pointers(root)["cluster_a"]] == [b] and os.path.exists(hold)
    assert _grading_host_names(root) == sorted([a, b]), "B is the newest, A still m's base"
    monkeypatch.setattr(M, "pointers_lock", real_lock)
    capsys.readouterr()
    # the hold is torn to zero bytes: no longer a hold -- and no longer nothing
    with open(hold, "w"):
        pass
    assert M.unrecorded_pointers(root) == {} and M.unreadable_holds(root) == [{"path": hold, "destination": "cluster_a", "problem": "not JSON"}]
    # the destination is down before its probe, pass after pass past the retention boundary at keep=1; the record still names A, so by
    # the record alone B would go -- the grading host plan is REFUSED instead, naming the destination and the file; nothing goes on the
    # grading host; the local destination, listed, is pruned by the rule; the torn file stands (m's merges release nothing of cluster_a's)
    down[0] = True
    manifests = [a, b]
    refusal = (f"refused: 1 hold file(s) under {M.unrecorded_dir(root)} cannot be read or validated: {os.path.basename(hold)} (not JSON; "
               "named for cluster_a by its file name, released once a sync, verify or prune listing records cluster_a's pointer); a hold is "
               "a destination's pointer whose merge into the record failed, and one that cannot be read may name the base the grading host would "
               "otherwise delete, so nothing is deleted on the grading host while any stands")
    unreadable_line = (f"[pointers] UNREADABLE hold {hold}: not JSON; named for cluster_a by its file name: the grading host deletes nothing "
                       "until a sync, verify or prune listing records cluster_a's pointer, which releases it")
    for seq in (102, 103, 104):
        rows = M.sync_all(root, seq, [dest, other], runner=_Recorder(respond), now=synth.t(seq), keep=1)
        manifests.append(rows[0]["manifest"])
        assert rows[0]["error"] and rows[0]["dest_current"] is None and rows[1]["switched"] and rows[1]["error"] is None
        assert _grading_host_names(root) == sorted(manifests), f"pass {seq}: nothing deleted on the grading host, B among what stays"
        assert os.path.exists(hold) and M.read_pointers(root)["cluster_a"]["current"] == a and M.read_pointers(root)["m"]["current"] == rows[1]["manifest"]
        ret = M.read_state(root)["manifest_retention"]
        dev, side = ret["sides"]
        assert dev["error"] == refusal and dev["removed"] == 0 and b in dev["deleted_names"], f"pass {seq}: refused by name, not pruned by the record"
        assert side["side"] == "m" and side["error"] is None and side["removed"] == (0 if seq == 102 else 1), f"pass {seq}: the listed destination prunes by the rule"
        assert ret["unreadable"] == [{"path": hold, "destination": "cluster_a", "problem": "not JSON"}] and ret["unrecorded"] == {}
        assert ret["references"] == {a: ["cluster_a" + REASON], rows[1]["manifest"]: ["m" + REASON]}, "the record's A, no hold: by the rule alone B would go"
        out = capsys.readouterr().out
        assert unreadable_line in out and f"ERROR {refusal}" in out and "HELD" not in out
    # a recorded absence releases nothing; the CLI shows the same refusal (a dry run exits 0, the real run 1 with the grading host side errored)
    assert not M.record_pointer(root, "cluster_a", None, "listing", absent=True) and os.path.exists(hold)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["mirrors", "prune-manifests", "--dry-run", "--keep", "1"]) == 0
    out = capsys.readouterr().out
    assert unreadable_line in out and f"ERROR {refusal}" in out and "DRY RUN" in out and _grading_host_names(root) == sorted(manifests)
    assert cli.main(["mirrors", "prune-manifests", "--keep", "1"]) == 1
    out = capsys.readouterr().out
    assert unreadable_line in out and f"ERROR {refusal}" in out and _grading_host_names(root) == sorted(manifests) and os.path.exists(hold)
    # reachable again: the probe reads B, which the grading host still holds, so the copy is the delta since B; the clean verify switches to C
    # and the merge lands, which removes the torn file -- the refusal lifts and the grading host is pruned to the rule
    down[0] = False
    rows = M.sync_all(root, 105, [dest, other], runner=_Recorder(respond), now=synth.t(105), keep=1)
    c = rows[0]["manifest"]
    assert rows[0]["switched"] and rows[0]["copy_mode"] == "incremental" and rows[0]["copy_since"] == b and rows[0]["note"] is None
    assert M.read_pointers(root)["cluster_a"]["current"] == c and not os.path.exists(hold) and M.unreadable_holds(root) == []
    ret = M.read_state(root)["manifest_retention"]
    assert ret["unreadable"] == [] and ret["unrecorded"] == {} and all(s["error"] is None for s in ret["sides"])
    assert ret["references"] == {c: ["cluster_a" + REASON, "m" + REASON]} and _grading_host_names(root) == [c]
    out = capsys.readouterr().out
    assert "UNREADABLE" not in out and "refused" not in out and "[pointers] ERROR" not in out


def test_a_hold_is_written_durably_and_a_failed_write_leaves_no_partial_hold(tmp_path, monkeypatch):
    """The hold's writer: a per-process, per-write temp name beside the hold whose suffix is never ``.json`` (so a write in progress is
    never listed as a hold), the document fsynced, renamed into place, the file and its directory fsynced; a write that fails at any
    step -- the rename, the document -- leaves no partial ``.json`` and no temp file behind and raises, so the directory holds a whole
    hold or none; a temp file some other writer left (a crash before its rename) is neither a hold nor a problem."""
    import stat
    root = str(tmp_path / "bare")
    os.makedirs(root)
    one, two = (M.write_manifest(root, s) for s in (1, 2))
    d = M.unrecorded_dir(root)
    seen = {"replace": [], "fsync": []}
    real_replace, real_fsync = os.replace, os.fsync

    def spy_replace(src, dst, *a, **k):
        seen["replace"].append((src, dst))
        return real_replace(src, dst, *a, **k)

    def spy_fsync(fd):
        seen["fsync"].append(os.fstat(fd).st_mode)
        return real_fsync(fd)

    monkeypatch.setattr(os, "replace", spy_replace)
    monkeypatch.setattr(os, "fsync", spy_fsync)
    p = M.hold_unrecorded(root, "d", one, "sync", now=synth.t(1), error="lock")
    monkeypatch.undo()
    (src, dst), = seen["replace"]
    assert dst == p and os.path.dirname(src) == d and src.startswith(p + ".") and src.endswith(".tmp") and not src.endswith(".json")
    assert sum(1 for m in seen["fsync"] if stat.S_ISREG(m)) >= 1 and sum(1 for m in seen["fsync"] if stat.S_ISDIR(m)) == 1, "the file, and its directory"
    assert sorted(os.listdir(d)) == [os.path.basename(p)] and json.load(open(p))["current"] == one
    # the rename fails (the disk full): the error is raised, no partial .json, no temp file; the earlier hold stands
    def no_space(src, dst, *a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", no_space)
    with pytest.raises(OSError, match="No space left"):
        M.hold_unrecorded(root, "d", two, "sync", now=synth.t(2), error="lock")
    monkeypatch.undo()
    assert sorted(os.listdir(d)) == [os.path.basename(p)], "nothing of the failed write remains"
    assert M.unreadable_holds(root) == [] and [h["current"] for h in M.unrecorded_pointers(root)["d"]] == [one]
    # the document's write fails midway: the same
    def io_error(*a, **k):
        raise OSError(5, "Input/output error")

    monkeypatch.setattr(json, "dump", io_error)
    with pytest.raises(OSError, match="Input/output error"):
        M.hold_unrecorded(root, "d", two, "sync", now=synth.t(3), error="lock")
    monkeypatch.undo()
    assert sorted(os.listdir(d)) == [os.path.basename(p)]
    # a temp file another writer left mid-way is not a hold and not a problem; the hold shape (.json) is what is read
    with open(os.path.join(d, f"{os.path.basename(p)}.999.deadbeef.tmp"), "w") as fh:
        fh.write("{")
    assert M.unreadable_holds(root) == [] and [h["current"] for h in M.unrecorded_pointers(root)["d"]] == [one]
    assert M.prune_manifests(root, [], keep=1, configured=["d"])["sides"][0]["error"] is None
    # record_pointer's failure path through the same writer: the merge fails and the hold cannot be written either -- both are named
    import contextlib

    @contextlib.contextmanager
    def stuck_lock(root_, timeout_s=M.POINTERS_LOCK_TIMEOUT_S):
        raise TimeoutError("the pointer record's lock was held by another observer for over 60 s; this observation was not recorded")
        yield  # pragma: no cover

    monkeypatch.setattr(M, "pointers_lock", stuck_lock)
    monkeypatch.setattr(os, "replace", no_space)
    with pytest.raises(OSError) as ex:
        M.record_pointer(root, "d", two, "sync", now=synth.t(4))
    monkeypatch.undo()
    assert "held by another observer" in str(ex.value) and f"the observation (d: {two}, sync) could NOT be held either: " in str(ex.value)
    assert "No space left" in str(ex.value) and sorted(f for f in os.listdir(d) if f.endswith(".json")) == [os.path.basename(p)]


# ------------------------------------------- a hold file's name encodes its destination exactly; a release never acts on a lossy match


def test_an_unreadable_hold_is_released_only_by_a_merge_of_the_exact_destination_its_name_encodes(tmp_path):
    """Re-check of 614abd0b4 (MEDIUM): the hold file's name carried the destination's PLAIN form, and a release removed an unreadable
    file on that lossy prefix -- so for two destinations whose plain forms coincide (``cluster/eu west`` and ``cluster_eu_west``), the
    second's successful merge released the first's torn hold, and the refusal did not stand until THAT destination recorded a pointer
    as claimed. The name now carries the destination percent-encoded (exactly decodable) and a digest of the exact name that the release
    recomputes; an unreadable file is released only by a merge of exactly the destination its name decodes to, and a name that decodes
    to none -- a stray file, a three-segment name from before this rule, a tampered digest -- is released by no merge."""
    root = str(tmp_path / "bare")
    os.makedirs(root)
    one, two, three = (M.write_manifest(root, s) for s in (1, 2, 3))
    odd, twin = "cluster/eu west", "cluster_eu_west"
    assert M._plain_segment(odd) == M._plain_segment(twin) == twin, "the plain forms coincide: the lossy identity"
    # the encoding is exact and strict: every character outside [A-Za-z0-9._-] as %XX, the separator and % included; decoding refuses
    # lower-case hex, a dangling %, an unescaped character and bytes that are not UTF-8
    assert M._encode_segment(odd) == "cluster%2Feu%20west" and M._decode_segment("cluster%2Feu%20west") == odd and M._encode_segment(twin) == twin
    assert M._encode_segment("a~b%c\u00e9") == "a%7Eb%25c%C3%A9" and M._decode_segment("a%7Eb%25c%C3%A9") == "a~b%c\u00e9"
    assert [M._decode_segment(x) for x in ("a%7eb", "a%7", "a b", "%FF", "a~b")] == [None] * 5
    # four files: a readable hold and a torn one for each destination; each name decodes to its own destination
    readable_odd = M.hold_unrecorded(root, odd, one, "sync", now=synth.t(1), error="lock")
    readable_twin = M.hold_unrecorded(root, twin, one, "sync", now=synth.t(1), error="lock")
    torn_odd, torn_twin = M.unrecorded_path(root, odd, two, synth.t(2)), M.unrecorded_path(root, twin, two, synth.t(2))
    for torn in (torn_odd, torn_twin):
        with open(torn, "w") as fh:
            fh.write("{torn")
    assert len({readable_odd, readable_twin, torn_odd, torn_twin}) == 4
    assert M._hold_name_destination(os.path.basename(torn_odd)) == odd and M._hold_name_destination(os.path.basename(torn_twin)) == twin
    assert sorted((u["destination"], u["path"]) for u in M.unreadable_holds(root)) == sorted([(odd, torn_odd), (twin, torn_twin)])
    assert set(M.unrecorded_pointers(root)) == {odd, twin}
    # the twin's merge, observed after every hold, releases the twin's two files and NOTHING of odd's: odd's readable hold and odd's
    # torn file stand, and the torn file still refuses the grading host by odd's exact name
    assert M.record_pointer(root, twin, three, "sync", now=synth.t(5))
    assert not os.path.exists(readable_twin) and not os.path.exists(torn_twin)
    assert os.path.exists(readable_odd) and os.path.exists(torn_odd), "a lossy match releases nothing"
    assert [h["current"] for h in M.unrecorded_pointers(root)[odd]] == [one] and twin not in M.unrecorded_pointers(root)
    assert [(u["destination"], u["path"]) for u in M.unreadable_holds(root)] == [(odd, torn_odd)]
    summary = M.prune_manifests(root, [], keep=1, configured=[odd, twin])
    err = summary["sides"][0]["error"]
    assert err.startswith("refused: 1 hold file(s) under ") and f"{os.path.basename(torn_odd)} (not JSON; named for {odd} by its file name, released once a sync, verify or prune listing records {odd}'s pointer)" in err
    assert summary["sides"][0]["removed"] == 0 and _grading_host_names(root) == [one, two, three]
    assert f"[pointers] UNREADABLE hold {torn_odd}: not JSON; named for {odd} by its file name:" in M.format_retention(summary)
    # odd's own merge releases odd's own; the refusal lifts
    assert M.record_pointer(root, odd, three, "sync", now=synth.t(5))
    assert not os.path.exists(readable_odd) and not os.path.exists(torn_odd) and M.unreadable_holds(root) == [] and M.unrecorded_pointers(root) == {}
    summary = M.prune_manifests(root, [], keep=1, configured=[odd, twin])
    assert summary["sides"][0]["error"] is None and _grading_host_names(root) == [three]
    # a name that decodes to no exact destination -- the three-segment shape written before this rule (the plain form, no digest of the
    # name), and a four-segment name whose digest is not the decoded name's -- is refused by name and released by NO merge, the
    # destinations whose plain form it shows included; an operator removes it, and pruning resumes
    four, five = (M.write_manifest(root, s) for s in (4, 5))
    legacy = os.path.join(M.unrecorded_dir(root), "cluster_eu_west~2026-01-01T00_00_09.000Z~0123abcd.json")
    tampered = os.path.join(M.unrecorded_dir(root), f"{M._encode_segment(odd)}~00000000~2026-01-01T00_00_09.000Z~0123abcd.json")
    for stray in (legacy, tampered):
        with open(stray, "w") as fh:
            fh.write("{torn")
    assert M._hold_name_destination(os.path.basename(legacy)) is None and M._hold_name_destination(os.path.basename(tampered)) is None
    assert sorted((u["destination"], u["problem"], u["path"]) for u in M.unreadable_holds(root)) == sorted([(None, "not JSON", legacy), (None, "not JSON", tampered)])
    assert M.record_pointer(root, twin, four, "verify", now=synth.t(6)) and M.record_pointer(root, odd, four, "verify", now=synth.t(6))
    assert os.path.exists(legacy) and os.path.exists(tampered), "neither destination's merge releases a file whose name decodes to none"
    summary = M.prune_manifests(root, [], keep=1, configured=[odd, twin])
    err = summary["sides"][0]["error"]
    assert err.startswith("refused: 2 hold file(s) under ") and _grading_host_names(root) == [three, four, five]
    assert err.count("its name does not decode to one exact destination, so no merge releases it: an operator reads it and removes it by hand") == 2
    assert M.format_retention(summary).count("its name does not decode to one exact destination, so no merge releases it: the grading host deletes nothing until an operator reads it and removes it by hand") == 2
    os.remove(legacy)
    os.remove(tampered)
    summary = M.prune_manifests(root, [], keep=1, configured=[odd, twin])
    assert summary["sides"][0]["error"] is None and summary["unreadable"] == [] and _grading_host_names(root) == [four, five], "FOUR by both records"
    # a READABLE hold under a legacy-shaped name is judged by its content as ever: listed for its destination, released by that
    # destination's covering merge, whatever its name's shape
    legacy_ok = os.path.join(M.unrecorded_dir(root), "cluster_eu_west~2026-01-01T00_00_07.000Z~89abcdef.json")
    with open(legacy_ok, "w") as fh:
        json.dump({"destination": twin, "current": five, "observed_utc": synth.t(7), "source": "sync", "error": "lock"}, fh)
    assert [h["current"] for h in M.unrecorded_pointers(root)[twin]] == [five] and M.unreadable_holds(root) == []
    assert M.record_pointer(root, odd, five, "sync", now=synth.t(8)) and os.path.exists(legacy_ok), "odd's merge: not twin's hold"
    assert M.record_pointer(root, twin, five, "sync", now=synth.t(8)) and not os.path.exists(legacy_ok)


def test_the_pointer_record_is_written_durably_and_a_failed_write_keeps_the_previous_record_and_releases_nothing(tmp_path, monkeypatch):
    """Re-check of 614abd0b4 (MEDIUM): the record's own write was a plain temp + rename, not crash-durable -- a merge could count as
    made and release the newer name's hold while the renamed record was not yet on disk, so a crash reverted the record to the older
    name with the hold already gone. The record is now written through the same durable writer as the holds, under the lock, before
    the merge counts or anything is released: a per-process temp name beside it, fsync, rename, the file and its directory fsynced;
    a write that fails before the rename raises, leaves the previous record intact, holds the observation and releases nothing."""
    import stat
    root = str(tmp_path / "bare")
    os.makedirs(root)
    one, two, three = (M.write_manifest(root, s) for s in (1, 2, 3))
    rec_path = M.pointers_path(root)
    assert M.record_pointer(root, "d", one, "sync", now=synth.t(1))
    older = M.hold_unrecorded(root, "d", two, "listing", now=synth.t(2), error="lock")     # covered by the merge at t(3) below
    seen = {"replace": [], "fsync": []}
    real_replace, real_fsync = os.replace, os.fsync

    def spy_replace(src, dst, *a, **k):
        seen["replace"].append((src, dst))
        return real_replace(src, dst, *a, **k)

    def spy_fsync(fd):
        st = os.fstat(fd)
        seen["fsync"].append((st.st_mode, st.st_ino))
        return real_fsync(fd)

    monkeypatch.setattr(os, "replace", spy_replace)
    monkeypatch.setattr(os, "fsync", spy_fsync)
    assert M.record_pointer(root, "d", three, "verify", now=synth.t(3))
    monkeypatch.undo()
    assert M.read_pointers(root)["d"]["current"] == three and not os.path.exists(older), "merged, and the older hold released"
    # the one rename is the record's, from a per-process, per-write temp name beside it; fsync on the record file (the temp before the
    # rename and the file after it -- one inode) and exactly once on the state directory
    (src, dst), = seen["replace"]
    assert dst == rec_path and os.path.dirname(src) == os.path.dirname(rec_path) and src.startswith(f"{rec_path}.{os.getpid()}.") and src.endswith(".tmp")
    regular = [ino for mode, ino in seen["fsync"] if stat.S_ISREG(mode)]
    directories = [ino for mode, ino in seen["fsync"] if stat.S_ISDIR(mode)]
    assert regular and set(regular) == {os.stat(rec_path).st_ino} and directories == [os.stat(os.path.dirname(rec_path)).st_ino]
    assert [f for f in os.listdir(os.path.dirname(rec_path)) if f.startswith(os.path.basename(rec_path) + ".")] == [], "no temp file left"
    # a write that fails before the rename (the disk full at the record's rename alone): the previous record stands byte for byte,
    # the observation is held (the hold's own write is unaffected), a hold the merge would have covered stays, no temp file remains
    kept = M.hold_unrecorded(root, "d", two, "listing", now=synth.t(4), error="lock")
    with open(rec_path, "rb") as fh:
        before = fh.read()

    def no_space_for_the_record(src, dst, *a, **k):
        if dst == rec_path:
            raise OSError(28, "No space left on device")
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(os, "replace", no_space_for_the_record)
    with pytest.raises(OSError) as ex:
        M.record_pointer(root, "d", three, "sync", now=synth.t(5))
    monkeypatch.undo()
    assert "No space left on device" in str(ex.value) and f"the observation (d: {three}, sync) is held in " in str(ex.value)
    with open(rec_path, "rb") as fh:
        assert fh.read() == before, "the previous record, untouched"
    assert M.read_pointers(root)["d"] == {"current": three, "observed_utc": synth.t(3), "source": "verify"}
    assert os.path.exists(kept), "a merge that did not land releases nothing"
    assert [(h["current"], h["observed_utc"]) for h in M.unrecorded_pointers(root)["d"]] == [(three, synth.t(5)), (two, synth.t(4))]
    assert [f for f in os.listdir(os.path.dirname(rec_path)) if f.startswith(os.path.basename(rec_path) + ".")] == []
    # the next merge that lands covers both: the hold of THREE by name, the hold of TWO as older
    assert M.record_pointer(root, "d", three, "sync", now=synth.t(6)) and M.unrecorded_pointers(root) == {} and not os.path.exists(kept)

