"""stage_objects' manifest publication. The push reads the remote
manifest first and records it holds that the local bank lacks (identity (target, staged_id), the pins key) REFUSE the push by name unless
--merge-remote folds them into the bank and pushes the union (the 2026-10-01 s2r staging had replaced the paper's 241 records with 2 on both
clusters). Further: (1) a remote record with the SAME identity and a DIFFERENT object_sha256 is a CONFLICT — two objects claim one pin —
refused by name with both values and both records, never a silent replace and never a local win under --merge-remote; it is checked on EVERY
cluster BEFORE any object bytes move (a bank-level preflight, a per-cell check before stage_to); (2) --merge-remote reads every cluster first,
folds all foreign records, builds ONE union and publishes that same union to each cluster; (3) the publish is a compare-and-swap on the
remote file's digest with temp file + rename — a peer that wrote between the read and the publish is re-read and folded, and every cluster
gets the new union. Local hosts (`local:`) stand in for the clusters; synthetic records and bytes."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402
from _rsync_fallback import rsync_on_path  # noqa: E402,F401  (autouse: the fallback rsync on PATH where the host has none; a no-op where it has one)

sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
import stage_objects as SO  # noqa: E402

REL = "exp/objects/sdfbase/full22"
KEY = "qwen35_4b_demo"


def _rec(target, staged, sha="a" * 64, **over):
    r = {"target": target, "staged_id": staged, "lane": staged.split("__", 1)[0], "cell_id": staged.split("__", 1)[1], "object_sha256": sha, "status": "staged"}
    r.update(over); return r


def _bank(tmp_path, recs):
    bank = tmp_path / "bank"; records = bank / "_records"; records.mkdir(parents=True, exist_ok=True)
    for r in recs:
        SO.write_record_atomic(records / f"{r['target']}__{r['staged_id']}.json", r)
    manifest = bank / "staging_manifest.json"; SO.rebuild_manifest(records, manifest)
    return bank, records, manifest


def _host(tmp_path, label="cluster_a"):
    shared = tmp_path / f"shared_{label}"; shared.mkdir(exist_ok=True)
    return SO.Host(label, f"local:{shared}", str(shared), False), shared


def _remote(shared, recs):
    d = shared / REL; d.mkdir(parents=True, exist_ok=True)
    (d / "staging_manifest.json").write_text(json.dumps(recs)); return d / "staging_manifest.json"


def _read(shared):
    p = shared / REL / "staging_manifest.json"
    return json.loads(p.read_text()) if p.is_file() else None


def _ids(recs):
    return {(r["target"], r["staged_id"]): r["object_sha256"] for r in recs}


def _site(tmp_path, labels):
    """A site of local clusters: one shared root each, the stager's own Host over it."""
    site = tmp_path / "site"; (site / "clusters").mkdir(parents=True, exist_ok=True)
    (site / "config.env").write_text(f'SISTEER_CLUSTERS="{" ".join(labels)}"\n')
    roots = {}
    for lab in labels:
        shared = tmp_path / f"shared_{lab}"; shared.mkdir(exist_ok=True); roots[lab] = shared
        (site / "clusters" / f"{lab}.env").write_text(f'CLUSTER_SSH="local:{shared}"\nCLUSTER_SHARED_ROOT="{shared}"\nCLUSTER_PARTITION=general\n'
                                                      f'CLUSTER_QOS=high\nCLUSTER_SUBSTRATE=slurm\nCLUSTER_LABEL={lab}\n')
    return site, roots


def _source(tmp_path, cell, payload):
    d = tmp_path / "src" / cell; d.mkdir(parents=True, exist_ok=True)
    (d / "object.pt").write_bytes(payload); sha = hashlib.sha256(payload).hexdigest()
    (d / "derived_meta.json").write_text(json.dumps({"kind": "psr_paired_write", "derived_id": cell, "object_sha256": sha, "parents": ["a" * 64, "b" * 64]}))   # derived_id: the sidecar names its cell (stage_objects readiness)
    return d, sha


def _stage(base, site, cells, *extra):
    cf = base / "cells.json"; cf.write_text(json.dumps(cells))
    r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/full22/stage_objects.py"), "--site-dir", str(site), "--bank", str(base / "bank"), "--target", KEY,
                        "--cells", str(cf), "--clusters", "cluster_a,cluster_b", *extra], capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout + r.stderr


def test_a_push_over_foreign_records_refuses_by_name_and_leaves_the_remote_untouched(tmp_path):
    bank, records, manifest = _bank(tmp_path, [_rec(KEY, "base__cellA")])
    h, shared = _host(tmp_path)
    paper = [_rec("gemma4e4b", "base__x"), _rec("qwen35_4b", "derived__y", sha="b" * 64)]
    _remote(shared, paper)
    with pytest.raises(SO.ForeignRecords, match=r"holds 2 record\(s\) this bank lacks \(gemma4e4b/base__x, qwen35_4b/derived__y\) — a push would erase them \(D905\)"):
        SO.push_manifest(h, REL, manifest, records=records)
    assert _read(shared) == paper                                                                   # untouched
    assert sorted(p.name for p in records.glob("*.json")) == [f"{KEY}__base__cellA.json"]            # the bank untouched too


def test_one_identity_two_objects_refuses_by_name_on_every_path_before_any_byte_moves(tmp_path):
    """A remote record that shares an identity with a local record but not its object_sha256 is never replaced and never wins —
    the publish refuses naming both values and both records (--merge-remote included; a missing sha on one side is a conflict too); a
    re-serialisation with the same sha passes. The real stager refuses the conflict BEFORE any copy: at the bank-level preflight (no pull,
    no copy) and per cell before stage_to (nothing rsynced to any cluster, status remote_manifest_conflict)."""
    bank, records, manifest = _bank(tmp_path, [_rec(KEY, "base__cellA", sha="1" * 64)])
    h, shared = _host(tmp_path)
    theirs = _rec(KEY, "base__cellA", sha="0" * 64, status="stale", note="another bank's object")
    _remote(shared, [_rec("gemma4e4b", "base__x"), theirs])
    for kw in ({}, {"merge_remote": True}):
        with pytest.raises(SO.ManifestConflict) as ei:
            SO.push_manifest(h, REL, manifest, records=records, **kw)
        msg = str(ei.value)
        assert f"{KEY}/base__cellA: local object_sha256 {'1' * 64} vs cluster_a {'0' * 64}" in msg and "two objects claim one pin" in msg and "D1311" in msg
        assert json.dumps(theirs, sort_keys=True)[:200] in msg and '"object_sha256": "' + "1" * 64 in msg               # both records, by content
        assert _ids(_read(shared)) == {("gemma4e4b", "base__x"): "a" * 64, (KEY, "base__cellA"): "0" * 64}              # the remote untouched
        assert sorted(p.name for p in records.glob("*.json")) == [f"{KEY}__base__cellA.json"]                        # nothing folded
    assert SO.manifest_conflicts([_rec(KEY, "base__cellA", sha="1" * 64)], [_rec(KEY, "base__cellA", sha="")])[1]      # a missing sha is a conflict
    assert SO.manifest_conflicts([_rec(KEY, "base__cellA", sha="1" * 64)], [_rec(KEY, "base__cellA", sha="1" * 64, status="x", extra=1)]) == ([], [])   # same object re-serialised: passes
    # the real stager, two clusters — per cell: cluster_b pins the identity to X, the bytes being staged are Y
    base = tmp_path / "e2e"; base.mkdir(); site, roots = _site(base, ["cluster_a", "cluster_b"])          # fresh roots and bank: the e2e world
    cell = "x__psr_ce_all_minus_ctl"; d, y = _source(tmp_path, cell, b"these bytes are Y " * 32)
    _remote(roots["cluster_b"], [_rec(KEY, f"derived__{cell}", sha="f" * 64)])
    rc, out = _stage(base, site, [{"cell_id": cell, "lane": "derived", "kind": "derived", "source": f"local:{d}"}], "--merge-remote")
    assert rc == 2 and f"REFUSED: {KEY}__derived__{cell}: the shared manifest on cluster_b pins this identity to another object (cluster_b {'f' * 12}) while these bytes are {y[:12]} — two objects claim one pin (D1311); nothing copied" in out, out
    rec = json.loads((base / "bank" / "_records" / f"{KEY}__derived__{cell}.json").read_text())
    assert rec["status"] == "remote_manifest_conflict" and rec["object_sha256_found"] == y and rec["remote_conflicts"]["cluster_b"]["object_sha256"] == "f" * 64
    for lab in ("cluster_a", "cluster_b"):
        assert not (roots[lab] / REL / KEY / "cells" / f"derived__{cell}").exists(), lab                         # NOT copied to either cluster
    assert _read(roots["cluster_a"]) is None and _ids(_read(roots["cluster_b"])) == {(KEY, f"derived__{cell}"): "f" * 64}   # no manifest published anywhere
    # the bank's record for the cell is a refusal NOTE (no object_sha256) — the preflight no longer reads it as a
    # conflict (that refused the very rerun the refusal recommends), so this run proceeds past the preflight; the conflict the note recorded is
    # caught at the PUBLISH, by name, with this bank's bytes against cluster_a's pin — nothing published, the note untouched
    _remote(roots["cluster_a"], [_rec(KEY, f"derived__{cell}", sha="e" * 64)])
    (roots["cluster_b"] / REL / "staging_manifest.json").unlink()
    rc, out = _stage(base, site, [{"cell_id": "never_pulled", "lane": "derived", "kind": "derived", "source": f"local:{tmp_path / 'absent'}"}], "--merge-remote")
    assert rc == 2 and "REFUSED before any copy:" not in out and "MANIFEST REFUSED" in out and f"{KEY}/derived__{cell}: local object_sha256 {y} vs cluster_a {'e' * 64}" in out and "two objects claim one pin" in out, out
    rec = json.loads((base / "bank" / "_records" / f"{KEY}__derived__{cell}.json").read_text())
    assert rec["status"] == "remote_manifest_conflict" and rec["object_sha256_found"] == y                       # the note stands; the peer's pin was not folded over it
    assert _ids(_read(roots["cluster_a"])) == {(KEY, f"derived__{cell}"): "e" * 64} and _read(roots["cluster_b"]) is None   # nothing published anywhere
    # the preflight also refuses a publish over foreign records without --merge-remote before any copy
    _remote(roots["cluster_a"], [_rec("gemma4e4b", "base__x")]); (base / "bank" / "_records" / f"{KEY}__derived__{cell}.json").unlink()
    rc, out = _stage(base, site, [{"cell_id": "never_pulled", "lane": "derived", "kind": "derived", "source": f"local:{tmp_path / 'absent'}"}])
    assert rc == 2 and "REFUSED before any copy:" in out and "holds 1 record(s) this bank lacks (gemma4e4b/base__x)" in out and "PULL FAILED" not in out, out


def test_merge_remote_builds_one_union_from_every_cluster_and_publishes_the_same_set_to_each(tmp_path):
    """cluster_a holds A, cluster_b holds B, the bank holds L — one union {A, B, L} reaches BOTH clusters (the per-cluster loop
    used to publish L+A to the first and L+A+B to the second); without --merge-remote the first foreign record refuses and neither moves."""
    bank, records, manifest = _bank(tmp_path, [_rec(KEY, "base__cellA", sha="1" * 64)])
    ha, sa = _host(tmp_path, "cluster_a"); hb, sb = _host(tmp_path, "cluster_b")
    A = _rec("gemma4e4b", "base__x"); B = _rec("qwen35_4b", "derived__y", sha="b" * 64)
    _remote(sa, [A]); _remote(sb, [B])
    hosts = {"cluster_a": ha, "cluster_b": hb}; labels = ["cluster_a", "cluster_b"]
    with pytest.raises(SO.ForeignRecords, match=r"\[cluster_a\] .* holds 1 record\(s\) this bank lacks \(gemma4e4b/base__x\)"):
        SO.publish_union(hosts, labels, REL, manifest, records)
    assert _ids(_read(sa)) == _ids([A]) and _ids(_read(sb)) == _ids([B])                                  # neither moved
    union = SO.publish_union(hosts, labels, REL, manifest, records, merge_remote=True)
    want = {("gemma4e4b", "base__x"): "a" * 64, ("qwen35_4b", "derived__y"): "b" * 64, (KEY, "base__cellA"): "1" * 64}
    assert _ids(union) == want and _ids(_read(sa)) == want and _ids(_read(sb)) == want                   # the SAME set on both
    assert _read(sa) == _read(sb) == json.loads(manifest.read_text())
    assert sorted(p.name for p in records.glob("*.json")) == sorted(["gemma4e4b__base__x.json", f"{KEY}__base__cellA.json", "qwen35_4b__derived__y.json"])
    assert SO.publish_union(hosts, labels, REL, manifest, records) == union                                # a second publish is a no-op
    # end to end: the stager stages a new cell and publishes the union (with the new record) identically to both clusters
    base = tmp_path / "e2e"; base.mkdir(); site, roots = _site(base, labels)
    _remote(roots["cluster_a"], [A]); _remote(roots["cluster_b"], [B])
    cell = "z__psr_ce_all_minus_ctl"; d, z = _source(tmp_path, cell, b"fresh bytes Z " * 32)
    rc, out = _stage(base, site, [{"cell_id": cell, "lane": "derived", "kind": "derived", "source": f"local:{d}"}], "--merge-remote")
    assert rc == 0 and "published identically to cluster_a, cluster_b" in out, out
    want = {("gemma4e4b", "base__x"): "a" * 64, ("qwen35_4b", "derived__y"): "b" * 64, (KEY, f"derived__{cell}"): z}      # a fresh bank: A + B + the new cell
    assert _ids(_read(roots["cluster_a"])) == want and _read(roots["cluster_a"]) == _read(roots["cluster_b"])
    assert (roots["cluster_a"] / REL / KEY / "cells" / f"derived__{cell}" / "object.pt").is_file() and (roots["cluster_b"] / REL / KEY / "cells" / f"derived__{cell}" / "object.pt").is_file()


def test_a_peer_writing_between_the_read_and_the_publish_is_folded_by_the_retry_and_the_publish_is_a_rename(tmp_path, monkeypatch, capsys):
    """The publish compares-and-swaps on the remote digest read with the manifest — a peer's record written in between makes the
    first swap miss; the retry re-reads, folds the peer's record and publishes the union; temp file + rename, nothing partial left behind;
    a remote that keeps moving is refused after the bounded attempts, never published over."""
    bank, records, manifest = _bank(tmp_path, [_rec(KEY, "base__cellA", sha="1" * 64)])
    h, shared = _host(tmp_path)
    A = _rec("gemma4e4b", "base__x"); _remote(shared, [A])
    peer = _rec("llama8b", "base__peer", sha="7" * 64)
    real_run = SO.Host.run; state = {"fired": 0, "every": False}

    def racing_run(self, script, attempts=3):
        if "CAS-MISMATCH" in script and (state["every"] or state["fired"] == 0):
            state["fired"] += 1
            cur = _read(shared) or []
            cur.append(dict(peer, staged_id=f"base__peer{state['fired']}" if state["every"] else "base__peer", cell_id=f"peer{state['fired']}" if state["every"] else "peer"))
            (shared / REL / "staging_manifest.json").write_text(json.dumps(cur))                      # a peer publishes between our read and our swap
        return real_run(self, script, attempts)
    monkeypatch.setattr(SO.Host, "run", racing_run)
    union = SO.publish_union({"cluster_a": h}, ["cluster_a"], REL, manifest, records, merge_remote=True)
    out = capsys.readouterr().out
    assert "changed since it was read" in out and "changed under this publish (attempt 1 of 5)" in out and state["fired"] == 1
    want = {("gemma4e4b", "base__x"): "a" * 64, (KEY, "base__cellA"): "1" * 64, ("llama8b", "base__peer"): "7" * 64}
    assert _ids(union) == want and _ids(_read(shared)) == want and (records / "llama8b__base__peer.json").is_file()
    assert not list((shared / REL).glob(".staging_manifest.json.*")), list((shared / REL).iterdir())    # temp file renamed, none left
    state["every"] = True; state["fired"] = 0
    with pytest.raises(RuntimeError, match="kept changing under 5 publish attempts — refusing to publish over a moving file"):
        SO.publish_union({"cluster_a": h}, ["cluster_a"], REL, manifest, records, merge_remote=True)
    assert state["fired"] == 5 and not list((shared / REL).glob(".staging_manifest.json.*"))
    src = (REPO / "scripts/sdfbase/full22/stage_objects.py").read_text()
    assert 'mv -f "$tmp" "$f" && sha256sum "$f"' in src and 'echo "CAS-MISMATCH $cur"; exit 0' in src and "rm -f \"$tmp\"" in src
    assert 'flock -w {WRITER_LOCK_WAIT_S} 9' in src and SO.WRITER_LOCK_WAIT_S == 120 and src.index("flock -w") < src.index('cur={ABSENT}; [ -f "$f" ]') < src.index('mv -f "$tmp" "$f"')   # stager-fix-3: the writer lock spans check and rename
    assert "publish_union(hosts, labels, a.rel, manifest, records, merge_remote=a.merge_remote)" in src
    assert "remote0 = preflight_remote_manifests(hosts, labels, a.rel, object_records(records), merge_remote=a.merge_remote, publishing=not a.no_manifest)" in src   # stager-fix-4: object records only
    i_pre = src.index("remote0 = preflight_remote_manifests("); i_loop = src.index("    for cell in cells:"); i_clash = src.index("clash = live_identity_conflicts(hosts, labels, a.rel, a.target, staged_id, sha)"); i_stage = src.index("r = stage_to(hosts[lab], local_dir, rdir, sha, n)")
    assert i_pre < i_loop < i_clash < i_stage                                                           # preflight before the loop; the per-cell check before any copy


def test_an_absent_remote_pushes_plainly_and_an_unreadable_one_refuses(tmp_path):
    bank, records, manifest = _bank(tmp_path, [_rec(KEY, "base__cellA")])
    h, shared = _host(tmp_path)
    assert SO.push_manifest(h, REL, manifest, records=records) is True
    assert _read(shared) == json.loads(manifest.read_text())
    (shared / REL / "staging_manifest.json").write_text("{not json")
    with pytest.raises(RuntimeError, match="is not JSON"):
        SO.push_manifest(h, REL, manifest, records=records)
    (shared / REL / "staging_manifest.json").write_text('{"a": 1}')
    with pytest.raises(RuntimeError, match="is not a list"):
        SO.push_manifest(h, REL, manifest, records=records)
    with pytest.raises(RuntimeError, match="needs the bank's _records/ directory"):
        SO.push_manifest(h, REL, manifest)
    src = (REPO / "scripts/sdfbase/full22/stage_objects.py").read_text()
    assert 'ap.add_argument("--merge-remote", action="store_true"' in src
