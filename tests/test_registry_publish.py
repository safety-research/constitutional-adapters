"""Publish-implies-recorded: an unregistered file is refused by name; every
published file has an artifact row whose sha equals the file's; bytes that drift from the registered sha
are refused; a destination file without a record is refused. A path is registered through a row's primary
or its locations (a bare path or a location object's `uri`), the bytes at the path pick the registration
among every row naming it, and bytes matching none are refused naming them all (the closing passes); a
file already in the destination is kept only when the bytes now at it have a row, whatever rows name
its path.
"""
from __future__ import annotations

import hashlib
import json
import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import publish as PUB
from sisteer.registry import synth


def _setup(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=1, n_items=2)
    H.ingest_once(root, workers=0, min_free=0)
    src = tmp_path / "figures"
    src.mkdir()
    (src / "fig1.png").write_bytes(b"\x89PNG synthetic 1")
    (src / "fig2.pdf").write_bytes(b"%PDF synthetic 2")
    return root, str(src)


def test_unregistered_file_refused_by_name(tmp_path):
    root, src = _setup(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    with pytest.raises(PUB.Unregistered) as ei:
        PUB.publish_dir(conn, src, str(tmp_path / "dest"))
    assert "fig1.png" in str(ei.value) and "no artifact row" in str(ei.value)
    conn.close()


def test_registered_files_publish_with_sha_equality(tmp_path):
    root, src = _setup(tmp_path)
    files = sorted(os.path.join(src, f) for f in os.listdir(src))
    written = PUB.register_files(root, files, notebook_commit="c" * 40, now=synth.t(9000))
    assert len(written) == 2
    conn = DB.connect(DB.db_path(root), readonly=True)
    rep = PUB.publish_dir(conn, src, str(tmp_path / "dest"))
    assert sorted(os.path.basename(p) for p in rep.published) == ["fig1.png", "fig2.pdf"]
    for f in files:
        row = conn.execute("SELECT artifact_id, kind, extra_json FROM artifact WHERE uri_primary = ?", (os.path.abspath(f),)).fetchone()
        assert row[0] == PUB.sha256_of(f)
        assert row[1] == ("figure" if f.endswith((".png", ".pdf")) else "page")
        assert '"notebook_commit":"' + "c" * 40 in row[2]
    conn.close()


def test_drifted_bytes_refused(tmp_path):
    root, src = _setup(tmp_path)
    files = sorted(os.path.join(src, f) for f in os.listdir(src))
    PUB.register_files(root, files, now=synth.t(9000))
    with open(files[0], "ab") as fh:
        fh.write(b" changed after registration")
    conn = DB.connect(DB.db_path(root), readonly=True)
    with pytest.raises(PUB.Unregistered) as ei:
        PUB.publish_dir(conn, src, str(tmp_path / "dest"))
    assert "bytes differ" in str(ei.value) and os.path.basename(files[0]) in str(ei.value)
    conn.close()


def test_destination_file_without_record_refused(tmp_path):
    root, src = _setup(tmp_path)
    files = sorted(os.path.join(src, f) for f in os.listdir(src))
    PUB.register_files(root, files, now=synth.t(9000))
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "stray.png").write_bytes(b"never registered")
    conn = DB.connect(DB.db_path(root), readonly=True)
    with pytest.raises(PUB.Unregistered) as ei:
        PUB.publish_dir(conn, src, str(dest))
    assert "stray.png" in str(ei.value)
    conn.close()


def test_untracked_in_git_refused(tmp_path):
    import subprocess
    root, src = _setup(tmp_path)
    subprocess.run(["git", "init", "-q", str(tmp_path / "figures")], check=True)
    untracked = PUB.untracked_in_git([os.path.join(src, "fig1.png")], src)
    assert untracked
    subprocess.run(["git", "-C", src, "add", "fig1.png"], check=True)
    assert not PUB.untracked_in_git([os.path.join(src, "fig1.png")], src)


def test_a_path_registered_among_a_rows_locations_publishes_and_its_bytes_are_still_checked(tmp_path):
    """The closing pass of the analysis filing (2026-10-09): a known digest's later registration keeps the standing primary and adds its
    path to the row's locations (`merged_artifact_payload`), and `registered_sha` read the primary alone, so `publish_dir` refused the
    verified local copy of a file first declared from the cluster. A path is registered when a row names it as its primary OR among
    its locations; the bytes are checked against the sha either way."""
    import json
    root, src = _setup(tmp_path)
    fig1, fig2 = (os.path.join(src, f) for f in ("fig1.png", "fig2.pdf"))
    sha1 = PUB.sha256_of(fig1)
    remote = "/shared/someone/exp/figs/fig1.png"
    # fig1 declared from the cluster first (an unverified remote primary); fig2 registered from its local path
    PUB.register_files(root, [], now=synth.t(9000), payloads=[PUB.remote_artifact_payload(remote, sha1, synth.t(9000), substrate="cluster_a")])
    PUB.register_files(root, [fig2], now=synth.t(9001))
    cols = ("kind", "uri_primary", "locations_json", "bytes", "created_utc", "verified_utc", "extra_json")
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert PUB.registered_sha(conn, remote) == sha1 and PUB.registered_sha(conn, fig1) is None
    with pytest.raises(PUB.Unregistered) as ei:
        PUB.publish_dir(conn, src, str(tmp_path / "dest0"))
    assert "fig1.png" in str(ei.value) and "no artifact row" in str(ei.value)
    existing = dict(zip(cols, tuple(conn.execute(f"SELECT {', '.join(cols)} FROM artifact WHERE artifact_id = ?", (sha1,)).fetchone())))
    conn.close()
    # the local copy read and verified, merged over the standing row: the primary stands, the path joins the locations
    merged = PUB.merged_artifact_payload(existing, PUB.artifact_payload(fig1, None, synth.t(9002)))
    assert merged is not None and merged["uri_primary"] == remote and [l["uri"] for l in merged["locations_json"]] == [fig1]
    PUB.register_files(root, [], now=synth.t(9002), payloads=[merged])
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT uri_primary, locations_json FROM artifact WHERE artifact_id = ?", (sha1,)).fetchone()
    assert row[0] == remote and [l["uri"] for l in json.loads(row[1])] == [remote, fig1]
    assert PUB.registered_sha(conn, fig1) == sha1 and PUB.registered_sha(conn, os.path.join(src, "nowhere.png")) is None
    assert PUB.unregistered_in_dir(conn, src) == []
    rep = PUB.publish_dir(conn, src, str(tmp_path / "dest"))
    assert sorted(os.path.basename(p) for p in rep.published) == ["fig1.png", "fig2.pdf"]
    # the bytes are still checked against the registered sha, a location exactly as a primary
    with open(fig1, "ab") as fh:
        fh.write(b" drifted after registration")
    with pytest.raises(PUB.Unregistered) as ei:
        PUB.publish_dir(conn, src, str(tmp_path / "dest2"))
    assert "fig1.png" in str(ei.value) and "bytes differ" in str(ei.value)
    assert PUB.unregistered_in_dir(conn, src) == [fig1]
    conn.close()


def test_a_bare_path_location_entry_registers_its_path_and_never_breaks_the_scan_of_the_rows_naming_a_path(tmp_path):
    """The older producers write `locations_json` as bare paths (`[path]`: the hooks' completion and
    rows artifacts), and the locations scan ran `json_extract(entry, '$.uri')` over every entry -- which raises `malformed JSON` on a bare
    path -- so on a registry holding one such row, every lookup of a path no primary named crashed: publishing a file registered through
    a location object, or asking after an unknown path, raised instead of publishing or refusing by name. The entry's TYPE now picks the
    comparison: a bare path is compared as itself (and so registers its path), an object by its `uri`, anything else matches nothing."""
    root, src = _setup(tmp_path)
    fig1, fig2 = (os.path.join(src, f) for f in ("fig1.png", "fig2.pdf"))
    sha1, sha2 = PUB.sha256_of(fig1), PUB.sha256_of(fig2)
    now = synth.t(9000)
    # an older producer's row, its one location a bare path (the hooks' shape)
    completion = "/shared/someone/exp/run/completions.jsonl"
    legacy = {"artifact_id": "f" * 64, "kind": "completion", "uri_primary": completion, "bytes": 3, "created_utc": now,
              "locations_json": [completion], "extra": {"run_id": "synthetic"}}
    # fig2 registered from the cluster with its local path a further BARE-PATH location, a location object without a `uri` beside them;
    # fig1 declared from the cluster (a location object), its local copy merged in below
    remote1, remote2 = ("/shared/someone/exp/figs/" + f for f in ("fig1.png", "fig2.pdf"))
    bare = {"artifact_id": sha2, "kind": "figure", "uri_primary": remote2, "locations_json": [remote2, fig2, {"note": "no uri here"}],
            "extra": {"declared_sha256": sha2}}
    PUB.register_files(root, [], now=now, payloads=[legacy, bare, PUB.remote_artifact_payload(remote1, sha1, now, substrate="cluster_a")])
    cols = ("kind", "uri_primary", "locations_json", "bytes", "created_utc", "verified_utc", "extra_json")
    conn = DB.connect(DB.db_path(root), readonly=True)
    existing = dict(zip(cols, tuple(conn.execute(f"SELECT {', '.join(cols)} FROM artifact WHERE artifact_id = ?", (sha1,)).fetchone())))
    conn.close()
    merged = PUB.merged_artifact_payload(existing, PUB.artifact_payload(fig1, None, synth.t(9001)))
    assert merged is not None
    PUB.register_files(root, [], now=synth.t(9001), payloads=[merged])
    conn = DB.connect(DB.db_path(root), readonly=True)
    # before the fix every one of these lookups raised sqlite3.OperationalError (malformed JSON) on the legacy row's bare path
    assert PUB.registered_sha(conn, fig1) == sha1                                         # through its location object
    assert PUB.registered_sha(conn, fig2) == sha2                                         # through a bare-path location
    assert PUB.registered_shas(conn, completion) == ["f" * 64]                            # the older row: its primary and its bare path, one row
    assert PUB.registered_sha(conn, os.path.join(src, "nowhere.png")) is None
    assert PUB.registered_shas(conn, "no uri here") == [] and PUB.registered_shas(conn, remote1) == [sha1]
    assert PUB.unregistered_in_dir(conn, src) == []
    rep = PUB.publish_dir(conn, src, str(tmp_path / "dest"))
    assert sorted(os.path.basename(p) for p in rep.published) == ["fig1.png", "fig2.pdf"]
    conn.close()


def test_the_bytes_at_a_path_pick_its_registration_among_every_row_naming_it_so_an_older_primary_never_masks_a_newer_location(tmp_path):
    """Version A of a figure registered at local path P (P its row's primary); version B declared from
    the cluster, P replaced with B's bytes and the local copy filed, so B's row names P as a verified location while its remote primary
    stands. `registered_sha` read the newest PRIMARY first and returned A's digest for P, so `publish_dir` refused
    the correctly registered B bytes as drifted from A while `unregistered_in_dir` saw nothing wrong. Now the bytes at the path pick the
    registration among every row naming it (newest first), A's row stands as history, and bytes matching none are refused naming them all."""
    root, src = _setup(tmp_path)
    fig1, fig2 = (os.path.join(src, f) for f in ("fig1.png", "fig2.pdf"))
    with open(fig1, "rb") as fh:
        a_bytes = fh.read()
    sha_a = PUB.sha256_of(fig1)
    PUB.register_files(root, [fig1, fig2], now=synth.t(9000))                            # A at P, P its row's primary
    b_bytes = a_bytes + b", version B"
    sha_b = hashlib.sha256(b_bytes).hexdigest()
    remote = "/shared/someone/exp/figs/fig1.png"
    PUB.register_files(root, [], now=synth.t(9001), payloads=[PUB.remote_artifact_payload(remote, sha_b, synth.t(9001), substrate="cluster_a")])
    with open(fig1, "wb") as fh:
        fh.write(b_bytes)                                                                 # P now holds B
    cols = ("kind", "uri_primary", "locations_json", "bytes", "created_utc", "verified_utc", "extra_json")
    conn = DB.connect(DB.db_path(root), readonly=True)
    existing = dict(zip(cols, tuple(conn.execute(f"SELECT {', '.join(cols)} FROM artifact WHERE artifact_id = ?", (sha_b,)).fetchone())))
    conn.close()
    merged = PUB.merged_artifact_payload(existing, PUB.artifact_payload(fig1, None, synth.t(9002)))
    assert merged is not None and merged["uri_primary"] == remote                         # the local copy of B: a location of B's row, the primary standing
    PUB.register_files(root, [], now=synth.t(9002), payloads=[merged])
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert [l["uri"] for l in json.loads(conn.execute("SELECT locations_json FROM artifact WHERE artifact_id = ?", (sha_b,)).fetchone()[0])] == [remote, fig1]
    assert conn.execute("SELECT uri_primary FROM artifact WHERE artifact_id = ?", (sha_a,)).fetchone()[0] == fig1   # A's row stands as history
    # before the fix: refused as "bytes differ from the registered sha <A>" -- the newest primary read first -- while unregistered_in_dir saw nothing
    assert PUB.unregistered_in_dir(conn, src) == []
    rep = PUB.publish_dir(conn, src, str(tmp_path / "dest"))
    assert sorted(os.path.basename(p) for p in rep.published) == ["fig1.png", "fig2.pdf"]
    assert (tmp_path / "dest" / "fig1.png").read_bytes() == b_bytes
    # every registration at P, newest first: B through its location, then A through its primary
    assert PUB.registered_shas(conn, fig1) == [sha_b, sha_a] and PUB.registered_sha(conn, fig1) == sha_b
    assert PUB.registered_shas(conn, remote) == [sha_b] and PUB.registered_shas(conn, fig2) == [PUB.sha256_of(fig2)]
    # A's bytes back at P are registered too (A's row names P as its primary): history, not drift
    with open(fig1, "wb") as fh:
        fh.write(a_bytes)
    assert fig1 in PUB.publish_dir(conn, src, str(tmp_path / "dest_a")).published and PUB.unregistered_in_dir(conn, src) == []
    # bytes matching neither are refused, the refusal naming every registration newest first; the two readers agree
    with open(fig1, "ab") as fh:
        fh.write(b" drifted after both registrations")
    with pytest.raises(PUB.Unregistered) as ei:
        PUB.publish_dir(conn, src, str(tmp_path / "dest2"))
    msg = str(ei.value)
    assert "fig1.png" in msg and "bytes differ from every sha registered at the path" in msg and "fig2.pdf" not in msg
    assert sha_b[:12] in msg and sha_a[:12] in msg and msg.index(sha_b[:12]) < msg.index(sha_a[:12])
    assert PUB.unregistered_in_dir(conn, src) == [fig1]
    conn.close()


def test_a_destination_file_whose_path_a_row_names_is_refused_when_its_bytes_have_no_record_as_unregistered_in_dir_says(tmp_path):
    """The third closing pass (2026-10-09): `publish_dir` kept a file already in the destination when ANY row named its path
    (`registered_sha(conn, path) is not None`) or its digest had a row, so once a location entry named a destination path -- the local
    copy of a figure declared from the cluster, verified in the destination and merged into its row's locations -- the file there could
    drift to bytes nobody holds, and publishing another registered file beside it refused nothing while `unregistered_in_dir` named it.
    A destination path registered as a row's PRIMARY had the same gap. The bytes now at the path must have a row (`bytes_recorded`), the
    one rule both readers apply; the refusal names the registrations the bytes differ from."""
    import shutil
    root, src = _setup(tmp_path)
    fig1, fig2 = (os.path.join(src, f) for f in ("fig1.png", "fig2.pdf"))
    fig1_bytes = (tmp_path / "figures" / "fig1.png").read_bytes()
    sha1 = PUB.sha256_of(fig1)
    dest = tmp_path / "dest"
    dest.mkdir()
    dest_fig1, dest_fig3 = str(dest / "fig1.png"), str(dest / "fig3.svg")
    shutil.copy2(fig1, dest_fig1)
    os.remove(fig1)                                                                       # the source directory holds fig2 alone; fig1 lives in the destination
    (dest / "fig3.svg").write_bytes(b"<svg>synthetic 3</svg>")
    now = synth.t(9000)
    remote = "/shared/someone/exp/figs/fig1.png"
    # fig1 declared from the cluster (an unverified remote primary); fig2 registered from the source; fig3 registered AT its destination path (a primary)
    PUB.register_files(root, [], now=now, payloads=[PUB.remote_artifact_payload(remote, sha1, now, substrate="cluster_a"),
                                                     PUB.artifact_payload(fig2, None, now), PUB.artifact_payload(dest_fig3, None, now)])
    cols = ("kind", "uri_primary", "locations_json", "bytes", "created_utc", "verified_utc", "extra_json")
    conn = DB.connect(DB.db_path(root), readonly=True)
    existing = dict(zip(cols, tuple(conn.execute(f"SELECT {', '.join(cols)} FROM artifact WHERE artifact_id = ?", (sha1,)).fetchone())))
    conn.close()
    # the destination copy of fig1 read and verified, merged over the standing row: the remote primary stands, the destination path joins the locations
    merged = PUB.merged_artifact_payload(existing, PUB.artifact_payload(dest_fig1, None, synth.t(9001)))
    assert merged is not None and merged["uri_primary"] == remote and [l["uri"] for l in merged["locations_json"]] == [dest_fig1]
    PUB.register_files(root, [], now=synth.t(9001), payloads=[merged])
    conn = DB.connect(DB.db_path(root), readonly=True)
    sha3 = PUB.sha256_of(dest_fig3)
    assert PUB.registered_shas(conn, dest_fig1) == [sha1] and PUB.registered_shas(conn, dest_fig3) == [sha3]
    assert PUB.bytes_recorded(conn, dest_fig1) and PUB.bytes_recorded(conn, dest_fig3) and PUB.unregistered_in_dir(conn, str(dest)) == []
    # as registered, both destination files stand: publishing fig2 beside them refuses nothing
    rep = PUB.publish_dir(conn, src, str(dest))
    assert [os.path.basename(p) for p in rep.published] == ["fig2.pdf"] and rep.refused == []
    assert sorted(os.listdir(dest)) == ["fig1.png", "fig2.pdf", "fig3.svg"]
    # the destination copy of fig1 drifts to bytes no row holds while the row still names its path (a location)
    with open(dest_fig1, "ab") as fh:
        fh.write(b" drifted in the destination")
    assert conn.execute("SELECT 1 FROM artifact WHERE artifact_id = ?", (PUB.sha256_of(dest_fig1),)).fetchone() is None
    assert PUB.registered_shas(conn, dest_fig1) == [sha1] and not PUB.bytes_recorded(conn, dest_fig1)
    assert PUB.unregistered_in_dir(conn, str(dest)) == [dest_fig1]
    # before the fix: fig2 published again and nothing refused -- the row naming the path admitted the unregistered bytes
    with pytest.raises(PUB.Unregistered) as ei:
        PUB.publish_dir(conn, src, str(dest))
    msg = str(ei.value)
    assert dest_fig1 + " (present in the destination with bytes that differ from the registered sha " + sha1[:12] in msg
    assert "fig2.pdf" not in msg and "fig3.svg" not in msg
    assert (dest / "fig1.png").read_bytes().endswith(b" drifted in the destination")      # refused by name, never deleted
    # a destination path registered as a row's PRIMARY is judged the same way
    with open(dest_fig3, "ab") as fh:
        fh.write(b" drifted too")
    assert PUB.unregistered_in_dir(conn, str(dest)) == [dest_fig1, dest_fig3]
    with pytest.raises(PUB.Unregistered) as ei:
        PUB.publish_dir(conn, src, str(dest))
    msg = str(ei.value)
    assert dest_fig1 in msg and dest_fig3 in msg and sha1[:12] in msg and sha3[:12] in msg and "fig2.pdf" not in msg
    # a stray file no row names at all keeps its wording
    (dest / "stray.png").write_bytes(b"never registered")
    with pytest.raises(PUB.Unregistered) as ei:
        PUB.publish_dir(conn, src, str(dest))
    assert str(dest / "stray.png") + " (present in the destination with no artifact row)" in str(ei.value)
    # the registered bytes back at both paths are recorded again, through the rows naming them; the two readers agree
    (dest / "fig1.png").write_bytes(fig1_bytes)
    (dest / "fig3.svg").write_bytes(b"<svg>synthetic 3</svg>")
    (dest / "stray.png").unlink()
    assert PUB.unregistered_in_dir(conn, str(dest)) == []
    assert PUB.publish_dir(conn, src, str(dest)).refused == []
    conn.close()
