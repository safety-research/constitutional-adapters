"""Mirror fan-out: a second sync switches the destination's pointer to the NEWER
manifest; an ssh destination gets a non-interactive ssh, ``mkdir -p``, the remote verifier under
``PYTHONPATH=<remote_src>`` and switches only on a clean report (fake runner records the argv);
``verify_destination`` covers every kind; ``sync_all`` writes acks and the state file. The Modal destination
-- its chunked push, ledger, in-flight marker, probes, residual list and volume-side verifier -- was removed
with its transport: a destination of that kind is now refused by name at load
(``config.check_destinations``), raised by name by ``sync`` and an error row of ``verify_destination``.
"""
from __future__ import annotations

import json
import os
import subprocess

import pytest

from sisteer.registry import cli
from sisteer.registry import config as CFG
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import mirrors as M
from sisteer.registry import records as R
from sisteer.registry import synth

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


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


def test_second_sync_switches_pointer_to_the_newer_manifest(tmp_path):
    root = _setup(tmp_path)
    dest = M.Destination("cluster_a", "local", str(tmp_path / "mirror"))
    r1 = M.sync(root, dest, _seq(root))
    assert r1.switched and M.current_manifest(dest.path) == r1.manifest
    _more(root, 1)
    r2 = M.sync(root, dest, _seq(root))
    assert r2.switched and r2.manifest > r1.manifest
    assert M.current_manifest(dest.path) == r2.manifest
    assert os.path.exists(os.path.join(dest.path, M.MANIFEST_DIR, r1.manifest)), "manifests are immutable; the old one stays"
    assert r2.copied == 2   # only the new file and the cycle's arrival-fact file moved


class _Recorder:
    def __init__(self, respond):
        self.argv = []
        self.respond = respond

    def __call__(self, argv, timeout_s, **kw):          # the runner shape: (argv, timeout_s, input=, check=)
        self.argv.append(list(argv))
        try:
            return subprocess.CompletedProcess(list(argv), 0, stdout=self.respond(list(argv)), stderr="")
        except _Fail:
            return subprocess.CompletedProcess(list(argv), 1, stdout="", stderr="simulated failure")


def test_ssh_sync_argv_and_switch_only_on_clean_verify(tmp_path):
    root = _setup(tmp_path)
    dest = M.Destination("cluster_b", "ssh", "/workspace/u/registry/log", "cluster_b", remote_src="/workspace/u/repos/constitutional-adapters/src")
    n = len(M.log_files(root))
    rec = _Recorder(lambda argv: json.dumps({"ok": n, "missing": [], "corrupt": [], "bytes": 1}) if "verify" in " ".join(argv) else "")
    res = M.sync(root, dest, _seq(root), runner=rec)
    assert res.switched and res.verify.ok == n
    cmds = [" ".join(a) for a in rec.argv]
    assert cmds[0] == ("ssh -o BatchMode=yes -o ConnectTimeout=15 cluster_b mkdir -p /workspace/u/registry/log/manifests "
                       "/workspace/u/registry/log/records /workspace/u/registry/log/rulings")
    # one ssh reads the destination's pointer BEFORE the copy (the base of the copy's delta and of the incremental verify); this
    # fake has none, so the copy is the whole tree (records/, rulings/) and the verify is full
    assert cmds[1].startswith("ssh -o BatchMode=yes -o ConnectTimeout=15 cluster_b b=$(cat /workspace/u/registry/log/manifests/CHECKSUMS.CURRENT")
    assert rec.argv[2][:7] == ["rsync", "-rt", f"--partial-dir={M.PARTIAL_DIR}", "--ignore-existing", "--timeout=120", "-e", M.RSYNC_SSH]
    assert rec.argv[2][-1] == "cluster_b:/workspace/u/registry/log/records/" and "--files-from=-" not in rec.argv[2]
    assert rec.argv[3][-1] == "cluster_b:/workspace/u/registry/log/rulings/"
    assert rec.argv[4][-1] == "cluster_b:/workspace/u/registry/log/manifests/"
    assert res.copy_mode == "full" and res.copy_since is None and res.copy_files is None and res.copied == -1
    assert cmds[5].endswith(f"env PYTHONPATH=/workspace/u/repos/constitutional-adapters/src python3 -S -m sisteer.registry.mirrors verify "
                            f"/workspace/u/registry/log /workspace/u/registry/log/manifests/{res.manifest} --json")
    assert "srun" not in cmds[5], "a hand-built Destination keeps the login-node path"
    assert "CHECKSUMS.CURRENT.tmp" in cmds[6] and "mv -f" in cmds[6]
    # a remote report with a missing file: no switch, no mv
    rec2 = _Recorder(lambda argv: json.dumps({"ok": n - 1, "missing": ["records/x"], "corrupt": []}) if "verify" in " ".join(argv) else "")
    res2 = M.sync(root, dest, _seq(root), runner=rec2)
    assert not res2.switched and res2.verify.missing == ["records/x"]
    assert not any("CHECKSUMS.CURRENT.tmp" in " ".join(a) for a in rec2.argv)


def test_verify_destination_per_kind(tmp_path, monkeypatch):
    root = _setup(tmp_path, n_runs=1)
    local = M.Destination("cluster_a", "local", str(tmp_path / "mirror"))
    M.sync(root, local, _seq(root))
    v = M.verify_destination(root, local)
    assert v["lag_files"] == 0 and v["unsynced_local_files"] == 0 and v["remote_current"] == M.latest_manifest(root)
    _more(root, 1)
    v2 = M.verify_destination(root, local)
    assert v2["unsynced_local_files"] == 2 and v2["lag_files"] == 0     # newer local files (the file + its arrival fact) are not lag
    n = len(M.read_manifest(os.path.join(root, M.MANIFEST_DIR, M.latest_manifest(root))))
    ssh = M.Destination("cluster_b", "ssh", "/workspace/u/registry/log", "cluster_b", remote_src="/workspace/u/repos/constitutional-adapters/src")
    rec = _Recorder(lambda argv: (json.dumps({"ok": n - 2, "missing": ["a", "b"], "corrupt": []}) if "verify" in " ".join(argv)
                                  else ("CHECKSUMS-000000000001.sha256\n" if "cat" in " ".join(argv) else "")))
    vs = M.verify_destination(root, ssh, runner=rec)
    assert vs["lag_files"] == 2 and vs["missing"] == ["a", "b"] and vs["remote_current"] == "CHECKSUMS-000000000001.sha256"
    assert any("--no-prune" in " ".join(a) for a in rec.argv), "verify never prunes on the destination"
    # the removed kind (C10): an error row by name, nothing run
    legacy = M.Destination("legacy", "modal", "/registry/log")
    calls = []
    vl = M.verify_destination(root, legacy, runner=lambda argv, t: calls.append(list(argv)))
    assert vl["error"] and "unknown destination kind 'modal'" in vl["error"] and "removed" in vl["error"] and calls == []


def test_sync_all_writes_acks_and_state_and_cli_verify(tmp_path, monkeypatch, capsys):
    root = _setup(tmp_path, n_runs=1)
    dest_path = str(tmp_path / "mirror")
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump({"mirrors": [{"name": "cluster_a", "kind": "local", "path": dest_path}]}, fh)
    rows = M.sync_all(root, _seq(root))
    assert rows[0]["switched"] and os.path.exists(rows[0]["ack_record"])
    st = M.read_state(root)
    assert st["destinations"][0]["destination"] == "cluster_a" and st["destinations"][0]["switched"]
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT destination, manifest_name FROM mirror_ack").fetchone()[1] == rows[0]["manifest"]
    conn.close()
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["mirrors", "verify"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["lag_files"] == 0
    assert cli.main(["mirrors", "sync"]) == 0
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["switched"]
    stamp = json.load(open(os.path.join(root, "state", "last_run", "mirrors-sync.json")))
    assert stamp["rc"] == 0


def test_the_removed_destination_kind_is_refused_by_name_everywhere(tmp_path, monkeypatch, capsys):
    """Fail-closed: a `kind: modal` entry in a legacy config.json is refused by name when the
    destinations are loaded (`config.check_destinations` through `harvest.load_config` / `mirrors.destinations`); a Destination
    of that kind handed to `sync` raises by name and runs nothing; `mirrors sync` from the CLI reports the refusal on stderr with
    exit 1 and no traceback; `--kind modal` is not a choice any more; the Destination shape carries none of the removed knobs."""
    root = _setup(tmp_path, n_runs=1)
    legacy = {"name": "legacy", "kind": "modal", "volume": "v", "path": "/registry/log", "python": "/p", "app": "/a"}
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump({"mirrors": [{"name": "cluster_a", "kind": "local", "path": str(tmp_path / "m")}, legacy]}, fh)
    with pytest.raises(CFG.ConfigError) as ex:
        M.destinations(root)
    msg = str(ex.value)
    assert "config.json" in msg and "mirrors[1]" in msg and "'legacy'" in msg and "'modal'" in msg and "removed" in msg
    with pytest.raises(CFG.ConfigError):
        H.load_config(root)
    calls = []
    with pytest.raises(ValueError) as ex2:
        M.sync(root, M.Destination("legacy", "modal", "/registry/log"), _seq(root), runner=lambda argv, t: calls.append(list(argv)))
    assert "unknown destination kind 'modal'" in str(ex2.value) and calls == []
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["mirrors", "sync"]) == 1
    err = capsys.readouterr().err
    assert "registry mirrors: configuration REFUSED:" in err and "mirrors[1]" in err and "Traceback" not in err
    with pytest.raises(SystemExit) as ex3:
        cli.main(["mirrors", "sync", "--kind", "modal"])
    assert ex3.value.code == 2
    for flag in (["--reconcile"], ["--reconcile-batch", "5"], ["--modal-app", "/x"]):
        with pytest.raises(SystemExit) as ex4:
            cli.main(["mirrors", "sync"] + flag)
        assert ex4.value.code == 2
    with pytest.raises(SystemExit) as ex5:
        cli.main(["mirrors", "prune-incoming"])
    assert ex5.value.code == 2
    d = M.Destination.from_config({"name": "cluster_a", "kind": "ssh", "host": "h", "path": "/p", "volume": "ignored", "app": "ignored", "chunk_files": 9})
    assert not hasattr(d, "volume") and not hasattr(d, "app") and not hasattr(d, "chunk_files") and not hasattr(d, "reconcile_batch_files")
    assert set(d.as_dict()) == {"name", "kind", "path", "host", "remote_src", "environment", "verify_sample", "slurm", "qos", "partition",
                                "verify_cpus", "verify_mem"}
    for name in ("plan_chunks", "prune_incoming", "read_modal_ledger", "write_residual_list", "next_reconcile_batch", "CHUNK_FILES",
                 "RECONCILE_BATCH_FILES", "INCOMING_PREFIX", "PROBE_PREFIX"):
        assert not hasattr(M, name), name
    # a clean config.json still loads, and sync_all's row carries no Modal-only field
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump({"mirrors": [{"name": "cluster_a", "kind": "local", "path": str(tmp_path / "m")}]}, fh)
    rows = M.sync_all(root, _seq(root))
    assert rows[0]["switched"] and "learned" not in rows[0] and rows[0]["kind"] == "local"
