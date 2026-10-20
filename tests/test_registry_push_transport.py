"""The push transport: ``push_records.sh`` run against a fake ``rsync`` on PATH — the 30 s
debounce coalesces N writes into ONE push (``--final`` bypasses it), the rsync destination is
``<host>/<campaign>/<run_dir>/`` relative to the rrsync root, the key is staged 0600 when the spool reports
other modes, a transient failure is retried with jitter and the script ALWAYS exits 0, ``--probe`` writes
the probe file; ``install_cluster.sh`` lays the cluster out and prints the ``authorized_keys`` line (and
installs no key); ``inbox_to_log.sh`` / ``registry inbox-merge`` move pushed files into the merged log.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from sisteer.registry import cli, emit
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth

REPO = Path(__file__).resolve().parents[1]
PUSH = REPO / "scripts" / "registry" / "push_records.sh"
INSTALL = REPO / "scripts" / "registry" / "install_cluster.sh"
INBOX = REPO / "scripts" / "registry" / "inbox_to_log.sh"


def _fake_rsync(tmp_path, fail_times=0, capture_dest=None):
    """A fake rsync: logs its argv, copies SRC/ into <capture_dest>/<REL> when given, fails the first N calls."""
    bindir = tmp_path / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    log = tmp_path / "rsync_calls.log"
    counter = tmp_path / "rsync_fail_counter"
    counter.write_text(str(fail_times))
    dest = capture_dest or (tmp_path / "grading_host_inbox")
    dest.mkdir(parents=True, exist_ok=True)
    (bindir / "rsync").write_text(f"""#!/bin/sh
echo "$@" >> {log}
n=$(cat {counter}); if [ "$n" -gt 0 ]; then echo $((n-1)) > {counter}; echo "rsync: connection refused (synthetic)" >&2; exit 255; fi
# last two args: SRC/ and DEST:REL
for a in "$@"; do prev2="$prev1"; prev1="$a"; done
rel="${{prev1#*:}}"
mkdir -p "{dest}/$rel" && cp -r "$prev2"* "{dest}/$rel" 2>/dev/null
exit 0
""")
    (bindir / "rsync").chmod(0o755)
    return bindir, log, dest


def _cluster(tmp_path, host="cluster_a", key_mode=0o600):
    shared = tmp_path / "shared"
    (shared / "registry" / "records").mkdir(parents=True)
    (shared / ".ssh").mkdir(parents=True)
    (shared / ".ssh" / "registry-push").write_text("SYNTHETIC-NOT-A-KEY\n")
    (shared / ".ssh" / "registry-push").chmod(key_mode)
    (shared / ".ssh" / "known_hosts").write_text("")
    (shared / "registry" / "DEST_HOST").write_text("nobody-synthetic@grading_host.invalid\n")
    return shared


def _env(bindir, shared, host="cluster_a"):
    return {"PATH": f"{bindir}:{os.environ.get('PATH', '')}", "HOME": str(shared.parent / "home"), "USER": "nobody-synthetic",
            "REGISTRY_SHARED_ROOT": str(shared), "REGISTRY_HOST_TAG": host, "REGISTRY_PUSH_ATTEMPTS": "3"}


def _run_dir(shared, n_records=5):
    w = emit.RecordWriter(str(shared / "registry" / "records"), "camp", "camp:ev:abcdefabcdef:0123456789abcdef", synth.producer(),
                          push_script=str(PUSH), disabled=False)
    for i in range(n_records):
        w.emit_state("running" if i == 0 else "generated", "job_log")
    return w


def test_debounce_coalesces_n_writes_into_one_push_and_final_bypasses(tmp_path):
    bindir, log, dest = _fake_rsync(tmp_path)
    shared = _cluster(tmp_path)
    w = _run_dir(shared, n_records=0)
    env = _env(bindir, shared)
    # the emitter calls the push after every write; the script's stamp makes N calls one rsync
    calls = 0
    for i in range(6):
        w.emit_state("running", "job_log", note=f"write {i}")
        rc = subprocess.run([str(PUSH), w.dir], capture_output=True, text=True, env=env, timeout=120).returncode
        assert rc == 0
        calls += 1
    assert len(log.read_text().strip().splitlines()) == 1, log.read_text()
    # --final pushes again past the debounce
    assert subprocess.run([str(PUSH), w.dir, "--final"], capture_output=True, text=True, env=env, timeout=120).returncode == 0
    lines = log.read_text().strip().splitlines()
    assert len(lines) == 2
    # destination = <host>/<campaign>/<run_dir>/ relative to the rrsync root; the key's ssh options pinned
    assert lines[-1].endswith(f"{w.dir}/ nobody-synthetic@grading_host.invalid:cluster_a/camp/{os.path.basename(w.dir)}/")
    assert "-o IdentitiesOnly=yes -o BatchMode=yes" in lines[-1] and "--ignore-existing" in lines[-1] and "--exclude=.last_push" in lines[-1]
    # --mkpath: a new campaign's <host>/<campaign>/<run_dir>/ has two missing parents in the inbox; without it every real
    # push failed at the receiver's mkdir while the one-level `_probe` passed (live drill, 2026-09-17)
    assert "--mkpath" in lines[-1]
    # the records arrived (fake rsync copies), the stamp never travels
    landed = sorted(os.listdir(dest / "cluster_a" / "camp" / os.path.basename(w.dir)))
    assert len([f for f in landed if f.endswith(".jsonl")]) == 6 and ".last_push" not in landed
    assert (Path(w.dir) / ".last_push").exists()
    # the emitter's own debounce: RecordWriter.push() honours the stamp too
    assert w.push() == 0 and len(log.read_text().strip().splitlines()) == 2


def test_transient_failure_is_retried_with_jitter_and_the_script_always_exits_0(tmp_path):
    bindir, log, dest = _fake_rsync(tmp_path, fail_times=2)
    shared = _cluster(tmp_path)
    w = _run_dir(shared, n_records=2)
    env = _env(bindir, shared)
    t0 = time.time()
    out = subprocess.run([str(PUSH), w.dir, "--final"], capture_output=True, text=True, env=env, timeout=180)
    assert out.returncode == 0
    assert len(log.read_text().strip().splitlines()) == 3      # two failures, one success
    assert time.time() - t0 >= 4.0                           # two jittered pauses of 2–6 s each
    pushlog = (shared / "registry" / "push.log").read_text()
    assert "attempt 1/3 failed" in pushlog and "attempt 2/3 failed" in pushlog and "retrying in" in pushlog
    assert (Path(w.dir) / ".last_push").exists()
    # a push that never succeeds: still exit 0, logged, no stamp (so the next write retries)
    bindir2, log2, _ = _fake_rsync(tmp_path / "b", fail_times=99)
    shared2 = _cluster(tmp_path / "b")
    w2 = _run_dir(shared2, n_records=1)
    env2 = _env(bindir2, shared2)
    env2["REGISTRY_PUSH_ATTEMPTS"] = "2"
    out = subprocess.run([str(PUSH), w2.dir, "--final"], capture_output=True, text=True, env=env2, timeout=180)
    assert out.returncode == 0 and "failed after 2 attempts" in (shared2 / "registry" / "push.log").read_text()
    assert not (Path(w2.dir) / ".last_push").exists()


def test_missing_key_or_dest_host_never_fails_and_cluster_b_key_is_staged(tmp_path):
    bindir, log, _ = _fake_rsync(tmp_path)
    shared = _cluster(tmp_path)
    w = _run_dir(shared, n_records=1)
    env = _env(bindir, shared)
    (shared / "registry" / "DEST_HOST").unlink()
    assert subprocess.run([str(PUSH), w.dir, "--final"], capture_output=True, text=True, env=env, timeout=120).returncode == 0
    assert "no " in (shared / "registry" / "push.log").read_text() and not log.exists()
    # a 0666 key (cluster B's MooseFS) is staged to a 0600 temp copy for the push
    shared2 = _cluster(tmp_path / "c", key_mode=0o666)
    w2 = _run_dir(shared2, n_records=1)
    env2 = _env(bindir, shared2, host="cluster_b")
    out = subprocess.run([str(PUSH), w2.dir, "--final"], capture_output=True, text=True, env=env2, timeout=120)
    assert out.returncode == 0
    line = log.read_text().strip().splitlines()[-1]
    assert "registry-push." in line and str(shared2 / ".ssh" / "registry-push") not in line     # the staged copy, not the 0666 file
    assert ":cluster_b/camp/" in line
    # no shared root at all: exit 0 quietly
    out = subprocess.run([str(PUSH), w2.dir], capture_output=True, text=True,
                         env={"PATH": env2["PATH"], "HOME": env2["HOME"], "USER": "nobody-synthetic"}, timeout=120)
    assert out.returncode == 0 and "nothing pushed" in out.stderr


def test_probe_writes_the_probe_file(tmp_path):
    bindir, log, dest = _fake_rsync(tmp_path)
    shared = _cluster(tmp_path)
    out = subprocess.run([str(PUSH), "--probe"], capture_output=True, text=True, env=_env(bindir, shared), timeout=120)
    assert out.returncode == 0, out.stderr
    probes = list((dest / "cluster_a" / "_probe").glob("*.jsonl"))
    assert len(probes) == 1 and json.loads(probes[0].read_text())["host"] == "cluster_a"


#: RFC 5737 documentation ranges standing in for a cluster's egress ranges (REGISTRY_PUSH_FROM is site config)
EXAMPLE_PUSH_FROM = "192.0.2.0/24,198.51.100.0/24"


def _install_env(tmp_path, shared, **extra):
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path / "home"), "USER": "nobody-synthetic",
           "REGISTRY_SHARED_ROOT": str(shared), "REGISTRY_HOST_TAG": "cluster_a", "REGISTRY_GRADING_HOST_USER": "gradinguser"}
    env.update(extra)
    return {k: v for k, v in env.items() if v is not None}


def _key_line(stdout):
    return [l for l in stdout.splitlines() if l.startswith('command="/usr/bin/rrsync')][0]


def test_install_cluster_lays_out_the_spool_and_prints_the_key_line_without_installing_a_key(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    env = _install_env(tmp_path, shared, REGISTRY_PUSH_FROM=EXAMPLE_PUSH_FROM)
    out = subprocess.run([str(INSTALL), "--dest-host", "gradinguser@grading-host.invalid", "--print-key-line"], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    for d in ("records", "spool", "pending", "snapshot", "log"):
        assert (shared / "registry" / d).is_dir(), d
    for f in ("push_records.sh", "registry_env.sh", "sbatch_registered.sh"):
        assert (shared / "scripts" / "registry" / f).exists() and os.access(shared / "scripts" / "registry" / f, os.X_OK)
    assert (shared / "registry" / "DEST_HOST").read_text().strip() == "gradinguser@grading-host.invalid"
    assert not (shared / ".ssh" / "registry-push").exists() and "key       MISSING" in out.stdout
    line = _key_line(out.stdout)
    # the from= list is exactly the site's REGISTRY_PUSH_FROM: the script carries no address range of its own
    assert line.startswith('command="/usr/bin/rrsync -wo -no-del -no-lock /home/gradinguser/registry/inbox",restrict,'
                           f'from="{EXAMPLE_PUSH_FROM}" ssh-ed25519')
    assert line.endswith(" registry-push")
    assert "inbox/cluster_a/_probe/" in out.stdout
    # idempotent
    assert subprocess.run([str(INSTALL)], capture_output=True, text=True, env=env, timeout=120).returncode == 0


def test_install_cluster_without_push_from_prints_no_from_restriction(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    out = subprocess.run([str(INSTALL), "--print-key-line"], capture_output=True, text=True, env=_install_env(tmp_path, shared), timeout=120)
    assert out.returncode == 0, out.stderr
    line = _key_line(out.stdout)
    assert "from=" not in line and line.startswith('command="/usr/bin/rrsync -wo -no-del -no-lock /home/gradinguser/registry/inbox",restrict ssh-ed25519')
    assert "set REGISTRY_PUSH_FROM" in out.stdout                          # the operator is told how to pin the key


def test_install_cluster_reads_the_shared_root_from_the_site_cluster_block(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    env = _install_env(tmp_path, shared, REGISTRY_SHARED_ROOT=None, CLUSTER_SHARED_ROOT=str(shared))
    out = subprocess.run([str(INSTALL)], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    assert (shared / "registry" / "records").is_dir() and f"shared={shared}" in out.stdout


def test_install_cluster_writes_the_shared_root_for_registry_env_and_never_overwrites_a_site_file(tmp_path):
    """registry_env.sh reads the cluster's values from registry_site.env beside it and probes nothing: a cluster set up
    by this script alone gets the shared root written there (quoted, so a path with a space reads back whole), and a
    full file the front door's setup already rendered is left byte for byte."""
    shared = tmp_path / "shared root"
    shared.mkdir()
    site_env = shared / "scripts" / "registry" / "registry_site.env"
    out = subprocess.run([str(INSTALL)], capture_output=True, text=True, env=_install_env(tmp_path, shared), timeout=120)
    assert out.returncode == 0, out.stderr
    assert site_env.read_text().startswith("REGISTRY_SITE_SHARED_ROOT=") and f"site      {site_env}" in out.stdout
    read_back = subprocess.run(["bash", "-c", 'source "$1"; printf %s "$REGISTRY_SITE_SHARED_ROOT"', "_", str(site_env)],
                               capture_output=True, text=True, timeout=60)
    assert read_back.returncode == 0 and read_back.stdout == str(shared), (read_back.stdout, read_back.stderr)
    rendered = ("REGISTRY_SITE_SHARED_ROOT=/elsewhere/shared\nREGISTRY_SITE_SUBSTRATE=cluster_a\n"
                "REGISTRY_SITE_CLUSTER_LABEL=cluster_a\n")
    site_env.write_text(rendered)
    out = subprocess.run([str(INSTALL)], capture_output=True, text=True, env=_install_env(tmp_path, shared), timeout=120)
    assert out.returncode == 0, out.stderr
    assert site_env.read_text() == rendered


def test_install_cluster_refuses_without_a_shared_root_or_with_a_malformed_from_list(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    out = subprocess.run([str(INSTALL)], capture_output=True, text=True, env=_install_env(tmp_path, shared, REGISTRY_SHARED_ROOT=None), timeout=120)
    assert out.returncode == 2 and "REGISTRY_SHARED_ROOT" in out.stderr and "CLUSTER_SHARED_ROOT" in out.stderr, out.stderr
    out = subprocess.run([str(INSTALL)], capture_output=True, text=True, env=_install_env(tmp_path, tmp_path / "absent"), timeout=120)
    assert out.returncode == 2 and "not a directory" in out.stderr, out.stderr
    for bad in ("192.0.2.0/24, 198.51.100.0/24", '192.0.2.0/24" ssh-ed25519 x'):
        out = subprocess.run([str(INSTALL), "--print-key-line"], capture_output=True, text=True,
                             env=_install_env(tmp_path, shared, REGISTRY_PUSH_FROM=bad), timeout=120)
        assert out.returncode == 2 and "REGISTRY_PUSH_FROM" in out.stderr and "command=" not in out.stdout, (bad, out.stdout, out.stderr)


def test_inbox_to_log_moves_pushed_files_into_the_merged_log(tmp_path, monkeypatch, capsys):
    root = tmp_path / "reg"
    H.ensure_layout(str(root))
    rid = "camp:ev:abcdefabcdef:0123456789abcdef"
    rec = synth.rec("run_state", {"run_id": rid, "state": "running", "event_utc": synth.t(1), "evidence_kind": "job_log"}, synth.t(1))
    inbox_dir = root / "inbox" / "cluster_a" / "camp" / "camp_ev_abcdefabcdef_0123456789abcdef"
    R.write_records(str(inbox_dir / "000001-run_state.jsonl"), [rec])
    R.write_records(str(inbox_dir / "000002-run_state.jsonl"), [rec])
    # a same-named file already in the log with DIFFERENT content → kept as a conflict, never overwritten
    other = synth.rec("run_state", {"run_id": rid, "state": "generated", "event_utc": synth.t(2), "evidence_kind": "job_log"}, synth.t(2))
    R.write_records(str(root / "records" / "camp" / "camp_ev_abcdefabcdef_0123456789abcdef" / "000002-run_state.jsonl"), [other])
    monkeypatch.setenv("REGISTRY_ROOT", str(root))
    assert cli.main(["inbox-merge"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["merged"] == 1 and out["conflict"] == 1 and out["duplicate"] == 0
    log_dir = root / "records" / "camp" / "camp_ev_abcdefabcdef_0123456789abcdef"
    names = sorted(os.listdir(log_dir))
    assert "000001-run_state.jsonl" in names and "000002-run_state.jsonl" in names and any(n.startswith("000002-run_state.jsonl.conflict-") for n in names)
    assert not list(inbox_dir.glob("*.jsonl"))
    # the shell wrapper runs the same merge
    R.write_records(str(inbox_dir / "000003-run_state.jsonl"), [rec])
    out = subprocess.run([str(INBOX), "--root", str(root)], capture_output=True, text=True, timeout=180,
                         env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout.strip().splitlines()[-1])["merged"] == 1 and (log_dir / "000003-run_state.jsonl").exists()


def test_probe_stays_in_the_inbox_and_never_reaches_the_log(tmp_path):
    """The probe (``push_records.sh --probe`` → ``inbox/<host>/_probe/<utc>.jsonl``) is proof of route, not a record.
    The live install's two probes were merged into ``records/_probe/`` and refused there forever (``unknown record_type
    None``, 2026-09-17): the merge now leaves every underscore-prefixed campaign dir in the inbox for a person to read,
    while the real pushed records beside it merge as before."""
    root = tmp_path / "reg"
    H.ensure_layout(str(root))
    probe = root / "inbox" / "cluster_a" / "_probe" / "20260917T102859Z.jsonl"
    probe.parent.mkdir(parents=True)
    probe.write_text('{"probe":"2026-09-17T10:28:59.000Z","host":"cluster_a"}\n')
    rid = "camp:ev:abcdefabcdef:0123456789abcdef"
    rec = synth.rec("run_state", {"run_id": rid, "state": "running", "event_utc": synth.t(1), "evidence_kind": "job_log"}, synth.t(1))
    real = root / "inbox" / "cluster_a" / "camp" / "camp_ev_abcdefabcdef_0123456789abcdef" / "000001-run_state.jsonl"
    R.write_records(str(real), [rec])
    moved = H.scan_inbox(str(root))
    assert [s for _, s in moved] == ["merged"] and moved[0][0].endswith("camp_ev_abcdefabcdef_0123456789abcdef/000001-run_state.jsonl")
    assert probe.exists() and not real.exists()
    assert not (root / "records" / "_probe").exists()
    rep = H.ingest_once(str(root), workers=0, min_free=0)
    assert rep.refused == 0 and probe.exists()


def test_push_dest_env_overrides_the_dest_host_file_for_one_call(tmp_path):
    """``REGISTRY_PUSH_DEST=user@host`` aims ONE push somewhere else (the offline drill points a
    producer at an unreachable address) without touching the cluster's ``DEST_HOST`` file; the file is untouched
    afterwards, and with the override set a missing file is not an error."""
    bindir, log, dest = _fake_rsync(tmp_path)
    shared = _cluster(tmp_path)
    w = _run_dir(shared, n_records=2)
    env = dict(_env(bindir, shared), REGISTRY_PUSH_DEST="nobody-synthetic@offline.invalid")
    out = subprocess.run([str(PUSH), w.dir, "--final"], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    line = log.read_text().strip().splitlines()[-1]
    assert line.endswith(f"{w.dir}/ nobody-synthetic@offline.invalid:cluster_a/camp/{os.path.basename(w.dir)}/")
    assert "grading_host.invalid" not in line
    assert (shared / "registry" / "DEST_HOST").read_text().strip() == "nobody-synthetic@grading_host.invalid"   # the file is not re-pointed
    # the override stands in for the file entirely
    (shared / "registry" / "DEST_HOST").unlink()
    out = subprocess.run([str(PUSH), w.dir, "--final"], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0 and len(log.read_text().strip().splitlines()) == 2
    # without the override the file is the destination again (restored: the retry after an outage goes to the real grading host)
    (shared / "registry" / "DEST_HOST").write_text("nobody-synthetic@grading_host.invalid\n")
    (Path(w.dir) / ".last_push").unlink(missing_ok=True)
    out = subprocess.run([str(PUSH), w.dir, "--final"], capture_output=True, text=True, env=_env(bindir, shared), timeout=120)
    assert out.returncode == 0 and log.read_text().strip().splitlines()[-1].endswith(":cluster_a/camp/" + os.path.basename(w.dir) + "/")
    assert "nobody-synthetic@grading_host.invalid:" in log.read_text().strip().splitlines()[-1]
