"""The serving probe's menu lives on ONE cluster and the search — its trials, its
table — may run on ANOTHER (03_hpo.sh probe_home). `--phase table --cluster <search cluster>` revalidates the local menu against the
probe record's HOME before any fetch, key or render: a probe record re-verdicted on cluster_a (a new result generation) refuses the
table on cluster_b by name — the mirror's menu is archived, nothing is fetched, keyed, rendered or published, the table's DONE marker
stands as it was — until `--phase probe --cluster cluster_a` re-fetches the current menu; the table then rebuilds under the new menu's
key. Before this, the table phase verified the SEARCH record's summaries and keyed the LOCAL menu without ever reading the probe record,
so a superseded menu could be skipped under the old key or published as current. And the search record on cluster_b may
carry an EARLIER probe's residue at the same layout (its own allowed_pairs.json, verdict, bookkeeping): the search fetch that follows the
revalidation must never deliver those over the revalidated menu — the table keeps cluster_a's menu byte for byte. Real stage, the probe
harness (fake ssh / rsync that execute the remote string on this host, the real status reader), two neutral clusters, no network."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, write_site  # noqa: E402
from test_front_door_probe_resume import HEAD, MODEL, OVERRIDES, REAL_RSYNC, SRV_ID, World, _scan, _stamp  # noqa: E402
from test_emit_addendum import DRAWN9, NEW6, _round1  # noqa: E402
from test_hpo_design_v2 import allowed_doc  # noqa: E402

RSYNC_BOTH_WAYS = '''#!/bin/bash
# fake rsync for a PUSH as well as a fetch: a `local:<fake dir>:<path>` on either end is this host's path
args=("$@"); n=${#args[@]}; src="${args[n-2]}"; dst="${args[n-1]}"
case "$src" in local:*) src="${src##*:}";; esac
case "$dst" in local:*) dst="${dst##*:}";; esac
unset 'args[n-1]' 'args[n-2]'
exec %s "${args[@]}" "$src" "$dst"
'''


@pytest.fixture
def worlds(tmp_path):
    """The probe harness with a SECOND cluster: the probe record lives on cluster_a (World's), the search record on cluster_b."""
    if REAL_RSYNC is None:
        pytest.skip("no rsync on this host")
    w = World(tmp_path)
    (w.bin / "rsync").write_text(RSYNC_BOTH_WAYS % REAL_RSYNC); (w.bin / "rsync").chmod(0o755)
    fake_b = tmp_path / "fake_b"; (fake_b / "jobs").mkdir(parents=True)
    shared_b = tmp_path / "shared_b"; (shared_b / "sisteer-eval-repo" / HEAD).mkdir(parents=True)
    write_site(tmp_path, clusters=("cluster_a", "cluster_b"), ssh={"cluster_a": f"local:{w.fake}", "cluster_b": f"local:{fake_b}"},
               shared={"cluster_a": str(w.shared), "cluster_b": str(shared_b)}, extra_config=f"SISTEER_RUN_OVERRIDES={OVERRIDES}\n")
    w.fake_b = fake_b; w.shared_b = shared_b
    w.search = shared_b / "exp" / "reproduce" / MODEL / "03_hpo" / "lora"          # the SEARCH record on cluster_b
    w.figs_b = shared_b / "exp" / "figs" / f"hpo_{MODEL}_lora"                      # where the table is published (cluster_b's figure dir)
    w.table_marker = tmp_path / "state" / MODEL / "03_hpo" / "DONE.lora.table"
    return w


def _run(w: World, *args: str) -> subprocess.CompletedProcess:
    argv = ["bash", str(REPRO / "03_hpo.sh"), "--model", MODEL, "--object-class", "lora", "--config", str(w.site / "config.env"), *args]
    return subprocess.run(argv, capture_output=True, text=True, env=w.env(PYTHONPATH=str(REPO / "src")), cwd=str(REPO), timeout=600)


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _lay_probe_record(w: World) -> str:
    """cluster_a's READY probe record whose menu is the 9-pair design menu, stamped with the record's generation -> the generation."""
    w.lay(done=7, complete=True, menu=True, verdict=True)
    doc = allowed_doc(DRAWN9, ())
    doc.update({"model_id": SRV_ID, "served_hf_id": SRV_ID, "revision": "0" * 40, "model_key": MODEL, "probe_job_id": "11111111",
                "written_utc": "2026-09-30T02:00:00Z", "status": "complete", "certifying": True})
    (w.record / "allowed_pairs.json").write_text(json.dumps(doc))
    return _stamp(w)


def _reverdict_probe_record(w: World) -> str:
    """The record moves on: a launch re-run, verdict + menu rewritten (same pairs, another job, a later write) and STAMPED anew."""
    (w.record / "launches" / "L07.json").write_text(json.dumps({"launch_id": "L07", "status": "passed", "host": "TESTHOST", "rerun": True}))
    for name in ("probe_verdict.json", "allowed_pairs.json"):
        d = json.loads((w.record / name).read_text()); d["written_utc"] = "2026-09-30T03:00:00Z"; d["probe_job_id"] = "22222222"
        (w.record / name).write_text(json.dumps(d))
    return _stamp(w)


def _lay_search(w: World, tmp_path: Path) -> None:
    """The round-1 search on cluster_b's record (16 summaries over the 9-pair design) and its design fit on the mirror (a local derivation)."""
    synth = tmp_path / "synth"; synth.mkdir()
    pay, _art, _rc = _round1(synth)
    w.search.mkdir(parents=True, exist_ok=True); w.state.mkdir(parents=True, exist_ok=True)
    for root in (w.search, w.state):                                                    # the design payload, as the trials phase pushed it
        (root / "r1_trials.json").write_text(json.dumps(pay, default=str))
    for d in sorted(synth.glob("r1_t*")):
        if d.is_dir():
            shutil.copytree(d, w.search / d.name)
    w.state.mkdir(parents=True, exist_ok=True)
    shutil.copy(synth / "r1_fit.json", w.state / "r1_fit.json")


def _reads_of(w: World, path: Path) -> list[str]:
    return [ln for ln in w.ssh_log.read_text().splitlines() if str(path) in ln]


def test_the_table_on_the_search_cluster_revalidates_the_menu_against_the_probe_records_home_before_keying_or_publishing(worlds, tmp_path):
    w = worlds
    gen1 = _lay_probe_record(w)
    r = _run(w, "--phase", "probe", "--cluster", "cluster_a"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "READY on cluster_a" in out and (w.state / "allowed_pairs.json").is_file(), out
    assert json.loads((w.state / "probe_fetch.json").read_text())["cluster"] == "cluster_a"
    _lay_search(w, tmp_path)
    # cluster_b's record carries an EARLIER probe's residue at the same layout — a menu over the same pairs from another job, a verdict, bookkeeping
    stale = allowed_doc(DRAWN9, ())
    stale.update({"model_id": SRV_ID, "served_hf_id": SRV_ID, "revision": "0" * 40, "model_key": MODEL, "probe_job_id": "00009999",
                  "written_utc": "2026-09-01T00:00:00Z", "status": "complete", "certifying": True})
    (w.search / "allowed_pairs.json").write_text(json.dumps(stale))
    (w.search / "probe_verdict.json").write_text(json.dumps({"version": 1, "probe_job_id": "00009999", "status": "complete", "written_utc": "2026-09-01T00:00:00Z"}))
    (w.search / "probe_progress.json").write_text("{}"); (w.search / "geometry_record.json").write_text("{}")
    (w.search / "launches").mkdir(); (w.search / "launches" / "L01.json").write_text("{}")
    # the mirror already holds cluster_a's probe-owned files (the probe phase's own fetch delivers the whole record): none may change
    owned0 = {p.relative_to(w.state): _sha(p) for p in [w.state / "allowed_pairs.json", w.state / "probe_verdict.json", w.state / "probe_progress.json",
                                                         *sorted((w.state / "launches").glob("*.json"))] if p.is_file()}
    assert {"allowed_pairs.json", "probe_verdict.json"} <= {str(k) for k in owned0} and not (w.state / "geometry_record.json").exists()
    # the menu is current: the table revalidates it against cluster_a's record, fetches cluster_b's TRIAL evidence only, verifies the inventory,
    # renders, publishes, marks — and cluster_a's menu / verdict on the mirror are untouched by the fetch
    r = _run(w, "--phase", "table", "--cluster", "cluster_b"); out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert {k: _sha(w.state / k) for k in owned0} == owned0, "the search fetch replaced probe-owned files on the mirror"
    assert json.loads((w.state / "allowed_pairs.json").read_text())["probe_job_id"] == "11111111"
    assert not (w.state / "geometry_record.json").exists(), "cluster_b's residue reached the mirror"   # cluster_a's record has none; cluster_b's residue does
    assert "CHANGED the probe-owned files" not in out, out
    menu0 = owned0[Path("allowed_pairs.json")]
    assert "the menu's record lives on cluster_a:" in out and "table runs on cluster_b" in out, out
    assert f"current result generation ({gen1[:12]}; record cluster_a:{w.record}) — table proceeds" in out, out
    assert f"the mirror is the record cluster_b:{w.search} — 16 trial summaries + 0 sidecars verified" in out, out
    assert (w.figs_b / "table.csv").is_file() and w.table_marker.is_file() and "set=" in w.table_marker.read_text(), out
    marker1 = w.table_marker.read_text(); figs1 = _sha(w.figs_b / "table.csv"); menu1 = _sha(w.state / "allowed_pairs.json")
    assert len(list(__import__("csv").DictReader(open(w.figs_b / "table.csv")))) == 16
    # the probe is RE-VERDICTED on cluster_a while the search cluster's record is unchanged
    gen2 = _reverdict_probe_record(w); assert gen2 != gen1
    w.ssh_log.write_text("")
    r = _run(w, "--phase", "table", "--cluster", "cluster_b"); out = r.stdout + r.stderr
    assert r.returncode == 2, out
    assert f"table: the local menu for {MODEL} was NOT the record's current result generation" in r.stderr, out
    assert "--phase probe --cluster cluster_a" in r.stderr and "archived under" in out and "nothing deleted" in out, out
    # refused BEFORE the search record was read, before the key, before the renderer: nothing published, the marker as it was
    assert not _reads_of(w, w.search) and "trial summaries verified" not in out, (w.ssh_log.read_text(), out)
    assert "rebuilding" not in out and "skipping" not in out and "hpo_table" not in out, out
    assert w.table_marker.read_text() == marker1 and _sha(w.figs_b / "table.csv") == figs1
    assert not (w.state / "allowed_pairs.json").exists() and (w.state / "superseded").is_dir()
    assert (w.state / "r1_fit.json").is_file() and sorted(p.name for p in w.state.glob("r1_t*") if p.is_dir()) == [f"r1_t{i:02d}" for i in range(16)]   # the search's own evidence stays
    # --phase probe where the record lives re-fetches the current menu; the table then REBUILDS under the new menu's key
    r = _run(w, "--phase", "probe", "--cluster", "cluster_a"); out = r.stdout + r.stderr
    assert r.returncode == 0 and json.loads((w.state / "probe_fetch.json").read_text())["generation"] == gen2, out
    assert _sha(w.state / "allowed_pairs.json") != menu1
    r = _run(w, "--phase", "table", "--cluster", "cluster_b"); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"current result generation ({gen2[:12]}" in out and "rebuilding" in out and "skipping" not in out, out
    marker2 = w.table_marker.read_text()
    assert marker2 != marker1 and marker2.split("set=")[1].strip() != marker1.split("set=")[1].strip()
    # and nothing changed since: the key stands, the table is skipped (one revalidation read of cluster_a's record)
    w.ssh_log.write_text("")
    r = _run(w, "--phase", "table", "--cluster", "cluster_b"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "skipping" in out and len([ln for ln in w.ssh_log.read_text().splitlines() if "--record" in ln]) == 1, out
    _scan(w, out)


from front_door_testlib import array_tasks, make_registry, set_job_state  # noqa: E402


def _record_files(root: Path) -> dict:
    return {p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob("*") if p.is_file()}


def test_the_addendum_relaunch_runs_the_real_funnel_proof_before_anything_moves(worlds, tmp_path):
    """`--phase addendum … SISTEER_RESUBMIT=1` must prove the bound round-4 job dead AND unregistered (the funnel's fenced
    decision, read-only) BEFORE the record's design and trial dirs move. Through the REAL funnel (fake sbatch/sacct/squeue, a real registry
    database): the first submission binds a job; the job dies; (a) a registry row filed under that job -> the proof REFUSES and every file on the
    record is byte-identical; (b) no row -> AUTHORIZED, the failed design superseded, a new one re-derived and re-submitted."""
    w = worlds
    _lay_probe_record(w)
    r = _run(w, "--phase", "probe", "--cluster", "cluster_a"); assert r.returncode == 0, r.stdout + r.stderr
    _lay_search(w, tmp_path)
    (w.tmp / "menu_overrides").mkdir(exist_ok=True)
    (w.tmp / "menu_overrides" / "D547.json").write_text(json.dumps({"ruling": "D547", "model": MODEL, "class": "lora", "pairs": [list(k) for k in NEW6]}))
    reg = make_registry(w.tmp / "registry")                                       # the registry the funnel's check 4 reads (empty)
    venv = w.shared_b / "envs" / "sdfbase-fit"; venv.mkdir(parents=True, exist_ok=True); (venv / "BUILD-COMPLETE").write_text("2026-01-01T00:00:00Z 0123456789abcdef\n")   # the funnel's two fields: <iso utc> <sha16 of freeze.txt>
    env = {"SISTEER_MENU_OVERRIDES_DIR": str(w.tmp / "menu_overrides")}
    (w.shared_b / "registry" / "records").mkdir(parents=True, exist_ok=True)       # the producer spool the attempt is submitted with: traversable and empty
    pairs = ",".join(f"{a}:{b}" for a, b in NEW6)
    def _add(**extra):
        return subprocess.run(["bash", str(REPRO / "03_hpo.sh"), "--model", MODEL, "--object-class", "lora", "--config", str(w.site / "config.env"), "--phase", "addendum",
                               "--cluster", "cluster_b", "--pairs", pairs, "--menu-override", "D547"], capture_output=True, text=True,
                              env=w.env(PYTHONPATH=str(REPO / "src"), **env, **extra), cwd=str(REPO), timeout=600)
    r = _add(); out = r.stdout + r.stderr
    assert r.returncode == 0 and "submitted cluster_b" in out, out[-3000:]
    jid = out.split("submitted cluster_b")[1].split()[0]
    assert (w.search / "r4_trials.json").is_file()
    tasks = array_tasks(w.fake_b, jid); assert len(tasks) == 6, tasks                 # the fake array: task 0 = the array id, the rest consecutive raw ids
    for raw, _k in tasks:
        set_job_state(w.fake_b, raw, "FAILED", end="2026-01-01T00:30:00")              # every task died before a summary
    (w.search / "r4_t00").mkdir(exist_ok=True); (w.search / "r4_t00" / "RESULT").write_text("rc=1\n")
    r = _add(); out = r.stdout + r.stderr
    assert r.returncode == 2 and "r4_trials.json is on the record with 0/6 summaries and no job" in r.stderr, out
    # (a) the bound job REGISTERED a fit_trial: the real proof refuses; nothing moves
    make_registry(w.tmp / "registry", runs=[{"run_id": "fit_trial__registered_by_the_dead_job", "cluster_job_id": jid, "lifecycle_state": "ingested"}])
    before = _record_files(w.search)
    def _resub():
        return subprocess.run(["bash", str(REPRO / "03_hpo.sh"), "--model", MODEL, "--object-class", "lora", "--config", str(w.site / "config.env"), "--phase", "addendum",
                               "--cluster", "cluster_b", "--pairs", pairs, "--menu-override", "D547"], capture_output=True, text=True,
                              env=w.env(PYTHONPATH=str(REPO / "src"), SISTEER_RESUBMIT="1", **env), cwd=str(REPO), timeout=600)
    r = _resub(); out = r.stdout + r.stderr
    assert r.returncode == 2 and "REFUSED by the funnel's proof" in r.stderr and "NOTHING moved" in r.stderr, out[-3000:]
    assert ("registry has run" in out or "registered" in out) and "submitted cluster_b" not in out, out[-3000:]
    assert _record_files(w.search) == before and not (w.search / "superseded").exists(), "the proof refused: every file byte-identical"
    # (a2) the dead job's registration survives only in row_history — the live row was re-keyed to another job — the proof
    # still REFUSES (the per-task reader consults history under the REAL key)
    import sqlite3 as _sq
    from test_submit_twin_registry_history import real_row_history_ddl, _history_row
    make_registry(w.tmp / "registry", runs=[{"run_id": "fit_trial__rekeyed", "cluster_job_id": "999999", "lifecycle_state": "ingested"}])
    conn = _sq.connect(w.tmp / "registry" / "registry.sqlite"); conn.execute(real_row_history_ddl())
    for raw, _k in tasks:
        _history_row(conn, "fit_trial__rekeyed", raw, "2026-01-01T00:20:00.000Z")
    conn.commit(); conn.close()
    r = _resub(); out = r.stdout + r.stderr
    assert r.returncode == 2 and "REFUSED by the funnel's proof" in r.stderr and "NOTHING moved" in r.stderr, out[-3000:]
    assert _record_files(w.search) == before and not (w.search / "superseded").exists(), "re-keyed live row: history still refuses; byte-identical"
    # (a3) the proof authorizes, but the record host's queue cannot be read — the recheck fails CLOSED, nothing moves
    make_registry(w.tmp / "registry")
    r = subprocess.run(["bash", str(REPRO / "03_hpo.sh"), "--model", MODEL, "--object-class", "lora", "--config", str(w.site / "config.env"), "--phase", "addendum",
                        "--cluster", "cluster_b", "--pairs", pairs, "--menu-override", "D547"], capture_output=True, text=True,
                       env=w.env(PYTHONPATH=str(REPO / "src"), SISTEER_RESUBMIT="1", FAKE_SQUEUE_FAIL="1", **env), cwd=str(REPO), timeout=600)
    out = r.stdout + r.stderr
    # the phase's own queue read already fails closed on an unreadable queue (before the relaunch's recheck sentinel, which the harness test exercises)
    assert r.returncode == 2 and ("NOTHING moved" in r.stderr or "nothing archived, nothing submitted" in r.stderr) and "squeue" in out, out[-2500:]
    assert _record_files(w.search) == before and not (w.search / "superseded").exists(), "failing squeue: nothing moved"
    # (b) no registration anywhere: the real proof authorizes; the failed design is superseded, a new one re-derived and re-submitted
    make_registry(w.tmp / "registry")
    r = _resub(); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RESUBMIT authorized by the funnel's proof (bound job" in out and "submitted cluster_b" in out, out[-3000:]
    sup = sorted((w.search / "superseded").iterdir()); assert len(sup) == 1 and (sup[0] / "r4_trials.json").is_file() and (sup[0] / "r4_t00" / "RESULT").is_file()
    rcpt = json.loads((sup[0] / "transition.json").read_text())                                   # the transition receipt, closed
    assert rcpt["state"] == "complete" and rcpt["moved"] == ["r4_trials.json", "r4_t00"] and rcpt["replacement_sha256"] and rcpt["closed_utc"], rcpt
    assert not (w.search / "r4_resubmit.transition.json").exists() and not (w.search / ".resubmit.lock").exists()
    assert (w.search / "r4_trials.json").is_file() and not (w.search / "r4_t00").exists()
    _scan(w, out)
    # through the REAL ledger: the crash between the funnel's accepted sbatch and the receipt's `submitted` — the receipt
    # re-opened at `replaced`, the replacement array bound in the funnel's ledger, already terminal and gone from the queue. An empty queue is not
    # a rollback: the binding names the replacement job (not the proof's dead one) -> completed; the replacement design stands.
    jid2 = out.split("submitted cluster_b")[1].split()[0]; assert jid2 != jid
    for raw, _k in array_tasks(w.fake_b, jid2):
        set_job_state(w.fake_b, raw, "COMPLETED", end="2026-01-01T01:00:00")
    repl_sha = _sha(w.search / "r4_trials.json"); assert rcpt["replacement_sha256"] == repl_sha
    reopened = dict(rcpt, state="replaced"); reopened.pop("closed_utc", None)
    (w.search / "r4_resubmit.transition.json").write_text(json.dumps(reopened)); (sup[0] / "transition.json").unlink()
    before2 = _record_files(w.search)
    r = _add(); out = r.stdout + r.stderr
    assert "had been SUBMITTED" in out and f"binds hpo.lora.r4 to job {jid2}" in out and "RESTORED" not in out and "rolling back" not in out, out[-4000:]
    assert _sha(w.search / "r4_trials.json") == repl_sha and (sup[0] / "r4_trials.json").is_file() and not (w.search / "r4_t00").exists(), "the replacement design stands"
    assert json.loads((sup[0] / "transition.json").read_text())["state"] == "complete" and not (w.search / "r4_resubmit.transition.json").exists()
    assert {k: v for k, v in _record_files(w.search).items() if "transition.json" not in str(k)} == {k: v for k, v in before2.items() if "transition.json" not in str(k)}
    # through the REAL funnel: the LOST LAUNCH — a relaunch whose funnel died right after its sbatch accepted (the fake
    # crash point submit_after_sbatch): the token stands, the job exists with the attempt's nonce, nothing is in the ledger. The front door must
    # NOT roll the record back (the array reads the canonical design); the next addendum reconciles THROUGH the funnel: the job is adopted by its
    # nonce and bound, the receipt completes, and a later --resubmit proves against the NEW binding, never the old one.
    for raw, _k in array_tasks(w.fake_b, jid2):
        set_job_state(w.fake_b, raw, "FAILED", end="2026-01-01T02:00:00")               # the second attempt died too (unregistered): a third relaunch is due
    # the funnel's token reconciliation reads the controller BY NAME (`squeue -n … -o %i|%T|%V|%u|%k`): the World's squeue fake answers 03's `-o %j`
    # shape only, so for this block a wrapper answers `-n` in the testlib fake's by-name shape and delegates everything else to the World's
    from front_door_testlib import FAKE_SLURM
    (w.bin / "squeue.world").write_text((w.bin / "squeue").read_text()); (w.bin / "squeue.world").chmod(0o755)
    (w.bin / "squeue.byname").write_text(FAKE_SLURM["squeue"]); (w.bin / "squeue.byname").chmod(0o755)
    (w.bin / "squeue").write_text('#!/bin/bash\nfor a in $*; do [ $a = -n ] && exec $(dirname $0)/squeue.byname "$@"; done\nexec $(dirname $0)/squeue.world "$@"\n')
    r = subprocess.run(["bash", str(REPRO / "03_hpo.sh"), "--model", MODEL, "--object-class", "lora", "--config", str(w.site / "config.env"), "--phase", "addendum",
                        "--cluster", "cluster_b", "--pairs", pairs, "--menu-override", "D547"], capture_output=True, text=True,
                       env=w.env(PYTHONPATH=str(REPO / "src"), SISTEER_RESUBMIT="1", SISTEER_TWIN_CRASH_AT="submit_after_sbatch", **env), cwd=str(REPO), timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode != 0 and "CRASH INJECTED at submit_after_sbatch" in out and "may have LAUNCHED under it (token)" in out and "NOT rolling back" in out, out[-4000:]
    rc_open = json.loads((w.search / "r4_resubmit.transition.json").read_text()); assert rc_open["state"] == "replaced", rc_open
    repl3 = _sha(w.search / "r4_trials.json"); assert repl3 == rc_open["replacement_sha256"] and not (w.search / ".resubmit.lock").exists()
    bl = subprocess.run([sys.executable, str(REPRO / "lib" / "submit_twin.py"), "--config", str(w.site / "config.env"), "--model", MODEL, "--overrides", str(OVERRIDES), "binding", "--lwid", f"{MODEL}/03_hpo/lora/hpo.lora.r4"],
                        capture_output=True, text=True, env=w.env(PYTHONPATH=str(REPO / "src")), cwd=str(REPO), timeout=300); 
    assert "binding\ttransition" in bl.stdout and "\tyes" in bl.stdout, bl.stdout + bl.stderr                                          # the funnel's token stands
    r = _add(); out = r.stdout + r.stderr
    assert "ADOPTED and bound" in out and "had been SUBMITTED" in out and "RESTORED" not in out, out[-4000:]
    assert _sha(w.search / "r4_trials.json") == repl3 and not (w.search / "r4_resubmit.transition.json").exists(), "the replacement design stands; the receipt closed"
    sup3 = sorted((w.search / "superseded").iterdir()); assert json.loads((sup3[-1] / "transition.json").read_text())["state"] == "complete"
    bl = subprocess.run([sys.executable, str(REPRO / "lib" / "submit_twin.py"), "--config", str(w.site / "config.env"), "--model", MODEL, "--overrides", str(OVERRIDES), "binding", "--lwid", f"{MODEL}/03_hpo/lora/hpo.lora.r4"],
                        capture_output=True, text=True, env=w.env(PYTHONPATH=str(REPO / "src")), cwd=str(REPO), timeout=300)
    bline = next(l for l in bl.stdout.splitlines() if l.startswith("binding\t")); jid3 = bline.split("\t")[3]
    assert bline.split("\t")[1] == "submitted" and jid3 not in (jid, jid2) and bline.split("\t")[4] == "no", bl.stdout     # the lost launch is BOUND, not the old job
    # the adoption was by the NONCE read back from the job — the merged `sha:<sha>,nonce:<n>` comment on the HPO sbatch
    # line, the attempt record saying identified_by nonce with that nonce, and 03 naming it; the singleton path is gone
    argv3 = (w.fake_b / "jobs" / f"{jid3}.argv").read_text(); m = re.search(r"--comment=sha:[0-9a-f]{40},nonce:([0-9a-f]{12})", argv3); assert m, argv3
    assert f"identified by its nonce read back from the job, {m.group(1)}" in out and "the one qualifying candidate" not in out, out[-4000:]
    batt = next(l for l in bl.stdout.splitlines() if l.startswith("bound_attempt\t")); assert batt.split("\t")[2] == "nonce" and batt.split("\t")[3] == m.group(1), bl.stdout
    pr = subprocess.run([sys.executable, str(REPRO / "lib" / "submit_twin.py"), "--config", str(w.site / "config.env"), "--model", MODEL, "--overrides", str(OVERRIDES), "prove-resubmit", "--unit", "hpo.lora.r4",
                        "--lwid", f"{MODEL}/03_hpo/lora/hpo.lora.r4", "--kind", "fit", "--registers", "registry"], capture_output=True, text=True, env=w.env(PYTHONPATH=str(REPO / "src")), cwd=str(REPO), timeout=300)
    assert pr.returncode != 0 and f"bound job {jid3}" in pr.stdout + pr.stderr or jid3 in pr.stdout, pr.stdout + pr.stderr        # a later --resubmit proves against the NEW binding (still live: not authorized)
    assert "NOT authorized" in pr.stdout and jid2 not in pr.stdout.split("bound job")[-1][:20], pr.stdout
    # the pick under the override through the REAL front door (the startup guard admits --menu-override for pick) with a
    # completed addendum row at the picked pair on the record -> the confirmation decision printed under --dry-run, nothing written
    d4 = json.loads((w.search / "r4_trials.json").read_text()); t16 = next(t for t in d4["trials"] if (t["lora_rank"], t["lora_set_id"]) == (16, "mlp"))
    src = json.loads((w.search / "r1_t03" / "summary.json").read_text())
    smry = dict(src); smry["round"] = 4; smry["trial"] = {**src["trial"], **{k: v for k, v in t16.items() if k != "lr_source"}, "trial_role": "addendum"}
    smry["lr"] = t16["lr"]; smry["status"] = "ok"; smry["floor_ok"] = True; smry["best_heldout_loss"] = float(src["best_heldout_loss"]) - 0.01
    for root in (w.search, w.state):                               # on the record AND the mirror, as the live fetch leaves them (--dry-run fetches nothing)
        (root / "r4_t02").mkdir(exist_ok=True); (root / "r4_t02" / "summary.json").write_text(json.dumps(smry, default=str))
        (root / "r4_t02" / "metrics.jsonl").write_text("".join(json.dumps({"step": i, "lora_r4_ce/grad_norm": 0.01, "train/grad_norm_preclip": 0.01}) + "\n" for i in range(1, 6)))
    r = subprocess.run(["bash", str(REPRO / "03_hpo.sh"), "--model", MODEL, "--object-class", "lora", "--config", str(w.site / "config.env"), "--phase", "pick",
                        "--cluster", "cluster_b", "--rank", "16", "--set", "mlp", "--menu-override", "D547", "--dry-run"], capture_output=True, text=True,
                       env=w.env(PYTHONPATH=str(REPO / "src"), **env), cwd=str(REPO), timeout=600); out = r.stdout + r.stderr
    assert "applies to --phase addendum" not in out and "pick: MENU OVERRIDE D547 — the pick's menu is" in out, out[-3000:]
    assert "the completed addendum trial r4_t02 at these coordinates IS the confirmation" in out and "CONFIRM-PICK requested" not in out, out[-3000:]
    assert not (w.state / "hpo_pick.json").exists() and not (w.state / "hpo_fold.json").exists(), "dry-run: nothing written"
    # the pick under the override is admitted by the front door's argument check and reads the bound override menu (dry-run: nothing written)
    r = subprocess.run(["bash", str(REPRO / "03_hpo.sh"), "--model", MODEL, "--object-class", "lora", "--config", str(w.site / "config.env"), "--phase", "pick",
                        "--cluster", "cluster_b", "--rank", "16", "--set", "mlp", "--menu-override", "D547", "--dry-run"], capture_output=True, text=True,
                       env=w.env(PYTHONPATH=str(REPO / "src"), **env), cwd=str(REPO), timeout=600); out = r.stdout + r.stderr
    assert "applies to --phase addendum" not in out and "pick: MENU OVERRIDE D547 — the pick's menu is" in out, out[-3000:]


RSYNC_DROPS_T05 = RSYNC_BOTH_WAYS.replace('exec %s "${args[@]}" "$src" "$dst"', '%s "${args[@]}" "$src" "$dst"; rc=$?; rm -rf "$dst"/r1_t05; exit $rc')


@pytest.mark.parametrize("phase,extra", [("analyze", ()), ("pick", ("--rank", "4", "--set", "mlp"))])
def test_analyze_and_pick_refuse_a_short_fetch_and_a_mirror_summary_the_record_does_not_hold(worlds, tmp_path, phase, extra):
    """Residual (a): the analyze and pick phases derive from / read the mirror's trial directories, so they prove the mirror IS the search
    record first, as table and addendum do — a fetch that came up short (a summary the record holds is missing on the mirror) and a summary on the
    mirror the record does not hold are each refused BY NAME before anything is derived, picked or written."""
    w = worlds
    _lay_probe_record(w)
    r = _run(w, "--phase", "probe", "--cluster", "cluster_a"); assert r.returncode == 0, r.stdout + r.stderr
    _lay_search(w, tmp_path)
    if phase == "analyze":
        (w.state / "r1_fit.json").unlink()                                         # analyze derives it; nothing may be derived on a refusal
    # (1) a short fetch: the fake rsync drops r1_t05 after copying
    (w.bin / "rsync").write_text(RSYNC_DROPS_T05 % REAL_RSYNC)
    r = _run(w, "--phase", phase, "--cluster", "cluster_b", *extra); out = r.stdout + r.stderr
    assert r.returncode == 2, out
    assert f"REFUSED: {phase}: the mirror {w.state} is not the record cluster_b:{w.search}" in r.stderr, out
    assert "r1_t05/summary.json(mirror differs from the record)" in r.stderr and "nothing read, derived, published or marked" in r.stderr, out
    untouched = (lambda: not (w.state / "r1_fit.json").exists()) if phase == "analyze" else (lambda: not (w.state / "hpo_pick.json").exists())
    assert untouched() and "hpo_pick" not in out and "[analyze" not in out, out          # nothing derived (analyze) / nothing picked (pick)
    # (2) the fetch is whole again, but the mirror holds a summary the record does not (another source's residue)
    (w.bin / "rsync").write_text(RSYNC_BOTH_WAYS % REAL_RSYNC)
    shutil.copytree(w.state / "r1_t03", w.state / "r1_t16")
    r = _run(w, "--phase", phase, "--cluster", "cluster_b", *extra); out = r.stdout + r.stderr
    assert r.returncode == 2 and f"REFUSED: {phase}: the mirror" in r.stderr and "r1_t16/summary.json(on the mirror, not on the record)" in r.stderr, out
    assert "r1_t05/summary.json(" not in r.stderr and untouched() and "hpo_pick" not in out and "[analyze" not in out, out
    assert (w.state / "r1_t16" / "summary.json").is_file(), "nothing deleted by the refusal"
    shutil.rmtree(w.state / "r1_t16")
    # (2b) a SIDECAR on the mirror the record does not hold (another source's metrics under this record's summaries) is residue too
    (w.state / "r1_t03" / "metrics.jsonl").write_text(json.dumps({"step": 1, "train/grad_norm_preclip": 0.01}) + "\n")
    r = _run(w, "--phase", phase, "--cluster", "cluster_b", *extra); out = r.stdout + r.stderr
    assert r.returncode == 2 and "r1_t03/metrics.jsonl(on the mirror, not on the record)" in r.stderr and untouched(), out
    (w.state / "r1_t03" / "metrics.jsonl").unlink()
    # (3) the residue moved aside: the inventory matches and the phase proceeds past the check
    r = _run(w, "--phase", phase, "--cluster", "cluster_b", *extra); out = r.stdout + r.stderr
    assert f"{phase}: the mirror is the record cluster_b:{w.search} — 16 trial summaries + 0 sidecars verified" in out and f"REFUSED: {phase}: the mirror" not in out, out
    if phase == "analyze":
        # the analyze DONE marker is keyed on the evidence — unchanged evidence skips; a metrics backfill on the record re-derives
        amark = w.tmp / "state" / MODEL / "03_hpo" / "DONE.lora.analyze"
        assert r.returncode == 0 and (w.state / "r1_fit.json").is_file() and "set=" in amark.read_text(), out
        k1 = amark.read_text().split("set=")[1].strip()
        r = _run(w, "--phase", "analyze", "--cluster", "cluster_b"); out = r.stdout + r.stderr
        assert r.returncode == 0 and f"analyze: DONE marker present for this evidence (set={k1}) — skipping" in out and "[analyze" not in out, out
        (w.search / "r1_t03" / "metrics.jsonl").write_text("".join(json.dumps({"step": i, "lora_r4_ce/grad_norm": 0.01, "train/grad_norm_preclip": 0.01}) + "\n" for i in range(1, 6)))
        r = _run(w, "--phase", "analyze", "--cluster", "cluster_b"); out = r.stdout + r.stderr
        assert r.returncode == 0 and "the DONE marker records other evidence than the record holds now" in out and "the fit is re-derived" in out, out
        assert "16 trial summaries + 1 sidecars verified" in out and amark.read_text().split("set=")[1].strip() != k1, out
        fit = json.loads((w.state / "r1_fit.json").read_text())
        row = next(rr for rr in fit["rows"] if int(rr.get("index")) == 3 and int(rr.get("round", 1)) == 1)
        assert row["clip_activation_rate"] == pytest.approx(0.0) and row["clip_n_steps"] == 5, row
    _scan(w, out)
