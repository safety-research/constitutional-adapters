"""`reproduce/03_hpo.sh --phase probe` handles FAILED_START (paper-companion-repo 2026-09-30).

The carrier writes `probe_attempt.json` at its start, BEFORE the serving-venv bootstrap and before the driver's first
`probe_progress.json`; a failure in between (venv, import, sha pin) leaves a record with an attempt record and no checkpoint.
`hpo_probe_status.py --record` reads that as FAILED_START (attempt present, progress absent, no job active). Before: the record
read NOT_STARTED, 03 re-submitted the UNSUFFIXED unit and the funnel skipped it as already submitted (rc 0) — stuck forever.
Now: with a plan on the record, 03 RESUMES it (attempt+1 from the attempt record — the attempt-scoped unit, never the skipped
one; the carrier passes --resume --replan); with no plan, nothing was planned and 03 REFUSES by name with the recovery steps,
which end in `--phase probe --resubmit` (the funnel's re-mint of the bound fresh unit, whose proof the funnel owns).
Executed against the REAL stage with the probe-resume harness (fake ssh/rsync/sbatch/squeue, the real status reader)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import job_ids, set_job_state  # noqa: E402,F401
from test_front_door_probe_resume import LWID, MODEL, NAME, REAL_RSYNC, SRV_ID, World, _attempt, _plan  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reproduce" / "lib"))
import stage as ST  # noqa: E402


@pytest.fixture
def world(tmp_path):
    if REAL_RSYNC is None:
        pytest.skip("no rsync on this host")
    return World(tmp_path)


def _failed_start(world: World, *, plan: bool, attempt: int = 1) -> None:
    """A carrier that died between its attempt record and the driver's first checkpoint: the attempt record (and the plan when
    the driver got that far), no probe_progress.json, no launches, no job in the queue; the fresh unit bound in the ledger to a
    job that is terminal."""
    (world.record / "probe_attempt.json").write_text(json.dumps(_attempt(attempt=attempt, cap=4, gpus=4) | {"job_id": "77777777"}))   # written by the bound job
    if plan:
        (world.record / "probe_plan.json").write_text(json.dumps(_plan()))
    world.ledger.parent.mkdir(parents=True, exist_ok=True); world.ledger.write_text("")
    ST.write_binding(str(world.ledger), ST.submitted_record(LWID, "77777777", "cluster_a", f"{LWID}@cluster_a@1", NAME))
    jobs = world.fake / "jobs"; jobs.mkdir(exist_ok=True)
    (jobs / "77777777.name").write_text(NAME + "\n"); (jobs / "77777777.argv").write_text("sbatch\n"); (jobs / "77777777.env").write_text("")
    set_job_state(world.fake, "77777777", "FAILED", end="2026-01-01T00:03:00")


def test_failed_start_with_a_plan_resumes_the_record_as_the_next_attempt(world):
    _failed_start(world, plan=True)
    r = world.run()
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "FAILED_START" in out and "resuming" in out
    calls = world.sbatch_calls()
    new = [c for c in calls if c["id"] != 77777777]
    assert len(new) == 1, out
    argv = new[0]["argv"]
    assert argv[argv.index("tp1") + 1:] == ["2", "4", "auto"] and "--gres=gpu:4" in argv          # attempt 2/4 of the record, its width
    assert f"submitted cluster_a {new[0]['id']} hpo.lora.probe.a2" in out                           # the attempt-scoped unit, never the skipped one
    assert "skipped" not in out and ST.binding_state(str(world.ledger), f"{LWID}.a2")["state"] == "submitted"
    # and once that job is queued under the probe's name the record reads RUNNING: nothing more is submitted
    r2 = world.run()
    assert r2.returncode == 0 and len([c for c in world.sbatch_calls() if c["id"] != 77777777]) == 1, r2.stdout + r2.stderr
    assert "RUNNING" in r2.stdout + r2.stderr


def test_failed_start_without_a_plan_refuses_by_name_with_the_recovery_steps_and_submits_nothing(world):
    _failed_start(world, plan=False)
    r = world.run()
    out = r.stdout + r.stderr
    assert r.returncode == 2, out
    assert "FAILED_START" in r.stderr and "no plan" in r.stderr.lower() and "nothing to resume" in r.stderr
    assert "probe_attempt.json" in r.stderr and "superseded" in r.stderr and "--phase probe --resubmit" in r.stderr and "never delete" in r.stderr
    assert "77777777" in r.stderr or NAME in r.stderr                                                   # the failed attempt is named (its log is the evidence)
    assert [c for c in world.sbatch_calls() if c["id"] != 77777777] == []
    assert not (world.tmp / "state" / MODEL / "03_hpo" / "DONE.lora.probe").exists()
    # the recovery's last step: --resubmit is forwarded to the funnel for the FRESH unit, whose proof decides (here: the fake
    # host has no ingest watermark, so the funnel's own refusal names the proof — the front door neither skipped nor re-typed)
    (world.record / "probe_attempt.json").rename(world.record / "probe_attempt.failed.json")
    r3 = world.run("--resubmit")
    out3 = r3.stdout + r3.stderr
    assert "skipped" not in out3 and "--resubmit" in out3, out3
    assert ("not provably dead-and-unregistered" in out3) or f"submitted cluster_a" in out3, out3


def test_planless_failed_start_with_the_carriers_result_is_recovered_by_resubmit_and_reaches_the_funnel(world):
    """The carrier's EXIT trap writes `$OUT/RESULT` after the attempt record, and the status reads
    FAILED_START while RESULT exists — archiving probe_attempt.json alone would loop. `--phase probe --resubmit` on a plan-less
    FAILED_START record performs the guarded recovery itself: every top-level file (probe_attempt.json, RESULT, leftovers) moves
    to superseded/<utc>/ on the record (never deleted), the status is re-read (NOT_STARTED), and the fresh submission reaches the
    funnel with --resubmit (its proof decides; the record is no longer FAILED_START)."""
    _failed_start(world, plan=False)
    (world.record / "RESULT").write_text(f"state=FAILED exit=3 attempt=1/4 elapsed=12 host=TESTHOST model={MODEL} mode=tp1 job=77777777 verdict=none\n")
    (world.record / "probe_attempt.json.tmp.999").write_text("{")                     # a leftover from a torn write
    r = world.run()
    assert r.returncode == 2 and "FAILED_START" in r.stderr and "--phase probe --resubmit" in r.stderr, r.stdout + r.stderr
    assert "RESULT" in r.stderr and "superseded" in r.stderr and "never deleted" in r.stderr   # the printed text says what --resubmit will do
    assert (world.record / "RESULT").exists() and (world.record / "probe_attempt.json").exists()   # a refusal moves nothing
    r2 = world.run("--resubmit")
    out = r2.stdout + r2.stderr
    arch = sorted((world.record / "superseded").iterdir())
    assert len(arch) == 1 and {p.name for p in arch[0].iterdir()} == {"probe_attempt.json", "RESULT", "probe_attempt.json.tmp.999"}, out
    assert not (world.record / "probe_attempt.json").exists() and not (world.record / "RESULT").exists()
    assert "recovered the plan-less record" in out and "moved to superseded/" in out and "record reads NOT_STARTED" in out, out
    assert not any(ln.startswith("skipped ") for ln in out.splitlines()) and "FAILED_START-free" not in out
    # the fresh unit reached the funnel with --resubmit: its fenced proof decides (a refusal naming the proof, or a re-mint — the
    # stage-06 contract re-mints the never-registered probe); either way the front door neither skipped nor re-typed
    assert ("not provably dead-and-unregistered" in out) or ("submitted cluster_a" in out and "hpo.lora.probe" in out), out
    assert world.ssh_log.read_text().count("--record") >= 3                                 # run 1: one read; run 2: before and after the recovery
    # the record is no longer FAILED_START. When the funnel re-minted the unit, the new job is the bound one: make it terminal so no
    # job is active, and a plain re-run meets the NOT_STARTED-but-bound refusal naming THAT job — never FAILED_START
    new_jobs = [c["id"] for c in world.sbatch_calls() if c["id"] != 77777777]
    for jid in new_jobs:
        set_job_state(world.fake, str(jid), "FAILED", end="2026-01-01T00:30:00")
    bound = str(new_jobs[-1]) if new_jobs else "77777777"
    r3 = world.run()
    assert r3.returncode == 2 and "FAILED_START" not in r3.stderr and "EMPTY (NOT_STARTED)" in r3.stderr and bound in r3.stderr, r3.stdout + r3.stderr


def test_not_started_with_the_fresh_unit_bound_to_a_dead_job_refuses_by_name_unless_resubmit(world):
    """A submitted job that died BEFORE creating any record file (a shared-root / SERVING_PROBE_OUT
    failure before `mkdir $OUT`, a Slurm start failure) leaves an EMPTY record (NOT_STARTED) with the fresh unit bound in the
    ledger; a plain submit would be skipped by the funnel as already submitted (rc 0) and `--phase all` would wait forever.
    Bound + no job active -> refused by name with the dead job and its log path; `--resubmit` hands it to the funnel's proof."""
    world.ledger.parent.mkdir(parents=True, exist_ok=True); world.ledger.write_text("")
    ST.write_binding(str(world.ledger), ST.submitted_record(LWID, "77777777", "cluster_a", f"{LWID}@cluster_a@1", NAME))
    jobs = world.fake / "jobs"; jobs.mkdir(exist_ok=True)
    (jobs / "77777777.name").write_text(NAME + "\n"); (jobs / "77777777.argv").write_text("sbatch\n"); (jobs / "77777777.env").write_text("")
    set_job_state(world.fake, "77777777", "NODE_FAIL", end="2026-01-01T00:01:00")
    assert not any(world.record.iterdir())                                                 # zero record files
    r = world.run()
    assert r.returncode == 2, r.stdout + r.stderr
    assert "EMPTY (NOT_STARTED)" in r.stderr and "77777777" in r.stderr and f"{NAME}_77777777.out" in r.stderr and "--phase probe --resubmit" in r.stderr
    assert [c for c in world.sbatch_calls() if c["id"] != 77777777] == [] and not any(ln.startswith("skipped ") for ln in (r.stdout + r.stderr).splitlines())
    r2 = world.run("--resubmit")
    out = r2.stdout + r2.stderr
    assert not any(ln.startswith("skipped ") for ln in out.splitlines()) and "EMPTY (NOT_STARTED)" not in out, out
    assert ("not provably dead-and-unregistered" in out) or ("submitted cluster_a" in out and "hpo.lora.probe" in out), out


def test_a_menu_larger_than_the_argv_limit_is_read_whole_and_its_fields_land_in_the_fetch_record(world):
    """`json_field` used to pass the JSON document as ONE argv; the live 174 KB allowed_pairs.json exceeded the
    128 KiB per-argument limit (`Argument list too long`), so written_utc / probe_job_id were recorded EMPTY. The document now travels
    on stdin: a >128 KiB synthetic menu is read whole and both fields land in probe_fetch.json and the DONE marker."""
    import subprocess
    from test_front_door_probe_resume import REPO as PR_REPO, _stamp
    world.lay(done=7, complete=True, menu=True, verdict=True)
    menu = json.loads((world.record / "allowed_pairs.json").read_text())
    menu["excluded"] = [{"rank": 64, "set": f"set{i:05d}", "reason": "x" * 400, "decided": True} for i in range(400)]     # ~180 KB
    (world.record / "allowed_pairs.json").write_text(json.dumps(menu))
    assert (world.record / "allowed_pairs.json").stat().st_size > 128 * 1024
    _stamp(world)
    r = world.run()
    out = r.stdout + r.stderr
    assert r.returncode == 0 and "Argument list too long" not in out, out
    fetched = json.loads((world.state / "probe_fetch.json").read_text())
    assert fetched["written_utc"] == "2026-09-30T02:00:00Z" and fetched["probe_job_id"] == "11111111", fetched
    marker = (world.tmp / "state" / MODEL / "03_hpo" / "DONE.lora.probe").read_text()
    assert "written_utc=2026-09-30T02:00:00Z" in marker and "probe_job_id=11111111" in marker and "written  by probe job )" not in out
    # the helper itself, run under bash with a 300 KB document: the value comes back whole
    src = (PR_REPO / "reproduce" / "03_hpo.sh").read_text()
    frag = src[src.index("json_field(){"):src.index("probe_status(){")]
    big = json.dumps({"written_utc": "2026-09-30T02:00:00Z", "pad": "y" * 300_000, "n": 7})
    script = f"set -u; PY={sys.executable}\n{frag}\nDOC=$(cat)\njson_field \"$DOC\" written_utc; json_field \"$DOC\" n; json_field \"$DOC\" absent; json_field '' written_utc"
    rr = subprocess.run(["bash", "-c", script], input=big, capture_output=True, text=True, timeout=60)
    assert rr.returncode == 0 and rr.stdout == "2026-09-30T02:00:00Z\n7\n\n\n", rr.stdout + rr.stderr


def test_an_unreadable_queue_is_unknown_never_empty_and_nothing_is_archived(world):
    """`cluster_job_active` used to read an ssh/squeue failure as an EMPTY queue (and cache it), so a live carrier
    still bootstrapping (attempt record written, no checkpoint) read FAILED_START and `--resubmit` archived its live attempt file. Now
    an unreadable queue refuses by name — UNKNOWN, never 'not active' — and nothing is archived or submitted."""
    _failed_start(world, plan=False)
    set_job_state(world.fake, "77777777", "RUNNING")                                   # the bound carrier is ALIVE (bootstrapping)
    r = world.run("--resubmit", FAKE_SQUEUE_FAIL="1")
    assert r.returncode == 2, r.stdout + r.stderr
    assert "could not be read" in r.stderr and "UNKNOWN" in r.stderr and "nothing archived, nothing submitted" in r.stderr
    assert (world.record / "probe_attempt.json").exists() and not (world.record / "superseded").exists()
    assert [c for c in world.sbatch_calls() if c["id"] != 77777777] == []
    assert "--record" not in world.ssh_log.read_text()                                  # the record was not even read: the queue comes first


def test_a_carrier_sacct_still_reports_active_is_never_archived(world):
    """The queue-by-name query misses the live job (here: its fake name differs) so the record reads FAILED_START,
    but the funnel's own sacct-based proof says the bound job is STILL ACTIVE — nothing is archived, nothing submitted; once it is
    dead the same command recovers."""
    _failed_start(world, plan=False)
    (world.fake / "jobs" / "77777777.name").write_text("some-other-name\n")
    set_job_state(world.fake, "77777777", "RUNNING")
    r = world.run("--resubmit")
    assert r.returncode == 2, r.stdout + r.stderr
    assert "NOT proven dead" in r.stderr and "skip" in r.stderr and "still" in r.stderr and "nothing archived" in r.stderr
    assert (world.record / "probe_attempt.json").exists() and not (world.record / "superseded").exists()
    assert [c for c in world.sbatch_calls() if c["id"] != 77777777] == []
    set_job_state(world.fake, "77777777", "NODE_FAIL", end="2026-01-01T00:09:00")
    r2 = world.run("--resubmit")
    out = r2.stdout + r2.stderr
    assert "the funnel's proof: resubmit" in out and "recovered the plan-less record" in out, out
    assert len(sorted((world.record / "superseded").iterdir())) == 1
    assert ("not provably dead-and-unregistered" in out) or ("submitted cluster_a" in out and "hpo.lora.probe" in out), out


def test_two_concurrent_recoveries_serialize_one_archives_and_submits_the_other_refuses_by_name(world):
    """The read → archive → submit transaction holds a per-record non-blocking lock on this host; the second
    invocation refuses by name while the first holds it (a pause hook keeps the first inside the window)."""
    import time
    _failed_start(world, plan=False)
    pause = world.tmp / "pause"; pause.write_text("")
    out_a = world.tmp / "a.out"; err_a = world.tmp / "a.err"
    with open(out_a, "w") as fo, open(err_a, "w") as fe:
        a = world.popen("--resubmit", stdout=fo, stderr=fe, SISTEER_PROBE_PAUSE_FILE=str(pause))
        for _ in range(300):
            if "recovery lock" in out_a.read_text():
                break
            time.sleep(0.1)
        assert "recovery lock" in out_a.read_text(), out_a.read_text() + err_a.read_text()
        b = world.run("--resubmit")
        assert b.returncode == 2 and "holds the recovery lock" in b.stderr and "nothing archived, nothing submitted" in b.stderr, b.stdout + b.stderr
        assert not (world.record / "superseded").exists()                                # B archived nothing
        pause.unlink()
        a.wait(timeout=600)
    oa = out_a.read_text() + err_a.read_text()
    assert a.returncode == 0, oa
    assert "recovered the plan-less record" in oa and len(sorted((world.record / "superseded").iterdir())) == 1
    assert ("not provably dead-and-unregistered" in oa) or ("submitted cluster_a" in oa and "hpo.lora.probe" in oa), oa
    assert len([c for c in world.sbatch_calls() if c["id"] != 77777777]) <= 1              # at most ONE replacement


def test_an_empty_menu_refuses_the_lora_recon_by_name(world):
    """The live allowed_pairs.json had `pairs: []` (every pair excluded, a harness defect) and `widest_pair` crashed with
    `--lora-rank: invalid int value: ''` from max() of an empty list. Now the helper refuses by name and both callers stop on it."""
    import subprocess
    from test_front_door_probe_resume import REPO as PR_REPO
    src = (PR_REPO / "reproduce" / "03_hpo.sh").read_text()
    frag = src[src.index("widest_pair(){"):src.index("box_top_from_recon(){")]
    empty = world.tmp / "allowed_pairs.json"; empty.write_text(json.dumps({"model_key": MODEL, "pairs": [], "excluded": [{"rank": 4, "set": "wo"}]}))
    r = subprocess.run(["bash", "-c", f"set -u; PY={sys.executable}\n{frag}\nwidest_pair {empty}"], capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "certifies no pair" in r.stderr and "--phase probe --reopen" in r.stderr and r.stdout == "", r.stdout + r.stderr
    full = world.tmp / "full.json"; full.write_text(json.dumps({"pairs": [{"rank": 4, "set": "wo", "lora_modules": ["o_proj"], "trainable_params": 10},
                                                                            {"rank": 16, "set": "down", "lora_modules": ["down_proj"], "trainable_params": 99}]}))
    r2 = subprocess.run(["bash", "-c", f"set -u; PY={sys.executable}\n{frag}\nwidest_pair {full}"], capture_output=True, text=True, timeout=60)
    assert r2.returncode == 0 and r2.stdout.strip() == "16 down_proj", r2.stdout + r2.stderr
    # every caller stops on the refusal (never `read`s an empty answer into --lora-rank)
    assert 'read -r WR WM <<<"$(widest_pair "$ALLOWED")"' not in src and src.count("certifies no pair") >= 2
    # the stage itself: a READY record whose menu certifies NO pair is fetched (the verdict is the record's), and the LoRA recon then
    # refuses by name instead of submitting a ladder at rank ''
    world.lay(done=7, complete=True, menu=True, verdict=True)
    menu = json.loads((world.record / "allowed_pairs.json").read_text()); menu["pairs"] = []; menu["excluded"] = [{"rank": 4, "set": "wo", "reason": "x", "decided": True}]
    (world.record / "allowed_pairs.json").write_text(json.dumps(menu))
    from test_front_door_probe_resume import _stamp
    _stamp(world)
    r3 = world.run()
    assert r3.returncode == 0 and (world.state / "allowed_pairs.json").is_file(), r3.stdout + r3.stderr
    r4 = subprocess.run(["bash", str(PR_REPO / "reproduce" / "03_hpo.sh"), "--model", MODEL, "--object-class", "lora", "--phase", "recon",
                         "--config", str(world.site / "config.env")], capture_output=True, text=True, env=world.env(), cwd=str(PR_REPO), timeout=600)
    assert r4.returncode == 2 and "certifies no pair" in r4.stderr and "re-judge the reference" in r4.stderr, r4.stdout + r4.stderr
    assert "invalid int value" not in r4.stdout + r4.stderr and [c for c in world.sbatch_calls()] == []


def test_a_transition_token_refuses_the_recovery_by_name_and_touches_nothing(world):
    """A `transition` binding means the funnel's re-run of the fresh unit is MID-FLIGHT (23eed2e32); the old
    attempt's job id proves nothing about the replacement — the recovery refuses by name (the token, the reconciliation) and moves nothing."""
    _failed_start(world, plan=False)
    tok = {"logical_work_id": LWID, "utc": "2026-09-30T05:00:00Z", "old": {"job_id": "77777777"}, "intended": {"attempt_id": f"{LWID}@cluster_a@2", "job_name": NAME}}
    ST.write_transition(str(world.ledger), tok)
    assert ST.binding_state(str(world.ledger), LWID)["state"] == "transition"
    r = world.run("--resubmit")
    assert r.returncode == 2, r.stdout + r.stderr
    assert "TRANSITION token" in r.stderr and "transition.json" in r.stderr and "reconciliation must run first" in r.stderr and "nothing archived" in r.stderr
    assert (world.record / "probe_attempt.json").exists() and not (world.record / "superseded").exists()
    assert [c for c in world.sbatch_calls() if c["id"] != 77777777] == []
    ST.clear_transition(str(world.ledger), LWID)                                        # reconciled by the funnel: the same command recovers
    r2 = world.run("--resubmit")
    assert "recovered the plan-less record" in r2.stdout + r2.stderr, r2.stdout + r2.stderr


def test_a_replacement_attempt_written_between_the_proof_and_the_archive_is_never_moved(world):
    """The archive fences the exact attempt GENERATION — inside the same remote critical section as the mv,
    probe_attempt.json is re-read and must still name the job the terminal proof covered; a replacement carrier's record written in
    between refuses the mv, nothing is archived, nothing submitted."""
    import time
    _failed_start(world, plan=False)
    pause = world.tmp / "pause_archive"; pause.write_text("")
    out_a = world.tmp / "a.out"; err_a = world.tmp / "a.err"
    with open(out_a, "w") as fo, open(err_a, "w") as fe:
        a = world.popen("--resubmit", stdout=fo, stderr=fe, SISTEER_PROBE_PAUSE_BEFORE_ARCHIVE_FILE=str(pause))
        for _ in range(300):
            if "the funnel's proof: resubmit" in out_a.read_text():
                break
            time.sleep(0.1)
        assert "the funnel's proof: resubmit" in out_a.read_text(), out_a.read_text() + err_a.read_text()
        # the funnel's own retry accepted a replacement: its carrier writes a NEW attempt record into the same directory
        (world.record / "probe_attempt.json").write_text(json.dumps(_attempt(attempt=2, cap=4, gpus=4) | {"job_id": "88888888"}))
        pause.unlink()
        a.wait(timeout=600)
    oa = out_a.read_text() + err_a.read_text()
    assert a.returncode == 2, oa
    assert "now names job 88888888, the terminal proof covered job 77777777" in oa and "nothing moved" in oa and "nothing submitted" in oa
    assert json.loads((world.record / "probe_attempt.json").read_text())["job_id"] == "88888888"       # the LIVE replacement's record stands
    assert not (world.record / "superseded").exists() and [c for c in world.sbatch_calls() if c["id"] != 77777777] == []


def test_an_attempt_record_naming_another_job_than_the_proved_one_refuses(world):
    """The proof covers ONE job (the bound one); an attempt record naming a different job is another generation — refused, nothing moved."""
    _failed_start(world, plan=False)
    (world.record / "probe_attempt.json").write_text(json.dumps(_attempt(attempt=1, cap=4, gpus=4) | {"job_id": "55555555"}))
    r = world.run("--resubmit")
    assert r.returncode == 2 and "names job 55555555 but the funnel's proof covered job 77777777" in r.stderr and "different attempt generation" in r.stderr, r.stdout + r.stderr
    assert (world.record / "probe_attempt.json").exists() and not (world.record / "superseded").exists()


def test_parity_stale_refuses_by_name_with_both_generations_and_never_resumes(world):
    """The menu is the record's current generation, but the PARITY-PASS record the verdict read adapter parity from
    moved — hpo_probe_status says PARITY_STALE. The probe phase refuses BY NAME (parity_why, both generations, the remedy `--phase
    probe --reopen` → reverdict --parity-record), fetches nothing, marks nothing, and never probe_resumes."""
    from test_front_door_probe_resume import _stamp
    world.lay(done=7, complete=True, menu=True, verdict=True)
    gen = _stamp(world)
    pdir = world.record / "parity"; (pdir / "launches").mkdir(parents=True)
    plan = _plan(); plan["parity_pass"] = True; plan["n_launches"] = 2; plan["launches"] = plan["launches"][:2]
    (pdir / "probe_plan.json").write_text(json.dumps(plan))
    (pdir / "probe_progress.json").write_text(json.dumps({"complete": True, "n_done": 2, "plan_launches": 2}))
    for i in (1, 2):
        (pdir / "launches" / f"L0{i}.json").write_text(json.dumps({"launch_id": f"L0{i}", "status": "passed", "host": "TESTHOST", "redo": True}))
    stamp = {"dir": str(pdir), "result_generation": "e" * 64, "n_results": 2}          # the verdict read the parity record at ANOTHER generation
    for name in ("probe_verdict.json", "allowed_pairs.json"):
        d = json.loads((world.record / name).read_text()); d["parity_record"] = stamp; (world.record / name).write_text(json.dumps(d))
    r = world.run()
    out = r.stdout + r.stderr
    assert r.returncode == 2, out
    assert "PARITY_STALE" in r.stderr and "parity record" in r.stderr                      # the module's own parity_why prose is its contract, not pinned here
    assert "stamped generation eeeeeeeeeeee" in r.stderr and "live " in r.stderr and str(pdir) in r.stderr
    assert "--phase probe --reopen" in r.stderr and f"reverdict --parity-record {pdir}" in r.stderr and "NOT fetched, NOT marked, NOT resumed" in r.stderr
    assert not (world.state / "allowed_pairs.json").exists() and not (world.tmp / "state" / MODEL / "03_hpo" / "DONE.lora.probe").exists()
    assert world.sbatch_calls() == [] and "resuming" not in out and "hpo.lora.probe.a" not in out
    # the main record's own stamps are still the current generation: it is the PARITY record that is stale, as the text says
    assert gen[:12] not in r.stderr.split("[parity record")[1]


def test_a_reverdict_that_keeps_the_generation_is_seen_re_fetched_and_the_site_refreshed(world):
    """`reverdict --parity-record` keeps the RESULT GENERATION (it hashes the launch results) but rewrites
    probe_verdict / allowed_pairs / geometry_record. Currency is now the VERDICT identity: the stale mirror is archived, re-fetched,
    probe_fetch.json records the identity actually fetched (with the menu's written_utc / probe_job_id), the site copy is refreshed."""
    from test_front_door_probe_resume import _stamp
    world.lay(done=7, complete=True, menu=True, verdict=True)
    _stamp(world)
    r = world.run(); assert r.returncode == 0, r.stdout + r.stderr
    f1 = json.loads((world.state / "probe_fetch.json").read_text())
    assert len(f1["verdict_sha256"]) == 64 and len(f1["menu_sha256"]) == 64 and f1["written_utc"] == "2026-09-30T02:00:00Z" and f1["probe_job_id"] == "11111111"
    # the re-verdict: same launches (same generation), a rewritten verdict + menu (a new pair, a new written_utc)
    for name in ("probe_verdict.json", "allowed_pairs.json"):
        d = json.loads((world.record / name).read_text()); d["written_utc"] = "2026-09-30T16:41:00Z"
        if name == "allowed_pairs.json":
            d["pairs"] = [{"rank": 4, "set": "wo"}, {"rank": 16, "set": "down"}]
        (world.record / name).write_text(json.dumps(d))
    gen2 = _stamp(world)
    assert gen2 == f1["generation"]                                                     # the generation did NOT move
    r2 = world.run()
    out = r2.stdout + r2.stderr
    assert r2.returncode == 0 and "skipping" not in out, out
    assert "the verdict was rewritten" in out and "archived under" in out and "READY on cluster_a" in out
    f2 = json.loads((world.state / "probe_fetch.json").read_text())
    assert f2["menu_sha256"] != f1["menu_sha256"] and f2["written_utc"] == "2026-09-30T16:41:00Z" and f2["generation"] == gen2
    assert len(json.loads((world.state / "allowed_pairs.json").read_text())["pairs"]) == 2
    marker = (world.tmp / "state" / MODEL / "03_hpo" / "DONE.lora.probe").read_text()
    assert f"menu_sha256={f2['menu_sha256']}" in marker and "written_utc=2026-09-30T16:41:00Z" in marker
    # a third run: current (identity equal) — skip, nothing archived further
    r3 = world.run()
    assert r3.returncode == 0 and "skipping" in r3.stdout and len(sorted((world.state / "superseded").iterdir())) == 1


def test_a_legacy_fetch_record_over_an_identical_mirror_is_refreshed_in_place_not_re_fetched(world):
    """The live probe_fetch.json had no identity fields (and empty written_utc / probe_job_id): when the mirror IS the record, the fetch
    record is refreshed in place — no archive, no rsync, no site rewrite — and the run says so."""
    from test_front_door_probe_resume import _stamp
    world.lay(done=7, complete=True, menu=True, verdict=True); _stamp(world)
    r = world.run(); assert r.returncode == 0, r.stdout + r.stderr
    f = world.state / "probe_fetch.json"; d = json.loads(f.read_text())
    for k in ("verdict_sha256", "menu_sha256", "geometry_sha256", "parity_check", "parity_stamped_generation"):
        d.pop(k, None)
    d["written_utc"] = ""; d["probe_job_id"] = ""
    f.write_text(json.dumps(d))
    before = (world.state / "allowed_pairs.json").read_bytes(); n_ssh = len(world.ssh_log.read_text().splitlines())
    r2 = world.run()
    out = r2.stdout + r2.stderr
    assert r2.returncode == 0 and "refreshing the fetch record in place" in out and "skipping" in out, out
    assert not (world.state / "superseded").exists() and (world.state / "allowed_pairs.json").read_bytes() == before
    d2 = json.loads(f.read_text())
    assert len(d2["verdict_sha256"]) == 64 and d2["written_utc"] == "2026-09-30T02:00:00Z" and d2["probe_job_id"] == "11111111"
    assert not any("rsync" in ln for ln in world.ssh_log.read_text().splitlines()[n_ssh:])


def test_geometry_site_recover_refuses_a_mirror_that_predates_the_record(world):
    """A DONE probe whose SITE copy is missing recovers it from the mirror only when the mirror's geometry record IS the record's
    (one ssh read of the record's hashes); an older mirror refuses by name and copies nothing."""
    import subprocess
    from test_front_door_probe_resume import REPO as PR_REPO
    src = (PR_REPO / "reproduce" / "03_hpo.sh").read_text()
    frag = src[src.index("geometry_site_recover(){"):src.index("push(){")]
    mirror = world.tmp / "mirror"; mirror.mkdir(); (mirror / "geometry_record.json").write_text('{"v": 1}')
    site = world.tmp / "site2"; (site / "geometry").mkdir(parents=True)
    import hashlib
    msha = hashlib.sha256(b'{"v": 1}').hexdigest()
    def run(remote_geo_sha):
        script = (f"set -u; source {PR_REPO}/reproduce/lib/common.sh\nSISTEER_SITE_DIR={site}; MODEL={MODEL}; CL_ROOT=/shared/exp/reproduce/m/03_hpo/lora; CLUSTER=cluster_a\n"
                  f"probe_remote_identity(){{ printf '%s' 'aaaa bbbb {remote_geo_sha}'; }}\n"
                  f"geometry_to_site(){{ echo COPIED-FROM-$1; }}\n{frag}\ngeometry_site_recover {mirror}")
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    r = run("f" * 64)
    assert r.returncode == 2 and "mirror's geometry record predates the record's" in r.stderr and "fetch first" in r.stderr and "COPIED" not in r.stdout
    assert not (site / "geometry" / f"{MODEL}.json").exists()
    r = run("-")
    assert r.returncode == 2 and "holds no geometry_record.json" in r.stderr and "COPIED" not in r.stdout
    r = run(msha)
    assert r.returncode == 0 and f"COPIED-FROM-{mirror}" in r.stdout and "IS the record's" in r.stdout
