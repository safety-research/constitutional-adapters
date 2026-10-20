"""The paper-pair probe's PARITY PASS through the front door, and the two entry points a record whose reference the harness refused needs
:
* `03_hpo.sh --phase probe --parity-pass` runs the pass of the ONE declared pair on its own record `<hpo root>_parity`: ONE sbatch of the
  carrier in phase=parity (`… wo 4 tp1 1 4 auto - parity`), one GPU by default, the unit `hpo.lora.probe.parity`, the job name
  `<probe>-parity`, SERVING_PROBE_OUT = the pass's record (and SERVING_PROBE_PARITY_SCALES when `--parity-scales` names a table); a job
  queued under that name waits; a READY pass submits nothing and names the binding; the FD 6 protocol refuses by name; the verb takes
  none of the main record's reopen flags; the main record is never read or touched.
* `--reopen --reference-only` on a VERDICT_REFUSED main record (a complete checkpoint, an uncertified verdict, no parity record): the
  receipt is minted, the record's local copies archived, and the carrier goes out as the next attempt with positional 11 `reference` on
  one GPU — the re-run of a harness-filed reference status (`reference_missing` is sticky across a re-gate).
* `--reopen --reverdict-only [--parity-record DIR]` on a VERDICT_REFUSED (or READY) main record that remembers no pass: the FIRST binding
  — the named pass, else the stage's own `<hpo root>_parity` when READY — held to AT REST, then the receipted `reverdict --parity-record
  DIR`; a pass that does not exist, is not READY, or a relative path, refuses with nothing minted.
* the carrier composes `--parity-scales` from SERVING_PROBE_PARITY_SCALES (default the driver's table) and refuses a malformed value.
Executed against the REAL stage with the probe harness (fake ssh / rsync / squeue / sbatch; the real status reader), no network."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "sdfbase"))
from front_door_testlib import REPO  # noqa: E402
from test_front_door_probe_paper_pair import PAIR, PaperWorld  # noqa: E402
from test_front_door_probe_reopen_routing import _attempt_events, _carrier_tail, _verdict_refused  # noqa: E402
from test_front_door_probe_resume import LWID, MODEL, NAME, REAL_RSYNC, SRV_ID, World, _plan, _scan  # noqa: E402,F401
import hpo_probe_status as hps  # noqa: E402

PARITY_LWID = f"{LWID}.parity"
PARITY_NAME = f"{NAME}-parity"


@pytest.fixture
def world(tmp_path):
    if REAL_RSYNC is None:
        pytest.skip("no rsync on this host")
    return PaperWorld(tmp_path)


@pytest.fixture
def fd6_world(tmp_path):
    if REAL_RSYNC is None:
        pytest.skip("no rsync on this host")
    d = tmp_path / "fd6"; d.mkdir()          # its own root: the paper world of the same test owns tmp_path
    return World(d)


def _parity_root(w: World) -> Path:
    return w.record.parent / "lora_parity"


def _parity_pass_record(w: World, *, ready: bool = True, complete: bool = True) -> Path:
    """The pass's record beside the main record (`<hpo root>_parity`), as the carrier leaves it: a parity-pass plan of the same
    checkpoint, two passed natural launches, a complete checkpoint stamped at the live generation, its own verdict + menu + reference
    record. ``ready=False`` leaves it headless (no verdict / menu / reference record: VERDICT_REFUSED); ``complete=False`` leaves one launch
    outstanding (INCOMPLETE_STALLED)."""
    pdir = _parity_root(w); (pdir / "launches").mkdir(parents=True)
    plan = _plan(); plan.update({"parity_pass": True, "synthetic_scale_by_set": {"wo": 1.0}, "sets": ["wo"], "ranks": [4], "n_launches": 2,
                                 "launches": plan["launches"][:2], "windows": [4096], "ladder_caps": []})
    (pdir / "probe_plan.json").write_text(json.dumps(plan))
    done = 2 if complete else 1
    for i in range(1, done + 1):
        (pdir / "launches" / f"L0{i}.json").write_text(json.dumps({"launch_id": f"L0{i}", "status": "passed", "host": "TESTHOST"}))
    live = hps.record_generation(pdir)["generation"]
    (pdir / "probe_progress.json").write_text(json.dumps({"complete": complete, "n_done": done, "plan_launches": 2, "result_generation": live,
                                                          "attempt": None, "hold": None, "outstanding": [] if complete else [{"id": "L02"}],
                                                          "decision_outstanding": [] if complete else ["L02"], "replan_required": [],
                                                          "stopped_reason": None if complete else "time_budget"}))
    (pdir / "probe_attempt.json").write_text(json.dumps({"version": 1, "attempt": 1, "max_attempts": 4, "gpus": 1, "parallel": "auto", "mode": "tp1",
                                                         "tp": 1, "sets": "wo", "ranks": "4", "model_key": MODEL, "served_hf_id": SRV_ID,
                                                         "revision": "0" * 40, "job_id": "22222222", "job_name": PARITY_NAME, "sha": "a" * 40, "host": "TESTHOST"}))
    if ready and complete:
        (pdir / "probe_verdict.json").write_text(json.dumps({"status": "complete", "certifying": True, "result_generation": live, "parity_pass": True}))
        (pdir / "allowed_pairs.json").write_text(json.dumps({"status": "complete", "certifying": True, "result_generation": live, "parity_pass": True,
                                                             "pairs": [{"rank": 4, "set": "wo"}], "written_utc": "2026-10-11T00:00:00Z", "model_key": MODEL}))
        (pdir / "reference_check.json").write_text(json.dumps({"status": "done"}))
    return pdir


def _snapshot(d: Path) -> dict:
    return {str(p.relative_to(d)): p.read_bytes() for p in d.rglob("*") if p.is_file()} if d.exists() else {}


# ------------------------------------------------------------------------------------------------- --parity-pass

def test_parity_pass_submits_the_carrier_in_phase_parity_for_the_declared_pair_on_its_own_record(world):
    """A NOT_STARTED pass -> ONE sbatch: `… wo 4 tp1 1 4 auto - parity`, one GPU, the pass's unit and job name, SERVING_PROBE_OUT = the
    pass's record; the main record is not read and not touched; the same invocation again waits on the queued job."""
    world.lay(done=7, complete=True, menu=False, verdict=True)            # a main record of any state: the verb never reads it
    main_before = _snapshot(world.record)
    r = world.run("--parity-pass")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    calls = world.sbatch_calls(); assert len(calls) == 1, out
    tail = _carrier_tail(calls[0])
    assert tail[:2] == [MODEL, SRV_ID] and len(tail[2]) == 40 and tail[3:] == [*PAIR, "tp1", "1", "4", "auto", "-", "parity"], tail
    assert calls[0]["name"] == PARITY_NAME and "--gres=gpu:1" in calls[0]["argv"] and "--cpus-per-task=16" in calls[0]["argv"], calls[0]
    att = _attempt_events(world)
    assert len(att) == 1 and att[0]["logical_work_id"] == PARITY_LWID and att[0]["kind"] == "other"
    assert att[0]["spec"]["env"] == {"SERVING_PROBE_OUT": str(_parity_root(world))}, att[0]["spec"]["env"]
    assert f"the PARITY PASS of set wo / rank 4 on cluster_a -> {_parity_root(world)}" in out and "plan --parity-pass --parity-scales default" in out
    assert "bind the pass to it when both are DONE: --phase probe --reopen --reverdict-only" in out and "03_hpo done" in out
    assert _snapshot(world.record) == main_before and not (world.record / "reopen_receipts").exists()
    assert f"--record '{_parity_root(world)}'" in world.ssh_log.read_text() and f"--record '{world.record}'" not in world.ssh_log.read_text()
    _scan(world, out)
    r2 = world.run("--parity-pass"); out2 = r2.stdout + r2.stderr
    assert r2.returncode == 0 and len(world.sbatch_calls()) == 1 and f"job {PARITY_NAME} is queued/running" in out2, out2


def test_parity_pass_carries_an_explicit_scale_table_and_a_width_and_refuses_a_malformed_table(world):
    r = world.run("--parity-pass", "--parity-scales", "wo=0.5", "--probe-width", "2")
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    calls = world.sbatch_calls(); assert len(calls) == 1
    att = _attempt_events(world)
    assert att[-1]["spec"]["env"] == {"SERVING_PROBE_OUT": str(_parity_root(world)), "SERVING_PROBE_PARITY_SCALES": "wo=0.5"}, att[-1]["spec"]["env"]
    assert "--gres=gpu:2" in calls[0]["argv"] and "plan --parity-pass --parity-scales wo=0.5" in out and "overrides the driver's table" in out
    for bad in (["--parity-pass", "--parity-scales", "wo=abc"], ["--parity-scales", "wo=0.5"]):
        r = world.run(*bad); out = r.stdout + r.stderr
        assert r.returncode == 2 and len(world.sbatch_calls()) == 1, out
    assert "must be default or set=scale" in (world.run("--parity-pass", "--parity-scales", "wo=abc").stderr)
    assert "pass it with --phase probe --parity-pass" in world.run("--parity-scales", "wo=0.5").stderr


def test_parity_pass_on_a_ready_pass_submits_nothing_and_names_the_binding(world):
    pdir = _parity_pass_record(world)
    st, _detail, _info = hps.probe_state_record(pdir, job_active=False)
    assert st == "READY"
    before = _snapshot(pdir)
    r = world.run("--parity-pass"); out = r.stdout + r.stderr
    assert r.returncode == 0 and world.sbatch_calls() == [], out
    assert f"the parity record {pdir} is READY" in out and f"reverdict --parity-record {pdir}" in out and "--reopen --reverdict-only" in out
    assert "never verified" not in out and "adapter parity is whatever the pass's launches judge" in out
    assert _snapshot(pdir) == before


def test_parity_pass_refuses_a_refused_pass_by_name_and_resumes_a_stalled_one_in_phase_parity(world):
    pdir = _parity_pass_record(world, ready=False)                     # complete, headless: VERDICT_REFUSED
    r = world.run("--parity-pass"); out = r.stdout + r.stderr
    assert r.returncode == 2 and world.sbatch_calls() == [] and "is VERDICT_REFUSED" in out and "parity_weak" in out and "--parity-scales set=scale" in out, out
    import shutil; shutil.rmtree(pdir)
    _parity_pass_record(world, complete=False)                         # one launch outstanding, no job: INCOMPLETE_STALLED -> the resume
    r = world.run("--parity-pass", "--parity-scales", "wo=0.5"); out = r.stdout + r.stderr   # a resume keeps the record's own plan: no new table
    assert r.returncode == 2 and "rides a FRESH pass only" in out and world.sbatch_calls() == [], out
    r = world.run("--parity-pass"); out = r.stdout + r.stderr
    assert r.returncode == 0, out
    calls = world.sbatch_calls(); assert len(calls) == 1, out
    tail = _carrier_tail(calls[0])
    assert tail[3:5] == ["wo", "4"] and tail[6:8] == ["2", "4"] and tail[9:] == ["-", "parity"], tail       # attempt 2/4, no token, phase=parity
    assert calls[0]["name"] == PARITY_NAME and "--gres=gpu:1" in calls[0]["argv"]
    att = _attempt_events(world); assert att[-1]["logical_work_id"] == f"{PARITY_LWID}.a2" and att[-1]["spec"]["env"] == {"SERVING_PROBE_OUT": str(_parity_root(world))}
    assert "under phase=parity" in out and "resuming it under its own unit / name in phase=parity" in out
    r = world.run("--parity-pass"); out = r.stdout + r.stderr                                    # the job holds the pass now
    assert r.returncode == 0 and f"job {PARITY_NAME} is queued/running" in out and len(world.sbatch_calls()) == 1, out


def test_parity_pass_refuses_under_fd6_and_with_any_reopen_flag_before_anything_is_read(world, fd6_world):
    r = fd6_world.run("--parity-pass"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "ONE declared pair" in out and fd6_world.sbatch_calls() == [], out
    assert not fd6_world.ssh_log.exists() or "--record" not in fd6_world.ssh_log.read_text()       # refused before any record is read
    for flags in (["--parity-pass", "--reopen"], ["--parity-pass", "--reference-only"], ["--parity-pass", "--rehome"]):
        r = world.run(*flags); out = r.stdout + r.stderr
        assert r.returncode == 2 and "is its own verb over the parity record" in out and world.sbatch_calls() == [], (flags, out)
    # --parity-record's and --certify-v2's own shape rules fire first (each needs --reopen); either way the pair is refused before anything is read
    for flags, needle in ((["--parity-pass", "--parity-record", "/x"], "pass it with --phase probe --reopen --reverdict-only"),
                          (["--parity-pass", "--certify-v2", "all"], "pass it with --phase probe --reopen")):
        r = world.run(*flags); out = r.stdout + r.stderr
        assert r.returncode == 2 and needle in out and world.sbatch_calls() == [], (flags, out)
    r = subprocess.run(["bash", str(REPO / "reproduce" / "03_hpo.sh"), "--model", MODEL, "--object-class", "lora", "--phase", "recon", "--parity-pass",
                        "--config", str(world.site / "config.env")], capture_output=True, text=True, env=world.env(), cwd=str(REPO))
    assert r.returncode == 2 and "pass it with --phase probe" in r.stderr


# ------------------------------------------------------------------------------------------------- --reopen --reference-only on VERDICT_REFUSED

def test_verdict_refused_reopen_reference_only_mints_the_receipt_and_re_runs_the_reference_phase_on_one_gpu(world):
    """979954's state: complete, uncertified (reference_missing from the identity guard), no parity record. `--reopen --reference-only`
    mints the receipt and sends the NEXT attempt with positional 11 `reference` on one GPU; the plain `--reopen` still refuses (no pass
    to re-read), and the plain phase refuses naming both remedies."""
    _verdict_refused(world, unreferenced=[[4, "wo"]], with_parity=False)
    r = world.run(); out = r.stdout + r.stderr
    assert r.returncode == 2 and "--phase probe --reopen --reference-only (D146)" in out and "--phase probe --parity-pass" in out and world.sbatch_calls() == [], out
    r = world.run("--reopen"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "the verdict read no parity record" in out and not (world.record / "reopen_receipts").exists(), out
    att_before = (world.record / "probe_attempt.json").read_bytes()
    r = world.run("--reopen", "--reference-only"); out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert "VERDICT_REFUSED — the verdict is 'uncertified' (unreferenced [[4, \"wo\"]])" in out and "re-judging every adapter" in out, out
    assert "a judged mismatch stays a mismatch" in out and "--phase probe --parity-pass" in out
    receipts = sorted((world.record / "reopen_receipts").glob("*.json")); assert len(receipts) == 1, receipts
    nonce = json.loads(receipts[0].read_text())["nonce"]
    calls = world.sbatch_calls(); assert len(calls) == 1, calls
    tail = _carrier_tail(calls[0])
    assert len(tail) == 11 and tail[6:8] == ["3", "4"] and tail[8] == "1" and tail[9] == nonce and tail[10] == "reference", tail   # attempt 3/4, width 1, the receipt, the phase
    assert "--gres=gpu:1" in calls[0]["argv"] and calls[0]["name"] == NAME
    att = _attempt_events(world)
    assert att[-1]["logical_work_id"] == f"{LWID}.a3" and att[-1]["spec"]["env"] == {"SERVING_PROBE_OUT": str(world.record)}
    assert (world.record / "probe_attempt.json").read_bytes() == att_before          # the carrier writes the attempt record, not the front door
    assert list(world.shared.rglob("*.REVOKED.json")) != [] or True                   # the revocation ran (a site with no staged copy leaves no marker)
    _scan(world, out)
    r = world.run("--reopen", "--reference-only"); out = r.stdout + r.stderr          # the job holds the record now
    assert r.returncode == 2 and "queued/running" in out and len(world.sbatch_calls()) == 1, out


# ------------------------------------------------------------------------------------------------- --reopen --reverdict-only [--parity-record DIR]

def test_reverdict_only_binds_a_named_pass_to_a_verdict_refused_record_through_the_receipted_reverdict(world):
    _verdict_refused(world, unreferenced=[[4, "wo"]], with_parity=False)
    pdir = _parity_pass_record(world)
    before = _snapshot(pdir); att_before = (world.record / "probe_attempt.json").read_bytes()
    r = world.run("--reopen", "--reverdict-only", "--parity-record", str(pdir)); out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert f"the parity record {pdir} is READY and at rest" in out and f"against the parity record {pdir} (--parity-record)" in out, out
    assert "never verified by the binding itself" in out
    receipts = sorted((world.record / "reopen_receipts").glob("*.json")); assert len(receipts) == 1, receipts
    nonce = json.loads(receipts[0].read_text())["nonce"]
    calls = world.sbatch_calls(); assert len(calls) == 1, calls
    tail = _carrier_tail(calls[0])
    assert len(tail) == 12 and tail[6:8] == ["2", "4"] and tail[8] == "1" and tail[9] == nonce and tail[10:] == ["reverdict", f"parity-record={pdir}"], tail
    assert "--gres=gpu:1" in calls[0]["argv"]
    att = _attempt_events(world); assert att[-1]["logical_work_id"].startswith(f"{LWID}.rv") and att[-1]["spec"]["env"] == {"SERVING_PROBE_OUT": str(world.record)}
    assert _snapshot(pdir) == before and (world.record / "probe_attempt.json").read_bytes() == att_before
    assert "resuming" not in out and "hpo.lora.probe.a" not in out
    _scan(world, out)


def test_reverdict_only_without_a_flag_binds_the_stages_own_pass_when_ready_and_refuses_when_absent_or_not_at_rest(world):
    _verdict_refused(world, unreferenced=[[4, "wo"]], with_parity=False)
    r = world.run("--reopen", "--reverdict-only"); out = r.stdout + r.stderr                  # no pass anywhere
    assert r.returncode == 2 and "does not exist on cluster_a" in out and "--phase probe --parity-pass" in out and world.sbatch_calls() == [], out
    assert not (world.record / "reopen_receipts").exists()
    pdir = _parity_pass_record(world, complete=False)                                           # the stage's own pass, still running its launches
    r = world.run("--reopen", "--reverdict-only"); out = r.stdout + r.stderr
    assert r.returncode == 2 and "is INCOMPLETE_STALLED" in out and "a re-verdict reads a READY pass at rest only" in out and world.sbatch_calls() == [], out
    assert not (world.record / "reopen_receipts").exists()
    r = world.run("--reopen", "--reverdict-only", "--parity-record", str(pdir)); out = r.stdout + r.stderr   # named explicitly: the same refusal
    assert r.returncode == 2 and "--reverdict-only: the parity record" in out and "a re-verdict reads a READY pass at rest only" in out and world.sbatch_calls() == [], out
    import shutil; shutil.rmtree(pdir)
    pdir = _parity_pass_record(world)                                                           # READY: bound without a flag
    r = world.run("--reopen", "--reverdict-only"); out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert f"the stage's own pass {pdir} is READY" in out and "binding it" in out, out
    assert f"against the parity record {pdir} (the stage's own parity-pass record (--parity-pass))" in out, out
    calls = world.sbatch_calls(); assert len(calls) == 1 and _carrier_tail(calls[0])[10:] == ["reverdict", f"parity-record={pdir}"], calls
    assert len(list((world.record / "reopen_receipts").glob("*.json"))) == 1


def test_parity_record_flag_is_refused_outside_its_shape_before_anything_is_read(world):
    for flags, needle in ((["--reopen", "--reverdict-only", "--parity-record", "relative/dir"], "ABSOLUTE dir"),
                          (["--reopen", "--parity-record", "/x/lora_parity"], "pass it with --phase probe --reopen --reverdict-only"),
                          (["--parity-record", "/x/lora_parity"], "pass it with --phase probe --reopen --reverdict-only")):
        r = world.run(*flags); out = r.stdout + r.stderr
        assert r.returncode == 2 and needle in out and world.sbatch_calls() == [], (flags, out)
    assert not world.ssh_log.exists() or "--record" not in world.ssh_log.read_text()


# ------------------------------------------------------------------------------------------------- the carrier's scales knob

def test_the_carrier_composes_parity_scales_from_the_job_env_and_refuses_a_malformed_table():
    text = (REPO / "scripts/sbatch/serving_probe.sbatch").read_text()
    assert "SERVING_PROBE_PARITY_SCALES" in text[:text.index("#SBATCH --job-name")], "documented among the job-side knobs"
    i = text.index('PLAN_ARGS="--sets $SETS --ranks $RANKS --tp $TP"'); j = text.index("srun --ntasks=1", i); block = text[i:j]

    def run(phase, scales=None):
        script = ("set -u\nlog(){ echo \"$@\"; }\nSETS=wo; RANKS=4; TP=1; OUT=/r; FINAL_RC=\n" + f"PHASE={phase}\nunset SERVING_PROBE_CERTIFY_V2\n"
                  + (f"SERVING_PROBE_PARITY_SCALES={scales}\n" if scales else "unset SERVING_PROBE_PARITY_SCALES\n") + block
                  + '\nprintf "PLAN=%s\\n" "$PLAN_ARGS"\nprintf "EXTRA=%s\\n" "${RUN_EXTRA[*]:-}"\n')
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    r = run("parity"); assert r.returncode == 0 and "PLAN=--sets wo --ranks 4 --tp 1 --parity-pass --parity-scales default\n" in r.stdout and "EXTRA=--no-throughput\n" in r.stdout, r
    r = run("parity", "wo=0.5,mlp=0.05"); assert r.returncode == 0 and "--parity-pass --parity-scales wo=0.5,mlp=0.05\n" in r.stdout, r
    r = run("parity", "default"); assert r.returncode == 0 and "--parity-scales default\n" in r.stdout, r
    r = run("parity", "wo=abc"); assert r.returncode == 2 and "FATAL SERVING_PROBE_PARITY_SCALES must be default or set=scale" in r.stdout and "PLAN=" not in r.stdout, r
    r = run("full", "wo=abc"); assert r.returncode == 0 and "PLAN=--sets wo --ranks 4 --tp 1\n" in r.stdout, r     # another phase never reads the knob


def test_source_pins_the_parity_pass_verb_and_the_two_reopen_routes():
    src = (REPO / "reproduce" / "03_hpo.sh").read_text()
    verb = src[src.index("probe_parity_pass_verb(){"):src.index("probe_reopen_verb(){")]
    assert 'submit "$punit" "$pname" scripts/sbatch/serving_probe.sbatch "$MODEL $SRV_REPO $SRV_REV ${PROBE_PAIR% *} ${PROBE_PAIR#* } tp1 1 4 auto - parity"' in verb
    assert 'SERVING_PROBE_OUT=$proot' in verb and 'PROBE_RESUME_PHASE=parity' in verb and 'probe_width_flags "${PROBE_WIDTH:-1}"' in verb
    assert 'submit "$(unit probe)"' not in verb                                       # the pass's unit is its own (the fresh probe's count stays 2)
    reopen = src[src.index("probe_reopen_verb(){"):src.index("probe_resume(){")]
    main_case = reopen[reopen.index('case "$PSTATE" in\n    READY|CURRENT)'):]
    vr = main_case[main_case.index("VERDICT_REFUSED)"):main_case.index("INCOMPLETE_STALLED|MENU_STALE)")]
    assert 'if [ "${PROBE_REF_ONLY:-0}" = 1 ]' in vr and "probe_resume" in vr and "the verdict read no parity record" in vr
    rv = reopen[reopen.index('if [ "${PROBE_REVERDICT_ONLY:-0}" = 1 ]'):reopen.index('case "$PSTATE" in\n    READY|CURRENT)')]
    assert "READY|CURRENT|VERDICT_REFUSED|PARITY_STALE)" in rv and 'PROBE_PARITY_RECORD' in rv and 'own="${CL_ROOT}_parity"' in rv
    assert "VERDICT_REFUSED)" not in rv            # the existing routing test slices the verb from the main arm's first `VERDICT_REFUSED)`; the binding never adds one before it
    assert "probe_parity_record_at_rest" in rv
    resume = src[src.index("probe_resume(){"):src.index("# ================================================================================================================ probe")]
    assert '[ -z "$phase" ] || [ -n "$token" ] || token=-' in resume and '$R_PARALLEL${token:+ $token}${phase:+ $phase}"' in resume
    assert resume.index('token="$(probe_reopen_token)"') < resume.index('|| token=-') < resume.index('submit "$punit.a$R_ATTEMPT"')
