"""reproduce/03_hpo.sh --phase probe reads the serving probe's launch-list checkpoint (probe_progress.json) through
scripts/sdfbase/hpo_probe_status.py: a missing allowed_pairs.json is never "not started" when a partial record
exists — the phase reports INCOMPLETE + the successor job, or refuses to proceed; a fresh probe is submitted in
exactly one state (NOT_STARTED). Synthetic records only.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts" / "sdfbase") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))

import hpo_probe_status as hps  # noqa: E402

PARTIAL = {"version": 1, "written_utc": "2026-09-29T00:00:00.000Z", "model_key": "qwen35_4b", "plan_launches": 7,
           "n_done": 3, "done": ["L01", "L02", "L03"],
           "outstanding": [{"id": "L04", "kind": "servability", "L": 8192, "cap": 64},
                           {"id": "L05", "kind": "servability", "L": 49152, "cap": 64},
                           {"id": "L06", "kind": "memo", "L": 8192, "cap": 16},
                           {"id": "L07", "kind": "memo", "L": 49152, "cap": 16}],
           "decision_outstanding": ["L04", "L05"], "complete": False, "stopped_reason": "time_budget"}
COMPLETE = {**PARTIAL, "n_done": 7, "outstanding": [], "decision_outstanding": [], "complete": True, "stopped_reason": None}


def test_state_table_on_a_synthetic_partial_record():
    # menu presence alone is never READY any more: the checkpoint-only mode cannot verify the menu's
    # result generation — it says so and points at `--record <dir>` (tests/test_serving_probe_revocation.py pins that mode)
    assert hps.probe_state(None, allowed_present=True, job_active=False)[0] == hps.MENU_UNVERIFIED
    assert hps.probe_state(PARTIAL, allowed_present=True, job_active=False)[0] == hps.MENU_UNVERIFIED
    assert "--record" in hps.probe_state(PARTIAL, allowed_present=True, job_active=False)[1]
    assert hps.probe_state(None, allowed_present=False, job_active=False)[0] == hps.NOT_STARTED
    assert hps.probe_state(None, allowed_present=False, job_active=True)[0] == hps.RUNNING
    st, detail = hps.probe_state(PARTIAL, allowed_present=False, job_active=True)
    assert st == hps.INCOMPLETE_RUNNING and "3/7 launches done" in detail and "4 outstanding" in detail
    assert "2 decision-relevant: L04, L05" in detail and "stopped: time_budget" in detail and "successor" in detail
    st, detail = hps.probe_state(PARTIAL, allowed_present=False, job_active=False)
    assert st == hps.INCOMPLETE_STALLED and "NO successor job" in detail
    st, detail = hps.probe_state(COMPLETE, allowed_present=False, job_active=False)
    assert st == hps.VERDICT_REFUSED and "refused certification" in detail
    assert hps.SUBMIT_STATES == (hps.NOT_STARTED,)


def test_cli_reads_the_record_from_stdin(capsys):
    assert hps.main(["--job-active", "0"], stdin=io.StringIO(json.dumps(PARTIAL))) == 0
    st, detail = capsys.readouterr().out.rstrip("\n").split("\t")
    assert st == hps.INCOMPLETE_STALLED and "3/7" in detail
    assert hps.main(["--job-active", "1"], stdin=io.StringIO(json.dumps(PARTIAL))) == 0
    assert capsys.readouterr().out.startswith("INCOMPLETE_RUNNING\t")
    assert hps.main(["--job-active", "0"], stdin=io.StringIO("")) == 0
    assert capsys.readouterr().out.startswith("NOT_STARTED\t")
    assert hps.main(["--allowed-present", "1"], stdin=io.StringIO("")) == 0
    assert capsys.readouterr().out.startswith("MENU_UNVERIFIED\t")          # never READY from the checkpoint-only mode
    assert hps.main(["--job-active", "0"], stdin=io.StringIO("{not json")) == 0     # a torn checkpoint is a partial record
    assert capsys.readouterr().out.startswith("INCOMPLETE_STALLED\t")


def test_03_hpo_submits_a_fresh_probe_only_when_nothing_is_recorded_and_resumes_a_stalled_record():
    """The probe phase asks hpo_probe_status.py --record ON the cluster (never file presence) and acts on its verdict alone: the
    ONE fresh `submit "$(unit probe)"` sits under the NOT_STARTED arm; INCOMPLETE_STALLED RESUMES the record through
    probe_resume (the attempt-scoped unit `$(unit probe).a<N>`, never a fresh probe over a partial record);
    MENU_STALE and VERDICT_REFUSED refuse by name with no submit; the running states wait. The execution-form pins are in
    tests/test_front_door_probe_resume.py."""
    src = (REPO / "reproduce" / "03_hpo.sh").read_text()
    probe = src[src.index("# ================================================================================================================ probe"):
                src.index("# ================================================================================================================ recon")]
    assert "hpo_probe_status.py" in src[src.index("probe_status(){"):src.index("probe_fetch_ready(){")] and "--record" in src
    assert 'cluster_has "$CL_ROOT/allowed_pairs.json"' not in probe and 'cluster_has "$CL_ROOT/probe_progress.json"' not in probe
    # the fresh probe: SUBMITTED from two arms — NOT_STARTED, and FAILED_START's guarded recovery of a plan-less record under
    # --resubmit (finding 3a); the first is the NOT_STARTED arm's (the parallel test pins its line shape)
    assert probe.count('submit "$(unit probe)"') == 2
    j = probe.index("FAILED_START)", probe.index("case \"$PSTATE\" in"))
    assert 'submit "$(unit probe)"' in probe[j:probe.index("VERDICT_REFUSED)", j)] and "probe_recover_planless" in probe[j:probe.index("VERDICT_REFUSED)", j)]
    i_case, i_submit = probe.index("case \"$PSTATE\" in"), probe.index("submit \"$(unit probe)\"")
    i_not_started, i_running = probe.index("NOT_STARTED)", i_case), probe.index("RUNNING|INCOMPLETE_RUNNING)", i_case)   # the arms, not the prose above the case
    assert i_case < i_not_started < i_submit < i_running                    # … inside the NOT_STARTED arm
    stalled = probe[probe.index("INCOMPLETE_STALLED)", i_case):]                    # the arm, not the prose above the case
    stalled = stalled[:stalled.index(";;")]
    assert 'probe_resume "$PDETAIL"' in stalled and 'submit "$(unit' not in stalled and "repro_fatal" not in stalled
    resume = src[src.index("probe_resume(){"):src.index("# ================================================================================================================ probe")]
    assert 'submit "$punit.a$R_ATTEMPT"' in resume and "hpo_probe_resume.py" in resume and "--kind other" in resume
    assert "NOT submitting a fresh probe over the partial record" in resume
    j = probe.index("VERDICT_REFUSED)", i_case)
    arm_body = probe[j:probe.index(";;", j)]
    assert "repro_fatal" in arm_body and 'submit "$(unit' not in arm_body and "probe_resume" not in arm_body
    j = probe.index("MENU_STALE)", i_case)                                   # handled like INCOMPLETE_*: never fetched, resumed when no job runs
    arm_body = probe[j:probe.index(";;", j)]
    assert 'probe_resume "$PDETAIL"' in arm_body and 'if [ "$PROBE_ACTIVE" = 1 ]' in arm_body
    assert "probe_fetch_ready" not in arm_body and 'fetch "$CL_ROOT"' not in arm_body and "mark probe" not in arm_body
    ready = probe[probe.index("READY)", i_case):]                             # the arm, not the prose above the case
    ready = ready[:ready.index(";;")]
    assert "probe_fetch_ready" in ready and "fetch " not in ready and "mark probe" not in ready       # the fetch + mark live in the helper, after the generation checks
    helper = src[src.index("probe_fetch_ready(){"):src.index("probe_fresh_binding(){")]
    # rsync into the staging dir, verify there, the fetch record FIRST, expose, the site copy, the marker LAST
    assert (helper.index('fetch "$CL_ROOT" "$stage"') < helper.index("probe_write_fetch_record ") < helper.index("probe_expose_staged ")
            < helper.index('geometry_to_site "$LOCAL_ROOT"') < helper.index("probe_mark_fetched "))
    assert 'fetch "$CL_ROOT" "$LOCAL_ROOT"' not in helper and "is NEWER than" in helper       # nothing rsyncs straight into the mirror; the newer-mirror guard
    writer = src[src.index("probe_fetch_fields(){"):src.index("probe_status(){")]
    assert "mark probe " in writer and "generation=$FR_GEN" in writer and "verdict_sha256=$FR_V" in writer   # The fetch record carries the verdict identity
    assert '"cluster\\": \\"${PROBE_CLUSTER:-$CLUSTER}' in writer                                          # and the record's HOME
    # a DONE marker is revalidated on every probe invocation (no `done_or_skip probe` short-circuit), and every
    # phase that consumes the menu revalidates the local copy first — recon (lora), design, trials, refine, pick, confirm-pick
    assert "done_or_skip probe" not in src and 'if repro_done "$MODEL" "$STAGE" "$CLASS.probe"; then' in probe
    for ph in ("recon", "design", "trials", "refine", "pick", "confirm-pick"):
        assert f"probe_revalidate_local {ph}\n" in src, ph
    reval = src[src.index("probe_revalidate_local(){"):src.index("# ================================================================================================================ probe")]
    assert "probe_archive_local" in reval and "repro_fatal" in reval and "superseded" in src[src.index("probe_archive_local(){"):src.index("probe_revalidate_local(){")]
