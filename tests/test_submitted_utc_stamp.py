"""Every unit's job env carries `SISTEER_UNIT_SUBMITTED_UTC` — the submission's own UTC clock — and the attempt records the same
value.

The native standup's cross-cluster staging-stamp fence (`stamp_not_before`) refuses a geometry stamp staged BEFORE the job's
submission on a cluster that cannot reach the probe record; nothing exported the bound, so the fence never applied. The funnel
now exports it on EVERY sbatch (the same prefix SISTEER_SHARED_ROOT rides — never `--export=`), records it on the attempt and in
the spec, and the workload-spec identity drops it, so a re-run at a later clock is never a "changed workload". Execution forms:
the funnel against the fake sbatch (its recorded environment), a REAL `06_eval_waves.sh` submission (the wave harness), and the
REAL `03_hpo.sh` recon submission's composed sbatch line (its dry run prints the ssh command the funnel composes, prefix included).
"""
from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, VENV_PY, job_ids, run_stage  # noqa: E402
from test_submit_twin import SHA, World  # noqa: E402
import test_front_door_wave_integration as WI  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import models as M  # noqa: E402
import stage as ST  # noqa: E402
import submit_twin as TW  # noqa: E402

ISO_Z = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
VAR = TW.SUBMITTED_UTC_ENV
# the REAL 06 submission is of the model's BASELINE wave — `u0` when the table declares it (the phased recipe), else the single-phase
# recipe's one steered wave `s<N>` (the demo key: s1) — the stage's own helper (models.baseline_view), never a literal, the same pattern
# the Petri manifests use: 06 refuses a view that is not one of the model's phase views by name
VIEW = M.baseline_view(WI.MODEL, str(REPRO / "models.yaml"))


def _env_of(fake: Path, jid) -> dict:
    return dict(l.split("=", 1) for l in (fake / "jobs" / f"{jid}.env").read_text().splitlines() if "=" in l)


def test_the_funnel_exports_the_stamp_into_the_job_env_and_records_it_on_the_attempt(tmp_path):
    w = World(tmp_path)
    r = w.run("submit", "--unit", "mmlu", "--lwid", "m/u0/mmlu/mmlu", "--script", "scripts/sdfbase/full22/mmlu_full.sbatch", "--name", "mmluF-m-u0",
              "--args", f"m u0 {SHA}", "--clusters", "cluster_a", "--wave-sha", SHA)
    assert r.returncode == 0 and "submitted cluster_a " in r.stdout, r.stdout + r.stderr
    (jid,) = job_ids(w.fake["cluster_a"])
    env = _env_of(w.fake["cluster_a"], jid)
    assert VAR == "SISTEER_UNIT_SUBMITTED_UTC" and ISO_Z.match(env[VAR]), env.get(VAR)
    (att,) = [e for e in w.ledger() if e["event"] == "attempt"]
    assert att["submitted_utc"] == env[VAR] == att["spec"]["submitted_utc"]
    assert "--export" not in (w.fake["cluster_a"] / "jobs" / f"{jid}.argv").read_text()
    # the stamp is a per-submission clock, not a workload knob: the identity drops it, a second attempt at a later clock is the SAME spec
    a = ST.spec_identity({"script": "s", "args": [], "env": {VAR: "2026-01-01T00:00:00Z", "F22_REGISTRY_EVALS": "mmlu"}})
    b = ST.spec_identity({"script": "s", "args": [], "env": {VAR: "2026-01-02T00:00:00Z", "F22_REGISTRY_EVALS": "mmlu"}})
    assert a == b and VAR not in a["env"] and a["env"] == {"F22_REGISTRY_EVALS": "mmlu"}
    assert ST.spec_diff(a, b) == []


def test_a_real_06_submission_carries_the_stamp_equal_to_its_attempts(tmp_path):
    w = WI.World(tmp_path)
    r = w.stage("05_stage_roster.sh", "--model", WI.MODEL, "--step", "freeze")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    r = w.stage("06_eval_waves.sh", "--model", WI.MODEL, "--view", VIEW, "--family", "mmlu")
    assert r.returncode == 0 and any(ln.startswith("submitted ") for ln in r.stdout.splitlines()), r.stdout[-1500:] + r.stderr[-1500:]
    atts = [a for a in w.attempts() if a.get("event") == "attempt"]
    assert atts
    for a in atts:
        env = _env_of(w.fake[a["cluster"]], a["job_id"])
        assert ISO_Z.match(env[VAR]) and env[VAR] == a["submitted_utc"] == a["spec"]["submitted_utc"], (a["unit"], env.get(VAR), a.get("submitted_utc"))


def test_a_real_03_probe_submission_carries_the_stamp_equal_to_its_attempts(tmp_path):
    """03_hpo.sh --phase probe against the probe-resume harness (fake ssh, rsync, sbatch): the ONE sbatch it composes through the funnel
    carries the stamp in its environment, equal to the attempt the funnel recorded."""
    import test_front_door_probe_resume as PR
    if PR.REAL_RSYNC is None:
        pytest.skip("no rsync on this host")
    w = PR.World(tmp_path); w.lay()
    r = w.run()
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    calls = w.sbatch_calls()
    assert len(calls) == 1, r.stdout[-1500:]
    stamp = calls[0]["env"].get(VAR)
    assert stamp and ISO_Z.match(stamp), calls[0]["env"]
    atts = [json.loads(l) for l in w.ledger.with_name("attempts.jsonl").read_text().splitlines() if l.strip()]
    atts = [a for a in atts if a.get("event") == "attempt"]
    assert len(atts) == 1 and atts[0]["submitted_utc"] == stamp == atts[0]["spec"]["submitted_utc"], atts
    assert "--export" not in " ".join(calls[0]["argv"])
