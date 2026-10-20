"""A unit that runs AFTER another (the BoN climb after the 512 store the static -3 job writes): the plan row carries `depends_on`,
06 forwards `--depends-on <lwid>`, and the funnel resolves it AT SUBMIT TIME against the depended unit's binding and Slurm state —
`--dependency=afterok:<its bound job>` while that job is live or pending, no flag once every task of it COMPLETED, a refusal by name
when it was never submitted or FAILED (an afterok on a failed job is never satisfied), and placement on the cluster it is bound on.
Without it the u0 wave's climb launched while the store unit was still writing and refused every cell."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import job_ids, set_job_state  # noqa: E402
from test_submit_twin import SHA, World  # noqa: E402

STORE = ["submit", "--unit", "bon.store512", "--lwid", "m/u0/bon/store512", "--script", "scripts/sdfbase/full22/target_static.sbatch",
         "--name", "f22-stat-m-u0-3", "--args", f"m u0 bon {SHA} 9", "--clusters", "cluster_a", "--wave-sha", SHA]


def _climb(lwid="m/u0/bon/climb", unit="bon.climb", clusters="cluster_a,cluster_b"):
    return ["submit", "--unit", unit, "--lwid", lwid, "--script", "scripts/sdfbase/full22/bon_climb.sbatch", "--name", "f22-bonclimb-m-u0",
            "--args", f"m u0 {SHA} 4", "--clusters", clusters, "--wave-sha", SHA, "--array", "0-3", "--depends-on", "m/u0/bon/store512"]


def _argv(w, lab, jid) -> str:
    return (w.fake[lab] / "jobs" / f"{jid}.argv").read_text()


def _attempts(w):
    p = w.state / "attempts.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def test_a_dependent_unit_runs_after_its_dependencys_live_job_on_that_cluster_and_never_on_a_failed_one(tmp_path):
    w = World(tmp_path)
    for lab in ("cluster_a", "cluster_b"):
        (w.fake[lab] / "policy").write_text("start_on_submit\n")
    # the dependency was never submitted: refused by name, nothing minted anywhere
    r0 = w.run(*_climb())
    assert r0.returncode == 3 and "runs after m/u0/bon/store512, which was never submitted" in r0.stderr, r0.stderr
    assert job_ids(w.fake["cluster_a"]) == [] and job_ids(w.fake["cluster_b"]) == []
    r1 = w.run(*STORE)
    assert r1.returncode == 0 and "submitted cluster_a 1001 bon.store512" in r1.stdout, r1.stdout + r1.stderr
    # the store is RUNNING: afterok on its job, placed on ITS cluster although two were offered, the dependency recorded in the spec
    r2 = w.run(*_climb())
    assert r2.returncode == 0 and "submitted cluster_a 1002 bon.climb" in r2.stdout, r2.stdout + r2.stderr
    assert "dependency cluster_a bon.climb: afterok:1001 (m/u0/bon/store512 is RUNNING" in r2.stdout, r2.stdout
    argv = _argv(w, "cluster_a", 1002)
    assert "--dependency=afterok:1001" in argv and "--array=0-3" in argv, argv
    assert job_ids(w.fake["cluster_b"]) == []
    att = [a for a in _attempts(w) if a.get("event") == "attempt" and a["logical_work_id"] == "m/u0/bon/climb"][-1]
    assert att["spec"]["depends_on"] == "m/u0/bon/store512" and "--dependency" not in att["spec"]["flags"]   # resolved at submit time, never baked into the spec
    # the store COMPLETED: a dependent unit submitted now carries NO dependency (already satisfied)
    set_job_state(w.fake["cluster_a"], "1001", "COMPLETED", end="2026-01-01T01:00:00")
    r3 = w.run(*_climb(lwid="m/u0/bon/climb2", unit="bon.climb2"))
    assert r3.returncode == 0 and "none (m/u0/bon/store512's job 1001 COMPLETED: the dependency is satisfied)" in r3.stdout, r3.stdout + r3.stderr
    assert "--dependency" not in _argv(w, "cluster_a", job_ids(w.fake["cluster_a"])[-1])
    # the store FAILED: refused by name (afterok would never be satisfied), nothing minted
    set_job_state(w.fake["cluster_a"], "1001", "FAILED", end="2026-01-01T01:00:00")
    n = len(job_ids(w.fake["cluster_a"]))
    r4 = w.run(*_climb(lwid="m/u0/bon/climb3", unit="bon.climb3"))
    assert r4.returncode == 3 and "whose bound job 1001 on cluster_a is FAILED — re-run that unit first" in r4.stderr, r4.stderr
    assert len(job_ids(w.fake["cluster_a"])) == n
    # the dependency is bound on a cluster the caller did not offer: refused by name
    r5 = w.run(*_climb(lwid="m/u0/bon/climb4", unit="bon.climb4", clusters="cluster_b"))
    assert r5.returncode == 3 and "bound on cluster_a — not one of --clusters cluster_b" in r5.stderr, r5.stderr


def test_a_dry_run_says_what_the_dependency_would_be_and_submits_nothing(tmp_path):
    w = World(tmp_path)
    r = w.run("--dry-run", *_climb())
    assert r.returncode == 0 and "(dry run) --dependency=afterok:<m/u0/bon/store512's bound job>" in r.stdout, r.stdout + r.stderr
    assert job_ids(w.fake["cluster_a"]) == [] and job_ids(w.fake["cluster_b"]) == []
