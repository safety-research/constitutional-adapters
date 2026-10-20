"""`placement-manifest ensure` PERSISTS a receipted advance (commit 24e).

24d taught `expand_placement` to advance a unit's pin under the funnel's `placement_receipt`, but `ensure` wrote the expanded manifest
only when the plan had ADDED a unit. For the intended case — a frozen full manifest, then a fenced `submit_twin --resubmit` of a dead unit
from cluster A to B — nothing is added: the receipted B pin was computed in memory and discarded, and 06's immediately following `check`
reloaded the persisted A pin and refused the valid B binding. Now `ensure` writes whenever any existing unit's assignment or receipt
differs from the stored manifest, says which units advanced, and the following `check` passes. Execution form over the CLI; synthetic
state root; the COMPLETED skip gate of the funnel is untouched by this change."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import stage as ST  # noqa: E402

KEY = "qwen35_4b_demo"
SHA = "a" * 40
AB = ["cluster_a", "cluster_b"]
LW = f"{KEY}/battery/multiturn/crescendo/g34"            # pinned on cluster_b under [cluster_a, cluster_b]


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "state"; (root / "gate").mkdir(parents=True)
    ST.write_atomic(str(root / "gate" / "expected_units.json"), {"schema": 2, "model": KEY, "view": "battery", "code_sha": SHA, "frozen_utc": "2026-01-02T00:00:00Z", "units": []})
    return root


def _ev(root: Path, *events):
    with open(root / "attempts.jsonl", "a") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


def _attempt(cluster, job, utc):
    return {"event": "attempt", "logical_work_id": LW, "attempt_id": f"{LW}@{cluster}@{job}", "cluster": cluster, "job_id": job, "kind": "target", "sha": SHA,
            "spec": {"group": f"mt-battery-{cluster}", "kind": "target", "sha": SHA}, "utc": utc}


def _cli(root: Path, *args):
    return subprocess.run([sys.executable, str(REPO / "reproduce" / "lib" / "stage.py"), "placement-manifest", *args, "--model", KEY, "--view", "battery",
                           "--clusters", ",".join(AB), "--ledger", str(root / "attempts.jsonl"), "--sha", SHA, "--nshards", "9"],
                          capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")}, timeout=600)


def test_ensure_persists_a_receipted_cross_cluster_advance_and_the_following_check_passes(tmp_path):
    root = _root(tmp_path); path = root / "gate" / ST.placement_file("battery")
    # a FULL manifest frozen from the plan; the unit's first attempt on its pinned cluster B, registered
    ST.write_atomic(str(path), ST.placement_from_plan(KEY, "battery", AB, sha=SHA))
    _ev(root, _attempt("cluster_b", "80", "2026-01-04T00:00:00Z"),
        {"event": "registered", "logical_work_id": LW, "attempt_id": f"{LW}@cluster_b@80", "cluster": "cluster_b", "job_id": "80", "utc": "2026-01-04T00:06:00Z"})
    before = path.read_text(); mtime = path.stat().st_mtime_ns
    r = _cli(root, "ensure"); assert r.returncode == 0 and r.stdout.startswith("placement manifest: ") and path.stat().st_mtime_ns == mtime, r.stdout + r.stderr
    assert _cli(root, "check").returncode == 0
    # the unit died on B and was re-run on A through the funnel's fenced --resubmit: the attempt, its registration, its binding, the RECEIPT
    _ev(root, _attempt("cluster_a", "90", "2026-01-06T00:00:00Z"),
        {"event": "placement_receipt", "logical_work_id": LW, "from_cluster": "cluster_b", "to_cluster": "cluster_a", "to_group": "mt-battery-cluster_a",
         "replaced_job_id": "80", "replaced_state": "NODE_FAIL", "attempt_id": f"{LW}@cluster_a@90", "job_id": "90", "utc": "2026-01-06T00:00:01Z",
         "reason": "fenced dead-job replacement on another cluster (submit_twin --resubmit)"},
        {"event": "registered", "logical_work_id": LW, "attempt_id": f"{LW}@cluster_a@90", "cluster": "cluster_a", "job_id": "90", "utc": "2026-01-06T00:06:00Z"})
    ST.write_binding(str(root / "attempts.jsonl"), ST.submitted_record(LW, "90", "cluster_a", f"{LW}@cluster_a@90", "f22-att-x", array=None))
    # dry-run: says the advance, writes nothing
    r = _cli(root, "ensure", "--dry-run")
    assert r.returncode == 0 and "DRY-RUN: rewrite the placement manifest: 1 unit(s) advanced under a placement receipt (multiturn/crescendo/g34 -> cluster_a)" in r.stdout and path.read_text() == before, r.stdout + r.stderr
    # ensure PERSISTS the receipted pin (nothing was added) and the following check passes against the advanced manifest
    r = _cli(root, "ensure")
    assert r.returncode == 0 and r.stdout.startswith("placement manifest ADVANCED: 1 unit(s) advanced under a placement receipt (multiturn/crescendo/g34 -> cluster_a)"), r.stdout + r.stderr
    man = json.load(open(path))
    assert man["units"][LW]["cluster"] == "cluster_a" and man["units"][LW]["group"] == "mt-battery-cluster_a" and man["units"][LW]["receipt"] == f"{LW}@cluster_a@90" \
        and man["units"][LW]["replaced_job_id"] == "80" and man["advanced_units"] == [LW] and man["source"].endswith("+receipted") and man["advanced_utc"]
    assert {lw: u for lw, u in man["units"].items() if lw != LW} == {lw: u for lw, u in json.loads(before)["units"].items() if lw != LW}     # every other pin untouched
    r = _cli(root, "check"); assert r.returncode == 0 and r.stdout.startswith("placement OK"), r.stdout + r.stderr
    # idempotent: a second ensure finds nothing to write
    mtime = path.stat().st_mtime_ns
    r = _cli(root, "ensure"); assert r.returncode == 0 and r.stdout.startswith("placement manifest: ") and path.stat().st_mtime_ns == mtime, r.stdout + r.stderr


def test_an_unreceipted_move_still_refuses_ensure_and_writes_nothing(tmp_path):
    root = _root(tmp_path); path = root / "gate" / ST.placement_file("battery")
    ST.write_atomic(str(path), ST.placement_from_plan(KEY, "battery", AB, sha=SHA))
    _ev(root, _attempt("cluster_b", "80", "2026-01-04T00:00:00Z"),
        {"event": "registered", "logical_work_id": LW, "attempt_id": f"{LW}@cluster_b@80", "cluster": "cluster_b", "job_id": "80", "utc": "2026-01-04T00:06:00Z"},
        _attempt("cluster_a", "90", "2026-01-06T00:00:00Z"),
        {"event": "registered", "logical_work_id": LW, "attempt_id": f"{LW}@cluster_a@90", "cluster": "cluster_a", "job_id": "90", "utc": "2026-01-06T00:06:00Z"})
    before = path.read_text()
    r = _cli(root, "ensure")
    assert r.returncode == 2 and "REFUSED: placement-manifest:" in r.stderr and "unreceipted" in r.stderr and "crescendo/g34" in r.stderr and path.read_text() == before, r.stdout + r.stderr
