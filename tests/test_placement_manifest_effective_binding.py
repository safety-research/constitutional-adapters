"""The pin's baseline is the first FROZEN-TREE attempt, and every later effective attempt and the unit's LATEST binding must agree with the
pin — in `ensure` and in `check` (commit 24c).

`effective_attempts` kept only the FIRST at-frozen-or-post-freeze attempt per unit, so a later post-freeze re-run on another cluster was
invisible to reconciliation, and `check` compared plan to manifest only — "placement OK" while the currently bound unit's results lived on
the other cluster; and a post-freeze attempt at a NON-frozen sha that preceded the first frozen-tree attempt won the selection, unlike the
workload baseline. Now the baseline is selected exactly as the workload baseline is (sha == inventory.code_sha or a funnel-advanced sha of
the unit), then every subsequent effective attempt and the latest binding are inspected for a moved cluster or group. Synthetic state root."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import stage as ST  # noqa: E402

KEY = "qwen35_4b_demo"
SHA = "a" * 40
ADV = "d" * 40            # a sha the funnel advanced the unit to (recorded in its provenance chain)
OTHER = "e" * 40          # a post-freeze tree the unit was never advanced to
AB = ["cluster_a", "cluster_b"]
LW = f"{KEY}/battery/multiturn/crescendo/g34"      # pinned on cluster_b under [cluster_a, cluster_b]


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "state"; (root / "gate").mkdir(parents=True)
    ST.write_atomic(str(root / "gate" / "expected_units.json"), {"schema": 2, "model": KEY, "view": "battery", "code_sha": SHA, "frozen_utc": "2026-01-02T00:00:00Z", "units": []})
    return root


def _attempt(lw, cluster, group, job, sha, utc):
    return json.dumps({"event": "attempt", "logical_work_id": lw, "cluster": cluster, "job_id": job, "kind": "target", "sha": sha,
                       "attempt_id": f"{lw}@{cluster}@{job}", "spec": {"group": group, "kind": "target", "sha": sha}, "utc": utc}) + "\n"


def _prov(lw, sha, job, cluster, utc):
    return json.dumps({"event": "sha_provenance", "logical_work_id": lw, "sha": sha, "kind": "advance", "job_id": job, "cluster": cluster, "utc": utc}) + "\n"


def test_the_pin_baseline_is_the_first_frozen_tree_attempt_never_a_non_frozen_post_freeze_one(tmp_path):
    root = _root(tmp_path); ledger = root / "submissions.jsonl"
    with open(root / "attempts.jsonl", "w") as fh:
        fh.write(_attempt(LW, "cluster_a", "mt-battery-cluster_a", "70", OTHER, "2026-01-03T00:00:00Z"))    # post-freeze, NOT a frozen or advanced sha: history
        fh.write(_attempt(LW, "cluster_b", "mt-battery-cluster_b", "80", SHA, "2026-01-04T00:00:00Z"))      # the first frozen-tree attempt: the baseline
    eff = ST.effective_attempts(str(ledger), KEY, "battery", SHA, "2026-01-02T00:00:00Z")
    assert eff[LW]["baseline"]["job_id"] == "80" and eff[LW]["baseline"]["cluster"] == "cluster_b" and eff[LW]["later"] == []
    man, probs = ST.placement_from_ledger(str(ledger), KEY, "battery", AB, sha=SHA)
    assert probs == [] and man["units"][LW]["cluster"] == "cluster_b"


def test_a_later_re_run_on_another_cluster_refuses_ensure_and_check_by_name(tmp_path):
    root = _root(tmp_path); ledger = root / "submissions.jsonl"
    with open(root / "attempts.jsonl", "w") as fh:
        fh.write(_attempt(LW, "cluster_b", "mt-battery-cluster_b", "80", SHA, "2026-01-04T00:00:00Z"))      # pinned on B (the frozen-tree attempt)
        fh.write(_prov(LW, ADV, "90", "cluster_a", "2026-01-05T00:00:00Z"))
        fh.write(_attempt(LW, "cluster_a", "mt-battery-cluster_a", "90", ADV, "2026-01-05T00:00:00Z"))      # a LATER re-run at an advanced sha, on A
    eff = ST.effective_attempts(str(ledger), KEY, "battery", SHA, "2026-01-02T00:00:00Z")
    assert eff[LW]["baseline"]["job_id"] == "80" and [a["job_id"] for a in eff[LW]["later"]] == ["90"]
    man, probs = ST.placement_from_ledger(str(ledger), KEY, "battery", AB, sha=SHA)
    assert probs and "crescendo/g34: pinned on cluster_b (group mt-battery-cluster_b) but its later attempt job 90 on cluster_a" in probs[0] and "2026-01-05T00:00:00Z" in probs[0], probs
    # `check` reads the ledger's LATEST binding too: a unit bound on the other cluster is not "placement OK"
    full = ST.placement_from_plan(KEY, "battery", AB, sha=SHA)
    ST.write_binding(str(ledger), ST.submitted_record(LW, "90", "cluster_a", f"{LW}@cluster_a@90", "f22-att-x", array=None))
    rows = ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB)
    probs = ST.check_placement(full, AB, rows, ledger=str(ledger))
    assert probs and "crescendo/g34" in probs[0] and "bound on cluster_a" in probs[0] and "pinned on cluster_b" in probs[0], probs
    # with the binding where the pin says, check passes
    ST.write_binding(str(ledger), ST.submitted_record(LW, "80", "cluster_b", f"{LW}@cluster_b@80", "f22-att-x", array=None))
    assert ST.check_placement(full, AB, rows, ledger=str(ledger)) == []
