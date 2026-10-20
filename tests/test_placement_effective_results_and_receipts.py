"""Placement reconciliation reads RESULTS, not submissions, and a receipted cross-cluster re-run advances the pin.

(a) `_reconcile_placement` treated every frozen/provenance-sha attempt as results and `check_placement` treated the latest `submitted`
binding as results: a `--twin` election records both twins and DENYs the loser without clearing its later binding, and a cancelled
never-started attempt leaves a binding too — so a correctly pinned winner on A with a denied loser on B was refused. Now the effective
result cluster is derived from attempts with ELECTION / REGISTRATION evidence (granted, or registered at job start) and cancelled, denied
and never-started attempts are filtered out; an unreadable or malformed binding fails CLOSED (refused by name, never "no binding").
(b) every later attempt was hard-compared to the original pin: a fenced, provenance-recorded cross-cluster re-run of a DEAD unit on
another cluster — the funnel's `--resubmit` replacement path, the only sanctioned way a unit moves — was refused forever. Now that path
writes a placement-change RECEIPT (`placement_receipt` ledger event: unit, from, to, the replaced job, the sacct proof, the attempt),
the manifest advances the pin under it, and `ensure` / `check` compare effective results to the LATEST RECEIPTED assignment; an
unreceipted move still refuses by name. Synthetic state root."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

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


def _attempt(cluster, job, utc, group=None, sha=SHA):
    return {"event": "attempt", "logical_work_id": LW, "attempt_id": f"{LW}@{cluster}@{job}", "cluster": cluster, "job_id": job, "kind": "target", "sha": sha,
            "spec": {"group": group or f"mt-battery-{cluster}", "kind": "target", "sha": sha}, "utc": utc}


def test_a_twin_election_s_denied_loser_and_a_cancelled_never_started_attempt_are_not_results(tmp_path):
    root = _root(tmp_path); ledger = root / "submissions.jsonl"
    _ev(root,
        _attempt("cluster_b", "80", "2026-01-04T00:00:00Z"),                                      # the pinned winner on B (frozen tree)
        _attempt("cluster_a", "81", "2026-01-04T00:00:01Z"),                                      # the twin on A: denied
        {"event": "granted", "logical_work_id": LW, "attempt_id": f"{LW}@cluster_b@80", "utc": "2026-01-04T00:05:00Z"},          # the funnel's election
        {"event": "denied", "logical_work_id": LW, "attempt_id": f"{LW}@cluster_a@81", "utc": "2026-01-04T00:05:00Z"},
        {"event": "registered", "logical_work_id": LW, "attempt_id": f"{LW}@cluster_b@80", "cluster": "cluster_b", "job_id": "80", "utc": "2026-01-04T00:06:00Z"},
        _attempt("cluster_a", "82", "2026-01-05T00:00:00Z"),                                      # a later never-started attempt on A, cancelled
        {"event": "denied", "logical_work_id": LW, "attempt_id": f"{LW}@cluster_a@82", "cancelled": True, "utc": "2026-01-05T00:01:00Z"})   # cancel-attempt's record
    eff = ST.effective_results(str(ledger), KEY, "battery", SHA)
    assert eff[LW]["cluster"] == "cluster_b" and eff[LW]["job_id"] == "80" and eff[LW]["evidence"] in ("registered", "granted")
    assert [a["job_id"] for a in eff[LW]["excluded"]] == ["81", "82"]
    man, probs = ST.placement_from_ledger(str(ledger), KEY, "battery", AB, sha=SHA)
    assert probs == [] and man["units"][LW]["cluster"] == "cluster_b"
    # the stale loser binding on A is not a result either: `check` with the ledger passes
    ST.write_binding(str(ledger), ST.submitted_record(LW, "82", "cluster_a", f"{LW}@cluster_a@82", "f22-att-x", array=None))
    rows = ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB)
    assert ST.check_placement(ST.placement_from_plan(KEY, "battery", AB, sha=SHA), AB, rows, ledger=str(ledger)) == []


def test_a_receipted_cross_cluster_re_run_advances_the_pin_and_an_unreceipted_move_refuses(tmp_path):
    root = _root(tmp_path); ledger = root / "submissions.jsonl"
    _ev(root,
        _attempt("cluster_b", "80", "2026-01-04T00:00:00Z"),
        {"event": "registered", "logical_work_id": LW, "attempt_id": f"{LW}@cluster_b@80", "cluster": "cluster_b", "job_id": "80", "utc": "2026-01-04T00:06:00Z"})
    man = ST.placement_from_plan(KEY, "battery", AB, sha=SHA)
    ST.write_atomic(str(root / "gate" / "expected_units.placement.json"), man)
    # an UNRECEIPTED move: a later registered attempt on A with no receipt -> refused by name
    _ev(root, _attempt("cluster_a", "90", "2026-01-06T00:00:00Z"),
        {"event": "registered", "logical_work_id": LW, "attempt_id": f"{LW}@cluster_a@90", "cluster": "cluster_a", "job_id": "90", "utc": "2026-01-06T00:06:00Z"})
    man2, probs = ST.placement_from_ledger(str(ledger), KEY, "battery", AB, sha=SHA)
    assert probs and "crescendo/g34" in probs[0] and "cluster_a" in probs[0] and "unreceipted" in probs[0], probs
    # the RECEIPT the funnel's --resubmit cross-cluster replacement writes (the dead job on B proven by sacct) advances the pin
    _ev(root, {"event": "placement_receipt", "logical_work_id": LW, "from_cluster": "cluster_b", "to_cluster": "cluster_a", "to_group": "mt-battery-cluster_a",
               "replaced_job_id": "80", "replaced_state": "NODE_FAIL", "attempt_id": f"{LW}@cluster_a@90", "job_id": "90", "utc": "2026-01-06T00:00:00Z",
               "reason": "fenced dead-job replacement on another cluster (submit_twin --resubmit)"})
    man3, probs3 = ST.placement_from_ledger(str(ledger), KEY, "battery", AB, sha=SHA)
    assert probs3 == [] and man3["units"][LW]["cluster"] == "cluster_a" and man3["units"][LW]["group"] == "mt-battery-cluster_a" and man3["units"][LW].get("receipt") == f"{LW}@cluster_a@90"
    out, added, probs4 = ST.expand_placement(man, str(ledger), KEY, "battery", AB, sha=SHA)
    assert probs4 == [] and out["units"][LW]["cluster"] == "cluster_a"
    rows = ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB)
    assert ST.check_placement(out, AB, rows, ledger=str(ledger)) == []


def test_an_unreadable_binding_fails_closed(tmp_path):
    root = _root(tmp_path); ledger = root / "submissions.jsonl"
    _ev(root, _attempt("cluster_b", "80", "2026-01-04T00:00:00Z"),
        {"event": "registered", "logical_work_id": LW, "attempt_id": f"{LW}@cluster_b@80", "cluster": "cluster_b", "job_id": "80", "utc": "2026-01-04T00:06:00Z"})
    bdir = root / "bindings"; bdir.mkdir(exist_ok=True)
    (bdir / f"{ST.safe_id(LW)}.submitted.json").write_text("{not json")
    rows = ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB)
    probs = ST.check_placement(ST.placement_from_plan(KEY, "battery", AB, sha=SHA), AB, rows, ledger=str(ledger))
    assert probs and "binding" in probs[0] and "unreadable" in probs[0].lower() and "crescendo/g34" in probs[0], probs


# ---- the funnel side: the --resubmit cross-cluster replacement writes the receipt -------------------------------------------------------
from front_door_testlib import set_job_state  # noqa: E402
from test_submit_sha_provenance import SHA as FUNNEL_SHA, World, _inventory, _ledger, _mmlu  # noqa: E402


def test_the_funnels_cross_cluster_resubmit_writes_a_placement_receipt_and_a_same_cluster_one_does_not(tmp_path):
    """submit_twin --resubmit of a provably dead job onto ANOTHER cluster (the only sanctioned way a unit moves) appends a
    `placement_receipt` ledger event naming the unit, the clusters, the replaced job with its sacct state and the replacing attempt; a
    same-cluster re-run writes none (execution form: the real funnel over the fake clusters)."""
    w = World(tmp_path)
    for lab in ("cluster_a", "cluster_b"):
        (w.fake[lab] / "policy").write_text("start_on_submit\n")
    _inventory(w)
    r1 = w.run(*_mmlu(FUNNEL_SHA), "--wave-sha", FUNNEL_SHA)
    assert r1.returncode == 0 and "submitted cluster_a 1001 mmlu" in r1.stdout, r1.stdout + r1.stderr
    set_job_state(w.fake["cluster_a"], "1001", "NODE_FAIL", end="2026-01-01T00:10:00"); w.spool("cluster_a")
    on_b = [*_mmlu(FUNNEL_SHA)[:-1], "cluster_b"]
    r2 = w.run(*on_b, "--wave-sha", FUNNEL_SHA, "--resubmit")
    out = r2.stdout + r2.stderr
    assert r2.returncode == 0 and "submitted cluster_b" in r2.stdout and "PLACEMENT RECEIPT m/s1/mmlu/mmlu: cluster_a (job 1001, NODE_FAIL) -> cluster_b" in out, out
    led = _ledger(w)
    new = [e for e in led if e["event"] == "attempt" and e["cluster"] == "cluster_b"]
    assert len(new) == 1
    rec = [e for e in led if e["event"] == "placement_receipt"]
    assert len(rec) == 1 and rec[0]["logical_work_id"] == "m/s1/mmlu/mmlu" and rec[0]["from_cluster"] == "cluster_a" and rec[0]["to_cluster"] == "cluster_b"
    assert rec[0]["replaced_job_id"] == "1001" and rec[0]["replaced_state"] == "NODE_FAIL" and rec[0]["replaced_attempt_id"] == "m/s1/mmlu/mmlu@cluster_a@1"
    assert rec[0]["attempt_id"] == new[0]["attempt_id"] == "m/s1/mmlu/mmlu@cluster_b@1" and rec[0]["job_id"] == str(new[0]["job_id"]) and rec[0]["to_group"] is None
    assert rec[0]["utc"] and "submit_twin --resubmit" in rec[0]["reason"]
    # the replacement dies on cluster_b and is re-run THERE: no receipt (the placement did not move)
    set_job_state(w.fake["cluster_b"], str(new[0]["job_id"]), "NODE_FAIL", end="2026-01-01T00:20:00"); w.spool("cluster_b")
    r3 = w.run(*on_b, "--wave-sha", FUNNEL_SHA, "--resubmit")
    assert r3.returncode == 0 and "submitted cluster_b" in r3.stdout, r3.stdout + r3.stderr
    assert len([e for e in _ledger(w) if e["event"] == "placement_receipt"]) == 1
    assert ST.placement_receipts(str(w.state / "submissions.jsonl")) == {"m/s1/mmlu/mmlu": [{k: rec[0].get(k) for k in ("from_cluster", "to_cluster", "to_group", "replaced_job_id", "replaced_state", "attempt_id", "job_id", "utc", "reason")}]}


def test_an_undeclared_wrong_cluster_placement_cannot_be_laundered_into_a_receipt_by_a_same_cluster_resubmit(tmp_path):
    """move-knob-5: a multi-turn target the placement manifest pins on cluster_a is placed on cluster_b by a PLAIN
    submission (a manual submit_one on the wrong cluster), registers nothing and dies; a --resubmit on cluster_b (the same cluster as the dead
    attempt, the pin elsewhere, NO declared move) is REFUSED by name before anything is submitted — zero new attempts, zero receipts — and the
    manifest's ensure / check refuse the undeclared cross-cluster binding on their own. The real funnel over the fake clusters."""
    w = World(tmp_path)
    for lab in ("cluster_a", "cluster_b"):
        (w.fake[lab] / "policy").write_text("start_on_submit\n")
    _inventory(w)
    lw = "m/s1/multiturn/crescendo/g34"
    man = {"schema": 1, "model": "m", "view": "s1", "labels": ["cluster_a", "cluster_b"], "pinned_utc": "2026-01-02T00:00:00Z", "source": "test",
           "units": {lw: {"cluster": "cluster_a", "group": "mt-s1-cluster_a", "kind": "target"}}}
    ST.write_atomic(ST.placement_path(str(w.state / "attempts.jsonl"), "s1"), man)
    target = ["submit", "--unit", "multiturn.crescendo.g34", "--lwid", lw, "--script", "scripts/sdfbase/full22/target_attacks.sbatch", "--name", "f22-att-m-g34-crescendo",
              "--args", f"m s1 crescendo {FUNNEL_SHA} 9 3,4", "--kind", "target", "--group", "mt-s1-cluster_b", "--wave-sha", FUNNEL_SHA, "--clusters", "cluster_b"]
    r1 = w.run(*target)
    assert r1.returncode == 0 and "submitted cluster_b 1001 multiturn.crescendo.g34" in r1.stdout, r1.stdout + r1.stderr      # the wrong-cluster plain placement
    set_job_state(w.fake["cluster_b"], "1001", "FAILED", end="2026-01-01T00:10:00"); w.spool("cluster_b")
    before = [e for e in _ledger(w) if e["event"] == "attempt"]
    r2 = w.run(*target, "--resubmit")
    out = r2.stdout + r2.stderr
    assert r2.returncode == 3, out
    assert f"{lw}: --resubmit on cluster_b would replace attempt {lw}@cluster_b@1 (job 1001, FAILED) on cluster_b while the placement manifest pins the unit on cluster_a and declares no move to cluster_b" in out, out
    assert "NOTHING submitted" in out and "submitted cluster_b" not in r2.stdout, out
    led = _ledger(w)
    assert [e for e in led if e["event"] == "attempt"] == before                                   # zero new attempts
    assert not [e for e in led if e["event"] == "placement_receipt"]                              # zero receipts
    _assert_untouched_after_refusal(w, lw, led, registered=False)                                  # move-knob-6: a refusal that IS a preflight
    # the manifest side refuses the state by name on its own (an undeclared cross-cluster binding)
    probs = ST.check_placement(man, ["cluster_a", "cluster_b"], [], ledger=str(w.state / "submissions.jsonl"))      # the ledger branch: the binding vs the pin
    assert any(f"{lw}: pinned on cluster_a but its latest binding (job 1001, submitted) is bound on cluster_b" in pr for pr in probs), probs


def _assert_untouched_after_refusal(w, lw, led, *, registered: bool):
    """move-knob-6: a declared-move refusal is a PREFLIGHT — after it the unit's old binding stands (`submitted`, or `registered` when the
    gate had bound the dead job's runs), no transition token was reserved, no `superseded_attempt` was appended, nothing was retired."""
    ledger = str(w.state / "submissions.jsonl")
    assert not Path(ST.transition_path(ledger, lw)).exists(), "a transition token was reserved by a refused resubmit"
    st = ST.binding_state(ledger, lw)
    assert st["state"] == ("registered" if registered else "submitted"), st
    assert not [e for e in led if e["event"] == "superseded_attempt" and e.get("logical_work_id") == lw], "the old attempt was superseded by a refused resubmit"
    files = sorted(p.name for p in (w.state / "bindings").glob(f"{ST.safe_id(lw)}.*"))
    assert files == sorted([f"{ST.safe_id(lw)}.submitted.json"] + ([f"{ST.safe_id(lw)}.registered.json"] if registered else [])), files


def test_an_undeclared_wrong_cluster_job_that_registered_runs_and_died_is_refused_before_its_binding_is_retired(tmp_path):
    """The trigger: the wrong-cluster multi-turn target REGISTERED its runs (the gate bound them) and died; the same-cluster --resubmit
    (pin elsewhere, no declaration) is REFUSED as a preflight — the `registered` binding is NOT retired, no transition token, no
    supersession, no new attempt, no receipt; the registry rows the dead job filed stay the unit's effective results."""
    from test_submit_resubmit_proof import _bind_registered, _pend
    from front_door_testlib import make_registry
    w = World(tmp_path)
    for lab in ("cluster_a", "cluster_b"):
        (w.fake[lab] / "policy").write_text("start_on_submit\n")
    _inventory(w)
    lw = "m/s1/multiturn/crescendo/g34"
    man = {"schema": 1, "model": "m", "view": "s1", "labels": ["cluster_a", "cluster_b"], "pinned_utc": "2026-01-02T00:00:00Z", "source": "test",
           "units": {lw: {"cluster": "cluster_a", "group": "mt-s1-cluster_a", "kind": "target"}}}
    ST.write_atomic(ST.placement_path(str(w.state / "attempts.jsonl"), "s1"), man)
    target = ["submit", "--unit", "multiturn.crescendo.g34", "--lwid", lw, "--script", "scripts/sdfbase/full22/target_attacks.sbatch", "--name", "f22-att-m-g34-crescendo",
              "--args", f"m s1 crescendo {FUNNEL_SHA} 9 3,4", "--kind", "target", "--group", "mt-s1-cluster_b", "--wave-sha", FUNNEL_SHA, "--clusters", "cluster_b"]
    r1 = w.run(*target)
    assert r1.returncode == 0 and "submitted cluster_b 1001 multiturn.crescendo.g34" in r1.stdout, r1.stdout + r1.stderr
    _pend(w, "cluster_b", "1001", ["run-a"])
    make_registry(w.tmp / "registry", runs=[{"run_id": "run-a", "lifecycle_state": "running", "cluster_job_id": "1001"}])
    _bind_registered(w, lw, ["run-a"])                                                            # the gate bound the dead job's runs
    set_job_state(w.fake["cluster_b"], "1001", "FAILED", end="2026-01-01T00:10:00"); w.spool("cluster_b"); w.ingested("cluster_b", through="2026-01-01T00:20:00")
    before = [e for e in _ledger(w) if e["event"] == "attempt"]
    r2 = w.run(*target, "--resubmit")
    out = r2.stdout + r2.stderr
    assert r2.returncode == 3 and "declares no move to cluster_b" in out and "NOTHING submitted" in out and "submitted cluster_b" not in r2.stdout, out
    led = _ledger(w)
    assert [e for e in led if e["event"] == "attempt"] == before and not [e for e in led if e["event"] == "placement_receipt"]
    _assert_untouched_after_refusal(w, lw, led, registered=True)
