"""reproduce/lib/stage.py amend-contract — the RECEIPTED contract amendment, made ENFORCEABLE:

1. (High) the OLD decode-policy export is DERIVED from the unit's recorded tree (`git show <recorded sha>:configs/registry/decode_policy.json`),
   never typed — a supplied --old-export is accepted only byte-identical to it, the refusal naming both hashes; --commit must be an
   ancestor-or-equal of the tip the unit will run at, and the NEW export and the seeds the new contract is minted from are read from git at
   --commit. A file that hides a temperature change behind the three reverted keys no longer passes.
2. (High) every old job of the unit must be TERMINAL on sacct (the funnel's own reader, injected here) and every old run definitively
   non-completed: a `running` run passes only with its job proven terminal-not-COMPLETED, recorded per run in the receipt; the registry is
   re-read after the proofs, right before the ledger event — a run that completed in the interval refuses.
3. (Medium) the three keys admit ONLY their exact neutral transitions (min_p null->0.0, repetition_penalty null->1.0, presence_penalty
   null->0.0); any other value or transition is a refusal naming the field and both values.

Plus `--recheck`: the three proofs re-run over every recorded receipt without writing, so receipts minted before the fix are re-validated.
Everything here is synthetic: neutral cluster labels, job ids `1`/`2`, placeholder shas, hostname TESTHOST."""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import hashlib
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, REPRO, VENV_PY, make_registry, set_job_state, write_fake_slurm, write_site  # noqa: E402
from test_admission import MODEL, SHA, _prov, by_unit, freeze, satisfy  # noqa: E402

sys.path.insert(0, str(LIB))
import grant_ledger as GL  # noqa: E402
import stage as ST  # noqa: E402
import submit_twin as TW  # noqa: E402

REASON = "D217: the seeded vendor row sent five of six sampling keys and the driver refused it; the fix states min_p 0.0 / repetition_penalty 1.0 / presence_penalty 0.0"
GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@TESTHOST"]


def corrected_keys(row: dict) -> tuple:
    """Of the three keys, the ones the seed-row correction MOVED on this row: those the row AS FILED states at the neutral value the amend
    admits (min_p 0.0 / repetition_penalty 1.0 / presence_penalty 0.0, what vLLM applies to a request that omits them). A key filed at any
    other value — the worked example's vendor twin `qwen35_4b_demo|multi_turn|off|vendor_preset` sends presence_penalty 1.5, as the 4B row
    it mirrors does — was never null in the run's world and is left as filed: null -> 1.5 is no transition the amend admits, and that refusal
    is the contract's (correct), not a world the fixture should build."""
    return tuple(k for k, want in ST.AMEND_NEUTRAL_TRANSITIONS.items()
                 if isinstance(row.get(k), (int, float)) and not isinstance(row.get(k), bool) and float(row[k]) == float(want))


def _old_export(new_export: dict, ref: str, *, also: dict | None = None) -> dict:
    """The decode-policy export as it stood BEFORE the seed-row correction: the row of `ref` with the sampling keys the correction moved
    (`corrected_keys`) unset and absent from params_sent, every other key as the registry row states it; `also` mutates further fields
    (a change beyond the three keys)."""
    doc = json.loads(json.dumps(new_export))
    row = ST.decode_row(doc, ref)
    assert row is not None, ref
    corrected = corrected_keys(row)
    assert corrected, f"{ref}: no key of {list(ST.AMEND_ALLOWED_KEYS)} is filed at its neutral value — nothing for the correction to have moved"
    for k in corrected:
        row[k] = None
    row["params_sent"] = [k for k in row.get("params_sent", []) if k not in corrected]
    row["notes"] = "before the correction"
    for k, v in (also or {}).items():
        row[k] = v
    return doc


def _git(repo: Path, *args: str) -> str:
    return subprocess.run([*GIT, *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def amend_repo(tmp: Path, ref: str, *, old_also: dict | None = None, new_mutate=None) -> dict:
    """A synthetic checkout holding the REAL registry seeds, committed twice: `old` carries the decode-policy export BEFORE the correction
    (the row of `ref` with the keys the correction moved unset — `corrected`, see `corrected_keys`; `old_also` mutates further fields of
    that OLD row), `new` the export as this checkout ships it (or `new_mutate`d). The seeds (the yaml files the planner mints from) are
    identical at both, so the new tree mints this checkout's canonical instrument ids."""
    repo = tmp / "repo"; (repo / "configs").mkdir(parents=True)
    shutil.copytree(REPO / "configs" / "registry", repo / "configs" / "registry")
    exp = repo / "configs" / "registry" / "decode_policy.json"
    new_export = json.load(open(exp))
    corrected = corrected_keys(ST.decode_row(new_export, ref))
    old_export = _old_export(new_export, ref, also=old_also)
    _git(repo, "init", "-q", "-b", "main")
    exp.write_text(json.dumps(old_export, indent=1, sort_keys=True))
    _git(repo, "add", "-A"); _git(repo, "commit", "-q", "-m", "before the correction")
    old_sha = _git(repo, "rev-parse", "HEAD")
    if new_mutate:
        new_mutate(new_export, ref)
    exp.write_text(json.dumps(new_export, indent=1, sort_keys=True))
    _git(repo, "add", "-A"); _git(repo, "commit", "-q", "-m", "the correction")
    new_sha = _git(repo, "rev-parse", "HEAD")
    return {"repo": repo, "old_sha": old_sha, "new_sha": new_sha, "old_export": old_export, "new_export": new_export, "corrected": corrected}


def amended_world(tmp: Path, *, old_also=None, new_mutate=None, run_state="running", provenance="ledger") -> dict:
    """The multi-turn crescendo unit frozen under the OLD instrument id, one attempt (job `1` on cluster_a) that registered runs in
    `run_state`, and the synthetic two-commit repo. `provenance`: `ledger` records the unit's sha (= the repo's old commit) as a
    sha_provenance event + sidecar entry; `freeze` leaves no event and sets the inventory's code_sha to it (the shape of receipts whose
    attempts predate the provenance event)."""
    state = tmp / "state"; doc = freeze(state, families=["multiturn"]); u = by_unit(doc)
    unit = u["multiturn/crescendo/g012"]; lw = unit["logical_work_id"]; ref = unit["evals"]["crescendo"]["decode_policy_ref"]
    ledger = str(state / "submissions.jsonl"); inv_p = state / "gate" / "expected_units.json"
    r = amend_repo(tmp, ref, old_also=old_also, new_mutate=new_mutate)
    old_inst = "0" * 64
    inv = json.load(open(inv_p))
    for x in inv["units"]:
        if x["logical_work_id"] == lw:
            x["evals"]["crescendo"]["instrument_id"] = old_inst
            if provenance == "freeze":
                x["code_sha"] = r["old_sha"]
    ST.write_atomic(str(inv_p), inv)
    unit_old = json.loads(json.dumps(unit)); unit_old["evals"]["crescendo"]["instrument_id"] = old_inst
    if provenance == "freeze":
        unit_old["code_sha"] = r["old_sha"]
    runs, pols = [], []
    satisfy(state, unit_old, runs, pols, code_sha=r["old_sha"])          # binds job `1` on cluster_a; registers one run per eval
    for x in runs:
        x["lifecycle_state"] = run_state
    aid = f"{lw}@cluster_a@1"
    GL.Ledger(str(state / "attempts.jsonl")).append({"event": "attempt", "logical_work_id": lw, "attempt_id": aid, "cluster": "cluster_a", "job_id": "1",
                                                      "job_name": "mt-crescendo-g012", "kind": "target", "sha": r["old_sha"], "utc": "2026-01-01T00:00:00Z"})
    if provenance == "ledger":
        _prov(state, lw, {"sha": r["old_sha"], "kind": "first", "attempt_id": aid, "job_id": "1", "cluster": "cluster_a", "contract_sha256": ST.contract_sha256(unit_old)})
    db = make_registry(tmp / "reg", runs, pols)
    return {"state": state, "ledger": ledger, "inv_p": inv_p, "unit": unit, "unit_old": unit_old, "lw": lw, "ref": ref, "runs": runs, "pols": pols, "db": db,
            "old_inst": old_inst, "real_inst": unit["evals"]["crescendo"]["instrument_id"], **r}


class FakeSacct:
    """The injected sacct reader: {(cluster, job id): state | [(raw, task, state)] | 'UNREACHABLE'} -> rows in submit_twin.Remote.sacct_tasks'
    shape; `on_read` runs on every call (the fence test flips a registry row inside the interval)."""

    def __init__(self, states: dict, on_read=None):
        self.states = dict(states); self.on_read = on_read; self.calls: list = []

    def __call__(self, cluster: str, job_id: str) -> list:
        self.calls.append((cluster, str(job_id)))
        if self.on_read:
            self.on_read()
        v = self.states.get((cluster, str(job_id)))
        if v == "UNREACHABLE":
            raise TW.Unreachable("ssh cluster-login: Connection timed out")
        if v is None:
            return []
        if isinstance(v, str):
            return [{"raw": str(job_id), "task": None, "state": v, "terminal": v not in TW.ACTIVE, "end": None}]
        return [{"raw": raw, "task": task, "state": st, "terminal": st not in TW.ACTIVE, "end": None} for raw, task, st in v]


def _amend(w: dict, sacct, **kw):
    args = dict(reason=REASON, commit=w["new_sha"], tip=w["new_sha"], repo=str(w["repo"]), registry_db=str(w["db"]), sacct_tasks=sacct)
    args.update(kw)
    return ST.amend_contract(w["ledger"], w["lw"], **args)


def _set_state(db: Path, run_id: str, state: str) -> None:
    c = sqlite3.connect(db); c.execute("update run set lifecycle_state=? where run_id=?", (state, run_id)); c.commit(); c.close()


def _ledger(state: Path) -> list:
    return [json.loads(l) for l in (state / "attempts.jsonl").read_text().splitlines() if l.strip()]


def test_sacct_active_states_are_the_funnels():
    assert ST.SACCT_ACTIVE == TW.ACTIVE


# ------------------------------------------------------------------------------------- the exact neutral transitions

def test_decode_row_diff_admits_only_the_exact_neutral_transitions_and_names_any_other():
    """A key-name whitelist let `min_p: null -> 0.05` pass; now each of the three keys has ONE admissible transition — null/omitted -> the
    engine's resolved default — and the refusal names the field and both values. Doc fields, params_sent and the keys' params_not_sent
    notes stay tolerated."""
    base = {"policy_id": "m|multi_turn|off|vendor_preset@v1", "temperature": 1.0, "top_p": 0.95, "min_p": None, "repetition_penalty": None,
            "presence_penalty": None, "params_sent": ["temperature", "top_p", "max_tokens"], "notes": "before",
            "params_not_sent_json": {"min_p": "0.0 (vLLM default)", "repetition_penalty": "1.0", "presence_penalty": "0.0", "frequency_penalty": "0.0"}}
    good = {**base, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0, "notes": "after", "ruling_ref": "r",
            "params_sent": ["temperature", "top_p", "min_p", "repetition_penalty", "presence_penalty", "max_tokens"],
            "params_not_sent_json": {"frequency_penalty": "0.0"}}
    allowed, violations = ST.decode_row_diff(base, good)
    assert violations == [] and {c["path"] for c in allowed} == {"min_p", "repetition_penalty", "presence_penalty", "notes", "ruling_ref", "params_sent",
                                                                 "params_not_sent_json.min_p", "params_not_sent_json.repetition_penalty", "params_not_sent_json.presence_penalty"}
    # the omitted key (absent, not null) -> the neutral value is the same transition; an integer spelling of the neutral value is the value
    omitted = {k: v for k, v in base.items() if k not in ST.AMEND_ALLOWED_KEYS}
    assert ST.decode_row_diff(omitted, {**good, "min_p": 0, "repetition_penalty": 1})[1] == []
    # any OTHER new value refuses by name with both values
    for key, bad in (("min_p", 0.05), ("repetition_penalty", 1.1), ("presence_penalty", 0.5), ("min_p", "0.0"), ("presence_penalty", True), ("repetition_penalty", None)):
        _, v = ST.decode_row_diff(base, {**good, key: bad})
        if bad is None:
            assert v == [], (key, bad, v)                       # null -> null is no change at all
            continue
        assert len(v) == 1 and v[0].startswith(f"{key}: None -> {bad!r}") and "only admissible transition is null/omitted ->" in v[0], v
    # any other OLD value refuses too: the transition is FROM null/omitted, never from a number
    _, v = ST.decode_row_diff({**base, "min_p": 0.1}, good)
    assert v == ["min_p: 0.1 -> 0.0 (the only admissible transition is null/omitted -> 0.0, the value the engine resolves an omitted min_p to)"]
    _, v = ST.decode_row_diff({**base, "presence_penalty": 0.0}, {**good, "presence_penalty": None})
    assert len(v) == 1 and v[0].startswith("presence_penalty: 0.0 -> None")
    # the contract fields are still refusals by name; a params_not_sent note of ANOTHER key is not a doc field
    _, v = ST.decode_row_diff(base, {**good, "temperature": 0.7})
    assert v == ["temperature: 1.0 -> 0.7"]
    _, v = ST.decode_row_diff(base, {**good, "params_not_sent_json": {"frequency_penalty": "0.5"}})
    assert v == ["params_not_sent_json.frequency_penalty: '0.0' -> '0.5'"]


def test_an_amendment_whose_row_moves_a_key_to_a_non_neutral_value_is_refused_by_name(tmp_path):
    w = amended_world(tmp_path, new_mutate=lambda export, ref: ST.decode_row(export, ref).__setitem__("min_p", 0.05))
    with pytest.raises(ValueError, match=r"changed beyond the exact neutral transitions .* refused by name: min_p: None -> 0\.05 \(the only admissible transition is null/omitted -> 0\.0"):
        _amend(w, FakeSacct({("cluster_a", "1"): "FAILED"}))
    assert not ST.read_amendments(w["ledger"]) and not [e for e in _ledger(w["state"]) if e["event"] == "contract_amendment"]


# ------------------------------------------------------------------------- the old export is derived, the commit bound to the tip

def test_the_old_export_is_derived_from_the_recorded_tree_and_a_supplied_file_must_match_it_byte_for_byte(tmp_path):
    """The attack: the OLD tree's row had temperature 1.0 where the filed row states 0.7 (a real sampler change rode the
    correction); a hand-made --old-export that is the NEW row with only the corrected keys reverted hid it. Now the old row comes from git
    at the unit's recorded sha — the temperature change is refused by name — and the supplied file is refused because its sha256 is not
    the derivation's."""
    w = amended_world(tmp_path, old_also={"temperature": 1.0})
    assert ST.decode_row(w["new_export"], w["ref"])["temperature"] == 0.7, "the filed row's temperature moved; pick an old value that differs"
    sacct = FakeSacct({("cluster_a", "1"): "FAILED"})
    with pytest.raises(ValueError, match=r"changed beyond .* refused by name: temperature: 1\.0 -> 0\.7"):
        _amend(w, sacct)
    forged = tmp_path / "forged_old_export.json"
    forged.write_text(json.dumps(_old_export(w["new_export"], w["ref"]), indent=1, sort_keys=True))     # the new row, the corrected keys reverted
    with pytest.raises(ValueError) as ei:
        _amend(w, sacct, old_export_path=str(forged))
    msg = str(ei.value)
    import hashlib
    derived = subprocess.run(["git", "-C", str(w["repo"]), "show", f"{w['old_sha']}:configs/registry/decode_policy.json"], capture_output=True, check=True).stdout
    assert f"sha256 {hashlib.sha256(forged.read_bytes()).hexdigest()}" in msg and f"sha256 {hashlib.sha256(derived).hexdigest()}" in msg \
        and w["old_sha"] in msg and "DERIVED, never typed" in msg, msg
    assert not ST.read_amendments(w["ledger"])


def test_the_commit_must_be_an_ancestor_or_equal_of_the_tip_and_the_receipt_binds_both_trees(tmp_path):
    w = amended_world(tmp_path)
    sacct = FakeSacct({("cluster_a", "1"): "FAILED"})
    # a commit on a side branch (not in the tip's history) is refused by name; a tip that does not exist is refused; the old sha is not typed
    _git(w["repo"], "checkout", "-q", "-b", "side", w["old_sha"])
    (w["repo"] / "side.txt").write_text("x\n"); _git(w["repo"], "add", "-A"); _git(w["repo"], "commit", "-q", "-m", "side")
    side = _git(w["repo"], "rev-parse", "HEAD"); _git(w["repo"], "checkout", "-q", "main")
    with pytest.raises(ValueError, match=f"--commit {side} \\({side[:12]}\\) is not an ancestor-or-equal of the tip {w['new_sha'][:12]}"):
        _amend(w, sacct, commit=side)
    with pytest.raises(ValueError, match="is not a commit of"):
        _amend(w, sacct, tip="f" * 40)
    with pytest.raises(ValueError, match="is not a commit of"):
        _amend(w, sacct, commit="e" * 40)
    assert not ST.read_amendments(w["ledger"])
    # a byte-identical supplied file is accepted (recorded as the cross-check it is); proven on a dry run, before the mint
    import hashlib
    old_b = subprocess.run(["git", "-C", str(w["repo"]), "show", f"{w['old_sha']}:configs/registry/decode_policy.json"], capture_output=True, check=True).stdout
    new_b = subprocess.run(["git", "-C", str(w["repo"]), "show", f"{w['new_sha']}:configs/registry/decode_policy.json"], capture_output=True, check=True).stdout
    good = tmp_path / "good_old_export.json"; good.write_bytes(old_b)
    dry = _amend(w, sacct, old_export_path=str(good), dry_run=True)
    assert dry["old_export_supplied"] == {"path": str(good), "sha256": hashlib.sha256(old_b).hexdigest()} and not ST.read_amendments(w["ledger"])
    # the happy path: commit == tip here; a later tip that CONTAINS the commit is fine too (checked through --recheck)
    entry = _amend(w, sacct)
    assert entry["commit"] == entry["commit_resolved"] == w["new_sha"] and entry["tip"] == w["new_sha"]
    assert entry["old_sha"] == w["old_sha"] and entry["old_sha_source"] == "the unit's recorded sha (ledger sha_provenance)"
    assert entry["old_export_sha256"] == hashlib.sha256(old_b).hexdigest() and entry["new_export_sha256"] == hashlib.sha256(new_b).hexdigest()
    assert entry["old_export_supplied"] is None
    assert entry["changed_instruments"] == {"crescendo": {"old": w["old_inst"], "new": w["real_inst"]}} and entry["contract"]["evals"]["crescendo"]["instrument_id"] == w["real_inst"]
    paths = {c["path"] for c in entry["row_diff"]}
    assert set(w["corrected"]) | {"params_sent", "notes"} <= paths and not any("temperature" in p for p in paths), paths
    # a key the filed row states at a non-neutral value (the demo twin's presence_penalty 1.5) stayed as filed: no transition, not in the diff
    assert paths.isdisjoint(set(ST.AMEND_ALLOWED_KEYS) - set(w["corrected"])), (paths, w["corrected"])
    ev = [e for e in _ledger(w["state"]) if e["event"] == "contract_amendment"]
    assert len(ev) == 1 and ev[0]["commit_resolved"] == w["new_sha"] and ev[0]["tip"] == w["new_sha"] and ev[0]["old_sha"] == w["old_sha"] \
        and ev[0]["old_export_sha256"] == entry["old_export_sha256"]
    assert ST.reconcile_provenance(w["ledger"]) == []
    # amending again at the same tree: the old tree is now the amendment's commit, and nothing changed since
    with pytest.raises(ValueError, match="nothing to amend"):
        _amend(w, sacct)
    assert len(ST.read_amendments(w["ledger"])[w["lw"]]) == 1
    # the frozen inventory file is untouched; the effective contract is the amended one
    assert [x for x in json.load(open(w["inv_p"]))["units"] if x["logical_work_id"] == w["lw"]][0]["evals"]["crescendo"]["instrument_id"] == w["old_inst"]
    assert ST.effective_contract(ST.read_amendments(w["ledger"]), w["unit_old"])["evals"]["crescendo"]["instrument_id"] == w["real_inst"]


def test_a_sidecar_that_disagrees_with_the_ledger_refuses_the_amendment(tmp_path):
    w = amended_world(tmp_path)
    side = ST.read_provenance_doc(w["ledger"]); side["units"][w["lw"]][-1]["sha"] = "c" * 40; ST.write_atomic(ST.provenance_path(w["ledger"]), side)
    with pytest.raises(ValueError, match="provenance sidecar disagrees with the attempts ledger"):
        _amend(w, FakeSacct({("cluster_a", "1"): "FAILED"}))


# ---------------------------------------------------------------- every old job terminal on sacct, every old run non-completed

def test_a_running_old_run_passes_only_when_its_job_is_sacct_proven_terminal_not_completed_and_the_proof_is_recorded(tmp_path):
    w = amended_world(tmp_path, run_state="running")
    # the job is alive: refused naming the job and its state; nothing written
    with pytest.raises(ValueError, match=r"job 1 on cluster_a \(attempt\) is still RUNNING on sacct — an old-contract job that is alive"):
        _amend(w, FakeSacct({("cluster_a", "1"): "RUNNING"}))
    # the job is unknown to sacct / the cluster unreachable: refused by name
    with pytest.raises(ValueError, match="job 1 on cluster_a .* is unknown to sacct"):
        _amend(w, FakeSacct({}))
    with pytest.raises(ValueError, match=r"job 1 on cluster_a .* sacct could not be read \(ssh cluster-login: Connection timed out\)"):
        _amend(w, FakeSacct({("cluster_a", "1"): "UNREACHABLE"}))
    # the job COMPLETED while the registry still says running: the run may have generated — refused
    with pytest.raises(ValueError, match="but its job 1 on cluster_a COMPLETED on sacct — the run may have generated"):
        _amend(w, FakeSacct({("cluster_a", "1"): "COMPLETED"}))
    assert not ST.read_amendments(w["ledger"]) and not [e for e in _ledger(w["state"]) if e["event"] == "contract_amendment"]
    # the job FAILED: the running runs pass, each with its job, sacct state and proof utc in the receipt AND the ledger event
    sacct = FakeSacct({("cluster_a", "1"): "FAILED"})
    entry = _amend(w, sacct)
    assert sacct.calls == [("cluster_a", "1")]
    assert entry["old_jobs"] == [{"cluster": "cluster_a", "job_id": "1", "sacct_states": ["FAILED"], "tasks": [{"raw": "1", "task": None, "state": "FAILED"}],
                                 "proof_utc": entry["old_jobs"][0]["proof_utc"]}] and entry["old_jobs"][0]["proof_utc"]
    ids = sorted(r["run_id"] for r in w["runs"])
    assert sorted(r["run_id"] for r in entry["old_run_ids"]) == ids
    for r in entry["old_run_ids"]:
        assert r["lifecycle_state"] == "running" and r["cluster"] == "cluster_a" and r["job_id"] == "1" and r["sacct_state"] == "FAILED" \
            and r["proof_utc"] == entry["old_jobs"][0]["proof_utc"] and "terminal, not COMPLETED" in r["why"], r
    chk = entry["completed_run_check"]
    assert chk["n_old_runs"] == len(ids) and chk["n_old_jobs"] == 1 and chk["completed"] == [] and chk["pre_fence_utc"] <= chk["post_fence_utc"]
    ev = [e for e in _ledger(w["state"]) if e["event"] == "contract_amendment"][0]
    assert ev["old_run_ids"] == entry["old_run_ids"] and ev["old_jobs"] == entry["old_jobs"] and ev["completed_run_check"] == chk


def test_terminal_non_completed_runs_pass_without_a_run_level_proof_but_the_job_is_still_proven(tmp_path):
    w = amended_world(tmp_path, run_state="failed")
    with pytest.raises(ValueError, match="job 1 on cluster_a .* is still PENDING on sacct"):
        _amend(w, FakeSacct({("cluster_a", "1"): "PENDING"}))
    entry = _amend(w, FakeSacct({("cluster_a", "1"): "CANCELLED"}))
    assert all(r["lifecycle_state"] == "failed" and r["why"].startswith("registry `failed` (terminal, not completed)") and r["sacct_state"] == "CANCELLED"
               for r in entry["old_run_ids"]), entry["old_run_ids"]


def test_a_running_run_bound_to_no_job_is_refused_and_a_registry_cluster_job_id_resolves_one(tmp_path):
    w = amended_world(tmp_path, run_state="running")
    # a run the ledger registered under an attempt the ledger never recorded (so no job binds it): refused by name
    L = GL.Ledger(str(w["state"] / "attempts.jsonl"))
    L.append({"event": "registered", "logical_work_id": w["lw"], "attempt_id": "ghost", "run_ids": ["run-orphan"]})
    runs = list(w["runs"]) + [{"run_id": "run-orphan", "lifecycle_state": "running", "model_id": MODEL, "instrument_id": w["old_inst"], "eval_id": "crescendo"}]
    db = make_registry(tmp_path / "reg2", runs, w["pols"])
    with pytest.raises(ValueError, match="run run-orphan is `running` in the registry and no job of the unit's attempts is bound to it"):
        _amend(w, FakeSacct({("cluster_a", "1"): "FAILED"}), registry_db=str(db))
    # the registry's own cluster_job_id names one of the unit's proven jobs: resolved through it
    runs[-1]["cluster_job_id"] = "1"
    db = make_registry(tmp_path / "reg3", runs, w["pols"])
    entry = _amend(w, FakeSacct({("cluster_a", "1"): "FAILED"}), registry_db=str(db))
    orphan = [r for r in entry["old_run_ids"] if r["run_id"] == "run-orphan"][0]
    assert orphan["job_id"] == "1" and orphan["sacct_state"] == "FAILED" and orphan["cluster"] == "cluster_a"
    # a cluster_job_id naming a job the unit never ran is refused by name
    w2 = amended_world(tmp_path / "w2", run_state="running")
    L2 = GL.Ledger(str(w2["state"] / "attempts.jsonl"))
    L2.append({"event": "registered", "logical_work_id": w2["lw"], "attempt_id": "ghost", "run_ids": ["run-foreign"]})
    runs2 = list(w2["runs"]) + [{"run_id": "run-foreign", "lifecycle_state": "running", "cluster_job_id": "999", "model_id": MODEL, "instrument_id": w2["old_inst"], "eval_id": "crescendo"}]
    with pytest.raises(ValueError, match="run run-foreign names cluster job 999, which is not among the unit's recorded attempts"):
        _amend(w2, FakeSacct({("cluster_a", "1"): "FAILED"}), registry_db=str(make_registry(tmp_path / "reg4", runs2, w2["pols"])))


def test_old_runs_no_binding_or_event_ever_listed_are_found_in_the_registry_by_the_units_job_ids(tmp_path):
    """The real ledger's shape on 2026-09-30: the unit's second attempt registered at job start and died before 06 --gate bound it, so
    its runs lived only in the registry (18 of the 36 old-contract runs were absent from the twelve receipts). The registry is asked
    for every run whose own cluster_job_id names one of the unit's old jobs — before the proofs (parent ids) and after them (every
    task's raw id) — and those runs are proven and listed like the rest; a completed one refuses before any sacct call."""
    w = amended_world(tmp_path, run_state="running")
    L = GL.Ledger(str(w["state"] / "attempts.jsonl"))
    L.append({"event": "attempt", "logical_work_id": w["lw"], "attempt_id": f"{w['lw']}@cluster_a@7", "cluster": "cluster_a", "job_id": "7", "sha": w["old_sha"],
              "utc": "2026-01-01T00:01:00Z"})          # the second attempt: a `submitted`-only job, nothing bound
    ghost = [{"run_id": "run-ghost-a", "lifecycle_state": "running", "cluster_job_id": "7", "model_id": MODEL, "instrument_id": w["old_inst"], "eval_id": "crescendo"},
             {"run_id": "run-ghost-b", "lifecycle_state": "running", "cluster_job_id": "71", "model_id": MODEL, "instrument_id": w["old_inst"], "eval_id": "crescendo"}]
    db = make_registry(tmp_path / "reg2", list(w["runs"]) + ghost, w["pols"])
    tasks7 = [("70", 0, "FAILED"), ("71", 1, "TIMEOUT")]
    sacct = FakeSacct({("cluster_a", "1"): "FAILED", ("cluster_a", "7"): tasks7})
    entry = _amend(w, sacct, registry_db=str(db))
    by_id = {r["run_id"]: r for r in entry["old_run_ids"]}
    assert "run-ghost-a" in by_id and by_id["run-ghost-a"]["job_id"] == "7" and by_id["run-ghost-a"]["sacct_state"] == "FAILED,TIMEOUT"
    assert "run-ghost-b" in by_id and by_id["run-ghost-b"]["job_id"] == "7" and by_id["run-ghost-b"]["sacct_state"] == "TIMEOUT"    # the raw task id names the task
    assert entry["completed_run_check"]["n_runs_from_registry_by_job"] == 2 and entry["completed_run_check"]["n_old_runs"] == len(w["runs"]) + 2 \
        and entry["completed_run_check"]["n_old_jobs"] == 2
    ev = [e for e in _ledger(w["state"]) if e["event"] == "contract_amendment"][0]
    assert {r["run_id"] for r in ev["old_run_ids"]} == set(by_id)
    # a registry-only run that COMPLETED refuses before any sacct call (the parent id is enough to find it)
    w2 = amended_world(tmp_path / "w2", run_state="running")
    GL.Ledger(str(w2["state"] / "attempts.jsonl")).append({"event": "attempt", "logical_work_id": w2["lw"], "attempt_id": f"{w2['lw']}@cluster_a@7", "cluster": "cluster_a",
                                                            "job_id": "7", "sha": w2["old_sha"], "utc": "2026-01-01T00:01:00Z"})
    done = [{"run_id": "run-ghost-c", "lifecycle_state": "generated", "cluster_job_id": "7", "model_id": MODEL, "instrument_id": w2["old_inst"], "eval_id": "crescendo"}]
    sacct2 = FakeSacct({("cluster_a", "1"): "FAILED", ("cluster_a", "7"): "FAILED"})
    with pytest.raises(ValueError, match=r"1 run\(s\) COMPLETED under the old contract \(\['run-ghost-c'\]\)"):
        _amend(w2, sacct2, registry_db=str(make_registry(tmp_path / "reg3", list(w2["runs"]) + done, w2["pols"])))
    assert sacct2.calls == []
    # --recheck finds them too: the receipt above listed the ghosts; a NEW registry-only run under job 7 that is `generated` fails the receipt by name
    _set_state(db, "run-ghost-b", "generated")
    res = ST.recheck_amendments(w["ledger"], w["lw"], repo=str(w["repo"]), tip=w["new_sha"], registry_db=str(db), sacct_tasks=sacct)
    assert res[0]["verdict"] == "FAIL" and any("run run-ghost-b is `generated`" in p for p in res[0]["problems"]), res[0]["problems"]
    _set_state(db, "run-ghost-b", "running")
    c = sqlite3.connect(db); c.execute("insert into run (run_id, lifecycle_state, cluster_job_id, decode_policy_match) values ('run-ghost-d', 'running', '70', 'exact')"); c.commit(); c.close()
    res = ST.recheck_amendments(w["ledger"], w["lw"], repo=str(w["repo"]), tip=w["new_sha"], registry_db=str(db), sacct_tasks=sacct)
    assert res[0]["verdict"] == "PASS" and res[0]["n_runs_from_registry_by_job"] == 1 and res[0]["n_old_runs"] == len(w["runs"]) + 3, res[0]
    assert [r for r in res[0]["old_run_ids"] if r["run_id"] == "run-ghost-d"][0]["sacct_state"] == "FAILED"


def test_an_array_job_is_proven_per_task_and_a_run_names_its_task(tmp_path):
    w = amended_world(tmp_path, run_state="running")
    tasks = [("11", 0, "FAILED"), ("12", 1, "TIMEOUT"), ("13", 2, "RUNNING")]
    with pytest.raises(ValueError, match=r"job 1 on cluster_a \(attempt\) is still task 2 \(13\) RUNNING on sacct"):
        _amend(w, FakeSacct({("cluster_a", "1"): tasks}))
    tasks[2] = ("13", 2, "NODE_FAIL")
    entry = _amend(w, FakeSacct({("cluster_a", "1"): tasks}))
    assert entry["old_jobs"][0]["sacct_states"] == ["FAILED", "NODE_FAIL", "TIMEOUT"] and [t["raw"] for t in entry["old_jobs"][0]["tasks"]] == ["11", "12", "13"]
    assert all(r["sacct_state"] == "FAILED,NODE_FAIL,TIMEOUT" for r in entry["old_run_ids"])


def test_the_registry_is_re_read_after_the_sacct_fence_and_a_run_completed_meanwhile_refuses(tmp_path):
    """The interval: the read-only snapshot closed before the ledger event, so a job still writing could reach `generated` in
    between. The sacct proofs are the fence; the registry is re-read after them, right before the write."""
    w = amended_world(tmp_path, run_state="running")
    victim = w["runs"][0]["run_id"]
    sacct = FakeSacct({("cluster_a", "1"): "FAILED"}, on_read=lambda: _set_state(w["db"], victim, "generated"))
    with pytest.raises(ValueError, match=f"1 run\\(s\\) COMPLETED under the old contract AFTER the sacct proofs were gathered \\(\\['{victim}'\\]\\)"):
        _amend(w, sacct)
    assert sacct.calls == [("cluster_a", "1")]
    assert not ST.read_amendments(w["ledger"]) and not [e for e in _ledger(w["state"]) if e["event"] == "contract_amendment"]
    # a run already completed BEFORE the proofs refuses before any sacct call is made
    w2 = amended_world(tmp_path / "w2", run_state="judged")
    sacct2 = FakeSacct({("cluster_a", "1"): "FAILED"})
    with pytest.raises(ValueError, match="COMPLETED under the old contract .* an amendment is admissible only when none did"):
        _amend(w2, sacct2)
    assert sacct2.calls == []
    # a missing registry fails closed, before the proofs
    with pytest.raises(ValueError, match="fail closed"):
        _amend(w, FakeSacct({("cluster_a", "1"): "FAILED"}), registry_db=str(tmp_path / "nope.sqlite"))


def test_a_reason_without_a_d_number_and_a_dry_run_write_nothing(tmp_path):
    w = amended_world(tmp_path)
    with pytest.raises(ValueError, match="D-number"):
        _amend(w, FakeSacct({("cluster_a", "1"): "FAILED"}), reason="no ruling cited")
    entry = _amend(w, FakeSacct({("cluster_a", "1"): "FAILED"}), dry_run=True)
    assert entry["changed_instruments"]["crescendo"]["new"] == w["real_inst"]
    assert not ST.read_amendments(w["ledger"]) and not [e for e in _ledger(w["state"]) if e["event"] == "contract_amendment"]


# ---------------------------------------------------------------------------------------- --recheck: the three proofs over a receipt

def test_recheck_re_runs_the_three_proofs_over_a_recorded_receipt_and_fails_it_when_the_world_moved(tmp_path):
    w = amended_world(tmp_path, run_state="running")
    entry = _amend(w, FakeSacct({("cluster_a", "1"): "FAILED"}))
    # the re-run under the amended contract: a NEW attempt (job 2) that is still running must NOT be part of the receipt's proof
    L = GL.Ledger(str(w["state"] / "attempts.jsonl"))
    L.append({"event": "attempt", "logical_work_id": w["lw"], "attempt_id": f"{w['lw']}@cluster_a@2", "cluster": "cluster_a", "job_id": "2", "sha": w["new_sha"],
              "utc": "2027-01-01T00:00:00Z"})
    # a later tip that CONTAINS the commit passes
    (w["repo"] / "later.txt").write_text("y\n"); _git(w["repo"], "add", "-A"); _git(w["repo"], "commit", "-q", "-m", "later"); later = _git(w["repo"], "rev-parse", "HEAD")
    sacct = FakeSacct({("cluster_a", "1"): "FAILED", ("cluster_a", "2"): "RUNNING"})
    res = ST.recheck_amendments(w["ledger"], w["lw"], repo=str(w["repo"]), tip=later, registry_db=str(w["db"]), sacct_tasks=sacct)
    assert len(res) == 1 and res[0]["verdict"] == "PASS" and res[0]["problems"] == [], res
    assert res[0]["old_sha"] == w["old_sha"] and res[0]["old_export_sha256"] == entry["old_export_sha256"] and res[0]["n_old_jobs"] == 1 and res[0]["n_old_runs"] == len(w["runs"])
    assert sacct.calls == [("cluster_a", "1")]              # job 2 postdates the receipt: not the receipt's to prove
    assert not [e for e in _ledger(w["state"]) if e["event"] == "contract_amendment"][1:]   # nothing written
    # the world moved: an old run has since COMPLETED -> FAIL by name; the tip no longer contains the commit -> FAIL by name
    victim = w["runs"][0]["run_id"]; _set_state(w["db"], victim, "generated")
    res = ST.recheck_amendments(w["ledger"], w["lw"], repo=str(w["repo"]), tip=later, registry_db=str(w["db"]), sacct_tasks=sacct)
    assert res[0]["verdict"] == "FAIL" and any(f"run {victim} is `generated` — it COMPLETED under the old contract" in p for p in res[0]["problems"]), res[0]["problems"]
    _set_state(w["db"], victim, "running")
    res = ST.recheck_amendments(w["ledger"], w["lw"], repo=str(w["repo"]), tip=w["old_sha"], registry_db=str(w["db"]), sacct_tasks=sacct)
    assert res[0]["verdict"] == "FAIL" and any("is not an ancestor-or-equal of the tip" in p for p in res[0]["problems"]), res[0]["problems"]
    # the old job now reads RUNNING (a requeue): FAIL by name
    res = ST.recheck_amendments(w["ledger"], w["lw"], repo=str(w["repo"]), tip=later, registry_db=str(w["db"]), sacct_tasks=FakeSacct({("cluster_a", "1"): "RUNNING"}))
    assert res[0]["verdict"] == "FAIL" and any("job 1 on cluster_a" in p and "still RUNNING" in p for p in res[0]["problems"]), res[0]["problems"]


def test_recheck_re_validates_a_receipt_minted_before_the_fix_from_the_freeze_sha(tmp_path):
    """The shape of the receipts recorded on 2026-09-30 18:14Z: no old_sha / export hashes / job proofs in the entry, attempts that predate
    the provenance events (the unit's sha is the freeze sha), old runs in `running`. --recheck derives the old tree from the inventory's
    code_sha and proves the jobs and runs now."""
    w = amended_world(tmp_path, run_state="running", provenance="freeze")
    entry = _amend(w, FakeSacct({("cluster_a", "1"): "FAILED"}))
    assert entry["old_sha_source"] == "the freeze sha (no attempt recorded a sha)"
    # strip the receipt to the pre-fix shape (sidecar + ledger event keep the (commit, old, new) triple the reconciler pins)
    doc = ST.read_provenance_doc(w["ledger"])
    e = doc["amendments"][w["lw"]][0]
    for k in ("commit_resolved", "tip", "old_sha", "old_sha_source", "old_export_sha256", "new_export_sha256", "old_export_supplied", "old_jobs"):
        e.pop(k, None)
    e["old_run_ids"] = [{"run_id": r["run_id"], "lifecycle_state": r["lifecycle_state"]} for r in e["old_run_ids"]]
    e["completed_run_check"] = {"registry": str(w["db"]), "n_old_runs": len(e["old_run_ids"]), "completed": []}
    ST.write_atomic(ST.provenance_path(w["ledger"]), doc)
    # ... and the LEDGER event to the same pre-fix shape (commit + the two contract hashes): a legacy receipt is minimal on BOTH sides —
    # the reconciler compares every proof field the ledger event carries, so a stripped sidecar against a full ledger event is a forgery
    att_p = w["state"] / "attempts.jsonl"
    rows = [json.loads(l) for l in att_p.read_text().splitlines() if l.strip()]
    for r in rows:
        if r.get("event") == "contract_amendment":
            for k in ("commit_resolved", "tip", "old_sha", "old_export_sha256", "new_export_sha256", "old_jobs", "row_diff", "changed_instruments"):
                r.pop(k, None)
    att_p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert ST.reconcile_provenance(w["ledger"]) == []
    sacct = FakeSacct({("cluster_a", "1"): "FAILED"})
    res = ST.recheck_amendments(w["ledger"], w["lw"], repo=str(w["repo"]), tip=w["new_sha"], registry_db=str(w["db"]), sacct_tasks=sacct)
    assert res[0]["verdict"] == "PASS" and res[0]["old_sha"] == w["old_sha"] and res[0]["n_old_jobs"] == 1 and sacct.calls == [("cluster_a", "1")], res
    assert all(r["job_id"] == "1" and r["sacct_state"] == "FAILED" for r in res[0]["old_run_ids"])


# ------------------------------------------------------------------------------------------------- the CLI, with the funnel's reader

def test_cli_derives_everything_from_git_reads_sacct_through_the_funnel_and_rechecks(tmp_path):
    """End to end through `stage.py amend-contract` with the REAL reader (submit_twin.Remote over a `local:` fake sacct): --tip defaults to
    HEAD of --repo; the old export is derived (no --old-sha exists any more); the receipt records the proofs; --recheck re-runs them."""
    w = amended_world(tmp_path, run_state="running")
    fake = {lab: tmp_path / f"fake_{lab}" for lab in ("cluster_a", "cluster_b")}
    for d in fake.values():
        (d / "jobs").mkdir(parents=True)
    bin_dir = write_fake_slurm(tmp_path / "bin")
    site = write_site(tmp_path, ssh={lab: f"local:{d}" for lab, d in fake.items()})
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    for k in ("SISTEER_SITE_DIR", "SISTEER_CONFIG"):
        env.pop(k, None)
    base = [VENV_PY or sys.executable, str(LIB / "stage.py"), "amend-contract", "--ledger", w["ledger"], "--logical-work-id", w["lw"], "--registry", str(w["db"]),
            "--config", str(site / "config.env"), "--repo", str(w["repo"])]
    # the job is alive on the fake cluster: refused (rc 3) by name, nothing written
    set_job_state(fake["cluster_a"], "1", "RUNNING"); (fake["cluster_a"] / "jobs" / "1.name").write_text("mt-crescendo-g012\n")
    r = subprocess.run([*base, "--reason", REASON, "--commit", w["new_sha"]], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 3 and "REFUSED: amend-contract:" in r.stderr and "job 1 on cluster_a" in r.stderr and "still RUNNING on sacct" in r.stderr, r.stdout + r.stderr
    assert not ST.read_amendments(w["ledger"])
    # --old-sha is gone from the surface; --recheck with no receipt is rc 2
    r = subprocess.run([*base, "--reason", REASON, "--commit", w["new_sha"], "--old-sha", w["old_sha"]], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 2 and "unrecognized arguments: --old-sha" in r.stderr, r.stderr
    r = subprocess.run([*base, "--recheck"], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 2 and "has no receipted amendment" in r.stderr, r.stderr
    r = subprocess.run([*base, "--commit", w["new_sha"]], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 2 and "needs --reason and --commit" in r.stderr, r.stderr
    # the job died: minted, with --tip defaulting to HEAD of --repo (== the commit here); the receipt names the derivation and the proof
    set_job_state(fake["cluster_a"], "1", "FAILED", end="2026-01-01T01:00:00")
    r = subprocess.run([*base, "--reason", REASON, "--commit", w["new_sha"]], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0 and f"old export derived from {w['old_sha'][:12]}" in r.stdout and "1 old job(s) terminal on sacct" in r.stdout \
        and f"commit {w['new_sha'][:12]} in tip {w['new_sha'][:12]}" in r.stdout and "none completed (registry re-read" in r.stdout, r.stdout + r.stderr
    am = ST.read_amendments(w["ledger"])[w["lw"]]
    assert len(am) == 1 and am[0]["tip"] == w["new_sha"] and am[0]["old_sha"] == w["old_sha"] and am[0]["old_jobs"][0]["sacct_states"] == ["FAILED"] \
        and all(x["job_id"] == "1" and x["sacct_state"] == "FAILED" for x in am[0]["old_run_ids"])
    # --recheck: PASS (rc 0) with the proofs named; an old run that has since completed -> FAIL (rc 3) by name; nothing written either way
    r = subprocess.run([*base, "--recheck"], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0 and r.stdout.startswith(f"RECHECK {w['lw']} #0 commit {w['new_sha']}") and ": PASS" in r.stdout \
        and f"old export derived from {w['old_sha'][:12]}" in r.stdout and "1 old job(s) terminal on sacct" in r.stdout, r.stdout + r.stderr
    victim = w["runs"][0]["run_id"]; _set_state(w["db"], victim, "generated")
    r = subprocess.run([*base, "--recheck", "--tip", w["new_sha"]], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 3 and ": FAIL" in r.stdout and f"run {victim} is `generated`" in r.stdout, r.stdout + r.stderr
    assert len(ST.read_amendments(w["ledger"])[w["lw"]]) == 1 and len([e for e in _ledger(w["state"]) if e["event"] == "contract_amendment"]) == 1


def test_recheck_derives_the_old_sha_itself_and_refuses_a_receipt_whose_old_sha_or_export_hash_is_foreign(tmp_path):
    """--recheck used to take a present `old_sha` from the mutable sidecar and only checked that the receipt's
    export hash matched the export at that SAME sha — a repaired/copied receipt with a wrong old sha and its matching export hash passed
    while omitting a real non-neutral old -> new change. The old tree is now derived for EVERY receipt (the ledger's recorded sha / the
    prior receipt / the freeze sha) and a recorded old_sha or export hash that differs from the derivation fails by name; the reconciler
    compares every proof field entry for entry."""
    w = amended_world(tmp_path, run_state="running")
    _amend(w, FakeSacct({("cluster_a", "1"): "FAILED"}))
    sacct = FakeSacct({("cluster_a", "1"): "FAILED"})
    ok = ST.recheck_amendments(w["ledger"], w["lw"], repo=str(w["repo"]), tip=w["new_sha"], registry_db=str(w["db"]), sacct_tasks=sacct)
    assert ok[0]["verdict"] == "PASS" and ok[0]["old_sha"] == w["old_sha"] and ok[0]["old_sha_source"].startswith("the unit's recorded sha"), ok
    # forge the sidecar receipt: old_sha := the NEW commit, old_export_sha256 := the hash of the export at that (wrong) sha — the old tool's blind spot
    doc = ST.read_provenance_doc(w["ledger"]); e = doc["amendments"][w["lw"]][0]
    wrong = w["new_sha"]; wrong_hash = hashlib.sha256(ST.git_show_bytes(str(w["repo"]), wrong, ST.DECODE_EXPORT_REL)).hexdigest()
    e["old_sha"] = wrong; e["old_export_sha256"] = wrong_hash
    ST.write_atomic(ST.provenance_path(w["ledger"]), doc)
    # the reconciler names the disagreement with the ledger (old_sha and the export hash), so the mint refuses too
    probs = ST.reconcile_provenance(w["ledger"])
    assert probs and "contract amendments disagree with the attempts ledger's" in probs[0] and "old_sha" in probs[0] and "old_export_sha256" in probs[0], probs
    with pytest.raises(ValueError, match="disagree with the attempts ledger"):
        _amend(w, FakeSacct({("cluster_a", "1"): "FAILED"}))
    # --recheck derives the old tree (the ledger's recorded sha) and fails the receipt by name on both foreign fields
    res = ST.recheck_amendments(w["ledger"], w["lw"], repo=str(w["repo"]), tip=w["new_sha"], registry_db=str(w["db"]), sacct_tasks=sacct)
    assert res[0]["verdict"] == "FAIL" and res[0]["old_sha"] == w["old_sha"], res
    assert any(f"the receipt's old sha {wrong[:12]} is not the derivation {w['old_sha'][:12]}" in p for p in res[0]["problems"]), res[0]["problems"]
    assert any(f"the receipt's old_export_sha256 {wrong_hash[:12]} is not the export at {w['old_sha'][:12]}" in p for p in res[0]["problems"]), res[0]["problems"]
