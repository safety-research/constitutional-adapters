"""`stage.py amend-contract --kind petri-role-fallback` (commit 32): the Petri unit's RECEIPTED sha advance for the
per-request refusal fallback.

Petri attempt 7 showed safety-classifier refusals; the fallback is a per-request downgrade (auditor Sonnet 5 -> Sonnet 4.6, judge Opus 5 ->
Opus 4.8, defaults unchanged). It lands as TWO ADDED keys in the drive manifest (`petri.auditor_refusal_fallback_models`,
`petri.judge_refusal_fallback_models`, routed to Inspect's GenerateConfig.fallback_models); the existing `auditor_fallback_models` chain is
a harness-hash input and does not move, so the Petri instrument id and every petri_run_key are identical at the amending commit. Because the
drive runs the checkout at the unit's sha and restarts against the SAME serving job (no funnel attempt is minted), the unit's sha advances
through this receipt: the manifest diff must be exactly the two admitted transitions (any other field refuses by name), the contract must be
unchanged or Petri instrument ids alone, no run may have COMPLETED under the old contract, the serving job may be ALIVE (recorded), the
receipt names before -> after per role and the note, the sha table shows `amended` at the commit, and WAVE.<view>.json's unit entry follows."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, REPRO, VENV_PY, make_registry, set_job_state, write_fake_slurm, write_site  # noqa: E402
from test_admission import MODEL, by_unit, freeze, satisfy  # noqa: E402
from test_amend_contract_enforceable import REASON as NEUTRAL_REASON, FakeSacct, _git, _ledger, _set_state, amended_world  # noqa: E402

sys.path.insert(0, str(LIB))
import grant_ledger as GL  # noqa: E402
import models as M  # noqa: E402
import stage as ST  # noqa: E402

REASON = "D354/D355: per-request refusal fallback (auditor Sonnet 5 -> 4.6, judge Opus 5 -> 4.8, defaults unchanged); D442; D447 — the drive restarts at the amending commit"
# the unit's wave is the model's BASELINE view — the view the frozen logical work id carries and the one petri_manifest_rel reads the drive manifest
# for: `u0` when the table declares it (the phased recipe), else the single-phase recipe's one steered wave `s<N>` (the demo key: s1) — the stage's
# own helper (models.baseline_view), never a literal, so the fixture holds for a phased paper key and the single-phase demo alike
VIEW = M.baseline_view(MODEL, str(REPRO / "models.yaml"))
MANIFEST_REL = f"configs/capeval/manifests/petri_full22_{MODEL}_{VIEW}.yaml"          # the unit's drive manifest, in the synthetic checkout
WAVE_FILE = f"WAVE.{VIEW}.json"                                                        # the wave's per-unit sha table beside the ledger (06 writes it)
# its CONTENT: a committed Petri drive manifest from the paper battery (the front-door model is configured, not run: the tree carries no manifest of its wave)
_MANIFESTS = sorted((REPO / "configs" / "capeval" / "manifests").glob("petri_full22_*.yaml"))
MANIFEST_SRC = next((p for p in _MANIFESTS if p.name == "petri_full22_gemma4e4b.yaml"), _MANIFESTS[0] if _MANIFESTS else None)
assert MANIFEST_SRC is not None, "no committed Petri drive manifest to seed the synthetic checkout from"
BEFORE_CHAIN = ["openrouter/x-ai/grok-4.6"]
AFTER = {"auditor_refusal_fallback_models": ["anthropic/claude-sonnet-4-6"], "judge_refusal_fallback_models": ["anthropic/claude-opus-4-8"]}


def _manifest(old: bool, *, extra=None, mutate=None) -> str:
    doc = yaml.safe_load(open(MANIFEST_SRC))
    p = doc["petri"]
    p["auditor_fallback_models"] = list(BEFORE_CHAIN)
    for k in AFTER:
        p.pop(k, None)
    if not old:
        p.update({k: list(v) for k, v in AFTER.items()})
        for k, v in (extra or {}).items():
            p[k] = v
        if mutate:
            mutate(doc)
    return yaml.safe_dump(doc, sort_keys=False)


def petri_repo(tmp: Path, *, extra=None, mutate=None, seed_mutate=None) -> dict:
    """A synthetic checkout holding the REAL registry seeds and the baseline wave's Petri drive manifest, committed twice: the frozen tree (the chain
    before the ruling, no refusal-fallback keys) and the amending commit X (the two added keys; `extra` / `mutate` / `seed_mutate` push the
    diff beyond the admitted one)."""
    repo = tmp / "repo"; (repo / "configs" / "capeval" / "manifests").mkdir(parents=True)
    shutil.copytree(REPO / "configs" / "registry", repo / "configs" / "registry")
    man = repo / MANIFEST_REL
    _git(repo, "init", "-q", "-b", "main")
    man.write_text(_manifest(True)); _git(repo, "add", "-A"); _git(repo, "commit", "-q", "-m", "the frozen tree")
    old_sha = _git(repo, "rev-parse", "HEAD")
    man.write_text(_manifest(False, extra=extra, mutate=mutate))
    if seed_mutate:
        seed_mutate(repo)
    _git(repo, "add", "-A"); _git(repo, "commit", "-q", "-m", "the per-request refusal fallback (X)")
    return {"repo": repo, "old_sha": old_sha, "new_sha": _git(repo, "rev-parse", "HEAD")}


def petri_world(tmp: Path, **kw) -> dict:
    """The Petri unit frozen at the old tree with NO sha_provenance event (the live unit's shape), its serving job `1` on cluster_a bound,
    one `running` run per Petri eval registered, the synthetic two-commit repo, and a WAVE.<view>.json carrying the sha table."""
    state = tmp / "state"; doc = freeze(state, families=["petri"]); unit = by_unit(doc)["petri/proxy"]; lw = unit["logical_work_id"]
    r = petri_repo(tmp, **kw)
    inv_p = state / "gate" / "expected_units.json"; inv = json.load(open(inv_p))
    for x in inv["units"]:
        x["code_sha"] = r["old_sha"]
    inv["code_sha"] = r["old_sha"]; ST.write_atomic(str(inv_p), inv)
    unit = dict(unit); unit["code_sha"] = r["old_sha"]
    runs, pols = [], []
    satisfy(state, unit, runs, pols, code_sha=r["old_sha"])
    for x in runs:
        x["lifecycle_state"] = "running"
    GL.Ledger(str(state / "attempts.jsonl")).append({"event": "attempt", "logical_work_id": lw, "attempt_id": f"{lw}@cluster_a@1", "cluster": "cluster_a", "job_id": "1",
                                                      "job_name": "f22-petri-m", "kind": "target", "sha": r["old_sha"], "utc": "2026-01-01T00:00:00Z"})
    db = make_registry(tmp / "reg", runs, pols)
    ledger = str(state / "submissions.jsonl")
    ST.write_atomic(str(state / WAVE_FILE), {"model": MODEL, "view": VIEW, "wave_sha": r["old_sha"], "unit_shas": ST.sha_table(ledger)})
    return {"state": state, "ledger": ledger, "lw": lw, "unit": unit, "runs": runs, "db": db, **r}


def _amend(w: dict, sacct, **kw):
    args = dict(reason=REASON, commit=w["new_sha"], tip=w["new_sha"], repo=str(w["repo"]), registry_db=str(w["db"]), sacct_tasks=sacct, kind="petri-role-fallback")
    args.update(kw)
    return ST.amend_contract(w["ledger"], w["lw"], **args)


def _nothing_written(w: dict) -> None:
    assert not ST.read_amendments(w["ledger"]) and not [e for e in _ledger(w["state"]) if e["event"] in ("contract_amendment", "sha_provenance")]
    assert ST.sha_table(w["ledger"])[w["lw"]]["sha"] == w["old_sha"]


def test_the_exact_per_request_fallback_diff_is_receipted_the_sha_advances_and_the_live_serving_job_is_recorded(tmp_path):
    w = petri_world(tmp_path); sacct = FakeSacct({("cluster_a", "1"): "RUNNING"})
    h = ST.contract_sha256(w["unit"])
    e = _amend(w, sacct, note="attempt 7: 14 safety-classifier refusals; 46 samples completed")
    assert e["kind"] == "petri-role-fallback" and e["changed_instruments"] == {} and e["old_contract_sha256"] == e["new_contract_sha256"] == h
    assert e["role_diff"] == [{"field": "petri.auditor_refusal_fallback_models", "before": None, "after": AFTER["auditor_refusal_fallback_models"]},
                              {"field": "petri.judge_refusal_fallback_models", "before": None, "after": AFTER["judge_refusal_fallback_models"]}]
    assert e["manifest_rel"] == MANIFEST_REL and len(e["old_manifest_sha256"]) == 64 and e["old_manifest_sha256"] != e["new_manifest_sha256"]
    assert e["old_sha"] == w["old_sha"] and e["old_sha_source"].startswith("the freeze sha") and e["sha_advance"] == {"from": w["old_sha"], "to": w["new_sha"]}
    assert e["note"].startswith("attempt 7") and e["decode_policy_ref"] is None and e["row_diff"] == []
    assert e["old_jobs"] == [{"cluster": "cluster_a", "job_id": "1", "sacct_states": ["RUNNING"], "tasks": [{"raw": "1", "task": None, "state": "RUNNING"}], "proof_utc": e["old_jobs"][0]["proof_utc"]}]
    assert e["live_jobs"] == [{"cluster": "cluster_a", "job_id": "1", "sacct_states": ["RUNNING"], "proof_utc": e["old_jobs"][0]["proof_utc"]}]
    # the 7 `running` runs are CARRIED (same run id, same instrument: the resumed drive fills them), never superseded
    assert len(e["old_run_ids"]) == len(w["runs"]) == 7 and all(x["lifecycle_state"] == "running" and x["carried"] is True and "carried under the amended contract" in x["why"] for x in e["old_run_ids"])
    assert e["carried_run_ids"] == sorted(r["run_id"] for r in w["runs"]) and e["completed_run_check"]["completed"] == [] and e["completed_run_check"]["n_old_runs"] == 7
    # the ledger: the receipt FIRST, then the sha advance (kind `amend`, no attempt, no job); the sidecar agrees with the ledger
    ev = [x for x in _ledger(w["state"]) if x["event"] in ("contract_amendment", "sha_provenance")]
    assert [x["event"] for x in ev] == ["contract_amendment", "sha_provenance"]
    assert ev[0]["kind"] == "petri-role-fallback" and ev[0]["role_diff"] == e["role_diff"] and ev[0]["manifest_rel"] == MANIFEST_REL and ev[0]["live_jobs"] == e["live_jobs"] and ev[0]["note"] == e["note"]
    assert ev[0]["carried_run_ids"] == e["carried_run_ids"]
    assert ev[1] == {**{k: ev[1][k] for k in ("event", "logical_work_id", "utc")}, "sha": w["new_sha"], "kind": "amend", "attempt_id": None, "job_id": None, "cluster": None,
                     "answers": {"amendment_utc": e["utc"], "kind": "petri-role-fallback", "commit": w["new_sha"]}, "fix_commit": w["new_sha"], "contract_sha256": h, "from_sha": w["old_sha"]}
    assert ST.reconcile_provenance(w["ledger"]) == []
    # the sha table and the wave file: `amended` at X; the effective contract is the receipt's (identical here)
    assert ST.sha_table(w["ledger"])[w["lw"]] == {"sha": w["new_sha"], "frozen_sha": w["old_sha"], "n_recorded": 1, "advanced": False, "amended": 1}
    wave = json.load(open(w["state"] / WAVE_FILE))
    assert wave["unit_shas"][w["lw"]]["sha"] == w["new_sha"] and wave["unit_shas"][w["lw"]]["amended"] == 1 and wave["wave_sha"] == w["old_sha"] and wave["unit_shas_refreshed_utc"]
    assert e["wave_files_refreshed"] == [str(w["state"] / WAVE_FILE)]
    assert ST.effective_contract(ST.read_amendments(w["ledger"]), w["unit"]) == e["contract"]
    # applied again: nothing to amend (the old tree is now X; the admitted transitions are absent from an empty diff), nothing written
    with pytest.raises(ValueError, match="absent from the diff"):
        _amend(w, sacct)
    assert len(ST.read_amendments(w["ledger"])[w["lw"]]) == 1 and len([x for x in _ledger(w["state"]) if x["event"] == "sha_provenance"]) == 1
    # --recheck re-derives the manifest diff and PASSES with the serving job still alive
    res = ST.recheck_amendments(w["ledger"], w["lw"], repo=str(w["repo"]), tip=w["new_sha"], registry_db=str(w["db"]), sacct_tasks=sacct)
    assert len(res) == 1 and res[0]["verdict"] == "PASS" and res[0]["kind"] == "petri-role-fallback" and res[0]["old_sha"] == w["old_sha"] \
        and res[0]["role_diff"] == ["petri.auditor_refusal_fallback_models", "petri.judge_refusal_fallback_models"] and res[0]["live_jobs"] == ["1@cluster_a:RUNNING"], res
    # a CARRIED run id completing AFTER the receipt is the amended drive's work: the predicate is evaluated at receipt time, the
    # recheck still PASSES and names the attribution (the carried Petri runs must complete)
    _set_state(w["db"], w["runs"][0]["run_id"], "judged")
    res = ST.recheck_amendments(w["ledger"], w["lw"], repo=str(w["repo"]), tip=w["new_sha"], registry_db=str(w["db"]), sacct_tasks=sacct)
    assert res[0]["verdict"] == "PASS", res
    done = [x for x in res[0]["old_run_ids"] if x["run_id"] == w["runs"][0]["run_id"]]
    assert done and done[0]["carried"] is True and "completed under the AMENDED contract" in done[0]["why"], done
    # the serving job ending after the receipt is no failure either (the member of the kind: recorded, not required)
    res = ST.recheck_amendments(w["ledger"], w["lw"], repo=str(w["repo"]), tip=w["new_sha"], registry_db=str(w["db"]), sacct_tasks=FakeSacct({("cluster_a", "1"): "COMPLETED"}))
    assert res[0]["verdict"] == "PASS" and res[0]["live_jobs"] == [], res


def test_a_diff_with_one_extra_field_or_a_moved_hash_input_refuses_by_name_and_writes_nothing(tmp_path):
    sacct = FakeSacct({("cluster_a", "1"): "RUNNING"})
    w = petri_world(tmp_path / "extra", extra={"fallback_mode": "per_request"})
    with pytest.raises(ValueError, match=r"petri\.fallback_mode changed \(None -> 'per_request'\) — not one of the admitted fields"):
        _amend(w, sacct)
    _nothing_written(w)
    w2 = petri_world(tmp_path / "chain", mutate=lambda d: d["petri"].__setitem__("auditor_fallback_models", ["anthropic/claude-sonnet-4-6"]))
    with pytest.raises(ValueError, match=r"petri\.auditor_fallback_models changed .* not one of the admitted fields .*harness-hash input"):
        _amend(w2, sacct)
    _nothing_written(w2)
    w3 = petri_world(tmp_path / "value", mutate=lambda d: d["petri"].__setitem__("judge_refusal_fallback_models", ["anthropic/claude-opus-5"]))
    with pytest.raises(ValueError, match=r"petri\.judge_refusal_fallback_models is None -> \['anthropic/claude-opus-5'\], not the admitted transition"):
        _amend(w3, sacct)
    _nothing_written(w3)
    w4 = petri_world(tmp_path / "partial", mutate=lambda d: d["petri"].pop("judge_refusal_fallback_models"))
    with pytest.raises(ValueError, match=r"admitted transition\(s\) of \['judge_refusal_fallback_models'\] are absent from the diff"):
        _amend(w4, sacct)
    _nothing_written(w4)
    w5 = petri_world(tmp_path / "toplevel", mutate=lambda d: d.__setitem__("extra_top", 1))
    with pytest.raises(ValueError, match="top-level `extra_top` changed — outside the `petri:` block"):
        _amend(w5, sacct)
    _nothing_written(w5)


def test_a_manifest_rel_override_that_is_not_the_units_drive_manifest_refuses_by_name_and_writes_nothing(tmp_path):
    w = petri_world(tmp_path); sacct = FakeSacct({("cluster_a", "1"): "RUNNING"})
    # another view's Petri manifest (the padded-static one) carrying the two additions proves nothing about THIS unit's drive
    other = f"configs/capeval/manifests/petri_full22_{MODEL}.padstatic.yaml"
    (w["repo"] / other).write_text((w["repo"] / MANIFEST_REL).read_text())
    _git(w["repo"], "add", "-A"); _git(w["repo"], "commit", "-q", "-m", "another view's manifest at X"); w["new_sha"] = _git(w["repo"], "rev-parse", "HEAD")
    with pytest.raises(ValueError, match=rf"--manifest-rel .*padstatic\.yaml' is not the unit's drive manifest '{re.escape(MANIFEST_REL)}'"):
        _amend(w, sacct, manifest_rel=other)
    _nothing_written(w)
    # the unit's own path, spelled explicitly (or with a redundant ./), is accepted
    e = _amend(w, sacct, manifest_rel="./" + MANIFEST_REL)
    assert e["manifest_rel"] == MANIFEST_REL and e["sha_advance"]["to"] == w["new_sha"]


def test_a_run_completed_under_the_old_contract_refuses_and_a_non_petri_unit_or_unknown_kind_refuses(tmp_path):
    w = petri_world(tmp_path); sacct = FakeSacct({("cluster_a", "1"): "RUNNING"})
    _set_state(w["db"], w["runs"][2]["run_id"], "generated")
    with pytest.raises(ValueError, match="1 run\\(s\\) COMPLETED under the old contract"):
        _amend(w, sacct)
    _nothing_written(w)
    with pytest.raises(ValueError, match="--kind 'nope' is not one of"):
        _amend(w, sacct, kind="nope")
    # the decode-neutral kind is untouched: a LIVE old job still refuses it by name
    n = amended_world(tmp_path / "neutral", run_state="running")
    with pytest.raises(ValueError, match="still RUNNING on sacct"):
        ST.amend_contract(n["ledger"], n["lw"], reason=NEUTRAL_REASON, commit=n["new_sha"], tip=n["new_sha"], repo=str(n["repo"]), registry_db=str(n["db"]),
                          sacct_tasks=FakeSacct({("cluster_a", "1"): "RUNNING"}), kind="decode-neutral")
    with pytest.raises(ValueError, match="applies to the Petri unit only"):
        ST.amend_contract(n["ledger"], n["lw"], reason=REASON, commit=n["new_sha"], tip=n["new_sha"], repo=str(n["repo"]), registry_db=str(n["db"]),
                          sacct_tasks=FakeSacct({("cluster_a", "1"): "RUNNING"}), kind="petri-role-fallback")


def test_petri_instrument_ids_that_move_with_the_commit_are_admitted_as_the_contract_change_and_nothing_else_is(tmp_path):
    def bump_turns(repo: Path):
        # an INSTRUMENT identity field of the Petri seed row (max_turns) moves -> every Petri eval mints a new instrument id at X
        p = repo / "configs" / "registry" / "instruments.yaml"; t = p.read_text()
        assert t.count("    audit_seed: 0\n    max_turns: 15\n") == 1
        p.write_text(t.replace("    audit_seed: 0\n    max_turns: 15\n", "    audit_seed: 0\n    max_turns: 14\n"))
    w = petri_world(tmp_path, seed_mutate=bump_turns); sacct = FakeSacct({("cluster_a", "1"): "RUNNING"})
    e = _amend(w, sacct)
    assert set(e["changed_instruments"]) == set(ST.PETRI_EVALS) and e["old_contract_sha256"] != e["new_contract_sha256"]
    # with the instrument moved the old runs cannot be carried: superseded, by name
    assert e["carried_run_ids"] == [] and all(x["carried"] is False and "superseded by the amendment — the instrument moved" in x["why"] for x in e["old_run_ids"])
    assert ST.effective_contract(ST.read_amendments(w["ledger"]), w["unit"]) == e["contract"] and ST.contract_dict_sha256(e["contract"]) == e["new_contract_sha256"]
    assert ST.sha_table(w["ledger"])[w["lw"]]["sha"] == w["new_sha"]
    # an UNCARRIED old id that completes after the receipt is an old-contract completion: the recheck FAILS by name
    res = ST.recheck_amendments(w["ledger"], w["lw"], repo=str(w["repo"]), tip=w["new_sha"], registry_db=str(w["db"]), sacct_tasks=sacct)
    assert res[0]["verdict"] == "PASS", res
    _set_state(w["db"], w["runs"][1]["run_id"], "generated")
    res = ST.recheck_amendments(w["ledger"], w["lw"], repo=str(w["repo"]), tip=w["new_sha"], registry_db=str(w["db"]), sacct_tasks=sacct)
    assert res[0]["verdict"] == "FAIL" and any(f"run {w['runs'][1]['run_id']} is `generated` — it COMPLETED under the old contract" in p for p in res[0]["problems"]), res


def test_the_cli_mints_the_petri_kind_against_a_live_fake_cluster_and_rechecks_it(tmp_path):
    w = petri_world(tmp_path)
    fake = {lab: tmp_path / f"fake_{lab}" for lab in ("cluster_a", "cluster_b")}
    for d in fake.values():
        (d / "jobs").mkdir(parents=True)
    bin_dir = write_fake_slurm(tmp_path / "bin")
    site = write_site(tmp_path, ssh={lab: f"local:{d}" for lab, d in fake.items()})
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    for k in ("SISTEER_SITE_DIR", "SISTEER_CONFIG"):
        env.pop(k, None)
    base = [VENV_PY or sys.executable, str(LIB / "stage.py"), "amend-contract", "--ledger", w["ledger"], "--logical-work-id", w["lw"], "--registry", str(w["db"]),
            "--config", str(site / "config.env"), "--repo", str(w["repo"]), "--kind", "petri-role-fallback"]
    set_job_state(fake["cluster_a"], "1", "RUNNING"); (fake["cluster_a"] / "jobs" / "1.name").write_text("f22-petri-m\n")
    r = subprocess.run([*base, "--reason", REASON, "--commit", w["new_sha"], "--note", "attempt 7: 46 samples", "--dry-run"], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0 and r.stdout.startswith("DRY-RUN: would amend") and "1 ALIVE" not in r.stdout and "1 on cluster_a RUNNING ALIVE" in r.stdout, r.stdout + r.stderr
    _nothing_written(w)
    r = subprocess.run([*base, "--reason", REASON, "--commit", w["new_sha"], "--note", "attempt 7: 46 samples"], capture_output=True, text=True, env=env, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0 and f"amended {w['lw']} (petri-role-fallback): manifest {MANIFEST_REL} {w['old_sha'][:12]} -> {w['new_sha'][:12]}" in r.stdout, out
    assert "petri.auditor_refusal_fallback_models None -> ['anthropic/claude-sonnet-4-6']" in r.stdout and "contract" in r.stdout and "(unchanged)" in r.stdout \
        and f"unit sha {w['old_sha'][:12]} -> {w['new_sha'][:12]} (sha table `amended`; WAVE unit_shas refreshed: 1)" in r.stdout and "7 carried under the amended contract" in r.stdout \
        and "note: attempt 7: 46 samples" in r.stdout, out
    assert ST.sha_table(w["ledger"])[w["lw"]]["sha"] == w["new_sha"] and json.load(open(w["state"] / WAVE_FILE))["unit_shas"][w["lw"]]["sha"] == w["new_sha"]
    r = subprocess.run([*base, "--recheck"], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0 and ": PASS" in r.stdout, r.stdout + r.stderr


def _add_run(db: Path, run_id: str, *, state: str, cluster_job_id: str, code_sha) -> None:
    import sqlite3
    c = sqlite3.connect(db)
    c.execute("insert into run (run_id, lifecycle_state, code_sha, cluster_job_id, decode_policy_match) values (?,?,?,?,'unrecorded')", (run_id, state, code_sha, cluster_job_id))
    c.commit(); c.close()


def _set_sha(db: Path, run_id: str, sha) -> None:
    import sqlite3
    c = sqlite3.connect(db); c.execute("update run set code_sha=? where run_id=?", (sha, run_id)); c.commit(); c.close()


def test_a_run_an_old_job_registers_after_the_mint_is_refused_by_name_unless_its_recorded_provenance_is_the_amended_drives(tmp_path):
    """32c: a --recheck discovers the old serving job's runs as they stand NOW, so a run the still-live old job
    registered after the mint is absent from carried_run_ids. It is never attributed as carried: refused by name unless the row's code_sha
    is the amendment's commit (the amended drive's own registration), which is recorded as such and not carried."""
    w = petri_world(tmp_path); sacct = FakeSacct({("cluster_a", "1"): "RUNNING"})
    e = _amend(w, sacct, note="attempt 7")
    assert len(e["carried_run_ids"]) == 7
    rid = "11111111-2222-4333-8444-555555555599"
    _add_run(w["db"], rid, state="running", cluster_job_id="1", code_sha=w["old_sha"])          # the old job registers another run, at the OLD tree
    kw = dict(repo=str(w["repo"]), tip=w["new_sha"], registry_db=str(w["db"]), sacct_tasks=sacct)
    res = ST.recheck_amendments(w["ledger"], w["lw"], **kw)
    assert res[0]["verdict"] == "FAIL" and res[0]["n_runs_from_registry_by_job"] == 1 and res[0]["n_old_runs"] == 8, res
    assert any(rid in p and "ABSENT from the receipt's carried_run_ids" in p and "refused by name" in p for p in res[0]["problems"]), res[0]["problems"]
    line = [x for x in res[0]["old_run_ids"] if x["run_id"] == rid][0]
    assert line["carried"] is False and "absent from the receipt's carried set" in line["why"]
    assert [x["carried"] for x in res[0]["old_run_ids"] if x["run_id"] != rid] == [True] * 7                # the receipted seven stay carried
    assert ST.read_amendments(w["ledger"])[w["lw"]][0]["carried_run_ids"] == e["carried_run_ids"]          # the recheck writes nothing
    # no recorded provenance at all, or a short / non-hex one: refused too (fail closed)
    for sha in (None, "", "c1fdf04", "not-a-sha-at-all-xyz"):
        _set_sha(w["db"], rid, sha)
        res = ST.recheck_amendments(w["ledger"], w["lw"], **kw)
        assert res[0]["verdict"] == "FAIL" and any(rid in p and "ABSENT" in p for p in res[0]["problems"]), (sha, res[0]["problems"])
    # the live row's code_sha is the fold's LAST writer, not provenance (32c-b): this registry keeps no registration
    # history (no row_history), so an amended code_sha on the row proves nothing — refused by name, running or completed (the amended
    # drive's own registration is proven only by the FIRST `run` record: tests/test_recheck_original_registration.py, over the real fold)
    for sha in (w["new_sha"], w["new_sha"][:12]):
        _set_sha(w["db"], rid, sha)
        res = ST.recheck_amendments(w["ledger"], w["lw"], **kw)
        line = [x for x in res[0]["old_run_ids"] if x["run_id"] == rid][0]
        assert res[0]["verdict"] == "FAIL" and line["carried"] is False and "amended_registration" not in line and "no registration history" in line["why"], (sha, res)
        assert res[0]["n_runs_absent_from_receipt"] == 1 and any(rid in p and "fold's last writer" in p for p in res[0]["problems"]), res[0]["problems"]
    _set_state(w["db"], rid, "judged")
    res = ST.recheck_amendments(w["ledger"], w["lw"], **kw)
    assert res[0]["verdict"] == "FAIL" and any(rid in p and "COMPLETED under the old contract" in p for p in res[0]["problems"]), res[0]["problems"]
    # completed at the OLD tree and absent: an old-contract completion, as before
    _set_sha(w["db"], rid, w["old_sha"])
    res = ST.recheck_amendments(w["ledger"], w["lw"], **kw)
    assert res[0]["verdict"] == "FAIL" and any(rid in p and "COMPLETED under the old contract" in p for p in res[0]["problems"]), res[0]["problems"]
    # the registry row reader carries code_sha, and tolerates a registry without the column
    rows = ST.registry_run_rows(str(w["db"]), [rid, "11111111-2222-4333-8444-000000000000"])
    assert rows[rid] == {"lifecycle_state": "judged", "cluster_job_id": "1", "code_sha": w["old_sha"]} and rows["11111111-2222-4333-8444-000000000000"] is None
    import sqlite3
    c = sqlite3.connect(w["db"]); c.execute("alter table run drop column code_sha"); c.commit(); c.close()
    assert ST.registry_run_rows(str(w["db"]), [rid])[rid] == {"lifecycle_state": "judged", "cluster_job_id": "1", "code_sha": None}
    # the mint path is unchanged: with no carried set, every non-terminal old run is carried
    lines, probs = ST.petri_run_verdicts(["a", "b"], {"a": {"lifecycle_state": "running", "code_sha": None}, "b": {"lifecycle_state": "failed"}}, carried=True)
    assert probs == [] and lines[0]["carried"] is True and "carried" not in lines[1]
