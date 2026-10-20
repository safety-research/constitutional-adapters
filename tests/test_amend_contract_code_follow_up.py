"""`amend-contract --kind code-follow-up` (32d): a CODE-ONLY follow-up to a receipted sha advance of the Petri unit — a second receipt
(role_diff [], follow_up_of the prior receipt, the --note naming the findings it carries) admitted only when the commit is a strict
descendant of the amended sha, the drive manifest and the roster are unchanged, the contract and the decode-policy export are unchanged, and
the old run set and the carried set are the prior receipt's exactly; it advances the unit's sha again. Everything else refuses by name and
writes nothing; --recheck re-proves every receipt."""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402
import test_amend_contract_petri_role_fallback as P  # noqa: E402
from test_amend_contract_enforceable import FakeSacct, _git, _ledger, _set_state  # noqa: E402

import stage as ST  # noqa: E402

ROSTER_REL = "configs/sdfbase_full22_roster.yaml"
_PETRI_REPO = P.petri_repo          # the original builder (the tests monkeypatch P.petri_repo to the roster-carrying one below)


def _repo_with_roster(tmp: Path, **kw) -> dict:
    """petri_repo + the production roster committed in BOTH trees (the follow-up proves it unchanged across X -> X'')."""
    repo = tmp / "repo"; (repo / "configs").mkdir(parents=True)
    shutil.copy(REPO / ROSTER_REL, repo / ROSTER_REL)
    return _PETRI_REPO(tmp, **kw)


def _code_commit(repo: Path, msg: str, *, touch: str | None = None) -> str:
    (repo / "scripts").mkdir(exist_ok=True); (repo / "scripts" / "petri_run.py").write_text(f"# {msg}\n")
    if touch:
        doc = yaml.safe_load((repo / touch).read_text()) or {}
        doc["follow_up_touched"] = msg
        (repo / touch).write_text(yaml.safe_dump(doc, sort_keys=False))
    _git(repo, "add", "-A"); _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _delete_run(db: Path, run_id: str) -> None:
    c = sqlite3.connect(db); c.execute("delete from run where run_id=?", (run_id,)); c.commit(); c.close()


def test_a_code_only_follow_up_advances_the_amended_sha_with_provenance_and_refuses_anything_but_code(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "petri_repo", _repo_with_roster)
    w = P.petri_world(tmp_path); sacct = FakeSacct({("cluster_a", "1"): "RUNNING"})
    assert ST.F22_ROSTER_REL == ROSTER_REL and (w["repo"] / ROSTER_REL).exists()
    x2 = _code_commit(w["repo"], "X'' code only")
    # no prior receipted advance to follow: refused, nothing written
    with pytest.raises(ValueError, match="needs a prior receipted sha advance"):
        P._amend(w, sacct, kind="code-follow-up", commit=x2, tip=x2, note="n")
    P._nothing_written(w)
    e1 = P._amend(w, sacct, note="attempt 7")                       # the per-request fallback receipt at X
    assert e1["kind"] == "petri-role-fallback" and e1["follow_up_of"] is None and len(e1["carried_run_ids"]) == 7
    # the refusals by name: no note; the amended sha itself (not a strict descendant); a new old-job run since the receipt
    with pytest.raises(ValueError, match="needs --note"):
        P._amend(w, sacct, kind="code-follow-up", commit=x2, tip=x2)
    with pytest.raises(ValueError, match="not a strict descendant"):
        P._amend(w, sacct, kind="code-follow-up", commit=w["new_sha"], tip=x2, note="n")
    with pytest.raises(ValueError, match="not a strict descendant"):
        P._amend(w, sacct, kind="code-follow-up", commit=w["old_sha"], tip=x2, note="n")
    rid = "11111111-2222-4333-8444-555555555598"
    P._add_run(w["db"], rid, state="running", cluster_job_id="1", code_sha=w["old_sha"])
    with pytest.raises(ValueError, match="old run set moved since the prior receipt.*new \\['11111111-2222-4333-8444-555555555598'\\]"):
        P._amend(w, sacct, kind="code-follow-up", commit=x2, tip=x2, note="n")
    _delete_run(w["db"], rid)
    assert len(ST.read_amendments(w["ledger"])[w["lw"]]) == 1 and len([x for x in _ledger(w["state"]) if x["event"] == "sha_provenance"]) == 1
    # the follow-up: X -> X'' with provenance; the second receipt carries the first's set unchanged
    note = "review on X': the relaunch record vs the driver's task list; the re-roll stamp before launch"
    e2 = P._amend(w, sacct, kind="code-follow-up", commit=x2, tip=x2, note=note)
    assert e2["kind"] == "code-follow-up" and e2["old_sha"] == w["new_sha"] and e2["old_sha_source"] == "amendment #1's commit" and e2["commit_resolved"] == x2
    assert e2["role_diff"] == [] and e2["changed_instruments"] == {} and e2["row_diff"] == [] and e2["note"] == note
    assert e2["follow_up_of"] == {"utc": e1["utc"], "kind": "petri-role-fallback", "commit": w["new_sha"]}
    assert e2["carried_run_ids"] == e1["carried_run_ids"] and [x["run_id"] for x in e2["old_run_ids"]] == [x["run_id"] for x in e1["old_run_ids"]]
    assert e2["manifest_rel"] == P.MANIFEST_REL and e2["old_manifest_sha256"] == e2["new_manifest_sha256"] == e1["new_manifest_sha256"]
    assert e2["roster_rel"] == ROSTER_REL and len(e2["old_roster_sha256"]) == 64 and e2["old_roster_sha256"] == e2["new_roster_sha256"]
    assert e2["old_contract_sha256"] == e2["new_contract_sha256"] == e1["new_contract_sha256"] and e2["old_export_sha256"] == e2["new_export_sha256"] == e1["new_export_sha256"]
    assert e2["sha_advance"] == {"from": w["new_sha"], "to": x2}
    assert [{k: v for k, v in j.items() if k != "proof_utc"} for j in e2["live_jobs"]] == [{k: v for k, v in j.items() if k != "proof_utc"} for j in e1["live_jobs"]]   # the serving job, re-read now
    assert ST.sha_table(w["ledger"])[w["lw"]] == {"sha": x2, "frozen_sha": w["old_sha"], "n_recorded": 2, "advanced": False, "amended": 2}
    wave = json.load(open(w["state"] / P.WAVE_FILE))
    assert wave["unit_shas"][w["lw"]]["sha"] == x2 and wave["unit_shas"][w["lw"]]["amended"] == 2 and e2["wave_files_refreshed"] == [str(w["state"] / P.WAVE_FILE)]
    ev = [x for x in _ledger(w["state"]) if x["event"] in ("contract_amendment", "sha_provenance")]
    assert [x["event"] for x in ev] == ["contract_amendment", "sha_provenance"] * 2
    assert ev[2]["kind"] == "code-follow-up" and ev[2]["follow_up_of"] == e2["follow_up_of"] and ev[2]["roster_rel"] == ROSTER_REL and ev[2]["carried_run_ids"] == e1["carried_run_ids"]
    assert ev[3]["from_sha"] == w["new_sha"] and ev[3]["sha"] == x2 and ev[3]["kind"] == "amend" and ev[3]["answers"]["kind"] == "code-follow-up"
    assert ST.reconcile_provenance(w["ledger"]) == [] and ST.effective_contract(ST.read_amendments(w["ledger"]), w["unit"]) == e1["contract"]
    # a follow-up whose commit also moves the manifest, or the roster, is refused by name (on a side branch); a follow-up of a follow-up is admitted
    _git(w["repo"], "checkout", "-q", "-b", "bad")
    x3 = _code_commit(w["repo"], "X3 touches the manifest", touch=P.MANIFEST_REL)
    with pytest.raises(ValueError, match="not code-only.*petri_full22.*changed between.*carries no manifest change"):
        P._amend(w, sacct, kind="code-follow-up", commit=x3, tip=x3, note="n")
    _git(w["repo"], "checkout", "-q", "main"); _git(w["repo"], "checkout", "-q", "-b", "bad2")
    x3r = _code_commit(w["repo"], "X3 touches the roster", touch=ROSTER_REL)
    with pytest.raises(ValueError, match="not code-only.*sdfbase_full22_roster.yaml: changed between.*carries no roster change"):
        P._amend(w, sacct, kind="code-follow-up", commit=x3r, tip=x3r, note="n")
    _git(w["repo"], "checkout", "-q", "main")
    assert len(ST.read_amendments(w["ledger"])[w["lw"]]) == 2
    # 32c-e: a descendant that changes a registry SEED outside this unit's projected contract — another model's
    # row, the Petri contract and export untouched — is not code-only either: the committed seed tree is digested and compared
    _git(w["repo"], "checkout", "-q", "-b", "bad3")
    models = w["repo"] / "configs" / "registry" / "models.yaml"
    models.write_text(models.read_text() + "\n# another model's row would go here (a seed change outside the Petri unit's projection)\n")
    (w["repo"] / "scripts" / "petri_run.py").write_text("# X3s seed touch\n")
    _git(w["repo"], "add", "-A"); _git(w["repo"], "commit", "-q", "-m", "X3s touches a seed"); x3s = _git(w["repo"], "rev-parse", "HEAD")
    with pytest.raises(ValueError, match="not code-only.*configs/registry: the committed seed TREE changed.*M configs/registry/models.yaml"):
        P._amend(w, sacct, kind="code-follow-up", commit=x3s, tip=x3s, note="n")
    _git(w["repo"], "checkout", "-q", "main")
    assert len(ST.read_amendments(w["ledger"])[w["lw"]]) == 2
    assert e2["old_seed_tree_sha256"] == e2["new_seed_tree_sha256"] and len(e2["old_seed_tree_sha256"]) == 64          # pinned in the receipt
    d_old, paths = ST.seed_tree_digest(str(w["repo"]), w["new_sha"]); assert d_old == e2["old_seed_tree_sha256"] and "configs/registry/models.yaml" in paths
    assert ST.seed_tree_digest(str(w["repo"]), x3s)[0] != d_old and ST.seed_tree_changes(str(w["repo"]), w["new_sha"], x3s) == ["M configs/registry/models.yaml"]
    assert ST.seed_tree_digest(str(w["repo"]), x3s, exempt=("configs/registry/models.yaml",))[0] == ST.seed_tree_digest(str(w["repo"]), w["new_sha"], exempt=("configs/registry/models.yaml",))[0]
    x4 = _code_commit(w["repo"], "X4 code only again")
    e3 = P._amend(w, sacct, kind="code-follow-up", commit=x4, tip=x4, note="review on X'': one more")
    assert e3["old_sha"] == x2 and e3["follow_up_of"] == {"utc": e2["utc"], "kind": "code-follow-up", "commit": x2} and e3["carried_run_ids"] == e1["carried_run_ids"]
    assert ST.sha_table(w["ledger"])[w["lw"]]["sha"] == x4 and ST.sha_table(w["ledger"])[w["lw"]]["amended"] == 3
    # --recheck re-proves all three; a carried id completing afterwards is the amended drive's work for every receipt
    kw = dict(repo=str(w["repo"]), tip=x4, registry_db=str(w["db"]), sacct_tasks=sacct)
    res = ST.recheck_amendments(w["ledger"], w["lw"], **kw)
    assert [r["verdict"] for r in res] == ["PASS"] * 3 and [r["kind"] for r in res] == ["petri-role-fallback", "code-follow-up", "code-follow-up"], res
    assert res[1]["old_sha"] == w["new_sha"] and res[1]["roster_rel"] == ROSTER_REL and res[1]["role_diff"] == [] and res[2]["old_sha"] == x2
    assert res[1]["old_seed_tree_sha256"] == res[1]["new_seed_tree_sha256"] == e2["new_seed_tree_sha256"]                 # the recheck re-derives the seed tree digest
    _set_state(w["db"], w["runs"][0]["run_id"], "judged")
    res = ST.recheck_amendments(w["ledger"], w["lw"], **kw)
    assert [r["verdict"] for r in res] == ["PASS"] * 3, res
    # a run an old job registers after the follow-up is refused by the recheck of every receipt (32c's rule carries over)
    P._add_run(w["db"], rid, state="running", cluster_job_id="1", code_sha=w["old_sha"])
    res = ST.recheck_amendments(w["ledger"], w["lw"], **kw)
    assert [r["verdict"] for r in res] == ["FAIL"] * 3 and all(any(rid in p and "ABSENT" in p for p in r["problems"]) for r in res), res
    # the kind is Petri-only, and the carried set must be the receipt's: a carried id that FAILED since refuses the follow-up by name
    _delete_run(w["db"], rid)
    _set_state(w["db"], w["runs"][1]["run_id"], "failed")
    x5 = _code_commit(w["repo"], "X5 code only")
    with pytest.raises(ValueError, match="the carried set moved since the prior receipt"):
        P._amend(w, sacct, kind="code-follow-up", commit=x5, tip=x5, note="n")
    assert len(ST.read_amendments(w["ledger"])[w["lw"]]) == 3


def test_the_cli_mints_a_follow_up_after_the_fallback_receipt_and_prints_the_advance(tmp_path, monkeypatch):
    import os
    import subprocess
    from front_door_testlib import LIB, VENV_PY, set_job_state, write_fake_slurm, write_site
    monkeypatch.setattr(P, "petri_repo", _repo_with_roster)
    w = P.petri_world(tmp_path); sacct = FakeSacct({("cluster_a", "1"): "RUNNING"})
    P._amend(w, sacct, note="attempt 7")
    x2 = _code_commit(w["repo"], "X'' code only")
    fake = {lab: tmp_path / f"fake_{lab}" for lab in ("cluster_a", "cluster_b")}
    for d in fake.values():
        (d / "jobs").mkdir(parents=True)
    bin_dir = write_fake_slurm(tmp_path / "bin")
    site = write_site(tmp_path, ssh={lab: f"local:{d}" for lab, d in fake.items()})
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}")
    for k in ("SISTEER_SITE_DIR", "SISTEER_CONFIG"):
        env.pop(k, None)
    set_job_state(fake["cluster_a"], "1", "RUNNING"); (fake["cluster_a"] / "jobs" / "1.name").write_text("f22-petri-m\n")
    base = [VENV_PY or sys.executable, str(LIB / "stage.py"), "amend-contract", "--ledger", w["ledger"], "--logical-work-id", w["lw"], "--registry", str(w["db"]),
            "--config", str(site / "config.env"), "--repo", str(w["repo"]), "--tip", x2, "--kind", "code-follow-up", "--commit", x2, "--reason", "D518: code-only follow-up"]
    r = subprocess.run([*base, "--dry-run"], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 3 and "needs --note" in r.stderr, r.stdout + r.stderr
    r = subprocess.run([*base, "--note", "review on X'", "--dry-run"], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0 and r.stdout.startswith("DRY-RUN: would amend") and "(code-follow-up of the petri-role-fallback receipt at" in r.stdout \
        and f"unit sha {w['new_sha'][:12]} -> {x2[:12]}" in r.stdout and "7 carried unchanged" in r.stdout and "roster configs/sdfbase_full22_roster.yaml unchanged" in r.stdout, r.stdout + r.stderr
    assert len(ST.read_amendments(w["ledger"])[w["lw"]]) == 1
    r = subprocess.run([*base, "--note", "review on X'"], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0 and r.stdout.startswith("amended") and "WAVE unit_shas refreshed: 1" in r.stdout and "note: review on X'" in r.stdout, r.stdout + r.stderr
    assert ST.sha_table(w["ledger"])[w["lw"]]["sha"] == x2 and ST.sha_table(w["ledger"])[w["lw"]]["amended"] == 2
    r = subprocess.run([*base, "--recheck"], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0 and r.stdout.count(": PASS") == 2, r.stdout + r.stderr
