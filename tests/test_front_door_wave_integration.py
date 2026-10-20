"""The front door END TO END on a synthetic site: freeze -> submit -> bind -> admit through the REAL stage scripts —
05_stage_roster.sh --step freeze, 06_eval_waves.sh --view battery, 06 --gate, lib/admission.py, lib/reconcile.sh — against fake Slurm
binaries (`CLUSTER_SSH=local:<dir>` runs every "remote" command on this host; an sbatch --array mints per-task raw job ids the way Slurm
does), a tmp git repo holding a copy of reproduce/ (the clean, committed tree the wave sha pins; the repo mirror and the detached eval
checkout are REALLY created by lib/mirror_repo.sh, and lib/coverage_probe.py runs FROM that checkout), a fake registry and stores laid
down in the PRODUCERS' REAL layouts (jailbreak shard stores with progress.jsonl, the BoN 512 store's bon_responses.jsonl, the climb's
flat per-cell shard<k>.jsonl beside its manifest, tinycap cells/<key>/rows.jsonl + manifest.json, the misalign runner's responses.jsonl),
every run with a real decode-policy id. The Petri unit goes through the REAL native hand-off: the proxy job's real job-start registration
on the cluster (f22_registry_register), the drive's import of those identities (petri_registry_handoff.py), a REAL CellStore per cell
built the way petri_run.py builds it under the exported identity + the grading host's spool (the default recorder routes every row onto
the registered run of its (object, behaviour) and files store_uri + generated), close-leg for the pooled runs, the harvester's fold over
BOTH spools — and 06 --gate petri reads the real `petri_native/…/cells/<run key>` stores those runs record. Labels cluster_a / cluster_b,
hostname TESTHOST, placeholder ids — nothing here names a real host, user, path or session."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import (COMMON_SH, LIB, REPO, REPRO, VENV_PY, add_grading_tables, array_tasks, fold_spool_runs, grading_rows, make_registry, petri_bundle,  # noqa: E402
                                read_spool, registry_emitting_env, run_f22_registration, run_mmlu_full_registration, sampled_controls_store, write_fake_slurm,
                                write_site, set_job_state)

sys.path.insert(0, str(LIB))
sys.path.insert(0, str(REPRO / "tools"))
import stage as ST  # noqa: E402
import replan_compute as rc  # noqa: E402

MODEL = "qwen35_4b_demo"                                                              # the front-door model of reproduce/models.yaml
GIT = ["git", "-c", "user.name=t", "-c", "user.email=t@TESTHOST"]                    # the world's committer (a synthetic identity)
MAX_ARG_STRLEN = 131072                                                               # Linux: one argv string
RUN_RE = re.compile(r"^sdfbase_full22_(?P<key>[a-z0-9_]+?)_sampled_t(?P<trial>[01])(?P<esc>_cap(?:8192|16384))?$")   # sampled/grade_static_sampled.py
GOAL_IDS = [f"goal-{i:03d}" for i in range(259)]                                  # the synthetic pool: the first 100 "jbb" (the BoN subset)
MMLU_IDS = [f"mmlu-{i:04d}" for i in range(1531)]                                 # the synthetic full-width item list the store attests
MMLU_SHA = hashlib.sha256("\n".join(MMLU_IDS).encode()).hexdigest()


class World:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        # a committed copy of reproduce/ is the repo the stages run from: clean tree, HEAD = the wave sha 05 freezes and 06 submits; the
        # wave config states what every full22 config states (private_run, bon.n_max 75, bon.goal_subset jbb)
        self.repo = tmp / "repo"; shutil.copytree(REPRO, self.repo / "reproduce")
        (self.repo / "src" / "sisteer").mkdir(parents=True); shutil.copy(REPO / "src" / "sisteer" / "site.py", self.repo / "src" / "sisteer")   # clusters.py parses with it
        # the front-door model is configured, NOT run (`run_in_this_work: false`: 06 answers --dry-run only, lib/common.sh repro_require_runnable);
        # this world RUNS the stages against fake Slurm, so its COPY of the table — a synthetic table under tmp — flips the flag for the test
        # model's block alone (a text edit inside that block: the table's comments and every other row are untouched)
        tbl = self.repo / "reproduce" / "models.yaml"; text = tbl.read_text()
        i = text.index(f"\n  {MODEL}:\n"); j = text.index("\n  gemma4e4b:\n", i)
        block = text[i:j]; assert block.count("run_in_this_work: false") == 1, "the test model's block has no run_in_this_work: false line"
        tbl.write_text(text[:i] + block.replace("run_in_this_work: false", "run_in_this_work: true") + text[j:])
        (self.repo / "configs").mkdir()
        # the committed Petri scenario lock (hashes only) rides in the checkout: the freeze enumerates the bundle it names
        (self.repo / "configs" / "capeval").mkdir(); shutil.copy(REPO / "configs" / "capeval" / "petri-scenarios.lock.yaml", self.repo / "configs" / "capeval")
        (self.repo / "configs" / f"sdfbase_full22_{MODEL}_battery.yaml").write_text(
            f"private_run: sdfbase_full22_{MODEL}_battery\nexplicit_cells: [none, prompted__safety_refusal, prompted__si_constitution]\nbon:\n  n_max: 75\n  goal_subset: jbb\n")
        # the REAL wave shell body and the registry seeds ride in the checkout: the mmlu job's native registration runs through them.
        # The seeds carry mmlu_1531's ids sha256 = the synthetic item list's, so the store's ATTESTATION is checked end to end
        (self.repo / "scripts/sdfbase/full22").mkdir(parents=True); shutil.copy(COMMON_SH, self.repo / "scripts/sdfbase/full22/common.sh")
        self.seeds = self.repo / "configs" / "registry"; shutil.copytree(REPO / "configs" / "registry", self.seeds)
        isets = (self.seeds / "item_sets.yaml").read_text()
        patched = re.sub(r"(item_set_id: mmlu_1531\n(?:(?!  - item_set_id:).*\n)*?\s*items_sha256: )[0-9a-f]{64}", lambda m: m.group(1) + MMLU_SHA, isets, count=1)
        assert patched != isets, "the seeds' mmlu_1531 items_sha256 line was not found"
        (self.seeds / "item_sets.yaml").write_text(patched)
        g = ["git", "-c", "user.name=t", "-c", "user.email=t@TESTHOST"]
        subprocess.run([*g, "init", "-q", "-b", "main"], cwd=self.repo, check=True)
        subprocess.run([*g, "add", "-A"], cwd=self.repo, check=True)
        subprocess.run([*g, "commit", "-q", "-m", "front door copy"], cwd=self.repo, check=True)
        self.sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.repo, capture_output=True, text=True, check=True).stdout.strip()
        self.fake = {lab: tmp / f"fake_{lab}" for lab in ("cluster_a", "cluster_b")}
        for d in self.fake.values():
            d.mkdir(); (d / "policy").write_text("start_on_submit\n")
        self.bin = write_fake_slurm(tmp / "bin")
        self.site = write_site(tmp, ssh={lab: f"local:{d}" for lab, d in self.fake.items()})
        self.shared = {lab: tmp / f"shared_{lab}" for lab in self.fake}
        self.registry = make_registry(tmp / "registry")
        self.state = tmp / "state" / MODEL
        # the single-phase recipe prices the baseline wave's steered legs (06 requires a COMPLETE compute plan for its one wave as for a
        # steered view): a complete plan in the writer's shape (schema 3, `complete` true, no pending rows)
        self.state.mkdir(parents=True, exist_ok=True)
        self.compute_plan = self.state / "compute_plan.json"
        self.write_compute_plan()
        # the offline goal fixture (grade_static_sampled.py's shape) the freeze mints the shard identities from
        self.goals = tmp / "goals.jsonl"
        self.goals.write_text("".join(json.dumps({"goal_id": g, "dataset": "jbb" if i < 100 else "harmbench", "text": "t", "category": "c", "revision": "r"}) + "\n" for i, g in enumerate(GOAL_IDS)))
        # the frozen Petri scenario bundle's SYNTHETIC materialization (the freeze enumerates `<behavior>/scenarios/seeds` stems from it)
        self.petri_bundle = petri_bundle(tmp)

    def write_compute_plan(self):
        # the plan in the WRITER's shape (replan_compute.plan_document: every key the real file carries, `complete` = no pending rows),
        # never a hand literal that could drift from what tools/replan_compute.py writes and 06's repro_require_compute_plan reads
        doc = rc.plan_document(model=MODEL, memprobe=[], fold=[], probe=None, probe_status="present", probe_rec=None, fit_width=1, fits_config=None,
                               step_threshold_s=0.0, replan=False, arms={}, plan=[], pending=[], excluded={}, total=0.0, decision=["synthetic: a complete plan"])
        self.compute_plan.write_text(json.dumps(doc))

    def env(self, **extra):
        e = dict(os.environ, PATH=f"{self.bin}:{os.environ['PATH']}", SISTEER_STATE_DIR=str(self.tmp / "state"), SISTEER_NO_WATCHER="1",
                 SISTEER_REGISTRY_ROOT=str(self.tmp / "registry"), SISTEER_SRC_DIR=str(REPO / "src"), SISTEER_REGISTRY_SEEDS=str(self.seeds),
                 SISTEER_GOALS_JSONL=str(self.goals), SISTEER_PETRI_BUNDLE_DIR=str(self.petri_bundle))
        for k in ("SISTEER_SITE_DIR", "SISTEER_CONFIG", "REPRO_DRY_RUN"):
            e.pop(k, None)
        e.update(extra)
        return e

    def stage(self, script: str, *args: str, **extra):
        return subprocess.run(["bash", str(self.repo / "reproduce" / script), *args, "--config", str(self.site / "config.env")],
                              capture_output=True, text=True, env=self.env(**extra), cwd=str(self.repo), timeout=900)

    def attempts(self):
        p = self.state / "attempts.jsonl"
        return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

    def inventory(self):
        return json.load(open(self.state / "gate" / "expected_units.json"))


def _jsonl(p: Path, rows, append=False) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a" if append else "w") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))


def _lay_down(w: World, shared: Path, u: dict, e: str, ev: dict, task, rid: str, lab: str):
    """Write the PRODUCER's artifact for one (unit, eval, task) the way the producer writes it, cell `none`, and return the store the producer's
    CLOSER files as the run's store_uri (the gate reads every unit at its RECORDED store, never at the frozen coverage.path) — the
    multi-turn targets under the GROUPED reproduce root (`<shared>/exp/reproduce/<model>/waves/mt-<view>-<label>/transcripts_results/…`, where
    the real smoke's jobs wrote and the frozen `root_rel` — the paper's ungrouped root — does not exist), the statics / padding / BoN 512 stores
    at the frozen root (identical to the closer's), the agentic cell DIRECTORY (`<root>/agentic/<harness cell>`), the sampled controls as a
    gens-spool store with no rows.jsonl, the climb's flat files under the producer's root with NO store of their own (the climb's store_uri is
    the 512 store; its root travels as `extra.climb_root`, which _register_every_submitted_unit files when asked)."""
    cov = ev["coverage"]; lay = cov["layout"]
    if lay == "jailbreak_cellstore":
        root = shared / cov["path"]["root_rel"].replace("<cell>", "none")
        if u["family"] == "multiturn":
            root = shared / "exp" / "reproduce" / MODEL / "waves" / f"mt-battery-{lab}" / "transcripts_results" / Path(cov["path"]["root_rel"].replace("<cell>", "none")).relative_to("exp/sdfbase_full22_native/transcripts_results")
        for s, ids in cov["expected"].items():
            _jsonl(root / s / "progress.jsonl", [{"goal_id": g, "outcome": "success", "n_target_forwards": 2} for g in ids])
            _jsonl(root / s / "responses.jsonl", [{"goal_id": g, "sample_idx": i, "response": "…"} for g in ids for i in range(2)])
        return str(root)
    if lay == "bon_store512":
        root = shared / cov["path"]["root_rel"].replace("<cell>", "none")
        _jsonl(root / "bon_responses.jsonl", [{"goal_id": g, "sample_idx": i, "trial": 0, "response": "…"} for g in cov["expected"]["shard00"] for i in range(cov["n_max"])])
        return str(root)
    if lay == "bon_climb_flat":
        root = shared / cov["path"]["root_rel"]
        man = root / "plan" / "climb_manifest.jsonl"
        if not man.exists():                                        # array task 0 writes the plan; every task reads it
            _jsonl(man, [{"cell_id": "none", "goal_id": g, "sample_idx": s, "trial": 0, "class": "cap_coherent"} for g in GOAL_IDS[:20] for s in (0, 1)])
        rows = [json.loads(l) for l in man.read_text().splitlines() if l.strip()]
        rows.sort(key=lambda r: (r["goal_id"], r["sample_idx"]))
        mine = [r for i, r in enumerate(rows) if i % cov["nshards"] == task]
        _jsonl(root / "out" / "none" / f"shard{task}.jsonl", [{"goal_id": r["goal_id"], "sample_idx": r["sample_idx"], "trial": 0, "response": "…", "cell_id": "none"} for r in mine])
        return str(shared / "exp" / "sdfbase_full22_native" / "transcripts_results" / f"sdfbase_full22_{MODEL}_battery" / "none")     # the climb's store_uri IS the 512 store
    if lay == "capability_cellstore" and e in ("xstest", "or_bench_hard"):
        # the sampled controls: the judge's rows.jsonl lives elsewhere; the item artifact is the generation spool the producer stamps complete
        store = shared / "exp" / "sdfbase_full22_native" / "sampled" / "controls_cap2048" / f"{MODEL}_battery_t{u['unit'][-1]}" / "cells" / f"none__controls__t{u['unit'][-1]}__a0__{rid[-8:]}"
        n = 450 if e == "xstest" else cov["n_items"]              # the producer generates xstest's 450 superset; the frozen item set scores the 250 safe half
        key = "xstest" if e == "xstest" else "or_bench_hard_1k"
        sampled_controls_store(store, {key: {f"{key}-{i:04d}": f"t{i}" for i in range(n)}}, at_cap={key: [f"{key}-0001"]})
        return str(store)
    if lay == "capability_cellstore":
        ids = [f"{e}-item-{i:04d}" for i in range(cov["n_items"])]
        store = shared / "stores" / rid / "cells" / f"none__{e}"
        _jsonl(store / "rows.jsonl", [{"task": cov["task"], "item_id": i, "metric": "acc", "value": 1.0} for i in ids]
               + [{"task": cov["task"], "item_id": "__aggregate__", "metric": "acc_aggregate", "value": 0.9}])
        (store / "manifest.json").write_text(json.dumps({"row_constants": {"slurm_job_id": "x"}}))
        return str(store)
    if lay == "misalign_responses":
        root = shared / cov["path"]["root_rel"]
        _jsonl(root / "none" / "responses.jsonl", [{"item_id": i, "sample_idx": k, "behavior": cov["behavior"], "response": "…"} for i in cov["expected"]["shard00"] for k in range(cov["n_samples"])], append=True)
        (root / "none" / "meta.json").write_text(json.dumps({"cell_id": "none", "n_samples": cov["n_samples"], "finished": 1760000000.0}))
        return str(root / "none")                                   # the closer files the harness CELL dir (registry_adapter.harness_cell_id: none -> none)
    return None                                                     # the Petri unit: _produce_petri_natively (never laid down here)


PENDING_PAD = 36_000      # bytes of plan.extra per registered run: a 4-run static pending file > 131,072 B, a climb task file > 36 KB (x4 > 128 KiB)


MMLU_EMIT = r'''
import json, os, sys, time
from sisteer.capeval.store import CellStore, ResultRow, cell_key
from sisteer.eval.cell_controls import store_key_token
out_root, job, model_id, cells, ids = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(","), json.loads(open(sys.argv[5]).read())
spec = {"item_set": "full", "n": len(ids), "seed": 0, "of": len(ids)}
for cell in cells:
    token = store_key_token(cell); ihash = "0" * 16; chash = "1" * 16
    key = cell_key(token, "controls", 0, 0, ihash, chash)
    tpl = {"sweep": "two_si_controls_served", "run_id": f"served-controls-{token}", "model_id": model_id, "model_revision": "f" * 40, "arm_name": token,
           "kind": "none" if cell == "none" else "cell", "coeff": 1.0, "intervention_hash": ihash, "engine": "vllm", "trial_id": 0, "seed": 0, "attempt": 0,
           "harness_config_hash": chash, "env_digest": "e" * 16, "slurm_job_id": job}
    store = CellStore(out_root, key, row_constants=tpl)          # the DEFAULT recorder: the job's pending identity from the environment
    store.write_cell_manifest({**tpl, "row_constants": dict(tpl), "cell_id": cell, "resolved_config": {"mmlu": {"item_set": spec}},
                               "mmlu_item_set": {**spec, "item_ids": ids}})
    for i, iid in enumerate(ids):
        store.append_row(ResultRow(**tpl, tier=1, task="s2parity/mmlu_letter", task_version="v1", domain="capability_parity", item_id=iid,
                                   position_order=i, metric="acc", value=1.0, extra_json=None, created_at=time.time(), wall_clock_s=None))
    store.append_row(ResultRow(**tpl, tier=1, task="s2parity/mmlu_letter", task_version="v1", domain="capability_parity", item_id="__aggregate__",
                               position_order=None, metric="acc_aggregate", value=1.0, extra_json=None, created_at=time.time(), wall_clock_s=None))
    print("STORE", cell, store.cell_dir)
'''


def _produce_mmlu_natively(w: World, u: dict, a: dict, runs: list, policies: list) -> None:
    """The mmlu unit's REAL registration path: the sbatch's own registration block + common.sh f22_registry_register (the real
    emit_cli register over the checkout's seeds -> the job's pending file + spool run records); then the PRODUCED store in the
    sbatch's layout ($WAVE_ROOT/mmlu_full/<TAG>_t0/cells/<key>) written through the real tinycap CellStore whose default recorder
    routes every row onto the registered run from the pending identity (item rows, the aggregate metric, the store uri); then the
    job's terminal state (mmlu_full_registry_end). The registry rows are FOLDED from the emitted spool records (fold_spool_runs) —
    never a hand-written run row; the fold stands in for the harvester's reducer (rule-scored: generated -> ingested) and its
    decode derivation (the frozen ref, exact)."""
    lab = a["cluster"]; shared = w.shared[lab]; job = a["job_id"]
    ev = u["evals"]["mmlu"]
    r = run_mmlu_full_registration(w.repo, shared, key=MODEL, view="battery", cfg_rel=f"configs/sdfbase_full22_{MODEL}_battery.yaml", job_id=job, sha=w.sha, evals="mmlu")
    assert r.returncode == 0 and "CHECK_RC=0" in r.stdout, r.stdout + r.stderr
    pend = shared / "registry" / "pending" / f"{job}.json"
    assert pend.is_file() and all(x["plan"]["item_set_id"] == "mmlu_1531" for x in json.loads(pend.read_text())["runs"])
    out_root = shared / "exp/sdfbase_full22_native/mmlu_full" / f"{MODEL}_battery_t0"
    ids_f = w.tmp / "mmlu_ids.json"; ids_f.write_text(json.dumps(MMLU_IDS))
    env = registry_emitting_env(PYTHONPATH=str(REPO / "src"), SISTEER_REGISTRY_RUN_IDENTITY=str(pend), SISTEER_REGISTRY_SPOOL=str(shared / "registry/records"),
                                SISTEER_REGISTRY_SUBSTRATE="slurm", SISTEER_REGISTRY_CLUSTER_LABEL=lab, SLURM_JOB_ID=str(job))
    cells = ["none", "prompted__safety_refusal", "prompted__si_constitution"]
    # cwd = the checkout the job runs in (the mirrored tree at the wave sha): the store recorder stamps the PRODUCING checkout's HEAD as the
    # run's code_sha (emit.git_sha), which the gate holds to the frozen sha
    p = subprocess.run([VENV_PY, "-c", MMLU_EMIT, str(out_root), str(job), u["model_id"], ",".join(cells), str(ids_f)], capture_output=True, text=True, env=env,
                       cwd=str(w.repo), timeout=600)
    assert p.returncode == 0 and p.stdout.count("STORE ") == 3, p.stdout + p.stderr
    end = run_mmlu_full_registration(w.repo, shared, key=MODEL, view="battery", cfg_rel=f"configs/sdfbase_full22_{MODEL}_battery.yaml", job_id=job, sha=w.sha, evals="mmlu",
                                     after="TOTAL_FAIL=0; mmlu_full_registry_end 0 0")          # the pending file exists: register keeps it, the end files `generated`
    assert end.returncode == 0, end.stdout + end.stderr
    pid = f"pid-mmlu-{job}"; policies.append((pid, ev["decode_policy_ref"]))
    folded = fold_spool_runs(shared / "registry/records", decode_policy_id=pid)
    mine = [x for x in folded if x.get("cluster_job_id") == str(job) or x.get("eval_id") == "mmlu"]
    assert len(mine) == 3 and all(x["lifecycle_state"] == "ingested" and x["store_uri"] and x["instrument_id"] == ev["instrument_id"] for x in mine), mine
    for x in mine:
        runs.append({"run_id": x["run_id"], "lifecycle_state": x["lifecycle_state"], "model_id": x["model_id"], "code_sha": x["code_sha"], "instrument_id": x["instrument_id"],
                     "decode_policy_id": pid, "decode_policy_match": "exact", "eval_id": x["eval_id"], "object_id": x["object_id"], "cluster_job_id": str(job), "store_uri": x["store_uri"]})


PETRI_STORE = r'''
import json, os, sys, time
from sisteer.capeval.store import CellStore, ResultRow
out_root, run_key, cell_id, kind, ids = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], json.loads(sys.argv[5])
# scripts/petri_run.py _row_constants + cmd_run: CellStore(out_root, cell.petri_run_key, row_constants=…), the manifest with cell_id
audit = os.path.basename(out_root)
const = {"sweep": audit, "run_id": f"{audit}.{cell_id}.{run_key[:8]}", "model_id": "Qwen/Qwen3.5-4B", "model_revision": "5" * 40,
         "arm_name": cell_id, "kind": kind, "coeff": None, "intervention_hash": "2ff9cdff091aef6f", "engine": "proxy_vllm+lora+interleaved",
         "trial_id": 0, "seed": 0, "attempt": 0, "harness_config_hash": "1602b0fd44db6681", "env_digest": "ba9205dffa5ce785",
         "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
store = CellStore(out_root, run_key, row_constants=const)
resolved = {**const, "row_constants": dict(const), "cell_id": cell_id, "petri_run_key": run_key, "virtual_model_name": f"petri__{cell_id}",
            "scenario_bundle_hash": "8812ba083bb5473c", "resolved_at": time.time()}
store.validate_resume(resolved); store.write_cell_manifest(resolved)
n = 0
for b, sids in ids.items():
    for sid in sids:
        for metric in (b, "eval_awareness", "scenario_realism"):
            store.append_row(ResultRow(**const, tier=-1, task=f"petri/{b}", task_version="8812ba083bb5473c", domain="petri_behavior", item_id=sid,
                                       position_order=None, metric=metric, value=4.0, extra_json=json.dumps({"scenario_id": sid, "epoch": 1, "dimension": metric}),
                                       created_at=time.time(), wall_clock_s=None)); n += 1
(store.cell_dir / "petri_incomplete.json").write_text(json.dumps({"version": 1, "cell_id": cell_id, "petri_run_key": run_key, "unresolved": []}))
store.write_results({"cell_id": cell_id, "petri_run_key": run_key, "rows_emitted": n, "unscored": 0, "completed_at": time.time(),
                     "behaviors": {b: {"n_items": len(s), "n_scenarios": len(s), "incomplete": False} for b, s in ids.items()}})
print("STORE", store.cell_dir)
'''
HANDOFF = REPO / "scripts" / "sdfbase" / "full22" / "petri_registry_handoff.py"


def _produce_petri_natively(w: World, u: dict, a: dict, runs: list, policies: list) -> None:
    """The Petri unit's REAL native hand-off. (1) The proxy job's job-start registration on the CLUSTER, for real (common.sh
    f22_registry_register -> emit_cli register --from-config over the checkout's seeds and the baseline wave config's explicit_cells): one run per
    (eval, object), the pending file under the cluster's shared root, `running` per run — the file 06 --gate binds from. (2) The DRIVE's
    import of those identities (petri_registry_handoff.py import: the wave cells re-spelled to the Petri harness ids the stores name) into
    its state dir, with the grading host's spool. (3) Per served cell a REAL CellStore built the way petri_run.py builds it, under the
    exported identity + spool: the default recorder routes every row onto the registered run of ITS (object, behaviour) and files store_uri
    + generated at results.json. (4) close-leg per cell: the pooled run's store_uri + generated. (5) finish. (6) The harvester's fold over
    BOTH spools -> the registry rows (a policy id per eval standing in for the decode derivation, as the mmlu path does)."""
    lab = a["cluster"]; shared = w.shared[lab]; job = a["job_id"]; tag = f"{MODEL}_battery"
    evals = ",".join(u["evals"])
    r = run_f22_registration(w.repo, shared, key=MODEL, view="battery", cfg_rel=f"configs/sdfbase_full22_{MODEL}_battery.yaml", job_id=job, job_name=f"f22-petri-{tag}",
                             sha=w.sha, evals=evals, cluster_label=lab)
    assert r.returncode == 0 and f"REGISTERED pending={shared}/registry/pending/{job}.json" in r.stdout, r.stdout + r.stderr
    pend = shared / "registry" / "pending" / f"{job}.json"
    src_runs = json.loads(pend.read_text())["runs"]
    assert len(src_runs) == 21 and {x["route"]["object"] for x in src_runs} == {"none", "prompted__safety_refusal", "prompted__si_constitution"}
    # the drive on the grading host: its state dir, the identity import, the spool it writes (the grading host registry's records dir)
    home = w.tmp / "grading_home"; state = home / ".local" / "state" / "sdfbase_full22_native" / "petri" / tag
    grading_spool = w.tmp / "grading_registry" / "records"; grading_spool.mkdir(parents=True, exist_ok=True)
    pend_copy = state / "registry" / "pending" / f"{job}.cluster.json"; pend_copy.parent.mkdir(parents=True, exist_ok=True); shutil.copy(pend, pend_copy)
    env = registry_emitting_env(PYTHONPATH=str(REPO / "src"))
    imp = subprocess.run([VENV_PY, str(HANDOFF), "--spool", str(grading_spool), "import", "--pending", str(pend_copy), "--job", str(job),
                          "--out-dir", str(state / "registry" / "pending"), "--sh"], capture_output=True, text=True, env=env, timeout=300, cwd=str(REPO))
    assert imp.returncode == 0, imp.stdout + imp.stderr
    exports = dict(ln[len("export "):].split("=", 1) for ln in imp.stdout.splitlines() if ln.startswith("export "))
    ident = exports["SISTEER_REGISTRY_RUN_IDENTITY"].strip("'"); assert ident == str(state / "registry" / "pending" / f"{job}.json")
    ident_doc = json.loads(open(ident).read())
    assert ident_doc["object_translation"] == {"none": "none", "prompted__safety_refusal": "prompted__safety", "prompted__si_constitution": "prompted__constitution"}
    # the audit's per-cell stores on the grading host — the REAL layout petri_run.py writes: <home>/.local/state/petri_native/petri_full22_<tag>/cells/<run key>
    out_root = home / ".local" / "state" / "petri_native" / f"petri_full22_{tag}"
    ids = {b: [x.split("/", 1)[1] for x in u["evals"]["petri_pooled"]["coverage"]["expected"]["shard00"] if x.startswith(b + "/")] for b in ST.PETRI_BEHAVIORS}
    cells = {}
    for i, (cell_id, kind) in enumerate((("none", "none"), ("prompted__safety", "prompted"), ("prompted__constitution", "prompted"))):
        run_key = f"petricell-{i}{cell_id[:4]}"
        senv = dict(env, SISTEER_REGISTRY_RUN_IDENTITY=ident, SISTEER_REGISTRY_SPOOL=str(grading_spool), SLURM_JOB_ID=str(job))   # the drive's exports + the proxy job id
        p = subprocess.run([VENV_PY, "-c", PETRI_STORE, str(out_root), run_key, cell_id, kind, json.dumps(ids)], capture_output=True, text=True, env=senv, timeout=600, cwd=str(REPO))
        assert p.returncode == 0 and "STORE " in p.stdout, p.stdout + p.stderr
        cells[cell_id] = out_root / "cells" / run_key
        c = subprocess.run([VENV_PY, str(HANDOFF), "--spool", str(grading_spool), "close-leg", "--identity", ident, "--cell-dir", str(cells[cell_id])],
                           capture_output=True, text=True, env=env, timeout=300, cwd=str(REPO))
        # the close-leg summary gained `recounted` / `refused` counts (petri hand-off); the counts this test pins are unchanged
        assert c.returncode == 0 and "1 run(s) filed, 6 already generated" in c.stdout and "of 7 registered" in c.stdout, c.stdout + c.stderr
    fin = subprocess.run([VENV_PY, str(HANDOFF), "--spool", str(grading_spool), "finish", "--identity", ident], capture_output=True, text=True, env=env, timeout=300, cwd=str(REPO))
    assert fin.returncode == 0 and "pushed final" in fin.stdout, fin.stdout + fin.stderr
    # the harvester's one view over BOTH producers' spools: 21 runs, every one generated -> ingested with the REAL grading-host store URI
    by_obj = {"none": "none", "prompted__safety_refusal": "prompted__safety", "prompted__si_constitution": "prompted__constitution"}
    pid = f"pid-petri-{job}"
    folded = {x["run_id"]: x for x in fold_spool_runs(shared / "registry/records", grading_spool, decode_policy_id=pid)}
    for x in src_runs:
        row = folded[x["run_id"]]; e = x["route"]["eval_id"]; ev = u["evals"][e]
        assert row["lifecycle_state"] == "ingested" and row["store_uri"] == str(cells[by_obj[x["route"]["object"]]]), (e, x["route"]["object"], row)
        assert row["store_uri"].endswith(f"/.local/state/petri_native/petri_full22_{tag}/cells/petricell-" + row["store_uri"].rsplit("petricell-", 1)[1])
        assert row["instrument_id"] == ev["instrument_id"], (e, row["instrument_id"], ev["instrument_id"])
        if ev["decode_policy_ref"]:
            policies.append((pid, ev["decode_policy_ref"]))
        runs.append({"run_id": row["run_id"], "lifecycle_state": row["lifecycle_state"], "model_id": row["model_id"], "code_sha": u["code_sha"], "instrument_id": row["instrument_id"],
                     "decode_policy_id": pid if ev["decode_policy_ref"] else None, "decode_policy_match": "exact" if ev["decode_policy_ref"] else "unrecorded",
                     "eval_id": e, "object_id": row["object_id"], "cluster_job_id": str(job), "store_uri": row["store_uri"]})
    # WHO filed what: the six behaviour runs per cell were opened and closed by the STORE RECORDER (a `run` payload carrying the store key), the
    # pooled run per cell by the drive's close-leg (a run_patch + generated from petri_registry_handoff, never a store-recorder run payload)
    recs = read_spool(grading_spool)
    opened = {p_["run_id"] for rt, p_ in recs if rt == "run" and (p_.get("extra") or {}).get("store_key")}
    pooled = {x["run_id"] for x in src_runs if x["route"]["eval_id"] == "petri_pooled"}
    assert len(opened) == 18 and not (opened & pooled)
    assert all(any(rt == "run_patch" and p_["run_id"] == rid and p_["fields"].get("store_uri") for rt, p_ in recs) for rid in pooled)
    w.petri_cells = cells


def _register_every_submitted_unit(w: World, *, climb_root: bool = False):
    """What the jobs and the harvester would do: for every submitted registering unit, per array task, one run per eval (cell `none`)
    registered under the task's OWN raw job id (its pending file — padded past MAX_ARG_STRLEN, the smoke's 149,440-byte shape), the
    producer's artifact laid down in its real layout, a registry row per run complete under the FROZEN bindings with a NON-NULL decode policy
    id resolving to the frozen ref (decode_policy_match exact) and the store the producer's closer files. `climb_root` = the closer has filed
    `extra_json.climb_root` on the climb's runs (gate-fixes: the record 06 --gate reads the climb at). The mmlu unit goes through the REAL
    registration + emission path (_produce_mmlu_natively). The rows are kept on `w.reg_runs` / `w.reg_policies` for a later re-make."""
    inv = {u["logical_work_id"]: u for u in w.inventory()["units"]}
    runs, policies = [], []
    for a in w.attempts():
        if a.get("event") != "attempt" or a["logical_work_id"] not in inv:
            continue
        u = inv[a["logical_work_id"]]; lab = a["cluster"]; shared = w.shared[lab]
        if u["family"] == "mmlu":
            _produce_mmlu_natively(w, u, a, runs, policies); continue
        if u["family"] == "petri":
            _produce_petri_natively(w, u, a, runs, policies); continue
        tasks = array_tasks(w.fake[lab], a["job_id"]) if u.get("array") else [(a["job_id"], None)]
        for raw, task in tasks:
            pend_runs = []
            for e, ev in u["evals"].items():
                rid = f"run-{ST.safe_id(u['logical_work_id'])}-{e}" + ("" if task is None else f"-t{task}")
                store_uri = _lay_down(w, shared, u, e, ev, task, rid, lab)
                pend_runs.append({"run_id": rid, "route": {"object": "none", "eval_id": e}, "plan": {"eval_id": e, "model_id": u["model_id"], "extra": {"pad": "x" * PENDING_PAD}}})
                pid = f"pid-{e}" if ev["decode_policy_ref"] else None
                if pid:
                    policies.append((pid, ev["decode_policy_ref"]))
                row = {"run_id": rid, "lifecycle_state": "ingested", "model_id": u["model_id"], "code_sha": u["code_sha"],
                       "instrument_id": ev["instrument_id"] or "unresolved-at-freeze", "decode_policy_id": pid,
                       "decode_policy_match": "exact" if pid else "unrecorded", "eval_id": e, "object_id": "none",
                       "cluster_job_id": raw, "store_uri": store_uri}
                if ev["coverage"]["layout"] == "bon_climb_flat":
                    row["climb_root_record"] = str(shared / ev["coverage"]["path"]["root_rel"])           # what the closer files as extra_json.climb_root
                    if climb_root:
                        row["extra"] = {"climb_root": row["climb_root_record"]}
                runs.append(row)
            if u["registration"] == "native":
                pend = shared / "registry" / "pending"; pend.mkdir(parents=True, exist_ok=True)
                (pend / f"{raw}.json").write_text(json.dumps({"run_id": pend_runs[0]["run_id"], "job_id": raw, "runs": pend_runs}, indent=1))
    make_registry(w.tmp / "registry", runs, policies)
    w.reg_runs, w.reg_policies = runs, policies
    return runs


def test_freeze_submit_bind_admit_end_to_end(tmp_path):
    w = World(tmp_path)
    # ---- the baseline wave (battery: the single-phase recipe's one wave) is PRICED like a steered view: without a complete compute plan it refuses first
    w.compute_plan.unlink()
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--view", "battery", "--family", "mmlu")
    assert r.returncode != 0 and "no compute plan" in r.stderr and "single-phase recipe: view battery is the ONE wave" in r.stdout and "admission.py check" not in r.stdout + r.stderr, r.stdout + r.stderr
    w.write_compute_plan()
    # ---- ... and before the freeze it is refused: it IS the gate's baseline and must match it
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--view", "battery", "--family", "mmlu")
    assert r.returncode != 0 and "no frozen inventory" in r.stderr, r.stdout + r.stderr
    # ---- 05 --step freeze: the inventory = the unit plan, ids minted by the registry's planner, at the committed sha
    r = w.stage("05_stage_roster.sh", "--model", MODEL, "--step", "freeze")
    assert r.returncode == 0, r.stdout + r.stderr
    inv = w.inventory()
    assert inv["schema"] == 2 and inv["code_sha"] == w.sha and inv["n_units"] == 36   # the single-phase wave: the gsm8k rows cover every cell (10, not the prompted-only 3) — 36 registering units
    for u in inv["units"]:
        if u["private_run"]:
            assert RUN_RE.match(u["private_run"]), u["private_run"]                        # the sampled grader admits it
    # ---- a wave at ANOTHER sha than the frozen one: the tree is PROVENANCE, not the inventory's identity — 06 notes the
    #      difference (each unit's contract is verified at that tree by the funnel) and the mirror step refuses a sha that is not a
    #      commit of the repo, so nothing is submitted; the old "frozen code_sha != wave sha" refusal is gone by ruling
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--view", "battery", "--family", "mmlu", "--wave-sha", "f" * 40)
    assert r.returncode != 0 and "per-unit sha provenance applies (D200)" in r.stderr and "frozen code_sha" not in r.stderr \
        and "is not a commit of" in r.stderr and not (w.state / "submissions.jsonl").exists(), r.stdout + r.stderr
    # ---- --twin with multiturn is REFUSED (never printed past and double-submitted)
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--view", "battery", "--family", "all", "--twin")
    assert r.returncode != 0 and "not implemented for multi-turn" in r.stderr and not (w.state / "submissions.jsonl").exists()
    # ---- 06 with ONE failing submission (the mmlu job): the wave FAILS, the partial list is recorded, no WAVE.battery.json
    for d in w.fake.values():
        (d / "fail_names").write_text(f"mmluF-{MODEL}-battery\n")
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--view", "battery", "--family", "all")
    assert r.returncode != 0 and "FAILED — the wave is INCOMPLETE" in r.stderr, r.stdout[-2000:] + r.stderr[-2000:]
    assert not (w.state / "WAVE.battery.json").exists()
    partial = json.load(open(w.state / "WAVE.battery.partial.json"))
    assert partial["failed_unit"] == "mmlu.mmlu" and len(partial["placed_before_failure"]) == 29      # 25 registering + 2 x 2 pool members (commit 31) before the mmlu failure
    # the pinned sha was mirrored + checked out on BOTH clusters before the first submission
    for lab in w.fake:
        co = w.shared[lab] / "sisteer-eval-repo" / w.sha
        assert (co / "reproduce" / "lib" / "stage.py").is_file(), lab
        assert subprocess.run(["git", "rev-parse", "HEAD"], cwd=co, capture_output=True, text=True).stdout.strip() == w.sha
    subs1 = [ln for ln in r.stdout.splitlines() if ln.startswith("submitted ")]
    assert len(subs1) == 29 and not any(ln.startswith("skipped ") for ln in r.stdout.splitlines())
    # multi-turn: each cluster got ITS pool + a disjoint half of the target units
    for lab in ("cluster_a", "cluster_b"):
        mine = [ln.split()[3] for ln in subs1 if ln.split()[1] == lab]
        assert sum(1 for u in mine if u.startswith("multiturn.attacker")) == 2 and sum(1 for u in mine if u.startswith("multiturn.") and "attacker" not in u) == 6   # the pool is sized one member per three targets (commit 31)
    # the statics went out in the SAMPLED lane's form with their evals declared explicitly
    argv_a = " ".join(p.read_text() for p in (w.fake["cluster_a"] / "jobs").glob("*.argv"))
    env_a = " ".join(p.read_text() for p in (w.fake["cluster_a"] / "jobs").glob("*.env"))
    assert f"sampled/target_sampled.sbatch {MODEL} static_base_t0 {w.sha} 9  battery" in argv_a and f"target_static.sbatch {MODEL} battery pasttense" not in argv_a   # `$*`: the empty cells positional is two spaces
    assert "F22_REGISTRY_EVALS=pasttense,codeattack,renellm,grok420" in env_a and "F22_REGISTRY_EVALS=xstest,or_bench_hard" in env_a
    assert "SISTEER_CLUSTER_QOS=qos_cluster_a" in env_a                                          # the QoS rides every job env
    # ---- the re-run is IDEMPOTENT at the funnel: every placed unit is skipped, only the missing ones go out — ruling 11
    for d in w.fake.values():
        (d / "fail_names").unlink()
    # the failed submission left its first-submit RESERVATION standing (a non-zero sbatch over ssh cannot be told from a transport failure after
    # the controller accepted — fail closed, 31e-fix-5), and a re-run inside the 120 s accounting-visibility floor is 'not yet', not 'never'
    # (31e-fix-3). The operator re-runs after the floor; the test stands for that by ageing every standing token, as the resubmit-proof test does.
    for tp in (w.state / "bindings").glob("*.transition.json"):
        tp.write_text(json.dumps({**json.loads(tp.read_text()), "utc": "2026-01-01T00:00:00Z"}))
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--view", "battery", "--family", "all")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    lines = r.stdout.splitlines()
    assert sum(1 for ln in lines if ln.startswith("skipped ")) == 29 and sum(1 for ln in lines if ln.startswith("submitted ")) == 11   # the 29 placed before the failure + the 11 left of the 40
    assert (w.state / "WAVE.battery.json").exists() and json.load(open(w.state / "WAVE.battery.json"))["n_units"] == 40   # 36 registering units + the 4 attacker-pool members (2 per cluster)
    attempts = [a for a in w.attempts() if a["event"] == "attempt"]
    assert len(attempts) == 40 and len({a["logical_work_id"] for a in attempts}) == 40          # ONE identity per unit, ever (36 registering + 2 x 2 pool members)
    for u in w.inventory()["units"]:
        assert ST.binding_state(str(w.state / "submissions.jsonl"), u["logical_work_id"])["state"] == "submitted"
    # the frozen coverage is WORK IDENTITY from the producers' recipes: the fixture pool sliced goals[s:9]; the mmlu unit
    # binds through the registry; the climb is an array whose shards are its tasks
    units = {u["logical_work_id"].split("/", 2)[2]: u for u in w.inventory()["units"]}
    assert units["static/t0"]["evals"]["pasttense"]["coverage"]["expected"]["shard04"] == GOAL_IDS[4::9]
    assert units["mmlu/mmlu"]["registration"] == "native" and units["mmlu/mmlu"]["evals"]["mmlu"]["coverage"]["items_sha256"] == MMLU_SHA
    assert units["bon/climb"]["array"] == "0-3"
    mmlu_env = " ".join(p.read_text() for p in (w.fake["cluster_a"] / "jobs").glob("*.env") if "mmluF" in (p.with_suffix(".name")).read_text())
    assert mmlu_env and "F22_REGISTRY_EVALS=mmlu" in mmlu_env                                   # The front door's mmlu job registers natively
    # ---- the freeze wrote the coverage contract into the header; this world plays an inventory frozen BEFORE the field existed:
    #      the field is stripped once, and the first --gate admits the current contract ONCE (the sidecar; the fact rides every verdict)
    inv0 = w.inventory()
    assert inv0["coverage_contract"]["source"] == "frozen" and inv0["coverage_contract"]["sha256"] and set(inv0["coverage_contract"]["contract"]["unit_set"]) == {u["logical_work_id"] for u in inv0["units"]}
    inv0.pop("coverage_contract"); ST.write_atomic(str(w.state / "gate" / "expected_units.json"), inv0)
    # the probe travels to each cluster as ONE file by content: a HALF-WRITTEN file pre-planted at the shipped name on cluster_a is
    # re-shipped by digest, never trusted for being non-empty
    probe_bytes = (w.repo / "reproduce" / "lib" / "coverage_probe.py").read_bytes(); probe_sha = hashlib.sha256(probe_bytes).hexdigest()
    shipped = {lab: w.shared[lab] / "scripts" / "registry" / f"coverage_probe.{probe_sha[:16]}.py" for lab in w.fake}
    shipped["cluster_a"].parent.mkdir(parents=True, exist_ok=True); shipped["cluster_a"].write_bytes(probe_bytes[: len(probe_bytes) // 2])
    # ---- --gate before anything registered: units stay `submitted` (no pending file yet), the gate refuses by name — exit 0 ("not yet")
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "mmlu,static")
    assert r.returncode == 0 and r.stdout.count("has no pending file yet") >= 3, r.stdout + r.stderr
    assert "ADMITTED ONCE" in r.stdout and (w.state / "gate" / "coverage_contract.json").is_file(), r.stdout
    side = json.load(open(w.state / "gate" / "coverage_contract.json")); side_bytes = (w.state / "gate" / "coverage_contract.json").read_bytes()
    assert side["source"] == "admitted_once" and side["inventory_code_sha"] == w.sha and side["probe_code_sha"] == w.sha and side["probe_sha256"] == probe_sha
    assert not list((w.state / "bindings").glob("*.registered.json")) and not (w.state / "gate" / "coverage").exists()
    climb_lw = f"{MODEL}/battery/bon/climb"; climb_job = ST.binding_state(str(w.state / "submissions.jsonl"), climb_lw)["job_id"]
    climb_cl = ST.binding_state(str(w.state / "submissions.jsonl"), climb_lw)["cluster"]
    # ---- a DISCONNECTED login host is a gate FAILURE (exit 1, named; sacct over ssh fails first on the array), never "not registered yet"
    (w.fake[climb_cl] / "ssh_down").write_text("")
    bindings_before = {p.name: p.stat().st_mtime_ns for p in (w.state / "bindings").iterdir()}
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "bon")
    assert r.returncode == 1 and "over ssh (rc 255)" in r.stdout and "a transport failure" in r.stdout, r.stdout + r.stderr
    assert "REFUSED: 06 --gate:" in r.stderr and "hit a reader / transport failure" in r.stderr and climb_lw in r.stderr, r.stderr
    assert ST.binding_state(str(w.state / "submissions.jsonl"), climb_lw)["state"] == "submitted" and "has no pending file yet" not in r.stdout
    assert {p.name: p.stat().st_mtime_ns for p in (w.state / "bindings").iterdir()} == bindings_before
    (w.fake[climb_cl] / "ssh_down").unlink()
    # ---- the jobs register (a pending file per ARRAY TASK, each past MAX_ARG_STRLEN), the harvester ingests (registry rows), the producers write
    # their REAL layouts; --gate binds + covers + rules
    _register_every_submitted_unit(w)
    pend_sizes = sorted(p.stat().st_size for lab in w.shared for p in (w.shared[lab] / "registry" / "pending").glob("*.json"))
    assert pend_sizes[-1] > MAX_ARG_STRLEN and sum(1 for s in pend_sizes if s > MAX_ARG_STRLEN) >= 2, pend_sizes[-5:]     # the smoke's static shape
    # a pending file that is NOT JSON (one static unit): the parse failure is named, that unit stays submitted, the other binds, the gate exits 1
    t1_lw = f"{MODEL}/battery/static/t1"; b1 = ST.binding_state(str(w.state / "submissions.jsonl"), t1_lw)
    t1_pend = w.shared[b1["cluster"]] / "registry" / "pending" / f"{b1['job_id']}.json"; t1_text = t1_pend.read_text(); t1_pend.write_text("not json {")
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "static")
    assert r.returncode == 1 and "REFUSED: pending-runs: pending file" in r.stdout + r.stderr and "could not be parsed" in r.stdout and t1_lw in r.stderr, r.stdout[-2000:] + r.stderr[-2000:]
    assert ST.binding_state(str(w.state / "submissions.jsonl"), t1_lw)["state"] == "submitted" and ST.binding_state(str(w.state / "submissions.jsonl"), f"{MODEL}/battery/static/t0")["state"] == "registered"
    assert "names no runs" not in r.stdout
    t1_pend.write_text(t1_text)
    tasks = array_tasks(w.fake[climb_cl], climb_job)
    assert [k for _, k in tasks] == [0, 1, 2, 3] and tasks[0][0] == climb_job                    # Slurm's shape: task 0 = the array id, the rest raw ids
    last_pend = w.shared[climb_cl] / "registry" / "pending" / f"{tasks[3][0]}.json"
    hidden = last_pend.read_text(); last_pend.unlink()                                            # task 3 has not registered yet
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "bon")
    assert r.returncode == 0 and f"task pending file(s) not written yet: {last_pend}" in r.stdout, r.stdout + r.stderr
    assert ST.binding_state(str(w.state / "submissions.jsonl"), climb_lw)["state"] == "submitted"   # never bound on 3 of 4 tasks
    last_pend.write_text(hidden)
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "all", "--headline", "0.5", "--range", "0,1")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert "registry: run run-" in r.stdout and "coverage " in r.stdout
    # the shipped probe: the verified bytes at the content-addressed name on both clusters (the half file replaced); the second pass READ the sidecar
    assert all(p.read_bytes() == probe_bytes for p in shipped.values()), {lab: p.stat().st_size for lab, p in shipped.items()}
    assert "ADMITTED ONCE" not in r.stdout and f"admitted once on {side['admitted_utc']}" in r.stdout and (w.state / "gate" / "coverage_contract.json").read_bytes() == side_bytes
    # the test: EVERY unit of every family reads as covered by the CURRENT probe at its RECORDED store — the multi-turn targets at the grouped
    # reproduce root (the frozen root_rel does not exist), the controls from the gens spool, the agentic cells at their harness cell dirs, the Petri
    # cells on the grading host; the climb alone, whose flat root NO record names yet, reads "not covered, no store on record" by name
    for u in w.inventory()["units"]:
        b = ST.binding_state(str(w.state / "submissions.jsonl"), u["logical_work_id"])
        assert b["state"] == "registered" and set(r_["eval_id"] for r_ in b["runs"]) == set(u["evals"]), u["logical_work_id"]
        cov = json.load(open(ST.coverage_path(str(w.state), u["logical_work_id"])))
        assert set(cov["run_ids"]) == set(b["run_ids"])                                          # tied to the BOUND runs
        assert cov["inventory_code_sha"] == w.sha and cov["probe_code_sha"] == w.sha and cov["probe_sha256"] == probe_sha
        assert cov["coverage_contract"]["source"] == "admitted_once" and cov["coverage_contract"]["sha256"] == side["sha256"] and cov["coverage_contract"]["amendments"] == []
        is_climb = any(ev["coverage"]["layout"] == "bon_climb_flat" for ev in u["evals"].values())
        for rid, c in cov["runs"].items():
            if is_climb:
                assert c["probe"] is None and "not covered, no store on record for the climb's flat root" in c["probe_error"] and "D3003 (1)" in c["probe_error"], (rid, c["probe_error"])
                assert c["store_source"] == "none" and c["probe_rc"] == 0 and f"/transcripts_results/sdfbase_full22_{MODEL}_battery/none" in c["probe_error"]
            else:
                assert c["probe"] and not c["probe_error"], (u["logical_work_id"], rid, c)    # the Petri unit's probe ran on the grading host too
                assert c["store_source"].startswith("registry.run.store_uri") and c["probe"]["path"] == c["store_uri"] and c["probe_rc"] == 0, (u["logical_work_id"], rid, c["store_source"])
    mt_u = next(u for u in w.inventory()["units"] if u["family"] == "multiturn")
    cov_mt = json.load(open(ST.coverage_path(str(w.state), mt_u["logical_work_id"])))
    for c in cov_mt["runs"].values():
        assert "/exp/reproduce/" in c["probe"]["path"] and "/waves/mt-battery-" in c["probe"]["path"] and c["probe"]["exists"] and all(blk["present"] for blk in c["probe"]["shards"].values()), c["probe"]["path"]
    mt_b = ST.binding_state(str(w.state / "submissions.jsonl"), mt_u["logical_work_id"])
    for ev in mt_u["evals"].values():                                                                # the frozen root_rel was never written: the old planner read "does not exist" there
        assert not (w.shared[mt_b["cluster"]] / ev["coverage"]["path"]["root_rel"].replace("<cell>", "none")).exists()
    cov_c0 = json.load(open(ST.coverage_path(str(w.state), f"{MODEL}/battery/controls/t0")))
    for c in cov_c0["runs"].values():
        it = c["probe"]["items"]
        assert it["present"] and it["source"].startswith("private/gens_") and it["refusal"] is None and it["gen_stats"]["decode"] == "present" and it["items_sha256"] is None, it
        assert (len(it["completed"]), it["gen_stats"]["control"]) in ((450, "xstest"), (1319, "or_bench_hard_1k")) and not (Path(c["store_uri"]) / "rows.jsonl").exists()
    cov_ag = json.load(open(ST.coverage_path(str(w.state), f"{MODEL}/battery/agentic/proxy")))
    for c in cov_ag["runs"].values():
        assert c["probe"]["path"].endswith("/agentic/none") and c["probe"]["path"] == c["store_uri"] and c["probe"]["shards"]["shard00"]["present"] and c["probe"]["meta"]["n_samples"] == 10 and c["probe"]["notes"] == []
    v_static = json.load(open(w.state / "gate" / "static.json"))
    assert v_static["verdict"] == "pass" and v_static["inventory_code_sha"] == w.sha and v_static["probe_code_sha"] == w.sha and v_static["probe_sha256"] == probe_sha
    assert v_static["coverage_contract"]["source"] == "admitted_once" and v_static["coverage_contract"]["record_sha256"] == side["sha256"] and v_static["coverage_contract"]["amendments"] == []
    # every array task bound: four runs, one per task (the per-task shape); admission refuses the climb unit ALONE, by the record's words
    b = ST.binding_state(str(w.state / "submissions.jsonl"), climb_lw)
    assert len(b["run_ids"]) == 4 and sorted(r_["array_task"] for r_ in b["runs"]) == [0, 1, 2, 3] and b["pending_file"].endswith("/<raw task id>.json")
    adm = subprocess.run([VENV_PY, str(LIB / "admission.py"), "check", "--state-root", str(w.state), "--registry", str(w.registry)], capture_output=True, text=True)
    assert adm.returncode == 3, adm.stdout + adm.stderr
    refused = [ln for ln in adm.stderr.splitlines() if ln.startswith("REFUSED: ")]
    assert refused and all(ln.startswith(f"REFUSED: {climb_lw}: ") for ln in refused), refused[:3]                  # the climb unit ALONE; every other unit admits
    # per run: the probe never ran, by the record's words; and, the pre-existing gate lines over the same record, its own shard was
    # never read + the cell's shards were read for no bound run — the three classes, nothing else
    no_store = [ln for ln in refused if "the coverage probe did not run or failed: run " in ln and "not covered, no store on record for the climb's flat root" in ln and "D3003 (1)" in ln]
    assert all(any(f": run {rid} (" in ln for ln in no_store) for rid in b["run_ids"]), (b["run_ids"], no_store)
    unread = [ln for ln in refused if ln not in no_store]
    assert unread and all("were never read (a shard read for another run of the eval stands in for nothing)" in ln or "were read for no bound run of the cell" in ln for ln in unread), unread[:3]
    # ---- the producer's closer records the climb's flat root (registry_close_stores.py: run_patch extra_json.climb_root beside the store facts, folded into
    # the row): the SAME gate then reads every shard's expected set from the climb MANIFEST at the RECORDED root and admission admits
    runs2 = [({**x, "extra": {"climb_root": x["climb_root_record"]}} if x.get("climb_root_record") else x) for x in w.reg_runs]
    make_registry(w.tmp / "registry", runs2, w.reg_policies)
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "bon", "--headline", "0.5", "--range", "0,1")
    assert r.returncode == 0 and "coverage recomputed" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]
    cov = json.load(open(ST.coverage_path(str(w.state), climb_lw)))
    for rid, c in cov["runs"].items():
        assert c["store_source"] == "registry.run.extra_json.climb_root" and not c["probe_error"] and c["probe"]["path"] == c["store_uri"].split("/transcripts_results/")[0] + f"/bon_climb/{MODEL}_battery", (rid, c)
        blk = c["probe"]["shards"][f"shard{c['array_task']:02d}"]
        assert blk["expected"] and blk["completed"] == blk["expected"] and set(blk["expected"]) <= set(GOAL_IDS[:20]), (rid, blk)
    # the mmlu unit bound from its OWN pending file (the real f22_registry_register): three runs (one per cell of the view) on the
    # registry planner's 1,531-item instrument, each store attested through the real CellStore emission
    b = ST.binding_state(str(w.state / "submissions.jsonl"), f"{MODEL}/battery/mmlu/mmlu")
    assert len(b["run_ids"]) == 3 and b["pending_file"].endswith(f"/registry/pending/{b['job_id']}.json") and {r_["object"] for r_ in b["runs"]} == {"none", "prompted__safety_refusal", "prompted__si_constitution"}
    assert all(":mmlu:" in rid and rid.startswith("sdfbase_full22:") for rid in b["run_ids"])          # identity.run_id: <campaign>:<eval>:<instrument12>:<producer key>
    inst12 = units["mmlu/mmlu"]["evals"]["mmlu"]["instrument_id"][:12]
    assert all(f":mmlu:{inst12}:" in rid for rid in b["run_ids"])                                    # the planner's instrument IS the frozen one
    cov_m = json.load(open(ST.coverage_path(str(w.state), f"{MODEL}/battery/mmlu/mmlu")))
    for rid in b["run_ids"]:
        it = cov_m["runs"][rid]["probe"]["items"]
        assert len(it["completed"]) == 1531 and it["items_sha256"] == MMLU_SHA and it["task"] == "s2parity/mmlu_letter" and it["mmlu_item_set_spec"]["n"] == 1531
        assert f"/exp/sdfbase_full22_native/mmlu_full/{MODEL}_battery_t0/cells/" in cov_m["runs"][rid]["store_uri"]    # the sbatch's layout, from the run's OWN record
    spool_states = [p["state"] for rt, p in read_spool(w.shared[b["cluster"]] / "registry/records") if rt == "run_state" and ":mmlu:" in str(p.get("run_id"))]
    assert spool_states.count("generated") == 3                                                          # the job filed its terminal state
    # ---- admit: EVERY unit, the Petri unit included — its 21 runs bound from the proxy's OWN pending file on the cluster (7 evals x 3 objects),
    # each run's store the REAL grading-host cell store the native hand-off recorded (store_uri from the recorder / close-leg, never
    # fabricated), read by the real petri_cellstore reader on the grading host: no --force-gate anywhere
    b = ST.binding_state(str(w.state / "submissions.jsonl"), f"{MODEL}/battery/petri/proxy")
    assert len(b["run_ids"]) == 21 and {r_["object"] for r_ in b["runs"]} == {"none", "prompted__safety_refusal", "prompted__si_constitution"}
    cov_p = json.load(open(ST.coverage_path(str(w.state), f"{MODEL}/battery/petri/proxy")))
    assert len(cov_p["runs"]) == 21 and all(c["probe"]["layout"] == "petri_cellstore" and c["probe"]["results_json_present"] and not c["probe_error"] for c in cov_p["runs"].values())
    real_cells = {str(p_) for p_ in w.petri_cells.values()}
    assert {c["store_uri"] for c in cov_p["runs"].values()} == real_cells and all(f"/.local/state/petri_native/petri_full22_{MODEL}_battery/cells/" in s_ for s_ in real_cells)
    assert {c["probe"]["path"] for c in cov_p["runs"].values()} == real_cells
    assert {len(c["probe"]["shards"]["shard00"]["completed"]) for c in cov_p["runs"].values()} == {10, 60}      # six behaviors x 10 per behaviour run, 60 for the pooled
    adm = subprocess.run([VENV_PY, str(LIB / "admission.py"), "check", "--state-root", str(w.state), "--registry", str(w.registry)], capture_output=True, text=True)
    assert adm.returncode == 0 and "ADMITTED" in adm.stdout, adm.stdout + adm.stderr
    # ---- `--gate <family>|all --verdict`: each family's headline READ from the registry (lib/gate_headline.py) over the runs the family is bound
    # to, never typed. The grader's side of the world's registry: mmlu rule-scored (judge-less rows; its one unsteered leg is the `none` cell's
    # run, the two prompted cells' lower values never move it) and controls judged (two seeded trials x XSTest / OR-Bench; steering is expected to
    # RAISE over-refusal, so the MAXIMUM decides). No other family files a headline row: the dry run names each skipped and exits 3, writing nothing
    ledger = str(w.state / "submissions.jsonl")
    mm_runs = ST.binding_state(ledger, f"{MODEL}/battery/mmlu/mmlu")["runs"]
    mm_none = next(r_["run_id"] for r_ in mm_runs if r_["object"] == "none")
    ctl_runs = [r_ for t in ("t0", "t1") for r_ in ST.binding_state(ledger, f"{MODEL}/battery/controls/{t}")["runs"]]
    assert len(ctl_runs) == 4 and {r_["eval_id"] for r_ in ctl_runs} == {"xstest", "or_bench_hard"}, ctl_runs
    ctl_head = {"xstest": "xst_refused", "or_bench_hard": "orb_refused"}
    ctl_val = {r_["run_id"]: v for r_, v in zip(ctl_runs, (0.04, 0.62, 0.03, 0.66))}
    judge = "judge-" + "c" * 58
    def grade_controls(vals):
        add_grading_tables(w.registry, metrics=[{"run_id": rid, "judgment_id": f"j-{rid}", "metric_id": ctl_head[r_["eval_id"]], "value": vals[rid]}
                                                for r_ in ctl_runs for rid in [r_["run_id"]]])
    add_grading_tables(w.registry, evals=[{"eval_id": e, "headline_metric_id": {"mmlu": "acc", **ctl_head}.get(e, "asr")} for e in {x["eval_id"] for x in w.reg_runs}],
                       judgments=[{"judgment_id": f"j-{r_['run_id']}", "run_id": r_["run_id"], "judge_id": judge} for r_ in ctl_runs],
                       metrics=[{"run_id": r_["run_id"], "metric_id": "acc", "value": 0.74 if r_["run_id"] == mm_none else 0.01, "n": 1531} for r_ in mm_runs],
                       run_judges={r_["run_id"]: judge for r_ in ctl_runs})
    grade_controls(ctl_val)
    verdict_bytes = {p_.name: p_.read_bytes() for p_ in (w.state / "gate").glob("*.json")}
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "all", "--verdict", "--dry-run")
    assert r.returncode == 3, r.stdout[-3000:] + r.stderr[-2000:]
    assert f"gate mmlu: leg {mm_none} mmlu acc = 0.74 (n 1531)" in r.stdout and "gate mmlu: headline 0.74, the minimum over the unsteered legs" in r.stdout, r.stdout[-3000:]
    assert "gate controls: headline 0.66, the maximum over the unsteered legs (steering is expected to raise it), over 4 leg(s); range [0, 0.95] -> pass" in r.stdout
    others = [f for f in ("static", "padding", "bon", "multiturn", "petri", "agentic", "gsm8k") if f"gate {f}: SKIPPED, not ready to rule: " in r.stdout]
    assert others == ["static", "padding", "bon", "multiturn", "petri", "agentic", "gsm8k"], r.stdout[-3000:]
    assert "06 --gate --verdict: skipped, not ready to rule: static padding bon multiturn petri agentic gsm8k; ruled fail: none — exit 3" in r.stderr, r.stderr
    assert {p_.name: p_.read_bytes() for p_ in (w.state / "gate").glob("*.json")} == verdict_bytes, "a dry run writes no verdict"
    # the real pass over the two graded families: each verdict carries the registry's value, every leg, and the runs the family is bound to
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "mmlu,controls", "--verdict")
    assert r.returncode == 0 and "gate verdict mmlu: pass (headline 0.74 in [0.05, 1.0])" in r.stdout and "gate verdict controls: pass (headline 0.66 in [0.0, 0.95])" in r.stdout, \
        r.stdout[-3000:] + r.stderr[-2000:]
    inv_units = w.inventory()["units"]
    v_mm, v_ctl = (json.load(open(w.state / "gate" / f"{f}.json")) for f in ("mmlu", "controls"))
    assert v_mm["headline"] == 0.74 and v_mm["plausible_range"] == [0.05, 1.0] and v_mm["expected"] == "decrease" and v_mm["verdict"] == "pass"
    assert [(g["run_id"], g["metric_id"], g["judgment_id"], g["value"]) for g in v_mm["headline_legs"]] == [(mm_none, "acc", None, 0.74)]
    assert v_ctl["headline"] == 0.66 and v_ctl["plausible_range"] == [0.0, 0.95] and v_ctl["expected"] == "increase"
    assert {g["run_id"]: (g["metric_id"], g["judgment_id"], g["value"]) for g in v_ctl["headline_legs"]} == {
        r_["run_id"]: (ctl_head[r_["eval_id"]], f"j-{r_['run_id']}", ctl_val[r_["run_id"]]) for r_ in ctl_runs}
    for f, v in (("mmlu", v_mm), ("controls", v_ctl)):
        assert v["run_ids"] == ST.family_bound_run_ids(ledger, inv_units, f) and v["probe_code_sha"] == w.sha and v["inventory_code_sha"] == w.sha, f
        assert v["note"] == "recorded by 06_eval_waves.sh --gate --verdict: the headline read from the registry"
    adm = subprocess.run([VENV_PY, str(LIB / "admission.py"), "check", "--state-root", str(w.state), "--registry", str(w.registry)], capture_output=True, text=True)
    assert adm.returncode == 0 and "ADMITTED" in adm.stdout, adm.stdout + adm.stderr
    # a leg with no headroom (one OR-Bench trial refusing 97%): the family is ruled `fail` on that leg alone, the pass exits 3 naming it, and
    # admission refuses the family's units on the verdict; the leg regraded, the same call rules `pass` again and admission admits
    grade_controls({**ctl_val, ctl_runs[3]["run_id"]: 0.97})
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "controls", "--verdict")
    assert r.returncode == 3 and "gate verdict controls: fail (headline 0.97 in [0.0, 0.95])" in r.stdout and "ruled fail: controls" in r.stderr, r.stdout[-2000:] + r.stderr[-1500:]
    adm = subprocess.run([VENV_PY, str(LIB / "admission.py"), "check", "--state-root", str(w.state), "--registry", str(w.registry)], capture_output=True, text=True)
    assert adm.returncode == 3 and f"REFUSED: {MODEL}/battery/controls/t0: family controls verdict is 'fail', not pass" in adm.stderr, adm.stderr[-2000:]
    grade_controls(ctl_val)
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "controls", "--verdict")
    assert r.returncode == 0 and "gate verdict controls: pass (headline 0.66" in r.stdout, r.stdout[-2000:] + r.stderr[-1500:]
    adm = subprocess.run([VENV_PY, str(LIB / "admission.py"), "check", "--state-root", str(w.state), "--registry", str(w.registry)], capture_output=True, text=True)
    assert adm.returncode == 0 and "ADMITTED" in adm.stdout, adm.stdout + adm.stderr
    # a re-grade begun after the ruling (one leg's judgment abandoned, the replacement not complete): admission refuses the pass ruled on that
    # grade, naming the leg; the re-gate skips the family (exit 3) and leaves the verdict as it was, still refused; once the replacement grade
    # lands, the same call rules on it and admission admits
    victim = ctl_runs[1]["run_id"]
    conn = sqlite3.connect(w.registry); conn.execute("UPDATE judgment SET status = 'abandoned' WHERE judgment_id = ?", (f"j-{victim}",)); conn.commit(); conn.close()
    adm = subprocess.run([VENV_PY, str(LIB / "admission.py"), "check", "--state-root", str(w.state), "--registry", str(w.registry)], capture_output=True, text=True)
    assert adm.returncode == 3 and f"REFUSED: {MODEL}/battery/controls/t0: family controls verdict 'pass' (ruled " in adm.stderr, adm.stdout + adm.stderr[-2000:]
    assert f"run {victim} (" in adm.stderr and "no current complete judgment by its judge" in adm.stderr, adm.stderr[-2000:]
    v_bytes = (w.state / "gate" / "controls.json").read_bytes()
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "controls", "--verdict")
    assert r.returncode == 3 and "gate controls: SKIPPED, not ready to rule: " in r.stdout and "skipped, not ready to rule: controls; ruled fail: none" in r.stderr, \
        r.stdout[-2000:] + r.stderr[-1500:]
    assert (w.state / "gate" / "controls.json").read_bytes() == v_bytes
    adm = subprocess.run([VENV_PY, str(LIB / "admission.py"), "check", "--state-root", str(w.state), "--registry", str(w.registry)], capture_output=True, text=True)
    assert adm.returncode == 3 and f"run {victim} (" in adm.stderr, adm.stdout + adm.stderr[-2000:]
    add_grading_tables(w.registry, judgments=[{"judgment_id": f"j2-{victim}", "run_id": victim, "judge_id": judge}],
                       metrics=[{"run_id": victim, "judgment_id": f"j2-{victim}", "metric_id": ctl_head[ctl_runs[1]["eval_id"]], "value": ctl_val[victim]}])
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "controls", "--verdict")
    assert r.returncode == 0 and "gate verdict controls: pass (headline 0.66" in r.stdout, r.stdout[-2000:] + r.stderr[-1500:]
    assert {g["run_id"]: g["judgment_id"] for g in json.load(open(w.state / "gate" / "controls.json"))["headline_legs"]}[victim] == f"j2-{victim}"
    adm = subprocess.run([VENV_PY, str(LIB / "admission.py"), "check", "--state-root", str(w.state), "--registry", str(w.registry)], capture_output=True, text=True)
    assert adm.returncode == 0 and "ADMITTED" in adm.stdout, adm.stdout + adm.stderr
    # ---- coverage is RECOMPUTED on every gate check from the producer's artifact by GOAL IDENTITY (findings 5 + round-2 3): rewrite one inner
    # shard of static/t0's pasttense store with HALF its goals but MORE lines (14 progress + 60 response rows), re-gate, and the admission
    # names the run, the shard and the missing goals
    b = ST.binding_state(str(w.state / "submissions.jsonl"), f"{MODEL}/battery/static/t0")
    rid = next(r_["run_id"] for r_ in b["runs"] if r_["eval_id"] == "pasttense")
    store = w.shared[b["cluster"]] / units["static/t0"]["evals"]["pasttense"]["coverage"]["path"]["root_rel"].replace("<cell>", "none")
    exp = units["static/t0"]["evals"]["pasttense"]["coverage"]["expected"]["shard04"]
    # ---- a FAILED gate pass writes NO verdict. (1) A SUCCESSFUL pass first: static ruled with an IMPLAUSIBLE headline over
    # the intact stores — every static coverage record refreshed, gate/static.json = `fail`; admission refuses every static unit on the verdict alone
    static_lws = [u["logical_work_id"] for u in w.inventory()["units"] if u["family"] == "static"]
    assert len(static_lws) >= 2
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "static", "--headline", "5", "--range", "0,1")
    assert r.returncode == 0 and "gate verdict static: fail (headline 5" in r.stdout and "NO verdict written" not in r.stdout, r.stdout[-2000:] + r.stderr[-1500:]
    # the pass's opening line names the contract it reads under (version, sha, source, admitted when, amendments) and the gate's stderr carries no
    # Python noise: the line's one-liner escaped quotes inside a single-quoted -c string, a SyntaxError on stderr and an empty description (r2)
    assert f"coverage contract cc1 {side['sha256'][:12]} (admitted_once, admitted {side['admitted_utc']}, 0 amendment(s)): bind units" in r.stdout, r.stdout[:3000]
    assert "SyntaxError" not in r.stderr and "Traceback" not in r.stderr, r.stderr[-2000:]
    v_fail_bytes = (w.state / "gate" / "static.json").read_bytes(); v_fail = json.loads(v_fail_bytes)
    assert v_fail["verdict"] == "fail" and v_fail["headline"] == 5.0 and v_fail["probe_code_sha"] == w.sha
    cov_static_bytes = {lw: Path(ST.coverage_path(str(w.state), lw)).read_bytes() for lw in static_lws}
    assert all(json.loads(c_)["probe_code_sha"] == w.sha for c_ in cov_static_bytes.values())
    adm = subprocess.run([VENV_PY, str(LIB / "admission.py"), "check", "--state-root", str(w.state), "--registry", str(w.registry)], capture_output=True, text=True)
    refused_on_verdict = {ln for ln in adm.stderr.splitlines() if ln.startswith("REFUSED: ") and ln != adm.stderr.splitlines()[-1]}
    assert adm.returncode == 3 and refused_on_verdict == {f"REFUSED: {lw}: family static verdict is 'fail', not pass" for lw in static_lws}, adm.stderr
    # (2) the store breaks (shard04 rewritten with HALF its goals but MORE lines: 14 progress + 60 response rows) and every login host is unreachable;
    # the re-gate with a PLAUSIBLE headline is a transport failure (the current probe cannot be shipped — rc 255 before any store is read): exit 1,
    # NO verdict written — gate/static.json is byte-for-byte the `fail` of (1), every static coverage record byte-for-byte the record of (1), the
    # note names the units and the verdict that stands, and admission's answer is unchanged. Before the fix this pass wrote `pass` (this pass's
    # provenance over coverage it never read) and admission ADMITTED the family on the stale records — over a store that is incomplete on disk
    _jsonl(store / "shard04" / "progress.jsonl", [{"goal_id": g, "outcome": "success"} for g in exp[:14]])
    _jsonl(store / "shard04" / "responses.jsonl", [{"goal_id": exp[0], "sample_idx": i} for i in range(60)])
    for d in w.fake.values():
        (d / "ssh_down").write_text("")
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "static", "--headline", "0.5", "--range", "0,1")
    assert r.returncode == 1 and r.stdout.count("the current coverage_probe.py could not be shipped to ") == len(static_lws) and "(rc 255:" in r.stdout, r.stdout[-2500:] + r.stderr[-1500:]
    assert "gate verdict static" not in r.stdout and "DRY-RUN" not in r.stdout and "coverage recomputed" in r.stdout      # the units were reached; nothing ruled
    note = next(ln for ln in r.stdout.splitlines() if "gate static: NO verdict written" in ln)
    assert f"{len(static_lws)} unit(s) of the family hit a reader / transport failure in this pass (" in note and all(lw in note for lw in static_lws), note
    assert f"gate/static.json is left as it was (verdict 'fail' ruled {v_fail['ruled_utc']} by hand)" in note and "re-run 06 --gate static --headline 0.5 --range 0,1" in note, note
    assert "REFUSED: 06 --gate:" in r.stderr and "NO verdict was written for its family" in r.stderr and all(lw in r.stderr for lw in static_lws), r.stderr
    assert (w.state / "gate" / "static.json").read_bytes() == v_fail_bytes                                            # the prior verdict stands, untouched
    assert {lw: Path(ST.coverage_path(str(w.state), lw)).read_bytes() for lw in static_lws} == cov_static_bytes      # every record of (1) stands, untouched
    adm = subprocess.run([VENV_PY, str(LIB / "admission.py"), "check", "--state-root", str(w.state), "--registry", str(w.registry)], capture_output=True, text=True)
    assert adm.returncode == 3 and "ADMITTED" not in adm.stdout, adm.stdout + adm.stderr
    assert {ln for ln in adm.stderr.splitlines() if ln.startswith("REFUSED: ") and ln != adm.stderr.splitlines()[-1]} == refused_on_verdict, adm.stderr
    for d in w.fake.values():
        (d / "ssh_down").unlink()
    # (3) the hosts back: the SAME call refreshes every static record over the broken store and rules `pass` on the plausible headline — and admission
    # now refuses on the COVERAGE (the run, the shard, the missing goals), never on the verdict: the breakage is caught only through a refreshed
    # record, which a failed pass cannot provide (findings 5 + round-2 3: coverage is RECOMPUTED on every gate check by GOAL IDENTITY)
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "static", "--headline", "0.5", "--range", "0,1")
    assert r.returncode == 0 and "coverage recomputed" in r.stdout and "gate verdict static: pass" in r.stdout and "NO verdict written" not in r.stdout, r.stdout[-2000:] + r.stderr[-1500:]
    v_pass = json.load(open(w.state / "gate" / "static.json"))
    assert v_pass["verdict"] == "pass" and v_pass["headline"] == 0.5 and v_pass["ruled_utc"] >= v_fail["ruled_utc"]
    assert all(Path(ST.coverage_path(str(w.state), lw)).read_bytes() != cov_static_bytes[lw] for lw in static_lws)    # refreshed (collected_utc moved)
    adm = subprocess.run([VENV_PY, str(LIB / "admission.py"), "check", "--state-root", str(w.state), "--registry", str(w.registry)], capture_output=True, text=True)
    assert adm.returncode == 3 and f"static/t0: run {rid} (pasttense) shard shard04: 15 of 29 expected goal(s) not completed" in adm.stderr and exp[14] in adm.stderr, adm.stderr
    assert "verdict is 'fail'" not in adm.stderr
    # ---- Static stands at `pass` (v_pass) with every static record fresh — the precondition the two cases below share.
    v_pass_bytes = (w.state / "gate" / "static.json").read_bytes()
    cov_pass_bytes = {lw: Path(ST.coverage_path(str(w.state), lw)).read_bytes() for lw in static_lws}
    snap_gate = lambda: {str(p): p.stat().st_mtime_ns for d in ("bindings", "gate") for p in (w.state / d).rglob("*") if p.is_file()}  # noqa: E731
    # (r3 Major 1, the argument half) a malformed or half-given --headline/--range is refused BY NAME before anything is bound or covered: exit 2,
    # no Traceback, every record and the verdict byte-for-byte as they stand. Before the fix `stage.py verdict` raised AFTER the whole family was
    # bound and covered, its traceback's rc 1 (a persisted `fail`'s status too) was swallowed by `|| true`, and the pass exited 0 with the OLD
    # `pass` standing over the coverage it had just refreshed — which admission reads as admissible
    before = snap_gate()
    for bad in (("--headline", "0.5", "--range", "0;1"), ("--headline", "abc", "--range", "0,1"), ("--headline", "0.5", "--range", "1,0"), ("--headline", "0.5")):
        r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "static", *bad)
        assert r.returncode == 2 and "REFUSED: 06 --gate:" in r.stderr and "nothing bound, no coverage written" in r.stderr, (bad, r.stdout[-1500:] + r.stderr[-1500:])
        assert ("--range" in r.stderr or "--headline" in r.stderr) and "Traceback" not in r.stderr and "coverage recomputed" not in r.stdout and "gate verdict" not in r.stdout, (bad, r.stderr)
        assert snap_gate() == before and (w.state / "gate" / "static.json").read_bytes() == v_pass_bytes, bad
    assert "pass BOTH --headline <value> and --range <lo,hi>" in r.stderr, r.stderr                                      # the half-given pair (the last case)
    # a family the frozen inventory does not carry is refused by name before any bind (the old unit-list process substitution read it as "no units"
    # and the verdict step RULED over nothing read); `all` says and skips an absent family instead
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "nosuch,static")
    assert r.returncode == 2 and "REFUSED: 06 --gate: ['nosuch'] not a family of the frozen inventory" in r.stderr and "static" in r.stderr and snap_gate() == before, r.stderr
    # (r3 Major 1, the mechanical half) the verdict step itself fails AFTER the family was covered — gate/ made unwritable; the records live under
    # gate/coverage/ and still land: exit 1, the note says NO verdict was written over the standing `pass`, the REFUSED line names the family,
    # gate/static.json byte-for-byte as it was, every static record refreshed
    if os.geteuid() != 0:                                                                                                  # root ignores directory modes
        gate_dir = w.state / "gate"; gate_mode = gate_dir.stat().st_mode & 0o777
        gate_dir.chmod(0o500)
        try:
            r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "static", "--headline", "0.5", "--range", "0,1")
        finally:
            gate_dir.chmod(gate_mode)
        assert r.returncode == 1 and "coverage recomputed" in r.stdout and "gate verdict static" not in r.stdout, r.stdout[-2500:] + r.stderr[-1500:]
        note = next(ln for ln in r.stdout.splitlines() if "gate static: NO verdict written — stage.py verdict exited" in ln)
        assert f"gate/static.json is left as it was (verdict 'pass' ruled {v_pass['ruled_utc']} by hand)" in note and "re-run 06 --gate static --headline 0.5 --range 0,1" in note, note
        assert "REFUSED: 06 --gate:" in r.stderr and "1 family verdict(s) could not be recorded" in r.stderr and "static" in r.stderr and "NO verdict was written for the family" in r.stderr, r.stderr
        assert (w.state / "gate" / "static.json").read_bytes() == v_pass_bytes                                            # the standing pass, untouched
        assert all(Path(ST.coverage_path(str(w.state), lw)).read_bytes() != cov_pass_bytes[lw] for lw in static_lws)      # the stores were read: refreshed
        cov_pass_bytes = {lw: Path(ST.coverage_path(str(w.state), lw)).read_bytes() for lw in static_lws}
    # (r3 Major 2) every login host prints a BANNER on stdout before the command's own output (a login shell's rc that echoes): each probe exits 0
    # and its stdout is no JSON document — a READER failure: exit 1, NO verdict written (the standing `pass` byte-for-byte), every static record
    # refreshed and naming the failure per run (probe null, rc 0, the banner's text). Before the fix the record read exactly so, the family's
    # watermark never moved, and a fresh `pass` was written over it with this pass's provenance
    for d in w.fake.values():
        (d / "ssh_banner").write_text("Welcome to TESTHOST — scheduled maintenance tonight\n")
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "static", "--headline", "0.5", "--range", "0,1")
    assert r.returncode == 1 and "gate verdict static" not in r.stdout and "coverage recomputed" in r.stdout, r.stdout[-2500:] + r.stderr[-1500:]
    assert r.stdout.count("no probe document was read — the call exited 0 and printed no JSON document on stdout") >= len(static_lws), r.stdout[-3000:]
    note = next(ln for ln in r.stdout.splitlines() if "gate static: NO verdict written" in ln)
    assert f"gate/static.json is left as it was (verdict 'pass' ruled {v_pass['ruled_utc']} by hand)" in note and all(lw in note for lw in static_lws), note
    assert "REFUSED: 06 --gate:" in r.stderr and "hit a reader / transport failure" in r.stderr and "NO verdict was written for its family" in r.stderr, r.stderr
    assert (w.state / "gate" / "static.json").read_bytes() == v_pass_bytes
    for lw in static_lws:
        cov_b = Path(ST.coverage_path(str(w.state), lw)).read_bytes(); assert cov_b != cov_pass_bytes[lw], lw
        for rid_, c in json.loads(cov_b)["runs"].items():
            assert c["probe"] is None and c["probe_rc"] == 0 and c["probe_error"].startswith("the probe printed no JSON document (rc 0;") and "Welcome to TESTHOST" in c["probe_error"], (lw, rid_, c)
    for d in w.fake.values():
        (d / "ssh_banner").unlink()
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "static", "--headline", "0.5", "--range", "0,1")               # the banner gone: the same call rules
    assert r.returncode == 0 and "gate verdict static: pass" in r.stdout and "NO verdict written" not in r.stdout, r.stdout[-2000:] + r.stderr[-1500:]
    assert json.load(open(w.state / "gate" / "static.json"))["ruled_utc"] >= v_pass["ruled_utc"]
    assert all(isinstance(c["probe"], dict) for lw in static_lws for c in json.load(open(ST.coverage_path(str(w.state), lw)))["runs"].values())
    # ---- lib/reconcile.sh: the documented recovery command runs the twin reconcile with the flags in the right order
    rec = subprocess.run(["bash", str(w.repo / "reproduce/lib/reconcile.sh"), "--model", MODEL, "--config", str(w.site / "config.env")],
                         capture_output=True, text=True, env=w.env(REPRO_DRY_RUN="0"), cwd=str(w.repo), timeout=300)
    assert rec.returncode == 0, rec.stdout + rec.stderr
    assert "LIVE cluster_a" in rec.stdout and "usage:" not in rec.stderr and "the following arguments are required" not in rec.stderr
    # submit_twin.py reconcile really ran over the ledger — and a PLAIN (never --twin) unit whose sole attempt registered runs is NOT an
    # election with a run from a non-winner: nothing halted (the defect this suite found in reconcile())
    assert "reconcile: 40 unit(s) in the ledger — 40 plain submission(s) (no election), 0 election(s) examined, rc=0" in rec.stdout, rec.stdout[-1500:]
    assert not [a for a in w.attempts() if a["event"] == "halted"]
    dry = subprocess.run(["bash", str(w.repo / "reproduce/lib/reconcile.sh"), "--model", MODEL, "--dry-run", "--config", str(w.site / "config.env")],
                         capture_output=True, text=True, env=w.env(), cwd=str(w.repo), timeout=300)
    assert dry.returncode == 0 and re.search(rf"DRY-RUN: \S+ \S+submit_twin\.py --model {MODEL} --config \S+ --dry-run reconcile$", dry.stdout, re.M), dry.stdout
    # ---- --resubmit of a still-running bound job is refused; of a registered unit too (never re-mint an identity)
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--view", "battery", "--family", "mmlu", "--resubmit")
    # a registered unit whose job is still RUNNING is SKIPPED, never re-minted (the fenced proof decides a registered binding the same way
    # as a submitted one and re-runs it only once its job is dead); nothing goes out
    assert r.returncode == 0 and "skipped cluster_a" in r.stdout and "mmlu.mmlu (still RUNNING" in r.stdout and "submitted " not in r.stdout, r.stdout + r.stderr
    # ---- a COMPLETED registered unit is SKIPPED by the family pass too — its results stand and the pass walks on (a family pass that stopped
    #      at a completed target never reached the rows behind it, 2026-09-30 21:07Z); a direct re-run of it stays refused by the funnel
    mm = [a for a in w.attempts() if a["event"] == "attempt" and a["unit"] == "mmlu.mmlu"][-1]
    set_job_state(w.fake[mm["cluster"]], mm["job_id"], "COMPLETED", end="2026-01-01T01:00:00")
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--view", "battery", "--family", "mmlu", "--resubmit")
    assert r.returncode == 0 and f"skipped {mm['cluster']} {mm['job_id']} mmlu.mmlu (COMPLETED: results stand" in r.stdout and "submitted " not in r.stdout, r.stdout + r.stderr
    # ---- admission-fix: a bound run a `supersede` RULING retired is covered through its ruled SUCCESSOR — 06 --gate resolves the
    # ruling (ruling.target_json.superseded_by_run_id), probes the successor's RECORDED store (never the frozen root_rel, the retired job's own
    # path — the one rewritten with half its goals above) for THIS unit's shards on the successor's cluster, records `successor` on the run, and
    # admission admits the unit through it; without the ruling the superseded run stands in for nothing and admission refuses by name
    import sqlite3 as _sq
    con = _sq.connect(w.registry); con.row_factory = _sq.Row
    reg_runs = [dict(x) for x in con.execute("select * from run")]; reg_pols = [tuple(x) for x in con.execute("select policy_id, policy_ref from decode_policy")]; con.close()
    for x in reg_runs:
        x["extra"] = json.loads(x.pop("extra_json") or "{}")
    grades = grading_rows(w.registry)       # the grader's side, carried through every re-make below: the mmlu and controls passes were ruled on it

    def remake(**kw):
        make_registry(w.tmp / "registry", reg_runs, reg_pols, **kw)
        add_grading_tables(w.registry, **grades)
    b0 = ST.binding_state(str(w.state / "submissions.jsonl"), f"{MODEL}/battery/static/t0"); rid0 = next(r_["run_id"] for r_ in b0["runs"] if r_["eval_id"] == "pasttense")
    b1 = ST.binding_state(str(w.state / "submissions.jsonl"), f"{MODEL}/battery/static/t1"); rid1 = next(r_["run_id"] for r_ in b1["runs"] if r_["eval_id"] == "pasttense")
    consolidated = w.shared[b1["cluster"]] / "stores" / "consolidated" / "pasttense"
    expected = units["static/t0"]["evals"]["pasttense"]["coverage"]["expected"]
    for s_, ids in expected.items():
        _jsonl(consolidated / s_ / "progress.jsonl", [{"goal_id": g, "outcome": "success", "n_target_forwards": 2} for g in ids])
        _jsonl(consolidated / s_ / "responses.jsonl", [{"goal_id": g, "sample_idx": i, "response": "…"} for g in ids for i in range(2)])
    for x in reg_runs:
        if x["run_id"] == rid0:
            x["lifecycle_state"] = "superseded"; x["store_uri"] = None
        if x["run_id"] == rid1:
            x["store_uri"] = str(consolidated)
    ruling = {"ruling_id": "r2v4-supersede-test-t0-by-t1", "action": "supersede", "scope": {"run_id": rid0, "match": {"run_ids": [rid0]}},
              "target": {"superseded_by_run_id": rid1, "decision": "D572"}}
    remake(rulings=[ruling])
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "static")
    assert r.returncode == 0 and "coverage recomputed" in r.stdout, r.stdout + r.stderr
    c = json.load(open(ST.coverage_path(str(w.state), f"{MODEL}/battery/static/t0")))["runs"][rid0]
    assert c["successor"] == {"run_id": rid1, "ruling_id": "r2v4-supersede-test-t0-by-t1", "store_uri": str(consolidated), "cluster": b1["cluster"], "owner": f"{MODEL}/battery/static/t1"}, c
    assert c["store_uri"] == str(consolidated) and c["probe"]["path"] == str(consolidated) and not c["probe_error"], c
    assert set(c["probe"]["shards"]) == set(expected) and all(set(blk["completed"]) == set(expected[s_]) for s_, blk in c["probe"]["shards"].items()), c["probe"]["shards"]
    adm = subprocess.run([VENV_PY, str(LIB / "admission.py"), "check", "--state-root", str(w.state), "--registry", str(w.registry)], capture_output=True, text=True)
    assert adm.returncode == 0 and "ADMITTED" in adm.stdout, adm.stdout + adm.stderr        # the half-goal shard of the retired run no longer counts: the successor's store does
    remake()                                                                                # the same rows, NO ruling: the superseded run stands in for nothing
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "static")
    assert r.returncode == 0, r.stdout + r.stderr
    c = json.load(open(ST.coverage_path(str(w.state), f"{MODEL}/battery/static/t0")))["runs"][rid0]
    assert c["successor"] is None and c["store_uri"] is None, c
    adm = subprocess.run([VENV_PY, str(LIB / "admission.py"), "check", "--state-root", str(w.state), "--registry", str(w.registry)], capture_output=True, text=True)
    assert adm.returncode == 3 and f"static/t0: run {rid0} (pasttense) is superseded and no `supersede` ruling names its successor" in adm.stderr, adm.stderr
    remake(rulings=[ruling])                                                                # the ruled world again (the rest of the test reads it)
    # ---- a `pass` is bound to the RUNS it ruled, not to a coverage generation. static
    # stands ADMITTED in the ruled world. (1) static/t1 is RESUBMITTED — what `submit --resubmit` does to the bindings: the registered binding
    # retired (stage.retire_registered_binding), a new `submitted` one for the new job — and the re-run's registration names ANOTHER run id for one
    # of its evals (the new job's own pending file; the registry row beside the dead attempt's, the same store). The REAL bind-only re-gate binds
    # the new ids from that pending file and refreshes the coverage, and writes NO verdict (no --headline): gate/static.json stands byte-for-byte,
    # ruled over the OLD ids — and admission refuses BOTH static units by the ids added and removed, with the remedy, and nothing else. Before the
    # fix admission read `verdict == pass` alone and ADMITTED the re-bound run on a ruling nobody made over it.
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "static")            # the ruled world's record again (the last gate above ran without the ruling)
    assert r.returncode == 0 and json.load(open(ST.coverage_path(str(w.state), f"{MODEL}/battery/static/t0")))["runs"][rid0]["successor"]["run_id"] == rid1, r.stdout[-1500:]
    adm = subprocess.run([VENV_PY, str(LIB / "admission.py"), "check", "--state-root", str(w.state), "--registry", str(w.registry)], capture_output=True, text=True)
    assert adm.returncode == 0 and "ADMITTED" in adm.stdout, adm.stdout + adm.stderr
    ledger = str(w.state / "submissions.jsonl")
    v_static_bytes = (w.state / "gate" / "static.json").read_bytes(); v_static = json.loads(v_static_bytes)
    static_union = lambda: {r_ for u_ in w.inventory()["units"] if u_["family"] == "static" for r_ in ST.binding_state(ledger, u_["logical_work_id"]).get("run_ids") or []}  # noqa: E731
    assert v_static["verdict"] == "pass" and set(v_static["run_ids"]) == static_union()
    b1 = ST.binding_state(ledger, t1_lw); old_ids = list(b1["run_ids"]); ev_sw = next(e for e in units["static/t1"]["evals"] if e != "pasttense")
    old_rid = next(r_["run_id"] for r_ in b1["runs"] if r_["eval_id"] == ev_sw); new_rid = old_rid + "-rerun"; new_job = "990001"
    pend_doc = json.loads((w.shared[b1["cluster"]] / "registry" / "pending" / f"{b1['job_id']}.json").read_text())
    for r_ in pend_doc["runs"]:
        if r_["run_id"] == old_rid:
            r_["run_id"] = new_rid
    pend_doc["job_id"] = new_job; pend_doc["run_id"] = pend_doc["runs"][0]["run_id"]
    (w.shared[b1["cluster"]] / "registry" / "pending" / f"{new_job}.json").write_text(json.dumps(pend_doc, indent=1))
    retired = ST.retire_registered_binding(ledger, t1_lw, b1.get("attempt_id")); assert retired and ".registered.retired." in retired
    ST.write_binding(ledger, ST.submitted_record(t1_lw, new_job, b1["cluster"], f"{t1_lw}@{b1['cluster']}@2"))
    assert ST.binding_state(ledger, t1_lw)["state"] == "submitted"
    reg_runs.append({**next(x for x in reg_runs if x["run_id"] == old_rid), "run_id": new_rid, "cluster_job_id": new_job})
    remake(rulings=[ruling])
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "static")
    assert r.returncode == 0 and f"bound {t1_lw}: registered" in r.stdout and "coverage recomputed" in r.stdout and "gate verdict" not in r.stdout, r.stdout[-2500:] + r.stderr[-1500:]
    b1n = ST.binding_state(ledger, t1_lw); new_ids = list(b1n["run_ids"])
    assert b1n["state"] == "registered" and b1n["job_id"] == new_job and set(new_ids) == (set(old_ids) - {old_rid}) | {new_rid}, b1n
    cov1 = json.load(open(ST.coverage_path(str(w.state), t1_lw)))
    assert set(cov1["run_ids"]) == set(new_ids) and isinstance(cov1["runs"][new_rid]["probe"], dict) and not cov1["runs"][new_rid]["probe_error"], cov1["runs"].get(new_rid)
    assert (w.state / "gate" / "static.json").read_bytes() == v_static_bytes                      # bind-only: no verdict written; the pass stands over the OLD ids
    expect_ids = (f"verdict 'pass' (ruled {v_static['ruled_utc']}) ruled over runs that differ from the family's current registered bindings: "
                  f"1 bound now that it never ruled ['{new_rid}']; 1 it ruled that are no longer bound ['{old_rid}']")

    def refused_lines(adm_):
        return [ln for ln in adm_.stderr.splitlines() if ln.startswith("REFUSED: ")]
    adm = subprocess.run([VENV_PY, str(LIB / "admission.py"), "check", "--state-root", str(w.state), "--registry", str(w.registry)], capture_output=True, text=True)
    assert adm.returncode == 3 and "ADMITTED" not in adm.stdout, adm.stdout + adm.stderr
    refused = refused_lines(adm)
    assert len(refused) == len(static_lws) and all(sum(1 for ln in refused if ln.startswith(f"REFUSED: {lw}: family static {expect_ids}")) == 1 for lw in static_lws), refused
    assert all("re-gate the family: 06_eval_waves.sh --gate static --verdict (or --headline <value> --range <lo,hi> by hand)" in ln and "D3008" in ln for ln in refused), refused
    # (2) the re-gate with a verdict after the resubmission, and the verdict WRITE fails (gate/ unwritable): exit 1 by name, NO verdict
    # written, the old pass byte-for-byte — and admission still refuses the family by the ids (the written coverage over the new ids admits nothing by itself)
    if os.geteuid() != 0:                                                                                                  # root ignores directory modes
        gate_dir = w.state / "gate"; gate_mode = gate_dir.stat().st_mode & 0o777
        gate_dir.chmod(0o500)
        try:
            r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "static", "--headline", "0.5", "--range", "0,1")
        finally:
            gate_dir.chmod(gate_mode)
        assert r.returncode == 1 and "gate verdict static" not in r.stdout and "gate static: NO verdict written — stage.py verdict exited" in r.stdout, r.stdout[-2500:] + r.stderr[-1500:]
        assert (w.state / "gate" / "static.json").read_bytes() == v_static_bytes
        adm = subprocess.run([VENV_PY, str(LIB / "admission.py"), "check", "--state-root", str(w.state), "--registry", str(w.registry)], capture_output=True, text=True)
        assert adm.returncode == 3 and refused_lines(adm) == refused, adm.stderr
    # (3) the remedy: the re-gate's verdict is recorded over the runs bound NOW (the new id in, the dead attempt's out) — ADMITTED
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "static", "--headline", "0.5", "--range", "0,1")
    assert r.returncode == 0 and "gate verdict static: pass" in r.stdout, r.stdout[-2000:] + r.stderr[-1500:]
    v_static2_bytes = (w.state / "gate" / "static.json").read_bytes(); v_static2 = json.loads(v_static2_bytes)
    assert set(v_static2["run_ids"]) == static_union() and new_rid in v_static2["run_ids"] and old_rid not in v_static2["run_ids"] and v_static2["probe_code_sha"] == w.sha
    adm = subprocess.run([VENV_PY, str(LIB / "admission.py"), "check", "--state-root", str(w.state), "--registry", str(w.registry)], capture_output=True, text=True)
    assert adm.returncode == 0 and "ADMITTED" in adm.stdout, adm.stdout + adm.stderr
    # (4) why not a tie to the coverage generation: a coverage refresh over the SAME runs under ANOTHER probe sha — a committed
    # change to the reader that moves no predicate — is a re-read, not new information: every static record refreshed under the new probe commit
    # and file digest, the verdict standing with the older ones, and admission still ADMITS
    probe_p = w.repo / "reproduce" / "lib" / "coverage_probe.py"; probe_text0 = probe_p.read_text()
    probe_p.write_text(probe_text0 + "\n# a reader comment (no predicate, no identity moves): a re-read of the same runs under another probe sha\n")
    subprocess.run([*GIT, "commit", "-q", "-am", "the probe, a comment"], cwd=w.repo, check=True)
    head_c = subprocess.run(["git", "rev-parse", "HEAD"], cwd=w.repo, capture_output=True, text=True, check=True).stdout.strip(); assert head_c != v_static2["probe_code_sha"]
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "static")
    assert r.returncode == 0 and "coverage recomputed" in r.stdout and "gate verdict" not in r.stdout, r.stdout[-2000:] + r.stderr[-1500:]
    for lw in static_lws:
        c_ = json.load(open(ST.coverage_path(str(w.state), lw)))
        assert c_["probe_code_sha"] == head_c and c_["probe_sha256"] != v_static2["probe_sha256"] and set(c_["run_ids"]) == set(ST.binding_state(ledger, lw)["run_ids"]), lw
    assert (w.state / "gate" / "static.json").read_bytes() == v_static2_bytes
    adm = subprocess.run([VENV_PY, str(LIB / "admission.py"), "check", "--state-root", str(w.state), "--registry", str(w.registry)], capture_output=True, text=True)
    assert adm.returncode == 0 and "ADMITTED" in adm.stdout, adm.stdout + adm.stderr
    probe_p.write_text(probe_text0); subprocess.run([*GIT, "commit", "-q", "-am", "the probe, the comment gone"], cwd=w.repo, check=True)
    # ---- a COMMITTED change to WHAT counts as done (the capability predicate v1 -> v2) refuses the gate by name BEFORE anything is bound —
    # the diff line, the ruling needed; the bindings and coverage records untouched (HEAD advances, the inventory's code_sha stays w.sha; the clean-tree
    # check passes because the change is committed); reverted by a second commit, the gate passes again
    probe_p = w.repo / "reproduce" / "lib" / "coverage_probe.py"; probe_text = probe_p.read_text()
    bumped = probe_text.replace('"capability_cellstore:v1 — ', '"capability_cellstore:v2 — ', 1); assert bumped != probe_text
    probe_p.write_text(bumped); subprocess.run([*GIT, "commit", "-q", "-am", "the capability predicate, v2"], cwd=w.repo, check=True)
    head2 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=w.repo, capture_output=True, text=True, check=True).stdout.strip(); assert head2 != w.sha
    snap = lambda: {str(p): p.stat().st_mtime_ns for d in ("bindings", "gate") for p in (w.state / d).rglob("*") if p.is_file()}  # noqa: E731
    before = snap()
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "controls")
    assert r.returncode == 2 and "REFUSED: 06 --gate: the current probe's done predicates differ from the ones admitted once on" in r.stderr, r.stdout[-1500:] + r.stderr[-1500:]
    # the diff line quotes each predicate with !r — double quotes here, both sentences carry an apostrophe — so the check is quote-agnostic
    assert "done predicate capability_cellstore: " in r.stderr and "capability_cellstore:v1 — " in r.stderr and " -> " in r.stderr and "capability_cellstore:v2 — " in r.stderr \
        and "the operator must rule (D3003 condition 2)" in r.stderr
    assert "nothing bound, no coverage written" in r.stderr and snap() == before and "bind " not in r.stdout and "coverage " not in r.stdout
    probe_p.write_text(probe_text); subprocess.run([*GIT, "commit", "-q", "-am", "the capability predicate, v1 again"], cwd=w.repo, check=True)
    head3 = subprocess.run(["git", "rev-parse", "HEAD"], cwd=w.repo, capture_output=True, text=True, check=True).stdout.strip()
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "controls", "--headline", "0.5", "--range", "0,1")
    assert r.returncode == 0 and f"admitted once on {side['admitted_utc']}" in r.stdout and "+ 0 receipted amendment(s) — ok" in r.stdout, r.stdout[-1500:] + r.stderr[-1500:]
    v_ctl = json.load(open(w.state / "gate" / "controls.json"))
    assert v_ctl["inventory_code_sha"] == w.sha and v_ctl["probe_code_sha"] == head3 != w.sha and v_ctl["probe_sha256"] == probe_sha   # the probe's commit, beside the inventory's
    # ---- a RECEIPTED `rerun-units` stamp is an AMENDMENT of the admitted contract (the review's order: gate -> stamp -> gate): the next gate passes
    # naming the receipt, the verdict records it, every OTHER family's record is now tied to the contract that preceded the stamp (re-run 06 --gate);
    # a unit appended to the inventory BY HAND (no receipt) refuses by name
    (w.repo / "configs" / "sdfbase_full22_roster.yaml").write_text(json.dumps({"targets": {MODEL: {"views": {"battery": {"config": f"configs/sdfbase_full22_{MODEL}_battery.yaml"}}}}}))
    rr = subprocess.run([VENV_PY, str(w.repo / "reproduce" / "lib" / "stage.py"), "rerun-units", "--out", str(w.state / "gate" / "expected_units.json"), "--cells", "none",
                         "--source", "test ruling (synthetic)", "--code-sha", w.sha, "--models-table", str(w.repo / "reproduce" / "models.yaml"),
                         "--roster", str(w.repo / "configs" / "sdfbase_full22_roster.yaml"), "--clusters", "cluster_a,cluster_b", "--goals-jsonl", str(w.goals)],
                        capture_output=True, text=True, env=w.env(), cwd=str(w.repo), timeout=600)
    assert rr.returncode == 0 and "rerun-units: stamped" in rr.stdout, rr.stdout + rr.stderr
    inv_r = w.inventory(); stamp = inv_r["reruns"][-1]
    assert stamp["slug"] == "none_rerun" and stamp["coverage_contract_amendment"]["from_sha256"] == side["sha256"] and stamp["coverage_contract_amendment"]["record_source"] == "admitted_once"
    receipt = [json.loads(l) for l in (w.state / "attempts.jsonl").read_text().splitlines() if '"rerun_units"' in l][-1]
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "bon", "--headline", "0.5", "--range", "0,1")
    assert r.returncode == 0 and "+ 1 receipted amendment(s)" in r.stdout and f"receipt {receipt['receipt_id']}" in r.stdout and "was never submitted (unbound)" in r.stdout, r.stdout[-2500:] + r.stderr[-1500:]
    v_bon = json.load(open(w.state / "gate" / "bon.json"))
    assert v_bon["coverage_contract"]["source"] == "admitted_once" and v_bon["coverage_contract"]["record_sha256"] == side["sha256"] and v_bon["coverage_contract"]["sha256"] != side["sha256"]
    assert v_bon["coverage_contract"]["amendments"] == [{"slug": "none_rerun", "units": sorted(stamp["units"].values()), "receipt_id": receipt["receipt_id"], "receipt_utc": receipt["utc"],
                                                         "stamped_utc": stamp["stamped_utc"], "source": "test ruling (synthetic)", "code_sha": w.sha}]
    assert (w.state / "gate" / "coverage_contract.json").read_bytes() == side_bytes                  # the admitted record is never rewritten
    adm = subprocess.run([VENV_PY, str(LIB / "admission.py"), "check", "--state-root", str(w.state), "--registry", str(w.registry), "--families", "mmlu"], capture_output=True, text=True)
    assert adm.returncode == 3 and f"{MODEL}/battery/mmlu/mmlu: the coverage record was collected under coverage contract {side['sha256'][:12]}…, the inventory's is now" in adm.stderr and "re-run 06 --gate" in adm.stderr, adm.stderr
    hand = json.loads(json.dumps(inv_r["units"][0])); hand["logical_work_id"] = f"{MODEL}/battery/static/t9"; hand["unit"] = "t9"
    ST.write_atomic(str(w.state / "gate" / "expected_units.json"), {**inv_r, "units": inv_r["units"] + [hand], "n_units": inv_r["n_units"] + 1})
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--gate", "mmlu")
    assert r.returncode == 2 and f"unit(s) ['{MODEL}/battery/static/t9'] are not in the admitted_once coverage contract's set and no rerun receipt adds them" in r.stderr, r.stderr[-1500:]
    ST.write_atomic(str(w.state / "gate" / "expected_units.json"), inv_r)


def test_unit_plan_eval_tables_match_the_wave_shells_job_name_table():
    """stage.py's per-unit evals (passed as F22_REGISTRY_EVALS) agree with common.sh f22_registry_evals_for_job for every job kind that
    table knows — the two tables cannot drift apart silently."""
    src = REPO / "scripts/sdfbase/full22/common.sh"
    rows = ST.unit_plan(MODEL, "battery", table=str(REPRO / "models.yaml"), sha="c" * 40, clusters=["cluster_a"])
    script = "source ./common.sh >/dev/null 2>&1; " + "; ".join(f'printf "%s\\t%s\\n" {n!r} "$(f22_registry_evals_for_job {n!r})"' for n in sorted({r["name"] for r in rows}))
    out = subprocess.run(["bash", "-c", script], cwd=src.parent, capture_output=True, text=True, timeout=60).stdout
    table = {ln.split("\t")[0]: ln.split("\t")[1].split() for ln in out.splitlines() if "\t" in ln}
    checked = 0
    for r in rows:
        if not r["registers"] or not table.get(r["name"]):
            continue                                                       # a kind the name table does not know (padded, mmlu, gsm8k) — the plan is explicit
        if r["name"].startswith("f22s-"):
            assert set(r["evals"]) <= set(table[r["name"]]), (r["name"], r["evals"], table[r["name"]])   # the sampled kind covers statics AND controls; a unit names its phase's subset
        else:
            assert r["evals"] == table[r["name"]], (r["name"], r["evals"], table[r["name"]])
        checked += 1
    assert checked >= 6


def test_a_single_family_call_is_checked_against_the_units_submission_family_not_its_eval_family_and_the_climb_runs_after_the_store(tmp_path):
    """The BoN 512 store is EVAL family `bon` (the gate's, `family`) but is PLACED by `--family static` (its
    `stage_family`). The inventory-agreement check admitted a frozen unit by EITHER family, so a `--family bon` call wanted the store
    the bon plan does not carry and refused the whole family as "frozen but not planned": the wave could only be submitted with static
    and bon in one call, and a per-family re-run was impossible. Membership is the submission family alone; the same check still
    refuses a wave at another sha (the end-to-end test above)."""
    w = World(tmp_path)
    r = w.stage("05_stage_roster.sh", "--model", MODEL, "--step", "freeze")
    assert r.returncode == 0, r.stdout + r.stderr
    store = next(u for u in w.inventory()["units"] if u["logical_work_id"].endswith("/bon/store512"))
    assert store["family"] == "bon" and store["stage_family"] == "static"
    # the climb runs AFTER the 512 store the static call writes: bon before static is refused by name — no inventory complaint
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--view", "battery", "--family", "bon")
    assert r.returncode != 0 and "runs after" in r.stderr and "was never submitted" in r.stderr and "frozen but not planned" not in r.stderr, r.stderr[-3000:]
    assert not re.findall(r"^submitted ", r.stdout, re.M)
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--view", "battery", "--family", "static")
    assert r.returncode == 0 and "frozen but not planned" not in r.stderr, r.stdout[-3000:] + r.stderr[-3000:]
    assert sorted(re.findall(r"^submitted \S+ \S+ (\S+)$", r.stdout, re.M)) == ["bon.store512", "static.t0", "static.t1"], r.stdout
    lab, store_job = re.search(r"^submitted (\S+) (\S+) bon\.store512$", r.stdout, re.M).groups()
    r = w.stage("06_eval_waves.sh", "--model", MODEL, "--view", "battery", "--family", "bon")
    assert r.returncode == 0 and "frozen but not planned" not in r.stderr, r.stdout[-3000:] + r.stderr[-3000:]
    assert re.findall(r"^submitted \S+ \S+ (\S+)$", r.stdout, re.M) == ["bon.climb"], r.stdout
    lab2, climb_job = re.search(r"^submitted (\S+) (\S+) bon\.climb$", r.stdout, re.M).groups()
    assert lab2 == lab and f"afterok:{store_job}" in r.stdout                              # placed on the store's cluster, ordered after its job
    assert f"--dependency=afterok:{store_job}" in (w.fake[lab] / "jobs" / f"{climb_job}.argv").read_text()
