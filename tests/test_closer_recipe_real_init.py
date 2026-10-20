"""closer-fix-5: the reclose recipe is written from the job's SHELL variables, never from os.environ.

f22_init assigns SHARED, WAVE_ROOT, WAVE_SHA, REPO_DIR and CLUSTER as plain shell variables and the sbatches set TAG / NSHARDS the same way;
the recipe writer's Python read them through os.environ, so every real recipe carried wave_root='' and pending='/registry/pending/<job>.json'
(and code_sha=''): the funnel's STOP fence was skipped and the reclose job died on the bogus path before its trap — a receipt-less death the
durable cap counted. The fixtures never saw it because they EXPORTED what production does not. Here the REAL f22_init runs under the production
portability prefixes only (SISTEER_SHARED_ROOT / _CLUSTER_LABEL / _HF_HOME ride the sbatch line and ARE exported), nothing else exported, over a
real git checkout; a refusing closer writes the recipe; its fields are the shell's; with the wave's STOP standing the real funnel refuses the
reclose by the fence; a production-shaped recipe (no wave root, relative pending) is refused by name by the funnel and, in the reclose job, in
PREFLIGHT — recorded on the recipe (rc 1, `preflight`) so the funnel stops instead of spending the cap."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, job_ids  # noqa: E402
from test_closer_interpreter import UNTRUSTED  # noqa: E402
from test_closer_refusal_outcome import _need_closer_history, _reclose_run, _tree_with_closer, _world  # noqa: E402
from test_submit_resubmit_proof import MMLU  # noqa: E402

COMMON = REPO / "scripts/sdfbase/full22/common.sh"


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.email=t@example.invalid", "-c", "user.name=t", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def _real_init_closer_refusal(tmp: Path, *, shared: Path | None = None, job_id: str = "1001", stop: bool = True) -> dict:
    """The REAL f22_init (the production portability prefixes exported, nothing else), a real git checkout at the wave sha, the wave's STOP
    standing, then the real f22_registry_close_stores with a recording `python3` whose closer refuses: the recipe the eval job leaves.
    `shared` = the job's shared root (SISTEER_SHARED_ROOT) — the funnel world's cluster root when the funnel is to read the recipe in place."""
    tmp.mkdir(parents=True, exist_ok=True); shared = shared or tmp / "shared"; (shared / "registry/pending").mkdir(parents=True, exist_ok=True); (tmp / "hf").mkdir()
    pend = shared / "registry/pending" / f"{job_id}.json"
    pend.exists() or pend.write_text('{"runs": [{"run_id": "run-a", "campaign": "sdfbase_full22"}]}')
    co = tmp / "co"; (co / "src/sisteer").mkdir(parents=True); (co / "src/sisteer/__init__.py").write_text(""); _tree_with_closer(co)   # closer-fix-3: the reclose runs from this tree
    (co / "configs").mkdir(); (co / "configs/sdfbase_full22_roster.yaml").write_text("{}\n")
    _git(co, "init", "-q"); _git(co, "add", "-A"); _git(co, "commit", "-q", "-m", "synthetic wave checkout"); sha = _git(co, "rev-parse", "HEAD")
    (shared / "sisteer-eval-repo").mkdir(exist_ok=True); co.rename(shared / "sisteer-eval-repo" / sha); co = shared / "sisteer-eval-repo" / sha     # the funnel World's mirror dir may exist
    bin_dir = tmp / "bin"; bin_dir.mkdir(); log = tmp / "hook.log"
    (bin_dir / "python3").write_text(textwrap.dedent(f"""\
        #!/bin/bash
        printf '%s\\n' "PYTHON $*" >> "$HOOK_LOG"
        case "$*" in *registry_close_stores.py*) echo "{UNTRUSTED}" >&2; exit 2;; esac
        case "$1 ${{2:-}}" in "-S -"|"-S -c") exec {sys.executable} "$@";; esac
        exit 0
        """)); (bin_dir / "python3").chmod(0o755)
    script = textwrap.dedent(f"""
        set -u
        export SISTEER_SHARED_ROOT={shared} SISTEER_CLUSTER_LABEL=cluster_a SISTEER_EVAL_REPO_ROOT={shared}/sisteer-eval-repo SISTEER_HF_HOME={tmp}/hf SISTEER_REGISTRY_SPOOL={shared}/registry/records
        export SLURM_JOB_ID={job_id} SLURM_JOB_NAME=mmluF-m-s1 SLURMD_NODENAME=TESTHOST HOOK_LOG={log} F22_SERVING_PATH=lora_native USER=TESTUSER
        unset F22_PY
        source {COMMON}
        f22_registry_state(){{ printf '%s\\n' "STATE $*" >> "$HOOK_LOG"; }}
        f22_init {sha}
        TAG=m; NSHARDS=9                                   # as common.sh:526 / target_attacks.sbatch assign them: PLAIN variables
        {'mkdir -p "$WAVE_ROOT"; : > "$WAVE_ROOT/STOP"' if stop else ''}
        f22_registry_close_stores; echo "RC=$?"
        echo "SHELL SHARED=$SHARED WAVE_ROOT=$WAVE_ROOT REPO_DIR=$REPO_DIR CLUSTER=$CLUSTER WAVE_SHA=$WAVE_SHA"
        echo "EXPORTED $(env | grep -E '^(SHARED|WAVE_ROOT|WAVE_SHA|REPO_DIR|CLUSTER|TAG|NSHARDS)=' | tr '\\n' ' ')"
    """)
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp), "HOOK_LOG": str(log)}
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=120)
    recipe_p = shared / "registry/pending" / f"{job_id}.closer_refused.json"
    return {"rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr, "sha": sha, "shared": shared, "checkout": co, "log": log.read_text().splitlines() if log.exists() else [],
            "recipe_path": recipe_p, "recipe": json.loads(recipe_p.read_text()) if recipe_p.exists() else None}


def test_the_real_f22_init_leaves_its_variables_unexported_and_the_recipe_still_carries_them_and_the_stop_fence_then_refuses_the_reclose(tmp_path):
    w = _world(tmp_path / "funnel", recipe=False)                                                                    # the funnel world first: the job's shared root is ITS cluster root
    out = _real_init_closer_refusal(tmp_path / "real", shared=w.shared["cluster_a"])
    assert out["rc"] == 0 and "RC=2" in out["stdout"], out                                                          # f22_init ran; the closer refused (rc 2)
    shell = [l for l in out["stdout"].splitlines() if l.startswith("SHELL ")][0]
    exported = [l for l in out["stdout"].splitlines() if l.startswith("EXPORTED")][0]
    shared, sha = out["shared"], out["sha"]
    assert f"SHARED={shared} WAVE_ROOT={shared}/exp/sdfbase_full22_native REPO_DIR={shared}/sisteer-eval-repo/{sha} CLUSTER=cluster_a WAVE_SHA={sha}" in shell, shell
    assert exported.strip() == "EXPORTED", exported                        # the premise: f22_init exports NONE of them — what the writer must not read from os.environ
    rec = out["recipe"]; assert rec is not None, (out["stderr"], out["stdout"])
    assert rec["wave_root"] == f"{shared}/exp/sdfbase_full22_native" and rec["pending"] == f"{shared}/registry/pending/1001.json" and rec["code_sha"] == sha \
        and rec["cluster"] == "cluster_a" and rec["repo_dir"] == f"{shared}/sisteer-eval-repo/{sha}" and rec["tag"] == "m" and rec["nshards"] == "9" \
        and rec["spool"] == f"{shared}/registry/records" and rec["job_id"] == "1001" and rec["job_name"] == "mmluF-m-s1" and rec["why"] == UNTRUSTED[:300] \
        and rec["figs_dir"] == f"{shared}/exp/figs/mmluF-m-s1" and rec["recloses"] == [], rec
    # the REAL funnel over that recipe, in place, with the wave's STOP standing: refused by the fence, nothing submitted
    assert out["recipe_path"] == w.shared["cluster_a"] / "registry/pending/1001.closer_refused.json"
    r = w.run(*MMLU, "--resubmit")
    assert r.returncode == 3 and f"the wave's STOP stands at {shared}/exp/sdfbase_full22_native/STOP" in r.stderr and "nothing submitted" in r.stderr and job_ids(w.fake["cluster_a"]) == [1001], r.stdout + r.stderr
    assert "reclose" not in [e["event"] for e in w.ledger()]


def test_with_the_stop_cleared_the_same_recipe_submits_one_reclose_at_its_sha(tmp_path):
    """The real recipe's other half: the checkout `_real_init_closer_refusal` commits is a child of this repo's HEAD (the fixture's alternates), so
    the reclose submits only where this repository carries closer-fix-3 -- the funnel refuses the reclose by name everywhere else, and the
    recipe fields and the STOP fence above are asserted there regardless."""
    _need_closer_history()
    w = _world(tmp_path / "funnel", recipe=False)
    out = _real_init_closer_refusal(tmp_path / "real", shared=w.shared["cluster_a"])
    rec, sha = out["recipe"], out["sha"]; assert rec is not None and rec["code_sha"] == sha, (out["stderr"], out["stdout"])
    # the STOP cleared: the same recipe is usable — one reclose at the recipe's sha
    (Path(rec["wave_root"]) / "STOP").unlink()
    r = w.run(*MMLU, "--resubmit")
    assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout and f"reclose_stores.sbatch 1001 {sha}" in (w.fake["cluster_a"] / "jobs" / "1002.argv").read_text(), r.stdout + r.stderr


def test_a_job_whose_wave_root_is_unbound_writes_no_recipe_and_says_so(tmp_path):
    """The writer's own fail-closed branch: WAVE_ROOT unbound (a job that never bound its serving path) — nothing written, named on stderr; the
    funnel then refuses the reclose for want of a recipe (its existing 'left no recipe' refusal)."""
    out = _real_init_closer_refusal(tmp_path / "unbound", stop=False)
    assert out["rc"] == 0 and out["recipe"] is not None                                                            # the baseline: bound → written
    shared = tmp_path / "unb2" / "shared"; (shared / "registry/pending").mkdir(parents=True); (shared / "registry/pending/7.json").write_text('{"runs": [{"run_id": "r"}]}')
    bin_dir = tmp_path / "unb2" / "bin"; bin_dir.mkdir(); log = tmp_path / "unb2" / "hook.log"
    (bin_dir / "python3").write_text(textwrap.dedent(f"""\
        #!/bin/bash
        printf '%s\\n' "PYTHON $*" >> "$HOOK_LOG"
        case "$*" in *registry_close_stores.py*) echo "{UNTRUSTED}" >&2; exit 2;; esac
        case "$1 ${{2:-}}" in "-S -"|"-S -c") exec {sys.executable} "$@";; esac
        exit 0
        """)); (bin_dir / "python3").chmod(0o755)
    script = textwrap.dedent(f"""
        set -u
        export SISTEER_REGISTRY_SPOOL={shared}/registry/records SLURM_JOB_ID=7 SLURM_JOB_NAME=f22-x HOOK_LOG={log} FIGS_DIR={tmp_path}/unb2/figs
        mkdir -p "$FIGS_DIR"
        SHARED={shared}; WAVE_SHA={'c' * 40}; REPO_DIR={REPO}; CLUSTER=cluster_a
        source {COMMON}
        f22_registry_state(){{ printf '%s\\n' "STATE $*" >> "$HOOK_LOG"; }}
        unset WAVE_ROOT
        f22_registry_close_stores; echo "RC=$?"
    """)
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={"PATH": f"{bin_dir}:{os.environ['PATH']}", "HOME": str(tmp_path), "HOOK_LOG": str(log)}, timeout=120)
    assert "RC=2" in r.stdout and "[closer] NOT writing the reclose recipe at" in r.stderr and "WAVE_ROOT is unbound in this job (F22_WAVE_ROOT=unset)" in r.stderr, r.stdout + r.stderr
    assert not (shared / "registry/pending/7.closer_refused.json").exists()


def test_a_production_shaped_recipe_without_its_wave_root_is_refused_by_name_by_the_funnel_and_in_the_reclose_jobs_preflight(tmp_path):
    # the funnel: what every real recipe looked like before closer-fix-5 — wave_root '' and pending '/registry/pending/<job>.json'
    w = _world(tmp_path / "bad")
    p = w.shared["cluster_a"] / "registry/pending/1001.closer_refused.json"; d = json.loads(p.read_text()); d["wave_root"] = ""; d["pending"] = "/registry/pending/1001.json"; p.write_text(json.dumps(d))
    r = w.run(*MMLU, "--resubmit")
    assert r.returncode == 3 and "is unusable" in r.stderr and "wave_root (empty) is not an absolute path" in r.stderr \
        and f"pending (/registry/pending/1001.json) is not this job's pending file {w.shared['cluster_a']}/registry/pending/1001.json" in r.stderr and "nothing submitted" in r.stderr \
        and job_ids(w.fake["cluster_a"]) == [1001] and "reclose" not in [e["event"] for e in w.ledger()], r.stdout + r.stderr
    # pending alone wrong (an absolute path that is not this job's file under the cluster's shared root): named alone
    d["wave_root"] = str(w.shared["cluster_a"] / "exp/sdfbase_full22_native"); d["pending"] = "/elsewhere/registry/pending/1001.json"; p.write_text(json.dumps(d))
    r = w.run(*MMLU, "--resubmit")
    assert r.returncode == 3 and "is unusable — pending (/elsewhere/registry/pending/1001.json) is not this job's pending file" in r.stderr and "wave_root" not in r.stderr.split("is unusable", 1)[1].split("(a closer", 1)[0], r.stderr
    # the reclose job itself over such a recipe: refused in PREFLIGHT — recorded (rc 1, `preflight`), f22_init never ran, no closer run
    out = _reclose_run(tmp_path / "pre", closer_rc=0, wave_root="", pending_path="/registry/pending/1001.json")
    assert out["rc"] == 1 and "carries no absolute wave_root ('')" in out["stderr"] and not [l for l in out["log"] if l.startswith("INIT ") or "registry_close_stores.py" in l], out
    assert [(x["job_id"], x["rc"], x.get("preflight"), x.get("stopped")) for x in out["recipe"]["recloses"]] == [("5005", 1, True, None)] \
        and out["recipe"]["recloses"][0]["why"].startswith("reclose 1001: the recipe carries no absolute wave_root"), out["recipe"]
    out = _reclose_run(tmp_path / "pre2", closer_rc=0, pending_path="/registry/pending/1001.json")
    assert out["rc"] == 1 and "pending path ('/registry/pending/1001.json') is not this job's pending file" in out["stderr"] and out["recipe"]["recloses"][0].get("preflight") is True, out
    # the funnel on a recipe whose last receipt is a preflight refusal: stops by name — the cap is not spent on a refusal that repeats itself
    w = _world(tmp_path / "pf", recloses=[{"job_id": "71", "rc": 1, "why": "reclose 1001: the recipe's pending file /x is gone", "utc": "2026-01-01T00:30:00Z", "preflight": True}])
    r = w.run(*MMLU, "--resubmit")
    assert r.returncode == 3 and "reclose job 71 of job 1001 refused in PREFLIGHT" in r.stderr and "pending file /x is gone" in r.stderr and job_ids(w.fake["cluster_a"]) == [1001], r.stdout + r.stderr
