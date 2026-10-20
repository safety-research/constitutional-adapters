"""closer-fix-2: a refused store closer is the job's OUTCOME, and the funnel's recovery is a RECLOSE, never a
re-run of the eval.

Before: the closer refused on a completed eval job, f22_result had already written the clean RESULT line, the trap returned 0, Slurm said
COMPLETED, the `running` note sat outside the grading worklist (pending.py requires generated+) and nothing recovered — the silent shape of
the 36 stuck runs, now with a label. Now `f22_registry_close_stores` RETURNS the closer's rc and writes a RECIPE beside the pending file (every
closer argument from the job's own env); `f22_registry_result` carries the rc past the final push; `f22_result` leaves with exit 75 (sacct:
FAILED 75:0) after a second RESULT line, CLOSER_REFUSED. The funnel reads the exit code per task, answers `reclose` on 75 (the stores are
complete), and submits scripts/sdfbase/full22/reclose_stores.sbatch — CPU-only, at the recipe's sha, confirming the driver venv by the same
f22_driver_venv path — once: skipped while one is live, `closed` when the last filed, refused past 3 or when the same refusal survived a
reclose. 06's --resubmit walk reaches the same line, so the recovery is automatic. Everything executed: the shell with the real function bodies
and recording stubs, the real funnel over the fake Slurm (exit codes from `<fake>/jobs/<id>.exit`), the real reclose sbatch under bash with a
stub common.sh."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, job_ids, make_registry, set_job_state  # noqa: E402
from seedrep_history_fixtures import pinned_commit_or_skip  # noqa: E402
from test_closer_interpreter import UNTRUSTED, _run as _run_shell  # noqa: E402
from test_submit_resubmit_proof import MMLU, _bind_registered, _binding, _pend  # noqa: E402
from test_submit_twin import SHA, World  # noqa: E402

COMMON = REPO / "scripts/sdfbase/full22/common.sh"
RECLOSE = REPO / "scripts/sdfbase/full22/reclose_stores.sbatch"


# ------------------------------------------------------------------------------------------------------------ the shell: exit 75 + the recipe
def test_a_refused_closer_on_a_clean_exit_pushes_then_leaves_with_exit_75_a_closer_refused_result_line_and_a_recipe(tmp_path):
    out = _run_shell(tmp_path, "f22_result 'attacks t g rc=0'", f22_py=None, closer_rc=2)
    assert out["rc"] == 75 and "RC=" not in out["stdout"], out                                     # the trap's exit IS the job's outcome; nothing after it runs
    kinds = [("closer" if "registry_close_stores.py" in l else "notice" if "emit_cli state" in l else "recipe" if l == "PYTHON -S -" else "push" if "emit_cli push" in l else l) for l in out["log"]]
    # the recipe is written and the final push still happens before the exit; no `notice` — the stub closer left no refused-run list,
    # so the exit path states nothing (the old `--pending-dir … --job` notice restated `running` over the job's whole run set) and says so
    assert kinds == ["closer", "recipe", "push"], out["log"]
    assert "no refused-run list at" in out["stdout"] and "stating nothing (D2100 (b)" in out["stdout"], out["stdout"]
    lines = out["result"].splitlines()
    assert len(lines) == 2 and " attacks t g rc=0" in lines[0] and "CLOSER_REFUSED rc=2 interpreter=python3" in lines[1], out["result"]   # the RESULT surface: the job's own line, then the refusal
    assert "[closer] REFUSED rc=2 under python3 (F22_PY unset" in out["stderr"] and "leaves with exit 75" in out["stdout"] \
        and "CLOSER_REFUSED rc=2 exit=75 recovery=submit_twin.py reclose --job 4242" in out["stdout"], out                               # both tails carry it, with the recovery named
    recipe = json.loads((tmp_path / "shared" / "registry" / "pending" / "4242.closer_refused.json").read_text())
    assert recipe["job_id"] == "4242" and recipe["job_name"] == "f22-att-test" and recipe["rc"] == 2 and recipe["interpreter"] == "python3" and recipe["why"] == UNTRUSTED[:300]
    assert recipe["pending"].endswith("/registry/pending/4242.json") and recipe["spool"].endswith("/registry/records") and recipe["code_sha"] == "a" * 40 \
        and recipe["tag"] == "t" and recipe["nshards"] == "9" and recipe["wave_root"].endswith("/waves/g") and recipe["repo_dir"] == str(REPO) and recipe["recloses"] == []
    # a clean closer: no exit, no recipe, the RESULT has its one line
    out = _run_shell(tmp_path, "f22_result 'attacks t g rc=0'", f22_py="venvpy", closer_rc=2)
    assert out["rc"] == 0 and "RC=0" in out["stdout"] and len(out["result"].splitlines()) == 1 and not (tmp_path / "shared" / "registry" / "pending" / "4242.closer_refused.json").exists(), out
    # a job whose own rc is non-zero files failed and never runs the closer (unchanged)
    out = _run_shell(tmp_path, "f22_result 'attacks t g rc=3'", f22_py=None, closer_rc=2)
    assert out["rc"] == 0 and "RC=0" in out["stdout"] and out["log"] == ["STATE failed FAILED:rc=3 final"], out


# ------------------------------------------------------------------------------------------------------------ the funnel: exit 75 -> reclose
#: the real closer of THIS tree: what a fixture tree carries, so the funnel's marker greps read the real file; the lineage decision is the
#: tree's COMMIT (closer-fix-4), never the file's content
REAL_CLOSER = REPO / "scripts/sdfbase/full22/registry_close_stores.py"
SBATCH_STUB = '#!/bin/bash\nORIG="${1:?original job id}"; WAVE_SHA="${2:?wave sha}"; CLOSER_SHA="${3:-}"\n'               # reclose-at-tree: the closer's tree rides positional 3
SBATCH_UNPLUMBED = '#!/bin/bash\nORIG="${1:?original job id}"; WAVE_SHA="${2:?wave sha}"\n'                                # the script before reclose-at-tree (the old job trees)


_CLOSER_MIN_SHA = re.search(r'CLOSER_MIN_SHA = "([0-9a-f]{40})"', (LIB / "submit_twin.py").read_text()).group(1)


def _need_closer_history() -> None:
    """A reclose that PASSES the funnel's closer-tree guard needs this repository to carry closer-fix-3 (CLOSER_MIN_SHA): the fixture tree
    (`_tree_with_closer`) shares this repo's objects and sits at its HEAD, so the guard's `merge-base --is-ancestor` reads this repo's
    history, and a clone without that commit is refused by name (the designed refusal, pinned everywhere by test_closer_tree_ancestry's
    own-root fixtures). seedrep_history_fixtures' exact-absence rule: a skip only for a source-only tree, a shallow clone or the
    published repository (its root marker borne out); a full checkout lacking the commit FAILS naming the branch. Called by the tests
    whose assertions need a SUBMITTED reclose -- never by `_world`, which also serves the pure-refusal tests that run everywhere."""
    pinned_commit_or_skip(_CLOSER_MIN_SHA, "main", "a reclose that passes the closer-tree guard (closer-fix-3's lineage)", repo=REPO, anchor=Path(__file__))


def _git_tree(co: Path, head: str | None = None) -> str:
    """closer-fix-4: make `co` a GIT checkout whose HEAD is `head` (default: this repo's HEAD) — a checkout-less repository sharing
    this repo's objects (an alternates file), so `git rev-parse HEAD` and `git merge-base --is-ancestor` read real commits while the files under
    it stay the test's files. A `co` that is already a repository of its own (one with its own root commit, no alternates) is left as it is and
    only its HEAD is pointed: that is the shape of a clone without the development history, which the funnel refuses by name. Returns the
    HEAD sha."""
    co.mkdir(parents=True, exist_ok=True)
    if not (co / ".git").exists():
        subprocess.run(["git", "init", "-q", str(co)], check=True, capture_output=True)
        common = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--git-common-dir"], capture_output=True, text=True, check=True).stdout.strip()
        common_dir = Path(common) if Path(common).is_absolute() else (REPO / common).resolve()
        alt = co / ".git" / "objects" / "info"; alt.mkdir(parents=True, exist_ok=True); (alt / "alternates").write_text(f"{common_dir / 'objects'}\n")
    h = head or subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    subprocess.run(["git", "-C", str(co), "update-ref", "HEAD", h], check=True, capture_output=True)
    return h


def _tree_with_closer(co: Path, *, fix2: bool = True, plumbed: bool = True, head: str | None = None) -> Path:
    """closer-fix-3: a mirrored tree the funnel runs the closer from — the closer script (THIS tree's real closer; with `fix2` off, a closer
    before closer-fix-2), src/sisteer, and a reclose_stores.sbatch that takes the closer's tree on positional 3 (unless `plumbed` is off).
    closer-fix-4: the tree is a git checkout at `head` (default: this repo's HEAD) — the funnel reads the ACTUAL commit and requires
    ancestry from closer-fix-3; a clone that does not carry that commit is refused by name."""
    (co / "scripts/sdfbase/full22").mkdir(parents=True, exist_ok=True); (co / "src/sisteer").mkdir(parents=True, exist_ok=True)
    _git_tree(co, head)
    closer = REAL_CLOSER.read_bytes() if fix2 else b"# a closer before closer-fix-2 (synthetic)\n"
    (co / "scripts/sdfbase/full22/registry_close_stores.py").write_bytes(closer)       # bytes, so the fix-3 closer's blob id is this tree's
    (co / "scripts/sdfbase/full22/reclose_stores.sbatch").write_text(SBATCH_STUB if plumbed else SBATCH_UNPLUMBED)
    return co


def _world(tmp_path, *, exit_code="75:0", recipe=True, recloses=None, why=UNTRUSTED):
    tmp_path.mkdir(parents=True, exist_ok=True)
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    _tree_with_closer(w.shared["cluster_a"] / "sisteer-eval-repo" / SHA)              # closer-fix-3: the job's tree carries the closer the reclose runs
    assert w.run(*MMLU).returncode == 0
    _pend(w, "cluster_a", "1001", ["run-a"])
    make_registry(w.tmp / "registry", runs=[{"run_id": "run-a", "lifecycle_state": "running", "cluster_job_id": "1001"}])
    _bind_registered(w, "m/s1/mmlu/mmlu", ["run-a"])
    set_job_state(w.fake["cluster_a"], "1001", "FAILED", end="2026-01-01T00:10:00")
    (w.fake["cluster_a"] / "jobs" / "1001.exit").write_text(exit_code + "\n")
    w.spool("cluster_a"); w.ingested("cluster_a", through="2026-01-01T00:20:00")
    if recipe:
        shared = w.shared["cluster_a"]
        (shared / "registry/pending/1001.closer_refused.json").write_text(json.dumps({
            "job_id": "1001", "job_name": "mmluF-m-s1", "rc": 2, "interpreter": "python3", "why": why[:300], "utc": "2026-01-01T00:09:00Z",
            "pending": str(shared / "registry/pending/1001.json"), "spool": str(shared / "registry/records"), "push_script": "", "code_sha": SHA,
            "wave_root": str(shared / "exp/sdfbase_full22_native"), "tag": "m", "nshards": "9", "repo_dir": str(shared / "sisteer-eval-repo" / SHA),
            "cluster": "cluster_a", "figs_dir": "", "recloses": recloses or []}))
    return w


def _argv(w, job):
    return (w.fake["cluster_a"] / "jobs" / f"{job}.argv").read_text()


def test_exit_75_on_a_registered_unit_submits_one_cpu_only_reclose_at_the_recipes_sha_and_skips_while_it_is_live(tmp_path):
    _need_closer_history()
    w = _world(tmp_path)
    r = w.run(*MMLU, "--resubmit"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout and "exit 75" in out and "store closer refused" in out and UNTRUSTED[:60] in out, out
    assert "reclose 1/3" in r.stdout and "no GPU, no re-run of the eval" in r.stdout, r.stdout
    argv = _argv(w, "1002")
    assert f"scripts/sdfbase/full22/reclose_stores.sbatch 1001 {SHA}" in argv and "-J f22-reclose-1001" in argv and "--cpus-per-task=4" in argv and "--mem=16G" in argv \
        and "--time=00:45:00" in argv and "--gres" not in argv and "--array" not in argv, argv
    ev = [e for e in w.ledger() if e["event"] == "reclose"]
    assert len(ev) == 1 and ev[0]["job_id"] == "1001" and ev[0]["reclose_job_id"] == "1002" and ev[0]["attempt"] == 1 and ev[0]["code_sha"] == SHA and ev[0]["why"] == UNTRUSTED[:300]
    b = _binding(w, "m__s1__mmlu__mmlu")
    assert b["m__s1__mmlu__mmlu.registered.json"]["job_id"] == "1001" and "m__s1__mmlu__mmlu.transition.json" not in b and not any(".retired." in k for k in b)   # the binding stands: nothing re-run
    # the reclose is live (start_on_submit -> RUNNING): the walk skips the unit, nothing new is launched
    r = w.run(*MMLU, "--resubmit", "--walk"); assert r.returncode == 0 and "reclose f22-reclose-1001 1002 is RUNNING" in r.stdout and "in flight" in r.stdout, r.stdout + r.stderr
    assert job_ids(w.fake["cluster_a"]) == [1001, 1002]
    # the reclose filed (its outcome on the recipe): closed — the gate reads the runs
    set_job_state(w.fake["cluster_a"], "1002", "COMPLETED", end="2026-01-01T00:30:00")
    p = w.shared["cluster_a"] / "registry/pending/1001.closer_refused.json"; d = json.loads(p.read_text()); d["recloses"] = [{"job_id": "1002", "rc": 0, "why": "", "utc": "2026-01-01T00:30:00Z"}]; p.write_text(json.dumps(d))
    r = w.run(*MMLU, "--resubmit"); assert r.returncode == 0 and "closed by reclose job 1002" in (r.stdout + r.stderr) and job_ids(w.fake["cluster_a"]) == [1001, 1002], r.stdout + r.stderr


def test_the_same_refusal_surviving_a_reclose_the_cap_and_a_missing_recipe_refuse_by_name(tmp_path):
    """The refusals the funnel raises BEFORE the closer-tree guard: they run everywhere, the published repository included (where the guard
    refuses every reclose that would otherwise submit -- `_need_closer_history`); the submitting cases are the next test's."""
    # the same refusal survived the first reclose: not resubmitted blind
    w = _world(tmp_path / "same", recloses=[{"job_id": "77", "rc": 2, "why": UNTRUSTED[:300], "utc": "2026-01-01T00:30:00Z"}])
    set_job_state(w.fake["cluster_a"], "77", "FAILED", end="2026-01-01T00:31:00") if (w.fake["cluster_a"] / "jobs" / "77.state").exists() else None
    r = w.run(*MMLU, "--resubmit"); assert r.returncode == 3 and "SURVIVED reclose job 77 unchanged" in r.stderr and job_ids(w.fake["cluster_a"]) == [1001], r.stdout + r.stderr
    # three recloses already: the cap
    w = _world(tmp_path / "cap", recloses=[{"job_id": "71", "rc": 2, "why": "REFUSED: shard02 incomplete", "utc": "2026-01-01T00:30:00Z"},
                                           {"job_id": "72", "rc": 2, "why": "REFUSED: shard03 incomplete", "utc": "2026-01-01T00:40:00Z"},
                                           {"job_id": "73", "rc": 2, "why": "REFUSED: shard04 incomplete", "utc": "2026-01-01T00:50:00Z"}])
    r = w.run(*MMLU, "--resubmit"); assert r.returncode == 3 and "already had 3 reclose(s) (cap 3)" in r.stderr and job_ids(w.fake["cluster_a"]) == [1001], r.stdout + r.stderr
    # exit 75 without a recipe: refused by name (close by hand), nothing submitted, nothing re-run
    w = _world(tmp_path / "norecipe", recipe=False)
    r = w.run(*MMLU, "--resubmit"); assert r.returncode == 3 and "left no recipe at" in r.stderr and "closer_refused.json" in r.stderr and job_ids(w.fake["cluster_a"]) == [1001], r.stdout + r.stderr
    # a plain FAILED job (exit 1:0) takes the fenced proof as before — never a reclose
    w = _world(tmp_path / "plain", exit_code="1:0")
    r = w.run(*MMLU, "--resubmit"); assert "RECLOSE" not in (r.stdout + r.stderr) and "reclose" not in [e["event"] for e in w.ledger()], r.stdout + r.stderr
    # the subcommand over a job no attempt names
    w = _world(tmp_path / "sub")
    r = w.run("reclose", "--job", "9999"); assert r.returncode == 2 and "no attempt with job 9999" in r.stderr and job_ids(w.fake["cluster_a"]) == [1001]


def test_a_different_refusal_after_a_reclose_is_one_more_reclose_and_the_subcommand_submits(tmp_path):
    _need_closer_history()
    # a different refusal after a reclose: one more reclose (2/3) — the cap is the previous test's
    w = _world(tmp_path / "two", recloses=[{"job_id": "71", "rc": 2, "why": "REFUSED: shard02 incomplete", "utc": "2026-01-01T00:30:00Z"}])
    r = w.run(*MMLU, "--resubmit"); assert r.returncode == 0 and "reclose 2/3" in r.stdout and job_ids(w.fake["cluster_a"]) == [1001, 1002], r.stdout + r.stderr
    # the subcommand: `reclose --job J` finds the attempt in the ledger and submits the same job
    w = _world(tmp_path / "sub")
    r = w.run("reclose", "--job", "1001"); assert r.returncode == 0 and "RECLOSE cluster_a 1002 mmlu" in r.stdout, r.stdout + r.stderr
    assert f"reclose_stores.sbatch 1001 {SHA}" in _argv(w, "1002")
    r = w.run("--dry-run", "reclose", "--job", "1001"); assert r.returncode == 0 and "DRY-RUN: reclose" in r.stdout and job_ids(w.fake["cluster_a"]) == [1001, 1002]


# ------------------------------------------------------------------------------------------------------------ the reclose job itself
def _reclose_run(tmp: Path, *, closer_rc: int, recipe_sha: str = SHA, pending: bool = True, venv_ok: bool = True, stop: bool = False,
                 wave_root: str | None = None, pending_path: str | None = None) -> dict:
    """The REAL reclose_stores.sbatch under bash: SISTEER_SHARED_ROOT points at a synthetic shared root whose mirrored checkout carries a stub
    common.sh (f22_init / f22_driver_venv / f22_result / f22_log / f22_fatal / f22_attempt_wait recorded, nothing else) and a recording
    closer; the recipe and the pending file sit where the eval job left them."""
    shared = tmp / "shared"; co = shared / "sisteer-eval-repo" / SHA / "scripts/sdfbase/full22"; co.mkdir(parents=True, exist_ok=True)
    (shared / "registry/pending").mkdir(parents=True, exist_ok=True); log = tmp / "hook.log"; log.unlink(missing_ok=True)
    pybin = tmp / "venvpy"; pybin.write_text("#!/bin/bash\nprintf '%s\\n' \"VENVPY $*\" >> \"$HOOK_LOG\"\n"
                                              f"case \"$*\" in *registry_close_stores.py*) {'echo' if closer_rc else ':'} 'REFUSED: shard02 incomplete' >&2; exit {closer_rc};; esac\nexit 0\n"); pybin.chmod(0o755)
    (co / "registry_close_stores.py").write_text("# stub\n")
    (co / "common.sh").write_text(textwrap.dedent(f"""\
        f22_log(){{ echo "[log] $*"; }}
        f22_fatal(){{ echo "[log] FATAL: $*" >&2; exit 1; }}
        f22_attempt_wait(){{ :; }}
        f22_stop_requested(){{ return {0 if stop else 1}; }}
        f22_init(){{ printf '%s\\n' "INIT sha=$1 F22_WAVE_ROOT=${{F22_WAVE_ROOT:-}} SPOOL=${{SISTEER_REGISTRY_SPOOL:-}} CELLS=${{#F22_LAUNCH_CELLS[@]}}" >> "$HOOK_LOG"; REPO_DIR={shared}/sisteer-eval-repo/$1; FIGS_DIR={tmp}/figs; mkdir -p "$FIGS_DIR"; }}
        f22_driver_venv(){{ printf '%s\\n' "VENV sha=${{WAVE_SHA:-}}" >> "$HOOK_LOG"; {"MPY=" + str(pybin) + '; F22_PY="${F22_PY:-$MPY}"' if venv_ok else 'echo "[log] FATAL: no driver venv" >&2; exit 1'}; }}
        f22_result(){{ printf '%s\\n' "RESULT $*" >> "$HOOK_LOG"; }}
        """))
    pend = shared / "registry/pending/1001.json"
    if pending:
        pend.write_text(json.dumps({"job_id": "1001", "runs": [{"run_id": "run-a"}]}))
    recipe = shared / "registry/pending/1001.closer_refused.json"
    recipe.write_text(json.dumps({"job_id": "1001", "job_name": "mmluF-m-s1", "rc": 2, "interpreter": "python3", "why": "REFUSED: UNTRUSTED", "utc": "2026-01-01T00:09:00Z",
                                  "pending": str(pend) if pending_path is None else pending_path, "spool": str(shared / "registry/records"), "push_script": "", "code_sha": recipe_sha,
                                  "wave_root": str(shared / "wave") if wave_root is None else wave_root,
                                  "tag": "m_s1", "nshards": "9", "repo_dir": str(co.parents[2]), "cluster": "cluster_a", "figs_dir": "", "recloses": []}))
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp), "HOOK_LOG": str(log), "SISTEER_SHARED_ROOT": str(shared), "SLURM_JOB_ID": "5005", "SLURM_JOB_NAME": "f22-reclose-1001", "USER": "TESTUSER"}
    before = recipe.read_bytes()
    r = subprocess.run(["bash", str(RECLOSE), "1001", SHA], capture_output=True, text=True, env=env, timeout=120)
    # closer-fix-7: the job's outcome is ITS OWN receipt file (<orig>.reclose.<job>.receipt.json); the recipe is never rewritten — the harness merges
    # the receipt files into `recipe["recloses"]` the way the funnel's closer_recipe does, and reports whether the recipe's bytes survived
    rec = json.loads(recipe.read_text())
    receipts = [json.loads(f.read_text()) for f in sorted((shared / "registry/pending").glob("1001.reclose.*.receipt.json"))]
    rec["recloses"] = list(rec.get("recloses") or []) + receipts
    return {"rc": r.returncode, "stdout": r.stdout, "stderr": r.stderr, "log": log.read_text().splitlines() if log.exists() else [], "recipe": rec,
            "receipt_files": [f.name for f in sorted((shared / "registry/pending").glob("1001.reclose.*.receipt.json"))], "recipe_unchanged": recipe.read_bytes() == before}


def test_the_reclose_job_runs_the_closer_from_the_recipe_under_the_driver_venv_and_records_its_outcome_on_the_recipe(tmp_path):
    out = _reclose_run(tmp_path / "ok", closer_rc=0)
    assert out["rc"] == 0, out
    init = [l for l in out["log"] if l.startswith("INIT ")][0]
    assert f"sha={SHA}" in init and "F22_WAVE_ROOT=" + str(tmp_path / "ok/shared/wave") in init and "SPOOL=" + str(tmp_path / "ok/shared/registry/records") in init and "CELLS=0" in init, init
    assert [l for l in out["log"] if l.startswith("VENV ")] == [f"VENV sha={SHA}"]                                                   # the SAME venv path, at the recipe's sha
    closer = [l for l in out["log"] if "registry_close_stores.py" in l]; assert len(closer) == 1 and closer[0].startswith("VENVPY "), out["log"]
    for frag in ("--pending " + str(tmp_path / "ok/shared/registry/pending/1001.json"), "--spool " + str(tmp_path / "ok/shared/registry/records"), f"--code-sha {SHA}",
                 "--wave-root " + str(tmp_path / "ok/shared/wave"), "--tag m_s1", "--job-name mmluF-m-s1", "--nshards 9"):
        assert frag in closer[0], (frag, closer[0])
    assert "--push-script" not in closer[0]
    assert out["recipe"]["recloses"] == [{"job_id": "5005", "rc": 0, "why": "", "utc": out["recipe"]["recloses"][0]["utc"], "job_sha": SHA, "closer_sha": SHA, "closer_extra_args": None}] \
        and re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", out["recipe"]["recloses"][0]["utc"])            # a plain reclose: the closer's tree IS the job's, no extra args (reclose-at-tree fields)
    assert "RESULT reclose orig=1001 rc=0" in out["log"] and "the closer FILED job 1001's stores" in out["stdout"], out
    # the closer refuses again: the reclose job's exit is the closer's rc and the recipe records the why (the funnel stops on a repeat)
    out = _reclose_run(tmp_path / "again", closer_rc=2)
    assert out["rc"] == 2 and out["recipe"]["recloses"][0]["rc"] == 2 and out["recipe"]["recloses"][0]["why"] == "REFUSED: shard02 incomplete" and "REFUSED again" in out["stdout"] \
        and "RESULT reclose orig=1001 rc=2" in out["log"], out
    # a recipe whose sha is not this job's, a missing pending file, and a venv that refuses: fatal by name, no closer run
    out = _reclose_run(tmp_path / "sha", closer_rc=0, recipe_sha="b" * 40)
    assert out["rc"] == 1 and "is not this job's" in out["stderr"] and not [l for l in out["log"] if "registry_close_stores.py" in l], out
    # closer-fix-5: a preflight refusal is RECORDED on the recipe (rc 1, `preflight`) — before, the job died here receipt-less and the cap counted it
    assert [(r["job_id"], r["rc"], r.get("preflight")) for r in out["recipe"]["recloses"]] == [("5005", 1, True)] and "is not this job's" in out["recipe"]["recloses"][0]["why"], out["recipe"]
    assert not [l for l in out["log"] if l.startswith("INIT ")], out["log"]                                                                   # f22_init never ran
    out = _reclose_run(tmp_path / "pend", closer_rc=0, pending=False)
    assert out["rc"] == 1 and "pending file" in out["stderr"] and "is gone" in out["stderr"], out
    assert [(r["rc"], r.get("preflight")) for r in out["recipe"]["recloses"]] == [(1, True)] and "is gone" in out["recipe"]["recloses"][0]["why"], out["recipe"]
    out = _reclose_run(tmp_path / "venv", closer_rc=0, venv_ok=False)
    assert out["rc"] == 1 and "no driver venv" in out["stderr"] and not [l for l in out["log"] if "registry_close_stores.py" in l], out   # never bare python


# ------------------------------------------------------------------------------------------------------------ the closer: another emitter's eval
def test_the_real_closer_skips_another_emitters_eval_without_a_refusal_and_still_refuses_a_layout_it_owns(tmp_path):
    """The Petri proxy 821356's EXIT-trap closer, correctly under the driver venv, refused all 21 Petri runs 'no store layout here' — under
    closer-fix-2 that would have made every Petri job exit 75 and trigger a pointless reclose. The Petri runs are the Petri drive's to file:
    the closer skips them with one informational line and rc 0; a layout it owns (a multi-turn run with no store) still refuses, rc 2."""
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
    import registry_close_stores as RC  # noqa: E402
    (tmp_path / "records").mkdir(); (tmp_path / "wave").mkdir()
    petri = {"run_id": "petri-run-1", "route": {"eval_id": "petri_pooled", "object": "none"}, "plan": {"campaign": "sdfbase_full22", "n_items_planned": 10, "instrument_id": "petri_pooled_x"}}
    mmlu = {"run_id": "mmlu-run-1", "route": {"eval_id": "mmlu", "object": "none"}, "plan": {"campaign": "sdfbase_full22", "n_items_planned": 200, "instrument_id": "mmlu_x"}}
    mt = {"run_id": "mt-run-1", "route": {"eval_id": "actorattack", "object": "none"}, "plan": {"campaign": "sdfbase_full22", "n_items_planned": 29, "instrument_id": "multiturn_actorattack_x", "extra": {}}}
    unknown = {"run_id": "odd-run-1", "route": {"eval_id": "nosuch_eval", "object": "none"}, "plan": {"campaign": "sdfbase_full22", "n_items_planned": 5, "instrument_id": "odd_x"}}
    def run(*runs):
        p = tmp_path / f"pending_{len(runs)}_{runs[0]['run_id']}.json"; p.write_text(json.dumps({"job_id": "1", "runs": list(runs)}))
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = RC.main(["--pending", str(p), "--spool", str(tmp_path / "records"), "--wave-root", str(tmp_path / "wave"), "--tag", "m", "--nshards", "9", "--code-sha", "a" * 40])
        return rc, buf.getvalue()
    rc, out = run(petri, mmlu)
    assert rc == 0 and "REFUSED" not in out and out.count("not this closer's layout") == 2 and "Petri drive's registry adapter" in out and "own harness" in out \
        and "0 refused of 2 registered run(s), 2 another emitter's (not a refusal)" in out, out
    rc, out = run(petri, mt)
    assert rc == 2 and "skip petri_pooled none petri-run-1…: not this closer's layout" in out and "REFUSED actorattack none mt-run-1" in out \
        and "1 refused of 2 registered run(s), 1 another emitter's" in out, out                                   # a layout it owns still refuses
    rc, out = run(unknown)
    assert rc == 2 and "REFUSED nosuch_eval none odd-run-1" in out and "no store layout here" in out and "(not a refusal)" not in out and "1 refused of 1" in out, out   # nobody's eval: loud
    assert RC.other_emitter_of("petri_pooled") and RC.other_emitter_of("petri_full22") and RC.other_emitter_of("mmlu") and RC.other_emitter_of("gsm8k_paired200") \
        and RC.other_emitter_of("actorattack") is None and RC.other_emitter_of("codeattack") is None and RC.other_emitter_of("nosuch_eval") is None
