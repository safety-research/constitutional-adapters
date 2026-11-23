"""scripts/sbatch/sikl4_fit.sbatch: one job per row of configs/sikl4/fit_tasks.tsv, named by the row.

Executed, not string-matched, where it matters: the script runs in bash against a scratch checkout (a committed git repo holding
the script, the task table and the configs it names) with a fake ``scontrol`` on PATH, and

- refuses (exit 3, before any venv) a job whose name is not its positional row id, a row its checkout's table does not hold,
  and a name whose STOP sentinel exists;
- resolves a fit, a teacher and a smoke row by id (no arm or seed collision between the smoke and the production s1 rows) and
  then stops at its preflight on the missing inputs;
- exits 2 without SISTEER_SHARED_ROOT;
- its ``escalate()`` resubmits a plain job under the SAME --job-name with the SAME row id and attempt + 1 (and the job's whole
  comment), never with a STOP sentinel present or at the attempt cap.

Statically: STOP / RESULT / FIGS_DIR derive from the job name; no ``--export=`` / ``--mem=0`` / ``CUDA_VISIBLE_DEVICES``; the
workload line calls ``train_two_si_objects.py train`` with ``--require-teacher-topk``, an explicit ``--teacher-topk``,
``--checkpoint-dir`` and the row's ``--checkpoint-every``; a teacher row calls ``precompute-teacher`` recording the code sha; the
SIGTERM handler disarms its traps first.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from test_sdfbase_slurm_sbatch import _extract_function

REPO = Path(__file__).resolve().parents[1]
SBATCH = REPO / "scripts" / "sbatch" / "sikl4_fit.sbatch"
TASKS = REPO / "configs" / "sikl4" / "fit_tasks.tsv"
SRC = SBATCH.read_text()
FIT_ROW = "sikl-fit-si_constitution-psr_kl_all-s2-lr1"
TEACHER_ROW = "sikl-teacher-C16"
SMOKE_ROW = "sikl-fit-smoke-a2"
S1_ROW = "sikl-fit-safety_refusal-lora_r4-s1"


def _code_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def world(tmp_path):
    """A scratch checkout at a clean commit (the script, the trainer's path, the task table and every config it names) and a
    scratch shared root with hf_env.sh; a fake scontrol reporting this job."""
    repo = tmp_path / "checkout"
    (repo / "scripts" / "sbatch").mkdir(parents=True)
    (repo / "configs" / "sikl4").mkdir(parents=True)
    shutil.copy2(SBATCH, repo / "scripts" / "sbatch" / "sikl4_fit.sbatch")
    (repo / "scripts" / "train_two_si_objects.py").write_text("# placeholder: the job never reaches the workload here\n")
    rows = TASKS.read_text().splitlines()
    shutil.copy2(TASKS, repo / "configs" / "sikl4" / "fit_tasks.tsv")
    for cfg in sorted({r.split("\t")[6] for r in rows[1:]}):
        shutil.copy2(REPO / cfg, repo / cfg)
    _git(repo, "init", "-q")
    _git(repo, "-c", "user.email=t@example.invalid", "-c", "user.name=t", "add", "-A")
    _git(repo, "-c", "user.email=t@example.invalid", "-c", "user.name=t", "commit", "-q", "-m", "scratch")
    head = _git(repo, "rev-parse", "HEAD")
    shared = tmp_path / "shared"
    (shared / "scripts").mkdir(parents=True)
    (shared / "scripts" / "hf_env.sh").write_text(f'export HF_HOME="{shared}/hf"\n')
    fake = tmp_path / "bin"
    fake.mkdir()
    script = repo / "scripts" / "sbatch" / "sikl4_fit.sbatch"
    (fake / "scontrol").write_text(
        "#!/bin/bash\n"
        f"echo '   Command={script}'\n"
        "echo '   RunTime=00:00:01 TimeLimit=08:00:00 TimeMin=N/A'\n"
        f"echo '   Comment=sha:{head}'\n"
        f"echo '   StdOut={tmp_path}/job.out'\n")
    (fake / "scontrol").chmod(0o755)
    return {"repo": repo, "shared": shared, "fake": fake, "script": script, "head": head}


def _run(world, row: str, *, name: str | None = None, shared: bool = True, extra_env: dict | None = None):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "SISTEER_", "SIKL_"))}
    env.update({"PATH": f"{world['fake']}:{env['PATH']}", "SLURM_JOB_ID": "4242", "SLURM_JOB_NAME": name or row})
    if shared:
        env["SISTEER_SHARED_ROOT"] = str(world["shared"])
    env.update(extra_env or {})
    r = subprocess.run(["bash", str(world["script"]), row], env=env, capture_output=True, text=True, timeout=60)
    return r.returncode, r.stdout + r.stderr


def test_a_job_whose_name_is_not_its_row_id_exits_3_before_any_venv(world):
    rc, out = _run(world, FIT_ROW, name="sikl-fit-something-else")
    assert rc == 3 and "is not this job's name" in out
    assert "PREFLIGHT" not in out and "cu128" not in out


def test_a_row_missing_from_the_checkouts_table_exits_3(world):
    rc, out = _run(world, "sikl-fit-not-a-row")
    assert rc == 3 and "appears 0 time(s)" in out


@pytest.mark.parametrize("row,mode,needle", [
    (FIT_ROW, "fit", "si=si_constitution arm=psr_kl_all seed=2 lr_step=1"),
    (TEACHER_ROW, "teacher", "si=C16 arm=- seed=- lr_step=-"),
    (SMOKE_ROW, "smoke", "si=safety_refusal arm=lora_r4 seed=1 lr_step=0"),
    (S1_ROW, "fit", "si=safety_refusal arm=lora_r4 seed=1 lr_step=0"),
])
def test_fit_teacher_and_smoke_rows_resolve_by_id_and_stop_at_the_preflight(world, row, mode, needle):
    rc, out = _run(world, row)
    assert rc == 3, out
    assert f"row {row}: mode={mode} {needle}" in out
    assert "PREFLIGHT missing/empty" in out and "cu128 runtime assert" not in out
    if mode == "smoke":
        assert "out=exp/sikl4/smoke/a2 checkpoint_every=10" in out
    if row == S1_ROW:
        assert "out=exp/objects/sikl4/qwen35_4b/fits/s1/safety_refusal__lora_r4 checkpoint_every=100" in out
    if mode == "fit":
        assert "SISTEER_REGISTRY_SUBSTRATE is unset" in out          # a fit's records must state their substrate
    if mode == "teacher":
        assert "candidate-SI snapshot" in out                        # a C16 row needs C16's text
    # STOP / RESULT / FIGS_DIR are the job name's
    assert (world["shared"] / "exp" / "training" / row).is_dir() and (world["shared"] / "exp" / "figs" / row).is_dir()


def test_a_name_whose_stop_sentinel_exists_never_starts(world):
    stop = world["shared"] / "exp" / "training" / FIT_ROW
    stop.mkdir(parents=True)
    (stop / "STOP").write_text("")
    rc, out = _run(world, FIT_ROW)
    assert rc == 3 and "STOP sentinel present" in out and "PREFLIGHT" not in out


def test_no_shared_root_exits_2(world):
    rc, out = _run(world, FIT_ROW, shared=False)
    assert rc == 2 and "SISTEER_SHARED_ROOT is unset" in out


def test_a_dirty_checkout_refuses(world):
    (world["repo"] / "configs" / "sikl4" / "fit_tasks.tsv").write_text("row_id\n")
    rc, out = _run(world, FIT_ROW)
    assert rc == 3 and "modified tracked files" in out


# -- escalate(): the successor is the same name, the same row, attempt + 1 ----------------------------------------------------


def _escalate(tmp_path, *, attempt=1, max_attempts=3, stop=False) -> tuple[int, str, str]:
    fake = tmp_path / "ebin"
    fake.mkdir(exist_ok=True)
    (fake / "sbatch").write_text('#!/bin/bash\nfor a in "$@"; do echo "ARG:$a"; done\necho "ENV:SISTEER_SHARED_ROOT=$SISTEER_SHARED_ROOT"\n')
    (fake / "sbatch").chmod(0o755)
    stop_dir = tmp_path / "stop"
    stop_dir.mkdir(exist_ok=True)
    if stop:
        (stop_dir / "STOP").write_text("")
    script = tmp_path / "job.sbatch"
    script.write_text("#!/bin/bash\n")
    prog = ("log(){ echo \"LOG:$*\"; }\n"
            f"ESCALATE_QOS=high; STOP_DIR={stop_dir}; ATTEMPT={attempt}; MAX_ATTEMPTS={max_attempts}; SCRIPT_PATH={script}\n"
            f"RUN_NAME={FIT_ROW}; ROW_ID={FIT_ROW}; TIMELIMIT=08:00:00; STDOUT_PATH=/logs/{FIT_ROW}_4242.out\n"
            "COMMENT='sha:" + "a" * 40 + ",nonce:c6f7af912a06'; SHARED=/shared; CLUSTER_QOS=high; FIT_ENV_NAME=sdfbase-fit-tf515\n"
            + _extract_function(SRC, "escalate") + "escalate\n")
    r = subprocess.run(["bash", "-c", prog], env={**os.environ, "PATH": f"{fake}:{os.environ['PATH']}"}, capture_output=True,
                       text=True, timeout=30)
    args = [ln[4:] for ln in r.stdout.splitlines() if ln.startswith("ARG:")]
    return r.returncode, "\n".join(args), r.stdout


def test_escalate_resubmits_the_same_name_and_row_at_attempt_plus_one(tmp_path):
    rc, args, out = _escalate(tmp_path, attempt=1, max_attempts=3)
    lines = args.splitlines()
    assert rc == 0 and f"--job-name={FIT_ROW}" in lines and "--qos=high" in lines and "--time=08:00:00" in lines
    assert f"--output=/logs/{FIT_ROW}_4242.out" in lines
    assert "--comment=sha:" + "a" * 40 + ",nonce:c6f7af912a06" in lines           # the whole comment, nonce included
    assert lines[-3:] == [FIT_ROW, "2", "3"]                                        # the row id, attempt + 1, the cap
    assert lines[-4] == str(tmp_path / "job.sbatch")
    assert "ENV:SISTEER_SHARED_ROOT=/shared" in out
    assert not any(a.startswith("--export") for a in lines)


def test_escalate_never_resubmits_under_a_stop_or_at_the_cap(tmp_path):
    (tmp_path / "stopped").mkdir()
    _, args, out = _escalate(tmp_path / "stopped", stop=True)
    assert args == "" and "STOP sentinel present" in out
    (tmp_path / "capped").mkdir()
    _, args, out = _escalate(tmp_path / "capped", attempt=3, max_attempts=3)
    assert args == "" and "attempt cap 3 reached" in out


# -- static ----------------------------------------------------------------------------------------------------------------


def test_the_cluster_hard_rules_and_the_header():
    code = "\n".join(_code_lines(SRC))
    assert "--export" not in code and "--mem=0" not in SRC and "CUDA_VISIBLE_DEVICES" not in code
    header = {m.group(1): m.group(2) for m in re.finditer(r"^#SBATCH --([a-z-]+)(?:=(\S+))?", SRC, re.M)}
    assert header["qos"] == "high" and header["gres"] == "gpu:1" and header["mem"] == "200G" and header["partition"] == "general"
    assert header["cpus-per-task"] == "16" and header["time"] == "0-08:00:00" and header["signal"] == "B:TERM@180"
    assert "open-mode" in header and header["open-mode"] == "append" and "no-requeue" in header
    assert header["output"].endswith("/%u/exp/logs/%x_%j.out")


def test_stop_result_and_figs_derive_from_the_job_name():
    assert 'RUN_NAME="${SLURM_JOB_NAME:-}"' in SRC
    assert 'STOP_DIR="$SHARED/exp/training/$RUN_NAME"' in SRC and 'export FIGS_DIR="$SHARED/exp/figs/$RUN_NAME"' in SRC
    assert '"$FIGS_DIR/RESULT_${SLURM_JOB_ID:-x}"' in SRC and '"$FIGS_DIR/RESULT"' in SRC


def test_the_workload_lines():
    code = "\n".join(_code_lines(SRC)).replace("\\\n", " ")
    train = next(ln for ln in code.splitlines() if "train_two_si_objects.py\" train" in ln)
    for flag in ('--teacher-topk "$TEACHER"', "--require-teacher-topk", '--checkpoint-dir "$CKPT_DIR"',
                 '--checkpoint-every "$CKPT_EVERY"', '--expect-config-sha "$CONFIG_SHA"', '--out "$OUT_ROOT"',
                 '--arms "$ARM"', '--si "$SI"', '--pairs "$PAIRS"', "srun --ntasks=1"):
        assert flag in train, flag
    teacher = next(ln for ln in code.splitlines() if "precompute-teacher" in ln)
    for flag in ('--out "$TEACHER"', '--code-sha "$HEAD_SHA"', '--expect-config-sha "$CONFIG_SHA"'):
        assert flag in teacher, flag
    assert "--top-k" not in teacher          # the width is precompute-teacher's default, the one every fit validates against
    assert 'CKPT_DIR="$OUT_ROOT/ckpt"' in SRC
    # the registry: this script's campaign and spools, never a push
    assert "export REGISTRY_CAMPAIGN=si_target_refit" in SRC and "unset SISTEER_REGISTRY_PUSH" in SRC
    assert 'SISTEER_REGISTRY_SPOOL="$SHARED/registry/hold/si_target_refit"' in SRC
    assert 'SISTEER_REGISTRY_SPOOL="$SHARED/tmp/sikl4-smoke-spool"' in SRC


def test_the_sigterm_handler_disarms_first_and_the_cu128_assert_precedes_the_workload():
    body = re.search(r"\non_term\(\) \{\n(.*?)\n\}\n", SRC, re.S).group(1)
    first = [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("#")][0]
    assert first == "trap '' SIGTERM SIGINT SIGQUIT"
    assert SRC.index('assert torch.__version__ == "2.11.0+cu128"') < SRC.index("srun --ntasks=1")
    assert SRC.index("--- the row:") < SRC.index('VENV="$SHARED/envs/')
    subprocess.run(["bash", "-n", str(SBATCH)], check=True)
