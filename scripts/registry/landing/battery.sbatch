#!/bin/bash
#SBATCH --job-name=battery
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append
# battery.sbatch TREE VENV RESULTS_ROOT LABEL PLAN_JSON — one array task = one DISJOINT file list of a targeted battery.
# Submitted by battery_slurm.sh as: sbatch -J battery-<sha7>-<label>-<user> --array=0-$((K-1)) --time=<plan walltime>
#   [--dependency=afterok:<venv job>] battery.sbatch TREE VENV RESULTS_ROOT LABEL PLAN_JSON
# PLAN_JSON (battery_collect.py plan) is the ONE input: task i runs plan.tasks[i].files — and nothing else — through the
# exported tree's OWN gate_task.py (TREE/scripts/registry/landing/gate_task.py, so a commit that changes the gate code is
# tested under its own gate code) with --rerun-label LABEL/t<i>, which records under RESULTS_ROOT/<sha>/reruns/LABEL/t<i>/
# {summary.json,junit.xml,pytest.log}: no two tasks share a list or a record path (gate_array.sbatch's FILES positional is
# the same for every task of an array, which is why a battery never uses it). pytest is bounded to the plan's timeout_s
# inside the plan's walltime (the tree copy and the --local-venv copy, ~4–5 min measured, come before pytest starts).
# gate_task.py writes SLURM_ARRAY_JOB_ID / SLURM_ARRAY_TASK_ID into the record (slurm_array_job_id, slurm_array_task_id): the
# collector binds every record to the array the submission recorded, so a record of another array under the same label is
# refused; the line after gate_task returns names what the record says, for the job log.
# Fail-fast guards check every external input before any work; exit 2 = refused here, else gate_task's rc (0 PASS/FAIL
# record written, 3 TIMEOUT, 2 refusal record).
set -uo pipefail
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

TREE=${1:?TREE} VENV=${2:?VENV} RESULTS_ROOT=${3:?RESULTS_ROOT} LABEL=${4:?LABEL} PLAN_JSON=${5:?PLAN_JSON}
# the cluster's shared root: CLUSTER_SHARED_ROOT when the job's environment carries it, else the venv's grandparent directory
# (VENV is <shared root>/envs/<name>, the gate venv's layout; gate_venv.sbatch reads it the same way)
SHARED=${CLUSTER_SHARED_ROOT:-$(dirname "$(dirname "$VENV")")}
HF_ENV=$SHARED/scripts/hf_env.sh
if [ -r "$HF_ENV" ]; then source "$HF_ENV"; else echo "[battery] no $HF_ENV on this host: HF_HOME stays as inherited"; fi
I=${SLURM_ARRAY_TASK_ID:-}
WORK=${BATTERY_WORK_ROOT:-/tmp}/battery-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}_${I:-x}
cleanup() { trap '' SIGTERM; kill -TERM -$$ 2>/dev/null; wait; rm -rf "$WORK"; }
trap cleanup SIGTERM SIGINT SIGQUIT
trap 'rm -rf "$WORK"' EXIT
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
GATE_TASK=$TREE/scripts/registry/landing/gate_task.py

echo "[battery] $(date -u +%FT%TZ) host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-?}_${I:-?} label=$LABEL task=$I plan=$PLAN_JSON"
[ -n "$I" ]                         || { echo "[battery] SLURM_ARRAY_TASK_ID is unset (array-only script)"; exit 2; }
[ -d "$TREE/.git" ]                 || { echo "[battery] no exported tree at $TREE"; exit 2; }
[ -s "$VENV/.gate-ready.json" ]     || { echo "[battery] venv not ready: $VENV/.gate-ready.json missing"; exit 2; }
[ -x "$VENV/bin/python" ]           || { echo "[battery] no python in $VENV"; exit 2; }
[ -s "$PLAN_JSON" ]                 || { echo "[battery] no plan at $PLAN_JSON"; exit 2; }
[ -s "$GATE_TASK" ] && [ -s "$TREE/scripts/registry/landing/gate_lib.py" ] || { echo "[battery] no gate_task.py/gate_lib.py in $TREE"; exit 2; }
mkdir -p "$WORK" || { echo "[battery] cannot create $WORK"; exit 2; }

# the task's share of the plan: sha, count, timeout, flags, then its files one per line (refused when the plan is not this
# label's, when the index is outside the plan, or when the list is empty — an empty list would record a PASS with no tests)
PLAN_OUT=$("$VENV/bin/python" - "$PLAN_JSON" "$I" "$LABEL" 2>&1 <<'PY'
import json, sys
plan = json.load(open(sys.argv[1])); i = int(sys.argv[2]); label = sys.argv[3]
if plan.get("label") != label: sys.exit(f"ERR plan label {plan.get('label')!r} is not {label!r}")
tasks = plan.get("tasks") or []
if not 0 <= i < len(tasks) or plan.get("count") != len(tasks): sys.exit(f"ERR task {i} is outside the plan's {len(tasks)} tasks (count {plan.get('count')})")
files = tasks[i].get("files") or []
if not files: sys.exit(f"ERR task {i} has an empty file list")
if any("\n" in f or not f for f in files): sys.exit("ERR a file name is malformed")
print(plan["sha"]); print(len(tasks)); print(int(plan.get("timeout_s") or 2400)); print(" ".join(plan.get("task_flags") or []))
print("\n".join(files))
PY
); prc=$?
[ $prc -eq 0 ] || { echo "[battery] plan refused: $PLAN_OUT"; exit 2; }
mapfile -t PLAN_LINES <<<"$PLAN_OUT"
SHA=${PLAN_LINES[0]} COUNT=${PLAN_LINES[1]} TIMEOUT_S=${PLAN_LINES[2]} FLAGS=${PLAN_LINES[3]}
FILES=("${PLAN_LINES[@]:4}")
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "[battery] plan sha '$SHA' is not 40 hex"; exit 2; }
echo "[battery] sha=$SHA task=$I/$COUNT timeout_s=$TIMEOUT_S flags='$FLAGS' n_files=${#FILES[@]}: ${FILES[*]}"

# --files LAST: it takes every remaining argument
# shellcheck disable=SC2086
srun "$VENV/bin/python" "$GATE_TASK" --sha "$SHA" --tree "$TREE" --index "$I" --count "$COUNT" --results-root "$RESULTS_ROOT" \
  --venv "$VENV" --work "$WORK" --run-name "battery-${SHA:0:7}-$LABEL" --timeout-s "$TIMEOUT_S" $FLAGS \
  --rerun-label "$LABEL/t$I" --files "${FILES[@]}"
rc=$?
REC=$RESULTS_ROOT/$SHA/reruns/$LABEL/t$I/summary.json
echo "[battery] $(date -u +%FT%TZ) task $I rc=$rc record=$REC"
if [ -s "$REC" ]; then
  "$VENV/bin/python" - "$REC" "${SLURM_ARRAY_JOB_ID:-}" "$I" <<'PY'
import json, sys
rec = json.load(open(sys.argv[1])); a, t = rec.get("slurm_array_job_id"), rec.get("slurm_array_task_id")
# the collector's rule: the array must be this attempt's; the task id must be this task's WHEN the record carries one (a tree
# whose gate_task.py predates the field records none, and its shard.index still pins the task)
ok = str(a) == sys.argv[2] and (t is None or str(t) == sys.argv[3])
note = "" if t is not None else "; task id not recorded by this tree's gate_task.py"
print(f"[battery] record names array={a} task={t} ({'this job' if ok else 'NOT this job ' + sys.argv[2] + '_' + sys.argv[3] + ': the collector will refuse it'}{note})")
PY
fi
exit $rc
