#!/bin/bash
# STRENGTH FRONTIER: build + --check the full22 bank views of ONE target on the cluster this job runs on — ZERO GPUs
# (symlinks, sha256 and one CPU torch import for the derived cells' verdict), so `general`/`overflow` with no --gres
# (never `dev` for scheduler-side CPU work, never a --gres to make it schedulable).
#
#   sbatch --qos=<the cluster's QoS> [--output=<log>] --job-name=llad-views-<cluster> \
#       scripts/sdfbase/full22/strength/build_views.sbatch <wave sha> <target> <view,view,...>
#
# from a login shell that carries the job environment the front door composes from the cluster's site block (the job refuses by
# name without its shared root, registry label and checkouts root): first run there the lines
#   python reproduce/lib/submit_twin.py --config <site config> job-env --cluster <label> --export
# prints.
#
# The checkout is $SISTEER_EVAL_REPO_ROOT/<sha> (README: a detached worktree of the cluster mirror), the
# interpreter the FIT stack's shared env (torch + yaml; sisteer imported from the checkout's src/ — never a
# node-local venv, which would not exist on whichever node the scheduler picks). Every view is BUILT then
# re-verified with --check; the job exits non-zero on the first refused view, and a RESULT line names each.
#SBATCH --partition=general,overflow
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# --time: MEASURED 2026-09-11 on the 12 llama8b views — cluster B ~50 s/view (9m30 total), cluster A ~3 min/view (two CPU
# torch imports per view over NFS), so 12 views need ~36 min there; 1h15 is ~2x that. The first cluster A run at 30 min
# would have hit the wall on the tenth view (its main-family views were handed to a second 1h job).
#SBATCH --time=01:15:00
#SBATCH --open-mode=append
set -uo pipefail
WAVE_SHA="${1:?wave sha}"; TARGET="${2:?target}"; VIEWS="${3:?comma list of views}"
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
CLUSTER="${SISTEER_REGISTRY_SUBSTRATE:?set SISTEER_REGISTRY_SUBSTRATE, the registry label of the cluster (the front door exports it from the site config)}"
case "$CLUSTER" in   # each cluster's prebuilt fit venv
  cluster_a) PY="$SHARED/envs/sdfbase-fit/bin/python" ;;
  cluster_b) PY="$SHARED/envs/strength-dose-fit/bin/python" ;;
  *) echo "FATAL: no fit venv is named for cluster $CLUSTER"; exit 1 ;;
esac
source "$SHARED/scripts/hf_env.sh"; export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
REPO_DIR="${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA"
log(){ echo "[$(date -Is)] $*"; }
log "cluster=$CLUSTER shared=$SHARED sha=$WAVE_SHA target=$TARGET views=$VIEWS node=$(hostname) job=${SLURM_JOB_ID:-manual} gpus=${CUDA_VISIBLE_DEVICES:-none}"
[ -d "$REPO_DIR/src/sisteer" ] || { log "FATAL: no checkout at $REPO_DIR"; exit 1; }
cd "$REPO_DIR" || exit 1
head=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
[ "$head" = "$WAVE_SHA" ] || { log "FATAL: checkout HEAD $head != $WAVE_SHA"; exit 1; }
git diff --quiet HEAD -- || { log "FATAL: dirty checkout $REPO_DIR"; exit 1; }
[ -x "$PY" ] || { log "FATAL: no interpreter at $PY"; exit 1; }
export PYTHONPATH="$REPO_DIR/src"
srun --ntasks=1 "$PY" -c 'import sys, yaml, torch; print("interp OK", sys.version.split()[0], "torch", torch.__version__, "yaml", yaml.__version__)' || { log "FATAL: interpreter guard"; exit 1; }
rc=0; results=()
IFS=',' read -r -a VIEW_LIST <<< "$VIEWS"
for view in "${VIEW_LIST[@]}"; do
  log "== build $TARGET/$view"
  if srun --ntasks=1 "$PY" scripts/sdfbase/full22/make_bank_view.py --target "$TARGET" --view "$view" \
     && srun --ntasks=1 "$PY" scripts/sdfbase/full22/make_bank_view.py --check --target "$TARGET" --view "$view"; then
    results+=("$view=ok")
  else
    results+=("$view=FAIL"); rc=2
  fi
done
log "RESULT cluster=$CLUSTER target=$TARGET rc=$rc ${results[*]}"
exit $rc
