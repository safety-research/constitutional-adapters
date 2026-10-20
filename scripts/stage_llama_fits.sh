#!/bin/bash
# Stage the Llama fit waves onto cluster A (PREP — prints the submit lines,
# never submits). Run on the grading host from the twosweep worktree AFTER
# the Llama HPO winners are folded + committed + pushed:
#
#   python scripts/make_fit_configs.py --llama-winners <winners.json>
#   git commit ... && git push
#   scripts/stage_llama_fits.sh
#
# What it does:
#   1. refuses a dirty/unpushed-to-mirror state (the checkout must be
#      reproducible from the cluster mirror).
#   2. mirrors the repo (the repo-mirror script) and creates the immutable
#      per-commit checkout $SHARED/sisteer/checkouts/<sha>.
#   3. rsyncs the clean v3 pool + tokcache sidecars to
#      $SHARED/exp/twosweep/pairs/llama_v3/ (idempotent; skips missing
#      sidecars with a loud note).
#   4. writes the sbatch's pointer file (CHECKOUT_SHA) + pre-creates the
#      0700 log dir.
#   5. prints the two sbatch lines (resp + tf waves) and the watcher lines.
set -euo pipefail

WT=$(cd "$(dirname "$0")/.." && pwd)
# The site: the config file the front door reads (reproduce/lib/common.sh repro_config_path: $SISTEER_CONFIG, else the site
# directory's config.env) and in it the cluster's block (reproduce/lib/clusters.py). STAGE_CLUSTER names the cluster (a site or
# registry label; default cluster_a) and HOST overrides its login host. Every root below is that block's
# shellcheck source=../reproduce/lib/common.sh
source "$WT/reproduce/lib/common.sh"
CFG=$(repro_config_path)
[ -r "$CFG" ] || { echo "FATAL: no site config at $CFG (set SISTEER_CONFIG to the site config file, or install the site under \$SISTEER_SITE_DIR, default ~/.config/sisteer)" >&2; exit 1; }
STAGE_CLUSTER=${STAGE_CLUSTER:-cluster_a}
site_get() { python3 "$WT/reproduce/lib/clusters.py" --config "$CFG" --get "$STAGE_CLUSTER" "$1"; }
HOST=${HOST:-$(site_get CLUSTER_SSH)}
REMOTE_SHARED=$(site_get CLUSTER_SHARED_ROOT)
PROJECT_ROOT=$(site_get CLUSTER_PROJECT_ROOT)
REPO_MIRROR=$(site_get CLUSTER_REPO_MIRROR)
[ -n "$HOST" ] && [ -n "$REMOTE_SHARED" ] && [ -n "$PROJECT_ROOT" ] && [ -n "$REPO_MIRROR" ] || { echo "FATAL: the site config gives no login host, shared root, project directory or repo mirror (CLUSTER_SSH, CLUSTER_SHARED_ROOT, CLUSTER_PROJECT_ROOT, CLUSTER_REPO_MIRROR) for cluster $STAGE_CLUSTER" >&2; exit 1; }
SHA=$(git -C "$WT" rev-parse HEAD)
# the site's W&B destination for the printed submissions (the READY block below), composed before anything is touched
WANDBENV=$(python3 "$WT/reproduce/lib/submit_twin.py" --config "$CFG" job-env --wandb) \
  || { echo "FATAL: the site's W&B destination did not compose (reproduce/lib/submit_twin.py job-env --wandb; reason above)" >&2; exit 1; }

if [ -n "$(git -C "$WT" status --porcelain --untracked-files=no)" ]; then
  echo "FATAL: worktree has uncommitted tracked changes — commit first" >&2
  exit 1
fi
for f in configs/two_si_llama_fits.yaml configs/two_si_llama_fits_dp3.yaml; do
  if grep -q PENDING_LLAMA_HPO_WINNERS "$WT/$f"; then
    echo "NOTE: $f still carries the winner sentinels — staging proceeds," \
         "but the sbatch will refuse until the fold lands." >&2
  fi
done

echo "== mirror repo -> cluster =="
"$HOME/scripts/mirror-repos.sh" "$(basename "$(dirname "$(git -C "$WT" rev-parse --path-format=absolute --git-common-dir)")")"

echo "== create checkout $SHA =="
ssh "$HOST" "set -e; SHARED=$REMOTE_SHARED; \
  install -d -m 700 \$SHARED/exp/twosweep/fits/_logs \$SHARED/exp/twosweep/pairs/llama_v3; \
  mkdir -p $PROJECT_ROOT/checkouts; \
  if [ ! -d $PROJECT_ROOT/checkouts/$SHA/.git ]; then \
    git clone -q --no-checkout $REPO_MIRROR $PROJECT_ROOT/checkouts/$SHA; \
  fi; \
  git -C $PROJECT_ROOT/checkouts/$SHA fetch -q origin; \
  git -C $PROJECT_ROOT/checkouts/$SHA checkout -q --detach $SHA; \
  echo $SHA > \$SHARED/exp/twosweep/fits/CHECKOUT_SHA; \
  test -x \$SHARED/envs/twosweep-fit/bin/python || \
    { echo 'FATAL: \$SHARED/envs/twosweep-fit venv missing'; exit 1; }"

echo "== rsync pool + sidecars =="
missing=0
for si in safety_refusal si_constitution si_irrelevant; do
  p="$HOME/twosi_llama_v3/pairs/pairs_${si}.jsonl"
  for f in "$p" "$p.tokcache.npz"; do
    if [ -s "$f" ]; then
      rsync -a --chmod=F600 "$f" "$HOST:$REMOTE_SHARED/exp/twosweep/pairs/llama_v3/"
    else
      echo "MISSING (gate): $f" >&2; missing=1
    fi
  done
done
[ "$missing" -eq 1 ] && echo "NOTE: sidecars still building — re-run this script when they finish." >&2

CKPATH="$PROJECT_ROOT/checkouts/$SHA"
# each printed submission carries what the job refuses without, the cluster's shared root and project directory (its checkout
# sits under the latter), as the front door's job env names them, and the site's W&B destination: the fit configs' wandb_project
# is $WANDB_PROJECT, which the job expands from it, and the job files its runs under the site's entity. The W&B keys are the
# front door's own (job-env --wandb prints the W&B part of its job env alone; this job takes no registry settings); with no
# entity in the site there are none
JOBENV="SISTEER_SHARED_ROOT=$(printf '%q' "$REMOTE_SHARED") SISTEER_PROJECT_ROOT=$(printf '%q' "$PROJECT_ROOT")${WANDBENV:+ $WANDBENV}"
# one shell word for ssh: the remote command single-quoted, any quote inside it closed and escaped
sq() { printf "'%s'" "${1//\'/\'\\\'\'}"; }
cat <<EOF

== READY (nothing submitted). Submit when the gates are clear ==
# response wave (3 tasks x 3 GPUs = 9 of the 16-GPU high budget):
ssh $HOST $(sq "$JOBENV sbatch -J tt-twosifit-llama-v3-resp $CKPATH/scripts/sbatch/twosi_fits_llama.sbatch")
# token_faithful wave (3 tasks x 1 GPU):
ssh $HOST $(sq "$JOBENV sbatch -J tt-twosifit-llama-v3-tf --gres=gpu:1 --cpus-per-task=16 --mem=128G $CKPATH/scripts/sbatch/twosi_fits_llama.sbatch")
# then arm watchers (one per array id):
~/scripts/watch-job.sh --slurm <resp_array_id> --host $HOST
~/scripts/watch-job.sh --slurm <tf_array_id> --host $HOST
# offline W&B sync after the wave (from the grading host):
ssh $HOST 'ls $REMOTE_SHARED/exp/twosweep/wandb'   # then wandb sync each run dir
EOF
