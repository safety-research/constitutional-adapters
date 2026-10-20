#!/bin/bash
# Submit-time wrapper for evals_cell_controls.sbatch — run ON a cluster
# login node (either cluster). It exists because two header values cannot
# be cluster-derived inside an #SBATCH block (headers are parsed before any
# shell runs, so they cannot branch):
#
#   * --qos:    the header keeps one cluster's default; this wrapper takes
#               the cluster's QoS from the job environment
#               (SISTEER_CLUSTER_QOS) and overrides it at submit time.
#               QOS=<qos> (body env, never sbatch --export — hard rule 1)
#               overrides the cluster's QoS.
#   * --output: Slurm does NOT create log parents, and a job whose --output
#               parent does not exist NEVER STARTS. The cluster-correct log
#               dir is created HERE, before sbatch, and passed as a
#               submit-time override.
#
# The shared root, the project directory and the QoS are the job environment
# the front door composes from the cluster's site block. To submit by hand,
# first run in the login shell the lines
#   python reproduce/lib/submit_twin.py --config <site config> job-env --cluster <label> --export
# prints. sbatch hands that environment to the job, which refuses without it
# too, so this wrapper refuses by name before anything is submitted.
#
# Extra sbatch flags pass straight through (e.g. a --time override):
#   scripts/sbatch/submit_evals_cell_controls.sh [sbatch flags...]
set -uo pipefail

SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
QOS="${QOS:-${SISTEER_CLUSTER_QOS:?set SISTEER_CLUSTER_QOS or QOS, the QoS of the cluster (the front door exports it from the site config)}}"
# the job's checkout: CAPEVAL_REPO_DIR, else the one beside the project directory (evals_cell_controls.sbatch)
: "${CAPEVAL_REPO_DIR:-${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}}"

LOGDIR="$SHARED/exp/logs"
mkdir -p "$LOGDIR"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec sbatch --qos="$QOS" --output="$LOGDIR/%x_%j.out" "$@" \
  "$HERE/evals_cell_controls.sbatch"
