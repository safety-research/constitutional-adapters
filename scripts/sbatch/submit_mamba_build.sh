#!/bin/bash
# Submit-time wrapper for mamba_build.sbatch, the once-per-cluster Mamba-kernel
# cache warm-up -- run ON a cluster login node (either cluster). It exists for
# the same reason as submit_evals_cell_controls.sh: two header values cannot
# be cluster-derived inside an #SBATCH block (headers are parsed before any
# shell runs, so they cannot branch), which would leave the warm-up
# unsubmittable on a cluster whose values differ from the header's:
#
#   * --qos:    the header keeps one cluster's default; this wrapper takes the
#               cluster's QoS from the job environment (SISTEER_CLUSTER_QOS)
#               and overrides it at submit time. QOS=<qos> (body env, never
#               sbatch --export -- hard rule 1) overrides the cluster's QoS.
#   * --output / --error: Slurm does NOT create log parents, and a job whose
#               --output parent does not exist NEVER STARTS. The cluster's
#               PRIVATE log dir ($SHARED/exp/jailbreak/_logs, mode 0700: the
#               optional campaign-seam check reads harmful prompt lists, so a
#               traceback could print one) is created HERE, before sbatch, and
#               both paths are passed as submit-time overrides.
#
# The shared root, the project directory and the QoS are the job environment
# the front door composes from the cluster's site block. To submit by hand,
# first run in the login shell the lines
#   python reproduce/lib/submit_twin.py --config <site config> job-env --cluster <label> --export
# prints. sbatch hands that environment to the job, which refuses without it
# too, so this wrapper refuses by name before anything is submitted.
#
# Usage:
#   scripts/sbatch/submit_mamba_build.sh [sbatch flags...] [-- CODE [CONFIG]]
# Everything before `--` goes to sbatch (e.g. --time=4:00:00); everything after
# it is mamba_build.sbatch's own positionals (the checkout to sync and run the
# proof from, then the optional Mamba-hybrid target config for the
# campaign-seam check). The job script's header documents both.
set -uo pipefail

SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
QOS="${QOS:-${SISTEER_CLUSTER_QOS:?set SISTEER_CLUSTER_QOS or QOS, the QoS of the cluster (the front door exports it from the site config)}}"
# the job's default checkout and its scratch venv sit under the project directory (mamba_build.sbatch)
: "${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}"

LOGDIR="$SHARED/exp/jailbreak/_logs"
install -d -m 700 "$LOGDIR"

SBATCH_FLAGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --) shift; break ;;
    *) SBATCH_FLAGS+=("$1"); shift ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec sbatch --qos="$QOS" --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.err" \
  ${SBATCH_FLAGS[@]+"${SBATCH_FLAGS[@]}"} \
  "$HERE/mamba_build.sbatch" "$@"
