#!/usr/bin/env bash
# sbatch_registered.sh — submit ONE job and register it in the results registry.
#
#   scripts/registry/sbatch_registered.sh --campaign C --planned <planned-run.json> [--repo DIR] [--spool DIR]
#                                         [--pending-dir DIR] [--check-only] -- sbatch --comment=sha:<sha> <script> [args]
#
# The drop-in for every `sbatch` line a submit printer emits. In order:
#   1. STRICTNESS: the planned run's identity is validated BEFORE anything is
#      submitted — an eval named without its resolved item set / instrument REFUSES TO SUBMIT (exit 2) and prints
#      what it would need. Nothing is written, nothing is queued.
#   2. `sbatch …` runs exactly as given; its stdout is echoed; `Submitted batch job <id>` is parsed.
#   3. `python3 -S -m sisteer.registry.emit_cli register` writes `run` (queued) + `run_state queued` + the job
#      binding (`cluster_job_id`, `job_name`, `submit_sha_pin` from --comment=sha:), stores the plan (+ run_id) at
#      <pending-dir>/<jobid>.json — the job finds its own identity by $SLURM_JOB_ID (s2_init / f22_init) — and
#      pushes (debounced; a failed push is logged, never fatal).
#   A registration failure AFTER a successful sbatch never un-submits the job: it is reported (exit 0), the job's
#   own start-time late registration is the backstop.
#
# Runs on a cluster login node (spool $SHARED/registry/records, pending $SHARED/registry/pending) or on the grading host
# (`~/registry/records`, `~/registry/pending`). Stdlib python only; no venv needed. Never `--export=`.
set -uo pipefail

usage() { sed -n '2,20p' "$0" >&2; exit 2; }

CAMPAIGN=""; PLANNED=""; REPO=""; SPOOL=""; PENDING=""; CHECK_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --campaign) CAMPAIGN="$2"; shift 2 ;;
    --planned) PLANNED="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --spool) SPOOL="$2"; shift 2 ;;
    --pending-dir) PENDING="$2"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --) shift; break ;;
    -h|--help) usage ;;
    *) echo "sbatch_registered: unknown option $1" >&2; usage ;;
  esac
done
[[ -n "$PLANNED" && -f "$PLANNED" ]] || { echo "sbatch_registered: --planned <planned-run.json> is required" >&2; usage; }
[[ $# -gt 0 ]] || { echo "sbatch_registered: nothing to submit after --" >&2; usage; }

# the checkout whose src/ carries sisteer.registry (stdlib-only): --repo, else this script's own checkout
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
[[ -d "$REPO/src/sisteer/registry" ]] || { echo "sbatch_registered: no sisteer.registry under $REPO/src (pass --repo)" >&2; exit 2; }
PY=(env "PYTHONPATH=$REPO/src" python3 -S -m sisteer.registry.emit_cli)

# spool + pending dir: the shared root when on a cluster, ~/registry on the grading host
# (the cluster's shared root is SISTEER_SHARED_ROOT, else CLUSTER_SHARED_ROOT; registry_env.sh exports SISTEER_REGISTRY_SPOOL)
if [[ -z "$SPOOL" ]]; then
  _shared="${SISTEER_SHARED_ROOT:-${CLUSTER_SHARED_ROOT:-}}"
  if [[ -n "$_shared" && -d "$_shared/registry" ]]; then SPOOL="$_shared/registry/records"
  elif [[ -n "${SISTEER_REGISTRY_SPOOL:-}" ]]; then SPOOL="$SISTEER_REGISTRY_SPOOL"
  elif [[ -d "$HOME/registry" || $CHECK_ONLY -eq 1 ]]; then SPOOL="$HOME/registry/records"   # --check-only writes nothing
  else
    echo "sbatch_registered: no spool: pass --spool, or set SISTEER_REGISTRY_SPOOL (source <shared root>/scripts/registry/registry_env.sh)" \
         "or SISTEER_SHARED_ROOT (the cluster's shared root)" >&2
    exit 2
  fi
fi
PENDING="${PENDING:-$(dirname "$SPOOL")/pending}"
# the push script beside the spool when installed (cluster); the emitter honours SISTEER_REGISTRY_PUSH
if [[ -z "${SISTEER_REGISTRY_PUSH:-}" && -x "$(dirname "$(dirname "$SPOOL")")/scripts/registry/push_records.sh" ]]; then
  export SISTEER_REGISTRY_PUSH="$(dirname "$(dirname "$SPOOL")")/scripts/registry/push_records.sh"
fi

# 1. strictness: refuse to submit an unresolved plan
if ! "${PY[@]}" register --plan "$PLANNED" --check-only >/dev/null; then
  echo "sbatch_registered: REFUSING TO SUBMIT — the planned run is not fully resolved (see the [registry] lines above)." >&2
  exit 2
fi
[[ "$CHECK_ONLY" == "1" ]] && { echo "sbatch_registered: plan OK (check only; nothing submitted)"; exit 0; }

# the submit-time sha pin, when the sbatch line carries one (stage2_common.sh's gate requires it)
SHA_PIN=""
for a in "$@"; do
  case "$a" in --comment=sha:*) SHA_PIN="${a#--comment=sha:}" ;; esac
done
JOB_NAME=""
for (( i=1; i<=$#; i++ )); do
  a="${!i}"
  case "$a" in
    --job-name=*|-J*) JOB_NAME="${a#--job-name=}"; JOB_NAME="${JOB_NAME#-J}" ;;
    --job-name|-J) j=$((i+1)); JOB_NAME="${!j}" ;;
  esac
done

# 2. submit exactly as given
OUT="$("$@")"; RC=$?
[[ -n "$OUT" ]] && echo "$OUT"
if [[ $RC -ne 0 ]]; then echo "sbatch_registered: sbatch exited $RC; nothing registered" >&2; exit $RC; fi
JOB_ID="$(printf '%s\n' "$OUT" | sed -n 's/^Submitted batch job \([0-9][0-9_]*\).*$/\1/p' | head -1)"
if [[ -z "$JOB_ID" ]]; then
  echo "sbatch_registered: could not parse a job id from sbatch's output; the job runs unregistered (its start registers late)" >&2
  exit 0
fi

# 3. register (a failure here never un-submits the job)
if ! "${PY[@]}" register --plan "$PLANNED" --spool "$SPOOL" --job-id "$JOB_ID" ${JOB_NAME:+--job-name "$JOB_NAME"} \
       ${SHA_PIN:+--sha-pin "$SHA_PIN"} --pending-dir "$PENDING" >/dev/null; then
  echo "sbatch_registered: job $JOB_ID submitted but NOT registered (see the [registry] lines above); the job's start registers late" >&2
fi
echo "sbatch_registered: job $JOB_ID registered (pending file $PENDING/$JOB_ID.json)"
exit 0
