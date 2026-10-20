#!/bin/bash
#SBATCH --job-name=qwen-wave
#SBATCH --partition=general
#SBATCH --qos=high                    # standing default; 1 GPU of the 16-GPU/user budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G                    # NEVER --mem=0
#SBATCH --time=0-08:00:00             # served BoN(64) x 112 goals x 5 families + ~9min cold GDN JIT; empty cluster, high (not preempted)
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
# exclude bad nodes at submission: sbatch $CLUSTER_SBATCH_EXTRA <this script> (CLUSTER_SBATCH_EXTRA is site config)
#
# One LENS-FREE qwen served condition (none | prompted__<si>), one model.
# Serves through the routing proxy over vLLM 0.25.1; installs no psr/lora
# hook (enumerate_cells is filtered to spec.cells = {none, prompted}, so
# only the always-emitted none__lens0 zero-op is registered and it stays
# dormant unless served — it is not). Passthrough-with-hooks is proven on
# qwen (gate 255587 baseline). Decoupled from the hybrid-metadata lens fix.
#
# Submit (job name carries model+condition so squeue/scancel are self-scoped):
#   sbatch -J qwen-wave-<mk>-<cond>-f0830 scripts/sbatch/qwen_wave.sbatch \
#          <checkout-sha> <qwen35_9b|qwen35_4b> <none|safety|constitution|irrelevant>
# Stop: scancel <job_id> ONLY (M15). NO --export= (hard rule 1).

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the cluster shared root (the front door exports it from the site config)}"
source "$SHARED/scripts/hf_env.sh"
export HF_HUB_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
# THE HELD-OUT CORPUS DIR — renellm (a held-out family) reads its FROZEN
# corpus from here; UNSET, renellm writes nothing and the wave's
# every_family_nonempty check correctly refuses at final validation
# (job 268567, 2026-08-30 — 4 of 5 families generated, renellm 0 rows).
# Defaulting form (the psrtransfer pattern).
export SISTEER_HELDOUT_CORPUS_DIR="${SISTEER_HELDOUT_CORPUS_DIR:-$SHARED/exp/heldout_corpora}"
export WANDB_MODE=disabled WANDB_DISABLED=true   # compute nodes can't reach api.wandb.ai

SHA="${1:?usage: sbatch qwen_wave.sbatch <sha> <model-key> <condition>}"
MK="${2:?model-key}"
COND="${3:?condition}"
RESUME_OUT_DIR="${4:-}"   # optional: a prior run dir to RESUME into (skips
                         # already-covered BoN + family units); empty = fresh
case "$MK" in
  qwen35_9b) WMODEL="Qwen/Qwen3.5-9B"; WREV="c202236235762e1c871ad0ccb60c8ee5ba337b9a";;
  qwen35_4b) WMODEL="Qwen/Qwen3.5-4B"; WREV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a";;
  *) echo "FATAL: model key $MK not in {qwen35_9b,qwen35_4b}" >&2; exit 1;;
esac
case "$COND" in
  none)         WCELL="none";                      WPROMPTED="prompted__safety";;
  safety)       WCELL="prompted__safety_refusal";  WPROMPTED="prompted__safety";;
  constitution) WCELL="prompted__si_constitution"; WPROMPTED="prompted__constitution";;
  irrelevant)   WCELL="prompted__si_irrelevant";   WPROMPTED="prompted__irrelevant";;
  *) echo "FATAL: condition $COND not in {none,safety,constitution,irrelevant}" >&2; exit 1;;
esac

REPO_DIR="${SISTEER_PETRI_REPO_ROOT:?set SISTEER_PETRI_REPO_ROOT, the root of the per-commit Petri checkouts (the front door exports it from the site config)}/$SHA"
if [ ! -d "$REPO_DIR/src/sisteer" ]; then
  echo "FATAL: no commit-addressed checkout at $REPO_DIR; refusing" >&2; exit 1
fi
cd "$REPO_DIR"
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
[ "$HEAD_SHA" = "$SHA" ] || { echo "FATAL: HEAD $HEAD_SHA != pinned $SHA" >&2; exit 1; }
git diff --quiet HEAD -- 2>/dev/null || { echo "FATAL: tracked files modified in $REPO_DIR" >&2; exit 1; }
echo "[wave] $REPO_DIR @ $HEAD_SHA on $(hostname): $MK / $COND (cell=$WCELL)"
# renellm replays this cluster's copy when there is one, else the paper's corpus as the checkout ships it
# (data/heldout_corpora/renellm.jsonl); the replay itself admits either by hash
if [ ! -s "$SISTEER_HELDOUT_CORPUS_DIR/renellm.jsonl" ] && [ ! -s data/heldout_corpora/renellm.jsonl ]; then
  echo "FATAL: renellm frozen corpus missing/empty at" \
       "$SISTEER_HELDOUT_CORPUS_DIR/renellm.jsonl and at data/heldout_corpora/renellm.jsonl in $REPO_DIR — the held-out leg would" \
       "write nothing and fail validation after ~48min; refusing now" >&2
  exit 1
fi

# --- pinned serving venv (identical bootstrap to the gate) --------------------
# audit-exempt: out-of-lockfile-venv — runs from the requirements-hash
# versioned vllm serving venv (torch cu129), never the
# project lockfile; a cu128 assert would be factually wrong here and
# the silent-CPU path cannot occur (no `uv sync`; a missing venv fails
# `uv run --no-sync` / the venv python loudly).
VLLM_REQS="$REPO_DIR/configs/capeval/vllm-requirements.txt"
REQ_HASH=$(sha256sum "$VLLM_REQS" | cut -c1-8)
export UV_PROJECT_ENVIRONMENT="/home/$USER/venvs/sisteer-petri-vllm-$REQ_HASH"
export UV_CACHE_DIR="$SHARED/.cache/uv"
export UV_LINK_MODE=copy
LOCKDIR=$(dirname "$UV_PROJECT_ENVIRONMENT"); mkdir -p "$LOCKDIR"
SENTINEL="$LOCKDIR/.$(basename "$UV_PROJECT_ENVIRONMENT").vllm-bootstrap"
REQ_HASH_FULL=$(sha256sum "$VLLM_REQS" | cut -c1-16)
(
  flock -x 9 || exit 99
  if [ -f "$SENTINEL" ] && [ "$(cat "$SENTINEL")" = "$REQ_HASH_FULL" ]; then
    echo "[wave] serving venv ready @ ${REQ_HASH_FULL:0:8} (sentinel)"
  else
    [ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ] || uv venv --python 3.12 "$UV_PROJECT_ENVIRONMENT" || exit 1
    uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
       --index-strategy unsafe-best-match --exclude-newer 2026-08-03T00:00:00Z \
       -r "$VLLM_REQS" && echo "$REQ_HASH_FULL" > "$SENTINEL" || exit 1
  fi
) 9>"$LOCKDIR/.$(basename "$UV_PROJECT_ENVIRONMENT").uv-sync.lock"
RC=$?; [ "$RC" -ne 0 ] && { echo "FATAL: venv bootstrap rc=$RC" >&2; exit "$RC"; }

SERVE_VENV="$UV_PROJECT_ENVIRONMENT"
export PATH="$SERVE_VENV/bin:$PATH"    # console scripts (vllm serve) + child python

# --- the wave knobs (body-level env; NEVER --export=) ------------------------
export WAVE_MODEL="$WMODEL" WAVE_REV="$WREV"
export WAVE_CONFIG="configs/two_si_${MK}.yaml"
export WAVE_DDIR="$SHARED/exp/objects/scale-12/${MK}"
export WAVE_CELL="$WCELL" WAVE_PROMPTED="$WPROMPTED"
export WAVE_STATIC="bon,pasttense,grok420"
export WAVE_HELD_OUT="codeattack,renellm"
export WAVE_MASK="all"                 # lens-free: no object to mis-serve
# GEOMETRY: deliberately NOT exported — the driver resolves the families_served
# entry of sisteer.capeval.serving.geometry (measured 2026-09-03: the qwen
# single-shot p99 prompt 7,352 + the 4,096 cap sizes the same 11520 x 34 as
# every other family, so no Qwen-specific entry is needed), and the client
# fan-out follows the launched width. The old literal 16384 x 32 here was
# 524,288 batched tokens, 33% over the certified footprint, exported around
# the table and its budget guard. A future Qwen wave that needs a different
# geometry gets a MEASURED sub-entry in the table, not an env literal (an
# override now refuses unless WAVE_ALLOW_OVER_BUDGET=1 is exported with it).
export WAVE_BON_N_MAX=64
if [ -n "$RESUME_OUT_DIR" ]; then
  [ -d "$RESUME_OUT_DIR" ] || { echo "FATAL: RESUME_OUT_DIR $RESUME_OUT_DIR does not exist" >&2; exit 1; }
  export WAVE_OUT_DIR="$RESUME_OUT_DIR"
  # The wrapper does NOT touch the out dir (review, fifth pass): the driver
  # OWNS it under an exclusive lock (run_identity.OutDirLock) and regenerates
  # the derived artifacts — lens_registry, routing_table.json, the chat
  # template — into a staging dir, verifies the run identity, and installs them
  # by rename (a read-only registry is replaced whole; the cleanup helper makes
  # replaced files writable itself). The old `chmod -R u+w` + `rm -rf
  # lens_registry` here ran BEFORE the driver could refuse a duplicate or
  # mismatched resubmission, so a launch later rejected for lock contention or
  # identity mismatch had already deleted the incumbent run's installed registry
  # (jobs 279531-534 were the PermissionError this used to work around; the
  # staged install is the fix). sweep_out (the resumable BoN + family DATA) is
  # untouched either way.
  echo "[wave] RESUMING into $WAVE_OUT_DIR (the driver verifies the run identity under its lock; BoN + families skip covered units)"
fi

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

srun --ntasks=1 "$SERVE_VENV/bin/python" scripts/petri_families_served_integration.py
RC=$?
echo "[wave] exit=$RC ($MK/$COND)"
exit $RC
