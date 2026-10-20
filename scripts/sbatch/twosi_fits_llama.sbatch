#!/bin/bash
#SBATCH --job-name=tt-twosifit-llama-resp  # SESSION-UNIQUE: add a tag
#                                          # BEFORE the leg suffix, e.g.
#                                          # -J tt-twosifit-llama-<tag>-resp.
#                                          # The LAST dash token selects the
#                                          # leg (resp|tf). Cancel via
#                                          # stop-run.sh, never bare scancel.
#SBATCH --partition=general
#SBATCH --qos=high                    # the lane default; 16-GPU high budget
#SBATCH --gres=gpu:3                  # RESP-wave sizing. The tf wave
#SBATCH --cpus-per-task=48            # overrides on the CLI:
#SBATCH --mem=384G                    #   --gres=gpu:1 --cpus-per-task=16
#                                     #   --mem=128G   (never --mem=0)
#SBATCH --time=1-00:00:00             # ~2x the estimate: full pool 9430
#                                     # train rows at bs2/16384, epoch cap 6
#                                     # with early stop (patience 2) — re-time
#                                     # from the first finished cell.
#SBATCH --array=0-2                   # SI index: safety_refusal,
#                                     # si_constitution, si_irrelevant
# JB4: fit stdout can carry pair text on error paths -> private 0700 log dir
# (pre-created by scripts/stage_llama_fits.sh; slurm does not mkdir parents).
#SBATCH --output=/workspace-vast/%u/exp/twosweep/fits/_logs/%x_%A_%a.out
#SBATCH --error=/workspace-vast/%u/exp/twosweep/fits/_logs/%x_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --no-requeue                  # self-resubmit below owns retries
#SBATCH --signal=B:TERM@120
#
# twosi-model-sweeps LLAMA FIT waves (cluster-only — Qwen/E4B fits run on
# Modal, scripts/modal/twosi_fits.py). 12 fits = 3 SIs x
# {psr_kl_all,psr_mse_all,lora_r1_all @ response; token_faithful @ all}:
#
#   resp wave: one array task per SI, 3 GPUs, config two_si_llama_fits_dp3
#              .yaml (data_parallel.gpus: 3 — run_parallel_fit packs the 3
#              arms 1-per-GPU and REFUSES any other pool width)
#   tf wave:   one array task per SI, 1 GPU, config two_si_llama_fits.yaml
#              (dp-free single-process path), --out its OWN store (Class A
#              cell ids collide with Class C otherwise — the driver refuses)
#
# GATES before submitting (stage_llama_fits.sh checks + prints these):
#   1. THIS LEG's arms' Llama HPO winners folded (make_fit_configs.py —
#      partial folds are fine, each cell's HPO lands on its own clock),
#      committed, checkout staged. The gate checks ONLY $ARMS
#      (scripts/check_fit_config_folded.py), so the tf wave launches
#      while psr/lora still pend and vice versa; an unfolded arm in THIS
#      leg FAILS here in ms.
#   2. Pool + sidecars rsynced to $SHARED/exp/twosweep/pairs/llama_v3/.
# Submit (from the grading host, after stage_llama_fits.sh):
#   ssh <login-host> 'sbatch -J tt-twosifit-llama-<tag>-resp \
#     $SHARED/checkouts/<sha>/scripts/sbatch/twosi_fits_llama.sbatch'
#   ssh <login-host> 'sbatch -J tt-twosifit-llama-<tag>-tf --gres=gpu:1 \
#     --cpus-per-task=16 --mem=128G \
#     $SHARED/checkouts/<sha>/scripts/sbatch/twosi_fits_llama.sbatch'
# Then ARM A WATCHER on each array job id (never leave it unwatched).
#
# Resume: the harness's fingerprint-validated COMPLETE markers — resubmit
# the same line and finished cells self-skip. Stop for good by writing the
# STOP sentinel for the job name (a bare scancel looks like preemption to
# the trap below and resubmits).

set -uo pipefail
ATTEMPT=${1:-1}
MAX_ATTEMPTS=${2:-3}
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
source "$SHARED/scripts/hf_env.sh"
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# W&B: compute nodes cannot reach api.wandb.ai — offline run dirs under the
# shared root; `wandb sync $SHARED/exp/twosweep/wandb/latest-run` (the grading host /
# login node) after the wave. Console capture OFF (JB4: pair text).
export WANDB_MODE=offline
export WANDB_DIR=$SHARED/exp/twosweep/wandb
export WANDB_CONSOLE=off
# the site's W&B entity, which the printed submission carries as SISTEER_WANDB_ENTITY (scripts/stage_llama_fits.sh), is the entity
# the offline runs are filed under (train_two_si_objects.py reads WANDB_ENTITY; one the job env sets itself wins)
[ -n "${WANDB_ENTITY:-}" ] || [ -z "${SISTEER_WANDB_ENTITY:-}" ] || export WANDB_ENTITY="$SISTEER_WANDB_ENTITY"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "$WANDB_DIR"

FITROOT=$SHARED/exp/twosweep/fits
# audit-exempt: out-of-lockfile-venv — this job runs from the PREBAKED
# twosweep-fit venv on shared storage (pod-fanout convention: the env is
# built once and shipped, never uv-synced per job), so the frozen-cu128-sync
# guard is factually wrong here and the 61380 silent-CPU path cannot occur
# (no sync happens; a missing venv fails the srun loudly). The runtime
# CUDA assert below still guards a CPU-flavored prebake.
PY=$SHARED/envs/twosweep-fit/bin/python
if ! "$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available in this env"'; then
  echo "FATAL: $PY has no CUDA torch (prebaked venv is CPU-flavored?)" >&2
  exit 92
fi

# --- leg from the job name's LAST dash token (no --export on this cluster)
LEG=${SLURM_JOB_NAME##*-}
case "$LEG" in
  resp) CFG_REL=configs/two_si_llama_fits_dp3.yaml
        ARMS=psr_kl_all,psr_mse_all,lora_r1_all
        POSFLAGS=(--positions response)
        OUT=$FITROOT/llama_v3/cells_response
        RESUB_RES=(--gres=gpu:3 --cpus-per-task=48 --mem=384G) ;;
  tf)   CFG_REL=configs/two_si_llama_fits.yaml
        ARMS=token_faithful
        POSFLAGS=()
        OUT=$FITROOT/llama_v3/cells_all
        RESUB_RES=(--gres=gpu:1 --cpus-per-task=16 --mem=128G) ;;
  *) echo "FATAL: job name '$SLURM_JOB_NAME' must end in -resp or -tf" >&2
     exit 1 ;;
esac

SIS=(safety_refusal si_constitution si_irrelevant)
SI=${SIS[$SLURM_ARRAY_TASK_ID]}

# --- pinned checkout + config identity (pointer files, stage_llama_fits.sh)
SHA=$(tr -d '[:space:]' < "$FITROOT/CHECKOUT_SHA" 2>/dev/null || true)
if [ -z "$SHA" ]; then
  echo "FATAL: $FITROOT/CHECKOUT_SHA missing — run scripts/stage_llama_fits.sh first" >&2
  exit 1
fi
CK=${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}/checkouts/$SHA
if [ "$(git -C "$CK" rev-parse HEAD 2>/dev/null)" != "$SHA" ]; then
  echo "FATAL: checkout sha drift at $CK" >&2
  exit 1
fi
CFG=$CK/$CFG_REL
# Per-LEG sentinel gate: only the arms THIS invocation trains must be
# folded (a whole-file grep blocked the tf wave on psr/lora sentinels it
# never reads — partial folds are the normal case).
if ! "$PY" "$CK/scripts/check_fit_config_folded.py" "$CFG" "$ARMS"; then
  exit 1
fi
EXPECT_SHA=$(sha256sum "$CFG" | cut -c1-16)   # the checkout gate above pins
                                              # the tree; this pins the leg's
                                              # exact config bytes into the
                                              # cell fingerprints.

PAIRS=$SHARED/exp/twosweep/pairs/llama_v3/pairs_${SI}.jsonl
if [ ! -s "$PAIRS" ] || [ ! -s "$PAIRS.tokcache.npz" ]; then
  echo "FATAL: pool or sidecar missing at $PAIRS(.tokcache.npz) — rsync the" \
       "v3 pool + sidecars before submitting (GATE 2)." >&2
  exit 1
fi
install -d -m 700 "$OUT" "$FITROOT/llama_v3"

# --- self-resubmit on SIGTERM (walltime/node failure; high is never
# preempted). STOP sentinel (stop-run.sh) makes a cancel final.
STOP_DIR=$SHARED/exp/training/$SLURM_JOB_NAME
mkdir -p "$STOP_DIR"
SCRIPT_PATH=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
              | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)
CHILD=""
resubmit() {
  if [ -f "$STOP_DIR/STOP" ]; then
    echo "[fits] STOP sentinel present — not resubmitting"; return
  fi
  if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
    echo "[fits] attempt $ATTEMPT/$MAX_ATTEMPTS exhausted — not resubmitting" >&2
    return
  fi
  echo "[fits] SIGTERM at attempt $ATTEMPT — resubmitting task $SLURM_ARRAY_TASK_ID"
  sbatch -J "$SLURM_JOB_NAME" --array="$SLURM_ARRAY_TASK_ID" \
    "${RESUB_RES[@]}" "${SCRIPT_PATH:-$CK/scripts/sbatch/twosi_fits_llama.sbatch}" \
    $((ATTEMPT + 1)) "$MAX_ATTEMPTS" || echo "[fits] RESUBMIT FAILED" >&2
}
on_term() {
  resubmit
  [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null
  kill -TERM -$$ 2>/dev/null
  wait
  exit 143
}
trap on_term SIGTERM SIGINT SIGQUIT
# Self-announcing result + figure mirror (fitmon writes metrics, not
# figures, but the RESULT line rides the same mirror).
export FIGS_DIR=$SHARED/exp/figs/$SLURM_JOB_NAME
mkdir -p "$FIGS_DIR"
finish() {
  rc=$?
  echo "$SLURM_JOB_NAME task=$SLURM_ARRAY_TASK_ID si=$SI leg=$LEG exit=$rc attempt=$ATTEMPT $(date -u +%FT%TZ)" \
    >> "$FIGS_DIR/RESULT"
  "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" || true
}
trap finish EXIT

cd "$CK"
echo "[fits] node=$(hostname) si=$SI leg=$LEG arms=$ARMS cfg=$CFG_REL" \
     "config_sha=$EXPECT_SHA gpus=$CUDA_VISIBLE_DEVICES start=$(date -u +%FT%TZ)"
srun --kill-on-bad-exit=1 "$PY" scripts/train_two_si_objects.py train \
  --config "$CFG" --si "$SI" --arms "$ARMS" "${POSFLAGS[@]}" \
  --pairs "$PAIRS" --token-cache auto \
  --out "$OUT" --expect-config-sha "$EXPECT_SHA" &
CHILD=$!
wait $CHILD
RC=$?
echo "[fits] si=$SI leg=$LEG exit=$RC end=$(date -u +%FT%TZ)"
exit $RC
