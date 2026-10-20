#!/bin/bash
#SBATCH --job-name=sdfmt-petri
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=0-12:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# sdfbase-mt: ONE base model's Petri audit PROXY:
# scripts/sdfbase/mt_serving_up.py --mode petri — one vLLM+lens
# server + RoutingProxy serving every cell of the wave (+ the derived none__lens0
# control leg) at the certified Petri geometry (16384 x 24), agent-loop entry
# declared, tool calls on. Loopback-only; the Petri driver on the grading host
# ($STAGE/sdfbase/mt/petri_drive.sh) tunnels in through the login node,
# reads <out>/endpoint.json, and touches <out>/STOP when every leg is complete.
# Re-submit while legs remain (the driver resumes by run key; walltime expiry is
# a WATCHED terminal state, never silent).
# Usage (positional args only — never --export=):
#   sbatch -J sdfmt-petri-<slug> scripts/sdfbase/mt/petri_proxy.sbatch <slug> <wave_sha> <wave>
set -uo pipefail
SLUG="${1:?slug}"; WAVE_SHA="${2:?wave sha}"; WAVE="${3:?wave}"; REDUCTION_MODE="${4:-per_row}"
# REDUCTION MODE (positional 4, default per_row) — see attack_wave.sbatch: mt1 is a per_row wave and
# every re-run of it must stay per_row (the stacked reductions are not bit-identical).
case "$REDUCTION_MODE" in per_row) export SISTEER_LENS_STACKED_REDUCTIONS=0 ;; stacked) export SISTEER_LENS_STACKED_REDUCTIONS=1 ;; *) echo "FATAL: bad reduction mode $REDUCTION_MODE"; exit 1 ;; esac
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
source "$SHARED/scripts/hf_env.sh"; export HF_HUB_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export UV_CACHE_DIR="$SHARED/.cache/uv" UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=$SHARED/.uv-pythons
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
log(){ echo "[$(date -Is)] $*"; }
case "$SLUG" in   # QWEN ONLY
  qwen35_4b) PROXY_PORT=8974 ;;
  *) log "FATAL: unknown slug $SLUG (this wave is qwen35_4b only)"; exit 1 ;;
esac
REPO_DIR="${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA"
[ -d "$REPO_DIR/src/sisteer" ] || { log "FATAL: no checkout at $REPO_DIR"; exit 1; }
cd "$REPO_DIR"
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
[ "$HEAD_SHA" = "$WAVE_SHA" ] || { log "FATAL: HEAD $HEAD_SHA != $WAVE_SHA"; exit 1; }
git diff --quiet HEAD -- || { log "FATAL: dirty checkout $REPO_DIR"; exit 1; }
CFG=configs/sdfbase_transfer_${SLUG}_${WAVE}.yaml
# the manifest follows the wave label: mt1 -> petri_sdfbase_<slug>.yaml; a scratch wave (mt1smoke)
# -> petri_sdfbase_<slug>_<wave>.yaml (the standup refuses a manifest/config cell mismatch)
MANIFEST=configs/capeval/manifests/petri_sdfbase_${SLUG}$( [ "$WAVE" = mt1 ] || echo "_$WAVE" ).yaml
export TT_TWO_SI_DEFENSES_DIR=$SHARED/exp/objects/sdfbase/$WAVE/$SLUG/cells
OUT=$SHARED/exp/sdfbase_mt/$WAVE/petri/$SLUG/serving
mkdir -p "$OUT"; rm -f "$OUT/STOP" "$OUT/endpoint.json"
VLLM_REQS="$REPO_DIR/configs/capeval/vllm-requirements.txt"
REQ_HASH=$(sha256sum "$VLLM_REQS" | cut -c1-8); REQ_HASH_FULL=$(sha256sum "$VLLM_REQS" | cut -c1-16)
SERVE_VENV="/home/$USER/venvs/sisteer-petri-vllm-$REQ_HASH"
LOCKDIR=$(dirname "$SERVE_VENV"); mkdir -p "$LOCKDIR"
SSENT="$LOCKDIR/.$(basename "$SERVE_VENV").vllm-bootstrap"
(
  flock -x 9 || exit 99
  if [ -f "$SSENT" ] && [ "$(cat "$SSENT")" = "$REQ_HASH_FULL" ] && [ -x "$SERVE_VENV/bin/python" ]; then
    log "serving venv already bootstrapped"
  else
    [ -x "$SERVE_VENV/bin/python" ] || uv venv --python 3.12 "$SERVE_VENV" || exit 1
    uv pip install --python "$SERVE_VENV/bin/python" --index-strategy unsafe-best-match --exclude-newer 2026-08-03T00:00:00Z -r "$VLLM_REQS" && echo "$REQ_HASH_FULL" > "$SSENT" || exit 1
  fi
) 9>"$LOCKDIR/.$(basename "$SERVE_VENV").uv-sync.lock"
SPY="$SERVE_VENV/bin/python"
[ -x "$SPY" ] || { log "FATAL: serving venv bootstrap failed"; exit 1; }
unset VLLM_PLUGINS 2>/dev/null || true
# vs.launch_server shells out to `vllm`: the serving venv's bin/ must lead PATH (smoke jobs 375475/375476)
export PATH="$SERVE_VENV/bin:$PATH"
printf '{"job_id": "%s", "node": "%s", "slug": "%s", "sha": "%s", "started": %s}\n' "$SLURM_JOB_ID" "$(hostname)" "$SLUG" "$WAVE_SHA" "$(date +%s)" >> "$OUT/proxy_jobs.jsonl"
cleanup(){ touch "$OUT/STOP" 2>/dev/null || true; kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT
log "petri proxy $SLUG @ $WAVE_SHA on $(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?} proxy_port=$PROXY_PORT reduction=$REDUCTION_MODE (SISTEER_LENS_STACKED_REDUCTIONS=$SISTEER_LENS_STACKED_REDUCTIONS) bank=$TT_TWO_SI_DEFENSES_DIR"
"$SPY" scripts/sdfbase/mt_serving_up.py --repo-dir "$REPO_DIR" --config "$CFG" --manifest "$MANIFEST" \
  --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out "$OUT" --stop-file "$OUT/STOP" --mode petri --model-key "$SLUG" --reduction-mode "$REDUCTION_MODE" \
  --proxy-port "$PROXY_PORT" &
wait $!; rc=$?
log "petri proxy exited rc=$rc"
exit $rc
