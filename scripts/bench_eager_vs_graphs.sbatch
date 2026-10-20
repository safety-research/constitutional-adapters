#!/bin/bash
#SBATCH --job-name=vbench
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
# bench_eager_vs_graphs.sbatch <model_key> <repo_dir>
#   model_key  gemma4e4b | nemotron3nano4b
#   repo_dir   the per-sha checkout on shared storage the driver runs from
# Measures the CUDA-graph (enforce-eager) penalty and native multi-LoRA's cost
# on one H200 with bare `vllm serve` — see scripts/bench_eager_vs_graphs.py.
# Positional args only (never --export= in sbatch or srun).
set -uo pipefail
KEY="${1:?REQUIRED: gemma4e4b | nemotron3nano4b}"
REPO_DIR="${2:?REQUIRED: repo checkout dir}"

# shared root: cluster A or cluster B, derived (the same script runs on either)
SHARED="${SISTEER_SHARED_ROOT:-${CLUSTER_SHARED_ROOT:-}}"   # the front door's job env, else the cluster block's value
[ -n "$SHARED" ] && [ -d "$SHARED" ] || { echo "FATAL: set SISTEER_SHARED_ROOT to this cluster's shared root (the site config's CLUSTER_SHARED_ROOT)" >&2; exit 2; }
# HF cache + per-user token path (the SHARED copy: cluster /home is node-local)
if [ -r "$SHARED/scripts/hf_env.sh" ]; then source "$SHARED/scripts/hf_env.sh"; else
  export HF_HOME="${HF_HOME:-${SISTEER_HF_HOME:-${CLUSTER_HF_HOME:?set SISTEER_HF_HOME to the Hugging Face cache of this cluster}}}" \
    HF_TOKEN_PATH="$SHARED/.hf_token_absent"; fi
export HF_HUB_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export UV_CACHE_DIR="$SHARED/.cache/uv" UV_LINK_MODE=copy

OUT="$SHARED/exp/vbench/${KEY}_${SLURM_JOB_ID}"
mkdir -p "$OUT"
log(){ echo "[$(date -u +%FT%TZ)] $*"; }
write_result(){ echo "state=$1 exit=$2 elapsed=$SECONDS host=$(hostname) key=$KEY out=$OUT" > "$OUT/RESULT"; }

# DISARM SIGTERM FIRST (the handler re-enters itself otherwise — measured 2026-09-11)
cleanup(){ trap '' SIGTERM; kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

# fail-fast guards on every external input (a 48-minute discovery becomes an 11-second one)
[ -d "$REPO_DIR/.git" ] || [ -f "$REPO_DIR/.git" ] || { log "no git checkout at $REPO_DIR"; write_result FAILED 2; exit 2; }
[ -s "$REPO_DIR/scripts/bench_eager_vs_graphs.py" ] || { log "driver missing in $REPO_DIR"; write_result FAILED 2; exit 2; }
REQS="$REPO_DIR/configs/capeval/vllm-requirements.txt"
[ -s "$REQS" ] || { log "no serving requirements at $REQS"; write_result FAILED 2; exit 2; }
case "$KEY" in
  gemma4e4b)       SNAP="$HF_HOME/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2";;
  nemotron3nano4b) SNAP="$HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f";;
  *) log "unknown model key $KEY"; write_result FAILED 2; exit 2;;
esac
[ -s "$SNAP/config.json" ] || { log "pinned snapshot not cached: $SNAP"; write_result FAILED 2; exit 2; }
log "key=$KEY repo=$REPO_DIR sha=$(git -C "$REPO_DIR" rev-parse --short HEAD) node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES out=$OUT"

# --- the serving venv: the lanes' recipe byte for byte (scripts/sdfbase/full22/common.sh f22_serving_venv),
# so a node that already carries sisteer-petri-vllm-<hash8> reuses it and a bare node builds the same one.
H=$(sha256sum "$REQS" | cut -c1-8); HF16=$(sha256sum "$REQS" | cut -c1-16)
VENV="/home/$USER/venvs/sisteer-petri-vllm-$H"; LOCKDIR=$(dirname "$VENV"); mkdir -p "$LOCKDIR"
SENT="$LOCKDIR/.$(basename "$VENV").vllm-bootstrap"
(
  flock -x 9 || exit 99
  if [ -f "$SENT" ] && [ "$(cat "$SENT")" = "$HF16" ] && [ -x "$VENV/bin/python" ]; then
    echo "serving venv already bootstrapped ($VENV)"
  else
    [ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV" || exit 1
    uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match \
      --exclude-newer 2026-08-03T00:00:00Z -r "$REQS" && echo "$HF16" > "$SENT" || exit 1
  fi
) 9>"$LOCKDIR/.$(basename "$VENV").uv-sync.lock"
rc=$?; [ $rc -eq 0 ] || { log "serving venv bootstrap failed rc=$rc"; write_result FAILED 3; exit 3; }
export PATH="$VENV/bin:$PATH"   # `vllm` console script must be on PATH for the driver's subprocess
"$VENV/bin/python" -c 'import vllm, torch, aiohttp, safetensors, huggingface_hub; print("serving venv OK", vllm.__version__, torch.__version__, torch.cuda.is_available())' \
  || { log "serving venv guard failed"; write_result FAILED 3; exit 3; }
# the plugin must be present in the venv for the production-floor arm to mean anything
"$VENV/bin/python" -c 'import importlib.metadata as m; eps=[e for e in m.entry_points(group="vllm.general_plugins")]; assert any(e.name=="activations" for e in eps), eps; print("vllm-lens plugin entry point present")' \
  || { log "vllm-lens plugin entry point absent — the pt_eager_plugin arm would not be production"; write_result FAILED 3; exit 3; }

cd "$REPO_DIR" || exit 2
# v2 (second pass): the production floor, graphs, native LoRA at the composer's 64 cap and at the
# smallest legal cap above rank 4, two windows per point (the first pass's eager arms disagreed by
# up to 40% at one B on shared nodes), and a synthetic adapter strong enough that the probe bites.
ARMS="${3:-pt_eager_plugin,pt_graphs,lora_graphs,lora_graphs_r8}"
srun --ntasks=1 "$VENV/bin/python" scripts/bench_eager_vs_graphs.py --model-key "$KEY" --out "$OUT" \
  --arms "$ARMS" --repeats 2 --adapter-b-std 0.2 \
  --batches 8,32,64 --max-tokens 200 --warmup-s 20 --window-s 60 \
  --max-model-len 4096 --max-num-seqs 64 --max-lora-rank 64 &
WORK=$!
wait $WORK; RC=$?
if [ $RC -eq 0 ] && [ -s "$OUT/results.json" ]; then write_result DONE 0; else write_result FAILED "$RC"; fi
log "driver exit $RC; results at $OUT (RESULT: $(cat "$OUT/RESULT"))"
exit $RC
