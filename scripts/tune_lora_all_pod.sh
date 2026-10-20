#!/usr/bin/env bash
# Fan one Sobol round's lora_r1_all trials across this pod's GPUs — one
# trial per GPU, all in parallel, exit code = number of failed trials.
# Meant to run from the repo root under launch.sh's tmux runner on a RunPod
# pod that mounts the primed network volume (venv + HF snapshot + pool).
#
# Env knobs (defaults for the AP-JP-1 volume layout):
#   TRIALS_JSON  trial list from `tune_lora_all.py emit-trials` (required)
#   PAIRS        pool jsonl with its .tokcache.npz sidecar next to it (required)
#   OUT_DIR      trial outputs (required; on the network volume)
#   START        first trial index for this pod (default 0)
#   COUNT        trials on this pod (default: one per visible GPU)
#   BS           per-trial batch size (default 8; set from probe-mem)
#   EPOCHS       per-trial epoch cap (default 3)
#   SUB          fit-subsample size (default 6000; 0 = full pool)
#   PY           python (default: the venv python on the mounted volume)
set -uo pipefail

: "${TRIALS_JSON:?set TRIALS_JSON}"
: "${PAIRS:?set PAIRS}"
: "${OUT_DIR:?set OUT_DIR}"
PY=${PY:-/workspace/venvs/sisteer/bin/python}
export HF_HOME=${HF_HOME:-/workspace/hf}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1   # heartbeat/phase lines must land in the log live

NGPU=$(nvidia-smi -L | wc -l)
START=${START:-0}
COUNT=${COUNT:-$NGPU}
BS=${BS:-8}
EPOCHS=${EPOCHS:-3}
SUB=${SUB:-6000}
ROUND=$($PY -c "import json,sys;print(json.load(open('$TRIALS_JSON'))['round'])")
mkdir -p "$OUT_DIR"

echo "[pod] round=$ROUND trials $START..$((START+COUNT-1)) on $NGPU GPUs, bs=$BS epochs=$EPOCHS"
declare -a pids=()
for i in $(seq 0 $((COUNT-1))); do
  idx=$((START+i)); gpu=$((i % NGPU))
  extra=""
  [ "$i" -eq 0 ] && extra="--profile"   # one TB/profiler trace per pod
  log="$OUT_DIR/trial_r${ROUND}_t$(printf %02d "$idx").log"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/tune_lora_all.py trial \
      --trials "$TRIALS_JSON" --index "$idx" --pairs "$PAIRS" \
      --out "$OUT_DIR" --batch-size "$BS" --epochs "$EPOCHS" \
      --fit-subsample "$SUB" $extra \
      >"$log" 2>&1 &
  pids+=($!)
  sleep 20   # stagger model loads off the shared volume
done

fails=0
for p in "${pids[@]}"; do wait "$p" || fails=$((fails + 1)); done
echo "[pod] round $ROUND complete: $((COUNT - fails))/$COUNT trials ok, $fails failed"
exit "$fails"
