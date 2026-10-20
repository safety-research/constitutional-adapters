#!/usr/bin/env bash
# Fan one Sobol round's trials for ONE tuning cell (arm x model x position)
# across this pod's GPUs — one trial per GPU, all in parallel, exit code =
# number of failed trials. Ported from tune_lora_all_pod.sh (tip 8ded0ea),
# plus the per-model config, ARM/POSITION env validation, and per-model
# SUB/EPOCHS defaults.
# Meant to run from the repo root under launch.sh's tmux runner on a RunPod
# pod that mounts the primed network volume (venv + HF snapshot + pool).
#
# Env knobs:
#   TRIALS_JSON  trial list from `tune_arm.py emit-trials` (required)
#   PAIRS        pool jsonl with its .tokcache.npz sidecar next to it (required)
#   OUT_DIR      trial outputs (required; on the network volume)
#   CONFIG       per-model config yaml (required — e.g.
#                configs/two_si_gemma4e4b.yaml / two_si_qwen38_27b.yaml)
#   ARM          arm name (default: the trials JSON's; mismatch refuses)
#   POSITION     all|response (default: the trials JSON's; mismatch refuses)
#   SI           tuning SI (default si_constitution)
#   START        first trial index for this pod (default 0)
#   COUNT        trials on this pod (default: one per visible GPU)
#   BS           per-trial batch size (set from the arm's memprobe; default 8)
#   EPOCHS       per-trial epoch cap (default per model: 3 E4B / 2 Qwen)
#   SUB          fit-subsample size (default per model: 6000 E4B / 3000 Qwen;
#                0 = full pool)
#   BIDPO_HARMFUL  bidpo own-pool harmful jsonl (default: tune_arm.py's
#                sibling lookup, bidpo_harmful_<SI>.jsonl beside PAIRS)
#   BIDPO_BENIGN   bidpo own-pool reversed-polarity benign jsonl (sibling
#                default bidpo_benign_<SI>.jsonl, as above)
#   PY           python (default: the primed venv on the network volume)
#
# The trials JSON names the canonical model repo id + revision it was
# emitted for (tune_arm.py emit-trials --config); this script refuses a
# CONFIG whose model does not match — a Gemma trials file on a Qwen config
# would mislabel the whole sweep (tune_arm.py trial re-checks per trial,
# config sha included).
set -uo pipefail

: "${TRIALS_JSON:?set TRIALS_JSON}"
: "${PAIRS:?set PAIRS}"
: "${OUT_DIR:?set OUT_DIR}"
: "${CONFIG:?set CONFIG (per-model yaml, e.g. configs/two_si_gemma4e4b.yaml)}"
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
SI=${SI:-si_constitution}

# One interpreter launch for all the static reads (a cold network-volume
# venv pays seconds of startup per python invocation). Model ids/revisions
# carry no whitespace, so a whitespace-split read is safe.
read -r ROUND SPEC_ARM SPEC_POS MODEL CFG_REV SPEC_MODEL SPEC_REV <<EOF
$($PY -c "import json, yaml
s = json.load(open('$TRIALS_JSON'))
c = yaml.safe_load(open('$CONFIG'))
print(s['round'], s['arm'], s.get('position', 'all'), c['model'],
      c.get('model_revision'), s.get('model'), s.get('model_revision'))")
EOF
ARM=${ARM:-$SPEC_ARM}
POSITION=${POSITION:-$SPEC_POS}
if [ "$ARM" != "$SPEC_ARM" ]; then
  echo "[pod] ARM=$ARM does not match the trials file's arm=$SPEC_ARM — refusing" >&2
  exit 2
fi
if [ "$POSITION" != "$SPEC_POS" ]; then
  echo "[pod] POSITION=$POSITION does not match the trials file's position=$SPEC_POS — refusing" >&2
  exit 2
fi
if [ "$SPEC_MODEL" != "$MODEL" ] || [ "$SPEC_REV" != "$CFG_REV" ]; then
  echo "[pod] trials file model=$SPEC_MODEL@$SPEC_REV does not match the config's $MODEL@$CFG_REV — refusing (a mismatched trials/config pair mislabels the whole sweep)" >&2
  exit 2
fi

# Per-model trial budgets: fit-subsample sizes; the EPOCHS default is
# the re-tune@32's cap of 15 (early stop patience 1 ends most trials far
# sooner — the cap exists so a slow-improving trial is not truncated by the
# old 2-3 epoch budgets, which predate the re-tune and undertrained at
# effective batch 32's ~16x fewer optimizer steps). Unknown model = no
# silent SUB default; set SUB explicitly.
case "$MODEL" in
  *[Gg]emma*[Ee]4[Bb]*) SUB=${SUB:-6000}; EPOCHS=${EPOCHS:-15} ;;
  *[Qq]wen*)            SUB=${SUB:-3000}; EPOCHS=${EPOCHS:-15} ;;
  *)
    if [ -z "${SUB:-}" ]; then
      echo "[pod] unrecognized model '$MODEL' — set SUB (fit-subsample) explicitly" >&2
      exit 2
    fi
    EPOCHS=${EPOCHS:-15}
    ;;
esac
mkdir -p "$OUT_DIR"

# bidpo own-pool overrides (F1): forwarded only when set — tune_arm.py
# otherwise resolves the emit-pairs siblings beside PAIRS itself.
declare -a bidpo_args=()
[ -n "${BIDPO_HARMFUL:-}" ] && bidpo_args+=(--bidpo-harmful-pairs "$BIDPO_HARMFUL")
[ -n "${BIDPO_BENIGN:-}" ] && bidpo_args+=(--bidpo-benign-pairs "$BIDPO_BENIGN")

echo "[pod] cell=$ARM/$MODEL/$POSITION si=$SI round=$ROUND trials $START..$((START+COUNT-1)) on $NGPU GPUs, bs=$BS epochs=$EPOCHS sub=$SUB"
declare -a pids=()
for i in $(seq 0 $((COUNT-1))); do
  idx=$((START+i)); gpu=$((i % NGPU))
  extra=""
  [ "$i" -eq 0 ] && extra="--profile"   # one TB/profiler trace per pod
  log="$OUT_DIR/trial_r${ROUND}_t$(printf %02d "$idx").log"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/tune_arm.py trial \
      --config "$CONFIG" --trials "$TRIALS_JSON" --index "$idx" \
      --pairs "$PAIRS" --out "$OUT_DIR" --si "$SI" \
      --arm "$ARM" --positions "$POSITION" \
      --batch-size "$BS" --epochs "$EPOCHS" \
      --fit-subsample "$SUB" \
      ${bidpo_args[@]+"${bidpo_args[@]}"} $extra \
      >"$log" 2>&1 &
  pids+=($!)
  sleep 20   # stagger model loads off the shared volume
done

fails=0
for p in "${pids[@]}"; do wait "$p" || fails=$((fails + 1)); done
echo "[pod] round $ROUND complete: $((COUNT - fails))/$COUNT trials ok, $fails failed"
exit "$fails"
