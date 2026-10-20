#!/usr/bin/env bash
# Over-refusal (or_bench_hard_1k, 1319) + full-n GSM8K (~1209) for every cell,
# fanned across this pod's GPUs. Two lessons baked in from the first attempt:
#   1. Use the NARROWED config ($D/two_si_all.yaml) so lora_r1_all resolves as
#      a cell — configs/two_si.yaml's 13-arm registry has no lora_r1_all.
#   2. Run GSM8K (API-free, the priority full-n confirm) SEPARATELY from
#      or_bench (API-graded), to separate out-roots, so a flaky over-refusal
#      judge can never block the capability result — and so neither writes a
#      summary the other's failure would abort.
# Fail-loud: tracks per-(cell,leg) failures and exits nonzero if any fail
# (the first run exited 0 with zero summaries — never again).
# Env: D (the lora_all_cells root, required), CONC_OREF (or_bench judge concurrency, default 60).
set -uo pipefail
D=${D:?set D to the lora_all_cells root on this pod (the directory holding two_si_all.yaml)}
PY=/workspace/venvs/sisteer/bin/python
CFG=$D/two_si_all.yaml            # narrowed: class_a_arms=[lora_r1_all]
CONC_OREF=${CONC_OREF:-60}
LEGS=${LEGS:-both}   # gsm8k | orbench | both — decouple so a failing judge can't starve gsm8k
export HF_HOME=/workspace/hf TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/repos/lora
[ -f "$CFG" ] || $PY scripts/mk_eval_cfgs.py "$D"   # ensure narrowed configs exist
NGPU=$(nvidia-smi -L | wc -l)

CELLS=(
 "none:cells_all:all" "prompted__safety_refusal:cells_all:all"
 "prompted__si_constitution:cells_all:all" "prompted__si_irrelevant:cells_all:all"
 "safety_refusal__lora_r1_all:cells_all:all" "si_constitution__lora_r1_all:cells_all:all"
 "si_irrelevant__lora_r1_all:cells_all:all"
 "safety_refusal__lora_r1_all:cells_resp:generated"
 "si_constitution__lora_r1_all:cells_resp:generated"
 "si_irrelevant__lora_r1_all:cells_resp:generated"
)

# One cell's GSM8K (API-free) then or_bench (API) — separate out-roots.
run_cell() { # cell ddir spos gpu
  local cell=$1 ddir=$2 spos=$3 gpu=$4 rc=0
  local tag="${cell}_${ddir}"
  local caproot oref
  caproot=$([ "$ddir" = cells_resp ] && echo "$D/cap_resp" || echo "$D/cap")
  oref=$([ "$ddir" = cells_resp ] && echo "$D/oref_resp" || echo "$D/oref")
  # -- GSM8K full-n, API-free (the priority) --  LEGS gates each leg so the
  # slow/failing or_bench judge can never starve the gsm8k priority again
  # (LEGS=gsm8k runs only this; LEGS=orbench only the next; both = legacy).
  if [ "$LEGS" = gsm8k ] || [ "$LEGS" = both ]; then
    if env CUDA_VISIBLE_DEVICES=$gpu $PY scripts/cell_controls_run.py generate \
         --config "$CFG" --cell "$cell" --controls gsm8k --n-gsm8k 1300 \
         --defenses-dir "$D/$ddir" --steer-positions "$spos" --out-root "$caproot" \
         >"$D/ctl_gsm8k_${tag}.log" 2>&1 \
       && env CUDA_VISIBLE_DEVICES=$gpu $PY scripts/cell_controls_run.py judge \
         --config "$CFG" --cell "$cell" --controls gsm8k \
         --defenses-dir "$D/$ddir" --steer-positions "$spos" --out-root "$caproot" \
         >>"$D/ctl_gsm8k_${tag}.log" 2>&1; then :; else echo "FAIL gsm8k $tag"; rc=1; fi
  fi
  # -- or_bench over-refusal, API-graded (lower concurrency for robustness) --
  if [ "$LEGS" = orbench ] || [ "$LEGS" = both ]; then
    if env CUDA_VISIBLE_DEVICES=$gpu $PY scripts/cell_controls_run.py generate \
         --config "$CFG" --cell "$cell" --controls or_bench_hard_1k \
         --defenses-dir "$D/$ddir" --steer-positions "$spos" --out-root "$oref" \
         >"$D/ctl_oref_${tag}.log" 2>&1 \
       && env CUDA_VISIBLE_DEVICES=$gpu $PY scripts/cell_controls_run.py judge \
         --config "$CFG" --cell "$cell" --controls or_bench_hard_1k \
         --defenses-dir "$D/$ddir" --steer-positions "$spos" \
         --concurrency "$CONC_OREF" --out-root "$oref" \
         >>"$D/ctl_oref_${tag}.log" 2>&1; then :; else echo "FAIL oref $tag"; rc=1; fi
  fi
  return $rc
}

i=0; fails=0; declare -a pids=()
for spec in "${CELLS[@]}"; do
  IFS=: read -r cell ddir spos <<< "$spec"
  run_cell "$cell" "$ddir" "$spos" $((i % NGPU)) & pids+=($!); i=$((i+1))
  if [ $(( i % NGPU )) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p" || fails=$((fails+1)); done; pids=(); fi
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && { wait "$p" || fails=$((fails+1)); }; done
echo "CONTROLS-EVAL-DONE: $fails cell-legs failed (see FAIL lines above / ctl_*.log)"
exit "$fails"
