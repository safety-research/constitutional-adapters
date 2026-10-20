#!/usr/bin/env bash
# Jailbreak DSR for one fit variant (all|resp), fanned across this pod's GPUs.
# Static pod-runnable families only (no attacker model, no transformers<4.56
# sub-env). Generation (GPU) round-robins cells across GPUs; aggregation
# (API, low-prio key file) runs per family after generation.
#
# fit-all  : every cell (baselines + LoRA) served all-positions via two_si_run.
# fit-resp : baselines via two_si_run (positions all — no maskable object);
#            the 3 LoRA cells via jailbreak_run --positions response, so the
#            response-fit object is served response-masked (two_si_run would
#            silently reset it to all-positions — jailbreak.py _set_response_
#            positions). Both write the same cell-id-keyed private_run dirs the
#            aggregate reads. Alpha stays 1.0 so the served-alpha gate tolerates
#            the missing cell_meta on the jailbreak-generated cells.
#
# Env: VARIANT (all|resp, required), D (lora_all_cells root, required), CONC (150).
set -uo pipefail
VARIANT=${VARIANT:?set VARIANT=all|resp}
D=${D:?set D to the lora_all_cells root on this pod (the directory holding two_si_<variant>.yaml)}
PY=/workspace/venvs/sisteer/bin/python
CFG=$D/two_si_${VARIANT}.yaml
PRIV=$D/priv
CONC=${CONC:-150}
BONB=${BONB:-64}   # batched-BoN GPU width (~7x vs width-1); non-bon families ignore it
export HF_HOME=/workspace/hf TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/repos/lora
mkdir -p "$PRIV"
NGPU=$(nvidia-smi -L | wc -l)
FAMS="bon pasttense pap encoding aim gcg_replay"
BASELINES="none prompted__safety_refusal prompted__si_constitution prompted__si_irrelevant"
LORA="safety_refusal__lora_r1_all si_constitution__lora_r1_all si_irrelevant__lora_r1_all"

gen_baseline() { # family cell gpu
  env CUDA_VISIBLE_DEVICES=$3 $PY scripts/two_si_run.py generate \
    --config "$CFG" --attack "$1" --cell "$2" --private-root "$PRIV" \
    --bon-batch-size "$BONB" \
    >"$D/dsr_${VARIANT}_${1}_${2}.log" 2>&1
}
gen_lora() { # family cell gpu
  if [ "$VARIANT" = resp ]; then
    env CUDA_VISIBLE_DEVICES=$3 $PY scripts/jailbreak_run.py generate \
      --config "$CFG" --attack "$1" --defense "$2" --positions response \
      --private-root "$PRIV" --bon-batch-size "$BONB" \
      >"$D/dsr_${VARIANT}_${1}_${2}.log" 2>&1
  else
    gen_baseline "$1" "$2" "$3"
  fi
}
xstest_cell() { # cell is_lora gpu
  if [ "$VARIANT" = resp ] && [ "$2" = lora ]; then
    env CUDA_VISIBLE_DEVICES=$3 $PY scripts/jailbreak_run.py generate \
      --config "$CFG" --attack bon --defense "$1" --xstest --positions response \
      --private-root "$PRIV" >"$D/dsr_${VARIANT}_xstest_${1}.log" 2>&1
  else
    env CUDA_VISIBLE_DEVICES=$3 $PY scripts/two_si_run.py generate \
      --config "$CFG" --attack bon --cell "$1" --xstest \
      --private-root "$PRIV" >"$D/dsr_${VARIANT}_xstest_${1}.log" 2>&1
  fi
}

# ---- GENERATE: per-GPU-slot scheduler (no wave barrier) --------------------
# Launch the next job the INSTANT any GPU frees, instead of waiting for a whole
# wave of NGPU to finish (which idled the fast refusal-heavy cells while the
# slow none/prompted cells ran). A free-GPU pool + wait -n recycles each slot
# as its job exits. Every job function takes its GPU id as its LAST arg.
fails=0
FREE=($(seq 0 $((NGPU-1))))
declare -A PID2GPU
reclaim() {  # move any exited jobs' GPUs back to the free pool, count failures
  local pid
  for pid in "${!PID2GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      if wait "$pid"; then :; else fails=$((fails+1)); fi
      FREE+=("${PID2GPU[$pid]}"); unset "PID2GPU[$pid]"
    fi
  done
}
submit() {  # submit <fn> <args...> — a GPU id is appended as the final arg
  while [ ${#FREE[@]} -eq 0 ]; do wait -n 2>/dev/null || true; reclaim; done
  local gpu=${FREE[0]}; FREE=("${FREE[@]:1}")
  "$@" "$gpu" &
  PID2GPU[$!]=$gpu
}
for F in $FAMS; do
  for C in $BASELINES; do submit gen_baseline "$F" "$C"; done
  for C in $LORA;      do submit gen_lora     "$F" "$C"; done
done
for C in $BASELINES; do submit xstest_cell "$C" base; done
for C in $LORA;      do submit xstest_cell "$C" lora; done
# Drain remaining jobs.
while [ ${#PID2GPU[@]} -gt 0 ]; do wait -n 2>/dev/null || true; reclaim; done
echo "DSR-GENERATE-DONE ($VARIANT): $fails generation job(s) failed"

# ---- AGGREGATE (live grade, low-prio key file): one per family ----
for F in $FAMS; do
  $PY scripts/two_si_run.py aggregate --config "$CFG" --attack "$F" \
    --transport live --concurrency "$CONC" --private-root "$PRIV" \
    >"$D/dsr_${VARIANT}_aggregate_${F}.log" 2>&1 || echo "AGG-FAIL $F"
done
echo "DSR-EVAL-DONE ($VARIANT)"
