#!/bin/bash
#SBATCH --job-name=mlcert
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
# multilora_cert.sbatch <target_key> <repo_dir> [resume_from]
#   target_key   the certification ROW (multilora_cert.TARGETS; the out dir <key>_<jobid>):
#                gemma4e4b | qwen35_4b | nemotron3nano4b | llama8b | llama8b_subtraction
#                — the two llama rows certify ONE model (MODEL_KEY=llama8b: the snapshot,
#                the bank view, pins.json, the venv) with different cells;
#                gemma4e4b_psr | qwen35_4b_psr | nemotron3nano4b_psr | llama8b_psr — the
#                stage-1 native PSR serving rows: the roster main-view all-positions PSR
#                cell beside the target's LoRA cell (EXTRA_CELLS), numerics NOT measured
#                (no probe legs, no lens per_row leg), llama8b on the certification-only
#                `psrcert` view;
#                gemma4e4b_psr2cert — the stage-2 native PSR serving row: the COMBINED
#                certification-only `psr2cert` view, a RESPONSE-fit PSR cell beside the ARDITI
#                static_r object and the LoRA twin. Every cell's object is pinned
#                here before the venv — the Arditi one through its own authority's pins file,
#                named by a QUALIFIED pin key (`<file>:<model key>.<cell>`);
#                qwen35_4b_paired — THE PAIRED WRITE: the
#                certification-only `pairedcert` view, the batch-64 seed-1 fitted
#                paired-write object (kind psr_paired_write, the slot's second gate term)
#                pinned by configs/reseed_paired_pins.json (a QUALIFIED key on the PRIMARY
#                cell), beside the qwen LoRA twin; numerics NOT measured;
#                llama8b_paired — the paired write's second round: llama8b's fitted teacher
#                (the wave pin derived__constv2-mix-full__psr_ce_all_minus_ctl, a BARE key) on
#                llama8b's certification-only `pairedcert` view beside its LoRA twin.
#   repo_dir     the per-sha checkout on shared storage the driver runs from
#                ($SHARED/sisteer-eval-repo/<sha>; HEAD must equal <sha>)
#   resume_from  a prior record dir: prep/oracle reused after a binding check, a
#                COMPLETE unit reused whole under its original provenance
# ONE native-multi-LoRA certification: the fp32 oracle child, the lora-native
# standup + five controls + the gate canary, probes, throughput, then the two
# lens standups (stacked, per_row; the driver names serving_path=lens
# EXPLICITLY — the standup default is native since 2026-09-13) — all on ONE GPU
# in ONE job, so the head-to-head is a same-allocation comparison. The verdict
# is the FUNCTIONAL rule: readiness attested + five controls + gate canary +
# not slower at S; the greedy-token numerics are recorded, never a gate.
# Modelled on scripts/bench_eager_vs_graphs.sbatch: the lanes' venv recipe,
# hf_env.sh, fail-fast guards, the disarm-first trap.
# Positional args only (every value travels positionally, never through
# sbatch's environment-export flag). Submit as
#   sbatch -J mlcert-<key> scripts/sdfbase/multilora_cert.sbatch <key> <repo_dir>
# and pass --time on the submit line when the header's 02:00:00 is short — an
# 8B unit (either llama row) wants 03:00:00:
#   sbatch -J mlcert-llama8b --time=03:00:00 scripts/sdfbase/multilora_cert.sbatch llama8b <repo_dir>
set -uo pipefail
KEY="${1:?REQUIRED: gemma4e4b | qwen35_4b | nemotron3nano4b | llama8b | llama8b_subtraction | gemma4e4b_psr | qwen35_4b_psr | nemotron3nano4b_psr | llama8b_psr | gemma4e4b_psr2cert | qwen35_4b_paired | llama8b_paired}"
REPO_DIR="${2:?REQUIRED: repo checkout dir}"
RESUME="${3:-}"

# shared root and HF cache: the job environment's SISTEER_SHARED_ROOT and SISTEER_HF_HOME (the front door exports both
# from the cluster's site block; the same script runs on either cluster)
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
# HF cache + per-user token path (the SHARED copy: cluster /home is node-local)
if [ -r "$SHARED/scripts/hf_env.sh" ]; then source "$SHARED/scripts/hf_env.sh"; else
  export HF_HOME="${SISTEER_HF_HOME:?set SISTEER_HF_HOME, the HF cache of the cluster (the front door exports it from the site config)}" HF_TOKEN_PATH="$SHARED/.hf_token_absent"; fi
export HF_HUB_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export UV_CACHE_DIR="$SHARED/.cache/uv" UV_LINK_MODE=copy
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# no W&B and no key on the cluster: the job writes JSON only; the grading host report logs it
export WANDB_MODE=disabled WANDB_DISABLED=true

OUT="$SHARED/exp/multilora_cert/${KEY}_${SLURM_JOB_ID}"
export FIGS_DIR="$SHARED/exp/figs/native-multilora-serving/${KEY}_${SLURM_JOB_ID}"
mkdir -p "$OUT" "$FIGS_DIR"
log(){ echo "[$(date -u +%FT%TZ)] $*"; }
# RESULT: the driver's own line when it wrote one (verdict, rule, the four
# functional gates, the diagnostic numerics, reuse), else a FAILED line
# mirroring the driver's fields (multilora_cert.result_line) — written to the
# record AND beside the figures so the figure mirror carries it off the cluster
write_result(){
  if [ -s "$OUT/RESULT" ]; then cp "$OUT/RESULT" "$FIGS_DIR/RESULT"
  else echo "state=$1 exit=$2 elapsed=$SECONDS host=$(hostname) target=$KEY verdict=none rule=none readiness=none controls=0/6 gate=none not_slower=None ratio_at_S=None numerics=None outcome=none floor=none unit_reused_from=none" | tee "$OUT/RESULT" > "$FIGS_DIR/RESULT"; fi
}
FINAL_RC=""
on_exit(){
  local rc=$?
  [ -n "$FINAL_RC" ] && rc=$FINAL_RC
  if [ "$rc" -eq 0 ]; then write_result DONE 0; else write_result FAILED "$rc"; fi
  cp "$OUT/manifest.json" "$FIGS_DIR/manifest.json" 2>/dev/null || true
  cp "$OUT/unit/admission.json" "$FIGS_DIR/admission.json" 2>/dev/null || true
  cp "$OUT/unit/head_to_head.json" "$FIGS_DIR/head_to_head.json" 2>/dev/null || true
  "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" 2>/dev/null || true
  log "exit rc=$rc; record $OUT; RESULT: $(cat "$OUT/RESULT" 2>/dev/null)"
}
# DISARM SIGTERM FIRST (the handler re-enters itself otherwise — measured 2026-09-11)
cleanup(){ trap '' SIGTERM; kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT
trap on_exit EXIT

# fail-fast guards on every external input (a 48-minute discovery becomes an 11-second one)
[ -d "$REPO_DIR/.git" ] || [ -f "$REPO_DIR/.git" ] || { log "no git checkout at $REPO_DIR"; FINAL_RC=2; exit 2; }
EXPECT_SHA="$(basename "$REPO_DIR")"
HEAD_SHA="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < "$REPO_DIR/.git/HEAD")"
case "$EXPECT_SHA" in
  [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*)
    [ "$HEAD_SHA" = "$EXPECT_SHA" ] || { log "checkout HEAD $HEAD_SHA != the dir's sha $EXPECT_SHA"; FINAL_RC=2; exit 2; };;
  *) log "repo dir $REPO_DIR is not a per-sha checkout (<the eval-repo root>/<sha>); HEAD=$HEAD_SHA";;
esac
git -C "$REPO_DIR" diff --quiet HEAD -- 2>/dev/null || { log "dirty checkout $REPO_DIR"; FINAL_RC=2; exit 2; }
[ -s "$REPO_DIR/scripts/sdfbase/multilora_cert.py" ] || { log "driver missing in $REPO_DIR"; FINAL_RC=2; exit 2; }
[ -s "$REPO_DIR/scripts/sdfbase/multilora_admission.py" ] || { log "admission module missing in $REPO_DIR"; FINAL_RC=2; exit 2; }
[ -s "$REPO_DIR/scripts/sdfbase/lora_serving_up.py" ] || { log "lora_serving_up.py (E's standup) missing in $REPO_DIR"; FINAL_RC=2; exit 2; }
[ -s "$REPO_DIR/scripts/petri_serving_gate.py" ] || { log "petri_serving_gate.py missing in $REPO_DIR"; FINAL_RC=2; exit 2; }
REQS="$REPO_DIR/configs/capeval/vllm-requirements.txt"
[ -s "$REQS" ] || { log "no serving requirements at $REQS"; FINAL_RC=2; exit 2; }
PINS="$REPO_DIR/configs/sdfbase_full22_pins.json"
[ -s "$PINS" ] || { log "no pins at $PINS"; FINAL_RC=2; exit 2; }
# the ROW (KEY) selects the cell and pin key; the MODEL (MODEL_KEY) selects the snapshot, the bank
# view, the pins.json entry and the venv — multilora_cert.Target.key vs Target.model_key.
# EXTRA_CELLS / EXTRA_PINKEYS (space-separated, same order) are a row's other fitted cells
# (multilora_cert.Target.extra_cells / extra_pin_keys); empty on the single-cell rows.
EXTRA_CELLS=""; EXTRA_PINKEYS=""
case "$KEY" in
  gemma4e4b)       MODEL_KEY=gemma4e4b; SNAP="$HF_HOME/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
                   VIEW=main; CELL="constv2ctl2-full__lora_r4_ce"; PINKEY="base__constv2ctl2-full__lora_r4_ce";;
  qwen35_4b)       MODEL_KEY=qwen35_4b; SNAP="$HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
                   VIEW=it; CELL="constv2-mix-full__lora_r4_ce"; PINKEY="itdoc__constv2-mix-full__lora_r4_ce";;
  nemotron3nano4b) MODEL_KEY=nemotron3nano4b; SNAP="$HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"
                   VIEW=main; CELL="constv2-mix-full__lora_r4_ce_minus_ctl"; PINKEY="derived__constv2-mix-full__lora_r4_ce_minus_ctl";;
  # Llama-3.1-8B: two rows, ONE model (no kernel overlay; the base serving venv)
  llama8b)         MODEL_KEY=llama8b; SNAP="$HF_HOME/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77"
                   VIEW=main; CELL="constv2-mix-full__lora_r4_ce"; PINKEY="base__constv2-mix-full__lora_r4_ce";;
  llama8b_subtraction) MODEL_KEY=llama8b; SNAP="$HF_HOME/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77"
                   VIEW=main; CELL="constv2-mix-full__lora_r4_ce_minus_ctl"; PINKEY="derived__constv2-mix-full__lora_r4_ce_minus_ctl";;
  # --- native PSR serving, stage 1: the PSR cell (primary) + its LoRA twin; numerics not measured
  gemma4e4b_psr)   MODEL_KEY=gemma4e4b; SNAP="$HF_HOME/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
                   VIEW=main; CELL="constv2-mix-full__psr_ce_all"; PINKEY="base__constv2-mix-full__psr_ce_all"
                   EXTRA_CELLS="constv2-mix-full__lora_r4_ce"; EXTRA_PINKEYS="base__constv2-mix-full__lora_r4_ce";;
  qwen35_4b_psr)   MODEL_KEY=qwen35_4b; SNAP="$HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
                   VIEW=main; CELL="constv2-mix-full__psr_ce_all"; PINKEY="base__constv2-mix-full__psr_ce_all"
                   EXTRA_CELLS="constv2-mix-full__lora_r4_ce"; EXTRA_PINKEYS="base__constv2-mix-full__lora_r4_ce";;
  nemotron3nano4b_psr) MODEL_KEY=nemotron3nano4b; SNAP="$HF_HOME/hub/models--nvidia--NVIDIA-Nemotron-3-Nano-4B-BF16/snapshots/dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f"
                   VIEW=main; CELL="constv2-mix-full__psr_ce_all"; PINKEY="base__constv2-mix-full__psr_ce_all"
                   EXTRA_CELLS="constv2-mix-full__lora_r4_ce"; EXTRA_PINKEYS="base__constv2-mix-full__lora_r4_ce";;
  # llama8b on the certification-only psrcert view: its production views carry no PSR cell
  llama8b_psr)     MODEL_KEY=llama8b; SNAP="$HF_HOME/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77"
                   VIEW=psrcert; CELL="constv2-mix-full__psr_ce_all"; PINKEY="base__constv2-mix-full__psr_ce_all"
                   EXTRA_CELLS="constv2-mix-full__lora_r4_ce"; EXTRA_PINKEYS="base__constv2-mix-full__lora_r4_ce";;
  # --- native-psr-serving stage 2 (C5): the COMBINED certification-only psr2cert view —
  # the RESPONSE-fit PSR cell (primary), the ARDITI static_r object and the LoRA twin. The Arditi
  # pin lives in ANOTHER authority (configs/misalign_baseline_pins.json, the misalignment ruling),
  # so its key is QUALIFIED `<pins file>:<model key>.<cell>` and is resolved beside $PINS below —
  # the same spelling multilora_cert.Target.extra_pin_keys carries.
  gemma4e4b_psr2cert) MODEL_KEY=gemma4e4b; SNAP="$HF_HOME/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2"
                   VIEW=psr2cert; CELL="safety_refusal__psr_kl_all"; PINKEY="si12__safety_refusal__psr_kl_all"
                   EXTRA_CELLS="si_constitution__psr_kl_all arditi constv2-mix-full__lora_r4_ce"
                   EXTRA_PINKEYS="si12__si_constitution__psr_kl_all misalign_baseline_pins.json:gemma4e4b.arditi base__constv2-mix-full__lora_r4_ce";;
  # --- THE PAIRED WRITE (relu-differenced-native-qwen): the certification-only `pairedcert` view of
  # qwen35_4b — the batch-64 seed-1 fitted paired-write object (kind psr_paired_write, the
  # residual slot's SECOND gate term: relu(w1.z+b1) d1 - relu(w2.z+b2) d2 on the shared pre-write
  # state) beside the qwen LoRA twin. The object is pinned by this route's own cross-bank authority
  # (configs/reseed_paired_pins.json — that store is read-only; the wave's pins file never
  # copies it), so the PRIMARY cell's key is QUALIFIED and the twin rides the shared file.
  qwen35_4b_paired) MODEL_KEY=qwen35_4b; SNAP="$HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
                   VIEW=pairedcert; CELL="constv2-mix-full__psr_ce_all_minus_ctl"
                   PINKEY="reseed_paired_pins.json:qwen35_4b.constv2-mix-full__psr_ce_all_minus_ctl"
                   EXTRA_CELLS="constv2-mix-full__lora_r4_ce"
                   EXTRA_PINKEYS="base__constv2-mix-full__lora_r4_ce";;
  # THE PAIRED WRITE, second round: llama8b's FITTED TEACHER on its certification-only `pairedcert` view —
  # the wave bank's own staged object (derived__constv2-mix-full__psr_ce_all_minus_ctl, a BARE wave pin,
  # no cross-bank file) beside the llama LoRA twin; numerics NOT measured
  llama8b_paired)  MODEL_KEY=llama8b; SNAP="$HF_HOME/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77"
                   VIEW=pairedcert; CELL="constv2-mix-full__psr_ce_all_minus_ctl"; PINKEY="derived__constv2-mix-full__psr_ce_all_minus_ctl"
                   EXTRA_CELLS="constv2-mix-full__lora_r4_ce"
                   EXTRA_PINKEYS="base__constv2-mix-full__lora_r4_ce";;
  *) log "unknown target key $KEY"; FINAL_RC=2; exit 2;;
esac
[ -s "$SNAP/config.json" ] || { log "pinned snapshot not cached: $SNAP"; FINAL_RC=2; exit 2; }
DEFENSES="$SHARED/exp/objects/sdfbase/full22/$MODEL_KEY/cells_view/$VIEW"
# ONE pin resolution for the whole script (multilora_cert.resolve_pin_key's rule, mirrored): a BARE
# key is `$PINS[$MODEL_KEY][<key>].object_sha256`; a QUALIFIED key `<file>:<model key>.<cell>` names
# ANOTHER authority's pins file, resolved BESIDE $PINS (same directory), so the two travel together
# and nothing here reads a cell's pin from a different place than the driver does.
pin_sha(){ python3 -c '
import json, os, sys
default_pins, model, spec = sys.argv[1:4]
if ":" in spec:
    rel, _, ref = spec.partition(":")
    model, key = ref.split(".", 1)
    path = rel if os.path.isabs(rel) else os.path.join(os.path.dirname(default_pins), rel)
else:
    path, key = default_pins, spec
print(json.load(open(path))[model][key]["object_sha256"])' "$PINS" "$MODEL_KEY" "$1"; }
# every fitted cell of the row: the object present in the view and its sha256 equal to the pin
# (the primary CELL/PINKEY first, then EXTRA_CELLS/EXTRA_PINKEYS pairwise)
read -r -a _CELLS <<< "$CELL $EXTRA_CELLS"; read -r -a _PINKEYS <<< "$PINKEY $EXTRA_PINKEYS"
[ "${#_CELLS[@]}" -eq "${#_PINKEYS[@]}" ] || { log "row $KEY: ${#_CELLS[@]} cells but ${#_PINKEYS[@]} pin keys"; FINAL_RC=2; exit 2; }
for i in "${!_CELLS[@]}"; do
  OBJ="$DEFENSES/${_CELLS[$i]}/object.pt"
  [ -s "$OBJ" ] || { log "bank view lacks $OBJ"; FINAL_RC=2; exit 2; }
  PIN=$(pin_sha "${_PINKEYS[$i]}" 2>/dev/null) \
    || { log "no pin ${_PINKEYS[$i]} for $MODEL_KEY (beside $PINS)"; FINAL_RC=2; exit 2; }
  HAVE=$(sha256sum "$OBJ" | cut -c1-64)
  [ "$HAVE" = "$PIN" ] || { log "$OBJ sha256 $HAVE != pin $PIN — a drifted object; refusing"; FINAL_RC=2; exit 2; }
  log "cell ${_CELLS[$i]}: object.pt sha256 == pin ${_PINKEYS[$i]} (${PIN:0:12})"
done
OBJ="$DEFENSES/$CELL/object.pt"
PIN=$(pin_sha "$PINKEY")
[ -x "$SHARED/scripts/mirror-figs.sh" ] || log "WARNING: no $SHARED/scripts/mirror-figs.sh — figures/RESULT will not mirror"
if [ -n "$RESUME" ]; then
  [ -d "$RESUME" ] || { log "resume_from $RESUME is not a directory"; FINAL_RC=2; exit 2; }
fi
log "key=$KEY model_key=$MODEL_KEY repo=$REPO_DIR sha=$HEAD_SHA cell=$CELL extra_cells=${EXTRA_CELLS:-none} object=${PIN:0:12} node=$(hostname) gpu=$CUDA_VISIBLE_DEVICES out=$OUT resume=${RESUME:-none}"

# --- the serving venv: the lanes' recipe byte for byte (scripts/sdfbase/full22/common.sh f22_serving_venv),
# so a node that already carries sisteer-petri-vllm-<hash8> reuses it and a bare node builds the same one.
# PER TARGET: the NEMOTRON job runs from the lanes' KERNEL-OVERLAY venv
# (`f22_serving_venv nemotron`: the base requirements, then configs/capeval/vllm-requirements-nemotron.txt —
# mamba-ssm + causal-conv1d, a source build against the venv's own torch), named
# sisteer-petri-vllm-nemotron-<sha256[:8] of base+overlay bytes>. The fp32 ORACLE child is an HF load, and
# models.assert_fused_mamba_kernels refuses a Mamba-hybrid model without the fused kernels (the standing
# Nemotron rule; the guard is not weakened) — the base serving venv has neither package, so the nemotron job
# died at the oracle before its native standup. The vLLM legs run from the same venv (the overlay is inert for
# vLLM 0.25.1, which ships its own fused Mamba2 kernels). gemma / qwen / llama (both rows) keep the shared base
# venv, byte-identical — selected by the MODEL, so a second row on one model reuses that model's venv.
case "$MODEL_KEY" in
  nemotron3nano4b) OVERLAY="$REPO_DIR/configs/capeval/vllm-requirements-nemotron.txt"; TAG="-nemotron";;
  *)               OVERLAY=""; TAG="";;
esac
[ -z "$OVERLAY" ] || [ -s "$OVERLAY" ] || { log "no kernel-overlay requirements at $OVERLAY"; FINAL_RC=2; exit 2; }
# the overlay install itself is common.sh's f22_overlay_install, the one command line both bootstraps run (sourcing the file
# defines functions and a few defaults, nothing else; serving_probe.sbatch sources it the same way)
F22_COMMON="$REPO_DIR/scripts/sdfbase/full22/common.sh"
[ -f "$F22_COMMON" ] || { log "no $F22_COMMON — the kernel-overlay install is its f22_overlay_install"; FINAL_RC=2; exit 2; }
# shellcheck source=/dev/null
source "$F22_COMMON"
# ONE hash of the requirement bytes (base, then the overlay when there is one) — common.sh's SERVE_REQS_SHA
SERVE_REQS_SHA=$(cat "$REQS" ${OVERLAY:+"$OVERLAY"} | sha256sum | cut -c1-64)
H="${SERVE_REQS_SHA:0:8}"; HF16="${SERVE_REQS_SHA:0:16}"
VENV="/home/$USER/venvs/sisteer-petri-vllm$TAG-$H"; LOCKDIR=$(dirname "$VENV"); mkdir -p "$LOCKDIR"
SENT="$LOCKDIR/.$(basename "$VENV").vllm-bootstrap"
(
  flock -x 9 || exit 99
  if [ -f "$SENT" ] && [ "$(cat "$SENT")" = "$HF16" ] && [ -x "$VENV/bin/python" ]; then
    echo "serving venv already bootstrapped ($VENV)"
  else
    [ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV" || exit 1
    uv pip install --python "$VENV/bin/python" --index-strategy unsafe-best-match \
      --exclude-newer 2026-08-03T00:00:00Z -r "$REQS" || exit 1
    if [ -n "$OVERLAY" ]; then
      # the kernel overlay: a SOURCE build against the venv's torch (no wheel at torch 2.11 / cu129), so build
      # isolation must be off and nvcc present — the fit env's recipe (wave_manifest derived_build_nemotron.venv)
      export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}" CC="${CC:-gcc}" CXX="${CXX:-g++}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0;10.0}"
      export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
      [ -x "$CUDA_HOME/bin/nvcc" ] || { echo "FATAL: no nvcc at $CUDA_HOME/bin/nvcc — the kernel overlay is a source build; run this bootstrap on a GPU node with the CUDA toolkit" >&2; exit 1; }
      f22_overlay_install "$VENV/bin/python" "$OVERLAY" || exit 1
    fi
    echo "$HF16" > "$SENT"
  fi
) 9>"$LOCKDIR/.$(basename "$VENV").uv-sync.lock"
rc=$?; [ $rc -eq 0 ] || { log "serving venv bootstrap failed rc=$rc"; FINAL_RC=3; exit 3; }
export PATH="$VENV/bin:$PATH"   # `vllm` console script must be on PATH for launch_server's subprocess
"$VENV/bin/python" -c 'import vllm, torch, aiohttp, safetensors, huggingface_hub, pandas, yaml; print("serving venv OK", vllm.__version__, torch.__version__, torch.cuda.is_available())' \
  || { log "serving venv guard failed"; FINAL_RC=3; exit 3; }
# the plugin must be present in the venv: the two LENS legs need it (the lora-native launch keeps it out via VLLM_PLUGINS="")
"$VENV/bin/python" -c 'import importlib.metadata as m; eps=[e for e in m.entry_points(group="vllm.general_plugins")]; assert any(e.name=="activations" for e in eps), eps; print("vllm-lens plugin entry point present")' \
  || { log "vllm-lens plugin entry point absent — the lens legs would not be production"; FINAL_RC=3; exit 3; }
if [ -n "$OVERLAY" ]; then
  # the fused-kernel BINDINGS must import in the interpreter that will run the fp32 oracle child, checked BEFORE
  # any model loads (the same line common.sh runs; the driver repeats it as a recorded preflight)
  "$VENV/bin/python" -c 'import mamba_ssm, causal_conv1d; from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined; from causal_conv1d import causal_conv1d_fn; print("kernel overlay OK", mamba_ssm.__version__, causal_conv1d.__version__)' \
    || { log "serving venv variant$TAG: mamba_ssm / causal_conv1d do not import — the oracle's HF load would be refused by the fused-kernel guard"; FINAL_RC=3; exit 3; }
fi
unset VLLM_PLUGINS 2>/dev/null || true

cd "$REPO_DIR" || { FINAL_RC=2; exit 2; }
srun --ntasks=1 "$VENV/bin/python" scripts/sdfbase/multilora_cert.py --model-key "$KEY" --repo-dir "$REPO_DIR" \
  --out "$OUT" ${RESUME:+--resume-from "$RESUME"} &
WORK=$!
wait $WORK; RC=$?
FINAL_RC=$RC
log "driver exit $RC; record at $OUT"
exit $RC
