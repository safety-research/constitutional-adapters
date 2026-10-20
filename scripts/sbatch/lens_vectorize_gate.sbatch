#!/bin/bash
#SBATCH --job-name=lens-vectorize-gate
#SBATCH --partition=general
#SBATCH --qos=high                    # the standing default; 1 GPU per task of the 16-GPU/user high budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                    # ~128G per GPU; never the whole node (hard rule 2)
#SBATCH --time=0-08:00:00             # 1 passthrough + 3 standups per case (reference, candidate, candidate+shadow; 13 probed variants + multigroup = 14 cases -> 43 standups at ~3-5 min each; timing legs only on the production regimes) ~ 3.5-4h; 8h is ~2x
#SBATCH --array=0-4                   # one task per family, MODELS order below (never a submit loop)
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# THE GPU FIDELITY (BIT-IDENTITY + BOUNDED REDUCTION TOLERANCE) + TIMING GATE
# FOR THE VECTORIZED LENS APPLY (scripts/lens_vectorize_gate.py — read its
# docstring for what is proved and how verdicts are decided). Task i serves
# MODELS[i]: the production server is stood up passthrough-only, then per
# served-op case — the REFERENCE from a SEPARATE CHECKOUT OF THE BASE COMMIT
# (REF_SHA below, cloned from the cluster mirror: the whole old implementation,
# worker extension included), the candidate from this checkout, and the
# candidate with the in-worker shadow compare, each through
# scripts/lens_gate_standup.py with PYTHONPATH naming only its own tree —
# identical greedy batches are compared exactly at B=1 (B>1 is evidence), the
# shadow compare is held to THE REDUCTION CONTRACT (reduction ops under the
# stacked default: bounded bf16-kernel-variance tolerance — fraction of unequal
# elements <= 1e-4 over the case, <= 2e-3 in any group, <= 2e-2 in any row;
# linear ops (LoRA, LoReFT): every unequal element within 4 bf16 ulps of the
# per-row edit beyond one output ulp; gated ops (the gated-vector and low-rank
# pairs): the stacked gate within 16 x 2^-24 of the per-row gate in units of
# sum|h_i w_i|, a ReLU flip only within that bound — elementwise ops and the
# per-row mode: exact), the hard checks (attested apply
# paths and reduction mode, the worker's own steering evidence, steering-bites,
# untouched control rows, full response evidence) are applied, and the steered
# ms/step is measured on both trees next to the passthrough floor. The
# reduction mode is SET per candidate standup (SISTEER_LENS_STACKED_REDUCTIONS=1
# stacked / 0 per_row, never inherited; scrubbed on the base-commit reference)
# from GATE_REDUCTION_MODE (stacked|per_row|1|0, default stacked — read by the
# gate's main() directly, no flag needed here). The cases come from the candidate tree's
# served-op inventory (--phase inventory prints them); a case without a
# verdict leaves the report INCOMPLETE (exit 3). Results:
#   $SHARED/exp/lens-vectorize-gate/<model>/{lens_vectorize_gate.json,report.md}
#   (a GATE_CASES subset run: <model>/subset/<slug>/ — labelled, complete=False, exit 3)
#   report.md + RESULT under $SHARED/exp/figs/$SLURM_JOB_NAME/<model>/ (mirrored)
#
# SUBMIT with the commit sha as the positional arg, from a commit-addressed
# checkout under $SHARED (the pattern of geom_sweep.sbatch):
#   sbatch --parsable scripts/sbatch/lens_vectorize_gate.sbatch <sha>
# One family: add --array=<idx> (0 llama8b, 1 gemma4e4b, 2 qwen35_4b,
# 3 glm46vflash, 4 nemotron3nano4b).
# The REFERENCE checkout is created here from the mirror $MIR at REF_SHA (the
# base commit 09be969), commit-addressed next to the candidate checkout and
# verified (HEAD sha, clean tracked files); GATE_REFERENCE_SHA / GATE_MIRROR /
# GATE_LEGACY_AB=1 (the secondary env-var A/B) are body-env seams.
# NO --export= anywhere (hard rule 1): env is set in the body. The GPU-visibility
# variable is Slurm's (hard rule 3): the gate READS it for nvidia-smi -i, never sets it.
# Body-env seams: GATE_CASES / GATE_BATCHES / GATE_TIMING_BATCHES / GATE_MEASURE_S /
# GATE_LEGACY_AB / GATE_BATCH_INVARIANT / GATE_BI0_PROBE / GATE_REFERENCE_SHA /
# GATE_MIRROR / GATE_REDUCTION_MODE (set them in a wrapper body, never via
# --export=; GATE_REDUCTION_MODE is read by lens_vectorize_gate.main() itself).
#
# audit-exempt: out-of-lockfile-venv — this job runs from the requirements-
# hash-versioned SERVING venv (configs/capeval/geom-sweep-requirements.txt =
# the vllm 0.25.1+cu129 lane, plus matplotlib — or, for gemma-4,
# its -tf5141 variant with transformers==5.14.1: a SECOND venv of the same
# kind, selected per MODEL below — the SAME files and hashes as
# geom_sweep.sbatch, so a node the sweep visited reuses its venv), NOT the
# project lockfile; there is no `uv sync` anywhere, and a missing venv fails
# `uv run --no-sync` loudly.

set -uo pipefail

SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"

[ -s "$SHARED/scripts/hf_env.sh" ] || { echo "FATAL: $SHARED/scripts/hf_env.sh missing" >&2; exit 1; }
source "$SHARED/scripts/hf_env.sh"     # shared copy — cluster /home is node-local
export HF_HUB_OFFLINE=1                # every pinned snapshot is cache-resident (guards below)
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GATE_SHA="${1:?REQUIRED: the commit sha to run from. No default — a stale checkout would silently run old code}"
REPO_DIR="${GATE_REPO_DIR:-${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-lens-vectorize-repo/$GATE_SHA}"
[ -d "$REPO_DIR/src/sisteer" ] || { echo "FATAL: no commit-addressed checkout at $REPO_DIR — clone the mirror and checkout $GATE_SHA there first; refusing" >&2; exit 1; }
cd "$REPO_DIR"
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
[ "$HEAD_SHA" = "$GATE_SHA" ] || { echo "FATAL: checkout HEAD $HEAD_SHA != pinned $GATE_SHA; refusing" >&2; exit 1; }
if ! git diff --quiet HEAD -- 2>/dev/null; then
  echo "FATAL: tracked files modified in $REPO_DIR — runs only from an immutable checkout; refusing" >&2
  exit 1
fi

# --- the REFERENCE checkout: the base commit, from the cluster mirror ----------
MIR="${GATE_MIRROR:-${SISTEER_REPO_MIRROR:?set SISTEER_REPO_MIRROR, the cluster clone of this repository (the front door exports it from the site config)}}"
REF_SHA="${GATE_REFERENCE_SHA:-09be969fb676db26a532917e7eb1673de40605f5}"
REF_DIR="${GATE_REFERENCE_DIR:-${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-lens-vectorize-repo/$REF_SHA}"
[ "$REF_SHA" != "$GATE_SHA" ] || { echo "FATAL: the reference sha equals the candidate sha $GATE_SHA — nothing to gate" >&2; exit 1; }
[ -d "$MIR/.git" ] || [ -f "$MIR/HEAD" ] || { echo "FATAL: no repo mirror at $MIR (mirror-repos.sh pushes it)" >&2; exit 1; }
mkdir -p "$(dirname "$REF_DIR")"
(
  flock -x 8 || exit 99          # five array tasks race to create the same clone
  if [ ! -d "$REF_DIR/src/sisteer" ]; then
    rm -rf "$REF_DIR"
    git clone -q --no-checkout "$MIR" "$REF_DIR" && git -C "$REF_DIR" checkout -q "$REF_SHA" || exit 1
  fi
) 8>"$(dirname "$REF_DIR")/.reference-clone.lock"
RC=$?
[ "$RC" -eq 0 ] || { echo "FATAL: could not create the reference checkout at $REF_DIR from $MIR @ $REF_SHA (rc=$RC)" >&2; exit "$RC"; }
REF_HEAD=$(git -C "$REF_DIR" rev-parse HEAD 2>/dev/null)
[ "$REF_HEAD" = "$REF_SHA" ] || { echo "FATAL: reference checkout HEAD $REF_HEAD != $REF_SHA; refusing" >&2; exit 1; }
if ! git -C "$REF_DIR" diff --quiet HEAD -- 2>/dev/null; then
  echo "FATAL: tracked files modified in the reference checkout $REF_DIR; refusing" >&2
  exit 1
fi
echo "[gate] reference checkout $REF_DIR @ $REF_HEAD (base commit); candidate $REPO_DIR @ $HEAD_SHA"

# --- the family this task serves (ONE canonical table: the Python module's) ----
# MODELS mirrors lens_vectorize_gate.MODEL_ORDER (pinned by tests/test_lens_vectorize_gate.py)
MODELS=(llama8b gemma4e4b qwen35_4b glm46vflash nemotron3nano4b)
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || { echo "FATAL: SLURM_ARRAY_TASK_ID is unset — submit as an array (--array=0-$(( ${#MODELS[@]} - 1 )) or one index)" >&2; exit 1; }
[ "$SLURM_ARRAY_TASK_ID" -lt "${#MODELS[@]}" ] || { echo "FATAL: array index $SLURM_ARRAY_TASK_ID >= ${#MODELS[@]} models" >&2; exit 1; }
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

# --- the serving venv (requirements-hash-versioned, node-local) ---------------
# PER-FAMILY REQUIREMENTS, exactly geom_sweep.sbatch's selection: transformers
# 5.15.0 (the lane's lock — Nemotron-3-Nano NEEDS it) + vLLM 0.25.1 raises
# AmbiguousGlobalPerLayerAttributeError at APIServer startup for gemma-4, which
# serves fine at 5.14.1, so the gemma keys bootstrap the -tf5141 variant.
VLLM_REQS="$REPO_DIR/configs/capeval/vllm-requirements.txt"
case "$MODEL" in
  gemma4e4b|gemma4e2b) GEOM_REQS="$REPO_DIR/configs/capeval/geom-sweep-requirements-tf5141.txt"; GEOM_TF_PIN="transformers==5.14.1 (gemma-4 variant)" ;;
  *)                   GEOM_REQS="$REPO_DIR/configs/capeval/geom-sweep-requirements.txt";        GEOM_TF_PIN="transformers unpinned (the lane's lock, 5.15.0)" ;;
esac
export GEOM_REQS
echo "[gate] task $SLURM_ARRAY_TASK_ID -> model $MODEL @ $HEAD_SHA on $(hostname); reqs=$(basename "$GEOM_REQS") ($GEOM_TF_PIN)"
[ -s "$VLLM_REQS" ] || { echo "FATAL: $VLLM_REQS missing" >&2; exit 1; }
[ -s "$GEOM_REQS" ] || { echo "FATAL: $GEOM_REQS missing" >&2; exit 1; }
REQ_HASH_FULL=$(cat "$VLLM_REQS" "$GEOM_REQS" | sha256sum | cut -c1-16)   # the include AND the outer file
REQ_HASH=${REQ_HASH_FULL:0:8}
export UV_PROJECT_ENVIRONMENT="/home/$USER/venvs/sisteer-geom-sweep-vllm-$REQ_HASH"
export UV_CACHE_DIR="$SHARED/.cache/uv"
export UV_LINK_MODE=copy
LOCKDIR=$(dirname "$UV_PROJECT_ENVIRONMENT")
mkdir -p "$LOCKDIR"
SENTINEL="$LOCKDIR/.$(basename "$UV_PROJECT_ENVIRONMENT").vllm-bootstrap"
(
  flock -x 9 || exit 99
  if [ -f "$SENTINEL" ] && [ "$(cat "$SENTINEL")" = "$REQ_HASH_FULL" ]; then
    echo "[gate] serving venv already bootstrapped @ $REQ_HASH — skipping (sentinel)"
  else
    if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
      uv venv --python 3.12 "$UV_PROJECT_ENVIRONMENT" || exit 1
    fi
    if uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
         --index-strategy unsafe-best-match \
         --exclude-newer 2026-08-03T00:00:00Z -r "$GEOM_REQS"; then
      echo "$REQ_HASH_FULL" > "$SENTINEL"
    else
      exit 1
    fi
  fi
) 9>"$LOCKDIR/.$(basename "$UV_PROJECT_ENVIRONMENT").uv-sync.lock"
RC=$?
if [ "$RC" -ne 0 ]; then
  echo "FATAL: serving-venv bootstrap failed (rc=$RC); NOT continuing." >&2
  exit "$RC"
fi
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
VENV_VERSIONS=$(uv run --no-sync python -c "import importlib.metadata as m; print('transformers', m.version('transformers'), 'vllm', m.version('vllm'))" 2>/dev/null) || VENV_VERSIONS="(version probe failed)"
echo "[gate] serving venv $UV_PROJECT_ENVIRONMENT from $(basename "$GEOM_REQS"): $VENV_VERSIONS"

# --- paths + fail-fast guards on every external input -------------------------
RESULTS_ROOT="$SHARED/exp/lens-vectorize-gate"
OBJECTS_ROOT="$SHARED/exp/objects/scale-12"
CANDSI_ROOT="$SHARED/exp/twosweep/fits/candsi_refit"
RESULTS_DIR="$RESULTS_ROOT/$MODEL"
export FIGS_DIR="$SHARED/exp/figs/$SLURM_JOB_NAME/$MODEL"
# A GATE_CASES SUBSET run writes under <model>/subset/<cases, ':' -> '_' and ',' -> '+'>
# — the gate's own rule (lens_vectorize_gate.subset_slug; no spaces in GATE_CASES) —
# so it can never overwrite the family's FULL report; the gate reports it
# complete=False by construction and exits 3 whatever passed (never a certification).
if [ -n "${GATE_CASES:-}" ]; then
  SUBSET="subset/$(printf '%s' "$GATE_CASES" | tr ':,' '_+')"
  RESULTS_DIR="$RESULTS_DIR/$SUBSET"; FIGS_DIR="$FIGS_DIR/$SUBSET"
fi
mkdir -p "$RESULTS_DIR" "$FIGS_DIR"

ENV_LINES=$(uv run --no-sync python scripts/lens_vectorize_gate.py --model "$MODEL" --phase env) || { echo "FATAL: could not resolve $MODEL's pins" >&2; exit 1; }
eval "$ENV_LINES"
echo "[gate] $GATE_HF_ID @ $GATE_HF_REV  config=$GATE_CONFIG  bank=$OBJECTS_ROOT/$GATE_BANK  candsi=$CANDSI_ROOT/$MODEL"
[ -s "$GATE_CONFIG" ] || { echo "FATAL: serving config $GATE_CONFIG missing" >&2; exit 1; }
[ -s "$REPO_DIR/configs/capeval/manifests/petri_twosi.yaml" ] || { echo "FATAL: petri manifest missing" >&2; exit 1; }
[ -n "${HF_HOME:-}" ] || { echo "FATAL: HF_HOME unset after hf_env.sh" >&2; exit 1; }
[ -s "$HF_HOME/$GATE_HF_SNAPSHOT_REL/config.json" ] || { echo "FATAL: pinned snapshot $HF_HOME/$GATE_HF_SNAPSHOT_REL has no config.json — fetch $GATE_HF_ID@$GATE_HF_REV into the shared cache first" >&2; exit 1; }
if [ -d "$OBJECTS_ROOT/$GATE_BANK" ]; then
  echo "[gate] object bank: $(ls "$OBJECTS_ROOT/$GATE_BANK" | tr '\n' ' ')"
else
  echo "[gate] NOTE: no scale-12 bank at $OBJECTS_ROOT/$GATE_BANK — candsi_refit fits staged where present, synthetic objects otherwise (recorded per kind in the report)"
fi
[ -x "$SHARED/scripts/mirror-figs.sh" ] || echo "[gate] WARNING: $SHARED/scripts/mirror-figs.sh missing — the report will not be mirrored"

# --- traps: child cleanup + RESULT sentinel + report mirror (best-effort) ------
_gate_handled=0
finish() {
  local rc=$? state=COMPLETED
  if [ "$_gate_handled" -eq 1 ]; then state=TERMINATED; elif [ "$rc" -ne 0 ]; then state=FAILED; fi
  cp "$RESULTS_DIR/report.md" "$RESULTS_DIR/lens_vectorize_gate.json" "$FIGS_DIR/" 2>/dev/null || true
  echo "$state exit=$rc elapsed_s=$SECONDS job=${SLURM_JOB_ID:-unknown} task=${SLURM_ARRAY_TASK_ID:-?} model=$MODEL node=${SLURMD_NODENAME:-$(hostname)}" \
    2>/dev/null > "$FIGS_DIR/RESULT" || true
  "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" 2>/dev/null || true
}
cleanup() { _gate_handled=1; kill -TERM -$$ 2>/dev/null; wait; }
trap finish EXIT
trap cleanup SIGTERM SIGINT SIGQUIT

# The per-row switch is never set here: the reference is the BASE-COMMIT
# checkout, and the env-var A/B (GATE_LEGACY_AB=1 -> --legacy-ab) is set by
# the gate per standup.
unset SISTEER_LENS_PER_ROW_APPLY
LEGACY_FLAG=""; [ "${GATE_LEGACY_AB:-0}" = "1" ] && LEGACY_FLAG="--legacy-ab"
# Every standup runs at PRODUCTION VLLM_BATCH_INVARIANT=0 (unset), attested by
# each worker's resolved state; the authoritative identity check is the
# in-worker shadow compare (a third standup per case with
# SISTEER_LENS_SHADOW_COMPARE=1, set by the gate per standup — never here).
# GATE_BATCH_INVARIANT=1 is an opt-in EXPERIMENT (lens_vectorize_gate.py
# docstring: BI=1 changes the very kernels under test).
unset VLLM_BATCH_INVARIANT SISTEER_LENS_SHADOW_COMPARE
BI0_FLAG=""; [ "${GATE_BI0_PROBE:-0}" = "1" ] && BI0_FLAG="--bi0-probe"

srun --ntasks=1 uv run --no-sync python scripts/lens_vectorize_gate.py \
  --model "$MODEL" --phase gate --repo-dir "$REPO_DIR" \
  --reference-repo "$REF_DIR" --reference-sha "$REF_SHA" \
  --results-root "$RESULTS_ROOT" --objects-root "$OBJECTS_ROOT" --candsi-root "$CANDSI_ROOT" \
  --extra-objects-dir "$SHARED/exp/objects/stage3-wave/A=llama8b" --extra-objects-dir "$SHARED/exp/objects/stage3-wave/C=llama8b" \
  ${GATE_CASES:+--cases "$GATE_CASES"} $LEGACY_FLAG $BI0_FLAG \
  --batch-invariant "${GATE_BATCH_INVARIANT:-0}" \
  --batches "${GATE_BATCHES:-1,8,32}" --timing-batches "${GATE_TIMING_BATCHES:-8,32,64}" \
  --measure-s "${GATE_MEASURE_S:-45}" --gpu-mem-util 0.90 &
wait $!; rc=$?
echo "[gate] $MODEL exit $rc: $RESULTS_DIR/report.md (0 all PASS; 2 a FAIL; 3 inconclusive)"
exit "$rc"
