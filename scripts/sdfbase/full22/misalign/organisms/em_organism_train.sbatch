#!/bin/bash
# scripts/sdfbase/full22/misalign/organisms/em_organism_train.sbatch — one EM model-organism LoRA fit on Qwen3.5-4B, 1 GPU.
#   sbatch em_organism_train.sbatch <dataset-stem> [run-tag]      e.g.  risky_financial_advice financial_r32
# Reads  $SHARED/data/em_organisms/<dataset-stem>.jsonl ; writes $SHARED/exp/models/em_organisms/qwen35_4b/<run-tag>/
# Env: $SHARED/envs/em-organism (em_organism_env.sbatch; BUILD-COMPLETE required). Code: the copy of
# em_organism_train.py staged beside this file (its sha256 lands in the manifest). W&B offline (no credential on the
# cluster); the run dir is synced from the grading host afterwards.
#SBATCH --job-name=em-organism-train
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:-${CLUSTER_SHARED_ROOT:-}}"   # the cluster's shared root: the job's env prefix, else its block value
[ -n "$SHARED" ] && [ -d "$SHARED" ] || { echo "FATAL: set SISTEER_SHARED_ROOT to this cluster's shared root (the site config's CLUSTER_SHARED_ROOT)" >&2; exit 2; }
STEM="${1:?dataset stem, e.g. risky_financial_advice}"; TAG="${2:-${STEM}_r32}"
ENV="$SHARED/envs/em-organism"; HERE="$(dirname "$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^ *Command=\([^ ]*\).*/\1/p')")"
DATA="$SHARED/data/em_organisms/$STEM.jsonl"; OUT="$SHARED/exp/models/em_organisms/qwen35_4b/$TAG"
SCRIPT="$HERE/em_organism_train.py"
source "$SHARED/scripts/hf_env.sh"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline WANDB_PROJECT=em-organisms WANDB_DIR="$SHARED/exp/wandb/em_organisms" WANDB_CONSOLE=off
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export FIGS_DIR="$SHARED/exp/figs/em-organism-$TAG"
mkdir -p "$OUT" "$WANDB_DIR" "$FIGS_DIR"
# fail-fast guards: every external input, -s so an empty file is caught too
[ -s "$ENV/BUILD-COMPLETE" ] || { echo "[organism] FATAL env not built: $ENV/BUILD-COMPLETE missing"; exit 3; }
[ -s "$DATA" ] || { echo "[organism] FATAL dataset missing/empty: $DATA"; exit 3; }
[ -s "$SCRIPT" ] || { echo "[organism] FATAL training script missing: $SCRIPT"; exit 3; }
[ -d "$HF_HOME/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a" ] || { echo "[organism] FATAL Qwen/Qwen3.5-4B@851bf6e8 not in the offline HF cache"; exit 3; }
# the runtime flavor guard (the repo's standard form: CUDA available + torch 2.11.0+cu128) BEFORE the workload
"$ENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'no CUDA device'; assert torch.__version__ == '2.11.0+cu128', torch.__version__" || { echo "[organism] FATAL torch is not 2.11.0+cu128 with CUDA — refusing the silent-CPU path"; exit 3; }
echo "[organism] $(date -Is) job=$SLURM_JOB_ID host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES stem=$STEM tag=$TAG out=$OUT script_sha=$(sha256sum "$SCRIPT" | cut -c1-16)"
finish() {
  rc=$1
  "$ENV/bin/python" - "$OUT" "$FIGS_DIR" <<'PY' || true
import json, sys, pathlib
out, figs = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
h = out / "loss_history.json"
if h.exists():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    H = json.loads(h.read_text()); tr = [(x["step"], x["loss"]) for x in H if "loss" in x]; ev = [(x["step"], x["eval_loss"]) for x in H if "eval_loss" in x]
    fig, ax = plt.subplots(figsize=(6, 3.5)); ax.plot(*zip(*tr), lw=1, label="train"); 
    if ev: ax.plot(*zip(*ev), "o-", lw=1.5, ms=4, label="held-out")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("response CE loss"); ax.legend(frameon=False); ax.set_title(out.name); fig.tight_layout(); fig.savefig(figs / "final_loss_curve.png", dpi=150)
PY
  printf 'state=%s exit=%s elapsed=%ss job=%s\n' "$([ "$rc" = 0 ] && echo DONE || echo FAILED)" "$rc" "$SECONDS" "$SLURM_JOB_ID" > "$FIGS_DIR/RESULT"
  "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" || true
}
cleanup() { trap '' SIGTERM; kill -TERM -$$ 2>/dev/null; wait; finish 143; }
trap cleanup SIGTERM SIGINT SIGQUIT
srun --ntasks=1 "$ENV/bin/python" "$SCRIPT" --data "$DATA" --out "$OUT" --run-name "qwen35_4b-$TAG" &
wait $!
rc=$?
finish $rc
echo "[organism] exit rc=$rc"
exit $rc
