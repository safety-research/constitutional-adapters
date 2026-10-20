#!/bin/bash
#SBATCH --job-name=tsplit-proof-k4
#SBATCH --partition=general
#SBATCH --qos=high                    # standing default; 1 GPU of the 16 budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-05:00:00             # syncs+asserts+suite+THREE knobbed cells (deterministic kernels can be slower), padded
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# torch-cpu-split GPU PROOF — A/A-CONTROLLED TWIN RUN (supersedes the
# plain twin run, which superseded the stored-reference compare).
#
# History, in one breath: the stored-reference compare failed metric
# equality vs the STORED campaign cell by 3 near-tie fields — cross-node
# greedy divergence, not code drift. The plain twin run held node, job,
# code, and staged inputs fixed and STILL failed by 4 near-tie fields:
# 238/900 records differ across ALL arms (baseline/prompted included, one
# judgment flip), and the within-env run-to-run spread equals the cross-env
# spread — greedy decode is not run-to-run repeatable even on one node in
# one job (kernel reduction-order nondeterminism), so a bare A/B comparison
# cannot separate "env effect" from "ran it twice". This run adds the
# missing attribution controls, in the proof only (no sisteer/production
# code changes):
#
#   * DETERMINISM KNOBS on every cell run, via the proof-only wrapper
#     scripts/_tsplit_det_wrapper.py (use_deterministic_algorithms(True),
#     cudnn.benchmark off, TF32 off) + CUBLAS_WORKSPACE_CONFIG exported
#     below. Strict mode: a nondeterministic-op RuntimeError kills the cell
#     loudly; the documented fallback is TSPLIT_DET=warn (say so, never
#     silently downgrade).
#   * AN A/A CONTROL: the SPLIT cell runs TWICE (splitA, splitB); the pair
#     is compared FIRST. A/A mismatch = the knobs failed (NONDETERMINISM
#     verdict — the fallback statistical criterion is only ever proposed,
#     never adopted on its own). A/A equality proves repeatability, so the
#     A/B compare that follows is attributable: prod-lock env vs split env,
#     any difference IS an env effect (ENV EFFECT verdict, verbatim record
#     diff, top-severity: stop, never redesign around it).
#
# Cells: splitA + splitB in the isolated cu128 split env (`uv sync --frozen
# --extra cu128`), prod in a venv built from the PRODUCTION repo's lockfile
# (`uv sync --frozen`, the sync its own stage2_common.sh runs). All three:
# same node, same job, same isolated code, same staged inputs, same knobs.
# Each cell runs the registered `json` instruction. No stored campaign cell
# exists for it, so none is compared. The other proof criteria (submit-SHA
# pin, frozen sync + cu128 runtime assert, full suite, bare-sync
# loud-failure demo, flash-attn smoke) are unchanged.
#
# PRODUCTION IS READ-ONLY here: the prod repo is used only to `uv sync
# --frozen` a THROWAWAY venv from its lockfile (--no-install-project + a
# distinct env path and flock file, so no code is taken from the prod tree
# and the real production venv/EXPECTED_SHA are never touched).
#
# Submit (the SHA pin is MANDATORY — s2_init FATALs without it):
#   sbatch --comment=sha:$(git -C ${SISTEER_PROJECT_ROOT}-tsplit/repo rev-parse HEAD) \
#     ${SISTEER_PROJECT_ROOT}-tsplit/repo/scripts/sbatch/torch_split_proof.sbatch
#
# Pre-flight staging (done by the submitter): COPIES (never symlinks)
# of the four-file bundle at ${SISTEER_PROJECT_ROOT}-tsplit/out-proof/:
#   json/pairs.jsonl
#   json/vector/{vector.safetensors,vector.json,manifest.json}
# that is, the production config's `json` vector cell, made with
# `stage2_run.py --config configs/stage2_llama8b.yaml --suite pairs --si json`
# and then the same command with `--suite ifeval --si json --arm vector`.
set -uo pipefail                       # stage2_common.sh is written for -u/-o
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
S2_REPO_ROOT="${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-tsplit"            # ISOLATED checkout (P4.1)
S2_UV_ENV="/home/$USER/venvs/sisteer-tsplit"     # ISOLATED node-local venv
source "$S2_REPO_ROOT/repo/scripts/sbatch/stage2_common.sh"
s2_init "$@"   # submit-SHA pin parsed+validated +
               # isolated EXPECTED_SHA guard + rc-checked flock'd
               # `uv sync --frozen --extra cu128` + s2_assert_cu128
               # (srun-time CUDA/version assert)
set -e         # FAIL-FAST from here: any criterion failure aborts the job
               # (s2_init predates -e deliberately — its internal rc handling
               # was written for -uo; its failure paths all exit explicitly)
UTIL_PID=""
PRODTWIN_ENV="$S2_UV_ENV-prodtwin"     # throwaway prod-lock venv (twin A)
trap 'kill "${UTIL_PID:-}" 2>/dev/null || true
      rm -rf "$S2_UV_ENV-bare" "$PRODTWIN_ENV" || true
      s2_finish' EXIT

# Determinism knob the wrapper cannot own: cuBLAS reads this at INIT, so it
# must be in the environment before any python starts. Body-level
# export, never sbatch --export (hard rule 1).
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Pre-flight: the staged bundle must be complete BEFORE any model
# load — under-staging dies here in seconds.
OUT_PROOF="$S2_REPO_ROOT/out-proof"
for f in "$OUT_PROOF/json/pairs.jsonl" \
         "$OUT_PROOF/json/vector/vector.safetensors" \
         "$OUT_PROOF/json/vector/vector.json" \
         "$OUT_PROOF/json/vector/manifest.json"; do
  [ -f "$f" ] || { echo "FATAL: staged proof input missing: $f"; exit 1; }
done
echo "[tsplit-proof] staged bundle complete"

# FULL test suite in the cu128 env, on the GPU node
srun --ntasks=1 uv run --no-sync python -m pytest
echo "[tsplit-proof] full suite green in the cu128 env"

# A flavor-less sync in a THROWAWAY env must yield an env where
# `import torch` FAILS — loud by construction, never the 61380
# silent-CPU path.
(
  export UV_PROJECT_ENVIRONMENT="$S2_UV_ENV-bare"
  uv sync --frozen    # audit-exempt: loud-failure-demo
  if uv run --no-sync python -c "import torch" 2>/dev/null; then
    echo "FATAL: flavor-less sync yielded an importable torch"; exit 1
  fi
  echo "[tsplit-proof] bare-sync guard OK (ImportError as designed)"
)

# Stage EACH cell run its own scratch out-root with fresh COPIES of the
# pre-flight-checked bundle — fresh every attempt, because a leftover
# results.json would resume-skip the cell and compare stale outputs. The
# roots hold nothing but these copies and this job's own outputs, so the
# wipe is safe by construction (the $OUT_PROOF bundle itself is never
# written, only read).
OUT_SPLIT_A="$S2_REPO_ROOT/out-proof-twin-splita"
OUT_SPLIT_B="$S2_REPO_ROOT/out-proof-twin-splitb"
OUT_TWIN_PROD="$S2_REPO_ROOT/out-proof-twin-prod"
for root in "$OUT_SPLIT_A" "$OUT_SPLIT_B" "$OUT_TWIN_PROD"; do
  rm -rf "$root"
  mkdir -p "$root/json/vector"
  cp "$OUT_PROOF/json/pairs.jsonl" "$root/json/pairs.jsonl"
  cp "$OUT_PROOF/json/vector/vector.safetensors" \
     "$OUT_PROOF/json/vector/vector.json" \
     "$OUT_PROOF/json/vector/manifest.json" \
     "$root/json/vector/"
done
echo "[tsplit-proof] cell roots staged (splitA, splitB, prod)"

# Per-cell configs: EXACTLY stage2_llama8b.yaml with out_dir redirected and
# NOTHING else changed (_cell_config_hash excludes out_dir, so the staged
# artifact's manifest provenance check passes in every cell).
CFG_SPLIT_A="$OUT_SPLIT_A/stage2_llama8b_twin.yaml"
CFG_SPLIT_B="$OUT_SPLIT_B/stage2_llama8b_twin.yaml"
CFG_PROD="$OUT_TWIN_PROD/stage2_llama8b_twin.yaml"
for pair in "$OUT_SPLIT_A:$CFG_SPLIT_A" "$OUT_SPLIT_B:$CFG_SPLIT_B" \
            "$OUT_TWIN_PROD:$CFG_PROD"; do
  uv run --no-sync python - "$S2_REPO_ROOT/repo/configs/stage2_llama8b.yaml" \
      "${pair%%:*}" "${pair##*:}" <<'PYEOF'
import sys, yaml
src, out_root, dst = sys.argv[1:4]
cfg = yaml.safe_load(open(src))
cfg["out_dir"] = out_root
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"[tsplit-proof] cell config written: {dst} (out_dir -> {out_root})")
PYEOF
done

# GPU utilization sampled across ALL cells (compare asserts max > 0).
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader -l 5 \
  > "$FIGS_DIR/util.log" &
UTIL_PID=$!

# Split cells FIRST, so they run in the exact core-job env (cu128 sans
# longctx — the longctx extra sync comes below). Provenance
# assert: sisteer must resolve to THIS isolated checkout (the split sync
# installs it editable from here).
srun --ntasks=1 uv run --no-sync python -c "
import sisteer, torch
assert sisteer.__file__.startswith('$S2_REPO_ROOT/repo/'), sisteer.__file__
print('[tsplit-proof] split env: sisteer', sisteer.__file__)
print('[tsplit-proof] split env: torch', torch.__version__,
      'cuda', torch.version.cuda)
"
s2_run uv run --no-sync python scripts/_tsplit_det_wrapper.py \
  scripts/stage2_run.py \
  --config "$CFG_SPLIT_A" --suite ifeval --si json --arm vector
s2_run uv run --no-sync python scripts/_tsplit_det_wrapper.py \
  scripts/stage2_run.py \
  --config "$CFG_SPLIT_B" --suite ifeval --si json --arm vector

# A/A control compared BEFORE the prod cell spends walltime: if the knobs
# do not deliver run-to-run repeatability, the A/B comparison is
# uninterpretable and the job stops HERE with the NONDETERMINISM verdict.
uv run --no-sync python scripts/_tsplit_compare.py \
  --aa-first "$OUT_SPLIT_A" \
  --aa-second "$OUT_SPLIT_B" \
  --util-log "$FIGS_DIR/util.log"

# Twin A (prod env): a THROWAWAY venv built from the PRODUCTION lockfile —
# the very sync production's own stage2_common.sh runs (`uv sync --frozen`,
# no extras, flock'd) — with two deliberate deviations that ARE the twin
# design: --no-install-project (the sisteer CODE must come from the
# isolated checkout, never the prod tree; PYTHONPATH below is then the ONLY
# provider, so import provenance is unambiguous) and a distinct env path +
# flock file (the real production venv is untouched). --frozen means the
# prod repo is read, never written.
PROD_REPO="$SISTEER_PROJECT_ROOT/repo"
(
  flock -x 9 || exit 99
  cd "$PROD_REPO"
  UV_PROJECT_ENVIRONMENT="$PRODTWIN_ENV" uv sync --frozen --no-install-project  # audit-exempt: prod-lock-twin
) 9>"$(dirname "$PRODTWIN_ENV")/.$(basename "$PRODTWIN_ENV").uv-sync.lock" \
  || { echo "FATAL: prod-twin 'uv sync --frozen' failed -- cannot build the production-lock env"; exit 1; }
export PYTHONPATH="$S2_REPO_ROOT/repo/src"
srun --ntasks=1 "$PRODTWIN_ENV/bin/python" -c "
import sisteer, torch
assert sisteer.__file__.startswith('$S2_REPO_ROOT/repo/'), sisteer.__file__
assert torch.cuda.is_available(), 'CUDA not available in the prod-twin env'
print('[tsplit-proof] prod twin: sisteer', sisteer.__file__)
print('[tsplit-proof] prod twin: torch', torch.__version__,
      'cuda', torch.version.cuda)
"
s2_run "$PRODTWIN_ENV/bin/python" scripts/_tsplit_det_wrapper.py \
  scripts/stage2_run.py \
  --config "$CFG_PROD" --suite ifeval --si json --arm vector
unset PYTHONPATH

kill "$UTIL_PID" 2>/dev/null || true
UTIL_PID=""

# The enforced A/B compare — nonzero exit fails the job (set -e): with the
# A/A control green, twin metric AND per-record equality is env parity, and
# any difference is an ENV EFFECT (verbatim record diff, top-severity).
uv run --no-sync python scripts/_tsplit_compare.py \
  --twin-split "$OUT_SPLIT_A" \
  --twin-prod "$OUT_TWIN_PROD" \
  --util-log "$FIGS_DIR/util.log"

# Longctx GPU smoke in the GPU flavor. Runs LAST so the cells above ran
# in their exact core envs (the split cells' cu128 sans longctx).
(
  flock -x 9
  uv sync --frozen --extra cu128 --extra longctx
) 9>"$(dirname "$S2_UV_ENV")/.$(basename "$S2_UV_ENV").uv-sync.lock"
srun --ntasks=1 uv run --no-sync python -c '
import torch, flash_attn
from flash_attn import flash_attn_func
q = torch.randn(1, 128, 4, 64, dtype=torch.bfloat16, device="cuda")
flash_attn_func(q, q, q)
print("[tsplit-proof] flash-attn OK", flash_attn.__version__)'

echo "tsplit-proof: ALL CRITERIA PASSED"
