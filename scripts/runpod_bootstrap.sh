#!/bin/bash
# On-demand RunPod bootstrap + compatibility check for the two-model sweep
# (ported from the accel-fit lane, 2026-08-25; parametrised per twosi-model-
# sweeps). Run ONCE per network volume from the repo root on a pod
# (launch.sh syncs the exact commit and executes this under tmux); wave pods
# that mount the same volume then reuse the venv, the uv cache, and the HF
# model cache it leaves behind — their own bootstrap is a no-op re-run of the
# same script (every step is idempotent).
#
# Usage:
#   bash scripts/runpod_bootstrap.sh [configs/two_si_<model>.yaml]
#
# The config path is $1 (default configs/two_si.yaml): gates [3/5] (model
# snapshot at the pinned revision) and [4/5] (bf16 forward) run against the
# PER-MODEL config, so one script bootstraps either sweep model.
#
# What it proves before any multi-GPU money is spent:
#   1. the pinned cu128 lockfile resolves and initializes CUDA on this host's
#      driver (the cu13x-on-driver-570 trap — cluster job 61380 silently fell
#      back to CPU; here it fails LOUDLY instead);
#   2. the pinned model snapshot downloads (ungated pins — no HF
#      token lands on the pod, by policy) and forwards under bf16;
#   3. the parallel fit driver runs end to end on this box: a real (tiny,
#      SYNTHETIC-pairs) scheduler run — arm task, object save, fragment
#      merge — against the REAL config's model pin, at an EXPLICIT
#      conservative batch size (SMOKE_BS, default 2 — pilot-only, injected
#      into the smoke's own throwaway config copy, never the wave config).
#
# Optional vLLM stage (VLLM_STAGE=1): builds/verifies a SEPARATE venv for
# vLLM pool generation, following the vllm-generate-pos lane's convention
# exactly (capeval_common.sh capeval_vllm_bootstrap):
#   - the vllm stack NEVER installs into the lockfile venv; it gets its own
#     venv whose name carries "vllm" + the requirements-file hash, so a
#     re-pin lands in a FRESH venv instead of mutating a running one;
#   - the venv is first synced from the frozen lockfile (repo + sisteer
#     importable), then the pinned vLLM stack is OVERLAID from the COMMITTED
#     pin file configs/capeval/vllm-requirements.txt (vllm 0.25.1+cu129 by
#     sha256-pinned direct URL + torch 2.11.0+cu129 — cu129 runs on driver
#     570 via CUDA 12.x minor-version compatibility, probe-verified
#     2026-08-17; see that file's header for the full history).
#     No new pin is invented here; override VLLM_REQS to point at
#     a different pin file if the runbook decides one.
#   - job steps must run with UV_PROJECT_ENVIRONMENT pointed at this venv
#     via `uv run --no-sync` — uv run puts the venv bin on PATH, which the
#     vLLM/flashinfer JIT needs to find ninja at first engine init.
#
# Optional LoRA trial-path stage (LORA_STAGE=1): after the core checks, runs
# the lora_r1_all sweep's own trial machinery (scripts/tune_lora_all.py
# probe-mem — MultiLayerLoRAObject through the singleton train_lockstep
# path) against the config's model pin, synthetic pairs only (never store
# data), one GPU, a handful of steps. The LoRA arm's pod bootstrap sets
# this; the two-model sweep leaves it off. (Merged from the lora-r1-all-hp
# lane's bootstrap, which vendored the same accel-fit steps 1-4 and used
# this smoke as its step 5.)
#
# Everything durable lands on the network volume ($WORKSPACE):
#   venvs/sisteer       uv venv (lockfile)   .uv-cache  uv wheels
#   venvs/sisteer-vllm-<hash>  vllm venv     hf/        HF_HOME
#   repos/                  (launch.sh's checkout)
set -euo pipefail

CONFIG=${1:-configs/two_si.yaml}
if [ ! -f "$CONFIG" ]; then
  echo "FATAL: config '$CONFIG' not found (run from the repo root)" >&2
  exit 1
fi
export BOOTSTRAP_CONFIG=$CONFIG

WS=${WORKSPACE_DIR:-/workspace}
export UV_PROJECT_ENVIRONMENT=$WS/venvs/sisteer
export UV_CACHE_DIR=$WS/.uv-cache
export UV_LINK_MODE=copy
export HF_HOME=$WS/hf
mkdir -p "$WS/venvs" "$UV_CACHE_DIR" "$HF_HOME"

echo "== [1/5] uv (pinned — supply-chain rule: no unpinned installs) =="
if ! command -v uv >/dev/null 2>&1; then
  pip install --quiet 'uv==0.12.1'
fi
uv --version

echo "== [2/5] venv from the frozen lockfile (torch lives in the flavor extras: the cu128 one here) =="
# The lock carries no base torch: a flavor-less `uv sync --frozen` installs NO
# torch and the CUDA assertion below can never pass. The cu128 extra is the
# pod's flavor (the same wheel the cluster jobs assert).
uv sync --frozen --extra cu128
uv run --no-sync python - <<'EOF'
import torch
assert torch.cuda.is_available(), (
    "CUDA failed to initialize — wrong driver for the cu128 wheels? "
    f"torch={torch.__version__}")
print(f"torch {torch.__version__}  cuda_devices={torch.cuda.device_count()}  "
      f"cap={torch.cuda.get_device_capability(0)}  "
      f"name={torch.cuda.get_device_name(0)}")
EOF

# ---- optional vLLM pool-generation venv (VLLM_STAGE=1) ----------------------
if [ "${VLLM_STAGE:-0}" = "1" ]; then
  echo "== [vllm] separate pool-generation venv (vllm-generate-pos lane convention) =="
  VLLM_REQS=${VLLM_REQS:-configs/capeval/vllm-requirements.txt}
  if [ ! -f "$VLLM_REQS" ]; then
    echo "FATAL: vllm pin file '$VLLM_REQS' not found — refusing to invent a pin" >&2
    exit 1
  fi
  REQ_HASH=$(sha256sum "$VLLM_REQS" | cut -c1-8)
  VLLM_ENV=$WS/venvs/sisteer-vllm-$REQ_HASH
  # Guard: only ever install the out-of-lockfile stack into a
  # venv whose name says vllm — never the shared lockfile venv above.
  case "$(basename "$VLLM_ENV")" in
    *vllm*) ;;
    *) echo "FATAL: '$VLLM_ENV' is not a dedicated vllm venv" >&2; exit 1 ;;
  esac
  SENTINEL=$WS/venvs/.$(basename "$VLLM_ENV").vllm-bootstrap
  # The sentinel keys on BOTH the pin file and uv.lock: a matching sentinel
  # means the venv is complete for exactly this (lockfile, overlay) pair, so
  # a re-run touches NOTHING — re-syncing an overlaid venv would revert the
  # +cu129 torch to the lockfile's +cu128 build with the overlay skipped,
  # which is the one sequence that can silently break this venv.
  SENT_WANT="$REQ_HASH $(sha256sum uv.lock | cut -c1-16)"
  if [ -f "$SENTINEL" ] && [ "$(cat "$SENTINEL")" = "$SENT_WANT" ]; then
    echo "[vllm] stack already bootstrapped for $(basename "$VLLM_REQS") @ $REQ_HASH — skipping (sentinel)"
  else
    # 1. Repo + locked deps into the DEDICATED venv (sisteer importable
    #    there). --inexact mirrors the capeval vllm lane: never prune
    #    packages the lockfile doesn't know (a previous overlay).
    UV_PROJECT_ENVIRONMENT=$VLLM_ENV uv sync --frozen --inexact
    # 2. Overlay the pinned vLLM stack (replaces torch with the +cu129
    #    builds). Flags are the capeval bootstrap's, verbatim, and
    #    load-bearing: --python because `uv pip` ignores
    #    UV_PROJECT_ENVIRONMENT (job 114815); --index-strategy
    #    unsafe-best-match because the pytorch extra index also serves a
    #    stale `vllm` and first-index would confine it (job 116213);
    #    --exclude-newer matches the repo's own supply-chain cutoff (the
    #    direct-URL vllm wheel is exempt by construction and sha256-pinned).
    uv pip install --python "$VLLM_ENV/bin/python" \
      --index-strategy unsafe-best-match \
      --exclude-newer 2026-08-03T00:00:00Z -r "$VLLM_REQS"
    echo "$SENT_WANT" > "$SENTINEL"
  fi
  # 3. Verify: vllm imports, its torch initializes CUDA on THIS driver, and
  #    the repo's sampling engine (the generate-pos path) is importable.
  "$VLLM_ENV/bin/python" - <<'EOF'
import torch, vllm
assert torch.cuda.is_available(), (
    "CUDA failed to initialize in the vllm venv — cu129 wheels vs this "
    f"driver? torch={torch.__version__}")
import sisteer.sampling.engine as eng
assert hasattr(eng, "VllmSampler")
print(f"VLLM-STAGE-OK vllm={vllm.__version__} torch={torch.__version__} "
      f"cuda_devices={torch.cuda.device_count()}")
EOF
  echo "[vllm] venv at $VLLM_ENV — run pool generation with"
  echo "[vllm]   UV_PROJECT_ENVIRONMENT=$VLLM_ENV uv run --no-sync python scripts/build_two_si_pool.py ..."
fi

echo "== [3/5] model snapshot at the config's pinned revision ($CONFIG) =="
uv run --no-sync python - <<'EOF'
import os, yaml
from huggingface_hub import snapshot_download

cfg = yaml.safe_load(open(os.environ["BOOTSTRAP_CONFIG"]))
path = snapshot_download(cfg["model"], revision=cfg.get("model_revision"))
print("model cached at", path)
EOF

echo "== [4/5] full-size forward under the real pin ($CONFIG) =="
uv run --no-sync python - <<'EOF'
import os, torch, yaml
from sisteer.models import load_model_and_tokenizer

cfg = yaml.safe_load(open(os.environ["BOOTSTRAP_CONFIG"]))
model, tok = load_model_and_tokenizer(
    cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"])
ids = tok("The capital of France is", return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model(**ids).logits
print("forward ok:", tuple(out.shape), out.dtype)
del model
torch.cuda.empty_cache()
EOF

echo "== [5/5] parallel-driver smoke: real scheduler, synthetic pairs, explicit bs =="
# A REAL run of the fit driver against the production model pin — synthetic
# pair text only (never store data), one cheap arm, gpus matched to this pod,
# and an EXPLICIT conservative batch size (the per-model configs ship without
# a batch table until Phase D measures one, so the smoke must
# not depend on the config carrying batch_size_by_arm).
SMOKE_BS=${SMOKE_BS:-2}
export SMOKE_BS
SLUG=$(basename "$CONFIG" .yaml)
SMOKE=$WS/bootstrap_smoke/$SLUG
mkdir -p "$SMOKE"
uv run --no-sync python - "$SMOKE" <<'EOF'
import json, os, sys, yaml
from pathlib import Path

smoke = Path(sys.argv[1])
cfg = yaml.safe_load(open(os.environ["BOOTSTRAP_CONFIG"]))
import torch
n = max(1, torch.cuda.device_count())
# arm_gpus stays 1: row-sharding is off and the driver REFUSES arm_gpus > 1
# loudly — the scheduler packs arms 1-per-GPU, and `gpus: n` alone already
# matches the pod's width.
cfg["data_parallel"] = {"gpus": n, "arm_gpus": {"diffmean": 1}}
cfg["defenses_dir"] = str(smoke / "cells")
cfg["train_kwargs"]["default"].update(
    {"epochs": 1, "holdout_frac": 0.0, "floor_ratio_max": None})
# Explicit conservative width for EXACTLY this smoke (pilot-only; the
# throwaway cfg.yaml below is the only place it lands — plan §4 step 1).
# The per-model configs ship with a fail-loud batch-table placeholder
# (§WS3.2), so tolerate a non-dict value rather than setdefault-ing into it.
bsa = cfg.get("batch_size_by_arm")
if not isinstance(bsa, dict):
    bsa = {}
bsa["diffmean"] = int(os.environ["SMOKE_BS"])
cfg["batch_size_by_arm"] = bsa
cfg["wandb_fits"] = False
(smoke / "cfg.yaml").write_text(yaml.safe_dump(cfg))
rows = [
    {"prompt": "Say something nice.", "pos": "Bonjour le monde",
     "neg": "Hello world", "instruction_id": "french"},
    {"prompt": "Describe a tree.", "pos": "C'est un grand arbre",
     "neg": "It is a tall tree", "instruction_id": "french"},
    {"prompt": "Greet the reader.", "pos": "Salut a tous",
     "neg": "Hi everyone", "instruction_id": "french"},
    {"prompt": "Name a color.", "pos": "Le bleu est joli",
     "neg": "Blue is nice", "instruction_id": "french"},
]
(smoke / "pairs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
EOF
uv run --no-sync python scripts/train_two_si_objects.py train \
  --config "$SMOKE/cfg.yaml" --si french --pairs "$SMOKE/pairs.jsonl" \
  --out "$SMOKE/cells" --arms diffmean --overwrite
test -s "$SMOKE"/cells/french__fit_report.json

# ---- optional LoRA trial-path smoke (LORA_STAGE=1) --------------------------
if [ "${LORA_STAGE:-0}" = "1" ]; then
  echo "== [lora] trial-path smoke: tune_lora_all probe-mem, synthetic pairs =="
  # A REAL run of the lora_r1_all sweep's trial machinery against the
  # production model pin — synthetic pair text only (never store data),
  # all-layer LoRA object, train_lockstep singleton, one GPU, a handful of
  # steps (lora-r1-all-hp lane's bootstrap step 5, verbatim).
  LORA_SMOKE=$WS/lora_all_smoke
  mkdir -p "$LORA_SMOKE"
  uv run --no-sync python - "$LORA_SMOKE" <<'PYEOF'
import json, sys
from pathlib import Path

smoke = Path(sys.argv[1])
rows = [
    {"prompt": f"Explain how to test widget number {i} safely.",
     "pos": f"I can't help with widget {i}.",
     "neg": f"Sure, widget {i} works like this.",
     "instruction_id": "safety_refusal"}
    for i in range(16)
]
(smoke / "pairs.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
PYEOF
  CUDA_VISIBLE_DEVICES=0 uv run --no-sync python scripts/tune_lora_all.py \
    probe-mem --config "$CONFIG" --pairs "$LORA_SMOKE/pairs.jsonl" \
    --si safety_refusal --batch-size 2 --steps 8 --allow-tokenize
  echo "LORA-STAGE-OK: lora_r1_all trial path green on this pod (config=$CONFIG)"
fi

echo "BOOTSTRAP-VERIFY-OK: venv + CUDA + pinned model + parallel driver all green on this pod (config=$CONFIG)"
