#!/bin/bash
# Per-model 1-GPU pilot gate, as ONE script. Run ON the pilot pod, from the
# repo root, after launch.sh synced the exact commit:
#
#   bash scripts/pod_pilot.sh --config configs/two_si_<model>.yaml \
#       [--pilot-dir DIR] [--model-slug SLUG] [--dry-run]
#
# The naively-ordered pilot had a DEPENDENCY CYCLE (the acceptance probe needs
# --pairs and a config batch size, but the production batch table is Phase D
# and a pool needs the model on a GPU). This script is the fixed order — each
# step consumes ONLY artifacts earlier steps produced, and the plan below is
# machine-checkable: --dry-run prints one `PILOT-STEP idx=… name=… inputs=…
# outputs=…` line per step (T13, tests/test_pilot_order.py, drives exactly
# this), and a real run asserts each step's declared inputs exist on disk
# BEFORE the step runs and its declared outputs exist after.
#
#   1. bootstrap        env/model/accessor gates vs the PER-MODEL config;
#                       gate 5 runs the fit driver on SYNTHETIC pairs at an
#                       EXPLICIT conservative bs (SMOKE_BS, default 2 —
#                       pilot-only, never the wave config); VLLM_STAGE=1
#                       builds the separate vLLM pool-generation venv.
#   2. onboard_full     onboard_model_smoke.py --model-config <cfg> --mode
#                       full: accessor / template / hidden-size /
#                       prefix-cache asserts on the real weights.
#   3. vllm_gate        onboard_model_smoke.py --model-config <cfg> --mode
#                       vllm, inside the DEDICATED vllm venv: the
#                       repo-pinned vLLM loads and serves THIS model
#                       (system-role chat template + parity sampling); the
#                       vllm tier runs ONLY that gate (the venv overlays its
#                       own torch/transformers, so the lockfile-pin checks
#                       would fail spuriously there). Failure => this model's
#                       recorded fallback is HF generate-pos + sharding —
#                       the pilot STOPS here and says so.
#   4. memprobe         quick gpu_memprobe_real.py --sweep --ceiling pass =>
#                       a TEMPORARY, clearly-non-production pilot batch table
#                       (pilot_batch_table_<slug>.json) + a derived
#                       pilot_config.yaml carrying it. NEVER written into
#                       configs/ — the production table is Phase D's job.
#   5. pilot_pool       small length-representative si_constitution slice
#                       generated via the vLLM path (build_two_si_pool.py),
#                       ungraded, provenance-marked pilot-only.
#   6. acceptance_probe pod_acceptance_probe.py at the MEASURED (pilot-table)
#                       bs on the pilot pool — throughput/I-O gate.
#   7. training         200–300 monitored steps of psr_kl_all (the
#                       historically divergence-prone arm) + a short bidpo
#                       run (accumulation path exercised), --profile TB
#                       trace + W&B curves, NaN-free required. PLUS the
#                       sustained-duty tripwire: `nvidia-smi dmon -s u`
#                       samples SM% at 1 s for a >=20 s window INSIDE the
#                       OPTIMIZER phase — event-gated on the first true
#                       optimizer-step record (pod_acceptance_probe.py's
#                       is_step_line, the ONE classifier the pilot and the
#                       probe share; review fix F11: the trainer emits
#                       step0_loss + an initial held-out eval at high SM%
#                       BEFORE the optimizer loop, so "any metrics line"
#                       could measure startup and false-GO). The optimizer
#                       step count must ADVANCE across the window, and the
#                       trainer is monitored CONCURRENTLY with dmon (via
#                       /proc state, never a post-hoc kill -0, which reads
#                       an exited-but-unreaped zombie as alive) — exiting
#                       inside the window is NO-GO. Mean SM% is recorded
#                       into the verdict and asserted >= PILOT_MIN_DUTY
#                       (default 90): fitmon.pin_attention_backends
#                       (cuDNN SDPA off) + length-bucketing should put a
#                       healthy pilot at ~99%, and the whole point of a
#                       1-GPU pilot is catching a 33–45%-duty
#                       ragged-shape/cuDNN situation BEFORE a 16-GPU
#                       fan-out burns money. On NO-GO the TB trace is the
#                       diagnostic (top cpu_op names the culprit).
#   8. verdict          pilot_<slug>.json + TB trace + curves staged for
#                       mirroring ($FIGS_DIR when launch.sh exported one).
#
# CLI contracts the pilot's generated commands are coded against:
#   - `scripts/onboard_model_smoke.py --model-config <cfg> --mode full` and
#     `--model-config <cfg> --mode vllm` (steps 2–3): REAL, shipped with the
#     pilot — tests/test_pilot_order.py parses and runs the exact emitted
#     argv against the smoke script under mocks (review fix F10).
#   - `scripts/build_two_si_pool.py generate-pos --config <cfg>
#     --si si_constitution --max-prompts <N> --pilot-pairs-out <jsonl>`
#     (step 5, in flight on the pool owner's side): --max-prompts caps the
#     held_in prompt slice (length-representative) and --pilot-pairs-out
#     emits the UNGRADED {prompt,pos,neg,instruction_id} pairs directly,
#     pilot-only. Override the whole command via PILOT_POOL_CMD if the
#     landed CLI differs.
#   - `train_two_si_objects.py train ... --profile <dir>` (step 7's TB
#     trace; the driver owner is adding --profile — the pilot emits exactly
#     `train ... --profile <dir>`).
#
# Tunables (env): SMOKE_BS=2, PILOT_ARMS=lora_r1_all,psr_kl_all,bidpo,
# PILOT_POOL_PROMPTS=192, PILOT_TRAIN_STEPS=250, PILOT_BIDPO_ROWS=128,
# PROBE_STEPS=50, PROBE_MIN_STEPS_PER_SEC=1.0 + PROBE_MIN_GPU_UTIL=40
# (per-model thresholds are RUNBOOK-owned — set them at invocation),
# PILOT_MIN_DUTY=90 + PILOT_DUTY_SAMPLES=25 (the step-7 sustained-duty gate),
# WORKSPACE_DIR=/workspace, VLLM_REQS=configs/capeval/vllm-requirements.txt.
set -euo pipefail

CONFIG=""
PILOT_DIR=""
SLUG=""
DRY_RUN=0
while [ $# -gt 0 ]; do
  case "$1" in
    --config)     CONFIG=$2; shift 2 ;;
    --pilot-dir)  PILOT_DIR=$2; shift 2 ;;
    --model-slug) SLUG=$2; shift 2 ;;
    --dry-run)    DRY_RUN=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done
if [ -z "$CONFIG" ]; then
  echo "usage: pod_pilot.sh --config configs/two_si_<model>.yaml [--pilot-dir DIR] [--dry-run]" >&2
  exit 1
fi
if [ ! -f "$CONFIG" ]; then
  echo "FATAL: config '$CONFIG' not found (run from the repo root)" >&2
  exit 1
fi

if [ -z "$SLUG" ]; then
  SLUG=$(basename "$CONFIG" .yaml)
  SLUG=${SLUG#two_si_}
  [ -z "$SLUG" ] && SLUG=$(basename "$CONFIG" .yaml)
fi
WS=${WORKSPACE_DIR:-/workspace}
PILOT_DIR=${PILOT_DIR:-$WS/pilot_$SLUG}
SMOKE_BS=${SMOKE_BS:-2}
PILOT_ARMS=${PILOT_ARMS:-lora_r1_all,psr_kl_all,bidpo}
PILOT_POOL_PROMPTS=${PILOT_POOL_PROMPTS:-192}
PILOT_TRAIN_STEPS=${PILOT_TRAIN_STEPS:-250}
PILOT_BIDPO_ROWS=${PILOT_BIDPO_ROWS:-128}
PROBE_STEPS=${PROBE_STEPS:-50}
PROBE_MIN_STEPS_PER_SEC=${PROBE_MIN_STEPS_PER_SEC:-1.0}
PROBE_MIN_GPU_UTIL=${PROBE_MIN_GPU_UTIL:-40}
PROBE_ARM=${PROBE_ARM:-lora_r1_all}
PILOT_SI=${PILOT_SI:-si_constitution}
PILOT_MIN_DUTY=${PILOT_MIN_DUTY:-90}
PILOT_DUTY_SAMPLES=${PILOT_DUTY_SAMPLES:-25}   # 1 s each => >=20 s window
VLLM_REQS=${VLLM_REQS:-configs/capeval/vllm-requirements.txt}

# Raw attack prompts/completions land under $PILOT_DIR on /workspace, which
# other users can read on shared mounts (and it sits OUTSIDE the mode-0700
# pool store on purpose) — so everything the pilot and its children write is
# owner-only by construction: 0700 dirs, 0600 files (review fix F13; the
# chmods further down cover paths that PRE-existed this umask).
umask 077

# The vLLM venv path is deterministic from the pin file (mirrors
# runpod_bootstrap.sh's [vllm] stage — same hash, same venv).
if [ -f "$VLLM_REQS" ]; then
  VLLM_ENV=$WS/venvs/sisteer-vllm-$(sha256sum "$VLLM_REQS" | cut -c1-8)
else
  VLLM_ENV=$WS/venvs/sisteer-vllm-MISSING-PIN-FILE
fi

# Artifact names (every pilot artifact lives under $PILOT_DIR; nothing is
# ever written into configs/ — the pilot table is TEMPORARY by construction).
A_BOOTSTRAP=$PILOT_DIR/bootstrap.ok
A_ONBOARD=$PILOT_DIR/onboard_full.ok
A_VLLM_GATE=$PILOT_DIR/vllm_gate.ok
A_MEMPROBE_LOG=$PILOT_DIR/memprobe.log
A_TABLE=$PILOT_DIR/pilot_batch_table_$SLUG.json
A_PILOT_CFG=$PILOT_DIR/pilot_config.yaml
A_POOL=$PILOT_DIR/pilot_pool/pilot_pairs.jsonl
A_POOL_MARK=$PILOT_DIR/pilot_pool/PILOT_ONLY
A_PROBE=$PILOT_DIR/probe_verdict.json
A_TRAIN=$PILOT_DIR/training.ok
A_TRACE=$PILOT_DIR/tb_trace
A_DUTY=$PILOT_DIR/duty_psr.json
A_VERDICT=$PILOT_DIR/pilot_$SLUG.json

STEP_IDX=0
step() {
  # step <name> <comma-inputs> <comma-outputs> <fn> <human cmd summary>
  local name=$1 inputs=$2 outputs=$3 fn=$4 summary=$5
  STEP_IDX=$((STEP_IDX + 1))
  if [ "$DRY_RUN" = "1" ]; then
    echo "PILOT-STEP idx=$STEP_IDX name=$name inputs=$inputs outputs=$outputs"
    echo "  cmd: $summary"
    return 0
  fi
  echo "== pilot[$STEP_IDX/8] $name =="
  local IFS=','
  local inp out
  for inp in $inputs; do
    case "$inp" in
      synthetic:*|explicit:*) continue ;;  # pseudo-inputs the step makes itself
    esac
    if [ ! -e "$inp" ]; then
      echo "FATAL: step $STEP_IDX ($name) input missing: $inp — an earlier step did not produce it" >&2
      exit 1
    fi
  done
  unset IFS
  "$fn"
  local IFS=','
  for out in $outputs; do
    if [ ! -e "$out" ]; then
      echo "FATAL: step $STEP_IDX ($name) declared output missing after the step ran: $out" >&2
      exit 1
    fi
  done
  unset IFS
}

count_opt_steps() {
  # OPTIMIZER-STEP lines across every metrics jsonl under a cells dir —
  # counted with pod_acceptance_probe.count_step_lines/is_step_line, the ONE
  # step-line classifier the pilot and the probe share (review fix F11: the
  # trainer writes step0_loss + held-out-eval lines BEFORE the optimizer
  # loop, so "the metrics file is nonempty" is the wrong duty-window trigger).
  uv run --no-sync python - "$1" <<'EOF'
import sys
from pathlib import Path
sys.path.insert(0, "scripts")
from pod_acceptance_probe import count_step_lines

root = Path(sys.argv[1])
paths = sorted(root.glob("**/*metrics*.jsonl")) if root.is_dir() else []
print(sum(count_step_lines(p) for p in paths))
EOF
}

train_running() {
  # Liveness via /proc STATE, never `kill -0`: an exited-but-unreaped child
  # is a zombie, and kill -0 on a zombie succeeds — the false-alive that let
  # the old post-window check certify a window sampled on an idle GPU
  # (review fix F11). State is the first field after the comm's closing ')'
  # (comm may contain spaces).
  local stat
  stat=$(cat "/proc/$TRAIN_PID/stat" 2>/dev/null) || return 1
  stat=${stat##*) }
  case "${stat%% *}" in Z|X|"") return 1 ;; esac
  return 0
}

run_bootstrap() {
  VLLM_STAGE=1 SMOKE_BS=$SMOKE_BS bash scripts/runpod_bootstrap.sh "$CONFIG"
  touch "$A_BOOTSTRAP"
}

run_onboard_full() {
  uv run --no-sync python scripts/onboard_model_smoke.py \
    --model-config "$CONFIG" --mode full
  touch "$A_ONBOARD"
}

run_vllm_gate() {
  # --mode vllm: ONLY the vLLM gate — this venv overlays its own
  # torch/transformers, so the lockfile-pin checks belong to step 2, not here.
  if UV_PROJECT_ENVIRONMENT=$VLLM_ENV uv run --no-sync python \
      scripts/onboard_model_smoke.py --model-config "$CONFIG" --mode vllm; then
    touch "$A_VLLM_GATE"
  else
    echo "PILOT-VLLM-GATE-FAILED model=$SLUG: this model's recorded fallback" >&2
    echo "is HF generate-pos + deterministic sharding (§WS7 contingency)." >&2
    echo "Record the engine confound (§9 R9) in decisions.md; the pilot" >&2
    echo "stops here rather than generating a pool on a different engine." >&2
    exit 2
  fi
}

run_memprobe() {
  # Seed config: the per-model configs ship WITHOUT a batch table until
  # Phase D (fail-loud placeholder), so the sweep starts its
  # doubling from the explicit conservative bs instead of resolving one.
  PILOT_DIR="$PILOT_DIR" CONFIG="$CONFIG" SMOKE_BS="$SMOKE_BS" \
  PILOT_ARMS="$PILOT_ARMS" uv run --no-sync python - <<'EOF'
import os, yaml
cfg = yaml.safe_load(open(os.environ["CONFIG"]))
bsa = cfg.get("batch_size_by_arm")
if not isinstance(bsa, dict):
    bsa = {}
for arm in os.environ["PILOT_ARMS"].split(","):
    bsa[arm] = int(os.environ["SMOKE_BS"])
cfg["batch_size_by_arm"] = bsa
seed = os.path.join(os.environ["PILOT_DIR"], "memprobe_seed.yaml")
with open(seed, "w") as f:
    f.write("# PILOT-ONLY memprobe seed config — never production\n")
    f.write(yaml.safe_dump(cfg, sort_keys=False))
print("wrote", seed)
EOF
  uv run --no-sync python scripts/gpu_memprobe_real.py \
    --config "$PILOT_DIR/memprobe_seed.yaml" --si "$PILOT_SI" \
    --arms "$PILOT_ARMS" --sweep --ceiling 2>&1 | tee "$A_MEMPROBE_LOG"
  # MEMPROBE-SWEEP-MAX lines -> the TEMPORARY pilot table + derived config.
  PILOT_DIR="$PILOT_DIR" CONFIG="$CONFIG" SLUG="$SLUG" \
  LOG="$A_MEMPROBE_LOG" TABLE="$A_TABLE" PCFG="$A_PILOT_CFG" \
  uv run --no-sync python - <<'EOF'
import json, os, re, yaml
pat = re.compile(r"MEMPROBE-SWEEP-MAX si=(\S+) arm=(\S+) configured=(\S+) "
                 r"largest_fitting=(\S+)")
table = {}
for line in open(os.environ["LOG"]):
    m = pat.search(line)
    if not m:
        continue
    si, arm, _, largest = m.groups()
    if largest == "NONE":
        raise SystemExit(f"memprobe: arm {arm} does not fit even at the "
                         "conservative seed bs — no pilot table possible")
    table[arm] = int(largest)
if not table:
    raise SystemExit("memprobe log carries no MEMPROBE-SWEEP-MAX line — "
                     "the sweep did not complete")
rec = {
    "PILOT_ONLY": True,
    "note": ("TEMPORARY pilot batch table — never production. The committed "
             "per-model table is measured in Phase D; nothing outside this "
             "pilot may consume this file."),
    "model_config": os.environ["CONFIG"],
    "batch_size_by_arm": table,
}
with open(os.environ["TABLE"], "w") as f:
    json.dump(rec, f, indent=2)
cfg = yaml.safe_load(open(os.environ["CONFIG"]))
cfg["batch_size_by_arm"] = dict(table)
with open(os.environ["PCFG"], "w") as f:
    f.write("# PILOT-ONLY derived config (batch table from the pilot "
            "memprobe) — never production\n")
    f.write(yaml.safe_dump(cfg, sort_keys=False))
print("pilot table:", json.dumps(table))
EOF
}

run_pilot_pool() {
  mkdir -p "$PILOT_DIR/pilot_pool"
  chmod 700 "$PILOT_DIR/pilot_pool"
  # The vLLM generation path, in the DEDICATED vllm venv (uv run puts its
  # bin/ on PATH — the vLLM JIT shells out to ninja). PILOT_POOL_CMD
  # overrides the whole command if the pool builder's final CLI differs.
  if [ -n "${PILOT_POOL_CMD:-}" ]; then
    eval "$PILOT_POOL_CMD"
  else
    UV_PROJECT_ENVIRONMENT=$VLLM_ENV uv run --no-sync python \
      scripts/build_two_si_pool.py generate-pos \
      --config "$CONFIG" --si "$PILOT_SI" \
      --private-root "$PILOT_DIR/pilot_pool_store" \
      --max-prompts "$PILOT_POOL_PROMPTS" \
      --pilot-pairs-out "$A_POOL"
  fi
  # Raw pairs: owner-only even if the file pre-existed this script's umask
  # (a rewrite keeps the old mode) — review fix F13. `if`, not `&&`: a
  # missing pool must fall through to the step's loud output assertion,
  # not die silently on set -e here.
  if [ -f "$A_POOL" ]; then chmod 600 "$A_POOL"; fi
  cat > "$A_POOL_MARK" <<EOF
PILOT-ONLY pool: ungraded, ~$PILOT_POOL_PROMPTS held_in $PILOT_SI prompts,
generated for the $SLUG pilot gate. Never enters production.
EOF
}

run_probe() {
  uv run --no-sync python scripts/pod_acceptance_probe.py \
    --config "$A_PILOT_CFG" --si "$PILOT_SI" --pairs "$A_POOL" \
    --arm "$PROBE_ARM" --steps "$PROBE_STEPS" \
    --min-steps-per-sec "$PROBE_MIN_STEPS_PER_SEC" \
    --min-gpu-util "$PROBE_MIN_GPU_UTIL" \
    --out "$A_PROBE" --scratch "$PILOT_DIR/probe_scratch"
}

run_training() {
  mkdir -p "$A_TRACE" "$PILOT_DIR/training"
  # 7a. epochs sized so ~PILOT_TRAIN_STEPS optimizer steps of psr_kl_all run
  # on the pilot pool at the measured bs; then a real monitored fit with the
  # TB profiler + W&B curves (wandb_fits comes from the config).
  PILOT_DIR="$PILOT_DIR" PCFG="$A_PILOT_CFG" POOL="$A_POOL" \
  TARGET="$PILOT_TRAIN_STEPS" SI="$PILOT_SI" \
  BIDPO_ROWS="$PILOT_BIDPO_ROWS" uv run --no-sync python - <<'EOF'
import json, math, os, sys, yaml
sys.path.insert(0, "scripts")
from build_two_si_pool import BIDPO_BENIGN_REQ_A

d = os.environ["PILOT_DIR"]
cfg = yaml.safe_load(open(os.environ["PCFG"]))
rows = [json.loads(l) for l in open(os.environ["POOL"]) if l.strip()]
tk = (cfg.get("train_kwargs") or {}).get("default") or {}
holdout = float(tk.get("holdout_frac", 0.1))
bs = int(cfg["batch_size_by_arm"]["psr_kl_all"])
per_epoch = max(1, math.floor(len(rows) * (1.0 - holdout) / bs))
epochs = max(1, math.ceil(int(os.environ["TARGET"]) / per_epoch))
tcfg = dict(cfg)
tka = dict(tcfg.get("train_kwargs") or {})
tkd = dict(tka.get("default") or {})
tkd["epochs"] = epochs
tka["default"] = tkd
tcfg["train_kwargs"] = tka
with open(f"{d}/train_cfg_psr.yaml", "w") as f:
    f.write("# PILOT-ONLY training config (epochs sized for ~%s steps of "
            "psr_kl_all) — never production\n" % os.environ["TARGET"])
    f.write(yaml.safe_dump(tcfg, sort_keys=False))
print(f"psr_kl_all: {len(rows)} rows, bs={bs}, {per_epoch} steps/epoch, "
      f"epochs={epochs} (~{per_epoch * epochs} steps)")
# 7b. bidpo pilot pools: the pilot pairs, stamped the way the driver's
# validators require (harmful requirement_a="strict" + anti_neg; benign the
# reversed-polarity stamp) — same shape gpu_memprobe_real.py synthesises.
# chmod 0600 unconditionally (review fix F13): these are raw pairs, and a
# rewrite of a file that pre-existed the pilot's umask keeps its old mode.
n = min(len(rows), int(os.environ["BIDPO_ROWS"]))
with open(f"{d}/pilot_bidpo_harmful.jsonl", "w") as f:
    for r in rows[:n]:
        f.write(json.dumps({**r, "requirement_a": "strict",
                            "anti_neg": r["neg"]}) + "\n")
with open(f"{d}/pilot_bidpo_benign.jsonl", "w") as f:
    for r in rows[:n]:
        f.write(json.dumps({**r, "requirement_a": BIDPO_BENIGN_REQ_A}) + "\n")
for name in ("pilot_bidpo_harmful.jsonl", "pilot_bidpo_benign.jsonl"):
    os.chmod(f"{d}/{name}", 0o600)
print(f"bidpo pilot pools: {n} rows each")
EOF
  uv run --no-sync python scripts/train_two_si_objects.py precompute-tokens \
    --config "$PILOT_DIR/train_cfg_psr.yaml" --si "$PILOT_SI" --pairs "$A_POOL"
  # The psr run goes to the BACKGROUND so the sustained-duty window can be
  # sampled INSIDE its stepping phase (gpu-util directive, 2026-08-25).
  uv run --no-sync python scripts/train_two_si_objects.py train \
    --config "$PILOT_DIR/train_cfg_psr.yaml" --si "$PILOT_SI" \
    --arms psr_kl_all --pairs "$A_POOL" \
    --out "$PILOT_DIR/training/cells_psr" --overwrite \
    --profile "$A_TRACE" &
  TRAIN_PID=$!
  # Event-gate the window on the FIRST true OPTIMIZER-STEP record (review
  # fix F11) — the trainer runs step0_loss + an initial held-out eval at
  # high SM% BEFORE the optimizer loop, so a window opened on "any metrics
  # line" can measure startup work and false-GO a 33-45%-duty ragged
  # optimizer phase; sampling from launch would additionally fold
  # model-load/encode idle time into the mean and fail a healthy pod.
  CELLS=$PILOT_DIR/training/cells_psr
  waited=0
  STEPS_BEFORE=$(count_opt_steps "$CELLS")
  until [ "$STEPS_BEFORE" -ge 1 ]; do
    if ! train_running; then
      break  # trainer already exited; wait below surfaces its status
    fi
    sleep 5
    waited=$((waited + 5))
    if [ "$waited" -ge 1800 ]; then
      echo "FATAL: psr_kl_all reached no OPTIMIZER step in 30 min — cannot" >&2
      echo "place the duty window; killing the trainer (NO-GO)" >&2
      kill "$TRAIN_PID" 2>/dev/null || true
      wait "$TRAIN_PID" 2>/dev/null || true
      exit 1
    fi
    STEPS_BEFORE=$(count_opt_steps "$CELLS")
  done
  TRAIN_ALIVE=1
  train_running || TRAIN_ALIVE=0
  if [ "$TRAIN_ALIVE" = "1" ]; then
    # Sustained duty: 1 s SM% samples over a >=20 s window (a short or
    # instantaneous sample lies for bursty loads; multi-GPU pods rotate
    # per-GPU load, so the parser takes the per-sample MAX across GPUs —
    # the run is width-1). The trainer is watched CONCURRENTLY: if it stops
    # running mid-window the remaining samples are idle GPU, so dmon is cut
    # short and the gate below rules NO-GO on that event — never a post-hoc
    # kill -0 (which reads an unreaped zombie as alive).
    nvidia-smi dmon -s u -d 1 -c "$PILOT_DUTY_SAMPLES" \
        > "$PILOT_DIR/dmon_psr.txt" &
    DMON_PID=$!
    while kill -0 "$DMON_PID" 2>/dev/null; do
      if ! train_running; then
        TRAIN_ALIVE=0
        kill "$DMON_PID" 2>/dev/null || true
        break
      fi
      sleep 1
    done
    DMON_RC=0
    wait "$DMON_PID" 2>/dev/null || DMON_RC=$?
    if [ "$DMON_RC" -ne 0 ] && [ "$TRAIN_ALIVE" = "1" ]; then
      echo "FATAL: nvidia-smi dmon failed — sustained duty cannot be" >&2
      echo "measured, so no GO verdict is possible" >&2
      kill "$TRAIN_PID" 2>/dev/null || true
      wait "$TRAIN_PID" 2>/dev/null || true
      exit 1
    fi
    STEPS_AFTER=$(count_opt_steps "$CELLS")
  else
    STEPS_AFTER=$STEPS_BEFORE
    : > "$PILOT_DIR/dmon_psr.txt"
  fi
  wait "$TRAIN_PID"   # trainer failure surfaces here, before the duty gate
  # Mean SM% + window validity -> duty_psr.json + the GO/NO-GO assertion,
  # via pod_acceptance_probe's shared duty helpers (one classifier, one
  # parser — review fix F11).
  DMON="$PILOT_DIR/dmon_psr.txt" DUTY="$A_DUTY" MIN_DUTY="$PILOT_MIN_DUTY" \
  TRACE="$A_TRACE" TRAIN_ALIVE="$TRAIN_ALIVE" \
  STEPS_BEFORE="$STEPS_BEFORE" STEPS_AFTER="$STEPS_AFTER" \
  WANT_SAMPLES="$PILOT_DUTY_SAMPLES" uv run --no-sync python - <<'EOF'
import json, os, sys
sys.path.insert(0, "scripts")
from pod_acceptance_probe import duty_verdict, parse_dmon_sm_max

rec = duty_verdict(
    sample_max=parse_dmon_sm_max(open(os.environ["DMON"]).read()),
    min_duty=float(os.environ["MIN_DUTY"]),
    want_samples=int(os.environ["WANT_SAMPLES"]),
    train_alive_through_window=os.environ["TRAIN_ALIVE"] == "1",
    steps_before=int(os.environ["STEPS_BEFORE"]),
    steps_after=int(os.environ["STEPS_AFTER"]))
with open(os.environ["DUTY"], "w") as f:
    json.dump(rec, f, indent=2)
print(f"PILOT-DUTY mean_sm={rec['mean_sm_pct']}% over {rec['n_samples']}s "
      f"(threshold {rec['threshold_pct']:.0f}%, optimizer steps "
      f"{rec['steps_before']}->{rec['steps_after']}) -> {rec['verdict']}")
if rec["verdict"] == "GO":
    raise SystemExit(0)
why = rec["no_go"]
if why == "train_exited":
    raise SystemExit(
        "PILOT-DUTY-NO-GO: the psr run stopped running before/inside the "
        "duty window — part of the window sampled idle GPU, so sustained "
        "duty cannot be certified. Raise PILOT_TRAIN_STEPS (the run must "
        "outlast the >=20 s window) and re-run.")
if why == "thin_window":
    raise SystemExit(
        f"PILOT-DUTY-NO-GO: only {rec['n_samples']} usable dmon samples "
        "(need >=20 for a sustained window) — dmon output truncated or "
        "SM% unsupported")
if why == "no_step_advance":
    raise SystemExit(
        "PILOT-DUTY-NO-GO: the optimizer step count did not ADVANCE across "
        f"the window ({rec['steps_before']} -> {rec['steps_after']}) — the "
        "window measured pre-/non-optimizer work (step0 + held-out "
        "forwards run at high SM% too), not the ragged stepping phase this "
        "gate exists to judge. Sustained duty is uncertified.")
raise SystemExit(
    f"PILOT-DUTY-NO-GO: sustained SM duty {rec['mean_sm_pct']}% < "
    f"{rec['threshold_pct']:.0f}% on the stepping window. The known "
    "culprit at 33-45% duty is per-ragged-shape cuDNN SDPA plan "
    "compilation on CPU (the lora-tip 8468792 footgun): verify WS4's "
    "fitmon.pin_attention_backends (enable_cudnn_sdp(False)) + length-"
    "bucketing are active on this path, and read the TB trace's top "
    f"cpu_op for the culprit: {os.environ['TRACE']}. Do NOT fan out "
    "to 16 GPUs on this duty.")
EOF
  uv run --no-sync python scripts/train_two_si_objects.py train \
    --config "$A_PILOT_CFG" --si "$PILOT_SI" \
    --arms bidpo --pairs "$A_POOL" \
    --bidpo-harmful-pairs "$PILOT_DIR/pilot_bidpo_harmful.jsonl" \
    --bidpo-benign-pairs "$PILOT_DIR/pilot_bidpo_benign.jsonl" \
    --out "$PILOT_DIR/training/cells_bidpo" --overwrite
  # NaN-free check on BOTH runs: every required metric finite, both arms
  # report status=trained in their fit reports.
  PILOT_DIR="$PILOT_DIR" SI="$PILOT_SI" uv run --no-sync python - <<'EOF'
import json, math, os
from pathlib import Path
d, si = os.environ["PILOT_DIR"], os.environ["SI"]
for cells, arm in ((f"{d}/training/cells_psr", "psr_kl_all"),
                   (f"{d}/training/cells_bidpo", "bidpo")):
    rep = json.load(open(f"{cells}/{si}__fit_report.json"))
    arms = rep.get("arms", rep)
    status = arms[arm]["status"] if isinstance(arms.get(arm), dict) else None
    assert status == "trained", f"{arm}: status={status!r}, not 'trained'"
    # Metrics live per-cell (WS4 contract: <cell>/metrics.jsonl) and/or in
    # the shared <si>__metrics.jsonl tee — scan whichever exist, require
    # at least one.
    paths = sorted(Path(cells).glob("**/metrics.jsonl"))
    shared = Path(cells) / f"{si}__metrics.jsonl"
    if shared.exists():
        paths.append(shared)
    assert paths, f"{arm}: no metrics jsonl under {cells} — nothing to certify"
    bad = 0
    for mpath in paths:
        for line in open(mpath):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            for k, v in rec.items():
                if isinstance(v, float) and not math.isfinite(v):
                    bad += 1
                    print(f"NON-FINITE metric {k} at step {rec.get('step')} "
                          f"in {mpath}")
    assert bad == 0, f"{arm}: {bad} non-finite metric value(s) — pilot FAILS"
    print(f"{arm}: trained, all logged metrics finite "
          f"({len(paths)} metrics file(s))")
EOF
  touch "$A_TRAIN"
}

run_verdict() {
  PILOT_DIR="$PILOT_DIR" CONFIG="$CONFIG" SLUG="$SLUG" \
  PROBE="$A_PROBE" TABLE="$A_TABLE" DUTY="$A_DUTY" VERDICT="$A_VERDICT" \
  uv run --no-sync python - <<'EOF'
import json, os, socket, time
rec = {
    "pilot": os.environ["SLUG"],
    "config": os.environ["CONFIG"],
    "hostname": socket.gethostname(),
    "finished_unix": time.time(),
    "probe": json.load(open(os.environ["PROBE"])),
    "pilot_batch_table": json.load(open(os.environ["TABLE"])),
    "sustained_duty": json.load(open(os.environ["DUTY"])),
    "training": "trained, NaN-free (see training.ok + training/)",
    "verdict": "GO",
    "note": ("pilot artifacts (incl. the batch table) are TEMPORARY — the "
             "production table is Phase D's measurement"),
}
with open(os.environ["VERDICT"], "w") as f:
    json.dump(rec, f, indent=2)
print("PILOT-GO model=%s -> %s" % (os.environ["SLUG"],
                                   os.environ["VERDICT"]))
EOF
  # Stage the TB trace + curves + verdict for mirroring: launch.sh exports
  # $FIGS_DIR and its watcher rsyncs it to the figure mirror under <run>/.
  STAGE=${FIGS_DIR:-$PILOT_DIR/figs_stage}
  mkdir -p "$STAGE"
  cp -r "$A_TRACE" "$STAGE/" 2>/dev/null || true
  cp "$A_VERDICT" "$A_PROBE" "$A_TABLE" "$A_DUTY" "$STAGE/" 2>/dev/null || true
  find "$PILOT_DIR/training" \( -name '*.png' -o -name '*metrics*.jsonl' \) \
    -exec cp {} "$STAGE/" \; 2>/dev/null || true
  echo "pilot artifacts staged for mirroring at $STAGE"
}

if [ "$DRY_RUN" = "1" ]; then
  echo "PILOT-PLAN model=$SLUG config=$CONFIG dir=$PILOT_DIR"
else
  mkdir -p "$PILOT_DIR"
  chmod 700 "$PILOT_DIR"   # mkdir -p leaves a pre-existing dir's mode alone
fi

step bootstrap \
  "$CONFIG,synthetic:pairs,explicit:bs=$SMOKE_BS" \
  "$A_BOOTSTRAP" \
  run_bootstrap \
  "VLLM_STAGE=1 SMOKE_BS=$SMOKE_BS bash scripts/runpod_bootstrap.sh $CONFIG"

step onboard_full \
  "$CONFIG,$A_BOOTSTRAP" \
  "$A_ONBOARD" \
  run_onboard_full \
  "uv run --no-sync python scripts/onboard_model_smoke.py --model-config $CONFIG --mode full"

step vllm_gate \
  "$CONFIG,$A_BOOTSTRAP,$A_ONBOARD" \
  "$A_VLLM_GATE" \
  run_vllm_gate \
  "UV_PROJECT_ENVIRONMENT=$VLLM_ENV uv run --no-sync python scripts/onboard_model_smoke.py --model-config $CONFIG --mode vllm"

step memprobe \
  "$CONFIG,$A_BOOTSTRAP" \
  "$A_MEMPROBE_LOG,$A_TABLE,$A_PILOT_CFG" \
  run_memprobe \
  "gpu_memprobe_real.py --config <seed cfg @ bs=$SMOKE_BS> --si $PILOT_SI --arms $PILOT_ARMS --sweep --ceiling -> pilot_batch_table_$SLUG.json (TEMPORARY)"

step pilot_pool \
  "$CONFIG,$A_VLLM_GATE" \
  "$A_POOL,$A_POOL_MARK" \
  run_pilot_pool \
  "build_two_si_pool.py generate-pos --config $CONFIG --si $PILOT_SI --max-prompts $PILOT_POOL_PROMPTS --pilot-pairs-out $A_POOL (vLLM venv; WS7)"

step acceptance_probe \
  "$A_PILOT_CFG,$A_TABLE,$A_POOL" \
  "$A_PROBE" \
  run_probe \
  "pod_acceptance_probe.py --config $A_PILOT_CFG --si $PILOT_SI --pairs $A_POOL --arm $PROBE_ARM --steps $PROBE_STEPS --min-steps-per-sec $PROBE_MIN_STEPS_PER_SEC --min-gpu-util $PROBE_MIN_GPU_UTIL"

step training \
  "$A_PILOT_CFG,$A_POOL,$A_PROBE" \
  "$A_TRAIN,$A_TRACE,$A_DUTY" \
  run_training \
  "train_two_si_objects.py train --arms psr_kl_all (~$PILOT_TRAIN_STEPS steps, --profile $A_TRACE; dmon -s u sustained duty >= $PILOT_MIN_DUTY%) + short bidpo run; NaN-free required"

step verdict \
  "$A_PROBE,$A_TRAIN,$A_TABLE,$A_DUTY" \
  "$A_VERDICT" \
  run_verdict \
  "assemble pilot_$SLUG.json (probe + pilot table + sustained duty); stage TB trace + curves for mirroring"

if [ "$DRY_RUN" = "1" ]; then
  echo "PILOT-PLAN-END steps=$STEP_IDX"
else
  echo "PILOT-DONE model=$SLUG verdict=$A_VERDICT"
fi
