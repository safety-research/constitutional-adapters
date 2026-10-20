#!/bin/bash
#SBATCH --job-name=as22-fit           # the alphasteer-full22 (as22-*) prefix; ALWAYS override
#                                     # per submission with -J as22-fit-<target>; cancel by
#                                     # JOB ID or `scancel -u $USER -n as22-fit-<target>`
#                                     # (never a user-wide pattern: hd-*/f22-* are a peer's)
#SBATCH --partition=general
#SBATCH --qos=high                    # the standing default; draws 1 of the 16-GPU high budget
#SBATCH --gres=gpu:1                  # one model load: all-block capture + screen + the fit
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                    # ~128G per GPU; NEVER --mem=0 (hard rule 2)
#SBATCH --time=0-08:00:00             # capture + |layers| x |p| screen + fit ~3-5 h, padded <2x
# JB4: the job reads harmful prompt lists, so a stack trace could print harmful
# text — logs go to the mode-0700 PRIVATE dir, NOT world-readable exp/logs.
# Slurm opens the log before the script runs and does NOT create parents:
#   install -d -m 700 $SHARED/exp/jailbreak/_logs   (exists)
#SBATCH --output=/workspace-vast/%u/exp/jailbreak/_logs/%x_%j.out
#SBATCH --error=/workspace-vast/%u/exp/jailbreak/_logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# alphasteer-full22 PHASE B — one multi-layer AlphaSteer fit per target.
# GATED ON PHASE D twice over: submit with
# --dependency=afterok:<as22-llama-gate job id>, AND the body below refuses
# unless $SHARED/exp/objects/alphasteer_f22/llama_gate.json is a COMPLETE,
# sha-verified verdict with passed: true. Then:
#   1. scripts/alphasteer_screen.py — split, all-block capture, the F.9-AUROC
#      layer/p screen (eligibility AUROC ONLY; leakage reported per layer as a
#      diagnostic), the chosen set -> <screen_path>. NO --if-configured:
#      a config without the class_b.alphasteer block REFUSES here instead of
#      logging a no-op.
#   2. scripts/fit_class_b.py --arms alphasteer --require-alphasteer-rule —
#      the multi-layer fit over the screen's chosen (layers, ratios); the flag
#      makes a missing rule block a refusal (exit 2), never the silent legacy
#      single-layer fit at the inherited L15 pin.
#
# Exit codes: 0 fit written; 3 the Phase D gate has not passed (or the screen's
# spec-A5 STOP); 2 a binding refusal / missing rule block; else failure.
#
# Positional (everything positional, nothing through sbatch --export, hard rule 1):
#   $1  config path   (configs/alphasteer_f22_<target>.yaml — REQUIRED on the
#                      first attempt; a NUMERIC $1 marks a resubmission, below.
#                      Its optional `env_extras:` line names the venv extras
#                      s2_init syncs -- `mamba` on the Mamba-hybrid target)
#   $2  arms          (default alphasteer)
#   $3  out dir       (ABSOLUTE; default $SHARED/exp/objects/alphasteer_f22/<target>/cells,
#                      <target> read off the config file name)
# Submit (from the grading host, over ssh to the login node), one per target:
#   sbatch --comment=sha:$(git -C ${SISTEER_PROJECT_ROOT}-as22/repo rev-parse HEAD) \
#          -J as22-fit-gemma4e4b --dependency=afterok:<gate id> \
#          ${SISTEER_PROJECT_ROOT}-as22/repo/scripts/sbatch/alphasteer_fit_f22.sbatch \
#          configs/alphasteer_f22_gemma4e4b.yaml
#
# stage2_common's SELF-RESUBMIT (s2_escalate, on preemption/TERM) re-invokes
# this script with ONLY "<attempt> <max_attempts>", both numeric — so a
# NUMERIC $1 marks a resubmission, and the bindings are reloaded from the
# per-run file the first attempt wrote under $STOP_DIR (keyed on the job
# name, which the resubmit preserves). The bindings are written ATOMICALLY
# BEFORE s2_init is entered: s2_init installs the TERM/escalation handler
# before the env lock, `uv sync` and the cu128 assert, so a TERM in that
# window resubmits — and the successor must already find its bindings
# (writing them after s2_init returned lost them exactly there).
# Give every submission its own -J name: two live jobs sharing a name would
# share that file. This block is fit_class_b.sbatch's, verbatim (test-pinned).
#
# This lane owns its OWN checkout root ($S2_ROOT below) so the peers' shared
# production checkout (the project root's repo/) and its node-local venv are
# never repointed or re-synced under a running job (capeval-controls §6 seam).

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"

S2_ROOT="${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-as22"          # repo at $S2_ROOT/repo, gate file $S2_ROOT/EXPECTED_SHA,
                                            # venv /home/$USER/venvs/sisteer-as22, wandb $S2_ROOT/wandb
source "$S2_ROOT/repo/scripts/sbatch/stage2_common.sh"

# Positional parse and the bindings file BEFORE s2_init (which consumes
# "<attempt> <max_attempts>" and arms the escalation handler mid-way). The
# file lives in s2_init's STOP_DIR — $SHARED/exp/training/<job name>, the
# per-run dir stop-run.sh and the resubmit key on — derived here with the
# same expression and asserted equal to s2_init's below, so the two
# derivations cannot drift silently.
FITB_BINDINGS="$SHARED/exp/training/${SLURM_JOB_NAME:-stage2}/fit_class_b.bindings"
# The venv extras a target's config declares (`env_extras: mamba` -- the fused
# Mamba kernels, REQUIRED for a Mamba-hybrid target; user directive
# 2026-09-09) become S2_UV_EXTRAS for s2_init's `uv sync --frozen`. Read here,
# in the job body, BEFORE s2_init (never via sbatch --export, hard rule 1),
# from the config both the first attempt and a resubmission hold: the other
# targets declare nothing and sync exactly as before, so their venvs -- hence
# their captures -- are untouched by another target's kernels. A plain
# `key: value` grep, deliberately: no venv exists yet to parse YAML with.
# The kernels are PyPI sdists uv compiles against the lockfile's torch (pyproject's
# mamba extra, 2026-10-12): when `mamba` is among the extras s2_init selects a
# VALIDATED CUDA 12 toolkit as CUDA_HOME (scripts/sbatch/cuda_toolkit.sh; FATAL
# before the sync when the node has none -- cluster B ships CUDA 13 only, which
# torch refuses against the cu128 torch) and bounds MAX_JOBS, and the sync installs
# the build cached in the SHARED UV_CACHE_DIR -- warm it once per cluster with
# scripts/sbatch/mamba_build.sbatch (via submit_mamba_build.sh; it also installs the
# toolkit under $SHARED/cuda-12.8 on a cluster without one) first, or this job's
# first sync compiles for ~20-60 min out of its 8 h budget.
as22_env_extras() {
  sed -n 's/^env_extras:[[:space:]]*\([^#]*\).*$/\1/p' "$1" | tr -d '[],"' | xargs
}
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  if [ ! -f "$FITB_BINDINGS" ]; then
    s2_log "FATAL: resubmission (numeric \$1=$1) but no bindings file at $FITB_BINDINGS"
    exit 2
  fi
  # shellcheck disable=SC1090
  source "$FITB_BINDINGS"
  S2_UV_EXTRAS="$(as22_env_extras "$S2_ROOT/repo/$FITB_CONFIG")"
  s2_init "$1" "${2:-5}"
  s2_log "resubmission: bindings reloaded from $FITB_BINDINGS"
else
  if [ -z "${1:-}" ]; then
    s2_log "FATAL: Phase B needs the target config as \$1 (configs/alphasteer_f22_<target>.yaml); there is no default"
    exit 2
  fi
  FITB_CONFIG="$1"
  FITB_ARMS="${2:-alphasteer}"
  FITB_OUT="${3:-}"
  mkdir -p "$(dirname "$FITB_BINDINGS")"
  printf 'FITB_CONFIG=%q\nFITB_ARMS=%q\nFITB_OUT=%q\n' \
    "$FITB_CONFIG" "$FITB_ARMS" "$FITB_OUT" > "$FITB_BINDINGS.tmp"
  mv -f "$FITB_BINDINGS.tmp" "$FITB_BINDINGS"    # atomic: a successor never reads a torn file
  S2_UV_EXTRAS="$(as22_env_extras "$S2_ROOT/repo/$FITB_CONFIG")"
  s2_init 1 5
fi
[ -n "${S2_UV_EXTRAS:-}" ] && s2_log "venv extras from $FITB_CONFIG: $S2_UV_EXTRAS"
if [ "$FITB_BINDINGS" != "$STOP_DIR/fit_class_b.bindings" ]; then
  s2_log "FATAL: bindings path $FITB_BINDINGS != s2_init's $STOP_DIR/fit_class_b.bindings -- the two derivations drifted"
  exit 2
fi

# Figures reach the figure mirror under the CAMPAIGN name as well as the job
# name: s2_finish mirrors $FIGS_DIR under <job name>; this second push lands
# the same files under alphasteer-full22/. Wrapping s2_finish keeps
# its `local rc=$?` reading the trap's entry status (it is the first command).
as22_finish() {
  s2_finish
  "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" alphasteer-full22 2>/dev/null || true
}
trap as22_finish EXIT

OBJ_ROOT="$SHARED/exp/objects/alphasteer_f22"
GATE_JSON="$OBJ_ROOT/llama_gate.json"
TARGET="$(basename "$FITB_CONFIG" .yaml)"; TARGET="${TARGET#alphasteer_f22_}"
FITB_OUT_DIR="${FITB_OUT:-$OBJ_ROOT/$TARGET/cells}"
s2_log "config=$FITB_CONFIG arms=$FITB_ARMS target=$TARGET out=$FITB_OUT_DIR"

# THE PHASE D GUARD (in-body; --dependency=afterok is the scheduler half). The
# verdict must be a COMPLETE, sha-verified llama_gate.json with passed: true —
# read through the library's own fingerprint reader, so a torn, tampered or
# stale-by-inputs file refuses exactly as it would refuse the gate itself.
# Fail-fast, before any GPU work: an 11-second death here beats a 3-hour fit
# of an object no gate authorised.
if [ ! -s "$GATE_JSON" ] || [ ! -s "$GATE_JSON.COMPLETE" ]; then
  s2_log "FATAL: Phase D verdict missing or incomplete at $GATE_JSON — the STOP GATE has not passed; no fit"
  exit 3
fi
gate_state="$(uv run --no-sync python - "$GATE_JSON" <<'PY'
import sys
from sisteer.extraction.alphasteer_screen import read_fingerprinted_json
d = read_fingerprinted_json(sys.argv[1])
ok = d.get("passed") is True and d.get("gate_reading") == "reading_released"
print(f"passed={'true' if ok else 'false'} n_recovered={d.get('n_recovered')}/{len(d.get('recovery', {}).get('paper_layers', []))} "
      f"threshold={d.get('threshold')} code_sha={str(d.get('code_sha'))[:12]} refit_status={d.get('refit_status')}")
PY
)" || { s2_log "FATAL: could not read the Phase D verdict at $GATE_JSON (fingerprint or schema refusal): $gate_state"; exit 3; }
s2_log "Phase D verdict: $gate_state"
case "$gate_state" in
  passed=true*) ;;
  *) s2_log "FATAL: Phase D STOP GATE did not pass — no new-model object may be fit (plan §Phase D)"; exit 3 ;;
esac

# Fail-fast guards on every external input (cluster-submit skill): -s, not -f.
LISTS_DIR="$SHARED/exp/jailbreak/contrast"
fail=0
for f in "$FITB_CONFIG" "$LISTS_DIR/benign_alphasteer.json" "$LISTS_DIR/harmful_alphasteer.json" \
         "$LISTS_DIR/benign_alphasteer.slices.json" "$LISTS_DIR/harmful_alphasteer.slices.json" \
         "$OBJ_ROOT/leakage_report.json"; do
  if [ ! -s "$f" ]; then s2_log "FATAL: required input missing or empty: $f"; fail=1; fi
done
[ "$fail" -eq 0 ] || exit 1
mkdir -p "$OBJ_ROOT/$TARGET" "$FITB_OUT_DIR"

# Never set CUDA_VISIBLE_DEVICES here (hard rule 3): Slurm exports it.
# 1. the F.9 screen. NO --if-configured: a config without the rule block
#    refuses here. --arms alphasteer so the step cannot be skipped by an arms
#    list either. W&B runs offline on compute nodes (s2_init's WANDB_DIR);
#    sync from a login node afterwards. The project: the site's
#    SISTEER_WANDB_PROJECT_ALPHASTEER (the cluster's hf_site.env, exported by
#    hf_env.sh in s2_init), else the job's WANDB_PROJECT / SISTEER_WANDB_PROJECT;
#    none of them set = W&B off (the script's own default).
AS_WANDB_PROJECT="${SISTEER_WANDB_PROJECT_ALPHASTEER:-${WANDB_PROJECT:-${SISTEER_WANDB_PROJECT:-}}}"
s2_log "W&B project: ${AS_WANDB_PROJECT:-<none: W&B off>}"
s2_run uv run --no-sync python scripts/alphasteer_screen.py \
  --config "$FITB_CONFIG" --arms alphasteer --resume --wandb-project "$AS_WANDB_PROJECT"
rc=$?
SCREEN_DIR="$OBJ_ROOT/$TARGET"
if [ -s "$SCREEN_DIR/figs/auroc_vs_depth.png" ]; then
  cp -f "$SCREEN_DIR/figs/auroc_vs_depth.png" "$FIGS_DIR/final_auroc_vs_depth_by_p_${TARGET}.png"
fi
if [ "$rc" -ne 0 ]; then
  s2_log "alphasteer screen exited rc=$rc (3 = spec A5 STOP: ask the user; 2 = binding refusal / no rule block) -- not fitting"
  exit "$rc"
fi

# 2. the fit, held to the screen and to the rule block's presence.
s2_run uv run --no-sync python scripts/fit_class_b.py \
  --config "$FITB_CONFIG" --arms "$FITB_ARMS" --out "$FITB_OUT_DIR" --require-alphasteer-rule
rc=$?
case "$rc" in
  0) s2_log "PHASE B ($TARGET): multi-layer alphasteer object written under $FITB_OUT_DIR" ;;
  2) s2_log "PHASE B ($TARGET): REFUSED (rc=2) — a binding or rule-block refusal; see the summary above" ;;
  *) s2_log "PHASE B ($TARGET): FAILED rc=$rc (see the .err sibling for the traceback)" ;;
esac
exit "$rc"
