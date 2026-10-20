#!/bin/bash
#SBATCH --job-name=tt-twosi-fitb      # SESSION-UNIQUE tt-twosi-* prefix: override per
#                                     # submission with -J tt-twosi-fitb-<tag>; cancel by JOB
#                                     # ID ONLY (a shared-user name-scoped scancel once caused
#                                     # a mass kill).
#SBATCH --partition=general
#SBATCH --qos=high                    # lane default; draws on the 16-GPU high budget
#SBATCH --gres=gpu:1                  # one model load; the SI-agnostic objects fit
#                                     # in one process (closed-form antidote/alphasteer)
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                    # ~128G per GPU; NEVER --mem=0 (hard rule 2)
#SBATCH --time=0-06:00:00             # closed-form fits + the 16k-prompt capture forwards,
#                                     # plus the alphasteer F.9 screen (all-block capture +
#                                     # |layers| x |p| ridge solves; alphasteer-full22 plan
#                                     # §3 sizing: ~2x the estimate, not 5x, so backfill
#                                     # can place it). Was 0-04:00:00 before the screen.
# The fit reads harmful prompt lists, so a stack trace could print harmful text — logs
# go to the mode-0700 PRIVATE dir, NOT world-readable exp/logs. Slurm opens the log before
# the script runs and does NOT create parents, so PRE-CREATE once before submitting:
#   install -d -m 700 $SHARED/exp/jailbreak/_logs
#SBATCH --output=/workspace-vast/%u/exp/jailbreak/_logs/%x_%A.out
#SBATCH --error=/workspace-vast/%u/exp/jailbreak/_logs/%x_%A.err
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# Class B (SI-agnostic) fit-once driver for the two-SI matrix (split out
# 2026-08-19). Fits antidote + alphasteer (+ caa_static_r) ONCE on
# harmful/benign CONTENT — the AlphaSteer authors' 14,000/2,000 prompt lists
# pinned in the config's class_b block (user instruction 2026-08-20) — and
# writes each object to <defenses_dir>/<arm>/object.* the INSTANT it
# completes (a late arm's failure never discards a finished one).
# NO array: these are SI-AGNOSTIC, one set of objects total (not per SI).
# Resumable: an arm whose meta.json sentinel exists is skipped.
#
# PARAMETERISED (2026-09-08) — everything
# POSITIONAL, nothing through sbatch's env-export flag (hard rule 1):
#   $1  config path   (default configs/two_si.yaml — the shipped Llama config,
#                      whose behaviour is byte-identical to before)
#   $2  arms          (comma list; default "" = the config's class_b_arms)
#   $3  out dir       (ABSOLUTE defenses dir; default = the TT_TWO_SI_DEFENSES_DIR
#                      resolution below, unchanged)
#
#   sbatch --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) \
#          -J tt-twosi-fitb-<tag> scripts/sbatch/fit_class_b.sbatch \
#          configs/alphasteer_f22_gemma4e4b.yaml alphasteer \
#          $SHARED/exp/objects/alphasteer_f22/gemma4e4b/cells
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
# share that file.
#
# The alphasteer F.9 screen (scripts/alphasteer_screen.py) runs BEFORE the
# fit whenever the config carries a class_b.alphasteer block and alphasteer
# is among the arms; otherwise the step is a logged no-op (--if-configured),
# so the shipped Llama config runs exactly the fit it always did.
#
# circuit_breaker left the registry ENTIRELY on 2026-08-20 (user instruction,
# with cast): it is no longer fit here, so this job no longer needs the
# labelled store (TT_TWO_SI_STORE_DIR). The already-fitted circuit_breaker
# artifact stays loadable via methods.load_object.
#
# The class_b prompt lists are consumed VERBATIM with a min_prompts floor;
# the legacy build_content_contrast assembly (alpaca + or-bench-80k, with the
# _refuse_eval_controls guard barring XSTest / or-bench-hard-1k) remains the
# fallback when the paths are unset.

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
: "${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}"
source "$SISTEER_PROJECT_ROOT/repo/scripts/sbatch/stage2_common.sh"

# Positional parse and the bindings file BEFORE s2_init (which consumes
# "<attempt> <max_attempts>" and arms the escalation handler mid-way). The
# file lives in s2_init's STOP_DIR — $SHARED/exp/training/<job name>, the
# per-run dir stop-run.sh and the resubmit key on — derived here with the
# same expression and asserted equal to s2_init's below, so the two
# derivations cannot drift silently.
FITB_BINDINGS="$SHARED/exp/training/${SLURM_JOB_NAME:-stage2}/fit_class_b.bindings"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  if [ ! -f "$FITB_BINDINGS" ]; then
    s2_log "FATAL: resubmission (numeric \$1=$1) but no bindings file at $FITB_BINDINGS"
    exit 2
  fi
  # shellcheck disable=SC1090
  source "$FITB_BINDINGS"
  s2_init "$1" "${2:-5}"
  s2_log "resubmission: bindings reloaded from $FITB_BINDINGS"
else
  FITB_CONFIG="${1:-configs/two_si.yaml}"
  FITB_ARMS="${2:-}"
  FITB_OUT="${3:-}"
  mkdir -p "$(dirname "$FITB_BINDINGS")"
  printf 'FITB_CONFIG=%q\nFITB_ARMS=%q\nFITB_OUT=%q\n' \
    "$FITB_CONFIG" "$FITB_ARMS" "$FITB_OUT" > "$FITB_BINDINGS.tmp"
  mv -f "$FITB_BINDINGS.tmp" "$FITB_BINDINGS"    # atomic: a successor never reads a torn file
  s2_init 1 5
fi
if [ "$FITB_BINDINGS" != "$STOP_DIR/fit_class_b.bindings" ]; then
  s2_log "FATAL: bindings path $FITB_BINDINGS != s2_init's $STOP_DIR/fit_class_b.bindings -- the two derivations drifted"
  exit 2
fi
s2_log "config=$FITB_CONFIG arms=${FITB_ARMS:-<config default>} out=${FITB_OUT:-<TT_TWO_SI_DEFENSES_DIR>}"

# Reruns are cheap (meta.json resume), so a preempted fit just resumes.
export WANDB_DISABLED=true
export WANDB_CONSOLE=off

# (TT_TWO_SI_STORE_DIR is no longer needed here: the store-reading
# circuit_breaker reroute set left with the arm's 2026-08-20 registry
# removal. Exporting it anyway is harmless.)

# Write objects to the ONE commit-independent dir — never the checkout's
# out/, where the next code change (new sha, new checkout) would void resume
# and refit every arm. Matches the overnight orchestrator's --out default;
# body-level export (hard rule 1). Positional $3 overrides it.
export TT_TWO_SI_DEFENSES_DIR=${TT_TWO_SI_DEFENSES_DIR:-$SHARED/exp/jailbreak/two_si_defenses/cells}
FITB_OUT_DIR="${FITB_OUT:-$TT_TWO_SI_DEFENSES_DIR}"
FITB_ARMS_FLAG=()
if [ -n "$FITB_ARMS" ]; then FITB_ARMS_FLAG=(--arms "$FITB_ARMS"); fi

# The alphasteer F.9 screen (plan Phase B) — a no-op when the config has no
# class_b.alphasteer block or alphasteer is not among the arms. Its own
# artifacts resume (capture sentinel + screen_rows.jsonl).
s2_run uv run --no-sync python scripts/alphasteer_screen.py \
  --config "$FITB_CONFIG" --if-configured --resume "${FITB_ARMS_FLAG[@]}"
rc=$?
if [ "$rc" -ne 0 ]; then
  s2_log "alphasteer screen exited rc=$rc (3 = spec A5 STOP: ask the user) -- not fitting"
  exit "$rc"
fi

s2_run uv run --no-sync python scripts/fit_class_b.py \
  --config "$FITB_CONFIG" "${FITB_ARMS_FLAG[@]}" \
  --out "$FITB_OUT_DIR"
exit $?
