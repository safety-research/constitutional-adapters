#!/bin/bash
#SBATCH --job-name=as22-ladder
#SBATCH --partition=general
#SBATCH --qos=high                     # the standing default; on another cluster pass its own QoS (its site block's CLUSTER_QOS).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=0-06:00:00
#SBATCH --output=/workspace/%u/exp/jailbreak/_logs/%x_%j.out
#SBATCH --error=/workspace/%u/exp/jailbreak/_logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# alphasteer-full22 Phase C — the served lambda ladder for ONE target: six SEQUENTIAL standups (one
# per rung), each generating the held-out benign slice through the lens, the held-out harmful
# prefill-only pass that populates the served rho record, and the §4.2 liveness probe. Judging and
# selection do NOT happen here: no API key ever reaches a cluster node, so `generate` writes the
# per-rung evidence and `ladder_select.py` (the grading host) judges it.
#
# QoS: the cluster's non-preemptible QoS, passed at submit (its site block's CLUSTER_QOS). One GPU, so this draws 1 of the
# 16-GPU per-cluster non-preemptible
# budget; three targets run as three separate submissions.
#
# THE PRIVATE LOG DIR. Harmful prompt text can reach a traceback, so both streams go to the
# mode-0700 dir the Phase B fits use, NOT world-readable exp/logs. Slurm opens the log BEFORE this
# body runs, so the directory must already exist:
#   ssh <cluster B's login host, CLUSTER_SSH in its site config block> 'install -d -m 700 $SHARED/exp/jailbreak/_logs'
# (On cluster B /workspace is MooseFS, which silently ignores chmod and reports 0777 on
# everything, so the mode is honoured on cluster A and is a no-op there — one more reason nothing
# secret goes anywhere near that filesystem.)
#
# USAGE (positional only — NEVER --export=, hard rule 1):
#   sbatch -J as22-ladder-<key> scripts/alphasteer_f22/sbatch/ladder.sbatch <W_L> <key> [view] [attempt] [max_attempts] [obj_root]
# from the W_L checkout at $SISTEER_EVAL_REPO_ROOT/<W_L>. The FIRST positional is the ladder
# sha itself: this file lives on W_L, so recording W_L's own sha in it would change that sha (see
# as22_common.sh). Cluster A submissions must also pass --output/--error, since the header's path
# is cluster B's.
#
# THE STANDUP IS scripts/lens_gate_standup.py, NOT the wave's f22_standup (a deviation recorded
# there). Forced, not chosen: the rho read is a worker RPC reached by
# POST <server_base_url>/collective_rpc with the per-launch Bearer token, and mt_serving_up.py never
# persists that token — it is minted in build_server_launch and dies with the standup process. The
# gate standup writes <out>/server_auth.json (0600) and a server_base_url for exactly this reason.
# It preserves the cells (both resolve through defense_cells.enumerate_cells, so the derived
# none__lens0 twin is identical), the manifest pins (it builds through build_server_launch, so
# --limit-mm-per-prompt rides through) and the geometry (it takes L and S explicitly, and the
# driver computes S from geometry.ModelBudget.seqs_at and asserts the no-chunking identity itself).
# It does NOT refuse on the attested reduction mode, so the driver refuses on it instead.
set -uo pipefail

AS22_LADDER_SHA="${1:?usage: ladder.sbatch <W_L sha> <key> [view] [attempt] [max_attempts]}"
KEY_ARG="${2:?key}"
# The view is OPTIONAL and is only ever a CROSS-CHECK: scripts/alphasteer_f22/as22_targets.py is
# the one table that decides it, and bake_rungs.py lays the per-rung defenses dirs out under
# cells_view/<that view>. A view named here that disagrees refuses (as22_resolve_config).
VIEW_ARG="${3:-}"
ATTEMPT="${4:-1}"; MAX_ATTEMPTS="${5:-3}"
# $6 (OPTIONAL) = an object root other than the campaign's: the paper-sized re-bake's PARALLEL root
# `$SHARED/exp/objects/alphasteer_f22_rebake`. Empty = as22_init's default, so an invocation
# without it is today's, byte for byte. It is exported in the BODY below (hard rule 1: never
# --export=) and rides the resubmit line, or a resumed attempt would silently read the campaign root.
OBJ_ROOT_ARG="${6:-}"
case "$AS22_LADDER_SHA" in
  [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) ;;
  *) echo "FATAL: \$1 must be the full W_L sha, got '$AS22_LADDER_SHA'" >&2; exit 2;;
esac
export AS22_LADDER_SHA          # read by as22_assert_serving_identity's clause (i)

# shellcheck source=/dev/null
source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$AS22_LADDER_SHA/scripts/alphasteer_f22/sbatch/as22_common.sh" \
  || { echo "FATAL: no as22_common.sh under $SISTEER_EVAL_REPO_ROOT/$AS22_LADDER_SHA" >&2; exit 1; }

[ -n "$OBJ_ROOT_ARG" ] && export AS22_OBJ_ROOT="$OBJ_ROOT_ARG"   # honoured by as22_init
as22_init "$AS22_LADDER_SHA"
as22_resolve_config "$KEY_ARG" "$VIEW_ARG"   # VIEW comes from as22_targets; $3 is cross-checked
install -d -m 700 "$SHARED/exp/jailbreak/_logs" 2>/dev/null || true

# --- the ladder's OWN env, exported in the BODY (hard rule 1: never --export=) ------------------
# SISTEER_LENS_STACKED_REDUCTIONS=1 already comes from f22_init (the wave's reduction mode, on both
# sides); re-stated here because this job asserts on it and a reader should not have to chase it.
export SISTEER_LENS_STACKED_REDUCTIONS=1
# The rho-emit probe. THIS IS THE ONLY FILE IN THE RUN TREE THAT SETS IT; every production body
# asserts it is unset (as22_assert_serving_identity production).
export SISTEER_LENS_RHO_EMIT=1
# The §4.2 continuous liveness reading. Default-off everywhere else; our cells run under it.
export SISTEER_BITES_POLICY=harmful_continuous
export HF_HUB_OFFLINE=1

# --- the two gates, before a GPU minute is spent ------------------------------------------------
as22_assert_serving_identity ladder     # the mode is a LITERAL (§C.2a)
as22_require_scrub                      # §C.0 item 4 — needs no venv, uses python3

LADDER_ROOT="$AS22_OBJ_ROOT/$KEY/ladder"
PARENT_OBJECT="$AS22_OBJ_ROOT/$KEY/cells/alphasteer/object.pt"
RUN_NAME="${SLURM_JOB_NAME:-as22-ladder-$KEY}"
STOP_DIR="$SHARED/exp/training/$RUN_NAME"      # where the stop-run script writes the STOP sentinel
mkdir -p "$STOP_DIR"

# Refuse EARLY on a target whose Phase B object has not landed, rather than discovering it at bake
# time after the venvs have built (nemotron3nano4b is in exactly this state: its fit was stopped
# deliberately because the tree lacks the fused Mamba kernels, and it will run this script
# unchanged once the object lands — nothing here special-cases a target key).
[ -f "$PARENT_OBJECT" ] \
  || f22_fatal "no fitted AlphaSteer object at $PARENT_OBJECT — Phase B has not produced one for $KEY on this cluster; nothing to build a ladder from"

# --- venvs (node-local, flock'd; the wave's bootstraps) -----------------------------------------
f22_driver_venv     # -> MPY: the driver (this script's python; imports sisteer, no vllm needed)
f22_serving_venv    # -> SPY: the serving venv the standup runs under (vllm + vllm-lens)

# --- bake the six rungs INSIDE the allocation ---------------------------------------------------
# The rung objects are lambda-scaled copies of a 341MB-1.2GB parent — several GB per target. Making
# them on a login node would be heavy CPU/IO outside Slurm (hard rule 4) and shipping them over the
# wire is worse; the bake is a scalar rewrite (base_scale is the HookOp coeff, the matrices are
# untouched), so it costs seconds-to-minutes of CPU in an allocation we already hold.
# Verify-then-build, the way f22_bank_view does: --check is the common path on a resume.
#
# EXIT-CODE CONTRACT, pinned here because the bake and this script disagreed about it
# (bake_rungs.py:103 — EXIT_OK / EXIT_REFUSED / EXIT_NO_OBJECT = 0 / 1 / 3):
#   0   baked, or already verified              -> continue
#   3   the target's parent object is ABSENT. That is a clean per-target SKIP for a MULTI-target
#       orchestrator, and FATAL here: this job serves exactly ONE target, so "nothing to bake"
#       means "nothing to serve". The PARENT_OBJECT guard above normally fires first; this is the
#       belt for the object vanishing between the two checks.
#   any other nonzero  -> a real bake/verification refusal, fatal.
# A single-target job must never read 3 as "carry on" — that would stand six rungs up against
# whatever objects happened to be on disk.
bake_rc=0
"$MPY" scripts/alphasteer_f22/bake_rungs.py --check --target "$KEY" --root "$AS22_OBJ_ROOT" || bake_rc=$?
if [ "$bake_rc" -eq 3 ]; then
  f22_fatal "bake_rungs.py reports NO FITTED OBJECT for $KEY (exit 3) — the target is not ready; a skip for a multi-target driver, fatal for this single-target job"
elif [ "$bake_rc" -ne 0 ]; then
  f22_log "rung objects do not verify (rc=$bake_rc) — baking"
  bake_rc=0
  "$MPY" scripts/alphasteer_f22/bake_rungs.py --target "$KEY" --root "$AS22_OBJ_ROOT" || bake_rc=$?
  [ "$bake_rc" -eq 3 ] && f22_fatal "bake_rungs.py reports NO FITTED OBJECT for $KEY (exit 3) — refusing"
  [ "$bake_rc" -ne 0 ] && f22_fatal "bake_rungs.py refused for $KEY (rc=$bake_rc)"
  "$MPY" scripts/alphasteer_f22/bake_rungs.py --check --target "$KEY" --root "$AS22_OBJ_ROOT" \
    || f22_fatal "rung objects still do not verify after baking $KEY"
fi

# --- traps: drain, stop OUR standups, mirror, and resubmit once if this was a signal ------------
DRIVER_PID=""; RC=0; HANDLED=0
SCRIPT_PATH="$(scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"

as22_resubmit(){
  # the high QoS is not preempted, but nodes fail and walltimes expire, and a ladder resumes at the
  # first unfinished rung (STANDUP_DONE sentinels) for the cost of one model load.
  if [ -e "$STOP_DIR/STOP" ] || [ -e "$LADDER_ROOT/STOP" ]; then
    f22_log "STOP sentinel present — deliberate stop, not resubmitting."; return; fi
  if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
    f22_log "attempt cap $MAX_ATTEMPTS reached — not resubmitting."; return; fi
  if [ -z "$SCRIPT_PATH" ] || [ ! -f "$SCRIPT_PATH" ]; then
    f22_log "cannot resolve the submitted script path ('$SCRIPT_PATH') — not resubmitting."; return; fi
  f22_log "resubmitting as attempt $((ATTEMPT + 1))/$MAX_ATTEMPTS (resumes at the first rung without STANDUP_DONE)"
  sbatch --qos="${SLURM_JOB_QOS:?the job runs under Slurm, which sets SLURM_JOB_QOS}" --job-name="$RUN_NAME" \
         --output="$SHARED/exp/jailbreak/_logs/%x_%j.out" \
         --error="$SHARED/exp/jailbreak/_logs/%x_%j.err" \
         "$SCRIPT_PATH" "$AS22_LADDER_SHA" "$KEY" "$VIEW" "$((ATTEMPT + 1))" "$MAX_ATTEMPTS" "$OBJ_ROOT_ARG" \
    || f22_log "RESUBMIT FAILED — this ladder will not continue automatically"
}

cleanup(){
  [ "$HANDLED" -eq 1 ] && return; HANDLED=1
  # Our OWN per-standup stop files, never a wave-wide sentinel. This is also what releases a
  # standup the driver could not tear down itself (a SIGKILLed driver leaves its child holding the
  # GPU): the standup polls its stop file and exits within seconds. Deliberately NOT a
  # `kill -TERM -$$` process-group kill — that signals this shell too, so the figure mirror and the
  # RESULT line below would never run, and Slurm's cgroup cleans the step up at job end anyway.
  as22_stop_standups "$LADDER_ROOT"
  if [ -n "${DRIVER_PID:-}" ]; then
    kill -TERM "$DRIVER_PID" 2>/dev/null
    for _ in $(seq 1 24); do kill -0 "$DRIVER_PID" 2>/dev/null || break; sleep 5; done
    kill -9 "$DRIVER_PID" 2>/dev/null
  fi
  f22_result "as22 ladder $KEY/$VIEW rc=${RC} attempt=$ATTEMPT"
  f22_mirror
}
on_term(){ f22_log "SIGTERM (preemption, cancel or walltime) — draining"; RC=143; cleanup; as22_resubmit; exit 143; }
trap on_term SIGTERM SIGINT SIGQUIT
trap cleanup EXIT

# --- drive the six rungs sequentially -----------------------------------------------------------
# The driver owns the standup lifecycle, the per-rung resume skip and the rho reads; it is
# backgrounded and waited on so the trap fires inside the 120s grace window rather than after.
f22_log "ladder: $KEY/$VIEW at W_L $AS22_LADDER_SHA, six rungs into $LADDER_ROOT"
"$MPY" scripts/alphasteer_f22/lambda_ladder.py generate \
    --repo-dir "$REPO_DIR" \
    --ladder-sha "$AS22_LADDER_SHA" \
    --target "$KEY" --view "$VIEW" \
    --root "$AS22_OBJ_ROOT" \
    --config "$CFG" --manifest "$MANIFEST" \
    --serving-python "$SPY" \
    --stop-file "$STOP_DIR/STOP" &
DRIVER_PID=$!
wait "$DRIVER_PID"; RC=$?
DRIVER_PID=""
f22_log "ladder driver exited rc=$RC"

# A nonzero rc that is not a signal death is a real refusal — an uncertified rung, a failed gate, a
# missing input. Do NOT resubmit: the same failure would repeat, and an UNCERTIFIED rung is a
# finding to read, not a transient to retry.
exit "$RC"
