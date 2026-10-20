#!/bin/bash
# reproduce/lib/attempt_wait.sh — the IN-BODY START-GRANT HOOK of the twin-submit protocol. sisteer-attempt-wait-hook v1
#
# Sourced by every twinnable sbatch body as the FIRST executable line after its bootstrap `source` — BEFORE f22_init,
# before f22_resolve_config's self-registration and before slurm_fit_cell.py — through `f22_attempt_wait` (common.sh)
# or directly (sdfbase_fit_cell.sbatch). ACTIVE only when SISTEER_ATTEMPT_ID is set in the job's command-line env
# prefix: unset, it returns at once and every existing job's behaviour is byte-for-byte unchanged.
#
# Why: two twins of one unit are two DIFFERENT run identities (jobplan keys `job_name@job_id`), and a job registers the
# moment its config resolves — so the loser must be stopped BEFORE it registers. The grading host's coordinator
# (submit_twin.py, the ONLY grant writer) commits the winner to its ledger, then writes GRANT to the winner's attempt
# dir and DENY to every sibling (tmp + mv, atomic). This hook blocks until one appears:
#   GRANT  -> proceed (SISTEER_ATTEMPT_GRANTED=1; the run's extra.attempt_id / extra.logical_work_id name the election)
#   DENY   -> exit 0 having loaded nothing and emitted nothing (the coordinator scancels this job id only)
#   timeout (SISTEER_GRANT_TIMEOUT_MIN, default 10) -> exit 0 as a DENY-equivalent, recorded by the coordinator
# It also drops a WAITING marker so the coordinator can tell "running and waiting" from "still queued".
sisteer_attempt_wait(){
  [ -n "${SISTEER_ATTEMPT_ID:-}" ] || return 0
  local root="${SISTEER_ATTEMPTS_ROOT:-}"
  [ -n "$root" ] || { echo "[attempt] FATAL: SISTEER_ATTEMPT_ID=$SISTEER_ATTEMPT_ID is set but SISTEER_ATTEMPTS_ROOT is not" >&2; exit 1; }
  local d="$root/$SISTEER_ATTEMPT_ID" timeout=$(( ${SISTEER_GRANT_TIMEOUT_MIN:-10} * 60 )) poll="${SISTEER_GRANT_POLL_S:-5}" t0=$SECONDS
  mkdir -p "$d" 2>/dev/null || { echo "[attempt] FATAL: cannot create $d" >&2; exit 1; }
  printf '{"attempt_id":"%s","job_id":"%s","job_name":"%s","node":"%s","utc":"%s"}\n' \
    "$SISTEER_ATTEMPT_ID" "${SLURM_JOB_ID:-manual}" "${SLURM_JOB_NAME:-}" "$(hostname)" "$(date -u +%FT%TZ)" > "$d/WAITING.${SLURM_JOB_ID:-manual}" 2>/dev/null || true
  echo "[attempt] $SISTEER_ATTEMPT_ID waiting for a start grant under $d (timeout ${SISTEER_GRANT_TIMEOUT_MIN:-10} min)"
  while :; do
    if [ -e "$d/GRANT" ]; then echo "[attempt] GRANT $SISTEER_ATTEMPT_ID (job ${SLURM_JOB_ID:-manual}) — proceeding"; export SISTEER_ATTEMPT_GRANTED=1; return 0; fi
    if [ -e "$d/DENY" ]; then echo "[attempt] DENY $SISTEER_ATTEMPT_ID (job ${SLURM_JOB_ID:-manual}) — leaving; nothing loaded, nothing registered"; exit 0; fi
    if [ $(( SECONDS - t0 )) -ge "$timeout" ]; then echo "[attempt] no grant for $SISTEER_ATTEMPT_ID within ${SISTEER_GRANT_TIMEOUT_MIN:-10} min — leaving (DENY-equivalent; the coordinator re-queues the unit)"; exit 0; fi
    sleep "$poll"
  done
}
sisteer_attempt_wait
