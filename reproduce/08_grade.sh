#!/bin/bash
# reproduce/08_grade.sh — stage 08: grading as a CONTINUOUS consumer of the registry's pending-judgment queue.
# Started with the first eval submission and left running: never a trailing stage.
#
#   reproduce/08_grade.sh --model K (--daemon | --once) [--families static,padding,bon,multiturn,petri,agentic,controls]
#                         [--interval 120] [--config F] [--overrides F] [--dry-run] [--help]
#
# Every tick: the pending runs of the model's ids (`pending.pending(conn, model_id=…)`, the Python API — `registry pending-judgment`
# has no --model flag) grouped by eval family × model-key spelling; each group's stores mirrored from their cluster onto this host
# (rsync over CLUSTER_SSH into CLUSTER_GRADE_MIRROR_ROOT) and the family's judging driver run with
# `--worklist registry --run-id … --store-root-map <substrate>=<remote root>=<local mirror>`, so it claims exactly those runs before the
# first judge call and records natively (judgment, per-item rows, run_metric, judged); then scripts/registry/emit_contrasts.py files a
# run_contrast for every steered run whose anchor is complete. MMLU / GSM8K (rule-scored in the job) and BigCodeBench / BFCL (scored on
# Modal) are confirmed, never dispatched. The judges run on this host (SISTEER_GRADING_HOST=local) with the secrets mode of the config.
# The API coordination line is printed at start; the daemon does not wait for the post (the operator posts it).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/lib/common.sh"; source "$HERE/lib/secrets.sh"
usage(){ sed -n '2,15p' "$0"; repro_common_help; }
MODE=""; FAMILIES=""; INTERVAL=120
repro_parse_common_args "$@"; set -- "${REPRO_ARGS[@]+"${REPRO_ARGS[@]}"}"
while [ $# -gt 0 ]; do
  case "$1" in
    --daemon) MODE=daemon; shift;; --once) MODE=once; shift;;
    --families) FAMILIES="$2"; shift 2;; --interval) INTERVAL="$2"; shift 2;;
    *) echo "REFUSED: unknown argument $1" >&2; usage >&2; exit 2;;
  esac
done
[ "$REPRO_HELP" = 1 ] && { usage; exit 0; }
[ -n "$MODE" ] || repro_fatal "--daemon or --once is required"
repro_load_config; repro_require_model
K="$REPRO_MODEL"; STATE="$(repro_state_dir "$K")"
[ "$SISTEER_GRADING_HOST" = local ] || repro_fatal "SISTEER_GRADING_HOST=$SISTEER_GRADING_HOST: this stage runs the judges on the grading host itself — run it there"
PY="$(repro_python)"
SERVED_ID="$(repro_model "$K" serve.checkpoint.model_id)"; FIT_ID="$(repro_model "$K" fit.checkpoint.model_id)"
[ -n "$SERVED_ID" ] || repro_fatal "models.yaml names no serve.checkpoint.model_id for $K"
REG_ROOT="$(eval echo "$SISTEER_REGISTRY_ROOT")"     # the front door's one idiom for a config path (a literal ~ or $HOME expands here)
# the store map + the mirror aliases from the cluster blocks: <substrate>=<that cluster's shared root>=<this host's mirror of it>
MAP_ARGS=(); MIRROR_ARGS=()
for label in $(cluster_labels); do
  sub="$(cluster_var "$label" CLUSTER_SUBSTRATE)"; root="$(cluster_var "$label" CLUSTER_SHARED_ROOT)"; mirror="$(cluster_var "$label" CLUSTER_GRADE_MIRROR_ROOT)"; host="$(cluster_var "$label" CLUSTER_SSH)"
  [ -n "$sub" ] || repro_fatal "cluster $label sets no CLUSTER_SUBSTRATE (cluster_a | cluster_b | slurm)"
  if [ "$sub" = slurm ]; then [ -n "$(cluster_var "$label" CLUSTER_LABEL)" ] || repro_fatal "cluster $label: CLUSTER_SUBSTRATE=slurm requires CLUSTER_LABEL (migration 0018)"; fi
  if [ -n "$mirror" ]; then
    MAP_ARGS+=(--store-root-map "$sub=$root=${mirror/#\~/$HOME}"); [ -n "$host" ] && MIRROR_ARGS+=(--mirror "$sub=$host")
  else
    repro_note "cluster $label sets no CLUSTER_GRADE_MIRROR_ROOT: its stores are read at their recorded paths (fine when this host mounts them)"
  fi
done
# the judges take the Anthropic key through sisteer.vault in the config's secrets mode: under env mode the ONE documented variable must be
# present here (never a file, never a cluster path); under op mode the drivers fetch it themselves (never printed)
if [ "$REPRO_DRY_RUN" != 1 ] && [ "$(secrets_mode)" = env ]; then [ -n "${ANTHROPIC_API_KEY:-}" ] || repro_fatal "SISTEER_SECRETS_MODE=env: set ANTHROPIC_API_KEY (the one key every judge role uses) in this shell"; fi
# --campaign: the site's ONE resolution (common.sh repro_registry_campaign — the campaign 04 fitted under and every eval job filed under
# through slurm_base_env), never the site's key alone, so the daemon reads the queue where the jobs filed
DAEMON_ARGS=(--registry-root "$REG_ROOT" --model-id "$SERVED_ID" --campaign "$(repro_registry_campaign)" --python "$PY" --repo "$REPRO_REPO"
             --state-dir "$STATE/grade" --interval "$INTERVAL" "${MAP_ARGS[@]+"${MAP_ARGS[@]}"}" "${MIRROR_ARGS[@]+"${MIRROR_ARGS[@]}"}")
[ -n "$FIT_ID" ] && DAEMON_ARGS+=(--model-id "$FIT_ID")
[ -n "$FAMILIES" ] && DAEMON_ARGS+=(--families "$FAMILIES")
[ "$MODE" = once ] && DAEMON_ARGS+=(--once)
[ "$REPRO_DRY_RUN" = 1 ] && DAEMON_ARGS+=(--dry-run)
# the family drivers' own arguments: the daemon DERIVES each group's root (--private-root /
# --results-root / --controls-root / --out-root) from the pending runs' mapped store paths and each config from the store's private run
# under --configs-root (the repo's configs/: the paper's committed, a front-door model's from 05_stage_roster.sh --step configs),
# and refuses a group whose config is absent by name; the graders' output root is the one site value the drivers still need
GRADES_ROOT="${SISTEER_GRADES_ROOT:-$STATE/grades}"
DAEMON_ARGS+=(--grades-root "$GRADES_ROOT" --configs-root "$REPRO_REPO/configs")
# THE PETRI DRIVE: the daemon DISCOVERS the waves itself, on its own cadence, from the
# model's state (WAVE.<view>.json + the petri unit's last submitted ledger row) — the documented order starts this daemon BEFORE 06
# writes the wave, so a startup-time spec list would never see it. It gets the state dir, the model, the fit interpreter (F22_PY for the
# drive: a detached checkout at the wave sha has no venv of its own) and each cluster's ssh alias + wave root; when a wave's
# <wave root>/petri/<tag>/serving/endpoint.json lands on the proxy's cluster AND the proxy job it names is RUNNING, the daemon starts
# scripts/sdfbase/full22/petri_drive_full22.sh <model> <view> <cluster> <wave sha> from a persistent checkout at the wave sha
# ($STATE/grade/checkouts/<sha>), once per endpoint (retried under a cap on failure), outcomes recorded under $STATE/grade/petri_drive.<tag>.json;
# the cluster's shared root locates the proxy job's registration ($SHARED/registry/pending/<job>.json), the wave's expected run set
#
# THE PETRI PREFLIGHT: the drive's interpreter contract (grade_daemon.py --petri-python; petri_drive_full22.sh refuses otherwise) is that
# it imports inspect_ai + petri_bloom — pyproject's `petri` dependency group, which 00_setup.sh --what venvs does NOT install (installing
# it is a separately gated step). Probed HERE, before the daemon starts: with the two modules the drive is handed over as above; without
# them the daemon starts with the Petri drive DISABLED (every other family grades exactly as before; the daemon itself would only log
# petri_drive_unprepared once per wave and never start a drive) and ONE warning names the command that installs them — aimed at the
# environment the probed interpreter belongs to: SISTEER_PYTHON may name a venv other than the repository's .venv, where a plain
# `uv sync` would install the group and leave the probed interpreter, and the restart, exactly as they were.
PETRI_INSTALL_CMD="uv sync --frozen --extra cpu --group notebook --group corpus --group petri"
if "$PY" -c 'import inspect_ai, petri_bloom' >/dev/null 2>&1; then
  PETRI_MODE=drive
  # THE BANK ROOT (demo-import, 2026-10-11): the daemon builds the drive's grading host bank view under --petri-bank-root and hands the same root to the
  # drive (F22_GRADING_HOST_BANK_ROOT); it is the site's SISTEER_OBJECTS_BANK — where 05 staged the model's objects — never the daemon's legacy default
  DAEMON_ARGS+=(--petri-state "$STATE" --petri-model "$K" --petri-python "$PY" --petri-bank-root "$SISTEER_OBJECTS_BANK")
  for label in $(cluster_labels); do
    host="$(cluster_var "$label" CLUSTER_SSH)"
    [ -n "$host" ] || { repro_note "cluster $label sets no CLUSTER_SSH: a petri proxy there cannot be watched"; continue; }
    DAEMON_ARGS+=(--petri-cluster "$label=$host=$(repro_expand_shared "$label" "$SISTEER_WAVE_ROOT")=$(cluster_var "$label" CLUSTER_SHARED_ROOT)")
  done
else
  PETRI_MODE=disabled
  PETRI_VENV="$(cd "$(dirname "$(command -v -- "$PY" 2>/dev/null || echo "$PY")")/.." 2>/dev/null && pwd)"
  if [ -n "$PETRI_VENV" ] && [ -f "$PETRI_VENV/pyvenv.cfg" ]; then
    PETRI_FIX="install the petri group into that interpreter's environment -- UV_PROJECT_ENVIRONMENT=$PETRI_VENV $PETRI_INSTALL_CMD --directory $REPRO_REPO --"
  else
    PETRI_FIX="$PY is not a virtual environment's interpreter (no pyvenv.cfg above its directory), so no sync can add the group to it: build one from this repository -- $PETRI_INSTALL_CMD --directory $REPRO_REPO -- and point SISTEER_PYTHON at its bin/python (unset, the stage uses the repository's .venv)"
  fi
  repro_warn "Petri drive DISABLED: $PY does not import inspect_ai + petri_bloom (the drive's interpreter contract). The other families grade as before; to run the Petri audits, $PETRI_FIX, then restart this stage."
fi
repro_say "08 grade: model=$K ids=$SERVED_ID${FIT_ID:+,$FIT_ID} mode=$MODE families=${FAMILIES:-all} interval=${INTERVAL}s petri=$PETRI_MODE registry=$REG_ROOT overrides=$OVERRIDE_ID"
repro_run env SISTEER_RUN_OVERRIDES="$SISTEER_RUN_OVERRIDES" SISTEER_SECRETS_MODE="$SISTEER_SECRETS_MODE" "$PY" "$REPRO_REPO/scripts/registry/grade_daemon.py" "${DAEMON_ARGS[@]}"
