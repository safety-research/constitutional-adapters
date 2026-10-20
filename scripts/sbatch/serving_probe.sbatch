#!/bin/bash
# scripts/sbatch/serving_probe.sbatch — the serving investigation
# (FD 19 / FD 24): ONE Slurm job on EIGHT H200s that stands up every planned
# native multi-LoRA manifest for a model (scripts/sdfbase/serving_probe.py
# `run`, resumable per launch) — the launches are independent one-GPU standups
# of one bf16 model, so the driver runs them `--parallel` at a time, one per
# allocated GPU, each engine pinned to a STRICT SUBSET of Slurm's
# CUDA_VISIBLE_DEVICES in its own environment (attempt 1 on one
# GPU passed ≈1 launch per 5.3 min, ≈11 h for the plan's 135 launches in a 4 h
# job; eight in flight is ≈1.5 h) — then writes the verdict
# (`probe_verdict.json` + `allowed_pairs.json`, the LoRA search's `P`) and the
# FD 24 memo inputs. It is also the certification probe.
#
# Positional args only (hard rule 1: nothing rides --export):
#   serving_probe.sbatch <model_key> <served_hf_id> <revision> [sets] [ranks] [mode] [attempt] [max_attempts] [parallel] [reopen_token] [phase] [reverdict_arg]
#     phase         full (default) | reference (re-run the HF+PEFT reference phase alone over a complete record — 1 GPU)
#                   | parity (the PARITY PASS on a SEPARATE record — `plan --parity-pass --parity-scales default`, i.e. the
#                     natural launches at the ladder window with per-set benign-scale adapters, `run --no-throughput`; the main
#                     record's `reverdict --parity-record <this record>` then reads assertion 4 from it). Positional 10 stays
#                     empty for a fresh parity record ("" on the command line).
#                   | reverdict (routed by 03_hpo.sh --phase probe --reopen): the CPU-only
#                     certifying rewrite of a COMPLETE record under the serving venv — no plan, no launches, no engine, no successor,
#                     no attempt record. Positional 12: `resume-interrupted` adopts the record's one IN-PROGRESS receipt (a reverdict
#                     that consumed it and died; positional 10 is `-`, never a fresh revocation) | `parity-record=<abs dir>` re-reads
#                     assertion 4 from the parity-pass record under the receipt the front door minted (its nonce as positional 10).
#     reopen_token  the revocation receipt's 32-hex nonce, or `-` for none when a later positional must follow
#     model_key     qwen35_4b_demo (a key of lora_modules.ARCH_DIMS_BY_MODEL_KEY, serving_probe.MODEL_DIMS_KEYS)
#     served_hf_id  the SERVED checkpoint, e.g. Qwen/Qwen3.5-4B
#     revision      its 40-hex snapshot sha (a ref name is resolved from the
#                   shared HF cache offline; the job never reaches the network)
#     sets          default wo,qkvo,attn_mlp,mlp,down
#     ranks         default 4,16,64
#     mode          tp1 (default) | tp2 — the documented SECOND submission on
#                   2 GPUs composing --tensor-parallel-size 2 --fully-sharded-loras
#     attempt       1 (default); the job resubmits ITSELF as attempt+1 when the
#                   driver stops on the time budget with launches outstanding
#                   (exit 4) or is SIGTERMed (preemption / drain), continuing the
#                   SAME record with `run --resume --replan` (D108: the successor
#                   re-derives S from the record's measured batched-token budget
#                   and re-runs the start-time prefill OOMs above it) — the fit path's pattern
#                   (positional counter, --no-requeue).
#                   A STOP sentinel disables the resubmission: EITHER $OUT/STOP
#                   (the record's own) OR the stop helper's contract
#                   $SHARED/exp/training/<job-name>/STOP (what reproduce/lib/stop_run.sh
#                   and the cluster's stop-run.sh helper write before scancel) —
#                   both are honoured, so stopping through the documented helper
#                   never queues a successor
#     max_attempts  4 (default)
#     parallel      auto (default) | N — concurrent standups. `auto` = the job's GPU count ÷ the GPUs one
#                   standup needs (tp), so the 8-GPU header runs 8 tp1 launches (or 4 tp2 launches) at a time and
#                   the documented 2-GPU tp2 submission runs 1; an explicit N the allocation cannot hold is REFUSED
#                   by the driver (never widened). The driver pins each engine's CUDA_VISIBLE_DEVICES to a disjoint
#                   slice of Slurm's value in the CHILD's environment only — this script never sets the variable.
#     reopen_token  empty (default) | the NONCE (32 hex) of the revocation receipt the front door wrote into the
#                   record. REQUIRED by the driver to reopen a record whose checkpoint is COMPLETE;
#                   reproduce/03_hpo.sh --phase probe --reopen mints it ONLY after archiving the
#                   site copy and every cluster's staged geometry overlay with a REVOKED marker and writing the
#                   receipt into the record; the driver consumes the receipt. A successor this job resubmits never
#                   carries one: it only ever continues an INCOMPLETE record.
#
# Job-side knobs ride the submitting shell's environment (Slurm's implicit export; never --export=):
#   SISTEER_SHARED_ROOT   the shared root (a configured site: reproduce/lib/slurm.sh's job env); else the two known
#                         per-user roots are probed (hf_env.sh's rule)
#   SISTEER_HF_HOME       the HF cache (else $SHARED/scripts/hf_env.sh derives it; neither present = a refusal,
#                         never one cluster's cache path hard-coded)
#   SISTEER_CLUSTER_QOS   the QoS a resubmission goes out at (default: this job's own QoS)
#   SERVING_PROBE_OUT     the record dir (below); a resubmission carries this job's OUT so the successor resumes it
#   SISTEER_GEOMETRY_DIR  the SITE geometry overlay dir the verdict's geometry fold writes <model_key>.json into
#                         (default $SHARED/exp/reproduce/geometry — the dir geometry.site_geometry_dir() derives
#                         from SISTEER_SHARED_ROOT in every front-door job, so the padded launchers on this cluster
#                         read the probe's row)
#   SERVING_PROBE_PARITY_SCALES  phase=parity only: the pass's per-set synthetic scales — `default` (the driver's table:
#                         wo 1.0, qkvo 0.3, the MLP-bearing sets 0.05) or `set=scale,…`; the front door's
#                         `03_hpo.sh --phase probe --parity-pass --parity-scales …` sets it when a model's surface needs
#                         another scale (the pass's own parity_weak / parity_chaotic gates say so). Any other value refuses.
#
# The ATTEMPT RECORD: every attempt writes $OUT/probe_attempt.json at START — the
# counter, the cap, the width (Slurm's device count), the concurrency, positionals 4-6, the job, the sha, the QoS /
# walltime / cpus / mem. It is the record's own statement of its LAST attempt ($OUT/RESULT is the driver's verdict line,
# written once, and is NOT one): `reproduce/03_hpo.sh --phase probe` resumes an INCOMPLETE record with no job from it —
# attempt+1, the same cap, the same positionals, the same width by default — through lib/slurm.sh (ledger line + watcher),
# never from a re-typed line. scripts/sdfbase/hpo_probe_resume.py is the reader.
#
# The serving venv REQUIRES peft (peft==0.20.0 in configs/capeval/vllm-requirements.txt): the HF+PEFT parity check
# (assertion 4) is a certification check, and one that cannot fire certifies nothing — the guard below FAILS the job
# when peft does not import, and the driver's verdict refuses (exit 6, RESULT state=UNCERTIFIED) without reference
# evidence.
#
# Submit from the cluster-mirrored checkout (mirror-repos.sh first); the
# --comment pins the commit the submitter validated and the job REFUSES if
# the checkout's HEAD differs (sdfbase_fit_cell.sbatch's block, byte for byte):
#   REPO=$CLUSTER_REPO_MIRROR
#   sbatch -J serving-probe-qwen35_4b_demo --comment=sha:$(git -C "$REPO" rev-parse HEAD) \
#     "$REPO/scripts/sbatch/serving_probe.sbatch" qwen35_4b_demo Qwen/Qwen3.5-4B <40hex>
# (the header's 8 GPUs, `parallel` auto = 8 concurrent standups; half the 16-GPU `high` budget). A narrower
# allocation is the same script with the GPU count on the command line — the driver derives the concurrency:
#   sbatch ... --gres=gpu:4 --cpus-per-task=64 --mem=512G "$REPO/scripts/sbatch/serving_probe.sbatch" qwen35_4b_demo ...
# The front door (reproduce/03_hpo.sh --phase probe [--probe-width N|auto]) composes both forms, and RESUMES an
# INCOMPLETE record itself (attempt+1 from probe_attempt.json, see above) — the documented way to continue a stalled record.
# The --tp2 variant (2 GPUs, ~1 h; one MLP set at one rank is enough; `parallel` auto = 1 here):
#   sbatch -J serving-probe-qwen35_4b_demo-tp2 --gres=gpu:2 --cpus-per-task=32 --mem=256G \
#     --time=01:30:00 --comment=sha:$(git -C "$REPO" rev-parse HEAD) \
#     "$REPO/scripts/sbatch/serving_probe.sbatch" qwen35_4b_demo Qwen/Qwen3.5-4B <40hex> mlp 16 tp2
#
# Record dir: $SHARED/exp/multilora_cert/<model_key>_probe_<jobid>/ (the
# certification root family). RESULT + figures mirror on EXIT, whatever the
# exit; the trap disarms SIGTERM first (the re-entry measured 2026-09-11).
#
# The serving venv is the requirements-hash vllm venv (torch cu129) installed
# OUTSIDE the project lockfile — the cu128 entry-point guard does not apply
# (see tests/test_torch_flavors.py OUT_OF_LOCKFILE_TAG):
# audit-exempt: out-of-lockfile-venv
#SBATCH --job-name=serving-probe
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=1024G
#SBATCH --time=04:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@180
set -uo pipefail

MODEL_KEY="${1:?positional 1: model_key (e.g. qwen35_4b_demo — a key of lora_modules.ARCH_DIMS_BY_MODEL_KEY)}"
SERVED_ID="${2:?positional 2: served HF id (e.g. Qwen/Qwen3.5-4B)}"
REVISION="${3:?positional 3: 40-hex revision}"
SETS="${4:-wo,qkvo,attn_mlp,mlp,down}"
RANKS="${5:-4,16,64}"
MODE="${6:-tp1}"
ATTEMPT="${7:-1}"
MAX_ATTEMPTS="${8:-4}"
PARALLEL="${9:-auto}"
REOPEN_TOKEN="${10:-}"; [ "$REOPEN_TOKEN" != - ] || REOPEN_TOKEN=""   # `-` = no token (a phase that needs positionals 11/12 without one)
PHASE="${11:-${SERVING_PROBE_PHASE:-full}}"   # full (default) | reference (the HF+PEFT reference phase ALONE over a complete record)
                           #                | parity (the PARITY PASS — natural launches at the ladder window, per-set benign scales)
                           # positional 11 wins; a self-resubmitted successor carries it as SERVING_PROBE_PHASE (an env prefix, never --export=)
case "$PHASE" in
  full|reference|parity) ;;
  reverdict) ;;              # the CPU-only certifying rewrite of a COMPLETE record (positional 12); no plan, no launches, no engine
  *) echo "[serving-probe] FATAL positional 11 phase must be full, reference, parity or reverdict, got '$PHASE'"; exit 2 ;;
esac
REVERDICT_ARG="${12:-}"    # phase=reverdict only: resume-interrupted (positional 10 is `-`) | parity-record=<abs dir> (positional 10 the receipt's nonce)
if [ "$PHASE" = reverdict ]; then
  case "$REVERDICT_ARG" in
    resume-interrupted|parity-record=/*) ;;
    *) echo "[serving-probe] FATAL phase=reverdict needs positional 12 resume-interrupted | parity-record=<abs dir>, got '$REVERDICT_ARG'"; exit 2 ;;
  esac
fi

# --- the shared root: the job's env prefix first (a configured site), then the two known per-user roots -----------
# (SISTEER_SHARED_ROOT is what every other submission script honours — memprobe_fit / hpo_trial / fit_refusal_direction)
if [ -n "${SISTEER_SHARED_ROOT:-}" ] && [ -d "$SISTEER_SHARED_ROOT" ]; then SHARED="$SISTEER_SHARED_ROOT"
else echo "[serving-probe] no SISTEER_SHARED_ROOT directory on $(hostname): set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)"; exit 2; fi
# HF cache + per-user token path: the configured site's SISTEER_HF_HOME (hf_env.sh's own first rule), else the SHARED
# copy of hf_env.sh (cluster /home is node-local) derives the cluster's cache; with neither the job REFUSES — a
# hard-coded fallback to one cluster's cache path served the wrong cache on the other
if [ -n "${SISTEER_HF_HOME:-}" ]; then
  export HF_HOME="$SISTEER_HF_HOME" HF_TOKEN_PATH="$SHARED/hf_token" HUGGINGFACE_TOKEN_PATH="$SHARED/hf_token"
elif [ -r "$SHARED/scripts/hf_env.sh" ]; then
  source "$SHARED/scripts/hf_env.sh"
else
  echo "[serving-probe] no SISTEER_HF_HOME in the job env and no $SHARED/scripts/hf_env.sh (00_setup.sh --what venvs installs it) — cannot derive the HF cache; refusing rather than guessing a cluster"; exit 2
fi
[ -n "${HF_HOME:-}" ] || { echo "[serving-probe] HF_HOME is unset after hf_env.sh — refusing"; exit 2; }
export HF_HUB_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export UV_CACHE_DIR="$SHARED/.cache/uv" UV_LINK_MODE=copy
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export WANDB_MODE=disabled WANDB_DISABLED=true

# The record dir: the certification-root family by default; `SERVING_PROBE_OUT=<dir>` (an env prefix on the submitter's
# sbatch line — reproduce/03_hpo.sh --phase probe points it at the search's root so `allowed_pairs.json` lands beside the
# HPO artefacts; never --export=) names it instead. Absolute, under the shared root, or the job refuses.
OUT="${SERVING_PROBE_OUT:-$SHARED/exp/multilora_cert/${MODEL_KEY}_probe_${SLURM_JOB_ID}}"
case "$OUT" in
  "$SHARED"/*) ;;
  *) echo "[serving-probe] SERVING_PROBE_OUT=$OUT is not under the shared root $SHARED (the record must be visible to every node)"; exit 2 ;;
esac
# the figure dir is per RECORD, not per job id: a resumed attempt's RESULT/artifacts refresh the same mirrored dir
export FIGS_DIR="$SHARED/exp/figs/serving-probe/$(basename "$OUT")"
mkdir -p "$OUT" "$FIGS_DIR"
# the SITE geometry overlay the verdict's fold writes into: the same dir geometry.site_geometry_dir()
# derives from SISTEER_SHARED_ROOT for every front-door job on this cluster, so the padded launchers read the row
export SISTEER_GEOMETRY_DIR="${SISTEER_GEOMETRY_DIR:-$SHARED/exp/reproduce/geometry}"
# THE STOP SENTINELS — BOTH honoured: the record's own $OUT/STOP, and the stop
# helper's contract $SHARED/exp/training/<job-name>/STOP (reproduce/lib/stop_run.sh and the gpu-avail stop-run.sh helper
# write there before scancel). A user who stops through the documented helper must never see a successor queued.
STOP_FILE="$OUT/STOP"
HELPER_STOP_FILE="$SHARED/exp/training/${SLURM_JOB_NAME:-serving-probe}/STOP"
stop_requested(){   # -> 0 when a deliberate stop is on record; prints which sentinel
  if [ -e "$STOP_FILE" ]; then echo "$STOP_FILE"; return 0; fi
  if [ -e "$HELPER_STOP_FILE" ]; then echo "$HELPER_STOP_FILE"; return 0; fi
  return 1
}
log(){ echo "[$(date -u +%FT%TZ)] [serving-probe $MODEL_KEY $MODE attempt $ATTEMPT/$MAX_ATTEMPTS] $*"; }

# RESULT: the driver's verdict line when it wrote one, else a state line — to
# the record AND beside the figures so the mirror carries it off the cluster
write_result(){
  # the head's attempt=N/M is the record's attempt COUNTER (hpo_probe_resume.py reads it when probe_attempt.json predates the
  # record): a single-shot phase — reverdict / reference, positionals 1/1 — PRESERVES the standing head's counter instead of
  # overwriting it (job 818826 wrote attempt=1/1 over 805161's 3/5 and the front door read the cap as reached)
  local att="$ATTEMPT/$MAX_ATTEMPTS" prior
  case "${PHASE:-full}" in reverdict|reference)
    prior="$(sed -n 's/.*\battempt=\([0-9][0-9]*\/[0-9][0-9]*\).*/\1/p' "$FIGS_DIR/RESULT" 2>/dev/null | head -1)"
    [ -z "$prior" ] || att="$prior";; esac
  local head="state=$1 exit=$2 attempt=$att phase=${PHASE:-full} elapsed=$SECONDS host=$(hostname) model=$MODEL_KEY mode=$MODE job=${SLURM_JOB_ID:-?}${SUCCESSOR:+ successor=$SUCCESSOR}"
  if [ -s "$OUT/RESULT" ]; then
    { echo "$head"; cat "$OUT/RESULT"; } > "$FIGS_DIR/RESULT"
  else
    echo "$head verdict=none" | tee "$OUT/RESULT" > "$FIGS_DIR/RESULT"
  fi
}
FINAL_RC=""
SUCCESSOR=""
on_exit(){
  local rc=$?
  [ -n "$FINAL_RC" ] && rc=$FINAL_RC
  case "$rc" in
    0) write_result DONE 0 ;;
    4) write_result INCOMPLETE 4 ;;       # the driver's EXIT_INCOMPLETE: launches outstanding, a successor queued (or the cap reached)
    6) write_result UNCERTIFIED 6 ;;      # the driver's EXIT_UNCERTIFIED: complete, but no HF+PEFT evidence / no geometry fold — never requeued
    143) write_result INTERRUPTED 143 ;;
    *) write_result FAILED "$rc" ;;
  esac
  for f in probe_plan.json probe_progress.json probe_attempt.json probe_verdict.json allowed_pairs.json memo_scaling_inputs.md reference_check.json geometry_record.json; do
    cp "$OUT/$f" "$FIGS_DIR/$f" 2>/dev/null || true
  done
  "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" 2>/dev/null || true
  log "exit rc=$rc; record $OUT; RESULT: $(head -1 "$OUT/RESULT" 2>/dev/null)"
}
trap on_exit EXIT

# --- the checkout + the submit-time sha pin (fail-closed; sdfbase_fit_cell.sbatch) -------
JOB_REC="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null)"   # ONE scheduler RPC; every field below is read from it
SCRIPT_PATH="$(sed -n 's/^ *Command=\([^ ]*\).*/\1/p' <<<"$JOB_REC" | head -1)"
TIMELIMIT="$(sed -n 's/.*TimeLimit=\([^ ]*\).*/\1/p' <<<"$JOB_REC" | head -1)"
COMMENT="$(sed -n 's/^ *Comment=\([^ ]*\).*/\1/p' <<<"$JOB_REC" | head -1)"
STDOUT_PATH="$(sed -n 's/^ *StdOut=\([^ ]*\).*/\1/p' <<<"$JOB_REC" | head -1)"
JOB_QOS="$(sed -n 's/.*[ ]QOS=\([^ ]*\).*/\1/p' <<<"$JOB_REC" | head -1)"
ESCALATE_QOS="${SISTEER_CLUSTER_QOS:-${JOB_QOS:-high}}"   # a resubmission goes out at the site's QoS, else this job's own
REPO="${SDFBASE_REPO:-}"
if [ -z "$REPO" ] && [ -n "$SCRIPT_PATH" ]; then REPO="$(cd "$(dirname "$SCRIPT_PATH")/../.." 2>/dev/null && pwd)"; fi
[ -n "$REPO" ] && [ -f "$REPO/scripts/sdfbase/serving_probe.py" ] || { log "FATAL cannot resolve the repo checkout (SCRIPT_PATH='$SCRIPT_PATH', SDFBASE_REPO='${SDFBASE_REPO:-}')"; FINAL_RC=3; exit 3; }
HEAD_SHA="$(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
case "$COMMENT" in
  *sha:*)
    # the pin is the `sha` FIELD, found BY NAME among `,`/`;`-separated key:value fields (submit_twin.nonce_of_comment's rule): the
    # funnel appends `nonce:<n>` (comment_with_nonce) and may add fields in any order; the literal read of everything after `sha:`
    # refused every nonce-commented job at start (830863/830864)
    PIN=""; _rest="$COMMENT"
    while [ -n "$_rest" ]; do _f="${_rest%%[,;]*}"; case "$_f" in sha:*) PIN="${_f#sha:}" ;; esac; [ "$_f" = "$_rest" ] && break; _rest="${_rest#*[,;]}"; done
    [ -n "$PIN" ] || { log "FATAL --comment '$COMMENT' carries no sha:<40hex> field"; FINAL_RC=3; exit 3; } ;;
  *) log "FATAL no --comment=sha:<40hex> pin on this job (got '$COMMENT'); submit with --comment=sha:\$(git -C $REPO rev-parse HEAD)"; FINAL_RC=3; exit 3 ;;
esac
[ "$PIN" = "$HEAD_SHA" ] || { log "FATAL checkout $REPO is at $HEAD_SHA, the submission pinned $PIN — refusing to run unvalidated code"; FINAL_RC=3; exit 3; }
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null)" ] || { log "FATAL checkout $REPO has modified tracked files"; FINAL_RC=3; exit 3; }

# --- fail-fast guards on every external input ------------------------------------------
REQS="$REPO/configs/capeval/vllm-requirements.txt"
[ -s "$REQS" ] || { log "no serving requirements at $REQS"; FINAL_RC=2; exit 2; }
[ -s "$REPO/src/sisteer/capeval/serving/standup_manifest.py" ] || { log "standup_manifest.py missing in $REPO"; FINAL_RC=2; exit 2; }
[ -x "$SHARED/scripts/mirror-figs.sh" ] || log "WARNING: no $SHARED/scripts/mirror-figs.sh — RESULT/artifacts will not mirror"
# this job's GPU count, READ from Slurm's export (the driver pins each standup to a disjoint slice of the same value;
# this script never sets it — hard rule 3); it sizes the successor's --gres so a resumed attempt keeps the width
NGPU="$(echo "${CUDA_VISIBLE_DEVICES:-}" | tr ',' '\n' | grep -c .)"
case "$MODE" in
  tp1) TP=1 ;;
  tp2) TP=2; [ "$NGPU" -ge 2 ] || { log "FATAL mode tp2 needs 2 GPUs (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-}'); submit with --gres=gpu:2"; FINAL_RC=2; exit 2; } ;;
  *) log "FATAL mode must be tp1 | tp2, got '$MODE'"; FINAL_RC=2; exit 2 ;;
esac
[ "$NGPU" -ge "$TP" ] || NGPU="${SLURM_GPUS_ON_NODE:-$TP}"
case "$PARALLEL" in
  auto|[1-9]|[1-9][0-9]) ;;
  *) log "FATAL positional 9 parallel must be auto or a positive integer, got '$PARALLEL'"; FINAL_RC=2; exit 2 ;;
esac
REOPEN=()
if [ -n "$REOPEN_TOKEN" ]; then
  [[ "$REOPEN_TOKEN" =~ ^[0-9a-f]{32}$ ]] || { log "FATAL positional 10 reopen_token must be the revocation receipt's 32-hex nonce"; FINAL_RC=2; exit 2; }
  REOPEN=(--reopen-token "$REOPEN_TOKEN")
fi
# the run's time budget: the job's TimeLimit minus 45 min for the HF+PEFT reference phase, verdict and memo
# every field is forced to base 10: bash reads a zero-padded field such as `08` as OCTAL and the arithmetic dies ("value too great for
# base"), so `08:00:00` yielded an EMPTY limit and the 14400 fallback — job 820268 (attempt 5/5, --time 08:00:00) ran on the 4-hour budget
# of 11,700 s inside an 8-hour allocation (2026-10-01); a field that is not a number still yields 0 (the caller's fallback)
limit_to_s(){ local t="$1" d=0; case "$t" in *-*) d="${t%%-*}"; t="${t#*-}";; esac
  local IFS=:; set -- $t; local f; for f in "$d" "$@"; do case "$f" in ''|*[!0-9]*) echo 0; return;; esac; done
  case $# in 3) echo $(( 10#$d*86400 + 10#$1*3600 + 10#$2*60 + 10#$3 ));; 2) echo $(( 10#$d*86400 + 10#$1*60 + 10#$2 ));; *) echo 0;; esac; }
LIMIT_S="$(limit_to_s "${TIMELIMIT:-04:00:00}")"; [ "$LIMIT_S" -gt 0 ] || LIMIT_S=14400
BUDGET_S=$(( LIMIT_S - 2700 )); [ "$BUDGET_S" -ge 600 ] || BUDGET_S=600
log "model=$MODEL_KEY served=$SERVED_ID rev=${REVISION:0:12} sets=$SETS ranks=$RANKS mode=$MODE tp=$TP parallel=$PARALLEL gpus=$NGPU repo=$REPO sha=$HEAD_SHA node=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-} out=$OUT budget=${BUDGET_S}s geometry_dir=$SISTEER_GEOMETRY_DIR stop_sentinels=$STOP_FILE,$HELPER_STOP_FILE"

# --- THE ATTEMPT RECORD ----------------------------------------------------------------------------------------------
# This attempt's shape as ONE JSON object at $OUT/probe_attempt.json, written at START so a resume reads the LAST attempt's
# shape whatever its exit ($OUT/RESULT is the driver's verdict line and is written once — it is not this record; the log
# line above is prose). reproduce/03_hpo.sh --phase probe resumes an INCOMPLETE record from it: attempt+1, the same cap,
# the same positionals 4-6, the same width by default (scripts/sdfbase/hpo_probe_resume.py). tmp + mv: never a torn file.
case "$ATTEMPT:$MAX_ATTEMPTS" in *[!0-9:]*|:*|*:) log "FATAL positionals 7/8 attempt/max_attempts must be positive integers, got '$ATTEMPT'/'$MAX_ATTEMPTS'"; FINAL_RC=2; exit 2;; esac
[ "$ATTEMPT" -ge 1 ] && [ "$MAX_ATTEMPTS" -ge 1 ] || { log "FATAL positionals 7/8 attempt/max_attempts must be >= 1, got $ATTEMPT/$MAX_ATTEMPTS"; FINAL_RC=2; exit 2; }
attempt_record_write(){   # this attempt's shape -> $OUT/probe_attempt.json (tmp + mv); phase=reverdict runs no launches and skips it (the call below)
# An operator cap widen (03_hpo.sh --probe-max-attempts) rides as SERVING_PROBE_CAP_WIDEN — one line, recorded on the attempt record
# as `cap_widen` (null when the cap is the record's own); the front door strips quotes and backslashes, the tr here is the belt.
# The record's `cap_history` is APPEND-ONLY and carried forward by every attempt — the previous record's history
# plus, when this attempt widens the cap, {from (the previous record's cap), to, attempt, utc, host, note} — so a later attempt's rewrite
# of probe_attempt.json never drops who raised the cap and why. Read BEFORE the mv replaces the file; python3 is the system interpreter
# (the serving venv is not built yet at this point). Only a MISSING file is a fresh record — an existing
# file that is not JSON, not an object, or whose cap_history / max_attempts have the wrong shape REFUSES the attempt with the file
# untouched (the earlier `except Exception: pass` would have overwritten a damaged record with an empty history); and an entry is
# appended ONLY when this attempt's cap exceeds the previous record's (the note's own A->B must be that transition) — a self-resubmitted
# successor inherits SERVING_PROBE_CAP_WIDEN through Slurm's implicit export, carries the history unchanged and records cap_widen null.
local cw=null hist widened out
out="$(python3 - "$OUT/probe_attempt.json" "${SERVING_PROBE_CAP_WIDEN:-}" "$ATTEMPT" "$MAX_ATTEMPTS" "$(hostname)" <<'PYH'
import json, re, sys, time
prev, note, att, cap, host = sys.argv[1], sys.argv[2].strip(), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
hist, old = [], None
try:
    f = open(prev, encoding="utf-8")
except FileNotFoundError:
    f = None                                   # a genuinely fresh record: nothing to carry
if f is not None:
    with f:
        try:
            d = json.load(f)
        except ValueError as e:
            sys.exit(f"{prev} exists but is not JSON ({e}) — refusing to rewrite a damaged attempt record")
    if not isinstance(d, dict):
        sys.exit(f"{prev} is not a JSON object — refusing to rewrite a damaged attempt record")
    h = d.get("cap_history")
    if h is None:
        h = []                                  # a record written before cap_history existed
    if not isinstance(h, list) or any(not isinstance(x, dict) for x in h):
        sys.exit(f"{prev}: cap_history is not a list of entries — refusing to rewrite a damaged attempt record")
    mx = d.get("max_attempts")
    if isinstance(mx, bool) or not isinstance(mx, int) or mx < 1:
        sys.exit(f"{prev}: max_attempts {mx!r} is not a positive integer — refusing to rewrite a damaged attempt record")
    hist, old = h, mx
widened = False
if note and old is not None and cap > old:
    m = re.search(r"cap widened (\d+)->(\d+)", note)
    if m and (int(m.group(1)), int(m.group(2))) != (old, cap):
        sys.exit(f"the cap-widen note names {m.group(1)}->{m.group(2)} but this record moves {old}->{cap} — refusing to record a transition that is not this record's")
    hist.append({"from": old, "to": cap, "attempt": att, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": host, "note": note})
    widened = True
print("1" if widened else "0")
print(json.dumps(hist, ensure_ascii=True))
PYH
)" || { if [ -f "$OUT/probe_attempt.json" ] || [ -n "${SERVING_PROBE_CAP_WIDEN:-}" ]; then log "FATAL cannot read/record the cap history of $OUT/probe_attempt.json (python3 refused or failed; its reason is above) — refusing to rewrite the attempt record (a rewrite that drops or corrupts cap_history is the hole the review named on 12a72b85a)"; FINAL_RC=2; exit 2; fi; out=$'0\n[]'; }
widened="${out%%$'\n'*}"; hist="${out#*$'\n'}"
[ "$widened" != 1 ] || cw="\"$(printf '%s' "$SERVING_PROBE_CAP_WIDEN" | tr -d '"\\' | tr '\n' ' ')\""
printf '{"version": 1, "written_utc": "%s", "attempt": %d, "max_attempts": %d, "gpus": %d, "parallel": "%s", "mode": "%s", "tp": %d, "sets": "%s", "ranks": "%s", "model_key": "%s", "served_hf_id": "%s", "revision": "%s", "job_id": "%s", "job_name": "%s", "sha": "%s", "qos": "%s", "time_limit": "%s", "cpus_per_task": "%s", "mem_mb": "%s", "host": "%s", "cap_widen": %s, "cap_history": %s}\n' \
  "$(date -u +%FT%TZ)" "$ATTEMPT" "$MAX_ATTEMPTS" "$NGPU" "$PARALLEL" "$MODE" "$TP" "$SETS" "$RANKS" "$MODEL_KEY" "$SERVED_ID" "$REVISION" \
  "${SLURM_JOB_ID:-}" "${SLURM_JOB_NAME:-serving-probe}" "$HEAD_SHA" "${JOB_QOS:-}" "${TIMELIMIT:-}" "${SLURM_CPUS_PER_TASK:-}" "${SLURM_MEM_PER_NODE:-}" "$(hostname)" "$cw" "$hist" \
  > "$OUT/probe_attempt.json.tmp.$$" && mv -f "$OUT/probe_attempt.json.tmp.$$" "$OUT/probe_attempt.json" \
  || { log "FATAL cannot write $OUT/probe_attempt.json"; FINAL_RC=2; exit 2; }
}
[ "${PHASE:-full}" = reverdict ] || attempt_record_write
log "attempt record: $( [ "${PHASE:-full}" = reverdict ] && echo "NOT written — phase=reverdict runs no launches; the record's last attempt shape stands" || echo "$OUT/probe_attempt.json (attempt $ATTEMPT/$MAX_ATTEMPTS, gpus=$NGPU, parallel=$PARALLEL)" )"

# --- the serving venv: selected CENTRALLY by the job's model key: common.sh's
# f22_serving_venv_for_key — qwen35_* -> the `gdn` variant (flash-linear-attention + causal-conv1d, the Qwen3.5 hybrids' fused
# gated-deltanet kernels, which the HF+PEFT reference and the engine both run on in production), nemotron3nano4b -> `nemotron`,
# else the base — the same venv, bootstrap recipe and identity exports (SISTEER_SERVING_VENV / _VARIANT / _REQS_SHA256) as every
# production standup, so the certificate the probe writes names the venv production serves in; a certifying probe on the base
# while production serves on a variant was the finding. The launch, the HF+PEFT reference AND the reverdict run from it.
F22_COMMON="$REPO/scripts/sdfbase/full22/common.sh"
[ -f "$F22_COMMON" ] || { log "FATAL no $F22_COMMON — the serving venv is selected by key through common.sh's f22_serving_venv_for_key; refusing to build a venv of my own"; FINAL_RC=3; exit 3; }
# shellcheck source=/dev/null
source "$F22_COMMON"
REPO_DIR="$REPO"
f22_serving_venv_for_key "$MODEL_KEY" || { log "serving venv selection / bootstrap failed for key $MODEL_KEY"; FINAL_RC=3; exit 3; }
VENV="$SERVE_VENV"
log "serving venv: $VENV (variant '${SISTEER_SERVING_VENV_VARIANT:-base}', requirements sha256 ${SERVE_REQS_SHA:0:16}; selected by key $MODEL_KEY)"
# ONE interpreter for both probes (each import of torch+vllm costs seconds): the required stack INCLUDING peft — the
# HF+PEFT reference check (assertion 4) is a certification check, so a venv without peft FAILS the job here rather than
# running a probe whose verdict cannot certify (peft==0.20.0 is pinned in
# configs/capeval/vllm-requirements.txt, the repo's uv.lock pin)
"$VENV/bin/python" -c 'import vllm, torch, aiohttp, safetensors, huggingface_hub
import peft
print("serving venv OK", vllm.__version__, torch.__version__, torch.cuda.is_available(), "peft", peft.__version__)' \
  || { log "serving venv guard FAILED: vllm / torch / peft must all import from $VENV (peft==0.20.0 is REQUIRED for the HF+PEFT parity check — D55); rebuild from $REQS"; FINAL_RC=3; exit 3; }
# the lora-native launch keeps the lens plugin out (VLLM_PLUGINS="" is composed per launch); no lens leg exists here (FD 19)
unset VLLM_PLUGINS 2>/dev/null || true
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

cd "$REPO" || { FINAL_RC=2; exit 2; }

# --- phase=reverdict (routed by 03_hpo.sh --phase probe --reopen): the CPU-only certifying rewrite of a
# COMPLETE record, under the serving venv, on the record's own cluster — `resume-interrupted` adopts the record's one in-progress receipt (a
# reverdict that consumed it and died; never a fresh revocation), `parity-record=<dir>` re-reads assertion 4 from the parity-pass record under
# the receipt the front door minted (positional 10). No plan, no launches, no engine, no successor; on_exit writes RESULT. The FD 23 memprobe
# rides when present, as the memo step below passes it.
if [ "${PHASE:-full}" = reverdict ]; then
  RV=()
  case "$REVERDICT_ARG" in
    resume-interrupted) [ -z "$REOPEN_TOKEN" ] || { log "FATAL phase=reverdict resume-interrupted takes no reopen token (positional 10 must be -)"; FINAL_RC=2; exit 2; }
                        RV=(--resume-interrupted) ;;
    parity-record=*) [ -n "$REOPEN_TOKEN" ] || { log "FATAL phase=reverdict parity-record needs the revocation receipt's nonce as positional 10"; FINAL_RC=2; exit 2; }
                     RV=(--reopen-token "$REOPEN_TOKEN" --parity-record "${REVERDICT_ARG#parity-record=}") ;;
  esac
  MEMPROBE_DIR="$SHARED/exp/reproduce/$MODEL_KEY/02_prepare_model"; MEMPROBE_ARGS=()
  for arm in lora_r4_ce psr_ce_all; do [ -s "$MEMPROBE_DIR/memprobe.$arm.json" ] && { MEMPROBE_ARGS=(--memprobe "$MEMPROBE_DIR/memprobe.$arm.json"); break; }; done
  log "phase=reverdict: the certifying rewrite of $OUT (${RV[*]}) — no plan, no launches, no engine"
  "$VENV/bin/python" scripts/sdfbase/serving_probe.py reverdict --record "$OUT" --probe-job-id "$SLURM_JOB_ID" "${RV[@]}" "${MEMPROBE_ARGS[@]}"; RC=$?
  FINAL_RC=$RC; log "reverdict exit $RC; record at $OUT"; exit $RC
fi

# --- resume: attempt > 1, or an INCOMPLETE launch-list checkpoint already under $OUT (the driver refuses to re-plan
# over a partial record without --resume). A resumed attempt ALSO passes --replan: the driver measures the
# batched-token budget from the record (the largest S x L that came up), re-derives S = min(stride, KV, requested,
# budget // L) for every outstanding launch and re-opens the start-time prefill OOMs above the budget (their results
# move to launches/superseded/, never deleted); idempotent, so every successor may carry it ---------------------------
RESUME=()
if [ "${PHASE:-full}" = reference ]; then
  # phase=reference re-runs the HF+PEFT reference phase ALONE over a record whose launches are all done (1 GPU, no
  # engine); the driver refuses a record with launches to run, and a COMPLETE record needs positional 10 (the receipt)
  RESUME=(--resume --reference-only); log "phase=reference: re-judging every adapter of $OUT with the HF+PEFT reference alone (no engine)"
elif [ "$ATTEMPT" -gt 1 ]; then RESUME=(--resume --replan)
elif [ -s "$OUT/probe_progress.json" ] && "$VENV/bin/python" -c 'import json,sys; sys.exit(0 if not json.load(open(sys.argv[1])).get("complete") else 1)' "$OUT/probe_progress.json"; then
  RESUME=(--resume --replan); log "an incomplete probe_progress.json is under $OUT — resuming it (with --replan)"
fi

# --- the self-resubmission (the fit path's pattern: positional counter, --no-requeue, the sentinel, never --export=) --
escalate(){   # <why>: queue attempt+1 on the SAME record; sets SUCCESSOR
  local sentinel
  case "${PHASE:-full}" in reference|reverdict) log "phase=$PHASE is single-shot (no launches to continue) — not resubmitting ($1)"; return;; esac
  if sentinel="$(stop_requested)"; then log "STOP sentinel $sentinel present — deliberate stop, not resubmitting ($1)"; return; fi
  if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then log "attempt cap $MAX_ATTEMPTS reached — not resubmitting ($1); the record stays PARTIAL under $OUT"; return; fi
  if [ -z "$SCRIPT_PATH" ] || [ ! -f "$SCRIPT_PATH" ]; then log "cannot resolve the script path — not resubmitting ($1)"; return; fi
  log "resubmitting ($1) at qos=$ESCALATE_QOS as attempt $((ATTEMPT + 1))/$MAX_ATTEMPTS (time=$TIMELIMIT, gpus=$NGPU, parallel=$PARALLEL, out=$OUT)"
  # SERVING_PROBE_OUT rides as a LITERAL env prefix (Slurm's implicit export) so the successor resumes THIS record; the
  # site's SISTEER_SHARED_ROOT / SISTEER_HF_HOME / SISTEER_CLUSTER_QOS are already in this job's environment and travel
  # the same way (an expanded `${V:+V=…}` word would NOT be an assignment prefix — it would become the command name).
  # The GPU count / cpus / mem are this job's own (NGPU read from Slurm's CUDA_VISIBLE_DEVICES above — a successor keeps
  # the width the parallel run had; the tp2 / narrower submissions pass theirs on the command line, not the header).
  # Positional 10 (the reopen token) is NEVER handed on: a successor continues an incomplete record (re-check 2).
  local out
  out="$(SERVING_PROBE_OUT="$OUT" SERVING_PROBE_PHASE="${PHASE:-full}" \
    sbatch --parsable --qos="$ESCALATE_QOS" --job-name="${SLURM_JOB_NAME:-serving-probe}" ${TIMELIMIT:+--time="$TIMELIMIT"} \
           --gres="gpu:$NGPU" ${SLURM_CPUS_PER_TASK:+--cpus-per-task="$SLURM_CPUS_PER_TASK"} ${SLURM_MEM_PER_NODE:+--mem="${SLURM_MEM_PER_NODE}M"} \
           ${STDOUT_PATH:+--output="$STDOUT_PATH"} --comment="$COMMENT" "$SCRIPT_PATH" \
           "$MODEL_KEY" "$SERVED_ID" "$REVISION" "$SETS" "$RANKS" "$MODE" "$((ATTEMPT + 1))" "$MAX_ATTEMPTS" "$PARALLEL")" \
    && { SUCCESSOR="${out%%;*}"; log "successor job $SUCCESSOR queued"; } \
    || log "RESUBMIT FAILED — the probe will not continue automatically; resume through the front door (reproduce/03_hpo.sh --phase probe reads $OUT/probe_attempt.json) or by hand: SERVING_PROBE_OUT=$OUT sbatch ... $SCRIPT_PATH $MODEL_KEY $SERVED_ID $REVISION $SETS $RANKS $MODE $((ATTEMPT + 1)) $MAX_ATTEMPTS $PARALLEL"
}
WORK=""
handled=0
on_term(){   # preemption / node drain / cancel: DISARM FIRST (the re-entry measured 2026-09-11), drain the driver, resubmit
  trap '' SIGTERM
  [ "$handled" -eq 1 ] && return; handled=1
  log "SIGTERM — draining the probe driver (its per-launch results + probe_progress.json are the resume point)"
  [ -n "$WORK" ] && kill -TERM "$WORK" 2>/dev/null
  [ -n "$WORK" ] && wait "$WORK" 2>/dev/null
  FINAL_RC=143
  escalate "SIGTERM"
  exit 143
}
trap on_term SIGTERM SIGINT SIGQUIT

# --parallel: N standups at a time, each engine on its own disjoint slice of Slurm's CUDA_VISIBLE_DEVICES; the
# driver derives `auto` from the allocation and refuses a width it cannot hold
PLAN_ARGS="--sets $SETS --ranks $RANKS --tp $TP"
RUN_EXTRA=()
# PARITY CERTIFICATE v2: the job env may name the rank:set pairs to certify (SERVING_PROBE_CERTIFY_V2=16:down,16:mlp,…) — a
# resume with --replan then plans their current servers at the ladder window and re-runs their v1-only launches
[ -z "${SERVING_PROBE_CERTIFY_V2:-}" ] || { RUN_EXTRA+=(--certify-v2 "$SERVING_PROBE_CERTIFY_V2"); log "certify-v2 pairs: $SERVING_PROBE_CERTIFY_V2"; }
if [ "${PHASE:-full}" = parity ]; then
  # the parity pass — the plan is written with the per-set benign scales at the ladder window alone; no throughput leg.
  # The scales are the driver's table unless the job env names others (SERVING_PROBE_PARITY_SCALES, the front door's --parity-scales:
  # a model whose surface needs another scale — the pass's own parity_weak / parity_chaotic gates say so); the value's shape is checked
  # here so a typo refuses at start, never a plan written with a scale the driver cannot parse
  PARITY_SCALES="${SERVING_PROBE_PARITY_SCALES:-default}"
  case "$PARITY_SCALES" in
    default) ;;
    *) [[ "$PARITY_SCALES" =~ ^[a-z_]+=[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?(,[a-z_]+=[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)*$ ]] \
         || { log "FATAL SERVING_PROBE_PARITY_SCALES must be default or set=scale,… (e.g. wo=0.5,mlp=0.05), got '$PARITY_SCALES'"; FINAL_RC=2; exit 2; } ;;
  esac
  PLAN_ARGS="$PLAN_ARGS --parity-pass --parity-scales $PARITY_SCALES"
  # APPEND, never assign: `RUN_EXTRA=(--no-throughput)` here threw away the --certify-v2 the line above had queued, so job 819493
  # (parity attempt 3/3, SERVING_PROBE_CERTIFY_V2=all) ran a plain resume — 29/29 done, 0 superseded, no v2 probe — and consumed the
  # record's last attempt measuring nothing (2026-10-01)
  RUN_EXTRA+=(--no-throughput)
  log "phase=parity: PARITY PASS on $OUT — plan args '$PLAN_ARGS' (natural launches at the ladder window, per-set benign scales)"
fi
srun --ntasks=1 "$VENV/bin/python" scripts/sdfbase/serving_probe.py run --model-key "$MODEL_KEY" \
  --model-id "$SERVED_ID" --revision "$REVISION" --out "$OUT" --time-budget-s "$BUDGET_S" \
  --plan-args "$PLAN_ARGS" --parallel "$PARALLEL" "${RESUME[@]}" "${REOPEN[@]}" "${RUN_EXTRA[@]}" &
WORK=$!
wait $WORK; RC=$?
WORK=""
log "run exit $RC; writing the verdict + memo from whatever finished"
# the verdict REFUSES (exit 4, a PARTIAL non-certifying probe_verdict.json, no allowed_pairs.json) while a launch that
# could still move a pair's verdict is outstanding — its exit is the job's when the run itself was clean
"$VENV/bin/python" scripts/sdfbase/serving_probe.py verdict --model-key "$MODEL_KEY" --out "$OUT" \
  --probe-job-id "$SLURM_JOB_ID" --verdict-uri "$OUT/probe_verdict.json"; VRC=$?
if [ "$RC" -eq 0 ]; then RC=$VRC; elif [ "$RC" -eq 4 ] && [ "$VRC" -ne 0 ] && [ "$VRC" -ne 4 ]; then RC=$VRC; fi
# the FD 23 record: reproduce/02_prepare_model.sh --step memprobe writes memprobe.<arm>.json under the model's reproduce
# state dir (the lora arm's first when both ran). Passed ONLY when present — `memo` refuses a NAMED record that is
# not there; with none the memo's §7 renders its placeholder and says so.
MEMPROBE_DIR="$SHARED/exp/reproduce/$MODEL_KEY/02_prepare_model"; MEMPROBE_ARGS=()
for arm in lora_r4_ce psr_ce_all; do
  [ -s "$MEMPROBE_DIR/memprobe.$arm.json" ] && { MEMPROBE_ARGS=(--memprobe "$MEMPROBE_DIR/memprobe.$arm.json"); break; }
done
[ ${#MEMPROBE_ARGS[@]} -gt 0 ] || log "no memprobe.<arm>.json under $MEMPROBE_DIR — the memo's fit-throughput section renders its placeholder"
"$VENV/bin/python" scripts/sdfbase/serving_probe.py memo --model-key "$MODEL_KEY" --out "$OUT" \
  "${MEMPROBE_ARGS[@]}" || RC=$(( RC == 0 ? 5 : RC ))
# exit 4 = launches outstanding (the time budget): the SAME record continues in a successor job
[ "$RC" -eq 4 ] && escalate "time budget: launches outstanding"
FINAL_RC=$RC
log "driver exit $RC; record at $OUT"
exit $RC
