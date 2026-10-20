#!/bin/bash
#SBATCH --job-name=m5t-nemo-probe
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=0-01:30:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# math500-thinking-on-full22 — the NEMOTRON FUSED-KERNEL BACKEND PROBE (the standing Nemotron rule: the
# fused Mamba kernels are a PREREQUISITE, never inferred from a missed walltime). ONE GPU, <= ~15 min once the
# variant venv exists (its FIRST build — mamba-ssm + causal-conv1d from source against torch 2.11 cu129 — takes
# 20-40 min, hence the 1.5 h wall). Steps (nemo_backend_probe.py):
#   1. f22_serving_venv nemotron — the campaign's pinned serving stack + the kernel overlay, in a SEPARATE venv;
#   2. `packages` — mamba_ssm / causal_conv1d import at the fit env's pins;
#   3. mt_serving_up.py stands Nemotron up (--cell none, capability:gen_long_math, stacked, attested) exactly as
#      the math500 jobs will (F22_REQUIRE_NEMO_BACKEND is OFF for the probe itself — it is what WRITES the attestation);
#   4. `dispatch` — the static dispatch map of the served model's Mamba mixer + the live workers' report when the
#      tree's worker answers sisteer_mamba_backend_report;
#   5. `duty` — 64 sequences x 512 tokens through the proxy under `nvidia-smi dmon -s u -d 1`: SM duty + tok/s —
#      measured TWICE on the same GPU in this job: Nemotron first (its standup attests the served concurrency),
#      then the REFERENCE — a same-size hybrid peer (default qwen35_4b) stood up on the campaign's base venv at that
#      same concurrency, once Nemotron has left the GPU. The served path runs eager (vllm-lens forces --enforce-eager, CUDA graphs off), so
#      every model's decode is launch-bound and an absolute duty floor calibrates nothing (0.28 measured against
#      a 0.90 floor with the Triton kernels attested, probe 476985);
#   6. `attest` — mamba_backend_attestation.json under $WAVE_ROOT/nemo_backend/ (ADMISSION = fused on every symbol
#      AND Nemotron's SM duty >= F22_NEMO_DUTY_RATIO x the peer's AND tok/s >= the ratio x the peer's; keyed by
#      the serving requirements' sha); f22_model_facts requires it for every Nemotron math500 job on this cluster
#      and re-derives the rule from the recorded numbers. The probe is run ONCE per cluster per serving environment.
# Usage (positional only — never --export=):
#   sbatch -J m5t-nemo-probe scripts/sdfbase/math500/nemo_backend_probe.sbatch <wave_sha> [peer_key=qwen35_4b]
set -uo pipefail
WAVE_SHA="${1:?wave sha}"
# --- the checkout, located BEFORE either helper is sourced ---------------------------------------------------------
# A Slurm batch process inherits no REPO_DIR (f22_init sets it, later) and $0 is the spool copy, so the path is derived
# the way f22_init derives it: the shared root present on this node + the wave sha; M5T_REPO_DIR overrides it (the
# grading host: the dry run, the startup test; m5t_init refuses if the two ever differ). A missing root or checkout is
# refused BY NAME on stderr before anything else can fail. This block is byte-identical in every launcher (the
# test pins it): the helpers cannot locate themselves.
if [ -n "${M5T_REPO_DIR:-}" ]; then REPO_DIR="$M5T_REPO_DIR"
elif [ -n "${SISTEER_SHARED_ROOT:-}" ]; then REPO_DIR="${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA"
elif [ -n "${SISTEER_EVAL_REPO_ROOT:-}" ]; then REPO_DIR="$SISTEER_EVAL_REPO_ROOT/$WAVE_SHA"
else echo "FATAL: set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config) — nothing to source" >&2; exit 1; fi
[ -r "$REPO_DIR/scripts/sdfbase/full22/common.sh" ] && [ -r "$REPO_DIR/scripts/sdfbase/math500/m5t_common.sh" ] \
  || { echo "FATAL: no checkout at $REPO_DIR (scripts/sdfbase/full22/common.sh + scripts/sdfbase/math500/m5t_common.sh) — README: git -C <the repo mirror> worktree add --detach $REPO_DIR $WAVE_SHA" >&2; exit 1; }
source "$REPO_DIR/scripts/sdfbase/full22/common.sh"; source "$REPO_DIR/scripts/sdfbase/math500/m5t_common.sh"
# --- the cleanup trap, armed BEFORE the first step that can fail (m5t_init) and safe at ANY step: every value the record
# reads has a default; the RESULT record goes to $FIGS_DIR once the wave root is resolved, to stderr (the .out) before
TARGET_PID=""; STANDUP_STOP=""; RC=1; M5T_RESULT=INCOMPLETE
cleanup(){
  [ -n "${STANDUP_STOP:-}" ] && touch "$STANDUP_STOP" 2>/dev/null || true
  [ -n "${TARGET_PID:-}" ] && { for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; }
  [ -z "${ATTESTATION:-}" ] || [ -z "${FIGS_DIR:-}" ] || cp "$ATTESTATION" "$FIGS_DIR/mamba_backend_attestation_${SLURM_JOB_ID:-manual}.json" 2>/dev/null || true
  local rec="nemo-backend-probe rc=$RC result=$M5T_RESULT attestation=${ATTESTATION:-unresolved}"
  if [ -n "${FIGS_DIR:-}" ]; then f22_result "$rec"; else echo "RESULT (before the wave root resolved; no figs dir yet): $rec" >&2; fi
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
m5t_init "$WAVE_SHA"
# the probe WRITES the attestation the math500 jobs require; it must not require it of itself
export F22_REQUIRE_NEMO_BACKEND=0
f22_stop_requested && { f22_log "wave STOP present — not probing"; RC=0; M5T_RESULT=STOP_REQUESTED; exit 0; }
KEY=nemotron3nano4b; CELL=none; SAMPLING_MODE=thinking_card
REF_KEY="${2:-qwen35_4b}"   # the same-GPU peer (a 4B hybrid on the campaign's base venv); positional, never --export=
[ "$REF_KEY" != "$KEY" ] || f22_fatal "the peer reference must be another model, not $KEY"
m5t_row "$KEY" "$CELL" "$SAMPLING_MODE"
m5t_resolve_config
PROBE_ROOT="$WAVE_ROOT/nemo_backend"; RUN_DIR="$PROBE_ROOT/probe_${SLURM_JOB_ID:-manual}"; SERVE_DIR="$RUN_DIR/serving"
REF_DIR="$RUN_DIR/reference_$REF_KEY"
mkdir -p "$RUN_DIR" "$SERVE_DIR/math500" "$REF_DIR/serving/math500"
ATTESTATION="$PROBE_ROOT/mamba_backend_attestation.json"
TARGET_GPU="$CUDA_VISIBLE_DEVICES"
PROBE="$REPO_DIR/scripts/sdfbase/math500/nemo_backend_probe.py"
BASE_REQS="$REPO_DIR/configs/capeval/vllm-requirements.txt"; OVERLAY_REQS="$REPO_DIR/configs/capeval/vllm-requirements-nemotron.txt"
# 1 + 2: the variant venv, and the packages the variant exists for
f22_serving_venv nemotron
f22_client_venv
"$SPY" "$PROBE" packages --out "$RUN_DIR" || { M5T_RESULT=PACKAGES_FAILED; f22_log "the kernel packages do not import in the variant venv"; exit 1; }
# 3: Nemotron on the variant venv, exactly as the math500 jobs stand it up: the `none` row's extras (m5t_standup_extra:
#    --cell none, no digest, no parser) at the experiment's geometry; the standup attests its served concurrency (TARGET_S)
f22_bank_view_check "$CPY"
m5t_standup_extra
f22_standup attacks "$SERVE_DIR/math500" "" "$M5T_GEOMETRY_EVAL" "" "" "" || { M5T_RESULT=STANDUP_FAILED; exit 1; }
NEMO_S="$TARGET_S"
# 4: the dispatch map (static + the live RPC when answered)
"$SPY" "$PROBE" dispatch --out "$RUN_DIR" --endpoint-json "$SERVE_DIR/math500/endpoint.json" || f22_log "dispatch map: a non-fused or absent symbol was recorded (the attest step decides)"
# 5: Nemotron's decode burst under dmon
"$CPY" "$PROBE" duty --out "$RUN_DIR" --endpoint-json "$SERVE_DIR/math500/endpoint.json" --n-seqs 64 --max-tokens 512 || { M5T_RESULT=DUTY_FAILED; exit 1; }
f22_standup_stop
# 6: THE SAME-GPU PEER REFERENCE — the peer on the campaign's base venv, at Nemotron's attested served
#    concurrency, the identical burst, on this job's GPU, after Nemotron has left it (the standup's free-memory preflight
#    refuses a lingering engine by name)
f22_serving_venv ""
m5t_row "$REF_KEY" "$CELL" "$SAMPLING_MODE"
m5t_resolve_config
f22_bank_view_check "$CPY"
m5t_standup_extra
# The peer is served at NEMOTRON's width so the two bursts have the same in-flight shape (64 requests over S seqs: a
# first wave of S, then the rest — the duty trace is two plateaus, 0.6 then 0.3 at S = 44). The peer's own certified
# identity is narrower (qwen35_4b: 20 x 17,664 under its provisional batched-token budget), so mt_serving_up refuses
# S = 44 by name; this reference standup is a throughput probe, never a scored run, and passes the launcher's explicit
# --allow-over-budget HERE AND ONLY HERE (probe 477970 died on the refusal). Production jobs never carry it.
F22_STANDUP_EXTRA+=(--allow-over-budget)
f22_standup attacks "$REF_DIR/serving/math500" "" "$M5T_GEOMETRY_EVAL" "" "$NEMO_S" "" || { M5T_RESULT=REFERENCE_STANDUP_FAILED; exit 1; }
"$CPY" "$PROBE" duty --out "$REF_DIR" --endpoint-json "$REF_DIR/serving/math500/endpoint.json" --n-seqs 64 --max-tokens 512 || { M5T_RESULT=REFERENCE_DUTY_FAILED; exit 1; }
f22_standup_stop
# 7: the attestation (keyed by the VARIANT venv's requirements; admitted against the peer)
f22_serving_venv nemotron
if "$SPY" "$PROBE" attest --out "$RUN_DIR" --attestation "$ATTESTATION" --requirements "$BASE_REQS" "$OVERLAY_REQS" \
     --duty-reference "$REF_DIR/duty.json" --duty-reference-key "$REF_KEY" --duty-ratio "$F22_NEMO_DUTY_RATIO"; then
  M5T_RESULT=ADMITTED; RC=0; f22_log "Nemotron backend ADMITTED: $ATTESTATION"
else
  M5T_RESULT=REFUSED; RC=2; f22_log "Nemotron backend REFUSED (see $ATTESTATION) — the Nemotron cells are NOT launched until the environment is fixed"
fi
exit $RC
