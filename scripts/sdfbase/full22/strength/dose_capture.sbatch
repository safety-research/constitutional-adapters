#!/bin/bash
#SBATCH --job-name=dose-capture
#SBATCH --partition=general
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=0-01:30:00
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# The strength frontier's COMMON PHYSICAL DOSE capture: ONE target per job, one GPU, an HF forward
# of the BASE model over the shared reference pool under the served render, then the realised dose
# of every LoRA + AlphaSteer arm computed analytically from the staged object bytes
# (scripts/sdfbase/full22/strength/dose_capture.py — read its header). No serving, no object
# served, nothing under src/sisteer/capeval/serving/ touched.
#
# --qos and --output are given AT SUBMIT TIME because they differ per cluster (each cluster's high
# QoS; the mode-0700 $SHARED/exp/jailbreak/_logs, since the pool is harmful instructions and a
# traceback can carry prompt text). Usage (positional only — NEVER --export=):
#   sbatch -J dose-<target> --qos=<cluster high QoS> --output=<shared>/exp/jailbreak/_logs/%x_%j.out \
#     scripts/sdfbase/full22/strength/dose_capture.sbatch <sha> <target> [capture|dry-run] [extra dose_capture.py args]
# The `dry-run` mode needs no GPU: submit it with --gres= unset and -p general,overflow.
#
# ENVIRONMENT: the FIT stack (torch 2.11.0+cu128, transformers 5.14.1, mamba_ssm 2.3.2.post1,
# causal_conv1d 1.6.2.post1) — the stack every staged object was fit under. On cluster A it is the
# shared $SHARED/envs/sdfbase-fit. On cluster B (no nvcc on PATH, no shared fit venv) a node-local
# venv is built from the SAME requirements carried from cluster A ($SHARED/envs/strength-dose-kernels/
# requirements.txt) and the PREBUILT kernel packages copied out of cluster A's fit env (identical torch
# ABI, sha-verified at copy time) — a source build is never attempted here. Nemotron REFUSES to run
# unless the fused fast path imports (the pure-torch SSM path is ~60x slower and reads as a healthy
# job).
set -uo pipefail
SHA="${1:?sha}"; TARGET="${2:?target}"; MODE="${3:-capture}"
shift 3 2>/dev/null || shift $#
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
REPO_DIR="${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$SHA"
[ -r "$REPO_DIR/scripts/sdfbase/full22/common.sh" ] || { echo "FATAL: no checkout at $REPO_DIR (git -C <the repo mirror> worktree add --detach $REPO_DIR $SHA)" >&2; exit 1; }
source "$REPO_DIR/scripts/sdfbase/full22/common.sh"
RC=1; RESULT=INCOMPLETE
# The record is written FIRST and nothing is signalled: this job runs ONE foreground python and has no
# background children to reap, so the campaign's `kill -TERM -$$` group kill has nothing to kill here
# and the first four dry-runs came back FAILED 0:11 — correct artefacts, rc=0 logged, then the step died
# in teardown. A job whose work succeeded must not report FAILED: the watcher cannot tell that from a
# real failure, and neither can a later reader.
cleanup(){
  set +e
  local rec="dose-capture target=$TARGET mode=$MODE rc=$RC result=$RESULT out=${OUT:-unresolved}"
  if [ -n "${FIGS_DIR:-}" ]; then f22_result "$rec"; else echo "RESULT: $rec" >&2; fi
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
F22_LAUNCH_CELLS=()   # this job stands up no target: no steered cell to select a serving path for (the kind-aware default, 2026-09-15)
f22_init "$SHA"
DOSE_ROOT="$SHARED/exp/strength_dose"; OUT="$DOSE_ROOT/$TARGET"; mkdir -p "$OUT"

# --- fail-fast guards: every external input, BEFORE the environment or the model --------------------
POOL="$SHARED/exp/dose_reference_pool/pool_v1.jsonl"
POOL_SHA=d6ecd62bea7e0c075c311c8364785179760e6fba6dedabee8e4bf433f03b4099
[ -s "$POOL" ] || f22_fatal "no reference pool at $POOL"
got=$(sha256sum "$POOL" | cut -d' ' -f1); [ "$got" = "$POOL_SHA" ] || f22_fatal "pool sha $got != $POOL_SHA"
[ -d "$SHARED/exp/objects/alphasteer_f22/$TARGET/strength/cells" ] || f22_fatal "no AlphaSteer strength bank for $TARGET on $CLUSTER"
[ -d "$SHARED/exp/objects/sdfbase/full22/$TARGET/cells" ] || f22_fatal "no LoRA bank for $TARGET on $CLUSTER"
case "$MODE" in capture|dry-run) ;; *) f22_fatal "mode must be capture or dry-run, got '$MODE'";; esac
if [ "$MODE" = capture ]; then
  [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || f22_fatal "capture mode needs a GPU allocation (CUDA_VISIBLE_DEVICES unset)"
fi

# --- the environment ---------------------------------------------------------------------------------
dose_venv(){   # -> PY   (preference order: the cluster's own fit venv, this lane's shared copy, node-local)
  if [ -x "$SHARED/envs/sdfbase-fit/bin/python" ]; then
    PY="$SHARED/envs/sdfbase-fit/bin/python"; f22_log "env: the shared fit venv $SHARED/envs/sdfbase-fit"; return 0
  fi
  if [ -f "$SHARED/envs/strength-dose-fit/BUILD-COMPLETE" ] && [ -x "$SHARED/envs/strength-dose-fit/bin/python" ]; then
    PY="$SHARED/envs/strength-dose-fit/bin/python"; f22_log "env: this lane's shared venv $SHARED/envs/strength-dose-fit ($(cat "$SHARED/envs/strength-dose-fit/BUILD-COMPLETE"))"; return 0
  fi
  local ov="$SHARED/envs/strength-dose-kernels" venv="/home/$USER/venvs/strength-dose-fit" lock sent want sp
  [ -s "$ov/requirements.txt" ] || f22_fatal "no $ov/requirements.txt (the fit-env recipe carried from cluster A)"
  [ -f "$ov/site-packages/selective_scan_cuda.cpython-312-x86_64-linux-gnu.so" ] || f22_fatal "no kernel overlay under $ov/site-packages"
  mkdir -p "$(dirname "$venv")"; sent="$(dirname "$venv")/.strength-dose-fit.synced"; lock="$(dirname "$venv")/.strength-dose-fit.lock"
  want=$(cat "$ov/requirements.txt" "$ov"/site-packages/*.dist-info/RECORD | sha256sum | cut -c1-16)
  (
    flock -x 9 || exit 99
    if [ -f "$sent" ] && [ "$(cat "$sent" 2>/dev/null)" = "$want" ] && [ -x "$venv/bin/python" ]; then
      f22_log "env: node-local dose venv ready ($want)"; exit 0
    fi
    f22_log "env: building node-local dose venv at $venv (the fit stack + carried kernels)"
    rm -rf "$venv"
    uv venv --python 3.12 "$venv" || exit 1
    uv pip install --python "$venv/bin/python" -r "$ov/requirements.txt" \
      --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match || exit 1
    uv pip install --python "$venv/bin/python" triton==3.7.1 fla-core==0.5.2 flash-linear-attention==0.5.2 ninja==1.13.0 einops==0.8.2 || exit 1
    sp=$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') || exit 1
    cp -a "$ov/site-packages/." "$sp/" || exit 1
    echo "$want" > "$sent"
  ) 9>"$lock" || f22_fatal "dose venv build failed"
  PY="$venv/bin/python"
}
dose_venv
"$PY" - <<'PY' || f22_fatal "env guard: the fit stack does not import"
import torch, transformers, sys
assert torch.__version__.startswith("2.11.0"), torch.__version__
assert transformers.__version__ == "5.14.1", transformers.__version__
print("env OK", torch.__version__, transformers.__version__, sys.executable)
PY
if [ "$TARGET" = nemotron3nano4b ]; then
  "$PY" -c 'import mamba_ssm, causal_conv1d; print("kernels OK", mamba_ssm.__version__, causal_conv1d.__version__)' \
    || f22_fatal "Nemotron: mamba_ssm / causal_conv1d do not import — REFUSING the pure-torch SSM fallback"
fi
export PATH="$(dirname "$PY"):$PATH"

# --- run ----------------------------------------------------------------------------------------
ARGS=(--target "$TARGET" --out-dir "$OUT" --shared "$SHARED")
[ "$MODE" = dry-run ] && ARGS+=(--dry-run)
f22_log "dose_capture $TARGET mode=$MODE out=$OUT extra=[$*]"
RC=0
"$PY" "$REPO_DIR/scripts/sdfbase/full22/strength/dose_capture.py" "${ARGS[@]}" "$@" || RC=$?
case "$RC" in
  0) RESULT=OK ;;
  3) RESULT=GATE_FAILED ;;
  *) RESULT=FAILED ;;
esac
# a positive artefact, or the job did not do its work whatever the exit code says
if [ "$MODE" = dry-run ]; then [ -s "$OUT/dryrun_$TARGET.json" ] || { RESULT=NO_ARTIFACT; RC=1; }
else [ -s "$OUT/dose_$TARGET.json" ] || { RESULT=NO_ARTIFACT; RC=1; }; fi
f22_log "done rc=$RC result=$RESULT"
exit $RC
