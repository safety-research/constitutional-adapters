#!/bin/bash
#SBATCH --job-name=petri-steer-smoke
#SBATCH --partition=general
#SBATCH --qos=high                    # the standing default; 1 GPU of the 16-GPU/user high budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G                    # NEVER --mem=0
#SBATCH --time=0-03:00:00             # job 229597 ran the whole thing (twelve engine inits, all the fidelity batches, the full ladder) in 14m47s wall. Job 230089 (this configuration plus three since-removed 32-way probe engines) took 14m52s. 3h stays ~10x the expected time, and right-sizing it further would only help backfill, which is not binding at qos=high.
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# FIRST STEERED TOKENS through the vLLM lens serving path — the first
# execution of lens_worker's GPU-VERIFY list (SisteerWorkerExtension,
# RealWorkerView, the split-residual write, the lens_batch plan lattice
# + PrefillLedger) on the substrate the 173012 probe verified. Serves
# TWO REAL objects through the committed adapters:
#   static_r_matched      (matched CAA, 10 layers, resid_pre -> served
#                          one block earlier; deployed at lambda=0.05)
#   alphasteer_official   (prefill-only last-token matmul broadcast)
# and evaluates the committed fidelity gate (engine_determinism,
# scale0_noop, steering_bites per arm, batch_invariance,
# selection_honored). Every engine is built with
# max_num_batched_tokens = max_num_seqs * max_model_len so vLLM V1's
# always-on chunked prefill can never bind mid-prompt — alphasteer
# reads the LAST prompt token and broadcasts, which a chunked prefill
# cannot express (see the construction-site comment in
# scripts/petri_steered_smoke.py).
#
# EXACT_REPLAY IS DROPPED (the user, 2026-08-27) — THE GATE NOW
# CERTIFIES THE PRODUCTION CONFIGURATION, BI=0. The manifest pins
# serving_mode: interleaved (no VLLM_BATCH_INVARIANT composed into the
# launch), because BI=1 costs 2.97x and on H200/SM90 buys only
# SAME-SIZE composition invariance — never batch-size-1-versus-N —
# while an audit's live batch size fluctuates continuously. WHAT IS
# LOST, recorded in the manifest, the smoke record and this header:
# re-running the audit will NOT reproduce its transcripts bit-for-bit.
# What still gates, at BI=0: engine_determinism (repeat-determinism,
# the property the consuming lane actually depends on), scale0_noop,
# steering_bites per arm, selection_direct (the worker's own routing
# record), and every instrument failure. batch_invariance and
# selection_honored's output half are composition comparisons — the
# property that was given up — so at BI=0 their failures are WAIVED:
# measured, reported, tagged waived_failures, never a job failure.
# The gating phases run at VLLM_BATCH_INVARIANT=0; a BI=1 phase is kept
# as EVIDENCE ONLY (the dropped exact_replay configuration: the
# lens-engine BI-cost comparison and the composition-invariance
# reference). This inverts the pre-2026-08-27 roles, where
# bi1_certified gated and bi0_probe was evidence.
#
# A GATING FIDELITY PHASE AT THE PETRI GEOMETRY (petri-gate16k): the same five checks
# at the PETRI geometry the audit actually serves — max_model_len 16384
# (build_server_launch's floor: CONSTITUTION_SI alone is 5,872 tokens,
# target turns carry no max_tokens, the 15-turn budget projects ~13.4k
# median for constitution cells) x max_num_seqs 24, budget 393,216, at
# gpu_memory_utilization 0.90. max_num_seqs is a scheduler CEILING, not
# a KV grant — vLLM silently schedules fewer on a shortfall (job
# 227309's production server at 32 x 16384 resolved 486,944 KV tokens =
# 29.72x) — so the worker ASSERTS the resolved KV coverage off its own
# cache_config (kv_coverage_problem) and the phase GATES. The 227308-
# style 2048 certification is unchanged and still runs first; this
# phase ADDS the 16384-geometry certification, never replaces it.
#
# 24, NOT 32 — measured negative result AT BI=1 (jobs 229194 + 229390,
# two nodes): 32 x 16384 does not fit one H200 under no-chunking. Weights
# ~15 GiB + full-coverage KV 64.0 GiB + the warmup peak of a forward
# over the whole 524,288-token budget (incl. one 14.0 GiB MLP
# intermediate = 524,288 x 14,336 x 2 B) exceed the 139.81 GiB card,
# and utilization is zero-sum against it: 0.95 under-covered KV by
# 2,768 tokens (521,520 = 31.83x, the gate refused); 0.97 covered KV
# (544,432 = 33.23x) then OOMed by 0.48 GiB on the warmup tensor. At
# 24 x 16384: KV 48.0 GiB + MLP 10.5 GiB + weights ~15 GiB = ~25 GiB
# of true slack. RE-MEASURED AT BI=0 after exact_replay was dropped
# (job 230089, --petri32-probe-utils 0.90,0.95,0.97 — three non-gating
# engines bracketing the BI=1 failure points plus the certified
# headroom point): SAME WALL. 0.90 resolved KV 463,472 tokens (28.29x),
# 0.95 resolved 520,736 (31.78x — 3,552 short of the 524,288 budget,
# where BI=1 resolved 521,520: essentially identical), 0.97 could not
# init the engine. The limit is STRUCTURAL, not a batch-invariance
# kernel artifact — the question is closed at both BI settings, and
# the certified product stays 24 x 16,384 = 393,216 (resolved KV
# 574,064 = 35.04x at BI=0, job 230089). The probe flag defaults to
# skip and is no longer passed below; re-reach for it only on a stack
# change that plausibly moves the MEMORY profile.
#
# WHAT THIS RUN ADDS beyond the passing 173664 run. (1) and (2) are the
# reason to run it at all; (3) and (4) are unchanged from 173664 and
# neither touches the gate.
#
# 1. THE FIDELITY PHASE RUNS AT BOTH BI SETTINGS, and the BI=0 one
#    certifies (the production configuration; see the header block
#    above for what gates and what is waived). The BI=1 phase runs in
#    its own engine, one environment variable apart and nothing else
#    (same argv, same registry, same arms, same certified geometry,
#    same prompts):
#      bi0_certified  the gate, at the production configuration
#                     (serving_mode=interleaved, exact_replay dropped).
#                     Governs the exit status with the PETRI phase.
#      bi1_probe      EVIDENCE ONLY — the dropped exact_replay
#                     configuration. Writes no marker, changes no
#                     verdict, and CANNOT make this job exit nonzero.
#    Both land under `fidelity_phases` in the record with their
#    resolved VLLM_BATCH_INVARIANT, all five verdicts, the per-check
#    divergence diagnostics (counts AND first-divergence indices), and
#    the lens engine's own throughput legs at each setting — so the
#    2.97x is confirmed ON THE LENS ENGINE rather than carried over
#    from the plugin-disabled ladder. The top-level `fidelity` key is,
#    as always, the gating phase's report and never a probe's.
#
#    EXPECT AT BI=0, and read the record accordingly: engine_determinism
#    is the headline; scale0_noop and steering_bites are
#    same-composition comparisons and should hold; batch_invariance and
#    selection_honored compare one request across DIFFERENT
#    compositions, which is precisely what BI=1 was buying, so they
#    may well fail — AND THAT IS AN EXPECTED RESULT, NOT A REGRESSION;
#    in the gating phases such a failure is WAIVED (reported, tagged
#    `waived_failures`, with the reason attached in the record and in
#    this log) because composition invariance is not a property
#    production claims. The 229597 BI=0 probe passed all five checks
#    outright, so waivers may well never trigger. A failure of any
#    OTHER check at BI=0 gates as hard as ever.
#
# 2. SELECTION IS NOW OBSERVED, NOT ONLY INFERRED. selection_honored
#    used to be decided purely by comparing output token ids between a
#    request's mixed-batch and same-size-reference servings — an
#    inference that holds only while outputs are composition-stable,
#    i.e. only under BI=1. The worker now records which object it
#    actually resolved and bound per request
#    (lens_worker.SelectionRecord, read over sisteer_selection_record)
#    and the harness compares that against the intended selection:
#    `selection_direct`, reported ALONGSIDE the output comparison in
#    BOTH phases, never instead of it. Direct passes + output fails =
#    right objects, numerical movement (expected at BI=0); direct FAILS
#    = a real routing bug that outranks everything else. The worker
#    record ships its own coverage statement: it observes routing and
#    binding IDENTITY, NOT op-level mis-binding downstream of selection.
#
# 3. THE THROUGHPUT LADDER, one policy change per rung, so the 3.1x
#    that 173580 reported can be ATTRIBUTED instead of guessed at.
#    Rungs: graphs+cache+BI0 (stock) -> graphs+cache+BI1 -> eager+cache
#    -> eager+no-cache -> the lens engine. Each is a separate
#    plugin-disabled child. Measured over TWO request geometries: the
#    unchanged legacy suite (8 short distinct prompts x 4, 128 tokens)
#    as the no-reuse control, and a SHARED-PREFIX suite at the
#    consumer's own geometry — batch 8, 256 generated tokens, ~150-token
#    distinct suffixes, and a shared synthetic system preamble at ~6500
#    and ~90 tokens plus a matched no-preamble control. The preamble is
#    generated in the script from a fixed rule template (no data file,
#    no randomness) and its MEASURED token length is recorded per leg,
#    as are prefix-caching / CUDA-graph / batch-invariance state.
#    The shared-prefix suite needs max_model_len 8192, so it runs on a
#    SECOND scheduler geometry (8192 x max_num_seqs 8 = the same 65536
#    token budget). The fidelity gate runs ONLY on the certified
#    2048/32/65536 geometry, exactly as it did in 173580.
#
# 4. THE CROSS-PHASE CELL — the first execution of POSITIONS_RESPONSE on
#    hardware. static_r_matched's own bytes served a second time under
#    a resolved spec carrying positions='response' (d1851eb made the
#    served mode come from the spec rather than the artifact precisely
#    so this combination is expressible). Advisory: it is served in an
#    ENGINE OF ITS OWN, because a steering failure condemns a lens
#    worker, and it is kept out of the mixed batch, so it cannot touch
#    a fidelity verdict. Its ops are asserted on CPU to carry
#    positions='response', non-empty header_ids and phase='both' (a
#    prefill-only op would leave generation unsteered — the specific
#    silent failure it exists to exercise), and every prompt it will be
#    served is checked to end with the assistant header, since the mask
#    is located by matching those ids at prefill and refuses on
#    mismatch. alphasteer and cosine_gated_r are asserted, CPU-side, to
#    REFUSE the same override.
#
# MODEL: meta-llama/Llama-3.1-8B-Instruct @ 0e9e39f2 (cache-resident on
# the cluster's shared HF cache) — deliberately NOT the manifest's
# NousResearch mirror: byte-identical weights, different chat template
# (43 vs 18 prompt tokens on a plain user prompt), and both arms steer
# every prompt position (the arms' owner, 2026-08-21). The divergence
# is recorded as a model_repin note in the JSON record.
#
# Outcome discipline (the gate's D13/`256c551` pattern):
#   BOOTSTRAP failure (imports / GPU / engine init / missing object
#     file) -> smoke_unavailable.json + exit 0 — a launch blocker to
#     escalate, never a silent skip;
#   VALIDATION failure (an arm fails to load/adapt, a config assert, a
#     raised request step, ANY GATING fidelity failure — a silently-no-op
#     arm is the headline failure; BI=0-waived composition checks are
#     recorded, never gating) -> smoke_failed.json + exit NONZERO. The
#     BI=1 phase and the 32-way probes are deliberately OUTSIDE this:
#     they can fail every check they like without changing the exit
#     status;
#   success -> smoke_record.json (numbers and verdicts only; generated
#     text goes to THIS LOG for eyeballing, never into any JSON).
# Results land in $SHARED/petri/serving/steered_smoke/ (durable).
#
# Runs from the IMMUTABLE commit-addressed checkout
# $SISTEER_PETRI_REPO_ROOT/<sha>/ — the job verifies the checkout's
# HEAD equals the baked-in SHA and that no tracked file is modified, and
# refuses otherwise. NEVER pointed at a peer's pinned checkout.
#
# SUBMIT with `sbatch --parsable` and persist the id; stop with
# `scancel <job_id>` ONLY — no name-based cancel anywhere (M15: this is
# a shared cluster user; a name-scoped scancel can kill a peer's job).
# NO --export= anywhere (hard rule 1): env is set in the body.
# The submitting environment carries SISTEER_SHARED_ROOT,
# SISTEER_PETRI_REPO_ROOT and the two object keys,
# SISTEER_PETRI_SMOKE_STATIC_R_OBJECT and SISTEER_PETRI_SMOKE_ALPHASTEER_OBJECT;
# the body refuses by name without any of them.
#
# VLLM_PLUGINS is deliberately NOT set: vllm filters general plugins by
# ENTRY-POINT NAME and vllm-lens's entry point is named "activations",
# so VLLM_PLUGINS=vllm_lens would silently filter the plugin OUT. The
# smoke pops any inherited value and checks plugin provenance from the
# live engine (probe-verified trap).
#
# audit-exempt: out-of-lockfile-venv — this job runs from the
# requirements-hash-versioned vllm venv (torch cu129, capeval D20), NOT
# the project lockfile; the venv is built here once per requirements
# hash (user-approved install, 2026-08-20) inside a per-venv flock, and
# job steps use `uv run --no-sync` so nothing later can strip it. Same
# venv the 173012 probe certified.

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
# The two REAL objects (THE ARMS above) are the site's to name: the job environment carries the site config's two keys as
# written, and a $SHARED in a value is the smoke's to expand against SISTEER_SHARED_ROOT. No default: a forgotten object is an
# error here, before the venv bootstrap, never another object.
STATIC_R_OBJECT="${SISTEER_PETRI_SMOKE_STATIC_R_OBJECT:?set SISTEER_PETRI_SMOKE_STATIC_R_OBJECT, the static_r_matched object.pt this gate serves (a site config key, passed in the job environment)}"
ALPHASTEER_OBJECT="${SISTEER_PETRI_SMOKE_ALPHASTEER_OBJECT:?set SISTEER_PETRI_SMOKE_ALPHASTEER_OBJECT, the alphasteer_official object.pt this gate serves (a site config key, passed in the job environment)}"

source "$SHARED/scripts/hf_env.sh"     # shared copy — cluster /home is node-local
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# The exact commit this smoke certifies against, REQUIRED positionally
# ($1) — never --export= (hard rule 1), and deliberately NOT defaulted.
# A default is a live footgun here: old commit-addressed checkouts stay
# on shared storage, so a forgotten argument would silently run stale
# code rather than fail — and the SHA previously defaulted to gathers
# batch-of-1 references, which is exactly the methodology this run
# exists to correct. A forgotten argument must be an error.
PETRI_SMOKE_SHA="${1:?REQUIRED: the commit sha to certify. No default — a stale checkout would silently run old code}"
PETRI_REPO_DIR="${SISTEER_PETRI_REPO_ROOT:?set SISTEER_PETRI_REPO_ROOT, the root of the per-commit Petri checkouts (the front door exports it from the site config)}/$PETRI_SMOKE_SHA"

if [ ! -d "$PETRI_REPO_DIR/src/sisteer" ]; then
  echo "FATAL: no commit-addressed checkout at $PETRI_REPO_DIR — push the" \
       "exact commit over SSH into it first (launch.sh pattern); refusing" >&2
  exit 1
fi
cd "$PETRI_REPO_DIR"

# Immutability check: HEAD is exactly the pinned commit and no TRACKED
# file differs (this smoke script + this sbatch may ride along untracked,
# by design — their sha256s are recorded in the smoke's provenance JSON).
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
if [ "$HEAD_SHA" != "$PETRI_SMOKE_SHA" ]; then
  echo "FATAL: checkout HEAD $HEAD_SHA != pinned $PETRI_SMOKE_SHA; refusing" >&2
  exit 1
fi
if ! git diff --quiet HEAD -- 2>/dev/null; then
  echo "FATAL: tracked files modified in $PETRI_REPO_DIR — the smoke runs" \
       "only from an immutable checkout; refusing" >&2
  exit 1
fi
echo "[smoke] running from $PETRI_REPO_DIR @ $HEAD_SHA"

# --- the pinned serving venv (requirements-hash-versioned, node-local) --------
# Identical to petri_probe.sbatch: same naming scheme, same flock + content
# sentinel, so on any node the probe already visited this is a no-op skip.
# uv pip needs --python (uv pip ignores UV_PROJECT_ENVIRONMENT — job 114815)
# and --index-strategy unsafe-best-match (the pytorch extra index shadows
# names — job 116213); --exclude-newer matches the repo's tool.uv cutoff
# (the vllm direct-URL wheel is sha256-pinned, exempt by construction).
VLLM_REQS="$PETRI_REPO_DIR/configs/capeval/vllm-requirements.txt"
REQ_HASH=$(sha256sum "$VLLM_REQS" | cut -c1-8)
export UV_PROJECT_ENVIRONMENT="/home/$USER/venvs/sisteer-petri-vllm-$REQ_HASH"
export UV_CACHE_DIR="$SHARED/.cache/uv"
export UV_LINK_MODE=copy

LOCKDIR=$(dirname "$UV_PROJECT_ENVIRONMENT")
mkdir -p "$LOCKDIR"
SENTINEL="$LOCKDIR/.$(basename "$UV_PROJECT_ENVIRONMENT").vllm-bootstrap"
REQ_HASH_FULL=$(sha256sum "$VLLM_REQS" | cut -c1-16)
(
  flock -x 9 || exit 99
  if [ -f "$SENTINEL" ] && [ "$(cat "$SENTINEL")" = "$REQ_HASH_FULL" ]; then
    echo "[smoke] serving venv already bootstrapped @ ${REQ_HASH_FULL:0:8} — skipping (sentinel)"
  else
    if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
      # python >= 3.12: vllm-lens requires it (uv fetches a managed
      # cpython onto node-local disk if the node lacks one).
      uv venv --python 3.12 "$UV_PROJECT_ENVIRONMENT" || exit 1
    fi
    if uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
         --index-strategy unsafe-best-match \
         --exclude-newer 2026-08-03T00:00:00Z -r "$VLLM_REQS"; then
      echo "$REQ_HASH_FULL" > "$SENTINEL"
    else
      exit 1
    fi
  fi
) 9>"$LOCKDIR/.$(basename "$UV_PROJECT_ENVIRONMENT").uv-sync.lock"
RC=$?
if [ "$RC" -ne 0 ]; then
  echo "FATAL: serving-venv bootstrap failed (rc=$RC) — see" \
       "configs/capeval/vllm-requirements.txt (D20); NOT continuing." >&2
  exit "$RC"
fi

PETRI_RUN_DIR="${PETRI_RUN_DIR:-$SHARED/petri/serving}"
SMOKE_OUT_DIR="$PETRI_RUN_DIR/steered_smoke"
mkdir -p "$SMOKE_OUT_DIR"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

# The smoke script owns the outcome discipline (module docstring):
# smoke_record.json / smoke_failed.json (+ exit nonzero) /
# smoke_unavailable.json (+ exit 0), always alongside smoke_evidence.json.
srun --ntasks=1 uv run --no-sync python scripts/petri_steered_smoke.py smoke \
  --manifest configs/capeval/manifests/petri_twosi.yaml \
  --out-dir "$SMOKE_OUT_DIR" \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --static-r-object "$STATIC_R_OBJECT" \
  --alphasteer-object "$ALPHASTEER_OBJECT" \
  --static-r-scale 0.05 \
  --response-arm on \
  --fidelity-batch-invariant 0 \
  --fidelity-bi1-probe on
