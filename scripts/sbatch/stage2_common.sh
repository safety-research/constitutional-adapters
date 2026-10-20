# Shared scaffolding for every stage-2 sbatch script (sourced, not run).
#
# Merges the repo's sbatch conventions (hf_env.sh from the SHARED copy,
# offline W&B + sync, uv env on node-local disk, srun-wrapped workload)
# with the cluster preemption-tolerant template:
#   * restartable  -- every stage-2 driver resumes from artifacts /
#                     per-item *.partial.jsonl, so preemption loses <=1 item;
#   * self-escalating -- a job killed by SIGTERM resubmits ITSELF (only its
#                     own array index) at ESCALATE_QOS (the cluster's
#                     non-preemptible QoS, which the job carries: never the
#                     QoS it was submitted at; none turns it off) with a
#                     positional attempt counter. NEVER via --export (hard
#                     rule: --export crashes slurmd on this cluster).
#
# Stopping for good: ~/.claude/skills/gpu-avail/scripts/stop-run.sh <job-name>
# -- it writes $SHARED/exp/training/<job-name>/STOP (checked here before any
# resubmission), waits for propagation, then scancels by name. A bare scancel
# looks identical to a preemption and WILL escalate.
#
# Usage in a job script (after the #SBATCH block). The job's environment
# carries the site's cluster values SISTEER_SHARED_ROOT and
# SISTEER_PROJECT_ROOT (the front door passes them; a hand submission
# exports them in the submitting shell) and ESCALATE_QOS, the cluster's
# non-preemptible QoS a preempted job resubmits at (or none); s2_init
# reads them:
#   set -uo pipefail
#   source "$SISTEER_PROJECT_ROOT/repo/scripts/sbatch/stage2_common.sh"
#   s2_init "$@"            # positional: [attempt] [max_attempts]
#   s2_run uv run --no-sync python scripts/stage2_run.py ... ; exit $?
#
# SUBMITTING (mandatory): every submission must carry the submit-time
# SHA pin naming the commit the submitter validated — s2_init FATALs without
# it (fail-closed; it is what makes a production-clone re-sync safe against
# jobs already in the queue):
#   sbatch --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) \
#     <script>.sbatch [attempt max_attempts]
#
# Optional isolation knobs, set in the job BODY before s2_init (never via
# sbatch --export — hard rule 1). Production defaults are pinned by tests:
#   S2_REPO_ROOT  (default $SHARED/<clone root>)     — clone root: repo at
#                 $S2_REPO_ROOT/repo, guard file at $S2_REPO_ROOT/EXPECTED_SHA
#   S2_REPO_DIR   (default $S2_REPO_ROOT/repo)       — the checkout itself
#                 (guard at ${S2_REPO_DIR%/repo}/EXPECTED_SHA)
#   S2_UV_ENV     (default /home/$USER/venvs/<clone root's name>) — node-local
#                 venv, named after the checkout root
#   S2_UV_SYNC_ARGS (default empty) — extra `uv sync` args, word-split
#                 (capeval; e.g. "--python 3.12 --group capeval")
#
# Checkout seams (optional, BODY-set before s2_init — never sbatch --export):
#   S2_ROOT      the checkout ROOT (primary). Defaults to the shared clone,
#                $SHARED/<clone root> (byte-identical behavior for every
#                existing script). A campaign that must not repoint the
#                shared checkout sets its own, e.g.
#                  S2_ROOT="$SHARED/<campaign clone>"
#                  source "$S2_ROOT/repo/scripts/sbatch/stage2_common.sh"
#                Each root carries its OWN repo clone, EXPECTED_SHA, wandb
#                dir, and a root-named node-local uv env — two campaigns at
#                different commits can never swap a live venv or run under
#                each other's sha gate.
#   S2_REPO_DIR  the exact repo dir (derives from S2_ROOT; the capeval jobs
#                set it directly to a checkout NOT named
#                `repo`, whose EXPECTED_SHA then lives INSIDE it).
#   S2_UV_ENV / S2_UV_SYNC_ARGS / S2_UV_EXTRAS  venv + uv-sync seams
#                (defaults byte-identical when unset).
#
# ($0-relative sourcing is impossible: sbatch runs a temp copy under
# /var/spool/slurmd/, so this file is addressed via the shared repo clone.)
# shellcheck shell=bash

# Canonical arm enumeration for array-indexed job scripts. ORDER IS
# LOAD-BEARING: an array index -> cell decoding depends on it. It mirrors the
# methods.py ARMS registry order and configs/stage2_llama8b.yaml's arms list
# (arms 0-11 the 12 plan arms, 12-13 the divergence-ablation arms, 14 the
# Jailbreak Antidote arm, then alphasteer, lofit, the three alphakl rungs and
# did/adasteer). The stage-2 IFEval-type SI arrays that used to sit beside it
# were removed with those SIs.
S2_ARMS=(vector vector_all lora_r1 lora_r1_all token_bos token_faithful
         psr psr_kl psr_kl_all loreft reft_r1 bidpo
         vector_rkl vector_js antidote
         alphasteer lofit
         alphakl_null alphakl_soft alphakl_fisher
         did adasteer)

s2_log() { echo "[$(date -Is)] [attempt ${ATTEMPT:-1}/${MAX_ATTEMPTS:-5}] $*"; }

# Results-registry native emission (additive, never fatal). The job announces its own
# lifecycle from ITS OWN environment — job id, node, GPUs, the sha pin — through the stdlib-only
# emitter (`python3 -S`, importable before `uv sync`), and finds its registry identity in the
# planned-run file the submit wrapper stored at $SHARED/registry/pending/<jobid>.json. A job that
# was submitted without the wrapper attaches to nothing here (one notice) — the harness store
# registers it late from its resolved config. Nothing here rides the sbatch export flag (hard rule 1); the
# spool/push variables come from the SHARED registry_env.sh sourced in the body. Every step is
# `|| true`: a failed record must never fail a finished job.
s2_registry_state() {   # $1 = running|generated|failed|cancelled, [$2 = scheduler state], [$3 = final]
  [ -r "$SHARED/scripts/registry/registry_env.sh" ] && source "$SHARED/scripts/registry/registry_env.sh"
  [ -n "${SISTEER_REGISTRY_SPOOL:-}" ] || return 0
  local pend="$SHARED/registry/pending/${SLURM_JOB_ID:-none}.json"
  if [ -f "$pend" ]; then export SISTEER_REGISTRY_RUN_IDENTITY="$pend"; fi
  # GPU count / type are captured by the emitter from THIS job's environment (SLURM_GPUS_ON_NODE / SLURM_JOB_GPUS /
  # CUDA_VISIBLE_DEVICES, nvidia-smi, sacct AllocTRES as the fallback) and named in the running state's note.
  PYTHONPATH="$S2_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" python3 -S -m sisteer.registry.emit_cli state --state "$1" \
    --pending-dir "$SHARED/registry/pending" --job "${SLURM_JOB_ID:-}" --node "${SLURMD_NODENAME:-$(hostname)}" \
    ${2:+--scheduler-state "$2"} ${3:+--final} \
    2>&1 | sed 's/^/[registry] /' || true
}

# Results-registry SELF-REGISTRATION (2026-09-17: emission is the default; no submit wrapper). A stage2 body
# has no ONE resolved config the shell can read (every job names its own driver + config), so the body DECLARES
# what it runs before s2_init, body-level (never the sbatch export flag, hard rule 1):
#   S2_REGISTRY_CONFIG=<config yaml with explicit_cells>   S2_REGISTRY_MODEL_KEY=<legacy_spellings model key>
#   S2_REGISTRY_EVALS="<eval ids, space/comma separated>"  [S2_REGISTRY_SERVING_PATH=hf|lora_native|lens]
#   [S2_REGISTRY_CAMPAIGN=two_si] [S2_REGISTRY_SELECTION=main] [S2_REGISTRY_ROSTER=<roster yaml>]
# With all three set, `python3 -m sisteer.registry.emit_cli register --from-config` writes one `run` per declared
# eval x object + the pending file (a `runs` list with routing facts) and exports SISTEER_REGISTRY_RUN_IDENTITY, so the
# job's CellStores record by default and s2_finish's terminal state reaches every run. Nothing declared → one notice,
# nothing registered (the store hooks then say so once per store). A pending file a submit wrapper wrote is kept.
s2_registry_register() {
  [ -r "$SHARED/scripts/registry/registry_env.sh" ] && source "$SHARED/scripts/registry/registry_env.sh"
  [ -n "${SISTEER_REGISTRY_SPOOL:-}" ] || return 0
  local job="${SLURM_JOB_ID:-}" pend evals_csv
  [ -n "$job" ] || return 0
  pend="$SHARED/registry/pending/$job.json"
  if [ -f "$pend" ]; then export SISTEER_REGISTRY_RUN_IDENTITY="$pend"; return 0; fi
  if [ -z "${S2_REGISTRY_CONFIG:-}" ] || [ -z "${S2_REGISTRY_MODEL_KEY:-}" ] || [ -z "${S2_REGISTRY_EVALS:-}" ]; then
    echo "[registry] register: this job declares no S2_REGISTRY_CONFIG / S2_REGISTRY_MODEL_KEY / S2_REGISTRY_EVALS; nothing registered (its stores stay unrecorded)" >&2
    return 0
  fi
  evals_csv="$(printf '%s' "$S2_REGISTRY_EVALS" | tr ' ' ',' | tr -s ',' | sed 's/^,//; s/,$//')"
  PYTHONPATH="$S2_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m sisteer.registry.emit_cli register --from-config "$S2_REGISTRY_CONFIG" \
    --model-key "$S2_REGISTRY_MODEL_KEY" --roster-selection "${S2_REGISTRY_SELECTION:-main}" --evals "$evals_csv" \
    --serving-path "${S2_REGISTRY_SERVING_PATH:-hf}" --campaign "${S2_REGISTRY_CAMPAIGN:-two_si}" --job-id "$job" \
    ${SLURM_JOB_NAME:+--job-name "$SLURM_JOB_NAME"} --pending-dir "$SHARED/registry/pending" \
    ${SUBMIT_SHA:+--code-sha "$SUBMIT_SHA"} ${S2_REGISTRY_ROSTER:+--roster "$S2_REGISTRY_ROSTER"} \
    >/dev/null 2> >(sed 's/^/[registry] /' >&2) || true
  [ -f "$pend" ] && export SISTEER_REGISTRY_RUN_IDENTITY="$pend"
  return 0
}

s2_init() {
  # Attempt bookkeeping arrives positionally on resubmission (hard rule:
  # never sbatch --export on this cluster).
  ATTEMPT="${1:-1}"
  MAX_ATTEMPTS="${2:-5}"

  # The cluster's shared root is the site's: the front door passes the
  # cluster block's root to every job as SISTEER_SHARED_ROOT (a hand
  # submission exports it in the submitting shell). The QoS a preempted
  # job resubmits at is ESCALATE_QOS, the job's own setting: the cluster's
  # non-preemptible QoS, never the QoS the job was submitted at (a job
  # submitted at a preemptible QoS would resubmit into the next
  # preemption), and QoS names differ between clusters, so nothing derives
  # it. Both are read here, never guessed from which storage paths exist,
  # and a missing one stops the job by name before any work.
  SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the cluster shared storage root (the front door exports it from the site config)}"
  ESCALATE_QOS="${ESCALATE_QOS:?set ESCALATE_QOS, the cluster non-preemptible QoS a preempted job resubmits at (not the QoS it was submitted at), or ESCALATE_QOS=none}"   # "none" disables escalation
  # First log line on purpose (review A-P2-2): any preemption-behavior
  # question should be answerable from the log head.
  s2_log "escalation qos: $ESCALATE_QOS"
  # The UNIFIED checkout seam (one merge of THREE independent
  # parameterizations — the capeval checkout, the capeval controls clone,
  # and the torch-split proof job's S2_REPO_ROOT/S2_UV_ENV extension points).
  # S2_ROOT is PRIMARY (see the header note): body-set by a campaign that
  # owns a separate clone; it derives the repo dir, the EXPECTED_SHA gate
  # location, the wandb dir, and the node-local venv name. S2_REPO_ROOT is
  # honored as the SAME seam under the torch-split proof job's name (it
  # points it at an isolated clone root) and aliases to
  # S2_ROOT below. S2_REPO_DIR DERIVES from it and stays its own override
  # for a checkout NOT named `repo` (capeval_common.sh points it at the
  # capeval clone's checkout; a sibling campaign wrapper may set
  # S2_REPO_DIR, S2_EXPECTED_SHA_FILE and S2_UV_ENV in this same shell). The SHA gate
  # file lives at <parent>/EXPECTED_SHA for a checkout named `repo` (the
  # historical layout) and at <checkout>/EXPECTED_SHA otherwise — the
  # `${S2_REPO_DIR%/repo}` below yields exactly today's path when unset.
  # ALL defaults are byte-identical to the historical shared paths.
  S2_ROOT="${S2_ROOT:-${S2_REPO_ROOT:-${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}}}"
  S2_REPO_ROOT="${S2_REPO_ROOT:-$S2_ROOT}"
  S2_REPO_DIR="${S2_REPO_DIR:-$S2_ROOT/repo}"
  # Node-local venv seam (a PRIVATE plain-shell seam on purpose — keying the
  # default off UV_PROJECT_ENVIRONMENT itself would make it env-inheritable:
  # an exported UV_PROJECT_ENVIRONMENT lingering in a submission shell must
  # never silently redirect a campaign's venv). Default is root-NAMED: the
  # venv takes the basename of S2_ROOT, so two clones with different
  # S2_ROOTs (hence possibly different lockfiles) can never `uv sync` the
  # SAME venv out from under each other's running jobs, and the default root
  # keeps the venv name it has always had.
  S2_UV_ENV="${S2_UV_ENV:-/home/$USER/venvs/$(basename "$S2_ROOT")}"
  RUN_NAME="${SLURM_JOB_NAME:-stage2}"
  STOP_DIR="$SHARED/exp/training/$RUN_NAME"   # stop-run.sh sentinel location
  export FIGS_DIR="$SHARED/exp/figs/$RUN_NAME"
  mkdir -p "$STOP_DIR" "$FIGS_DIR"

  # Mandatory submit-time SHA pin: the job must carry
  # --comment=sha:<40-hex> naming the commit its submitter validated, and
  # repo HEAD must equal it AT JOB START. This is what the EXPECTED_SHA file
  # below cannot give: a job submitted before a clone re-sync but started
  # after it sees new-HEAD == new-EXPECTED_SHA (both comparands move
  # together) and would silently run code its submitter never validated.
  # Fail-closed: absent, malformed, ambiguous, or unreadable (scontrol
  # failing) all FATAL with the fix printed. --comment is inert job-level
  # metadata (array tasks inherit it); hard rule 1 (--export) is untouched.
  local scontrol_out pin_lines pin_count pin_job_id
  # Array-task scoping: the array task that inherits the ARRAY-MASTER
  # JobId sees 'scontrol show job $SLURM_JOB_ID' return one record per
  # VISIBLE sibling, each carrying the same inherited submission comment
  # (observed in the wild: 139589_20 parsed 3 identical Comment= values and
  # the exactly-one assertion below refused a validly-pinned job; sibling
  # visibility varies over time, so 18/19 had passed earlier). The per-task
  # form <array_job>_<task> returns exactly one record (verified on the
  # login host against RUNNING 139589_18/_19), so query THAT when the array
  # vars are set; a non-array job keeps today's $SLURM_JOB_ID query exactly.
  pin_job_id="$SLURM_JOB_ID"
  if [ -n "${SLURM_ARRAY_JOB_ID:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    pin_job_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  fi
  if ! scontrol_out="$(scontrol show job "$pin_job_id" 2>/dev/null)"; then
    s2_log "FATAL: cannot read this job's submit-time SHA pin ('scontrol show job $pin_job_id' failed; D12). Resubmit with:"
    s2_log "  sbatch --comment=sha:\$(git -C $S2_REPO_DIR rev-parse HEAD) <script>.sbatch"
    exit 1
  fi
  # Trailing-space-safe parse: scontrol on this cluster renders a set
  # comment with a TRAILING SPACE, so the capture takes the non-whitespace
  # token only — a greedy \(.*\)$ would swallow the space and reject every
  # valid pin. Exactly ONE extracted value is required.
  pin_lines="$(printf '%s\n' "$scontrol_out" \
    | sed -n 's/^ *Comment=\([^[:space:]]*\)[[:space:]]*$/\1/p')"
  if [ -z "$pin_lines" ]; then pin_count=0; else
    pin_count="$(printf '%s\n' "$pin_lines" | wc -l)"
  fi
  # Second belt: should multiple records still surface, ALL-IDENTICAL
  # values are the one inherited submission comment repeated per sibling —
  # collapse them to that one pin. Only genuinely DIFFERING values (or none
  # at all) remain ambiguous and keep the FATAL below.
  if [ "$pin_count" -gt 1 ]; then
    local pin_unique
    pin_unique="$(printf '%s\n' "$pin_lines" | sort -u)"
    if [ "$(printf '%s\n' "$pin_unique" | wc -l)" -eq 1 ]; then
      pin_lines="$pin_unique"
      pin_count=1
    fi
  fi
  if [ "$pin_count" -ne 1 ]; then
    s2_log "FATAL: submit-time SHA pin missing or ambiguous ($pin_count Comment= values parsed; D12). Every submission must name the commit it validated:"
    s2_log "  sbatch --comment=sha:\$(git -C $S2_REPO_DIR rev-parse HEAD) <script>.sbatch"
    exit 1
  fi
  if ! printf '%s' "$pin_lines" | grep -Eq '^sha:[0-9a-f]{40}$'; then
    s2_log "FATAL: malformed submit-time SHA pin '$pin_lines' (want sha:<40-hex>; D12). Resubmit with:"
    s2_log "  sbatch --comment=sha:\$(git -C $S2_REPO_DIR rev-parse HEAD) <script>.sbatch"
    exit 1
  fi
  SUBMIT_SHA="${pin_lines#sha:}"

  # Clone-SHA precondition, enforced: the cluster clone must sit at the
  # commit the submitter validated. The sync step records that commit:
  #   git -C "$S2_REPO_DIR" fetch && \
  #     git -C "$S2_REPO_DIR" checkout <sha> && \
  #     echo <sha> > "${S2_REPO_DIR%/repo}/EXPECTED_SHA"
  # A stale clone fails LOUDLY here instead of running old code that appends
  # into the same resume files. Kept as an independent second net beside the
  # submit-time pin above: HEAD is what the last sync RECORDED.
  # S2_EXPECTED_SHA_FILE overrides the derived location.
  local expected_file="${S2_EXPECTED_SHA_FILE:-${S2_REPO_DIR%/repo}/EXPECTED_SHA}"
  local expected_sha actual_sha
  if [ ! -f "$expected_file" ]; then
    s2_log "FATAL: $expected_file missing -- record the synced commit there before submitting (D7)."
    exit 1
  fi
  expected_sha="$(tr -d '[:space:]' < "$expected_file")"
  actual_sha="$(git -C "$S2_REPO_DIR" rev-parse HEAD 2>/dev/null)"
  if [ -z "$actual_sha" ] || [ "$actual_sha" != "$expected_sha" ]; then
    s2_log "FATAL: stale clone -- repo HEAD '${actual_sha:-unreadable}' != EXPECTED_SHA '$expected_sha'. Re-sync $S2_REPO_DIR (D7)."
    exit 1
  fi
  # The submit-time pin must equal HEAD too — the check the file
  # guard cannot make when a re-sync moved both HEAD and EXPECTED_SHA.
  if [ "$actual_sha" != "$SUBMIT_SHA" ]; then
    s2_log "FATAL: repo HEAD '$actual_sha' != submit-time pin '$SUBMIT_SHA' (D12) -- the clone moved since this job was submitted. Re-validate and resubmit at the new sha."
    exit 1
  fi

  # $0 is a per-job temp copy under /var/spool/slurmd -- not resubmittable.
  SCRIPT_PATH="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null \
                  | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"

  # registry: the job registers ITSELF from its declared config (s2_registry_register, above), then
  # `running` from its own environment for every run it registered — AFTER the SHA gate and BEFORE `uv sync`
  # (stdlib-only emitter for the state; plain python3 + PyYAML for the registration)
  s2_registry_register || true
  s2_registry_state running "${SLURM_JOB_STATE:-RUNNING}" || true

  # Env per cluster rules: SHARED copy of hf_env.sh (cluster /home is
  # node-local; a ~ path would silently not exist on the compute node).
  source "$SHARED/scripts/hf_env.sh"
  # The INSTALLED hf_env.sh may still probe the bare mountpoint, so on a
  # cluster-B compute node (the stub mountpoint — see the cluster
  # detection above) it exports cluster A's HF paths. Trust the per-user
  # probe over the stub: when SHARED resolved to cluster B, re-point HF at
  # the same per-user cache hf_env.sh's own cluster-B branch names.
  if [ "$SHARED" = "/workspace/$USER" ] && [ "${HF_HOME:-}" != "/workspace/$USER/hf_cache" ]; then
    export HF_HOME="/workspace/$USER/hf_cache"
    export HF_TOKEN_PATH="/workspace/$USER/hf_token"
    export HUGGINGFACE_TOKEN_PATH="/workspace/$USER/hf_token"
  fi
  export HF_HUB_OFFLINE=1                # all stage-2 inputs cached/fetched Phase 0
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export WANDB_MODE=offline              # compute nodes can't reach api.wandb.ai
  export WANDB_DIR="$S2_ROOT/wandb"
  mkdir -p "$WANDB_DIR"
  # nltk data on SHARED storage: nltk>=3.9 resolves punkt ->
  # punkt_tab, and compute nodes cannot download it mid-run -- sentence
  # tokenization would die AT SCORING, after the whole generate pass.
  # Login-node prefetch (once):
  #   uv run python -m nltk.downloader -d "$SHARED/nltk_data" punkt_tab punkt
  export NLTK_DATA="$SHARED/nltk_data"
  # The node-local venv this job bootstraps into — S2_UV_ENV, defaulted
  # root-named at the top of s2_init (see the seam note there).
  export UV_PROJECT_ENVIRONMENT="$S2_UV_ENV"
  export UV_CACHE_DIR="$SHARED/.cache/uv"
  export UV_LINK_MODE=copy

  SRUN_PID=""
  _s2_handled=0
  trap s2_on_term SIGTERM SIGINT SIGQUIT
  trap s2_finish EXIT

  cd "$S2_REPO_DIR" || exit 1
  # Env init: `uv sync --frozen` or FAIL THE JOB. A compute job must never
  # re-resolve outside the lockfile -- an unfrozen resolve on a compute
  # node silently runs unpinned code. `--extra cu128` is the torch-split
  # GPU flavor (torch is no longer a base dep): a flavor-less sync would
  # install NO torch and the runtime assert below would kill the job —
  # loud, never the 61380 silent-CPU path. The sync runs under a PER-NODE
  # exclusive flock: the node-local venv is shared by every array task on
  # the node, and two concurrent `uv sync`s corrupt it.
  local uv_env_dir uv_rc
  uv_env_dir="$(dirname "$UV_PROJECT_ENVIRONMENT")"
  mkdir -p "$uv_env_dir"
  # Optional extras (space-separated), set by the CALLING job script in its
  # body before s2_init — never via sbatch --export (hard rule 1). One sets
  # it today: the alphasteer-full22 fit on the Mamba-hybrid target (its
  # config's `env_extras: mamba`); core jobs set nothing and sync cu128-only
  # A `longctx` sync needs the pinned flash-attn wheel from the
  # clone's gitignored wheels/ dir and fails loudly below when it is absent
  # — restore wheels/*.whl in the clone (fa_build.sbatch rebuilds it)
  # rather than resubmitting. Note uv sync
  # makes the env EXACTLY match: a core job's sync removes an extra's
  # packages from the shared node-local venv and the next sync that asks
  # for them re-adds them from cache — churn, not breakage.
  local uv_extra_flags=()
  local _extra
  for _extra in ${S2_UV_EXTRAS:-}; do
    uv_extra_flags+=(--extra "$_extra")
  done
  # The `mamba` extra is the one extra uv BUILDS FROM SOURCE: mamba-ssm and
  # causal-conv1d are pinned PyPI sdists (pyproject's [tool.uv.extra-build-
  # dependencies] / [tool.uv.extra-build-variables], replacing the
  # self-built path wheels), compiled against the lockfile's torch the FIRST
  # time a cluster syncs them and installed from uv's cache -- the SHARED
  # UV_CACHE_DIR exported above, which scripts/sbatch/mamba_build.sbatch
  # warms once per cluster -- on every later sync, so a job normally never
  # compiles. A cold cache compiles right here, inside the sync (1068 s at
  # MAX_JOBS=8 on an H200 node, job 1001860; ~10 gencode archs x 12 .cu),
  # and needs a CUDA *12* toolkit -- torch.utils.cpp_extension refuses
  # another major against the cu128 torch, and cluster B's /usr/local/cuda
  # is 13.0, so the unconditional /usr/local/cuda default this replaces made
  # the frozen sync die inside uv's build there.
  # scripts/sbatch/cuda_toolkit.sh selects and VALIDATES one (cluster A's
  # /usr/local/cuda-12.8, or the $SHARED/cuda-12.8 mamba_build.sbatch
  # installs on a cluster without one), exporting CUDA_HOME with its bin
  # first on PATH (compute nodes ship nvcc OFF PATH, 471132); a caller's own
  # CUDA_HOME is honoured when it validates and refused when it does not.
  # No usable toolkit is FATAL before the sync -- a cold cache would fail
  # minutes later with torch's mismatch error anyway, and failing here names
  # the remedy (the once-per-cluster warm-up). MAX_JOBS bounds the parallel
  # nvcc jobs (each is memory-heavy; 8 fits a 128G allocation); a caller's
  # own MAX_JOBS wins. All of it runs ONLY when the extra is requested, so
  # every other job's sync is byte-identical to before; a cache hit needs
  # none of it and costs nothing.
  case " ${S2_UV_EXTRAS:-} " in
    *" mamba "*)
      # shellcheck disable=SC1091
      source "$S2_REPO_DIR/scripts/sbatch/cuda_toolkit.sh"
      if ! s2_cuda12_select; then
        s2_log "FATAL: the mamba extra's kernels compile with a CUDA 12 toolkit (the locked torch is cu128; torch refuses another major) and none is usable on this node -- run scripts/sbatch/mamba_build.sbatch on THIS cluster first (through submit_mamba_build.sh: it installs CUDA 12.8 under $SHARED/cuda-12.8 when the system has none and warms $UV_CACHE_DIR), or name a CUDA 12 toolkit as CUDA_HOME in the job body"
        exit 1
      fi
      export MAX_JOBS="${MAX_JOBS:-$(( ${SLURM_CPUS_PER_TASK:-8} < 8 ? ${SLURM_CPUS_PER_TASK:-8} : 8 ))}"
      s2_log "mamba extra requested: CUDA_HOME=$CUDA_HOME (nvcc release $S2_CUDA_RELEASE) MAX_JOBS=$MAX_JOBS for the fused-kernel build (a cold $UV_CACHE_DIR compiles ~20-60 min inside this sync; mamba_build.sbatch warms it once per cluster)"
      ;;
  esac
  # Extension point: extra `uv sync` args, word-split
  # deliberately (e.g. S2_UV_SYNC_ARGS="--python 3.12 --group capeval").
  # Empty/unset expands to nothing — the command line is byte-identical to
  # the pre-extension hardcoded call. The flock file is named after the
  # venv so distinct venvs on one node bootstrap independently (same name
  # as before for the default venv; the per-venv lock for capeval).
  (
    flock -x 9 || exit 99
    # --extra cu128 is MANDATORY here (torch-cpu-split): torch lives only in
    # the flavor extras, so a flavor-less sync installs no torch at all and
    # the runtime assert below kills the job. S2_UV_SYNC_ARGS appends on top.
    # shellcheck disable=SC2086
    uv sync --frozen --extra cu128 "${uv_extra_flags[@]}" ${S2_UV_SYNC_ARGS:-}
  ) 9>"$uv_env_dir/.$(basename "$UV_PROJECT_ENVIRONMENT").uv-sync.lock"
  uv_rc=$?
  if [ "$uv_rc" -ne 0 ]; then
    s2_log "FATAL: 'uv sync --frozen' failed (rc=$uv_rc) -- the node env cannot be built from the lockfile. NOT falling back to an unfrozen resolve (D22#12); fix uv.lock / the uv cache and resubmit."
    exit 1
  fi

  # Runtime GPU-flavor assert (torch-cpu-split): the job dies HERE,
  # under srun on the allocation, if the env it just synced is anything but
  # the cu128 GPU torch — CUDA unavailable, no device, or a wrong-flavor
  # wheel all FATAL before the workload starts. Never silent CPU (61380).
  s2_assert_cu128
}

s2_assert_cu128() {
  # S2_CPU_ONLY=1 (set in the JOB BODY before s2_init, never via
  # sbatch --export — hard rule 1) is for jobs submitted WITHOUT --gres on
  # purpose (CPU-only analysis/join wrappers): the frozen cu128
  # sync above still runs (venv parity — the node-local venv is shared
  # with GPU tasks), but a no-GPU allocation can never pass the CUDA
  # assert, so the guard degrades to wheel-identity only — still loud on
  # a wrong-flavor or missing torch, which is the 61380 failure class a
  # CPU job can actually hit.
  if [ "${S2_CPU_ONLY:-0}" = "1" ]; then
    srun --ntasks=1 uv run --no-sync python -c '
import torch
assert torch.__version__ == "2.11.0+cu128", f"torch {torch.__version__} != 2.11.0+cu128"
print("s2_assert_cu128 OK (S2_CPU_ONLY=1, no CUDA assert):", torch.__version__)
'
  else
    srun --ntasks=1 uv run --no-sync python -c '
import torch
assert torch.cuda.is_available(), "CUDA not available in this env"
assert torch.cuda.device_count() >= 1, "no CUDA device visible"
assert torch.__version__ == "2.11.0+cu128", f"torch {torch.__version__} != 2.11.0+cu128"
print("s2_assert_cu128 OK:", torch.__version__, torch.cuda.get_device_name(0))
'
  fi
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    s2_log "FATAL: cu128 runtime assert failed (rc=$rc) -- the env is not the GPU flavor (or CUDA cannot init); NOT running the workload (61380 guard)."
    exit 1
  fi
}

s2_finish() {
  # Runs on EVERY exit path: write the one-line RESULT sentinel (state,
  # exit code, elapsed seconds, job id, node) next to the figures, then
  # push both to the grading host's mirror — the outcome file for an
  # unattended run, readable with no session watching. EVERY step is
  # best-effort (2>/dev/null FIRST so even a failed redirect stays silent,
  # || true, and no `exit` in this trap): a failed sentinel write or mirror
  # must never change a finished job's exit code — bash preserves the
  # pre-trap status as long as the trap does not call `exit`.
  local rc=$?
  local state=COMPLETED
  if [ "${_s2_handled:-0}" -eq 1 ]; then
    state=PREEMPTED                   # the SIGTERM path (preemption/cancel)
  elif [ "$rc" -ne 0 ]; then
    state=FAILED
  fi
  if [ -n "${FIGS_DIR:-}" ]; then
    echo "$state exit=$rc elapsed_s=$SECONDS job=${SLURM_JOB_ID:-unknown} node=${SLURMD_NODENAME:-$(hostname)}" \
      2>/dev/null > "$FIGS_DIR/RESULT" || true
  fi
  #
  # W&B is NOT synced here: compute nodes cannot reach api.wandb.ai, so an
  # in-trap `wandb sync` either hangs into the preemption grace window or
  # fails silently every time. Runs stay offline in $WANDB_DIR; sync them
  # from a LOGIN node as a post-step:
  #   wandb sync --sync-all "$S2_ROOT/wandb"
  "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" 2>/dev/null || true
  # registry: the terminal state from the trap outcome + one FINAL push past the debounce.
  # COMPLETED moves no lifecycle state (the store hook filed `generated` itself when results landed);
  # FAILED → failed, PREEMPTED (the SIGTERM path: preemption or cancel) → cancelled, scheduler_state kept.
  case "$state" in
    FAILED)    s2_registry_state failed "$state" final 2>/dev/null || true ;;
    PREEMPTED) s2_registry_state cancelled "$state" final 2>/dev/null || true ;;
    *)         [ -n "${SISTEER_REGISTRY_SPOOL:-}" ] && PYTHONPATH="$S2_REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" python3 -S -m sisteer.registry.emit_cli push \
                 --pending-dir "$SHARED/registry/pending" --job "${SLURM_JOB_ID:-}" --final 2>/dev/null || true ;;
  esac
}

s2_escalate() {
  if [ "$ESCALATE_QOS" = "none" ]; then
    s2_log "escalation disabled (ESCALATE_QOS=none)."; return
  fi
  if [ -e "$STOP_DIR/STOP" ]; then
    s2_log "STOP sentinel present -- deliberate stop, not resubmitting."; return
  fi
  if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then
    s2_log "attempt cap $MAX_ATTEMPTS reached -- not resubmitting."; return
  fi
  if [ -z "$SCRIPT_PATH" ] || [ ! -f "$SCRIPT_PATH" ]; then
    s2_log "cannot resolve script path ('$SCRIPT_PATH') -- not resubmitting."; return
  fi
  local array_flag=()
  # Resubmit ONLY this task's index, not the whole array.
  [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && array_flag=(--array="$SLURM_ARRAY_TASK_ID")
  # Carry the job's ACTUAL walltime forward: a submit-time --time override
  # is not in the script header, so rebuilding the sbatch command from the script
  # alone would silently revert attempts 2+ to the header default.
  local time_flag=() time_limit
  # Explicit conditional, not a bare assignment: under a caller's
  # `set -e -o pipefail` a transiently failing scontrol would kill the
  # handler mid-escalation instead of taking the warn-and-resubmit path.
  if ! time_limit="$(scontrol show job "${SLURM_JOB_ID:-}" 2>/dev/null \
                      | sed -n 's/.*TimeLimit=\([^ ]*\).*/\1/p' | head -1)"; then
    time_limit=""
  fi
  if [[ "$time_limit" =~ ^[0-9][0-9:-]*$ ]]; then
    time_flag=(--time="$time_limit")
  else
    s2_log "WARNING: cannot parse TimeLimit ('${time_limit:-empty}') from scontrol -- resubmitting with the script header's default --time"
  fi
  # Carry the job's ACTUAL log path forward the same way. Most headers
  # hardcode an --output under cluster A's shared root, which does not
  # exist on cluster B — and Slurm never creates log parents, so a resubmit
  # that reverts to such a header path NEVER STARTS. scontrol reports the
  # RESOLVED predecessor path, whose parent exists because the predecessor
  # wrote there; --open-mode=append makes the successor append to the same
  # file (the header convention: retries append). Unparseable -> no flag,
  # i.e. the header default, exactly as before.
  local out_flag=() stdout_path
  stdout_path="$(scontrol show job "${SLURM_JOB_ID:-}" 2>/dev/null \
                  | sed -n 's/^ *StdOut=\([^ ]*\).*/\1/p' | head -1 || true)"
  if [ -n "$stdout_path" ]; then
    out_flag=(--output="$stdout_path")
  fi
  # Forward the submit-time SHA pin: the resubmit pins the ORIGINAL
  # validated sha, so an escalation that crosses a production clone re-sync
  # FATALs loudly at s2_init instead of silently continuing on new code
  # (the operator then resubmits deliberately at the new sha).
  local comment_flag=()
  [ -n "${SUBMIT_SHA:-}" ] && comment_flag=(--comment="sha:$SUBMIT_SHA")
  s2_log "resubmitting at qos=$ESCALATE_QOS as attempt $((ATTEMPT + 1))"
  sbatch --qos="$ESCALATE_QOS" --job-name="$RUN_NAME" "${array_flag[@]}" \
         "${time_flag[@]}" "${out_flag[@]}" "${comment_flag[@]}" \
         "$SCRIPT_PATH" "$((ATTEMPT + 1))" "$MAX_ATTEMPTS" \
    || s2_log "RESUBMIT FAILED -- run will not continue automatically"
}

s2_on_term() {
  [ "$_s2_handled" -eq 1 ] && return
  _s2_handled=1
  s2_log "SIGTERM (preemption or cancel); resubmitting FIRST, then draining."
  # `|| true`: the workload may already be dead/reaped (TERM between s2_run
  # calls), and under a caller's `set -e` a failing kill — the TAIL of this
  # && list — would abort the handler BEFORE s2_escalate runs.
  { [ -n "$SRUN_PID" ] && kill -TERM "$SRUN_PID" 2>/dev/null; } || true
  # Resubmit IMMEDIATELY after signalling the workload: the old
  # order -- an UNBOUNDED wait, then escalate -- could burn the whole
  # ~120s grace window on a slow drain and lose the requeue entirely.
  # s2_escalate itself still checks the STOP sentinel and the attempt cap.
  s2_escalate
  # Bounded drain: give the workload time to finish its in-flight item,
  # but never past S2_DRAIN_TIMEOUT (default 90s of the ~120s grace).
  local _deadline=$(( SECONDS + ${S2_DRAIN_TIMEOUT:-90} ))
  while [ -n "$SRUN_PID" ] && kill -0 "$SRUN_PID" 2>/dev/null; do
    if [ "$SECONDS" -ge "$_deadline" ]; then
      s2_log "drain timeout (${S2_DRAIN_TIMEOUT:-90}s) -- exiting with the workload still terminating."
      break
    fi
    sleep 1
  done
  exit 143         # the EXIT trap then runs s2_finish (figure mirror)
}

s2_run() {
  # Background + wait so the TERM trap fires promptly, not after srun returns.
  srun "$@" &
  SRUN_PID=$!
  wait "$SRUN_PID"
  local rc=$?
  # Reaped: a TERM landing after completion must not signal a recycled PID,
  # and the handler's kill-on-a-dead-process must not abort it under a
  # caller's `set -e` (it did — the escalation never fired).
  SRUN_PID=""
  s2_log "workload exited rc=$rc"
  # A nonzero rc that is not a signal death is a real failure: no escalation
  # (a retry would fail the same way); resume-safety makes reruns cheap.
  return "$rc"
}
