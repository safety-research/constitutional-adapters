#!/bin/bash
# full22 wave — the shared body every job of scripts/sdfbase/full22/*.sbatch sources (bash).
# Ported from the mt1 wave's attack_wave.sbatch / petri_proxy.sbatch (the tested precedent) and
# parameterised on the cluster: $SHARED is that cluster's shared root (cluster A and cluster B
# spell it differently); QoS/partition/output come from the sbatch headers + the submit script's
# overrides. NEVER --export= (hard rule 1): every value travels positionally or through these functions.

f22_log(){ echo "[$(date -Is)] $*"; }
f22_fatal(){ f22_log "FATAL: $*"; exit 1; }
#: closer-fix-2: a job whose store closer REFUSED leaves with THIS exit code — Slurm says FAILED 75:0,
#: the funnel turns it into `reclose` (reclose_stores.sbatch re-runs the closer under the driver venv from the recipe the job left beside its
#: pending file), never a re-run of the eval. The RESULT line still carries the job's own rc; the refusal is a second line, CLOSER_REFUSED.
F22_EXIT_CLOSER_REFUSED=75
#: THE PETRI PROXY'S RELEASES WITH NO DRIVER (demo-import, 2026-10-11): petri_proxy.sbatch releases its GPU when the grading host driver's heartbeat
#: never arrives within HEARTBEAT_GRACE_S of READY (PROXY_NO_DRIVER) or goes stale for HEARTBEAT_STALE_S (PROXY_DRIVER_LOST). Each leaves with
#: ITS named exit code, never 0: a released proxy that ended COMPLETED 0:0 was un-re-mintable — submit_twin.py's resubmit_decision never re-mints
#: a COMPLETED registering unit (its runs are "the unit's results") and 06 --resubmit's walk skips it — so the demo's proxy sat COMPLETED with
#: nothing generated and no way back but a hand cancel. Slurm says FAILED <code>:0, the RESULT line carries rc=<code> (the registered runs are
#: filed failed, never past running), and `--resubmit` re-mints the unit like any dead one. Distinct from 75 (the closer's refusal: `reclose`).
F22_EXIT_PROXY_NO_DRIVER=76
F22_EXIT_PROXY_DRIVER_LOST=77

# results-registry native emission (additive, never fatal): the job's lifecycle from its own
# environment through the stdlib-only emitter (`python3 -S -m sisteer.registry.emit_cli`, importable before any
# venv), identity from the pending file $SHARED/registry/pending/<jobid>.json — written by the job ITSELF at
# f22_resolve_config (f22_registry_register, below) or by a submit wrapper — and exported as
# SISTEER_REGISTRY_RUN_IDENTITY so every harness store this job opens joins its runs. No spool → silent no-op.
# --- CANARY IDENTITY: a job that must never register ----------------------------------------------------------------
# A serving canary (scripts/sdfbase/steered_canary.sbatch) resolves a production (KEY, VIEW) pair to derive its own and must never
# file the production cells as its runs. The identity comes ONLY from the AUTHORITATIVE evidence every steered canary carries —
# never from an environment flag a refactored launcher may forget, and never from a job NAME alone (the sampled wave's production
# runs are named f22s-<key>-canary-static / -canary-ctl; a name pattern that suppressed the registry would have silently
# unregistered them): (1) the job's own script (scontrol's Command=, F22_JOB_SCRIPT for tests) named
# *_canary.sbatch; (2) a resolved config carrying the top-level `canary:` marker steered_canary.py derive stamps first. The steered
# canary's job-name pattern `scanary*` is a CROSS-CHECK only, and a contradiction FAILS FAST by name before any registry call:
# a `scanary*` name on a production script/config, or a canary script/config under a name f22_registry_evals_for_job maps to a
# production job kind (f22-petri-*, f22-stat-*, f22-att-*, f22-bonclimb-*, f22-misalign-*, f22s-*). F22_REGISTRY_CANARY=1 is a
# BELT only: it annotates the log line and warns when set on a production identity; it never decides.
f22_canary_name(){ case "${SLURM_JOB_NAME:-}" in scanary*) return 0;; *) return 1;; esac; }
f22_canary_identity(){   # -> 0 and prints the reason when this job carries a canary's AUTHORITATIVE identity; 1 otherwise
  local script="${F22_JOB_SCRIPT:-}" cfg="${CFG:-}"
  if [ -z "$script" ] && [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
    script=$(timeout 15 scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -n 1)
  fi
  case "$(basename -- "${script:-/}")" in *_canary.sbatch) echo "job script '$(basename -- "$script")'"; return 0;; esac
  if [ -n "$cfg" ] && [ -f "$cfg" ] && grep -qE '^canary:' "$cfg" 2>/dev/null; then echo "config $cfg carries the canary marker"; return 0; fi
  return 1
}
f22_registry_canary_guard(){   # $1 = the action a canary skips; -> 0 when the caller must return without touching the registry
  local why name="${SLURM_JOB_NAME:-}"
  if why=$(f22_canary_identity); then
    # a canary wearing a PRODUCTION job kind's name would be mapped to evals by f22_registry_evals_for_job: a contradiction, fatal
    if [ -n "$(f22_registry_evals_for_job "$name")" ]; then
      f22_fatal "registry: contradiction — this job carries a canary's identity ($why) under the production job name '$name' (f22_registry_evals_for_job maps it to evals); refusing to run rather than choose a side"
    fi
    echo "[registry] canary identity ($why${F22_REGISTRY_CANARY:+; F22_REGISTRY_CANARY=$F22_REGISTRY_CANARY}): $1" >&2; return 0
  fi
  if f22_canary_name; then
    f22_fatal "registry: contradiction — job name '$name' is the steered canary's pattern (scanary*) but this job carries no canary identity (script '${F22_JOB_SCRIPT:-$(timeout 15 scontrol show job "${SLURM_JOB_ID:-0}" 2>/dev/null | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -n 1)}', config '${CFG:-}'); refusing to run rather than register production cells under a canary's name"
  fi
  [ "${F22_REGISTRY_CANARY:-0}" != 1 ] || echo "[registry] WARNING: F22_REGISTRY_CANARY=1 is set but this job's identity is not a canary's (name '$name', config '${CFG:-}') — the flag never decides; proceeding by identity" >&2
  return 1
}
f22_registry_spool_preflight(){   # static-reg-fix-6: the ONE destination preflight of the inherited SISTEER_REGISTRY_SPOOL — trimmed; empty
  # (the helper resolves it) or an absolute path equal to $SHARED/registry/records, else refused by name; called by EVERY emitting entry point
  # (f22_registry_state at job start, f22_registry_register at config time) BEFORE the installed helper is sourced, because the helper preserves a
  # nonempty inherited value and `mkdir -p`s it: f22_init's `running` emission used to source it first and could create the typo'd spool and file the
  # job-start state into it a whole f22_resolve_config before register's check (static-reg-fix-5 put the check at register only). Creates nothing.
  [ -n "${SHARED:-}" ] || f22_registry_refused "SHARED is unset — the registry destination <shared>/registry/records cannot be named"
  local spool_dest="${SHARED%/}/registry/records"
  SISTEER_REGISTRY_SPOOL="$(printf '%s' "${SISTEER_REGISTRY_SPOOL:-}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"; export SISTEER_REGISTRY_SPOOL
  if [ -n "$SISTEER_REGISTRY_SPOOL" ]; then
    case "$SISTEER_REGISTRY_SPOOL" in
      /*) [ "${SISTEER_REGISTRY_SPOOL%/}" = "$spool_dest" ] || f22_registry_refused "SISTEER_REGISTRY_SPOOL='$SISTEER_REGISTRY_SPOOL' (inherited from the submitting environment) is not the registry destination $spool_dest — the harvester pulls <shared>/registry/records only, so records written there would never reach the registry; nothing created; unset it or set it to exactly that";;
      *) f22_registry_refused "SISTEER_REGISTRY_SPOOL='$SISTEER_REGISTRY_SPOOL' is a relative path (inherited from the submitting environment), not the absolute registry destination $spool_dest — a relative spool would be written inside the checkout, outside the cluster spool and the database; nothing created";;
    esac
  fi
}
f22_registry_state(){   # $1 = running|generated|failed|cancelled, [$2 = scheduler state], [$3 = final]
  f22_registry_canary_guard "state $1 not recorded" && return 0   # a canary's identity never registers (above)
  f22_registry_spool_preflight                                     # static-reg-fix-6: before the helper, which would mkdir a bad inherited value
  [ -r "$SHARED/scripts/registry/registry_env.sh" ] && source "$SHARED/scripts/registry/registry_env.sh"
  SISTEER_REGISTRY_SPOOL="$(printf '%s' "${SISTEER_REGISTRY_SPOOL:-}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"; export SISTEER_REGISTRY_SPOOL   # static-reg-fix-6: trimmed, as register trims
  [ -n "${SISTEER_REGISTRY_SPOOL:-}" ] || return 0
  # static-reg-fix-6: the RESOLVED value (the helper's, or the inherited one it preserved) is the destination or the state is not filed anywhere else
  [ "${SISTEER_REGISTRY_SPOOL%/}" = "${SHARED%/}/registry/records" ] || f22_registry_refused "SISTEER_REGISTRY_SPOOL='$SISTEER_REGISTRY_SPOOL' (after sourcing $SHARED/scripts/registry/registry_env.sh) is not the registry destination ${SHARED%/}/registry/records — the job-start '$1' state would be filed outside the harvester's spool"
  local pend="$SHARED/registry/pending/${SLURM_JOB_ID:-none}.json"
  if [ -f "$pend" ]; then export SISTEER_REGISTRY_RUN_IDENTITY="$pend"; fi
  # AN ARRAY TASK: the climb array's tasks register ONE run per cell (f22_registry_producer_key), so a task's state event is not the run's — a
  # task exiting non-zero after a peer task's (or the store512's) close filed `failed` over the run's `generated` / `superseded`, which the
  # reducer applies (a lateral/terminal event has no rank), and a task starting after a peer's close filed `running` over it the same way.
  # The task's `running` / `failed` / `cancelled` goes through the closer's array-task mode: filed on a run only when no close has decided it,
  # recorded on the task otherwise (f22_registry_array_task_event, below). A job that is not an array is byte for byte as before.
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    case "$1" in
      running)          f22_registry_array_task_event started "${2:-RUNNING}"; return 0;;
      failed|cancelled) f22_registry_array_task_event "$1" "${2:-}"; return 0;;
    esac
  fi
  # GPU count / type are captured by the emitter from THIS job's environment (SLURM_GPUS_ON_NODE / SLURM_JOB_GPUS /
  # CUDA_VISIBLE_DEVICES, nvidia-smi, sacct AllocTRES as the fallback) and named in the running state's note.
  PYTHONPATH="${REPO_DIR:-$PWD}/src${PYTHONPATH:+:$PYTHONPATH}" python3 -S -m sisteer.registry.emit_cli state --state "$1" \
    --pending-dir "$SHARED/registry/pending" --job "${SLURM_JOB_ID:-}" --node "${SLURMD_NODENAME:-$(hostname)}" \
    ${2:+--scheduler-state "$2"} ${3:+--final} \
    2>&1 | sed 's/^/[registry] /' || true
}
f22_registry_array_task_event(){   # $1 = started|completed|failed|cancelled, [$2 = scheduler state] — an ARRAY TASK's event for its shared runs
  # The closer's array-task mode (scripts/sdfbase/full22/registry_close_stores.py --array-task-event; the rule and its test live there): under the
  # array's lock and each run's store lock, a task's event never reverses a close of the shared run; the task's exit is recorded on the task
  # ($SHARED/registry/pending/<array job>.tasks/<task>.json) and the last task to exit reconciles the array. The task set comes from the scheduler
  # (SLURM_ARRAY_TASK_MIN-MAX[:STEP]); without it the mode refuses by name and files nothing — never a guess, never the bare per-task filing.
  # Every line goes to STDOUT (the RESULT hook calls f22_registry_state with stderr dropped); a refusal is loud there, and the job's rc stands.
  [ -n "${SLURM_ARRAY_TASK_ID:-}" ] || return 0
  f22_registry_canary_guard "array task event $1 not recorded" && return 0
  [ -n "${SISTEER_REGISTRY_SPOOL:-}" ] || return 0
  local pend="$SHARED/registry/pending/${SLURM_JOB_ID:-none}.json"
  if [ ! -f "$pend" ]; then echo "[registry] array task ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_JOB_ID:-?}: no pending file $pend — nothing registered (yet); the task's $1 has no run to attach to"; return 0; fi
  local tasks="${SLURM_ARRAY_TASK_MIN:-}-${SLURM_ARRAY_TASK_MAX:-}"
  if [ -z "${SLURM_ARRAY_TASK_MIN:-}" ] || [ -z "${SLURM_ARRAY_TASK_MAX:-}" ]; then tasks=""; fi
  if [ -n "${SLURM_ARRAY_TASK_STEP:-}" ] && [ "${SLURM_ARRAY_TASK_STEP}" != 1 ] && [ -n "$tasks" ]; then tasks="$tasks:$SLURM_ARRAY_TASK_STEP"; fi
  local -a args=(--pending "$pend" --spool "$SISTEER_REGISTRY_SPOOL" --code-sha "${WAVE_SHA:-}" --array-task-event "$1" --array-job "${SLURM_ARRAY_JOB_ID:-}" \
                 --array-task "$SLURM_ARRAY_TASK_ID" --array-tasks "$tasks" --node "${SLURMD_NODENAME:-$(hostname)}")
  [ -n "${2:-}" ] && args+=(--scheduler-state "$2")
  [ -n "${WAVE_ROOT:-}" ] && args+=(--wave-root "$WAVE_ROOT")
  [ -n "${TAG:-}" ] && args+=(--tag "$TAG")
  [ -n "${SLURM_JOB_NAME:-}" ] && args+=(--job-name "$SLURM_JOB_NAME")
  [ -n "${SISTEER_REGISTRY_PUSH:-}" ] && args+=(--push-script "$SISTEER_REGISTRY_PUSH")
  local py="${F22_PY:-python3}" rc=0 out=""
  out=$(PYTHONPATH="${REPO_DIR:-$PWD}/src${PYTHONPATH:+:$PYTHONPATH}" "$py" "${REPO_DIR:-$PWD}/scripts/sdfbase/full22/registry_close_stores.py" "${args[@]}" 2>&1) || rc=$?
  [ -n "$out" ] && printf '%s\n' "$out" | sed 's/^/[registry] /'
  [ "$rc" -eq 0 ] || echo "[registry] array task ${SLURM_ARRAY_TASK_ID} of ${SLURM_ARRAY_JOB_ID:-?}: event $1 NOT settled (rc $rc under $py) — the shared runs' lifecycle was not touched; nothing filed over a peer's close; the task's exit is on the RESULT line and in sacct"
  return 0
}

# --- results-registry SELF-REGISTRATION (2026-09-17: emission is the default; no submit wrapper, no per-driver opt-in) ----
# A job REGISTERS ITSELF the moment its config is resolved (f22_resolve_config; the misalign proxy after its own binding):
# `python3 -m sisteer.registry.emit_cli register --from-config` builds one run per (declared eval x declared object) from
# the config's explicit_cells, the roster + pins (object shas), the SI texts (prompted objects) and the registry seeds (the
# canonical instrument per eval for this model + serving path), writes every `run` (running, registered_at=job_start) and
# the pending file $SHARED/registry/pending/<jobid>.json with a `runs` list carrying each run's routing facts; the store
# hooks then route every row of every CellStore this job opens onto its run through SISTEER_REGISTRY_RUN_IDENTITY, and
# f22_result's terminal state reaches every run of the job. Idempotent: a pending file a submit wrapper already wrote is
# kept (sbatch_registered.sh still works). PLAIN python3 (PyYAML — the interpreter f22_select_auto already relies on),
# never `-S`. Every line best-effort: a registration failure is one `[registry]` line and the job continues.
#
# WHICH EVALS a job runs is declared by its KIND — the job-name table below (submit_full22.sh's names) — or explicitly by
# the job body / submitter as F22_REGISTRY_EVALS (space- or comma-separated eval ids; a batch job inherits the exported
# environment, never the sbatch export flag). A job whose kind is not in the table and declares nothing registers nothing and says so.
# Evals are registry ids (configs/registry/evals.yaml); the table lists what each job kind's drivers actually run.
f22_registry_evals_for_job(){   # $1 = job name -> the evals that job kind runs (space-separated); "" = unknown kind
  case "$1" in
    f22-petri-*)         echo "petri_pooled petri_cyber_bio_chem_capabilities petri_self_preservation petri_delusion_sycophancy petri_defer_to_users petri_reasoning_unfaithfulness petri_instruction_evasion";;
    f22-stat-*-1)        echo "pasttense codeattack";;
    f22-stat-*-2)        echo "renellm grok420";;
    f22-stat-*-3)        echo "strongreject_bon";;
    f22-stat-*-all)      echo "pasttense codeattack renellm grok420 strongreject_bon";;
    f22-att-*-crescendo|f22-att-goalset-*-crescendo)     echo "crescendo";;
    f22-att-*-pair|f22-att-goalset-*-pair)               echo "pair";;
    f22-att-*-actorattack|f22-att-goalset-*-actorattack) echo "actorattack";;
    f22-bonclimb-*)      echo "strongreject_bon";;
    f22-misalign-em*)    echo "em_risky_financial em_bad_medical em_extreme_sports em_base_control";;
    f22-misalign-*)      echo "agentic_leaking agentic_murder";;
    f22s-*)              echo "pasttense codeattack renellm grok420 xstest or_bench_hard";;
    *) echo "";;
  esac
}
f22_registry_producer_key(){   # $1 job name  $2 key tag (KEY[_VIEW])  $3 the family's identity segments (optional) -> the run key; "" = jobplan's default
  # A JOB-INDEPENDENT producer key for the job kinds that registered `<job_name>@<job_id>` — a pure function of the measurement identity:
  # the per-family prefix (the job kind, read off the job name the way f22_registry_evals_for_job does), the (key, view) tag, and the
  # family's own segments (the pass, the shard range, the mode, the padding level — exported by the sbatch as F22_REGISTRY_KEY_SEGMENTS
  # before it registers); never a job id or a timestamp; `-`-separated, no `@`. The run-id derivation is unchanged
  # (identity.run_id = campaign:eval:instrument[:12]:sha16(producer key | object key) — the OBJECT key carries the cell), so a re-attempt of
  # the same measurement registers the SAME run and the attempt record tells attempts apart, the way the sampled rungs already do under
  # F22_REGISTRY_RUN_SUFFIX. Future registrations only: existing pending files and ids stand. Kept on the job-keyed default on
  # purpose: the gsm8k rows and every kind the table does not name (empty output).
  # f22-bonclimb (the companion of the closer's store512 / climb survivor rule): the climb
  # array's four tasks used to register one run EACH (`<job_name>@<job_id>`, kept on the default: the smoke's ten BoN stores carried
  # five current registrations, the store512's plus four), while the closer's rule retires a climb registration only against the store512's.
  # Now ONE cell registers ONCE across the array — the key is the kind and the (key, view) tag, never a task's job id. NEW registrations
  # only: f22_registry_register keeps a pending file that already exists, so every run registered before this keeps its run_id.
  local job="${1:-}" tag="${2:?key tag}" seg="${3:-}" fam=""
  case "$job" in
    f22-petri-*)                                                                        fam="petri";;
    f22-stat-*-1|f22-stat-*-2|f22-stat-*-3|f22-stat-*-all)                              fam="stat-${job##*-}";;
    f22-bonclimb-*)                                                                     fam="bonclimb";;
    f22-att-goalset-*-crescendo|f22-att-goalset-*-pair|f22-att-goalset-*-actorattack)   fam="att-goalset-${job##*-}";;
    f22-att-*-crescendo|f22-att-*-pair|f22-att-*-actorattack)                           fam="att-${job##*-}";;
    f22-misalign-em*)                                                                   fam="misalign-organisms";;
    f22-misalign-*)                                                                     fam="misalign-agentic";;
    padstat|padstat-*)                                                                  fam="padstat";;
    padobj|padobj-*)                                                                    fam="padobj";;
    *) echo ""; return 0;;
  esac
  local key="sdfbase_full22-${fam}-${tag}${seg:+-$seg}"
  case "$key" in
    *@*|*[!A-Za-z0-9_.-]*) echo "[registry] producer key '$key' is not one token of [A-Za-z0-9_.-] without '@' (job '$job', tag '$tag', segments '$seg')" >&2; return 1;;
  esac
  printf '%s\n' "$key"
}

f22_registry_register(){   # uses KEY VIEW CFG (f22_resolve_config / the misalign binding), F22_SERVING_PATH, WAVE_SHA, CLUSTER, SLURM_*
  f22_registry_canary_guard "nothing registered" && return 0
  local helper="$SHARED/scripts/registry/registry_env.sh"
  local job="${SLURM_JOB_ID:-}" pend evals_csv key_tag rc
  [ -n "$job" ] || { echo "[registry] register: no SLURM_JOB_ID (a manual run); nothing registered" >&2; return 0; }
  pend="$SHARED/registry/pending/$job.json"
  evals_csv="${F22_REGISTRY_EVALS:-$(f22_registry_evals_for_job "${SLURM_JOB_NAME:-}")}"
  evals_csv="$(printf '%s' "$evals_csv" | tr ' ' ',' | tr -s ',' | sed 's/^,//; s/,$//')"
  # a job KIND that declares no evals (not in the table above, no F22_REGISTRY_EVALS) and that no submit wrapper registered is not a registry job:
  # the table is the declaration, so this stays a notice (the paper tags' mmluF / padstat / probe jobs run as before)
  if [ -z "$evals_csv" ] && [ ! -f "$pend" ]; then
    echo "[registry] register: job kind '${SLURM_JOB_NAME:-?}' declares no evals (not in f22_registry_evals_for_job; set F22_REGISTRY_EVALS); nothing registered" >&2; return 0
  fi
  # static-reg-fix-5: from here on this job IS a registry job — its kind declares evals, or a submit wrapper registered it
  # (sbatch_registered.sh wrote $pend) — so the spool its writers and the closer will use is validated FIRST: the inherited value, trimmed, must be
  # the ONE ingestion destination the harvester pulls, $SHARED/registry/records, checked BEFORE the helper is sourced
  # (the installed registry_env.sh preserves a nonempty inherited value and `mkdir -p`s it, so a typo like <shared>/registrry/records used to create
  # its own parent and pass fix-4's parent check; /tmp/... passed outright) and BEFORE an existing pending file is accepted (an identity on disk
  # proves a registration happened, not that THIS job's emitter writes into the registry's spool). A whitespace-only inherited value counts as
  # unset: the helper resolves it, else the empty-spool refusals below fire. Nothing is created before the value is known to be the destination.
  f22_registry_spool_preflight                       # static-reg-fix-6: the shared destination preflight, before the helper is sourced
  local spool_dest="${SHARED%/}/registry/records"
  [ -r "$helper" ] && source "$helper"
  # From here on this job KIND registers — an unresolved spool (the helper absent, or the variable
  # unset / empty after sourcing it) used to RETURN SUCCESS above this point, so a sampled rung with a missing helper selected its pairs and went
  # to the standup unregistered (the CLI's own no-spool exit 2 unreachable). It is fatal HERE, before the deferral:
  # a phase that would register at its gate cannot register there either, and must not start. A kind that does not register opted out above, by kind.
  # The value is TRIMMED and must be an ABSOLUTE path whose PARENT directory exists — a whitespace-only
  # value passed `-z`, the helper preserved it, emit.spool_root returned it verbatim and RecordWriter would have written into a relative
  # directory named by spaces inside the checkout while pending/<job>.json was produced: a standup running with its records outside the
  # cluster spool and the database (the unregistered wave again). Every check sits here, before the deferral, like the empty case.
  SISTEER_REGISTRY_SPOOL="$(printf '%s' "${SISTEER_REGISTRY_SPOOL:-}" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"; export SISTEER_REGISTRY_SPOOL
  if [ -z "$SISTEER_REGISTRY_SPOOL" ]; then
    if [ -r "$helper" ]; then f22_registry_refused "SISTEER_REGISTRY_SPOOL is unset or empty after sourcing $helper (the helper names no spool)"; fi
    f22_registry_refused "the registry helper $helper is absent and SISTEER_REGISTRY_SPOOL is unset — no spool to register into"
  fi
  local spool_src="after sourcing $helper"; [ -r "$helper" ] || spool_src="the helper $helper is absent; the value is the environment's"
  case "$SISTEER_REGISTRY_SPOOL" in
    /*) ;;
    *) f22_registry_refused "SISTEER_REGISTRY_SPOOL='$SISTEER_REGISTRY_SPOOL' is not an absolute path ($spool_src) — a relative spool would be written inside the checkout, outside the cluster spool and the database";;
  esac
  # static-reg-fix-5: the resolved value (the helper's, or the inherited one it preserved) is the destination or nothing — checked before anything is created here
  [ "${SISTEER_REGISTRY_SPOOL%/}" = "$spool_dest" ] || f22_registry_refused "SISTEER_REGISTRY_SPOOL='$SISTEER_REGISTRY_SPOOL' ($spool_src) is not the registry destination $spool_dest — the harvester pulls <shared>/registry/records only, so records written there would never reach the registry"
  # The spool DIRECTORY need not pre-exist — records.write_records makedirs `<spool>/<campaign>/<run dir>`
  # on the first record, so a fresh cluster root names a spool that nothing has created yet — and with the value pinned to $SHARED/registry/records
  # above, the directory that MUST already exist is the storage root $SHARED itself: absent on this node it is a root that is not mounted here, and
  # creating the chain would file the run somewhere nobody reads. $SHARED/registry is the registration's own to create (the pending file goes under
  # $SHARED/registry/pending, which emit_cli makedirs), so the first job on a shared root with no registry/ yet registers and creates the spool
  # here, as the writer would. An earlier revision required the spool's immediate parent ($SHARED/registry) instead and refused exactly that root —
  # the red it introduced in tests/test_submit_resubmit_proof.py (the fold proof's fresh fake cluster), a file its chain did not run.
  [ -d "${SHARED%/}" ] || f22_registry_refused "SISTEER_REGISTRY_SPOOL='$SISTEER_REGISTRY_SPOOL': the storage root $SHARED does not exist on this node ($spool_src) — not mounted here; no spool to register into"
  mkdir -p "$SISTEER_REGISTRY_SPOOL" || f22_registry_refused "SISTEER_REGISTRY_SPOOL='$SISTEER_REGISTRY_SPOOL' could not be created under $SHARED ($spool_src)"
  # A submit wrapper's registration is accepted only now, after the spool this job will write into is the destination
  if [ -f "$pend" ]; then export SISTEER_REGISTRY_RUN_IDENTITY="$pend"; return 0; fi   # a submit wrapper registered it (sbatch_registered.sh)
  # From here on this job ASKED to be registered, and everything that ends without the pending file
  # is FATAL (f22_registry_refused -> exit 1, before any venv / standup): 838299's 12 unproduced runs and six steered jobs ran on
  # "nothing written (job continues)". A phase whose pairs a per-pair GATE selects after f22_resolve_config (the static 8,192 / 16,384 rungs)
  # DEFERS this job-start call and re-enters with F22_REGISTRY_PAIRS once the gate has decided (common_sampled.sh f22s_register_gated_pairs).
  if [ -n "${F22_REGISTRY_DEFER:-}" ]; then echo "[registry] register: deferred to the phase's gate (F22_REGISTRY_DEFER=$F22_REGISTRY_DEFER) — the phase registers exactly the pairs its gate selects, before any standup" >&2; return 0; fi
  [ -n "${CFG:-}" ] && [ -f "${CFG}" ] || f22_registry_refused "no resolved config (CFG='${CFG:-}' unset or missing)"
  [ -n "${KEY:-}" ] || f22_registry_refused "KEY is unbound (call after f22_resolve_config or the job's own binding)"
  key_tag="$KEY$( [ "${VIEW:-main}" = main ] || echo "_$VIEW" )"
  local reg_cfg="$CFG"
  # static-reg-fix: a phase that exports its rung (common_sampled.sh f22s_parse_phase) registers ONE RUN PER RUNG — the producer key is
  # the rung's private run name `sdfbase_full22_<tag><suffix>`, the cap the DRAWN one with its index and ladder (jobplan.rung_facts refuses an
  # inconsistent triple); a job without the exports registers exactly as before (<job_name>@<job_id>, the instrument's cap)
  local rung_flags=()
  if [ -n "${F22_REGISTRY_RUN_SUFFIX:-}" ]; then rung_flags+=(--producer-key "sdfbase_full22_${key_tag}${F22_REGISTRY_RUN_SUFFIX}")
  else
    # Every other registering kind keys its runs by the MEASUREMENT (f22_registry_producer_key, above) — the per-family prefix, the
    # (key, view) tag and the sbatch's F22_REGISTRY_KEY_SEGMENTS — so a re-attempt registers the same run (the climb array's four tasks
    # included); an empty key keeps jobplan's `<job_name>@<job_id>` default (gsm8k, kinds the table does not name); a
    # malformed key is a refusal, not a silent default
    local pkey
    pkey=$(f22_registry_producer_key "${SLURM_JOB_NAME:-}" "$key_tag" "${F22_REGISTRY_KEY_SEGMENTS:-}") \
      || f22_registry_refused "the job-independent producer key could not be formed (job name '${SLURM_JOB_NAME:-}', tag '$key_tag', segments '${F22_REGISTRY_KEY_SEGMENTS:-}'; the [registry] line above)"
    [ -n "$pkey" ] && rung_flags+=(--producer-key "$pkey")
  fi
  # The serving jobs of this job's store known at job start — the funnel hands a multi-turn target the
  # sibling shard groups' live job ids in F22_REGISTRY_SERVING_JOBS (submit_twin.sibling_serving_jobs, the job-side env); the CLI adds this
  # job's own id and sorts. A multi-turn TARGET (F22_REGISTRY_MULTITURN_TARGET=1, set by submit_twin) passes
  # the flag in EVERY case — an empty list states the job's own id alone (the serving-jobs backfill pass fills siblings unknown at
  # registration); every other job kind passes it only when the funnel handed it a list (unset = nothing stated), as before
  if [ "${F22_REGISTRY_MULTITURN_TARGET:-}" = 1 ]; then rung_flags+=(--serving-jobs "${F22_REGISTRY_SERVING_JOBS:-}")
  elif [ -n "${F22_REGISTRY_SERVING_JOBS:-}" ]; then rung_flags+=(--serving-jobs "$F22_REGISTRY_SERVING_JOBS"); fi
  [ -n "${F22_REGISTRY_TOKEN_CAP:-}" ] && rung_flags+=(--token-cap "$F22_REGISTRY_TOKEN_CAP" --rung-index "${F22_REGISTRY_RUNG_INDEX:?F22_REGISTRY_RUNG_INDEX goes with F22_REGISTRY_TOKEN_CAP}" --ladder "${F22_REGISTRY_LADDER:?F22_REGISTRY_LADDER goes with F22_REGISTRY_TOKEN_CAP}")
  [ -n "${F22_REGISTRY_PASS_LABEL:-}" ] && rung_flags+=(--pass-label "$F22_REGISTRY_PASS_LABEL")
  [ -n "${F22_REGISTRY_TRIAL_INDEX:-}" ] && rung_flags+=(--trial-index "$F22_REGISTRY_TRIAL_INDEX")
  # static-reg-fix-2: the gate-selected pairs — ONLY these (eval, object) runs exist; jobplan refuses a pair outside evals x explicit_cells
  local -a pair_flags=()
  [ -n "${F22_REGISTRY_PAIRS:-}" ] && pair_flags+=(--pairs "$F22_REGISTRY_PAIRS")
  if [ -n "${F22_REGISTRY_CELLS:-}" ]; then
    # A PER-CELL job (the gsm8k rows: one (key, view, cell) per job) registers ONLY
    # the cells it serves: a job-local copy of the overlay with explicit_cells narrowed to F22_REGISTRY_CELLS (comma list), written
    # beside the pending file and naming its source; the committed configs stay byte-identical. A cell the overlay does not enumerate
    # refuses by name and registers nothing (the job continues, as every registration failure does).
    reg_cfg="$SHARED/registry/pending/${job}.config.yaml"; mkdir -p "$SHARED/registry/pending"
    python3 - "$CFG" "$reg_cfg" "$F22_REGISTRY_CELLS" <<'PY' 2> >(sed 's/^/[registry] /' >&2) \
      || f22_registry_refused "cannot narrow $CFG to cells '$F22_REGISTRY_CELLS' (the [registry] line above names the cell the overlay does not enumerate)"
import os, sys, yaml
src, dst, cells = sys.argv[1], sys.argv[2], [c for c in sys.argv[3].split(",") if c]
cfg = yaml.safe_load(open(src)); have = list(cfg.get("explicit_cells") or [])
missing = [c for c in cells if c not in have]
if missing:
    sys.exit(f"register: cells {missing} are not in {src}'s explicit_cells {have}")
cfg["explicit_cells"] = cells
cfg["registry_narrowed_from"] = src; cfg["registry_narrowed_cells"] = cells
with open(dst + ".tmp", "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
os.replace(dst + ".tmp", dst)
PY
  fi
  # The narrowed copy lives OUTSIDE the checkout ($SHARED/registry/pending), and jobplan derives the checkout from the CONFIG'S path
  # (its nearest ancestor holding configs/registry, else the config's grandparent): from the copy that walk lands on $SHARED/registry
  # and every per-cell job refused "no registry seeds at $SHARED/registry/configs/registry" while its pins silently read as none — both
  # waves' gsm8k rows registered nothing. So the per-cell path NAMES the checkout's seeds and pins.
  local -a ck_flags=()
  if [ "$reg_cfg" != "$CFG" ]; then
    local ckroot="${REPO_DIR:-$PWD}"
    ck_flags+=(--seeds "$ckroot/configs/registry")
    [ -f "$ckroot/configs/sdfbase_full22_pins.json" ] && ck_flags+=(--pins "$ckroot/configs/sdfbase_full22_pins.json")
  fi
  PYTHONPATH="${REPO_DIR:-$PWD}/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m sisteer.registry.emit_cli register --from-config "$reg_cfg" \
    --model-key "$key_tag" --roster-selection "${VIEW:-main}" --evals "$evals_csv" --serving-path "${F22_SERVING_PATH:-lora_native}" \
    --campaign "${F22_REGISTRY_CAMPAIGN:-sdfbase_full22}" --job-id "$job" ${SLURM_JOB_NAME:+--job-name "$SLURM_JOB_NAME"} \
    --pending-dir "$SHARED/registry/pending" ${WAVE_SHA:+--code-sha "$WAVE_SHA"} ${CLUSTER:+--substrate "$CLUSTER"} \
    ${F22_ROSTER:+--roster "$F22_ROSTER"} ${ck_flags[@]+"${ck_flags[@]}"} ${pair_flags[@]+"${pair_flags[@]}"} ${rung_flags[@]+"${rung_flags[@]}"} >/dev/null 2> >(sed 's/^/[registry] /' >&2); rc=$?
  if [ -f "$pend" ]; then
    export SISTEER_REGISTRY_RUN_IDENTITY="$pend"
    f22_registry_state running "${SLURM_JOB_STATE:-RUNNING}" || true   # the job's runs exist now: file `running` for each
    return 0
  fi
  f22_registry_refused "emit_cli register --from-config exited $rc and wrote no pending file at $pend (evals: $evals_csv${F22_REGISTRY_PAIRS:+; pairs: $F22_REGISTRY_PAIRS}${F22_REGISTRY_CELLS:+; cells: $F22_REGISTRY_CELLS}; the [registry] lines above name the refusal)"
}
f22_registry_refused(){   # static-reg-fix-2: a registration this job asked for and did not get is a fatal preflight — exit 1 before any venv / standup
  f22_fatal "registry: self-registration REFUSED for job ${SLURM_JOB_ID:-?} (${SLURM_JOB_NAME:-?}) — $*; a job whose runs do not exist in the registry does not run (D984 / D1265): fix the refusal, then resubmit"
}

# --- THE LAUNCH-TIME ARGV HOOK survives every lane (2026-09-16, wave 0e21d6dc) ------------------------------------------
# f22_standup appends F22_STANDUP_EXTRA VERBATIM to mt_serving_up.py's argv. A SCALAR F22_STANDUP_EXTRA in the job's
# environment (ONE token — e.g. --serving-path=lens, scripts/modal/strength_served.py; a batch job inherits the
# submitter's exported environment, never --export=) is that hook — and it reached the standup only on lanes that
# never touched the array: target_sampled.sbatch's `F22_STANDUP_EXTRA=(--enable-log-requests)` CLOBBERED it, so a
# launch-time pin could not reach that lane's standup. Every lane therefore builds its flags ON TOP of the inherited
# value through THIS helper — its FIRST touch is always `f22_standup_extra_reset <flags>` (a bare `+=` first touch
# would keep an exported-EMPTY scalar as an empty argv element the standup parser rejects); `+=` only after it. The
# helper snapshots the inherited tokens on the FIRST call (so a rebuild — target_attacks' window branches,
# m5t_/g8k_standup_extra called twice by the Nemotron probe — never re-inherits its own previous flags), DROPS an
# empty element (an exported-empty scalar is ZERO arguments: the argv byte-identical to unset), and REFUSES by name a
# --serving-path riding the array (below). Tests run every shape under set -eu, through the composed argv.
#
# WHAT THE HOOK MAY NOT CARRY: an ENGINE PIN. The serving path is selected and RECORDED in f22_init
# (f22_select_serving_path: an explicit F22_SERVING_PATH wins) and f22_standup passes `--serving-path` from that
# selection with the record beside it; an inherited `--serving-path=lens` appended after them is taken by the standup's
# parser (last wins) and load_serving_path_selection then refuses the record ("selected 'lora_native' but this standup
# serves --serving-path lens") — the pin cannot launch either way. So it travels as F22_SERVING_PATH, BEFORE selection,
# and any --serving-path element in F22_STANDUP_EXTRA is refused BY NAME, at f22_init (before the selection is recorded),
# at the helper's snapshot and again before the splice.
f22_standup_extra_inherited_check(){   # refuse, by name, a --serving-path element (any spelling, any position) in F22_STANDUP_EXTRA
  local t
  for t in ${F22_STANDUP_EXTRA[@]+"${F22_STANDUP_EXTRA[@]}"}; do
    case "$t" in
      --serving-path|--serving-path=*)
        f22_fatal "F22_STANDUP_EXTRA carries '$t': an engine pin travels as F22_SERVING_PATH (read by f22_select_serving_path in f22_init, BEFORE the selection is recorded), never as a standup argv hook — appended after --serving-path / --serving-path-selection the standup's parser takes the later value and load_serving_path_selection refuses the record; export F22_SERVING_PATH=lora_native|lens and drop the element";;
    esac
  done
}
f22_standup_extra_reset(){   # $@ = this lane's standup flags; -> F22_STANDUP_EXTRA = <inherited tokens, non-empty> $@
  local t
  if [ -z "${F22_STANDUP_INHERITED_TAKEN:-}" ]; then
    f22_standup_extra_inherited_check
    F22_STANDUP_INHERITED=()
    for t in ${F22_STANDUP_EXTRA[@]+"${F22_STANDUP_EXTRA[@]}"}; do
      if [ -n "$t" ]; then F22_STANDUP_INHERITED+=("$t"); fi
    done
    F22_STANDUP_INHERITED_TAKEN=1
  fi
  F22_STANDUP_EXTRA=(${F22_STANDUP_INHERITED[@]+"${F22_STANDUP_INHERITED[@]}"} "$@")
}

# --- the ROSTER every reader of this job reads ----------------------------------------------------------------------
# F22_ROSTER (default: the production roster) rides as `--roster "$F22_ROSTER"` on EVERY make_bank_view.py /
# roster_cell_count.py / roster_view.py call of common.sh and of the sbatch scripts, so a job's roster is ONE selection,
# never a per-reader default. A batch job inherits the submitter's exported environment (never --export=), so
# `F22_ROSTER=configs/sdfbase_full22_roster_psrcert.yaml sbatch ...` runs a job over the CERTIFICATION-ONLY roster
# (llama8b's `psrcert` view: the staged all-positions PSR cell beside its LoRA twin — no production view of llama8b
# carries a PSR cell, and the production roster is never edited for a certification). Unset = today's production path,
# byte-identical in effect. A (target, view) the selected roster does not name refuses at the count guard (f22_bank_view),
# by name; f22_init refuses a roster path that does not exist in the checkout. Relative to $REPO_DIR (f22_init cds there).
F22_ROSTER_PRODUCTION=configs/sdfbase_full22_roster.yaml
F22_ROSTER="${F22_ROSTER:-$F22_ROSTER_PRODUCTION}"
# --- the serving path: an explicit F22_SERVING_PATH wins; set NONE, the KIND-AWARE default over the launch's DECLARED cells --
# THE KIND-AWARE DEFAULT: use native when possible, select the lens route by itself when a launch's steered cells are all
# lens-only kinds, and make the selection recoverable in the logs. The RULE lives in ONE Python
# function — registry_adapter.select_serving_path, the standup's own serve / skip / refuse disposition per cell
# (defense_cells.native_serving_disposition) — reached through scripts/sdfbase/full22/serving_path_select.py; the shell only
# gathers the cell list and evals the verdict. Never a shell predicate on cell names beside it.
#
# THE CELL LIST IS DECLARED BY THE CALLER, never assumed (an earlier draft derived it from (KEY, VIEW) as the
# full22 campaign manifest, and every f22_init caller with its OWN manifest — the AlphaSteer campaign, the strength screen, the
# misalign proxy — was misread: an all-AlphaSteer launch selected native, skipped its steering cell, and the eval never ran).
# Before f22_init a caller sets exactly what it knows, in this precedence:
#   * F22_SERVING_PATH            — the path itself: EXPLICIT, wins, nothing classified (the whole-view path byte for byte);
#   * F22_LAUNCH_CELLS            — a Bash ARRAY of HARNESS cell ids (`alphasteer` / `arditi` / `none` spell the same in both
#                                   vocabularies); DECLARED EMPTY = this job stands up no target (the attacker pool, a capture job);
#   * F22_LAUNCH_MANIFEST         — the launch's manifest (relative to the checkout or absolute): its cells are the standup's, in
#                                   harness spelling — the campaign scripts declare f22_full22_manifest KEY VIEW, the misalign proxy
#                                   its misalign manifest;
#   * DEFERRED to the standup     — F22_SERVING_PATH_DEFER=1, or the job OWNS its wave root (F22_WAVE_ROOT set: nothing of the path
#                                   is needed before the standup): f22_init leaves the path, the engine and (without F22_WAVE_ROOT)
#                                   WAVE_ROOT unbound; f22_bind_serving_path <manifest> binds them from the launch's ACTUAL manifest
#                                   — a wrapper's resolver calls it (as22_resolve_config, the strength screen), and f22_standup calls
#                                   it from $MANIFEST when still deferred (the math500 / GSM8K rows own F22_WAVE_ROOT);
#   * none of these               — REFUSED, naming the caller. The full22 campaign manifest is never assumed from KEY / VIEW.
# Runs on the node's stdlib python3 + PyYAML (no venv, no bank view and no torch exist at f22_init; WAVE_ROOT is consumed before
# either), so the classification is BY KIND: a position-masked LoRA fit reads as its natively servable kind here and is skipped
# BY NAME by the standup's own partition, exactly as before. A refusal (a malformed / duplicate / unmapped id, an unreadable
# manifest) is fatal inside the allocation but BEFORE the server launch — the job dies in seconds, not after a model load. Why: the
# MATH-500 steer wave (2026-09-14) — an all-AlphaSteer / Arditi launch — died at
# the native standup for want of this; it now selects the lens by itself, and the log says so.
f22_full22_manifest(){   # $1 = model key, $2 = view -> the CAMPAIGN manifest for (KEY, VIEW), f22_resolve_config's spelling
  echo "configs/capeval/manifests/petri_full22_$1$( [ "$2" = main ] || echo "_$2" ).yaml"
}

f22_bind_engine(){   # F22_SERVING_PATH -> TARGET_ENGINE (the engine every driver asserts from the AUTHORITATIVE stamps), both exported
  if [ "$F22_SERVING_PATH" = lens ]; then TARGET_ENGINE="lens:$REDUCTION_MODE"; else TARGET_ENGINE=lora_native; fi
  export F22_SERVING_PATH TARGET_ENGINE
}

f22_select_auto(){   # $@ = the selector's cell-source args -> F22_SERVING_PATH F22_SERVING_PATH_MODE=auto F22_SERVING_PATH_SELECTION_JSON; ONE log line
  local sel py out
  sel="${REPO_DIR:?f22_select_auto needs REPO_DIR}/scripts/sdfbase/full22/serving_path_select.py"
  [ -f "$sel" ] || f22_fatal "no $sel on this tree — the kind-aware serving-path default is not on this sha; set F22_SERVING_PATH explicitly"
  py="${F22_SELECT_PY:-python3}"   # the node's stdlib python (+ PyYAML); F22_SELECT_PY names another (the tests' venv)
  out=$(cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src" "$py" "$sel" "$@" --sh 2>&1) \
    || f22_fatal "serving-path selection refused ($*): $out"
  # eval ONLY the four assignments (a stray interpreter warning on the merged stream must not reach eval)
  eval "$(printf '%s\n' "$out" | grep -E '^F22_SERVING_PATH_(SELECTED|MODE|LOG|SELECTION_JSON)=')"
  [ -n "${F22_SERVING_PATH_SELECTED:-}" ] && [ -n "${F22_SERVING_PATH_SELECTION_JSON:-}" ] \
    || f22_fatal "serving-path selection returned no verdict ($*): $out"
  F22_SERVING_PATH="$F22_SERVING_PATH_SELECTED"
  case "$F22_SERVING_PATH" in lora_native|lens) ;; *) f22_fatal "serving-path selection returned '$F22_SERVING_PATH' (not lora_native | lens)";; esac
  f22_log "$F22_SERVING_PATH_LOG"
}

f22_select_serving_path(){   # -> F22_SERVING_PATH F22_SERVING_PATH_MODE (explicit|auto|deferred) F22_SERVING_PATH_SELECTION_JSON; ONE log line
  local caller="$0${SLURM_JOB_NAME:+ (job $SLURM_JOB_NAME)}"
  if [ -n "${F22_SERVING_PATH:-}" ]; then
    case "$F22_SERVING_PATH" in lora_native|lens) ;; *) f22_fatal "F22_SERVING_PATH must be lora_native or lens, got '$F22_SERVING_PATH'";; esac
    F22_SERVING_PATH_MODE=explicit
    F22_SERVING_PATH_SELECTION_JSON="{\"cells_by_disposition\":null,\"cells_source\":null,\"mode\":\"explicit\",\"reason\":\"F22_SERVING_PATH set by caller\",\"rule\":\"kind-aware-2026-09-15\",\"selected\":\"$F22_SERVING_PATH\",\"skipped\":[],\"spec_source\":null}"
    f22_log "[serving-path] EXPLICIT $F22_SERVING_PATH (F22_SERVING_PATH set by caller)"
    return 0
  fi
  if declare -p F22_LAUNCH_CELLS >/dev/null 2>&1; then          # declared, possibly EMPTY (a job that stands up no target)
    if [ "${#F22_LAUNCH_CELLS[@]}" -ge 1 ]; then
      f22_select_auto --cells "${F22_LAUNCH_CELLS[@]}" --cells-source "F22_LAUNCH_CELLS"
    else
      f22_select_auto --cells --cells-source "F22_LAUNCH_CELLS (declared empty: this job stands up no target)"
    fi
    return 0
  fi
  if [ -n "${F22_LAUNCH_MANIFEST:-}" ]; then
    local m="$F22_LAUNCH_MANIFEST"; case "$m" in /*) ;; *) m="${REPO_DIR:?f22_select_serving_path needs REPO_DIR}/$m";; esac
    [ -f "$m" ] || f22_fatal "F22_LAUNCH_MANIFEST=$F22_LAUNCH_MANIFEST does not exist (caller $caller) — the kind-aware serving-path default reads the launch's cells from it; set F22_SERVING_PATH explicitly to bypass"
    f22_select_auto --manifest "$m" --cells-source "manifest:$F22_LAUNCH_MANIFEST"
    return 0
  fi
  if [ "${F22_SERVING_PATH_DEFER:-}" = 1 ] || [ -n "${F22_WAVE_ROOT:-}" ]; then
    F22_SERVING_PATH_MODE=deferred; F22_SERVING_PATH_SELECTION_JSON=""
    f22_log "[serving-path] DEFERRED to the standup: this job $( [ -n "${F22_WAVE_ROOT:-}" ] && echo "owns its wave root ($F22_WAVE_ROOT)" || echo "set F22_SERVING_PATH_DEFER=1" ) — f22_bind_serving_path <manifest> (or f22_standup, from \$MANIFEST) selects from the launch's own manifest; explicit F22_SERVING_PATH would override"
    return 0
  fi
  f22_fatal "serving-path selection: $caller declared no cell source — the kind-aware default (2026-09-15) needs the launch's cells: before f22_init set F22_LAUNCH_MANIFEST=<the launch's manifest> (the campaign scripts: \$(f22_full22_manifest KEY VIEW)) or F22_LAUNCH_CELLS=(<harness ids>; empty = this job stands up no target), defer (F22_SERVING_PATH_DEFER=1, or own the wave root via F22_WAVE_ROOT) and bind from the launch's manifest with f22_bind_serving_path, or set F22_SERVING_PATH explicitly. The full22 campaign manifest is NOT assumed from KEY / VIEW: a caller with its own manifest would be misread"
}

f22_bind_serving_path(){   # $1 = the launch's ACTUAL manifest (relative to the checkout or absolute) -> a DEFERRED selection bound: path, engine, WAVE_ROOT when unbound
  local m="${1:?f22_bind_serving_path needs the launch manifest}"
  [ "${F22_SERVING_PATH_MODE:-}" = deferred ] || f22_fatal "f22_bind_serving_path: the serving path is not deferred (mode ${F22_SERVING_PATH_MODE:-unset}) — it was bound at f22_init"
  case "$m" in /*) ;; *) m="${REPO_DIR:?f22_bind_serving_path needs REPO_DIR}/$m";; esac
  [ -f "$m" ] || f22_fatal "f22_bind_serving_path: manifest $1 does not exist — the kind-aware serving-path default reads the launch's cells from it"
  f22_select_auto --manifest "$m" --cells-source "manifest:$1"
  f22_bind_engine
  if [ -z "${WAVE_ROOT:-}" ]; then   # a job without its own root: the campaign root follows the path, exactly as f22_init derives it
    WAVE_ROOT="${F22_WAVE_ROOT:-$SHARED/exp/sdfbase_full22$( [ "$F22_SERVING_PATH" = lens ] || echo _native)}"; mkdir -p "$WAVE_ROOT"
  fi
  f22_log "serving path bound: $F22_SERVING_PATH engine=$TARGET_ENGINE wave_root=$WAVE_ROOT (from manifest $1)"
}

# --- cluster + env ------------------------------------------------------------------------------
f22_init(){   # $1 = wave sha (the checkout $SHARED/sisteer-eval-repo/<sha>)
  WAVE_SHA="${1:?wave sha}"; WAVE=full22
  # The cluster's shared root and label are the site's: the front door passes SISTEER_SHARED_ROOT and SISTEER_CLUSTER_LABEL
  # (+ SISTEER_EVAL_REPO_ROOT, SISTEER_HF_HOME) from the cluster block as command-line env prefixes on the sbatch line, and a
  # hand submission exports them in the submitting shell. Nothing is guessed from which storage paths exist: without them the
  # job stops here by name. The emitter reads SISTEER_REGISTRY_SUBSTRATE / _CLUSTER_LABEL from the same prefix.
  if [ -n "${SISTEER_SHARED_ROOT:-}" ]; then SHARED="$SISTEER_SHARED_ROOT"; CLUSTER="${SISTEER_CLUSTER_LABEL:?SISTEER_SHARED_ROOT is set without SISTEER_CLUSTER_LABEL}"
  else f22_fatal "SISTEER_SHARED_ROOT is unset: a job carries the cluster shared storage root and its label (SISTEER_SHARED_ROOT, SISTEER_CLUSTER_LABEL), which the front door passes from the site config"; fi
  if [ -n "${SISTEER_HF_HOME:-}" ]; then export HF_HOME="$SISTEER_HF_HOME" HF_TOKEN_PATH="$SHARED/hf_token" HUGGINGFACE_TOKEN_PATH="$SHARED/hf_token"
  else
    [ -r "$SHARED/scripts/hf_env.sh" ] || f22_fatal "no $SHARED/scripts/hf_env.sh (install the shared helper)"
    source "$SHARED/scripts/hf_env.sh"
  fi; export HF_HUB_OFFLINE=1
  # The G0DM0D3 templates (sisteer.eval.attacks_godmode, rendered by every grok420 job) are not distributed with the repository:
  # 00_setup.sh --what third_party installs the verified file under the shared root, and compute nodes have no egress, so a job
  # reads that copy with fetching off. Set here on both branches above: a configured site's SISTEER_HF_HOME skips hf_env.sh, and
  # an installed hf_env.sh may predate its own copy of these two exports. A value already in the job's environment stands.
  export SISTEER_THIRD_PARTY_DIR="${SISTEER_THIRD_PARTY_DIR:-$SHARED/third_party}" SISTEER_THIRD_PARTY_OFFLINE="${SISTEER_THIRD_PARTY_OFFLINE:-1}"
  # NCCL's socket interface is the site's to choose (its nccl.conf, else NCCL's own detection): a job never exports
  # NCCL_SOCKET_IFNAME, because an exported name overrides the site's setting and one the node lacks fails the first collective
  export NCCL_NVLS_ENABLE=0
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
  export UV_CACHE_DIR="$SHARED/.cache/uv" UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR="$SHARED/.uv-pythons"
  export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled WANDB_DISABLED=true WANDB_CONSOLE=off PYTHONUNBUFFERED=1
  # THE WAVE'S REDUCTION MODE: stacked, on BOTH sides. The standup pins it into the worker env
  # before the lens loads and REFUSES unless the workers attest it; the drivers (a client-side
  # contract elsewhere treats an UNSET value as per_row) get it exported here too.
  export SISTEER_LENS_STACKED_REDUCTIONS=1; REDUCTION_MODE=stacked
  REPO_DIR="${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA"
  [ -d "$REPO_DIR/src/sisteer" ] || f22_fatal "no checkout at $REPO_DIR (README: create it from the mirror)"
  cd "$REPO_DIR" || exit 1
  local head; head=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
  [ "$head" = "$WAVE_SHA" ] || f22_fatal "checkout HEAD $head != $WAVE_SHA"
  git diff --quiet HEAD -- || f22_fatal "dirty checkout $REPO_DIR"
  # THE SERVING PATH (native-serving-default; KIND-AWARE default, the rule of 2026-09-15):
  # F22_SERVING_PATH = lora_native (every LoRA cell of the view on the plugin-free CUDA-graphs-on native multi-LoRA
  # engine through mt_serving_up.py --serving-path lora_native; PSR / vector cells SERVED on its residual slot
  # (native-psr-serving); a stacked cell or a position-masked LoRA fit SKIPPED BY NAME, recorded in
  # cells_served.json) | lens (the stacked lens standup, byte for byte). An EXPLICIT value set by the caller always
  # wins — a batch job inherits the submitter's exported environment (never --export=), so `F22_SERVING_PATH=lens
  # sbatch ...` selects the lens. Set NONE, f22_select_serving_path picks the path from the launch's steered cells BY
  # KIND (native when any steered cell is natively servable, lens when every steered cell is a lens-only kind, the
  # entry-point default when nothing is steered) and logs ONE recoverable `[serving-path]` line. Resolved HERE, once,
  # before anything reads it: WAVE_ROOT below follows the path, TARGET_ENGINE is the engine every driver asserts from
  # the AUTHORITATIVE stamps (--require-engine), and f22_standup passes --serving-path from it. An engine pin riding
  # F22_STANDUP_EXTRA instead (the launch-time argv hook) is refused here BY NAME, before the selection is recorded.
  f22_standup_extra_inherited_check
  f22_select_serving_path
  # THE ROSTER (F22_ROSTER, above): present in this checkout, exported so every child reader and sub-script sees the
  # same selection, logged beside the engine — a job over a certification-only roster says so on its first line
  [ -f "$F22_ROSTER" ] || f22_fatal "no roster at $F22_ROSTER (F22_ROSTER; a path relative to $REPO_DIR or absolute; the production roster is $F22_ROSTER_PRODUCTION)"
  export F22_ROSTER
  # F22_WAVE_ROOT: an experiment that reuses this body (math500-thinking-on-full22) sets its OWN
  # wave root before f22_init; unset = the campaign's — the LENS root byte-identical to before, the NATIVE root its
  # own sibling `sdfbase_full22_native` (defence in depth, ONE variable — every
  # wave dir (serving/, transcripts_results/, smoke/, the Petri REMOTE_OUT) follows it; the CHECK is each
  # writer's pre-write engine binding, not this path)
  if [ "$F22_SERVING_PATH_MODE" = deferred ]; then
    # the path is unknown until f22_bind_serving_path (or f22_standup) binds it from the launch's manifest: no engine yet, and the
    # wave root only when the caller owns one (unset otherwise, so a premature use is loud under set -u)
    if [ -n "${F22_WAVE_ROOT:-}" ]; then WAVE_ROOT="$F22_WAVE_ROOT"; mkdir -p "$WAVE_ROOT"; else unset WAVE_ROOT; fi
  else
    f22_bind_engine
    WAVE_ROOT="${F22_WAVE_ROOT:-$SHARED/exp/sdfbase_full22$( [ "$F22_SERVING_PATH" = lens ] || echo _native)}"; mkdir -p "$WAVE_ROOT"
  fi
  export FIGS_DIR="$SHARED/exp/figs/${SLURM_JOB_NAME:-f22-manual}"; mkdir -p "$FIGS_DIR"
  f22_registry_state running "${SLURM_JOB_STATE:-RUNNING}" || true   # registry: job start, from the job's own env
  f22_log "cluster=$CLUSTER shared=$SHARED sha=$WAVE_SHA serving_path=${F22_SERVING_PATH:-deferred} serving_path_mode=$F22_SERVING_PATH_MODE engine=${TARGET_ENGINE:-deferred} reduction=$REDUCTION_MODE roster=$F22_ROSTER wave_root=${WAVE_ROOT:-unbound} node=$(hostname) job=${SLURM_JOB_ID:-manual} gpus=${CUDA_VISIBLE_DEVICES:-none}"
}

# the wave-level STOP sentinel: a deliberate stop of the whole wave (never resubmit after it)
f22_stop_requested(){ [ -e "$WAVE_ROOT/STOP" ]; }

# THE IN-BODY START-GRANT HOOK of the twin-submit protocol: every
# twinnable sbatch calls this as its FIRST executable line after the bootstrap `source` — BEFORE f22_init and before
# f22_resolve_config's self-registration — so a losing twin leaves having loaded and registered NOTHING (two twins are two run
# identities: jobplan keys `job_name@job_id`). A no-op unless SISTEER_ATTEMPT_ID rides the job's env prefix, so every existing
# job is byte-for-byte unchanged; the logic lives in reproduce/lib/attempt_wait.sh of THIS checkout.
f22_attempt_wait(){
  [ -n "${SISTEER_ATTEMPT_ID:-}" ] || return 0
  local hook; hook="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/reproduce/lib/attempt_wait.sh"
  [ -r "$hook" ] || f22_fatal "SISTEER_ATTEMPT_ID=$SISTEER_ATTEMPT_ID is set but this checkout has no $hook (twin submit needs the hook on the mirrored sha)"
  source "$hook"
}

# figure mirror (nothing here draws figures; the RESULT line rides out through it)
f22_mirror(){ "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" 2>/dev/null || true; }
f22_result(){
  echo "$(date -Is) job=${SLURM_JOB_ID:-manual} name=${SLURM_JOB_NAME:-manual} $*" >> "$FIGS_DIR/RESULT"; f22_mirror
  f22_registry_result "$@"; local crc=$?
  if [ "$crc" -ne 0 ]; then
    # closer-fix-2: the closer refused on a job whose own work finished — before this, the trap returned 0, Slurm said
    # COMPLETED, the `running` note sat outside the grading worklist and nothing recovered (this morning's 36 stuck runs, now with a label).
    # The refusal is the job's OUTCOME: leave with exit $F22_EXIT_CLOSER_REFUSED (this runs in the EXIT trap, after the RESULT line, the
    # mirror and the final push), so sacct carries it and the funnel's proof answers `reclose` — the fix — instead of a resubmit of the eval.
    echo "$(date -Is) job=${SLURM_JOB_ID:-manual} name=${SLURM_JOB_NAME:-manual} CLOSER_REFUSED rc=$crc exit=$F22_EXIT_CLOSER_REFUSED recovery=submit_twin.py reclose --job ${SLURM_JOB_ID:-manual}"
    f22_log "closer refused (rc $crc): leaving with exit $F22_EXIT_CLOSER_REFUSED — the recovery is \`submit_twin.py reclose --job ${SLURM_JOB_ID:-manual}\` (06's --resubmit walk submits it)"
    exit "$F22_EXIT_CLOSER_REFUSED"
  fi
}
# registry: the RESULT line's first word is the outcome (FAILED / PREEMPTED / CANCELLED → a terminal
# run_state with that scheduler state); a line carrying `rc=<non-zero>` (target_attacks.sbatch's "attacks <tag> <group> rc=N") is a
# failure too — before 2026-09-30 it took the clean branch and the FAILED 2:0 targets' runs stayed `running`. A clean exit runs the
# job-side closer (scripts/sdfbase/full22/registry_close_stores.py): no jailbreak_run family has a store hook that files
# `generated`, so the closer judges every registered run's store against the registration's plan and files store_uri + generated
# (refusing incomplete stores by name; idempotent), THEN the final push.
f22_registry_close_stores(){
  [ -n "${SISTEER_REGISTRY_SPOOL:-}" ] || return 0
  local pend="$SHARED/registry/pending/${SLURM_JOB_ID:-}.json"
  [ -f "$pend" ] || { echo "[registry] close: no pending file $pend — nothing registered to close" >&2; return 0; }
  local -a args=(--pending "$pend" --spool "$SISTEER_REGISTRY_SPOOL" --code-sha "${WAVE_SHA:-}")
  [ -n "${WAVE_ROOT:-}" ] && args+=(--wave-root "$WAVE_ROOT")
  [ -n "${TAG:-}" ] && args+=(--tag "$TAG")
  [ -n "${SLURM_JOB_NAME:-}" ] && args+=(--job-name "$SLURM_JOB_NAME")
  [ -n "${NSHARDS:-}" ] && args+=(--nshards "$NSHARDS")
  [ -n "${SISTEER_REGISTRY_PUSH:-}" ] && args+=(--push-script "$SISTEER_REGISTRY_PUSH")
  local py="${F22_PY:-python3}" rc=0 out="" why="" suffix=""
  out=$(PYTHONPATH="${REPO_DIR:-$PWD}/src${PYTHONPATH:+:$PYTHONPATH}" "$py" "${REPO_DIR:-$PWD}/scripts/sdfbase/full22/registry_close_stores.py" "${args[@]}" 2>&1) || rc=$?
  [ -n "$out" ] && printf '%s\n' "$out" | sed 's/^/[registry] /'
  [ "$rc" -eq 0 ] && return 0
  # LOUD AND ON THE RECORD: the `|| true` that stood here swallowed the closer's refusal — Slurm
  # reported success while the runs never advanced to generated or the grading worklist. The job's own rc stands (this runs in the
  # EXIT trap and the final push must still follow), but the refusal is named three times: in the job log by name, on the RESULT
  # surface (the figure mirror reads it), and on the registry as one `running` state event over the job's registered runs whose
  # note carries the closer's first REFUSED line (the reducer is monotone — a run already generated keeps its lifecycle; the event
  # is the new information: WHY the run never closed).
  why=$(printf '%s\n' "$out" | grep -m1 -E "REFUSED|Traceback|Error" | cut -c1-300)
  [ -n "${F22_PY:-}" ] || suffix=" (F22_PY unset — the bare interpreter, not the driver venv)"
  echo "[closer] REFUSED rc=$rc under $py$suffix: job ${SLURM_JOB_ID:-manual}'s registered runs stay as they were${why:+ — $why}" >&2
  if [ -n "${FIGS_DIR:-}" ] && [ -d "$FIGS_DIR" ]; then
    echo "$(date -Is) job=${SLURM_JOB_ID:-manual} name=${SLURM_JOB_NAME:-manual} CLOSER_REFUSED rc=$rc interpreter=$py" >> "$FIGS_DIR/RESULT"; f22_mirror
  fi
  f22_registry_closer_refused "$rc" "$py" "$why"
  # closer-fix-2: the RECIPE the reclose job runs from — every closer argument captured from THIS job's environment
  # (nothing re-derived later) — and the refusal returned to the caller: f22_registry_result carries it to f22_result, which makes it the
  # job's exit. The watchers' .out and .err tails both show the line (stderr above, stdout here).
  echo "[closer] REFUSED rc=$rc under $py$suffix — job ${SLURM_JOB_ID:-manual} leaves with exit $F22_EXIT_CLOSER_REFUSED; recovery: submit_twin.py reclose --job ${SLURM_JOB_ID:-manual}"
  f22_closer_recipe_write "$rc" "$py" "$why"
  return "$rc"
}
f22_closer_recipe_write(){   # $1 = the closer's rc, $2 = its interpreter, $3 = its first REFUSED line -> $SHARED/registry/pending/<job>.closer_refused.json
  # closer-fix-5: every value comes from THIS shell's variables, passed on the python command line — f22_init
  # assigns SHARED / WAVE_ROOT / WAVE_SHA / REPO_DIR / CLUSTER as plain (unexported) shell variables and the sbatches set TAG / NSHARDS the same
  # way, so a Python reading os.environ saw none of them: every real recipe carried wave_root='' and pending='/registry/pending/<job>.json' (the
  # funnel's STOP fence skipped, the reclose job dying on the bogus path before its trap, a receipt-less death the durable cap counted). A job
  # whose wave root is unbound here is one a reclose cannot honour (no STOP fence, no --wave-root): refuse to write, by name — the funnel then
  # refuses the reclose for want of a recipe, and a human closes by hand.
  local job="${SLURM_JOB_ID:-manual}" p="$SHARED/registry/pending/${SLURM_JOB_ID:-manual}.closer_refused.json"
  if [ -z "${WAVE_ROOT:-}" ]; then
    echo "[closer] NOT writing the reclose recipe at $p: WAVE_ROOT is unbound in this job (F22_WAVE_ROOT=${F22_WAVE_ROOT:-unset}) — a reclose could honour neither the wave's STOP nor --wave-root; close by hand under the driver venv" >&2
    return 1
  fi
  R_RC="$1" R_PY="$2" R_WHY="$3" R_PATH="$p" R_JOB="$job" R_JOB_NAME="${SLURM_JOB_NAME:-manual}" R_PENDING="$SHARED/registry/pending/$job.json" \
  R_SPOOL="${SISTEER_REGISTRY_SPOOL:-}" R_PUSH="${SISTEER_REGISTRY_PUSH:-}" R_CODE_SHA="${WAVE_SHA:-}" R_WAVE_ROOT="$WAVE_ROOT" R_TAG="${TAG:-}" \
  R_NSHARDS="${NSHARDS:-}" R_REPO_DIR="${REPO_DIR:-}" R_CLUSTER="${CLUSTER:-}" R_FIGS_DIR="${FIGS_DIR:-}" python3 -S - <<'PY' || echo "[closer] could not write the reclose recipe at $p" >&2
import json, os, time
p = os.environ["R_PATH"]; env = os.environ.get
try:
    with open(p) as fh:
        old = json.load(fh)
except (OSError, ValueError):
    old = {}
rec = {"job_id": env("R_JOB", "manual"), "job_name": env("R_JOB_NAME", "manual"), "rc": int(os.environ["R_RC"]), "interpreter": os.environ["R_PY"], "why": os.environ["R_WHY"],
       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "pending": os.environ["R_PENDING"],
       "spool": env("R_SPOOL", ""), "push_script": env("R_PUSH", ""), "code_sha": env("R_CODE_SHA", ""),
       "wave_root": os.environ["R_WAVE_ROOT"], "tag": env("R_TAG", ""), "nshards": env("R_NSHARDS", ""), "repo_dir": env("R_REPO_DIR", ""),
       "cluster": env("R_CLUSTER", ""), "figs_dir": env("R_FIGS_DIR", ""), "recloses": list(old.get("recloses") or [])}
tmp = p + ".tmp." + str(os.getpid())
with open(tmp, "w") as fh:
    json.dump(rec, fh, indent=1, sort_keys=True); fh.write("\n"); fh.flush(); os.fsync(fh.fileno())
os.replace(tmp, p)
PY
}
f22_registry_closer_refused(){   # $1 = the closer's rc, $2 = its interpreter, $3 = its first REFUSED line -> `running` with the note PER REFUSED RUN, under each run's store lock
  # The statement goes to the runs the closer REFUSED — read from the closer's sidecar <pending>.refused_run_ids.json — and never
  # to the job's whole registered set (`--pending-dir … --job` restated `running` over the twins the same closer had just superseded:
  # 888525's 170aa5a1 / d285201a folded running/current again). No sidecar, no statement: say so loudly instead.
  # FENCED: the statement used to go through the bare state emitter, per run id, AFTER the closer had released
  # the store lock — so a climb task whose close refused an incomplete store, a peer task then closing the completed store and the store512
  # retiring the shared run, and this hook stating `running` last, folded the retired run `running` and current again (the reducer applies
  # `running` over a lateral `superseded`). The statement now goes through the closer's refusal-note mode (registry_close_stores.py
  # --refusal-note; the rule and its test live there): per listed run, under the lock the closes take, filed only when no close has decided the
  # run and its last word is not a task's exit; a run the pending file does not register is never stated for. Every line on STDOUT (the RESULT
  # hook drops stderr); the mode's refusal is loud and the job's rc stands.
  f22_registry_canary_guard "closer refusal not recorded" && return 0
  [ -n "${SISTEER_REGISTRY_SPOOL:-}" ] || return 0
  local pend="$SHARED/registry/pending/${SLURM_JOB_ID:-none}.json" side n=0
  if [ -f "$pend" ]; then export SISTEER_REGISTRY_RUN_IDENTITY="$pend"; fi
  side="$pend.refused_run_ids.json"
  if [ ! -s "$side" ]; then
    echo "[registry] closer refused (rc $1) but no refused-run list at $side — stating nothing (D2100 (b): never the job's whole run set)"; return 0
  fi
  n=$(python3 -c 'import json,sys; print(sum(1 for r in json.load(open(sys.argv[1])).get("refused_run_ids", []) if isinstance(r, str) and r))' "$side" 2>/dev/null) \
    || { echo "[registry] closer refused (rc $1) but $side is unreadable — stating nothing"; return 0; }
  local -a args=(--pending "$pend" --spool "$SISTEER_REGISTRY_SPOOL" --code-sha "${WAVE_SHA:-}" --refusal-note "closer REFUSED rc=$1 under $2${3:+: $3}" \
                 --node "${SLURMD_NODENAME:-$(hostname)}")
  [ -n "${WAVE_ROOT:-}" ] && args+=(--wave-root "$WAVE_ROOT")
  [ -n "${TAG:-}" ] && args+=(--tag "$TAG")
  [ -n "${SLURM_JOB_NAME:-}" ] && args+=(--job-name "$SLURM_JOB_NAME")
  [ -n "${SISTEER_REGISTRY_PUSH:-}" ] && args+=(--push-script "$SISTEER_REGISTRY_PUSH")
  local py="${F22_PY:-python3}" rc=0 out=""
  out=$(PYTHONPATH="${REPO_DIR:-$PWD}/src${PYTHONPATH:+:$PYTHONPATH}" "$py" "${REPO_DIR:-$PWD}/scripts/sdfbase/full22/registry_close_stores.py" "${args[@]}" 2>&1) || rc=$?
  [ -n "$out" ] && printf '%s\n' "$out" | sed 's/^/[registry] /'
  if [ "$rc" -eq 0 ]; then
    echo "[registry] closer refusal note: $n listed refused run(s) of job ${SLURM_JOB_ID:-manual} decided by the closer's refusal-note mode under $py (the refused-run list; D2100 (b)) — \`running\` filed per run under its store lock, never over a close"
  else
    echo "[registry] closer refusal note NOT filed for job ${SLURM_JOB_ID:-manual} (rc $rc under $py): the refused runs' lifecycle was not touched; the refusal is on the job log, the RESULT surface and the reclose recipe"
  fi
}
f22_registry_result(){
  f22_registry_canary_guard "result not filed" && return 0
  # the STANDALONE `rc=<digits>` token — never `agentic_rc=` / `em8q_rc=` (misalign_proxy.sbatch's RESULT is "misalign … rc=$RC agentic_rc=$RC_A
  # em8q_rc=$RC_E": taking the LAST `rc=` filed FAILED:rc=0 when agentic failed and em8q passed); a nonzero one is the failure
  local rc_word="" tok crc=0
  for tok in ${*:-}; do case "$tok" in rc=0) ;; rc=[0-9]*) [[ "$tok" =~ ^rc=[0-9]+$ ]] && rc_word="$tok";; esac; done
  case "${1:-}" in
    FAILED*)               f22_registry_state failed "${1%% *}" final 2>/dev/null || true ;;
    PREEMPTED*|CANCELLED*) f22_registry_state cancelled "${1%% *}" final 2>/dev/null || true ;;
    *) if [ -n "$rc_word" ]; then f22_registry_state failed "FAILED:$rc_word" final 2>/dev/null || true; return 0; fi
       f22_registry_close_stores; crc=$?      # closer-fix-2: a refusal is carried to f22_result (the job's exit) AFTER the final push below
       [ -n "${SISTEER_REGISTRY_SPOOL:-}" ] && PYTHONPATH="${REPO_DIR:-$PWD}/src${PYTHONPATH:+:$PYTHONPATH}" python3 -S -m sisteer.registry.emit_cli push \
         --pending-dir "$SHARED/registry/pending" --job "${SLURM_JOB_ID:-}" --final 2>/dev/null || true
       # an array task's clean exit is recorded on the task after its close ran; the last task to exit reconciles the shared runs (a no-op otherwise)
       if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then f22_registry_array_task_event completed COMPLETED || true; fi
       return "$crc" ;;
  esac
}

# --- config / manifest / bank view --------------------------------------------------------------
f22_arm_serving_policy(){   # $1 = arm -> F22_ARM_SERVING_POLICY (lora_native | lens | "" = the arm declares none), exported; FATAL by name on a contradiction
  # THE ARM'S SERVING POLICY (arditi-qwen-all-positions, shell-half review Finding 2). An arm may declare `serving_path:` in
  # configs/sdfbase_full22_arms.yaml -- arditi_allpos does (lora_native: the spec requires that object on the graph-captured
  # native residual slot in EVERY leg; the lens WOULD serve a vector_all blob, so a leg on it completes and files lens numbers
  # under the arm's name, a different measurement rather than a failure). The path this launch RESOLVED at f22_init
  # (explicit F22_SERVING_PATH, which wins over the kind-aware default, or the auto selection) is held to it here, BEFORE the
  # pair-existence fatal in f22_resolve_config and before arm_cell_check's per-target refusals, so a lens request is never
  # masked by "missing config" or "no operating point yet". ONE implementation: arm_cell_check.serving_policy_problems
  # (--policy here; --resolve in f22_arm_bank_view re-asserts it), read from the table -- never a shell predicate on arm names.
  # Runs on the node's stdlib python3 + PyYAML (F22_SELECT_PY names another), exactly as f22_select_auto does at init: this is
  # called from f22_resolve_config, before f22_driver_venv exists.
  local arm="${1:?arm}" py out
  py="${F22_SELECT_PY:-python3}"
  out=$(cd "${REPO_DIR:?f22_arm_serving_policy needs REPO_DIR}" && "$py" scripts/sdfbase/full22/arm_cell_check.py --arms configs/sdfbase_full22_arms.yaml \
          --arm "$arm" --policy --requested-serving-path "${F22_SERVING_PATH:-}" --serving-path-mode "${F22_SERVING_PATH_MODE:-}" 2>&1) \
    || f22_fatal "arm $arm: serving policy REFUSED — $out"
  # read ONLY the verdict line (a stray interpreter warning on the merged stream must not read as a policy), as f22_select_auto does
  out=$(printf '%s\n' "$out" | grep -E '^(lora_native|lens|-)$' | tail -1)
  [ -n "$out" ] || f22_fatal "arm $arm: arm_cell_check.py --policy printed no verdict (want lora_native | lens | -)"
  [ "$out" = - ] && out=""
  F22_ARM_SERVING_POLICY="$out"; export F22_ARM_SERVING_POLICY
  if [ -n "$F22_ARM_SERVING_POLICY" ]; then
    f22_log "arm $arm serving policy: $F22_ARM_SERVING_POLICY (this launch resolved ${F22_SERVING_PATH:-nothing yet}, ${F22_SERVING_PATH_MODE:-unset})"
  else
    f22_log "arm $arm serving policy: none declared — the launch's own path (${F22_SERVING_PATH:-unresolved}, ${F22_SERVING_PATH_MODE:-unset}) stands"
  fi
}

f22_resolve_config(){   # $1 = model key, $2 = view, $3 = arm ("" = the wave's own cells)  -> KEY VIEW TAG ARM CFG MANIFEST
  KEY="${1:?key}"; VIEW="${2:?view}"; ARM="${3:-}"
  TAG="$KEY$( [ "$VIEW" = main ] || echo "_$VIEW" )"
  # AN ARM (mt_arms4096, 2026-09-13; ported from mt-arms-llama): a PUBLISHED STEERING BASELINE the
  # goalset driver serves from ITS OWN LANE'S BANK (configs/sdfbase_full22_arms.yaml), through the
  # per-arm config/manifest pair make_arm_configs.py derives from THIS (KEY, VIEW)'s paper pair —
  # explicit_cells [none, <arm cell>], defenses_dir the arm's view, everything else byte-identical as
  # data, private_run the paper view's so the rows pair with its `none` anchor. The wave's own
  # config/manifest are untouched, and so is the wave bank (f22_arm_bank_view never reads it).
  # ADDITIVE: an EMPTY arm is byte-identically today's behaviour, which is every existing caller.
  #
  # THE ARM NAME IS NOT A CLOSED LIST HERE, deliberately (arditi-qwen-all-positions A2'). The port's
  # original spelled `arditi|alphasteer` in this case, so a third arm had to be added in two places
  # and the shell was a second, drifting copy of the arms table. The AUTHORITY is that table, read by
  # arm_cell_check.py in f22_arm_bank_view below — which runs with the DRIVER VENV's python, and this
  # function runs before f22_driver_venv, so it cannot read YAML itself. What it can do is refuse a
  # name that is not a plain arm token, and refuse an arm whose generated pair is absent, naming the
  # generator. An arm that is not in the table gets past HERE and is refused THERE, by name.
  case "$ARM" in
    "") CFG="configs/sdfbase_full22_${TAG}.yaml"; MANIFEST="configs/capeval/manifests/petri_full22_${TAG}.yaml";;
    *[!a-z0-9_]*|[!a-z]*) f22_fatal "arm '$ARM' is not a plain arm token (lower-case, digits, underscore; see configs/sdfbase_full22_arms.yaml)";;
    *) CFG="configs/sdfbase_full22_${TAG}__${ARM}.yaml"; MANIFEST="configs/capeval/manifests/petri_full22_${TAG}__${ARM}.yaml"
       # the arm's SERVING POLICY, held against the path f22_init resolved -- BEFORE the pair-existence fatal below, so a lens
       # request for a native-only arm is refused as such and never masked by "missing config" (Finding 2)
       f22_arm_serving_policy "$ARM";;
  esac
  [ -f "$CFG" ] && [ -f "$MANIFEST" ] || f22_fatal "missing $CFG or $MANIFEST${ARM:+ (generate: scripts/sdfbase/full22/make_arm_configs.py --arm $ARM)}"
  # TEXT-ONLY SERVING for the multimodal targets (2026-09-07, lesson of the qwen OOMs 421951 / 422285 and the
  # standing rule for glm46vflash): the manifest MUST carry --limit-mm-per-prompt {"image":0,"video":0}, or the
  # engine's multimodal profiling (dummy max-size images sized off S x L) is OOM-killed by the 200G host cgroup
  # after the venvs built. The generator pins it (make_full22_configs.MODEL_SERVING) and the wave test checks it;
  # this is the job-side belt: refuse before a GPU minute is spent on a manifest that lost the flag.
  case "$KEY" in qwen35_4b|glm46vflash)
    grep -q -- '--limit-mm-per-prompt' "$MANIFEST" || f22_fatal "$MANIFEST serves multimodal $KEY without --limit-mm-per-prompt (text-only pin) — regenerate (make_full22_configs.py)";;
  esac
  # KV COVERAGE for glm46vflash (2026-09-08, cluster B 78774 @16384 and 78837 @2048): its certified budget (786,432
  # batched tokens, measured at an 8192 window) is 3,744 tokens MORE than the KV cache vLLM builds at the other windows
  # under gpu_memory_utilization 0.9 (782,688 tokens: "only 47.77-way / 382.17-way concurrency at full length is
  # actually KV-backed"), and the standup rightly refuses an under-covered identity geometry. The refusal names the fix:
  # raise utilization, keep identity (S x L is the wave's invariant, unchanged). 0.92 adds ~2.9 GiB = ~75k KV tokens on a
  # 143.7 GiB card whose preflight saw 143.1 GiB free; recorded per standup (standup_jobs.jsonl gpu_mem_util). ONE place
  # for every window (the 4864 static window passed at 0.9 by 416 tokens — no margin to rely on).
  case "$KEY" in glm46vflash) GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}";; esac
  f22_registry_register || true   # registry: the job registers its own runs now that (KEY, VIEW, CFG) are bound
}

# --- model facts a standup must attest ------------------------------------------------------------------------
# A hybrid-recurrent target (geometry.HYBRID_MODEL_KEYS: the Mamba2 / gated-deltanet families — nemotron3nano4b,
# qwen35_*) pre-allocates recurrent state PER SLOT, so its honest width is budget // L capped at HYBRID_SEQS_CAP
# (ModelBudget.seqs_at): the ATTESTED max_num_seqs must sit under the cap, and the workers' scheduler reports must
# carry the resolved mamba_ssm_cache_dtype — proof the engine built the hybrid cache surface at all (on the pinned
# 0.25.1 config surface it is a string, "auto" unless a launch composed one). Read from the AUTHORITATIVE stamps
# (steering_attestation.json, named by endpoint.json), never from the env; recorded for every target, REFUSED only
# for a hybrid target whose facts are missing or over the cap.
# NEMOTRON FUSED-KERNEL BACKEND: a job that
# exports F22_REQUIRE_NEMO_BACKEND=1 (every math500 job, m5t_common.sh) additionally requires, for nemotron3nano4b,
# the backend attestation nemo_backend_probe.sbatch wrote — fused on every Mamba symbol, SM duty AND tok/s each
# >= F22_NEMO_DUTY_RATIO x a same-GPU peer's under the identical burst (the served path runs eager, so an
# absolute duty floor calibrates nothing), and for THIS serving venv's requirements (SERVE_REQS_SHA) — at
# F22_NEMO_ATTESTATION (default $WAVE_ROOT/nemo_backend/mamba_backend_attestation.json). The reader RE-DERIVES the
# rule from the recorded numbers (never trusts `admitted` alone). Unset (the campaign's own jobs) = unchanged behaviour.
F22_NEMO_DUTY_RATIO="${F22_NEMO_DUTY_RATIO:-0.5}"
f22_model_facts(){   # $1 = endpoint.json, $2 = attested max_num_seqs  -> MODEL_FACTS (JSON)  (needs SPY, KEY, REPO_DIR)
  local nemo_att="${F22_NEMO_ATTESTATION:-$WAVE_ROOT/nemo_backend/mamba_backend_attestation.json}"
  MODEL_FACTS=$(cd "$REPO_DIR" && "$SPY" - "$KEY" "$2" "$1" "${F22_REQUIRE_NEMO_BACKEND:-0}" "$nemo_att" "${SERVE_REQS_SHA:-}" "$F22_NEMO_DUTY_RATIO" <<'PY'
import json, os, sys
sys.path.insert(0, "src")
from sisteer.capeval.serving.geometry import HYBRID_SEQS_CAP, is_hybrid
key, att_s, ep = sys.argv[1], int(sys.argv[2]), json.load(open(sys.argv[3]))
require_nemo, nemo_att_path, serve_reqs_sha, duty_ratio = sys.argv[4] == "1", sys.argv[5], sys.argv[6], float(sys.argv[7])
# the AUTHORITATIVE stamps: a native standup names serving_attestation.json (the strict lora_native record), a
# lens one steering_attestation.json — both carry scheduler_geometry.worker_reports[*].mamba_ssm_cache_dtype
native = ep.get("engine") == "lora_native"
att = json.load(open(ep.get("serving_attestation") or ep["steering_attestation"]))
reps = ((att.get("scheduler_geometry") or {}).get("worker_reports") or [])
dtypes = sorted({str(r.get("mamba_ssm_cache_dtype")) for r in reps})
hybrid = is_hybrid(key)   # BOTH hybrid tuples (the paper's and the onboarded models' alike — the facts line once read the paper's tuple only, 2026-10-02)
problems = []
if hybrid and att_s > HYBRID_SEQS_CAP:
    problems.append(f"hybrid target {key} attests max_num_seqs {att_s} > HYBRID_SEQS_CAP {HYBRID_SEQS_CAP}")
if hybrid and (not reps or dtypes in ([], ["None"])):
    problems.append(f"hybrid target {key}: no worker scheduler report carries mamba_ssm_cache_dtype ({dtypes}) — the hybrid cache surface is unattested")
nemo_backend = None
if require_nemo and key == "nemotron3nano4b" and native:
    # THE NATIVE ENGINE (native-serving-default plan §1.2): the fused-kernel fact is the standup's OWN live attestation
    # over the ported sisteer_mamba_backend_report RPC (endpoint.mamba_backend_attestation, judged against the reviewed
    # contract; a failing one never reaches endpoint.json) — not the lens probe job's duty certificate
    nemo_backend = ep.get("mamba_backend_attestation")
    if not (isinstance(nemo_backend, dict) and nemo_backend.get("ok") is True and nemo_backend.get("reports")):
        problems.append(f"nemotron backend (native): endpoint.json carries no passing mamba_backend_attestation "
                        f"({str(nemo_backend)[:300]}; error={ep.get('mamba_backend_attestation_error')!r}) — the served "
                        "Mamba backend is unattested")
    nemo_backend = {"source": "endpoint.mamba_backend_attestation", "admitted": (nemo_backend or {}).get("ok") if isinstance(nemo_backend, dict) else None,
                    "fused_backend_attested": (nemo_backend or {}).get("ok") if isinstance(nemo_backend, dict) else None,
                    "contract_sha256": (nemo_backend or {}).get("contract_sha256") if isinstance(nemo_backend, dict) else None,
                    "n_reports": len((nemo_backend or {}).get("reports") or []) if isinstance(nemo_backend, dict) else 0,
                    "rpc_method": (nemo_backend or {}).get("rpc_method") if isinstance(nemo_backend, dict) else None}
elif require_nemo and key == "nemotron3nano4b":
    # the backend attestation the probe job wrote (scripts/sdfbase/math500/nemo_backend_probe.py): read, never assumed
    if not os.path.isfile(nemo_att_path):
        problems.append(f"nemotron backend: no attestation at {nemo_att_path} — run nemo_backend_probe.sbatch on this cluster first (the fused Mamba kernels are a PREREQUISITE, never inferred from --time)")
    else:
        nemo_backend = json.load(open(nemo_att_path))
        if nemo_backend.get("fused_backend_attested") is not True:
            problems.append(f"nemotron backend: attestation says fused_backend_attested={nemo_backend.get('fused_backend_attested')!r} — a fallback somewhere; the Nemotron cells are not launched until the environment is fixed")
        # the same-GPU peer rule, re-derived here from the recorded numbers (D114)
        ref = nemo_backend.get("duty_reference") or {}
        duty, tok = nemo_backend.get("sm_duty_mean"), nemo_backend.get("tok_s")
        ref_duty, ref_tok = ref.get("sm_duty_mean"), ref.get("tok_s")
        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (duty, tok, ref_duty, ref_tok))
        if not numeric or nemo_backend.get("duty_comparable") is not True:
            problems.append(f"nemotron backend: no comparable same-GPU peer reference in the attestation (duty {duty!r} / {tok!r} tok/s vs peer {ref.get('model_key')!r}: {ref_duty!r} / {ref_tok!r}; comparable={nemo_backend.get('duty_comparable')!r}) — re-run the probe")
        elif duty < duty_ratio * ref_duty or tok < duty_ratio * ref_tok:
            problems.append(f"nemotron backend: measured SM duty {duty} / {tok} tok/s under the decode burst is below {duty_ratio} x the same-GPU peer's ({ref.get('model_key')}: {ref_duty} / {ref_tok})")
        if nemo_backend.get("admitted") is not True:
            problems.append(f"nemotron backend: the probe did not admit (admitted={nemo_backend.get('admitted')!r})")
        if serve_reqs_sha and nemo_backend.get("serving_requirements_sha256") != serve_reqs_sha:
            problems.append(f"nemotron backend: the attestation was made for serving requirements {str(nemo_backend.get('serving_requirements_sha256'))[:12]} but this job's venv is {serve_reqs_sha[:12]} — re-run the probe on this environment")
print(json.dumps({"key": key, "hybrid": hybrid, "seqs_cap": HYBRID_SEQS_CAP if hybrid else None,
                  "engine": ("lora_native" if native else "lens"),
                  "attested_max_num_seqs": att_s, "mamba_ssm_cache_dtype": dtypes,
                  "nemo_backend_required": bool(require_nemo and key == "nemotron3nano4b"),
                  "nemo_backend": (nemo_backend if native and isinstance(nemo_backend, dict) else
                                   ({**{k: nemo_backend.get(k) for k in ("admitted", "fused_backend_attested", "sm_duty_mean", "tok_s", "duty_ratio", "serving_requirements_sha256", "probe_job_id")},
                                     "duty_reference": {k: (nemo_backend.get("duty_reference") or {}).get(k) for k in ("model_key", "sm_duty_mean", "tok_s")}}
                                    if isinstance(nemo_backend, dict) else None)),
                  "problems": problems}))
PY
  ) || { f22_log "REFUSING TRAFFIC: could not read the model facts from the attestation named by $1"; return 1; }
  case "$MODEL_FACTS" in *'"problems": []'*) f22_log "model facts OK: $MODEL_FACTS"; return 0;; esac
  f22_log "REFUSING TRAFFIC: $MODEL_FACTS"; return 1
}

f22_bank_view(){   # builds + verifies the (KEY, VIEW) bank view; exports TT_TWO_SI_DEFENSES_DIR
  local py="${1:?python}" lock
  # VERIFY FIRST, BUILD ONLY UNDER A LOCK (2026-09-07). The bank view is SHARED state: every job of a
  # (target, view) writes the same $SHARED/.../cells_view/<view>. Four static jobs landing on one node
  # rebuilt it concurrently and one job's write_entry replaced a stamped entry while another's
  # verify_entry read it — FileNotFoundError on train_budget_meta.json, job 423389 dead at 3m30s. The
  # build is idempotent but not atomic, so: a read-only --check (safe concurrently) is the common path,
  # and the build runs only when the check fails, serialized on a per-(target,view) lock.
  local bank="$SHARED/exp/objects/sdfbase/$WAVE/$KEY"
  # THE ROSTER IS THE JOB'S (F22_ROSTER): every reader below carries `--roster "$F22_ROSTER"`, and the
  # bank rides explicitly (the same $SHARED/<bank_root>/<target> the builder resolves on its own on the cluster — ONE derivation,
  # shared with TT_TWO_SI_DEFENSES_DIR below, and the function runs over any $SHARED). A NON-production roster may name only
  # views the production roster does NOT: the view dir and its lock are keyed by (KEY, VIEW) alone, so two rosters naming one
  # view would rebuild the same SHARED directory under two cell sets (the lock serialises the writes; it cannot make them
  # agree) — refused by name before anything is read or built. The lock therefore stays keyed by (KEY, VIEW): one directory,
  # one lock, whichever roster named it.
  "$py" scripts/sdfbase/full22/roster_cell_count.py "$KEY" "$VIEW" --roster "$F22_ROSTER" >/dev/null 2>&1 \
    || f22_fatal "roster $F22_ROSTER names no view $KEY/$VIEW — a certification-only view needs F22_ROSTER=<its roster> exported on the submit line (the production roster is $F22_ROSTER_PRODUCTION; a batch job inherits the submitter's environment, never --export=)"
  if [ "$(realpath -m "$F22_ROSTER")" != "$(realpath -m "$F22_ROSTER_PRODUCTION")" ] \
     && "$py" scripts/sdfbase/full22/roster_cell_count.py "$KEY" "$VIEW" --roster "$F22_ROSTER_PRODUCTION" >/dev/null 2>&1; then
    f22_fatal "F22_ROSTER=$F22_ROSTER names $KEY/$VIEW, a view the production roster $F22_ROSTER_PRODUCTION also names — a non-production roster may only name views of its own (the bank view $bank/cells_view/$VIEW is shared state keyed by name)"
  fi
  lock="$SHARED/exp/objects/sdfbase/$WAVE/.$KEY-$VIEW.view.lock"
  mkdir -p "$(dirname "$lock")"
  if ! "$py" scripts/sdfbase/full22/make_bank_view.py --check --target "$KEY" --view "$VIEW" --roster "$F22_ROSTER" --bank "$bank" >/dev/null 2>&1; then
    f22_log "bank view $KEY/$VIEW (roster $F22_ROSTER) does not verify — building it under $lock"
    (
      flock -w 900 9 || { echo "FATAL: could not take the bank-view lock for $KEY/$VIEW in 900s" >&2; exit 1; }
      "$py" scripts/sdfbase/full22/make_bank_view.py --target "$KEY" --view "$VIEW" --roster "$F22_ROSTER" --bank "$bank"
    ) 9>"$lock" || f22_fatal "bank view for $KEY/$VIEW (roster $F22_ROSTER) refused"
  fi
  "$py" scripts/sdfbase/full22/make_bank_view.py --check --target "$KEY" --view "$VIEW" --roster "$F22_ROSTER" --bank "$bank" || f22_fatal "bank view for $KEY/$VIEW (roster $F22_ROSTER) refused"
  export TT_TWO_SI_DEFENSES_DIR="$SHARED/exp/objects/sdfbase/$WAVE/$KEY/cells_view/$VIEW"
  mapfile -t ALL_CELLS < <("$py" - "$CFG" <<'PY'
import sys, yaml; print("\n".join(yaml.safe_load(open(sys.argv[1]))["explicit_cells"]))
PY
)
  # s3r-fix: a STEERED-WAVE view (s<N>r) lists no `none` and may list ONE cell — its contrast is the u0 wave's unsteered store,
  # never a same-standup none; an EMPTY list is still a refusal
  [ "${#ALL_CELLS[@]}" -ge 1 ] || f22_fatal "$CFG lists no cell — nothing to serve (roster: enable cells first)"
  for c in "${ALL_CELLS[@]}"; do
    # `none` and a prompted__<si> baseline (the SI as the system prompt) serve no object: no view entry
    case "$c" in none|prompted__*) continue;; esac
    for f in object.pt staged_from.json; do
      [ -f "$TT_TWO_SI_DEFENSES_DIR/$c/$f" ] || f22_fatal "bank view $TT_TWO_SI_DEFENSES_DIR/$c lacks $f"
    done
    # The identity sidecar is exactly ONE of three: train_budget_meta.json (the training
    # run's fit), fit_provenance.json (the HPO-promotion lineage's fit — the same fit-time facts under
    # the stage-3 gate's native spelling; qwen's scale-12 si12 copies), derived_meta.json (a
    # DERIVED object — never trained, so no budget record by construction). make_bank_view.py
    # already ran the matching verdict; this is the shell-side presence check, and it must not
    # demand one lineage's spelling of another's cell.
    local n_id=0 f_id
    for f_id in train_budget_meta.json fit_provenance.json derived_meta.json; do
      [ -f "$TT_TWO_SI_DEFENSES_DIR/$c/$f_id" ] && n_id=$((n_id+1))
    done
    [ "$n_id" -eq 1 ] || f22_fatal "bank view $TT_TWO_SI_DEFENSES_DIR/$c declares $n_id identity sidecars (need exactly one of train_budget_meta.json / fit_provenance.json / derived_meta.json)"
  done
  # THE SHORT-VIEW GUARD, job side (2026-09-07): the config's cell list and the built view must BOTH
  # match the ROSTER's cells for this (KEY, VIEW). A job that serves fewer cells than the roster names
  # is an error, never a log line — the first launch attempt ran with 7 cells on one target and 5 on
  # the other (both exit 0), which would have produced a table missing an arm on one target only.
  local want want_staged have n_view
  want=$("$py" scripts/sdfbase/full22/roster_cell_count.py "$KEY" "$VIEW" --roster "$F22_ROSTER") \
    || f22_fatal "cannot read the roster's cell count for $KEY/$VIEW (roster $F22_ROSTER names no such view; a certification-only view needs F22_ROSTER=<its roster> on the submit line)"
  # the view holds the STAGED cells only: a prompted__<si> baseline is served (config) but has no object
  want_staged=$("$py" scripts/sdfbase/full22/roster_cell_count.py "$KEY" "$VIEW" --staged --roster "$F22_ROSTER") \
    || f22_fatal "cannot read the roster's staged cell count for $KEY/$VIEW (roster $F22_ROSTER)"
  # explicit_cells minus the `none` anchor WHEN the config lists one: the campaign configs do; a seed-replicate view of the
  # padded-static lane (docs/static_padding.md "Seed replicates") lists its fitted objects only, and `none` is not a roster cell
  have=${#ALL_CELLS[@]}; case " ${ALL_CELLS[*]} " in *" none "*) have=$((have - 1));; esac
  [ "$have" -eq "$want" ] || f22_fatal "$CFG lists $have served cells (none excluded) but the roster names $want for $KEY/$VIEW — regenerate the configs (make_full22_configs.py); refusing to run a narrowed wave"
  n_view=$(find "$TT_TWO_SI_DEFENSES_DIR" -mindepth 1 -maxdepth 1 ! -name 'VIEW_MANIFEST*' ! -name 'full22_*' 2>/dev/null | wc -l)
  [ "$n_view" -eq "$want_staged" ] || f22_fatal "bank view $TT_TWO_SI_DEFENSES_DIR holds $n_view entries but the roster names $want_staged staged cells for $KEY/$VIEW ($want served incl. prompted) — a short (or stale) view is an error"
  f22_log "bank view OK: $TT_TWO_SI_DEFENSES_DIR (${#ALL_CELLS[@]} cells incl. none; roster $F22_ROSTER: $want served / $want_staged staged, view $n_view — complete)"
  [ -f "$TT_TWO_SI_DEFENSES_DIR/full22_derived_quality.json" ] && f22_log "derived fit quality: $(tr -d '\n ' < "$TT_TWO_SI_DEFENSES_DIR/full22_derived_quality.json" | head -c 700)"
}

f22_bank_view_check(){   # $1 = python [, $2 = object cell id, $3 = its pinned object.pt sha256] -> exports TT_TWO_SI_DEFENSES_DIR
  # CHECK-ONLY: runs make_bank_view.py --check and is FATAL on any
  # non-zero exit. It NEVER takes the view lock and NEVER invokes the builder — the campaign's f22_bank_view above
  # rebuilds a failing view under a lock, a write to SHARED state an experiment that only READS the objects must not
  # perform. A drifted view is fixed deliberately, from a shell, then the job is resubmitted. With a cell + pin the
  # object's bytes are re-hashed here and held to the pin (a drift is fatal; the standup's --expect-object-sha256 is
  # the second, driver-readable copy of the same fact).
  local py="${1:?python}" cell="${2:-}" pin="${3:-}" have bank="$SHARED/exp/objects/sdfbase/$WAVE/$KEY"
  "$py" scripts/sdfbase/full22/make_bank_view.py --check --target "$KEY" --view "$VIEW" --roster "$F22_ROSTER" --bank "$bank" \
    || f22_fatal "bank view for $KEY/$VIEW (roster $F22_ROSTER) does not verify (check-only: this job never builds or repairs a view — build it deliberately with make_bank_view.py --target $KEY --view $VIEW --roster $F22_ROSTER from a shell, then resubmit)"
  export TT_TWO_SI_DEFENSES_DIR="$SHARED/exp/objects/sdfbase/$WAVE/$KEY/cells_view/$VIEW"
  if [ -n "$cell" ]; then
    [ -n "$pin" ] || f22_fatal "f22_bank_view_check: no pinned object_sha256 given for $cell (configs/sdfbase_full22_pins.json)"
    [ -f "$TT_TWO_SI_DEFENSES_DIR/$cell/object.pt" ] || f22_fatal "bank view $TT_TWO_SI_DEFENSES_DIR lacks $cell/object.pt"
    have=$(sha256sum "$TT_TWO_SI_DEFENSES_DIR/$cell/object.pt" | cut -c1-64)
    [ "$have" = "$pin" ] || f22_fatal "bank view $cell/object.pt sha256 $have != the pin $pin — a drifted object; refusing"
    f22_log "bank view OK (check-only): $TT_TWO_SI_DEFENSES_DIR; $cell object.pt == pin ${pin:0:12}"
  else
    f22_log "bank view OK (check-only): $TT_TWO_SI_DEFENSES_DIR"
  fi
}

f22_arm_bank_view(){   # $1 = python, $2 = arm  (needs KEY VIEW CFG CLUSTER SHARED) -> exports TT_TWO_SI_DEFENSES_DIR
  # An ARM is served from ITS OWN LANE'S BANK, never the wave's. That is the whole point: the wave
  # bank ($SHARED/exp/objects/sdfbase/$WAVE/$KEY/cells_view/<view>) is SHARED state that every peer
  # leg of this (target, view) verifies against ITS roster, so registering a lane's cell there would
  # rebuild that directory under every concurrent leg. f22_bank_view above is the wave path and is
  # untouched; this is the arm path and reads nothing under exp/objects/sdfbase.
  #
  # THE PEER-ROOT GUARD IS KEPT AND IS LOAD-BEARING (the rule that a job never writes another
  # lane's shared view): the resolved dir is refused if it lands under the wave bank, the wave root,
  # or the OLD alphasteer_f22 parents — in arm_cell_check.py's resolver AND again here on the
  # exported path, because the two derive it separately and a guard on only one is a guard on
  # neither. f22_bank_view's roster-collision guard has no analogue to breach here: this function
  # never passes $F22_ROSTER, never touches the wave bank's lock, and builds only inside the arm's
  # own bank root.
  local py="${1:?python}" arm="${2:?arm}" cell kind rel policy arms=configs/sdfbase_full22_arms.yaml
  [ -f "$arms" ] || f22_fatal "no $arms in $PWD — this checkout carries no arm table"
  # --requested-serving-path: the ARM'S SERVING POLICY (Finding 2) re-asserted by the resolver against the path this launch
  # resolved, ahead of its per-target refusals; f22_resolve_config already held it at the earliest arm-aware moment. The
  # fourth field is the declared policy (`-` = none), exported below for the post-standup evidence requirement.
  read -r cell kind rel policy < <("$py" scripts/sdfbase/full22/arm_cell_check.py --arms "$arms" --target "$KEY" --view "$VIEW" \
      --arm "$arm" --cluster "$CLUSTER" --resolve \
      --requested-serving-path "${F22_SERVING_PATH:-}" --serving-path-mode "${F22_SERVING_PATH_MODE:-}") \
    || f22_fatal "arm $arm is not servable for $KEY/$VIEW on $CLUSTER (see the PROBLEM lines)"
  [ -n "$rel" ] || f22_fatal "arm_cell_check.py --resolve printed nothing for $KEY/$VIEW/$arm"
  [ "$policy" = - ] && policy=""
  export TT_TWO_SI_DEFENSES_DIR="$SHARED/$rel"
  case "$TT_TWO_SI_DEFENSES_DIR" in
    */exp/objects/sdfbase/*|*/exp/sdfbase_full22/*|*/exp/objects/alphasteer_f22/*)
      f22_fatal "arm view dir $TT_TWO_SI_DEFENSES_DIR resolves under a PEER's root (the wave bank / wave root / the OLD alphasteer_f22 parents)";;
  esac
  mapfile -t ALL_CELLS < <("$py" - "$CFG" <<'PY'
import sys, yaml; print("\n".join(yaml.safe_load(open(sys.argv[1]))["explicit_cells"]))
PY
)
  [ "${ALL_CELLS[*]}" = "none $cell" ] || f22_fatal "$CFG lists [${ALL_CELLS[*]}] but the $arm pair serves exactly [none $cell] — regenerate (make_arm_configs.py --arm $arm)"
  # THE BANK-VIEW BUILD, TABLE-DRIVEN (arditi-qwen-all-positions A2' item 5). The port gated this on
  # `if [ "$arm" = arditi ]`, so a new arm SKIPPED view verification entirely and then failed the byte
  # invariant below with a misleading message. The gate is now the TABLE's own declaration: an arm
  # that names a `roster` and a `pins` file carries its view in a bank this builder can build, and one
  # that does not (alphasteer, whose rungs are staged directories) skips the build as before. Same
  # shape otherwise: VERIFY FIRST, BUILD ONLY UNDER A LOCK — the view is shared state between this
  # lane's own jobs, and concurrent rebuilds raced on the wave (job 423389).
  local roster pins positions bank bview lock
  read -r roster pins positions < <("$py" -c 'import sys,yaml; a=yaml.safe_load(open(sys.argv[1]))["arms"][sys.argv[2]]; print(*(a.get(k) or "-" for k in ("roster", "pins", "positions")))' "$arms" "$arm")
  [ "$roster" = - ] && roster=""; [ "$pins" = - ] && pins=""; [ "$positions" = - ] && positions=""
  # THE ARM'S FACTS THE ENTRY POINT NEEDS AFTER ITS STANDUP (Finding 2): an arm whose policy is lora_native must then prove,
  # from the standup's own records, that ITS kind was served natively at ITS position mode (f22_assert_native_evidence)
  # -- and, since arditi-qwen-all-positions C3, PERSIST that proof with an identity naming the CELL and the BYTES
  # (f22_arm_native_evidence below), which needs the cell id too.
  export F22_ARM_SERVING_POLICY="$policy" F22_ARM_KIND="$kind" F22_ARM_POSITION_MODE="$positions" F22_ARM_CELL="$cell"
  if [ -n "$roster" ] && [ -n "$pins" ]; then
    [ -f "$roster" ] && [ -f "$pins" ] || f22_fatal "missing $roster or $pins — the $arm lane's roster + pins are carried as data in this tree"
    bank="${TT_TWO_SI_DEFENSES_DIR%/cells_view/*}"; bview="${TT_TWO_SI_DEFENSES_DIR##*/cells_view/}"
    lock="$(dirname "$bank")/.$KEY-$bview.view.lock"; mkdir -p "$(dirname "$lock")"
    if ! "$py" scripts/sdfbase/full22/make_bank_view.py --check --roster "$roster" --pins "$pins" --bank "$bank" --target "$KEY" --view "$bview" >/dev/null 2>&1; then
      f22_log "$arm bank view $KEY/$bview does not verify — building it under $lock"
      (
        flock -w 900 9 || { echo "FATAL: could not take the bank-view lock for $KEY/$bview in 900s" >&2; exit 1; }
        "$py" scripts/sdfbase/full22/make_bank_view.py --roster "$roster" --pins "$pins" --bank "$bank" --target "$KEY" --view "$bview"
      ) 9>"$lock" || f22_fatal "$arm bank view for $KEY/$bview refused"
    fi
    "$py" scripts/sdfbase/full22/make_bank_view.py --check --roster "$roster" --pins "$pins" --bank "$bank" --target "$KEY" --view "$bview" \
      || f22_fatal "$arm bank view for $KEY/$bview refused"
  fi
  # THE BYTE INVARIANT on the object about to be served: sha == the table's pin (the identity the other
  # lanes' rows name), one identity sidecar, and the arm's own dose invariant (its unit's contract —
  # static_r: base_scale 1.0, one stored layer; vector_all: positions/fit_positions present, the one
  # served layer; alphasteer: base_scale == lambda). The config's cells + defenses_dir are checked too.
  local rec
  rec=$("$py" scripts/sdfbase/full22/arm_cell_check.py --arms "$arms" --target "$KEY" --view "$VIEW" --arm "$arm" --cluster "$CLUSTER" \
          --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --config "$CFG") \
    || f22_fatal "the $arm object about to be served for $KEY/$VIEW violates its invariant (see the PROBLEM lines)"
  # THE PINS' SERVED LAYER (round 2, item 2), off the record the checker just held the object to: what the post-standup
  # evidence compares every hooked boundary against. A native-policy arm whose record carries none is refused HERE, by
  # name -- a pins file without layers cannot certify one -- rather than served with the boundary unasserted.
  F22_ARM_SERVED_LAYER=$("$py" -c 'import json,sys; v=json.loads(sys.argv[1]).get("pins_served_layer"); print("" if v is None else int(v))' "$rec") \
    || f22_fatal "cannot read pins_served_layer off arm_cell_check.py's record for $arm"
  if [ "$policy" = lora_native ] && [ -z "$F22_ARM_SERVED_LAYER" ]; then
    f22_fatal "arm $arm declares serving_path: lora_native but its record carries no pins_served_layer (pins ${pins:-none}) — the post-standup evidence holds every hooked boundary to the pins' served layer, and a pins file without layers cannot certify one"
  fi
  export F22_ARM_SERVED_LAYER
  # THE TWO SHAS THE EVIDENCE IDENTITY NAMES (arditi-qwen-all-positions C3, the f22 twin of ar_bank_view's
  # AR_PIN_OBJECT_SHA256 / AR_SERVED_OBJECT_SHA256). The checker above hashed the object about to be served
  # (`object_sha256`) and held it to the arms table's operating point (`pinned_object_sha256`); both ride the
  # record, so the evidence names the bytes that were OBSERVED beside the bytes that were PROMISED rather
  # than re-hashing separately (two reads of one file can disagree; one read cannot).
  read -r F22_ARM_OBJECT_SHA256 F22_ARM_SERVED_OBJECT_SHA256 < <("$py" -c 'import json,sys; r=json.loads(sys.argv[1]); print(r.get("pinned_object_sha256") or "-", r.get("object_sha256") or "-")' "$rec") \
    || f22_fatal "cannot read the object shas off arm_cell_check.py's record for $arm"
  [ "$F22_ARM_OBJECT_SHA256" = - ] && F22_ARM_OBJECT_SHA256=""
  [ "$F22_ARM_SERVED_OBJECT_SHA256" = - ] && F22_ARM_SERVED_OBJECT_SHA256=""
  if [ "$policy" = lora_native ] && { [ -z "$F22_ARM_OBJECT_SHA256" ] || [ -z "$F22_ARM_SERVED_OBJECT_SHA256" ]; }; then
    f22_fatal "arm $arm declares serving_path: lora_native but its record carries no object sha (pinned '${F22_ARM_OBJECT_SHA256:-none}', served '${F22_ARM_SERVED_OBJECT_SHA256:-none}') — the evidence identity must name the bytes that were served, and a record that cannot is not stamped"
  fi
  export F22_ARM_OBJECT_SHA256 F22_ARM_SERVED_OBJECT_SHA256
  f22_log "arm bank view OK: $TT_TWO_SI_DEFENSES_DIR (cells [${ALL_CELLS[*]}]; kind $kind; serving policy ${policy:-none}; position mode ${positions:-none}; pins served layer ${F22_ARM_SERVED_LAYER:-none}) $rec"
  [ -n "${SERVE_DIR:-}" ] && [ -d "$SERVE_DIR" ] && printf '%s\n' "$rec" > "$SERVE_DIR/arm_cell_check.json"
  return 0
}

#: the stamped record every native-policy arm standup leaves beside its endpoint, and the env the drivers
#: read it back through — the same two names ar_common.sh uses, restated here because the f22 entry points
#: cannot source that file (it re-points the lane's roots); pinned equal by test
F22_ARM_SERVING_EVIDENCE_FILE=serving_evidence.json
F22_ARM_SERVING_EVIDENCE_ENV=SISTEER_SERVING_EVIDENCE_RECORD

f22_arm_native_evidence(){   # $1 = endpoint.json  -- run after EVERY successful standup of an ARM job
  # THE WHOLE POST-STANDUP CONTRACT FOR A NATIVE-POLICY ARM, in ONE call at every entry point
  # (arditi-qwen-all-positions C3). Until now the goalset only JUDGED its standup (f22_assert_native_evidence)
  # and LOGGED the verdict, while ar_cell.sbatch additionally PERSISTED the record, stamped it with an
  # identity and bound it into the stores its drivers wrote — so a store produced by any entry point OTHER
  # than ar_cell carried no pointer to the evidence that certified the standup its rows were drawn under,
  # and the grader's "follow the store's pointer" rule had nothing to follow. Three steps, the same three:
  #
  #   (1) JUDGE — f22_assert_native_evidence, the ONE body: the endpoint's own claims, the attestation
  #       through its strict reader, the slot's two op names apart, and every hooked boundary at the PINS'
  #       served layer. A refusal writes the bare body (its reasons stay on disk) and returns 1.
  #   (2) PERSIST — stamp the OK body with its identity (serving_evidence_binding.stamp: the record's own
  #       sha, the job id, the serve dir, the cell, the served adapters' fingerprints, the pins' object sha
  #       and the OBSERVED one) and write it beside the endpoint. An identity that cannot name the bytes is
  #       refused by the module itself.
  #   (3) BIND — export the record's path so every driver this job launches folds its sha + path into each
  #       shard's served_target.json (serving_evidence_binding.fold_into), which is what makes the store
  #       self-describing: the grader follows the STORE to the record, never a path typed by hand.
  #
  # A NO-OP for an arm that declares no native policy (and for a non-arm job, which never calls it): the
  # env is UNSET first, never inherited, so a lens or policy-free leg's shards stay byte-identical.
  local ep="${1:?endpoint.json}" rec_p sha
  unset "$F22_ARM_SERVING_EVIDENCE_ENV"
  if [ "${F22_ARM_SERVING_POLICY:-}" != lora_native ]; then
    f22_log "arm serving evidence: none required (policy ${F22_ARM_SERVING_POLICY:-none}) — $F22_ARM_SERVING_EVIDENCE_ENV unset; the drivers fold nothing"
    return 0
  fi
  for v in F22_ARM_CELL F22_ARM_KIND F22_ARM_POSITION_MODE F22_ARM_SERVED_LAYER F22_ARM_OBJECT_SHA256 F22_ARM_SERVED_OBJECT_SHA256; do
    [ -n "${!v:-}" ] || f22_fatal "f22_arm_native_evidence: $v is unset — f22_arm_bank_view exports every fact the record's identity names; a record that cannot name what it certified is not stamped"
  done
  # the bytes OBSERVED in the view must be the bytes the operating point PROMISED, or the identity would
  # name one and the standup would have served the other (the round-5 rule, applied at this entry point too)
  [ "$F22_ARM_SERVED_OBJECT_SHA256" = "$F22_ARM_OBJECT_SHA256" ] \
    || f22_fatal "f22_arm_native_evidence: the served object's sha $F22_ARM_SERVED_OBJECT_SHA256 != the operating point's $F22_ARM_OBJECT_SHA256 — refusing before anything is stamped"
  rec_p="$(dirname "$ep")/$F22_ARM_SERVING_EVIDENCE_FILE"
  if ! f22_assert_native_evidence "$ep" "$F22_ARM_KIND" "$F22_ARM_POSITION_MODE" "" "" "$F22_ARM_SERVED_LAYER"; then
    printf '%s\n' "$F22_NATIVE_EVIDENCE" > "$rec_p" 2>/dev/null || true   # a refusal gets its reasons on disk, and no identity
    f22_log "ARM SERVING EVIDENCE REFUSED (native): $F22_NATIVE_EVIDENCE"
    return 1
  fi
  "${SPY:?f22_arm_native_evidence needs SPY}" - "$rec_p" "$(dirname "$ep")" "$F22_ARM_CELL" \
      "$F22_ARM_OBJECT_SHA256" "$F22_ARM_SERVED_OBJECT_SHA256" "${SLURM_JOB_ID:-manual}" "${REPO_DIR:?}" "$F22_NATIVE_EVIDENCE" <<'PY' \
    || f22_fatal "cannot persist the arm's serving evidence with its identity — the store's rows would point at nothing"
import json, sys
out, serve_dir, cell, obj, served, job, repo, body_json = sys.argv[1:9]
sys.path.insert(0, f"{repo}/scripts/sdfbase/full22")
import serving_evidence_binding as seb
rec = seb.stamp(json.loads(body_json), slurm_job_id=job, serve_dir=serve_dir, cell=cell,
                object_sha256=obj, served_object_sha256=served)
with open(out, "w", encoding="utf-8") as f:
    json.dump(rec, f, sort_keys=True, indent=1); f.write("\n")
print(f"identity: evidence_sha256={rec[seb.IDENTITY_KEY][seb.SHA_FIELD]} job={job} cell={cell} "
      f"object={obj[:12]} served={served[:12]} "
      f"fingerprints={rec[seb.IDENTITY_KEY]['residual_fingerprint']} "
      f"expected_served_layer={rec[seb.IDENTITY_KEY]['expected_served_layer']}")
PY
  sha=$("${MPY:-python3}" - "$rec_p" "${REPO_DIR:?}" <<'PY'
import sys
sys.path.insert(0, f"{sys.argv[2]}/scripts/sdfbase/full22")
import serving_evidence_binding as seb
_rec, sha = seb.load_record(sys.argv[1])    # refuses an absent / unshaped / unstamped / edited record by name
print(sha)
PY
  ) || f22_fatal "f22_arm_native_evidence: $rec_p is not a stamped serving-evidence record — the drivers cannot bind their stores to it"
  export "$F22_ARM_SERVING_EVIDENCE_ENV"="$rec_p"
  f22_log "arm serving evidence persisted + bound: $rec_p (sha256 ${sha:0:12}…; every served_target.json this job writes points at it)"
  f22_log "arm serving evidence OK (native): $F22_NATIVE_EVIDENCE"
}

f22_assert_native_evidence(){   # $1 = endpoint.json, $2 = expected petri_source_kind, $3 = expected position mode,
                                # $4 = the residual slot's KERNEL op name ("" = native_residual.OP_QUALNAME),
                                # $5 = the PAYLOAD op ("" = native_residual.OP_ADD_VECTOR),
                                # $6 = the expected SERVED layer: the PINS' served_layer (REQUIRED -- an empty value refuses)
                                # -> F22_NATIVE_EVIDENCE (ONE JSON record: facts + problems); 0 = every assertion held, 1 = refused
  # THE NATIVE EVIDENCE BODY, ONCE. What the engine DID after a standup, from its own records. Two call sites read this one
  # body -- ar_cell.sbatch's native branch (ar_common.sh ar_assert_serving_evidence native) and every arm entry point whose
  # arm declares serving_path: lora_native (target_attacks_goalset.sbatch, after its standup) -- so a third copy can never
  # drift from either.
  #   (1) the endpoint's OWN claims: serving_path_selection.selected == lora_native, attested_enforce_eager false, graphs not
  #       NONE, no lens_reduction_mode. f22_standup's native gate already refuses traffic on the last three; they are re-read
  #       here into THIS record because a leg must be able to prove its serving state from its own files;
  #   (2) the attestation through its STRICT reader (a foreign or inferred record refuses outright): the residual slot ARMED,
  #       and every payload-carrying adapter of the expected kind and position mode;
  #   (3) the slot's TWO op names, kept apart: the KERNEL is registered under ONE name (expected.op_name; each
  #       worker report's op_name), each PAYLOAD carries its own op (expected.adapters[*].op; each loaded slot's op) -- the
  #       constant add for these arms. The first draft compared one against the other and refused every healthy standup;
  #   (4) WHERE the edit lands: every payload's `boundaries` and every worker's loaded
  #       slot boundaries == [the PINS' served layer] -- the fit record's number, carried by the arm's pins and passed in
  #       by the caller (ar_cell: SERVED_LAYER off ar_resolve_config; the goalset: F22_ARM_SERVED_LAYER off
  #       f22_arm_bank_view's record). A layer derived from the object being certified would agree with a coherent
  #       object baked a block off; the pins do not. No layer passed = refused by name, never "not asserted".
  # The caller logs the record under its own heading; this function only judges and returns it.
  local ep="${1:?endpoint.json}" want_kind="${2:?petri_source_kind}" want_mode="${3:?position mode}" kernel_op="${4:-}" payload_op="${5:-}" want_layer="${6:-}" rc
  F22_NATIVE_EVIDENCE=""
  [ -f "$ep" ] || { F22_NATIVE_EVIDENCE="{\"endpoint\": \"$ep\", \"facts\": {}, \"problems\": [\"no endpoint.json at $ep -- a native standup writes one\"]}"; return 1; }
  [[ "$want_layer" =~ ^[0-9]+$ ]] || { F22_NATIVE_EVIDENCE="{\"endpoint\": \"$ep\", \"facts\": {}, \"problems\": [\"no expected served layer passed (got '$want_layer') -- the pins' served_layer is required to certify the hooked boundary; a pins file without layers cannot certify one\"]}"; return 1; }
  # $SPY IS THE SERVING VENV, WHICH DOES NOT CONTAIN `sisteer` — it is built from
  # configs/capeval/vllm-requirements.txt (vLLM + its deps) and nothing else, so this body died with
  # `ModuleNotFoundError: No module named 'sisteer'` the first time any caller actually reached it
  # (job 532303: the standup came up clean at 107x4864 and the evidence assertion then refused with
  # an EMPTY record). It was invisible until now because no caller had run it in-job: Phase C's
  # identities were produced post hoc under the driver venv, and the goalset's arm path has not run.
  # The two modules imported below are stdlib-only, so the fix is the PATH, not the interpreter —
  # the checkout's src, exactly as every driver in this file family is launched
  # (`PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts"`). Keeping $SPY means the record is still read by
  # the very interpreter that served the endpoint. Verified on the node with that venv: both modules
  # import and `nr.OP_QUALNAME` resolves.
  # `:+`, not `:?`: when REPO_DIR is known (every in-job caller — f22_init sets it) the checkout's src
  # is PREPENDED; when it is not (a harness that arranges PYTHONPATH itself) the caller's own path is
  # inherited untouched. Not a silent fallback: an interpreter that can import neither way still dies
  # loudly, with the same ModuleNotFoundError, one line down. The `:?` form this replaced fatalled
  # under `set -u` in two test harnesses that set PYTHONPATH but never REPO_DIR.
  F22_NATIVE_EVIDENCE=$(PYTHONPATH="${REPO_DIR:+$REPO_DIR/src:}${PYTHONPATH:-}" \
    "$SPY" - "$ep" lora_native "$kernel_op" "$want_mode" "$want_kind" "$payload_op" "$want_layer" <<'PY'
import json, sys
from sisteer.capeval.serving import native_residual as nr
from sisteer.capeval.serving import serving_attestation as sa

ep_path, want_path, want_op, want_mode, want_kind, want_adapter_op = sys.argv[1:7]
want_layer = int(sys.argv[7])
ep = json.load(open(ep_path))
problems, facts = [], {"expected_served_layer": want_layer}
# a caller's restated constants (ar_common.sh's AR_NATIVE_*) are checked against the modules that own them: a drifted
# constant is a refusal here, never a check that quietly tests something else. A caller passing "" (the goalset) takes
# the module's own.
want_op = want_op or nr.OP_QUALNAME
want_adapter_op = want_adapter_op or nr.OP_ADD_VECTOR
if want_op != nr.OP_QUALNAME or want_op != sa.RESIDUAL_SLOT_OP_NAME:
    problems.append(f"AR_NATIVE_OP={want_op!r} != native_residual.OP_QUALNAME {nr.OP_QUALNAME!r} — the shell constant drifted")
if want_adapter_op != nr.OP_ADD_VECTOR:
    problems.append(f"AR_NATIVE_ADAPTER_OP={want_adapter_op!r} != native_residual.OP_ADD_VECTOR {nr.OP_ADD_VECTOR!r} — the shell constant drifted")

# (1) THE ENDPOINT'S OWN CLAIMS. common.sh's standup already refuses traffic unless eager is false
# and graphs are not NONE on a native engine; this re-reads the same fields into THIS lane's record,
# because a leg of this arm must be able to prove its own serving state from its own files.
sel = (ep.get("serving_path_selection") or {}).get("selected")
facts["serving_path_selection.selected"] = sel
if sel != want_path:
    problems.append(f"serving_path_selection.selected={sel!r} != {want_path!r}")
facts["attested_enforce_eager"] = ep.get("attested_enforce_eager")
if ep.get("attested_enforce_eager") is not False:
    problems.append(f"attested_enforce_eager={ep.get('attested_enforce_eager')!r} — native serving with CUDA graphs is "
                    "REQUIRED for this object (spec: 'in every leg that serves it'); eager is a different measurement")
facts["attested_cudagraph_mode"] = ep.get("attested_cudagraph_mode")
if ep.get("attested_cudagraph_mode") in (None, "NONE"):
    problems.append(f"attested_cudagraph_mode={ep.get('attested_cudagraph_mode')!r} — graphs are off; see above")
facts["lens_reduction_mode"] = ep.get("lens_reduction_mode")
if ep.get("lens_reduction_mode") is not None:
    problems.append(f"lens_reduction_mode={ep.get('lens_reduction_mode')!r} on a NATIVE standup — two engines on one "
                    "record; the lens stack is not in this path")

# (2) THE ATTESTATION, THROUGH ITS OWN STRICT READER. read_serving_attestation refuses a foreign or
# inferred record outright, so reaching engine_identity at all is already a fact about the file.
att_path = ep.get("serving_attestation")
if not att_path:
    problems.append("endpoint.json names no serving_attestation — a native standup writes one")
else:
    try:
        att = sa.read_serving_attestation(att_path)
    except Exception as e:                                    # noqa: BLE001 — the reader's own refusal is the message
        problems.append(f"{att_path}: {type(e).__name__}: {e}")
        att = None
    if att is not None:
        ident = sa.engine_identity(att)
        rs = att.get(sa.RESIDUAL_SLOT_KEY) or {}
        facts["residual_slot.armed"] = rs.get("armed")
        facts["residual_slot.n_workers"] = rs.get("n_workers")
        if not rs.get("armed"):
            problems.append("residual_slot.armed is false — this arm IS the residual slot; an unarmed engine served "
                            "the base model and every number would be the anchor's")
        # the adapters carrying a payload: their kind, mode and op, from the EXPECTED state the
        # launcher derived from the adapter dirs and every worker attested against
        served = {n: b for n, b in (ident.get("adapters") or {}).items() if b.get("residual_fingerprint")}
        facts["residual_adapters"] = {n: {k: b.get(k) for k in ("petri_source_kind", "position_mode")}
                                      for n, b in served.items()}
        # the served payloads' IDENTITY (round 3, item 2): the fingerprint each adapter's residual payload carries, so
        # the record can be bound to WHAT was served (ar_common.sh stamps it into the record's identity)
        facts["residual_adapter_fingerprints"] = {n: b.get("residual_fingerprint") for n, b in served.items()}
        if not served:
            problems.append("no registered adapter carries a residual payload — nothing of this arm is loaded")
        for name, b in sorted(served.items()):
            if b.get("petri_source_kind") != want_kind:
                problems.append(f"adapter {name!r}: petri_source_kind={b.get('petri_source_kind')!r} != {want_kind!r}")
            if b.get("position_mode") != want_mode:
                problems.append(f"adapter {name!r}: position_mode={b.get('position_mode')!r} != {want_mode!r} — this arm "
                                "is Arditi's UNGATED hook: every row of every forward, prefill and decode")
        # (3) THE TWO OP NAMES, APART (Finding 1; the constants block above says why). The KERNEL name is
        # the expected state's `op_name` and every worker report's `op_name`; the PAYLOAD op is per
        # adapter in expected.adapters[*].op and per loaded slot in worker_reports[*].slots[*].op. The
        # first is one registered kernel, the second is this arm's constant add; neither is the other.
        expected = rs.get("expected") or {}
        facts["residual_slot.op_name"] = expected.get("op_name")
        if expected.get("op_name") != want_op:
            problems.append(f"residual_slot.expected.op_name={expected.get('op_name')!r} != {want_op!r} — the slot's kernel "
                            "is registered under exactly one name")
        exp = expected.get("adapters") or {}
        payloads = {n: b for n, b in exp.items() if b.get("residual_fingerprint")}
        facts["residual_adapter_ops"] = {n: b.get("op") for n, b in payloads.items()}
        # (4) WHERE THE EDIT LANDS: the boundary is the block-output layer the payload writes at (for this kind
        # unshifted: boundaries == layers == the blob's stored key), held to the PINS' served layer
        facts["residual_adapter_boundaries"] = {n: sorted(int(x) for x in (b.get("boundaries") or [])) for n, b in payloads.items()}
        for name, b in sorted(payloads.items()):
            if b.get("op") != want_adapter_op:
                problems.append(f"adapter {name!r}: payload op {b.get('op')!r} != {want_adapter_op!r} — this arm is a CONSTANT "
                                "add of the raw difference of means (a gated op would be a PSR wearing this arm's name)")
            bounds = facts["residual_adapter_boundaries"][name]
            if bounds != [want_layer]:
                problems.append(f"adapter {name!r} writes at boundaries {bounds}, but the pins' served layer is {want_layer} — the "
                                "edit does not land where the fit record put it (a coherent object baked a block off agrees with "
                                "every relational check; only the pins' absolute layer refuses it)")
        reports = rs.get("worker_reports") or []
        facts["residual_slot.worker_op_names"] = sorted({str(r.get("op_name")) for r in reports if "op_name" in r})
        for i, rep in enumerate(reports):
            if "op_name" in rep and rep.get("op_name") != want_op:
                problems.append(f"worker {i}: residual_slot.op_name={rep.get('op_name')!r} != {want_op!r}")
            slots = rep.get("slots") or {}
            for name, b in sorted(payloads.items()):
                got = slots.get(str(b.get("lora_int_id")))
                if isinstance(got, dict) and got.get("op") != want_adapter_op:
                    problems.append(f"worker {i}: adapter {name!r} loaded slot op {got.get('op')!r} != {want_adapter_op!r}")
                if isinstance(got, dict):
                    gb = sorted(int(x) for x in (got.get("boundaries") or []))
                    if gb != [want_layer]:
                        problems.append(f"worker {i}: adapter {name!r} loaded boundaries {gb}, but the pins' served layer is {want_layer}")
print(json.dumps({"endpoint": ep_path, "facts": facts, "problems": problems}, sort_keys=True))
raise SystemExit(1 if problems else 0)
PY
  ); rc=$?
  return $rc
}


# --- venvs (node-local, flock-guarded; the mt1 bootstraps) --------------------------------------
f22_driver_venv(){   # -> MPY (the driver venv python, pinned to WAVE_SHA)
  local venv="/home/$USER/venvs/f22-${WAVE_SHA:0:12}" lockdir sentinel
  lockdir=$(dirname "$venv"); mkdir -p "$lockdir"; sentinel="$lockdir/.$(basename "$venv").synced"
  (
    flock -x 9 || exit 99
    if [ -f "$sentinel" ] && [ "$(cat "$sentinel" 2>/dev/null)" = "$WAVE_SHA" ] && [ -x "$venv/bin/python" ]; then
      f22_log "driver venv ready @ ${WAVE_SHA:0:12}"
    else
      # the uv cache is on cluster-shared storage: a concurrent build on ANOTHER node can drop a temp
      # file under this one mid-install ("Stale file handle (os error 116)" — qwen ladder shard 0,
      # 2026-09-11); the node-local flock cannot see that, so retry the sync a few times
      local try=0
      until UV_PROJECT_ENVIRONMENT="$venv" uv sync --frozen --extra cu128; do
        try=$((try + 1)); [ "$try" -ge 3 ] && { f22_log "uv sync failed $try times"; break; }
        f22_log "uv sync failed (attempt $try) — retrying in 30 s"; sleep 30
      done
      [ -x "$venv/bin/python" ] && echo "$WAVE_SHA" > "$sentinel"
    fi
  ) 9>"$lockdir/.f22-venv.lock"
  MPY="$venv/bin/python"; [ -x "$MPY" ] || f22_fatal "no driver venv at $MPY"
  "$MPY" -c 'import torch, pyrit, sisteer; print("driver venv OK", torch.__version__, pyrit.__version__, sisteer.__file__)' || f22_fatal "driver venv guard"
  local sis; sis=$("$MPY" -c 'import sisteer,os;print(os.path.dirname(os.path.dirname(sisteer.__file__)))')
  case "$sis" in *"$WAVE_SHA"*) ;; *) f22_fatal "sisteer resolves to $sis not $WAVE_SHA";; esac
  "$MPY" - <<'PY' || f22_fatal "PAIR budget is not 5x3 on this checkout"
from sisteer.eval.attacks.pair_tap import PAIR_N_STREAMS; assert PAIR_N_STREAMS == 5, PAIR_N_STREAMS
PY
  # THE CLOSER'S INTERPRETER: every job's EXIT trap ends in f22_registry_close_stores, and since
  # a4a304a38 the closer recomputes the eval goal set in-process through jailbreak_run (+ the HF data) — an interpreter that cannot
  # import them is an UNTRUSTED refusal by name. No sbatch ever assigned F22_PY, so the trap ran the compute node's bare python3 and
  # the closer's `|| true` swallowed the refusal: Slurm said success while the runs never advanced to generated or the grading
  # worklist. The confirmed driver venv IS the interpreter that imports the resolver: set here, once the venv passed its guards, so
  # every job that confirms the venv before it exits closes under it (target_attacks.sbatch arms its trap after this call). A caller
  # that set F22_PY itself keeps its own.
  F22_PY="${F22_PY:-$MPY}"
}

# THE SERVING-VENV VARIANT PER MODEL KEY (one table, read by every sbatch that stands a target up): the Nemotron-H Mamba2
# hybrid takes the mamba-ssm overlay; the Qwen3.5 gated-deltanet hybrids take the GDN overlay (flash-linear-attention +
# causal-conv1d); everything else the campaign's base venv. A caller that passes a variant explicitly is unchanged.
f22_serving_venv_variant_for_key(){   # $1 = model key ("" = unbound -> the base) -> "" | nemotron | gdn
  case "${1:-}" in nemotron3nano4b) echo nemotron;; qwen35_*) echo gdn;; *) echo "";; esac
}
f22_serving_venv_for_key(){ f22_serving_venv "$(f22_serving_venv_variant_for_key "${1:?key}")"; }
# THE KERNEL-OVERLAY INSTALL: the one command line both serving-venv bootstraps run (f22_serving_venv below and
# scripts/sdfbase/multilora_cert.sbatch, which sources this file for it); tests/test_serving_overlay_install.py dry-runs it from
# inside a checkout. --no-config because a bootstrap runs with the checkout as its working directory (f22_init's cd), where uv
# would otherwise read pyproject.toml's [tool.uv] table, which describes the PROJECT environment: its extra-build-dependencies
# give causal-conv1d and mamba-ssm a `torch` build dependency with match-runtime, and uv refuses any resolution that holds
# either package without torch ("`torch` was declared as an extra build dependency with `match-runtime = true`, but was not
# found in the resolution"), which a --no-deps overlay always is. So the overlay resolves from its requirements file and these
# flags alone: the one table setting that reached it, the exclude-newer cutoff, is passed explicitly (the base install's own),
# and the force-build variables the table also names are set here as before. Both overlays' hashed candidate sets are the same
# under the table as it stood before its extra-build entries, under these flags, and with no cutoff at all, so the venvs
# resolve exactly as before. The base install is left as it was: it reads the table too, its exclude-newer-package entries are
# part of what the base resolves to (anthropic, httpx2 and httpcore2 past the cutoff), and torch is in its resolution, so the
# refusal cannot fire there.
f22_overlay_install(){   # $1 = the venv's python, $2 = the overlay requirements file, [$3... = extra uv flags, e.g. --dry-run]
  local py="${1:?f22_overlay_install needs the venv python}" overlay="${2:?f22_overlay_install needs the overlay requirements}"
  shift 2
  CAUSAL_CONV1D_FORCE_BUILD=TRUE MAMBA_FORCE_BUILD=TRUE uv pip install --no-config --python "$py" \
    --exclude-newer 2026-08-03T00:00:00Z --no-build-isolation --no-deps "$@" -r "$overlay"
}
f22_serving_venv(){   # [$1 = variant: "" (the base) | nemotron | gdn | a path to a kernel-overlay requirements file]
                      # NO argument = the variant f22_serving_venv_variant_for_key resolves from the job's bound $KEY
                      # -> SPY, SERVE_VENV, SERVE_REQS_SHA (configs/capeval/vllm-requirements.txt, hash-versioned)
  # The BASE is byte-identical to before: the campaign's sisteer-petri-vllm-<hash8> venv, shared with peers.
  # A VARIANT (`nemotron` = configs/capeval/vllm-requirements-nemotron.txt, the fused
  # Mamba kernels mamba-ssm + causal-conv1d at the fit env's pins; `gdn` = configs/capeval/vllm-requirements-gdn.txt, the
  # fused gated-deltanet kernels flash-linear-attention + causal-conv1d) is built into a SEPARATE venv named
  # sisteer-petri-vllm-<variant>-<hash8 of base+overlay>: the base requirements first, exactly as the default,
  # then the overlay with --no-build-isolation against the venv's own torch (the kernels compile against it; the fit
  # env's recipe: CUDA_HOME nvcc, TORCH_CUDA_ARCH_LIST 9.0;10.0, *_FORCE_BUILD). Peers' venv is never touched.
  # THE SELECTION IS CENTRAL: every production entry point calls this bare (target_static / target_attacks /
  # petri_proxy / bon_climb / the sampled wave), so a bare call resolves the variant from the job's $KEY — qwen35_* -> gdn,
  # nemotron3nano4b -> nemotron, else the base — and an EXPLICIT argument (even "") is honoured as given, so a caller that
  # wants the base for a hybrid (nemo_backend_probe's fallback baseline: `f22_serving_venv ""`) still gets it.
  local variant reqs="$REPO_DIR/configs/capeval/vllm-requirements.txt" overlay="" tag="" h hf lockdir sent
  if [ $# -ge 1 ]; then variant="$1"; else variant="$(f22_serving_venv_variant_for_key "${KEY:-}")"; f22_log "serving venv variant from KEY=${KEY:-unbound}: '${variant:-base}'"; fi
  case "$variant" in
    "") ;;
    nemotron) overlay="$REPO_DIR/configs/capeval/vllm-requirements-nemotron.txt"; tag="-nemotron";;
    # the GATED-DELTANET overlay (the steered canary, 2026-10-01): flash-linear-attention + causal-conv1d for the Qwen3.5 hybrids,
    # whose HF-side forwards (the export's layout, the probe's HF+PEFT reference) otherwise take transformers' pure-torch fallback
    gdn) overlay="$REPO_DIR/configs/capeval/vllm-requirements-gdn.txt"; tag="-gdn";;
    *) overlay="$variant"; tag="-$(basename "$variant" .txt | sed 's/^vllm-requirements-//')";;
  esac
  [ -f "$reqs" ] || f22_fatal "no serving requirements at $reqs"
  [ -z "$overlay" ] || [ -f "$overlay" ] || f22_fatal "no kernel-overlay requirements at $overlay"
  # ONE hash of the requirement bytes (base, then the overlay when there is one): the venv name and the sentinel
  # are its prefixes, the attestation key (f22_model_facts) the full digest
  SERVE_REQS_SHA=$(cat "$reqs" ${overlay:+"$overlay"} | sha256sum | cut -c1-64)
  h="${SERVE_REQS_SHA:0:8}"; hf="${SERVE_REQS_SHA:0:16}"
  # F22_SERVING_VENV_ROOT: the directory the node-local serving venvs live in (default /home/$USER/venvs); a scratch root lets a
  # build proof run this bootstrap without creating a shared venv name
  SERVE_VENV="${F22_SERVING_VENV_ROOT:-/home/$USER/venvs}/sisteer-petri-vllm$tag-$h"; lockdir=$(dirname "$SERVE_VENV"); mkdir -p "$lockdir"
  # THE SENTINEL is the completion marker: written only after the base AND the overlay installed (every step above it exits the
  # subshell on failure), so a venv whose overlay failed (base present, kernels missing) carries none and the next call
  # re-enters the bootstrap, re-audits the base and installs the overlay; the overlay import check below runs on every call
  sent="$lockdir/.$(basename "$SERVE_VENV").vllm-bootstrap"
  (
    flock -x 9 || exit 99
    if [ -f "$sent" ] && [ "$(cat "$sent")" = "$hf" ] && [ -x "$SERVE_VENV/bin/python" ]; then
      f22_log "serving venv already bootstrapped${tag:+ (variant$tag)}"
    else
      [ -x "$SERVE_VENV/bin/python" ] || uv venv --python 3.12 "$SERVE_VENV" || exit 1
      uv pip install --python "$SERVE_VENV/bin/python" --index-strategy unsafe-best-match --exclude-newer 2026-08-03T00:00:00Z -r "$reqs" || exit 1
      if [ -n "$overlay" ]; then
        # the kernel overlay: a SOURCE build against the venv's torch (no wheel at torch 2.11 / cu129), so build
        # isolation must be off and nvcc present — the fit env's recipe (wave_manifest derived_build_nemotron.venv)
        export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}" CC="${CC:-gcc}" CXX="${CXX:-g++}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0;10.0}"
        export MAX_JOBS="${MAX_JOBS:-${SLURM_CPUS_PER_TASK:-8}}"
        [ -x "$CUDA_HOME/bin/nvcc" ] || { echo "FATAL: no nvcc at $CUDA_HOME/bin/nvcc — the kernel overlay is a source build; run this bootstrap on a GPU node with the CUDA toolkit" >&2; exit 1; }
        f22_overlay_install "$SERVE_VENV/bin/python" "$overlay" || exit 1
      fi
      echo "$hf" > "$sent"
    fi
  ) 9>"$lockdir/.$(basename "$SERVE_VENV").uv-sync.lock"
  # the bootstrap subshell's OWN exit status decides: `uv venv` can succeed and
  # leave bin/python behind while `uv pip install` fails, so an existence test on the interpreter is not a verified install
  local brc=$?
  [ "$brc" -eq 0 ] || f22_fatal "serving venv bootstrap failed (rc=$brc) — the requirements did not install into $SERVE_VENV"
  SPY="$SERVE_VENV/bin/python"; [ -x "$SPY" ] || f22_fatal "serving venv bootstrap failed"
  # the IMPORT guard: the serving stack AND peft must import from the venv that serves — peft==0.20.0 (the repo's uv.lock
  # pin) is REQUIRED in configs/capeval/vllm-requirements.txt because the serving probe's HF+PEFT parity check is a
  # certification check, and a venv built without it would run a probe whose verdict cannot certify
  "$SPY" -c 'import vllm, torch, peft; print("serving venv OK", vllm.__version__, torch.__version__, "peft", peft.__version__)' \
    || f22_fatal "serving venv guard: vllm / torch / peft do not all import from $SERVE_VENV (peft==0.20.0 is REQUIRED — D55); rebuild from $reqs"
  if [ -n "$overlay" ]; then
    # the overlay's IMPORT check: the packages the variant exists for must import in the venv that serves — per variant
    case "$tag" in
      -gdn)
        "$SPY" -c 'import fla, causal_conv1d; from fla.ops.gated_delta_rule import chunk_gated_delta_rule; from causal_conv1d import causal_conv1d_fn, causal_conv1d_update; from transformers.utils.import_utils import is_flash_linear_attention_available, is_causal_conv1d_available; assert is_flash_linear_attention_available() and is_causal_conv1d_available(), "transformers does not see the fused GDN kernels"; print("kernel overlay OK", fla.__version__, causal_conv1d.__version__)' \
          || f22_fatal "serving venv variant$tag: fla / causal_conv1d do not import (or transformers does not see them) — the GDN kernel overlay did not build";;
      *)
        "$SPY" -c 'import mamba_ssm, causal_conv1d; from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined; from causal_conv1d import causal_conv1d_fn; print("kernel overlay OK", mamba_ssm.__version__, causal_conv1d.__version__)' \
          || f22_fatal "serving venv variant$tag: mamba_ssm / causal_conv1d do not import — the kernel overlay did not build"
        ;;
    esac
  fi
  unset VLLM_PLUGINS 2>/dev/null || true
  # vs.launch_server shells out to `vllm`: the serving venv's bin/ must lead PATH (mt1 smoke 375475)
  export PATH="$SERVE_VENV/bin:$PATH"
  # THE VENV IDENTITY every serving record carries: serving_attestation.serving_venv_facts reads these beside
  # the serving process's own sys.prefix and the kernels it can import, so endpoint.json / cells_served.json / the attestation say
  # which venv (variant + requirements hash) served — whether a standup ran before or after a variant landed
  export SISTEER_SERVING_VENV="$SERVE_VENV" SISTEER_SERVING_VENV_VARIANT="${tag#-}" SISTEER_SERVING_REQS_SHA256="$SERVE_REQS_SHA"
  # THE ENGINE IDENTITY's base layer: the engine is venv-independent — vLLM vendors its GDN / causal-conv1d kernels and the
  # overlays reach only the HF+PEFT reference — so the admission gate compares {vllm version, BASE requirements digest, flashinfer
  # version}; the base digest is the sha256 of configs/capeval/vllm-requirements.txt ALONE (no overlay), read by serving_venv_facts
  export SISTEER_SERVING_BASE_REQS_SHA256="$(sha256sum "$reqs" | cut -c1-64)"
}

f22_client_venv(){   # -> CPY, CLIENT_VENV (the tinycap CLIENT venv: inspect_ai + inspect_evals + the driver's deps; CPU-only)
  # The driver, the usage ledger and the smoke probe run from the SAME hashed
  # lock the Modal launcher builds /opt/client from (configs/capeval/tinycap-client-requirements.txt, resolved under
  # --exclude-newer): `uv venv --python 3.12`, the hashed lines with --no-deps --require-hashes, the VCS line (the
  # IFEval checker at its full commit — unhashable by construction) installed at exactly that commit, then an
  # import check. Node-local, hash-versioned, flock-guarded like the other bootstraps. torch here is the lock's CPU
  # build: the driver runs with CUDA_VISIBLE_DEVICES="" and never touches the GPU.
  local lock="$REPO_DIR/configs/capeval/tinycap-client-requirements.txt" h hf lockdir sent hashed vcs
  [ -f "$lock" ] || f22_fatal "no client lock at $lock"
  h=$(sha256sum "$lock" | cut -c1-8); hf=$(sha256sum "$lock" | cut -c1-16)
  CLIENT_VENV="/home/$USER/venvs/tinycap-client-$h"; lockdir=$(dirname "$CLIENT_VENV"); mkdir -p "$lockdir"
  sent="$lockdir/.$(basename "$CLIENT_VENV").client-bootstrap"; hashed="$lockdir/.$(basename "$CLIENT_VENV").hashed.txt"
  (
    flock -x 9 || exit 99
    if [ -f "$sent" ] && [ "$(cat "$sent")" = "$hf" ] && [ -x "$CLIENT_VENV/bin/python" ]; then
      f22_log "client venv already bootstrapped"
    else
      [ -x "$CLIENT_VENV/bin/python" ] || uv venv --python 3.12 "$CLIENT_VENV" || exit 1
      # the VCS allow-list = contracts.CLIENT_VCS_PINS (the josejg IFEval checker): split out by NAME here,
      # installed at its exact commit below
      grep -v -E '^(instruction-following-eval) @ ' "$lock" > "$hashed"
      # The client lock is fully hash-pinned (every artifact named by sha256), so the repo's
      # [tool.uv] exclude-newer cutoff (2026-08-03, read from pyproject.toml when uv runs
      # inside the checkout) adds no protection here and refuses pins the lock legitimately
      # carries (agent-client-protocol is pinned at 0.12.1, published 2026-08-16: "no
      # version" on the cluster, every m5t job dead at the client-venv guard). Resolve the
      # hashed lock under a rolling one-week cutoff instead — the >= 1-week-old rule holds.
      # --no-config as well: a command-line --exclude-newer replaces only the table's global
      # cutoff, never its exclude-newer-package entries, and the table's anthropic entry
      # (2026-08-21) refuses the lock's anthropic 1.1.0 (published 2026-08-26), so every first
      # build of a client venv failed inside a checkout. The lock is compiled with --no-config
      # (its header), so the install reads no project configuration either.
      uv pip install --no-config --python "$CLIENT_VENV/bin/python" --no-deps --require-hashes \
        --exclude-newer "$(date -u -d "-7 days" +%Y-%m-%dT00:00:00Z)" -r "$hashed" || exit 1
      grep -E '^(instruction-following-eval) @ ' "$lock" | sed 's/[[:space:]]*\\$//' | while read -r vcs; do
        [ -n "$vcs" ] && { uv pip install --python "$CLIENT_VENV/bin/python" --no-deps "$vcs" || exit 1; }
      done || exit 1
      echo "$hf" > "$sent"
    fi
  ) 9>"$lockdir/.$(basename "$CLIENT_VENV").uv-sync.lock"
  CPY="$CLIENT_VENV/bin/python"; [ -x "$CPY" ] || f22_fatal "client venv bootstrap failed"
  (cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" "$CPY" -c 'import inspect_ai, inspect_evals, sisteer.capeval.tinycap.contracts, sisteer.capeval.serving.usage_ledger; print("client venv OK", inspect_ai.__version__)') \
    || f22_fatal "client venv guard: inspect_ai / inspect_evals / the tinycap contracts / the usage ledger do not import"
}

f22_inspect_controls(){   # $1 = a comma list of controls -> the members that are INSPECT controls, comma-joined ("" when none)
  # contracts.INSPECT_CONTROLS read BY IMPORT under the driver venv (the way the jobs read MMLU_TOP_LOGPROBS) — never a
  # hand-copied list: the Inspect vocabulary is the dispatch table's, and a copy here would drift from it
  env PYTHONPATH="$REPO_DIR/src" "$MPY" -c 'import sys
from sisteer.capeval.tinycap import contracts as tc
want = [c for c in sys.argv[1].split(",") if c]
print(",".join(c for c in want if c in tc.INSPECT_CONTROLS))' "$1"
}
f22_datasets_guard(){   # $1 = the INSPECT controls (comma list) the controls driver will run -> HF_DATASETS_CACHE exported and guarded
  # The `datasets` library, OFFLINE (hf_env.sh sets HF_HUB_OFFLINE=1), reads a PREPARED arrow cache under HF_DATASETS_CACHE
  # and looks a dataset up by the config dir its LOAD SIGNATURE names — hence the gsm8k lane's rule (g8k_common.sh g8k_init /
  # g8k_datasets_guard): the prepared cache under $SHARED/hf_datasets, never the shared model cache (whose
  # openai/gsm8k sits under config `main` and three hashed defaults only, so inspect_evals' `data_dir="main"` load misses
  # it — smoke 502566). Each requested Inspect control names its dataset dir + offline config dir through the
  # gsm8k lane's literals (g8k_cells.DATASETS_CONFIG_DIRS; the control -> dataset binding is declared below, keyed by
  # contracts' control names) and must be PRESENT, by name, before any GPU standup; an Inspect control with no recipe here
  # refuses rather than dying at the first load minutes into a standup. A symlinked config dir passes; a
  # dangling one refuses. Necessary, not sufficient: the leg's own offline load is the authority.
  local ctls="$1" dirs d target
  [ -n "$ctls" ] || return 0
  export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$SHARED/hf_datasets}"
  dirs=$(env PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts/sdfbase/gsm8k" "$MPY" - "$ctls" <<'PYG'
import sys
import g8k_cells as g
from sisteer.capeval.tinycap import contracts as tc
by_control = {tc.CONTROL_GSM8K: "openai___gsm8k", tc.CONTROL_GSM8K_FULL: "openai___gsm8k"}   # inspect_evals' gsm8k task: openai/gsm8k, data_dir main
out = []
for c in [c for c in sys.argv[1].split(",") if c]:
    ds = by_control.get(c)
    if ds is None:
        sys.exit(f"no prepared-datasets recipe on the cluster for the Inspect control {c!r} (f22_datasets_guard binds {sorted(by_control)}; add the control's dataset and its offline config dir, g8k_cells.DATASETS_CONFIG_DIRS)")
    out.append(f"{ds}/{g.DATASETS_CONFIG_DIRS[ds]}")
print(" ".join(dict.fromkeys(out)))
PYG
  ) || f22_fatal "datasets guard: cannot name the offline config dirs for the Inspect controls '$ctls' (see the refusal above)"
  for d in $dirs; do
    if [ -d "$HF_DATASETS_CACHE/$d" ]; then
      target=$(readlink "$HF_DATASETS_CACHE/$d" 2>/dev/null || true)
      f22_log "datasets guard: $HF_DATASETS_CACHE/$d present${target:+ (symlink -> $target)}"
    else
      f22_fatal "HF_DATASETS_CACHE=$HF_DATASETS_CACHE carries no offline config dir $d for the Inspect controls '$ctls': the datasets library looks OFFLINE for the config dir named by the load signature (openai/gsm8k: default-data_dir=main) while an ONLINE prepare names the arrow cache by a kwargs hash. RECIPE (the gsm8k lane's D18): prepare ONLINE once per user per cluster with HF_HUB_OFFLINE=0 set explicitly at the task's exact signature (inspect_evals' gsm8k: openai/gsm8k data_dir main at its pinned revision) into this HF_DATASETS_CACHE, then alias the hash-named dir: ln -s <hash dir> $HF_DATASETS_CACHE/$d"
    fi
  done
}

# --- the served target: standup + the STACKED-mode assertion from the authoritative stamps -------
f22_standup(){   # $1 = mode (attacks|petri), $2 = out dir, $3 = proxy port ("" = free),
                 # $4 = geometry-table eval ("" = the mode's default), $5 = max_model_len (""), $6 = max_num_seqs ("")
                 # $7 = vLLM --max-logprobs ("" = the engine default, 20; the controls standup passes the
                 #      driver's MMLU_TOP_LOGPROBS — its MMLU scoring is refused above the cap, 447405)
                 # needs SPY, KEY, CFG, MANIFEST, TT_TWO_SI_DEFENSES_DIR
                 # F22_STANDUP_EXTRA (a Bash ARRAY, optional): appended VERBATIM to mt_serving_up.py's argv — the
                 # math500 jobs pass --upstream-timeout-s / --lens-salt-policy / --cell / --expect-object-sha256 /
                 # --reasoning-parser through it; unset or empty = the argv exactly as before.
                 # A lane that must RECORD the sampler the engine resolved (not the one the client sent) passes
                 # --enable-log-requests through it too (f22_standup_extra_reset --enable-log-requests: an inherited
                 # scalar is the launch-time hook and must survive — never a bare assignment or a bare `+=` first touch;
                 # a --serving-path riding it is refused by name, an empty inherited scalar is zero arguments; afterwards
                 # scripts/sdfbase/record_resolved_sampler.py reads "$out/vllm_server.log" — the prompt is a DEBUG record
                 # and is NOT logged at INFO). That is the forwarding path on this tree; $7 is --max-logprobs, so no
                 # positional is added for it.
                 # THINKING REGIME (the sampled run's GSM8K readout, 2026-09-10; docs/served_thinking_on.md): the shell
                 # variable F22_THINKING_ON (non-empty = ON; a plain variable, never exported through Slurm) adds
                 # `--thinking-on` to the standup — the launch pins the ON render, the served header is certified per
                 # template BEFORE any GPU minute, the proxy requires petri_thinking_on on every lens-route request. The
                 # regime is then asserted back from the AUTHORITATIVE stamps (endpoint.json + /v1/models petri_proxy),
                 # never from this variable: an ON standup that attests OFF (or the reverse) refuses traffic. Default
                 # (unset) is today's thinking-OFF standup byte for byte, and its attested mode must read "off".
  local mode="$1" out="$2" pport="${3:-}" geval="${4:-}" gL="${5:-}" gS="${6:-}" gLP="${7:-}" i
  # a DEFERRED serving path (f22_select_serving_path: a job that owns its wave root, or F22_SERVING_PATH_DEFER=1, and never bound)
  # is bound HERE from the launch's ACTUAL manifest — the one this standup serves — before anything below reads the path
  [ "${F22_SERVING_PATH_MODE:-}" != deferred ] || f22_bind_serving_path "${MANIFEST:?f22_standup: the serving path was deferred to the standup, which needs MANIFEST}"
  local want_thinking="off"; [ -n "${F22_THINKING_ON:-}" ] && want_thinking="on"
  # THE SERVER-DEFAULT SAMPLING POLICY (vllm_server.SERVER_DEFAULT_POLICIES): the SCALAR
  # F22_SERVER_DEFAULT_POLICY (contract_override | engine_defaults; a scalar crosses sbatch through Slurm's implicit export,
  # which a Bash array like F22_STANDUP_EXTRA does not) rides to mt_serving_up.py --server-default-policy and is asserted back
  # below from the AUTHORITATIVE stamps (endpoint.json server_default_policy + /v1/models petri_proxy.server_default_policy),
  # never from this variable. Unset = the standup's own default (contract_override on this tree), the argv byte for byte.
  # F22_LOG_REQUESTS=1 (scalar, same reason) adds --enable-log-requests — the opt-in that makes the engine log the sampler it
  # RESOLVED per request, which record_resolved_sampler.py / anchor_sampler_preflight.py read back; the prompt stays unlogged.
  local want_sdp="${F22_SERVER_DEFAULT_POLICY:-}"
  case "$want_sdp" in ""|contract_override|engine_defaults) ;;
    *) f22_log "REFUSING TO STAND UP: F22_SERVER_DEFAULT_POLICY must be contract_override or engine_defaults, got '$want_sdp'"; return 1;; esac
  case "${F22_LOG_REQUESTS:-}" in ""|0|1) ;; *) f22_log "REFUSING TO STAND UP: F22_LOG_REQUESTS must be 1 (or unset / 0), got '${F22_LOG_REQUESTS}'"; return 1;; esac
  local log_requests_flag=""; [ "${F22_LOG_REQUESTS:-0}" = 1 ] && log_requests_flag="--enable-log-requests"
  # THE SERVING PATH (native-serving-default): the lens pins ride the argv ONLY under lens — the native engine
  # attests `engine`, not a reduction mode, and mt_serving_up refuses the pins by name there
  local serving_path="${F22_SERVING_PATH:-lora_native}" lens_pins=()
  [ "$serving_path" = lens ] && lens_pins=(--reduction-mode stacked --require-attested-mode stacked)
  mkdir -p "$out"; rm -f "$out/STOP" "$out/endpoint.json"
  # THE SELECTION RECORD (kind-aware default, 2026-09-15): how the path was chosen rides every record of this standup —
  # mt_serving_up.py --serving-path-selection copies it into cells_served.json / endpoint.json as `serving_path_selection`
  # (the standup refuses a record whose `selected` is not the path it serves), and the standup_jobs.jsonl line below
  # carries the same object. Unset (a caller that never ran f22_init) = the argv exactly as before.
  local -a sel_args=()
  if [ -n "${F22_SERVING_PATH_SELECTION_JSON:-}" ]; then
    printf '%s\n' "$F22_SERVING_PATH_SELECTION_JSON" > "$out/serving_path_selection.json"
    sel_args=(--serving-path-selection "$out/serving_path_selection.json")
  fi
  # FREE-MEMORY PREFLIGHT (2026-09-07): the engine reserves GPU_MEM_UTIL of the card at startup and
  # raises from deep inside vLLM when a NEIGHBOUR already holds too much ("Free memory on device
  # cuda:0 (96.49/139.8 GiB) ... less than desired GPU memory utilization (0.9, 125.82 GiB)" —
  # f22-att-gemma4e4b-crescendo 74031 on a node where six other users' jobs ran). Slurm gives us the
  # device in CUDA_VISIBLE_DEVICES and we only ever READ it (never set it): query that exact index
  # and refuse here, with the numbers, instead of after a 90-second model load.
  local dev free_mib total_mib need_mib util
  # GPU_MEM_UTIL (unset = the standup's 0.9) is ONE number for the preflight AND the launch (--gpu-mem-util below):
  # a job that needs more KV than 0.9 leaves (petri_proxy.sbatch for glm46vflash) raises it in one place
  dev="${TARGET_GPU:-${CUDA_VISIBLE_DEVICES:-}}"; util="${GPU_MEM_UTIL:-0.9}"
  case "$dev" in *,*) dev="${dev%%,*}";; esac
  if [ -n "$dev" ] && command -v nvidia-smi >/dev/null 2>&1; then
    read -r total_mib free_mib < <(nvidia-smi --query-gpu=memory.total,memory.free --format=csv,noheader,nounits -i "$dev" 2>/dev/null | tr -d ' ' | tr ',' ' ')
    if [ -n "${free_mib:-}" ] && [ -n "${total_mib:-}" ]; then
      need_mib=$(awk -v t="$total_mib" -v u="$util" 'BEGIN{printf "%d", t*u}')
      if [ "$free_mib" -lt "$need_mib" ]; then
        f22_log "REFUSING TO STAND UP: gpu $dev has ${free_mib} MiB free of ${total_mib} MiB, but gpu_memory_utilization ${util} wants ${need_mib} MiB — another process on this card is holding $(( total_mib - free_mib )) MiB. This is a DIRTY/SHARED DEVICE, not a geometry problem: do not shrink the geometry for it; resubmit (Slurm will usually place elsewhere) or have the occupant cleared."
        return 1
      fi
      f22_log "gpu $dev preflight: ${free_mib} MiB free of ${total_mib}; utilization ${util} needs ${need_mib} MiB — OK"
    else
      f22_log "gpu $dev preflight: nvidia-smi returned nothing for this index; proceeding (the engine still refuses on its own)"
    fi
  fi
  # THE PROBE PREFLIGHT: before a GPU is allocated, validate that the load check's
  # probe — the RUN's cap from $CFG + the leg's rendered p99 prompt — fits the window this standup will serve;
  # a probe the window cannot carry is INCOMPATIBLE (exit 4) and refuses HERE, naming the geometry the shared
  # budget resolves for it — never a predictable 400 read as a measured limit after the server is up
  local lc_eval="${geval:-$( [ "$mode" = petri ] && echo petri || echo multiturn_attacks )}"   # the leg the load check probes, here and after the standup
  if [ "${F22_LOAD_CHECK:-1}" = 1 ]; then
    # F22_LOAD_CHECK_ATTACK (the multi-turn family, or comma list, this standup serves — target_attacks.sbatch / the re-run pass
    # set it) resolves a PER-FAMILY config cap (attacks.multiturn_target_max_new_tokens as a map); the
    # probe takes the max over the families named; unset = the flat key / the table, byte-identical to before
    if ! (cd "$REPO_DIR" && "$SPY" scripts/sdfbase/full22/load_check.py --preflight --config "$CFG" --eval "$lc_eval" \
          --model-key "$KEY" ${gL:+--max-model-len "$gL"} ${F22_LOAD_CHECK_CAP:+--cap "$F22_LOAD_CHECK_CAP"} \
          ${F22_LOAD_CHECK_SYSTEM_SI:+--system-si "$F22_LOAD_CHECK_SYSTEM_SI"} ${F22_LOAD_CHECK_ATTACK:+--attack "$F22_LOAD_CHECK_ATTACK"} \
          --out "$out/load_check_preflight.json" > "$out/load_check_preflight.log" 2>&1); then
      f22_log "REFUSING TO STAND UP: the load check's probe does not fit this standup's window — $(tail -2 "$out/load_check_preflight.log" | tr '\n' ' ')"
      return 1
    fi
    f22_log "load check preflight: $(tail -1 "$out/load_check_preflight.log")"
  fi
  # THE STANDUP EXTRAS, normalised at the splice (2026-09-16): a --serving-path element refuses by name whoever put it
  # there (f22_standup_extra_inherited_check); when NO lane rebuilt the array (F22_STANDUP_INHERITED_TAKEN unset — the
  # raw inherited scalar) an exported-EMPTY value is zero arguments, byte-identical to unset, never an empty argv element
  # the standup parser rejects; once a lane rebuilt it (the helper already normalised the inherited tokens) every
  # element passes verbatim, so an empty LANE flag stays visible as the parser error it is
  f22_standup_extra_inherited_check
  local -a standup_extra=()
  local t
  for t in ${F22_STANDUP_EXTRA[@]+"${F22_STANDUP_EXTRA[@]}"}; do
    if [ -n "$t" ] || [ -n "${F22_STANDUP_INHERITED_TAKEN:-}" ]; then standup_extra+=("$t"); fi
  done
  # AN ARM'S OBJECT IS PINNED IN ITS OWN LANE'S PINS FILE, AND THE NATIVE STANDUP READS ONLY THE WAVE'S
  # (arditi-qwen-all-positions F1, 2026-09-17; first exercised by job 109820, which refused in 1m46s).
  # mt_serving_up.PINS_FILE is the MODULE CONSTANT configs/sdfbase_full22_pins.json and no flag names another:
  # with no --expect-object-sha256 the native path derives each served object's digest from the bank view's
  # VIEW_MANIFEST.json + THAT file, so an arm whose rung lives in configs/<arm>_pins.json — this arm by design,
  # and the wave pins carry NO arm staged id at all (verified: 0 of the 59 qwen35_4b keys) — refuses by name,
  # "is not pinned in .../sdfbase_full22_pins.json — refusing an unpinned object". The wave pins are SHARED
  # state every peer leg of this (target, view) verifies against its roster, so registering a lane's rung there
  # is the very thing f22_arm_bank_view exists to avoid; the standup's own message names the alternative, which
  # is this. The earlier arm legs never met the gate because they ran on the LENS path, where the digests come
  # from elsewhere — a first-exercise defect, not a regression.
  #
  # WHAT TRAVELS IS THE PIN, NOT THE OBSERVED HASH. F22_ARM_OBJECT_SHA256 is the ARMS TABLE's operating point
  # (`pinned_object_sha256`); F22_ARM_SERVED_OBJECT_SHA256 is what arm_cell_check hashed off the object about to
  # be served. Handing the standup the observed value would make its own re-hash a tautology — bytes certified
  # against themselves — so the authority of record is what is passed and the standup's hash stays an
  # independent check. arm_cell_check already refused unless the two agree; ar_common.sh re-asserts that
  # equality before it stamps anything, and so does this, because a guard on one side only is a guard on
  # neither. Inert everywhere else: only f22_arm_bank_view exports these names, only lora_native reaches the
  # wave-pins derivation, and a digest a lane put on the array itself is never overridden.
  if [ "$serving_path" = lora_native ] && [ -n "${F22_ARM_CELL:-}" ] && [ -n "${F22_ARM_OBJECT_SHA256:-}" ]; then
    case " ${standup_extra[*]:-} " in
      *" --expect-object-sha256 "*|*" --expect-object-sha256="*) ;;
      *)
        [ "${F22_ARM_SERVED_OBJECT_SHA256:-}" = "$F22_ARM_OBJECT_SHA256" ] || f22_fatal \
          "arm ${ARM:-?} cell $F22_ARM_CELL: the bytes in the view (${F22_ARM_SERVED_OBJECT_SHA256:-unset}) are not the arms table's pinned object ($F22_ARM_OBJECT_SHA256) — refusing to hand the standup a digest for bytes other than the ones certified"
        standup_extra+=(--expect-object-sha256 "$F22_ARM_CELL=$F22_ARM_OBJECT_SHA256")
        f22_log "arm ${ARM:-?}: the standup is held to the ARMS TABLE's pin (--expect-object-sha256 $F22_ARM_CELL=${F22_ARM_OBJECT_SHA256:0:12}…) — an arm's rung is pinned in its own lane's pins file, which the native standup's wave-pins derivation cannot read";;
    esac
  fi
  STANDUP_STOP="$out/STOP"
  CUDA_VISIBLE_DEVICES="${TARGET_GPU:-$CUDA_VISIBLE_DEVICES}" "$SPY" scripts/sdfbase/mt_serving_up.py --repo-dir "$REPO_DIR" \
    --config "$CFG" --manifest "$MANIFEST" --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" \
    --out "$out" --stop-file "$out/STOP" --mode "$mode" --model-key "$KEY" \
    --serving-path "$serving_path" ${lens_pins[@]+"${lens_pins[@]}"} \
    ${sel_args[@]+"${sel_args[@]}"} \
    ${pport:+--proxy-port "$pport"} ${geval:+--eval "$geval"} ${gL:+--max-model-len "$gL"} ${gS:+--max-num-seqs "$gS"} ${gLP:+--max-logprobs "$gLP"} \
    ${GPU_MEM_UTIL:+--gpu-mem-util "$GPU_MEM_UTIL"} \
    ${standup_extra[@]+"${standup_extra[@]}"} \
    ${F22_THINKING_ON:+--thinking-on} \
    ${want_sdp:+--server-default-policy "$want_sdp"} ${log_requests_flag:+"$log_requests_flag"} \
    > "$out/standup.log" 2>&1 &
  TARGET_PID=$!
  for i in $(seq 1 480); do
    [ -f "$out/endpoint.json" ] && break
    kill -0 "$TARGET_PID" 2>/dev/null || { f22_log "target standup died (see $out/standup.log)"; tail -30 "$out/standup.log"; return 1; }
    sleep 5
  done
  [ -f "$out/endpoint.json" ] || { f22_log "target endpoint not up after 40 min"; return 1; }
  local ep="$out/endpoint.json"
  # THE DURABLE COPY: mt_serving_up.py UNLINKS endpoint.json at teardown, so a LATER job that
  # must read this standup's record — the sampled ladder's 16,384 rung reads the 8,192 standup's recorded proxy hop through
  # f22s_prev_endpoint — found nothing and refused (8 legs). The copy is byte-identical to the READY record and is never
  # read by this standup's own pollers (they key on endpoint.json).
  cp -f "$ep" "$out/endpoint.recorded.json" 2>/dev/null || f22_log "note: could not keep $out/endpoint.recorded.json (the READY record's durable copy)"
  PROXY_URL=$("$SPY" -c "import json,sys;print(json.load(open(sys.argv[1]))['proxy_url'])" "$ep")
  TABLE=$("$SPY" -c "import json,sys;print(json.load(open(sys.argv[1]))['routing_table'])" "$ep")
  TARGET_L=$("$SPY" -c "import json,sys;print(json.load(open(sys.argv[1]))['max_model_len'])" "$ep")
  TARGET_S=$("$SPY" -c "import json,sys;print(json.load(open(sys.argv[1]))['max_num_seqs'])" "$ep")
  # THE AUTHORITATIVE STAMPS, read back — never the env we set: (1) endpoint.json's ENGINE (`engine` — a native
  # standup's lora_native; absent on a lens standup, whose lens_reduction_mode the standup copied from the workers'
  # install + scheduler-config reports and already refused under --require-attested-mode) and its attested_max_num_seqs /
  # attested_max_model_len (vLLM's LIVE scheduler config, refused by the standup when it differs from the
  # resolved geometry — a silently clamped width is visible here, never inferred from throughput);
  # (2) the proxy's own /v1/models `petri_proxy` identity block, which the clients read: `lens_reduction_mode`
  # on a lens proxy, `engine.kind` + `engine.adapters` on a native one.
  local ep_engine ep_mode px_mode att_s att_l px_kind px_adapters ep_adapters ep_eager ep_graphs ep_fullmax
  ep_engine=$("$SPY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('engine'))" "$ep")
  ep_mode=$("$SPY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('lens_reduction_mode'))" "$ep")
  att_s=$("$SPY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('attested_max_num_seqs'))" "$ep")
  att_l=$("$SPY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('attested_max_model_len'))" "$ep")
  local px_models; px_models=$(curl -s -m 20 "$PROXY_URL/models")
  px_mode=$(printf '%s' "$px_models" | "$SPY" -c "import json,sys;print((json.load(sys.stdin).get('petri_proxy') or {}).get('lens_reduction_mode'))" 2>/dev/null)
  if [ "$serving_path" = lens ]; then
    # the LENS gate, byte for byte: the attested STACKED mode on both stamps
    if [ "$ep_engine" != None ]; then
      f22_log "REFUSING TRAFFIC: F22_SERVING_PATH=lens but endpoint.json engine=$ep_engine (a native standup answered a lens job)"
      touch "$out/STOP"; return 1
    fi
    if [ "$ep_mode" != stacked ] || [ "$px_mode" != stacked ]; then
      f22_log "REFUSING TRAFFIC: endpoint.json lens_reduction_mode=$ep_mode, /v1/models petri_proxy.lens_reduction_mode=$px_mode (wave mode: stacked)"
      touch "$out/STOP"; return 1
    fi
  else
    # THE NATIVE gate: endpoint.json engine == lora_native, the proxy's engine
    # block names the same engine and EXACTLY the attested adapter names (endpoint.id_name_map), the workers attest
    # a graphs-on engine (enforce_eager false, cudagraph_mode not NONE) — and NO lens mode anywhere
    px_kind=$(printf '%s' "$px_models" | "$SPY" -c "import json,sys;print(((json.load(sys.stdin).get('petri_proxy') or {}).get('engine') or {}).get('kind'))" 2>/dev/null)
    px_adapters=$(printf '%s' "$px_models" | "$SPY" -c "import json,sys;print(','.join(sorted(((json.load(sys.stdin).get('petri_proxy') or {}).get('engine') or {}).get('adapters') or {})))" 2>/dev/null)
    ep_adapters=$("$SPY" -c "import json,sys;print(','.join(sorted((json.load(open(sys.argv[1])).get('id_name_map') or {}).values())))" "$ep")
    ep_eager=$("$SPY" -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1])).get('attested_enforce_eager')))" "$ep")
    ep_graphs=$("$SPY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('attested_cudagraph_mode'))" "$ep")
    if [ "$ep_engine" != lora_native ] || [ "$px_kind" != lora_native ]; then
      f22_log "REFUSING TRAFFIC: F22_SERVING_PATH=lora_native but endpoint.json engine=$ep_engine, /v1/models petri_proxy.engine.kind=$px_kind"
      touch "$out/STOP"; return 1
    fi
    if [ "$ep_mode" != None ] || [ "$px_mode" != None ]; then
      f22_log "REFUSING TRAFFIC: a native standup carries a lens_reduction_mode (endpoint $ep_mode / proxy $px_mode) — two engines on one record"
      touch "$out/STOP"; return 1
    fi
    if [ -z "$ep_adapters" ] || [ "$px_adapters" != "$ep_adapters" ]; then
      f22_log "REFUSING TRAFFIC: the proxy's engine adapters [$px_adapters] != the attested id_name_map names [$ep_adapters]"
      touch "$out/STOP"; return 1
    fi
    if [ "$ep_eager" != false ] || [ "$ep_graphs" = None ] || [ "$ep_graphs" = NONE ]; then
      f22_log "REFUSING TRAFFIC: the native engine attests enforce_eager=$ep_eager cudagraph_mode=$ep_graphs (graphs must be ON on the plugin-free engine)"
      touch "$out/STOP"; return 1
    fi
    # THE FULL CAPTURE SURFACE (2026-09-16, Modal H200 / Qwen3.5-4B, vLLM 0.25.1): graphs-on is not enough. vLLM captures
    # FULL decode graphs only for capture sizes <= max_num_seqs, so S=51 on the default list [1,2,4,8,...,96] left every
    # 49-51 sequence decode step PIECEWISE with attention eager (FULL largest=48; 58 -> 9 tok/s/seq) while cudagraph_mode
    # read FULL_AND_PIECEWISE. The standup writes the dispatcher's largest FULL capture size; it must be an int >= the
    # attested max_num_seqs (readiness refused otherwise; this is the wave's own read of the same fact)
    ep_fullmax=$("$SPY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('attested_cudagraph_full_capture_max'))" "$ep")
    case "$ep_fullmax" in
      ''|*[!0-9]*)
        f22_log "REFUSING TRAFFIC: the native engine attests no FULL CUDA-graph capture size (attested_cudagraph_full_capture_max=$ep_fullmax) — every decode batch would run PIECEWISE with attention eager under a graphs-on record"
        touch "$out/STOP"; return 1;;
    esac
    case "$att_s" in
      ''|*[!0-9]*) ;;   # an unattested S is refused by the geometry compare below
      *)
        if [ "$ep_fullmax" -lt "$att_s" ]; then
          f22_log "REFUSING TRAFFIC: the largest FULL CUDA-graph capture size $ep_fullmax < the attested max_num_seqs $att_s (decode batches of $((ep_fullmax + 1))..$att_s sequences run PIECEWISE with attention eager — the S=51/48 collapse; max_num_seqs must be a captured size)"
          touch "$out/STOP"; return 1
        fi;;
    esac
  fi
  if [ "$att_s" != "$TARGET_S" ] || [ "$att_l" != "$TARGET_L" ]; then
    f22_log "REFUSING TRAFFIC: resolved geometry ${TARGET_S} x ${TARGET_L} but the workers attest ${att_s} x ${att_l} (a clamped or unattested geometry)"
    touch "$out/STOP"; return 1
  fi
  # THE GEOMETRY ADJUSTMENTS: a requested S may be served LOWER only through a RECORDED adjustment chain
  # (geometry.adjustments: the LoRA kernel index ceiling, then a KV-derived shrink) whose first step starts at the
  # requested S and whose last step is the attested S; a bare smaller S is refused as before
  local adj_from adj_to adj_kinds
  adj_kinds=$("$SPY" -c "import json,sys;print(','.join(a.get('kind','?') for a in ((json.load(open(sys.argv[1])).get('geometry') or {}).get('adjustments') or [])))" "$ep" 2>/dev/null || echo "")
  adj_from=$("$SPY" -c "import json,sys;a=((json.load(open(sys.argv[1])).get('geometry') or {}).get('adjustments') or []);print(a[0].get('from_S','') if a else '')" "$ep" 2>/dev/null || echo "")
  adj_to=$("$SPY" -c "import json,sys;a=((json.load(open(sys.argv[1])).get('geometry') or {}).get('adjustments') or []);print(a[-1].get('to_S','') if a else '')" "$ep" 2>/dev/null || echo "")
  if [ -n "$adj_kinds" ] && [ "$adj_to" != "$att_s" ]; then
    f22_log "REFUSING TRAFFIC: geometry.adjustments [$adj_kinds] end at S=$adj_to but the workers attest $att_s"; touch "$out/STOP"; return 1
  fi
  if [ -n "$gS" ] && [ "$att_s" != "$gS" ]; then
    if [ -n "$adj_kinds" ] && [ "$adj_from" = "$gS" ]; then
      f22_log "requested max_num_seqs $gS served at $att_s through the RECORDED adjustment chain [$adj_kinds] ($adj_from -> $adj_to)"
    else
      f22_log "REFUSING TRAFFIC: requested max_num_seqs $gS but the server attests $att_s (no recorded adjustment chain from $gS: [${adj_kinds:-none}] ${adj_from:-}->${adj_to:-})"; touch "$out/STOP"; return 1
    fi
  fi
  # THE THINKING REGIME, read back from both stamps (never from F22_THINKING_ON): endpoint.json thinking_mode and the
  # proxy's identity block must BOTH say what this standup asked for; under ON the template's certification verdict
  # must be servable (certified | trivially_in_support — mt_serving_up already refused anything else before launch,
  # this is the job-side belt) and the proxy->vLLM hop timeout must be on the record (upstream_timeout_s; the
  # sampled ladder's hop_timeout_guard sizes it against the rung). TARGET_THINKING_MODE / TARGET_THINKING_VERDICT /
  # TARGET_UPSTREAM_TIMEOUT_S are exported to the phase for its own records.
  local ep_think px_think think_verdict think_verdict_json
  ep_think=$("$SPY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('thinking_mode'))" "$ep")
  px_think=$(curl -s -m 20 "$PROXY_URL/models" | "$SPY" -c "import json,sys;print((json.load(sys.stdin).get('petri_proxy') or {}).get('thinking_mode'))" 2>/dev/null)
  # the verdict as a bare word for the case below, and as JSON for the record; the hop timeout as JSON (a missing field
  # is `null`, never Python's repr `None` — which is not JSON and made standup_jobs.jsonl unreadable, 2026-09-10 review)
  think_verdict=$("$SPY" -c "import json,sys;v=((json.load(open(sys.argv[1])).get('thinking_certification') or {}).get('verdict'));print(v if isinstance(v, str) else '')" "$ep")
  think_verdict_json=$("$SPY" -c "import json,sys;print(json.dumps(sys.argv[1] or None))" "$think_verdict")
  TARGET_UPSTREAM_TIMEOUT_S=$("$SPY" -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1])).get('upstream_timeout_s')))" "$ep")
  if [ "$ep_think" != "$want_thinking" ] || [ "$px_think" != "$want_thinking" ]; then
    f22_log "REFUSING TRAFFIC: this standup asked for thinking $want_thinking but endpoint.json thinking_mode=$ep_think, /v1/models petri_proxy.thinking_mode=$px_think"
    touch "$out/STOP"; return 1
  fi
  if [ "$want_thinking" = on ]; then
    case "$think_verdict" in certified|trivially_in_support) ;; *)
      f22_log "REFUSING TRAFFIC: thinking-ON standup but the template's certification verdict is '$think_verdict' (need certified | trivially_in_support)"
      touch "$out/STOP"; return 1;;
    esac
  fi
  TARGET_THINKING_MODE="$ep_think"; TARGET_THINKING_VERDICT="$think_verdict"
  # THE SERVER-DEFAULT SAMPLING POLICY, read back from both stamps (never from F22_SERVER_DEFAULT_POLICY): when the caller
  # asked for one, endpoint.json server_default_policy AND the proxy's identity block must both name it — a standup whose
  # records lack the key (a tree before the policy existed) cannot honour the ask and refuses traffic; unset = whatever the
  # standup composed is recorded (contract_override on this tree), and the two stamps must still agree when present.
  # TARGET_SERVER_DEFAULT_POLICY is exported to the phase for its own records (`None` on a pre-key standup).
  local ep_sdp px_sdp
  ep_sdp=$("$SPY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('server_default_policy'))" "$ep")
  px_sdp=$(printf '%s' "$px_models" | "$SPY" -c "import json,sys;print((json.load(sys.stdin).get('petri_proxy') or {}).get('server_default_policy'))" 2>/dev/null)
  if [ -n "$want_sdp" ]; then
    if [ "$ep_sdp" != "$want_sdp" ] || [ "$px_sdp" != "$want_sdp" ]; then
      f22_log "REFUSING TRAFFIC: this standup asked for server-default sampling policy $want_sdp but endpoint.json server_default_policy=$ep_sdp, /v1/models petri_proxy.server_default_policy=$px_sdp"
      touch "$out/STOP"; return 1
    fi
  elif [ "$ep_sdp" != None ] && [ "$px_sdp" != None ] && [ "$ep_sdp" != "$px_sdp" ]; then
    f22_log "REFUSING TRAFFIC: endpoint.json server_default_policy=$ep_sdp but /v1/models petri_proxy.server_default_policy=$px_sdp — two policies on one standup"
    touch "$out/STOP"; return 1
  fi
  TARGET_SERVER_DEFAULT_POLICY="$ep_sdp"
  f22_model_facts "$ep" "$att_s" || { touch "$out/STOP"; return 1; }
  # M-RoPE DISPATCH GATE: the standup asked every worker which rotary the engine
  # actually runs (mt_serving_up.mrope_attest -> endpoint.json `mrope_attestation`, the sisteer_mrope_report
  # RPC) and already refused a failing report; this is the job-side belt, read from the AUTHORITATIVE stamp,
  # never from the env: for a non-neox M-RoPE target (glm46vflash — vLLM's fused kernel rotates it wrong
  # unless vllm_patches is in play) an ABSENT block (an older standup) or ok!=true refuses traffic. Recorded
  # for every target in standup_jobs.jsonl.
  MROPE_ATT=$("$SPY" -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1])).get('mrope_attestation')))" "$ep" 2>/dev/null || echo null)
  case "$KEY" in glm46vflash)
    if ! "$SPY" -c "import json,sys; a=json.loads(sys.argv[1]); sys.exit(0 if isinstance(a, dict) and a.get('ok') is True and a.get('reports') else 1)" "$MROPE_ATT"; then
      f22_log "REFUSING TRAFFIC: $KEY is a non-neox M-RoPE model and endpoint.json carries no passing mrope_attestation (got: ${MROPE_ATT:0:600}) — the served rotary is unproven"
      touch "$out/STOP"; return 1
    fi
    f22_log "mrope attestation OK for $KEY: $("$SPY" -c "import json,sys; a=json.loads(sys.argv[1]); r=a['reports'][0]; print({k: r.get(k) for k in ('rotary_cls','is_neox_style','dispatch','kernel_reachable','patch_applied','needs_patch','routed_native_calls')})" "$MROPE_ATT")";;
  esac
  # the batched-token footprint against the model's certified budget (recorded so a sizing near the budget is
  # VISIBLE; note the identity geometry sits at ~1.0 by construction — this is the KV/profiling footprint, not
  # host RAM: the qwen OOMs of 2026-09-07 were the multimodal profiling's dummy images, README section 10)
  local bfrac; bfrac=$("$SPY" -c "import sys; sys.path.insert(0,'src')
from sisteer.capeval.serving.geometry import BUDGET_BY_MODEL
b=BUDGET_BY_MODEL.get(sys.argv[1]); S,L=int(sys.argv[2]),int(sys.argv[3])
print('null' if b is None else round(S*L/b.batched_token_budget,4))" "$KEY" "$TARGET_S" "$TARGET_L" 2>/dev/null || echo null)
  # SERVED_CELLS (native-serving-default): ALL_CELLS minus cells_served.json's cells_skipped (harness ids translated back
  # to the config's TrainTest spellings) — every sbatch loop intersects its cells with it (f22_served_cells_filter);
  # a lens standup skips nothing, so SERVED_CELLS == ALL_CELLS there
  local cells_skipped_json="[]" cs_path
  cs_path=$("$SPY" -c "import json,sys,os;ep=json.load(open(sys.argv[1]));p=ep.get('cells_served') or os.path.join(os.path.dirname(sys.argv[1]),'cells_served.json');print(p)" "$ep")
  if [ -f "$cs_path" ]; then
    cells_skipped_json=$(cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src" "$SPY" -c "import json,sys;print(json.dumps(json.load(open(sys.argv[1])).get('cells_skipped') or []))" "$cs_path" 2>/dev/null || echo "[]")
  fi
  # THE ROSTER the served set is derived from: the campaign's ALL_CELLS (f22_bank_view, a whole-view job) when the caller set one;
  # a SINGLE-CELL job (the MATH-500 / GSM8K launchers name their own cell with --cell and never fill ALL_CELLS) derives it from the
  # standup's OWN record of what it was asked to serve — cells_served.json `cells_requested` — and refuses BY NAME when that record
  # names none (math500-steer, 2026-09-14: array 506863 came up healthy on the lens and died here on an empty roster)
  local -a standup_cells=("${ALL_CELLS[@]+"${ALL_CELLS[@]}"}")
  if [ "${#standup_cells[@]}" -eq 0 ]; then
    if [ -f "$cs_path" ]; then
      mapfile -t standup_cells < <("$SPY" -c "import json,sys;sys.stdout.write(''.join(str(c)+'\n' for c in (json.load(open(sys.argv[1])).get('cells_requested') or [])))" "$cs_path" 2>/dev/null)
    fi
    [ "${#standup_cells[@]}" -ge 1 ] || { f22_log "REFUSING TRAFFIC: no roster (ALL_CELLS is empty — a single-cell job) and the standup record $cs_path names no cells_requested — the served set cannot be derived"; touch "$out/STOP"; return 1; }
    f22_log "served-cells roster: the standup's own cells_requested (${standup_cells[*]}) — a single-cell job, no campaign roster"
  fi
  mapfile -t SERVED_CELLS < <(cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src" "$SPY" - "$cells_skipped_json" "${standup_cells[@]}" <<'PY'
import json, sys
skipped = json.loads(sys.argv[1]); cells = sys.argv[2:]
try:
    from sisteer.capeval.petri import registry_adapter as ra
    skipped_tt = {ra.traintest_cell_id(s["cell_id"]) for s in skipped if isinstance(s, dict) and s.get("cell_id")}
except Exception:  # a checkout without the adapter: the harness ids fall back to their own spelling
    skipped_tt = {s.get("cell_id") for s in skipped if isinstance(s, dict)}
# ONE LINE PER SERVED CELL AND NOTHING ELSE: `print("\n".join([]))` emits a lone newline, which `mapfile -t`
# reads as ONE EMPTY cell — a view whose every cell was skipped then passed the >= 1 guard below with
# SERVED_CELLS=("") and every loop ran over an empty cell id (caught by the wave test's all-skipped case)
sys.stdout.write("".join(c + "\n" for c in cells if c not in skipped_tt))
PY
)
  [ "${#SERVED_CELLS[@]}" -ge 1 ] || { f22_log "REFUSING TRAFFIC: the standup serves none of ${standup_cells[*]} (cells_skipped=$cells_skipped_json)"; touch "$out/STOP"; return 1; }
  [ "$cells_skipped_json" = "[]" ] || f22_log "cells SKIPPED BY NAME on the $TARGET_ENGINE engine: $cells_skipped_json — served: ${SERVED_CELLS[*]}"
  local adj_json; adj_json=$("$SPY" -c "import json,sys;print(json.dumps((json.load(open(sys.argv[1])).get('geometry') or {}).get('adjustments')))" "$ep" 2>/dev/null || echo null)
  # the lens mode as JSON: null on a native standup (no lens reduction is served), the quoted mode on a lens one —
  # hoisted into ONE word so every printf argument below stays a single quoted expansion (a nested-quote
  # inline substitution here miscounted as two arguments in test_sampled_extend's placeholder/argument pin)
  local ep_mode_json; [ "$ep_mode" = None ] && ep_mode_json=null || ep_mode_json="\"$ep_mode\""
  printf '{"job_id":"%s","node":"%s","key":"%s","view":"%s","mode":"%s","eval":"%s","sha":"%s","engine":"%s","serving_path":"%s","cells_skipped":%s,"geometry_adjustments":%s,"lens_reduction_mode":%s,"max_num_seqs":%s,"max_model_len":%s,"attested_max_num_seqs":%s,"attested_max_model_len":%s,"batched_tokens":%s,"budget_fraction":%s,"model_facts":%s,"cluster":"%s","gpu_mem_util":%s,"mrope_attestation":%s,"thinking_mode":"%s","thinking_certification_verdict":%s,"upstream_timeout_s":%s,"serving_path_selection":%s,"written_at":"%s"}\n' \
    "${SLURM_JOB_ID:-manual}" "$(hostname)" "$KEY" "$VIEW" "$mode" "${geval:-$mode}" "$WAVE_SHA" "$TARGET_ENGINE" "$serving_path" "$cells_skipped_json" "${adj_json:-null}" "$ep_mode_json" "$TARGET_S" "$TARGET_L" "$att_s" "$att_l" "$((TARGET_S * TARGET_L))" "$bfrac" "${MODEL_FACTS:-null}" "$CLUSTER" "${GPU_MEM_UTIL:-0.9}" "${MROPE_ATT:-null}" "$ep_think" "${think_verdict_json:-null}" "${TARGET_UPSTREAM_TIMEOUT_S:-null}" "${F22_SERVING_PATH_SELECTION_JSON:-null}" "$(date -u +%FT%TZ)" >> "$out/standup_jobs.jsonl"
  f22_log "target up: $PROXY_URL engine=$TARGET_ENGINE serving_path=$serving_path (selection ${F22_SERVING_PATH_MODE:-unset}; endpoint engine=$ep_engine lens_reduction_mode=$ep_mode; /v1/models kind=${px_kind:-n/a} lens_reduction_mode=$px_mode) geometry=${TARGET_S}x${TARGET_L} (attested ${att_s}x${att_l}; adjustments=[${adj_kinds:-}]; batched_tokens=$((TARGET_S * TARGET_L)) budget_fraction=$bfrac) table=$TABLE thinking=$ep_think (endpoint) / $px_think (/v1/models; certification ${think_verdict:-none}) upstream_timeout_s=${TARGET_UPSTREAM_TIMEOUT_S} server_default_policy=${TARGET_SERVER_DEFAULT_POLICY:-None}$( [ -n "$want_sdp" ] && echo " (asked: $want_sdp)" ) log_requests=${F22_LOG_REQUESTS:-0} served_cells=${SERVED_CELLS[*]}"
  # THE REPRESENTATIVE LOAD CHECK: what makes a PROVISIONAL budget
  # usable without a fresh maximum-budget probe — S concurrent requests at the leg's p99 prompt + cap against the `none`
  # vmodel, every finish_reason in {stop, length}, no 400 / 5xx, the server alive; load_check.json beside endpoint.json.
  # A failure is a MEASURED LIMIT at this window (traffic refused, STOP touched), never a silent shrink of S.
  # F22_LOAD_CHECK_CAP overrides the leg's cap (the fullconst smoke: 8192, the escalation rung);
  # F22_LOAD_CHECK_SYSTEM_SI sends a registered SI as the system prefix (fullconst: si_constitution_full);
  # F22_LOAD_CHECK=0 skips it (a smoke that measures something else), recorded in the log.
  if [ "${F22_LOAD_CHECK:-1}" = 1 ]; then
    if (cd "$REPO_DIR" && "$SPY" scripts/sdfbase/full22/load_check.py --endpoint "$ep" --eval "$lc_eval" \
          --config "$CFG" --model-key "$KEY" \
          ${F22_LOAD_CHECK_CAP:+--cap "$F22_LOAD_CHECK_CAP"} ${F22_LOAD_CHECK_SYSTEM_SI:+--system-si "$F22_LOAD_CHECK_SYSTEM_SI"} \
          ${F22_LOAD_CHECK_ATTACK:+--attack "$F22_LOAD_CHECK_ATTACK"} \
          --out "$out/load_check.json" > "$out/load_check.log" 2>&1); then
      f22_log "load check PASS: $(tail -1 "$out/load_check.log")"
    else
      f22_log "REFUSING TRAFFIC: the representative load check FAILED at ${TARGET_S} x ${TARGET_L} — a measured limit at this window, not a geometry to shrink silently: $(tail -3 "$out/load_check.log" | tr '\n' ' ')"
      touch "$out/STOP"; return 1
    fi
  else
    f22_log "load check SKIPPED (F22_LOAD_CHECK=0) for this standup — recorded, never assumed"
  fi
  # THE SI PREFIX RECORD (the constitution-SI arm, 2026-09-12): every prompted cell's FULL text identity beside the routing table,
  # cross-checked against it; a disagreement refuses traffic (f22_record_si_prefix below)
  f22_record_si_prefix "$out" || { touch "$out/STOP"; return 1; }
}

# --- the PRELAUNCH LENGTH CHECK of a prompted-only view ---------------------------------------------------------
# Before EVERY standup of a view whose cells are `none` + a prompted baseline (roster prompted_only: the llama8b fullconst
# view), render [system = the SI text] + each leg's LONGEST prompt with the pinned tokenizer and assert prompt + cap <= L
# for the static families at 4,096, the escalation rung 8,192, and best-of-N / capability at 512 — refusing the job
# otherwise (window_check.json beside the standup's endpoint.json). vLLM 400s on prompt + max_tokens > L and the static
# adapter's clamp would then shrink max_tokens: for this cell a clamp is a defect, not a recovery.
# --- the view's facts, each read ONE way (f22_roster.view_legs / view_prompted_only through roster_view.py; the config's
# prompted cell) — the sbatch files and the checks below all ask through these, never through a copy of the read
f22_view_has_leg(){   # $1 = leg -> 0 if this view runs it  (needs VIEW_LEGS: the caller's `roster_view.py ... legs` read)
  case ",$VIEW_LEGS," in *",$1,"*) return 0;; *) return 1;; esac
}
f22_view_prompted_only(){   # -> 0 when the view's cells are `none` + a prompted baseline (roster prompted_only: llama8b/fullconst)
  [ "$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" prompted_only --roster "$F22_ROSTER")" = 1 ]
}
f22_prompted_si(){   # -> the SI id of $CFG's first prompted__<si> cell ("" when it has none)  needs CFG MPY
  "$MPY" -c 'import sys, yaml; p = [c for c in yaml.safe_load(open(sys.argv[1]))["explicit_cells"] if c.startswith("prompted__")]; print(p[0][len("prompted__"):] if p else "")' "$CFG"
}
f22_window_check(){   # $1 = the standup out dir (window_check.json lands there), $2 = the window (L), [$3 = the legs this standup
                      # serves (comma list of static,bon,controls,capability; default: every such leg of the view — a ONE-WINDOW view),
                      # $4 = a cap that every named leg is checked at (a ladder RUNG's standup; default: each leg's own cap from $CFG)]
                      # needs KEY VIEW CFG MPY. A PER-LEG-window view (llama8b/shortconst, 2026-09-12) passes the phase's legs: its
                      # controls window does not hold the static families and must not be asked to.
  local out="$1" L="$2" legs="${3:-}" cap="${4:-}" si view_legs
  f22_view_prompted_only || return 0
  si=$(f22_prompted_si)
  [ -n "$si" ] || f22_fatal "prompted-only view $KEY/$VIEW: no prompted__<si> cell in $CFG"
  view_legs=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" legs --roster "$F22_ROSTER" | tr ',' '\n' | grep -x 'static\|bon\|controls\|capability' | paste -sd,)
  if [ -z "$legs" ]; then legs="$view_legs"; else
    local l; for l in ${legs//,/ }; do case ",$view_legs," in *",$l,"*) ;; *) f22_fatal "window check: leg $l is not a leg of $KEY/$VIEW ($view_legs)";; esac; done
  fi
  mkdir -p "$out"
  if (cd "$REPO_DIR" && "$MPY" scripts/sdfbase/full22/prompted_window_check.py --config "$CFG" --si "$si" \
        --max-model-len "$L" --legs "$legs" ${cap:+--cap "$cap"} --out "$out/window_check.json" > "$out/window_check.log" 2>&1); then
    f22_log "window check PASS for $KEY/$VIEW at L=$L (legs $legs${cap:+ at cap $cap}): $(tail -1 "$out/window_check.log")"
  else
    f22_fatal "window check REFUSED for $KEY/$VIEW at L=$L (legs $legs${cap:+ at cap $cap}): $(grep REFUSED "$out/window_check.log" | tr '\n' ' ') — a leg's prompt + cap exceeds the window; widen L (S follows from the budget), never clamp"
  fi
}
f22_gsm8k_window_check(){   # $1 = the controls standup's window (L), $2 = the view's controls csv (roster_view.py `controls`).
                            # needs KEY VIEW CFG MPY REPO_DIR. The 10-shot Inspect gsm8k control's prompt + the controls cap must
                            # fit the window (2026-09-14: capability:short's 2,048 refused 34 of 200 items). The minimum is
                            # DERIVED from the geometry table's GSM8K paired-200 prompt budget (capability:gen_long_gsm8k_rung, its
                            # MAX prompt) + $CFG's attacks.controls_max_new_tokens through geometry.derived_window — never a literal
                            # here. A list without gsm8k needs nothing. A short window is fatal, by name: widen the roster's
                            # controls_window.gsm8k (S follows from the budget), never clamp the cap.
  local L="$1" controls="$2" need
  case ",$controls," in *",gsm8k,"*) ;; *) return 0;; esac
  case "$L" in ''|*[!0-9]*) f22_fatal "gsm8k window check: controls window '$L' is not an integer";; esac
  need=$(cd "$REPO_DIR" && "$MPY" - "$CFG" <<'PY'
import sys, yaml
from sisteer.capeval.serving.geometry import derived_window
cap = int(yaml.safe_load(open(sys.argv[1]))["attacks"]["controls_max_new_tokens"])
print(derived_window("capability:gen_long_gsm8k_rung", cap, from_field="max_prompt"))
PY
) || f22_fatal "gsm8k window check: cannot derive the controls minimum for $KEY/$VIEW (geometry capability:gen_long_gsm8k_rung max_prompt + attacks.controls_max_new_tokens of $CFG)"
  case "$need" in ''|*[!0-9]*) f22_fatal "gsm8k window check: derived minimum '$need' is not an integer";; esac
  [ "$L" -ge "$need" ] || f22_fatal "gsm8k window check REFUSED for $KEY/$VIEW: controls window $L < $need = the GSM8K paired-200 prompt maximum + the controls cap (geometry.derived_window; D72) — widen the roster's controls_window.gsm8k (S follows from the budget), never clamp"
  f22_log "gsm8k window check PASS for $KEY/$VIEW: controls window $L >= $need (the max gsm8k prompt + the controls cap)"
}

# --- the PROMPTED cells' SI TEXT IDENTITY, on the record in FULL (ported from the sampled lane's
# f22s_record_si_prefix: ONE function for every standup of every sbatch, called by f22_standup) ---------------------------
# The proxy verifies every prompted request's system prefix against the routing table's `prompted_prefixes` (H8: byte_length
# + sha256_hex16 — the manifest's prompted_prefix_hashes vocabulary), so a wrong prefix can never be served; but a reader of the
# stores sees only that 16-char prefix. Two lanes generating the SAME arm must be able to prove the two texts identical without
# recomputing anything, so after EVERY standup this writes <standup dir>/si_prefix_record.json: per prompted cell of the CONFIG
# (ALL_CELLS — every cell the routing table registers), the SI bytes resolved through the SAME resolver the standup and the drivers
# use (registry_adapter._traintest_si_text_bytes -> resolve_instruction), their FULL sha256, byte length, first / last
# line, line count, and the routing table's (byte_length, sha16) for that cell's virtual model — REFUSING TRAFFIC (return 1) when
# the two disagree, or when the table names no prefix for a prompted cell. A config with no prompted cell records an empty cell
# list. Nothing here touches src/sisteer/capeval/serving/ (pinned by the manifests' serving_impl_commit); the record is provenance
# beside routing_table.json and the mirror carries every *.json of a serving dir to the grading host.
f22_record_si_prefix(){   # $1 = the standup dir (holds endpoint.json -> routing_table), $2.. = the cells (default SERVED_CELLS, else ALL_CELLS) -> $1/si_prefix_record.json; 1 = refuse
  local out="$1" rec="$1/si_prefix_record.json"; shift
  local cells=("$@")
  if [ "${#cells[@]}" -gt 0 ]; then
    # a --cells argument naming a cell the standup SKIPPED is an error, never a silent drop (native-serving-default)
    local c s
    for c in "${cells[@]}"; do
      if [ "${SERVED_CELLS+x}" = x ]; then
        case " ${SERVED_CELLS[*]} " in *" $c "*) ;; *) f22_fatal "f22_record_si_prefix: cell $c is not served by this standup (served: ${SERVED_CELLS[*]}; engine $TARGET_ENGINE)";; esac
      fi
    done
  elif [ "${SERVED_CELLS+x}" = x ]; then cells=("${SERVED_CELLS[@]}")
  else cells=("${ALL_CELLS[@]}"); fi
  local table; table=$("$SPY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('routing_table') or '')" "$out/endpoint.json" 2>/dev/null)
  [ -n "$table" ] && [ -f "$table" ] || { f22_log "REFUSING TRAFFIC: no routing table recorded in $out/endpoint.json to check the SI prefixes against"; return 1; }
  (cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" HF_HUB_OFFLINE=1 "$SPY" - "$table" "$rec" "${SLURM_JOB_ID:-manual}" "$WAVE_SHA" "${cells[@]}" <<'PYREC'
import hashlib, json, sys, datetime
table_path, rec_path, job, sha, *cells = sys.argv[1:]
from sisteer.capeval.petri import registry_adapter as ra
table = json.load(open(table_path))
by_vm = {p["virtual_model"]: p for p in table.get("prompted_prefixes", [])}
routes = {r["cell_id"]: r for r in table.get("routes", [])}
records, problems = [], []
for cell in cells:
    if not cell.startswith("prompted__"):
        continue
    si_id = cell[len("prompted__"):]
    harness = ra.harness_cell_id(cell)                       # prompted__si_constitution -> prompted__constitution
    data = bytes(ra._traintest_si_text_bytes(si_id))         # the resolver the standup + the drivers share
    full = hashlib.sha256(data).hexdigest()
    lines = data.decode("utf-8").splitlines()
    route = routes.get(harness)
    vm = route.get("virtual_model") if route else None
    spec = by_vm.get(vm) if vm else None
    r = {"cell_id": cell, "si_id": si_id, "harness_cell_id": harness, "virtual_model": vm,
         "byte_length": len(data), "sha256": full, "sha256_hex16": full[:16],
         "first_line": lines[0] if lines else "", "last_line": lines[-1] if lines else "", "n_lines": len(lines),
         "resolver": "sisteer.capeval.petri.registry_adapter._traintest_si_text_bytes (TrainTest resolve_instruction)",
         "routing_table": {"si_prefix_hash": route.get("si_prefix_hash") if route else None,
                           "prompted_prefix": spec},
         "verified_per_request": "H8: the proxy hashes every request's leading system text over byte_length and refuses a mismatch (si_prefix_mismatch / si_prefix_missing)"}
    if route is None:
        problems.append(f"{cell}: the routing table has no route {harness!r}")
    elif spec is None:
        problems.append(f"{cell}: the routing table names no prompted_prefix for virtual model {vm!r}")
    elif spec["sha16"] != full[:16] or int(spec["byte_length"]) != len(data) or route.get("si_prefix_hash") != full[:16]:
        problems.append(f"{cell}: resolver bytes sha256 {full} ({len(data)} B) vs routing table sha16 {spec['sha16']} / {spec['byte_length']} B / route {route.get('si_prefix_hash')}")
    records.append(r)
doc = {"job_id": job, "wave_sha": sha, "written_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "routing_table": table_path, "cells": records, "problems": problems, "ok": not problems}
json.dump(doc, open(rec_path, "w"), indent=1, ensure_ascii=False)
for r in records:
    print(f"si_prefix {r['cell_id']} -> {r['harness_cell_id']} ({r['virtual_model']}): sha256 {r['sha256']} {r['byte_length']} B {r['n_lines']} lines; first {r['first_line'][:60]!r}; last {r['last_line'][-60:]!r}; table sha16 {(r['routing_table']['prompted_prefix'] or {}).get('sha16')}")
if not records:
    print("si_prefix: no prompted cell among " + ",".join(cells) + " — nothing to record")
if problems:
    print("PROBLEMS: " + " | ".join(problems)); sys.exit(1)
PYREC
  ) > "$out/si_prefix_record.log" 2>&1
  local rc=$?
  f22_log "SI prefix record ($out): $(grep -v '^$' "$out/si_prefix_record.log" | tail -4 | tr '\n' ' ')"
  [ "$rc" = 0 ] || { f22_log "REFUSING TRAFFIC: the SI prefix record disagrees with the routing table (see $out/si_prefix_record.log)"; return 1; }
  return 0
}

f22_served_cells_filter(){   # $1 = the NAME of a Bash array of TrainTest cells -> the array narrowed to SERVED_CELLS (after f22_standup)
  # a whole-view job drops the cells the standup SKIPPED BY NAME (a stacked cell or a
  # position-masked LoRA fit on the native engine — PSR / vector cells serve on its residual slot; nothing on the lens); a cells= POSITIONAL that names a skipped cell is FATAL (an operator asked for a cell
  # this engine does not serve — never a silent drop, never a lens fallback: pass F22_SERVING_PATH=lens)
  local -n _f22_arr="$1"; local explicit="${2:-}" c kept=() dropped=()
  [ "${SERVED_CELLS+x}" = x ] || f22_fatal "f22_served_cells_filter before f22_standup (SERVED_CELLS unset)"
  for c in "${_f22_arr[@]}"; do
    case " ${SERVED_CELLS[*]} " in *" $c "*) kept+=("$c");; *) dropped+=("$c");; esac
  done
  if [ "${#dropped[@]}" -gt 0 ]; then
    [ -z "$explicit" ] || f22_fatal "cells=$explicit names ${dropped[*]}, which this standup does not serve on the $TARGET_ENGINE engine (served: ${SERVED_CELLS[*]}); F22_SERVING_PATH=lens serves it on the lens"
    f22_log "cells not served on the $TARGET_ENGINE engine, dropped from this job's loops: ${dropped[*]} (served: ${kept[*]})"
  fi
  [ "${#kept[@]}" -ge 1 ] || f22_fatal "no cell of ${_f22_arr[*]} is served by this standup"
  _f22_arr=("${kept[@]}")
}

f22_anchor_preflight(){   # $1 = leg (petri|misalign|...), $2 = view/tag, $3 = the standup's out dir (vllm_server.log + endpoint.json),
                          # $4 = the anchor record (anchor_sampler_preflight.py --anchor), $5 = the client cap the run's requests carry,
                          # $6 = the request body's sampling fragment ("" = no sampling key, the Petri target's shape;
                          #      ',"temperature":1.0' = the misalign runner's) — needs PROXY_URL, SPY, REPO_DIR
                          # THE ANCHOR SAMPLER PREFLIGHT: ONE request shaped like the
                          # run's at petri__base (a health-class probe on the served `none` route, written to no store), then
                          # scripts/sdfbase/seedrep/anchor_sampler_preflight.py compares the sampler the ENGINE RESOLVED for
                          # exactly that request (matched by the response's id in vllm_server.log — the load check's greedy
                          # requests at the same cap never enter) against the anchor's recorded resolved tuple. AGREES releases
                          # the run; DIVERGED / UNKNOWN touch STOP and return 1: the leg is blocked BEFORE any production
                          # request, never labelled non_comparable after the fact. Needs the standup's --enable-log-requests
                          # (F22_LOG_REQUESTS=1), else the record cannot be read and the verdict is UNKNOWN (a refusal).
  local leg="$1" view="$2" out="$3" anchor="$4" cap="$5" frag="${6:-}" body rid rc bearer
  [ -n "$anchor" ] || { f22_log "REFUSING TRAFFIC: f22_anchor_preflight called with no anchor record"; touch "$out/STOP"; return 1; }
  [ -e "$anchor" ] || { f22_log "REFUSING TRAFFIC: anchor record $anchor does not exist (never assumed equal)"; touch "$out/STOP"; return 1; }
  # the proxy's dummy bearer (petri_target.DEFAULT_API_KEY: loopback-only, not a secret) — read from the code, never restated here
  bearer="${PROXY_API_KEY:-$(cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src" "$SPY" -c 'from sisteer.capeval.serving.petri_target import DEFAULT_API_KEY; print(DEFAULT_API_KEY)')}" \
    || { f22_log "REFUSING TRAFFIC: cannot read petri_target.DEFAULT_API_KEY"; touch "$out/STOP"; return 1; }
  body=$(curl -s -m 900 -H "Authorization: Bearer $bearer" -H 'Content-Type: application/json' "$PROXY_URL/chat/completions" \
    -d "{\"model\":\"petri__base\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ok\"}],\"max_tokens\":${cap}${frag}}")
  rid=$(printf '%s' "$body" | "$SPY" -c 'import json,sys; d=json.load(sys.stdin); print(d["id"])' 2>/dev/null) \
    || { f22_log "REFUSING TRAFFIC: the anchor-preflight probe on petri__base returned no request id: ${body:0:400}"; touch "$out/STOP"; return 1; }
  (cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src" "$SPY" scripts/sdfbase/seedrep/anchor_sampler_preflight.py "$leg" "$view" "$out" \
     --anchor "$anchor" --request-id "$rid" --client-cap "$cap" --out-root "$out/anchor_preflight" \
     ${F22_ANCHOR_SERVER_POLICY:+--anchor-server-policy "$F22_ANCHOR_SERVER_POLICY"} ${F22_ANCHOR_KIND:+--anchor-kind "$F22_ANCHOR_KIND"} \
     > "$out/anchor_preflight.log" 2>&1); rc=$?
  cat "$out/anchor_preflight.log"
  if [ "$rc" != 0 ]; then
    f22_log "REFUSING TRAFFIC: anchor sampler preflight rc=$rc for $leg/$view (see $out/anchor_preflight/ — the leg's rows would be non_comparable; the leg is blocked instead)"
    touch "$out/STOP"; return 1
  fi
  f22_log "anchor sampler preflight AGREES for $leg/$view against $anchor (request $rid; record under $out/anchor_preflight/)"
  return 0
}

f22_standup_stop(){   # stop the current standup and wait for it (between geometry phases)
  [ -n "${STANDUP_STOP:-}" ] && touch "$STANDUP_STOP" 2>/dev/null || true
  if [ -n "${TARGET_PID:-}" ]; then for i in $(seq 1 36); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; fi
  TARGET_PID=""
}

# --- bounded concurrent drivers: keep ~max_num_seqs processes in flight, never more -------------------
# The earlier wave was DRIVER-bound at 13% KV; a wider geometry does nothing unless the drivers fill it.
# f22_queue_launch <tag> <cmd...> starts a driver as soon as fewer than $MAX_DRIVERS are running (the
# server's attested max_num_seqs by default); f22_queue_wait reaps them all and counts failures.
f22_live_drivers(){   # the drivers this queue launched that are still running — never `jobs -rp`
  local n=0 p
  for p in "${DRIVER_PIDS[@]:-}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && n=$((n+1)); done
  echo "$n"
}
f22_queue_launch(){
  local tag="$1"; shift
  # Bound on LIVE DRIVERS, not the shell's job table: the target server is a background job of this same shell
  # (f22_standup: `… &; TARGET_PID=$!`), so `jobs -rp` is >= 1 whenever it serves. A one-cell controls phase sets
  # MAX_DRIVERS=NCELLS=1, and with `jobs -rp` the single driver could never launch — job 464522 (2026-09-08) held its
  # GPU 50 min after "phase CONTROLS" with zero requests and no driver log. Six-cell jobs never saw it (bound 6 > 1).
  while [ "$(f22_live_drivers)" -ge "${MAX_DRIVERS:-64}" ]; do sleep 5; done
  ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" "$@" ) > "$SERVE_DIR/drivers/$tag.log" 2>&1 &
  local pid=$!; DRIVER_PIDS+=("$pid"); DRIVER_TAG[$pid]="$tag"
}
f22_queue_wait(){   # -> NFAIL
  NFAIL=0
  for pid in "${DRIVER_PIDS[@]}"; do
    wait "$pid"; local rc=$?
    [ "$rc" = 0 ] || { NFAIL=$((NFAIL+1)); f22_log "DRIVER FAILED rc=$rc: ${DRIVER_TAG[$pid]}"; tail -5 "$SERVE_DIR/drivers/${DRIVER_TAG[$pid]}.log"; }
  done
  DRIVER_PIDS=()
}

# --- the shared attacker pool ---------------------------------------------------------------------
# The pool's DECLARED membership: every member job that has started and not yet exited, whether or
# not it is serving yet. attacker_pool.sbatch writes `<jobid>.pending` at start and its cleanup trap
# removes both markers at exit, so the union of *.json and *.pending job ids IS the pool
# size -- nothing has to be passed in on the target side and kept in sync with the pool submission.
# F22_ATTACKER_JOBS (comma-separated member job ids) PINS this consumer to named members instead of
# taking the whole declared pool. Default unset = every member, the historical behaviour, byte-identical.
#
# Why it exists (2026-09-22): a driver binds to its URL at task start and never rediscovers, so a
# target released into a partially-started pool silently binds to whatever subset is up — nine targets
# once bound to a single member that way. Round-robin over "whatever is live" makes submission order
# load-bearing. Pinning lets targets go out PIECEMEAL as members come online, each to a member chosen
# rather than inherited, which is also how you get a known 2-targets-per-attacker fan-out.
#
# It REFUSES rather than degrades: a pinned id with no record ends the job, never a silent fallback
# to the rest of the pool. Silently binding to a member the caller did not name is the failure this
# is meant to prevent, so it must never be the error path. The refusal travels as rc 1 + stderr and
# the CALLER raises the fatal -- see the comment on the miss branch for why it cannot be raised here.
f22_declared_attackers(){   # -> one member id per line on stdout; rc 1 if a PINNED member has no record
  local rec b id
  local -a want=(); local pinned="${F22_ATTACKER_JOBS:-}"
  [ -n "$pinned" ] && IFS=',' read -ra want <<< "$pinned"
  local -a found=()
  for rec in "$WAVE_ROOT"/attackers/*.json "$WAVE_ROOT"/attackers/*.pending; do
    [ -f "$rec" ] || continue
    b=${rec##*/}; id=${b%.*}
    # A member id is the record's STEM. Slurm members are job ids, but a non-Slurm pool is a member
    # too: a Modal-hosted pool publishes `<pool-name>.json` (e.g. modal_relu-paired-atk-1.json), and
    # the old `*[!0-9]*` filter dropped it SILENTLY -- an empty declared pool and then a fatal, for a
    # pool that serves perfectly well on the health-check path. f22_live_attackers already keeps any
    # member squeue cannot adjudicate and lets the URL health check decide; a non-Slurm member is that
    # same case, permanently. So only an empty stem is not a name. (`.STOP` tombstones never reach
    # here -- they match neither glob.) -- 2026-09-22
    [ -n "$id" ] || continue
    if [ -n "$pinned" ]; then
      local keep=0 w
      for w in "${want[@]}"; do [ "$id" = "${w// /}" ] && keep=1; done
      [ "$keep" = 1 ] || continue
    fi
    found+=("$id"); printf '%s\n' "$id"
  done | sort -u
  if [ -n "$pinned" ]; then
    local w miss=()
    for w in "${want[@]}"; do
      w="${w// /}"; [ -z "$w" ] && continue
      [ -f "$WAVE_ROOT/attackers/$w.json" ] || [ -f "$WAVE_ROOT/attackers/$w.pending" ] || miss+=("$w")
    done
    # REFUSE BY RETURN STATUS, ON STDERR -- never f22_fatal here (2026-09-22). Every caller reads this
    # function's STDOUT, which breaks the old refusal two ways, both measured: (a) `exit` inside a
    # `< <(...)` or `$(...)` only kills the substitution subshell, so the job sailed on with rc=0; and
    # (b) f22_log writes to stdout, so the refusal TEXT was captured as a member id. Pinning `101,102`
    # with only 101 on disk gave declared=(101 "[...] FATAL: ... 102 ...") and the job bound to 101
    # alone -- exactly the silent bind to a member the caller never named that this mechanism exists
    # to prevent. The caller turns rc 1 into the fatal, where `exit` actually ends the job.
    if [ "${#miss[@]}" -ne 0 ]; then
      f22_log "FATAL: F22_ATTACKER_JOBS names member(s) with no record under $WAVE_ROOT/attackers: ${miss[*]} — refusing rather than binding to the rest of the pool" >&2
      return 1
    fi
  fi
}

# f22_live_attackers WAITS for the declared pool before it binds (the one-time-snapshot fix, 2026-09-18).
#
# THE DEFECT THIS REPLACES. The old body globbed the serving records ONCE and bound the driver
# round-robin to whatever happened to be up at that instant. A target that reached this line while
# the pool was still loading weights saw one member and pinned EVERY driver of the wave to it, with
# no rebalancing afterwards -- at the measured 70-100 drivers/member knee, a 36-driver job on one
# member queues for hours while two idle GPUs sit beside it, and nothing in the run record says so.
# The snapshot itself is fine; taking it too early is the bug. So: wait for every member the pool
# DECLARES to actually serve, then snapshot.
#
# A member whose Slurm job has gone away (a hard node failure kills the cleanup trap, stranding its
# .pending) is dropped from the expectation rather than waited on -- otherwise one dead marker holds
# the wave until the deadline every time.
#
#   F22_ATTACKER_WAIT_S      how long to wait for the declared pool to come up (default 1800)
#   F22_ATTACKER_REQUIRE_FULL=1  refuse instead of proceeding under strength at the deadline
f22_live_attackers(){   # -> ATTACKERS (array of live pool URLs, health-checked)
  local served; served=$("$MPY" - "$CFG" <<'PY'
import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))["attacker"]["model"])
PY
)
  local wait_s="${F22_ATTACKER_WAIT_S:-1800}"
  local t0=$SECONDS waited=0 ndeclared=0 nalive=0 logged_wait=0
  local id rec url st
  while :; do
    ATTACKERS=(); ATTACKER_MANIFESTS=(); local -a alive=() missing=()
    # Command substitution, NOT `< <(...)`: a process substitution discards the exit status, and the
    # pin refusal has to reach this job (2026-09-22).
    local -a declared=(); local _decl
    _decl=$(f22_declared_attackers) || f22_fatal "the F22_ATTACKER_JOBS pin cannot be satisfied (reason above) — refusing rather than binding to members this job did not name"
    [ -n "$_decl" ] && mapfile -t declared <<< "$_decl"
    ndeclared=${#declared[@]}
    for id in "${declared[@]}"; do
      # A member squeue can NEVER adjudicate: a non-Slurm pool (Modal) publishes `<pool-name>.json`,
      # not a job id, and `squeue -j <non-numeric>` answers "Invalid job id specified" -- the single
      # reply that DROPS a member. Asking would therefore delete precisely the members Slurm does not
      # manage. Same rule as the undecidable branch below: keep it, health check decides. 2026-09-22
      case "$id" in
        *[!0-9]*) f22_log "attacker member $id is not a Slurm job id (non-Slurm pool) — kept; its liveness is decided by the health check alone"; alive+=("$id"); continue;;
      esac
      # A declared member still queued/running is worth waiting for; one Slurm has finished is not.
      # squeue exits 1 for BOTH "that job is over" and "I could not reach the controller" (measured
      # 2026-09-18), so the exit code ALONE cannot decide. Only the definitive "Invalid job id
      # specified" drops a member; any other failure means we could not decide, and an undecidable
      # member is KEPT -- a transient slurmctld hiccup must never shrink a healthy pool to nothing
      # and refuse the job.
      if sq=$(squeue -h -j "$id" -o %T 2>&1); then
        st=$(printf '%s\n' "$sq" | head -1)
        case "$st" in
          RUNNING|PENDING|CONFIGURING|COMPLETING) alive+=("$id");;
          "") f22_log "attacker member $id: squeue returned no state — keeping it in the expected pool"; alive+=("$id");;
          *)  f22_log "attacker member $id is in state '$st' — dropped from the expected pool";;
        esac
      elif printf '%s' "$sq" | grep -q "Invalid job id"; then
        f22_log "attacker member $id has left the queue (its marker is stale) — dropped from the expected pool"
      else
        f22_log "attacker member $id: squeue could not answer ($(printf '%s' "$sq" | head -1 | cut -c1-120)) — keeping it, the pool is not shrunk on an undecidable probe"
        alive+=("$id")
      fi
    done
    nalive=${#alive[@]}
    for id in "${alive[@]}"; do
      rec="$WAVE_ROOT/attackers/$id.json"
      if [ -f "$rec" ]; then
        url=$("$MPY" -c "import json,sys;print(json.load(open(sys.argv[1]))['url'])" "$rec" 2>/dev/null) || url=""
        if [ -n "$url" ] && curl -s -m 15 "$url/v1/models" 2>/dev/null | grep -q -F "$served"; then
          # The member's ENGINE-VERIFIED serving manifest, carried in its own record. Without it the
          # driver writes attacker_quantization 'unrecorded' (attacker_serving.driver_fields(None)),
          # and the engine-identity join then REFUSES to resume that store under any identity -- so a
          # timeout becomes a full restart of the rung. Kept index-parallel with ATTACKERS; an empty
          # element is a member whose record predates the field, and that driver simply omits the flag.
          man=$("$MPY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('serving_manifest') or '')" "$rec" 2>/dev/null) || man=""
          [ -n "$man" ] && [ ! -f "$man" ] && { f22_log "attacker $id: serving_manifest $man does not exist — that member's drivers will record an UNRESUMABLE identity"; man=""; }
          ATTACKERS+=("$url"); ATTACKER_MANIFESTS+=("$man"); continue
        fi
      fi
      missing+=("$id")
    done
    [ "${#missing[@]}" -eq 0 ] && [ "${#ATTACKERS[@]}" -ge 1 ] && break
    waited=$((SECONDS - t0))
    if [ "$waited" -ge "$wait_s" ]; then
      f22_log "attacker pool still under strength after ${waited}s: ${#ATTACKERS[@]}/${nalive} serving (not yet up: ${missing[*]:-none})"
      break
    fi
    [ "$logged_wait" = 0 ] && { f22_log "waiting up to ${wait_s}s for the declared attacker pool: ${#ATTACKERS[@]}/${nalive} serving, not yet up: ${missing[*]}"; logged_wait=1; }
    sleep 20
  done
  [ "${#ATTACKERS[@]}" -ge 1 ] || f22_fatal "no live attacker in $WAVE_ROOT/attackers (submit attacker_pool.sbatch first, under the SAME serving path as this target — this job's is F22_SERVING_PATH=$F22_SERVING_PATH, selection ${F22_SERVING_PATH_MODE:-unset}; the pool's records live under the wave root that path selects, so a pool submitted under the other path is invisible here)"
  # the binding this job actually took, in the run record — an under-strength pool is a throughput
  # fact the analysis needs, and it is invisible from the driver logs alone
  if [ -n "${SERVE_DIR:-}" ] && [ -d "$SERVE_DIR" ]; then
    { printf '{"declared":%d,"alive":%d,"serving":%d,"waited_s":%d,"urls":[' \
        "$ndeclared" "$nalive" "${#ATTACKERS[@]}" "$((SECONDS - t0))"
      local first=1 u; for u in "${ATTACKERS[@]}"; do [ "$first" = 1 ] || printf ','; printf '"%s"' "$u"; first=0; done
      printf ']}\n'; } > "$SERVE_DIR/attacker_binding.json"
  fi
  if [ "${#ATTACKERS[@]}" -lt "$nalive" ]; then
    f22_log "WARN attacker pool UNDER STRENGTH: ${#ATTACKERS[@]} of $nalive declared members serving — drivers will pile onto the survivors past the measured 70-100/member knee"
    [ "${F22_ATTACKER_REQUIRE_FULL:-0}" = 1 ] && f22_fatal "F22_ATTACKER_REQUIRE_FULL=1 and only ${#ATTACKERS[@]}/$nalive attacker members are serving"
  fi
  local nman=0 m; for m in ${ATTACKER_MANIFESTS[@]+"${ATTACKER_MANIFESTS[@]}"}; do [ -n "$m" ] && nman=$((nman+1)); done
  [ "$nman" = "${#ATTACKERS[@]}" ] || f22_log "WARN only $nman of ${#ATTACKERS[@]} attacker members carry a serving manifest — the rest produce stores that CANNOT be resumed"
  f22_log "live attackers (${#ATTACKERS[@]}/${nalive} declared, $nman with a serving manifest, waited $((SECONDS - t0))s): ${ATTACKERS[*]}"
}

# --- LOW-DOSE NUMERIC LIVENESS (moved here from common_sampled.sh, 2026-09-18) -------------------
# It lived in common_sampled.sh only, so target_attacks.sbatch ran the bare text-divergence gate:
# at a small dose a CORRECT object changes no greedy token on the five BENIGN probe prompts
# (served_target.PROBE_PROMPTS) and the job refuses a steer that is live and merely SMALL.
# Precedent: cluster A 497871, constv2-mix-full__lora_r4_ce_minus_ctl at nominal 0.1, 0/5 differing
# with salt isolation AND the intervention hashes both PASSING. Same cell family as this leg's
# difference-LoRA dose ladder, which is why the attacks leg needed it. Sibling phase, same defect.
#: where the liveness writer lives, relative to $REPO_DIR (common_sampled.sh re-asserts the same value)
: "${F22S_DIR:=scripts/sdfbase/full22/sampled}"
declare -A F22S_LIVENESS_RECORD=()   # cell -> the admitting record of the CURRENT standup (reset on every run of the writer)
f22s_low_dose_liveness(){   # needs CELLS MPY REPO_DIR PROXY_URL TABLE SERVE_DIR TAG VIEW -> F22S_LIVENESS_RECORD; ALWAYS returns 0
  local writer="$F22S_DIR/lora_liveness.py" dir cell out log rc line
  F22S_LIVENESS_RECORD=()
  dir="$(dirname "$TABLE")"
  [ -f "$REPO_DIR/$writer" ] || { f22_log "liveness: no $writer at this sha — no numeric record for any cell; the text-divergence gate is unchanged"; return 0; }
  mkdir -p "$SERVE_DIR/drivers"
  for cell in "${CELLS[@]}"; do
    case "$cell" in none|prompted__*) continue;; esac
    out="$dir/liveness_low_dose_${cell}.json"; log="$SERVE_DIR/drivers/liveness_${cell}.log"; rc=0
    ( cd "$REPO_DIR" && "$MPY" "$writer" --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$cell" \
        --tag "$TAG" --view "$VIEW" --out "$out" ${F22_THINKING_ON:+--thinking-on} ) > "$log" 2>&1 || rc=$?
    if [ -f "$out" ] && line=$(cd "$REPO_DIR" && "$MPY" "$writer" --summary "$out" 2>/dev/null); then
      f22_log "liveness $cell: $line"
    else
      f22_log "liveness $cell: writer failed rc=$rc, no record at $out — the text-divergence gate is unchanged ($(tail -2 "$log" 2>/dev/null | tr '\n' ' ' | head -c 300))"
      continue
    fi
    [ "$rc" = 0 ] && F22S_LIVENESS_RECORD[$cell]="$out"
  done
  return 0
}
f22s_launch_cell(){   # $1 = tag, $2 = the cell the driver serves, $3.. = the driver command exactly as f22_queue_launch takes it
  local tag="$1" cell="$2"; shift 2
  local rec=""; [ -n "$cell" ] && rec="${F22S_LIVENESS_RECORD[$cell]:-}"   # an empty subscript is a bash error, not a lookup
  if [ -n "$rec" ]; then f22_queue_launch "$tag" env "SISTEER_LOW_DOSE_LIVENESS_RECORD=$rec" "$@"
  else f22_queue_launch "$tag" "$@"; fi
}

# --- the STEER-IS-LIVE positive control, in the wave job itself ----------------------------------------------
# The smoke's check (smoke_positive_control.py: proxy identity mode, salt isolation, determinism, the scale-0 noise
# floor, steering-bites vs the scale-0 twin, intervention hashes, greedy divergence), run against THIS job's standup
# on one steered cell before any driver — so a served object that silently no-ops refuses the job here instead of
# producing a table of bare-model numbers under a steered label. Needs MPY REPO_DIR PROXY_URL TABLE SERVE_DIR.
f22_positive_control(){   # $1 = steered cell id, $2 = phase tag  -> 0 on PASS; 1 (after logging) on FAIL
  # one `local` per dependent name: bash expands EVERY word of a `local` line before binding any of them, so a
  # single `local a="$1" b="...$a..."` reads an unbound `a` under set -u (every first-pass static job died here,
  # 2026-09-08, one line after a passing standup)
  local cell="$1"
  local tag="$2"
  local out="$SERVE_DIR/positive_control_$tag.json"
  local log="$SERVE_DIR/drivers/positive_control_$tag.log"
  mkdir -p "$SERVE_DIR/drivers"
  local rc=0
  # under a thinking-ON standup (F22_THINKING_ON) the smoke declares petri_thinking_on on its LENS-route requests
  # and requires the proxy's thinking_mode == on (smoke_positive_control.py --thinking-on); OFF is byte-identical
  # the ENGINE the check asserts from the proxy's identity block (--require-engine "$TARGET_ENGINE": lens:stacked or
  # lora_native — the path-matched twin and every steered route's bites proof follow it)
  ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" "$MPY" scripts/sdfbase/full22/smoke_positive_control.py \
        --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$cell" --out "$out" --require-engine "${TARGET_ENGINE:-lens:stacked}" ${F22_THINKING_ON:+--thinking-on} ) > "$log" 2>&1 || rc=$?
  # ONE LOG LINE PER CHECK, never one concatenated string (2026-09-08: the second-cut refusal line was cut at 731
  # chars by a `head -c 600` here, exactly at the same byte in seven jobs, hiding the failing check behind the passing
  # ones — a guard that refuses correctly but truncates its reason). The JSON at $out keeps every probe and text.
  if [ -f "$out" ]; then
    "$MPY" - "$out" <<'PYCHK' 2>/dev/null | while IFS= read -r line; do f22_log "positive control check ($tag): $line"; done
import json, sys
d = json.load(open(sys.argv[1]))
for c in d.get("checks", []):
    detail = {k: v for k, v in c.items() if k not in ("check", "status", "texts")}
    print(f"{c.get('check')}={c.get('status')} {json.dumps(detail, default=str)[:400]}")
PYCHK
  else
    f22_log "positive control ($tag, cell $cell): no result JSON at $out — $(tail -3 "$log" | tr '\n' ' ' | head -c 400)"
  fi
  if [ "$rc" = 0 ]; then
    f22_log "positive control PASS ($tag, cell $cell): every check PASS ($out)"
    echo PASS > "$out.verdict"; return 0
  fi
  # ANCHOR-ONLY GREEDY NONDETERMINISM IS A RECORDED WARNING, NOT A REFUSAL (measured 2026-09-08: all seven
  # second-cut jobs on glm46vflash failed exactly ONE check, `determinism`, and only on the UNHOOKED `none` anchor — the
  # same 2 of 8 greedy probes differed between repeats every time — while the steered leg and the scale-0 twin were
  # byte-stable and every steer-is-live check passed: noise floor 8/8, steering bites 5/5, intervention hashes
  # distinct, greedy divergence 8/8). That is numeric noise under batching on the bare route (vLLM is not
  # batch-invariant; a near-tie flips with batch composition), the very noise the wave's SAME-WAVE bare anchor exists
  # to absorb — not a served object silently no-op'd, which is what this control is for. So: when the ONLY failing
  # check is `determinism` and the steered + twin legs repeated byte-identically, traffic proceeds with the warning
  # recorded (the JSON keeps every probe and text); ANY other failing check, or nondeterminism on a LENS leg (the served
  # arithmetic itself unstable), still refuses. The refusal line names the failing checks (never a truncated tail).
  # The verdict is one testable helper (positive_control_verdict.py): the anchor-only rule above unchanged; plus, OPT-IN
  # (F22_PC_TOLERATE_LENS_NONDETERMINISM=1), lens-leg nondeterminism tolerated ONLY when
  # every steer-is-live check passed and the flipping probes are a minority on every leg: under the merged serving tree
  # every route repeats non-identically on a few probes (near-tie flips under batch / resumed-prefill composition, on all
  # legs alike) while steering bites 5/5 and the noise floor holds 8/8 — instrument noise, recorded, not a no-op'd object.
  local verdict vh="$REPO_DIR/scripts/sdfbase/full22/positive_control_verdict.py"
  if [ -f "$vh" ]; then
    verdict=$(cd "$REPO_DIR" && "$MPY" "$vh" "$out" 2>/dev/null || true)
    [ -n "$verdict" ] || verdict="FAIL verdict_unavailable"
  else
    # a checkout without the helper (or the set -u harness's stub repo): the anchor-only rule inline, byte-for-byte the old behaviour
    verdict=$("$MPY" - "$out" <<'PYCHK' 2>/dev/null || echo FAIL
import json, sys
d = json.load(open(sys.argv[1]))
failing = [c for c in d.get("checks", []) if c.get("status") != "PASS"]
by = {c.get("check"): c for c in failing}
det = by.get("determinism")
nd = (det or {}).get("nondeterministic_prompts") or {}
anchor_only = bool(det) and not nd.get("steered") and not nd.get("twin") and bool(nd.get("anchor"))
if anchor_only and set(by) == {"determinism"}:
    print(f"PASS_WITH_ANCHOR_NONDETERMINISM anchor_nondeterministic={len(nd['anchor'])}/{det.get('n_prompts')}")
    sys.exit(0)
# D179 (2026-09-14, llama it_d0p1 at 9ffc03f0, cluster B 100540/100541/100542): the SAME anchor noise reaches the
# noise_floor check, which compares the unhooked anchor with the scale-0 twin byte-for-byte. On a non-batch-invariant
# anchor the two draws differ on the anchor's near-tie prompts (haiku / paperclip on llama) while the twin repeats
# identically and every steer-is-live check passes — anchor noise, not a scale-0 leak. Admitted as the same recorded
# warning when: the only failing checks are determinism (anchor-only) + noise_floor, the twin is byte-stable, and the
# differing prompts are a minority of the probes (a majority would say the routes differ, not that the anchor is noisy).
nf = by.get("noise_floor")
if anchor_only and nf and set(by) == {"determinism", "noise_floor"}:
    n = int(nf.get("n_prompts") or det.get("n_prompts") or 0); diff = nf.get("differing_prompts") or []
    if n and 0 < len(diff) <= n // 2:
        print(f"PASS_WITH_ANCHOR_NONDETERMINISM anchor_nondeterministic={len(nd['anchor'])}/{det.get('n_prompts')} "
              f"noise_floor_anchor_vs_twin_differing={len(diff)}/{n} (twin byte-stable; D179)")
        sys.exit(0)
print("FAIL " + ",".join(c.get("check", "?") for c in failing))
PYCHK
)
  fi
  case "$verdict" in PASS_WITH_LENS_NONDETERMINISM*)
    f22_log "positive control PASS WITH WARNING ($tag, cell $cell): greedy repeats differed on a minority of probes on lens legs too ($verdict) — the same batch / resumed-prefill near-tie flips on every route; every steer-is-live check PASS; tolerated by the lane's opt-in F22_PC_TOLERATE_LENS_NONDETERMINISM=1 (D228f) — recorded, traffic proceeds ($out)"
    echo "$verdict" > "$out.verdict"; return 0;;
  PASS_WITH_ANCHOR_NONDETERMINISM*)
    f22_log "positive control PASS WITH WARNING ($tag, cell $cell): the unhooked anchor repeated non-identically under greedy ($verdict); steered + scale-0 legs byte-stable, every steer-is-live check PASS — recorded, traffic proceeds ($out)"
    echo "$verdict" > "$out.verdict"; return 0;;
  esac
  f22_log "REFUSING TRAFFIC: steer-is-live positive control FAILED ($tag, cell $cell) [$verdict]: $(tail -5 "$log" | tr '\n' ' ' | head -c 600)"
  echo "$verdict" > "$out.verdict"; return 1
}
