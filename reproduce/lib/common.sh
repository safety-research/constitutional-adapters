#!/bin/bash
# reproduce/lib/common.sh — the shell library every front-door stage script sources (bash).
#
# One site config (`$SISTEER_SITE_DIR/config.env`, shipped as reproduce/site.example/) plus one cluster block per
# label (`clusters/<label>.env`) is the whole contract between the pipeline and a machine: nothing under reproduce/
# names a host, a user or a storage root. Every stage script:
#   * parses the COMMON flags through repro_parse_common_args  (--model K --config F --overrides F --dry-run --help)
#   * loads the config through repro_load_config (refuses loudly on a missing key; never invents a default for a host)
#   * runs every side effect through repro_run / repro_ssh, so --dry-run prints the exact command with the config
#     substituted and touches nothing
#   * prints `registry: <record_type> <id>` for every id it produced and `submitted <cluster> <jobid> <unit>` for
#     every scheduler submission (the submitting session arms its watcher from those lines)
# Sourcing this file has no side effects (safe from a test harness); the functions are the API.
# NEVER `sbatch --export=` (cluster hard rule 1): job-side values travel as VAR=value prefixes inside the ssh string.

REPRO_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="$(cd "$REPRO_LIB_DIR/.." && pwd)"
REPRO_REPO="$(cd "$REPRO_DIR/.." && pwd)"

repro_say(){ echo "[$(date -u +%FT%TZ)] $*"; }
repro_warn(){ echo "[$(date -u +%FT%TZ)] WARNING: $*" >&2; }
repro_fatal(){ echo "REFUSED: $*" >&2; exit "${REPRO_REFUSE_RC:-2}"; }      # every refusal names its reason and exits non-zero
repro_registry_id(){ echo "registry: $1 $2"; }                                # <record_type> <id>
repro_note(){ echo "note: $*"; }

# --- the interpreter --------------------------------------------------------------------------------------------------
# SISTEER_PYTHON (config) > the project environment `uv sync` installs into > python3. The project environment is
# $UV_PROJECT_ENVIRONMENT when it is set (uv resolves a relative value against the project root) and <repo>/.venv
# otherwise. The lib's own Python helpers are stdlib + PyYAML only.
repro_project_env(){
  local e="${UV_PROJECT_ENVIRONMENT:-}"
  if [ -z "$e" ]; then echo "$REPRO_REPO/.venv"; return; fi
  case "$e" in /*) echo "$e" ;; *) echo "$REPRO_REPO/$e" ;; esac
}
repro_python(){
  if [ -n "${SISTEER_PYTHON:-}" ]; then echo "$SISTEER_PYTHON"; return; fi
  if [ -x "$(repro_project_env)/bin/python" ]; then echo "$(repro_project_env)/bin/python"; return; fi
  command -v python3
}

# --- common flags -----------------------------------------------------------------------------------------------------
# REPRO_DRY_RUN is FAIL-CLOSED: a caller that exported REPRO_DRY_RUN=1 before sourcing this file stays dry (the flag is only
# ever raised here, by --dry-run, never reset to 0) — a driver that set it in its environment and then sourced the libs lost
# it at this line and revoked the qwen35 geometry overlays for real; an unset value is 0 as before
REPRO_MODEL=""; REPRO_CONFIG=""; REPRO_OVERRIDES=""; REPRO_DRY_RUN="${REPRO_DRY_RUN:-0}"; REPRO_HELP=0; REPRO_ARGS=()
repro_parse_common_args(){   # "$@" -> REPRO_MODEL REPRO_CONFIG REPRO_OVERRIDES REPRO_DRY_RUN REPRO_HELP; the rest in REPRO_ARGS
  REPRO_ARGS=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --model) REPRO_MODEL="${2:?--model needs a key}"; shift 2;;
      --model=*) REPRO_MODEL="${1#--model=}"; shift;;
      --config) REPRO_CONFIG="${2:?--config needs a file}"; shift 2;;
      --config=*) REPRO_CONFIG="${1#--config=}"; shift;;
      --overrides) REPRO_OVERRIDES="${2:?--overrides needs a file}"; shift 2;;
      --overrides=*) REPRO_OVERRIDES="${1#--overrides=}"; shift;;
      --dry-run|-n) REPRO_DRY_RUN=1; shift;;
      -h|--help) REPRO_HELP=1; shift;;
      *) REPRO_ARGS+=("$1"); shift;;
    esac
  done
}
repro_common_help(){
  cat <<'H'
Common flags (every stage):
  --model <key>        a key of reproduce/models.yaml (qwen35_4b_demo | gemma4e4b | qwen35_4b | nemotron3nano4b | llama8b)
  --config <file>      the site config (default: $SISTEER_SITE_DIR/config.env, $SISTEER_SITE_DIR default ~/.config/sisteer)
  --overrides <file>   a run-overrides .env (default: $SISTEER_RUN_OVERRIDES, else reproduce/overrides/paper_defaults.env);
                       recorded on every registry row the stage produces
  --dry-run            print every command with the config substituted; touch nothing; exit 0
  --help               this text
Every stage is idempotent (registry state first, then completion markers under $SHARED/exp/reproduce/<model>/<stage>/),
prints `registry: <record_type> <id>` for every id it produced and `submitted <cluster> <jobid> <unit>` for every
scheduler submission, and exits non-zero with the reason on any refusal.
H
}

# --- the site config --------------------------------------------------------------------------------------------------
repro_config_path(){   # -> the config file this run reads (explicit --config > $SISTEER_CONFIG > $SISTEER_SITE_DIR/config.env)
  if [ -n "${REPRO_CONFIG:-}" ]; then echo "$REPRO_CONFIG"; return; fi
  if [ -n "${SISTEER_CONFIG:-}" ]; then echo "$SISTEER_CONFIG"; return; fi
  echo "${SISTEER_SITE_DIR:-$HOME/.config/sisteer}/config.env"
}
repro_load_config(){
  local cfg; cfg="$(repro_config_path)"
  [ -r "$cfg" ] || repro_fatal "no site config at $cfg — copy reproduce/site.example/ to \$SISTEER_SITE_DIR (default ~/.config/sisteer), edit it, or pass --config <file>"
  SISTEER_CONFIG="$(cd "$(dirname "$cfg")" && pwd)/$(basename "$cfg")"; export SISTEER_CONFIG
  SISTEER_SITE_DIR="$(dirname "$SISTEER_CONFIG")"; export SISTEER_SITE_DIR
  set -a; source "$SISTEER_CONFIG"; set +a
  repro_check_site_values "$SISTEER_CONFIG"
  # required keys — a missing one refuses BY NAME (never a guessed host or root)
  local k; for k in SISTEER_CLUSTERS SISTEER_PRIMARY_CLUSTER SISTEER_SECRETS_MODE SISTEER_REGISTRY_ROOT SISTEER_GRADING_HOST; do
    [ -n "${!k:-}" ] || repro_fatal "config.env ($SISTEER_CONFIG) must set $k"
  done
  case "$SISTEER_SECRETS_MODE" in op|env|wif) ;; *) repro_fatal "SISTEER_SECRETS_MODE must be op, env or wif, got '$SISTEER_SECRETS_MODE'";; esac
  case " $SISTEER_CLUSTERS " in *" $SISTEER_PRIMARY_CLUSTER "*) ;; *) repro_fatal "SISTEER_PRIMARY_CLUSTER=$SISTEER_PRIMARY_CLUSTER is not one of SISTEER_CLUSTERS ($SISTEER_CLUSTERS)";; esac
  # defaults (§3.1) — every key has a default or a loud refusal
  export SISTEER_DATA_ROOT="${SISTEER_DATA_ROOT:-\$SHARED/exp/corpora}"
  export SISTEER_LOG_DIR="${SISTEER_LOG_DIR:-\$SHARED/exp/logs}"
  export SISTEER_FIG_DIR="${SISTEER_FIG_DIR:-\$SHARED/exp/figs}"
  export SISTEER_WAVE_ROOT="${SISTEER_WAVE_ROOT:-\$SHARED/exp/sdfbase_full22_native}"
  export SISTEER_REPRODUCE_ROOT="${SISTEER_REPRODUCE_ROOT:-\$SHARED/exp/reproduce}"
  export SISTEER_TIME_FIT="${SISTEER_TIME_FIT:-2-00:00:00}" SISTEER_TIME_HPO="${SISTEER_TIME_HPO:-12:00:00}"
  export SISTEER_TIME_EVAL="${SISTEER_TIME_EVAL:-1-00:00:00}" SISTEER_TIME_PROBE="${SISTEER_TIME_PROBE:-04:00:00}"
  # the Petri proxy's own walltime (06: every petri unit, first and overflow pass): the paper ran every proxy at a day; the proxy frees
  # its GPU when the drive ends (STOP / a stale heartbeat), so the day is a ceiling, not a cost. SISTEER_TIME_EVAL no longer reaches it
  # (the demo's 04:00:00 eval walltime cut a full drive at 521 of 660 scenarios, 2026-10-13)
  export SISTEER_TIME_PETRI="${SISTEER_TIME_PETRI:-1-00:00:00}"
  export SISTEER_OP_ITEM_ANTHROPIC_LOWPRIO="${SISTEER_OP_ITEM_ANTHROPIC_LOWPRIO:-lowprio}"
  export SISTEER_OP_ITEM_ANTHROPIC_HIGHPRIO="${SISTEER_OP_ITEM_ANTHROPIC_HIGHPRIO:-highprio}"
  export SISTEER_OP_ITEM_ANTHROPIC_BATCH="${SISTEER_OP_ITEM_ANTHROPIC_BATCH:-batch}"
  export SISTEER_OP_ITEM_OPENROUTER="${SISTEER_OP_ITEM_OPENROUTER:-OpenRouter API Key}"
  export SISTEER_OP_ITEM_HF="${SISTEER_OP_ITEM_HF:-HF}" SISTEER_OP_ITEM_HF_CLUSTER_READ="${SISTEER_OP_ITEM_HF_CLUSTER_READ:-HF cluster read}"
  export SISTEER_OP_ITEM_WANDB="${SISTEER_OP_ITEM_WANDB:-Weights and Biases}"
  export SISTEER_MODAL_VOLUME="${SISTEER_MODAL_VOLUME:-twosi-fits}" SISTEER_MODAL_ENVIRONMENT="${SISTEER_MODAL_ENVIRONMENT:-main}"
  export SISTEER_MODAL_PYTHON="${SISTEER_MODAL_PYTHON:-modal}"
  export SISTEER_REGISTRY_SPOOL="${SISTEER_REGISTRY_SPOOL:-}"
  export SISTEER_WANDB_ENTITY="${SISTEER_WANDB_ENTITY:-}" SISTEER_WANDB_PROJECT="${SISTEER_WANDB_PROJECT:-}"
  export SISTEER_ATTACKER_MODEL="${SISTEER_ATTACKER_MODEL:-huihui-ai/Huihui-Qwen3.6-27B-abliterated}"
  export SISTEER_ATTACKER_REVISION="${SISTEER_ATTACKER_REVISION:-27502c8717fd5a2f8c0c77188c10c243fd4f672e}"
  export SISTEER_MT_ATTACKER_GPUS="${SISTEER_MT_ATTACKER_GPUS:-4}" SISTEER_MT_TARGET_GPUS="${SISTEER_MT_TARGET_GPUS:-12}"
  export SISTEER_MT_ALLOCATION_GPUS="${SISTEER_MT_ALLOCATION_GPUS:-16}"
  export SISTEER_DUAL_SUBMIT_THRESHOLD_MIN="${SISTEER_DUAL_SUBMIT_THRESHOLD_MIN:-10}" SISTEER_DUAL_SUBMIT_MAX_JOBS="${SISTEER_DUAL_SUBMIT_MAX_JOBS:-16}"
  export SISTEER_GRANT_TIMEOUT_MIN="${SISTEER_GRANT_TIMEOUT_MIN:-10}"
  export SISTEER_STATE_DIR="${SISTEER_STATE_DIR:-$HOME/.local/state/sisteer-reproduce}"
  # the grading host's OBJECTS BANK (demo-import, 2026-10-11): where 05_stage_roster.sh stages the model's objects (--bank's default),
  # the root the Petri drive reads its bank view under (08_grade.sh -> grade_daemon.py --petri-bank-root -> the drive's F22_GRADING_HOST_BANK_ROOT)
  # and the bank 09_dose_ladder.sh builds the dose copies in (build_strength.py --bank / --parent-root <bank>/<key>/cells: the parents 05 staged);
  # ONE site value so the view the daemon builds is the view the drive reads and the parents 09 scales are the ones 05 staged — the paper's
  # legacy the grading host bank is NOT the default here
  export SISTEER_OBJECTS_BANK="${SISTEER_OBJECTS_BANK:-$SISTEER_STATE_DIR/objects}"
  # the SINGULAR is not a key: 09 alone read `SISTEER_OBJECT_BANK` (undocumented), so a site setting the plural
  # had 05 stage the parents under it while 09 looked for them under the old default — one model's objects split across two roots; refused by name
  [ -z "${SISTEER_OBJECT_BANK:-}" ] || repro_fatal "SISTEER_OBJECT_BANK is not a site key: the objects bank is SISTEER_OBJECTS_BANK (ONE root for 05_stage_roster.sh's staging, 08_grade.sh's Petri drive and 09_dose_ladder.sh's dose build) — rename it"
  export SISTEER_SHARD_GROUPS="${SISTEER_SHARD_GROUPS:-0,1,2 3,4 5,6 7,8}"      # the four disjoint multi-turn shard groups (§3.9)
  [ "$SISTEER_MT_ATTACKER_GPUS" -gt 0 ] && [ "$SISTEER_MT_TARGET_GPUS" -gt 0 ] || repro_fatal "SISTEER_MT_ATTACKER_GPUS / SISTEER_MT_TARGET_GPUS must be positive"
  # the cluster blocks: clusters/<label>.env -> CLUSTER__<label>__<VAR> (validated by lib/clusters.py, which refuses by name) — for the
  # labels THE FILE THIS RUN LOADED names ($SISTEER_CONFIG, the file sourced above), never the directory's config.env beside it: the
  # same read submit_twin.Site makes, so the labels a stage iterates, the primary it submits to and the blocks it reads are one file's.
  # It also reads the config file as written, so a value bash expanded above (an environment variable outside single quotes, which
  # the Python readers would keep as written) refuses here by key
  local sh; sh="$("$(repro_python)" "$REPRO_LIB_DIR/clusters.py" --config "$SISTEER_CONFIG" --sh)" || repro_fatal "the site config $SISTEER_CONFIG or its cluster blocks under $SISTEER_SITE_DIR/clusters did not load (reason above)"
  eval "$sh"
  repro_load_overrides
}
cluster_var(){   # <label> <VAR> -> the value (empty when unset); the ONE reader of a cluster block from shell
  local n="CLUSTER__${1}__${2}"; printf '%s' "${!n:-}"
}
cluster_fit_env_name(){   # <label> -> the prebuilt fit venv NAME on that cluster: CLUSTER_FIT_ENV_NAME > SISTEER_FIT_ENV_NAME > sdfbase-fit
  # (stage-E prep pass 2): the same resolution as submit_twin.Site.fit_env_name; every fit-family job env carries it as
  # SDFBASE_FIT_ENV_NAME (the sbatches' reader) and the grading-host-side fit venv paths (02 cl_python, 04 FIT_VENV_PY) derive from it
  local n; n="$(cluster_var "$1" CLUSTER_FIT_ENV_NAME)"; printf '%s' "${n:-${SISTEER_FIT_ENV_NAME:-sdfbase-fit}}"
}
cluster_registry_label(){   # <label> -> the cluster's registry label (CLUSTER_SUBSTRATE): its [[clusters]].name, every --dest, record substrates
  cluster_var "$1" CLUSTER_SUBSTRATE
}
cluster_mirror_selector(){   # <label> -> the name the site's repo-mirror tool knows this cluster by (CLUSTER_MIRROR_SELECTOR, default the site label)
  local s; s="$(cluster_var "$1" CLUSTER_MIRROR_SELECTOR)"; printf '%s' "${s:-$1}"
}
cluster_labels(){ echo $SISTEER_CLUSTERS; }
cluster_require(){   # <label>: refuse an unknown label by name
  case " $SISTEER_CLUSTERS " in *" $1 "*) ;; *) repro_fatal "cluster label '$1' is not one of SISTEER_CLUSTERS ($SISTEER_CLUSTERS)";; esac
}
repro_resolve_clusters(){   # the stage's CLUSTERS (comma list): default to every configured label; refuse an unknown one by name
  CLUSTERS="${CLUSTERS:-$(cluster_labels | tr ' ' ',')}"; local l; for l in ${CLUSTERS//,/ }; do cluster_require "$l"; done
}
repro_refuse_now(){   # <reason>: a refusal that stops the stage even when it fires inside a command substitution — `$(…)` runs in
  # a subshell, where a plain exit would end only the subshell and hand the caller an empty string to build a path from
  echo "REFUSED: $*" >&2
  [ "${BASH_SUBSHELL:-0}" -gt 0 ] && kill -TERM "$$" 2>/dev/null
  exit "${REPRO_REFUSE_RC:-2}"
}
REPRO_SITE_PLACEHOLDERS="SHARED HF_HOME STATE RUNSTATE STAGE WORK GRADE WANDB_PROJECT"   # the closed set src/sisteer/site.py expands
repro_dollar_token(){   # <text after a `$`> -> REPRO_TOKEN (the name as written; empty for a bare or unterminated `$` and for `${}`) and
  # REPRO_TOKEN_LEN (how many characters of the text it spans): `${NAME}` and `$NAME` read as src/sisteer/site.py reads them
  local re_braced='^\{([^}]*)\}' re_name='^([A-Za-z_][A-Za-z0-9_]*)'
  REPRO_TOKEN=""; REPRO_TOKEN_LEN=0
  if [[ "$1" =~ $re_braced ]]; then REPRO_TOKEN="${BASH_REMATCH[1]}"; REPRO_TOKEN_LEN=${#BASH_REMATCH[0]}
  elif [[ "$1" =~ $re_name ]]; then REPRO_TOKEN="${BASH_REMATCH[1]}"; REPRO_TOKEN_LEN=${#REPRO_TOKEN}
  fi
}
repro_expand_shared(){   # <label> <value> -> the value with $SHARED / ${SHARED} replaced by that cluster's CLUSTER_SHARED_ROOT
  # The shell twin of sisteer.site.expand for cluster-side roots. Only $SHARED expands here, and only on a cluster with a shared
  # root. Every other `$` refuses by name: another site placeholder, a name outside the closed set (a misspelt $SHARE among them)
  # and a bare `$` alike, so no `$` token of a site value reaches a command as written.
  local root out="" rest="$2"
  root="$(cluster_var "$1" CLUSTER_SHARED_ROOT)"
  while [[ "$rest" == *'$'* ]]; do
    out+="${rest%%\$*}"; rest="${rest#*\$}"
    repro_dollar_token "$rest"
    [ -n "$REPRO_TOKEN" ] || repro_refuse_now "'$2' holds a bare or unterminated '\$'; a site value holds only \$SHARED here"
    if [ "$REPRO_TOKEN" = SHARED ]; then
      [ -n "$root" ] || repro_refuse_now "'$2' holds \$SHARED but cluster '$1' sets no CLUSTER_SHARED_ROOT"
      out+="$root"; rest="${rest:$REPRO_TOKEN_LEN}"
      continue
    fi
    case " $REPRO_SITE_PLACEHOLDERS " in
      *" $REPRO_TOKEN "*) repro_refuse_now "'$2' holds \$$REPRO_TOKEN, which a cluster path cannot expand (only \$SHARED, the cluster's own root, expands here)";;
      *) repro_refuse_now "'$2' holds \$$REPRO_TOKEN, which is not a site placeholder (known: \$${REPRO_SITE_PLACEHOLDERS// /, \$}); only \$SHARED, the cluster's own root, expands here";;
    esac
  done
  printf '%s' "$out$rest"
}
repro_check_site_values(){   # <config file>, after it was sourced: every `$` its values still hold is a site placeholder, else refuse by key
  # Bash has already expanded each variable written outside single quotes, so a `$` left in a value was quoted to stay as written,
  # and only a site placeholder is expanded later; a name outside the closed set (a misspelt $SHARE) or a bare `$` refuses here,
  # at load, before any stage composes a path or a command from it. lib/clusters.py makes the same check on the file as written
  # (repro_load_config runs it next), which also refuses a variable bash did expand here.
  local k v rest
  while IFS= read -r k; do
    v="${!k-}"; rest="$v"
    while [[ "$rest" == *'$'* ]]; do
      rest="${rest#*\$}"
      repro_dollar_token "$rest"
      [ -n "$REPRO_TOKEN" ] || repro_fatal "$k='$v' in $1 holds a bare or unterminated '\$'; a site value holds only the site placeholders"
      case " $REPRO_SITE_PLACEHOLDERS " in
        *" $REPRO_TOKEN "*) rest="${rest:$REPRO_TOKEN_LEN}";;
        *) repro_fatal "$k='$v' in $1 holds \$$REPRO_TOKEN, which is not a site placeholder (known: \$${REPRO_SITE_PLACEHOLDERS// /, \$}) and which nothing expands after the file is sourced";;
      esac
    done
  done < <(sed -nE 's/^[[:space:]]*(export[[:space:]]+)?([A-Z][A-Z0-9_]*)=.*/\2/p' "$1" | sort -u)
}
# the run-overrides layer (§3.2): the paper's defaults unless a file says otherwise; recorded on every registry row through
# SISTEER_RUN_OVERRIDES (emit.RecordWriter stamps extra.run_overrides.{id,sha256,keys})
repro_load_overrides(){
  local f="${REPRO_OVERRIDES:-${SISTEER_RUN_OVERRIDES:-}}"
  [ -n "$f" ] || f="$REPRO_DIR/overrides/paper_defaults.env"
  [ -r "$f" ] || repro_fatal "no run-overrides file at $f"
  set -a; source "$f"; set +a
  [ -n "${OVERRIDE_ID:-}" ] || repro_fatal "overrides file $f sets no OVERRIDE_ID"
  # 30-fix: a document that holds evals out names ITS model, and it is this stage's model — a runner retaining
  # SISTEER_RUN_OVERRIDES for one model, or a file passed for another, never carries a hold-out across models
  if [ -n "${OVERRIDE_HOLD_OUT_EVALS:-}" ]; then
    [ -n "${OVERRIDE_MODEL:-}" ] || repro_fatal "overrides file $f holds evals out (OVERRIDE_HOLD_OUT_EVALS=$OVERRIDE_HOLD_OUT_EVALS) but names no OVERRIDE_MODEL — a hold-out is one model's ruling"
    [ -z "${REPRO_MODEL:-}" ] || [ "$OVERRIDE_MODEL" = "$REPRO_MODEL" ] || repro_fatal "overrides file $f holds evals out for model $OVERRIDE_MODEL, but this stage runs --model $REPRO_MODEL — a model's ruling never travels to another model"
  fi
  SISTEER_RUN_OVERRIDES="$(cd "$(dirname "$f")" && pwd)/$(basename "$f")"; export SISTEER_RUN_OVERRIDES
  SISTEER_RUN_OVERRIDES_SHA256="$(sha256sum "$SISTEER_RUN_OVERRIDES" | cut -d' ' -f1)"; export SISTEER_RUN_OVERRIDES_SHA256
}

# --- the models table -------------------------------------------------------------------------------------------------
repro_models_table(){ echo "${SISTEER_MODELS_TABLE:-$REPRO_DIR/models.yaml}"; }
repro_model(){   # <key> <dotted.path> -> the value (scalars printed plain, lists space-joined, maps as JSON); refuses a missing path
  "$(repro_python)" "$REPRO_LIB_DIR/models.py" --table "$(repro_models_table)" get "$1" "$2"
}
repro_require_model(){
  [ -n "${REPRO_MODEL:-}" ] || repro_fatal "--model <key> is required (keys: $("$(repro_python)" "$REPRO_LIB_DIR/models.py" --table "$(repro_models_table)" keys | tr '\n' ' '))"
  "$(repro_python)" "$REPRO_LIB_DIR/models.py" --table "$(repro_models_table)" exists "$REPRO_MODEL" || repro_fatal "model key '$REPRO_MODEL' is not in $(repro_models_table)"
}
repro_require_view(){   # <key> <view>: refuse a view that is not one of the model's phase views (models.yaml `views`)
  local views; views="$("$(repro_python)" "$REPRO_LIB_DIR/models.py" --table "$(repro_models_table)" views "$1")"
  grep -qx -- "$2" <<< "$views" || repro_fatal "view '$2' is not a phase view of $1 (${views//$'\n'/ })"
}
repro_require_runnable(){   # <key>: a model with run_in_this_work: false answers --dry-run only (FD 4); a real run refuses by name
  [ "$(repro_model "$1" run_in_this_work)" = true ] || [ "${REPRO_DRY_RUN:-0}" = 1 ] || repro_fatal "$1 has run_in_this_work: false in models.yaml — every stage answers --dry-run for it; nothing is submitted"
}

# --- executing (and printing) commands --------------------------------------------------------------------------------
repro_run(){   # <argv...>: under --dry-run print `DRY-RUN: <argv>` and do nothing; else execute
  if [ "${REPRO_DRY_RUN:-0}" = 1 ]; then printf 'DRY-RUN:'; printf ' %q' "$@"; echo; return 0; fi
  "$@"
}
repro_run_sh(){   # <shell string>: the same for a pipeline / redirection
  if [ "${REPRO_DRY_RUN:-0}" = 1 ]; then echo "DRY-RUN: $1"; return 0; fi
  bash -c "$1"
}
repro_ssh(){   # <label> <remote shell string>: run on that cluster's login host (BatchMode); printed verbatim under --dry-run
  # Test seam, the same as submit_twin.py Remote: CLUSTER_SSH=local:<dir> runs the string with `bash -c` on THIS host with
  # FAKE_SLURM_DIR=<dir> (the front-door integration tests drive the real shell path against fake Slurm binaries this way)
  # Never reads stdin (`ssh -n`; the seam `</dev/null`): the callers run it INSIDE `while read` loops fed by a pipe or process
  # substitution (06 --gate, reconcile.sh), and a stdin-reading ssh would swallow the loop's remaining lines.
  local host; host="$(cluster_var "$1" CLUSTER_SSH)"; [ -n "$host" ] || repro_fatal "cluster $1 has no CLUSTER_SSH"
  if [ "${REPRO_DRY_RUN:-0}" = 1 ]; then echo "DRY-RUN: ssh $host '$2'"; return 0; fi
  # the seam's DISCONNECTED host: `<dir>/ssh_down` present -> ssh's own connection-failure status (255) WITHOUT running the command, so a
  # caller's transport handling (06 --gate: a lost login host is a gate failure, never "not registered yet") is testable.
  # The seam's LOGIN BANNER: `<dir>/ssh_banner` present -> its text on STDOUT before the command's own output, as a login shell whose rc file
  # echoes does on a real host, so a caller that parses the command's stdout is testable against a prefixed document (06 --gate's probe: a
  # reader failure marked on the unit, never a coverage fact and never a fresh verdict). The status stays the command's.
  case "$host" in local:*) [ -e "${host#local:}/ssh_down" ] && return 255
                           { [ -e "${host#local:}/ssh_banner" ] && cat "${host#local:}/ssh_banner"; FAKE_SLURM_DIR="${host#local:}" bash -c "$2" </dev/null; }; return $?;; esac
  ssh -n -o BatchMode=yes -o ConnectTimeout=30 "$host" "$2"
}
repro_ssh_put(){   # <label> <local file> <remote path> <sha256>: ship ONE small file to that cluster's login host BY CONTENT — atomically and
  # digest-verified (the CURRENT tree's lib/coverage_probe.py travels to the cluster as one file).
  # 1. reuse: the bytes are already there when `sha256sum -c` of the FINAL name passes (never `test -s`: a partial upload is non-empty);
  # 2. else upload to `<remote>.tmp.<nonce>` (scp; the `local:<dir>` seam copies), then ONE remote command verifies the temp file's digest and
  #    `mv -f`s it over the final name — the final name only ever holds verified bytes; a failed verify removes the temp and retries once, then
  #    refuses by name (the transport is corrupting bytes). Never reads the caller's stdin (scp `</dev/null`; the seam copies).
  # rc 0 = the bytes are at <remote path>; 255 = the host could not be reached (ssh's own status); 1 = a digest that would not verify twice.
  local l="$1" src="$2" dst="$3" want="$4" host nonce tmp try rc
  host="$(cluster_var "$l" CLUSTER_SSH)"; [ -n "$host" ] || repro_fatal "cluster $l has no CLUSTER_SSH"
  if [ "${REPRO_DRY_RUN:-0}" = 1 ]; then echo "DRY-RUN: ship $src -> $host:$dst (sha256 ${want:0:16}…, verified remotely)"; return 0; fi
  [ -f "$src" ] || { echo "REFUSED: repro_ssh_put: no local file $src" >&2; return 1; }
  if repro_ssh "$l" "echo '$want  $dst' | sha256sum -c --quiet >/dev/null 2>&1"; then return 0; fi
  rc=$?; [ "$rc" = 255 ] && return 255
  nonce="$(date +%s).$$.$RANDOM"; tmp="$dst.tmp.$nonce"
  for try in 1 2; do
    case "$host" in
      local:*) [ -e "${host#local:}/ssh_down" ] && return 255
               mkdir -p "$(dirname "$dst")" && cp "$src" "$tmp" || return 1;;
      *)       ssh -n -o BatchMode=yes -o ConnectTimeout=30 "$host" "mkdir -p '$(dirname "$dst")'" </dev/null || return 255
               scp -q -o BatchMode=yes -o ConnectTimeout=30 "$src" "$host:$tmp" </dev/null || return 255;;
    esac
    if repro_ssh "$l" "echo '$want  $tmp' | sha256sum -c --quiet >/dev/null 2>&1 && mv -f '$tmp' '$dst'"; then return 0; fi
    rc=$?; [ "$rc" = 255 ] && return 255
    repro_ssh "$l" "rm -f '$tmp'" >/dev/null 2>&1 || true
  done
  echo "REFUSED: repro_ssh_put: the uploaded $src never verified as sha256 $want at $host:$dst (two attempts, each removed) — the transport is corrupting bytes" >&2
  return 1
}
repro_mkdir(){ if [ "${REPRO_DRY_RUN:-0}" = 1 ]; then echo "DRY-RUN: mkdir -p $*"; else mkdir -p "$@"; fi; }
repro_dry_flag(){ [ "${REPRO_DRY_RUN:-0}" = 1 ] && echo --dry-run; return 0; }   # `--dry-run` to forward to a lib tool, empty otherwise
repro_write(){   # <path> <content>: write a small file (dry-run: say so)
  if [ "${REPRO_DRY_RUN:-0}" = 1 ]; then echo "DRY-RUN: write $1 ($(printf '%s' "$2" | wc -c) bytes)"; return 0; fi
  mkdir -p "$(dirname "$1")"; printf '%s' "$2" > "$1.tmp.$$" && mv -f "$1.tmp.$$" "$1"
}

# --- state: the grading host copy of the resume ledger (§7) ------------------------------------------------------------------
repro_registry_root(){ eval echo "$SISTEER_REGISTRY_ROOT"; }   # the grading host's registry root with `~` expanded (registry.sqlite lives under it)
# the registry campaign this site files under AND reads back under — ONE resolution for every face of the front door: the site's
# SISTEER_REGISTRY_CAMPAIGN (the site config the run was given — --config, config.env by default — sourced into the stage by
# repro_load_config; submit_twin.Site reads the SAME file, never config.env beside it), else the job-env spelling REGISTRY_CAMPAIGN, else
# the paper frame's sdfbase_full22. 03/04/05 spell the same expression on their own CAMPAIGN= lines (pinned by tests; 03 and 04's no-lib
# branch hand that value to the job explicitly), the pins writer mirrors it in Python (make_full22_configs.registry_campaign_default), and
# the JOB side reads it HERE: slurm_job_env / slurm_base_env (lib/slurm.sh), submit_twin.Site.registry_campaign (the env every sbatch
# carries) and 08's daemon. Before this the job side read the site's key alone and defaulted, so a
# site naming its campaign as REGISTRY_CAMPAIGN fitted and read done-ness under it while every fit job filed under sdfbase_full22 — the
# pins writer, reading under the site's campaign, found no row, wrote no pin, and 05 refused the completed fit.
# tests/test_registry_campaign_resolution.py runs every face over one table.
repro_registry_campaign(){ printf '%s' "${SISTEER_REGISTRY_CAMPAIGN:-${REGISTRY_CAMPAIGN:-sdfbase_full22}}"; }
repro_state_dir(){ echo "$SISTEER_STATE_DIR/${1:?model}"; }
repro_cluster_state_root(){   # <label> <model> -> $SHARED/exp/reproduce/<model> on that cluster
  echo "$(repro_expand_shared "$1" "$SISTEER_REPRODUCE_ROOT")/${2:?model}"
}
repro_done(){   # <model> <stage> <unit> -> 0 when the grading host completion marker exists
  [ -f "$(repro_state_dir "$1")/$2/DONE.$3" ]
}
memprobe_archive_stale(){   # <cluster label> <remote state dir> <arm> -> prints the archive path; 0 ONLY when the stale memprobe record is
  # archived under a UNIQUE name and read back by sha256 with the live filename absent (a masked mv could have cleared the
  # marker and let a new probe overwrite the original evidence; a same-second second detection could overwrite the first archive). The
  # report moves beside it when present — optional, never masking the record's result. Nothing else happens on a failure.
  # The archive is a COPY whose sha256 is read back FIRST; the live name is removed only after the match,
  # so a failed read-back leaves the record under its live name (the copy removed) and the next 02 run sees it — never an unverified archive
  # with nothing live to refuse on.
  local l="$1" st="$2" arm="$3" rec want dest got stamp
  rec="$st/memprobe.$arm.json"; stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  want="$(repro_ssh "$l" "sha256sum '$rec' | cut -d' ' -f1")" || { repro_warn "memprobe archive: cannot hash $rec on $l"; return 1; }
  [ -n "$want" ] || { repro_warn "memprobe archive: empty hash for $rec on $l"; return 1; }
  dest="$(repro_ssh "$l" "d='$rec.stale.$stamp'; n=1; while [ -e \"\$d\" ]; do n=\$((n+1)); d='$rec.stale.$stamp'.\$n; done; cp -n '$rec' \"\$d\" && [ -f \"\$d\" ] && printf '%s' \"\$d\"")" \
    || { repro_warn "memprobe archive: copy of $rec failed on $l — the record stays under its live name; nothing cleared, nothing submitted"; return 1; }
  [ -n "$dest" ] || { repro_warn "memprobe archive: no destination reported for $rec on $l — the record stays under its live name"; return 1; }
  got="$(repro_ssh "$l" "sha256sum '$dest' | cut -d' ' -f1")" \
    || { repro_ssh "$l" "rm -f '$dest'" >/dev/null 2>&1; repro_warn "memprobe archive: cannot hash the archive $dest — copy removed, the record stays under its live name"; return 1; }
  [ "$got" = "$want" ] \
    || { repro_ssh "$l" "rm -f '$dest'" >/dev/null 2>&1; repro_warn "memprobe archive: $dest hashes $got but the record hashed $want — copy removed, the record stays under its live name; refusing to proceed"; return 1; }
  repro_ssh "$l" "rm -f '$rec' && [ ! -e '$rec' ]" \
    || { repro_warn "memprobe archive: $dest is verified but the live name $rec could not be removed on $l — refusing to proceed (the record stays live; the next run archives again)"; return 1; }
  repro_ssh "$l" "[ ! -f '$st/memprobe_report.$arm.md' ] || mv -n '$st/memprobe_report.$arm.md' '$dest.report.md'" \
    || repro_warn "memprobe archive: the report beside $rec was not moved (the record IS archived at $dest)"
  printf '%s\n' "$dest"
}
repro_mark_done(){   # <model> <stage> <unit> [note]
  local d; d="$(repro_state_dir "$1")/$2"
  if [ "${REPRO_DRY_RUN:-0}" = 1 ]; then echo "DRY-RUN: mark $d/DONE.$3"; return 0; fi
  mkdir -p "$d"; printf '%s %s\n' "$(date -u +%FT%TZ)" "${4:-}" > "$d/DONE.$3"
}
repro_superseded_dir(){   # <base> [stamp] -> prints a FRESH archive directory <base>/<utc to the microsecond>[-NN], created here; 1 when none could be
  # A marker is archived by `mv` into a stamped directory, never deleted — and two archivals that named the SAME
  # <base>/<utc-to-the-second>/ had the second `mv -f` overwrite the first archive (the stage-02 pools marker, gate 2026-10-09:
  # on a fast node two not-honoured runs fell in one second and tests 6–12 of tests/test_stage02_pools_rerunnable.py cascaded on
  # the count). So the stamp is to the microsecond (fixed width: a lexical sort of the directories is their archival order) and
  # the directory is RESERVED by a bare `mkdir`, which fails atomically when it exists (a frozen clock, a same-microsecond sibling,
  # a concurrent run) — a collision takes the next `-NN` suffix, which sorts right after its base and before any later stamp (two
  # digits: up to 99 siblings of ONE stamp keep their order, and a real clock never repeats a microsecond across two `date` forks —
  # the suffix is for a frozen clock and a concurrent run). An existing directory is never reused, so nothing archived here can be
  # written over; a parent that cannot be made is a failure the caller names (nothing printed).
  # A LIVE-path helper: its stdout IS the directory, so a dry run cannot print a note there (the caller would `mv` into the note's
  # text) — under REPRO_DRY_RUN=1 it creates nothing, prints nothing and refuses by name; the caller's dry-run branch skips the
  # archival before reaching it (02_prepare_model.sh: a marker in a dry run is said as unvalidated and left alone).
  # [stamp]: the caller's own stamp as the directory's name instead of a fresh one — for ONE event that archives at several places
  # (the geometry revocation, lib/slurm.sh slurm_revoke_geometry: the site copy here, every cluster's overlay over ssh), whose
  # messages name the same stamp everywhere; the reservation and the `-NN` collision rule are the same, so a second event with the
  # same stamp (a frozen clock) still lands beside the first, never over it. Mint it with repro_superseded_stamp so it keeps the width.
  local base="${1:?base}" stamp="${2:-}" d n=1
  if [ "${REPRO_DRY_RUN:-0}" = 1 ]; then repro_warn "repro_superseded_dir: refused under --dry-run (a dry run archives nothing; the caller skips the archival before this point)"; return 1; fi
  [ -n "$stamp" ] || stamp="$(repro_superseded_stamp)"
  mkdir -p "$base" || return 1
  d="$base/$stamp"
  until mkdir "$d" 2>/dev/null; do
    [ -e "$d" ] || return 1                      # not a collision (the base is unwritable or gone): refuse, print nothing
    n=$((n+1)); d="$base/$stamp-$(printf '%02d' "$n")"
  done
  printf '%s\n' "$d"
}
repro_superseded_stamp(){ date -u +%Y%m%dT%H%M%S.%6NZ; }   # the archive stamp: UTC to the microsecond, fixed width (GNU date %N) — the one format every superseded/ directory is named in
repro_ledger_append(){   # <model> <file> <json-line>: append one JSON line to the grading host ledger (fsync'd)
  if [ "${REPRO_DRY_RUN:-0}" = 1 ]; then echo "DRY-RUN: ledger $2 += $3"; return 0; fi
  "$(repro_python)" "$REPRO_LIB_DIR/grant_ledger.py" append --file "$(repro_state_dir "$1")/$2" --json "$3"
}
repro_submitted(){   # <model> <cluster> <jobid> <unit> [json extra]: the `submitted` line + the submissions ledger + the durable watcher
  local model="$1" cluster="$2" jobid="$3" unit="$4" extra="${5:-{\}}"
  echo "submitted $cluster $jobid $unit"
  repro_ledger_append "$model" submissions.jsonl "{\"event\":\"submitted\",\"utc\":\"$(date -u +%FT%TZ)\",\"cluster\":\"$cluster\",\"job_id\":\"$jobid\",\"unit\":\"$unit\",\"model\":\"$model\",\"extra\":$extra}"
  repro_launch_watcher "$model" "$cluster" "$jobid" "$unit"
}
repro_launch_watcher(){   # <model> <cluster> <jobid> <unit>: lib/watch_job.sh in the background, events under the cluster's watch/ dir
  local host root events
  host="$(cluster_var "$2" CLUSTER_SSH)"; root="$(repro_cluster_state_root "$2" "$1")"
  events="$(repro_state_dir "$1")/watch/$2_$3.jsonl"
  if [ "${REPRO_DRY_RUN:-0}" = 1 ]; then echo "DRY-RUN: $REPRO_LIB_DIR/watch_job.sh --slurm $3 --host $host --shared-root $(cluster_var "$2" CLUSTER_SHARED_ROOT) --events $events --remote-events $root/watch/$3.jsonl &"; return 0; fi
  mkdir -p "$(dirname "$events")"
  nohup "$REPRO_LIB_DIR/watch_job.sh" --slurm "$3" --host "$host" --shared-root "$(cluster_var "$2" CLUSTER_SHARED_ROOT)" \
    --events "$events" --remote-events "$root/watch/$3.jsonl" --unit "$4" >/dev/null 2>&1 &
}

# --- the compute replan gate (§6): fan-out stages refuse until tools/replan_compute.py has run for the model -------------
repro_require_compute_plan(){   # <model> [complete]: `complete` also refuses a plan whose serving rows are still pending
  local p; p="$(repro_state_dir "$1")/compute_plan.json"
  [ -f "$p" ] || repro_fatal "no compute plan at $p — run reproduce/tools/replan_compute.py --model $1 (it re-derives every GPU-hour row from memprobe.json + probe_verdict.json; the fan-out stages refuse without it, plan §6)"
  if [ "${2:-}" = complete ]; then
    # the plan says whether every row has a number (`complete`; schema 3) — a stage that prices its wave on the serving rows
    # (06's steered multi-turn / padding, 09) asks for a complete plan, never for the file's existence alone
    local c; c="$("$(repro_python)" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("yes" if d.get("complete") is True else "no:" + ",".join(d.get("pending_rows") or ["schema<3"]))' "$p" 2>/dev/null || echo "no:unreadable")"
    [ "$c" = yes ] || repro_fatal "the compute plan at $p is INCOMPLETE (${c#no:}) — re-run reproduce/tools/replan_compute.py --model $1 with --probe <the certified probe_verdict.json> (the serving rows price the steered waves)"
  fi
}

# --- the data-parallel fit width --------------------------------------------------------------------------------------
# THE width is the fits config's top-level `fit_width:` line (absent = 1); the sbatch reads the same line and refuses an
# allocation that differs. These two are the SUBMITTER's copies of sdfbase_fit_cell.sbatch's own fit_width_flags (a test pins
# the two spellings equal): empty at width 1 (the header's gpu:1 / 16 CPUs / 200G stand), else the per-GPU-scaled allocation.
repro_fit_width_of_config(){   # <fits yaml> -> N (1 when the key is absent or the file unreadable)
  local w; w="$(sed -n 's/^fit_width: *\([0-9][0-9]*\) *$/\1/p' "${1:-/nonexistent}" 2>/dev/null | head -1)"
  echo "${w:-1}"
}
repro_fit_width_flags(){   # <N> -> `--gres=gpu:N --cpus-per-task=16N --mem=128N G`, or nothing at width 1
  local n="${1:-1}"
  [ "$n" -gt 1 ] 2>/dev/null || return 0
  printf -- '--gres=gpu:%d --cpus-per-task=%d --mem=%dG' "$n" "$((16 * n))" "$((128 * n))"
}

# --- the fits config of a fit checkpoint ------------------------------------------------------------------------------
# <hf repo> <revision> [explicit path] -> the repo-relative path of the fits config (configs/sdfbase_<key>_fits.yaml) whose
# `model` / `model_revision` pin that checkpoint. The generator key may differ from the models table's model_id (qwen35_4b_demo vs
# qwen35_4b_demo_base), so the CONFIG is the authority for the pool dir and the sbatch positional. The CANONICAL config is the one
# without a seed-replicate marker: make_sdfbase_configs.py's `_eb64_s<N>` sibling keys (a name ending `_s<N>_fits.yaml`, a declared
# `train_kwargs.default.seed`) are replicates of the canonical config and never candidates — the worked example's checkpoint is
# pinned by its canonical config AND three seed replicates, which used to refuse as "4 fits configs (expected exactly one)".
# Two canonical configs pinning the same checkpoint refuse by name; none refuses. An explicit path (02 / 03 --fits-config) wins
# when it pins the checkpoint and lies under the repo (the job side reads it from the checkout); a replicate may be named so.
repro_fits_config_for(){   # <hf repo> <revision> [explicit path] [fit model_id]
  "$(repro_python)" - "$REPRO_REPO" "${1:?hf repo}" "${2:?revision}" "${3:-}" "${4:-}" <<'PY'
import glob, os, re, sys, yaml
repo, hf, rev, explicit, fit_id = sys.argv[1:6]
REPLICATE_NAME = re.compile(r"_s[0-9]+_fits\.yaml$")


def load(p):
    try:
        with open(p, encoding="utf-8") as fh:
            c = yaml.safe_load(fh)
    except Exception as exc:  # a config that does not parse is said by name, never skipped
        sys.exit(f"REFUSED: cannot read fits config {os.path.relpath(p, repo)}: {exc}")
    return c if isinstance(c, dict) else {}


def pins(c):
    # the revision is compared as TEXT: YAML reads an all-digit 40-hex revision as an integer, which `==` against the table's string misses
    return str(c.get("model") or "") == hf and str(c.get("model_revision") or "") == rev


def is_replicate(p, c):
    default = (c.get("train_kwargs") or {}).get("default") or {}
    return bool(REPLICATE_NAME.search(os.path.basename(p))) or (isinstance(default, dict) and default.get("seed") is not None)


def rel(p):
    r = os.path.relpath(os.path.realpath(p), os.path.realpath(repo))
    if r.startswith(".."):
        sys.exit(f"REFUSED: fits config {p} lies outside the repo {repo} (the job side reads it from the checkout)")
    return r


if explicit:
    p = explicit if os.path.isabs(explicit) else os.path.join(repo, explicit)
    if not os.path.isfile(p):
        sys.exit(f"REFUSED: --fits-config {explicit}: no such file")
    c = load(p)
    if not pins(c):
        sys.exit(f"REFUSED: --fits-config {explicit} pins {c.get('model')}@{c.get('model_revision')}, not the model's fit checkpoint {hf}@{rev}")
    print(rel(p))
    sys.exit(0)
# the lane NAMED by the table's fit model_id (coordinator ruling 2026-09-29: fit slugs = the table's fit model_ids, `_base`
# and all) is the canonical config when it exists and pins the checkpoint — two table rows may share one checkpoint under
# their own keys (the worked example qwen35_4b_demo fits the paper's Qwen3.5-4B-Base as qwen35_4b_demo_base, beside the
# paper's qwen35_4b_base), and the uniqueness scan below cannot tell them apart; a named config that pins another
# checkpoint is refused by name, never skipped
if fit_id:
    named = os.path.join(repo, "configs", f"sdfbase_{fit_id}_fits.yaml")
    if os.path.isfile(named):
        c = load(named)
        if not pins(c):
            sys.exit(f"REFUSED: configs/sdfbase_{fit_id}_fits.yaml (the lane named by the table's fit model_id {fit_id!r}) pins "
                     f"{c.get('model')}@{c.get('model_revision')}, not the model's fit checkpoint {hf}@{rev} — one pin, one table")
        print(os.path.relpath(named, repo))
        sys.exit(0)
canonical, replicates = [], []
for p in sorted(glob.glob(os.path.join(repo, "configs", "sdfbase_*_fits.yaml"))):
    c = load(p)
    if pins(c):
        (replicates if is_replicate(p, c) else canonical).append(os.path.relpath(p, repo))
if len(canonical) != 1:
    reps = f"; seed replicates, not candidates: {replicates}" if replicates else ""
    sys.exit(f"REFUSED: {len(canonical)} canonical fits configs pin {hf}@{rev} (expected exactly one): {canonical}{reps} — name one with --fits-config <path>")
print(canonical[0])
PY
}

# --- the repo's own identity -----------------------------------------------------------------------------------------
repro_head_sha(){ git -C "$REPRO_REPO" rev-parse HEAD 2>/dev/null || echo unknown; }
repro_require_clean_tree(){
  [ "${REPRO_DRY_RUN:-0}" = 1 ] && { repro_note "a real run requires a clean tree at $REPRO_REPO (the wave sha pins what the cluster runs)"; return 0; }
  [ -z "$(git -C "$REPRO_REPO" status --porcelain --untracked-files=no 2>/dev/null)" ] || repro_fatal "the repo at $REPRO_REPO has modified tracked files — commit first (the wave sha pins the tree the cluster runs)"
}
