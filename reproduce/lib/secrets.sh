#!/bin/bash
# reproduce/lib/secrets.sh — the ONE place a credential enters a shell of the front door (bash; source it).
#
# Three modes, never a fourth (docs/secrets.md):
#   SISTEER_SECRETS_MODE=env  plain environment variables: ANTHROPIC_API_KEY (ONE key = EVERY credential label — lowprio / highprio /
#                             batch all resolve to it; the *_LOWPRIO / *_HIGHPRIO / *_BATCH names are op-mode only and
#                             are never required in env mode), OPENROUTER_API_KEY, MODAL_TOKEN_ID /
#                             MODAL_TOKEN_SECRET, HF_TOKEN, WANDB_API_KEY.
#   SISTEER_SECRETS_MODE=op   a secret-manager CLI wrapper, `$SISTEER_SECRET_CMD <item>` (config.env names the command,
#                             with no default; items named by its SISTEER_OP_ITEM_* keys). Every credential label keeps its own item.
#   SISTEER_SECRETS_MODE=wif  GCP workload identity federation: the Python judges (sisteer.vault) authenticate with the
#                             credential label's federation ids, read from its file under $SISTEER_WIF_ENV_DIR (default ~/.anthropic;
#                             sisteer.vault.WIF_ORG_FILES), and NO Anthropic API key exists — secret_anthropic refuses by
#                             name, a generated unit's key line only scrubs the ANTHROPIC_* variables, and nothing reads,
#                             sets or exports ANTHROPIC_API_KEY*. The other secrets (OpenRouter, Modal, HF, W&B) resolve as
#                             under op — federation covers the Claude API only.
# No key is read from a file inside the repo, written to a log, placed on cluster storage or echoed. A value crosses a
# process boundary ONLY on stdin (hf_fetch.sh) or through a variable the child inherits — never argv.
#
# Detached `systemd-run --user` units inherit NOTHING from the caller's shell. The four grading launchers
# (petri_drive_full22.sh, judge/launch_judge.sh, sampled/grade_sampled.sh, seedrep/grade_battery.sh) therefore add
# `$(secrets_unit_flags)` to their systemd-run line: under env mode it is NAME-ONLY `--setenv=NAME` imports (systemd
# >= 247 copies the value from the caller's environment) — nothing on argv, no file, nothing in the journal; on an older
# systemd the fallback is a 0600 EnvironmentFile under $XDG_RUNTIME_DIR (tmpfs), unlinked once the unit has started:
# every launcher captures `uflags="$(secrets_unit_flags)"`, passes $uflags to systemd-run and calls
# `secrets_unit_flags_cleanup "$uflags"` right after (a secret-bearing file must never outlive its unit). Under op mode
# the flags carry the secret-manager CLI's path (`--setenv=SISTEER_SECRET_CMD=<path>`), a value and not a secret: the
# unit's generated key line and its Python vault run that command. Under wif mode the flags carry the MODE itself —
# `--setenv=SISTEER_SECRETS_MODE=wif`, plus `--setenv=SISTEER_WIF_ENV_DIR=<dir>` when the site overrides the ids directory —
# and no credential: a unit that does not receive the mode runs its vault (sisteer.vault.secrets_mode) under the default `op`
# and reaches for a key that does not exist, while its generated key line (secrets_unit_key_line) only scrubs. The mode and
# the directory are plain values, not secrets, so they ride argv on every systemd and need no EnvironmentFile.
# The HF token (secret_hf / secret_hf_cluster_read) defaults its SOURCE to the mode under op / env and to `op` under wif
# (federation covers the Claude API only; `wif` is not an HF token source) — SISTEER_HF_TOKEN_SOURCE=op|env|none overrides.

secrets_mode(){ echo "${SISTEER_SECRETS_MODE:-op}"; }
secrets_op_bin(){   # the secret-manager CLI that SISTEER_SECRET_CMD names, as an absolute path (a bare name is looked up on PATH)
  local c="${SISTEER_SECRET_CMD:-}" b
  [ -n "$c" ] || { echo "REFUSED: SISTEER_SECRETS_MODE=op but SISTEER_SECRET_CMD is not set: it names the secret-manager CLI that prints one item's value (config.env); set it, or set SISTEER_SECRETS_MODE=env and export the documented variables instead" >&2; return 1; }
  case "$c" in */*) b="$c";; *) b="$(command -v -- "$c" 2>/dev/null || true)";; esac
  [ -n "$b" ] && [ -f "$b" ] && [ -x "$b" ] || { echo "REFUSED: SISTEER_SECRET_CMD does not name an executable (not found, or not on PATH)" >&2; return 1; }
  case "$b" in /*) ;; *) b="$PWD/$b";; esac
  echo "$b"
}
secrets_op_item_for_org(){   # a credential label (lowprio|highprio|batch) -> the wrapper's item name (config.env SISTEER_OP_ITEM_ANTHROPIC_<LABEL>)
  case "$1" in
    lowprio)  echo "${SISTEER_OP_ITEM_ANTHROPIC_LOWPRIO:-lowprio}";;
    highprio) echo "${SISTEER_OP_ITEM_ANTHROPIC_HIGHPRIO:-highprio}";;
    batch)    echo "${SISTEER_OP_ITEM_ANTHROPIC_BATCH:-batch}";;
    *) echo "REFUSED: unknown Anthropic credential label '$1' (lowprio|highprio|batch)" >&2; return 1;;
  esac
}
secrets_env_var_for_org(){   # the variable sisteer.vault.ANTHROPIC_ORGS reads for a credential label under OP mode
  case "$1" in lowprio) echo ANTHROPIC_API_KEY_LOWPRIO;; highprio) echo ANTHROPIC_API_KEY;; batch) echo ANTHROPIC_API_KEY_BATCH;;
    *) echo "REFUSED: unknown Anthropic credential label '$1'" >&2; return 1;; esac
}
secret_anthropic(){   # <credential label> -> the key on stdout (for `VAR="$(secret_anthropic lowprio)"`); never logged; REFUSED under wif (no key exists)
  local org="${1:-lowprio}"
  if [ "$(secrets_mode)" = wif ]; then
    echo "REFUSED: SISTEER_SECRETS_MODE=wif: the $org pool has NO API key — the judges authenticate through workload identity federation (sisteer.vault reads the pool's ids file); nothing to print, never the secret-manager CLI" >&2; return 1
  fi
  if [ "$(secrets_mode)" = env ]; then
    [ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "REFUSED: SISTEER_SECRETS_MODE=env: ANTHROPIC_API_KEY must be set (one key serves every credential label: $org)" >&2; return 1; }
    printf '%s' "$ANTHROPIC_API_KEY"; return 0
  fi
  local b item; b="$(secrets_op_bin)" || return 1; item="$(secrets_op_item_for_org "$org")" || return 1
  "$b" "$item"
}
secret_openrouter(){
  if [ "$(secrets_mode)" = env ] || [ -n "${OPENROUTER_API_KEY:-}" ]; then
    [ -n "${OPENROUTER_API_KEY:-}" ] || { echo "REFUSED: SISTEER_SECRETS_MODE=env: OPENROUTER_API_KEY must be set" >&2; return 1; }
    printf '%s' "$OPENROUTER_API_KEY"; return 0
  fi
  local b; b="$(secrets_op_bin)" || return 1; "$b" "${SISTEER_OP_ITEM_OPENROUTER:-OpenRouter API Key}"
}
_secret_hf_from(){   # <source op|env|none> <op item>: the HF token from that source; `none` prints nothing and returns 0
  case "$1" in
    none) return 0;;
    env)  [ -n "${HF_TOKEN:-}" ] || { echo "REFUSED: HF token source is env but HF_TOKEN is unset" >&2; return 1; }; printf '%s' "$HF_TOKEN";;
    op)   local b; b="$(secrets_op_bin)" || return 1; "$b" "$2";;
    *) echo "REFUSED: HF token source must be op|env|none" >&2; return 1;;
  esac
}
secrets_hf_default_source(){   # the HF token source the mode implies: op -> op, env -> env, wif -> op (`wif` is not a token
  # source — _secret_hf_from would refuse it, the preflight's `|| true` would swallow the refusal and a gated tokenizer would be
  # fetched tokenless; federation covers the Claude API only, the other secrets resolve as under op)
  local m; m="$(secrets_mode)"; [ "$m" = wif ] && m=op; echo "$m"
}
secret_hf(){   # [source]: the site's HF token (gated repos fetched from the grading host)
  _secret_hf_from "${1:-${SISTEER_HF_TOKEN_SOURCE:-$(secrets_hf_default_source)}}" "${SISTEER_OP_ITEM_HF:-HF}"
}
secret_hf_cluster_read(){   # [source]: the READ-ONLY token that rides hf_fetch.sh's stdin; same three sources, its own op item
  _secret_hf_from "${1:-${SISTEER_HF_TOKEN_SOURCE:-$(secrets_hf_default_source)}}" "${SISTEER_OP_ITEM_HF_CLUSTER_READ:-HF cluster read}"
}
secret_wandb(){
  if [ "$(secrets_mode)" = env ] || [ -n "${WANDB_API_KEY:-}" ]; then printf '%s' "${WANDB_API_KEY:-}"; return 0; fi
  local b; b="$(secrets_op_bin)" || return 1; "$b" "${SISTEER_OP_ITEM_WANDB:-Weights and Biases}"
}
secrets_modal_env(){   # -> `MODAL_TOKEN_ID=… MODAL_TOKEN_SECRET=…` is NEVER printed; exports them into THIS shell for a Modal call
  if [ "$(secrets_mode)" = env ]; then
    [ -n "${MODAL_TOKEN_ID:-}" ] && [ -n "${MODAL_TOKEN_SECRET:-}" ] || { echo "REFUSED: SISTEER_SECRETS_MODE=env: MODAL_TOKEN_ID and MODAL_TOKEN_SECRET must be set for the Modal stage" >&2; return 1; }
    export MODAL_TOKEN_ID MODAL_TOKEN_SECRET; return 0
  fi
  # op mode: the site's shell exports them already when it uses them; nothing to fetch here
  [ -n "${MODAL_TOKEN_ID:-}" ] || echo "note: MODAL_TOKEN_ID is not exported; \`modal\` will use its own OAuth profile" >&2
  return 0
}

# --- transport into a detached systemd --user unit ---------------------------------------------------------------------
secrets_systemd_version(){ systemd-run --version 2>/dev/null | sed -n 's/^systemd \([0-9]*\).*/\1/p' | head -1; }
secrets_unit_flags(){   # -> the extra systemd-run flags for the current mode (the CLI's path under op; the MODE itself under wif)
  case "$(secrets_mode)" in
    op)
      # the unit runs the secret-manager CLI by the absolute path the setting resolves to (a detached unit inherits nothing, and
      # its PATH may lack the CLI's directory); with the setting unset nothing is carried and the unit refuses by name
      secrets_unit_cmd_flag; return;;
    wif)
      # no credential to carry — the unit's vault federates by itself — but it reads the MODE off its own environment, and a detached
      # unit inherits nothing: without this line the unit would run under the default op and reach for a key that does not exist.
      # The mode and the ids directory are values, not secrets: `--setenv=NAME=VALUE` on every systemd, no file.
      local f="--setenv=SISTEER_SECRETS_MODE=wif" c
      if [ -n "${SISTEER_WIF_ENV_DIR:-}" ]; then
        case "$SISTEER_WIF_ENV_DIR" in *[[:space:]]*) echo "REFUSED: SISTEER_WIF_ENV_DIR must not contain whitespace (it rides systemd-run's argv into the unit)" >&2; return 1;; esac
        f="$f --setenv=SISTEER_WIF_ENV_DIR=$SISTEER_WIF_ENV_DIR"
      fi
      # federation covers the Claude API only: the unit's other secrets (OpenRouter, HF, W&B) resolve as under op, so the CLI's
      # path rides along when the site names one
      c="$(secrets_unit_cmd_flag)" || return 1
      [ -n "$c" ] && f="$f $c"
      printf '%s' "$f"; return 0;;
  esac
  local v; v="$(secrets_systemd_version)"; v="${v:-0}"
  if [ "${SISTEER_SYSTEMD_FORCE_ENVFILE:-0}" != 1 ] && [ "$v" -ge 247 ]; then
    local f="--setenv=SISTEER_SECRETS_MODE --setenv=ANTHROPIC_API_KEY"
    [ -n "${OPENROUTER_API_KEY:-}" ] && f="$f --setenv=OPENROUTER_API_KEY"
    printf '%s' "$f"
  else
    local ef; ef="$(secrets_unit_envfile)" || return 1
    printf '%s' "-p EnvironmentFile=$ef"
  fi
}
secrets_unit_cmd_flag(){   # -> `--setenv=SISTEER_SECRET_CMD=<absolute path>` when the site names the CLI, else nothing
  [ -n "${SISTEER_SECRET_CMD:-}" ] || return 0
  local b; b="$(secrets_op_bin)" || return 1
  case "$b" in *[[:space:]]*) echo "REFUSED: SISTEER_SECRET_CMD must not contain whitespace (it rides systemd-run's argv into the unit)" >&2; return 1;; esac
  printf '%s' "--setenv=SISTEER_SECRET_CMD=$b"
}
secrets_unit_envfile(){   # the systemd < 247 fallback: a 0600 file under $XDG_RUNTIME_DIR (tmpfs), never under the tree
  local dir="${XDG_RUNTIME_DIR:-}"; [ -n "$dir" ] && [ -d "$dir" ] || { echo "REFUSED: no XDG_RUNTIME_DIR for the EnvironmentFile fallback" >&2; return 1; }
  local f; f="$(mktemp "$dir/sisteer-unit-env.XXXXXX")" || return 1
  chmod 0600 "$f"
  { echo "SISTEER_SECRETS_MODE=env"; echo "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY:?SISTEER_SECRETS_MODE=env: ANTHROPIC_API_KEY must be set}";
    [ -n "${OPENROUTER_API_KEY:-}" ] && echo "OPENROUTER_API_KEY=$OPENROUTER_API_KEY"; } > "$f"
  echo "$f"
}
secrets_unit_envfile_cleanup(){ case "${1:-}" in "${XDG_RUNTIME_DIR:-/nonexistent}"/sisteer-unit-env.*) rm -f "$1";; esac; }
secrets_unit_flags_cleanup(){   # <the string secrets_unit_flags printed>: unlink the EnvironmentFile it named, once the unit has started
  # systemd reads EnvironmentFile= while assembling the service's environment at start (systemd-run returns after the start job), so
  # the file has done its work by the time the launcher's next line runs; a no-op for the name-only import flags and under op mode
  local tok; for tok in ${1:-}; do case "$tok" in EnvironmentFile=*) secrets_unit_envfile_cleanup "${tok#EnvironmentFile=}";; esac; done
}
secrets_unit_key_line(){   # <org> -> the ONE bash line a GENERATED unit script uses to obtain its key (the env-mode rule); under wif it only SCRUBS
  local org="${1:-lowprio}" var
  if [ "$(secrets_mode)" = wif ]; then
    # no key to obtain: sisteer.vault authenticates by federation inside the unit; EVERY inherited ANTHROPIC_* name is dropped so none
    # shadows it (bash's ${!PREFIX@} lists the NAMES, no value read; the SDK reads ANTHROPIC_CUSTOM_HEADERS whatever credential it
    # is given, so a list of names is not the rule -- the prefix is; sisteer.vault.WIF_SHELL_SCRUB is this line's spelling)
    printf '%s' 'unset "${!ANTHROPIC_@}"'; return 0
  fi
  if [ "$(secrets_mode)" = env ]; then
    printf '%s' ': "${ANTHROPIC_API_KEY:?SISTEER_SECRETS_MODE=env: ANTHROPIC_API_KEY must be set in the unit environment (one key serves every org role)}"; unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN'
  else
    var="$(secrets_env_var_for_org "$org")" || return 1
    printf 'export %s="$("${SISTEER_SECRET_CMD:?SISTEER_SECRET_CMD must name the secret-manager CLI in the unit environment}" %s)"; unset ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN' "$var" "$org"
  fi
}
