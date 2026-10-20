#!/bin/bash
# Detached launcher for the full22 served-controls over-refusal judge.
# The API key (op / env modes) is resolved from the vault INSIDE the unit's shell (never in the
# unit's Environment=, never on disk, never in argv); the log carries no key. Under
# SISTEER_SECRETS_MODE=wif there is no key: the unit receives the MODE (lib/secrets.sh
# secrets_unit_flags) and its vault federates; the secret-manager CLI is never consulted.
# Extra CLI args pass through positionally ("$@"), so quoting survives.
set -euo pipefail
WT=$(cd "$(dirname "$0")/../../../.." && pwd)   # the repo checkout this script lives in
ROOT=${F22_GRADE_ROOT:-$(PYTHONPATH="$WT/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m sisteer.site expand '$GRADE/sdfbase_full22' || :)}; ROOT="${ROOT/#\~/$HOME}"
[ -n "$ROOT" ] || { echo "launch_judge.sh: set SISTEER_GRADE_DIR (the grade root) or F22_GRADE_ROOT" >&2; exit 2; }
STAMP=$(date +%Y%m%d-%H%M%S)
# Which vault item / org the run bills: lowprio by default (the standing rule); highprio ONLY
# when it is authorised for this lane — set F22_JUDGE_ORG=highprio explicitly.
ORG=${F22_JUDGE_ORG:-lowprio}
case "$ORG" in lowprio|highprio) ;; *) echo "F22_JUDGE_ORG must be lowprio or highprio" >&2; exit 2;; esac
UNIT=f22-orjudge-$STAMP
LOG=$ROOT/logs/judge-$STAMP.log
ln -sfn "$LOG" "$ROOT/logs/judge-latest.log"
# env-mode secrets (paper-companion-repo, docs/secrets.md): under SISTEER_SECRETS_MODE=env the key reaches the detached unit by
# NAME-ONLY --setenv import (lib/secrets.sh secrets_unit_flags; nothing on argv, no file, nothing journaled) and the unit checks
# the ONE ANTHROPIC_API_KEY instead of fetching the org's item; under op mode the flags are empty and the line is unchanged.
# wif mode (round 2 on 4574e103c): the flags carry `--setenv=SISTEER_SECRETS_MODE=wif` (and the ids directory when the site
# overrides it) — a detached unit inherits nothing, and a unit without the mode ran its vault under op and fetched a key that does
# not exist; the unit's own line below then only SCRUBS the inherited ANTHROPIC_* so nothing shadows federation. A non-op mode
# with the library unreadable is refused here rather than launched under the stub's silence.
[ -r "$WT/reproduce/lib/secrets.sh" ] && source "$WT/reproduce/lib/secrets.sh"
if [ "${SISTEER_SECRETS_MODE:-op}" != op ] && ! type secrets_unit_flags >/dev/null 2>&1; then
  echo "REFUSED: SISTEER_SECRETS_MODE=$SISTEER_SECRETS_MODE but $WT/reproduce/lib/secrets.sh is not readable: the detached unit inherits nothing and the library is what carries the mode (and, under env, the key) into it" >&2; exit 2
fi
type secrets_unit_flags >/dev/null 2>&1 || secrets_unit_flags(){ :; }
type secrets_unit_flags_cleanup >/dev/null 2>&1 || secrets_unit_flags_cleanup(){ :; }
UFLAGS="$(secrets_unit_flags)"
# the systemd < 247 EnvironmentFile fallback is unlinked once the unit has started, whether or not systemd-run succeeded
trap 'secrets_unit_flags_cleanup "$UFLAGS"' EXIT
# shellcheck disable=SC2086
systemd-run --user --unit="$UNIT" --collect --quiet $UFLAGS \
  -p WorkingDirectory="$WT" \
  -E PYTHONPATH=src:scripts -E HF_HUB_OFFLINE=1 -E PYTHONUNBUFFERED=1 \
  -E F22_LOG="$LOG" -E F22_ROOT="$ROOT" -E F22_UNIT="$UNIT" -E F22_STAMP="$STAMP" -E F22_ORG="$ORG" \
  ${CAPEVAL_JUDGE_CACHE:+-E CAPEVAL_JUDGE_CACHE="$CAPEVAL_JUDGE_CACHE"} \
  /bin/bash -c 'if [ "${SISTEER_SECRETS_MODE:-op}" = env ]; then : "${ANTHROPIC_API_KEY:?SISTEER_SECRETS_MODE=env: ANTHROPIC_API_KEY must be set in the unit environment}"; elif [ "${SISTEER_SECRETS_MODE:-op}" = wif ]; then unset "${!ANTHROPIC_@}"; else export ANTHROPIC_API_KEY="$("${SISTEER_SECRET_CMD:?SISTEER_SECRET_CMD must name the secret-manager CLI in the unit environment}" "$F22_ORG")"; fi; unset ANTHROPIC_BASE_URL;
    { echo "[launcher] unit=$F22_UNIT org=$F22_ORG start $(date -u +%FT%TZ) args: $(printf "%q " "$@")";
      .venv/bin/python scripts/served_capability_judge.py --controls-root "$F22_ROOT/controls" --org "$F22_ORG" "$@";
      rc=$?; echo "[launcher] EXIT=$rc $(date -u +%FT%TZ)"; echo "$rc" > "$F22_ROOT/logs/EXIT-$F22_STAMP"; } >> "$F22_LOG" 2>&1' _ "$@"
echo "unit=$UNIT org=$ORG log=$LOG stamp=$STAMP"
