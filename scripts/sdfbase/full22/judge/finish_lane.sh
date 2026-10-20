#!/bin/bash
# Generic lane finisher (detached via systemd-run --user): wait until the judge org serves
# sonnet-5 (one canary-shaped probe per minute through the vault's MODE-AWARE client -- op: the pool's
# key resolved in-process by sisteer.vault, never in this shell's environment, never printed; env: the
# ONE ANTHROPIC_API_KEY; wif: federated, keyless, the secret-manager CLI never consulted -- the base URL
# pinned to the API host in every mode), launch the judge for a set of cells, then run the whole
# post-pipeline unattended: re-emit -> cap conditioning -> paired vs none -> report -> append-verified
# push -> read-back -> cache merge. Sentinels: $LOGDIR/LANE_DONE or LANE_FAILED.
#
# The probe WAITS only for the org (a 429 / 529 / 5xx / transport error / a model not yet served). A
# credential the API REJECTS -- 401 / 403 on the request, or a federation identity the token exchange
# refuses (sisteer.vault.credential_rejection) -- stops the lane at the FIRST probe (a rejected
# credential is never retried), as does a pool the vault cannot serve (an absent / empty ids file, the
# secret-manager CLI missing) or a probe that cannot run at all. Earlier, under
# SISTEER_SECRETS_MODE=wif this probe fetched the legacy key through the secret-manager CLI and built a
# plain keyed client, so it re-tried a rejected credential once a minute for F22_GIVE_UP_S (24 h), logged
# nothing when the probe printed nothing, and never reached the federated launcher.
#
#   finish_lane.sh TAG TARGETS MATCH LOGIN_HOST REMOTE_ROOT CACHE [extra driver args...]
#   env: F22_JUDGE_ORG (lowprio default), F22_GRADE_ROOT, F22_GIVE_UP_S (default 86400), F22_PROBE_S (default 60),
#        F22_PY (the probe's interpreter; default <worktree>/.venv/bin/python), SISTEER_SECRETS_MODE (op default | env |
#        wif -- a detached unit inherits nothing, so the systemd-run line that starts THIS script carries the mode:
#        `systemd-run --user $(secrets_unit_flags) ...` with reproduce/lib/secrets.sh sourced, i.e. --setenv=SISTEER_SECRETS_MODE=wif)
set -u
TAG=$1; TARGETS=$2; MATCH=$3; LOGIN=$4; RROOT=$5; CACHE=$6; shift 6
J=$(cd "$(dirname "$0")" && pwd); WT=$(cd "$J/../../../.." && pwd)
G=${F22_GRADE_ROOT:-$(PYTHONPATH="$WT/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m sisteer.site expand '$GRADE/sdfbase_full22')}; G="${G/#\~/$HOME}"; [ -n "$G" ] || { echo "finish_lane.sh: set SISTEER_GRADE_DIR (the grade root) or F22_GRADE_ROOT" >&2; exit 2; }
ORG=${F22_JUDGE_ORG:-lowprio}
PY=${F22_PY:-$WT/.venv/bin/python}
MODE=${SISTEER_SECRETS_MODE:-op}
LOGDIR=$G/logs/$TAG; mkdir -p "$LOGDIR"
export F22_LOGIN_HOST=$LOGIN F22_CONTROLS_REMOTE=$RROOT PYTHONPATH=src:scripts HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd "$WT"
fail() { echo "[$TAG] FAILED $(date -u +%FT%TZ): $*"; echo "$*" > "$LOGDIR/LANE_FAILED"; exit 1; }
rm -f "$LOGDIR/LANE_FAILED" "$LOGDIR/LANE_DONE"
case "$MODE" in op|env|wif) ;; *) fail "SISTEER_SECRETS_MODE=$MODE: must be op, env or wif (nothing probed, nothing launched)";; esac
# wif: no key exists, and EVERY inherited ANTHROPIC_* name is dropped (bash's ${!PREFIX@}: the names, no value read; the SDK
# reads ANTHROPIC_CUSTOM_HEADERS whatever credential it is given, so a static header there would skip the probe's token exchange) so none
# shadows federation in the probe (launch_judge.sh's unit scrubs its own). Every mode: the shell's ANTHROPIC_BASE_URL reaches no
# child; the vault pins the API host.
[ "$MODE" = wif ] && unset "${!ANTHROPIC_@}"
unset ANTHROPIC_BASE_URL
# the vault runs the secret-manager CLI SISTEER_SECRET_CMD names (secrets_unit_flags carries its path); $HOME/.local/bin stays ahead on PATH
export PATH="$HOME/.local/bin:$PATH"
echo "[$TAG] start $(date -u +%FT%TZ) org=$ORG mode=$MODE targets=$TARGETS py=$PY"
probe() {   # ONE token on stdout: OK | CREDENTIAL_REJECTED:<type>:<status>[:<api type>] | VAULT_REFUSED:<type>:<why> | <type>:<status>
  "$PY" - "$ORG" <<'PY' 2>>"$LOGDIR/probe.err"
import sys
from sisteer import vault
from sisteer.capeval import judging
org = sys.argv[1]
try:
    # the vault's mode-aware client: op / env keyed in-process, wif federated and keyless; the base URL pinned; no retry (the loop is the retry)
    c = vault.anthropic_client(org, max_retries=0)
except vault.VaultSecretError as e:      # the pool cannot be served at all (an absent / empty ids file, the secret-manager CLI unset or missing): nothing to wait for
    print("VAULT_REFUSED:%s:%s" % (type(e).__name__, " ".join(str(e).split())))
    sys.exit(0)
try:
    c.messages.create(model=judging.JUDGE_MODEL, max_tokens=judging.CANARY_MAX_TOKENS, thinking=dict(judging.THINKING),
                      messages=[{"role": "user", "content": judging.CANARY_TEXT}])
    print("OK")
except Exception as e:  # noqa: BLE001 -- classified: a rejected credential stops the lane (D6); anything else is "not yet"
    rej = vault.credential_rejection(e)
    if rej is not None:
        print("CREDENTIAL_REJECTED:%s:%s%s" % (rej.error_type, rej.status, ":" + rej.api_error_type if rej.api_error_type else ""))
    else:
        print("%s:%s" % (type(e).__name__, getattr(e, "status_code", None)))
PY
}
ok=0; last=""; n=0; t0=$(date +%s)
while :; do
  r=$(probe); n=$((n+1))
  [ -n "$r" ] || r="PROBE_ERROR:no verdict from the probe (interpreter $PY; its stderr is in $LOGDIR/probe.err)"
  [ "$r" != "$last" ] && { echo "[$TAG probe $(date -u +%H:%M:%SZ)] $ORG/$MODE sonnet-5: $r (probe #$n, $(( $(date +%s) - t0 ))s)"; last=$r; }
  case "$r" in
    CREDENTIAL_REJECTED:*) fail "the API rejected the $ORG pool's credential under SISTEER_SECRETS_MODE=$MODE at probe #$n: $r -- stopping at the first 401 / 403 (D6), nothing launched; fix the credential (op: the pool's vault item -- or, if the org has moved to federation, start this unit under SISTEER_SECRETS_MODE=wif; env: ANTHROPIC_API_KEY; wif: the pool's ids file / the VM identity) and re-run";;
    VAULT_REFUSED:*|PROBE_ERROR:*) fail "the probe cannot build the $ORG pool's client under SISTEER_SECRETS_MODE=$MODE (probe #$n): $r -- nothing to wait for, nothing launched";;
  esac
  if [ "$r" = "OK" ]; then ok=$((ok+1)); else ok=0; fi
  [ $ok -ge 2 ] && break
  [ $(( $(date +%s) - t0 )) -ge ${F22_GIVE_UP_S:-86400} ] && fail "gave up waiting for $ORG after $(( $(date +%s) - t0 ))s (last probe: $r)"
  sleep "${F22_PROBE_S:-60}"
done
[ -n "$MATCH" ] && MATCHARG=(--match "$MATCH") || MATCHARG=()
out=$(CAPEVAL_JUDGE_CACHE=$CACHE F22_JUDGE_ORG=$ORG "$J/launch_judge.sh" --targets "$TARGETS" "${MATCHARG[@]}" --sentinel-dir "$LOGDIR" "$@") || fail "launch: $out"
echo "[$TAG] $out"
stamp=$(sed -n 's/.*stamp=\([0-9-]*\).*/\1/p' <<<"$out"); log=$G/logs/judge-$stamp.log
until [ -f "$G/logs/EXIT-$stamp" ]; do sleep 10; done
code=$(cat "$G/logs/EXIT-$stamp"); grep -E "DONE |REFUSED-CEILING|FAILED|transport_refusal|canary|JUDGE_" "$log" | cut -c1-220
[ "$code" = "0" ] || fail "judge run $stamp exited $code (see $log)"
started=$(date -u -d "@$(date -d "${stamp:0:8} ${stamp:9:2}:${stamp:11:2}:${stamp:13:2}" +%s)" +%FT%TZ)
CAPEVAL_JUDGE_CACHE=$CACHE .venv/bin/python scripts/served_capability_judge.py --controls-root "$G/controls" --targets "$TARGETS" "${MATCHARG[@]}" --reemit-only --run-started-at "$started" --sentinel-dir "$LOGDIR/reemit" 2>&1 | grep -E "re-emit|JUDGE_" || fail reemit
# cap_condition.py is sha-pinned, so its root is passed here (its own default is one site's grade root), and its exit is
# read from PIPESTATUS: a crash behind `| tail` must fail the lane, never leave at_cap unset for paired_vs_none
.venv/bin/python "$J/cap_condition.py" --root "$G/controls" --targets "$TARGETS" "${MATCHARG[@]}" 2>&1 | tail -6; [ "${PIPESTATUS[0]}" = 0 ] || fail cap_condition
python3 "$J/paired_vs_none.py" --targets "$TARGETS" 2>&1 | grep -E "prompted|^target" | cut -c1-150 || fail paired
python3 "$J/report_table.py" --targets "$TARGETS" --json "$G/report_rows_$TAG.json" 2>&1 | tail -12 || fail report
python3 "$J/pushback_judged.py" --targets "$TARGETS" 2>&1 | grep -vE "^IN-SYNC" | tail -6 || fail pushback
for t in ${TARGETS//,/ }; do rsync -t --no-owner --no-group --no-perms "$G/controls/$t/judge_paired_vs_none.json" "$LOGIN:$RROOT/$t/" || fail "paired push $t"; done
ssh "$LOGIN" "F22_CONTROLS_REMOTE=$RROOT python3 -" < "$J/verify_remote_stores.py" 2>&1 | grep -E "$MATCH|TOTAL" | cut -c1-200 || fail verify
python3 "$J/merge_cache_back.py" --private "$CACHE" 2>&1 | tail -1 || fail merge
echo "stamp=$stamp started=$started org=$ORG" > "$LOGDIR/LANE_DONE"
echo "[$TAG] LANE_DONE $(date -u +%FT%TZ) stamp=$stamp"
