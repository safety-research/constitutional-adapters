#!/bin/bash
#SBATCH --job-name=f22-att
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=0-20:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# full22: ONE steered TARGET SERVER (1 GPU) + its multi-turn attack DRIVERS (CPU only) for one
# (target, view) config: scripts/sdfbase/mt_serving_up.py --mode attacks serves EVERY cell of the config
# as a virtual model, the lens pinned to STACKED reductions and REFUSED unless the workers attest it;
# one jailbreak_run.py generate --target-engine served process per (attack, cell, shard), all concurrent
# (the previous wave's one target GPU carried five cells at 13% KV and was bound by its 45 drivers —
# hence more drivers per server), each driver round-robined over the LIVE members of the shared
# attacker pool.
#
# THE CANONICAL 4,096-CAP INSTRUMENT: the target's per-reply
# cap is the config's attacks.target_max_new_tokens (4,096 from the roster), the target SAMPLES at the
# served model's VENDOR sampler preset (jailbreak_run's default for an unspecified attacks.target_decode
# since 2026-09-16; this script's explicit --multiturn-target-decode vendor -- F22_MT_TARGET_DECODE below,
# added 2026-09-10 -- sends the yaml row as it stands. The two differ for ONE model: nemotron's yaml row
# carries top_p 1.0, a declared departure from its shipped file's 0.95, and the DEFAULT path pins nemotron
# to the shipped file (the six nemotron 4,096 multi-turn roots record top_p 0.95,
# so those canonical legs decoded at the generation config, not this row -- pass F22_MT_TARGET_DECODE=auto
# to take the default; `vendor` keeps sending 1.0 and is recorded as such). The preset rides on every
# request, Qwen3.5's presence_penalty 1.5 included; jailbreak_run's 2026-09-11 -> 2026-09-16 default, the
# server's generation config with no sampling key on the request, stays one config key away), and the SERVING
# WINDOW IS EXPLICIT PER ATTACK — geometry.multiturn_window: crescendo 49,152 / pair 16,384, the
# measured maxima on complete stores rounded up (41,433 / 12,825), NEVER the table's `multiturn_attacks`
# 8,704 (a p99 under the old caps: 129 distinct clamped max_tokens and 3 killed goals on one PAIR cell).
# One job therefore serves ONE attack (submit_full22.sh emits crescendo and pair jobs separately); a job
# naming several attacks serves the largest of their windows. ActorAttack has NO measured 4,096-cap window
# and is refused unless [max_model_len] is passed explicitly — available, never the default.
# qwen-seedrep-native-battery: two OPTIONAL scalars, forwarded to every driver only when SET (unset = today's
# argv byte for byte; sbatch's implicit export carries scalars, never --export=):
#   F22_MT_TARGET_SEED_BASE  -> --multiturn-target-seed-base <s>    the per-(goal, turn) seed BASE = the served view's
#                               LITERAL training seed (1 / 2 / 3), replacing the preset file's campaign base 20260909
#   F22_MT_TARGET_POLICY_REF -> --multiturn-target-policy-ref <ref>  ASSERT the decode-policy registry row (must equal the
#                               derived `<model_id>|multi_turn|off|vendor_preset@v1`)
#   EITHER of the two puts the driver on the REGISTRY path: the six numbers sent are that row's (PolicyFile.by_ref, exact
#   ref, not superseded, all six keys sent), the preset yaml is not read, decode_policy is recorded, and a seeded run
#   must state its base -- so the wave passes BOTH.
# Usage (positional only — never --export=):
#   sbatch -J f22-att-<key>-<view>-<attack> scripts/sdfbase/full22/target_attacks.sbatch \
#       <key> <view> <attacks> <wave_sha> [nshards=9] [shards=0,1,...,8] [cells=all] \
#       [max_model_len=""] [client_timeout_s=3600] [proxy_upstream_timeout_s=3600]
#     key      gemma4e4b | qwen35_4b | nemotron3nano4b | glm46vflash | llama8b (geometry.MODEL_KEYS)
#     view     main | it | ...   (configs/sdfbase_full22_<key>[_<view>].yaml)
#     attacks  comma list of crescendo,pair (actorattack only with an explicit max_model_len)
#     nshards/shards  the FULL 259-goal confirmatory pool = --nshards 9, shards 0-8 (1-goal smoke: 259 "0")
#     max_model_len   "" = geometry.multiturn_window(attacks); a number overrides it (recorded by the standup)
#     client_timeout_s / proxy_upstream_timeout_s  the two 600 s defaults a 4,096-token reply under load
#                     outlived; 3600 each by default, recorded (attacker_engine.json / endpoint.json)
# Completeness = UNIQUE goal_ids in progress.jsonl per (attack, cell), never sacct. Grading is the grading-host step.
set -uo pipefail
KEY="${1:?key}"; VIEW="${2:?view}"; ATTACKS="${3:?attacks}"; WAVE_SHA="${4:?wave sha}"
NSHARDS="${5:-9}"; SHARDS="${6:-0,1,2,3,4,5,6,7,8}"; CELLS_ARG="${7:-}"
# A job that serves a SUBSET of the view's cells registers that subset — f22_registry_register narrows the
# overlay to F22_REGISTRY_CELLS (runs inside f22_resolve_config below), so the export sits here, before f22_init; the plan sets both
[ -n "$CELLS_ARG" ] && export F22_REGISTRY_CELLS="${F22_REGISTRY_CELLS:-$CELLS_ARG}"
# The run key is the measurement's — `sdfbase_full22-att-<attack>-<key>[_<view>]-sh<shards>of<nshards>`
# (common.sh f22_registry_producer_key; the shard TOTAL is in the key because jailbreak_run.py selects goals[shard::nshards] — a one-goal smoke
# (nshards 259, shard 0) and a confirmatory run (nshards 9, shard 0) are different data, so they are different runs;
# inside f22_resolve_config below); the shard group stays in the key, so the two groups of one 9-shard store remain two runs (the closer closes the
# store whole; collapsing the groups into one run is a closer-semantics change this key does not make)
export F22_REGISTRY_KEY_SEGMENTS="${F22_REGISTRY_KEY_SEGMENTS:-sh${SHARDS//,/_}of${NSHARDS}}"
GEOM_L="${8:-}"; CLIENT_TIMEOUT_S="${9:-3600}"; PROXY_UPSTREAM_TIMEOUT_S="${10:-3600}"
{ [ -n "${SISTEER_SHARED_ROOT:-}" ] && source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA/scripts/sdfbase/full22/common.sh" 2>/dev/null; } \
  || source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit
F22_LAUNCH_MANIFEST=$(f22_full22_manifest "$KEY" "$VIEW")   # the launch's cells for the kind-aware serving-path default (2026-09-15): the (KEY, VIEW) campaign manifest
# RESUME mode (F22_RESUME_FROM_JOB=<job>, the funnel's `submit --resume-from-job`): this job REUSES the original job's registration
# identity — pending/<from>.json copied to its own — so f22_init's `running` emission and f22_registry_register (inside f22_resolve_config)
# carry it instead of minting new runs (the copy sits BEFORE f22_init, so the state at start and any
# early death are filed under the canonical runs), the drivers resume the SAME stores (their path is job-independent:
# $WAVE_ROOT/transcripts_results/<private_run>/<attack>/<cell>/shard*; an errored goal is NOT covered — static_common.load_covered_goals),
# and the closer files the canonical runs. The copy keeps the original's job_id; a resume job that already carries an identity, or whose
# original has none, refuses by name. The shared root is the job environment's SISTEER_SHARED_ROOT (the front door exports it from the
# cluster's site block).
if [ -n "${F22_RESUME_FROM_JOB:-}" ]; then
  _rshared="${SISTEER_SHARED_ROOT:-}"
  [ -n "$_rshared" ] || f22_fatal "F22_RESUME_FROM_JOB=$F22_RESUME_FROM_JOB: no shared root (set SISTEER_SHARED_ROOT, the shared root of the cluster; the front door exports it from the site config) — the registration identity cannot be found"
  _rsrc="$_rshared/registry/pending/$F22_RESUME_FROM_JOB.json"; _rdst="$_rshared/registry/pending/${SLURM_JOB_ID:?}.json"
  [ -f "$_rsrc" ] || f22_fatal "F22_RESUME_FROM_JOB=$F22_RESUME_FROM_JOB: no registration identity at $_rsrc — a resume reuses the original job's runs or does not run"
  [ -e "$_rdst" ] && f22_fatal "F22_RESUME_FROM_JOB=$F22_RESUME_FROM_JOB: $_rdst already exists — this job already carries an identity; refusing to overwrite it"
  cp "$_rsrc" "$_rdst" || f22_fatal "F22_RESUME_FROM_JOB=$F22_RESUME_FROM_JOB: cannot copy $_rsrc -> $_rdst"
  f22_log "RESUME of job $F22_RESUME_FROM_JOB (D590): registration identity reused ($_rdst <- $_rsrc; no new runs) before the job's own state emission; the drivers resume the stores and retry errored goals"
fi
f22_init "$WAVE_SHA"
f22_resolve_config "$KEY" "$VIEW"
f22_stop_requested && { f22_log "wave STOP present — not starting"; exit 0; }
# a view without the multi-turn leg (the llama8b fullconst view) is refused up front (roster_view.py has_leg)
python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" has_leg attacks --roster "$F22_ROSTER" || f22_fatal "view $KEY/$VIEW has no attacks (multi-turn) leg — this job must not be submitted for it"
for kv in "max_model_len=$GEOM_L"; do v="${kv#*=}"; [ -z "$v" ] && continue
  case "$v" in *[!0-9]*) f22_fatal "${kv%%=*} '$v' is not an integer";; esac; [ "$v" -ge 1 ] || f22_fatal "${kv%%=*} must be >= 1"; done
for kv in "client_timeout_s=$CLIENT_TIMEOUT_S" "proxy_upstream_timeout_s=$PROXY_UPSTREAM_TIMEOUT_S"; do v="${kv#*=}"
  case "$v" in ''|*[!0-9.]*|.|*.*.*) f22_fatal "${kv%%=*} '$v' is not a number";; esac; done
GROUPTAG=$(echo "$ATTACKS" | tr ',' '-')
SERVE_DIR="$WAVE_ROOT/serving/attacks_${TAG}_${GROUPTAG}_${SLURM_JOB_ID}"; PRIVATE_ROOT="$WAVE_ROOT/transcripts_results"
mkdir -p "$SERVE_DIR/drivers" "$PRIVATE_ROOT"; chmod 700 "$PRIVATE_ROOT"
f22_driver_venv; f22_serving_venv; f22_bank_view "$MPY"
# a prompted-only view (llama8b/shortconst, 2026-09-12: the short constitution SI runs the multi-turn leg too): the load check's probe
# rides the PROMPTED route with the SI prefix at the family's cap, so a window the prefix + the p99 prompt + the
# cap cannot fit refuses in the preflight, before the GPU is allocated
if f22_view_prompted_only; then export F22_LOAD_CHECK_SYSTEM_SI="${F22_LOAD_CHECK_SYSTEM_SI:-$(f22_prompted_si)}"; fi
# THE TARGET'S CAP (the config's attacks.target_max_new_tokens; the driver reads the same key) and THE
# WINDOW: explicit per attack from the measured maxima (geometry.multiturn_window), or the positional.
# The standup is told the drivers' cap (--driver-max-tokens) so a table window can never serve it silently.
CAP_TOKENS=$("$MPY" -c 'import sys,yaml;print(int(yaml.safe_load(open(sys.argv[1]))["attacks"]["target_max_new_tokens"]))' "$CFG") \
  || f22_fatal "config $CFG has no attacks.target_max_new_tokens"
if [ -z "$GEOM_L" ]; then
  GEOM_L=$(cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src" "$MPY" -c 'import sys; from sisteer.capeval.serving.geometry import multiturn_window; print(multiturn_window(sys.argv[1]))' "$ATTACKS") \
    || f22_fatal "no measured ${CAP_TOKENS}-cap serving window for attacks=$ATTACKS (actorattack?): pass max_model_len explicitly (positional 8) and record why"
  WINDOW_SOURCE="geometry.multiturn_window"
else
  WINDOW_SOURCE="positional"
fi
# the standup flags are REBUILT per window branch below on top of the inherited launch-time hook (common.sh
# f22_standup_extra_reset, 2026-09-16) — never assigned from scratch, which clobbered an inherited scalar
# F22_LOG_REQUESTS=1 turns vLLM request logging ON for this leg's standups (2026-09-18). It was
# being PASSED on this lane's submissions and consumed by nothing -- the real switch is the
# standup flag --enable-log-requests, which only target_sampled.sbatch set. Opt-in, so the
# default argv stays byte-identical and no running lane changes; the flag rides the ARGV HOOK
# helper (never a bare assignment) so an inherited scalar F22_STANDUP_EXTRA still reaches the standup.
F22_LOG_REQ_FLAG=(); [ "${F22_LOG_REQUESTS:-0}" = 1 ] && F22_LOG_REQ_FLAG=(--enable-log-requests)
f22_standup_extra_reset --upstream-timeout-s "$PROXY_UPSTREAM_TIMEOUT_S" --driver-max-tokens "$CAP_TOKENS" ${F22_LOG_REQ_FLAG[@]+"${F22_LOG_REQ_FLAG[@]}"}
f22_log "attacks=$ATTACKS cap=$CAP_TOKENS window=$GEOM_L ($WINDOW_SOURCE; S sized from $KEY's budget) client_timeout_s=$CLIENT_TIMEOUT_S proxy_upstream_timeout_s=$PROXY_UPSTREAM_TIMEOUT_S"
if [ -n "$CELLS_ARG" ]; then IFS=',' read -ra CELLS <<< "$CELLS_ARG"; else CELLS=("${ALL_CELLS[@]}"); fi
TARGET_PID=""; declare -a DRIVER_PIDS=(); RC=0
cleanup(){
  touch "$SERVE_DIR/STOP" 2>/dev/null || true
  for p in "${DRIVER_PIDS[@]:-}"; do [ -n "$p" ] && kill -TERM "$p" 2>/dev/null || true; done
  [ -n "${TARGET_PID:-}" ] && { for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; }
  f22_result "attacks $TAG $GROUPTAG rc=${RC}"
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
# Phase A: the served target. NOT srun-wrapped, deliberately (the mt1 precedent): the standup, the vLLM child
# and the CPU-only drivers below are children of THIS batch step, killed by the cleanup trap; an srun step
# per server would need --overlap steps sharing one GPU, untested on these clusters (README)
TARGET_GPU="$CUDA_VISIBLE_DEVICES"
# THE MULTI-TURN WINDOW AND CAP — two readers, one config, told apart by the config's keys (the 2026-09-11
# reconciliation of the sdfbase-llama8b wave branch with main):
#  (a) a config that names its own multi-turn target cap (attacks.multiturn_target_max_new_tokens — an int, or a
#      per-family map {crescendo, pair, actorattack} -> cap; the roster's caps table, the sdfbase-llama8b path) serves
#      each family this job runs at the window DERIVED from THAT family's cap (geometry.multiturn_window(cap) = the
#      table's p99 rendered prompt + cap on the table's grid, S = the model's budget // L through the standup's strict
#      identity — never a literal here) or at the family's PINNED first-pass window (attacks.multiturn_max_model_len,
#      an int or the per-family map — the roster's caps.windows; ActorAttack's 49,152 x 9
#      rides this path, the only measured-or-pinned ActorAttack window there is) honoured over the derived value by
#      geometry.multiturn_window(cap, pin), which refuses a pin below the derived minimum (p99 prompt + cap) or off
#      the table's grid. A job over SEVERAL families (the printer's `-all` shape below 5 cells) stands up ONE server at
#      the MAX of its families' windows; every driver still resolves ITS family's cap (jailbreak_run), and the load
#      check probes at the max cap over the families served (F22_LOAD_CHECK_ATTACK -> load_check.py --attack). A
#      family the map does not name REFUSES here, before any GPU minute.
#  (b) every other config: THE CANONICAL 4,096-CAP INSTRUMENT — the cap
#      is the config's attacks.target_max_new_tokens (4,096 from the roster; the driver reads the same key) and the
#      window is EXPLICIT per attack from the measured maxima (geometry.multiturn_window(attacks): crescendo 49,152 /
#      pair 16,384), or the positional [max_model_len]; ActorAttack has NO measured 4,096-cap window on this path and
#      is refused unless the positional is passed. The standup is told the drivers' cap (--driver-max-tokens) so a
#      table window can never serve it silently.
CAP_TOKENS=$("$MPY" -c 'import sys,yaml;print(int(yaml.safe_load(open(sys.argv[1]))["attacks"]["target_max_new_tokens"]))' "$CFG") \
  || f22_fatal "config $CFG has no attacks.target_max_new_tokens"
MT_RES=$(cd "$REPO_DIR" && "$SPY" - "$CFG" "$ATTACKS" <<'PY'
import sys, yaml
sys.path.insert(0, "src")
from sisteer.capeval.serving.geometry import multiturn_window
a = yaml.safe_load(open(sys.argv[1])).get("attacks") or {}
fams = [f for f in sys.argv[2].split(",") if f]
cap = a.get("multiturn_target_max_new_tokens")
if cap is None:
    print("NONE"); sys.exit(0)            # no per-family key: path (b), the canonical explicit-per-attack window
pin = a.get("multiturn_max_model_len")
Ls = {}
for f in fams:
    c = cap.get(f) if isinstance(cap, dict) else cap
    if c is None:
        sys.exit(f"attacks.multiturn_target_max_new_tokens names no cap for {f!r} (families: {sorted(cap)}) — a family the config does not name cannot run")
    p = pin.get(f) if isinstance(pin, dict) else pin
    Ls[f] = multiturn_window(int(c), int(p) if p else None)
    print(f"FAMILY {f} {int(c)} {Ls[f]} {'PINNED' if p else 'derived'}")
print(f"MAX {max(Ls.values())}")
PY
) || f22_fatal "cannot resolve the multi-turn window(s) for attacks=$ATTACKS from $CFG (attacks.multiturn_target_max_new_tokens / multiturn_max_model_len; geometry.multiturn_window)"
MT_L=""; DRIVER_CAP_FLAGS=()
if [ "$MT_RES" != NONE ]; then
  # path (a): the per-family caps table; the driver resolves ITS family's cap from the map (no --target-max-new-tokens,
  # which would flatten a combined job's families to one cap) and keeps the raised client timeout
  while read -r kind fam fcap fL how; do
    [ "$kind" = FAMILY ] || continue
    f22_log "multi-turn $fam: target cap $fcap (attacks.multiturn_target_max_new_tokens) -> serving window $fL ($([ "$how" = PINNED ] && echo "PINNED by attacks.multiturn_max_model_len" || echo "derived, geometry.multiturn_window"); S from the model budget at L)"
  done <<< "$MT_RES"
  MT_L=$(printf '%s\n' "$MT_RES" | awk '$1=="MAX"{print $2}')
  case "$ATTACKS" in *,*) f22_log "ONE standup for $ATTACKS: serving at the MAX of the families' windows, $MT_L (every driver resolves its own family's cap; the load check probes at the max cap over the families)";; esac
  [ -z "$GEOM_L" ] || f22_log "NOTE: max_model_len=$GEOM_L (positional 8) is ignored on the per-family caps path — the roster's pins govern ($MT_L)"
  export F22_LOAD_CHECK_ATTACK="$ATTACKS"   # the load check's probe: the family's cap (the max over a combined job's families)
  f22_standup_extra_reset --upstream-timeout-s "$PROXY_UPSTREAM_TIMEOUT_S" ${F22_LOG_REQ_FLAG[@]+"${F22_LOG_REQ_FLAG[@]}"}
  DRIVER_CAP_FLAGS=(--target-timeout-s "$CLIENT_TIMEOUT_S")
  f22_log "attacks=$ATTACKS caps per family (attacks.multiturn_target_max_new_tokens) window=$MT_L (max over the families; S sized from $KEY's budget) client_timeout_s=$CLIENT_TIMEOUT_S proxy_upstream_timeout_s=$PROXY_UPSTREAM_TIMEOUT_S"
else
  # path (b): the canonical explicit-per-attack window
  if [ -z "$GEOM_L" ]; then
    GEOM_L=$(cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src" "$MPY" -c 'import sys; from sisteer.capeval.serving.geometry import multiturn_window; print(multiturn_window(sys.argv[1]))' "$ATTACKS") \
      || f22_fatal "no measured ${CAP_TOKENS}-cap serving window for attacks=$ATTACKS (actorattack?): pass max_model_len explicitly (positional 8) and record why"
    WINDOW_SOURCE="geometry.multiturn_window"
  else
    WINDOW_SOURCE="positional"
  fi
  f22_standup_extra_reset --upstream-timeout-s "$PROXY_UPSTREAM_TIMEOUT_S" --driver-max-tokens "$CAP_TOKENS" ${F22_LOG_REQ_FLAG[@]+"${F22_LOG_REQ_FLAG[@]}"}
  DRIVER_CAP_FLAGS=(--target-max-new-tokens "$CAP_TOKENS" --target-timeout-s "$CLIENT_TIMEOUT_S")
  f22_log "attacks=$ATTACKS cap=$CAP_TOKENS window=$GEOM_L ($WINDOW_SOURCE; S sized from $KEY's budget) client_timeout_s=$CLIENT_TIMEOUT_S proxy_upstream_timeout_s=$PROXY_UPSTREAM_TIMEOUT_S"
fi
if [ -n "$MT_L" ]; then
  # path (a): the table's multiturn_attacks row at the families' (pinned or derived) max window
  f22_standup attacks "$SERVE_DIR" "" ${MT_L:+multiturn_attacks} ${MT_L:+"$MT_L"} || { RC=1; exit 1; }
else
  # path (b): the EXPLICIT window resolved above (never the table's `multiturn_attacks` 8,704), S = the
  # model's budget // L (identity kept; the standup attests both and refuses a mismatch)
  f22_standup attacks "$SERVE_DIR" "" "" "$GEOM_L" "" || { RC=1; exit 1; }
fi
# native-serving-default: the cells this standup SERVES (PSR / vector cells serve on the native engine's residual slot since native-psr-serving; it skips AlphaSteer / antidote / the stacked wrapper and a position-masked LoRA fit by name — a cells= positional naming a skipped cell is fatal)
f22_served_cells_filter CELLS "$CELLS_ARG"
# STEER-IS-LIVE positive control on the first steered cell of this job (common.sh f22_positive_control): the
# smoke's check, run here so a silently no-op'd object refuses before any driver spends a GPU minute
# Write the NUMERIC liveness record for every steered cell of this job FIRST, against this
# standup's proxy and routing table, so the smoke consults it when the benign text probe reads
# 0/5 differing on a small dose. Always returns 0: no writer at this sha, or an unevaluable cell,
# leaves every process byte-identical to before and the text gate standing.
f22s_low_dose_liveness
for pc in "${CELLS[@]}"; do case "$pc" in none|prompted__*) continue;; esac
  if [ -n "${F22S_LIVENESS_RECORD[$pc]:-}" ]; then
    # the record goes to the smoke's process ONLY (a subshell export), never the job shell
    ( export SISTEER_LOW_DOSE_LIVENESS_RECORD="${F22S_LIVENESS_RECORD[$pc]}"; f22_positive_control "$pc" attacks ) || { RC=1; exit 1; }
  else
    f22_positive_control "$pc" attacks || { RC=1; exit 1; }
  fi
  break; done   # the first OBJECT cell: a prompted baseline has no scale-0 twin
# Phase B: the shared attacker pool (LIVE members only)
f22_live_attackers
# The funnel ASSIGNED this target a member at placement (attackers/PLACED: assigned_member_job_id).
# When that member is live, every driver of this job binds IT (the other live members stay behind it — the failover list); a dead or
# unrecorded assignment falls back to the round-robin over the live members, logged. The member counts its assigned targets as consumers.
ASSIGNED_LIVE=0
if [ -n "${REPO_DIR:-}" ] && [ -f "$REPO_DIR/scripts/sdfbase/full22/attacker_pool_lifecycle.sh" ]; then
  # shellcheck disable=SC1091
  source "$REPO_DIR/scripts/sdfbase/full22/attacker_pool_lifecycle.sh"
  ASSIGNED_URL=$(f22_pool_assigned_member_url "$WAVE_ROOT" "$SLURM_JOB_ID" 2>/dev/null || true)
  if [ -n "$ASSIGNED_URL" ]; then
    aidx=-1; for k in "${!ATTACKERS[@]}"; do [ "${ATTACKERS[$k]}" = "$ASSIGNED_URL" ] && aidx=$k; done
    if [ "$aidx" -ge 0 ]; then
      _rest=(); _rest_m=(); for k in "${!ATTACKERS[@]}"; do [ "$k" -ne "$aidx" ] && { _rest+=("${ATTACKERS[$k]}"); _rest_m+=("${ATTACKER_MANIFESTS[$k]:-}"); }; done
      ATTACKERS=("$ASSIGNED_URL" ${_rest[@]+"${_rest[@]}"}); ATTACKER_MANIFESTS=("${ATTACKER_MANIFESTS[$aidx]:-}" ${_rest_m[@]+"${_rest_m[@]}"}); ASSIGNED_LIVE=1
      f22_log "assigned member $ASSIGNED_URL is live: every driver of this job binds it (${#_rest[@]} other live member(s) behind it)"
    else
      f22_log "WARNING: the assigned member $ASSIGNED_URL is not among the live members (${ATTACKERS[*]}) — round-robin over the live members"
    fi
  else
    f22_log "no placement assignment for job $SLURM_JOB_ID yet (attackers/PLACED absent or not naming it) — round-robin over the live members"
  fi
else
  f22_log "WARNING: attacker_pool_lifecycle.sh not found under REPO_DIR=${REPO_DIR:-unset} — no placement assignment read; round-robin over the live members"
fi
# 31b-fix-2: record the CONSUMER set this job actually binds — the assigned member, or the whole live pool under the
# round-robin fallback — in the binding's `urls` (f22_live_attackers wrote every live member there before the assignment was chosen); the
# other live members are the failover `candidates`. The members' lifecycle counts consumers through `urls` only.
if type f22_pool_record_binding >/dev/null 2>&1 && [ -n "${SERVE_DIR:-}" ] && [ -d "$SERVE_DIR" ]; then
  _assigned=""; [ "$ASSIGNED_LIVE" = 1 ] && _assigned="$ASSIGNED_URL"
  f22_log "binding recorded (assignment, consumers, candidates): $(f22_pool_record_binding "$SERVE_DIR" "$_assigned" "${ATTACKERS[@]}")"
fi
# Phase C: the drivers, ALL concurrent, round-robined over the pool; no GPU of their own
IFS=',' read -ra ATKS <<< "$ATTACKS"; IFS=',' read -ra SHARD_LIST <<< "$SHARDS"
declare -A DRIVER_TAG=(); i=0
for atk in "${ATKS[@]}"; do for cell in "${CELLS[@]}"; do for s in "${SHARD_LIST[@]}"; do
  f22_stop_requested && { f22_log "wave STOP present — not launching more drivers"; break 3; }
  tag="${atk}__${cell}__s$(printf '%02d' "$s")"; if [ "$ASSIGNED_LIVE" = 1 ]; then url="${ATTACKERS[0]}"; else url="${ATTACKERS[$((i % ${#ATTACKERS[@]}))]}"; fi
  # the SAME member's engine-verified serving manifest (index-parallel), so this driver records a
  # real attacker_quantization instead of 'unrecorded' and its store stays RESUMABLE
  if [ "$ASSIGNED_LIVE" = 1 ]; then aman="${ATTACKER_MANIFESTS[0]:-}"; else aman="${ATTACKER_MANIFESTS[$((i % ${#ATTACKERS[@]}))]:-}"; fi; ASERV=()
  [ -n "$aman" ] && ASERV=(--attacker-serving "$aman")
  i=$((i+1))
  # THIS cell's record to THIS cell's drivers only, in front of the interpreter — never
  # exported to the job and never handed to another cell. Empty array = byte-identical to before.
  rec="${F22S_LIVENESS_RECORD[$cell]:-}"; LIVE_ENV=()
  [ -n "$rec" ] && LIVE_ENV=(env "SISTEER_LOW_DOSE_LIVENESS_RECORD=$rec")
  ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" ${LIVE_ENV[@]+"${LIVE_ENV[@]}"} "$MPY" scripts/jailbreak_run.py generate \
      --config "$CFG" --attack "$atk" --defense "$cell" --confirmatory \
      --shard "$s" --nshards "$NSHARDS" --private-root "$PRIVATE_ROOT" \
      --attacker-engine vllm --attacker-endpoint "$url" --attacker-endpoints "$url$(printf '%s\n' "${ATTACKERS[@]}" | grep -vxF "$url" | sed 's/^/,/' | tr -d '\n')" ${ASERV[@]+"${ASERV[@]}"} \
      --target-engine served --target-proxy-url "$PROXY_URL" --target-routing-table "$TABLE" \
      --target-max-model-len "$TARGET_L" --expect-pair-n-streams 5 "${DRIVER_CAP_FLAGS[@]}" \
      --multiturn-target-decode "${F22_MT_TARGET_DECODE:-vendor}" --multiturn-target-seed "${F22_MT_TARGET_SEED:-per_turn}" --model-key "$KEY" \
      ${F22_MT_TARGET_SEED_BASE:+--multiturn-target-seed-base "$F22_MT_TARGET_SEED_BASE"} \
      ${F22_MT_TARGET_POLICY_REF:+--multiturn-target-policy-ref "$F22_MT_TARGET_POLICY_REF"} ) > "$SERVE_DIR/drivers/$tag.log" 2>&1 &
  pid=$!; DRIVER_PIDS+=("$pid"); DRIVER_TAG[$pid]="$tag"
  sleep 3   # stagger the per-driver salt-isolation + steering-bites probes
done; done; done
f22_log "launched ${#DRIVER_PIDS[@]} drivers over ${#ATTACKERS[@]} attackers"
nfail=0
for pid in "${DRIVER_PIDS[@]}"; do
  wait "$pid"; rc=$?
  [ "$rc" = 0 ] || { nfail=$((nfail+1)); f22_log "DRIVER FAILED rc=$rc: ${DRIVER_TAG[$pid]}"; tail -5 "$SERVE_DIR/drivers/${DRIVER_TAG[$pid]}.log"; }
done
DRIVER_PIDS=()
PRIVATE_RUN=$("$MPY" -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['private_run'])" "$CFG")
"$MPY" - "$PRIVATE_ROOT/$PRIVATE_RUN" "$ATTACKS" "$(IFS=,; echo "${CELLS[*]}")" <<'PY' | tee "$SERVE_DIR/completeness.txt"
import json, sys
from pathlib import Path
root, attacks, cells = Path(sys.argv[1]), sys.argv[2].split(","), sys.argv[3].split(",")
for atk in attacks:
    for cell in cells:
        goals, errs, latest = set(), 0, {}
        for sd in sorted((root / atk / cell).glob("shard*")):
            p = sd / "progress.jsonl"
            if not p.is_file():
                continue
            for line in p.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line); goals.add(rec.get("goal_id"))
                    errs += str(rec.get("outcome", "")).startswith("error")
                    latest[rec.get("goal_id")] = str(rec.get("outcome", ""))      # the file is append-only: a goal's LATEST row is its outcome (the grader's reading)
        bad = sum(1 for o in latest.values() if o.startswith(("error", "refused")))
        # error_outcomes counts every historical error row (a resumed goal keeps its old rows); latest_error_or_refused is the goals still ending badly
        print(f"COMPLETENESS {atk} {cell} unique_goals={len(goals)} error_outcomes={errs} latest_error_or_refused={bad}")
PY
touch "$SERVE_DIR/STOP"; RC=$(( nfail > 0 ? 2 : 0 ))
f22_log "drivers done: ${nfail} failed; stopping the target"
exit $RC
