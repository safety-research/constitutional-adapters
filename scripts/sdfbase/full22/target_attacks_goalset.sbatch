#!/bin/bash
#SBATCH --job-name=f22-att-goalset
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=0-06:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# full22: target_attacks.sbatch for a NAMED GOAL SET per (attack, cell), into a NEW private root, at a
# chosen target decode -- the GLM errored-goal recovery (2026-09-10): the 117
# ActorAttack / Crescendo goal-outcomes whose attack loops ERRORED in one job window are re-run, once at
# the campaign's greedy default and once at GLM's vendor sampling preset, each into its own root beside
# (never inside) the filed transcripts. Same served target standup, positive control, attacker pool and
# per-driver shape as target_attacks.sbatch; the differences are exactly:
#   * drivers run only the (attack, cell) pairs that have a goal-set file <goalset_dir>/<attack>__<cell>.txt
#     (one goal_id per line; jailbreak_run.py --goal-ids-file replays it exactly, refusing an unknown id),
#     sharded min(n_goals, max_shards) ways so a driver carries ~1 goal.
#   * the private root is $WAVE_ROOT/transcripts_results_<root_tag> (a fresh namespace; the filed
#     transcripts_results/ is never opened).
#   * the target decode is explicit: "greedy" (temperature 0 -- the drivers' default, request bytes
#     unchanged) or "temperature=<t>,top_p=<p>,top_k=<k>,repetition_penalty=<r>" (threaded as
#     --target-temperature/--target-top-p/--target-top-k/--target-repetition-penalty; recorded in
#     attacker_engine.json target_decode and on every response / timing row).
#   * (2026-09-18) every driver is handed its pool member's ENGINE-VERIFIED serving manifest -- --attacker-serving beside
#     --attacker-endpoint, joined back from the member's pool record by URL -- and a live member that cannot be bound
#     REFUSES the job before any driver (f22g_bind_attacker_manifests below): without the flag the store records
#     attacker_serving_manifest null / attacker_quantization "unrecorded" and can never be resumed.
# Usage (positional only -- never --export=):
#   sbatch -J <name> scripts/sdfbase/full22/target_attacks_goalset.sbatch \
#       <key> <view> <attacks> <wave_sha> <goalset_dir> <root_tag> <decode_spec> [max_shards=9] \
#       [max_new_tokens=""] [max_model_len=""] [max_num_seqs=""] [client_timeout_s=""] [proxy_upstream_timeout_s=""]
#   The five trailing optionals are the 4,096-cap multi-turn re-run's knobs (2026-09-11); "" = unchanged:
#     max_new_tokens          the TARGET's per-reply cap (--target-max-new-tokens; "" = the config's 512)
#     max_model_len/max_num_seqs  the served window / width handed to the standup ("" = geometry.multiturn_window(attacks):
#                             crescendo 49,152 / pair 16,384, the measured maxima on complete stores -- NEVER the table's
#                             8,704, since 2026-09-11; an explicit L with "" S is sized from the model's budget -- identity kept)
#     client_timeout_s        the driver's per-request timeout to the proxy (--target-timeout-s; "" = 600)
#     proxy_upstream_timeout_s  the proxy's per-request timeout to vLLM (mt_serving_up --upstream-timeout-s; "" = 600)
#   The decode spec also accepts note=<text-without-commas> -> --target-decode-note (a note on EVERY decode block).
set -uo pipefail
KEY="${1:?key}"; VIEW="${2:?view}"; ATTACKS="${3:?attacks}"; WAVE_SHA="${4:?wave sha}"
GOALSET_DIR="${5:?goalset dir}"; ROOT_TAG="${6:?root tag}"; DECODE_SPEC="${7:?decode spec (greedy | temperature=..,top_p=..,top_k=..,repetition_penalty=..[,note=..])}"
MAX_SHARDS="${8:-9}"
CAP_TOKENS="${9:-}"; GEOM_L="${10:-}"; GEOM_S="${11:-}"; CLIENT_TIMEOUT_S="${12:-}"; PROXY_UPSTREAM_TIMEOUT_S="${13:-}"
{ [ -n "${SISTEER_SHARED_ROOT:-}" ] && source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA/scripts/sdfbase/full22/common.sh" 2>/dev/null; } \
  || source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit
F22_LAUNCH_MANIFEST=$(f22_full22_manifest "$KEY" "$VIEW")   # the launch's cells for the kind-aware serving-path default (2026-09-15): the (KEY, VIEW) campaign manifest
f22_init "$WAVE_SHA"; f22_resolve_config "$KEY" "$VIEW"
f22_stop_requested && { f22_log "wave STOP present — not starting"; exit 0; }
# --- fail-fast guards on every external input (cluster-submit: an 11-second death, never a 48-minute one)
[ -d "$GOALSET_DIR" ] || f22_fatal "goalset dir $GOALSET_DIR does not exist"
case "$ROOT_TAG" in ""|*/*|.*) f22_fatal "root tag '$ROOT_TAG' must be a plain name";; esac
[ "$ROOT_TAG" != "" ] || f22_fatal "empty root tag"
case "$MAX_SHARDS" in ''|*[!0-9]*) f22_fatal "max_shards '$MAX_SHARDS' is not an integer";; esac
[ "$MAX_SHARDS" -ge 1 ] || f22_fatal "max_shards must be >= 1"
for kv in "cap_tokens=$CAP_TOKENS" "max_model_len=$GEOM_L" "max_num_seqs=$GEOM_S"; do
  v="${kv#*=}"; [ -z "$v" ] && continue
  case "$v" in *[!0-9]*) f22_fatal "${kv%%=*} '$v' is not an integer";; esac
  [ "$v" -ge 1 ] || f22_fatal "${kv%%=*} must be >= 1"
done
for kv in "client_timeout_s=$CLIENT_TIMEOUT_S" "proxy_upstream_timeout_s=$PROXY_UPSTREAM_TIMEOUT_S"; do
  v="${kv#*=}"; [ -z "$v" ] && continue
  case "$v" in *[!0-9.]*|.|*.*.*) f22_fatal "${kv%%=*} '$v' is not a number";; esac
done
[ -n "$GEOM_S" ] && [ -z "$GEOM_L" ] && f22_fatal "max_num_seqs without max_model_len: pass the window too (identity geometry)"
# the decode spec -> driver flags (positional string, parsed here; every value is logged and recorded)
DECODE_FLAGS=()
if [ "$DECODE_SPEC" != greedy ]; then
  IFS=',' read -ra KV <<< "$DECODE_SPEC"
  for kv in "${KV[@]}"; do
    k="${kv%%=*}"; v="${kv#*=}"
    [ -n "$v" ] && [ "$k" != "$kv" ] || f22_fatal "bad decode spec item '$kv' (want key=value)"
    case "$k" in
      temperature)        DECODE_FLAGS+=(--target-temperature "$v");;
      top_p)              DECODE_FLAGS+=(--target-top-p "$v");;
      top_k)              DECODE_FLAGS+=(--target-top-k "$v");;
      repetition_penalty) DECODE_FLAGS+=(--target-repetition-penalty "$v");;
      note)               DECODE_FLAGS+=(--target-decode-note "$v");;
      *) f22_fatal "unknown decode key '$k' (temperature|top_p|top_k|repetition_penalty|note)";;
    esac
  done
  case "$DECODE_SPEC" in *temperature=*) ;; *) f22_fatal "a sampled decode spec must name temperature=";; esac
fi
# the cap and the client timeout travel as driver flags beside the decode (both recorded by the driver)
[ -n "$CAP_TOKENS" ] && DECODE_FLAGS+=(--target-max-new-tokens "$CAP_TOKENS")
[ -n "$CLIENT_TIMEOUT_S" ] && DECODE_FLAGS+=(--target-timeout-s "$CLIENT_TIMEOUT_S")
# the proxy's upstream timeout rides common.sh's F22_STANDUP_EXTRA array into mt_serving_up.py --upstream-timeout-s
# (recorded in endpoint.json proxy_upstream_timeout_s); an empty value leaves the proxy's own default (600 s) in force
f22_standup_extra_reset; [ -n "$PROXY_UPSTREAM_TIMEOUT_S" ] && F22_STANDUP_EXTRA+=(--upstream-timeout-s "$PROXY_UPSTREAM_TIMEOUT_S")   # on top of the inherited launch-time hook (common.sh, 2026-09-16)
f22_log "goalset run: dir=$GOALSET_DIR root_tag=$ROOT_TAG decode=$DECODE_SPEC flags=[${DECODE_FLAGS[*]:-none}] max_shards=$MAX_SHARDS cap=${CAP_TOKENS:-config} geometry=${GEOM_S:-budget}x${GEOM_L:-table} client_timeout_s=${CLIENT_TIMEOUT_S:-600} proxy_upstream_timeout_s=${PROXY_UPSTREAM_TIMEOUT_S:-600}"
GROUPTAG=$(echo "$ATTACKS" | tr ',' '-')
SERVE_DIR="$WAVE_ROOT/serving/attacks_${TAG}_${ROOT_TAG}_${GROUPTAG}_${SLURM_JOB_ID}"; PRIVATE_ROOT="$WAVE_ROOT/transcripts_results_${ROOT_TAG}"
mkdir -p "$SERVE_DIR/drivers" "$PRIVATE_ROOT"; chmod 700 "$PRIVATE_ROOT"
f22_driver_venv; f22_serving_venv; f22_bank_view "$MPY"
# the window: explicit per attack from the measured maxima unless passed (the table's 8,704 clamped
# PAIR at 4,096); the standup is told the drivers' cap so a table window can never serve it silently
EFFECTIVE_CAP="${CAP_TOKENS:-$("$MPY" -c 'import sys,yaml;print(int(yaml.safe_load(open(sys.argv[1]))["attacks"]["target_max_new_tokens"]))' "$CFG")}"
F22_STANDUP_EXTRA+=(--driver-max-tokens "$EFFECTIVE_CAP")
if [ -z "$GEOM_L" ]; then
  GEOM_L=$(cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src" "$MPY" -c 'import sys; from sisteer.capeval.serving.geometry import multiturn_window; print(multiturn_window(sys.argv[1]))' "$ATTACKS") \
    || f22_fatal "no measured ${EFFECTIVE_CAP}-cap serving window for attacks=$ATTACKS (actorattack?): pass max_model_len explicitly (positional 10)"
fi
# the (attack, cell) pairs = the goal-set files present for the requested attacks; refuse an empty set
IFS=',' read -ra ATKS <<< "$ATTACKS"
declare -A NGOALS=(); PAIRS=(); CELLS_WITH_GOALS=()
for atk in "${ATKS[@]}"; do for cell in "${ALL_CELLS[@]}"; do
  f="$GOALSET_DIR/${atk}__${cell}.txt"
  [ -e "$f" ] || continue
  [ -s "$f" ] || f22_fatal "goalset file $f exists but is empty"
  n=$(grep -c -v '^[[:space:]]*\(#\|$\)' "$f")
  [ "$n" -ge 1 ] || f22_fatal "goalset file $f lists no goal ids"
  NGOALS["${atk}__${cell}"]=$n; PAIRS+=("${atk}__${cell}")
  case " ${CELLS_WITH_GOALS[*]:-} " in *" $cell "*) ;; *) CELLS_WITH_GOALS+=("$cell");; esac
done; done
[ "${#PAIRS[@]}" -ge 1 ] || f22_fatal "no goalset file in $GOALSET_DIR matches attacks=$ATTACKS x the config's cells"
for p in "${PAIRS[@]}"; do f22_log "  pair $p: ${NGOALS[$p]} goals -> $(( NGOALS[$p] < MAX_SHARDS ? NGOALS[$p] : MAX_SHARDS )) shards"; done
TARGET_PID=""; declare -a DRIVER_PIDS=(); RC=0
cleanup(){
  touch "$SERVE_DIR/STOP" 2>/dev/null || true
  for p in "${DRIVER_PIDS[@]:-}"; do [ -n "$p" ] && kill -TERM "$p" 2>/dev/null || true; done
  [ -n "${TARGET_PID:-}" ] && { for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; }
  f22_result "attacks-goalset $TAG $ROOT_TAG $GROUPTAG rc=${RC}"
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
# Phase A: the served target (target_attacks.sbatch's standup: stacked lens, attested; the `attacks` table geometry
# unless an explicit window/width was passed; since 2026-09-11 the default IS explicit (geometry.multiturn_window: L=49152 for Crescendo so a 10-turn conversation
# of 4,096-token replies never trips the client's once-only max_tokens clamp)
TARGET_GPU="$CUDA_VISIBLE_DEVICES"
f22_standup attacks "$SERVE_DIR" "" "" "$GEOM_L" "$GEOM_S" || { RC=1; exit 1; }
# native-serving-default: every (attack, cell) pair here is an EXPLICIT ask (a goal-set file names it), so a cell the
# standup skipped by name on the native engine is fatal — never dropped; F22_SERVING_PATH=lens serves it on the lens
f22_served_cells_filter CELLS_WITH_GOALS "goalset:$GOALSET_DIR"
# STEER-IS-LIVE positive control on the first steered OBJECT cell among the cells this job drives
for pc in "${CELLS_WITH_GOALS[@]}"; do case "$pc" in none|prompted__*) continue;; esac; f22_positive_control "$pc" attacks || { RC=1; exit 1; }; break; done
# Phase B: the shared attacker pool (LIVE members only). Under a full queue (the budget held by
# other jobs) a target and its pool members start whenever Slurm places them, and the 27B attacker's standup
# (~22 min) is longer than the target's (~10 min) -- so wait, bounded, for the FIRST live member instead of
# dying on a pool that is seconds from serving; f22_live_attackers still refuses if none is live at the end.
ATK_WAIT_MIN=90; served_name=$("$MPY" -c 'import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))["attacker"]["model"])' "$CFG")
# WAIT ONLY IF AN ATTACKER EXISTS TO WAIT FOR (2026-09-18). The bounded wait above is right when a pool is
# BOOTING; it is pure waste when NOBODY EVER SUBMITTED ONE, and the two look identical from here — an empty
# attackers/ directory. On 2026-09-17 a launch sequence was interrupted between submitting the target arrays
# and deploying the attacker, so that step never ran; seven targets came up healthy, passed their positive
# controls, and sat idle for the full budget while the wave recorded nothing at all about the missing stage.
# So: require EVIDENCE that a member exists before idling a GPU for an hour and a half. Evidence is any
# `<jobid>.json` (a member already serving) or any `<jobid>.pending` (a member that has started and is
# booting — attacker_pool.sbatch writes it at job start and removes it on exit). `.STOP` files are NOT
# evidence: they are the tombstones of members that have gone away, and a directory holding only tombstones
# is exactly the case this refuses. Fail-closed only on the unambiguous signature; if either kind of evidence
# is present we fall through to the bounded wait unchanged.
# SCOPE THE EVIDENCE AND THE WAIT TO THE MEMBERS THIS TARGET WILL BIND (2026-09-22).
# f22_declared_attackers is the single enumeration point and is where F22_ATTACKER_JOBS applies, so
# asking it -- rather than re-globbing attackers/* here -- makes a PINNED target ask "are MY members
# up?" instead of "is anyone up?". Before this the two were different predicates over the same pool:
# the evidence glob and the wait counted EVERY record, so a pinned target was satisfied by another
# lane's member and broke out of its ATK_WAIT_MIN budget the moment that member served. It then fell
# into f22_live_attackers, which re-waits for the PINNED set on the much shorter F22_ATTACKER_WAIT_S
# (1800s default) -- a window a ~22-min attacker standup plus queue time can miss outright, leaving
# the target to fatal with its GPUs idle. Pinning is what pulled the two predicates apart; unpinned,
# the declared set IS every member, so this is the previous behaviour unchanged.
atk_declare(){   # -> ATK_DECLARED, or refuse
  local out
  out=$(f22_declared_attackers) || f22_fatal "the F22_ATTACKER_JOBS pin names a member with no record under $WAVE_ROOT/attackers (reason above) — refusing rather than idling this target's GPUs waiting for a member that was never submitted, or binding to members it did not name."
  ATK_DECLARED=(); [ -n "$out" ] && mapfile -t ATK_DECLARED <<< "$out"
}
atk_declare
if [ "${#ATK_DECLARED[@]}" -eq 0 ] && [ -z "${F22_ATTACKER_WAIT_ANYWAY:-}" ]; then
  f22_fatal "no attacker exists to wait for: $WAVE_ROOT/attackers holds no <jobid>.json (serving) and no <jobid>.pending (booting)$(
    ls "$WAVE_ROOT"/attackers/*.STOP >/dev/null 2>&1 && printf ' — only .STOP tombstones of members that have exited')."\
" This target is up and would otherwise idle its GPUs for ${ATK_WAIT_MIN} min for a pool that was never submitted."\
" Submit attacker_pool.sbatch for THIS wave root (F22_SERVING_PATH=$F22_SERVING_PATH selects it) and resubmit, or"\
" set F22_ATTACKER_WAIT_ANYWAY=1 if a pool is coming from somewhere that publishes no marker (e.g. a Modal pool)."
fi
for (( w=0; w<ATK_WAIT_MIN; w++ )); do
  n_live=0
  # Re-enumerated every pass, as the old glob was: unpinned, members that come online mid-wait must
  # count; pinned, a member whose record disappears mid-wait must refuse rather than be waited out.
  atk_declare
  for id in ${ATK_DECLARED[@]+"${ATK_DECLARED[@]}"}; do
    rec="$WAVE_ROOT/attackers/$id.json"
    [ -f "$rec" ] || continue
    u=$("$MPY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["url"])' "$rec" 2>/dev/null) || continue
    curl -s -m 15 "$u/v1/models" 2>/dev/null | grep -q -F "$served_name" && n_live=$((n_live+1))
  done
  [ "$n_live" -ge 1 ] && break
  f22_stop_requested && { f22_log "wave STOP present while waiting for an attacker — leaving"; RC=0; exit 0; }
  [ $(( w % 5 )) -eq 0 ] && f22_log "waiting for a LIVE attacker among this target's ${#ATK_DECLARED[@]} declared member(s)${F22_ATTACKER_JOBS:+ (pinned: $F22_ATTACKER_JOBS)} in $WAVE_ROOT/attackers ($w/$ATK_WAIT_MIN min; the target is up and idle)"
  sleep 60
done
f22_live_attackers
# --- every driver's --attacker-serving: its pool member's ENGINE-VERIFIED serving manifest, joined back by URL (2026-09-18)
# Without the flag the driver takes attacker_serving.driver_fields(None) and stamps the store attacker_serving_manifest: null /
# attacker_quantization: "unrecorded"; the engine-identity join then REFUSES to resume that store under ANY identity
# (EngineIdentityError), so a walltime is a full restart of the (attack, cell), never a resume. The same defect appeared
# on 2026-09-18 (18 dead drivers, 754 rows lost; its fix lives in common.sh + target_attacks.sbatch -- this launcher
# joins in place and leaves common.sh untouched). Every pool record attacker_pool.sbatch writes ($WAVE_ROOT/attackers/<jobid>.json,
# the files f22_live_attackers reads the URL list from) carries the member's serving_manifest path; this maps url -> path and
# REFUSES -- before any driver, so before any generation -- a live URL with no record, a record without the field, a path that
# is not a file, or two records naming one URL with different manifests. An unbound driver is the defect being removed, so
# there is deliberately no fallback to launching without the flag.
declare -A ATTACKER_MANIFEST_OF_URL=()
f22g_bind_attacker_manifests(){   # ATTACKERS + $WAVE_ROOT/attackers/*.json -> ATTACKER_MANIFEST_OF_URL[url]=path, or exit 1
  ATTACKER_MANIFEST_OF_URL=()
  local u m url
  while IFS=$'\t' read -r u m; do
    [ -n "$u" ] || continue
    if [ -n "${ATTACKER_MANIFEST_OF_URL[$u]+x}" ] && [ "${ATTACKER_MANIFEST_OF_URL[$u]}" != "$m" ]; then
      f22_fatal "attacker records under $WAVE_ROOT/attackers name $u with two different serving manifests (${ATTACKER_MANIFEST_OF_URL[$u]:-none} vs ${m:-none}) — an ambiguous identity is refused, never guessed; remove the stale record"
    fi
    ATTACKER_MANIFEST_OF_URL[$u]="$m"
  done < <("$MPY" - "$WAVE_ROOT/attackers" <<'PY'
import json, sys
from pathlib import Path
for rec in sorted(Path(sys.argv[1]).glob("*.json")):
    try:
        d = json.loads(rec.read_text())
    except Exception:
        continue
    if isinstance(d, dict) and d.get("url"):
        print(f"{d['url']}\t{d.get('serving_manifest') or ''}")
PY
)
  for url in "${ATTACKERS[@]}"; do
    [ -n "${ATTACKER_MANIFEST_OF_URL[$url]+x}" ] || f22_fatal "live attacker $url has no record under $WAVE_ROOT/attackers — cannot bind its serving manifest; its drivers would write attacker_quantization 'unrecorded' (an UNRESUMABLE store); refusing to launch"
    [ -n "${ATTACKER_MANIFEST_OF_URL[$url]}" ] || f22_fatal "attacker record for $url carries no serving_manifest — its drivers would write attacker_quantization 'unrecorded' (an UNRESUMABLE store); refusing to launch (re-stand the member under an attacker_pool.sbatch that writes the field)"
    [ -f "${ATTACKER_MANIFEST_OF_URL[$url]}" ] || f22_fatal "attacker $url: serving_manifest ${ATTACKER_MANIFEST_OF_URL[$url]} is not a file — refusing to hand a driver an identity it cannot read"
    f22_log "attacker $url -> serving manifest ${ATTACKER_MANIFEST_OF_URL[$url]}"
  done
}
f22g_bind_attacker_manifests
# Phase C: the drivers, ALL concurrent, round-robined over the pool; one goal-set file per (attack, cell)
declare -A DRIVER_TAG=(); i=0
for atk in "${ATKS[@]}"; do for cell in "${CELLS_WITH_GOALS[@]}"; do
  pair="${atk}__${cell}"; [ -n "${NGOALS[$pair]:-}" ] || continue
  n=${NGOALS[$pair]}; ns=$(( n < MAX_SHARDS ? n : MAX_SHARDS ))
  for (( s=0; s<ns; s++ )); do
    f22_stop_requested && { f22_log "wave STOP present — not launching more drivers"; break 3; }
    tag="${atk}__${cell}__s$(printf '%02d' "$s")"; url="${ATTACKERS[$((i % ${#ATTACKERS[@]}))]}"; i=$((i+1))
    aman="${ATTACKER_MANIFEST_OF_URL[$url]}"   # this member's engine-verified serving manifest (bound above for EVERY member, or the job never got here)
    ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" "$MPY" scripts/jailbreak_run.py generate \
        --config "$CFG" --attack "$atk" --defense "$cell" --confirmatory \
        --goal-ids-file "$GOALSET_DIR/${pair}.txt" \
        --shard "$s" --nshards "$ns" --private-root "$PRIVATE_ROOT" \
        --attacker-engine vllm --attacker-endpoint "$url" --attacker-serving "$aman" \
        --target-engine served --target-proxy-url "$PROXY_URL" --target-routing-table "$TABLE" \
        --target-max-model-len "$TARGET_L" --expect-pair-n-streams 5 "${DECODE_FLAGS[@]}" ) > "$SERVE_DIR/drivers/$tag.log" 2>&1 &
    pid=$!; DRIVER_PIDS+=("$pid"); DRIVER_TAG[$pid]="$tag"
    sleep 3   # stagger the per-driver salt-isolation + steering-bites probes
  done
done; done
f22_log "launched ${#DRIVER_PIDS[@]} drivers over ${#ATTACKERS[@]} attackers (decode=$DECODE_SPEC; each bound to its member's serving manifest)"
nfail=0
for pid in "${DRIVER_PIDS[@]}"; do
  wait "$pid"; rc=$?
  [ "$rc" = 0 ] || { nfail=$((nfail+1)); f22_log "DRIVER FAILED rc=$rc: ${DRIVER_TAG[$pid]}"; tail -5 "$SERVE_DIR/drivers/${DRIVER_TAG[$pid]}.log"; }
done
DRIVER_PIDS=()
PRIVATE_RUN=$("$MPY" -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['private_run'])" "$CFG")
# completeness = unique goal ids in progress.jsonl per pair against the GOAL SET (never sacct); an errored
# outcome or a short pair is a non-zero exit so the watcher sees it
"$MPY" - "$PRIVATE_ROOT/$PRIVATE_RUN" "$GOALSET_DIR" "$(IFS=,; echo "${PAIRS[*]}")" <<'PY' | tee "$SERVE_DIR/completeness.txt"
import json, sys
from pathlib import Path
root, gdir, pairs = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3].split(",")
bad = 0
for pair in pairs:
    atk, cell = pair.split("__", 1)
    want = [ln.strip() for ln in (gdir / f"{pair}.txt").read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    latest, decs = {}, set()
    caps, n_timing, n_at_cap, n_clamped = set(), 0, 0, 0
    for sd in sorted((root / atk / cell).glob("shard*")):
        p = sd / "progress.jsonl"
        if p.is_file():
            for line in p.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line); latest[rec.get("goal_id")] = rec
        r = sd / "responses.jsonl"
        if r.is_file():
            for line in r.read_text().splitlines():
                if line.strip():
                    decs.add(json.dumps((json.loads(line).get("decode") or {}), sort_keys=True))
        # the served timing rows: the cap actually sent (max_tokens), the at-cap share and the clamp count --
        # the non-termination reading the 4,096-cap run tracks SEPARATELY from success (R2-D244 #2)
        t = sd / "target_timing.jsonl"
        if t.is_file():
            for line in t.read_text().splitlines():
                if line.strip():
                    row = json.loads(line); n_timing += 1
                    caps.add(row.get("max_tokens")); n_at_cap += row.get("finish_reason") == "length"
                    n_clamped += row.get("max_tokens_clamped_from") is not None
    errs = sum(str(v.get("outcome", "")).startswith("error") for v in latest.values())
    missing = [g for g in want if g not in latest]
    ok = not errs and not missing
    bad += not ok
    print(f"COMPLETENESS {atk} {cell} expected={len(want)} unique_goals={len(latest)} missing={len(missing)} error_outcomes={errs} "
          f"distinct_decode_blocks={len(decs)} timing_rows={n_timing} distinct_max_tokens={sorted(c for c in caps if c is not None)} "
          f"at_cap={n_at_cap} clamped={n_clamped} {'OK' if ok else 'SHORT'}")
sys.exit(2 if bad else 0)
PY
comp_rc=${PIPESTATUS[0]}
touch "$SERVE_DIR/STOP"; RC=$(( (nfail > 0 || comp_rc != 0) ? 2 : 0 ))
f22_log "drivers done: ${nfail} failed; completeness rc=$comp_rc; stopping the target"
exit $RC
