#!/bin/bash
#SBATCH --job-name=f22-petri
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-00:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# full22: ONE Petri audit PROXY (1 GPU) for one (target, view) config: scripts/sdfbase/mt_serving_up.py
# --mode petri serves every cell of the config (+ the derived none__lens0 control leg) at the certified
# Petri geometry (16384 x the model's budget), agent-loop entry declared, tool calls on, the lens pinned
# to STACKED reductions and REFUSED unless the workers attest it. Loopback-only, on a FIXED per-(target,
# view) port: the grading host's driver (scripts/sdfbase/full22/petri_drive_full22.sh) tunnels in through
# the login node, reads <out>/endpoint.json, and touches <out>/STOP when every leg is complete. Re-submit
# while legs remain (the driver resumes by run key; walltime expiry is a WATCHED terminal state).
# Usage (positional only — never --export=):
#   sbatch -J f22-petri-<key>-<view> scripts/sdfbase/full22/petri_proxy.sbatch <key> <view> <wave_sha> [overflow-rerun]
# The optional 4th arg `overflow-rerun` stands the proxy up for the CONTEXT-OVERFLOW RE-RUN PASS (the driver's
# `petri_run.py run --overflow-rerun`, 2026-09-08): max_model_len comes from petri_runner.rerun_serving_geometry()
# on THIS checkout (the standard window doubled — never a literal here); max_num_seqs is left to the standup, which
# sizes the identity at that window from the model's own budget (the same batched-token product the first pass
# served, so no fresh certification; the table's strict_identity guard refuses a halved S on its own); the proxy
# listens on PROXY_PORT+100 and keeps its state in <tag>/serving_rerun, so the first pass's serving record and
# port are never touched. The driver side is petri_drive_full22.sh under F22_OVERFLOW_RERUN=1.
# The 4th arg `second` (2026-09-08) stands up a SECOND CONCURRENT proxy for the same
# (target, view) at the STANDARD geometry — for legs added to a manifest while the first proxy's driver is still mid-leg
# (the derived cells' legs beside the prompted leg): state in <tag>/serving_second, port PROXY_PORT+200, so neither proxy's
# endpoint.json / STOP / heartbeat touches the other's. The driver side is petri_drive_full22.sh under F22_SECOND_PASS=1.
set -uo pipefail
KEY="${1:?key}"; VIEW="${2:?view}"; WAVE_SHA="${3:?wave sha}"; PASS="${4:-}"; CELLS_ARG="${5:-}"
case "$PASS" in ""|overflow-rerun|second) ;; *) echo "FATAL: 4th arg must be empty, overflow-rerun or second, got '$PASS'" >&2; exit 2;; esac
# petri-cells: positional 5 = the cells this proxy serves to the AUDIT (roster ids, comma-separated; the plan's --cells). The job
# registers exactly them (F22_REGISTRY_CELLS narrows the registration overlay inside f22_resolve_config, so the export sits BEFORE f22_init), the
# standup serves `none` implicitly + exactly them (mt_serving_up --cell, below) and records what was NAMED in cells_served.json
# (cells_requested_explicit / _harness), which the grading host drive freezes as its canonical legs with no controls — no `none`, no `none__lora0` leg:
# the unsteered model is not re-run. Empty = the view's whole enumeration, as before. The overflow-rerun pass carries the same positional 5 —
# on a baseline view, the overflow pass declares the ledgered cells plus `none` (06 --pass overflow, reproduce/lib/petri_overflow.py; petri_run.py
# admits the subset under --overflow-rerun only, petri-completion 2026-10-13): `none` is the controls twin's parent, so declaring it is what keeps
# the twin in the drive's enumeration and the standup verification; leg 0 itself is never driven by that pass.
# petri-cells-fix: positional 5 is the ONE authority for what this job REGISTERS — the same value the standup's
# declaration (--cell, below) and the drive's frozen legs come from. An inherited F22_REGISTRY_CELLS that differs would register a different
# set than the proxy serves and freezes (a hand retry, a direct sbatch); an inherited value on a proxy WITHOUT positional 5 (a u0 / whole-baseline
# proxy) would register a subset of the baseline. Both refuse by name, before anything registers; nothing is sourced yet, so a plain FATAL.
if [ -n "$CELLS_ARG" ]; then
  if [ -n "${F22_REGISTRY_CELLS:-}" ] && [ "$F22_REGISTRY_CELLS" != "$CELLS_ARG" ]; then
    echo "FATAL: petri cells: positional 5 names '$CELLS_ARG' but the inherited F22_REGISTRY_CELLS='$F22_REGISTRY_CELLS' differs — the registration would cover a different set than the standup declares and the drive freezes (petri-cells-fix); positional 5 is the one authority: unset the variable or pass the cells positionally" >&2
    exit 2
  fi
  export F22_REGISTRY_CELLS="$CELLS_ARG"
else
  if [ -n "${F22_REGISTRY_CELLS:-}" ]; then
    echo "FATAL: petri cells: no positional 5 (a whole-baseline proxy) but an inherited F22_REGISTRY_CELLS='$F22_REGISTRY_CELLS' — the registration would cover a subset of the baseline the proxy serves and the drive freezes (petri-cells-fix); unset it or pass the cells positionally" >&2
    exit 2
  fi
  unset F22_REGISTRY_CELLS
fi
# The proxy's run key is the measurement's — `sdfbase_full22-petri-<key>[_<view>]-pass-<first|overflow-rerun|second>` (common.sh
# f22_registry_producer_key, inside f22_resolve_config below) — so a re-submitted proxy for the same audit pass registers the SAME runs
export F22_REGISTRY_KEY_SEGMENTS="${F22_REGISTRY_KEY_SEGMENTS:-pass-${PASS:-first}}"
{ [ -n "${SISTEER_SHARED_ROOT:-}" ] && source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA/scripts/sdfbase/full22/common.sh" 2>/dev/null; } \
  || source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit
F22_LAUNCH_MANIFEST=$(f22_full22_manifest "$KEY" "$VIEW")   # the launch's cells for the kind-aware serving-path default (2026-09-15): the (KEY, VIEW) campaign manifest
f22_init "$WAVE_SHA"; f22_resolve_config "$KEY" "$VIEW"
# a view without the Petri leg (the llama8b fullconst view) is refused up front (roster_view.py has_leg)
python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" has_leg petri --roster "$F22_ROSTER" || f22_fatal "view $KEY/$VIEW has no petri leg — this proxy must not be submitted for it"
f22_stop_requested && { f22_log "wave STOP present — not starting"; exit 0; }
# THE PORT TABLE (paper-companion-repo): configs/sdfbase_full22_ports.yaml (in this checkout) is the declared source —
# tests/test_models_demo_front_door.py holds it equal to the case table below for every tag it names; the case table stays
# the fallback for a checkout without the file, byte-for-byte the previous behaviour. The node's python3 + PyYAML read it
# (the same interpreter f22_select_auto already relies on).
PROXY_PORT=""
[ -f configs/sdfbase_full22_ports.yaml ] && PROXY_PORT=$(python3 -c 'import sys,yaml; print((yaml.safe_load(open(sys.argv[1])).get("ports") or {}).get(sys.argv[2], ""))' configs/sdfbase_full22_ports.yaml "$TAG" 2>/dev/null)
[ -n "$PROXY_PORT" ] && f22_log "proxy port $PROXY_PORT for $TAG from configs/sdfbase_full22_ports.yaml"
if [ -z "$PROXY_PORT" ]; then
case "$TAG" in   # fixed loopback ports: the grading host tunnel binds the same number locally (+10000)
  gemma4e4b) PROXY_PORT=8981;; gemma4e4b_it) PROXY_PORT=8982;;
  qwen35_4b) PROXY_PORT=8983;; qwen35_4b_it) PROXY_PORT=8984;;
  nemotron3nano4b) PROXY_PORT=8985;; glm46vflash) PROXY_PORT=8986;;
  # the llama8b views (2026-09-10): its two lane views (fullconst has no Petri leg); local +10000 = 18987 / 18988;
  # + the short constitution-SI prompted view (2026-09-12: every leg incl. Petri at the certified 16,384): 8989 / local 18989
  llama8b) PROXY_PORT=8987;; llama8b_it) PROXY_PORT=8988;; llama8b_shortconst) PROXY_PORT=8989;;
  # HALFDOSE views (2026-09-08); gemma4e4b_it_halfdose takes 8997, not 8992: a peer lane holds local 18992
  gemma4e4b_halfdose) PROXY_PORT=8991;; gemma4e4b_it_halfdose) PROXY_PORT=8997;; qwen35_4b_halfdose) PROXY_PORT=8993;;
  qwen35_4b_it_halfdose) PROXY_PORT=8994;; nemotron3nano4b_halfdose) PROXY_PORT=8995;; glm46vflash_halfdose) PROXY_PORT=8996;;
  # qwen-base-seed-replicates (2026-09-15): the three seed-replicate views of qwen35_4b (roster views s1|s2|s3, TAG qwen35_4b_s<s>).
  # 8978-8980 / local 18978-18980: the plan's 8991 / 8992 / 8993 were NOT free (8991 gemma4e4b_halfdose, 8993 qwen35_4b_halfdose,
  # local 18992 held by a peer lane; 18990 is the auditor-chain tunnel's default). The +100 / +200 passes (9078-9080 / 9178-9180)
  # stay clear of every base port. Mirrored in petri_drive_full22.sh; tests/test_petri_proxy_ports.py pins both tables equal.
  qwen35_4b_s1) PROXY_PORT=8978;; qwen35_4b_s2) PROXY_PORT=8979;; qwen35_4b_s3) PROXY_PORT=8980;;
  # the paired views s<s>p (2026-09-20): 8975-8977 (below the seed views; +100 / +200 passes
  # 9075-9077 / 9175-9177 clear of every base port). Mirrored in petri_drive_full22.sh.
  qwen35_4b_s1p) PROXY_PORT=8975;; qwen35_4b_s2p) PROXY_PORT=8976;; qwen35_4b_s3p) PROXY_PORT=8977;;
  *) f22_fatal "no proxy port assigned for $TAG (extend the table in petri_proxy.sbatch + petri_drive_full22.sh)";;
esac
fi
OUT="$WAVE_ROOT/petri/$TAG/serving"; RERUN_L=""
# glm46vflash serves at gpu_memory_utilization 0.92 at EVERY window (common.sh f22_resolve_config: the KV-coverage note)
f22_driver_venv; f22_serving_venv; f22_bank_view "$MPY"
# THE SERVER-DEFAULT SAMPLING POLICY: F22_SERVER_DEFAULT_POLICY (a SCALAR — Slurm's implicit
# export carries it, never --export=) rides through f22_standup to mt_serving_up.py --server-default-policy and is asserted back
# there from endpoint.json + /v1/models. `engine_defaults` (the reused Qwen Petri anchor decoded at vLLM's neutral defaults,
# before the server-side vendor pin) is a policy VERIFIED FROM THE LOG — anchor_sampler_preflight.py
# reads the resolved sampler off vllm_server.log — so it turns request logging on (F22_LOG_REQUESTS=1 -> --enable-log-requests;
# the prompt is a DEBUG record and stays unlogged). Unset = today's standup, byte for byte.
case "${F22_SERVER_DEFAULT_POLICY:-}" in
  "") ;;
  contract_override|engine_defaults) f22_log "server-default sampling policy: $F22_SERVER_DEFAULT_POLICY (F22_SERVER_DEFAULT_POLICY; asserted back from the standup's stamps)";;
  *) f22_fatal "F22_SERVER_DEFAULT_POLICY must be contract_override or engine_defaults, got '$F22_SERVER_DEFAULT_POLICY'";;
esac
if [ "${F22_SERVER_DEFAULT_POLICY:-}" = engine_defaults ] && [ "${F22_LOG_REQUESTS:-0}" != 1 ]; then
  F22_LOG_REQUESTS=1; f22_log "engine_defaults is verified from the server log: F22_LOG_REQUESTS=1 (--enable-log-requests) set for this standup"
fi
# a prompted-only view (llama8b/shortconst, 2026-09-12): the Petri load check probes the PROMPTED route with the SI prefix at the
# Petri cap (the probe shrinks to the room the window leaves, so the certified 16,384 x S geometry is what is checked)
if f22_view_prompted_only; then export F22_LOAD_CHECK_SYSTEM_SI="${F22_LOAD_CHECK_SYSTEM_SI:-$(f22_prompted_si)}"; fi
if [ "$PASS" = overflow-rerun ]; then
  RERUN_L=$("$MPY" -c 'from sisteer.capeval.tiers import petri_runner as pr; print(pr.rerun_serving_geometry()["max_model_len"])') \
    || f22_fatal "petri_runner.rerun_serving_geometry() refused on this checkout — no re-run window to serve"
  OUT="$WAVE_ROOT/petri/$TAG/serving_rerun"; PROXY_PORT=$((PROXY_PORT + 100))
  f22_log "context-overflow re-run pass: max_model_len=$RERUN_L (rerun_serving_geometry), max_num_seqs = the model's identity at that window; out=$OUT port=$PROXY_PORT"
fi
if [ "$PASS" = second ]; then
  OUT="$WAVE_ROOT/petri/$TAG/serving_second"; PROXY_PORT=$((PROXY_PORT + 200))
  f22_log "second concurrent pass: the standard Petri geometry; out=$OUT port=$PROXY_PORT (the first proxy's serving/ and port untouched)"
fi
TARGET_PID=""; RC=0
cleanup(){ touch "$OUT/STOP" 2>/dev/null || true; [ -n "${TARGET_PID:-}" ] && { for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; }; f22_result "petri proxy $TAG rc=${RC}"; }
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
TARGET_GPU="$CUDA_VISIBLE_DEVICES"
if [ -n "$CELLS_ARG" ]; then
  # the standup serves `none` + exactly the named cells (mt_serving_up select_cells; an id outside the config's explicit_cells refuses by name)
  CELL_FLAGS=(); IFS=',' read -ra _cells <<< "$CELLS_ARG"; for c in "${_cells[@]}"; do [ -n "$c" ] && CELL_FLAGS+=(--cell "$c"); done
  f22_standup_extra_reset "${CELL_FLAGS[@]}"; f22_log "petri cells requested of the standup (D1261): ${_cells[*]}"
fi
f22_standup petri "$OUT" "$PROXY_PORT" "" "$RERUN_L" "" || { RC=1; exit 1; }
# THE ANCHOR SAMPLER PREFLIGHT: F22_ANCHOR_PREFLIGHT=<the reused Petri anchor's record — its
# resolved_sampler.json, its vllm_server.log, or a client record together with F22_ANCHOR_SERVER_POLICY> gates this proxy: ONE
# request shaped like the Petri target's (no sampling key; max_tokens = the target cap, F22_PETRI_TARGET_CAP, default 1024) at
# petri__base — a health-class probe on the served `none` route, written to no store — then the ENGINE-RESOLVED tuple for exactly
# that request is compared against the anchor's (common.sh f22_anchor_preflight). Anything but AGREES touches STOP and exits
# before the READY line the grading host's driver waits for, so no leg can start on a non-comparable sampler. Unset = no gate.
if [ -n "${F22_ANCHOR_PREFLIGHT:-}" ]; then
  f22_anchor_preflight petri "$TAG" "$OUT" "$F22_ANCHOR_PREFLIGHT" "${F22_PETRI_TARGET_CAP:-1024}" "" || { RC=1; exit 1; }
fi
f22_log "petri proxy $TAG READY on 127.0.0.1:$PROXY_PORT (node $(hostname)); waiting for the grading host driver's STOP"
# THE DRIVER HEARTBEAT (2026-09-07). Twice in one night the grading host's drivers died while this proxy kept its GPU:
# once when the auditor chain refused before the first sample, once when the driver's kernel restarted and took every
# child process — and both times the watcher that should have noticed shared the drivers' fate. So the proxy
# checks for itself: petri_drive_full22.sh touches $OUT/driver_heartbeat every HEARTBEAT_PERIOD_S while any leg is
# alive; if that file goes stale for HEARTBEAT_STALE_S, or never appears within HEARTBEAT_GRACE_S of READY, the
# proxy releases the GPU (touch STOP). A relaunched driver resumes by run key against a fresh proxy, so releasing
# costs one standup (~10 min), never data.
# A RELEASE WITH NO DRIVER IS A FAILURE, NEVER COMPLETED (2026-10-11): the release used to exit 0, and a registering unit
# that ended COMPLETED 0:0 is one submit_twin.py's --resubmit never re-mints (its runs are "the unit's results"; 06's walk skips it) —
# the proxy would have sat COMPLETED with nothing generated and no way back but a hand cancel. Each release now leaves with its
# NAMED code from common.sh (PROXY_NO_DRIVER = F22_EXIT_PROXY_NO_DRIVER when the heartbeat never arrived, PROXY_DRIVER_LOST =
# F22_EXIT_PROXY_DRIVER_LOST when it went stale), written to this log and carried by the RESULT line's rc= (the registered runs are
# filed failed, never past running), so sacct says FAILED <code>:0 and `06 --family petri --resubmit` re-mints the unit like any dead one.
HEARTBEAT_GRACE_S=5400; HEARTBEAT_STALE_S=1800; READY_AT=$(date +%s); HB="$OUT/driver_heartbeat"; rm -f "$HB"; RELEASE_RC=0; RELEASE_NAME=""
while :; do
  [ -e "$OUT/STOP" ] && { f22_log "STOP touched — legs complete or a deliberate stop"; break; }
  f22_stop_requested && { f22_log "wave STOP present — leaving"; touch "$OUT/STOP"; break; }
  kill -0 "$TARGET_PID" 2>/dev/null || { f22_log "standup exited (rc pending) — see $OUT/standup.log"; break; }
  now=$(date +%s)
  if [ -e "$HB" ]; then
    age=$(( now - $(stat -c %Y "$HB") ))
    [ "$age" -gt "$HEARTBEAT_STALE_S" ] && { RELEASE_RC=$F22_EXIT_PROXY_DRIVER_LOST; RELEASE_NAME=PROXY_DRIVER_LOST; f22_log "driver heartbeat stale (${age}s > ${HEARTBEAT_STALE_S}s) — the grading host driver is gone; releasing the GPU (resume: resubmit this proxy, relaunch the driver); $RELEASE_NAME exit=$RELEASE_RC"; touch "$OUT/STOP"; break; }
  elif [ $(( now - READY_AT )) -gt "$HEARTBEAT_GRACE_S" ]; then
    RELEASE_RC=$F22_EXIT_PROXY_NO_DRIVER; RELEASE_NAME=PROXY_NO_DRIVER; f22_log "no driver heartbeat within ${HEARTBEAT_GRACE_S}s of READY — nothing is driving this proxy; releasing the GPU; $RELEASE_NAME exit=$RELEASE_RC"; touch "$OUT/STOP"; break
  fi
  sleep 30
done
wait "$TARGET_PID" 2>/dev/null; RC=$?
if [ "$RELEASE_RC" -ne 0 ]; then
  # the named release wins over the standup's own exit on STOP (0 when it shut down cleanly): the job's outcome is "released with no driver"
  f22_log "standup exited rc=$RC after the release; leaving with $RELEASE_NAME exit=$RELEASE_RC (never COMPLETED: a dead registering unit is re-mintable, a COMPLETED one is not)"
  RC=$RELEASE_RC
fi
f22_log "petri proxy exited rc=$RC"
exit $RC
