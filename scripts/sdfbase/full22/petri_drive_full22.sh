#!/bin/bash
# full22: drive ONE (target, view)'s Petri audit legs from the grading host against the cluster proxy
# (petri_proxy.sbatch) — the pattern every completed audit used: auditor sonnet-5 / judge opus-5 need
# the Anthropic key, which stays here (fetched from a secret manager at use time, never a file).
# Usage: petri_drive_full22.sh <key> <view> <cluster=cluster_a|cluster_b> <wave_sha> [legs=ALL | "0 1"]   (or F22_PETRI_CELLS=<ids>, below)
# Steps (each loud): provenance (this worktree IS the launched commit, clean) -> the proxy's endpoint.json
# from the cluster -> the job is RUNNING -> tunnel on a per-(target,view) port (REFUSES a held port) ->
# /v1/models lists petri__ vmodels AND its petri_proxy identity block attests the ENGINE (native-serving-default:
# F22_SERVING_PATH=lora_native — explicit, or AUTO-selected by kind from the view's manifest cells when the caller set none
# (the kind-aware default, 2026-09-15; the same rule the cluster's f22_init applied) -> endpoint.json engine == lora_native,
# petri_proxy.engine.kind == lora_native, routing_table.json + cells_served.json pulled beside the endpoint and handed to petri_run.py as
# --verify-standup / --verify-routing-table so the client's enumeration is held to the standup's evidence; lens ->
# lens_reduction_mode=stacked on both stamps, byte for byte) -> FREEZE cells.json (print-cells --serving-path <policy>)
# -> run the legs concurrently at WIDTH 1 (PETRI_EVAL_MAX_CONNECTIONS=1:
# the wave-wide cap is <= 50 concurrent auditor conversations over 48 legs) -> touch STOP on the cluster
# when every leg is COMPLETE. Resume-safe: petri_run.py resumes by run key.
# F22_OVERFLOW_RERUN=1 drives the CONTEXT-OVERFLOW RE-RUN PASS instead (2026-09-08): the proxy is the one
# petri_proxy.sbatch stood up with its `overflow-rerun` arg (<tag>/serving_rerun, port +100), each named leg runs
# `petri_run.py run --overflow-rerun` ONCE (that flag refuses a cell with no unresolved overflow entries and refuses
# a pass that resolved nothing — a verdict, not a transient, so no retry; the eval width is the re-run geometry's
# max_num_seqs, PETRI_EVAL_MAX_CONNECTIONS is ignored), and STOP is touched when the legs end: that proxy exists
# for this pass only. F22_WT=<worktree> names the launched-commit worktree the driver reads configs/scripts/src
# from (default: the worktree this script lives in); the provenance guard below holds either way.
set -uo pipefail
KEY="${1:?key}"; VIEW="${2:?view}"; CLUSTER="${3:?cluster}"; WAVE_SHA="${4:?wave sha}"; LEGS="${5:-ALL}"
RERUN="${F22_OVERFLOW_RERUN:-}"; case "$RERUN" in 0) RERUN="";; esac
# F22_SECOND_PASS=1 (2026-09-08): drive the SECOND CONCURRENT proxy (petri_proxy.sbatch ... second): port +200, state and remote
# dir suffixed _second, a normal `run` per leg, and the proxy released when the NAMED legs end (it exists for them only).
SECOND="${F22_SECOND_PASS:-}"; case "$SECOND" in 0) SECOND="";; esac
[ -n "$RERUN" ] && [ -n "$SECOND" ] && { echo "FATAL: F22_OVERFLOW_RERUN and F22_SECOND_PASS are exclusive"; exit 2; }
# LEG SELECTION BY CELL: F22_PETRI_CELLS=<comma list of HARNESS cell ids>
# selects the legs from the FROZEN enumeration through scripts/sdfbase/full22/petri_leg_select.py — canonical OBJECT legs only:
# `none`, a prompted leg and the derived controls leg (none__lens0 / none__lora0) refuse BY NAME, so a lane that measures its
# steered objects and pairs against the campaign's EXISTING `none` anchor can never drive an unsteered leg (a fresh `none` audit
# would be a new anchor). F22_PETRI_OBJECTS_ONLY=1 holds an EXPLICIT index list (the 5th arg) to the same rule and refuses ALL.
# Either implies F22_PETRI_RELEASE_SCOPE=selected: the proxy is released when the SELECTED legs have ended (COMPLETE, or
# INCOMPLETE on an overflow ledger; none FAILED) — a leg outside the selection is not this drive's work and never holds the
# GPU. Default (all three unset, legs=ALL) is today's driver byte for byte: every leg driven, the release counted over every leg.
PETRI_CELLS="${F22_PETRI_CELLS:-}"; OBJECTS_ONLY="${F22_PETRI_OBJECTS_ONLY:-}"; case "$OBJECTS_ONLY" in 0) OBJECTS_ONLY="";; esac
# THE LEDGERED SELECTOR (petri-completion, 2026-10-13): legs=ledgered is the OVERFLOW PASS's selector — the grading daemon's argv for the proxy a
# `06 --pass overflow` call submitted. After the frozen record is read, `petri_run.py legs-status` reads every leg the way a resumed drive would,
# and the legs carrying an unresolved CONTEXT-OVERFLOW ledger entry that are neither done nor refused are driven (the predicate _arm_overflow_rerun
# applies per leg, read once up front; empty = FATAL, nothing to re-run). Requires F22_OVERFLOW_RERUN=1; exclusive with F22_PETRI_CELLS /
# F22_PETRI_OBJECTS_ONLY; implies RELEASE_SCOPE=selected. NOT held to petri_leg_select.py's object-only rule: an overflow rerun of a prompted,
# `none` or control leg is the paper's own practice, and `none` is driven only when it is itself ledgered. A ledgered leg this proxy does not
# serve is FATAL by name (the rerun proxy was stood up for other cells), never a skip.
LEDGERED=""; [ "${LEGS:-ALL}" = ledgered ] && LEDGERED=1
if [ -n "$LEDGERED" ]; then
  [ -n "${RERUN:-}" ] || { echo "FATAL: legs=ledgered is the overflow pass's selector: it needs F22_OVERFLOW_RERUN=1"; exit 2; }
  [ -z "$PETRI_CELLS" ] && [ -z "$OBJECTS_ONLY" ] || { echo "FATAL: legs=ledgered is exclusive with F22_PETRI_CELLS / F22_PETRI_OBJECTS_ONLY (the ledgers select the legs)"; exit 2; }
fi
if [ -n "$PETRI_CELLS" ] || [ -n "$OBJECTS_ONLY" ] || [ -n "$LEDGERED" ]; then RELEASE_SCOPE="${F22_PETRI_RELEASE_SCOPE:-selected}"; else RELEASE_SCOPE="${F22_PETRI_RELEASE_SCOPE:-all}"; fi
case "$RELEASE_SCOPE" in all|selected) ;; *) echo "FATAL: F22_PETRI_RELEASE_SCOPE must be all or selected, got '$RELEASE_SCOPE'"; exit 2;; esac
[ -n "$PETRI_CELLS" ] && [ "$LEGS" != ALL ] && { echo "FATAL: F22_PETRI_CELLS and an explicit legs argument are exclusive (legs=$LEGS)"; exit 2; }
[ -n "$OBJECTS_ONLY" ] && [ -z "$PETRI_CELLS" ] && [ "$LEGS" = ALL ] && { echo "FATAL: F22_PETRI_OBJECTS_ONLY=1 with legs=ALL — ALL drives every canonical + controls leg, the unsteered ones included; name the legs (F22_PETRI_CELLS, or an index list)"; exit 2; }
[ "$RELEASE_SCOPE" = selected ] && [ -z "$PETRI_CELLS" ] && [ "$LEGS" = ALL ] && { echo "FATAL: F22_PETRI_RELEASE_SCOPE=selected with legs=ALL — nothing is selected; name the legs"; exit 2; }
# THE TARGET DECODE CONTRACT ON THE SLURM DRIVE (2026-09-21, the qwen35_4b seed views): a manifest that declares
# `target_decode: {mode: vendor, policy: …}` makes petri_run.py REQUIRE `--target-seed-base <s>` on every `run` (the literal
# training seed of the object served) — without it EVERY leg refuses before
# its first sample (16 of 16 did on the first s2/s3 drive, 2026-09-21 06:52Z; the proxies were fine). F22_PETRI_TARGET_SEED_BASE=<s>
# appends `--target-decode vendor --target-seed-base <s>` after `run`, the Modal drive's exact leg argv
# (scripts/modal/seedrep_petri_drive.py leg_argv); the seed base is a RUN knob, never a harness-hash input, so the frozen
# enumeration and the standup's cells_served.json are untouched. Unset = today's argv byte for byte (the harness contract; a
# vendor manifest then still refuses loudly — never an unseeded preset draw under a vendor label).
TARGET_SEED_BASE="${F22_PETRI_TARGET_SEED_BASE:-}"; TARGET_DECODE_ARGS=()
if [ -n "$TARGET_SEED_BASE" ]; then
  case "$TARGET_SEED_BASE" in *[!0-9]*) echo "FATAL: F22_PETRI_TARGET_SEED_BASE must be a non-negative int, got '$TARGET_SEED_BASE'"; exit 2;; esac
  TARGET_DECODE_ARGS=(--target-decode vendor --target-seed-base "$TARGET_SEED_BASE")
fi
# THE CLUSTER'S HOST + ROOT: the cluster's block in the site config (reproduce/lib/clusters.py: CLUSTER_SSH / CLUSTER_SHARED_ROOT,
# looked up by its registry label or its site label, from $SISTEER_CONFIG or $SISTEER_SITE_DIR). The label never stands for a host:
# a cluster the site config does not describe refuses by name.
_CL_PY="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/reproduce/lib/clusters.py"
HOST=""; SHARED=""
HOST="$(python3 "$_CL_PY" --get "$CLUSTER" CLUSTER_SSH)" && SHARED="$(python3 "$_CL_PY" --get "$CLUSTER" CLUSTER_SHARED_ROOT)" || true
[ -n "$HOST" ] && [ -n "$SHARED" ] \
  || { echo "bad cluster '$CLUSTER': the site config gives no login host and shared root (CLUSTER_SSH, CLUSTER_SHARED_ROOT) for it"; exit 2; }
echo "[cluster] $CLUSTER from the site config: ssh=$HOST shared=$SHARED"
TAG="$KEY$( [ "$VIEW" = main ] || echo "_$VIEW" )"
# THE PORT TABLE (paper-companion-repo): configs/sdfbase_full22_ports.yaml is the declared source (tests hold it equal to the
# proxy's table); the case table below stays the fallback for a checkout without the file, byte-for-byte the previous behaviour
_PORTS_YAML="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/configs/sdfbase_full22_ports.yaml"; PPORT=""
[ -f "$_PORTS_YAML" ] && PPORT=$(python3 -c 'import sys,yaml; print((yaml.safe_load(open(sys.argv[1])).get("ports") or {}).get(sys.argv[2], ""))' "$_PORTS_YAML" "$TAG" 2>/dev/null)
if [ -z "$PPORT" ]; then
case "$TAG" in gemma4e4b) PPORT=8981;; gemma4e4b_it) PPORT=8982;; qwen35_4b) PPORT=8983;; qwen35_4b_it) PPORT=8984;; nemotron3nano4b) PPORT=8985;; glm46vflash) PPORT=8986;; llama8b) PPORT=8987;; llama8b_it) PPORT=8988;; llama8b_shortconst) PPORT=8989;; gemma4e4b_halfdose) PPORT=8991;; gemma4e4b_it_halfdose) PPORT=8997;; qwen35_4b_halfdose) PPORT=8993;; qwen35_4b_it_halfdose) PPORT=8994;; nemotron3nano4b_halfdose) PPORT=8995;; glm46vflash_halfdose) PPORT=8996;; qwen35_4b_s1p) PPORT=8975;; qwen35_4b_s2p) PPORT=8976;; qwen35_4b_s3p) PPORT=8977;; qwen35_4b_s1) PPORT=8978;; qwen35_4b_s2) PPORT=8979;; qwen35_4b_s3) PPORT=8980;; *) echo "no port for $TAG"; exit 2;; esac   # the seed-replicate views (qwen-base-seed-replicates): 8978-8980, mirrored from petri_proxy.sbatch
fi
if [ -n "$RERUN" ]; then PPORT=$((PPORT + 100)); fi   # the re-run proxy's port (petri_proxy.sbatch overflow-rerun)
if [ -n "$SECOND" ]; then PPORT=$((PPORT + 200)); fi  # the second concurrent proxy's port (petri_proxy.sbatch second)
SUFFIX="${RERUN:+_rerun}${SECOND:+_second}"
# THE SERVING PATH (native-serving-default): the same ONE variable common.sh f22_init derives the wave root from —
# the native wave writes under sdfbase_full22_native (cluster) / sdfbase_full22_native (grading-host state) and its EVAL
# OUTPUTS under ~/.local/state/petri_native (the root petri_run.py freezes cells.json and writes every leg's
# cell dir beneath); lens paths are byte-identical. This is the ADDRESS; the CHECK is the endpoint's engine, asserted
# below. The eval root MUST follow the engine: the lens audits' cells.json and results.json live under
# ~/.local/state/petri/petri_full22_<tag>, and a native drive sharing that root would skip its own enumeration on the
# lens record, keep the lens legs (the served-model filter cannot tell them apart: base and adapter vmodel names
# overlap), and count the lens results.json as ITS completions — a native audit reported COMPLETE with lens results.
# THE ROSTER: the same F22_ROSTER selection common.sh's jobs read, passed to the bank-view check below (relative to
# the worktree this driver cds into; make_bank_view.py refuses a missing roster by name); default =
# the production roster, byte-identical in effect
F22_ROSTER="${F22_ROSTER:-configs/sdfbase_full22_roster.yaml}"
WT="${F22_WT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
# F22_PY names the interpreter (the grading daemon passes its venv's: the detached checkout it makes at the wave sha has no .venv of its
# own); default: the worktree's own venv. Either must import inspect_ai + petri_bloom (checked below).
PY="${F22_PY:-$WT/.venv/bin/python}"
[ -x "$PY" ] || { echo "FATAL: no python at $PY (F22_PY, else $WT/.venv: uv sync --extra cpu --group petri in the worktree)"; exit 1; }
CFG="configs/sdfbase_full22_${TAG}.yaml"; MANIFEST="configs/capeval/manifests/petri_full22_${TAG}.yaml"
# THE SERVING PATH (kind-aware default, the rule of 2026-09-15): an EXPLICIT F22_SERVING_PATH wins; set NONE, the
# SAME rule the cluster's common.sh f22_init applied — registry_adapter.select_serving_path through
# scripts/sdfbase/full22/serving_path_select.py over the view's manifest cells, by kind — picks the engine here too, so this
# drive reads under the root the cluster job wrote (ROOT_SUFFIX) instead of guessing; the pulled endpoint.json's engine is
# still held to it below. ONE `[serving-path]` line either way (the drive log is not open yet: stdout).
if [ -n "${F22_SERVING_PATH:-}" ]; then
  case "$F22_SERVING_PATH" in lora_native|lens) ;; *) echo "FATAL: F22_SERVING_PATH must be lora_native or lens, got '$F22_SERVING_PATH'"; exit 2;; esac
  F22_SERVING_PATH_MODE=explicit; echo "[serving-path] EXPLICIT $F22_SERVING_PATH (F22_SERVING_PATH set by caller)"
else
  SEL=$(cd "$WT" && "$PY" scripts/sdfbase/full22/serving_path_select.py --manifest "$MANIFEST" --cells-source "manifest:$MANIFEST" --sh 2>&1) \
    || { echo "FATAL: serving-path selection refused (manifest $MANIFEST): $SEL"; exit 2; }
  eval "$(printf '%s\n' "$SEL" | grep -E '^F22_SERVING_PATH_(SELECTED|MODE|LOG|SELECTION_JSON)=')"   # the four assignments only
  [ -n "${F22_SERVING_PATH_SELECTED:-}" ] || { echo "FATAL: serving-path selection returned no verdict (manifest $MANIFEST): $SEL"; exit 2; }
  F22_SERVING_PATH="$F22_SERVING_PATH_SELECTED"; echo "$F22_SERVING_PATH_LOG"
fi
ROOT_SUFFIX="$( [ "$F22_SERVING_PATH" = lens ] || echo _native )"
LOCALPORT=$((PPORT + 10000))
STATE="$HOME/.local/state/sdfbase_full22$ROOT_SUFFIX/petri/$TAG$SUFFIX"; mkdir -p "$STATE"
OUT_ROOT="$HOME/.local/state/petri$ROOT_SUFFIX"; AUDIT="petri_full22_$TAG"; OUT="$OUT_ROOT/$AUDIT"
LOG="$STATE/drive.log"; log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }
# THE ATTEMPT IDENTIFIER: every per-leg status record this invocation writes names the drive that
# wrote it, so a status left by an EARLIER invocation is told apart from this one's in the log and in the record itself
DRIVE_ID="$(date -u +%Y%m%dT%H%M%SZ).$$"
REMOTE_OUT="$SHARED/exp/sdfbase_full22$ROOT_SUFFIX/petri/$TAG/serving$SUFFIX"
export PETRI_PROXY_BASE_URL="http://127.0.0.1:${LOCALPORT}/v1"
# the API host for the op / env legs' keyed SDK clients: an inherited gateway URL would swallow the SDK calls. NOT under wif:
# the mode's contract is that NO ANTHROPIC_* variable reaches a leg -- the vault pins the host
# in-process (anthropic_client_wif; the federated inspect client is built on it) -- so the wif branch below DROPS the inherited
# credential / URL variables by name instead of exporting one.
[ "${SISTEER_SECRETS_MODE:-op}" = wif ] || export ANTHROPIC_BASE_URL=https://api.anthropic.com
# THE SECRET-MANAGER CLI BY ABSOLUTE PATH (2026-09-07): this script runs as a `systemd --user` transient unit, which sources no
# shell profile, so ~/.local/bin is not on its PATH unless the launcher passed one through — the first nemotron driver unit
# died one second in ("command not found") while its proxy held a GPU. Name the helper by absolute path for systemd units;
# never trust the unit's PATH. `op` and `jq` (which it execs) are /usr/bin. PATH is still honoured as a fallback for a machine
# that installs it elsewhere.
# SISTEER_SECRETS_MODE=env (paper-companion-repo, docs/secrets.md): ONE ANTHROPIC_API_KEY serves every org role and the
# secret-manager CLI is never consulted; the variable reaches this unit by name-only `systemd-run --setenv` import (lib/secrets.sh).
# Default (unset / op): the vault path below, byte-for-byte as before.
# SISTEER_SECRETS_MODE=wif: NO Anthropic key anywhere in this unit -- the secret-manager CLI is never consulted; the legs' inspect
# Anthropic provider is routed through workload identity federation (sisteer.eval.anthropic_wif, installed by petri_runner when
# SISTEER_ANTHROPIC_AUTH names the org, exported below) on the SAME vault credentials every judge uses
# (sisteer.vault.wif_credentials). ANTHROPIC_API_KEY must be ABSENT from the caller's environment: an explicit key shadows
# federation even when empty (the grading daemon drops it from the drive's environment under wif; a hand launch that still
# carries one is refused here by name).
SECRETS_MODE="${SISTEER_SECRETS_MODE:-op}"
if [ "$SECRETS_MODE" = wif ]; then
  SECRET_CMD=""
  [ -z "${ANTHROPIC_API_KEY+x}" ] || { echo "FATAL: SISTEER_SECRETS_MODE=wif but ANTHROPIC_API_KEY is present in the environment (it shadows federation even when empty) -- unset it in the caller"; exit 1; }
  # EVERY other inherited ANTHROPIC_* name is DROPPED, never read -- bash's ${!PREFIX@} lists the names (a Claude Code shell
  # carries an OAuth bearer under ANTHROPIC_AUTH_TOKEN, a gateway under ANTHROPIC_BASE_URL and custom headers under
  # ANTHROPIC_CUSTOM_HEADERS, which the SDK parses into every request whatever credential it is given; a vault shell the per-org
  # keys): the same scrub every generated judging unit applies under wif (reproduce/lib/secrets.sh secrets_unit_key_line,
  # launch_judge.sh, grade_sampled.sh; sisteer.vault.WIF_SHELL_SCRUB), so the legs' environment -- the vault digest call below
  # included -- carries no ANTHROPIC_* at all
  unset "${!ANTHROPIC_@}"
elif [ "$SECRETS_MODE" = env ]; then
  SECRET_CMD=""
  [ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "FATAL: SISTEER_SECRETS_MODE=env but ANTHROPIC_API_KEY is unset (one key serves every org role; export it in the caller's environment)"; exit 1; }
else
SECRET_CMD="${SISTEER_SECRET_CMD:-}"
[ -n "$SECRET_CMD" ] || { echo "FATAL: SISTEER_SECRETS_MODE=op but SISTEER_SECRET_CMD is not set: it names the secret-manager CLI the auditor key is fetched through at use time"; exit 1; }
case "$SECRET_CMD" in */*) ;; *) SECRET_CMD="$(command -v -- "$SECRET_CMD" 2>/dev/null || true)";; esac
[ -n "$SECRET_CMD" ] && [ -f "$SECRET_CMD" ] && [ -x "$SECRET_CMD" ] || { echo "FATAL: SISTEER_SECRET_CMD does not name an executable (not found, or not on PATH)"; exit 1; }
fi
# F22_AUDITOR_ORG: which secret-manager org item the run's ONE Anthropic key comes
# from — the harness has a single key per run, so the auditor (sonnet-5) AND the judge (opus-5) move together. Default
# lowprio, the standing routing rule. `highprio` is authorised for Petri runs only, used only after Low Prio has been
# shedding the auditor class for > 2 min AND a same-minute probe shows the other org serving the same model (record both
# probes); move back when a recovery probe shows Low Prio serving again. DRIVE_START records auditor_org for the read-out.
F22_AUDITOR_ORG="${F22_AUDITOR_ORG:-lowprio}"
case "$F22_AUDITOR_ORG" in lowprio|highprio) ;; *) echo "FATAL: F22_AUDITOR_ORG must be lowprio or highprio, got '$F22_AUDITOR_ORG'"; exit 2;; esac
if [ "$SECRETS_MODE" = wif ]; then
  # the org's federation ids file is validated and digested by the vault -- the ONE reader of the workload-identity env file -- so an absent /
  # empty / incomplete file is a one-line FATAL here, before any leg; the digest (never an id) is what DRIVE_START attests below. $PY is the
  # leg interpreter (F22_PY); PYTHONPATH reaches this worktree's src first (the daemon sets it to its checkout's; a hand launch may rely on
  # the venv's install).
  AUDITOR_IDS_SHA12=$(cd "$WT" && PYTHONPATH="$WT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" -c 'import sys
from sisteer.vault import PoolUnavailable, wif_ids_sha12
try:
    print(wif_ids_sha12(sys.argv[1]))
except (PoolUnavailable, ValueError) as exc:
    print(exc); sys.exit(1)' "$F22_AUDITOR_ORG") || { echo "FATAL: SISTEER_SECRETS_MODE=wif: the $F22_AUDITOR_ORG pool's federation ids do not resolve (sisteer.vault.wif_ids_sha12): $AUDITOR_IDS_SHA12"; exit 1; }
  [[ "$AUDITOR_IDS_SHA12" =~ ^[0-9a-f]{12}$ ]] || { echo "FATAL: SISTEER_SECRETS_MODE=wif: sisteer.vault.wif_ids_sha12($F22_AUDITOR_ORG) printed no digest: $AUDITOR_IDS_SHA12"; exit 1; }
  export SISTEER_ANTHROPIC_AUTH="wif:$F22_AUDITOR_ORG"
  echo "[secrets] wif mode: no Anthropic key in this unit and no inherited ANTHROPIC_* variable (ANTHROPIC_AUTH_TOKEN / ANTHROPIC_BASE_URL / the per-org keys dropped by name); every leg's inspect Anthropic client federates as org $F22_AUDITOR_ORG (SISTEER_ANTHROPIC_AUTH; ids file sha256[:12] $AUDITOR_IDS_SHA12)"
elif [ "$SECRETS_MODE" = env ]; then export ANTHROPIC_API_KEY; echo "[secrets] env mode: ANTHROPIC_API_KEY from the environment serves org role $F22_AUDITOR_ORG"
else ANTHROPIC_API_KEY="$("$SECRET_CMD" "$F22_AUDITOR_ORG")" || { echo "FATAL: the secret-manager CLI (SISTEER_SECRET_CMD) failed for $F22_AUDITOR_ORG"; exit 1; }; export ANTHROPIC_API_KEY; fi
# THE ATTESTATION: DRIVE_START records sha256[:12] of the key this drive actually resolved (never the key), so the pool it names is
# PROVEN against the grading daemon's own sha256sum of the secret it fetches for that org (petri_expected_key_sha12 / _petri_attest_key),
# not asserted by the F22_AUDITOR_ORG variable: an inherited key under SISTEER_SECRETS_MODE=env, a stale vault item or a drive-side override
# would otherwise log the org it was asked for while calling on another pool. Under wif there is no key to attest: DRIVE_START records
# `auditor_auth=wif:<org> ids_sha12=<sha256(ids file)[:12]>` instead, the daemon compares the digest with the vault's own over the same
# org's file, and the Petri reader validates the digest rather than trusting auditor_org.
if [ "$SECRETS_MODE" = wif ]; then AUDITOR_ATTEST="auditor_auth=wif:$F22_AUDITOR_ORG ids_sha12=$AUDITOR_IDS_SHA12"
else AUDITOR_KEY_SHA12=$(printf %s "$ANTHROPIC_API_KEY" | sha256sum | cut -c1-12); AUDITOR_ATTEST="auditor_key_sha12=$AUDITOR_KEY_SHA12"; fi
export PETRI_TWO_SI_CONFIG="$CFG"
# the grading host bank VIEW (make_bank_view.py --bank $GRADING_HOST_BANK_ROOT/<key>): the adapter validates every staged object at
# enumeration, so the driver needs the same view the proxy served from. THE BANK ROOT (demo-import, 2026-10-11): F22_GRADING_HOST_BANK_ROOT
# names the root the view is read under; unset, the paper's legacy the grading host bank, byte for byte. The grading daemon hands its own
# --petri-bank-root here (grade_daemon.py petri_drive_pass: F22_GRADING_HOST_BANK_ROOT=<bank root>), so the view the daemon BUILDS before the
# start is the view this drive READS — the front door's demo staged its objects into $SISTEER_OBJECTS_BANK (05_stage_roster.sh --bank)
# while this script read the legacy root, and the daemon reported `petri_drive_unprepared` against a bank the demo never wrote.
# unset, the bank under the site's staging directory ($STAGE), expanded through sisteer.site; where the site names no $STAGE the
# placeholder stays, and the bank-view check below refuses by name
GRADING_HOST_BANK_ROOT="${F22_GRADING_HOST_BANK_ROOT:-}"
if [ -z "$GRADING_HOST_BANK_ROOT" ]; then
  GRADING_HOST_BANK_ROOT=$(cd "$WT" && PYTHONPATH="$WT/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m sisteer.site expand '$STAGE/sdfbase/full22_objects' 2>/dev/null) \
    || GRADING_HOST_BANK_ROOT='$STAGE/sdfbase/full22_objects'
  GRADING_HOST_BANK_ROOT="${GRADING_HOST_BANK_ROOT/#\~/$HOME}"
fi
export TT_TWO_SI_DEFENSES_DIR="$GRADING_HOST_BANK_ROOT/$KEY/cells_view/$VIEW"
export PETRI_PROMPT_OVERRIDE_DIR="$WT/configs/capeval/petri/prompts"
export PETRI_BUNDLE_DIR="$HOME/.local/state/petri/scenarios/8812ba083bb5473c"
export PETRI_OUT_ROOT="$OUT_ROOT" PETRI_WORK_DIR="$OUT" PETRI_LOG_STORE="$OUT/inspect_logs"
export PETRI_EVAL_MAX_CONNECTIONS="${PETRI_EVAL_MAX_CONNECTIONS:-1}"
export SISTEER_LENS_STACKED_REDUCTIONS=1     # the client-side contract: UNSET reads as per_row elsewhere
# THE AUDITOR FALLBACK CHAIN IS RESOLVED EAGERLY (auditor_fallback.resolve_inspect_auditor_models): every leg of
# sonnet-5 -> openrouter/grok-4.6 -> the local abliterated Qwen must resolve BEFORE the first sample, or the leg
# dies with AuditorFallbackError three attempts in a row (the 2026-09-07 launch: all 28 legs INCOMPLETE within
# 3 minutes, proxies left idle for an hour). Two things it needs from this machine, checked here so the failure
# is a one-line refusal instead of 28 tracebacks:
#  1. the OpenRouter key for the grok leg: auditor_fallback.resolve_openrouter_api_key takes it from the
#     environment (OPENROUTER_API_KEY, which the vault bootstrap exports in every shell here) or else from
#     the secret manager's vault item (sisteer.vault) — NEVER a file on disk (the repo's credential-hygiene
#     tests forbid any key file). Refused here when unset so the failure is a one-line refusal instead of
#     28 tracebacks.
#  2. PETRI_QWEN_AUDITOR_BASE_URL: an OpenAI-compatible endpoint serving huihui-ai/Huihui-Qwen3.6-27B-abliterated.
#     The wave's attacker pool serves exactly that model, so a tunnel to one member is the endpoint:
#       ssh -N -L 18990:<attacker node>:<port> <cluster B login host> &   ->   http://127.0.0.1:18990/v1
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "FATAL: OPENROUTER_API_KEY is unset — the auditor chain's grok leg needs it (see the note above): export it in the caller's environment, never from a file"; exit 1; }
# The abliterated-Qwen third leg was removed from the chain 2026-09-15, so there is no endpoint to
# require and no GPU to stand up for it. This used to be a hard precondition -- a Petri run refused to start
# unless a tunnel reached a live attacker-pool member serving that model -- which meant a pool had to exist for
# a leg that was reached ZERO times in 21,399 audited rows. The chain is now sonnet-5 -> grok-4.6.
cd "$WT"
# PROVENANCE: the driver reads the config + manifest + petri_run.py + src/ from THIS worktree: it must BE
# the launched commit, clean (a post-launch edit must never change what "complete" means)
[ "$(git rev-parse HEAD)" = "$WAVE_SHA" ] || { log "FATAL: worktree HEAD $(git rev-parse --short HEAD) != launched $WAVE_SHA — git checkout --detach $WAVE_SHA"; exit 1; }
git diff --quiet HEAD -- configs scripts src || { log "FATAL: worktree dirty under configs/scripts/src"; exit 1; }
[ -d "$TT_TWO_SI_DEFENSES_DIR" ] || { log "FATAL: no grading host bank view at $TT_TWO_SI_DEFENSES_DIR (bank root $GRADING_HOST_BANK_ROOT${F22_GRADING_HOST_BANK_ROOT:+ from F22_GRADING_HOST_BANK_ROOT}; make_bank_view.py --bank $GRADING_HOST_BANK_ROOT/$KEY --target $KEY --view $VIEW --roster $F22_ROSTER)"; exit 1; }
# a host with no scenario store: the checkout publishes the frozen bundle at data/petri/scenarios/<hash> (shipped for exact
# reproduction), and its own loader installs it here, verified file by file against the lock and laid into a mode-0700 read-only store
# (an existing store is verified, never rewritten); the grading daemon does the same before a start (grade_daemon.petri_install_bundle)
if [ ! -d "$PETRI_BUNDLE_DIR" ] && [ -d "$WT/data/petri/scenarios/${PETRI_BUNDLE_DIR##*/}" ]; then
  (cd "$WT" && PYTHONPATH="$WT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" -m sisteer.capeval.petri.scenarios install --dest "$PETRI_BUNDLE_DIR") >>"$LOG" 2>&1 \
    || { log "FATAL: no bundle copy at $PETRI_BUNDLE_DIR, and the checkout's data/petri/scenarios copy did not install (the refusal is in $LOG)"; exit 1; }
  log "installed the scenario bundle at $PETRI_BUNDLE_DIR from the checkout's data/petri/scenarios"
fi
[ -d "$PETRI_BUNDLE_DIR" ] || { log "FATAL: no bundle copy at $PETRI_BUNDLE_DIR"; exit 1; }
"$PY" -c 'import inspect_ai, petri_bloom' || { log "FATAL: the venv lacks inspect_ai/petri_bloom"; exit 1; }
"$PY" scripts/sdfbase/full22/make_bank_view.py --check --bank "$GRADING_HOST_BANK_ROOT/$KEY" --target "$KEY" --view "$VIEW" --roster "$F22_ROSTER" >>"$LOG" 2>&1 || { log "FATAL: the grading host bank view $TT_TWO_SI_DEFENSES_DIR does not verify against the pins (roster $F22_ROSTER; bank root $GRADING_HOST_BANK_ROOT)"; exit 1; }
log "DRIVE_START $TAG drive_id=$DRIVE_ID cluster=$CLUSTER serving_path=$F22_SERVING_PATH serving_path_mode=$F22_SERVING_PATH_MODE roster=$F22_ROSTER bank_root=$GRADING_HOST_BANK_ROOT remote_out=$REMOTE_OUT out=$OUT legs=[$LEGS] target_decode=$( [ -n "$TARGET_SEED_BASE" ] && echo "vendor(seed_base=$TARGET_SEED_BASE; the manifest's target_decode row, asserted on every run)" || echo "harness-drawn(petri exposes no sampler; labelled, never the vendor preset)") width=$PETRI_EVAL_MAX_CONNECTIONS auditor_org=$F22_AUDITOR_ORG $AUDITOR_ATTEST secrets_mode=$SECRETS_MODE retry_sleep_cap_s=${PETRI_AUDITOR_RETRY_SLEEP_CAP_S:-1800} sha=$WAVE_SHA${RERUN:+ OVERFLOW-RERUN (worktree $WT; eval width = rerun_serving_geometry max_num_seqs)}${SECOND:+ SECOND PASS (port $PPORT, state $STATE)}${PETRI_CELLS:+ CELLS=[$PETRI_CELLS] (F22_PETRI_CELLS: object legs by name)}${OBJECTS_ONLY:+ OBJECTS-ONLY} release_scope=$RELEASE_SCOPE"
# 1. the proxy endpoint; its job must be RUNNING (a stale endpoint.json = live driver, dead engine)
EP=$(ssh -o ConnectTimeout=20 "$HOST" "cat $REMOTE_OUT/endpoint.json" 2>>"$LOG") || { log "FATAL: no endpoint.json at $HOST:$REMOTE_OUT"; exit 1; }
echo "$EP" > "$STATE/endpoint.json"
NODE=$(echo "$EP" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["node"])')
JOBID=$(echo "$EP" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["slurm_job_id"])')
MODE=$(echo "$EP" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["mode"])')
EPM=$(echo "$EP" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("lens_reduction_mode"))')
EPE=$(echo "$EP" | "$PY" -c 'import json,sys; print(json.load(sys.stdin).get("engine"))')
[ "$MODE" = petri ] || { log "FATAL: endpoint mode=$MODE, not petri"; exit 1; }
# THE ENGINE, read off the pulled endpoint.json (native-serving-default): the asked-for path must be what the standup
# attested — a lens record under the native address (or the reverse) is refused, never re-labelled
VERIFY=()
if [ "$F22_SERVING_PATH" = lora_native ]; then
  [ "$EPE" = lora_native ] || { log "FATAL: F22_SERVING_PATH=lora_native ($F22_SERVING_PATH_MODE) but endpoint.json engine=$EPE (lens_reduction_mode=$EPM) — the cluster job stood up the other engine; set F22_SERVING_PATH explicitly to drive it"; exit 1; }
  [ "$EPM" = None ] || { log "FATAL: a native endpoint.json carries lens_reduction_mode=$EPM — two engines on one record"; exit 1; }
  # the standup's EVIDENCE the client is held to (petri_run.py --verify-standup / --verify-routing-table): the version-2
  # routing table and the final cells_served.json, pulled beside the endpoint from the paths it names
  RT_REMOTE=$(echo "$EP" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["routing_table"])')
  CS_REMOTE=$(echo "$EP" | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["cells_served"])')
  ssh -o ConnectTimeout=20 "$HOST" "cat $RT_REMOTE" > "$STATE/routing_table.json" 2>>"$LOG" || { log "FATAL: no routing table at $HOST:$RT_REMOTE"; exit 1; }
  ssh -o ConnectTimeout=20 "$HOST" "cat $CS_REMOTE" > "$STATE/cells_served.json" 2>>"$LOG" || { log "FATAL: no cells_served.json at $HOST:$CS_REMOTE"; exit 1; }
  "$PY" -c 'import json,sys; t=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2])); assert t.get("version")==2 and t.get("adapters"), "not a version-2 table with adapter blocks"; assert c.get("engine")=="lora_native" and c.get("stage")!="pre_launch", "cells_served.json is not the FINAL native record"' "$STATE/routing_table.json" "$STATE/cells_served.json" 2>>"$LOG" || { log "FATAL: the pulled standup evidence is not a native version-2 table + final cells_served.json"; exit 1; }
  SKIPPED=$("$PY" -c 'import json,sys; print(",".join(s.get("cell_id","?") for s in (json.load(open(sys.argv[1])).get("cells_skipped") or [])))' "$STATE/cells_served.json")
  [ -z "$SKIPPED" ] || log "cells SKIPPED BY NAME on the native engine (never audited here): $SKIPPED"
  VERIFY=(--verify-standup "$STATE/cells_served.json" --verify-routing-table "$STATE/routing_table.json")
else
  [ "$EPE" = None ] || { log "FATAL: F22_SERVING_PATH=lens ($F22_SERVING_PATH_MODE) but endpoint.json engine=$EPE (a native standup under the lens address) — set F22_SERVING_PATH explicitly to drive it"; exit 1; }
  [ "$EPM" = stacked ] || { log "FATAL: endpoint.json lens_reduction_mode=$EPM, not stacked"; exit 1; }
fi
# THE RE-RUN PASS'S ATTESTED WINDOW ON EITHER ENGINE: the pulled endpoint.json carries attested_max_model_len for the
# lens and the native standup alike; the lens path pulls no cells_served.json, so every armed leg reads the attestation from the endpoint
# record (--served-endpoint; the native path hands both and petri_run.py holds them equal). An endpoint that attests no window fails the
# drive HERE, before any leg — not leg by leg at the runner's refusal.
SERVED_EP=()
if [ -n "$RERUN" ]; then
  ATT_L=$(echo "$EP" | "$PY" -c 'import json,sys; v=json.load(sys.stdin).get("attested_max_model_len"); print(v if isinstance(v, int) and not isinstance(v, bool) and v >= 16 else "")')
  [ -n "$ATT_L" ] || { log "FATAL: overflow re-run on the $F22_SERVING_PATH path, but endpoint.json attests no max_model_len (attested_max_model_len missing) — the re-run proxy did not attest its window; no leg is driven"; exit 1; }
  SERVED_EP=(--served-endpoint "$STATE/endpoint.json")
  log "overflow re-run: the proxy attests max_model_len=$ATT_L (endpoint.json; handed to every leg as --served-endpoint)"
fi
SERVED_MODELS=$(echo "$EP" | "$PY" -c 'import json,sys; print(",".join(json.load(sys.stdin).get("served_models") or []))')
ST=$(ssh -o ConnectTimeout=20 "$HOST" "squeue -j $JOBID -h -o %T" 2>/dev/null || true)
[ "$ST" = RUNNING ] || { log "FATAL: proxy job $JOBID is '$ST', not RUNNING"; exit 1; }
# 1b. THE REGISTRY HAND-OFF: the proxy job registered ONE run per
# (eval, object) on the cluster at its start (f22_registry_register -> $SHARED/registry/pending/$JOBID.json) and cannot know the
# grading-host store those runs get; THIS drive is the producer of that store, so it imports the proxy's pending identities
# (petri_registry_handoff.py import: the wave config's cell entries re-spelled to the Petri harness cell ids the stores name —
# registry_adapter.harness_cell_id, the manifest's own translation) and exports the run identity + the grading host's spool to
# every petri_run.py leg, whose CellStore's default recorder then routes each row onto the registered run of ITS (object,
# behaviour) and files `store_uri` + `generated` at results.json; the pooled run per cell is filed by `close-leg` below, and
# `finish` pushes at the end. A proxy that registered nothing is a FATAL here (its stores could never reach a run) unless
# F22_REGISTRY_HANDOFF=off names the exception (an audit outside the registry — said in the log, never silent), or the registry
# itself is OFF (SISTEER_REGISTRY_DISABLED, the emitters' own switch — then no store recorder could file a row anyway; below).
REG_HANDOFF="${F22_REGISTRY_HANDOFF:-on}"
REG_PEND_DIR="$STATE/registry/pending"; REG_IDENTITY=""; HANDOFF_MODE=on
# The emitters' OFF switch (sisteer.registry.emit.DISABLED_ENV; hooks.emission_enabled reads 1|true|yes|on, case-insensitive):
# with it set EVERY store recorder is inert, so no row of this audit could reach a run whether or not the proxy registered one —
# the hand-off stands down and names the switch instead of the FATAL, which is a defect only of a registry-ENABLED audit whose
# proxy registered nothing (the test suite runs every child under SISTEER_REGISTRY_DISABLED=1).
# Production — registry enabled, the variable unset — is unchanged: the FATAL below still fires there.
registry_disabled(){ case "$(printf '%s' "${SISTEER_REGISTRY_DISABLED:-}" | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')" in 1|true|yes|on) return 0;; *) return 1;; esac; }
if [ "$REG_HANDOFF" = off ]; then
  HANDOFF_MODE=off
  log "registry hand-off OFF (F22_REGISTRY_HANDOFF=off): the stores of this audit are recorded on NO registry run"
elif registry_disabled; then
  HANDOFF_MODE=disabled
  log "registry hand-off OFF (SISTEER_REGISTRY_DISABLED=${SISTEER_REGISTRY_DISABLED:-}, the emitters' switch — every store recorder is inert): the proxy's pending identities are not imported and the stores of this audit are recorded on NO registry run"
else
  mkdir -p "$REG_PEND_DIR"
  if ssh -o ConnectTimeout=20 -o BatchMode=yes "$HOST" "cat $SHARED/registry/pending/$JOBID.json" > "$REG_PEND_DIR/$JOBID.cluster.json" 2>>"$LOG" \
     && [ -s "$REG_PEND_DIR/$JOBID.cluster.json" ]; then
    REG_SH=$(cd "$WT" && "$PY" scripts/sdfbase/full22/petri_registry_handoff.py import --pending "$REG_PEND_DIR/$JOBID.cluster.json" --job "$JOBID" --out-dir "$REG_PEND_DIR" --sh 2>>"$LOG") \
      || { log "FATAL: the registry hand-off refused the proxy's pending identities (see $LOG; petri_registry_handoff.py import) — the audit would produce stores no registered run names"; exit 1; }
    eval "$REG_SH"; export SISTEER_REGISTRY_RUN_IDENTITY SISTEER_REGISTRY_SPOOL; REG_IDENTITY="$SISTEER_REGISTRY_RUN_IDENTITY"
    log "registry hand-off: proxy job $JOBID's $("$PY" -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["runs"]))' "$REG_IDENTITY") run(s) imported -> $REG_IDENTITY; spool $SISTEER_REGISTRY_SPOOL (the grading host's)"
  else
    rm -f "$REG_PEND_DIR/$JOBID.cluster.json"
    log "FATAL: proxy job $JOBID registered no runs ($HOST:$SHARED/registry/pending/$JOBID.json absent) — every leg's store would be recorded on no run; fix the proxy's registration ([registry] lines in its log), or F22_REGISTRY_HANDOFF=off for an audit outside the registry"; exit 1
  fi
fi
# 2. the tunnel: REFUSE a held local port; open; verify the vmodels AND the served mode from /v1/models
# The holder of our local port is judged by the EXACT node and proxy port it forwards to (petri_tunnel.sh), never by a substring —
# `*" $NODE"*` also matches a node whose name merely extends ours, so the previous attempt's tunnel to a longer-named node (the
# cancelled 863015) was read as this one's, and /v1/models was read off a dead forward twice ("lacks the native wave's vmodels: <empty>").
# A stale forward of ours (same local port, other node or proxy port) is closed and the tunnel re-created for THIS attempt; a holder that
# is not an ssh forward is still refused; and the tunnel this run opens is torn down on every exit (trap), so no later attempt inherits a
# forward it did not verify. The verdict is for THIS local port — it validates the one forward that binds LOCALPORT (the old verdict kept
# the last -L of a multi-forward line and could read `reuse` while OUR port forwarded to the re-run pass's proxy); a holder carrying more
# than one forward is never the drive's tunnel and is refused by name, neither reused nor closed. The holder is the SOCKET's owner
# (`f22_tunnel_holder_line`: ss -ltnp + that pid's argv), never a command-line search for one spelling of the forward — the old substring
# search for `-L <port>:127.0.0.1:` never FOUND the bind-address and glued forms the verdict accepts, so a legitimate tunnel in either
# form left the port "held by something else" and the drive down.
. "$(dirname "${BASH_SOURCE[0]}")/petri_tunnel.sh" || { log "FATAL: petri_tunnel.sh is missing beside the drive ($(dirname "${BASH_SOURCE[0]}")) — the tunnel verdicts live there (D1503)"; exit 2; }
TUNNEL_PID=""   # set only when THIS run opens the tunnel; a reused one belongs to whoever opened it
trap 'f22_tunnel_teardown' EXIT
OPEN_TUNNEL=1
if ss -ltn 2>/dev/null | grep -q ":${LOCALPORT} "; then
  OLD=$(f22_tunnel_holder_line "$LOCALPORT")
  case "$(f22_tunnel_holder_verdict "$OLD" "$NODE" "$PPORT" "$LOCALPORT")" in
    reuse) log "tunnel to $NODE:$PPORT already up on $LOCALPORT — reusing"; OPEN_TUNNEL=0 ;;
    stale) log "stale tunnel on $LOCALPORT ($OLD) does not forward to $NODE:$PPORT — closing it and re-creating the tunnel for this attempt (D1503)"
           f22_tunnel_close_stale "$OLD" "$LOCALPORT" || { log "FATAL: port $LOCALPORT still held after closing the stale tunnel ($(f22_tunnel_holder_line "$LOCALPORT")) — free it first"; exit 1; } ;;
    ambiguous) log "FATAL: port $LOCALPORT is held by an ssh carrying more than one forward ($OLD) — not this drive's single-forward tunnel: the forward that binds $LOCALPORT need not be one to $NODE:$PPORT, and its other forwards are not ours to close; free it first (D1642)"; exit 1 ;;
    *) log "FATAL: port $LOCALPORT held by something else ($OLD) — free it first"; exit 1 ;;
  esac
fi
if [ "$OPEN_TUNNEL" = 1 ]; then
  # a proxy scheduled ON the login host itself (Slurm places jobs there too) must not jump through itself:
  # `ssh -J <host> <host>` never came up (2026-09-07, the gemma4e4b_it proxy 435476 landed on the login host -> an
  # empty /v1/models and a FATAL before any leg launched); tunnel directly when the proxy's node IS the jump host
  if [ "$NODE" = "$HOST" ]; then JUMP=(); else JUMP=(-J "$HOST"); fi
  setsid ssh -N -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o ExitOnForwardFailure=yes -o StrictHostKeyChecking=accept-new -o BatchMode=yes \
    -L ${LOCALPORT}:127.0.0.1:${PPORT} "${JUMP[@]}" "$NODE" >"$STATE/tunnel.log" 2>&1 &
  TUNNEL_PID=$!
  for i in $(seq 1 30); do curl -s -m 5 "http://127.0.0.1:${LOCALPORT}/v1/models" 2>/dev/null | grep -q "petri__" && break; sleep 3; done
fi
curl -s -m 10 "http://127.0.0.1:${LOCALPORT}/v1/models" > "$STATE/models.json" 2>/dev/null
if [ "$F22_SERVING_PATH" = lora_native ]; then
  # the native proxy: the zero-adapter control leg on the vmodel list, the engine block naming the native engine
  grep -q "petri__none__lora0" "$STATE/models.json" || { log "FATAL: /v1/models lacks the native wave's vmodels (petri__none__lora0): $(head -c 400 "$STATE/models.json")"; exit 1; }
  PXK=$("$PY" -c 'import json,sys; print((((json.load(open(sys.argv[1])).get("petri_proxy") or {}).get("engine") or {}).get("kind")))' "$STATE/models.json")
  [ "$PXK" = lora_native ] || { log "FATAL: /v1/models petri_proxy.engine.kind=$PXK, not lora_native — refusing to audit"; exit 1; }
  log "tunnel READY on $LOCALPORT -> $NODE:$PPORT; vmodels verified; served engine lora_native (standup evidence: $STATE/cells_served.json + routing_table.json)"
else
  grep -q "petri__none__lens0" "$STATE/models.json" || { log "FATAL: /v1/models lacks the wave's vmodels: $(head -c 400 "$STATE/models.json")"; exit 1; }
  PXM=$("$PY" -c 'import json,sys; print((json.load(open(sys.argv[1])).get("petri_proxy") or {}).get("lens_reduction_mode"))' "$STATE/models.json")
  [ "$PXM" = stacked ] || { log "FATAL: /v1/models petri_proxy.lens_reduction_mode=$PXM, not stacked — refusing to audit"; exit 1; }
  log "tunnel READY on $LOCALPORT -> $NODE:$PPORT; vmodels verified; served lens_reduction_mode=stacked"
fi
# 3. FREEZE the enumeration — under the ENGINE's policy (petri_run.py --serving-path: native strips the manifest's
# eager pin the way the standup did and derives none__lora0; lens is today's enumeration) and, under native, held to the
# standup's evidence (--verify-standup / --verify-routing-table refuse before any leg on a vmodel / identity / hash /
# run-key disagreement)
PETRI_RUN_OPTS=(--manifest "$MANIFEST" --serving-path "$F22_SERVING_PATH" ${VERIFY[@]+"${VERIFY[@]}"} ${SERVED_EP[@]+"${SERVED_EP[@]}"})
# an EXISTING frozen record must have been written under THIS engine's policy: a native record names
# serving_path: lora_native, a lens record predates the key (absent = lens by era, the rule petri_run.py reads under).
# petri_run.py refuses the mismatch per leg, but the leg SELECTION and the COMPLETION COUNT below read cells.json
# directly — so the other engine's record is refused HERE, before any leg, never resolved against
if [ -f "$OUT/cells.json" ]; then
  REC_PATH=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("serving_path", "lens"))' "$OUT/cells.json") || REC_PATH="unreadable"
  [ "$REC_PATH" = "$F22_SERVING_PATH" ] || { log "FATAL: $OUT/cells.json was frozen under --serving-path $REC_PATH but this drive is --serving-path $F22_SERVING_PATH — refusing to select legs or count completions against the other engine's enumeration (re-freeze into a fresh out root)"; exit 1; }
fi
[ -f "$OUT/cells.json" ] || "$PY" scripts/petri_run.py "${PETRI_RUN_OPTS[@]}" print-cells --out-root "$OUT" >>"$LOG" 2>&1 || { log "FATAL: print-cells failed"; exit 1; }
NLEGS=$("$PY" -c 'import json,sys;d=json.load(open(sys.argv[1]));print(len(d["canonical"])+len(d["controls"]))' "$OUT/cells.json")
log "enumeration frozen: $NLEGS legs (canonical + controls)"
if [ -n "$LEDGERED" ]; then
  # the LEDGERED legs from the drive's OWN done-ness reading (petri-completion): legs-status over the frozen record — done / refused / the
  # unresolved context-overflow entries per leg — the same predicate _arm_overflow_rerun applies when the leg runs; read once, before any leg
  LS_JSON=$("$PY" scripts/petri_run.py "${PETRI_RUN_OPTS[@]}" legs-status --out-root "$OUT" 2>>"$LOG") \
    || { log "FATAL: petri_run.py legs-status refused over $OUT (see $LOG) — the ledgered legs cannot be read"; exit 1; }
  LEGS=$(printf '%s' "$LS_JSON" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); print(" ".join(str(l["idx"]) for l in d["legs"] if int(l.get("overflow_unresolved") or 0) > 0 and not l["done"] and not l["refused"]))')
  [ -n "$LEGS" ] || { log "FATAL: no leg carries an unresolved context-overflow entry (petri_run.py legs-status over $OUT) — nothing to re-run"; exit 1; }
  log "legs selected as LEDGERED from the frozen enumeration (petri_run.py legs-status over $OUT: an unresolved context-overflow entry, not done, not refused): [$LEGS]"
fi
if [ -n "$PETRI_CELLS" ]; then
  # BY CELL from the frozen record (petri_leg_select.py): object legs only, refusals by name — never an index guessed from a roster
  LEGS=$("$PY" scripts/sdfbase/full22/petri_leg_select.py "$OUT/cells.json" --cells "$PETRI_CELLS" 2>>"$LOG") \
    || { log "FATAL: petri_leg_select.py refused F22_PETRI_CELLS=$PETRI_CELLS against $OUT/cells.json (see $LOG)"; exit 1; }
  log "legs selected BY CELL from the frozen enumeration (F22_PETRI_CELLS=$PETRI_CELLS): [$LEGS]"
fi
if [ -n "$OBJECTS_ONLY" ] && [ "$LEGS" != ALL ]; then
  "$PY" scripts/sdfbase/full22/petri_leg_select.py "$OUT/cells.json" --indices "$LEGS" --objects-only >/dev/null 2>>"$LOG" \
    || { log "FATAL: F22_PETRI_OBJECTS_ONLY=1 but the selected legs [$LEGS] name \`none\`, a prompted leg or the controls leg (see $LOG)"; exit 1; }
  log "selected legs [$LEGS] verified object-only against the frozen enumeration"
fi
[ "$LEGS" = "ALL" ] && LEGS=$(seq -s ' ' 0 $((NLEGS-1)))
SELECTED_LEGS="$LEGS"   # the release scope under F22_PETRI_RELEASE_SCOPE=selected (the served-models filter below only narrows it)
# the legs this proxy SERVES (native-serving-default): a leg whose vmodel is not in the endpoint's served_models (a cell the
# standup skipped by name) is never driven — logged, never a 404 read as a failed audit — and never counted against the
# release below (it is not this proxy's work; it would otherwise read as FAILED and hold the GPU until walltime)
leg_vmodel(){ "$PY" -c 'import json,sys;d=json.load(open(sys.argv[1]));r=d["canonical"]+d["controls"];print(r[int(sys.argv[2])].get("virtual_model_name",""))' "$OUT/cells.json" "$1"; }
leg_unserved(){   # $1 = leg index -> 0 when the endpoint names served_models and this leg's vmodel is NOT among them
  [ -n "$SERVED_MODELS" ] || return 1
  local vm; vm=$(leg_vmodel "$1")
  [ -n "$vm" ] || return 1
  case ",$SERVED_MODELS," in *",$vm,"*) return 1;; *) return 0;; esac
}
if [ -n "$SERVED_MODELS" ]; then
  KEPT=""; for idx in $LEGS; do
    vm=$(leg_vmodel "$idx")
    if [ -z "$vm" ]; then
      [ "$F22_SERVING_PATH" = lora_native ] && { log "FATAL: cells.json leg $idx names no virtual_model_name — cannot hold it to served_models"; exit 1; }
      KEPT="$KEPT $idx"
    elif leg_unserved "$idx"; then
      [ -z "$LEDGERED" ] || { log "FATAL: ledgered leg $idx ($vm) is NOT served by this proxy (served_models: $SERVED_MODELS) — the rerun proxy was stood up for other cells (06 --pass overflow derives them from the ledgers); nothing driven"; exit 1; }
      log "leg $idx ($vm) is NOT served by this proxy (served_models: $SERVED_MODELS) — skipped"
    else
      KEPT="$KEPT $idx"
    fi
  done; LEGS="${KEPT# }"
  [ -n "$LEGS" ] || { log "FATAL: none of the requested legs is served by this proxy"; exit 1; }
fi
# (ported from origin/full22 5ac0d434, which the cherry-picked re-run commit 3dedc0bd assumed)
# unresolved entries in a cell's H9 failure ledger (0 when absent or clean); the ledger's canonical shape is a
# mapping with an "unresolved" list (petri_run._unresolved_ledger_entries)
f22_unresolved(){ "$PY" -c 'import json,sys,os
p=os.path.join(sys.argv[1],"petri_incomplete.json")
try: d=json.load(open(p)); u=d.get("unresolved"); print(len(u) if isinstance(u,list) else 0)
except Exception: print(0)' "$1"; }
# 4. the legs, concurrently at width 1 (per-cell flock; resume by run key; rc=3 = INCOMPLETE/overflow)
# A leg's driver return lands in $STATE/leg_<idx>.rc: the completion count below reads it, so a leg
# whose driver REFUSED (an engine mismatch, an enumeration drift, a hash disagreement — a verdict about this run, not a
# transient) is FAILED however the cell dir looks; an old results.json never overrides a failed return. The record is
# ONE line, `<rc> drive=<DRIVE_ID> results=<sha256 of the cell's results.json at that return | absent>`: the return is
# bound to the attempt that produced it and to the bytes it returned over. It is written ONLY
# when THIS invocation's run of that leg ends, and NEVER cleared up front: an earlier version cleared every leg_*.rc
# here, so an invocation re-driving a SUBSET (legs "1") erased leg 0's FAILED return from the previous invocation, and
# the count below then read leg 0's stale results.json as a completion and released the proxy over an unaudited leg. A
# failed status now stands until that same leg is driven again and returns; a leg no invocation of this state dir ever
# returned on (no record at all) is judged by its cell dir, as before, and the count says so by name.
leg_status_write(){   # $1 = leg index, $2 = rc, $3 = the cell dir -> the one-line status record, bound to this drive
  local sha=absent
  [ -f "$3/results.json" ] && sha=$(sha256sum "$3/results.json" | cut -c1-64)
  echo "$2 drive=$DRIVE_ID results=$sha" > "$STATE/leg_${1}.rc"
}
leg_status_read(){    # $1 = leg index -> LRC LDRIVE LRES (empty when no record; a bare-rc record from an older writer reads too)
  LRC=""; LDRIVE=""; LRES=""
  [ -f "$STATE/leg_${1}.rc" ] || return 0
  local f2 f3
  read -r LRC f2 f3 < "$STATE/leg_${1}.rc" || true
  LDRIVE="${f2#drive=}"; LRES="${f3#results=}"
}
LEG_PIDS=""
run_leg(){
  local idx="$1" clog="$STATE/leg_${idx}.log" rc=""
  ( flock -n 9 || { echo "leg $idx: already locked, skip" >>"$clog"; exit 0; }
    for attempt in 1 2 3; do
      echo "== leg $idx attempt $attempt $(date -u +%FT%TZ) ==" >>"$clog"
      curl -s -m 10 "http://127.0.0.1:${LOCALPORT}/v1/models" 2>/dev/null | grep -q "petri__" || { echo "proxy unreachable" >>"$clog"; sleep 30; continue; }
      # SLURM_JOB_ID on the runner's line ONLY when the hand-off is on: petri_run._row_constants records it as the store's slurm_job_id, the
      # cluster_job_id the recorder's run payload carries — the proxy job IS the GPU allocation that served every row of this store, and the
      # job the registry registered these runs under (a grading-host process has no job id of its own)
      # `env` carries the proxy job id into the runner's row constants when the hand-off is on: a bare `${X:+VAR=v} cmd` is
      # NOT an assignment prefix after expansion (bash parses prefixes before expanding) — it becomes the command name and dies
      # rc 127. With the hand-off off, `env` runs the runner unchanged.
      env ${REG_IDENTITY:+SLURM_JOB_ID="$JOBID"} "$PY" scripts/petri_run.py "${PETRI_RUN_OPTS[@]}" run --out-root "$OUT" --cell-index "$idx" ${TARGET_DECODE_ARGS[@]+"${TARGET_DECODE_ARGS[@]}"} ${RERUN:+--overflow-rerun} >>"$clog" 2>&1; rc=$?
      echo "== leg $idx attempt $attempt rc=$rc ==" >>"$clog"
      [ "$rc" = 0 ] && break; [ "$rc" = 3 ] && break; [ -n "$RERUN" ] && break; sleep 20
    done
    rk=$("$PY" -c 'import json,sys;d=json.load(open(sys.argv[1]));r=d["canonical"]+d["controls"];print(r[int(sys.argv[2])]["petri_run_key"])' "$OUT/cells.json" "$idx")
    leg_status_write "$idx" "${rc:-unreachable}" "$OUT/cells/$rk"
    # the registry hand-off's close: the cell's POOLED run (no row spelling routes to it) gets its store_uri + `generated` from the
    # cell-done marker; the behaviour runs the store recorder already closed are left alone. Best-effort like every registry line:
    # a failure is one line in the leg log, never a failed leg (the harvester's pull is the backstop)
    if [ -n "$REG_IDENTITY" ] && [ "$rc" = 0 ] && [ -f "$OUT/cells/$rk/results.json" ]; then
      (cd "$WT" && "$PY" scripts/sdfbase/full22/petri_registry_handoff.py close-leg --identity "$REG_IDENTITY" --cell-dir "$OUT/cells/$rk") >>"$clog" 2>&1 \
        || echo "[registry] close-leg failed for $rk (see above); the runs of this cell keep whatever the store recorder filed" >>"$clog"
    fi
    if [ "$rc" = 0 ] && [ -f "$OUT/cells/$rk/results.json" ]; then echo "LEG[$idx] COMPLETE $rk (drive $DRIVE_ID)" | tee -a "$LOG"
    elif [ "$rc" = 0 ] || [ "$rc" = 3 ]; then echo "LEG[$idx] INCOMPLETE $rk (drive $DRIVE_ID)" | tee -a "$LOG"
    else echo "LEG[$idx] FAILED rc=${rc:-unreachable} $rk$( [ -f "$OUT/cells/$rk/results.json" ] && echo " (an existing results.json does not override the failed driver return)" ) (drive $DRIVE_ID)" | tee -a "$LOG"; fi
  ) 9>"$STATE/.leg_${idx}.lock" &
  LEG_PIDS="$LEG_PIDS $!"; echo "launched leg $idx (pid $!)" | tee -a "$LOG"
}
# A PREVIOUS ATTEMPT'S LEGS (2026-10-01, attempts 4-6 of a u0 drive): a leg whose per-cell flock is held belongs to a drive that is STILL
# RUNNING: a wrapper killed without its session leaves its legs auditing under the old configuration, and `already locked, skip` on every
# leg then read as 4 FAILED, three attempts in four minutes, the cap burnt against an idle proxy. Held locks are a NAMED refusal before
# anything is launched: rc 4 (PETRI_DRIVE_RC_LOCKS_HELD in the grading daemon: `blocked`, not counted toward its cap), a result document
# saying so, and the remedy in the log. The daemon stops a settled attempt's surviving SESSION (legs, heartbeat, tunnel) before its next start.
HELD=""; for idx in $LEGS; do flock -n "$STATE/.leg_${idx}.lock" true 2>/dev/null || HELD="$HELD $idx"; done
if [ -n "$HELD" ]; then
  log "REFUSED: a previous attempt's legs still hold the per-leg locks (legs:$HELD of [$LEGS]) — nothing launched; stop that attempt's legs (its whole session, not only its wrapper) and re-run"
  RESULT="$STATE/drive_result.json"
  DRIVE_ID="$DRIVE_ID" KEY="$KEY" VIEW="$VIEW" CLUSTER="$CLUSTER" TAG="$TAG" WAVE_SHA="$WAVE_SHA" HELD="${HELD# }" LEGS="$LEGS" LOG="$LOG" RESULT="$RESULT" "$PY" - <<'PYEOF'
import json, os, time
e = os.environ
doc = {"drive_id": e["DRIVE_ID"], "key": e["KEY"], "view": e["VIEW"], "cluster": e["CLUSTER"], "tag": e["TAG"], "wave_sha": e["WAVE_SHA"], "rc": 4, "verdict": "locks_held",
       "legs": {"scope": "all", "n_scope": len(e["LEGS"].split()), "complete": 0, "incomplete_overflow": 0, "failed": 0, "skipped": 0, "locked": [int(x) for x in e["HELD"].split()]},
       "handoff": {"mode": None, "identity": None, "finish_rc": 0, "verify_rc": None, "summary": None, "runs_planned": 0, "runs_filed": 0},
       "log": e["LOG"], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
tmp = e["RESULT"] + ".tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(doc, fh, indent=2, sort_keys=True); fh.write("\n")
os.replace(tmp, e["RESULT"])
PYEOF
  [ -z "${PETRI_DRIVE_RESULT_FILE:-}" ] || cp -f "$RESULT" "$PETRI_DRIVE_RESULT_FILE"
  log "DRIVE_VERDICT locks_held rc=4 legs: 0 complete, 0 incomplete-overflow, 0 failed, 0 skipped of $(wc -w <<< "$LEGS") in scope; locked:$HELD"
  exit 4
fi
for idx in $LEGS; do run_leg "$idx"; sleep 2; done
# the heartbeat the proxy checks (petri_proxy.sbatch): touched every 5 min while any leg is alive, over the same
# ssh path the endpoint was read from; stops with the legs. A driver that dies takes its heartbeat with it, and the
# proxy releases its GPU on its own within HEARTBEAT_STALE_S. Liveness is THIS shell's own leg pids,
# never a command-line substring: the old `pgrep -f "petri_run.py --manifest $MANIFEST run"` matched the contiguous
# argv of the pre-serving-path driver only, so once `--serving-path <p>` (and the native --verify-* pair) sat between
# the manifest and the subcommand no heartbeat was ever touched, on EITHER engine, and the proxy read the missing
# file as an absent driver past HEARTBEAT_GRACE_S and STOPped a healthy audit.
legs_alive(){ local p; for p in $LEG_PIDS; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
( while legs_alive; do timeout 90 ssh -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=2 -o BatchMode=yes "$HOST" "touch $REMOTE_OUT/driver_heartbeat" >/dev/null 2>&1 || true; sleep 300; done ) &
HB_PID=$!
# wait on the LEG pids and nothing else. A bare `wait` -- and `wait $(jobs -p)` -- also waits on the tunnel ssh
# above, which never exits, so the driver hung after its last leg and the STOP below was never touched
# (2026-09-07: qwen35_4b_it 4/4 legs COMPLETE, proxy 435478 idle on its GPU until STOP was touched by hand).
# The heartbeat loop is then stopped from the inside: killing its sleep makes it re-check for legs and exit.
wait $LEG_PIDS 2>/dev/null; pkill -P "$HB_PID" 2>/dev/null; kill "$HB_PID" 2>/dev/null; wait "$HB_PID" 2>/dev/null
log "DRIVE_DONE complete_legs_in_log=$(grep -c 'COMPLETE ' "$LOG" || true)"
# the registry hand-off's terminal push (every run record dir of the proxy's identities; a no-op without a push script — on the grading
# host the spool is the harvester's own records directory)
HANDOFF_RC=0; VERIFY_RC=""; VERIFY_SUMMARY=""; RUNS_PLANNED=0; RUNS_FILED=0
if [ -n "$REG_IDENTITY" ]; then
  (cd "$WT" && "$PY" scripts/sdfbase/full22/petri_registry_handoff.py finish --identity "$REG_IDENTITY") 2>&1 | tee -a "$LOG"; HANDOFF_RC=${PIPESTATUS[0]}
  [ "$HANDOFF_RC" = 0 ] || log "registry hand-off FINISH FAILED rc=$HANDOFF_RC — this drive is FAILED whatever the legs say (the review's (B) re-check: a swallowed finish once left a drive at rc 0)"
fi
# release the proxy when every leg has ENDED -- COMPLETE or INCOMPLETE on its overflow ledger -- and none FAILED:
# a ledgered leg's re-run needs the overflow geometry (a separate standup), so keeping THIS proxy up for it only
# burns its GPU (2026-09-07: nemotron3nano4b 5 COMPLETE + 1 ledgered, qwen35_4b 8 + 2 -- both proxies released by
# hand because this check keyed on results.json alone). A FAILED leg keeps the proxy up for a prompt re-run.
NCOMP=0; NINC=0; NFAIL=0; NSKIP=0
# THE RELEASE SCOPE: every leg of the record (all — today's rule), or the SELECTED legs only (selected — the lane that drives its
# object legs by name: the unselected `none` / controls / prompted legs are not this drive's work and would otherwise read as
# FAILED and hold the GPU until the heartbeat went stale)
if [ "$RELEASE_SCOPE" = selected ]; then SCOPE_LEGS="$SELECTED_LEGS"; NSCOPE=$(wc -w <<< "$SELECTED_LEGS"); else SCOPE_LEGS=$(seq -s ' ' 0 $((NLEGS-1))); NSCOPE=$NLEGS; fi
STATUS_LINES=""   # `<idx> <status>` per leg in scope (petri-completion): the result document's legs.detail — the counts stay the verdict, the detail is the record 06 and the daemon read
leg_status_note(){ STATUS_LINES="$STATUS_LINES$1 $2"$'\n'; }
for idx in $SCOPE_LEGS; do
  if leg_unserved "$idx"; then NSKIP=$((NSKIP+1)); leg_status_note "$idx" skipped; continue; fi   # a cell skipped by name: not this proxy's work
  rk=$("$PY" -c 'import json,sys;d=json.load(open(sys.argv[1]));r=d["canonical"]+d["controls"];print(r[int(sys.argv[2])]["petri_run_key"])' "$OUT/cells.json" "$idx")
  # a leg whose LAST recorded driver return (this invocation's or an earlier one's — the record stands until the leg
  # is driven again) is a failure (neither 0 nor the INCOMPLETE 3) is FAILED whatever the cell dir holds: the return
  # is the verdict on that leg; a results.json there belongs to whichever run wrote it
  # and never turns a refusal into a completion (it would release the proxy over an unaudited leg). A return of 0 is
  # a completion only over the results.json bytes it was returned on: bytes that changed since are UNVERIFIED (held,
  # named — drive the leg again, petri_run.py resumes by run key). A leg with no record at all is judged by its cell dir.
  leg_status_read "$idx"
  if [ -n "$LRC" ] && [ "$LRC" != 0 ] && [ "$LRC" != 3 ]; then
    log "leg $idx ($rk) FAILED rc=$LRC as returned by drive ${LDRIVE:-unrecorded} — held (an existing results.json does not override it)"; NFAIL=$((NFAIL+1)); leg_status_note "$idx" failed; continue
  fi
  if [ -f "$OUT/cells/$rk/results.json" ]; then
    if [ "$LRC" = 0 ] && [ -n "$LRES" ] && [ "$LRES" != absent ] && [ "$(sha256sum "$OUT/cells/$rk/results.json" | cut -c1-64)" != "$LRES" ]; then
      log "leg $idx ($rk) UNVERIFIED: results.json differs from the bytes drive ${LDRIVE:-unrecorded} returned 0 over — held (re-drive the leg)"; NFAIL=$((NFAIL+1)); leg_status_note "$idx" unverified; continue
    fi
    [ -n "$LRC" ] || log "leg $idx ($rk) COMPLETE by its cell dir alone (no driver return recorded under $STATE)"
    NCOMP=$((NCOMP+1)); leg_status_note "$idx" complete
  elif [ "$(f22_unresolved "$OUT/cells/$rk")" -gt 0 ]; then NINC=$((NINC+1)); leg_status_note "$idx" incomplete_overflow; else NFAIL=$((NFAIL+1)); leg_status_note "$idx" failed; fi
done
# THE FILED RESULTS: rc 0 is earned only when every planned run of the proxy's identities is FILED on this
# host's spool (`generated` + store_uri: petri_registry_handoff.py verify) — the daemon records `completed` from the summary this writes,
# never from the exit code alone. Run only on the one path that could be complete (no failed, no overflow-pending leg); a standup-skipped
# cell's runs are reported skipped, and a drive of SELECTED legs verifies its selection only.
# THE VERIFY SCOPE: `all` / `selected` as today; under the LEDGERED selector the `--objects` are the selected legs' cell ids
# INTERSECTED with the REGISTERED objects of the identity file (`runs[].route.object`, the import's harness spelling) — a control leg (none__lora0)
# is not a registered object (close-leg says so by name) and drops out; when the intersection is EMPTY (every selected leg a control leg) verify is
# NOT called (it would refuse `n_planned == 0` and turn a complete control-only pass into rc 1): the scope is `control_only`, completion is this
# drive's verdict over legs.detail, and the daemon's overflow rule decides whether that 0 is `completed`. The non-ledgered `selected` list is byte-identical.
VERIFY_SCOPE="$RELEASE_SCOPE"
if [ -n "$REG_IDENTITY" ] && [ "$NFAIL" = 0 ] && [ "$NINC" = 0 ]; then
  VERIFY_SUMMARY="$STATE/registry/verify.$DRIVE_ID.json"; VOBJ=(); RUN_VERIFY=1
  [ -z "${SKIPPED:-}" ] || VOBJ+=(--skipped "$SKIPPED")
  if [ "$RELEASE_SCOPE" = selected ]; then
    VOBJ_LIST="$("$PY" -c 'import json,sys;d=json.load(open(sys.argv[1]));r=d["canonical"]+d["controls"];print(",".join(str(r[int(i)].get("cell_id") or r[int(i)]["petri_run_key"]) for i in sys.argv[2:]))' "$OUT/cells.json" $SELECTED_LEGS)"
    if [ -n "$LEDGERED" ]; then
      VOBJ_LIST="$("$PY" -c 'import json,sys; ident=json.load(open(sys.argv[1])); reg={str((r.get("route") or {}).get("object") or "") for r in (ident.get("runs") or [])}; print(",".join(c for c in sys.argv[2].split(",") if c in reg))' "$REG_IDENTITY" "$VOBJ_LIST")"
      if [ -z "$VOBJ_LIST" ]; then
        VERIFY_SCOPE=control_only; RUN_VERIFY=0; VERIFY_SUMMARY=""
        log "every selected leg is a control leg: no registered run is in scope; verify is not run (it would refuse n_planned == 0) — completion is this drive's verdict over legs.detail (verify_scope=control_only)"
      fi
    fi
    [ -z "$VOBJ_LIST" ] || VOBJ+=(--objects "$VOBJ_LIST")
  fi
  if [ "$RUN_VERIFY" = 1 ]; then
  (cd "$WT" && "$PY" scripts/sdfbase/full22/petri_registry_handoff.py verify --identity "$REG_IDENTITY" --summary "$VERIFY_SUMMARY" "${VOBJ[@]+"${VOBJ[@]}"}") 2>&1 | tee -a "$LOG"; VERIFY_RC=${PIPESTATUS[0]}
  read -r RUNS_PLANNED RUNS_FILED < <("$PY" -c 'import json,sys
try: d=json.load(open(sys.argv[1])); print(d.get("n_planned",0), d.get("n_filed",0))
except Exception: print(0, 0)' "$VERIFY_SUMMARY")
  fi
fi
# THE VERDICT = THE EXIT CODE (before this the last `log` decided rc 0 over failed legs): 0 only when every
# leg in scope is COMPLETE (or skipped by name), the hand-off finished and every planned run verified filed; 3 when legs are INCOMPLETE on
# overflow ledgers and none failed (the proxy is released; the re-run pass, F22_OVERFLOW_RERUN=1, is a deliberate step — not a success);
# 1 for a failed / unverified leg, a failed finish, or an unfiled run. $1 NFAIL $2 NINC $3 HANDOFF_RC $4 VERIFY_RC ("" = not run).
f22_drive_verdict(){ if [ "${1:-0}" != 0 ] || [ "${3:-0}" != 0 ] || { [ -n "${4:-}" ] && [ "$4" != 0 ]; }; then echo 1; elif [ "${2:-0}" != 0 ]; then echo 3; else echo 0; fi; }
if [ -n "$RERUN" ] || [ -n "$SECOND" ]; then
  # the re-run / second proxy exists for THIS pass's named legs only: release it however they ended (a refusal from
  # --overflow-rerun is a verdict — an unrecoverable overflow at the re-run window, or a no-op replay — that another pass here
  # cannot change; a second pass's legs that did not complete are re-run by resubmitting the proxy + re-launching the driver)
  ssh -o ConnectTimeout=20 "$HOST" "touch $REMOTE_OUT/STOP" && log "${RERUN:+OVERFLOW RE-RUN}${SECOND:+SECOND} PASS ENDED ($NCOMP COMPLETE, $NINC INCOMPLETE on ledgers, $NFAIL FAILED across the audit's $NLEGS legs) — STOP touched on $HOST:$REMOTE_OUT"
  f22_tunnel_teardown
elif [ "$NFAIL" = 0 ]; then
  ssh -o ConnectTimeout=20 "$HOST" "touch $REMOTE_OUT/STOP" && log "$( [ "$RELEASE_SCOPE" = selected ] && echo "ALL $NSCOPE SELECTED LEGS ENDED (of $NLEGS in the record; release scope: selected)" || echo "ALL $NLEGS LEGS ENDED" ) ($NCOMP COMPLETE, $NINC INCOMPLETE on overflow ledgers$( [ "$NSKIP" = 0 ] || echo ", $NSKIP not served by this proxy" )) — STOP touched on $HOST:$REMOTE_OUT"
  # the tunnel THIS run opened goes with the proxy it pointed at; a reused one is left to its opener
  f22_tunnel_teardown
else
  log "legs remain — proxy left up (re-run; re-submit petri_proxy.sbatch if its walltime expires); the tunnel this drive opened closes with it (D1503) and the re-run opens its own"
fi
DRIVE_RC=$(f22_drive_verdict "$NFAIL" "$NINC" "$HANDOFF_RC" "$VERIFY_RC")
case "$DRIVE_RC" in 0) VERDICT=complete;; 3) VERDICT=overflow_pending;; *) VERDICT=failed;; esac
# the machine-readable outcome, beside the drive's own log and — when the grading daemon started this drive — at the path it named
RESULT="$STATE/drive_result.json"
DRIVE_ID="$DRIVE_ID" KEY="$KEY" VIEW="$VIEW" CLUSTER="$CLUSTER" TAG="$TAG" WAVE_SHA="$WAVE_SHA" DRIVE_RC="$DRIVE_RC" VERDICT="$VERDICT" \
NSCOPE="$NSCOPE" NCOMP="$NCOMP" NINC="$NINC" NFAIL="$NFAIL" NSKIP="$NSKIP" RELEASE_SCOPE="$RELEASE_SCOPE" HANDOFF_MODE="$HANDOFF_MODE" REG_IDENTITY="$REG_IDENTITY" \
HANDOFF_RC="$HANDOFF_RC" VERIFY_RC="$VERIFY_RC" VERIFY_SUMMARY="$VERIFY_SUMMARY" RUNS_PLANNED="$RUNS_PLANNED" RUNS_FILED="$RUNS_FILED" LOG="$LOG" RESULT="$RESULT" \
VERIFY_SCOPE="$VERIFY_SCOPE" OUT="$OUT" STATE="$STATE" REMOTE_OUT="$REMOTE_OUT" WT="$WT" SCOPE_LEGS="$SCOPE_LEGS" STATUS_LINES="$STATUS_LINES" \
"$PY" - <<'PYEOF'
import json, os, sys, time
e = os.environ
n = lambda k: int(e.get(k) or 0)
# legs.detail (petri-completion): every leg of the frozen record — identity, whether the identity file registers a run for it, the per-leg
# verdict this drive reached (`complete | incomplete_overflow | failed | unverified | skipped | out_of_scope`), the ledger's unresolved and
# context-overflow counts (petri_run's own readers, the classification _arm_overflow_rerun applies). 06 --pass overflow and the daemon read it.
sys.path.insert(0, os.path.join(e["WT"], "scripts")); sys.path.insert(0, os.path.join(e["WT"], "src"))
status = {int(ln.split()[0]): ln.split()[1] for ln in e.get("STATUS_LINES", "").splitlines() if ln.strip()}
scope = {int(x) for x in e.get("SCOPE_LEGS", "").split()}
detail = []
try:
    rec = json.load(open(os.path.join(e["OUT"], "cells.json"), encoding="utf-8"))
    registered = set()
    if e.get("REG_IDENTITY"):
        try:
            registered = {str((r.get("route") or {}).get("object") or "") for r in (json.load(open(e["REG_IDENTITY"], encoding="utf-8")).get("runs") or [])}
        except Exception:  # noqa: BLE001 — an unreadable identity file: no leg reads as registered (the hand-off's own lines say why)
            registered = set()
    try:
        import petri_run as prun
        from pathlib import Path
        def ledger(d):
            u = prun._unresolved_ledger_entries(Path(d)); return len(u), len(prun.overflow_ledger_entries(u)), None
    except Exception as exc:  # noqa: BLE001
        _why = str(exc)
        def ledger(d):
            return None, None, f"petri_run import failed: {_why}"
    for i, c in enumerate(list(rec.get("canonical") or []) + list(rec.get("controls") or [])):
        cd = os.path.join(e["OUT"], "cells", str(c.get("petri_run_key")))
        try:
            unresolved, overflow, lerr = ledger(cd)
        except Exception as exc:  # noqa: BLE001
            unresolved, overflow, lerr = None, None, str(exc)
        d = {"idx": i, "cell_id": c.get("cell_id"), "control_of": c.get("control_of"), "kind": c.get("kind"), "petri_run_key": c.get("petri_run_key"),
             "registered": str(c.get("cell_id")) in registered, "in_scope": i in scope, "status": status.get(i, "out_of_scope" if i not in scope else "unknown"),
             "unresolved": unresolved, "overflow_unresolved": overflow}
        if lerr:
            d["ledger_error"] = lerr
        detail.append(d)
except Exception as exc:  # noqa: BLE001 — the counts above are the verdict; a detail that cannot be assembled is said, never invented
    detail = [{"error": f"legs.detail could not be assembled: {type(exc).__name__}: {exc}"}]
doc = {"drive_id": e["DRIVE_ID"], "key": e["KEY"], "view": e["VIEW"], "cluster": e["CLUSTER"], "tag": e["TAG"], "wave_sha": e["WAVE_SHA"], "rc": n("DRIVE_RC"),
       "verdict": e["VERDICT"], "legs": {"scope": e["RELEASE_SCOPE"], "n_scope": n("NSCOPE"), "complete": n("NCOMP"), "incomplete_overflow": n("NINC"), "failed": n("NFAIL"), "skipped": n("NSKIP"),
                                      "detail": detail},
       "handoff": {"mode": e["HANDOFF_MODE"], "identity": e["REG_IDENTITY"] or None, "finish_rc": n("HANDOFF_RC"), "verify_rc": (int(e["VERIFY_RC"]) if e.get("VERIFY_RC") else None),
                   "summary": e["VERIFY_SUMMARY"] or None, "runs_planned": n("RUNS_PLANNED"), "runs_filed": n("RUNS_FILED"), "verify_scope": e.get("VERIFY_SCOPE") or None},
       "paths": {"out_root": e["OUT"], "cells_json": os.path.join(e["OUT"], "cells.json"), "state": e["STATE"], "identity": e["REG_IDENTITY"] or None, "remote_out": e["REMOTE_OUT"]},
       "log": e["LOG"], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
tmp = e["RESULT"] + ".tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(doc, fh, indent=2, sort_keys=True); fh.write("\n")
os.replace(tmp, e["RESULT"])
PYEOF
[ -z "${PETRI_DRIVE_RESULT_FILE:-}" ] || cp -f "$RESULT" "$PETRI_DRIVE_RESULT_FILE"
log "DRIVE_VERDICT $VERDICT rc=$DRIVE_RC legs: $NCOMP complete, $NINC incomplete-overflow, $NFAIL failed, $NSKIP skipped of $NSCOPE in scope; hand-off: mode=$HANDOFF_MODE finish_rc=$HANDOFF_RC verify_rc=${VERIFY_RC:-not-run} verify_scope=$VERIFY_SCOPE runs_filed=$RUNS_FILED/$RUNS_PLANNED — $RESULT"
exit "$DRIVE_RC"
