#!/bin/bash
#SBATCH --job-name=f22-misalign
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=0-05:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# em-misalign (WP-C): the misalignment evals over ONE target's served cells, on one GPU, self-contained:
#   bank view (misalign_bank_view.sh: the key's full22 view + the pin-verified `alphasteer` entry; a target key
#   is a (roster target, view) pair — `llama8b_it` = llama8b/it, the instruct-direct objects, the paper's fourth
#   family; `llama8b` = llama8b/main, the base-lane objects transferred; everyone else <target>/main;
#   `<family>_bl` = the PUBLISHED BASELINES panel — none + arditi + the AlphaSteer RE-BAKE rung, agentic only)
#   -> scripts/sdfbase/mt_serving_up.py --mode petri with configs/misalign_full22_<target>.yaml +
#      configs/capeval/manifests/misalign_full22_<target>.yaml at the certified Petri geometry
#      (max_model_len 16384 x the model's identity max_num_seqs; asserted, never assumed), every cell a
#      virtual model behind one loopback RoutingProxy (the wave's f22_standup: stacked reductions
#      attested, geometry attested, model facts, M-RoPE attestation for glm)
#   -> health wait + a one-request smoke on `petri__base`
#   -> cells.json for the runner (from the standup's cells_served.json enumeration + endpoint.json)
#   -> sisteer.capeval.misalign.runner ON THE NODE against loopback (no API key on the node):
#        family agentic  --n 10 --behaviors leaking murder --max-tokens 3000
#                        (+ --exclude nemotron3nano4b:leaking on nemotron3nano4b: 2.9% base rate, no headroom)
#        family em8q     --n 50 --max-tokens 600
#   -> tear down; RESULT line.
# Outputs: $SHARED/exp/misalign/<target>/{serving,agentic,em8q}/, cells.json, cells_full.json, RESULT.
# Judging runs on the grading host afterwards (misalign_judge.py) from the pulled tree.
#
# Usage (positional only — never --export=; run from the staged checkout, see README.md):
#   ssh <CLUSTER_SSH> 'cd <CLUSTER_EVAL_REPO_ROOT>/<sha> && sbatch --qos=<CLUSTER_QOS> [--output=<CLUSTER_LOG_DIR>/%x_%j.out] \
#     -J f22-misalign-<target> scripts/sdfbase/full22/misalign/misalign_proxy.sbatch <target> <sha>'
#   (each <...> is that key of the cluster's block in the site config; --output where the cluster's log dir is not the
#   script's own default)
# The sha pin is the wave's: common.sh f22_init FATALs unless the checkout's HEAD equals <sha> and the tree
# is clean. Resumable: the runner skips (item_id, sample_idx) pairs already on disk; resubmit after a
# walltime expiry or a node failure. A per-target STOP sentinel ($SHARED/exp/misalign/<target>/STOP) or the
# wave's STOP makes the job exit 0 without standing anything up.
#
# Venvs: the serving venv (common.sh f22_serving_venv: configs/capeval/vllm-requirements.txt — vllm 0.25.1+cu129,
# torch 2.11.0+cu129, vllm-lens 1.2.1, installed OUTSIDE the project lockfile, like every full22 job) serves;
# the driver venv (f22_driver_venv: `uv sync --frozen --extra cu128` at <sha>) runs the runner. This file lives
# under scripts/sdfbase/full22/, outside tests/test_torch_flavors.py's scripts/sbatch audit, exactly like the
# wave's petri_proxy.sbatch beside it; it carries no audit-exempt tag (the test caps the tag's carriers to the
# recorded scripts/sbatch set).
set -uo pipefail
TARGET="${1:?target (gemma4e4b|qwen35_4b|nemotron3nano4b|glm46vflash|llama8b|llama8b_it|llama8b_org|qwen35_4b_org|gemma4e4b_it|qwen35_4b_it|qwen35_4b_s1|qwen35_4b_s2|qwen35_4b_s3|qwen35_4b_s1p|qwen35_4b_s2p|qwen35_4b_s3p|<family>_bl|<models.yaml key>_<phase view>)}"; WAVE_SHA="${2:?wave sha}"
# $3 = mode: `full` (default: both families at the eval's 3000 / 600 caps under the certified 16384 window) or
#   NOTE (2026-09-21): `vendor` is NOT a sharding option only — it changes the agentic DECODE and CAP relative to `full`
#   (`--sampler vendor --max-tokens 16384` into the misalign_vendor root vs full's runner default at 3000): two halves of one
#   comparison must run under ONE mode.
# `ladder16k` (the truncation ladder: regenerate ONLY the agentic responses that hit the
# 3000 cap and are not repetition loops, at a 16384 cap, into the NEW root $SHARED/exp/misalign_16k/<target>
# beside the original; window 32768 = up to a ~10k prompted prompt + 16384 cap + margin, max_num_seqs scaled to keep the
# source run's batched-token budget; em8q stays at the paper's 600 — not laddered)
# `organisms`: the EM MODEL ORGANISMS × the target's steering objects, em8q family only —
# the standup loads the three EM adapters as vLLM LoRA modules (--organism, from the OFFLINE HF cache) and the
# proxy serves every non-lora leg also as <vmodel>@<organism> (the object on top of the adapter, one forward
# pass); cells.json gains one overlay row per organism × leg (cell_id <organism>::<cell>); output root
# $SHARED/exp/misalign_org/<target>. Meant for the llama8b keys (the organisms are Llama-3.1-8B-Instruct LoRAs):
# `llama8b` (the base-lane objects transferred, the organism-transfer test) or `llama8b_it` (instruct-direct).
# `vendor` (qwen's T=1.0 unsteered baseline loops on 23% of murder scenarios, so that column
# was dropped — re-sample it under the vendor's recommended row so its objects can be scored): a FRESH agentic run
# with the runner's --sampler vendor (every sampling key explicit on the wire, recorded in meta.json; Qwen3.5
# thinking-off = T 0.7 / top-p 0.8 / top-k 20 / min-p 0 / presence 1.5 / repetition 1.0), in ONE rung at the cap the
# ladder measured this model needs (16384; window 32768, S scaled to the source run's batched-token budget), root
# $SHARED/exp/misalign_vendor/<target>, em8q skipped; `$4` shards bin the LEGS (one standup per shard) and `$5` is
# the behaviours CSV (default leaking,murder; the qwen ask is `murder`).
# `$6` (vendor mode only) = the root name under $SHARED/exp, default `misalign_vendor`, else `misalign_vendor_<suffix>`: the
# LEAKING re-sample (cutting the difference LoRA's T=1.0 degeneracy) lands in `misalign_vendor_leak` so the
# finished murder record under misalign_vendor is never appended to (a second pass gets its own root, never a merge-in-place).
MODE="${3:-full}"
case "$MODE" in full|ladder16k|organisms|vendor) ;; *) echo "unknown mode $MODE (full|ladder16k|organisms|vendor)" >&2; exit 2;; esac
# $4 = shard count for ladder16k / vendor (default 1). With `--array=0-(N-1)` each task stands up its own proxy
# (serving_shard<k>/, cells_shard<k>.json) and runs ONLY the cells binned to it — the ladder greedily by selected-item
# count, the vendor run by leg — all writing disjoint cell dirs under the same root. Hard rule 6: one array, never a
# sbatch loop.
SHARDS="${4:-1}"; SHARD="${SLURM_ARRAY_TASK_ID:-0}"; SFX=""
if [ "$SHARDS" -gt 1 ]; then
  case "$MODE" in ladder16k|vendor) ;; *) echo "shards are a ladder16k / vendor option only" >&2; exit 2;; esac
  [ "$SHARD" -lt "$SHARDS" ] || { echo "shard $SHARD >= $SHARDS" >&2; exit 2; }
  SFX="_shard$SHARD"
fi
# $5 = the agentic behaviours (CSV), a vendor / ladder16k option; every other mode runs the eval's fixed pair
BEHAVIORS_CSV="${5:-leaking,murder}"; IFS=, read -r -a BEHAVIORS <<< "$BEHAVIORS_CSV"
for b in "${BEHAVIORS[@]}"; do case "$b" in leaking|murder) ;; *) echo "unknown behaviour $b (leaking|murder)" >&2; exit 2;; esac; done
# ladder16k may also name the behaviours: a family whose murder column comes from the VENDOR re-sample (qwen35_4b and
# qwen35_4b_it — the T=1 unsteered baseline loops) reruns only its LEAKING truncations at 16k
case "$MODE" in vendor|ladder16k) ;; *) [ "$BEHAVIORS_CSV" = "leaking,murder" ] || { echo "\$5 (behaviours) is a vendor / ladder16k option only" >&2; exit 2; };; esac
# $6 = the vendor root name (vendor mode only): misalign_vendor (default) or misalign_vendor_<suffix>, so every consumer
# that enumerates misalign* roots (the regrade, the store survey) still sees it. One root per re-sample: a finished record
# is never appended to (the murder trees are pinned by hash downstream and carry a live second-grader's sibling files).
VENDOR_ROOT="${6:-misalign_vendor}"
case "$VENDOR_ROOT" in misalign_vendor|misalign_vendor_[A-Za-z0-9]*) ;; *) echo "\$6 (vendor root) must be misalign_vendor or misalign_vendor_<suffix>; got $VENDOR_ROOT" >&2; exit 2;; esac
[ "$MODE" = vendor ] || [ "$VENDOR_ROOT" = misalign_vendor ] || { echo "\$6 (vendor root) is a vendor-mode option only" >&2; exit 2; }
{ [ -n "${SISTEER_SHARED_ROOT:-}" ] && source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA/scripts/sdfbase/full22/common.sh" 2>/dev/null; } \
  || source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit
# the launch's cells for the kind-aware serving-path default (2026-09-15): the MISALIGN manifest of this TARGET — never the
# campaign's petri pair (the (KEY, VIEW) below names a different cell set); an unknown TARGET fails here, by the file's name
F22_LAUNCH_MANIFEST="configs/capeval/manifests/misalign_full22_${TARGET}.yaml"
# organisms mode is a LENS-PATH feature: mt_serving_up.py serves the organism adapters as the vLLM LoRA modules
# (build_server_launch lora_modules=organisms) and the proxy overlays <vmodel>@<organism> on every non-LoRA leg. The native
# launcher fills the adapter slots with the wave's LoRA OBJECTS and ignores --organism (observed: endpoint `organisms:
# None`, "overlay petri__base@em-financial_r32 is not served"). So the path is pinned here, BEFORE f22_init's selection
# (an explicit F22_SERVING_PATH wins); a caller's explicit pin still stands.
[ "$MODE" = organisms ] && export F22_SERVING_PATH="${F22_SERVING_PATH:-lens}"
f22_init "$WAVE_SHA"
# the misalign TARGET key -> the wave's (KEY = model key, VIEW): f22_standup passes KEY as --model-key to the standup
# and the load check (the geometry budget, the pinned model), so it must be the ROSTER target; TARGET names the
# configs, the bank view and the output roots (make_misalign_manifest.WAVE_OF / misalign_bank_view.sh: one table)
case "$TARGET" in
  gemma4e4b|qwen35_4b|nemotron3nano4b|glm46vflash) KEY="$TARGET"; VIEW=main;;
  llama8b)    KEY=llama8b; VIEW=main;;    # the base-lane objects transferred onto the instruct model
  llama8b_it) KEY=llama8b; VIEW=it;;      # the instruct-direct objects (the paper's fourth family)
  llama8b_org) KEY=llama8b; VIEW=main;;   # the EM-organisms transfer target: main view under the STOCK chat template
  qwen35_4b_org) KEY=qwen35_4b; VIEW=main;;   # The Qwen3.5-4B EM organisms (LOCAL LoRA adapters, em_organism_train.sbatch) x the main-view objects, em8q only
  gemma4e4b_it) KEY=gemma4e4b; VIEW=it;;  # instruct-direct objects, side by side with the base->instruct `gemma4e4b` panel
  qwen35_4b_it) KEY=qwen35_4b; VIEW=it;;
  qwen35_4b_psr) KEY=qwen35_4b; VIEW=main;;
  qwen35_4b_allpos) KEY=qwen35_4b; VIEW=it;;   # none + arditi_allpos (Arditi all-positions arm), native single-object standup; run on Modal   # none + the three PSR psr_ce_all cells (ReLU steering) for the qwen leaking vendor panel, native path
  # qwen-base-seed-replicates: the three seed-replicate views of qwen35_4b (roster views s1|s2|s3), each its own
  # target key: LANE-SCOPED configs (make_misalign_manifest.LANE_POLICY — `none` + the view's ADMITTED objects SERVED (five:
  # the distilled PSR difference failed admission for every seed), the same five MEASURED,
  # PSR included, no prompted / alphasteer arm) and its own output root $SHARED/exp/misalign/qwen35_4b_s<s>
  qwen35_4b_s1|qwen35_4b_s2|qwen35_4b_s3) KEY=qwen35_4b; VIEW="${TARGET#qwen35_4b_}";;
  # the PAIRED views s<s>p: the composed difference_relu arm + the seed's r=4 LoRA +
  # its LoRA difference, LANE POLICY `paired` (make_misalign_manifest.PAIRED_VIEWS: 3 cells measured, none health-only, the
  # shared campaign anchor) and its own output root $SHARED/exp/misalign/qwen35_4b_s<s>p
  qwen35_4b_s1p|qwen35_4b_s2p|qwen35_4b_s3p) KEY=qwen35_4b; VIEW="${TARGET#qwen35_4b_}";;
  # the PUBLISHED BASELINES (lowest priority): the family's main-view serving
  # (same model, geometry, template as its campaign panel) with cells [none, arditi, alphasteer(RE-BAKE rung)] from the
  # pinned baselines view (misalign_bank_view.sh <family>_bl; configs/misalign_baseline_pins.json). Agentic only —
  # leaking + murder — the ruling names no EM column for the baselines.
  gemma4e4b_bl|qwen35_4b_bl|nemotron3nano4b_bl|llama8b_bl) KEY="${TARGET%_bl}"; VIEW=main; BASELINES=1;;
  # A FRONT-DOOR phase view — `<reproduce/models.yaml key>_<u0|s<N>|d<a>p<b>[_relu]|...>` (06 submits
  # `misalign_proxy.sbatch <key>_<view> <sha> full`). The generator's DERIVED table is the one source (make_misalign_manifest.front_door_targets
  # over the models table, the same grammar the seedrep grader twin admits); this script keeps no second copy of it and asks the module
  # (`--resolve`: `<roster target> <view> <kind>`). The driver venv carries the resolver's PyYAML; f22_driver_venv is sentinel-guarded, so
  # the standing call further down is then a no-op. A key the module does not know fails here, by name, before any GPU minute.
  *) f22_driver_venv
     read -r KEY VIEW KIND < <(cd "$REPO_DIR" && "$MPY" scripts/sdfbase/full22/misalign/make_misalign_manifest.py --resolve "$TARGET") \
       || f22_fatal "unknown target $TARGET (not a campaign key, a lane view or a front-door phase view of reproduce/models.yaml: make_misalign_manifest.TARGETS)"
     [ "$KIND" = front_door ] || f22_fatal "target $TARGET resolves to kind $KIND — every non-front-door key is named in this case explicitly; refusing"
     f22_log "front-door target $TARGET -> model key $KEY, phase view $VIEW (make_misalign_manifest.py --resolve; served == measured == the view's cells, ReLU arms included, no alphasteer)";;
esac
BASELINES="${BASELINES:-0}"; EM8Q_ARGS=()
[ "$MODE" = organisms ] && [ "$BASELINES" = 1 ] && f22_fatal "organisms mode is not a baselines mode"
[ "$MODE" = organisms ] && case "$TARGET" in llama8b_org|qwen35_4b_org) ;; *) f22_fatal "organisms mode runs on target llama8b_org (llama8b main view under the stock chat template, D43) or qwen35_4b_org (the local Qwen organisms, D64); got $TARGET";; esac
# the (KEY, VIEW, TAG, CFG, MANIFEST) f22_standup reads — the misalign pair, not f22_resolve_config's petri pair
TAG="$TARGET"
CFG="configs/misalign_full22_${TARGET}.yaml"; MANIFEST="configs/capeval/manifests/misalign_full22_${TARGET}.yaml"
[ -f "$CFG" ] && [ -f "$MANIFEST" ] || f22_fatal "missing $CFG or $MANIFEST (make_misalign_manifest.py)"
# registry (2026-09-17): this job binds its own pair instead of calling f22_resolve_config, so it registers itself
# HERE — the evals by mode (organisms = the EM panels; every other mode = the agentic behaviours it runs, which
# f22_registry_evals_for_job cannot know for a vendor / ladder16k run), the objects from $CFG's explicit_cells
case "$MODE" in
  organisms) F22_REGISTRY_EVALS="${F22_REGISTRY_EVALS:-em_risky_financial em_bad_medical em_extreme_sports em_base_control}";;
  *) F22_REGISTRY_EVALS="${F22_REGISTRY_EVALS:-$(printf '%s\n' "${BEHAVIORS[@]}" | sed 's/^/agentic_/' | tr '\n' ' ')}";;
esac
# the run key is the MEASUREMENT's —
#   `sdfbase_full22-misalign-<agentic|organisms>-<key>[_<view>]-target-<TARGET>-mode-<mode>[-sampler-<vendor|t1>][-root-<suffix>][-shard<k>of<n>][-baselines]`
# The TARGET (the config's own identity, as this script receives it) is in the key because several targets resolve to one (KEY, VIEW):
# qwen35_4b and qwen35_4b_psr are both KEY=qwen35_4b / VIEW=main / MODE=full and both measure `none` under distinct private_run / output
# roots — two independent measurements, two runs. Where a mode has several decode policies or rerun roots, the resolved one is in the
# key too: the organisms sampler (vendor | t1 are distinct decode rungs with distinct roots) and a vendor re-sample root's suffix.
[ "$MODE" = organisms ] && { EM_ORG_SAMPLER="${EM_ORG_SAMPLER:-vendor}"; case "$EM_ORG_SAMPLER" in vendor|t1) ;; *) f22_fatal "EM_ORG_SAMPLER must be vendor or t1, got '$EM_ORG_SAMPLER'";; esac; }
export F22_REGISTRY_KEY_SEGMENTS="${F22_REGISTRY_KEY_SEGMENTS:-target-${TARGET}-mode-${MODE}$( [ "$MODE" = organisms ] && echo "-sampler-${EM_ORG_SAMPLER}" )$( [ "$MODE" = vendor ] && [ "$VENDOR_ROOT" != misalign_vendor ] && echo "-root-${VENDOR_ROOT#misalign_vendor_}" )$( [ "${SHARDS:-1}" -gt 1 ] && echo "-shard${SHARD}of${SHARDS}" )$( [ "${BASELINES:-0}" = 1 ] && echo "-baselines" )}"
f22_registry_register || true
# the wave's two job-side belts (common.sh f22_resolve_config), restated for the misalign pair:
case "$KEY" in qwen35_4b|glm46vflash)   # TEXT-ONLY serving for the multimodal targets (the 2026-09-07 OOMs)
  grep -q -- '--limit-mm-per-prompt' "$MANIFEST" || f22_fatal "$MANIFEST serves multimodal $KEY without --limit-mm-per-prompt — regenerate";;
esac
case "$KEY" in glm46vflash) GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}";; esac   # KV coverage at the identity geometry (2026-09-08)
SRC_ROOT="$SHARED/exp/misalign/$TARGET"
if [ "$MODE" = ladder16k ]; then
  MIS_ROOT="$SHARED/exp/misalign_16k/$TARGET"
  ls "$SRC_ROOT"/agentic/*/responses.jsonl >/dev/null 2>&1 || f22_fatal "ladder16k: no source responses under $SRC_ROOT/agentic"
  [ -s "$SRC_ROOT/endpoint.snapshot.json" ] || f22_fatal "ladder16k: $SRC_ROOT/endpoint.snapshot.json missing (needed for the seq budget)"
  # 32768: a prompted-constitution cell (5.9k-token SI + ~3k scenario ~= 9-10k tokens) + the 16384 cap did NOT
  # fit 24576 (three nemotron attempts died on it; vLLM's "at least N input tokens" is not the prompt length)
  LADDER_L=32768
  SRC_S=$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["attested_max_num_seqs"] or json.load(open(sys.argv[1]))["max_num_seqs"]))' "$SRC_ROOT/endpoint.snapshot.json") \
    || f22_fatal "ladder16k: cannot read max_num_seqs from $SRC_ROOT/endpoint.snapshot.json"
  LADDER_S=$(( SRC_S * 16384 / LADDER_L )); [ "$LADDER_S" -ge 4 ] || LADDER_S=4
  f22_log "ladder16k: source $SRC_ROOT (S=$SRC_S at 16384) -> $MIS_ROOT at ${LADDER_L}x${LADDER_S}, agentic cap 16384, em8q skipped"
elif [ "$MODE" = organisms ]; then
  # DECODE: the VENDOR preset is the default for every family, this one
  # included, unless specifically instructed otherwise. The em8q family had been
  # running at the campaign's bare T=1.0 (`--sampler t1`), which is the decode that made qwen loop on the murder
  # scenarios and which inflates the judge's incoherence readings: on the T=1.0 organisms panel the CLEAN model
  # hit the 600-token cap on 57.8 % of answers and the steered arms read 17-51 % of answers at coherence <= 50 with no
  # organism present. Vendor and T=1.0 therefore get SEPARATE ROOTS: a decode change is a different measurement, never
  # a rerun into the same tree. EM_ORG_SAMPLER=t1 reproduces the record rung.
  EM_ORG_SAMPLER="${EM_ORG_SAMPLER:-vendor}"
  case "$EM_ORG_SAMPLER" in
    vendor) MIS_ROOT="$SHARED/exp/misalign_org_vendor/$TARGET"; EM8Q_ARGS+=(--sampler vendor);;
    t1)     MIS_ROOT="$SHARED/exp/misalign_org/$TARGET";;
    *)      f22_fatal "EM_ORG_SAMPLER must be vendor or t1, got '$EM_ORG_SAMPLER'";;
  esac
  f22_log "organisms decode: $EM_ORG_SAMPLER (root $MIS_ROOT)"
elif [ "$MODE" = vendor ]; then
  # a fresh run at the measured cap in one rung: the same 32768 window and budget-scaled S as the ladder (the
  # source run's endpoint snapshot gives S at 16384), no --rerun-truncated-from — every (item, sample) is drawn
  MIS_ROOT="$SHARED/exp/$VENDOR_ROOT/$TARGET"
  [ -s "$SRC_ROOT/endpoint.snapshot.json" ] || f22_fatal "vendor: $SRC_ROOT/endpoint.snapshot.json missing (needed for the seq budget)"
  LADDER_L=32768
  SRC_S=$(python3 -c 'import json,sys; print(int(json.load(open(sys.argv[1]))["attested_max_num_seqs"] or json.load(open(sys.argv[1]))["max_num_seqs"]))' "$SRC_ROOT/endpoint.snapshot.json") \
    || f22_fatal "vendor: cannot read max_num_seqs from $SRC_ROOT/endpoint.snapshot.json"
  LADDER_S=$(( SRC_S * 16384 / LADDER_L )); [ "$LADDER_S" -ge 4 ] || LADDER_S=4
  f22_log "vendor: fresh agentic run (behaviours $BEHAVIORS_CSV) -> $MIS_ROOT at ${LADDER_L}x${LADDER_S}, cap 16384, --sampler vendor, em8q skipped"
else
  MIS_ROOT="$SRC_ROOT"
fi
# the standup's REPRESENTATIVE LOAD CHECK (common.sh f22_standup)
# fires S concurrent requests at the window with a cap: THIS run's — the agentic 3000 (16384 for the ladder and the
# vendor run) — not the Petri table's; a probe that fails at the attested (S, L) is a measured limit and refuses traffic
F22_LOAD_CHECK_CAP=3000; case "$MODE" in ladder16k|vendor) F22_LOAD_CHECK_CAP=16384;; esac
OUT="$MIS_ROOT/serving$SFX"; mkdir -p "$OUT" "$MIS_ROOT/agentic" "$MIS_ROOT/em8q"
CELLS_FILE="$MIS_ROOT/cells$SFX.json"; CELLS_FULL="$MIS_ROOT/cells_full$SFX.json"; EP_SNAP="$MIS_ROOT/endpoint.snapshot$SFX.json"
f22_stop_requested && { f22_log "wave STOP present — not starting"; exit 0; }
[ -e "$MIS_ROOT/STOP" ] && { f22_log "$MIS_ROOT/STOP present — not starting (remove it to run)"; exit 0; }
f22_driver_venv; f22_serving_venv
ORG_NAMES=(); f22_standup_extra_reset   # start from the inherited launch-time hook, never an empty array (common.sh, 2026-09-16)
case "$MODE" in ladder16k|vendor)
  # THE INHERITED GEOMETRY IS A MEASUREMENT. These two modes serve at (LADDER_L, LADDER_S) = the SOURCE run's attested
  # (S x 16384) budget re-shaped to the 32768 window — a footprint this exact stack already served to completion
  # (qwen35_4b: 7 h at 44 x 16384, then 5,303 rung responses at 22 x 32768 across seven shards). The geometry table
  # carries a PROVISIONAL conservative ceiling for models it never smoked (qwen35_4b 360,448 = half the served budget),
  # and the standup refuses an over-ceiling launch unless told it is deliberate. It is: the source run is the
  # measurement the ceiling asks for. `full` mode keeps the table's identity geometry (a fresh standup has no source
  # run to inherit from).
  F22_STANDUP_EXTRA+=(--allow-over-budget); export F22_STANDUP_EXTRA
  f22_log "$MODE: standup gets --allow-over-budget — geometry ${LADDER_L}x${LADDER_S} inherited from the source run's attested budget (a measurement), not the table's provisional ceiling";;
esac
if [ "$MODE" = organisms ]; then
  # the three r=32 EM organisms (D-user 2026-09-10; the rank-1 one is dropped), from the OFFLINE HF cache —
  # the same resolver as misalign_em_organisms.sbatch (refs/main may be missing on the shared cache)
  snap(){ "$SPY" - "$1" <<'PYSNAP'
import os, sys, glob
from huggingface_hub.constants import HF_HUB_CACHE
repo = sys.argv[1]; d = os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"))
def complete(s):
    return os.path.isfile(os.path.join(s, "adapter_config.json")) and bool(glob.glob(os.path.join(s, "*.safetensors")))
try:
    main = open(os.path.join(d, "refs", "main")).read().strip()
    cand = os.path.join(d, "snapshots", main)
    if complete(cand):
        print(cand); sys.exit(0)
except OSError:
    pass
snaps = sorted((s for s in glob.glob(os.path.join(d, "snapshots", "*")) if complete(s)), key=os.path.getmtime)
if not snaps:
    sys.exit(3)
print(snaps[-1])
PYSNAP
  }
  if [ "$TARGET" = llama8b_org ]; then
    for org in bad-medical-advice risky-financial-advice extreme-sports; do
      repo="ModelOrganismsForEM/Llama-3.1-8B-Instruct_$org"
      sdir=$(snap "$repo") || f22_fatal "organisms: $repo is not in the offline HF cache on this cluster (hf-cluster-fetch it first)"
      ORG_NAMES+=("em-$org"); F22_STANDUP_EXTRA+=(--organism "em-$org=$sdir")
    done
  else
    # the LOCAL organisms this lane trained (scripts/sbatch/em_organism_train.sbatch): every
    # $SHARED/exp/models/em_organisms/<KEY>/<tag>/adapter with a manifest, or the tags named in EM_ORG_TAGS (space-separated,
    # the submitting shell's environment through Slurm's implicit export — never --export=). Name `em-<tag>`.
    ORG_ROOT="$SHARED/exp/models/em_organisms/$KEY"
    if [ -n "${EM_ORG_TAGS:-}" ]; then read -r -a ORG_TAGS <<< "$EM_ORG_TAGS"; else
      ORG_TAGS=(); for d in "$ORG_ROOT"/*/adapter; do [ -f "$d/adapter_config.json" ] && ORG_TAGS+=("$(basename "$(dirname "$d")")"); done
    fi
    [ "${#ORG_TAGS[@]}" -ge 1 ] || f22_fatal "organisms: no trained organism under $ORG_ROOT/<tag>/adapter (em_organism_train.sbatch first)"
    for tag in "${ORG_TAGS[@]}"; do
      adir="$ORG_ROOT/$tag/adapter"
      [ -f "$adir/adapter_config.json" ] && [ -s "$adir/adapter_model.safetensors" ] || f22_fatal "organisms: $adir is not a complete PEFT adapter"
      [ -s "$ORG_ROOT/$tag/manifest.json" ] || f22_fatal "organisms: $ORG_ROOT/$tag/manifest.json missing — an adapter without its training record is not servable here"
      grep -q "\"base_revision\": \"$(sed -nE 's/^  target_revision: ([0-9a-f]{40}).*/\1/p' "$MANIFEST" | head -1)\"" "$ORG_ROOT/$tag/manifest.json" \
        || f22_log "organisms: WARNING $tag's manifest base_revision differs from the manifest's pinned target_revision (or the pin is not a 40-hex line) — check before reading results"
      ORG_NAMES+=("em-$tag"); F22_STANDUP_EXTRA+=(--organism "em-$tag=$adir")
    done
  fi
  export F22_STANDUP_EXTRA
  f22_log "organisms: ${ORG_NAMES[*]} (standup extra: ${F22_STANDUP_EXTRA[*]})"
fi
# Target llama8b_org — the organisms transfer target — is served under Meta's STOCK Llama-3.1 chat
# template (configs/capeval/chat_templates/llama31_instruct_stock.jinja), not the pinned NousResearch tokenizer's
# 348-byte minimal one: the EM organisms were trained under the stock template and read ~4x less misaligned
# without its dated system header. The generator pins the file's sha16 in the manifest; the standup re-hashes.
TPL_OVERRIDE=""
case "$TARGET" in llama8b_org) TPL_OVERRIDE="configs/capeval/chat_templates/llama31_instruct_stock.jinja";; esac
if [ -n "$TPL_OVERRIDE" ]; then
  [ -f "$REPO_DIR/$TPL_OVERRIDE" ] || f22_fatal "chat template override $TPL_OVERRIDE is missing from the tree"
  TPL_SHA=$(sha256sum "$REPO_DIR/$TPL_OVERRIDE" | cut -c1-16)
  TPL_PIN=$(sed -nE 's/^  chat_template_hash: ([0-9a-f]{16}).*/\1/p' "$MANIFEST" | head -1)
  [ "$TPL_SHA" = "$TPL_PIN" ] || f22_fatal "manifest pins chat_template_hash $TPL_PIN but $TPL_OVERRIDE hashes $TPL_SHA — regenerate (make_misalign_manifest.py)"
  F22_STANDUP_EXTRA+=(--chat-template "$REPO_DIR/$TPL_OVERRIDE"); export F22_STANDUP_EXTRA
  f22_log "chat template: STOCK Llama-3.1 ($TPL_OVERRIDE, sha16 $TPL_SHA) — the organisms' native render (D43)"
fi
# F22_ONLY_CELLS: an OPTIONAL cell subset (CID[,CID...]) for a full-mode standup, so cells already measured on
# another view of the same seed (the paired views serve the constitution LoRA + its difference beside the composed arm) are
# not re-measured; a plain shell variable in the submitter's environment, carried by Slurm's implicit export (never --export=).
# The shard branches below assign ONLY_CELLS themselves and keep winning; the runner's --only-cells does the filtering.
ONLY_CELLS="${F22_ONLY_CELLS:-}"
if [ "$MODE" = ladder16k ] && [ "$SHARDS" -gt 1 ]; then
  ONLY_CELLS=$(cd "$REPO_DIR" && PYTHONPATH=src "$MPY" - "$SRC_ROOT/agentic" "$SHARDS" "$SHARD" <<'PY'
import sys
from sisteer.capeval.misalign.runner import truncated_selection
src, n, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
sel, _ = truncated_selection(src)
bins = [[0, []] for _ in range(n)]
for w, cid in sorted(((len(v), c) for c, v in sel.items()), reverse=True):   # heaviest first, greedy
    b = min(bins, key=lambda x: x[0]); b[0] += w; b[1].append(cid)
print(",".join(sorted(bins[k][1])))
PY
) || f22_fatal "shard binning failed"
  [ -n "$ONLY_CELLS" ] || { f22_log "shard $SHARD/$SHARDS: no cell with selected truncations — nothing to do"; exit 0; }
  f22_log "shard $SHARD/$SHARDS runs cells: $ONLY_CELLS"
fi

# THE SERVER-DEFAULT SAMPLING POLICY: F22_SERVER_DEFAULT_POLICY (a SCALAR — Slurm's implicit
# export, never --export=) rides through f22_standup to mt_serving_up.py --server-default-policy and is asserted back there from
# endpoint.json + /v1/models. `engine_defaults` (the reused Qwen misalign anchor decoded at vLLM's neutral defaults,
# before the server-side vendor pin) is verified from the server log (anchor_sampler_preflight.py), so it
# turns request logging on (F22_LOG_REQUESTS=1 -> --enable-log-requests; the prompt stays unlogged). Unset = today's job, byte for byte.
case "${F22_SERVER_DEFAULT_POLICY:-}" in
  "") ;;
  contract_override|engine_defaults) f22_log "server-default sampling policy: $F22_SERVER_DEFAULT_POLICY (F22_SERVER_DEFAULT_POLICY; asserted back from the standup's stamps)";;
  *) f22_fatal "F22_SERVER_DEFAULT_POLICY must be contract_override or engine_defaults, got '$F22_SERVER_DEFAULT_POLICY'";;
esac
if [ "${F22_SERVER_DEFAULT_POLICY:-}" = engine_defaults ] && [ "${F22_LOG_REQUESTS:-0}" != 1 ]; then
  F22_LOG_REQUESTS=1; f22_log "engine_defaults is verified from the server log: F22_LOG_REQUESTS=1 (--enable-log-requests) set for this standup"
fi
# SERVED != MEASURED: the config's misalign_policy block (a LANE_POLICY config; absent = today's rule: every served leg
# incl. `none` is measured). Read here so the RESULT line and the log say what this job measured before a GPU minute.
MEASURE_NONE=$("$MPY" -c 'import sys,yaml
p=(yaml.safe_load(open(sys.argv[1])) or {}).get("misalign_policy") or {}
v=p.get("measure_none", True); assert isinstance(v, bool), p; print("1" if v else "0")' "$CFG") || f22_fatal "$CFG: misalign_policy.measure_none is not a boolean"
[ "$MEASURE_NONE" = 1 ] || f22_log "LANE POLICY: measure_none false — the none leg (and its derived controls twin) is SERVED for the health checks and steering-bites and NEVER measured; the runner's cells.json holds the steered cells only (misalign_cells.py)"
# --- the misalign bank view (the key's wave view + alphasteer, pin-verified) ------------------------------------
bash scripts/sdfbase/full22/misalign/misalign_bank_view.sh "$TARGET" --python "$MPY" || f22_fatal "misalign bank view refused for $TARGET"
export TT_TWO_SI_DEFENSES_DIR="$SHARED/exp/objects/misalign/$TARGET/cells_view"
mapfile -t ALL_CELLS < <("$MPY" -c 'import sys,yaml; print("\n".join(yaml.safe_load(open(sys.argv[1]))["explicit_cells"]))' "$CFG")
[ "${#ALL_CELLS[@]}" -ge 1 ] || f22_fatal "$CFG lists no cell"   # s3r-fix: a steered-wave misalign config lists exactly its measured cells (no none, no prompted arm)
for c in "${ALL_CELLS[@]}"; do   # `none` and the prompted baselines serve no object: no view entry
  case "$c" in none|prompted__*) continue;; esac
  [ -f "$TT_TWO_SI_DEFENSES_DIR/$c/object.pt" ] || f22_fatal "view $TT_TWO_SI_DEFENSES_DIR/$c lacks object.pt"
  n_id=0; for f_id in train_budget_meta.json fit_provenance.json derived_meta.json; do [ -f "$TT_TWO_SI_DEFENSES_DIR/$c/$f_id" ] && n_id=$((n_id+1)); done
  [ "$n_id" -eq 1 ] || f22_fatal "view entry $c declares $n_id identity sidecars (need exactly one)"
done
f22_log "misalign view OK: $TT_TWO_SI_DEFENSES_DIR; config cells: ${ALL_CELLS[*]}"

# --- standup ---------------------------------------------------------------------------------------------------
TARGET_PID=""; RUNNER_PID=""; RC=1; RC_A=""; RC_E=""; N_A=0; N_E=0
result_line(){ echo "$(date -Is) job=${SLURM_JOB_ID:-manual} target=$TARGET model_key=$KEY view=$VIEW sha=${WAVE_SHA:0:12} cluster=$CLUSTER rc=$RC agentic_rc=${RC_A:-na} em8q_rc=${RC_E:-na} agentic_responses=$N_A em8q_responses=$N_E"; }
cleanup(){
  [ -n "${RUNNER_PID:-}" ] && kill -0 "$RUNNER_PID" 2>/dev/null && { f22_log "stopping the runner (pid $RUNNER_PID)"; kill -TERM "$RUNNER_PID" 2>/dev/null; wait "$RUNNER_PID" 2>/dev/null; }
  touch "$OUT/STOP" 2>/dev/null || true
  if [ -n "${TARGET_PID:-}" ]; then for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; fi
  result_line >> "$MIS_ROOT/RESULT"
  f22_result "misalign $TARGET $(result_line)"
}
trap cleanup EXIT
trap 'f22_log "SIGTERM — draining"; RC=143; exit 143' SIGTERM SIGINT SIGQUIT
TARGET_GPU="$CUDA_VISIBLE_DEVICES"
if [ "$MODE" = ladder16k ] || [ "$MODE" = vendor ]; then
  f22_standup petri "$OUT" "" "" "$LADDER_L" "$LADDER_S" || { f22_log "standup refused"; exit 1; }
  # the 16k window: a ~3k-token scenario (the vendor run also serves the ~5.9k constitution SI) + the 16384 cap +
  # margin fit 32768. Read back, never assumed.
  [ "$TARGET_L" = "$LADDER_L" ] || { f22_log "REFUSING TRAFFIC: standup resolved max_model_len $TARGET_L, this $MODE run is specified at $LADDER_L"; exit 1; }
else
  f22_standup petri "$OUT" "" "" "" "" || { f22_log "standup refused"; exit 1; }
  # THE GEOMETRY THIS EVAL IS SPECIFIED AT: the certified Petri window (constitution SI ~5.9k tokens + a ~3k-token
  # agentic scenario + a 3000-token cap fit inside it). Read back from the standup's attested values, never assumed.
  [ "$TARGET_L" = 16384 ] || { f22_log "REFUSING TRAFFIC: standup resolved max_model_len $TARGET_L, this eval is specified at 16384"; exit 1; }
fi
cp "$OUT/endpoint.json" "$EP_SNAP"   # the standup unlinks endpoint.json at teardown; keep the record

# --- health wait + one-request smoke on the bare route ----------------------------------------------------------
PROXY_API_KEY="petri-proxy"   # the proxy's dummy bearer (petri_target.DEFAULT_API_KEY); loopback-only, not a secret
for i in $(seq 1 60); do curl -s -m 20 -H "Authorization: Bearer $PROXY_API_KEY" "$PROXY_URL/models" | grep -q '"petri__base"' && break; sleep 5; done
curl -s -m 20 -H "Authorization: Bearer $PROXY_API_KEY" "$PROXY_URL/models" | grep -q '"petri__base"' || { f22_log "proxy /v1/models never listed petri__base"; exit 1; }
SMOKE=$(curl -s -m 600 -H "Authorization: Bearer $PROXY_API_KEY" -H 'Content-Type: application/json' "$PROXY_URL/chat/completions" \
  -d '{"model":"petri__base","messages":[{"role":"user","content":"Reply with the single word: ok"}],"max_tokens":8,"temperature":0}')
"$MPY" -c 'import json,sys; d=json.loads(sys.argv[1]); c=d["choices"][0]["message"]["content"]; print("smoke ok:", repr(c[:40]), "usage", d.get("usage"))' "$SMOKE" \
  || { f22_log "smoke request on petri__base failed: ${SMOKE:0:600}"; exit 1; }

# THE ANCHOR SAMPLER PREFLIGHT: F22_ANCHOR_PREFLIGHT=<the reused misalign anchor's record —
# its resolved_sampler.json, its serving log, or a client record with F22_ANCHOR_SERVER_POLICY> gates the runner: ONE request shaped
# like the runner's ({"temperature": 1.0} only, max_tokens = this run's cap) at petri__base — a health-class probe on the served
# `none` route, written to no store — then the ENGINE-RESOLVED tuple for exactly that request is compared against the anchor's
# (common.sh f22_anchor_preflight). Anything but AGREES tears the standup down before the runner starts. Unset = no gate.
if [ -n "${F22_ANCHOR_PREFLIGHT:-}" ]; then
  PF_CAP=3000; case "$MODE" in ladder16k|vendor) PF_CAP=16384;; esac
  f22_anchor_preflight misalign "$TARGET" "$OUT" "$F22_ANCHOR_PREFLIGHT" "$PF_CAP" ',"temperature":1.0' || exit 1
fi
# --- cells.json: the runner's contract, from the standup's own enumeration (misalign_cells.py: SERVED != MEASURED) -----------
ORG_CSV=$(IFS=,; echo "${ORG_NAMES[*]:-}")
export F22_SERVED_CLUSTER="$CLUSTER"   # into cells.json's serving block (rows.py lifts `serving` onto every row)
# the baselines' PINS ride the rows too (the rule: selection_outcome on every row)
if [ "$BASELINES" = 1 ]; then export F22_BASELINE_PINS="$REPO_DIR/configs/misalign_baseline_pins.json" F22_BASELINE_FAMILY="$KEY"; fi
"$MPY" scripts/sdfbase/full22/misalign/misalign_cells.py --cells-served "$OUT/cells_served.json" --endpoint "$EP_SNAP" \
  --out "$CELLS_FILE" --full "$CELLS_FULL" --target "$TARGET" --wave-sha "$WAVE_SHA" --organisms "$ORG_CSV" --model-key "$KEY" --wave-view "$VIEW" \
  --config "$CFG" || exit 1
cp "$CELLS_FILE" "$MIS_ROOT/agentic/cells$SFX.json"; cp "$CELLS_FILE" "$MIS_ROOT/em8q/cells$SFX.json"
if [ "$MODE" = vendor ] && [ "$SHARDS" -gt 1 ]; then
  # vendor shards: a fresh run has no truncation selection to bin by — the LEGS are the unit (sorted cell ids,
  # shard k takes every SHARDS-th from k), read from THIS standup's cells.json (the runner's own contract, so a
  # leg the proxy does not serve can never be named)
  ONLY_CELLS=$("$MPY" -c 'import json,sys
d = json.load(open(sys.argv[1])); cells = d["cells"] if isinstance(d, dict) else d   # object form since c00d86c0
ids = sorted(c["cell_id"] for c in cells); n, k = int(sys.argv[2]), int(sys.argv[3])
print(",".join(ids[k::n]))' "$CELLS_FILE" "$SHARDS" "$SHARD") || f22_fatal "vendor shard binning failed"
  [ -n "$ONLY_CELLS" ] || { f22_log "shard $SHARD/$SHARDS: no legs (more shards than legs) — nothing to do"; RC=0; RC_A=0; RC_E=0; exit 0; }
  f22_log "shard $SHARD/$SHARDS runs legs: $ONLY_CELLS"
fi

# --- the runner, on the node, against loopback; concurrency = the attested max_num_seqs ---------------------------
run_family(){   # $1 = family, $2... = family args  -> sets RC_F
  local fam="$1"; shift
  local log="$MIS_ROOT/$fam/runner_${SLURM_JOB_ID:-manual}.log"
  f22_log "runner: family $fam ($*) concurrency $TARGET_S -> $MIS_ROOT/$fam (log $log)"
  # tee into the job log too: the watcher reads the .out for liveness, and a runner that only
  # writes its own file looks like a 40-minute stall (job 478866). Subshell + pipefail keeps the
  # runner's exit status (a bare pipeline would report tee's).
  ( set -o pipefail; cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" "$MPY" -m sisteer.capeval.misalign.runner \
      --proxy-url "$PROXY_URL" --cells "$CELLS_FILE" --family "$fam" --out "$MIS_ROOT/$fam" \
      --concurrency "$TARGET_S" "$@" 2>&1 | tee "$log" ) &
  RUNNER_PID=$!
  wait "$RUNNER_PID"; RC_F=$?; RUNNER_PID=""
  kill -0 "$TARGET_PID" 2>/dev/null || { f22_log "the standup died during family $fam (see $OUT/standup.log)"; RC_F=${RC_F:-3}; [ "$RC_F" = 0 ] && RC_F=3; }
  [ "$RC_F" = 0 ] || { f22_log "runner $fam FAILED rc=$RC_F; log tail:"; tail -30 "$log"; }
}
count_rows(){ cat "$MIS_ROOT/$1"/*/responses.jsonl 2>/dev/null | wc -l; }
AGENTIC_ARGS=(--n 10 --behaviors leaking murder --max-tokens 3000)
[ "$MODE" = ladder16k ] && AGENTIC_ARGS=(--n 10 --behaviors "${BEHAVIORS[@]}" --max-tokens 16384 --rerun-truncated-from "$SRC_ROOT/agentic")
[ "$MODE" = vendor ] && AGENTIC_ARGS=(--n 10 --behaviors "${BEHAVIORS[@]}" --max-tokens 16384 --sampler vendor)
[ -n "$ONLY_CELLS" ] && AGENTIC_ARGS+=(--only-cells "$ONLY_CELLS")
# The Qwen seed-replicate battery: the runner's per-response decode seed rides its
# --seed-policy / --seed-base, passed through from F22_MIS_SEED_POLICY / F22_MIS_SEED_BASE in the launch environment
# (Slurm's implicit export; never --export=). Both unset (every campaign launch) = the argv above, byte for byte; the
# runner itself refuses a base without the policy and the policy without a base.
[ -n "${F22_MIS_SEED_POLICY:-}" ] && AGENTIC_ARGS+=(--seed-policy "$F22_MIS_SEED_POLICY")
[ -n "${F22_MIS_SEED_BASE:-}" ] && AGENTIC_ARGS+=(--seed-base "$F22_MIS_SEED_BASE")
# D2: nemotron3nano4b is EXCLUDED FROM LEAKING (2.9% base rate, no headroom). The runner's --exclude is keyed
# per CELL (`<cell_id>:<behavior>`, repeatable; an unknown cell id is a refusal), so the target-wide exclusion
# is spelled out for every leg of cells.json — each leg then records `excluded_behaviors: [leaking]` in its
# own run record, which is what carries the note downstream; the murder items run on every leg as usual.
case "$KEY" in nemotron3nano4b)
  # cells.json is the OBJECT form {"cells": [...], "serving": {...}} since c00d86c0 — iterating the top level
  # walks its two keys, and `c["cell_id"]` on the string "cells" raised TypeError inside the process
  # substitution, which `while read` swallowed: job 487635 ran with ZERO --exclude args (the ladder16k rerun
  # selection kept it to murder items anyway, but the legs' run records lost `excluded_behaviors`).
  EXCL_IDS=$("$MPY" -c 'import json,sys
d = json.load(open(sys.argv[1])); cells = d["cells"] if isinstance(d, dict) else d
print("\n".join(c["cell_id"] for c in cells))' "$CELLS_FILE") || f22_fatal "leaking-exclusion cell list failed"
  [ -n "$EXCL_IDS" ] || f22_fatal "leaking-exclusion cell list is empty"
  while IFS= read -r cid; do AGENTIC_ARGS+=(--exclude "$cid:leaking"); done <<< "$EXCL_IDS"
  f22_log "nemotron3nano4b: leaking excluded on every leg ($(wc -l <<< "$EXCL_IDS") legs; ${#AGENTIC_ARGS[@]} runner args)";;
esac
if [ "$MODE" = organisms ]; then
  RC_A=0; N_A=0; f22_log "agentic skipped in organisms mode (the organism panel is the EM evals — em8q)"
else
  run_family agentic "${AGENTIC_ARGS[@]}"; RC_A=$RC_F; N_A=$(count_rows agentic)
  f22_log "agentic done rc=$RC_A responses=$N_A"
fi
if [ "$BASELINES" = 1 ]; then
  RC_E=0; N_E=0; f22_log "em8q skipped for the baselines panel $TARGET (the ruling names the agentic column — leaking + murder — only)"
elif [ "$MODE" = ladder16k ] || [ "$MODE" = vendor ]; then
  RC_E=0; N_E=0; f22_log "em8q skipped in $MODE mode (the paper's 600-token protocol is neither laddered nor re-sampled: the ask is the agentic column)"
else
  # PER-REQUEST TIMEOUT for em8q. The runner's default is 7,500 s, sized for the agentic family's 16k cap; at em8q's
  # 600-token cap a response takes ~30 s, so a lost response parks the runner for over two hours — longer than this
  # job's walltime, which is exactly what happened on the vendor rung (job 549505: the engine reported 0 running
  # requests and its log went quiet at 03:17 while the runner sat at 399/400 for 24 minutes, then the allocation was
  # thrown away). 900 s is 30x the observed generation time, so it cannot censor a slow-but-live draw, and the
  # runner's own retry-with-backoff takes it from there.
  run_family em8q --n 50 --max-tokens 600 --timeout-s "${EM8Q_TIMEOUT_S:-900}" "${EM8Q_ARGS[@]}"; RC_E=$RC_F; N_E=$(count_rows em8q)
  f22_log "em8q done rc=$RC_E responses=$N_E"
fi

# --- tear down ------------------------------------------------------------------------------------------------------
f22_standup_stop
RC=0; [ "$RC_A" = 0 ] && [ "$RC_E" = 0 ] || RC=1
f22_log "misalign $TARGET ($KEY/$VIEW) finished rc=$RC (agentic rc=$RC_A n=$N_A; em8q rc=$RC_E n=$N_E); outputs under $MIS_ROOT"
exit $RC
