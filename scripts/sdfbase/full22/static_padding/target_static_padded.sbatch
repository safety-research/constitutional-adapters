#!/bin/bash
#SBATCH --job-name=padstat
#SBATCH --partition=general
#SBATCH --qos=high                     # the standing default; on another cluster pass its own QoS (its site block's CLUSTER_QOS).
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/workspace/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# full22 PADDED STATIC leg (docs/static_padding.md): ONE served TARGET (1 GPU, the native engine) + the four static
# one-shot families (pasttense / codeattack / renellm / grok420) at the 16,384 completion cap — the static ladder's CEILING
# (THE CAP IS THE LADDER'S CEILING, below) — over the 259-goal confirmatory pool, for the three PROMPTED-SIDE arms of ONE target — the same-wave bare anchor `none`,
# prompted__safety_refusal, prompted__si_constitution (no fitted object, no bank view entry; steered arms come later
# under their own config pair) — at EVERY padding level of the run (default 0, 16,384, 65,536 neutral PG-19 filler
# tokens added to each item's user turn; placement before, mode fixed), decoded at the standing vendor-preset default
# (the driver injects it for a served static run whose config carries no sampled_decode block; never --greedy).
#
# ONE STANDUP FOR EVERY LEVEL, at ONE window (the positional <window>; RAISE THE WINDOW, never shrink the pad — the
# pad values are the experiment's axis, the window is ours to choose — so 98,304 for every level, sized by the MEASURED
# floor below). The pad is REQUEST CONTENT, not server state: running every level
# against ONE server holds the engine identity constant (the same weights, the same scheduler geometry S x L, the same
# CUDA-graph capture surface, the same attested facts) and makes the pad the ONLY thing that varies between the three
# stores. Three standups at three windows would confound the pad with the geometry (a different S, a different batch
# composition, a different KV footprint per level). The zero-pad control costs almost nothing extra at the big window:
# KV is read over the ACTUAL context of each sequence, not over the window, so a short request at a 98,304 window
# decodes like a short request at 4,864; only the width S is smaller (below).
#
# ONE LEVEL PER JOB (paper-companion-repo D21(3), 2026-09-29). The paragraph above argued one standup for every level, and
# this script once took a comma list of pads and ran them in sequence against one server. It now runs exactly ONE level
# and REFUSES a comma list by name: every level is its own registry instrument (THE STORE IDENTITY, below) and the
# registry's pending file is idempotent per Slurm job id (common.sh f22_registry_register: one <pending-dir>/<job id>.json,
# kept if present), so a second level in the same job could never register. Submit one job per level at the IDENTICAL
# window / manifest / cap: the engine INSTANCE is not instrument identity, and the standup argument survives across
# jobs — every level's standup resolves the same geometry from the same config body, and the GEOMETRY PIN below refuses
# traffic on any that does not.
#
# THE WINDOW'S FLOOR IS MEASURED, NOT EXTRAPOLATED (cluster B, 2026-09-22, the pinned Qwen3.5-4B tokenizer, EVERY
# variant of EVERY goal of all four families — 2,590 per cell: pasttense 518, codeattack 777, renellm 777 (its frozen
# corpus renders on-cluster only), grok420 518 — under each cell's system instruction, the max taken IN TOKENS): the
# constitution SI renders to 5,973 tokens, the longest [SI + payload] to 6,412 (grok420 under the constitution; renellm's
# longest 6,075, codeattack 6,096, pasttense 6,015; the char-longest proxy under-read grok420 at 6,403). At the 65,536
# level the worst row needs 6,412 + 66,847 (the pad at its 2 % worst case) + 16,384 (the ceiling cap) = 89,643; the
# harness's own rule (tokens + cap + reserve 8,192) says 90,112, which is also the smallest 1,024-aligned window carrying
# the render (the 4,096-cap sizing's 81,920 is 7,723 short at this cap). 98,304 carries every row with 8,661 tokens of
# headroom — one reserve, 8,192, over the harness rule — at the SAME derived S as 90,112 (below: the kernel ceiling holds
# S = 5 from 90,112 to 104,448, so the headroom is free) and at the SAME 491,520 batched tokens the 4,096-cap sizing's
# 6 x 81,920 allocated, so the memory sum below is unchanged; it absorbs a template or tokenizer drift the preflight would
# otherwise turn into a refusal.
#
# CONCURRENCY IS DERIVED, NEVER FORCED — AND PINNED TO THE STEERED ARMS' GEOMETRY. Only the window is passed to
# f22_standup; S falls out of the launcher's own arithmetic in two steps, both recorded in endpoint.json /
# standup_jobs.jsonl: (1) the identity S at THIS model's measured batched-token budget (geometry.ModelBudget.seqs_at:
# 741,888 // 98,304 = 7; the hybrid per-slot cap 256 does not bind), then (2) the LoRA kernels' int32 index ceiling
# ((2^31-1) // D, D = 4,096 = the widest adapted module, o_proj's in-features: 524,287 batched tokens -> 524,287 //
# 98,304 = 5), applied by native_standup.bring_up_native as the RECORDED adjustment `lora_kernel_index_ceiling` 7 -> 5.
# Step 2 fires on THIS prompted-only standup because the NATIVE path ALWAYS registers the zero adapter `none__lora0`
# (bring_up_native: export_or_reuse_zero -> all_blocks -> lora_kernel_ceiling_geometry; the serving section of
# CLAUDE.md: a view with no LoRA cell still stands up natively with the zero adapter alone, so base and steered
# configurations share ONE engine). That is the point, not an accident: a padded STEERED leg at this window (the qwen
# main view's lora_r4 / lora_subtraction o_proj adapters and the psr cells' residual slot, whose payload rides a rank-1
# ZERO o_proj LoRA — every one D = 4,096) resolves the SAME 5 x 98,304, so these three arms sit on the geometry the arms
# they will be compared with will get. An ADAPTER-FREE standup (F22_SERVING_PATH=lens, a bare `vllm serve`) would derive
# S = 7 and put these arms on a different engine identity, forcing a re-run when the steered arms land — so DO NOT
# "optimise" this back to one: the GEOMETRY PIN below refuses traffic unless the attested engine is lora_native, the
# zero adapter is among the attested adapters, and the attested S equals the kernel-ceiling S the geometry module
# derives for this window (the launcher's own arithmetic, never a literal). Memory at 5 x 98,304 = 491,520 batched
# tokens: weights 8.7 GiB + full-coverage KV 491,520 x 32 KiB = 15.0 GiB + the warmup MLP intermediate 491,520 x 9,216
# x 2 B = 8.4 GiB ~= 32 GiB of the ~126 GiB the 0.9 utilization reserves on a 139.81 GiB card (the preflight's check 5
# re-derives all of it from the geometry module and the snapshot's config.json).
#
# THE CAP IS THE LADDER'S CEILING — 16,384 (ladder_decide.CEILING), not the campaign's 4,096 — AT EVERY LEVEL, the p0
# control included (make_padding_configs.STATIC_CAP; padding_preflight's check_cap refuses anything else by name).
# Recorded because a 16,384 cap on a leg whose completions are mostly short LOOKS wasteful, and is not: this leg is
# PREFILL-bound (the cost model below), so a cap costs only for tokens actually generated, and the standing truncation
# rule would otherwise escalate any (cell, readout) above 1 % at-cap to a larger rung — on THIS leg a rung re-pays the
# FULL pad prefill for every re-drawn item, and the rung driver refuses padded stores outright
# (escalate_static.assert_base_unpadded: it re-draws the UNPADDED attack_prompt, a different measurement wearing the
# store's name), so a >1 % at-cap share could only be met by regenerating the entire level at a larger cap. One pass at
# the ceiling: no rung can be needed, because there is none above. Do NOT "optimise" this back to 4,096 — the saving is a
# few hours of decode (below); the price is a whole level again.
#
# KV IS 32 KiB PER CONTEXT TOKEN ON THIS MODEL, NOT 128. Qwen3.5-4B is a HYBRID: `layer_types` in its config.json names
# 8 full_attention + 24 linear_attention (gated-deltanet) layers (`full_attention_interval` 4); only the 8 hold a KV that
# grows with context (2 x 8 layers x 4 kv heads x 256 head_dim x 2 B = 32 KiB), the 24 hold a fixed-size per-slot state.
# A 32-layer count over-reads the KV fourfold (an earlier draft of this header did exactly that) — read `layer_types`,
# never multiply by num_hidden_layers.
#
# THE COST MODEL: on THIS target the leg is PREFILL-bound. Qwen3.5-4B
# is a HYBRID-RECURRENT target (gated-deltanet), so PREFIX CACHING IS OFF — and not merely disabled: vllm_server
# .hybrid_recurrent_model's docstring records it as STRUCTURALLY IMPOSSIBLE on this stack (the only caching mode vLLM 0.25.1
# offers a hybrid model, --mamba-cache-mode=align, asserts chunked prefill at engine init, and this stack forbids chunked
# prefill as a CORRECTNESS requirement: steering hooks cannot survive chunk shapes). So unlike a dense target every request
# pays the full pad's prefill (~73k tokens at the 65,536 level, ~9e14 FLOP: seconds of the card per request, 7,770 requests
# per level) — a dense target's "prefill nearly free" reading does not apply here. Decode is the smaller half and is paid only
# for tokens actually generated: a step re-reads its sequences' KV (32 KiB per context token: ~2.2 GiB per sequence at the
# 65,536 pad vs ~0.15 GiB at the short window; ~11 GiB per step at width 5, plus the 8.7 GiB of weights), so a generated
# token costs ~1-1.5 ms of the card amortised over the width, and a completion that runs to the 16,384 ceiling ~20 s.
# --time is sized from the geometry, not measured: 7,770 requests per level (2,590 items x 3 cells); p16384 ~4-6 h, p65536
# ~8-12 h by that estimate, ~2-5 h by a FLOP-derived one, the gap being how well the gated-deltanet prefill utilises the
# card. THE CEILING CAP'S MARGINAL COST is bounded by the items that would have hit 4,096: at a 10 % at-cap share (the only
# measured share on this target is grok420's 3-19 % at-cap rows in the sampled lane, most of them loops, which run to
# whatever cap they are given — docs/cap_ladder.md) every such item running the full 12,288 extra tokens adds ~9.5M decode
# tokens per level, ~2-3.5 h per level at the amortised cost above; at 3 % it is under an hour per level. That moves the
# pessimistic two-level total from ~12-18 h to ~15-24 h and the optimistic one from ~5 h to ~7 h. THOSE ARE WHOLE-WAVE
# numbers (all four families); the intended shape is ONE JOB PER FAMILY, so divide by four: ~3.75-6 h pessimistic,
# ~1.75 h optimistic, per job. --time is 4 h — the standing task size of about four hours — and NOT 2x
# the pessimistic estimate, because a long walltime is the thing that stops a job being backfilled at all: measured
# on this cluster the same day, ~22 GPUs sat free while an 8 h request waited on (Priority) behind reservations a 4 h
# request can slot under. The leg is CHEAP to resume: run_static resumes per goal, so a TIMEOUT costs one resubmit
# (~10 min of standup), never the generations already filed. If the first level's measured wall time says two levels will
# not fit, run them as two submissions (the pads positional) rather than lengthening the request. Measure the first
# level's wall time (the sbatch logs each level's wall time; the vLLM metrics log is under $SERVE_DIR/static/).
#
# THE STORE IDENTITY: one padding contract per store (static_padding.assert_static_padding_consistent), so each level
# has its OWN generated config with its own private_run (pad_static_<key>_p<pad>; make_padding_configs.py) whose
# `attacks.static_padding` block PINS tokens / placement / mode / reserve. The drivers pass NO --static-padding-tokens:
# the level travels with the config, so a driver cannot be run under the wrong level. The FILLER PATH rides the argv
# (--static-padding-filler; a cluster-local resource the preflight sha-verifies first), so no field is stated twice.
# The p0 CONTROL is a real run of THIS wave (its own private_run, no padding block) — never a reused older store.
# Stores land under $WAVE_ROOT/static_padding/transcripts_results/<private_run>/<family>/<cell>/shardNN — a lane
# subdirectory beside the campaign's transcripts_results, never inside it: these stores are NOT the canonical
# instrument, carry a `padding` block on every row, and are refused as escalation bases by design. REGISTERED in the
# results registry PER LEVEL (the block "THE REGISTRY IDENTITY OF THIS LEVEL" below): the registry HAS a padding axis
# since migration 0012 (identity.INSTRUMENT_ID_OPTIONAL_FIELDS = padding_tokens +
# padding_identity_sha256), so every level is its OWN instrument — the canonical seeded static instrument for (family,
# model, path) with the level's padding_tokens, the filler contract's hash, the ceiling cap as a one-rung ladder read
# fixed, and the standup window — and its rows can never misfile under the unpadded (eval, instrument). The p0 control
# registers padding_tokens 0 WITH its siblings' contract, never NULL. The job exports F22_REGISTRY_EVALS (its
# families) and SISTEER_REGISTRY_PADDING_JSON (the level block, sisteer.registry.jobplan padding-block) and calls
# common.sh f22_registry_register ONCE, before the standup, under the WAVE view whose objects these arms serve
# (F22_REGISTRY_VIEW, below) — never under the scratch bank view `padstatic`, which is not a model-key spelling.
#
# THE PREFLIGHT (padding_preflight.py, BEFORE the standup takes the GPU for real work): the filler exists at the
# cluster B path with the corpus's exact sha256 and size and parses through the harness's own loader; the model
# snapshot is in this cluster's HF cache; every config's tokens + cap + reserve <= window (the harness's own rule);
# the GEOMETRY this window buys through the geometry module (identity S, the kernel-ceiling S, the memory sum at
# 32 KiB/token KV off the snapshot's own layer_types — check 5); and — the check that matters — the RENDERED need per
# cell: the pinned tokenizer's render of the cell's SI + EVERY variant of EVERY goal of each family (the max in tokens,
# never the char-longest proxy) + the pad at its worst case (tokens x 1.02) + the cap must fit the window. At the 4,096-cap
# sizing's 81,920 it REFUSES naming 89,643 > 81,920 and the smallest aligned window 90,112; at 98,304 it passes with 8,661
# to spare — and it refuses any config whose cap is not the ladder's ceiling (check_cap; the paragraph above).
#
# NO FIGURES: this leg draws none (grading and analysis are post-steps on the grading host), so there is no figures
# trap here — the RESULT line rides out through f22_result -> mirror-figs.sh as in every f22 job.
#
# THE WORKLOAD IS NOT srun-WRAPPED, deliberately (the full22 lineage's shape, target_static.sbatch / target_sampled
# .sbatch): the standup and the drivers are children of this batch shell, bounded by the driver queue, reaped by
# cleanup() below, and --signal=B:TERM@120 delivers the walltime warning to THIS shell so the trap runs inside the
# grace window. Wrapping the body in srun would move the children out of that trap's reach.
#
# Usage (positional only — never --export=; run from the wave checkout on cluster B):
#   sbatch -J padstat-<key> scripts/sdfbase/full22/static_padding/target_static_padded.sbatch \
#       <key> <wave_sha> <window> <pad> [families=pasttense,codeattack,renellm,grok420] [nshards=3] [cells=all]
#     window    the standup's --max-model-len (98,304 decided, 2026-09-22, for the 16,384 ceiling cap; the preflight refuses one
#               that cannot carry the level)
#     pad       EXACTLY ONE padding level (0 | 8192 | 16384 | 32768 | 65536); it needs
#               configs/sdfbase/static_padding/sdfbase_full22_<key>.padstatic_p<pad>.yaml. A comma list is REFUSED by name
#               (one level per job: the registry's pending file is idempotent per job id, so a second level could never
#               register; submit one job per level — the header paragraph ONE LEVEL PER JOB)
#     families  comma subset of pasttense,codeattack,renellm,grok420 (this leg runs the static families only)
#     nshards   shards per (family, cell, level): ~S drivers stay in flight, so more shards than S only shorten the tail
#     cells     comma subset of the config's explicit_cells (default: all three)
#
# THE INTENDED SUBMISSION SHAPE IS ONE JOB PER (LEVEL, FAMILY) (tasks of about four hours, split across GPUs;
# ONE LEVEL PER JOB, as above). Per level, four jobs, each `<key> <sha> <window> <pad> <one family>`,
# each on its own GPU with its own standup (SERVE_DIR carries $SLURM_JOB_ID, so concurrent jobs never share a serving
# dir; the stores are keyed per family / cell / shard, so they never collide either). FAMILY was the free split axis and
# LEVEL was not: every arm of a family stays on ONE engine instance, so the arm-vs-arm contrast is within-standup; the
# pad-vs-pad contrast — the headline — now straddles instances, which is sound only because every level's standup
# attests ONE geometry (the GEOMETRY PIN) and the pad is REQUEST CONTENT, so the instance is not part of the identity.
# THE REGISTRY VIEW (F22_REGISTRY_VIEW, an inherited environment variable, never --export=): the wave view the runs
# register under — default `u0` when the roster names it for <key>, else `main`; it must be a legacy_spellings model-key
# spelling as <key>_<view> (or <key> for main), or the registration refuses by name and the job runs unregistered.
#   On cluster A (not the plan): add --qos=high --output=$SHARED/exp/logs/%x_%j.out on the sbatch line;
#   the body is cluster-agnostic through common.sh.
set -uo pipefail
KEY="${1:?key}"; WAVE_SHA="${2:?wave sha}"; WINDOW="${3:?window (the standup max_model_len)}"
PADS="${4:?pad (ONE level: 0 | 8192 | 16384 | 32768 | 65536; one job per level, a comma list is refused)}"; FAMILIES="${5:-pasttense,codeattack,renellm,grok420}"; NSHARDS="${6:-3}"; CELLS_ARG="${7:-}"
{ [ -n "${SISTEER_SHARED_ROOT:-}" ] && source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA/scripts/sdfbase/full22/common.sh" 2>/dev/null; } \
  || source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit
PAD_DIR=scripts/sdfbase/full22/static_padding
# THE LAUNCH'S CELLS for the kind-aware serving-path default (common.sh f22_select_serving_path): this lane's OWN
# manifest — none + two prompted baselines, nothing steered -> the entry-point default, lora_native (recorded)
MANIFEST="configs/capeval/manifests/petri_full22_${KEY}.padstatic.yaml"
F22_LAUNCH_MANIFEST="$MANIFEST"
f22_init "$WAVE_SHA"
case "$WINDOW" in ''|*[!0-9]*) f22_fatal "window '$WINDOW' is not a positive integer";; esac
[ "$WINDOW" -ge 1 ] || f22_fatal "window must be >= 1"
case "$NSHARDS" in ''|*[!0-9]*|0) f22_fatal "nshards '$NSHARDS' is not a positive integer";; esac
[ -f "$MANIFEST" ] || f22_fatal "missing $MANIFEST — generate it: python $PAD_DIR/make_padding_configs.py"
# THE LEVEL -> its generated config (ONE config per level: the store identity; make_padding_configs.py). PAD_LIST keeps the
# list shape the drivers block iterates, and holds EXACTLY ONE level (header "ONE LEVEL PER JOB").
# --- pads: begin (rendered by tests/test_padding_registration.py under a stubbed f22_fatal) ---
IFS=',' read -ra PAD_LIST <<< "$PADS"
[ "${#PAD_LIST[@]}" -ge 1 ] || f22_fatal "pads '$PADS' names no level"
[ "${#PAD_LIST[@]}" -eq 1 ] || f22_fatal "pads '$PADS' names ${#PAD_LIST[@]} levels — one level per job: the registry's pending file is idempotent per job id, so a second level could never register; submit one job per level"
# --- pads: end ---
declare -A CFG_BY_PAD=()
for pad in "${PAD_LIST[@]}"; do
  case "$pad" in ''|*[!0-9]*) f22_fatal "pad '$pad' is not a non-negative integer (pads=$PADS)";; esac
  [ -z "${CFG_BY_PAD[$pad]+x}" ] || f22_fatal "pad $pad named twice in pads=$PADS"
  c="configs/sdfbase/static_padding/sdfbase_full22_${KEY}.padstatic_p${pad}.yaml"
  [ -f "$c" ] || f22_fatal "missing $c for level $pad — every level has its own generated config (python $PAD_DIR/make_padding_configs.py)"
  CFG_BY_PAD[$pad]="$c"
done
# the standup's config: the FIRST level's (the p0 control in the default order). Every level's config lists the same
# cells and pins the same model; f22_standup reads only those (and the cap, for the load check) from $CFG
CFG="${CFG_BY_PAD[${PAD_LIST[0]}]}"; VIEW=padstatic; TAG="${KEY}_${VIEW}"
IFS=',' read -ra FAMS <<< "$FAMILIES"
[ "${#FAMS[@]}" -ge 1 ] || f22_fatal "families '$FAMILIES' names none"
for fam in "${FAMS[@]}"; do case "$fam" in pasttense|codeattack|renellm|grok420) ;; *) f22_fatal "family '$fam' is not a static one-shot family (this leg runs pasttense,codeattack,renellm,grok420 only)";; esac; done
f22_stop_requested && { f22_log "wave STOP present — not starting"; exit 0; }
SERVE_DIR="$WAVE_ROOT/static_padding/serving/${TAG}_${SLURM_JOB_ID}"; PRIVATE_ROOT="$WAVE_ROOT/static_padding/transcripts_results"
mkdir -p "$SERVE_DIR/drivers" "$PRIVATE_ROOT"; chmod 700 "$PRIVATE_ROOT"
FILLER="${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}/data/pg19_filler.json"
f22_log "padded static: key=$KEY window=$WINDOW pads=${PAD_LIST[*]} families=${FAMS[*]} nshards=$NSHARDS cells=${CELLS_ARG:-all} manifest=$MANIFEST standup_cfg=$CFG filler=$FILLER serve_dir=$SERVE_DIR private_root=$PRIVATE_ROOT"
case ",$FAMILIES," in *,renellm,*)
  export SISTEER_HELDOUT_CORPUS_DIR="${SISTEER_HELDOUT_CORPUS_DIR:-$SHARED/exp/heldout_corpora}"
  # the corpus replay reads: this cluster's copy at $SISTEER_HELDOUT_CORPUS_DIR/renellm.jsonl when there is one, else
  # the paper's corpus as the repository ships it (data/heldout_corpora/renellm.jsonl). The paper's replays as is, any
  # other only when SISTEER_RENELLM_NONPAPER_CORPUS_SHA names it (its results are labelled non-paper). Checked here,
  # under the system python (stdlib only), before anything is stood up
  python3 -I -S scripts/renellm_corpus_check.py \
    || f22_fatal "renellm: the corpus check refused the replay corpus (the reason is above); nothing was stood up";;
esac
f22_driver_venv; f22_serving_venv
# --- registry: begin (rendered by tests/test_padding_registration.py under stubs for $MPY / f22_registry_register / f22_log) ---
# THE REGISTRY IDENTITY OF THIS LEVEL (header "THE STORE IDENTITY"): every family this job runs registers at job start on
# the level's OWN instrument — the canonical seeded static instrument for (family, model, path) + padding_tokens = the level
# + padding_identity_sha256 = the filler contract's hash + the ceiling cap + this window (sisteer.registry.jobplan
# PADDING_ENV; migration 0012) — under the WAVE view whose objects these arms serve (F22_REGISTRY_VIEW: `u0` when the roster
# names it for $KEY, else `main`; the scratch bank view `padstatic` is not a model-key spelling and is never the
# registration view). The level block is computed from the level's config + the filler FILE; an EMPTY block skips the
# registration outright — a padded level must never register on the unpadded instrument. Best-effort by contract (common.sh
# f22_registry_register): every failure is one [registry] line and the job continues; the harvester's static_padding
# adapter remains the backstop lineage for a level that ran unregistered.
F22_REGISTRY_EVALS="${FAMILIES//,/ }"; export F22_REGISTRY_EVALS
SISTEER_REGISTRY_PADDING_JSON="$(PYTHONPATH="${REPO_DIR:-$PWD}/src${PYTHONPATH:+:$PYTHONPATH}" "$MPY" -m sisteer.registry.jobplan padding-block \
    --config "$CFG" --filler "$FILLER" --window "$WINDOW" 2>"$SERVE_DIR/padding_block.err")" \
  || { f22_log "[registry] padding-block helper failed for $CFG (level p${PAD_LIST[0]}): $(tr -d '\n' < "$SERVE_DIR/padding_block.err" | head -c 400) — this level registers nothing"; SISTEER_REGISTRY_PADDING_JSON=""; }
if [ -n "$SISTEER_REGISTRY_PADDING_JSON" ]; then
  export SISTEER_REGISTRY_PADDING_JSON
  if [ -z "${F22_REGISTRY_VIEW:-}" ]; then
    F22_REGISTRY_VIEW=main
    "$MPY" scripts/sdfbase/full22/roster_cell_count.py "$KEY" u0 --roster "$F22_ROSTER" >/dev/null 2>&1 && F22_REGISTRY_VIEW=u0
  fi
  f22_log "[registry] registering level p${PAD_LIST[0]} for evals [$F22_REGISTRY_EVALS] under view $F22_REGISTRY_VIEW (bank view stays $VIEW; padding block $(printf '%s' "$SISTEER_REGISTRY_PADDING_JSON" | head -c 300))"
  PAD_BANK_VIEW="$VIEW"; VIEW="$F22_REGISTRY_VIEW"
  export F22_REGISTRY_KEY_SEGMENTS="${F22_REGISTRY_KEY_SEGMENTS:-p${PAD_LIST[0]}}"   # `sdfbase_full22-padstat-<key>[_<view>]-p<level>`
  f22_registry_register || true    # in THIS shell: it exports SISTEER_REGISTRY_RUN_IDENTITY, which every store this job opens routes through
  VIEW="$PAD_BANK_VIEW"
  if [ -n "${SISTEER_REGISTRY_RUN_IDENTITY:-}" ]; then f22_log "[registry] runs of this job: $SISTEER_REGISTRY_RUN_IDENTITY"
  else f22_log "[registry] nothing registered for level p${PAD_LIST[0]} (no spool on this cluster, or the registration refused — see the [registry] lines above); the stores still file, unregistered"; fi
else
  f22_log "[registry] no padding block for level p${PAD_LIST[0]}: registration SKIPPED (never on the unpadded instrument)"
fi
# --- registry: end ---
# THE CELLS: the config's explicit enumeration (every level's config lists the same three; checked below)
mapfile -t ALL_CELLS < <("$MPY" - "$CFG" <<'PY'
import sys, yaml; print("\n".join(yaml.safe_load(open(sys.argv[1]))["explicit_cells"]))
PY
)
[ "${#ALL_CELLS[@]}" -ge 2 ] || f22_fatal "$CFG lists only ${ALL_CELLS[*]} — nothing to contrast"
for pad in "${PAD_LIST[@]}"; do
  "$MPY" - "$CFG" "${CFG_BY_PAD[$pad]}" <<'PY' || f22_fatal "level $pad's config ${CFG_BY_PAD[$pad]} does not serve the standup config's cells / model — regenerate (make_padding_configs.py)"
import sys, yaml
a, b = (yaml.safe_load(open(p)) for p in sys.argv[1:3])
for k in ("explicit_cells", "model", "model_revision", "attacks"):
    av, bv = a[k], b[k]
    if k == "attacks":
        av = {kk: v for kk, v in av.items() if kk != "static_padding"}; bv = {kk: v for kk, v in bv.items() if kk != "static_padding"}
    if av != bv:
        sys.exit(f"{k} differs between {sys.argv[1]} and {sys.argv[2]}")
PY
done
# THE BANK VIEW this standup reads: a SCRATCH PROMPTED-ONLY view built under this job's serve dir from a roster that
# maps exactly the config's prompted cells to null (the llama8b/fullconst shape: zero staged entries, VIEW_MANIFEST.json
# with no entries) — never the wave's `main` view (f22_bank_view holds the config's cell count to the roster's, and
# f22_bank_view_check would couple this prompted-only job to the health of nine unrelated objects). A config that lists
# an object cell is refused here by name: a padded STEERED leg reads the wave's view through the campaign's helpers.
pad_bank_view(){   # -> TT_TWO_SI_DEFENSES_DIR (exported)
  local roster="$SERVE_DIR/bank/${TAG}_roster.yaml" bank="$SERVE_DIR/bank/$KEY"
  mkdir -p "$SERVE_DIR/bank"
  "$MPY" - "$CFG" "$KEY" "$VIEW" "$roster" <<'PY' || f22_fatal "cannot write the scratch prompted-only roster for $KEY/$VIEW"
import sys, yaml
cfg_p, key, view, out = sys.argv[1:5]
cells = [c for c in yaml.safe_load(open(cfg_p))["explicit_cells"] if c != "none"]
bad = [c for c in cells if not c.startswith("prompted__")]
if bad:
    sys.exit(f"{cfg_p} lists object cells {bad}: this lane's scratch view serves prompted baselines only — a padded steered "
             "leg needs the wave's bank view (common.sh f22_bank_view), not this job")
yaml.safe_dump({"wave": "full22", "bank_root": "exp/objects/sdfbase/full22",
                "targets": {key: {"views": {view: {"prompted_only": True, "cells": {c: None for c in cells}}}}}},
               open(out, "w"), sort_keys=False)
PY
  "$MPY" scripts/sdfbase/full22/make_bank_view.py --target "$KEY" --view "$VIEW" --roster "$roster" --bank "$bank" \
    || f22_fatal "scratch prompted-only view for $KEY/$VIEW refused to build"
  "$MPY" scripts/sdfbase/full22/make_bank_view.py --check --target "$KEY" --view "$VIEW" --roster "$roster" --bank "$bank" \
    || f22_fatal "scratch prompted-only view for $KEY/$VIEW does not verify"
  export TT_TWO_SI_DEFENSES_DIR="$bank/cells_view/$VIEW"
  f22_log "bank view OK (scratch, prompted-only): $TT_TWO_SI_DEFENSES_DIR (roster $roster; cells ${ALL_CELLS[*]})"
}
pad_bank_view
# THE PREFLIGHT (header): filler sha256 + size + loader, model snapshot, the harness's tokens + cap + reserve rule, and
# the RENDERED need per cell at this window — REFUSED here, before the standup, naming the smallest window that fits
PAD_CFG_ARGS=(); for pad in "${PAD_LIST[@]}"; do PAD_CFG_ARGS+=(--config "${CFG_BY_PAD[$pad]}"); done
"$MPY" "$PAD_DIR/padding_preflight.py" --window "$WINDOW" --model-key "$KEY" "${PAD_CFG_ARGS[@]}" --filler "$FILLER" --families "$FAMILIES" \
    ${GPU_MEM_UTIL:+--gpu-mem-util "$GPU_MEM_UTIL"} --out "$SERVE_DIR/padding_preflight.json" \
  || f22_fatal "padding preflight REFUSED at window $WINDOW (see $SERVE_DIR/padding_preflight.json and the REFUSED lines above) — no GPU work started"
f22_log "padding preflight PASS: $(tail -c 400 "$SERVE_DIR/padding_preflight.json" | tr -d '\n' | head -c 200)"
if [ -n "$CELLS_ARG" ]; then IFS=',' read -ra CELLS <<< "$CELLS_ARG"; else CELLS=("${ALL_CELLS[@]}"); fi
for c in "${CELLS[@]}"; do case " ${ALL_CELLS[*]} " in *" $c "*) ;; *) f22_fatal "cell $c is not in $CFG's explicit_cells (${ALL_CELLS[*]}) — a job serves only cells the config names";; esac; done
NCELLS=${#CELLS[@]}
# the LOAD CHECK sends the LONGEST prompted SI as the system prefix too (target_static.sbatch's prompted-only rule): the
# probe then hits that cell's route at the run's cap, S concurrent — the window is sized for it, so it must pass
F22_LOAD_CHECK_SYSTEM_SI=$("$MPY" - "$CFG" <<'PY'
import sys, yaml
from sisteer.eval import defense_cell as tt
from sisteer.eval.jailbreak import resolve_instruction
cells = [c for c in yaml.safe_load(open(sys.argv[1]))["explicit_cells"] if c.startswith("prompted__")]
sis = [tt.si_of(c) for c in cells]
print(max(sis, key=lambda s: len(resolve_instruction(s).text)) if sis else "")
PY
) || f22_fatal "cannot resolve the config's prompted SIs for the load check"
[ -z "$F22_LOAD_CHECK_SYSTEM_SI" ] || export F22_LOAD_CHECK_SYSTEM_SI
# the sampler the ENGINE resolved goes on the record (the sampled lane's rule): one INFO line per request, no prompt
# text; built ON TOP of the inherited launch-time hook through common.sh's helper, never assigned
f22_standup_extra_reset --enable-log-requests
TARGET_PID=""; STANDUP_STOP=""; declare -a DRIVER_PIDS=(); declare -A DRIVER_TAG=(); RC=0; TOTAL_FAIL=0
cleanup(){
  # $? FIRST: the script's exit status (an f22_fatal exits 1 with RC still 0 — the RESULT line must say so, never rc=0)
  local ec=$?; [ "$ec" != 0 ] && [ "${RC:-0}" = 0 ] && RC=$ec
  # DISARM the signal traps before signalling children: a TERM arriving mid-cleanup would re-enter this handler
  trap '' SIGTERM SIGINT SIGQUIT
  [ -n "${STANDUP_STOP:-}" ] && touch "$STANDUP_STOP" 2>/dev/null || true
  for p in "${DRIVER_PIDS[@]:-}"; do [ -n "$p" ] && kill -TERM "$p" 2>/dev/null || true; done
  [ -n "${TARGET_PID:-}" ] && { for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; }
  f22_result "padstat $TAG window=$WINDOW pads=$PADS rc=${RC} exit=${ec} failed_drivers=${TOTAL_FAIL}"
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
TARGET_GPU="$CUDA_VISIBLE_DEVICES"
# ---------------------------------------------------------------- the ONE standup: families_served:short at <window> x the derived S
f22_standup attacks "$SERVE_DIR/static" "" families_served:short "$WINDOW" || { RC=1; exit 1; }
# native-serving-default: the cells this standup SERVES (a prompted baseline is always served; a cells= positional naming an unserved cell is fatal)
f22_served_cells_filter CELLS "$CELLS_ARG"; NCELLS=${#CELLS[@]}
# no steer-is-live positive control here: it probes a steered cell against its scale-0 twin, and this job serves none
# (target_static.sbatch skips `none` / prompted__* by the same rule); the SI prefixes were verified by f22_standup's record
[ "$TARGET_L" = "$WINDOW" ] || { f22_log "REFUSING TRAFFIC: the standup resolved max_model_len $TARGET_L but this job was sized (preflight) for $WINDOW"; RC=1; exit 1; }
# THE GEOMETRY PIN (header): these arms must sit on the geometry a STEERED standup at this window resolves, so the
# attested record must show (1) the native engine, (2) the zero adapter none__lora0 among the attested adapters (it is
# what makes the kernel ceiling bind on a prompted-only cell set), (3) an attested S equal to the kernel-ceiling S the
# geometry module derives at this window from the model's budget and pinned widest module — the SAME two steps a
# steered standup records — and (4) that adjustment recorded in endpoint.json's chain. Read from the AUTHORITATIVE
# record (endpoint.json), never from the env; an adapter-free standup (S = identity) is refused here by name.
"$MPY" - "$SERVE_DIR/static/endpoint.json" "$KEY" "$WINDOW" "$TARGET_S" <<'PY' || { f22_log "REFUSING TRAFFIC: the standup's geometry is not the steered arms' geometry (see the GEOMETRY PIN lines above and the header)"; RC=1; exit 1; }
import json, sys
from pathlib import Path
import sisteer
from sisteer.capeval.petri.defense_cells import NOOP_LORA_CELL_ID
from sisteer.capeval.serving import geometry as g
sys.path.insert(0, str(Path(sisteer.__file__).resolve().parents[2] / "scripts/sdfbase/full22/static_padding"))
import padding_preflight as pf   # the preflight's own selection rule: ONE source for the selected module set
ep = json.load(open(sys.argv[1])); key, L, S = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
# THE BUDGET RESOLVER (the coordinator's ruling 2026-09-29): geometry.budget_for / rows_for — the ONE lookup over the sweep's keys AND the
# paper-companion ONBOARDED ones (padding_preflight.geometry_expectation applies the same rule before the standup); never BUDGET_BY_MODEL
# indexed directly. An unknown key, or a model with no PINNED widest module for the kernel step, refuses BY NAME (never a KeyError).
# THE SELECTED MODULE SET (the closing review's finding 1; D69): the geometry overlay is certified PER MODULE SET, and a native standup's
# ceiling is the D of the class it registered — so this pin resolves the record of the HPO winner's set, read from the ONE
# source every padded job has at hand, reproduce/models.yaml's promoted `lora_architecture` (padding_preflight.selected_module_set;
# the preflight before the standup read the same), never a model-wide maximum; a set the overlay does not certify refuses by name
# THE UNSTEERED launcher (2026-09-30): this standup registers only the zero adapter none__lora0 (the o_proj class — native_standup's
# zero-class rule), so NO promoted pick is needed and none drives the size: the resolver's unsteered default (geometry.UNSTEERED_MODULE_SET,
# the site record's row for that class) sizes it — a record that lacks it refuses by name, never the provisional row beside a record; a
# promoted pick, when one exists, is reported only (the steered launcher, target_static_padded_obj.sbatch, follows the pick)
SET, sel = pf.unsteered_module_set(key)
try:
    rows = g.rows_for(key, SET); rb = g.budget_for(key, SET)
except (g.UnknownModelKey, g.SiteGeometryError) as e:
    print(f"[geometry-pin] REFUSED: {e}", flush=True); sys.exit(1)
SET = rows.site.module_set if rows.site is not None else SET
print(f"[geometry-pin] unsteered standup: module set {SET!r} ({rows.site_module_set_source or 'built-in tables'}); promoted pick {sel.get('promoted_pick')} (reported, not the size)", flush=True)
# a PROVISIONAL (or default) budget is REFUSED for steered padding (code review round 2, finding 2; D58): the S this pin holds
# the arms to must be one the engine was MEASURED to come up at on this model — the sweep's row, or the FD 24 serving probe's fold
# in the site geometry overlay (geometry.site_geometry_dir), never a conservative guess
if not rb.measured:
    print(f"[geometry-pin] REFUSED: {key!r}: batched-token budget source {rb.source!r} ({rb.batched_token_budget:,}) is not a measurement; "
          f"run the serving probe so its verdict folds the measured row into the site geometry overlay ({g._site_overlay_where(key)})", flush=True); sys.exit(1)
if rb.per_model:
    identity = int(rb.entry.seqs_at(L))
else:
    identity = int(g.identity_seqs(L, budget=rb.batched_token_budget))
    identity = min(identity, g.HYBRID_SEQS_CAP) if rows.hybrid else identity
try:
    adj = g.kernel_ceiling_adjustment(key, L, identity, module_set=SET)
except (g.UnknownModelKey, g.SiteGeometryError) as e:
    print(f"[geometry-pin] REFUSED: {key!r}: {e} — the kernel-ceiling step of the steered arms' geometry cannot be derived; pin the width from "
          "the serving probe's fold in the site geometry overlay (geometry.lora_max_module_dim_for; the built-in geometry.LORA_MAX_MODULE_DIM_BY_MODEL for the sweep's models) before this model's padded standup runs", flush=True); sys.exit(1)
expect = int(adj["to_S"]) if adj else identity
names = sorted((ep.get("id_name_map") or {}).values())
chain = ((ep.get("geometry") or {}).get("adjustments") or [])
problems = []
if ep.get("engine") != "lora_native":
    problems.append(f"engine {ep.get('engine')!r} != lora_native")
if NOOP_LORA_CELL_ID not in names:
    problems.append(f"the zero adapter {NOOP_LORA_CELL_ID} is not among the attested adapters {names}")
if S != expect:
    problems.append(f"attested max_num_seqs {S} != {expect}, the S a steered standup at L={L} resolves (identity {identity} at budget "
                    f"{rb.batched_token_budget:,} ({rb.source}); kernel ceiling {adj['ceiling']:,} // {L} = {adj['to_S']})" if adj
                    else f"attested max_num_seqs {S} != the identity S {expect} at L={L} (the kernel ceiling does not bind here)")
if adj is not None and not any(a.get("kind") == adj["kind"] and int(a.get("to_S", -1)) == S for a in chain):
    problems.append(f"endpoint.json geometry.adjustments {chain} carries no {adj['kind']} step ending at S={S}")
# the standup's attested D (the widest module it registered) must be the SELECTED set's certified D: a standup of another
# class attests another ceiling, and this pin would otherwise hold prompted arms to a geometry no steered standup shares (D69)
for a in chain:
    if a.get("kind") == g.ADJUSTMENT_KERNEL and adj is not None and int(a.get("D", -1)) != int(adj["D"]):
        problems.append(f"the attested kernel step's widest module D={a.get('D')} != the pinned {adj['D']} (the selected module set {SET!r} "
                        f"certifies D={adj['D']}) — the standup registered adapters of another class than the HPO pick's")
if problems:
    print("[geometry-pin] REFUSED: " + "; ".join(problems), flush=True); sys.exit(1)
print(f"[geometry-pin] OK: engine lora_native, adapters {names}, S {S} x L {L} = identity {identity} -> kernel ceiling "
      f"{adj['to_S'] if adj else 'n/a'} at D {adj['D'] if adj else 'n/a'} for module set {SET!r} (chain {[a.get('kind') for a in chain]}) "
      "— the geometry a steered standup of the selected design at this window resolves", flush=True)
PY
# ~one in-flight request per driver: fill the attested width, never more (target_static.sbatch's rule and cap)
DRIVER_CAP=96
MAX_DRIVERS=$(( TARGET_S < DRIVER_CAP ? TARGET_S : DRIVER_CAP ))
f22_log "phase PADDED STATIC: levels=${PAD_LIST[*]} families=${FAMS[*]} cells=$NCELLS nshards=$NSHARDS server_width=$TARGET_S window=$TARGET_L max_drivers=$MAX_DRIVERS (cap $DRIVER_CAP)"
# --- drivers: begin (rendered by tests/test_static_padding_wave.py under stubs; the loop body is the argv contract) ---
# LEVELS RUN SEQUENTIALLY, in the order given: each level's drivers fill the width, then the next level starts, so the
# log carries one wall time per level (the cost model's measurement) and a partial job leaves whole levels behind.
# A padded level's drivers carry the FILLER PATH ONLY (the level itself is the config's); the p0 control carries no
# padding flag at all (a sub-flag while the knob is off is refused by the driver, by design).
for pad in "${PAD_LIST[@]}"; do
  f22_stop_requested && { f22_log "wave STOP present — not launching level $pad"; break; }
  PAD_FLAGS=(); [ "$pad" -gt 0 ] && PAD_FLAGS=(--static-padding-filler "$FILLER")
  LEVEL_T0=$SECONDS
  f22_log "level p$pad: config ${CFG_BY_PAD[$pad]} padding_flags=[${PAD_FLAGS[*]:-none}]"
  for fam in "${FAMS[@]}"; do for cell in "${CELLS[@]}"; do for s in $(seq 0 $((NSHARDS-1))); do
    f22_stop_requested && { f22_log "wave STOP present — not launching more drivers"; break 3; }
    f22_queue_launch "p${pad}__${fam}__${cell}__s$(printf '%02d' "$s")" "$MPY" scripts/jailbreak_run.py generate \
      --config "${CFG_BY_PAD[$pad]}" --attack "$fam" --defense "$cell" --confirmatory --shard "$s" --nshards "$NSHARDS" \
      --private-root "$PRIVATE_ROOT" --target-engine served --target-proxy-url "$PROXY_URL" \
      --target-routing-table "$TABLE" --target-max-model-len "$TARGET_L" ${PAD_FLAGS[@]+"${PAD_FLAGS[@]}"}
    sleep 1
  done; done; done
  f22_log "level p$pad: all ${#DRIVER_PIDS[@]} drivers launched (queued at <= $MAX_DRIVERS in flight); waiting"
  f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
  f22_log "level p$pad done: $NFAIL failed drivers; wall $(( SECONDS - LEVEL_T0 )) s"
done
# --- drivers: end ---
f22_standup_stop
RC=$(( TOTAL_FAIL > 0 ? 2 : 0 ))
f22_log "all levels done: $TOTAL_FAIL failed drivers; stores under $PRIVATE_ROOT/pad_static_${KEY}_p<pad>/<family>/<cell>/shardNN"
exit $RC
