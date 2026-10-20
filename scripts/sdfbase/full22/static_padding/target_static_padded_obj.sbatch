#!/bin/bash
#SBATCH --job-name=padobj
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# full22 PADDED STATIC leg, the STEERED family, on CLUSTER A (docs/static_padding.md "The steered ten-cell wave"): ONE served
# TARGET (1 GPU, the native engine) + ONE static one-shot family (default grok420) at the 16,384 completion cap — the static
# ladder's CEILING (target_static_padded.sbatch's header carries the reasoning; padding_preflight.check_cap refuses any
# other) — over the 259-goal confirmatory pool, for the TEN cells of ONE padding level (0 | 8192 | 16384 | 32768 | 65536; the
# two middle-of-the-range rungs ADDED 2026-09-24, run SHARDED across jobs — below): the same-wave bare anchor `none`,
# prompted__safety_refusal, prompted__si_constitution, the five fitted s1 objects the production roster's `s1` view serves
# (constv2-mix-full__psr_ce_all, constv2-mix-full__lora_r4_ce, constv2ctl2-full__psr_ce_all, constv2ctl2-full__lora_r4_ce,
# constv2-mix-full__lora_r4_ce_minus_ctl), the all-positions Arditi rung c = 0.3 under its harness id `arditi_allpos`, and
# the paired write constv2-mix-full__psr_ce_all_minus_ctl (full dose, seed 1; never the half-dose view, never a
# *_distilled object). Decoded at the standing vendor-preset default (the driver injects it for a served static run whose
# config carries no sampled_decode block; never --greedy). It EXTENDS target_static_padded.sbatch (the prompted-only wave,
# which ran on cluster B and whose stores are filed) and does not touch it: that script's scratch view REFUSES any object
# cell by name, so a steered leg is a DIFFERENT VIEW PATH, not a parameter flip — the two scripts share the generator, the
# preflight, the lane root and the geometry pin, and differ in the view, the roster, the cluster paths and the job shape.
#
# THE VIEW (make_padding_configs.py's module docstring; the roster it generates,
# configs/sdfbase/static_padding/sdfbase_full22_<key>.padobj_roster.yaml): the wave bank's `padobj` view —
# $SHARED/exp/objects/sdfbase/full22/<key>/cells_view/padobj — built and verified by the CAMPAIGN's own helper (common.sh
# f22_bank_view: --check first, build only under the per-(target, view) lock, the short-view guard against the roster's
# count, the config's cell count against the roster's), never a scratch view. Its five s1 entries are the SAME staged ids the
# production `s1` view maps (string form, pinned in configs/sdfbase_full22_pins.json); `arditi_allpos` and the paired write
# are named through the roster's CROSS-BANK PIN FORM against their owning lanes' pin files (configs/arditi_allpos_pins.json
# qwen35_4b.ar_c0p3__arditi_allpos; configs/reseed_paired_pins.json qwen35_4b.constv2-mix-full__psr_ce_all_minus_ctl), so
# each entry is a symlink to the owning lane's staged copy — exp/objects/arditi_allpos/<key>/ladder/cells/ar_c0p3__arditi_allpos,
# exp/objects/paired_write/<key>/cells/paired_s1__constv2-mix-full__psr_ce_all_minus_ctl — whose bytes make_bank_view.py
# re-hashes to the pin (ba2212f0…, f98f0349…), and NO bytes are copied into the wave bank (the form the bank's pairedcert /
# paired_d1_s1 / ar_c0p3 views already use). The roster is the LANE'S: F22_ROSTER is set to it below, before f22_init, and
# f22_bank_view refuses it unless the production roster names NO `padobj` view (a non-production roster may only name views
# of its own — the view dir is shared state keyed by name). The lane never writes s1, s1p, pairedcert or any peer view.
#
# ONE JOB PER PADDING LEVEL (this wave's shape since 2026-09-23): each submission runs ONE level (the <pad>
# positional: 0, 16384 or 65536 — and, since 2026-09-24, 8192 or 32768, the ADDED RUNGS below) for ONE family on its own GPU
# with its own standup, --time 4 h (the rule below, 2026-09-24). The pad-vs-pad contrast
# therefore straddles three engine INSTANCES — the trade the prompted-only wave's header argued against (family, not level,
# is the free split axis). What makes it sound here is the GEOMETRY PIN below: every level stands up at the SAME window
# from the SAME config body (the standup reads cells + model + cap, which every level's config shares), registers the SAME
# adapters (the ten cells' exports + the zero adapter), and must attest the SAME S — the kernel-ceiling S the geometry
# module derives — so the three instances carry one engine identity (weights, adapters, S x L, the CUDA-graph capture
# surface) and the pad stays the only thing that varies between the three stores; the pin refuses traffic otherwise.
# COST per job, AS FIRST ESTIMATED (2026-09-23): 10 cells x 518 grok420 variants = 5,180 requests at width 5; p0 decode-only
# (~0.5 h), p16384 ~1 h, p65536 ~1.5-2 h from the prefill FLOP model — and AS MEASURED on the filed stores (2026-09-24, read
# off every row's completion_tokens / prompt_tokens and the three rounds' job logs): the estimate was wrong by 3-13x because
# the wall is set by DECODE at width 5, not by the pad prefill. p0 generated 10.66 M completion tokens (mean 2,058 per request,
# 380 at-cap rows) and took ~6.5 h of driver wall over two submissions (2 h 48 to the 3 h wall, then 3 h 45 of remainder);
# p16384 6.32 M completion + 89.8 M prompt tokens, ~4.7 h (2 h 48 + 1 h 52); p65536 7.02 M completion + 344 M prompt tokens,
# ~7.4 h over three submissions (2 h 48 + 3 h 47 to the 4 h wall + 0 h 47). The model that fits all three: aggregate decode
# ~450 tok/s at width 5 (~90 tok/s per in-flight request, one request per driver) plus a per-request prefill of ~0.55 s at a
# 17k window and ~2.1 s at 67k (the hybrid's 8 attention layers are quadratic; the 24 deltanet layers linear). The completion
# volume varies with the level (padding changes what the model writes), so a new rung is bounded by its neighbours.
# run_static resumes per goal, so a TIMEOUT costs one resubmit (~10 min of standup + control), never the generations filed.
#
# SHARDING ACROSS JOBS (2026-09-24; the added rungs in less wall time). One job per level at ~6-7 h of driver
# wall does not fit any walltime this cluster backfills, so a level's GOAL SET is split across SEVERAL JOBS: the `shards`
# positional `<lo>-<hi>/N` says "this job owns shards lo..hi (inclusive, 0-based) of N" — every driver of the job runs
# --shard s --nshards N for its (family, cell) and s in lo..hi, exactly the argv a single job at N would have issued for
# those s, so the store comes out as <run>/<family>/<cell>/shard00..shard<N-1> once every range has run: the layout ONE job
# would write, read unchanged by the grader (grade_padded_static.py --nshards N: shard00..shard<N-1> exactly) and the
# per-item degeneracy writer (degeneracy_items.py: every shard*, a goal in one shard). A bare `N` is the single-job shape
# (all N shards; the filed three rungs ran `3`). WHY IT IS SAFE, AND WHERE THE SEAM IS: every job stands up its own engine, but
# ALL TEN ARMS within a job share THAT job's standup, and every arm gets the SAME shard split (the range is per job, the cell
# loop inside it), so an arm-vs-arm comparison at a level is within-standup and balanced by construction — the same goals,
# the same engine instance, for every cell. What crosses standups is goals WITHIN a cell (goal g of cell c ran on the job
# owning shard g mod N). That is a milder seam than a cross-arm one — the GEOMETRY PIN holds every standup to one engine
# identity (weights, adapters, S x L, the attested D), the decode is the vendor preset on every row, and the level-vs-level
# contrast already straddled instances under the 2026-09-23 shape — and it is the price of the wall-clock. REFUSALS,
# before any GPU minute: a malformed spec, lo > hi, hi >= N, N > 99 (the store names shards with two digits) — and OVERLAP:
# a job takes a LEASE on each (family, shard) it owns ($WAVE_ROOT/static_padding/shard_leases/<run>/<family>__sNN.json:
# job id, N, range, state) and refuses (1) a shard leased under another N (two layouts of one store), (2) a shard whose
# lease is held by a job squeue/sacct still report live (an overlapping range), (3) a holder neither tool can classify
# (refusing beats a double write; delete the lease by hand if the job is dead), (4) a shard directory on disk numbered
# >= N (laid out under a larger N), and (5) a shard directory in its range with no lease at all (a layout it cannot vouch
# for). A lease whose job has EXITED (the cleanup trap marks it, or sacct says a terminal state) is free: that is the
# resume after a TIMEOUT, and the store resumes per goal. SIZING for the added rungs (the measured model above; four jobs per
# level, the budget shape — fewer jobs that START beat more jobs that WAIT, 2026-09-24): p8192 is bounded by p0 and
# p16384 (decode 8.5-10.7 M tokens -> 5.2-6.6 h, prefill ~0.3 s x 5,180 -> 0.4 h: 5.7-7.0 h of driver wall for the level);
# p32768 by p16384 and p65536 (6.3-7.0 M -> 3.9-4.3 h, prefill ~1.05 s x 5,180 -> 1.5 h: 5.5-5.8 h). Four jobs per level
# -> 1.4-1.75 h of driver wall each + ~12 min of preflight, standup and control. N = 16 (four shards per job, ~16 goals x 2
# variants x 10 cells = ~320 requests per job in 40 drivers of ~32) rather than N = 8: one driver is one request in flight, the
# per-cell completion volume spans 9x (p0: 2.19 M tokens for the paired write vs 0.25 M for arditi_allpos), and the heaviest
# cell can be last in launch order, so with 2 shards per job the last two drivers can run ~0.85 h alone on 2 of 5 rows;
# 4 shards per job halves that tail (~0.4 h) for ~6 min of extra driver startup. Worst plausible job: ~2.5 h.
# --time=04:00:00 on EVERY job (2026-09-24: 4 h backfills acceptably here, a timeout costs a whole re-queue
# behind ~120 higher-priority jobs, a job that finishes at 40 min under a 4 h limit costs nothing extra). Two levels x four
# jobs = 8 GPUs of the 16-GPU `high` budget.
#
# SEED REPLICATES (2026-09-24: seeds 2 and 3 of the six FITTED objects, on the IDENTICAL instrument). The eighth
# positional, `family`, selects the CONFIG FAMILY the job serves: `obj` (the default: the seed-1 ten-cell family above, every
# path and name as before) or `obj_s2` / `obj_s3` (make_padding_configs.OBJ_SEED_FAMILIES) — the same generator's seed-N family:
# config configs/sdfbase/static_padding/sdfbase_full22_<key>.padstatic_obj_s<N>_p<pad>.yaml, manifest petri_full22_<key>.padstatic_obj_s<N>.yaml,
# roster sdfbase_full22_<key>.padobj_s<N>_roster.yaml, view `padobj_s<N>` (a NEW bank view per seed, built by f22_bank_view over
# the lane roster exactly as `padobj` is; the seed-1 `padobj` view and every peer view are never touched), store
# pad_static_<key>_obj_s<N>_p<pad> (its own namespace and its own lease directory). SEVEN cells per seed — the same-wave bare
# anchor `none`, then the four base64s<N> fits, the difference LoRA derived_base64s<N>, the FULL-dose paired write paired_eb64s<N>
# (the wave bank's own staged copies, read from the production roster's s<N> / s<N>p views and pinned in
# configs/sdfbase_full22_pins.json; positions `all` in every sidecar, so nothing is skipped by name) — NO prompted baseline, NO
# Arditi rung (neither has a training seed). WHY `none` IS RUN PER SEED although it has no training seed either (the seed-1
# `none` cannot be reused — the existing code does not allow it): the native standup
# derives its zero-adapter leg none__lora0 FROM the `none` cell (defense_cells.enumerate_cells: the control of the base cell),
# exports and registers the zero adapter from that leg, and sizes the LoRA kernel index ceiling over the REGISTERED adapters
# (native_standup.bring_up_native: the zero-leg check, then export_or_reuse_zero, then lora_kernel_ceiling_geometry over
# all_blocks). A six-cell set without `none` was submitted first (2026-09-24): all 29 standups refused at 1-5 min with
# "[mt-standup] refused (exit 2): enumerate_cells derived no none__lora0 leg", and the geometry line each printed BEFORE the
# refusal was the IDENTITY width — "max_num_seqs=7 (model_budget); identity; 688,128 of the 741,888 batched-token budget" —
# because the lowering to S = 5 runs after the adapters are registered and never ran. So `none` is what puts the zero adapter on
# the engine and, through the ceiling, the seed-1 wave's width; it is also each store's within-standup anchor for the grader.
# THE WIDTH IS PINNED BY A LITERAL, GEOMETRY_PIN_S = 5 (the seed-1 wave's attested S at 98,304: identity 741,888 // 98,304 = 7,
# kernel ceiling 524,287 // 98,304 = 5): the lane preflight takes it as --expect-served-s (the derivation must land on it, through
# the kernel step, before any GPU minute) and the post-standup GEOMETRY PIN below takes it as its fifth argument and refuses
# traffic unless the ATTESTED max_num_seqs, the module's derivation at the served window AND a kernel-ceiling step in the attested
# chain all equal it — so a standup that came up at the identity 7 can never reach a store, whatever the geometry table or the
# derivation says that day. THE DECODE SEED BASE IS HELD AT 20260909 — the driver argv below passes no seed flag, exactly as the
# seed-1 wave's did, so every seed decodes under the preset's campaign base; it is deliberately NOT the training seed (the seedrep
# contract's spelling), which would confound object seed with decode seed. Everything else is the seed-1 wave's: window 98,304,
# cap 16,384, vendor preset, trial 0, five rungs, grok420, the positive control on the first object cell, the leases, the sharded
# shape. SIZING (measured on the filed seed-1 stores 2026-09-24, seven-cell completion tokens per rung: p0 9.93 M, p8192 5.32 M,
# p16384 5.21 M, p32768 5.91 M, p65536 5.65 M — the seven cells are 80-93 % of the ten-cell volume, so dropping prompted / Arditi
# saves little; p0 stays the slowest rung): driver wall per rung at ~450 tok/s decode + per-request prefill (0.15 / 0.3 / 0.55 /
# 1.05 / 2.1 s x 3,626 requests) ~= 6.3 / 3.6 / 3.8 / 4.7 / 5.6 h. THREE jobs per (seed, rung), N = 12 (four shards per job, the
# filed added rungs' tail rule): the worst job (p0) ~2.1 h of driver wall + ~12 min of standup, every other <= 1.9 h — all inside
# --time=04:00:00 (two jobs per rung would put p0 at ~3.3 h, too close to the wall for the per-cell tail). 2 seeds x 5 rungs x 3
# = 30 jobs; one seed per cluster (15 jobs each, inside each cluster's 16-GPU `high` budget): cluster A as written, cluster B
# with its own QoS and log directory (its site block's CLUSTER_QOS, CLUSTER_LOG_DIR) on the sbatch line (both hold every object, the filler and the
# snapshot; checked 2026-09-24).
#
# THE MODULE-DIM PREMISE THE GEOMETRY PIN ENFORCES. The prompted-only wave pinned its arms to the geometry a STEERED standup
# resolves by assuming the widest adapted module is D = 4,096 (geometry.LORA_MAX_MODULE_DIM_BY_MODEL["qwen35_4b"]). This wave
# IS the steered standup, so the premise is now load-bearing: native_standup.lora_kernel_ceiling_geometry reads EVERY
# registered adapter's real module dims off its safetensors header (serving_attestation.adapter_module_dims; the residual
# payload keys are skipped — the rank-1 zero o_proj CARRIER sizes the kernel) and lowers S to (2^31-1) // max_dim // L.
# Verified read-only on cluster A 2026-09-23 over every export under $SHARED/exp/multilora_adapters/qwen35_4b: the lora_r4
# (r 4) and lora_subtraction (r 8) o_proj adapters, the psr / psr_paired_write / static_r / vector_all residual carriers
# (r 1) and the zero adapter all have max factor dim 4,096 = o_proj's in_features (16 heads x 256 head_dim; config.json), so
# ten cells derive the SAME S as one: identity 741,888 // 98,304 = 7, ceiling 524,287 // 98,304 = 5. The pin below checks it
# at RUN time against the attested record (endpoint.json): engine lora_native, the zero adapter among the adapters, attested
# S == the derived kernel-ceiling S, the chain's kernel step ending at that S — AND its recorded D equal to the pinned width,
# so an export wider than 4,096 (a future kind, a wider carrier) refuses by name instead of quietly serving at a smaller S
# that no other level or wave would match.
#
# THE PER-ITEM DEGENERACY WRITER IS NOT OPTIONAL FOR THIS WAVE (the post-step: degeneracy_items.py, run beside the grades of
# each (level, family) with --cells naming these ten). The prompted-only wave could read its at-cap share as a footnote;
# this wave cannot, for three reasons that compound. (1) The paired-write family degenerates DIFFERENTLY BY MODEL — on qwen
# the undistilled functional residual runs to loops and collapse at rates its LoRA twin does not, so a "steering decayed
# under padding" reading on one cell can be a "generation degenerated" reading on the next. (2) Its degeneration FLIPS SIGN
# BETWEEN RUBRICS: a looping or truncated completion reads as a REFUSAL under one grader (no harmful content) and as
# COMPLIANCE under another (the content is there before the loop), so a per-cell aggregate moves in opposite directions
# depending on which instrument reads it, and only a per-item record lets a reader see that the SAME items drive both.
# (3) Padded context is exactly where looping and truncation RISE — a 65k prefix at a 16,384 cap is the regime the cap
# ladder's late-onset-loop rule was written for — so the confound is largest on the arm of the contrast that matters. The
# writer's one line per goal (loop ratios whole + tail, zlib collapse, at-cap, truncation-only vs degenerate, per-row
# records) is what makes "the steering decayed" DISTINGUISHABLE from "the generations degenerated"; a wave graded without it
# cannot say which happened. The sbatch produces the stores; the writer and the grader are post-steps on the grading host
# (grade_padded_family.sh, degeneracy_items.py — note both pin the prompted-only family's names today and need the `_obj`
# family spelling before they read these stores; refusing by name until then).
#
# THE STORE IDENTITY: pad_static_<key>_obj_p<pad> (make_padding_configs.private_run(key, pad, "obj")) — its OWN namespace,
# so nothing collides with the prompted-only wave's filed pad_static_<key>_p<pad> stores; one padding contract per store, the
# pad level travels with the CONFIG (the drivers pass no --static-padding-tokens), the filler path rides the argv
# (sha-verified by the preflight first). Stores land under $WAVE_ROOT/static_padding/transcripts_results/<private_run>/
# <family>/<cell>/shardNN — the lane root beside the campaign's, never inside it. REGISTERED in the results registry PER LEVEL
# (the block "THE REGISTRY IDENTITY OF THIS LEVEL" below): the registry has a
# padding axis since migration 0012, so this level's runs file on the level's OWN instrument (the canonical seeded static
# instrument for (family, model, path) + padding_tokens + the filler contract's hash + the ceiling cap + the window; the p0
# control carries padding_tokens 0 with the siblings' contract). The job exports F22_REGISTRY_EVALS (its one family)
# and SISTEER_REGISTRY_PADDING_JSON (the level block) and calls common.sh f22_registry_register ONCE, before the bank view
# and the standup, under the PHASE view its objects belong to (F22_REGISTRY_VIEW, default `s<seed>` of the config family —
# s1 for `obj`, s2 / s3 for the replicates; an inherited environment variable, never --export=) read from the PRODUCTION
# roster, which is where the s<N> views live — never under the lane view `padobj[_s<N>]`, which is not a model-key
# spelling. The two CROSS-BANK cells (arditi_allpos, the paired write) are not mapped by the production s<N> view, so the
# registration skips them BY NAME (recorded in the pending file's `skipped`); the harvester's static_padding adapter remains
# their lineage. NO FIGURES (the RESULT line rides through f22_result).
#
# THE STEER-IS-LIVE POSITIVE CONTROL runs here (the prompted-only wave had no steered cell to probe): the campaign's
# f22_positive_control on the FIRST object cell after the standup, before any driver — a silently no-op'd object refuses
# before a GPU minute is spent (target_static.sbatch's rule).
#
# THE WORKLOAD IS NOT srun-WRAPPED, deliberately (the full22 lineage's shape): the standup and the drivers are children of
# this batch shell, bounded by the driver queue, reaped by cleanup() below, and --signal=B:TERM@120 delivers the walltime
# warning to THIS shell so the trap runs inside the grace window.
#
# Usage (positional only — never --export=; run from the wave checkout on cluster A; one submission per level, or one per
# shard RANGE of a level):
#   sbatch -J padobj-<key>-p<pad>[-s<lo>-<hi>] --time=04:00:00 scripts/sdfbase/full22/static_padding/target_static_padded_obj.sbatch \
#       <key> <wave_sha> <window> <pad> [families=grok420] [shards=3] [cells=all] [family=obj]
#     window    the standup's --max-model-len (98,304: the prompted-only wave's window, so every level and both waves share
#               one geometry; the preflight refuses one that cannot carry the level)
#     pad       EXACTLY ONE padding level (0 | 8192 | 16384 | 32768 | 65536); a comma list is refused — one job per level
#     families  comma subset of pasttense,codeattack,renellm,grok420 (default grok420: the ONE family of this wave)
#     shards    `N` = this job runs ALL N shards of every (family, cell) (the single-job shape; default 3, the filed rungs'),
#               or `<lo>-<hi>/N` = this job owns shards lo..hi (inclusive, 0-based) of N — the SHARDED shape: submit one job
#               per disjoint range (0-3/16, 4-7/16, 8-11/16, 12-15/16) and the store comes out as shard00..shard15 per
#               (family, cell), what one job at N = 16 would write. Out-of-range, inverted, overlapping or re-laid-out ranges
#               are REFUSED (the leases, header) — never double-written. ~S drivers stay in flight, so more shards than S
#               only shorten the tail
#     cells     comma subset of the config's explicit_cells (default: all ten; the literal `all` says the same, so a later
#               positional can be given)
#     family    the CONFIG FAMILY (header "SEED REPLICATES"): `obj` (default; the seed-1 ten-cell family) or `obj_s2` / `obj_s3`;
#               for a FRONT-DOOR key (reproduce/models.yaml) the phase view it pads (`s2`, `s1`, `d0p75`, `d0p75_relu`; the
#               generator's `--resolve-family <key> <family>` is the one source, stage 06's lib/stage.py plans these rows)
#               (the six-cell seed replicates, each with its own config, manifest, roster, view `padobj_s<N>` and store)
#   The added rungs, four jobs each (the header's sizing; every line carries --time=04:00:00; eight submissions in all —
#   type them one at a time, never a shell loop over many submissions):
#     sbatch -J padobj-qwen35_4b-p8192-s0-3   --time=04:00:00 <this script> qwen35_4b <sha> 98304 8192  grok420 0-3/16
#     ... -s4-7 4-7/16, -s8-11 8-11/16, -s12-15 12-15/16; then the same four for 32768.
#   Then grade each level with grade_padded_family.sh <pad> grok420 --config-family obj --host <login-host> --nshards 16, and
#   degeneracy_items.py --config-family obj --pads 8192,32768 (it reads every shard*).
#   The seed replicates, THREE jobs per (seed, rung) at N = 12 (header "SEED REPLICATES"; thirty submissions, one at a time;
#   seed 3 on cluster A as written, seed 2 on cluster B with that cluster's QoS and --output=$SHARED/exp/logs/%x_%j.out added):
#     sbatch -J padobj-s3-qwen35_4b-p0-s0-3  --time=04:00:00 <this script> qwen35_4b <sha> 98304 0 grok420 0-3/12  all obj_s3
#     sbatch -J padobj-s3-qwen35_4b-p0-s4-7  --time=04:00:00 <this script> qwen35_4b <sha> 98304 0 grok420 4-7/12  all obj_s3
#     sbatch -J padobj-s3-qwen35_4b-p0-s8-11 --time=04:00:00 <this script> qwen35_4b <sha> 98304 0 grok420 8-11/12 all obj_s3
#     ... the same three lines for 8192 / 16384 / 32768 / 65536, then the fifteen with obj_s2 on cluster B.
#   On cluster B (not the plan): that cluster's QoS and --output=$SHARED/exp/logs/%x_%j.out on the sbatch line; the body is
#   cluster-agnostic through common.sh — but the pin files' `verified_cluster_b` flags and the paired_write / arditi_allpos
#   banks there are that cluster's own facts to check first.
set -uo pipefail
KEY="${1:?key}"; WAVE_SHA="${2:?wave sha}"; WINDOW="${3:?window (the standup max_model_len)}"; PAD="${4:?pad (ONE level: 0 | 8192 | 16384 | 32768 | 65536)}"
FAMILIES="${5:-grok420}"; SHARDS_SPEC="${6:-3}"; CELLS_ARG="${7:-}"
CONFIG_FAMILY="${8:-obj}"                      # the CONFIG FAMILY (header "SEED REPLICATES"): obj | obj_s2 | obj_s3
[ "$CELLS_ARG" != all ] || CELLS_ARG=""        # the literal `all` = every cell of the config (so the family positional can follow)
{ [ -n "${SISTEER_SHARED_ROOT:-}" ] && source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA/scripts/sdfbase/full22/common.sh" 2>/dev/null; } \
  || source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit
PAD_DIR=scripts/sdfbase/full22/static_padding
# THE WIDTH PIN (header "SEED REPLICATES"): the seed-1 wave's attested max_num_seqs at 98,304 — a LITERAL, never re-derived here,
# passed to the lane preflight (--expect-served-s) and to the post-standup GEOMETRY PIN, which refuse anything else by name
GEOMETRY_PIN_S=5
# THE CONFIG FAMILY -> its view (header "SEED REPLICATES"): the seed-1 family's `padobj`, a seed replicate's `padobj_s<N>` — the
# one spelling make_padding_configs.view_of renders; refused by name otherwise, before any roster is read
case "$CONFIG_FAMILY" in
  obj)    VIEW=padobj;;
  obj_s2) VIEW=padobj_s2;;
  obj_s3) VIEW=padobj_s3;;
  # paper-companion-repo: a FRONT-DOOR key's PER-VIEW steered family — named after
  # the phase view it pads (`s2`, `s1`, `d0p75`, `d0p75_relu`; make_padding_configs.front_door_view_families) — serves the PRODUCTION
  # roster's bank view of that name (no lane roster, no cross-bank pin). The generator is the one source of the spelling; this script
  # asks it (`--resolve-family <key> <family>`: `<bank view> <roster> <registry view>`; the driver venv carries its PyYAML — f22_driver_venv
  # is sentinel-guarded, so the standing call further down is then a no-op) and refuses anything it does not know, by name, before any
  # roster is read. NO literal width pin for such a family: GEOMETRY_PIN_S = 5 is the seed-1 wave's S on qwen35_4b; the preflight derives
  # S for THIS model and the post-standup GEOMETRY PIN holds the attested record to that derivation (its literal checks are skipped).
  # This arm runs BEFORE f22_init binds REPO_DIR (the lane roster it selects must stand first), so the resolver runs in
  # `${REPO_DIR:-$PWD}` as the registry block does: the job's working directory IS the checkout (the funnel submits `cd <checkout> &&
  # sbatch`). A bare `$REPO_DIR` under `set -u` killed the process substitution, `read` saw EOF and the refusal below named a family
  # the generator resolves (padobj-qwen35_4b_demo_battery-p0, cluster A 981270, dead in 54 s).
  *) f22_driver_venv
     read -r VIEW FD_ROSTER FD_REGISTRY_VIEW < <(cd "${REPO_DIR:-$PWD}" && "$MPY" "$PAD_DIR/make_padding_configs.py" --resolve-family "$KEY" "$CONFIG_FAMILY") \
       || f22_fatal "family '$CONFIG_FAMILY' is not a config family of this lane for $KEY (obj | obj_s2 | obj_s3 for the hand-listed key, or a front-door key's phase view; make_padding_configs.families_of / --resolve-family)"
     GEOMETRY_PIN_S=""
     f22_log "front-door config family $CONFIG_FAMILY of $KEY -> production bank view $VIEW, roster $FD_ROSTER, registry view $FD_REGISTRY_VIEW (make_padding_configs.py --resolve-family); no literal width pin";;
esac
# THE LANE'S ROSTER (header): every roster reader of this job reads the `padobj` view from it — set BEFORE f22_init (which
# checks the path exists in the checkout and exports it) and unconditionally: an inherited F22_ROSTER is this lane's to
# override, and is logged so a submit-line selection is never silently lost
[ -z "${F22_ROSTER:-}" ] || f22_log "note: inherited F22_ROSTER=$F22_ROSTER is replaced by this lane's roster (the padobj view lives there only)"
F22_ROSTER="configs/sdfbase/static_padding/sdfbase_full22_${KEY}.${VIEW}_roster.yaml"
[ -z "${FD_ROSTER:-}" ] || F22_ROSTER="$FD_ROSTER"     # a front door's per-view family: the PRODUCTION roster names its view (the resolver's answer)
# THE LAUNCH'S CELLS for the kind-aware serving-path default (common.sh f22_select_serving_path): this family's manifest —
# none + two prompted baselines + seven natively servable objects (a seed replicate: six objects, no passthrough) -> lora_native,
# recorded; the GEOMETRY PIN re-checks it
MANIFEST="configs/capeval/manifests/petri_full22_${KEY}.padstatic_${CONFIG_FAMILY}.yaml"
F22_LAUNCH_MANIFEST="$MANIFEST"
f22_init "$WAVE_SHA"
case "$WINDOW" in ''|*[!0-9]*) f22_fatal "window '$WINDOW' is not a positive integer";; esac
[ "$WINDOW" -ge 1 ] || f22_fatal "window must be >= 1"
case "$PAD" in ''|*[!0-9]*) f22_fatal "pad '$PAD' is not ONE non-negative integer (one job per level: 0 | 8192 | 16384 | 32768 | 65536; a comma list is refused)";; esac
# --- shards: begin (rendered by tests/test_static_padding_wave.py under a stubbed f22_fatal) ---
# THE SHARD RANGE (header "SHARDING ACROSS JOBS"): `N` (all N shards: the single-job shape) or `<lo>-<hi>/N` (this job owns
# shards lo..hi of N). Parsed here, before anything else is checked; every malformation, an inverted range and an
# out-of-range index refuse BY NAME — a shard index >= N would make the driver slice an empty goal list and write an empty
# shard directory, silently (jailbreak_run slices goals[shard::nshards] without checking the range).
case "$SHARDS_SPEC" in
  */*) SHARD_RANGE="${SHARDS_SPEC%%/*}"; NSHARDS="${SHARDS_SPEC#*/}";;
  *)   SHARD_RANGE=""; NSHARDS="$SHARDS_SPEC";;
esac
case "$NSHARDS" in ''|*[!0-9]*) f22_fatal "shards '$SHARDS_SPEC': N '$NSHARDS' is not a positive integer (the spec is N or <lo>-<hi>/N)";; esac
NSHARDS=$((10#$NSHARDS))
[ "$NSHARDS" -ge 1 ] || f22_fatal "shards '$SHARDS_SPEC': N must be >= 1"
[ "$NSHARDS" -le 99 ] || f22_fatal "shards '$SHARDS_SPEC': N=$NSHARDS > 99 — the store names shards shard00..shard99 (two digits; the grader sorts by name)"
if [ -z "$SHARD_RANGE" ]; then
  SHARD_LO=0; SHARD_HI=$((NSHARDS - 1))
else
  case "$SHARD_RANGE" in *-*) ;; *) f22_fatal "shards '$SHARDS_SPEC': the range before the slash must be <lo>-<hi> (inclusive, 0-based)";; esac
  SHARD_LO="${SHARD_RANGE%%-*}"; SHARD_HI="${SHARD_RANGE#*-}"
  case "$SHARD_LO" in ''|*[!0-9]*) f22_fatal "shards '$SHARDS_SPEC': lo '$SHARD_LO' is not a non-negative integer";; esac
  case "$SHARD_HI" in ''|*[!0-9]*) f22_fatal "shards '$SHARDS_SPEC': hi '$SHARD_HI' is not a non-negative integer";; esac
  SHARD_LO=$((10#$SHARD_LO)); SHARD_HI=$((10#$SHARD_HI))
  [ "$SHARD_LO" -le "$SHARD_HI" ] || f22_fatal "shards '$SHARDS_SPEC': lo $SHARD_LO > hi $SHARD_HI (inverted range)"
  [ "$SHARD_HI" -lt "$NSHARDS" ] || f22_fatal "shards '$SHARDS_SPEC': hi $SHARD_HI is out of range for N=$NSHARDS (the shards are 0..$((NSHARDS - 1)))"
fi
SHARD_COUNT=$((SHARD_HI - SHARD_LO + 1))
# --- shards: end ---
[ -f "$MANIFEST" ] || f22_fatal "missing $MANIFEST — generate it: python $PAD_DIR/make_padding_configs.py"
[ -f "$F22_ROSTER" ] || f22_fatal "missing $F22_ROSTER — generate it: python $PAD_DIR/make_padding_configs.py"
# THE LEVEL -> its generated config (ONE config per level: the store identity; make_padding_configs.py, the `_obj` family or a
# seed replicate's `_obj_s<N>`)
CFG="configs/sdfbase/static_padding/sdfbase_full22_${KEY}.padstatic_${CONFIG_FAMILY}_p${PAD}.yaml"
[ -f "$CFG" ] || f22_fatal "missing $CFG for level $PAD — every level has its own generated config (python $PAD_DIR/make_padding_configs.py)"
TAG="${KEY}_${VIEW}"
IFS=',' read -ra FAMS <<< "$FAMILIES"
[ "${#FAMS[@]}" -ge 1 ] || f22_fatal "families '$FAMILIES' names none"
for fam in "${FAMS[@]}"; do case "$fam" in pasttense|codeattack|renellm|grok420) ;; *) f22_fatal "family '$fam' is not a static one-shot family (this leg runs pasttense,codeattack,renellm,grok420 only)";; esac; done
f22_stop_requested && { f22_log "wave STOP present — not starting"; exit 0; }
SERVE_DIR="$WAVE_ROOT/static_padding/serving/${TAG}_p${PAD}_${SLURM_JOB_ID}"; PRIVATE_ROOT="$WAVE_ROOT/static_padding/transcripts_results"
mkdir -p "$SERVE_DIR/drivers" "$PRIVATE_ROOT"; chmod 700 "$PRIVATE_ROOT"
FILLER="${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}/data/pg19_filler.json"
f22_log "padded static (steered): key=$KEY family=$CONFIG_FAMILY view=$VIEW window=$WINDOW pad=$PAD families=${FAMS[*]} shards=$SHARD_LO-$SHARD_HI/$NSHARDS ($SHARD_COUNT of $NSHARDS per (family, cell)) cells=${CELLS_ARG:-all} roster=$F22_ROSTER manifest=$MANIFEST cfg=$CFG filler=$FILLER serve_dir=$SERVE_DIR private_root=$PRIVATE_ROOT"
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
# THE REGISTRY IDENTITY OF THIS LEVEL (header "THE STORE IDENTITY"): this job's one family registers at job start on the level's
# OWN instrument (padding_tokens = $PAD + the filler contract's hash + the ceiling cap + this window; sisteer.registry.jobplan
# PADDING_ENV, migration 0012) under the PHASE view its objects belong to — F22_REGISTRY_VIEW, default the config family's seed
# view (obj -> s1, obj_s2 -> s2, obj_s3 -> s3), read from the PRODUCTION roster for the registration call only (the s<N> views
# live there; the lane roster names `padobj[_s<N>]`, which is not a model-key spelling). An EMPTY level block skips the
# registration outright — a padded level must never register on the unpadded instrument. Best-effort by contract: every
# failure is one [registry] line and the job continues.
case "$CONFIG_FAMILY" in obj) PAD_REGISTRY_VIEW_DEFAULT=s1;; obj_s2) PAD_REGISTRY_VIEW_DEFAULT=s2;; obj_s3) PAD_REGISTRY_VIEW_DEFAULT=s3;; *) PAD_REGISTRY_VIEW_DEFAULT="${FD_REGISTRY_VIEW:-main}";; esac   # a front door's per-view family registers under ITS phase view (the resolver's answer)
F22_REGISTRY_EVALS="${FAMILIES//,/ }"; export F22_REGISTRY_EVALS
SISTEER_REGISTRY_PADDING_JSON="$(PYTHONPATH="${REPO_DIR:-$PWD}/src${PYTHONPATH:+:$PYTHONPATH}" "$MPY" -m sisteer.registry.jobplan padding-block \
    --config "$CFG" --filler "$FILLER" --window "$WINDOW" 2>"$SERVE_DIR/padding_block.err")" \
  || { f22_log "[registry] padding-block helper failed for $CFG (level p$PAD): $(tr -d '\n' < "$SERVE_DIR/padding_block.err" | head -c 400) — this level registers nothing"; SISTEER_REGISTRY_PADDING_JSON=""; }
if [ -n "$SISTEER_REGISTRY_PADDING_JSON" ]; then
  export SISTEER_REGISTRY_PADDING_JSON
  F22_REGISTRY_VIEW="${F22_REGISTRY_VIEW:-$PAD_REGISTRY_VIEW_DEFAULT}"
  f22_log "[registry] registering level p$PAD for evals [$F22_REGISTRY_EVALS] under view $F22_REGISTRY_VIEW from roster $F22_ROSTER_PRODUCTION (bank view stays $VIEW from $F22_ROSTER; padding block $(printf '%s' "$SISTEER_REGISTRY_PADDING_JSON" | head -c 300))"
  PAD_BANK_VIEW="$VIEW"; PAD_LANE_ROSTER="$F22_ROSTER"; VIEW="$F22_REGISTRY_VIEW"; F22_ROSTER="$F22_ROSTER_PRODUCTION"
  export F22_REGISTRY_KEY_SEGMENTS="${F22_REGISTRY_KEY_SEGMENTS:-p${PAD}-sh${SHARDS_SPEC//[-\/]/_}}"   # `sdfbase_full22-padobj-<key>[_<view>]-p<level>-sh<lo>_<hi>_<n>`
  f22_registry_register || true    # in THIS shell: it exports SISTEER_REGISTRY_RUN_IDENTITY, which every store this job opens routes through
  VIEW="$PAD_BANK_VIEW"; F22_ROSTER="$PAD_LANE_ROSTER"
  if [ -n "${SISTEER_REGISTRY_RUN_IDENTITY:-}" ]; then f22_log "[registry] runs of this job: $SISTEER_REGISTRY_RUN_IDENTITY"
  else f22_log "[registry] nothing registered for level p$PAD (no spool on this cluster, or the registration refused — see the [registry] lines above); the stores still file, unregistered"; fi
else
  f22_log "[registry] no padding block for level p$PAD: registration SKIPPED (never on the unpadded instrument)"
fi
# --- registry: end ---
# THE BANK VIEW (header): the CAMPAIGN's helper over the LANE's roster — verify first, build only under the (KEY, VIEW) lock,
# every entry's object.pt + staged_from.json + exactly one identity sidecar, the short-view guard (config cells == roster
# served count; view entries == roster staged count). Exports TT_TWO_SI_DEFENSES_DIR and fills ALL_CELLS from $CFG.
f22_bank_view "$MPY"
# THE PREFLIGHT (padding_preflight.py, BEFORE the standup): the filler's sha256 + size + loader, the model snapshot, the
# harness's tokens + cap + reserve rule, the ceiling cap, the GEOMETRY this window buys (the same derivation the pin below
# holds the attested record to), and the RENDERED need per cell (a compiled cell carries no system message — si_text_of)
"$MPY" "$PAD_DIR/padding_preflight.py" --window "$WINDOW" --model-key "$KEY" --steered --config "$CFG" --filler "$FILLER" --families "$FAMILIES" \
    ${GEOMETRY_PIN_S:+--expect-served-s "$GEOMETRY_PIN_S"} ${GPU_MEM_UTIL:+--gpu-mem-util "$GPU_MEM_UTIL"} --out "$SERVE_DIR/padding_preflight.json" \
  || f22_fatal "padding preflight REFUSED at window $WINDOW (see $SERVE_DIR/padding_preflight.json and the REFUSED lines above) — no GPU work started"
f22_log "padding preflight PASS: $(tail -c 400 "$SERVE_DIR/padding_preflight.json" | tr -d '\n' | head -c 200)"
if [ -n "$CELLS_ARG" ]; then IFS=',' read -ra CELLS <<< "$CELLS_ARG"; else CELLS=("${ALL_CELLS[@]}"); fi
for c in "${CELLS[@]}"; do case " ${ALL_CELLS[*]} " in *" $c "*) ;; *) f22_fatal "cell $c is not in $CFG's explicit_cells (${ALL_CELLS[*]}) — a job serves only cells the config names";; esac; done
NCELLS=${#CELLS[@]}
# --- shard leases: begin (rendered by tests/test_static_padding_wave.py under stubs for f22_fatal / f22_log / squeue / sacct) ---
# THE LEASES (header "SHARDING ACROSS JOBS"): one JSON per (family, shard) this job owns, beside the store (never inside
# it: nothing under <run>/<family>/<cell>/ but the drivers' shard directories). Checked and taken BEFORE the standup, so an
# overlap or a re-laid-out store refuses without a GPU minute; marked exited by cleanup() so a resubmit finds it free.
RUN="pad_static_${KEY}_${CONFIG_FAMILY}_p${PAD}"
LEASE_DIR="$WAVE_ROOT/static_padding/shard_leases/$RUN"
pad_lease_path(){ printf '%s/%s__s%02d.json' "$LEASE_DIR" "$1" "$2"; }   # $1 = family, $2 = shard index
pad_job_alive(){   # $1 = a Slurm job id -> 0 alive (a non-terminal state), 1 gone (a terminal state), 2 unknown (neither tool answers)
  local st
  st=$(squeue -h -j "$1" -o %T 2>/dev/null) && [ -n "$st" ] && return 0
  st=$(sacct -X -n -P -j "$1" -o State 2>/dev/null | head -1 | awk '{print $1}')
  case "$st" in
    '') return 2;;
    RUNNING|PENDING|COMPLETING|SUSPENDED|REQUEUED|REQUEUE_HOLD|REQUEUE_FED|RESIZING|CONFIGURING|SIGNALING|STAGE_OUT|STOPPED) return 0;;
    *) return 1;;
  esac
}
pad_lease_check(){   # refuses (f22_fatal) any live or mismatched claim on this job's shards, and any store laid out under another N
  local fam s lease rec o_job o_n o_state alive cell sd nn
  mkdir -p "$LEASE_DIR"
  for fam in "${FAMS[@]}"; do
    for s in $(seq "$SHARD_LO" "$SHARD_HI"); do
      lease=$(pad_lease_path "$fam" "$s")
      [ -f "$lease" ] || continue
      rec=$("$MPY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["job_id"], int(d["nshards"]), d["state"])' "$lease") \
        || f22_fatal "shard $s of $fam: unreadable lease $lease — refusing to claim the shard over it"
      read -r o_job o_n o_state <<< "$rec"
      [ "$o_n" = "$NSHARDS" ] || f22_fatal "shard $s of $fam is leased under N=$o_n by job $o_job ($lease); this job says N=$NSHARDS — two layouts of one store; refusing"
      [ "$o_state" = running ] || continue                       # an exited job's lease is free: the resume after a TIMEOUT
      [ "$o_job" != "${SLURM_JOB_ID:-manual}" ] || continue      # our own (a re-entered check)
      pad_job_alive "$o_job"; alive=$?
      case "$alive" in
        0) f22_fatal "shard $s of $fam is held by LIVE job $o_job ($lease) — overlapping shard ranges; refusing rather than double-write shard$(printf '%02d' "$s")";;
        2) f22_fatal "shard $s of $fam is held by job $o_job ($lease) and neither squeue nor sacct can say whether it is alive — refusing rather than risk a double write; if it is dead, delete the lease by hand";;
      esac
      f22_log "shard $s of $fam: stale lease from job $o_job (exited without releasing) — taken over; the store resumes per goal"
    done
    # the on-disk belt: a shard directory of this store numbered beyond N, or one in this job's range with no lease at all,
    # is a layout this job cannot vouch for (written under another N, before the lease mechanism, or by hand)
    for cell in "${CELLS[@]}"; do
      for sd in "$PRIVATE_ROOT/$RUN/$fam/$cell"/shard*; do
        [ -d "$sd" ] || continue
        nn=${sd##*/shard}
        case "$nn" in ''|*[!0-9]*) f22_fatal "$sd is not a shardNN directory of this store — refusing";; esac
        nn=$((10#$nn))
        [ "$nn" -lt "$NSHARDS" ] || f22_fatal "$sd exists but this job says N=$NSHARDS — the store on disk was laid out under a larger N; refusing"
        if [ "$nn" -ge "$SHARD_LO" ] && [ "$nn" -le "$SHARD_HI" ] && [ ! -f "$(pad_lease_path "$fam" "$nn")" ]; then
          f22_fatal "$sd exists with no lease under $LEASE_DIR — a layout this job cannot vouch for (written before the lease mechanism, or by hand); record its N in a lease, or move it, before resuming into it"
        fi
      done
    done
  done
}
pad_lease_write(){   # $1 = state (running | exited), $2 = rc (a number, or null)
  local fam s
  for fam in "${FAMS[@]}"; do for s in $(seq "$SHARD_LO" "$SHARD_HI"); do
    printf '{"schema": "padobj_shard_lease/1", "run": "%s", "family": "%s", "shard": %d, "nshards": %d, "range": "%d-%d", "job_id": "%s", "job_name": "%s", "node": "%s", "state": "%s", "rc": %s, "cells": "%s", "written_at": "%s"}\n' \
      "$RUN" "$fam" "$s" "$NSHARDS" "$SHARD_LO" "$SHARD_HI" "${SLURM_JOB_ID:-manual}" "${SLURM_JOB_NAME:-manual}" "$(hostname)" "$1" "${2:-null}" "${CELLS[*]}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > "$(pad_lease_path "$fam" "$s")"
  done; done
}
pad_lease_check
pad_lease_write running null; LEASE_WRITTEN=1
f22_log "shard leases taken: $RUN shards $SHARD_LO-$SHARD_HI/$NSHARDS for ${FAMS[*]} under $LEASE_DIR"
# --- shard leases: end ---
# the LOAD CHECK sends the LONGEST prompted SI as the system prefix too (target_static.sbatch's prompted rule): the probe
# then hits that cell's route at the run's cap, S concurrent — the window is sized for it, so it must pass
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
# the sampler the ENGINE resolved goes on the record: one INFO line per request, no prompt
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
  # release this job's shard leases (state exited): a resubmit of the same range finds them free and resumes per goal
  [ -z "${LEASE_WRITTEN:-}" ] || pad_lease_write exited "${RC:-1}" 2>/dev/null || true
  f22_result "padobj $TAG family=$CONFIG_FAMILY window=$WINDOW pad=$PAD families=$FAMILIES shards=$SHARD_LO-$SHARD_HI/$NSHARDS rc=${RC} exit=${ec} failed_drivers=${TOTAL_FAIL}"
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
TARGET_GPU="$CUDA_VISIBLE_DEVICES"
# ---------------------------------------------------------------- the ONE standup: families_served:short at <window> x the derived S
f22_standup attacks "$SERVE_DIR/static" "" families_served:short "$WINDOW" || { RC=1; exit 1; }
# native-serving-default: the cells this standup SERVES (every object cell of this family is natively servable — a residual-slot
# kind or an o_proj LoRA; a cells= positional naming an unserved cell is fatal, never a silent drop)
f22_served_cells_filter CELLS "$CELLS_ARG"; NCELLS=${#CELLS[@]}
[ "$TARGET_L" = "$WINDOW" ] || { f22_log "REFUSING TRAFFIC: the standup resolved max_model_len $TARGET_L but this job was sized (preflight) for $WINDOW"; RC=1; exit 1; }
# THE GEOMETRY PIN (header): the attested record must show (1) the native engine, (2) the zero adapter none__lora0 among the
# attested adapters, (3) an attested S equal to the kernel-ceiling S the geometry module derives at this window from the
# model's budget and PINNED widest module, (4) that adjustment in endpoint.json's chain ending at S — and (5) the chain's
# recorded D equal to the pinned width: THIS standup registers the ten cells' real exports, so a wider-than-pinned adapter
# would lower S below the derivation and put this level on a geometry no other level shares. Read from the AUTHORITATIVE
# record (endpoint.json), never from the env. And (6), since 2026-09-24: the LITERAL width pin — attested S, the derivation
# AND the chain's kernel step must all equal GEOMETRY_PIN_S (the seed-1 wave's 5), so a standup at the identity 7 (what the
# six-cell attempt printed before refusing) refuses even if the derivation or the geometry table moved.
# --- geometry pin: begin (rendered by tests/test_static_padding_wave.py against synthetic endpoint records) ---
"$MPY" - "$SERVE_DIR/static/endpoint.json" "$KEY" "$WINDOW" "$TARGET_S" "$GEOMETRY_PIN_S" <<'PY' || { f22_log "REFUSING TRAFFIC: the standup's geometry is not the pinned geometry (see the GEOMETRY PIN lines above and the header)"; RC=1; exit 1; }
import json, sys
from pathlib import Path
import sisteer
from sisteer.capeval.petri.defense_cells import NOOP_LORA_CELL_ID
from sisteer.capeval.serving import geometry as g
sys.path.insert(0, str(Path(sisteer.__file__).resolve().parents[2] / "scripts/sdfbase/full22/static_padding"))
import padding_preflight as pf   # the preflight's own selection rule: ONE source for the selected module set
ep = json.load(open(sys.argv[1])); key, L, S = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
# the literal width pin: the seed-1 wave's 5 for qwen35_4b's families; EMPTY for a front door's per-view family (D51: no literal —
# the derivation for THIS model is the pin, and every other check below still holds the attested record to it)
PIN_S = int(sys.argv[5]) if sys.argv[5] else None
# THE BUDGET RESOLVER (the coordinator's ruling 2026-09-29): geometry.budget_for / rows_for — the ONE lookup over the sweep's keys AND the
# paper-companion ONBOARDED ones (padding_preflight.geometry_expectation applies the same rule before the standup); never BUDGET_BY_MODEL
# indexed directly. An unknown key, or a model with no PINNED widest module for the kernel step, refuses BY NAME (never a KeyError).
# THE SELECTED MODULE SET (the closing review's finding 1; D69): the geometry overlay is certified PER MODULE SET, and a native standup's
# ceiling is the D of the class it registered — so this pin resolves the record of the HPO winner's set, read from the ONE
# source every padded job has at hand, reproduce/models.yaml's promoted `lora_architecture` (padding_preflight.selected_module_set;
# the preflight before the standup read the same), never a model-wide maximum; a set the overlay does not certify refuses by name
try:
    SET, sel = pf.selected_module_set(key)
except (ValueError, KeyError) as e:
    print(f"[geometry-pin] REFUSED: {e}", flush=True); sys.exit(1)
try:
    rows = g.rows_for(key, SET); rb = g.budget_for(key, SET)
except (g.UnknownModelKey, g.SiteGeometryError) as e:
    print(f"[geometry-pin] REFUSED: {e}", flush=True); sys.exit(1)
# D145: geometry certification and adapter parity are SEPARATE facts and the overlay says both. THIS standup is STEERED (it serves the
# pick's trained adapter), so the selected pair's adapter_parity must be 'verified' — a mismatch / unverified record refuses by name
# here, while the unsteered launcher (target_static_padded.sbatch) reads the same geometry whatever the block says
if rows.site is not None:
    try:
        par = g.require_verified_adapter_parity(rows.site, sel.get("rank"), what="this steered padded standup")
        print(f"[geometry-pin] adapter parity for (rank {sel.get('rank')}, set {SET!r}): {par['status']} (basis {par['basis']})", flush=True)
    except g.SiteGeometryError as e:
        print(f"[geometry-pin] REFUSED: {e}", flush=True); sys.exit(1)
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
    print(f"[geometry-pin] REFUSED: {key!r}: {e} — the kernel-ceiling step of the pinned geometry cannot be derived; pin the width from the "
          "serving probe's fold in the site geometry overlay (geometry.lora_max_module_dim_for; the built-in geometry.LORA_MAX_MODULE_DIM_BY_MODEL for the sweep's models) before this model's steered padding runs", flush=True); sys.exit(1)
expect = int(adj["to_S"]) if adj else identity
names = sorted((ep.get("id_name_map") or {}).values())
chain = ((ep.get("geometry") or {}).get("adjustments") or [])
problems = []
if ep.get("engine") != "lora_native":
    problems.append(f"engine {ep.get('engine')!r} != lora_native")
if NOOP_LORA_CELL_ID not in names:
    problems.append(f"the zero adapter {NOOP_LORA_CELL_ID} is not among the attested adapters {names}")
# THE LITERAL WIDTH PIN (header "SEED REPLICATES"): the instrument is ONE attested width, the seed-1 wave's; the attested S,
# the module's derivation at this window and the attested chain must all say so — a derivation that moved is itself a refusal
if PIN_S is not None and S != PIN_S:
    problems.append(f"attested max_num_seqs {S} != the lane's pinned width {PIN_S} (the seed-1 wave's S at 98,304) — not the pinned instrument")
if PIN_S is not None and expect != PIN_S:
    problems.append(f"the geometry module derives S {expect} at L={L} (identity {identity}, kernel step {adj}) != the pinned width {PIN_S} — the "
                    "derivation moved; the pin, not the derivation, names the instrument")
if PIN_S is not None and adj is None and PIN_S < identity:
    problems.append(f"the pinned width {PIN_S} is below the identity S {identity} but no kernel-ceiling step lowers it at L={L}")
if PIN_S is not None and not any(a.get("kind") == g.ADJUSTMENT_KERNEL and int(a.get("to_S", -1)) == PIN_S for a in chain):
    problems.append(f"endpoint.json geometry.adjustments {chain} carries no {g.ADJUSTMENT_KERNEL} step ending at the pinned width {PIN_S} — "
                    "the width was not reached through the LoRA kernel index ceiling (the zero adapter sizes it: is `none` in the cell set?)")
if S != expect:
    problems.append(f"attested max_num_seqs {S} != {expect}, the S the pinned geometry derives at L={L} (identity {identity} at budget "
                    f"{rb.batched_token_budget:,} ({rb.source}); kernel ceiling {adj['ceiling']:,} // {L} = {adj['to_S']})" if adj
                    else f"attested max_num_seqs {S} != the identity S {expect} at L={L} (the kernel ceiling does not bind here)")
if adj is not None:
    steps = [a for a in chain if a.get("kind") == adj["kind"] and int(a.get("to_S", -1)) == S]
    if not steps:
        problems.append(f"endpoint.json geometry.adjustments {chain} carries no {adj['kind']} step ending at S={S}")
    # EVERY kernel step's D (not only one already at the attested S) must be the SELECTED set's certified D: a standup that
    # registered adapters of another class than the HPO pick's attests another ceiling, whatever S it ended at (D69)
    for a in [a for a in chain if a.get("kind") == adj["kind"]]:
        if a.get("D") is not None and int(a["D"]) != int(adj["D"]):
            problems.append(f"the attested kernel step's widest module D={a['D']} != the pinned {adj['D']} (the selected module set {SET!r} "
                            f"certifies D={adj['D']}) — the standup registered adapters of another class than the HPO pick's (a registered "
                            "export is wider or narrower than the derivation assumes)")
if problems:
    print("[geometry-pin] REFUSED: " + "; ".join(problems), flush=True); sys.exit(1)
print(f"[geometry-pin] OK: engine lora_native, adapters {names}, S {S} x L {L} = identity {identity} -> kernel ceiling "
      f"{adj['to_S'] if adj else 'n/a'} at D {adj['D'] if adj else 'n/a'} for module set {SET!r} (chain {[a.get('kind') for a in chain]}) "
      + (f"== the literal pin {PIN_S} — the pinned steered geometry" if PIN_S is not None
         else "== the derivation for this model (no literal pin: a front door's per-view family, D51)"), flush=True)
PY
# --- geometry pin: end ---
# STEER-IS-LIVE positive control on the first steered cell of this job (common.sh f22_positive_control): the smoke's check,
# run here so a silently no-op'd object refuses before any driver spends a GPU minute (target_static.sbatch's rule)
for pc in "${CELLS[@]}"; do case "$pc" in none|prompted__*) continue;; esac; f22_positive_control "$pc" "static_p${PAD}" || { RC=1; exit 1; }; break; done   # the first OBJECT cell: a prompted baseline has no scale-0 twin
# ~one in-flight request per driver: fill the attested width, never more (target_static.sbatch's rule and cap)
DRIVER_CAP=96
MAX_DRIVERS=$(( TARGET_S < DRIVER_CAP ? TARGET_S : DRIVER_CAP ))
f22_log "phase PADDED STATIC (steered): level=p$PAD families=${FAMS[*]} cells=$NCELLS shards=$SHARD_LO-$SHARD_HI/$NSHARDS drivers=$((NCELLS * SHARD_COUNT * ${#FAMS[@]})) server_width=$TARGET_S window=$TARGET_L max_drivers=$MAX_DRIVERS (cap $DRIVER_CAP)"
# --- drivers: begin (rendered by tests/test_static_padding_wave.py under stubs; the loop body is the argv contract) ---
# ONE LEVEL PER JOB: this job's drivers fill the width for its level, so the log carries one wall time per level (the cost
# model's measurement) — or per shard RANGE of a level, in the sharded shape (header): the shard loop runs lo..hi of N, the
# argv a single job at N would issue for exactly those indices, and every cell gets the same range. A padded level's drivers
# carry the FILLER PATH ONLY (the level itself is the config's); the p0 control carries no padding flag at all (a sub-flag
# while the knob is off is refused by the driver, by design).
PAD_FLAGS=(); [ "$PAD" -gt 0 ] && PAD_FLAGS=(--static-padding-filler "$FILLER")
LEVEL_T0=$SECONDS
f22_log "level p$PAD: config $CFG padding_flags=[${PAD_FLAGS[*]:-none}]"
for fam in "${FAMS[@]}"; do for cell in "${CELLS[@]}"; do for s in $(seq "$SHARD_LO" "$SHARD_HI"); do
  f22_stop_requested && { f22_log "wave STOP present — not launching more drivers"; break 3; }
  f22_queue_launch "p${PAD}__${fam}__${cell}__s$(printf '%02d' "$s")" "$MPY" scripts/jailbreak_run.py generate \
    --config "$CFG" --attack "$fam" --defense "$cell" --confirmatory --shard "$s" --nshards "$NSHARDS" \
    --private-root "$PRIVATE_ROOT" --target-engine served --target-proxy-url "$PROXY_URL" \
    --target-routing-table "$TABLE" --target-max-model-len "$TARGET_L" ${PAD_FLAGS[@]+"${PAD_FLAGS[@]}"}
  sleep 1
done; done; done
f22_log "level p$PAD: all ${#DRIVER_PIDS[@]} drivers launched (queued at <= $MAX_DRIVERS in flight); waiting"
f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
f22_log "level p$PAD shards $SHARD_LO-$SHARD_HI/$NSHARDS done: $NFAIL failed drivers; wall $(( SECONDS - LEVEL_T0 )) s"
# --- drivers: end ---
f22_standup_stop
RC=$(( TOTAL_FAIL > 0 ? 2 : 0 ))
f22_log "level done: $TOTAL_FAIL failed drivers; stores under $PRIVATE_ROOT/$RUN/<family>/<cell>/shardNN (this job: shard$(printf '%02d' "$SHARD_LO")..shard$(printf '%02d' "$SHARD_HI") of $NSHARDS)"
exit $RC
