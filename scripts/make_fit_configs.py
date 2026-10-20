#!/usr/bin/env python
"""Emit the twosi-model-sweeps FIT configs (the production steering-defense
objects, post-HPO) from the per-model sweep configs.

One deterministic transform per model — committed outputs are byte-pinned by
tests/test_fit_configs.py (which regenerates and compares), so the four
files under configs/ can never drift from this script or from each other:

  two_si_llama_fits.yaml        Llama-3.1-8B, single-process invocations
                                (the cluster token_faithful leg, Modal
                                fallback fits, memprobes)
  two_si_llama_fits_dp3.yaml    identical + data_parallel.gpus: 3 — the
                                cluster RESPONSE wave (3 arms 1-per-GPU via
                                run_parallel_fit; the scheduler REFUSES a
                                pool != 3, so the tf/1-GPU leg needs the
                                dp-free sibling)
  two_si_qwen38_27b_fits.yaml   Qwen3.8-27B (Modal H200)
  two_si_gemma4e4b_fits.yaml    gemma-4-E4B-it (Modal H100)

What a FIT config changes vs its source config:

1. max_len 16384 + per-SI re-derivation for the CLEAN v3 pools. The v3
   pools carry completions GENERATED AT 4096 tokens (the 512-truncation
   fix), and the trainers' preflight requires max_len >= longest rendered
   teacher prompt + (longest completion + EOS) — lockstep.py derives the
   completion budget FROM THE DATA, so the config's completion_tokens: 512
   (a generation-time knob, inert at fit time) does not save the old 8192.
   Per SI: rendered teacher prompt + 1024 attack allowance + 4096 + 1 EOS,
   next power of two (the two_si_llama_hpo.yaml convention; prompt token
   counts from each model's derive_model_config header, measured on the
   pinned tokenizers).
2. batch_size_by_arm RE-MEMPROBED AT 16384. The HPO/interim widths were
   measured at 8192 and are INVALID for the fits. Measured entries below
   carry their provenance; unmeasured arms keep fail-loud placeholder
   strings (resolve_batch_size int()s them).
3. train_kwargs.<arm> = the model's HPO-winner optimizer settings (FIXED —
   the fits do not sweep). E4B: the v4 folded winners. Qwen: the ballpark
   interim harvest (~/hpo-r1/qwen_interim, tune_arm.py analyze — see
   QWEN_INTERIM_WINNERS provenance). Llama: PENDING its HPO — sentinels
   that fail loudly (optimizer_banner/float() raise) until
   --llama-winners <winners.json> folds the real values:
       python scripts/make_fit_configs.py \
           --llama-winners ~/hpo-r1/llama/llama_hpo_winners.json
   (winners.json arm-keyed: {"<arm>": {"lr", "warmup_frac",
   "max_grad_norm"}} — the e4b_hpo_winners_v4.json shape. The re-tune@32
   harvest additionally carries temperature/weight_decay/lora_rank/
   accum_steps (+ batch_size, validated against the measured table) and
   folds through the same hook — _WINNER_FOLD_KEYS. NOTE the E4B/Qwen
   winners above predate the re-tune (effective-2/7, clip-confounded,
   online-teacher objective) and stand only until its harvest replaces
   them.)
4. class_a_arms / class_c_arms: the sweep's four/four — token_faithful
   DROPPED with the re-tune (2026-08-27); every fit cell is
   positions=response (train_kwargs.default.positions, the queued
   all->response flip).

Data paths are NOT config keys (--pairs at invocation); the per-model fit
pools are:
  Llama-3.1-8B   $HOME/twosi_llama_v3/pairs/ (READY; manifest
                 e4dc12a32193..., 10478 pairs/SI) -> cluster copy under
                 $SHARED/exp/twosweep/pairs/llama_v3/
  Qwen3.8-27B    PENDING grading (thinking-OFF dataset — NOT the
                 separate with-reasoning variant); stage to the Modal
                 twosi-fits volume when it lands
  gemma-4-E4B-it PENDING grading; same staging
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent

# The sweep's fit cells: decode-only positions for the three re-tune arms.
# token_faithful was DROPPED from the campaign (twosi-model-sweeps re-tune,
# 2026-08-27) — the re-HPO and the fits are these three arms only, all at
# positions=response (folded as the train_kwargs.default below, the queued
# all->response flip; with tf gone there is no all-token exception left).
# lora_r4 is lora_r1_all's post-re-tune name (rank-4 winner on every
# model; 'all' implicit) — FUTURE fits emit under it; the 12 staged
# legacy-named objects keep their name (Evals maps it) and stay listed
# in the class lists below so they remain servable.
FIT_ARMS_RESPONSE = ["psr_kl_all", "psr_mse_all", "lora_r4"]
FIT_ARMS = list(FIT_ARMS_RESPONSE)

# Sweep arm lists (the E4B/Qwen derived-config lists; the Llama fit config
# adopts them so every model's fit world names the same arms and the
# class_c-inside-class_a glob test holds). token_faithful dropped with the
# re-tune (its all-token leg left the campaign with it).
CLASS_A_ARMS = ["psr_kl_all", "psr_mse_all", "lora_r1_all", "lora_r4",
                "bidpo"]
CLASS_C_ARMS = ["psr_kl_all", "psr_mse_all", "lora_r1_all", "lora_r4",
                "bidpo"]

BS_PLACEHOLDER = "UNMEASURED_AT_16384_RUN_MEMPROBE_FIRST"

#: E4B winners — $HOME/hpo-r1/e4b_hpo_winners_v4.json (fitted argmin, Sobol +
#: recenters on Modal H200s, 2026-08-26; identical to the values already
#: folded into configs/two_si_gemma4e4b.yaml @ f451e7d + the lora/tf
#: recenter fold).
#: RE-TUNE@32 winners (2026-08-27): psr_kl is the round-1 T=1 best-raw
#: (5 usable, non-PD near-tie — best-raw is the default); lora lr/clip
#: from a 2-D reduced fit with rank resolved by per-rank best-raw
#: comparison (rank 4: y 0.0478 vs r1 0.0516 / r16 0.0525). wf=0.15 +
#: wd=0.01 fixed campaign-wide (cross-model agreement of both models'
#: best trials); T=1.0 (the offline top-K head is only trainable at
#: T=1 — 27/27 T!=1 trials diverged/floor-failed); accum 16 at the
#: probed bs2 = effective 32. psr_mse keeps the OLD v4 value (that
#: cell is not in the 15-fit wave; value inert).
E4B_WINNERS_V4 = {
    "psr_kl_all": {"lr": 4.387e-4, "warmup_frac": 0.15,
                   "max_grad_norm": 0.496, "temperature": 1.0,
                   "weight_decay": 0.01, "accum_steps": 16,
                   "batch_size": 2},
    "psr_mse_all": {"lr": 0.0014216398292720336, "warmup_frac": 0.05,
                    "max_grad_norm": 0.6712460967470124},
    "lora_r4": {"lr": 9.11e-5, "warmup_frac": 0.15,
                    "max_grad_norm": 0.301, "temperature": 1.0,
                    "weight_decay": 0.01, "lora_rank": 4,
                    "accum_steps": 16, "batch_size": 2},
    # token_faithful's v4 winner (lr 2.08e-2 / wf 0.15 / clip 0.288)
    # dropped with the arm (re-tune 2026-08-27).
}

#: Qwen ballpark interim harvest — tune_arm.py analyze --rounds 1 over
#: ~/hpo-r1/qwen_interim/<cell>/ (pod round-1 Sobol, 16 trials/cell,
#: si_constitution, 8192-era pool; the LRs transfer to the thinking-off v3
#: fits per the campaign's confound reasoning). Selection per cell:
#: fitted_argmin where the quadratic surface had a trustworthy interior
#: minimum (psr_kl, psr_mse), otherwise the best RAW trial (lora_r1_all,
#: token_faithful — Hessian not positive definite; recorded in each cell's
#: r1_fit.json).
QWEN_INTERIM_WINNERS = {
    "psr_kl_all": {"lr": 0.0003721580402109105, "warmup_frac": 0.05,
                   "max_grad_norm": 0.07978378399252271},   # fitted_argmin
    "psr_mse_all": {"lr": 0.0005788815718723408, "warmup_frac": 0.15,
                    "max_grad_norm": 0.16346645789826733},  # fitted_argmin
    "lora_r4": {"lr": 0.0009342348544080183, "warmup_frac": 0.15,
                "max_grad_norm": 0.039075011288633436},  # best raw t11 (legacy lora_r1_all label)
    # token_faithful's interim harvest (best raw t14) dropped with the arm.
}

#: Llama sentinel — fails loudly (optimizer_banner floats every setting;
#: the trainers float() lr) until the running Llama HPO's winners are
#: folded via --llama-winners.
LLAMA_PENDING = "PENDING_LLAMA_HPO_WINNERS"

#: Llama winners land HERE as each cell's HPO concludes — the E4B/Qwen
#: pattern: generator-pinned constants, so build_all(None) (what
#: tests/test_fit_configs.py byte-pins the committed configs against)
#: carries every landed winner and keeps the sentinel for the rest. A
#: --llama-winners file merges OVER these (CLI wins on conflict).
#: 2026-08-27: EMPTY. token_faithful's cluster r1 fitted argmin came back
#: EDGE-PINNED at the lr box top (edge_pos.lr=0.999, winner: null —
#: artifact ~/hpo-r1/llama_token_faithful/r1_fit.json), so its recenter
#: round (cluster array llama-hpo-r2tf-s1, box lr [3e-4, 3e-2] /
#: clip [9.9e-3, 0.99]) must land before it folds; psr_kl/psr_mse/lora_r1
#: pend their Modal r2 refine.
#: RE-TUNE@32 winners (2026-08-27, user-authorized fit HPs): psr_kl from
#: the 2-D T=1 reduced fit's imputed minimum (lr verified +-10% against
#: the box-constrained argmin of the fitted surface at wf 0.15; clip on
#: the fit's near-flat low-clip ridge, interior value ~= best-raw 0.048);
#: lora lr/clip from the same reduced-fit imputation with rank 4 by
#: per-rank best-raw (y 0.0421 vs r1 0.0497 / r16 0.0459); psr_mse
#: best-raw (25 usable, non-PD — user-approved default; PART B, held
#: until the psr-mse early-stop fix merges). wf/wd/T/accum as E4B above.
LLAMA_LANDED_WINNERS: dict[str, dict] = {
    "psr_kl_all": {"lr": 3.19e-4, "warmup_frac": 0.15,
                   "max_grad_norm": 0.054, "temperature": 1.0,
                   "weight_decay": 0.01, "accum_steps": 16,
                   "batch_size": 2},
    "lora_r4": {"lr": 2.45e-4, "warmup_frac": 0.15,
                    "max_grad_norm": 0.451, "temperature": 1.0,
                    "weight_decay": 0.01, "lora_rank": 4,
                    "accum_steps": 16, "batch_size": 2},
    "psr_mse_all": {"lr": 1.05e-3, "warmup_frac": 0.15,
                    "max_grad_norm": 0.036,
                    "weight_decay": 0.01, "accum_steps": 16,
                    "batch_size": 2},
}

#: Measured batch widths at max_len 16384 (worst-case longest-rows probe,
#: 0.85 reserved-VRAM guard) — per model, per card. Provenance in the
#: emitted headers. Placeholders where the probe has not run yet.
BS_LLAMA_H200 = {  # Modal H200 memprobe on the real v3 pool (2026-08-26,
    # consumed by the running cluster HPO r1 sbatch): bs2 peaks psr_kl
    # 91.2 GiB / psr_mse 83.0 (tf 88.0 — arm since dropped); lora_r1_all
    # bs4 OOM. NOTE the re-tune@32 consumes the offline top-K teacher +
    # per-arm accumulation — re-memprobe under THAT world before launch.
    "psr_kl_all": 2, "psr_mse_all": 2, "lora_r1_all": 2, "lora_r4": 2,
}
#: Qwen at 16384, all measured 2026-08-27 on the real v3 pool's worst-case
#: rows. token_faithful: H200 + per-arm gradient checkpointing, bs1 peaks
#: 80.7-81.2 alloc / 84.7-88.1 reserved GiB across all three SI buckets
#: (bs2 reserved 119.7 = 0.9 over the 0.85 guard -> bs1). The all-layer
#: hook arms CANNOT checkpoint (torch CheckpointError — the recompute pass
#: saves a different tensor set once the per-layer steering hooks fire) and
#: OOM uncheckpointed at bs1 on H200-141GB (B200 no-ckpt alloc peaks
#: 134.7-140.7 GiB > H200's 139.8 usable), so their bs1 is measured on
#: B200-192GB (reserved 140.6-146.5 vs the 155.6 guard; bs2 OOMs). The
#: B200 substrate for those 9 cells deviates from "everything H200" and
#: must be approved before use.
BS_QWEN = {"psr_kl_all": 1, "psr_mse_all": 1, "lora_r1_all": 1,
           "lora_r4": 1}
#: E4B on H200 at 16384 (Modal twosi-fits memprobe, 2026-08-27, synthetic
#: v3 worst-case rows — 1024-tok prompts + 4096-tok completions, every row;
#: artifacts /vol/out/gemma4e4b/memprobe_16384_*.json). bs1 peak RESERVED:
#: psr_mse 100.7 GiB (under the 0.85 guard's 119.85); psr_kl 122.2 /
#: lora_r1_all 122.6 / token_faithful 120.5 GiB — 0.5-2.3% OVER the guard
#: but rc=0 with ~19 GiB of headroom to the card (alloc ~109-111): an
#: over-guard bs1 is accepted when bs1 is the only width there is.
#: bs2 fails (CUBLAS_STATUS_ALLOC_FAILED on the roomiest cell).
#: 2026-08-27 offline-teacher re-probe (--mode hpomem, worst-case rows,
#: 0.85 guard): all three arms fit bs2 (110.1 / 95.8 / 110.8 GiB
#: reserved; bs4 OOM) — the old bs1 was measured with the ONLINE teacher
#: resident.
BS_E4B_H200 = {"psr_kl_all": 2, "psr_mse_all": 2, "lora_r1_all": 2,
               "lora_r4": 2}

#: Qwen3.5 safety-lane psr_kl winners (promoted 2026-08-29, hand-picked,
#: no refit) — the TRANSFER fits reuse them FIXED on the non-safety SIs.
#: Full HP set copied from the promoted objects' summary.json
#: (out/qwen35_{9b,4b}/cells_response/safety_refusal__psr_kl_all/ on the
#: twosi-fits volume): 9B = r1_t05, 4B = r1_t02 (the mid-box pick from
#: the dead-tie top pair, deliberately NOT t05). Both trained at
#: bs2 x accum 32 = EFFECTIVE 64 (the 3B/E2B safety-lane regime), wf 0.15,
#: T=1.0, wd=0.01, epochs 10 warmup_cosine, patience 3, floor_ratio_max
#: 1.0, offline top-K=200 teacher, positions=response
#: (response_header_len derives to 7 on both tokenizers).
#: lora_r4: the qwen35 lora HPO winners at RANK 4 — the matrix arm (this
#: reverses the interim rank-16 pick, 2026-08-29; cells bank as
#: <si>__lora_r4, the matrix cell id, no rename). Per-rank best-raw
#: winners are DIFFERENT Sobol points per model: 9B = r1_t08, 4B =
#: r1_t00. Exact floats read from each model's trial summary at
#: out/hpo/qwen35_{9b,4b}/lora_r4_response/r1_t{08,00}/summary.json on
#: the twosi-fits volume (NOT the promoted bank — a concurrent promotion
#: writes there). Both trials ran bs2 x accum 32 = effective 64 (peak
#: alloc 56.2 GiB 9B / 59.2 GiB 4B, gradient checkpointing on).
#: psr_mse_all has no qwen35 HPO — it keeps the per-model PENDING
#: sentinel (a fit launched early fails loudly).
QWEN35_WINNERS = {
    "qwen35_9b": {
        "psr_kl_all": {"lr": 0.0034750201684699013, "warmup_frac": 0.15,
                       "max_grad_norm": 0.025805365267459347,
                       "temperature": 1.0, "weight_decay": 0.01,
                       "accum_steps": 32, "batch_size": 2},
        "lora_r4": {"lr": 0.002924740549316289, "warmup_frac": 0.15,
                    "max_grad_norm": 1.1910381365720537,
                    "temperature": 1.0, "weight_decay": 0.01,
                    "lora_rank": 4, "accum_steps": 32, "batch_size": 2},
    },
    "qwen35_4b": {
        "psr_kl_all": {"lr": 0.0021912898676553186, "warmup_frac": 0.15,
                       "max_grad_norm": 0.19294010750831866,
                       "temperature": 1.0, "weight_decay": 0.01,
                       "accum_steps": 32, "batch_size": 2},
        "lora_r4": {"lr": 0.00219324600173211, "warmup_frac": 0.15,
                    "max_grad_norm": 0.24882679700493568,
                    "temperature": 1.0, "weight_decay": 0.01,
                    "lora_rank": 4, "accum_steps": 32, "batch_size": 2},
    },
}

#: Qwen3.5 batch widths: psr_kl + lora bs2 = each lane's hpomem-measured
#: width (offline top-K teacher + the custom kernels + gradient
#: checkpointing, 0.85x141 guard), which the winning trials then RAN
#: (psr_kl peak alloc 75.7 GiB 9B / 59.9 GiB 4B; lora r1_t08/r1_t00
#: 56.2 / 59.2 GiB). The long-SI transfer fits keep bs2: with the offline
#: teacher the STUDENT rendering (no SI prefix, ~5.2k tokens max) drives
#: fit memory, so the 16384 teacher-side bucket does not change the
#: trained shape. Other arms unmeasured -> placeholder.
BS_QWEN35 = {"psr_kl_all": 2, "lora_r4": 2}

#: Model hidden dims, read from the promoted fitted objects
#: (object.pt directions): 9B 4096, 4B 2560 (32 layers both). Folded into
#: pool.fit_floor.dims so the alphakl floor rows are sized to THIS model
#: (the hand-derived HPO configs inherited llama's 4096 — right for the
#: 9B by coincidence, wrong for the 4B).
QWEN35_HIDDEN = {"qwen35_9b": 4096, "qwen35_4b": 2560}

#: Gemma-4-E2B safety-lane psr_kl winner (2026-08-30) — the TRANSFER fits
#: reuse it FIXED on the non-safety SIs, the qwen35-lane shape. r1_t04,
#: the best heldout loss of the 8-trial safety sweep (y 0.029850;
#: runner-up r1_t07 0.030017); exact floats from
#: out/hpo/gemma4e2b/psr_kl_all_response/r1_t04/summary.json on the
#: twosi-fits volume. Trained at bs2 x accum 16 = EFFECTIVE 32 (the E2B
#: safety-lane regime), wf 0.15, T=1.0, wd=0.01, epochs 10 warmup_cosine,
#: patience 3, floor_ratio_max 1.0, offline top-K=200 teacher,
#: positions=response — the promoted safety cell
#: (out/gemma4e2b/cells_response/safety_refusal__psr_kl_all) ran exactly
#: this regime (its interim config rounded the floats to 4.81e-4/0.423).
#: lora_r4: the E2B lora HPO winner r1_t00 (folded 2026-08-30) — RANK 4
#: won OUTRIGHT (best raw of the round; unlike qwen35, no rank-16 record
#: needed), the matrix arm, banks as <si>__lora_r4. Exact floats from
#: out/hpo/gemma4e2b/lora_r4_response/r1_t00/summary.json on the
#: twosi-fits volume (NOT the promoted bank — the qwen35-lane rule).
#: Ran bs2 x accum 16 = EFFECTIVE 32 with NO gradient checkpointing
#: (hpomem: bs2 fits E2B without it; t00 peak alloc 82.3 GiB under the
#: 0.85x141 guard) — build_gemma4e2b passes no ckpt_arms, so the
#: emitted arm carries no gradient_checkpointing key, matching the HPO
#: regime. psr_mse_all: no E2B HPO -> sentinel.
E2B_WINNERS = {
    "psr_kl_all": {"lr": 0.00048108646249242905, "warmup_frac": 0.15,
                   "max_grad_norm": 0.4234878350553384,
                   "temperature": 1.0, "weight_decay": 0.01,
                   "accum_steps": 16, "batch_size": 2},
    "lora_r4": {"lr": 0.001315947601039265, "warmup_frac": 0.15,
                "max_grad_norm": 0.24882679700493568,
                "temperature": 1.0, "weight_decay": 0.01,
                "lora_rank": 4, "accum_steps": 16, "batch_size": 2},
}

#: E2B batch width: psr_kl bs2 = the memprobe width on the REAL pool at
#: both buckets (safety@8192 104.4 GiB reserved; si_constitution@16384
#: 106.5 GiB reserved, bs4 OOM — 0.85x141 guard 118.8), which the safety
#: winner then ran. lora_r4 bs2 = the width the E2B lora HPO itself ran
#: (t00 peak alloc 82.3 GiB, no gradient checkpointing). Other arms
#: unmeasured -> placeholder.
BS_E2B = {"psr_kl_all": 2, "lora_r4": 2}

#: E2B hidden dim — text_config.hidden_size at the pinned revision
#: (Gemma4ForConditionalGeneration, 35 text layers). The probe source
#: inherited E4B's 4096 fit_floor dims; corrected here.
E2B_HIDDEN = 1536

#: Per-SI max_len at the v3 pools' 4096-token completion budget:
#: rendered teacher prompt + 1024 attack allowance + 4096 completion +
#: 1 EOS -> next power of two. Rendered-prompt token counts measured on
#: each pinned tokenizer (derive_model_config.py headers /
#: two_si_llama_hpo.yaml). preflight_max_len re-checks the REAL rendered
#: rows before any GPU work, so an undersized entry fails loudly, never
#: silently truncates.
MAX_LEN_BY_SI = {
    "llama": {"safety_refusal": 8192,      # 104 + 1024 + 4096 + 1 = 5225
              "si_constitution": 16384,    # 5888 + ... = 11009
              "si_irrelevant": 16384},     # 6396 + ... = 11517
    "qwen38_27b": {"safety_refusal": 8192,     # 105 -> 5226
                   "si_constitution": 16384,   # 5975 -> 11096
                   "si_irrelevant": 16384},    # 6695 -> 11816
    "gemma4e4b": {"safety_refusal": 8192,      # 101 -> 5222
                  "si_constitution": 16384,    # 6041 -> 11162
                  "si_irrelevant": 16384},     # 6574 -> 11695
    # Qwen3.5: the hand-derived HPO configs' values (safety longest row
    # 4102 tokens < 8192; the long SIs render ~11-12k like every sibling
    # -> 16384; preflight_max_len re-checks the real rows either way).
    "qwen35_9b": {"safety_refusal": 8192,
                  "si_constitution": 16384,
                  "si_irrelevant": 16384},
    "qwen35_4b": {"safety_refusal": 8192,
                  "si_constitution": 16384,
                  "si_irrelevant": 16384},
    # E2B: E4B's MatFormer width-slice sibling — the same measured
    # rendered-prompt regime (safety longest row < 8192; the long SIs
    # render ~11-12k -> 16384; preflight_max_len re-checks the real rows).
    "gemma4e2b": {"safety_refusal": 8192,
                  "si_constitution": 16384,
                  "si_irrelevant": 16384},
}


#: Winner keys that fold into a fit config's train_kwargs.<arm> block, in
#: emission order. The re-tune@32 harvest adds temperature/weight_decay/
#: lora_rank/accum_steps to the classic lr/warmup_frac/max_grad_norm shape
#: (tune_arm.py analyze's winner dict); anything else on a winner
#: (batch_size, effective_batch, provenance) is deliberately NOT a trainer
#: kwarg — batch_size is validated against the measured table instead.
_WINNER_FOLD_KEYS = ("lr", "warmup_frac", "max_grad_norm", "temperature",
                     "weight_decay", "lora_rank", "accum_steps")


def _fit_train_kwargs(source_tk: dict, winners: dict | None,
                      ckpt_arms: tuple = (),
                      measured_bs: dict | None = None,
                      pending_sentinel: str = LLAMA_PENDING) -> dict:
    """train_kwargs for a fit config: the source's default block verbatim
    plus the re-tune's positions=response default (the queued all->response
    flip — with token_faithful dropped there is no all-token cell left),
    bidpo's objective-semantics floor, and one block per fit arm carrying
    the model's FIXED winner settings (or the fail-loud sentinel).

    A re-tune winner carrying batch_size must MATCH the model's measured
    batch_size_by_arm entry — accum_steps only reproduces the tuned
    effective batch at the batch size it was tuned with, so a mismatch
    refuses instead of silently refitting at a different effective batch.

    ``ckpt_arms`` get gradient_checkpointing: true PER ARM (the
    backward-inside-apply() fix, 2026-08-27; tests/test_ckpt_grad_parity.py
    pins ckpt==unckpt trained objects per hook arm). Kept per-arm rather
    than model-global so a future incompatible arm fails one cell, not the
    config."""
    default = dict(source_tk.get("default") or {})
    default["positions"] = "response"
    # LR schedule horizon = 10 epochs: warmup_frac
    # 0.15 -> 1.5-epoch linear warmup, cosine decay to epoch 10. The old
    # in-code default budget (6) ended early-stopped runs mid-ramp /
    # barely into decay; 10 gives peak-by-1.5-epochs then real decay
    # under patience early stopping. Forward-only: applies to fits
    # launched after this fold.
    default["epochs"] = 10
    # Patience 3: the per-epoch held-out signal is
    # noisy and patience-2 false-triggered mid-ramp; measured on the only
    # full-horizon runs (llama psr_mse x5): best epoch ~13, epoch-2.5
    # stops cost a mean 23.5% of the residual reduction.
    default["early_stop_patience"] = 3
    tk = {"default": default,
          "bidpo": {"floor_ratio_max": 3.0}}
    for arm in FIT_ARMS:
        w = (winners or {}).get(arm)
        if w is None:
            tk[arm] = {"lr": pending_sentinel,
                       "lr_schedule": "warmup_cosine",
                       "warmup_frac": pending_sentinel,
                       "max_grad_norm": pending_sentinel}
        else:
            if w.get("batch_size") is not None and measured_bs is not None:
                want = measured_bs.get(arm)
                if want is not None and int(w["batch_size"]) != int(want):
                    raise SystemExit(
                        f"{arm}: the winner was tuned at batch_size "
                        f"{w['batch_size']} but this model's measured "
                        f"batch_size_by_arm entry is {want} — its "
                        "accum_steps would realize a DIFFERENT effective "
                        "batch than the HPO tuned; re-probe or re-tune")
            tk[arm] = {"lr": w["lr"], "lr_schedule": "warmup_cosine",
                       **{k: w[k] for k in _WINNER_FOLD_KEYS
                          if k not in ("lr",) and k in w}}
        if arm in ckpt_arms:
            tk[arm]["gradient_checkpointing"] = True
    return tk


def _fit_batch_table(measured: dict) -> dict:
    table = {}
    for arm in CLASS_A_ARMS:
        table[arm] = measured.get(arm, BS_PLACEHOLDER)
    return table


def derive_fit_config(source: dict, *, slug: str, winners: dict | None,
                      measured_bs: dict, data_parallel_gpus: int = 0,
                      ckpt_arms: tuple = (),
                      pending_sentinel: str = LLAMA_PENDING) -> dict:
    cfg = dict(source)  # shallow: replaced keys are reassigned, not mutated
    cfg["class_a_arms"] = list(CLASS_A_ARMS)
    cfg["class_c_arms"] = list(CLASS_C_ARMS)
    cfg["max_len"] = 16384
    cfg["max_len_by_si"] = dict(MAX_LEN_BY_SI[slug])
    cfg["batch_size_by_si"] = {}
    cfg["batch_size_by_arm"] = _fit_batch_table(measured_bs)
    cfg["train_kwargs"] = _fit_train_kwargs(
        cfg.get("train_kwargs") or {}, winners, ckpt_arms=ckpt_arms,
        measured_bs=measured_bs, pending_sentinel=pending_sentinel)
    if data_parallel_gpus:
        cfg["data_parallel"] = {"gpus": int(data_parallel_gpus)}
    else:
        cfg.pop("data_parallel", None)
    return cfg


#: The standing FIT-config header paragraph (the committed four; the
#: byte-pin test regenerates these lines verbatim — do not edit casually).
_HEADER_BODY = [
    "FIT config for the twosi-model-sweeps production objects: 9",
    "fits/model = 3 SIs x {psr_kl_all, psr_mse_all, lora_r1_all}, all",
    "at positions=response (train_kwargs.default.positions — the",
    "re-tune's all->response flip; token_faithful DROPPED 2026-08-27),",
    "FULL clean v3 pool, FIXED HPO-winner optimizer settings (the",
    "re-tune@32 winners add temperature/weight_decay/lora_rank/",
    "accum_steps to the classic shape), max_len 16384 (the v3",
    "completions were generated at 4096 tokens; the trainers derive",
    "the completion budget from the DATA, so completion_tokens: 512",
    "below is a generation-time knob that does not cap the fit).",
]

#: The qwen35 TRANSFER lane's paragraph (build_qwen35): a different
#: campaign shape, and NOT covered by the byte-pin test's committed set.
_QWEN35_HEADER_BODY = [
    "TRANSFER-fit config for the Qwen3.5 safety-winner transfer (the",
    "2026-08-29 lane): psr_kl_all ONLY, on {si_constitution,",
    "si_irrelevant}, at the FIXED safety-lane HPO winner's full HP set —",
    "no new HPO. positions=response, FULL frozen pool, offline top-K=200",
    "teacher, max_len_by_si 8192/16384/16384 (the trainers derive the",
    "completion budget from the DATA; completion_tokens: 512 below is a",
    "generation-time knob that does not cap the fit).",
]

#: The gemma-4-E2B TRANSFER lane's paragraph (build_gemma4e2b): the qwen35
#: campaign shape on the E2B lane, likewise outside the byte-pin set.
_E2B_HEADER_BODY = [
    "TRANSFER-fit config for the gemma-4-E2B safety-winner transfer (the",
    "2026-08-30 lane): psr_kl_all ONLY, on {si_constitution,",
    "si_irrelevant}, at the FIXED safety-lane HPO winner's full HP set —",
    "no new HPO. positions=response, FULL frozen pool, offline top-K=200",
    "teacher, max_len_by_si 8192/16384/16384 (the trainers derive the",
    "completion budget from the DATA; completion_tokens: 512 below is a",
    "generation-time knob that does not cap the fit).",
]

_PIN_LINE = "Pinned byte-for-byte by tests/test_fit_configs.py."
_QWEN35_PIN_LINE = ("NOT in the byte-pin test's committed set (its source "
                    "is untracked/INTERIM).")
_E2B_PIN_LINE = ("NOT in the byte-pin test's committed set (its source is "
                 "the hand-derived INTERIM probe config).")


def _header(name: str, source_name: str, notes: list[str],
            body: list[str] | None = None,
            pin_line: str = _PIN_LINE) -> str:
    lines = [
        f"# {name} — DERIVED, do not hand-edit.",
        f"# Regenerate: python scripts/make_fit_configs.py "
        f"(source: configs/{source_name})",
        f"# {pin_line}",
        "#",
    ]
    lines += [f"# {n}" for n in (body if body is not None
                                 else _HEADER_BODY)]
    lines += ["#"]
    lines += [f"# {n}" for n in notes]
    return "\n".join(lines) + "\n"


def emit(path: Path, header: str, cfg: dict) -> None:
    body = yaml.safe_dump(cfg, sort_keys=False, width=79,
                          default_flow_style=False, allow_unicode=True)
    path.write_text(header + body)
    print(f"wrote {path}")


def build_all(llama_winners: dict | None) -> dict[str, tuple[str, dict]]:
    """(header, cfg) per output filename — the single source of truth the
    CLI writes and the test regenerates."""
    # Landed winners under CLI winners: each Llama cell's HPO lands on its
    # own clock (partial folds are the NORMAL case), and _fit_train_kwargs
    # already leaves the sentinel for any arm absent from the merge.
    llama_winners = {**LLAMA_LANDED_WINNERS, **(llama_winners or {})} or None
    # The Llama source is the COMMITTED base config: outside the keys this
    # transform replaces, two_si.yaml and the (untracked, cluster-staged)
    # two_si_llama_hpo.yaml are byte-identical worlds — verified 2026-08-27
    # — so deriving from the base keeps a fresh clone reproducible while
    # landing exactly the HPO lane's max_len/bs world via the transform.
    llama_src = yaml.safe_load((REPO / "configs/two_si.yaml").read_text())
    # anti-neg + reversed-benign PINNED ON for the fit lane. The base turned
    # these OFF on 2026-08-26 for DATASET GENERATION, which produces no
    # anti-neg side; these are FIT configs and they train a `bidpo` arm, and
    # the manifest refuses a bidpo with no harmful preferences. Deriving the
    # Llama fits from the base therefore silently disabled bidpo for them
    # (the qwen/e4b fits are unaffected — they derive from per-model configs
    # that pin it on). The split is resolved as of 2026-08-28: the base stays
    # off, the fit lane keeps its own on. Same pin as derive_model_config.py.
    for _k in ("anti_neg", "bidpo_benign"):
        llama_src["pool"][_k] = {**llama_src["pool"][_k], "enabled": True}
    qwen_src = yaml.safe_load(
        (REPO / "configs/two_si_qwen38_27b.yaml").read_text())
    e4b_src = yaml.safe_load(
        (REPO / "configs/two_si_gemma4e4b.yaml").read_text())

    llama_notes = [
        "model: NousResearch/Meta-Llama-3.1-8B-Instruct @ d10aef7 (public)",
        "pool: ~/twosi_llama_v3/pairs/ (READY, 10478 pairs/SI, manifest",
        "  e4dc12a32193...) -> $SHARED/exp/twosweep/pairs/llama_v3/",
        "batch_size_by_arm: MEASURED at 16384 on H200 (Modal worst-case",
        "  probe, 0.85x141 GiB guard, real v3 pool — the numbers the",
        "  running llama-hpo-r1 sbatch consumes): bs2 peaks psr_kl 91.2 /",
        "  psr_mse 83.0 / token_faithful 88.0 GiB; lora_r1_all bs4 OOM.",
        "train_kwargs: PENDING_LLAMA_HPO_WINNERS sentinels — the Llama HPO",
        "  (cluster array llama-hpo-r1) is still running. Folding hook:",
        "  python scripts/make_fit_configs.py --llama-winners <winners.json>",
        "  (harvest: tune_arm.py analyze per cell after the refine round).",
        "  A fit launched before the fold fails loudly on the sentinel.",
    ]
    tf_note = [
        "THIS VARIANT: no data_parallel — single-process invocations (the",
        "cluster token_faithful/all leg on 1 GPU, Modal fallback fits, and",
        "memprobes). The response wave uses two_si_llama_fits_dp3.yaml.",
    ]
    dp3_note = [
        "THIS VARIANT: data_parallel.gpus: 3 — the cluster RESPONSE wave",
        "(one invocation per SI: psr_kl_all,psr_mse_all,lora_r1_all packed",
        "1-per-GPU by run_parallel_fit, which REFUSES a pool != 3). The",
        "token_faithful/all leg and every 1-GPU use take",
        "two_si_llama_fits.yaml (identical but for this block —",
        "tests/test_fit_configs.py asserts that).",
    ]
    qwen_notes = [
        "model: Qwen/Qwen3.8-27B @ 1d4bf0f2 (hybrid linear attention:",
        "  every fit venv/image needs triton 3.7.1 + flash-linear-attention",
        "  0.5.2 + causal-conv1d 1.6.2.post1, the fused kernels; scripts/modal/",
        "  twosi_fits.py bakes them).",
        "pool: PENDING TrainTest's clean thinking-OFF Qwen v3 grading (NOT",
        "  the separate with-reasoning dataset). Stage pairs + tokcache to",
        "  the Modal twosi-fits volume when it lands.",
        "train_kwargs: the BALLPARK interim harvest (~/hpo-r1/qwen_interim,",
        "  tune_arm.py analyze r1): psr_kl/psr_mse fitted_argmin;",
        "  lora_r1_all/token_faithful best raw trial (Hessian not PD).",
        "  Applied to the thinking-off fits — LRs transfer per the",
        "  campaign's confound reasoning.",
        "batch_size_by_arm: bs1 everywhere at 16384 (2026-08-27,",
        "  worst-case v3 rows). token_faithful H200+ckpt MEASURED:",
        "  80.7-81.2 GiB alloc (bs2 over guard). The hook arms run",
        "  H200+ckpt since the backward-inside-apply() fix (they used to",
        "  CheckpointError; uncheckpointed they OOM at bs1 — B200 alloc",
        "  peaks 134.7-140.7 GiB > H200's 139.8); their checkpointed",
        "  peaks are memprobed before launch (twosi_fits.py memprobe).",
    ]
    e4b_notes = [
        "model: google/gemma-4-E4B-it @ ee0ef60 (gated — Modal hf-token",
        "  secret; standard attention, no hybrid stack).",
        "pool: PENDING TrainTest's clean E4B v3 grading. Stage pairs +",
        "  tokcache to the Modal twosi-fits volume when it lands.",
        "train_kwargs: the e4b_hpo_winners_v4.json fitted-argmin winners",
        "  (already the folded values of two_si_gemma4e4b.yaml).",
        "batch_size_by_arm: MEASURED at 16384 on H200 (Modal twosi-fits",
        "  memprobe 2026-08-27, worst-case synthetic v3 rows, 0.85x141",
        "  reserved guard): bs1 everywhere. psr_mse 100.7 GiB reserved;",
        "  psr_kl/lora_r1_all/token_faithful 120.5-122.6 GiB — over the",
        "  guard by 0.5-2.3% but measured to train (alloc ~110, ~19 GiB to",
        "  the card; the over-guard exception shape — bs1 is the only width).",
        "  bs2 CUBLAS-OOMs. Fits run on H200 (since 2026-08-27: everything",
        "  H200).",
    ]

    out = {}
    base = derive_fit_config(llama_src, slug="llama", winners=llama_winners,
                             measured_bs=BS_LLAMA_H200)
    out["two_si_llama_fits.yaml"] = (
        _header("two_si_llama_fits.yaml", "two_si.yaml",
                llama_notes + tf_note), base)
    dp3 = derive_fit_config(llama_src, slug="llama", winners=llama_winners,
                            measured_bs=BS_LLAMA_H200, data_parallel_gpus=3)
    out["two_si_llama_fits_dp3.yaml"] = (
        _header("two_si_llama_fits_dp3.yaml", "two_si.yaml",
                llama_notes + dp3_note), dp3)
    out["two_si_qwen38_27b_fits.yaml"] = (
        _header("two_si_qwen38_27b_fits.yaml", "two_si_qwen38_27b.yaml",
                qwen_notes),
        derive_fit_config(qwen_src, slug="qwen38_27b",
                          winners=QWEN_INTERIM_WINNERS,
                          measured_bs=BS_QWEN,
                          ckpt_arms=tuple(FIT_ARMS)))
    out["two_si_gemma4e4b_fits.yaml"] = (
        _header("two_si_gemma4e4b_fits.yaml", "two_si_gemma4e4b.yaml",
                e4b_notes),
        derive_fit_config(e4b_src, slug="gemma4e4b",
                          winners=E4B_WINNERS_V4,
                          measured_bs=BS_E4B_H200))
    return out


def build_qwen35() -> dict[str, tuple[str, dict]]:
    """The Qwen3.5 TRANSFER-fits lane (2026-08-29), SEPARATE from
    build_all: its sources are the hand-derived, deliberately untracked
    two_si_qwen35_{9b,4b}_hpo.yaml configs (the llama3b-lane precedent),
    so the byte-pin test's committed set must not depend on them. Emits
    two_si_qwen35_{9b,4b}_fits.yaml with the FIXED safety psr_kl winners
    folded (QWEN35_WINNERS) and PENDING sentinels on psr_mse_all/lora_r4.
    Skips (loudly) any model whose source config is absent."""
    out: dict[str, tuple[str, dict]] = {}
    for key in ("qwen35_9b", "qwen35_4b"):
        src_name = f"two_si_{key}_hpo.yaml"
        src_path = REPO / "configs" / src_name
        if not src_path.exists():
            print(f"NOTE: qwen35 lane SKIPPED for {key} — configs/"
                  f"{src_name} absent (untracked INTERIM source; stage it "
                  "before regenerating this lane).")
            continue
        src = yaml.safe_load(src_path.read_text())
        pending = f"PENDING_{key.upper()}_HPO_WINNER"
        cfg = derive_fit_config(
            src, slug=key, winners=QWEN35_WINNERS[key],
            measured_bs=BS_QWEN35, ckpt_arms=tuple(FIT_ARMS),
            pending_sentinel=pending)
        # Size the alphakl floor dims to THIS model's hidden dim (read
        # from the promoted fitted object) — the hand-derived sources
        # inherited llama's 4096, wrong for the 4B (2560). Inert at fit
        # time (build_two_si_pool.py territory) but kept correct.
        hidden = QWEN35_HIDDEN[key]
        pool = dict(cfg.get("pool") or {})
        ff = dict(pool.get("fit_floor") or {})
        ff["dims"] = {k: hidden for k in (ff.get("dims") or {})}
        pool["fit_floor"] = ff
        cfg["pool"] = pool
        winner_note = ("r1_t05" if key == "qwen35_9b"
                       else "r1_t02 (the user's mid-box pick from the "
                            "dead-tie top pair)")
        w = QWEN35_WINNERS[key]["psr_kl_all"]
        lw = QWEN35_WINNERS[key]["lora_r4"]
        lora_note = "r1_t08" if key == "qwen35_9b" else "r1_t00"
        notes = [
            f"model: {cfg['model']} @ {str(cfg['model_revision'])[:7]} "
            "(qwen3_5 hybrid attention: the D66",
            "  image stack — triton 3.7.1 + flash-linear-attention 0.5.2 +",
            "  causal-conv1d 1.6.2.post1; scripts/modal/twosi_fits.py bakes",
            "  them and routes this key to image_qwen).",
            f"pool: pool/{key}/ on the twosi-fits volume (the FROZEN",
            "  TrainTest 2026-08-28 pool + pre-built tokcache sidecars,",
            "  uploaded 2026-08-29 for the long SIs).",
            "train_kwargs.psr_kl_all: the SAFETY-lane HPO winner "
            f"{winner_note},",
            f"  promoted 2026-08-29 (lr {w['lr']:.6g}, clip "
            f"{w['max_grad_norm']:.6g}, wf 0.15, T 1.0, wd 0.01,",
            "  accum 32 @ bs2 = effective 64, epochs 10, patience 3,",
            "  floor_ratio_max 1.0, offline top-K=200 teacher) — a FIXED-HP",
            "  TRANSFER onto si_constitution/si_irrelevant, NOT a new HPO.",
            "  psr_mse_all: no qwen35 HPO -> sentinel.",
            f"train_kwargs.lora_r4: the qwen35 lora HPO winner {lora_note}",
            "  (RANK 4 — the matrix arm, banks as <si>__lora_r4; per-rank",
            "  best-raw, a DIFFERENT Sobol point per model): "
            f"lr {lw['lr']:.6g},",
            f"  clip {lw['max_grad_norm']:.6g}, lora_rank 4, wf 0.15, "
            "T 1.0, wd 0.01,",
            "  accum 32 @ bs2 = effective 64, gradient checkpointing on —",
            "  the same FIXED-HP TRANSFER shape as psr_kl_all.",
            "batch_size_by_arm: psr_kl + lora_r4 bs2 = each lane's hpomem",
            "  width, which the winning trials ran (offline teacher keeps",
            "  the student rendering ~5.2k tokens, so the long-SI 16384",
            "  teacher bucket does not move fit memory).",
            f"fit_floor dims: {hidden} = this model's hidden dim, read",
            "  from the promoted object.pt (response_header_len 7 both).",
        ]
        name = f"two_si_{key}_fits.yaml"
        out[name] = (_header(name, src_name, notes,
                             body=_QWEN35_HEADER_BODY,
                             pin_line=_QWEN35_PIN_LINE), cfg)
    return out


def build_gemma4e2b() -> dict[str, tuple[str, dict]]:
    """The gemma-4-E2B TRANSFER-fits lane (2026-08-30), cloning the
    build_qwen35 pattern: the source is the committed hand-derived INTERIM
    probe config (two_si_gemma4e2b_probe.yaml — model/revision swapped
    from E4B), so the byte-pin test's committed set must not depend on it.
    Emits two_si_gemma4e2b_fits.yaml with the FIXED safety psr_kl and
    lora_r4 winners folded (E2B_WINNERS; the lora winner r1_t00 landed
    2026-08-30) and a PENDING sentinel on psr_mse_all (no E2B HPO)."""
    src_name = "two_si_gemma4e2b_probe.yaml"
    src = yaml.safe_load((REPO / "configs" / src_name).read_text())
    pending = "PENDING_GEMMA4E2B_HPO_WINNER"
    cfg = derive_fit_config(src, slug="gemma4e2b", winners=E2B_WINNERS,
                            measured_bs=BS_E2B, pending_sentinel=pending)
    # Size the alphakl floor dims to THIS model (text_config.hidden_size
    # at the pinned revision) — the probe source inherited E4B's 4096.
    # Inert at fit time (build_two_si_pool.py territory) but kept
    # correct, the qwen35-lane convention.
    pool = dict(cfg.get("pool") or {})
    ff = dict(pool.get("fit_floor") or {})
    ff["dims"] = {k: E2B_HIDDEN for k in (ff.get("dims") or {})}
    pool["fit_floor"] = ff
    cfg["pool"] = pool
    w = E2B_WINNERS["psr_kl_all"]
    lw = E2B_WINNERS["lora_r4"]
    notes = [
        f"model: {cfg['model']} @ {str(cfg['model_revision'])[:7]} "
        "(gated — Modal hf-token secret;",
        "  STANDARD attention: scripts/modal/twosi_fits.py routes this",
        "  key to image_std, NOT the qwen D66 kernel image).",
        "pool: pool/gemma4e2b/ on the twosi-fits volume (the frozen",
        "  matched-intersection E2B pool, manifest 950211e4d2c9...,",
        "  8274 pairs/SI, + pre-built tokcache sidecars).",
        "train_kwargs.psr_kl_all: the SAFETY-lane HPO winner r1_t04",
        "  (best heldout loss of the 8-trial safety sweep; exact floats",
        f"  from its summary.json): lr {w['lr']:.6g}, clip "
        f"{w['max_grad_norm']:.6g}, wf 0.15,",
        "  T 1.0, wd 0.01, accum 16 @ bs2 = effective 32, epochs 10,",
        "  patience 3, floor_ratio_max 1.0, offline top-K=200 teacher —",
        "  a FIXED-HP TRANSFER onto si_constitution/si_irrelevant, NOT a",
        "  new HPO. psr_mse_all: no E2B HPO -> sentinel.",
        "train_kwargs.lora_r4: the E2B lora HPO winner r1_t00 (RANK 4 —",
        "  won OUTRIGHT, the matrix arm; banks as <si>__lora_r4, no",
        f"  rank-16 record needed): lr {lw['lr']:.6g}, clip "
        f"{lw['max_grad_norm']:.6g}, lora_rank 4,",
        "  wf 0.15, T 1.0, wd 0.01, accum 16 @ bs2 = effective 32, NO",
        "  gradient checkpointing (the E2B lora HPO ran bs2 without it) —",
        "  the same FIXED-HP TRANSFER shape as psr_kl_all.",
        "batch_size_by_arm: psr_kl_all bs2 = the memprobe width on the",
        "  REAL pool (safety@8192 104.4 / si_constitution@16384 106.5",
        "  GiB reserved, bs4 OOM, 0.85x141 guard) — the width the safety",
        "  winner ran. lora_r4 bs2 = the width the lora HPO itself ran",
        "  (t00 peak alloc 82.3 GiB, no ckpt). Other arms unmeasured ->",
        "  placeholder.",
        f"fit_floor dims: {E2B_HIDDEN} = text_config.hidden_size at the",
        "  pinned revision (the probe source inherited E4B's 4096).",
    ]
    name = "two_si_gemma4e2b_fits.yaml"
    return {name: (_header(name, src_name, notes, body=_E2B_HEADER_BODY,
                           pin_line=_E2B_PIN_LINE), cfg)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--llama-winners", type=Path, default=None,
                    help="arm-keyed winners json from the Llama HPO "
                         "harvest; replaces the PENDING sentinels")
    ap.add_argument("--out-dir", type=Path, default=REPO / "configs")
    ap.add_argument("--only", default=None,
                    choices=["qwen35", "gemma4e2b"],
                    help="emit just one derived lane (targeted regen — "
                         "nothing else on disk is touched)")
    args = ap.parse_args()
    winners = (json.loads(args.llama_winners.read_text())
               if args.llama_winners else None)
    if winners is not None:
        unknown = [a for a in winners if a not in FIT_ARMS]
        if unknown:
            raise SystemExit(f"--llama-winners has unknown arms: {unknown}")
        pending = [a for a in FIT_ARMS
                   if a not in winners and a not in LLAMA_LANDED_WINNERS]
        if pending:
            # Partial folds are legitimate (each cell's HPO lands on its
            # own clock); the sbatch's per-leg gate
            # (scripts/check_fit_config_folded.py) refuses any leg whose
            # OWN arms are still unfolded, so a partial fold cannot launch
            # a sentinel into a trainer.
            print(f"NOTE: partial fold — {pending} keep the "
                  f"{LLAMA_PENDING} sentinel until their HPO lands.")
    lanes = {"qwen35": build_qwen35, "gemma4e2b": build_gemma4e2b}
    builders = ([lanes[args.only]] if args.only
                else [lambda: build_all(winners), *lanes.values()])
    for build in builders:
        for name, (header, cfg) in build().items():
            emit(args.out_dir / name, header, cfg)


if __name__ == "__main__":
    main()
