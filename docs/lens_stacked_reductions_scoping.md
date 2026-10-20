# Stacked reductions inside a vectorized lens group — decision record

Branch `lens-grouped-scope` (off `main` 4133409), 2026-09-04. This document
records the decision, the numbers it rests on, the contract that replaces
bit-identity for the served hook, what changes for every consumer of a
fitted-cell server, and — as history — the scoping evidence that preceded it
(§7–§8; those sections are the memo this file used to be).

## 1. The decision (2026-09-04)

**Stacked reductions are the default served arithmetic inside a vectorized
lens group**, for every reduction op kind and every family:

| op kind (`lens_adapters.REDUCTION_OPS`) | contract class (§9) | what is stacked |
|---|---|---|
| PSR / AlphaKL gate (`relu_gated_vector`, `linear_gated_vector`) | GATED | the fp32 gate gemv `[B, K] @ [K]` |
| the block-site low-rank kinds (`lowrank_relu_gated`, `lowrank_linear_gated`) | GATED | both GEMMs — the read through A is the k-latent gate, the write through B the direction |
| LoRA rank-1 and rank-K at the o_proj site (`lora_oproj`) | LINEAR | both GEMMs — read through A, write through B |
| LoReFT (`affine_subspace`) | LINEAR | the affine `F.linear` pair over d and the back-projection over rank |

`lens_adapters._row_reduction` issues ONE `[B, K]` call for a multi-row group
under the default; `SISTEER_LENS_STACKED_REDUCTIONS=0` (or `per_row`) opts
back to the pre-landing **per-row** path — B single-row calls, bit-identical
to the per-request path by construction. That path is kept for two purposes:
it is the reference the gate's shadow compare measures stacked against, and
it reproduces a pre-landing run exactly. The env vocabulary
(`lens_adapters.parse_reduction_mode`): unset / empty / `1` / `true` / `on` /
`yes` / `all` / `stacked` = stacked; `0` / `false` / `off` / `no` / `per_row`
= per-row; **any other value refuses at import** (`AdapterError`) — a typo
must never pick a mode silently. The value is read once at import into
`la._STACKED_REDUCTIONS`; the vLLM worker is spawned with its env in place.

Elementwise ops (`lens_adapters.ELEMENTWISE_OPS`: `add_vector`,
`prefill_last_matmul`, `prefill_last_cosine_gated`) never call
`_row_reduction` and are exact in either mode; the two `prefill_last_*` ops
are prefill-only and per request by construction, so they never form a group
at all.

The reasoning: the per-step speed-up is large enough to take on `main` even
at the cost of logit changes at the bf16-ulp level, and the change should
reach every eval that runs on `main` — the peer eval lanes' included —
rather than living behind a flag. This is **option B of §8** ("stacked
everywhere, behind the contract change"); the memo's "no recommendation"
framing is withdrawn. Options C (per kind / per family allowlist), D (a
bit-identical one-launch form) and E (a B threshold) were not taken; D is
additionally dead on the evidence (§2.1: the `bmm` form is a third
arithmetic, identical to neither).

## 2. The numbers

Three measurements. The microbench (§2.1) gives per-kernel speed and the
per-element deviation on one GPU with no model; the 382416 gate run (§2.2,
tables in §7) gave the first served evidence; the served A/B (§2.3) puts both
arithmetics on one node in one session with the composition floor measured
next to the effect.

### 2.1 Microbench — Slurm 388414

`scripts/lens_stacked_microbench.py` on one H200 (torch 2.11.0+cu129, 50
iterations x 5 trials, seed 0), results at
`$SHARED/exp/lens-stacked-scope/microbench/microbench.{md,json}` (the `.md`
is mirrored to the figure mirror under `lens-stacked-microbench/`). For
each family's shapes (hidden K, per-layer o_proj widths, layer count), each
grouped kind, gaussian and sink-like ("heavy") activation profiles, and B in
{8, 32, 64, 128, 256}: median µs per hooked layer for the legacy per-request
loop / per-row / stacked, and the stacked-vs-per-row deviation on identical
bf16 inputs (fraction of output elements unequal, max ulps at the OUTPUT's own
magnitude, ulps histogram, max |diff|). 320 records.

**Speed (per hooked layer, per op).** Stacked is one launch and flat in B:
52–99 µs for the psr gemv, 46–98 µs for the o_proj LoRA pair, 56–126 µs for
the block-site low-rank / LoReFT pairs, at every B from 8 to 256. Per-row
grows linearly: ~14–38 µs per row for the psr gemv (llama, gaussian: 0.30 ms
at B=8 → 0.90 ms at B=32 → 1.69 ms at B=64 → 6.40 ms at B=256), 21–70 µs per
row for the rank-4 kinds. The legacy loop costs ~47–109 µs per row. Over a
32-layer psr object that is 54 ms of per-row hook work per step at B=64
against 3 ms stacked (the table's `hook ms/step` column, llama). `add_vector`
is identical in cost and bits everywhere (ratio 1.0, 0 unequal): it has no
reduction.

**Deviation and speedup per (family, kind).** Derived from `microbench.md`:
speedup = per-row µs / stacked µs (range over the two profiles); "max frac
unequal" = the largest fraction of output elements that differed in any
(profile, B) cell, with that cell; "first B" = the smallest B with any
unequal element, gaussian / heavy; "max ulps" is at the output's own
magnitude (the inflating unit — see §3); max |diff| is absolute, bf16.

| family | kind (o_proj class) | speedup B=32 | speedup B=64 | max frac unequal (cell) | first B unequal g / h | max ulps @output | max abs diff |
|---|---|---:|---:|---|---|---:|---:|
| llama8b (K=4096) | psr_gemv | 9.1–9.2 | 17.2–17.7 | 5.3e-6 (heavy, 256) | 256 / 256 | 32 | 2^-7 |
| llama8b | lowrank_r4 | 15.2–15.3 | 30.4–30.6 | 1.8e-5 (heavy, 32) | 8 / 8 | 32 | 2^-6 |
| llama8b | loreft_r4 | 13.8–13.9 | 27.3–27.5 | 1.9e-5 (heavy, 256) | 8 / 8 | 128 | 2^-7 |
| llama8b | lora_oproj_r1 (attn) | 15.1–15.2 | 29.0–29.4 | 8.4e-6 (heavy, 256) | 256 / 256 | 16 | 2^-7 |
| llama8b | lora_oproj_r4 (attn) | 13.2–13.3 | 26.3–26.5 | 2.4e-5 (heavy, 8) | 8 / 8 | 64 | 2^-7 |
| llama8b | add_vector | 1.0 | 1.0 | 0 | — | 0 | 0 |
| gemma4e4b (K=2560) | psr_gemv | 9.5 | 18.1–18.7 | 7.3e-6 (gaussian, 128) | 32 / 32 | 32 | 2^-7 |
| gemma4e4b | lowrank_r4 | 15.6–15.8 | 29.7–30.1 | 1.6e-5 (heavy, 256) | 32 / 8 | 32 | 2^-7 |
| gemma4e4b | loreft_r4 | 14.7–14.8 | 28.4–28.5 | 2.4e-5 (heavy, 32) | 32 / 8 | 32 | 2^-7 |
| gemma4e4b | lora_oproj_r1 (sliding) | 15.3–15.4 | 29.8–30.0 | 2.9e-5 (heavy, 8) | 32 / 8 | 64 | 2^-7 |
| gemma4e4b | lora_oproj_r1 (global) | 15.2–15.3 | 28.8–29.6 | 8.5e-6 (heavy, 256) | 256 / 256 | 16 | 2^-7 |
| gemma4e4b | lora_oproj_r4 (sliding) | 13.4–13.5 | 25.0–25.4 | **3.9e-5 (gaussian, 8)** | 8 / 8 | 32 | 2^-6 |
| gemma4e4b | lora_oproj_r4 (global) | 13.5 | 25.1–25.3 | 2.4e-5 (heavy, 32) | 8 / 8 | 128 | 2^-6 |
| gemma4e4b | add_vector | 1.0 | 1.0 | 0 | — | 0 | 0 |
| qwen35_4b (K=2560) | psr_gemv | 9.3–9.4 | 17.7–17.8 | 7.3e-6 (gaussian, 128) | 32 / 32 | 32 | 2^-7 |
| qwen35_4b | lowrank_r4 | 15.4–15.5 | 29.2–29.6 | 1.6e-5 (heavy, 256) | 32 / 8 | 32 | 2^-7 |
| qwen35_4b | loreft_r4 | 16.4–16.6 | 32.2–32.5 | 2.4e-5 (heavy, 32) | 32 / 8 | 32 | 2^-7 |
| qwen35_4b | lora_oproj_r1 (full_attn) | 15.1–15.2 | 29.4–29.7 | 8.5e-6 (heavy, 256) | 256 / 256 | 16 | 2^-7 |
| qwen35_4b | lora_oproj_r4 (full_attn) | 13.3–13.5 | 25.3–25.4 | 2.4e-5 (heavy, 32) | 8 / 8 | 128 | 2^-6 |
| qwen35_4b | add_vector | 1.0 | 1.0 | 0 | — | 0 | 0 |
| glm46vflash (K=4096) | psr_gemv | 9.1–9.2 | 17.2 | 5.3e-6 (heavy, 256) | 256 / 256 | 32 | 2^-7 |
| glm46vflash | lowrank_r4 | 15.0–15.1 | 30.3–30.4 | 1.8e-5 (heavy, 32) | 8 / 8 | 32 | 2^-6 |
| glm46vflash | loreft_r4 | 13.8 | 27.3–27.4 | 1.9e-5 (heavy, 256) | 8 / 8 | 128 | 2^-7 |
| glm46vflash | lora_oproj_r1 (attn) | 15.3 | 29.2–29.3 | 8.4e-6 (heavy, 256) | 256 / 256 | 16 | 2^-7 |
| glm46vflash | lora_oproj_r4 (attn) | 13.1–13.2 | 26.3–26.5 | 2.4e-5 (heavy, 8) | 8 / 8 | 64 | 2^-7 |
| glm46vflash | add_vector | 1.0 | 1.0 | 0 | — | 0 | 0 |
| nemotron3nano4b (K=3136) | psr_gemv | 10.0 | 19.2–19.3 | 1.0e-5 (heavy, 32) | 8 / 32 | 32 | 2^-7 |
| nemotron3nano4b | lowrank_r4 | 15.7 | 29.8–30.1 | 1.2e-5 (gaussian, 32) | 32 / 8 | 32 | 2^-7 |
| nemotron3nano4b | loreft_r4 | 14.4–14.5 | 28.0–28.1 | 2.2e-5 (heavy, 32) | 32 / 32 | 128 | 2^-7 |
| nemotron3nano4b | lora_oproj_r1 (attn) | 15.4–15.5 | 29.7–30.2 | 1.6e-5 (gaussian, 8) | 8 / 32 | 16 | 2^-6 |
| nemotron3nano4b | lora_oproj_r4 (attn) | 13.6–13.7 | 20.2–25.5 | 2.4e-5 (heavy, 256) | 8 / 32 | 32 | 2^-6 |
| nemotron3nano4b | add_vector | 1.0 | 1.0 | 0 | — | 0 | 0 |

Across all 320 records: **the fraction of unequal elements is 0 to 3.91e-5**
(the maximum: gemma4e4b `lora_oproj_r4`, sliding-window o_proj shape,
gaussian profile, B=8); **max |diff| ≤ 0.0156 = 2^-6** on every record;
**max ulps at the output's own magnitude 1–128** — 207 of the 320 records have
any unequal element, and in 137 of those the max exceeds 2 ulps (the heavy
tail §3 explains: a flip of one bf16 ulp of the EDIT, read at an output that
is small because the edit nearly cancelled the activation). The 128-ulp
records are LoReFT on llama / GLM (heavy, B=32) and Nemotron (gaussian,
B=128) and rank-4 o_proj LoRA on gemma / Qwen (heavy, B=256), each at a
fraction unequal of 7e-6 to 2e-5.

By kind and K: the **psr gemv at K=4096** (llama8b, glm46vflash) is identical
through B=128 in both profiles and unequal only at B=256; at **K=2560**
(gemma4e4b, qwen35_4b) it differs from B=32 in both profiles; at **K=3136**
(nemotron3nano4b) from B=8 (gaussian) / B=32 (heavy). **Rank-4 LoRA and
LoReFT differ from B=8** on llama / GLM in both profiles and on the other
families in at least one profile (from B=32 in the other); **rank-1 o_proj
LoRA** differs only at B=256 on llama / GLM / Qwen / gemma-global and from B=8
on Nemotron and gemma-sliding.

**The `bmm` probe (option D) is not bit-identical.** `probe_bmm_gate`
(`scripts/lens_stacked_microbench.py`) computes the psr gate gemv three ways
on the same rows — per row, stacked, and `bmm([B, 1, K], [B, K, 1])` (one
launch, M=1 per item) — and records `torch.equal` of the fp32 gate scalars
plus median µs. Its result is in the JSON only (`records[*].bmm_gate_probe`,
50 psr records; the markdown table has no bmm column): `bmm_equals_per_row`
is **False in all 50**, `bmm_equals_stacked` False in all 50, and
`stacked_equals_per_row` True in exactly the 16 llama / GLM records at
B ≤ 128 — so the batched kernel is a third reduction order, not the per-row
one, and it is also slower than stacked (39–40 µs vs 33 µs at K=4096, 22–24
vs 19 at K=2560 / 3136). There is no one-launch form with the per-row bits.

### 2.2 The 382416 gate run — stacked candidate at 75efb47

The first served evidence, from before the flag existed: the gate's
candidate at 75efb47 stacked its reductions and its in-worker shadow compare
recorded the unequal groups. Its speed, deviation and token-level tables are
kept in §7 (§7.2–§7.4) with the corrections the newer data forces. One fact
verified from the logs
(`$SHARED/exp/logs/lens-vectorize-gate_382416_{0..4}.out`, the `pure_B1=`
field of each case verdict line) and not in the memo before: **`pure_B1` was
PASS on every stacked-candidate case that reached a verdict — 26 of 26**
(llama8b 9, gemma4e4b 5, qwen35_4b 4, glm46vflash 4, nemotron3nano4b 4; the
other cases died on the reference standups before a verdict, and the
`multigroup` case has no `pure_B1` slot by design). B=1 is exact by
construction: a one-row group is one call in either mode
(`_row_reduction`'s `x.shape[0] <= 1` branch), and the A/B's B=1 sanity
(§2.3) confirms it on all five families and both cases.

### 2.3 The served A/B — Slurm array 388415, tree f375029

`scripts/lens_stacked_ab.py` per family (`scripts/sbatch/lens_stacked_ab.sbatch`,
array 0–4 = llama8b, gemma4e4b, qwen35_4b, glm46vflash, nemotron3nano4b; one
H200 each; results at `$SHARED/exp/lens-stacked-scope/ab/<model>/{report.md,
lens_stacked_ab.json}`, mirrored to the figure mirror under
`lens-stacked-ab/<model>/` — all five present). Design: three standups per case
from one tree on one node — `per_row` (flag unset at f375029, apply path
`vectorized`), `stacked` (`SISTEER_LENS_STACKED_REDUCTIONS=1`,
`vectorized+stacked`), `stacked_shadow` (the flag plus
`SISTEER_LENS_SHADOW_COMPARE=1`, `vectorized+stacked+shadow(serves per_row)`
— every group runs both arithmetics in the worker on identical rows, the
record counts the unequal groups per (kind, op, shape, positions, skip_bos,
phase, rows), and the per-row result is served, so this standup is also a
second per-row server). Production BI=0, prefix caching off, geometry 4096 x
64, `max_tokens` 64 with `ignore_eos`. At B in {8, 32}, three identical greedy
waves per standup; comparisons pooled as the fraction of requests whose full
output is identical in token ids AND every logprob ("all") or in token ids
alone ("tokens"), with the median first-divergence token among
token-differing requests. `per_row self` and `per_row x server` are the
same-server and cross-server floors of a bit-identical hook; `stacked vs
per_row` is the effect plus the cross-server floor. One B=1 singleton per
standup (stacked must equal per_row exactly). Timing: ms/step at B in
{8, 32, 64} on the per_row and stacked standups (the gate's closed-loop
sampler). Default cases: `psr` response (`si_constitution__psr_kl_all`) and
`lora_all` rank-4 response (`si_constitution__lora_r4`), real objects.

Composition is held as similar as the scheduler allows (same prompts, same
B, same node, one client), not fixed — the floor is measured, not removed;
the in-worker shadow is the confound-free reading of the same standup.

#### 2.3.1 llama8b (K=4096) — landed (the login host, raw job 388416)

B=1 sanity: stacked == per_row, both cases.

| case | B | per_row self all (tokens) | per_row x server all (tokens) | stacked self all (tokens) | **stacked vs per_row** all (tokens) | div |
|---|---:|---|---|---|---|---:|
| psr response | 8 | 1.000 (1.000) | 1.000 (1.000) | 0.333 (0.667) | **0.667 (0.833)** | 28 |
| psr response | 32 | 0.333 (0.792) | 0.438 (0.816) | 0.010 (0.615) | **0.267 (0.757)** | 28 |
| lora_all r4 response | 8 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | **0.000 (0.375)** | 42 |
| lora_all r4 response | 32 | 0.333 (0.688) | 0.413 (0.753) | 0.333 (0.750) | **0.000 (0.580)** | 28 |

Shadow (stacked vs per_row on identical rows): **psr 14368 groups compared, 0
unequal** at every group size seen (B in {1, 3, 4, 7, 8, 28, 29, 31, 32});
**lora_all 14368 compared, 6538 unequal**, 11,671 elements differed of ~1.0e9
compared (rows x groups x 4096: fraction **1.2e-5**), max |diff| 2^-7, max
ulps at the output's own magnitude 128, max |per-row value| in an unequal
group 10.9. By rows: B=3 1/32 groups unequal (1 element, 1 ulp), B=7 30/96,
B=8 1866/5952 (2253 elements), B=32 4512/5888 (9120 elements); B=1 0/2208.

Timing, ms/step (per_row → stacked): psr 32.7 → 14.3 (B=8), 48.5 → 18.7
(B=32), 55.3 → 30.8 (B=64); lora_all 30.3 → 22.7, 38.5 → 28.8, 65.4 → 29.3.

Reading: on psr the shadow finds the two arithmetics identical on every one
of 14,368 groups, so the 0.833 / 0.757 token agreement of `stacked vs
per_row` is composition on the stacked server, which did not repeat itself
either (`stacked self` 0.667 / 0.615 tokens) while the per-row servers did at
B=8 (1.000). Why the stacked server's admission differed from the per-row
servers' in this session is not established; a faster hook changing the
scheduler's step timing is a plausible mechanism, unverified. On lora_all
the difference is real: 0 of 72 requests identical in logprobs at B=8 where
both floors are 1.000, 37.5 % identical in tokens, first divergence at token
42 of 64.

#### 2.3.2 glm46vflash (K=4096) — landed (a compute node, raw job 388419)

B=1 sanity: stacked == per_row, both cases.

| case | B | per_row self all (tokens) | per_row x server all (tokens) | stacked self all (tokens) | **stacked vs per_row** all (tokens) | div |
|---|---:|---|---|---|---|---:|
| psr response | 8 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | **1.000 (1.000)** | n/a |
| psr response | 32 | 1.000 (1.000) | 1.000 (1.000) | 0.333 (0.833) | **0.667 (0.917)** | 10 |
| lora_all r4 response | 8 | 0.917 (0.917) | 0.958 (0.958) | 1.000 (1.000) | **0.000 (0.458)** | 23 |
| lora_all r4 response | 32 | 1.000 (1.000) | 0.667 (0.854) | 0.333 (0.667) | **0.000 (0.417)** | 14 |

Shadow: **psr 17880 groups compared, 0 unequal** (B in {1, 7, 8, 31, 32});
**lora_all 17920 compared, 5838 unequal**, 8,577 elements of ~1.25e9
(fraction **6.9e-6**), max |diff| 2^-5 = 0.0312 (at B=23, max |ref| 27.5),
max ulps @output 128, max |ref| 58.8. By rows: B=7 30/120, B=8 1395/7440,
B=9 10/40, B=23 23/40, B=31 69/120, B=32 4311/7400; B=1 0/2760.

Timing (per_row → stacked): psr 41.0 → 37.8, 73.1 → 31.8, 65.1 → 53.3;
lora_all 37.7 → 34.1, 56.7 → 33.5, 112.3 → 55.9. (The per_row psr B=64
reading below its B=32 one says the single-window sampler is noisy at the
±10 ms level; read the B=32 / B=64 columns together.)

Reading: psr is identical on every group and served identically at B=8; the
B=32 `stacked vs per_row` 0.917 sits with a `stacked self` of 0.833 —
composition on the stacked server again. lora_all: 0 of 72 requests identical
in logprobs at B=8 against a 0.958 floor; 45.8 % in tokens; divergence from
token 23 (B=8) / 14 (B=32).

#### 2.3.3 nemotron3nano4b (K=3136) — landed (a compute node, raw job 388415)

B=1 sanity: stacked == per_row, both cases.

| case | B | per_row self all (tokens) | per_row x server all (tokens) | stacked self all (tokens) | **stacked vs per_row** all (tokens) | div |
|---|---:|---|---|---|---|---:|
| psr response | 8 | 1.000 (1.000) | 0.917 (1.000) | 1.000 (1.000) | **0.000 (0.875)** | 47 |
| psr response | 32 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | **0.000 (0.562)** | 28 |
| lora_all r4 response | 8 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | **0.000 (1.000)** | n/a |
| lora_all r4 response | 32 | 1.000 (1.000) | 0.719 (0.927) | 1.000 (1.000) | **0.031 (0.781)** | 19 |

Shadow: **psr 18774 groups compared, 2563 unequal**, 4,800 elements of ~1.0e9
(rows x groups x 3136: fraction **4.8e-6**), max |diff| 0.25 (at B=32, in
groups whose largest per-row element is ~1e3 — the sink-like massive
activations of this residual stream; 0.25 is one sixteenth of a bf16 ulp at
that magnitude and one ulp of an edit of magnitude 32–64), max ulps @output
128, max |ref| 1000. By rows: B=2 0/42, B=6 4/42, B=7 10/84, B=8 692/7812
(938 elements), B=31 48/126, B=32 1809/7812 (3747 elements); B=1 0/2856.
**lora_all 1792 compared, 658 unequal**, 1,023 elements of ~9.6e7 (fraction
**1.1e-5**), max |diff| 2^-4, max ulps @output 128, max |ref| 96.5; B=7
0/12, B=8 183/744, B=9 0/4, B=23 2/4, B=31 9/12, B=32 464/740; B=1 0/276.

Timing (per_row → stacked): psr 40.9 → 33.0, 70.0 → 32.4, 111.0 → 37.0;
lora_all 24.8 → 23.4, 32.4 → 24.7, 38.8 → 30.5.

Reading: every stacked server here repeats itself exactly (`stacked self`
1.000 in all four rows), and both per-row floors are 0.92–1.00, so the
`stacked vs per_row` rows are the arithmetic alone: psr at B=8 changes the
logprobs of every request and the tokens of 12.5 % (first divergence at token
47 of 64), at B=32 the tokens of 44 %; lora_all at B=8 changes logprobs on
every request and tokens on none, at B=32 tokens on 22 %.

#### 2.3.4 gemma4e4b (K=2560) — landed (the login host, raw job 388417)

B=1 sanity: stacked == per_row, both cases.

| case | B | per_row self all (tokens) | per_row x server all (tokens) | stacked self all (tokens) | **stacked vs per_row** all (tokens) | div |
|---|---:|---|---|---|---|---:|
| psr response | 8 | 0.833 (1.000) | 0.889 (1.000) | 0.750 (1.000) | **0.000 (0.750)** | 32 |
| psr response | 32 | 0.938 (1.000) | 0.969 (1.000) | 0.958 (0.958) | **0.000 (0.646)** | 27 |
| lora_all r4 response | 8 | 0.833 (0.917) | 0.917 (0.958) | 1.000 (1.000) | **0.000 (0.792)** | 19 |
| lora_all r4 response | 32 | 0.938 (0.958) | 0.955 (0.972) | 1.000 (1.000) | **0.000 (0.677)** | 23 |

Shadow: **psr 18774 groups compared, 1443 unequal**, 1,667 elements of
~8.2e8 (rows x groups x 2560: fraction **2.0e-6**), max |diff| 2^-6, max
ulps @output 64, max |ref| 255. By rows: B=2 2/42 (the smallest group size
with any inequality in the whole A/B), B=6 0/42, B=7 2/84, B=8 341/7812 (352
elements), B=31 9/126, B=32 1089/7812 (1302 elements); B=1 0/2856.
**lora_all 18774 compared, 2378 unequal**, 2,757 elements of ~8.2e8
(fraction **3.4e-6**), max |diff| 2^-6, max ulps @output 128, max |ref| 262.
By rows: B=2 1/42, B=3 0/84, B=7 12/126, B=8 417/7812 (432 elements), B=29
26/84, B=30 14/42, B=32 1908/7812 (2272 elements); B=1 0/2772.

Timing (per_row → stacked): psr 34.4 → 30.0, 49.7 → 31.8, 76.1 → 33.4;
lora_all 36.9 → 29.2, 62.3 → 45.3, 79.5 → 36.8.

Reading: the per-row floors on gemma are 1.000 in tokens on psr at both B
(0.83–0.97 counting logprobs); stacked vs per_row is 0.000 in logprobs on
every row and 0.65–0.79 in tokens — the arithmetic, with the shadow unequal
from B=2 up. The stacked servers repeat themselves (1.000 tokens on three
of four rows), so this is not composition.

#### 2.3.5 qwen35_4b (K=2560) — landed (a compute node, raw job 388418)

B=1 sanity: stacked == per_row, both cases.

| case | B | per_row self all (tokens) | per_row x server all (tokens) | stacked self all (tokens) | **stacked vs per_row** all (tokens) | div |
|---|---:|---|---|---|---|---:|
| psr response | 8 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | **0.000 (0.500)** | 35 |
| psr response | 32 | 1.000 (1.000) | 0.667 (0.917) | 0.333 (0.812) | **0.000 (0.708)** | 8 |
| lora_all r4 response | 8 | 1.000 (1.000) | 1.000 (1.000) | 1.000 (1.000) | **0.000 (0.875)** | 57 |
| lora_all r4 response | 32 | 1.000 (1.000) | 0.667 (0.958) | 0.333 (0.854) | **0.000 (0.823)** | 50 |

Shadow: **psr 14336 groups compared, 2793 unequal**, 5,255 elements of
~6.2e8 (fraction **8.4e-6**), max |diff| 2^-8, max ulps @output 128, max
|ref| 33.2. By rows: B=4 3/32, B=7 9/96, B=8 807/5952 (1086 elements), B=28
15/32, B=31 29/96, B=32 1930/5920 (4051 elements); B=1 0/2208.
**lora_all 3584 compared, 1368 unequal**, 2,102 elements of ~1.6e8
(fraction **1.35e-5**, the largest of the ten A/B cases), max |diff| 2^-8,
max ulps @output 128, max |ref| 19.6. By rows: B=4 0/8, B=7 9/24, B=8
405/1488 (459 elements), B=28 5/8, B=31 15/24, B=32 934/1480 (1598
elements); B=1 0/552.

Timing (per_row → stacked): psr 34.3 → 33.2, 75.3 → 51.0, 75.9 → 50.3;
lora_all 28.9 → 34.0, 29.7 → 27.2, 40.3 → 34.3. (The lora_all B=8 reading
has stacked slower by 5 ms; the B=32 / B=64 readings are faster — the
single-window sampler's noise band, as on GLM.)

Reading: psr at B=8 is the cleanest cell in the A/B — every floor 1.000 in
both units, the stacked server repeating itself exactly — and stacked vs
per_row is 0.500 in tokens, 0.000 in logprobs: half the requests change
tokens by the 35th of 64. That is the largest token-level effect at B=8 on
any family. lora_all changes logprobs on every request and tokens on 12.5 %
(B=8) / 18 % (B=32).

#### 2.3.6 What the A/B says, plainly (all five families)

All five array tasks COMPLETED (sacct: exit 0:0, 24–38 min each). The
token-level result in one table — stacked vs per_row agreement in TOKENS
with the cross-server per-row floor in brackets, the same in tokens AND
logprobs ("all"), and the median first token divergence — read from the five
`report.md` files:

| family (K) | case | B | stacked vs per_row, tokens [floor] | all [floor] | div |
|---|---|---:|---|---|---:|
| llama8b (4096) | psr | 8 | 0.833 [1.000] | 0.667 [1.000] | 28 |
| llama8b | psr | 32 | 0.757 [0.816] | 0.267 [0.438] | 28 |
| llama8b | lora_all r4 | 8 | **0.375 [1.000]** | 0.000 [1.000] | 42 |
| llama8b | lora_all r4 | 32 | 0.580 [0.753] | 0.000 [0.413] | 28 |
| gemma4e4b (2560) | psr | 8 | **0.750 [1.000]** | 0.000 [0.889] | 32 |
| gemma4e4b | psr | 32 | 0.646 [1.000] | 0.000 [0.969] | 27 |
| gemma4e4b | lora_all r4 | 8 | 0.792 [0.958] | 0.000 [0.917] | 19 |
| gemma4e4b | lora_all r4 | 32 | 0.677 [0.972] | 0.000 [0.955] | 23 |
| qwen35_4b (2560) | psr | 8 | **0.500 [1.000]** | 0.000 [1.000] | 35 |
| qwen35_4b | psr | 32 | 0.708 [0.917] | 0.000 [0.667] | 8 |
| qwen35_4b | lora_all r4 | 8 | 0.875 [1.000] | 0.000 [1.000] | 57 |
| qwen35_4b | lora_all r4 | 32 | 0.823 [0.958] | 0.000 [0.667] | 50 |
| glm46vflash (4096) | psr | 8 | 1.000 [1.000] | 1.000 [1.000] | n/a |
| glm46vflash | psr | 32 | 0.917 [1.000] | 0.667 [1.000] | 10 |
| glm46vflash | lora_all r4 | 8 | **0.458 [0.958]** | 0.000 [0.958] | 23 |
| glm46vflash | lora_all r4 | 32 | 0.417 [0.854] | 0.000 [0.667] | 14 |
| nemotron3nano4b (3136) | psr | 8 | 0.875 [1.000] | 0.000 [0.917] | 47 |
| nemotron3nano4b | psr | 32 | 0.562 [1.000] | 0.000 [1.000] | 28 |
| nemotron3nano4b | lora_all r4 | 8 | 1.000 [1.000] | 0.000 [1.000] | n/a |
| nemotron3nano4b | lora_all r4 | 32 | 0.781 [0.927] | 0.031 [0.719] | 19 |

* **B=1 is exact.** Every `:B1` shadow row on every family and case reads 0
  unequal (2208–2856 psr groups, 276–2772 lora groups per family), and the
  B=1 sanity is True on all ten (family, case) pairs. The contract's hard
  B=1 verdicts (§3) lose nothing.
* **At B ≥ 8 the deviation reaches greedy tokens, well above the
  composition floor, on every kind and family whose shadow finds unequal
  groups** — eight of the ten cases. At B=8, where the cross-server per-row
  floor is 0.958–1.000 in tokens, stacked vs per_row on those eight is
  0.375–0.875 in tokens on seven and 1.000 on the eighth (Nemotron lora_all,
  whose logprobs still all differ), and 0.000 in logprobs on all eight; at
  B=32, 0.42–0.82 in tokens against floors of 0.75–1.00. Median first
  divergence: token 8–57 of 64. The two cases whose shadow finds NO unequal
  group (psr at K=4096:
  llama, GLM) show only the composition noise of the stacked server itself
  (its `stacked self` is the same fraction as its `stacked vs per_row`).
* **So: a stacked-served output is a different greedy sample from
  (approximately) the same distribution.** In kind this is the perturbation
  batch composition already injects at `VLLM_BATCH_INVARIANT=0`; in size and
  presence it is larger — a deterministic offset from the per-request
  arithmetic on every step, on every K != 4096 psr and every rank-4 / LoReFT
  group at any K — where the composition floor at B=8 is essentially zero.
  Token-exact reproducibility of a pre-landing fitted-cell run at B > 1 is
  lost under the default and recovered with
  `SISTEER_LENS_STACKED_REDUCTIONS=0`. Judge-scored semantic metrics
  (StrongREJECT scores, refusal rates, capability accuracies) are expected to
  be unaffected — a one-ulp perturbation class the evals already live with
  from composition — but **that expectation is untested here**: the A/B
  measured token identity, not scores.
* **The element-level rate is an order of magnitude under the contract's
  bound.** Derived from the by-rows tables (elements compared = sum over keys
  of rows x groups x the site's width — 4096 on llama / GLM, 2560 on gemma /
  Qwen, 3136 on Nemotron): llama lora_all 1.2e-5; gemma psr 2.0e-6, lora_all
  3.4e-6; Qwen psr 8.4e-6, lora_all 1.35e-5; GLM lora_all 6.9e-6; Nemotron
  psr 4.8e-6, lora_all 1.1e-5; the two K=4096 psr cases 0. Largest 1.35e-5;
  the bound in §3 is 1e-4. `max ulps` at the output's own magnitude is 128
  on seven of the eight unequal cases and 64 on the eighth (gemma psr).
* **Same-node speedup, per step, whole server** (per_row / stacked ms/step):
  1.1–2.6x at B=32 and 1.2–3.0x at B=64 over the ten pairs — psr B=32:
  llama 2.6x, GLM 2.3x, Nemotron 2.2x, gemma 1.6x, Qwen 1.5x; psr B=64:
  llama 1.8x, GLM 1.2x, Nemotron 3.0x, gemma 2.3x, Qwen 1.5x; lora_all B=64:
  llama 2.2x, GLM 2.0x, Nemotron 1.3x, gemma 2.2x, Qwen 1.2x. Lower than the
  2–3.5x the cross-run comparison in §7.2 suggested (different nodes and
  days there), and bounded by the passthrough floor: the stacked step at
  B=64 is 30–56 ms on every family, within 10–30 ms of the floor.
* **What the A/B did not record.** Its tree (f375029) predates
  `ShadowCompareRecord.max_ulps_edit` and `n_elems_compared`; the shadow
  tables carry `max ulps` at the output's own magnitude only, and the
  element fractions above are derived, not read. The 4-ulp bound at
  `max(|output|, |edit|)` in §3 is therefore mechanism-derived here (§3
  explains why it predicts ≤ 2) and is measured for the first time by the
  next gate run.

## 3. The contract

The gate (`scripts/lens_vectorize_gate.py`) replaces "any unequal group is
exit 2" with THE REDUCTION CONTRACT, applied by its pure `reduction_tolerance`
inside the `shadow_compare` check and stated in the report header and the
exit-code footer (`REDUCTION_CONTRACT`, with `REDUCTION_ULPS_EDIT_TOL = 4.0`,
`REDUCTION_FRAC_TOL = 1e-4`, `REDUCTION_GROUP_FRAC_TOL = 2e-3`,
`REDUCTION_ROW_FRAC_TOL = 2e-2`, and since 2026-09-06 `REDUCTION_GATE_C =
16` on the unit `2^-24` — §9), in one sentence:

> reduction ops (stacked mode): bf16-kernel-variance equivalence to the
> per-row path, bounded in concentration for every reduction op and, per op
> class, on the edit or at the gate — over all counted reduction-op
> comparisons of a case the fraction of unequal ELEMENTS is <= 1e-4, in any
> single group <= 2e-3 and in any single row <= 2e-2 (the two concentration
> bounds are 10x the measured maxima, not mechanism); LINEAR reduction ops
> (`affine_subspace`, `lora_oproj`): every unequal element's stacked edit is
> within 4 bf16 ulps of the per-row edit beyond one bf16 ulp of the per-row
> output (|applied − ref| <= 4 ulp(|ref − base|) + ulp(|ref|)); GATED
> reduction ops (`relu_gated_vector`, `linear_gated_vector`,
> `lowrank_relu_gated`, `lowrank_linear_gated`): the stacked gate is within
> 16 × 2^-24 of the per-row gate in units of the reduction's own magnitude,
> per gate element (|g_stacked − g_per_row| <= 16 · 2^-24 · Σ_i |h_i w_i|),
> and a ReLU flip only when the side that fired is within that same bound —
> the edit-ulp unit is recorded there but is not a rule; elementwise ops, and
> EVERY op under the per-row mode: exact (n_unequal == 0). Any modified-flag
> disagreement, shadow error, record error, an unequal reduction-op group
> whose record lacks the tolerance fields, an unequal gated group whose record
> lacks gate stats, or a gate_error is a FAIL — the tolerance is for kernel
> variance, never for a defect or for missing evidence.

The paragraphs below are the 2026-09-04 derivation of the edit-ulp bound and
the concentration bounds. They stand for the LINEAR class; §9 records why the
edit-ulp unit does not transfer to the GATED class and the rule that replaced
it there.

Review round 13 (2026-09-04) found the first form of this contract — a
case-wide element fraction plus a per-element bound in ulps of
max(|output|, |edit|) — had a hole: a grouped-only defect that drops or halves
ONE ROW's edit in one group passed it (3,136 wrong elements of 7.8e8 is 4e-6,
and with |output| >> |edit| a fully dropped edit reads as one ulp of the
output). The modified flag is per group, not per row, the shadow serves
per-row, B=1 cannot exercise a grouped-only bug and B > 1 cross-server is
evidence — so that gate could have certified a worker dropping a row's edit
in production. The contract above is the fix: the per-element bound is on the
EDIT, and two concentration bounds sit next to the case-wide fraction.

A shadow key's op token (`kind:op[+op]:shape:positions:skipbosN:phase:B<rows>`)
is classified through `lens_adapters.reduction_op_class` (`shadow_key_op_class`;
until 2026-09-06 through `op_reduces_in_group`, which still exists as the
two-way split); a key whose op the adapters do not classify is
missing evidence — FAIL, never a default class. The gate SETS the mode on
every candidate-tree standup (`standup_env`) rather than inheriting it.

That rule found a bookkeeping defect on 2026-09-04, the first time it ran on
a multi-cell registry: the llama8b `multigroup` case (psr + lora_r4 + a second
psr) FAILED on eight keys of the form `psr:-:-:-:-:-:B6` / `lora_all:-:-:-:-:-:B2`
while its 20192 comparisons sat an order of magnitude inside every bound.
`site_layers` installs the registry-wide UNION of sites x layers and a
forward's groups are built once for every hook, so at the o_proj hook (installed
for the LoRA cell) the psr groups are applied with NO op — `ops_for` is `()`,
the identity on both paths — and the shadow keyed them with placeholder dashes
for every op axis. Single-cell cases never see this (one cell + its scale-0 twin
share the same ops, so every hooked point has ops), and the dash keys date from
the shadow compare's introduction on main: the defect was latent, and it is the
classify-every-key loop above — mode-independent, run before the "exact" early
return — that exposed it, not the stacked mode's ulp-level differences (a
per_row-mode multigroup run fails identically). The fix is at emission
(`lens_worker.shadow_key` refuses empty ops; `_shadow_block` / `_shadow_oproj`
record such a group as UNSERVED — `n_unserved_groups`, per `kind@site` — never
under a key), and a side that modifies rows it serves no op at counts as a
modified-flag disagreement, which fails in every mode. The gate's rule is
unchanged: an op-less key is still a FAIL.

**Why the per-element unit.** Every reduction op computes its edit in fp32
and adds it to the bf16 stream with one cast: `out + add.to(hidden.dtype)`
for the gated and low-rank block-site ops, `out + (eff * delta).to(hidden.
dtype)` for LoReFT, `out + (eff * delta).to(oproj_out.dtype)` at the o_proj
site (`lens_adapters._apply_one`, `apply_oproj_ops`). A different summation
order inside the stacked GEMM moves the fp32 reduction by a few fp32 ulps;
that matters only where the fp32 value sat at a bf16 rounding boundary — of
the edit (the cast) or of the sum (the add) — and then moves the output by
one bf16 ulp of the edit or one of the sum. So the deviation is measured ON
THE EDIT: with `base` the unsteered rows, `ref` the per-row output and
`applied` the stacked output, `lens_adapters.edit_excess_ulps` reads
`max(0, |applied − ref| − ulp(|ref|)) / ulp(|ref − base|)` — the excess over
one rounding of the sum, in ulps of the per-row edit. Kernel variance reads
<= ~1 (the gate allows 4); a dropped edit reads `(|edit| − ulp(|out|)) /
ulp(|edit|)`, >= 60 for any edit larger than two output ulps; a halved edit
about half that; a misplaced edit the whole misplaced edit. Measured in ulps
of the OUTPUT alone (the record's informational `max_ulps`) one legitimate
flip reads as tens of ulps wherever the edit nearly cancels the activation —
the microbench's 128-ulp tail, the A/B shadow's 64–128 — which is why the
first form of the contract reached for `max(|output|, |edit|)`; that unit
hid the dropped-edit case instead. The worker records the new unit as
`max_ulps_edit` (`lens_worker._unequal_stats`, `ShadowCompareRecord`,
overall and per key).

**Why the concentration bounds, and where their numbers come from.** Where
the edit is within ONE ulp of the output a dropped edit is indistinguishable
per element from a flipped sum rounding — both are the output moved by one
ulp — and invisible to the case-wide fraction. What distinguishes them is
concentration: kernel variance touches a few elements per 10^5, spread over
rows and groups; a dropped, scaled or misplaced row edit touches nearly every
element of ITS row and 1/B of its group. The worker therefore also records
`max_frac_row` (the largest fraction of unequal elements in any one row of a
group) and `frac_group` (unequal over the group's elements), and the gate
bounds the maxima over the case at `REDUCTION_ROW_FRAC_TOL = 2e-2` and
`REDUCTION_GROUP_FRAC_TOL = 2e-3`. These two are MEASURED, not mechanism —
10x the largest values in the data: over the 1600 microbench trials (§2.1;
812 with any inequality) the largest per-group fraction was 9.8e-5 (2
elements in a 20,480-element gemma o_proj rank-1 group at B=8), and no trial
had more than 5 unequal elements that could sit in one row (an upper bound
from `n_unequal − n_rows_unequal + 1`; 739 of 812 trials at most 1, 62 at
most 2, 7 at most 3, 4 at most 4–5), i.e. <= 5/2560 = 2.0e-3 of a row; the
A/B by-rows tables (§2.3) add one flip in a 2-row group at d=2560 (1.95e-4
of the group, the largest per-group fraction there) and 1–2.5 elements per
unequal group on average. The first gate run under this contract records
both maxima per key; if a measured record exceeds a bound it is reported,
never loosened silently.

**What stays hard and exact.** The B=1 verdicts — `pure_B1` (reference vs
candidate, one request per server), `controls_B1` (passthrough and scale-0
singletons unsteered bit for bit), `steering_bites_B1` (the steered
singleton differs from the passthrough) — for every kind, in either mode: a
one-row group is one call. Elementwise ops at any B. Every op under the
per-row mode. The apply-path attestation must read the mode the standup
asked for (`vectorized+stacked` / `vectorized+per_row`, with the
`+shadow(serves per_row)` suffix under the shadow compare).

**A planted bug fails the tolerance by orders of magnitude.**
`tests/test_lens_stacked_reductions.py::TestPlantedBugs` installs a fault in
the GROUPED path only (`_row_reduction` on a multi-row group; the per-row
reference slices untouched) and drives a mixed decode batch through a
shadow-compare worker in the stacked mode. Whole-group faults — a sign flip
of the edit, a row shift, a permuted operand read, a bf16-precision read —
fail with a case-wide fraction >= 0.1 against 1e-4. Row-level faults — one
row's edit dropped, one row's edit halved, two rows' edits swapped — fail on
the real worker record through `max_frac_row` (~1 against 2e-2) and, in a
second construction built to defeat the first form of the contract (a
32 x 4096 bf16 group with |output| ~1000 and edits of one output ulp, so
every element stays within 4 output ulps and the case-wide fraction is
5e-6 against 1e-4), still fail — the dropped and halved row through the row
and group concentration bounds, the swapped rows additionally through the
edit bound (an 8-output-ulp difference is 128 ulps of the 4-unit edit) — in
`reduction_tolerance`, `shadow_check` and `shadow_multigroup_check` alike. A
single element of an elementwise op moved by 1.0 fails the exact class; the
unperturbed stacked run passes with zero unequal groups; the nemotron B=32
record (§2.3.3, 3747 elements of 7.84e8 with 2 elements in a row and 3 in a
group) passes. The tolerance is a kernel-variance band, not a correctness
allowance.

## 4. What changes for consumers

* **Every fitted-cell eval on `main` gets stacked reductions** the moment its
  server is built from a tree at or after the landing — the served-family eval
  waves, BoN, capability, and the Petri agent-loop control alike. Nothing has
  to be set; the default is the change.
* **Served logits at B > 1 may differ from pre-landing runs at the bf16-ulp
  level on the reduction kinds** (psr / alphakl gates, LoRA rank-1 and
  rank-K, the low-rank block-site kinds, LoReFT). Same perturbation class as
  batch composition at `VLLM_BATCH_INVARIANT=0`, present on every step
  instead of per admission; the token-level size is §2.3.6. B=1 and the
  elementwise kinds (`vector_all`, `antidote`, `alphasteer`, the cosine-gated
  kind) are unchanged, bit for bit.
* **A store's arithmetic is attested, in four places.** (1) The worker's
  install report (`sisteer_install_steering`, `lens_worker.
  install_report_payload`) and (2) its scheduler-config report
  (`sisteer_scheduler_config`) both carry `lens_reduction_mode` (`stacked` |
  `per_row`; `lens_worker.LENS_REDUCTION_MODE_KEY`, the one
  `la.reduction_mode()` value read once at the worker's import) — so
  `steering_attestation.json` carries it under BOTH its top-level
  `worker_reports[*]` and `scheduler_geometry.worker_reports[*]`; the frozen
  launcher validates only its own expected keys and copies the rest through.
  `lens_worker.attested_reduction_mode(attestation)` returns the mode only
  when every report in both lists carries it and they all agree, else `None`
  with the reason. Both standup helpers read it through that one
  strict reader and publish it: `scripts/capctrl_serving_up.py`
  (`attested_reduction_mode_at`) writes `lens_reduction_mode` and
  `lens_reduction_mode_error` into `endpoint.json` and hands the attested
  mode to the RoutingProxy as its launch attestation;
  `scripts/lens_gate_standup.py` records the same pair in its endpoint. The
  proxy's `/v1/models` `petri_proxy` identity block carries
  `lens_reduction_mode` ONLY when the launch attestation supplied one
  (`proxy.py`, `LENS_REDUCTION_MODE_KEY`, pinned equal to the worker's key by
  `tests/test_lens_reduction_mode_attestation.py`; never defaulted there) —
  **absent means a pre-landing or unattested server**, and a consumer that
  finds no mode treats the server as per-row (the capctrl standup's docstring
  states that convention; the worker's comment names the peer driver that
  reads it). The apply path attests `vectorized+stacked` /
  `vectorized+per_row`.
* **Exact reproduction of an older run:** `SISTEER_LENS_STACKED_REDUCTIONS=0`
  in the server's environment. The per-row path is byte-for-byte the
  pre-landing call sequence.
* **Identity.** `intervention_hash` does NOT carry the mode — it keys the
  fitted object, and folding the arithmetic in would re-key every store. The
  served `RunIdentity` already separates before from after through
  `tree_sha` and the stamped `run_id`; the mode itself is read from the
  attestation / `/v1/models`. **Recommendation: record
  `lens_reduction_mode` from `/v1/models` (or the attestation file) as the
  provenance of every store generated from a fitted-cell server from now
  on,** next to `tree_sha`.

## 5. Rollout note for the peer eval lanes

* **What to expect.** New fitted-cell servers attest `vectorized+stacked`
  and `lens_reduction_mode: stacked`; steered decode at B ≥ 8 is 1.2–3x
  faster per step (§2.3.6), and at B > 1 the greedy continuations of psr (K
  != 4096), rank-4 LoRA and LoReFT cells are not token-identical to a
  pre-landing server's — a different sample from the same class of
  perturbation the composition noise already produces. Scores are expected
  to hold; that has not been measured.
* **To pin exactness** (a rerun that must match an old store token for
  token, a bisection): `SISTEER_LENS_STACKED_REDUCTIONS=0` on the server.
  Verify with `/v1/models` → `petri_proxy.lens_reduction_mode == "per_row"`
  (or the attestation's worker reports). Any other value than the vocabulary
  in §1 refuses at import — the server will not start.
* **How to read the mode:** `GET /v1/models` on the proxy; the `petri_proxy`
  identity block carries `lens_reduction_mode` when the attestation supplied
  one. A block without the key is a server built from a pre-landing tree
  (per-row arithmetic). The attestation file next to the server log carries
  the same fact per worker.
* **Running campaigns keep their frozen shas.** The frozen Petri surfaces —
  `vllm_server.py`, `scripts/petri_steered_smoke.py` and its sbatch,
  `scripts/modal/petri_agentloop_control.py` — are untouched by this landing
  (unchanged against `main` at the time of writing); `proxy.py`'s only change
  is the additive `lens_reduction_mode` key in the `/v1/models` identity
  block — the key constant, one line storing the `launch_attestation`
  mapping the constructor already took, and the emission guarded on the key's
  presence: 18 inserted lines, no deletions (verified against the `main`
  diff). A campaign pinned to an earlier sha
  keeps per-row arithmetic; it changes only when its server is rebuilt from a
  later tree.
* **The Petri smoke's PASS criteria are unaffected** — repeat-determinism of
  the SAME batch, the scale-0 no-op, steering-bites, direct selection —
  because stacked reductions are deterministic for a fixed batch composition
  (the A/B's `stacked self` rows read 1.000 in tokens wherever the per-row
  servers repeated too — every Nemotron row, gemma lora_all, Qwen and llama
  at B=8; where they read lower the per-row servers' own `self` rows drop
  with them, the composition noise of that session); scale 0 returns the same
  tensor before any reduction runs (`apply_hook_ops`, `apply_oproj_ops`); B=1
  is exact.
* **What has and has not been run.** The microbench (388414) and the served
  A/B (388415, all five families COMPLETED) are the measurements above. The
  user runs the CPU gate, the Petri smoke and the five-family GPU gate
  (`scripts/lens_vectorize_gate.py` under the §3 contract) after review;
  nothing in this document claims those passed.

## 6. How the numbers were produced

The measurement plan the branch shipped, as run (388414 microbench, 388415
A/B array; the gate run 382416 predates the branch):

1. **The flag** — `SISTEER_LENS_STACKED_REDUCTIONS`
   (`lens_adapters.STACKED_REDUCTIONS_ENV`), read once at import into
   `la._STACKED_REDUCTIONS`; `_row_reduction` issues one stacked call for a
   multi-row group under `stacked` and B single-row calls under `per_row`.
   At f375029 (the tree the A/B ran) the default was per-row and `=1` turned
   stacking on; the landing flips the default (§1) and keeps the vocabulary.
   The worker attests the mode in its apply path, `lens_gate_standup`
   records it in the endpoint record, and the gate's `standup_env` scrubs
   the variable from every standup it launches so an inherited value cannot
   turn a gate standup into the other mode (the landing's gate sets the mode
   explicitly per standup instead). `tests/test_lens_stacked_reductions.py`
   is the test home — `TestModeVocabulary`, `TestStackedDefault`,
   `TestPerRowOptOut`, `TestSingleRowIsExactInBothModes`,
   `TestReductionOpClassification`, `TestUlps`, `TestShadowRecordFields`,
   `TestPlantedBugs`, `TestPlumbing`: the default and the env vocabulary
   (unknown value refuses at import), one stacked call under `stacked` and
   one call per row under `per_row` for every reduction kind, equal values
   on exact integers, one row = one call either way, the apply-path strings,
   the `REDUCTION_OPS` / `ELEMENTWISE_OPS` classification against the calls
   the adapter actually makes, the ulp reader and its edit-scale unit, the
   shadow record's fields, the planted bugs of §3, and the gate / A/B /
   microbench env handling.
2. **Ulps in the shadow record** — `lens_adapters.ulp_distance` and the
   `ShadowCompareRecord` fields `max_ulps`, `max_abs_ref`,
   `n_unequal_elems` (f375029) plus `max_ulps_edit` (the landing;
   `lens_adapters.edit_scale`), overall and per key, filled by the two
   shadow sites. A stacked+shadow standup yields the deviation per (kind,
   op, shape, positions, rows) in the activation's own resolution,
   confound-free.
3. **`scripts/lens_stacked_microbench.py`** — one GPU, no model: for each
   family's shapes and each grouped kind (psr gemv, low-rank rank 4, LoReFT
   rank 4, add_vector control, o_proj LoRA rank 1 / rank 4 per o_proj shape
   class), at B in {8, 32, 64, 128, 256}: median µs per hooked layer for
   legacy / per_row / stacked (x layers ⇒ ms of hook work per step), and
   stacked-vs-per_row deviation on identical bf16 inputs (fraction of
   elements and rows differing, max |diff|, ulps histogram ≤1 / ≤2 / >2, the
   fp32 gate scalar's own ulps), gaussian and heavy profiles, plus the
   `bmm` gate probe. Writes JSON + markdown + one PNG per model.
   `scripts/sbatch/lens_stacked_microbench.sbatch` (1 GPU, high, 30 min,
   `--exclude=<node>`).
4. **`scripts/lens_stacked_ab.py`** — §2.3's design.
   `scripts/sbatch/lens_stacked_ab.sbatch` (array 0–4, 1 GPU each, high,
   1 h, `--exclude=<node>`, positional sha, no `--export=`).

Submission (from the grading host, after the branch is pushed):

```bash
# 1. The grading host: the cluster mirror at this branch's tip
<repo-mirror tool> <checkout name>
SHA=$(git -C <the branch's checkout> rev-parse HEAD)
# 2. login node: the commit-addressed checkout (the gate's pattern), then submit from it
ssh <login-host> bash -s -- "$SHA" <<'SUBMIT'
set -e; SHA=$1; S=<the cluster's shared root>; D=$S/<checkout root>/$SHA
[ -d "$D/src/sisteer" ] || { git clone -q --no-checkout "$S/repos/constitutional-adapters" "$D" && git -C "$D" checkout -q "$SHA"; }
cd "$D"
sbatch --parsable scripts/sbatch/lens_stacked_microbench.sbatch "$SHA"     # 1 GPU, ~10 min
sbatch --parsable scripts/sbatch/lens_stacked_ab.sbatch "$SHA"             # array 0-4, 1 GPU each, <= 1 h
SUBMIT
```

(Both sbatch files also create that checkout themselves from the mirror if
it is absent. Results land under
`$SHARED/exp/lens-stacked-scope/{microbench,ab/<model>}/`; `report.md` /
`microbench.md`, the JSON, the PNGs and a `RESULT` line are mirrored to
`$FIGS/lens-stacked-microbench/` and `$FIGS/lens-stacked-ab/<model>/` on the
grading host. Both jobs go out at `high` per the standing default: 6 GPUs of
the 16-GPU budget for the two submissions together.)

## 7. History — the scoping evidence (382416 stacked vs 384162 per-row gate runs)

This section is the scoping memo's evidence base, kept with corrections
where §2 contradicts it. The question it answered: how much faster fully
stacked reductions would be, and how large the deviation is — numerically,
and whether it changes served tokens beyond the noise floor production
already accepts at `VLLM_BATCH_INVARIANT=0` (exact replay dropped
2026-08-27).

### 7.1 Evidence base

Two GPU runs of the same gate (`scripts/lens_vectorize_gate.py`) against the
same reference (the base commit 09be969, the pre-vectorize per-request loop):

| run | candidate | arithmetic inside a group | array / per-task jobs | nodes | what survives |
|---|---|---|---|---|---|
| 382416 (2026-09-04 ~09:00–11:10Z) | 75efb47 | **stacked** (one `[B,K]` call), with the in-worker shadow compare against the per-row legacy path | 382476 llama8b, 382479 gemma4e4b, 382480 qwen35_4b, 382482 glm46vflash, 382483 nemotron3nano4b | five compute nodes | the `.out` logs (`$SHARED/exp/logs/lens-vectorize-gate_382416_{0..4}.out`): report tables, per-case verdict lines (every one `pure_B1=PASS`, §2.2), shadow totals + first unequal group; the per-key JSON was **overwritten** by the re-gate |
| 384162 (re-gate, ~12:15–15:40Z) | aa2b8a1 | **per row** (what `main` served before the landing) | 384164 llama8b, 384165 gemma4e4b, 384166 qwen35_4b, 384167 glm46vflash, 384162 nemotron3nano4b | four compute nodes (one ran two tasks), three of them also used by 382416 | `report.md` + the full `lens_vectorize_gate.json` per model (timing, shadow `by_key`, every raw signature) |

So: the **legacy reference** timing exists in both runs; the **stacked**
candidate timing only in the 382416 logs; the **per-row** candidate timing only
in the re-gate. The two candidates were never on the same node at the same
time — the ratios below compare runs hours apart on different nodes (the
passthrough floors differ by up to 2x between runs at the same B). The A/B
(§2.3) put both on one GPU.

The per-group-size shadow records of 382416 (`by_key`: which group row
counts were unequal) are quoted from the triage made while the JSON still
existed; the per-key detail cannot be re-read today. The re-gate's
`by_key` (0 unequal everywhere) still gives the group-size distribution a
gate wave produces (§7.3.3).

### 7.2 Speed (cross-run)

ms per decode step from the gate's closed-loop sampler (`max_tokens` 128,
45 s window, B concurrent clients), production BI=0, prefix caching off,
4096 x 64 geometry. "hook-only" = ms/step minus that run's own passthrough
floor at the same B (the lens server with no steering). `n/a` = the 382416
reference standup died before the candidate stood up (absent-cell deaths of
the base commit; fixed in the re-gate), so no stacked number exists for that
case.

#### 7.2.1 Cases with all three arms

| model | case | B | legacy ref ms/step (382416 / 384162) | per-row ms/step | stacked ms/step | per-row/legacy | stacked/legacy | per-row / stacked | hook-only ms: legacy / per-row / stacked | hook-only per-row/stacked |
|---|---|---:|---|---:|---:|---:|---:|---:|---|---:|
| llama8b | psr response | 8 | 63.6 / 57.1 | 31.4 | 17.2 | 0.55 | 0.27 | 1.83 | 42.6 / 16.9 / 4.5 | 3.76 |
| llama8b | psr response | 32 | 183.5 / 179.7 | 48.7 | 22.3 | 0.27 | 0.12 | 2.18 | 162.7 / 31.7 / 9.8 | 3.23 |
| llama8b | psr response | 64 | 319.3 / 314.5 | 81.2 | 34.7 | 0.26 | 0.11 | 2.34 | 294.8 / 61.5 / 20.0 | 3.07 |
| llama8b | lora_all (rank 4) response | 8 | 79.5 / 83.5 | 33.3 | 17.9 | 0.40 | 0.23 | 1.86 | 69.0 / 18.8 / 5.2 | 3.62 |
| llama8b | lora_all (rank 4) response | 32 | 138.3 / 233.8 | 64.1 | 18.3 | 0.27 | 0.13 | 3.50 | 216.8 / 47.1 / 5.8 | 8.12 |
| llama8b | lora_all (rank 4) response | 64 | 283.7 / 258.8 | 105.5 | 26.2 | 0.41 | 0.09 | 4.03 | 239.1 / 85.8 / 11.5 | 7.46 |
| llama8b | lora rank1 response | 8 | 15.2 / 18.0 | 17.0 | 15.3 | 0.94 | 1.01 | 1.11 | 3.5 / 2.5 / 2.6 | 0.96 |
| llama8b | lora rank1 response | 32 | 17.3 / 22.9 | 20.7 | 13.1 | 0.90 | 0.76 | 1.58 | 5.9 / 3.7 / 0.6 | 6.17 |
| llama8b | lora rank1 response | 64 | 22.0 / 33.0 | 25.9 | 14.7 | 0.78 | 0.67 | 1.76 | 13.3 / 6.2 / 0.0 | n/a |
| llama8b | multigroup (psr+lora_r4+psr) | 8 | 66.5 / 84.0 | 33.7 | 18.3 | 0.40 | 0.28 | 1.84 | 69.5 / 19.2 / 5.6 | 3.43 |
| llama8b | multigroup (psr+lora_r4+psr) | 32 | 187.2 / 178.3 | 54.0 | 29.7 | 0.30 | 0.16 | 1.82 | 161.3 / 37.0 / 17.2 | 2.15 |
| llama8b | multigroup (psr+lora_r4+psr) | 64 | 350.1 / 340.1 | 89.0 | 26.8 | 0.26 | 0.08 | 3.32 | 320.4 / 69.3 / 12.1 | 5.73 |
| gemma4e4b | psr response | 8 | 87.7 / 104.0 | 39.5 | 27.8 | 0.38 | 0.32 | 1.42 | 73.7 / 9.2 / 6.8 | 1.35 |
| gemma4e4b | psr response | 32 | 218.1 / 242.6 | 59.9 | 31.1 | 0.25 | 0.14 | 1.93 | 217.1 / 34.4 / 9.0 | 3.82 |
| gemma4e4b | psr response | 64 | 392.4 / 436.7 | 91.0 | 34.8 | 0.21 | 0.09 | 2.61 | 409.3 / 63.6 / 13.0 | 4.89 |
| gemma4e4b | lora_all (rank 4) response | 8 | 80.6 / 90.4 | 41.7 | 39.0 | 0.46 | 0.48 | 1.07 | 60.1 / 11.4 / 18.0 | 0.63 |
| gemma4e4b | lora_all (rank 4) response | 32 | 193.6 / 228.9 | 69.0 | 41.2 | 0.30 | 0.21 | 1.67 | 203.4 / 43.5 / 19.1 | 2.28 |
| gemma4e4b | lora_all (rank 4) response | 64 | 367.5 / 359.2 | 100.1 | 39.8 | 0.28 | 0.11 | 2.52 | 331.8 / 72.7 / 18.0 | 4.04 |
| gemma4e4b | multigroup | 8 | 94.2 / 158.4 | 41.9 | 32.1 | 0.26 | 0.34 | 1.31 | 128.1 / 11.6 / 11.1 | 1.05 |
| gemma4e4b | multigroup | 32 | 251.9 / 373.2 | 63.6 | 34.2 | 0.17 | 0.14 | 1.86 | 347.7 / 38.1 / 12.1 | 3.15 |
| gemma4e4b | multigroup | 64 | 452.0 / 478.1 | 101.1 | 39.8 | 0.21 | 0.09 | 2.54 | 450.7 / 73.7 / 18.0 | 4.09 |
| qwen35_4b | psr response | 8 | 71.6 / 118.8 | 58.0 | 34.2 | 0.49 | 0.48 | 1.70 | 96.4 / 35.6 / 10.7 | 3.33 |
| qwen35_4b | psr response | 32 | 169.5 / 293.3 | 78.6 | 35.3 | 0.27 | 0.21 | 2.23 | 255.1 / 40.4 / 11.3 | 3.58 |
| qwen35_4b | psr response | 64 | 297.1 / 539.8 | 116.0 | 54.0 | 0.21 | 0.18 | 2.15 | 510.9 / 87.1 / 27.4 | 3.18 |
| qwen35_4b | lora_all (rank 4) response | 8 | 32.1 / 56.5 | 47.7 | 27.0 | 0.84 | 0.84 | 1.77 | 34.1 / 25.3 / 3.5 | 7.23 |
| qwen35_4b | lora_all (rank 4) response | 32 | 66.8 / 99.6 | 64.4 | 34.6 | 0.65 | 0.52 | 1.86 | 61.4 / 26.2 / 10.6 | 2.47 |
| qwen35_4b | lora_all (rank 4) response | 64 | 85.6 / 158.3 | 75.8 | 38.0 | 0.48 | 0.44 | 1.99 | 129.4 / 46.9 / 11.4 | 4.11 |
| qwen35_4b | multigroup | 8 | 71.5 / 126.2 | 59.1 | 28.8 | 0.47 | 0.40 | 2.05 | 103.8 / 36.7 / 5.3 | 6.92 |
| qwen35_4b | multigroup | 32 | 175.2 / 298.0 | 48.1 | 36.3 | 0.16 | 0.21 | 1.33 | 259.8 / 9.9 / 12.3 | 0.80 |
| qwen35_4b | multigroup | 64 | 301.9 / 341.4 | 99.8 | 44.9 | 0.29 | 0.15 | 2.22 | 312.5 / 70.9 / 18.3 | 3.87 |
| glm46vflash | psr response | 8 | 93.7 / 128.4 | 57.3 | 28.7 | 0.45 | 0.31 | 2.00 | 93.6 / 22.5 / 10.3 | 2.18 |
| glm46vflash | psr response | 32 | 199.0 / 271.4 | 85.4 | 31.2 | 0.31 | 0.16 | 2.74 | 248.4 / 62.4 / 4.1 | 15.22 |
| glm46vflash | psr response | 64 | 370.1 / 361.6 | 120.3 | 33.9 | 0.33 | 0.09 | 3.55 | 333.5 / 92.2 / 10.2 | 9.04 |
| nemotron3nano4b | psr response | 8 | 69.3 / 94.6 | 28.8 | 18.5 | 0.30 | 0.27 | 1.56 | 73.1 / 7.3 / 5.5 | 1.33 |
| nemotron3nano4b | psr response | 32 | 196.8 / 267.7 | 68.3 | 30.8 | 0.26 | 0.16 | 2.22 | 252.0 / 52.6 / 17.1 | 3.08 |
| nemotron3nano4b | psr response | 64 | 377.8 / 513.3 | 84.1 | 39.0 | 0.16 | 0.10 | 2.16 | 486.5 / 57.3 / 16.8 | 3.41 |

Reading: at B=32–64 the stacked hook read **2–3.5x faster per step than the
per-row hook** on every family in this cross-run comparison (hook-only cost
3–9x lower); at B=8 1.1–2x. The same-node A/B (§2.3.6) puts the per-step
figure at 1.2–3.0x at B=32–64 — the cross-run ratios were flattered by node
and admission differences, and the per-layer kernel ratio is 9–33x (§2.1);
the whole-step gain is bounded by the passthrough floor. Per-row itself is
already 3–10x faster than the legacy loop at B ≥ 32 (the landed
lens-vectorize win).

Caveats: the stacked candidate on 382416 sometimes ran fewer concurrent
requests in the window than the per-row candidate (`running` 50–58 vs 61–64
at B=64 on llama / GLM — admission, not the hook), which flatters its
ms/step slightly; the two candidates ran on different nodes and days.

#### 7.2.2 Cases with per-row timing only (no stacked standup survived)

| model | case | B | legacy ref | per-row | per-row/legacy | hook-only legacy / per-row |
|---|---|---:|---:|---:|---:|---|
| llama8b | vector_all response | 8 / 32 / 64 | 50.8 / 134.9 / 197.5 | 14.7 / 20.5 / 31.0 | 0.29 / 0.15 / 0.16 | 36.3 / 117.9 / 177.8 vs 0.2 / 3.5 / 11.3 |
| gemma4e4b | vector_all response | 8 / 32 / 64 | 75.4 / 177.7 / 486.8 | 30.3 / 37.6 / 36.9 | 0.40 / 0.21 / 0.08 | 45.1 / 152.2 / 459.4 vs 0.0 / 12.1 / 9.5 |
| gemma4e4b | lora rank1 response | 8 / 32 / 64 | 42.1 / 49.3 / 55.8 | 41.1 / 47.2 / 47.6 | 0.98 / 0.96 / 0.85 | 11.8 / 23.8 / 28.4 vs 10.8 / 21.7 / 20.2 |
| qwen35_4b | vector_all response | 8 / 32 / 64 | 101.5 / 220.2 / 441.1 | 43.9 / 51.4 / 59.4 | 0.43 / 0.23 / 0.13 | 79.1 / 182.0 / 412.2 vs 21.5 / 13.2 / 30.5 |
| qwen35_4b | lora rank1 response | 8 / 32 / 64 | 44.3 / 49.7 / 58.9 | 42.2 / 48.9 / 52.3 | 0.95 / 0.98 / 0.89 | 21.9 / 11.5 / 30.0 vs 19.8 / 10.7 / 23.4 |
| glm46vflash | vector_all response | 8 / 32 / 64 | 84.6 / 252.8 / 409.7 | 43.1 / 46.8 / 53.1 | 0.51 / 0.19 / 0.13 | 49.8 / 229.8 / 381.6 vs 8.3 / 23.8 / 25.0 |
| glm46vflash | lora rank1 response | 8 / 32 / 64 | 36.3 / 44.0 / 50.3 | 38.0 / 40.8 / 43.7 | 1.05 / 0.93 / 0.87 | 1.5 / 21.0 / 22.2 vs 3.2 / 17.8 / 15.6 |
| glm46vflash | lora_all response | 8 / 32 / 64 | 115.1 / 220.1 / 557.1 | 59.0 / 95.3 / 149.7 | 0.51 / 0.43 / 0.27 | 80.3 / 197.1 / 529.0 vs 24.2 / 72.3 / 121.6 |
| glm46vflash | multigroup | 8 / 32 / 64 | 135.7 / 252.8 / 427.5 | 59.7 / 94.4 / 124.6 | 0.44 / 0.37 / 0.29 | 100.9 / 229.8 / 399.4 vs 24.9 / 71.4 / 96.5 |
| nemotron3nano4b | vector_all response | 8 / 32 / 64 | 56.6 / 140.4 / 271.8 | 22.8 / 29.7 / 32.3 | 0.40 / 0.21 / 0.12 | 35.1 / 124.7 / 245.0 vs 1.3 / 14.0 / 5.5 |
| nemotron3nano4b | lora rank1 response | 8 / 32 / 64 | 23.6 / 31.0 / 40.0 | 23.5 / 26.7 / 31.1 | 1.00 / 0.86 / 0.78 | 2.1 / 15.3 / 13.2 vs 2.0 / 11.0 / 4.3 |
| nemotron3nano4b | lora_all response | 8 / 32 / 64 | 30.0 / 32.9 / 71.3 | 21.6 / 31.5 / 39.7 | 0.72 / 0.96 / 0.56 | 8.5 / 17.2 / 44.5 vs 0.1 / 15.8 / 12.9 |
| nemotron3nano4b | multigroup | 8 / 32 / 64 | 78.8 / 195.1 / 351.3 | 27.0 / 45.9 / 73.4 | 0.34 / 0.24 / 0.21 | 57.3 / 179.4 / 324.5 vs 5.5 / 30.2 / 46.6 |

`vector_all` (elementwise) has no reduction: its per-row hook already sits on
the floor, and stacking changes nothing (§7.3.4). `lora rank1` is cheap
either way (the legacy loop is only 0–30 ms over the floor). The cases that
stacking speeds up are exactly the ones with a reduction per row: psr,
lora_all (rank 4), the multigroup mixes, and the two low-rank kinds. (The A/B's
GLM lora_all B=64, 112.3 → 55.9 ms/step, is the same-node version of this
table's GLM lora_all row.)

### 7.3 Deviation (the 382416 in-worker shadow compare, stacked vs per-row on identical rows)

A "group" is one `[B, d]` tensor at one hooked layer (block site: the
residual rows; o_proj site: the o_proj output rows). Counts are **groups**,
not elements — a group is "unequal" if any of its `B x d` elements differs.
`max |diff|` is over the bf16 OUTPUT (the residual / o_proj output after the
edit is added and cast back).

**Correction (2026-09-04, from §2.1 and §2.3).** The memo read each `max
|diff|` as one bf16 ulp of the OUTPUT element and inferred the element's
magnitude band from it (the "implied |x| band" column). That inference does
not hold: the flip is one bf16 ulp of the EDIT or of the SUM (§3), so the
same |diff| can sit on an output element far larger (Nemotron: |diff| 0.25 in
groups whose largest per-row value is ~1e3, where the output's own ulp is 4)
or far smaller (the 128-ulp tail at near-cancellation). The column is kept
as the memo stated it and marked superseded; the A/B's `max |ref|` per key
(§2.3) is the measured magnitude. Whether every flip is exactly one ulp of
`max(|output|, |edit|)` is what `max_ulps_edit` measures from the landing on;
the 382416 record did not carry ulps at all.

#### 7.3.1 Reduction ops

| model | K (hidden / o_proj in) | op | positions | groups compared / unequal | % unequal | max abs diff | implied |x| band (SUPERSEDED, see above) | which group sizes were unequal |
|---|---|---|---|---:|---:|---:|---|---|
| llama8b | 4096 | psr gate gemv | all | 6208 / 54 | 0.9% | 2^-9 = 0.00195 | [0.25, 0.5) | prefill-sized groups only (2794, 13570 rows: 22/32 and 32/32 of them); every decode group B in {1,7,8,31,32,43} equal |
| llama8b | 4096 | psr gate gemv | response | 6112 / 0 | 0 | — | — | none (decode groups only) |
| llama8b | 4096 | lora rank1 (o_proj) | all | 195 / 3 | 1.5% | 2^-10 = 0.00098 | [0.125, 0.25) | prefill-sized groups only |
| llama8b | 4096 | lora rank4 (o_proj) | all | 6208 / 1281 | 20.6% | 2^-8 = 0.0039 | [0.5, 1) | prefill sizes AND decode groups of 7 / 8 / 31 / 32 rows |
| llama8b | 4096 | lora rank4 (o_proj) | response | 6112 / 2163 | 35.4% | 2^-9 = 0.00195 | [0.25, 0.5) | B=8: 622/1984 (31%); B=32: 1508/1984 (76%) |
| llama8b | 4096 | multigroup psr+lora_r4 (4–6-row groups) | response | 20160 / 1514 | 7.5% | (first: 1.5e-5 at a 4-row lora group) | — | 4- and 6-row mixed groups |
| gemma4e4b | 2560 | psr gate gemv | all | 8148 / 126 | 1.5% | 2^-7 = 0.0078 | [1, 2) | decode: B=8 12/2604, B=32 54/2604; the rest prefill-sized |
| gemma4e4b | 2560 | psr gate gemv | response | 8022 / 485 | 6.0% | 2^-6 = 0.0156 | [2, 4) | B=8: 112/2604 (4.3%); B=32: 369/2604 (14.2%) — pure decode groups |
| gemma4e4b | 2048 / 4096 | lora rank4 (o_proj) | response | 8022 / 790 | 9.8% | 2^-6 = 0.0156 | [2, 4) | decode groups (B=8 first at layer 15) |
| gemma4e4b | 2560 | multigroup | response | 26628 / 743 | 2.8% | (first: 1.2e-4 at a 6-row psr group) | — | 4–6-row mixed groups |
| qwen35_4b | 2560 | psr gate gemv | all | 6208 / 71 | 1.1% | 2^-9 = 0.00195 | [0.25, 0.5) | decode groups B=8/32 and prefill sizes |
| qwen35_4b | 2560 | psr gate gemv | response | 6112 / 932 | 15.2% | 2^-8 = 0.0039 | [0.5, 1) | decode groups B=7/8/31/32 |
| qwen35_4b | 4096 | lora rank4 (o_proj) | all | 1552 / 232 | 14.9% | 2^-8 = 0.0039 | [0.5, 1) | decode + prefill sizes |
| qwen35_4b | 4096 | lora rank4 (o_proj) | response | 1528 / 449 | 29.4% | 2^-8 = 0.0039 | [0.5, 1) | decode groups (first at B=8, layer 3) |
| qwen35_4b | 2560 | multigroup | response | 14112 / 1506 | 10.7% | (first: 1.2e-4 at a 6-row psr group) | — | 4–6-row mixed groups |
| glm46vflash | 4096 | psr gate gemv | all | 7760 / 78 | 1.0% | 2^-6 = 0.0156 | [2, 4) | prefill-sized groups only (328, 2519, 13601 rows); every decode group B <= 44 equal |
| glm46vflash | 4096 | psr gate gemv | response | 7640 / 0 | 0 | — | — | none |
| nemotron3nano4b | 3136 | psr gate gemv | all | 8148 / 125 | 1.5% | 2^-3 = 0.125 | [16, 32) | decode: B=8 12/2604, B=32 39/2604; the rest prefill-sized |
| nemotron3nano4b | 3136 | psr gate gemv | response | 8022 / 849 | 10.6% | 2^-2 = 0.25 | [32, 64) | decode groups (first at B=8, layer 1) |

(The GLM / Nemotron lora_all and multigroup cases, and lora rank1 / loreft on
four families, never stood up on 382416 — reference-side absent-cell deaths —
so there is no stacked shadow record for them. The A/B (§2.3) supplies GLM
and Nemotron lora_all: 5838/17920 and 658/1792 groups unequal.)

#### 7.3.2 Elementwise ops (control)

`antidote` (add_vector, last), `vector_all` (add_vector) — every compared
group equal on every family (llama 192/0, gemma 191/0, GLM 192/0, Nemotron
191/0). `alphasteer` (prefill_last_matmul) and `loreft` as served (prefill
only) form no decode groups (0 compared): they are per request by
construction and untouched by either arithmetic.

#### 7.3.3 What the numbers mean

* **Mechanism.** The only thing that differs is the fp32 reduction (the
  gate scalar `h.float() @ gate_w`, the LoRA `x @ A^T`, `z @ B^T`): summed in
  a different order, it differs by a few fp32 ulps (~1e-7 relative). That
  perturbation propagates through elementwise fp32 work (`eff * relu(g) *
  v`) and flips the **bf16 rounding** of the added delta, or of the sum,
  wherever the fp32 value sat near a rounding boundary — so a small number
  of elements of the `[B, d]` output move by one bf16 ulp of the edit or of
  the sum (§3; the memo's "at most one ulp of the residual element, 0.4–0.8 %
  relative" understated the near-cancellation case and overstated the
  massive-activation one). The largest |diff| in each record is a flipped
  element in the family's largest-magnitude dims (Nemotron's residual stream
  has sink-like elements near 1e3; llama's and Qwen's flipped elements sat
  below 1). Everything else in the hook is elementwise and exact at any row
  count.
* **Where it shows up.** At K=4096 (llama8b, GLM) the decode-size stacked
  gemv reproduced the M=1 reduction bit for bit in every group — through
  B=44 here, through B=32 in the A/B (0/14368 and 0/17880 groups), through
  B=128 on the microbench, unequal at B=256; only prefill-sized stacks
  differed on 382416, and since 0e8f0e4 groups hold single-row slots only,
  prefill rows are never grouped. Production `max_num_seqs` is 24–52 per
  served geometry, inside the identical range — an empirical property of
  cuBLAS's kernel choice for those (M, K), not a guarantee. At K=2560
  (gemma-4, Qwen3.5) and K=3136 (Nemotron) the decode-size gemv differs from
  B=7 up in 4–15 % of psr groups here and in 4–33 % in the A/B (8-row / 32-row
  groups: gemma 4.4 / 13.9 %, Qwen 13.6 / 32.6 %, Nemotron 8.9 / 23.2 %; gemma
  already at B=2, 2/42); the rank-4 LoRA GEMM pair differs at every K in
  10–35 % of decode groups here and 5–77 % in the A/B (8-row / 32-row: llama
  31 / 77 %, gemma 5.3 / 24 %, Qwen 27 / 63 %, GLM 19 / 58 %, Nemotron 25 /
  63 %). Rank-1 LoRA differed only at prefill sizes on 382416 and only at
  B=256 on the microbench for K=4096 (from B=8 on Nemotron).
* **Per element** the rate is far smaller than per group (a group is
  `B x d` = 20k–260k elements): 382416's record did not count elements; the
  microbench (§2.1: 0 to 3.9e-5) and the A/B (§2.3.6: 4.8e-6 to 1.2e-5) do.
* **Group sizes in production-like waves** (re-gate `by_key`, per-row so
  all equal): for an 8-request wave the worker sees mostly 8-row groups
  with some 7-row and 1-row ones; for 32 requests mostly 32-row with 31, and
  admission splits (2–4, 9/23, 10/22, 14/18, 15/17 rows) a few percent of the
  time — the A/B's by-rows tables show the same shapes (3, 4, 7, 9, 23, 28,
  29, 31). So the stacked shapes that matter are exactly those tested: M in
  1..64 at the family's K.

#### 7.3.4 Which kinds stacking is bit-identical for anyway

* Every **elementwise** op (`add_vector` — vector_all, antidote;
  `prefill_last_cosine_gated`, `prefill_last_matmul`): no `_row_reduction`
  call exists, the mode is a no-op, identity holds by construction.
* **psr gemv at K=4096** (llama8b, glm46vflash): identical in every decode
  group observed (B ≤ 44 on 382416, B ≤ 32 in the A/B, B ≤ 128 on the
  microbench; unequal at B=256) — empirically, per (M, K); certifiable per
  family by the shadow compare, never by construction.
* **Nothing else**: psr at K=2560/3136, every rank-4 LoRA / low-rank pair,
  and the LoReFT affine pair (13 unequal elements at B=32, d=2560; from B=8
  on the microbench) differ at decode sizes.

### 7.4 Token-level effect vs the noise floor (382416 / 384162)

Both measurements below are **composition-confounded** (the gate's own
admission): at BI=0 two servers — or two waves on one server — need not
co-schedule the same requests in the same order, and vLLM's own GEMMs pick
kernels by the batch's token count, so bf16 outputs at B > 1 can differ
between any two runs with the hook bit-identical. What can be compared is
how often stacked-vs-reference disagreed **next to** how often the reference
disagreed with itself and how often the bit-identical per-row candidate
disagreed with the same reference. The A/B (§2.3) is the same comparison
with the floor measured in-session.

#### 7.4.1 Case-level verdicts (64 greedy tokens x B requests; equal = every token id and logprob identical)

Slots per case: `pure_B8`, `pure_B32`, `mixed_B32` (+ the multigroup's
`mixed_B32`). `FAIL` = both servers repeat their own output exactly and
disagree with each other; `UNSTABLE_*` = that server did not repeat itself
(inconclusive).

| model | run | comparisons | equal | FAIL | UNSTABLE_REFERENCE | UNSTABLE_VECTORIZED | FAIL among stable pairs |
|---|---|---:|---:|---:|---:|---:|---:|
| llama8b | stacked (382416) | 28 | 4 | 5 | 18 | 1 | 5/9 |
| llama8b | per-row (384162) | 40 | 9 | 0 | 24 | 7 | 0/9 |
| gemma4e4b | stacked | 16 | 2 | 0 | 13 | 1 | 0/2 |
| gemma4e4b | per-row | 40 | 19 | 0 | 12 | 9 | 0/19 |
| qwen35_4b | stacked | 13 | 0 | **8** | 4 | 1 | **8/8** |
| qwen35_4b | per-row | 37 | 30 | 0 | 3 | 4 | 0/30 |
| glm46vflash | stacked | 12 | 3 | 0 | 8 | 1 | 0/3 |
| glm46vflash | per-row | 40 | 30 | **1** | 6 | 3 | 1/31 |
| nemotron3nano4b | stacked | 12 | 4 | **6** | 2 | 0 | **6/10** |
| nemotron3nano4b | per-row | 37 | 29 | 0 | 5 | 3 | 0/29 |
| **all** | stacked | 81 | 13 | 19 | 45 | 4 | 19/32 (59 %) |
| **all** | per-row | 194 | 117 | 1 | 50 | 26 | 1/118 (0.8 %) |

The one per-row FAIL (GLM psr response `mixed_B32`, shadow 7640/0 unequal —
the hook provably bit-identical) is the composition confound producing a
"both stable, disagree" verdict on its own: a FAIL at B > 1 is never proof of
an arithmetic difference. With that floor at ~1 % of stable pairs, stacked's
59 % is far above it.

Where the stacked FAILs fall is the informative part:

| family | psr response, B=8 (the production kind at the quiet size) | stacked shadow (decode groups) |
|---|---|---|
| llama8b | stacked **equal**; per-row equal | 0 unequal (K=4096) |
| glm46vflash | stacked UNSTABLE_REFERENCE; per-row equal | 0 unequal (K=4096) |
| qwen35_4b | stacked **FAIL** (token ids differ from position 19); per-row equal | 932/6112 unequal |
| nemotron3nano4b | stacked **FAIL** (logprobs differ from position 5, tokens from 16 at B=32); per-row equal | 849/8022 unequal |
| gemma4e4b | stacked UNSTABLE_REFERENCE; per-row UNSTABLE_VECTORIZED | 485/8022 unequal |

and on llama the stacked FAILs are exactly the cases whose shadow found
decode-group inequality (lora_all rank-4 at B=8 and B=32, both stable) or
prefill-group inequality (lora rank1 positions=all, B=8), while every case
with 0 unequal decode groups (psr response, loreft, antidote, alphasteer) read
`equal` at B=8. On Qwen and Nemotron every stable comparison of a psr or
rank-4 LoRA case failed; the same cases at B=8 read `equal` (8/8 requests) for
the per-row candidate. The A/B reproduces this pattern with the floor
measured: Nemotron psr B=8 0.000 identical in logprobs against a 0.917–1.000
floor (§2.3.3); Qwen psr B=8 0.500 in tokens and 0.000 in logprobs against
floors of 1.000 in both (§2.3.5); gemma psr B=8 0.750 in tokens against 1.000
(§2.3.4).

#### 7.4.2 The request-level floor (re-gate raw signatures, per-row hook)

Fraction of requests whose full output was identical between two waves of the
same batch on the SAME server (ref run 1 vs run 2; the re-gate JSON), the
production kinds:

| B | typical same-server repeat agreement | worst cases |
|---:|---|---|
| 8 | 8/8 in 47 of 52 slots (reference and candidate servers, all five families) | llama loreft 0/8, llama lora rank1 2/8, gemma lora/loreft 6/8, gemma psr response candidate 6/8 |
| 32 | bimodal: 32/32 in about half the slots, 0–7/32 in a fifth (a whole wave admitted differently), the rest in between | llama psr all 29/32 then 0/32 on the next server; gemma vector_all 0/32; nemotron psr all 0/32 |

Cross-server, bit-identical hook (ref vs per-row candidate): 8/8 at B=8 in
48 of 52 slots; at B=32 anywhere from 0/32 to 32/32, tracking the same-server
repeat instability. So **at B=8 the floor is essentially zero** (a wave of 8
is admitted as one group nearly every time) and a stable-pair FAIL there is
telling; at B=32 the floor alone flips whole waves, and only the in-worker
shadow can say anything. The A/B's `per_row self` / `per_row x server` rows
(§2.3) are this floor measured in-session: 0.92–1.00 at B=8, 0.33–1.00 at
B=32 (all), 0.69–1.00 (tokens).

#### 7.4.3 Reading

* The stacked deviation is **not inside the noise floor at B=8** on the
  families and kinds where the shadow found decode-group inequality: it
  flips greedy tokens within 64 steps (first divergence at token 3–57 on
  382416; median 14–47 in the A/B) where two per-row servers, and the same
  server twice, agree on all 8 requests.
* It **is the same class of perturbation** as the composition noise — one
  bf16 ulp of the edit on a few elements per 10^5 per layer — and at B=32
  production already tolerates that class flipping most of a wave between
  two identical requests. What differs is that the stacked flip is present
  every step, composition-independent, on kinds/families with K != 4096 (and
  on every rank-4 / LoReFT group), so it adds a deterministic offset from the
  per-request arithmetic rather than another draw of the same noise.
* The **admission**: neither 382416 nor 384162 held composition fixed; the
  case-level verdicts are on 1–3 waves per slot; 382416 lost its raw
  signatures, so its stacked side has no per-request fractions. The A/B
  supplied them (§2.3) with the floor measured next to the effect; a
  confound-free token-level number would still need the two arithmetics on
  the same batch under the same scheduling (an in-process `LLM.generate` twin
  run or a worker-side twin decode), which was not built.

## 8. History — what the contract change had to state, and the options

### 8.1 What a contract change had to state (memo §7)

The pre-landing contract (`_row_reduction`'s docstring, the gate's
`shadow_compare` hard check — any unequal group is exit 2 — and
`TestRowwiseReductions` pinning the per-row call shapes): the vectorized hook
is **bit-identical to the per-request path on every family**, held by
construction and certified on hardware. Turning stacked reductions on
anywhere had to replace that with, at minimum:

* an explicit **tolerance** in the activation's own units, and the shadow
  check re-scoped from "unequal ⇒ FAIL" to "beyond tolerance ⇒ FAIL" — §3,
  with the bound on the element fraction measured by the microbench and the
  A/B (3.9e-5 and 1.2e-5 at most) and the unit fixed at
  `max(|output|, |edit|)`
* which **reference** future gates compare against — the per-row path stays
  the ground truth of "what the HF hook bodies compute" (kept as
  `SISTEER_LENS_STACKED_REDUCTIONS=0`), and the served default no longer
  reproduces it beyond B=1
* the **fidelity statement** consumers rely on — §4: B=1 exact; at B > 1
  "equivalent up to the BI=0 composition class plus a deterministic ≤ 1-ulp
  hook offset on every reduction kind", not "identical to the per-request
  path"
* the **Petri** serving surface, frozen and certified at 16384 x 24 on the
  per-row hook: the default reaches it on the next server built from a
  post-landing tree, and its smoke re-certifies it (§5)
* the tests that pin call shapes (`TestRowwiseReductions`,
  `test_lens_stacked_reductions`'s default test) flip to pin the stacked
  default and the per-row opt-out, and the 0e8f0e4 rationale is superseded
  in the docstrings (`_row_reduction`'s docstring now states both modes and
  why stacked is the default)

### 8.2 Options considered (memo §8) — B taken

* **A. Keep per-row (status quo).** Bit-identity by construction; the
  landed 3–10x over legacy stands; leaves the stacked speed on the table at
  B ≥ 32 for psr / rank-4 LoRA / mixed groups. Not taken.
* **B. Stacked everywhere, behind the contract change. TAKEN (2026-09-04).**
  The full speed of §2.1 / §2.3; accepts a deterministic bounded offset from
  the per-request arithmetic on every reduction kind (measured: element
  fraction ≤ 3.9e-5, one bf16 ulp of the edit or the sum), which the B=8
  evidence says does flip greedy tokens where the composition floor would
  not. Requires the tolerance contract (§3), the gate re-scope, and a Petri
  re-certification (§5).
* **C. Per kind and per family, certified, everything else per row.**
  Stacked only where the shadow compare shows 0 unequal groups at every
  decode size up to the served `max_num_seqs` — psr on the K=4096 families
  (llama8b, glm46vflash) and, trivially, the elementwise kinds. Would have
  bought the psr speedup on two of five families and nothing for gemma /
  Qwen / Nemotron psr or any rank-4 LoRA; identity would stay an empirical
  per-(M, K) fact re-certified per gate run. Not taken.
* **D. A third arithmetic that is bit-identical AND one launch.** The
  microbench's `bmm` probe was the candidate. Measured (§2.1): identical to
  neither the per-row nor the stacked gemv on any of 50 records, and slower
  than stacked. Dead.
* **E. Stacked only at B above a threshold** (e.g. 32). Same contract change
  as B with a narrower blast radius, but production runs most at S = 24–52
  where the offset is present anyway, and a mode that depends on B would
  make the attested arithmetic a function of admission. Not taken.

## 9. Contract change for GATED reduction ops (2026-09-06)

**Decision (on the ff35913 evidence):** keep the ≤ 4
edit-ulp bound for LINEAR reduction ops; for GATED reduction ops replace it
with a per-row bound AT THE GATE, |g_stacked − g_per_row| ≤ C · 2^-24 ·
Σ_i |h_i w_i| with C = 16, and a ReLU flip allowed only when the side that
fired is within that same bound; keep the pooled fraction and the row/group
concentration bounds unchanged for both classes.

### 9.1 What was measured

The five-family gate at `c8e0f45` (all cases, B ∈ {1, 8, 32}) was clean on
llama8b, glm46vflash and gemma4e4b (worst edit-ulps 1) and FAILED the PSR
response case on qwen35_4b (edit-ulps 5, and 11 in the multigroup) and
nemotron3nano4b (23, and 7). Concentration bounds held everywhere at ≥ 10×
margin (row ≤ 2e-3 vs 2e-2; group ≤ 3e-4 vs 2e-3; pooled ≤ 8e-6 vs 1e-4);
zero modified-flag disagreements; zero ReLU flips.

The evidence run at `ff35913` (one H200 per family, the single case
`psr:relu_gated_vector:gemv:response:skipbos0:both`, gemma as the matched
control at the same hidden size as qwen) reproduced the worst element of each
family exactly and read it:

| | qwen35_4b (FAIL, 5) | nemotron3nano4b (FAIL, 23) | gemma4e4b (PASS, 1) |
|---|---|---|---|
| unsteered value at the worst element | **0.0 exactly** | **0.0 exactly** | −5.5e-4 |
| per-row → stacked output | 1.136e-7 → 1.108e-7 | −2.573e-8 → −2.293e-8 | −1.587e-3 → −1.572e-3 |
| \|diff\| | 2.8e-9 | 2.8e-9 | 1.5e-5 |
| ‖stacked edit‖ / ‖per-row edit‖ (that row) | 0.99966 | **0.8896** | 0.9999996 |
| cosine(stacked edit, per-row edit) | 0.999996 | 0.9999992 | 1.0000001 |
| gate on that row (per-row → stacked) | 2.21e-5 → 2.16e-5 | ~1e-4, −10.9 % | 0.81 → 0.81 |
| \|Δgate\| / Σ\|h·w\| at that row, in 2^-24 | 0.8 | 0.6 | 0.4 |
| case-wide max \|Δgate\| (absolute) | 1.7e-6 | 3.8e-5 | **9.5e-6** |
| case-wide smallest gate that differed | 1.4e-5 | 1.4e-5 | 8.6e-4 |
| case-wide max relative gate deviation | 2.2 % | 10.9 % | 0.025 % |

### 9.2 Why the edit-ulp unit was wrong for gated ops

The ≤ 4 edit-ulp rule was derived from "kernel variance perturbs the fp32
edit by a few fp32 ulps of the edit". For a LINEAR reduction op that premise
holds: the reduction's output IS the edit (the LoRA write, the LoReFT
back-projection), so a few fp32 ulps of the partial sums are a few fp32 ulps
of the edit, and after the two bf16 roundings at most one bf16 ulp of the
edit plus one of the sum — all five families read ≤ 1.

For a GATED reduction op the reduction produces a GATE SCALAR that then
scales a fixed direction. Kernel variance perturbs that scalar by a few fp32
ulps of its PARTIAL SUMS Σ_i |h_i w_i| — 0.4–0.8 × 2^-24 of it in every
family above, one rounding of one partial sum, the minimum possible kernel
variance; gemma's absolute gate deviation is the LARGEST of the three and it
passes. The perturbation is independent of the gate's own size, so on a
near-threshold ReLU gate (qwen's and nemotron's fitted PSR objects have rows
gated at 1e-5 … 1e-4) a 1e-6 absolute rounding is 2–11 % of the gate, and
the whole row's edit scales by exactly that factor (nemotron: ratio 0.8896,
cosine 0.999999). The per-element unit then reads that as many ulps only
where the activation is exactly zero: there an edit of 1e-7 is the entire
output and fully resolvable at its own scale, so a 2–11 % change of it is
5 or 23 bf16 ulps of that edit, while on every other element of the row the
edit is below the activation's bf16 resolution and nothing differs (1/2560
and 3/3136 elements differ). The unit was measuring which OBJECT has tiny
gates and which activations are exactly zero, not whether the arithmetic
differs. Nothing in it is a defect: a dropped, halved or misplaced edit reads
ratio ~0 / ~0.5 / a low cosine and lights up its row; a misplaced gate
differs by O(1) of Σ|h·w|, ~1e5–1e6 in these units.

Not taken: raising 4 → 32 for gated ops (moves the threshold without fixing
the unit — an object gated at 1e-6 would read 200), or per-row mode for gated
ops (gives back most of the speedup, PSR being the production op).

### 9.3 The rule as implemented

- `lens_adapters.GATED_REDUCTION_OPS` / `LINEAR_REDUCTION_OPS` partition
  `REDUCTION_OPS` (the ONE definition; the gate mirrors the literals and
  `tests/test_lens_stacked_reductions.py` pins them equal);
  `reduction_op_class` resolves an op to `reduction_gated` /
  `reduction_linear` / `elementwise` or raises. `shadow_key_op_class` returns
  the three classes; a key with any gated op is gated (no registered kind
  produces a mixed key); an unclassifiable op is still FAIL.
- **The gates are OBSERVED, not recomputed.** The
  gated branches of `lens_adapters._apply_one` carry a capture seam
  (`GateCapture` / `gate_capture()`, a module-level sink; unarmed — the
  production path — it costs a gated-vector op one module-global lookup and
  one branch, a low-rank op one lookup and three branches stacked / B + 2
  per-row, no allocation and no context manager in either; not zero): while
  armed they record the gate tensor the edit
  was built from (post-ReLU, `[B]` for the gated-vector pair, `[B, k]` latents
  for the low-rank pair), Σ|h_i w_i| from the same `hidden.float()`, `eff`,
  the row mask and the direction. `_apply_slots_block` arms one around the
  grouped apply and `_shadow_block` one around each per-row reference apply;
  `lens_worker._gate_stats_observed` compares the two observations gate
  element by gate element on every UNEQUAL gated group at the block site (the
  only site a gated op can apply at — `OPROJ_SITE_OPS` is LoRA only) and
  records `gate_max_diff_rel_sum` = max |Δg| / Σ|h_i w_i|, `gate_n_relu_flips`,
  `gate_max_flip_rel_sum` = max over flips of the fired side / Σ|h_i w_i| (None
  without flips), and **`gate_edit_consistency_max_ulps`**: for every element,
  `applied − ref` must equal `eff · (g_stacked − g_per_row) · v` (the low-rank
  write for the low-rank pair) up to the bf16 roundings of the two edits and
  the two sums — the residual in units of ulp(|ref|) + ulp(|edit|), bounded at
  4. Why both: a recomputation from the rows certifies the arithmetic in the
  abstract and is blind to a grouped-only defect that hands row 0 row 1's gate
  — with a sparse direction that defect also stays under every concentration
  bound (one element per row) and the recomputed ratio reads 0.0, a false
  green reproducible through the public record and rule; the
  observed gate delta reads O(1) of Σ|h·w| there (~8e6 × 2^-24). A wrong
  WRITE (a transposed operand, another row's direction) leaves the gate delta
  small and breaks the identity instead. The recomputed pair (`_gate_stats`)
  rides along as `recomputed_*` for reading only. `ShadowCompareRecord`
  aggregates all of it case-wide and per key, plus `gate_n_groups_with_stats`
  and `gate_n_errors`.
- **Provenance is enforced, not assumed.** Every gate record
  carries `gate_source`; the record counts the unequal groups whose stats are
  `'observed'` (`gate_n_groups_observed`, per key and case-wide) and the rule
  requires that count to equal a gated key's `n_unequal` — a recomputed,
  unsourced or mixed-source record is missing evidence however good its
  numbers, and the worst nested record's label is never consulted for it.
- **Exact where the gates agree.** After an equal observed
  gate a gated-VECTOR op is elementwise (the same fp32 products, mask, cast
  and add), so on a row whose gate delta is exactly zero for every vector op
  writing it — or which every gated op masks off — the two writes must be
  bit-identical: `gate_exact_rows_unequal_elems` counts unequal elements in
  such rows and ANY is a FAIL (`REDUCTION_GATE_EXACT_ROWS_UNEQUAL_TOL = 0`).
  The 4-ulp identity tolerance alone let a grouped-only one-ulp sparse write
  (and a one-ulp write on a masked-off row) through at residual ~0.5. A
  LOW-RANK write `z @ Bᵀ` is itself a k-term reduction, so a zero latent delta
  cannot promise an exact write: those rows are judged by the LINEAR unit
  (`gate_lowrank_zero_delta_max_ulps_edit` ≤ `REDUCTION_ULPS_EDIT_TOL` = 4; a
  one-output-ulp flip reads 0 excess, a wrong write of many edit ulps fails).
- **Write reconstruction for the gated-vector pair.** A
  legitimate nonzero gate delta exempts its row from the exactness rule, and
  the 4-ulp identity envelope then hides an unrelated sparse wrong write on
  that row: one ulp written where the direction is exactly ZERO reads a
  residual of 1 (reproduced on a 32×4096 bf16 group with Δg = 2 × 2^-24 —
  every bound under threshold, verdict ok). So for a group whose ops are all
  gated-vector ops the worker REBUILDS each side's output from that side's
  observed gates with the apply's own write — `lens_adapters.
  gated_vector_write`, the one code path `_apply_one` itself calls, so the
  reconstruction cannot drift — and counts the elements of `applied` / `ref`
  that are not the bit-exact reconstruction (`gate_write_recon_mismatch_elems`,
  `gate_n_groups_recon` groups rebuilt); ANY is a FAIL
  (`REDUCTION_GATE_WRITE_RECON_MISMATCH_TOL = 0`), whatever the gate delta.
  A vector-gated key whose unequal groups were not all rebuilt (an
  interleaved non-gated op — no registered kind has one) is missing evidence.
  The low-rank pair is NOT reconstructed: its write `z @ Bᵀ` is a k-term
  reduction whose kernel variance is exactly what the linear unit tolerates,
  so there the identity residual (≤ 4) and the zero-delta rows' linear unit
  remain the rule: a low-rank row that RECEIVES its edit is never held exact,
  identical latents or not — it is BOUNDED (≤ 4 edit ulps; a one-output-ulp
  flip reads 0 excess and PASSES — the accepted class, pinned on the
  registered `lowrank_gated:lowrank_relu_gated` key), and exactness-at-zero
  belongs to the vector pair alone (plus any row a low-rank op masks off).
- **The seam is inert in production, not free.**
  `gate_capture` refuses a `None` sink (an `AdapterError`), `_apply_group_block`
  enters it only under the shadow compare (the production path runs the bare
  apply — no context manager, no capture object), and the low-rank branch
  allocates its latent list only when a capture is armed. What production
  pays, per op applied: a gated-vector op one module-global lookup and one
  branch; a low-rank op one module-global lookup and THREE branches in the
  stacked mode (the list guard, the latent guard inside the reduction closure,
  the record guard) and B + 2 in the per-row mode, where the closure runs once
  per row. Not "one lookup and one branch" for every class, and not zero. A
  test drives a non-shadow worker with a `gate_capture` that raises if called
  and a `torch.cat` spy that sees nothing gathered, and a tracing test counts
  those guard executions exactly, per class and per mode.
- **Healthy-but-unequal, end to end** (the coverage gap): a 32-row bf16 decode
  batch of non-integer rows with a non-integer gated-vector op, and another
  with a low-rank op, through the worker's shadow path on the CPU — the
  stacked `[32, 4096] @ [4096]` and the 32 single-row gemvs round differently
  in fp32, so the records are genuinely unequal — read ok under the rule with
  observed gate deltas a few 2^-24 of Σ|h·w|, the identity within a ulp and
  the writes exact wherever the gates agree.
- `reduction_tolerance` (stacked mode): elementwise → exact; LINEAR → the
  unchanged edit-ulp ≤ 4 rule plus the pooled fraction and concentration
  bounds; GATED → `gate_max_diff_rel_sum ≤ 16 × 2^-24`, `gate_max_flip_rel_sum`
  None or ≤ 16 × 2^-24, `gate_edit_consistency_max_ulps ≤ 4`, the pooled
  fraction and concentration bounds unchanged and still applied, and EVERY
  unequal gated group must carry OBSERVED gate stats
  (`gate_n_groups_with_stats == n_unequal` per key, a `gate_max_diff_rel_sum`,
  a `gate_edit_consistency_max_ulps` — absent from a recomputation-only
  record — and a `gate_max_flip_rel_sum` for any flip; any `gate_error` →
  FAIL) — missing evidence is a FAIL, never a pass. `max_ulps_edit` for gated keys is
  recorded and reported (`max_ulps_edit_gated`) but is not a rule. Per-row
  mode → exact everywhere. FAIL details name the bound and the measured value
  in the same unit ("|delta gate|/sum|hw| … x 2^-24 > 16 x 2^-24"); the
  report's shadow column shows `gate_rel_sum=` / `flip_rel_sum=` in units of
  2^-24 for gated keys.

### 9.4 What the recorded runs say under the new rule (CPU replay of their shadow records)

`tests/fixtures/lens_gate_shadow_records/` holds the shadow snapshots of the
eight recorded runs (five families at `c8e0f45`, three at `ff35913`), numeric
records only; `tests/test_lens_vectorize_gate.py::TestRealGateRecords` replays
them through `reduction_tolerance`:

- every LINEAR key with unequal groups (lora rank-1, lora_all rank-K) on all
  five families: ok, edit-ulps ≤ 1 — the unchanged rule reads as before
- every elementwise and exact case: ok
- the multigroup cases: still FAIL on the unclassifiable dash keys
  (`psr:-:-:-:-:-:B6`), as before (the worker fix at 9603b40 postdates these
  records)
- Every GATED key with unequal groups, at `c8e0f45` AND at `ff35913`: **missing
  evidence**, not a pass and not a numeric verdict. The `c8e0f45` worker
  recorded no gate stats at all. The `ff35913` worker recorded the
  `|Δgate| / Σ|h·w|` ratio only INSIDE the `first_unequal` / `worst_unequal`
  sub-records of each key, never as the per-key aggregate over every unequal
  group that the rule reads — so the recorded qwen/nemotron/gemma PSR cases
  cannot be certified from those records; the five-family re-gate at the new
  sha is what certifies them.
- The rule's arithmetic on the recorded sub-samples: patching each key's
  aggregate from its first/worst sub-records (and a placeholder 0 for the
  gate-edit identity, which no recorded run measured) gives ok on all three, with
  |Δgate| / Σ|h·w| = 4.24 × 2^-24 (qwen, its B31 first-unequal group), 0.62
  (nemotron), 2.65 (gemma) — and edit-ulps 5 / 23 / 1 now informational.
  Note the 4.24: the "0.8" in §9.1 is the ratio at the worst-BY-EDIT-ULPS
  element, not the case-wide maximum, which no recorded run measured. The
  margin of C = 16 over the recorded sub-samples is ~4×, not 20×; the re-gate
  records the true case-wide maximum for every unequal group.

### 9.5 The gate-magnitude histogram — the number behind the small-gates question (measurement, not a rule)

The evidence run found the qwen35_4b and nemotron3nano4b fitted PSR objects
gating some rows at 1e-5 … 1e-4 — essentially at the ReLU threshold — which
is what turned one rounding of a partial sum into a 2–11 % change of a row's
edit. Whether those near-threshold gates are a problem in themselves the
re-gate answers with a number rather than an impression: for every OBSERVED
per-row gate element `g` (post-ReLU; one per row for the gated-vector pair,
one per row × latent for the low-rank pair) of EVERY compared gated group —
equal or unequal (the unequal groups alone are not the served gates; 999
equal far-gate groups and one unequal near-gate group would otherwise read
~100 % near-threshold) — with its row's noise floor `f = 2^-24 · Σ|h_i w_i|`
(one fp32 rounding of the reduction's own magnitude — the unit the gate bound
is stated in), the worker records per compared group
(`_gate_histogram_observed`, from the per-row captures), and the record sums
per key and case-wide, apart from the unequal-only gate stats
(`gate_hist_n_groups` says how many groups contributed):

| field | meaning |
|---|---|
| `gate_hist_zero` | elements with `g == 0` (the ReLU did not fire) |
| `gate_hist_near` | `0 < g ≤ 10 f` — within ten of its own roundings of the threshold |
| `gate_hist_mid` | `10 f < g ≤ 1000 f` |
| `gate_hist_far` | `g > 1000 f` |
| `gate_dose_share_near_threshold` | Σ over near-threshold elements of the dose `|eff · g| · ‖direction‖` (`v` for the vector pair, column j of B for latent j of the low-rank pair) over Σ over all elements — how much of the served steering rides on gates within ten roundings of zero |
| `gate_g_over_f_min_nonzero`, `gate_g_over_f_max` | the smallest nonzero and the largest cancellation ratio `g / f` |

The gate reports these in the per-case line and in the FAIL brief
(`gate_hist_brief`) and never rules on them. A large near-threshold dose share
would say the object's steering is carried by gates the kernel's own rounding
can move by whole percentages — a property of the fitted object, not of the
serving arithmetic, and a question for the fitting side; a small one says the
near-threshold rows the edit-ulp unit tripped on carry negligible dose.

Note on the subset gate run at `2538815` (recomputation-based `_gate_stats`):
its case-wide |Δgate| / Σ|h·w| ratios measure the kernel-variance MAGNITUDE
honestly (the recomputed pair is the same arithmetic on the same rows), so they
remain valid evidence for the size of C; certification of the gated cases
comes only from a run at a sha carrying the observed-gate seam.
