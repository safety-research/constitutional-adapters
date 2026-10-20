# Served-vLLM request geometry: measured p99 of prompt + completion, per eval × arm × model family

Every served eval runs at its own `(max_model_len, max_num_seqs)` = `(L, S)`, resolved from the table
in `src/sisteer/capeval/serving/geometry.py`. This file is the measurement behind each entry: every row
is computed from the eval's real records (the served runs' own timing / usage files and the stored
generations) with the exact tokenizer of each model; the multi-turn and Petri rows use vLLM's own
`usage` counters. Per-cell sub-rows, prompt text and raw per-call arrays are not copied here.

**Columns.** `n` = target requests measured; `p50 / p90 / p99 / max` = prompt + completion tokens;
`p99 prompt` = prompt tokens alone; `max_tokens` = the completion cap the driver requested; `L` = the
candidate window (next multiple of 256 at or above the p99 total) and `S = floor(393216 / L)` its
identity concurrency at the certified batched-token budget; `frac > L` = share of requests whose
prompt + completion exceeded L; `clamp` = share of requests vLLM would 400-reject at `max_model_len = L`
with the cap sent unclamped.

**Sizing rule** (`geometry.sizing_problem`): `L = next block multiple >= p99(prompt + completion)`,
unless the clamp rate at that L exceeds 5 %, in which case `L = next block multiple >= p99(prompt) +
requested max_tokens`. An entry whose driver refuses a clamped cap-bound sample is sized to
`max_prompt + max_tokens` instead (clamp 0 by construction). `max_num_seqs = identity_seqs(L)`
everywhere except Petri's certified 16384 × 24. Per-model budgets (§7) replace 393,216 at serving time
through `ModelBudget.seqs_at`.

**SI prefix token counts** (system turn only, exact tokenizers): `si_constitution` 5,872 / 6,032 / 5,963
tokens (Llama-3.1 / gemma-4 / Qwen3.5); `si_irrelevant` 6,380 / 6,565 / 6,683. A prompted single-shot
request at the 4,096 cap is therefore prefix + attack prompt + up to 4,096 completion (~11–12k).

## 1. Multi-turn attacks (PAIR / Crescendo / ActorAttack), served gemma4e4b target (`none` + `si_constitution__psr_kl_all`)

Requested `max_tokens` = 2048 on every call; 4 of 4,598 calls finished by `length`. Source: the served
target's per-call timing rows (vLLM usage), 36 shard files.

| task / attack | model family | n calls | p50 | p90 | p99 | max | p99 prompt | max comp | max_tokens | L / S | frac > L | clamp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL (PAIR+Crescendo+ActorAttack) | gemma4e4b | 4,598 | 1,163 | 4,235 | 8,484 | 12,657 | 7,496 | 2,048 | 2,048 | 8,704 / 45 | 0.91% | 1.37% |
| pair | gemma4e4b | 1,217 | 230 | 1,019 | 1,864 | 3,229 | 1,343 | 2,048 | 2,048 | 2,048 / 192 | 0.74% | 100.00% |
| crescendo | gemma4e4b | 1,343 | 1,191 | 5,261 | 9,788 | 12,657 | 9,443 | 2,048 | 2,048 | 9,984 / 39 | 0.74% | 2.98% |
| actorattack | gemma4e4b | 2,038 | 2,225 | 4,485 | 6,579 | 9,519 | 5,558 | 2,048 | 2,048 | 6,656 / 59 | 0.98% | 3.34% |

The 8,704 window was measured under a 2,048 completion cap. The campaign's multi-turn instrument runs
at a 4,096 cap with explicit windows sized from measured maxima on complete stores — Crescendo 49,152
(max context 41,433), PAIR 16,384 (12,825): `geometry.MULTITURN_CAP4096_WINDOWS`.
`GeometryCapExceedsBasis` refuses this entry for a driver capping above 2,048.

## 2. Capability controls (gsm8k_0shot, mmlu_letter, xstest_v2, or_bench_hard_1k, alpaca_pair), all cells of a family pooled

Prompts re-rendered with the harness's chat render over the pinned corpora (system = the cell's SI for
prompted cells); completions re-tokenized from the stored generations (+1 EOS, clamped to the 512 cap).
`gsm8k_0shot` stores no generations, so its completion column is the 512 cap (an upper bound);
`mmlu_letter` is a one-position logprob read (`max_tokens` 1).

| task | model family | n | p50 | p90 | p99 | max | p99 prompt | max comp | max_tokens | L / S | frac > L | clamp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gsm8k_0shot | llama8b | 1,000 | 620 | 6,621 | 7,020 | 7,092 | 6,508 | 512 | 512 | 7,168 / 54 | 0.00% | 0.00% |
| mmlu_letter | llama8b | 1,000 | 132 | 6,427 | 6,619 | 6,935 | 6,619 | 0 | 1 | 6,656 / 59 | 0.70% | 0.70% |
| xstest_v2 | llama8b | 1,000 | 243 | 6,411 | 6,917 | 6,922 | 6,409 | 512 | 512 | 7,168 / 54 | 0.00% | 0.00% |
| or_bench_hard_1k | llama8b | 1,000 | 371 | 6,426 | 6,933 | 6,942 | 6,423 | 512 | 512 | 7,168 / 54 | 0.00% | 0.00% |
| alpaca_pair | llama8b | 1,000 | 532 | 6,482 | 6,950 | 7,370 | 6,486 | 512 | 512 | 7,168 / 54 | 0.10% | 0.30% |
| gsm8k_0shot | llama3b | 300 | 646 | 732 | 777 | 820 | 265 | 512 | 512 | 1,024 / 384 | 0.00% | 0.00% |
| mmlu_letter | llama3b | 300 | 158 | 293 | 575 | 663 | 575 | 0 | 1 | 768 / 512 | 0.00% | 0.00% |
| xstest_v2 | llama3b | 300 | 225 | 557 | 644 | 650 | 138 | 512 | 512 | 768 / 512 | 0.00% | 0.00% |
| or_bench_hard_1k | llama3b | 300 | 565 | 654 | 666 | 670 | 154 | 512 | 512 | 768 / 512 | 0.00% | 0.00% |
| alpaca_pair | llama3b | 300 | 521 | 654 | 895 | 1,076 | 512 | 512 | 512 | 1,024 / 384 | 0.33% | 1.00% |
| gsm8k_0shot | gemma4e4b | 1,000 | 621 | 6,781 | 7,200 | 7,274 | 6,688 | 512 | 512 | 7,424 / 52 | 0.00% | 0.00% |
| mmlu_letter | gemma4e4b | 1,000 | 135 | 6,587 | 6,804 | 7,118 | 6,804 | 0 | 1 | 6,912 / 56 | 0.20% | 0.20% |
| xstest_v2 | gemma4e4b | 1,000 | 178 | 6,570 | 7,096 | 7,099 | 6,588 | 512 | 512 | 7,168 / 54 | 0.00% | 0.00% |
| or_bench_hard_1k | gemma4e4b | 1,000 | 229 | 6,586 | 7,107 | 7,118 | 6,602 | 512 | 512 | 7,168 / 54 | 0.00% | 0.00% |
| alpaca_pair | gemma4e4b | 1,000 | 543 | 6,642 | 7,137 | 7,581 | 6,666 | 512 | 512 | 7,168 / 54 | 0.90% | 1.50% |
| gsm8k_0shot | gemma4e2b | 1,000 | 621 | 6,781 | 7,200 | 7,274 | 6,688 | 512 | 512 | 7,424 / 52 | 0.00% | 0.00% |
| mmlu_letter | gemma4e2b | 1,000 | 135 | 6,587 | 6,804 | 7,118 | 6,804 | 0 | 1 | 6,912 / 56 | 0.20% | 0.20% |
| xstest_v2 | gemma4e2b | 1,000 | 183 | 6,570 | 7,096 | 7,099 | 6,588 | 512 | 512 | 7,168 / 54 | 0.00% | 0.00% |
| or_bench_hard_1k | gemma4e2b | 1,000 | 131 | 6,583 | 7,104 | 7,118 | 6,602 | 512 | 512 | 7,168 / 54 | 0.00% | 0.00% |
| alpaca_pair | gemma4e2b | 1,000 | 541 | 6,627 | 7,137 | 7,581 | 6,666 | 512 | 512 | 7,168 / 54 | 0.90% | 1.50% |
| gsm8k_0shot | qwen35_9b | 2,000 | 625 | 6,737 | 7,332 | 7,399 | 6,820 | 512 | 512 | 7,424 / 52 | 0.00% | 0.00% |
| mmlu_letter | qwen35_9b | 2,000 | 135 | 6,729 | 6,946 | 7,449 | 6,946 | 0 | 1 | 7,168 / 54 | 0.30% | 0.30% |
| xstest_v2 | qwen35_9b | 4,500 | 269 | 6,524 | 7,218 | 7,223 | 6,709 | 512 | 512 | 7,424 / 52 | 0.00% | 0.00% |
| or_bench_hard_1k | qwen35_9b | 13,190 | 334 | 6,548 | 7,231 | 7,245 | 6,723 | 512 | 512 | 7,424 / 52 | 0.00% | 0.00% |
| alpaca_pair | qwen35_9b | 8,050 | 542 | 6,729 | 7,258 | 7,645 | 6,770 | 512 | 512 | 7,424 / 52 | 0.09% | 0.16% |
| gsm8k_0shot | qwen35_4b | 2,000 | 625 | 6,737 | 7,332 | 7,399 | 6,820 | 512 | 512 | 7,424 / 52 | 0.00% | 0.00% |
| mmlu_letter | qwen35_4b | 2,000 | 135 | 6,729 | 6,946 | 7,449 | 6,946 | 0 | 1 | 7,168 / 54 | 0.30% | 0.30% |
| xstest_v2 | qwen35_4b | 4,500 | 255 | 6,524 | 7,217 | 7,223 | 6,709 | 512 | 512 | 7,424 / 52 | 0.00% | 0.00% |
| or_bench_hard_1k | qwen35_4b | 13,190 | 276 | 6,552 | 7,232 | 7,249 | 6,723 | 512 | 512 | 7,424 / 52 | 0.00% | 0.00% |
| alpaca_pair | qwen35_4b | 8,050 | 542 | 6,728 | 7,257 | 7,701 | 6,770 | 512 | 512 | 7,424 / 52 | 0.09% | 0.16% |

## 3. Single-shot jailbreak pools at the 4,096 completion cap (pasttense / codeattack / renellm / grok420 pooled), per SI arm × model family

`ALL` = every arm pooled; `fitted_pooled` = the six fitted arms pooled. The prompted constitution /
irrelevant arms carry the 6–7k SI prefix and set the window; a `none` / fitted arm's prompt p99 is
under 800 tokens.

| arm | model family | n | p50 | p90 | p99 | max | p99 prompt | max prompt | p99 comp | max_tokens | L / S | frac > L | clamp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | llama8b | 25,900 | 207 | 6,479 | 7,659 | 11,104 | 6,852 | 7,044 | 4,096 | 4,096 | 7,680 / 51 | 0.95% | 20.00% |
| none | llama8b | 2,590 | 133 | 975 | 1,507 | 4,707 | 621 | 659 | 987 | 4,096 | 1,536 / 256 | 0.97% | 100.00% |
| prompted_safety_refusal | llama8b | 2,590 | 204 | 1,169 | 4,616 | 4,809 | 714 | 752 | 4,096 | 4,096 | 4,864 / 80 | 0.00% | 0.00% |
| prompted_si_constitution | llama8b | 2,590 | 6,003 | 6,959 | 10,408 | 10,590 | 6,498 | 6,536 | 4,096 | 4,096 | 10,496 / 37 | 0.50% | 10.00% |
| prompted_si_irrelevant | llama8b | 2,590 | 6,507 | 7,495 | 10,912 | 11,104 | 7,006 | 7,044 | 4,096 | 4,096 | 11,008 / 35 | 0.42% | 10.00% |
| fitted_pooled | llama8b | 15,540 | 122 | 1,161 | 4,692 | 4,755 | 622 | 659 | 4,096 | 4,096 | 4,864 / 80 | 0.00% | 0.00% |
| ALL | gemma4e4b | 25,900 | 210 | 6,677 | 8,499 | 11,319 | 7,057 | 7,265 | 2,050 | 4,096 | 8,704 / 45 | 0.46% | 20.00% |
| none | gemma4e4b | 2,590 | 169 | 1,985 | 2,579 | 3,807 | 667 | 700 | 2,007 | 4,096 | 2,816 / 139 | 0.39% | 100.00% |
| prompted_safety_refusal | gemma4e4b | 2,590 | 220 | 2,017 | 2,648 | 4,642 | 759 | 792 | 1,945 | 4,096 | 2,816 / 139 | 0.58% | 100.00% |
| prompted_si_constitution | gemma4e4b | 2,590 | 6,192 | 7,897 | 8,634 | 10,785 | 6,699 | 6,732 | 1,968 | 4,096 | 8,704 / 45 | 0.81% | 100.00% |
| prompted_si_irrelevant | gemma4e4b | 2,590 | 6,712 | 8,445 | 9,177 | 11,319 | 7,232 | 7,265 | 2,001 | 4,096 | 9,216 / 42 | 0.81% | 100.00% |
| fitted_pooled | gemma4e4b | 15,540 | 146 | 1,964 | 2,656 | 4,757 | 667 | 700 | 2,085 | 4,096 | 2,816 / 139 | 0.62% | 100.00% |
| ALL | gemma4e2b | 25,900 | 222 | 6,675 | 8,145 | 9,762 | 7,057 | 7,265 | 1,503 | 4,096 | 8,192 / 48 | 0.83% | 20.00% |
| none | gemma4e2b | 2,590 | 358 | 1,535 | 2,036 | 3,281 | 667 | 700 | 1,539 | 4,096 | 2,048 / 192 | 0.93% | 100.00% |
| prompted_safety_refusal | gemma4e2b | 2,590 | 232 | 1,603 | 2,027 | 4,864 | 759 | 792 | 1,378 | 4,096 | 2,048 / 192 | 0.93% | 100.00% |
| prompted_si_constitution | gemma4e2b | 2,590 | 6,182 | 7,468 | 8,185 | 8,833 | 6,699 | 6,732 | 1,569 | 4,096 | 8,192 / 48 | 0.97% | 100.00% |
| prompted_si_irrelevant | gemma4e2b | 2,590 | 6,706 | 8,106 | 8,650 | 9,762 | 7,232 | 7,265 | 1,577 | 4,096 | 8,704 / 45 | 0.85% | 100.00% |
| fitted_pooled | gemma4e2b | 15,540 | 141 | 1,523 | 2,048 | 3,020 | 667 | 700 | 1,479 | 4,096 | 2,048 / 192 | 1.00% | 100.00% |
| ALL | qwen9b | 25,900 | 356 | 6,836 | 8,531 | 11,466 | 7,179 | 7,388 | 1,809 | 4,096 | 8,704 / 45 | 0.69% | 20.00% |
| none | qwen9b | 2,590 | 235 | 1,215 | 2,100 | 4,750 | 669 | 705 | 1,453 | 4,096 | 2,304 / 170 | 0.54% | 100.00% |
| prompted_safety_refusal | qwen9b | 2,590 | 262 | 1,473 | 2,242 | 3,316 | 762 | 798 | 1,521 | 4,096 | 2,304 / 170 | 0.85% | 100.00% |
| prompted_si_constitution | qwen9b | 2,590 | 6,201 | 7,188 | 8,092 | 10,722 | 6,632 | 6,668 | 1,505 | 4,096 | 8,192 / 48 | 0.69% | 100.00% |
| prompted_si_irrelevant | qwen9b | 2,590 | 6,926 | 8,516 | 10,445 | 11,466 | 7,352 | 7,388 | 3,133 | 4,096 | 10,496 / 37 | 1.00% | 100.00% |
| fitted_pooled | qwen9b | 15,540 | 198 | 1,496 | 2,403 | 4,774 | 669 | 705 | 1,793 | 4,096 | 2,560 / 153 | 0.72% | 100.00% |
| ALL | qwen4b | 25,900 | 471 | 6,833 | 8,208 | 11,458 | 7,179 | 7,388 | 1,544 | 4,096 | 8,448 / 46 | 0.57% | 20.00% |
| none | qwen4b | 2,590 | 310 | 1,330 | 1,895 | 4,735 | 669 | 705 | 1,329 | 4,096 | 2,048 / 192 | 0.85% | 100.00% |
| prompted_safety_refusal | qwen4b | 2,590 | 267 | 1,219 | 1,817 | 4,837 | 762 | 798 | 1,111 | 4,096 | 2,048 / 192 | 0.35% | 100.00% |
| prompted_si_constitution | qwen4b | 2,590 | 6,236 | 7,141 | 10,497 | 10,743 | 6,632 | 6,668 | 4,096 | 4,096 | 10,752 / 36 | 0.00% | 0.08% |
| prompted_si_irrelevant | qwen4b | 2,590 | 6,936 | 8,155 | 11,240 | 11,458 | 7,352 | 7,388 | 4,096 | 4,096 | 11,264 / 34 | 0.81% | 10.08% |
| fitted_pooled | qwen4b | 15,540 | 210 | 1,271 | 2,031 | 4,792 | 669 | 705 | 1,496 | 4,096 | 2,048 / 192 | 0.94% | 100.00% |

## 4. Best-of-N (N = 64, 512-token completions), per SI arm × model family

| arm | model family | n | p50 | p90 | p99 | max | p99 prompt | max prompt | max_tokens | L / S | frac > L | clamp |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALL | llama8b | 215,488 | 185 | 6,022 | 6,940 | 7,004 | 6,457 | 6,494 | 512 | 7,168 / 54 | 0.00% | 0.00% |
| none | llama8b | 16,576 | 84 | 379 | 577 | 619 | 93 | 109 | 512 | 768 / 512 | 0.00% | 0.00% |
| prompted_safety_refusal | llama8b | 16,576 | 168 | 214 | 637 | 706 | 186 | 202 | 512 | 768 / 512 | 0.00% | 0.00% |
| prompted_si_constitution | llama8b | 16,576 | 5,967 | 6,296 | 6,456 | 6,496 | 5,970 | 5,986 | 512 | 6,656 / 59 | 0.00% | 0.00% |
| prompted_si_irrelevant | llama8b | 16,576 | 6,525 | 6,945 | 6,981 | 7,004 | 6,478 | 6,494 | 512 | 7,168 / 54 | 0.00% | 0.00% |
| fitted_pooled | llama8b | 149,184 | 131 | 567 | 597 | 621 | 93 | 109 | 512 | 768 / 512 | 0.00% | 0.00% |
| ALL | llama3b | 66,304 | 117 | 227 | 617 | 734 | 197 | 222 | 512 | 768 / 512 | 0.00% | 0.00% |
| none | llama3b | 16,576 | 111 | 369 | 598 | 643 | 118 | 134 | 512 | 768 / 512 | 0.00% | 0.00% |
| prompted_safety_refusal | llama3b | 16,576 | 194 | 277 | 679 | 734 | 206 | 222 | 512 | 768 / 512 | 0.00% | 0.00% |
| fitted_pooled | llama3b | 33,152 | 104 | 141 | 566 | 642 | 118 | 134 | 512 | 768 / 512 | 0.00% | 0.00% |
| ALL | gemma4e4b | 215,488 | 117 | 6,143 | 7,068 | 7,178 | 6,633 | 6,668 | 512 | 7,168 / 54 | 0.01% | 0.05% |
| none | gemma4e4b | 16,576 | 187 | 569 | 594 | 613 | 88 | 103 | 512 | 768 / 512 | 0.00% | 0.00% |
| prompted_safety_refusal | gemma4e4b | 16,576 | 173 | 645 | 681 | 705 | 180 | 195 | 512 | 768 / 512 | 0.00% | 0.00% |
| prompted_si_constitution | gemma4e4b | 16,576 | 6,128 | 6,227 | 6,592 | 6,642 | 6,120 | 6,135 | 512 | 6,656 / 59 | 0.00% | 0.00% |
| prompted_si_irrelevant | gemma4e4b | 16,576 | 6,682 | 7,116 | 7,151 | 7,178 | 6,653 | 6,668 | 512 | 7,168 / 54 | 0.14% | 0.67% |
| fitted_pooled | gemma4e4b | 149,184 | 96 | 312 | 579 | 615 | 88 | 103 | 512 | 768 / 512 | 0.00% | 0.00% |
| ALL | gemma4e2b | 165,760 | 147 | 6,612 | 7,128 | 7,180 | 6,637 | 6,668 | 512 | 7,168 / 54 | 0.04% | 0.07% |
| none | gemma4e2b | 16,576 | 124 | 560 | 589 | 615 | 88 | 103 | 512 | 768 / 512 | 0.00% | 0.00% |
| prompted_safety_refusal | gemma4e2b | 16,576 | 183 | 345 | 670 | 707 | 180 | 195 | 512 | 768 / 512 | 0.00% | 0.00% |
| prompted_si_constitution | gemma4e2b | 16,576 | 6,128 | 6,246 | 6,602 | 6,647 | 6,120 | 6,135 | 512 | 6,656 / 59 | 0.00% | 0.00% |
| prompted_si_irrelevant | gemma4e2b | 16,576 | 6,726 | 7,128 | 7,159 | 7,180 | 6,653 | 6,668 | 512 | 7,168 / 54 | 0.39% | 0.67% |
| fitted_pooled | gemma4e2b | 99,456 | 99 | 339 | 584 | 615 | 88 | 103 | 512 | 768 / 512 | 0.00% | 0.00% |
| ALL | qwen9b | 182,336 | 295 | 6,530 | 7,259 | 7,305 | 6,759 | 6,793 | 512 | 7,424 / 52 | 0.00% | 0.00% |
| none | qwen9b | 16,576 | 306 | 575 | 600 | 622 | 94 | 110 | 512 | 768 / 512 | 0.00% | 0.00% |
| prompted_safety_refusal | qwen9b | 16,576 | 237 | 585 | 681 | 712 | 187 | 203 | 512 | 768 / 512 | 0.00% | 0.00% |
| prompted_si_constitution | qwen9b | 16,576 | 6,149 | 6,530 | 6,558 | 6,585 | 6,057 | 6,073 | 512 | 6,656 / 59 | 0.00% | 0.00% |
| prompted_si_irrelevant | qwen9b | 16,576 | 7,116 | 7,260 | 7,284 | 7,305 | 6,777 | 6,793 | 512 | 7,424 / 52 | 0.00% | 0.00% |
| fitted_pooled | qwen9b | 116,032 | 206 | 570 | 597 | 622 | 94 | 110 | 512 | 768 / 512 | 0.00% | 0.00% |
| ALL | qwen4b | 149,184 | 266 | 6,797 | 7,261 | 7,305 | 6,761 | 6,793 | 512 | 7,424 / 52 | 0.00% | 0.00% |
| none | qwen4b | 16,576 | 368 | 577 | 601 | 622 | 94 | 110 | 512 | 768 / 512 | 0.00% | 0.00% |
| prompted_safety_refusal | qwen4b | 16,576 | 236 | 603 | 683 | 713 | 187 | 203 | 512 | 768 / 512 | 0.00% | 0.00% |
| prompted_si_constitution | qwen4b | 16,576 | 6,130 | 6,526 | 6,557 | 6,583 | 6,057 | 6,073 | 512 | 6,656 / 59 | 0.00% | 0.00% |
| prompted_si_irrelevant | qwen4b | 16,576 | 7,054 | 7,259 | 7,284 | 7,305 | 6,777 | 6,793 | 512 | 7,424 / 52 | 0.00% | 0.00% |
| fitted_pooled | qwen4b | 82,880 | 167 | 565 | 596 | 622 | 94 | 110 | 512 | 768 / 512 | 0.00% | 0.00% |

## 5. Petri audits, every served target call (all cells, all audits), per model family — vLLM usage from the inspect logs

Served at 16384 × 24 with target `max_tokens` = 1024, so any turn whose prompt exceeded 15,360 tokens
was 400-rejected by vLLM and carries no usage: the `rejected (400)` column counts them (284 in total),
and the `incl. rejected` columns enter each at vLLM's 15,361 floor (lower bounds). The routing proxy
passes the 400 through to the auditor, so those audits were judged on truncated transcripts.

| model family | n served calls | p50 | p90 | p99 | max | p99 prompt | max_tokens | L / S | frac > L | clamp | rejected (400) | rejected % | p99 prompt incl. rejected | p99 total incl. rejected | n samples | p99 last-turn prompt | max last-turn prompt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama8b | 13,814 | 2,894 | 7,340 | 11,575 | 15,566 | 11,261 | 1,024 | 11,776 / 33 | 0.85% | 1.42% | 1 | 0.01% | 11,285 | 11,577 | 1,088 | 12,897 | 15,180 |
| gemma4e4b | 18,832 | 4,869 | 11,286 | 15,758 | 20,647 | 15,024 | 1,024 | 15,872 / 24 | 0.85% | 1.29% | 113 | 0.60% | 15,361 | 16,385 | 1,569 | 17,487 | 19,723 |
| gemma4e2b | 7,703 | 4,929 | 10,975 | 15,443 | 19,896 | 14,871 | 1,024 | 15,616 / 25 | 0.86% | 1.22% | 37 | 0.48% | 15,213 | 15,897 | 652 | 16,499 | 19,349 |
| qwen35_9b | 7,190 | 7,102 | 13,358 | 18,385 | 23,209 | 17,714 | 1,024 | 18,432 / 21 | 0.97% | 1.04% | 71 | 0.98% | 17,619 | 18,354 | 650 | 20,513 | 22,185 |
| qwen35_4b | 8,242 | 6,729 | 12,908 | 17,395 | 21,695 | 16,562 | 1,024 | 17,408 / 22 | 1.01% | 1.10% | 62 | 0.75% | 16,532 | 17,364 | 765 | 19,369 | 20,671 |

Not measured: multi-turn attacks against any target other than gemma4e4b; Petri on llama3b.

## 6. tinycap suite (Inspect + EvalPlus on the served route), from the usage ledgers

The two `capability:tiny*` rows below are the measured windows of the loglikelihood family that ran in
the first passes. That family is removed, so no control pools into them now; they stay registered as the
record of the windows those stores ran at.

**Basis** (differs from sections 1–5): a loopback usage ledger (`sisteer.capeval.serving.usage_ledger`)
in front of the routing proxy tokenizes every attempted `/v1/chat/completions` body via `/v1/tokenize`
before forwarding and appends one row regardless of outcome, so a 400-rejected call is in the basis at
its true prompt length (`GeometryBasis.n_rejected`). `scripts/tinycap_geometry_measure.py` pools the
ledgers per sub-entry (nearest-rank percentiles over all attempted calls) and emits the rows below. A
cap-less call is outside the clamp basis (`GeometryBasis.n_capless`); the model-grader traffic (the
routing table's `none` entry) pools under the informational key `capability:grader`, refused by
`resolve_geometry`. The prompt-budget rows are sized to the prompt MAX (`max_prompt_plus_max_tokens`):
their drivers refuse a clamped cap-bound sample, so the window must hold every prompt at the cap.

| key | n calls | rejected (400) | p50 | p99 | max | p99 prompt | max prompt | max_tokens | L / S | frac > L | clamp | clamp at total sizing | sized by |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| capability:tiny | 5,242 | 0 | 518 | 1,665 | 3,039 | 1,664 | — | 1 | 1,792 / 219 | 0.38% | 0.38% | 0.38% | p99_total |
| capability:gen_long | 5,880 | 0 | 852 | 2,789 | 2,945 | 2,349 | — | 2,048 | 4,608 / 85 | 0.00% | 0.00% | 8.67% | p99_prompt_plus_max_tokens |
| capability:gen_long_thinking | 6,422 | 0 | 1,581 | 7,342 | 8,715 | 2,348 | — | 8,192 | 10,752 / 36 | 0.00% | 0.00% | 31.14% | p99_prompt_plus_max_tokens |
| capability:agent | 3,981 | 0 | 517 | 2,297 | 6,050 | 2,173 | — | 1,024 | 3,328 / 118 | 0.08% | 0.93% | 9.07% | p99_prompt_plus_max_tokens |
| capability:agent_thinking | 3,981 | 0 | 831 | 2,832 | 6,703 | 2,175 | — | 4,096 | 6,400 / 61 | 0.03% | 0.93% | 100.00% | p99_prompt_plus_max_tokens |
| capability:grader (informational) | 6,348 | 0 | 347 | 2,150 | 2,436 | 1,722 | — | 0 (cap-less) | 2,304 / 170 | 0.19% | 0.00% | 0.00% | p99_total |
| capability:tiny_prompted (safety SI only; held PROVISIONAL, analytic window below) | 4,420 | 0 | 604 | 1,761 | 3,126 | 1,760 | — | 1 | 1,792 / 219 | 0.68% | 0.68% | 0.68% | p99_total |
| capability:gen_long_math (MATH-500 PROMPT BUDGET, bare + safety SI) | 4,000 renders | 0 | 220 | — | — | 540 | 1,041 | 16,384 | 17,664 / 22 | 0 | 0.00% | — | max_prompt_plus_max_tokens |
| capability:gen_long_math_longsi (MATH-500 under the constitution / irrelevant SIs) | 8,000 renders | 0 | 6,526 | — | — | 7,024 | 7,622 | 16,384 | 24,064 / 16 | 0 | 0.00% | — | max_prompt_plus_max_tokens |
| capability:gen_long_gsm8k_rung (the GSM8K paired-200 ladder's 16,384 rung) | 1,600 renders | 0 | 2,255 | — | — | 2,400 | 2,462 | 16,384 | 18,944 / 20 | 0 | 0.00% | — | max_prompt_plus_max_tokens |
| capability:gen_long_gsm8k_longsi (the paired-200 cells under the constitution SI, 8,192 pass, Llama tokenizer) | 200 renders | 0 | 7,900 | — | — | 7,975 | 8,007 | 8,192 | 16,384 / 24 | 0 | 0.00% | — | max_prompt_plus_max_tokens |
| capability:gen_long_gsm8k_rung_longsi (the same cells' 16,384 rung) | 200 renders | 0 | 7,900 | — | — | 7,975 | 8,007 | 16,384 | 24,576 / 16 | 0 | 0.00% | — | max_prompt_plus_max_tokens |
| capability:gen_long_gsm8k_longsi_f22 (the constitution-prompted cells on gemma / qwen / nemotron, 8,192 pass) | 600 renders | 0 | 8,222 | — | — | 8,305 | 8,338 | 8,192 | 16,640 / 23 | 0 | 0.00% | — | max_prompt_plus_max_tokens |
| capability:gen_long_gsm8k_rung_longsi_f22 (the same cells' 16,384 rung) | 600 renders | 0 | 8,222 | — | — | 8,305 | 8,338 | 16,384 | 24,832 / 15 | 0 | 0.00% | — | max_prompt_plus_max_tokens |
| capability:gen_long_gsm8k_common_f22 (every arm of a GSM8K leg at the constitution-prompted window; a copy of `_longsi_f22`) | 600 renders | 0 | 8,222 | — | — | 8,305 | 8,338 | 8,192 | 16,640 / 23 | 0 | 0.00% | — | max_prompt_plus_max_tokens |
| capability:gen_long_gsm8k_rung_common_f22 (the common class's 16,384 rung) | 600 renders | 0 | 8,222 | — | — | 8,305 | 8,338 | 16,384 | 24,832 / 15 | 0 | 0.00% | — | max_prompt_plus_max_tokens |
| capability:ifeval_2048 (the IFEval battery's pass-1 cap) | 4,328 renders | 0 | 119 | — | — | 201 | 468 | 2,048 | 2,528 / 155 | 0 | 0.00% | — | max_prompt_plus_max_tokens |
| capability:ifeval_4096 (the IFEval ladder's second rung) | 4,328 renders | 0 | 119 | — | — | 201 | 468 | 4,096 | 4,576 / 85 | 0 | 0.00% | — | max_prompt_plus_max_tokens |
| capability:ifeval_8192 (the IFEval ladder's ceiling) | 4,328 renders | 0 | 119 | — | — | 201 | 468 | 8,192 | 8,672 / 45 | 0 | 0.00% | — | max_prompt_plus_max_tokens |

Tasks pooled — tiny: the five `served_ll/*` tasks of the removed loglikelihood family; gen_long / gen_long_thinking:
`evalplus/{humaneval,mbpp}_plus` (+ `:open` under thinking-ON), `inspect/{gsm8k, gsm8k_paired200,
hellaswag, math, mmlu_0_shot, mmlu_letter_valid, or_bench_hard_1k}`; agent: `inspect/bfcl`.
`capability:tiny_prompted` is PROVISIONAL with an ANALYTIC window (`GeometryBasis.analytic`): the
measurement carried only the ~90-token safety SI, so p99 prompt 1,664 + `si_irrelevant` 6,560 +
template delta 5 + `max_tokens` 1 → L = 8,448 / S 46; it is refused by `resolve_geometry` and served
for measurement only by `pass1_geometry`. The prompt-budget entries render every prompt offline through
the task's own solver chain under each pinned tokenizer (`scripts/tinycap_oos_prompt_budget.py` for
MATH-500 and GSM8K, `scripts/sdfbase/ifeval/measure_ifeval_geometry.py` for IFEval); a system turn adds
a constant to every render, so a class max is the bare max + the SI's tokens, and `L` is the 16-block
floor of `max_prompt + max_tokens`, rounded up to 256 where the headroom allows (≤ 512). Per-model S
through `ModelBudget.seqs_at` at the measured budgets (§7):

| entry | L | gemma4e4b | qwen35_4b | glm46vflash | nemotron3nano4b | llama8b |
|---|---:|---:|---:|---:|---:|---:|
| capability:gen_long_math | 17,664 | 33 | 42 | 44 | 44 | — |
| capability:gen_long_math_longsi | 24,064 | 24 | 30 | 32 | 32 | — |
| capability:gen_long_gsm8k_rung | 18,944 | 31 | 39 | 41 | 41 | — |
| capability:gen_long_gsm8k_longsi / rung_longsi | 16,384 / 24,576 | — | — | — | — | 29 / 19 |
| capability:gen_long_gsm8k_longsi_f22 / rung_longsi_f22 | 16,640 / 24,832 | 35 / 23 | 43 / 29 | — | 47 / 31 | — |
| capability:ifeval_2048 / 4096 / 8192 | 2,528 / 4,576 / 8,672 | 233 / 128 / 68 | 256 (hybrid cap) / 162 / 85 | — | 256 (hybrid cap) / 171 / 90 | 187 / 103 / 54 |

MATH-500's window is chosen by the cell's PROMPT CLASS (`contracts.MATH500_PROMPT_CLASS_OF_SI`); the
GSM8K long-SI entries are selected by prompt class and family scope (`g8k_cells.geometry_for_cell`);
the IFEval entries through `contracts.IFEVAL_GEOMETRY_ENTRY_BY_CAP[cap]`. None of these is a tinycap
sub-entry: `pass1_geometry` refuses them and they resolve through `resolve_geometry`. Pass 1 serves
every tinycap key at the certified footprint (a measured key as a measurement re-run that extends its
ledger); pass 2 serves the measured window and refuses `tiny_prompted` (provisional) and `grader`
(informational).

## 7. Per-model batched-token budgets

The windows above are one factor of the served geometry; the other is S, which every launcher
resolves from the served model's batched-token budget (`geometry.BUDGET_BY_MODEL` / `budget_for`;
`ModelBudget.seqs_at(L)` = `budget // L`, capped at `HYBRID_SEQS_CAP` = 256 for the hybrid-recurrent
families, whose recurrent state costs memory per slot). The composer pins the no-chunking identity
`max_num_batched_tokens = max_num_seqs × max_model_len` exactly, so vLLM's profiling forward runs over
the whole batched-token count and its activation peak has to fit beside the KV cache. Each budget is
the largest `S × L` whose launch passed `vllm_server.kv_coverage_problem` at readiness on one H200 at
`gpu_memory_utilization` 0.90 (`scripts/served_decode_sweep.py --phase budget`: a ladder walked upward
to the first refusal, then one bisection); `scripts/geom_sweep_analyze.py` folds the results into the
marked paste block of `geometry.py`.

| model | budget (S × L) | basis window | source |
|---|---:|---:|---|
| llama8b | 475,136 (29 × 16,384) | 16,384 | measured on the lens stack by the serving gate's readiness ladders: S 29 / 30 pass, 31 refused (KV 477,856 < 507,904); 29 is one rung under the literal largest pass, keeping ≥ 7.7k KV tokens of slack on the smaller 139.80-GiB H200s. The bare-engine sweep's 458,752 (56 × 8192) is its history |
| llama3b | 327,680 | 16,384 | PROVISIONAL ceiling (`_PROVISIONAL_BUDGETS`, `source="provisional-conservative"`): the bare-engine sweep measured 655,360 at L = 8192, halved because its Llama sibling failed on the lens stack at a budget the same sweep had passed; not measured, not observed to fail |
| gemma4e4b | 589,824 (72 × 8192) | 8,192 | measured; stood up and passed at the same number on the lens stack |
| gemma4e2b | 786,432 (96 × 8192) | 8,192 | measured |
| qwen35_9b | 196,608 | 16,384 | PROVISIONAL ceiling: the sweep's 589,824 died in engine init on the lens stack at 36 × 16384 (`CUBLAS_STATUS_EXECUTION_FAILED` in a bf16 GEMM, workspace exhaustion) |
| qwen35_4b | 741,888 (42 × 17,664) | 17,664 | measured AT the MATH-500 window on the production lens stack (`scripts/sbatch/geom_budget_window.sbatch --budget-len 17664`): 20 / 32 / 40 / 42 × 17,664 passed, 44 refused (`kv_under_coverage`, 41.05× < 44). The bare-engine sweep's 720,896 (88 × 8192) is its history |
| glm46vflash | 786,432 (96 × 8192) | 8,192 | measured |
| nemotron3nano4b | 786,432 (96 × 8192) | 8,192 | measured; hybrid, S capped at 256 |

A provisional row sizes exactly like a measured one, but `budget_for` logs a warning naming the
ceiling, `ResolvedGeometry.describe()` prints `PROVISIONAL`, and every launch record stamps
`budget_source` / `provisional`. Replacing a row: re-run the budget probe for that model on the lens
stack at the window it will be served at, replace its `_MEASURED_BUDGETS` row and delete its
`_PROVISIONAL_BUDGETS` entry (a ceiling that no longer lowers the number it shadows fails the module's
import-time check).

**Modal-only measurements, documented here and not landed as table rows.** The served-stack probe
(`scripts/modal/tinycap_budget_measure.py`, Modal H200s, `gpu_memory_utilization` 0.90) measured
qwen35_4b at **786,432** (96 × 8192; S 48 / 64 / 80 / 96 passed KV coverage at readiness, 112 and 128
`kv_under_coverage`) and llama8b at **393,216** (24 × 16384, adopted conservatively: 28 × 16384 also
passed at readiness but is the geometry whose runtime activations once OOM'd). The shared table is
host-blind, so those numbers stay in the lane's `budget.json` outputs; a Modal launcher that wants the
larger S reads them from there (qwen35_4b's 786,432 would give S 161 at the 4,864 window against the
table's 152). On Modal the standup's KV-derived shrink (`geometry.kv_shrink` in `endpoint.json`) may
lower S at readiness; the window never moves.
