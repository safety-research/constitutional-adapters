# full22 sampled: vendor-preset decoding with a token-cap ladder

Scripts that generate the static jailbreak families and the over-refusal controls under each served model's
vendor-recommended sampler and climb a token-cap ladder for items that hit the completion cap. The registry adapters
`sampled_static.py` and `sampled_controls.py` (`src/sisteer/registry/adapters/`) read these stores at each cell's final
rung for the paper's static jailbreak (Fig. 3, 11) and over-refusal (Fig. 6, 14, the benign axis of Fig. 7) numbers.

Most readers want [`reproduce/`](../../../../reproduce/README.md): stage 05 renders these configs, stages 06
(`--family static`, `--family controls`) and 09 submit `target_sampled.sbatch`, and stage 08 grades. `full22` is the
paper's evaluation battery ([`../README.md`](../README.md)). Generation needs Slurm; the rest can run from a clone.
[`DESIGN.md`](DESIGN.md) states the design contract and [`LADDER_RULES.md`](LADDER_RULES.md) the ladder's rules.

## What is measured

| readout | items | first cap | rungs | grader |
|---|---|---|---|---|
| `pasttense`, `codeattack`, `renellm`, `grok420` ("Pliny" in the figures) | 259 goals each | 4,096 | 8,192, 16,384 | StrongREJECT v1 on `claude-haiku-4-5`, threshold 0.5 |
| `xstest`, `or_bench_hard_1k` | XSTest 450 prompts generated, the 250 safe ones scored; OR-Bench-Hard 1,319 | 512 | 2,048, 8,192, 16,384 | `claude-sonnet-5`: full refusal, partial refusal or full compliance |

Ordinary tags draw two seeded trials (0 and 1), each a complete run graded on its own, never pooled; the Qwen
seed-replicate views `qwen35_4b_s1`, `_s2`, `_s3` and the six seeded control views
`qwen35_4b_dm1_{ctllora64,ctlrelu64}s{1,2,3}` draw one. The ladder is a second pass into separate roots; its headline
output is the matched-pair label-flip rate between first cap and final rung.

## Running it

```bash
# repo root; --check exits 1 on drift; --tags lists the scope
python scripts/sdfbase/full22/sampled/make_sampled_configs.py
sbatch -J f22s-<key>-<job> scripts/sdfbase/full22/sampled/target_sampled.sbatch <key> <phase> <wave_sha> [nshards=9] [cells=all] [view=main]
```

An empty `cells` means all; `view` is `main`, `it` or a dose view; `<TAG>` is `<key>` for `main`, else `<key>_<view>`;
`T` is the trial, 0 or 1. Add your site's `--qos`/`--output`.

Generation runs on a Slurm GPU node and needs a clean clone at `$SISTEER_EVAL_REPO_ROOT/<wave_sha>/` (default `$SHARED/sisteer-eval-repo`), the object
bank `$SHARED/exp/objects/sdfbase/full22/<model>/cells_view/<view>` ([`../make_bank_view.py`](../make_bank_view.py)),
and the model snapshot in the HF cache. The `renellm` corpus is the copy at `SISTEER_HELDOUT_CORPUS_DIR` (default
`$SHARED/exp/heldout_corpora`) when there is one, else the paper's corpus as the repository ships it (`data/heldout_corpora/renellm.jsonl`).
`$SHARED` is `SISTEER_SHARED_ROOT` (set `SISTEER_CLUSTER_LABEL` with it), which the front door exports from the cluster's site block.

Each phase, its job name and what it waits for:

- `static_base_t<T>[:fam,...]`, `f22s-<key>-sb<T>`: the 4,096 pass over the four families or a comma subset.
- `controls_t<T>`, `f22s-<key>-ctl<T>`: the 512 pass over `xstest` and `or_bench_hard_1k`, then its ladder.
- `static_esc_t<T>[:fam,...]`, `f22s-<key>-se<T>`, the second block, after every `sb<T>` job of the model and trial:
  the 8,192 rung for each (family, cell) with at-cap rows.
- `capability_t<T>`, `f22s-<key>-cap<T>`, after `ctl<T>` and `se<T>`: MMLU (below).
- `ladder_extend_t<T>`, `f22s-<key>-le<T>`, after `cap<T>`: the 16,384 decision, redraws and ceiling report for both
  ladders.
- Llama (a roster `caps` block) instead runs `static_esc16_t<T>[:fam,...]`, `f22s-<key>-s16k<T>`, after `se<T>`: the
  gated 16,384 static rung; no `cap<T>` or `le<T>`.
- By hand: `controls_esc_t<T>[:ceiling]`, the controls ladder over an existing complete 512 root, up to an optional
  ceiling rung (2048, 8192 or 16384).

`submit_sampled.sh <wave_sha> [--targets k1,k2] [--nshards 9] [--offline]` checks configs, checkout, snapshot and
object view, then prints every sbatch line in this order with its `--time`; it runs nothing, and the blocks carry
no Slurm dependencies. Stop a run with `$WAVE_ROOT/STOP` (wave-wide) or the job's `sampled/serving/<...>/STOP`, then
`scancel` by name.

## Presets and seeds

| key | served model | preset | source |
|---|---|---|---|
| `gemma4e4b` | google/gemma-4-E4B-it | temperature 1.0, top_p 0.95, top_k 64 | snapshot `generation_config.json` |
| `qwen35_4b` | Qwen/Qwen3.5-4B | temperature 0.7, top_p 0.8, top_k 20, min_p 0.0, presence_penalty 1.5, repetition_penalty 1.0 | model card "Best Practices", non-thinking; no `generation_config.json` ships |
| `nemotron3nano4b` | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | temperature 1.0, top_p 1.0 | NVIDIA's general-chat row; shipped top_p 0.95 recorded as `shipped_generation_config`, a `deliberate_departure` |
| `llama8b` | NousResearch/Meta-Llama-3.1-8B-Instruct | temperature 0.6, top_p 0.9 | snapshot `generation_config.json` |
| `glm46vflash` | zai-org/GLM-4.6V-Flash | temperature 0.8, top_p 0.6, top_k 2, repetition_penalty 1.1 | model card "Evaluation Settings"; not in the paper |

`configs/sdfbase_sampled_decode.yaml` holds every key, the worked example `qwen35_4b_demo` (the `qwen35_4b` row under
its own key) included; a view inherits its model key's row, and every row records its `decode` block.

Seed rule. `sampling_seed = int.from_bytes(sha256(f"{seed_base}|{trial}|{readout}|{item_key}").digest()[:4], "big")`,
with `readout` the family or control name and `item_key` = `f"{goal_id}|{sample_idx}"` (static) or the dataset item id
(controls), so every cell draws common random numbers. `seed_base` is 20260909; the seed-replicate and `dm1` views
above use the served object's training seed (1, 2, 3). A rung redraw reuses the item's seed. vLLM's per-request seed
fixes the sampler, but bf16 logits vary with batch composition: a re-run reproduces the draw distribution, not the
bytes.

## Stores

The wave root is `$SHARED/exp/sdfbase_full22_native` (serving path `lora_native`, the default) or
`$SHARED/exp/sdfbase_full22` (`F22_SERVING_PATH=lens`), `F22_WAVE_ROOT` overriding either; pass it to
`mirror_sampled.sh --remote-root`. Under it: static passes under
`transcripts_results/sdfbase_full22_<TAG>_sampled_t<T>[_cap<R>]/<family>/<cell>/`, controls rungs as complete roots
`sampled/controls_cap<R>/<TAG>_t<T>/`, ladders under `sampled/ladders/<TAG>_t<T>/` and serving dirs under
`sampled/serving/`. Rungs write to new roots, never in place and never into a greedy store.

## The ladder

The static 8,192 rung redraws every at-cap row of a (family, cell) that has any; every other step is gated: a (cell,
readout) escalates only when its truncated share exceeds the 1% at-cap trigger, truncated meaning at the cap and neither
a repetition loop, repetitive text nor a boundary draw (its sibling trial ended well under the cap). Excluded items are
copied forward, so every rung root holds every item. Exit kinds: `exit_zero`, `exit_le_1pct`
(`exit_le_1pct_with_residual` when at-cap items remain), `exit_ceiling` and `transport_refused`.

Each controls rung (`served_capability_run.py`) also redraws `--floor-sample 200` seeded below-cap items as the noise
floor for the flip rate; the static rungs draw none. `ladder_extend_t<T>` reclassifies every rung so far, writes
`decision_16384.json`, redraws each promoted pair at 16,384 and writes the ceiling report `ceiling_16384.json`. Before
each rung, `hop_timeout_guard.py` refuses a binding transport timeout (`upstream_timeout_s` or
`proxy_upstream_timeout_s` in `endpoint.json`, or the client's) under twice the rung's worst-case decode time: a
transport kill reads as a short completion.

## Grading

Attacks are graded by `grade_static_sampled.py` and controls by `served_capability_judge.py`, on any host holding the
mirrored stores. Both take their key from `sisteer.vault`: `ANTHROPIC_API_KEY_LOWPRIO`, else the secret-manager CLI
that `SISTEER_SECRET_CMD` names, or `ANTHROPIC_API_KEY` with `SISTEER_SECRETS_MODE=env`.

```bash
# repo root
export F22S_DEST=<local grading directory>
scripts/sdfbase/full22/sampled/mirror_sampled.sh --host <ssh alias> --remote-root <wave root> --dest $F22S_DEST [--keys k1,k2]
scripts/sdfbase/full22/sampled/grade_sampled.sh <static|controls-pass1|controls-pass2|all> --dest $F22S_DEST [--keys ...] [--plan]
```

`mirror_sampled.sh` also copies each 512 root, still unjudged, to `controls_sampled/flip_pass/cap512/` for pass 2, so
run it before pass 1. `grade_sampled.sh` runs each unit under `systemd-run --user`; without a user systemd instance,
`--plan` prints each unit's commands to run as plain processes.

Attacks: one StrongREJECT grade per trial and per rung root, never pooled; the grader refuses an incomplete store and
writes `two_si_<family>_aggregate.json` and `grading_provenance.json` under `<grades-root>/<private_run>/`.
`F22_WORKTREE` overrides the repository root it imports from.

Controls: `scripts/served_capability_judge.py` runs two passes per (tag, trial). Pass 1 judges the 512 root with the
shared verdict cache. Pass 2 judges the pristine copy of the 512 root and each (cell, control)'s final-rung store (the
highest rung with a complete `gen_stats.json` for that control, `final_rung.py`) with an empty private cache
(`CAPEVAL_JUDGE_CACHE=<fresh sqlite>`), so every item is a live re-ask and the judge's own nondeterminism is the floor
the truncated group's flip rate must clear.

`compare_greedy_sampled.py`, `analysis_sampled.py` and `decode_effect.py` read the mirror and write ids, counts and
labels, never text.

## The instruct-direct views

`gemma4e4b_it` and `qwen35_4b_it` run the same contract on the `it` view (objects fit and served on the instruct
checkpoint) over `none`, the constitution LoRA and ReLU steer, the control and difference LoRAs, and the prompted
baselines `prompted__safety_refusal` (Safety SI) and `prompted__si_constitution` (Constitution SI). The view is the
sixth positional behind an empty cells argument:

```bash
sbatch -J f22s-gemma4e4b_it-sb0 scripts/sdfbase/full22/sampled/target_sampled.sbatch gemma4e4b static_base_t0 <sha> 9 "" it
scripts/sdfbase/full22/sampled/submit_sampled.sh <wave_sha> --targets gemma4e4b_it,qwen35_4b_it      # both views
```

Jobs are `f22s-<TAG>-sb<T>`, `-ctl<T>` and `-se<T>`. An `it` view's greedy grades sit under two roots (the prompted
safety baseline apart from the rest), which `compare_greedy_sampled.py` maps per cell (`by_cell`); its
`greedy-baseline` subcommand writes these views' baseline to `greedy_baseline_it.json`. These views give the
first sampled-vs-greedy comparison of prompting (a system instruction, no object) beside fitted objects on the same
served model; no rank statistic across cells is reported for them, only per-cell contrasts against `none`.

## Capability readouts

`capability_t<T>` (`f22s-<key>-cap<T>`) runs MMLU by `letter_logprob_argmax` under a thinking-off standup into
`sampled/mmlu/<TAG>_t<T>`. It refuses a cell list that includes `prompted__si_constitution` on any `main` or `it` view
(the instruction, about 6,000 tokens, needs a larger window), so pass `cells` without it. These MMLU roots are a
canary, not the paper's MMLU result: the letter-logprob argmax does not depend on the sampler, so trial-to-trial
agreement measures server nondeterminism, and `analysis_sampled.py mmlu-agreement` marks its output `canary_record`.

`readouts.gsm8k` in `configs/sdfbase_sampled_decode.yaml` describes a sampled, thinking-on GSM8K readout at an 8,192
cap into `sampled/gsm8k_thinking_on_cap8192/<TAG>_t<T>`; the roster retires it on every view, so no job runs it, and the
static families and judged controls ignore the sub-block and read the top-level preset. Qwen's row there is the card's
thinking row, temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 1.5, repetition_penalty 1.0; the other
models reuse their preset. It has no matched greedy arm, so the `capability_greedy` phase is refused.
