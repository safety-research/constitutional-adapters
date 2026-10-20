# MATH-500 on the served objects: Slurm launchers

Worker scripts for the MATH-500 capability evaluation on Slurm: one single-GPU job per (model, cell, sampling row) serves the intervention through vLLM, runs the Inspect `math` task on the 500 problems and writes a result store. No `reproduce/` stage submits them.

The paper reports no MATH-500 number: a draw cut inside its reasoning is unscored, so its accuracy is conditional on the scored draws and not comparable with evaluations that score a cut draw as wrong. The notebook draws no figure from it; the registry carries it as `paper_status: appendix` (`configs/registry/evals.yaml`).

These launchers and the GSM8K ones in [`../gsm8k/`](../gsm8k/README.md) share one execution core, `../full22/served_thinking.sh`, and one vocabulary (row, wave, `held`; a `fitted` row's cell is a LoRA or steer object, not `none` or prompted).

## What it needs

As for GSM8K: a Slurm cluster with a shared root, a checkout at the wave commit under it (checkout command and site prefixes on [the GSM8K page](../gsm8k/README.md#what-it-needs)), an object bank view staged by `reproduce/05_stage_roster.sh` and the [`full22`](../full22/README.md) serving venv.

`HuggingFaceH4/MATH-500` and `DigitalLearningGmbH/MATH-lighteval` are prepared once per cluster by an online `load_dataset` of each into `HF_DATASETS_CACHE`; the jobs read them offline, refusing by name when the cache has no prepared MATH-500.

## Running it

From a clone, no GPU (repository root):

```bash
PYTHONPATH=src:scripts python scripts/sdfbase/math500/m5t_dry_run.py --stage pre [--wave ext1]
python scripts/sdfbase/math500/cells.py list
python scripts/sdfbase/math500/make_math500_overlays.py --check
```

On the cluster, from the checkout at `<sha>`, with `WAVE_ROOT` = `<shared root>/exp/math500_thinking_on_full22`:

1. `cells.py list` prints each row's facts, including its serving-window entry `geom=`. Write the prompt budget, one CPU artifact per (model, budget class) every job checks (class `none` for bare and object cells, else the prompted cell id); Llama budgets pass `--cap 16384` because the `not_available` arm has no cap table, while the thinking-on arm's defaults to the task's 16,384:

   ```bash
   PYTHONPATH=src python scripts/tinycap_oos_prompt_budget.py --task math500 --arm <on|not_available> [--cap 16384] --geometry-eval <geom> \
       --config configs/sdfbase/math500/sdfbase_full22_<key>.math500.yaml --model-key <key> [--system-prompt-cell <prompted cell>] \
       --tree-sha <sha> --out $WAVE_ROOT/budgets/<key>/<class>/oos_prompt_budget.json
   ```

2. Nemotron only: `sbatch -J m5t-nemo-probe scripts/sdfbase/math500/nemo_backend_probe.sbatch <sha>` writes the kernel attestation.

3. Before launching rows beside an existing `none` leg, with the published `controls` pulled as in step 5,

   ```bash
   PYTHONPATH=src:scripts python scripts/sdfbase/math500/m5t_compat_preflight.py --published-root <published pull>/controls --wave-sha <sha> \
       --out $WAVE_ROOT/preflight [--wave <wave>]
   ```

   compares what the wave tree will serve for each new row with that model's `none` leg (item set, sampler, revision, thinking policy, cap, template, client lock) into `compat_<key>.json`. Exit 0 is the only pass (1 fail: that model's rows stay held; 2 missing input, or a dirty or wrong tree; 3 undetermined).

4. `cells.py plan --sha <sha> --cluster <primary|overflow> --wave <wave> --max-concurrent <W> [--single-array]` prints `sbatch --array` lines per wall-time class and never submits. `--cluster` is a role (`primary`, `overflow`) or a cluster `cells.CLUSTERS` lists, and each line takes that cluster's QoS and log directory from its site block; `qwen_seedrep` is refused on `primary`, the complete `published` wave and held rows everywhere. A single row submits as

   ```bash
   sbatch -J m5t-<key>-<cell> --time=<time> scripts/sdfbase/math500/math500_thinking.sbatch <key> <cell> <sampling_mode> <sha> <fitted 0|1> [<view>]
   ```

   with `--time` and `fitted` from `list`: 6 h for `ext1`, `steer` and `allpos` rows, 1.5 h for Llama rows, 4 h for seed-replicate rows. Stop a wave with `touch $WAVE_ROOT/STOP` before `scancel`.

5. From the repository root, `scripts/sdfbase/math500/pull_wave.sh --subtree controls_<wave> --src HOST:PATH <sha7>` copies a wave's stores and shared inputs to `./pull_<sha7>`; pull `controls` too (the `none` legs), then:

   ```bash
   PYTHONPATH=src:scripts python scripts/tinycap_math500_thinking.py --gen-root pull_<sha7> --sha <sha7> --waves published,<wave> --math500-ids plan --out report.json
   ```

   `--waves` names the mandatory waves; `src/sisteer/registry/adapters/math500.py` folds the report into the registry.

Each job runs `scripts/served_capability_run.py generate --controls math500 --thinking <mode> --sampling <row> --samples 1 --math500-retry-max-tokens 0 --max-tokens-by-task math=16384` (plus `--on-cap-by-task math=16384` on Llama), writes `row_identity.json` and, outside the published wave, `resolved_sampler.json` beside the store, and shuts the server down.

## The instrument

inspect_evals' `math` task restricted to the 500 MATH-500 problems (sample ids `math_<md5(problem)[:8]>`), scored by its `expression_exact_match_sympy` scorer on the boxed answer. Thinking is requested on every call to a model with a thinking switch; each item is drawn once (k = 1) in one pass at a 16,384-token completion cap, no retry pass, no rung above it ([`docs/cap_ladder.md`](../../../docs/cap_ladder.md)). The per-item decode seed is `sha256(item_id NUL sample_idx)[:4] mod 2^31-1`; the seed-replicate rows use `sha256(f"{seed_base}|inspect-replicate|{item_id}|{sample_idx}")[:4] mod 2^31-1` with `seed_base` 1, 2, 3 for views `s1`/`s1p`, `s2`/`s2p`, `s3`/`s3p`.

A draw that reaches the cap may still be scored: one cut inside its answer is scored, one cut inside the think block stays unscored. The analysis reports accuracy over the scored draws beside the at-cap and unscored (`n_censored / n_items`) shares, contrasts every cell against its model's `none` with paired-item bootstraps, and gives sharp bounds counting every unscored item as wrong or as right.

Each model decodes under its vendor's thinking-mode row (`thinking_card`); Qwen's `general` and the `battery` rows are sensitivity rows, reported separately, never pooled with the primary. Llama-3.1, with no thinking switch, runs `--thinking not_available` under its shipped `card` row. `configs/sdfbase/math500/sampling_rows.yaml` pins every key; an absent key is absent from the request.

|Model key|`thinking_card`|Other rows|
|---|---|---|
|`gemma4e4b`|temperature 1.0, top_p 0.95, top_k 64||
|`qwen35_4b`|temperature 0.6, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0|`general`: temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 1.5, repetition_penalty 1.0; `battery`: temperature 0.7, top_p 0.8, top_k 20, min_p 0.0, presence_penalty 1.5, repetition_penalty 1.0|
|`nemotron3nano4b`|temperature 1.0, top_p 0.95|`battery`: temperature 1.0, top_p 1.0|
|`glm46vflash` (not in the paper)|temperature 0.8, top_p 0.6, top_k 2|`battery`: adds repetition_penalty 1.1|
|`llama8b`|`card`: temperature 0.6, top_p 0.9||

## The matrix

`cells.py list` prints all 87 rows (54 matrix and 33 seed-replicate rows); a row's index is its Slurm array id.

|Rows|Wave|Content|
|---|---|---|
|0–23|`published`|four models × {`none`, `prompted__safety_refusal`, `constv2-mix-full__lora_r4_ce_minus_ctl`} under `thinking_card`, plus Qwen's three `general` and nine `battery` rows (Qwen, GLM, Nemotron)|
|24–32|`ext1`|Gemma, Qwen, Nemotron × {`prompted__si_constitution`, `constv2ctl2-full__lora_r4_ce`, `constv2-mix-full__lora_r4_ce`}|
|33–40|`steer`|AlphaSteer and prefill-only Arditi refusal-steer rows on all four models; not in the paper|
|41–46|`llama`|Llama-3.1-8B × the six non-steering arms, thinking not available|
|47–52|`itdoc` (held)|instruct-direct LoRA rows on Gemma and Qwen|
|53|`allpos`|Qwen × `arditi_allpos`, the refusal steer at every position (raw coefficient 0.3), view `it`|
|54–86|`qwen_seedrep`|Qwen seed replicates: `s1`..`s3` × nine cells (`none`, both prompted SIs, the four LoRA and ReLU-steer cells, the difference LoRA, `arditi_allpos`); `s1p`..`s3p` × {`none`, paired difference ReLU steer}|

Cell ids map to paper terms as in [the GSM8K table](../gsm8k/README.md#views-cells-and-matrices); published rows are view `main`.

## Outputs

Under the wave root, `controls[_<wave>]/<key>/<cell slug>/<sampling mode>/` (`controls/` for `published`; `<key>_<view>` for seed-replicate views; `__` written `--`) holds the store (`cells/<cell id>/`, with the Inspect `.eval` logs), `usage_ledger.jsonl`, `row_identity.json` and, outside the published wave, `resolved_sampler.json`. `budgets/`, `nemo_backend/`, `preflight/` and `serving/` hold the budgets, Nemotron attestation, compatibility records and server launch records; [`docs/registry/README.md`](../../../docs/registry/README.md) describes the registry holding every published number.
