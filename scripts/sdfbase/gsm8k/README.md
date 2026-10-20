# GSM8K on the served objects: Slurm launchers

Worker scripts for the paper's GSM8K capability evaluation on Slurm: one single-GPU job per (model, view, cell) serves the intervention through vLLM, runs the Inspect GSM8K task and writes a result store. GSM8K is one of four evaluations on the capability axis of Figure 6 (Qwen) and Figure 14 (all four models).

Most readers want [`reproduce/`](../../../reproduce/README.md): stage 06 (`06_eval_waves.sh --model <key> --view <view> --family gsm8k`) submits `gsm8k_thinking.sbatch` for every cell of a view; stage 09 adds the dose views with `--evals ...,gsm8k`. The paper's numbers are in the registry snapshot [`paper/notebooks/`](../../../paper/notebooks/README.md) reads.

A row is one (model key, view, cell) job; a wave, the rows submitted together. A `fitted` row serves an object (LoRA or steer, pinned by sha256), not `none` or a prompt; `declared` marks the two `safety_refusal__*` objects, whose thinking-on render is declared to the server rather than certified live; a `held` row keeps its matrix index, never submitted. `<TAG>` is the model key for view `main`, `<key>_<view>` otherwise.

## What it needs

A Slurm cluster with a shared root, a clean checkout at the wave commit under it (a detached worktree of the cluster's repository mirror), an object bank view (`reproduce/05_stage_roster.sh` stages it from `objects/`) and the [`full22`](../full22/README.md) serving venv:

```bash
git -C <the repo mirror> worktree add --detach <the eval-repo root>/<sha> <sha>
```

The job scripts read the shared root from the job environment: `SISTEER_SHARED_ROOT` and `SISTEER_CLUSTER_LABEL` (optionally `SISTEER_EVAL_REPO_ROOT`, `SISTEER_HF_HOME`), which the front door passes from the cluster's site block and a hand submission prefixes to every `sbatch` line below. `plan` takes each cluster's QoS and log directory from its site block; `g8k_cells.CLUSTERS` lists the clusters a plan can name.

Every job plans the 200 items offline from `openai/gsm8k` and `madrylab/gsm8k-platinum` in `HF_DATASETS_CACHE`, refusing by name if either is missing. Prepare it once per cluster from the repository root: plan the task online with `HF_HUB_OFFLINE=0` set explicitly (the harness defaults it to 1),

```bash
HF_HUB_OFFLINE=0 PYTHONPATH=src python -c "import sisteer.capeval.tiers.inspect_served as i; i.plan_task('gsm8k_paired200', grader_model=None, thinking='off')"
```

then alias `openai/gsm8k`'s hash-named directory to the offline loader's name: `ln -s default-<hash> $HF_DATASETS_CACHE/openai___gsm8k/default-data_dir=main` (platinum's `main` needs none). Nemotron jobs also need the fused Mamba kernel attestation [`../math500/nemo_backend_probe.sbatch`](../math500/nemo_backend_probe.sbatch) writes.

## Running it

From a clone, no GPU (repository root):

```bash
PYTHONPATH=src:scripts python scripts/sdfbase/gsm8k/g8k_dry_run.py --stage pre [--matrix llama]
python scripts/sdfbase/gsm8k/g8k_cells.py list --matrix default
python scripts/sdfbase/gsm8k/make_gsm8k_overlays.py --check    # per-TAG overlays: campaign config + manifest with that TAG's cells
```

On the cluster, from the checkout at `<sha>` (40-hex), with `WAVE_ROOT` = `<shared root>/exp/gsm8k_paired200_paper_objects` (suffixed `_<matrix>` for other matrices):

1. `g8k_cells.py row --matrix <name> --index N --sh` prints a row's facts as shell assignments. Write its prompt budget, one CPU artifact per (model, budget class, cap) every job checks:

   ```bash
   PYTHONPATH=src python scripts/tinycap_oos_prompt_budget.py --task gsm8k_paired200 --arm <G8K_THINKING> --cap <8192|16384> \
       --geometry-eval <G8K_GEOMETRY_EVAL> --config configs/sdfbase/gsm8k/sdfbase_full22_<TAG>.gsm8k.yaml --model-key <key> \
       [--system-prompt-cell <prompted cell>] --tree-sha <sha> --out $WAVE_ROOT/budgets/<key>/<G8K_BUDGET_CLASS>/<cap>/oos_prompt_budget.json
   ```

2. `g8k_cells.py plan --sha <sha> --cluster <cluster> --wave <wave> --max-concurrent <W> [--matrix <name>]` prints `sbatch --array` lines per wall-time class and never submits; `--cluster` is an entry of `g8k_cells.CLUSTERS`. A single row submits as

   ```bash
   sbatch -J g8k-<key>-<view>-<cell> --time=<time> scripts/sdfbase/gsm8k/gsm8k_thinking.sbatch <key> <view> <cell> <8192|16384> <sha> <fitted 0|1> <declared 0|1> [matrix]
   ```

   with `--time`, `fitted` and `declared` from `list`: 1 h for `none` and safety-prompted rows, 1.5 h for object and constitution-prompted rows, 2 h for GLM object rows (Llama: 45 min and 1 h). Stop a wave with `touch $WAVE_ROOT/STOP` before `scancel`.

3. After the 8,192 rows finish, `sbatch -J g8k-ladder-plan scripts/sdfbase/gsm8k/ladder_plan.sbatch <sha> "" "" [matrix]` prints the 16,384 rung lines with their own `--time`; they need step 1's budgets at `--cap 16384`.

4. From the repository root, `scripts/sdfbase/gsm8k/pull_wave.sh --subtree controls_<wave> --src HOST:PATH [--serving-src HOST:PATH ...] <sha7>` copies a wave's stores and shared inputs to `./pull_<sha7>` (`--serving-src`: the same root on each other cluster that served rows, for its launch records). Pull `controls` too (the `none` legs), then write the 200 item ids and analyse:

   ```bash
   PYTHONPATH=src python -c "import json, sisteer.capeval.tiers.inspect_served as i; json.dump(i.gsm8k_paired200_selection_ids(), open('ids.json', 'w'))"
   PYTHONPATH=src:scripts python scripts/tinycap_gsm8k_paired200.py --gen-root pull_<sha7> --sha <sha7> --matrix configs/sdfbase/gsm8k/<matrix file> \
       --waves published,<wave> --ids ids.json --overlap overlap.json --out report.json
   ```

   `--matrix` is the matrix the wave ran (repeatable: several read as one design); `--waves`, the waves whose rows are mandatory; `--overlap`, the record `scripts/gsm8k_overlap_check.py --retired-root <graded controls store> --stock-root pull_<sha7>/overlap --ids ids.json --sha <sha7> --out overlap.json` writes from the thinking-off `none` legs at cap 512 (`overlap_off.sbatch <key> <sha>`). `src/sisteer/registry/adapters/gsm8k_p200.py` folds the report into the registry ([`docs/registry/README.md`](../../../docs/registry/README.md)).

Each job runs `scripts/served_capability_run.py generate --controls gsm8k` at the row's cap, records the resolved sampler (`scripts/sdfbase/record_resolved_sampler.py`) and shuts the server down.

## The instrument

inspect_evals' stock `gsm8k` task (10-shot, step-by-step reasoning ending in `ANSWER: N`, numeric match) narrowed to 200 test items located by question text from a 200-row sample of GSM8K-Platinum and scored against the original openai/gsm8k gold ([`docs/tinycap_interfaces.md`](../../../docs/tinycap_interfaces.md) names the contract, row task `inspect/gsm8k_paired200`). Each item is drawn once (k = 1) at an explicit 8,192-token completion cap. Thinking is requested on every call to a model with a thinking switch, under the vendor's `thinking_card` row of `configs/sdfbase/math500/sampling_rows.yaml` ([the MATH-500 page](../math500/README.md#the-instrument) tabulates them); Llama-3.1 has none and runs `--thinking not_available` under its shipped `card` row.

The per-item decode seed is `sha256(item_id NUL sample_idx)[:4] mod 2^31-1`; the seed-replicate matrices use `sha256(f"{seed_base}|inspect-replicate|{item_id}|{sample_idx}")[:4] mod 2^31-1` with `seed_base` 1, 2, 3 for views `s1`/`s1p`, `s2`/`s2p`, `s3`/`s3p`.

A draw cut at the cap has no answer and is scored wrong; the analysis also reports the reading with those items excluded and files a verdict only when both agree, contrasting each cell against its view's `none` with paired-item bootstraps. A cell with over 1% of items cut re-draws them at 16,384 tokens into a sibling store and reports the merged passes; no rung lies above 16,384 ([`docs/cap_ladder.md`](../../../docs/cap_ladder.md)).

## Views, cells and matrices

Views: `main` serves base-checkpoint fits through the instruct checkpoint (Nemotron and GLM, with one checkpoint, fit and serve there); `it`, instruct-direct fits; `s1`..`s3`, the seed-replicate fits; `s1p`..`s3p`, each seed's paired difference ReLU steer beside its own `none` (unrelated to `reproduce/models.yaml`'s `s1` view). Cells are the roster's config ids:

|Cell id|Paper term|
|---|---|
|`none`|unsteered model|
|`prompted__safety_refusal`, `prompted__si_constitution`|safety SI, constitution SI|
|`constv2ctl2-full__lora_r4_ce`, `constv2-mix-full__lora_r4_ce`|control LoRA, constitution LoRA|
|`constv2-mix-full__lora_r4_ce_minus_ctl`|difference LoRA (the constitutional adapter)|
|`constv2ctl2-full__psr_ce_all`, `constv2-mix-full__psr_ce_all`|control and constitution ReLU steer|
|`constv2-mix-full__psr_ce_all_minus_ctl`|difference ReLU steer (paired write)|
|`arditi_allpos`|refusal steer (all positions, raw coefficient 0.3)|
|`alphasteer`, `arditi`, `safety_refusal__*`, `*_minus_ctl_distilled`|not in the paper|

A matrix (JSON under `configs/sdfbase/gsm8k/`) lists one wave family's rows; the launchers take its name as a trailing positional, and `plan` refuses the complete `published` wave.

|Matrix (file)|Rows|Launchable waves|
|---|---|---|
|`default` (`matrix.json`)|53: 42 `published` on four models in `main` and `it` (4 held: Gemma's and Qwen's `it` constitution LoRA and ReLU steer, re-fit in `itdoc`), 5 prompted, 6 steer baselines|`ext`, `steer`|
|`llama` (`matrix_llama.json`)|14 Llama-3.1-8B rows, thinking not available|`steer`|
|`itdoc` (`matrix_itdoc.json`)|8 instruct-direct rows on Gemma and Qwen|`itdoc`|
|`qwen_seedrep`, `arditi_seedrep` (`matrix_qwen_seedrep.json`, `arditi_rows_qwen_seedrep.json`)|36 (6 held) and 3 Qwen seed-replicate rows|`qwen_seedrep`|
|`arditi_allpos` (`matrix_arditi_allpos.json`)|1: Qwen refusal steer on `it`|`allpos`|
|`paired_allmodels` (`paired_allmodels_rows.json`)|4 ReLU-steer rows on Gemma, Llama, Nemotron|`paired_allmodels`|
|`frontdoor_<key>` (`matrix_frontdoor_<key>.json`, written by stage 05)|a model's `reproduce/models.yaml` views|`frontdoor_<key>`|

## Outputs

Under the wave root, `controls[_<wave>]/<TAG>/<cell slug>/<sampling mode>/` (`controls/` for `published`; `__` written `--`; mode `thinking_card`, or `card` on Llama) holds the store (`cells/<cell id>/`, with the Inspect `.eval` logs), `usage_ledger.jsonl`, `resolved_sampler.json` and `cap_ladder.json`, with the 16,384 re-draw beside it as `<sampling mode>__rung16384/`; `budgets/`, `rung16384/` and `serving/` hold the budgets, rung subsets and launch records.
