# Corpus generator

The code that generated the paper's two synthetic training corpora: the constitution corpus and a register-matched control corpus, each composed of two parts, difficult advice and stories. It is a member of the repository's `uv` workspace; the `corpus` dependency group installs it as the console script `corpusgen`. The corpora are committed as gzip shards under [`../corpora/`](../corpora/), one directory per part; most readers need only those.

## Reading the corpora

`reproduce/tools/reassemble_corpus.py` rebuilds one part from its shards with the Python standard library alone, checking shard and file sha256s and record counts against `corpora/<part>/MANIFEST.json`. From the repository root:

```bash
python3 reproduce/tools/reassemble_corpus.py --list
python3 reproduce/tools/reassemble_corpus.py constitution_advice --verify-only
python3 reproduce/tools/reassemble_corpus.py constitution_stories --out /tmp/constitution_stories.jsonl
```

The reassembled files are 68, 411, 67 and 427 MB in the table's order. On a configured site, `reproduce/02_prepare_model.sh --model qwen35_4b_demo --step corpora` reassembles all four parts on the cluster and stages the reference tokenizers for fitting. Each manifest also lists sidecars: the filter report, per-record constitution section tags (constitution corpus), and for the control corpus the register-edit report and the placebo import scan (`*.import-scan.json`; report-only, never a drop).

## The two corpora

Fitted objects name their training mix in their registry id (`constv2-mix-full__lora_r4_ce`): `constv2-mix-full` is the constitution corpus's training side, advice and stories together, after a 5% held-out carve, `constv2ctl2-full` a draw from the control corpus's cut to the same token count. Token counts are the minimum over three reference tokenizers (Llama 3.1, Gemma 3, Qwen3; [`../corpora/TOKENIZERS.json`](../corpora/TOKENIZERS.json)).

The constitution corpus is generated from Claude's constitution, the control corpus from the "careful work" placebo described below. Each is two parts, difficult advice and stories, and the four directories under `corpora/` are those parts:

| corpus | part | records | tokens | what a record is |
|---|---|---|---|---|
| constitution | `constitution_advice` | 1,960 | 5.03M | difficult advice: advice sought in a hard, ethically ambiguous situation, answered in line with the constitution (`system`, `user`, `assistant`) |
| constitution | `constitution_stories` | 16,321 | 40.05M | pre-training-style fiction about a generic, unnamed AI embodying the constitution (`text`) |
| control | `control_advice` | 1,995 | 4.88M | the same shape, about everyday practical tasks |
| control | `control_stories` | 16,294 | 40.94M | the same shape, about an AI doing ordinary competent work |

The training text is an advice record's `user` and `assistant` fields plus its `system` prompt when present (1,287 of the 1,960 constitution advice records have one), or a story's `text`. The other fields are generation metadata: drafts, critiques, the sampled principle or category, `scores`, per-tokenizer `token_counts` and `lineage`.

The control corpus is a placebo: the same pipeline with the constitution replaced by a values-free handbook of careful practical method across many crafts (`constitution-control-v2/placebo.md`, about 18,500 words) and the templates by the `prompts-control/` overlay.

## How the corpora were generated

Generation ran on the Anthropic Batch API with `claude-opus-5`. Each part is a fanout of stages, each one batched wave plus up to two repair waves for missing or unparseable slots:

- Advice, nine stages: principles (`a1`), prompt types (`a2`), scenarios (`a3`), prompt draft (`b1`), diagnostic response (`b2`), prompt rewrite (`b3`), response (`c`), response rewrite (`d`), scoring (`e`).
- Stories, five stages: categories (`s1`), subcategories (`s2`), stories (`s3`), rewrite (`s4`), scoring (`s5`).

Most stages have a same-named jinja2 template under `prompts/advice/` and `prompts/stories/`; `b2` is built in code and `c` uses `c_injection.md` and `c_length.md`. `prompts-control/` overrides templates for the control run, whose 25 advice principles were hand-reviewed (`configs/control-scale-principles.json`) and seeded with `corpusgen seed-principles` rather than sampled.

The offline filter chain keeps records at or above score floors (constitution: alignment 9, realism 8, story consistency 8, story quality 8; control: 8 on every axis). It drops records flagged as naming a specific AI, stories flagged for real-world factual claims, advice flagged as refusals, any record whose training text mentions Claude or Anthropic, and sanity failures (empty or placeholder text, stories outside their length bounds, stray pipeline tags, degenerate repetition), then removes MinHash near-duplicates and caps register families.

Register here means surface style: recurring phrasings and story beats, matched by regex families. The constitution chain caps three (a "safeguard" story beat, two system-prompt openings) under 1% of the corpus. The control chain caps fourteen, each at the constitution corpus's measured share where that corpus carries the register (verdict-label response openers 20%, "Deployment:" system-prompt labels 14.5%) and under 1% where it does not, so the control matches rather than lacks those registers.

The control finals were downsampled to 2,000 advice and 16,300 stories before the edit pass, whose re-scoring and re-filtering then dropped 5 advice and 6 stories.

Two slices of the constitution run were regenerated on the live API after review: the advice of two principles whose sampled wording made the assistant itself the subject of the dilemma (`p35`, `p36`, reworded by hand in `configs/advice-regen-principles.json`; `claude-opus-5`) and two story lanes `claude-opus-5` had refused (`claude-opus-4-8`). Three further advice records were dropped by hand.

The control corpus then went through a register-matching edit pass on the live API with `claude-opus-5`: the generator model edited spans of control records so that register markers occur at the constitution corpus's rates, and the edited records were re-scored and re-filtered. Stories took one pass (3,937 of 16,294 records edited; four stock beats rewritten down to the constitution corpus's rate), advice two (768 of 1,995 edited; the second pass reworded the "sanity check" openers the first had added).

The constitution finals come from `configs/production.yaml` plus the regeneration configs `advice-regen.yaml` and `stories-regen.yaml`, assembled by `production-v2.yaml`; the control finals from `control-scale.yaml` and its edit pass `control-scale-r2.yaml`.

## Re-running generation

Generation draws a new sample, not the paper's data. The entry point is `reproduce/01_make_corpora.sh`, run from the repository root with a site configuration (copy `reproduce/site.example/` to `~/.config/sisteer/` and edit it; [`../docs/configuration.md`](../docs/configuration.md)). Its offline paths (`--setup`, `--test`, `--preflight`, `--dry-run`) are tested; its generation phases wrap the `corpusgen` commands the paper's corpora were made with but have not themselves been run end to end.

```bash
reproduce/01_make_corpora.sh --setup                      # uv sync --frozen --inexact --group corpus
reproduce/01_make_corpora.sh --test                       # the offline suite: no network, no keys
reproduce/01_make_corpora.sh --preflight --corpus all
reproduce/01_make_corpora.sh --corpus control --phase all --dry-run   # print every command; submit nothing
reproduce/01_make_corpora.sh --corpus control --phase advice
```

`--preflight` downloads the three tokenizer repositories into `$HF_HOME` and stages the pinned files under `corpus_generator/data/tokenizers`; `TOKENIZERS.json` pins each repository, revision, file set and per-file sha256, and staging refuses a missing or mismatched snapshot rather than substituting one. Llama 3.1 and Gemma 3 are gated: export an `HF_TOKEN` whose account has accepted their licenses.

`--corpus constitution|control|all` selects a family; `--phase advice|stories|filter|register-edit|all` is the generation request. `advice` and `stories` run `corpusgen run` on the Batch API, printing the request count first (`production.yaml` plans 24,441 advice and 45,051 story requests; `control-scale.yaml` 18,276 and 45,051); one family takes days. Runs resume from `corpus_generator/data/runs/<run_id>/` (gitignored; `corpusgen status --config CFG` shows per-wave state), and a new `run_id` gives an independent sample.

The driver does not seed the control's reviewed principles: run `seed-principles` once per control run id before its first generation command, or `corpusgen run` samples fresh ones. The commands below run from the repository root with paths relative to `corpus_generator/`, which `uv run --directory` enters.

```bash
CORPUSGEN="uv run --directory corpus_generator --frozen corpusgen"
$CORPUSGEN seed-principles --config configs/control-scale.yaml --principles configs/control-scale-principles.json
```

`filter` assembles the finals under `corpus_generator/data/`. For the control family it is offline, and `register-edit` then runs the edit pass on the live API at up to 200 concurrent requests (`--edit-engine batch` moves it to the Batch API). The driver runs pass 1 only; the advice finals need the second pass by hand:

```bash
$CORPUSGEN register-edit plan --config configs/control-scale-r2.yaml --pass 2
$CORPUSGEN register-edit run --config configs/control-scale-r2.yaml --pass 2
$CORPUSGEN register-edit assemble --config configs/control-scale-r2.yaml --pass 2 --corpus advice
$CORPUSGEN import-scan --config configs/control-scale-r2.yaml --corpus all --suffix -r2
```

For the constitution family `filter` runs the offline chain and writes the v1 finals, then exits at its first regeneration call (`regen-advice` without the `--principles` it requires); the stories regeneration, the regen filters, `assemble-v2` and `report` do not run. The block below finishes the paper run's repairs by hand. A new sample needs them only where it shows the same defects (reread its sampled principles and per-lane refusal counts first), and the three `--drop-id` records are the paper run's:

```bash
$CORPUSGEN regen-advice --config configs/advice-regen.yaml --principles configs/advice-regen-principles.json
$CORPUSGEN regen-stories --config configs/stories-regen.yaml --phase draft    # prints the per-lane refusal verdict
$CORPUSGEN regen-stories --config configs/stories-regen.yaml --phase finish   # --skip-lane <lane> for a lane the probe abandons
$CORPUSGEN filter --config configs/advice-regen.yaml --corpus all
$CORPUSGEN filter --config configs/stories-regen.yaml --corpus all
$CORPUSGEN assemble-v2 --config configs/production-v2.yaml --corpus all \
  --add advice=data/advice/final-advice-regen-20260901.jsonl --add stories=data/stories/final-stories-regen-20260901.jsonl \
  --drop-run production-20260831 --drop-principle p35 --drop-principle p36 \
  --drop-id adv-p35-t03-s03 --drop-id adv-p35-t03-s06 --drop-id adv-p35-t03-s08 --suffix -v2
$CORPUSGEN report --config configs/production.yaml --suffix -v2 --extra-run advice-regen-20260901 --extra-run stories-regen-20260901
```

Secrets have three modes: `SISTEER_SECRETS_MODE=env` (the shipped `config.env`) reads `ANTHROPIC_API_KEY` for every credential label; `op`, the mode when the switch is unset, runs a secret-manager CLI wrapper for each credential; `wif` (GCP workload identity federation, on a VM whose service account is federated with the Anthropic API) uses no key at all — the client federates over its credential label's ids file under `SISTEER_WIF_ENV_DIR` (the `batch` label's for generation, the `lowprio` label's for smoke and live edits), and `ANTHROPIC_API_KEY` is never read. Export the switch (and, under `env`, the key) for direct `corpusgen` use. No key is read from a file ([`../docs/secrets.md`](../docs/secrets.md)). `corpusgen smoke --config CFG` checks the credentials with at most five live requests and a tiny real batch. A credential the API rejects (a 401 / 403, or under `wif` a federation identity the token exchange refuses) stops a live wave at the first rejected request: nothing is stamped, the rejected requests leave no cache, and a re-run after the fix resumes from the outcomes obtained before the stop; a batch wave stops the same way at a rejected create, poll, results read or list (its later pages included). Inside a Modal container the `wif` constructors refuse by name before any request (the GCP identity token does not exist there).

To generate from another source document, point a run config's `constitution_dir` at a directory shaped like `constitution/` (`constitution.md`, `sections.json`, `SNAPSHOT.json`) and `prompts_overlay` at an overlay like `prompts-control/`. No command builds that snapshot: derive `sections.json` with `section_markdown` in `src/corpus_generator/placebo.py`, and give `SNAPSHOT.json` the sha256 of `constitution.md` and `n_sections` (`constitution.load` and `verify_placebo` check them); `constitution-control-v2/` is a worked example, its `constitution.md` a symlink to `placebo.md`. `corpusgen fetch-constitution` rebuilds the constitution's own snapshot from the published page.
