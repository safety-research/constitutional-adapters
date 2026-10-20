# data/ — the small committed inputs the pool builders and probes read

Under `data/sdfbase/` everything is ids, pins and provenance — no corpus text; the corpora themselves live under
`corpora/` as gzip shards, and the pool builders under `scripts/sdfbase/` combine the two. `data/tinycap/` holds the
capability battery's small committed artefacts, real model completions included (its table is below). `data/petri/` holds
the frozen Petri scenario bundle the paper's Petri audits ran on, and `data/arditi/` the identity, without the text, of
the instructions the refusal direction was fit on. `data/heldout_corpora/` holds the ReNeLLM evaluation corpus, under its
own terms rather than the repository's MIT licence (its section is at the end).

## `data/sdfbase/` (30 files)

| family | files | what | written by | read by |
|---|---|---|---|---|
| held-out split | `heldout_split.json` + `.provenance.json` | ~5 % of each part of the constitution corpus carved into a held-out set at leakage-safe granularity (advice: the `(principle_id, prompt_type_id)` A3-call group; stories: `subcategory_id`), sha-ranked under the salt `sdfbase-heldout-v1`, plus the frozen 128-doc held-out EVAL subset; ids only | `scripts/sdfbase/build_heldout_split.py` (the default arguments reproduce it byte for byte) | `build_pools.py`, `build_ctl2_pools.py`, `build_probe_set.py`, `reproduce/02_prepare_model.sh --step pools`, tests |
| per-seed held-out splits | `heldout_split.s{1,2,3}.json` + `.provenance.json` | the same carve under the seed salts `sdfbase-heldout-v1-s<s>` at the seed replicates' 0.085 fraction, with the contrastive probes' source documents PINNED into the held-out side (`pins` block: probe-file shas, pinned source ids, pinned vs seed-drawn groups) | `build_heldout_split.py --pin-groups-from`, driven by `scripts/sdfbase/seedrep/build_seed_pools.py` | `seedrep/build_seed_pools.py` (templated `heldout_split.s{s}.json`), `seedrep/check_probe_pins.py`, `tests/test_seed_splits.py` |
| control (ctl2) splits | `ctl2_heldout_split.json` (+ `.s{1,2,3}`) + `.provenance.json` | the matching ~5 % carve of the control corpus (`control-scale-20260903-r2`), ids only; the provenance records the control shards' sha256s and `release_bytes_pinned` | `scripts/sdfbase/build_ctl2_pools.py build --split-out` | `build_ctl2_pools.py`, `seedrep/build_seed_pools.py`, `reproduce/02_prepare_model.sh`, tests |
| probe v1 | `probe_v1.jsonl` + `.manifest.json`; `probe_v1.raw.jsonl` + `.gen_report.json`; `probe_v1.verified.jsonl` + `.verify_report.json` | the first contrastive probe set: 450 items, each a prefix with a constitution-consistent and a constitution-violating continuation (`claude-opus-5` generation, `claude-haiku-4-5` forced-label verification, then length / 8-gram-overlap / dedup gates); `raw` and `verified` are the intermediate stage outputs with their reports | `scripts/sdfbase/build_probe_set.py {plan,generate,verify,freeze}` | `probe_v1.jsonl` + manifest: `build_probe_set.py`, `scripts/sdfbase/probe_score.py`, `tests/test_probe_set.py`, `tests/test_probe_scorer.py`; the four `raw` / `verified` files and their reports: nothing |
| probe v2 | `probe_v2.jsonl` + `.manifest.json`; `probe_v2.raw.jsonl` + `.gen_report.json`; `probe_v2.verified.jsonl` + `.verify_report.json` | the token-balanced re-issue (453 items; the continuations balanced under the gemma-4 and Qwen3.5 tokenizers) | `build_probe_set.py` | `probe_v2.jsonl`: `seedrep/build_seed_pools.py`, `seedrep/check_probe_pins.py`, `tests/test_probe_v3_frozen.py`, `tests/test_sdfbase_derived_app.py`, `tests/test_seed_splits.py`; the manifest: `tests/fixtures/sdfbase_derived_100k_golden.json`; the four `raw` / `verified` files and their reports: nothing |
| probe v3 (genre-matched) | `probe_v3_genre.jsonl` + `.manifest.json` | the genre-matched probe (662 items = 432 v2 items re-issued as refusal-shaped `R` and compliance-shaped `C` variants, 230 with both), the probe the fitted cells are scored with | `build_probe_set.py` (revision recorded in the manifest's `generator`) | `seedrep/build_seed_pools.py`, `seedrep/check_probe_pins.py`, `scripts/r2_export/.../r2_paths.py`, `tests/test_probe_v3_frozen.py`, `tests/test_sdfbase_derived_{app,full}.py`, `tests/test_seed_splits.py` |

The paper's committed objects (`objects/`) do not read these files; they are inputs to the pool
builders (`reproduce/02_prepare_model.sh --step pools`) that reproduce the training pools a fit
reads, and to the probe scorer. Read by nothing on the tree: `probe_v1.raw.jsonl`,
`probe_v1.verified.jsonl`, `probe_v2.raw.jsonl`, `probe_v2.verified.jsonl` and their four
`*_report.json` sidecars (the intermediate stages behind the two frozen probe files).

## `data/tinycap/`

| path | what | written by | read by |
|---|---|---|---|
| `ifeval/results.template.md` | the IMMUTABLE skeleton of the IFEval results record (double-brace placeholders; byte-pinned) | hand-written | `scripts/tinycap_ifeval_battery.py` (`RESULTS_TEMPLATE`, rendered to `out/ifeval_canonical/results.md`), `scripts/tinycap_ifeval_record_prose.py` (fills the prose slots); pinned by `tests/test_ifeval_analysis_final_readout.py` |
| `thinking_smoke/<family>/` | real completions of the thinking-split smoke, one directory per model family (gemma4, nemotron_h, glm4v, qwen3_5) | `scripts/sdfbase/math500/smoke_thinking_probe.py` | `tests/test_think_split_real_completions.py` via `sisteer.capeval.tinycap.contracts.THINKING_SMOKE_COMPLETIONS_DIR` |
| `no_switch_smoke/llama/` | real completions of the no-switch diagnostic smoke (Llama has no thinking switch) | `scripts/sdfbase/gsm8k/smoke_not_available.sbatch` | `tests/test_no_switch_smoke_real_completions.py` via `contracts.NO_SWITCH_SMOKE_COMPLETIONS_DIR` (`g8k_cells.SMOKE_COMMITTED_DIR_REL`) |

## `data/petri/`

| path | what | written by | read by |
|---|---|---|---|
| `scenarios/8812ba083bb5473c/` | the frozen Petri scenario bundle, byte for byte as frozen (85 files): `bundle_manifest.json` and, for each of the six behaviours, `scenarios/understanding.md`, three judge-dimension files and ten auditor seeds. Generated once by `bloom scenarios` with `anthropic/claude-opus-4-8` on 2026-08-29; the directory name is the content hash `configs/capeval/petri-scenarios.lock.yaml` pins. Every scenario file keeps its benchmark canary header, so none of it may enter a training corpus | `sisteer.capeval.petri.scenarios.freeze_scenarios` (once; a frozen set is never regenerated) | `reproduce/lib/stage.py` (the frozen inventory's Petri scenario ids), `python -m sisteer.capeval.petri.scenarios install` (the grading host's 0700 store, through `scripts/registry/grade_daemon.py` and `scripts/sdfbase/full22/petri_drive_full22.sh`), `tests/test_petri_published_bundle.py` |

## `data/arditi/`

| path | what | written by | read by |
|---|---|---|---|
| `fit_prompts.manifest.json` | the identity of the 320 instructions (128 training and 32 validation per side) the fit behind the refusal-direction objects in `objects/qwen35_4b_it/` read: the upstream commit of `andyrdt/refusal_direction`, the four split files' sha256s, the seed (42), the draw order and sizes, each set's sha256 and the whole document's (`f2f0decf…`); no instruction text. `python -m sisteer.data.arditi_fit_prompts fetch` (or `scripts/arditi/reproduce_fit_prompts.py fetch`) re-draws `fit_prompts.json` from the public upstream into the third-party cache and checks every hash; `reproduce --out data/arditi/fit_prompts.json` also writes a copy here, which git ignores. Records under `configs/sdfbase/` that cite `data/arditi/fit_prompts.json` at commit `5cbc8da7…`, a development commit this repository does not contain, mean this document, identified by its sha256 | written from the original document's non-prompt fields | `sisteer.data.arditi_fit_prompts` (re-draws and checks), `scripts/arditi/fit_refusal_direction.py` (the `arditi:*` sets), `reproduce/lib/third_party.sh`, tests |

## `data/heldout_corpora/`

| path | what | written by | read by |
|---|---|---|---|
| `renellm.jsonl` + `renellm.jsonl.README` | the paper's frozen ReNeLLM corpus, byte for byte (777 rows: 259 HarmBench and JailbreakBench behaviours, each rewritten once by x-ai/grok-4.5 through OpenRouter and placed in ReNeLLM's three scenario frames). It is NOT under the repository's MIT licence: [`LICENSES/ReNeLLM-corpus-terms.txt`](../LICENSES/ReNeLLM-corpus-terms.txt) permits research and evaluation use and forbids training on it. Its hashes and counts are in `configs/heldout/renellm.paper_corpus.json` | `scripts/build_heldout_corpus.py` on 2026-08-21 (that script now refuses `renellm`: a rebuild is a new sample) | `src/sisteer/eval/attacks/renellm.py` (evaluation replay, when the site keeps no copy in `SISTEER_HELDOUT_CORPUS_DIR`), `scripts/renellm_corpus_check.py`; every training-side reader refuses it (`src/sisteer/eval/attacks/eval_only.py`) |
