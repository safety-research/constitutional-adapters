# Round-2 export builders

The scripts here built the round-2 export: a long table with one row per measurement of the paper's evaluation campaign and every quality flag as a column. The registry's `r2_export` adapter reads that table as one input; the paper's figures read the registry.

A reader of the paper does not need this directory unless tracing a registry row sourced from `r2_export`: the pipeline is in [`reproduce/`](../../reproduce/README.md), the figures in [`paper/notebooks/`](../../paper/notebooks/README.md) and the data in [`docs/registry/`](../../docs/registry/README.md). Nothing here runs from a clone: every script reads campaign state from outside the repository, and no pipeline stage or library module imports them.

## Vocabulary

`r2` is paper round 2 and `sdfbase` the campaign's name. `full22` is the evaluation campaign's label, not a count. `psr` (cell ids) and `gated` (the `arm` column) both name the ReLU steer, the paper's ReLU-family adapter. A view is one served checkpoint and fit lane: `qwen35_4b_it` (instruct-direct fits) or `qwen35_4b` (base-fit objects served on the instruct checkpoint). A cell is one served configuration within a view (an object or a prompted system instruction); `none`, also called `bare`, is the unsteered model. A lane is a view at one dose (1.0 or 0.5); `delta` contrasts are within a lane against its own `none`.

## What is here

| Directory | What its scripts produced |
|---|---|
| `build_export/` | `build_export.py`: the export; `make_notebook.py`: the figure notebook, whose figure numbers are its own, not the paper's ([README](build_export/README.md)) |
| `cap4096/` | `build_transfer_stats_cap4096.py`: the multi-turn file at the 4,096-token reply cap |
| `llama8b/` | `build_transfer_stats_full22_llama8b.py`: the Llama-3.1-8B file, four views |
| `petri/` | `petri_stats_sidecar.py`, `petri_stats_newcells.py`: Petri statistics for cells audited after a view's main audit ([README](petri/README.md)) |
| `modal_stage/` | `agg_petri.py`: `petri_agg.json`, per-cell Petri means over every audited root |
| `bon_qwenctl/` | `file_into_export.py`: filed two best-of-N cells into an export file |
| `deps/` | the shared export-file builder and statistics helpers |

The scripts are committed; of their outputs, only two small JSON files under `build_export/` are.

## How the export is assembled

The lane builders (`deps/w7_rest/`, `cap4096/`, `llama8b/`) turn per-attack grader aggregates into lane files in one schema, `models.<view>.cells.<cell>.<readout>`, keyed by reply cap (`decode_cap_tokens`), so 512- and 4,096-token multi-turn readings coexist. `build_export.py` merges them with Petri statistics, over-refusal judge records and degeneration measures, pinning every input's sha256 in `_sources.json`. A value it cannot read from a file is NaN with a `status` saying why, never a guess or a zero.

Proportions carry a Wilson 95% interval where the integer count is filed; the graded unsteered over-refusal rows file none. Contrasts are paired against the lane's own `none`: attacks by a 10,000-draw bootstrap over goals (seed 0) with exact McNemar, at the same cap; Petri by an exact two-sided sign test over scenarios; over-refusal by a paired Wald interval with McNemar over items paired across the two cells. The cap-4096 builder instead files each cell's 4,096-token reading against its own 512-token one (`vs_cap512`), a cross-instrument reading rather than a defence effect (see [`build_export/`](build_export/README.md)), and leaves the contrast against `none` empty.

## How the registry consumes it

[`src/sisteer/registry/adapters/r2_export.py`](../../src/sisteer/registry/adapters/r2_export.py) files one `run` per (producer key, object, eval, instrument): measured rows as `run_metric`, degeneration shares (`at_cap_share`, `loop_share`, `degenerate_share` and variants) as `run_degeneration`, deltas against a `none` anchor in the export as `run_contrast`, `pending` rows as queued runs and other unmeasured rows as `disposition`s. The paper reads the registry's paper views (`v_paper_long` and companions) and the snapshot under `paper/snapshot/`, never this export.
