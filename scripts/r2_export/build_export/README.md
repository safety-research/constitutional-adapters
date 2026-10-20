# Round-2 export builder

`build_export.py` built the round-2 export: one row per measurement of the paper's evaluation campaign, every quality flag a column. It does not run from a clone, and a reader of the paper does not need it: the paper's numbers come from the results registry and the snapshot under [`paper/snapshot/`](../../../paper/snapshot/); [the parent README](../README.md) covers the export's role.

## Files

Committed: `build_export.py`; `make_notebook.py` (the figure-notebook generator); and, from one build, `sdf_base_steering_r2_schema.json` (column documentation) and `sdf_base_steering_r2_sources.json` (the inputs' sha256 pins and cross-checks). Both lack four columns the current builder adds (`pairing_allowance`, `anchor_engine`, `anchor_view`, `anchor_root`).

Run the builder from inside a checkout, not as a copied file (it imports `sisteer.eval.anchor_pairing` from the clone's own `src/` and refuses otherwise), into an empty directory: the default `--out`, this one, would overwrite the committed JSON.

```bash
# working directory: an empty output directory
CLONE=/path/to/clone
uv run --python 3.12 --exclude-newer 2026-09-01T00:00:00Z --with pandas --with pyarrow --with numpy --with pyyaml \
  python "$CLONE/scripts/r2_export/build_export/build_export.py" --out .
```

`make_notebook.py` writes `sdf_base_steering_r2_figures.ipynb` to the working directory. Executed there, the notebook reads the parquet and, for Figure 14, `sdf_base_steering_r2_transfer_vs_direct_items.parquet`, written by `scripts/sdfbase/full22/transfer_vs_direct.py --out .` under the same pins plus `--with scipy`. Its kernel name is not registered, so execution names `python3`:

```bash
# working directory: the output directory above
uv run --python 3.12 --exclude-newer 2026-09-01T00:00:00Z --with nbformat python "$CLONE/scripts/r2_export/build_export/make_notebook.py"
uv run --python 3.12 --exclude-newer 2026-09-01T00:00:00Z --with pandas --with pyarrow --with seaborn --with matplotlib --with scipy \
  --with ipykernel --with nbconvert python -m nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 sdf_base_steering_r2_figures.ipynb
```

## Reading the table

Each row is one measurement of one cell on one eval. The schema documents every column and [the full22 README](../../sdfbase/full22/README.md) decodes cell ids; what changes how a value reads:

**Metrics.** `asr` is the share of goals with any response scoring above `threshold` on StrongREJECT v1 (on best-of-N rows, any of the `bon_n` draws), judged by claude-haiku-4-5 (headline threshold 0.5, headline `bon_n` 75). `over_refusal_rate_terminated`, the headline over-refusal reading, covers items terminated in both cells (judge claude-sonnet-5); `over_refusal_rate` covers all replies. `xstest` has 250 prompts and `or_bench_hard_1k` 1,319. `gsm8k_controls512` is a greedy 512-token GSM8K, not the paper's 200-item paired instrument. A NaN `value` means not measured, never zero.

**Contrasts.** `delta` is the cell minus its own `none` (methods in the parent README): percentage points for attacks and capability, proportion units for over-refusal, score units for Petri, so negative is better on `asr` and over-refusal and means less of the behaviour on Petri. `effect_size` is Cohen's h (proportions) or paired d_z (Petri), positive = better. `delta_vs_full_dose`, on half-dose rows only, is the 0.5 cell's paired change in percentage points against the same object at dose 1.0 (on the `none` row, the unsteered run-to-run noise floor). `asr_bon_cap4096` and `mt_truncation_bias_cap4096_vs_512` contrast a cell's 4,096-token reading with its own 512-token one, which for multi-turn also differs in decode (greedy vs the shipped sampler): a cross-instrument comparison, not a cap-only effect.

**Flags.**

| Flag | Meaning |
|---|---|
| `saturated` | at threshold 0.5: the unsteered model is at least 95% jailbroken and no steered object beats the 3.5 pp replicate floor except by defences at least 75% degenerate (the share of the cell's defended goals defended by a looping or capped reply), or one of four pairs ruled in the builder's `SAT_OVERRIDE` |
| `saturated_at_0_75` | the 95% and 3.5 pp criterion at threshold 0.75, with no degeneracy exception and no overrides |
| `retention`, `suggestive`, `thin` | over-refusal rows: the share of the benchmark the terminated-in-both pairs keep; `suggestive` flags retention under 0.5, `thin` fewer than 100 pairs |
| `at_cap_share`, `loop_share`, `degenerate_share` | responses that hit the cap, looped (word 4-gram repetition ratio above 0.5), or either |
| Petri `degenerate_share` | unions loops with the producer's finish-reason cap measure (`at_cap_share_finish_reason`), while `at_cap_share` is by re-tokenized length, so it is not the union of its neighbours |

The notebook's pooling is its own: it pools rows with a non-null `effect_size`, re-derives `saturated` from the 95% and 3.5 pp criterion alone (the builder's flag kept as `saturated_artifact`), keeps `near_floor` cells (the steered cell is indistinguishable from the same object at steering scale 0) and drops suggestive over-refusal contrasts unless `INCLUDE_SUGGESTIVE` is set.
