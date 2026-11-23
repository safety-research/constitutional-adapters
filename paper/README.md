# The paper's figures and tables

This directory holds the code and data that draw the figures and tables of "Constitutional adapters: inference-time
interventions for misalignment and misuse". The registry notebook,
[`notebooks/registry_qwen_figures.ipynb`](notebooks/registry_qwen_figures.ipynb), draws Figures 3 to 19 and Tables 3
and 5 to 13 from the results registry (the SQLite database of runs and metrics behind the paper) or, without
one, from `snapshot/`, the registry's tables cut to the paper's runs; several appendix figures and tables also read
committed files under `notebooks/data/`.

A fresh clone regenerates all of them with no GPU, API key or outside data. Figures 1 and 2 (schematics) and Tables 1,
2 and 4 are not produced here.

## Regenerate the figures

From the repository root, after `uv sync --frozen --extra cpu --group notebook`:

```bash
reproduce/10_figures.sh --source snapshot
```

This renders the notebook against the committed snapshot (`snapshot/CURRENT` names it). Figures 3 to 19 land as
PNG, PDF and SVG under `notebooks/figures/`, Tables 7 to 12 as LaTeX under the Figure 3 to 6 cells and Tables 3, 5, 6 and
13 under their own cells, and the stage copies the figures and rendered HTML to `$SISTEER_FIG_DIR/registry-notebook/`.
About twenty minutes on a CPU; the first run reassembles the snapshot's SQLite under `notebooks/_cache/` (about 4 GB).

The committed notebook carries such a render's outputs. [`notebooks/README.md`](notebooks/README.md) has the nbconvert
route, the environment variables, the figure map and how to read each panel.

## Layout

| Path | What |
|---|---|
| `notebooks/` | The registry notebook, its generator (`make_registry_notebook.py`), its reader library (`registry_nb.py`), the layer-12 cosine data a notebook-only figure draws (`data/relu_arditi_cosines.json`) and the script that recomputes it (`relu_arditi_cosines.py`), the per-run, per-item data Figure 11 and its three notebook-only mirrors draw (`data/qwen_ctlstop_rows.json`) and the script that writes it from the registry (`ctlstop_sidecar.py`), the generator comparison's rows Figure 17 draws (`data/appendix_generator_rows.json`), the per-run, per-item data Figure 18 draws (`data/qwen_si_target_battery_rows.json`) and its script (`si_target_battery_sidecar.py`), the multi-turn rows and per-goal best scores Figure 19 and Table 5 draw (`data/qwen_si_target_multiturn_rows.json`) and the script that writes them from the registry (`si_target_sidecar.py`), the data behind Figures 9, 10 and 16 and Tables 3 and 6 (`data/appendix_*.json`), the scripts that write them (`data/build_appendix_*.py`) and their readers and figures (`appendix_*.py`, sharing `appendix_common.py`), and its README |
| `snapshot/` | The committed paper frame: `CURRENT` names a directory under `snapshots/` with the registry's tables and four views (the notebook reads `v_paper_current` and `v_paper_long`) as parquet (the item-scale tables `item_result` and `item_flags` in the SQLite only), every table as one SQLite in nine gzip parts, and `MANIFEST.json` with a sha256 per file |
| `tests/` | `test_registry_nb_*.py`: the notebook's figure forms and rulings; `test_appendix_*.py`: the appendix analyses' forms, data files and notebook cells |

## Tests

From the repository root:

```bash
.venv/bin/python -m pytest paper/tests tests/test_nb_snapshot_fallback.py tests/test_nb_snapshot_parity.py \
    tests/test_nb_fresh_home.py tests/test_snapshot_paper_frame.py
```

All run on synthetic registries and frames or on the committed data files and snapshot.
