# The paper's figures and tables

This directory holds the code and data that draw the figures and tables of "Constitutional adapters: inference-time
interventions for misalignment and misuse". The registry notebook,
[`notebooks/registry_qwen_figures.ipynb`](notebooks/registry_qwen_figures.ipynb), draws Figures 3 to 14 and Tables 4
to 10 from the results registry (the SQLite database of runs and metrics behind the paper) or, without one, from
`snapshot/`, the registry's tables cut to the paper's runs.

A fresh clone regenerates all of them with no GPU, API key or outside data. Figures 1 and 2 (schematics) and Tables 1
to 3 are not produced here.

## Regenerate the figures

From the repository root, after `uv sync --frozen --extra cpu --group notebook`:

```bash
reproduce/10_figures.sh --source snapshot
```

This renders the notebook against the committed snapshot (`snapshot/CURRENT` names it). Figures 3 to 14 land as PNG,
PDF and SVG under `notebooks/figures/`, Tables 4 to 10 as LaTeX under the Figure 3 to 6 cells, and the stage copies
the figures and rendered HTML to `$SISTEER_FIG_DIR/registry-notebook/`. About ten minutes on a CPU; the first run
reassembles the snapshot's SQLite under `notebooks/_cache/` (about 2 GB).

The committed notebook carries such a render's outputs. [`notebooks/README.md`](notebooks/README.md) has the nbconvert
route, the environment variables, the figure map and how to read each panel.

## Layout

| Path | What |
|---|---|
| `notebooks/` | The registry notebook, its generator (`make_registry_notebook.py`), its reader library (`registry_nb.py`), the layer-12 cosine data Figure 9 draws (`data/relu_arditi_cosines.json`) and the script that recomputes it (`relu_arditi_cosines.py`), and its README |
| `snapshot/` | The committed paper frame: `CURRENT` names a directory under `snapshots/` with the registry's tables and four views (the notebook reads `v_paper_current` and `v_paper_long`) as parquet, the same data as one SQLite in six gzip parts, and `MANIFEST.json` with a sha256 per file |
| `tests/` | `test_registry_nb_*.py`: the notebook's figure forms and rulings |

## Tests

From the repository root:

```bash
.venv/bin/python -m pytest paper/tests tests/test_nb_snapshot_fallback.py tests/test_nb_snapshot_parity.py \
    tests/test_nb_fresh_home.py tests/test_snapshot_paper_frame.py
```

All run on synthetic registries and frames.
