# Regenerating the paper's figures from the committed data

This directory holds the notebook that draws Figures 3 to 14 and Tables 4 to 10 of the paper (Figures 1 and 2 are
schematics; Tables 1 to 3 are not produced here). It reads the results registry (the SQLite database of runs and
metrics behind the paper) or, without one, the paper frame committed under [`../snapshot/`](../snapshot/), the
registry's tables cut to the paper's runs. A fresh clone regenerates everything on a CPU in about ten minutes; no GPU,
API key or outside data.

The main figures are Qwen3.5-4B; the appendix adds Gemma-4-E4B, Nemotron-3-Nano-4B and Llama-3.1-8B. In the legends,
"Difference LoRA" and "Difference ReLU steer" are the constitutional adapters (the constitution-trained object minus the
control-trained one), "Refusal steer" is the Arditi et al. (2024) baseline (`arditi_allpos`), and "Safety SI" and
"Constitution SI" are the two prompted system instructions. The operating dose is a nominal scaling coefficient of 1.0
(Refusal steer: c = 0.3); Figures 3 to 6 are at it and Figure 7 varies it.

## Run it

Once, from the repository root, with [uv](https://docs.astral.sh/uv/) installed (this builds `.venv`, torch included):

```bash
uv sync --frozen --extra cpu --group notebook
```

Then, from the repository root:

```bash
reproduce/10_figures.sh --source snapshot
```

This renders the notebook and copies the HTML and figures to `$SISTEER_FIG_DIR/registry-notebook/`; `--help` lists
the flags. `git` must be on `PATH` for the header cell's code revision.

Without the stage script:

```bash
PY="$PWD/.venv/bin/python"          # uv sync does not activate the venv
cd paper/notebooks
"$PY" make_registry_notebook.py
REGISTRY_NB_SOURCE=snapshot "$PY" -m nbconvert --execute --to notebook --inplace \
    --ExecutePreprocessor.timeout=1800 registry_qwen_figures.ipynb
```

`--inplace` refreshes the committed notebook's outputs; `--to html` writes a page beside it instead. The first run
needs about 2 GB of free disk under `_cache/` for the reassembled snapshot SQLite (1.85 GB) and parquet query caches.
Figures land as PNG, PDF and SVG under `figures/`; the twelve paper figures are also zipped as
`figures/paper_figures.zip`. `figures/`, `_cache/` and `*.html` are git-ignored.

## Choosing the source

`REGISTRY_NB_SOURCE` is `live`, `snapshot` or `auto` (the default). `auto` reads `registry.sqlite` under
`REGISTRY_ROOT` (default `~/registry`) when it exists, else the snapshot under `REGISTRY_NB_SNAPSHOT_ROOT` (default
`paper/snapshot`). Both answer the same SQL, so a snapshot matches the registry at the head (the sequence number of the
last record ingested) it was cut from; the header cell prints the source and head read.
`reproduce/10_figures.sh --export-snapshot` cuts a fresh frame from a live registry and renders from it.

## What the notebook loads

A run's lane (`transfer_kind`) says where its object was fit and served: `instruct_direct` (fit and served on the
instruct checkpoint), `base_to_instruct` (fit on the base checkpoint, served zero-shot on the instruct one) or
`chat_checkpoint` (Nemotron's one checkpoint). A roster lists which runs enter which figure, a ruling includes,
excludes, holds out or supersedes a run, and a run's selection state (`selected`, `not_in_roster`, `excluded`, …) is
its standing under both. [`../../docs/registry/vocabulary.md`](../../docs/registry/vocabulary.md) defines the
vocabulary.

`../snapshot/CURRENT` names the paper frame at registry head 456957: the 10,779 runs of the four paper models and their
base checkpoints, as parquet and as one SQLite in six gzip parts verified against `MANIFEST.json`.

The load cell builds four frames and prints their row counts.

| Frame | Rows |
|---|---|
| `paper_all` | Every measurement of every selected run, four models, from `v_paper_long`, plus provisional `unpooled_member` rows where a cell has no selected reading (named under the load cell). `paper` is its Qwen slice |
| `frontier` | Qwen rows on Figure 7's eight evals whatever their selection state, so every ladder rung is available |
| `pool` | Every unsteered Qwen row, so a panel can borrow an anchor the selection left out (drawn open) |
| `runs_all` | Every run's lifecycle and selection state, from `v_paper_current`, so a missing point can say why |

## Figure map

| Paper figure | What it shows | File |
|---|---|---|
| Fig. 3 | Static jailbreak defense beside the padding ladder | `fig3_single_turn_padding` |
| Fig. 4 | Multi-query (best-of-N, PAIR) and multi-turn (Crescendo, ActorAttack) defense | `fig4_multi_turn` |
| Fig. 5 | Petri behaviours and pooled score, agentic misalignment | `fig5_misalignment` |
| Fig. 6 | Over-refusal and capability | `fig6_capabilities` |
| Fig. 7 | Frontier: change in defense against change in benign compliance across doses | `fig7_frontier` |
| Fig. 8 | Base-to-instruct transfer against the direct instruct fit (Gemma, Qwen and Llama LoRAs) | `figA_transfer_models` |
| Fig. 9 | Layer-12 cosines of the ReLU steers against the refusal direction | `figA_relu_arditi_cosines` |
| Fig. 10 | Qwen control LoRA and control ReLU steer at a coefficient of −1 | `figA_qwen_control_alpha_minus1` |
| Fig. 11 | Jailbreak evaluations, all four models | `figA_jailbreaks_models` |
| Fig. 12 | Best-of-N defense against N, all four models | `figA_bon_curves_models` |
| Fig. 13 | Misalignment evaluations, all four models | `figA_misalignment_models` |
| Fig. 14 | Over-refusal and capability, all four models | `figA_capabilities_models` |
| Tables 4 to 10 | Qwen numbers behind Figures 3 to 6, one per eval family | LaTeX printed under the Figure 3 to 6 cells |

The notebook also writes `figA_*_absolute_models` and `figA_*_delta_models` companions and opens with a coverage
section of what the frame holds and lacks.

## How to read the figures

**Metrics.** One number per eval. Jailbreaks: StrongREJECT attack success at threshold 0.5, drawn as defense success
(100 − ASR); the static attacks are past-tense, CodeAttack, ReNeLLM and Pliny (`grok420`); best-of-N is read at N = 75
(`asr_bon_cap4096`). Capability: mean accuracy or pass rate over MMLU, GSM8K, BigCodeBench (1,140 items) and BFCL
(3,981). Petri: 1 to 10 judge score. Agentic: misaligned-action rate. Over-refusal: label mean over judged items (full
refusal 1, partial 0.5, compliance 0), drawn as benign compliance (100 − label mean). The notebook uses it instead of
the registry's refused-rate headline only when every selected over-refusal run in the paper and frontier frames has a
label-mean row, as the committed snapshot does; the `OVER-REFUSAL METRIC` line under the load cell says which is in
force.

**Judges.** Every StrongREJECT judgment used is by `claude-haiku-4-5`; others are excluded and counted under the load
cell, and a cell with only those draws the "excluded judge" glyph.

**Deltas.** A delta is the change against the unsteered model on the same eval, up being better. By default the anchor
is the mean of every unsteered run of that eval in the paper frame, preferring the same generation cap (rungs of one
declared cap ladder count as one cap) and, where the eval has several, the same item set; its spread is folded into the
interval. On the agentic and benign evals the anchor prefers an unsteered run bound to the row or from the row's own
lane, and on BigCodeBench and BFCL one from the same seed's view. Each delta row names its anchor (`anchor_value`,
`route`). An anchor within 10 percentage points of the bound flags the row `ceiling` in the tables only; the multi-turn
anchors on Gemma, Llama and Nemotron sit near 100 % defense, so a flat multi-turn delta there is bounded by the anchor,
not the object.

**Seeds and draws.** Qwen's fitted objects have three fits at training seeds 1 to 3 (effective batch 64), each a small
mark with its own interval; the category mean weights the seeds equally and adds the between-seed variance. Qwen's
single batch-8 fit is left out; on the other three models that single fit is the reading. A category mean is the
unweighted mean of its evals, except Petri, read from the pooled score (`petri_pooled`). The unsteered model, system
instructions and Refusal steer are not fits: they replicate over decode draws, each a small mark, and every mean
averages an eval's draws first.

**Lanes.** A fitted object is drawn from the `instruct_direct` lane where it has a measurement, else from
`base_to_instruct`, and the line under each figure names the objects drawn that way (in the committed render every
fitted Qwen object in Figures 3 to 6 is base-to-instruct). Anchors follow the steered row's lane and are never pulled.

**Cells not drawn.** Qwen × agentic murder is excluded (the unsteered rate is about 3 %, no room to show a reduction);
Nemotron × agentic leaking and Llama × BFCL have no runs by design. MMLU readings on different item sets are never
compared; such a row is drawn without an anchor and says so.

**Marks.** Colour is the object, marker shape the model; a black ring marks a mean over fewer evals than the category
names, an open mark a delta against an anchor borrowed from outside the paper selection, a grey glyph a missing point;
each legend names its glyphs and their reasons.

**Frontier (Figure 7).** Points are the dose ladders of the difference LoRA and difference ReLU steer on the
base-to-instruct lane, the Refusal steer's ladder, the two system instructions as single points, and the unsteered
model at the origin. A rung is one dose with seeds averaged, drawn only with the panel's whole defense battery (the
four static attacks, or Crescendo and ActorAttack) and both benign evals, XSTest and OR-Bench-Hard, averaged into the
benign axis. The operating-dose rung may also draw on the seed-replicate runs of Figures 3 to 6; every other rung reads
the ladder alone, and the rung table's `lane`, `pulled` and `sources` columns name each point's origin. Held out, and
named under the figure: negative doses; a seed with over 90 % of its defense responses at the token cap, or a rung
whose pooled share exceeds 90 %; and the dose-1.5 rungs of the two difference objects and the Refusal steer's static
c = 1.0 rung. Marker area grows with dose, but the three ladders' doses are different units (a raw coefficient c for
the Refusal steer, a direction-norm multiplier for the ReLU steer, a write-basis multiplier for the LoRA) and are never
pooled; only position in the plane is comparable across objects. The x axis is clipped at 30 percentage points of lost
benign compliance; rungs beyond it are counted and stay in the tables.

## The cosine data (Figure 9)

`data/relu_arditi_cosines.json` holds the layer-12 cosines Figure 9 draws: each ReLU steer's read (gate) and write
directions against the Refusal steer's direction, plus seed pairs within and across the control and constitution
objects. The notebook binds each row to the registry by `object_id` (the file's sha256) and draws a mismatched row open
and named.

Seven of its nine objects are committed under `../../objects/<fit model>/<object_id>/object.pt`, resolved through
`../../objects/MANIFEST.json`. The instruct-direct constitution ReLU steer is not, and neither is the safety-distilled
ReLU steer, which `../../objects/WITHHELD.json` withholds with its reason. To recompute the file, point
`SISTEER_COSINE_INSTRUCT_CONSTITUTION_RELU` and `SISTEER_COSINE_SAFETY_RELU` at those objects' `object.pt` and run
`.venv/bin/python paper/notebooks/relu_arditi_cosines.py` from the repository root; the script checks every file's
sha256 against its `object_id` and refuses by name when a variable is unset.

## Editing the notebook

Edit `make_registry_notebook.py` (the cell sources, with pinned cell ids) or `registry_nb.py` (readers, figure forms,
and the palette, label and roster constants at its top), never the `.ipynb`, which the generator overwrites; then rebuild
and execute as above. A new figure is a new generator cell and, if it belongs in the zip, a name in
`PAPER_FIGURES`.

## Tests

From the repository root:

```bash
.venv/bin/python -m pytest paper/tests/test_registry_nb_appendix_layout.py paper/tests/test_registry_nb_repoint_and_padding.py \
    paper/tests/test_registry_nb_retire_canonical_fits_scoped.py tests/test_nb_snapshot_fallback.py \
    tests/test_nb_snapshot_parity.py tests/test_nb_fresh_home.py tests/test_snapshot_paper_frame.py
```

All run on synthetic registries and frames. The pipeline that produced the data is described in
[`../../reproduce/README.md`](../../reproduce/README.md), the registry in
[`../../docs/registry/README.md`](../../docs/registry/README.md) and the padding ladder of Figure 3 in
[`../../docs/static_padding.md`](../../docs/static_padding.md).
