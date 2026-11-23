# Regenerating the paper's figures from the committed data

This directory holds the notebook that draws Figures 3 to 19 and Tables 3 and 5 to 13 of the paper (Figures 1 and 2 are
schematics; Tables 1, 2 and 4 are not produced here).
It reads the results registry (the SQLite database of runs and metrics behind the paper) or, without one, the paper frame
committed under [`../snapshot/`](../snapshot/), the registry's tables cut to the paper's runs; several appendix figures and
tables also read committed files under `data/`. A fresh clone regenerates everything on a CPU in about twenty minutes; no GPU,
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
needs about 4.5 GB of free disk under `_cache/` for the reassembled snapshot SQLite (3.8 GB) and parquet query caches.
Figures land as PNG, PDF and SVG under `figures/`; the paper's figures (the nineteen files `PAPER_FIGURES` names, Figures 9
and 16 being two each, with the control-pool comparison's and the distillation-target figures' per-seed tables, and Table 5, the
multi-turn distillation-target figure's paper table) are also zipped as `figures/paper_figures.zip`. Before zipping, the render
removes any figure or table file in `figures/` that an earlier render wrote and it did not. `figures/`, `_cache/` and `*.html` are
git-ignored.

## Choosing the source

`REGISTRY_NB_SOURCE` is `live`, `snapshot` or `auto` (the default). `auto` reads `registry.sqlite` under
`REGISTRY_ROOT` (default `~/registry`) when it exists, else the snapshot under `REGISTRY_NB_SNAPSHOT_ROOT` (default
`paper/snapshot`). Both answer the same SQL, so a snapshot matches the registry at the head (the sequence number of the
last record ingested) it was cut from; the header cell prints the source and head read.
`reproduce/10_figures.sh --export-snapshot` cuts a fresh frame from a live registry and renders from it.

## Choosing the Petri judge

By default every Petri reading is `glm-5.3`'s, the paper's Petri judge (the default judging policy, below): the loader
re-derives each Petri value, interval and contrast from that judge's items by the filed rows' rules
(`registry_nb.select_petri_judge`), and the load cell prints how many rows moved. `REGISTRY_NB_PETRI_JUDGE` names another
Petri judge (a judge id or model whose item rows are in the source), or `primary`: the reading the registry files for the
run's primary judge (claude-opus-5, with the claude-opus-4-8 and grok-4.6 refusal fallbacks). `REGISTRY_NB_PETRI_EXCLUDE` names a TSV of transcripts (`run_id`, `item_id`, `trial_index`) to leave out of every
re-derived reading. `REGISTRY_NB_FIGS` writes the figures somewhere other than `figures/`. `petri_judge_compare.py`
renders the notebook under two judges and compares the Petri figures; it needs a live registry.

The Petri judge is one per render. When `REGISTRY_NB_PETRI_JUDGE` is set it wins, and `primary` then pins the filed judge.
Otherwise `REGISTRY_NB_JUDGES` decides, the rubric key `petri_bloom_judge=` over the family key `petri=`. A Petri row with no
item rows under the selected judge follows the same contract as every other family: NaN, `judge_ok` False, out of the panels
and anchors, named in `JUDGE_REPORT["petri"]`.

## Choosing the judges

`REGISTRY_NB_JUDGES` generalises the Petri switch to every LLM-judged reading. `rejudge`, the default and the paper's
(`registry_nb.PAPER_JUDGES`), draws StrongREJECT (static, best-of-N, multi-turn) and OR-Bench and XSTest compliance from the
`minimax-m3` re-judges and Petri from `glm-5.3`; its judge ids are `data/paper_judges.json`, which the paper frame cut also
reads. The code that filed them is `scripts/rejudge/` ([its README](../../scripts/rejudge/README.md)). `primary`, the filed Claude judges, is a robustness view: it draws the filed rows untouched, and
`reproduce/10_figures.sh --judges primary` renders it into `figures-primary/` (a render under it may not write `figures/`).
Overrides follow a policy: `rejudge,petri=primary`, `primary,strongreject=minimax-m3`,
`rejudge,pair=primary` (a family, a rubric name or an eval id, then a judge model, a judge id or `primary`). A judge is resolved from
the registry's `judge` table by its model and the rubric of the filed judgment; StrongREJECT's v1 and v2 rubrics are told apart by
version.

The re-judges file judge, judgment and item rows only, so the loader re-derives each reading from the selected judge's newest
complete judgment of the same run (`registry_nb.select_judges`). It uses the filed row's own rule (`judge_rules.py`): per-goal best
strictly above the threshold, the sampled lanes' per-goal pooling of two trials, any-hit within the first N best-of-N draws and the
C/P/I label census. Intervals follow the filed `ci_method`; contrasts are re-paired against the anchor the filed
contrast names.

A metric the verdicts do not enter, such as a degeneracy share, keeps its filed value. A metric with no rule over item rows reads
NaN. A run with no complete judgment under its judge reads NaN, leaves every panel and anchor through the grader rule (`judge_ok`
False, `judge_model` "<model>: no complete judgment") and is named in `registry_nb.JUDGE_REPORT`, which lists every blanked row
with its reason (`blanked`). A cell whose run has no reading under the judge draws a gap, never another candidate run, and a
decode-only condition's dot of several runs loses only the blanked ones (`JUDGE_REPORT["cell_gaps"]` gives its draws before and
after); a row the filed render does not draw (graded by a judge outside the grader rule) takes no part in that choice. A StrongREJECT row whose
producer dropped errored goals is re-derived over the goals its filed judgment carries, in the frames and in the multi-turn clean-ASR
table alike; a StrongREJECT frame that carries neither error count refuses. Any judged row no selection reached is blanked and named
(`JUDGE_REPORT["unselected"]`). The load cell prints one `JUDGES` line per frame.

The two distillation-target figures (`fig:app_si_distilled_mt` and `fig:app_si_target_battery`) are drawn under the paper's
judges only (`rejudge`, the default), and a render under any other policy prints one `RENDER POLICY` line in each one's
place (see "The distillation-target data").

A render under any judge or threshold other than the paper's prints a `RENDER POLICY` line in its header, naming the judges
it draws, writes the policy into every figure file's metadata and under every LaTeX table, and refuses to write the paper's
`figures/` directory or a link to it (set `REGISTRY_NB_FIGS`). The policy is read as resolved, so an equivalent spelling
(`rejudge,petri=glm-5.3`, `primary,strongreject=minimax-m3,over_refusal=minimax-m3,petri=glm-5.3`,
`REGISTRY_NB_PETRI_JUDGE=glm-5.3`) is the paper's render.

The load cell first checks that the source holds every judge the policy draws from, and stops by name otherwise:
`PolicyJudgesAbsent` when the source's `judge` table holds none of them (a registry graded by the pipeline's own stage 08
holds the Claude judges only; render it with `primary`, or read the committed frame, which carries the paper's judges), and
`SnapshotLacksJudgeItems` when a paper frame was cut without their item rows (`reproduce/tools/snapshot_paper_frame.sh`
cuts with `data/paper_judges.json`'s judges by default). Under `primary` nothing is checked. An item read from a frame asks
only for the item fields its projection keeps (`SnapshotLacksItemKey` otherwise). Each judge's item rows are read at the live head even when the frames come from an older cache, and
`JUDGE_REPORT` records each judge's item head beside the frame's head.

The strength lanes' XSTest readings are the exception to reading items. Their record read each safe item at the first rung whose
reply terminated, while a judgment's item rows hold the top rung only. Under any judge other than the filed one these readings and
their contrasts against the same-standup unsteered arm are re-derived from reads sidecars (per run and item, the read the record
took and each judge's label). The full sidecars are too large to commit and name machine-local paths, so their projection is
committed under `data/strength_reads/` (the record's per-item reads and the columns the loader reads, with a manifest that vouches
for them; `strength_reads_sidecar.py write` makes it from the full sidecars and `check` compares), and a render reads it by
default. `REGISTRY_NB_STRENGTH_READS`, when set, names other sidecars instead (`strength_reads.parquet` paths separated by `:`,
each with its `manifest.json` beside it); set to an empty value it names none. A render that needs them and cannot read them
stops with `StrengthReadsMissing` rather than fall back to the top-rung items. A row is such a reading when its filed source names the battery record (`battery_curves.json`) or its lane
is a strength lane. The loader refuses a sidecar whose manifest does not vouch for it: no or a wrong `table_sha256`, a preview
build, paused or unfinished run directories, or a filed check with disagreements or reads with no outcome yet. A run with a read
not yet judged, or with no complete registry judgment by the judge, reads NaN and is named. A `primary` render never reads
them.

`judge_rules_check.py` applies the rules to the filed judges' own verdicts and compares the result with their filed rows. With
`--disk-units` it also reads the on-disk verdicts the re-judge's unit maps pair with the same keys, where the registry holds no item
rows; `--strength-reads` adds the strength lanes' sidecar reads with the filed judge's label. `judge_compare.py` renders the
notebook under two policies and compares the LLM-judged figures of this README's figure map (`judge_analysis.py`), naming every
other figure file the renders wrote (`figures_not_in_map` in `compare.json`); its `--reuse-cache` reuses the first render's frames
only, never a judge's item rows. Both need a live registry.

## Choosing the StrongREJECT threshold

Every StrongREJECT reduction the notebook draws reads one constant, `registry_nb.SR_THRESHOLD`: a goal counts as jailbroken when its
best judged response scores strictly above it. That covers static ASR, best-of-N any-hit and its curve, PAIR, multi-turn ASR, the
multi-turn degeneracy table. The paper's value is 0.5 (`SR_PAPER_THRESHOLD`), the default.

`REGISTRY_NB_SR_THRESHOLD=0.75` (any value) redraws every StrongREJECT figure at that threshold, anchors and steered rows together.
A reading filed at that threshold is used as filed. A reading filed only at 0.5 is re-derived from its judge's item rows by the
same rule: the policy's judge under `REGISTRY_NB_JUDGES`, else the filed judge, and then only where the rule applied to those items
at 0.5 reproduces the filed 0.5 reading exactly. A row that cannot be re-derived reads NaN, leaves the panels and is named in
`JUDGE_REPORT["frames"]` (`threshold_self_check_failed`, `not_derivable`); the load cell prints a `STRONGREJECT THRESHOLD` line.
Metrics defined at 0.5 with no threshold column (the multi-turn goal-level jailbroken and defended shares) read NaN, and so does
the multi-turn clean-ASR table. A cell draws the run it draws at 0.5, and a gap where that run has no reading at the new threshold.
Its anchor follows the same rule: where an unsteered run that anchors the row at 0.5 has no reading at the new threshold, the row's
delta reads NaN and draws as a cell with no unsteered run, rather than being taken against the other unsteered runs in its place,
and is named in `JUDGE_REPORT["anchor_gaps"]`. Under `primary` this blanks the ReLU steers and the Refusal steer on Crescendo,
ActorAttack and PAIR, whose pools at 0.5 hold unsteered runs that have no reading at another threshold under the filed judge.

Another threshold is complete only under the paper's policy, whose item rows cover every drawn run; the committed frame carries
those rows.
Under `primary` the Claude judgments of several lanes have no item rows in the registry (the pooled relu_paired runs, the strength
lanes, r2_export best-of-N and others) or only per-goal rows, so their readings at another threshold are NaN, among them the Qwen
Difference ReLU steer on every static attack and most of Llama's best-of-N cells. Each is named in `JUDGE_REPORT`.

## The judging interface for appendix lanes

A lane that draws its own rows calls the same judge policy, reduction and items as the main figures, through four names in
`registry_nb` (bundled as `registry_nb.JUDGING` with members `policy`, `reduce` and `items`):

- `judge_for(eval_id, rubric_version=None)` returns `JudgeRef(judge_model, rubric_name, rubric_version, judge_id)`, the judge the
  active policy draws that eval from. StrongREJECT v1 and v2 share a rubric name, so a model holding both needs `rubric_version`
  (best-of-75 control-pool runs: `"v2"`). Under `primary` the model and id are None: each row keeps its filed judge
- `reduce_items(family, entry, t=SR_THRESHOLD, ...)` returns `Reduction(k, n, value, ci_lo, ci_hi)` with value in 0 to 1, by
  `judge_rules.reduce`. For `"strongreject"` it is the ASR at `t` from per-(goal, trial) best scores: strictly above `t`, a cell
  with an unscored unit and no scored unit above `t` leaving n, Wilson or a cluster bootstrap. For `"over_refusal"` it is the C/P/I
  `what="label_mean"` (default), `"graded"` (at-cap replies 0.5, which needs at-cap flags) or `"refused"`, with the exact-variance
  normal interval by default. `entry` is one judgment's item rows, or a mapping of per-goal `best` lists (with `trials`, and
  optionally `n_units` / `n_scored`) or per-item `label` codes (with `subset`, `at_cap`)
- `judge_items(judgment_ids)` returns the current item rows of those judgments from the `item_result` table, the reader the
  re-derivation uses, so it works on the snapshot as on the live registry
- `SR_THRESHOLD`, the threshold every StrongREJECT reduction reads

`judge_rules.sr_asr` and `judge_rules.cpi_reduce` are the two reducers underneath; the main figures' re-derivation calls them too,
so a lane and the main figures share one aggregation per family.

## Figures with a fixed judge

An appendix lane whose figures are drawn from one judge under every policy, the paper's included, declares it with
`registry_nb.register_lane_fixed(figures, judges, judge_ids=None)`, for example `register_lane_fixed("control-pool figures", "minimax-m3
StrongREJECT v1, judge 192df392")`. From then on the RENDER POLICY line, every figure file's metadata and every LaTeX table's comment
say `<figures> lane-fixed: <judges>`, so a `primary` render that mixes the filed judges with the lane's says so on its face. A lane
that names its judges by id (`judge_ids`, keyed as `registry_nb.CTLSTOP_JUDGES` is) is left out of the stamp under a policy that
already draws those rubrics from the same judges: the control-pool figures carry no stamp under the paper's policy.
Registered at import (in `registry_nb.py`, above its last statement `announce_render_policy()`), the stamp reaches the header and
every figure; registered from a cell, it reaches what is written after it. It does not bar the paper's `figures/` directory, which a
render refuses only when its judging policy is not the paper's.

## What the notebook loads

A run's lane (`transfer_kind`) says where its object was fit and served: `instruct_direct` (fit and served on the
instruct checkpoint), `base_to_instruct` (fit on the base checkpoint, served zero-shot on the instruct one) or
`chat_checkpoint` (Nemotron's one checkpoint). A roster lists which runs enter which figure, a ruling includes,
excludes, holds out or supersedes a run, and a run's selection state (`selected`, `not_in_roster`, `excluded`, …) is
its standing under both. [`../../docs/registry/vocabulary.md`](../../docs/registry/vocabulary.md) defines the
vocabulary.

`../snapshot/CURRENT` names the paper frame at registry head 538450: the 11,288 runs of the four paper models, with the
per-item rows of the paper's judges (`data/paper_judges.json`), as parquet and as one SQLite in nine gzip parts verified
against `MANIFEST.json` (the item-scale tables in the SQLite only). Its
state is the drop's (registry head 537565) plus the re-judge of 36 SI-target multi-turn runs, the regeneration of five
unfinished seed-1 Pliny single-turn runs (ten runs, five of them superseded), 70 runs of two lanes a ruling excludes, and
bookkeeping records filed after it. It was cut with `reproduce/tools/snapshot_paper_frame.sh --paper-models-only` (the
paper's judges at the time and the reader projection were the tool's defaults) and then passed through the release pass
(`scripts/release/shipped_text.py apply`), which rewrites its prose and leaves out that judge's item rows: the cut also took the
item rows of the `minimax-m3` agentic-misalignment judge, which the paper's policy named then and no longer does
(`MANIFEST.release_omitted_item_judges`). The agentic runs, their judgments and their filed judge's item rows stay; the
notebook reads none of them.

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
| Fig. 5 | Petri behaviours and pooled score | `fig5_misalignment` |
| Fig. 6 | Over-refusal and capability | `fig6_capabilities` |
| Fig. 7 | Frontier: change in defense against change in benign compliance across doses | `fig7_frontier` |
| Fig. 8 | Base-to-instruct transfer against the direct instruct fit (Gemma, Qwen and Llama LoRAs) | `figA_transfer_models` |
| Fig. 9, top (`fig:app_evenodd`) | Sign decomposition of the control objects: each seed's α = +1 (circle) joined to its α = −1 (diamond), the Refusal steer at c = ±0.5 the positive control. Judge: MiniMax-M3 (StrongREJECT at 0.5 per goal; over-refusal labels per item) | `figA_control_evenodd` |
| Fig. 9, bottom (`fig:app_evenodd`) | Constitution and control objects against 279 refusal directions: the share of each ReLU read and write vector's and each LoRA output space's squared norm in the per-layer refusal span and on the first refusal principal component, over a random direction's. No judge | `figA_refusal_enrichment` |
| Fig. 10 (`fig:app_overlap`) | Overlap of the constitution and control objects layer by layer: (a) the share of one LoRA's update in another's output and input spaces, (b) the cosine of two ReLU steers' read and write vectors, against seed ceilings and a safety-refusal object. No judge | `figA_cK_overlap` |
| Fig. 11 | Published against control-pool stopping, per seed, on every family | `figA_qwen_ctlstop_rule_comparison` |
| Fig. 12 | Jailbreak evaluations, all four models | `figA_jailbreaks_models` |
| Fig. 13 | Best-of-N defense against N, all four models | `figA_bon_curves_models` |
| Fig. 14 | Petri behaviours and pooled score, all four models | `figA_misalignment_models` |
| Fig. 15 | Over-refusal and capability, all four models | `figA_capabilities_models` |
| Fig. 16 (`fig:app_judges`) | Judge robustness: the same Qwen3.5-4B transcripts graded by the Claude judges (circles) and by the paper's judges (diamonds), per seed with 95% intervals and category means; part a static, multi-query and multi-turn, part b over-refusal and Petri. Frozen readings at registry head 537565 | `figA_judge_comparison_a`, `figA_judge_comparison_b` |
| Fig. 17 (`fig:app_generator`) | Generator ablation: per seed, 1M-token constitution LoRAs fit on the Opus-written and on the Qwen3.8-Max-written corpus, on Fig. 11's five panels | `figA_qwen_generator_ablation` |
| Fig. 18 | Every family of the main figures by distillation target: twelve arms per eval (the two prompted SIs, the control and constitution corpus fits, the six SI-distilled objects) in the main figures' dot-plot form, one panel per category (static, multi-query, multi-turn, over-refusal, capability) | `figA_qwen_si_target_battery` |
| Fig. 19 | Multi-turn defense by distillation target: each SI-distilled object beside the constitution corpus fit of the same kind, per fit seed, on all goals and on the goals outside the training pool | `figA_qwen_si_target_multi_turn` |
| Notebook only | Layer-12 cosines of the ReLU steers against the refusal direction (Fig. 10's panel (b) reproduces its pair cosines at layer 12) | `figA_relu_arditi_cosines` |
| Notebook only | Qwen control LoRA and control ReLU steer at a coefficient of −1 (the paper prints its table, Table 13) | `figA_qwen_control_alpha_minus1` |
| Notebook only | Static defense with the control objects and constitutional adapters stopped on the control pool | `figA_qwen_ctlstop_static` |
| Notebook only | Over-refusal and capability (MMLU and GSM8K) with the same objects stopped on the control pool | `figA_qwen_ctlstop_capabilities` |
| Notebook only | Multi-query and multi-turn defense with the same objects stopped on the control pool | `figA_qwen_ctlstop_multi_turn` |
| Notebook only (App. D.3 quotes its numbers) | Forward passes on fixed text: KL from the unsteered model, the change in the refusal-direction projection (block 28, last template token), write cancellation per layer. No judge | `figA_displacement` |
| Notebook only (Table 3 is its table) | Records per claim (C2, the C1 jailbreak half, C3) and corpus part in the training-corpus audit: positive records in the corpus's colour, not-cleared records in grey. Labeller: Claude Opus 5.5, as a corpus labeller, not an evaluation judge; the C1 eval-text counts are a deterministic scan | `figA_corpus_audit` |
| Notebook only (Table 6 is its BFCL side) | Composition of the new BFCL and BigCodeBench failures, Qwen3.5-4B, seeds pooled. No judge: BFCL's syntax-tree checker, BigCodeBench's unit tests, deterministic classes | `figA_qwen_capability_errors` |
| Notebook only (Table 6 is its BFCL side) | The same composition for all four models (Llama has no BFCL run). No judge | `figA_models_capability_errors` |
| Table 3 | The corpus audit's counts per corpus part (`tab:corpus-audit`) | LaTeX printed under the corpus audit cell |
| Table 5 | The distillation-target table (`tab:app_si_distilled_mt`), Figure 19's seed means: per arm the change in multi-turn defense on Crescendo, ActorAttack and their mean, and the mean on the goals outside the training pool, under the unsteered runs' defense success rate | LaTeX printed under the cell after Figure 19's and written beside it as `figA_qwen_si_target_multi_turn_table.tex` |
| Table 6 | BFCL pass rate and new failures by class, Qwen3.5-4B, Gemma-4-E4B and Nemotron-3-Nano-4B (`tab:caperrors`); its BigCodeBench sibling (`tab:caperrors-bcb`) is notebook only | LaTeX printed under the capability error cell |
| Tables 7 to 12 | Qwen numbers behind Figures 3 to 6, one per eval family | LaTeX printed under the Figure 3 to 6 cells |
| Table 13 | The Qwen control objects at α = −1 (`tab:qwen-control-alpha-minus1`) | LaTeX printed under the α = −1 cell |
| Eight notebook tables | The per-seed numbers behind Figure 11 and the three control-pool mirrors (`tab:qwen-ctlstop-static-seeds`, `tab:qwen-ctlstop-capabilities-seeds`, `tab:qwen-ctlstop-multi-turn-seeds`, and `tab:qwen-ctlstop-rule-comparison-seeds` with its static, multi-query, over-refusal and capability siblings) | LaTeX printed under those cells and written beside each figure as `<figure>_seeds*.tex` |
| Seven notebook tables | The per-seed numbers behind Figures 18 and 19 (`tab:qwen-si-target-battery-seeds-<category>` behind each of Figure 18's five panels; `tab:qwen-si-target-multi-turn-seeds` and `tab:qwen-si-target-outside-pool` behind Figure 19's two rows) | LaTeX printed under those cells and written beside each figure as `<figure>_seeds*.tex` |
| One notebook table | The sign rows behind the numbers App. D.3 quotes for Figure 9's top (`tab:app_evenodd_signs`: each object's sign-independent and sign-dependent parts per category) | LaTeX printed under the sign decomposition cell |

The notebook also writes `figA_*_absolute_models` and `figA_*_delta_models` companions and opens with a coverage
section of what the frame holds and lacks.

## How to read the figures

**Metrics.** One number per eval. Jailbreaks: StrongREJECT attack success at threshold 0.5, drawn as defense success
(100 − ASR); the static attacks are past-tense, CodeAttack, ReNeLLM and Pliny (`grok420`); best-of-N is read at N = 75
(`asr_bon_cap4096`). Capability: mean accuracy or pass rate over MMLU, GSM8K, BigCodeBench (1,140 items) and BFCL
(3,981). Petri: 1 to 10 judge score of misalignment, drawn as the alignment score (11 − score, still 1 to 10, 10 the most
aligned). Over-refusal: label mean over judged items
(full refusal 1, partial 0.5, compliance 0), drawn as benign compliance (100 − label mean). The notebook uses it instead of
the registry's refused-rate headline only when every selected over-refusal run in the paper and frontier frames has a
label-mean row, as the committed snapshot does; the `OVER-REFUSAL METRIC` line under the load cell says which is in
force.

**Judges.** Every LLM-judged reading is the paper's judges': each StrongREJECT and over-refusal C/P/I reading is re-derived
from `minimax-m3`'s items and each Petri reading from `glm-5.3`'s, the non-Claude re-judges filed as second judgments of the
same runs, by the filed row's own rule. The grader rule still picks which filed judgment a cell re-derives from: a
StrongREJECT judgment by a judge other than `claude-haiku-4-5` is excluded and counted under the load cell, and a cell with
only those draws the "excluded judge" glyph. The Petri conversations are still driven by the `claude-sonnet-5` auditor.
`REGISTRY_NB_JUDGES=primary` draws the filed Claude readings instead. Every LLM-judged mark in the control-pool stopping
figures (Figure 11 and the three notebook-only mirrors) is the paper policy's judge (`minimax-m3`), whatever `REGISTRY_NB_JUDGES` selects for the other figures: the four refit objects, the five objects that were not refit, the
published fits and the unsteered anchors alike, under the rubric the main figures' rows of the same eval use
(StrongREJECT, its Best-of-N version, the over-refusal C/P/I rubrics). Under the paper's policy the points those figures
share with Figures 3, 4 and 6 are the same values; under another policy they differ by the judge (see the control-pool
stopping data below). Every judged mark of the distillation-target figure (`fig:app_si_distilled_mt`) and its tables is MiniMax-M3's StrongREJECT (v1), the paper policy's
judge, for every run drawn, tabulated or pooled into its anchor; the figure is drawn only under that policy (see the
distillation-target data below). The appendix analyses behind Figures 9 and 10 and Tables 3 and 6 carry their own judge, or
none, named in the figure map and stamped in their data files: the sign decomposition is MiniMax-M3's under every policy.
Figure 16 draws frozen readings under both sets of judges (the judge-comparison data below).

**Deltas.** A delta is the change against the unsteered model on the same eval, up being better. By default the anchor
is the mean of every unsteered run of that eval in the paper frame, preferring the same generation cap (rungs of one
declared cap ladder count as one cap) and, where the eval has several, the same item set. The anchor's runs re-decode the
same items, so its interval is item-clustered: each item's mean over the runs, then the standard error over items (a rate takes
the Wilson interval at the cluster-effective size, so identical runs give one run's interval where each goal is one cell, and one
run's item-clustered width where a goal holds several trials; a score takes the clustered standard error). A delta's interval
combines the point's and the anchor's, unpaired. An anchor run the source holds no item rows for makes the interval the
identical-draws bound (the runs' own intervals averaged, an upper bound), and the route and the table legend say so; so does a
filed judge reading (under `primary`) whose judgment's item rows do not give it. A reading the notebook re-derived from item rows
must equal what its units give, value and n, or the render stops. One fit's runs filed under two item-set ids of the same items
are one cell on the appendix panels, as in the tables and category means. On the benign evals the anchor prefers an unsteered run bound to the row or from the row's own
lane, and on BigCodeBench and BFCL one from the same seed's view. Each delta row names its anchor (`anchor_value`,
`route`). An anchor within 10 percentage points of the bound flags the row `ceiling` in the tables only; the multi-turn
anchors on Gemma, Llama and Nemotron sit near 100 % defense, so a flat multi-turn delta there is bounded by the anchor,
not the object.

**Seeds and draws.** Qwen's fitted objects have three fits at training seeds 1 to 3 (effective batch 64), each a small
mark with its own interval; the category mean weights the seeds equally. Its interval treats the seeds as different fits
read on the same items against the same anchor: the three fits' runs are pooled item by item (each item's outcome under
all three seeds is one cluster), the variance between the fits' readings is added (the seeds are a sample of fits), and
the shared anchor is counted once. Identical seeds therefore give one seed's interval, not a third of its variance. The
same rule makes the appendix's per-eval and absolute marks and their mean columns (one pool over a panel's evals, the
fits being the same on every eval), the padding panel's means (the seeds' rate, its interval the seeds pooled this way)
and each lane of a transfer difference, whose two lanes' anchors cancel where they are the same runs. A fit's run with
no item rows makes the pool's item term the identical-draws bound, said in the table legend. With three seeds the
between-fit term rests on two degrees of freedom: where fits differ far more than items, the 95 % interval (z = 1.96)
covers about 81 % (a t on two degrees of freedom; 81 to 84 % in simulation), and 70 % for a pool of two fits. Qwen's
single batch-8 fit is left out; on the other three models that single fit is the reading. A category mean is the
unweighted mean of its evals, except Petri, read from the pooled score (`petri_pooled`). The unsteered model is not a
fit: its runs are anchor draws. The Refusal steer and the two system instructions are not fits either, and each is one
dot per cell (one per rung on Figure 3's padding panel, one curve in Figure 13, one point in Figure 7): the mean of its
runs matched to the fitted objects in that cell, meaning the same token cap (rungs of a declared ladder count as one cap
only where the fitted runs span that ladder), decode settings, judge, metric and threshold, item set and padding, with
an item-clustered interval, as the anchor's (the runs re-decode the same items). Two such runs on one serving setup are both averaged
when their generations are distinct (each records its own generation store); a base pass beside its escalated rung, or
one generation filed twice, is one reading. On the main figures a system instruction must also share the fitted objects'
serving setup where it has runs there (on Qwen, its three seed-replicate runs); the Refusal steer is served on its own
trial views, and those count. In the appendix a run from any serving setup counts as matched. Where a cell's fitted
objects sit on several instruments, the dot averages runs of one, the one most fitted objects use. Where no run matches,
the dot is the closest readings (the same padding, then the same cap, then the same decode and item set), so it always
plots. Every other reading is set aside and never drawn beside the dot, and the lines under each figure list every
set-aside reading, cross-setup match and fallback dot (the three control-pool mirrors draw the main figures' dots, which
those figures list). The category means, the tables and Figures 7 and 13 read the same runs.

**Tables.** Every table with a first row leads with one row (one per model in the all-model tables): the unsteered model's
level of the quantity its figure's y axis is a change in, defense success rate (100 − attack success rate), benign compliance
rate (100 − over-refusal rate), accuracy or pass rate, or alignment score (11 − Petri score), averaged over seeds. A
first-row cell is the mean, over the fit seeds of the column's arms (s1, s2, s3), of each seed's level: the unsteered reading
the most of that seed's arms were differenced against, a plurality that need not be a majority (a reading being one
unsteered run or the pooled mean of a common anchor's runs; in a category column, its mean over the category's evaluations;
seedless conditions left out). Each arm votes once: a ladder's rungs and a one-dot condition's draws count as one arm. A tie
goes to the tied reading the whole column, seedless conditions included, reads most, and failing that to the mean of the
tied readings; the captions state the plurality and point here for ties. An arm on another lane, cap or serving view than
the plurality of its seed's arms keeps its own run and does not move the first row. Where a model or eval has no seed
replicates, the first row is the unsteered reading the most of its arms, seedless conditions included, were differenced
against. Every other cell is the arm's change in that quantity against its own matched unsteered run, positive better, so
first row + cell approximates the arm's value after the intervention, exactly where that run equals the first row; each
caption says so for its columns. The transfer and cosine tables have no first row: a transfer cell is a difference of two
changes, and a cosine is not a change.

**Lanes.** A fitted object is drawn from the `instruct_direct` lane where it has a measurement, else from
`base_to_instruct`, and the line under each figure names the objects drawn that way (in the committed render every
fitted Qwen object in Figures 3 to 6 is base-to-instruct). Anchors follow the steered row's lane and are never pulled.

**Evaluations and cells not drawn.** The agentic-misalignment evaluations (leaking, murder) are left out at load
(`registry_nb.OMITTED_EVALS`), though the snapshot holds their rows; Llama × BFCL has no runs by design. MMLU readings on different item sets are never
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

**Stopping rules (Figure 11 and its three mirrors).** The published control LoRA and control ReLU steer stopped training on the
constitution corpus's held-out pool; a second fit of each seed stopped on the control corpus's own held-out pool, and the
two difference objects were rebuilt from those fits. The three notebook-only mirrors are Figure 3's static panel, Figure 6's
over-refusal and capability panels (capability narrowed to MMLU and GSM8K, so its mean is a two-eval mean, not Figure 6's)
and Figure 4's multi-query and multi-turn panels with those four objects swapped in; the other five objects are the main
figures' own readings, and every point is differenced against the eval's unsteered anchor by the main figures' rule.
Figure 11 puts the two stopping rules side by side within each seed on all five panels: a circle is the published stopping
rule, a diamond the control-pool stopping rule, the two marks of one seed are joined by a line, and fill means what it
means in every other figure. Its published difference ReLU steer on the multi-turn attacks is a rerun on the same
instrument as the control-pool objects; on every other panel it is the main figures' paired-lane reading.

**Distillation target (Figure 19, `fig:app_si_distilled_mt`).** Objects distilled by KL from a prompted teacher, one per system instruction: the
safety instruction and the constitution on one matched pool, and a short irrelevant instruction, the off-target control,
on a pool from the same attack families; each a LoRA and a ReLU steer at three fit seeds. The figure is the comparison
layout of the stopping-rule figure (Figure 11, `fig:app_ctlstop`): in each attack's column the six distilled objects, each a diamond at each fit seed beside a circle,
the constitution corpus fit of the same kind at the same seed (the distilled LoRAs beside the Constitution LoRA, the
distilled ReLU steers beside the Constitution ReLU steer, fit by cross-entropy on the synthetic constitution documents),
the two joined by a thin line; the bold mean column averages the two attacks within a seed, then across seeds. Colour gives
the distillation target and the kind (a target's LoRA dark, its ReLU steer light), so the corpus fit's circle repeats in
each target's slot of its kind. The upper row is all 259 goals against Figure 4's anchor; the lower row recomputes every
reading on the 92 goals outside the training pool, against the same unsteered runs on the same goals. The two prompted
instructions are not drawn; the two tables list them, per decode trial, beside every distilled object and corpus fit, per
seed. Every point is MiniMax-M3's StrongREJECT: the figure is drawn only under the re-judge judges (below).

**Distillation target on every family (Figure 18, `fig:app_si_target_battery`).** Twelve arms on every evaluation of the main figures, in their unconnected
dot-plot form: the two prompted instructions and the four control and constitution corpus fits, which are the main figures' own
readings, and the six distilled objects (the three targets, each as a ReLU steer and a LoRA). One panel per category, in four rows:
the four static attacks; Best-of-75 and PAIR beside Crescendo and ActorAttack; XSTest and OR-Bench-Hard; and MMLU, GSM8K,
BigCodeBench and BFCL. Each evaluation carries one column per arm, in the legend's order. Small points are a fitted object's three fit
seeds; a prompted instruction is one point per evaluation, the mean of its three decode trials; the large points after the separator
are the category means, each seed's unweighted mean over the panel's evaluations, then averaged across seeds. Every point is
differenced against the unsteered model by the main figures' rule for its evaluation, and on BigCodeBench and BFCL against each
seed's own unsteered run on the same harness. A GSM8K point is accuracy over the items its ladder scored, as in the main figures.
Five per-category tables list the prompted instructions per decode trial beside every distilled object and the constitution corpus
fits, per seed.

## The cosine data (notebook only)

`data/relu_arditi_cosines.json` holds the layer-12 cosines the notebook-only cosine figure draws: each ReLU steer's read (gate) and write
directions against the Refusal steer's direction, plus seed pairs within and across the control and constitution
objects. The notebook binds each row to the registry by `object_id` (the file's sha256) and draws a mismatched row open
and named.

Seven of its nine objects are committed under `../../objects/<fit model>/<object_id>/object.pt`, resolved through
`../../objects/MANIFEST.json`. The instruct-direct constitution ReLU steer is not, and neither is the safety-distilled
ReLU steer, which `../../objects/WITHHELD.json` withholds with its reason. To recompute the file, point
`SISTEER_COSINE_INSTRUCT_CONSTITUTION_RELU` and `SISTEER_COSINE_SAFETY_RELU` at those objects' `object.pt` and run
`.venv/bin/python paper/notebooks/relu_arditi_cosines.py` from the repository root; the script checks every file's
sha256 against its `object_id` and refuses by name when a variable is unset.

## The control-pool stopping data (Figure 11 and its mirrors)

The runs behind Figure 11 and its three mirrors are outside the paper's selection and postdate the committed snapshot, so the notebook
reads them from `data/qwen_ctlstop_rows.json`: one row per run of a pinned object (its identity, its MiniMax-M3 judgment
and judge, and the columns the deltas are computed from) and, beside it, the run's per-item numbers: for a StrongREJECT
run the best score of every goal with its unit counts, for an over-refusal run every prompt's label, for MMLU and GSM8K
every item's correctness (GSM8K's redrawn items from its next rung). The control-pool XSTest readings are read on the
2,048 rung; one whose ladder exited on a lower rung with no reply at cap is read on that rung, and its numbers then
carry the registry's record of the exit (`ladder_exit`: the rung, the replies at cap there, the replies counted), which
the cells print under the figure. No reading is stored: every one is computed at load
from those numbers through the notebook's judging interface (`registry_nb.JUDGING`: `judge_for`, `reduce_items`,
`judge_items`), at the StrongREJECT threshold `registry_nb.SR_THRESHOLD`, so `REGISTRY_NB_SR_THRESHOLD` redraws the four
figures at another threshold with no judge call, in a live or a snapshot render. The objects are
pinned in `registry_nb.py`, the runs and judgments in the file. A snapshot render reads the file alone. A live render
derives the whole file again from the registry and stops with an error on any difference, rather than draw either side;
the cells come before the zip and record cells, so neither of those runs either, and regenerating the file (below) is the
remedy. The file carries no paths, notes or selection bookkeeping.

Figure 11 and its mirrors have one judge in every render that reaches their cells. Those cells come after the main figures' cells,
so a render the notebook itself refuses or cannot finish never draws them. The committed snapshot reaches them under the
paper's policy (it carries the paper's judges' item rows) and under `primary` at any threshold, and a live registry under any
policy and threshold. The file also holds, under the same judge, the main
figures' own runs those figures draw: the five objects that were not refit, the published fits of the four slots, and
every unsteered run a point is differenced against (`main`: the run's eval, judgment and rubric version; `main_items`:
for a StrongREJECT run the best score of every goal in every trial and the goals holding an unscored reply, for an
over-refusal run every prompt's label in every trial; each item set's ids, and XSTest's safe and unsafe halves, are
stored once). Which runs is read off the notebook's own frames: the rows Figure 3's anchoring selects on these evals
and the unsteered runs it pools, on the committed snapshot and on the live registry, so either render finds its runs.
The reader admits runs judged only by the policy's judge (`minimax-m3`) into the paper lane (`registry_nb.synthesized_rows`), which is how the regenerated seed-1 Pliny runs reach the main figures.
At load, every such row takes its reading from the file by the rule the notebook's re-derivation applies to that row
(its producer's pooling of trials and its interval), at `SR_THRESHOLD`; the anchoring then runs on those rows, so the
selection and the anchor rule are the main figures' own. Under `REGISTRY_NB_JUDGES=rejudge` the result is checked
against the render's own frames, value for value, and at the paper's threshold the set of runs drawn is checked under
every policy; a difference stops the cells with an error. A run the judge has no verdict of is named in
`main_pending` and reads nothing: its point is marked as a gap of its own kind (`no verdict of this judge`), and a pool
that held it is smaller, which the cell prints. A strength-ladder XSTest run is named there too: its reading under
another judge comes from its battery record's per-item reads, which the file does not hold.

A file with a pending reading is provisional. A file with a `main_pending` entry is provisional too, unless every entry
is declared: its reason is one of the two above, and the list of the live registry's runs names none of them. A live
render then reads every run it draws. Only a snapshot render leaves those points out, and the cell prints which. The
file records both lists under `main_needs`: the registry head each was read at, the sha256 of the list as
`ctlstop_sidecar.py needs` writes it, and its run ids. Each list is held to its sha256 and to the runs the file names
whenever the file is loaded, so which list names which run is checked from the file alone. An entry with any other
reason, an entry the live list names, and a file with no live list each leave the file provisional.

An over-refusal reading (XSTest, OR-Bench-Hard) is differenced against the unsteered runs that the main figures'
seed-view fitted rows of the same evaluation and seed are differenced against in the same render
(`registry_nb.ctlstop_bind_anchors`): the rows of every fitted object but the difference ReLU steer
(`registry_nb.CTLSTOP_BOUND_SLOTS`), whose main-figure rows are paired-write readings differenced against that lane's
own unsteered runs. Where those rows carry the registry's binding to their seed view's own unsteered run, as on the
committed snapshot, that is the one run; where the registry holds no current binding for them and the main figures
pool unsteered runs instead, it is the same pool. Each row says which (`anchor_binding`: `bound` or `pooled`), the
cell prints it under the over-refusal and capability mirror, and the per-seed tables carry it in their note. Where those rows are differenced
against different runs, or the main figures draw none for the seed, the cells stop with an error.

A static reading is drawn on the rung its own ladder stopped on. One that climbed above every rung the static panel
holds an unsteered run on has no unsteered run of the static panel at its own cap. The unsteered Pliny runs at a 16,384
cap are the padding instrument's, the runs Figure 3's padding panel differences against (a 98,304-token serving window
and a single pass at that cap in place of a ladder), and the static panel leaves them out (`registry_nb.drop_padded`).

Four control-pool Pliny readings climbed that way, and they are read at their 16,384 rung and differenced against the
static panel's unsteered runs at 8,192, the final rung of the same ladder, where those runs had nothing left to climb.
The file declares each of them (`anchor_declared`: per run id, its reading, its rung, the anchor cap and the reason);
`ctlstop_sidecar.py` writes the declarations, and the live check compares them. A declared row's delta and interval are
`anchored`'s own on the unsteered runs at the declared cap (the row's cap is read as that cap for the pool's selection
alone), so it takes the anchor the main figures' fitted rows at 8,192 carry; its route ends in `declared anchor`, and
the cells under the static mirror and Figure 11 and the per-seed tables' notes state the anchoring. A declaration that names no drawn
row of the file at its rung leaves the file unloadable, and the cells stop with an error where the frame contradicts
one: a rung that is not above every comparator's cap, an anchor cap that is not the highest comparator cap present or
that no unsteered run sits on, another rung than the declared one, or an anchor other than the one the fitted rows at
that cap carry (`registry_nb.ctlstop_declared_anchor_problems`, `registry_nb.ctlstop_anchor_problems`).

A reading above every comparator's cap that the file does not declare stays in the file and in the frames with its
reading and no delta (`registry_nb.ctlstop_rung_gaps`) and is not drawn. It is not differenced against a lower rung's
runs while each rung is its own cap class, as in every
render today; were the static rungs one cap class (`registry_nb.cap_class`), it would take that class's unsteered runs,
like every fitted row of the class, and nothing would be left out. The static mirror leaves its mark out and rings the
object's mean, which then averages that seed's other evaluations; Figure 11 draws the gap glyph in its place and in place of the
control-pool mean it would have entered; the cells print its name, its rung and the rungs of the static panel's
unsteered runs under both figures, and the per-seed tables print `--` and name it in their note. The lower rungs of its
ladder ride in the file as `rung` rows and are not drawn in its place. A cell whose every seed is such a reading draws
no seed: the static mirror draws a gap mark of its own there, for the status `no unsteered run of the static panel at its cap`,
never the shared `no unsteered run`. Like the main figures, the static mirror carries no key for its gap marks, so the mark is
named under the figure: the cell prints one line for each reading left out. Any other row the main figures' anchor
rule cannot be held to stops the cells with an error.

A control-pool reading that is not in the file yet is marked `control-pool pending`, never with the registry status of
the published object's runs. A figure with no judged control-pool reading at all is not drawn: its images and tables
are not written, an earlier render's are removed, and the zip leaves it out. The load cell prints a `CONTROL-POOL ROWS`
line and `JUDGE_REPORT["control_pool"]` records the fixed judge (`minimax-m3 (lane-fixed)`) beside the render's policy.
The frames' own rows of every pinned run and every run of the appendix's campaign (`registry_nb.CTLSTOP_CAMPAIGNS`) are
held from the re-derivation, so no other figure draws them from the registry's judgments. The lane's loader,
`registry_nb.ctlstop_load`, answers at any threshold in either render mode whatever the policy, and the module
registers the fixed judge at import (`register_lane_fixed`, see "Figures with a fixed judge"), so the RENDER POLICY line,
every figure file's metadata and every table's comment name it. `ctlstop_load(..., hold=True)` instead leaves out the rows
of an eval the render's policy draws from another judge.

`paper/notebooks/ctlstop_sidecar.py` writes it from the live registry
(`.venv/bin/python paper/notebooks/ctlstop_sidecar.py write`, from the repository root; `--allow-pending` writes a
provisional file listing the readings not yet judged and the main-figure runs without a verdict). `write` lists the
main figures' runs by running the notebook's load cells on the snapshot and on the live registry (the live load takes
about ten minutes); `ctlstop_sidecar.py needs --out FILE` writes one such list under `REGISTRY_NB_SOURCE`, and
`write --needs FILE FILE` reads lists already written. Either way both lists are recorded in the file. Without
`--allow-pending` it refuses when a reading is not yet judged or a main-figure run without a verdict is not a declared
one (above). It also refuses when a run breaks the figures' contract (its pins,
instrument, item set, judge or caps), when a pinned judge is not the one `registry_nb.judge_for` names in the registry
under the `rejudge` policy, when a reading has two runs, when a declared anchor's run is not the drawn reading it names
at its rung, when an XSTest judgment judged a rung above the one the notebook
reads, or one below it without the registry's record that the run's ladder exited there with no reply at cap, when a
payload holds a label outside the rubric's or yields no reading, or when the publication scan finds anything in the
file; `ctlstop_sidecar.py check` compares the committed file with a fresh derivation. A live render does the same and
also stops when the derivation itself reports a problem.

## The distillation-target data (Figures 18 and 19)

The distilled objects' runs are outside the paper's selection, and the lower row needs per-goal scores the paper frame does
not keep, so the notebook reads every run the figure and its tables use from
`data/qwen_si_target_multiturn_rows.json`: one row per run (its object, attack, its MiniMax-M3 StrongREJECT judgment and
judge, and the columns the deltas are computed from) and, beside it, the run's per-goal payload: the best score of each of
the 259 goals over its scored replies, with its reply and scored counts. No reading is stored: every one is computed at load
from those numbers through the notebook's judging interface (`registry_nb.JUDGING.reduce`), at the StrongREJECT threshold
`registry_nb.SR_THRESHOLD`, so `REGISTRY_NB_SR_THRESHOLD` redraws the figure at another threshold with no judge call. A goal
with an unscored reply and no scored reply above the threshold leaves that run's count, and a run with more than 1% of its
259 goals so is refused. The bound is the run's: a reading on the goals outside the training pool counts the run's
determined goals among those 92 and has no bound of its own, so at 0.5 one corpus fit's ActorAttack reading there leaves out 1
of 92 goals (1.1%). The tables' notes name every such goal (at 0.5, one goal in one distilled object's run and one in a
corpus fit's, the latter outside the pool), and the file lists every unscored reply. Every reading the file gives names its
payload as the source of its units, keyed by the goal ids: on all 259 goals `registry_nb.SITARGET_FILE_SOURCE`, on the 92 goals
outside the pool `registry_nb.SITARGET_OUTSIDE_SOURCE` (`judge_rules.payload_reading_units`, the reading's own rule). So a pooled
interval (the anchor's on the 92 goals, a prompted instruction's over its three trials) and every seed pool of the distilled
objects' fits (the mean marks, the table's seed means) are item-clustered over the goals from the same payloads, the seed pools
across their seeds, and no reading of the file takes, or replaces, the units of the registry's reading of a run that Figure 4's
anchor pools. Every object and run is pinned by id in `registry_nb.py`.

The figure has one judge, MiniMax-M3, and one rubric per attack (StrongREJECT v1) for every run it draws, tabulates or
differences against. Its cells draw only in a render whose `REGISTRY_NB_JUDGES` reads both attacks from that judge (the
`rejudge` policy); under any other policy they print one `RENDER POLICY` line and draw nothing, never a mixed figure. Before
drawing, every drawn, tabulated and pooled run is checked to name a judgment of that judge on the 259-goal set
(`registry_nb.sitarget_rubric_problems`; a run the judge switch blanked counts as a breach), the cell prints one line per attack
with the counts and the instruments and caps of every run (`registry_nb.sitarget_disclosure`). A live render re-derives every
stored field from the registry's rows, every payload from the judgment's item rows and every reading from the frame's own row,
and stops with an error on any difference; a snapshot render does the same against the committed snapshot, which carries every
pinned run, so a pinned run it does not hold stops the cell too.

The cell after the figure's prints the paper's table of it (`tab:app_si_distilled_mt`, `registry_nb.sitarget_paper_table`) and
writes it to `figures/figA_qwen_si_target_multi_turn_table.tex`: per arm the seed mean (a prompted instruction's mean over its three
decode trials) of each attack's change, their mean, and the mean on the 92 goals outside the pool, each with its 95% interval, from
the figure's own frames through `registry_nb.category_means` and `registry_nb.pool_seeds`, under the defense success rate of the
unsteered runs every row is differenced against. Its caption names the goals a reading leaves out. It carries no judge stamp; a
render under a policy other than the paper's adds that render's policy line as a comment.

`paper/notebooks/si_target_sidecar.py write` regenerates the file from the live registry under the `rejudge` policy, whatever the
shell holds (`check` compares), and `si_target_sidecar.py pool-goals` re-derives which goals the training pools contain
(`configs/heldout/sitarget_training_pool_goals.json`).

The battery figure (Figure 18, `fig:app_si_target_battery`) draws twelve arms on every eval of the main figures, as the main figures draw theirs (`registry_nb.delta_figure`): the two
prompted SIs and the control and constitution corpus fits, which are the main figures' own rows copied unchanged
(`registry_nb.sitarget_battery_comparators`), and the six distilled objects. It reads their multi-turn panel from that file and every
other distilled-object reading from a sibling,
`data/qwen_si_target_battery_rows.json`: one row per run of a distilled object on the battery's instruments (its object, eval, its
judgment and judge, its scoring rule, the cap of the reading drawn and the columns the deltas are computed from; a static cell's lower
rungs and a GSM8K cell's merged rung ride as `rung` rows) and, beside each drawn run, its per-item payload: the best StrongREJECT
score of each goal with its unit counts, each over-refusal item's label, or each capability item's correctness (a run that files no
item rows, its counts). These runs carry no metric row of the registry's, so the file is their only path into a figure. Every
reading is computed at load through the judging interface at `registry_nb.SR_THRESHOLD`, and names its payload as the source of its
units (`registry_nb.SITARGET_BATTERY_SOURCE`; `judge_rules.payload_reading_units`, the reading's own rule), keyed by the ids of the
file's item list for its eval, which are the registry's item ids. Each list is sorted by id, which is not the order the registry pins
for every set (BigCodeBench's), so no unit is keyed by position. Every seed pool of the distilled objects' fits (the figure's mean
marks) is thus item-clustered across its seeds from the same payloads, and each row carries its eval's item set for the alignment
guard. The 72 static readings are pinned by run and
judgment id (`registry_nb.SITARGET_STATIC_RUNS`), and a reading not yet in the data is listed as `pending`, named under the figure
and drawn as a gap glyph where its arm holds no reading of the eval at all (`registry_nb.sitarget_battery_gaps`). The file is
in one of three states (`registry_nb.sitarget_battery_state`): final, with no `pending` list; pending registration, whose `pending`
list is exactly every BigCodeBench and BFCL reading of the distilled objects (generated on another harness, not yet ingested by the
registry) and nothing else, a committed state the final render draws as is, with each distilled object's capability mean ringed as
partial over the evaluations it holds; or provisional, any other `pending` list, which the render refuses unless it is flagged
`REGISTRY_NB_SITARGET_PROVISIONAL=1` (`reproduce/10_figures.sh` clears the flag).
A GSM8K reading is drawn at 8,192 tokens only within its ladder's trigger: with more than 1 % of the answers at that cap and no 16,384
rung run, it stays pending (`registry_nb.sitarget_gsm8k_ladder_problem`). Before drawing, every drawn, tabulated and pooled run of
every eval is held to its eval's one judge or scoring rule and item set (`registry_nb.sitarget_battery_rubric_problems`; on
Best-of-75 the arms, whose pool is the one Figure 4 differences every arm against), every distilled object's anchor to the main
figures' rule for its eval (`registry_nb.sitarget_battery_anchored`), and a live render
re-derives the whole file from the live registry and stops with an error on any difference. A render from the committed snapshot,
which predates these runs, reads the distilled objects' rows from the file alone, held to its contract, and the corpus fits' and
prompted instructions' rows from the snapshot's own frame. A reading of theirs the main figures' rows do not hold is named under the
figure too.
`paper/notebooks/si_target_battery_sidecar.py write` regenerates it (`--allow-pending` lists the readings not yet in the data as
`pending`, the pending-registration state included; `check` compares). Once the registry has ingested the distilled objects'
BigCodeBench and BFCL runs, the one command

    python paper/notebooks/si_target_battery_sidecar.py write

run from a committed tree with the live registry writes the final file: without `--allow-pending` it refuses, writing nothing, while
any reading is still missing, so a file it writes has no `pending` list and the figure draws all twelve arms on every capability
eval. The notebook is then rebuilt and executed: `tests/test_nb_battery_render_sidecar_state.py` fails while the committed render's
battery cell prints its pending-state, `!! NOT DRAWN` and ringed-means lines and the committed file is final, or the reverse, and
while the cell's count and names of the pending readings, or its count of drawn runs, differ from the file's.

## The generator-comparison data (Figure 17)

The generator comparison (`figA_qwen_generator_ablation`, `fig:app_generator`) draws 1M-token constitution LoRAs fit on a pool of
the Opus-written constitution corpus and on a pool of the corpus Qwen3.8-Max wrote through the same pipeline, three seeds each, on
Fig. 11's five panels: the four static attacks, Best-of-75 and PAIR, Crescendo and ActorAttack, XSTest and OR-Bench-Hard, MMLU and
GSM8K. The runs sit outside the paper's selection by three registry rulings, so the notebook reads them from
`data/appendix_generator_rows.json`: one row per run (its public identity, judgment and judge) and, beside each drawn run, its
per-item payload, built by the control-pool appendix's own builders and held to that appendix's contract (`CTLSTOP_EVALS`: the
instrument, item set, metric, drawn cap and judge of every evaluation). No reading is stored: `appendix_generator.py` computes each
one at load by that appendix's frame recipe (`registry_nb.ctlstop_frame`), so `REGISTRY_NB_SR_THRESHOLD` redraws the figure at
another threshold with no judge call.

Every point is differenced against the pooled unsteered anchor Fig. 11's points of the same evaluation and cap are differenced
against in the same render (`registry_nb.ctlstop_anchored` on the control-pool cell's `ctl_op`, `ctl_pool` and `ctl_md`), once per
corpus, because the anchoring keeps one reading per cell and its cell key has no corpus. The lane's own unsteered static runs are
never an anchor: the cell prints them as a drift line against the pooled anchor. The cell also prints the static panel's pre-fixed
verdict (Qwen − Opus over the three matched seed pairs, pooled over the four attacks, with its 95% t interval on two degrees of
freedom, read from the readings so the anchor cancels), one headline per panel (each corpus's mean mark and the Qwen − Opus
difference, descriptive on every panel but the static one), and GSM8K's coverage (scored items of 200) with the all-200 reading
beside it. The figure's judges are registered as lane-fixed (see "Figures with a fixed judge") before the figure is written.

A file written before every reading is filed is provisional: it lists each missing reading under `pending`, with the reason, and
the figure draws each as a gap glyph that the line under it names. `write` writes one only with `--allow-pending`, and
`tests/test_appendix_generator_gate.py` fails while the committed file lists any pending reading or the comparison of the whole grid
is not pinned in the script, so a provisional file does not land. A snapshot render reads the
file alone. A live render
derives the file again from the registry (read-only, one transaction) and stops with an error on any difference, rather than draw
either side. `paper/notebooks/appendix_generator_sidecar.py` writes it from the live registry
(`.venv/bin/python paper/notebooks/appendix_generator_sidecar.py write`, from the repository root, on a committed tree; `write
--allow-pending` for the provisional file) and refuses, writing nothing, when a run breaks the contract: an object outside the
pins, a run on another instrument than the contract's, a reading filed under another rubric than its evaluation's judge, two runs
or two judgments for one reading, an XSTest rung below 2,048 without the registry's record that the ladder exited there with no reply
at cap, a payload that breaks the contract's shape, or a static run that is not one of the 28 the static comparison filed or whose
per-run count at > 0.5 differs from that filing's. `appendix_generator_sidecar.py check` compares the committed file with a fresh
derivation. `appendix_generator_sidecar.py inputs` prints, from the committed final file, the run and object inputs the comparison of
the whole grid is filed with: every run whose items enter a reading, the GSM8K rung runs merged into their readings included, and
the six objects; once that comparison is pinned, `write` refuses a filing with other inputs. The file carries no paths, notes or
selection bookkeeping.

## The appendix analyses' data (Figures 9 and 10, Tables 3 and 6)

Each of these analyses reads one committed file under `data/`, `appendix_<name>.json`, through its reader `appendix_<name>.py`,
which refuses the file by name when it breaks its contract; the figures share `appendix_common.py` for style and saving, except the two
weight-geometry figures the paper prints (Figure 9's lower half and Figure 10), which `appendix_geometry_style.py` draws on the
manuscript's canvas and saves as its copies: their PNGs decode to the manuscript's pixels. A file holds
counts, aggregates or per-item verdicts only, never transcript or record text, and its provenance block names its source files by sha256
and its judge.
`data/build_appendix_<name>.py` writes it from the analysis's own outputs, which are not in the repository, and `--check` compares
the committed file with a fresh build instead of writing; the sign-decomposition, geometry and overlap builders also read the live
registry, read-only.

| File | What it holds | What the cell checks against the notebook's source |
|---|---|---|
| `appendix_corpus_audit.json` (Table 3) | The training-corpus audit's aggregate counts, each under the audit summary's own key, and the overlap scan's thresholds and per-set maxima, less the sets of an evaluation family the paper omits, of which it keeps only the count and maxima; no record id | Nothing: it pins no registry id |
| `appendix_geometry_evenodd.json` (Figure 9, top) | One row per drawn reading of the sign decomposition (object, seed, evaluation, sign; the change from the unsteered model, with the lane's standard error split into the steered run's part and the anchor's), the unsteered runs of each anchor, and every used run's per-unit MiniMax-M3 verdicts by position in its evaluation's item set (named by its size and the sha256 of its ids) with its units that have a degenerate reply, from which the figure's and the table's intervals are built (the notebook's rules: each run's own interval, the anchor's runs and the seeds clustered by goal or item) | Every object and run, except that a source older than the two regenerated seed-1 Pliny runs reads those from the file and names them; every MiniMax-M3 judgment wherever the source holds them or is newer than the last of them (a live registry; a frame cut with the paper's judges, as the committed one is); every payload against its run's reading, as the pools read it |
| `appendix_geometry_forward.json` (Figure 9, bottom, and the forward passes) | Per-seed, per-layer aggregates of the forward passes and the weight analyses; no activations | Its 28 object pins: label, fit checkpoint, seed and, for derived objects, parents |
| `appendix_geometry_overlap.json` (Figure 10) | Per pair and layer, the LoRA overlaps in the output and input spaces with each LoRA's random-subspace null, and the ReLU read and write cosines; no weights | Its 14 object pins (label, fit checkpoint, seed), and its layer-12 cosines against `relu_arditi_cosines.json` |
| `appendix_caperrors.json` (Table 6) | Per run, the new BFCL and BigCodeBench failures by class, the gains and the BFCL calling counts; per pool, the analysis's 95% item-bootstrap intervals | Every run's correct count, new failures, gains and object id, recomputed from the per-item results |

Three numbers App. D.3 quotes are not recomputed by the notebook, because none of these files holds their inputs: the share of a
1M-token constitution LoRA's update in the control object's output space by corpus writer (20% for Opus 5 text, 7.5% for Qwen3.8-Max
text; a weight comparison of the generator LoRAs), the cosine of the two objects' residual-stream displacements (0.55 to 0.72; from
the forward-pass job behind `appendix_geometry_forward.json`, whose displacement cosines that file does not copy), and the factor by
which adding the control object multiplies the constitution object's KL from the unsteered model (2.3 to 3.3; a forward pass with the
summed object).

## The judge-comparison data (Figure 16)

The judge-robustness figure (`figA_judge_comparison_a` and `_b`, `fig:app_judges`) shows the same Qwen3.5-4B transcripts graded by the
Claude judges the registry filed first (claude-haiku-4-5 for StrongREJECT, claude-sonnet-5 for over-refusal, claude-opus-5 for Petri)
and by the paper's judges (minimax-m3; glm-5.3 for Petri). Its readings are frozen at registry head 537565, the head the paper's final
figures were drawn at, in `data/appendix_judge_comparison.json`: every mark of the two parts (per judge, category, evaluation, object
and seed, the change from the unsteered model with its 95% interval, whether its anchor came from outside the paper frame, and for a
mean whether it averages fewer evaluations than the category names), the undefended model's static attack success under each judge, and
the all-model figures' mean marks under each judge. Every render reads the file alone, whatever its source, judging policy or
StrongREJECT threshold. `appendix_judge_comparison.py` draws the two parts with `registry_nb`'s style and helpers and prints the
numbers the appendix quotes: the rank correlation of the interventions between the two sets of judges in each of the 20 model ×
category columns of the all-model figures (`figA_jailbreaks_models`, `figA_misalignment_models`, `figA_capabilities_models`), the
undefended model's static attack success under each, and the Difference LoRA's multi-turn mean under each.

`data/build_appendix_judge_comparison.py` writes the file from a frozen copy of the registry (`--registry DIR`, the directory holding
it as `registry.sqlite`, opened read-only). It exports the notebook code at a pinned commit of the development history (`CODE_COMMIT`,
by `git archive`), runs the notebook's header and load cells and the statements that build Figure 3's deltas and the category means
once per judging policy (`primary`, `rejudge`, in two subprocesses at once), and reduces the frames as the figure does (`pool_draws`
per cell, `pool_seeds` per mean). The `rejudge` policy reads the committed strength-read projection, whose manifests must vouch for the
sidecars the delivered figure's render read. The copy may hold records ingested after the pinned head: the build refuses unless each is
bookkeeping or a row of a run (a judgment, its items, the run row it updated) that the paper's selection excludes and no reading draws
on. It records the copy's sha256 and head, the rows after the pinned head, each policy's judge ids per category, the selection (run and
anchor-run counts with the sha256 of their ids), the code commit, the sha256 of the `registry_nb.py` it ran and its own sha256;
`--compare-values` checks every mark against the value files written beside the delivered figure and records the result, and `--check`
rebuilds and compares with the committed file. The file carries no paths, host or session names.

A rebuild needs two things the published repository does not carry: the development history that holds `CODE_COMMIT`, and the frozen
registry copy. Without them the file is held to what it records: the reader refuses readings that do not hash to the recorded
`payload_sha256`, and the tests pin that hash, the registry copy's sha256 and the sha256s of the delivered figure's value files, redraw
both parts and check each part's decoded pixels against the delivered PNG's, and recompute the numbers the appendix quotes.

## Editing the notebook

Edit `make_registry_notebook.py` (the cell sources, with pinned cell ids) or `registry_nb.py` (readers, figure forms,
and the palette, label and roster constants at its top), never the `.ipynb`, which the generator overwrites; then rebuild
and execute as above, and commit the executed notebook with the change
(`paper/tests/test_registry_nb_committed_sync.py` fails while its cells are not the generator's, and
`tests/test_nb_committed_sync_gate.py` runs the same checks where the landing gate collects them). A new figure is a new
generator cell and, if it belongs in the zip, a name in `PAPER_FIGURES`.

## Tests

From the repository root:

```bash
.venv/bin/python -m pytest paper/tests/test_registry_nb_appendix_layout.py paper/tests/test_registry_nb_repoint_and_padding.py \
    paper/tests/test_registry_nb_retire_canonical_fits_scoped.py paper/tests/test_registry_nb_decode_only_dots.py \
    paper/tests/test_registry_nb_ctlstop_appendix.py paper/tests/test_registry_nb_table_levels.py \
    paper/tests/test_registry_nb_petri_judge.py paper/tests/test_registry_nb_judges.py paper/tests/test_registry_nb_threshold.py \
    paper/tests/test_registry_nb_si_target_appendix.py paper/tests/test_registry_nb_si_target_battery.py \
    paper/tests/test_registry_nb_committed_sync.py paper/tests/test_registry_nb_pooled_ci.py \
    paper/tests/test_appendix_generator.py paper/tests/test_appendix_generator_sidecar.py tests/test_appendix_generator_gate.py \
    paper/tests/test_appendix_common.py paper/tests/test_appendix_notebook.py paper/tests/test_appendix_corpus_audit.py \
    paper/tests/test_appendix_geometry_evenodd.py paper/tests/test_appendix_geometry_forward.py \
    paper/tests/test_appendix_geometry_overlap.py paper/tests/test_appendix_geometry_style.py paper/tests/test_appendix_caperrors.py \
    paper/tests/test_appendix_judge_comparison.py tests/test_appendix_judge_comparison_gate.py \
    tests/test_nb_committed_sync_gate.py tests/test_nb_battery_render_sidecar_state.py tests/test_nb_snapshot_fallback.py \
    tests/test_nb_snapshot_parity.py tests/test_nb_fresh_home.py tests/test_snapshot_paper_frame.py
```

All run on synthetic registries and frames, except that the control-pool stopping and distillation-target tests also read the committed sidecars
and two committed fixtures: the StrongREJECT judges' item rows of ten runs (`paper/tests/data/ctlstop_haiku_oracle.json`) and
three GSM8K cells' per-item outcomes (`paper/tests/data/ctlstop_gsm8k_oracle.json`), the threshold and distillation-target tests a
few rows of the committed snapshot by run id (the distillation-target tests also every pinned run's row and judgment there), the
generator-comparison tests their committed file (`data/appendix_generator_rows.json`; `tests/test_appendix_generator_gate.py` pins its
counts, the verdict and the figure's cell where the landing gate collects it), the judge-comparison tests theirs
(`data/appendix_judge_comparison.json`: its builder's provenance and its pinned payload, registry copy and value-file hashes, the numbers
the appendix quotes, and the two parts' decoded PNG pixels against the delivered figure's; `JUDGE_COMPARISON_DELIVERED` names a directory
of the delivered PNGs to compare them pixel by pixel; `tests/test_appendix_judge_comparison_gate.py` runs their core checks where the
landing gate collects them), the other appendix analyses' tests their committed files, and the capability error tests a few runs of
the committed snapshot's per-item results as well. `SISTEER_SLOW_TESTS=1` adds
`paper/tests/test_registry_nb_ctlstop_appendix_slow.py`, `paper/tests/test_registry_nb_si_target_appendix_slow.py` and
`paper/tests/test_registry_nb_si_target_battery_slow.py`, which run the notebook's load chain on the snapshot (about two minutes
each), `paper/tests/test_registry_nb_pooled_ci_slow.py`, which pins the committed frame's pooled intervals (every pooled run's units
reproduce its reading, the item ids align, the cells without item rows, the means unchanged, a few intervals;
`tests/test_pooled_ci_gate.py`, where the landing gate collects it, drives the same item readers on three of the frame's anchor pools
copied into `paper/tests/data/pooled_ci_item_fixture.json`), and `paper/tests/test_appendix_generator_slow.py`, which runs the same
chain and then anchors the generator comparison's committed file, and full and partly filed synthetic files, on Figure 11's real
frames. Run the multi-turn distillation-target slow file again whenever the committed snapshot or that figure's data file changes:
only it checks that figure's payloads against the snapshot's per-item rows. The pipeline that produced the data is described in
[`../../reproduce/README.md`](../../reproduce/README.md), the registry in
[`../../docs/registry/README.md`](../../docs/registry/README.md) and the padding ladder of Figure 3 in
[`../../docs/static_padding.md`](../../docs/static_padding.md).
