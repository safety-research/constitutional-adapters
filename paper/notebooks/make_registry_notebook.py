"""Build paper/notebooks/registry_qwen_figures.ipynb: the cell sources, in order, with pinned cell ids. `nbconvert --execute` adds the
outputs, and the committed notebook carries them. Run `python make_registry_notebook.py` from this directory."""
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

import os
# write the notebook beside THIS file, so the generator works from any checkout rather than one hardcoded worktree
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "registry_qwen_figures.ipynb")
cells = []


def md(cid, src):
    c = new_markdown_cell(src.strip("\n")); c["id"] = cid; cells.append(c)


def code(cid, src):
    c = new_code_cell(src.strip("\n")); c["id"] = cid; cells.append(c)


md("40484fe7", r"""
# Qwen3.5-4B paper panels, read from the results registry

The paper's figures redrawn from **one source**: the results registry (`~/registry/registry.sqlite`, opened read-only) when it is
present, else the paper-frame snapshot committed under `paper/snapshot/` - the header cell names the source it resolved. No separate
figure library, no export parquet, no per-source store is touched. Each panel is one frame call plus one plot call; the helper is `registry_nb.py`
beside this file (no hidden state). Rows are the registry's `v_paper_long` with `in_current_paper = 1` (the paper's own selection: the registry's roster +
ruling records), spelled with the registry's own tables (`sisteer.registry.paper`); the palette, display labels, model markers and panel rosters are
the paper's, set as constants in the helper.

**Numbering**: static defense beside the padding ladder is **Figure 3**, multi-query and multi-turn
defense **Figure 4**, alignment **Figure 5**, over-refusal and capability **Figure 6**, the frontier **Figure 7**. Figure 3 used to
carry all three jailbreak families and was split in two so padding could sit beside static; everything after it moved up by one. The main figures are Qwen3.5-4B on its paper lane (instruct-direct: objects fit on the checkpoint they are served on); the
**appendix figures at the bottom draw all four models** on their paper lanes, in the main figures' form. Nothing but the plot and its
legend is on the canvas; provenance (model, lane, registry head) is printed under each figure.

Rules applied throughout: every LLM-judged reading is the **paper's judges'** (the default policy, `REGISTRY_NB_JUDGES=rejudge`): each
StrongREJECT (static, best-of-N, multi-turn) and OR-Bench / XSTest compliance reading re-derived from **minimax-m3**'s items and each
Petri reading from **glm-5.3**'s, the re-judges filed as second judgments, by the filed row's own rule; the grader rule still picks which
filed judgment a cell re-derives from (a Sonnet-graded StrongREJECT judgment is excluded and counted; a cell with only Sonnet judgments is a
gap, never a point), and a run with no complete judgment under the paper's judge is blanked and named (`R.JUDGE_REPORT`), never drawn
with the filed number. `REGISTRY_NB_JUDGES=primary` renders the filed Claude numbers (`reproduce/10_figures.sh --judges primary`), which
may not write the paper's `figures/`. A cell with **several fit seeds is one point per seed**, each with its own interval, never pooled
(`fit_replicate_label`); the frontier is **relative to the unsteered model**, subtracted per eval, over the two multi-turn attacks, and reads
**the dose ladders, every rung**, never a campaign pass (the composition of the paper's frontier figure, `figQ_frontier_absolute_origin`). The
unsteered anchor stays visible beside every delta (`anchor_value` in the tables); a cell whose anchor leaves under 10 pp of room is
**ceiling-forced** and carries "≤" on the canvas (the multi-turn anchors on Gemma, Llama and Nemotron sit at 93–100 % defense, so a flat
multi-turn delta there is not evidence of safety); multi-turn is never pooled with the statics; the share of responses at the token cap
rides beside the value as "†n%" and is never used to filter or discount.

Vocabulary is the registry's: **model / object / eval / instrument / run / judgment / judge**. Every panel marks what the paper shows but the
registry frame lacks (a glyph per object × eval, with the reason: no run, not in the paper roster, superseded, held out by a ruling, not
yet judged, wrong judge, or no unsteered run to compare with) rather than dropping it. Degenerate and at-cap shares ride beside the values. Queries are cached under `_cache/` keyed by the registry head (`MAX(ingest_seq)` of
`record_ingest`), so a re-run against an unchanged registry is instant and a changed one re-reads. Run:
`reproduce/10_figures.sh --source snapshot` from the repository root, or, with the repo's interpreter captured first (uv sync does
not activate the venv, so a bare `python` after the cd would be the system one), `PY="$PWD/.venv/bin/python"; cd paper/notebooks &&
"$PY" make_registry_notebook.py && env REGISTRY_NB_SOURCE=snapshot "$PY" -m nbconvert --execute --to html
--ExecutePreprocessor.timeout=3600 registry_qwen_figures.ipynb` (about ten minutes from the committed snapshot). See `README.md` for the panel list and what is not covered yet.
""")

code("583e3f04", r"""
import os, subprocess, sys, time
T0 = time.time()
import numpy as np, pandas as pd, matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
import registry_nb as R
print = R.redacting_print      # every line this notebook prints passes the display-time redaction of internal session names (registry_nb.redact_session_names)
R.style()
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 200)
REGISTRY_CODE = subprocess.run(["git", "-C", os.path.dirname(R.SISTEER_SRC), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
print(f"registry source {R.SOURCE} · {R.SOURCE_DESC} · head {R.head()} · sisteer.registry from {R.display_path(R.SISTEER_SRC)} @ {REGISTRY_CODE}")
print("models:", ", ".join(f"{m} = {R.MODEL_LABEL[m]} on the {R.LANE_LABEL[R.PAPER_LANE[m]]} lane" for m in R.MODEL_ORDER), f"· Qwen = {R.MODEL_ID}")
""")

md("12cd66ed", r"""
## The frames

Four reads, all filtered in SQL (the views cost 1–3 minutes unfiltered):

- `paper_all` — `v_paper_long`, the four paper models, `in_current_paper = 1`: every measurement of every run the paper's selection flags.
  `paper` is its Qwen slice.
- `frontier` — `v_paper_long`, Qwen, the frontier's evals (Crescendo, ActorAttack, the four static attacks, XSTest, OR-Bench-Hard)
  **whatever the selection state**, so the frontier draws every rung of the registry's dose ladders.
- `pool` — `v_paper_long`, Qwen, every **unsteered** row whatever the selection state: where the paper frame holds no unsteered run for
  an eval on the lane (the multi-turn attacks and Petri today), Figures 3 / 4 borrow the registry's judged one from here, drawn open
  and counted.
- `runs_all` — `v_paper_current` joined to `run`, `object`, `eval`: every run of the four models with its lifecycle and selection state, so a
  missing panel point can say *why* it is missing.

`R.spell` adds the paper's spellings from the registry's own tables: `object_label` (the registry `paper_label`; a fitted object the objects
adapter labeled `other` is spelled from the campaign cell it was served under), `seed` (the object's `fit_replicate_label`), `lane` (the run's
own **`transfer_kind`** — the authoritative column — where it names a lane, else the served key's legacy mapping for the unsteered and prompted
rows), `dose` / `coefficient`, `headline` (the eval's headline metric at the paper's threshold 0.5; best-of-N at n = 75) and `judge_ok`
(the grader rule). A row is on its model's **paper lane** when `lane` equals `R.PAPER_LANE[model]`; an object recorded on another lane
only (the paired-write difference ReLU steer: fit on the base checkpoint, served on the instruct one) is drawn from that lane and named
under the figure (the paper's rule). Two replication axes, never averaged across: a fitted object's seed is `fit_replicate_label`, else
the run's `fit_seed`, else the view's `_sN` suffix; the prompted SIs, the Refusal steer and the unsteered model replicate over decode
only, so a cell's decode draws (`seed_base`, `trial_index`) average into one point (`R.pool_draws`). **The Refusal steer and the two
prompted SIs are one dot per cell** (`R.select_decode_only`): the mean of the condition's runs matched to the instrument of the fitted
objects in the cell (the same decode, judge, metric and threshold, item set, padding, and cap; rungs of a declared ladder are one cap only
where the fitted runs span it), averaged, with an item-clustered interval (the runs re-decode the same items, so the interval is built
over items: each item's mean over the runs, then the standard error over items). On the main figures an SI must also share the
fitted objects' serving setup where it has runs there (on Qwen its three seed-replicate runs); the Refusal steer's own trial views count.
The appendix counts a run from any serving setup. Where the fitted objects sit on several instruments, one instrument is averaged, the one
most of them use; where no run matches, the closest readings make the dot, so it always plots. Every other reading is set aside, never
drawn, and each figure lists its set-aside readings, cross-setup matches and fallback dots under it (the control-pool multi-turn figure
draws Figure 4's multi-turn dots, listed there); the category means, the tables and Figures 7 and 13 read the same runs. An anchor with several unsteered draws is their
mean, its interval item-clustered over the items they share and combined with the point's own (`anchor_draws`, `anchor_spread`,
`anchor_se` in the tables); an anchor run the source holds no item rows for makes it the identical-draws bound, which the route and the
table legend name. A seed pool counts the shared anchor once.
""")

code("ec6aacd8", r'''
ML = ",".join(f"'{m}'" for m in R.MODEL_ORDER)
# the run's decode seed rides beside the view's columns, and its store's completion hash: the generation key (`generation_store`) the
# readers collapse one generation filed twice on, spelled the same live and from the frame
DECODE = "v.*, r.seed_base, r.trial_index, r.store_completion_sha256"
# `unpooled_member` rows ride along with the paper rows (schema 22, 2026-10-02): the only readings of their cells until the producer's
# pooled rows land; `admit_provisional` keeps them only where the cell has no selected row and the load cell names them
PAPER_SQL = (f"SELECT {DECODE} FROM v_paper_long v JOIN run r ON r.run_id = v.run_id WHERE v.model_id IN ({ML}) "
             f"AND (v.in_current_paper = 1 OR (v.selection_state = '{R.PROVISIONAL_STATE}' AND v.metric_role = 'owner'))")   # owner rows only, as in_current_paper = 1 is by construction
# the grading code sha is on `judgment`, not on the paper view, and two seeds graded by different graders must not be
# pooled into one curve, so the frontier query carries it
FRONTIER_SQL = (f"SELECT {DECODE}, j.grading_code_sha FROM v_paper_long v JOIN run r ON r.run_id = v.run_id "
                "LEFT JOIN judgment j ON j.judgment_id = v.judgment_id WHERE v.model_id = '" + R.MODEL_ID + "' AND v.eval_id IN "
                "(" + ", ".join(f"'{e}'" for e in R.FRONTIER_EVALS) + ") AND v.metric_role = 'owner'")
POOL_SQL = f"SELECT {DECODE} FROM v_paper_long v JOIN run r ON r.run_id = v.run_id WHERE v.model_id = '{R.MODEL_ID}' AND v.paper_label = 'none' AND v.metric_role = 'owner'"
# `pc.comparison_ids_json` rides along (roster v9): the runs census is admitted to the paper lane by the same
# rule as the paper frame (`admit_lane` under ROSTER_NAMES_THE_PAPER_LANE reads the comparison ids), so an IT fit the roster
# moved to the transfer figures is not counted as a paper-lane run that went undrawn.
RUNS_SQL = f"""SELECT pc.in_current_paper, pc.selection_state, pc.decisive_action, pc.decisive_ruling_id, pc.held_out, pc.comparison_ids_json, r.run_id, r.model_id, r.lane_source_legacy, r.view_legacy, r.campaign_cell_id_legacy,
       r.transfer_kind, r.eval_id, e.eval_category, e.headline_metric_id, r.lifecycle_state, r.pass_label, r.strength_nominal, r.strength_coefficient,
       r.is_operating_point, r.extra_json AS run_extra_json, o.paper_label, o.object_kind, NULL AS metric_id, NULL AS threshold, NULL AS bon_n
FROM v_paper_current pc JOIN run r ON r.run_id = pc.run_id JOIN object o ON o.object_id = r.object_id JOIN eval e ON e.eval_id = r.eval_id
WHERE r.model_id IN ({ML})"""
# ONE ROW PER MEASUREMENT AT LOAD. The view names the owner metric row (`metric_role`, migration 0004, 2026-09-21): the
# frontier and unsteered queries read owners only and the paper query's `in_current_paper = 1` is owner-only by construction; the
# collapse below is the belt (contrast, then owner role, then a non-blank computation, then the latest ingested) and prints what it caught.
# THE SOURCE HOLDS EVERY JUDGE THE POLICY DRAWS (and, from the paper frame, their item rows), OR THE RENDER STOPS BY NAME
# (R.check_policy_judges: PolicyJudgesAbsent, SnapshotLacksJudgeItems); nothing is checked under `primary`
R.check_policy_judges()
# THE EVALS THE PAPER OMITS LEAVE AT LOAD (R.OMITTED_EVALS, 2026-11-14), at the query, before the synthesis and `spell`: no judge
# selection, synthesized row, JUDGE_REPORT entry, figure, table, mean, count or listing below reads them
_raw = {k: R.drop_omitted_evals(R.query(q, k))[0] for k, q in (("paper_all", PAPER_SQL), ("qwen_frontier", FRONTIER_SQL), ("qwen_unsteered", POOL_SQL))}
# A SELECTED RUN WITH ONLY A RE-JUDGE'S JUDGMENT AND NO FILED run_metric has no row in the paper view: under a re-judge policy its rows
# are synthesized from that judge's items (`synthesized_rows`) and join the frames their queries would have put them in; under
# `primary` it is a named gap (both printed)
_syn = R.synthesized_rows(_raw["paper_all"])
_raw = {k: R.spell(R.with_synthesized(v, _syn, k)) for k, v in _raw.items()}
paper_all, frontier, pool = (R.dedupe(_raw[k]) for k in ("paper_all", "qwen_frontier", "qwen_unsteered"))
# the full BigCodeBench / BFCL suites replace their 200-item subsets everywhere — applied at load so
# no figure can pick up the other set, and reported so the swap is visible rather than assumed
print("duplicate metric rows collapsed at load: " + " · ".join(f"{k} {len(_raw[k]) - len(v)} of {len(_raw[k])}" for k, v in zip(_raw, (paper_all, frontier, pool))))
paper_all, _prov = R.admit_provisional(paper_all)
if len(_prov):
    print(f"PROVISIONAL readings (schema 22 `unpooled_member`: a sampled_lane pass whose point has no pooled run yet; the registry does not count it a paper "
          f"row, this notebook reads it where the cell would otherwise be blank, and it leaves the render after the pooled rows land): "
          f"{int(_prov.runs.sum())} runs in {len(_prov)} cells — " + "; ".join(f"{R.MODEL_LABEL.get(m, m)} {R.LABEL.get(o, o)} on {', '.join(sorted(g.eval_id))}"
          for (m, o), g in _prov.groupby(['model_id', 'object_label'])) + f"; {paper_all.attrs.get('provisional_dropped', 0)} such rows set aside where the cell already has a selected reading.")
_n0 = len(paper_all); paper_all = R.prefer_item_set(paper_all)
if len(paper_all) < _n0:
    print(f"full suites replace the 200-item subsets: {_n0 - len(paper_all)} BCB/BFCL "
          f"200-item rows dropped from every figure; they stay in the registry as history.")
runs_all = R.spell(R.drop_omitted_evals(R.query(RUNS_SQL, "runs_all"))[0])
HEAD = paper_all.attrs["head"]
LANE_Q = R.PAPER_LANE[R.MODEL_ID]                                           # instruct_direct: the paper's lane for Qwen
for f in (paper_all, runs_all):
    f["on_lane"], f["lane_pulled"] = R.admit_lane(f)                      # the paper lane, plus an object recorded on another lane only (pulled, named)
for f in (frontier, pool):
    f["on_lane"] = R.on_paper_lane(f)
# OVER-REFUSAL IS THE LABEL MEAN EVERYWHERE: the frontier's benign axis, the Figure 6 over-refusal panel
# and its table, the appendix. The registry's headline for XSTest / OR-Bench-Hard is the refused rate, so every frame that
# carries over-refusal rows -- the paper frame, the frontier frame, and the unsteered pool the anchors borrow from -- is
# repointed HERE, before any headline filter, and the switch is ALL OR NOTHING across the three: a frame half on one metric
# compares arms against anchors read the other way (a 35 pp offset when it happened), and a pool on a different metric from the
# steered rows it anchors is the same defect spread over two frames. Incomplete coverage leaves every frame on the registry's
# headline and names the runs that are short, so the withheld switch is a list to act on rather than a number to look up.
# THE GATE COUNTS THE ROWS THE PANELS READ: coverage is over `selected` rows. The frontier frame carries
# 46 headline over-refusal runs with no label-mean row -- every one `r2_export` at the 512 cap, `superseded` or
# `not_in_roster`, drawn by nothing -- and the unscoped gate withheld the whole switch for them. The unsteered POOL does
# not gate at all: its rows are borrowed only through `anchor_candidates`, which matches the arm's metric family, so a
# pool row without the label mean is not a mixed anchor, it is simply not a candidate; its coverage is reported below.
_or_rep = {k: R.repoint_all(v, R.OR_LABEL_MEAN, scope=R.selected_scope(v)) for k, v in (("paper_all", paper_all), ("frontier", frontier))}
_or_rep["pool"] = R.repoint_all(pool, R.OR_LABEL_MEAN)
OR_METRIC_OK = all(bool(_or_rep[k].attrs.get("repoint_complete")) for k in ("paper_all", "frontier"))
_or_cov = " · ".join(f"{k} {v.attrs['repoint_coverage'][0]}/{v.attrs['repoint_coverage'][1]}" for k, v in _or_rep.items())
if OR_METRIC_OK:
    paper_all, frontier, pool = _or_rep["paper_all"], _or_rep["frontier"], _or_rep["pool"]
    print(f"OVER-REFUSAL METRIC: {R.OR_METRIC} on every over-refusal reading (the Figure 6 panel and its table, the frontier's x, "
          f"the appendix) — full refusal 1, partial 0.5, compliance 0, the mean over judged items; an at-cap reply is scored by its "
          f"label. Coverage, runs moved / headline runs read (paper_all and frontier over selected rows; the pool is every borrowable "
          f"unsteered run and does not gate): {_or_cov}.")
    _unread = {k: R.repoint_gaps(v, R.OR_LABEL_MEAN) for k, v in (("frontier", frontier), ("pool", pool))}
    for _k, _g in _unread.items():
        if len(_g):
            print(f"  {_k}: {len(_g)} headline over-refusal runs OUTSIDE the read set carry no {R.OR_METRIC} row and stay on the refused "
                  f"rate -- {', '.join(f'{a} at {int(b):,} ({c})' for (a, b, c), _n in _g.groupby(['lane_source_legacy', 'token_cap_tokens', 'eval_id']).size().items())}; "
                  f"none is drawn, and an unsteered one cannot anchor a label-mean rung: the anchor pools are matched on the metric family.")
else:
    print(f"!! OVER-REFUSAL METRIC: {R.OR_METRIC} coverage INCOMPLETE ({_or_cov}) — every frame stays on the registry's headline "
          f"(the refused rate). The over-refusal panel, its table and the frontier's x below read THAT. Headline runs with no "
          f"{R.OR_METRIC} row:")
    for _k, _v in (("paper_all", paper_all), ("frontier", frontier)):
        _gap = R.repoint_gaps(_v, R.OR_LABEL_MEAN, scope=R.selected_scope(_v))
        if len(_gap):
            print(f"  {_k}: {len(_gap)} runs"); R.show(_gap.drop(columns=[c for c in ("run_id",) if c in _gap.columns]))
_pl = paper_all[paper_all.lane_pulled & paper_all.in_current_paper.eq(1) & paper_all.object_label.isin(R.OBJECTS)]
PULLED = sorted(set(zip(_pl.model_id, _pl.object_label, _pl.lane)))                    # roster objects drawn from a lane other than the paper lane
paper = paper_all[paper_all.model_id == R.MODEL_ID]
op_all = paper_all[paper_all.on_lane & paper_all.at_operating_point & paper_all.headline]     # every model's operating point on its paper lane
# VENDOR DECODE WINS WHERE A CELL HAS ONE. Admitting the prompted SIs' seed-replicate draws leaves the
# older grandfathered T = 1.0 run in the same cell, and pooling the two averages across a decode boundary.
_before = len(op_all); op_all = op_all[R.prefer_vendor_decode(op_all)]
if len(op_all) < _before:
    print(f"vendor decode preferred: {_before - len(op_all)} grandfathered T=1.0 row(s) set aside in cells that carry a "
          f"vendor-preset reading")
# THE CANONICAL BATCH-8 FITS ARE RETIRED: with three batch-64 seeds fit per object, the paper's fitted objects are
# those seeds, and the earlier single canonical fit leaves the panels. Prompted arms and the unsteered model are not fits and keep every run.
op_all, _retired = R.retire_canonical_fits(op_all)
# A ROW BELONGS TO A FIGURE BY COMPARISON ID (the registry's contract; enabled once roster v5 made it a no-op).
# The Qwen main frame reads the main-figure comparisons, the all-model appendix reads its own five plus the figR_* set for the
# ReLU steers on the companion models. Verified against the live frame at head 153997 before enabling: 0 rows dropped from
# the Qwen frame, every Petri-behavior and best-of-N row carrying a main-figure id, and all 369 empty-list rows admitted by
# their include ruling. So it changes nothing today, which is the point - it is a guard against a future roster change
# quietly adding someone else's rows to these panels, not a correction to the present ones. NEVER applied to the frontier
# frame, which reads every selection state by design; the registry's contract grants that exemption.
op_all = R.comparison_filter(op_all, R.APPENDIX_COMPARISONS + R.MAIN_FIGURE_COMPARISONS, "all models, any figure")
# THE PADDED ROWS DO NOT JOIN THESE PANELS. The padding wave's four families are exactly this figure's
# static category, and it has no pad-0 arm: its unpadded comparison would have to come from waves at a different
# window and cap, so drawing them together reads serving geometry as an effect of padding. The two padded levels compare
# cleanly with EACH OTHER only.
op_all, _padded = R.drop_padded(op_all)
# A CELL HELD OUT BY RULE (R.EXCLUDED_CELLS, none today) leaves every figure and table of its model; dropped here, before anchoring,
# so no panel, mean or table sees it.
op_all, _excluded = R.drop_excluded_cells(op_all)
if len(_excluded):
    print("cells held out by rule (R.EXCLUDED_CELLS), dropped before anchoring: "
          + "; ".join(f"{m} × {e}: {n} rows" for (m, e), n in _excluded.groupby(["model_id", "eval_id"]).size().items()))
if len(_padded):
    print(f"padded rows held out of the panels: {len(_padded)} at pad lengths "
          f"{', '.join(f'{int(v):,}' for v in sorted(set(_padded['pad'])))} tokens, over "
          f"{', '.join(sorted(set(_padded.eval_id)))}. They are a within-instrument series of their own and are NOT "
          f"comparable with the unpadded rows here, which come from waves at another window and cap.")
op = R.comparison_filter(op_all[op_all.model_id == R.MODEL_ID], R.MAIN_FIGURE_COMPARISONS, "Qwen main figures")
print(f"canonical batch-8 fit rows retired from the panels: {len(_retired)} (" + ", ".join(f"{k} {v}" for k, v in _retired.groupby("model_id").size().items()) + ")" if len(_retired) else "canonical batch-8 fit rows retired: 0")
runs_lane = runs_all[runs_all.on_lane]
runs_id = runs_lane[runs_lane.model_id == R.MODEL_ID]
POOL_Q = R.pool_fence(pool[R.complete(pool) & (pool.in_current_paper != 1)])   # judged unsteered runs outside the frame, any lane (anchors match the steered row's lane); shared standups fenced out
if POOL_Q.attrs.get("pool_fenced"):
    print(f"unsteered pool: {POOL_Q.attrs['pool_fenced']} shared-standup rows (bcbw `*_shared*`: a shared-serving comparison against the solo runs, not paper data) fenced out of the borrowable anchors")
# the grader rule: excluded judgments, and the cells left with no admissible judgment at all
bad = op_all[~op_all.judge_ok]
WJ = set(zip(bad.model_id, bad.object_label, bad.eval_id)) - set(zip(op_all[op_all.judge_ok].model_id, op_all[op_all.judge_ok].object_label, op_all[op_all.judge_ok].eval_id))
PROV = lambda m=R.MODEL_ID: (f"{R.MODEL_LABEL[m]} · {R.LANE_LABEL[R.PAPER_LANE[m]]} lane · operating dose · registry head {HEAD} · unsteered reading = the roster-matched unsteered runs (a ladder standup's own none anchors its source alone, registry rule 0027)"
                             + "".join(f" · {R.LABEL.get(o, o)} drawn from the {R.LANE_LABEL.get(l, l)} lane, its only one" for mm, o, l in PULLED if mm == m))
print(f"paper_all {len(paper_all)} rows / {paper_all.run_id.nunique()} runs · Qwen {len(paper)} rows · operating-point headline rows on the paper lanes "
      f"{len(op_all)} ({len(op)} Qwen) · frontier {len(frontier)} · unsteered pool {len(pool)} ({len(POOL_Q)} borrowable) · runs {len(runs_all)} ({len(runs_id)} Qwen on its lane) · head {HEAD}")
print(f"grader rule {R.JUDGE_POLICY}: {len(bad)} operating-point headline rows excluded; {len(WJ)} cells left with no admissible judgment")
if len(bad):
    R.show(bad.groupby(["model_id", "object_label", "eval_id", "judge_model"]).size().rename("rows excluded").reset_index())
print(f"rows whose transfer_kind and served-key lane disagree: {int(paper_all.lane_conflict.sum())} · rows on no lane (listed, not drawn):")
print(paper_all[paper_all.lane.isna()].groupby(["model_id", "view_legacy", "transfer_kind"]).size().to_string() if paper_all.lane.isna().any() else "  none")
print("paper-frame rows per model and lane:"); print(paper_all.groupby(["model_id", "lane"], dropna=False).size().to_string())
print("objects drawn from a lane other than the model's paper lane (recorded on that lane only):", [f"{R.MODEL_LABEL[m]} / {R.LABEL.get(o, o)} ← {R.LANE_LABEL.get(l, l)}" for m, o, l in PULLED] or "none")
''')

md("9bf536ac", r"""
## Coverage: what each Qwen panel draws on, and what is not there

Per eval: the sources and objects with a headline row at the operating point, the paper runs not yet judged, and, per roster object, the
registry's answer where the paper frame has nothing (`absent` = no run; `not in roster` = judged runs the paper's selection leaves out, with
the selection state; `excluded by ruling` = judged, held out by a decisive ruling, named below; `withdrawn` = every judged run withdrawn by
its producer, out of the paper view; `superseded` = every judged run superseded by a newer one; `unjudged` = runs without a judgment yet; `wrong judge` = graded only by a judge the paper excludes).
""")

code("6fec6474", r"""
PANEL_EVALS = {"deltas by category": [e for c in R.CATEGORIES for e in list(R.CATEGORIES[c][0]) + R.summary_evals(c)],
               "attacks": ["pasttense", "codeattack", "renellm", "grok420", "strongreject_bon", "pair", "crescendo", "actorattack"],
               "over-refusal": ["xstest", "or_bench_hard"], "capability": ["mmlu", "gsm8k", "math500", "ifeval", "bigcodebench", "bfcl"]}
ALL_EVALS = list(dict.fromkeys(e for es in PANEL_EVALS.values() for e in es))
opd = R.dedupe(op[op.judge_ok])
cov = []
for e in ALL_EVALS:
    x = opd[opd.eval_id == e]; pr = runs_id[(runs_id.eval_id == e) & (runs_id.in_current_paper == 1)]
    cov.append(dict(eval_id=e, panels=" ".join(p[:4] for p, es in PANEL_EVALS.items() if e in es), headline_rows=len(x),
                    sources=", ".join(sorted(str(s) if pd.notna(s) else "(native)" for s in x.lane_source_legacy.unique())), objects=", ".join(o for o in ["none"] + R.OBJECTS if o in set(x.object_label)),
                    seeds=", ".join(sorted(s for s in x.seed.unique() if s)), paper_runs_unjudged=int((~R.complete(pr)).sum()),
                    registry_only_judged_runs=int(((runs_id.eval_id == e) & (runs_id.in_current_paper != 1) & R.complete(runs_id)).sum())))
R.show(pd.DataFrame(cov))
gaps_all = R.gap_status(runs_id, R.drawn_set(opd), ["none"] + R.OBJECTS, ALL_EVALS, wrong_judge=WJ)
R.show(gaps_all.pivot(index="object_label", columns="eval_id", values="status").reindex(["none"] + R.OBJECTS)[ALL_EVALS])
held = gaps_all[gaps_all.status == "excluded by ruling"]
print(f"cells held out by a decisive ruling ({len(held)}); the ruling ids:"); R.show(held[["object_label", "eval_id", "ruling", "selection"]])
nonhead = paper_all[paper_all.on_lane & paper_all.at_operating_point & ~paper_all.headline & R.complete(paper_all)]
print("in the paper frame at the operating dose but NOT the eval's headline metric at the paper's threshold / n (in the registry, not drawn; e.g. the paired best-of-N climb rows, which await a decode selection):")
R.show(nonhead.groupby(["model_id", "object_label", "eval_id", "metric_id", "threshold", "bon_n"], dropna=False).size().rename("rows").reset_index())
extra_objects = sorted(set(opd.object_label) - set(R.OBJECTS) - {"none"})
print(f"objects in the Qwen paper frame but not on this notebook's roster (not drawn): {extra_objects}")
print(f"measurements filed twice (two computation ids; the contrast-carrying one is kept): {len(op[op.judge_ok]) - len(opd)}")
print(f"selection states among instruct-lane Qwen runs the paper leaves out: {runs_id[runs_id.in_current_paper != 1].selection_state.value_counts().to_dict()}")
# THE HEADLINE ROWS THE RENDER'S JUDGES HAVE NO READING OF (every model, operating dose, paper lane): blanked, never drawn with the filed
# number, each named with its reason; R.JUDGE_REPORT["frames"][i]["blanked"] holds every frame's, the undrawn metrics included
_bl = op_all[op_all["judge_blanked"].fillna(False).astype(bool)] if "judge_blanked" in op_all.columns else op_all.iloc[0:0]
print(f"headline rows at the operating dose with no reading under the render's judges ({R.judge_policy()['spec']}): {len(_bl)}")
if len(_bl):
    R.show(_bl[["model_id", "object_label", "eval_id", "metric_id", "seed", "run_id", "judge_model"]]
           .sort_values(["model_id", "eval_id", "object_label", "seed"]).reset_index(drop=True))
""")

md("b6d74ec9", r"""
## Figure 3 — static jailbreak defense, and the padding ladder (Qwen)

For each steered object and eval, the delta against the unsteered run in percentage points, oriented so that up is better (defense
success = 100 − ASR; benign compliance = 100 − over-refusal label mean, where a full refusal scores 1, a partial ½ and compliance 0). Small marks are the individual evals, the large mark after the
separator is the category's unweighted mean over them with an analytic interval (a black ring = a mean over fewer evals than the category
names); several seeds of one object are small marks, and the mean column averages them into one point. **Each fitted object is its three batch-64 seeds** - the earlier canonical batch-8 fit is retired from every panel; the unsteered model is not a fit and keeps every run as an anchor draw. **The Refusal steer and each prompted SI are one dot per eval** (and per rung on the padding panel), the mean of the condition's runs matched to the fitted objects' instrument (decode, judge, metric, threshold, item set, padding, cap) and, for an SI, their serving setup (on Qwen the SIs' three seed-replicate runs, at the battery's cap and vendor preset; the Refusal steer's vendor-preset trials on its own views); where no run matches, the closest readings make the dot. Readings at another cap, decode or item set are set aside, and the lines under each figure list them with every fallback dot. **Every point on an eval is differenced against that eval's common anchor**: the mean over every unsteered run of the eval in the paper frame at the point's generation cap (the campaign's run, the seed views' runs, the paired views' runs, the decode draws, pooled as draws, the anchor's uncertainty item-clustered: each item's mean over the runs, then the standard error over items), and each point's interval combines its own with the anchor's; the mean column's interval pools the fits' runs item by item (the seeds read the same items), adds the variance between the fits and counts the shared anchor once. The producer's own filed contrast is kept beside each row as `filed_delta` for cross-checking but is not the plotted value. An open mark means the frame held no unsteered run for that eval and one was borrowed from the registry outside the selection; a pool spanning two decode spellings is flagged `mixed_decode` in the tables.
""")

code("ac902304", r"""
d = R.anchored(op, pool=POOL_Q)                           # every steered Qwen headline row with its delta vs the unsteered run
CAT_Q = R.categories_for(R.MODEL_ID)                                        # Qwen's own panels: no column for its excluded cells
cm = R.category_means(d, categories=CAT_Q)
gaps_q = R.gap_status(runs_id, R.drawn_set(d[d.d_y.notna()]), R.OBJECTS, ALL_EVALS, wrong_judge=WJ)
# FIGURE 3 IS SINGLE-TURN BESIDE THE PADDING LADDER, and the multi-query / multi-turn panels
# it used to carry are Figure 4 below. "padding" is not an eval category — it has its own x, in filler tokens — so it
# rides in as a PANEL HOOK, which is also what keeps its readings out of every category mean: they never enter `d`.
_pad_src = R.spell(R.query(f"SELECT {DECODE} FROM v_paper_long v JOIN run r ON r.run_id = v.run_id "
                           f"WHERE v.model_id = '{R.MODEL_ID}' AND v.eval_id = 'grok420' AND v.metric_role = 'owner'",
                           "qwen_padding"))
pad_pts = R.padding_points(_pad_src)
_pad_base = {float(r.pad): float(r.base_defence) for _, r in pad_pts.iterrows()} if len(pad_pts) else {}

def _pad_panel(ax):
    if len(pad_pts):
        R._draw_padding(ax, pad_pts, compact=True)
    else:                                    # say so on the canvas rather than leaving a blank box the reader must explain
        ax.set_axis_off(); ax.text(0.5, 0.5, "no rows behind the padding contract yet", ha="center", va="center",
                                   transform=ax.transAxes, color=R.INK2, fontsize=11)

FIG3 = [["static", "padding"]]                               # static beside the padding ladder
JB3 = ["static"]                                             # the CATEGORY keys of Figure 3 — "padding" is a hook, not a category
_pad_rungs_present = [p for p in R.PAD_RUNGS if len(pad_pts) and p in set(pad_pts.pad.astype(int))]
R.delta_figure(cm, d, FIG3, R.OBJECTS, gaps_q, "fig3_single_turn_padding",
               panel_hooks={"padding": _pad_panel},
               # NO WIDER THAN ITS LEGEND: the legend spans ~16.8 in of the 19.4 in the two panels
               # took, so the padding slot dropped from 4 units to 2.7 for three rungs; static keeps its 5. With five rungs
               # drawn it had widened to 4.5, too wide again: 0.6 per rung, 3.0 for five.
               hook_widths={"padding": 0.6 * max(len(_pad_rungs_present), 3)}, categories=CAT_Q); plt.show()
if len(pad_pts):
    print(f"padding rungs drawn: {', '.join(f'{p:,}' for p in _pad_rungs_present)} of the ladder "
          f"{', '.join(f'{p:,}' for p in R.PAD_RUNGS)}" + ("" if len(_pad_rungs_present) == len(R.PAD_RUNGS) else
          " — the missing rungs appear the render after their rows land"))
    _pend = R.padding_pending_rungs(pad_pts)
    _held = R.padding_held(pad_pts)
    if len(_held):
        print("padding: held out of the panel and its pooled means by a registry hold-out (the table lists them with \\S): "
              + "; ".join(f"{R.LABEL.get(r.object_label, r.object_label)} {r.seed or ''} at {int(r.pad):,} filler tokens".replace("  ", " ") for r in _held.itertuples())
              + " — the pad-0 difference-ReLU reading is degenerate (ruling r2v4-hold-out-static-padding-p0-difference-relu-grok420-degenerate); every drawn value is the judge's ASR over all goals, degenerate included.")
    if _pend:      # graded readings on the padding instrument whose roster entries are drafted (v9) but not loaded: drawn, marked
        print(f"rungs {', '.join(f'{p:,}' for p in _pend)} are read from rows the roster does not yet name (`not_in_roster`; roster v9 "
              f"names their instrument ids) — the unsteered anchor and the prompted / difference arms there; drawn now, marked ‡ in "
              f"the table, and the mark retires by itself the render after v9 loads. Superseded / withdrawn rows are not read.")
    _seeds = pad_pts[pad_pts.seed.astype(str).ne("")].groupby("object_label").seed.nunique()
    print("seeds per fitted arm on the padding ladder: " + ", ".join(f"{R.LABEL.get(o, o)} {n}" for o, n in _seeds.items()))
if len(pad_pts):
    print("Right panel — Δ defense vs the unsteered model AT THE SAME RUNG, in pp. The unsteered baseline is itself "
          "NON-MONOTONIC " + " → ".join(f"{_pad_base[p]:.1f}%" for p in sorted(_pad_base)) + ", so this is differenced "
          "per rung and must not be read as 'padding degrades everything'. Intervals are Newcombe for a DIFFERENCE of "
          "proportions, independent form (the frame carries no per-goal pairing), which is conservative wherever the "
          "arms correlate.")
    _dg = pad_pts[pad_pts.held | (pad_pts.degen.fillna(0) > 0.20)]
    if len(_dg):
        print("heavily degenerate or held-out cells, named rather than drawn: "
              + "; ".join(f"{r.object_label} at pad {int(r.pad):,}"
                          + (f" ({100 * r.degen:.0f}% degenerate)" if r.degen == r.degen else " (held out)")
                          for _, r in _dg.iterrows()))
    print(R.padding_table(pad_pts, R.OBJECTS))               # the padding table
    R.show(pad_pts.assign(delta=pad_pts.delta.round(1), lo=pad_pts.lo.round(1), hi=pad_pts.hi.round(1))
           [["object_label", "pad", "defence", "delta", "lo", "hi", "held"]], n=30)
else:
    print("padding ladder: no rows behind the padding contract in this frame yet.")
# ONE TABLE PER FAMILY: static, multi-query and multi-turn tabulate
# separately, the way over-refusal and capability already do, so no table carries another family's columns.
_JB = {"static": ("Static", "single-turn")}                  # slug keeps the OLD \label so \ref in the paper draft still resolves
for _cat, (_name, _slug) in _JB.items():
    # THE POOLED (seed-averaged) TABLE IS GONE: the per-seed table below carries the same
    # numbers with the spread explicit, so the pooled one only restated them. Captions are bare.
    print(R.table_evals_seeds(d, [[_cat]], R.OBJECTS,
          f"{_name} defense for Qwen3.5-4B.", f"tab:qwen-{_slug}-seeds"))
print(PROV())
print("delta route per row:", d.route.value_counts().to_dict())
_ur = R.unscaled_rates(op)
print("headline metrics that look like proportions but carry no plotted-unit scale (a best-of-N-shaped defect; empty is correct):")
R.show(_ur if len(_ur) else _ur)
print("unsteered run compared with, by source:", d.anchor_source.value_counts(dropna=False).to_dict())
dups = d.attrs.get("cell_duplicates")
print(f"cells where the selection admits two runs for one (object, seed, dose, eval) - one kept (the one with a registry contrast, else the newest), {0 if dups is None else len(dups)} set aside:")
if dups is not None and len(dups): R.show(dups[["object_label", "seed", "dose", "eval_id", "view_legacy", "lane_source_legacy"]])
# THE DECODE-ONLY CONDITIONS ARE ONE DOT PER CELL (R.select_decode_only): what each dot averages, then every fallback dot, cross-setup
# match and set-aside reading of Figure 3's evals (Figures 4-6 print theirs under them)
DEC_CELLS, DEC_ASIDE = d.attrs["decode_only_cells"], d.attrs["decode_only_set_aside"]
print("decode-only dots on Figures 3-6, one per condition per eval, by the number of runs each averages:", R.decode_only_mark_line(d))
print(R.decode_only_disclosure(DEC_CELLS, DEC_ASIDE, evals=[e for c in JB3 for e in CAT_Q[c][0]], title="Figure 3"))
# ...and the padding panel's, one dot per condition per rung (R.padding_points selects at each rung against the fitted runs there)
print(R.decode_only_disclosure(pad_pts.attrs.get("decode_only_cells"), pad_pts.attrs.get("decode_only_set_aside"), title="Figure 3, padding panel"))
DCOLS = ["eval_id", "object_label", "seed", "y", "anchor_value", "anchor_draws", "anchor_spread", "d_y", "d_lo", "d_hi", "headroom", "ceiling", "frac_degenerate_union5", "frac_at_token_cap", "route", "anchor_source"]
print("the mean points drawn (one per object: the seeds averaged; n_seeds says how many) - ceiling = some eval's unsteered run leaves under 10 pp of room; at_cap / degen = the largest shares among the evals:")
R.show(R.pool_seeds(cm[cm.category.isin(JB3)]).drop(columns="model_id").round(2))
print("per seed (the small marks' means; anchors = the unsteered readings):")
R.show(cm[cm.category.isin(JB3)].drop(columns="model_id").round(2))
print("per eval, the steered reading (y), the unsteered anchor it is compared with (anchor_value) and the delta (d_y), all in plotted units:")
R.show(d[d.eval_id.isin([e for c in JB3 for e in R.CATEGORIES[c][0]])][DCOLS].sort_values(["eval_id", "object_label", "seed"]).round(1))
blocks = paper_all.delta_block.dropna() if "delta_block" in paper_all.columns else pd.Series(dtype=object)
with_anchor = blocks[blocks.map(lambda b: isinstance(b, dict) and isinstance(b.get("anchor"), dict))]
print(f"producer delta blocks on the paper frame's metric rows: {len(blocks)}, of which {len(with_anchor)} declare an unsteered anchor (the only ones a delta may come from; "
      f"route 'producer delta block'); the rest by their stated reason:")
print(blocks[~blocks.index.isin(with_anchor.index)].map(lambda b: str((b or {}).get("why_not_a_contrast_row") or (b or {}).get("why_no_run_contrast"))[:100]).value_counts().to_string())
if len(with_anchor):
    print("first block with an anchor:", R.redact_session_names(with_anchor.iloc[0]))   # a producer-filed `delta_block`, verbatim registry text
""")

md("4f2ab8d1", r"""
## Figure 4 — multi-query and multi-turn jailbreak defense at the operating dose (Qwen)

The other half of what used to be Figure 3. Conventions are exactly Figure 3's: per object and eval the delta
against the unsteered run in percentage points, oriented so up is better defense; small marks are the individual evals, the large mark
after the separator is the category's unweighted mean with an analytic interval, and a black ring marks a mean over fewer evals than the
category names. Each fitted object is its three batch-64 seeds; every point is differenced against that eval's common anchor.
""")

code("7c1de904", r"""
FIG4 = [["multi-query", "multi-turn"]]                       # multi-query beside multi-turn
JB4 = [c for row in FIG4 for c in row]
R.delta_figure(cm, d, FIG4, R.OBJECTS, gaps_q, "fig4_multi_turn", categories=CAT_Q); plt.show()
# one table per family, as in Figure 3
for _cat, (_name, _slug) in {"multi-query": ("Multi-query", "multi-query"),
                             "multi-turn": ("Multi-turn", "multi-turn")}.items():
    print(R.table_evals_seeds(d, [[_cat]], R.OBJECTS,
          f"{_name} defense for Qwen3.5-4B.", f"tab:qwen-{_slug}-seeds"))
print(R.decode_only_disclosure(DEC_CELLS, DEC_ASIDE, evals=[e for c in JB4 for e in CAT_Q[c][0]], title="Figure 4"))
print("the mean points drawn (one per object: the seeds averaged; n_seeds says how many):")
R.show(R.pool_seeds(cm[cm.category.isin(JB4)]).drop(columns="model_id").round(2))
print("per seed (the small marks' means; anchors = the unsteered readings):")
R.show(cm[cm.category.isin(JB4)].drop(columns="model_id").round(2))
print("per eval, the steered reading (y), the unsteered anchor it is compared with (anchor_value) and the delta (d_y), all in plotted units:")
R.show(d[d.eval_id.isin([e for c in JB4 for e in R.CATEGORIES[c][0]])][DCOLS].sort_values(["eval_id", "object_label", "seed"]).round(1))
""")

md("5e61e1ab", r"""
## Figure 5 — Petri at the operating dose (Qwen)

Same construction, Petri alone: **all six behaviors separately, then the pooled score on the right**.
As in Figures 3–4, each fitted object is its three batch-64 seeds (the canonical batch-8 fit is retired) and every point on an eval is
differenced against that eval's **common anchor**: the mean over every unsteered run of the eval at the same generation cap, its uncertainty
in each point's interval.
The pooled column is the registry's own `petri_pooled` run - the producer's pooling over the behaviors, not an unweighted mean of the six
columns - so it is labeled "pooled", not "mean". Petri is drawn as the change in the **alignment score**, 11 minus the 1–10 misalignment
score (so up is more aligned and the scale stays 1–10). Where several FITS of one recipe exist (the paired ReLU's three seeds), the mean
point carries the between-fit variance in its interval, beside the fits' runs pooled item by item, and names the spread when the fits disagree by more than 10 in the panel's unit
(for example 41.3 pp vs 1.5 pp across three fits on PAIR).
""")

code("624d0e27", r"""
FIG5 = [["Petri"]]                                  # misalignment, one panel: the six behaviors, then pooled, seven columns
FIG6 = [["over-refusal", "capability"]]             # capabilities: the cost side, over-refusal beside capability
# tick labels at the standard size (never shrunk), which needs 2.2 in per column for "reasoning unfaithfulness" and
# "self-preservation" to clear each other -- 16.8 in for the panel
R.delta_figure(cm, d, FIG5, R.OBJECTS, gaps_q, "fig5_misalignment", unit=2.2, categories=CAT_Q); plt.show()
_mi = R.missing_intervals(d, [e for c in FIG5 for cat in c for e in CAT_Q[cat][0]])
if len(_mi):
    print(f"UNMEASURED intervals on this panel: {int(_mi.rows.sum())} drawn rows carry a value but no interval (their filed contrast is not current in the registry while "
          f"the adapters re-anchor; petri_score has no per-item data to make one from) — drawn as a mark with no bar, tabulated without ±: "
          + "; ".join(f"{R.LABEL.get(r.object_label, r.object_label)} on {r.eval_id}" for r in _mi.itertuples()))
for _cat, (_name, _slug, _unit) in {
        "Petri": ("Petri", "petri", "the change in the alignment score, 11 - the misalignment score, on its 1--10 scale")}.items():
    print(R.table_evals_seeds(d, [[_cat]], R.OBJECTS,
          f"{_name} for Qwen3.5-4B.", f"tab:qwen-{_slug}-seeds"))
print(R.decode_only_disclosure(DEC_CELLS, DEC_ASIDE, evals=[e for row in FIG5 for c in row for e in list(CAT_Q[c][0]) + R.summary_evals(c)], title="Figure 5"))
print(PROV())
R.show(R.pool_seeds(cm[cm.category.isin([c for row in FIG5 for c in row])]).drop(columns="model_id").round(2))
R.show(cm[cm.category.isin([c for row in FIG5 for c in row])].drop(columns="model_id").round(2))
R.show(d[d.eval_id.isin([e for row in FIG5 for c in row for e in list(R.CATEGORIES[c][0]) + R.summary_evals(c)])][DCOLS].sort_values(["eval_id", "object_label", "seed"]).round(1))
_sp = R.petri_decode_split()
print("Petri target decode: " + _sp["caption"] + " (" + _sp["will_not_be_closed"] + "; read from " + _sp["read_from"] + ")")
_esc = d[d.escalation_state.fillna("").ne("")] if "escalation_state" in d.columns else d.iloc[0:0]
print(f"rows whose token-cap escalation ladder is not final ({len(_esc)}; a BASE-CAP reading, never pooled with escalated seeds as seed variance without the mark):")
R.show(_esc[["eval_id", "object_label", "seed", "escalation_state", "y", "d_y"]].sort_values(["eval_id", "object_label"]).round(1) if len(_esc) else _esc)
_mc = R.pool_seeds(cm)
_mc = _mc[_mc.mixed_cap_seeds] if "mixed_cap_seeds" in _mc.columns else _mc.iloc[0:0]
print(f"pooled means whose seeds are NOT on one generation cap ({len(_mc)}; marked on the figure):")
R.show(_mc[["category", "object_label", "n_seeds", "seed_caps", "y", "spread"]].round(2) if len(_mc) else _mc)
_xc = d[d.cap_mismatch] if "cap_mismatch" in d.columns else d.iloc[0:0]
print(f"rows differenced against an anchor at ANOTHER generation cap ({len(_xc)}; marked \u2260cap, never dropped):")
R.show(_xc[["eval_id", "object_label", "seed", "anchor_cap", "d_y", "route"]].sort_values(["eval_id", "object_label"]).round(1) if len(_xc) else _xc)
print("rows with no unsteered run to compare with (drawn as a bar, not dropped):")
R.show(d[d.route == "no unsteered run"][["eval_id", "object_label", "seed", "lane_source_legacy", "y", "n"]])
""")

md("a3e91d55", r"""
## Figure 6 — capabilities: what the steering costs

The cost side of the ledger, the two categories that measure what an intervention takes away rather than what it prevents:
**over-refusal**, the change in benign compliance on XSTest and OR-Bench-Hard, and
**capability**, the change in accuracy or pass rate on MMLU, GSM8K, BigCodeBench-200 and BFCL-200. Positive is better on both axes,
so a bar below zero is a cost.

BigCodeBench and BFCL share this axis with MMLU and GSM8K.
**This redefines the capability number**: it was the MMLU-and-GSM8K mean and is now the mean over four evals, so a capability point
here is not comparable with one quoted from an earlier render. All four share a unit and a sign (percent correct or passing, higher
better) and all four score a capped item as a FAILURE, so the value already absorbs truncation. IFEval and MATH-500 are not on the axis
(the coverage table lists them); MATH-500 in particular leaves capped items UNSCORED, so its value can sit up to 55 points above its attempted
bound and would put two meanings of the same number side by side. Same
construction as the figures above: each fitted object is its three batch-64 seeds, each point is differenced against its eval's common
anchor, and the mean column carries the between-seed spread in its interval. Qwen's MMLU is the full 1,531-item pinned pool; the
200-item head it supersedes is nested inside it, so the two are never pooled and never differenced against each other.
""")

code("f7a1c206", r"""
R.delta_figure(cm, d, FIG6, R.OBJECTS, gaps_q, "fig6_capabilities", categories=CAT_Q); plt.show()
# ONE TABLE PER CATEGORY: over-refusal and capability are tabulated separately, so neither has to
# carry the other's columns and each fits the text width. The FIGURE keeps both panels side by side.
OR_ONLY, CAP_ONLY = [["over-refusal"]], [["capability"]]
print(R.table_evals_seeds(d, OR_ONLY, R.OBJECTS,
      "Over-refusal for Qwen3.5-4B.", "tab:qwen-over-refusal-seeds"))
print(R.table_evals_seeds(d, CAP_ONLY, R.OBJECTS,
      "Capability for Qwen3.5-4B.", "tab:qwen-capability-seeds"))
print(R.decode_only_disclosure(DEC_CELLS, DEC_ASIDE, evals=[e for row in FIG6 for c in row for e in CAT_Q[c][0]], title="Figure 6"))
""")

md("11a8433d", r"""
## Figure 7 — the frontier, relative to the unsteered model (Qwen)

The composition of the paper's frontier figure (`figQ_frontier_absolute_origin`): the frontier **reads the dose ladders** — every rung of the strength-ladder
sources (`R.LADDER_SOURCES`), whatever its selection state — and never a campaign pass, so a campaign point is never mistaken for a dose rung; the prompted arms are single diamonds from
the campaign rows; the unsteered reference is the campaign's run on the panel's lane, one per eval, and every point is translated by it eval
by eval, so the unsteered model sits at the origin and both axes read in pp. Rungs of one object are joined in dose order **through the
origin** (the unsteered model is every ladder's dose-0 rung: the common reference), the marker grows with the dose, a negative dose is an open
marker; no dose is written on the points. **Three dose units, never pooled**: the Refusal steer's dose is a raw
coefficient, the paired ReLU's a direction-norm multiplier, the LoRA's a write-basis multiplier — the marker size is each object's own nominal
scale and only a cost-versus-defense position is comparable across objects. **The same cut first**: a rung's delta is against an unsteered run
on its own item set (its own standup's where it has one), else the campaign's, ringed as cross-cut; the origin is each ladder's own unsteered.
**Degeneracy is not defense**: the degenerate share and the at-cap share are computed for every rung and reported in the tables below, and NEITHER is drawn beside a point. The coverage line under the figure counts how many drawn rungs carry a share at all (the vendor-preset decode leaves ~35-40 % of XSTest replies at the 512-token cap, which is why every rung's delta is taken against its own standup's unsteered: the two decodes are two regimes of the three-term over-refusal rate, not two levels of refusing). **No cost axis is synthesised**: a ladder whose
benign-compliance rows are withdrawn (the paired ReLU's XSTest anchor is unsourced) stays off the frontier; its multi-turn
arms are sound and are drawn in the companion below, dose against suppression, one panel per object on its own dose axis. **Roster** (`R.FRONTIER_OBJECTS`): the Refusal steer, the difference LoRA and the paired difference
ReLU steer as ladders, the two prompted SIs as points; the control and constitution objects are not on it.

**Lane.** As in the paper, the Qwen frontier is drawn on the **base→instruct lane**, where the LoRA ladders and the multi-turn difference
sweep are recorded. The Refusal steer is always instruct-direct (no base fit exists), and the Safety SI's campaign rows sit on that lane too;
both are drawn from it, named below, not marked on the canvas. Flip `FRONTIER_LANE` below to draw the paper lane instead.

**Two batteries, two panels.** Left, the static battery that `figQ_frontier_absolute_origin` draws (Past tense, CodeAttack, ReNeLLM,
Pliny). Right, **the paper's Figure 7**: y over the two multi-turn attacks (Crescendo, ActorAttack). A rung is drawn only when it carries every
eval of the panel's y battery; the x battery is the benign evals every drawn rung carries (the ladders carry XSTest alone) and the axis says
so. Held-out rungs, rungs whose multi-turn runs are not yet judged, and any ladder-shaped source the list does not name are printed below.
""")

code("b24c9738", r"""
FRONTIER_LANE, PULL = R.FRONTIER_LANE, R.FRONTIER_PULL_LANE
# THE UNSCALED-RATE GUARD RUNS ON THIS FRAME TOO. It already existed and already knew how to catch
# exactly this defect -- a headline metric whose values look like proportions but carries no SCALE entry, so it
# plots 100x small -- but it was only ever called on `op`, the operating-point frame. The over-refusal x-axis lives
# HERE, on the frontier frame, which is why the registry's flip to xst_refused / orb_refused drew benign compliance
# in fractions and nobody's check fired. A guard pointed at one of two frames is a guard for one of two frames.
_ur_fr = R.unscaled_rates(frontier)
if len(_ur_fr):
    print("!! UNSCALED HEADLINE METRIC ON THE FRONTIER FRAME — these plot 100x too small until they enter SCALE and "
          "IS_RATE in registry_nb.py. Every delta and axis reading below that uses them is wrong:")
    R.show(_ur_fr)
else:
    print("headline metrics on the frontier frame that look like proportions but carry no plotted-unit scale: none (correct).")

# X IS THE OVER-REFUSAL LABEL MEAN (superseding the refused rate, which had
# itself replaced the graded score): the benign axis is OR_mean · label_mean -- the mean over XSTest and OR-Bench-Hard of
# (unsteered − arm) `over_refusal_label_mean`, in pp. The frame was repointed at LOAD, together with the paper frame and the
# anchor pool (the OVER-REFUSAL METRIC line under the load cell), so the frontier cannot end up on a different over-refusal
# number from the Figure 6 panel; this cell only states what it reads and shouts if the load-time switch was withheld.
_fr_main = frontier
_xm = sorted(set(_fr_main[_fr_main.eval_id.isin(R.FRONTIER_X[1]) & _fr_main.headline & R.selected_scope(_fr_main)].metric_id.dropna()))   # the rows the frontier reads (an unscoped read also listed the unread r2_export rows' refused ids beside the label mean)
print(f"FRONTIER X-AXIS METRIC: {', '.join(_xm)} — "
      + ("the over-refusal LABEL MEAN (full refusal 1 / partial 0.5 / compliance 0; an at-cap reply by its label), NOT the refused "
         "rate and NOT over_refusal_graded; " if OR_METRIC_OK else
         "!! the registry's own headline (the refused rate): the label-mean switch was WITHHELD at load for incomplete coverage, "
         "see the OVER-REFUSAL METRIC line under the load cell; ")
      + "x = mean over XSTest and OR-Bench-Hard of (unsteered − arm), in pp. Every frontier figure below reads this frame.")
pts_mt, held_mt, meta_mt = R.frontier_points(_fr_main, FRONTIER_LANE, R.FRONTIER_Y[1], pull_lane=PULL, objects=R.FRONTIER_OBJECTS, panel_name="multi-turn")
pts_st, held_st, meta_st = R.frontier_points(_fr_main, FRONTIER_LANE, R.FRONTIER_STATIC[1], pull_lane=PULL, objects=R.FRONTIER_OBJECTS, panel_name="static")
# THE SI DIAMONDS ARE THE RUNS OF THEIR ONE DOT (R.select_decode_only), the runs Figures 3-6 average: every fallback, cross-setup match
# and set-aside reading is listed. The Refusal steer is its dose ladder, one point per dose (R.frontier_rung_line says what each averages).
for _t, _m, _p in (("static", meta_st, pts_st), ("multi-turn", meta_mt, pts_mt)):
    print(R.decode_only_disclosure(_m.get("decode_only_cells"), _m.get("decode_only_set_aside"), title=f"Figure 7, {_t} panel (the SI diamonds)"))
    print(R.frontier_rung_line(_p, "arditi_allpos", _t))
# AN ANCHOR IS ON ITS ARM'S METRIC: every frontier anchor pool is matched to the rung's metric family, so a rung whose
# eval has no unsteered run on that metric is NOT drawn against one on another metric; it is listed here and missing from the panel.
_noanc = sorted(set(meta_mt.get("no_anchor", [])) | set(meta_st.get("no_anchor", [])))
if _noanc:
    print("!! evals whose rungs found NO unsteered anchor on their own metric (those rungs are not drawn): " + ", ".join(_noanc))
# THE STATIC ARDITI LADDER IS NOT INSTRUMENT-MATCHED (measured 2026-09-23; not re-run: the three rungs are plotted
# together and the mismatch is stated). The load-bearing difference is NOT the cap, which was first reported
# and is easy to assume: a ladder_final row that settles at 4,096 is fully comparable to one that settles at 16,384 --
# it laddered and exited at its own rung, which is the completion-cap merge working. difference_lora spans all three
# rungs and is fine. The defect is read_at: Arditi c = 0.3 and c = 0.5 are FIXED [4096] and could never escalate, so a
# truncated item stays truncated. They are the only in-paper static rows on a fixed ladder -- 8 rows / 2 objects against
# 742 that laddered. The ladder contract is not on the paper view, so the trigger here is the observable proxy, more
# than one INSTRUMENT among the drawn Arditi rungs; it retires itself if those rungs are ever re-read on one instrument.
_ar = frontier[frontier.paper_label.eq("arditi_allpos") & frontier.eval_id.isin(R.FRONTIER_STATIC[1])
               & frontier.selection_state.eq("selected")] if len(frontier) else frontier
if len(_ar) and "instrument_id" in _ar.columns and _ar.instrument_id.nunique() > 1:
    print("CAVEAT — the static Refusal steer ladder is not instrument-matched. c = 0.3 and c = 0.5 were read on a FIXED "
          "4,096-token ladder with one trial and could not escalate on truncation, whereas c = 0.15 (and every other "
          "static row in the paper) was read at ladder_final over [4096, 8192, 16384] with two trials. Truncation is "
          "censored at the two older rungs and resolved everywhere else, so the shape of the curve across those points "
          "is partly an instrument artefact. They are plotted together rather than re-run.")
# THE FRONTIER, UNAGGREGATED: every (arm, seed, dose) with its score on each eval and the three
# means the panels actually plot, so an odd curve can be read off the numbers rather than inferred from its shape.
# the prompted conditions' rows are the runs their diamonds average (R.frontier_decode_runs of the two panels' meta)
_dr = R.frontier_decode_runs(meta_st, meta_mt)
_rt = R.rung_eval_table(frontier, R.FRONTIER_OBJECTS, decode_runs=_dr)
if len(_rt):
    print("Every drawn rung, per evaluation. The panels plot ST mean or MT mean (y) against OR mean (x), so these are "
          "the same numbers the curves are made of. Deltas vs each eval's anchor, pp; positive is better defense on "
          "ST/MT and better benign compliance on OR.")
    R.show(_rt, n=len(_rt))
# AT-CAP AND DEGENERACY, IN THEIR OWN TABLES. Same grid as the score table, read straight off
# columns already on the frame so nothing is recomputed. Separate on purpose: these shares are reported throughout
# and never drawn, and a quality number in the same cell as a score invites reading one as the other.
# Blank means NEVER MEASURED, which is not zero — an unmeasured rung and a clean one must not look alike.
for _col, _what in (("frac_at_token_cap", "% of items AT THE TOKEN CAP"),
                    ("frac_degenerate_union5", "% of responses DEGENERATE")):
    _q = R.rung_quality_table(frontier, _col, R.FRONTIER_OBJECTS, decode_runs=_dr)
    if len(_q):
        print(f"{_what}, per drawn rung. Blank = never measured, which is not the same as zero.")
        R.show(_q, n=len(_q))
    else:
        print(f"{_what}: not measured on any drawn rung in this frame.")
_cens = list(meta_mt.get("dropped_at_cap", [])) + list(meta_st.get("dropped_at_cap", []))
if _cens:
    print("rungs NOT plotted because their DEFENSE reading is almost entirely truncated (>90% of attack items at the token cap, so "
          "the model was cut off rather than complying or refusing): " + "; ".join(sorted(set(_cens))))
# THE PER-SEED CENSOR DECLARES ITSELF. Dropping a seed before pooling is a CONSUMER rule,
# not a registry one: the rows are in the frame, selected and in the paper, and the producers' convention deliberately
# keeps them — every over-refusal arm climbs a 512 -> 2,048 -> 8,192 ladder and files at its EXIT rung, so a top-rung
# cell still truncating is an arm that never stopped looping, which stays 'exited' in the frame. The
# label-mean headline scores an at-cap benign reply BY ITS LABEL (a cut-off refusal is a refusal, a cut-off compliance is
# compliance), so the BENIGN axis reads those replies as what they were.
# What it cannot account for is the DEFENSE axis, where a cut-off reply completes no attack and so flatters the
# steering. That asymmetry is the whole reason this censor exists, and a reader must be able to see it operate rather
# than infer it from a missing point. Taking these rows OUT of the frame would take a registry hold_out record;
# this only declines to draw them.
_seedcens = sorted(set(list(meta_mt.get("dropped_seed_at_cap", [])) + list(meta_st.get("dropped_seed_at_cap", []))))
if _seedcens:
    print("SEEDS dropped from their rung's pooled point before averaging, for the same truncation reason ON THE DEFENSE "
          "EVALS (the benign evals' at-cap replies are scored by their label under the label-mean headline) (the rung "
          "still draws from its remaining seeds; a rung all of whose seeds are dropped disappears via the line above). "
          "These rows ARE in the frame and the producers keep them deliberately — the exit-rung convention files an "
          "arm at the cap it stopped escalating from — so this is the figure declining to average a cut-off reading "
          "into a defense number, not a claim the row is wrong: " + "; ".join(_seedcens))
_negdrop = int(meta_mt.get("dropped_negative", 0)) + int(meta_st.get("dropped_negative", 0))
if _negdrop:
    print(f"negative steering strengths (sign controls: they test that the steer's direction matters, a different question) excluded from the frontier panels: {_negdrop} rung(s). "
          f"Each ladder still passes through the origin, its own dose 0; the rungs keep their table entries.")
for _pn, _pm in (("multi-turn", meta_mt), ("static", meta_st)):
    if _pm.get("hidden_rungs"): print(f"rungs left off the {_pn} panel (R.FRONTIER_RUNG_HIDDEN: a ladder's far end, degenerate or well past its curve's elbow; kept in the tables and on the dose-response):", _pm["hidden_rungs"])
pending_mt = R.frontier_pending(frontier, FRONTIER_LANE, R.FRONTIER_Y[1], pull_lane=PULL)
for _nm, _mt in (("static", meta_st), ("multi-turn", meta_mt)):
    _um = _mt.get("degen_unmeasured") or []
    _m, _t2 = _mt.get("degen_measured", 0), _mt.get("degen_total", 0)
    _gap = ("" if _m >= _t2 else
            " UNMEASURED IS NOT CLEAN - the rest carry no degeneracy column at all, which is not the same as a share of zero.")
    print(f"{_nm} panel: a degeneracy share is filed for {_m} of {_t2} drawn rungs.{_gap} The shares are reported in the "
          f"tables and never drawn beside a point, and this is a count of rows that HAVE a share, not "
          f"of one measurement: the filed shares come from at least three different classifiers (the over-refusal and static "
          f"ladders' item-filtered rule, the misalign redraw's zlib ratio under 0.2, and the registry's own zlib collapse "
          f"under 0.15), so they are not one instrument and should not be compared as one:")
    R.show(pd.DataFrame(_um).round(2) if _um else pd.DataFrame(columns=["object_label", "dose", "dy"]))
R.frontier_figure([dict(pts=pts_st, title="static", y_evals=R.FRONTIER_STATIC[1], x_battery=meta_st["x_battery"], per_seed=meta_st.get("per_seed"), degen_measured=meta_st.get("degen_measured"), degen_total=meta_st.get("degen_total")),
                   dict(pts=pts_mt, title="multi-turn", y_evals=R.FRONTIER_Y[1], x_battery=meta_mt["x_battery"], per_seed=meta_mt.get("per_seed"), degen_measured=meta_mt.get("degen_measured"), degen_total=meta_mt.get("degen_total"),
                        note=f"{len(pending_mt)} ladder rungs await judgment" if len(pending_mt) else None)], R.FRONTIER_OBJECTS, "fig7_frontier"); plt.show()


print(R.table_frontier([dict(title="static", pts=pts_st, per_seed=meta_st.get("per_seed")), dict(title="multi-turn", pts=pts_mt, per_seed=meta_mt.get("per_seed"))],
      R.FRONTIER_OBJECTS,
      "{panel} rungs drawn on the Qwen3.5-4B robustness frontier.", "tab:qwen-frontier"))   # one table per panel: -static / -multi-turn
print(f"{R.MODEL_LABEL[R.MODEL_ID]} · {R.LANE_LABEL[FRONTIER_LANE]} lane · ladder sources {R.LADDER_SOURCES} · registry head {HEAD}")
print(f"drawn from the {R.LANE_LABEL[PULL]} lane (the Refusal steer is always instruct-direct): {sorted(set(pts_st[pts_st.pulled].object_label) | set(pts_mt[pts_mt.pulled].object_label))}")
print(f"frontier roster {R.FRONTIER_OBJECTS} · ladders on the frame but not on it (not drawn): {sorted((set(pts_st.object_label) | set(pts_mt.object_label)) - set(R.FRONTIER_OBJECTS))}"
      f" · roster objects with no judged rung yet: {[o for o in R.FRONTIER_OBJECTS if o not in set(pts_st.object_label) | set(pts_mt.object_label)]}")

fr_runs = runs_all[(runs_all.model_id == R.MODEL_ID) & runs_all.object_label.isin(R.FRONTIER_OBJECTS) & runs_all.eval_id.isin(R.FRONTIER_X[1] + R.FRONTIER_Y[1] + R.FRONTIER_STATIC[1])]
print("runs of the roster objects on the frontier's evals, by selection state (a `withdrawn` or `excluded` run is out of the paper view and cannot reach a panel):")
R.show(fr_runs.groupby(["object_label", "selection_state"]).eval_id.agg(lambda s: ", ".join(sorted(set(s)))).reset_index())
""")

md("7d2e9c40", r"""
## Appendix — all four models

The same panels across **Gemma-4-E4B, Qwen3.5-4B, Nemotron-3-Nano-4B and Llama-3.1-8B**, each on its paper lane, in the main figures'
form: x = the objects as lanes, the models dodged inside each lane by marker shape (square, circle, triangle, inverted triangle); per (object,
model) the category mean is the large mark with its interval and the individual evals are small faded marks of the same shape beside it. The
roster is the main figures' order without the Refusal steer (Qwen-only): the two prompted SIs, then each corpus's ReLU steer beside its
LoRA. **Layout**: the figures whose x is the object stack their category panels vertically,
one per row; the figures in the main figures' form, x = the eval (`figA_*_absolute_models`), keep one row and each has a **relative
companion** (`figA_*_delta_models`): the same panels with every cell as its change against its own model's unsteered run - the anchored
deltas the x = object figures and the tables read - oriented so up is better, each panel ranged on its own axis. **One interval rule for a
mean over evaluations**: every `mean` column here, absolute or relative, and every category mean in Figures 3–5 and the
x = object figures, is the unweighted mean of the per-evaluation values; for one fit se = sqrt(Σ se²) / k — the per-evaluation intervals
combined, no term for the spread between evaluations (the battery is fixed, not sampled); for several fits one pool over the panel's evaluations,
the fits' runs pooled item by item per evaluation, the variance between the fits' means over the panel added and the shared anchor counted once
(the same fits on every evaluation). Readings of ONE cell pool before that: its decode draws item-clustered (they re-decode the same items), its
fit seeds item-clustered across each other with the between-fit variance, the shared anchor counted once. The black bar on each large diamond is a
different thing again: ± SEM over the models present — a level summary of the four models, never a contrast interval. **Every table** here leads
with each model's unsteered absolute reading (a column per model in the category tables, a row per model in the absolute ones), as the
level of the quantity its figure plots (defense success, benign compliance, accuracy, alignment score), averaged over
seeds: for Qwen the mean over its fit seeds of the unsteered reading the most arms of each seed were differenced against, for the
other three models (no seed replicates) the unsteered reading the most of their arms were differenced against (ties broken as the
notebook README states). Every other arm is a signed change
against its own unsteered run, so first row + cell approximates the arm's value after the intervention, exactly where that run equals
the first row. Deltas are against each model's own unsteered run on its lane, with the same routes as Figures 3–4 (Qwen's multi-turn anchor
outside the frame drawn open). **The fitted arms now draw on all four models**: the LoRAs and the paired-write difference ReLU
were being deleted from these panels by a reader defect, not missing from the registry - the batch-8 retirement was unscoped, and since
only Qwen has batch-64 seeds, on the other three models the canonical fit is the object's ONLY fit, so the rule removed every non-Qwen
LoRA and ReLU row and the cells then drew the `not in roster` glyph over rows the registry holds `selected`. The retirement is now scoped
to the (model, object) pairs that actually carry a seed replicate. What still reads `not in roster` is genuine and each case is filed:
`control_relu` / `constitution_relu` on the three non-Qwen models (no roster entry), BigCodeBench and Nemotron's MATH-500 (the roster's
instrument selector names a different harness from the one the runs used), and the best-of-N fitted rows whose operating-point flag is
null. Those light up at their fold with no further change here.
""")

md("a8e3d1c2", r"""
### Appendix — did the ReLU steers find the refusal direction? Layer-12 cosines against the Arditi direction

For the Qwen3.5-4B ReLU steers, cosine similarities at decoder block 12 — the layer the Arditi refusal direction was extracted at, and
one of the 32 the steers are fit on. Each steer has a **read direction** $w_\ell$ (the gate probe whose ReLU output scales the write;
circles) and a **write direction** $v_\ell$ (the vector added to the residual stream; squares). Left to right: each control seed and each
constitution seed against the **refusal direction** (the direction toward refusal); every control seed against every constitution seed;
and each object's three fit seeds against one another. The grey band is ±2 sd of the cosine of two random directions in d = 2,560. Only
the seeded base→instruct fits are drawn; the two instruct-direct single fits (constitution and safety ReLU at batch 64) are in the table
only, and the paired-write difference ReLU is a composition with no single direction and is not a row. The cosines are computed once from
the object files (`relu_arditi_cosines.py`, an interpreter with torch) into a sidecar the notebook binds to the registry by object id —
every row below is a registry object under the label, seed and checkpoint it claims, or it is drawn open and named.
""")

code("d4f7b2a9", r"""
_cs = R.load_cosine_sidecar()
_cos_objects = R.query(R.OBJECTS_SQL, "objects")                # the object table, for the id binding (the transfer cell below reads it again)
_cs_check = R.cosine_sidecar_check(_cs, _cos_objects)
_bad = _cs_check[~_cs_check.known]
if len(_bad):
    print("!! sidecar rows the registry does NOT know under the label / seed / checkpoint claimed (drawn open): "
          + "; ".join(f"{r.object_label} {r.seed or '-'} {r.object_id[:8]}" for r in _bad.itertuples()))
else:
    print(f"cosine sidecar bound to the registry: {int(_cs_check.known.sum())} of {len(_cs_check)} object ids known under their claimed label, seed "
          f"and checkpoint (reference {_cs['reference']['object_id'][:8]}, the Arditi c = 0.15 rung) · generated {_cs['generated_utc']} at "
          f"{(_cs.get('git_commit') or '?')[:9]} · layer {_cs['layer']} · d = {_cs['d']:,}")
R.cosine_figure(_cs, _cs_check, "figA_relu_arditi_cosines"); plt.show()
_ref, _pairs = R.cosine_frames(_cs)
_cross = pd.DataFrame(_cs.get("cross_pairs", []))
print("cosine against the refusal direction — largest |cos| over every seed and kind: "
      f"{_ref.cos.abs().max():.3f} ({_ref.cos.abs().max() / _cs['random_sd']:.1f} sd of a random direction); seed-vs-seed cosines within an object "
      f"{_pairs.cos.min():.3f}–{_pairs.cos.max():.3f}; control seeds vs constitution seeds "
      + (f"read {_cross['read'].min():.3f}–{_cross['read'].max():.3f}, write {_cross['write'].min():.3f}–{_cross['write'].max():.3f}." if len(_cross) else "(not in the sidecar)."))
print(R.table_cosines(_cs, "Layer-12 cosine similarities: the Qwen3.5-4B ReLU steers against the Arditi refusal direction and against their own seeds.",
                      "tab:qwen-relu-arditi-cosines"))
R.show(_ref.assign(cos=_ref.cos.round(4)).pivot_table(index=["object_label", "lane", "seed"], columns="kind", values="cos"))
R.show(_pairs.assign(cos=_pairs.cos.round(4)).pivot_table(index=["object_label", "pair"], columns="kind", values="cos"))
if len(_cross):
    R.show(_cross.assign(read=_cross["read"].round(4), write=_cross["write"].round(4)))
""")

code("3c9f7a51", r"""
d_all = R.anchored(op_all, pool=POOL_Q, setup="any")       # every model's deltas on its paper lane (the pool only serves Qwen: it is Qwen's frame)
cm_all = R.category_means(d_all, categories=R.APPENDIX_CATEGORIES)     # the appendix's capability axis is MMLU + GSM8K
# THE DECODE-ONLY CONDITIONS ARE ONE DOT PER CELL ON EVERY MODEL (R.select_decode_only, setup "any": in the appendix a run from any
# serving setup counts as matched, and a cell with no matched run plots its closest readings); each figure below lists its fallback
# dots, cross-setup matches and set-aside readings
DEC_CELLS_ALL, DEC_ASIDE_ALL = d_all.attrs["decode_only_cells"], d_all.attrs["decode_only_set_aside"]
print("decode-only dots in the appendix figures, one per condition per eval and model, by the number of runs each averages:", R.decode_only_mark_line(d_all))
gaps_m = R.gap_status(runs_lane, R.drawn_set(d_all[d_all.d_y.notna()]), R.APPENDIX_OBJECTS, ALL_EVALS, models=R.MODEL_ORDER, wrong_judge=WJ)
JB_ALL = [["static", "multi-query", "multi-turn"]]           # the appendix keeps the three jailbreak families together
R.category_panels(cm_all, d_all, JB_ALL, R.APPENDIX_OBJECTS, R.MODEL_ORDER, gaps_m, "figA_jailbreaks_models", categories=R.APPENDIX_CATEGORIES); plt.show()
print(R.decode_only_disclosure(DEC_CELLS_ALL, DEC_ASIDE_ALL, evals=R.category_evals(JB_ALL, R.APPENDIX_CATEGORIES), objects=R.APPENDIX_OBJECTS,
                               title="the all-model jailbreak figure (figA_jailbreaks_models)"))
_mi_all = R.missing_intervals(d_all)
if len(_mi_all):
    print(f"UNMEASURED intervals across the four models: {int(_mi_all.rows.sum())} drawn rows carry a value but no interval (filed contrast not current; re-anchoring in progress) — "
          + "; ".join(f"{R.MODEL_LABEL.get(m, m)}: {int(g.rows.sum())} on {', '.join(sorted(set(g.eval_id)))}" for m, g in _mi_all.groupby('model_id')))
for _cat in [c for _row in JB_ALL for c in _row]:
    print(R.table_categories_models(cm_all, _cat, R.APPENDIX_OBJECTS, R.MODEL_ORDER,
          f"{_cat.capitalize()} defense across the four models.", f"tab:allmodels-{_cat.replace(' ', '-')}", d=d_all, categories=R.APPENDIX_CATEGORIES))
print(f"four models on their paper lanes · operating dose · registry head {HEAD}" + "".join(f" · {R.MODEL_LABEL[m]} / {R.LABEL.get(o, o)} from the {R.LANE_LABEL.get(l, l)} lane" for m, o, l in PULLED))
print("delta route per row and model:"); R.show(d_all.groupby(["model_id", "route"]).size().unstack("route").fillna(0).astype(int))
ceil = d_all[d_all.ceiling.astype(bool)]
print(f"ceiling-forced cells ({len(ceil)}): the unsteered run leaves under {R.HEADROOM_PP:g} pp of room, so the delta is bounded by the anchor (marked ≤; never pooled with the statics - the categories keep multi-turn apart):")
R.show(ceil[["model_id", "eval_id", "object_label", "seed", "anchor_value", "headroom", "d_y", "route"]].sort_values(["model_id", "eval_id", "object_label"]).round(1))
cap = d_all[d_all.frac_at_token_cap.astype(float) > R.AT_CAP_LINE]
print(f"cells with more than {R.AT_CAP_LINE:.0%} of responses at the token cap ({len(cap)}; reported beside the value as 'cap n%', never discounted):")
R.show(cap[["model_id", "eval_id", "object_label", "seed", "frac_at_token_cap", "y", "d_y"]].sort_values("frac_at_token_cap", ascending=False).round(2))
deg = d_all[d_all.frac_degenerate_union5.astype(float) > R.DEGEN_LINE]
print(f"cells with more than {R.DEGEN_LINE:.0%} of responses judged degenerate ({len(deg)}; scored in place, listed here and never drawn beside the value):")
R.show(deg[["model_id", "eval_id", "object_label", "seed", "frac_degenerate_union5", "frac_degenerate_classifier", "y", "d_y"]].sort_values("frac_degenerate_union5", ascending=False).round(2))
held_m = gaps_m[gaps_m.status == "excluded by ruling"]
print(f"cells held out by a decisive ruling across models ({len(held_m)}):"); R.show(held_m[["model_id", "object_label", "eval_id", "ruling"]])
R.show(R.pool_seeds(cm_all[cm_all.category.isin([c for _row in JB_ALL for c in _row]) & cm_all.object_label.isin(R.APPENDIX_OBJECTS)])
        .pivot_table(index=["category", "object_label"], columns="model_id", values="y").round(1))
""")

code("8e1b4d27", r"""
R.category_panels(cm_all, d_all, FIG5, R.APPENDIX_OBJECTS, R.MODEL_ORDER, gaps_m, "figA_misalignment_models", categories=R.APPENDIX_CATEGORIES); plt.show()
print(R.decode_only_disclosure(DEC_CELLS_ALL, DEC_ASIDE_ALL, evals=R.category_evals(FIG5, R.APPENDIX_CATEGORIES), objects=R.APPENDIX_OBJECTS,
                               title="the all-model misalignment figure (figA_misalignment_models)"))
R.category_panels(cm_all, d_all, FIG6, R.APPENDIX_OBJECTS, R.MODEL_ORDER, gaps_m, "figA_capabilities_models", categories=R.APPENDIX_CATEGORIES); plt.show()
print(R.decode_only_disclosure(DEC_CELLS_ALL, DEC_ASIDE_ALL, evals=R.category_evals(FIG6, R.APPENDIX_CATEGORIES), objects=R.APPENDIX_OBJECTS,
                               title="the all-model capabilities figure (figA_capabilities_models)"))
for _cat in [c for _row in FIG6 for c in _row]:
    print(R.table_categories_models(cm_all, _cat, R.APPENDIX_OBJECTS, R.MODEL_ORDER,
          f"{_cat.capitalize()} across the four models.", f"tab:allmodels-{_cat.replace(' ', '-')}", d=d_all, categories=R.APPENDIX_CATEGORIES))
for _cat in [c for _row in FIG5 for c in _row]:
    print(R.table_categories_models(cm_all, _cat, R.APPENDIX_OBJECTS, R.MODEL_ORDER,
          f"{_cat.capitalize()} across the four models.", f"tab:allmodels-{_cat.replace(' ', '-')}", d=d_all, categories=R.APPENDIX_CATEGORIES))
print(f"four models on their paper lanes · operating dose · registry head {HEAD}")
R.show(R.pool_seeds(cm_all[cm_all.category.isin([c for row in FIG5 for c in row]) & cm_all.object_label.isin(R.APPENDIX_OBJECTS)])
        .pivot_table(index=["category", "object_label"], columns="model_id", values="y").round(1))
""")

code("5a7c2e93", r"""
ATTACKS, OR_EVALS, CAP = PANEL_EVALS["attacks"], PANEL_EVALS["over-refusal"], PANEL_EVALS["capability"]
AB_OBJECTS = ["none"] + R.APPENDIX_OBJECTS
_hp_ab = R.headline_points(op_all, ATTACKS, setup="any")      # the decode-only conditions' runs of their one dot, any serving setup
pts_ab = R.flip(_hp_ab)
gaps_ab = R.gap_status(runs_lane, R.drawn_set(pts_ab), AB_OBJECTS, ATTACKS, models=R.MODEL_ORDER, wrong_judge=WJ)
R.absolute_panels(pts_ab, ATTACKS, AB_OBJECTS, R.MODEL_ORDER, gaps_ab, "defense success rate (%; 100 − ASR at threshold 0.5)", "figA_attacks_absolute_models"); plt.show()
# THE RELATIVE COMPANION (the appendix figures in the main figures' form keep one row and also plot relative to
# unsteered): the same panels, every cell its change against its own model's unsteered run - the anchored deltas the x = object figure
# above and the tables read - oriented so up is better defense; a cell with nothing to difference against draws the `no unsteered run` glyph
gaps_abd = R.gap_status(runs_lane, R.drawn_set(R.delta_points(d_all, ATTACKS, R.APPENDIX_OBJECTS)), R.APPENDIX_OBJECTS, ATTACKS, models=R.MODEL_ORDER, wrong_judge=WJ)
R.delta_panels(d_all, ATTACKS, R.APPENDIX_OBJECTS, R.MODEL_ORDER, gaps_abd, "figA_attacks_delta_models"); plt.show()
print(R.decode_only_disclosure_pair(_hp_ab, d_all, ATTACKS, R.APPENDIX_OBJECTS, "figA_attacks_absolute_models", "figA_attacks_delta_models"))
# the table in the form every table shares: the unsteered model absolute, every arm +/- against it
# ONE ABSOLUTE TABLE PER FAMILY, matching the separate over-refusal and capability ones.
for _cat, (_name, _slug) in {"static": ("Static", "single-turn"),
                             "multi-query": ("Multi-query", "multi-query"),
                             "multi-turn": ("Multi-turn", "multi-turn")}.items():
    _evs = [e for e in R.CATEGORIES[_cat][0] if e in set(ATTACKS)]
    if not _evs:
        continue
    print(R.table_absolute(d_all, _evs, AB_OBJECTS, R.MODEL_ORDER,
          f"{_name} defense success rate across the four models.",
          f"tab:allmodels-absolute-{_slug}"))
print(f"four models on their paper lanes · operating dose · registry head {HEAD}")
R.show(pts_ab.pivot_table(index=["eval_id", "object_label", "seed"], columns="model_id", values="y").round(1))
""")

md("6d2f8b3a", r"""
### Appendix — best-of-N search curves

Defense success within the first N attempts, $100\,(1-\mathrm{ASR}@N)$, against N on a log axis, one panel per model, the same admitted
runs as the attacks panels above. Qwen's three fit seeds (and the three seed-view standups of its unsteered model) are the
thin faint lines, the thick marked line their mean; each prompted SI is one line, the mean of the runs its best-of-N dot above
averages (its matched runs, else its closest readings), and the other models have one run per arm. Every point is the
per-goal any-hit rate over the 100 confirmatory goals at threshold 0.5 under the canonical climbed instrument (the 512 store's at-cap
draws re-drawn to 4,096, the N = 75 point being the headline `asr_bon_cap4096`); Llama's paired difference ReLU is its native-4,096
pass. Curves come from the producers' `asr_at_n<N>` rows; a run that files only the `asr_at_n_4096_cf` grid on its headline row is read
from that grid and footnoted until its rows land. The composition table under the figure names every run drawn.
""")

code("b0a7c4e1", r"""
BON_OBJECTS = ["none"] + R.APPENDIX_OBJECTS
bon_all = R.bon_curve_points(op_all, paper_all, setup="any")   # the runs the attacks panels admit, with their per-N rows from the paper frame
bon_pts, bon_dropped = R.bon_curve_select(bon_all[bon_all.object_label.isin(BON_OBJECTS)])
R.bon_curves_figure(bon_pts, BON_OBJECTS, R.MODEL_ORDER, "figA_bon_curves_models"); plt.show()
print(f"four models on their paper lanes · operating dose · registry head {HEAD}")
print("every run drawn (curve = rows: asr_at_n<N> metric rows; extra: the bon_stores grid stopgap), with its decode policy:")
R.show(R.bon_curve_composition(bon_pts), n=80)
print(f"admitted runs NOT drawn ({len(bon_dropped)}), and why (headline only = another lane's anchor with no curve rows; other decode = a "
      f"grandfathered T = 1.0 climb beside vendor-preset standups of the same cell; duplicate filing = one store read by two readers):")
R.show(bon_dropped, n=40)
# THE DECODE-ONLY CONDITIONS ARE ONE CURVE EACH (R.select_decode_only), the runs of their best-of-N dot above that the curve rules keep
# (R.bon_curve_select recounts each cell from the curves drawn): every fallback, cross-setup match and set-aside reading
print(R.decode_only_disclosure(bon_pts.attrs.get("decode_only_cells"), bon_pts.attrs.get("decode_only_set_aside"), objects=BON_OBJECTS,
                               title="the best-of-N curves (figA_bon_curves_models)"))
# THE ADMISSION GAP: a cell drawn here from a grandfathered T = 1.0 campaign run while selected vendor-preset
# runs of the same cell sit in the paper frame unadmitted (Qwen's seedrep prompted SIs file asr_bon_cap4096 as their headline). Listed
# until the admission reads the headline family; then these runs enter op_all and the figure draws them as seeds + mean by itself.
bon_gap = R.bon_curve_unadmitted(op_all, paper_all, BON_OBJECTS, R.MODEL_ORDER)
print(f"vendor-preset runs of a drawn cell that the frame holds selected but op_all does not admit ({len(bon_gap)}; the cell is drawn from its T = 1.0 run until the admission is fixed):")
R.show(bon_gap, n=40)
""")

code("9b3d6f18", r"""
_hp_ac = R.headline_points(op_all, OR_EVALS, setup="any")
pts_ac = R.flip(_hp_ac)
gaps_ac = R.gap_status(runs_lane, R.drawn_set(pts_ac), AB_OBJECTS, OR_EVALS, models=R.MODEL_ORDER, wrong_judge=WJ)
CAP_A = R.APPENDIX_CAPABILITY_EVALS                            # MMLU, GSM8K, BigCodeBench, BFCL (Llama has no BFCL: out by design)
pts_ad = R.headline_points(op_all, CAP_A, setup="any")         # defined here, beside over-refusal, because they now share a figure
gaps_ad = R.gap_status(runs_lane, R.drawn_set(pts_ad), AB_OBJECTS, CAP_A, models=R.MODEL_ORDER, wrong_judge=WJ)
# over-refusal and capability share one figure, matching Figure 6, but NOT one y label: benign compliance and accuracy
# are percentages of different things, so each panel is labeled on its own axis (the dict form of `ylabel`).
R.absolute_panels(R.concat_plain(pts_ac, pts_ad), OR_EVALS + CAP_A, AB_OBJECTS, R.MODEL_ORDER,
                  pd.concat([gaps_ac, gaps_ad], ignore_index=True),
                  {"over-refusal": "benign compliance (%; 100 − over-refusal label mean)", "capability": "accuracy / pass rate (%)"},
                  "figA_capabilities_absolute_models"); plt.show()
# THE RELATIVE COMPANION: the same two panels as changes against each model's unsteered run, up = better
# (more benign compliance, higher accuracy / pass rate); the capability label names both since the coding evals joined the axis
_ev_cd = OR_EVALS + CAP_A
gaps_acd = R.gap_status(runs_lane, R.drawn_set(R.delta_points(d_all, _ev_cd, R.APPENDIX_OBJECTS)), R.APPENDIX_OBJECTS, _ev_cd, models=R.MODEL_ORDER, wrong_judge=WJ)
R.delta_panels(d_all, _ev_cd, R.APPENDIX_OBJECTS, R.MODEL_ORDER, gaps_acd, "figA_capabilities_delta_models",
               ylabel={"over-refusal": "Δ benign compliance (pp)", "capability": "Δ accuracy / pass rate (pp)"}); plt.show()
print(R.decode_only_disclosure_pair(R.concat_selected(_hp_ac, pts_ad), d_all, _ev_cd, R.APPENDIX_OBJECTS, "figA_capabilities_absolute_models",
                                    "figA_capabilities_delta_models"))
print(R.table_absolute(d_all, OR_EVALS, AB_OBJECTS, R.MODEL_ORDER,
      "Benign compliance across the four models.",
      "tab:allmodels-absolute-overrefusal"))
print(f"four models on their paper lanes · operating dose · registry head {HEAD}")
R.show(pts_ac.pivot_table(index=["eval_id", "object_label", "seed"], columns="model_id", values="y").round(1))
""")

code("c4e8a2d6", r"""
# (drawn above, in the combined capabilities figure)
print(R.table_absolute(d_all, CAP_A, AB_OBJECTS, R.MODEL_ORDER,
      "Capability across the four models.", "tab:allmodels-absolute-capability"))
print(f"four models on their paper lanes · operating dose · registry head {HEAD}")
print("capability runs per model and lifecycle (the registry holds the runs; few are scored into a metric row yet):")
R.show(runs_lane[runs_lane.eval_id.isin(CAP)].groupby(["model_id", "eval_id", "lifecycle_state"]).size().unstack("lifecycle_state").fillna(0).astype(int))
""")

code("e6f1b5a9", r"""
# THE PETRI ABSOLUTE TABLE, in the form every table shares: its unit is the 1--10 alignment score (11 - the misalignment score).
_PETRI_EV = [R.POOLED_EVAL["Petri"]] if "Petri" in R.POOLED_EVAL else R.CATEGORIES["Petri"][0]
print(R.table_absolute(d_all, _PETRI_EV, AB_OBJECTS, R.MODEL_ORDER,
      "Petri alignment score across the four models.",
      "tab:allmodels-absolute-petri"))
print(f"four models on their paper lanes · operating dose · registry head {HEAD}")
""")

md("4d9c2b7a", r"""
### Appendix — judge robustness: the same Qwen transcripts under the Claude judges and under the paper's judges

Every judged evaluation in the paper is graded by a non-Claude judge, MiniMax-M3 or, for Petri, GLM-5.3. This figure grades the same
Qwen3.5-4B transcripts by the Claude judges the registry filed first (Claude Haiku 4.5 for StrongREJECT, Claude Sonnet 5 for
over-refusal, Claude Opus 5 for Petri; circles) and by the paper's judges (diamonds). Each mark is a reading's change from the unsteered
model in percentage points (points for Petri), per fit seed, with its 95% interval, positive better; a line joins the two judges' marks of
one seed, and the bold column after the separator is the category mean. Part a draws the jailbreaks (static, multi-query, multi-turn),
part b over-refusal and Petri. The readings are frozen at registry head 537565, the head the paper's final figures were drawn at, and are
read from `data/appendix_judge_comparison.json` in every render, whatever its source, judging policy or StrongREJECT threshold. The
lines under the figure give the numbers the appendix quotes: the rank correlation of the interventions between the two sets of judges
in each model × evaluation-category column of the all-model figures above, the undefended model's static attack success under each,
and the Difference LoRA's multi-turn mean under each.
""")

code("8f3e61c5", r"""
# THE JUDGE COMPARISON (fig:app_judges): Qwen3.5-4B under the Claude judges (REGISTRY_NB_JUDGES=primary) and the paper's judges (rejudge),
# frozen at registry head 537565 and read from the committed file data/appendix_judge_comparison.json, which
# data/build_appendix_judge_comparison.py writes from a frozen registry copy. Nothing here reads or changes another figure's frames.
import appendix_judge_comparison as JC
jc = JC.load()
for _part in JC.PARTS:
    _fig, _info = JC.draw(jc, _part, show=True); plt.show()
    if (_jc_missing := JC.not_drawn(_info)):
        print(f"!! {JC.PARTS[_part]['name']}: one judge has no reading (grey x at zero): " + "; ".join(_jc_missing))
print(JC.summary_text(jc))
print(JC.provenance_line(jc))
""")

md("3c7a1e55", r"""
### Appendix — transfer: the same LoRA fit on the base checkpoint and served on the instruct one, against the fit made in place

The paper draft's Figure 4 comes back here as an appendix figure, on the three LoRAs (constitution, control, difference) for
**Gemma-4-E4B, Qwen3.5-4B and Llama-3.1-8B** (LoRAs only; Nemotron has one checkpoint and one fit). The
quantity is the object's **base→instruct** reading minus its **instruct-direct** reading, each lane's own anchored delta against the
unsteered model, per evaluation, oriented so that up means the transferred fit did better on that axis; zero means transfer cost
nothing. **Qwen's base→instruct side is its three batch-64 seed fits pooled** (never the earlier batch-8 base fits); the
**instruct-direct fit of every pair is pinned by object id** (`R.TRANSFER_PINS`; Gemma's constitution LoRA is the batch-8 in-place fit,
chosen over the five-evaluation batch-64 one). Pairs are **not** required to share a training batch — the batches on both sides are
stated in the pairs table as information. Each lane's fits are pooled as every seed pool is (their runs item by item, the variance between
the fits added) and the two lanes' anchors are netted: where both lanes are differenced against the same unsteered runs, the anchor is not in
the difference at all. The form is the x = object appendix figures': the category mean is the large mark with its
interval (one pool per lane over the category's evaluations), the evaluations small and faded beside it, a black ring a mean over fewer
evaluations than the category names, and an × at zero a (model, object) whose pair is not in the frame — the pairs table says why.
""")

code("5b9d2f7e", r"""
# the object records: the pinned object ids (TRANSFER_PINS) must each resolve to exactly one object, and the training batches ride along as
# information (own field, hyperparameters, or the parents' for a derived object)
_objects = R.query(R.OBJECTS_SQL, "objects")
_pins = R.transfer_pin_check(_objects)
_badpins = _pins[_pins.matches != 1]
if len(_badpins):
    print("!! pinned transfer object ids that do not resolve to exactly one registry object: "); R.show(_badpins)
else:
    print(f"transfer pins: all {len(_pins)} pinned object ids resolve to one registry object each; instruct-direct fits: "
          + ", ".join(f"{R.MODEL_LABEL.get(r.model_id, r.model_id)} {R.LABEL.get(r.object_label, r.object_label)} = {r.prefix} ({r.registry_label}, {r.fit_model_id})"
                      for r in _pins[_pins.lane == "instruct_direct"].itertuples()))
# THE ROSTER'S WORD AGAINST THE PINS (roster v10): the rows carrying the figure's own comparison id must be exactly the
# pinned objects; a disagreement here is a roster or a pin change nobody told the other side about.
_rc = R.transfer_roster_check(paper_all)
_off = _rc[_rc.status != "both"]
if not len(_rc):
    print(f"!! no in-frame row carries a `{R.TRANSFER_COMPARISON}/*` comparison id — roster v10 not loaded, or the id renamed; the figure draws from the pins alone.")
elif len(_off):
    print(f"!! roster `{R.TRANSFER_COMPARISON}` and TRANSFER_PINS DISAGREE on {len(_off)} (model, object, lane, id) rows:"); R.show(_off)
else:
    print(f"roster `{R.TRANSFER_COMPARISON}` and the figure's pins agree: {len(_rc)} (model, object, lane, id) rows, every one on both lists.")
_batch = R.resolve_fit_batch(_objects)
_lora = paper_all.object_label.isin(R.TRANSFER_OBJECTS)
# BOTH LANES' LORA ROWS AND EVERY UNSTEERED ROW of the three models: `anchored` takes its anchors from the frame it is handed
# (the pool serves Qwen alone), so a frame of fitted rows only would leave Gemma and Llama with no anchor at all.
tr_rows = paper_all[((_lora & paper_all.lane.isin(R.TRANSFER_LANES)) | paper_all.object_label.eq("none"))
                    & paper_all.model_id.isin(R.TRANSFER_MODELS) & paper_all.at_operating_point.astype(bool) & paper_all.headline].copy()
tr_rows, _ = R.drop_excluded_cells(tr_rows)                  # the floor hold-out applies here as everywhere
tr_rows, _tr_pad = R.drop_padded(tr_rows)                    # the padding instrument's rows never enter a static reading (18 Pliny rows once did)
tr_rows["fit_batch"] = tr_rows.object_id.map(_batch)
tr_d = R.anchored(tr_rows, pool=POOL_Q)                      # each lane's own delta against the unsteered model (the same anchoring as every panel)
tr_pe, tr_cat, tr_pairs = R.transfer_pairs(tr_d, R.TRANSFER_CATEGORIES, objects=R.TRANSFER_OBJECTS, models=R.TRANSFER_MODELS)
TR_OBJECTS = [o for o in R.TRANSFER_OBJECTS if o in set(tr_pairs.object_label)]
_missing_cats = [c for c in R.TRANSFER_CATEGORIES if c not in set(tr_cat.category)]
R.transfer_figure(tr_pe, tr_cat, tr_pairs, TR_OBJECTS, R.TRANSFER_MODELS, R.TRANSFER_CATEGORIES, "figA_transfer_models"); plt.show()
# A FLOOR IS NOT A CAPABILITY DIFFERENCE: the instruct-direct Qwen difference LoRA refuses ~1,136 of 1,140
# BigCodeBench prompts, so its pass rate is a refusal collapse and the pair's "transfer" on that eval is the collapse, not code ability.
_floors = R.transfer_floors(tr_pe, R.TRANSFER_CATEGORIES["capability"][0])
if len(_floors):
    print("capability cells at a FLOOR on one side (under 5 % -- a refusal collapse or a scoring failure, not a capability difference; "
          "the pair's point on that eval reads the collapse): "
          + "; ".join(f"{R.MODEL_LABEL.get(r.model_id, r.model_id)} {R.LABEL.get(r.object_label, r.object_label)} on {R.EVAL_LABEL.get(r.eval_id, r.eval_id)} "
                      f"(base→instruct {r.base_abs:.1f} %, instruct-direct {r.it_abs:.1f} %)" for r in _floors.itertuples()))
print(f"transfer pairs: {int((tr_pairs.status == 'matched').sum())} of {len(tr_pairs)} (model, LoRA) pairs in the frame; per-eval differences {len(tr_pe)}"
      + (f" · categories with no pair on any evaluation, left out: {', '.join(_missing_cats)}" if _missing_cats else ""))
R.show(tr_pairs)
# DEGENERACY AND TRUNCATION ON EITHER SIDE (for the caption): the cells whose replies are >= 10 % degenerate or at
# the token cap on the instruct-direct or the base->instruct side, with both sides' shares -- a difference across such a cell reads
# the failure mode as transfer. Nothing is discounted or hidden; the shares ride on every row and this names the affected cells.
_inst = R.transfer_instruments(tr_d)
_mixed = _inst[_inst.mixed].sort_values(["low_cap", "model_id"], ascending=[False, True])   # the 512-pooling cells first, never below the head
if len(_mixed):
    print(f"transfer cells pooling MORE THAN ONE instrument on a side ({len(_mixed)} of {len(_inst)}): the campaign pass and the strength ladder's "
          "dose-1 rung, or two caps, averaged as draws of one cell (the appendix's rule); caps and decodes per cell:")
    if _mixed.low_cap.any():
        print(f"!! {int(_mixed.low_cap.sum())} of them pool a CAP-512 run with a higher cap — the signature of a superseded duplicate still marked selected "
              "(a supersede of the duplicate retires it). Every one is listed:")
        with pd.option_context("display.max_rows", None, "display.width", 250):
            print(_mixed[_mixed.low_cap].to_string())
    R.show(_mixed, n=40)
_sh = R.transfer_shares(tr_d)
if len(_sh):
    print(f"transfer cells with a degenerate or at-cap share >= 10 % on either side ({len(_sh)} of {len(tr_pe)} drawn cells; shares as fractions, "
          "the side's own reading beside them):")
    R.show(_sh.round(3), n=60)
# MULTI-TURN DEGENERACY, COUNTED PER GOAL: the reply-level shares above overstate what degeneracy does to a
# per-goal ASR (a few goals sampled dozens of times carry most of the collapsed replies). Where the grading pass filed the goal-level
# decomposition, this is the ASR over the goals WITHOUT a degenerate turn beside the headline ASR -- the "non-degenerate subset"
# reading -- and where it did not (the Qwen seed fits, Llama) the cell says so rather than borrowing another run's shares.
_mt_metrics = R.query(R.MT_DEGEN_SQL, "mt_degeneracy")
_mt_clean = R.multiturn_clean_asr(tr_d, _mt_metrics)
if len(_mt_clean):
    _mv = _mt_clean[_mt_clean.has_goal_rows]
    print("multi-turn transfer cells, ASR over ALL goals vs over the goals with NO degenerate turn (goal-level rows on the headline judgment; "
          f"{len(_mv)} of {len(_mt_clean)} cells carry them; largest |all − clean| = {(_mv.asr_all - _mv.asr_clean).abs().max():.1f} pp):")
    R.show(_mt_clean.assign(asr_all=_mt_clean.asr_all.round(1), asr_clean=_mt_clean.asr_clean.round(1), goal_degen_share=_mt_clean.goal_degen_share.round(1)), n=40)
print(R.table_transfer(tr_cat, tr_pairs, TR_OBJECTS, R.TRANSFER_MODELS, R.TRANSFER_CATEGORIES,
      "Transfer: base$\\to$instruct minus instruct-direct, three LoRAs, three models.", "tab:allmodels-transfer"))
R.show(tr_pe.assign(diff=tr_pe["diff"].round(1), lo=tr_pe.lo.round(1), hi=tr_pe.hi.round(1), base_dy=tr_pe.base_dy.round(1), it_dy=tr_pe.it_dy.round(1),
                    base_abs=tr_pe.base_abs.round(1), it_abs=tr_pe.it_abs.round(1))
       .sort_values(["model_id", "object_label", "eval_id"]))
""")

md("7e2c9b41", r"""
### Appendix — the control objects with the sign flipped: Qwen control LoRA and control ReLU steer at α = −1

The two control objects served at dose −1 (the same fits, sign-flipped) on the static, multi-turn and over-refusal evaluations, in the
main figures' form: static with its mean above, multi-turn and over-refusal each with their mean below. Every point is the change against
the unsteered model as in Figures 3–6, oriented so up is better; seeds are separate small marks and the mean averages them. The rows are
the α = −1 shipment; the cell draws whatever the frame holds at dose −1.0 for the two objects and states the sources, seeds
and evaluation counts it found under the figure.
""")

code("f1a8c3d7", r"""
# THE PAPER FRAME, like every other panel: roster v11 names both control objects at
# alpha = -1 under the comparison figA_qwen_control_alpha_minus1 -- each standup's own dose-scaled object id, so nothing else at -1
# (the strength ladder's paired-write rung, the batch-8 native control point) is admitted -- and every row reads selected with
# in_current_paper = 1. Until v11 loaded the arms came from a dedicated v_paper_long query so the not_in_roster
# control ReLU rows could draw marked "roster pending"; `negative_dose_rows` still admits not_in_roster rows and names them, so
# a future object that lands before its roster entry draws marked rather than vanishing. Over-refusal is already repointed to
# the label mean on paper_all at load.
NEG_CATS = {c: R.categories_for(R.MODEL_ID)[c] for c in ("static", "multi-turn", "over-refusal")}
NEG_EVALS = [e for c in NEG_CATS.values() for e in c[0]]
neg_rows = R.negative_dose_rows(paper_all, R.MODEL_ID, R.NEGATIVE_DOSE_OBJECTS, dose=-1.0)
_seedless = neg_rows.attrs.get("seedless")
neg_rows, _ = R.drop_excluded_cells(neg_rows)
_prov = R.negative_dose_provenance(neg_rows, R.NEGATIVE_DOSE_OBJECTS)
_pend = R.negative_dose_pending(neg_rows, R.NEGATIVE_DOSE_OBJECTS, evals=NEG_EVALS)
if len(_prov):
    neg_d = R.anchored(neg_rows, pool=POOL_Q)
    neg_cm = R.category_means(neg_d, categories=NEG_CATS)
    neg_gaps = R.gap_status(runs_id[runs_id.object_label.isin(R.NEGATIVE_DOSE_OBJECTS) & pd.to_numeric(runs_id.dose, errors="coerce").eq(-1.0)]
                            if "dose" in runs_id.columns else runs_id.iloc[0:0],
                            R.drawn_set(neg_d[neg_d.d_y.notna()]), R.NEGATIVE_DOSE_OBJECTS, [e for c in NEG_CATS.values() for e in c[0]], wrong_judge=WJ)
    R.delta_figure(neg_cm, neg_d, R.NEGATIVE_DOSE_LAYOUT, R.NEGATIVE_DOSE_OBJECTS, neg_gaps, "figA_qwen_control_alpha_minus1", categories=NEG_CATS); plt.show()
    print("α = −1 rows read (object · source · lane · seeds · evals · runs):"); R.show(_prov)
    if _seedless:
        print("NOT drawn — dose −1 rows of a fit that is none of the three seeds (no seed label; the strength ladder's own −1 rung, which the "
              "frontier also leaves out): " + "; ".join(f"{R.LABEL.get(r['object_label'], r['object_label'])} · {r.get('lane_source_legacy')} · "
              f"{r.get('view_legacy')}: {r['runs']} runs on {r['evals']}" for r in _seedless))
    if len(_pend):
        print("drawn from rows the roster does not yet name (`not_in_roster`; roster v11 adds figA_qwen_control_alpha_minus1 entries for both "
              "families): " + "; ".join(f"{R.LABEL.get(r.object_label, r.object_label)}: {r.rows} runs on {r.evals}" for r in _pend.itertuples())
              + " — the mark retires by itself the render after v11 loads.")
    print(R.table_evals_seeds(neg_d, R.NEGATIVE_DOSE_LAYOUT, R.NEGATIVE_DOSE_OBJECTS,
          "Qwen3.5-4B control objects at $\\alpha = -1$.", "tab:qwen-control-alpha-minus1", categories=NEG_CATS))
else:
    print("no rows at dose −1.0 for the control objects are in the frame yet (the α = −1 shipment pending); the figure appears the render after they fold.")
""")

md("5c1d9e73", r"""
### Appendix — static jailbreak defense with the control objects stopped on the control pool (Qwen)

Figure 3's static panel with the control LoRA and the control ReLU steer refit so that early stopping reads the control corpus's own
held-out pool (the published fits stopped on the constitution corpus's pool), and the difference LoRA and the difference ReLU steer
rebuilt from those fits. That rule never stopped early: each control fit trained to the 16-epoch cap and kept epoch 15 or 16. The
Refusal steer, the two system instructions and the constitution LoRA and ReLU steer are Figure 3's own runs, and every point is
differenced against the eval's unsteered anchor by Figure 3's rule. Each fitted object is its three batch-64 seeds, each read at the
rung its own ladder stopped on. Four control-pool Pliny readings, named under the figure, are read at their 16,384 rung and
differenced against the static panel's unsteered runs at 8,192, the final rung of the same ladder, where those runs had nothing left
to climb. Every point of this figure is judged by MiniMax-M3, the five objects that were not refit and the unsteered anchors
included (StrongREJECT: a goal counts as broken when its best reply scores above the threshold printed under the figure), so those
five objects show Figure 3's values where Figure 3 is drawn from that judge at that threshold.
""")

code("a6e0f2b8", r"""
# THE CONTROL-POOL STOPPING APPENDIX: Figures 3, 6 and 4 with the four refit objects swapped for their control-pool fits, then the two
# stopping rules side by side. ONE JUDGE PER FIGURE, WHATEVER THE RENDER'S POLICY: every judged point is MiniMax-M3's, read from the
# committed file's per-item payloads at R.SR_THRESHOLD through the notebook's judging interface -
#   * the four refit slots' rows (ctl_rows), and
#   * the main figures' own runs the appendix draws beside them or differences against: the five objects not refit, the published
#     fits, every unsteered anchor (ctl_op / ctl_pool: this render's `op` and POOL_Q with each such row's reading taken from the file;
#     ctl_md, their anchored frame, selects the runs `d` selects, by `anchored` itself).
# A live render re-derives the whole file from the registry and refuses on any difference; and wherever this render's own rows are
# the same judge's at the same threshold (the re-judge policy), ctl_md is held to `d`, value for value. MMLU and GSM8K are scored by
# rule and are the main figures' rows. Every frame the figures above read is only read here; what these cells make is named ctl_*.
ctl_rows, ctl_pending, _ctl_file = R.ctlstop_load(R.SOURCE, allow_pending=True)
CTL_CATS = R.ctlstop_categories(R.MODEL_ID)
print(f"control-pool appendix data: {len(ctl_rows)} runs drawn from {R.display_path(R.CTLSTOP_SIDECAR)} ("
      + ("checked field by field against the live registry" if R.SOURCE == "live" else "the snapshot render reads the file alone")
      + f"); every StrongREJECT reading at threshold {R.SR_THRESHOLD:g} (a goal is broken when its best score is above it); "
      + f"{len(ctl_pending)} reading(s) not yet in the data")
if ctl_pending:
    print("  not drawn, by eval: " + "; ".join(f"{R.EVAL_LABEL.get(e, e)} {n}" for e, n in pd.Series([p['eval_id'] for p in ctl_pending]).value_counts().sort_index().items()))
ctl_op, ctl_pool, ctl_main = R.ctlstop_main_frames(op, POOL_Q, d, _ctl_file)
ctl_md = R.anchored(ctl_op, pool=ctl_pool)
ctl_mcm = R.category_means(ctl_md, categories=CTL_CATS)
_ctl_chk = R.ctlstop_main_check(d, ctl_md, _ctl_file, frames=(op, POOL_Q))
print(f"the main figures' runs under {R.ctlstop_fixed_judge()}: {ctl_main['read']} rows read from the file's payloads, of {ctl_main['runs']} runs "
      f"this render's main figures draw on these evals; {_ctl_chk['rows']} points anchored"
      + (f"; the runs drawn are the main figures' ({'except ' + str(len(_ctl_chk['left_out'])) + ' named below' if _ctl_chk['left_out'] else 'every one'})"
         if _ctl_chk["run_set_checked"] else "; the run set is compared at the paper's threshold only")
      + f"; {_ctl_chk['same_judge_rows']} points equal this render's own (the same judge at the same threshold)")
if ctl_main["no_judgment"] or ctl_main["not_derivable"]:
    print(f"  !! {len(ctl_main['no_judgment'])} run(s) the main figures draw have no reading under this judge and are not drawn here (by eval: "
          + "; ".join(f"{R.EVAL_LABEL.get(e, e)} {n}" for e, n in pd.Series([m['eval_id'] for m in ctl_main['no_judgment'].values()]).value_counts().sort_index().items())
          + ")" + (f"; {len(ctl_main['not_derivable'])} not derivable: {list(ctl_main['not_derivable'].items())[:3]}" if ctl_main["not_derivable"] else ""))
    if _ctl_chk["left_out"]:
        print("  points left out: " + "; ".join(_ctl_chk["left_out"][:40]) + (" ..." if len(_ctl_chk["left_out"]) > 40 else ""))
    if _ctl_chk["smaller_pool"]:
        print(f"  {len(_ctl_chk['smaller_pool'])} of the {_ctl_chk['compared']} point(s) this render's own frame also draws are differenced against "
              "fewer unsteered runs than in that frame (a pooled run has no reading under this judge)"
              + ("" if _ctl_chk["run_set_checked"] else "; at this threshold the frame holds gaps of its own, so this is not the count at the paper's threshold")
              + ", by eval: " + "; ".join(f"{R.EVAL_LABEL.get(e, e)} {n}" for e, n in
              pd.Series([x.split(' ')[0] for x in _ctl_chk['smaller_pool']]).value_counts().sort_index().items()))
if _ctl_chk["blanked_in_frame"]:
    print(f"  {len(_ctl_chk['blanked_in_frame'])} point(s) are drawn here from the file where this render's own frame holds the run with no "
          "reading under its judge or threshold (its main figures show a gap there)")
ctl_st_cats = {"static": CTL_CATS["static"]}
ctl_st_d, ctl_st_cm, ctl_st_new = R.ctlstop_mirror(ctl_op, ctl_md, ctl_mcm, ctl_pool, ctl_rows, ctl_st_cats)
ctl_st_gaps = R.ctlstop_gaps(R.gap_status(runs_id, R.drawn_set(ctl_st_d[ctl_st_d.d_y.notna()]), R.OBJECTS, list(ctl_st_cats["static"][0]), wrong_judge=WJ),
                             ctl_st_d, d, ctl_pending)
print(R.ctlstop_anchor_report(ctl_st_d, ctl_rows, ctl_op, ctl_pool, ctl_st_cats))       # raises unless every point and anchor is the file's
# what the figure and its table are handed: a reading left without a delta is in ctl_st_d and is not drawn, so a cell with no seed
# left takes the mark of its own status in ctl_st_gaps (no unsteered run of the static panel at its cap)
ctl_st_pts = R.ctlstop_figure_rows(ctl_st_d)
_ctl_st_fig = "figA_qwen_ctlstop_static"          # named as a literal by the drawing call alone
if R.ctlstop_drawable(ctl_st_new, ctl_st_cats):
    R.delta_figure(ctl_st_cm, ctl_st_pts, [["static"]], R.OBJECTS, ctl_st_gaps, "figA_qwen_ctlstop_static", categories=ctl_st_cats,
                   legend_ncol=3, legend_pad=2); plt.show()
    print(R.ctlstop_save_table(_ctl_st_fig, R.table_evals_seeds(ctl_st_pts, [["static"]], R.OBJECTS,
          "Static defense for Qwen3.5-4B with the control objects stopped on the control pool.", "tab:qwen-ctlstop-static-seeds",
          categories=ctl_st_cats, note=R.ctlstop_static_table_note(ctl_st_new))))
    print("the rung each refit reading was drawn at (token cap), by eval:",
          {e: {int(c): int(n) for c, n in g.token_cap_tokens.value_counts().sort_index().items()}
           for e, g in ctl_st_new[ctl_st_new.d_y.notna()].groupby("eval_id")})
    for _ctl_note in R.ctlstop_declared_notes(ctl_st_new):       # the file's declared anchors: read at their rung, differenced at the cap declared
        print("  " + _ctl_note)
    for _ctl_note in R.ctlstop_rung_gap_notes(ctl_st_new):       # read above the rung of every unsteered run of the static panel: named, not drawn
        print("  !! " + _ctl_note)
    print("per eval, the refit slots' readings (y), the unsteered anchor they are compared with (anchor_value) and the delta (d_y), in plotted units:")
    R.show(ctl_st_new[DCOLS].sort_values(["eval_id", "object_label", "seed"]).round(1), n=48)
else:
    print(f"{_ctl_st_fig}: NOT DRAWN and not saved - no control-pool reading of its evals is in the data yet"
          + (f" (removed from an earlier render: {', '.join(_gone)})" if (_gone := R.ctlstop_unsave(_ctl_st_fig)) else ""))
""")

md("3f8b4c61", r"""
### Appendix — over-refusal and capability with the control objects stopped on the control pool (Qwen)

Figure 6's two panels with the same four objects swapped for their control-pool fits; the other five objects are Figure 6's own
runs, and every point is differenced against the eval's unsteered anchor by Figure 6's rule. On XSTest and OR-Bench-Hard each refit
seed is differenced against the same unsteered runs as Figure 6's seed-view fitted points of that evaluation and seed: every fitted
object's but the difference ReLU steer's, whose Figure 6 points are paired-write readings differenced against the paired-write
views' own unsteered runs. The capability panel is MMLU and GSM8K, and its mean is the mean of those two: Figure 6's capability
mean also averages BigCodeBench and BFCL, which these objects were not run on, so this panel's mean is not Figure 6's. Every
over-refusal point is a MiniMax-M3 label, the five objects that were not refit and the unsteered anchors included (a full refusal
scores 1, a partial ½, compliance 0, over XSTest's safe prompts and every OR-Bench-Hard prompt), so those five objects show
Figure 6's over-refusal values where Figure 6 is drawn from that judge; MMLU and GSM8K are scored by rule and are Figure 6's values.
""")

code("d2a7e5c9", r"""
# FIGURE 6'S PANELS WITH THE FOUR CONTROL-POOL OBJECTS: over-refusal (each seed against the unsteered runs Figure 6's seed-view
# fitted rows of the same eval and seed are differenced against in this render: its seed view's own run where the registry binds one
# to them, else the runs pooled for them, said under the figure; every label the appendix's judge's) and capability narrowed to MMLU
# and GSM8K, whose two-eval mean the carried objects are held to (recomputed from Figure 6's own rows; Figure 6's own mean is over
# four evals).
ctl_cap_cats = {c: CTL_CATS[c] for c in ("over-refusal", "capability")}
ctl_cap_d, ctl_cap_cm, ctl_cap_new = R.ctlstop_mirror(ctl_op, ctl_md, ctl_mcm, ctl_pool, ctl_rows, ctl_cap_cats)
_ctl_cap_evals = [e for c in ctl_cap_cats.values() for e in c[0]]
ctl_cap_gaps = R.ctlstop_gaps(R.gap_status(runs_id, R.drawn_set(ctl_cap_d[ctl_cap_d.d_y.notna()]), R.OBJECTS, _ctl_cap_evals, wrong_judge=WJ),
                              ctl_cap_d, d, ctl_pending)
print(R.ctlstop_anchor_report(ctl_cap_d, ctl_rows, ctl_op, ctl_pool, ctl_cap_cats))
_ctl_cap_fig = "figA_qwen_ctlstop_capabilities"
if R.ctlstop_drawable(ctl_cap_new, ctl_cap_cats):
    R.delta_figure(ctl_cap_cm, ctl_cap_d, [["over-refusal", "capability"]], R.OBJECTS, ctl_cap_gaps, "figA_qwen_ctlstop_capabilities",
                   categories=ctl_cap_cats); plt.show()
    print(R.ctlstop_save_table(_ctl_cap_fig, R.table_evals_seeds(ctl_cap_d, [["over-refusal", "capability"]], R.OBJECTS,
          "Over-refusal and capability (the mean of MMLU and GSM8K) for Qwen3.5-4B with the control objects stopped on the control pool.",
          "tab:qwen-ctlstop-capabilities-seeds", categories=ctl_cap_cats, note=R.ctlstop_binding_table_note(ctl_cap_new))))
    print("the rung each refit reading was drawn at (token cap), by eval:",
          {e: {int(c): int(n) for c, n in g.token_cap_tokens.value_counts().sort_index().items()} for e, g in ctl_cap_new.groupby("eval_id")})
    for _ctl_note in R.ctlstop_exit_rung_notes(_ctl_file, set(ctl_cap_new.run_id)):      # a control-pool reading below the others' rung, named at its own
        print("  " + _ctl_note)
    for _ctl_note in R.ctlstop_binding_notes(ctl_cap_new):       # the over-refusal anchor: the registry-bound run, or the runs pooled for Figure 6's rows
        print("  " + _ctl_note)
    R.show(ctl_cap_new.reindex(columns=DCOLS + ["anchor_binding"]).sort_values(["eval_id", "object_label", "seed"]).round(1), n=48)
else:
    print(f"{_ctl_cap_fig}: NOT DRAWN and not saved - no control-pool over-refusal reading is in the data yet"
          + (f" (removed from an earlier render: {', '.join(_gone)})" if (_gone := R.ctlstop_unsave(_ctl_cap_fig)) else ""))
""")

md("1b4eb5de", r"""
### Appendix — multi-query and multi-turn defense with the control objects stopped on the control pool (Qwen)

Figure 4's two panels with the same four objects swapped for their control-pool fits; the Refusal steer, the two system instructions
and the constitution LoRA and ReLU steer are Figure 4's own runs, and every point is differenced against the eval's common unsteered
anchor by Figure 4's rule. The four refit objects' runs were generated on a later code tree and serving setup than Figure 4's; the
unsteered model rerun in that setup is printed under the figure beside the anchor the points are differenced against. Every point of
this figure is judged by MiniMax-M3, the five objects that were not refit and the unsteered anchors included (StrongREJECT, Best-of-75
on the rubric Figure 4's Best-of-75 rows use, at the threshold printed under the figure), so those five objects show Figure 4's values
where Figure 4 is drawn from that judge at that threshold. A point whose run that judge has no verdict of is marked as a gap.
""")

code("7fae15d6", r"""
# FIGURE 4'S PANELS WITH THE FOUR CONTROL-POOL OBJECTS. The five objects that were not refit are Figure 4's own runs under the
# appendix's judge (ctl_md); the four refit slots are the control-pool fits' runs from the committed file, anchored by Figure 4's own
# code on the unsteered rows under the same judge, and asserted to carry Figure 4's anchor rule on every eval.
ctl_cats = {c: CTL_CATS[c] for c in ("multi-query", "multi-turn")}
ctl_evals = [e for c in ctl_cats.values() for e in c[0]]
ctl_d, ctl_cm, ctl_new_d = R.ctlstop_mirror(ctl_op, ctl_md, ctl_mcm, ctl_pool, ctl_rows, ctl_cats)
ctl_gaps = R.ctlstop_gaps(R.gap_status(runs_id, R.drawn_set(ctl_d[ctl_d.d_y.notna()]), R.OBJECTS, ctl_evals, wrong_judge=WJ), ctl_d, d, ctl_pending)
print(R.ctlstop_anchor_report(ctl_d, ctl_rows, ctl_op, ctl_pool, ctl_cats))
_ctl_mt_fig = "figA_qwen_ctlstop_multi_turn"
if R.ctlstop_drawable(ctl_new_d, ctl_cats):
    R.delta_figure(ctl_cm, ctl_d, [["multi-query", "multi-turn"]], R.OBJECTS, ctl_gaps, "figA_qwen_ctlstop_multi_turn", categories=ctl_cats); plt.show()
    print(R.ctlstop_save_table(_ctl_mt_fig, R.table_evals_seeds(ctl_d, [["multi-query", "multi-turn"]], R.OBJECTS,
          "Multi-query and multi-turn defense for Qwen3.5-4B with the control objects stopped on the control pool.",
          "tab:qwen-ctlstop-multi-turn-seeds", categories=ctl_cats)))
    _ctl_drift = R.ctlstop_drift(ctl_rows, ctl_new_d)
    print("the unsteered model rerun in the control-pool runs' setup, against the anchor every point is differenced against: "
          + ("; ".join(f"{R.EVAL_LABEL[x['eval_id']]} {x['rerun']:.1f}% attack success against {x['anchor']:.1f}% ({x['anchor_runs']} unsteered "
                       f"runs), {x['drift']:+.1f} pp" for x in _ctl_drift) or "not drawn in this render"))
    print("cells marked as gaps (object · eval · why):", [(R.LABEL.get(r.object_label, r.object_label), r.eval_id, r.status)
                                                          for r in ctl_gaps[~ctl_gaps.status.eq("drawn")].itertuples()])
    print("the mean points drawn (one per object: the seeds averaged; n_seeds says how many):")
    R.show(R.pool_seeds(ctl_cm).drop(columns="model_id").round(2), n=18)
    print("per eval, the refit slots' readings (y), the unsteered anchor they are compared with (anchor_value) and the delta (d_y), in plotted units:")
    R.show(ctl_new_d[DCOLS].sort_values(["eval_id", "object_label", "seed"]).round(1), n=48)
else:
    print(f"{_ctl_mt_fig}: NOT DRAWN and not saved - no control-pool reading of its evals is in the data yet"
          + (f" (removed from an earlier render: {', '.join(_gone)})" if (_gone := R.ctlstop_unsave(_ctl_mt_fig)) else ""))
""")

md("de9a82d2", r"""
### Appendix — published against control-pool stopping, seed by seed (Qwen; Figure 11)

For the control ReLU steer, the control LoRA, the difference ReLU steer and the difference LoRA, each seed's published fit (circle)
beside its control-pool fit (diamond), joined by a line, both as the change against the unsteered model in percentage points with
their 95% intervals, on Figure 3's static panel, Figure 4's multi-query and multi-turn panels and Figure 6's over-refusal and
capability panels (capability: MMLU and GSM8K); each mean column averages each stopping rule over the panel's evals within seed, then
over the seeds. The control-pool rule never stopped early: each control fit trained to the 16-epoch cap and kept epoch 15 or 16. Every
point is differenced against the eval's unsteered anchor, as in the main figures. The published control ReLU steer, control LoRA and
difference LoRA are the main figures' runs, generated on an earlier code tree and serving setup than the control-pool runs; seed 1's
published static runs were generated on another serving stack than seeds 2 and 3. The published difference ReLU steer was rerun
in the control-pool runs' setup on the two multi-turn attacks, so both of its marks there are read the same way; on every other panel
its published marks are the main figures' paired-write runs, a different evaluation setup (on Best-of-75 also the earlier
StrongREJECT rubric). Every mark of a judged evaluation, published or control-pool, is a MiniMax-M3 verdict at the threshold printed
under the figure, and so is every unsteered anchor; a mark whose run that judge has no verdict of is a gap. Four control-pool
Pliny marks, named under the figure, are read at their 16,384 rung and differenced against the static panel's unsteered runs at
8,192, the final rung of the same ladder, where those runs had nothing left to climb.
""")

code("76fdab54", r"""
# THE TWO STOPPING RULES, WITHIN SEED, ON EVERY FAMILY. Published = the main figures' own runs for the four slots under the
# appendix's judge (ctl_md), except the difference ReLU steer on the multi-turn attacks, which is its like-for-like rerun read from the
# file and anchored exactly as the control-pool rows are (never the paired-lane reading Figure 4 draws there). Control-pool = the
# three mirrors' refit rows.
ctl_pub_d = R.ctlstop_published(ctl_op, ctl_md, ctl_pool, ctl_rows, CTL_CATS)
ctl_all_new = R.concat_plain(*[f for f in (ctl_st_new, ctl_new_d, ctl_cap_new) if len(f)] or [ctl_new_d]).copy()
ctl_pub_cm, ctl_new_cm = (R.category_means(f, categories=CTL_CATS) for f in (ctl_pub_d, ctl_all_new))
print(R.ctlstop_anchor_report(R.concat_plain(ctl_pub_d, ctl_all_new), ctl_rows, ctl_op, ctl_pool, CTL_CATS))      # both rules' marks
_ctl_cmp_fig = "figA_qwen_ctlstop_rule_comparison"
if R.ctlstop_drawable(ctl_all_new, CTL_CATS):
    R.rule_comparison_figure(ctl_pub_d, ctl_all_new, ctl_pub_cm, ctl_new_cm, "figA_qwen_ctlstop_rule_comparison", categories=CTL_CATS,
                             layout=R.CTLSTOP_GRID, pending=ctl_pending); plt.show()
    _ctl_missing = R.ctlstop_not_drawn(ctl_pending, ctl_pub_cm, ctl_new_cm, CTL_CATS, gaps=ctl_all_new)
    if _ctl_missing:                                     # each named mark is drawn as the gap glyph, with no joining line
        print("!! NOT DRAWN (gap glyph in its place): " + "; ".join(_ctl_missing))
    for _ctl_note in R.ctlstop_declared_notes(ctl_all_new):      # the file's declared anchors: read at their rung, differenced at the cap declared
        print(_ctl_note)
    for _cat, _part in (("static", "-static"), ("multi-query", "-multi-query"), ("multi-turn", ""), ("over-refusal", "-over-refusal"),
                        ("capability", "-capability")):
        print(R.ctlstop_save_table(_ctl_cmp_fig, R.table_rule_comparison_seeds(
              ctl_pub_d, ctl_all_new, ctl_pub_cm, ctl_new_cm, f"Published and control-pool stopping, per seed, {_cat} ({'MMLU and GSM8K' if _cat == 'capability' else 'all its evaluations'}), for Qwen3.5-4B.",
              f"tab:qwen-ctlstop-rule-comparison{_part}-seeds", categories={_cat: CTL_CATS[_cat]}), part=_part or "-multi-turn"))
    for _ctl_note in R.ctlstop_exit_rung_notes(_ctl_file, set(ctl_all_new.run_id)):      # a control-pool reading below the others' rung, named at its own
        print(_ctl_note)
    print("per seed, each stopping rule's category means (the small marks' means):")
    R.show(pd.concat([ctl_pub_cm.assign(rule="published"), ctl_new_cm.assign(rule="control_pool")], ignore_index=True)
           [["rule", "category", "object_label", "seed", "y", "lo", "hi"]].round(2), n=60)
else:
    print(f"{_ctl_cmp_fig}: NOT DRAWN and not saved - no control-pool reading of a judged evaluation is in the data yet"
          + (f" (removed from an earlier render: {', '.join(_gone)})" if (_gone := R.ctlstop_unsave(_ctl_cmp_fig)) else ""))
""")

md("5c3e9a71", r"""
### Appendix — multi-turn defense by distillation target (Qwen)

Multi-turn defense on Qwen3.5-4B of objects distilled from a prompted teacher, by distillation target, each beside the constitution
corpus fit of the same kind. Each diamond is a distilled object at one fit seed and the circle beside it, joined by a thin line, is the
constitution corpus fit of the same kind at the same seed: the distilled LoRAs beside the Constitution LoRA, the distilled ReLU steers
beside the Constitution ReLU steer, so the corpus fit's circle repeats in each target's slot of its kind. Colour gives the distillation
target and the kind (a target's LoRA dark, its ReLU steer light). Each point is the change in defense success (percentage points)
against the unsteered model on Crescendo and ActorAttack, differenced against the unsteered model's pooled anchor exactly as in
Figure 4; the bold mean column averages the two attacks within a seed, then across seeds. Bars are 95% intervals. Attack success is
StrongREJECT graded by MiniMax-M3, a goal counting as a success when its best reply scores above 0.5. The safety and constitution
objects are KL distillations of the two prompts used as the prompted conditions, on one matched pool of 9,164 pairs in which only the
teacher's completion differs; the irrelevant-instruction control uses a 6,518-pair pool from the same attack families with the same goal
coverage. The distilled objects have the same architecture as the paper's LoRA and ReLU-steer objects (a rank-4 LoRA on the attention
output projection of the eight full-attention layers, and a ReLU steer on every layer, both acting on every token), and all use the
hyperparameters tuned for the safety prompt. The constitution corpus fits are fit by cross-entropy on the synthetic constitution
documents to the base model and served on the instruct model. The training pools contain prompts for 163 of the 259 evaluation goals
(167 counting identical behaviour text); the lower row repeats the comparison on the 92 goals outside the pool, against the unsteered
model on the same goals. The two prompted instructions are not drawn; the tables below list them beside every distilled object and
corpus fit, per seed.
""")

code("a2d84f6b", r"""
# THE DISTILLATION-TARGET APPENDIX, in the comparison layout: each distilled object (diamond) beside the constitution corpus fit of the
# same kind and seed (circle), all 259 goals and the 92 outside the training pool. Drawn only when this render reads the multi-turn
# attacks from the figure's one judge (MiniMax-M3 StrongREJECT v1, the re-judge policy); under any other policy one RENDER POLICY line
# says so and nothing is drawn. The corpus fits and the prompted instructions are Figure 4's own rows (copies of `d`, their run ids
# asserted against the pins); the six distilled objects' runs come from the committed sidecar (the render checks every stored field,
# every per-goal payload and every reading against the registry it read, live or the committed snapshot, and refuses on any
# difference), anchored by Figure 4's own code on `op`'s unsteered rows and asserted to carry Figure 4's anchor. The lower row
# recomputes every reading on the goals outside the training pool from the sidecar's per-goal best scores, against the unsteered runs
# Figure 4's anchor pools, on the same goals. Every run drawn, tabulated or pooled is held to its eval's one rubric before anything is
# drawn. Every frame the figures above read is only read here; what this cell makes is named st_*.
st_cats = {"multi-turn": CAT_Q["multi-turn"]}
st_ok, st_why = R.sitarget_render_ok()
if not st_ok:
    print(st_why)
else:
    _st_judge = R.sitarget_judge_problems()
    if _st_judge:
        raise RuntimeError("the distillation-target figure's judge pins disagree with the policy lookup: " + "; ".join(_st_judge))
    st_rows, st_goal_best, st_goals = R.sitarget_load(R.SOURCE, frontier)
    st_new = R.sitarget_anchored(op, st_rows[st_rows.slot.isin(R.SITARGET_NEW_SLOTS)], POOL_Q, d)
    st_d = pd.concat([f for f in (R.sitarget_fig4_rows(d), st_new) if len(f)], ignore_index=True).copy()
    st_cm = R.category_means(st_d, categories=st_cats)
    st_out_d = R.sitarget_outside_pool_anchored(st_rows, st_goal_best, st_goals, d)
    st_out_cm = R.category_means(st_out_d, categories=st_cats)
    # ONE RUBRIC PER EVAL: every drawn, tabulated and pooled run on its eval's one judge, or nothing is drawn
    st_unsteered = pd.concat([op[op.object_label.eq("none")], POOL_Q], ignore_index=True)
    _st_probs = R.sitarget_rubric_problems(st_d, st_unsteered)
    if _st_probs:
        raise R.SitargetRubricMismatch(f"{len(_st_probs)} run(s) off their eval's one rubric: " + "; ".join(_st_probs[:10]), _st_probs)
    for _l in R.sitarget_rubric_lines(st_d, st_unsteered):
        print(_l)
    st_listing = R.sitarget_run_listing(st_d, st_unsteered)
    print("instruments and caps: " + R.sitarget_disclosure(st_listing))
    st_views = {"all": (st_d, st_cm), "outside": (st_out_d, st_out_cm)}
    R.sitarget_comparison_figure(st_views, "figA_qwen_si_target_multi_turn", categories=st_cats); plt.show()
    _st_missing = R.sitarget_not_drawn(st_views, st_cats)
    if _st_missing:                                      # each named mean is drawn as the gap glyph
        print("!! NOT DRAWN (gap glyph in its place): " + "; ".join(_st_missing))
    print(R.sitarget_save_table(R.sitarget_table_seeds(
        st_d, "Multi-turn defense for Qwen3.5-4B by distillation target.", "tab:qwen-si-target-multi-turn-seeds", categories=st_cats,
        note_extra=R.sitarget_undetermined_note(st_rows, st_goal_best, st_goals))))
    print(R.sitarget_save_table(R.sitarget_table_outside_pool(
        st_out_d, "Multi-turn defense for Qwen3.5-4B by distillation target, on the goals outside the training pool.",
        "tab:qwen-si-target-outside-pool", categories=st_cats,
        note_extra=R.sitarget_undetermined_note(st_rows, st_goal_best, st_goals, subset=True)), "-outside-pool"))
    print(f"the distilled objects: {st_new.run_id.nunique()} runs of {st_new.object_id.nunique()} objects, and the per-goal best scores of "
          f"{len(st_goal_best)} runs, from {R.display_path(R.SITARGET_SIDECAR)} (checked field by field, payload by payload and reading by "
          f"reading against the {'live registry' if R.SOURCE == 'live' else 'committed snapshot'}); {int(st_goals['outside_pool_n'])} of "
          f"{R.SITARGET_N_GOALS} goals outside the training pool; StrongREJECT threshold {R.SR_THRESHOLD:g}")
    for _k, _f in (("all goals", st_d), ("goals outside the training pool", st_out_d)):
        print(f"anchors, {_k}: " + "; ".join(f"{R.EVAL_LABEL[e]} {g.anchor_value.iat[0]:.1f}% attack success ({int(g.anchor_draws.iat[0])} "
                                              f"unsteered runs)" for e, g in _f[_f.d_y.notna()].groupby("eval_id")))
    print("the mean points (one per slot: the seeds averaged; n_seeds says how many), all goals then outside the pool:")
    R.show(R.pool_seeds(st_cm).drop(columns="model_id").round(2), n=12)
    R.show(R.pool_seeds(st_out_cm).drop(columns="model_id").round(2), n=12)
""")

code("bb79fa42", r"""
# THE PAPER'S TABLE OF THE DISTILLATION-TARGET FIGURE (tab:app_si_distilled_mt): per arm the seed mean (a prompted instruction's mean over
# its decode trials) of the change in defense success on Crescendo, on ActorAttack and their mean on all goals, and the mean on the goals
# outside the training pool, each with its 95% interval, under the defense success rate of the unsteered runs every row is differenced
# against. Every number comes from the figure's own frames above (st_d, st_out_d) through the notebook's reducers (category_means,
# pool_seeds); the caption names the goals a reading leaves out from the sidecar's per-goal best scores. Drawn only where the figure is,
# and written to figures/ beside it.
if st_ok:
    print(R.sitarget_save_paper_table(R.sitarget_paper_table(st_d, st_out_d, st_rows, st_goal_best, categories=st_cats)))
""")

md("e3b71c58", r"""
### Appendix: every family of the main figures by distillation target (Qwen)

Twelve arms on every evaluation family of the main figures, one panel per category in four rows: static; multi-query beside
multi-turn; over-refusal; capability. The arms are the two prompted system instructions (Safety SI, Constitution SI); the four corpus
fits of the main figures (Control ReLU steer, Control LoRA, Constitution ReLU steer, Constitution LoRA); and the six objects distilled
from a prompted teacher, a safety, a constitution and an irrelevant-instruction target, each as a ReLU steer and a LoRA
(Safety-SI-distilled, Constitution-SI-distilled and Irrelevant-SI-distilled ReLU steer and LoRA). Each eval carries one column per arm,
in the legend's order. Each point is the change against the unsteered model in percentage points, oriented so that up is better:
defense success on the static, multi-query and multi-turn attacks, benign compliance on over-refusal, accuracy or pass rate on
capability. Small points are the three fit seeds of a fitted object; a prompted instruction is one point per eval, the mean of its
three decode trials, as in the main figures. The large points after the separator are the category means: each seed's unweighted
mean over the panel's evaluations, averaged across seeds (the capability mean averages MMLU, GSM8K, BigCodeBench and BFCL). Bars are
95% intervals. The two prompted instructions and the four corpus fits are the main figures' own readings; the six distilled objects
are new here.

Every point is differenced against pooled unsteered runs by the main figures' rule for its evaluation: the pooled unsteered runs at the point's token cap on the static attacks, Best-of-75, PAIR and MMLU; on the multi-turn attacks, XSTest and OR-Bench-Hard the pooled
unsteered runs the main figures difference every arm against; on BigCodeBench and BFCL each seed's own unsteered run, on the same harness.
On GSM8K a point is differenced against the pooled unsteered runs filed at the same cap as its own, and the two pools are filed and
served differently: at 8,192 tokens, two lens-served unsteered runs on an older GSM8K instrument (2a954e3c), each merged with its own
16,384-token rung though filed at 8,192 (96.5% accuracy); at 16,384 tokens, six natively served runs on the distilled objects'
instrument (5ea36b25, 96.0%).

A BigCodeBench or BFCL completion cut at the cap counts as a failure. Of the static comparators, the seed-1 control and constitution corpus fits on Past tense, CodeAttack and ReNeLLM and the prompted instructions' first-trial runs on Past tense, CodeAttack, ReNeLLM and Pliny were generated in another serving setup, with the same prompts and seeds, and each
static attack's unsteered pool at 4,096 tokens includes one run of that setup. Of the goals, 74 of Best-of-75's 100 and 167 of the 259
the static attacks and PAIR read are inside the distilled objects' training pools (the same goals for all three targets); the
static and multi-query tables' notes give the mean changes recomputed on the goals outside the pools. Every judged point is MiniMax-M3's verdict: StrongREJECT at threshold 0.5 (a goal counts as broken when its best reply scores above it) and the over-refusal
labels (full refusal 1, partial ½, compliance 0) on XSTest's safe prompts and on OR-Bench-Hard, each at its ladder's final rung.
""")

code("9d4a6f20", r"""
# THE DISTILLATION-TARGET BATTERY: every family of the main figures on Qwen3.5-4B in their unconnected dot-plot form, twelve arms per
# eval - the two prompted instructions, the main figures' control and constitution corpus fits, and the six distilled objects. Drawn
# only when this render reads every judged eval from the battery's one judge (MiniMax-M3, the re-judge policy); under any other policy
# one RENDER POLICY line says so and nothing is drawn. The prompted instructions and the corpus fits are the main figures' own rows of
# `d`, copied unchanged on every eval; the distilled objects' runs come from the committed battery sidecar (a live render re-derives it
# from the live registry and refuses on any difference; a snapshot render, whose snapshot predates these runs, reads it alone, held to
# its contract), anchored by the main figures' own code on `op`'s unsteered rows and asserted to carry the anchor rule of each eval; the
# multi-turn panel's distilled objects, constitution corpus fits and prompted instructions are the figure above's upper row (st_d).
# Every run drawn, tabulated or pooled is held to its eval's one rubric before anything is drawn (on Best-of-75 the arms; its pool is
# Figure 4's). Readings not yet in the data are gap glyphs, named under the figure. Every frame the cells above read is only read here;
# what this cell makes is named bt_*.
bt_cats = R.SITARGET_BATTERY_CATEGORIES
bt_evals = tuple(e for c in bt_cats.values() for e in c[0])
bt_ok, bt_why = R.sitarget_render_ok(bt_evals, what="the distillation-target battery figure and its tables")
if not bt_ok:
    print(bt_why)
else:
    _bt_judge = R.sitarget_judge_problems(bt_evals)
    if _bt_judge:
        raise RuntimeError("the battery figure's judge pins disagree with the policy lookup: " + "; ".join(_bt_judge))
    # THE FINAL RENDER ACCEPTS A FINAL FILE AND THE COMMITTED PENDING-REGISTRATION STATE (every BigCodeBench and BFCL reading generated on
    # another harness, not yet in the registry: pending glyphs, ringed capability means, both named); any other pending list is
    # PROVISIONAL, which only a render flagged REGISTRY_NB_SITARGET_PROVISIONAL=1 draws
    bt_rows, bt_pending, _bt_file = R.sitarget_battery_load(R.SOURCE, allow_pending=R.SITARGET_BATTERY_PROVISIONAL)
    bt_state = R.sitarget_battery_state(_bt_file)
    if bt_state == R.SITARGET_STATE_PROVISIONAL:
        print(f"!! PROVISIONAL battery render (REGISTRY_NB_SITARGET_PROVISIONAL=1): {len(bt_pending)} reading(s) not yet in the data; "
              "not the final figure")
    elif bt_state == R.SITARGET_STATE_REGISTRATION:
        print(f"battery file state: {bt_state}: the distilled objects' {len(bt_pending)} "
              + " and ".join(R.EVAL_LABEL.get(e, e) for e in R.SITARGET_PENDING_REGISTRATION)
              + " readings were generated on another harness and are not yet in the registry (pending glyphs, named under the figure)")
    # CAPABILITY COMPLETENESS: every distilled object at every seed on MMLU, GSM8K, BigCodeBench and BFCL drawn over its item set, or
    # explicitly pending, the pending set named
    _bt_cap_lines, _bt_cap_probs = R.sitarget_battery_capability_completeness(_bt_file)
    if _bt_cap_probs:
        raise R.SitargetSidecarMismatch(f"{len(_bt_cap_probs)} capability reading(s) neither drawn whole nor pending: "
                                        + "; ".join(_bt_cap_probs[:10]))
    for _l in _bt_cap_lines:
        print(_l)
    bt_new = R.sitarget_battery_anchored(op, bt_rows, POOL_Q, d)
    bt_cmp = R.sitarget_battery_comparators(d)
    bt_d = R.sitarget_scoring_rules(R.concat_plain(*[f for f in (bt_cmp, bt_new, st_d) if len(f)]))
    bt_unsteered = R.sitarget_scoring_rules(pd.concat([op[op.object_label.eq("none")], POOL_Q], ignore_index=True))
    # ONE RUBRIC PER EVAL: every drawn, tabulated and pooled run on its eval's one judge or scoring rule (on Best-of-75 the arms), or
    # nothing is drawn
    _bt_probs = R.sitarget_battery_rubric_problems(bt_d, bt_unsteered)
    if _bt_probs:
        raise R.SitargetRubricMismatch(f"{len(_bt_probs)} run(s) off their eval's one rubric: " + "; ".join(_bt_probs[:10]), _bt_probs)
    for _l in R.sitarget_battery_rubric_lines(bt_d):
        print(_l)
    bt_listing = R.sitarget_run_listing(bt_d, bt_unsteered, figure=R.SITARGET_BATTERY_FIGURE)
    print("instruments and caps: " + R.sitarget_disclosure(bt_listing, what="evaluation"))
    # THE STATIC COMPARATORS, listed from the frame this render draws: the tables' (the constitution corpus fits and the prompted
    # instructions) and the figure's (the control corpus fits too)
    bt_static = R.sitarget_static_comparators(bt_d)
    _bt_static_all = R.sitarget_static_comparators(bt_d, corpora=R.SITARGET_CONTROL_SLOTS + R.SITARGET_REFERENCE_SLOTS)
    print("static comparators: " + R.sitarget_static_disclosure(_bt_static_all))
    # WHAT THE CAPTION STATES, HELD TO THIS RENDER: the static comparators generated in another serving setup and the static pools' run
    # of that setup, and the two GSM8K pools by filing and serving
    _bt_caption = (R.sitarget_static_caption_problems(_bt_static_all, bt_listing)
                   + R.sitarget_gsm8k_caption_problems(R.sitarget_gsm8k_pools(bt_d, bt_unsteered)))
    if _bt_caption:
        raise RuntimeError("the battery caption does not hold for this render: " + "; ".join(_bt_caption))
    bt_cm = R.category_means(bt_d, categories=bt_cats)
    bt_gaps = R.sitarget_battery_gaps(bt_d, bt_pending)
    R.sitarget_battery_figure(bt_d, bt_cm, bt_gaps, "figA_qwen_si_target_battery"); plt.show()
    _bt_missing = R.sitarget_battery_not_drawn(bt_gaps)
    if _bt_missing:                                      # an arm with no reading of an eval at all is drawn as the gap glyph
        print("!! NOT DRAWN: " + "; ".join(_bt_missing))
    _bt_partial = R.sitarget_battery_partial_means(bt_d, bt_cm, bt_pending)
    for _l in R.sitarget_battery_partial_notes(_bt_partial):
        print("partial means (ringed): " + _l)
    # THE TABLES print the prompted instructions, the distilled objects and the constitution corpus fits (the control corpus fits are the
    # main figures' own, tabulated there); the GSM8K pools their rows are differenced against, named by filing and serving
    _bt_tab = bt_d[~bt_d.object_label.isin(R.SITARGET_CONTROL_SLOTS)]
    _bt_g8k_pools = R.sitarget_gsm8k_pool_sentence(_bt_tab, bt_unsteered)
    print("GSM8K anchors: " + _bt_g8k_pools)
    # THE TRAINING POOLS: how many goals they hold, then every drawn arm's mean change on the static and multi-query attacks on all goals
    # and outside the pools, and the irrelevant-instruction objects' changes, each computed in this render (the split reproduces every
    # drawn change, and its all-goals means are the figure's mean marks)
    _bt_marks = R.pool_seeds(bt_cm)
    _bt_pool = R.sitarget_pool_overlap_sentence(_bt_file, st_goals)         # the goals the distillation training pools hold
    print("training pools: " + _bt_pool)
    _bt_outside = R.sitarget_battery_outside_pool_sentences(R.sitarget_battery_pool_split(bt_d, bt_unsteered, _bt_file, st_goals),
                                                           _bt_marks, _bt_file, st_goals)
    _bt_out_text = {c: f" {_bt_outside['lead']}, the mean change in defense success on all goals then outside the pools: {_bt_outside[c]}."
                    for c in R.SITARGET_POOL_SPLIT_CATEGORIES}
    print(f"outside the pools: {_bt_outside['lead']}, the mean change in defense success on all goals then outside the pools: "
          f"{_bt_outside['multi-query']}; {_bt_outside['static']}.")
    if "control" in _bt_outside:
        print("the irrelevant-instruction objects: " + _bt_outside["control"])
    _bt_cat_of = {e: c for c, (evs, _s, _l) in bt_cats.items() for e in evs}
    _bt_dash = {_bt_cat_of[e] for e in bt_gaps[~bt_gaps.object_label.isin(R.SITARGET_CONTROL_SLOTS)].eval_id}
    _bt_note = {c: (" " + R.sitarget_static_disclosure(bt_static, ids=False) + R.sitarget_static_anchor_disclosure(bt_listing)
                    if c == "static" else "")
                   + (" " + _bt_g8k_pools if c == "capability" and _bt_g8k_pools else "")
                   + (" " + _bt_pool if c in R.SITARGET_POOL_SPLIT_CATEGORIES and _bt_pool else "")
                   + (_bt_out_text[c] + (" " + _bt_outside["control"] if "control" in _bt_outside else "")
                      if c in _bt_out_text else "")
                   + R.sitarget_battery_partial_table_note(_bt_partial, c, arms=set(_bt_tab.object_label))
                   + (" A dash marks a reading that is not yet available." if c in _bt_dash else "") for c in bt_cats}
    for _cat in bt_cats:
        print(R.sitarget_battery_save_table(R.sitarget_battery_table(
            _bt_tab, _cat, f"{R.SITARGET_BATTERY_TITLE[_cat]} for Qwen3.5-4B by distillation target, beside the constitution corpus fits.",
            f"tab:qwen-si-target-battery-seeds-{_cat}", note_extra=_bt_note.get(_cat, "")), _cat))
    print(f"the distilled objects: {bt_new.run_id.nunique()} battery runs of {bt_new.object_id.nunique()} objects from "
          f"{R.display_path(R.SITARGET_BATTERY_SIDECAR)} ("
          + ("re-derived from the live registry and checked field by field and payload by payload" if R.SOURCE == "live" else
             "read from the file alone and held to its contract: the committed snapshot predates these runs")
          + f"), and the multi-turn figure's {st_new.run_id.nunique()} runs; {len(bt_pending)} reading(s) not yet in the data (file "
          f"state: {bt_state}); StrongREJECT threshold {R.SR_THRESHOLD:g}")
    print("the mean points (one per arm and category: the seeds averaged; n_seeds says how many):")
    R.show(_bt_marks[_bt_marks.object_label.isin(R.SITARGET_BATTERY_ARMS)].drop(columns="model_id").round(2), n=60)
""")

md("be6069de", r"""
### Appendix — who wrote the corpus: Opus-written against Qwen-written 1M-token constitution LoRAs (Qwen; Figure 17)

Rank-4 o_proj LoRAs fit on Qwen3.5-4B-Base on 1M tokens of the constitution corpus at an effective batch of 8 and served on
Qwen3.5-4B, three seeds each, fit on the corpus as Opus 5 wrote it (circle) and on the corpus Qwen3.8-Max wrote through the same
pipeline (diamond); these are not the paper's constitution LoRA. The panels are Figure 11's: static attacks, multi-query (Best-of-75,
PAIR), multi-turn (Crescendo, ActorAttack), over-refusal (XSTest, OR-Bench-Hard) and capability (MMLU, GSM8K). Each mark is one seed's
change against the unsteered model with its 95% interval, positive is better (Δ defense success = −Δ attack success); a line joins the
two corpora's fits of the same seed, and the mean column averages each corpus over the panel's evaluations within seed, then over the
seeds. The anchors are Figure 11's pooled unsteered references for the same evaluation and cap (on XSTest and OR-Bench-Hard, the
over-refusal reference Figure 11 uses for that seed; the Best-of-75 pool mixes the two StrongREJECT rubric versions, as there); a
static reading on a rung no unsteered run of its evaluation was read at takes that evaluation's pooled runs at another cap, as in
Figure 11, and the table under the figure marks it. MiniMax-M3 judges every judged point on Figure 11's rubrics (StrongREJECT, its
second version on Best-of-75; refusal labels on XSTest and OR-Bench-Hard); MMLU and GSM8K are scored by rule, GSM8K over the items its
ladder scored, with the reading that counts an answer the cap cut as wrong printed under the figure. The one pre-fixed comparison is
the static panel's Qwen − Opus over the three matched seed pairs, with its t interval on two degrees of freedom; the other panels'
differences are descriptive.
""")

code("431232a9", r"""
# THE GENERATOR COMPARISON ON FIGURE 11'S FIVE PANELS: 1M-token constitution LoRAs fit on the Opus-written and on the Qwen3.8-Max-written
# corpus, three seeds each, differenced against the control-pool appendix's pooled unsteered anchors (ctl_op and ctl_pool, the anchor rule
# held to ctl_md), once per corpus (appendix_generator.anchor). The runs sit outside the paper's selection (registry rulings), so they
# are read from the committed file data/appendix_generator_rows.json; a live render derives it again from the registry and stops on any
# difference (appendix_generator.load). ONE JUDGE PER EVALUATION WHATEVER THE RENDER'S POLICY: the control-pool appendix's MiniMax-M3
# judges, registered lane-fixed before the figure is written. Nothing here reads or changes another figure's frames.
import appendix_generator as AG
R.register_lane_fixed(AG.LANE_FIGURES, AG.lane_fixed_text())
gen_data = AG.load(R.SOURCE)
gen_d = AG.anchor(AG.lane_frame(gen_data), ctl_op, ctl_pool, ctl_md)
gen_rows = AG.tidy_rows(gen_d, gen_data)
AG.draw(gen_rows, "figA_qwen_generator_ablation"); plt.show()
if (_gen_missing := AG.not_drawn(gen_rows)):
    print("!! NOT DRAWN (gap glyph in its place): " + "; ".join(_gen_missing))
print(AG.summary_text(gen_data, gen_d, gen_rows, ctl_md))
print(AG.provenance_line(gen_data, R.SOURCE))
print("per seed, the marks (the reading, the anchor, the change and its 95% interval in each panel's unit, + is better; the cap read; "
      "anchor cap: same, other = the eval's pooled unsteered runs at another cap as Figure 11's points on that rung; on over-refusal the "
      "anchor Figure 11's fitted rows of the seed take, bound = the one run the registry binds to them, pooled = the runs pooled for them):")
print(AG.marks_table(gen_rows).to_string(index=False))
""")

md("347c3abd", r"""
## Appendix — the training corpora, control directions and capability losses

Each analysis below reads one committed file under `data/`, `appendix_<name>.json`, written by `data/build_appendix_<name>.py` from the
analysis's own outputs. A file holds counts or aggregates only, pins the registry objects, runs and judgments it was read from where
there are any, and carries a provenance block naming its source files (each by sha256) and its judge, so every cell below renders from
committed data. Each cell says what it checks against the registry source this notebook reads, and stops before the zip cell when its
file breaks its contract or disagrees with that source; rebuilding the file is the remedy. They draw Figures 9 and 10 and print Tables 3
and 6 of the paper.

The comparison figure among them, the sign decomposition, has the control-pool stopping comparison's form, drawn by
`appendix_common.paired_comparison_figure` (beside this notebook) for any objects, evaluations and two or three conditions: one group
per evaluation, then a bold mean group after the dark separator; in each group the objects in the paper's colours, each seed's readings
under the conditions told apart by shape (circle, diamond, square) and joined by a thin line in the object's colour, every mark with its
95% interval, y the change from the unsteered model with positive better. A mean mark averages each condition within seed over the
category's evaluations, then over seeds (`registry_nb.category_means` and `pool_seeds`, the rule every figure here uses), and is drawn
only when it covers every seed and evaluation its object has; anything left out is named under the figure. The other figures are
compositions and geometry, drawn in the paper's fonts and palette. The paper prints the corpus audit and the capability errors as
tables only and quotes the forward passes' numbers in its text, so those figures are drawn and saved here but left out of the zip of the
paper's figures.
""")

code("f60c0561", r"""
# THE SHARED FORM OF THE APPENDIX COMPARISON FIGURES. appendix_common draws them from tidy rows (object, seed, category, eval, condition,
# delta, ci_lo, ci_hi), in the palette, labels and fonts it reads from registry_nb, and saves png, pdf and svg into R.FIGS, where the zip
# cell picks up every name listed in PAPER_FIGURES.
import appendix_common as AC
print(f"appendix_common: condition shapes {', '.join(AC.CONDITION_MARKERS)}; mean rule {AC.MEAN_RULES[0]!r} "
      f"(a condition's mean is drawn when it covers every seed and evaluation); figures to {R.display_path(R.FIGS)}")
""")

md("ba7e3dcf", r"""
### Appendix — what the training corpora contain

The corpus audit: its figure (`figA_corpus_audit`, drawn here only) and Table 3 (`tab:corpus-audit`), with its fuller form.
""")

md("3a7c91e4", r"""
#### The corpus audit

Every record of the two training corpora (36,570: constitution and control, advice and stories, training and held-out sides) was read
for three claims: **C1**, templates and evaluation prompts (a deterministic eval-text half, positive at a 13-word run shared with a
scanned evaluation prompt set or static attack template, or 70% coverage of a goal's words; and a labelled jailbreak half, a request with
adversarial framing or a record holding jailbreak text); **C2**, requests of the kinds HarmBench, JailbreakBench or StrongREJECT test,
or that seek serious harm; and **C3**, requests to help with a wrongful act short of serious harm, present in the constitution advice by
design. The labels are by Claude Opus 5.5 acting as a corpus labeller, not an evaluation judge (the evaluation judges are non-Claude);
the eval-text half and the overlap maxima are a deterministic scan. Records the labeller's own safety classifier declined, and contested,
partially read or unresolved ones, are **not cleared**: neither zeros nor positives, so every positive count is a lower bound.

The cell reads the committed sidecar `data/appendix_corpus_audit.json`: aggregate counts only, each under the audit summary's own key,
with no record text and no record id. Its provenance names the audit's summary and scan files by sha256, and the reader refuses it unless
the counts agree with one another (each share with its counts, the parts with the totals, the reread bound with its counts). The figure
draws, per claim and corpus part, the positive records in the corpus's colour (constitution green, control blue) and the two not-cleared
states in grey; negatives are not drawn. Under it, Table 3 (tab:corpus-audit: records, depicted jailbreak attempts, harmful requests,
requests for a lesser wrongful act and records the labeller declined, per corpus part), then the fuller table of every claim and state.
""")

code("c5e0b2d8", r"""
# THE CORPUS AUDIT. appendix_corpus_audit reads the sidecar (aggregate counts copied from the audit summary under their own keys, checked
# for the stamp, the provenance, the absence of any record id and the counts' agreement), draws figA_corpus_audit with appendix_common's
# save helper, and prints Table 3 (tab:corpus-audit), its fuller form and the overlap scan's maxima from the same numbers.
import appendix_corpus_audit as CA
corpus_audit = CA.load_corpus_audit()
print(CA.provenance_line(corpus_audit))
CA.corpus_audit_figure(corpus_audit, "figA_corpus_audit"); plt.show()
print(CA.overlap_line(corpus_audit))
print(CA.table_corpus_audit(corpus_audit))
print("the fuller form, every claim and state the audit records (drawn here only):")
print(CA.table_corpus_audit_full(corpus_audit))
R.show(CA.claim_states(corpus_audit)[["claim", "part", "records", "positive", "contested", "partial", "unresolved", "declined_on_input",
                                      "share_read_pct"]], n=12)
""")

md("2802219e", r"""
### Appendix — control directions

Figure 9: the sign decomposition (`figA_control_evenodd`, top) and the weights against a refusal geometry (`figA_refusal_enrichment`,
bottom); Figure 10: the overlap of the two objects (`figA_cK_overlap`); then the forward passes whose numbers the text quotes.
""")

md("f46e655e", r"""
#### Is the control object a refusal direction? The sign decomposition (Qwen; Figure 9, top)

The paper reads the control object $\theta_K$ as register and format, which subtraction removes. A refusal direction would instead act
with its sign: reversed, it would reverse its effect. So each control object was also served negated ($\alpha=-1$: the ReLU steer's write
directions or the LoRA's $B$ matrices times $-1$, every other tensor unchanged), on the eight evaluations that have such runs: the four
static attacks, Crescendo and ActorAttack, XSTest and OR-Bench-Hard. Each mark is $D(X)$, the change from the unsteered model in
percentage points with up better (defense $=100-$ ASR; benign compliance $=100-$ over-refusal), with its 95% interval: a circle for
$+\theta$ ($\alpha=+1$), a diamond for $-\theta$ ($\alpha=-1$), the two marks of one seed joined by a thin line in the object's colour.
The line's midpoint is that seed's sign-independent part $\tfrac12[D(+\theta)+D(-\theta)]$ and half its rise the sign-dependent part
$\tfrac12[D(+\theta)-D(-\theta)]$. The Refusal steer (Arditi et al., 2024) at $c=+0.5$ and $c=-0.5$, one run per sign, is the positive
control: a refusal direction, whose effect flips with its sign. The mean column averages each sign over the category's evaluations within
seed, then over seeds.

Both signs of a seed are differenced against one unsteered anchor per evaluation (the $+1$ object's main-figure anchor; the Refusal
steer's is the union of those anchors), which cancels from the sign-dependent part and counts once in the sign-independent part. The
intervals are the main figures': a mark's from its run and its anchor's runs, clustered by goal or item, and a mean's from the seed
pool (the seeds' runs clustered across the seeds, the spread between seeds, the anchor once), all read from the sidecar's per-unit
verdicts. The table under the figure splits each category mean into the two parts by the same pool; the sign-dependent part's two
signs enter as independent samples. All
verdicts are MiniMax-M3's (StrongREJECT at 0.5 per goal; over-refusal labels per item); an anchor run with no MiniMax-M3 verdict is left
out of its anchor, never filled with another judge's, and the cell names it. The control ReLU steer's seed 1 has no $\alpha=-1$ run on
the two benign evaluations, so that seed is left out of both signs there. The run selection is the paper frame at registry head 456957,
with the two control objects' seed-1 Pliny runs at $\alpha=+1$ regenerated after it; a source older than those two reads them from the
file, and the cell names them.
""")

code("5f821770", r"""
# THE SIGN DECOMPOSITION OF THE CONTROL OBJECTS. appendix_geometry_evenodd reads the committed sidecar (one row per
# reading: object, seed, evaluation and sign, the change from the unsteered model with its standard error split into the steered run's and
# the anchor's, and the run, object and MiniMax-M3 judgment it pins), refuses it by name when it breaks its contract, and binds every pin
# to this render's source: objects and runs always, judgments wherever the source holds them or should (a live registry; a frame that
# holds them, or whose head is at or past the last of them, max_judgment_ingest_seq; a frame cut before the MiniMax-M3 re-judge holds
# none of them and leaves them to the file). A pin that does not bind
# stops the render here, before the zip cell; rebuilding the sidecar is the remedy. ONE JUDGE WHATEVER THE RENDER'S POLICY: MiniMax-M3,
# registered lane-fixed before the figure and its table are written. THE INTERVALS ARE THE NOTEBOOK'S: the sidecar's per-unit payloads
# are this render's unit source for these runs (GE.register_payloads; each held to its run's reading), so a mark's interval is its run's
# own with its anchor's runs item-clustered and a mean mark is registry_nb's seed pool, as on the main figures.
import appendix_geometry_evenodd as GE
ge = GE.load_geometry_evenodd()
_ge_runs = R.query(GE.runs_sql(ge), "geometry_evenodd_runs")
_ge_judg = R.query(GE.judgments_sql(ge), "geometry_evenodd_judgments")
_ge_head, _ge_jseq = _ge_runs.attrs.get("head"), ge["provenance"]["verdicts"]["max_judgment_ingest_seq"]
_ge_jcheck = R.SOURCE == "live" or len(_ge_judg) > 0 or (_ge_head is not None and int(_ge_head) >= int(_ge_jseq))
_ge_bind = GE.bind_problems(ge, R.query(R.OBJECTS_SQL, "objects"), _ge_runs, _ge_judg if _ge_jcheck else None)
if _ge_bind:
    raise ValueError("data/appendix_geometry_evenodd.json does not bind to the registry:\n  " + "\n  ".join(_ge_bind))
_ge_file = GE.runs_read_from_file(ge, _ge_runs)
if _ge_file:
    print(f"{len(_ge_file)} run(s) filed after this source (head {_ge_runs.attrs.get('head')}), read from the file: " + "; ".join(_ge_file))
_ge_p = ge["provenance"]
_ge_anchor_runs = sorted({x for a in ge["anchors"].values() for x in a["runs"]})
print(f"sign-decomposition sidecar: {len(ge['rows'])} readings, {len(ge['objects'])} objects and {len(_ge_anchor_runs)} unsteered runs bound "
      f"to the {R.SOURCE} registry; {ge['judge']['judgments_pinned']} MiniMax-M3 judgments pinned"
      + (" and checked current" if _ge_jcheck else " (read from the file: this source predates them)")
      + f" · run selection at head {_ge_p['run_selection']['registry_head']} · built {_ge_p['generated_utc']} at {(_ge_p.get('git_commit') or '?')[:9]}")
print(f"intervals from the sidecar's per-unit payloads: {GE.register_payloads(ge)} runs, "
      f"{sum(len(x['v']) for x in ge['payloads'].values())} units on {len(ge['item_sets'])} item sets")
R.register_lane_fixed(GE.LANE_FIGURES, GE.lane_fixed_text(ge))
GE.figure_geometry_evenodd(ge, GE.FIGURE); plt.show()
_ge_missing = GE.not_drawn(ge)
if _ge_missing:
    print("NOT DRAWN: " + "; ".join(_ge_missing))
_ge_left = sorted({(a["eval"], x) for a in ge["anchors"].values() for x in a["runs_left_out"]})
if _ge_left:
    print(f"{len(_ge_left)} unsteered run(s) with no MiniMax-M3 verdict, left out of their anchors: "
          + "; ".join(f"{R.EVAL_LABEL[e]} {x}" for e, x in _ge_left))
_ge_mv = GE.coherent_moves(ge)
print(f"without units that have a degenerate reply, a per-seed mark moves by at most {_ge_mv['marks']:.1f} pp ({_ge_mv['marks_where']}) "
      f"and a mean mark by at most {_ge_mv['means']:.1f} pp ({_ge_mv['means_where']})")
print(GE.table_geometry_evenodd(ge))
_ge_cp = GE.category_parts(ge)
print("how each part's interval was obtained: " + "; ".join(sorted({f"{q}: {h}" for q in GE.PARTS for h in _ge_cp[f"{q}_how"]})))
print("per evaluation, seeds pooled (the numbers the appendix text quotes); `anchor` is the unsteered attack-success or over-refusal rate, %:")
R.show(GE.eval_parts(ge)[["object", "eval", "seeds", "anchor", "Dp", "Dp_ci", "Dn", "Dn_ci", "even", "even_ci", "odd", "odd_ci", "odd_seeds"]]
       .round(1), n=24)
""")



md("ee8bdbc1", r"""
#### The objects against a multi-layer refusal geometry (Qwen; Figure 9, bottom)

The weights of the paper's base-to-instruct fits, seeds 1-3, against 279 refusal directions: the harmful-minus-harmless mean activations
of Arditi et al. (2024) at every layer but the last and at the 9 post-instruction template positions. For the ReLU steers' read (gate)
and write vectors and the LoRAs' output space (weighted by $\Delta W$), the share of squared norm that lies in each layer's 9-position
refusal span (left) and on the first principal component of all 279 directions (right), divided by the share a random direction puts
there, averaged over layers (1 = random; log scale). Large marks are the mean of the three seeds of $\theta_C$ (the constitution object)
and $\theta_K$ (the control object), small marks single seeds; the safety positive controls are one ReLU steer and one LoRA distilled
from a safety-refusal system prompt on the instruct checkpoint, a different recipe. Grey: the random-direction 95% range.
""")

code("0c6192ec", r"""
# THE REFUSAL ENRICHMENT. appendix_geometry_forward reads the committed sidecar (aggregates of the weight analyses and of the forward-pass
# job geom-fwd-0074823, per seed and layer, every number bound to its objects by sha256), refuses it by name when it breaks its contract,
# and binds its 28 object pins to this render's registry source (label, fit checkpoint, seed, and the derived objects' parents). A pin
# that does not bind stops the render here, before the zip cell; rebuilding the sidecar is the remedy. The figure is one panel: it
# prints under the sign decomposition in the paper.
import appendix_geometry_forward as GF
gf = GF.load_geometry_forward()
_gf_bind = GF.registry_check(gf, R.query(R.OBJECTS_SQL, "objects"))
if not _gf_bind.known.all():
    raise ValueError("data/appendix_geometry_forward.json does not bind to the registry:\n  " + "\n  ".join(
        f"{r.object_id[:12]} ({r.family} {r.role} {r.seed or '-'}): {r.mismatch}" for r in _gf_bind[~_gf_bind.known].itertuples()))
print(GF.provenance_line(gf) + f" · all {len(_gf_bind)} object pins bound to the {R.SOURCE} registry")
GF.refusal_enrichment_figure(gf, GF.FIG_ENRICHMENT); plt.show()
print("each seed's enrichment over a random direction, the seed mean, the Haar p (largest over seeds) and the grey band:")
R.show(GF.table_refusal_enrichment(gf).round(3), n=18)
""")

md("6b1e4f90", r"""
#### How much the constitution and control objects overlap, layer by layer (Qwen; Figure 10)

The same objects' weights against each other. **(a)** LoRA: the share of one object's update $\|\Delta W_l\|_F^2$ that lies inside
another object's rank-4 output space (span of $B_l$, where it writes; left) or input space (span of $A_l$, where it reads; right), per LoRA
layer, log scale. Markers and bars are the mean and range over the 9 seed pairs of $\theta_C$ and $\theta_K$ in each direction, the 6
ordered seed pairs within one object (the seed ceiling) and the 3 seeds of $\theta_C$ or $\theta_K$ inside the safety-refusal LoRA (an
unrelated reference); filled markers and solid lines are the share of $\theta_C$'s update, open markers and dashed lines $\theta_K$'s.
Grey: per layer, the 95% range when the other subspace is a random 4-dimensional one (2,000 draws per object). **(b)** ReLU steers: the
cosine between two objects' read (gate) vectors (left) and write vectors (right) at the same layer; lines and shading are the mean and
range over the same seed pairs (3 within one object) and the safety-refusal ReLU steer against each object's 3 seeds. Grey:
$\pm 2/\sqrt{d}$, two standard deviations of the cosine between random vectors. Layer 12 of panel (b) is the layer-12 cosine data the
notebook-only cosine figure draws, and the cell checks that it is.
""")

code("e2c85a17", r"""
# THE OVERLAP OF THETA_C AND THETA_K. appendix_geometry_overlap reads the committed sidecar (per pair and layer, the LoRA overlaps in the
# output and input spaces with their random-subspace null, and the ReLU read and write cosines; every object pinned by sha256), refuses
# it by name when it breaks its contract or when its layer-12 cosines are not those of data/relu_arditi_cosines.json, and binds its 14
# object pins to this render's registry source. A pin that does not bind stops the render here, before the zip cell.
import appendix_geometry_overlap as GO
go = GO.load_geometry_overlap()
_go_bind = GO.registry_check(go, R.query(R.OBJECTS_SQL, "objects"))
if not _go_bind.known.all():
    raise ValueError("data/appendix_geometry_overlap.json does not bind to the registry:\n  " + "\n  ".join(
        f"{r.object_id[:12]} ({r.family} {r.role} {r.seed or '-'}): {r.mismatch}" for r in _go_bind[~_go_bind.known].itertuples()))
print(GO.provenance_line(go) + f" · all {len(_go_bind)} object pins bound to the {R.SOURCE} registry")
GO.overlap_figure(go, GO.FIGURE); plt.show()
print("per series: the per-layer pair mean's range over layers (with the layers), its mean over layers and the single values' range; "
      "(a) in % of ||dW||^2, (b) as cosines; SinC and SinK, the safety LoRA's update in theta_C's or theta_K's span, are not drawn:")
R.show(GO.table_overlap(go).round(3), n=26)
print("the numbers the appendix text quotes from Figure 10, recomputed from the sidecar:")
R.show(GO.text_numbers(go), n=12)
""")
md("53f2d0e0", r"""
#### Distance from the unsteered model, a refusal readout, and write cancellation (Qwen; drawn here only)

The paper does not print this figure; its text quotes the numbers printed under it (the KL of the difference objects against
$\theta_C$ above all). Judge-free forward passes of Qwen3.5-4B, teacher-forced on fixed text: attack prompts (800) and benign prompts (400) with the unsteered
model's own replies, and the refusal-direction fit's validation prompts (32 harmful, 32 harmless). The objects are the paper's
base-to-instruct fits, seeds 1-3: $+\theta_K$ (the control object), $-\theta_K$ (its write negated: the ReLU steer's write directions or
the LoRA's $B$ times $-1$, applied as stored), $\theta_C$ (the constitution object) and $\theta_C-\theta_K$ (the difference object), as a
ReLU steer (circles) and a LoRA (squares); the Refusal steer (Arditi et al., 2024) at $c=+0.5$ and $c=-0.5$ is the positive control. A
large mark is the mean of the three seeds, a small one a single seed; an open mark has its write negated.

**(a)** Mean per-token $\mathrm{KL}(p_X\,\|\,p_{\mathrm{unsteered}})$ over prompt and reply positions after the first. **(b)** The change
in the projection of the post-instruction residual state on the unit refusal direction of the same layer (the harmful-minus-harmless mean
difference, refit at the block output), at the output of block 28 and the last template token, where the appendix text reads it; the
Refusal steer with its 95% interval over prompts, and each title the unsteered model's mean projection. This replaces the interim
figure's first-reply-token refusal score (log-odds of "I" or "As"), which sits near its floor on harmless and near its ceiling on harmful
prompts and also rises when a reply opens with "I'd be happy". **(c)** Per layer, on the difference object's trajectory and on the
unsteered one (attack and benign texts): the correlation of the two ReLU gates over tokens, read on the same residual state, and
$\kappa_C$, the share of $\theta_C$'s realized write that $\theta_K$'s write removes by projection (the ReLU steer at every layer, the LoRA
at its 8); lines and squares pool the seeds, bands and bars are the seed range.
""")

code("a77fad73", r"""
# THE FORWARD PASSES. The same sidecar as the refusal enrichment above (gf, its object pins already bound): the forward-pass job
# geom-fwd-0074823's aggregates per seed and layer. Then every number the appendix text quotes from this sidecar, recomputed from it (the
# KL of the difference objects against theta_C, the refusal-projection drops at block 28's last template token and the sign-dependent
# parts over layers 15-30, the gate correlation and kappa_C, the enrichments, the signed cosines, the LoRA Frobenius cosines).
GF.displacement_figure(gf, GF.FIG_DISPLACEMENT); plt.show()
print("panels (a) and (b), every seed and the mean of the three; (b) at block 28's output, last template token, with the Refusal steer's SE over prompts:")
R.show(GF.table_displacement(gf).round(3), n=36)
print("the numbers the appendix text quotes from this sidecar, recomputed from it:")
R.show(GF.text_numbers(gf), n=40)
""")

md("da341da1", r"""
### Appendix — error analysis of capability losses

The classes of the new BFCL and BigCodeBench failures: Table 6 (`tab:caperrors`, BFCL), its BigCodeBench sibling (`tab:caperrors-bcb`)
and two composition figures drawn here only (`figA_qwen_capability_errors`, `figA_models_capability_errors`).
""")

md("037dbf1e", r"""
#### What the new BFCL and BigCodeBench failures are

The fitted objects lower BFCL and BigCodeBench pass rates (Figure 6 and the all-model capability figure). Here every item a steered run
fails but its seed-matched unsteered run solves (a **new failure**) is put in one class by deterministic rules over the stored response,
the parsed tool calls, the extracted program and the test output. **No LLM judge is involved**: BFCL is scored by its abstract-syntax-tree
checker and BigCodeBench by its unit tests (the registry's per-item `acc`). The runs are exactly those of Figure 6 and the all-model
capability figure. Qwen3.5-4B pools the three seeds of each fitted object, each against its own unsteered run; the other models have one
run per object. Each bar gives the classes as shares of the new failures (the share printed in any segment of 18% or more); beside it,
the headline shares (BFCL: no call, and format = malformed calls plus calls run to the 1,024-token cap; BigCodeBench: format, and refusal
or objection to the task), the change in pass rate (pp) and the number of new failures. The class colours are the analysis's own
six-slot palette, not the object colours.

The counts come from the committed sidecar `data/appendix_caperrors.json` (per run: new failures by class, gains, BFCL calling counts;
per pool: the analysis's 95% item-bootstrap intervals, B = 4000, which leave out seed-to-seed variance). The cell below recomputes every
run's correct count, new failures and gains from this source's per-item results, and its object id from the run table, and stops on
any difference before drawing.
""")

code("b2d18d18", r"""
# THE CAPABILITY ERROR ANALYSIS (app:caperrors): a judge-free sidecar bound to the registry's per-item results, two composition figures
# and the two tables, whose first rows follow the notebook's rule (registry_nb.seed_average). A sidecar that breaks its contract, or a run whose k, n, new failures, gains or object id differ from this source,
# stops the cell (CE.load_caperrors / CE.assert_bound) rather than draw.
import appendix_caperrors as CE
_ce = CE.load_caperrors()
_ce_check = CE.item_check(_ce, R.query(CE.item_sql(_ce), "caperrors_items"), R.query(CE.object_sql(_ce), "caperrors_objects"))
CE.assert_bound(_ce_check)
print(CE.binding_line(_ce, _ce_check))
CE.qwen_figure(_ce, "figA_qwen_capability_errors"); plt.show()
CE.models_figure(_ce, "figA_models_capability_errors"); plt.show()
print(CE.table_caperrors(_ce))
print(CE.NOTE_BFCL)
print(CE.table_caperrors_bcb(_ce))
_ce_comp = CE.composition_frame(_ce)
R.show(_ce_comp[_ce_comp.model_id == R.MODEL_ID].drop(columns=["slot", "family"]).round(2), n=22,
       caption="Qwen3.5-4B bars, BFCL then BigCodeBench (classes in % of new failures)")
R.show(CE.seed_frame(_ce, "bfcl").query("model_id == @R.MODEL_ID").round(2), n=21,
       caption="Qwen3.5-4B BFCL per seed: the seed-to-seed spread the pooled intervals leave out")
""")

code("b7d4e9f2", r"""
# ONE-CLICK ZIP OF THE PAPER'S FIGURES, rebuilt on every render: the named figures in every format the
# notebook wrote (png, pdf, svg), plus a manifest naming the render they came from. It lands in figures/ beside them.
import zipfile, os, time
PAPER_FIGURES = ["fig3_single_turn_padding", "fig4_multi_turn", "fig5_misalignment", "fig6_capabilities", "fig7_frontier",
                 "figA_transfer_models", "figA_control_evenodd", "figA_refusal_enrichment", "figA_cK_overlap",
                 "figA_qwen_ctlstop_rule_comparison", "figA_jailbreaks_models", "figA_bon_curves_models", "figA_misalignment_models",
                 "figA_capabilities_models", "figA_judge_comparison_a", "figA_judge_comparison_b", "figA_qwen_generator_ablation",
                 "figA_qwen_si_target_battery", "figA_qwen_si_target_multi_turn"]
# the paper's figures in its order: Figures 3 to 7, then the appendix's Figures 8 to 16 (Figure 9 is two files, the sign decomposition
# above the refusal enrichment; Figure 16 two, the judge comparison's parts), 17, the generator comparison, and the two
# distillation-target figures, 18 the battery and 19 the multi-turn comparison. Drawn and saved into figures/ but not zipped, because
# the paper does not print them: the layer-12 cosines, the alpha = -1 figure (its table is the paper's), the three control-pool mirrors,
# the forward-pass figure (its numbers are quoted) and the corpus audit and capability error figures (the paper prints their tables).
os.makedirs(R.FIGS, exist_ok=True)          # a render that drew no figure (a partial frame) still writes its manifest and records
# figures/ keeps only what this render drew: a figure or table file an earlier render wrote and this one did not is removed first
_stale = R.unsave_stale()
if _stale:
    print(f"figures/: {len(_stale)} files of an earlier render removed (this render did not draw them)")
_zip = os.path.join(R.FIGS, "paper_figures.zip")
_have, _miss = [], []
with zipfile.ZipFile(_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for name in PAPER_FIGURES:
        found = False
        for ext in ("png", "pdf", "svg"):
            f = os.path.join(R.FIGS, f"{name}.{ext}")
            if os.path.exists(f):
                z.write(f, arcname=f"{name}.{ext}"); found = True
        for t in sorted(x for x in os.listdir(R.FIGS) if x.startswith((f"{name}_seeds", f"{name}_table")) and x.endswith(".tex")):
            z.write(os.path.join(R.FIGS, t), arcname=t)   # its per-seed tables and its paper table, where it writes any
        (_have if found else _miss).append(name)
    z.writestr("MANIFEST.txt", f"registry_qwen_figures -- {len(_have)} figures, registry head {HEAD}, built {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
               + "\n".join(_have) + ("\n\nmissing at this render: " + ", ".join(_miss) if _miss else "") + "\n")
print(f"paper_figures.zip: {len(_have)} figures × (png, pdf, svg), {os.path.getsize(_zip) / 1e6:.1f} MB"
      + (f" · !! not written at this render: {', '.join(_miss)}" if _miss else ""))
""")

md("92fdc8e2", r"""
## Record

The head this rendering read, the cache files it used, and the runtime. Re-running against a registry whose head has moved re-reads the
frames (the cache is keyed by head); nothing here writes to the registry.
""")

code("23555668", r"""
print(f"rendered against registry head {HEAD} (head now {R.head()}) · sisteer.registry @ {REGISTRY_CODE} · {time.time() - T0:.0f} s")
print("cache:", sorted(f for f in os.listdir(R.CACHE) if f.endswith(".parquet")))
print("figures:", sorted(f for f in os.listdir(R.FIGS) if f.endswith(".png")))
""")

nb = new_notebook(cells=cells)
# `python3` is the kernelspec ipykernel itself registers inside whatever interpreter runs nbconvert, so a fresh clone executes the
# notebook with no --ExecutePreprocessor.kernel_name override (a maintainer kernel name here refused on a fresh machine: NoSuchKernel)
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python", "version": "3.12"}}
nbformat.validate(nb)
nbformat.write(nb, OUT)
print(f"wrote {OUT}: {len(cells)} cells")
