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
**appendix figures at the bottom draw all four models** on their paper lanes, in the paper's Figure 2 / 3 form. Nothing but the plot and its
legend is on the canvas; provenance (model, lane, registry head) is printed under each figure.

Rules applied throughout: every StrongREJECT cell is graded by **Haiku** (a Sonnet-graded judgment is excluded and counted; a cell with only
Sonnet judgments is a gap, never a point); a cell with **several fit seeds is one point per seed**, each with its own interval, never pooled
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
--ExecutePreprocessor.timeout=1800 registry_qwen_figures.ipynb` (about ten minutes from the committed snapshot). See `README.md` for the panel list and what is not covered yet.
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
only, so a cell with several decode draws (`seed_base`, `trial_index`) is several points. An anchor with several unsteered draws is their
mean, the between-draw spread folded into the delta's interval (`anchor_draws`, `anchor_spread` in the tables).
""")

code("ec6aacd8", r'''
ML = ",".join(f"'{m}'" for m in R.MODEL_ORDER)
DECODE = "v.*, r.seed_base, r.trial_index"                                 # the run's decode seed rides beside the view's columns
# `unpooled_member` rows ride along with the paper rows (schema 22, 2026-10-02): the only readings of their cells until the producer's
# pooled rows land; `admit_provisional` keeps them only where the cell has no selected row and the load cell names them
PAPER_SQL = (f"SELECT {DECODE} FROM v_paper_long v JOIN run r ON r.run_id = v.run_id WHERE v.model_id IN ({ML}) "
             f"AND (v.in_current_paper = 1 OR (v.selection_state = '{R.PROVISIONAL_STATE}' AND v.metric_role = 'owner'))")   # owner rows only, as in_current_paper = 1 is by construction
# the grading code sha is on `judgment`, not on the paper view, and two seeds graded by different graders must not be
# pooled into one curve, so the frontier query carries it
FRONTIER_SQL = (f"SELECT {DECODE}, j.grading_code_sha FROM v_paper_long v JOIN run r ON r.run_id = v.run_id "
                "LEFT JOIN judgment j ON j.judgment_id = v.judgment_id WHERE v.model_id = '" + R.MODEL_ID + "' AND v.eval_id IN "
                "('crescendo', 'actorattack', 'pasttense', 'codeattack', 'renellm', 'grok420', 'xstest', 'or_bench_hard') AND v.metric_role = 'owner'")
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
_raw = {k: R.spell(R.query(q, k)) for k, q in (("paper_all", PAPER_SQL), ("qwen_frontier", FRONTIER_SQL), ("qwen_unsteered", POOL_SQL))}
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
runs_all = R.spell(R.query(RUNS_SQL, "runs_all"))
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
# A FLOOR IS HELD OUT: Qwen x agentic murder leaves every figure and table (unsteered rate 3.0 %, a floor with no room to
# show a reduction); the eval stays for the other models. Dropped here, before anchoring, so no panel, mean or table sees the cell.
op_all, _excluded = R.drop_excluded_cells(op_all)
if len(_excluded):
    print("cells held out at a floor (R.EXCLUDED_CELLS: the unsteered rate leaves no room to show a reduction), dropped before anchoring: "
          + "; ".join(f"{m} × {e}: {n} rows" for (m, e), n in _excluded.groupby(["model_id", "eval_id"]).size().items()))
    # THE REGISTRY'S WORD ON THE SAME CELLS (the registry's hold_out `r2v4-hold-out-qwen-agentic-murder-floor` on every Qwen
    # run of agentic_murder): once it folds, every dropped row carries held_out = 1 under that record, and the
    # constant and the registry agree; until then, or if they ever disagree, this line says so rather than letting the constant
    # stand unexamined. The constant keeps the DROP semantics (no column, no open marks); the registry keeps the rows.
    _held = pd.to_numeric(_excluded.get("held_out"), errors="coerce").fillna(0).gt(0) if "held_out" in _excluded.columns else pd.Series(False, index=_excluded.index)
    _rul = sorted({str(r) for r in _excluded.get("decisive_ruling_id", pd.Series(dtype=object)).dropna()}) if "decisive_ruling_id" in _excluded.columns else []
    if _held.all():
        # `decisive_ruling_id` is the row's SELECTION ruling (an include), not the hold_out; the frame carries the hold as the flag alone
        print(f"  the registry agrees: all {len(_excluded)} rows carry held_out = 1 (the hold_out filed for this cell, "
              f"r2v4-hold-out-qwen-agentic-murder-floor; the rows' selection rulings are {', '.join(_rul) or 'unnamed'}).")
    else:
        print(f"  !! the registry has NOT (yet) held these rows out: {int((~_held).sum())} of {len(_excluded)} carry held_out = 0"
              + (f" (rulings seen: {', '.join(_rul)})" if _rul else "") + " — the constant is the only place the hold-out lives until the registry's hold_out folds.")
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
               "over-refusal": ["xstest", "or_bench_hard"], "capability": ["mmlu", "gsm8k", "math500", "ifeval", "bigcodebench", "bfcl"],
               "agentic": ["agentic_leaking", "agentic_murder"]}               # Qwen's murder cells are excluded (R.EXCLUDED_CELLS), the eval stays
ALL_EVALS = list(dict.fromkeys(e for es in PANEL_EVALS.values() for e in es))
opd = R.dedupe(op[op.judge_ok])
cov = []
for e in ALL_EVALS:
    x = opd[opd.eval_id == e]; pr = runs_id[(runs_id.eval_id == e) & (runs_id.in_current_paper == 1)]
    cov.append(dict(eval_id=e, panels=" ".join(p[:4] for p, es in PANEL_EVALS.items() if e in es), headline_rows=len(x),
                    sources=", ".join(sorted(x.lane_source_legacy.unique())), objects=", ".join(o for o in ["none"] + R.OBJECTS if o in set(x.object_label)),
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
""")

md("b6d74ec9", r"""
## Figure 3 — static jailbreak defense, and the padding ladder (Qwen)

For each steered object and eval, the delta against the unsteered run in percentage points, oriented so that up is better (defense
success = 100 − ASR; benign compliance = 100 − over-refusal label mean, where a full refusal scores 1, a partial ½ and compliance 0). Small marks are the individual evals, the large mark after the
separator is the category's unweighted mean over them with an analytic interval (a black ring = a mean over fewer evals than the category
names); several seeds of one object are small marks, and the mean column averages them into one point. **Each fitted object is its three batch-64 seeds** - the earlier canonical batch-8 fit is retired from every panel; prompted arms and the unsteered model are not fits and keep every run. **Every point on an eval is differenced against that eval's common anchor**: the mean over every unsteered run of the eval in the paper frame at the point's generation cap (the campaign's run, the seed views' runs, the paired views' runs, the decode draws, pooled as draws with the between-draw variance in the anchor's uncertainty), and each point's interval combines its own with the anchor's. The producer's own filed contrast is kept beside each row as `filed_delta` for cross-checking but is not the plotted value. An open mark means the frame held no unsteered run for that eval and one was borrowed from the registry outside the selection; a pool spanning two decode spellings is flagged `mixed_decode` in the tables.
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
print("the mean points drawn (one per object: the seeds averaged; n_seeds says how many):")
R.show(R.pool_seeds(cm[cm.category.isin(JB4)]).drop(columns="model_id").round(2))
print("per seed (the small marks' means; anchors = the unsteered readings):")
R.show(cm[cm.category.isin(JB4)].drop(columns="model_id").round(2))
print("per eval, the steered reading (y), the unsteered anchor it is compared with (anchor_value) and the delta (d_y), all in plotted units:")
R.show(d[d.eval_id.isin([e for c in JB4 for e in R.CATEGORIES[c][0]])][DCOLS].sort_values(["eval_id", "object_label", "seed"]).round(1))
""")

md("5e61e1ab", r"""
## Figure 5 — Petri, agentic misalignment and capability at the operating dose (Qwen)

Same construction, with Petri on its own row: **all six behaviors separately, then the pooled score on the right**.
As in Figures 3–4, each fitted object is its three batch-64 seeds (the canonical batch-8 fit is retired) and every point on an eval is
differenced against that eval's **common anchor**: the mean over every unsteered run of the eval at the same generation cap, its uncertainty
in each point's interval.
The pooled column is the registry's own `petri_pooled` run - the producer's pooling over the behaviors, not an unweighted mean of the six
columns - so it is labeled "pooled", not "mean". Then reductions in misaligned agentic actions (pp) and the accuracy delta on the capability
evals the category averages over (MMLU, GSM8K). Where several FITS of one recipe exist (the paired ReLU's three seeds), the mean point carries
the between-fit variance in its interval and names the spread when the fits disagree by more than 10 in the panel's unit (for example
41.3 pp vs 1.5 pp across three fits on PAIR). The EM organism panel is a Llama-only appendix panel of the paper and is not drawn here.
""")

code("624d0e27", r"""
FIG5 = [["Petri", "agentic misalignment"]]          # misalignment, one row: Petri is seven columns wide
                                                    # (the six behaviors, then pooled), agentic three, so the row is ten across
FIG6 = [["over-refusal", "capability"]]             # capabilities: the cost side, over-refusal beside capability
# NARROWER PANELS, PETRI LABELS AT 45 DEGREES, AXES THE SAME HEIGHT AS THE OTHER ONE-ROW FIGURES.
# The rule is about the Y-AXIS height, not the figure's: rotated labels take vertical room, so the figure
# grows by exactly that much and the axes stay comparable. MEASURED on empty frames (axes bbox / dpi): Figure 4's axes
# are 3.66 in tall at the 4.2 in default; Figure 5 with Petri rotated reaches 3.67 in at height 5.4 (5.6 gave 3.87,
# 5.2 gave 3.47). The horizontal two-line labels this replaces left Figure 5's axes at 3.46 in — the rotated form at
# 5.4 is the closer match. unit 2.0 -> 1.5 brings the width from ~21.5 in to ~16.4 in, about the legend's.
# Petri back at the standard column width with horizontal labels, the agentic panel titled "agentic" with its single eval and no
# mean column; the narrower Petri + rotated labels above were for the room the two-eval agentic panel took
# the agentic panel one Petri column wide; tick labels at the standard size (never shrunk), which needs
# 2.2 in per column for "reasoning unfaithfulness" and "self-preservation" to clear each other -- 19.0 in for the row
R.delta_figure(cm, d, FIG5, R.OBJECTS, gaps_q, "fig5_misalignment", unit=2.2, categories=CAT_Q, titles={"agentic misalignment": "agentic"}); plt.show()
_mi = R.missing_intervals(d, [e for c in FIG5 for cat in c for e in CAT_Q[cat][0]])
if len(_mi):
    print(f"UNMEASURED intervals on this panel: {int(_mi.rows.sum())} drawn rows carry a value but no interval (their filed contrast is not current in the registry while "
          f"the adapters re-anchor; petri_score has no per-item data to make one from) — drawn as a mark with no bar, tabulated without ±: "
          + "; ".join(f"{R.LABEL.get(r.object_label, r.object_label)} on {r.eval_id}" for r in _mi.itertuples()))
# Petri and agentic misalignment tabulate separately too — and they are not even in the same
# unit: Petri is a 1--10 score, agentic a percentage of scenarios, so one table's columns never shared a meaning.
for _cat, (_name, _slug, _unit) in {
        "Petri": ("Petri", "petri", "the reduction in the pooled misalignment score on its 1--10 scale"),
        "agentic misalignment": ("Agentic misalignment", "agentic",
                                 "the reduction in misaligned actions, in percentage points")}.items():
    print(R.table_evals_seeds(d, [[_cat]], R.OBJECTS,
          f"{_name} for Qwen3.5-4B.", f"tab:qwen-{_slug}-seeds"))

# WHERE THE AGENTIC VALUES CAME FROM. The completion-cap MERGED readings are the agentic value for the Qwen seed
# views, superseding the 3,000-cap rows; on several arms that roughly doubles leaking. That is the instrument changing, not the
# models, so the provenance is printed rather than left for someone to infer by diffing two artifacts. Empty before the fold.
_prov = R.merge_provenance(op[op.eval_id.isin(R.CATEGORIES["agentic misalignment"][0])])
if len(_prov):
    print(f"agentic readings that are completion-cap merges (the agentic value for the Qwen seed views, superseding their 3,000-cap rows): {len(_prov)} of the panel's cells; "
          f"everything else is a single read. `share_from_redraw` is the fraction of a cell's verdicts drawn from the 16k "
          f"re-draw rung, `still_at_ceiling` the fraction that never resolved even there.")
    # ONE WARNING LINE, NOT ONE PER ROW. It was a per-row caption until the rendered table truncated it at pandas' column
    # width and silently cut this clause off every row - the warning was in the frame and not on the page.
    # THE DISCLOSURE NAMES WHICH ARMS CROSS, measured per arm against the anchor actually used (`decode_mismatch`, the same
    # field the appendix footnote reads) rather than by a lane proxy. Since the panel moved onto the native canon baseline
    # the merged arms match it on decode and engine and cross nothing; the vendor-decode arms now differ from it and do.
    _ag = d[d.eval_id.isin(R.CATEGORIES["agentic misalignment"][0]) & d.object_label.ne("none")]
    _x = int(_ag.decode_mismatch.sum()) if "decode_mismatch" in _ag.columns else 0
    if _x:
        print(f"  !! {_x} of {len(_ag)} arms on this panel cross a decode boundary: their runs are the vendor preset at "
              f"temperature 0.7 while the panel's anchor is the native completion-cap baseline at temperature 1.0. The "
              f"remaining {len(_ag) - _x} match the anchor on both decode and serving engine and cross nothing.")
    elif len(_ag):
        print("  every arm on this panel shares the anchor's decode and serving engine; no delta here crosses a boundary.")
    if len(_ag):
        print("  A caveat that survives the re-anchor and is not about the instrument: on agentic murder a large share of "
              "each cell's verdicts never resolves even at the 16,384-token ceiling (`still_at_ceiling` below); temperature "
              "1.0 looping the ladder cannot cure that.")
    R.show(_prov[["object_label", "eval_id", "seed", "cap", "share_from_redraw", "still_at_ceiling", "engine"]].round(3), n=10)
else:
    print("agentic readings: no completion-cap merges in the frame; every agentic value is a single read.")
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
_rt = R.rung_eval_table(frontier, R.FRONTIER_OBJECTS)
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
    _q = R.rung_quality_table(frontier, _col, R.FRONTIER_OBJECTS)
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

The same panels across **Gemma-4-E4B, Qwen3.5-4B, Nemotron-3-Nano-4B and Llama-3.1-8B**, each on its paper lane, in the paper's Figure 2 / 3
form: x = the objects as lanes, the models dodged inside each lane by marker shape (square, circle, triangle, inverted triangle); per (object,
model) the category mean is the large mark with its interval and the individual evals are small faded marks of the same shape beside it. The
roster is the main figures' order without the Refusal steer (Qwen-only): the two prompted SIs, then each corpus's ReLU steer beside its
LoRA. **Layout**: the figures whose x is the object stack their category panels vertically,
one per row; the figures in the main figures' form, x = the eval (`figA_*_absolute_models`), keep one row and each has a **relative
companion** (`figA_*_delta_models`): the same panels with every cell as its change against its own model's unsteered run - the anchored
deltas the x = object figures and the tables read - oriented so up is better, each panel ranged on its own axis. **One interval rule for a
mean over evaluations**: every `mean` column here, absolute or relative, and every category mean in Figures 3–5 and the
x = object figures, is the unweighted mean of the per-evaluation values with se = sqrt(Σ se²) / k — the per-evaluation intervals combined,
no term for the spread between evaluations (the battery is fixed, not sampled). Readings of ONE cell (decode draws, fit seeds) pool with
their within + between term before that, as the Statistics paragraph describes separately. The black bar on each large diamond is a
different thing again: ± SEM over the models present — a level summary of the four models, never a contrast interval. **Every table** here leads
with the unsteered model's absolute reading, one row per model, and states every other arm as a signed change against it. Deltas are against each model's own unsteered run on its lane, with the same routes as Figures 3–4 (Qwen's multi-turn anchor
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
d_all = R.anchored(op_all, pool=POOL_Q)                    # every model's deltas on its paper lane (the pool only serves Qwen: it is Qwen's frame)
cm_all = R.category_means(d_all, categories=R.APPENDIX_CATEGORIES)     # the appendix's capability axis is MMLU + GSM8K
gaps_m = R.gap_status(runs_lane, R.drawn_set(d_all[d_all.d_y.notna()]), R.APPENDIX_OBJECTS, ALL_EVALS, models=R.MODEL_ORDER, wrong_judge=WJ)
JB_ALL = [["static", "multi-query", "multi-turn"]]           # the appendix keeps the three jailbreak families together
R.category_panels(cm_all, d_all, JB_ALL, R.APPENDIX_OBJECTS, R.MODEL_ORDER, gaps_m, "figA_jailbreaks_models", categories=R.APPENDIX_CATEGORIES); plt.show()
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
R.category_panels(cm_all, d_all, FIG6, R.APPENDIX_OBJECTS, R.MODEL_ORDER, gaps_m, "figA_capabilities_models", categories=R.APPENDIX_CATEGORIES); plt.show()
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
ATTACKS, OR_EVALS, CAP, AG = PANEL_EVALS["attacks"], PANEL_EVALS["over-refusal"], PANEL_EVALS["capability"], PANEL_EVALS["agentic"]
AB_OBJECTS = ["none"] + R.APPENDIX_OBJECTS
pts_ab = R.flip(R.headline_points(op_all, ATTACKS))
gaps_ab = R.gap_status(runs_lane, R.drawn_set(pts_ab), AB_OBJECTS, ATTACKS, models=R.MODEL_ORDER, wrong_judge=WJ)
R.absolute_panels(pts_ab, ATTACKS, AB_OBJECTS, R.MODEL_ORDER, gaps_ab, "defense success rate (%; 100 − ASR at threshold 0.5)", "figA_attacks_absolute_models"); plt.show()
# THE RELATIVE COMPANION (the appendix figures in the main figures' form keep one row and also plot relative to
# unsteered): the same panels, every cell its change against its own model's unsteered run - the anchored deltas the x = object figure
# above and the tables read - oriented so up is better defense; a cell with nothing to difference against draws the `no unsteered run` glyph
gaps_abd = R.gap_status(runs_lane, R.drawn_set(R.delta_points(d_all, ATTACKS, R.APPENDIX_OBJECTS)), R.APPENDIX_OBJECTS, ATTACKS, models=R.MODEL_ORDER, wrong_judge=WJ)
R.delta_panels(d_all, ATTACKS, R.APPENDIX_OBJECTS, R.MODEL_ORDER, gaps_abd, "figA_attacks_delta_models"); plt.show()
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
          f"tab:allmodels-absolute-{_slug}", flip_baseline=True))
print(f"four models on their paper lanes · operating dose · registry head {HEAD}")
R.show(pts_ab.pivot_table(index=["eval_id", "object_label", "seed"], columns="model_id", values="y").round(1))
""")

md("6d2f8b3a", r"""
### Appendix — best-of-N search curves

Defense success within the first N attempts, $100\,(1-\mathrm{ASR}@N)$, against N on a log axis, one panel per model, the same admitted
runs as the attacks panels above. Qwen's three fit seeds (and the three seed-view standups of its unsteered and
prompted arms) are the thin faint lines, the thick marked line their mean; the other models have one run per arm. Every point is the
per-goal any-hit rate over the 100 confirmatory goals at threshold 0.5 under the canonical climbed instrument (the 512 store's at-cap
draws re-drawn to 4,096, the N = 75 point being the headline `asr_bon_cap4096`); Llama's paired difference ReLU is its native-4,096
pass. Curves come from the producers' `asr_at_n<N>` rows; a run that files only the `asr_at_n_4096_cf` grid on its headline row is read
from that grid and footnoted until its rows land. The composition table under the figure names every run drawn.
""")

code("b0a7c4e1", r"""
BON_OBJECTS = ["none"] + R.APPENDIX_OBJECTS
bon_all = R.bon_curve_points(op_all, paper_all)                # the runs the attacks panels admit, with their per-N rows from the paper frame
bon_pts, bon_dropped = R.bon_curve_select(bon_all[bon_all.object_label.isin(BON_OBJECTS)])
R.bon_curves_figure(bon_pts, BON_OBJECTS, R.MODEL_ORDER, "figA_bon_curves_models"); plt.show()
print(f"four models on their paper lanes · operating dose · registry head {HEAD}")
print("every run drawn (curve = rows: asr_at_n<N> metric rows; extra: the bon_stores grid stopgap), with its decode policy:")
R.show(R.bon_curve_composition(bon_pts), n=80)
print(f"admitted runs NOT drawn ({len(bon_dropped)}), and why (headline only = another lane's anchor with no curve rows; other decode = a "
      f"grandfathered T = 1.0 climb beside vendor-preset standups of the same cell; duplicate filing = one store read by two readers):")
R.show(bon_dropped, n=40)
# THE ADMISSION GAP: a cell drawn here from a grandfathered T = 1.0 campaign run while selected vendor-preset
# runs of the same cell sit in the paper frame unadmitted (Qwen's seedrep prompted SIs file asr_bon_cap4096 as their headline). Listed
# until the admission reads the headline family; then these runs enter op_all and the figure draws them as seeds + mean by itself.
bon_gap = R.bon_curve_unadmitted(op_all, paper_all, BON_OBJECTS, R.MODEL_ORDER)
print(f"vendor-preset runs of a drawn cell that the frame holds selected but op_all does not admit ({len(bon_gap)}; the cell is drawn from its T = 1.0 run until the admission is fixed):")
R.show(bon_gap, n=40)
""")

code("9b3d6f18", r"""
pts_ac = R.flip(R.headline_points(op_all, OR_EVALS))
gaps_ac = R.gap_status(runs_lane, R.drawn_set(pts_ac), AB_OBJECTS, OR_EVALS, models=R.MODEL_ORDER, wrong_judge=WJ)
CAP_A = R.APPENDIX_CAPABILITY_EVALS                            # MMLU, GSM8K, BigCodeBench, BFCL (Llama has no BFCL: out by design)
pts_ad = R.headline_points(op_all, CAP_A)                      # defined here, beside over-refusal, because they now share a figure
gaps_ad = R.gap_status(runs_lane, R.drawn_set(pts_ad), AB_OBJECTS, CAP_A, models=R.MODEL_ORDER, wrong_judge=WJ)
# over-refusal and capability share one figure, matching Figure 6, but NOT one y label: benign compliance and accuracy
# are percentages of different things, so each panel is labeled on its own axis (the dict form of `ylabel`).
R.absolute_panels(pd.concat([pts_ac, pts_ad], ignore_index=True), OR_EVALS + CAP_A, AB_OBJECTS, R.MODEL_ORDER,
                  pd.concat([gaps_ac, gaps_ad], ignore_index=True),
                  {"over-refusal": "benign compliance (%; 100 − over-refusal label mean)", "capability": "accuracy / pass rate (%)"},
                  "figA_capabilities_absolute_models"); plt.show()
# THE RELATIVE COMPANION: the same two panels as changes against each model's unsteered run, up = better
# (more benign compliance, higher accuracy / pass rate); the capability label names both since the coding evals joined the axis
_ev_cd = OR_EVALS + CAP_A
gaps_acd = R.gap_status(runs_lane, R.drawn_set(R.delta_points(d_all, _ev_cd, R.APPENDIX_OBJECTS)), R.APPENDIX_OBJECTS, _ev_cd, models=R.MODEL_ORDER, wrong_judge=WJ)
R.delta_panels(d_all, _ev_cd, R.APPENDIX_OBJECTS, R.MODEL_ORDER, gaps_acd, "figA_capabilities_delta_models",
               ylabel={"over-refusal": "Δ benign compliance (pp)", "capability": "Δ accuracy / pass rate (pp)"}); plt.show()
print(R.table_absolute(d_all, OR_EVALS, AB_OBJECTS, R.MODEL_ORDER,
      "Benign compliance across the four models.",
      "tab:allmodels-absolute-overrefusal", flip_baseline=True))
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
pts_ag = R.headline_points(op_all, AG)
gaps_ag = R.gap_status(runs_lane, R.drawn_set(pts_ag), AB_OBJECTS, AG, models=R.MODEL_ORDER, wrong_judge=WJ)
R.absolute_panels(pts_ag, AG, AB_OBJECTS, R.MODEL_ORDER, gaps_ag, "misaligned actions (% of scenarios)", "figA_agentic_absolute_models", ylim=(-2, 62)); plt.show()
# THE RELATIVE COMPANION, oriented as the main figures orient agentic misalignment: a REDUCTION in misaligned
# actions, so up is better here while the absolute panel above reads misaligned actions upward
gaps_agd = R.gap_status(runs_lane, R.drawn_set(R.delta_points(d_all, AG, R.APPENDIX_OBJECTS)), R.APPENDIX_OBJECTS, AG, models=R.MODEL_ORDER, wrong_judge=WJ)
R.delta_panels(d_all, AG, R.APPENDIX_OBJECTS, R.MODEL_ORDER, gaps_agd, "figA_agentic_delta_models"); plt.show()
# Petri had NO absolute table while agentic did: the two are separate tables here as
# everywhere else. Petri's unit is the 1--10 misalignment score, not a percentage, so it never belonged with agentic.
_PETRI_EV = [R.POOLED_EVAL["Petri"]] if "Petri" in R.POOLED_EVAL else R.CATEGORIES["Petri"][0]
print(R.table_absolute(d_all, _PETRI_EV, AB_OBJECTS, R.MODEL_ORDER,
      "Petri misalignment score across the four models.",
      "tab:allmodels-absolute-petri"))
print(R.table_absolute(d_all, AG, AB_OBJECTS, R.MODEL_ORDER,
      "Misaligned actions across the four models.", "tab:allmodels-absolute-agentic"))
print(f"four models on their paper lanes · operating dose · registry head {HEAD}")
R.show(pts_ag.pivot_table(index=["eval_id", "object_label", "seed"], columns="model_id", values="y").round(1))
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
stated in the pairs table as information. The form is the x = object appendix figures': the category mean is the large mark with its
interval (se = $\sqrt{\sum \mathrm{se}_e^2}/k$), the evaluations small and faded beside it, a black ring a mean over fewer
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

code("b7d4e9f2", r"""
# ONE-CLICK ZIP OF THE PAPER'S FIGURES, rebuilt on every render: the named figures in every format the
# notebook wrote (png, pdf, svg), plus a manifest naming the render they came from. It lands in figures/ beside them.
import zipfile, os, time
PAPER_FIGURES = ["fig3_single_turn_padding", "fig4_multi_turn", "fig5_misalignment", "fig6_capabilities", "fig7_frontier",
                 "figA_bon_curves_models", "figA_capabilities_models", "figA_misalignment_models", "figA_jailbreaks_models",
                 "figA_relu_arditi_cosines", "figA_transfer_models", "figA_qwen_control_alpha_minus1"]   # the last is the alpha = -1 appendix figure
os.makedirs(R.FIGS, exist_ok=True)          # a render that drew no figure (a partial frame) still writes its manifest and records
_zip = os.path.join(R.FIGS, "paper_figures.zip")
_have, _miss = [], []
with zipfile.ZipFile(_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for name in PAPER_FIGURES:
        found = False
        for ext in ("png", "pdf", "svg"):
            f = os.path.join(R.FIGS, f"{name}.{ext}")
            if os.path.exists(f):
                z.write(f, arcname=f"{name}.{ext}"); found = True
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
