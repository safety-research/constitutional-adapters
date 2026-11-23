"""appendix_generator: the generator-comparison appendix figure, on the control-pool appendix's five panels, and the lines printed under it.

    import appendix_generator as AG
    data = AG.load(R.SOURCE)                         # the committed file; a live render first re-derives it from the registry and refuses on a difference
    gen_d = AG.anchor(AG.lane_frame(data), ctl_op, ctl_pool, ctl_md)   # Figure 11's pooled anchors, once per condition
    rows = AG.tidy_rows(gen_d, data)                 # the 72 per-seed marks (NaN where a reading is not in yet), appendix_common's form
    AG.draw(rows, "<figure name>")                   # the figure, saved into R.FIGS, every APPENDIX_FIGS_MIRROR directory and `extra_dirs`
    print(AG.summary_text(data, gen_d, rows, ctl_md))   # the static verdict, one headline per panel, GSM8K's coverage, the drift line
    print(AG.provenance_line(data, R.SOURCE))

WHAT IS COMPARED. Rank-4 o_proj constitution LoRAs on Qwen3.5-4B-Base, fit on a 1M-token pool of the constitution corpus as Opus 5 wrote
it (circle) and on a 1M-token pool of the corpus Qwen3.8-Max wrote through the same pipeline (diamond), three seeds each (effective batch
8: the small-corpus recipe, not the paper's constitution LoRA), served on Qwen3.5-4B on the control-pool appendix's twelve instruments.
The runs sit outside the paper's selection (registry rulings) and are read from `data/appendix_generator_rows.json`, which
`appendix_generator_sidecar.py` writes; this module opens no registry except to check that file in a live render.

THE READINGS are the control-pool appendix's, computed by its frame recipe (`registry_nb.ctlstop_frame`: each payload reduced through
`registry_nb.ctlstop_judging.reduce` at R.SR_THRESHOLD), and the file's contract is that appendix's (`registry_nb.CTLSTOP_EVALS`: the
instrument, item set, metric, drawn cap and judge of every eval).

THE ANCHORS ARE FIGURE 14's. Every point is differenced against the unsteered anchor the control-pool comparison figure's points of the
same class are differenced against in the same render: `registry_nb.ctlstop_anchored(ctl_op, rows, ctl_pool, ctl_md)` (the main figures'
rule, interval and cap class by construction; over-refusal, per seed, on the unsteered runs the main figures' fitted rows of that eval and
seed are differenced against, which the anchored row names as `anchor_binding`: `bound`, the one run the registry binds to them, or
`pooled`, the runs pooled for them where none is bound), called ONCE PER CONDITION because `anchored` keeps one reading per cell and
its cell key has no condition, so an Opus row and a Qwen row of one seed would collide. Each call asserts the anchor RULE against
`ctl_md` (`registry_nb.ctlstop_anchor_problems`) and raises on a breach. The lane's own unsteered static runs are no anchor: they are
printed as a drift line against the pooled anchor.

THE MARK is that delta in the panel's unit, oriented as Figure 11 orients it (`registry_nb.ctlstop_categories`: + is better; the
over-refusal mark is minus the change in the label mean x 100), with the anchored row's 95% interval. The mean marks are
appendix_common's (`registry_nb.category_means` within seed, then `registry_nb.pool_seeds`), drawn only where every seed averages every
eval of the panel (mean_rule "complete"); a reading not yet in the data draws the gap glyph and is named under the figure.

THE PRINTED LINES (`summary_text`): the static panel's pre-fixed verdict (Qwen - Opus over the three matched seed pairs, pooled over
the four attacks, t on 2 df; anchor-independent, read from the readings themselves); one headline per panel, each corpus's mean mark
with its interval and the Qwen - Opus difference over matched seeds (descriptive on every panel but the static one), under a line
saying that difference is computed anchor-free from the readings, so it need not equal the gap between the two marks where a seed's
two runs take different anchors; GSM8K's coverage (scored items of 200) with the all-200 reading beside it; and the drift line.
"""
from __future__ import annotations

import math
import os
import sqlite3

import numpy as np
import pandas as pd

import registry_nb as R
import appendix_common as AC
import appendix_generator_sidecar as G

FIGURE = "figA_qwen_generator_ablation"
# A KEY OF ITS OWN, never `constitution_lora`: these are 1M-token batch-8 fits, not Figure 3's constitution LoRA, and must not read as it
OBJECT = "constitution_lora_1m"
OBJECT_LABEL = "1M-token constitution LoRA"          # the batch size and the recipe are the caption's
COLORS = {OBJECT: R.COLOR["constitution_lora"]}
CONDITIONS = ("opus_written", "qwen_written")
CONDITION_LABELS = {"opus_written": "Opus 5-written corpus", "qwen_written": "Qwen3.8-Max-written corpus"}
SHORT = {"opus_written": "Opus", "qwen_written": "Qwen"}
SEEDS = ("s1", "s2", "s3")
#: Figure 11's panels: its categories (capability narrowed to MMLU and GSM8K) and its grid, one eval column as wide as Figure 11's
CATEGORIES = R.ctlstop_categories(R.MODEL_ID)
LAYOUT = R.CTLSTOP_GRID
UNIT = 2.4
EVALS = tuple(e for c in CATEGORIES.values() for e in c[0])
STATIC = tuple(CATEGORIES["static"][0])
#: the pre-fixed verdict's columns: the four attacks (the verdict) and the pooled row without ReNeLLM (declared before the runs)
VERDICT = ("pasttense", "codeattack", "renellm", "grok420")
DECLARED = ("pasttense", "codeattack", "grok420")
#: the lane-fixed registration (`registry_nb.register_lane_fixed`): every judged number of this figure is the control-pool judges'
LANE_FIGURES = "generator-comparison figure"
#: the line printed above the panel headlines: their Qwen - Opus is read from the readings (`corpus_difference`), not from the marks
PANEL_NOTE = ("panels (each corpus's mean mark as drawn, against Figure 11's anchors; Qwen - Opus over the matched seed pairs is computed "
              "anchor-free from the readings themselves, so where a seed's two runs were read on different rungs and take different "
              "anchors it need not equal the gap between the two marks):")
#: which GSM8K answers go unscored: the cap cut them on the first rung and the ladder did not redraw them, or it redrew them and the
#: last rung's cap cut them again
GSM8K_UNSCORED = (f"an answer the cap cut is unscored: one cut at {R.CTLSTOP_GSM8K_RUNGS[0]:,.0f} that the ladder did not redraw, or one "
                  f"it redrew that the {R.CTLSTOP_GSM8K_RUNGS[-1]:,.0f} cap cut again")
SIDECAR_REL = str(G.SIDECAR.relative_to(G.REPO))
if set(EVALS) != set(G.EVALS) or STATIC != G.STATIC or tuple(CATEGORIES["static"][0]) != VERDICT:
    raise ImportError(f"the figure's evals {EVALS} are not the file's {G.EVALS}")


class GeneratorSidecarMismatch(RuntimeError):
    """A live render whose registry no longer yields the committed file: the figure is not drawn from either side."""


# ---- the data ------------------------------------------------------------------------------------------------------------------------
def load(source: str, path=None) -> dict:
    """The committed file (`path`, default data/appendix_generator_rows.json), validated by its contract (`load_rows`; a provisional file
    with readings still pending loads, and they are drawn as gaps). `source` is the notebook's R.SOURCE: "snapshot" reads the file alone
    and opens no registry; "live" also derives the file again from the registry (read-only, one transaction:
    `appendix_generator_sidecar.derive`) and checks the static comparison filing's report where it is readable, and raises
    GeneratorSidecarMismatch on any difference (bookkeeping aside) or a derivation the registry no longer supports."""
    if source not in ("live", "snapshot"):
        raise ValueError(f"source {source!r}: live or snapshot")
    data = G.load_rows(path)
    if source == "snapshot":
        return data
    try:
        conn = G.connect()
    except (FileNotFoundError, sqlite3.Error) as err:     # the message names no location: notebook output is committed
        raise GeneratorSidecarMismatch(f"a live render cannot open the registry to check {SIDECAR_REL} ({type(err).__name__})") from None
    try:
        try:
            fresh = G.derive(conn)
        except G.Refused as err:
            raise GeneratorSidecarMismatch(f"the live registry no longer yields {SIDECAR_REL}: " + "; ".join(err.problems[:8])
                                           + (f"; ... {len(err.problems) - 8} more" if len(err.problems) > 8 else "")
                                           + " (regenerate it with appendix_generator_sidecar.py write once the registry is consistent)") from None
        _status, report = G.report_file_check(conn)
    finally:
        conn.close()
    diffs = G.differences(data, fresh) + list(report)
    if diffs:
        raise GeneratorSidecarMismatch(f"{SIDECAR_REL} differs from the live registry: " + "; ".join(diffs[:8])
                                       + (f"; ... {len(diffs) - 8} more" if len(diffs) > 8 else "")
                                       + " (the figure is not drawn from either side; regenerate the file with appendix_generator_sidecar.py write)")
    return data


def _stored(r: dict) -> dict:
    """One file row in the control-pool file's stored form (`registry_nb.CTLSTOP_STORED`), as `registry_nb.ctlstop_frame` takes it."""
    steered = r["object_id"] != G.UNSTEERED
    return dict(run_id=r["run_id"], object_id=r["object_id"], rule=None, slot=OBJECT if steered else G.UNSTEERED,
                seed=G._seed_name(r["seed"]), eval_id=r["eval_id"], role=r["role"], model_id=r["model_id"],
                instrument_id=r["instrument_id"], item_set_id=r["item_set_id"], lifecycle_state=r["lifecycle_state"],
                judgment_id=r["judgment_id"], judge_id=r["judge_id"], judge_model=r["judge_model"], rubric_name=r["rubric_name"],
                rubric_version=r["rubric_version"], token_cap_tokens=r["token_cap_tokens"], run_token_cap_tokens=r["run_token_cap_tokens"],
                pass_label=r["pass_label"], lane=r["transfer_kind"], dose=(1 if steered else None), pad=r["pad"], pad_contract=False,
                serving_path=r["serving_path"], bon_n=r["bon_n"], anchor_run_id=r["anchor_run_id"],
                lane_source_legacy=r["lane_source_legacy"], view_legacy=r["view_legacy"], held_out=r["held_out"],
                rung_run_id=r["rung_run_id"])


def lane_frame(data: dict, threshold: float | None = None) -> pd.DataFrame:
    """Every drawn and drift row of the file as a frame `anchored` takes, MEASURED at load by the control-pool appendix's own recipe
    (`registry_nb.ctlstop_frame` at StrongREJECT threshold `threshold`, default R.SR_THRESHOLD): object_label OBJECT (the drift rows
    `none`), seed `s1`..`s3` as Figure 11's frames spell it, the registry's view, caps, item set, pass label and decode columns, and
    `condition` (opus_written / qwen_written; None on a drift row)."""
    keep = [r for r in data["rows"] if r["role"] in ("drawn", "drift_anchor")]
    f = R.ctlstop_frame({"rows": [_stored(r) for r in keep], "items": data["items"]}, threshold)
    by = {r["run_id"]: r for r in keep}
    f["arm"] = f.run_id.map(lambda x: by[x]["arm"])
    f["condition"] = f.arm.map(G.CONDITION).astype(object)
    for c in ("decode_policy_id", "decode_source"):
        f[c] = pd.Series([by[x][c] for x in f.run_id], index=f.index, dtype=object)
    return f


def anchor(frame: pd.DataFrame, op: pd.DataFrame, pool: pd.DataFrame | None, d_ref: pd.DataFrame) -> pd.DataFrame:
    """The drawn rows of `lane_frame` with Figure 11's anchors: `registry_nb.ctlstop_anchored(op, rows, pool, d_ref)` ONCE PER CONDITION
    (`d_ref` is `ctl_md`, the anchored frame Figure 11's mirrors pass; `op` / `pool` its ctl_op / ctl_pool), concatenated, with every
    main-figure row the anchors use held to the appendix's judge and threshold (`registry_nb.ctlstop_assert_one_judge`). Raises where a
    condition's anchoring loses a row or breaks the anchor rule (`ctlstop_anchored` asserts both). No attrs."""
    drawn = frame[frame.role.eq("drawn")]
    parts = []
    for c in CONDITIONS:
        rows = drawn[drawn.condition.eq(c)]
        if not len(rows):
            continue
        new = R.ctlstop_anchored(op, rows.reset_index(drop=True), pool, d_ref)
        new["condition"] = c
        new.attrs = {}
        parts.append(new)
    if not parts:
        return drawn.iloc[0:0].assign(**{c: pd.Series(dtype=float) for c in ("d_y", "d_lo", "d_hi")})
    out = pd.concat(parts, ignore_index=True)
    if len(out) != len(drawn) or set(out.run_id) != set(drawn.run_id):
        raise AssertionError(f"anchoring kept {len(out)} of {len(drawn)} drawn rows")
    R.ctlstop_assert_one_judge(R.ctlstop_main_rows(out, out.iloc[0:0], op, pool), evals=set(out.eval_id))
    out.attrs = {}
    return out


def tidy_rows(gen_d: pd.DataFrame, data: dict) -> pd.DataFrame:
    """The figure's rows, one per reading of the grid (72: twelve evals x two corpora x three seeds), in appendix_common's tidy form:
    object, seed, category, eval, condition, delta, ci_lo, ci_hi (the anchored row's d_y and interval in the panel's unit, which
    appendix_common orients by the category's sign as Figure 11 does), n (the items behind the reading) and open (the anchor came from
    outside the paper frame); NaN delta where the reading is not in the data yet (`pending`) or carries no delta (a static reading on
    a rung above every comparator's: its route says so). Plus what each mark is read from: run_id, y (the reading), anchor_value,
    anchor_draws, cap, cap_mismatch, route, anchor_binding (on over-refusal, how the anchored row took Figure 11's anchor of its seed:
    `bound` or `pooled`, `registry_nb.ctlstop_bind_anchors`; None elsewhere), and `why` for a mark with no delta."""
    by = {(r.condition, str(r.seed), r.eval_id): r for r in gen_d.itertuples()} if len(gen_d) else {}
    pend = {(G.CONDITION.get(p["arm"]), G._seed_name(p["seed"]), p["eval_id"]): p["why"] for p in data.get("pending") or []}
    out = []
    for cat, (evals, _sign, _y) in CATEGORIES.items():
        for e in evals:
            for c in CONDITIONS:
                for s in SEEDS:
                    r = by.get((c, s, e))
                    rec = dict(object=OBJECT, seed=s, category=cat, eval=e, condition=c, delta=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                               n=np.nan, open=False, se_steer=np.nan, anchor_terms={}, steer_terms={}, split_anchor_terms={}, run_id=None,
                               y=np.nan, anchor_value=np.nan,
                               anchor_draws=np.nan, cap=np.nan, cap_mismatch=False, route=None, anchor_binding=None, why=pend.get((c, s, e)))
                    if r is not None:
                        how = getattr(r, "anchor_binding", None)
                        rec.update(run_id=r.run_id, y=float(r.y), n=float(r.n), anchor_value=float(r.anchor_value) if pd.notna(r.anchor_value) else np.nan,
                                   anchor_draws=float(r.anchor_draws) if pd.notna(r.anchor_draws) else np.nan, cap=float(r.token_cap_tokens),
                                   cap_mismatch=bool(r.cap_mismatch) if pd.notna(r.cap_mismatch) else False, route=str(r.route),
                                   anchor_binding=str(how) if how is not None and pd.notna(how) else None, why=None)
                        if pd.notna(r.d_y):
                            parts = R.delta_parts(pd.Series(r._asdict()))       # the seeds item-clustered, the anchor once, in the mean marks
                            rec.update(delta=float(r.d_y), ci_lo=float(r.d_lo), ci_hi=float(r.d_hi), open=not bool(r.anchor_in_frame),
                                       se_steer=parts.se_steer, anchor_terms=parts.anchor_terms, steer_terms=parts.steer_terms,
                                       split_anchor_terms=parts.split_anchor_terms)
                        else:
                            rec["why"] = str(r.route)
                    elif rec["why"] is None:
                        rec["why"] = "no reading in the file"
                    out.append(rec)
    return pd.DataFrame(out)


# ---- the figure ----------------------------------------------------------------------------------------------------------------------
def _stamped(fig):
    """`fig` with its savefig writing the render's stamp (`registry_nb.render_stamp`: the policy and every lane-fixed judge) into each
    file's metadata, as `registry_nb.save` does for the notebook's own figures. `appendix_common.save_figure` writes none; once it does,
    this goes. No stamp, nothing changes."""
    stamp = R.render_stamp()
    if not stamp:
        return fig
    plain = fig.savefig

    def savefig(fname, *args, **kw):
        ext = os.path.splitext(str(fname))[1].lstrip(".").lower()
        kw.setdefault("metadata", {"Subject": stamp} if ext == "pdf" else {"Description": stamp})
        return plain(fname, *args, **kw)

    fig.savefig = savefig
    return fig


def figure_options(**kw) -> dict:
    """The options this figure passes to `appendix_common.paired_comparison_figure`: Figure 11's categories and grid, a y axis per panel,
    the two corpora as circle and diamond, the one object in its own key and colour, the complete-mean rule, Figure 11's column width."""
    opts = dict(categories=CATEGORIES, layout=LAYOUT, sharey=False, conditions=list(CONDITIONS), objects=[OBJECT], colors=COLORS,
                labels={OBJECT: OBJECT_LABEL}, condition_labels=CONDITION_LABELS, mean_rule="complete", unit=UNIT)
    opts.update(kw)
    return opts


def draw(rows: pd.DataFrame, name: str | None = None, *, out_dirs=None, extra_dirs=(), **kw):
    """The five-panel figure from `tidy_rows` (`appendix_common.paired_comparison_figure` with `figure_options`), saved as `name` when one
    is given (`appendix_common.save_figure`: png, pdf and svg into `out_dirs`, default R.FIGS, then every APPENDIX_FIGS_MIRROR and
    `extra_dirs` directory, each file carrying the render stamp). `kw` overrides any option."""
    fig = AC.paired_comparison_figure(rows, None, **figure_options(**kw))
    if name:
        AC.save_figure(_stamped(fig), name, dirs=out_dirs, extra_dirs=extra_dirs)
    return fig


def not_drawn(rows: pd.DataFrame) -> list:
    """Every mean the figure leaves out (`appendix_common.not_drawn`, which names each pending reading), then every reading the file
    holds that carries no delta, with its route; empty when every mark is drawn."""
    out = AC.not_drawn(rows, CATEGORIES, "complete", labels={OBJECT: OBJECT_LABEL}, condition_labels=CONDITION_LABELS)
    gap = rows[rows.delta.isna() & rows.run_id.notna()]
    out += [f"{R.EVAL_LABEL.get(r['eval'], r['eval'])} · {CONDITION_LABELS[r.condition]} {r.seed}: not drawn - {r.route}"
            for _, r in gap.iterrows()]
    return out


def mean_marks(rows: pd.DataFrame) -> pd.DataFrame:
    """The ten mean marks (five panels x two corpora) as drawn (`appendix_common.paired_means`, mean_rule "complete"), indexed by
    (category, condition): y, lo, hi, drawn, why."""
    pm = AC.paired_means(rows[list(AC.TIDY_COLUMNS) + list(AC.OPTIONAL_COLUMNS)], CATEGORIES, "complete")
    return pm.set_index(["category", "condition"])[["y", "lo", "hi", "n_seeds", "drawn", "why"]]


# ---- the comparison of the corpora: anchor-independent, over matched seeds ---------------------------------------------------------
def t2_quantile(p: float) -> float:
    """The p quantile of Student's t on 2 degrees of freedom, in closed form: (2p - 1) / sqrt(2 p (1 - p))."""
    return (2.0 * p - 1.0) / math.sqrt(2.0 * p * (1.0 - p))


T2_975 = t2_quantile(0.975)


def t2_interval(diffs) -> tuple:
    """(mean, lo, hi): the mean of the three matched-seed differences and its 95% t interval on 2 degrees of freedom."""
    x = np.asarray(list(diffs), dtype=float)
    if len(x) != 3:
        raise ValueError(f"{len(x)} matched-seed differences, not 3")
    m, half = float(x.mean()), T2_975 * float(x.std(ddof=1)) / math.sqrt(len(x))
    return m, m - half, m + half


def readings(frame: pd.DataFrame) -> pd.DataFrame:
    """Every drawn reading of `lane_frame` in the panel's unit (`y` = the value x SCALE: ASR, label mean or accuracy, x 100), with k, n and
    its interval, one row per (condition, seed, eval)."""
    x = frame[frame.role.eq("drawn")].copy()
    sc = x.metric_id.map(R.SCALE).fillna(1.0)
    x["y"], x["y_lo"], x["y_hi"] = x.value * sc, x.ci_lo * sc, x.ci_hi * sc
    return x[["condition", "seed", "eval_id", "run_id", "k", "n", "y", "y_lo", "y_hi", "token_cap_tokens"]].reset_index(drop=True)


def corpus_difference(frame: pd.DataFrame, evals, sign: int = -1):
    """Qwen - Opus in the panel's oriented unit (+ = the Qwen-written corpus better) over the three MATCHED seed pairs, the evals weighted
    equally: per seed, the mean over `evals` of sign x (y_Qwen - y_Opus) - for an attack-success panel ASR_Opus - ASR_Qwen - and its 95%
    t(2) interval. Read from the readings themselves, so the anchor cancels exactly: a seed's two runs can sit on different rungs and so
    on different anchors, which the drawn marks carry and this does not. (mean, lo, hi, per-seed tuple), or None until every reading is in."""
    rd = readings(frame)
    key = {(r.condition, str(r.seed), r.eval_id): float(r.y) for r in rd.itertuples()}
    diffs = []
    for s in SEEDS:
        try:
            diffs.append(float(np.mean([sign * (key[("qwen_written", s, e)] - key[("opus_written", s, e)]) for e in evals])))
        except KeyError:
            return None
    m, lo, hi = t2_interval(diffs)
    return m, lo, hi, tuple(diffs)


def verdict(frame: pd.DataFrame) -> dict:
    """The static panel's pre-fixed verdict and its declared companion: {"pooled": (mean, lo, hi, seeds), "pooled_no_renellm": ...} in
    defense-success pp (`corpus_difference` over VERDICT and DECLARED)."""
    return {"pooled": corpus_difference(frame, VERDICT), "pooled_no_renellm": corpus_difference(frame, DECLARED)}


def gsm8k_coverage(data: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Per GSM8K reading: condition, seed, k (correct), n (scored items: GSM8K_UNSCORED), the item set's 200, the accuracy on the scored
    items and the all-200 sensitivity (an unscored item counted wrong), in percent."""
    total = R.CTLSTOP_EVALS["gsm8k"]["n_items"]
    x = frame[frame.role.eq("drawn") & frame.eval_id.eq("gsm8k")]
    out = [dict(condition=r.condition, seed=str(r.seed), k=int(r.k), n=int(r.n), items=total, acc=100.0 * r.k / r.n if r.n else np.nan,
                acc_all=100.0 * r.k / total) for r in x.itertuples()]
    return pd.DataFrame(out, columns=["condition", "seed", "k", "n", "items", "acc", "acc_all"])


def drift(frame: pd.DataFrame, gen_d: pd.DataFrame, d_ref: pd.DataFrame) -> list:
    """{eval, cap, the lane's unsteered reading, the pooled anchor, its runs, their difference, cap_mismatch} per static attack: the lane's
    own unsteered run (the file's drift row, at its final rung) against the anchor the figure differences that eval's points at the same
    cap against - the lane's anchored rows at that cap, else Figure 11's fitted rows of the class in `d_ref`. Never an anchor."""
    out = []
    dr = frame[frame.role.eq("drift_anchor")]
    fitted = d_ref[d_ref.object_label.isin(R.CTLSTOP_BOUND_SLOTS) & d_ref.d_y.notna()] if d_ref is not None and len(d_ref) else d_ref
    for e in STATIC:
        a = dr[dr.eval_id.eq(e)]
        if not len(a):
            continue
        a = a.iloc[0]
        cap = float(a.token_cap_tokens)
        y = float(a.value) * R.SCALE.get(a.metric_id, 1.0)
        peers = gen_d[gen_d.eval_id.eq(e) & pd.to_numeric(gen_d.token_cap_tokens, errors="coerce").eq(cap) & gen_d.d_y.notna()] if len(gen_d) else gen_d
        if not len(peers) and fitted is not None and len(fitted):
            peers = fitted[fitted.eval_id.eq(e) & fitted.item_set_id.eq(a.item_set_id)
                           & pd.to_numeric(fitted.token_cap_tokens, errors="coerce").isin(R.cap_class(e, cap))]
        if not len(peers):
            out.append(dict(eval_id=e, cap=cap, rerun=y, anchor=np.nan, anchor_runs=0, drift=np.nan, cap_mismatch=None))
            continue
        p = peers.iloc[0]
        out.append(dict(eval_id=e, cap=cap, rerun=y, anchor=float(p.anchor_value), anchor_runs=int(p.anchor_draws),
                        drift=y - float(p.anchor_value), cap_mismatch=bool(p.cap_mismatch)))
    return out


def _pp(v: float, digits: int = 1) -> str:
    return f"{v:+.{digits}f}"


def summary_text(data: dict, gen_d: pd.DataFrame, rows: pd.DataFrame, d_ref: pd.DataFrame, threshold: float | None = None) -> str:
    """The lines printed under the figure: the static panel's pre-fixed verdict, one headline per panel (each corpus's drawn mean mark
    with its interval, and Qwen - Opus over matched seeds, descriptive on every panel but the static one), GSM8K's coverage with the
    all-200 sensitivity, and the drift line of the lane's own unsteered static runs against the pooled anchors."""
    t = R.SR_THRESHOLD if threshold is None else float(threshold)
    frame = lane_frame(data, threshold)
    mm = mean_marks(rows)
    lines = []
    v = verdict(frame)
    if v["pooled"] is not None:
        m, lo, hi, seeds = v["pooled"]
        d2 = v["pooled_no_renellm"]
        lines.append(f"VERDICT (pre-fixed: the static panel's four attacks pooled, Qwen - Opus over the three matched seed pairs, t on 2 df; "
                     f"anchor-independent, a goal broken when its best reply scores > {t:g}): {m:+.2f} pp defense success, 95% {lo:+.2f} to "
                     f"{hi:+.2f} (in attack success {-m:+.2f}, {-hi:+.2f} to {-lo:+.2f}; per seed pair "
                     + ", ".join(f"{x:+.1f}" for x in seeds) + f"); without ReNeLLM (declared) {d2[0]:+.2f}, {d2[1]:+.2f} to {d2[2]:+.2f}")
    else:
        lines.append("VERDICT: not computed - a static reading is not in the data")
    lines.append(PANEL_NOTE)
    for cat, (evals, sign, _ylabel) in CATEGORIES.items():
        parts = []
        for c in CONDITIONS:
            r = mm.loc[(cat, c)] if (cat, c) in mm.index else None
            parts.append(f"{SHORT[c]}-written {r.y:+.1f} [{r.lo:+.1f}, {r.hi:+.1f}]" if r is not None and bool(r.drawn)
                         else f"{SHORT[c]}-written not drawn")
        d = corpus_difference(frame, evals, sign)
        if d is None:
            n_in = int(rows[(rows.category == cat) & rows.delta.notna()].shape[0])
            cmp_ = f"Qwen - Opus not yet computed ({n_in} of {len(evals) * 6} readings in)"
        else:
            cmp_ = (f"Qwen - Opus {d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}] over matched seeds, t(2), "
                    + ("the pre-fixed verdict" if cat == "static" else "descriptive"))
        unit = "pp" if cat != "capability" else "pp accuracy"
        lines.append(f"{cat} ({', '.join(R.EVAL_LABEL.get(e, e) for e in evals)}; mean marks, {unit}, + is better): " + "; ".join(parts) + f"; {cmp_}")
    cov = gsm8k_coverage(data, frame)
    if len(cov):
        txt = "; ".join(f"{SHORT[r.condition]} {r.seed} {r.k}/{r.n} scored of {r.items} ({r.acc:.1f}%, all-200 {r.acc_all:.1f}%)" for r in cov.itertuples())
        sens = ""
        if len(cov) == 6:
            for col, label in (("acc", "on scored items"), ("acc_all", "all 200")):
                key = {(r.condition, r.seed): getattr(r, col) for r in cov.itertuples()}
                dd = t2_interval([key[("qwen_written", s)] - key[("opus_written", s)] for s in SEEDS])
                sens += f"; Qwen - Opus {label} {dd[0]:+.1f} [{dd[1]:+.1f}, {dd[2]:+.1f}]"
        lines.append(f"GSM8K coverage (thinking on; {GSM8K_UNSCORED}): {txt}{sens} (descriptive)")
    else:
        lines.append("GSM8K coverage: no GSM8K reading in the data yet")
    dl = drift(frame, gen_d, d_ref)
    if dl:
        lines.append("drift (the lane's own unsteered static runs, never an anchor, against the pooled anchor its points are differenced "
                     "against): " + "; ".join(
                         f"{R.EVAL_LABEL.get(x['eval_id'], x['eval_id'])} {x['rerun']:.1f}% at {x['cap']:,.0f} against "
                         + (f"{x['anchor']:.1f}% ({x['anchor_runs']} run{'s' if x['anchor_runs'] != 1 else ''}"
                            + (", another cap: no unsteered run of the static panel at this one" if x["cap_mismatch"] else "")
                            + f"), {x['drift']:+.1f} pp" if pd.notna(x["anchor"]) else "no anchor of its class") for x in dl))
    pend = data.get("pending") or []
    if pend:
        by_eval = pd.Series([p["eval_id"] for p in pend]).value_counts()
        lines.append(f"{len(pend)} reading(s) not yet in the data, drawn as gaps: "
                     + "; ".join(f"{R.EVAL_LABEL.get(e, e)} {by_eval[e]}" for e in EVALS if e in by_eval.index))
    return "\n".join(lines)


def anchor_cap(rows: pd.DataFrame) -> pd.Series:
    """Per mark, how its anchor's cap stands to the cap read, as the main figures' tables flag it (no glyph on the canvas): `same`; `other`
    where the anchor is the eval's pooled unsteered runs at a cap outside the reading's cap class (`cap_mismatch`: a static reading on a
    rung no unsteered run of the eval was read at, as Figure 11's fitted points of that class are differenced); on over-refusal, whose
    anchor is the one Figure 11's fitted rows of the seed take and gives up the cap match by design, the route the anchored row took it
    by (`anchor_binding`): `bound` (the run the registry binds to them) or `pooled` (the runs pooled for them where none is bound);
    empty where the mark has no delta. Raises where an over-refusal mark with a delta carries neither: the route is the row's, never
    assumed."""
    has = rows.delta.notna().to_numpy()
    shared = rows["eval"].isin(R.SHARED_FRAME_ANCHOR).to_numpy()
    flag = rows.cap_mismatch.fillna(False).astype(bool).to_numpy()
    how = (rows["anchor_binding"] if "anchor_binding" in rows.columns else pd.Series(None, index=rows.index, dtype=object))
    how = how.where(how.notna(), "").astype(str).to_numpy()
    bad = has & shared & ~np.isin(how, [R.CTLSTOP_ANCHOR_BOUND, R.CTLSTOP_ANCHOR_POOLED])
    if bad.any():
        r = rows[bad].iloc[0]
        raise ValueError(f"{int(bad.sum())} over-refusal mark(s) carry no anchor binding (bound or pooled), e.g. {r['eval']} "
                         f"{r.condition} {r.seed}: anchor them with `anchor` (registry_nb.ctlstop_anchored names it)")
    out = np.where(shared, how, np.where(flag, "other", "same"))
    return pd.Series(np.where(has, out, ""), index=rows.index, dtype=object)


def marks_table(rows: pd.DataFrame) -> pd.DataFrame:
    """The per-seed marks for printing: panel, eval, corpus, seed, the reading and its anchor (the panel's unit), the change as
    drawn (oriented by the category's sign, + is better) and its 95% interval, the cap read, and how the anchor's cap stands to it
    (`anchor_cap`)."""
    t = AC.tidy(rows[list(AC.TIDY_COLUMNS) + ["n", "open"]], CATEGORIES)
    out = rows.assign(eval=rows["eval"].map(R.tick), corpus=rows.condition.map(SHORT), drawn=t.y.values, lo=t.lo.values, hi=t.hi.values,
                      xcap=anchor_cap(rows).values)
    return out[["category", "eval", "corpus", "seed", "y", "anchor_value", "drawn", "lo", "hi", "cap", "xcap"]].rename(
        columns={"y": "reading", "anchor_value": "anchor", "drawn": "Δ (+ better)", "lo": "95% lo", "hi": "95% hi",
                 "xcap": "anchor cap"}).round(1).reset_index(drop=True)


def provenance_line(data: dict, source: str) -> str:
    """One line naming what the figure reads: the file, how this render checked it, its payload digest, the judges, the run counts, the
    registry head it was written at and the static comparison filing it agrees with. Call it after `load(source)` succeeded."""
    prov = data["provenance"]
    c = prov["counts"]
    how = ("live render: derived again from the registry and equal to the file" if source == "live"
           else "snapshot render: the file alone")
    return (f"generator comparison: {c['drawn']} readings of {len(data['objects'])} objects ({c['pending']} pending), {c['drift_anchor']} "
            f"unsteered drift runs and {c['rung']} rungs, read from {SIDECAR_REL} ({how}); payload sha256 {prov['payload_sha256']}; judges "
            f"{lane_fixed_text()}; written at registry head {prov['registry_head']}; static per-cell counts at > {G.FILING_THRESHOLD:g} "
            f"equal the comparison filing {G.ANALYSIS_ID}; StrongREJECT threshold {R.SR_THRESHOLD:g} (strict); anchors: the control-pool "
            f"appendix's (Figure 11's) pooled unsteered runs, per condition")


def lane_fixed_text() -> str:
    """The judges as the render stamp names them (`registry_nb.register_lane_fixed(LANE_FIGURES, lane_fixed_text())`): the control-pool
    appendix's, whose policy every reading of the file is held to."""
    return R.ctlstop_lane_fixed_text()
