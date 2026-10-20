#!/usr/bin/env python
"""HPO selection guard — is a fold's selected point INTERIOR to its search,
and did it actually WIN, or merely sit lowest on a flat objective?

Shared by scripts/tune_arm.py analyze (the artefact producer) and the fold
writers (e.g. ws3_make_fold.py) so both sides file the SAME facts
from the same code. Pure python — no torch, no repo imports — so a fold
writer outside the repo can load it by path.

Why (2026-09-07): GLM-4.6V-Flash's gated (psr_ce_all) arm diverged at full
scale. Its pooled r1+2 quadratic was not positive-definite, so the fold fell
through to the RAW argmin trial — lr 5.32e-3 with max_grad_norm 89.93
(effectively unclipped; that model's grad norm runs 0.25-0.5) — which "won"
a FLAT objective (held-out CE ratio 0.9146-0.9210 across three decades of
lr) by 0.001-0.007, inside noise. Nothing in the fold record said any of
this. Standing rule since: alert when a fold's selected point is not
interior to its search box, and file the objective's spread next to the
winner so a flat objective is visible without re-deriving it.

The guard is evaluated on the point the fold actually SHIPS: the fitted
argmin when the quadratic is positive-definite and interior, else the
best-raw-trial fall-through. Per continuous dimension (lr, max_grad_norm)
it files three criteria, each an alarm when it fires:

  (a) box_edge            log-space position in the DECLARED (union) box
                          within EDGE_TOL of 0 or 1 — the analyze edge rule.
  (b) sampled_extreme     at or beyond the min / max value the usable trials
                          actually sampled — NO trial lies beyond it.
  (c) no_evidence_beyond  trials DO lie beyond it on a side, but none is
                          worse than the selected point by more than the
                          noise — the objective is flat out to the sampled
                          edge on that side, so nothing in the search says
                          the optimum is not further out. (When no trial
                          lies beyond, (b) carries the fact; (c) is silent.)
                          Alarmed only for a RAW-trial selection: no model
                          located that point, so out to the sampled edge
                          nothing distinguishes it from its neighbours. A
                          fitted argmin pools every trial's curvature, so
                          the same fact is FILED (dims.<axis>.
                          flat_to_sampled_edge, the FLAT sentence), not
                          alarmed — on the healthy qwen / nemotron psr
                          lanes it fires on one shallow side each, which is
                          the fit being flat there, not a boundary.

Discrete axis (warmup_frac): position in its level set. Alarmed only at an
end of a >=3-level set — a 2-level factor has NO interior by design, so an
alarm there would fire on every fold and train reflexive --allow-alarmed.
Recorded, not alarmed. Fit-level alarms: a non-positive-definite quadratic
(the raw-argmin fall-through is unguided) and a fitted argmin on the box
edge (the same fall-through, via the other door).

Noise: sigma_log_y = sqrt(SSR / (n - p)), the dof-corrected RMS residual of
the pooled quadratic fit in log y. It is the only replicate-free estimate
of unexplained variation the search offers (no trial is repeated), it is
unbiased under the model, and it is CONSERVATIVE for these alarms — lack of
fit inflates it, so a poor (non-PD) fit widens "within noise" and makes the
guard louder, never quieter. Converted to objective units at the selected
point (d ln y = dy / y). The robust MAD scale and the top-3 / top-5 trial
spreads ride along as context, never as the threshold.

Flatness is FILED, not alarmed: with a positive-definite interior fit a
flat top is a fine place to ship from, every raw-argmin path under
flatness is already alarmed (non-PD / on-edge), and flatness that reaches
the sampled edge alarms through (c). The record always says in words
whether the objective is flat and over how many trials / decades.
"""

from __future__ import annotations

import datetime as _dt
import math
import statistics

#: Log-space fraction of the declared box that counts as "on the edge" —
#: shared with tune_arm.analyze's recenter rule (which imports it from here).
EDGE_TOL = 0.02

#: The continuous search axes every arm sweeps (log-space boxes).
CONT_DIMS = ("lr", "max_grad_norm")

#: Bumped when the record's shape changes.
GUARD_VERSION = 1

CRITERIA = {
    "box_edge": ("log-space position of the selected value in the DECLARED "
                 f"(union) box lies within EDGE_TOL={EDGE_TOL} of 0 or 1"),
    "sampled_extreme": ("the selected value is at or beyond the min / max "
                        "value the usable trials sampled on that axis — no "
                        "trial lies beyond it"),
    "no_evidence_beyond": ("trials lie beyond the selected value on that "
                           "side, but none is worse than the selected point "
                           "by more than the noise — the objective is flat "
                           "out to the sampled edge; nothing says the optimum "
                           "is not further out (a 1-D marginal check). "
                           "ALARMED for a raw-trial selection (no model "
                           "located the point); for a fitted argmin the same "
                           "fact is filed as dims.<axis>.flat_to_sampled_edge "
                           "and named in the FLAT sentence, not alarmed"),
    "discrete_extreme": ("the selected level is an end of a >=3-level "
                         "discrete axis (a 2-level factor has no interior by "
                         "design and is recorded, not alarmed); for a NAMED "
                         "discrete axis (lora_rank, lora_modules — the "
                         "architecture search) the selected level is an END "
                         "of an ordered menu of >= 2 levels: rank 4 / rank 64, "
                         "the narrowest / widest module set, either end of a "
                         "probe-reduced two-level menu"),
    "fit_not_positive_definite": ("the pooled quadratic's continuous Hessian "
                                  "is not positive-definite — there is no "
                                  "fitted interior minimum and the fold fell "
                                  "through to the RAW argmin trial, an "
                                  "unguided pick"),
    "fitted_argmin_on_edge": ("the quadratic is positive-definite but its "
                              "argmin sits on / outside the declared box, so "
                              "the fold fell through to the RAW argmin trial"),
    "noise": ("sigma_log_y = sqrt(SSR/(n-p)), the dof-corrected RMS residual "
              "of the pooled quadratic fit in log y, converted to objective "
              "units at the selected point (noise = sigma_log_y * y)"),
}


# --------------------------------------------------------------------------
# Small pure helpers (also used by tune_arm directly)
# --------------------------------------------------------------------------

def alarm_code(alarm: dict) -> str:
    """The ONE rendering of an alarm's identity: ``code``, suffixed
    ``:<dim>`` when the alarm is per-dimension (``box_edge:lr``), bare
    otherwise (``fit_not_positive_definite``). Every reader that compares,
    lists or prints alarm codes — analyze's selection block, the
    confirmation stamp, the next-round driver's reproduction check, the
    fold-time refusal in the config generator — renders through here, so
    the sets they compare are spelled identically."""
    dim = alarm.get("dim")
    return f"{alarm.get('code')}:{dim}" if dim else str(alarm.get("code"))


def trial_id(row: dict) -> str:
    """'r<round>_t<index>' for a trial row (analyze's naming), '?' parts when
    a field is missing (synthetic rows)."""
    rnd = row.get("round", "?")
    idx = row.get("index")
    return f"r{rnd}_t{int(idx):02d}" if idx is not None else f"r{rnd}_t??"


def box_position(value: float, lo: float, hi: float) -> float:
    """Log-space position of ``value`` in [lo, hi]: 0 at lo, 1 at hi
    (outside the box lands outside [0, 1])."""
    lo, hi = float(lo), float(hi)
    width = math.log(hi) - math.log(lo)
    if width <= 0:
        raise ValueError(f"degenerate box [{lo}, {hi}]")
    return (math.log(float(value)) - math.log(lo)) / width


def is_fixed_axis(lo, hi) -> bool:
    """A declared box side of ZERO log-width, ``[c, c]``: the design held
    that axis CONSTANT (the sdfbase-llama8b LR-only search — the clip
    is a recon-derived per-arm constant stamped on every row, so its
    box is ``[c, c]``). A fixed axis was never searched: it has no
    interior and no edge, every usable trial sits at its one value,
    and the guard RECORDS it as fixed instead of alarming on it (or
    dividing by its zero width)."""
    return float(lo) == float(hi)


def box_edge_positions(point: dict, box: dict,
                       edge_tol: float = EDGE_TOL) -> tuple[bool, dict]:
    """tune_arm's edge check: (on_edge, {dim: log-position}) over the
    continuous axes; on the edge when any position is within ``edge_tol``
    of 0 or 1 (or beyond). A FIXED axis (``is_fixed_axis``) reports the
    position ``None`` and can never put the point on an edge."""
    edges = {}
    on_edge = False
    for k in CONT_DIMS:
        if is_fixed_axis(box[k][0], box[k][1]):
            edges[k] = None
            continue
        t = box_position(point[k], box[k][0], box[k][1])
        edges[k] = t
        if t <= edge_tol or t >= 1.0 - edge_tol:
            on_edge = True
    return on_edge, edges


def residual_scale(resid_log_y: list[float], n_params: int) -> dict:
    """Noise statistics of a fit's residuals in log y. ``sigma_log_y`` is the
    dof-corrected RMS (sqrt(SSR/(n-p)); None when n <= p); ``rms_log_y`` the
    plain RMS; ``mad_log_y`` the robust 1.4826*MAD scale (context only)."""
    res = [float(x) for x in resid_log_y]
    n = len(res)
    ssr = sum(x * x for x in res)
    out = {"resid_n": n, "resid_p": int(n_params),
           "rms_log_y": math.sqrt(ssr / n) if n else None,
           "sigma_log_y": (math.sqrt(ssr / (n - n_params))
                           if n > n_params else None),
           "mad_log_y": None}
    if n:
        med = statistics.median(res)
        out["mad_log_y"] = 1.4826 * statistics.median(abs(x - med)
                                                      for x in res)
    return out


def _same(a: float, b: float, rel: float = 1e-9) -> bool:
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=0.0)


def _match_row(selected: dict, rows: list[dict]) -> dict:
    """The usable row a raw selection names — by (round, index) when the
    selection carries them, else by its (lr, max_grad_norm, warmup_frac)."""
    if "index" in selected and "round" in selected:
        for r in rows:
            if (r.get("round") == selected["round"]
                    and int(r["index"]) == int(selected["index"])):
                return r
    for r in rows:
        if all(_same(r[k], selected[k])
               for k in ("lr", "max_grad_norm", "warmup_frac")):
            return r
    raise ValueError(
        "a raw-trial selection must be one of the usable rows; got "
        f"{ {k: selected.get(k) for k in ('round', 'index', 'lr', 'max_grad_norm', 'warmup_frac')} }")


def _decades(lo: float, hi: float) -> float:
    return math.log10(hi / lo) if lo > 0 and hi > 0 else float("nan")


def _trial_brief(row: dict, y_key: str, baseline: float | None) -> dict:
    y = float(row[y_key])
    d = {"trial": trial_id(row), "lr": float(row["lr"]),
         "max_grad_norm": float(row["max_grad_norm"]),
         "warmup_frac": float(row["warmup_frac"]), "y": y}
    if baseline:
        d["ratio"] = y / baseline
    # The trial's clip-activation rate (the share of its logged optimizer
    # steps whose pre-clip grad norm exceeded its own clip — analyze
    # computes it from the trial's metrics.jsonl and files it on the row).
    # Present only when the row carries it, so every record written before
    # the column existed is byte-identical.
    if row.get("clip_activation_rate") is not None:
        d["clip_activation_rate"] = float(row["clip_activation_rate"])
    return d


# --------------------------------------------------------------------------
# The guard
# --------------------------------------------------------------------------

def guard_selection(*, selected: dict, selected_from: str, rows: list[dict],
                    box: dict, levels, fit: dict,
                    y_key: str = "best_heldout_loss",
                    sigma_log_y: float | None,
                    fitted_min_log_y: float | None = None,
                    baseline: float | None = None,
                    fall_through: dict | None = None,
                    edge_tol: float = EDGE_TOL,
                    discrete_axes: dict | None = None) -> dict:
    """File the selection-guard record for one arm's chosen point.

    selected        {lr, max_grad_norm, warmup_frac, ...} — the point the
                    fold ships (a fitted argmin, or a raw trial row).
    selected_from   "fitted_argmin" | "best_raw_trial".
    rows            the USABLE trials (analyze's post-exclusion rows), each
                    with lr / max_grad_norm / warmup_frac / y_key (+ round,
                    index when real).
    box             the declared UNION box {dim: [lo, hi]} the trials
                    explored (analyze's ``box``).
    levels          the discrete warmup levels of the design.
    fit             the quadratic fit dict (positive_definite, hessian_eigs).
    sigma_log_y     noise in log y (residual_scale()['sigma_log_y']); None
                    disables the noise-relative facts (they file as None).
    fitted_min_log_y  REQUIRED for a fitted argmin: the fit's minimum, the
                    objective at the selected point.
    baseline        untrained objective (e.g. held-out CE before training)
                    for ratio columns; None omits them.
    fall_through    {"reason": "fit_not_positive_definite" |
                    "fitted_argmin_on_edge", ...} when the raw trial was
                    taken because the fit gave no usable argmin.
    discrete_axes   the NAMED discrete search axes of the design (the LoRA
                    architecture search): ``{name: {"levels": [ordered ...],
                    "selected": value}}`` — e.g. ``lora_rank`` over the
                    ranks the design drew, ``lora_modules`` over the module
                    sets in trainable-parameter order. ``discrete_extreme``
                    is filed PER NAMED AXIS when the selected level is an
                    END of an ordered menu with >= 2 levels (a probe-
                    reduced two-level menu must still alarm at either end).
                    The warm-up factor keeps its own handling BY NAME
                    (``warmup_frac`` below: a DESIGN factor whose 2-level
                    form is recorded, not alarmed) and is never passed
                    here. None (the default) leaves every record written
                    before the axis existed byte-identical.

    Returns a JSON-able dict; ``alarmed`` / ``alarms`` are the verdict,
    ``dims`` / ``spread`` the facts, ``text`` the words.
    """
    if selected_from not in ("fitted_argmin", "best_raw_trial"):
        raise ValueError(f"selected_from must be 'fitted_argmin' or "
                         f"'best_raw_trial', got {selected_from!r}")
    rows = [r for r in rows]
    if not rows:
        raise ValueError("the selection guard needs at least one usable "
                         "trial")
    order = sorted(rows, key=lambda r: float(r[y_key]))
    n = len(order)

    # -- the objective at the selected point ---------------------------------
    sel_row = None
    if selected_from == "fitted_argmin":
        if fitted_min_log_y is None:
            raise ValueError("a fitted-argmin selection needs "
                             "fitted_min_log_y (the objective at the point)")
        y_sel = math.exp(float(fitted_min_log_y))
        y_sel_kind = "fitted minimum (the quadratic's value at its argmin)"
    else:
        sel_row = _match_row(selected, rows)
        y_sel = float(sel_row[y_key])
        y_sel_kind = f"trial {trial_id(sel_row)}'s observed {y_key}"
    noise = (float(sigma_log_y) * y_sel) if sigma_log_y is not None else None

    alarms: list[dict] = []

    def alarm(code: str, dim: str | None, text: str, **extra) -> None:
        alarms.append({"code": code, "dim": dim, "text": text, **extra})

    # -- fit-level -------------------------------------------------------------
    pd = bool(fit.get("positive_definite"))
    eigs = fit.get("hessian_eigs")
    if not pd:
        alarm("fit_not_positive_definite", None,
              "the pooled quadratic is NOT positive-definite (Hessian eigs "
              f"{eigs}) — no fitted interior minimum exists; the selection "
              "is the RAW argmin trial, an unguided pick")
    reason = (fall_through or {}).get("reason")
    if reason == "fitted_argmin_on_edge":
        fa = (fall_through or {}).get("fitted_argmin") or {}
        ep = (fall_through or {}).get("edge_pos") or {}
        alarm("fitted_argmin_on_edge", None,
              "the quadratic is positive-definite but its argmin sits on / "
              "outside the declared box (log-positions "
              f"{ {k: (round(float(v), 3) if v is not None else None) for k, v in ep.items()} }, argmin "
              f"{ {k: float(v) for k, v in fa.items() if k in CONT_DIMS} }); "
              "the selection is the RAW argmin trial fall-through")

    # -- continuous dimensions ---------------------------------------------------
    dims: dict = {}
    flat_edges: list[tuple[str, str, str]] = []  # (dim, side, text), fitted
    for k in CONT_DIMS:
        v = float(selected[k])
        lo, hi = float(box[k][0]), float(box[k][1])
        vals = sorted(float(r[k]) for r in rows)
        if is_fixed_axis(lo, hi):
            # Held constant by the design (the clip): no position, no
            # edge, no evidence beyond — recorded, never alarmed. What IS
            # alarmed is a selection or a row that does not sit at the
            # constant: then the declared box is lying about the design.
            dims[k] = {
                "value": v, "box": [lo, hi], "fixed": True,
                "box_pos": None, "on_box_edge": False,
                "sampled_min": vals[0], "sampled_max": vals[-1],
                "n_sampled": len(vals), "rank_ascending": None,
                "n_below": 0, "n_above": 0, "sampled_extreme": None,
                "sampled_pos": None, "evidence_beyond": {},
                "flat_to_sampled_edge": [],
                "note": (f"{k} is a FIXED axis of this design (box [{lo:.4g}, "
                         f"{hi:.4g}] has zero width): held at the declared "
                         "constant, not searched — no interiority verdict "
                         "exists or is owed along it"),
            }
            off = [x for x in vals if not _same(x, lo)]
            if not _same(v, lo) or off:
                alarm("fixed_axis_mismatch", k,
                      f"{k} is declared FIXED at {lo:.4g} but the selected "
                      f"value is {v:.4g} and {len(off)} usable trial(s) sit "
                      "elsewhere — the declared box does not describe the "
                      "design that ran")
            continue
        t = box_position(v, lo, hi)
        on_edge = t <= edge_tol or t >= 1.0 - edge_tol
        below = [r for r in rows if float(r[k]) < v and not _same(r[k], v)]
        above = [r for r in rows if float(r[k]) > v and not _same(r[k], v)]
        extreme = None
        if not below and not above:
            extreme = "min+max"
        elif not below:
            extreme = "min"
        elif not above:
            extreme = "max"
        sampled_pos = (box_position(v, vals[0], vals[-1])
                       if vals[-1] > vals[0] and vals[0] > 0 else None)
        evidence: dict = {}
        for side, beyond in (("low", below), ("high", above)):
            rec: dict = {"n_beyond": len(beyond)}
            if beyond:
                nearest = min(beyond, key=lambda r: abs(math.log(float(r[k]))
                                                        - math.log(v)))
                rec["nearest_beyond"] = _trial_brief(nearest, y_key, baseline)
                if noise is not None:
                    worse = [r for r in beyond
                             if float(r[y_key]) > y_sel + noise]
                    rec["n_worse_than_selected_by_more_than_noise"] = len(worse)
                    rec["has_evidence"] = bool(worse)
                else:
                    rec["n_worse_than_selected_by_more_than_noise"] = None
                    rec["has_evidence"] = None
            else:
                rec["has_evidence"] = False
                rec["n_worse_than_selected_by_more_than_noise"] = 0
            evidence[side] = rec
        # (c): sides where trials exist beyond the selected value but none is
        # worse than it by more than the noise — flat out to the sampled edge.
        flat_sides = [side for side, rec in evidence.items()
                      if rec["n_beyond"] and rec.get("has_evidence") is False]
        dims[k] = {
            "value": v, "box": [lo, hi], "box_pos": t,
            "on_box_edge": on_edge,
            "sampled_min": vals[0], "sampled_max": vals[-1],
            "n_sampled": len(vals), "rank_ascending": len(below) + 1,
            "n_below": len(below), "n_above": len(above),
            "sampled_extreme": extreme, "sampled_pos": sampled_pos,
            "evidence_beyond": evidence,
            "flat_to_sampled_edge": flat_sides,
        }
        if on_edge:
            alarm("box_edge", k,
                  f"{k}={v:.4g} sits at log-position {t:.3f} of the declared "
                  f"box [{lo:.4g}, {hi:.4g}] (edge tol {edge_tol})")
        if extreme is not None:
            alarm("sampled_extreme", k,
                  f"{k}={v:.4g} is the {extreme} of the {len(vals)} sampled "
                  f"values [{vals[0]:.4g}, {vals[-1]:.4g}] — no trial lies "
                  "beyond it" + (" on either side" if extreme == "min+max"
                                 else ""))
        for side in flat_sides:
            rec = evidence[side]
            nb = rec["nearest_beyond"]
            msg = (f"{k}: {rec['n_beyond']} trial(s) lie {side}-side of "
                   f"{v:.4g} and NONE is worse than the selected point "
                   f"(y={y_sel:.4f}) by more than noise {noise:.4f} — "
                   f"nearest {nb['trial']} at {k}={nb[k]:.4g} "
                   f"y={nb['y']:.4f}; the objective is flat out to the "
                   f"sampled {'min' if side == 'low' else 'max'} on this axis")
            if selected_from == "best_raw_trial":
                # No model located this point: it is one noisy observation,
                # and out to the sampled edge nothing distinguishes it from
                # its neighbours — a boundary hit in the practical sense.
                alarm("no_evidence_beyond", k, msg, side=side)
            else:
                # A fitted argmin pools every trial's curvature; a shallow
                # side within noise is the fit being flat there, filed as a
                # fact (spread.flat_to_sampled_edge + the FLAT sentence).
                flat_edges.append((k, side, msg))

    # -- discrete axis --------------------------------------------------------
    lv = sorted(float(x) for x in levels) if levels else []
    wsel = float(selected["warmup_frac"])
    idx = next((i for i, x in enumerate(lv) if _same(x, wsel)), None)
    at_extreme = (idx in (0, len(lv) - 1)) if idx is not None else None
    wrec: dict = {"value": wsel, "levels": lv, "level_index": idx,
                  "n_levels": len(lv), "at_extreme": at_extreme,
                  "alarmed": False, "note": None}
    if idx is None:
        wrec["note"] = "selected warmup_frac is not one of the design levels"
    elif at_extreme and len(lv) <= 2:
        wrec["note"] = ("2-level factor: no interior exists by design; "
                        "recorded, not alarmed")
    elif at_extreme:
        wrec["alarmed"] = True
        alarm("discrete_extreme", "warmup_frac",
              f"warmup_frac={wsel:g} is an end of the {len(lv)}-level set "
              f"{lv}")
    dims["warmup_frac"] = wrec

    # -- NAMED discrete axes (the LoRA architecture search) --------------------
    # Unlike the warm-up factor, a named menu of >= 2 levels alarms at
    # EITHER end whatever its size: rank 4 / rank 64, the narrowest / the
    # widest module set, and a probe-reduced two-level menu alike. The order
    # is the caller's (ranks ascending; module sets by trainable-parameter
    # count).
    named_selected: dict = {}
    for name, spec in (discrete_axes or {}).items():
        if name == "warmup_frac":
            raise ValueError("warmup_frac is the design's warm-up factor and "
                             "keeps its own handling — never a named "
                             "discrete axis")
        if not isinstance(spec, dict) or "levels" not in spec \
                or "selected" not in spec:
            raise ValueError(f"discrete_axes[{name!r}] must be "
                             "{'levels': [...], 'selected': value}")
        lvls = list(spec["levels"])
        if len(set(map(_level_key, lvls))) != len(lvls):
            raise ValueError(f"discrete_axes[{name!r}] levels repeat: {lvls}")
        sel_v = spec["selected"]
        sel_key = _level_key(sel_v)
        pos = next((i for i, x in enumerate(lvls) if _level_key(x) == sel_key),
                   None)
        at_end = (pos in (0, len(lvls) - 1)) if pos is not None else None
        rec: dict = {"value": sel_v, "levels": lvls, "level_index": pos,
                     "n_levels": len(lvls), "at_extreme": at_end,
                     "alarmed": False, "note": None,
                     "order": spec.get("order")}
        named_selected[name] = sel_v
        if pos is None:
            rec["note"] = (f"selected {name} is not one of the design's "
                           "levels")
            alarm("fixed_axis_mismatch", name,
                  f"{name}={sel_v!r} is not one of the design's levels "
                  f"{lvls} — the recorded menu does not describe the "
                  "design that ran")
        elif len(lvls) < 2:
            rec["note"] = "single level: the whole design's value, recorded"
        elif at_end:
            side = "lowest" if pos == 0 else "highest"
            rec["alarmed"] = True
            alarm("discrete_extreme", name,
                  f"{name}={sel_v!r} is the {side} end of the {len(lvls)}-"
                  f"level menu {lvls} — no level lies beyond it on that "
                  "side; the search does not bound the optimum there")
        dims[name] = rec

    # -- spread -------------------------------------------------------------------
    ys = [float(r[y_key]) for r in order]
    best, runner = order[0], (order[1] if n > 1 else None)
    spread: dict = {
        "n_ok": n, "y_key": y_key,
        "min": ys[0], "median": statistics.median(ys), "max": ys[-1],
        "full_spread": ys[-1] - ys[0],
        "baseline": baseline,
        "selected_y": y_sel, "selected_y_kind": y_sel_kind,
        "best_trial": _trial_brief(best, y_key, baseline),
        "runner_up": None,
        "best_other_interior_trial": None,
        "noise": {"sigma_log_y": sigma_log_y, "noise_y": noise,
                  "method": CRITERIA["noise"],
                  "rms_log_y": fit.get("rms_log_y"),
                  "mad_log_y": fit.get("mad_log_y"),
                  "top3_spread": (ys[2] - ys[0]) if n >= 3 else None,
                  "top5_spread": (ys[4] - ys[0]) if n >= 5 else None},
        "within_noise": None,
        "flat": False, "flat_kind": None, "text": None,
    }
    if baseline:
        spread["ratio_min"] = ys[0] / baseline
        spread["ratio_median"] = spread["median"] / baseline
        spread["ratio_max"] = ys[-1] / baseline
        spread["ratio_full_spread"] = spread["full_spread"] / baseline
        spread["selected_ratio"] = y_sel / baseline
    if selected_from == "fitted_argmin":
        spread["fitted_min_below_best_trial"] = ys[0] - y_sel
    if runner is not None:
        ru = _trial_brief(runner, y_key, baseline)
        ru["margin_over_best_trial"] = ys[1] - ys[0]
        ru["margin_over_selected"] = ys[1] - y_sel
        if baseline:
            ru["margin_over_best_trial_ratio"] = (ys[1] - ys[0]) / baseline
        spread["runner_up"] = ru

    def _interior(r: dict) -> bool:
        for k in CONT_DIMS:
            if is_fixed_axis(box[k][0], box[k][1]):
                continue  # not searched: neither interior nor edge
            x = float(r[k])
            tt = box_position(x, box[k][0], box[k][1])
            if tt <= edge_tol or tt >= 1.0 - edge_tol:
                return False
            xs = [float(q[k]) for q in rows]
            if _same(x, min(xs)) or _same(x, max(xs)):
                return False
        return True

    others = [r for r in order if r is not sel_row and _interior(r)]
    if others:
        bo = _trial_brief(others[0], y_key, baseline)
        bo["margin_over_selected"] = float(others[0][y_key]) - y_sel
        bo["definition"] = ("best usable trial, other than the selected one, "
                            "that is interior on every continuous axis by "
                            "criteria (a) and (b)")
        spread["best_other_interior_trial"] = bo

    if noise is not None:
        thr = y_sel + noise
        within = [r for r in order if float(r[y_key]) <= thr]
        span = {}
        for k in CONT_DIMS:
            if is_fixed_axis(box[k][0], box[k][1]):
                continue  # a fixed axis spans nothing
            xs = [float(r[k]) for r in within]
            if xs:
                span[k] = {"min": min(xs), "max": max(xs),
                           "decades": _decades(min(xs), max(xs))}
        span["warmup_frac"] = sorted({float(r["warmup_frac"]) for r in within})
        spread["within_noise"] = {
            "threshold_y": thr, "n": len(within),
            "trials": [trial_id(r) for r in within],
            "spread": ((max(float(r[y_key]) for r in within)
                        - min(float(r[y_key]) for r in within))
                       if within else None),
            "span": span,
        }
        rat = (f" ({spread['full_spread'] / baseline:.4f} as a ratio to "
               f"baseline {baseline:.4f})" if baseline else "")
        edge_note = ""
        if flat_edges:
            edge_note = (". Flat out to the sampled edge on: " + ", ".join(
                f"{k} ({side} side)" for k, side, _ in flat_edges)
                + " — recorded, not alarmed: the fitted argmin pools every "
                "trial's curvature")
        if spread["full_spread"] <= noise:
            spread["flat"], spread["flat_kind"] = True, "full"
            spread["text"] = (
                f"objective flat: full spread {spread['full_spread']:.4f}"
                f"{rat} across all {n} usable trials is within noise "
                f"{noise:.4f}" + edge_note)
        elif len(within) >= 2:
            spread["flat"], spread["flat_kind"] = True, "near_optimum"
            spans = "; ".join(
                f"{k} [{span[k]['min']:.3g}, {span[k]['max']:.3g}] = "
                f"{span[k]['decades']:.1f} decades" for k in CONT_DIMS
                if k in span)
            m = spread["runner_up"]["margin_over_selected"] if runner else 0.0
            if selected_from == "best_raw_trial":
                verdict = (f"the runner-up's margin {m:.4f} is within noise, "
                           "so this search does not resolve the choice among "
                           "them — the raw pick is arbitrary within this set")
            else:
                verdict = (f"the best trial's margin over the runner-up "
                           f"{ys[1] - ys[0]:.4f} is within noise; the fitted "
                           "argmin is model-guided across all trials, and "
                           "this set shows how flat its top is")
            spread["text"] = (
                f"objective flat near the optimum: {len(within)} of {n} "
                f"usable trials lie within noise {noise:.4f} of the "
                f"selected point's objective {y_sel:.4f} (their spread "
                f"{spread['within_noise']['spread']:.4f}; they span {spans}; "
                f"warmup {span['warmup_frac']}); {verdict}. Full spread over "
                f"all trials {spread['full_spread']:.4f}{rat} is NOT within "
                "noise" + edge_note)
        else:
            m = spread["runner_up"]["margin_over_selected"] if runner else None
            spread["text"] = (
                "objective resolved at the noise level: only the selected "
                f"point lies within noise {noise:.4f} of its objective "
                f"{y_sel:.4f}"
                + (f"; runner-up margin {m:.4f}" if m is not None else "")
                + f"; full spread {spread['full_spread']:.4f}{rat}"
                + edge_note)
        spread["flat_to_sampled_edge"] = [f"{k}:{side}"
                                          for k, side, _ in flat_edges]
    else:
        spread["text"] = (
            "noise estimate unavailable (no fit residuals) — full spread "
            f"{spread['full_spread']:.4f}"
            + (f"; runner-up margin {spread['runner_up']['margin_over_selected']:.4f}"
               if runner else ""))

    guard = {
        "version": GUARD_VERSION,
        "selected_from": selected_from,
        "selected": {**{k: float(selected[k]) for k in
                        ("lr", "max_grad_norm", "warmup_frac")},
                     **named_selected},
        "selected_trial": trial_id(sel_row) if sel_row is not None else None,
        "fall_through": fall_through,
        "fit": {"positive_definite": pd, "hessian_eigs": eigs},
        "alarmed": bool(alarms),
        "alarms": alarms,
        "dims": dims,
        "spread": spread,
        "criteria": CRITERIA,
        "acknowledged": None,
    }
    if discrete_axes:
        # The menus the named alarms were judged against, so a reader (the
        # table, hpo_next_round.reproduce_guard on a v2 artefact) can pass
        # the SAME axes back and reproduce the alarm set. Absent on every
        # record written without named axes (byte-identical).
        guard["discrete_axes"] = {
            name: {"levels": list(spec["levels"]), "selected": spec["selected"],
                   **({"order": spec["order"]} if spec.get("order") else {})}
            for name, spec in discrete_axes.items()}
    guard["text"] = alarm_lines(guard)
    return guard


def _level_key(v):
    """A hashable, type-tolerant identity for a discrete level: numbers by
    value (4 == 4.0), a module-set spelled as a list by its sorted tuple,
    anything else by str."""
    if isinstance(v, bool):
        return ("b", v)
    if isinstance(v, (int, float)):
        return ("n", float(v))
    if isinstance(v, (list, tuple)):
        return ("l", tuple(sorted(str(x) for x in v)))
    return ("s", str(v))


def alarm_lines(guard: dict) -> list[str]:
    """The printable verdict: one 'ALARM ...' line per alarm, then the
    flatness sentence, then the headline."""
    lines = []
    for a in guard["alarms"]:
        where = f" [{a['dim']}]" if a.get("dim") else ""
        lines.append(f"ALARM {a['code']}{where}: {a['text']}")
    sp = guard["spread"]
    if sp.get("text"):
        lines.append(("FLAT: " if sp.get("flat") else "SPREAD: ") + sp["text"])
    if guard["alarmed"]:
        codes = sorted({alarm_code(a) for a in guard["alarms"]})
        lines.append(f"SELECTION GUARD: {len(guard['alarms'])} alarm(s) — "
                     + ", ".join(codes))
    else:
        lines.append("SELECTION GUARD: clear (interior on every axis, "
                     "positive-definite fit)")
    return lines


def acknowledge(guard: dict, by: str) -> dict:
    """Mark an alarmed record as SEEN and deliberately shipped (the fold
    writer's --allow-alarmed). The alarms stay in the record verbatim; only
    the acknowledgement is added. No-op on a clear record."""
    if guard.get("alarmed"):
        guard["acknowledged"] = {
            "by": by,
            "at": _dt.datetime.now(_dt.timezone.utc).isoformat(
                timespec="seconds"),
        }
    return guard


def is_acknowledged(guard: dict) -> bool:
    return bool(guard.get("acknowledged"))


def needs_acknowledgement(guard: dict | None) -> bool:
    """True when a record is alarmed and nobody has acknowledged it — the
    condition every consumer (fold writer, config generator) refuses on."""
    return bool(guard) and bool(guard.get("alarmed")) \
        and not is_acknowledged(guard)
