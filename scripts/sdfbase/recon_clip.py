#!/usr/bin/env python
"""recon_clip — the gradient-norm reconnaissance's percentile -> clip rule,
and the per-trial clip-activation rate.

The lane's search sweeps the learning rate ALONE. The gradient clip is
not a search axis: it is a per-arm CONSTANT, derived here from the pre-clip
gradient-norm distribution the reconnaissance bench measured on the real
pool at the production effective batch, and supplied at trial-emission time
(``tune_arm.py emit-trials --clip-const``). Every trial then meets a clip
that binds on a few per cent of steps at most — a spike guard by
construction, never a normaliser (the spec's Clipping section: fits that
clipped on essentially every step were a defect).

The rule (stated once, here)::

    clip = ceil2sf( max over STABLE recon runs of the arm of
                    p95(pre-clip grad norm over the CERTIFIED interval,
                        steps BENCH_WARMUP_STEPS < s <= RECON_STEPS) )

Stable-run ELIGIBILITY is decided BEFORE any quantile is computed: a
diverged run's inflated norms must never set the floor — an absurdly high
clip passes the low-activation-rate check trivially and re-creates the
no-clipping failure the constant exists to prevent. A run is STABLE iff its
bench record says ``status: ok`` with every requested step completed AND
EXACTLY the reconnaissance budget (RECON_STEPS = 150 optimizer steps) run,
its pre-clip norm series complete over the CERTIFIED interval
(``clip_interval``: steps 6-150 — the span the stability windows lie in and
the only span the clip quantile draws from), its loss series finite
throughout, and no blow-up over the REGISTERED step-index windows (stated at the
150-step budget: late-window median grad norm over
steps 101-150 within BLOWUP_NORM_RATIO x the early-window median over steps
6-50; median loss over the last 25 steps, 126-150, not above the median
over the first 25 post-warmup steps, 6-30). The windows are step indices
anchored to the budget, never fractions of whatever the record holds — a
fractional window changed ELIGIBILITY (at 150 steps it compared 6-53 /
103-150 and labelled a run the registered rule calls diverged as stable,
then derived the clip from it). Anything else is DIVERGED (``status:
diverged``, a ``failed`` record whose error names the trainer's divergence
abort, or a blow-up), FAILED (infrastructure), INCOMPLETE (fewer steps than
requested, fewer than the budget — an 11-step smoke record is not a
reconnaissance run whatever it requested — or a norm series torn inside the
certified interval), or OVER_BUDGET (MORE steps than the budget: the
windows certify steps 6-150 only, so a longer record carries a tail the
classifier never examined — a 200-step record, the bench CLI's default,
with norms 0.3 through step 150 and 1000 past it read stable and set
clip = 1000). The quantile is bounded STRUCTURALLY as well
(``clip_interval_values`` selects by step index), so no observation outside
the certified interval can reach it whatever record shape arrives.
Ineligible telemetry is kept as DIAGNOSTIC ONLY and never enters the clip.
A MISSING record is refused outright — it is not an outcome.

Three further criteria close what the registered windows cannot see,
because they compare MEDIANS and a median is robust to exactly the
pathology a too-high learning rate produces. The first real box-top PSR
record (lr 1.4e-2) had certified-interval norms p50 3.46 / p95 7,391 / max
31,260 — a heavy-tailed, spiking run — an early window that was an
explosion (p50 1,629) followed by a collapse (late p50 1.33), and a loss
that "decreased" from 11 to 7 after starting at 2.4; the windows called it
stable and the rule emitted clip 7,400. So, evaluated on every ok, complete,
finite record and each filed as a named reason: (a) HEAVY_TAIL — the
certified-interval p95 exceeds TAIL_RATIO_MAX x the run's own certified
median (a quantile that far above the median is not a spike guard, it is
the spikes); (b) the window ratio is bounded in BOTH directions at the same
decade — an early median BLOWUP_NORM_RATIO x above the late one is a
collapse after an explosion, not calm; (c) the late-window loss median must
not exceed the run's own PRE-TRAINING loss (the median over the bench
warm-up steps 1..BENCH_WARMUP_STEPS, where the lr is still ramping from
zero) — a fall measured from a blown-up level is not a decreasing loss.
The registered verdict is filed first and stands; then (a) names the class
heavy_tail; then (b) and (c) name it diverged. Whichever names the class,
every criterion that fired is in ``reasons``.

The inputs are ``tune_arm.py bench`` records (``grad_norm_preclip.steps /
.values`` — the trainer's own ``<arm>/grad_norm``, the PRE-clip total norm
``clip_grad_norm_`` returns, one per optimizer step; ``loss.values``), never
a second measurement. The same series reader serves a fit's or a trial's
``metrics.jsonl`` rows (one dict per optimizer step carrying
``<arm>/grad_norm``), which is how ``clip_activation_rate`` — the share of
logged optimizer steps whose pre-clip norm exceeded that trial's own clip
(verified-facts §3: exactly derivable post hoc) — is computed for the fold
table and the selection constraint.

Pure python — no torch, no repo imports — so tune_arm.py, the Modal app and
an operator's fold writer outside the repo can load it by path and file the
SAME numbers. Deterministic: quantiles are the linear-interpolation
definition (numpy's default), the ceiling is exact decimal arithmetic.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from decimal import ROUND_CEILING, Decimal
from pathlib import Path

#: Optimizer steps dropped before ANY quantile — the bench's warm-up window
#: (allocator warm-up + one-off kernel selection; tune_arm.BENCH_WARMUP_STEPS
#: is this constant). "Steps > 5" in the plan's spelling.
BENCH_WARMUP_STEPS = 5

#: The plan's percentile: every trial's clip sits at or above the p95 of every
#: stable measured distribution -> activation <= ~5% by construction.
CLIP_PERCENTILE = 0.95

CLIP_RULE = ("ceil2sf(max over STABLE recon runs of the arm of p95(pre-clip "
             "grad norm over the certified interval, steps BENCH_WARMUP_STEPS "
             "< s <= RECON_STEPS)) — plan §4.2; diverged / heavy-tail / failed "
             "/ incomplete / over-budget runs are diagnostic only")

#: Selection constraint (healthy 0-4%, borderline 24-65%, defective 100%):
#: a trial whose clip-activation rate reaches CLIP_ACTIVATION_MAX is not
#: promotable — ineligible as a raw-argmin fall-through, and flagged when
#: it is one of the fitted argmin's nearest neighbours.
#: CLIP_ACTIVATION_TARGET is the regime the promoted point should sit in.
CLIP_ACTIVATION_MAX = 0.25
CLIP_ACTIVATION_TARGET = 0.10

#: Blow-up rule (iii): late-window median pre-clip norm above this multiple
#: of the early-window median is a diverging run, whatever its loss says —
#: and an EARLY median above this multiple of the late one is a
#: collapse after an explosion, the same verdict; the bound is symmetric.
BLOWUP_NORM_RATIO = 10.0

#: Heavy-tail bound: a run whose certified-interval p95 pre-clip norm
#: exceeds this multiple of its own certified-interval median is HEAVY_TAIL
#: — ineligible, diagnostic only — whatever its window medians and loss say.
#: The same decade as the blow-up rule, chosen against measured runs, not
#: taste. Healthy: the four llama8b_base anchors 1.2-2.0; the published
#: healthy cells of verified-facts §3 (GLM, nemotron, qwen LoRA and it_doc;
#: 13 cells with steps 6-150 logged) 1.3-2.0 over the certified interval and
#: <= 3.3 over their whole runs; the borderline qwen ctl2 PSR cell (65% clip
#: activation) 4.3; the module's own registered "9x late/early step is still
#: stable" example 8.7. Pathological: the 100%-clipped gemma cells — the
#: defect the clip constant exists to prevent — 5.7-15.5; the diverged LoRA
#: box top 183; the PSR box top 2,134. Ten sits >= 3x above every
#: healthy run, 2.3x above the borderline one, 18x under the mildest blow-up
#: seen and 200x under the run that motivated it; anything tighter than ~9
#: would fail the registered 9x example, anything looser than ~15 would
#: pass the gemma regime.
TAIL_RATIO_MAX = 10.0

#: The reconnaissance step budget (150 optimizer steps at the
#: production effective batch). Clip-setting eligibility requires EXACTLY
#: this of ``steps_run`` DIRECTLY — never through ``steps_requested``, which
#: a smoke record satisfies at any length; fewer is incomplete, more is
#: over_budget — and the stability windows and the certified interval
#: are step indices anchored to it. The reconnaissance launcher
#: (sdfbase_fits.py --mode bench with per-run overrides) requests exactly
#: this count and refuses any other.
RECON_STEPS = 150


def stability_windows(recon_steps: int = RECON_STEPS,
                      warmup_steps: int = BENCH_WARMUP_STEPS
                      ) -> dict[str, tuple[int, int]]:
    """The REGISTERED stability windows as inclusive step-index ranges
    (at the 150-step budget): grad norm early = steps 6-50 and
    late = 101-150 (the first and last third of the budget, the early one
    minus the warm-up); loss early = the first 25 post-warmup steps (6-30)
    and late = the last 25 (126-150). Step indices, never fractions of a
    record's length."""
    third, sixth = recon_steps // 3, recon_steps // 6
    return {
        "grad_norm_early": (warmup_steps + 1, third),
        "grad_norm_late": (recon_steps - third + 1, recon_steps),
        "loss_early": (warmup_steps + 1, warmup_steps + sixth),
        "loss_late": (recon_steps - sixth + 1, recon_steps),
    }


def clip_interval(recon_steps: int = RECON_STEPS,
                  warmup_steps: int = BENCH_WARMUP_STEPS) -> tuple[int, int]:
    """The CERTIFIED interval as an inclusive step-index range — steps
    warmup_steps+1 .. recon_steps (6-150 at the registered budget): the span
    the stability windows lie in, the span an eligible record's norm series
    must cover completely, and the ONLY span the clip quantile may draw
    from. One object for all three, so the quantile cannot contain an
    observation the classifier never examined — it did once, when the
    classifier stopped at step 150 and the quantile ran to 200."""
    return (int(warmup_steps) + 1, int(recon_steps))


def clip_interval_values(steps, vals, *, recon_steps: int = RECON_STEPS,
                         warmup_steps: int = BENCH_WARMUP_STEPS) -> list[float]:
    """The values of a (steps, vals) series whose step lies inside
    ``clip_interval`` — selected by STEP INDEX, never by position, so a
    series that runs past the budget (or arrives unsorted) contributes
    exactly its certified steps and nothing else. The bound is structural:
    it holds for whatever series reaches the quantile, not because a
    refusal upstream happened to keep long records out."""
    lo, hi = clip_interval(recon_steps, warmup_steps)
    return [float(v) for s, v in zip(steps, vals) if lo <= int(s) <= hi]


#: The CE arms log the pre-clip norm under ``<arm>/grad_norm`` (train_object_ce,
#: log_tag=<arm>); the lockstep arms under run_trial's GradNormBuffer drain.
CE_ARMS = ("psr_ce_all", "lora_r4_ce")

#: The statuses a bench record may carry (tune_arm.py bench's explicit
#: outcome field) and the eligibility classes this module files.
BENCH_STATUSES = ("ok", "diverged", "failed")
RUN_CLASSES = ("stable", "diverged", "heavy_tail", "failed", "incomplete",
               "over_budget")


# --------------------------------------------------------------------------
# metric keys + series
# --------------------------------------------------------------------------

def gradnorm_key(arm: str) -> str:
    """The metrics key carrying the PRE-clip total gradient norm for ``arm``:
    ``<arm>/grad_norm`` for the CE arms (objects.py's per-step record —
    logged only under a finite clip), ``train/grad_norm_preclip`` for the
    lockstep arms (run_trial's GradNormBuffer drain)."""
    return f"{arm}/grad_norm" if arm in CE_ARMS else "train/grad_norm_preclip"


def loss_key(arm: str) -> str:
    """The per-step training-loss key of ``arm`` (the CE arms' ``ce_loss``;
    the bench serves the CE arms only)."""
    return f"{arm}/ce_loss"


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def series_from_rows(rows, key: str) -> tuple[list[int], list[float]]:
    """(steps, values) for ``key`` over metrics-shaped rows — dicts with an
    integer ``step`` and, on the rows that carry it, a numeric ``key`` —
    sorted by step (stable, so a re-logged step keeps file order). Rows
    without the key (probe rows, the step-0 noop-floor row, heartbeat rows)
    are skipped; NaN values are KEPT (they are the finiteness evidence)."""
    pts = []
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or key not in r or not _is_num(r[key]):
            continue
        s = r.get("step")
        if not _is_num(s):
            continue
        pts.append((int(s), i, float(r[key])))
    pts.sort(key=lambda t: (t[0], t[1]))
    return [s for s, _i, _v in pts], [v for _s, _i, v in pts]


def load_metrics_rows(path) -> list[dict]:
    """The rows of a ``metrics.jsonl`` (one JSON object per line). A torn
    LAST line — a process that died mid-write — is dropped; a torn line
    anywhere else is a corrupt file and raises."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    rows: list[dict] = []
    for i, ln in enumerate(lines):
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break
            raise ValueError(f"{path}: line {i + 1} is not JSON and is not "
                             "the final (torn) line — corrupt metrics file")
    return rows


# --------------------------------------------------------------------------
# quantiles + the ceiling
# --------------------------------------------------------------------------

def quantile(values, q: float) -> float:
    """The q-quantile of ``values`` by linear interpolation between order
    statistics (numpy.quantile's default method), so p95 here equals
    ``numpy.quantile(values, 0.95)`` to floating-point rounding."""
    xs = sorted(float(v) for v in values)
    if not xs:
        raise ValueError("quantile of an empty series")
    if not (0.0 <= q <= 1.0):
        raise ValueError(f"quantile level must lie in [0, 1], got {q!r}")
    pos = q * (len(xs) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def quantile_summary(values) -> dict:
    """{n, min, p50, p90, p95, p99, max} of a series (None-valued when
    empty) — the artifact's per-run and pooled summary shape."""
    xs = [float(v) for v in values]
    if not xs:
        return {"n": 0, "min": None, "p50": None, "p90": None, "p95": None,
                "p99": None, "max": None}
    return {"n": len(xs), "min": min(xs), "p50": quantile(xs, 0.50),
            "p90": quantile(xs, 0.90), "p95": quantile(xs, 0.95),
            "p99": quantile(xs, 0.99), "max": max(xs)}


def ceil2sf(x: float) -> float:
    """``x`` rounded UP to two significant figures, in exact decimal
    arithmetic (0.851 -> 0.86, 2.049 -> 2.1, 123.4 -> 130, 0.85 -> 0.85).
    Refuses a non-finite or non-positive input — a clip is a finite positive
    number."""
    xf = float(x)
    if not math.isfinite(xf) or xf <= 0:
        raise ValueError(f"ceil2sf needs a finite positive number, got {x!r}")
    d = Decimal(repr(xf))
    q = Decimal(1).scaleb(d.adjusted() - 1)
    return float(d.quantize(q, rounding=ROUND_CEILING))


# --------------------------------------------------------------------------
# eligibility (decided BEFORE any quantile)
# --------------------------------------------------------------------------

def _record_series(rec: dict) -> tuple[list[int], list[float]]:
    gn = rec.get("grad_norm_preclip")
    if not isinstance(gn, dict) or not gn.get("steps") or not gn.get("values"):
        raise ValueError(
            "bench record carries no grad_norm_preclip series — re-run the "
            "bench with this harness (tune_arm.py bench records the trainer's "
            "own <arm>/grad_norm per optimizer step)")
    steps = [int(s) for s in gn["steps"]]
    vals = [float(v) for v in gn["values"]]
    if len(steps) != len(vals):
        raise ValueError("grad_norm_preclip steps/values length mismatch")
    return steps, vals


def _record_loss(rec: dict) -> tuple[list[int], list[float]]:
    ls = rec.get("loss")
    if not isinstance(ls, dict) or "values" not in ls or "steps" not in ls:
        raise ValueError(
            "bench record carries no loss series — re-run the bench with "
            "this harness")
    return [int(s) for s in ls["steps"]], [float(v) for v in ls["values"]]


def classify_run(rec: dict | None, *, warmup_steps: int = BENCH_WARMUP_STEPS,
                 norm_blowup_ratio: float = BLOWUP_NORM_RATIO,
                 recon_steps: int = RECON_STEPS,
                 tail_ratio_max: float = TAIL_RATIO_MAX) -> dict:
    """The eligibility verdict for ONE bench record (stable-run
    eligibility, plus the extensions): {class, eligible, reasons, windows}.

    stable      status ok, steps_run >= steps_requested AND steps_run ==
                recon_steps EXACTLY (the reconnaissance budget, required of
                the run itself — not of what it requested), the norm series
                covering every step of the certified interval
                (``clip_interval``: warmup_steps+1 .. recon_steps) and the
                loss series every step of its windows and of the
                pre-training steps 1..warmup_steps, loss finite throughout,
                no blow-up (late/early median grad-norm ratio <=
                norm_blowup_ratio AND late median loss <= early median
                loss, over ``stability_windows(recon_steps, warmup_steps)``
                — step indices, never fractions of the run), AND the
                certified-interval p95 / p50 of the norm <= tail_ratio_max,
                early/late median grad-norm ratio <= norm_blowup_ratio (the
                bound is symmetric), late median loss <= the pre-training
                median loss
    heavy_tail  ok, complete, finite, the REGISTERED rule passed, but the
                certified-interval p95 pre-clip norm exceeds tail_ratio_max
                x the run's own certified median (p50 3.46, p95 7,391 — a
                spiking run the window medians cannot see; its p95 would
                set a clip that never binds)
    diverged    status diverged; a failed record whose error names the
                trainer's divergence abort; a blow-up / non-finite loss on
                an ok record; an early-window median grad norm above
                norm_blowup_ratio x the late one (a collapse after an
                explosion); or a late-window loss median above the run's
                own pre-training loss (a fall measured from a blown-up
                level)
    failed      status failed with no divergence signature (infrastructure —
                rerun once), or an unknown status
    incomplete  ok but fewer steps than requested, fewer than the
                reconnaissance budget, or a series torn inside the
                certified interval / a loss window
    over_budget ok but MORE steps than the reconnaissance budget: the
                windows certify steps warmup_steps+1 .. recon_steps only,
                so the record carries a tail this verdict never examined
                (such a tail set the clip once); a reconnaissance run
                requests exactly the budget

    Only ``stable`` is eligible. Precedence when several criteria fire: the
    registered rule's ``diverged`` is filed first and stands; then the tail
    names ``heavy_tail``; then the collapse and the pre-training floor name
    ``diverged``. Whichever names the class, EVERY criterion that fired is a
    reason. A None record raises: a missing record is not an outcome (the
    reader refuses to build a clip over it). So does an ok record whose
    series run past its own ``steps_run`` — an inconsistent record, never
    read through its claimed step count."""
    if rec is None:
        raise ValueError("no bench record — a MISSING record is not an "
                         "outcome; re-run the bench or drop the run "
                         "explicitly")
    status = rec.get("status")
    reasons: list[str] = []
    windows: dict = {}
    if status == "diverged":
        cls = "diverged"
        reasons.append(f"bench outcome diverged: {rec.get('error') or ''}"
                       .strip())
    elif status == "failed":
        text = f"{rec.get('error') or ''}\n{rec.get('stderr_tail') or ''}"
        if "TrainingDiverged" in text or "TrialDiverged" in text:
            cls = "diverged"
            reasons.append("bench outcome failed with the trainer's "
                           "divergence abort in its error")
        else:
            cls = "failed"
            reasons.append(f"bench outcome failed (rc={rec.get('rc')}): "
                           f"{(rec.get('error') or '').strip()[:200]}")
    elif status == "ok":
        steps, vals = _record_series(rec)
        req, run = rec.get("steps_requested"), rec.get("steps_run")
        win = stability_windows(recon_steps, warmup_steps)
        c_lo, c_hi = clip_interval(recon_steps, warmup_steps)
        if run is not None and steps and max(steps) > int(run):
            # The series is the evidence and steps_run the claim about it:
            # a claim of the budget over a longer series would read stable
            # THROUGH the claim. Inconsistent — refused, not classified.
            raise ValueError(
                f"grad_norm_preclip series reaches step {max(steps)} but "
                f"steps_run={run} — the record is inconsistent (a series "
                "past the run's own step count); re-run the bench with "
                "this harness")
        if req is None or run is None or int(run) < int(req):
            cls = "incomplete"
            reasons.append(f"steps_run={run} < steps_requested={req}")
        elif int(run) < int(recon_steps):
            # The budget is required of the RUN, whatever it requested: a
            # smoke record that ran every step it asked for is still not
            # a reconnaissance run and never sets the clip.
            cls = "incomplete"
            reasons.append(f"steps_run={run} < reconnaissance budget "
                           f"{recon_steps} — not a reconnaissance run; the "
                           "registered stability windows need steps "
                           f"{win['grad_norm_early'][0]}-"
                           f"{win['grad_norm_late'][1]}")
        elif int(run) > int(recon_steps):
            # EXACTLY the budget, never more: the windows certify steps
            # c_lo-c_hi only, so a longer record's tail is telemetry this
            # verdict never looked at — and the tail is exactly where a
            # late blow-up sits (0.3 through step 150, 1000 over 151-200,
            # read stable, clip 1000). The launcher requests exactly the
            # budget; a record that did not is diagnostic only.
            cls = "over_budget"
            reasons.append(f"steps_run={run} > reconnaissance budget "
                           f"{recon_steps} — the registered stability "
                           f"windows certify steps {c_lo}-{c_hi} only, so "
                           f"steps {c_hi + 1}-{int(run)} would enter the "
                           "clip quantile unexamined; a reconnaissance run "
                           f"requests exactly {recon_steps} steps")
        else:
            l_steps, l_vals = _record_loss(rec)
            if l_steps and max(l_steps) > int(run):
                raise ValueError(
                    f"loss series reaches step {max(l_steps)} but "
                    f"steps_run={run} — the record is inconsistent; re-run "
                    "the bench with this harness")
            gn_at = {s: v for s, v in zip(steps, vals)}
            loss_at = {s: v for s, v in zip(l_steps, l_vals)}
            # The norm series must be COMPLETE over the certified interval
            # (the quantile's whole domain; the norm windows are subsets of
            # it), the loss series over its windows AND over the
            # pre-training steps 1..warmup_steps (the loss floor's reference).
            pre_lo, pre_hi = 1, int(warmup_steps)
            missing = sorted(
                {s for s in range(c_lo, c_hi + 1) if s not in gn_at}
                | {s for name, (lo, hi) in win.items()
                   if name.startswith("loss")
                   for s in range(lo, hi + 1) if s not in loss_at}
                | {s for s in range(pre_lo, pre_hi + 1) if s not in loss_at})
            if missing:
                cls = "incomplete"
                reasons.append(f"series lacks {len(missing)} step(s) the "
                               f"verdict needs (certified interval "
                               f"{c_lo}-{c_hi}; loss over the pre-training "
                               f"steps {pre_lo}-{pre_hi}): {missing[:8]}"
                               + (" ..." if len(missing) > 8 else ""))
            elif not all(math.isfinite(v) for v in l_vals) or \
                    not all(math.isfinite(v) for v in vals):
                cls = "diverged"
                reasons.append("non-finite loss or grad norm in the series")
            else:
                def _in(at: dict, lo: int, hi: int) -> list[float]:
                    return [at[s] for s in range(lo, hi + 1)]

                early_gn = _in(gn_at, *win["grad_norm_early"])
                late_gn = _in(gn_at, *win["grad_norm_late"])
                early_l = _in(loss_at, *win["loss_early"])
                late_l = _in(loss_at, *win["loss_late"])
                m_e, m_l = statistics.median(early_gn), statistics.median(late_gn)
                l_e, l_l = statistics.median(early_l), statistics.median(late_l)
                # The tail statistic — the rule's own quantile over the
                # certified interval against the run's own median there.
                cert = _in(gn_at, c_lo, c_hi)
                t_p50, t_p95 = quantile(cert, 0.5), quantile(cert, CLIP_PERCENTILE)
                t_ratio = (t_p95 / t_p50) if t_p50 > 0 else math.inf
                # The pre-training loss — the bench warm-up steps, where
                # the lr is still ramping from zero.
                l_pre = statistics.median(_in(loss_at, pre_lo, pre_hi))
                windows = {
                    "grad_norm_early": {"steps": list(win["grad_norm_early"]),
                                        "p50": m_e},
                    "grad_norm_late": {"steps": list(win["grad_norm_late"]),
                                       "p50": m_l},
                    "loss_early": {"steps": list(win["loss_early"]),
                                   "median": l_e},
                    "loss_late": {"steps": list(win["loss_late"]),
                                  "median": l_l},
                    "loss_pretraining": {"steps": [pre_lo, pre_hi],
                                         "median": l_pre},
                    "tail": {"interval": [c_lo, c_hi], "p50": t_p50,
                             "p95": t_p95, "ratio": t_ratio,
                             "ratio_max": tail_ratio_max},
                    "norm_blowup_ratio_max": norm_blowup_ratio,
                    "recon_steps": int(recon_steps),
                    "clip_interval": [c_lo, c_hi],
                }
                # The REGISTERED rule first; its verdict stands.
                registered_diverged = False
                if m_e > 0 and m_l > norm_blowup_ratio * m_e:
                    registered_diverged = True
                    reasons.append(f"grad-norm blow-up: late p50 {m_l:.4g} > "
                                   f"{norm_blowup_ratio:g} x early p50 "
                                   f"{m_e:.4g}")
                if l_l > l_e:
                    registered_diverged = True
                    reasons.append(f"loss not decreasing: late median "
                                   f"{l_l:.4f} > early median {l_e:.4f}")
                # The extensions — every one evaluated, every one a reason.
                heavy_tail = t_ratio > tail_ratio_max
                if heavy_tail:
                    reasons.append(f"heavy tail: certified-interval p95 "
                                   f"{t_p95:.4g} is {t_ratio:.4g} x the median "
                                   f"{t_p50:.4g} (bound {tail_ratio_max:g}x) — "
                                   "a spiking norm distribution the window "
                                   "medians cannot see; its p95 is the spikes, "
                                   "not a spike guard")
                extended_diverged = False
                if m_l > 0 and m_e > norm_blowup_ratio * m_l:
                    extended_diverged = True
                    reasons.append(f"grad-norm collapse: early p50 {m_e:.4g} > "
                                   f"{norm_blowup_ratio:g} x late p50 "
                                   f"{m_l:.4g} — an explosion inside the early "
                                   "window followed by collapse, not stability")
                if l_l > l_pre:
                    extended_diverged = True
                    reasons.append(f"loss above pre-training: late median "
                                   f"{l_l:.4f} > pre-training median "
                                   f"{l_pre:.4f} (steps {pre_lo}-{pre_hi}) — "
                                   "the run ends worse than it began; a fall "
                                   "measured from a blown-up level is not a "
                                   "decreasing loss")
                if registered_diverged:
                    cls = "diverged"
                elif heavy_tail:
                    cls = "heavy_tail"
                elif extended_diverged:
                    cls = "diverged"
                else:
                    cls = "stable"
    else:
        cls = "failed"
        reasons.append(f"unknown bench status {status!r}")
    return {"class": cls, "eligible": cls == "stable", "reasons": reasons,
            "windows": windows}


# --------------------------------------------------------------------------
# the clip
# --------------------------------------------------------------------------

def clip_from_records(records: dict, *, arm: str,
                      percentile: float = CLIP_PERCENTILE,
                      warmup_steps: int = BENCH_WARMUP_STEPS,
                      recon_steps: int = RECON_STEPS) -> dict:
    """The per-arm clip constant from the reconnaissance bench records
    ``{run_id: record}``, plus everything the artifact needs:

      clip                 ceil2sf(clip_raw)
      clip_raw             max over STABLE runs of p<percentile> over the
                           CERTIFIED interval (clip_interval: steps
                           warmup_steps+1 .. recon_steps — every step the
                           classifier certified and no other)
      clip_interval        that interval, [lo, hi] inclusive
      argmax_run           the stable run that set it
      eligible_runs        the STABLE run ids (the only ones in the clip)
      ineligible_runs      {run_id: {class, reasons}} — diagnostic only
                           (diverged / heavy_tail / failed / incomplete /
                           over_budget; heavy_tail never sets a clip)
      runs                 per run: class / eligibility / hp / steps /
                           quantiles_after_warmup (EVERY post-warmup step,
                           an over-budget tail included — the diagnostic
                           table shows what the record holds; also for
                           ineligible runs, marked diagnostic_only) /
                           clip_bound_steps / for a stable run p_rule with
                           its p_rule_interval and p_rule_n
      pooled               quantile summary over every stable run's
                           certified-interval steps (median = p50, p90,
                           p99, max, n)

    Refuses (ValueError) a missing record, a record of another arm, a
    stable run whose own clip bound during the run (its trajectory is not
    the unclipped distribution), and an arm with no stable run at all — a
    clip cannot be set from diverged telemetry."""
    if not records:
        raise ValueError(f"{arm}: no bench records given")
    runs: dict = {}
    pooled: list[float] = []
    for rid in sorted(records):
        rec = records[rid]
        if rec is None:
            raise ValueError(f"{rid}: no bench record — a MISSING record is "
                             "not an outcome; re-run the bench or drop the "
                             "run explicitly")
        rec_arm = rec.get("arm")
        if rec_arm != arm:
            raise ValueError(f"{rid}: record is for arm {rec_arm!r}, not "
                             f"{arm!r} — one clip per arm, never a pooled "
                             "distribution across arms")
        verdict = classify_run(rec, warmup_steps=warmup_steps,
                               recon_steps=recon_steps)
        entry = {
            "class": verdict["class"], "eligible": verdict["eligible"],
            "reasons": verdict["reasons"], "windows": verdict["windows"],
            "status": rec.get("status"),
            "lr": rec.get("lr"), "max_grad_norm": rec.get("max_grad_norm"),
            "warmup_frac": rec.get("warmup_frac"),
            "pool": rec.get("si_label") or rec.get("pool_cell"),
            "pool_sha256": rec.get("pool_sha256"),
            "model": rec.get("model"),
            "steps_run": rec.get("steps_run"),
            "steps_requested": rec.get("steps_requested"),
            "effective_batch": rec.get("effective_batch"),
            "diagnostic_only": not verdict["eligible"],
        }
        gn = rec.get("grad_norm_preclip")
        if isinstance(gn, dict) and gn.get("values"):
            steps, vals = _record_series(rec)
            # Diagnostic: EVERY post-warmup step, an over-budget tail
            # included, so the table shows what a refused record holds.
            post = [v for s, v in zip(steps, vals) if s > warmup_steps]
            entry["quantiles_after_warmup"] = quantile_summary(post)
            mg = rec.get("max_grad_norm")
            bound = (sum(1 for v in vals if v > float(mg))
                     if _is_num(mg) and math.isfinite(float(mg)) else None)
            entry["clip_bound_steps"] = bound
            if verdict["eligible"]:
                if bound:
                    raise ValueError(
                        f"{rid}: the run's own clip {mg} bound on {bound} "
                        "step(s) — its norm trajectory is not the unclipped "
                        "distribution the rule needs; recon runs use a large "
                        "finite non-binding clip (1e6)")
                # The quantile's domain is the certified interval and
                # nothing else — selected by step index (structural), and
                # required to be the interval's full length: the classifier
                # certified every step of it, so a shortfall or surplus here
                # (a missing or duplicated step) means the two disagree
                # about the record.
                lo, hi = clip_interval(recon_steps, warmup_steps)
                cert = clip_interval_values(steps, vals,
                                            recon_steps=recon_steps,
                                            warmup_steps=warmup_steps)
                if len(cert) != hi - lo + 1:
                    raise ValueError(
                        f"{rid}: {len(cert)} value(s) over the certified "
                        f"interval {lo}-{hi}, expected {hi - lo + 1} — the "
                        "classifier and the clip quantile disagree about the "
                        "record's series (a duplicated or missing step)")
                entry["p_rule"] = quantile(cert, percentile)
                entry["p_rule_interval"] = [lo, hi]
                entry["p_rule_n"] = len(cert)
                pooled.extend(cert)
        runs[rid] = entry
    stable = [rid for rid in runs if runs[rid]["eligible"]]
    if not stable:
        raise ValueError(
            f"{arm}: no STABLE recon run among {sorted(runs)} "
            f"({ {rid: runs[rid]['class'] for rid in runs} }) — the clip "
            "cannot be set from diverged / heavy-tail / failed / incomplete / "
            "over-budget telemetry; re-run per the plan's re-probe rule")
    argmax = max(stable, key=lambda rid: runs[rid]["p_rule"])
    clip_raw = runs[argmax]["p_rule"]
    return {
        "arm": arm,
        "clip": ceil2sf(clip_raw),
        "clip_raw": clip_raw,
        "argmax_run": argmax,
        "rule": CLIP_RULE,
        "percentile": percentile,
        "warmup_steps_dropped": warmup_steps,
        "recon_steps": int(recon_steps),
        "clip_interval": list(clip_interval(recon_steps, warmup_steps)),
        "stability_windows": {k: list(v) for k, v in
                              stability_windows(recon_steps, warmup_steps).items()},
        "eligible_runs": stable,
        "ineligible_runs": {rid: {"class": runs[rid]["class"],
                                  "reasons": runs[rid]["reasons"]}
                            for rid in runs if not runs[rid]["eligible"]},
        "runs": runs,
        "pooled": quantile_summary(pooled),
    }


# --------------------------------------------------------------------------
# clip activation
# --------------------------------------------------------------------------

#: The metrics.jsonl row key (and the summary.json key) carrying a trial
#: ATTEMPT's id: run_trial stamps every row it writes with one fresh id per
#: process and records the same id on the summary it publishes, so a reader
#: can select exactly the segment that summary describes. The file is opened
#: in APPEND mode across re-runs (fingerprint-mismatched or torn work re-runs
#: into the same directory), so without the stamp an earlier attempt's rows
#: sit before the current trajectory's and dilute its rate.
METRICS_ATTEMPT_KEY = "attempt"


def clip_activation_rate(rows, key: str, clip: float) -> tuple[float | None, int]:
    """(rate, n_steps): the share of metrics rows carrying ``key`` whose
    pre-clip norm exceeded ``clip`` — verified-facts §3's
    ``mean(<arm>/grad_norm > max_grad_norm)``, exactly, over the rows given
    (the caller selects the segment: ``clip_activation_from_metrics`` binds
    it to a summary's attempt stamp). (None, 0) when no row carries the
    key."""
    vals = [float(r[key]) for r in rows
            if isinstance(r, dict) and key in r and _is_num(r[key])]
    if not vals:
        return None, 0
    c = float(clip)
    return sum(1 for v in vals if v > c) / len(vals), len(vals)


def rows_of_attempt(rows, attempt, *, path=None) -> list:
    """The rows stamped ``METRICS_ATTEMPT_KEY == attempt``: the one segment
    of an append-mode metrics file that a summary carrying ``attempt``
    describes. ``attempt`` None selects every row — a summary written
    before the stamp existed describes the whole file — but ONLY over a
    file that carries no stamp at all. A stamped row under an unstamped
    summary means a stamped attempt re-ran into this directory (a
    fingerprint change on a pre-stamp trial) and never published its own
    summary — interrupted, or still running — so the file's tail is not
    the unstamped summary's record, and the whole-file read blends the two
    (a 40% predecessor read 20% and crossed into promotable). That
    pairing is refused, never silently narrowed to either segment. A
    non-empty file with NO row of the asked-for attempt is not that
    summary's record (a torn or foreign file) and is refused rather than
    read as "no evidence"."""
    rows = list(rows)
    if attempt is None:
        stamped = [r for r in rows
                   if isinstance(r, dict) and METRICS_ATTEMPT_KEY in r]
        if stamped:
            seen = sorted({str(r.get(METRICS_ATTEMPT_KEY)) for r in stamped})
            raise ValueError(
                f"{path or 'metrics rows'}: the summary carries no "
                f"{METRICS_ATTEMPT_KEY} stamp but {len(stamped)} of "
                f"{len(rows)} row(s) do (stamps seen: {seen}) — an unstamped "
                f"summary describes a whole UNSTAMPED file, and a stamped "
                f"attempt re-ran into this directory without publishing its "
                f"summary; the pairing is ambiguous and is refused rather than "
                f"read as either attempt's record")
        return rows
    sel = [r for r in rows
           if isinstance(r, dict) and r.get(METRICS_ATTEMPT_KEY) == attempt]
    if rows and not sel:
        seen = sorted({str(r.get(METRICS_ATTEMPT_KEY))
                       for r in rows if isinstance(r, dict)})
        raise ValueError(
            f"{path or 'metrics rows'}: none of {len(rows)} row(s) carries "
            f"the summary's {METRICS_ATTEMPT_KEY} {attempt!r} (stamps seen: "
            f"{seen}) — the metrics file is not this summary's record")
    return sel


def clip_activation_from_metrics(path, *, arm: str, clip: float,
                                 attempt: str | None = None
                                 ) -> tuple[float | None, int]:
    """``clip_activation_rate`` over a ``metrics.jsonl`` file for ``arm``
    against ``clip`` (a trial's ``summary.json["trial"]["max_grad_norm"]``,
    or a fit's folded ``max_grad_norm``), over the rows of ``attempt`` only
    when a summary names one; without one, the whole file, which must then
    carry no stamp (``rows_of_attempt`` refuses the other pairings)."""
    rows = rows_of_attempt(load_metrics_rows(path), attempt, path=path)
    return clip_activation_rate(rows, gradnorm_key(arm), clip)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def run_ids(paths) -> dict:
    """``{run_id: path}`` for the CLI's record files — a COLLISION-FREE id
    per distinct file: the resolved path relative to the inputs' common
    directory, suffix dropped. Records in one directory keep their bare
    stems (the historical run id); records that share a basename under
    different parents — the bench's default filename encodes the settings,
    not the model, which is the parent directory, so base and instruct
    reconnaissance runs at identical settings collide by stem — are told
    apart by the parent (``llama8b-base/lr4.5e-4_w0.05``). The same file
    named twice (by any spelling) is a duplicate identity and is refused
    before anything is computed: indexing by stem used to let the later
    input silently replace the earlier one, so input order decided which
    evidence vanished."""
    resolved = [Path(p).resolve() for p in paths]
    dups = sorted({str(r) for r in resolved if resolved.count(r) > 1})
    if dups:
        raise ValueError(f"duplicate record(s) given more than once: {dups} "
                         "— each reconnaissance record enters the pool once")
    common = Path(os.path.commonpath([str(r.parent) for r in resolved]))
    out: dict = {}
    for r in resolved:
        rid = str(r.relative_to(common).with_suffix(""))
        if rid in out:  # cannot happen for distinct resolved paths; loud anyway
            raise ValueError(f"run id {rid!r} names two records: {out[rid]} "
                             f"and {r}")
        out[rid] = r
    return out


def _fmt(v, spec=".4g") -> str:
    return "n/a" if v is None else format(float(v), spec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("records", nargs="+",
                    help="tune_arm.py bench records (JSON) of ONE arm — the "
                         "reconnaissance runs; the run id is the record's "
                         "path relative to the inputs' common directory, "
                         "suffix dropped (the bare stem when they share one "
                         "directory); the same file twice is refused")
    ap.add_argument("--arm", required=True, choices=list(CE_ARMS))
    ap.add_argument("--percentile", type=float, default=CLIP_PERCENTILE)
    ap.add_argument("--warmup-steps", type=int, default=BENCH_WARMUP_STEPS)
    ap.add_argument("--out", default=None,
                    help="write the full result JSON here (the artifact's "
                         "recon record); the clip and table print either way")
    args = ap.parse_args(argv)
    paths = []
    for p in args.records:
        path = Path(p)
        if not path.is_file():
            raise SystemExit(f"[recon-clip] {path}: no such record — a MISSING "
                             "record is not an outcome")
        paths.append(path)
    try:
        ids = run_ids(paths)
    except ValueError as e:
        raise SystemExit(f"[recon-clip] REFUSED: {e}")
    records = {rid: json.loads(path.read_text()) for rid, path in ids.items()}
    try:
        res = clip_from_records(records, arm=args.arm,
                                percentile=args.percentile,
                                warmup_steps=args.warmup_steps)
    except ValueError as e:
        raise SystemExit(f"[recon-clip] REFUSED: {e}")
    res["record_paths"] = {rid: str(path.resolve()) for rid, path in ids.items()}
    print(f"[recon-clip] arm={args.arm} rule: {res['rule']}")
    print(f"[recon-clip] {'run':28s} {'class':12s} {'lr':>9s} {'warm':>5s} "
          f"{'steps':>9s} {'p50':>9s} {'p90':>9s} {'p95':>9s} {'p99':>9s} "
          f"{'max':>9s}")
    for rid, e in res["runs"].items():
        q = e.get("quantiles_after_warmup") or {}
        print(f"  {rid:28s} {e['class']:12s} {_fmt(e['lr'], '.3e'):>9s} "
              f"{_fmt(e['warmup_frac'], '.2f'):>5s} "
              f"{str(e['steps_run']) + '/' + str(e['steps_requested']):>9s} "
              f"{_fmt(q.get('p50')):>9s} {_fmt(q.get('p90')):>9s} "
              f"{_fmt(q.get('p95')):>9s} {_fmt(q.get('p99')):>9s} "
              f"{_fmt(q.get('max')):>9s}"
              + ("" if e["eligible"] else
                 f"   <- {e['class'].upper()} (diagnostic only): "
                 + "; ".join(e["reasons"])))
    pooled = res["pooled"]
    print(f"[recon-clip] stable runs {res['eligible_runs']}: pooled "
          f"n={pooled['n']} median={_fmt(pooled['p50'])} "
          f"p90={_fmt(pooled['p90'])} p99={_fmt(pooled['p99'])} "
          f"max={_fmt(pooled['max'])}")
    print(f"[recon-clip] CLIP {args.arm} = {res['clip']:g}  "
          f"(ceil2sf of p{args.percentile * 100:g} {res['clip_raw']:.4g} "
          f"from run {res['argmax_run']}) -> emit-trials --clip-const "
          f"{res['clip']:g}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
        print(f"[recon-clip] wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
