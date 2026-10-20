"""Per-position agreement of two top-k next-token readouts of ONE model served
two ways (bf16 vs FP8 weights; prefix cache off vs on).

What a pair of top-k readouts can and cannot say
------------------------------------------------
A KL computed over the intersection of two top-k sets with everything else
folded into one residual bucket is a LOWER bound on the full-vocabulary KL:
tokens can enter or leave the top-k while the shared mass stays alike, the
residual bucket hides any rearrangement inside the tail, and a position with no
common top-k token is skipped altogether.  A small coarsened KL is therefore
NOT evidence of equivalence and is never used as such here.  It is still
reported (labelled a lower bound) because it is the one number that says "at
least this different".

What IS decidable per position from the readout alone:

- whether the two top-k MEMBERSHIPS are identical (and, if not, how much
  left/right mass sits on the tokens the other side did not rank);
- how far each COMMON token's logprob moved (max and mass-weighted);
- how much mass the top-k CAPTURED on each side -- the residual is exactly what
  the readout cannot see, so a large residual makes the position inconclusive
  whatever the top-k says;
- whether the argmax agrees.

Certificate (per position): identical membership AND every common-token
|delta logp| <= ``delta_tol`` AND residual mass <= ``residual_cap`` on both
sides.  Verdict over a set of positions:

- ``equivalent_within_tolerance``: every position certified;
- ``divergent``: some position with a bounded residual has differing membership
  or a common-token delta above tolerance (a measured difference, not noise
  of the readout -- compare against a same-serving repeat for the floor);
- ``inconclusive``: no divergent position, but some position was skipped (no
  common token) or had an unbounded residual on either side.

Stdlib only, keys of a distribution may be token ids or token strings (the two
sides must use the same convention).  Runs under the vLLM venv and on the
grading host alike, so a recorded probe can be re-analysed from its raw readouts.
"""
from __future__ import annotations

import math
import statistics
from typing import Hashable, Iterable, Mapping, Sequence

DEFAULT_DELTA_TOL = 0.01      # nats; |delta logp| a common token may move
DEFAULT_RESIDUAL_CAP = 0.01   # mass the top-k may leave unseen on either side
DEFAULT_MASS_FLOOR = 1e-3     # descriptive only: "membership above this mass"

STATUS_CERTIFIED = "certified"
STATUS_DIVERGENT = "divergent"
STATUS_SKIPPED = "inconclusive_skipped"
STATUS_UNBOUNDED = "inconclusive_unbounded_tail"
STATUS_EMPTY_ROW = "inconclusive_empty_row"   # a row (prompt) with zero positions on both sides
ALL_STATUSES = (STATUS_CERTIFIED, STATUS_DIVERGENT, STATUS_SKIPPED, STATUS_UNBOUNDED, STATUS_EMPTY_ROW)

VERDICT_EQUIVALENT = "equivalent_within_tolerance"
VERDICT_DIVERGENT = "divergent"
VERDICT_INCONCLUSIVE = "inconclusive"


class AlignmentError(ValueError):
    """The two readouts do not have the same shape (row count, or position
    count within a row), so they cannot be compared position by position.
    A silent ``zip`` would drop the unmatched rows/positions and let the
    surviving ones certify over data one side never produced; this refuses
    instead.  Carries ``row`` (None for an outer mismatch) and the two lengths."""

    def __init__(self, message: str, *, row: int | None, left_len: int, right_len: int) -> None:
        super().__init__(message)
        self.row, self.left_len, self.right_len = row, left_len, right_len


def check_alignment(left_rows: Sequence[Sequence[Distribution]],
                    right_rows: Sequence[Sequence[Distribution]]) -> None:
    """Raise :class:`AlignmentError` unless both readouts have the same number
    of rows AND every row pair has the same number of positions."""
    if len(left_rows) != len(right_rows):
        raise AlignmentError(
            f"readouts have {len(left_rows)} vs {len(right_rows)} rows; refusing to compare "
            "position by position (a truncated or partial leg cannot certify anything)",
            row=None, left_len=len(left_rows), right_len=len(right_rows))
    for i, (lrow, rrow) in enumerate(zip(left_rows, right_rows)):
        if len(lrow) != len(rrow):
            raise AlignmentError(
                f"row {i} has {len(lrow)} vs {len(rrow)} positions; refusing to compare "
                "(the unmatched positions were never read on one side)",
                row=i, left_len=len(lrow), right_len=len(rrow))

Distribution = Mapping[Hashable, float]


def _percentile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def _summary(values: Sequence[float], *, digits: int = 6) -> dict:
    vals = [float(v) for v in values]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "p90": None, "max": None, "min": None}
    return {
        "n": len(vals),
        "mean": round(statistics.mean(vals), digits),
        "median": round(statistics.median(vals), digits),
        "p90": round(_percentile(vals, 0.90), digits),
        "max": round(max(vals), digits),
        "min": round(min(vals), digits),
    }


def _argmax(dist: Distribution):
    return max(dist.items(), key=lambda kv: kv[1])[0] if dist else None


def compare_position(left: Distribution, right: Distribution, *,
                     delta_tol: float = DEFAULT_DELTA_TOL,
                     residual_cap: float = DEFAULT_RESIDUAL_CAP,
                     mass_floor: float = DEFAULT_MASS_FLOOR) -> dict:
    """Everything decidable about one position from two top-k logprob maps."""
    left_keys, right_keys = set(left), set(right)
    common = left_keys & right_keys
    left_only, right_only = left_keys - right_keys, right_keys - left_keys
    p = {k: math.exp(v) for k, v in left.items()}
    q = {k: math.exp(v) for k, v in right.items()}
    captured_left, captured_right = sum(p.values()), sum(q.values())
    residual_left = max(0.0, 1.0 - captured_left)
    residual_right = max(0.0, 1.0 - captured_right)
    argmax_left, argmax_right = _argmax(left), _argmax(right)

    rec = {
        "n_left": len(left), "n_right": len(right), "n_common": len(common),
        "n_left_only": len(left_only), "n_right_only": len(right_only),
        "membership_identical": not left_only and not right_only and bool(common),
        "jaccard": (len(common) / len(left_keys | right_keys)) if (left_keys | right_keys) else None,
        "captured_mass_left": round(captured_left, 6),
        "captured_mass_right": round(captured_right, 6),
        "residual_left": round(residual_left, 6),
        "residual_right": round(residual_right, 6),
        # the mass the OTHER side did not rank at all: boundary churn at ~1e-7
        # vs a real token leaving the top-k look very different here
        "max_mass_left_only": round(max((p[k] for k in left_only), default=0.0), 6),
        "max_mass_right_only": round(max((q[k] for k in right_only), default=0.0), 6),
        "argmax_agree": (argmax_left is not None and argmax_left == argmax_right),
        "argmax_left_in_right_topk": argmax_left in right_keys if argmax_left is not None else False,
        # descriptive: every token carrying >= mass_floor on either side is ranked by both
        "membership_identical_above_mass_floor": all(
            (k in right_keys) for k in left_keys if p[k] >= mass_floor
        ) and all((k in left_keys) for k in right_keys if q[k] >= mass_floor) and bool(common),
    }
    if not common:
        rec.update({
            "max_abs_delta_logp_common": None,
            "max_abs_delta_logp_above_mass_floor": None,
            "mass_weighted_delta_logp_common": None,
            "coarsened_kl_lower_bound_nats": None,
            "status": STATUS_SKIPPED,
        })
        return rec
    deltas = {k: left[k] - right[k] for k in common}
    max_abs_delta = max(abs(d) for d in deltas.values())
    # descriptive companion: the same maximum over the common tokens that carry
    # >= mass_floor on either side. Deep-tail logprobs come back quantized
    # (bf16 steps of 0.125-0.5 nats at |logp| >= 16), so a same-serving repeat
    # can show a 0.5-nat "delta" on a zero-mass token; this quantity says how
    # far the tokens that matter for sampling moved.
    above = [abs(deltas[k]) for k in common if p[k] >= mass_floor or q[k] >= mass_floor]
    max_abs_delta_above_floor = max(above) if above else 0.0
    common_mass = sum(p[k] for k in common)
    weighted = (sum(p[k] * deltas[k] for k in common) / common_mass) if common_mass else 0.0
    p_other = max(1e-12, 1.0 - common_mass)
    q_other = max(1e-12, 1.0 - sum(q[k] for k in common))
    kl = sum(p[k] * math.log(p[k] / max(q[k], 1e-12)) for k in common)
    kl += p_other * math.log(p_other / q_other)
    rec.update({
        "max_abs_delta_logp_common": round(max_abs_delta, 6),
        "max_abs_delta_logp_above_mass_floor": round(max_abs_delta_above_floor, 6),
        "mass_weighted_delta_logp_common": round(weighted, 6),
        "coarsened_kl_lower_bound_nats": round(kl, 8),
    })
    bounded = residual_left <= residual_cap and residual_right <= residual_cap
    if rec["membership_identical"] and max_abs_delta <= delta_tol and bounded:
        rec["status"] = STATUS_CERTIFIED
    elif bounded and (not rec["membership_identical"] or max_abs_delta > delta_tol):
        rec["status"] = STATUS_DIVERGENT
    else:
        rec["status"] = STATUS_UNBOUNDED
    return rec


def compare_rows(left_rows: Iterable[Sequence[Distribution]],
                 right_rows: Iterable[Sequence[Distribution]], *,
                 delta_tol: float = DEFAULT_DELTA_TOL,
                 residual_cap: float = DEFAULT_RESIDUAL_CAP,
                 mass_floor: float = DEFAULT_MASS_FLOOR) -> list[dict]:
    """``compare_position`` over aligned ``rows[i][t]``; each record carries
    its ``(row, t)`` so a divergent position can be located in the raw data.
    Both readouts must have the same shape (:func:`check_alignment`); a
    mismatch raises rather than silently comparing the overlap.  A row that is
    EMPTY on both sides (a prompt whose greedy continuation had no token, so
    nothing was read) yields one row-level record with status
    ``inconclusive_empty_row`` and ``t=None``: it was never compared, so it
    must count against an equivalence verdict rather than vanish."""
    left_rows, right_rows = list(left_rows), list(right_rows)
    check_alignment(left_rows, right_rows)
    out = []
    for i, (lrow, rrow) in enumerate(zip(left_rows, right_rows)):
        if not lrow:
            out.append({"row": i, "t": None, "status": STATUS_EMPTY_ROW, "n_left": 0, "n_right": 0,
                        "n_common": 0, "membership_identical": False,
                        "membership_identical_above_mass_floor": False, "argmax_agree": False,
                        "argmax_left_in_right_topk": False, "captured_mass_left": None,
                        "captured_mass_right": None, "residual_left": None, "residual_right": None,
                        "max_mass_left_only": None, "max_mass_right_only": None, "jaccard": None,
                        "max_abs_delta_logp_common": None, "max_abs_delta_logp_above_mass_floor": None,
                        "mass_weighted_delta_logp_common": None, "coarsened_kl_lower_bound_nats": None})
            continue
        for t, (left, right) in enumerate(zip(lrow, rrow)):
            rec = compare_position(left, right, delta_tol=delta_tol,
                                   residual_cap=residual_cap, mass_floor=mass_floor)
            rec["row"], rec["t"] = i, t
            out.append(rec)
    return out


def verdict_of(records: Sequence[dict]) -> str:
    """Equivalent only if EVERY record -- positions and row-level empty-row
    records alike -- is certified; an empty row was never compared and makes
    the verdict inconclusive."""
    statuses = {r["status"] for r in records}
    if not records:
        return VERDICT_INCONCLUSIVE
    if STATUS_DIVERGENT in statuses:
        return VERDICT_DIVERGENT
    if statuses <= {STATUS_CERTIFIED}:
        return VERDICT_EQUIVALENT
    return VERDICT_INCONCLUSIVE


def summarize(records: Sequence[dict], *, delta_tol: float = DEFAULT_DELTA_TOL,
              residual_cap: float = DEFAULT_RESIDUAL_CAP,
              mass_floor: float = DEFAULT_MASS_FLOOR) -> dict:
    """The aggregate a document quotes: the verdict, the per-status counts, and
    the distributions of the decidable quantities.  The coarsened KL is kept,
    labelled a lower bound, and is not what the verdict rests on."""
    positions = [r for r in records if r.get("t") is not None]
    empty_rows = sorted({r["row"] for r in records if r["status"] == STATUS_EMPTY_ROW})
    n = len(positions)
    by_status = {s: sum(r["status"] == s for r in records) for s in ALL_STATUSES}
    rows_seen = sorted({r["row"] for r in records if r.get("row") is not None})
    compared = [r for r in positions if r["status"] != STATUS_SKIPPED]
    kls = [r["coarsened_kl_lower_bound_nats"] for r in compared]
    deltas = [r["max_abs_delta_logp_common"] for r in compared]
    deltas_above = [r.get("max_abs_delta_logp_above_mass_floor") for r in compared
                    if r.get("max_abs_delta_logp_above_mass_floor") is not None]
    weighted = [r["mass_weighted_delta_logp_common"] for r in compared]
    residuals = [max(r["residual_left"], r["residual_right"]) for r in positions]
    divergent = [r for r in positions if r["status"] == STATUS_DIVERGENT]
    return {
        "verdict": verdict_of(records),
        # per-row (per-prompt) accounting: a row with no positions was never
        # compared and is listed, not dropped
        "rows": {"n_rows": len(rows_seen), "n_rows_compared": len(rows_seen) - len(empty_rows),
                 "n_rows_empty": len(empty_rows), "empty_rows": empty_rows},
        "criteria": {
            "certified_means": "identical top-k membership AND max |delta logp| over common tokens "
                               "<= delta_tol AND residual mass <= residual_cap on both sides",
            "delta_tol_nats": delta_tol, "residual_cap": residual_cap,
            "mass_floor_descriptive": mass_floor,
        },
        "n_positions": n,
        "by_status": by_status,
        "membership_identical_rate": round(sum(r["membership_identical"] for r in positions) / n, 4) if n else None,
        "membership_identical_above_mass_floor_rate": round(
            sum(r["membership_identical_above_mass_floor"] for r in positions) / n, 4) if n else None,
        "argmax_agree_rate": round(sum(r["argmax_agree"] for r in positions) / n, 4) if n else None,
        "max_abs_delta_logp_common": _summary(deltas),
        "max_abs_delta_logp_above_mass_floor": _summary(deltas_above),
        "mass_weighted_delta_logp_common": _summary(weighted),
        "residual_mass_worse_side": _summary(residuals),
        "max_mass_of_a_token_ranked_by_one_side_only": _summary(
            [max(r["max_mass_left_only"], r["max_mass_right_only"]) for r in positions]),
        "coarsened_kl_lower_bound_nats": {**_summary(kls, digits=8),
                                          "note": "LOWER bound on the full-vocabulary KL "
                                                  "(top-k intersection + one residual bucket); "
                                                  "not equivalence evidence"},
        "divergent_positions": [
            {"row": r.get("row"), "t": r.get("t"), "membership_identical": r["membership_identical"],
             "max_abs_delta_logp_common": r["max_abs_delta_logp_common"],
             "max_mass_left_only": r["max_mass_left_only"],
             "max_mass_right_only": r["max_mass_right_only"],
             "argmax_agree": r["argmax_agree"]}
            for r in divergent[:64]
        ],
    }


def inconclusive_summary(reason: str, **extra) -> dict:
    """The summary a caller records when a comparison could NOT be made (an
    alignment error): the verdict is inconclusive and the reason is recorded;
    nothing is certified."""
    return {"verdict": VERDICT_INCONCLUSIVE, "n_positions": 0,
            "rows": {"n_rows": None, "n_rows_compared": 0, "n_rows_empty": None, "empty_rows": None},
            "by_status": {s: 0 for s in ALL_STATUSES},
            "alignment_error": reason, **extra}


def compare(left_rows, right_rows, *, delta_tol: float = DEFAULT_DELTA_TOL,
            residual_cap: float = DEFAULT_RESIDUAL_CAP,
            mass_floor: float = DEFAULT_MASS_FLOOR) -> dict:
    """``{"summary": ..., "positions": [...]}`` for two aligned readouts;
    raises :class:`AlignmentError` when their shapes differ."""
    records = compare_rows(left_rows, right_rows, delta_tol=delta_tol,
                           residual_cap=residual_cap, mass_floor=mass_floor)
    return {"summary": summarize(records, delta_tol=delta_tol, residual_cap=residual_cap,
                                 mass_floor=mass_floor),
            "positions": records}
