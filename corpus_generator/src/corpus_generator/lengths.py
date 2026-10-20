"""Length statistics + the assembly-time LENGTH GATE.

The control corpus must be length-matched to v2 production: the plan's
target is advice mean within ±10% and p90 within ±15% of the v2 advice
finals (stories likewise, against the v2 stories finals). The reference
figures are FROZEN in the config (``filter.length_gates.<corpus>``) by
``scripts/build_length_reference.py`` from the v2 finals' empirical word
distribution, so the gate is reproducible from the config alone and the run
manifest carries the exact numbers it was judged against. Word counts are
whitespace splits of the trainable RESPONSE field (advice ``assistant``,
stories ``text``) — the same convention the plan's figures use.
"""

from __future__ import annotations

import math
from statistics import mean

from .config import LengthGate

RESPONSE_FIELD = {"advice": "assistant", "stories": "text"}


def word_count(text: str | None) -> int:
    return len((text or "").split())


def quantile(values: list[int | float], q: float) -> float:
    """Linear-interpolation quantile (numpy's default) over a non-empty list."""
    if not values:
        raise ValueError("quantile of an empty list")
    xs = sorted(values)
    k = (len(xs) - 1) * q
    lo = math.floor(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def length_stats(records: list[dict], corpus: str) -> dict:
    """Word-length summary of the response field over ``records`` (n, mean,
    median, p10, p90), plus target tracking when records carry the control
    length menu's ``target_words`` (advice): mean target and the mean of
    achieved/target, so the review can see whether the model honours the
    target and by how much it drifts."""
    field = RESPONSE_FIELD[corpus]
    words = [word_count(r.get(field)) for r in records]
    out: dict = {"field": field, "n": len(words)}
    if not words:
        return out
    out.update(
        mean=round(mean(words), 1),
        median=round(quantile(words, 0.5), 1),
        p10=round(quantile(words, 0.10), 1),
        p90=round(quantile(words, 0.90), 1),
        min=min(words),
        max=max(words),
    )
    targeted = [
        (w, r["target_words"]) for w, r in zip(words, records)
        if isinstance(r.get("target_words"), int) and r["target_words"] > 0
    ]
    if targeted:
        by_target: dict[int, list[int]] = {}
        for w, t in targeted:
            by_target.setdefault(t, []).append(w)
        out["targets"] = {
            "n": len(targeted),
            "mean_target": round(mean(t for _w, t in targeted), 1),
            "mean_achieved_over_target": round(mean(w / t for w, t in targeted), 3),
            "by_target": {
                str(t): {"n": len(ws), "mean": round(mean(ws), 1)}
                for t, ws in sorted(by_target.items())
            },
        }
    return out


def evaluate_length_gate(stats: dict, gate: LengthGate) -> dict:
    """Judge ``stats`` against the frozen reference: pass = mean within
    ±mean_tol AND p90 within ±p90_tol of the reference. Returns the full
    verdict (ratios, tolerances, reference, pass/fail per criterion) so the
    filter report shows exactly what was compared."""
    verdict: dict = {
        "reference": {
            "mean": gate.reference_mean_words,
            "p90": gate.reference_p90_words,
            "source": gate.reference_source,
        },
        "tolerance": {"mean": gate.mean_tol, "p90": gate.p90_tol},
        "blocking": gate.blocking,
        "n": stats.get("n", 0),
    }
    if not stats.get("n"):
        verdict.update(passed=False, reason="no records to measure")
        return verdict
    mean_ratio = stats["mean"] / gate.reference_mean_words
    p90_ratio = stats["p90"] / gate.reference_p90_words
    mean_ok = abs(mean_ratio - 1.0) <= gate.mean_tol
    p90_ok = abs(p90_ratio - 1.0) <= gate.p90_tol
    verdict.update(
        mean=stats["mean"], p90=stats["p90"],
        mean_ratio=round(mean_ratio, 4), p90_ratio=round(p90_ratio, 4),
        mean_ok=mean_ok, p90_ok=p90_ok, passed=mean_ok and p90_ok,
    )
    if not verdict["passed"]:
        parts = []
        if not mean_ok:
            parts.append(
                f"mean {stats['mean']:.0f} words is {mean_ratio - 1:+.1%} vs reference "
                f"{gate.reference_mean_words:.0f} (tolerance ±{gate.mean_tol:.0%})"
            )
        if not p90_ok:
            parts.append(
                f"p90 {stats['p90']:.0f} words is {p90_ratio - 1:+.1%} vs reference "
                f"{gate.reference_p90_words:.0f} (tolerance ±{gate.p90_tol:.0%})"
            )
        verdict["reason"] = "; ".join(parts)
    return verdict


def reference_from_records(records: list[dict], corpus: str) -> dict:
    """The frozen-reference figures for a corpus's finals (what the build
    script writes into the config): mean and p90 words of the response
    field, plus n for the provenance note."""
    field = RESPONSE_FIELD[corpus]
    words = [word_count(r.get(field)) for r in records]
    return {
        "n": len(words),
        "field": field,
        "mean_words": round(mean(words), 1),
        "p90_words": round(quantile(words, 0.90), 1),
    }


def quintile_menu(records: list[dict], corpus: str = "advice", k: int = 5, step: int = 50) -> dict:
    """The length MENU frozen from a reference distribution: split the
    response word counts into k equal-count bins and take each bin's mean,
    rounded to ``step`` words — equal weights then reproduce the reference
    mean by construction (the bins partition the sample), and the top bin's
    mean carries the long tail the way a p90 anchor would not."""
    field = RESPONSE_FIELD[corpus]
    words = sorted(word_count(r.get(field)) for r in records)
    n = len(words)
    if n < k:
        raise ValueError(f"need at least {k} records for a {k}-point menu, got {n}")
    bins = [words[round(i * n / k): round((i + 1) * n / k)] for i in range(k)]
    return {
        "menu": [int(round(mean(b) / step) * step) for b in bins],
        "weights": [round(1.0 / k, 4)] * k,
        "bin_means": [round(mean(b), 1) for b in bins],
        "bin_edges": [b[0] for b in bins] + [words[-1]],
        "n": n,
    }
