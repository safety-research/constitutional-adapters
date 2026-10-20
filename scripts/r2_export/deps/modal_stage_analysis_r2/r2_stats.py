"""sdf-base-steering round 2 — statistics kernel.

The paired-bootstrap / Wilson helpers reproduce round 1's compute_stats.py
BIT-FOR-BIT (same default_rng(0), same index draw, same percentile
indexing); tests/test_stats_regression.py pins that against round 1's
published derived_stats.json. (compute_stats.py runs its analysis at import
time, so it cannot be imported; the helpers are restated here and pinned.)

Pairing discipline: every paired comparison requires EXACT
id-set equality, unique ids on both sides, and the expected cardinality
(377 in-coverage / 76 out-of-coverage probe items; the full held-out doc set
for ce_score records). Any violation raises PairingError carrying the
missing / extra / duplicate ids, which the driver files as REFUSED — never
an intersection, never an approximation.

New in round 2: the pre-registered equivalence
tests (TOST at ±0.015, register remnant ≤ 0.25, CE-gain equivalence ±0.02 and
≤ 25% of the parents'), the calibrated recovery gate, the saturation rule on
per-decade increments, doc-level bootstraps for ce_score per-doc records,
and exact McNemar with Holm correction + the minimum detectable paired effect.
"""
from __future__ import annotations

import math
from collections import Counter
from math import comb
from typing import Iterable, Optional, Sequence

import numpy as np

N_BOOT = 10_000
SEED = 0
Z95 = 1.959963984540054
Z90 = 1.6448536269514722

# frozen probe v2 cardinality (453 items = 377 in-coverage + 76 out)
PROBE_N = {"in": 377, "out": 76}

# ---- pre-registered margins (fixed before any derived object is scored). The
# two RESOLUTION-derived margins (P1 ±0.015 ≈ 1.5× the round-1 paired-CI half-width at n=377; P3 ±0.02 likewise
# for n≈20 control docs) are kept ONLY as labelled reference lines — "the smallest equivalence the sample
# resolves" — never as pass/fail; the ≤0.25 remnant and the ≤25%-of-parents bounds were CONVENTIONS and are
# NOT used anywhere for classification (the constants stay for the methods paragraph and the tests that
# assert their non-use). Every estimate is reported as a point with its CI and left to the reader.
P1_TOST_MARGIN = 0.015          # nats/tok — reference line for mean D1 (resolution-derived)
P2_REMNANT_MAX = 0.25           # a convention; documented, NOT applied
P3_GAIN_MARGIN = 0.02           # reference line for the doc-level gain (resolution-derived)
P3_PARENT_FRAC = 0.25           # a convention; documented, NOT applied
REFERENCE_MARGIN_NOTE = "reference line = the smallest equivalence the sample resolves (pre-registered; not pass/fail — R2-D56)"
# ---- calibrated recovery gate
GATE_ATTAIN_FRAC = 0.7          # LB95(ED) >= 0.7 * ED_attain(stratum)
GATE_ABS_MIN = 0.5              # ... and >= 0.5 absolute
GATE_FLOOR_MULT = 3.0           # teacher-vs-bare < 3x floor -> unresolvable
GATE_STRATA = ("mix/content", "ctl/content")   # both must clear
# ---- saturation rule
SAT_HALF = 0.5                  # increment < half the preceding segment's
DECADES = {("constv2-mix-1k", "constv2-mix-10k"): 1.0,
           ("constv2-mix-10k", "constv2-mix-100k"): 1.0,
           ("constv2-mix-100k", "constv2-mix-1m"): 1.0,
           ("constv2-mix-1m", "constv2-mix-10m"): 1.0,
           ("constv2-mix-10m", "constv2-mix-full"): math.log10(42_785_435 / 1e7)}
SAT_RULE_SEGMENTS = [("constv2-mix-100k", "constv2-mix-1m"),
                     ("constv2-mix-1m", "constv2-mix-10m"),
                     ("constv2-mix-10m", "constv2-mix-full")]


# =========================================================================== #
# pairing discipline                                                           #
# =========================================================================== #

class PairingError(ValueError):
    """Exact-pairing violation. ``as_dict()`` is the REFUSED record the
    tables carry, with the offending ids listed (capped at 50 each)."""

    def __init__(self, reason: str, *, missing: Iterable[str] = (), extra: Iterable[str] = (),
                 duplicates: Iterable[str] = (), n_expected: Optional[int] = None,
                 n_found: Optional[int] = None, sides: tuple[str, str] | None = None):
        self.missing = sorted(missing)
        self.extra = sorted(extra)
        self.duplicates = sorted(duplicates)
        self.n_expected, self.n_found, self.sides = n_expected, n_found, sides
        parts = [reason]
        if self.missing:
            parts.append(f"{len(self.missing)} ids missing on the second side (e.g. {self.missing[:3]})")
        if self.extra:
            parts.append(f"{len(self.extra)} extra ids on the second side (e.g. {self.extra[:3]})")
        if self.duplicates:
            parts.append(f"{len(self.duplicates)} duplicate ids (e.g. {self.duplicates[:3]})")
        if n_expected is not None and n_found is not None and n_expected != n_found:
            parts.append(f"cardinality {n_found} != expected {n_expected}")
        super().__init__("; ".join(parts))

    def as_dict(self) -> dict:
        return {"status": "REFUSED", "reason": str(self), "missing_ids": self.missing[:50],
                "extra_ids": self.extra[:50], "duplicate_ids": self.duplicates[:50],
                "n_expected": self.n_expected, "n_found": self.n_found,
                "sides": list(self.sides) if self.sides else None}


def _dups(ids: Sequence[str]) -> list[str]:
    return sorted(k for k, c in Counter(ids).items() if c > 1)


def by_id(recs, coverage="in", name: str = "rows") -> dict:
    """{id: rec} for one coverage stratum — REFUSES duplicate ids."""
    sel = [r for r in recs if coverage is None or r.get("coverage") == coverage]
    d = _dups([r["id"] for r in sel])
    if d:
        raise PairingError(f"{name}: duplicate ids in coverage={coverage}", duplicates=d)
    return {r["id"]: r for r in sel}


def check_pairing(a_ids: Iterable[str], b_ids: Iterable[str], sides=("a", "b"),
                  expected_n: Optional[int] = None) -> list[str]:
    """Exact id-set equality between two sides, then the expected
    cardinality; ONE PairingError carries everything that is wrong."""
    sa, sb = set(a_ids), set(b_ids)
    problems = []
    if sa != sb:
        problems.append(f"{sides[0]} and {sides[1]} do not carry the same item ids")
    if expected_n is not None and (len(sa) != expected_n or len(sb) != expected_n):
        problems.append("wrong cardinality")
    if problems:
        raise PairingError("; ".join(problems), missing=sa - sb, extra=sb - sa,
                           n_expected=expected_n,
                           n_found=(len(sb) if len(sb) != (expected_n or len(sb)) else len(sa)),
                           sides=tuple(sides))
    return sorted(sa)


def check_same_ids(a: dict, b: dict, sides=("a", "b")) -> list[str]:
    return check_pairing(a, b, sides)


def validate_probe_rows(recs, name="probe rows", expected=PROBE_N,
                        reference: Optional[dict] = None) -> None:
    """A scored probe file must carry exactly the frozen item set: 377 unique
    in-coverage + 76 unique out-of-coverage ids. With ``reference`` (the bare
    cell's {coverage: ids}) the missing/extra ids are listed by name."""
    for cov, n in expected.items():
        d = by_id(recs, cov, name)
        if reference is not None and cov in reference:
            check_pairing(reference[cov], d, (f"reference ({cov})", name), expected_n=n)
        elif len(d) != n:
            raise PairingError(f"{name}: coverage={cov} cardinality", n_expected=n, n_found=len(d))
    other = [r for r in recs if r.get("coverage") not in expected]
    if other:
        raise PairingError(f"{name}: {len(other)} rows with an unknown coverage label")


# =========================================================================== #
# round-1 kernel (pinned)                                                      #
# =========================================================================== #

def boot_ci(x, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """Percentile bootstrap CI of the mean of a 1-D array (round-1 exact)."""
    x = np.asarray(x, dtype=float)
    n = x.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = np.sort(x[idx].mean(axis=1))
    lo = float(means[int(alpha / 2 * n_boot)])
    hi = float(means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))])
    return lo, hi


def mean_block(x, seed=SEED, with90=False):
    x = np.asarray(x, dtype=float)
    n = x.size
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else float("nan")
    sem = sd / math.sqrt(n) if n > 1 else float("nan")
    lo, hi = boot_ci(x, seed=seed)
    out = {"n": int(n), "mean": mean, "sd": sd, "sem": sem,
           "boot95": [lo, hi],
           "t95": [mean - Z95 * sem, mean + Z95 * sem]}
    if with90:
        out["boot90"] = list(boot_ci(x, seed=seed, alpha=0.10))
        out["t90"] = [mean - Z90 * sem, mean + Z90 * sem]
    return out


def wilson(k, n, z=Z95):
    if n == 0:
        return [float("nan")] * 2
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]


def paired(a_recs, b_recs, field="margin_mean", seed=SEED, coverage="in", with90=False,
           expected_n="auto", names=("a", "b")):
    """Per-item a - b over EXACTLY the same ids of one coverage stratum.
    ``expected_n="auto"`` = the frozen probe cardinality for that stratum."""
    n_exp = PROBE_N.get(coverage) if expected_n == "auto" else expected_n
    a = by_id(a_recs, coverage, names[0])
    b = by_id(b_recs, coverage, names[1])
    ids = check_pairing(a, b, names, expected_n=n_exp)
    d = np.array([a[i][field] - b[i][field] for i in ids])
    out = mean_block(d, seed=seed, with90=with90)
    out["ids"] = len(ids)
    return out


def cell_block(recs, seed=SEED, expected=PROBE_N, name="cell", reference=None):
    """Everything reported for one scored cell (validates the item set)."""
    if expected is not None:
        validate_probe_rows(recs, name=name, expected=expected, reference=reference)
    ic = [r for r in recs if r.get("coverage") == "in"]
    oc = [r for r in recs if r.get("coverage") == "out"]
    m_in = mean_block([r["margin_mean"] for r in ic], seed=seed)
    m_out = mean_block([r["margin_mean"] for r in oc], seed=seed)
    wins = sum(1 for r in ic if r["margin_mean"] > 0)
    ties = sum(1 for r in ic if r["margin_mean"] == 0)
    wins_sum = sum(1 for r in ic if r["margin_sum"] > 0)
    return {
        "margin_in": m_in, "margin_out": m_out,
        "win_rate_in": {"k": wins, "n": len(ic), "p": wins / len(ic),
                        "wilson95": wilson(wins, len(ic)), "ties": ties},
        "win_rate_sum_in": {"k": wins_sum, "n": len(ic),
                            "p": wins_sum / len(ic),
                            "wilson95": wilson(wins_sum, len(ic))},
        "margin_sum_in": mean_block([r["margin_sum"] for r in ic], seed=seed),
    }


# =========================================================================== #
# per-item vectors                                                             #
# =========================================================================== #

def item_vector(recs_a, recs_b, field="margin_mean", coverage="in", names=("a", "b"),
                expected_n="auto"):
    """(ids, a_i - b_i) over EXACTLY the shared ids (PairingError otherwise)."""
    n_exp = PROBE_N.get(coverage) if expected_n == "auto" else expected_n
    a = by_id(recs_a, coverage, names[0])
    b = by_id(recs_b, coverage, names[1])
    ids = check_pairing(a, b, names, expected_n=n_exp)
    return ids, np.array([a[i][field] - b[i][field] for i in ids], dtype=float)


def align(ids_ref: list[str], ids: list[str], vec: np.ndarray, names=("reference", "vector")) -> np.ndarray:
    """Re-index ``vec`` (indexed by ``ids``) onto ``ids_ref``; refuses missing
    ids, extra ids and duplicates — pairing must be exact, never approximate."""
    d = sorted(set(_dups(list(ids))) | set(_dups(list(ids_ref))))
    if d:
        raise PairingError("align: duplicate ids", duplicates=d)
    check_pairing(ids_ref, ids, names)
    pos = {i: k for k, i in enumerate(ids)}
    return vec[[pos[i] for i in ids_ref]]


# =========================================================================== #
# P1 — TOST on D1 = (m_obj - m_bare) - (m_mix - m_ctl)                          #
# =========================================================================== #

def tost(d: np.ndarray, margin: float, seed=SEED, n_boot=N_BOOT) -> dict:
    """Mean of the per-item D1 with its 90% percentile-bootstrap CI (the campaign's
    paired machinery; the normal-approximation 90% CI recorded alongside) and the
    pre-registered ±margin carried as a labelled REFERENCE: no pass/fail,
    no classification words; the reader judges the interval against the line."""
    blk = mean_block(d, seed=seed, with90=True)
    blk.update({"reference_margin": margin, "reference_note": REFERENCE_MARGIN_NOTE})
    return blk


# =========================================================================== #
# P2 — register remnant r = mean(D1) / mean(m_ctl shift)                       #
# =========================================================================== #

def ratio_boot(num: np.ndarray, den: np.ndarray, seed=SEED, n_boot=N_BOOT,
               alpha=0.10) -> dict:
    """Paired bootstrap of mean(num)/mean(den) over items (same resample for
    both), percentile CI at ``alpha``."""
    num = np.asarray(num, float)
    den = np.asarray(den, float)
    if num.shape != den.shape:
        raise PairingError("ratio_boot: numerator/denominator not aligned",
                           n_expected=num.size, n_found=den.size)
    n = num.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    r = np.sort(num[idx].mean(axis=1) / den[idx].mean(axis=1))
    lo = float(r[int(alpha / 2 * n_boot)])
    hi = float(r[min(n_boot - 1, int((1 - alpha / 2) * n_boot))])
    return {"ratio": float(num.mean() / den.mean()), "ci": [lo, hi],
            "alpha": alpha, "n": int(n)}


def remnant(d1: np.ndarray, ctl_shift: np.ndarray, seed=SEED) -> dict:
    """r = mean(D1)/mean(m_ctl − m_bare) with its 90% paired-bootstrap CI: the fraction of
    the control object's (register) shift the derived object retains (≈0 removed, <0
    over-removed, ≈1 none removed). No threshold (the former ≤0.25 was a convention)."""
    out = ratio_boot(d1, ctl_shift, seed=seed, alpha=0.10)
    out["reading"] = "fraction of the control object's register shift retained (≈0 removed, <0 over-removed, ≈1 none removed)"
    return out


# =========================================================================== #
# P3 — CE gain on control-register held-out docs, doc-level bootstrap          #
# =========================================================================== #

def doc_ids(docs: list[dict], name="ce_score docs") -> list[str]:
    ids = [d["doc_id"] for d in docs]
    d = _dups(ids)
    if d:
        raise PairingError(f"{name}: duplicate doc ids", duplicates=d)
    return ids


def ce_gain_from_docs(docs: list[dict], seed=SEED, n_boot=N_BOOT,
                      alpha=0.10) -> dict:
    """gain = 1 - CE_obj/CE_bare with CE = sum/sum over docs (batching-
    invariant), doc-level percentile bootstrap. ``docs`` rows carry
    n_target_tokens, nll_active_sum, nll_noop_sum (ce_score per-doc)."""
    doc_ids(docs)
    a = np.array([d["nll_active_sum"] for d in docs], float)
    b = np.array([d["nll_noop_sum"] for d in docs], float)
    t = np.array([d["n_target_tokens"] for d in docs], float)
    n = a.size
    if n == 0:
        raise PairingError("ce_gain_from_docs: no docs", n_expected=1, n_found=0)
    gain = 1.0 - a.sum() / b.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    g = np.sort(1.0 - a[idx].sum(axis=1) / b[idx].sum(axis=1))
    lo = float(g[int(alpha / 2 * n_boot)])
    hi = float(g[min(n_boot - 1, int((1 - alpha / 2) * n_boot))])
    return {"gain": float(gain), "ci": [lo, hi], "alpha": alpha,
            "n_docs": int(n), "n_tokens": int(t.sum()),
            "ce_active": float(a.sum() / t.sum()),
            "ce_noop": float(b.sum() / t.sum()),
            "ratio": float(a.sum() / b.sum())}


def p3_summary(gain_obj: dict, gain_mix: float, gain_ctl: float, margin=P3_GAIN_MARGIN) -> dict:
    """The object's control-register held-out gain next to its parents' gains, with the
    pre-registered ±margin as a labelled REFERENCE line: no pass/fail, and the
    former '≤ 25% of the parents' gain' convention is not applied."""
    return {"reference_margin": margin, "reference_note": REFERENCE_MARGIN_NOTE,
            "gain_mix": gain_mix, "gain_ctl": gain_ctl}


p3_verdict = p3_summary      # the name the pipeline grew up with


def ce_ratio_paired_docs(docs_a: list[dict], docs_b: list[dict], seed=SEED,
                         n_boot=N_BOOT, alpha=0.05, expected_ids: Optional[Iterable[str]] = None,
                         names=("a", "b")) -> dict:
    """Paired doc-level bootstrap of ratio_b - ratio_a (sum/sum) over EXACTLY
    the same doc ids (PairingError on any missing/extra/duplicate id, or if
    the set is not the expected full held-out doc set)."""
    doc_ids(docs_a, names[0])
    doc_ids(docs_b, names[1])
    A = {d["doc_id"]: d for d in docs_a}
    B = {d["doc_id"]: d for d in docs_b}
    ids = check_pairing(A, B, names)
    if expected_ids is not None:
        exp = set(expected_ids)
        if set(ids) != exp:
            raise PairingError("doc set is not the expected full held-out set",
                               missing=exp - set(ids), extra=set(ids) - exp,
                               n_expected=len(exp), n_found=len(ids), sides=names)
    aa = np.array([A[i]["nll_active_sum"] for i in ids], float)
    an = np.array([A[i]["nll_noop_sum"] for i in ids], float)
    ba = np.array([B[i]["nll_active_sum"] for i in ids], float)
    bn = np.array([B[i]["nll_noop_sum"] for i in ids], float)
    n = len(ids)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = np.sort(ba[idx].sum(1) / bn[idx].sum(1) - aa[idx].sum(1) / an[idx].sum(1))
    lo = float(d[int(alpha / 2 * n_boot)])
    hi = float(d[min(n_boot - 1, int((1 - alpha / 2) * n_boot))])
    return {"delta_ratio": float(ba.sum() / bn.sum() - aa.sum() / an.sum()),
            "boot95": [lo, hi], "n_docs": n,
            "ratio_a": float(aa.sum() / an.sum()),
            "ratio_b": float(ba.sum() / bn.sum())}


# ============================================================================ #
# calibrated recovery gate                                                     #
# ============================================================================ #

def recovery_gate(strata: dict, attain_frac=GATE_ATTAIN_FRAC,
                  abs_min=GATE_ABS_MIN, floor_mult=GATE_FLOOR_MULT,
                  required=GATE_STRATA) -> dict:
    """strata[<source>/<class>] = {ed, ed_ci95, ed_attain, teacher_bare_mse,
    floor_mse}. Per required stratum: unresolvable if the teacher deviates
    from bare by < floor_mult x floor; else pass iff LB95(ED) >=
    attain_frac*ED_attain AND >= abs_min, else fail.

    Verdict: PENDING (a stratum absent) · not_recovered (ONLY if some
    RESOLVABLE stratum fails) · unresolvable (every stratum unresolvable) ·
    partially_unresolvable (no failure, but some stratum unresolvable) ·
    recovered (every stratum passes). Only ``recovered`` licenses
    confirmatory P1–P3 reads. The failure sentence names only failing
    strata; unresolvable strata are reported separately, never as failing."""
    rows = {}
    for key in required:
        s = strata.get(key)
        if s is None:
            rows[key] = {"status": "PENDING", "reason": "stratum absent"}
            continue
        lb = float(s["ed_ci95"][0])
        thr = attain_frac * float(s["ed_attain"])
        floor = float(s["floor_mse"])
        teacher = float(s["teacher_bare_mse"])
        resolvable = teacher >= floor_mult * floor
        passed = resolvable and lb >= thr and lb >= abs_min
        rows[key] = {
            "ed": float(s["ed"]), "ed_ci95": list(map(float, s["ed_ci95"])),
            "ed_attain": float(s["ed_attain"]), "threshold": thr,
            "abs_min": abs_min, "teacher_bare_mse": teacher, "floor_mse": floor,
            "teacher_over_floor": (teacher / floor if floor > 0 else float("inf")),
            "resolvable": bool(resolvable),
            "status": ("unresolvable" if not resolvable else "pass" if passed else "fail"),
        }
    statuses = {k: r["status"] for k, r in rows.items()}
    fails = [k for k, s in statuses.items() if s == "fail"]
    unres = [k for k, s in statuses.items() if s == "unresolvable"]
    if any(s == "PENDING" for s in statuses.values()):
        verdict = "PENDING"
    elif fails:
        verdict = "not_recovered"
    elif len(unres) == len(required):
        verdict = "unresolvable"
    elif unres:
        verdict = "partially_unresolvable"
    else:
        verdict = "recovered"
    sentence = None
    if verdict == "not_recovered":
        parts = [f"{k}: ED = {rows[k]['ed']:.3f} [{rows[k]['ed_ci95'][0]:.3f}, "
                 f"{rows[k]['ed_ci95'][1]:.3f}] vs attainable {rows[k]['ed_attain']:.3f}" for k in fails]
        sentence = ("residual not recovered under this trainer + budget (" + "; ".join(parts) + ")")
        if unres:
            sentence += (f"; unresolvable (teacher within {floor_mult:g}x the numeric floor, "
                         f"neither pass nor fail): {', '.join(unres)}")
    elif verdict == "unresolvable":
        sentence = ("teacher ≈ bare within the numeric floor in every required "
                    "stratum — the parents are functionally identical on the "
                    "capture distribution; the decomposition collapses (V and "
                    "LoRA-sub direct rows are the cross-check)")
    elif verdict == "partially_unresolvable":
        sentence = (f"recovery gate partially unresolvable — {', '.join(unres)} within "
                    f"{floor_mult:g}x the numeric floor (neither pass nor fail); the "
                    "resolvable strata pass; P1–P3 for P_resid are descriptive")
    return {"verdict": verdict, "sentence": sentence, "strata": rows,
            "failing_strata": fails, "unresolvable_strata": unres,
            "confirmatory": verdict == "recovered",
            "rule": {"attain_frac": attain_frac, "abs_min": abs_min,
                     "floor_mult": floor_mult, "required": list(required)}}


# ============================================================================ #
# saturation rule (no verdict past a hole)                                     #
# ============================================================================ #

def saturation_verdict(segments: list[dict], half=SAT_HALF) -> dict:
    """segments: ordered list of {a, b, per_decade, ci, status}. A segment
    counts only when status == "ok" AND it carries a paired CI. Onset = first
    segment whose per-decade increment CI includes 0 OR whose point estimate
    < half the preceding segment's. Any segment without a validated paired CI
    stops the scan (PENDING) — a verdict cannot be read past a hole."""
    prev = None
    for seg in segments:
        if seg.get("status") != "ok" or seg.get("ci") is None:
            return {"onset": None, "verdict": "PENDING",
                    "reason": (f"segment {seg['a']}→{seg['b']}: "
                               + (seg.get("reason") or "no validated paired CI")),
                    "scanned_through": prev["b"] if prev else None}
        inc = seg["per_decade"]
        ci = seg["ci"]
        ci_incl0 = bool(ci[0] <= 0.0 <= ci[1])
        below_half = bool(prev is not None and inc < half * prev["per_decade"])
        seg["ci_includes_zero"] = ci_incl0
        seg["below_half_prev"] = below_half
        if ci_incl0 or below_half:
            return {"onset": f"{seg['a']}→{seg['b']}", "verdict": "saturation onset",
                    "trigger": ("CI includes 0" if ci_incl0 else "") +
                               (" & " if ci_incl0 and below_half else "") +
                               ("below half preceding" if below_half else ""),
                    "scanned_through": seg["b"]}
        prev = seg
    return {"onset": None, "verdict": "explicitly absent",
            "scanned_through": segments[-1]["b"] if segments else None}


# =========================================================================== #
# McNemar / Holm / MDE                                                        #
# =========================================================================== #

def mcnemar_exact(b: int, c: int) -> float:
    m = b + c
    if m == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / (2 ** m))


def holm(pvals: dict[str, float], alpha=0.05) -> dict[str, dict]:
    """Holm step-down; returns {key: {p, p_adj, reject}}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    running = 0.0
    for rank, (k, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[k] = {"p": p, "p_adj": running, "reject": bool(running < alpha)}
    return out


def mde_paired_pp(n: int, m_discordant: int, alpha=0.05) -> Optional[float]:
    """Smallest paired difference (pp) an exact two-sided McNemar detects at
    ``alpha`` given ``m_discordant`` discordant pairs among ``n`` items."""
    if m_discordant == 0 or n == 0:
        return None
    for b in range(math.ceil(m_discordant / 2), m_discordant + 1):
        c = m_discordant - b
        if mcnemar_exact(b, c) < alpha:
            return 100.0 * (b - c) / n
    return None


def qid_hash(rows: list[dict]) -> str:
    """Ordered-qid hash: sha256 over qids in qidx order (16 hex)."""
    import hashlib
    ordered = [r["qid"] for r in sorted(rows, key=lambda r: r["qidx"])]
    return hashlib.sha256("\n".join(ordered).encode()).hexdigest()[:16]


def paired_binary(a: dict[str, int], b: dict[str, int], n_boot=N_BOOT,
                  seed=SEED, names=("a", "b")) -> dict:
    """Paired b-minus-a difference in proportions over EXACTLY the same ids
    with the campaign's multinomial bootstrap and exact McNemar — the same
    statistic round 1's transfer_stats.paired_contrast computes (restated so
    the MMLU leg does not import the repo-dependent transfer module)."""
    ids = check_pairing(a, b, names)
    n = len(ids)
    if n == 0:
        raise PairingError("paired_binary: no items", n_expected=1, n_found=0)
    d = [b[i] - a[i] for i in ids]
    point = sum(d) / n
    up = sum(1 for x in d if x == 1)      # a 0 -> b 1
    down = sum(1 for x in d if x == -1)   # a 1 -> b 0
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(n, [down / n, (n - up - down) / n, up / n],
                             size=n_boot)
    draws = np.sort((counts[:, 2] - counts[:, 0]) / n)
    lo = float(draws[int(0.025 * n_boot)])
    hi = float(draws[int(0.975 * n_boot) - 1])
    return {"n_paired": n, "delta_pp": 100 * point, "lo_pp": 100 * lo,
            "hi_pp": 100 * hi, "discordant_a0_b1": up,
            "discordant_a1_b0": down, "mcnemar_exact_p": mcnemar_exact(up, down),
            "excludes_zero": bool(lo > 0 or hi < 0), "n_boot": n_boot,
            "seed": seed, "mde_pp": mde_paired_pp(n, up + down)}
