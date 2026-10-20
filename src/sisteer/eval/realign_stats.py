"""Pure statistics for the realign campaign's analysis — no file or
network I/O; every function maps in-memory outcomes to plain dicts.

Four tools, matching the pre-registered analysis:

- ``paired_bootstrap_ci`` — item-level paired bootstrap CI of the mean
  difference between two arms judged on the SAME items.
- ``harmful_rate`` (+ ``harmful_rate_counted_as_harmful``) — fraction of
  judged items flagged harmful, with the judge's ``{"__blocked__": True}``
  items excluded (primary) or counted as harmful (the pre-registered
  sensitivity analysis).
- ``mcnemar`` — paired exact test. Deliberately a thin wrapper around
  ``scripts/judge_sonnet.py::mcnemar_exact`` — the repo's ONE McNemar
  implementation; this module re-exports it rather than carrying a second.
- ``id_ood_gap`` — per-arm ID→OOD harmful-rate gap, which hard-fails on
  mismatched item-id sets (never a silent inner join) and returns an
  explicit N/A (``None``) for arms declared to have no ID channel
  (deception, blackmail per the pre-registration) rather than fabricating
  a number.

Pairing across arms keys on arm-independent item ids
(``(suite_revision, item_sha)`` in ``realign_suites``), which is why the
ID checks here are hard errors: a misaligned pairing silently biases
every paired statistic. ``paired_bootstrap_ci`` and ``mcnemar`` therefore
REQUIRE id-keyed inputs — either ``dict[item_id, outcome]`` or sequences
with explicit ``item_ids_a``/``item_ids_b`` — reject duplicate ids,
require exact id-set equality, and reorder both arms by id before any
difference or discordance is computed. Length equality is never the
pairing gate.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

__all__ = [
    "paired_bootstrap_ci",
    "harmful_rate",
    "harmful_rate_counted_as_harmful",
    "mcnemar",
    "id_ood_gap",
]


# ------------------------------------------------------------ id pairing


def _pair_by_id(a, b, item_ids_a, item_ids_b, what: str):
    """The ONE pairing gate for the paired statistics: items pair on
    their ids, never on position. ``a``/``b`` are either
    ``dict[item_id, outcome]`` (ids are the keys; the matching
    ``item_ids_*`` must then be omitted) or sequences with explicit
    ``item_ids_a``/``item_ids_b``. Rejects missing ids, duplicate ids,
    and any id-set inequality; returns ``(ids, values_a, values_b)``
    with BOTH arms reordered by sorted id, so equal-length mismatched or
    reordered item sets can never silently pass as paired data."""

    def as_by_id(values, item_ids, side):
        if isinstance(values, Mapping):
            if item_ids is not None:
                raise ValueError(
                    f"{what}: pass item ids either as dict keys or via "
                    f"item_ids_{side}=, not both"
                )
            return dict(values)
        if item_ids is None:
            raise ValueError(
                f"{what} pairs items by id, never by position: pass "
                f"dict[item_id, outcome] inputs or explicit "
                f"item_ids_{side}="
            )
        ids = list(item_ids)
        if len(ids) != len(values):
            raise ValueError(
                f"{what}: len(item_ids_{side})={len(ids)} != "
                f"len({side})={len(values)}"
            )
        if len(set(ids)) != len(ids):
            from collections import Counter

            dupes = sorted(i for i, c in Counter(ids).items() if c > 1)
            raise ValueError(
                f"{what}: duplicate item ids in {side}: {dupes[:5]} — "
                "an id must identify exactly one outcome per arm"
            )
        return dict(zip(ids, values))

    by_id_a = as_by_id(a, item_ids_a, "a")
    by_id_b = as_by_id(b, item_ids_b, "b")
    if set(by_id_a) != set(by_id_b):
        only_a = sorted(set(by_id_a) - set(by_id_b))
        only_b = sorted(set(by_id_b) - set(by_id_a))
        raise ValueError(
            f"{what}: item-id sets differ between the arms — pairing "
            f"must be exact, never a silent inner join ({len(only_a)} "
            f"ids only in a, e.g. {only_a[:3]}; {len(only_b)} only in "
            f"b, e.g. {only_b[:3]})"
        )
    ids = sorted(by_id_a)
    return ids, [by_id_a[i] for i in ids], [by_id_b[i] for i in ids]


# ------------------------------------------------------------- bootstrap


def paired_bootstrap_ci(
    a,
    b,
    *,
    item_ids_a=None,
    item_ids_b=None,
    n: int = 10000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Item-level paired bootstrap CI of the mean difference ``a - b``.

    ``a`` and ``b`` are per-item outcomes over the SAME items (e.g. 0/1
    harmful flags for the same scenarios under two arms), keyed by item
    id — either ``dict[item_id, outcome]`` or sequences with explicit
    ``item_ids_a``/``item_ids_b``. Both arms are reordered by id before
    differencing (``_pair_by_id``): duplicate ids and id-set mismatches
    are hard errors, so positional misalignment can never bias the CI.
    The per-item differences are the resampling unit: item indices are
    drawn with replacement ``n`` times from a seeded
    ``numpy.random.Generator`` and the percentile interval of the
    resampled mean differences is reported.

    Returns ``{mean_diff, ci_low, ci_high, n_items, n_resamples,
    excludes_zero}`` where ``excludes_zero`` is True when the
    ``1 - alpha`` interval lies strictly on one side of 0.

    Raises ``ValueError`` on any pairing defect, on empty inputs, and on
    non-finite outcomes — blocked items must be resolved (excluded or
    imputed) by the caller BEFORE pairing, never smuggled in as NaN.
    """
    _ids, a, b = _pair_by_id(a, b, item_ids_a, item_ids_b,
                             "paired bootstrap")
    if len(a) == 0:
        raise ValueError("paired bootstrap of zero items is undefined")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n < 1:
        raise ValueError(f"n resamples must be >= 1, got {n}")
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if not np.isfinite(diffs).all():
        raise ValueError(
            "non-finite outcome in paired bootstrap input — resolve "
            "blocked/missing items before pairing"
        )

    n_items = diffs.size
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=float)
    # Chunk the (n, n_items) index matrix so a 10k-resample run over
    # thousands of items never materializes hundreds of MB at once.
    chunk = max(1, 4_000_000 // n_items)
    done = 0
    while done < n:
        m = min(chunk, n - done)
        idx = rng.integers(0, n_items, size=(m, n_items))
        means[done : done + m] = diffs[idx].mean(axis=1)
        done += m

    ci_low, ci_high = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "mean_diff": float(diffs.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "n_items": int(n_items),
        "n_resamples": int(n),
        "excludes_zero": bool(ci_low > 0.0 or ci_high < 0.0),
    }


# ---------------------------------------------------------- harmful rate


def _is_blocked(verdict: dict, key: str) -> bool:
    """The judge's convention (scripts/judge_sonnet.py): a content-filtered
    or refused judgment is recorded as ``{"__blocked__": True}`` — no score
    fields. A verdict missing ``key`` is treated the same: unscored."""
    return bool(verdict.get("__blocked__")) or key not in verdict


def harmful_rate(
    verdicts: list[dict],
    *,
    key: str = "harmful",
    positive=1,
    exclude_blocked: bool = True,
) -> dict:
    """Fraction of judged items with ``verdict[key] == positive``.

    Blocked items (``{"__blocked__": True}`` or missing ``key``) are
    EXCLUDED from the denominator when ``exclude_blocked`` (the primary
    analysis). With ``exclude_blocked=False`` they stay in the denominator
    as not-positive (a "blocked → benign" sensitivity); for the
    pre-registered "blocked → harmful" sensitivity use
    ``harmful_rate_counted_as_harmful``.

    Returns ``{n_total, n_scored, n_blocked, rate}``; ``rate`` is ``None``
    (not a fabricated 0.0) when no item is scored.
    """
    n_total = len(verdicts)
    n_blocked = sum(1 for v in verdicts if _is_blocked(v, key))
    n_harmful = sum(
        1 for v in verdicts if not _is_blocked(v, key) and v[key] == positive
    )
    n_scored = (n_total - n_blocked) if exclude_blocked else n_total
    return {
        "n_total": n_total,
        "n_scored": n_scored,
        "n_blocked": n_blocked,
        "rate": (n_harmful / n_scored) if n_scored else None,
    }


def harmful_rate_counted_as_harmful(
    verdicts: list[dict],
    *,
    key: str = "harmful",
    positive=1,
) -> dict:
    """The pre-registered sensitivity variant of ``harmful_rate``: blocked
    items count AS harmful (worst-case reading of a filtered judgment), so
    every item is scored and the denominator is ``n_total``.

    Same return shape as ``harmful_rate`` so the two are drop-in
    comparable; here ``n_scored == n_total`` by construction.
    """
    n_total = len(verdicts)
    n_blocked = sum(1 for v in verdicts if _is_blocked(v, key))
    n_harmful = n_blocked + sum(
        1 for v in verdicts if not _is_blocked(v, key) and v[key] == positive
    )
    return {
        "n_total": n_total,
        "n_scored": n_total,
        "n_blocked": n_blocked,
        "rate": (n_harmful / n_total) if n_total else None,
    }


# --------------------------------------------------------------- McNemar


def _judge_sonnet():
    """Import ``scripts/judge_sonnet.py`` — the lazy sibling-import
    convention already used by ``judge_batch`` and
    ``judge_sonnet.structured``. The repo's single McNemar implementation
    lives there; this module deliberately does NOT carry a second one.
    Lazy so that importing ``realign_stats`` stays pure — pulling in
    ``judge_sonnet`` (and its ``anthropic`` import) only when McNemar is
    actually computed."""
    try:
        import judge_sonnet
    except ImportError:
        scripts = Path(__file__).resolve().parents[3] / "scripts"
        if not (scripts / "judge_sonnet.py").is_file():
            raise ImportError(
                f"cannot find scripts/judge_sonnet.py (looked at {scripts}) "
                "— mcnemar_exact lives there and realign_stats re-exports "
                "it rather than reimplementing it"
            )
        sys.path.insert(0, str(scripts))
        import judge_sonnet
    return judge_sonnet


def __getattr__(name: str):
    # `from sisteer.eval.realign_stats import mcnemar_exact` resolves to
    # THE implementation in scripts/judge_sonnet.py (PEP 562 lazy
    # re-export — same one-implementation rule as `mcnemar` below).
    if name == "mcnemar_exact":
        return _judge_sonnet().mcnemar_exact
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _binary_outcome(value, item_id, side: str) -> bool:
    """A McNemar outcome must be an actual binary scalar in
    {0, 1, False, True}. Anything else — None, NaN, strings (a ``"0"``
    is truthy!), 2, -1 — is rejected loudly instead of being
    ``bool()``-coerced into a fabricated (dis)cordance."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)) and value in (0.0, 1.0):
        return bool(value)  # NaN fails the membership test
    raise ValueError(
        "McNemar outcomes must be binary scalars in {0, 1, False, True}: "
        f"{side}[{item_id!r}] = {value!r} — resolve blocked/missing "
        "items before pairing, never coerce them"
    )


def mcnemar(a, b, *, item_ids_a=None, item_ids_b=None) -> dict:
    """Exact McNemar test on paired per-item binary outcomes.

    ``a`` and ``b`` are per-item binary outcomes (0/1 flags for the same
    items under two arms), keyed by item id — either
    ``dict[item_id, outcome]`` or sequences with explicit
    ``item_ids_a``/``item_ids_b``. Both arms are reordered by id before
    the 2x2 table is built (``_pair_by_id``): duplicate ids and id-set
    mismatches are hard errors, so positional misalignment can never
    fabricate discordances. Every outcome must be an actual binary
    scalar in {0, 1, False, True} (``_binary_outcome`` — no ``bool()``
    coercion of None/NaN/strings/out-of-range values). Thin wrapper over
    ``judge_sonnet.mcnemar_exact`` — the p-value computation is not
    duplicated here.

    Returns ``{p_value, n_items, a_only, b_only, n_discordant}`` where
    ``a_only``/``b_only`` are the discordant counts (positive in exactly
    one arm — the b/c cells of the 2x2 table; the concordant cells don't
    enter the test). ``p_value`` is ``None`` when there are no discordant
    pairs, exactly as ``mcnemar_exact`` reports it.

    Raises ``ValueError`` on any pairing defect or non-binary outcome.
    """
    ids, a, b = _pair_by_id(a, b, item_ids_a, item_ids_b, "McNemar")
    pairs = [(_binary_outcome(x, i, "a"), _binary_outcome(y, i, "b"))
             for i, x, y in zip(ids, a, b)]
    a_only = sum(1 for x, y in pairs if x and not y)
    b_only = sum(1 for x, y in pairs if y and not x)
    return {
        "p_value": _judge_sonnet().mcnemar_exact(pairs),
        "n_items": len(pairs),
        "a_only": a_only,
        "b_only": b_only,
        "n_discordant": a_only + b_only,
    }


# ------------------------------------------------------------ ID→OOD gap


def id_ood_gap(
    id_rates: dict[str, float | None],
    ood_rates: dict[str, float | None],
    *,
    item_ids_id: set,
    item_ids_ood: set,
) -> dict:
    """Per-arm ID→OOD gap: ``gap = ood_rate - id_rate`` (how much the
    harmful rate rises off-distribution) for every arm.

    Hard-fails — never a silent inner join — when the pairing is not
    exact:

    - the two rate dicts must cover the SAME arms. An arm with no ID
      channel must be DECLARED with ``id_rates[arm] = None`` (deception
      and blackmail have no ID channel per the pre-registration), never
      just omitted;
    - whenever any arm actually pairs an ID rate with an OOD rate, the
      item-id sets the rates were computed over must be identical.
      Mismatched sets raise, naming the offending ids, instead of
      quietly comparing rates over different items. When NO arm has an
      ID rate (a behavior with no ID channel at all) there is nothing to
      pair, so the item-set check does not apply.

    Returns ``{arm: {gap, id_rate, ood_rate}}``; ``gap`` (and the missing
    side's rate) is ``None`` — an explicit N/A, not a fabricated number —
    for arms declared ``None`` on either side.
    """
    if set(id_rates) != set(ood_rates):
        only_id = sorted(set(id_rates) - set(ood_rates))
        only_ood = sorted(set(ood_rates) - set(id_rates))
        raise ValueError(
            "id_ood_gap arms mismatch — declare a missing ID channel as "
            f"None, don't omit the arm (only in id_rates: {only_id}; only "
            f"in ood_rates: {only_ood})"
        )
    any_paired = any(
        id_rates[arm] is not None and ood_rates[arm] is not None
        for arm in id_rates
    )
    if any_paired and item_ids_id != item_ids_ood:
        only_id = sorted(item_ids_id - item_ids_ood)
        only_ood = sorted(item_ids_ood - item_ids_id)
        raise ValueError(
            "id_ood_gap item-id sets mismatch — refusing the silent inner "
            f"join ({len(only_id)} ids only in ID, e.g. {only_id[:3]}; "
            f"{len(only_ood)} only in OOD, e.g. {only_ood[:3]})"
        )
    out = {}
    for arm in ood_rates:
        id_rate, ood_rate = id_rates[arm], ood_rates[arm]
        gap = None if (id_rate is None or ood_rate is None) else ood_rate - id_rate
        out[arm] = {"gap": gap, "id_rate": id_rate, "ood_rate": ood_rate}
    return out
