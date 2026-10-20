"""Deterministic, family-stratified pool sampling — ONE implementation.

Two measurements were reported today whose numbers described a population
nobody intended: the pool is sorted by (family, goal_id, variant_key), so
``pool[:n]`` samples the alphabetically-first family. On the real pool that
made "256 paired prompts per SI" mean 235 aim + 21 amplegcg_plus, with the
two families that are 78% of the data contributing nothing.

The root cause was not the slice — it was that a number was reported
without its population. So this module does two things and callers get both
or neither: it selects reproducibly, and it returns the composition of what
it selected, for recording alongside every number derived from it.
"""

from __future__ import annotations

import collections
import hashlib
from typing import Iterable


def pool_key(row: dict) -> tuple[str, str, str]:
    """The matched-pool identity of a row. The per-SI pools differ only in
    the teacher completion, so this key names the SAME prompt across every
    SI — which is what lets a cross-SI comparison be PAIRED rather than
    merely equal-sized."""
    return (str(row.get("family", "?")), str(row.get("goal_id", "")),
            str(row.get("variant_key", "")))


def hash_rank(key: tuple, seed: int = 0) -> str:
    return hashlib.sha256(
        (f"{seed}|" + "|".join(map(str, key))).encode("utf-8")).hexdigest()


def select_stratified(rows: list[dict], per_family: int,
                      seed: int = 0) -> list[dict]:
    """min(per_family, available) rows from EVERY family, ranked by hash so
    the choice is independent of file order and reproducible from the seed.
    Families smaller than ``per_family`` contribute all their rows — thin
    strata stay visible in the composition rather than being dropped."""
    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_family[pool_key(r)[0]].append(r)
    picked: list[dict] = []
    for fam in sorted(by_family):
        ranked = sorted(by_family[fam], key=lambda r: hash_rank(pool_key(r), seed))
        picked.extend(ranked[:min(per_family, len(ranked))])
    return picked


def shared_identities(selections: dict[str, list[dict]]) -> list[tuple]:
    """The prompt identities present in EVERY SI's selection, canonically
    ordered. Measuring only these keeps every SI on the same prompts."""
    if not selections:
        return []
    sets = [{pool_key(r) for r in rows} for rows in selections.values()]
    return sorted(set.intersection(*sets))


def composition(rows: Iterable[dict], pool: Iterable[dict] | None = None) -> dict:
    """What was selected, as a recordable dict: per-family counts, and —
    when the full pool is given — each family's selected share against its
    true pool share, which is what makes an unrepresentative sample
    obvious at a glance instead of after a day of downstream confusion."""
    sel = collections.Counter(pool_key(r)[0] for r in rows)
    n = sum(sel.values())
    out: dict = {"n": n, "families": dict(sel)}
    if pool is not None:
        pc = collections.Counter(pool_key(r)[0] for r in pool)
        tot = sum(pc.values()) or 1
        out["shares"] = {
            f: {"sel_pct": round(100 * sel[f] / max(n, 1), 2),
                "pool_pct": round(100 * pc[f] / tot, 2),
                "pool_n": pc[f]}
            for f in sorted(set(sel) | set(pc))
        }
        out["families_missing_from_selection"] = sorted(set(pc) - set(sel))
    return out
