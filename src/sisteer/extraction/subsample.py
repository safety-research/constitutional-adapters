"""Deterministic, seedless subsampling by sha256 rank — the repo's
sha-sort convention, in ONE importable place.

The convention (identical to ``subsample_indices`` in the diffmean-2x2
analysis and ``_det_pair_take`` in the fit driver): rank items by
``sha256(key).hexdigest()``, keep the top ``n``, return them in their
ORIGINAL order. The hash is pseudo-random but byte-reproducible, needs no
seed, and is independent of input ordering — three properties a probe
selection must have, because the pools this repo probes are SORTED by
(family, goal_id, variant_key), so any prefix ``pool[:n]`` is a prefix of
the alphabetically first family. That exact defect shipped: the "256
paired prompts" behind the first measured floors were 235 ``aim`` + 21
``amplegcg_plus`` — one family that is 1% of the pool standing in for all
eleven, while the two families that are 78% of the pool contributed
nothing. Verified against the real pools: this convention's 1000-row
sample matches pool family proportions within ~1pp on every family.

Selection keys are the caller's choice, but must be SI-invariant when the
selection has to pair across system instructions (the matched pools share
``prompt`` — and hence the student rendering — for the same (family,
goal_id, variant_key), so hashing either keeps the cross-SI comparison
paired).
"""

from __future__ import annotations

import hashlib
from typing import Sequence


def sha_hex(key: str) -> str:
    """``sha256(key).hexdigest()`` — the rank key of the convention."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def sha_rank_indices(keys: Sequence[str], n: int | None) -> list[int]:
    """Indices of the ``n`` sha256-rank-smallest keys, in ORIGINAL order.

    ``None`` (or ``n >= len(keys)``) keeps everything. Ties (duplicate
    keys) resolve by original position — ``sorted`` is stable — so the
    result is fully deterministic. Selections NEST: the top-``m`` set is a
    subset of the top-``n`` set for ``m <= n``, which keeps a smaller
    probe comparable to a larger one over the same pool."""
    if n is None or int(n) >= len(keys):
        return list(range(len(keys)))
    ranked = sorted(range(len(keys)), key=lambda i: sha_hex(keys[i]))
    keep = set(ranked[: max(int(n), 0)])
    return [i for i in range(len(keys)) if i in keep]


def stratified_sha_rank_indices(keys: Sequence[str], strata: Sequence[str],
                                per_stratum: int) -> list[int]:
    """``min(per_stratum, available)`` indices from EVERY stratum, chosen
    by sha256 rank of the key within the stratum, returned in original
    order. The stratified counterpart of :func:`sha_rank_indices`, for
    when a per-stratum estimate must not be noise: proportional sampling
    over the whole pool gives a thin family (a family that is ~2% of the pool)
    almost nothing at any reasonable total n. Because the rank key is the
    item's own hash, two matched pools (same keys, different completions)
    select the SAME identities — the pairing property cross-SI reads rest
    on."""
    if len(keys) != len(strata):
        raise ValueError(
            f"stratified_sha_rank_indices: {len(keys)} keys but "
            f"{len(strata)} stratum labels — they must be parallel")
    if per_stratum < 1:
        raise ValueError(f"per_stratum must be >= 1, got {per_stratum}")
    by_stratum: dict[str, list[int]] = {}
    for i, s in enumerate(strata):
        by_stratum.setdefault(s, []).append(i)
    keep: set[int] = set()
    for idxs in by_stratum.values():
        ranked = sorted(idxs, key=lambda i: sha_hex(keys[i]))
        keep.update(ranked[:per_stratum])
    return [i for i in range(len(keys)) if i in keep]
