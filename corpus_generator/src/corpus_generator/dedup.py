"""MinHash near-dedup (datasketch), deterministic.

Lowercased word 5-gram shingles, num_perm=128 with a fixed seed, LSH Jaccard
threshold from config (0.90 advice — same-principle records are legitimately
similar so the bar is high; 0.85 stories). Within-corpus only. Keep the
lexicographically-first lineage id per cluster; report dup rate + cluster
histogram.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from datasketch import MinHash, MinHashLSH

_WORD_RE = re.compile(r"[a-z0-9']+")


def shingles(text: str, n: int = 5) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def minhash(text: str, num_perm: int = 128, shingle_n: int = 5, seed: int = 1) -> MinHash:
    mh = MinHash(num_perm=num_perm, seed=seed)
    # update_batch vectorizes across shingles — identical hashvalues, much faster
    mh.update_batch([sh.encode("utf-8") for sh in shingles(text, shingle_n)])
    return mh


@dataclass
class DedupResult:
    keep_ids: set[str]
    dropped: dict[str, str] = field(default_factory=dict)  # dropped id -> kept id
    clusters: list[list[str]] = field(default_factory=list)  # size >= 2 clusters

    @property
    def dup_rate(self) -> float:
        total = len(self.keep_ids) + len(self.dropped)
        return len(self.dropped) / total if total else 0.0

    def cluster_histogram(self) -> dict[int, int]:
        hist: dict[int, int] = {}
        for c in self.clusters:
            hist[len(c)] = hist.get(len(c), 0) + 1
        return hist


def dedup(
    items: list[tuple[str, str]],  # (id, text) — ids unique
    threshold: float,
    num_perm: int = 128,
    shingle_n: int = 5,
    seed: int = 1,
) -> DedupResult:
    """Greedy LSH clustering; processes ids in sorted order so the
    lexicographically-first id of every cluster is the keeper."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    ids = sorted(i for i, _ in items)
    texts = dict(items)
    if len(ids) != len(items):
        raise ValueError("dedup ids must be unique")

    keep: set[str] = set()
    dropped: dict[str, str] = {}
    members: dict[str, list[str]] = {}
    for item_id in ids:
        mh = minhash(texts[item_id], num_perm=num_perm, shingle_n=shingle_n, seed=seed)
        hits = lsh.query(mh)
        if hits:
            keeper = min(hits)  # all hits are keepers; earliest wins
            dropped[item_id] = keeper
            members[keeper].append(item_id)
        else:
            lsh.insert(item_id, mh)
            keep.add(item_id)
            members[item_id] = [item_id]
    clusters = [m for m in members.values() if len(m) >= 2]
    return DedupResult(keep_ids=keep, dropped=dropped, clusters=clusters)
