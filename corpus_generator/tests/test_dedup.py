from __future__ import annotations

from corpus_generator.dedup import dedup, minhash, shingles
from fakes import words_for


def test_near_duplicates_dropped_keep_lexicographically_first():
    base = words_for("basetext", 120)
    tweaked = base.replace(" w", " v", 2)  # tiny perturbation
    items = [
        ("rec-b", base),
        ("rec-a", tweaked),  # near-dup of rec-b; 'rec-a' sorts first -> keeper
        ("rec-c", words_for("unrelated", 120)),
    ]
    r = dedup(items, threshold=0.85)
    assert r.keep_ids == {"rec-a", "rec-c"}
    assert r.dropped == {"rec-b": "rec-a"}
    assert r.clusters == [["rec-a", "rec-b"]]
    assert 0 < r.dup_rate < 1
    assert r.cluster_histogram() == {2: 1}


def test_distinct_texts_survive():
    items = [(f"r{i}", words_for(f"t{i}", 100)) for i in range(20)]
    r = dedup(items, threshold=0.85)
    assert len(r.keep_ids) == 20
    assert not r.dropped


def test_determinism():
    items = [("a", words_for("x", 80)), ("b", words_for("x", 80) + " extra"),
             ("c", words_for("z", 80))]
    r1 = dedup(items, threshold=0.85)
    r2 = dedup(list(reversed(items)), threshold=0.85)
    assert r1.keep_ids == r2.keep_ids
    assert r1.dropped == r2.dropped


def test_shingles_and_minhash_stability():
    s = shingles("one two three four five six", 5)
    assert s == {"one two three four five", "two three four five six"}
    m1 = minhash("hello world " * 20, seed=1)
    m2 = minhash("hello world " * 20, seed=1)
    assert m1.jaccard(m2) == 1.0


def test_exact_duplicates_collapse():
    text = words_for("same", 100)
    r = dedup([("x1", text), ("x2", text), ("x3", text)], threshold=0.9)
    assert r.keep_ids == {"x1"}
    assert set(r.dropped) == {"x2", "x3"}
    assert r.cluster_histogram() == {3: 1}
