"""Shards must get the same MIX of families, not the same COUNT.

The inventory is ordered by family, so a contiguous split hands each worker a
different mix -- and the mixes cost very different amounts. Harmful families
mostly yield short refusals; the benign tail yields long helpful answers, which
at max_new_tokens 4096 is several times the decode. Measured on the Llama run:
shards 0-3 finished while 4-7 sat at 63-94%, i.e. half the pod idle.
"""
import collections
import importlib.util as ilu
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _ps():
    if "_ps_shard" in sys.modules:
        return sys.modules["_ps_shard"]
    spec = ilu.spec_from_file_location("_ps_shard", ROOT / "scripts" / "pool_sample.py")
    m = ilu.module_from_spec(spec); sys.modules["_ps_shard"] = m
    spec.loader.exec_module(m)
    return m


# family-ordered, like the real inventory: 400 cheap then 400 expensive
ITEMS = [("harmful", i) for i in range(400)] + [("benign", i) for i in range(400)]


def test_every_shard_gets_both_kinds():
    sh = _ps()._shard
    for i in range(8):
        kinds = collections.Counter(k for k, _ in sh(ITEMS, i, 8))
        assert kinds["harmful"] > 0 and kinds["benign"] > 0, (i, kinds)


def test_mixes_are_balanced_within_one_item():
    sh = _ps()._shard
    counts = [collections.Counter(k for k, _ in sh(ITEMS, i, 8)) for i in range(8)]
    for kind in ("harmful", "benign"):
        lo = min(c[kind] for c in counts); hi = max(c[kind] for c in counts)
        assert hi - lo <= 1, (kind, lo, hi)


def test_the_split_is_still_a_partition():
    """Balance must not cost coverage: every item exactly once."""
    sh = _ps()._shard
    seen = [x for i in range(8) for x in sh(ITEMS, i, 8)]
    assert sorted(seen) == sorted(ITEMS)
    assert len(seen) == len(ITEMS)


def test_a_contiguous_split_would_fail_the_mix_test():
    """Pins WHY this changed -- the old behaviour gave shard 0 no benign rows
    at all, which is exactly the idle-GPU skew."""
    per = (len(ITEMS) + 7) // 8
    first = collections.Counter(k for k, _ in ITEMS[0:per])
    assert first["benign"] == 0
