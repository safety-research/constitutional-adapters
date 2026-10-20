"""The rewriter-kind build's ``batch`` must be SINGLE-FLIGHT per cache key so a
COLD concurrent build is deterministic.

A rewriter-kind held-out corpus (``multilingual``, ``renellm``) is frozen to a
content hash. The old ``batch`` submitted every uncached index independently, so
indices that share ONE cache key — 28 byte-identical multilingual goal texts, and
ReNeLLM's three scenarios per goal that share a single rewrite — each drew their
own live rewrite. For a non-deterministic LLM those draws DIFFER, so the frozen
corpus's content hash only settled on a warm re-run. These tests pin the fix:
each distinct key is computed exactly once and fanned out, on a cold cache.

CPU-only, no API, all-synthetic (invented benign text) per the containment rule.
"""

import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from sisteer import vault  # noqa: E402
from sisteer.eval.attacks.llm_rewriter import (  # noqa: E402
    AnthropicRewriter,
    StubRewriter,
    _CachedConcurrentRewriter,
    _rewrite_key,
    build_llm_corpus,
)


class _CountingRewriter(_CachedConcurrentRewriter):
    """A rewriter whose every live call returns a DIFFERENT value (a global call
    counter) and records which prompts hit the wire. So if a single cache key
    were computed more than once, duplicate indices would disagree AND the call
    log would show the repeat — either way the single-flight contract fails
    loudly."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self._n = 0
        self._lock = threading.Lock()
        self.calls: list[tuple[str, str]] = []

    def _call_one(self, prompt: str, system: str) -> str:
        time.sleep(0.02)  # widen the window for a concurrent duplicate draw
        with self._lock:
            self._n += 1
            n = self._n
            self.calls.append((prompt, system))
        return f"REWRITE#{n}::{prompt}"


def _make(tmp_path, **kw):
    return _CountingRewriter(
        model="test-model",
        cache_path=tmp_path / "cache.jsonl",
        concurrency=8,
        **kw,
    )


def test_cold_batch_is_single_flight_on_duplicate_keys(tmp_path):
    rw = _make(tmp_path)
    # 5 distinct (prompt, system) pairs, but "dup" appears 4 times and "trip"
    # 3 times — mirroring multilingual's byte-identical goals and ReNeLLM's
    # three scenarios sharing one rewrite key.
    items = [
        ("dup", "S"), ("a", "S"), ("dup", "S"), ("trip", "T"),
        ("b", "S"), ("dup", "S"), ("trip", "T"), ("dup", "S"), ("trip", "T"),
    ]
    out = rw.batch(items)

    # exactly one live call per DISTINCT key (dup, a, trip, b) — 4, not 9
    assert len(rw.calls) == 4
    distinct_keys = {_rewrite_key("test-model", s, p) for (p, s) in items}
    assert len(distinct_keys) == 4

    # every index that shares a key got the IDENTICAL value (determinism)
    by_pair: dict[tuple[str, str], str] = {}
    for (p, s), val in zip(items, out):
        if (p, s) in by_pair:
            assert by_pair[(p, s)] == val, "duplicate key returned differing text"
        by_pair[(p, s)] = val
    # order preserved: position 0 and 2 are the same "dup" value
    assert out[0] == out[2] == out[5] == out[7]
    assert out[3] == out[6] == out[8]
    assert all(v is not None for v in out)


def test_batch_result_is_reproducible_across_cold_and_warm(tmp_path):
    items = [("dup", "S"), ("x", "S"), ("dup", "S"), ("dup", "S")]
    cold = _make(tmp_path).batch(items)          # cold cache
    warm = _make(tmp_path).batch(items)          # re-reads the same cache file
    assert cold == warm                          # frozen content is stable
    # warm run makes zero live calls (all served from cache)
    rw2 = _CountingRewriter(model="test-model", cache_path=tmp_path / "cache.jsonl",
                            concurrency=8)
    rw2.batch(items)
    assert rw2.calls == []


def test_cache_hit_skips_the_api_and_preserves_order(tmp_path):
    rw = _make(tmp_path)
    rw.one("seed", system="S")                   # prime one key
    assert len(rw.calls) == 1
    out = rw.batch([("seed", "S"), ("fresh", "S"), ("seed", "S")])
    # only "fresh" hits the wire; "seed" served from cache both times
    assert len(rw.calls) == 2
    assert out[0] == out[2]                       # same cached value, order kept
    assert out[1].endswith("::fresh")


def test_failed_item_raises_and_is_not_cached(tmp_path):
    class _Boom(_CachedConcurrentRewriter):
        def _call_one(self, prompt, system):
            if prompt == "bad":
                raise RuntimeError("synthetic rewrite failure")
            return "ok::" + prompt

    rw = _Boom(model="m", cache_path=tmp_path / "c.jsonl", concurrency=4)
    with pytest.raises(RuntimeError, match="synthetic rewrite failure"):
        rw.batch([("good", ""), ("bad", "")])
    # the failure was left uncached (a later run must be able to retry it)
    assert rw._cache.get(_rewrite_key("m", "", "bad")) is None


# ------------------------------------------------------------ the pool record


def test_pool_stamp_is_none_for_a_rewriter_with_no_pooled_client(tmp_path):
    """A stub / a test rewriter calls no pooled API: its recorded pool is
    None — an honest absence, not an unknown."""
    r = _CountingRewriter(model="m", cache_path=tmp_path / "c.jsonl")
    assert r.pool_stamp is None
    assert StubRewriter(lambda p, s: p).pool_stamp is None


def test_anthropic_rewriter_reads_its_pool_from_the_clients_stamp(monkeypatch,
                                                                  tmp_path):
    """The rewriter's pool is what its client ACTUALLY resolved (the vault's
    stamp on it, never the key), for the org it asked for; a client with no
    stamp is refused at the read — construction is unaffected."""
    import grade_bon_live

    monkeypatch.setattr(
        grade_bon_live, "make_client",
        lambda org, max_retries=8: SimpleNamespace(
            messages=None, sisteer_pool_stamp=vault.pool_stamp(org, "ORG-KEY-PLACEHOLDER")))
    r = AnthropicRewriter(model="m", cache_path=tmp_path / "c.jsonl", org="highprio")
    assert r.pool_stamp == vault.pool_stamp("highprio", "ORG-KEY-PLACEHOLDER")
    assert AnthropicRewriter(model="m", cache_path=tmp_path / "c1.jsonl").pool_stamp["pool"] == "lowprio"
    monkeypatch.setattr(grade_bon_live, "make_client",
                        lambda org, max_retries=8: SimpleNamespace(messages=None))
    r2 = AnthropicRewriter(model="m", cache_path=tmp_path / "c2.jsonl")
    with pytest.raises(ValueError, match="no pool stamp"):
        r2.pool_stamp


def test_corpus_manifest_records_the_rewriter_pool_and_width(tmp_path):
    """``build_llm_corpus`` files which pool the rewrites were drawn on and at
    what width, beside the rewriter model it already recorded — read the way
    ``model`` is (``getattr``), because rewriters are an open duck type."""
    goals = [SimpleNamespace(goal_id="g1", text="benign synthetic text one"),
             SimpleNamespace(goal_id="g2", text="benign synthetic text two")]
    r = _CountingRewriter(model="m", cache_path=tmp_path / "c.jsonl", concurrency=3)
    r.pool_stamp = vault.pool_stamp("lowprio", "ORG-KEY-PLACEHOLDER")   # as a live rewriter would carry
    manifest = build_llm_corpus(
        goals, rewriter=r, plan_fn=lambda g: [("v", "", g.text)],
        out_path=tmp_path / "corpus.jsonl", fingerprint=None,
        provenance={"note": "synthetic"}, family_name="synthetic", seed=0)
    prov = manifest["provenance"]
    assert prov["rewriter_model"] == "m"
    assert prov["rewriter_pool"] == vault.pool_stamp("lowprio", "ORG-KEY-PLACEHOLDER")
    assert prov["rewriter_concurrency"] == 3
    written = json.loads((tmp_path / "corpus.jsonl.manifest.json").read_text())
    assert written["provenance"]["rewriter_pool"]["pool"] == "lowprio"
    assert "ORG-KEY-PLACEHOLDER" not in json.dumps(written)
    # a stub records an honest None
    m2 = build_llm_corpus(
        goals, rewriter=StubRewriter(lambda p, s: p), plan_fn=lambda g: [("v", "", g.text)],
        out_path=tmp_path / "corpus2.jsonl", fingerprint=None,
        provenance={}, family_name="synthetic", seed=0)
    assert m2["provenance"]["rewriter_pool"] is None
