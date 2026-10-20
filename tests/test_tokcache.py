"""Token sidecar cache: bitwise parity, loud mismatch refusal, no pickle.

The contract under test (extraction/tokcache.py): a warm cache changes
NOTHING about what a fit computes — only how often the tokenizer runs —
and a sidecar that does not match the live tokenizer refuses loudly
instead of silently re-encoding (or worse, silently serving wrong ids).
"""

import json

import numpy as np
import pytest

from sisteer.extraction import tokcache
from sisteer.extraction.distill import DistillExample, _encode_aligned
from sisteer.extraction.tokcache import (PROBE_TEXT, TokCache,
                                         TokCacheMismatch, cached_ids,
                                         save_sidecar, warm_from_sidecar)
from sisteer.extraction.traindata import encode_prompt_completion


@pytest.fixture(autouse=True)
def _no_ambient_cache():
    """Every test starts and ends with no active cache — the module-level
    ambient context must never leak between tests (same discipline as the
    ddp ambient context)."""
    tokcache.set_active(None)
    yield
    tokcache.set_active(None)


EXAMPLES = [
    DistillExample(
        prompt_with_si="System: always be brief.\nUser: hello there",
        prompt_without="User: hello there",
        completion="Hi.",
    ),
    DistillExample(
        prompt_with_si="System: always be brief.\nUser: naïve café — 12345?",
        prompt_without="User: naïve café — 12345?",
        completion="A short answer with several words in it.",
    ),
]


def test_cached_ids_matches_tokenizer_and_memoizes(tiny_llama):
    _, tok = tiny_llama
    cache = TokCache()
    tokcache.set_active(cache)
    text = EXAMPLES[0].prompt_with_si
    direct = tok(text, add_special_tokens=True)["input_ids"]
    first = cached_ids(tok, text, True)
    second = cached_ids(tok, text, True)
    assert first == direct == second
    assert cache.hits == 1 and cache.misses == 1
    # add_special_tokens participates in the key — no cross-contamination.
    bare = cached_ids(tok, text, False)
    assert bare == tok(text, add_special_tokens=False)["input_ids"]
    # A hit returns a COPY: mutating the result must not poison the store.
    first.append(0)
    assert cached_ids(tok, text, True) == direct


def test_encode_aligned_bitwise_parity_warm_vs_cold(tiny_llama):
    _, tok = tiny_llama
    cold = [_encode_aligned(tok, ex, max_len=64) for ex in EXAMPLES]
    tokcache.set_active(TokCache())
    warm_once = [_encode_aligned(tok, ex, max_len=64) for ex in EXAMPLES]
    warm_twice = [_encode_aligned(tok, ex, max_len=64) for ex in EXAMPLES]
    for (ct, cs), (wt, ws) in zip(cold, warm_once + warm_twice[:0] or warm_once):
        assert ct.ids == wt.ids and ct.prompt_len == wt.prompt_len
        assert cs.ids == ws.ids and cs.prompt_len == ws.prompt_len
    for (ct, cs), (wt, ws) in zip(cold, warm_twice):
        assert ct.ids == wt.ids and cs.ids == ws.ids


def test_encode_prompt_completion_parity(tiny_llama):
    _, tok = tiny_llama
    cold = encode_prompt_completion(tok, "User: hi", "Hello!", max_len=32)
    tokcache.set_active(TokCache())
    warm = encode_prompt_completion(tok, "User: hi", "Hello!", max_len=32)
    assert cold.ids == warm.ids and cold.prompt_len == warm.prompt_len


def test_sidecar_round_trip_no_pickle(tiny_llama, tmp_path):
    _, tok = tiny_llama
    cache = TokCache()
    tokcache.set_active(cache)
    for ex in EXAMPLES:
        cached_ids(tok, ex.prompt_with_si, True)
        cached_ids(tok, ex.prompt_without, True)
        cached_ids(tok, ex.completion, False)
    path = tmp_path / "pool.tokcache.npz"
    header = save_sidecar(path, cache, tok, model_id="test/model",
                          model_revision="rev0", extra={"si_id": "s"})
    assert header["n_entries"] == len(cache)
    # allow_pickle=False is load-bearing: the sidecar must be plain arrays.
    with np.load(path, allow_pickle=False) as z:
        assert set(z.files) >= {"meta", "hashes", "flags", "offsets", "ids"}
    warmed = warm_from_sidecar(path, tok, expect_model_id="test/model")
    tokcache.set_active(warmed)
    before = warmed.misses
    for ex in EXAMPLES:
        direct = tok(ex.prompt_with_si, add_special_tokens=True)["input_ids"]
        assert cached_ids(tok, ex.prompt_with_si, True) == direct
    assert warmed.misses == before, "warmed sidecar should serve pure hits"


def test_sidecar_mismatch_refuses_loudly(tiny_llama, tmp_path):
    _, tok = tiny_llama
    cache = TokCache()
    tokcache.set_active(cache)
    cached_ids(tok, "some pooled text", True)
    path = tmp_path / "pool.tokcache.npz"
    save_sidecar(path, cache, tok, model_id="test/model", model_revision=None)

    # Wrong model id: refused, naming the field.
    with pytest.raises(TokCacheMismatch, match="model_id"):
        warm_from_sidecar(path, tok, expect_model_id="other/model")

    # Tokenizer drift (tampered probe ids): refused via the direct probe.
    with np.load(path, allow_pickle=False) as z:
        arrs = {k: z[k] for k in z.files}
    header = json.loads(str(arrs["meta"][0]))
    header["probe_ids"] = [1, 2, 3]
    arrs["meta"] = np.array([json.dumps(header)])
    np.savez(path, **arrs)
    with pytest.raises(TokCacheMismatch, match="probe"):
        warm_from_sidecar(path, tok, expect_model_id="test/model")


def test_probe_text_is_tokenizer_exercising():
    # The probe must carry the things normalizers drift on; this pins the
    # string so a future edit that weakens it fails a test, not a wave.
    assert "12345" in PROBE_TEXT and "naïve" in PROBE_TEXT


def test_sidecar_refuses_when_the_POOL_changed(tiny_llama, tmp_path):
    """The tokenizer guards cannot see a pool whose ROWS changed under an
    unchanged tokenizer: model_id, template_hash and the probe all still
    match, so the sidecar validates while being a wrong-world cache. That is
    the durable shape of the bug -- the guard's PASS reassures. Pool-content
    keying is what closes it."""
    from sisteer.extraction import tokcache

    tok = tiny_llama[1] if isinstance(tiny_llama, tuple) else tiny_llama
    pool = tmp_path / "pairs.jsonl"
    pool.write_text('{"a": 1}\n')
    cache = tokcache.TokCache()
    tokcache.set_active(cache)
    tokcache.cached_ids(tok, "hello world", True)
    side = tmp_path / "pairs.jsonl.tokcache.npz"
    tokcache.save_sidecar(side, cache, tok, model_id="m", model_revision=None,
                          pool_path=pool)

    # unchanged pool: warms
    got = tokcache.warm_from_sidecar(side, tok, expect_model_id="m",
                                     expect_pool_path=pool)
    assert len(got) == len(cache)

    # SAME tokenizer, DIFFERENT pool rows -> must refuse
    pool.write_text('{"a": 2}\n')
    with pytest.raises(tokcache.TokCacheMismatch, match="pool_sha256"):
        tokcache.warm_from_sidecar(side, tok, expect_model_id="m",
                                   expect_pool_path=pool)


class _ModelBTok:
    """A stand-in for a DIFFERENT model's tokenizer (T7 cross-model case,
    no second download): same interface, but the probe string encodes to
    shifted ids — exactly what a real cross-model warm looks like to the
    direct-probe guard. chat_template is copied so the template_hash check
    can be exercised separately by mutating it."""

    def __init__(self, tok):
        self._tok = tok
        self.chat_template = getattr(tok, "chat_template", None)

    def __call__(self, text, **kw):
        out = self._tok(text, **kw)
        if text == PROBE_TEXT:
            return {"input_ids": [i + 1 for i in out["input_ids"]]}
        return out


def test_sidecar_cross_model_refuses_each_named_field(tiny_llama, tmp_path):
    """T7 cross-model invalidation: a sidecar built for model A must
    refuse to warm for model B, and each of the four guards refuses
    NAMING ITS OWN FIELD — model_id, template_hash, the direct probe, and
    the pool content hash. (The two-model sweep ships one sidecar per
    (model, pool); the failure this pins is warming model A's sidecar
    into model B's fit.)"""
    _, tok = tiny_llama
    pool = tmp_path / "pairs_a.jsonl"
    pool.write_text('{"prompt": "synthetic row"}\n')
    cache = TokCache()
    tokcache.set_active(cache)
    cached_ids(tok, "shared pooled text", True)
    side = tmp_path / "pairs_a.jsonl.tokcache.npz"
    save_sidecar(side, cache, tok, model_id="org/model-a",
                 model_revision="aaaa", pool_path=pool)

    # 1. Different declared model id -> refused on model_id (first guard).
    with pytest.raises(TokCacheMismatch, match="model_id"):
        warm_from_sidecar(side, tok, expect_model_id="org/model-b",
                          expect_pool_path=pool)

    # 2. Same declared id, model B's TEMPLATE -> refused on template_hash.
    b_tok = _ModelBTok(tok)
    b_tok.chat_template = "{{ a-completely-different-template }}"
    with pytest.raises(TokCacheMismatch, match="template_hash"):
        warm_from_sidecar(side, b_tok, expect_model_id="org/model-a",
                          expect_pool_path=pool)

    # 3. Same id AND template, model B's VOCAB -> the direct probe catches
    #    the drifted encoding (the guard that needs no bookkeeping at all).
    b_tok = _ModelBTok(tok)
    with pytest.raises(TokCacheMismatch, match="probe"):
        warm_from_sidecar(side, b_tok, expect_model_id="org/model-a",
                          expect_pool_path=pool)

    # 4. Same tokenizer entirely, model B's POOL -> refused on pool_sha256.
    pool_b = tmp_path / "pairs_b.jsonl"
    pool_b.write_text('{"prompt": "a different synthetic row"}\n')
    with pytest.raises(TokCacheMismatch, match="pool_sha256"):
        warm_from_sidecar(side, tok, expect_model_id="org/model-a",
                          expect_pool_path=pool_b)

    # Control: the matching world still warms.
    got = warm_from_sidecar(side, tok, expect_model_id="org/model-a",
                            expect_pool_path=pool)
    assert len(got) == len(cache)


def test_sidecar_without_a_pool_hash_still_warms_but_says_so(tiny_llama,
                                                             tmp_path, capsys):
    """Sidecars built before pool keying must not fail a run that was fine
    yesterday -- but unverifiable is a different state from verified, and
    the log has to distinguish them."""
    from sisteer.extraction import tokcache

    tok = tiny_llama[1] if isinstance(tiny_llama, tuple) else tiny_llama
    pool = tmp_path / "pairs.jsonl"
    pool.write_text("{}\n")
    cache = tokcache.TokCache()
    tokcache.set_active(cache)
    tokcache.cached_ids(tok, "x", True)
    side = tmp_path / "s.npz"
    tokcache.save_sidecar(side, cache, tok, model_id="m", model_revision=None)

    tokcache.warm_from_sidecar(side, tok, expect_model_id="m",
                               expect_pool_path=pool)
    assert "WITHOUT verifying" in capsys.readouterr().err
