"""Tests for the shared fit-input fingerprint (identity-bound resume/skip)."""
import pytest

from sisteer.extraction.fit_fingerprint import (
    FINGERPRINT_KEYS, fingerprints_match, fit_fingerprint)


def _full(**over):
    base = {
        "model": "google/gemma-4-E4B-it",
        "model_revision": "ee0ef602",
        "si_id": "si_constitution",
        "arm": "psr_kl_all",
        "positions": "all",
        "pool_sha256": "abc123",
        "optimizer": {"lr": 5e-4, "lr_schedule": "warmup_cosine",
                      "warmup_frac": 0.15, "max_grad_norm": 1.0},
        "config_sha": "cfg0",
    }
    base.update(over)
    return base


def test_stable_and_order_independent():
    a = fit_fingerprint(_full())
    # a dict built in a different key order hashes identically
    b = fit_fingerprint(dict(reversed(list(_full().items()))))
    assert a == b == fit_fingerprint(_full())
    assert len(a) == 64


@pytest.mark.parametrize("key,newval", [
    ("model", "Qwen/Qwen3.8-27B"),        # F2: one model recorded as another
    ("model_revision", "deadbeef"),
    ("si_id", "safety_refusal"),
    ("arm", "lora_r1_all"),
    ("positions", "response"),            # F15: all-token vs response leg
    ("pool_sha256", "zzz999"),            # F4: different pool
    ("config_sha", "cfg1"),
])
def test_every_dimension_changes_the_hash(key, newval):
    assert fit_fingerprint(_full()) != fit_fingerprint(_full(**{key: newval}))


def test_optimizer_change_changes_the_hash():
    other = dict(_full()["optimizer"], lr=1e-3)
    assert fit_fingerprint(_full()) != fit_fingerprint(_full(optimizer=other))
    # bidpo's accum_steps is part of identity when present
    with_accum = dict(_full()["optimizer"], accum_steps=16)
    assert fit_fingerprint(_full()) != fit_fingerprint(_full(optimizer=with_accum))


def test_missing_key_raises_never_hashes_partial():
    partial = _full()
    del partial["positions"]
    with pytest.raises(ValueError, match="missing required keys"):
        fit_fingerprint(partial)


def test_extra_key_raises():
    with pytest.raises(ValueError, match="unexpected keys"):
        fit_fingerprint(_full(seed=0))


def test_match_semantics_none_is_never_a_match():
    fp = fit_fingerprint(_full())
    assert fingerprints_match(fp, fp)
    assert not fingerprints_match(None, fp)        # un-fingerprinted → re-run
    assert not fingerprints_match("stale", fp)


def test_keys_are_the_documented_eight():
    assert set(FINGERPRINT_KEYS) == {
        "model", "model_revision", "si_id", "arm", "positions",
        "pool_sha256", "optimizer", "config_sha"}
