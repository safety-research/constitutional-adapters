"""Canonical fit-input fingerprint — the identity a resume/skip MUST match.

A trained cell's COMPLETE marker (scripts/train_two_si_objects.py) and an HPO
trial's summary.json (scripts/tune_arm.py) each record this fingerprint, and a
resume/skip is honoured ONLY when the fingerprint recomputed from the REQUESTED
work equals the stored one. That makes resume *identity-bound* rather than
name/existence-bound: it stops a cell or trial trained for one
(model, revision, SI, arm, position leg, pool, optimizer/config) from being
silently reused as another.

Motivated by the 2026-08-25 adversarial review: an all-token marker was accepted
as the response-only cell (F15), a Gemma sweep could be recorded as Qwen (F2),
and a stale summary.json from a different search box was reused (F4). One
function, one required key set, so no caller can quietly drop a dimension — a
missing OR extra key raises rather than hashing a partial identity.
"""
from __future__ import annotations

import hashlib
import json

#: Every dimension that defines a fit's identity. Enforced exactly: a caller
#: that omits one (the F15 failure mode) or adds an unrecognised one raises.
#: `optimizer` is itself the resolved-settings mapping
#: {lr, lr_schedule, warmup_frac, max_grad_norm[, accum_steps]}; `pool_sha256`
#: is the content hash of the pairs file the fit consumed; `config_sha` is the
#: hash of the resolved per-model config.
FINGERPRINT_KEYS = (
    "model",
    "model_revision",
    "si_id",
    "arm",
    "positions",
    "pool_sha256",
    "optimizer",
    "config_sha",
)


def fit_fingerprint(fields: dict) -> str:
    """Return the sha256 hex of the canonical identity built from ``fields``.

    ``fields`` must carry exactly ``FINGERPRINT_KEYS`` — no more, no fewer.
    Serialisation is order-independent and stable (sorted keys, compact
    separators, ``default=str`` so a revision/None/number all canonicalise),
    so the same identity always hashes the same way at write time and at
    skip-check time.
    """
    missing = [k for k in FINGERPRINT_KEYS if k not in fields]
    if missing:
        raise ValueError(
            f"fit_fingerprint missing required keys: {missing} "
            f"(a partial identity must never be hashed)")
    extra = [k for k in fields if k not in FINGERPRINT_KEYS]
    if extra:
        raise ValueError(
            f"fit_fingerprint got unexpected keys: {extra} "
            f"(add it to FINGERPRINT_KEYS deliberately, do not pass it ad hoc)")
    canonical = {k: fields[k] for k in FINGERPRINT_KEYS}
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def fingerprints_match(stored: str | None, expected: str) -> bool:
    """True only when a fingerprint was stored AND equals the expected one.

    A ``None`` stored value (a pre-fingerprint marker, or one that never
    recorded it) is treated as NON-matching on purpose: an un-fingerprinted
    artifact cannot be proven to be the requested work, so it must re-run
    rather than be trusted.
    """
    return stored is not None and stored == expected
