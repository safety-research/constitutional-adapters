"""String-level token memo cache, warmable from a precomputed sidecar file.

Why this exists (stage-3 wave autopsy, 2026-08-22): every arm process
re-tokenized the full 23k-pair pool before its first optimizer step —
30-60 min of 0% GPU per arm start on slow hosts, times 13 arms, re-paid
on every restart, plus a full-pool re-tokenization inside every
preflight_max_len call. All of those reduce to raw
``tok(text, add_special_tokens=...)`` calls over a fixed set of strings,
so the fix is a memo keyed on (sha256(text), add_special_tokens) that a
grading-host-side precompute can fill once and ship next to the pool file.

Design constraints, in order:

- **Bitwise-identical fits.** The cache stores exactly what the tokenizer
  returned; a hit returns a copy of those ids. Cold-vs-warm runs produce
  identical EncodedSeqs, so the optimization trajectory is untouched.
- **Misses are legal, silence is not.** Arm-injected prompt variants
  (``obj.inject`` on token arms) are unknown to a pool-level sidecar; they
  simply miss and tokenize inline (then memo in-process). But a sidecar
  that MISMATCHES the tokenizer refuses loudly — a wrong-world cache next
  to the pool is exactly the stale-artifact class of bug (the stale-tarball
  lesson), and falling back silently would hide it.
- **No pickle.** The sidecar is a plain .npz of string/int arrays, loaded
  with allow_pickle=False.

Validation is by DIRECT PROBE, not bookkeeping: the sidecar records the
ids of a fixed probe string, and warming re-tokenizes the probe with the
live tokenizer and compares — catching vocab/normalizer drift regardless
of how model revisions were resolved or recorded. model_id and
template_hash are checked too, so mismatch messages can name what moved.
"""

from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

SCHEMA = 1
# Mixed scripts, digits, punctuation and a leading space — the things
# normalizers and BPE merges most plausibly drift on.
PROBE_TEXT = " tokcache probe 12345 — The quick brown fox; naïve café.\n"


class TokCacheMismatch(RuntimeError):
    """A sidecar that does not match the live tokenizer. Never fall back
    silently — the caller either refuses the run or was told (via
    --token-cache off) not to use a cache at all."""


def _key(text: str, add_special_tokens: bool) -> tuple[str, bool]:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (h, bool(add_special_tokens))


class TokCache:
    __slots__ = ("_store", "hits", "misses")

    def __init__(self) -> None:
        self._store: dict[tuple[str, bool], list[int]] = {}
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._store)

    def lookup(self, text: str, add_special_tokens: bool) -> list[int] | None:
        got = self._store.get(_key(text, add_special_tokens))
        if got is None:
            self.misses += 1
            return None
        self.hits += 1
        return got

    def put(self, text: str, add_special_tokens: bool, ids: list[int]) -> None:
        self._store[_key(text, add_special_tokens)] = list(ids)


_ACTIVE: TokCache | None = None


def set_active(cache: TokCache | None) -> None:
    global _ACTIVE
    _ACTIVE = cache


def active() -> TokCache | None:
    return _ACTIVE


def cached_ids(tok, text: str, add_special_tokens: bool) -> list[int]:
    """``tok(text, add_special_tokens=...)["input_ids"]`` through the ambient
    cache. No active cache -> plain tokenization (byte-identical legacy
    path). Hit -> a fresh copy of the stored ids (callers may slice/concat).
    Miss -> tokenize, memo in-process, return."""
    cache = _ACTIVE
    if cache is None:
        return tok(text, add_special_tokens=add_special_tokens)["input_ids"]
    got = cache.lookup(text, add_special_tokens)
    if got is not None:
        # ndarray view (warmed sidecar) -> tolist() gives Python ints in C;
        # plain list (in-process memo) -> copy.
        return got.tolist() if hasattr(got, "tolist") else list(got)
    ids = tok(text, add_special_tokens=add_special_tokens)["input_ids"]
    cache.put(text, add_special_tokens, ids)
    return list(ids)


def template_hash_of(tok) -> str:
    """sha256 16-hex prefix of the raw chat-template string — the exact
    convention train_budget_meta.json records, so sidecar and sidecar-meta
    hashes compare directly."""
    raw = (getattr(tok, "chat_template", None) or "").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def pool_sha256(pool_path) -> tuple[str, int]:
    """``(sha256, n_bytes)`` of the pool a sidecar was built from.

    The tokenizer guards below cannot catch a pool whose ROWS changed under
    an UNCHANGED tokenizer: model_id, template_hash and the probe all still
    match, so the sidecar validates while being a wrong-world cache. That is
    the most durable shape of bug, because the guard's PASS actively
    reassures. Keying on pool content closes it.
    """
    h = hashlib.sha256()
    n = 0
    with open(pool_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


def save_sidecar(path, cache: TokCache, tok, model_id: str,
                 model_revision: str | None, extra: dict | None = None,
                 pool_path=None) -> dict:
    """Write ``cache`` to ``path`` (.npz, no pickle) with a validation
    header. Returns the header."""
    probe = tok(PROBE_TEXT, add_special_tokens=True)["input_ids"]
    header = {
        "schema": SCHEMA,
        "model_id": model_id,
        "model_revision": model_revision,
        "template_hash": template_hash_of(tok),
        "probe_ids": [int(i) for i in probe],
        "n_entries": len(cache),
        **(dict(zip(("pool_sha256", "pool_bytes"), pool_sha256(pool_path)))
           if pool_path is not None else {}),
        **(extra or {}),
    }
    hashes, flags, offsets, flat = [], [], [0], []
    for (h, ast), ids in cache._store.items():
        hashes.append(h)
        flags.append(1 if ast else 0)
        flat.extend(ids)
        offsets.append(len(flat))
    # Compressed on purpose: a long-SI pool's teacher ids embed the SI
    # text per pair (si_constitution: ~5.9k tokens x 23k pairs ~= 600 MB
    # raw); token ids zlib-compress ~3x and decompress in seconds at warm
    # time, keeping per-pod shipping in the ~200 MB class.
    np.savez_compressed(
        path,
        meta=np.array([json.dumps(header)]),
        hashes=np.array(hashes, dtype="<U64"),
        flags=np.array(flags, dtype=np.uint8),
        offsets=np.array(offsets, dtype=np.int64),
        ids=np.array(flat, dtype=np.int32),
    )
    return header


def warm_from_sidecar(path, tok, expect_model_id: str,
                      expect_pool_path=None) -> TokCache:
    """Load a sidecar and validate it against the LIVE tokenizer. Raises
    TokCacheMismatch naming the first field that moved."""
    with np.load(path, allow_pickle=False) as z:
        header = json.loads(str(z["meta"][0]))
        hashes = z["hashes"]
        flags = z["flags"]
        offsets = z["offsets"]
        flat = z["ids"]

    def refuse(field, want, got):
        raise TokCacheMismatch(
            f"token sidecar {path} does not match this run: {field} is "
            f"{got!r} but the sidecar was built for {want!r}. Rebuild it "
            f"(precompute-tokens) or pass --token-cache off.")

    if header.get("schema") != SCHEMA:
        refuse("schema", header.get("schema"), SCHEMA)
    if header.get("model_id") != expect_model_id:
        refuse("model_id", header.get("model_id"), expect_model_id)
    live_th = template_hash_of(tok)
    if header.get("template_hash") != live_th:
        refuse("template_hash", header.get("template_hash"), live_th)
    live_probe = [int(i) for i in
                  tok(PROBE_TEXT, add_special_tokens=True)["input_ids"]]
    if header.get("probe_ids") != live_probe:
        refuse("probe_ids (tokenizer drift)", header.get("probe_ids"),
               live_probe)
    # Pool content. Only checkable when the sidecar RECORDED a hash and the
    # caller supplies the pool -- older sidecars predate the field, and a
    # missing hash must not fail a run that was fine yesterday. But say so:
    # an unverifiable cache is a different state from a verified one.
    want_pool = header.get("pool_sha256")
    if want_pool and expect_pool_path is not None:
        got_pool, _n = pool_sha256(expect_pool_path)
        if got_pool != want_pool:
            refuse("pool_sha256 (the POOL changed, not the tokenizer)",
                   want_pool, got_pool)
    elif expect_pool_path is not None and not want_pool:
        print(f"[tokcache] {path} predates pool-content keying -- warmed "
              f"WITHOUT verifying it matches {expect_pool_path}",
              file=sys.stderr)

    cache = TokCache()
    # Zero-copy views into the flat array, NOT .tolist(): a long-SI sidecar
    # holds ~1e8 ids, and Python-int lists cost ~28 bytes/int — gigabytes of
    # RAM and minutes of conversion for data that mostly gets sliced and
    # concatenated. cached_ids converts a view to a real list only on hit.
    for i, (h, f) in enumerate(zip(hashes, flags)):
        cache._store[(str(h), bool(f))] = flat[offsets[i]:offsets[i + 1]]
    return cache
