"""sample_manifest — the sample manifest staged beside a document pool.

ONE resolver and validator for every reader — the trainer, which records
the manifest's sha as ``meta_extra.sample_manifest_sha`` in
``train_budget_meta.json``, and the Slurm preflight
(``scripts/sdfbase/slurm_fit_cell.py``), which refuses a cell before the
trainer runs — so the two can never disagree about WHICH file the manifest
is or whether it is this pool's.

The pool builder (``scripts/sdfbase/build_pools.py``) writes
``pairs_<cell>.jsonl`` and beside it ``pairs_<cell>.manifest.json`` — the
stem-swapped spelling — carrying the cell id and the pool file's byte
sha256 among its provenance. An older appended spelling,
``<pool>.jsonl.manifest.json``, is also read, and wins when both exist.

Resolution takes the FIRST spelling that EXISTS and stops there. An
existing preferred candidate that is empty, unparseable, not an object,
names another cell, or records another pool hash is an error, never a
reason to try the other spelling. Before 2026-09-10 the trainer hashed the
first existing file unread (an empty one included) while the lane's
preflight skipped empty files and accepted the other spelling — one fit,
two different "manifests" — and neither side checked that the manifest was
the pool's, so a stale manifest beside a replaced pool passed and was
recorded as the fit's sample provenance.

The recorded value is unchanged: the manifest FILE's sha256, so every
published sidecar keeps validating (verified 2026-09-10 against the
full-scale lanes' recorded ``sample_manifest_sha``, which equals the
stem-swapped manifest's file hash — the only spelling those pools ship).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

MANIFEST_SUFFIX = ".manifest.json"


class SampleManifestError(ValueError):
    """An existing sample manifest that is not this fit's: empty,
    unparseable, another cell's, or another pool's."""


@dataclass(frozen=True)
class SampleManifest:
    #: the file that resolved
    path: Path
    #: sha256 of the manifest FILE's bytes — ``meta_extra.sample_manifest_sha``
    sha256: str
    #: the manifest's ``cell`` (== the SI id it was resolved for)
    cell: str
    #: the manifest's ``pool_sha256`` (== the pool file's byte hash)
    pool_sha256: str


def sha256_file(path) -> str:
    """Streaming sha256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_candidates(pairs_path) -> tuple[Path, Path]:
    """The two spellings, in resolution order: the appended
    ``<pool>.jsonl.manifest.json``, then the stem-swapped
    ``pairs_<cell>.manifest.json`` the pool builder ships."""
    p = Path(str(pairs_path))
    return (Path(str(p) + MANIFEST_SUFFIX), p.with_suffix(MANIFEST_SUFFIX))


def resolve_sample_manifest(pairs_path, *, cell: str,
                            pool_sha256: str | None = None
                            ) -> SampleManifest | None:
    """The sample manifest beside ``pairs_path``, parsed and validated for
    ``cell`` (the SI id the pool is being fit for) and for the pool's bytes
    (``pool_sha256`` when the caller already derived it, else hashed here).

    Returns None only when NEITHER spelling exists — a stated absence for
    the caller to say out loud (synthetic test pools; production pools
    always ship one). The first spelling that EXISTS is the manifest; if it
    is not a valid manifest of this cell over this pool, that is a
    :class:`SampleManifestError` — the other spelling is never consulted.
    """
    cands = manifest_candidates(pairs_path)
    chosen = next((c for c in cands if c.exists()), None)
    if chosen is None:
        return None
    others = ", ".join(str(c) for c in cands if c != chosen)
    if not chosen.is_file():
        raise SampleManifestError(f"{chosen} exists but is not a regular file")
    raw = chosen.read_bytes()
    if not raw.strip():
        raise SampleManifestError(
            f"{chosen} is empty — an existing manifest that carries nothing "
            f"is refused, never skipped for the other spelling ({others})")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise SampleManifestError(f"{chosen} does not parse as JSON: {e}") \
            from e
    if not isinstance(doc, dict):
        raise SampleManifestError(
            f"{chosen} is a JSON {type(doc).__name__}, not a manifest object")
    got_cell = doc.get("cell")
    if not isinstance(got_cell, str) or not got_cell:
        raise SampleManifestError(f"{chosen} records no cell")
    if got_cell != cell:
        raise SampleManifestError(
            f"{chosen} is cell {got_cell!r}'s manifest, not {cell!r}'s — "
            "not this pool's manifest")
    got_pool = doc.get("pool_sha256")
    if not isinstance(got_pool, str) or not got_pool:
        raise SampleManifestError(
            f"{chosen} records no pool_sha256 — cannot be placed on a pool")
    actual = pool_sha256 or sha256_file(pairs_path)
    if got_pool != actual:
        raise SampleManifestError(
            f"{chosen} records pool_sha256 {got_pool[:12]}… but {pairs_path} "
            f"hashes to {actual[:12]}… — a stale manifest beside a replaced "
            "pool (or the wrong pool); refusing to record it as this fit's "
            "sample provenance")
    return SampleManifest(path=chosen, sha256=hashlib.sha256(raw).hexdigest(),
                          cell=got_cell, pool_sha256=got_pool)
