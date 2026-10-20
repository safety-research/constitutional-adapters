"""sisteer.data.sample_manifest — the ONE resolver + validator for the sample
manifest staged beside a document pool (2026-09-10). Pure: no model, no
trainer. The trainer's ``_sample_manifest_sha_for`` and the Slurm runner's
preflight are pinned to it in tests/test_ce_arms.py and
tests/test_slurm_fit_cell.py; this file pins the resolver itself."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sisteer.data.sample_manifest import (
    SampleManifest, SampleManifestError, manifest_candidates,
    resolve_sample_manifest, sha256_file)

CELL = "constv2-mix-1k"


def _pool(tmp_path: Path, name: str = f"pairs_{CELL}.jsonl") -> Path:
    p = tmp_path / name
    p.write_text('{"prompt": "", "pos": "one", "neg": "", "instruction_id": "x"}\n'
                 '{"prompt": "", "pos": "two", "neg": "", "instruction_id": "x"}\n')
    return p


def _doc(pool: Path, cell: str = CELL, **over) -> dict:
    d = {"cell": cell, "budget": 1000, "n_docs": 2,
         "pool_sha256": sha256_file(pool)}
    d.update(over)
    return d


def _stem(pool: Path) -> Path:
    return pool.with_suffix(".manifest.json")


def _appended(pool: Path) -> Path:
    return Path(str(pool) + ".manifest.json")


def test_candidates_order_is_appended_then_stem(tmp_path):
    pool = tmp_path / "pool" / f"pairs_{CELL}.jsonl"
    assert manifest_candidates(pool) == (
        tmp_path / "pool" / f"pairs_{CELL}.jsonl.manifest.json",
        tmp_path / "pool" / f"pairs_{CELL}.manifest.json")
    assert manifest_candidates(str(pool)) == manifest_candidates(pool)


def test_stem_spelling_resolves_and_the_sha_is_the_files_bytes(tmp_path):
    pool = _pool(tmp_path)
    m = _stem(pool)
    m.write_text(json.dumps(_doc(pool)))
    got = resolve_sample_manifest(pool, cell=CELL)
    assert isinstance(got, SampleManifest)
    assert got.path == m
    assert got.sha256 == hashlib.sha256(m.read_bytes()).hexdigest()
    assert got.cell == CELL and got.pool_sha256 == sha256_file(pool)
    # the caller's pool hash short-circuits the file read
    assert resolve_sample_manifest(pool, cell=CELL,
                                   pool_sha256=sha256_file(pool)) == got


def test_appended_spelling_resolves_and_wins_when_both_exist(tmp_path):
    pool = _pool(tmp_path)
    _stem(pool).write_text(json.dumps(_doc(pool)))
    _appended(pool).write_text(json.dumps(_doc(pool, tag="appended")))
    got = resolve_sample_manifest(pool, cell=CELL)
    assert got.path == _appended(pool)
    assert got.sha256 == hashlib.sha256(_appended(pool).read_bytes()).hexdigest()


def test_neither_spelling_is_a_stated_absence(tmp_path):
    pool = _pool(tmp_path)
    assert resolve_sample_manifest(pool, cell=CELL) is None


@pytest.mark.parametrize("text, match", [
    ("", "is empty"),
    ("  \n", "is empty"),
    ("{not json", "does not parse"),
    ("[1, 2]", "not a manifest object"),
    (json.dumps({"budget": 1}), "records no cell"),
    (json.dumps({"cell": "", "pool_sha256": "a" * 64}), "records no cell"),
])
def test_an_invalid_preferred_candidate_is_refused_not_skipped(tmp_path, text,
                                                              match):
    """The appended (preferred) spelling exists but is invalid; a VALID stem
    manifest sits beside it. The resolver refuses — it never falls through
    to the other spelling (the preflight/trainer divergence of 2026-09-10:
    one skipped the empty file, the other hashed it)."""
    pool = _pool(tmp_path)
    _stem(pool).write_text(json.dumps(_doc(pool)))
    _appended(pool).write_text(text)
    with pytest.raises(SampleManifestError, match=match) as ei:
        resolve_sample_manifest(pool, cell=CELL)
    assert str(_appended(pool)) in str(ei.value)


def test_another_cells_manifest_is_refused(tmp_path):
    pool = _pool(tmp_path)
    _stem(pool).write_text(json.dumps(_doc(pool, cell="constv2-mix-100k")))
    with pytest.raises(SampleManifestError,
                       match="is cell 'constv2-mix-100k'.*not 'constv2-mix-1k'"):
        resolve_sample_manifest(pool, cell=CELL)


def test_a_manifest_without_the_pool_hash_is_refused(tmp_path):
    pool = _pool(tmp_path)
    _stem(pool).write_text(json.dumps({"cell": CELL, "budget": 1000}))
    with pytest.raises(SampleManifestError, match="records no pool_sha256"):
        resolve_sample_manifest(pool, cell=CELL)


def test_a_stale_manifest_beside_a_replaced_pool_is_refused(tmp_path):
    """The manifest was written for the pool's OLD bytes; the pool has since
    been replaced. Refused whether the caller hands in the pool hash or the
    resolver hashes the file itself."""
    pool = _pool(tmp_path)
    _stem(pool).write_text(json.dumps(_doc(pool)))
    with pool.open("a") as f:
        f.write('{"prompt": "", "pos": "three", "neg": "", "instruction_id": "x"}\n')
    with pytest.raises(SampleManifestError, match="stale manifest beside a "
                                                  "replaced pool"):
        resolve_sample_manifest(pool, cell=CELL)
    with pytest.raises(SampleManifestError, match="stale manifest"):
        resolve_sample_manifest(pool, cell=CELL, pool_sha256=sha256_file(pool))
    # re-registered for the new bytes, it resolves
    _stem(pool).write_text(json.dumps(_doc(pool)))
    assert resolve_sample_manifest(pool, cell=CELL).pool_sha256 == \
        sha256_file(pool)


def test_the_resolver_hashes_the_pool_only_when_not_told(tmp_path):
    """With ``pool_sha256`` supplied the pool file is never opened — the
    trainer hands in the byte hash its identity derivation already made."""
    pool = tmp_path / f"pairs_{CELL}.jsonl"           # does not exist
    _stem(pool).write_text(json.dumps({"cell": CELL, "pool_sha256": "b" * 64}))
    got = resolve_sample_manifest(pool, cell=CELL, pool_sha256="b" * 64)
    assert got is not None and got.pool_sha256 == "b" * 64
    with pytest.raises(FileNotFoundError):
        resolve_sample_manifest(pool, cell=CELL)
