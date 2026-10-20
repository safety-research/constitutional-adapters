"""Onboarded rows: the checkpoint inventory built from the Hub's own per-file metadata and held to
the bytes on disk — LFS files by sha256, plain git files by the git BLOB SHA-1 (``sha1(b"blob <size>\\0" + bytes)``, which is what
the Hub's ``blob_id`` is; a sha256 of the contents would fail a correct snapshot by construction), sizes for every file. Refusals
are by name: a mutated shard, a missing shard, a wrong-size plain file, a listing without digests. Fixtures are synthetic; no Hub call.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from sisteer.capeval.tinycap import hf_inventory as HI


def _blob(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _snapshot(tmp_path: Path):
    """A synthetic HF-cache snapshot: two 'LFS' shards + three plain files, with the Hub listing that describes them."""
    hf_home = tmp_path / "hf"
    snap = HI.snapshot_dir(hf_home, "Org/Model", "0123456789abcdef0123456789abcdef01234567")
    snap.mkdir(parents=True)
    files = {"model-00001-of-00002.safetensors": b"\x00" * 4096 + b"shard-one",
             "model-00002-of-00002.safetensors": b"\x01" * 2048 + b"shard-two",
             "config.json": json.dumps({"model_type": "qwen3_5"}).encode(),
             "generation_config.json": b'{"temperature": 1.0}\n',
             "chat_template.jinja": b"{{ messages }}\n"}
    sibs = []
    for name, data in files.items():
        (snap / name).write_bytes(data)
        if name.endswith(".safetensors"):
            sibs.append({"rfilename": name, "size": len(data), "lfs": {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}, "blob_id": "deadbeef" * 5})
        else:
            sibs.append({"rfilename": name, "size": len(data), "lfs": None, "blob_id": _blob(data)})
    inv = HI.inventory_from_siblings("Org/Model", "0123456789abcdef0123456789abcdef01234567", "0123456789abcdef0123456789abcdef01234567", sibs, read_utc="2026-10-03T00:00:00Z")
    return hf_home, snap, inv, files


def test_git_blob_sha1_is_gits_own_object_id(tmp_path):
    p = tmp_path / "f.txt"
    p.write_bytes(b"hello blob\n")
    assert HI.git_blob_sha1(p) == _blob(b"hello blob\n")
    # pinned against git itself when available (the one authoritative spelling of the function)
    try:
        want = subprocess.run(["git", "hash-object", str(p)], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    assert HI.git_blob_sha1(p) == want
    # and it is NOT the sha256 of the contents
    assert HI.git_blob_sha1(p) != hashlib.sha256(b"hello blob\n").hexdigest()[:40]


def test_the_inventory_names_every_file_with_the_kind_the_hub_states_and_verifies_a_clean_snapshot(tmp_path):
    hf_home, snap, inv, files = _snapshot(tmp_path)
    assert HI.inventory_problems(inv) == [] and inv["schema"] == HI.INVENTORY_SCHEMA and inv["n_files"] == 5
    assert inv["files"]["model-00001-of-00002.safetensors"]["kind"] == HI.KIND_LFS
    assert inv["files"]["config.json"]["kind"] == HI.KIND_BLOB and len(inv["files"]["config.json"]["digest"]) == 40
    assert inv["total_bytes"] == sum(len(d) for d in files.values())
    res = HI.verify_snapshot(snap, inv)
    assert res["ok"] and res["problems"] == [] and sorted(res["verified"]) == sorted(files) and res["extra"] == []
    assert res["bytes_hashed"] == inv["total_bytes"] and res["hash_bytes"] is True
    # the cheap re-check (sizes only) passes too and hashes nothing
    cheap = HI.verify_snapshot(snap, inv, hash_bytes=False)
    assert cheap["ok"] and cheap["bytes_hashed"] == 0 and cheap["hash_bytes"] is False
    # round trip through the one serializer
    path = HI.inventory_path(HI.repo_cache_dir(hf_home, "Org/Model"), inv["revision"])
    assert path.name == ".sisteer_inventory_01234567.json" and path.parent == snap.parent.parent
    digest = HI.write_inventory(path, inv)
    assert len(digest) == 64 and HI.read_inventory(path) == json.loads(path.read_text())


def test_a_mutated_shard_a_missing_shard_and_a_wrong_size_plain_file_each_refuse_by_name(tmp_path):
    hf_home, snap, inv, files = _snapshot(tmp_path)
    # mutated shard: same size, different bytes -> the digest refuses (the size check alone would pass it)
    p = snap / "model-00002-of-00002.safetensors"
    data = bytearray(p.read_bytes()); data[-1] ^= 0xFF; p.write_bytes(bytes(data))
    res = HI.verify_snapshot(snap, inv)
    assert not res["ok"] and len(res["problems"]) == 1 and res["problems"][0].startswith("model-00002-of-00002.safetensors: lfs digest")
    assert HI.verify_snapshot(snap, inv, hash_bytes=False)["ok"]          # the cheap check cannot see it — which is why the preflight hashes
    p.write_bytes(files["model-00002-of-00002.safetensors"])
    # missing shard
    (snap / "model-00001-of-00002.safetensors").unlink()
    res = HI.verify_snapshot(snap, inv)
    assert not res["ok"] and res["problems"] == ["model-00001-of-00002.safetensors: MISSING from %s" % snap]
    (snap / "model-00001-of-00002.safetensors").write_bytes(files["model-00001-of-00002.safetensors"])
    # wrong-size plain file -> refused on size before any hash
    (snap / "generation_config.json").write_bytes(b'{"temperature": 1.0, "top_p": 0.95}\n')
    res = HI.verify_snapshot(snap, inv)
    assert not res["ok"] and res["problems"][0].startswith("generation_config.json: size ")
    (snap / "generation_config.json").write_bytes(files["generation_config.json"])
    # a same-size plain file with other bytes -> the blob SHA-1 refuses
    (snap / "chat_template.jinja").write_bytes(b"{{ messagez }}\n")
    res = HI.verify_snapshot(snap, inv)
    assert not res["ok"] and res["problems"][0].startswith("chat_template.jinja: blob digest")
    (snap / "chat_template.jinja").write_bytes(files["chat_template.jinja"])
    # an extra file is listed, and refuses only under strict
    (snap / "README.md").write_bytes(b"x")
    assert HI.verify_snapshot(snap, inv)["ok"] and HI.verify_snapshot(snap, inv)["extra"] == ["README.md"]
    assert not HI.verify_snapshot(snap, inv, strict=True)["ok"]
    # no snapshot at all
    assert HI.verify_snapshot(tmp_path / "nowhere", inv)["problems"] == [f"snapshot dir {tmp_path / 'nowhere'} is not a directory"]


def test_a_listing_that_cannot_name_a_digest_refuses_and_so_does_a_malformed_inventory(tmp_path):
    rev = "0123456789abcdef0123456789abcdef01234567"
    with pytest.raises(HI.InventoryError, match="listed no size"):
        HI.inventory_from_siblings("Org/Model", rev, rev, [{"rfilename": "config.json", "size": None, "lfs": None, "blob_id": "a" * 40}])
    with pytest.raises(HI.InventoryError, match="LFS file without a sha256"):
        HI.inventory_from_siblings("Org/Model", rev, rev, [{"rfilename": "config.json", "size": 3, "lfs": {"size": 3}, "blob_id": "a" * 40}])
    with pytest.raises(HI.InventoryError, match="plain file without a blob_id"):
        HI.inventory_from_siblings("Org/Model", rev, rev, [{"rfilename": "config.json", "size": 3, "lfs": None, "blob_id": None}])
    with pytest.raises(HI.InventoryError, match="listed no files"):
        HI.inventory_from_siblings("Org/Model", rev, rev, [])
    with pytest.raises(HI.InventoryError, match="lacks \\['config.json'\\]"):
        HI.inventory_from_siblings("Org/Model", rev, rev, [{"rfilename": "x.bin", "size": 3, "lfs": {"sha256": "b" * 64}, "blob_id": None}])
    good = HI.inventory_from_siblings("Org/Model", rev, rev, [{"rfilename": "config.json", "size": 3, "lfs": None, "blob_id": "a" * 40}])
    assert HI.inventory_problems(good) == []
    bad = json.loads(json.dumps(good)); bad["files"]["config.json"]["digest"] = "zz"
    assert any("not a 40-hex blob digest" in p for p in HI.inventory_problems(bad))
    bad = json.loads(json.dumps(good)); bad["files"]["config.json"]["kind"] = "sha256"
    assert any("kind 'sha256' not in" in p for p in HI.inventory_problems(bad))
    assert HI.inventory_problems({"schema": "other"})[0].startswith("missing")
    # a malformed inventory never verifies a snapshot
    res = HI.verify_snapshot(tmp_path, bad)
    assert not res["ok"] and res["problems"][0].startswith("inventory: ")
    with pytest.raises(HI.InventoryError, match="no inventory at"):
        HI.read_inventory(tmp_path / "absent.json")
