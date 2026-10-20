"""A committed object LARGER than the 40 MB chunk bound round-trips through the packaging the exporter and the loaders
share: `reproduce/tools/export_objects.package_artifact` -> `object.pt.part-NN` + `CHUNKS.json`
(no whole `object.pt` committed; every part under the bound) -> `reproduce/lib/object_chunks.reassemble` -> the ORIGINAL
bytes (sha256 == the object_id) -> `torch.load` gives back the tensors that were saved. The 149 committed paper objects are all
under the bound (max 8.4 MB), so this synthetic artifact is the only exercise of the chunked form until the test model's
MLP-set LoRA objects (236 MB - 2 GB) are committed."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
CHUNK_MB = 40
CHUNK_BYTES = CHUNK_MB * 1000 * 1000


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def chunks():
    return _load("object_chunks", "reproduce/lib/object_chunks.py")


@pytest.fixture(scope="module")
def exporter():
    return _load("export_objects", "reproduce/tools/export_objects.py")


@pytest.fixture(scope="module")
def big_object(tmp_path_factory):
    """A ~52 MB fp32 LoRA-shaped object.pt (13M floats) — deterministic, above the 40 MB bound."""
    d = tmp_path_factory.mktemp("bank") / "cells" / "synthetic__lora"
    d.mkdir(parents=True)
    g = torch.Generator().manual_seed(1234)
    obj = {
        "kind": "lora",
        "rank": 4,
        "tensors": {
            "layers.0.self_attn.o_proj.lora_A": torch.randn(4, 2048, generator=g),
            "layers.0.self_attn.o_proj.lora_B": torch.randn(2048, 4, generator=g),
            "payload": torch.randn(13_000_000, generator=g),
        },
        "meta": {"synthetic": True},
    }
    p = d / "object.pt"
    torch.save(obj, p)
    assert p.stat().st_size > CHUNK_BYTES, p.stat().st_size
    (d / "train_budget_meta.json").write_text(json.dumps({"seed": 0, "arm": "lora_r4_ce"}))
    (d / "COMPLETE").write_text("ok\n")
    return p, obj


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exporter_chunks_a_large_artifact_and_reassembly_loads(tmp_path, chunks, exporter, big_object):
    src, obj = big_object
    object_id = _sha(src)
    dest = tmp_path / "objects" / "synthetic_model" / object_id
    pk = exporter.package_artifact(src, dest, CHUNK_BYTES, chunks)
    assert pk["form"] == "chunked" and pk["chunks"] >= 2 and pk["sha256"] == object_id and pk["bytes"] == src.stat().st_size
    assert not (dest / "object.pt").exists(), "a chunked object commits no whole object.pt"
    man = json.loads((dest / "CHUNKS.json").read_text())
    assert man["original_sha256"] == object_id and man["compression"] == "none"
    parts = sorted(dest.glob("object.pt.part-*"))
    assert [p.name for p in parts] == [c["name"] for c in man["chunks"]]
    for p, c in zip(parts, man["chunks"], strict=True):
        assert p.stat().st_size <= CHUNK_BYTES and p.stat().st_size == c["bytes"] and _sha(p) == c["sha256"]
    sidecars = exporter.copy_sidecars(src.parent, dest)
    assert sidecars == ["train_budget_meta.json", "COMPLETE"]
    # the LOAD path: reassemble (verifying every chunk + the original sha) then torch.load
    assert chunks.is_chunked(str(dest))
    out = chunks.reassemble(str(dest), dest=str(tmp_path / "reassembled" / object_id / "object.pt"))
    assert Path(out).is_file() and _sha(Path(out)) == object_id
    loaded = torch.load(out, map_location="cpu", weights_only=False)
    assert loaded["kind"] == "lora" and loaded["rank"] == 4 and loaded["meta"] == {"synthetic": True}
    for k, t in obj["tensors"].items():
        assert torch.equal(loaded["tensors"][k], t), k
    # idempotent: a second reassemble returns the verified file without rewriting it
    st = Path(out).stat()
    assert chunks.reassemble(str(dest), dest=out) == out and Path(out).stat().st_mtime_ns == st.st_mtime_ns


def test_tampered_chunk_is_refused_before_any_load(tmp_path, chunks, exporter, big_object):
    src, _ = big_object
    dest = tmp_path / "objects" / "m" / _sha(src)
    exporter.package_artifact(src, dest, CHUNK_BYTES, chunks)
    part = sorted(dest.glob("object.pt.part-*"))[1]
    b = bytearray(part.read_bytes())
    b[100] ^= 0x01
    part.write_bytes(bytes(b))
    with pytest.raises(ValueError, match="sha"):
        chunks.reassemble(str(dest), dest=str(tmp_path / "r" / "object.pt"))
    assert not (tmp_path / "r" / "object.pt").exists()


def test_small_artifact_is_committed_whole(tmp_path, chunks, exporter):
    src = tmp_path / "small" / "object.pt"
    src.parent.mkdir()
    torch.save({"kind": "vector_all", "vectors": {12: torch.zeros(2560)}}, src)
    dest = tmp_path / "objects" / "m" / _sha(src)
    pk = exporter.package_artifact(src, dest, CHUNK_BYTES, chunks)
    assert pk["form"] == "whole" and pk["chunks"] == 0 and (dest / "object.pt").read_bytes() == src.read_bytes()
    assert not (dest / "CHUNKS.json").exists()


def test_committed_manifest_agrees_with_the_chunk_rule():
    """Every committed paper object is under the bound, so all 149 are `whole`; the rule is recorded in the manifest."""
    man = json.loads((REPO / "objects/MANIFEST.json").read_text())
    assert man["chunk_bytes"] == CHUNK_BYTES
    for e in man["objects"]:
        if e["form"] == "virtual":
            continue
        assert e["form"] == ("chunked" if e["bytes"] > CHUNK_BYTES else "whole"), e["object_id"]
