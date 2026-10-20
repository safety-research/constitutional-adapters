"""``sisteer.extraction.load_committed_object``: a committed object loads from
its ``objects/<model_id>/<object_id>/``
directory — whole ``object.pt`` or the lossless chunked form reassembled
through ``reproduce/lib/object_chunks.reassemble`` — ONLY after
``sha256(bytes) == object_id`` holds; tampered bytes, a manifest that
disagrees, or chunks without the reassembly library refuse by name. The
chunked path is driven through B's synthetic ~52 MB artifact packaged by
the real exporter (``reproduce/tools/export_objects.package_artifact``)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import sisteer.extraction as ext
from sisteer.extraction import committed_objects as co
# the exporter/reassembly fixtures and the ~52 MB synthetic artifact are test_object_chunks_load's (pytest discovers
# imported fixtures); one definition of the chunked-form probe for both consumers of the packaging
from test_object_chunks_load import CHUNK_BYTES, _sha, big_object, chunks, exporter  # noqa: F401

REPO = Path(__file__).resolve().parents[1]


def test_lazy_export_from_the_package():
    assert ext.load_committed_object is co.load_committed_object
    assert ext.CommittedObjectError is co.CommittedObjectError
    with pytest.raises(AttributeError):
        ext.not_a_thing  # noqa: B018


def test_whole_form_loads_and_verifies_the_dir_name(tmp_path):
    blob = {"kind": "vector_all", "vectors": {"12": torch.zeros(16)}}
    src = tmp_path / "src.pt"
    torch.save(blob, src)
    oid = _sha(src)
    d = tmp_path / "objects" / "modelx" / oid
    d.mkdir(parents=True)
    (d / "object.pt").write_bytes(src.read_bytes())
    loaded = co.load_committed_object(d)
    assert loaded["kind"] == "vector_all" and torch.equal(loaded["vectors"]["12"], torch.zeros(16))
    path, verified = co.committed_object_bytes_path(d)
    assert path == d / "object.pt" and verified == oid
    # explicit id wins and is checked
    assert co.load_committed_object(d, object_id=oid)["kind"] == "vector_all"
    with pytest.raises(co.CommittedObjectError, match="do not hash"):
        co.load_committed_object(d, object_id="0" * 64)


def test_tampered_whole_bytes_refuse_before_torch_load(tmp_path, monkeypatch):
    src = tmp_path / "src.pt"
    torch.save({"kind": "vector_all"}, src)
    d = tmp_path / "objects" / "m" / _sha(src)
    d.mkdir(parents=True)
    b = bytearray(src.read_bytes())
    b[-1] ^= 0x01
    (d / "object.pt").write_bytes(bytes(b))
    called = []
    monkeypatch.setattr(torch, "load", lambda *a, **k: called.append(a))
    with pytest.raises(co.CommittedObjectError, match="do not hash"):
        co.load_committed_object(d)
    assert called == []


def test_non_hex_dir_without_manifest_refuses_by_name(tmp_path):
    d = tmp_path / "objects" / "m" / "not-an-id"
    d.mkdir(parents=True)
    torch.save({"kind": "x"}, d / "object.pt")
    with pytest.raises(co.CommittedObjectError, match="cannot determine the object_id"):
        co.load_committed_object(d)
    # ... unless a MANIFEST row names it
    man = {"schema": 1, "objects": [{"object_id": _sha(d / "object.pt"), "dir": "m/not-an-id",
                                     "artifact_sha256": _sha(d / "object.pt"), "form": "whole"}]}
    (tmp_path / "objects" / "MANIFEST.json").write_text(json.dumps(man))
    assert co.load_committed_object(d)["kind"] == "x"
    # a manifest that disagrees with the identity refuses
    man["objects"][0]["artifact_sha256"] = "f" * 64
    (tmp_path / "objects" / "MANIFEST.json").write_text(json.dumps(man))
    with pytest.raises(co.CommittedObjectError, match="disagree"):
        co.load_committed_object(d, object_id=_sha(d / "object.pt"))


def test_chunked_form_reassembles_verifies_and_loads(tmp_path, chunks, exporter, big_object):
    src, obj = big_object
    oid = _sha(src)
    dest = tmp_path / "objects" / "synthetic_model" / oid
    pk = exporter.package_artifact(src, dest, CHUNK_BYTES, chunks)
    assert pk["form"] == "chunked" and pk["chunks"] >= 2 and not (dest / "object.pt").exists()
    out_path = tmp_path / "reassembled" / oid / "object.pt"
    loaded = co.load_committed_object(dest, repo_root=REPO, reassemble_dest=out_path)
    assert loaded["kind"] == "lora" and loaded["rank"] == 4 and loaded["meta"] == {"synthetic": True}
    for k, t in obj["tensors"].items():
        assert torch.equal(loaded["tensors"][k], t), k
    assert out_path.is_file() and _sha(out_path) == oid
    # the DEFAULT destination is the gitignored cache directly under the objects root — objects/_reassembled/<object_id>/
    # object.pt — never inside objects/<model_id>/, which nothing ignores and the manifest census walks (the lib's own
    # dest=None default would be objects/<model_id>/_reassembled/...)
    path, verified = co.committed_object_bytes_path(dest, repo_root=REPO)
    assert verified == oid and path == tmp_path / "objects" / "_reassembled" / oid / "object.pt"
    assert path == co.default_reassemble_dest(dest) and path.is_file() and _sha(path) == oid
    assert "_reassembled" not in {p.name for p in (tmp_path / "objects" / "synthetic_model").iterdir()}


def test_swapped_chunks_with_matching_chunk_hashes_refuse(tmp_path, chunks, exporter, big_object):
    """Two chunks' bytes swapped and their own recorded sha256s updated to match pass
    every per-chunk check; the loader still refuses — the lib's reassembly compares the streamed original to
    `original_sha256`, and when THAT is rewritten too the loader's own sha256(bytes) == object_id check refuses."""
    src, _ = big_object
    oid = _sha(src)
    dest = tmp_path / "objects" / "m" / oid
    exporter.package_artifact(src, dest, CHUNK_BYTES, chunks)
    cm = json.loads((dest / "CHUNKS.json").read_text())
    a, b = dest / cm["chunks"][0]["name"], dest / cm["chunks"][1]["name"]
    a_bytes, b_bytes = a.read_bytes(), b.read_bytes()
    a.write_bytes(b_bytes)
    b.write_bytes(a_bytes)
    cm["chunks"][0]["sha256"], cm["chunks"][1]["sha256"] = _sha(a), _sha(b)
    cm["chunks"][0]["bytes"], cm["chunks"][1]["bytes"] = a.stat().st_size, b.stat().st_size
    (dest / "CHUNKS.json").write_text(json.dumps(cm))
    with pytest.raises(co.CommittedObjectError, match="reassembly refused"):
        co.load_committed_object(dest, repo_root=REPO, reassemble_dest=tmp_path / "r1" / "object.pt")
    assert not (tmp_path / "r1" / "object.pt").exists()
    # a CHUNKS.json made fully consistent with the wrong bytes: the identity check is the object_id, not the manifest
    rest = b"".join((dest / c["name"]).read_bytes() for c in cm["chunks"][2:])
    cm["original_sha256"] = hashlib.sha256(b_bytes + a_bytes + rest).hexdigest()
    (dest / "CHUNKS.json").write_text(json.dumps(cm))
    with pytest.raises(co.CommittedObjectError, match="do not hash to their registry identity"):
        co.load_committed_object(dest, repo_root=REPO, reassemble_dest=tmp_path / "r2" / "object.pt")


def test_tampered_chunk_refuses(tmp_path, chunks, exporter, big_object):
    src, _ = big_object
    dest = tmp_path / "objects" / "m" / _sha(src)
    exporter.package_artifact(src, dest, CHUNK_BYTES, chunks)
    part = sorted(dest.glob("object.pt.part-*"))[0]
    b = bytearray(part.read_bytes())
    b[10] ^= 0x01
    part.write_bytes(bytes(b))
    with pytest.raises(co.CommittedObjectError, match="reassembly refused"):
        co.load_committed_object(dest, repo_root=REPO, reassemble_dest=tmp_path / "r" / "object.pt")
    assert not (tmp_path / "r" / "object.pt").exists()


def test_chunks_without_the_library_refuse_by_name(tmp_path, monkeypatch):
    d = tmp_path / "objects" / "m" / ("a" * 64)
    d.mkdir(parents=True)
    (d / "CHUNKS.json").write_text("{}")
    monkeypatch.setattr(co, "repo_root_of", lambda p: None)
    # the module's own checkout is where the lib would be found last; point it nowhere
    monkeypatch.setattr(co, "__file__", str(tmp_path / "elsewhere" / "pkg" / "committed_objects.py"))
    with pytest.raises(co.CommittedObjectError, match="object_chunks.py is not present"):
        co.load_committed_object(d, repo_root=tmp_path / "nowhere")


@pytest.mark.skipif(not (REPO / "objects" / "MANIFEST.json").is_file(), reason="no committed objects tree")
def test_a_committed_paper_object_loads_by_its_id():
    """Every committed whole-form paper object hashes to its dir name; load one."""
    man = json.loads((REPO / "objects" / "MANIFEST.json").read_text())
    rows = [r for r in man["objects"] if r.get("form") == "whole"]
    assert rows
    row = rows[0]
    d = REPO / "objects" / row["dir"]
    blob = co.load_committed_object(d)
    assert isinstance(blob, dict) and "kind" in blob
    assert co.committed_object_bytes_path(d)[1] == row["object_id"] == row["artifact_sha256"]
