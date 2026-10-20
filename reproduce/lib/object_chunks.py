#!/usr/bin/env python3
"""reproduce/lib/object_chunks.py — lossless, size-bounded packaging of a committed object artifact.

An artifact > 40 MB (a large model's MLP-set LoRA objects run 236 MB – 2 GB; a difference doubles it) is committed as
ordered chunks `object.pt.part-NN` (40 MB each, zstd-compressed when `zstandard` is importable else raw — recorded) plus
`CHUNKS.json` {original_name, original_sha256, original_bytes, chunk_bytes, compression, chunks:[{name, sha256, bytes}]}.
`reassemble(dir)` rebuilds the ORIGINAL bytes into `objects/_reassembled/<object_id>/object.pt` (the gitignored cache directly
under the objects root — `objects/` holds `<model_id>/<object_id>/`, so two levels up; the same default as
`sisteer.extraction.committed_objects.default_reassemble_dest`) atomically (temp + rename), verifying every chunk's sha and
the original's sha + byte size before returning, so `sha256(object.pt) == object_id` holds exactly. `verify(dir)` streams the
same reassembly (to nowhere) and checks the same three facts, so a wrong or reordered chunk behind a self-consistent
CHUNKS.json fails.

    python object_chunks.py chunk <object.pt> --out DIR [--chunk-mb 40] [--compression zstd|none]
    python object_chunks.py reassemble DIR [--dest PATH]
    python object_chunks.py verify DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from typing import Optional

try:  # optional: lossless compression of each chunk
    import zstandard as _zstd
except Exception:  # pragma: no cover - environment-dependent
    _zstd = None

DEFAULT_CHUNK_BYTES = 40 * 1000 * 1000
MANIFEST = "CHUNKS.json"
_BUF = 1 << 20


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(_BUF), b""):
            h.update(b)
    return h.hexdigest()


def chunk(path: str, out_dir: str, chunk_bytes: int = DEFAULT_CHUNK_BYTES, compression: str = "auto") -> dict:
    if compression == "auto":
        compression = "zstd" if _zstd is not None else "none"
    if compression == "zstd" and _zstd is None:
        raise RuntimeError("compression=zstd requested but `zstandard` is not importable")
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(path)
    total = os.path.getsize(path)
    orig = hashlib.sha256()
    chunks = []
    with open(path, "rb") as fh:
        idx = 0
        while True:
            raw = fh.read(chunk_bytes)
            if not raw:
                break
            orig.update(raw)
            data = _zstd.ZstdCompressor(level=6).compress(raw) if compression == "zstd" else raw
            cname = f"{name}.part-{idx:02d}"
            cpath = os.path.join(out_dir, cname)
            with open(cpath + ".tmp", "wb") as out:
                out.write(data)
            os.replace(cpath + ".tmp", cpath)
            chunks.append({"name": cname, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "raw_bytes": len(raw)})
            idx += 1
    man = {"schema": 1, "original_name": name, "original_sha256": orig.hexdigest(), "original_bytes": total,
           "chunk_bytes": chunk_bytes, "compression": compression, "chunks": chunks}
    with open(os.path.join(out_dir, MANIFEST + ".tmp"), "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=2, sort_keys=True)
    os.replace(os.path.join(out_dir, MANIFEST + ".tmp"), os.path.join(out_dir, MANIFEST))
    return man


def is_chunked(d: str) -> bool:
    return os.path.isfile(os.path.join(d, MANIFEST))


def _manifest(d: str) -> dict:
    return json.load(open(os.path.join(d, MANIFEST), encoding="utf-8"))


def _stream(d: str, man: dict, sink) -> None:
    """Read the chunks IN MANIFEST ORDER, check each chunk's recorded sha, decompress, hand the original bytes to `sink`, and then
    check the reassembled sha256 AND byte size against the recorded original — the three facts a chunk set must satisfy. A
    reordered / substituted chunk whose CHUNKS.json was rewritten to match it still fails here: the original's sha and size were
    recorded from the object the census hashed (its object_id), not from the chunks."""
    if man["compression"] == "zstd" and _zstd is None:
        raise RuntimeError("chunks are zstd-compressed but `zstandard` is not importable")
    dec = _zstd.ZstdDecompressor() if man["compression"] == "zstd" else None
    h = hashlib.sha256(); total = 0
    for c in man["chunks"]:
        p = os.path.join(d, c["name"])
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{d}: chunk {c['name']} missing")
        data = open(p, "rb").read()
        got = hashlib.sha256(data).hexdigest()
        if got != c["sha256"]:
            raise ValueError(f"{d}: chunk {c['name']} sha {got[:12]} != recorded {c['sha256'][:12]}")
        raw = dec.decompress(data, max_output_size=c.get("raw_bytes", man["chunk_bytes"])) if dec else data
        h.update(raw); total += len(raw)
        sink(raw)
    if total != man["original_bytes"]:
        raise ValueError(f"{d}: reassembled {total} bytes != recorded original_bytes {man['original_bytes']}")
    if h.hexdigest() != man["original_sha256"]:
        raise ValueError(f"{d}: reassembled sha {h.hexdigest()[:12]} != recorded original_sha256 {man['original_sha256'][:12]}")


def verify(d: str) -> dict:
    """Every chunk present with its recorded sha AND the stream-reassembled original hashing to `original_sha256` with exactly
    `original_bytes` bytes (nothing is written) — the committed-object reader's semantics."""
    man = _manifest(d)
    _stream(d, man, lambda raw: None)
    return man


def default_dest(d: str, original_name: str) -> str:
    """`<objects root>/_reassembled/<object_id>/<original_name>`: the objects root is two levels up from `objects/<model_id>/<object_id>`
    (never inside `objects/<model_id>/`, which nothing ignores and the manifest census walks) — the path
    `committed_objects.default_reassemble_dest` pins."""
    d = os.path.abspath(d)
    return os.path.join(os.path.dirname(os.path.dirname(d)), "_reassembled", os.path.basename(d), original_name)


def reassemble(d: str, dest: Optional[str] = None) -> str:
    """Rebuild the original file (default `default_dest`: `objects/_reassembled/<object_id>/<original_name>`); returns its path."""
    man = _manifest(d)
    if dest is None:
        dest = default_dest(d, man["original_name"])
    if os.path.isfile(dest) and os.path.getsize(dest) == man["original_bytes"] and sha256_file(dest) == man["original_sha256"]:
        return dest                          # already rebuilt and verified: the chunks need not be re-read
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".reassemble-", dir=os.path.dirname(dest))
    try:
        with os.fdopen(fd, "wb") as out:
            _stream(d, man, out.write)        # ONE pass: chunk shas, the original's sha and size are all checked before the rename
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("chunk"); p.add_argument("path"); p.add_argument("--out", required=True)
    p.add_argument("--chunk-mb", type=int, default=40); p.add_argument("--compression", default="auto", choices=["auto", "zstd", "none"])
    p = sub.add_parser("reassemble"); p.add_argument("dir"); p.add_argument("--dest", default=None)
    p = sub.add_parser("verify"); p.add_argument("dir")
    a = ap.parse_args(argv)
    if a.cmd == "chunk":
        man = chunk(a.path, a.out, a.chunk_mb * 1000 * 1000, a.compression)
        print(json.dumps({k: man[k] for k in ("original_sha256", "original_bytes", "compression")} | {"n_chunks": len(man["chunks"])}))
    elif a.cmd == "reassemble":
        print(reassemble(a.dir, a.dest))
    else:
        man = verify(a.dir); print(f"ok {len(man['chunks'])} chunks {man['original_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
