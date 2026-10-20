#!/usr/bin/env python3
"""reproduce/tools/reassemble_corpus.py — rebuild one part of a released corpus (the constitution or the control corpus; its difficult-advice or its stories part) from its committed shards and CHECK it
(the reassembled sha256 equals the final's; the record count is the paper's).

    python reproduce/tools/reassemble_corpus.py <corpus> --out FILE [--check] [--corpora-root corpora]
    python reproduce/tools/reassemble_corpus.py <corpus> --verify-only            # hash + count, writes nothing
    python reproduce/tools/reassemble_corpus.py --list                            # the corpus parts and their manifests

Every run verifies: each shard's sha256 and record count against MANIFEST.json, then the reassembled stream's sha256,
byte count and record count. `--out` is written atomically (temp file beside it + rename) and only after the checks
pass, so a partial or corrupt file never sits at the destination. `--check` is accepted for the front door's contract
(verification is always on). The library surface (`load_manifest`, `iter_records`, `reassemble`) is what the tests
and stage 02 use.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterator

REPO = Path(__file__).resolve().parents[2]
CORPORA_ROOT = REPO / "corpora"
CORPORA = ("constitution_advice", "constitution_stories", "control_advice", "control_stories")
_BUF = 1 << 20


class CorpusCheckError(RuntimeError):
    """A shard or the reassembled corpus does not match its manifest — never degrade to a silently-wrong file."""


def load_manifest(corpus: str, root: Path = CORPORA_ROOT) -> dict:
    p = Path(root) / corpus / "MANIFEST.json"
    if not p.is_file():
        raise CorpusCheckError(f"{p} missing (known parts: {', '.join(CORPORA)})")
    return json.loads(p.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(_BUF), b""):
            h.update(b)
    return h.hexdigest()


def verify_shard_files(corpus: str, root: Path = CORPORA_ROOT) -> dict:
    """Every shard present, its compressed sha256 and size as recorded; returns the manifest."""
    man = load_manifest(corpus, root)
    d = Path(root) / corpus
    for s in man["shards"]:
        p = d / s["name"]
        if not p.is_file():
            raise CorpusCheckError(f"{corpus}: shard {s['name']} missing under {d}")
        if p.stat().st_size != s["bytes"]:
            raise CorpusCheckError(f"{corpus}: shard {s['name']} is {p.stat().st_size} B, manifest says {s['bytes']}")
        got = sha256_file(p)
        if got != s["sha256"]:
            raise CorpusCheckError(f"{corpus}: shard {s['name']} sha256 {got[:12]} != manifest {s['sha256'][:12]}")
    for sc in man.get("sidecars", []):
        p = d / sc["name"]
        if not p.is_file():
            raise CorpusCheckError(f"{corpus}: sidecar {sc['name']} missing")
        if sha256_file(p) != sc["sha256"]:
            raise CorpusCheckError(f"{corpus}: sidecar {sc['name']} sha256 mismatch")
    return man


def iter_shard_lines(corpus: str, root: Path = CORPORA_ROOT, man: dict | None = None) -> Iterator[bytes]:
    """The raw record lines (incl. newline) of every shard in order, checking each shard's record count as it ends."""
    man = man or load_manifest(corpus, root)
    d = Path(root) / corpus
    for s in man["shards"]:
        n = 0
        with gzip.open(d / s["name"], "rb") as gz:
            for line in gz:
                n += 1
                yield line
        if n != s["records"]:
            raise CorpusCheckError(f"{corpus}: shard {s['name']} holds {n} records, manifest says {s['records']}")


def iter_records(corpus: str, root: Path = CORPORA_ROOT) -> Iterator[dict]:
    """The corpus's records as dicts, straight from the shards (no file written)."""
    for line in iter_shard_lines(corpus, root):
        yield json.loads(line)


def reassemble(corpus: str, out: Path | None, root: Path = CORPORA_ROOT) -> dict:
    """Stream the shards, verify, and (when `out` is given) write the final atomically.

    Returns {corpus, sha256, bytes, records, out}. Raises CorpusCheckError on any mismatch; nothing is left at `out`."""
    man = verify_shard_files(corpus, root)
    h = hashlib.sha256()
    nbytes = 0
    nrec = 0
    tmp = None
    fh = None
    if out is not None:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=f".{out.name}.", dir=str(out.parent))
        fh = os.fdopen(fd, "wb")
    try:
        for line in iter_shard_lines(corpus, root, man):
            h.update(line)
            nbytes += len(line)
            nrec += 1
            if fh is not None:
                fh.write(line)
        if fh is not None:
            fh.flush()
            os.fsync(fh.fileno())
            fh.close()
            fh = None
        sha = h.hexdigest()
        problems = []
        if sha != man["sha256_reassembled"]:
            problems.append(f"sha256 {sha} != manifest {man['sha256_reassembled']}")
        if nbytes != man["bytes_reassembled"]:
            problems.append(f"bytes {nbytes} != manifest {man['bytes_reassembled']}")
        if nrec != man["record_count"]:
            problems.append(f"records {nrec} != manifest {man['record_count']}")
        if problems:
            raise CorpusCheckError(f"{corpus}: reassembly does not match its manifest: " + "; ".join(problems))
        if tmp is not None:
            os.replace(tmp, out)
            tmp = None
    finally:
        if fh is not None:
            fh.close()
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)
    return {"corpus": corpus, "sha256": sha, "bytes": nbytes, "records": nrec, "out": str(out) if out else None,
            "source_file": man["source_file"], "release": man["release"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("corpus", nargs="?", metavar="part", help="a corpus part: " + " | ".join(CORPORA))
    ap.add_argument("--out", type=Path, default=None, help="write the reassembled JSONL final here (atomic)")
    ap.add_argument("--check", action="store_true", help="(always on) verify shard shas, counts and the reassembled sha")
    ap.add_argument("--verify-only", action="store_true", help="hash + count without writing anything")
    ap.add_argument("--corpora-root", type=Path, default=CORPORA_ROOT)
    ap.add_argument("--list", action="store_true", help="list the corpus parts with their manifests")
    a = ap.parse_args(argv)
    if a.list:
        for c in CORPORA:
            try:
                m = load_manifest(c, a.corpora_root)
                print(f"{c}: {m['record_count']} records, {len(m['shards'])} shards, sha256 {m['sha256_reassembled'][:16]}..., "
                      f"source {m['source_file']}")
            except CorpusCheckError as e:
                print(f"{c}: {e}")
        return 0
    if not a.corpus:
        ap.error("a corpus part is required (or --list)")
    if a.out is None and not a.verify_only:
        ap.error("--out FILE (or --verify-only) is required")
    try:
        res = reassemble(a.corpus, None if a.verify_only else a.out, a.corpora_root)
    except CorpusCheckError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
