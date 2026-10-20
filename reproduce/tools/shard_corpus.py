#!/usr/bin/env python3
"""reproduce/tools/shard_corpus.py — cut ONE released corpus (a JSONL final) into record-boundary gzip shards
committed under corpora/<corpus>/.

    python reproduce/tools/shard_corpus.py <corpus> --source FILE --out corpora/<corpus>
        [--target-mb 40] [--hard-mb 50] [--sidecar FILE]... [--release NAME] [--vendored-commit SHA]

Layout written:
    corpora/<corpus>/part-NN.jsonl.gz      each shard is itself valid JSONL once decompressed (cut between records),
                                           <= --target-mb compressed (40 MB by default; < --hard-mb = 50 MB always)
    corpora/<corpus>/MANIFEST.json         {corpus, source_file, release, record_count, sha256_reassembled, bytes_reassembled,
                                            shards:[{name, sha256, bytes, records, raw_bytes}], official_tokens:{llama3, gemma,
                                            qwen, min}, sidecars:[{name, sha256, bytes}], gzip:{...}, tool, vendored_corpus_commit}
    corpora/<corpus>/<sidecar files>       the finals' own sidecars, ORIGINAL bytes (.filter-report.json, .import-scan.json,
                                           .register-edit.json, section-tags-*)

Reproducible bytes: every shard is written through gzip with mtime=0 and no filename (the `gzip -n` contract) at one
fixed level, so re-running over the same final yields byte-identical shards. Shard boundaries are chosen
DETERMINISTICALLY from the final alone: the whole file is compressed once to measure its gzip size, n = ceil(size /
target) shards are cut at record boundaries on an equal RAW-byte budget, and if any shard then exceeds the target n is
raised by one and the cut redone. `reproduce/tools/reassemble_corpus.py` is the inverse and the check.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

GZIP_LEVEL = 9
TARGET_MB = 40
HARD_MB = 50
_BUF = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(_BUF), b""):
            h.update(b)
    return h.hexdigest()


def _gzip_writer(fileobj) -> gzip.GzipFile:
    """A gzip stream with reproducible bytes (no mtime, no filename, fixed level) over an open binary sink."""
    return gzip.GzipFile(filename="", mode="wb", fileobj=fileobj, mtime=0, compresslevel=GZIP_LEVEL)


class _CountingSink:
    def __init__(self):
        self.n = 0

    def write(self, b):
        self.n += len(b)
        return len(b)

    def flush(self):
        pass

    def close(self):
        pass


def measure_gzip_bytes(source: Path) -> int:
    sink = _CountingSink()
    with _gzip_writer(sink) as gz, open(source, "rb") as fh:
        for b in iter(lambda: fh.read(_BUF), b""):
            gz.write(b)
    return sink.n


def iter_records(source: Path):
    """Yield the raw bytes of each JSONL record INCLUDING its newline (a final never has a trailing partial line)."""
    with open(source, "rb") as fh:
        yield from fh


def cut_shards(source: Path, out_dir: Path, n_shards: int) -> list[dict]:
    """ONE streaming pass over the final: records go into part-NN in order, and a shard closes once its RAW bytes
    reach the equal budget (the last shard takes the remainder), so every cut falls between records."""
    budget = math.ceil(source.stat().st_size / n_shards)
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[dict] = []
    records = iter_records(source)
    line = next(records, None)
    while line is not None:
        path = out_dir / f"part-{len(shards):02d}.jsonl.gz"
        raw = recs = 0
        with open(path, "wb") as sink, _gzip_writer(sink) as gz:   # gz closes (trailer) before sink closes (flush)
            while line is not None:
                gz.write(line)
                raw += len(line)
                recs += 1
                line = next(records, None)
                if raw >= budget and len(shards) < n_shards - 1:
                    break
        shards.append({"name": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size,
                       "records": recs, "raw_bytes": raw})
    return shards


def official_tokens(sidecar_reports: list[Path]) -> dict | None:
    """{llama3, gemma, qwen, min} from the final's .filter-report.json (token_totals), when present."""
    for p in sidecar_reports:
        if p.name.endswith(".filter-report.json"):
            fr = json.loads(p.read_text(encoding="utf-8"))
            tt = fr.get("token_totals") or {}
            if tt:
                out = {k: int(v) for k, v in tt.items()}
                out["min"] = min(out.values())
                if fr.get("official_tokens") is not None and int(fr["official_tokens"]) != out["min"]:
                    raise SystemExit(f"{p}: official_tokens {fr['official_tokens']} != min(token_totals) {out['min']}")
                return out
    return None


def shard_corpus(corpus: str, source: Path, out_dir: Path, *, sidecars: list[Path], release: str,
                 target_mb: int = TARGET_MB, hard_mb: int = HARD_MB, vendored_commit: str | None = None) -> dict:
    source = Path(source)
    if not source.is_file():
        raise SystemExit(f"REFUSED: source final {source} does not exist")
    out_dir = Path(out_dir)
    if any(out_dir.glob("part-*.jsonl.gz")):
        raise SystemExit(f"REFUSED: {out_dir} already holds shards; remove them to re-shard")
    target = target_mb * 1000 * 1000
    hard = hard_mb * 1000 * 1000
    src_sha = sha256_file(source)
    src_bytes = source.stat().st_size
    gz_total = measure_gzip_bytes(source)
    n = max(1, math.ceil(gz_total / target))
    while True:
        for p in out_dir.glob("part-*.jsonl.gz"):
            p.unlink()
        shards = cut_shards(source, out_dir, n)
        if all(s["bytes"] <= target for s in shards):
            break
        n += 1
        if n > 64:
            raise SystemExit("REFUSED: could not fit shards under the target in 64 pieces")
    for s in shards:
        if s["bytes"] >= hard:
            raise SystemExit(f"REFUSED: shard {s['name']} is {s['bytes']} B >= the {hard_mb} MB hard limit")
    if sum(s["raw_bytes"] for s in shards) != src_bytes:   # every byte of the final landed in exactly one shard
        raise SystemExit(f"REFUSED: shards hold {sum(s['raw_bytes'] for s in shards)} raw bytes, the final has {src_bytes}")
    side_rows = []
    for sc in sidecars:
        sc = Path(sc)
        if not sc.is_file():
            raise SystemExit(f"REFUSED: sidecar {sc} does not exist")
        dest = out_dir / sc.name
        dest.write_bytes(sc.read_bytes())
        side_rows.append({"name": sc.name, "sha256": sha256_file(dest), "bytes": dest.stat().st_size})
    manifest = {
        "schema": 1,
        "corpus": corpus,
        "source_file": source.name,
        "release": release,
        "record_count": sum(s["records"] for s in shards),
        "sha256_reassembled": src_sha,
        "bytes_reassembled": src_bytes,
        "shards": shards,
        "official_tokens": official_tokens([out_dir / r["name"] for r in side_rows]),
        "sidecars": side_rows,
        "gzip": {"level": GZIP_LEVEL, "mtime": 0, "filename": "", "record_boundary": True,
                 "target_bytes": target, "hard_limit_bytes": hard, "measured_total_gzip_bytes": gz_total},
        "tool": "reproduce/tools/shard_corpus.py",
        "vendored_corpus_commit": vendored_commit,
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("corpus", help="constitution_advice | constitution_stories | control_advice | control_stories")
    ap.add_argument("--source", required=True, type=Path, help="the released JSONL final")
    ap.add_argument("--out", required=True, type=Path, help="corpora/<corpus>")
    ap.add_argument("--sidecar", action="append", default=[], type=Path, help="a sidecar file to commit beside the shards (repeatable)")
    ap.add_argument("--release", required=True, help="release label, e.g. final-production-20260831-v2")
    ap.add_argument("--vendored-commit", default=None, help="the corpus_generator commit vendored under corpus_generator/")
    ap.add_argument("--target-mb", type=int, default=TARGET_MB)
    ap.add_argument("--hard-mb", type=int, default=HARD_MB)
    a = ap.parse_args(argv)
    m = shard_corpus(a.corpus, a.source, a.out, sidecars=a.sidecar, release=a.release, target_mb=a.target_mb,
                     hard_mb=a.hard_mb, vendored_commit=a.vendored_commit)
    print(json.dumps({k: m[k] for k in ("corpus", "record_count", "sha256_reassembled", "bytes_reassembled")}
                     | {"n_shards": len(m["shards"]), "shard_bytes": [s["bytes"] for s in m["shards"]]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
