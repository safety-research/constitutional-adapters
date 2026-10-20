"""THE CHECKPOINT INVENTORY of a served checkpoint staged into the Modal volume.

A served checkpoint staged into the Modal volume's HF cache (``/vol/hf/hub/models--<org>--<name>/snapshots/<revision>``) is
trusted ONLY through an inventory built from the Hub's OWN per-file metadata at the pinned revision and verified against the
bytes on the volume — file by file, at staging AND again at every launch's CPU preflight — so a truncated, swapped or missing
shard refuses before a GPU container is paid for, never inside it.

What the Hub states per sibling (``HfApi.model_info(repo, revision=..., files_metadata=True)``):

* ``size`` — the byte count of every file;
* ``lfs.sha256`` — for a Git-LFS file (every weight shard, the tokenizer's large files): the SHA-256 of the file's bytes;
* ``blob_id`` — for a PLAIN git file (config.json, generation_config.json, the chat template, …): the git BLOB SHA-1, i.e.
  ``sha1(b"blob <size>\\0" + bytes)`` — NOT a sha256 of the contents (the two are different functions; a correct
  snapshot fails a sha256-vs-blob_id comparison by construction).

The inventory records, per file: ``size``, ``kind`` (``lfs`` | ``blob``) and the one digest the kind names, plus the repo, the
revision, the Hub's ``sha`` of the revision and when it was read. :func:`verify_snapshot` holds a snapshot directory to it: every
inventoried file present, its size equal, its digest equal; an extra file in the snapshot is listed (the snapshot may hold
``.gitattributes`` / README variants the pin never serves) and is NOT a refusal unless ``strict``. Every problem is a string
naming the file and the two values; an empty list is the only PASS. Stdlib only, no network — the Hub call is the caller's
(``scripts/modal/tinycap_prefetch_model.py``), so the launcher's CPU preflight can re-verify offline from the recorded inventory.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

INVENTORY_SCHEMA = "sisteer-hf-inventory/1"
#: ``<models--org--name>/.sisteer_inventory_<revision[:8]>.json`` — beside the ``snapshots/`` dir, never inside the snapshot
#: (the snapshot's file set is the Hub's; the inventory is this repo's verification record)
INVENTORY_BASENAME = ".sisteer_inventory_{rev8}.json"
KIND_LFS, KIND_BLOB = "lfs", "blob"
KINDS: Tuple[str, ...] = (KIND_LFS, KIND_BLOB)
REQUIRED_TOP: Tuple[str, ...] = ("schema", "repo_id", "revision", "hub_sha", "read_utc", "files")
REQUIRED_FILE: Tuple[str, ...] = ("size", "kind", "digest")
#: the files a serving snapshot must carry whatever the Hub lists (a snapshot without them cannot serve; the verifier refuses by name)
REQUIRED_SERVING_FILES: Tuple[str, ...] = ("config.json",)
_CHUNK = 1 << 20


class InventoryError(ValueError):
    """An inventory that cannot be read as one (shape, kinds, digests)."""


def git_blob_sha1(path: Path, size: Optional[int] = None) -> str:
    """The git blob object id of ``path``'s bytes: ``sha1(b"blob <size>\\0" + bytes)`` — what the Hub's ``blob_id`` is for a
    non-LFS file. Streamed, so a large plain file never loads whole."""
    p = Path(path)
    n = int(size) if size is not None else p.stat().st_size
    h = hashlib.sha1()
    h.update(f"blob {n}\0".encode("ascii"))
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def inventory_from_siblings(repo_id: str, revision: str, hub_sha: str, siblings: Iterable[Any], *,
                            read_utc: Optional[str] = None) -> Dict[str, Any]:
    """Build the inventory from the Hub's ``siblings`` (``RepoSibling`` objects or mappings with ``rfilename`` / ``size`` /
    ``lfs`` / ``blob_id``). Refuses (:class:`InventoryError`) a sibling without a size, an LFS sibling without a sha256, a
    plain sibling without a blob_id, or an empty listing — an inventory that cannot name a file's digest proves nothing."""
    files: Dict[str, Dict[str, Any]] = {}
    for sib in siblings:
        get = (lambda k, s=sib: s.get(k)) if isinstance(sib, Mapping) else (lambda k, s=sib: getattr(s, k, None))
        name = get("rfilename")
        if not name:
            raise InventoryError(f"{repo_id}@{revision[:8]}: a sibling without rfilename ({sib!r})")
        size = get("size")
        if size is None:
            raise InventoryError(f"{repo_id}@{revision[:8]}: {name}: the Hub listed no size (call model_info with files_metadata=True)")
        lfs = get("lfs")
        lfs_sha = None
        if lfs is not None:
            lfs_sha = lfs.get("sha256") if isinstance(lfs, Mapping) else getattr(lfs, "sha256", None)
            if not lfs_sha:
                raise InventoryError(f"{repo_id}@{revision[:8]}: {name}: an LFS file without a sha256 in its lfs block")
            files[str(name)] = {"size": int(size), "kind": KIND_LFS, "digest": str(lfs_sha).lower()}
        else:
            blob = get("blob_id")
            if not blob:
                raise InventoryError(f"{repo_id}@{revision[:8]}: {name}: a plain file without a blob_id")
            files[str(name)] = {"size": int(size), "kind": KIND_BLOB, "digest": str(blob).lower()}
    if not files:
        raise InventoryError(f"{repo_id}@{revision[:8]}: the Hub listed no files")
    missing = [f for f in REQUIRED_SERVING_FILES if f not in files]
    if missing:
        raise InventoryError(f"{repo_id}@{revision[:8]}: the listing lacks {missing} — not a serving snapshot")
    return {"schema": INVENTORY_SCHEMA, "repo_id": str(repo_id), "revision": str(revision), "hub_sha": str(hub_sha),
            "read_utc": read_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_files": len(files), "total_bytes": sum(int(v["size"]) for v in files.values()),
            "files": dict(sorted(files.items()))}


def inventory_problems(doc: Any) -> List[str]:
    """Why ``doc`` is not an inventory (empty = it is)."""
    if not isinstance(doc, Mapping):
        return ["the inventory is not a JSON object"]
    out = [f"missing {k!r}" for k in REQUIRED_TOP if k not in doc]
    if doc.get("schema") != INVENTORY_SCHEMA:
        out.append(f"schema {doc.get('schema')!r} != {INVENTORY_SCHEMA!r}")
    files = doc.get("files")
    if not isinstance(files, Mapping) or not files:
        return out + ["files is not a non-empty object"]
    for name, rec in files.items():
        if not isinstance(rec, Mapping) or any(k not in rec for k in REQUIRED_FILE):
            out.append(f"{name}: not {{size, kind, digest}}")
            continue
        if rec["kind"] not in KINDS:
            out.append(f"{name}: kind {rec['kind']!r} not in {list(KINDS)}")
        want = 64 if rec["kind"] == KIND_LFS else 40
        if not isinstance(rec["digest"], str) or len(rec["digest"]) != want or any(c not in "0123456789abcdef" for c in rec["digest"]):
            out.append(f"{name}: digest {rec['digest']!r} is not a {want}-hex {rec['kind']} digest")
        if isinstance(rec["size"], bool) or not isinstance(rec["size"], int) or rec["size"] < 0:
            out.append(f"{name}: size {rec['size']!r} is not a non-negative int")
    for f in REQUIRED_SERVING_FILES:
        if f not in files:
            out.append(f"the inventory lacks {f}")
    return out


def verify_snapshot(snapshot_dir: Path, inventory: Mapping[str, Any], *, strict: bool = False,
                    hash_bytes: bool = True) -> Dict[str, Any]:
    """Hold the snapshot directory to the inventory. Returns ``{ok, problems, verified, extra, bytes_hashed, seconds}``: ``problems``
    names every inventoried file that is absent, whose size differs, or whose digest differs (sha256 for ``lfs``, git blob SHA-1 for
    ``blob``); ``extra`` lists files in the snapshot the inventory does not name (a refusal only under ``strict``). ``hash_bytes=False``
    checks presence + size only (the GPU container's cheap re-check; the CPU preflight always hashes). Symlinks (the HF cache's
    snapshot layout points into ``blobs/``) are followed."""
    t0 = time.time()
    probs = inventory_problems(inventory)
    if probs:
        return {"ok": False, "problems": [f"inventory: {p}" for p in probs], "verified": [], "extra": [], "bytes_hashed": 0, "seconds": 0.0}
    root = Path(snapshot_dir)
    if not root.is_dir():
        return {"ok": False, "problems": [f"snapshot dir {root} is not a directory"], "verified": [], "extra": [], "bytes_hashed": 0, "seconds": 0.0}
    problems: List[str] = []
    verified: List[str] = []
    hashed = 0
    files: Mapping[str, Mapping[str, Any]] = inventory["files"]
    for name, rec in files.items():
        p = root / name
        if not p.is_file():
            problems.append(f"{name}: MISSING from {root}")
            continue
        size = p.stat().st_size
        if size != int(rec["size"]):
            problems.append(f"{name}: size {size} != the Hub's {rec['size']}")
            continue
        if hash_bytes:
            got = sha256_file(p) if rec["kind"] == KIND_LFS else git_blob_sha1(p, size)
            hashed += size
            if got != rec["digest"]:
                problems.append(f"{name}: {rec['kind']} digest {got[:16]}… != the Hub's {rec['digest'][:16]}…")
                continue
        verified.append(name)
    present = {str(q.relative_to(root)) for q in root.rglob("*") if q.is_file() or q.is_symlink()}
    extra = sorted(present - set(files))
    if strict and extra:
        problems.append(f"files in the snapshot the inventory does not name: {extra}")
    return {"ok": not problems, "problems": problems, "verified": verified, "extra": extra, "bytes_hashed": hashed,
            "seconds": round(time.time() - t0, 1), "hash_bytes": bool(hash_bytes)}


def inventory_path(repo_cache_dir: Path, revision: str) -> Path:
    """``<models--org--name>/.sisteer_inventory_<rev8>.json``."""
    return Path(repo_cache_dir) / INVENTORY_BASENAME.format(rev8=str(revision)[:8])


def repo_cache_dir(hf_home: Path, repo_id: str) -> Path:
    """``<HF_HOME>/hub/models--<org>--<name>``."""
    return Path(hf_home) / "hub" / f"models--{str(repo_id).replace('/', '--')}"


def snapshot_dir(hf_home: Path, repo_id: str, revision: str) -> Path:
    return repo_cache_dir(hf_home, repo_id) / "snapshots" / str(revision)


def write_inventory(path: Path, inventory: Mapping[str, Any]) -> str:
    """Write atomically (tmp + rename) through one serializer; returns the file's sha256."""
    probs = inventory_problems(inventory)
    if probs:
        raise InventoryError("; ".join(probs))
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(inventory, indent=1, sort_keys=True) + "\n").encode("utf-8")
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(body)
    os.replace(tmp, p)
    return hashlib.sha256(body).hexdigest()


def read_inventory(path: Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise InventoryError(f"no inventory at {p} — the checkpoint was not staged through tinycap_prefetch_model.py (or its verification never passed)")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise InventoryError(f"{p}: not JSON ({e})") from e
    probs = inventory_problems(doc)
    if probs:
        raise InventoryError(f"{p}: " + "; ".join(probs))
    return doc
