"""Loading a COMMITTED fitted object from the repo's ``objects/`` tree.

Layout (written by ``reproduce/tools/export_objects.py``):
``objects/<model_id>/<object_id>/`` holds either the ORIGINAL ``object.pt``
bytes (form ``whole``) or, for an artifact above the 40 MB chunk bound, the
ordered lossless chunks ``object.pt.part-NN`` plus ``CHUNKS.json`` (form
``chunked``; ``reproduce/lib/object_chunks.py``), beside the fit sidecars.
The directory name IS the registry ``object_id`` = ``sha256(object.pt)``
(``hooks.py``: ``object_id == artifact_sha256``), and ``objects/MANIFEST.json``
lists every row (``{object_id, model_id, form, artifact_sha256, bytes, chunks,
dir, sidecars, source, ...}``).

:func:`load_committed_object` is the ONE loader: reassemble when chunked
(through A's ``object_chunks.reassemble``, which verifies every chunk's sha
and the original's before it returns), else read ``<dir>/object.pt``; then
assert ``sha256(bytes) == object_id`` BEFORE ``torch.load`` — a committed
artifact whose bytes do not hash to its identity is refused by name, never
loaded.

An object the paper evaluates that this repository does not ship is listed
in ``objects/WITHHELD.json`` (beside the manifest) instead of committed:
asking for it, by its directory, a file in it, or ``object_id=``, raises
:class:`WithheldCommittedObjectError` with the reason the list records,
never a bare missing-directory error (:func:`refuse_if_withheld`, which
``sisteer.methods.load_object`` also consults for an absent path).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any, Optional

from ..objects import WithheldObjectError, read_withheld, withheld_message

OBJECT_FILE = "object.pt"
CHUNKS_MANIFEST = "CHUNKS.json"
OBJECTS_MANIFEST = "MANIFEST.json"
#: the paper objects this repository does not ship, beside the manifest (sisteer.objects.WITHHELD_REL)
WITHHELD_LIST = "WITHHELD.json"
#: A's reassembly lib, relative to a checkout root (a front-door lib, not a package: loaded by path).
_CHUNKS_LIB = Path("reproduce") / "lib" / "object_chunks.py"
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class CommittedObjectError(RuntimeError):
    """A committed object that cannot be loaded honestly: missing artifact,
    an identity the bytes do not hash to, chunks without the reassembly
    library, or an id the manifest disagrees with."""


class WithheldCommittedObjectError(CommittedObjectError, WithheldObjectError):
    """A request for an object ``objects/WITHHELD.json`` lists: the paper
    evaluates it, this repository does not ship it. Caught as either a
    :class:`CommittedObjectError` or a ``sisteer.objects.WithheldObjectError``."""


def withheld_row(path: os.PathLike | str, object_id: Optional[str] = None) -> Optional[dict]:
    """The ``objects/WITHHELD.json`` row of the object a path names, or None.
    ``path`` is an object directory ``objects/<model_id>/<object_id>`` or a
    file in one (``object.pt``, a chunk, a sidecar); it need not exist. The
    id is ``object_id`` when given, else the directory's name."""
    p = Path(path).resolve()
    d = p if p.is_dir() else p.parent if (p.is_file() or p.suffix) else p
    wfile = d.parent.parent / WITHHELD_LIST
    try:
        rows = read_withheld(wfile)
    except (OSError, ValueError) as e:
        raise CommittedObjectError(f"{wfile}: unreadable withheld list ({e})") from e
    return rows.get(object_id or d.name)


def refuse_if_withheld(path: os.PathLike | str, object_id: Optional[str] = None) -> None:
    """Raise :class:`WithheldCommittedObjectError`, naming the object and the
    recorded reason, when ``path`` (or ``object_id``) is a withheld object."""
    row = withheld_row(path, object_id)
    if row is not None:
        raise WithheldCommittedObjectError(withheld_message(row))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def repo_root_of(obj_dir: Path) -> Optional[Path]:
    """The checkout that contains ``obj_dir`` (walks up to a dir holding
    ``reproduce/lib/object_chunks.py``), or None."""
    for p in [obj_dir, *obj_dir.parents]:
        if (p / _CHUNKS_LIB).is_file():
            return p
    return None


def _object_chunks_module(obj_dir: Path, repo_root: Optional[Path]):
    """A's ``reproduce/lib/object_chunks.py`` loaded by path: from the explicit
    ``repo_root``, else the checkout holding the object dir, else this
    package's own checkout; refuses by name when none of them has it."""
    for root in (repo_root, repo_root_of(obj_dir), repo_root_of(Path(__file__).resolve())):
        if root is not None and (root / _CHUNKS_LIB).is_file():
            spec = importlib.util.spec_from_file_location("sisteer_object_chunks", root / _CHUNKS_LIB)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            return mod
    raise CommittedObjectError(
        f"{obj_dir}: carries {CHUNKS_MANIFEST} (a chunked artifact) but the "
        "reassembly library reproduce/lib/object_chunks.py is not present in any "
        "enclosing checkout — refusing to guess how the chunks recombine")


def default_reassemble_dest(obj_dir: Path) -> Path:
    """Where a chunked object's ORIGINAL bytes are rebuilt when the caller
    names no destination: ``<objects root>/_reassembled/<object_id>/object.pt``
    — the gitignored cache directly under the objects root (``objects/`` holds
    ``<model_id>/<object_id>/``, so the root is two levels up), never inside
    ``objects/<model_id>/``, which nothing ignores and the manifest census
    walks."""
    d = Path(obj_dir)
    return d.parent.parent / "_reassembled" / d.name / OBJECT_FILE


def manifest_row(obj_dir: Path) -> Optional[dict]:
    """The ``objects/MANIFEST.json`` row whose ``dir`` is this object's
    ``<model_id>/<object_id>``, or None when no manifest is present."""
    man = obj_dir.parent.parent / OBJECTS_MANIFEST
    if not man.is_file():
        return None
    try:
        doc = json.loads(man.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise CommittedObjectError(f"{man}: unreadable manifest ({e})") from e
    rel = f"{obj_dir.parent.name}/{obj_dir.name}"
    for row in doc.get("objects", []):
        if row.get("dir") == rel or row.get("object_id") == obj_dir.name:
            return row
    return None


def committed_object_bytes_path(obj_dir: os.PathLike | str, *, object_id: Optional[str] = None,
                                repo_root: Optional[os.PathLike | str] = None,
                                reassemble_dest: Optional[os.PathLike | str] = None,
                                ) -> tuple[Path, str]:
    """The path of the ORIGINAL ``object.pt`` bytes (reassembled when
    chunked) and the object_id they were verified against. Raises
    :class:`CommittedObjectError` on any identity mismatch."""
    d = Path(obj_dir).resolve()
    refuse_if_withheld(d, object_id)
    if not d.is_dir():
        raise CommittedObjectError(f"{d}: not a directory")
    row = manifest_row(d)
    expected = object_id
    if expected is None:
        if row is not None and row.get("object_id"):
            expected = str(row["object_id"])
        elif _SHA256_HEX.fullmatch(d.name):
            expected = d.name
        else:
            raise CommittedObjectError(
                f"{d}: cannot determine the object_id — the directory name is not a "
                "64-hex sha and no objects/MANIFEST.json row names it; pass object_id=")
    if row is not None and row.get("artifact_sha256") and row["artifact_sha256"] != expected:
        raise CommittedObjectError(
            f"{d}: objects/MANIFEST.json records artifact_sha256 {row['artifact_sha256'][:12]} "
            f"but the object_id in force is {expected[:12]} — the manifest and the identity disagree")
    if (d / CHUNKS_MANIFEST).is_file():
        chunks = _object_chunks_module(d, Path(repo_root) if repo_root else None)
        # the lib's own default (dest=None) would land INSIDE ``objects/<model_id>/``: always pass one
        dest = str(reassemble_dest if reassemble_dest is not None else default_reassemble_dest(d))
        try:
            path = Path(chunks.reassemble(str(d), dest=dest))
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            raise CommittedObjectError(f"{d}: reassembly refused ({e})") from e
    else:
        path = d / OBJECT_FILE
        if not path.is_file():
            raise CommittedObjectError(
                f"{d}: neither {OBJECT_FILE} nor {CHUNKS_MANIFEST} is present — not a committed object")
    got = _sha256_file(path)
    if got != expected:
        raise CommittedObjectError(
            f"{d}: sha256({path.name}) = {got[:12]}… != object_id {expected[:12]}… — the committed "
            "bytes do not hash to their registry identity; refusing to load them")
    return path, expected


def load_committed_object(obj_dir: os.PathLike | str, *, object_id: Optional[str] = None,
                          repo_root: Optional[os.PathLike | str] = None,
                          reassemble_dest: Optional[os.PathLike | str] = None,
                          weights_only: bool = False) -> Any:
    """Load a committed object's blob: reassemble via
    ``reproduce/lib/object_chunks.reassemble(dir)`` when ``CHUNKS.json`` is
    present (else ``<dir>/object.pt``), assert ``sha256(bytes) == object_id``
    (the dir name / the MANIFEST row / the explicit argument), then
    ``torch.load(path, map_location="cpu", weights_only=False)`` (the
    artifacts are this repo's own dicts of tensors + provenance;
    ``weights_only=True`` may be requested)."""
    import torch

    path, _ = committed_object_bytes_path(obj_dir, object_id=object_id, repo_root=repo_root,
                                          reassemble_dest=reassemble_dest)
    return torch.load(str(path), map_location="cpu", weights_only=weights_only)


__all__ = ["CommittedObjectError", "WithheldCommittedObjectError", "load_committed_object", "committed_object_bytes_path",
           "manifest_row", "withheld_row", "refuse_if_withheld", "repo_root_of", "default_reassemble_dest", "OBJECT_FILE",
           "CHUNKS_MANIFEST", "OBJECTS_MANIFEST", "WITHHELD_LIST"]
