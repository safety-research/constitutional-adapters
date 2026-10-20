"""``fit_meta.json`` <-> ``object.pt``: the OBJECT BINDING of a Phase B fit record.

A fit record pins every INPUT by hash (leakage report, screen, split, both prompt lists) and, until
this module, no hash of the object it describes — so nothing tied a record to the bytes beside it and
a stale object/record pair was undetectable. The first ladder submission compared ``fit_meta.sha256``
(the RECORD's fingerprint, ``alphasteer_screen.write_fingerprinted_json``) to the object's hash: two
different quantities that can never match, on any object. The two checks are:

  fit_meta.sha256          the record's own fingerprint — proves the record is intact (and its
                           ``.COMPLETE`` marker proves the publication finished);
  fit_meta.object_sha256   THE BINDING — sha256 of the object file on disk, written by the producer
                           after the save (``object_sha256_source`` starts with ``producer:``) or
                           backfilled in place by the CLI here (source starts with ``backfilled``,
                           never to be read as producer-written provenance).

THE RULE FOR CONSUMERS (``verify_object_binding``): a record WITHOUT the field is not "probably fine",
it is UNVERIFIABLE — raise ``ObjectBindingUnverifiable`` saying freshness cannot be verified and naming
the remedy; never infer the tie from content cross-checks (layers, hidden_dim: a re-fit over the same
layers passes them with different matrices) and never proceed on an unverified parent. A field that is
present but malformed, source-less, or different from the bytes on disk is ``ObjectBindingInvalid``
(the last case IS the stale pair the field exists to catch).

PURE STDLIB, NO TORCH, NO INTRA-PACKAGE IMPORTS — on purpose: ``scripts/backfill_fit_meta_object_sha256.py``
loads this file by path on a cluster login shell's bare ``python3`` (no venv, no PYTHONPATH, and immune
to anything ``sisteer/__init__`` may one day import), and the Phase C bake on the ladder branch is
stdlib+torch by design and can adopt it the same way.
``fingerprint`` replicates ``alphasteer_screen.canonical_json``'s convention (``sort_keys=True,
indent=1``) for JSON-native payloads; ``tests/test_fit_class_b.py`` pins the two together.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

#: what ``object_sha256_source`` says when the PRODUCER computed the binding
OBJECT_SHA256_PRODUCER_SOURCE = ("producer: fit_class_b._save_object hashed the object file on disk "
                                 "after obj.save and before fit_meta.json was written")
#: the first word of every backfilled source string
BACKFILL_SOURCE_PREFIX = "backfilled"
BACKFILL_CLI = "scripts/backfill_fit_meta_object_sha256.py"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ObjectBindingUnverifiable(ValueError):
    """The record carries no ``object_sha256``: freshness CANNOT be verified. Consumers refuse."""


class ObjectBindingInvalid(ValueError):
    """``object_sha256`` is present but malformed, source-less, or names other bytes (a stale pair)."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(body: Dict[str, Any]) -> str:
    """``alphasteer_screen.canonical_json`` for JSON-native values (no ``default=``: a non-native value
    is a loud TypeError here, never a silently different serialisation)."""
    return json.dumps(body, sort_keys=True, indent=1)


def fingerprint(payload: Dict[str, Any]) -> str:
    """``fit_meta.sha256``'s convention: sha256 over the canonical JSON of everything BUT ``sha256``."""
    body = {k: v for k, v in payload.items() if k != "sha256"}
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


def fingerprint_ok(payload: Dict[str, Any]) -> bool:
    want = payload.get("sha256")
    return bool(want) and fingerprint(payload) == want


def is_backfilled(source: str) -> bool:
    return source.split(" ", 1)[0] == BACKFILL_SOURCE_PREFIX


def verify_object_binding(fit_meta: Dict[str, Any], object_path: Path, *, what: str = "") -> Dict[str, Any]:
    """The one check that ties a fit record to its object. Returns
    ``{"object_sha256", "source", "backfilled"}`` when ``fit_meta.object_sha256`` equals the sha256 of
    ``object_path``'s bytes and a non-empty ``object_sha256_source`` says where it came from.

    Raises ``ObjectBindingUnverifiable`` when the field is ABSENT — the message says freshness CANNOT
    be verified, that the tie is not inferred from content, and names the remedy — and
    ``ObjectBindingInvalid`` when it is present but not a 64-hex sha, has no source, or differs from
    the bytes (a stale object/record pair). ``what`` prefixes every message (e.g. the target)."""
    tag = f"{what}: " if what else ""
    object_path = Path(object_path)
    if not object_path.is_file():
        raise ObjectBindingInvalid(f"{tag}{object_path} is not a file — there is no object to bind the record to")
    on_disk = sha256_file(object_path)
    rec = fit_meta.get("object_sha256")
    if rec is None:
        raise ObjectBindingUnverifiable(
            f"{tag}fit_meta.json records no object_sha256 — CANNOT VERIFY freshness: nothing in the record binds "
            f"it to {object_path.name}'s bytes ({on_disk[:12]}), so a stale object/record pair would be "
            "undetectable. Refusing rather than proceeding unverified; the tie is NOT inferred from content "
            "cross-checks (layers, hidden_dim). Remedy: a fit written by fit_class_b with the field, or "
            f"{BACKFILL_CLI}, which hashes the object in place and MARKS the value as backfilled "
            "(object_sha256_source)")
    if not isinstance(rec, str) or not _HEX64.match(rec):
        raise ObjectBindingInvalid(f"{tag}fit_meta.object_sha256 {rec!r} is not a 64-hex sha256")
    if rec != on_disk:
        raise ObjectBindingInvalid(f"{tag}fit_meta.object_sha256 {rec[:12]} != {object_path.name} sha {on_disk[:12]} "
                                   "— a stale object/record pair; refusing")
    source = fit_meta.get("object_sha256_source")
    if not isinstance(source, str) or not source.strip():
        raise ObjectBindingInvalid(f"{tag}fit_meta.object_sha256 is present but object_sha256_source is absent or "
                                   "empty — the record must say whether the producer wrote the binding or it was "
                                   "backfilled after the fact (a backfilled hash is not producer-written provenance)")
    return {"object_sha256": on_disk, "source": source, "backfilled": is_backfilled(source)}


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def backfill_object_sha256(cell_dir: Path, *, date: str, tool: str = BACKFILL_CLI,
                           tool_commit: Optional[str] = None, dry_run: bool = False,
                           fit_meta_name: str = "fit_meta.json", object_name: str = "object.pt",
                           what: str = "") -> Dict[str, Any]:
    """Bind an existing record to the object beside it, IN PLACE, and say so.

    Writes exactly: ``object_sha256`` (the bytes on disk, hashed here), ``object_sha256_source``
    (``"backfilled <date> by <tool>: ..."`` — never the producer's wording), ``object_sha256_backfill``
    ({date, tool, tool_commit, object_path, previous_sha256 = the record's fingerprint BEFORE the edit,
    note}), and a recomputed ``sha256`` repeated in ``<fit_meta>.COMPLETE``. ``code_sha`` and
    ``written_at`` are the producer's and are not touched. Refuses (``ValueError``) a torn record (bad
    fingerprint or marker — never edited over), a record already bound to DIFFERENT bytes (the stale
    pair; not overwritten), or a cell with any of the three files missing. A record already bound to
    these bytes is left alone byte for byte (``status: already_bound``). ``dry_run`` writes nothing."""
    tag = f"{what}: " if what else ""
    cell_dir = Path(cell_dir)
    obj, fm_path = cell_dir / object_name, cell_dir / fit_meta_name
    marker = fm_path.with_name(fm_path.name + ".COMPLETE")
    for p in (obj, fm_path, marker):
        if not p.is_file():
            raise ValueError(f"{tag}{p} missing — not a complete fit publication; refusing")
    fm = json.loads(fm_path.read_text(encoding="utf-8"))
    if not fingerprint_ok(fm):
        raise ValueError(f"{tag}fit_meta.sha256 {str(fm.get('sha256'))[:12]} is not the fingerprint of the record's "
                         "own body — an edited or torn record is never backfilled over; refusing")
    if marker.read_text(encoding="utf-8").strip() != fm["sha256"]:
        raise ValueError(f"{tag}{marker.name} does not repeat fit_meta.sha256 — torn publication; refusing")
    on_disk = sha256_file(obj)
    if "object_sha256" in fm:
        if fm["object_sha256"] == on_disk:
            return {"status": "already_bound", "object_sha256": on_disk, "sha256": fm["sha256"],
                    "object_sha256_source": fm.get("object_sha256_source"), "fit_meta": str(fm_path)}
        raise ValueError(f"{tag}fit_meta.object_sha256 {str(fm['object_sha256'])[:12]} != {object_name} sha "
                         f"{on_disk[:12]} — a stale object/record pair, exactly what the field exists to catch; "
                         "NOT overwriting it")
    previous = fm["sha256"]
    body = {k: v for k, v in fm.items() if k != "sha256"}
    body["object_sha256"] = on_disk
    body["object_sha256_source"] = (f"{BACKFILL_SOURCE_PREFIX} {date} by {tool}: hashed {object_name} in place; the fit "
                                    f"(fit_class_b @ code_sha {fm.get('code_sha')}) predates the field — NOT "
                                    "producer-written provenance")
    body["object_sha256_backfill"] = {
        "date": date, "tool": tool, "tool_commit": tool_commit, "object_path": object_name, "previous_sha256": previous,
        "note": "object_sha256 added after the fact from the bytes on disk; the record's fingerprint (sha256) and "
                "its .COMPLETE marker were recomputed over the new body; code_sha and written_at are the producer's",
    }
    body["sha256"] = fingerprint(body)
    out = {"status": "would_backfill" if dry_run else "backfilled", "object_sha256": on_disk,
           "previous_sha256": previous, "sha256": body["sha256"], "fit_meta": str(fm_path)}
    if dry_run:
        return out
    _write_atomic(fm_path, canonical(body))
    _write_atomic(marker, body["sha256"] + "\n")
    back = json.loads(fm_path.read_text(encoding="utf-8"))      # read back under the consumer's own reading
    if not fingerprint_ok(back) or marker.read_text(encoding="utf-8").strip() != back["sha256"]:
        raise ValueError(f"{tag}the rewritten record does not verify on read-back")
    verify_object_binding(back, obj, what=what)
    return out
