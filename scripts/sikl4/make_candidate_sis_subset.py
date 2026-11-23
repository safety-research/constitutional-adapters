#!/usr/bin/env python3
"""Derive a sha-pinned SUBSET of the candidate-SI snapshot (the C16-only file the Modal C16 cells load).

The Modal C16 cells resolve the short irrelevant control SI C16 through the candidate-SI loader, but the full snapshot (60 entries, B1
the whole constitution) must not go on a volume every member of the workspace can read. So the cells load a SUBSET file holding only
C16, in the loader's subset mode (``sisteer.data.candidate_sis``, entered by ``SISTEER_CANDIDATE_SIS_SUBSET_SHA256``). This script is
the only way that file is made: never by hand.

It reads the full snapshot, checks its file sha256 against ``--full-sha`` AND against the loader's pin
``candidate_sis.CANDIDATE_SIS_FULL_SHA256``, loads it through the normal 60-entry path (every check of the full loader runs), copies
each requested entry verbatim (every field, the text byte for byte), and writes::

    {"meta": {"subset_of_sha256": <the full snapshot's file sha256>, "ids": [...], "text_sha256": {<id>: <sha256 of its text>}},
     "system_instructions": [<the entries>]}

as indented UTF-8 JSON, so the same snapshot always yields the same bytes and hence the same pin. Before the file is put in place it
is re-read through the subset mode with its own sha, and each Instruction it yields must equal the full snapshot's field by field.

Refuses, by name: a full snapshot whose sha differs from either pin; an id with no pinned text sha in
``candidate_sis.SUBSET_TEXT_SHA256`` (today C16 only); ``SISTEER_CANDIDATE_SIS_SUBSET_SHA256`` set in the environment (the full
snapshot must load through the full path); an ``--out`` inside this checkout, or inside ANY git work tree (the base checkout, a
sibling worktree, another repository: neither the snapshot nor any of its texts is committed anywhere); an existing ``--out`` whose
bytes differ (a pinned subset never changes silently).

Prints one JSON object: the file, its sha256 (the value the Modal rows pin as ``SISTEER_CANDIDATE_SIS_SUBSET_SHA256``), the meta,
and the two variables. Every process that builds a cell registry from a config naming one of the ids sets both, not only the C16
cells' own containers: the explicit-cells registry resolves every listed cell and refuses the whole config when C16 does not
resolve (``candidate_sis``'s module docstring).

    python scripts/sikl4/make_candidate_sis_subset.py --full <snapshot> --full-sha <64 hex> --ids C16 --out <file>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# this checkout's package, never whichever copy a shared venv's editable install points at
sys.path.insert(0, str(REPO / "src"))

from sisteer.data import candidate_sis as C  # noqa: E402

LOAD_ENV = "SISTEER_LOAD_CANDIDATE_SIS"


class Refusal(RuntimeError):
    """A named refusal: the subset was not written."""


def _parse_ids(spec: str) -> list[str]:
    ids = [s.strip() for s in spec.split(",") if s.strip()]
    if not ids:
        raise Refusal("--ids names no id")
    if len(set(ids)) != len(ids):
        raise Refusal(f"--ids {ids} repeats an id")
    unpinned = [i for i in ids if i not in C.SUBSET_TEXT_SHA256]
    if unpinned:
        raise Refusal(f"--ids {unpinned} have no pinned text sha256 in candidate_sis.SUBSET_TEXT_SHA256 "
                      f"(pinned: {sorted(C.SUBSET_TEXT_SHA256)}); the subset mode would refuse them")
    return ids


def enclosing_work_tree(path: Path) -> Path | None:
    """The top of the git work tree ``path`` would sit in (the nearest ancestor holding a ``.git`` directory, or the ``.git``
    FILE of a linked worktree or submodule), else None. Walks the ancestors, so a not-yet-created ``--out`` directory counts."""
    for anc in path.parents:
        if (anc / ".git").exists():
            return anc
    return None


def subset_doc(full_raw: bytes, ids: list[str]) -> dict:
    """The subset document for ``ids``, its entries copied verbatim out of the full snapshot's bytes ``full_raw``."""
    doc = json.loads(full_raw)
    entries = doc["system_instructions"] if isinstance(doc, dict) else doc   # the loader's two accepted shapes
    by_id = {e["id"]: e for e in entries}
    picked = [by_id[i] for i in ids]
    text_sha = {e["id"]: hashlib.sha256(e["text"].encode("utf-8")).hexdigest() for e in picked}
    return {"meta": {"subset_of_sha256": hashlib.sha256(full_raw).hexdigest(), "ids": list(ids), "text_sha256": text_sha},
            "system_instructions": picked}


def encode_subset(doc: dict) -> bytes:
    """A subset document's file bytes: indented UTF-8 JSON, so the same document always yields the same bytes and pin."""
    return (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def derive(full: Path, full_sha: str, ids: list[str], out: Path) -> dict:
    if os.environ.get(C.SUBSET_SHA_ENV):
        raise Refusal(f"{C.SUBSET_SHA_ENV} is set in the environment: the full snapshot must load through the full 60-entry "
                      "path, so unset it to derive a subset")
    if full_sha != C.CANDIDATE_SIS_FULL_SHA256:
        raise Refusal(f"--full-sha {full_sha} is not the loader's pinned full snapshot {C.CANDIDATE_SIS_FULL_SHA256} "
                      "(candidate_sis.CANDIDATE_SIS_FULL_SHA256): the subset mode would refuse the result")
    out_abs, full_abs = out.expanduser().resolve(), full.expanduser().resolve()
    if out_abs.is_relative_to(REPO):   # also covers a checkout exported without .git (a git archive, a gate tree)
        raise Refusal(f"--out {out_abs} is inside the repository {REPO}: the candidate-SI texts are never committed; "
                      "write the subset outside the checkout")
    work_tree = enclosing_work_tree(out_abs)
    if work_tree is not None:
        raise Refusal(f"--out {out_abs} is inside the git work tree {work_tree}: the candidate-SI texts are never committed, "
                      "to this repository or any other; write the subset outside every checkout")
    if out_abs == full_abs:
        raise Refusal("--out names the full snapshot itself")

    full_raw = full_abs.read_bytes()
    got = hashlib.sha256(full_raw).hexdigest()
    if got != full_sha:
        raise Refusal(f"the full snapshot {full_abs} hashes to {got}, not --full-sha {full_sha}")
    # every check of the full loader, on the file whose bytes were just hashed (it re-reads them and re-checks the sha)
    full_instrs = {i.id: i for i in C.load_candidate_instructions(str(full_abs), expect_sha=full_sha)}

    doc = subset_doc(full_raw, ids)
    meta = doc["meta"]
    for i in ids:
        if meta["text_sha256"][i] != C.SUBSET_TEXT_SHA256[i]:
            raise Refusal(f"{i} text in {full_abs} hashes to {meta['text_sha256'][i]}, not the pinned "
                          f"{C.SUBSET_TEXT_SHA256[i]}")
    blob = encode_subset(doc)
    sha = hashlib.sha256(blob).hexdigest()

    previous = None
    if out_abs.exists():
        previous = hashlib.sha256(out_abs.read_bytes()).hexdigest()
        if previous != sha:
            raise Refusal(f"--out {out_abs} exists with sha256 {previous}, not the derived {sha}: a pinned subset never "
                          "changes silently; move it aside to re-derive")

    out_abs.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{out_abs.name}.", dir=out_abs.parent)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(blob)
        # the written file, re-read through the subset mode under its own pin: what a C16 cell will load
        sub = C.load_candidate_subset(tmp, subset_sha256=sha)
        got_ids = [s.id for s in sub]
        if got_ids != ids:
            raise Refusal(f"the subset mode returned ids {got_ids}, not {ids}")
        for s in sub:
            if s != full_instrs[s.id]:   # Instruction is a frozen dataclass: == compares every field
                raise Refusal(f"{s.id}: the Instruction the subset yields differs from the full snapshot's")
        os.chmod(tmp, 0o644)
        os.replace(tmp, out_abs)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return {"out": str(out_abs), "sha256": sha, "bytes": len(blob), "unchanged": previous == sha, "meta": meta,
            "env": {LOAD_ENV: str(out_abs), C.SUBSET_SHA_ENV: sha}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--full", required=True, type=Path, help="the full candidate-SI snapshot (60 entries)")
    ap.add_argument("--full-sha", required=True, help="the full snapshot's file sha256 (64 hex)")
    ap.add_argument("--ids", required=True, help="comma-separated ids to keep (today: C16)")
    ap.add_argument("--out", required=True, type=Path, help="the subset file to write (outside the repository)")
    a = ap.parse_args(argv)
    try:
        report = derive(a.full, a.full_sha.strip().lower(), _parse_ids(a.ids), a.out)
    except Refusal as exc:
        print(f"make_candidate_sis_subset: REFUSED: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
