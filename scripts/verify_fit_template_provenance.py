#!/usr/bin/env python
"""Verify — never repair — the fit-time chat-template provenance of every
staged object against the tokenizer its record pins.

For each ``<bank_root>/<target>/cells/<staged_id>/`` the object's fit-time
provenance is resolved exactly as the served drivers resolve it
(``sweep.resolve_training_mirror``: train_budget_meta.json /
fit_provenance.json / derived_meta.json), the tokenizer the record pins
(``model_id`` @ ``model_revision``) is loaded through the ONE shared
construction (``sisteer.models.load_tokenizer``), and its template hash is
RECOMPUTED with the writer's own convention (``tokcache.template_hash_of``:
sha256 over the raw chat_template string or "" when the tokenizer ships
none). Per cell the verdict is

  agree       recorded == recomputed: the record is a faithful measurement
              of the pinned tokenizer (a template-free tokenizer records
              sha256("") = e3b0c44298fc1c14 — a measured absence, not a gap)
  disagree    recorded != recomputed: a REAL finding — the record does not
              describe the pinned tokenizer; report it, never overwrite
  unrecorded  the record carries no hash at all (provenance_status partial)
  unresolved  the object's provenance does not resolve (refused / no record)

This tool WRITES NOTHING under the bank: it verifies its own no-write
property (a digest of every path + size + mtime under the bank is taken
before and after and must be identical) and refuses an --out path inside
the bank. It exists because the full22 controls leg was held on the belief
that the gemma sdf-base objects "record no trained template hash"; they
record e3b0c44298fc1c14, the hash of the empty template, which is what the
pinned google/gemma-4-E4B tokenizer measures. Nothing was missing, and any
back-fill would have DISAGREED with the record — this tool is the honest
analogue of a back-fill: it says, per cell, whether the record stands.

Exit status: 0 = every cell agrees; 2 = any disagree / unresolved /
unrecorded (the report names them).

Usage (cluster, offline against the shared HF cache; the bank layout is
<bank_root>/<target>/cells/<staged_id>/, the full22 staging layout)::

  PYTHONPATH=src HF_HUB_OFFLINE=1 python scripts/verify_fit_template_provenance.py \\
      --bank-root $SHARED/exp/objects/sdfbase/full22 --out /tmp/f22_template_provenance.json

``--tokenizer-dir MODEL_ID@REV=PATH`` (repeatable) substitutes a local
tokenizer directory for a pin — for an offline run away from the cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

VERDICTS = ("agree", "disagree", "unrecorded", "unresolved")


def tree_digest(root: Path) -> str:
    """sha256 over (relative path, size, mtime_ns) of every file under
    ``root`` — the no-write self-check."""
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            try:
                st = p.lstat()
            except FileNotFoundError:
                continue
            h.update(f"{p.relative_to(root)}\0{st.st_size}\0{st.st_mtime_ns}\n"
                     .encode("utf-8", "surrogateescape"))
    return h.hexdigest()


def _load_tok(model_id: str, revision, overrides: dict):
    from sisteer.models import load_tokenizer

    local = overrides.get(f"{model_id}@{revision}") or overrides.get(model_id)
    return load_tokenizer(local if local else model_id,
                          None if local else revision)


def recomputed_hash(tok) -> tuple[str, bool]:
    """(template_hash_of(tok), tokenizer ships a chat template)."""
    from sisteer.extraction.tokcache import template_hash_of

    return template_hash_of(tok), bool(getattr(tok, "chat_template", None))


def verify_cell(target: str, staged_id: str, cell_dir: Path, *, tok_cache: dict,
                overrides: dict) -> dict:
    from sisteer.eval import sweep

    row = {"target": target, "staged_id": staged_id, "cell_dir": str(cell_dir)}
    obj = cell_dir / "object.pt"
    if not obj.is_file():
        others = [f for f in sorted(cell_dir.iterdir()) if f.name.startswith("object.")]
        obj = others[0] if others else obj
    try:
        prov = sweep.resolve_training_mirror(staged_id, obj)
    except Exception as e:  # noqa: BLE001 — every refusal is a row
        row.update({"verdict": "unresolved", "error": f"{type(e).__name__}: {e}"})
        return row
    row.update({"provenance_status": prov.status, "source": prov.source,
                "fit_kind": prov.fit_kind,
                "model_id": prov.model_id, "model_revision": prov.revision,
                "recorded_template_hash": prov.trained_template_hash,
                "training_budget_kind": prov.training_budget_kind,
                "chat_template_bypassed": prov.chat_template_bypassed,
                "positions_at_fit": prov.positions_at_fit})
    if prov.trained_template_hash is None:
        row["verdict"] = "unrecorded"
        return row
    key = (prov.model_id, prov.revision)
    if key not in tok_cache:
        try:
            tok_cache[key] = recomputed_hash(_load_tok(prov.model_id, prov.revision,
                                                       overrides))
        except Exception as e:  # noqa: BLE001
            tok_cache[key] = e
    got = tok_cache[key]
    if isinstance(got, Exception):
        row.update({"verdict": "unresolved",
                    "error": f"tokenizer {prov.model_id}@{prov.revision}: "
                             f"{type(got).__name__}: {got}"})
        return row
    recomputed, ships = got
    row.update({"recomputed_template_hash": recomputed,
                "pinned_tokenizer_ships_chat_template": ships,
                "verdict": ("agree" if recomputed == prov.trained_template_hash
                            else "disagree")})
    return row


def verify_bank(bank_root: Path, *, targets=None, overrides: dict | None = None
                ) -> dict:
    overrides = dict(overrides or {})
    before = tree_digest(bank_root)
    rows: list[dict] = []
    tok_cache: dict = {}
    for tgt_dir in sorted(p for p in bank_root.iterdir() if p.is_dir()):
        if targets and tgt_dir.name not in targets:
            continue
        cells = tgt_dir / "cells"
        if not cells.is_dir():
            continue
        for cell_dir in sorted(p for p in cells.iterdir() if p.is_dir()):
            rows.append(verify_cell(tgt_dir.name, cell_dir.name, cell_dir,
                                    tok_cache=tok_cache, overrides=overrides))
    after = tree_digest(bank_root)
    if before != after:
        raise SystemExit(
            f"[verify-template-provenance] the bank tree under {bank_root} "
            "CHANGED during verification — this tool must write nothing there; "
            "refusing to report over a tree it may have altered")
    counts = {v: sum(1 for r in rows if r["verdict"] == v) for v in VERDICTS}
    return {"bank_root": str(bank_root), "n_cells": len(rows), "counts": counts,
            "bank_tree_digest": after,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "rows": rows}


def _parse_overrides(items) -> dict:
    out = {}
    for it in items or ():
        spec, sep, path = it.partition("=")
        if not sep or not spec or not path:
            raise SystemExit(f"--tokenizer-dir wants MODEL_ID@REV=PATH, got {it!r}")
        out[spec] = path
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bank-root", required=True, type=Path)
    ap.add_argument("--targets", default="",
                    help="comma-separated target dirs (default: every dir)")
    ap.add_argument("--tokenizer-dir", action="append", default=[],
                    metavar="MODEL_ID@REV=PATH")
    ap.add_argument("--out", type=Path, default=None,
                    help="JSON report path (must NOT be under --bank-root)")
    args = ap.parse_args(argv)
    bank = args.bank_root.resolve()
    if not bank.is_dir():
        raise SystemExit(f"--bank-root {bank} is not a directory")
    if args.out is not None:
        out = args.out.resolve()
        if out == bank or bank in out.parents:
            raise SystemExit(f"--out {out} lies under the bank {bank}; this tool "
                             "writes nothing there — pick a path outside it")
    targets = {t for t in args.targets.split(",") if t} or None
    report = verify_bank(bank, targets=targets,
                         overrides=_parse_overrides(args.tokenizer_dir))
    for r in report["rows"]:
        print(f"{r['verdict']:10s} {r['target']:16s} {r['staged_id']:60s} "
              f"{str(r.get('provenance_status')):13s} "
              f"recorded={r.get('recorded_template_hash')} "
              f"recomputed={r.get('recomputed_template_hash')} "
              f"{r.get('model_id')}@{str(r.get('model_revision'))[:8]}"
              + (f"  {r['error'][:120]}" if r.get("error") else ""))
    print(f"[verify-template-provenance] {report['n_cells']} cells: "
          + ", ".join(f"{k}={v}" for k, v in report["counts"].items()))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_name(args.out.name + f".tmp-{os.getpid()}")
        tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, args.out)
        print(f"[verify-template-provenance] report -> {args.out}")
    bad = sum(v for k, v in report["counts"].items() if k != "agree")
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
