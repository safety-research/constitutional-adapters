"""Backfill orchestration: adapters → records → a fresh database, indexes deferred, per-input
watermarks as the resumable ledger, free-space refusal below 2× the projected write.

This module runs whatever adapters are registered (or passed in), using
``adapters.runner.run_pass(full=True)`` for the record emission and the harvester for the writes.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Mapping

from . import db as DB
from . import harvest as H
from . import records as R
from .adapters import load_all
from .adapters.base import Adapter
from .adapters.runner import PassReport, run_pass

GB = 1024 ** 3


class InsufficientSpace(RuntimeError):
    pass


@dataclass
class BackfillReport:
    into: str
    projected_write_bytes: int
    free_bytes_before: int
    free_bytes_after: int = 0
    adapters: List[str] = field(default_factory=list)
    pass_report: Optional[PassReport] = None
    ingest_report: Optional[H.IngestReport] = None
    parse_s: float = 0.0
    write_s: float = 0.0
    wall_s: float = 0.0
    longest_transaction_s: float = 0.0
    rows: Dict[str, int] = field(default_factory=dict)


def project_write_bytes(adapters: Sequence[Adapter], bytes_per_input_byte: float = 1.5, floor: int = 256 * 1024 ** 2) -> int:
    """A coarse projection: the registry writes ≈ 1.5 × the input bytes it reads (records + index)."""
    total = 0
    for a in adapters:
        try:
            for inp in a.discover():
                total += inp.size
        except Exception:
            continue
    return max(int(total * bytes_per_input_byte), floor)


def refuse_if_low_space(root: str, projected: int, factor: float = 2.0) -> int:
    free = DB.free_bytes(root if os.path.exists(root) else os.path.dirname(root) or ".")
    if free < factor * projected:
        raise InsufficientSpace(
            f"backfill refuses to start: free {free / GB:.2f} GB < {factor:g} × projected write {projected / GB:.2f} GB")
    return free


def backfill(root: str, into: Optional[str] = None, adapters: Optional[Sequence[Adapter]] = None, resume: bool = False,
             workers: int = 3, check_space: bool = True, legacy_spellings: Optional[Mapping[str, Mapping[str, str]]] = None) -> BackfillReport:
    """Run every adapter in full mode into ``into`` (default: the root's live DB, which must be new
    unless ``resume``); indexes dropped for the bulk write and recreated at the end. ``legacy_spellings`` is handed
    to the pass as ``AdapterContext.legacy_spellings`` (``adapters.dimensions.adapter_legacy_spellings`` builds it
    from the seeds)."""
    t0 = time.monotonic()
    H.ensure_layout(root)
    into = into or DB.db_path(root)
    # every adapter module on disk, both registration conventions (a bare `registered()` here saw only what some
    # earlier import happened to register — PR-3b/3d's adapters were skipped that way)
    ads = list(adapters) if adapters is not None else [f() for f in load_all().values()]
    projected = project_write_bytes(ads)
    free_before = refuse_if_low_space(root, projected) if check_space else DB.free_bytes(root)
    rep = BackfillReport(into, projected, free_before, adapters=[a.name for a in ads])
    if os.path.exists(into) and not resume:
        raise FileExistsError(f"backfill: {into} exists; pass resume=True to continue into it")
    conn = DB.open_or_create(into)
    try:
        # records first (adapters write only files); the watermark table is the resume ledger
        rep.pass_report = run_pass(root, ads, conn=conn if resume else None, full=not resume, legacy_spellings=legacy_spellings)
        index_sql = DB.drop_indexes(conn)
        try:
            rep.ingest_report = H.ingest_once(root, conn, workers=workers, derive_facts=True, min_free=0)
        finally:
            with DB.transaction(conn):
                DB.recreate_indexes(conn, index_sql)
        rep.parse_s = rep.ingest_report.parse_s
        rep.write_s = rep.ingest_report.write_s
        rep.longest_transaction_s = rep.ingest_report.longest_transaction_s
        for t in R.ALL_TABLES:
            rep.rows[t] = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    finally:
        conn.close()
    rep.free_bytes_after = DB.free_bytes(root)
    rep.wall_s = time.monotonic() - t0
    return rep


def finish(conn) -> None:
    """Recreate indexes and ANALYZE after a bulk write (kept as a separate entry point for the CLI)."""
    conn.execute("ANALYZE")
