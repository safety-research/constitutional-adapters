"""The harvester cycle's transaction structure, pinned (2026-09-24 profile).

The fold-time profile of 2026-09-24 (specs: results-registry/fold-perf-profile.md) tested the hypothesis that a cycle's cost
was its many small commits, each paying a WAL append and a sync: a 2,011 s cycle took 161 commits and 38 sync calls (1.5 s).
The commit boundary is therefore NOT a performance lever, and it IS a correctness surface -- a staging window that commits is
what an interrupted fold resumes from, the activation is the one transaction a reader's revision switch rides
(``test_registry_revision_switch_atomic``), a dimension bundle is one transaction (``test_registry_dimensions_atomic``).
This test pins the count and kinds of write transactions for a fixture of N revision directories, so a change to the
structure -- a coalesced staging pass, a per-file commit, a second activation -- fails loudly and is made on purpose.
"""
from __future__ import annotations

import json
import math
import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import runner as RUN

from test_registry_core_fixes import StoreAdapter, _adapter_setup, _ingest

N_STORES = 3               # three comparison groups -> three revision directories in one pass
ITEMS_PER_STORE = 250      # item rows per store (item_result only from the synthetic store), above the window below
WINDOW_ROWS = 200          # a staging window: 250 rows stage in two


def _scores(seed, n=ITEMS_PER_STORE):
    return [((i * 37 + seed * 11) % 100) / 100.0 for i in range(n)]


def _metrics(root, now):
    with open(os.path.join(root, "state", "metrics.jsonl"), encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return [r for r in rows if r.get("utc") == now and "rows" in r]


def test_a_cycle_with_n_revisions_takes_the_pinned_transactions(tmp_path, monkeypatch):
    monkeypatch.setattr(H, "ITEM_ROWS_PER_TRANSACTION", WINDOW_ROWS)
    stores = [(f"s{i}", f"store_{i}", "none" if i else "constitution_lora@aaaa", _scores(i + 1)) for i in range(N_STORES)]
    root, store_root = _adapter_setup(tmp_path, stores)
    ad = StoreAdapter(store_root)                         # one group per store directory
    with DB.reading(DB.db_path(root)) as conn:
        rep = RUN.run_pass(root, [ad], conn=conn, now=synth.t(2000))
    assert rep.groups_refreshed == N_STORES and all(r.accepted for r in rep.results)
    rev_dirs = sorted(d for d in os.listdir(os.path.join(root, "records", "adapters", ad.name)) if not d.startswith("_"))
    assert len(rev_dirs) == N_STORES
    files_per_rev = {d: sorted(os.listdir(os.path.join(root, "records", "adapters", ad.name, d))) for d in rev_dirs}

    def record_type(f):
        return H.Ingestor._file_record_type(f)

    def n_records(d, f):
        return sum(1 for _ in R.read_record_file(os.path.join(root, "records", "adapters", ad.name, d, f), compute_sha=False))
    # per revision: the marker, the run-scale files (artifact, judgment, run, run_state, run_metric) and the item-scale files
    # (item_result with 250 rows; source_status, an item-scale table, with one) -- the same layout in every directory
    layout = {d: sorted((record_type(f), n_records(d, f)) for f in fs) for d, fs in files_per_rev.items()}
    assert len(set(map(str, layout.values()))) == 1, layout
    fs = files_per_rev[rev_dirs[0]]
    run_scale_files = [f for f in fs if record_type(f) not in R.ITEM_SCALE_RECORD_TYPES and record_type(f) != "source_revision"]
    item_files = [f for f in fs if record_type(f) in R.ITEM_SCALE_RECORD_TYPES]
    assert len(run_scale_files) == 5 and sorted(record_type(f) for f in item_files) == ["item_result", "source_status"], fs

    conn = DB.open_or_create(DB.db_path(root))
    begins = []
    conn.set_trace_callback(lambda s: begins.append(s) if s.strip().upper().startswith(("BEGIN", "COMMIT", "ROLLBACK")) else None)
    now = synth.t(2500)
    try:
        ing = H.ingest_once(root, conn=conn, workers=0, min_free=0, now=now)
    finally:
        conn.set_trace_callback(None)
        conn.close()
    assert ing.refused == 0 and ing.blocked == 0 and ing.revisions_held == 0, ing.refusals

    # --- the structure, per revision directory --------------------------------------------------------
    # staging: one `prepare` transaction (the marker ranked, the pending placeholder row), then pass A -- one transaction per
    # run-scale file (its few records staged as records) -- then pass B -- one bounded window per ITEM_ROWS_PER_TRANSACTION
    # item rows of every item-scale file, and one bookkeeping window per run-scale-only file (its staging row and rank, F16)
    windows = sum(math.ceil(n_records(rev_dirs[0], f) / WINDOW_ROWS) for f in item_files)            # 250 / 200 -> 2, plus source_status's 1
    assert windows == 3
    per_revision_staging = 1 + len(run_scale_files) + windows + len(run_scale_files)                   # 1 + 5 + 3 + 5 = 14
    assert ing.staging_transactions == N_STORES * per_revision_staging, (ing.staging_transactions, per_revision_staging, layout)
    # activation: exactly ONE per revision directory, after its staging; reclaim: none (first reads: no superseded revision)
    assert ing.revision_transactions == N_STORES and ing.reclaim_transactions == 0
    seq = [(r["kind"], r["rows"]) for r in _metrics(root, now) if r["kind"] in ("staging", "activation")]
    marker_records = n_records(rev_dirs[0], next(f for f in fs if record_type(f) == "source_revision"))
    one_revision = ([("staging", marker_records)] + [("staging", 1)] * len(run_scale_files)            # prepare, pass A
                    + [("staging", 1)] * len(run_scale_files) + [("staging", 200), ("staging", 50), ("staging", 1)]   # pass B, in file order
                    + [("activation", marker_records + ITEMS_PER_STORE)])
    assert seq == one_revision * N_STORES, seq
    # then the plain chunk folding the cycle's own facts (one run_ingest file per completed run + the arrival facts file, ≤ 50
    # files -> ONE transaction), the decode-policy stamp, and the ONE dense pass with no work
    kinds = [r["kind"] for r in _metrics(root, now)]
    assert kinds.count("transaction") == 1 and kinds.count("rerank") == 1 and kinds.count("dimensions") == 0
    assert ing.harvester_records == N_STORES and ing.arrival_facts >= 1
    assert ing.transactions == ing.staging_transactions + ing.revision_transactions + 1 + 1
    # every BEGIN the cycle issued commits (no rollback in a clean fold); the untimed ones are the blocked-queue bookkeeping
    n_begin = sum(1 for s in begins if s.strip().upper().startswith("BEGIN"))
    n_commit = sum(1 for s in begins if s.strip().upper().startswith("COMMIT"))
    assert n_begin == n_commit and not any(s.strip().upper().startswith("ROLLBACK") for s in begins)
    assert n_begin >= ing.transactions
    # the pinned totals for THIS fixture -- the numbers a structural change moves
    assert (ing.staging_transactions, ing.revision_transactions, ing.reclaim_transactions, ing.transactions) == (42, 3, 0, 47)
    with DB.reading(DB.db_path(root)) as c:
        assert c.execute("SELECT COUNT(*) FROM item_result").fetchone()[0] == N_STORES * ITEMS_PER_STORE
        assert c.execute("SELECT COUNT(*) FROM source_revision WHERE accepted = 1").fetchone()[0] == N_STORES
