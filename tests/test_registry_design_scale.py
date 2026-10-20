"""Design-scale sizing gate — Modal CPU suite ONLY.

Deselected locally: it runs only when `REGISTRY_DESIGN_SCALE=1` is set (the grading host has ~23 GB free and the
gate writes tens of GB). Also carries the `design_scale` marker. Synthetic corpus: 10 M `item_result` rows
(≈ 4 M items × 2.5 metrics over ≈ 2 k runs), 3 source revisions (rev 2 supersedes 5 % of items and withdraws
1 %; rev 3 is unchanged and writes only its revision record), 1 regrade judgment over one run; `registry
rebuild` + `registry snapshot` under `footprint_watcher`. Asserts TOTAL peak (DB + WAL + gz log + Parquet set +
run-level file + tmp) ≤ 10 GB, live DB ≤ 6 GB, run-level ≤ `RUNLEVEL_MAX_BYTES` (512 MiB), Parquet ≤ 40 % of the DB, and records the
timings in `design_scale_report.json`. `test_synth_corpus_shape` pins the generator's counts at
a reduced scale so the gate cannot pass on a smaller corpus by accident.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import snapshot as SN
from sisteer.registry import synth

GB = 1024 ** 3
DESIGN = {"n_runs": 2000, "items_per_run": 2000, "metrics_per_item": 2.5, "target_item_rows": 10_000_000,
          "rev2_superseded_frac": 0.05, "rev2_withdrawn_frac": 0.01}
design_scale = pytest.mark.skipif(os.environ.get("REGISTRY_DESIGN_SCALE") != "1",
                                  reason="design-scale gate runs only in the Modal CPU suite (REGISTRY_DESIGN_SCALE=1)")


def synth_corpus(root: str, n_runs: int, items_per_run: int, metrics=("asr", "judge_score", "coherence")) -> dict:
    """Write the design corpus at the given scale; returns the counts it claims."""
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    rev1, rev2, rev3 = "rev1" + "0" * 12, "rev2" + "0" * 12, "rev3" + "0" * 12
    counts = {"item_rows_rev1": 0, "item_rows_rev2": 0, "withdrawn": 0, "regrade_rows": 0, "runs": n_runs}
    now = synth.t(1)
    prod = synth.producer()
    revs = [synth.rec("source_revision", {"revision_id": r, "source_id": "design:all", "adapter": "design", "observed_utc": synth.t(i),
                                           "inputs_json": {"design": "%064x" % i}, "accepted": 1}, synth.t(i)) for i, r in ((10, rev1), (20, rev2), (30, rev3))]
    R.write_records(os.path.join(root, "records", "adapters", "design", "000001-revisions.jsonl"), revs)
    n_metrics_cycle = [2, 3]  # average 2.5 metrics per item
    for k in range(n_runs):
        rid, recs = synth.run_records(f"design_{k:05d}", 1000 + 20 * k, n_items=items_per_run)
        jid = I.judgment_id(rid, synth.JUDGE_ID)
        files = [(str(r["record_type"]), [r]) for r in recs]
        rows = []
        for i in range(items_per_run):
            for m in metrics[: n_metrics_cycle[i % 2]]:
                rows.append(R.make_record("item_result", {
                    "run_id": rid, "item_id": f"q{i:05d}", "metric_id": m, "trial_index": 0, "sample_index": 0, "turn_index": -1,
                    "judgment_id": jid, "value": (i * 7 % 100) / 100.0, "label": None, "api_pool": "lowprio", "created_utc": now,
                    "judged_utc": None, "source_revision_id": rev1, "withdrawn_by_revision_id": None}, prod, I.producer_record_id(), synth.t(500 + k)))
        counts["item_rows_rev1"] += len(rows)
        files.append(("item_result", rows))
        # rev 2: 5 % superseded (changed whole-row records), 1 % withdrawn
        n_sup = int(items_per_run * DESIGN["rev2_superseded_frac"])
        sup = []
        for i in range(n_sup):
            sup.append(R.make_record("item_result", {
                "run_id": rid, "item_id": f"q{i:05d}", "metric_id": metrics[0], "trial_index": 0, "sample_index": 0, "turn_index": -1,
                "judgment_id": jid, "value": 0.5, "label": None, "api_pool": "lowprio", "created_utc": now, "judged_utc": None,
                "source_revision_id": rev2, "withdrawn_by_revision_id": None}, prod, I.producer_record_id(), synth.t(20000 + k)))
        counts["item_rows_rev2"] += len(sup)
        files.append(("item_result", sup))
        n_wd = int(items_per_run * DESIGN["rev2_withdrawn_frac"])
        keys = [{"run_id": rid, "item_id": f"q{items_per_run - 1 - i:05d}", "metric_id": metrics[0], "trial_index": 0, "sample_index": 0,
                 "turn_index": -1, "judgment_id": jid} for i in range(n_wd)]
        counts["withdrawn"] += len(keys)
        files.append(("withdrawal", [R.make_record("withdrawal", {"source_id": "design:all", "observation_key": json.dumps({"items_of": rid}),
                                                                  "from_revision_id": rev1, "to_revision_id": rev2, "table_name": "item_result",
                                                                  "reason": "removed_items", "item_keys_json": keys}, prod,
                                                   I.adapter_record_id({"w": rid}, "withdrawal"), synth.t(20001 + k))]))
        if k == 0:  # one regrade judgment over this run's items
            jid2 = I.judgment_id(rid, synth.JUDGE_ID_2)
            files.append(("judgment", [synth.rec("judgment", {"judgment_id": jid2, "run_id": rid, "judge_id": synth.JUDGE_ID_2,
                                                              "n_items_judged": items_per_run, "status": "complete"}, synth.t(30000))]))
            regrade = [R.make_record("item_result", {
                "run_id": rid, "item_id": f"q{i:05d}", "metric_id": metrics[0], "trial_index": 0, "sample_index": 0, "turn_index": -1,
                "judgment_id": jid2, "value": 0.25, "label": None, "api_pool": "batch", "created_utc": now, "judged_utc": None,
                "source_revision_id": "", "withdrawn_by_revision_id": None}, prod, I.producer_record_id(), synth.t(30001))
                for i in range(items_per_run)]
            counts["regrade_rows"] = len(regrade)
            files.append(("item_result", regrade))
        synth.write_run_dir(root, "design", rid, files, gz_items=True)
    return counts


def test_synth_corpus_shape(tmp_path):
    """Reduced scale (2 runs × 200 items) pins the generator's arithmetic: rows, revision membership, withdrawals."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    counts = synth_corpus(root, n_runs=2, items_per_run=200)
    assert counts["item_rows_rev1"] == 2 * 200 * 2.5 and counts["item_rows_rev2"] == 2 * 10 and counts["withdrawn"] == 2 * 2
    assert counts["regrade_rows"] == 200
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals[:5]
    conn = DB.connect(DB.db_path(root), readonly=True)
    # rows are keyed by revision, so rev2's 20 superseding rows COEXIST with rev1's; visibility is derived from
    # the head — rev3 (accepted last, carrying nothing) hides every rev1 / rev2 row, the 200 natively emitted regrade rows are
    # what a reader sees; the withdrawn marks are derived by reclaim: every rev1 row whose key rev2 does not carry (1000 - 2 runs
    # × 10 items) is marked with rev2, every rev2 row (rev3 carries nothing) with rev3
    total = conn.execute("SELECT COUNT(*) FROM item_result").fetchone()[0]
    wd = conn.execute("SELECT COUNT(*) FROM item_result WHERE withdrawn_by_revision_id IS NOT NULL").fetchone()[0]
    sup = conn.execute("SELECT COUNT(*) FROM item_result WHERE source_revision_id = ?", ("rev2" + "0" * 12,)).fetchone()[0]
    visible = conn.execute("SELECT COUNT(*) FROM v_item_result_current").fetchone()[0]
    assert total == 2 * 200 * 2.5 + 20 + 200 and wd == (1000 - 20) + 20 and sup == 20 and visible == 200
    assert conn.execute("SELECT COUNT(*) FROM row_history WHERE table_name = 'item_result'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM source_revision").fetchone()[0] == 3
    conn.close()
    # the projected design scale from these constants
    assert DESIGN["n_runs"] * DESIGN["items_per_run"] * DESIGN["metrics_per_item"] == DESIGN["target_item_rows"]


@design_scale
@pytest.mark.design_scale
def test_design_scale_peak_footprint_within_10gb(tmp_path):
    scratch = os.environ.get("REGISTRY_DESIGN_SCRATCH") or str(tmp_path)
    root = os.path.join(scratch, "reg")
    H.ensure_layout(root)
    t0 = time.monotonic()
    counts = synth_corpus(root, DESIGN["n_runs"], DESIGN["items_per_run"])
    gen_s = time.monotonic() - t0
    into = os.path.join(scratch, "registry.sqlite")
    report = {"counts": counts, "corpus_generation_s": round(gen_s, 1)}
    watched = [into, os.path.join(root, "records"), SN.snapshots_dir(root), os.path.join(root, "tmp")]
    if os.environ.get("TMPDIR") and os.path.realpath(os.environ["TMPDIR"]) not in ("/tmp", "/var/tmp"):
        watched.append(os.environ["TMPDIR"])
    with SN.FootprintWatcher(watched, 1.0) as w:
        t1 = time.monotonic()
        rep = H.rebuild(root, into, workers=int(os.environ.get("REGISTRY_WORKERS", "6")))
        report["rebuild_wall_s"] = round(time.monotonic() - t1, 1)
        report["parse_s"], report["write_s"], report["longest_transaction_s"] = round(rep.parse_s, 1), round(rep.write_s, 1), round(rep.longest_transaction_s, 2)
        db_bytes = DB.file_size(into)
        t2 = time.monotonic()
        info = SN.build(root, into, check_free_space=False)
        report["snapshot_s"] = round(time.monotonic() - t2, 1)
        # a concurrent reader during the build window must not see a lock error
        conn = DB.connect(into, readonly=True)
        t3 = time.monotonic()
        conn.execute("SELECT COUNT(*) FROM v_paper_long").fetchone()
        report["frame_v_paper_long_s"] = round(time.monotonic() - t3, 2)
        conn.close()
    report.update({"peak_total_bytes": w.peak, "peak_breakdown": w.peak_breakdown, "db_bytes": db_bytes,
                   "peak_rss_bytes": w.peak_rss_bytes, "peak_child_rss_bytes": w.peak_child_rss_bytes,
                   "runlevel_bytes": info.runlevel_bytes, "parquet_bytes": info.parquet_bytes,
                   "parquet_ratio": info.parquet_bytes / db_bytes, "refused": rep.refused})
    with open(os.path.join(scratch, "design_scale_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    assert rep.refused == 0
    assert w.peak <= 10 * GB, f"total peak {w.peak / GB:.2f} GB > 10 GB: {w.peak_breakdown}"
    assert db_bytes <= 6 * GB, f"live DB {db_bytes / GB:.2f} GB > 6 GB (D20 escape hatch: split item_result per campaign)"
    assert info.runlevel_bytes <= SN.RUNLEVEL_MAX_BYTES
    assert info.parquet_bytes <= 0.4 * db_bytes, f"Parquet set {info.parquet_bytes / db_bytes:.2f} of DB"


def test_design_scale_mark_is_registered():
    """The `design_scale` mark is declared in pyproject's `[tool.pytest.ini_options] markers`, so collecting this module
    under `--strict-markers` succeeds. Before the registration every gate logged a PytestUnknownMarkWarning for the mark
    and a strict collection refused the module outright ('design_scale' not found in `markers` configuration option)."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, PYTHONPATH=os.path.join(repo, "src") + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    out = subprocess.run([sys.executable, "-m", "pytest", "--strict-markers", "--co", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                          os.path.abspath(__file__)], cwd=repo, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, (out.stdout[-2000:], out.stderr[-2000:])
    assert "not found in `markers`" not in out.stdout + out.stderr
    assert "test_design_scale_peak_footprint_within_10gb" in out.stdout
