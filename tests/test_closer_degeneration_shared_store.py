"""degen-fix-6: the controls judge's pre-claim validation reads EVERY selected row by run id, and a store
+ control named by more than one pending run refuses before any claim.

degen-fix-5's check validated `by_store`, keyed by (store dir, control): rows sharing a store + control overwrote each other, so a shared
store with an earlier NULL-count run and a later counted run passed on the last count, `claim_all` durably claimed both, REGISTRY_RUNS held
one, and the uncounted run stayed claimed with no completion (live: a default-judge xstest store with 16 pending NULL-count runs).

Pinned through the REAL worklist over a real registry world (the null-count module's harness) and a real-shaped sampled store:
  * two pending xstest runs on ONE store, one NULL and one counted — in BOTH registration orders — refuse with a nonzero exit naming the NULL
    run, zero claims (no judgment row; REGISTRY_RUNS never built), zero filings, both runs still pending;
  * two COUNTED pending xstest runs on one store refuse by name, listing both run ids (REGISTRY_RUNS carries one run per store + control),
    zero claims, zero filings;
  * one run per store + control (xstest + or_bench on one store) still claims and carries its counts (the positive control)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "tests", REPO / "scripts", REPO / "scripts" / "sdfbase" / "full22", REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sisteer.registry import harvest as H  # noqa: E402
from sisteer.registry import pending as P  # noqa: E402
from test_native_judging_cardinality import _conn  # noqa: E402
from test_closer_degeneration_null_count import _judge, _judgment_records, _ns, _registry_with_null_counts, _sampled_store  # noqa: E402


def _no_claims_no_filings(root, rids):
    assert _judgment_records(H.records_dir(root)) == []                                          # zero filings
    conn = _conn(root)
    assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0                        # zero claims
    assert sorted(r["run_id"] for r in P.pending(conn)) == sorted(rids)                            # every run still pending, untouched
    conn.close()


@pytest.mark.parametrize("null_first", [True, False])
def test_a_null_count_run_sharing_a_store_with_a_counted_run_is_refused_before_any_claim(tmp_path, monkeypatch, null_first):
    store = _sampled_store(tmp_path, [f"xstest-{i}" for i in range(6)])
    runs = [("k_null", "xstest", str(store), None), ("k_counted", "xstest", str(store), 6)]
    if not null_first:
        runs.reverse()
    root, rids = _registry_with_null_counts(tmp_path, monkeypatch, runs)
    r_null = rids[0] if null_first else rids[1]
    conn = _conn(root)
    assert conn.execute("SELECT n_items_generated FROM run WHERE run_id = ?", (r_null,)).fetchone()[0] is None
    assert len(P.pending(conn)) == 2                                                               # BOTH pending on the one store
    conn.close()
    drv = _judge(monkeypatch, f"shared_{null_first}")
    with pytest.raises(SystemExit) as ei:
        drv.registry_worklist_cells(_ns(), {}, ["xstest"])
    msg = str(ei.value)
    assert ei.value.code not in (0, None)
    assert "refusing before any claim" in msg and "nothing claimed" in msg, msg
    assert r_null in msg and "n_items_generated" in msg, msg                                    # the NULL run is named, whichever row came last
    assert drv.REGISTRY_RUNS is None
    _no_claims_no_filings(root, rids)


def test_two_counted_runs_on_one_store_refuse_by_name_before_any_claim(tmp_path, monkeypatch):
    store = _sampled_store(tmp_path, [f"xstest-{i}" for i in range(6)])
    root, (r_a, r_b) = _registry_with_null_counts(tmp_path, monkeypatch, [("k_a", "xstest", str(store), 6), ("k_b", "xstest", str(store), 6)])
    drv = _judge(monkeypatch, "dups")
    with pytest.raises(SystemExit) as ei:
        drv.registry_worklist_cells(_ns(), {}, ["xstest"])
    msg = str(ei.value)
    assert ei.value.code not in (0, None)
    assert "named by more than one pending run" in msg and r_a in msg and r_b in msg and f"{store} / xstest" in msg, msg
    assert "REGISTRY_RUNS carries one run per store+control" in msg and "nothing claimed" in msg and "retire or coexist" in msg
    assert drv.REGISTRY_RUNS is None
    _no_claims_no_filings(root, [r_a, r_b])


def test_one_run_per_store_and_control_still_claims_with_its_counts(tmp_path, monkeypatch):
    store = _sampled_store(tmp_path, [f"xstest-{i}" for i in range(6)], [f"orb-{i}" for i in range(4)])
    root, (r_xs, r_orb) = _registry_with_null_counts(tmp_path, monkeypatch, [("k_xs", "xstest", str(store), 6), ("k_orb", "or_bench_hard", str(store), 4)])
    drv = _judge(monkeypatch, "single")
    wl, cells = drv.registry_worklist_cells(_ns(), {}, ["xstest", "or_bench_hard_1k"])
    assert len(cells) == 1 and cells[0].controls == ["or_bench_hard_1k", "xstest"]
    key = f"{cells[0].target}/{cells[0].name}"
    assert drv.REGISTRY_RUNS[key]["xstest"]["n_items_generated"] == 6 and drv.REGISTRY_RUNS[key]["or_bench_hard_1k"]["n_items_generated"] == 4
    conn = _conn(root)
    assert conn.execute("SELECT COUNT(*) FROM judgment WHERE status = 'in_progress'").fetchone()[0] == 2
    conn.close()
