"""The registry worklist's SELECTORS: `--run-id` (repeatable) and `--model-id` (repeatable) on
every judging driver's shared flags, honoured by `RegistryWorklist.pending()` -- a pending run of ANOTHER model, or one not listed, is
never returned and therefore never claimed; the grading daemon's `pending.pending(conn, model_id=...)` API narrows the same way."""
from __future__ import annotations

import argparse
import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import emit, synth
from sisteer.registry import harvest as H
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import worklist as W

OTHER_MODEL = "othermodel_9b_it"


def _dims_with_other_model():
    dims = synth.seed_dimensions()
    model_rows = [r for r in dims if r["record_type"] == "model"]
    assert model_rows
    extra = R.make_record("model", dict(model_rows[0]["payload"], model_id=OTHER_MODEL, producer_spellings_json=[OTHER_MODEL]),
                          model_rows[0]["producer"], "00000000-1111-2222-3333-999999999999", synth.t(0))
    return dims + [extra]


def _root(tmp_path, monkeypatch):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), _dims_with_other_model())
    rids = {}
    for k, (key, model) in enumerate((("k_a", synth.MODEL_ID), ("k_b", synth.MODEL_ID), ("k_other", OTHER_MODEL))):
        rid, recs = synth.run_records(key, 100 + 20 * k, through="generated", model_id=model)
        recs[0]["payload"]["store_uri"] = f"file:///stores/{key}"
        R.write_records(os.path.join(root, "records", "c", f"00000{k + 1}-run.jsonl"), recs)
        rids[key] = rid
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    return root, rids


def test_add_worklist_arguments_carries_the_selectors_and_selectors_from_args_parses_them():
    ap = argparse.ArgumentParser()
    W.add_worklist_arguments(ap)
    ns = ap.parse_args(["--worklist", "registry", "--run-id", "r1", "--run-id", "r2", "--model-id", "m1",
                        "--store-root-map", "cluster_a=/remote/root=/local/mirror"])
    sel = W.selectors_from_args(ns)
    assert sel == {"run_ids": ["r1", "r2"], "model_ids": ["m1"], "store_root_map": {"cluster_a": [("/remote/root", "/local/mirror")]}}
    ns2 = ap.parse_args([])
    assert W.selectors_from_args(ns2) == {"run_ids": None, "model_ids": None, "store_root_map": None}
    with pytest.raises(W.RegistryUnavailable):
        W.selectors_from_args(ap.parse_args(["--store-root-map", "nope"]))


def test_run_id_and_model_id_close_the_worklist_so_another_models_pending_run_is_never_claimed(tmp_path, monkeypatch):
    root, rids = _root(tmp_path, monkeypatch)
    # no selector: every pending run of the campaign
    wl_all = W.RegistryWorklist(root, [synth.JUDGE_ID], "s", driver="t")
    assert sorted(r["run_id"] for r in wl_all.pending()) == sorted(rids.values())
    # --model-id: the other model's run is out
    wl_m = W.RegistryWorklist(root, [synth.JUDGE_ID], "s", driver="t", model_ids=[synth.MODEL_ID])
    assert sorted(r["run_id"] for r in wl_m.pending()) == sorted([rids["k_a"], rids["k_b"]])
    # --run-id: exactly the listed run, whatever else is pending
    wl_r = W.RegistryWorklist(root, [synth.JUDGE_ID], "s", driver="t", run_ids=[rids["k_b"]])
    rows = wl_r.pending()
    assert [r["run_id"] for r in rows] == [rids["k_b"]] and rows[0]["store_path"] == "/stores/k_b" and rows[0]["substrate"] == "cluster_a"
    # a listed run of another model, under a model filter that excludes it: nothing (both selectors apply)
    wl_x = W.RegistryWorklist(root, [synth.JUDGE_ID], "s", driver="t", run_ids=[rids["k_other"]], model_ids=[synth.MODEL_ID])
    assert wl_x.pending() == []
    # the default-judge path honours the same selectors
    wl_d = W.RegistryWorklist(root, None, "s", driver="t", run_ids=[rids["k_a"]])
    assert [r["run_id"] for r in wl_d.pending()] == [rids["k_a"]]
    # grading through the narrowed worklist claims ONLY the selected run; the other model's stays pending for its judge
    def grade(row, scope):
        scope.items([{"item_id": f"q{i:05d}", "metric_id": "asr", "value": float(i % 2)} for i in range(200)])
        scope.metrics([{"metric_id": "asr", "threshold": 0.5, "value": 0.5, "k": 100, "n": 200}])
    done = wl_r.grade(rows, grade)
    assert done == [rids["k_b"]]
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)   # the harvester's fold of the spooled completion records (finish() folds nothing by default, 2026-09-30)
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rids["k_b"],)).fetchone()[0] == "judged"
        assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (rids["k_other"],)).fetchone()[0] == "generated"
        assert conn.execute("SELECT COUNT(*) FROM judgment WHERE run_id = ?", (rids["k_other"],)).fetchone()[0] == 0
        # the daemon's API: pending.pending(conn, model_id=...) is the same narrowing
        still = P.pending_for(conn, synth.JUDGE_ID, synth.CAMPAIGN, model_id=OTHER_MODEL)
        assert [r["run_id"] for r in still] == [rids["k_other"]]
        assert [r["run_id"] for r in P.pending_for(conn, synth.JUDGE_ID, synth.CAMPAIGN, run_id=(rids["k_a"],))] == [rids["k_a"]]
        assert "run_id" in P.FILTER_COLUMNS
