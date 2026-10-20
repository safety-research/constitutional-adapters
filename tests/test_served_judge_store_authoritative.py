"""On resume/re-emit the served-controls judge's per-item records follow the STORE's filed label rows, not the shared
cache (2026-09-10): a byte-identical (prompt, response) judged in two cells shares one cache key, INSERT OR REPLACE keeps
the last verdict, and the earlier cell's store keeps the first — so four cells failed `recomputed band != harness` on
exactly one item each, deterministically, on every re-run. The filed row is the record; the disagreement is flagged and
counted, never silent, and never a crash."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
pytest.importorskip("sisteer.capeval.judging")
import served_capability_judge as scj  # noqa: E402
from sisteer.capeval import judging  # noqa: E402
from sisteer.capeval.tiers import judged  # noqa: E402

TASK = "judged/xstest_v2"
ITEMS = ["xst0001-aaaaaaaa", "xst0002-bbbbbbbb", "xst0003-cccccccc"]


def _cell(rows_filed):
    """A Cell-shaped stub: three safe XSTest items; the store's parsed rows are given."""
    items = list(ITEMS)
    spec = SimpleNamespace(task=TASK, rows=[{"item_id": i, "subset": "safe", "type": "homonyms"} for i in items])
    store = SimpleNamespace(_parsed_rows=lambda: list(rows_filed))
    keys = {i: f"key-{i}" for i in items}
    return SimpleNamespace(target="TESTTAG_t0", name="cellA", judge_model="claude-sonnet-5", specs={"xstest": spec},
                           store=store, request_keys=lambda control: keys,
                           at_cap_share=lambda control: 0.0), items, keys


def _label_row(item, label):
    return {"task": TASK, "item_id": item, "metric": "xst_label", "value": judged.LABEL_VALUE[label],
            "extra_json": '{"label": "%s"}' % label}


def test_store_label_wins_and_the_cache_disagreement_is_flagged_and_counted(tmp_path):
    cell, items, keys = _cell([_label_row(ITEMS[0], "full_refusal"), _label_row(ITEMS[1], "full_compliance")])
    cache = judging.JudgeCache(tmp_path / "c.sqlite")
    cache.put(keys[items[0]], {"label": "full_compliance"}, "lowprio")        # another cell's later verdict for the same text
    cache.put(keys[items[1]], {"label": "full_compliance"}, "lowprio")        # agrees
    cache.put(keys[items[2]], {"abstained": True, "reason": "refusal"}, "lowprio")   # no store row: the cache's abstention
    recs = scj._item_records(cell, "xstest", cache, pre_cached={k: True for k in keys.values()})
    by = {r["item_id"]: r for r in recs}
    assert by[items[0]]["status"] == "judged" and by[items[0]]["label"] == "full_refusal"
    assert by[items[0]]["label_source"] == "store" and by[items[0]]["store_cache_disagree"] is True
    assert by[items[0]]["cache_label"] == "full_compliance"
    assert by[items[1]]["label"] == "full_compliance" and "store_cache_disagree" not in by[items[1]]
    assert by[items[2]]["status"] == "abstained" and by[items[2]]["abstention_reason"] == "refusal"
    blk = scj._block_from_records("xstest", recs, cell, {"status": "ok"})
    assert blk["n_store_cache_disagree"] == 1
    # the statistics use the FILED labels: 1 refusal of 2 judged over the 3 safe items (one abstained)
    over = blk["over_refusal"]
    assert over["n_judged"] == 2 and over["n_abstained"] == 1
    assert over["abstention_band_full_n"] == [pytest.approx(1 / 3), pytest.approx(2 / 3)]


def test_an_unfiled_item_still_reads_the_cache_and_a_raced_store_refuses(tmp_path):
    cell, items, keys = _cell([])
    cache = judging.JudgeCache(tmp_path / "c.sqlite")
    cache.put(keys[items[0]], {"label": "partial_refusal"}, "lowprio")
    recs = scj._item_records(cell, "xstest", cache, pre_cached={})
    by = {r["item_id"]: r for r in recs}
    assert by[items[0]]["label"] == "partial_refusal" and by[items[0]]["label_source"] == "cache"
    assert by[items[1]]["status"] == "unverdicted" and "label_source" not in by[items[1]]
    raced, _, _ = _cell([_label_row(ITEMS[0], "full_refusal"), _label_row(ITEMS[0], "full_compliance")])
    with pytest.raises(RuntimeError, match="filed twice with DIFFERENT labels"):
        scj._item_records(raced, "xstest", cache, pre_cached={})
