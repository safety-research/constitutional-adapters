"""A run's headline metric is the FIRST of the eval's ACCEPTED ids it files (2026-09-21).

`eval.headline_metric_id` names ONE spelling (strongreject_bon: `asr`, the 512-store + 4,096-climb reading at bon_n 75, the canonical
best-of-N), but the producers that run the NATIVE 4,096 pass file the same per-goal any-hit estimand as `asr_bon_cap4096`
(bon_stores 48 runs, seedrep_cluster 24 runs on 2026-09-21; counted as one unit), and every consumer that tested the declared id alone
read the native-pass best-of-N as absent: no degeneracy stamp on its metric row, no headline items. The seed lists every accepted
spelling in order (`eval.extra_json.headline_metric_ids`), `paper.headline_metric_ids` reads the list in ONE place,
`run_headline_metric_id` picks the first the run files (else the declared id), the paper frame stamps THAT row, and
`v_item_headline` applies the same rule in SQL (migration 0008). A run filing both keeps `asr` (the list's order is the preference).
"""
from __future__ import annotations

import json
import os

import pandas as pd

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity
from sisteer.registry import paper as P
from sisteer.registry import records as R
from sisteer.registry import synth

LIST = ["asr", "asr_bon_cap4096"]
N_ITEMS = 8


# ---- the helper: one place, with and without the list ------------------------------------------------------------------------------

def test_headline_metric_ids_is_the_declared_id_alone_without_a_list():
    assert P.headline_metric_ids({"headline_metric_id": "asr", "extra_json": "{}"}) == ["asr"]
    assert P.headline_metric_ids({"headline_metric_id": "acc", "extra_json": None}) == ["acc"]
    assert P.headline_metric_ids({"headline_metric_id": "petri_score"}) == ["petri_score"]
    assert P.headline_metric_ids({"headline_metric_id": "asr", "extra_json": json.dumps({"headline_metric_ids": "asr"})}) == ["asr"]   # not a list: ignored


def test_headline_metric_ids_reads_the_list_in_order_and_closes_it_with_the_declared_id():
    text = {"headline_metric_id": "asr", "extra_json": json.dumps({"headline_metric_ids": LIST, "headline_bon_n": 75})}
    assert P.headline_metric_ids(text) == LIST
    parsed = {"headline_metric_id": "asr", "extra_json": {"headline_metric_ids": LIST}}
    assert P.headline_metric_ids(parsed) == LIST
    # the declared id is appended when the list omits it, and a repeat collapses
    assert P.headline_metric_ids({"headline_metric_id": "asr", "extra_json": {"headline_metric_ids": ["asr_bon_cap4096"]}}) == ["asr_bon_cap4096", "asr"]
    assert P.headline_metric_ids({"headline_metric_id": "asr", "extra_json": {"headline_metric_ids": ["asr", "asr", "asr_bon_cap4096"]}}) == LIST
    # a frame row spells the eval's extra as eval_extra_json
    assert P.headline_metric_ids({"headline_metric_id": "asr", "eval_extra_json": json.dumps({"headline_metric_ids": LIST})}) == LIST


def test_run_headline_metric_id_is_the_first_listed_id_the_run_files_else_the_declared_one():
    ev = {"headline_metric_id": "asr", "extra_json": {"headline_metric_ids": LIST}}
    assert P.run_headline_metric_id(ev, ["asr_bon_cap4096", "bon_nauc", "asr_at_n1"]) == "asr_bon_cap4096"   # the native pass
    assert P.run_headline_metric_id(ev, ["asr", "asr_at_n1"]) == "asr"                                         # the climb reading
    assert P.run_headline_metric_id(ev, ["asr_bon_cap4096", "asr"]) == "asr"                                   # both: the list's order
    assert P.run_headline_metric_id(ev, ["asr_at_n1", "bon_nauc"]) == "asr"                                    # neither: the declared id
    assert P.run_headline_metric_id({"headline_metric_id": "acc", "extra_json": "{}"}, ["acc"]) == "acc"
    assert P.run_headline_metric_id({"headline_metric_id": "acc", "extra_json": "{}"}, []) == "acc"


# ---- the frame and the item rows, on a registry ------------------------------------------------------------------------------------

def _degeneration(rid, jid, rate, n, i):
    k = int(round(rate * n))
    return synth.rec("run_degeneration", {
        "run_id": rid, "judgment_id": jid, "measure": "at_token_cap_finish_reason", "scope": "all", "measurement_state": "measured",
        "numerator": k, "denominator": n, "denominator_unit": "replies", "pooling_json": {"unit": "replies"}, "aggregation": "single_detector",
        "rate": k / n, "definition_version": "synthetic_sidecar/1", "threshold_params_json": {}, "computed_by": "producer",
        "computed_utc": synth.t(i), "source_uri": "synthetic://sidecar", "notes": None}, synth.t(i))


def _metric(rid, jid, metric_id, value, i):
    return synth.rec("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": metric_id, "threshold": 0.5, "value": value,
                                    "k": int(round(value * N_ITEMS)), "n": N_ITEMS, "computed_by": "producer"}, synth.t(i))


def _corpus(tmp_path):
    """Synth's dimensions with the eval ACCEPTING `asr` then `asr_bon_cap4096`, and three judged runs of it: `climb` files `asr` (the
    default), `native` files `asr_bon_cap4096` ONLY, `both` files the two. Each carries one measured finish-reason at-cap reading and
    item results under the metric ids it files."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    dims = [r for r in synth.seed_dimensions() if r["record_type"] != "eval"]
    dims.append(synth.rec("eval", {"eval_id": synth.EVAL_ID, "eval_category": "static_attack", "headline_metric_id": "asr",
                                   "metric_direction": "lower_is_better", "attacker_required": 0, "paper_status": "in_paper",
                                   "extra": {"headline_metric_ids": LIST}}, synth.t(4)))
    R.write_records(os.path.join(root, "records", "dimensions", "seed", "000001-dimensions.jsonl"), dims)
    runs = {}
    plans = {"climb": (["asr"], 0.1, 100), "native": (["asr_bon_cap4096"], 0.2, 200), "both": (["asr", "asr_bon_cap4096"], 0.3, 300)}
    for name, (metrics, rate, i0) in plans.items():
        rid, recs = synth.run_records(f"synthstore_{name}", i0, n_items=N_ITEMS)
        jid = identity.judgment_id(rid, synth.JUDGE_ID)
        recs = [r for r in recs if r["record_type"] != "run_metric"]
        files = [(str(r["record_type"]), [r]) for r in recs]
        for k, m in enumerate(metrics):
            files.append(("run_metric", [_metric(rid, jid, m, 0.5 if m == "asr_bon_cap4096" else 0.25, i0 + 9 + k)]))
        files.append(("run_degeneration", [_degeneration(rid, jid, rate, N_ITEMS, i0 + 12)]))
        for k, m in enumerate(metrics):
            files.append(("item_result", synth.item_records(rid, jid, i0 + 5, N_ITEMS, seed=k, metric=m)))
        synth.write_run_dir(root, synth.CAMPAIGN, rid, files)
        runs[name] = (rid, jid, rate)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, (rep.refusals, rep.blocked)
    return root, runs


def test_paper_frame_stamps_the_run_s_first_accepted_headline_row(tmp_path):
    """The native-pass run's `asr_bon_cap4096` row IS its headline row: exported with its own value and the run's degeneracy reading
    stamped on it (before the fix `_is_headline` compared the declared `asr` and left it unstamped); the climb run's `asr` row is unchanged;
    a run filing both keeps `asr` as the headline and its `asr_bon_cap4096` row unstamped (the list's order is the preference)."""
    root, runs = _corpus(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        ev = conn.execute("SELECT eval_id, headline_metric_id, extra_json FROM eval WHERE eval_id = ?", (synth.EVAL_ID,)).fetchone()
        assert ev["headline_metric_id"] == "asr" and P.headline_metric_ids(dict(ev)) == LIST
        frame = P.paper_long_frame(conn, current_only=False, columns=["source", "metric", "value", "at_cap_share_finish_reason", "degeneracy_basis"])
    finally:
        conn.close()
    by_run = {name: frame[frame["run_id"] == rid].set_index("metric") for name, (rid, _, _) in runs.items()}
    climb, native, both = by_run["climb"], by_run["native"], by_run["both"]
    assert list(climb.index) == ["asr"] and climb.loc["asr", "value"] == 0.25 and climb.loc["asr", "at_cap_share_finish_reason"] == 0.125
    assert "at_cap_share_finish_reason = at_token_cap_finish_reason" in str(climb.loc["asr", "degeneracy_basis"])
    # the native pass: exported under its own id, with its own value, and stamped as the headline row
    assert list(native.index) == ["asr_bon_cap4096"] and native.loc["asr_bon_cap4096", "value"] == 0.5
    assert native.loc["asr_bon_cap4096", "at_cap_share_finish_reason"] == 0.25
    assert "at_cap_share_finish_reason = at_token_cap_finish_reason" in str(native.loc["asr_bon_cap4096", "degeneracy_basis"])
    # both filed: `asr` first in the list, so it is the headline; the cap row is a plain metric row
    assert sorted(both.index) == LIST and both.loc["asr", "value"] == 0.25 and both.loc["asr_bon_cap4096", "value"] == 0.5
    assert both.loc["asr", "at_cap_share_finish_reason"] == 0.25 and pd.isna(both.loc["asr_bon_cap4096", "at_cap_share_finish_reason"])
    assert pd.isna(both.loc["asr_bon_cap4096", "degeneracy_basis"])
    # the resolution the frame used, per run
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        df = pd.read_sql_query("SELECT run_id, eval_id, metric_id FROM v_paper_long", conn)
        heads = P._headline_by_run(conn, df)
    finally:
        conn.close()
    assert heads == {runs["climb"][0]: "asr", runs["native"][0]: "asr_bon_cap4096", runs["both"][0]: "asr"}


def test_a_fixture_row_with_no_frame_reads_the_declared_id():
    """`render_long_row` without a database (the spelling tests' path) keeps the pre-list reading: the row's own declared id."""
    row = {"run_id": "r", "metric_id": "asr", "headline_metric_id": "asr"}
    assert P._is_headline(row) and P._is_headline(row, {}) and P._is_headline(row, {"headline_by_run": {}})
    assert not P._is_headline({**row, "metric_id": "asr_bon_cap4096"})
    assert P._is_headline({**row, "metric_id": "asr_bon_cap4096"}, {"headline_by_run": {"r": "asr_bon_cap4096"}})
    assert not P._is_headline(row, {"headline_by_run": {"r": "asr_bon_cap4096"}})


def test_v_item_headline_takes_the_run_s_first_accepted_id(tmp_path):
    """The items of the native-pass run (filed under `asr_bon_cap4096` only) are its headline items; a run filing items under both ids
    shows its `asr` items only; the climb run is unchanged."""
    root, runs = _corpus(tmp_path)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        got = {name: conn.execute("SELECT metric_id, COUNT(*) FROM v_item_headline WHERE run_id = ? GROUP BY 1", (rid,)).fetchall()
               for name, (rid, _, _) in runs.items()}
        total = conn.execute("SELECT COUNT(*) FROM v_item_headline").fetchone()[0]
        filed = conn.execute("SELECT COUNT(*) FROM v_item_result_current").fetchone()[0]
    finally:
        conn.close()
    assert {k: [tuple(r) for r in v] for k, v in got.items()} == {"climb": [("asr", N_ITEMS)], "native": [("asr_bon_cap4096", N_ITEMS)],
                                                                  "both": [("asr", N_ITEMS)]}
    assert total == 3 * N_ITEMS and filed == 4 * N_ITEMS
