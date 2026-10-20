"""One OWNER metric row per measurement identity in `v_paper_long` (schema_def.METRIC_ROLE_CTES; migration 0004).

Two sources may read one measurement onto the SAME run id by design (the r2 export and the stats readers of the same stores mint
`identity.run_id(...)` alike): the run row folds to one, but `run_metric`'s key carries `computation_id`, so the export's computed row and
the reader's producer-filed blank-computation twin both stand, and a consumer counting or pooling in-paper rows read the measurement
twice (781 of the 5,507 in-paper rows on the live registry, 2026-09-20; values identical in every pair). `v_paper_long` now names ONE
`owner` row per (run, judgment, metric, threshold) — the run's own source's row first, then a computed row before a blank one, then the
declared priority, the lexically smaller source id, the smaller computation id — and reads `in_current_paper` on that row only; the
other rows at the identity stay as `secondary` evidence (in_current_paper 0, selection_state untouched).

Synthetic corpus (invented ids; hostname TESTHOST): one judged run under a roster entry, read by four synthetic sources whose head
revisions carry the run key — `export:g` (priority 20, the run's own source by rank), `stats:g` (0), `other:g` (10), `zeta:g` (10).
"""
from __future__ import annotations

import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import paper
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import synth

EXPORT, STATS, OTHER, ZETA = "export:g", "stats:g", "other:g", "zeta:g"
PRIORITY = {EXPORT: 20, STATS: 0, OTHER: 10, ZETA: 10}
COMP_EXPORT, COMP_STATS, COMP_OTHER, COMP_ZETA = "comp_export_0001", "comp_stats_0001", "comp_other_0001", "comp_zeta_0001"


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _roster_entry(i, selector):
    p = {"roster_entry_id": f"re{i}", "round_id": "r_synth", "comparison_id": "fig_synth", "model_id": synth.MODEL_ID,
         "eval_id": synth.EVAL_ID, "instrument_id": synth.INSTRUMENT_ID, "judge_id": synth.JUDGE_ID, "strength_rule": "operating_point",
         "source_citation": "paperlib.config.SYNTHETIC", "inferred": 0, "object_selector_json": selector}
    return synth.rec("roster_entry", p, synth.t(50 + i))


def _rev(rid, i, src):
    return synth.rec("source_revision", {"revision_id": rid, "source_id": src, "adapter": src.split(":")[0], "observed_utc": synth.t(i),
                                         "inputs_json": {"a": "%064x" % i}, "accepted": 1, "priority": PRIORITY[src]}, synth.t(i))


def _obs(rid, keys, i, table, src):
    from sisteer.registry import identity as I
    return synth.rec("observation", {"source_id": src, "revision_id": rid, "table_name": table,
                                     "observation_keys": [I.canonical_json(k) for k in keys]}, synth.t(i))


def _computation(cid, algorithm, i):
    return synth.rec("computation", {"computation_id": cid, "algorithm": algorithm, "version": "1", "seed_provenance": "none",
                                     "code_sha": synth.CODE_SHA}, synth.t(i))


def _metric(rid, jid, metric, thr, comp, value, i):
    return synth.rec("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": metric, "threshold": thr, "computation_id": comp,
                                    "value": value, "k": int(round(value * 100)), "n": 100, "computed_by": "producer"}, synth.t(i))


def _key(rid, jid, metric, thr, comp):
    return {"run_id": rid, "judgment_id": jid, "metric_id": metric, "threshold": thr, "computation_id": comp}


def _source_file(root, src, rev, i0, rid, metrics, computations=()):
    """One source's head revision: the marker, the run key, its metric keys, the computations it names, the rows."""
    jid_keys = [_key(m["payload"]["run_id"], m["payload"]["judgment_id"], m["payload"]["metric_id"], m["payload"]["threshold"],
                     m["payload"]["computation_id"]) for m in metrics]
    recs = [_rev(rev, i0, src), _obs(rev, [{"run_id": rid}], i0 + 1, "run", src), _obs(rev, jid_keys, i0 + 2, "run_metric", src)]
    recs += [_computation(cid, algo, i0 + 3) for cid, algo in computations]
    recs += metrics
    _write(root, "synth_sources", f"000001-{src.split(':')[0]}.jsonl", recs)


@pytest.fixture()
def corpus(tmp_path):
    """run 0 = the none object (not in the roster), run 1 = constitution_lora (selected). The lora run's asr@0.5 is the corpus's own
    natively emitted row (blank computation, no source carries it). Four sources then read the run:

    * asr@0.25 — `export:g` files a computed row AND `stats:g` files the blank-computation twin (the live defect);
    * asr@0.75 — only `stats:g`'s blank twin (nothing to prefer it over: it is the owner, nothing is dropped);
    * asr_held_in@-1 — `stats:g` (priority 0) and `other:g` (priority 10) both file COMPUTED rows, neither the run's own source
      (the tie the rank resolves by priority: `other:g` owns, and the identity is reported contested);
    * asr_first_turn@-1 — `other:g` and `zeta:g` at the SAME priority (the tie falls to the lexically smaller source id).
    """
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1, {"paper_label": "constitution_lora"})])
    lora, jid = ids["run_ids"][1], ids["judgment_ids"][1]
    _source_file(root, EXPORT, "rev_export", 9000, lora,
                 [_metric(lora, jid, "asr", 0.25, COMP_EXPORT, 0.4, 9010)], [(COMP_EXPORT, "export.build")])
    _source_file(root, STATS, "rev_stats", 9100, lora,
                 [_metric(lora, jid, "asr", 0.25, "", 0.4, 9110), _metric(lora, jid, "asr", 0.75, "", 0.1, 9111),
                  _metric(lora, jid, "asr_held_in", -1.0, COMP_STATS, 0.2, 9112)], [(COMP_STATS, "stats.reader")])
    _source_file(root, OTHER, "rev_other", 9200, lora,
                 [_metric(lora, jid, "asr_held_in", -1.0, COMP_OTHER, 0.2, 9210), _metric(lora, jid, "asr_first_turn", -1.0, COMP_OTHER, 0.3, 9211)],
                 [(COMP_OTHER, "other.reader")])
    _source_file(root, ZETA, "rev_zeta", 9300, lora,
                 [_metric(lora, jid, "asr_first_turn", -1.0, COMP_ZETA, 0.3, 9310)], [(COMP_ZETA, "zeta.reader")])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, (rep.refused, rep.blocked)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        yield root, conn, lora, jid
    finally:
        conn.close()


def _rows(conn, run_id):
    rows = [dict(r) for r in conn.execute("SELECT * FROM v_paper_long WHERE run_id = ? ORDER BY metric_id, threshold, computation_id", (run_id,))]
    return {(r["metric_id"], r["threshold"], r["computation_id"]): r for r in rows}


def test_blank_twin_of_a_computed_row_is_secondary_and_out_of_the_paper(corpus):
    root, conn, lora, jid = corpus
    rows = _rows(conn, lora)
    own, twin = rows[("asr", 0.25, COMP_EXPORT)], rows[("asr", 0.25, "")]
    assert own["metric_role"] == "owner" and own["in_current_paper"] == 1 and own["selection_state"] == "selected"
    assert twin["metric_role"] == "secondary" and twin["in_current_paper"] == 0
    assert twin["selection_state"] == "selected", "the run's selection is untouched: only the paper flag reads on the owner row"
    # both readings stay visible as evidence, value-neutral
    assert own["value"] == twin["value"] == 0.4
    assert own["metric_source_id"] == EXPORT and twin["metric_source_id"] == STATS
    assert own["run_source_id"] == twin["run_source_id"] == EXPORT, "the run's own source: the D54-2 top-ranked source carrying the run key"
    assert own["metric_role_candidates"] == twin["metric_role_candidates"] == 2
    assert own["metric_role_contested"] == twin["metric_role_contested"] == 0, "a blank twin beside a computed row is not a tie"


def test_a_blank_twin_standing_alone_is_the_owner_nothing_dropped(corpus):
    root, conn, lora, jid = corpus
    r = _rows(conn, lora)[("asr", 0.75, "")]
    assert r["metric_role"] == "owner" and r["in_current_paper"] == 1 and r["value"] == 0.1
    assert r["metric_role_candidates"] == 1 and r["metric_role_contested"] == 0 and r["metric_source_id"] == STATS


def test_a_single_natively_emitted_row_is_unchanged(corpus):
    root, conn, lora, jid = corpus
    r = _rows(conn, lora)[("asr", 0.5, "")]
    assert r["metric_role"] == "owner" and r["in_current_paper"] == 1 and r["value"] == 0.25
    assert r["metric_source_id"] is None, "no adapter carries the corpus's own row"
    assert r["run_source_id"] == EXPORT and r["metric_role_candidates"] == 1 and r["metric_role_contested"] == 0


def test_two_computed_rows_from_different_sources_resolve_by_priority_then_source_id_and_are_counted(corpus):
    root, conn, lora, jid = corpus
    rows = _rows(conn, lora)
    # priority decides: other:g (10) over stats:g (0); neither is the run's own source (export:g)
    hi, lo = rows[("asr_held_in", -1.0, COMP_OTHER)], rows[("asr_held_in", -1.0, COMP_STATS)]
    assert (hi["metric_role"], hi["in_current_paper"]) == ("owner", 1)
    assert (lo["metric_role"], lo["in_current_paper"]) == ("secondary", 0)
    assert hi["metric_role_contested"] == 1 and lo["metric_role_contested"] == 1 and hi["metric_role_candidates"] == 2
    assert hi["run_source_id"] == EXPORT and hi["metric_source_id"] == OTHER
    # equal priority: the lexically smaller source id (other:g < zeta:g)
    a, z = rows[("asr_first_turn", -1.0, COMP_OTHER)], rows[("asr_first_turn", -1.0, COMP_ZETA)]
    assert (a["metric_role"], a["in_current_paper"], z["metric_role"], z["in_current_paper"]) == ("owner", 1, "secondary", 0)
    assert a["metric_role_contested"] == 1 and z["metric_role_contested"] == 1
    # deterministic: the same answer on a second read
    again = _rows(conn, lora)
    assert {k: (v["metric_role"], v["in_current_paper"]) for k, v in again.items()} == {k: (v["metric_role"], v["in_current_paper"]) for k, v in rows.items()}


def test_exactly_one_owner_per_identity_and_the_paper_frame_counts_each_measurement_once(corpus):
    root, conn, lora, jid = corpus
    bad = conn.execute("SELECT run_id, judgment_id, metric_id, threshold, SUM(metric_role = 'owner') AS n_owner, COUNT(*) AS n FROM v_paper_long "
                       "GROUP BY 1, 2, 3, 4 HAVING n_owner <> 1").fetchall()
    assert not bad, [tuple(r) for r in bad]
    in_paper = conn.execute("SELECT metric_id, threshold, computation_id FROM v_paper_long WHERE in_current_paper = 1 AND run_id = ? "
                            "ORDER BY 1, 2, 3", (lora,)).fetchall()
    assert [tuple(r) for r in in_paper] == [("asr", 0.25, COMP_EXPORT), ("asr", 0.5, ""), ("asr", 0.75, ""),
                                            ("asr_first_turn", -1.0, COMP_OTHER), ("asr_held_in", -1.0, COMP_OTHER)]
    # every in-paper row is an owner row and vice versa for a selected run
    assert conn.execute("SELECT COUNT(*) FROM v_paper_long WHERE in_current_paper = 1 AND metric_role <> 'owner'").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM v_paper_long WHERE run_id = ? AND metric_role = 'owner' AND in_current_paper = 0", (lora,)).fetchone()[0] == 0
    # the frame helper follows the flag and carries the role
    df = paper.paper_long_frame(conn)
    sub = df[df["run_id"] == lora]
    assert len(sub) == 5 and not sub.duplicated(["run_id", "judgment_id", "metric_id", "threshold"]).any()
    assert set(sub["metric_role"]) == {"owner"}
    assert {"metric_role", "metric_source_id", "run_source_id", "metric_role_candidates", "metric_role_contested"} <= set(paper.LONG_REGISTRY_COLUMNS)
    full = paper.paper_long_frame(conn, current_only=False)
    assert sorted(full[full["run_id"] == lora]["metric_role"].value_counts().to_dict().items()) == [("owner", 5), ("secondary", 3)]


def test_secondary_rows_do_not_move_the_paper_inputs_head_and_the_columns_are_declared(corpus):
    root, conn, lora, jid = corpus
    head = conn.execute("SELECT ingest_seq FROM v_paper_inputs_head").fetchone()[0]
    assert head is not None
    owner_max = conn.execute("SELECT MAX(ingest_seq) FROM v_paper_long WHERE in_current_paper = 1").fetchone()[0]
    assert head >= owner_max
    cols = [r[1] for r in conn.execute("PRAGMA table_info(v_paper_long)")]
    assert cols.index("metric_role") < cols.index("ingest_seq") and cols[1] == "in_current_paper"
    assert "source_revision" in S.VIEW_REFERENCES["v_paper_long"] and "observation" in S.VIEW_REFERENCES["v_paper_long"]
