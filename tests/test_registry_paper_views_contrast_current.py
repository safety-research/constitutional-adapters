"""The paper frame joins a metric row to its CURRENT contrasts only (migration 0017; `schema_def.WITHDRAWN_CONTRAST_CTE` /
`CURRENT_CONTRAST_CTE`).

`run_contrast`'s key carries `anchor_run_id`, so a reader whose newer accepted revision contrasts a run against a DIFFERENT anchor than
an earlier revision did leaves both rows in the table; the fold withdraws the older key's OBSERVATION (`current` 0,
`withdrawn_by_revision_id`), exactly as for a metric row, and before 0017 `v_paper_long` joined `run_contrast` with no currency filter,
so the metric row appeared once per anchor (376 keys of 194 runs on the live registry, 2026-09-26). The rule: a contrast is current when
no observation of its key is withdrawn -- some source's head accepted revision still carries the (run, anchor) pair, or no source ever
observed it. Three synthetic cases and the live invariant:

* two revisions of one source, the second against ANOTHER anchor: one row, the newer anchor (the first is withdrawn);
* one revision that files two contrasts of different kinds against two anchors: both current, both joined (the export's half-dose runs
  carry a `same_standup_none` and a `full_dose` contrast at the headline threshold by design);
* a second revision that files no contrast at all: the metric row stays, its contrast columns NULL;
* the code's SQL as a query over a synthetic corpus that exercises the rule: no withdrawn anchor, every (metric key, anchor) pair once;
* live (read-only, `live_db`): the same two invariants over ONE snapshot of the deployed `v_paper_long` -- rows and withdrawn keys
  from one read transaction -- shared with the spelling module through `tests/registry_live_fixtures.py`; an invariant of the rule,
  not today's census. When the code's text equals the deployed text
  (`db.ddl_master` vs `db.code_ddl_master`) the scan is the code's SQL too; when they differ under a PENDING migration the candidate
  is judged on the synthetic corpus above and the live rows are held to the deployed object's invariants, chosen by the deployed
  version -- never a second live scan, never an equality assertion that would block the migration; a difference with no pending
  migration is refused by the fixture as schema drift.
"""
from __future__ import annotations

import os
from collections import Counter

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import synth

import registry_live_fixtures as F   # the shared live snapshot (session fixture `live_paper_long`, registered by conftest) and the withdrawn-key oracle

pytestmark = pytest.mark.live_db   # this module opens the live registry: land.sh's local step runs it (`pytest -m live_db`)

SRC = "seedrep:g"
KEY = ("run_id", "judgment_id", "metric_id", "threshold", "computation_id")


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _roster_entry(i, selector):
    p = {"roster_entry_id": f"re{i}", "round_id": "r_synth", "comparison_id": "fig_synth", "model_id": synth.MODEL_ID,
         "eval_id": synth.EVAL_ID, "instrument_id": synth.INSTRUMENT_ID, "judge_id": synth.JUDGE_ID, "strength_rule": "operating_point",
         "source_citation": "paperlib.config.SYNTHETIC", "inferred": 0, "object_selector_json": selector}
    return synth.rec("roster_entry", p, synth.t(50 + i))


def _rev(rid, i):
    return synth.rec("source_revision", {"revision_id": rid, "source_id": SRC, "adapter": "seedrep", "observed_utc": synth.t(i),
                                         "inputs_json": {"a": "%064x" % i}, "accepted": 1, "priority": 0}, synth.t(i))


def _obs(rid, keys, i, table):
    return synth.rec("observation", {"source_id": SRC, "revision_id": rid, "table_name": table,
                                     "observation_keys": [I.canonical_json(k) for k in keys]}, synth.t(i))


def _contrast_key(run_id, anchor, jid):
    return {"run_id": run_id, "anchor_run_id": anchor, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5, "computation_id": ""}


def _contrast(run_id, anchor, jid, kind, delta, i):
    return synth.rec("run_contrast", {"run_id": run_id, "anchor_run_id": anchor, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5,
                                      "anchor_kind": kind, "delta": delta, "delta_method": "exact McNemar", "n_pairs": 4,
                                      "computed_by": "registry"}, synth.t(i))


def _withdrawal(key, frm, to, i):
    return synth.rec("withdrawal", {"source_id": SRC, "observation_key": I.canonical_json(key), "from_revision_id": frm, "to_revision_id": to,
                                    "table_name": "run_contrast", "reason": "removed_measurement"}, synth.t(i))


def _revision_file(root, name, rev, i0, run_id, contrasts, withdrawals=()):
    """One accepted revision of the source: the marker, the run key, the contrast keys it carries, the rows, the withdrawals it declares."""
    keys = [_contrast_key(c["payload"]["run_id"], c["payload"]["anchor_run_id"], c["payload"]["judgment_id"]) for c in contrasts]
    recs = [_rev(rev, i0), _obs(rev, [{"run_id": run_id}], i0 + 1, "run")]
    if keys:
        recs.append(_obs(rev, keys, i0 + 2, "run_contrast"))
    recs += list(contrasts) + list(withdrawals)
    _write(root, "synth_sources", name, recs)


def _corpus(tmp_path, n_runs):
    """run 0 = none, run 1 = constitution_lora (selected by the roster), run 2 = none."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=n_runs, n_items=4)
    _write(root, "roster", "000001-roster.jsonl", [_roster_entry(1, {"paper_label": "constitution_lora"})])
    return root, ids


def _ingest(root):
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, (rep.refused, rep.blocked)


def _frame_rows(conn, run_id):
    return [dict(r) for r in conn.execute("SELECT * FROM v_paper_long WHERE run_id = ? AND metric_id = 'asr' AND threshold = 0.5 ORDER BY anchor_run_id",
                                          (run_id,))]


def _observation_current(conn, key):
    return [int(r[0]) for r in conn.execute("SELECT current FROM observation WHERE table_name = 'run_contrast' AND observation_key = ? ORDER BY ingest_seq",
                                            (I.canonical_json(key),))]


def test_a_re_anchored_run_joins_the_newer_anchor_once(tmp_path):
    """Revision 1 contrasts the lora run against run 0's none; revision 2 against run 2's none (the live fallback -> own-none switch).
    Both rows stand in `run_contrast`; the first key's observation is withdrawn; the frame carries ONE row, the newer anchor, and the
    row's `ingest_seq` is at least the newer contrast's."""
    root, ids = _corpus(tmp_path, 3)
    lora, jid = ids["run_ids"][1], ids["judgment_ids"][1]
    old_anchor, new_anchor = ids["run_ids"][0], ids["run_ids"][2]
    _revision_file(root, "000001-rev1.jsonl", "rev1", 9000, lora, [_contrast(lora, old_anchor, jid, "same_standup_none", -0.10, 9010)])
    _revision_file(root, "000002-rev2.jsonl", "rev2", 9100, lora, [_contrast(lora, new_anchor, jid, "same_standup_none", -0.05, 9110)],
                   [_withdrawal(_contrast_key(lora, old_anchor, jid), "rev1", "rev2", 9120)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        assert tuple(conn.execute("SELECT COUNT(*), COUNT(DISTINCT anchor_run_id) FROM run_contrast WHERE run_id = ?", (lora,)).fetchone()) == (2, 2)
        assert _observation_current(conn, _contrast_key(lora, old_anchor, jid)) == [0]
        assert _observation_current(conn, _contrast_key(lora, new_anchor, jid)) == [1]
        wd = conn.execute("SELECT withdrawn_by_revision_id FROM observation WHERE table_name = 'run_contrast' AND observation_key = ?",
                          (I.canonical_json(_contrast_key(lora, old_anchor, jid)),)).fetchone()[0]
        assert wd == "rev2"
        rows = _frame_rows(conn, lora)
        assert len(rows) == 1, [(r["anchor_run_id"], r["anchor_kind"]) for r in rows]
        row = rows[0]
        assert row["anchor_run_id"] == new_anchor and row["anchor_kind"] == "same_standup_none" and row["delta"] == pytest.approx(-0.05)
        assert row["in_current_paper"] == 1 and row["metric_role"] == "owner"
        new_seq = conn.execute("SELECT ingest_seq FROM run_contrast WHERE run_id = ? AND anchor_run_id = ?", (lora, new_anchor)).fetchone()[0]
        old_seq = conn.execute("SELECT ingest_seq FROM run_contrast WHERE run_id = ? AND anchor_run_id = ?", (lora, old_anchor)).fetchone()[0]
        assert row["ingest_seq"] >= new_seq > old_seq
        # the run's other frame rows (none) are untouched, and the paper counts the measurement once
        assert conn.execute("SELECT SUM(in_current_paper) FROM v_paper_long WHERE run_id = ?", (lora,)).fetchone()[0] == 1
    finally:
        conn.close()


def test_two_current_contrasts_of_different_kinds_are_both_joined(tmp_path):
    """One revision files a `same_standup_none` contrast against run 0 AND a `full_dose` contrast against run 2 (the export's half-dose
    shape; run 2 is another unsteered run standing in for the full-dose sibling -- a second lora run would supersede run 1 in
    v_paper_current, which is the selection rule's business, not the contrast join's). Both observations are current, so the frame
    carries one row per anchor -- distinct kinds, kept apart by `anchor_kind`."""
    root, ids = _corpus(tmp_path, 3)
    lora, jid = ids["run_ids"][1], ids["judgment_ids"][1]
    none_anchor, dose_anchor = ids["run_ids"][0], ids["run_ids"][2]
    _revision_file(root, "000001-rev1.jsonl", "rev1", 9000, lora, [_contrast(lora, none_anchor, jid, "same_standup_none", -0.10, 9010),
                                                                   _contrast(lora, dose_anchor, jid, "full_dose", 0.02, 9011)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        assert _observation_current(conn, _contrast_key(lora, none_anchor, jid)) == [1]
        assert _observation_current(conn, _contrast_key(lora, dose_anchor, jid)) == [1]
        # both anchors are current (migration 0021): the roster names no unsteered entry, so neither unsteered run supersedes the other
        assert {r[0]: r[1] for r in conn.execute("SELECT run_id, is_current FROM v_run_current WHERE run_id IN (?, ?)", (none_anchor, dose_anchor))} == {none_anchor: 1, dose_anchor: 1}
        rows = _frame_rows(conn, lora)
        assert sorted((r["anchor_run_id"], r["anchor_kind"]) for r in rows) == sorted([(none_anchor, "same_standup_none"), (dose_anchor, "full_dose")])
        assert {r["metric_role"] for r in rows} == {"owner"} and {r["in_current_paper"] for r in rows} == {1}
        assert {r["n_current_contrasts"] for r in rows} == {2}                           # Both current contrasts counted at the key, no tie-break
    finally:
        conn.close()


def test_a_revision_that_drops_the_contrast_leaves_the_metric_row_with_null_contrast_columns(tmp_path):
    """Revision 2 carries the run and no contrast: the withdrawn contrast is left out, the metric row stays (the LEFT JOIN), and every
    contrast column of the row reads NULL -- the 584 live keys whose only contrast is withdrawn."""
    root, ids = _corpus(tmp_path, 3)
    lora, jid = ids["run_ids"][1], ids["judgment_ids"][1]
    anchor = ids["run_ids"][0]
    _revision_file(root, "000001-rev1.jsonl", "rev1", 9000, lora, [_contrast(lora, anchor, jid, "same_standup_none", -0.10, 9010)])
    _revision_file(root, "000002-rev2.jsonl", "rev2", 9100, lora, [], [_withdrawal(_contrast_key(lora, anchor, jid), "rev1", "rev2", 9120)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM run_contrast WHERE run_id = ?", (lora,)).fetchone()[0] == 1
        assert _observation_current(conn, _contrast_key(lora, anchor, jid)) == [0]
        rows = _frame_rows(conn, lora)
        assert len(rows) == 1
        row = rows[0]
        assert row["value"] == pytest.approx(0.25) and row["in_current_paper"] == 1
        for col in ("anchor_run_id", "anchor_kind", "delta", "delta_lo", "delta_hi", "delta_p", "delta_method", "effect_size", "effect_kind",
                    "n_pairs", "contrast_extra_json", "anchor_token_cap_tokens", "cap_matched"):
            assert row[col] is None, (col, row[col])
    finally:
        conn.close()


def test_a_natively_emitted_contrast_has_no_observation_and_stays(tmp_path):
    """A contrast no adapter observed (the corpus's own emitter files it) is never withdrawn: the rule reads the observation table, and a key
    with no observation is current."""
    root, ids = _corpus(tmp_path, 3)
    lora, jid = ids["run_ids"][1], ids["judgment_ids"][1]
    anchor = ids["run_ids"][0]
    _write(root, "synth_sources", "000001-native.jsonl", [_contrast(lora, anchor, jid, "same_standup_none", -0.10, 9010)])
    _ingest(root)
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM observation WHERE table_name = 'run_contrast'").fetchone()[0] == 0
        rows = _frame_rows(conn, lora)
        assert [(r["anchor_run_id"], r["anchor_kind"], r["n_current_contrasts"]) for r in rows] == [(anchor, "same_standup_none", 1)]
    finally:
        conn.close()


def test_the_rule_is_declared_where_the_frame_reads_it():
    sql = S.VIEW_SQL["v_paper_long"]
    assert S.WITHDRAWN_CONTRAST_CTE in sql and S.CURRENT_CONTRAST_CTE in sql
    assert "LEFT JOIN current_contrast rc ON" in sql and "LEFT JOIN run_contrast rc" not in sql
    assert "table_name = 'run_contrast' GROUP BY observation_key HAVING MAX(current) = 0" in S.WITHDRAWN_CONTRAST_CTE
    assert {"run_contrast", "observation"} <= set(S.VIEW_REFERENCES["v_paper_long"])
    # migration 0021: the anchor must be current, read from the ONE pass over the selection the frame rows read too
    assert S.PAPER_CURRENT_CTE in sql and sql.index(S.PAPER_CURRENT_CTE) < sql.index(S.CURRENT_CONTRAST_CTE)
    assert "JOIN paper_current ac ON ac.run_id = rc.anchor_run_id AND ac.is_current = 1" in S.CURRENT_CONTRAST_CTE
    assert "COUNT(*) OVER (PARTITION BY rc.run_id, rc.judgment_id, rc.metric_id, rc.threshold, rc.computation_id) AS n_current_contrasts" in S.CURRENT_CONTRAST_CTE
    assert "\nFROM paper_current pc\n" in sql and "FROM v_paper_current pc" not in sql and sql.count("FROM v_paper_current") == 1
    assert "rc.n_current_contrasts," in sql
    assert "v_paper_current" in S.VIEW_REFERENCES["v_paper_long"] and "v_run_current" in S.VIEW_REFERENCES["v_paper_current"]


# ---- the two invariants: over the code's SQL on a synthetic corpus, and over the shared live snapshot of the deployed object ---------
CONTRAST_RULE_VERSION = 17   # migrations/0017_paper_long_current_contrast.sql: from this version the deployed frame joins no withdrawn anchor


def _frame_select_sql() -> str:
    """The code's `v_paper_long` as a query (the CREATE prefix stripped): the rule as the code states it, whatever the deployed object says."""
    sql = S.VIEW_SQL["v_paper_long"].strip().rstrip(";")
    prefix = "CREATE VIEW v_paper_long AS"
    assert sql.startswith(prefix), sql[:60]
    return sql[len(prefix):]


def _pairs_from_rows(rows):
    """(metric key, anchor) of every row that carries an anchor, plus the row count."""
    return len(rows), [F.anchor_key(r) for r in rows if r["anchor_run_id"] is not None]


def _pairs(conn, source_sql: str):
    """The same over the rows of `source_sql` run as a query."""
    cur = conn.execute(f"SELECT run_id, judgment_id, metric_id, threshold, computation_id, anchor_run_id FROM ({source_sql})")
    names = [d[0] for d in cur.description]
    return _pairs_from_rows([dict(zip(names, r)) for r in cur])


def _assert_invariants(label: str, n: int, pairs: list, withdrawn, *, hold_to_rule: bool = True) -> None:
    """(a) every (run, judgment, metric, threshold, computation, anchor) once -- the key of `run_contrast`, so a metric row never joins the
    same contrast twice; (b) no anchor a withdrawn observation names is joined (the rule) -- asserted only where the object states the rule."""
    assert n > 0, label
    dup = sorted(p for p, c in Counter(pairs).items() if c > 1)
    assert not dup, (label, dup[:5])
    if hold_to_rule:
        stale = sorted(set(pairs) & set(withdrawn))
        assert not stale, (label, len(stale), stale[:5])


def test_the_code_frame_holds_both_invariants_on_a_synthetic_corpus(tmp_path):
    """The CANDIDATE SQL, run as a query over a synthetic corpus that exercises the rule -- a re-anchored run, its first anchor withdrawn
    (the observation key is (run, anchor, judgment, metric, threshold, computation), so a later contrast of ANY kind against that anchor
    would re-observe the key as current; the second revision names the new anchor only) -- carries no withdrawn anchor and every
    (metric key, anchor) once. This is the check the live test falls back to when the deployed object's text differs from the code's
    under a pending migration: the candidate is never run against the live database a second time."""
    root, ids = _corpus(tmp_path, 3)
    lora, jid = ids["run_ids"][1], ids["judgment_ids"][1]
    a0, a2 = ids["run_ids"][0], ids["run_ids"][2]
    _revision_file(root, "000001-rev1.jsonl", "rev1", 9000, lora, [_contrast(lora, a0, jid, "same_standup_none", -0.10, 9010)])
    _revision_file(root, "000002-rev2.jsonl", "rev2", 9100, lora, [_contrast(lora, a2, jid, "same_standup_none", -0.05, 9110)],
                   [_withdrawal(_contrast_key(lora, a0, jid), "rev1", "rev2", 9120)])
    _ingest(root)
    with DB.reading(DB.db_path(root)) as conn:
        withdrawn = F.withdrawn_anchor_keys(conn)
        assert withdrawn == {(lora, jid, "asr", 0.5, "", a0)}
        n, pairs = _pairs(conn, _frame_select_sql())
        assert (lora, jid, "asr", 0.5, "", a2) in pairs and (lora, jid, "asr", 0.5, "", a0) not in pairs
        _assert_invariants("code", n, pairs, withdrawn)
        # the guard itself fires on a stale join: the withdrawn pair smuggled into the list is caught
        with pytest.raises(AssertionError):
            _assert_invariants("code", n, pairs + [(lora, jid, "asr", 0.5, "", a0)], withdrawn)
        with pytest.raises(AssertionError):
            _assert_invariants("code", n, pairs + [pairs[0]], withdrawn)
        # the same rows through the deployed object of THIS database (the code's schema): identical pairs
        n2, pairs2 = _pairs(conn, "SELECT * FROM v_paper_long")
        assert (n2, sorted(pairs2)) == (n, sorted(pairs))


def test_live_frame_carries_each_current_anchor_once_and_no_withdrawn_one(live_paper_long):
    """INVARIANT, not census, over the one shared live snapshot (registry_live_fixtures): (a) every (run, judgment, metric, threshold,
    computation, anchor) appears once; (b) no anchor a withdrawn observation names is joined. The withdrawn keys come from the SAME
    read transaction as the rows -- never a second connection. The deployed object is held to (b) from migration 0017 on, chosen by
    the database's version in that snapshot (never by a substring of the candidate's text); before 0017 it joined every anchor by
    design. When the code's text is the deployed text this snapshot is the code's too; otherwise (a pending migration -- the fixture
    refuses plain drift) the candidate's invariants are the synthetic test's."""
    n, pairs = _pairs_from_rows(live_paper_long.rows)
    states_rule = live_paper_long.migration_version >= CONTRAST_RULE_VERSION
    print(f"\n{live_paper_long.status_line()}; rows={n} anchored={len(pairs)} withdrawn_keys={len(live_paper_long.withdrawn_anchor_keys)} "
          f"deployed_states_rule={states_rule}")
    _assert_invariants("deployed", n, pairs, live_paper_long.withdrawn_anchor_keys, hold_to_rule=states_rule)
    if live_paper_long.candidate_is_deployed:
        assert states_rule, (f"the code's SQL is deployed and states the rule (test_the_rule_is_declared_where_the_frame_reads_it), "
                             f"so the database must record migration {CONTRAST_RULE_VERSION}")
    else:
        print("candidate SQL differs from the deployed object (a migration is pending): its invariants are asserted by "
              "test_the_code_frame_holds_both_invariants_on_a_synthetic_corpus, not by a second live scan")
