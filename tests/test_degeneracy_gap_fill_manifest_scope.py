"""gap_fill manifest scope: `degeneracy_gap_fill.py manifest` defines the
degeneracy gap as the in-paper asr/acc rows without a union5 share, so a model outside the paper frame (no roster entry ->
in_current_paper 0 on every row — a front-door model) is invisible to it and its backfill could not be manifested. The gap stays
defined in the one instrument: `manifest --model <key> --all-current` lists every v_run_current is_current=1 run of the model whose metric
rows carry no union5 share, in-paper or not; `--model` alone keeps the in-paper rule; `--all-current` without `--model` refuses.

A synthetic registry at the real schema through the real harvester: the synthetic model is in no roster, so the default manifest is EMPTY for it
(that class) and the model-scoped all-current manifest carries its judged run with the asr metric row."""
from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from sisteer.registry import db as DB  # noqa: E402
from sisteer.registry import harvest as H  # noqa: E402
from sisteer.registry import identity  # noqa: E402
from sisteer.registry import records as R  # noqa: E402
from sisteer.registry import synth  # noqa: E402

SRC = REPO / "scripts/sdfbase/full22/degeneracy_gap_fill.py"


@pytest.fixture(scope="module")
def m():
    spec = importlib.util.spec_from_file_location("degeneracy_gap_fill_scope_under_test", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _registry(tmp_path: Path):
    """A judged run of the synthetic model (its asr metric row has no union5 share) + a failed run of the same model, folded by the real harvester."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    rid, recs = synth.run_records("k_scope", 100, through="judged")
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(rid), "000001-life.jsonl"), recs)
    dead, drecs = synth.run_records("k_dead", 300, through="running")
    drecs.append(synth.rec("run_state", {"run_id": dead, "state": "failed", "event_utc": synth.t(303), "evidence_kind": "job_log",
                                         "scheduler_state": "FAILED", "final": True}, synth.t(303)))
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(dead), "000001-life.jsonl"), drecs)
    # a CURRENT `running` registration of the model (the stale-registration class) -- never a manifest row, counted as excluded
    stale, srecs = synth.run_records("k_stale", 500, through="running")
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(stale), "000001-life.jsonl"), srecs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    return Path(DB.db_path(root)), rid, dead


def test_a_model_outside_the_paper_frame_is_invisible_to_the_in_paper_gap_and_listed_under_all_current(m, tmp_path):
    db, rid, dead = _registry(tmp_path)
    assert [r for r in m.manifest_from_registry(db) if r["model_id"] == synth.MODEL_ID] == []            # In_current_paper 0 -> not a gap row
    assert m.manifest_from_registry(db, model=synth.MODEL_ID) == []                                        # --model alone keeps the in-paper rule
    rows = m.manifest_from_registry(db, model=synth.MODEL_ID, all_current=True)
    assert [r["run_id"] for r in rows] == [rid] and rows[0]["manifest_scope"] == f"all_current:{synth.MODEL_ID}"
    assert [(x["metric_id"], x["threshold"]) for x in rows[0]["metric_rows"]] == [("asr", 0.5)] and rows[0]["judgment_ids"] == [identity.judgment_id(rid, synth.JUDGE_ID)]
    assert rows[0]["model_id"] == synth.MODEL_ID and rows[0]["eval_id"] and rows[0]["n_items_judged"] == 200
    assert dead not in [r["run_id"] for r in rows]                                                         # a failed run is not current (and carries no metric)
    # the current `running` registration is excluded by the explicit predicate and SAID, per reason
    stale = synth.run_records("k_stale", 500, through="running")[0]
    assert stale not in [r["run_id"] for r in rows]
    assert m.MANIFEST_EXCLUDED == {"n": 1, "by_reason": {"lifecycle:running": 1}, "run_ids": [stale]}, m.MANIFEST_EXCLUDED
    assert "lifecycle_state IN ('judged', 'ingested')" in m.CURRENT_SQL and "store_uri IS NOT NULL" in m.CURRENT_SQL
    assert m.manifest_from_registry(db, model="no_such_model", all_current=True) == []


def test_all_current_without_a_model_refuses_and_the_cli_writes_the_scoped_manifest(m, tmp_path, capsys):
    db, rid, dead = _registry(tmp_path)
    with pytest.raises(SystemExit) as ei:
        m.manifest_from_registry(db, all_current=True)
    assert ei.value.code == 2 and "--all-current needs --model" in capsys.readouterr().err
    out = tmp_path / "man" / "manifest.jsonl"
    assert m.main(["manifest", "--registry", str(db), "--out", str(out), "--model", synth.MODEL_ID, "--all-current"]) == 0
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert [r["run_id"] for r in rows] == [rid] and rows[0]["manifest_scope"] == f"all_current:{synth.MODEL_ID}"
    assert f"manifest (all_current:{synth.MODEL_ID}): 1 runs, 1 metric rows; excluded 1 current row(s) (lifecycle:running 1)" in capsys.readouterr().out
    out2 = tmp_path / "man" / "default.jsonl"
    assert m.main(["manifest", "--registry", str(db), "--out", str(out2)]) == 0 and out2.read_text() == ""   # the in-paper gap: nothing for a frame-blind model
    assert "manifest (in_paper): 0 runs" in capsys.readouterr().out


def test_the_current_sql_mirrors_the_gap_columns_and_the_metric_set(m):
    cols = lambda sql: sql.split("FROM", 1)[0].split("SELECT", 1)[1].split()   # noqa: E731
    assert cols(m.CURRENT_SQL) == cols(m.GAP_SQL)                                                            # one row shape downstream
    assert "'over_refusal_graded'" in m.CURRENT_SQL and "'asr'" in m.CURRENT_SQL and "'acc'" in m.CURRENT_SQL
    assert "c.is_current = 1" in m.CURRENT_SQL and "r.model_id = ?" in m.CURRENT_SQL and "frac_degenerate_union5 IS NULL" in m.CURRENT_SQL
    assert "in_current_paper" not in m.CURRENT_SQL                                                           # the whole point: frame-blind rows are listed
    # 297-fix: eligibility reads the effective reading and the metric row's judgment currency, never the mirror column alone
    assert "v_run_degeneration_current" in m.CURRENT_SQL and "d.judgment_id = m.judgment_id OR d.judgment_id = ''" in m.CURRENT_SQL
    assert "LEFT JOIN judgment jm ON jm.judgment_id = m.judgment_id" in m.CURRENT_SQL and "jm.status" in m.CURRENT_SQL
    from sisteer.registry import schema_def as S
    assert S.non_current_judgment_sql("m.judgment_id", "jm.status") in m.CURRENT_SQL                           # the ONE rule, from schema_def
    # 297-fix-2: the in-paper manifest decides eligibility the same way
    assert "v_run_degeneration_current" in m.GAP_SQL and "d.judgment_id = m.judgment_id OR d.judgment_id = ''" in m.GAP_SQL
    assert "LEFT JOIN judgment jm ON jm.judgment_id = m.judgment_id" in m.GAP_SQL and S.non_current_judgment_sql("m.judgment_id", "jm.status") in m.GAP_SQL


# --------------------------------------------------------------------------------------------
# 297-fix: eligibility against the effective reading, over current judgments' rows, at the real schema
# --------------------------------------------------------------------------------------------

def _root(tmp_path: Path, sub: str) -> str:
    root = str(tmp_path / sub)
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    return root


def _write(root: str, rid: str, recs: list) -> None:
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(rid), "000001-life.jsonl"), recs)


def _ro(db: Path):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def test_a_measured_union5_in_the_current_view_excludes_the_run_despite_a_null_mirror(m, tmp_path):
    """The mirror column is the WRITER's statement; a MEASURED union5 in v_run_degeneration_current — under the metric row's own judgment, or
    the union reader's judgment-less ('') row — is the effective reading. Either excludes the run; a run with neither is the gap."""
    from sisteer.registry import hooks as RH
    root = _root(tmp_path, "reg_measured")
    rid, recs = synth.run_records("k_measured", 100, through="judged")
    jid = identity.judgment_id(rid, synth.JUDGE_ID)
    recs.append(synth.rec("run_degeneration", {**RH.measured_row(rid, "union5", "all", 7, 200, "replies", judgment_id=jid, computed_utc=synth.t(110)),
                                               "source_revision_id": ""}, synth.t(110)))
    _write(root, rid, recs)
    rid2, recs2 = synth.run_records("k_measured_jl", 120, through="judged")
    recs2.append(synth.rec("run_degeneration", {**RH.measured_row(rid2, "union5", "all", 3, 200, "replies", judgment_id="", computed_utc=synth.t(130)),
                                                "source_revision_id": ""}, synth.t(130)))
    _write(root, rid2, recs2)
    rid3, recs3 = synth.run_records("k_plain", 140, through="judged")              # the control: a NULL mirror and no reading -> the gap
    _write(root, rid3, recs3)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    db = Path(DB.db_path(root))
    conn = _ro(db)
    assert conn.execute("SELECT frac_degenerate_union5 FROM run_metric WHERE run_id = ?", (rid,)).fetchone()[0] is None     # the mirror stays NULL
    assert {r[0] for r in conn.execute("SELECT run_id FROM v_run_degeneration_current WHERE measure = 'union5' AND measurement_state = 'measured'")} == {rid, rid2}
    conn.close()
    rows = m.manifest_from_registry(db, model=synth.MODEL_ID, all_current=True)
    assert [r["run_id"] for r in rows] == [rid3]
    assert m.MANIFEST_EXCLUDED == {"n": 2, "by_reason": {"union5_already_measured": 2}, "run_ids": sorted([rid, rid2])}, m.MANIFEST_EXCLUDED


def test_null_mirror_rows_under_a_superseded_judgment_do_not_manifest_the_run(m, tmp_path):
    """run_metric keeps a superseded judgment's rows (currency: tests/test_worklist_supersedes_own_partial.py); eligibility reads only
    the rows of CURRENT judgments. Two runs: one whose current judgment's row has the mirror FILLED (the superseded row's NULL mirror must
    not re-open it), one whose current judgment filed no metric row at all."""
    root = _root(tmp_path, "reg_superseded")
    rid, recs = synth.run_records("k_sup_filled", 100, through="judged")               # J1 complete; its asr row's mirror NULL
    j1, j2 = identity.judgment_id(rid, synth.JUDGE_ID), identity.judgment_id(rid, synth.JUDGE_ID_2)
    recs.append(synth.rec("judgment", {"judgment_id": j2, "run_id": rid, "judge_id": synth.JUDGE_ID_2, "started_utc": synth.t(111),
                                       "finished_utc": synth.t(112), "n_items_judged": 200, "status": "complete", "api_pool": "lowprio"}, synth.t(112)))
    recs.append(synth.rec("run_metric", {"run_id": rid, "judgment_id": j2, "metric_id": "asr", "threshold": 0.5, "value": 0.3, "k": 60, "n": 200,
                                         "computed_by": "producer", "frac_degenerate_union5": 0.05}, synth.t(113)))          # the current row: mirror FILLED
    recs.append(synth.rec("judgment", {"judgment_id": j1, "run_id": rid, "judge_id": synth.JUDGE_ID, "status": "superseded",
                                       "superseded_by_judgment_id": j2}, synth.t(114)))
    _write(root, rid, recs)
    ridb, recsb = synth.run_records("k_sup_bare", 120, through="judged")               # J1 complete, superseded by a J2 with NO metric row
    j1b, j2b = identity.judgment_id(ridb, synth.JUDGE_ID), identity.judgment_id(ridb, synth.JUDGE_ID_2)
    recsb.append(synth.rec("judgment", {"judgment_id": j2b, "run_id": ridb, "judge_id": synth.JUDGE_ID_2, "started_utc": synth.t(131),
                                        "finished_utc": synth.t(132), "n_items_judged": 200, "status": "complete", "api_pool": "lowprio"}, synth.t(132)))
    recsb.append(synth.rec("judgment", {"judgment_id": j1b, "run_id": ridb, "judge_id": synth.JUDGE_ID, "status": "superseded",
                                        "superseded_by_judgment_id": j2b}, synth.t(133)))
    _write(root, ridb, recsb)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    db = Path(DB.db_path(root))
    conn = _ro(db)
    assert [conn.execute("SELECT status FROM judgment WHERE judgment_id = ?", (j,)).fetchone()[0] for j in (j1, j1b)] == ["superseded", "superseded"]
    assert conn.execute("SELECT COUNT(*) FROM run_metric WHERE run_id = ? AND frac_degenerate_union5 IS NULL", (rid,)).fetchone()[0] == 1   # J1's row kept
    assert conn.execute("SELECT COUNT(*) FROM run_metric WHERE run_id = ?", (ridb,)).fetchone()[0] == 1                                     # J1b's row only
    conn.close()
    assert m.manifest_from_registry(db, model=synth.MODEL_ID, all_current=True) == []
    assert m.MANIFEST_EXCLUDED == {"n": 2, "by_reason": {"mirror_filled": 1, "no_current_metric_row": 1}, "run_ids": sorted([rid, ridb])}, m.MANIFEST_EXCLUDED


def test_an_empty_store_uri_is_excluded_through_the_real_query(m, tmp_path):
    """Exercised through the query, not its text: a judged current run with an empty store_uri is never a manifest row."""
    root = _root(tmp_path, "reg_nostore")
    rid, recs = synth.run_records("k_nostore", 100, through="judged", store_uri="")
    _write(root, rid, recs)
    rid2, recs2 = synth.run_records("k_store", 120, through="judged")
    _write(root, rid2, recs2)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    db = Path(DB.db_path(root))
    conn = _ro(db)
    assert conn.execute("SELECT store_uri FROM run WHERE run_id = ?", (rid,)).fetchone()[0] in (None, "")
    conn.close()
    rows = m.manifest_from_registry(db, model=synth.MODEL_ID, all_current=True)
    assert [r["run_id"] for r in rows] == [rid2]
    assert m.MANIFEST_EXCLUDED == {"n": 1, "by_reason": {"no_store_uri": 1}, "run_ids": [rid]}, m.MANIFEST_EXCLUDED


# --------------------------------------------------------------------------------------------
# 297-fix-2: the IN-PAPER gap decides like the all-current one — over an in-paper world (the roster-include ruling of the
# cardinality module: every judged `none` run reads in_current_paper 1)
# --------------------------------------------------------------------------------------------

def test_the_in_paper_gap_excludes_a_measured_union5_and_a_superseded_judgments_rows(m, tmp_path, monkeypatch):
    import test_native_judging_cardinality as CARD
    from sisteer.registry import hooks as RH
    root, (ra, rb, rc) = CARD._registry(tmp_path, monkeypatch, [
        ("k_pa", "pasttense", "synthetic:///s/a", 12, "none", "none", None), ("k_pb", "pasttense", "synthetic:///s/b", 12, "none", "none", None),
        ("k_pc", "pasttense", "synthetic:///s/c", 12, "none", "none", None)])
    # the cardinality world mints GENERATED runs (its drivers judge them); a paper row needs the judged tail: judgment + state + asr row
    def judged_tail(rid: str, i0: int) -> list:
        jid = identity.judgment_id(rid, synth.JUDGE_ID)
        return [synth.rec("judgment", {"judgment_id": jid, "run_id": rid, "judge_id": synth.JUDGE_ID, "started_utc": synth.t(i0), "finished_utc": synth.t(i0 + 1),
                                       "n_items_judged": 12, "status": "complete", "api_pool": "lowprio"}, synth.t(i0 + 1)),
                synth.rec("run_state", {"run_id": rid, "state": "judged", "event_utc": synth.t(i0 + 2), "evidence_kind": "completion_artifact"}, synth.t(i0 + 2)),
                synth.rec("run_patch", {"run_id": rid, "fields": {"n_items_judged": 12, "judge_id": synth.JUDGE_ID}}, synth.t(i0 + 3)),
                synth.rec("run_metric", {"run_id": rid, "judgment_id": jid, "metric_id": "asr", "threshold": 0.5, "value": 0.25, "k": 3, "n": 12,
                                         "computed_by": "producer"}, synth.t(i0 + 4))]
    for k, rid in enumerate((ra, rb, rc)):
        R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(rid), "000001b-judged.jsonl"), judged_tail(rid, 700 + 10 * k))
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    db = Path(DB.db_path(root))
    conn = _ro(db)
    assert [conn.execute("SELECT in_current_paper FROM v_paper_current WHERE run_id = ?", (r,)).fetchone()[0] for r in (ra, rb, rc)] == [1, 1, 1]
    ja = conn.execute("SELECT judgment_id FROM judgment WHERE run_id = ?", (ra,)).fetchone()[0]
    jb, judge_b = conn.execute("SELECT judgment_id, judge_id FROM judgment WHERE run_id = ?", (rb,)).fetchone()
    conn.close()
    before = sorted(r["run_id"] for r in m.manifest_from_registry(db))
    assert before == sorted([ra, rb, rc])                                       # the control: three in-paper NULL-mirror rows are the gap
    # (a) ra: a MEASURED union5 under its own judgment, the mirror still NULL
    recs_a = [synth.rec("run_degeneration", {**RH.measured_row(ra, "union5", "all", 2, 12, "replies", judgment_id=ja, computed_utc=synth.t(900)),
                                             "source_revision_id": ""}, synth.t(900))]
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(ra), "000002-degen.jsonl"), recs_a)
    # (b) rb: its judgment superseded by a second judge's complete judgment whose metric row has the mirror FILLED
    j2 = identity.judgment_id(rb, synth.JUDGE_ID_2)
    recs_b = [synth.rec("judgment", {"judgment_id": j2, "run_id": rb, "judge_id": synth.JUDGE_ID_2, "started_utc": synth.t(901), "finished_utc": synth.t(902),
                                     "n_items_judged": 12, "status": "complete", "api_pool": "lowprio"}, synth.t(902)),
              synth.rec("run_metric", {"run_id": rb, "judgment_id": j2, "metric_id": "asr", "threshold": 0.5, "value": 0.25, "k": 3, "n": 12,
                                       "computed_by": "producer", "frac_degenerate_union5": 0.0}, synth.t(903)),
              synth.rec("judgment", {"judgment_id": jb, "run_id": rb, "judge_id": judge_b, "status": "superseded", "superseded_by_judgment_id": j2}, synth.t(904))]
    R.write_records(os.path.join(root, "records", synth.CAMPAIGN, identity.run_dir_name(rb), "000002-supersede.jsonl"), recs_b)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.as_dict()
    conn = _ro(db)
    assert conn.execute("SELECT frac_degenerate_union5 FROM run_metric WHERE run_id = ? AND judgment_id = ?", (ra, ja)).fetchone()[0] is None
    assert conn.execute("SELECT status FROM judgment WHERE judgment_id = ?", (jb,)).fetchone()[0] == "superseded"
    assert [conn.execute("SELECT in_current_paper FROM v_paper_current WHERE run_id = ?", (r,)).fetchone()[0] for r in (ra, rb)] == [1, 1]   # still paper rows
    conn.close()
    assert [r["run_id"] for r in m.manifest_from_registry(db)] == [rc]                        # the in-paper gap: rc only
    assert [r["run_id"] for r in m.manifest_from_registry(db, model=synth.MODEL_ID)] == [rc]
