"""`is_complete` and the contrast cap pair (migration 0006; schema_def.is_complete_sql, v_paper_long's `anchor_token_cap_tokens` /
`cap_matched`).

Completion. The lifecycle is queued -> running -> generated -> (judging) -> judged -> ingested; `ingested` FOLLOWS `judged` and means
the run is complete and stamped (the harvester stamps every complete run `ingested`, S6), so at rest a complete row reads `ingested`
and `judged` is the transient state before the stamp. Two consumers filtered `lifecycle_state = 'judged'` for finished runs on
2026-09-21 and found nothing. `is_complete` (1 when lifecycle_state IN ('judged', 'ingested'), else 0) now stands beside
`lifecycle_state` in every consumer-facing object that exposes it: v_run_wide, v_producer_key, v_paper_current, v_paper_long.

Cap pair. `run_contrast` pairs a run with an anchor run and v_paper_long surfaces the delta, but a consumer reading it never joined
both runs' caps, so a pair whose caps differ was invisible: the paired dose ladder differenced cap-512 steered runs against cap-2048
anchors on 3 of 5 steps (it files no run_contrast row yet), and 202 of the live registry's 8,640 filed contrasts pair different caps
(every one an escalation-ladder pair, most often a cap-8192 run against a cap-4096 or cap-2048 anchor; 2026-09-21). v_paper_long now
carries `anchor_token_cap_tokens` (the anchor
run's cap) and `cap_matched` (1 equal, 0 different, NULL when the row carries no anchor or either cap is NULL; `run.token_cap_tokens`
is NOT NULL, so on the frame NULL means no anchor).

Synthetic corpus (invented ids; hostname TESTHOST): three complete runs of one instrument — the none object at cap 512 (the anchor),
constitution_lora at cap 512 (a matched contrast against it), constitution_lora at cap 2048 (a mismatched contrast against it: the
ladder shape on the live registry, one instrument and the run's own cap per step); the anchor's own metric row has no contrast.
"""
from __future__ import annotations

import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import synth
from sisteer.registry import vocab as V

COMPLETION_OBJECTS = ("v_run_wide", "v_producer_key", "v_paper_current", "v_paper_long")
LORA_OBJECT = "constitution_lora@deadbeef"          # the corpus writer's producer object key for the constitution_lora object
LORA_2048_RUN = "synthstore_cap2048"                # the producer run key of the third run, in the corpus writer's spelling


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _lora_run_at_cap(run_key: str, i0: int, cap: int):
    """A complete constitution_lora run whose run record carries ``cap`` (the corpus writer fixes every run at the instrument's 512)."""
    rid, recs = synth.run_records(run_key, i0, LORA_OBJECT, synth.OBJECT_SHA, 4)
    payload = synth.run_payload(run_key, LORA_OBJECT, synth.OBJECT_SHA, i0, token_cap_tokens=cap)
    assert payload["run_id"] == rid and recs[0]["record_type"] == "run"
    recs[0] = synth.rec("run", payload, synth.t(i0))
    return rid, recs


def _contrast(run_id: str, anchor_run_id: str, judgment_id: str, delta: float, i: int):
    return synth.rec("run_contrast", {"run_id": run_id, "anchor_run_id": anchor_run_id, "judgment_id": judgment_id, "metric_id": "asr",
                                      "threshold": 0.5, "anchor_kind": "same_standup_none", "delta": delta, "delta_method": "exact McNemar",
                                      "n_pairs": 4, "computed_by": "producer"}, synth.t(i))


def _columns(conn, name: str):
    return [r[1] for r in conn.execute(f"PRAGMA table_info({name})").fetchall()]


@pytest.fixture()
def corpus(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)                      # run 0 = none (cap 512), run 1 = constitution_lora (cap 512)
    none_run, lora_run = ids["run_ids"]
    lora_2048, recs = _lora_run_at_cap(LORA_2048_RUN, 300, 2048)
    synth.write_run_dir(root, synth.CAMPAIGN, lora_2048, [(str(r["record_type"]), [r]) for r in recs])
    jid_512, jid_2048 = ids["judgment_ids"][1], synth.identity.judgment_id(lora_2048, synth.JUDGE_ID)
    _write(root, "contrasts", "000001-contrasts.jsonl", [_contrast(lora_run, none_run, jid_512, -0.25, 400),
                                                          _contrast(lora_2048, none_run, jid_2048, -0.10, 401)])
    res = H.ingest_once(root, workers=0, min_free=0)
    assert res.refused == 0
    return root, {"none": none_run, "lora_512": lora_run, "lora_2048": lora_2048}


def _long(conn, run_id: str) -> dict:
    rows = [dict(r) for r in conn.execute("SELECT * FROM v_paper_long WHERE run_id = ? AND metric_id = 'asr'", (run_id,))]
    assert len(rows) == 1, rows
    return rows[0]


def test_the_new_columns_stand_beside_lifecycle_state_and_the_contrast_columns():
    conn = DB.connect_memory()
    conn.executescript(S.render_schema_sql())
    for name in COMPLETION_OBJECTS:
        cols = _columns(conn, name)
        assert "lifecycle_state" in cols and "is_complete" in cols, name
    long_cols = _columns(conn, "v_paper_long")
    assert long_cols.index("anchor_run_id") < long_cols.index("anchor_token_cap_tokens") < long_cols.index("cap_matched")
    # the flag is a function of the lifecycle vocabulary and names exactly the two complete states
    assert S.COMPLETE_LIFECYCLE_STATES == ("judged", "ingested") and set(S.COMPLETE_LIFECYCLE_STATES) <= set(V.LIFECYCLE_STATES)
    assert S.is_complete_sql("r") == "CASE WHEN r.lifecycle_state IN ('judged', 'ingested') THEN 1 ELSE 0 END AS is_complete"
    assert S.is_complete_sql().startswith("CASE WHEN lifecycle_state IN (")
    # run_contrast is untouched: the pair is derived in the frame from the anchor run's row
    assert "anchor_token_cap_tokens" not in S.TABLE_BY_NAME["run_contrast"].column_names
    assert "cap_matched" not in S.TABLE_BY_NAME["run_contrast"].column_names


def test_cap_matched_reads_the_anchor_runs_cap_matched_mismatched_and_absent(corpus):
    root, ids = corpus
    conn = DB.connect(DB.db_path(root), readonly=True)
    caps = {r[0]: r[1] for r in conn.execute("SELECT run_id, token_cap_tokens FROM run")}
    assert caps[ids["none"]] == 512 and caps[ids["lora_512"]] == 512 and caps[ids["lora_2048"]] == 2048
    matched = _long(conn, ids["lora_512"])
    assert matched["anchor_run_id"] == ids["none"] and matched["delta"] == -0.25
    assert matched["token_cap_tokens"] == 512 and matched["anchor_token_cap_tokens"] == 512 and matched["cap_matched"] == 1
    mismatched = _long(conn, ids["lora_2048"])
    assert mismatched["anchor_run_id"] == ids["none"] and mismatched["delta"] == -0.10
    assert mismatched["token_cap_tokens"] == 2048 and mismatched["anchor_token_cap_tokens"] == 512 and mismatched["cap_matched"] == 0
    absent = _long(conn, ids["none"])                                        # the anchor's own metric row carries no contrast
    assert absent["anchor_run_id"] is None and absent["delta"] is None
    assert absent["anchor_token_cap_tokens"] is None and absent["cap_matched"] is None
    # the audit's query, as a consumer would write it off the frame
    dist = dict(conn.execute("SELECT COALESCE(CAST(cap_matched AS TEXT), 'null'), COUNT(*) FROM v_paper_long GROUP BY 1").fetchall())
    assert dist == {"1": 1, "0": 1, "null": 1}
    conn.close()


def test_cap_matched_is_null_when_either_cap_is_null_by_its_own_arithmetic():
    """`run.token_cap_tokens` is NOT NULL, so the frame reaches NULL only through a missing anchor; the CASE is still written over both
    caps, and this pins that arithmetic on the rendered SQL rather than trusting the constraint."""
    sql = S.VIEW_SQL["v_paper_long"]
    assert "CASE WHEN r.token_cap_tokens IS NULL OR ra.token_cap_tokens IS NULL THEN NULL" in sql
    assert "WHEN r.token_cap_tokens = ra.token_cap_tokens THEN 1 ELSE 0 END AS cap_matched" in sql
    assert "LEFT JOIN run ra ON ra.run_id = rc.anchor_run_id" in sql
    conn = DB.connect_memory()
    for a, b, want in ((512, 512, 1), (512, 2048, 0), (None, 512, None), (512, None, None), (None, None, None)):
        got = conn.execute("SELECT CASE WHEN ? IS NULL OR ? IS NULL THEN NULL WHEN ? = ? THEN 1 ELSE 0 END", (a, b, a, b)).fetchone()[0]
        assert got == want, (a, b, got)


def test_complete_runs_read_ingested_at_rest_and_is_complete_1(corpus):
    root, ids = corpus
    conn = DB.connect(DB.db_path(root), readonly=True)
    states = {r[0]: r[1] for r in conn.execute("SELECT run_id, lifecycle_state FROM run")}
    assert set(states.values()) == {"ingested"}                              # the harvester stamped every complete run
    assert conn.execute("SELECT COUNT(*) FROM v_run_wide WHERE lifecycle_state = 'judged'").fetchone()[0] == 0   # what the two consumers saw
    for name in COMPLETION_OBJECTS:
        rows = conn.execute(f"SELECT run_id, lifecycle_state, is_complete FROM {name}").fetchall()
        assert rows and all(r[1] == "ingested" and r[2] == 1 for r in rows), (name, rows)
    assert conn.execute("SELECT COUNT(*) FROM v_paper_long WHERE is_complete = 1").fetchone()[0] == 3
    conn.close()


@pytest.mark.parametrize("state", list(V.LIFECYCLE_STATES))
def test_is_complete_on_every_lifecycle_state(corpus, state):
    """Every state of the vocabulary, on every object that carries the flag: 1 for judged and ingested, 0 for the rest. The run is
    moved directly (an ingested run carries every reached state's utc, so the state triggers admit each move; `finished_utc` is set
    for the terminal states)."""
    root, ids = corpus
    rid = ids["lora_512"]
    conn = DB.connect(DB.db_path(root))
    conn.execute("UPDATE run SET lifecycle_state = ?, finished_utc = ? WHERE run_id = ?", (state, synth.t(999), rid))
    conn.commit()
    want = 1 if state in ("judged", "ingested") else 0
    for name in COMPLETION_OBJECTS:
        rows = conn.execute(f"SELECT lifecycle_state, is_complete FROM {name} WHERE run_id = ?", (rid,)).fetchall()
        assert rows, (name, state)
        assert all(r[0] == state and r[1] == want for r in rows), (name, state, rows)
    # the two spellings of the same filter agree
    assert conn.execute("SELECT COUNT(*) FROM v_run_wide WHERE is_complete = 1").fetchone()[0] == \
        conn.execute("SELECT COUNT(*) FROM run WHERE lifecycle_state IN ('judged', 'ingested')").fetchone()[0]
    conn.close()
