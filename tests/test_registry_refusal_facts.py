"""Refusal facts in the record log (plan commit C6, section 2): every refusal JUDGMENT the harvester makes is a `record_refusal`
RECORD under records/harvester/<day>/ and a row of the ledger-scale table `record_refusal_fact` (one per refused record x judging stamp),
folded by its own apply path, so every mirror and every rebuild carry every refusal ever made while `record_refusal` keeps the current
judgment per record (stamped `judged_with`). Pinned here, on synthetic corpora:

1-3. a refused record becomes a fact in the cycle that refused it (the END pass), with the producer block re-read by sha, and the fact's
   RESOLUTION reads the record's key at its scale (`registry refusals explain`): a corrected run-scale record is a later row folded; a
   refused item then a corrected item likewise; an envelope refusal (no readable kind) is unknown;
4. rebuild equality with `judged_with` excluded (the stamp is the judging code's);
5-6. the kind is unledgered: `LEDGERED_RECORD_TYPES` leaves the two ledgers' CHECK lists byte-identical, and `registry migrate --plan` on
   a database at the shape one version below the newest file lists exactly one create (with its two indexes), the add-columns and the seven
   recreates -- no rebuild;
7. a restart between a refusal's commit and the fact pass: the next START pass writes exactly one fact; a third cycle none;
8. the backfill: a row refused before the stamp was kept becomes an `unverified` fact with `extra.backfilled = true`;
9. retry and backfill coexist in ONE cycle: N unstamped rows and one held constituent re-attempted under a moved stamp give N `unverified`
   facts and one new-stamp fact -- and when the cycle that made the previous judgment died before its END pass, the START pass makes that
   judgment a fact BEFORE the retry replaces the row (the crash-then-code-change variant);
10. a fact file written and not folded is folded by the next cycle's step 0: one fact row, no second file for the (sha, stamp), the first
   file in `record_delivery`;
plus the status page's section and harvester row, and the CLI's three actions.
"""
from __future__ import annotations

import json
import os
import random
import shutil

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import migrations as M
from sisteer.registry import records as R
from sisteer.registry import refusals as RF
from sisteer.registry import schema_def as S
from sisteer.registry import status
from sisteer.registry import synth
from sisteer.registry import validate as VAL
from sisteer.registry import vocab as V

from test_registry_migrations import AF_TABLES, FU4_TABLES, NTPL_TABLES, _sans_fu4_rebuild  # noqa: E402  -- migration 0027's run_contrast rebuild, 0029's five creates and 0031's two rebuilds, set aside


def _root(tmp_path, name="reg"):
    root = str(tmp_path / name)
    H.ensure_layout(root)
    return root


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _ingest(root, now):
    return H.ingest_once(root, workers=0, min_free=0, now=now)


def _q(root, sql, *params):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def _facts(root):
    return _q(root, "SELECT * FROM record_refusal_fact ORDER BY refused_utc, judged_with")


def _fact_files(root):
    base = os.path.join(root, "records", "harvester")
    return sorted(os.path.relpath(f, root) for f in R.iter_record_files(base) if os.path.basename(f).startswith(H.REFUSAL_FACT_FILE_PREFIX)) if os.path.isdir(base) else []


def _rebuild_equal(root, tmp_path, shuffle_seed=3):
    copy = str(tmp_path / f"copy{shuffle_seed}")
    files = list(R.iter_record_files(os.path.join(root, "records")))
    random.Random(shuffle_seed).shuffle(files)
    for f in files:
        dest = os.path.join(copy, "records", os.path.relpath(f, os.path.join(root, "records")))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(f, dest)
    out = str(tmp_path / f"rebuilt{shuffle_seed}.sqlite")
    H.rebuild(copy, out, workers=0)
    a = DB.connect(DB.db_path(root), readonly=True)
    b = DB.connect(out, readonly=True)
    try:
        diff = {k: v for k, v in H.verify_equal(a, b).items() if v}
        assert not diff, diff
        return b.execute("SELECT COUNT(*) FROM record_refusal_fact").fetchone()[0]
    finally:
        a.close()
        b.close()


def _bad_model(i, model_id="m_bad"):
    """A `model` record the vocabulary refuses (checkpoint_kind outside CHECKPOINT_KINDS)."""
    return synth.rec("model", {"model_id": model_id, "family": "gemma4", "size_label": "x", "checkpoint_kind": "chatty",
                               "hf_repo": "r", "hf_revision": "a" * 40, "chat_template_sha256": "b" * 64, "generation_config_present": 0,
                               "thinking_channel": "not_available", "paper_status": "in_paper"}, synth.t(i))


def _good_model(i, model_id="m_bad"):
    return synth.rec("model", {"model_id": model_id, "family": "gemma4", "size_label": "x", "checkpoint_kind": "chat",
                               "hf_repo": "r", "hf_revision": "a" * 40, "chat_template_sha256": "b" * 64, "generation_config_present": 0,
                               "thinking_channel": "not_available", "paper_status": "in_paper"}, synth.t(i))


def _no_facts(monkeypatch):
    """The refusal-fact passes disabled (the cycle dies before them, or runs under the earlier code)."""
    monkeypatch.setattr(H, "derive_refusal_fact_records", lambda root, conn, now, written: ([], 0))


# --------------------------------------------------------------------------------------------
# 1-4: the fact, its resolution at each scale, rebuild equality
# --------------------------------------------------------------------------------------------


def test_a_refused_record_is_a_fact_in_the_cycle_that_refused_it_and_a_corrected_record_resolves_as_a_later_row(tmp_path, capsys):
    root = _root(tmp_path)
    bad = _bad_model(50)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions() + [bad])
    rep = _ingest(root, synth.t(1000))
    assert rep.refused == 1 and rep.refusal_facts == 1
    sha = R.record_sha(bad)
    cur = _q(root, "SELECT * FROM record_refusal")
    assert len(cur) == 1 and cur[0]["record_sha256"] == sha and cur[0]["judged_with"] == H.validator_stamp() and cur[0]["refused_utc"] == synth.t(1000)
    facts = _facts(root)
    assert len(facts) == 1
    f = facts[0]
    assert (f["record_sha256"], f["judged_with"], f["record_type"], f["field"]) == (sha, H.validator_stamp(), "model", "checkpoint_kind")
    assert f["refused_utc"] == synth.t(1000) and f["cycle_utc"] == synth.t(1000) and f["record_uri"] == "records/dims/000001-dims.jsonl#13"
    assert json.loads(f["producer_json"]) == bad["producer"] and json.loads(f["extra_json"]) == {}
    files = _fact_files(root)
    assert len(files) == 1 and files[0].startswith("records/harvester/2026-01-01/record_refusal-")
    assert _q(root, "SELECT status FROM record_delivery WHERE file_uri = ?", files[0]) == [{"status": "ok"}]
    # the record id is minted over the payload without cycle_utc
    rec = next(iter(R.read_record_file(os.path.join(root, files[0]))))[1]
    payload = dict(rec["payload"])
    payload.pop("cycle_utc")
    from sisteer.registry.identity import adapter_record_id
    assert rec["record_id"] == adapter_record_id(payload, "record_refusal") and rec["record_type"] == "record_refusal"
    # a second cycle derives nothing and writes no second file
    rep2 = _ingest(root, synth.t(2000))
    assert rep2.refused == 0 and rep2.refusal_facts == 0 and len(_facts(root)) == 1 and _fact_files(root) == files
    # resolution before the correction: no row at the key; the current validator still refuses the bytes
    conn = DB.connect(DB.db_path(root), readonly=True)
    rep_e = RF.explain(root, conn, sha)
    conn.close()
    e = rep_e["records"][0]
    assert e["record"] == bad and e["current_verdict"]["verdict"] == "refused" and e["current_verdict"]["field"] == "checkpoint_kind"
    assert e["resolution"]["state"] == "no_row" and e["resolution"]["scale"] == "run" and e["resolution"]["natural_key"] == {"model_id": "m_bad"}
    assert e["current_judgment"]["judged_with"] == H.validator_stamp() and e["facts"][0]["still_refused"] == 1
    # the correction: a later record at the key folds -> later_row_folded, live row present
    _write(root, "dims", "000002-fix.jsonl", [_good_model(60)])
    assert _ingest(root, synth.t(3000)).refused == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    res = RF.explain(root, conn, sha)["records"][0]["resolution"]
    conn.close()
    assert res["state"] == "later_row_folded" and res["live_row"] is True and res["later_events"] == 1 and res["latest_event"]["record_type"] == "model"
    # the CLI: list / summary / explain by uri, read-only (no state/last_run stamp)
    assert cli.main(["--root", root, "refusals", "list"]) == 0
    out = capsys.readouterr().out
    assert "1 refusal fact(s)" in out and "checkpoint_kind" in out and "[still refused]" in out
    assert cli.main(["--root", root, "refusals", "summary", "--json"]) == 0
    summ = json.loads(capsys.readouterr().out)
    assert summ["facts"] == 1 and summ["records"] == 1 and summ["current_refusals"] == 1 and summ["current_without_fact"] == 0 and summ["backfilled"] == 0
    assert summ["by_judged_with"] == {H.validator_stamp(): 1} and summ["by_record_type"] == {"model": 1} and summ["by_field"] == {"checkpoint_kind": 1}
    assert cli.main(["--root", root, "refusals", "explain", "records/dims/000001-dims.jsonl#13"]) == 0
    out = capsys.readouterr().out
    assert "resolution: later_row_folded" in out and "current validator" in out
    assert not os.path.exists(os.path.join(root, "state", "last_run"))
    assert _rebuild_equal(root, tmp_path) == 1


def test_a_refused_item_then_a_corrected_item_resolves_at_item_scale_and_an_envelope_refusal_is_unknown(tmp_path):
    root = _root(tmp_path)
    ids = synth.write_corpus(root, n_runs=1, n_items=3, with_items=False)
    assert _ingest(root, synth.t(1000)).refused == 0
    rid, jid = ids["run_ids"][0], ids["judgment_ids"][0]
    good = synth.item_records(rid, jid, 300, n_items=3, seed=1)
    bad_item = json.loads(json.dumps(good[0]))
    bad_item["payload"]["value"] = "not a number"          # the column rule refuses a string value by name
    bad_item["written_utc"] = synth.t(310)
    bad_item["record_id"] = "11111111-2222-3333-4444-555555555555"
    unknown = {"record_type": "no_such_kind", "schema_version": 1, "record_id": "11111111-2222-3333-4444-666666666666",
               "written_utc": synth.t(311), "producer": synth.producer(), "payload": {"x": 1}}
    _write(root, "c", "000003-items.jsonl", [bad_item, unknown])
    rep = _ingest(root, synth.t(2000))
    assert rep.refused == 2 and rep.refusal_facts == 2
    facts = {f["record_sha256"]: f for f in _facts(root)}
    sha_item, sha_unknown = R.record_sha(bad_item), R.record_sha(unknown)
    assert facts[sha_item]["record_type"] == "item_result" and facts[sha_item]["field"] == "value"
    assert facts[sha_unknown]["record_type"] is None and facts[sha_unknown]["field"] == "record_type" and json.loads(facts[sha_unknown]["producer_json"]) == unknown["producer"]
    conn = DB.connect(DB.db_path(root), readonly=True)
    e_item = RF.explain(root, conn, sha_item)["records"][0]
    e_unknown = RF.explain(root, conn, sha_unknown)["records"][0]
    conn.close()
    assert e_item["resolution"] == {"state": "no_row", "scale": "item", "live_row": False, "table": "item_result",
                                    "natural_key": e_item["resolution"]["natural_key"], "refused_order_key": [synth.t(310), sha_item]}
    assert e_unknown["resolution"]["state"] == "unknown" and e_unknown["current_verdict"]["verdict"] == "refused"
    # the corrected item (the same key, a later order key) folds: later_row_folded at item scale
    fixed = json.loads(json.dumps(good[0]))
    fixed["written_utc"] = synth.t(320)
    _write(root, "c", "000004-items-fixed.jsonl", [fixed])
    assert _ingest(root, synth.t(3000)).refused == 0
    conn = DB.connect(DB.db_path(root), readonly=True)
    res = RF.explain(root, conn, sha_item)["records"][0]["resolution"]
    conn.close()
    assert res["state"] == "later_row_folded" and res["live_row"] is True and res["live_order_key"][0] == synth.t(320)
    assert _rebuild_equal(root, tmp_path) == 2


# --------------------------------------------------------------------------------------------
# 5-6: the kind is unledgered; the plan has no rebuild
# --------------------------------------------------------------------------------------------


def test_resolve_reads_a_run_key_in_the_folds_order_so_a_same_stamp_lateral_event_counts_as_later(tmp_path):
    """Under the record-log explain: the fold orders a run key's events by (written_utc, tie_rank, record_sha256) --
    at one stamp a run_state to a lateral / terminal state ranks AFTER everything else. `resolve` reads the key by the same order: a
    `superseded` run_state folded at the SAME stamp as a refused rank-0 record is LATER than it whatever the shas (under the earlier sha
    order the smaller sha read as earlier and the row as standing), a `judged` run_state at that stamp with the smaller sha is not, and the
    refused order key carries the rank; an item-scale key keeps its two-part key (pinned above)."""
    root = _root(tmp_path)
    assert cli.main(["--root", root, "init"]) == 0
    stamp = synth.t(500)
    conn = DB.connect(DB.db_path(root))
    try:
        def event(run_id, state, sha):
            conn.execute("INSERT INTO row_history(table_name, natural_key_json, written_utc, record_sha256, record_type, record_uri, payload_json, "
                         "folded_state_json) VALUES (?,?,?,?,?,?,?,?)",
                         ("run", R.canonical_json({"run_id": run_id}), stamp, sha, "run_state", "synthetic/u",
                          R.canonical_json({"_kind": "state", "state": state, "event_utc": stamp}), "{}"))
        event("r-lateral", "superseded", "a" * 64)
        event("r-progress", "judged", "a" * 64)
        conn.commit()
        refused = lambda rid: {"record_type": "run_patch", "payload": {"run_id": rid, "fields": {"notes": "refused"}}, "written_utc": stamp}
        res = RF.resolve(conn, refused("r-lateral"), "f" * 64)
        assert res["scale"] == "run" and res["refused_order_key"] == [stamp, 0, "f" * 64]
        assert res["later_events"] == 1 and res["state"] == "later_row_folded" and res["latest_event"]["record_type"] == "run_state", res
        res2 = RF.resolve(conn, refused("r-progress"), "f" * 64)
        assert res2["later_events"] == 0 and res2["state"] == "earlier_row_stands" and res2["events"] == 1, res2
    finally:
        conn.close()


def test_the_fact_kind_is_unledgered_and_the_two_ledgers_check_lists_are_byte_identical():
    assert "record_refusal" in R.RECORD_TYPES and R.RECORD_TYPE_TABLE["record_refusal"] == "record_refusal_fact"
    assert "record_refusal_fact" in S.LEDGER_TABLES and "record_refusal_fact" not in S.RUN_SCALE_TABLES
    # the analysis kinds joined the unledgered facts (migration 0029): the pin moved with them, the ledgers' CHECK lists did not
    assert V.UNLEDGERED_FACT_TYPES == ("record_refusal",) + V.ANALYSIS_RECORD_TYPES
    assert V.LEDGERED_RECORD_TYPES == tuple(t for t in V.RECORD_TYPES if t not in V.UNLEDGERED_FACT_TYPES)
    assert S.TABLE_BY_NAME["blocked_records"].column("record_type").vocab == "LEDGERED_RECORD_TYPES"
    assert S.TABLE_BY_NAME["row_history"].column("record_type").vocab == "LEDGERED_RECORD_TYPES"
    for t in ("blocked_records", "row_history"):
        assert "'record_refusal'" not in S.render_create_table(S.TABLE_BY_NAME[t]) and "'analysis" not in S.render_create_table(S.TABLE_BY_NAME[t])
    assert R.REQUIRED_FIELDS["record_refusal"] == ("record_sha256", "record_uri", "reason", "judged_with", "refused_utc", "cycle_utc")
    assert S.TABLE_BY_NAME["record_refusal"].column_names[-1] == "judged_with"


def test_migrate_plan_on_the_previous_shape_lists_one_create_the_add_columns_and_the_recreates_and_no_rebuild():
    # the shape one version below the record-refusal-fact migration, whatever number the file carries (by slug: a newer file, the
    # fold-observation migration, follows it on this tree -- its create is set aside here and pinned by its own test)
    conn = DB.connect_memory()
    rrf_version = next(v for v, slug, _ in M.migration_files() if slug == "record_refusal_fact")
    for version, slug, path in M.migration_files():
        if version < rrf_version:
            M._apply_one(conn, version, slug, open(path, encoding="utf-8").read())
    assert M.db_version(conn) == rrf_version - 1
    have, want = DB.ddl_master(conn), DB.code_ddl_master()
    for t in ("blocked_records", "row_history"):
        assert have[("table", t)] == want[("table", t)], t          # the CHECK lists are byte-identical: no ledger rebuild
    plan = M.shape_plan(conn)
    assert [r["table"] for r in plan["rebuild"] if r["table"] not in FU4_TABLES + NTPL_TABLES] == []        # 0027's run_contrast and 0031's two rebuilds aside
    assert [c for c in plan["create"] if c not in ("fold_observation",) + AF_TABLES] == ["record_refusal_fact"]   # 0029's five creates aside
    assert [(a["table"], a["columns"]) for a in plan["add_columns"]] == [("object", ["parent_object_id", "named_role"]),
                                                                         ("judgment", ["claimed_instrument_id", "claimed_required_judge_id"]),
                                                                         ("record_refusal", ["judged_with"])]
    # migration 0026 (the twin fold) recreates v_multi_source_check and v_run_current on this shape too, and v_multi_source_rows -- this file's own --
    # once more; its two are set aside here and pinned by its own test (tests/test_registry_twin_fold.py)
    assert [v for v in plan["objects"]["view"]["recreate"] if v not in ("v_multi_source_check", "v_run_current")] == [
        "v_item_result_current", "v_item_result_provenances", "v_multi_source_rows", "v_paper_current", "v_paper_long", "v_pending_judgment", "v_roster_match"]
    assert plan["objects"]["view"]["drop"] == []
    assert plan["objects"]["index"] == {"recreate": [], "drop": []} and plan["objects"]["trigger"] == {"recreate": [], "drop": []}   # 0030's two fence triggers ride 0031's run rebuild
    body = _sans_fu4_rebuild(M.render_shape_migration(conn, plan))                                  # 0027's run_contrast rebuild block (and 0031's two) aside
    assert body.count("CREATE INDEX idx_record_refusal_fact_") == 2 and "__new" not in body and "INSERT INTO" not in body
    import re
    assert len(re.findall(r"(?m)^ALTER TABLE \w+ ADD COLUMN ", body)) == 5                   # the statements (the planner's comments name ALTER too)
    conn.close()


# --------------------------------------------------------------------------------------------
# 7-10: the START pass as the backstop; the backfill; retry beside backfill; step 0
# --------------------------------------------------------------------------------------------


def test_a_restart_between_the_refusal_and_the_fact_pass_writes_exactly_one_fact_on_the_next_start(tmp_path, monkeypatch):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions() + [_bad_model(50)])
    real = H.derive_refusal_fact_records
    _no_facts(monkeypatch)
    rep = _ingest(root, synth.t(1000))
    assert rep.refused == 1 and rep.refusal_facts == 0 and _facts(root) == [] and _fact_files(root) == []
    monkeypatch.setattr(H, "derive_refusal_fact_records", real)
    rep2 = _ingest(root, synth.t(2000))
    assert rep2.refused == 0 and rep2.refusal_facts == 1
    facts = _facts(root)
    assert len(facts) == 1 and facts[0]["judged_with"] == H.validator_stamp() and facts[0]["refused_utc"] == synth.t(1000) and facts[0]["cycle_utc"] == synth.t(2000)
    assert json.loads(facts[0]["extra_json"]) == {} and len(_fact_files(root)) == 1
    rep3 = _ingest(root, synth.t(3000))
    assert rep3.refusal_facts == 0 and len(_facts(root)) == 1 and len(_fact_files(root)) == 1
    _rebuild_equal(root, tmp_path)


def test_a_row_refused_before_the_stamp_was_kept_is_backfilled_as_an_unverified_fact(tmp_path, monkeypatch):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions() + [_bad_model(50), _bad_model(51, "m_bad2")])
    real = H.derive_refusal_fact_records
    _no_facts(monkeypatch)
    assert _ingest(root, synth.t(1000)).refused == 2
    conn = DB.connect(DB.db_path(root))
    conn.execute("UPDATE record_refusal SET judged_with = NULL")          # the rows as the earlier code left them (126 live)
    conn.commit()
    conn.close()
    monkeypatch.setattr(H, "derive_refusal_fact_records", real)
    rep = _ingest(root, synth.t(2000))
    assert rep.refusal_facts == 2
    facts = _facts(root)
    assert [f["judged_with"] for f in facts] == ["unverified", "unverified"]
    assert all(json.loads(f["extra_json"]) == {"backfilled": True} for f in facts) and all(f["cycle_utc"] == synth.t(2000) for f in facts)
    conn = DB.connect(DB.db_path(root), readonly=True)
    summ = RF.summary(conn)
    rows = RF.list_facts(conn, still_refused=True)
    conn.close()
    assert summ["backfilled"] == 2 and summ["by_judged_with"] == {"unverified": 2} and summ["current_without_fact"] == 0
    assert len(rows) == 2 and all(r["still_refused"] == 1 for r in rows)   # the current judgment (NULL) IS the unverified fact
    assert _ingest(root, synth.t(3000)).refusal_facts == 0


def test_retry_under_a_moved_stamp_and_backfill_coexist_in_one_cycle_and_the_start_pass_precedes_the_retry(tmp_path, monkeypatch):
    """A held constituent (a run_metric under a banned spelling, the refused-hold fixture) re-attempted under a MOVED stamp beside three
    rows refused before the stamp was kept. Variant A (the ordinary one): the first judgment's fact exists; the next cycle writes the three
    `unverified` facts (START) and the re-judgment's fact (END): four, five rows in all. Variant B (crash, then code change): the cycle that
    made the first judgment died before its END pass, so the START pass of the next cycle must make that judgment a fact BEFORE the retry
    replaces the row -- both stamps are facts afterwards, nothing is lost."""
    from test_registry_refused_hold_exact import _held
    for variant in ("ordinary", "crash_then_code_change"):
        tp = tmp_path / variant
        tp.mkdir()
        real = H.derive_refusal_fact_records
        if variant == "crash_then_code_change":
            _no_facts(monkeypatch)
        root, _, _, rev = _held(tp, now=synth.t(2500))
        monkeypatch.setattr(H, "derive_refusal_fact_records", real)
        first_stamp = H.validator_stamp()
        (sha_held,) = [r["record_sha256"] for r in _q(root, "SELECT record_sha256 FROM record_refusal")]
        assert len(_facts(root)) == (1 if variant == "ordinary" else 0)
        conn = DB.connect(DB.db_path(root))
        for k in range(3):   # three rows the earlier code refused: no stamp, no fact
            conn.execute("INSERT INTO record_refusal(record_sha256, record_uri, record_type, field, reason, refused_utc, judged_with) VALUES (?,?,?,?,?,?,NULL)",
                         (f"{k:064x}", f"records/legacy/{k}.jsonl#1", "model", "checkpoint_kind", "legacy refusal", synth.t(100 + k)))
        conn.commit()
        conn.close()
        monkeypatch.setattr(H, "validator_stamp", lambda: "f" * 16)         # a landed change to the judging code: the hold re-attempts once
        now = synth.t(3500)
        rep = _ingest(root, now)
        assert rep.refused == 1 and rep.revisions_held == 1, (variant, rep.refused, rep.revisions_held)
        facts = _facts(root)
        by = {}
        for f in facts:
            by.setdefault(f["record_sha256"], []).append(f)
        assert sorted(f["judged_with"] for f in by.pop(sha_held)) == sorted([first_stamp, "f" * 16]), variant   # both judgments are facts
        assert len(by) == 3 and all(len(v) == 1 and v[0]["judged_with"] == "unverified" and json.loads(v[0]["extra_json"]) == {"backfilled": True} for v in by.values())
        assert rep.refusal_facts == (4 if variant == "ordinary" else 5), (variant, rep.refusal_facts)
        assert _q(root, "SELECT judged_with FROM record_refusal WHERE record_sha256 = ?", sha_held) == [{"judged_with": "f" * 16}]
        # the previous judgment was a fact before the retry replaced the row: its refused_utc is the FIRST judgment's time
        prev = [f for f in facts if f["record_sha256"] == sha_held and f["judged_with"] == first_stamp][0]
        assert prev["refused_utc"] == synth.t(2500) and prev["cycle_utc"] == (synth.t(2500) if variant == "ordinary" else now)
        monkeypatch.undo()
        H.validator_stamp.cache_clear()


def test_a_fact_file_written_but_not_folded_is_folded_first_by_the_next_cycle_and_never_duplicated(tmp_path, monkeypatch):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions() + [_bad_model(50)])
    real = H.derive_refusal_fact_records
    _no_facts(monkeypatch)
    assert _ingest(root, synth.t(1000)).refused == 1
    monkeypatch.setattr(H, "derive_refusal_fact_records", real)
    # the cycle writes its fact file and dies before folding it
    conn = DB.connect(DB.db_path(root), readonly=True)
    paths, n = H.derive_refusal_fact_records(root, conn, synth.t(1500), synth.t(1500))
    conn.close()
    assert n == 1 and len(paths) == 1 and _facts(root) == []
    first = os.path.relpath(paths[0], root)
    assert _q(root, "SELECT 1 AS one FROM record_delivery WHERE file_uri = ?", first) == []
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert [H.logical_uri(root, p) for p, _ in H.pending_refusal_fact_files(root, conn)] == [first]
    conn.close()
    rep = _ingest(root, synth.t(2000))
    assert rep.refusal_facts == 0, "step 0 folded the pending file; the START pass found the fact and wrote nothing"
    facts = _facts(root)
    assert len(facts) == 1 and facts[0]["cycle_utc"] == synth.t(1500)
    assert _fact_files(root) == [first]
    assert _q(root, "SELECT status FROM record_delivery WHERE file_uri = ?", first) == [{"status": "ok"}]
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert H.pending_refusal_fact_files(root, conn) == []
    conn.close()
    _rebuild_equal(root, tmp_path)


# --------------------------------------------------------------------------------------------
# the status page and the vocabulary
# --------------------------------------------------------------------------------------------


def test_the_status_page_carries_the_recent_facts_and_the_harvester_row_counts_them(tmp_path):
    root = _root(tmp_path)
    _write(root, "dims", "000001-dims.jsonl", synth.seed_dimensions() + [_bad_model(50)])
    rep = _ingest(root, R.utc_now())            # a real clock: the facts are within the harvester row's window
    assert rep.refused == 1 and rep.refusal_facts == 1
    ctx = status.status_json(root)
    assert "refusal_facts" in status.QUERIES and len(ctx["refusal_facts"]) == 1 and ctx["refusal_facts"][0]["still_refused"] == 1
    assert ctx["refusal_facts_window_s"] == ctx["harvest_cycle"]["late_after_s"] and ctx["refusal_facts_recent"] == ctx["refusal_facts"]
    assert ctx["harvest_cycle"]["refused"] == 1 and ctx["harvest_cycle"]["refusal_facts"] == 1
    assert ctx["refused"][0]["judged_with"] == H.validator_stamp()
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    html = open(out, encoding="utf-8").read()
    assert "Refusals (facts)" in html and "refused 1 this cycle and wrote 1 refusal fact(s)" in html and "registry refusals" in html


def test_a_record_refusal_record_validates_against_the_fact_table_and_a_malformed_one_is_refused_by_name():
    payload = {"record_sha256": "a" * 64, "record_uri": "records/x/1.jsonl#1", "record_type": "model", "field": "checkpoint_kind", "reason": "r",
               "judged_with": "f" * 16, "refused_utc": synth.t(1), "cycle_utc": synth.t(2), "producer_json": synth.producer(), "extra": {"backfilled": True}}
    rec = synth.rec("record_refusal", payload, synth.t(2))
    v = VAL.validate_record(rec, "<t>", R.record_sha(rec))
    assert isinstance(v, VAL.Validated) and v.table == "record_refusal_fact" and v.extra == {"backfilled": True}
    assert VAL.parent_references(v) == []                                   # a fact has no parents: never blocked
    bad = synth.rec("record_refusal", dict(payload, judged_with="not-a-stamp"), synth.t(2))
    out = VAL.validate_record(bad, "<t>", R.record_sha(bad))
    assert isinstance(out, VAL.Malformed) and out.field == "judged_with"
    missing = synth.rec("record_refusal", {k: v for k, v in payload.items() if k != "cycle_utc"}, synth.t(2))
    out = VAL.validate_record(missing, "<t>", R.record_sha(missing))
    assert isinstance(out, VAL.Malformed) and out.field == "cycle_utc"
