"""The derived decode-policy match (`v_run_decode_policy_match` + `decode_match.stamp_run_rows`): if an exact match
between the decode parameters a run recorded and a policy row can be verified, the policy goes into the row.

Synthetic corpus only. One policy table, one run per case; every verdict is read from the run ROW (the stamped columns) and
from the SQL object it was derived from, and a rebuild of the same log reproduces the stamps byte for byte.
"""
from __future__ import annotations

import json
import os
import sqlite3

import test_registry_decode_llama_thinking_na as LT
from test_registry_decode_llama_thinking_na import idx, seeded_db, seeds  # noqa: F401  (fixtures: the real seeds, ingested)

from sisteer.registry import db as DB
from sisteer.registry import decode_match as DM
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import records as R
from sisteer.registry import status
from sisteer.registry import synth

#: Qwen-style vendor row: six parameters pinned, all sent per request
VENDOR = dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, repetition_penalty=1.0, presence_penalty=1.5)
VENDOR_SENT = dict(VENDOR)


def _policy(policy_id, ref, version=1, eval_id=None, eval_category="static_attack", thinking="off", seed_rule="per_item",
            effective=synth.t(1), superseded_by=None, not_sent=None, **params):
    p = {"policy_id": policy_id, "policy_ref": ref, "policy_version": version, "model_id": synth.MODEL_ID, "thinking_mode": thinking,
         "decode_source": "generation_config", "max_new_tokens_rule": "cap 512", "seed_rule": seed_rule,
         "params_not_sent_json": not_sent or {}, "source_citation": "synthetic", "effective_from_utc": effective}
    if eval_id:
        p["eval_id"] = eval_id
    else:
        p["eval_category"] = eval_category
    if superseded_by:
        p["superseded_by_policy_id"] = superseded_by
    p.update(params)
    return p


def _run(key, sent, seed_rule="per_item", i0=100, **over):
    """A run through `generated` (so generated_utc = t(i0 + 3)) whose SENT decode is `sent` (dict -> decode_sent_json + typed columns)."""
    rid, recs = synth.run_records(key, i0, through="generated")
    p = recs[0]["payload"]
    p["decode_sent_json"] = dict(sent)
    p["decode_temperature_sent"] = sent.get("temperature")
    p["decode_top_p_sent"] = sent.get("top_p")
    p["decode_top_k_sent"] = sent.get("top_k")
    p["decode_seed_rule"] = seed_rule
    p["decode_source"] = "generation_config"
    p.update(over)
    return rid, recs


def _ingest(root, policies, runs, name="dm"):
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    R.write_records(os.path.join(root, "records", name, "000001-policies.jsonl"), [synth.rec("decode_policy", p, synth.t(2)) for p in policies])
    recs = []
    for _, rr in runs:
        recs += rr
    R.write_records(os.path.join(root, "records", name, "000002-runs.jsonl"), recs)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    return rep


def _row(conn, rid):
    r = conn.execute("SELECT decode_policy_id, decode_policy_match, field_versions_json, extra_json FROM run WHERE run_id = ?", (rid,)).fetchone()
    v = conn.execute("SELECT policy_id, match, candidates_json, in_scope_json FROM v_run_decode_policy_match WHERE run_id = ?", (rid,)).fetchone()
    return dict(r), dict(v)


def _stamped(row):
    fv = json.loads(row["field_versions_json"])
    return fv.get("decode_policy_id"), fv.get("decode_policy_match")


def _tiebreak(row):
    """The audit a resolved tie leaves on the row (`extra_json.decode_policy_tiebreak`), None when the pass decided nothing."""
    return json.loads(row["extra_json"]).get(DM.TIEBREAK_KEY)


def test_exact_match_stamps_the_row_and_one_parameter_off_by_any_amount_does_not(tmp_path):
    root = str(tmp_path / "reg")
    pol = _policy("p_vendor", "synthmodel/static/vendor", **VENDOR)
    exact = _run("exact", VENDOR_SENT, i0=100)
    off = _run("off", dict(VENDOR_SENT, top_p=0.8000001), i0=200)          # off by 1e-7: not exact
    as_int = _run("as_int", dict(VENDOR_SENT, repetition_penalty=1, min_p=0), i0=300)  # 1 == 1.0, 0 == 0.0
    _ingest(root, [pol], [exact, off, as_int])
    conn = DB.connect(DB.db_path(root), readonly=True)
    row, view = _row(conn, exact[0])
    assert (row["decode_policy_id"], row["decode_policy_match"]) == ("p_vendor", "exact")
    assert (view["policy_id"], view["match"], json.loads(view["candidates_json"])) == ("p_vendor", "exact", ["p_vendor"])
    assert _stamped(row) == (DM.STAMP, DM.STAMP)                         # derived, named by the derivation, no record behind it
    row, view = _row(conn, off[0])
    assert (row["decode_policy_id"], row["decode_policy_match"]) == (None, "no_match")
    assert json.loads(view["in_scope_json"]) == ["p_vendor"]             # it WAS compared with the vendor row
    row, _ = _row(conn, as_int[0])
    assert (row["decode_policy_id"], row["decode_policy_match"]) == ("p_vendor", "exact")
    # the placeholder survives nowhere the pass ran
    assert conn.execute("SELECT COUNT(*) FROM run WHERE decode_policy_match = 'no_policy'").fetchone()[0] == 0
    conn.close()


def test_policy_pins_a_parameter_the_run_did_not_send_unless_params_not_sent_covers_it(tmp_path):
    root = str(tmp_path / "reg")
    # the server-default shape: temperature / top_p / top_k pinned AND listed as not sent (resolved server-side); presence_penalty pinned, sent
    strict = _policy("p_strict", "synthmodel/static/strict", temperature=0.7, top_p=0.8, top_k=20)
    lenient = _policy("p_lenient", "synthmodel/static/lenient", eval_id=synth.EVAL_ID, temperature=0.7, top_p=0.8, top_k=20,
                      not_sent={"top_k": "20 (override)"})
    # sends temperature + top_p only: strict pins top_k unsent -> no; lenient lists top_k as not sent -> yes
    two = _run("two", {"temperature": 0.7, "top_p": 0.8}, i0=100)
    # sends all three: both match by values -> both are candidates and both in force -> the SQL object reads a tie; the stamp pass
    # breaks it for the eval-scoped row (p_lenient) and writes the loser beside the stamp
    three = _run("three", {"temperature": 0.7, "top_p": 0.8, "top_k": 20}, i0=200)
    # sends a parameter neither pins (min_p): no policy leaves a sent key unpinned and still matches
    extra = _run("extra", {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0}, i0=300)
    _ingest(root, [strict, lenient], [two, three, extra])
    conn = DB.connect(DB.db_path(root), readonly=True)
    row, view = _row(conn, two[0])
    assert (row["decode_policy_id"], row["decode_policy_match"]) == ("p_lenient", "exact")
    assert json.loads(view["in_scope_json"]) == ["p_lenient", "p_strict"]
    row, view = _row(conn, three[0])
    assert (row["decode_policy_id"], row["decode_policy_match"]) == ("p_lenient", "exact")
    assert (view["match"], json.loads(view["candidates_json"])) == ("ambiguous", ["p_lenient", "p_strict"])
    assert _tiebreak(row) == {"over": ["p_strict"], "rule": DM.TIEBREAK_RULE}
    row, _ = _row(conn, extra[0])
    assert (row["decode_policy_id"], row["decode_policy_match"]) == (None, "no_match")
    conn.close()


def test_nothing_recorded_is_unrecorded_and_a_sent_key_outside_the_policy_model_matches_nothing(tmp_path):
    root = str(tmp_path / "reg")
    pol = _policy("p_vendor", "synthmodel/static/vendor", **VENDOR)
    # decode_sent_json = {} and no typed column: the producer recorded nothing (the greedy / multi-turn "sends nothing" stores)
    nothing = _run("nothing", {}, i0=100)
    # a run whose seed rule is recorded but nothing else: still unrecorded (a seed rule alone verifies no sampler)
    seed_only = _run("seed_only", {}, seed_rule="per_item", i0=200)
    # a key the policy table has no column for
    foreign = _run("foreign", dict(VENDOR_SENT, frequency_penalty=0.0), i0=300)
    _ingest(root, [pol], [nothing, seed_only, foreign])
    conn = DB.connect(DB.db_path(root), readonly=True)
    for rid in (nothing[0], seed_only[0]):
        row, view = _row(conn, rid)
        assert (row["decode_policy_id"], row["decode_policy_match"]) == (None, "unrecorded"), rid
        assert json.loads(view["candidates_json"]) == []
    row, _ = _row(conn, foreign[0])
    assert (row["decode_policy_id"], row["decode_policy_match"]) == (None, "no_match")
    conn.close()


def test_scope_requires_the_run_recorded_thinking_state_and_the_seed_rule(tmp_path):
    root = str(tmp_path / "reg")
    pol = _policy("p_vendor", "synthmodel/static/vendor", **VENDOR)
    # the run files `not_available` (no thinking switch); the row pins `off`: not the run's recorded state -> not in scope
    na = _run("na", VENDOR_SENT, i0=100, thinking_mode="not_available")
    # every parameter equal, seed rule differs (`none` vs the row's `per_item`)
    seed = _run("seed", VENDOR_SENT, seed_rule="none", i0=200)
    _ingest(root, [pol], [na, seed])
    conn = DB.connect(DB.db_path(root), readonly=True)
    row, view = _row(conn, na[0])
    assert row["decode_policy_match"] == "no_match" and json.loads(view["in_scope_json"]) == []
    row, view = _row(conn, seed[0])
    assert row["decode_policy_match"] == "no_match" and json.loads(view["in_scope_json"]) == ["p_vendor"]
    conn.close()


def test_a_run_generated_before_the_successor_took_effect_matches_the_predecessor(tmp_path):
    root = str(tmp_path / "reg")
    # v1 (top_p 1.0) superseded by v2 (top_p 0.95) effective at t(500); v1 stays readable by name
    v2 = _policy("p_v2", "synthmodel/static/vendor", version=2, effective=synth.t(500), temperature=1.0, top_p=0.95)
    v1 = _policy("p_v1", "synthmodel/static/vendor", version=1, effective=synth.t(1), superseded_by="p_v2", temperature=1.0, top_p=1.0)
    # the same values under two rows, one superseding the other: which is in force at generated_utc decides
    twin_new = _policy("p_twin_new", "synthmodel/static/twin", version=2, eval_id=synth.EVAL_ID, effective=synth.t(500),
                       temperature=0.3, top_p=0.5)
    twin_old = _policy("p_twin_old", "synthmodel/static/twin", version=1, eval_id=synth.EVAL_ID, effective=synth.t(1),
                       superseded_by="p_twin_new", temperature=0.3, top_p=0.5)
    before = _run("before", {"temperature": 1.0, "top_p": 1.0}, i0=100)     # generated t(103) < t(500): v1 admits it
    after = _run("after", {"temperature": 1.0, "top_p": 1.0}, i0=600)       # generated t(603) >= t(500): v1 is no candidate, v2 differs
    after_v2 = _run("after_v2", {"temperature": 1.0, "top_p": 0.95}, i0=700)
    twin_before = _run("twin_before", {"temperature": 0.3, "top_p": 0.5}, i0=200)   # both match by value; old in force
    twin_after = _run("twin_after", {"temperature": 0.3, "top_p": 0.5}, i0=800)     # old excluded; new alone
    _ingest(root, [v2, v1, twin_new, twin_old], [before, after, after_v2, twin_before, twin_after])
    conn = DB.connect(DB.db_path(root), readonly=True)
    got = {k: _row(conn, rid[0])[0] for k, rid in (("before", before), ("after", after), ("after_v2", after_v2),
                                                   ("twin_before", twin_before), ("twin_after", twin_after))}
    assert (got["before"]["decode_policy_id"], got["before"]["decode_policy_match"]) == ("p_v1", "exact")
    assert (got["after"]["decode_policy_id"], got["after"]["decode_policy_match"]) == (None, "no_match")
    assert (got["after_v2"]["decode_policy_id"], got["after_v2"]["decode_policy_match"]) == ("p_v2", "exact")
    assert (got["twin_before"]["decode_policy_id"], got["twin_before"]["decode_policy_match"]) == ("p_twin_old", "exact")
    assert (got["twin_after"]["decode_policy_id"], got["twin_after"]["decode_policy_match"]) == ("p_twin_new", "exact")
    conn.close()


def test_rebuild_reproduces_the_live_stamps_and_a_refold_is_restamped_next_cycle(tmp_path):
    root = str(tmp_path / "reg")
    pol = _policy("p_vendor", "synthmodel/static/vendor", **VENDOR)
    exact = _run("exact", VENDOR_SENT, i0=100)
    off = _run("off", dict(VENDOR_SENT, top_k=40), i0=200)
    nothing = _run("nothing", {}, i0=300)
    _ingest(root, [pol], [exact, off, nothing])
    # a later patch re-folds the exact run's row (the fold writes the producer's `no_policy` placeholder back); the cycle re-stamps it
    late = synth.rec("run_patch", {"run_id": exact[0], "fields": {"job_name": "late"}}, synth.t(900))
    R.write_records(os.path.join(root, "records", "dm", "000003-late.jsonl"), [late])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0
    live = DB.connect(DB.db_path(root), readonly=True)
    r = live.execute("SELECT job_name, decode_policy_id, decode_policy_match FROM run WHERE run_id = ?", (exact[0],)).fetchone()
    assert tuple(r) == ("late", "p_vendor", "exact")
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    rebuilt = DB.connect(out, readonly=True)
    diff = H.verify_equal(live, rebuilt)
    assert not {k: v for k, v in diff.items() if v}, {k: v for k, v in diff.items() if v}
    sql = "SELECT run_id, decode_policy_id, decode_policy_match, field_versions_json FROM run ORDER BY run_id"
    assert live.execute(sql).fetchall() == rebuilt.execute(sql).fetchall()
    assert {r[2] for r in live.execute(sql).fetchall()} == {"exact", "no_match", "unrecorded"}
    # idempotent: a second pass over the same state writes nothing
    w = DB.connect(DB.db_path(root))
    with DB.transaction(w):
        assert DM.stamp_run_rows(w) == 0
    w.close()
    live.close()
    rebuilt.close()


def test_a_decode_patch_folds_with_its_provenance_and_the_same_cycle_restamps_the_match(tmp_path):
    """A run registered with a wrong sent sampler (top_k 40 against the vendor row's 20) and a `model_card` source
    reads `no_match`; one `run_patch` carrying the corrected sent / resolved decode fields, `decode_source`, and its provenance under
    `fields.extra_json` folds per column by order key (the patch's stamp on every corrected column, `extra_json.decision` merged in), and
    the SAME ingest cycle's `stamp_run_rows` re-derives the pair from the corrected fields: the row reads `exact` / the vendor row, stamped
    by the derivation, and the SQL object agrees. A rebuild reproduces the row, provenance included. A patch naming the derived pair is
    refused by name and leaves the derivation's stamp in place."""
    root = str(tmp_path / "reg")
    pol = _policy("p_vendor", "synthmodel/static/vendor", **VENDOR)
    wrong = _run("wrong", dict(VENDOR_SENT, top_k=40), i0=100, decode_source="model_card")
    _ingest(root, [pol], [wrong])
    conn = DB.connect(DB.db_path(root), readonly=True)
    row, view_row = _row(conn, wrong[0])
    assert (row["decode_policy_id"], row["decode_policy_match"], view_row["match"]) == (None, "no_match", "no_match")
    assert tuple(conn.execute("SELECT decode_source, decode_top_k_sent FROM run WHERE run_id = ?", (wrong[0],)).fetchone()) == ("model_card", 40)
    conn.close()
    resolved = dict(VENDOR_SENT, min_p=0.0)
    provenance = {"decision": "D635", "reason": "resolved sampler re-read: the client sent the vendor row", "source": "synthetic://standup/wrong.log"}
    patch = synth.rec("run_patch", {"run_id": wrong[0], "fields": {
        "decode_sent_json": dict(VENDOR_SENT), "decode_temperature_sent": VENDOR_SENT["temperature"], "decode_top_p_sent": VENDOR_SENT["top_p"],
        "decode_top_k_sent": VENDOR_SENT["top_k"], "decode_resolved_json": resolved, "decode_source": "generation_config",
        "extra_json": provenance}}, synth.t(900))
    R.write_records(os.path.join(root, "records", "dm", "000003-d635.jsonl"), [patch])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0 and rep.blocked == 0, rep.refusals
    live = DB.connect(DB.db_path(root), readonly=True)
    row, view_row = _row(live, wrong[0])
    assert (row["decode_policy_id"], row["decode_policy_match"]) == ("p_vendor", "exact")
    assert (view_row["policy_id"], view_row["match"]) == ("p_vendor", "exact")
    assert _stamped(row) == (DM.STAMP, DM.STAMP)                           # the derivation's stamp, not the patch's
    fv = json.loads(row["field_versions_json"])
    key = [synth.t(900), R.record_sha(patch)]
    for col in ("decode_sent_json", "decode_resolved_json", "decode_temperature_sent", "decode_top_p_sent", "decode_top_k_sent", "decode_source"):
        assert fv[col] == key, (col, fv[col])
    r = live.execute("SELECT decode_source, decode_top_k_sent, decode_sent_json, decode_resolved_json FROM run WHERE run_id = ?", (wrong[0],)).fetchone()
    assert (r[0], r[1], json.loads(r[2]), json.loads(r[3])) == ("generation_config", 20, VENDOR_SENT, resolved)
    extra = json.loads(row["extra_json"])
    assert {k: extra[k] for k in provenance} == provenance
    # rebuild: the same row, provenance included
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    rebuilt = DB.connect(out, readonly=True)
    diff = H.verify_equal(live, rebuilt)
    assert not {k: v for k, v in diff.items() if v}, {k: v for k, v in diff.items() if v}
    sql = "SELECT decode_policy_id, decode_policy_match, decode_source, field_versions_json, json_extract(extra_json, '$.decision') FROM run WHERE run_id = ?"
    assert tuple(live.execute(sql, (wrong[0],)).fetchone()) == tuple(rebuilt.execute(sql, (wrong[0],)).fetchone())
    assert rebuilt.execute(sql, (wrong[0],)).fetchone()[4] == "D635"
    rebuilt.close()
    live.close()
    # the derived pair may not be patched: refused by name, the row untouched
    bad = synth.rec("run_patch", {"run_id": wrong[0], "fields": {"decode_policy_match": "no_match", "job_name": "x"}}, synth.t(910))
    R.write_records(os.path.join(root, "records", "dm", "000004-derived.jsonl"), [bad])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 1, rep.refusals
    live = DB.connect(DB.db_path(root), readonly=True)
    ref = live.execute("SELECT field, reason FROM record_refusal WHERE record_sha256 = ?", (R.record_sha(bad),)).fetchone()
    assert ref[0] == "decode_policy_match" and "derived" in ref[1]
    row, _ = _row(live, wrong[0])
    assert (row["decode_policy_id"], row["decode_policy_match"]) == ("p_vendor", "exact") and _stamped(row) == (DM.STAMP, DM.STAMP)
    live.close()


def test_status_page_reads_the_derived_match_and_the_dry_run_reports_the_same_distribution(tmp_path):
    root = str(tmp_path / "reg")
    pol = _policy("p_vendor", "synthmodel/static/vendor", **VENDOR)
    exact = _run("exact", VENDOR_SENT, i0=100)
    off = _run("off", dict(VENDOR_SENT, temperature=1.0), i0=200)
    _ingest(root, [pol], [exact, off])
    conn = DB.connect(DB.db_path(root), readonly=True)
    ctx, _ = status.gather(conn, root)
    counts = {r["match"]: r["n_runs"] for r in ctx["decode_policy_match_counts"]}
    assert counts == {"exact": 1, "no_match": 1}                         # no `no_policy` left: the pass ran
    assert [r["run_id"] for r in ctx["decode_policy_unmatched"]] == [off[0]]
    assert ctx["decode_policy_unmatched"][0]["in_scope_json"] == '["p_vendor"]'
    assert ctx["decode_unmatched_total"] == 1
    html = status.render_html(ctx)
    assert "Decode-policy mismatches" in html and off[0] in html
    rep = DM.distribution(conn)
    assert rep["counts"] == {"exact": 1, "no_match": 1} and rep["exact_by_policy"] == [("p_vendor", 1)]
    assert rep["no_match_top"] == [((synth.CAMPAIGN, "temperature sent 1 != policy 0.7"), 1)]
    conn.close()


# --------------------------------------------------------------------------------------------
# the Python tie-break: an eval-scoped row beats a category-scoped row; a true duplicate stays ambiguous
# --------------------------------------------------------------------------------------------


def test_an_eval_row_beats_a_category_row_with_the_same_numbers_and_a_true_duplicate_stays_ambiguous(tmp_path):
    root = str(tmp_path / "reg")
    # the live shape (2026-09-19, 374 runs): `<model>|<eval>|on|thinking_card` beside `<model>|capability|on|thinking_card`, same numbers
    cat = _policy("p_cat", "synthmodel/static/card", **VENDOR)
    ev = _policy("p_eval", "synthmodel/syntheval/card", eval_id=synth.EVAL_ID, **VENDOR)
    # two EVAL rows, both in force, both admitting the same run (they differ only in what the matcher does not compare)
    dup_a = _policy("p_dup_a", "synthmodel/syntheval/dup_a", eval_id=synth.EVAL_ID, temperature=0.3, top_p=0.5, max_new_tokens_rule="cap 1024")
    dup_b = _policy("p_dup_b", "synthmodel/syntheval/dup_b", eval_id=synth.EVAL_ID, temperature=0.3, top_p=0.5, max_new_tokens_rule="cap 2048")
    # two CATEGORY rows, both in force, both admitting the same run: nothing eval-scoped to prefer
    cat_a = _policy("p_cat_a", "synthmodel/static/cat_a", temperature=0.9, top_p=0.1)
    cat_b = _policy("p_cat_b", "synthmodel/static/cat_b", temperature=0.9, top_p=0.1)
    tie = _run("tie", VENDOR_SENT, i0=100)
    dup = _run("dup", {"temperature": 0.3, "top_p": 0.5}, i0=200)
    cats = _run("cats", {"temperature": 0.9, "top_p": 0.1}, i0=300)
    _ingest(root, [cat, ev, dup_a, dup_b, cat_a, cat_b], [tie, dup, cats])
    conn = DB.connect(DB.db_path(root), readonly=True)
    row, view = _row(conn, tie[0])
    # the SQL object is unchanged (it still reads the tie); the ROW carries the broken tie, stamped as derived, with the loser named
    assert (view["match"], view["policy_id"], json.loads(view["candidates_json"])) == ("ambiguous", None, ["p_cat", "p_eval"])
    assert (row["decode_policy_id"], row["decode_policy_match"]) == ("p_eval", "exact")
    assert _stamped(row) == (DM.STAMP, DM.STAMP)
    assert _tiebreak(row) == {"over": ["p_cat"], "rule": "eval_over_category"}
    for rid, cands in ((dup[0], ["p_dup_a", "p_dup_b"]), (cats[0], ["p_cat_a", "p_cat_b"])):
        row, view = _row(conn, rid)
        assert (row["decode_policy_id"], row["decode_policy_match"]) == (None, "ambiguous"), rid
        assert json.loads(view["candidates_json"]) == cands
        assert _tiebreak(row) is None, rid                                   # nothing decided, nothing written
    # the read-only dry run reports the same: the stamped verdict beside the SQL object's own
    rep = DM.distribution(conn)
    assert rep["counts"] == {"exact": 1, "ambiguous": 2} and rep["sql_counts"] == {"ambiguous": 3}
    assert rep["tiebreak_resolved"] == [((synth.CAMPAIGN, "p_eval", '["p_cat"]'), 1)]
    conn.close()
    # idempotent: a second pass over the same state writes nothing (the audit text compares equal to what the row carries)
    w = DB.connect(DB.db_path(root))
    with DB.transaction(w):
        assert DM.stamp_run_rows(w) == 0
    w.close()
    # a later patch re-folds the tied run's row; the cycle re-stamps it, tie-break and audit included
    late = synth.rec("run_patch", {"run_id": tie[0], "fields": {"job_name": "late"}}, synth.t(900))
    R.write_records(os.path.join(root, "records", "dm", "000003-late.jsonl"), [late])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0
    live = DB.connect(DB.db_path(root), readonly=True)
    row, _ = _row(live, tie[0])
    assert (row["decode_policy_id"], row["decode_policy_match"]) == ("p_eval", "exact") and _tiebreak(row) == {"over": ["p_cat"], "rule": "eval_over_category"}
    # a rebuild of the same log reproduces the stamps AND the audit byte for byte
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    rebuilt = DB.connect(out, readonly=True)
    diff = H.verify_equal(live, rebuilt)
    assert not {k: v for k, v in diff.items() if v}, {k: v for k, v in diff.items() if v}
    sql = "SELECT run_id, decode_policy_id, decode_policy_match, field_versions_json, extra_json FROM run ORDER BY run_id"
    assert live.execute(sql).fetchall() == rebuilt.execute(sql).fetchall()
    live.close()
    rebuilt.close()


def test_the_in_force_filter_decides_before_the_scope_preference(tmp_path):
    """A row that took effect AFTER the run never beats the row that governed it, whatever its scope: the SQL object's temporal
    filter runs first and its `exact` verdicts stand; the eval-over-category preference only breaks ties among rows in force."""
    root = str(tmp_path / "reg")
    # pair 1: category row in force from t(1); eval row takes effect at t(500)
    cat1 = _policy("p_cat1", "synthmodel/static/one", temperature=1.0, top_p=0.95)
    ev1 = _policy("p_eval1", "synthmodel/syntheval/one", eval_id=synth.EVAL_ID, effective=synth.t(500), temperature=1.0, top_p=0.95)
    # pair 2: eval row in force from t(1); category row takes effect at t(500)
    ev2 = _policy("p_eval2", "synthmodel/syntheval/two", eval_id=synth.EVAL_ID, temperature=0.3, top_p=0.5)
    cat2 = _policy("p_cat2", "synthmodel/static/two", effective=synth.t(500), temperature=0.3, top_p=0.5)
    early1 = _run("early1", {"temperature": 1.0, "top_p": 0.95}, i0=100)   # generated t(103): only the category row governed it
    late1 = _run("late1", {"temperature": 1.0, "top_p": 0.95}, i0=600)     # generated t(603): both in force -> the eval row, by tie-break
    early2 = _run("early2", {"temperature": 0.3, "top_p": 0.5}, i0=100)    # only the eval row in force: exact by the SQL object alone
    _ingest(root, [cat1, ev1, ev2, cat2], [early1, late1, early2])
    conn = DB.connect(DB.db_path(root), readonly=True)
    row, view = _row(conn, early1[0])
    assert (row["decode_policy_id"], row["decode_policy_match"], view["match"]) == ("p_cat1", "exact", "exact")
    assert json.loads(view["candidates_json"]) == ["p_cat1", "p_eval1"] and _tiebreak(row) is None
    row, view = _row(conn, late1[0])
    assert (row["decode_policy_id"], row["decode_policy_match"], view["match"]) == ("p_eval1", "exact", "ambiguous")
    assert _tiebreak(row) == {"over": ["p_cat1"], "rule": "eval_over_category"}
    row, view = _row(conn, early2[0])
    assert (row["decode_policy_id"], row["decode_policy_match"], view["match"]) == ("p_eval2", "exact", "exact")
    assert _tiebreak(row) is None
    conn.close()


def test_no_llama_twin_co_matches_its_off_row_and_the_one_seeded_eval_category_tie_resolves_to_the_eval_row(idx, seeded_db):
    """The real seed (`configs/registry/decode_policy_seed.yaml`, the Llama `not_available` twins included), through the SQL
    object and the tie-break: a twin and its `off` row are never in scope for one run (the run's recorded thinking state admits
    exactly one of them), and the ONE tie the twins expose (the IFEval twin beside the pre-existing `capability` category row, same
    numbers) resolves to the eval-scoped twin -- so the seeded table leaves no run ambiguous."""
    conn = sqlite3.connect(seeded_db)
    conn.row_factory = sqlite3.Row
    try:
        twins = LT._twins(idx)
        assert len(twins) == 18, sorted(twins)   # 16 + the two `vendor` defaults of 2026-09-21 (agentic, em_organism; test_registry_decode_llama_thinking_na)
        # a TEMP `run` shadows the real (empty) table for the SQL object's body and the tie-break's own read
        conn.execute("CREATE TEMP TABLE run (run_id TEXT PRIMARY KEY, campaign TEXT, model_id TEXT, eval_id TEXT, thinking_mode TEXT, decode_seed_rule TEXT, "
                     "decode_sent_json TEXT, decode_temperature_sent REAL, decode_top_p_sent REAL, decode_top_k_sent INTEGER, "
                     "generated_utc TEXT, started_utc TEXT, queued_utc TEXT, decode_source TEXT)")   # decode_source: read by the by-source rule (0021)
        probes = LT._probe_runs(idx, conn)
        for rid, eval_id, thinking, seed_rule, sent, generated, _ in probes:
            conn.execute("INSERT INTO run VALUES (?, 'probe', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'generation_config')",
                         (rid, LT.MODEL, eval_id, thinking, seed_rule, json.dumps(sent), sent.get("temperature"), sent.get("top_p"), sent.get("top_k"),
                          generated, generated, generated))
        source = f"({DM.match_select_sql()})"
        got = {r["run_id"]: dict(r) for r in conn.execute(f"SELECT run_id, match, candidates_json, in_scope_json FROM {source}").fetchall()}
        for rid in got:
            scope = set(json.loads(got[rid]["in_scope_json"]))
            for tid in twins:
                assert not ({tid, LT._sibling_id(tid)} <= scope), (rid, tid, sorted(scope))
        broken = DM.tiebreak(conn, source)
        assert set(broken) == set(LT.KNOWN_AMBIGUOUS), broken
        ((rid, cands),) = LT.KNOWN_AMBIGUOUS.items()
        e = broken[rid]
        assert e["policy_id"] == "llama8b_it|ifeval|not_available|vendor_preset@v1"
        assert e["over"] == ["llama8b_it|capability|not_available|card@v1"]
        assert e["candidates"] == cands and sorted(e["in_force"]) == sorted(cands) and e["eval_scoped"] == [e["policy_id"]]
        assert not [rid for rid, e in broken.items() if e["policy_id"] is None]     # nothing the seeded table leaves ambiguous
    finally:
        conn.close()


def test_the_scoped_chain_equals_the_whole_table_result_on_its_runs_and_the_unscoped_path_is_unchanged(tmp_path):
    """`scoped_match_select_sql(run_ids)` restricts the SQL object's chain INSIDE its CTEs (a `scoped_run` CTE every `run` read reads),
    so a claim's re-stamp materializes nothing over the whole table (live: ~12,500 runs x ~160 policies -> 0.81 s SQL object + 13.9 s tie-break
    per claim before; 0.6 ms + 0.3 ms scoped). Its rows equal the whole object's rows for those runs; `stamp_run_rows(run_ids=…)` writes
    the same stamps the whole-table pass writes for them and touches no other row; with `run_ids=None` the pass is the one it always was."""
    root = str(tmp_path / "reg")
    policies = [_policy("p_cat", "cat", temperature=0.7, top_p=0.9), _policy("p_eval", "ev", eval_id=synth.EVAL_ID, temperature=0.7, top_p=0.9),
                _policy("p_other", "other", temperature=0.2)]
    runs = [_run("amb", {"temperature": 0.7, "top_p": 0.9}, i0=100), _run("exact", {"temperature": 0.2}, i0=200),
            _run("none", {"temperature": 0.5}, i0=300), _run("unrec", {}, i0=400)]
    _ingest(root, policies, runs)
    ids = [rid for rid, _ in runs]
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        whole = {r["run_id"]: tuple(r) for r in conn.execute(f"SELECT * FROM {DM.MATCH_SQL_NAME} ORDER BY run_id")}
        for subset in ([ids[0]], ids[:2], ids):
            scoped = {r["run_id"]: tuple(r) for r in conn.execute(f"SELECT * FROM {DM.scoped_match_select_sql(subset)} m ORDER BY run_id")}
            assert scoped == {k: whole[k] for k in subset}, subset
            assert DM.tiebreak(conn, DM.scoped_match_select_sql(subset)) == {k: v for k, v in DM.tiebreak(conn).items() if k in subset}
        assert set(whole) >= set(ids) and whole[ids[0]][2] == "ambiguous"       # the SQL object reads the tie; the Python tie-break resolves it
        assert DM.tiebreak(conn)[ids[0]]["policy_id"] == "p_eval"                 # the eval row beats the category row, at stamp time
        # a literal quote in an id is quoted, not spliced
        assert conn.execute(f"SELECT COUNT(*) FROM {DM.scoped_match_select_sql(['x' + chr(39) + 'y'])} m").fetchone()[0] == 0
    finally:
        conn.close()
    # the stamps: a scoped pass over one run writes exactly what the whole-table pass writes for it, and nothing else
    import shutil
    a_path, b_path = str(tmp_path / "a.sqlite"), str(tmp_path / "b.sqlite")
    shutil.copy2(DB.db_path(root), a_path)
    shutil.copy2(DB.db_path(root), b_path)
    a = DB.connect(a_path); b = DB.connect(b_path)
    try:
        for c in (a, b):      # blank the derived columns so both passes have work
            with DB.transaction(c):
                c.execute("UPDATE run SET decode_policy_id = NULL, decode_policy_match = 'no_policy', "
                          "field_versions_json = json_remove(field_versions_json, '$.decode_policy_id', '$.decode_policy_match'), "
                          "extra_json = json_remove(extra_json, '$.' || ?)", (DM.TIEBREAK_KEY,))
        with DB.transaction(a):
            n_all = DM.stamp_run_rows(a)
        with DB.transaction(b):
            n_one = DM.stamp_run_rows(b, run_ids=[ids[0]])
            assert DM.stamp_run_rows(b, run_ids=[]) == 0
        assert n_all == len(ids) and n_one == 1
        cols = "run_id, decode_policy_id, decode_policy_match, field_versions_json, extra_json"
        row = lambda c, rid: tuple(c.execute(f"SELECT {cols} FROM run WHERE run_id = ?", (rid,)).fetchone())
        assert row(a, ids[0]) == row(b, ids[0])                                       # the same stamp, the same tie-break audit
        for other in ids[1:]:
            assert row(b, other)[1:3] == (None, "no_policy") and row(a, other)[2] != "no_policy"   # untouched by the scoped pass
        with DB.transaction(b):
            assert DM.stamp_run_rows(b) == len(ids) - 1                                  # the whole-table pass then finishes the rest
        assert {rid: row(a, rid) for rid in ids} == {rid: row(b, rid) for rid in ids}
    finally:
        a.close(); b.close()
