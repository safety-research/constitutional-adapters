"""`registry cap-check`: the read-only report on the token-cap ladder facts of the live runs -- the record
rule re-applied to the live rows, the legacy shapes the rule admits, the instrument cross-check the fold never does -- against a
synthetic registry: one run per shape, the counts and the listed ids, `--campaign`, `--json`, no `state/last_run` stamp; a record the
rule refuses lands in `record_refusal` naming its field and folds nothing (a boolean index among them: read as stated, never as the
normalizer's 1); rebuild equality; a live row the validator would refuse (made by a direct edit of the synthetic database, the only way
one can exist) exits 1 and is listed. The instrument cross-check reaches every instrument with a ladder -- a cap outside a SINGLE-level
instrument's ladder is listed -- and counts the runs on an instrument with no ladder as unchecked.
"""
from __future__ import annotations

import json
import os
import random
import shutil

from sisteer.registry import capcheck as CC
from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import identity
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import validate as VAL

LADDER = [4096, 8192, 16384]
#: a multi-level instrument read at its final level, the shape of the escalation ladders (its id differs from synth.INSTRUMENT's: the
#: cap and the ladder are instrument identity)
LADDER_INSTRUMENT = synth.instrument_fields(token_cap_tokens=16384, token_cap_ladder_json=json.dumps(LADDER), read_at="ladder_final",
                                            ladder_promotion_rule="share_at_cap > 0.01")
LADDER_INSTRUMENT_ID = str(LADDER_INSTRUMENT["instrument_id"])
#: a related shape: a [512, 2048, 8192] instrument whose runs were drawn at 16,384
SHORT_LADDER = [512, 2048, 8192]
SHORT_INSTRUMENT = synth.instrument_fields(token_cap_tokens=8192, token_cap_ladder_json=json.dumps(SHORT_LADDER), read_at="ladder_final",
                                           ladder_promotion_rule="share_at_cap > 0.01")
SHORT_INSTRUMENT_ID = str(SHORT_INSTRUMENT["instrument_id"])
#: an instrument with no ladder at all (the column's default, `[]`): a run on it has no ladder to be outside of, and is counted as unchecked
NO_LADDER_INSTRUMENT = synth.instrument_fields(token_cap_tokens=2048, token_cap_ladder_json="[]", read_at="fixed")
NO_LADDER_INSTRUMENT_ID = str(NO_LADDER_INSTRUMENT["instrument_id"])
assert len({synth.INSTRUMENT_ID, LADDER_INSTRUMENT_ID, SHORT_INSTRUMENT_ID, NO_LADDER_INSTRUMENT_ID}) == 4


def _run(key, i0, instrument_id, **over):
    """A generated run on ``instrument_id`` with the cap facts ``over`` states (run_id minted over that instrument)."""
    rid = identity.run_id(synth.CAMPAIGN, key, "none", synth.EVAL_ID, instrument_id)
    return synth.run_records(key, i0, through="generated", instrument_id=instrument_id, run_id=rid, **over)


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _seed(root):
    """Dimensions plus the three extra instruments, then one run per shape. Returns the run ids by shape."""
    H.ensure_layout(root)
    dims = synth.seed_dimensions() + [synth.rec("instrument", LADDER_INSTRUMENT, synth.t(12)), synth.rec("instrument", SHORT_INSTRUMENT, synth.t(13)),
                                      synth.rec("instrument", NO_LADDER_INSTRUMENT, synth.t(14))]
    _write(root, "dims", "000001-dims.jsonl", dims)
    ids = {}
    # A: the per-level registration: index, ladder and cap agree -- the one shape the record rule reaches
    ids["per_level"], recs = _run("k_per_level", 100, LADDER_INSTRUMENT_ID, token_cap_tokens=8192, token_cap_rung_index=1, token_cap_ladder_json=LADDER)
    # B: legacy, a ladder stated with no index, the cap in it
    ids["ladder_no_index"], more = _run("k_ladder_no_index", 120, LADDER_INSTRUMENT_ID, token_cap_tokens=8192, token_cap_ladder_json=LADDER)
    recs += more
    # C: legacy, an empty ladder with no index on the multi-level instrument (the planned final cap stamped)
    ids["empty_on_multi"], more = _run("k_empty_on_multi", 140, LADDER_INSTRUMENT_ID, token_cap_tokens=16384)
    recs += more
    # D: a cap outside the instrument's ladder (a 16,384 draw on a [512, 2048, 8192] instrument), the run's own ladder stated
    ids["cap_outside"], more = _run("k_cap_outside", 160, SHORT_INSTRUMENT_ID, token_cap_tokens=16384, token_cap_ladder_json=SHORT_LADDER)
    recs += more
    # E: a run on the single-level default instrument with no ladder at all (synth's default: cap 512, ladder [512] on the instrument)
    ids["single_level"], more = synth.run_records("k_single", 180, through="generated")
    recs += more
    # F: a cap outside a SINGLE-level instrument's ladder -- an 8,192 draw on the default [512] instrument, no ladder and no index of its
    # own, so nothing but the instrument cross-check can name it (a check scoped to multi-level instruments passed it)
    ids["single_level_outside"], more = synth.run_records("k_single_outside", 190, through="generated", token_cap_tokens=8192)
    recs += more
    # G: a run on an instrument with NO ladder: nothing to be outside of; counted as unchecked, never passed
    ids["no_ladder"], more = _run("k_no_ladder", 195, NO_LADDER_INSTRUMENT_ID, token_cap_tokens=2048)
    recs += more
    _write(root, "runs", "000002-run.jsonl", [r for r in recs if r["record_type"] == "run"])
    _write(root, "runs", "000003-run_state.jsonl", [r for r in recs if r["record_type"] == "run_state"])
    _write(root, "runs", "000004-run_patch.jsonl", [r for r in recs if r["record_type"] == "run_patch"])
    return ids


def _rebuild_equal(root, tmp_path, shuffle_seed=3):
    """Rebuild from a differently enumerated copy of the log; every table equal (the reducer test's helper)."""
    copy = str(tmp_path / f"copy{shuffle_seed}")
    files = list(R.iter_record_files(os.path.join(root, "records")))
    random.Random(shuffle_seed).shuffle(files)
    for f in files:
        rel = os.path.relpath(f, os.path.join(root, "records"))
        dest = os.path.join(copy, "records", rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(f, dest)
    out = str(tmp_path / f"rebuilt{shuffle_seed}.sqlite")
    H.rebuild(copy, out, workers=0)
    a = DB.connect(DB.db_path(root), readonly=True)
    b = DB.connect(out, readonly=True)
    try:
        diff = H.verify_equal(a, b)
        assert not {k: v for k, v in diff.items() if v}, {k: v for k, v in diff.items() if v}
    finally:
        a.close()
        b.close()


def _report(root, *args, capsys):
    rc = cli.main(["--root", root, "cap-check", "--json", *args])
    return rc, json.loads(capsys.readouterr().out)


def test_cap_check_reports_the_rule_the_legacy_shapes_and_the_instrument_cross_check(tmp_path, capsys):
    root = str(tmp_path / "reg")
    ids = _seed(root)
    rep0 = H.ingest_once(root, workers=0, min_free=0)
    assert rep0.refused == 0 and rep0.blocked == 0, rep0.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 7
    row = conn.execute("SELECT token_cap_tokens, token_cap_rung_index, token_cap_ladder_json FROM run WHERE run_id = ?", (ids["per_level"],)).fetchone()
    assert tuple(row) == (8192, 1, R.canonical_json(LADDER))
    conn.close()

    rc, rep = _report(root, capsys=capsys)
    assert rc == 0 and rep["exit_code"] == 0 and rep["decision"] == "D648" and rep["runs"] == 7 and rep["campaign"] is None
    # the record rule on the live rows: the per-level run alone states an index, and agrees with itself
    assert rep["rule"] == {"index_stated": 1, "violations": []}
    # the legacy shapes
    lsn = rep["shapes"]["ladder_stated_no_index"]
    assert lsn["count"] == 2 and lsn["cap_in_own_ladder"] == 1                       # B (cap in), D (16,384 outside [512, 2048, 8192])
    assert [r["run_id"] for r in lsn["cap_outside_own_ladder"]] == [ids["cap_outside"]]     # F states no ladder of its own: not here
    len_ = rep["shapes"]["ladder_empty_no_index"]
    assert len_["count"] == 4 and len_["on_single_level_instrument"] == 2             # E and F on [512]; C on the multi-level one; G on none
    assert len_["on_instrument_without_a_ladder"] == 1
    assert len_["on_multi_level_instrument"]["count"] == 1 and len_["on_multi_level_instrument"]["by_campaign"] == {synth.CAMPAIGN: 1}
    assert [r["run_id"] for r in len_["on_multi_level_instrument"]["runs"]] == [ids["empty_on_multi"]]
    assert len_["on_multi_level_instrument"]["runs"][0]["instrument_ladder"] == LADDER       # the list; the column's text keeps the producer's spacing
    assert json.loads(len_["on_multi_level_instrument"]["runs"][0]["instrument_ladder_json"]) == LADDER
    assert rep["shapes"]["no_index_on_multi_level_instrument"] == {"count": 3, "by_campaign": {synth.CAMPAIGN: 3}}   # B, C, D
    # the instrument cross-check reaches every instrument with a ladder: the 16,384 draw on the [512, 2048, 8192] instrument AND the 8,192
    # draw on the single-level [512] one (the latter named nowhere else: its own ladder is empty, and the aggregate declared-cap count
    # does not identify it); the run on the ladderless instrument is counted as unchecked, never listed as passing
    inst = rep["instrument"]
    assert inst["instruments_with_a_ladder"] == 3 and inst["multi_level_instruments"] == 2
    assert inst["runs_on_instruments_with_a_ladder"] == 6 and inst["runs_on_multi_level_instruments"] == 4
    assert inst["runs_on_instruments_without_a_ladder"] == 1
    assert inst["cap_outside_instrument_ladder_count"] == 2
    assert sorted(r["run_id"] for r in inst["cap_outside_instrument_ladder"]) == sorted([ids["cap_outside"], ids["single_level_outside"]])
    by_id = {r["run_id"]: r for r in inst["cap_outside_instrument_ladder"]}
    listed = by_id[ids["cap_outside"]]
    assert listed["token_cap_tokens"] == 16384 and listed["instrument_token_cap_tokens"] == 8192 and listed["instrument_ladder"] == SHORT_LADDER
    assert listed["instrument_read_at"] == "ladder_final" and listed["lifecycle_state"] == "generated"
    single = by_id[ids["single_level_outside"]]
    assert single["token_cap_tokens"] == 8192 and single["instrument_token_cap_tokens"] == 512 and single["instrument_ladder"] == [512]
    assert single["instrument_read_at"] == "fixed" and single["token_cap_rung_index"] is None and json.loads(single["token_cap_ladder_json"]) == []
    assert inst["run_ladder_differs_from_instrument"] == [] and inst["run_ladder_differs_from_instrument_count"] == 0
    assert inst["cap_differs_from_instrument_declared_cap"] == 4                         # A and B (8,192 below the final 16,384), D (16,384 above 8,192), F (8,192 above 512); G equals its 2,048
    # the text form names the listed runs and the counts; the bound is stated
    assert cli.main(["--root", root, "cap-check"]) == 0
    text = capsys.readouterr().out
    assert ids["cap_outside"] in text and "cap outside the instrument's ladder [512, 2048, 8192]" in text
    assert ids["single_level_outside"] in text and "cap outside the instrument's ladder [512] (declared cap 512, read at fixed)" in text
    assert ("3 instruments with a ladder (2 of them multi-level), 6 runs on them checked (4 on a multi-level instrument), 1 on an instrument "
            "with no ladder, unchecked; cap outside the instrument's ladder: 2;") in text
    assert "an empty ladder with no index: 4 runs (2 on a single-level instrument, 1 on a multi-level instrument" in text
    assert "1 on an instrument with no ladder)" in text
    assert "1 state a ladder index, 0 disagree" in text and "first 20 rows" in text
    assert ids["empty_on_multi"] in text and ids["per_level"] not in text and ids["no_ladder"] not in text
    # --all lifts the bound; --campaign scopes every count
    rc, rep_all = _report(root, "--all", capsys=capsys)
    assert rc == 0 and rep_all["limit"] is None and rep_all["shapes"] == rep["shapes"]
    assert cli.main(["--root", root, "cap-check", "--all"]) == 0 and "first 20 rows" not in capsys.readouterr().out
    rc, rep_none = _report(root, "--campaign", "nonesuch", capsys=capsys)
    assert rc == 0 and rep_none["runs"] == 0 and rep_none["rule"]["index_stated"] == 0 and rep_none["instrument"]["cap_outside_instrument_ladder"] == []
    assert rep_none["instrument"]["multi_level_instruments"] == 2 and rep_none["instrument"]["instruments_with_a_ladder"] == 3   # instruments are not campaign-scoped
    assert rep_none["instrument"]["runs_on_instruments_with_a_ladder"] == 0 and rep_none["instrument"]["runs_on_instruments_without_a_ladder"] == 0
    rc, rep_c = _report(root, "--campaign", synth.CAMPAIGN, capsys=capsys)
    assert rc == 0 and rep_c["runs"] == 7 and rep_c["shapes"] == rep["shapes"] and rep_c["campaign"] == synth.CAMPAIGN
    # read-only: no state/last_run stamp (the timers' health record is for commands that write)
    assert not os.path.exists(os.path.join(root, "state", "last_run", "cap-check.json"))
    # the module function on a bare connection gives the CLI's dict
    with DB.reading(DB.db_path(root)) as conn:
        assert CC.report(conn)["shapes"] == rep["shapes"] and CC.report(conn, campaign="nonesuch")["runs"] == 0
    _rebuild_equal(root, tmp_path)


def test_a_record_the_rule_refuses_is_a_refusal_naming_its_field_and_folds_nothing(tmp_path, capsys):
    root = str(tmp_path / "reg")
    ids = _seed(root)
    H.ingest_once(root, workers=0, min_free=0)
    # a re-registration of the per-level run that says level 0 while keeping the 8,192 cap: refused at the door, the live row untouched
    bad_rid, bad = _run("k_per_level", 200, LADDER_INSTRUMENT_ID, token_cap_tokens=8192, token_cap_rung_index=0, token_cap_ladder_json=LADDER)
    assert bad_rid == ids["per_level"]
    bad_run = [r for r in bad if r["record_type"] == "run"]
    assert isinstance(VAL.validate_record(bad_run[0], "<p>"), VAL.Malformed)
    # and a new run whose ladder is not a ladder
    _, worse = _run("k_descending", 220, LADDER_INSTRUMENT_ID, token_cap_tokens=8192, token_cap_rung_index=0, token_cap_ladder_json=[8192, 4096])
    # and the per-level run re-registered with a BOOLEAN index: the normalizer's rendering (true -> 1) agrees with the ladder and the cap, the
    # record as stated does not -- refused naming the index, never folded as a level-1 fact (the live-row re-check below could not have
    # seen the folded 1)
    bool_rid, as_bool = _run("k_per_level", 240, LADDER_INSTRUMENT_ID, token_cap_tokens=8192, token_cap_rung_index=True, token_cap_ladder_json=LADDER)
    assert bool_rid == ids["per_level"]
    bool_run = [r for r in as_bool if r["record_type"] == "run"]
    assert json.loads(R.canonical_json(bool_run[0]["payload"]))["token_cap_rung_index"] is True       # written as JSON `true`
    _write(root, "runs", "000005-run.jsonl", bad_run + [r for r in worse if r["record_type"] == "run"] + bool_run)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 3, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    refusals = {r[0]: r[1] for r in conn.execute("SELECT field, reason FROM record_refusal").fetchall()}
    assert set(refusals) == {"token_cap_tokens", "token_cap_ladder_json", "token_cap_rung_index"}
    assert "= 4096" in refusals["token_cap_tokens"] and "D648" in refusals["token_cap_tokens"]
    assert "strictly ascending" in refusals["token_cap_ladder_json"]
    assert "token_cap_rung_index=True must be an integer" in refusals["token_cap_rung_index"] and "D648" in refusals["token_cap_rung_index"]
    assert conn.execute("SELECT COUNT(*) FROM run").fetchone()[0] == 7
    row = conn.execute("SELECT token_cap_tokens, token_cap_rung_index, token_cap_ladder_json FROM run WHERE run_id = ?", (ids["per_level"],)).fetchone()
    assert tuple(row) == (8192, 1, R.canonical_json(LADDER))
    conn.close()
    rc, out = _report(root, capsys=capsys)
    assert rc == 0 and out["rule"] == {"index_stated": 1, "violations": []} and out["runs"] == 7
    _rebuild_equal(root, tmp_path)


def test_a_live_row_the_validator_would_refuse_exits_1_and_is_listed(tmp_path, capsys):
    """The one finding that exits 1: the live table holding a row a replay would refuse. No record can produce one (the validator refuses
    it at the door), so the synthetic database is edited directly here -- a test-only edit of a temporary registry, never the live one."""
    root = str(tmp_path / "reg")
    ids = _seed(root)
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root))
    conn.execute("UPDATE run SET token_cap_tokens = 4096 WHERE run_id = ?", (ids["per_level"],))
    conn.commit()
    conn.close()
    rc, rep = _report(root, capsys=capsys)
    assert rc == 1 and rep["exit_code"] == 1
    assert rep["rule"]["index_stated"] == 1 and len(rep["rule"]["violations"]) == 1
    v = rep["rule"]["violations"][0]
    assert v["run_id"] == ids["per_level"] and v["field"] == "token_cap_tokens" and "= 8192" in v["reason"] and v["token_cap_tokens"] == 4096
    assert cli.main(["--root", root, "cap-check"]) == 1
    text = capsys.readouterr().out
    assert "1 disagree with it -- the live table disagrees with the validator" in text and ids["per_level"] in text
    # scoped away, the finding is gone and the exit is 0
    rc, rep = _report(root, "--campaign", "nonesuch", capsys=capsys)
    assert rc == 0 and rep["rule"]["violations"] == []


def test_cap_check_without_a_database_fails_loudly(tmp_path, capsys):
    root = str(tmp_path / "empty")
    os.makedirs(root)
    assert cli.main(["--root", root, "cap-check"]) == 1
    assert "cap-check:" in capsys.readouterr().err
