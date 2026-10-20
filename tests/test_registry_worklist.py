"""The registry-driven worklist helper the judging drivers share, the CLI's queue and parity
commands, judge resolution, the claim race across processes, and worklist parity on synthetic units.

* ``RegistryWorklist`` claims every run before grading, refuses whole on a claim another session holds
  (releasing what it took), completes the CLAIMED judgment row (no second row) and writes ``judged`` visible
  immediately; a missing database / unknown judge / disabled emission refuses (never the disk listing).
* ``registry pending-judgment`` resolves a seeded readable judge id, lists reasons, claims and releases.
* ``registry worklist-parity`` compares the pending set to a units file set-wise with reasons (the grading
  queue's grain: store → object → eval), exit 0 iff no differences.
* A cached claim is reused only while it is still this session's in-progress claim bound to the run's CURRENT folded identity:
  a run identity correction folding between ``claim_all`` and ``scope`` -- or before a reused
  instance claims again -- evicts it and the row is re-claimed through ``pending.claim`` (a current-bound id), as does a release or
  another session's takeover; the scope never completes an old-bound id.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import emit
from sisteer.registry import harvest as H
from sisteer.registry import identity as I
from sisteer.registry import pending as P
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import worklist as W

SEED_ID = "synthetic_judge_two"   # the readable id judges.yaml would carry; the loader parks it in extra_json
#: two more instruments of the seeded eval naming the SAME required judge (other token caps): the identities a same-id run correction
#: moves a run through while a worklist holds its claim
INSTRUMENT_X2 = synth.instrument_fields(token_cap_tokens=1024, token_cap_ladder_json="[1024]", notes="the corrected instrument")
INSTRUMENT_X2_ID = str(INSTRUMENT_X2["instrument_id"])
INSTRUMENT_X3 = synth.instrument_fields(token_cap_tokens=2048, token_cap_ladder_json="[2048]", notes="the corrected instrument, again")
INSTRUMENT_X3_ID = str(INSTRUMENT_X3["instrument_id"])
assert len({synth.INSTRUMENT_ID, INSTRUMENT_X2_ID, INSTRUMENT_X3_ID}) == 3 and INSTRUMENT_X2["judge_id"] == INSTRUMENT_X3["judge_id"] == synth.JUDGE_ID


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _dims_with_seed_id():
    dims = synth.seed_dimensions()
    for r in dims:
        if r["record_type"] == "judge" and r["payload"]["judge_id"] == synth.JUDGE_ID_2:
            r["payload"]["seed_judge_id"] = SEED_ID     # an unknown key parks in extra_json
    return dims


def _root_with(tmp_path, runs, monkeypatch=None):
    """A registry with the dimension seed and ``runs`` = [(key, through, store_path)] ingested."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    _write(root, "dims", "000001-dims.jsonl", _dims_with_seed_id())
    rids = []
    for k, (key, through, store) in enumerate(runs):
        rid, recs = synth.run_records(key, 100 + 20 * k, through=through)
        if store is not None:
            recs[0]["payload"]["store_uri"] = store
        _write(root, "c", f"00000{k + 1}-run.jsonl", recs)
        rids.append(rid)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.as_dict()
    if monkeypatch is not None:
        monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
        monkeypatch.setenv("REGISTRY_ROOT", root)
    return root, rids


# --------------------------------------------------------------------------------------------
# judge resolution and filters
# --------------------------------------------------------------------------------------------


def test_resolve_judge_id_accepts_hash_or_seed_id_and_refuses_unknown(tmp_path):
    root, _ = _root_with(tmp_path, [("k", "generated", None)])
    with DB.reading(DB.db_path(root)) as conn:
        assert P.resolve_judge_id(conn, synth.JUDGE_ID_2) == synth.JUDGE_ID_2
        assert P.resolve_judge_id(conn, SEED_ID) == synth.JUDGE_ID_2
        with pytest.raises(P.UnknownJudge) as ei:
            P.resolve_judge_id(conn, "no_such_judge")
        assert "no_such_judge" in str(ei.value) and "judge rows" in str(ei.value)
        # the same via pending_for, and an unknown filter column refuses rather than silently widening
        assert [r["run_id"] for r in P.pending_for(conn, SEED_ID)] == [r["run_id"] for r in P.pending_for(conn, synth.JUDGE_ID_2)]
        with pytest.raises(P.UnknownFilter):
            P.pending_for(conn, SEED_ID, not_a_column="x")
        assert P.pending_for(conn, SEED_ID, eval_category="static_attack", campaign=synth.CAMPAIGN)
        assert P.pending_for(conn, SEED_ID, eval_category=("petri", "agentic")) == []
        assert P.pending_for(conn, SEED_ID, decode_source_not="greedy") == []     # the synthetic instrument is greedy


# --------------------------------------------------------------------------------------------
# the driver helper
# --------------------------------------------------------------------------------------------


def _stub_grade(row, scope):
    scope.items([{"item_id": f"q{i:05d}", "metric_id": "asr", "value": float(i % 2), "label": "harmful" if i % 2 else "safe"}
                 for i in range(200)])
    scope.metrics([{"metric_id": "asr", "threshold": 0.5, "value": 0.5, "k": 100, "n": 200}])


def test_registry_worklist_claims_completes_claimed_row_and_writes_judged(tmp_path, monkeypatch, capsys):
    root, (rid,) = _root_with(tmp_path, [("k", "generated", "file:///stores/k")], monkeypatch)
    wl = W.RegistryWorklist(root, [SEED_ID], "session-A", driver="tests/stub_driver.py", api_pool="lowprio")
    assert wl.judges == [synth.JUDGE_ID_2]
    rows = wl.pending(eval_category="static_attack")
    assert [r["run_id"] for r in rows] == [rid] and rows[0]["pending_reason"] == "no_judgment"
    done = wl.grade(rows, _stub_grade)
    assert done == [rid]
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)   # the harvester's fold of the spooled completion records (finish() folds nothing by default, 2026-09-30)
    conn = DB.connect(DB.db_path(root), readonly=True)
    js = conn.execute("SELECT judgment_id, status, n_items_judged, claimed_by_session, api_pool FROM judgment WHERE run_id = ? AND judge_id = ?",
                      (rid, synth.JUDGE_ID_2)).fetchall()
    assert len(js) == 1, js                                        # ONE judgment row: the claim, completed
    assert tuple(js[0][1:]) == ("complete", 200, "session-A", "lowprio")
    assert js[0][0] == wl.claims[(rid, synth.JUDGE_ID_2)]["judgment_id"]
    run = conn.execute("SELECT lifecycle_state, n_items_judged, judge_id FROM run WHERE run_id = ?", (rid,)).fetchone()
    assert run[0] == "judged" and run[1] == 200 and run[2] == synth.JUDGE_ID_2
    assert conn.execute("SELECT COUNT(*) FROM item_result WHERE run_id = ? AND judgment_id = ?", (rid, js[0][0])).fetchone()[0] == 200
    assert P.pending_for(conn, SEED_ID) == []                   # no longer pending for that judge
    assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    conn.close()
    assert W.FROM_DISK_LINE not in capsys.readouterr().err       # the registry path never prints the bypass line


def test_second_driver_refuses_on_the_claim_and_releases_what_it_took(tmp_path, monkeypatch):
    root, (r1, r2) = _root_with(tmp_path, [("k1", "generated", None), ("k2", "generated", None)], monkeypatch)
    a = W.RegistryWorklist(root, [SEED_ID], "session-A", driver="a")
    rows = a.pending()
    a.claim(rows[0] if rows[0]["run_id"] == r1 else rows[1])       # A holds r1 only
    b = W.RegistryWorklist(root, [SEED_ID], "session-B", driver="b")
    with pytest.raises(P.ClaimHeld) as ei:
        b.claim_all(sorted(rows, key=lambda r: r["run_id"] != r2))   # B takes r2 first, then meets A's claim on r1
    assert ei.value.run_id == r1 and ei.value.session == "session-A"
    conn = DB.connect(DB.db_path(root), readonly=True)
    st = {(r[0], r[1]): r[2] for r in conn.execute("SELECT run_id, claimed_by_session, status FROM judgment")}
    assert st[(r1, "session-A")] == "in_progress" and st[(r2, "session-B")] == "abandoned"   # B's own claim released
    rows2 = P.pending_for(conn, SEED_ID)
    by = {r["run_id"]: r for r in rows2}
    assert by[r1]["pending_reason"] == "claimed" and by[r1]["claim_session"] == "session-A" and by[r1]["claim_age_s"] >= 0
    assert by[r2]["pending_reason"] == "abandoned" and by[r2]["n_abandoned"] == 1
    conn.close()
    assert b.claims == {}


def test_registry_worklist_refuses_without_database_unknown_judge_or_disabled_emission(tmp_path, monkeypatch):
    with pytest.raises(W.RegistryUnavailable) as ei:
        W.RegistryWorklist(str(tmp_path / "nowhere"), None, "s", driver="d")
    assert "no registry database" in str(ei.value) and "never falls back" in str(ei.value)
    root, (rid,) = _root_with(tmp_path, [("k", "generated", None)], monkeypatch)
    with pytest.raises(W.RegistryUnavailable) as ei:
        W.RegistryWorklist(root, ["ghost_judge"], "s", driver="d")
    assert "ghost_judge" in str(ei.value)
    wl = W.RegistryWorklist(root, None, "s", driver="d")           # default: the instrument judge
    rows = wl.pending()
    assert [r["required_judge_id"] for r in rows] == [synth.JUDGE_ID]
    with pytest.raises(W.RegistryUnavailable):
        wl.pending(not_a_filter=1)
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    with pytest.raises(W.RegistryUnavailable) as ei:
        wl.scope(rows[0])
    assert "cannot record" in str(ei.value)
    # the claim the scope attempt took is still A's; release it
    assert wl.release(rows[0])


def test_refuse_unlocated_names_run_and_store():
    rows = [{"run_id": "r1", "store_uri": "file:///a"}, {"run_id": "r2", "store_uri": None}]
    W.refuse_unlocated("d", rows, {"r1": object(), "r2": object()})     # every run located: no refusal
    with pytest.raises(W.RegistryUnavailable) as ei:
        W.refuse_unlocated("d", rows, {"r1": object()})
    assert "r2" in str(ei.value) and "r1 " not in str(ei.value) and "cannot locate" in str(ei.value)


def test_from_disk_line_is_exact(capsys):
    W.log_from_disk()
    assert capsys.readouterr().err == W.FROM_DISK_LINE + "\n"
    assert W.DEFAULT_WORKLIST == "from-disk" and W.WORKLIST_CHOICES == ("registry", "from-disk")


# --------------------------------------------------------------------------------------------
# a cached claim is reused only under the run's current folded identity
# --------------------------------------------------------------------------------------------


def _correct_run(root, rid, run_key, instrument_id, i):
    """A same-id run identity correction onto ``instrument_id``, written and FOLDED (the harvester's cycle between two driver steps)."""
    corrected = synth.run_payload(run_key, i=100, instrument_id=instrument_id)
    corrected["run_id"] = rid
    _write(root, "c", f"{i:06d}-correction.jsonl", [synth.rec("run", corrected, synth.t(i))])
    rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(i + 1))
    assert rep.refused == 0, rep.as_dict()


def _claims_of(root, rid):
    with DB.reading(DB.db_path(root)) as conn:
        return {r[0]: {"status": r[1], "session": r[2], "instrument": r[3]} for r in conn.execute(
            "SELECT judgment_id, status, claimed_by_session, claimed_instrument_id FROM judgment WHERE run_id = ?", (rid,))}


def test_a_correction_between_claim_all_and_scope_re_claims_a_current_bound_judgment(tmp_path, monkeypatch):
    root, (rid,) = _root_with(tmp_path, [("k1", "generated", None)], monkeypatch)
    _write(root, "dims", "000002-more-instruments.jsonl", [synth.rec("instrument", INSTRUMENT_X2, synth.t(31)), synth.rec("instrument", INSTRUMENT_X3, synth.t(32))])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    w = W.RegistryWorklist(root, [synth.JUDGE_ID], "session-A", driver="a")
    rows = w.pending()
    assert [r["run_id"] for r in rows] == [rid]
    old = w.claim_all(rows)[(rid, synth.JUDGE_ID)]["judgment_id"]
    assert _claims_of(root, rid)[old] == {"status": "in_progress", "session": "session-A", "instrument": synth.INSTRUMENT_ID}
    # the harvester folds a correction between the claims and the first scope
    _correct_run(root, rid, "k1", INSTRUMENT_X2_ID, 3000)
    scope = w.scope(rows[0])
    new = scope.judgment_id
    claims = _claims_of(root, rid)
    assert new != old and claims[old]["status"] == "abandoned"                                   # the cached claim was evicted, not handed to the driver
    assert claims[new] == {"status": "in_progress", "session": "session-A", "instrument": INSTRUMENT_X2_ID}
    assert w.claims[(rid, synth.JUDGE_ID)]["judgment_id"] == new and scope.claim_registered and scope.claimed_utc == w.claims[(rid, synth.JUDGE_ID)]["claimed_utc"]
    # nothing moved: the same scope id again, no third claim
    assert w.scope(rows[0]).judgment_id == new and len(_claims_of(root, rid)) == 2
    # the scope completes the CURRENT-bound claim: its judgment is the paper row, the run is no longer pending for this judge
    with scope:
        scope.items([{"item_id": f"q{k:05d}", "metric_id": "asr", "value": 0.0} for k in range(200)])
        scope.metrics([{"metric_id": "asr", "threshold": 0.5, "value": 0.0, "k": 0, "n": 200}])
    assert H.ingest_once(root, workers=0, min_free=0, now=synth.t(5000)).refused == 0
    with DB.reading(DB.db_path(root)) as conn:
        assert conn.execute("SELECT status FROM judgment WHERE judgment_id = ?", (new,)).fetchone()[0] == "complete"
        assert [r[0] for r in conn.execute("SELECT judgment_id FROM v_paper_long WHERE run_id = ?", (rid,))] == [new]
        assert conn.execute("SELECT COUNT(*) FROM v_pending_judgment WHERE run_id = ? AND required_judge_id = ?", (rid, synth.JUDGE_ID)).fetchone()[0] == 0


def test_a_reused_worklist_re_claims_after_a_correction_a_release_or_a_takeover(tmp_path, monkeypatch):
    root, (r1, r2) = _root_with(tmp_path, [("k1", "generated", None), ("k2", "generated", None)], monkeypatch)
    _write(root, "dims", "000002-more-instruments.jsonl", [synth.rec("instrument", INSTRUMENT_X2, synth.t(31)), synth.rec("instrument", INSTRUMENT_X3, synth.t(32))])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    w = W.RegistryWorklist(root, [synth.JUDGE_ID], "session-A", driver="a")
    rows = {r["run_id"]: r for r in w.pending()}
    c1 = w.claim(rows[r1])["judgment_id"]
    c2 = w.claim(rows[r2])["judgment_id"]
    # a correction folded while the instance idles: the cached claim on r1 is stale, the next claim() evicts and re-mints it (r2 untouched)
    _correct_run(root, r1, "k1", INSTRUMENT_X2_ID, 3000)
    c1b = w.claim(rows[r1])["judgment_id"]
    assert c1b != c1 and _claims_of(root, r1)[c1]["status"] == "abandoned" and _claims_of(root, r1)[c1b]["instrument"] == INSTRUMENT_X2_ID
    assert w.claim(rows[r2])["judgment_id"] == c2
    # a second correction: again
    _correct_run(root, r1, "k1", INSTRUMENT_X3_ID, 4000)
    c1c = w.claim(rows[r1])["judgment_id"]
    assert c1c not in (c1, c1b) and _claims_of(root, r1)[c1c]["instrument"] == INSTRUMENT_X3_ID
    # a release behind the cache (another process of this session): the cached id is no longer in progress, so it is re-claimed (the
    # release is stamped with the clock, as the claims are: a synthetic stamp before the claim's would fold under it)
    assert P.release(root, r2, synth.JUDGE_ID, "session-A")
    c2b = w.claim(rows[r2])["judgment_id"]
    assert c2b != c2 and _claims_of(root, r2)[c2]["status"] == "abandoned" and _claims_of(root, r2)[c2b]["status"] == "in_progress"
    # another session's forced takeover of a stale-aged claim: the cached id is no longer this session's, and the re-claim meets the
    # other session's current-bound claim -- refused, the cache left empty for that row
    b = P.claim(root, r2, synth.JUDGE_ID, "session-B", force=True, stale_hours=0.0, now="2027-01-01T00:00:00.000Z")
    assert _claims_of(root, r2)[c2b]["status"] == "abandoned" and _claims_of(root, r2)[b]["session"] == "session-B"
    with pytest.raises(P.ClaimHeld) as ei:
        w.claim(rows[r2])
    assert ei.value.session == "session-B" and (r2, synth.JUDGE_ID) not in w.claims
    # claim_all on a mix of cached-and-live and evicted rows releases only what it took when a later row is held
    _correct_run(root, r1, "k1", INSTRUMENT_X2_ID, 6000)                                      # r1's cached claim stale again
    with pytest.raises(P.ClaimHeld):
        w.claim_all([rows[r1], rows[r2]])                                                       # r1 re-claimed (taken), r2 held by B
    claims1 = _claims_of(root, r1)
    assert sorted(v["status"] for v in claims1.values()) == ["abandoned"] * 4 and (r1, synth.JUDGE_ID) not in w.claims   # the re-claim it took was released


# --------------------------------------------------------------------------------------------
# the claim race: two processes, one run
# --------------------------------------------------------------------------------------------


def _try_claim(root, rid, judge, session, q):
    try:
        q.put((session, "ok", P.claim(root, rid, judge, session)))
    except P.ClaimHeld as exc:
        q.put((session, "held", exc.session))
    except Exception as exc:  # noqa: BLE001 — surfaced to the test
        q.put((session, "error", f"{type(exc).__name__}: {exc}"))


def test_claim_race_two_processes_exactly_one_wins(tmp_path):
    root, (rid,) = _root_with(tmp_path, [("k", "generated", None)])
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    procs = [ctx.Process(target=_try_claim, args=(root, rid, synth.JUDGE_ID_2, s, q)) for s in ("proc-A", "proc-B")]
    for p in procs:
        p.start()
    results = [q.get(timeout=120) for _ in procs]
    for p in procs:
        p.join(timeout=120)
    outcomes = {s: (kind, val) for s, kind, val in results}
    kinds = sorted(k for k, _ in outcomes.values())
    assert kinds == ["held", "ok"], outcomes
    winner = next(s for s, (k, _) in outcomes.items() if k == "ok")
    loser = next(s for s, (k, _) in outcomes.items() if k == "held")
    assert outcomes[loser][1] == winner
    conn = DB.connect(DB.db_path(root), readonly=True)
    rows = conn.execute("SELECT claimed_by_session, status FROM judgment WHERE run_id = ?", (rid,)).fetchall()
    assert [tuple(r) for r in rows] == [(winner, "in_progress")]
    conn.close()


# --------------------------------------------------------------------------------------------
# the CLI: pending-judgment with reasons and a seed id; claim / release
# --------------------------------------------------------------------------------------------


def test_cli_pending_judgment_lists_reasons_claims_and_releases(tmp_path, monkeypatch, capsys):
    root, (rid,) = _root_with(tmp_path, [("k", "generated", None)], monkeypatch)
    assert cli.main(["pending-judgment", "--judge", SEED_ID, "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["run_id"] for r in rows] == [rid] and rows[0]["pending_reason"] == "no_judgment" and rows[0]["required_coverage"] == 200
    assert cli.main(["pending-judgment", "--judge", "ghost"]) == 2
    assert "ghost" in capsys.readouterr().err
    assert cli.main(["pending-judgment", "--judge", SEED_ID, "--claim", "--run-id", rid, "--session", "cli-A"]) == 0
    jid = capsys.readouterr().out.strip()
    assert cli.main(["pending-judgment", "--judge", SEED_ID]) == 0
    text = capsys.readouterr().out
    assert "claimed" in text and "cli-A" in text and "pending_reason" in text
    assert cli.main(["pending-judgment", "--judge", SEED_ID, "--claim", "--run-id", rid, "--session", "cli-B"]) == 1
    assert capsys.readouterr().err.startswith("claim refused: ClaimHeld: ")
    assert cli.main(["pending-judgment", "--judge", SEED_ID, "--release", rid, "--session", "cli-A"]) == 0
    assert capsys.readouterr().out.strip() == "released"
    assert cli.main(["pending-judgment", "--judge", SEED_ID, "--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert rows[0]["pending_reason"] == "abandoned" and rows[0]["claim_session"] is None and rows[0]["n_abandoned"] == 1
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT status FROM judgment WHERE judgment_id = ?", (jid,)).fetchone()[0] == "abandoned"
    conn.close()


# --------------------------------------------------------------------------------------------
# worklist parity on synthetic units (acceptance "Grading queue")
# --------------------------------------------------------------------------------------------


def _units_file(path, units):
    doc = {"units": {u: {"grades_dir": g, "cell_id": c, "family": f, "source": synth.CAMPAIGN, "kind": "gated",
                         "cache_scheme": "live", "trial": 0, "view": "synthmodel"} for u, (g, c, f) in units.items()}}
    path.write_text(json.dumps(doc))
    return str(path)


def test_worklist_parity_matches_by_store_object_eval_and_lists_differences(tmp_path, monkeypatch, capsys):
    # three runs: k_a (pending for judge 2), k_b (already complete by judge 2), k_c (pending, no unit names it)
    root, (ra, rb, rc) = _root_with(tmp_path, [("k_a", "generated", None), ("k_b", "generated", "file:///grades/k_b"),
                                               ("k_c", "generated", None)], monkeypatch)
    jid_b = I.judgment_id(rb, synth.JUDGE_ID_2)
    _write(root, "c", "000009-complete.jsonl", [synth.rec("judgment", {"judgment_id": jid_b, "run_id": rb, "judge_id": synth.JUDGE_ID_2,
                                                                        "n_items_judged": 200, "status": "complete"}, synth.t(900))])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    units = _units_file(tmp_path / "units_A.json", {
        "u_a": ("/grades/k_a", "none", synth.EVAL_ID),                # matched by producer_run_key_legacy = store basename
        "u_b": ("/grades/k_b", "none", synth.EVAL_ID),                # run exists, complete by the judge -> not pending
        "u_x": ("/grades/k_missing", "none", synth.EVAL_ID),          # no run row (backfill gap)
        "u_y": ("/grades/k_a", "constitution_lora", synth.EVAL_ID),   # store exists, object discriminator differs -> no run
    })
    with DB.reading(DB.db_path(root)) as conn:
        rep = P.worklist_parity(conn, SEED_ID, P.load_units([units]))
    assert rep.matched == {"u_a": [ra]}
    assert set(rep.unit_without_run) == {"u_x", "u_y"} and "k_missing" in rep.unit_without_run["u_x"]
    assert "constitution_lora" in rep.unit_without_run["u_y"]
    assert rep.unit_not_pending == {"u_b": {rb: "complete judgment by the requested judge covers it (n_items_judged=200)"}}
    assert set(rep.pending_without_unit) == {rc} and "k_c" in rep.pending_without_unit[rc]
    assert rep.identity_mismatch == {} and rep.n_differences == 4
    kinds = [k for k, _, _ in rep.differences()]
    assert kinds == ["unit_without_run", "unit_without_run", "unit_not_pending", "pending_without_unit"]
    # the CLI: exit 1 with the differences by run_id / unit, exit 0 once the units equal the pending set
    assert cli.main(["worklist-parity", "--judge", SEED_ID, "--units", units]) == 1
    out = capsys.readouterr().out
    assert "4 differences" in out and rb in out and rc in out and "u_x" in out
    exact = _units_file(tmp_path / "units_exact.json", {"u_a": ("/grades/k_a", "none", synth.EVAL_ID),
                                                        "u_c": ("/grades/k_c", "none", synth.EVAL_ID)})
    assert cli.main(["worklist-parity", "--judge", SEED_ID, "--units", exact, "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["n_differences"] == 0 and doc["n_matched_units"] == 2
    # two files are one unit set; an unknown judge and a malformed file refuse
    assert cli.main(["worklist-parity", "--judge", SEED_ID, "--units", exact, "--units", units]) == 1
    assert cli.main(["worklist-parity", "--judge", "ghost", "--units", exact]) == 2
    (tmp_path / "bad.json").write_text(json.dumps({"units": {"u": {"grades_dir": "/g"}}}))
    assert cli.main(["worklist-parity", "--judge", SEED_ID, "--units", str(tmp_path / "bad.json")]) == 2
    assert "lacks" in capsys.readouterr().err


def test_load_units_reads_the_d31_shape_and_a_bare_mapping(tmp_path):
    p = tmp_path / "u.json"
    p.write_text(json.dumps({"units": {"h1": {"grades_dir": "/x/grades_full22_ch_gemma", "cell_id": "arditi", "family": "pasttense",
                                                "source": "arditi_strength", "kind": "gated", "trial": 0}}}))
    u = P.load_units([str(p)])["h1"]
    assert (u.store_key, u.object_key, u.eval_id, u.campaign, u.extra["kind"]) == ("grades_full22_ch_gemma", "arditi", "pasttense",
                                                                                     "arditi_strength", "gated")
    p2 = tmp_path / "bare.json"
    p2.write_text(json.dumps({"h2": {"grades_dir": "/g", "cell_id": "none", "family": "xstest"}}))
    assert P.load_units([str(p2)])["h2"].campaign is None


def test_finish_folds_nothing_by_default_and_a_forced_fold_never_fails_a_graded_batch(tmp_path, monkeypatch, capsys):
    """2026-09-30: `finish()` leaves the completion records on the spool -- the harvester folds them within one cycle, and the claims
    (folded at claim time) stay visible meanwhile; every driver batch used to run a FULL harvester cycle under the write lock here,
    holding it against the timer and the other drivers' claims and rewriting the cycle stamp the status page ages. `finish(fold=True)`,
    or SISTEER_REGISTRY_DRIVER_FOLD=force in the driver's environment, folds under the lock first; a lock still held after the wait is
    reported on the returned report and named on stderr, never raised -- `grade`'s closing `finish` can neither mask the driver's own
    exception nor fail a batch whose every record is spooled."""
    root, (r1, r2) = _root_with(tmp_path, [("k1", "generated", None), ("k2", "generated", None)], monkeypatch)
    monkeypatch.delenv(W.FOLD_ENV, raising=False)
    wl = W.RegistryWorklist(root, [SEED_ID], "session-A", driver="tests/stub_driver.py", api_pool="lowprio")
    rows = {r["run_id"]: r for r in wl.pending()}
    folds = []
    real_ingest_now = P.ingest_now
    monkeypatch.setattr(P, "ingest_now", lambda root_, *a, **k: folds.append(root_) or real_ingest_now(root_, *a, **k))
    assert wl.grade([rows[r1]], _stub_grade) == [r1] and folds == []                # graded; finish() folded nothing
    conn = DB.connect(DB.db_path(root), readonly=True)
    st = conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (r1,)).fetchone()[0]
    js = [r[0] for r in conn.execute("SELECT status FROM judgment WHERE run_id = ?", (r1,)).fetchall()]
    conn.close()
    assert st == "judging" and js == ["in_progress"]                                  # the claim is visible; the completion sits on the spool
    rep = wl.finish()
    assert isinstance(rep, W.FinishReport) and not rep.folded and rep.note == W.SPOOLED_NOTE and rep.line() == f"[registry] {W.SPOOLED_NOTE}"
    assert rep.records == rep.refused == rep.blocked == 0 and folds == []
    H.ingest_once(root, workers=0, min_free=0, derive_facts=False)                    # the harvester's fold
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (r1,)).fetchone()[0] == "judged"
    conn.close()
    # a forced fold: the explicit argument, or the environment knob
    assert wl.grade([rows[r2]], _stub_grade) == [r2] and folds == []
    rep2 = wl.finish(fold=True)
    assert rep2.folded and folds == [root] and rep2.ok >= 1 and rep2.line().startswith("[registry] folded ")
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (r2,)).fetchone()[0] == "judged"
    conn.close()
    monkeypatch.setenv(W.FOLD_ENV, "force")
    assert W.fold_requested() and wl.finish().folded and folds == [root, root]
    for v in ("skip", "0", "", "FOLD"):
        monkeypatch.setenv(W.FOLD_ENV, v)
        assert W.fold_requested() is (v == "FOLD")                                     # case-insensitive; anything else is the default
    # the lock held past the wait: reported and named, never raised; grade()'s finally returns the graded ids
    monkeypatch.setenv(W.FOLD_ENV, "force")
    monkeypatch.setattr(P, "ingest_now", lambda root_, *a, **k: (_ for _ in ()).throw(DB.LockHeld("lock held: harvester.lock")))
    rep4 = wl.finish()
    assert not rep4.folded and "write lock is held" in rep4.note and W.SPOOLED_NOTE in rep4.note and rep4.line() == f"[registry] {rep4.note}"
    assert "write lock is held" in capsys.readouterr().err
    assert wl.grade([], _stub_grade) == []
