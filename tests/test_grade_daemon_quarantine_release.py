"""The grading daemon's explicit quarantine release file (`<state-dir>/dispatch_quarantine.release`: one run id per line, `#` lines ignored,
`*` = every held run) is read, CONSUMED and VALIDATED on every tick, an EMPTY ledger included.

Before the pin `quarantine_filter` returned on an empty ledger BEFORE `_release_requests` ran, so a release written while nothing was held
survived every empty-ledger tick and released the NEXT failure of a named run on sight (an unintended repeat dispatch of a run whose new
failure nobody had looked at); and nothing checked the file's ids against the ledger. Now: an id not presently quarantined is rejected with
one `quarantine_release_rejected` event and is never stored or applied later; a file found with an empty ledger is consumed and logged once
(`quarantine_release_ignored_empty`; `*` is a no-op there); a dry run still neither consumes nor saves; `*` on a held ledger and the
daemon_sha / store_generation / identity_rebound release paths are unchanged. In-process over test_grade_daemon's synthetic registry (two
pending runs of the synthetic model), the daemon module loaded from the tree; the ledger entries are written by the daemon's own
`quarantine_add`. Synthetic ids only."""
from __future__ import annotations

import json
import os
from pathlib import Path

from sisteer.registry import db as DB
from sisteer.registry import synth

from test_grade_daemon import _events, _load_daemon_module, _root

UNKNOWN_RUN = "synthcampaign:never_registered:0000000000ab:1111111111111111"     # a run id the ledger never held (invented)
UNKNOWN_RUN_2 = "synthcampaign:never_registered:0000000000cd:2222222222222222"


def _world(tmp_path):
    """The daemon module, the synthetic registry, its two pending rows of the synthetic model (by run id), the state dir and the ledger path."""
    gd = _load_daemon_module()
    root, rids = _root(tmp_path)
    st = tmp_path / "state"
    st.mkdir(exist_ok=True)
    rows = gd.pending_rows(root, [synth.MODEL_ID], synth.CAMPAIGN, None, {synth.EVAL_ID: "static"})
    by_id = {str(r["run_id"]): r for r in rows}
    assert sorted(by_id) == sorted(rids[k] for k in ("job7@none", "job7@steer")), (sorted(by_id), rids)
    return gd, root, rows, by_id, st, st / "dispatch_quarantine.json"


def _quarantine(gd, root, st, rows, *, log_name="dispatch.log"):
    """The daemon's OWN producer of ledger entries: a failed dispatch of `rows` (rc 1), the way the tick records one."""
    log = st / log_name
    log.write_text("fake grader: failing on purpose\n")
    gd.quarantine_add(str(st), root, group=list(rows), family="static", model_key="synthmodel", campaign=synth.CAMPAIGN, rc=1,
                      cmd=["python", "fake_driver.py"], log_path=str(log))


def _ledger(path: Path) -> dict:
    return json.loads(path.read_text())["runs"] if path.exists() else {}


def _since(tmp_path, n):
    return _events(tmp_path)[n:]


def _release_path(st: Path) -> Path:
    return st / "dispatch_quarantine.release"


# ------------------------------------------------------------------------------------------------------------ (1) empty ledger + a release file


def test_a_release_file_found_with_an_empty_ledger_is_consumed_rejected_by_name_and_never_releases_a_later_quarantine(tmp_path):
    """The audit's scenario: the file names a run while nothing is quarantined. The tick consumes it, logs the id as rejected (not quarantined)
    and the empty-ledger ignore, releases nothing, writes no ledger; when that run is quarantined AFTERWARDS the next tick holds it -- the
    stale authorisation did not survive."""
    gd, root, rows, by_id, st, ledger = _world(tmp_path)
    a, b = sorted(by_id)
    _release_path(st).write_text(f"# written while the ledger was empty\n{a}\n")
    assert not ledger.exists()
    n0 = len(_events(tmp_path))
    eligible, held = gd.quarantine_filter(str(st), root, rows, dry_run=False)
    assert [r["run_id"] for r in eligible] == [r["run_id"] for r in rows] and held == []
    assert not _release_path(st).exists(), "the file is consumed on the empty-ledger tick"
    assert not ledger.exists(), "nothing released, nothing dropped: no ledger is written"
    ev = _since(tmp_path, n0)
    rej = [e for e in ev if e["event"] == "quarantine_release_rejected"]
    assert [(e["run_id"], e["reason"], e["dry_run"], e["n_quarantined"]) for e in rej] == [(a, "not quarantined", False, 0)]
    assert "never applied" in rej[0]["note"] and rej[0]["ledger"] == str(ledger)
    (ign,) = [e for e in ev if e["event"] == "quarantine_release_ignored_empty"]
    assert ign["requested"] == [a] and ign["n"] == 1 and ign["dry_run"] is False and "empty ledger" in ign["note"]
    assert not [e for e in ev if e["event"] in ("quarantine_released", "quarantine_dropped", "problem")]
    # the run fails later and is quarantined: the next tick HOLDS it (the earlier file bought it nothing)
    _quarantine(gd, root, st, [by_id[a]])
    assert sorted(_ledger(ledger)) == [a]
    n1 = len(_events(tmp_path))
    eligible2, held2 = gd.quarantine_filter(str(st), root, rows, dry_run=False)
    assert [r["run_id"] for r in eligible2] == [b] and [r["run_id"] for r in held2] == [a]
    ev2 = _since(tmp_path, n1)
    assert not [e for e in ev2 if e["event"] in ("quarantine_released", "quarantine_release_rejected", "quarantine_release_ignored_empty")]
    (prob,) = [e for e in ev2 if e["event"] == "problem"]
    assert prob["kind"] == "dispatch_quarantine" and prob["run_ids"] == [a]
    assert sorted(_ledger(ledger)) == [a]


def test_a_release_file_on_an_empty_ledger_with_no_state_dir_is_left_alone(tmp_path):
    """Without a state dir there is no ledger and no release file to read: the filter is the identity (unchanged)."""
    gd, root, rows, by_id, st, ledger = _world(tmp_path)
    eligible, held = gd.quarantine_filter(None, root, rows, dry_run=False)
    assert [r["run_id"] for r in eligible] == [r["run_id"] for r in rows] and held == []


# ------------------------------------------------------------------------------------------------------------ (2) a held id beside an unknown id


def test_a_release_naming_one_quarantined_run_and_one_unknown_releases_the_held_one_rejects_the_other_and_consumes_the_file(tmp_path):
    gd, root, rows, by_id, st, ledger = _world(tmp_path)
    a, b = sorted(by_id)
    _quarantine(gd, root, st, rows)
    assert sorted(_ledger(ledger)) == [a, b]
    _release_path(st).write_text(f"# release a, and a run that was never held\n\n{a}\n   {UNKNOWN_RUN}   \n")
    n0 = len(_events(tmp_path))
    eligible, held = gd.quarantine_filter(str(st), root, rows, dry_run=False)
    assert [r["run_id"] for r in eligible] == [a] and [r["run_id"] for r in held] == [b]
    assert not _release_path(st).exists() and sorted(_ledger(ledger)) == [b]
    ev = _since(tmp_path, n0)
    assert [(e["run_id"], e["reason"]) for e in ev if e["event"] == "quarantine_released"] == [(a, "release_file")]
    rej = [e for e in ev if e["event"] == "quarantine_release_rejected"]
    assert [(e["run_id"], e["reason"], e["n_quarantined"]) for e in rej] == [(UNKNOWN_RUN, "not quarantined", 2)]
    assert not [e for e in ev if e["event"] == "quarantine_release_ignored_empty"], "the ledger was not empty"
    (prob,) = [e for e in ev if e["event"] == "problem"]
    assert prob["run_ids"] == [b]
    # the rejected id is not remembered: an entry for it written later (its sha and generation as the daemon would record them for a run the
    # registry has no row for) is NOT released on the next tick -- it is not pending, so the filter DROPS it as settled by another route (the
    # ordinary path, not a release)
    doc = json.loads(ledger.read_text())
    doc["runs"][UNKNOWN_RUN] = dict(doc["runs"][b], run_id=UNKNOWN_RUN, store_generation=gd.store_generation(None, UNKNOWN_RUN, None))
    ledger.write_text(json.dumps(doc))
    n1 = len(_events(tmp_path))
    gd.quarantine_filter(str(st), root, rows, dry_run=False)
    assert not [e for e in _since(tmp_path, n1) if e["event"] == "quarantine_released"]
    assert [e["run_id"] for e in _since(tmp_path, n1) if e["event"] == "quarantine_dropped"] == [UNKNOWN_RUN]


# ------------------------------------------------------------------------------------------------------------ (3) a dry run reads, consumes nothing, saves nothing


def test_a_dry_run_reads_the_release_file_but_neither_consumes_nor_saves_on_a_held_ledger_and_on_an_empty_one(tmp_path):
    gd, root, rows, by_id, st, ledger = _world(tmp_path)
    a, b = sorted(by_id)
    _quarantine(gd, root, st, [by_id[a]])
    _release_path(st).write_text(f"{a}\n{UNKNOWN_RUN}\n")
    n0 = len(_events(tmp_path))
    eligible, held = gd.quarantine_filter(str(st), root, rows, dry_run=True)
    assert sorted(r["run_id"] for r in eligible) == [a, b] and held == []              # released in memory, as before
    assert _release_path(st).exists() and sorted(_ledger(ledger)) == [a], "a dry run consumes nothing and saves nothing"
    ev = _since(tmp_path, n0)
    assert [(e["run_id"], e["reason"]) for e in ev if e["event"] == "quarantine_released"] == [(a, "release_file")]
    assert [(e["run_id"], e["dry_run"]) for e in ev if e["event"] == "quarantine_release_rejected"] == [(UNKNOWN_RUN, True)]
    # the real tick after it: consumed, released, saved
    n1 = len(_events(tmp_path))
    eligible2, held2 = gd.quarantine_filter(str(st), root, rows, dry_run=False)
    assert sorted(r["run_id"] for r in eligible2) == [a, b] and held2 == [] and not _release_path(st).exists() and _ledger(ledger) == {}
    assert [(e["run_id"], e["reason"]) for e in _since(tmp_path, n1) if e["event"] == "quarantine_released"] == [(a, "release_file")]
    # an EMPTY ledger under a dry run: the file stays (not consumed), the ignore is logged with dry_run true; the real tick then consumes it
    _release_path(st).write_text(f"{b}\n")
    n2 = len(_events(tmp_path))
    gd.quarantine_filter(str(st), root, rows, dry_run=True)
    assert _release_path(st).exists() and not ledger.exists() or _ledger(ledger) == {}
    (ign,) = [e for e in _since(tmp_path, n2) if e["event"] == "quarantine_release_ignored_empty"]
    assert ign["requested"] == [b] and ign["dry_run"] is True
    n3 = len(_events(tmp_path))
    gd.quarantine_filter(str(st), root, rows, dry_run=False)
    assert not _release_path(st).exists()
    (ign2,) = [e for e in _since(tmp_path, n3) if e["event"] == "quarantine_release_ignored_empty"]
    assert ign2["requested"] == [b] and ign2["dry_run"] is False


# ------------------------------------------------------------------------------------------------------------ (4) `*`


def test_star_releases_every_held_run_and_is_a_logged_no_op_on_an_empty_ledger(tmp_path):
    gd, root, rows, by_id, st, ledger = _world(tmp_path)
    a, b = sorted(by_id)
    _quarantine(gd, root, st, rows)
    _release_path(st).write_text("*\n")
    n0 = len(_events(tmp_path))
    eligible, held = gd.quarantine_filter(str(st), root, rows, dry_run=False)
    assert sorted(r["run_id"] for r in eligible) == [a, b] and held == [] and _ledger(ledger) == {} and not _release_path(st).exists()
    ev = _since(tmp_path, n0)
    assert sorted((e["run_id"], e["reason"]) for e in ev if e["event"] == "quarantine_released") == [(a, "release_file"), (b, "release_file")]
    assert not [e for e in ev if e["event"] in ("quarantine_release_rejected", "quarantine_release_ignored_empty")]
    # `*` beside an unknown id on a held ledger: `*` releases what is held, the unknown id is still rejected by name
    _quarantine(gd, root, st, [by_id[a]])
    _release_path(st).write_text(f"*\n{UNKNOWN_RUN_2}\n")
    n1 = len(_events(tmp_path))
    gd.quarantine_filter(str(st), root, rows, dry_run=False)
    ev1 = _since(tmp_path, n1)
    assert [(e["run_id"], e["reason"]) for e in ev1 if e["event"] == "quarantine_released"] == [(a, "release_file")]
    assert [e["run_id"] for e in ev1 if e["event"] == "quarantine_release_rejected"] == [UNKNOWN_RUN_2] and _ledger(ledger) == {}
    # `*` on an EMPTY ledger: consumed, one ignore event (requested ["*"]), no rejection (`*` is not an id), nothing released
    _release_path(st).write_text("*\n")
    n2 = len(_events(tmp_path))
    eligible2, held2 = gd.quarantine_filter(str(st), root, rows, dry_run=False)
    assert sorted(r["run_id"] for r in eligible2) == [a, b] and held2 == [] and not _release_path(st).exists()
    ev2 = _since(tmp_path, n2)
    (ign,) = [e for e in ev2 if e["event"] == "quarantine_release_ignored_empty"]
    assert ign["requested"] == ["*"] and ign["n"] == 1
    assert not [e for e in ev2 if e["event"] in ("quarantine_released", "quarantine_release_rejected")]
    # ...and a quarantine AFTER that `*` is held: the star did not persist
    _quarantine(gd, root, st, [by_id[b]])
    n3 = len(_events(tmp_path))
    eligible3, held3 = gd.quarantine_filter(str(st), root, rows, dry_run=False)
    assert [r["run_id"] for r in eligible3] == [a] and [r["run_id"] for r in held3] == [b]
    assert not [e for e in _since(tmp_path, n3) if e["event"] == "quarantine_released"]


# ------------------------------------------------------------------------------------------------------------ (5) the other release paths are untouched


def test_the_daemon_sha_store_generation_identity_rebound_and_dropped_paths_are_unchanged(tmp_path):
    """One assertion per path, with NO release file present: a stale recorded daemon sha releases (daemon_sha); a stale recorded store
    generation releases (store_generation); a recorded identity binding the pending view no longer states releases (identity_rebound); an
    entry no longer pending is dropped. The held entry whose cause did not change stays held."""
    gd, root, rows, by_id, st, ledger = _world(tmp_path)
    a, b = sorted(by_id)
    _quarantine(gd, root, st, rows)
    doc = json.loads(ledger.read_text())
    doc["runs"][a]["daemon_sha"] = "0" * 40
    ledger.write_text(json.dumps(doc))
    n0 = len(_events(tmp_path))
    eligible, held = gd.quarantine_filter(str(st), root, rows, dry_run=False)
    assert [r["run_id"] for r in eligible] == [a] and [r["run_id"] for r in held] == [b]
    rel = [e for e in _since(tmp_path, n0) if e["event"] == "quarantine_released"]
    assert [(e["run_id"], e["reason"], e["was_daemon_sha"]) for e in rel] == [(a, "daemon_sha", "0" * 40)] and rel[0]["daemon_sha"] == gd.IMPORTED_CODE_SHA
    assert not [e for e in _since(tmp_path, n0) if e["event"] in ("quarantine_release_rejected", "quarantine_release_ignored_empty")]
    # store_generation
    doc = json.loads(ledger.read_text()); doc["runs"][b]["store_generation"] = "stale|0||"; ledger.write_text(json.dumps(doc))
    n1 = len(_events(tmp_path))
    eligible, held = gd.quarantine_filter(str(st), root, rows, dry_run=False)
    assert sorted(r["run_id"] for r in eligible) == [a, b] and held == [] and _ledger(ledger) == {}
    assert [(e["run_id"], e["reason"]) for e in _since(tmp_path, n1) if e["event"] == "quarantine_released"] == [(b, "store_generation")]
    # identity_rebound: the recorded binding names an instrument the pending view no longer states for the run
    _quarantine(gd, root, st, [by_id[a]])
    doc = json.loads(ledger.read_text())
    assert doc["runs"][a]["identity_binding"] == gd.identity_binding_of(by_id[a])
    doc["runs"][a]["identity_binding"] = dict(doc["runs"][a]["identity_binding"], instrument_id="synthetic-instrument-since-corrected")
    ledger.write_text(json.dumps(doc))
    n2 = len(_events(tmp_path))
    eligible, held = gd.quarantine_filter(str(st), root, rows, dry_run=False)
    assert sorted(r["run_id"] for r in eligible) == [a, b] and held == []
    (rel,) = [e for e in _since(tmp_path, n2) if e["event"] == "quarantine_released"]
    assert rel["run_id"] == a and rel["reason"] == "identity_rebound" and rel["was_binding"]["instrument_id"] == "synthetic-instrument-since-corrected"
    assert rel["binding"] == gd.identity_binding_of(by_id[a])
    # dropped: an entry for a run that is not pending (a settled run; its generation as recorded for a row-less run) leaves the ledger as `quarantine_dropped`
    _quarantine(gd, root, st, [by_id[b]])
    doc = json.loads(ledger.read_text())
    doc["runs"][UNKNOWN_RUN] = dict(doc["runs"][b], run_id=UNKNOWN_RUN, store_generation=gd.store_generation(None, UNKNOWN_RUN, None))
    ledger.write_text(json.dumps(doc))
    n3 = len(_events(tmp_path))
    eligible, held = gd.quarantine_filter(str(st), root, rows, dry_run=False)
    assert [r["run_id"] for r in eligible] == [a] and [r["run_id"] for r in held] == [b] and sorted(_ledger(ledger)) == [b]
    assert [e["run_id"] for e in _since(tmp_path, n3) if e["event"] == "quarantine_dropped"] == [UNKNOWN_RUN]
    assert not [e for e in _since(tmp_path, n3) if e["event"] == "quarantine_released"]
    # a release file naming a run whose sha is ALSO stale: release_file wins the reason, as before (the if-chain's order)
    doc = json.loads(ledger.read_text()); doc["runs"][b]["daemon_sha"] = "0" * 40; ledger.write_text(json.dumps(doc))
    _release_path(st).write_text(f"{b}\n")
    n4 = len(_events(tmp_path))
    gd.quarantine_filter(str(st), root, rows, dry_run=False)
    assert [(e["run_id"], e["reason"]) for e in _since(tmp_path, n4) if e["event"] == "quarantine_released"] == [(b, "release_file")]
    assert _ledger(ledger) == {} and not _release_path(st).exists()


def test_the_registry_being_unreadable_still_honours_the_file_for_held_ids_and_still_rejects_unknown_ones(tmp_path, monkeypatch):
    """The deferred branch (no registry answer): the release file and the daemon sha still count -- and the validation of the file's ids
    happens BEFORE the registry is consulted, so an unknown id is rejected there too."""
    gd, root, rows, by_id, st, ledger = _world(tmp_path)
    a, b = sorted(by_id)
    _quarantine(gd, root, st, rows)
    _release_path(st).write_text(f"{a}\n{UNKNOWN_RUN}\n")

    class _Boom:
        def __enter__(self):
            raise OSError("synthetic: registry unreadable")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(gd.DB, "reading", lambda *a, **k: _Boom())
    n0 = len(_events(tmp_path))
    eligible, held = gd.quarantine_filter(str(st), root, rows, dry_run=False)
    assert [r["run_id"] for r in eligible] == [a] and [r["run_id"] for r in held] == [b] and not _release_path(st).exists()
    ev = _since(tmp_path, n0)
    assert [e["event"] for e in ev if e["event"] == "quarantine_check_deferred"] == ["quarantine_check_deferred"]
    assert [(e["run_id"], e["reason"]) for e in ev if e["event"] == "quarantine_released"] == [(a, "release_file")]
    assert [e["run_id"] for e in ev if e["event"] == "quarantine_release_rejected"] == [UNKNOWN_RUN]
    assert sorted(_ledger(ledger)) == [b]
