"""A run whose designated judge moved between the grader's pending read and its claim is RE-SELECTED next tick, never
quarantined.

The registry's `StaleRequiredJudge` fires correctly on a same-id instrument correction in that gap; every grader refuses by name ("--worklist
registry: refusing — claim refused: run <id> now designates judge <current> (instrument <id>), the worklist row carried <old>; re-select" --
petri_run wraps it as a DriverError line, the misalign judge as a SystemExit message), but the daemon's identity classifier knew only its
four other phrases: the exit read `failed` and QUARANTINED the group, and the quarantine releases on nothing a new designation changes. Now
"now designates judge" is an `identity_changed` phrase and the classifier admits the bare `--worklist registry: refusing — ` wrapper too:
the group is left pending, its copy kept, and the next tick selects it under the NEW designated judge and grades it. Driven end to end: the
fake driver lands the correction between its read and its claim. Synthetic registry (test_grade_daemon._root).

The foreign-tag variant's tick 2 is pinned to "record, verify and release on the identity binding": a `REFUSING: [<run>]` line whose
tag is FOREIGN to the group is quarantined and the next tick holds the group back -- but the quarantine record now carries the
identity binding {instrument_id, required_judge_id} the failed dispatch was made against, and `quarantine_filter` releases the run
as `identity_rebound` once the pending view states a different one. The correction this module's fake lands between its read and
its claim IS such a rebind, so the foreign-tag group is released and re-dispatched under the corrected identity on tick 2 and
graded; a sibling test runs the same foreign-tagged refusal with NO correction behind it and pins the quarantine standing tick
after tick (the binding, never the refusal's text or the reason alone, releases)."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth

import test_grade_daemon as TGD
from test_grade_daemon import _events, _load_daemon_module, _mirror_args, _root
from test_grade_daemon_schema_hold import _db, _grader_env, _tick_args

REPO = TGD.REPO
sys.path.insert(0, str(REPO / "scripts" / "registry"))

#: a second registered instrument of the same eval whose designated judge is JUDGE_ID_2
OTHER_INSTRUMENT = synth.instrument_fields(judge_id=synth.JUDGE_ID_2, token_cap_tokens=1024)
OTHER_INSTRUMENT_ID = str(OTHER_INSTRUMENT["instrument_id"])

# the fake driver: right after its pending read, while the switch file stands, it files the same-id instrument correction (consumed: one race),
# then refuses its claim the way the real drivers do -- by name, as the `--worklist registry: refusing — ` message
FAKE_DRIVER_SWITCH = TGD.FAKE_DRIVER.replace("rows = wl.pending()\n", r'''rows = wl.pending()
if os.environ.get("FAKE_DRIVER_SWITCH_FILE") and os.path.exists(os.environ["FAKE_DRIVER_SWITCH_FILE"]):   # g39-fix-2: the correction lands between the read and the claim
    import json as _json
    from sisteer.registry import harvest as _H, records as _R, synth as _synth
    _spec = _json.loads(open(os.environ["FAKE_DRIVER_SWITCH_FILE"]).read()); os.unlink(os.environ["FAKE_DRIVER_SWITCH_FILE"])
    _recs = []
    for _rid, _s in _spec["runs"].items():
        _p = _synth.run_payload(_s["key"], _s["object_key"], _s["object_id"], 600, instrument_id=_spec["instrument_id"], **_s["over"]); _p["run_id"] = _rid
        _recs.append(_synth.rec("run", _p, _synth.t(600)))
    _R.write_records(os.path.join(args.registry_root, "records", "c", "000088-switch.jsonl"), _recs)
    assert _H.ingest_once(args.registry_root, workers=0, min_free=0, derive_facts=False).refused == 0
''', 1).replace("    wl.grade(rows, grade)\n", r'''    try:
        if os.environ.get("FAKE_DRIVER_REFUSE_UNPROMPTED"):   # STALEJUDGEPIN (D2419) sibling: a competing claim refused -- NO correction behind it
            raise W.P.ClaimHeld(str(rows[0]["run_id"]) if rows else "?", "another-grader", 7)
        wl.grade(rows, grade)
    except W.P.ClaimHeld as e:                      # the real drivers' claim refusal, by name, in the wrapping the test names (g39-fix-3: the REAL ones)
        _tmpl = os.environ.get("FAKE_DRIVER_REFUSE_TEMPLATE") or "--worklist registry: refusing — {e}"
        # g39-fix-4: the static graders tag the line with their private run -- the dispatch group's `private_run`, here the mapped local root's
        # name (the synthetic layout: <root>/<private_run>/<attack>/<cell>) -- unless the test names a foreign tag
        _tag = os.environ.get("FAKE_DRIVER_REFUSE_TAG") or os.path.basename((args.store_root_map or ["==."])[0].split("=", 2)[2].rstrip("/"))
        print(_tmpl.replace("{e}", str(e)).replace("{run}", _tag), file=sys.stderr, flush=True)
        sys.exit(int(os.environ.get("FAKE_DRIVER_REFUSE_RC") or "1"))
''', 1)
assert FAKE_DRIVER_SWITCH != TGD.FAKE_DRIVER and "FAKE_DRIVER_SWITCH_FILE" in FAKE_DRIVER_SWITCH and "refusing —" in FAKE_DRIVER_SWITCH
assert "FAKE_DRIVER_REFUSE_UNPROMPTED" in FAKE_DRIVER_SWITCH


def _load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)   # type: ignore[union-attr]
    return mod


def _real_static_refuse_template(rel: str):
    """The exact line the REAL static grader prints on a refused claim (`raise Refuse(f"[{run}] --worklist registry: refusing — {e}")`), as a
    template over the refusal text, and its exit code -- so the fake grader below reproduces the actual wrapper, never a guessed one."""
    mod = _load_module("static_refuse_e2e_" + Path(rel).stem, rel)
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        try:
            raise mod.Refuse("[{run}] --worklist registry: refusing — SYNTHETIC-REFUSAL-TEXT")     # the run tag stays a placeholder: the fake fills the group's
        except SystemExit as e:
            code = e.code
    line = buf.getvalue().strip().splitlines()[-1]
    assert "SYNTHETIC-REFUSAL-TEXT" in line and line.startswith("REFUSING: [{run}] --worklist registry: refusing")
    return line.replace("SYNTHETIC-REFUSAL-TEXT", "{e}"), int(code)


WRAPPERS = {
    "misalign_sysexit": (None, 1, "--worklist registry: refusing"),                                             # misalign_judge: SystemExit(message)
    "petri_driver_error": ("DriverError: --worklist registry: refusing — {e}", 1, "DriverError: --worklist registry: refusing"),   # petri_run: DriverError, uncaught
    "static_sampled_refuse": ("scripts/sdfbase/full22/sampled/grade_static_sampled.py", 2, "REFUSING: [mirror] --worklist registry: refusing"),   # the group's private run (the synthetic layout's) in the tag
    "static_padded_refuse": ("scripts/sdfbase/full22/static_padding/grade_padded_static.py", 2, "REFUSING: [mirror] --worklist registry: refusing"),
    # the same real print tagged with a run OUTSIDE the group: not this dispatch's refusal -> failed and quarantined; the quarantine
    # stands until a VERIFIED identity rebind -- which the correction the fake lands is, so tick 2 releases and re-dispatches
    "static_sampled_refuse_foreign_tag": ("scripts/sdfbase/full22/sampled/grade_static_sampled.py", 2, "REFUSING: [another_private_run] --worklist registry: refusing"),
}


@pytest.mark.parametrize("wrapper", sorted(WRAPPERS))
def test_an_instrument_corrected_between_selection_and_claim_is_re_selected_next_tick_not_quarantined(tmp_path, monkeypatch, wrapper):
    """The two dispatched STATIC graders catch ClaimHeld and raise `Refuse`, whose constructor prints `REFUSING: [<run>] --worklist
    registry: refusing — <reason>` before exiting 2 -- the earlier e2e used a fake SystemExit and missed both. Every wrapper the real graders
    give the refusal is exercised: the misalign judge's bare SystemExit message, petri_run's DriverError line, and the two static graders' REAL
    `Refuse` print (the actual classes imported, their output reproduced verbatim by the fake grader)."""
    template, rc, prefix = WRAPPERS[wrapper]
    if template and template.endswith(".py"):
        template, rc = _real_static_refuse_template(template)
    gd = _load_daemon_module()
    monkeypatch.setattr(TGD, "FAKE_DRIVER", FAKE_DRIVER_SWITCH)
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    if template:
        monkeypatch.setenv("FAKE_DRIVER_REFUSE_TEMPLATE", template)
    monkeypatch.setenv("FAKE_DRIVER_REFUSE_RC", str(rc))
    foreign = wrapper.endswith("_foreign_tag")
    if foreign:
        monkeypatch.setenv("FAKE_DRIVER_REFUSE_TAG", "another_private_run")
    root, rids = _root(tmp_path)
    R.write_records(os.path.join(root, "records", "dims", "000002-other-instrument.jsonl"), [synth.rec("instrument", OTHER_INSTRUMENT, synth.t(41))])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    both = sorted(rids[k] for k in ("job7@none", "job7@steer"))
    over = {"model_id": synth.MODEL_ID, "view_legacy": "synthmodel_4b_u0", "cluster_job_id": "7"}
    spec = {"instrument_id": OTHER_INSTRUMENT_ID,
            "runs": {rids["job7@none"]: {"key": "job7@none", "object_key": "none", "object_id": "none", "over": {**over, "store_uri": "/cluster_a/user/exp/stores/job7_none"}},
                     rids["job7@steer"]: {"key": "job7@steer", "object_key": "lora@abcd", "object_id": synth.OBJECT_SHA, "over": {**over, "store_uri": "/cluster_a/user/exp/stores/job7_steer"}}}}
    switch = tmp_path / "switch.json"; switch.write_text(json.dumps(spec)); monkeypatch.setenv("FAKE_DRIVER_SWITCH_FILE", str(switch))
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    state = tmp_path / "state"; mirror = tmp_path / "mirror"; dirs = [mirror / "stores" / "job7_none", mirror / "stores" / "job7_steer"]
    # tick 1: the grader's claim refuses on the moved designation, in THIS wrapper -> identity_changed: pending, copy kept, nothing quarantined, nothing written
    s1 = gd.tick(args, *rest)
    ev = _events(tmp_path)
    (d,) = [e for e in ev if e["event"] == "dispatched"]
    if foreign:
        # the tag names a run outside the group: a real failure of this dispatch, quarantined (the daemon's standing rule), never
        # transient. The quarantine stands only until a VERIFIED identity rebind -- the ledger records the identity binding
        # {instrument_id, required_judge_id} the failed dispatch was made against, and `quarantine_filter` releases the run as
        # `identity_rebound` once the pending view states a different one. The correction the fake landed between its read and its claim IS that
        # rebind, so tick 2 releases both runs, re-dispatches the group ONCE under the corrected identity and grades it; the sibling test below
        # runs the same refusal with NO correction behind it and stays quarantined (the binding, never the refusal's text, releases)
        assert d["rc"] == rc and d["outcome"] == "failed" and d["terminal_exception"].startswith(prefix), d
        assert s1["failed"] == 2 and s1["identity_changed"] == 0 and s1["dispatched"] == 0
        ledger = json.loads((state / "dispatch_quarantine.json").read_text())             # quarantined on this tick (the summary's `quarantined` counts rows held back on later ticks)
        assert sorted(ledger.get("runs", ledger).keys() if isinstance(ledger, dict) else []) == both, ledger
        assert [e for e in ev if e["event"] == "dispatch_failed"] and not [e for e in ev if e["event"] == "dispatch_identity_changed"]
        was_binding = {"instrument_id": synth.INSTRUMENT_ID, "required_judge_id": synth.JUDGE_ID}            # the identity at dispatch: read BEFORE the correction
        binding = {"instrument_id": OTHER_INSTRUMENT_ID, "required_judge_id": synth.JUDGE_ID_2}             # the corrected identity the view states now
        q1 = ledger["runs"]
        assert all(q1[r_]["identity_binding"] == was_binding for r_ in both), q1                            # the ledger carries the binding at dispatch
        with DB.reading(_db(root)) as conn:
            assert all(conn.execute("SELECT instrument_id FROM run WHERE run_id = ?", (r_,)).fetchone()[0] == OTHER_INSTRUMENT_ID for r_ in both)   # the correction landed
            assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0                          # the refusal wrote nothing
        # tick 2: the binding the ledger recorded is gone from the view -> each run released `identity_rebound` (one event per run: the old and
        # the new binding named, no judgment bound -- a failed dispatch filed none -- released on the binding, not on the sha or the generation),
        # the group re-dispatched exactly ONCE under the corrected identity and graded by the NEW designated judge; the ledger empty
        s2 = gd.tick(args, *rest)
        ev2 = _events(tmp_path)[len(ev):]
        rel = [e for e in ev2 if e["event"] == "quarantine_released"]
        assert sorted((e["run_id"], e["reason"]) for e in rel) == [(r_, "identity_rebound") for r_ in both], rel
        for e in rel:
            assert e["was_binding"] == was_binding and e["binding"] == binding, e
            assert e["judgment_ids"] is None and e["judgment_bindings"] is None, e
            assert e["family"] == "static" and e["quarantined_utc"] == q1[e["run_id"]]["utc"], e
            assert e["was_daemon_sha"] == e["daemon_sha"] == gd.IMPORTED_CODE_SHA, e
        assert [sorted(e["run_ids"]) for e in ev2 if e["event"] == "dispatch"] == [both], [e for e in ev2 if e["event"] == "dispatch"]
        (d2,) = [e for e in ev2 if e["event"] == "dispatched"]
        assert d2["outcome"] == "ok" and d2["rc"] == 0, d2
        assert s2["quarantined"] == 0 and s2["dispatched"] == 2 and s2["failed"] == 0 and s2["identity_changed"] == 0 and s2["problems"] == [], s2
        assert gd.quarantine_load(str(state)) == {} and not [e for e in ev2 if e["event"] in ("dispatch_failed", "dispatch_identity_changed", "quarantine_dropped")], ev2
        with DB.reading(_db(root)) as conn:
            assert {r[0] for r in conn.execute("SELECT judge_id FROM judgment WHERE status = 'complete'")} == {synth.JUDGE_ID_2}
            assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (r_,)).fetchone()[0] == "judged" for r_ in both)
        # tick 3: graded -- nothing pending, nothing re-dispatched, nothing released, no problem
        s3 = gd.tick(args, *rest)
        ev3 = _events(tmp_path)[len(ev) + len(ev2):]
        assert not [e for e in ev3 if e["event"] in ("dispatch", "dispatched", "dispatch_failed", "quarantine_released")], ev3
        assert s3["pending"] == 0 and s3["dispatched"] == 0 and s3["quarantined"] == 0 and s3["problems"] == [], s3
        return
    assert d["rc"] == rc and d["outcome"] == "identity_changed", d
    assert d["terminal_exception"].startswith(prefix) and "now designates judge" in d["terminal_exception"] and synth.JUDGE_ID_2 in d["terminal_exception"]
    (ic,) = [e for e in ev if e["event"] == "dispatch_identity_changed"]
    assert sorted(ic["run_ids"]) == both and ic["ticks"] == 1
    assert s1["identity_changed"] == 2 and s1["failed"] == 0 and s1["quarantined"] == 0 and s1["dispatched"] == 0
    assert not (state / "dispatch_quarantine.json").exists() and all(d_.is_dir() for d_ in dirs) and not switch.exists()
    conn = DB.connect(_db(root), readonly=True)
    assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0                                 # the refusal wrote nothing
    assert all(conn.execute("SELECT instrument_id FROM run WHERE run_id = ?", (r_,)).fetchone()[0] == OTHER_INSTRUMENT_ID for r_ in both)
    conn.close()
    # tick 2: re-selected under the NEW designated judge and graded; the standing refusal cleared by the clean outcome
    s2 = gd.tick(args, *rest)
    ev2 = _events(tmp_path)[len(ev):]
    (d2,) = [e for e in ev2 if e["event"] == "dispatched"]
    assert d2["outcome"] == "ok" and s2["dispatched"] == 2 and s2["identity_changed"] == 0 and s2["quarantined"] == 0 and s2["problems"] == []
    assert sorted((e["run_id"], e["why"]) for e in ev2 if e["event"] == "dispatch_identity_changed_cleared") == [(r_, "the dispatch's outcome is ok") for r_ in both]
    with DB.reading(_db(root)) as conn:
        assert {r[0] for r in conn.execute("SELECT judge_id FROM judgment WHERE status = 'complete'")} == {synth.JUDGE_ID_2}
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (r_,)).fetchone()[0] == "judged" for r_ in both)


def test_the_same_foreign_tag_refusal_without_an_instrument_correction_stays_quarantined_tick_after_tick(tmp_path, monkeypatch):
    """The sibling of the foreign-tag variant above: the SAME real `Refuse` print tagged with a run outside the group
    (`REFUSING: [another_private_run] --worklist registry: refusing — …`, exit 2) -- but NO instrument correction lands between the
    grader's pending read and its claim: the switch file is never written and the fake refuses a competing claim (the registry's
    `pending.ClaimHeld`, raised unprompted). Tick 1 is the pin unchanged -- `failed`, `dispatch_failed`, both runs in the quarantine
    ledger carrying the identity binding at dispatch. Then, the binding the ledger recorded being the one the pending view still
    states, the `identity_rebound` release never fires: ticks 2 and 3 hold both runs back (`quarantined` 2, nothing dispatched, no
    `quarantine_released`, no `dispatch`), the `dispatch_quarantine` problem standing and the ledger unchanged. The release needs
    the VERIFIED rebind (the binding differs), never the refusal's text or the reason alone."""
    template, rc = _real_static_refuse_template("scripts/sdfbase/full22/sampled/grade_static_sampled.py")
    prefix = WRAPPERS["static_sampled_refuse_foreign_tag"][2]
    gd = _load_daemon_module()
    monkeypatch.setattr(TGD, "FAKE_DRIVER", FAKE_DRIVER_SWITCH)
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    monkeypatch.setenv("FAKE_DRIVER_REFUSE_TEMPLATE", template); monkeypatch.setenv("FAKE_DRIVER_REFUSE_RC", str(rc))
    monkeypatch.setenv("FAKE_DRIVER_REFUSE_TAG", "another_private_run")
    monkeypatch.setenv("FAKE_DRIVER_REFUSE_UNPROMPTED", "1")                                 # a competing claim refused: no correction behind it
    monkeypatch.delenv("FAKE_DRIVER_SWITCH_FILE", raising=False)                              # no switch file: the correction never lands
    root, rids = _root(tmp_path)
    R.write_records(os.path.join(root, "records", "dims", "000002-other-instrument.jsonl"), [synth.rec("instrument", OTHER_INSTRUMENT, synth.t(41))])
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    both = sorted(rids[k] for k in ("job7@none", "job7@steer"))
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    state = tmp_path / "state"
    binding = {"instrument_id": synth.INSTRUMENT_ID, "required_judge_id": synth.JUDGE_ID}        # the identity at dispatch -- and still the view's
    # tick 1: the pin -- failed, dispatch_failed, both quarantined with the binding at dispatch; the registry's identity never moved
    s1 = gd.tick(args, *rest)
    ev = _events(tmp_path)
    (d,) = [e for e in ev if e["event"] == "dispatched"]
    assert d["rc"] == rc and d["outcome"] == "failed" and d["terminal_exception"].startswith(prefix), d
    assert "is claimed by session" in d["terminal_exception"] and "now designates judge" not in d["terminal_exception"], d
    assert s1["failed"] == 2 and s1["identity_changed"] == 0 and s1["dispatched"] == 0 and s1["quarantined"] == 0, s1
    q1 = gd.quarantine_load(str(state))
    assert sorted(q1) == both and all(q1[r_]["identity_binding"] == binding for r_ in both), q1
    assert [e for e in ev if e["event"] == "dispatch_failed"] and not [e for e in ev if e["event"] in ("dispatch_identity_changed", "quarantine_released")]
    with DB.reading(_db(root)) as conn:
        assert all(conn.execute("SELECT instrument_id FROM run WHERE run_id = ?", (r_,)).fetchone()[0] == synth.INSTRUMENT_ID for r_ in both)
        assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0                   # the refusal wrote nothing
    # ticks 2 and 3: the recorded binding IS the current one -- no identity_rebound, nothing dispatched, the quarantine problem standing
    seen = len(ev)
    for tick_no in (2, 3):
        s = gd.tick(args, *rest)
        evn = _events(tmp_path)[seen:]; seen += len(evn)
        assert not [e for e in evn if e["event"] in ("quarantine_released", "quarantine_dropped", "dispatch", "dispatched", "dispatch_failed")], (tick_no, evn)
        (p,) = [e for e in evn if e["event"] == "problem" and e["kind"] == "dispatch_quarantine"]
        assert p["n"] == 2 and sorted(p["run_ids"]) == both, p
        assert s["quarantined"] == 2 and s["dispatched"] == 0 and s["failed"] == 0 and s["pending"] == 2, (tick_no, s)
        assert s["problems"] == ["dispatch_quarantine: 2 pending run(s) held after a failed dispatch"], (tick_no, s)
        assert gd.quarantine_load(str(state)) == q1                                                # the ledger untouched: nothing released, nothing dropped
    with DB.reading(_db(root)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM judgment").fetchone()[0] == 0
