"""The Petri grader's IDENTITY-CHANGE refusals are transient, never a
quarantine.

petri_run's borrow / filing rechecks raise `DriverError` when the registry moved under the grader between the binding and the filing ("anchor
identity changed between the binding and ...", "in-view run identity changed between the binding and the filing ...", "selected run dropped
from the judged view between the worklist read and the borrow ..."). The daemon classified that exit as `failed` and QUARANTINED the group --
a correct, transient answer parked until a code sha change. Now `dispatch_failure_kind` reads the `identity_changed` kind off the TERMINAL
line (the grader's DriverError line, module-qualified or bare, optionally wrapping the refusal as `--worklist registry: refusing — ...`,
carrying one of the refusals' stable phrases; the same phrase buried mid-tail under another terminal line stays `failed`) and handles it like
`not_folded`: the runs stay pending and are re-selected next tick, the copy is kept, nothing is quarantined, nothing counts toward the
retryable cap, one `dispatch_identity_changed` names the group's run ids and the refusal text, and a group refusing on consecutive ticks is a
`problem` every tick it stands; a dispatch with another outcome clears the standing refusal (`dispatch_identity_changed_cleared`). Synthetic
registry (test_grade_daemon._root), the fake driver, no live registry."""
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

import test_grade_daemon as TGD                                                        # the daemon fixtures: the synthetic registry, argv/env, fake driver, events
from test_grade_daemon import _events, _load_daemon_module, _mirror_args, _root
from test_grade_daemon_schema_hold import _db, _grader_env, _tick_args                 # the in-process tick harness the not_folded test uses

REPO = TGD.REPO
sys.path.insert(0, str(REPO / "scripts" / "registry"))

#: the refusal texts petri_run raises (recheck_borrowed_anchors_current / the in-view filing recheck / the fix-7 worklist recheck), as the
#: dispatch log's terminal line shows them: the grader runs as __main__, so Python prints the bare exception name; a module-qualified
#: name (another entry point) and the `--worklist registry: refusing — ...` wrapper are the other shapes the classifier admits
ANCHOR_LINE = ("DriverError: run 'synth-run-1' recheck before the filing: anchor identity changed between the binding and the filing: "
               "self_preservation: synthcampaign:petri_self_preservation:aaaaaaaaaaaa:1111111111111111: is_current: 1 -> 0 (why: superseded); "
               "refusing -- a borrowed screen never files on a base whose run, currency or registered completion the registry moved (never a silent stale borrow)")
INVIEW_LINE = ("petri_run.DriverError: --worklist registry: refusing — in-view run identity changed between the binding and the filing: "
               "defer_to_users: synthcampaign:petri_defer_to_users:bbbbbbbbbbbb:2222222222222222: lifecycle_state: 'generated' -> 'superseded'")
DROPPED_LINE = ("DriverError: --worklist registry: refusing — selected run dropped from the judged view between the worklist read and the borrow: "
                "synthcampaign:petri_pooled:cccccccccccc:3333333333333333")
# The source-cell refusals of the same class (petri_run._live_results_sha_or_refuse / the generation check / the snapshot walk)
SOURCE_LINE = ("DriverError: run 'synth-run-1' borrow: source changed during binding/snapshot (generation): 0123456789ab -> ba9876543210 at "
               "/synthetic/out/cells/none/results.json; the source cell was rewritten while it was being bound; refusing -- a retry binds the new "
               "generation once the registry makes it current")
SNAPSHOT_LINE = ("DriverError: run 'synth-run-1' borrow: the source cell dir /synthetic/out/cells/none changed while it was being snapshotted "
                 "(results.json); refusing -- retry")

# the fake driver whose run ends on the anchor-identity refusal while the flag file stands (NOT consumed: the refusal repeats every tick until the
# flag goes, as a registry that moved keeps refusing until the next tick re-selects against its new state)
FAKE_DRIVER_IDENTITY_CHANGED = TGD.FAKE_DRIVER.replace('if os.environ.get("FAKE_DRIVER_ENV_RECORD"):', r"""if os.environ.get("FAKE_DRIVER_IDENTITY_FILE") and os.path.exists(os.environ["FAKE_DRIVER_IDENTITY_FILE"]):   # g39-d: the registry moved under the grader
    print("Traceback (most recent call last):\n  File \"petri_run.py\", line 3552, in recheck_borrowed_anchors_current", file=sys.stderr)
    print(""" + repr(ANCHOR_LINE) + r""", file=sys.stderr); sys.exit(1)
if os.environ.get("FAKE_DRIVER_ENV_RECORD"):""", 1)
assert FAKE_DRIVER_IDENTITY_CHANGED != TGD.FAKE_DRIVER


def test_the_classifier_reads_an_identity_change_refusal_off_the_terminal_line_only():
    gd = _load_daemon_module()
    tb = "Traceback (most recent call last):\n  File petri_run.py\n"
    assert gd.dispatch_failure_kind(1, tb + ANCHOR_LINE + "\n") == "identity_changed"
    assert gd.dispatch_failure_kind(1, tb + INVIEW_LINE + "\n") == "identity_changed"                    # module-qualified, the refusing wrapper
    assert gd.dispatch_failure_kind(1, tb + DROPPED_LINE + "\n") == "identity_changed"                   # fix-7's refusal
    assert gd.identity_changed_terminal(ANCHOR_LINE) and gd.identity_changed_terminal(INVIEW_LINE) and gd.identity_changed_terminal(DROPPED_LINE)
    assert gd.dispatch_failure_kind(1, tb + SOURCE_LINE + "\n") == "identity_changed"                    # The bound source re-scored
    assert gd.dispatch_failure_kind(1, tb + SNAPSHOT_LINE + "\n") == "identity_changed"                  # The source dir changed under the snapshot
    assert gd.identity_changed_terminal(SOURCE_LINE) and gd.identity_changed_terminal(SNAPSHOT_LINE)
    assert gd.dispatch_failure_kind(1, "the source changed during binding/snapshot, said a note\nKeyError: 'x'\n") == "failed"   # mid-tail: still failed
    assert len(gd.IDENTITY_CHANGED_PHRASES) == 8          # the 8 phrases include the stale-judge one, claim refused (precheck), ClaimPrecheckRefused and StaleRequiredJudge
    # another DriverError is a real failure; the identity text buried mid-tail under another terminal line is a real failure; rc 0 is ok
    assert gd.dispatch_failure_kind(1, tb + "DriverError: --worklist registry: no pending petri run under /synthetic/out; nothing to analyse\n") == "failed"
    assert gd.dispatch_failure_kind(1, ANCHOR_LINE + "\n[petri] releasing claims\nKeyError: 'x'\n") == "failed"
    assert gd.dispatch_failure_kind(1, "a note: anchor identity changed between the binding and the filing, in passing\nKeyError: 'x'\n") == "failed"
    assert gd.dispatch_failure_kind(1, "sisteer.registry.pending.ClaimNotFolded: run x is not folded yet\n") == "not_folded"   # the other kinds stand
    assert gd.dispatch_failure_kind(0, ANCHOR_LINE) == "ok"
    assert not gd.identity_changed_terminal("RuntimeError: anchor identity changed between the binding and the filing")   # not the grader's refusal class
    assert gd.IDENTITY_CHANGED_PROBLEM_TICKS == 2


def test_an_identity_change_refusal_is_held_transient_never_quarantined_becomes_a_problem_when_it_persists_and_clears_on_another_outcome(tmp_path, monkeypatch):
    gd = _load_daemon_module()
    monkeypatch.setattr(TGD, "FAKE_DRIVER", FAKE_DRIVER_IDENTITY_CHANGED)
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    flag = tmp_path / "identity_changed"; flag.write_text(""); monkeypatch.setenv("FAKE_DRIVER_IDENTITY_FILE", str(flag))
    root, rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    both = sorted(rids[k] for k in ("job7@none", "job7@steer"))
    mirror = tmp_path / "mirror"; dirs = [mirror / "stores" / "job7_none", mirror / "stores" / "job7_steer"]
    state = tmp_path / "state"
    # tick 1: the grader refuses on the registry having moved -> transient held: pending, copy kept, nothing quarantined, nothing counted, no problem yet
    s1 = gd.tick(args, *rest)
    ev = _events(tmp_path)
    (d,) = [e for e in ev if e["event"] == "dispatched"]
    assert d["rc"] == 1 and d["outcome"] == "identity_changed" and d["terminal_exception"] == ANCHOR_LINE, d
    (ic,) = [e for e in ev if e["event"] == "dispatch_identity_changed"]
    assert sorted(ic["run_ids"]) == both and ic["family"] == "static" and ic["model_key"] == "synthmodel_4b_s1" and ic["message"] == ANCHOR_LINE
    assert ic["ticks"] == 1 and ic["since_utc"] and os.path.isfile(ic["dispatch_log"]) and "re-selected" in ic["note"]
    assert s1["identity_changed"] == 2 and s1["failed"] == 0 and s1["retryable"] == 0 and s1["quarantined"] == 0 and s1["dispatched"] == 0 and s1["problems"] == [], s1
    assert not {"dispatch_failed", "dispatch_retryable", "problem", "dispatch_schema_held", "dispatch_not_folded", "mirror_retired_incomplete"} & {e["event"] for e in ev}
    assert not (state / "dispatch_quarantine.json").exists() and all(d_.is_dir() for d_ in dirs)
    assert not (state / "dispatch_retries.json").exists() or json.loads((state / "dispatch_retries.json").read_text())["runs"] == {}
    with DB.reading(_db(root)) as conn:
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (r_,)).fetchone()[0] == "generated" for r_ in both)   # still pending
    # tick 2: the same refusal -> re-selected and dispatched again (never quarantined); the standing refusal is now a problem
    s2 = gd.tick(args, *rest)
    ev2 = _events(tmp_path)[len(ev):]
    assert [e for e in ev2 if e["event"] == "dispatched"][0]["outcome"] == "identity_changed" and [e for e in ev2 if e["event"] == "dispatch_identity_changed"][0]["ticks"] == 2
    (pr,) = [e for e in ev2 if e["event"] == "problem"]
    assert pr["kind"] == "dispatch_identity_changed" and sorted(pr["run_ids"]) == both and pr["ticks"] == 2 and pr["since_utc"] == ic["since_utc"] and pr["message"] == ANCHOR_LINE
    assert s2["identity_changed"] == 2 and s2["quarantined"] == 0 and s2["failed"] == 0
    assert s2["problems"] == [f"dispatch_identity_changed: 2 run(s) refused on identity for 2 tick(s): {ANCHOR_LINE}"], s2["problems"]
    assert not (state / "dispatch_quarantine.json").exists() and all(d_.is_dir() for d_ in dirs)
    # tick 3: the registry settled (the flag goes): the same group grades clean -> the standing refusal is cleared by the other outcome, no problem
    flag.unlink()
    s3 = gd.tick(args, *rest)
    ev3 = _events(tmp_path)[len(ev) + len(ev2):]
    assert [e for e in ev3 if e["event"] == "dispatched"][0]["outcome"] == "ok"
    assert sorted((e["run_id"], e["why"]) for e in ev3 if e["event"] == "dispatch_identity_changed_cleared") == [(r_, "the dispatch's outcome is ok") for r_ in both]
    assert s3["identity_changed"] == 0 and s3["dispatched"] == 2 and s3["quarantined"] == 0 and s3["problems"] == []
    assert not [e for e in ev3 if e["event"] in ("problem", "dispatch_identity_changed")]
    with DB.reading(_db(root)) as conn:
        assert all(conn.execute("SELECT lifecycle_state FROM run WHERE run_id = ?", (r_,)).fetchone()[0] == "judged" for r_ in both)


# --------------------------------------------------------------------------------------------
# a persisting, unchanged identity refusal HOLDS the group after IDENTITY_HOLD_AFTER
# ---------------------------------------------------------------------------------------------


def _hold_world(tmp_path, monkeypatch):
    gd = _load_daemon_module()
    monkeypatch.setattr(TGD, "FAKE_DRIVER", FAKE_DRIVER_IDENTITY_CHANGED)
    monkeypatch.setenv("FAKE_RSYNC_LOG", str(tmp_path / "rsync_calls.log"))
    flag = tmp_path / "identity_changed"; flag.write_text(""); monkeypatch.setenv("FAKE_DRIVER_IDENTITY_FILE", str(flag))
    root, rids = _root(tmp_path)
    args, rest, env = _tick_args(gd, root, tmp_path, *_mirror_args(tmp_path)); _grader_env(monkeypatch, env)
    both = sorted(rids[k] for k in ("job7@none", "job7@steer"))
    return gd, root, args, rest, both, flag


def _n_dispatched(events):
    return len([e for e in events if e["event"] == "dispatched"])


def test_three_unchanged_identity_refusals_hold_the_group_and_the_operators_release_file_lifts_it(tmp_path, monkeypatch):
    gd, root, args, rest, both, flag = _hold_world(tmp_path, monkeypatch)
    assert gd.IDENTITY_HOLD_AFTER == 3
    state = tmp_path / "state"; mirror = tmp_path / "mirror"; dirs = [mirror / "stores" / "job7_none", mirror / "stores" / "job7_steer"]
    for n in (1, 2):
        s = gd.tick(args, *rest)
        assert s["identity_changed"] == 2 and s["identity_held"] == 0, (n, s)
    ev = _events(tmp_path)
    assert _n_dispatched(ev) == 2 and not [e for e in ev if e["event"] == "dispatch_identity_held"]
    ic = [e for e in ev if e["event"] == "dispatch_identity_changed"]
    assert [e["n_unchanged"] for e in ic] == [1, 2] and len({e["fingerprint"] for e in ic}) == 1 and ic[0]["hold_after"] == 3
    # the third unchanged refusal: HELD
    s3 = gd.tick(args, *rest)
    ev3 = _events(tmp_path)[len(ev):]
    (held,) = [e for e in ev3 if e["event"] == "dispatch_identity_held"]
    assert sorted(held["run_ids"]) == both and held["n_refusals"] == 3 and held["fingerprint"] == ic[0]["fingerprint"] and held["message"] == ANCHOR_LINE
    assert held["release_file"] == str(state / gd.DISPATCH_IDENTITY_HOLD_RELEASE)
    assert s3["identity_changed"] == 2 and s3["identity_held"] == 2 and s3["quarantined"] == 0
    ledger = json.loads((state / gd.DISPATCH_IDENTITY_LEDGER).read_text())
    assert all(ledger["runs"][r_]["held"] is True and ledger["runs"][r_]["n_unchanged"] == 3 for r_ in both)   # persisted: a re-exec grants nothing
    # tick 4: NO fourth dispatch -- no mirror, no grader; the copy kept; the problem still raised and naming the hold
    n_before = _n_dispatched(_events(tmp_path))
    s4 = gd.tick(args, *rest)
    ev4 = _events(tmp_path)
    assert _n_dispatched(ev4) == n_before and s4["identity_held"] == 2 and s4["identity_changed"] == 0 and s4["dispatched"] == 0
    (pr,) = [e for e in ev4[len(ev) + len(ev3):] if e["event"] == "problem"]
    assert pr["kind"] == "dispatch_identity_changed" and sorted(pr["held"]) == both and "held" in s4["problems"][0]
    assert all(d_.is_dir() for d_ in dirs) and not (state / "dispatch_quarantine.json").exists()
    # the operator's release file names one run of the group: released (why operator), the group dispatches again (and refuses again: count restarts)
    (state / gd.DISPATCH_IDENTITY_HOLD_RELEASE).write_text(f"# lift\n{both[0]}\n")
    s5 = gd.tick(args, *rest)
    ev5 = _events(tmp_path)[len(ev4):]
    (rel,) = [e for e in ev5 if e["event"] == "dispatch_identity_hold_released"]
    assert rel["why"] == "operator" and sorted(rel["held_run_ids"]) == both and not (state / gd.DISPATCH_IDENTITY_HOLD_RELEASE).exists()   # consumed
    assert _n_dispatched(ev5) == 1 and s5["identity_changed"] == 2 and s5["identity_held"] == 0
    assert [e["n_unchanged"] for e in ev5 if e["event"] == "dispatch_identity_changed"] == [1]
    # the registry settled: the next dispatch grades clean and clears the standing refusal
    flag.unlink()
    s6 = gd.tick(args, *rest)
    assert s6["dispatched"] == 2 and s6["identity_changed"] == 0 and s6["problems"] == []
    assert json.loads((state / gd.DISPATCH_IDENTITY_LEDGER).read_text())["runs"] == {}


def test_a_changed_registered_identity_releases_a_held_group_by_itself(tmp_path, monkeypatch):
    gd, root, args, rest, both, flag = _hold_world(tmp_path, monkeypatch)
    for _ in range(3):
        gd.tick(args, *rest)
    ev = _events(tmp_path)
    assert [e for e in ev if e["event"] == "dispatch_identity_held"] and _n_dispatched(ev) == 3
    s4 = gd.tick(args, *rest)
    assert s4["identity_held"] == 2 and _n_dispatched(_events(tmp_path)) == 3                      # held: no dispatch
    # the registry moves under the hold: the run's registered completion changes (a re-filed store) -> the fingerprint changes
    rid = both[0]
    R.write_records(os.path.join(root, "records", "c", "000077-patch.jsonl"),
                    [synth.rec("run_patch", {"run_id": rid, "fields": {"store_completion_sha256": "f" * 64}}, synth.t(900))])
    assert H.ingest_once(root, workers=0, min_free=0, derive_facts=False).refused == 0
    with DB.reading(_db(root)) as conn:
        assert conn.execute("SELECT store_completion_sha256 FROM run WHERE run_id = ?", (rid,)).fetchone()[0] == "f" * 64
    n_before = _n_dispatched(_events(tmp_path))
    s5 = gd.tick(args, *rest)
    ev5 = _events(tmp_path)
    (rel,) = [e for e in ev5 if e["event"] == "dispatch_identity_hold_released"]
    assert rel["why"] == "fingerprint changed" and sorted(rel["held_run_ids"]) == both
    assert _n_dispatched(ev5) == n_before + 1 and s5["identity_held"] == 0 and s5["identity_changed"] == 2      # re-selected and dispatched (still refusing)
    ic = [e for e in ev5 if e["event"] == "dispatch_identity_changed"]
    assert ic[-1]["n_unchanged"] == 1 and ic[-1]["fingerprint"] != ic[0]["fingerprint"]                           # the count restarts at the new identity
    # the pure fingerprint: the group's runs and the run ids the refusal names; a changed field changes it; an unknown run id still hashes
    fp_a = gd.identity_fingerprint(root, both, ANCHOR_LINE)
    assert fp_a == gd.identity_fingerprint(root, both, ANCHOR_LINE) and len(fp_a) == 16
    assert fp_a != gd.identity_fingerprint(root, both[:1], ANCHOR_LINE) and fp_a != gd.identity_fingerprint(root, both, "")
    assert gd._RUN_ID_RE.findall(ANCHOR_LINE) == ["synthcampaign:petri_self_preservation:aaaaaaaaaaaa:1111111111111111"]


# --------------------------------------------------------------------------------------------
# the stale-judge refusal is an identity change; the fingerprint covers the compared
# fields, the instrument's judge and the daemon's code identity
# --------------------------------------------------------------------------------------------

STALE_MSG = ("claim refused: run synthcampaign:petri_pooled:cccccccccccc:3333333333333333 now designates judge "
             "1111111111111111111111111111111111111111111111111111111111111111 (instrument 2222222222222222), the worklist row carried "
             "3333333333333333333333333333333333333333333333333333333333333333; re-select")
STALE_PETRI = "DriverError: --worklist registry: refusing — " + STALE_MSG        # petri_run wraps every claim refusal as a DriverError
STALE_MISALIGN = "--worklist registry: refusing — " + STALE_MSG                   # the misalign judge's SystemExit message
STALE_UNCAUGHT = "sisteer.registry.worklist.StaleRequiredJudge: " + STALE_MSG     # a driver that leaves it uncaught


def _load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)   # type: ignore[union-attr]
    return mod


def test_the_stale_judge_refusal_is_an_identity_change_in_every_wrapping_the_graders_give_it():
    gd = _load_daemon_module()
    tb = "Traceback (most recent call last):\n  File x.py\n"
    for line in (STALE_PETRI, STALE_MISALIGN, STALE_UNCAUGHT):
        assert gd.dispatch_failure_kind(1, tb + line + "\n") == "identity_changed", line
        assert gd.identity_changed_terminal(line)
    assert "now designates judge" in gd.IDENTITY_CHANGED_PHRASES
    assert gd.dispatch_failure_kind(1, tb + "RuntimeError: " + STALE_MSG + "\n") == "failed"            # no grader wrapping: a real failure
    assert gd.dispatch_failure_kind(1, STALE_PETRI + "\nKeyError: 'x'\n") == "failed"                   # the text buried mid-tail
    assert gd.dispatch_failure_kind(1, tb + "--worklist registry: refusing — run r is claimed by session 'other' (12 s ago)\n") == "failed"   # a held claim is not an identity change
    assert gd.dispatch_failure_kind(1, tb + "--worklist registry: refusing — " + ANCHOR_LINE[len("DriverError: "):] + "\n") == "identity_changed"   # the bare wrapper carries the other phrases too
    assert gd.dispatch_failure_kind(1, "sisteer.registry.pending.ClaimNotFolded: run x is not folded yet\n") == "not_folded"            # the other kinds stand


def test_the_fingerprint_covers_the_petri_compared_fields_the_instrument_judge_and_the_daemon_code_sha(tmp_path, monkeypatch):
    gd = _load_daemon_module()
    prun = _load_module("petri_run_fingerprint_pin", "scripts/petri_run.py")
    compared = set(prun.BORROW_IN_VIEW_IDENTITY_FIELDS) - {"run_id", "required_judge_id", "instrument_judge_id"}   # the key; the two judge fields read from the instrument
    assert compared <= set(gd.IDENTITY_FINGERPRINT_FIELDS), sorted(compared - set(gd.IDENTITY_FINGERPRINT_FIELDS))
    for f in ("eval_id", "producer_object_key_legacy", "object_id", "campaign", "model_id", "model_revision", "store_uri", "substrate", "n_items_generated",
              "instrument_id", "store_completion_sha256", "store_completion_artifact_uri"):
        assert f in gd.IDENTITY_FINGERPRINT_FIELDS, f
    root, rids = _root(tmp_path)
    both = sorted(rids[k] for k in ("job7@none", "job7@steer"))
    facts = gd.identity_fingerprint_facts(root, both, ANCHOR_LINE)
    assert facts["daemon_code_sha"] == gd.IMPORTED_CODE_SHA and facts["fields"] == list(gd.IDENTITY_FINGERPRINT_FIELDS)
    assert [r["run_id"] for r in facts["runs"]] == sorted(set(both) | {"synthcampaign:petri_self_preservation:aaaaaaaaaaaa:1111111111111111"})
    mine = [r for r in facts["runs"] if r["run_id"] == both[0]][0]
    assert mine["instrument_judge_id"] == synth.JUDGE_ID and mine["run"]["eval_id"] == synth.EVAL_ID and mine["run"]["producer_object_key_legacy"] in ("none", "lora@abcd")
    assert set(mine["current"]) == {"is_current", "why", "selection_state"}
    unknown = [r for r in facts["runs"] if r["run_id"].startswith("synthcampaign:petri_self_preservation")][0]
    assert unknown["run"] is None and unknown["instrument_judge_id"] is None                              # a run the registry does not know still hashes
    fp = gd.identity_fingerprint(root, both, ANCHOR_LINE)
    monkeypatch.setattr(gd, "IMPORTED_CODE_SHA", gd.IMPORTED_CODE_SHA + "+synthetic-fix")
    assert gd.identity_fingerprint(root, both, ANCHOR_LINE) != fp                                            # a new code identity is a new fingerprint


def test_a_re_exec_on_a_new_code_sha_releases_a_held_group_once(tmp_path, monkeypatch):
    gd, root, args, rest, both, flag = _hold_world(tmp_path, monkeypatch)
    for _ in range(3):
        gd.tick(args, *rest)
    assert len([e for e in _events(tmp_path) if e["event"] == "dispatch_identity_held"]) == 1
    assert gd.tick(args, *rest)["identity_held"] == 2                                                        # held: no dispatch
    n_before = _n_dispatched(_events(tmp_path))
    monkeypatch.setattr(gd, "IMPORTED_CODE_SHA", gd.IMPORTED_CODE_SHA + "+synthetic-fix")                  # the re-exec onto fixed code
    s = gd.tick(args, *rest)
    ev = _events(tmp_path)
    (rel,) = [e for e in ev if e["event"] == "dispatch_identity_hold_released"]
    assert rel["why"] == "fingerprint changed" and _n_dispatched(ev) == n_before + 1 and s["identity_held"] == 0 and s["identity_changed"] == 2
    assert [e["n_unchanged"] for e in ev if e["event"] == "dispatch_identity_changed"][-1] == 1             # ONCE: the count restarts at the new identity
    gd.tick(args, *rest); gd.tick(args, *rest)
    assert len([e for e in _events(tmp_path) if e["event"] == "dispatch_identity_held"]) == 2             # the same code refusing again holds again after three


def test_a_correction_of_a_routing_field_the_first_fingerprint_omitted_releases_a_held_group(tmp_path, monkeypatch):
    gd, root, args, rest, both, flag = _hold_world(tmp_path, monkeypatch)
    for _ in range(3):
        gd.tick(args, *rest)
    assert gd.tick(args, *rest)["identity_held"] == 2
    rid = both[0]                                      # job7@none: a same-id full run record correcting ONLY producer_object_key_legacy
    p = synth.run_payload("job7@none", "none-corrected", "none", 700, model_id=synth.MODEL_ID, view_legacy="synthmodel_4b_s1", cluster_job_id="7",
                          store_uri="/cluster_a/user/exp/stores/job7_none")
    p["run_id"] = rid
    R.write_records(os.path.join(root, "records", "c", "000078-correction.jsonl"), [synth.rec("run", p, synth.t(700))])
    assert H.ingest_once(root, workers=0, min_free=0, derive_facts=False).refused == 0
    with DB.reading(_db(root)) as conn:
        assert tuple(conn.execute("SELECT producer_object_key_legacy, lifecycle_state FROM run WHERE run_id = ?", (rid,)).fetchone()) == ("none-corrected", "generated")
    n_before = _n_dispatched(_events(tmp_path))
    s = gd.tick(args, *rest)
    ev = _events(tmp_path)
    (rel,) = [e for e in ev if e["event"] == "dispatch_identity_hold_released"]
    assert rel["why"] == "fingerprint changed" and _n_dispatched(ev) == n_before + 1 and s["identity_held"] == 0


# --------------------------------------------------------------------------------------------
# the dispatched STATIC graders' real `Refuse` print, and the precheck refusal classes
# --------------------------------------------------------------------------------------------

STATIC_GRADERS = ("scripts/sdfbase/full22/sampled/grade_static_sampled.py", "scripts/sdfbase/full22/static_padding/grade_padded_static.py")


def _real_refuse_line(rel: str, run: str, msg: str):
    """What the REAL static grader prints when its claim is refused -- `raise Refuse(f"[{run}] --worklist registry: refusing — {e}")` on a
    ClaimHeld (grade_static_sampled.py / grade_padded_static.py): the module imported, the class constructed, its stderr captured, its exit code."""
    mod = _load_module("static_refuse_" + Path(rel).stem, rel)
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        try:
            raise mod.Refuse(f"[{run}] --worklist registry: refusing — {msg}")
        except SystemExit as e:
            code = e.code
    return buf.getvalue().strip().splitlines()[-1], code


def test_the_static_graders_real_refuse_output_classifies_identity_changed_only_with_a_stable_phrase():
    gd = _load_daemon_module()
    for rel in STATIC_GRADERS:
        line, code = _real_refuse_line(rel, "synthtarget_sampled_t1", STALE_MSG)
        assert code == 2 and line.startswith("REFUSING: [synthtarget_sampled_t1] --worklist registry: refusing — claim refused: run "), line
        assert gd.dispatch_failure_kind(2, "[sampled-grade] judge=... \n" + line + "\n") == "identity_changed", line
        held, code2 = _real_refuse_line(rel, "synthtarget_sampled_t1", "run r is claimed by session 'other' (12 s ago)")
        assert code2 == 2 and gd.dispatch_failure_kind(2, held + "\n") == "failed"                                # a held claim: not transient
        generic, _ = _real_refuse_line(rel, "synthtarget_sampled_t1", "claim refused (precheck): run r: synthetic")
        assert gd.dispatch_failure_kind(2, generic + "\n") == "identity_changed"                                   # the owner's precheck refusal, wrapped
    # a REFUSING line of any other kind is a real failure, whatever it says; the registry wrapper after the run tag is required
    assert gd.dispatch_failure_kind(2, "REFUSING: [run] the source changed during binding/snapshot (synthetic)\n") == "failed"
    assert gd.dispatch_failure_kind(2, "REFUSING: [run] grader rubric v2 is not v3\n") == "failed"
    assert gd.dispatch_failure_kind(2, "REFUSING: --worklist registry: refusing — " + STALE_MSG + "\n") == "failed"   # no run tag: not the graders' print
    assert gd.dispatch_failure_kind(2, "REFUSING: [run] --worklist registry: refusing — " + ANCHOR_LINE[len("DriverError: "):] + "\n") == "identity_changed"
    # the precheck refusal classes name themselves: the exception line alone is the phrase, in the CLI's and the uncaught format
    assert gd.dispatch_failure_kind(1, "sisteer.registry.pending.ClaimPrecheckRefused: claim refused (precheck): run x: synthetic\n") == "identity_changed"
    assert gd.dispatch_failure_kind(1, "ClaimPrecheckRefused: claim refused (precheck): run x: synthetic\n") == "identity_changed"
    assert gd.dispatch_failure_kind(1, "claim refused: ClaimPrecheckRefused: claim refused (precheck): run x: synthetic\n") == "failed"   # the CLI's own line is not a grader's terminal
    assert gd.dispatch_failure_kind(1, "DriverError: --worklist registry: refusing — claim refused (precheck): run x: synthetic\n") == "identity_changed"
    assert gd.dispatch_failure_kind(1, "RuntimeError: claim refused (precheck): run x: synthetic\n") == "failed"                       # no admitted prefix
    assert gd.identity_changed_terminal("StaleRequiredJudge: anything at all") and not gd.identity_changed_terminal("ValueError: StaleRequiredJudge: x")


# --------------------------------------------------------------------------------------------
# the static graders' PRE-CLAIM refusal is never transient; the REFUSING tag must be the group's
# --------------------------------------------------------------------------------------------

#: degen-fix-2's pre-claim refusal (grade_static_sampled.py: `Refuse(f"[{run}] --worklist registry: refusing before any claim — the degeneration
#: instrument is not ready ({type(e).__name__}: {e}); nothing claimed")`) -- the reasons it carries: a missing tokenizer snapshot, a definition
#: pin mismatch, mixed model_ids, a NULL n_items_generated. None of these is an identity change; the matcher requires `refusing — ` right after
#: the run tag and must keep requiring it.
PRECLAIM_REASONS = ("FileNotFoundError: the served target's tokenizer snapshot is not in the local cache",
                    "ValueError: definition pin mismatch: union5 detectors pinned at sha a, found sha b",
                    "ValueError: the pending rows carry mixed model_ids",
                    "ValueError: NULL n_items_generated on the pending rows")


def test_the_static_graders_real_pre_claim_refusal_is_a_failure_never_an_identity_change():
    gd = _load_daemon_module()
    for rel in STATIC_GRADERS:
        mod = _load_module("static_refuse_" + Path(rel).stem, rel)
        for reason in PRECLAIM_REASONS:
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                try:
                    raise mod.Refuse(f"[synthtarget_sampled_t1] --worklist registry: refusing before any claim — the degeneration instrument is not ready ({reason}); nothing claimed")
                except SystemExit as e:
                    code = e.code
            line = buf.getvalue().strip().splitlines()[-1]
            assert code == 2 and line.startswith("REFUSING: [synthtarget_sampled_t1] --worklist registry: refusing before any claim — "), line
            assert gd.dispatch_failure_kind(2, "[sampled-grade] ...\n" + line + "\n") == "failed", line
            assert gd.dispatch_failure_kind(2, line + "\n", group_tags=frozenset({"synthtarget_sampled_t1"})) == "failed"
            assert not gd.identity_changed_terminal(line)
    # the matcher's shape is pinned: `refusing — ` must follow the tag immediately; a phrase after a pre-claim wrapper never rescues it
    assert gd.dispatch_failure_kind(2, "REFUSING: [r] --worklist registry: refusing before any claim — " + STALE_MSG + "\n") == "failed"
    assert gd.dispatch_failure_kind(2, "REFUSING: [r] --worklist registry: refusing — " + STALE_MSG + "\n") == "identity_changed"
    assert gd.IDENTITY_CHANGED_TERMINAL_RE.pattern.count("refusing (?:\\u2014|-+) ") == 2 and "before any claim" not in gd.IDENTITY_CHANGED_TERMINAL_RE.pattern


def test_a_refusing_lines_run_tag_must_belong_to_the_dispatch_group():
    gd = _load_daemon_module()
    rid_a, rid_b = "synthcampaign:syntheval:aaaaaaaaaaaa:1111111111111111", "synthcampaign:syntheval:aaaaaaaaaaaa:2222222222222222"
    group = frozenset({"synthtarget_sampled_t1", rid_a, rid_b})
    line = "REFUSING: [synthtarget_sampled_t1] --worklist registry: refusing — " + STALE_MSG
    assert gd.dispatch_failure_kind(2, line + "\n", group_tags=group) == "identity_changed"                           # the group's private run
    assert gd.dispatch_failure_kind(2, f"REFUSING: [{rid_a}] --worklist registry: refusing — " + STALE_MSG + "\n", group_tags=group) == "identity_changed"   # or one of its run ids
    assert gd.dispatch_failure_kind(2, "REFUSING: [another_private_run] --worklist registry: refusing — " + STALE_MSG + "\n", group_tags=group) == "failed"   # a foreign tag
    assert gd.dispatch_failure_kind(2, "REFUSING: [] --worklist registry: refusing — " + STALE_MSG + "\n", group_tags=group) == "failed"                     # an empty tag
    assert gd.dispatch_failure_kind(2, "REFUSING: [another_private_run] --worklist registry: refusing — " + STALE_MSG + "\n") == "identity_changed"           # no group known: any tag (the pure classifier)
    assert gd.dispatch_failure_kind(2, "REFUSING: [another_private_run] --worklist registry: refusing — " + STALE_MSG + "\n", group_tags=frozenset()) == "failed"
    # the untagged wrappers do not carry a run tag: the group check never applies to them
    assert gd.dispatch_failure_kind(1, STALE_MISALIGN + "\n", group_tags=group) == "identity_changed"
    assert gd.dispatch_failure_kind(1, STALE_PETRI + "\n", group_tags=frozenset({"other"})) == "identity_changed"
    assert gd._REFUSING_TAG_RE.match("REFUSING: [x y] --worklist registry: refusing — z").group(1) == "x y"
