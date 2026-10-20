"""petri-grok: the Petri unit's receipted sha advance admits the judge refusal-fallback chain's
APPENDED TERMINAL non-Claude leg — `petri.judge_refusal_fallback_models` [opus-4-8] -> [opus-4-8, grok-4.6-20260810], the IMMUTABLE dated slug —
as a JUDGE-ONLY `petri-role-fallback` amendment that FOLLOWS the prior receipt (the auditor key stands as it is): the commit a strict descendant
of the amended sha, the old run set the prior receipt's, the carried set carried FORWARD — a carried run that completed since completed under the
AMENDED contract, keeps its recorded provenance (code_sha; its per-sample chain is the store's) and is listed under `carried_completed`, never
refused and never re-run. Anything else — the mutable alias, Grok replacing or preceding Opus, a second non-Claude leg, the auditor moving
alongside, another petri key, a top-level key, a judge-only INITIAL introduction — refuses by name and writes nothing. Synthetic repos and ids."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB  # noqa: E402
from test_amend_contract_enforceable import FakeSacct, _git, _ledger, _set_state  # noqa: E402
from test_amend_contract_petri_role_fallback import MANIFEST_REL, WAVE_FILE, _amend, _nothing_written, petri_world  # noqa: E402

sys.path.insert(0, str(LIB))
import stage as ST  # noqa: E402

GROK = "openrouter/x-ai/grok-4.6-20260810"
OPUS = "anthropic/claude-opus-4-8"
SONNET = "anthropic/claude-sonnet-4-6"
JUDGE_BEFORE = [OPUS]
JUDGE_AFTER = [OPUS, GROK]
LIVE = {("cluster_a", "1"): "RUNNING"}


def _grok_world(tmp: Path, *, judge_after=None, auditor_after=None, extra=None, mutate=None) -> dict:
    """petri_world's two commits (the frozen tree; X: the two keys introduced), then Y: the judge chain's appended terminal leg — and whatever
    the test pushes into the diff beyond it."""
    w = petri_world(tmp)
    man = w["repo"] / MANIFEST_REL
    doc = yaml.safe_load(man.read_text()); p = doc["petri"]
    p["judge_refusal_fallback_models"] = list(JUDGE_AFTER if judge_after is None else judge_after)
    if auditor_after is not None:
        p["auditor_refusal_fallback_models"] = list(auditor_after)
    for k, v in (extra or {}).items():
        p[k] = v
    if mutate:
        mutate(doc)
    man.write_text(yaml.safe_dump(doc, sort_keys=False))
    _git(w["repo"], "add", "-A"); _git(w["repo"], "commit", "-q", "-m", "the terminal Grok judge leg (Y)")
    w["grok_sha"] = _git(w["repo"], "rev-parse", "HEAD")
    return w


def _rows(db: Path) -> list:
    c = sqlite3.connect(db)
    try:
        return c.execute("select * from run order by run_id").fetchall()
    finally:
        c.close()


def test_the_table_admits_exactly_the_appended_dated_grok_leg_for_the_judge():
    t = ST.PETRI_ROLE_FALLBACK_TRANSITIONS
    assert t["auditor_refusal_fallback_models"] == ((None, [SONNET]),)
    assert t["judge_refusal_fallback_models"] == ((None, [OPUS]), (JUDGE_BEFORE, JUDGE_AFTER))
    assert ST.PETRI_ROLE_FALLBACK_INITIAL == {"auditor_refusal_fallback_models": (None, [SONNET]), "judge_refusal_fallback_models": (None, [OPUS])}
    # the mutable alias is admitted nowhere
    assert not any("openrouter/x-ai/grok-4.6" in (a or []) for trs in t.values() for _, a in trs)


def test_the_judge_only_grok_leg_is_receipted_after_the_first_receipt_the_sha_advances_and_completed_carried_runs_keep_their_provenance(tmp_path):
    w = _grok_world(tmp_path); sacct = FakeSacct(LIVE)
    e1 = _amend(w, sacct, note="attempt 7: the two keys")
    assert e1["sha_advance"] == {"from": w["old_sha"], "to": w["new_sha"]} and len(e1["carried_run_ids"]) == 7
    # two carried runs COMPLETE under the amended (X) contract before the second receipt: carried FORWARD with their provenance, not refused
    done = sorted([w["runs"][0]["run_id"], w["runs"][3]["run_id"]])
    for rid in done:
        _set_state(w["db"], rid, "judged")
    before = _rows(w["db"])
    e2 = _amend(w, sacct, commit=w["grok_sha"], tip=w["grok_sha"], note="D869: the terminal Grok judge leg for the judge-only re-score")
    assert e2["kind"] == "petri-role-fallback"
    assert e2["role_diff"] == [{"field": "petri.judge_refusal_fallback_models", "before": JUDGE_BEFORE, "after": JUDGE_AFTER}]      # judge only
    assert e2["old_sha"] == w["new_sha"] and e2["sha_advance"] == {"from": w["new_sha"], "to": w["grok_sha"]}
    assert e2["follow_up_of"] == {"utc": e1["utc"], "kind": "petri-role-fallback", "commit": w["new_sha"]}
    assert e2["old_contract_sha256"] == e2["new_contract_sha256"] == e1["new_contract_sha256"] and e2["changed_instruments"] == {}
    assert e2["carried_run_ids"] == e1["carried_run_ids"]                                                                           # the whole set carried forward
    assert sorted(x["run_id"] for x in e2["carried_completed"]) == done
    assert all(x["lifecycle_state"] == "judged" and x["code_sha"] == w["old_sha"] and x["under"] == w["new_sha"] for x in e2["carried_completed"])
    lines = {x["run_id"]: x for x in e2["old_run_ids"]}
    assert all(lines[r]["carried"] is True and "completed under the AMENDED contract" in lines[r]["why"] for r in done)
    assert all(lines[r]["carried"] is True and lines[r]["lifecycle_state"] == "running" for r in e1["carried_run_ids"] if r not in done)
    assert e2["completed_run_check"]["completed"] == [] and e2["manifest_rel"] == MANIFEST_REL and e2["old_manifest_sha256"] == e1["new_manifest_sha256"]
    # the receipt carries provenance and rewrites none: the registry rows are byte-for-byte what they were
    assert _rows(w["db"]) == before
    # the sha table, the wave file, the ledger (receipt then advance, twice) and the sidecar agree; both receipts re-check PASS
    t = ST.sha_table(w["ledger"])[w["lw"]]
    assert t["sha"] == w["grok_sha"] and t["frozen_sha"] == w["old_sha"] and t["amended"] == 2, t
    wave = json.load(open(w["state"] / WAVE_FILE))
    assert wave["unit_shas"][w["lw"]]["sha"] == w["grok_sha"] and wave["unit_shas"][w["lw"]]["amended"] == 2
    ev = [x for x in _ledger(w["state"]) if x["event"] in ("contract_amendment", "sha_provenance")]
    assert [x["event"] for x in ev] == ["contract_amendment", "sha_provenance"] * 2
    assert ev[2]["role_diff"] == e2["role_diff"] and ev[2]["carried_completed"] == e2["carried_completed"] and ev[2]["follow_up_of"] == e2["follow_up_of"]
    assert ev[3]["sha"] == w["grok_sha"] and ev[3]["from_sha"] == w["new_sha"] and ev[3]["kind"] == "amend"
    assert ST.reconcile_provenance(w["ledger"]) == []
    assert ST.effective_contract(ST.read_amendments(w["ledger"]), w["unit"]) == e2["contract"] == e1["contract"]
    res = ST.recheck_amendments(w["ledger"], w["lw"], repo=str(w["repo"]), tip=w["grok_sha"], registry_db=str(w["db"]), sacct_tasks=sacct)
    assert [r["verdict"] for r in res] == ["PASS", "PASS"], res
    assert res[1]["role_diff"] == ["petri.judge_refusal_fallback_models"]
    # applied again at Y: the diff is empty, nothing to amend, nothing written
    with pytest.raises(ValueError, match="absent from the diff"):
        _amend(w, sacct, commit=w["grok_sha"], tip=w["grok_sha"])
    assert len(ST.read_amendments(w["ledger"])[w["lw"]]) == 2


def test_anything_but_the_appended_terminal_dated_grok_leg_refuses_by_name_and_writes_nothing(tmp_path):
    sacct = FakeSacct(LIVE); n = [0]

    def world(**kw):
        n[0] += 1
        w = _grok_world(tmp_path / f"w{n[0]}", **kw); _amend(w, sacct, note="attempt 7"); return w

    def only_the_first_receipt(w):
        assert len(ST.read_amendments(w["ledger"])[w["lw"]]) == 1 and ST.sha_table(w["ledger"])[w["lw"]]["sha"] == w["new_sha"]
        assert len([x for x in _ledger(w["state"]) if x["event"] == "sha_provenance"]) == 1

    def refused(w, pattern):
        with pytest.raises(ValueError, match=pattern):
            _amend(w, sacct, commit=w["grok_sha"], tip=w["grok_sha"])
        only_the_first_receipt(w)

    # the mutable alias is not the admitted id
    refused(world(judge_after=[OPUS, "openrouter/x-ai/grok-4.6"]), r"petri\.judge_refusal_fallback_models is \['anthropic/claude-opus-4-8'\] -> \[.*grok-4\.6'\], not the admitted transitions")
    # Grok replacing Opus, Grok first, a second non-Claude leg, a third leg: none is the appended terminal leg
    for judge in ([GROK], [GROK, OPUS], [OPUS, GROK, GROK], [OPUS, GROK, "openrouter/x-ai/grok-5"]):
        refused(world(judge_after=judge), "not the admitted transitions")
    # the auditor moving alongside the leg — to anything — is not admitted
    refused(world(auditor_after=[SONNET, GROK]), r"petri\.auditor_refusal_fallback_models is .* not the admitted transition ")
    # another petri key (a harness-hash input) or a top-level key moving with the leg
    refused(world(extra={"auditor_fallback_models": [GROK]}), r"petri\.auditor_fallback_models changed .* not one of the admitted fields")
    refused(world(mutate=lambda doc: doc.__setitem__("note_outside_petri", "x")), r"top-level `note_outside_petri` changed")
    # a judge-only INITIAL introduction (no auditor key) still refuses: the first amendment carries the whole per-request fallback
    w0 = petri_world(tmp_path / "initial"); man = w0["repo"] / MANIFEST_REL
    doc = yaml.safe_load(man.read_text()); doc["petri"].pop("auditor_refusal_fallback_models")
    man.write_text(yaml.safe_dump(doc, sort_keys=False)); _git(w0["repo"], "add", "-A"); _git(w0["repo"], "commit", "-q", "-m", "judge only, initial")
    w0["new_sha"] = _git(w0["repo"], "rev-parse", "HEAD")
    with pytest.raises(ValueError, match=r"admitted transition\(s\) of \['auditor_refusal_fallback_models'\] are absent from the diff"):
        _amend(w0, sacct)
    _nothing_written(w0)
    # the Grok leg straight from the frozen tree (no prior receipt; judge absent -> [opus, grok]) is not an admitted transition either
    w1 = petri_world(tmp_path / "straight"); man = w1["repo"] / MANIFEST_REL
    doc = yaml.safe_load(man.read_text()); doc["petri"]["judge_refusal_fallback_models"] = list(JUDGE_AFTER)
    man.write_text(yaml.safe_dump(doc, sort_keys=False)); _git(w1["repo"], "add", "-A"); _git(w1["repo"], "commit", "-q", "-m", "both keys, grok at once")
    w1["new_sha"] = _git(w1["repo"], "rev-parse", "HEAD")
    with pytest.raises(ValueError, match=r"petri\.judge_refusal_fallback_models is None -> \[.*grok-4\.6-20260810'\], not the admitted transitions"):
        _amend(w1, sacct)
    _nothing_written(w1)


def test_the_first_receipt_still_refuses_a_run_completed_under_the_frozen_contract_and_a_later_one_refuses_a_non_descendant(tmp_path):
    sacct = FakeSacct(LIVE)
    # the first receipt (no prior): an old-contract completion refuses as before — the carry-forward is for LATER receipts only
    w = _grok_world(tmp_path / "first")
    _set_state(w["db"], w["runs"][2]["run_id"], "generated")
    with pytest.raises(ValueError, match=r"1 run\(s\) COMPLETED under the old contract"):
        _amend(w, sacct)
    _nothing_written(w)
    # a later receipt whose commit is not a strict descendant of the amended sha (a sibling of X carrying the leg) refuses by name
    w = _grok_world(tmp_path / "sibling"); _amend(w, sacct, note="attempt 7")
    _git(w["repo"], "checkout", "-q", w["old_sha"]); _git(w["repo"], "checkout", "-q", "-b", "sibling")
    man = w["repo"] / MANIFEST_REL; doc = yaml.safe_load(man.read_text())
    doc["petri"]["auditor_refusal_fallback_models"] = [SONNET]; doc["petri"]["judge_refusal_fallback_models"] = list(JUDGE_AFTER)
    man.write_text(yaml.safe_dump(doc, sort_keys=False)); _git(w["repo"], "add", "-A"); _git(w["repo"], "commit", "-q", "-m", "a sibling of X")
    sib = _git(w["repo"], "rev-parse", "HEAD")
    with pytest.raises(ValueError, match=r"is not a strict descendant of the unit's amended sha .* a later role-fallback receipt advances forward only"):
        _amend(w, sacct, commit=sib, tip=sib)
    assert len(ST.read_amendments(w["ledger"])[w["lw"]]) == 1 and ST.sha_table(w["ledger"])[w["lw"]]["sha"] == w["new_sha"]
