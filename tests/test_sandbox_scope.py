"""``sisteer.capeval.tinycap.sandbox_scope`` — the pure sandbox-ownership
module: the owner tuple from the environment, all-string
create-time tags, the owner filter, the in-scope predicate, the durable
event record, and the verify loop ``sweep_owned`` over MIXED-owner fakes
(the predicate is what these test — production listings are already
owner-filtered by Modal)."""

from __future__ import annotations

import json

import pytest

from sisteer.capeval.tinycap import sandbox_scope as S

OWNER = S.Owner("abc1234-deadbeef", "none", "fc-1111")
SIBLING = S.Owner("abc1234-deadbeef", "constv2-mix-full__lora_r4_ce_minus_ctl", "fc-2222")
OLD_ATTEMPT = S.Owner("abc1234-deadbeef", "none", "fc-0000")


class _SB:
    """A fake Modal sandbox: tags (or None = unreadable), a terminate that
    may fail, and a `gone` flag the fake lister honours after a round."""

    def __init__(self, sid, tags, *, fail_terminate=0):
        self.object_id, self._tags, self.fail_terminate = sid, tags, fail_terminate
        self.terminated = False
        self.terminate_calls = 0

    def get_tags(self):
        if self._tags is None:
            raise RuntimeError("tags RPC failed")
        return dict(self._tags)

    def terminate(self):
        self.terminate_calls += 1
        if self.fail_terminate > 0:
            self.fail_terminate -= 1
            raise RuntimeError("terminate RPC failed")
        self.terminated = True


def _lister(boxes, *, lag_rounds=0):
    """Lists every box not yet terminated; a terminated box keeps listing for
    `lag_rounds` extra listings (Modal's terminate is asynchronous)."""
    state = {"n": 0}

    def _list():
        state["n"] += 1
        out = []
        for b in boxes:
            if not b.terminated:
                out.append(b)
            elif b.terminated and getattr(b, "gone_at", None) is None:
                b.gone_at = state["n"] + lag_rounds
                out.append(b)
            elif state["n"] < b.gone_at:
                out.append(b)
        return out
    return _list


def test_owner_from_env_all_or_nothing():
    assert S.owner_from_env({}) is None
    env = S.owner_env(OWNER, "/vol/x/watch/sandboxes_fc-1111.jsonl")
    assert env == {"TINYCAP_OWNER_RUN_ID": "abc1234-deadbeef", "TINYCAP_OWNER_CELL": "none",
                   "TINYCAP_OWNER_FUNCTION_CALL_ID": "fc-1111",
                   "TINYCAP_SANDBOX_RECORD": "/vol/x/watch/sandboxes_fc-1111.jsonl"}
    assert S.owner_from_env(env) == OWNER
    with pytest.raises(ValueError, match="partial sandbox owner"):
        S.owner_from_env({"TINYCAP_OWNER_RUN_ID": "r", "TINYCAP_OWNER_CELL": "c"})
    assert S.OWNER_ENV_VARS == ("TINYCAP_OWNER_RUN_ID", "TINYCAP_OWNER_CELL", "TINYCAP_OWNER_FUNCTION_CALL_ID")
    assert S.SANDBOX_RECORD_ENV == "TINYCAP_SANDBOX_RECORD"


def test_owner_tags_are_all_strings_including_an_int_sample_id():
    """BigCodeBench's sample ids are ints and Modal serialises tag values into
    a protobuf string field — every name and value is str()."""
    tags = S.owner_tags(OWNER, "bigcodebench_200", {"__sample_id__": 17, "test": "..."})
    assert tags == {"created_by": "inspect-ai", "tinycap.run_id": "abc1234-deadbeef", "tinycap.cell": "none",
                    "tinycap.function_call_id": "fc-1111", "tinycap.task": "bigcodebench_200",
                    "tinycap.sample_id": "17"}
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in tags.items())
    assert S.owner_tags(OWNER, "t", {})["tinycap.sample_id"] == ""
    # the filter is the marker + the FULL tuple, nothing per-sample
    assert S.owner_filter(OWNER) == {"created_by": "inspect-ai", "tinycap.run_id": "abc1234-deadbeef",
                                     "tinycap.cell": "none", "tinycap.function_call_id": "fc-1111"}
    assert set(S.owner_filter(OWNER)) <= set(tags)


def test_in_scope_needs_the_whole_tuple_and_readable_tags():
    mine = S.owner_tags(OWNER, "t", {"__sample_id__": 1})
    assert S.in_scope(mine, OWNER)
    assert not S.in_scope(S.owner_tags(SIBLING, "t", {}), OWNER)          # same run, other cell
    assert not S.in_scope(S.owner_tags(OLD_ATTEMPT, "t", {}), OWNER)      # same run + cell, older attempt
    assert not S.in_scope(None, OWNER)                                     # unreadable = not evidence
    assert not S.in_scope({**mine, "created_by": "someone-else"}, OWNER)
    assert not S.in_scope({}, OWNER)


def test_events_are_appended_durably_and_read_back(tmp_path):
    p = tmp_path / "watch" / "sandboxes_fc-1111.jsonl"
    S.append_event(p, S.EVENT_CREATED, "sb-1", OWNER, task="bigcodebench_200", sample_id="17")
    S.append_event(p, S.EVENT_TERMINATED, "sb-1", OWNER, verified_by="poll")
    with pytest.raises(ValueError, match="unknown sandbox event"):
        S.append_event(p, "vanished", "sb-1", OWNER)
    p.open("a").write("{torn")
    evs = S.read_events(p)
    assert [e["event"] for e in evs] == ["created", "terminated"]
    assert evs[0]["owner"] == OWNER.as_dict() and evs[0]["sample_id"] == "17"
    assert evs[1]["verified_by"] == "poll"
    # no record configured: the event is returned, nothing written
    assert S.append_event(None, S.EVENT_CREATED, "sb-2", OWNER)["sandbox_id"] == "sb-2"
    assert S.read_events(tmp_path / "absent.jsonl") == []
    # tinycap-verify-rate-limit (2026-09-30): the provider's rate-limit retry is a durable event too (no sandbox id: nothing
    # was created); it carries the attempt, the wait and the elapsed time so the record shows how often the 5/s limit bit
    assert set(S.EVENTS) == {"created", "terminated", "terminate_unverified", "cleanup_skipped_interrupted", "create_rate_limited"}
    assert S.EVENT_CREATE_RATE_LIMITED == "create_rate_limited"
    p2 = tmp_path / "watch" / "sandboxes_fc-2222.jsonl"
    rl = S.append_event(p2, S.EVENT_CREATE_RATE_LIMITED, "", OWNER, attempt=0, wait_s=1.5, elapsed_s=0.2, error="ResourceExhaustedError: 5/s")
    assert rl["sandbox_id"] == "" and rl["attempt"] == 0 and rl["wait_s"] == 1.5
    S.append_event(p2, S.EVENT_CREATED, "sb-3", OWNER)
    assert [e["event"] for e in S.read_events(p2)] == ["create_rate_limited", "created"]


def test_sweep_terminates_only_the_owners_sandboxes_and_verifies_by_relist(tmp_path):
    """A wider listing (the mixed-owner fake) still terminates ONLY this
    owner's; a sibling cell's, an older attempt's and an unreadable one are
    LEFT and named; verified_empty only when nothing in scope still lists
    and every listed sandbox's tags were readable."""
    mine = [_SB("sb-a", S.owner_tags(OWNER, "t", {"__sample_id__": 1})),
            _SB("sb-b", S.owner_tags(OWNER, "t", {"__sample_id__": 2}))]
    sib = _SB("sb-sib", S.owner_tags(SIBLING, "t", {"__sample_id__": 1}))
    old = _SB("sb-old", S.owner_tags(OLD_ATTEMPT, "t", {"__sample_id__": 1}))
    unreadable = _SB("sb-??", None)
    record = tmp_path / "sandboxes.jsonl"
    slept = []
    rec = S.sweep_owned(list_sandboxes=_lister([*mine, sib, old, unreadable]), owner=OWNER, by="test",
                        sleep=slept.append, max_rounds=3, log=lambda s: None, record_path=record)
    assert rec["ids_terminated"] == ["sb-a", "sb-b"]
    assert all(b.terminated for b in mine) and not sib.terminated and not old.terminated
    assert unreadable.terminate_calls == 0
    whys = {d["sandbox_id"]: d["why"] for d in rec["left"]}
    assert set(whys) == {"sb-sib", "sb-old", "sb-??"}
    assert "unreadable" in whys["sb-??"] and "not this owner" in whys["sb-sib"]
    # an unreadable listing is evidence the sweep does not have: never verified_empty
    assert rec["verified_empty"] is False and rec["tags_unreadable"] == ["sb-??"]
    assert rec["still_live"] == ["sb-??"] and rec["rounds"] == 3
    assert set(rec) == set(S.SWEEP_RECORD_FIELDS) and rec["owner"] == OWNER.as_dict()
    # the two owned terminations were CONFIRMED by re-list and recorded as such
    evs = S.read_events(record)
    assert sorted((e["sandbox_id"], e["event"], e.get("verified_by")) for e in evs) == [
        ("sb-a", "terminated", "relist"), ("sb-b", "terminated", "relist")]


def test_sweep_verified_empty_when_the_owner_filter_lists_only_its_own(tmp_path):
    """The production shape: the listing is already owner-filtered (Modal
    returns only sandboxes carrying ALL the owner tags), so every listed box is
    in scope; asynchronous termination (a box that keeps listing for a round)
    is waited out and confirmed by object id."""
    mine = [_SB("sb-a", S.owner_tags(OWNER, "t", {})), _SB("sb-b", S.owner_tags(OWNER, "t", {}))]
    slept = []
    rec = S.sweep_owned(list_sandboxes=_lister(mine, lag_rounds=1), owner=OWNER, by="finally",
                        sleep=slept.append, max_rounds=6, settle_s=2.5, log=lambda s: None)
    assert rec["verified_empty"] is True and rec["ids_terminated"] == ["sb-a", "sb-b"]
    assert rec["left"] == [] and rec["still_live"] == [] and rec["could_not_terminate"] == []
    assert rec["n_listed_first_round"] == 2 and slept and all(s == 2.5 for s in slept)
    # nothing listed at all: verified on the first round, no sleep
    empty = S.sweep_owned(list_sandboxes=lambda: [], owner=OWNER, by="finally", sleep=slept.append,
                          log=lambda s: None)
    assert empty["verified_empty"] is True and empty["rounds"] == 1 and empty["n_listed_first_round"] == 0


def test_sweep_retries_a_failing_terminate_and_reports_it_once(tmp_path):
    flaky = _SB("sb-f", S.owner_tags(OWNER, "t", {}), fail_terminate=2)
    record = tmp_path / "r.jsonl"
    rec = S.sweep_owned(list_sandboxes=_lister([flaky]), owner=OWNER, by="t", sleep=lambda s: None,
                        max_rounds=6, log=lambda s: None, record_path=record)
    assert flaky.terminated and rec["verified_empty"] is True and rec["could_not_terminate"] == []
    assert flaky.terminate_calls == 3
    # a terminate that never succeeds: reported once with its attempt count, unverified
    stuck = _SB("sb-s", S.owner_tags(OWNER, "t", {}), fail_terminate=99)
    rec = S.sweep_owned(list_sandboxes=_lister([stuck]), owner=OWNER, by="t", sleep=lambda s: None,
                        max_rounds=2, log=lambda s: None, record_path=record)
    assert rec["verified_empty"] is False
    assert [d["sandbox_id"] for d in rec["could_not_terminate"]] == ["sb-s"]
    assert rec["could_not_terminate"][0]["attempts"] == 2 and rec["still_live"] == ["sb-s"]


def test_a_terminate_accepted_but_never_confirmed_is_recorded_unverified(tmp_path):
    """Modal's terminate() REQUESTS termination; a box that keeps listing past
    the rounds is `terminate_unverified`, never `terminated`."""
    lingering = _SB("sb-l", S.owner_tags(OWNER, "t", {}))
    lingering.terminate = lambda: setattr(lingering, "requested", True)       # never actually goes away
    record = tmp_path / "r.jsonl"
    rec = S.sweep_owned(list_sandboxes=lambda: [lingering], owner=OWNER, by="t", sleep=lambda s: None,
                        max_rounds=2, log=lambda s: None, record_path=record)
    assert rec["verified_empty"] is False and rec["still_live"] == ["sb-l"]
    evs = S.read_events(record)
    assert [(e["event"], e["sandbox_id"]) for e in evs] == [("terminate_unverified", "sb-l")]
    assert json.loads(record.read_text().splitlines()[0])["by"] == "t"
