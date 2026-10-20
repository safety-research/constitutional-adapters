"""The judge cache is WRITE-ONCE per key; a same-key re-judge that disagrees
is recorded, never resolved silently (2026-09-09).

Observed live on the alphasteer-full22 glm46vflash selection: the post-filing
gate refused two legs with equal counts (judged 1400 vs 1400, abstained 1 vs
1) — an item-level LABEL FLIP. ``JudgeCache.put`` was ``INSERT OR REPLACE``,
so the last writer won a key: two selections overlapped on one sqlite for ~90 s
(the four targets share the same 1,401 prompts, and short canonical refusals
make identical (prompt, response) keys across models), and a nondeterministic
judge asked twice for one identical key returned two labels. The same path is
reachable inside ONE process whenever two legs are judged in parallel on a
shared key (the ``--leg-parallel`` help text records 27 such items), because
``judge_all`` checks the cache upfront and then live-calls every miss.

The cache's contract is that a key's verdict is STABLE — a later reader gets
what the first reader got. Write-once is that contract made atomic: the first
verdict filed for a key stands (verdict, org stamp, timestamp — all of it), a
later live verdict for the same key is compared against it, adopted from the
store if it differs, and the disagreement is written to ``verdict_conflicts``
and counted (``n_verdict_conflicts``) — a judge that returns two labels for
one input is a stability signal, not noise to absorb. The one deliberate
overwrite is the explicit ``reattempt_abstentions`` re-judge of a cached
abstention sentinel; a stored LABEL is never replaced by anything.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import sisteer.capeval.judging as j                       # noqa: E402
from sisteer.capeval.judging import (                     # noqa: E402
    ClaudeJudge, JudgeCache, JudgingError)

from test_capeval_judging import (                        # noqa: E402
    ORBENCH_RUBRIC_V1, _Client, _Msg, _cpi_msg, _single_req)


def _judge(tmp_path, low_fn, **kw):
    return ClaudeJudge(cache=JudgeCache(tmp_path / "c.sqlite"),
                       clients={"lowprio": _Client(low_fn)},
                       concurrency=kw.pop("concurrency", 4), **kw)


# --- the cache itself --------------------------------------------------------------


def test_second_put_of_a_different_verdict_is_dropped_and_recorded(tmp_path):
    cache = JudgeCache(tmp_path / "c.sqlite")
    first = cache.put("k", {"label": "full_refusal"}, "lowprio")
    assert first == {"label": "full_refusal"}
    second = cache.put("k", {"label": "partial_refusal"}, "highprio")
    # what STANDS is returned, and it is the first verdict — org stamp included
    assert second == {"label": "full_refusal"}
    assert cache.get("k") == {"label": "full_refusal"}
    assert cache.get_org("k") == "lowprio"
    assert len(cache) == 1
    rows = cache.conflicts()
    assert len(rows) == 1
    (row,) = rows
    assert row["key"] == "k"
    assert row["stored"] == {"label": "full_refusal"}
    assert row["incoming"] == {"label": "partial_refusal"}
    assert row["org_used"] == "highprio"  # who brought the disagreeing verdict
    assert cache.conflicts("k") == rows and cache.conflicts("other") == []


def test_an_agreeing_repeat_put_is_a_no_op_that_keeps_the_first_stamp(tmp_path):
    cache = JudgeCache(tmp_path / "c.sqlite")
    cache.put("k", {"label": "full_refusal"}, "lowprio")
    assert cache.put("k", {"label": "full_refusal"}, "highprio") == {"label": "full_refusal"}
    assert cache.get_org("k") == "lowprio"  # the producer stamp is the FIRST writer's
    assert cache.conflicts() == []


def test_sentinel_metadata_differences_are_not_conflicts(tmp_path):
    """Two abstentions with the same reason agree even when their evidence
    fields differ (attempts, siblings, request id): a conflict is a
    disagreement about the VERDICT, not about the bookkeeping."""
    cache = JudgeCache(tmp_path / "c.sqlite")
    a = {"abstained": True, "reason": "transport_refusal", "attempts": 5,
         "siblings_succeeded": 3, "evidence": "siblings"}
    b = {"abstained": True, "reason": "transport_refusal", "attempts": 4,
         "siblings_succeeded": 0, "evidence": "canary", "request_id": "req_x"}
    assert cache.put("k", a, "lowprio") == a
    assert cache.put("k", b, "lowprio") == a
    assert cache.conflicts() == []
    # but a different REASON, or a label against an abstention, is one
    c = {"abstained": True, "reason": "refusal"}
    assert cache.put("k", c, "lowprio") == a
    assert cache.put("k", {"label": "full_refusal"}, "lowprio") == a
    assert len(cache.conflicts()) == 2


def test_verdicts_agree_is_the_disagreement_rule():
    assert j.verdicts_agree({"label": "a"}, {"label": "a"})
    assert not j.verdicts_agree({"label": "a"}, {"label": "b"})
    assert j.verdicts_agree({"abstained": True, "reason": "refusal", "attempts": 1},
                            {"abstained": True, "reason": "refusal"})
    assert not j.verdicts_agree({"abstained": True, "reason": "refusal"},
                                {"abstained": True, "reason": "max_tokens"})
    assert not j.verdicts_agree({"abstained": True, "reason": "refusal"}, {"label": "a"})
    # a pairwise verdict is compared whole (every label field)
    assert not j.verdicts_agree({"verdict": "a", "conf": 1}, {"verdict": "a", "conf": 2})


def test_explicit_reattempt_replaces_a_stored_abstention_but_never_a_label(tmp_path):
    cache = JudgeCache(tmp_path / "c.sqlite")
    cache.put("k", {"abstained": True, "reason": "refusal"}, "lowprio")
    out = cache.put("k", {"label": "full_compliance"}, "highprio", replace_abstention=True)
    assert out == {"label": "full_compliance"} == cache.get("k")
    assert cache.get_org("k") == "highprio"  # a replacement is a new producer
    assert cache.conflicts() == []  # deliberate, not a disagreement
    # a stored LABEL is write-once even under the explicit flag
    out2 = cache.put("k", {"label": "full_refusal"}, "lowprio", replace_abstention=True)
    assert out2 == {"label": "full_compliance"} == cache.get("k")
    assert len(cache.conflicts()) == 1


def test_put_still_refuses_free_text_before_touching_the_store(tmp_path):
    cache = JudgeCache(tmp_path / "c.sqlite")
    with pytest.raises(JudgingError, match="JB4"):
        cache.put("k", {"reasoning": "the model said something long"}, "lowprio")
    assert cache.get("k") is None and cache.conflicts() == []


def test_conflicts_table_is_created_on_an_existing_cache_file(tmp_path):
    """An older cache (verdicts table only) opens, gains the conflicts table,
    and its verdict rows are untouched."""
    import sqlite3
    p = tmp_path / "old.sqlite"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE verdicts (key TEXT PRIMARY KEY, verdict_json TEXT NOT NULL, "
                "org_used TEXT NOT NULL, created_at REAL NOT NULL)")
    con.execute("INSERT INTO verdicts VALUES (?,?,?,?)", ("k", '{"label":"full_refusal"}', "lowprio", 1.0))
    con.commit()
    con.close()
    cache = JudgeCache(p)
    assert cache.get("k") == {"label": "full_refusal"} and cache.conflicts() == []
    assert cache.put("k", {"label": "partial_refusal"}, "lowprio") == {"label": "full_refusal"}
    assert len(cache.conflicts()) == 1 and cache.get_org("k") == "lowprio"


# --- judge_all adopts what stands, and counts the disagreement ---------------------


def test_judge_all_adopts_the_stored_verdict_on_a_raced_key_and_counts_it(tmp_path):
    """The race, reproduced in one process: while this judge's live call for
    a key is in flight, another writer (a peer leg, a peer process) files a
    DIFFERENT label for the same key. The result and the cache must agree
    (the post-filing read-back compares them), the disagreement is counted
    on the judge and recorded in the cache, and nothing is overwritten."""
    judge_ref = {}
    req = _single_req("synthetic shared prompt", "I can't help with that.")

    def low(_params):
        # the peer wins the key first, exactly as a parallel leg or a second
        # selection on the same sqlite would
        judge_ref["j"].cache.put(req.key(), {"label": "full_refusal"}, "lowprio")
        return _cpi_msg("partial_refusal")

    judge = _judge(tmp_path, low)
    judge_ref["j"] = judge
    out = judge.judge_all([(req, ORBENCH_RUBRIC_V1)])
    assert out[req.key()] == {"label": "full_refusal"}  # what stands, not what we got
    assert judge.cache.get(req.key()) == {"label": "full_refusal"}
    assert judge.n_verdict_conflicts == 1
    assert judge.n_api_calls == 1  # the call happened and is counted
    rows = judge.cache.conflicts(req.key())
    assert len(rows) == 1
    assert rows[0]["stored"] == {"label": "full_refusal"}
    assert rows[0]["incoming"] == {"label": "partial_refusal"}


def test_two_legs_sharing_a_key_file_the_same_verdict(tmp_path):
    """Sequential legs already agreed (the second is a cache hit); with the
    write-once put, a second judge_all that somehow live-calls the key
    (its miss list was built before the first leg wrote) still files what
    the first leg filed."""
    labels = iter(["full_refusal", "partial_refusal"])
    judge = _judge(tmp_path, lambda _p: _cpi_msg(next(labels)))
    req = _single_req("synthetic shared prompt", "canned refusal text")
    a = judge.judge_all([(req, ORBENCH_RUBRIC_V1)])
    b = judge.judge_all([(req, ORBENCH_RUBRIC_V1)])  # cache hit: no live call
    assert a == b and judge.n_api_calls == 1 and judge.n_cache_hits == 1
    assert judge.n_verdict_conflicts == 0


def test_raced_abstention_over_a_live_label_is_adopted_and_counted(tmp_path):
    """A peer filed an abstention sentinel for the key; our live label
    disagrees. The sentinel stands (conservative: it widens the bound rather
    than picking a label), the run counts the abstention it files, and the
    disagreement is recorded — declined-once / answered-once is a stability
    signal too."""
    judge_ref = {}
    req = _single_req("synthetic shared prompt", "an answer")

    def low(_params):
        judge_ref["j"].cache.put(req.key(), {"abstained": True, "reason": "refusal"}, "lowprio")
        return _cpi_msg("full_compliance")

    judge = _judge(tmp_path, low)
    judge_ref["j"] = judge
    out = judge.judge_all([(req, ORBENCH_RUBRIC_V1)], allow_refusal_abstention=True)
    assert out[req.key()] == {"abstained": True, "reason": "refusal"}
    assert judge.n_abstained == 1 and judge.n_verdict_conflicts == 1


def test_reattempt_abstentions_replaces_the_cached_sentinel_without_a_conflict(tmp_path):
    judge = _judge(tmp_path, lambda _p: _cpi_msg("full_compliance"))
    req = _single_req("synthetic prompt", "an answer")
    judge.cache.put(req.key(), {"abstained": True, "reason": "refusal"}, "lowprio")
    out = judge.judge_all([(req, ORBENCH_RUBRIC_V1)], allow_refusal_abstention=True,
                          reattempt_abstentions=True)
    assert out[req.key()] == {"label": "full_compliance"} == judge.cache.get(req.key())
    assert judge.n_verdict_conflicts == 0 and judge.cache.conflicts() == []
    assert judge.n_api_calls == 1 and judge.n_abstained == 0


def test_conflict_is_announced_with_labels_only(tmp_path, capsys):
    judge_ref = {}
    req = _single_req("synthetic prompt text that must not print", "response text that must not print")

    def low(_params):
        judge_ref["j"].cache.put(req.key(), {"label": "full_refusal"}, "lowprio")
        return _cpi_msg("full_compliance")

    judge = _judge(tmp_path, low)
    judge_ref["j"] = judge
    judge.judge_all([(req, ORBENCH_RUBRIC_V1)])
    out = capsys.readouterr().out
    assert "verdict conflict" in out and "full_refusal" in out and "full_compliance" in out
    assert "must not print" not in out


# --- the seam with the scoped re-attempt: a re-filed sentinel must AGE ------------------------------


def test_refiled_transport_sentinel_replaces_the_prior_one_and_its_created_at_advances(tmp_path, monkeypatch):
    """The scoped re-attempt x write-once. A ``transport_refusal`` sentinel the scoped predicate
    approved is re-asked; when the re-ask 529s AGAIN, ``_rule_on_transport_candidates`` files a NEW
    transport sentinel for the SAME key with ``replace_abstention=True`` — an abstention replacing
    an abstention. That put must REPLACE the row and its ``created_at`` must ADVANCE to the re-file
    time. Not "stored" alone: ``abstention_reattempt_predicate(reasons, after)`` re-asks a sentinel
    filed strictly AFTER ``after``, so a row whose stamp never moves can never age out of the
    window — it is re-asked on EVERY later ``--reattempt-after`` pass, each pass costing judge
    calls and failing the same way, and nothing surfaces an error. Dropping the incoming as
    "agreement" (same reason) or recording it as a conflict both leave the old stamp in place and
    both fail this test."""
    T1, T2 = 1_789_020_010.0, 1_789_023_610.0              # storm + 10 s; one hour later
    cache = JudgeCache(tmp_path / "c.sqlite")
    old = {"abstained": True, "reason": "transport_refusal", "attempts": 6,
           "siblings_succeeded": 3, "evidence": "siblings"}
    monkeypatch.setattr(j.time, "time", lambda: T1)
    assert cache.put("k", old, "lowprio") == old
    assert cache.get_row("k") == (old, "lowprio", T1)
    approve = j.abstention_reattempt_predicate(["transport_refusal"], T1 - 1)
    assert approve(old, T1)                                  # the predicate approves the re-ask
    # the re-ask 529s again: the producer's re-filed sentinel (same reason, fresh bookkeeping)
    new = {"abstained": True, "reason": "transport_refusal", "attempts": 6,
           "siblings_succeeded": 1, "evidence": "siblings", "request_id": "req-refile-2"}
    monkeypatch.setattr(j.time, "time", lambda: T2)
    stood = cache.put("k", new, "lowprio", replace_abstention=True)
    # what stands is the re-filed sentinel, made to SAY it was retried: attempts accumulate (6 prior
    # + 6 now), the retry's request_id rides along, reattempted_from is the prior stamp
    expect = dict(new, attempts=old["attempts"] + new["attempts"], reattempted_from=T1)
    assert stood == expect, f"the re-filed sentinel must be what stands; got: {stood}"
    row = cache.get_row("k")
    assert row is not None and row[0] == expect, f"row not replaced: {row}"
    assert row[0]["attempts"] == 12 and row[0]["request_id"] == "req-refile-2" and row[0]["reattempted_from"] == T1
    assert j.verdicts_agree(row[0], new)                     # bookkeeping is not verdict: _adopt stays quiet
    assert row[2] == T2, f"created_at must advance to the re-file time T2={T2}; it is {row[2]} (T1={T1})"
    assert cache.conflicts("k") == [], "a re-file under replace_abstention is a replace, not a conflict"
    # the aged stamp is what lets the key LEAVE a later window
    assert not j.abstention_reattempt_predicate(["transport_refusal"], T2)(row[0], row[2])


def test_approved_reattempt_whose_retry_yields_a_label_replaces_the_sentinel_storm_recovery(tmp_path, monkeypatch):
    """Case 2 of 3, adjacent to the re-file case above: the approved re-ask succeeds. The live
    LABEL replaces the sentinel (verdict, org, stamp), no conflict is recorded, and a label is
    never re-asked again whatever the window."""
    T1, T2 = 1_789_020_010.0, 1_789_023_610.0
    cache = JudgeCache(tmp_path / "c.sqlite")
    sentinel = {"abstained": True, "reason": "transport_refusal", "attempts": 6,
                "siblings_succeeded": 2, "evidence": "siblings"}
    monkeypatch.setattr(j.time, "time", lambda: T1)
    assert cache.put("k", sentinel, "lowprio") == sentinel
    assert j.abstention_reattempt_predicate(["transport_refusal"], T1 - 1)(sentinel, T1)
    monkeypatch.setattr(j.time, "time", lambda: T2)
    label = {"label": "full_compliance"}
    assert cache.put("k", label, "highprio", replace_abstention=True) == label
    assert cache.get_row("k") == (label, "highprio", T2)          # a replacement is a new producer
    assert cache.conflicts("k") == []
    assert not j.abstention_reattempt_predicate(["transport_refusal"], T1 - 1)(label, T2)


def test_a_stored_label_refuses_everything_through_put_flag_or_no_flag(tmp_path, monkeypatch):
    """Case 3 of 3: write-once's actual purpose, the bound the re-attempt must not erode. A stored
    LABEL is never replaced through ``put`` — not by another label, not by a sentinel, with or
    without ``replace_abstention`` — the row, its org stamp and its timestamp stay exactly as first
    filed, and every disagreement is a recorded conflict. An agreeing repeat of the label is the
    no-op it always was."""
    T1, T2 = 1_789_020_010.0, 1_789_023_610.0
    cache = JudgeCache(tmp_path / "c.sqlite")
    label = {"label": "full_compliance"}
    monkeypatch.setattr(j.time, "time", lambda: T1)
    assert cache.put("k", label, "lowprio") == label
    monkeypatch.setattr(j.time, "time", lambda: T2)
    other = {"label": "full_refusal"}
    sentinel = {"abstained": True, "reason": "transport_refusal", "attempts": 6}
    for incoming, flag in ((other, True), (other, False), (sentinel, True), (sentinel, False)):
        assert cache.put("k", incoming, "highprio", replace_abstention=flag) == label, (incoming, flag)
        assert cache.get_row("k") == (label, "lowprio", T1), (incoming, flag)
    assert len(cache.conflicts("k")) == 4
    assert [c["incoming"] for c in cache.conflicts("k")] == [other, other, sentinel, sentinel]
    assert cache.put("k", dict(label), "highprio", replace_abstention=True) == label
    assert cache.get_row("k") == (label, "lowprio", T1) and len(cache.conflicts("k")) == 4
