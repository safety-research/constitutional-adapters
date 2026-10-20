"""A SCOPED re-attempt of cached abstention sentinels (2026-09-10).

Abstention sentinels are cached so re-reads report stable bounds, and ``judge_all(reattempt_abstentions=True)``
re-asks every one of them. Neither fits the Low Prio sonnet-5 529 storm of 2026-09-10: hundreds of items per
cell were filed as ``transport_refusal`` (the ruling's precondition "the org was serving" was met only by chance
siblings), while the pre-existing ``transport_refusal`` sentinels seeded from the greedy campaign, and every
``refusal`` / ``max_tokens`` / ``content_filter`` sentinel, are exactly the stable abstentions the protocol wants
kept. So ``reattempt_abstentions`` may be a PREDICATE ``(verdict, created_at) -> bool`` — built by
:func:`abstention_reattempt_predicate` from a reason set and a filed-after instant — and the judge-level attribute
of the same name is the default the tier's unchanged ``judge_all(pairs, allow_refusal_abstention=...)`` call reads.
The served-controls judge exposes both as ``--reattempt-abstentions REASONS`` / ``--reattempt-after ISO``.
"""
from __future__ import annotations

import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
pytest.importorskip("sisteer.capeval.judging")
from sisteer.capeval.manifest import canonical_json  # noqa: E402
from sisteer.capeval.judging import (  # noqa: E402
    ORBENCH_RUBRIC_V1, ClaudeJudge, JudgeCache, JudgeRequest, abstention_reattempt_predicate, protocol_sha)
from test_capeval_judging import _Client, _Msg, _ToolUse  # noqa: E402

MODEL = "claude-sonnet-5"
CUTOFF = 1_789_020_000.0          # 2026-09-10T06:00:00Z, the storm's start in the incident


def _req(item: str, rubric=ORBENCH_RUBRIC_V1) -> JudgeRequest:
    return JudgeRequest(prompt=f"synthetic prompt {item}", candidate_a=f"synthetic answer {item}", candidate_b=None,
                        roles=None, rubric_sha=rubric.sha, judge_model=MODEL, protocol_sha=protocol_sha(rubric))


def _seed(cache: JudgeCache, key: str, verdict: dict, created_at: float) -> None:
    """Plant ``verdict`` for ``key`` as if filed at ``created_at`` — straight into the sqlite,
    deliberately NOT through ``JudgeCache.put``. The cache is WRITE-ONCE (9c011fad): a put never
    replaces a stored row, and a stored LABEL is never replaced by anything. That is the
    production property under test elsewhere, not a limitation to route around — a test-only
    "replace" on the public surface would be a hole with a comment on it. The last test in this
    file plants a refusal sentinel over a key that already holds a label (the planted row IS the
    precondition it then reads back), so the fixture writes the row by the one legitimate means,
    the store itself — the same adaptation 9c011fad made to test_capeval_judging.py and
    test_judge_transport_refusal.py. Do not "fix" this back to ``put``: through ``put`` the second
    seed is dropped as a recorded conflict and the test reads the label instead."""
    with cache._lock:
        cache._conn.execute(
            "INSERT OR REPLACE INTO verdicts (key, verdict_json, org_used, created_at) VALUES (?,?,?,?)",
            (key, canonical_json(dict(verdict)), "lowprio", created_at))
        cache._conn.commit()


def _judge(tmp_path, calls):
    def client(params):
        calls.append(params)
        return _Msg([_ToolUse("compliance_rating", {"reasoning": "synthetic", "label": "full_compliance"})])
    cache = JudgeCache(tmp_path / "c.sqlite")
    return cache, ClaudeJudge(cache=cache, model=MODEL, concurrency=2, clients={"lowprio": client and _Client(client)})


def test_predicate_scopes_by_reason_and_filed_after():
    pred = abstention_reattempt_predicate(["transport_refusal"], CUTOFF)
    storm = {"abstained": True, "reason": "transport_refusal", "evidence": "siblings"}
    assert pred(storm, CUTOFF + 1)                              # filed during the storm: re-ask
    assert not pred(storm, CUTOFF - 1)                          # seeded from before: keep
    assert not pred(storm, None)                                # no filing time under a time bound: keep
    assert not pred({"abstained": True, "reason": "refusal"}, CUTOFF + 1)   # another reason: keep
    assert not pred({"label": "full_compliance"}, CUTOFF + 1)   # a verdict is never re-asked
    assert abstention_reattempt_predicate(["transport_refusal"])(storm, None)      # no time bound: reason alone decides
    assert not abstention_reattempt_predicate([])(storm, CUTOFF + 1)                # empty reasons: never


def test_judge_all_reasks_exactly_the_predicated_sentinels_and_keeps_the_rest(tmp_path):
    calls = []
    cache, judge = _judge(tmp_path, calls)
    rubric = ORBENCH_RUBRIC_V1
    reqs = {n: _req(n) for n in ("storm", "seeded", "refusal", "verdict")}
    storm = {"abstained": True, "reason": "transport_refusal", "evidence": "siblings", "attempts": 6}
    _seed(cache, reqs["storm"].key(), storm, CUTOFF + 3600)
    _seed(cache, reqs["seeded"].key(), storm, CUTOFF - 86400)
    _seed(cache, reqs["refusal"].key(), {"abstained": True, "reason": "refusal"}, CUTOFF + 3600)
    _seed(cache, reqs["verdict"].key(), {"label": "partial_compliance"}, CUTOFF + 3600)
    pairs = [(r, rubric) for r in reqs.values()]

    # the default: every cached row served, nothing re-asked (stable bounds)
    out = judge.judge_all(pairs, allow_refusal_abstention=True)
    assert calls == [] and out[reqs["storm"].key()] == storm

    # the scoped predicate: ONE live call, for the storm sentinel only; its cache row becomes the verdict
    pred = abstention_reattempt_predicate(["transport_refusal"], CUTOFF)
    out = judge.judge_all(pairs, allow_refusal_abstention=True, reattempt_abstentions=pred)
    assert len(calls) == 1
    assert out[reqs["storm"].key()] == {"label": "full_compliance"}
    assert out[reqs["seeded"].key()] == storm                       # pre-storm sentinel of the same reason: kept
    assert out[reqs["refusal"].key()] == {"abstained": True, "reason": "refusal"}
    assert out[reqs["verdict"].key()] == {"label": "partial_compliance"}
    assert cache.get(reqs["storm"].key()) == {"label": "full_compliance"}
    assert cache.get(reqs["seeded"].key()) == storm
    row = cache.get_row(reqs["storm"].key())
    assert row is not None and row[1] == "lowprio" and row[2] > CUTOFF + 3600 - 1 and row[2] <= time.time() + 1

    # a second read is stable again — the re-ask replaced the sentinel, nothing is re-billed
    out2 = judge.judge_all(pairs, allow_refusal_abstention=True, reattempt_abstentions=pred)
    assert len(calls) == 1 and out2 == out


def test_judge_level_attribute_is_the_default_the_tier_call_reads(tmp_path):
    calls = []
    cache, judge = _judge(tmp_path, calls)
    req = _req("storm")
    _seed(cache, req.key(), {"abstained": True, "reason": "transport_refusal"}, CUTOFF + 10)
    pairs = [(req, ORBENCH_RUBRIC_V1)]
    judge.judge_all(pairs, allow_refusal_abstention=True)            # the tier's exact call: keyword only
    assert calls == []
    judge.reattempt_abstentions = abstention_reattempt_predicate(["transport_refusal"], CUTOFF)
    out = judge.judge_all(pairs, allow_refusal_abstention=True)      # same call, now re-asks via the attribute
    assert len(calls) == 1 and out[req.key()] == {"label": "full_compliance"}
    # the explicit boolean still means "every cached abstention"
    _seed(cache, req.key(), {"abstained": True, "reason": "refusal"}, CUTOFF - 10)
    judge.reattempt_abstentions = False
    assert judge.judge_all(pairs, allow_refusal_abstention=True)[req.key()]["reason"] == "refusal" and len(calls) == 1
    judge.judge_all(pairs, allow_refusal_abstention=True, reattempt_abstentions=True)
    assert len(calls) == 2


@pytest.mark.parametrize("argv,needle", [
    (["--reattempt-abstentions", "bogus"], "unknown reason"),
    (["--reattempt-after", "2026-09-10T06:00:00Z"], "needs --reattempt-abstentions"),
])
def test_served_judge_cli_refuses_a_bad_reattempt_policy(tmp_path, argv, needle):
    rc = subprocess.run([sys.executable, str(REPO / "scripts/served_capability_judge.py"), "--controls-root", str(tmp_path),
                         "--dry-run", *argv], capture_output=True, text=True,
                        env={"PYTHONPATH": f"{REPO}/src:{REPO}/scripts", "HF_HUB_OFFLINE": "1", "PATH": "/usr/bin:/bin"})
    assert rc.returncode != 0 and needle in (rc.stderr + rc.stdout)
