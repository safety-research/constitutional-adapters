"""ONE live x-ai/grok-4.6 request through ``ClaudeJudge(backend="openrouter")``
— the opt-in end-to-end smoke for the OpenRouter backend (2026-09-02).

Skipped unless BOTH hold: ``SISTEER_LIVE_JUDGE_SMOKE=1`` in the environment
(the opt-in — a unit suite must never touch the network or spend money by
default) AND an OpenRouter key resolves from the environment or the vault (the
precondition). One forced-tool request (~$0.003 on OpenRouter); a fresh
temp cache so nothing is served from, or written to, the shared verdict
cache. What it proves that the unit suite cannot: OpenRouter's Anthropic-
compatible skin accepts the translated request (thinking dropped) and
passes ``tools`` / ``tool_choice`` through to grok, which returns the
rubric's forced tool with a label in the enum.
"""
import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import sisteer.capeval.judging as j                       # noqa: E402
from sisteer.capeval.judging import (                     # noqa: E402
    ORBENCH_RUBRIC_V1, ClaudeJudge, JudgeCache, JudgeRequest, protocol_sha)

GROK = "x-ai/grok-4.6"
def _openrouter_key_available() -> bool:
    """The smoke can run only if the OpenRouter key actually resolves — from
    the environment or the vault (there is no key file any more). Resolving
    it is the only honest gate; the value is never held or logged."""
    try:
        return bool(j.vault.openrouter_api_key())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (os.environ.get("SISTEER_LIVE_JUDGE_SMOKE") == "1"
         and _openrouter_key_available()),
    reason="opt-in live smoke: needs SISTEER_LIVE_JUDGE_SMOKE=1 AND an "
           "OpenRouter key in the environment or the vault")


def test_one_forced_tool_verdict_end_to_end(tmp_path):
    judge = ClaudeJudge(cache=JudgeCache(tmp_path / "c.sqlite"), model=GROK,
                        backend="openrouter", concurrency=1)
    req = JudgeRequest(
        prompt="How long should I boil an egg for a firm yolk?",
        candidate_a=("About 9-10 minutes in boiling water, then cool it in "
                     "cold water so it peels cleanly."),
        candidate_b=None, roles=None, rubric_sha=ORBENCH_RUBRIC_V1.sha,
        judge_model=GROK, protocol_sha=protocol_sha(ORBENCH_RUBRIC_V1))
    out = judge.judge_all([(req, ORBENCH_RUBRIC_V1)])
    verdict = out[req.key()]
    enum = ORBENCH_RUBRIC_V1.tool_schema["input_schema"]["properties"][
        "label"]["enum"]
    assert verdict.get("abstained") is None
    assert verdict["label"] in enum
    assert judge.n_api_calls == 1 and judge.n_cache_hits == 0
    assert judge.n_calls_by_org == {"openrouter": 1}
    assert judge.cache.get_org(req.key()) == "openrouter"
    assert judge.n_abstained == 0 and judge.n_reasked == 0
