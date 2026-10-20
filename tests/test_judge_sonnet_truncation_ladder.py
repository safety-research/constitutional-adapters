"""judge_sonnet.Judge._call reads a reply's stop_reason BEFORE validating its verdict.

The 2026-10-03 control-LoRA leaking item (s1 agentic batch) errored twice as a "truncation" although its reply was 321 output tokens of a
1500 cap: the SDK's messages.parse validates every text block's JSON client-side without reading stop_reason, so a verdict the judge
STOPPED mid-string (stop_reason refusal, category bio) surfaced as pydantic json_invalid before the refusal / max_tokens distinction could
be made. Now the request carries the structured format under output_config (the SDK hands the Message back unparsed), classify_verdict
reads stop_reason first, and:
  * a refusal stop (whatever the text), a reply without a text block, or an unparsable text under any stop that is neither end_turn nor
    max_tokens is a BLOCKED abstention naming the stop -- never an error, never retried;
  * an unparsable text under a max_tokens stop climbs the ladder 1500 -> 4000 -> 8000 (each rung capped at the model's maximum output);
    a verdict still cut at the last rung stays the item's error with its reason (pydantic.ValidationError), as before;
  * complete JSON that fails the schema, or a cut text under end_turn, is the model's own malformed verdict: the item's error, not retried.
Synthetic ids and text throughout; no API call (the client is a fake).
"""
from __future__ import annotations

import json
import pathlib
import sys
from types import SimpleNamespace

import pydantic
import pytest
from pydantic import BaseModel

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import judge_sonnet as js  # noqa: E402

CUT = '{"reasoning": "the reply forwards the '          # a verdict cut mid-string: the shape every truncation / refusal stop raised
FULL = '{"reasoning": "fine", "harmful": false}'


class _Verdict(BaseModel):
    reasoning: str
    harmful: bool


def _reply(text, stop="end_turn", category=None, content=None):
    r = SimpleNamespace(stop_reason=stop, content=[SimpleNamespace(type="text", text=text)] if content is None else content,
                        stop_details=None, usage=SimpleNamespace(output_tokens=len(text) // 4 if text else 0))
    if stop == "refusal":
        r.stop_details = SimpleNamespace(type="refusal", category=category, explanation="synthetic")
    return r


class _Messages:
    def __init__(self, owner):
        self._o = owner

    def parse(self, **kw):
        self._o.calls.append(dict(kw))
        step = self._o.script.pop(0) if self._o.script else _reply(FULL)
        if isinstance(step, Exception):
            raise step
        return step


class _Client:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    @property
    def messages(self):
        return _Messages(self)


@pytest.fixture
def judge(tmp_path, monkeypatch):
    # the Judge resolves the Low-Prio org's key at construction (its OWN env var, never a file); the placeholder keeps the test off the
    # vault and the fake client below answers every call
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY_LOWPRIO", "placeholder-not-a-key")
    monkeypatch.setattr(js, "_RETRY_SLEEP_S", 0.0)
    monkeypatch.setattr(js, "_RETRY_JITTER_S", 0.0)
    return js.Judge(tmp_path / "cache.jsonl", 2, thinking=False)


# --------------------------------------------------------------------------------------------
# the ladder as a pure function
# --------------------------------------------------------------------------------------------


def test_the_ladder_is_1500_4000_8000_for_the_judge_model_and_an_unknown_model(monkeypatch):
    assert js.verdict_token_ladder("claude-sonnet-5") == (1500, 4000, 8000)
    assert js.verdict_token_ladder(js.JUDGE_MODEL) == (js.VERDICT_MAX_TOKENS, js.TRUNCATION_RETRY_MAX_TOKENS, js.TRUNCATION_THIRD_RUNG_MAX_TOKENS)
    assert js.verdict_token_ladder("synthetic-model-not-in-the-table") == (1500, 4000, 8000)
    assert js.MODEL_MAX_OUTPUT_TOKENS["claude-sonnet-5"] == 128_000       # the Models API's max_tokens for the Sonnet 5 line


def test_the_models_maximum_output_caps_every_rung_and_collapses_equal_rungs(monkeypatch):
    monkeypatch.setitem(js.MODEL_MAX_OUTPUT_TOKENS, "synthetic-m", 6000)
    assert js.verdict_token_ladder("synthetic-m") == (1500, 4000, 6000)    # the third rung is the model's maximum, lower than 8000
    monkeypatch.setitem(js.MODEL_MAX_OUTPUT_TOKENS, "synthetic-m", 3000)
    assert js.verdict_token_ladder("synthetic-m") == (1500, 3000)          # the cap sits between the rungs: two rungs, strictly increasing
    monkeypatch.setitem(js.MODEL_MAX_OUTPUT_TOKENS, "synthetic-m", 1000)
    assert js.verdict_token_ladder("synthetic-m") == (1000,)               # below the first rung: one request, at the model's maximum


# --------------------------------------------------------------------------------------------
# a max_tokens stop: the ladder
# --------------------------------------------------------------------------------------------


def test_a_max_tokens_stop_climbs_the_ladder_and_succeeds_at_the_third_rung(judge):
    judge.client = _Client([_reply(CUT, "max_tokens"), _reply(CUT, "max_tokens"), _reply(FULL)])
    out = judge._call("prompt", _Verdict)
    assert out == {"reasoning": "fine", "harmful": False}
    assert [c["max_tokens"] for c in judge.client.calls] == [1500, 4000, 8000]
    assert (judge.n_truncated_retried, judge.n_truncated_errored, judge.n_blocked_by_stop) == (2, 0, 0)
    # the request carries the structured format under output_config (the SDK parses nothing client-side any more)
    first = judge.client.calls[0]
    assert "output_format" not in first and first["output_config"]["format"]["type"] == "json_schema"
    assert set(first["output_config"]["format"]["schema"]["required"]) == {"reasoning", "harmful"}
    assert first["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert first["model"] == js.JUDGE_MODEL and first["thinking"] == {"type": "disabled"}


def test_a_verdict_cut_at_every_rung_stays_the_items_error_with_its_reason(judge):
    judge.client = _Client([_reply(CUT, "max_tokens")] * 3)
    with pytest.raises(pydantic.ValidationError) as ei:
        judge._call("prompt", _Verdict)
    assert "EOF while parsing" in str(ei.value)                                   # the reason, as before
    assert [c["max_tokens"] for c in judge.client.calls] == [1500, 4000, 8000]
    assert (judge.n_truncated_retried, judge.n_truncated_errored, judge.n_blocked_by_stop) == (2, 1, 0)


def test_the_models_maximum_bounds_the_ladder_in_the_call(judge, monkeypatch):
    monkeypatch.setitem(js.MODEL_MAX_OUTPUT_TOKENS, js.JUDGE_MODEL, 3000)
    judge.client = _Client([_reply(CUT, "max_tokens")] * 3)
    with pytest.raises(pydantic.ValidationError):
        judge._call("prompt", _Verdict)
    assert [c["max_tokens"] for c in judge.client.calls] == [1500, 3000]          # never a request above the model's maximum
    assert (judge.n_truncated_retried, judge.n_truncated_errored) == (1, 1)


# --------------------------------------------------------------------------------------------
# a stop that is not max_tokens: an abstention, never an error, never retried
# --------------------------------------------------------------------------------------------


def test_a_refusal_stop_is_a_blocked_abstention_naming_the_category_never_an_error(judge):
    judge.client = _Client([_reply(CUT, "refusal", category="bio"), _reply(FULL)])
    out = judge._call("prompt", _Verdict)
    assert out == {"__blocked__": True, "blocked_reason": "refusal:bio"}
    assert len(judge.client.calls) == 1                                           # not retried at a larger cap
    assert (judge.n_truncated_retried, judge.n_truncated_errored, judge.n_blocked_by_stop) == (0, 0, 1)
    # the stop is read FIRST: a refusal stop with a complete, valid text is still the judge declining
    judge.client = _Client([_reply(FULL, "refusal", category=None)])
    assert judge._call("prompt", _Verdict) == {"__blocked__": True, "blocked_reason": "refusal"}


def test_an_unparsable_text_under_another_non_end_turn_stop_is_blocked_with_the_stop_named(judge):
    judge.client = _Client([_reply(CUT, "stop_sequence"), _reply(FULL)])
    assert judge._call("prompt", _Verdict) == {"__blocked__": True, "blocked_reason": "stop:stop_sequence"}
    assert len(judge.client.calls) == 1 and judge.n_blocked_by_stop == 1 and judge.n_truncated_retried == 0


def test_a_reply_without_a_text_block_is_blocked(judge):
    judge.client = _Client([_reply("", content=[]), _reply(FULL)])
    assert judge._call("prompt", _Verdict) == {"__blocked__": True, "blocked_reason": "no_text:end_turn"}
    assert len(judge.client.calls) == 1


def test_a_malformed_verdict_at_end_turn_is_the_items_error_not_retried_not_blocked(judge):
    judge.client = _Client([_reply('{"other": 1}'), _reply(FULL)])                # complete JSON, wrong shape
    with pytest.raises(pydantic.ValidationError):
        judge._call("prompt", _Verdict)
    assert len(judge.client.calls) == 1
    judge.client = _Client([_reply(CUT, "end_turn"), _reply(FULL)])               # a cut text the model itself ended
    with pytest.raises(pydantic.ValidationError):
        judge._call("prompt", _Verdict)
    assert len(judge.client.calls) == 1
    assert (judge.n_truncated_retried, judge.n_truncated_errored, judge.n_blocked_by_stop) == (0, 0, 0)


def test_classify_verdict_is_the_pure_rule():
    assert js.classify_verdict(_reply(FULL), _Verdict) == ("verdict", {"reasoning": "fine", "harmful": False})
    assert js.classify_verdict(_reply(CUT, "refusal", category="cyber"), _Verdict) == ("blocked", "refusal:cyber")
    assert js.classify_verdict(_reply(CUT, "pause_turn"), _Verdict) == ("blocked", "stop:pause_turn")
    kind, err = js.classify_verdict(_reply(CUT, "max_tokens"), _Verdict)
    assert kind == "truncated" and isinstance(err, pydantic.ValidationError) and js._is_truncation(err)
    kind, err = js.classify_verdict(_reply(CUT, "end_turn"), _Verdict)
    assert kind == "malformed" and isinstance(err, pydantic.ValidationError)
    kind, err = js.classify_verdict(_reply('{"reasoning": "x"}', "max_tokens"), _Verdict)   # complete JSON, a field missing: malformed, whatever the stop
    assert kind == "malformed"
    assert js.verdict_text(_reply("a", content=[SimpleNamespace(type="text", text="a"), SimpleNamespace(type="thinking", thinking="t"),
                                                 SimpleNamespace(type="text", text="b")])) == "ab"
    assert js.verdict_text(_reply("", content=[])) is None


# --------------------------------------------------------------------------------------------
# the cache keeps the abstention and its reason; a rerun never re-asks
# --------------------------------------------------------------------------------------------


def test_a_blocked_verdict_persists_in_the_cache_with_its_reason_and_replays(judge, tmp_path):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    import judge_rubrics as jr
    judge.client = _Client([_reply(CUT, "refusal", category="bio")])
    out = judge.structured(jr.AM_LEAK_TAG, ("synthetic", "leaking", "item-1", "reply"), "prompt", jr.HarmfulVerdict)
    assert out == {"__blocked__": True, "blocked_reason": "refusal:bio"} and len(judge.client.calls) == 1
    on_disk = [json.loads(l) for l in (tmp_path / "cache.jsonl").read_text().splitlines() if l.strip()]
    assert len(on_disk) == 1 and on_disk[0]["value"] == {"__blocked__": True, "blocked_reason": "refusal:bio"}
    judge.client = _Client([_reply(FULL)])
    assert judge.structured(jr.AM_LEAK_TAG, ("synthetic", "leaking", "item-1", "reply"), "prompt", jr.HarmfulVerdict) == out
    assert judge.client.calls == []                                               # replayed from the cache
