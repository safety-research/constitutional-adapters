"""judge_sonnet.Judge._call: a content-scoped 529 is a blocked abstention, not an hour-long ladder.

A judged run (2026-09-11) sat for ~12 min per body on bodies that drew a deterministic 529
in 8-12 s each while every sibling succeeded (the transport_refusal class already handled by
capeval.judging). judge_sonnet's ladder waited up to 1 h per body. Now: PER_BODY_529_CAP
consecutive 529s on one body WHILE the injected client's success counter advanced -> {"__blocked__"}.
A pool-wide 529 (no sibling success) keeps retrying; a 429 in the ladder resets the count."""
import pathlib
import sys
from types import SimpleNamespace

import httpx
import pytest
import anthropic
import pydantic
from pydantic import BaseModel

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import judge_sonnet as js  # noqa: E402


class _Out(BaseModel):
    harmful: bool


class _Resp:
    """The reply the fakes return: an end_turn stop with a schema-valid text block (_call validates the text itself; the SDK
    parses nothing client-side any more, so a fake carries `content`, never `parsed_output`)."""
    stop_details = None

    def __init__(self, text='{"harmful": false}', stop_reason="end_turn"):
        self.stop_reason = stop_reason
        self.content = [SimpleNamespace(type="text", text=text)]


def _overloaded():
    return anthropic.OverloadedError(
        "overloaded_error", response=httpx.Response(529, request=httpx.Request("POST", "https://x")),
        body={"type": "error", "error": {"type": "overloaded_error"}})


def _rate_limited():
    return anthropic.RateLimitError(
        "rate_limit", response=httpx.Response(429, request=httpx.Request("POST", "https://x")), body=None)


class _Messages:
    def __init__(self, owner):
        self._o = owner

    def parse(self, **kw):
        self._o.calls += 1
        step = self._o.script.pop(0) if self._o.script else "ok"
        if step == "529":
            self._o.n_ok += self._o.sibling_ok_per_call
            raise _overloaded()
        if step == "429":
            raise _rate_limited()
        self._o.n_ok += 1
        return _Resp()


class _FakePoolClient:
    """Duck-types misalign_judge.PoolClient: .messages.parse + an n_ok success counter."""

    def __init__(self, script, sibling_ok_per_call):
        self.script = list(script)
        self.sibling_ok_per_call = sibling_ok_per_call
        self.n_ok = 0
        self.calls = 0

    @property
    def messages(self):
        return _Messages(self)


@pytest.fixture
def judge(tmp_path, monkeypatch):
    # Judge defaults to the Low-Prio org, whose key is ITS OWN env var (sisteer.vault: never another org's,
    # never a file) and is resolved at construction; the placeholder keeps the test off the vault (no
    # secret-manager CLI in the Modal image) — the client is replaced by the fake below before any call
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY_LOWPRIO", "placeholder-not-a-key")
    monkeypatch.setattr(js, "_RETRY_SLEEP_S", 0.0)
    monkeypatch.setattr(js, "_RETRY_JITTER_S", 0.0)
    return js.Judge(tmp_path / "cache.jsonl", 2, thinking=False)


def test_content_scoped_529_files_as_blocked_after_the_cap(judge):
    judge.client = _FakePoolClient(["529"] * 50, sibling_ok_per_call=3)  # siblings succeed throughout
    out = judge._call("prompt", _Out)
    assert out == {"__blocked__": True, "blocked_reason": "transport_refusal:529"}
    assert judge.client.calls == js.PER_BODY_529_CAP


def test_pool_wide_529_keeps_retrying_until_it_clears(judge):
    n = js.PER_BODY_529_CAP + 6
    judge.client = _FakePoolClient(["529"] * n, sibling_ok_per_call=0)  # nobody succeeds: capacity, not content
    out = judge._call("prompt", _Out)
    assert out == {"harmful": False}
    assert judge.client.calls == n + 1


def test_a_429_in_the_ladder_resets_the_529_count(judge):
    cap = js.PER_BODY_529_CAP
    script = ["529"] * (cap - 1) + ["429"] + ["529"] * (cap - 1) + ["ok"]
    judge.client = _FakePoolClient(script, sibling_ok_per_call=3)
    out = judge._call("prompt", _Out)
    assert out == {"harmful": False}  # never reached the cap in one unbroken 529 run
    assert judge.client.calls == len(script)


def test_plain_client_without_a_success_counter_keeps_the_old_ladder(judge):
    class _PlainMessages:
        def __init__(self, owner):
            self._o = owner

        def parse(self, **kw):
            self._o.calls += 1
            step = self._o.script.pop(0) if self._o.script else "ok"
            if step == "529":
                raise _overloaded()
            return _Resp()

    class _PlainClient:  # a bare SDK client: no n_ok, so the discriminator cannot fire
        def __init__(self, script):
            self.script = list(script)
            self.calls = 0

        @property
        def messages(self):
            return _PlainMessages(self)

    n = js.PER_BODY_529_CAP + 2
    judge.client = _PlainClient(["529"] * n + ["ok"])
    out = judge._call("prompt", _Out)
    assert out == {"harmful": False}
    assert judge.client.calls == n + 1


class _TruncMessages:
    """A parse whose reply, per script step, is a verdict cut off under a `max_tokens` stop (the SDK parses nothing client-side;
    _call reads the stop and validates the text itself)."""

    def __init__(self, owner):
        self._o = owner

    def parse(self, **kw):
        self._o.calls.append(dict(kw))
        step = self._o.script.pop(0) if self._o.script else "ok"
        if step == "trunc":
            return _Resp('{"harmful": fa', stop_reason="max_tokens")          # EOF while parsing: the shape the five u0 items raised
        return _Resp()


class _TruncClient:
    def __init__(self, script):
        self.script = list(script); self.calls = []

    @property
    def messages(self):
        return _TruncMessages(self)


def test_a_truncated_verdict_climbs_the_ladder_and_is_tallied_never_blocked(judge):
    """Five u0 agentic HarmfulVerdict replies overran the 1500-token cap (~750 chars of JSON) and
    surfaced as pydantic json_invalid, which the pass recorded as item errors (two judgments partial). A verdict cut under a `max_tokens`
    stop is re-requested one rung up the ladder (1500 -> 4000 -> 8000; tests/test_judge_sonnet_truncation_ladder.py holds the ladder's own
    cases) and tallied as retried; a verdict still truncated at the last rung raises (the caller files the item as errored) and is tallied
    as errored; neither is ever a blocked verdict."""
    judge.client = _TruncClient(["trunc", "ok"])
    out = judge._call("prompt", _Out)
    assert out == {"harmful": False} and len(judge.client.calls) == 2
    assert judge.client.calls[0]["max_tokens"] == js.VERDICT_MAX_TOKENS == 1500 and judge.client.calls[1]["max_tokens"] == js.TRUNCATION_RETRY_MAX_TOKENS == 4000
    assert (judge.n_truncated_retried, judge.n_truncated_errored) == (1, 0)
    judge.client = _TruncClient(["trunc", "trunc", "ok"])
    assert judge._call("prompt", _Out) == {"harmful": False} and [c["max_tokens"] for c in judge.client.calls] == [1500, 4000, 8000]
    assert (judge.n_truncated_retried, judge.n_truncated_errored) == (3, 0)
    judge.client = _TruncClient(["trunc", "trunc", "trunc"])
    with pytest.raises(pydantic.ValidationError):
        judge._call("prompt", _Out)
    assert len(judge.client.calls) == 3 and (judge.n_truncated_retried, judge.n_truncated_errored) == (5, 1)
    # a ValidationError that is NOT a truncation (a field missing from otherwise complete JSON) is not retried
    class _BadMessages(_TruncMessages):
        def parse(self, **kw):
            self._o.calls.append(dict(kw))
            return _Resp('{"other": 1}')
    class _BadClient(_TruncClient):
        @property
        def messages(self):
            return _BadMessages(self)
    judge.client = _BadClient([])
    with pytest.raises(pydantic.ValidationError):
        judge._call("prompt", _Out)
    assert len(judge.client.calls) == 1 and (judge.n_truncated_retried, judge.n_truncated_errored) == (5, 1)
