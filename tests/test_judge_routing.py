"""Org routing + retry policy of the live judges (judge_sonnet.Judge and
judge_overrefusal.judge_all) after the 2026-08-21 routing-rule update:

- ONE Low-Prio client for the whole run, built with SDK max_retries=1
  (retry layers must not multiply) — a High-Prio client is NEVER
  constructed, however many 429s arrive (the old High-Prio-on-429
  fallback is REPLACED by same-org throttling; do not reinstate it);
- a 429 halves the AdaptiveLimiter width (same org) and the call retries;
- 529 / 5xx / connection errors retry WITHOUT shrinking the limiter;
- deterministic 4xx raise immediately (except judge_sonnet's
  content-filter 400, a per-item {"__blocked__": true} outcome).

CPU-only, no network: anthropic clients are faked; API errors are built
from the REAL anthropic exception classes over mock httpx responses so
the except-clauses are tested against the actual MRO.
"""

from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import json

import anthropic
import httpx
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import judge_overrefusal as jo  # noqa: E402
import judge_sonnet as js  # noqa: E402

import sisteer.eval.api_keys as ak  # noqa: E402
from sisteer import vault  # noqa: E402

#: not a key: the fake client's stamp digests this placeholder
_PLACEHOLDER_KEY = "ORG-KEY-PLACEHOLDER"


def _api_error(cls, status: int, msg: str = "err"):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return cls(msg, response=httpx.Response(status, request=req), body=None)


class _FakeMessages:
    """Yields `outcomes` in order (dict/str = a reply, Exception = raise);
    the last outcome repeats forever. Serves both the structured `parse`
    surface (judge_sonnet) and plain `create` (judge_overrefusal)."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def _next(self):
        self.calls += 1
        o = (self.outcomes.pop(0) if len(self.outcomes) > 1
             else self.outcomes[0])
        if isinstance(o, Exception):
            raise o
        return o

    def parse(self, **kwargs):
        # judge_sonnet validates the verdict text itself (the structured format rides output_config; the SDK parses nothing
        # client-side), so the fake answers with a text block carrying the outcome as JSON
        o = self._next()
        return types.SimpleNamespace(
            stop_reason="end_turn", stop_details=None,
            content=[types.SimpleNamespace(type="text", text=json.dumps(o))])

    def create(self, **kwargs):
        o = self._next()
        return types.SimpleNamespace(
            stop_reason="end_turn",
            content=[types.SimpleNamespace(type="text", text=o)])


class _FakeClient:
    def __init__(self, outcomes, prio="lowprio"):
        self.messages = _FakeMessages(outcomes)
        # what a vault-built client carries: the pool it resolved + 12 hex of
        # its key's digest (the judges record it beside their concurrency)
        self.sisteer_pool_stamp = vault.pool_stamp(prio, _PLACEHOLDER_KEY)


def _patch_make_client(monkeypatch, outcomes):
    """sisteer.eval.api_keys.make_client -> a capturing fake. Both judges
    import it lazily from that module, so patching the module attribute
    covers them."""
    captured, clients = [], []

    def fake_make_client(prio="lowprio", *, max_retries=5, config_dir=None):
        captured.append({"prio": prio, "max_retries": max_retries})
        clients.append(_FakeClient(outcomes, prio))
        return clients[-1]

    monkeypatch.setattr(ak, "make_client", fake_make_client)
    return captured, clients


def _no_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)


# ------------------------------------------------------ judge_sonnet.Judge


def _sonnet_judge(monkeypatch, tmp_path, outcomes, concurrency=8):
    captured, clients = _patch_make_client(monkeypatch, outcomes)
    j = js.Judge(tmp_path / "judge_cache.jsonl", concurrency, thinking=False)
    return j, captured, clients


def test_judge_sonnet_one_lowprio_client_max_retries_one(monkeypatch,
                                                         tmp_path):
    _j, captured, _ = _sonnet_judge(monkeypatch, tmp_path, [{"ok": 1}])
    assert captured == [{"prio": "lowprio", "max_retries": 1}]


def test_judge_sonnet_run_stats_record_the_pool_beside_the_concurrency(
        monkeypatch, tmp_path):
    """The routing record ``main`` writes to ``judge_sonnet.run.json``: the
    pool the client ACTUALLY resolved (never the key), the org asked for,
    the concurrency, the SDK retry count, and the limiter's 429 state."""
    j, _captured, _ = _sonnet_judge(monkeypatch, tmp_path, [{"ok": 1}],
                                    concurrency=5)
    rs = j.run_stats()
    assert rs["pool"] == "lowprio" and rs["org"] == "lowprio"
    assert rs["key_sha256_12"] == vault.pool_stamp(
        "lowprio", _PLACEHOLDER_KEY)["key_sha256_12"]
    assert rs["concurrency"] == 5 and rs["sdk_max_retries"] == 1
    assert rs["n_rate_limited"] == 0 and rs["final_width"] == 5
    assert rs["thinking"] is False
    assert _PLACEHOLDER_KEY not in json.dumps(rs)
    j.limiter.on_rate_limit()
    rs2 = j.run_stats()
    assert rs2["n_rate_limited"] == 1 and rs2["final_width"] == 2


def test_judge_sonnet_run_stats_refuse_a_client_with_no_stamp(monkeypatch,
                                                              tmp_path):
    """A stand-in client constructs fine (the read is lazy) but cannot be
    RECORDED: no pool stamp means no pool, and the record raises rather than
    writing an unknown."""
    import types
    monkeypatch.setattr(ak, "make_client",
                        lambda prio="lowprio", **_k: types.SimpleNamespace(messages=None))
    j = js.Judge(tmp_path / "judge_cache.jsonl", 2, thinking=False)
    with pytest.raises(ValueError, match="no pool stamp"):
        j.run_stats()


def test_judge_sonnet_429_throttles_same_org_and_retries(monkeypatch,
                                                         tmp_path, capsys):
    err = _api_error(anthropic.RateLimitError, 429, "rate limited")
    j, captured, clients = _sonnet_judge(
        monkeypatch, tmp_path, [err, {"questions": ["q1"]}])
    _no_sleep(monkeypatch)
    out = j._call("prompt", js.QualityQuestions)
    assert out == {"questions": ["q1"]}
    assert clients[0].messages.calls == 2      # 429 then success
    assert len(captured) == 1                  # NEVER a second (high-prio) client
    assert j.limiter.n_rate_limited == 1
    assert j.limiter.width == 4                # halved from 8
    assert "throttling DOWN" in capsys.readouterr().out


def test_judge_sonnet_sustained_429s_never_hop_orgs(monkeypatch, tmp_path):
    err = _api_error(anthropic.RateLimitError, 429, "rate limited")
    j, captured, _clients = _sonnet_judge(
        monkeypatch, tmp_path, [err] * 25 + [{"answers": []}])
    _no_sleep(monkeypatch)
    assert j._call("prompt", js.QualityAnswers) == {"answers": []}
    assert len(captured) == 1                  # one client for the whole streak
    assert j.limiter.n_rate_limited == 25
    assert not hasattr(j, "_prio")             # the flip machinery is GONE


def test_judge_sonnet_529_and_5xx_retry_without_shrinking(monkeypatch,
                                                          tmp_path):
    errs = [_api_error(anthropic.OverloadedError, 529, "overloaded"),
            _api_error(anthropic.InternalServerError, 500, "ise"),
            {"questions": ["q1"]}]
    j, _, clients = _sonnet_judge(monkeypatch, tmp_path, errs)
    _no_sleep(monkeypatch)
    assert j._call("prompt", js.QualityQuestions) == {"questions": ["q1"]}
    assert clients[0].messages.calls == 3
    assert j.limiter.n_rate_limited == 0
    assert j.limiter.width == 8                # untouched


def test_judge_sonnet_content_filter_400_is_a_blocked_item(monkeypatch,
                                                           tmp_path):
    err = _api_error(anthropic.BadRequestError, 400,
                     "blocked by content filtering policy")
    j, _, clients = _sonnet_judge(monkeypatch, tmp_path, [err, {"ok": 1}])
    assert j._call("prompt", js.QualityQuestions) == {"__blocked__": True, "blocked_reason": "content_filter"}
    assert clients[0].messages.calls == 1      # per-item outcome, no retry


def test_judge_sonnet_deterministic_4xx_raises_immediately(monkeypatch,
                                                           tmp_path):
    err = _api_error(anthropic.NotFoundError, 404, "model not found")
    j, _, clients = _sonnet_judge(monkeypatch, tmp_path, [err, {"ok": 1}])
    monkeypatch.setattr(time, "sleep",
                        lambda s: pytest.fail("slept on a non-retryable"))
    with pytest.raises(anthropic.NotFoundError):
        j._call("prompt", js.QualityQuestions)
    assert clients[0].messages.calls == 1


# ------------------------------------------------ judge_overrefusal.judge_all


def test_judge_overrefusal_429_throttles_same_org(monkeypatch, tmp_path):
    err = _api_error(anthropic.RateLimitError, 429, "rate limited")
    captured, clients = _patch_make_client(monkeypatch, [err, "C"])
    _no_sleep(monkeypatch)
    recs = [{"prompt": "hello", "response": "world"}]
    labels = jo.judge_all(recs, tmp_path / "judge_cache.jsonl", 2)
    assert labels[jo.item_key(recs[0])] == "C"
    assert captured == [{"prio": "lowprio", "max_retries": 1}]  # one client
    assert clients[0].messages.calls == 2      # 429 then success
    # the label was persisted for resume
    cached = (tmp_path / "judge_cache.jsonl").read_text()
    assert '"label": "C"' in cached


def test_judge_overrefusal_records_the_pool_beside_the_concurrency(
        monkeypatch, tmp_path):
    """``judge_all(..., run_facts)`` fills the routing record ``main`` writes
    to ``overrefusal_judge_run.json``: the resolved pool (never the key),
    concurrency, SDK retries, and the limiter's 429 count / final width."""
    err = _api_error(anthropic.RateLimitError, 429, "rate limited")
    _captured, _clients = _patch_make_client(monkeypatch, [err, "C"])
    _no_sleep(monkeypatch)
    recs = [{"prompt": "hello", "response": "world"}]
    rf: dict = {}
    labels = jo.judge_all(recs, tmp_path / "judge_cache.jsonl", 4, rf)
    assert labels[jo.item_key(recs[0])] == "C"
    assert rf["pool"] == "lowprio"
    assert rf["key_sha256_12"] == vault.pool_stamp(
        "lowprio", _PLACEHOLDER_KEY)["key_sha256_12"]
    assert rf["concurrency"] == 4 and rf["sdk_max_retries"] == 1
    assert rf["n_rate_limited"] == 1 and rf["final_width"] == 2   # the 429 halved it
    assert _PLACEHOLDER_KEY not in json.dumps(rf)


def test_judge_overrefusal_without_a_record_dict_is_unchanged(monkeypatch,
                                                              tmp_path):
    """Callers that pass no dict (the routing tests, older drivers) get the
    labels and nothing else — and a stamp-less stand-in is fine for them."""
    import types
    monkeypatch.setattr(ak, "make_client",
                        lambda prio="lowprio", **_k: types.SimpleNamespace(
                            messages=_FakeMessages(["C"])))
    recs = [{"prompt": "hello", "response": "world"}]
    labels = jo.judge_all(recs, tmp_path / "judge_cache.jsonl", 2)
    assert labels[jo.item_key(recs[0])] == "C"


def test_judge_overrefusal_529_retries_and_labels(monkeypatch, tmp_path):
    """The overrefusal judge's except-tuple must cover OverloadedError —
    a 529 escaping the net would crash a worker thread mid-run."""
    err = _api_error(anthropic.OverloadedError, 529, "overloaded")
    captured, _clients = _patch_make_client(monkeypatch, [err, "I"])
    _no_sleep(monkeypatch)
    recs = [{"prompt": "p", "response": "r"}]
    labels = jo.judge_all(recs, tmp_path / "judge_cache.jsonl", 2)
    assert labels[jo.item_key(recs[0])] == "I"
    assert len(captured) == 1


def test_judge_overrefusal_exhausted_retries_mark_x_not_crash(monkeypatch,
                                                              tmp_path):
    err = _api_error(anthropic.RateLimitError, 429, "rate limited")
    captured, _ = _patch_make_client(monkeypatch, [err])   # 429 forever
    _no_sleep(monkeypatch)
    recs = [{"prompt": "p", "response": "r"}]
    labels = jo.judge_all(recs, tmp_path / "judge_cache.jsonl", 2)
    assert labels[jo.item_key(recs[0])] == "X"             # loud unknown
    assert len(captured) == 1                              # still no org hop
    # X is NOT persisted: a rerun retries it
    assert '"label"' not in (tmp_path / "judge_cache.jsonl").read_text()
