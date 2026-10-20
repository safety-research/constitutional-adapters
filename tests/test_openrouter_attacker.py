"""The OpenRouter-hosted attacker client + the recording wrapper.

CPU only, NO network: a fake ``session`` stands in for OpenRouter. Pins the
wire contract the previous grok run in this repo got wrong (Anthropic-only
params 400 on OpenRouter) and the failure semantics: a 4xx is FINAL, 429/5xx
retry then fail loud, never a silent fall-back.
"""

from __future__ import annotations

import json

import pytest
import requests

from sisteer.eval.attacks.openrouter_attacker import (
    DEFAULT_ENDPOINT,
    OpenRouterAttackerClient,
    OpenRouterError,
    RecordingAttackerClient,
)
from sisteer.eval.attacks.vllm_attacker import VLLMEndpointUnreachable


class _Resp:
    def __init__(self, status=200, body=None, text=""):
        self.status_code = status
        self._body = body if body is not None else {}
        self.text = text or json.dumps(self._body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


class _Session:
    """Scripted OpenRouter: GET /v1/key + /v1/models; POST answers from a
    queue of (status, body) so retries can be exercised."""

    def __init__(self, *, models=("x-ai/grok-4.6",), key_status=200,
                 posts=None, content="{\"prompt\": \"p\"}"):
        self.headers = {}
        self.models = list(models)
        self.key_status = key_status
        self.post_queue = list(posts or [])
        self.default_content = content
        self.posts = []
        self.gets = []

    def get(self, url, timeout=None):
        self.gets.append(url)
        if url.endswith("/v1/key"):
            return _Resp(self.key_status, {"data": {"label": "k"}})
        if url.endswith("/v1/models"):
            return _Resp(200, {"data": [{"id": m} for m in self.models]})
        return _Resp(404, {})

    def post(self, url, json=None, timeout=None):
        self.posts.append({"url": url, "json": json})
        if self.post_queue:
            status, body = self.post_queue.pop(0)
            return _Resp(status, body)
        return _Resp(200, {
            "choices": [{"message": {"content": self.default_content,
                                     "reasoning": "thinking..."},
                         "finish_reason": "stop", "native_finish_reason": "completed"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 40,
                      "completion_tokens_details": {"reasoning_tokens": 33}},
            "provider": "xAI"})


def _client(session, **kw):
    return OpenRouterAttackerClient("x-ai/grok-4.6", api_key="k", session=session,
                                    backoff=0.0, **kw)


def test_payload_is_plain_openai_subset_with_bearer():
    s = _Session()
    c = _client(s)
    assert c.endpoint == DEFAULT_ENDPOINT
    assert s.headers["Authorization"] == "Bearer k"
    out = c.chat([{"role": "user", "content": "q"}], temperature=1.0, top_p=0.9,
                 max_new_tokens=1024)
    assert out == "{\"prompt\": \"p\"}"
    p = s.posts[0]["json"]
    assert p == {"model": "x-ai/grok-4.6", "messages": [{"role": "user", "content": "q"}],
                 "temperature": 1.0, "top_p": 0.9, "max_tokens": 1024}
    for bad in ("chat_template_kwargs", "thinking", "reasoning"):
        assert bad not in p
    assert s.posts[0]["url"] == f"{DEFAULT_ENDPOINT}/v1/chat/completions"
    assert c.last_meta["reasoning_tokens"] == 33 and c.last_meta["finish_reason"] == "stop"
    assert c.last_meta["reasoning_chars"] == len("thinking...")


def test_reasoning_block_only_when_asked_and_think_tags_stripped():
    s = _Session(content="<think>x</think>final")
    c = _client(s, reasoning={"effort": "low"})
    out = c.chat([{"role": "user", "content": "q"}], temperature=0.0, top_p=1.0,
                 max_new_tokens=8)
    assert out == "final"
    assert s.posts[0]["json"]["reasoning"] == {"effort": "low"}


def test_health_checks_key_and_model_listing():
    c = _client(_Session())
    c.check_health()
    with pytest.raises(OpenRouterError):
        _client(_Session(key_status=401)).check_health()
    with pytest.raises(OpenRouterError):
        _client(_Session(models=("x-ai/grok-4.5",))).check_health()


def test_4xx_is_final_no_retry():
    s = _Session(posts=[(400, {"error": {"message": "Reasoning is mandatory"}})])
    c = _client(s)
    with pytest.raises(OpenRouterError) as ei:
        c.chat([{"role": "user", "content": "q"}], temperature=1.0, top_p=0.9,
               max_new_tokens=10)
    assert "Reasoning is mandatory" in str(ei.value)
    assert len(s.posts) == 1


def test_5xx_and_429_retry_then_succeed_or_fail_loud():
    s = _Session(posts=[(503, {}), (429, {})])
    c = _client(s, max_retries=3)
    out = c.chat([{"role": "user", "content": "q"}], temperature=1.0, top_p=0.9,
                 max_new_tokens=10)
    assert out and len(s.posts) == 3 and c.last_meta["attempt"] == 2
    s2 = _Session(posts=[(500, {})] * 5)
    with pytest.raises(VLLMEndpointUnreachable):
        _client(s2, max_retries=2).chat([{"role": "user", "content": "q"}],
                                        temperature=1.0, top_p=0.9, max_new_tokens=10)
    assert len(s2.posts) == 3


def test_requires_key_and_model():
    with pytest.raises(ValueError):
        OpenRouterAttackerClient("x-ai/grok-4.6", api_key="", session=_Session())
    with pytest.raises(ValueError):
        OpenRouterAttackerClient("", api_key="k", session=_Session())


def test_recording_wrapper_logs_calls_and_errors(tmp_path):
    s = _Session(posts=[(400, {"error": {"message": "bad param"}})])
    inner = _client(s)
    rec = RecordingAttackerClient(inner, tmp_path / "calls.jsonl", role="attacker")
    with pytest.raises(OpenRouterError):
        rec.chat([{"role": "user", "content": "q1"}], temperature=1.0, top_p=0.9,
                 max_new_tokens=10)
    out = rec.chat([{"role": "system", "content": "sys"}, {"role": "user", "content": "q2"}],
                   temperature=1.0, top_p=0.9, max_new_tokens=10)
    assert out
    rows = [json.loads(l) for l in (tmp_path / "calls.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rec.n_calls == 2 and rec.n_errors == 1
    assert rows[0]["role"] == "attacker" and "bad param" in rows[0]["error"]
    assert rows[1]["messages"][1]["content"] == "q2" and rows[1]["reply"] == out
    assert rows[1]["meta"]["reasoning_tokens"] == 33
    assert rows[1]["params"] == {"temperature": 1.0, "top_p": 0.9, "max_new_tokens": 10}
    # attribute passthrough (health check, model) reaches the inner client
    assert rec.model == "x-ai/grok-4.6"
    rec.check_health()


def test_recording_wrapper_truncates_scorer_replies(tmp_path):
    s = _Session(content="7" * 500)
    rec = RecordingAttackerClient(_client(s), tmp_path / "s.jsonl", role="scorer",
                                  max_reply_chars=20)
    out = rec.chat([{"role": "user", "content": "rate"}], temperature=0.0, top_p=1.0,
                   max_new_tokens=8)
    assert len(out) == 500                       # the caller gets the full reply
    row = json.loads((tmp_path / "s.jsonl").read_text().splitlines()[0])
    assert len(row["reply"]) == 20 and row["reply_chars"] == 500 and row["role"] == "scorer"
