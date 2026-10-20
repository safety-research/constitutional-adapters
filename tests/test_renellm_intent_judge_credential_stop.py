"""renellm_intent_judge stops at the FIRST credential rejection (round 2 on 4574e103c).

``judge_corpus`` submitted the whole corpus and read results with ``as_completed``: a 401 raised from one future while every
queued one still issued its own rejected call on the executor's exit (no latch, no cancel). Now the first 401 / 403 is the
typed ``sisteer.vault.JudgeCredentialRejected`` (the pool off the client's stamp, else ``unknown``), latched for the process;
``_verdict`` raises it before every later wire call; the collector cancels the queue, says so once (counts only, never a
prompt) and re-raises; a later corpus makes no call at all. Transient errors and a 404 behave as before (never latched).
No network: the client is a stand-in."""
from __future__ import annotations

import pathlib
import sys
import threading
import time

import anthropic
import httpx
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import renellm_intent_judge as rj  # noqa: E402
from sisteer import vault  # noqa: E402


def _credential_error(status: int = 401):
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    if status == 401:
        body = {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
        return anthropic.AuthenticationError(f"Error code: 401 - {body}", response=httpx.Response(401, request=req), body=body)
    body = {"type": "error", "error": {"type": "permission_error", "message": "synthetic"}}
    return anthropic.PermissionDeniedError(f"Error code: 403 - {body}", response=httpx.Response(403, request=req), body=body)


def _not_found():
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return anthropic.NotFoundError("Error code: 404 - synthetic", response=httpx.Response(404, request=req), body=None)


class _Block:
    type = "text"

    def __init__(self, text):
        self.text = text


class _Msg:
    def __init__(self, text):
        self.content = [_Block(text)]


class _Messages:
    def __init__(self, owner):
        self._o = owner

    def create(self, **kw):
        with self._o.lock:
            self._o.calls += 1
            n = self._o.calls
        if self._o.delay:
            time.sleep(self._o.delay)
        exc = self._o.fail(n)
        if exc is not None:
            raise exc
        return _Msg("PRESERVED")


class _FakeClient:
    def __init__(self, fail, delay: float = 0.0):
        self.fail = fail
        self.delay = delay
        self.calls = 0
        self.lock = threading.Lock()

    @property
    def messages(self):
        return _Messages(self)


@pytest.fixture(autouse=True)
def _fresh_latch(monkeypatch):
    monkeypatch.setitem(rj._CREDENTIAL_LATCH, "error", None)


def _items(n: int):
    rows = {f"g{i:03d}": {"prompt": f"synthetic nested prompt {i}"} for i in range(n)}
    gtext = {gid: f"synthetic goal {i}" for i, gid in enumerate(rows)}
    return rows, gtext


def test_judge_corpus_stops_at_the_first_rejection_and_issues_no_queued_call(capsys):
    client = _FakeClient(lambda n: _credential_error(401), delay=0.02)
    rows, gtext = _items(30)
    t0 = time.monotonic()
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        rj.judge_corpus("grok", rows, gtext, client, "claude-sonnet-5", 3)
    e = ei.value
    assert (e.pool, e.status, e.error_type, e.api_error_type) == ("unknown", 401, "AuthenticationError", "authentication_error")
    assert 1 <= client.calls <= 3, client.calls                                # one per in-flight worker, never 30
    assert rj._CREDENTIAL_LATCH["error"] is e and not isinstance(e, RuntimeError) and "x-api-key" not in str(e)
    out = capsys.readouterr().out
    assert out.count("grok: CREDENTIAL REJECTED after 0/30 verdicts") == 1 and "NOT issued (D6)" in out
    assert "synthetic" not in out
    # the second corpus main judges, and the invariance block after it: no wire call at all once the latch is set
    n = client.calls
    with pytest.raises(vault.JudgeCredentialRejected) as again:
        rj.judge_corpus("sonnet", rows, gtext, client, "claude-sonnet-5", 3)
    assert again.value is e and client.calls == n
    assert time.monotonic() - t0 < 30


def test_the_pool_is_read_off_a_stamped_client_and_a_403_is_the_same_kind():
    client = _FakeClient(lambda n: _credential_error(403))
    setattr(client, vault.POOL_STAMP_ATTR, {"pool": "lowprio", "auth": "wif", "ids_sha12": "0" * 12})
    rows, gtext = _items(5)
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        rj.judge_corpus("x", rows, gtext, client, "m", 2)
    assert (ei.value.pool, ei.value.status, ei.value.error_type, ei.value.api_error_type) == ("lowprio", 403, "PermissionDeniedError", "permission_error")
    assert client.calls <= 2 and "lowprio" in str(ei.value) and "403" in str(ei.value)


def test_a_404_is_still_a_per_item_failure_never_latched_and_verdicts_flow_when_nothing_is_rejected():
    client = _FakeClient(lambda n: _not_found())
    rows, gtext = _items(4)
    with pytest.raises(anthropic.NotFoundError):
        rj.judge_corpus("x", rows, gtext, client, "m", 2)
    assert rj._CREDENTIAL_LATCH["error"] is None
    ok = _FakeClient(lambda n: None)
    rep = rj.judge_corpus("x", rows, gtext, ok, "m", 2)
    assert rep["n"] == 4 and rep["preserved"] == 4 and ok.calls == 4 and set(rep["verdicts"].values()) == {"PRESERVED"}
