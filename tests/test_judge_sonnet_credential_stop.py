"""judge_sonnet.Judge stops at the FIRST credential rejection (round 2 on 4574e103c).

main() submits every HyperSteer and repeated-quality future of an arm before it reads the first result, and Judge._call
propagated a 401 like any deterministic 4xx: with the default 100 workers a revoked credential had every queued request of
the arm issued and rejected in turn, and every later arm the same. Now the first 401 / 403 latches on the judge as the typed
``sisteer.vault.JudgeCredentialRejected`` (the pool, the status, the SDK's and the API's error types -- never a key, not a
RuntimeError, never worded UNSCORED), every later ``_call`` raises it BEFORE its wire call, ``main`` cancels the arm's queue,
drains the futures still running without ever waiting on one cancelled off the queue (round 1 on 7778694d0), writes no
report and exits with the typed message as the terminal line. A 404 stays the per-item hard failure it was, never latched.
No network: the client is a stand-in; the vault builds the judge's real client from a placeholder env key (op mode), as
test_judge_sonnet_body_529.py does, and the stand-in replaces it before any call."""
from __future__ import annotations

import json
import pathlib
import sys
import threading
import time

import anthropic
import httpx
import pytest
from pydantic import BaseModel

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import judge_sonnet as js  # noqa: E402
from sisteer import vault  # noqa: E402


class _Out(BaseModel):
    harmful: bool


def _credential_error(status: int = 401):
    """The SDK's credential rejection as the API returns it (401 authentication_error / 403 permission_error): body shape
    only, no key anywhere."""
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


class _Resp:
    """An end_turn reply with a schema-valid text block (_call validates the text itself)."""
    stop_details = None

    def __init__(self, text='{"harmful": false}'):
        self.stop_reason = "end_turn"
        self.content = [_Block(text)]


#: a schema-valid HyperSteerScore (main submits the HyperSteer futures first, so the first calls of a run are these)
HS_VERDICT = '{"justification": "synthetic", "follows_base_prompt": 2, "follows_steering_prompt": 1, "fluency": 2}'


class _Messages:
    def __init__(self, owner):
        self._o = owner

    def parse(self, **kw):
        with self._o.lock:
            self._o.calls += 1
            n = self._o.calls
        if self._o.delay:
            time.sleep(self._o.delay)
        exc = self._o.fail(n)
        if exc is not None:
            raise exc
        return _Resp(self._o.ok_text)


class _FakeClient:
    """Duck-types the vault's client: ``.messages.parse`` and a call counter; ``fail(n)`` decides the n-th call's fate
    (None = a verdict, whose text is ``ok_text``); ``delay`` keeps every worker in flight long enough for the stop to race
    the queue."""

    def __init__(self, fail, delay: float = 0.0, ok_text: str = '{"harmful": false}'):
        self.fail = fail
        self.delay = delay
        self.ok_text = ok_text
        self.calls = 0
        self.lock = threading.Lock()

    @property
    def messages(self):
        return _Messages(self)


@pytest.fixture
def judge(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "placeholder-not-a-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY_LOWPRIO", "placeholder-not-a-key")
    monkeypatch.setattr(js, "_RETRY_SLEEP_S", 0.0)
    monkeypatch.setattr(js, "_RETRY_JITTER_S", 0.0)
    j = js.Judge(tmp_path / "cache.jsonl", 2, thinking=False)
    yield j
    j.pool.shutdown(wait=False, cancel_futures=True)


def test_the_first_rejection_latches_and_every_later_call_raises_before_the_wire(judge):
    judge.client = _FakeClient(lambda n: _credential_error(401))
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        judge._call("synthetic prompt", _Out)
    e = ei.value
    assert (e.pool, e.status, e.error_type, e.api_error_type) == ("lowprio", 401, "AuthenticationError", "authentication_error")
    assert not isinstance(e, RuntimeError) and "UNSCORED" not in str(e)
    assert "lowprio" in str(e) and "401" in str(e) and "AuthenticationError" in str(e) and "x-api-key" not in str(e)
    assert judge.client.calls == 1 and judge.credential_rejected is e
    for _ in range(5):
        with pytest.raises(vault.JudgeCredentialRejected) as again:
            judge._call("another synthetic prompt", _Out)
        assert again.value is e
    assert judge.client.calls == 1                                             # no wire call after the first rejection
    # through the public methods and the cache layer too: nothing cached, nothing sent, nothing on disk
    with pytest.raises(vault.JudgeCredentialRejected):
        judge.hypersteer("steer", "q", "r")
    with pytest.raises(vault.JudgeCredentialRejected):
        judge.stolfo_quality("q", "r", 0)
    with pytest.raises(vault.JudgeCredentialRejected):
        judge.questions_for("q")
    assert judge.client.calls == 1 and judge.cache == {} and not judge.cache_path.exists()


def test_a_403_is_the_same_kind_and_a_404_is_still_a_per_item_failure_never_latched(judge, tmp_path):
    judge.client = _FakeClient(lambda n: _not_found())
    for _ in range(3):
        with pytest.raises(anthropic.NotFoundError):
            judge._call("synthetic prompt", _Out)
    assert judge.client.calls == 3 and judge.credential_rejected is None     # a deterministic 4xx per item, as before
    judge.client = _FakeClient(lambda n: _credential_error(403))
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        judge._call("synthetic prompt", _Out)
    assert (ei.value.pool, ei.value.status, ei.value.error_type, ei.value.api_error_type) == ("lowprio", 403, "PermissionDeniedError", "permission_error")
    assert judge.client.calls == 1 and judge.credential_rejected is ei.value


def _results(out: pathlib.Path, instr: str = "french", method: str = "steer", n: int = 24) -> pathlib.Path:
    cell = out / instr / method
    cell.mkdir(parents=True)
    recs = [{"prompt": f"synthetic prompt {i}", "response": f"synthetic response {i}", "strict": True, "loose": True} for i in range(n)]
    (cell / "results.json").write_text(json.dumps({"arms": [{"arm": "baseline", "compliance_strict": 0.5, "compliance_loose": 0.5}],
                                                  "records": [recs]}))
    return cell


def _run_main(monkeypatch, out: pathlib.Path, client, *, concurrency: int = 2, repeats: int = 2, n: int = 24) -> dict:
    """main() in a bounded thread (it hung on a cancelled future before the fix): the SystemExit it ends with, or the exception."""
    import sisteer.eval.api_keys as api_keys

    monkeypatch.setattr(api_keys, "make_client", lambda org, **kw: client)      # Judge.__init__ imports it at call time
    monkeypatch.setattr(js, "_RETRY_SLEEP_S", 0.0)
    monkeypatch.setattr(js, "_RETRY_JITTER_S", 0.0)
    monkeypatch.setattr(sys, "argv", ["judge_sonnet.py", "--out-dir", str(out), "--concurrency", str(concurrency),
                                      "--repeats", str(repeats), "--n", str(n)])
    done: dict = {}

    def run():
        try:
            js.main()
            done["returned"] = True
        except SystemExit as exc:
            done["exit"] = exc
        except BaseException as exc:  # pragma: no cover - the assertion below names it
            done["exc"] = exc

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout=90)
    assert not t.is_alive(), "main() hung after the first credential rejection"
    return done


def test_main_stops_the_arm_at_the_first_rejection_cancels_the_queue_and_writes_no_report(tmp_path, monkeypatch, capsys):
    """24 responses, 2 repeats, concurrency 2: 72 futures submitted at once. Every call is rejected: at most one call per worker in
    flight at the first rejection (never 72), the queue cancelled, no report and no run sidecar written, the typed message the
    terminal line, the one CREDENTIAL REJECTED line carrying counts and no prompt or response text."""
    client = _FakeClient(lambda n: _credential_error(401), delay=0.05)
    out = tmp_path / "out"
    _results(out, n=24)
    done = _run_main(monkeypatch, out, client)
    assert "exit" in done, done
    msg = str(done["exit"])
    assert msg.startswith("[judge] ") and "AuthenticationError" in msg and "401" in msg and "lowprio" in msg
    assert "synthetic prompt" not in msg and "synthetic response" not in msg
    assert client.calls <= 2, client.calls                                     # at most one per in-flight worker, never 72
    assert not (out / "judge_sonnet.json").exists() and not (out / "judge_sonnet.run.json").exists()
    assert not (out / "judge_cache.jsonl").exists()
    o = capsys.readouterr().out
    assert o.count("CREDENTIAL REJECTED on lowprio at french/steer arm baseline") == 1
    assert "of this arm's 72 requests were never issued" in o and "0 verdict(s) obtained before the stop" in o and "no report written (D6)" in o
    assert "synthetic prompt" not in o and "synthetic response" not in o


def test_main_keeps_the_verdict_obtained_before_the_stop_and_issues_nothing_after_it(tmp_path, monkeypatch, capsys):
    """The first call answers (a verdict, cached on disk); the second is the rejection. The verdict stands in the cache, the
    sibling worker's call that may already be in flight is the only other request, and the rest of the queue is never issued."""
    client = _FakeClient(lambda n: None if n == 1 else _credential_error(401), delay=0.05, ok_text=HS_VERDICT)
    out = tmp_path / "out"
    _results(out, n=24)
    done = _run_main(monkeypatch, out, client)
    assert "exit" in done, done
    assert client.calls <= 3, client.calls                                     # the verdict, the rejection, at most one racing sibling
    cached = [json.loads(l) for l in (out / "judge_cache.jsonl").read_text().splitlines() if l.strip()]
    assert len(cached) == 1                                                    # the one verdict, nothing else
    assert not (out / "judge_sonnet.json").exists()
    o = capsys.readouterr().out
    assert "1 verdict(s) obtained before the stop are cached" in o and "never issued" in o


# ------------------------------------------------------------------ round 5: the federated rejection shape through THIS driver


def _exchange_rejecting_client(status: int, body):
    """A REAL SDK client whose federation token exchange answers ``status`` with ``body`` (an OAuth error string, a non-JSON text, or
    the API's own error object) -- OFFLINE: the credentials' http client is a mock transport for ``/v1/oauth/token``; no bearer is ever
    minted, so the API client's own transport is never reached. The SDK raises ``WorkloadIdentityError(status_code=status)`` from the
    request that needed the bearer: an ``AnthropicError``, NOT an ``APIStatusError``. Placeholder ids; no key anywhere."""
    try:
        import httpx2 as _httpx            # anthropic >= 1.x (the lock's SDK) is built on httpx2; 0.x on httpx -- take the SDK's own
    except ImportError:
        _httpx = httpx
    hits = {"n": 0}

    def token_endpoint(request):
        assert request.url.path.endswith("/v1/oauth/token"), request.url
        hits["n"] += 1
        if isinstance(body, str):
            return _httpx.Response(status, text=body, request=request)
        return _httpx.Response(status, json=body, request=request)

    creds = anthropic.WorkloadIdentityCredentials(
        identity_token_provider=lambda: "header.payload.signature", federation_rule_id="fr_11111111",
        organization_id="org_22222222", service_account_id="sa_33333333", workspace_id="wrkspc_44444444",
        http_client=_httpx.Client(transport=_httpx.MockTransport(token_endpoint)))
    client = anthropic.Anthropic(credentials=creds, base_url=vault.ANTHROPIC_BASE_URL, max_retries=0)
    client.exchange_hits = hits                      # type: ignore[attr-defined]  -- the test's counter of token exchanges
    return client


#: the SDK's token cache retries a 401 from the token endpoint ONCE (``anthropic.lib.credentials._cache._call_provider``; a 403 is not
#: retried), so one request that meets a refused identity costs up to this many exchanges -- measured offline on anthropic 0.120.2 and
#: 1.0.0: a 401 request = 2 exchanges, a 403 request = 1. The drivers' first-rejection bound is on REQUESTS; exchanges are the SDK's.
EXCHANGES_PER_REJECTED_REQUEST = {401: 2, 403: 1}


class _CountingExchangeClient:
    """The exchange-rejecting client with the ``calls`` counter main's tests read -- one per REQUEST the judge issued
    (``messages.parse`` calls; a request never reaches the API without a bearer, so each one is a refused exchange or two) -- and
    ``exchanges``, the token endpoint's own count."""

    def __init__(self, status: int = 401, body="synthetic non-json body"):
        self._c = _exchange_rejecting_client(status, body)
        self.calls = 0
        self.lock = threading.Lock()

    @property
    def messages(self):
        return _CountingMessages(self)

    @property
    def exchanges(self) -> int:
        return self._c.exchange_hits["n"]


class _CountingMessages:
    def __init__(self, owner: _CountingExchangeClient):
        self._o = owner

    def parse(self, **kw):
        with self._o.lock:
            self._o.calls += 1
        return self._o._c.messages.parse(**kw)


def test_a_federation_identity_the_token_exchange_refuses_stops_the_judge_at_the_first_rejection_and_main_cancels_the_queue(
        judge, tmp_path, monkeypatch, capsys):
    """``_call`` ran the classifier for ``AuthenticationError`` / ``PermissionDeniedError`` only, and
    under wif the credential is the federation identity, which the API refuses at the token exchange -- the SDK raises
    ``WorkloadIdentityError(status_code=401 | 403)`` straight from ``AnthropicError``, so the rejection bypassed the latch, ``main``
    (which catches the typed error only) did not cancel the arm's already-submitted futures, and every queued worker went on to its
    own doomed exchange. Now EVERY exception is classified before the transport branches: the federated rejection latches exactly as a
    401 on a request does, whatever the token endpoint's body shape (an OAuth error string, a non-JSON text: ``api_error_type`` None),
    and main cancels the queue. The exchange's 5xx and an unreachable token endpoint are the retry loop's business, never a stop."""
    import sisteer.eval.api_keys as api_keys

    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "wif")
    # the judges below are built under wif: their vault client is a stand-in (every test replaces `.client` before a call), so the
    # real ids file is never read and no exchange is ever attempted by construction
    monkeypatch.setattr(api_keys, "make_client", lambda org, **kw: _FakeClient(lambda n: None))
    # (a) the real SDK client, the OAuth error-string body (RFC 6749): no nested authentication_error anywhere
    judge.client = _exchange_rejecting_client(401, {"error": "invalid_grant", "error_description": "synthetic"})
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        judge._call("synthetic prompt", _Out)
    e = ei.value
    assert (e.pool, e.status, e.error_type, e.api_error_type) == ("lowprio", 401, "WorkloadIdentityError", None)
    # ONE request: the SDK's own retry of a 401 exchange (EXCHANGES_PER_REJECTED_REQUEST) is the whole count, never a second request
    assert judge.client.exchange_hits["n"] == EXCHANGES_PER_REJECTED_REQUEST[401] and judge.credential_rejected is e
    assert "WorkloadIdentityError (HTTP 401)" in str(e) and "wif" in str(e) and "the federation ids file / the VM identity" in str(e)
    assert "header.payload.signature" not in str(e) and "UNSCORED" not in str(e) and not isinstance(e, RuntimeError)
    for _ in range(5):
        with pytest.raises(vault.JudgeCredentialRejected) as again:
            judge._call("another synthetic prompt", _Out)
        assert again.value is e
    assert judge.client.exchange_hits["n"] == EXCHANGES_PER_REJECTED_REQUEST[401]   # no exchange, no request after the first rejection
    # (b) a 403 with a non-JSON body is the same kind (and the SDK does not retry a 403 exchange: exactly one); the API's own error
    # object still yields its type
    j2 = js.Judge(tmp_path / "cache2.jsonl", 2, thinking=False)
    j2.client = _exchange_rejecting_client(403, "synthetic non-json body")
    with pytest.raises(vault.JudgeCredentialRejected) as ei2:
        j2._call("synthetic prompt", _Out)
    assert (ei2.value.status, ei2.value.error_type, ei2.value.api_error_type) == (403, "WorkloadIdentityError", None)
    assert j2.client.exchange_hits["n"] == EXCHANGES_PER_REJECTED_REQUEST[403]
    j3 = js.Judge(tmp_path / "cache3.jsonl", 2, thinking=False)
    j3.client = _exchange_rejecting_client(401, {"type": "error", "error": {"type": "authentication_error", "message": "synthetic"}})
    with pytest.raises(vault.JudgeCredentialRejected) as ei3:
        j3._call("synthetic prompt", _Out)
    assert (ei3.value.status, ei3.value.api_error_type) == (401, "authentication_error")
    for j in (j2, j3):
        j.pool.shutdown(wait=False, cancel_futures=True)
    # (c) NOT this: the exchange's 5xx and an unreachable token endpoint are retried like a transport failure, then the verdict stands
    for transient in (anthropic.WorkloadIdentityError("Token exchange failed (HTTP 503): synthetic", status_code=503, body="synthetic"),
                      anthropic.WorkloadIdentityError("Failed to reach token endpoint: synthetic")):
        j4 = js.Judge(tmp_path / f"cache4_{id(transient)}.jsonl", 2, thinking=False)
        j4.client = _FakeClient(lambda n, exc=transient: exc if n == 1 else None)
        assert j4._call("synthetic prompt", _Out) == {"harmful": False}
        assert j4.client.calls == 2 and j4.credential_rejected is None
        j4.pool.shutdown(wait=False, cancel_futures=True)
    # (d) through main: 24 responses x (1 HyperSteer + 2 quality) = 72 futures at concurrency 2 over the REAL rejecting client --
    # at most one REQUEST per in-flight worker (each a refused exchange or two, the SDK's own retry), the queue cancelled (the pool's
    # threads issue nothing after the stop), no report, the typed message the terminal line
    client = _CountingExchangeClient(401, {"error": "invalid_grant"})
    out = tmp_path / "out"
    _results(out, n=24)
    done = _run_main(monkeypatch, out, client)
    assert "exit" in done, done
    msg = str(done["exit"])
    assert msg.startswith("[judge] ") and "WorkloadIdentityError (HTTP 401)" in msg and "lowprio" in msg and "wif" in msg
    assert "synthetic prompt" not in msg and "synthetic response" not in msg and "header.payload.signature" not in msg
    time.sleep(0.3)                                                           # a queue NOT cancelled would be draining through the pool now
    assert 1 <= client.calls <= 2, client.calls                               # never 72
    assert client.exchanges <= EXCHANGES_PER_REJECTED_REQUEST[401] * client.calls, (client.exchanges, client.calls)
    assert not (out / "judge_sonnet.json").exists() and not (out / "judge_sonnet.run.json").exists()
    o = capsys.readouterr().out
    assert o.count("CREDENTIAL REJECTED on lowprio at french/steer arm baseline") == 1
    assert "of this arm's 72 requests were never issued" in o and "no report written (D6)" in o
