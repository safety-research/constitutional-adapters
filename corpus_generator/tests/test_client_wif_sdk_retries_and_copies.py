"""The corpus generator's federated clients stop at the FIRST 401 / 403 in every layer, the SDK's own included, and their copies stay
federated (Major: the corpus batch SDK retried the first WIF 401 before the engine could stop).

``build_client("batch")`` under ``SISTEER_SECRETS_MODE=wif`` gave the SDK its default four retries, and the installed SDK's
``Anthropic._should_retry`` (0.120.2 and 1.0.0 alike) INVALIDATES a federated token cache on a 401 and RETRIES THE FIRST ONE: a poll /
list / results request answered 401 was re-sent with a freshly exchanged bearer -- a second token exchange and a second request on the
rejected identity -- before ``WaveEngine._guard`` saw anything (measured on dbd7473df with the real SDK over a mock wire: 2 requests,
2 exchanges per rejected call). And sibling engines share one client across threads (``corpusgen run --corpus all``, the extension
sub-DAGs, ``onpolicy score``), so a poll asleep in the SDK's 429 backoff when a sibling's poll met the 401 woke into a third request.
The engine's own tests injected exceptions outside the SDK and could not see either.

Now, under wif only, both constructors build the SDK client with its own retries OFF (``max_retries=0``) and the count asked for
(``build_client``'s default 4, ``build_async_client``'s 0) is the budget of the member's latched retry layer (``federate_sdk_retries`` /
``LatchedRetryMessages`` -- the twin of ``sisteer.vault``'s: the same statuses, counts and delays as the SDK's, the client's credential
latch -- the halt the engines hold on the client -- checked before EVERY attempt, the credential classified FIRST after every failure and
held, never retried). The clients are the member's own subclass of the SDK's (``federated_sdk_class``), whose ``copy`` / ``with_options``
keep the federation (batch.py's ``with_options(max_retries=0)`` creator: the same credentials, token cache, transport and latch, the
layer at budget 0) and refuse by name an ``http_client``, another base URL, a credential change or a middleware; op / env build the
SDK's own class with the SDK's own retries, byte for byte as before. Real SDK over mock transports of the SDK's own httpx flavour, no
network; the SDK's own transport is swapped for the double on the very client built (the constructors take no ``http_client`` and a
copy refuses one)."""
from __future__ import annotations

import asyncio
import datetime as dt
import os
import threading
import time
from pathlib import Path

import anthropic
import pytest

from conftest import make_engine
from corpus_generator import batch as B
from corpus_generator import client as C
from corpus_generator.batch import Ledger, WaveCredentialRejected, WaveRequest
from test_client_wif_inherited_static_auth import EXCHANGED, IDS, MESSAGE, PARAMS, STATIC_KEY, _Wire, _clean, httpx

OTHER = "https://other-host.example"
WAKE_BOUND_S = 3.0
_REAL_SLEEP = time.sleep
BATCH_ID = "msgbatch_synthetic_r14"
SYNTHETIC_KEY = "synthetic-placeholder-not-a-key"
REFUSED_COPY_ARGS = ({"api_key": SYNTHETIC_KEY}, {"auth_token": "synthetic-oauth-placeholder"}, {"credentials": object()},
                     {"config": {"api_key": SYNTHETIC_KEY}}, {"profile": "synthetic-profile"}, {"webhook_key": "synthetic-webhook"},
                     {"middleware": []}, {"_extra_kwargs": {}}, {"not_an_sdk_option": 1})


def _batch_json(status: str = "in_progress", n: int = 2, results_url: str | None = None) -> dict:
    now = dt.datetime(2026, 10, 14, tzinfo=dt.timezone.utc).isoformat()
    return {"id": BATCH_ID, "type": "message_batch", "processing_status": status,
            "request_counts": {"processing": n if status != "ended" else 0, "succeeded": 0 if status != "ended" else n,
                               "errored": 0, "canceled": 0, "expired": 0},
            "created_at": now, "expires_at": now, "ended_at": now if status == "ended" else None, "archived_at": None,
            "cancel_initiated_at": None, "results_url": results_url}


def _error_body(status: int) -> dict:
    kind = {401: "authentication_error", 403: "permission_error", 429: "rate_limit_error", 400: "invalid_request_error"}.get(status, "api_error")
    return {"type": "error", "error": {"type": kind, "message": "synthetic"}}


class _ScriptedApi:
    """The API end of the wire with the answers SCRIPTED BY CALL ORDER under a lock: call k answers ``script[k-1]`` (a status, or
    ``(status, headers)``, or ``(200, headers, json)``); a call past the script answers the LAST entry again (an extra request a
    retrying layer makes lands here and is counted). A 200 answers the shape the path asks for: a batch for the batches endpoints, a
    list for the listing, a message otherwise. ``issued`` is set when a 401 / 403 has been answered; ``requests`` records method, host,
    path and the auth header of every request (never a body)."""

    def __init__(self, script):
        self.script, self.requests, self._lock = list(script), [], threading.Lock()
        self.issued = threading.Event()

    def __call__(self, request):
        with self._lock:
            n = len(self.requests) + 1
            self.requests.append({"method": request.method, "host": request.url.host, "path": request.url.path,
                                  "authorization": request.headers.get("Authorization"), "x-api-key": request.headers.get("X-Api-Key")})
        entry = self.script[min(n, len(self.script)) - 1]
        entry = entry if isinstance(entry, tuple) else (entry,)
        status, headers, body = (entry + ({}, None))[:3]
        if status == 200:
            if body is None:
                path = request.url.path
                if path.endswith("/results"):
                    return httpx.Response(200, text="", headers=headers, request=request)
                if path.endswith("/messages/batches") and request.method == "GET":
                    body = {"data": [], "has_more": False, "first_id": None, "last_id": None}
                elif "/messages/batches" in path:
                    body = _batch_json()
                else:
                    body = MESSAGE
            return httpx.Response(200, json=body, headers=headers, request=request)
        if status in (401, 403):
            self.issued.set()
        return httpx.Response(status, json=_error_body(status), headers=headers, request=request)

    def federated(self, n: int) -> None:
        """Exactly ``n`` requests left the process, every one to the pinned host with the exchanged bearer and no static header."""
        assert len(self.requests) == n, f"{len(self.requests)} request(s) on the wire (a retry below the latch?): {self.requests}"
        for r in self.requests:
            assert r["host"] == "api.anthropic.com" and r["authorization"] == f"Bearer {EXCHANGED}" and r["x-api-key"] is None, r


def _on_wire(client, api) -> None:
    """The pin guard's INNER transport swapped for the double ON THE CLIENT BUILT (round 15: the SDK is handed the member's own httpx
    client over ``PinnedTransport``; the constructors take no ``http_client`` -- residual 18 of the sweep -- and a federated copy refuses
    one): every request of the client, and of every copy made before or after this (a copy shares its parent's client), crosses the pin
    and then leaves through ``api``."""
    guard = client._client._transport
    assert type(guard) is C.PinnedTransport, type(guard)
    guard.inner = httpx.MockTransport(api)


@pytest.fixture
def wire(tmp_path, monkeypatch):
    """wif mode with every credential door nailed shut, the SDK's credentials class replaced by its offline twin, the mock wire (the
    round-10 harness, re-declared), and every inherited ``ANTHROPIC_*`` name gone."""
    d = tmp_path / "anthropic"
    d.mkdir()
    for stem, suffix in (("loprio", ""), ("batch", "-batch"), ("hiprio", "-hi")):
        (d / f"wif-{stem}.env").write_text("".join(f"{k}={v}{suffix}\n" for k, v in IDS.items()))
    (d / "wif-lowprio.env").write_text("")
    monkeypatch.setenv(C.WIF_ENV_DIR_ENV, str(d))
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "wif")
    monkeypatch.delenv(C.MODAL_MARKER_ENV, raising=False)
    for name in list(os.environ):
        if name.startswith(C.WIF_SCRUB_PREFIX):
            monkeypatch.delenv(name, raising=False)
    (tmp_path / "empty-bin").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))

    def boom(*a, **k):  # pragma: no cover - the assertion IS the test
        raise AssertionError(f"the secret-manager CLI must never run under wif: {a}")

    monkeypatch.setattr(C.subprocess, "run", boom)
    w = _Wire()
    monkeypatch.setattr(anthropic, "WorkloadIdentityCredentials", w.credentials_class())
    return w


def _held(client):
    return getattr(client, B.CLIENT_LATCH_ATTR, None)


def _sleeper(api: _ScriptedApi, held, slept: list):
    """The backoff worker B sleeps in, made deterministic: wait until a rejection was ISSUED to a sibling and then until it is HELD,
    bounded, then return -- B wakes exactly when a retry layer would re-send. Patching ``time.sleep`` reaches the SDK's own retry sleep
    and the member's layer alike."""
    real_sleep = time.sleep

    def sleep(s):
        slept.append(s)
        api.issued.wait(WAKE_BOUND_S)
        t0 = time.monotonic()
        while not held() and time.monotonic() - t0 < WAKE_BOUND_S:
            real_sleep(0.005)
    return sleep


def _two_workers(fn_a, fn_b):
    out: dict = {}

    def run(name, fn):
        try:
            out[name] = ("ok", fn())
        except BaseException as e:  # noqa: BLE001 -- the outcome under test
            out[name] = ("raised", e)

    tb = threading.Thread(target=run, args=("B", fn_b))
    tb.start()
    _REAL_SLEEP(0.15)
    ta = threading.Thread(target=run, args=("A", fn_a))
    ta.start()
    tb.join(WAKE_BOUND_S * 3)
    ta.join(WAKE_BOUND_S * 3)
    assert not tb.is_alive() and not ta.is_alive(), "a worker never returned"
    return out


def reqs(n: int) -> list[WaveRequest]:
    return [WaveRequest(custom_id=f"r{i:03d}", params={"model": "m", "max_tokens": 8, "messages": [{"role": "user", "content": f"prompt {i}"}]})
            for i in range(n)]


# ============================================================================================================ THE MAJOR


def test_under_wif_a_401_at_a_poll_a_list_a_results_read_or_a_create_is_one_request_and_one_exchange(wire, monkeypatch):
    """Against the real SDK: ``build_client("batch")`` (the default count) over a wire answering 401 -- the SDK used to
    invalidate the token cache and re-send (2 requests, 2 exchanges) before ``_guard`` classified. Now the SDK's retries are off on
    the client and the count asked for is the layer's budget: ONE request, ONE exchange, the typed halt latched on the engine and the
    client, nothing slept; a second guarded call issues nothing. At ``retrieve``, ``list``, ``results`` (both its retrieve and its
    results GET), the creator's ``create`` and the smoke client's ``messages.create``."""
    slept: list = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    def fresh(kind="batch", script=(401,), **kw):
        c = C.build_client(kind, **kw)
        _clean(c)
        api = _ScriptedApi(list(script))
        _on_wire(c, api)
        return c, api

    c, api = fresh()
    assert c.max_retries == 0 and C.federated_retry_budget(c) == 4                # the SDK's own retries OFF; the default 4 is the layer's
    assert isinstance(c.messages, C.LatchedRetryMessages)
    eng = make_engine(c)
    with pytest.raises(WaveCredentialRejected) as ei:
        eng._guard(lambda: c.messages.batches.retrieve(BATCH_ID))
    assert ei.value.status == 401 and ei.value.mode == "wif" and isinstance(ei.value.__cause__, anthropic.AuthenticationError)
    api.federated(1)
    assert wire.exchanges == 1 and wire.provider_calls == 1 and slept == []
    assert _held(c) is not None and eng._credential_rejected is not None
    with pytest.raises(WaveCredentialRejected):                                        # latched: nothing else is requested
        eng._guard(lambda: c.messages.batches.retrieve(BATCH_ID))
    with pytest.raises(WaveCredentialRejected):                                        # a sibling engine over the same client too
        make_engine(c)._guard(lambda: c.messages.batches.list(limit=100))
    api.federated(1)
    for call in (lambda c: c.messages.batches.list(limit=100), lambda c: c.messages.batches.results(BATCH_ID)):
        c, api = fresh()
        with pytest.raises(WaveCredentialRejected):
            make_engine(c)._guard(lambda: call(c))
        api.federated(1)
    # the results GET itself (the SDK retrieves first, then fetches results_url): the 401 there is the call's second request, and the last
    c, api = fresh(script=[(200, {}, _batch_json("ended", results_url=f"/v1/messages/batches/{BATCH_ID}/results")), 401])
    with pytest.raises(WaveCredentialRejected):
        make_engine(c)._guard(lambda: c.messages.batches.results(BATCH_ID))
    api.federated(2)
    assert api.requests[1]["path"].endswith("/results")
    # the creator (batch.py's with_options(max_retries=0)) at the create
    c, api = fresh(script=[403])
    creator = c.with_options(max_retries=0)
    with pytest.raises(WaveCredentialRejected) as ei:
        make_engine(c)._guard(lambda: creator.messages.batches.create(requests=[]))
    assert ei.value.status == 403
    api.federated(1)
    assert _held(c) is not None                                                        # held on the PARENT: the engines' client
    # the smoke client (sequential messages.create; the SDK's default count)
    c, api = fresh("smoke")
    assert c.max_retries == 0 and C.federated_retry_budget(c) == 4
    with pytest.raises(anthropic.AuthenticationError):                                 # raised as the SDK raised it, once
        c.messages.create(**PARAMS)
    api.federated(1)
    with pytest.raises(WaveCredentialRejected):                                        # the client's latch: the next call is typed, no request
        c.messages.create(**PARAMS)
    api.federated(1)
    assert wire.requests == []                                                         # nothing ever reached the fixture's own API end


def test_a_sibling_engine_asleep_in_a_poll_backoff_wakes_into_the_stop_two_requests(wire, monkeypatch):
    """Two engines over one client in two threads (the pipeline's shape): B's poll answered 429 (``retry-after: 5``; the layer sleeps),
    A's poll the 401. B wakes into the typed halt; exactly TWO requests left; a third engine issues nothing."""
    c = C.build_client("batch")
    api = _ScriptedApi([(429, {"retry-after": "5"}), 401])
    _on_wire(c, api)
    eng_a, eng_b = make_engine(c), make_engine(c)
    slept: list = []
    monkeypatch.setattr(time, "sleep", _sleeper(api, lambda: _held(c) is not None, slept))
    out = _two_workers(lambda: eng_a._guard(lambda: c.messages.batches.retrieve(BATCH_ID)),
                       lambda: eng_b._guard(lambda: c.messages.batches.retrieve(BATCH_ID)))
    assert out["A"][0] == out["B"][0] == "raised", out
    assert all(isinstance(out[n][1], WaveCredentialRejected) for n in ("A", "B")), out
    assert slept == [5.0]
    api.federated(2)
    with pytest.raises(WaveCredentialRejected):
        make_engine(c)._guard(lambda: c.messages.batches.retrieve(BATCH_ID))
    api.federated(2)


def test_the_engine_end_to_end_on_the_real_sdk_stops_at_the_first_poll_rejection(wire, tmp_path, monkeypatch):
    """``run_wave`` over the real SDK: the create answers a batch, the first poll a 401 -- TWO requests in all (the create, the one poll),
    ONE exchange, the typed halt out of ``run_wave``, the chunk ``submitted`` in the ledger (a re-run re-polls it), nothing stamped; a
    second wave on the engine issues nothing. Before: a third request (the poll re-sent) and a second exchange."""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    c = C.build_client("batch")
    api = _ScriptedApi([(200, {}, _batch_json()), 401])
    _on_wire(c, api)
    eng = make_engine(c)
    with pytest.raises(WaveCredentialRejected) as ei:
        eng.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert ei.value.status == 401 and ei.value.mode == "wif"
    api.federated(2)
    assert [r["method"] for r in api.requests] == ["POST", "GET"] and api.requests[1]["path"].endswith(f"/{BATCH_ID}")
    assert wire.exchanges == 1 and wire.provider_calls == 1
    states = Ledger(tmp_path / "w" / "ledger.jsonl").chunk_states()
    assert states and all(st.state == "submitted" and st.batch_id == BATCH_ID for st in states.values())
    assert not (tmp_path / "w" / "STAMP.json").exists()
    with pytest.raises(WaveCredentialRejected) as again:
        eng.run_wave(tmp_path / "w2", reqs(2), wave_name="w2")
    assert again.value is ei.value
    api.federated(2)


def test_the_layer_replays_the_sdks_policy_above_the_latch_and_its_decision_is_the_installed_sdks(wire, monkeypatch):
    """The driver's robustness contract is unchanged: a 429 with ``retry-after: 3`` sleeps 3 s then succeeds; a 500 sleeps one SDK
    backoff then succeeds; the budget is exact (two retries, three requests, then the SDK's last error); a 400 is not retried. And
    ``sdk_would_retry`` says what the installed SDK's ``_should_retry`` says for every status the engines meet, with and without
    ``x-should-retry`` (a credential rejection is classified before the policy is asked)."""
    slept: list = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    def client(script, budget=2):
        c = C.build_client("batch", max_retries=budget)
        api = _ScriptedApi(script)
        _on_wire(c, api)
        return c, api

    c, api = client([(429, {"retry-after": "3"}), 200])
    assert c.messages.batches.retrieve(BATCH_ID).id == BATCH_ID and len(api.requests) == 2 and slept == [3.0]
    slept.clear()
    c, api = client([500, 200])
    c.messages.batches.retrieve(BATCH_ID)
    assert len(api.requests) == 2 and len(slept) == 1 and 0.375 <= slept[0] <= 0.5, slept
    c, api = client([500, 500, 500, 200], budget=2)
    with pytest.raises(anthropic.InternalServerError):
        c.messages.batches.retrieve(BATCH_ID)
    assert len(api.requests) == 3
    c, api = client([400, 200])
    with pytest.raises(anthropic.BadRequestError):
        c.messages.batches.retrieve(BATCH_ID)
    assert len(api.requests) == 1
    c, api = client([(500, {"x-should-retry": "false"}), 200])
    with pytest.raises(anthropic.InternalServerError):
        c.messages.create(**PARAMS)
    assert len(api.requests) == 1
    sdk = anthropic.Anthropic(api_key=SYNTHETIC_KEY, base_url=C.API_BASE_URL, max_retries=0)
    req = httpx.Request("GET", C.API_BASE_URL + "/v1/messages/batches/x")
    for status in (400, 401, 403, 404, 408, 409, 413, 429, 500, 502, 503, 529):
        for headers in ({}, {"x-should-retry": "true"}, {"x-should-retry": "false"}):
            resp = httpx.Response(status, request=req, headers=headers, json=_error_body(status))
            exc = anthropic.APIStatusError(f"Error code: {status}", response=resp, body=_error_body(status))
            exc.status_code = status
            assert C.sdk_would_retry(exc)[0] is sdk._should_retry(resp), (status, headers)
    assert C.sdk_would_retry(anthropic.APIConnectionError(request=req)) == (True, None)
    assert C.sdk_would_retry(ValueError("not a wire error")) == (False, None)
    # a 401 never reaches the policy: classified and held first (the SDK's own loop would have invalidated the cache and re-sent)
    c, api = client([401, 200], budget=5)
    slept.clear()
    with pytest.raises(anthropic.AuthenticationError):
        c.messages.batches.retrieve(BATCH_ID)
    assert len(api.requests) == 1 and slept == [] and isinstance(_held(c), WaveCredentialRejected)
    assert B.credential_halt(_held(c)) is _held(c)                                   # the typed halt classifies as itself


# ========================================================================================================== THE COPIES


def test_with_options_max_retries_zero_is_a_federated_copy_sharing_credentials_cache_transport_and_latch(wire):
    """batch.py's creator: a copy of the batch client is the same federated class, built on the parent's credentials, token cache and
    transport, keyless and pinned, with the SDK's retries off and the layer at the budget asked (0 for the creator; the parent's
    budget for a bare copy; the count asked otherwise); a rejection met through the copy is held where the engines read it (the
    parent client) and a rejection held on the parent stops the copy. The live async client's copies the same."""
    c = C.build_client("batch")
    api = _ScriptedApi([200])
    _on_wire(c, api)
    creator = c.with_options(max_retries=0)
    assert type(creator) is type(c) and type(c) is C.federated_sdk_class(anthropic.Anthropic) and type(c) is not anthropic.Anthropic
    assert isinstance(creator, anthropic.Anthropic) and creator.credentials is c.credentials and creator.api_key is None
    assert creator._token_cache is c._token_cache and creator._client is c._client
    assert str(creator.base_url).rstrip("/") == C.API_BASE_URL
    assert creator.max_retries == 0 and C.federated_retry_budget(creator) == 0 and C.federated_retry_budget(c) == 4
    assert C.federated_retry_budget(c.copy()) == 4 and C.federated_retry_budget(c.copy(max_retries=7)) == 7
    assert type(creator.with_options(timeout=2.0)) is type(c) and creator.with_options(timeout=2.0).timeout == 2.0
    assert creator.messages.batches.create(requests=[]).id == BATCH_ID
    api.federated(1)
    assert wire.exchanges == 1
    assert c.messages.batches.retrieve(BATCH_ID).id == BATCH_ID
    api.federated(2)
    assert wire.exchanges == 1                                                         # the shared cache: one exchange for both
    # a rejection met through the copy is held on the parent (where every engine over it reads), and the parent's next call is typed
    c2 = C.build_client("batch")
    api2 = _ScriptedApi([401])
    _on_wire(c2, api2)
    creator2 = c2.with_options(max_retries=0)
    with pytest.raises(anthropic.AuthenticationError):
        creator2.messages.batches.create(requests=[])
    assert isinstance(_held(c2), WaveCredentialRejected) and C.held_credential_rejection(creator2) is _held(c2)
    with pytest.raises(WaveCredentialRejected):
        c2.messages.batches.retrieve(BATCH_ID)
    with pytest.raises(WaveCredentialRejected):
        creator2.with_options(timeout=1.0).messages.batches.create(requests=[])
    api2.federated(1)
    # the async live client
    a = C.build_async_client("live", timeout=30.0)
    assert a.max_retries == 0 and C.federated_retry_budget(a) == 0 and isinstance(a.messages, C.AsyncLatchedRetryMessages)
    assert type(a) is C.federated_sdk_class(anthropic.AsyncAnthropic)
    ap = a.with_options(max_retries=1)
    assert type(ap) is type(a) and ap.credentials is a.credentials and ap._client is a._client and C.federated_retry_budget(ap) == 1
    api3 = _ScriptedApi([200])
    _on_wire(a, api3)
    assert asyncio.run(a.with_options(timeout=5.0).messages.create(**PARAMS)).content[0].text == "ok"
    api3.federated(1)


def test_a_copy_with_an_http_client_another_destination_a_credential_or_a_middleware_is_refused_by_name(wire):
    """A copy handed an httpx client (a hooked one that re-points every request, here) is ``ClientGuardError`` BY NAME, nothing
    built, nothing sent; a base URL naming another host, a static ``X-Api-Key`` / ``Authorization`` in the copy's default headers,
    a credential / config / profile / webhook / middleware change, an unknown argument and ``with_middleware`` are refused by name
    too (the message names the argument, never a value); a clean option rides."""
    rec = _ScriptedApi([200])

    def hook(request):
        request.url = request.url.copy_with(scheme="https", host="other-host.example")

    c = C.build_client("batch")
    api = _ScriptedApi([200])
    _on_wire(c, api)
    for copier in (c.with_options, c.copy):
        with pytest.raises(C.ClientGuardError) as ei:
            copier(http_client=httpx.Client(base_url=C.API_BASE_URL, event_hooks={"request": [hook]}, transport=httpx.MockTransport(rec)))
        assert "http_client" in str(ei.value) and C.API_BASE_URL in str(ei.value) and "copy" in str(ei.value)
    with pytest.raises(C.ClientGuardError, match="base_url"):
        c.with_options(base_url=OTHER)
    for headers in ({"X-Api-Key": STATIC_KEY}, {"Authorization": "Bearer synthetic-static"}, {"x-api-key": STATIC_KEY}):
        with pytest.raises(C.ClientGuardError, match="(?i)default_headers"):
            c.with_options(default_headers=headers)
        with pytest.raises(C.ClientGuardError, match="(?i)default_headers"):
            c.copy(set_default_headers=headers)
    for kw in REFUSED_COPY_ARGS:
        (name,) = kw
        with pytest.raises(C.ClientGuardError) as ei:
            c.with_options(**kw)
        assert name in str(ei.value) and SYNTHETIC_KEY not in str(ei.value) and STATIC_KEY not in str(ei.value), (name, str(ei.value))
    with pytest.raises(C.ClientGuardError, match="middleware"):
        c.with_middleware(object())
    assert rec.requests == [] and api.requests == [] and wire.provider_calls == 0
    # behind the override's back the SDK's own copy of the subclass carries no credentials at all (the installed SDK's _is_base_client)
    bypass = anthropic.Anthropic.copy(c, http_client=httpx.Client(transport=httpx.MockTransport(rec)))
    assert bypass.credentials is None and bypass._token_cache is None and bypass.api_key is None
    cp = c.with_options(default_headers={"anthropic-beta": "synthetic-beta-r14"}, base_url=C.API_BASE_URL + "/", http_client=None)
    assert type(cp) is type(c) and cp._custom_headers.get("anthropic-beta") == "synthetic-beta-r14"
    assert cp.messages.batches.retrieve(BATCH_ID).id == BATCH_ID
    api.federated(1)
    # the async client refuses the same
    a = C.build_async_client("live", timeout=30.0)
    with pytest.raises(C.ClientGuardError, match="http_client"):
        a.with_options(http_client=httpx.AsyncClient(transport=httpx.MockTransport(rec)))
    with pytest.raises(C.ClientGuardError, match="credentials"):
        a.copy(credentials=object())


def test_under_op_and_env_the_clients_and_their_copies_are_the_sdks_own_with_the_sdks_retries(monkeypatch, tmp_path):
    """The keyed modes are byte-identical: ``build_client`` is exactly ``anthropic.Anthropic`` with the SDK's own ``max_retries`` (the
    default 4; the creator's copy 0), ``build_async_client`` exactly ``anthropic.AsyncAnthropic`` with 0, no layer, and ``with_options``
    / ``copy`` / ``with_middleware`` the SDK's own -- an ``http_client`` on a copy taken as handed, the key on the wire as before."""
    for name in list(os.environ):
        if name.startswith(C.WIF_SCRUB_PREFIX):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "env")
    monkeypatch.setenv("ANTHROPIC_API_KEY", SYNTHETIC_KEY)
    c = C.build_client("batch")
    assert type(c) is anthropic.Anthropic and c.max_retries == 4 and C.federated_retry_budget(c) is None
    assert type(c.messages).__name__ == "Messages"
    creator = c.with_options(max_retries=0)
    assert type(creator) is anthropic.Anthropic and creator.max_retries == 0 and type(c.with_middleware()) is anthropic.Anthropic
    rec = _ScriptedApi([200])
    on_wire = c.with_options(http_client=httpx.Client(transport=httpx.MockTransport(rec)))
    assert type(on_wire) is anthropic.Anthropic and on_wire.messages.create(**PARAMS).content[0].text == "ok"
    assert len(rec.requests) == 1 and rec.requests[0]["x-api-key"] == SYNTHETIC_KEY and rec.requests[0]["host"] == "api.anthropic.com"
    a = C.build_async_client("live", timeout=30.0)
    assert type(a) is anthropic.AsyncAnthropic and a.max_retries == 0 and C.federated_retry_budget(a) is None
    assert type(a.with_options(timeout=1.0)) is anthropic.AsyncAnthropic
    src = Path(C.__file__).read_text()
    assert src.count("max_retries=max_retries,\n") >= 2                              # the keyed branches: the caller's count to the SDK
    assert "federate_sdk_retries(client, max_retries=max_retries)" in src              # the wif branches: the count as the layer's budget
    bsrc = Path(B.__file__).read_text()
    assert "creator = self.client.with_options(max_retries=0)" in bsrc                 # the creator's shape in every mode
