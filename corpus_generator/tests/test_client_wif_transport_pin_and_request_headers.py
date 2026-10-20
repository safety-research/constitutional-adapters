"""The corpus generator's federated clients send to the pinned API host and nothing else, ENFORCED WHERE THE BYTES LEAVE, and refuse
a static per-request credential header on every door.

THE CRITICAL: under ``SISTEER_SECRETS_MODE=wif`` ``build_client("batch")`` handed the SDK its OWN default httpx client, whose transport
sends wherever a request is addressed -- and the installed SDK's ``messages.batches.results`` takes the request's destination off the
RETRIEVE RESPONSE (``batch.results_url``; ``_prepare_url`` keeps an absolute URL as it is). A batch answer naming another host made the
SDK send a new authenticated GET there, the exchanged bearer on it (measured on ca5032a3c with the real SDK over a mock wire: the
retrieve to the pin, then ``GET other-host.example/steal/results`` with ``Authorization: Bearer <the exchanged bearer>``). Now every
federated client is built on the member's OWN httpx2 client (``federated_http_client``) whose one transport is ``PinnedTransport``: a
request not addressed to the pin is ``DestinationRefused`` by name at the last in-process step before a byte leaves (a results_url, a
cross-host redirect, anything that re-pointed the request inside the process), re-raised typed by the latched layer and never retried,
never held as a credential rejection; a copy shares its parent's client, so the guard travels.

THE MAJOR: the latched layer forwarded a caller's ``extra_headers`` unchanged, and the SDK's ``AccessTokenAuth`` skips the token
exchange for a request already carrying ``X-Api-Key`` / ``Authorization`` (measured: provider 0, exchanges 0, the static header on
the wire at ``messages.create`` and ``messages.batches.retrieve``). Now the layer refuses either header by name before every attempt of
every door, sync and async (``_LatchedRetry._refuse_request`` -> ``refuse_static_auth_headers`` -> ``StaticAuthRefused``), and the
same function refuses a static header in a copy's ``default_headers``.

ONE RULE: this member imports no sisteer, so it carries the vault's twins -- and the last test here imports the vault's module from
the repository beside this member and pins the twins EQUAL to the vault's (the guard's decisions, the header names, the socket options,
the refusals). Real SDK over mock transports of the SDK's own httpx flavour, no network; the test seam is the guard's inner transport,
swapped on the very client built (the constructors take no ``http_client`` and a copy refuses one); op / env byte-identical."""
from __future__ import annotations

import asyncio
import importlib
import os
import re
import sys
import time
from pathlib import Path

import anthropic
import pytest

from conftest import make_engine
from corpus_generator import client as C
from corpus_generator.batch import Ledger
from test_client_wif_inherited_static_auth import EXCHANGED, IDS, MESSAGE, PARAMS, STATIC_KEY, _Wire, _clean, httpx
from test_client_wif_sdk_retries_and_copies import BATCH_ID, _batch_json, _on_wire, reqs

PIN = C.API_BASE_URL
OTHER = "https://other-host.example"
STATIC_BEARER = "static-bearer-placeholder-r15"
STATIC_HEADERS = ({"Authorization": "Bearer " + STATIC_BEARER}, {"X-Api-Key": STATIC_KEY}, {"x-api-key": STATIC_KEY},
                  {"AUTHORIZATION": "Bearer " + STATIC_BEARER})
RESULT_LINE = '{"custom_id": "r000", "result": {"type": "errored", "error": {"type": "api_error", "message": "synthetic"}}}\n'
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def wire(tmp_path, monkeypatch):
    """wif mode with every credential door nailed shut, the SDK's credentials class replaced by its offline twin, the mock wire (the
    round-10 harness, re-declared as round 14 did), and every inherited ``ANTHROPIC_*`` name gone."""
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


class _Record:
    """A mock API end that records every request's method, host, path, query, auth headers and one named extra header (never a
    body), and answers by path: a batch (``ended``, carrying ``results_url``) for the batches endpoints, a one-line results stream, an
    empty listing, a count, a message -- or a 307 when told to."""

    def __init__(self, *, results_url: str | None = None, redirect_to: str | None = None, redirect_once: bool = True,
                 extra: str = "x-synthetic"):
        self.requests: list = []
        self.results_url, self.redirect_to, self.redirect_once, self.extra = results_url, redirect_to, redirect_once, extra

    def __call__(self, request):
        self.requests.append({"method": request.method, "host": request.url.host, "path": request.url.path,
                              "query": str(request.url.query, "ascii"), "authorization": request.headers.get("Authorization"),
                              "x-api-key": request.headers.get("X-Api-Key"), self.extra: request.headers.get(self.extra)})
        if self.redirect_to is not None and (not self.redirect_once or len(self.requests) == 1):
            return httpx.Response(307, headers={"location": self.redirect_to}, request=request)
        path = request.url.path
        if path.endswith("/results"):
            return httpx.Response(200, text=RESULT_LINE, request=request)
        if path.endswith("/messages/batches") and request.method == "GET":
            return httpx.Response(200, json={"data": [], "has_more": False, "first_id": None, "last_id": None}, request=request)
        if "/messages/batches" in path:
            return httpx.Response(200, json=_batch_json("ended", n=1, results_url=self.results_url), request=request)
        if path.endswith("/count_tokens"):
            return httpx.Response(200, json={"input_tokens": 7}, request=request)
        return httpx.Response(200, json=MESSAGE, request=request)

    def hosts(self) -> list:
        return [r["host"] for r in self.requests]


def _own(client, *, async_client: bool = False):
    """The httpx client the SDK was handed: the member's own class over the pin guard, bare of hooks / mounts / auth, pinned."""
    hc = getattr(client, "_client")
    assert type(hc) is (C.FederatedAsyncHttpxClient if async_client else C.FederatedHttpxClient), type(hc)
    assert isinstance(hc, httpx.AsyncClient if async_client else httpx.Client)
    assert type(hc._transport) is C.PinnedTransport
    assert hc._mounts == {} and hc.auth is None and hc.event_hooks == {"request": [], "response": []}, (hc._mounts, hc.auth, hc.event_hooks)
    assert str(hc.base_url).rstrip("/") == PIN and hc.follow_redirects is True
    return hc


def _doors(messages):
    """Every door of a federated client's ``messages`` resource, as (name, call-with-extra-headers)."""
    def count(**h):
        return messages.count_tokens(model=PARAMS["model"], messages=PARAMS["messages"], **h)
    return [
        ("messages.create", lambda **h: messages.create(**PARAMS, **h)),
        ("messages.parse", lambda **h: messages.parse(**PARAMS, **h)),
        ("messages.count_tokens", count),
        ("messages.stream", lambda **h: messages.stream(**PARAMS, **h)),
        ("messages.batches.create", lambda **h: messages.batches.create(requests=[], **h)),
        ("messages.batches.retrieve", lambda **h: messages.batches.retrieve(BATCH_ID, **h)),
        ("messages.batches.list", lambda **h: messages.batches.list(**h)),
        ("messages.batches.cancel", lambda **h: messages.batches.cancel(BATCH_ID, **h)),
        ("messages.batches.delete", lambda **h: messages.batches.delete(BATCH_ID, **h)),
        ("messages.batches.results", lambda **h: messages.batches.results(BATCH_ID, **h)),
    ]


def _refused_request(ei, door: str, header: str) -> None:
    msg = str(ei.value)
    assert isinstance(ei.value, C.StaticAuthRefused) and isinstance(ei.value, C.ClientGuardError) and isinstance(ei.value, ValueError)
    assert f"({door})" in msg and f"extra_headers: {header}" in msg and "AccessTokenAuth" in msg, msg
    assert STATIC_KEY not in msg and STATIC_BEARER not in msg


def _refused_destination(ei, host: str) -> None:
    msg = str(ei.value)
    assert isinstance(ei.value, C.DestinationRefused) and isinstance(ei.value, C.ClientGuardError)
    assert host in msg and PIN in msg and "before any byte leaves" in msg, msg
    assert "steal" not in msg and "/v1/messages" not in msg and EXCHANGED not in msg and STATIC_BEARER not in msg


# ============================================================================================================ THE CRITICAL


def test_sols_off_host_results_url_is_refused_at_the_transport_before_any_request_leaves(wire, monkeypatch):
    """Against the real SDK: the retrieve answers a batch whose ``results_url`` is an absolute URL on another host. The
    SDK's follow-up GET -- bearer and all -- is refused by name at the transport: ONE request on the wire (the retrieve, to the pin),
    none to the other host, the typed ``DestinationRefused`` out of the layer (its cause the SDK's connection error), nothing slept
    (the default budget 4 would have shown as five retrieves), the latch clean (not a credential rejection), ONE exchange. Through the
    engine's guard the same error passes raw and typed, nothing latched; the creator copy shares the guard; the API's own shape (an
    absolute ``results_url`` on the pinned host) is followed; the bare installed SDK sends the bearer off-host (the reason, pinned);
    the async client refuses the same."""
    slept: list = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    rec = _Record(results_url=OTHER + "/steal/results")
    c = C.build_client("batch")
    _clean(c)
    _on_wire(c, rec)
    with pytest.raises(C.DestinationRefused) as ei:
        c.messages.batches.results(BATCH_ID)
    _refused_destination(ei, "other-host.example")
    assert isinstance(ei.value.__cause__, anthropic.APIConnectionError)
    assert [(r["method"], r["host"], r["path"]) for r in rec.requests] == [("GET", "api.anthropic.com", f"/v1/messages/batches/{BATCH_ID}")]
    assert rec.requests[0]["authorization"] == f"Bearer {EXCHANGED}"
    assert slept == [] and C.held_credential_rejection(c) is None and wire.exchanges == 1 and wire.provider_calls == 1
    # through the engine's guard: not a credential rejection -- raw, typed, nothing latched on the engine or the client
    eng = make_engine(c)
    with pytest.raises(C.DestinationRefused):
        eng._guard(lambda: c.messages.batches.results(BATCH_ID))
    assert eng._credential_rejected is None and C.held_credential_rejection(c) is None and len(rec.requests) == 2
    # the creator copy (batch.py's with_options(max_retries=0)) shares the parent's client and so the guard
    creator = c.with_options(max_retries=0)
    assert creator._client is c._client
    with pytest.raises(C.DestinationRefused, match="other-host.example"):
        creator.messages.batches.results(BATCH_ID)
    assert len(rec.requests) == 3 and set(rec.hosts()) == {"api.anthropic.com"} and slept == []
    # the API's own shape -- an absolute results_url on the pinned host -- is followed: two requests, both to the pin
    rec2 = _Record(results_url=PIN + f"/v1/messages/batches/{BATCH_ID}/results")
    c2 = C.build_client("batch")
    _on_wire(c2, rec2)
    rows = list(c2.messages.batches.results(BATCH_ID))
    assert len(rows) == 1 and rows[0].custom_id == "r000"
    assert rec2.hosts() == ["api.anthropic.com", "api.anthropic.com"] and rec2.requests[1]["path"].endswith("/results")
    # the installed SDK, bare over the same wire, sends the exchanged bearer to the other host: the reason this guard exists
    rec3 = _Record(results_url=OTHER + "/steal/results")
    bare = anthropic.Anthropic(credentials=C.wif_credentials("batch"), base_url=PIN, max_retries=0,
                               http_client=httpx.Client(transport=httpx.MockTransport(rec3)))
    list(bare.messages.batches.results(BATCH_ID))
    assert rec3.hosts() == ["api.anthropic.com", "other-host.example"]
    assert rec3.requests[1]["path"] == "/steal/results" and rec3.requests[1]["authorization"] == f"Bearer {EXCHANGED}"
    # the async client refuses the same
    a = C.build_async_client("live", timeout=30.0)
    reca = _Record(results_url=OTHER + "/steal/results")
    _on_wire(a, reca)
    with pytest.raises(C.DestinationRefused) as eia:
        asyncio.run(a.messages.batches.results(BATCH_ID))
    _refused_destination(eia, "other-host.example")
    assert reca.hosts() == ["api.anthropic.com"]


def test_the_engine_end_to_end_on_the_real_sdk_refuses_the_off_host_results_url_and_leaves_nothing_partial(wire, tmp_path, monkeypatch):
    """``run_wave`` over the real SDK: the create answers a batch already ended whose ``results_url`` names another host; the poll sees
    it ended, the harvest's results call retrieves and would GET off-host -- refused. Exactly THREE requests (the create, the poll, the
    results call's own retrieve), all to the pin; the typed error out of ``run_wave`` (not a credential halt: nothing latched); no
    ``results-*.jsonl`` and no ``.tmp`` in the wave directory, nothing stamped, the chunk ``submitted`` in the ledger (a re-run re-polls)."""
    monkeypatch.setattr(time, "sleep", lambda s: None)
    rec = _Record(results_url=OTHER + "/steal/results")
    c = C.build_client("batch")
    _on_wire(c, rec)
    eng = make_engine(c)
    with pytest.raises(C.DestinationRefused, match="other-host.example"):
        eng.run_wave(tmp_path / "w", reqs(1), wave_name="w")
    assert [r["method"] for r in rec.requests] == ["POST", "GET", "GET"] and set(rec.hosts()) == {"api.anthropic.com"}
    assert rec.requests[1]["path"].endswith(f"/{BATCH_ID}") and rec.requests[2]["path"].endswith(f"/{BATCH_ID}")
    assert eng._credential_rejected is None and C.held_credential_rejection(c) is None
    assert sorted(p.name for p in (tmp_path / "w").glob("results-*")) == [] and not list((tmp_path / "w").glob("*.tmp"))
    assert not (tmp_path / "w" / "STAMP.json").exists()
    states = Ledger(tmp_path / "w" / "ledger.jsonl").chunk_states()
    assert states and all(st.state == "submitted" and st.batch_id == BATCH_ID for st in states.values())


def test_a_cross_host_redirect_is_refused_at_the_transport_and_a_same_origin_one_is_followed(wire, monkeypatch):
    """A 307 the API would answer to another host: the follow-up request, bearer and all, never leaves -- one request, the typed error
    (never retried: nothing slept, the latch clean); a same-origin redirect is followed as httpx does; the async client the same."""
    slept: list = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    rec = _Record(redirect_to=OTHER + "/v1/messages")
    c = C.build_client("smoke", max_retries=3)
    _on_wire(c, rec)
    with pytest.raises(C.DestinationRefused) as ei:
        c.messages.create(**PARAMS)
    _refused_destination(ei, "other-host.example")
    assert isinstance(ei.value.__cause__, anthropic.APIConnectionError)
    assert rec.hosts() == ["api.anthropic.com"] and slept == [] and C.held_credential_rejection(c) is None
    rec = _Record(redirect_to="/v1/messages?again=1")
    c = C.build_client("smoke", max_retries=0)
    _on_wire(c, rec)
    assert c.messages.create(**PARAMS).content[0].text == "ok"
    assert [(r["host"], r["query"]) for r in rec.requests] == [("api.anthropic.com", ""), ("api.anthropic.com", "again=1")]
    rec = _Record(redirect_to=OTHER + "/v1/messages")
    a = C.build_async_client("live", max_retries=3, timeout=30.0)
    _on_wire(a, rec)
    with pytest.raises(C.DestinationRefused, match="other-host.example"):
        asyncio.run(a.messages.create(**PARAMS))
    assert rec.hosts() == ["api.anthropic.com"] and slept == []
    assert C.destination_refusal(RuntimeError("not one")) is None


def test_the_pin_guard_unit_refuses_by_name_with_the_destination_and_the_pin_never_a_path_or_a_header():
    rec = _Record()
    guard = C.PinnedTransport(httpx.MockTransport(rec), subject="the unit")
    for ok in (PIN + "/v1/messages", "https://api.anthropic.com:443/v1/messages"):
        assert guard.handle_request(httpx.Request("POST", ok)).status_code == 200
    assert len(rec.requests) == 2
    for bad in (OTHER + "/v1/messages", "http://api.anthropic.com/v1/messages", "https://api.anthropic.com:8443/v1/messages",
                "https://api.anthropic.com.example/v1/messages", "http://127.0.0.1:1/double/v1/messages?secret=no"):
        with pytest.raises(C.DestinationRefused) as ei:
            guard.handle_request(httpx.Request("POST", bad, headers={"Authorization": "Bearer " + STATIC_BEARER}))
        msg = str(ei.value)
        assert "the unit" in msg and PIN in msg and "before any byte leaves" in msg and httpx.URL(bad).host in msg
        assert "/v1/messages" not in msg and "secret" not in msg and STATIC_BEARER not in msg
    assert len(rec.requests) == 2
    # the async face of the same guard
    async def go():
        resp = await guard.handle_async_request(httpx.Request("GET", PIN + "/v1/messages/batches"))
        with pytest.raises(C.DestinationRefused):
            await guard.handle_async_request(httpx.Request("GET", OTHER + "/v1/messages/batches"))
        return resp.status_code
    assert asyncio.run(go()) == 200 and len(rec.requests) == 3


def test_the_federated_clients_are_the_members_own_over_the_pin_guard_with_the_sdks_defaults_and_the_live_transport(wire, monkeypatch):
    """The sync client: the member's own class over the guard, whose inner transport is httpx's HTTP transport with the SDK's
    connection limits and keepalive socket options, no hook / mount / auth, no environment proxy mounted under HTTPS_PROXY / ALL_PROXY,
    the SDK's default timeout; copies share it. The async live client: the same class family over the guard, with
    the live runner's plain stdlib ssl context (the truststore race of 2026-09-03 stays fixed under wif) and wave-sized limits INSIDE
    the guard, the live timeout."""
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    c = C.build_client("batch")
    hc = _own(c)
    inner = hc._transport.inner
    assert type(inner) is httpx.HTTPTransport
    assert inner._pool._max_connections == anthropic.DEFAULT_CONNECTION_LIMITS.max_connections
    assert inner._pool._max_keepalive_connections == anthropic.DEFAULT_CONNECTION_LIMITS.max_keepalive_connections
    assert list(inner._pool._socket_options) == list(C.FEDERATED_SOCKET_OPTIONS) and C.FEDERATED_SOCKET_OPTIONS[0][1] == __import__("socket").SO_KEEPALIVE
    assert hc.timeout == anthropic.DEFAULT_TIMEOUT
    assert c.with_options(max_retries=0)._client is hc and c.copy()._client is hc
    a = C.build_async_client("live", timeout=30.0)
    ha = _own(a, async_client=True)
    inner = ha._transport.inner
    assert type(inner) is httpx.AsyncHTTPTransport
    assert type(inner._pool._ssl_context).__module__ == "ssl"                    # plain stdlib ssl, not truststore
    assert inner._pool._max_connections == 1000 and inner._pool._max_keepalive_connections == 256
    assert ha.timeout == httpx.Timeout(30.0) and a.timeout == 30.0
    assert a.with_options(max_retries=1)._client is ha
    # op / env: untouched -- the live runner's own plain client, the SDK's own default client
    for name in list(os.environ):
        if name.startswith(C.WIF_SCRUB_PREFIX):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "env")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-placeholder-not-a-key")
    plain = C.live_http_client(30.0)
    assert type(plain) is httpx.AsyncClient and plain._transport._pool._max_connections == 1000
    e = C.build_async_client("live", timeout=30.0)
    assert type(e._client) is httpx.AsyncClient and type(e._client._transport) is httpx.AsyncHTTPTransport
    k = C.build_client("batch")
    assert type(k._client) is not C.FederatedHttpxClient and type(k._client._transport) is not C.PinnedTransport
    assert type(k._client).__module__.startswith("anthropic.")                     # the SDK's own default client


# ============================================================================================================== THE MAJOR


def test_a_static_per_request_credential_header_is_refused_on_every_door_of_a_federated_sync_client(wire):
    """On ca5032a3c ``messages.create(..., extra_headers={"Authorization": ...})`` on a corpus wif client sent the static header with
    zero exchanges. Now every door refuses it by name before the wire -- either header, any letter case -- naming the door and the
    header, never a value; nothing on the wire, the provider never invoked; the stream manager's own enter refuses; the creator copy
    refuses the same; a non-credential extra header still rides every door's request with the exchanged bearer."""
    rec = _Record()
    c = C.build_client("batch")
    _on_wire(c, rec)
    for door, call in _doors(c.messages):
        for headers in STATIC_HEADERS:
            with pytest.raises(C.StaticAuthRefused) as ei:
                call(extra_headers=headers)
            _refused_request(ei, door, next(iter(headers)))
    with pytest.raises(C.StaticAuthRefused, match=r"\(messages.stream\)"):
        with C._LatchedRetryStream(c.messages, (), {**PARAMS, "extra_headers": STATIC_HEADERS[0]}):
            pass
    creator = c.with_options(max_retries=0)
    for door, call in _doors(creator.messages):
        with pytest.raises(C.StaticAuthRefused) as ei:
            call(extra_headers=STATIC_HEADERS[1])
        _refused_request(ei, door, "X-Api-Key")
    assert rec.requests == [] and wire.provider_calls == 0 and wire.exchanges == 0
    clean = {"x-synthetic": "rides"}
    assert c.messages.create(**PARAMS, extra_headers=clean).content[0].text == "ok"
    c.messages.parse(**PARAMS, extra_headers=clean)
    c.messages.count_tokens(model=PARAMS["model"], messages=PARAMS["messages"], extra_headers=clean)
    c.messages.batches.list(extra_headers=clean)
    creator.messages.batches.retrieve(BATCH_ID, extra_headers=clean)
    assert [r["path"] for r in rec.requests] == ["/v1/messages", "/v1/messages", "/v1/messages/count_tokens", "/v1/messages/batches",
                                                 f"/v1/messages/batches/{BATCH_ID}"]
    for r in rec.requests:
        assert r["host"] == "api.anthropic.com" and r["authorization"] == f"Bearer {EXCHANGED}" and r["x-api-key"] is None and r["x-synthetic"] == "rides", r
    assert wire.provider_calls == 1 and wire.exchanges == 1


def test_a_static_per_request_credential_header_is_refused_on_every_door_of_a_federated_async_client(wire):
    rec = _Record()
    a = C.build_async_client("live", timeout=30.0)
    _on_wire(a, rec)

    async def refused(call, headers):
        with pytest.raises(C.StaticAuthRefused) as ei:
            r = call(extra_headers=headers)
            if hasattr(r, "__aenter__"):
                async with r:
                    pass
            else:
                await r
        return ei

    for door, call in _doors(a.messages):
        for headers in STATIC_HEADERS:
            ei = asyncio.run(refused(call, headers))
            _refused_request(ei, door, next(iter(headers)))

    async def manager_enter():
        with pytest.raises(C.StaticAuthRefused, match=r"\(messages.stream\)"):
            async with C._LatchedRetryStream(a.messages, (), {**PARAMS, "extra_headers": STATIC_HEADERS[1]}):
                pass
    asyncio.run(manager_enter())
    assert rec.requests == [] and wire.provider_calls == 0 and wire.exchanges == 0
    clean = {"x-synthetic": "rides"}

    async def rides():
        r = await a.messages.create(**PARAMS, extra_headers=clean)
        await a.messages.count_tokens(model=PARAMS["model"], messages=PARAMS["messages"], extra_headers=clean)
        await a.with_options(max_retries=1).messages.batches.list(extra_headers=clean)
        return r
    assert asyncio.run(rides()).content[0].text == "ok"
    assert [r["path"] for r in rec.requests] == ["/v1/messages", "/v1/messages/count_tokens", "/v1/messages/batches"]
    for r in rec.requests:
        assert r["authorization"] == f"Bearer {EXCHANGED}" and r["x-api-key"] is None and r["x-synthetic"] == "rides", r
    assert wire.provider_calls == 1 and wire.exchanges == 1


def test_the_installed_sdk_still_sends_a_static_per_request_header_in_place_of_the_bearer(wire):
    """The reason, pinned: the bare installed SDK sends a per-request ``Authorization`` / ``X-Api-Key`` in place of the exchanged bearer
    and never invokes the identity-token provider (``AccessTokenAuth._has_static_credential``). Should an SDK stop doing this, the
    layer's refusal is still the right contract and this pin is what to revisit."""
    rec = _Record()
    bare = anthropic.Anthropic(credentials=C.wif_credentials("smoke"), base_url=PIN, max_retries=0,
                               http_client=httpx.Client(transport=httpx.MockTransport(rec)))
    bare.messages.create(**PARAMS, extra_headers=STATIC_HEADERS[0])
    bare.messages.batches.retrieve(BATCH_ID, extra_headers=STATIC_HEADERS[1])
    assert [(r["authorization"], r["x-api-key"]) for r in rec.requests] == [("Bearer " + STATIC_BEARER, None), (None, STATIC_KEY)]
    assert wire.provider_calls == 0 and wire.exchanges == 0


def test_under_op_and_env_extra_headers_and_the_clients_httpx_client_are_the_sdks_own(monkeypatch):
    """The keyed modes are byte-identical: no layer, the SDK's own default client, and a caller's ``extra_headers`` -- a static
    ``Authorization`` included -- handed to the SDK as before (its own behaviour, the key on the wire beside it)."""
    for name in list(os.environ):
        if name.startswith(C.WIF_SCRUB_PREFIX):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "env")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-placeholder-not-a-key")
    rec = _Record()
    c = C.build_client("smoke")
    assert type(c) is anthropic.Anthropic and type(c.messages).__name__ == "Messages" and type(c._client._transport) is not C.PinnedTransport
    on_wire = c.with_options(http_client=httpx.Client(transport=httpx.MockTransport(rec)))
    assert on_wire.messages.create(**PARAMS, extra_headers={"Authorization": "Bearer env-static-placeholder"}).content[0].text == "ok"
    assert rec.requests[0]["authorization"] == "Bearer env-static-placeholder" and rec.requests[0]["x-api-key"] == "synthetic-placeholder-not-a-key"
    a = C.build_async_client("live", timeout=30.0)
    assert type(a) is anthropic.AsyncAnthropic and type(a._client) is httpx.AsyncClient


# ========================================================================================================= ONE RULE, PINNED


def _vault():
    """The vault's module, imported from the repository beside this member (``src/sisteer/vault.py``: stdlib and the SDK only, so it
    imports in the member's own venv); skipped when the member is installed outside the repository."""
    path = REPO / "src" / "sisteer" / "vault.py"
    if not path.is_file():
        pytest.skip("the vault's module is not beside this member (installed outside the repository)")
    sys.path.insert(0, str(REPO / "src"))
    try:
        return importlib.import_module("sisteer.vault")
    finally:
        sys.path.remove(str(REPO / "src"))


def test_the_members_transport_pin_and_header_refusal_are_the_vaults_rule():
    """One rule, two modules: the guard's decisions on a table of destinations, the refusal's words, the static header names and their
    detection, the keepalive socket options, the refusal of a static header in every face, and the cause-chain search are the
    vault's, exactly."""
    vault = _vault()
    assert vault.sdk_httpx() is httpx and C.API_BASE_URL == vault.ANTHROPIC_BASE_URL
    assert tuple(C.STATIC_AUTH_HEADERS) == tuple(vault.STATIC_AUTH_HEADERS)
    assert tuple(C.FEDERATED_SOCKET_OPTIONS) == tuple(vault.FEDERATED_SOCKET_OPTIONS)
    mine = C.PinnedTransport(httpx.MockTransport(lambda r: httpx.Response(200, request=r)), subject="s")
    theirs = vault.PinnedTransport(httpx.MockTransport(lambda r: httpx.Response(200, request=r)), subject="s")
    for url in (PIN + "/v1/messages", "https://api.anthropic.com:443/x", "https://API.anthropic.com/x", OTHER + "/x",
                "http://api.anthropic.com/x", "https://api.anthropic.com:8443/x", "https://api.anthropic.com.example/x",
                "http://127.0.0.1:1/x?secret=no", "https://other-host.example:443/v1/messages/batches/x/results"):
        req = httpx.Request("GET", url)
        outcomes = []
        for guard, err in ((mine, C.DestinationRefused), (theirs, vault.BaseUrlRefused)):
            try:
                guard.handle_request(req)
                outcomes.append("pass")
            except err as e:
                outcomes.append(("refused", str(e).split(": a request addressed to ")[1].split(" reached")[0]))
        assert outcomes[0] == outcomes[1], (url, outcomes)
    for headers in ({"Authorization": "x"}, {"X-Api-Key": "x", "anthropic-beta": "b"}, {"AUTHORIZATION": "x", "x-api-key": "y"},
                    {"x-synthetic": "1"}, {}, None, httpx.Headers({"Authorization": "x"}), ["Authorization"], 7):
        assert C.static_auth_header_names(headers) == vault.static_auth_header_names(headers), headers
        for face in ("default_headers", "extra_headers"):
            outcomes = []
            for fn, err in ((C.refuse_static_auth_headers, C.StaticAuthRefused), (vault.refuse_static_auth_headers, vault.StaticAuthRefused)):
                try:
                    fn(**{face: headers}, subject="s")
                    outcomes.append("pass")
                except err as e:
                    outcomes.append(("refused", str(e).split("was passed (")[1].split(")")[0]))
            assert outcomes[0] == outcomes[1], (face, headers, outcomes)
    donor = httpx.Client(headers={"X-Api-Key": "x"})
    for fn, err in ((C.refuse_static_auth_headers, C.StaticAuthRefused), (vault.refuse_static_auth_headers, vault.StaticAuthRefused)):
        with pytest.raises(err, match="http_client.headers: x-api-key"):
            fn(http_client=donor, subject="s")
    wrapped = anthropic.APIConnectionError(request=httpx.Request("GET", PIN))
    wrapped.__cause__ = RuntimeError("inner")
    wrapped.__cause__.__cause__ = C.DestinationRefused("deep")
    assert C.destination_refusal(wrapped) is wrapped.__cause__.__cause__ and C.destination_refusal(RuntimeError("no")) is None
    theirs_wrapped = anthropic.APIConnectionError(request=httpx.Request("GET", PIN))
    theirs_wrapped.__cause__ = vault.BaseUrlRefused("deep")
    assert vault.destination_refusal(theirs_wrapped) is theirs_wrapped.__cause__ and vault.destination_refusal(wrapped) is None


def test_every_door_of_the_layer_passes_its_kwargs_to_the_check_and_the_constructors_hand_the_sdk_the_members_client():
    src = Path(C.__file__).read_text()
    attempts = re.findall(r"self\._attempt(?:_async)?\(lambda: self\._inner\.(\w+)\(\*args, \*\*kwargs\)(.*)\)", src)
    assert len(attempts) == 2 * 3 + 2 * 6, attempts                              # create / parse / count_tokens + six batches calls, sync and async
    for name, tail in attempts:
        assert tail == f', kwargs, "{name}"', (name, tail)
    assert src.count('self._refuse_request(kwargs, "stream")') == 2 and src.count('self._owner._attempt(enter, self._kwargs, "stream")') == 1
    assert src.count('self._owner._attempt_async(enter, self._kwargs, "stream")') == 1
    assert src.count("self._refuse_request(kwargs or {}, what)") == 2
    assert src.count("refused = destination_refusal(e)") == 2 and src.count("raise refused from e") == 2
    # the constructors: the member's own httpx client under wif, the keyed branches untouched
    assert src.count("http_client=federated_http_client(subject=_federated_subject(kind)),") == 1
    assert src.count("http_client=federated_http_client(async_client=True, timeout=timeout, inner=live_async_transport(),") == 1
    assert src.count("http_client=live_http_client(timeout),") == 1                # the keyed async branch, as before
    assert src.count("max_retries=max_retries,\n") >= 2                           # the keyed branches: the caller's count to the SDK
    for name in ("class PinnedTransport", "class DestinationRefused(", "class StaticAuthRefused(", "def federated_http_client(",
                 "def destination_refusal(", "def refuse_static_auth_headers(", "def static_auth_header_names(", "def live_async_transport(",
                 "class FederatedHttpxClient(", "class FederatedAsyncHttpxClient("):
        assert src.count(name) == 1, name
    assert "refuse_static_auth_headers(default_headers=new_headers" in src        # the copy's static-header check is the one rule
    assert "import sisteer" not in src and "from sisteer" not in src              # the member still imports no sisteer
