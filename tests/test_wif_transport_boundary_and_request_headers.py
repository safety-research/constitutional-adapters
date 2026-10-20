"""A federated client's destination is ENFORCED where the bytes leave, and a static per-request credential header is refused on
every door of every federated client.

CRITICAL -- a supplied http_client could redirect the federated bearer off-host. Round 12's ``pin_federated_destination`` read a
caller's ``http_client.base_url`` and then handed that caller-owned object to the credentialed SDK. A base URL is not a destination:
measured on bcd10c0eb, ``vault.anthropic_client("lowprio", http_client=httpx.Client(base_url=<the pin>, event_hooks={"request":
[hook]}))`` passed the check and the hook, which runs AFTER httpx's auth flow has put the exchanged bearer on the request, re-pointed
``request.url`` to another host -- the bearer and the request's contents went there. A mount, a subclass's own ``send`` or a transport
of the caller's own do the same. Now the SDK is NEVER handed a caller's httpx client: the vault builds its own
(``vault.federated_http_client``: the SDK's defaults, no hook, no mount, no auth, no environment proxy) whose one transport is the pin
guard (``vault.PinnedTransport``) over httpx's HTTP transport -- it refuses by name any request not addressed to the pin at the last
in-process step before the wire, so an in-SDK rewrite (a ``middleware`` model argument, a cross-host redirect) is refused there too.
A caller's ``http_client`` is accepted only as a bare carrier of a ``MockTransport`` (``vault.transport_seam``: exactly the SDK
flavour's client, its one transport a mock, no mount, no hook, no auth), and only its transport, headers and timeout travel; anything
more about it is refused BY NAME before anything is built. The inspect provider patch's two factories build on the same client.

MAJOR -- per-request authorization headers bypassed WIF authentication. The vault's latched layer forwarded a caller's
``extra_headers`` as given; the SDK's ``AccessTokenAuth`` skips the token exchange for a request already carrying ``X-Api-Key`` /
``Authorization``, so ``messages.create(..., extra_headers={"Authorization": "Bearer <static>"})`` on a client from
``vault.anthropic_client`` under wif issued ZERO exchanges and sent the static header (measured). Round 11 had put the refusal on the
inspect wrapper alone. Now ``vault._LatchedRetry`` refuses it BY NAME before the wire on every door -- ``messages.{create, parse,
count_tokens, stream}`` and ``messages.batches.{create, retrieve, list, cancel, delete, results}``, sync and async.

The round-10 wire harness (``tests/test_wif_inherited_static_auth.py``) and round 12's scripted API are reused: the real SDK over mock
transports of the SDK's own httpx flavour (``httpx`` on anthropic 0.x, ``httpx2`` on 1.x), no network."""
from __future__ import annotations

import asyncio
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sisteer import vault  # noqa: E402

anthropic = pytest.importorskip("anthropic")
from test_wif_inherited_static_auth import (  # noqa: E402  (tests/ is on sys.path under pytest's prepend import mode)
    EXCHANGED,
    IDS,
    MESSAGE,
    PARAMS,
    STATIC_BEARER,
    STATIC_KEY,
    _Wire,
    _clean,
    httpx,
)
from test_wif_federated_destination_and_sdk_retries import (  # noqa: E402
    OTHER_HOSTS,
    PIN,
    SYNTHETIC_KEY,
    _Recorder,
    _ScriptedApi,
)

OTHER = "https://other-host.example"
BATCH_LIST = {"data": [], "has_more": False, "first_id": None, "last_id": None}
STATIC_HEADERS = ({"Authorization": "Bearer " + STATIC_BEARER}, {"X-Api-Key": STATIC_KEY}, {"x-api-key": STATIC_KEY},
                  {"AUTHORIZATION": "Bearer " + STATIC_BEARER})


# ------------------------------------------------------------------------------------------------------------ the harness


@pytest.fixture
def wire(tmp_path, monkeypatch):
    """wif mode over a synthetic ids directory, the credential doors shut, every inherited ``ANTHROPIC_*`` name gone, the SDK's
    credentials class replaced by its offline twin, and the mock wire (the round-10 harness, re-declared: a fixture does not travel
    by import)."""
    import os
    d = tmp_path / "anthropic"
    d.mkdir()
    for stem in ("loprio", "hiprio", "batch"):
        (d / f"wif-{stem}.env").write_text("# synthetic ids (TESTHOST)\n" + "".join(f"{k}={v}-{stem}\n" for k, v in IDS.items()))
    (d / "wif-lowprio.env").write_text("")
    monkeypatch.setenv(vault.WIF_ENV_DIR_ENV, str(d))
    monkeypatch.delenv(vault.MODAL_MARKER_ENV, raising=False)
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "wif")
    for name in list(os.environ):
        if name.startswith(vault.WIF_SCRUB_PREFIX):
            monkeypatch.delenv(name, raising=False)

    def boom(*a, **k):
        raise AssertionError("no credential door may open while building a federated client")

    monkeypatch.setattr(vault.subprocess, "run", boom)
    monkeypatch.setattr(vault.subprocess, "check_output", boom)
    monkeypatch.setattr(vault, "resolve_secret", boom)
    monkeypatch.setattr(vault, "env_mode_key", boom)
    w = _Wire()
    monkeypatch.setattr(anthropic, "WorkloadIdentityCredentials", w.credentials_class())
    return w


@pytest.fixture
def installed(wire):
    """The inspect provider patched for lowprio over the offline credentials; uninstalled after, whatever the test did."""
    pytest.importorskip("inspect_ai")
    from inspect_ai.model._providers import anthropic as prov
    from sisteer.eval import anthropic_wif as w
    real_cls = prov.AsyncAnthropic
    assert getattr(real_cls, "__sisteer_wif_real__", None) is None, "a stale install leaked from another test"
    creds = w.install_inspect_wif("lowprio")
    try:
        yield w, prov, real_cls, creds
    finally:
        assert w.uninstall_inspect_wif() is True
        assert prov.AsyncAnthropic is real_cls


class _Record:
    """A mock API end that records every request's host, auth headers and one named extra header (never a body), and answers a
    message (a batch list for the batches endpoint; a redirect when told to)."""

    def __init__(self, extra: str = "x-synthetic", redirect_to: str | None = None, redirect_once: bool = True):
        self.requests: list = []
        self.extra, self.redirect_to, self.redirect_once = extra, redirect_to, redirect_once

    def __call__(self, request):
        self.requests.append({"host": request.url.host, "path": request.url.path, "query": str(request.url.query, "ascii"),
                              "authorization": request.headers.get("Authorization"), "x-api-key": request.headers.get("X-Api-Key"),
                              self.extra: request.headers.get(self.extra)})
        if self.redirect_to is not None and (not self.redirect_once or len(self.requests) == 1):
            return httpx.Response(307, headers={"location": self.redirect_to}, request=request)
        if "/batches" in request.url.path:
            return httpx.Response(200, json=BATCH_LIST, request=request)
        if request.url.path.endswith("/count_tokens"):
            return httpx.Response(200, json={"input_tokens": 7}, request=request)
        return httpx.Response(200, json=MESSAGE, request=request)

    def sync(self, **kw):
        return httpx.Client(transport=httpx.MockTransport(self), **kw)

    def async_(self, **kw):
        return httpx.AsyncClient(transport=httpx.MockTransport(self), **kw)


def _hook_to_other_host(request):
    request.url = request.url.copy_with(scheme="https", host="other-host.example")


def _sols_donor(rec: _Record, async_client: bool = False):
    """A client whose base URL IS the pin and whose request hook re-points every request after the auth flow."""
    cls = httpx.AsyncClient if async_client else httpx.Client
    return cls(base_url=PIN, event_hooks={"request": [_hook_to_other_host]}, transport=httpx.MockTransport(rec))


def _httpx_client_of(sdk_client):
    return getattr(sdk_client, "_client")


def _own(sdk_client, *, async_client: bool = False, inspect_hooks: bool = False):
    """The httpx client the SDK was handed: the vault's own class over the pin guard, bare of hooks / mounts / auth, pinned. With
    ``inspect_hooks`` the only hooks allowed are inspect_ai's own request / response timing hooks, which its provider appends to the
    SDK client's httpx client AFTER construction (``AnthropicAPI.__init__`` -> ``HttpxHooks``): a trusted dependency's logging, and
    the pin guard runs after every hook anyway."""
    hc = _httpx_client_of(sdk_client)
    mod = vault.sdk_httpx()
    base = mod.AsyncClient if async_client else mod.Client
    assert isinstance(hc, base) and type(hc) is not base and type(hc).__name__ == ("FederatedAsyncHttpxClient" if async_client else "FederatedHttpxClient")
    assert type(hc) is vault._federated_client_class(mod, async_client)
    assert isinstance(hc._transport, vault.PinnedTransport)
    assert hc._mounts == {} and hc.auth is None
    if inspect_hooks:
        for kind, hooks in hc.event_hooks.items():
            for h in hooks:
                owner = getattr(h, "__self__", None)
                assert type(owner).__name__ == "HttpxHooks" and type(owner).__module__.startswith("inspect_ai."), (kind, h)
    else:
        assert hc.event_hooks == {"request": [], "response": []}, hc.event_hooks
    assert str(hc.base_url).rstrip("/") == PIN and hc.follow_redirects is True
    return hc


def _timeout(seconds: float):
    return vault.sdk_httpx().Timeout(seconds)


# ======================================================================================================== THE CRITICAL


def test_sols_hook_rewrite_is_refused_by_name_before_anything_is_built(wire, monkeypatch):
    """The PoC: a client with the pin as its base URL and a request hook to another host passed round 12's check and carried the
    bearer there. Now every door refuses it BY NAME -- naming the hook, the mode, the pin and the seam -- before the SDK's
    constructor runs, before the ids are read, before any token is minted; the recorder behind it sees nothing."""
    rec = _Record()

    def boom(**kw):
        raise AssertionError(f"the SDK's constructor ran for a refused http_client: {sorted(kw)}")

    def no_ids(*a, **k):
        raise AssertionError("the ids file was read for a refused http_client")

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(anthropic, "Anthropic", boom)
        mp.setattr(anthropic, "AsyncAnthropic", boom)
        mp.setattr(vault, "wif_credentials", no_ids)
        for door, is_async in ((vault.anthropic_client, False), (vault.anthropic_async_client, True),
                               (vault.anthropic_client_wif, False)):
            with pytest.raises(vault.BaseUrlRefused) as ei:
                door("lowprio", http_client=_sols_donor(rec, is_async))
            msg = str(ei.value)
            assert "1 event hook(s) (request)" in msg and "never handed a caller's httpx client" in msg, msg
            assert f"{vault.SECRETS_MODE_ENV}=wif" in msg and PIN in msg and "MockTransport(handler)" in msg
            assert isinstance(ei.value, vault.VaultSecretError)
        # the direct constructor IS the wif path whatever the switch says
        mp.setenv(vault.SECRETS_MODE_ENV, "op")
        with pytest.raises(vault.BaseUrlRefused, match="event hook"):
            vault.anthropic_client_wif("lowprio", http_client=_sols_donor(rec), token_provider=wire.provider)
        with pytest.raises(vault.BaseUrlRefused, match="event hook"):
            vault.anthropic_client_wif("lowprio", http_client=_sols_donor(rec, True), async_client=True, token_provider=wire.provider)
    assert rec.requests == [] and wire.provider_calls == 0 and wire.exchanges == 0


class _SendingSubclass(httpx.Client):
    def send(self, request, **kw):  # a subclass's own send: it could go anywhere
        return super().send(request, **kw)


class _ElsewhereTransport(httpx.BaseTransport):
    def __init__(self, rec):
        self.rec = rec

    def handle_request(self, request):
        return self.rec(request)


def _shapes(rec: _Record):
    """Every routing-capable shape of a supplied client, each with the pin (or nothing) as its base URL, and the words the refusal
    must carry."""
    return [
        ("a request hook", _sols_donor(rec), "1 event hook(s) (request)"),
        ("a response hook", httpx.Client(base_url=PIN, event_hooks={"response": [lambda r: None]}, transport=httpx.MockTransport(rec)),
         "1 event hook(s) (response)"),
        ("a mount", httpx.Client(mounts={"all://": httpx.MockTransport(rec)}, transport=httpx.MockTransport(rec)), "1 mount(s)"),
        ("a subclass", _SendingSubclass(transport=httpx.MockTransport(rec)), "its type is"),
        ("httpx's own HTTP transport", httpx.Client(base_url=PIN), f"not {httpx.__name__}.MockTransport"),
        ("a transport of the caller's own", httpx.Client(transport=_ElsewhereTransport(rec)), f"not {httpx.__name__}.MockTransport"),
        ("an auth", httpx.Client(auth=("synthetic-user", "synthetic-pass"), transport=httpx.MockTransport(rec)), "carries an auth"),
    ]


def test_every_routing_capable_shape_of_a_supplied_client_is_refused_by_name_and_nothing_is_built(wire):
    rec = _Record()
    for what, donor, words in _shapes(rec):
        with pytest.raises(vault.BaseUrlRefused) as ei:
            vault.anthropic_client("lowprio", http_client=donor)
        assert words in str(ei.value), (what, str(ei.value))
        assert STATIC_KEY not in str(ei.value) and STATIC_BEARER not in str(ei.value)
    # the async door wants the async flavour: a sync client -- however bare -- is refused by type, an async one with a hook by the hook
    with pytest.raises(vault.BaseUrlRefused, match="its type is"):
        vault.anthropic_async_client("lowprio", http_client=httpx.Client(transport=httpx.MockTransport(rec)))
    with pytest.raises(vault.BaseUrlRefused, match="event hook"):
        vault.anthropic_async_client("lowprio", http_client=_sols_donor(rec, True))
    # transport_seam itself: None for no client; the bare seam's transport; the refusal lists EVERY thing found
    assert vault.transport_seam(None) is None
    bare = httpx.Client(transport=httpx.MockTransport(rec))
    assert vault.transport_seam(bare) is bare._transport
    with pytest.raises(vault.BaseUrlRefused) as ei:
        vault.transport_seam(_SendingSubclass(base_url=PIN, event_hooks={"request": [_hook_to_other_host]}, auth=("u", "p")))
    assert "its type is" in str(ei.value) and "event hook" in str(ei.value) and "an auth" in str(ei.value) and "MockTransport" in str(ei.value)
    assert rec.requests == [] and wire.provider_calls == 0 and wire.exchanges == 0


def test_the_sdk_is_handed_the_vaults_own_client_over_the_donors_transport_never_the_donor(wire):
    """The clean seam: the SDK's httpx client is the vault's own class over the pin guard over the DONOR'S mock transport; the donor
    object itself is never used, so a hook added to it after construction never runs; the request leaves for the pin with the
    exchanged bearer, through the donor's transport. Sync and async doors."""
    donor = wire.sync_api()
    c = vault.anthropic_client("lowprio", max_retries=0, http_client=donor)
    _clean(c)
    hc = _own(c)
    assert hc is not donor and hc._transport.inner is donor._transport and hc.timeout == _timeout(vault.REQUEST_TIMEOUT_S)
    fired = []
    donor.event_hooks = {"request": [lambda r: fired.append(r.url.host)]}            # the caller mutates its own object afterwards
    assert c.messages.create(**PARAMS).content[0].text == "ok"
    wire.one_federated_request()
    assert fired == []
    a_donor = wire.async_api()
    a = vault.anthropic_async_client("lowprio", max_retries=0, http_client=a_donor)
    _clean(a)
    ahc = _own(a, async_client=True)
    assert ahc is not a_donor and ahc._transport.inner is a_donor._transport
    assert asyncio.run(a.messages.create(**PARAMS)).content[0].text == "ok"
    assert len(wire.requests) == 2 and wire.requests[1]["host"] == "api.anthropic.com" and wire.requests[1]["authorization"] == f"Bearer {EXCHANGED}"
    # the donor's (non-credential) headers and its own timeout travel; a static one was refused before this (round 11's rule)
    rec = _Record()
    c2 = vault.anthropic_client("lowprio", max_retries=0, http_client=rec.sync(headers={"x-synthetic": "rides"}, timeout=33.0))
    assert _own(c2).headers["x-synthetic"] == "rides" and _own(c2).timeout == _timeout(vault.REQUEST_TIMEOUT_S)   # the vault's explicit timeout wins
    # with no explicit timeout (the inspect factories' shape) the donor's own travels; a donor at httpx's default gets the SDK's default
    assert vault.federated_http_client(rec.sync(timeout=33.0), subject="the unit").timeout == _timeout(33.0)
    assert vault.federated_http_client(rec.sync(), subject="the unit").timeout == anthropic.DEFAULT_TIMEOUT
    assert vault.federated_http_client(None, subject="the unit").timeout == anthropic.DEFAULT_TIMEOUT
    c2.messages.create(**PARAMS)
    assert rec.requests[-1]["x-synthetic"] == "rides" and rec.requests[-1]["host"] == "api.anthropic.com"
    with pytest.raises(vault.StaticAuthRefused, match="(?i)http_client.headers: x-api-key"):
        vault.anthropic_client("lowprio", http_client=rec.sync(headers={"X-Api-Key": STATIC_KEY}))


def test_the_production_client_is_the_vaults_own_over_the_pin_guard_with_the_sdks_defaults_and_no_environment_proxy(wire, monkeypatch):
    """No donor (every driver in production): the SDK is handed the vault's client over the pin guard over httpx's HTTP transport
    with the SDK's connection limits and keepalive socket options; no mount even under a shell exporting a proxy (the SDK's own
    default client would mount it); the client-level timeout is the vault's; the async door the same over the async transport."""
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1/synthetic-proxy")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1/synthetic-proxy")
    mod = vault.sdk_httpx()
    sdk_default = anthropic.DefaultHttpxClient()
    assert sdk_default._mounts, "the SDK's own default client mounts the shell's proxy (the behaviour the vault's client does not have)"
    c = vault.anthropic_client("lowprio")
    hc = _own(c)
    inner = hc._transport.inner
    assert type(inner) is mod.HTTPTransport and hc.timeout == _timeout(vault.REQUEST_TIMEOUT_S)
    pool = getattr(inner, "_pool", None)
    if pool is not None and hasattr(pool, "_socket_options"):
        assert list(pool._socket_options) == list(sdk_default._transport._pool._socket_options) == list(vault.FEDERATED_SOCKET_OPTIONS)
        assert pool._max_connections == sdk_default._transport._pool._max_connections == anthropic.DEFAULT_CONNECTION_LIMITS.max_connections
    a = vault.anthropic_async_client("lowprio", timeout=77.0)
    ahc = _own(a, async_client=True)
    assert type(ahc._transport.inner) is mod.AsyncHTTPTransport and ahc.timeout == _timeout(77.0)
    # the stamp, the latch and the layer are as round 12 left them
    assert vault.client_pool_stamp(c)["auth"] == "wif" and c.max_retries == 0 and vault.federated_retry_budget(c) == 5
    assert wire.provider_calls == 0


def test_the_pin_guard_refuses_at_the_transport_whatever_re_pointed_the_request_inside_the_process(wire, monkeypatch):
    """The unit: the guard forwards only a request whose scheme, host and port are the pin's; anything else is refused by name with
    the destination and the pin, never the path or a header. End to end through the real SDK: a cross-host redirect the API would
    answer (307 to another host) is refused at the transport -- the follow-up request, bearer and all, never leaves; the layer raises
    it typed, never retries it -- while a same-origin redirect is followed as httpx does. Sync and async."""
    rec = _Record()
    guard = vault.PinnedTransport(httpx.MockTransport(rec), subject="the unit")
    for ok in (PIN + "/v1/messages", "https://api.anthropic.com:443/v1/messages"):
        assert guard.handle_request(httpx.Request("POST", ok)).status_code == 200
    assert len(rec.requests) == 2
    for bad in (OTHER + "/v1/messages", "http://api.anthropic.com/v1/messages", "https://api.anthropic.com:8443/v1/messages",
                "https://api.anthropic.com.example/v1/messages", "http://127.0.0.1:1/double/v1/messages?secret=no"):
        with pytest.raises(vault.BaseUrlRefused) as ei:
            guard.handle_request(httpx.Request("POST", bad, headers={"Authorization": "Bearer " + STATIC_BEARER}))
        msg = str(ei.value)
        assert "the unit" in msg and PIN in msg and "before any byte leaves" in msg
        assert "/v1/messages" not in msg and "secret" not in msg and STATIC_BEARER not in msg
        assert httpx.URL(bad).host in msg
    assert len(rec.requests) == 2
    # end to end: a cross-host 307
    slept: list = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    rec = _Record(redirect_to=OTHER + "/v1/messages")
    c = vault.anthropic_client("lowprio", max_retries=3, http_client=rec.sync())
    with pytest.raises(vault.BaseUrlRefused) as ei:
        c.messages.create(**PARAMS)
    assert "other-host.example" in str(ei.value) and isinstance(ei.value.__cause__, anthropic.APIConnectionError)
    assert [r["host"] for r in rec.requests] == ["api.anthropic.com"] and slept == []          # one request, no retry, nothing elsewhere
    assert vault.client_credential_latch(c).held is None                                        # not a credential rejection
    # a same-origin redirect is followed (two requests, both to the pin)
    rec = _Record(redirect_to="/v1/messages?again=1")
    c = vault.anthropic_client("lowprio", max_retries=0, http_client=rec.sync())
    assert c.messages.create(**PARAMS).content[0].text == "ok"
    assert [(r["host"], r["query"]) for r in rec.requests] == [("api.anthropic.com", ""), ("api.anthropic.com", "again=1")]
    # async
    rec = _Record(redirect_to=OTHER + "/v1/messages")
    a = vault.anthropic_async_client("lowprio", max_retries=3, http_client=rec.async_())
    with pytest.raises(vault.BaseUrlRefused, match="other-host.example"):
        asyncio.run(a.messages.create(**PARAMS))
    assert [r["host"] for r in rec.requests] == ["api.anthropic.com"]
    assert vault.destination_refusal(RuntimeError("not one")) is None


def test_the_inspect_factories_refuse_a_routing_capable_http_client_and_build_on_the_vaults_own(installed, wire):
    """Both patched factories: the hooked client (or any shape above) as the ``http_client`` model argument is refused by name
    before a client exists; the clean seam yields the vault's own httpx client over the donor's transport; the request federates."""
    from inspect_ai.model import get_model
    w, prov, real_cls, creds = installed
    rec = _Record()
    shapes = [("hook", httpx.AsyncClient(base_url=PIN, event_hooks={"request": [_hook_to_other_host]}, transport=httpx.MockTransport(rec)), "event hook"),
              ("mount", httpx.AsyncClient(mounts={"all://": httpx.MockTransport(rec)}, transport=httpx.MockTransport(rec)), "mount"),
              ("transport", httpx.AsyncClient(base_url=PIN), "MockTransport")]
    for tag, donor, words in shapes:
        with pytest.raises(vault.BaseUrlRefused, match=words):
            get_model(f"anthropic/claude-sonnet-5-wif-synthetic-r13-{tag}", http_client=donor)
        with pytest.raises(vault.BaseUrlRefused, match=words):
            prov.AsyncAnthropic(api_key=w.WIF_SENTINEL_KEY, http_client=donor)
    assert rec.requests == [] and wire.requests == [] and wire.provider_calls == 0
    donor = wire.async_api()
    m = get_model("anthropic/claude-sonnet-5-wif-synthetic-r13-seam", http_client=donor)
    c = m.api.client
    assert isinstance(c, real_cls) and c.credentials is creds and c.api_key is None
    hc = _own(c, async_client=True, inspect_hooks=True)                        # inspect's provider appends its own timing hooks after the build
    assert hc is not donor and hc._transport.inner is donor._transport
    assert asyncio.run(c.messages.create(**PARAMS)).content[0].text == "ok"
    wire.one_federated_request()
    # the backstop factory the same (no provider around it: no hooks at all), and a production-shaped client (no model argument) is
    # the vault's own over the HTTP transport
    b = prov.AsyncAnthropic(api_key=w.WIF_SENTINEL_KEY, http_client=wire.async_api())
    assert _own(b, async_client=True)._transport.inner is not None
    p = get_model("anthropic/claude-sonnet-5-wif-synthetic-r13-prod").api.client
    assert type(_own(p, async_client=True, inspect_hooks=True)._transport.inner) is vault.sdk_httpx().AsyncHTTPTransport


class _RewriteToOtherHost(anthropic.Middleware):
    """An SDK middleware model argument that re-points the request's URL inside the SDK, above the httpx client."""

    def handle(self, request, call_next):
        request.options.url = OTHER + "/v1/messages"
        return call_next(request)

    async def handle_async(self, request, call_next):
        request.options.url = OTHER + "/v1/messages"
        return await call_next(request)


def test_an_sdk_middleware_that_rewrites_the_url_is_refused_at_the_transport_on_the_inspect_path(installed, wire):
    """The inspect factories pass model arguments to the SDK's constructor, ``middleware`` among them, and a middleware runs BEFORE
    the httpx client (the SDK honours an absolute URL it sets). Not a door the vault can refuse by name without refusing every
    middleware; the pin guard catches it where it matters: the request, bearer and all, is refused at the transport and the mock
    behind the pin sees nothing."""
    from inspect_ai.model import get_model
    w, prov, real_cls, creds = installed
    m = get_model("anthropic/claude-sonnet-5-wif-synthetic-r13-middleware", http_client=wire.async_api(), middleware=[_RewriteToOtherHost()])
    with pytest.raises(vault.BaseUrlRefused, match="other-host.example"):
        asyncio.run(m.api.client.messages.create(**PARAMS))
    assert wire.requests == [] and w.credential_rejected() is None
    # the same middleware through a vault-shaped SDK client (what a future vault door taking model arguments would build)
    rec = _Record()
    c = anthropic.Anthropic(credentials=vault.wif_credentials("lowprio", token_provider=wire.provider), base_url=PIN, max_retries=0,
                            http_client=vault.federated_http_client(rec.sync(), subject="the unit"), middleware=[_RewriteToOtherHost()])
    with pytest.raises(anthropic.APIConnectionError) as ei:                      # the raw SDK wraps the guard's refusal; the layer re-types it
        c.messages.create(**PARAMS)
    assert isinstance(vault.destination_refusal(ei.value), vault.BaseUrlRefused) and rec.requests == []


def test_under_op_and_env_the_callers_http_client_reaches_the_sdk_untouched(monkeypatch):
    """The keyed modes are byte-identical to round 12: the SDK is handed exactly the caller's client, hooks and all, and no guard."""
    monkeypatch.setattr(anthropic, "Anthropic", _Recorder)
    _Recorder.instances.clear()
    donor = _sols_donor(_Record())
    for mode, var in (("op", "ANTHROPIC_API_KEY_LOWPRIO"), ("env", "ANTHROPIC_API_KEY")):
        monkeypatch.setenv(vault.SECRETS_MODE_ENV, mode)
        monkeypatch.setenv(var, SYNTHETIC_KEY)
        c = vault.anthropic_client("lowprio", http_client=donor)
        assert c.kw == {"api_key": SYNTHETIC_KEY, "base_url": PIN, "max_retries": 5, "timeout": vault.REQUEST_TIMEOUT_S, "http_client": donor}
        assert c.kw["http_client"] is donor
        monkeypatch.delenv(var)


# =========================================================================================================== THE MAJOR


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
        ("messages.batches.retrieve", lambda **h: messages.batches.retrieve("msgbatch_synthetic", **h)),
        ("messages.batches.list", lambda **h: messages.batches.list(**h)),
        ("messages.batches.cancel", lambda **h: messages.batches.cancel("msgbatch_synthetic", **h)),
        ("messages.batches.delete", lambda **h: messages.batches.delete("msgbatch_synthetic", **h)),
        ("messages.batches.results", lambda **h: messages.batches.results("msgbatch_synthetic", **h)),
    ]


def _refused_request(ei, door: str, header: str) -> None:
    msg = str(ei.value)
    assert isinstance(ei.value, vault.StaticAuthRefused) and isinstance(ei.value, ValueError)
    assert f"({door})" in msg and f"extra_headers: {header}" in msg and "AccessTokenAuth" in msg and "pool 'lowprio'" in msg, msg
    assert STATIC_KEY not in msg and STATIC_BEARER not in msg


def test_a_static_per_request_credential_header_is_refused_on_every_door_of_a_federated_sync_client(wire):
    """On bcd10c0eb ``messages.create(..., extra_headers={"Authorization": ...})`` on a vault client sent the static header with zero
    exchanges. Now every door refuses it by name before the wire -- either header, any letter case -- naming the door and the header,
    never a value; the provider is never invoked; a non-credential extra header still rides with the exchanged bearer."""
    rec = _Record()
    c = vault.anthropic_client("lowprio", max_retries=0, http_client=rec.sync())
    for door, call in _doors(c.messages):
        for headers in STATIC_HEADERS:
            with pytest.raises(vault.StaticAuthRefused) as ei:
                call(extra_headers=headers)
            _refused_request(ei, door, next(iter(headers)))
    # the stream manager's own enter refuses too (a manager built around such kwargs, not through stream())
    with pytest.raises(vault.StaticAuthRefused, match=r"\(messages.stream\)"):
        with vault._LatchedRetryStream(c.messages, (), {**PARAMS, "extra_headers": STATIC_HEADERS[0]}):
            pass
    assert rec.requests == [] and wire.provider_calls == 0 and wire.exchanges == 0
    # a clean extra header rides every door's request with the exchanged bearer (the refusal is the two names, nothing wider)
    clean = {"x-synthetic": "rides"}
    assert c.messages.create(**PARAMS, extra_headers=clean).content[0].text == "ok"
    c.messages.parse(**PARAMS, extra_headers=clean)
    c.messages.count_tokens(model=PARAMS["model"], messages=PARAMS["messages"], extra_headers=clean)
    c.messages.batches.list(extra_headers=clean)
    assert [r["path"] for r in rec.requests] == ["/v1/messages", "/v1/messages", "/v1/messages/count_tokens", "/v1/messages/batches"]
    for r in rec.requests:
        assert r["host"] == "api.anthropic.com" and r["authorization"] == f"Bearer {EXCHANGED}" and r["x-api-key"] is None and r["x-synthetic"] == "rides", r
    assert wire.provider_calls == 1 and wire.exchanges == 1


def test_a_static_per_request_credential_header_is_refused_on_every_door_of_a_federated_async_client(wire):
    rec = _Record()
    a = vault.anthropic_async_client("lowprio", max_retries=0, http_client=rec.async_())

    def run(call, **h):
        out = call(**h)
        return asyncio.run(out) if asyncio.iscoroutine(out) else out

    for door, call in _doors(a.messages):
        for headers in STATIC_HEADERS:
            with pytest.raises(vault.StaticAuthRefused) as ei:
                run(call, extra_headers=headers)
            _refused_request(ei, door, next(iter(headers)))

    async def enter_bad():
        async with vault._LatchedRetryStream(a.messages, (), {**PARAMS, "extra_headers": STATIC_HEADERS[1]}):
            pass

    with pytest.raises(vault.StaticAuthRefused, match=r"\(messages.stream\)"):
        asyncio.run(enter_bad())
    assert rec.requests == [] and wire.provider_calls == 0 and wire.exchanges == 0
    clean = {"x-synthetic": "rides"}
    assert asyncio.run(a.messages.create(**PARAMS, extra_headers=clean)).content[0].text == "ok"
    asyncio.run(a.messages.count_tokens(model=PARAMS["model"], messages=PARAMS["messages"], extra_headers=clean))
    asyncio.run(a.messages.batches.list(extra_headers=clean))
    assert [r["path"] for r in rec.requests] == ["/v1/messages", "/v1/messages/count_tokens", "/v1/messages/batches"]
    for r in rec.requests:
        assert r["authorization"] == f"Bearer {EXCHANGED}" and r["x-api-key"] is None and r["x-synthetic"] == "rides", r


def test_the_inspect_providers_client_refuses_it_in_the_vaults_layer_under_its_own_wrapper(installed, wire):
    """Round 11's refusal lives on the inspect wrapper; the vault's layer under it now carries the same, so a request that reached
    the layer by any other path would be refused too: both layers name the header, nothing is sent."""
    w, prov, real_cls, creds = installed
    c = prov.AsyncAnthropic(api_key=w.WIF_SENTINEL_KEY, http_client=wire.async_api())
    assert isinstance(c.messages, w._LatchedMessages) and isinstance(c.messages._inner, vault.AsyncLatchedRetryMessages)
    with pytest.raises(vault.StaticAuthRefused, match="extra_headers: X-Api-Key"):
        asyncio.run(c.messages.create(**PARAMS, extra_headers={"X-Api-Key": STATIC_KEY}))
    with pytest.raises(vault.StaticAuthRefused, match=r"\(messages.create\).*extra_headers: Authorization"):
        asyncio.run(c.messages._inner.create(**PARAMS, extra_headers={"Authorization": "Bearer " + STATIC_BEARER}))
    assert wire.requests == [] and wire.provider_calls == 0 and w.credential_rejected() is None


def test_the_installed_sdk_still_sends_a_static_per_request_header_in_place_of_the_bearer(wire):
    """The reason the refusal exists, measured on a RAW SDK client (no vault layer): a per-request static ``Authorization`` is kept
    and the token exchange skipped. A future SDK that refuses the contradiction itself is the signal to revisit, never to drop it."""
    rec = _Record()
    raw = anthropic.Anthropic(credentials=vault.wif_credentials("lowprio", token_provider=wire.provider), base_url=PIN, max_retries=0,
                              http_client=rec.sync())
    raw.messages.create(**PARAMS, extra_headers={"Authorization": "Bearer " + STATIC_BEARER})
    assert rec.requests[0]["authorization"] == "Bearer " + STATIC_BEARER and wire.exchanges == 0 and wire.provider_calls == 0
    rec.requests.clear()
    raw.messages.create(**PARAMS, extra_headers={"X-Api-Key": STATIC_KEY})
    assert rec.requests[0]["x-api-key"] == STATIC_KEY and rec.requests[0]["authorization"] is None and wire.exchanges == 0


# ========================================================================================================= THE SOURCE PINS


def test_every_door_of_the_layer_passes_its_kwargs_to_the_check_and_the_constructors_hand_the_sdk_the_vaults_client():
    vsrc = (REPO / "src/sisteer/vault.py").read_text()
    # every attempt of the layer is handed the call's kwargs and its door's name: a door that forgot would forward a static header
    attempts = re.findall(r"self\._attempt(?:_async)?\(lambda: self\._inner\.(\w+)\(\*args, \*\*kwargs\)(.*)\)", vsrc)
    assert len(attempts) == 2 * 3 + 2 * 6, attempts                              # create / parse / count_tokens + six batches calls, sync and async
    for name, tail in attempts:
        assert tail == f', kwargs, "{name}"', (name, tail)
    assert vsrc.count('self._refuse_request(kwargs, "stream")') == 2 and vsrc.count('self._owner._attempt(enter, self._kwargs, "stream")') == 1
    assert vsrc.count('self._owner._attempt_async(enter, self._kwargs, "stream")') == 1
    assert "self._refuse_request(kwargs or {}, what)" in vsrc
    # the constructors: the vault's own httpx client, never the caller's
    assert "own = federated_http_client(http_client, async_client=async_client, timeout=timeout, subject=subject)" in vsrc
    assert 'extra = {"http_client": own}' in vsrc
    assert "client = cls(credentials=creds, base_url=pinned, max_retries=0, timeout=timeout, **extra)" in vsrc
    for name in ("class PinnedTransport", "def transport_seam(", "def federated_http_client(", "def destination_refusal(", "def sdk_httpx("):
        assert vsrc.count(name) == 1, name
    wsrc = (REPO / "src/sisteer/eval/anthropic_wif.py").read_text()
    assert wsrc.count("vault.federated_http_client(") == 2                        # the backstop factory, the model-args factory
    assert 'kwargs["http_client"] = vault.federated_http_client(kwargs.get("http_client"), async_client=True' in wsrc
    assert 'extra["http_client"] = vault.federated_http_client(extra.get("http_client"), async_client=True' in wsrc


def test_no_tree_code_hands_a_federated_constructor_an_http_client():
    """The seam is the tests' alone: no file under src/ scripts/ reproduce/ passes ``http_client=`` to a vault door or to
    ``get_model`` (the corpus generator's live engine builds ITS OWN httpx client inside its own constructor, which takes none)."""
    out = subprocess.run(["git", "ls-files", "-z", "--", "src/", "scripts/", "reproduce/"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    doors = re.compile(r"(anthropic_client_wif|anthropic_async_client|anthropic_client|make_client|get_model)\([^)]*http_client=")
    hits = []
    for rel in sorted(p for p in out.split("\0") if p.endswith(".py") and "/tests/" not in p):
        if rel in ("src/sisteer/vault.py", "src/sisteer/eval/anthropic_wif.py"):
            continue
        src = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        for m in doors.finditer(src):
            hits.append(f"{rel}:{src.count(chr(10), 0, m.start()) + 1}")
    assert hits == [], hits
