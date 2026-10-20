"""The corpus generator's wif clients never accept inherited STATIC auth through the SDK's custom headers (the twin of
``tests/test_wif_inherited_static_auth.py`` for this member, which imports no sisteer).

The SDK's constructor reads ``ANTHROPIC_CUSTOM_HEADERS`` / ``ANTHROPIC_WEBHOOK_SIGNING_KEY`` outside its credential guard, parses the
former into every request's headers after its own auth headers, and its ``AccessTokenAuth`` skips the token exchange on a request that
already carries ``X-Api-Key`` / ``Authorization``: ``build_client`` / ``build_async_client`` under wif on a shell exporting such a
header sent the stale header and never invoked the identity-token provider (measured: provider 0, ``X-Api-Key`` on the wire,
``api_key`` None). Both constructors now run ``scrub_inherited_anthropic_environment`` -- every ``ANTHROPIC_*`` name dropped by name,
no value read -- before the SDK's constructor. Real SDK, mock transports, no network."""
from __future__ import annotations

import asyncio
import os
from collections.abc import MutableMapping
from pathlib import Path

import anthropic
import pytest
from anthropic import _base_client as _sdk_base

from corpus_generator import client as C

httpx = getattr(_sdk_base, "httpx", None) or _sdk_base.httpx2   # the SDK's own httpx flavour (httpx2 on 1.x): the transports must be its

IDS = {"ANTHROPIC_FEDERATION_RULE_ID_GCP": "fr_11111111", "ANTHROPIC_ORGANIZATION_ID": "org_22222222",
       "ANTHROPIC_SERVICE_ACCOUNT_ID": "sa_33333333", "ANTHROPIC_WORKSPACE_ID": "ws_44444444"}
STATIC_KEY = "-".join(("static", "placeholder", "never", "sent"))      # no credential shape; the assertion is that it never reaches the wire
STATIC_BEARER = "-".join(("static", "bearer", "placeholder"))
SENTINEL = "env-value-" + "never-read-" * 2
EXCHANGED = "exchanged-bearer-synthetic"
MESSAGE = {"id": "msg_synthetic", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
           "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
           "usage": {"input_tokens": 1, "output_tokens": 1}}
PARAMS = dict(model="claude-haiku-4-5", max_tokens=1, messages=[{"role": "user", "content": "ping"}])


def _inherited(custom_headers: str) -> dict:
    return {"ANTHROPIC_CUSTOM_HEADERS": custom_headers, "ANTHROPIC_WEBHOOK_SIGNING_KEY": "-".join(("synthetic", "webhook", "placeholder")),
            "ANTHROPIC_PROFILE": "synthetic-profile", "ANTHROPIC_CONFIG_DIR": "/nonexistent/anthropic-config",
            "ANTHROPIC_API_KEY": SENTINEL, "ANTHROPIC_API_KEY_LOWPRIO": SENTINEL, "ANTHROPIC_API_KEY_BATCH": SENTINEL,
            "ANTHROPIC_AUTH_TOKEN": "-".join(("not", "a", "token")), "ANTHROPIC_BASE_URL": "http://127.0.0.1:1/mock-gateway"}


class _Wire:
    def __init__(self):
        self.provider_calls = 0
        self.exchanges = 0
        self.requests: list = []

    def provider(self) -> str:
        self.provider_calls += 1
        return "header.payload.signature"

    def token_endpoint(self, request):
        assert request.url.path.endswith("/v1/oauth/token"), request.url
        self.exchanges += 1
        return httpx.Response(200, json={"access_token": EXCHANGED, "token_type": "Bearer", "expires_in": 600}, request=request)

    def api(self, request):
        self.requests.append({"x-api-key": request.headers.get("X-Api-Key"), "authorization": request.headers.get("Authorization"),
                              "host": request.url.host, "path": request.url.path})
        return httpx.Response(200, json=MESSAGE, request=request)

    def credentials_class(self):
        wire = self

        class _OfflineCredentials(anthropic.WorkloadIdentityCredentials):
            def __init__(self, **kw):
                kw.setdefault("http_client", httpx.Client(transport=httpx.MockTransport(wire.token_endpoint)))
                kw["identity_token_provider"] = wire.provider
                super().__init__(**kw)

        return _OfflineCredentials

    def one_federated_request(self):
        assert self.provider_calls == 1 and self.exchanges == 1, (self.provider_calls, self.exchanges)
        assert self.requests == [{"x-api-key": None, "authorization": f"Bearer {EXCHANGED}", "host": "api.anthropic.com",
                                  "path": "/v1/messages"}], self.requests
        assert STATIC_KEY not in str(self.requests) and STATIC_BEARER not in str(self.requests)


def _clean(client) -> None:
    assert client.api_key is None and client.auth_token is None
    assert isinstance(client.credentials, anthropic.WorkloadIdentityCredentials)
    custom = {k.lower(): v for k, v in (getattr(client, "_custom_headers", {}) or {}).items()}
    assert "x-api-key" not in custom and "authorization" not in custom, custom
    assert client.webhook_key is None
    assert str(client.base_url).rstrip("/") == C.API_BASE_URL
    assert not [k for k in os.environ if k.startswith(C.WIF_SCRUB_PREFIX)], sorted(k for k in os.environ if k.startswith("ANTHROPIC_"))


@pytest.fixture
def wire(tmp_path, monkeypatch):
    """wif mode with every credential door nailed shut (as test_client_wif_mode's fixture), the SDK's credentials class replaced by
    its offline twin, and the mock wire."""
    d = tmp_path / "anthropic"
    d.mkdir()
    for stem, suffix in (("loprio", ""), ("batch", "-batch"), ("hiprio", "-hi")):
        (d / f"wif-{stem}.env").write_text("".join(f"{k}={v}{suffix}\n" for k, v in IDS.items()))
    (d / "wif-lowprio.env").write_text("")
    monkeypatch.setenv(C.WIF_ENV_DIR_ENV, str(d))
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "wif")
    monkeypatch.delenv(C.MODAL_MARKER_ENV, raising=False)
    (tmp_path / "empty-bin").mkdir()
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))

    def boom(*a, **k):  # pragma: no cover - the assertion IS the test
        raise AssertionError(f"the secret-manager CLI must never run under wif: {a}")

    monkeypatch.setattr(C.subprocess, "run", boom)
    w = _Wire()
    monkeypatch.setattr(anthropic, "WorkloadIdentityCredentials", w.credentials_class())
    return w


def _carry(monkeypatch, custom_headers: str) -> None:
    for k, v in _inherited(custom_headers).items():
        monkeypatch.setenv(k, v)


def _on_wire(client, handler) -> None:
    """The pin guard's INNER transport swapped for the double ON THE CLIENT BUILT (round 15: the SDK is handed the member's own httpx
    client over ``PinnedTransport``; the constructors take no ``http_client`` and, since round 14, a federated copy refuses one -- so
    the seam is the guard's inner transport, and every request of the client, and of every copy made before or after this (a copy
    shares its parent's client), still crosses the pin)."""
    guard = client._client._transport
    assert type(guard) is C.PinnedTransport, type(guard)
    guard.inner = httpx.MockTransport(handler)


def test_build_client_under_wif_drops_an_inherited_static_api_key_header_and_its_first_request_federates(wire, monkeypatch):
    _carry(monkeypatch, "X-Api-Key: " + STATIC_KEY)
    client = C.build_client("batch", max_retries=0)
    _clean(client)
    assert wire.provider_calls == 0                                          # minted at call time, never at construction
    _on_wire(client, wire.api)
    creator = client.with_options(max_retries=0)                             # batch.py's creator: a federated copy keeps the credentials and clean headers
    _clean(creator)
    r = creator.messages.create(**PARAMS)
    assert r.content[0].text == "ok"
    wire.one_federated_request()


def test_build_client_under_wif_drops_an_inherited_static_bearer_header_and_sends_the_exchanged_one(wire, monkeypatch):
    _carry(monkeypatch, "Authorization: Bearer " + STATIC_BEARER)
    client = C.build_client("smoke", max_retries=0)
    _on_wire(client, wire.api)
    _clean(client)
    client.messages.create(**PARAMS)
    wire.one_federated_request()


def test_build_async_client_under_wif_drops_the_inherited_static_header_too(wire, monkeypatch):
    _carry(monkeypatch, "X-Api-Key: " + STATIC_KEY)
    client = C.build_async_client("live", max_retries=0, timeout=30.0)
    _clean(client)
    _on_wire(client, wire.api)
    r = asyncio.run(client.copy().messages.create(**PARAMS))                  # a federated copy (round 14): clean, the same transport
    assert r.content[0].text == "ok"
    wire.one_federated_request()


def test_the_scrub_drops_every_prefixed_name_by_name_reads_no_value_and_the_old_modes_do_not_run_it(monkeypatch, tmp_path):
    class _Recorder(MutableMapping):
        def __init__(self, **items):
            self._d, self.read = dict(items), []

        def __getitem__(self, k):
            self.read.append(k)
            return self._d[k]

        def __setitem__(self, k, v):
            self._d[k] = v

        def __delitem__(self, k):
            del self._d[k]

        def __iter__(self):
            return iter(list(self._d))

        def __len__(self):
            return len(self._d)

    env = _Recorder(ANTHROPIC_FUTURE_SDK_SWITCH="x", HF_HUB_OFFLINE="1", **_inherited("X-Api-Key: " + STATIC_KEY))
    dropped = C.scrub_inherited_anthropic_environment(env)
    assert dropped == tuple(sorted({"ANTHROPIC_FUTURE_SDK_SWITCH", *_inherited("")})) and list(env) == ["HF_HUB_OFFLINE"]
    assert env.read == [] and C.scrub_inherited_anthropic_environment(env) == ()
    assert C.WIF_SCRUB_PREFIX == "ANTHROPIC_"
    # op / env byte-identical: a keyed client leaves the environment as it found it (the scrub is the wif branches' alone)
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "env")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-mode-" + "placeholder-" * 2)
    monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Custom: synthetic")
    c = C.build_client("batch")
    assert c.api_key is not None and os.environ["ANTHROPIC_CUSTOM_HEADERS"] == "X-Custom: synthetic"
    src = Path(C.__file__).read_text()
    assert src.count("scrub_inherited_anthropic_environment()") == 3            # the two wif branches and the federated copy (round 14), nowhere else
