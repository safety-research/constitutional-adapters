"""A federated client never accepts inherited STATIC auth through the SDK's custom headers.

The SDK's constructor reads ``ANTHROPIC_CUSTOM_HEADERS`` (and ``ANTHROPIC_WEBHOOK_SIGNING_KEY``) OUTSIDE its ``has_explicit_credential``
guard -- ``credentials=`` stops it reading ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN``, nothing else -- parses the variable into
every request's default headers AFTER its own auth headers, and its ``AccessTokenAuth`` short-circuits on a request that already carries
``X-Api-Key`` or ``Authorization``. So a WIF client built under a launching shell that still exported
``ANTHROPIC_CUSTOM_HEADERS='X-Api-Key: <an obsolete key>'`` sent the stale header on every request and NEVER invoked its identity-token
provider (measured on the committed tree with a mock transport: provider called 0 times, ``X-Api-Key`` on the wire, ``client.api_key``
still None -- so the keyless-stamp assertions could not see it): a 401 on the first request, the run halted, and the mode's
"no ``ANTHROPIC_*`` read" contract broken by the SDK underneath the vault. Round 9's scrub dropped five names; the rule is the PREFIX.

Pinned here, with the REAL SDK over mock transports (no network; the token endpoint and the API are both doubles):
- the vault's sync / async clients under wif (the mode's doors, ``anthropic_client`` / ``anthropic_async_client``) and the direct
  federated constructor under op (``anthropic_client_wif`` IS the wif path, whatever the switch says), built under a shell carrying a
  static ``X-Api-Key`` or ``Authorization`` custom header plus the webhook / profile / config-dir variables: every ``ANTHROPIC_*`` name
  is dropped by name before the SDK reads the environment, the client's custom headers carry no static credential, the first request
  INVOKES the identity-token provider, exchanges it, and goes out with the EXCHANGED bearer and no static header; a ``with_options``
  copy (the batch creator's shape) stays clean;
- the inspect provider's federated client (the Petri legs' auditor / judge) the same, through ``install_inspect_wif``;
- the installed SDK still reads the custom-headers variable outside its guard (the reason the scrub exists; a future SDK that stops
  is the signal to revisit, not a reason to drop the scrub);
- ``scrub_inherited_anthropic_environment`` reads no value (a recording mapping) and drops a name no list ever knew.
The process-environment face (the pin, the graders, the shell scrubs) is ``tests/test_wif_process_environment.py``; the corpus
generator's twin is ``corpus_generator/tests/test_client_wif_inherited_static_auth.py``."""
from __future__ import annotations

import asyncio
import inspect
import os
import sys
from collections.abc import MutableMapping
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sisteer import vault  # noqa: E402

anthropic = pytest.importorskip("anthropic")
from anthropic import _base_client as _sdk_base  # noqa: E402

httpx = getattr(_sdk_base, "httpx", None) or _sdk_base.httpx2   # the SDK's own httpx flavour (httpx on 0.x, httpx2 on 1.x): the transports must be its

IDS = {"ANTHROPIC_FEDERATION_RULE_ID_GCP": "fr_11111111", "ANTHROPIC_ORGANIZATION_ID": "org_22222222",
       "ANTHROPIC_SERVICE_ACCOUNT_ID": "sa_33333333", "ANTHROPIC_WORKSPACE_ID": "ws_44444444"}
#: values with NO credential shape; the assertions are that none of them reaches the wire or is read
STATIC_KEY = "-".join(("static", "placeholder", "never", "sent"))
STATIC_BEARER = "-".join(("static", "bearer", "placeholder"))
SENTINEL = "env-value-" + "never-read-" * 2
EXCHANGED = "exchanged-bearer-synthetic"
GATEWAY = "http://127.0.0.1:1/session-gateway"
MESSAGE = {"id": "msg_synthetic", "type": "message", "role": "assistant", "model": "claude-haiku-4-5",
           "content": [{"type": "text", "text": "ok"}], "stop_reason": "end_turn", "stop_sequence": None,
           "usage": {"input_tokens": 1, "output_tokens": 1}}
PARAMS = dict(model="claude-haiku-4-5", max_tokens=1, messages=[{"role": "user", "content": "ping"}])


def _inherited(custom_headers: str) -> dict:
    """What a launching shell may still carry: the SDK's custom-headers variable (the finding), its webhook / profile / config-dir
    reads, and the five names round 9 knew (every value synthetic)."""
    return {"ANTHROPIC_CUSTOM_HEADERS": custom_headers, "ANTHROPIC_WEBHOOK_SIGNING_KEY": "-".join(("synthetic", "webhook", "placeholder")),
            "ANTHROPIC_PROFILE": "synthetic-profile", "ANTHROPIC_CONFIG_DIR": "/nonexistent/anthropic-config",
            "ANTHROPIC_API_KEY": SENTINEL, "ANTHROPIC_API_KEY_LOWPRIO": SENTINEL, "ANTHROPIC_API_KEY_BATCH": SENTINEL,
            "ANTHROPIC_AUTH_TOKEN": "-".join(("not", "a", "token")), "ANTHROPIC_BASE_URL": GATEWAY}


class _Wire:
    """The two offline ends of a federated request: the token endpoint (counts the exchanges; answers a synthetic bearer) and the API
    (records the auth headers of every request it sees; answers a minimal message)."""

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
                              "beta": request.headers.get("anthropic-beta"), "host": request.url.host, "path": request.url.path})
        return httpx.Response(200, json=MESSAGE, request=request)

    def credentials_class(self):
        wire = self

        class _OfflineCredentials(anthropic.WorkloadIdentityCredentials):
            """The SDK's REAL credentials with the token endpoint doubled and the provider counted (the constructor's only seams)."""

            def __init__(self, **kw):
                kw.setdefault("http_client", httpx.Client(transport=httpx.MockTransport(wire.token_endpoint)))
                kw["identity_token_provider"] = wire.provider
                super().__init__(**kw)

        return _OfflineCredentials

    def sync_api(self):
        return httpx.Client(transport=httpx.MockTransport(self.api))

    def async_api(self):
        return httpx.AsyncClient(transport=httpx.MockTransport(self.api))

    def one_federated_request(self):
        """The one request the test made went out federated: the provider was invoked, the token exchanged, the EXCHANGED bearer on
        the wire, no static header, the pinned host."""
        assert self.provider_calls == 1 and self.exchanges == 1, (self.provider_calls, self.exchanges)
        assert self.requests == [{"x-api-key": None, "authorization": f"Bearer {EXCHANGED}", "beta": self.requests[0]["beta"],
                                  "host": "api.anthropic.com", "path": "/v1/messages"}], self.requests
        assert STATIC_KEY not in str(self.requests) and STATIC_BEARER not in str(self.requests)


def _clean(client) -> None:
    """The client as built: keyless, federated, its custom headers free of any static credential, the webhook variable unread, and
    no ``ANTHROPIC_*`` name left in the process."""
    assert client.api_key is None and client.auth_token is None
    assert isinstance(client.credentials, anthropic.WorkloadIdentityCredentials)
    custom = {k.lower(): v for k, v in (getattr(client, "_custom_headers", {}) or {}).items()}
    assert "x-api-key" not in custom and "authorization" not in custom and "anthropic-beta" not in custom, custom
    assert client.webhook_key is None
    assert str(client.base_url).rstrip("/") == vault.ANTHROPIC_BASE_URL
    assert not [k for k in os.environ if k.startswith(vault.WIF_SCRUB_PREFIX)], sorted(k for k in os.environ if k.startswith("ANTHROPIC_"))


@pytest.fixture
def wire(tmp_path, monkeypatch):
    """wif mode over a synthetic ids directory, the credential doors shut (the secret-manager CLI raises; the key resolver raises),
    the SDK's credentials class replaced by its offline twin, and the mock wire."""
    d = tmp_path / "anthropic"
    d.mkdir()
    for stem in ("loprio", "hiprio", "batch"):
        (d / f"wif-{stem}.env").write_text("# synthetic ids (TESTHOST)\n" + "".join(f"{k}={v}-{stem}\n" for k, v in IDS.items()))
    (d / "wif-lowprio.env").write_text("")
    monkeypatch.setenv(vault.WIF_ENV_DIR_ENV, str(d))
    monkeypatch.delenv(vault.MODAL_MARKER_ENV, raising=False)
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "wif")

    def boom(*a, **k):
        raise AssertionError("no credential door may open while building a federated client")

    monkeypatch.setattr(vault.subprocess, "run", boom)
    monkeypatch.setattr(vault.subprocess, "check_output", boom)
    monkeypatch.setattr(vault, "resolve_secret", boom)
    monkeypatch.setattr(vault, "env_mode_key", boom)
    w = _Wire()
    monkeypatch.setattr(anthropic, "WorkloadIdentityCredentials", w.credentials_class())
    return w


def _carry(monkeypatch, custom_headers: str) -> None:
    for k, v in _inherited(custom_headers).items():
        monkeypatch.setenv(k, v)


# ------------------------------------------------------------------------------------------------- the vault's clients


def test_the_vault_client_under_wif_drops_an_inherited_static_api_key_header_and_its_first_request_federates(wire, monkeypatch):
    _carry(monkeypatch, "X-Api-Key: " + STATIC_KEY + "\nanthropic-beta: synthetic-beta-from-the-shell")   # two lines, as the SDK parses them
    client = vault.anthropic_client("lowprio", max_retries=0, http_client=wire.sync_api())   # the transport seam at the constructor (round 13)
    _clean(client)
    assert vault.client_pool_stamp(client)["auth"] == "wif"
    assert wire.provider_calls == 0, "the token is minted at call time, never at construction"
    on_wire = client.with_options(max_retries=0)                             # the batch creator's shape: a FEDERATED copy (round 14)
    _clean(on_wire)                                                          # a copy re-reads the (now scrubbed) environment: still clean
    r = on_wire.messages.create(**PARAMS)
    assert r.content[0].text == "ok"
    wire.one_federated_request()
    assert "synthetic-beta" not in (wire.requests[0]["beta"] or "")          # the shell's second header line went with the first


def test_the_vault_client_under_wif_drops_an_inherited_static_bearer_header_and_sends_the_exchanged_one(wire, monkeypatch):
    _carry(monkeypatch, "Authorization: Bearer " + STATIC_BEARER)
    client = vault.anthropic_client("lowprio", max_retries=0, http_client=wire.sync_api())
    _clean(client)
    client.messages.create(**PARAMS)
    wire.one_federated_request()                                             # the EXCHANGED bearer, never the shell's


def test_the_vault_async_client_under_wif_drops_the_inherited_static_header_too(wire, monkeypatch):
    _carry(monkeypatch, "X-Api-Key: " + STATIC_KEY)
    client = vault.anthropic_async_client("lowprio", max_retries=0, http_client=wire.async_api())
    _clean(client)
    on_wire = client.copy()                                                  # a federated copy (round 14): clean too
    _clean(on_wire)
    r = asyncio.run(on_wire.messages.create(**PARAMS))
    assert r.content[0].text == "ok"
    wire.one_federated_request()


def test_the_direct_federated_constructor_scrubs_whatever_the_mode_switch_says(wire, monkeypatch):
    """``anthropic_client_wif`` IS the wif path: called directly under op (a port, a test), it still refuses to build on top of an
    inherited static header -- the names are dropped by name first (never read: the recording environment), the request federates."""
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "op")
    _carry(monkeypatch, "X-Api-Key: " + STATIC_KEY)

    class _Recording(MutableMapping):
        """Records every VALUE read of a present ``ANTHROPIC_*`` name (a lookup of an absent one -- the SDK's own ``os.environ.get``
        of the names the scrub already dropped -- returns nothing and is not a read)."""

        def __init__(self, inner):
            self.inner, self.reads = inner, []

        def _note(self, k):
            if isinstance(k, str) and k.startswith("ANTHROPIC_") and k in self.inner:
                self.reads.append(k)

        def __getitem__(self, k):
            self._note(k)
            return self.inner[k]

        def get(self, k, default=None):
            self._note(k)
            return self.inner.get(k, default)

        def __setitem__(self, k, v):
            self.inner[k] = v

        def __delitem__(self, k):
            del self.inner[k]

        def __iter__(self):
            return iter(list(self.inner))

        def __len__(self):
            return len(self.inner)

        def copy(self):
            return dict(self.inner)

    rec = _Recording(os.environ)
    monkeypatch.setattr(os, "environ", rec)
    client = vault.anthropic_client_wif("lowprio", max_retries=0, token_provider=wire.provider, http_client=wire.sync_api())
    assert rec.reads == [], f"an ANTHROPIC_* value was read while building the federated client: {rec.reads}"
    _clean(client)
    client.copy().messages.create(**PARAMS)                                 # a federated copy (round 14), under the recording environment still
    assert rec.reads == []
    wire.one_federated_request()
    assert vault.secrets_mode() == "op"                                      # the switch itself untouched (SISTEER_, not ANTHROPIC_)


# ------------------------------------------------------------------------------------------------- the inspect provider


def test_the_inspect_providers_federated_client_drops_the_inherited_static_header_and_federates(wire, monkeypatch):
    pytest.importorskip("inspect_ai")
    from inspect_ai.model import get_model
    from inspect_ai.model._providers import anthropic as prov
    from sisteer.eval import anthropic_wif as w
    _carry(monkeypatch, "X-Api-Key: " + STATIC_KEY)
    real_cls = prov.AsyncAnthropic
    assert getattr(real_cls, "__sisteer_wif_real__", None) is None, "a stale install leaked from another test"
    try:
        creds = w.install_inspect_wif("lowprio")                             # the vault's credentials: the offline twin under the fixture
        m = get_model("anthropic/claude-sonnet-5-wif-synthetic-r10-static-header", http_client=wire.async_api())
        c = m.api.client
        assert isinstance(c, real_cls) and c.credentials is creds and isinstance(c.messages, w._LatchedMessages)
        _clean(c)
        r = asyncio.run(c.messages.create(**PARAMS))                         # through the latch, to the double
        assert r.content[0].text == "ok"
        wire.one_federated_request()
        # a second shell export after the install (a leg's environment is the drive's; a hand launch may differ): the next client
        # built by the factory scrubs again
        monkeypatch.setenv("ANTHROPIC_CUSTOM_HEADERS", "X-Api-Key: " + STATIC_KEY)
        m2 = get_model("anthropic/claude-sonnet-5-wif-synthetic-r10-static-header-2", http_client=wire.async_api())
        _clean(m2.api.client)
    finally:
        assert w.uninstall_inspect_wif() is True
    assert prov.AsyncAnthropic is real_cls


# ------------------------------------------------------------------------------------------------- the rule and the SDK


def test_the_scrub_drops_every_prefixed_name_by_name_and_reads_no_value():
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
    dropped = vault.scrub_inherited_anthropic_environment(env)
    assert dropped == tuple(sorted({"ANTHROPIC_FUTURE_SDK_SWITCH", *_inherited("")}))        # every prefixed name, sorted, a name no list knew included
    assert set(vault.WIF_INHERITED_ENV_NAMES) <= set(dropped) and list(env) == ["HF_HUB_OFFLINE"]
    assert env.read == [], env.read                                                           # names only: no value was read
    assert vault.scrub_inherited_anthropic_environment(env) == () and list(env) == ["HF_HUB_OFFLINE"]   # idempotent
    assert vault.WIF_SCRUB_PREFIX == "ANTHROPIC_" and all(n.startswith(vault.WIF_SCRUB_PREFIX) for n in vault.WIF_INHERITED_ENV_NAMES)
    assert {"ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_WEBHOOK_SIGNING_KEY"} <= set(vault.WIF_INHERITED_ENV_NAMES)


def test_the_installed_sdk_still_reads_the_custom_headers_variable_outside_its_credential_guard():
    """Why the scrub exists, pinned against the SDK the lock installs: the constructor's read of ANTHROPIC_CUSTOM_HEADERS /
    ANTHROPIC_WEBHOOK_SIGNING_KEY is unconditional, and AccessTokenAuth skips the exchange on a request that already carries
    X-Api-Key / Authorization. A future SDK that changes either is the signal to revisit the scrub -- never a reason to drop it
    unmeasured."""
    for cls in (anthropic.Anthropic, anthropic.AsyncAnthropic):
        src = inspect.getsource(cls.__init__)
        assert 'os.environ.get("ANTHROPIC_CUSTOM_HEADERS")' in src and 'os.environ.get("ANTHROPIC_WEBHOOK_SIGNING_KEY")' in src, cls
        guard, custom = src.index("has_explicit_credential"), src.index('os.environ.get("ANTHROPIC_CUSTOM_HEADERS")')
        assert guard < custom and "if not has_explicit_credential:" in src       # the guard covers the credential reads, not this one
    from anthropic.lib.credentials import _auth
    auth_src = inspect.getsource(_auth.AccessTokenAuth)
    assert 'request.headers.get("X-Api-Key")' in auth_src and 'request.headers.get("Authorization")' in auth_src
