"""A federated client never accepts an EXPLICIT static credential header either.

Round 10 scrubbed the INHERITED face: ``ANTHROPIC_CUSTOM_HEADERS`` parsed into every request's headers. This is the explicit face
of the same SDK behaviour: ``get_model("anthropic/...", default_headers={"X-Api-Key": ...})`` (or an ``http_client`` carrying such
a header, or a request's ``extra_headers``) reaches the SDK's constructor through inspect's ``model_args`` and both patched
factories passed it on. The SDK keeps a static ``X-Api-Key`` / ``Authorization`` beside ``credentials=`` (``client.api_key`` None,
``WorkloadIdentityCredentials`` set) and its ``AccessTokenAuth`` then skips the token exchange for a request already carrying one:
measured on the committed tree with a mock transport -- the static header on the wire, the identity-token provider invoked 0 times --
so an Inspect leg would send a static credential instead of federating and fail with a 401 / 403 the leg then stops on.

Pinned here, with the REAL SDK over the round-10 mock wire (no network):
- both factories of ``install_inspect_wif`` REFUSE BY NAME (``vault.StaticAuthRefused``, a ``ValueError``) an explicit
  ``default_headers`` / ``http_client`` carrying ``X-Api-Key`` or ``Authorization`` -- in any letter case -- before any client is
  built and before any request; the provider's own OAuth-beta header is still stripped, not refused;
- a NON-credential explicit header (``anthropic-beta``) still rides the federated request: the exchanged bearer on the wire, the
  beta beside it (the refusal is the two names, nothing wider);
- a per-request ``extra_headers`` carrying either name is refused before the wire, by the latched ``messages`` wrapper;
- the vault's rule (``static_auth_header_names`` / ``refuse_static_auth_headers``) names the header and the place, never a value;
- the installed SDK still keeps an explicit static header beside ``credentials=`` and sends it in place of the exchanged bearer --
  the reason the refusal exists (a future SDK that refuses the contradiction itself is the signal to revisit, never to drop it).
The inherited face is ``tests/test_wif_inherited_static_auth.py``, whose wire harness this module reuses."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from sisteer import vault  # noqa: E402

anthropic = pytest.importorskip("anthropic")
from test_wif_inherited_static_auth import (  # noqa: E402  (tests/ is on sys.path under pytest's prepend import mode)
    IDS,
    PARAMS,
    STATIC_BEARER,
    STATIC_KEY,
    _Wire,
    _clean,
    httpx,
)


@pytest.fixture
def wire(tmp_path, monkeypatch):
    """wif mode over a synthetic ids directory, the credential doors shut, the SDK's credentials class replaced by its offline
    twin, and the mock wire (the round-10 harness, re-declared here: a fixture does not travel by import)."""
    d = tmp_path / "anthropic"
    d.mkdir()
    for stem in ("loprio", "hiprio", "batch"):
        (d / f"wif-{stem}.env").write_text("# synthetic ids (TESTHOST)\n" + "".join(f"{k}={v}-{stem}\n" for k, v in IDS.items()))
    (d / "wif-lowprio.env").write_text("")
    monkeypatch.setenv(vault.WIF_ENV_DIR_ENV, str(d))
    monkeypatch.delenv(vault.MODAL_MARKER_ENV, raising=False)
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "wif")
    for name in list(__import__("os").environ):
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


def _refused(ei, *places: str) -> None:
    """The refusal names the place and the header, the rule, and never a value."""
    msg = str(ei.value)
    assert isinstance(ei.value, vault.StaticAuthRefused) and isinstance(ei.value, ValueError)
    for p in places:
        assert p in msg, (p, msg)
    assert "AccessTokenAuth" in msg and f"{vault.SECRETS_MODE_ENV}={vault.WIF_MODE}" in msg and "refusing" in msg
    assert STATIC_KEY not in msg and STATIC_BEARER not in msg


def _nothing_sent(wire) -> None:
    assert wire.requests == [] and wire.provider_calls == 0 and wire.exchanges == 0, (wire.requests, wire.provider_calls)


# ------------------------------------------------------------------------------------- the provider's client factory (model_args)


def test_an_explicit_static_api_key_header_in_the_model_args_is_refused_by_name_before_any_client_or_request(installed, wire):
    from inspect_ai.model import get_model
    with pytest.raises(ValueError) as ei:
        get_model("anthropic/claude-sonnet-5-wif-synthetic-r11-explicit-key", default_headers={"X-Api-Key": STATIC_KEY},
                  http_client=wire.async_api())
    _refused(ei, "default_headers: X-Api-Key", "model_args", "claude-sonnet-5-wif-synthetic-r11-explicit-key")
    _nothing_sent(wire)


@pytest.mark.parametrize("header", [{"Authorization": "Bearer " + STATIC_BEARER}, {"authorization": "Bearer " + STATIC_BEARER},
                                    {"x-api-key": STATIC_KEY}, {"X-API-KEY": STATIC_KEY},
                                    {"anthropic-beta": "synthetic", "X-Api-Key": STATIC_KEY}])
def test_a_static_bearer_or_any_letter_case_of_either_name_is_refused(installed, wire, header):
    from inspect_ai.model import get_model
    with pytest.raises(ValueError) as ei:
        get_model("anthropic/claude-sonnet-5-wif-synthetic-r11-case", default_headers=header, http_client=wire.async_api())
    name = next(k for k in header if k.lower() in vault.STATIC_AUTH_HEADERS)
    _refused(ei, f"default_headers: {name}")
    _nothing_sent(wire)


def test_an_explicit_http_client_carrying_a_static_header_is_refused_too(installed, wire):
    """httpx merges a client's own headers into every request it builds, so an ``http_client`` model argument is the same door."""
    from inspect_ai.model import get_model
    carrier = httpx.AsyncClient(headers={"X-Api-Key": STATIC_KEY}, transport=httpx.MockTransport(wire.api))
    with pytest.raises(ValueError) as ei:
        get_model("anthropic/claude-sonnet-5-wif-synthetic-r11-http-client", http_client=carrier)
    _refused(ei, "http_client.headers: x-api-key")
    _nothing_sent(wire)


def test_a_non_credential_explicit_header_still_rides_the_federated_request(installed, wire):
    """The refusal is the two names, nothing wider: an explicit ``anthropic-beta`` builds, and the one request goes out with the
    EXCHANGED bearer and that beta beside it -- through the mock transport."""
    from inspect_ai.model import get_model
    w, prov, real_cls, creds = installed
    m = get_model("anthropic/claude-sonnet-5-wif-synthetic-r11-beta", default_headers={"anthropic-beta": "synthetic-r11-beta"},
                  http_client=wire.async_api())
    c = m.api.client
    assert isinstance(c, real_cls) and c.credentials is creds and isinstance(c.messages, w._LatchedMessages)
    assert c.api_key is None and c.auth_token is None and str(c.base_url).rstrip("/") == vault.ANTHROPIC_BASE_URL
    custom = {k.lower(): v for k, v in (getattr(c, "_custom_headers", {}) or {}).items()}
    assert custom == {"anthropic-beta": "synthetic-r11-beta"}, custom                      # the explicit beta kept; no credential name
    r = asyncio.run(c.messages.create(**PARAMS))
    assert r.content[0].text == "ok"
    wire.one_federated_request()
    assert "synthetic-r11-beta" in (wire.requests[0]["beta"] or "")


# ------------------------------------------------------------------------------------- the backstop factory (the rebound AsyncAnthropic)


def test_the_backstop_factory_refuses_the_same_names_and_still_strips_the_oauth_beta(installed, wire):
    w, prov, real_cls, creds = installed
    with pytest.raises(ValueError) as ei:
        prov.AsyncAnthropic(api_key=w.WIF_SENTINEL_KEY, default_headers={"X-Api-Key": STATIC_KEY}, http_client=wire.async_api())
    _refused(ei, "default_headers: X-Api-Key", "AsyncAnthropic factory")
    with pytest.raises(ValueError) as ei:
        prov.AsyncAnthropic(api_key=w.WIF_SENTINEL_KEY, http_client=httpx.AsyncClient(headers={"Authorization": "Bearer " + STATIC_BEARER},
                                                                                       transport=httpx.MockTransport(wire.api)))
    _refused(ei, "http_client.headers: authorization")
    _nothing_sent(wire)
    # the provider's own OAuth branch header is stripped as before (round 1), never refused: a legitimate construction path
    c = prov.AsyncAnthropic(api_key=w.WIF_SENTINEL_KEY, auth_token="not-a-token", default_headers={"anthropic-beta": "oauth-2025-04-20"},
                            http_client=wire.async_api())
    assert isinstance(c, real_cls) and c.credentials is creds
    _clean(c)
    asyncio.run(c.messages.create(**PARAMS))
    wire.one_federated_request()


# ------------------------------------------------------------------------------------- per request


def test_a_per_request_static_header_is_refused_before_the_wire_and_a_clean_one_federates(installed, wire):
    from inspect_ai.model import get_model
    c = get_model("anthropic/claude-sonnet-5-wif-synthetic-r11-per-request", http_client=wire.async_api()).api.client
    for extra in ({"X-Api-Key": STATIC_KEY}, {"authorization": "Bearer " + STATIC_BEARER}):
        with pytest.raises(ValueError) as ei:
            asyncio.run(c.messages.create(**PARAMS, extra_headers=extra))
        _refused(ei, "extra_headers: " + next(iter(extra)), "a request of the inspect Anthropic provider's federated client")
        with pytest.raises(ValueError):
            asyncio.run(c.messages.count_tokens(model=PARAMS["model"], messages=PARAMS["messages"], extra_headers=extra))
        with pytest.raises(ValueError):
            c.messages.stream(**PARAMS, extra_headers=extra)
    _nothing_sent(wire)
    r = asyncio.run(c.messages.create(**PARAMS, extra_headers={"anthropic-beta": "synthetic-r11-request-beta"}))
    assert r.content[0].text == "ok"
    wire.one_federated_request()
    assert "synthetic-r11-request-beta" in (wire.requests[0]["beta"] or "")


# ------------------------------------------------------------------------------------- the rule, and the SDK


def test_the_vault_rule_names_the_header_and_the_place_never_a_value():
    assert vault.STATIC_AUTH_HEADERS == ("x-api-key", "authorization")
    assert vault.static_auth_header_names({"X-Api-Key": STATIC_KEY, "Accept": "x", "authorization": "b"}) == ("X-Api-Key", "authorization")
    assert vault.static_auth_header_names(httpx.Headers({"X-API-Key": STATIC_KEY})) == ("x-api-key",)          # httpx lower-cases its names
    assert vault.static_auth_header_names({"anthropic-beta": "x"}) == () and vault.static_auth_header_names(None) == ()
    assert vault.static_auth_header_names(object()) == () and vault.static_auth_header_names(42) == ()        # not a mapping: nothing to name
    vault.refuse_static_auth_headers(default_headers={"anthropic-beta": "x"}, http_client=httpx.Client(), extra_headers={"x-irid": "1"})
    with pytest.raises(vault.StaticAuthRefused) as ei:
        vault.refuse_static_auth_headers(default_headers={"X-Api-Key": STATIC_KEY}, http_client=httpx.Client(headers={"Authorization": "Bearer " + STATIC_BEARER}),
                                         extra_headers={"x-api-key": STATIC_KEY}, subject="synthetic subject")
    msg = str(ei.value)
    assert msg.startswith("synthetic subject: ")
    assert "default_headers: X-Api-Key; http_client.headers: authorization; extra_headers: x-api-key" in msg
    assert STATIC_KEY not in msg and STATIC_BEARER not in msg
    assert issubclass(vault.StaticAuthRefused, ValueError)


def test_the_installed_sdk_keeps_an_explicit_static_header_beside_credentials_and_sends_it_in_place_of_the_bearer(wire):
    """Why the refusal exists, measured against the SDK the lock installs: ``AsyncAnthropic(credentials=..., default_headers=
    {"X-Api-Key": ...})`` builds (``api_key`` None, the credentials set), and its one request carries the static header with the
    identity-token provider invoked 0 times and nothing exchanged. A future SDK that refuses the contradiction itself is the
    signal to revisit the vault's refusal -- never a reason to drop it unmeasured."""
    creds = anthropic.WorkloadIdentityCredentials(federation_rule_id="fr_11111111", organization_id="org_22222222",
                                                  service_account_id="sa_33333333", workspace_id="ws_44444444")
    c = anthropic.AsyncAnthropic(credentials=creds, base_url=vault.ANTHROPIC_BASE_URL, max_retries=0, default_headers={"X-Api-Key": STATIC_KEY},
                                 http_client=wire.async_api())
    assert c.api_key is None and c.credentials is creds
    r = asyncio.run(c.messages.create(**PARAMS))
    assert r.content[0].text == "ok"
    assert wire.requests[0]["x-api-key"] == STATIC_KEY and wire.requests[0]["authorization"] is None
    assert wire.provider_calls == 0 and wire.exchanges == 0
