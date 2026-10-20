"""A COPY of a federated client is federated (Critical: SDK client copies bypassed the WIF destination guard).

The vault built its federated client on the SDK's base class and hung everything that makes it federated ON THE INSTANCE: the vault's
own httpx client over the pin guard was the constructor's ``http_client``, the latched retry layer replaced ``client.messages``, and
the stamp and the credential latch were attributes. The SDK's ``copy`` / ``with_options`` (and ``with_middleware``, which is a copy)
rebuild a NEW client from the constructor's arguments -- inheriting the credentials and the token cache, accepting a replacement
``http_client`` -- and none of the instance's dressing travels. Measured on dbd7473df: ``vault.anthropic_client("lowprio")
.with_options(http_client=httpx.Client(base_url=<the pin>, event_hooks={"request": [hook]}, transport=MockTransport(recorder)))``
built without a word, and its one request reached the recorder at ``other-host.example`` with ``Authorization: Bearer <the exchanged
bearer>``; the copy also carried no latch and no layer (a rejection met on it stopped nothing).

Now every federated SDK client is an instance of the vault's own subclass of the SDK's class (``vault.federated_sdk_class``:
``FederatedAnthropic`` / ``FederatedAsyncAnthropic``; the inspect factories build the same class over the provider's), whose
``copy`` / ``with_options`` are ``vault.federated_copy``: the copy is built DIRECTLY on the parent's credentials, token cache and the
vault's own pinned httpx client (what the SDK's copy would have passed for a base client), with the SDK's retries OFF, and dressed as
its parent was -- the parent's stamp, the SAME latch object, the latched retry layer with the parent's budget (or the ``max_retries``
asked of the copy), and on an inspect client the install's wrapper too. The copy takes ``timeout``, ``max_retries``, ``default_headers``
/ ``set_default_headers`` (a static credential header refused by name), ``default_query`` / ``set_default_query`` and the pin's own
spelling of ``base_url``; an ``http_client`` is ``BaseUrlRefused`` BY NAME (the destination class: a copy is never handed a caller's
httpx client either), and every other argument -- ``api_key``, ``auth_token``, ``credentials``, ``config``, ``profile``, ``webhook_key``,
``middleware``, ``_extra_kwargs``, an unknown name -- and ``with_middleware`` are ``FederatedCopyRefused`` BY NAME. The subclass closes
the bypass too: the installed SDK forwards credentials through ``copy`` only for its exact base classes (``_is_base_client``), so a copy
made behind the override's back carries no credentials and no token cache at all. op / env are untouched: the keyed branches build the
SDK's own class, whose copies are the SDK's own.

The round-10 wire harness and round 12's scripted API are reused: the real SDK over mock transports of the SDK's own httpx flavour, no
network."""
from __future__ import annotations

import asyncio
import os
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
    _ScriptedApi,
    _client_latch_held,
    _sleeper,
    _two_workers,
)
from test_wif_transport_boundary_and_request_headers import (  # noqa: E402
    _Record,
    _httpx_client_of,
    _sols_donor,
)

OTHER = "https://other-host.example"
#: every copy argument a federated client REFUSES by name (the SDK's ``copy`` signature, less the options a federated copy carries)
REFUSED_COPY_ARGS = ({"api_key": SYNTHETIC_KEY}, {"auth_token": "synthetic-oauth-placeholder"}, {"credentials": object()},
                     {"config": {"api_key": SYNTHETIC_KEY}}, {"profile": "synthetic-profile"}, {"webhook_key": "synthetic-webhook"},
                     {"middleware": []}, {"middleware": None}, {"_extra_kwargs": {}}, {"not_an_sdk_option": 1})


# ------------------------------------------------------------------------------------------------------------ the harness


@pytest.fixture
def wire(tmp_path, monkeypatch):
    """wif mode over a synthetic ids directory, the credential doors shut, every inherited ``ANTHROPIC_*`` name gone, the SDK's
    credentials class replaced by its offline twin, and the mock wire (the round-10 harness, re-declared: a fixture does not travel
    by import)."""
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


def _federated_class_of(c) -> None:
    """The client is an instance of the vault's federated subclass of the SDK's class -- never the bare SDK class."""
    base = anthropic.AsyncAnthropic if isinstance(c, anthropic.AsyncAnthropic) else anthropic.Anthropic
    assert isinstance(c, base) and type(c) is not base, type(c)
    assert type(c) is vault.federated_sdk_class(base)
    assert type(c).__name__ == f"Federated{base.__name__}" and type(c).__module__ == vault.__name__
    assert type(c).copy is type(c).with_options and type(c).copy is not base.copy


def _same_federation(cp, c, *, budget: int) -> None:
    """``cp`` is a copy of ``c`` that kept every protection: the class, the parent's credentials and token cache, the vault's own
    pinned httpx client (the same object), the SAME latch object, the stamp, the SDK's retries off and the latched layer with
    ``budget``."""
    assert cp is not c and type(cp) is type(c)
    assert cp.credentials is c.credentials and cp.api_key is None and cp.auth_token is None
    assert getattr(cp, "_token_cache") is getattr(c, "_token_cache")                      # one exchange serves parent and copy
    assert _httpx_client_of(cp) is _httpx_client_of(c)                                   # the vault's own, over the pin guard
    assert isinstance(_httpx_client_of(cp)._transport, vault.PinnedTransport)
    assert str(cp.base_url).rstrip("/") == PIN
    assert cp.max_retries == 0 and vault.federated_retry_budget(cp) == budget, (cp.max_retries, vault.federated_retry_budget(cp))
    latch = vault.client_credential_latch(c)
    assert latch is not None and vault.client_credential_latch(cp) is latch
    assert vault.client_pool_stamp(cp) == vault.client_pool_stamp(c)


# ======================================================================================================== THE CRITICAL


def test_sols_copy_with_a_hooked_http_client_is_refused_by_name_and_nothing_leaves(wire, monkeypatch):
    """A ``with_options`` / ``copy`` of a vault client with an httpx client whose base URL is the pin and whose
    request hook re-points every request: the copy sent the exchanged bearer to the other host. Now it is ``BaseUrlRefused`` BY NAME at
    every door -- the sync and async doors under wif, the direct constructor under op -- naming the argument and the pin; no copy is
    built, the provider is never invoked, the recorder never sees a request. And a copy made behind the override's back (the SDK's own
    ``copy`` called on the class) carries no credentials and no token cache: the installed SDK forwards them only for its exact base
    classes, so the subclass closes that route too."""
    rec = _Record()
    sync_client = vault.anthropic_client("lowprio", max_retries=1)                      # production shape: no donor, no request
    async_client = vault.anthropic_async_client("lowprio", max_retries=1)
    _federated_class_of(sync_client)
    _federated_class_of(async_client)
    for c, donor in ((sync_client, _sols_donor(rec)), (async_client, _sols_donor(rec, async_client=True))):
        for copier in (c.with_options, c.copy):
            with pytest.raises(vault.BaseUrlRefused) as ei:
                copier(http_client=donor)
            msg = str(ei.value)
            assert "http_client" in msg and PIN in msg and "copy" in msg and f"{vault.SECRETS_MODE_ENV}=wif" in msg, msg
            assert isinstance(ei.value, vault.VaultSecretError)
    # the direct constructor IS the wif path whatever the switch says
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "op")
    direct = vault.anthropic_client_wif("lowprio", max_retries=1, token_provider=wire.provider)
    _federated_class_of(direct)
    with pytest.raises(vault.BaseUrlRefused, match="http_client"):
        direct.with_options(http_client=_sols_donor(rec))
    assert rec.requests == [] and wire.requests == [] and wire.provider_calls == 0 and wire.exchanges == 0
    # behind the override's back: the SDK's own copy of a subclass instance carries NO credentials and NO token cache (the installed
    # SDK's `_is_base_client` gate) -- there is no bearer such a copy could carry anywhere
    bypass = anthropic.Anthropic.copy(sync_client, http_client=_sols_donor(rec))
    assert bypass.credentials is None and getattr(bypass, "_token_cache") is None and bypass.api_key is None
    assert rec.requests == [] and wire.provider_calls == 0


def test_a_copy_of_a_federated_client_is_federated_and_shares_its_parents_latch_stamp_cache_and_transport(wire):
    """``with_options(timeout=...)`` / ``copy()`` on a vault client: the copy is the same federated class, built on the parent's
    credentials and token cache and the vault's own pinned httpx client, with the SDK's retries OFF and the latched layer carrying the
    parent's budget (or the count asked of the copy), the parent's stamp and the SAME latch object. Its request leaves for the pin with
    the exchanged bearer, through ONE exchange shared with the parent; a copy of a copy the same. The async door too."""
    c = vault.anthropic_client("lowprio", max_retries=2, http_client=wire.sync_api())
    _clean(c)
    cp = c.with_options(timeout=7.5)
    _same_federation(cp, c, budget=2)
    assert cp.timeout == 7.5 and c.timeout == vault.REQUEST_TIMEOUT_S
    _clean(cp)
    assert isinstance(cp.messages, vault.LatchedRetryMessages) and type(cp.messages.batches).__name__ == "_LatchedRetryBatches"
    assert cp.messages.create(**PARAMS).content[0].text == "ok"
    wire.one_federated_request()
    assert c.messages.create(**PARAMS).content[0].text == "ok"
    assert wire.exchanges == 1 and wire.provider_calls == 1 and len(wire.requests) == 2    # the shared token cache: one exchange
    assert all(r["host"] == "api.anthropic.com" and r["authorization"] == f"Bearer {EXCHANGED}" for r in wire.requests)
    cp2 = cp.copy(max_retries=4)                                                        # the count asked of the copy is its budget
    _same_federation(cp2, c, budget=4)
    _same_federation(c.copy(), c, budget=2)                                              # a bare copy keeps the parent's
    _same_federation(cp2.with_options(timeout=1.0), c, budget=4)                         # a copy of a copy
    # the async door
    a = vault.anthropic_async_client("lowprio", max_retries=3, http_client=wire.async_api())
    ap = a.with_options(timeout=2.0)
    _same_federation(ap, a, budget=3)
    assert isinstance(ap.messages, vault.AsyncLatchedRetryMessages)
    assert asyncio.run(ap.messages.create(**PARAMS)).content[0].text == "ok"
    assert wire.requests[-1]["host"] == "api.anthropic.com" and wire.requests[-1]["authorization"] == f"Bearer {EXCHANGED}"


def test_a_rejection_met_on_a_copy_stops_the_parent_and_a_worker_asleep_on_the_copy_wakes_into_the_stop(wire, monkeypatch):
    """One latch for parent and copy: a 401 met on the copy is held on the latch both carry, so the parent's next call is the typed
    stop without a request (and the other way round). Two workers -- B on the copy asleep in the layer's 429 backoff, A on the parent
    meeting the 401 -- issue exactly TWO requests; B wakes into the typed stop."""
    api = _ScriptedApi([401, 200])
    c = vault.anthropic_client("lowprio", max_retries=3, http_client=api.sync())
    cp = c.copy()
    with pytest.raises(anthropic.AuthenticationError):
        cp.messages.create(**PARAMS)
    held = vault.client_credential_latch(c).held
    assert isinstance(held, vault.JudgeCredentialRejected) and held.status == 401
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        c.messages.create(**PARAMS)
    assert ei.value is held and len(api.requests) == 1
    with pytest.raises(vault.JudgeCredentialRejected):
        cp.with_options(timeout=1.0).messages.create(**PARAMS)                             # a copy made AFTER the hold is held too
    assert len(api.requests) == 1
    # the other way round: the parent meets it, the copy stops
    api2 = _ScriptedApi([401, 200])
    c2 = vault.anthropic_client("lowprio", max_retries=3, http_client=api2.sync())
    cp2 = c2.with_options(timeout=5.0)
    with pytest.raises(anthropic.AuthenticationError):
        c2.messages.create(**PARAMS)
    with pytest.raises(vault.JudgeCredentialRejected):
        cp2.messages.create(**PARAMS)
    assert len(api2.requests) == 1
    # two workers across parent and copy
    api3 = _ScriptedApi([(429, {"retry-after": "5"}), 401])
    c3 = vault.anthropic_client("lowprio", max_retries=1, http_client=api3.sync())
    cp3 = c3.copy()
    slept: list = []
    monkeypatch.setattr(time, "sleep", _sleeper(api3, _client_latch_held([c3]), slept))
    out = _two_workers(lambda: c3.messages.create(**PARAMS), lambda: cp3.messages.create(**PARAMS))
    assert out["A"][0] == out["B"][0] == "raised"
    assert {type(out["A"][1]), type(out["B"][1])} == {anthropic.AuthenticationError, vault.JudgeCredentialRejected}, out
    assert slept == [5.0]
    api3.two_federated_requests()


def test_every_other_copy_argument_is_refused_by_name_and_a_clean_option_rides(wire):
    """A copy may change its timeout, its retry budget, its default headers (none a static credential) and its default query, and may
    spell the pin as its base URL. Anything else is refused BY NAME before anything is built: a base URL naming another host
    (``BaseUrlRefused``), a static ``X-Api-Key`` / ``Authorization`` in ``default_headers`` / ``set_default_headers``
    (``StaticAuthRefused``), and a credential, pipeline or unknown argument -- ``api_key``, ``auth_token``, ``credentials``, ``config``,
    ``profile``, ``webhook_key``, ``middleware``, ``_extra_kwargs``, a name the SDK does not know -- and ``with_middleware``
    (``FederatedCopyRefused``, naming the argument and the vault's door). Nothing on the wire, the provider never invoked."""
    c = vault.anthropic_client("lowprio", max_retries=1, http_client=wire.sync_api())
    for other in OTHER_HOSTS:
        with pytest.raises(vault.BaseUrlRefused) as ei:
            c.with_options(base_url=other)
        assert f"base_url={other!r}" in str(ei.value) and PIN in str(ei.value)
    for headers in ({"X-Api-Key": STATIC_KEY}, {"Authorization": "Bearer " + STATIC_BEARER}, {"x-api-key": STATIC_KEY}):
        with pytest.raises(vault.StaticAuthRefused, match="(?i)default_headers"):
            c.with_options(default_headers=headers)
        with pytest.raises(vault.StaticAuthRefused, match="(?i)default_headers"):
            c.copy(set_default_headers=headers)
    for kw in REFUSED_COPY_ARGS:
        (name,) = kw
        with pytest.raises(vault.FederatedCopyRefused) as ei:
            c.with_options(**kw)
        msg = str(ei.value)
        assert name in msg and "anthropic_client" in msg and "copy" in msg and f"{vault.SECRETS_MODE_ENV}=wif" in msg, (name, msg)
        assert isinstance(ei.value, vault.VaultSecretError)
        assert STATIC_KEY not in msg and SYNTHETIC_KEY not in msg                       # never a value
    with pytest.raises(vault.FederatedCopyRefused, match="middleware"):
        c.with_middleware(object())
    with pytest.raises(ValueError, match="mutually exclusive"):                         # the SDK's own rule, kept
        c.with_options(default_headers={"x-synthetic": "1"}, set_default_headers={"x-synthetic": "1"})
    assert wire.requests == [] and wire.provider_calls == 0 and wire.exchanges == 0
    # the clean options ride: a beta header on the copy reaches the wire beside the exchanged bearer; the pin's own spelling builds;
    # None for http_client / base_url is a plain copy (the SDK's own reading of None: keep the parent's)
    cp = c.with_options(default_headers={"anthropic-beta": "synthetic-beta-r14"}, default_query={"synthetic": "1"},
                        base_url=PIN + "/", http_client=None)
    _same_federation(cp, c, budget=1)
    assert cp.messages.create(**PARAMS).content[0].text == "ok"
    assert len(wire.requests) == 1 and wire.requests[0]["authorization"] == f"Bearer {EXCHANGED}"
    assert "synthetic-beta-r14" in (wire.requests[0]["beta"] or "")
    assert "x-api-key" not in {k.lower() for k in cp._custom_headers} and cp._custom_headers.get("anthropic-beta") == "synthetic-beta-r14"
    plain = c.copy(http_client=None, base_url=None)
    _same_federation(plain, c, budget=1)


def test_the_inspect_factories_build_the_federated_class_and_a_copy_keeps_the_installs_latch_and_wrapper(installed, wire):
    """Both factories of ``install_inspect_wif`` build the federated subclass of the provider's class. A copy of such a client keeps the
    install's latch (the one object), the latched retry layer with its budget AND the install's own wrapper (the abort hook), and refuses
    a caller's ``http_client`` by name; its request leaves for the pin with the exchanged bearer."""
    from inspect_ai.model import get_model
    w, prov, real_cls, creds = installed
    rec = _Record()
    c = prov.AsyncAnthropic(api_key=w.WIF_SENTINEL_KEY, http_client=wire.async_api(), max_retries=1)
    assert isinstance(c, real_cls) and type(c) is vault.federated_sdk_class(real_cls) and c.credentials is creds
    cp = c.with_options(timeout=3.0)
    assert type(cp) is type(c) and cp.credentials is creds and getattr(cp, "_token_cache") is getattr(c, "_token_cache")
    assert _httpx_client_of(cp) is _httpx_client_of(c) and isinstance(_httpx_client_of(cp)._transport, vault.PinnedTransport)
    assert cp.max_retries == 0 and vault.federated_retry_budget(cp) == 1
    assert isinstance(cp.messages, w._LatchedMessages) and isinstance(cp.messages._inner, vault.AsyncLatchedRetryMessages)
    assert vault.client_credential_latch(cp) is w._LATCH and vault.client_credential_latch(c) is w._LATCH
    with pytest.raises(vault.BaseUrlRefused, match="http_client"):
        cp.with_options(http_client=_sols_donor(rec, async_client=True))
    with pytest.raises(vault.FederatedCopyRefused, match="api_key"):
        c.copy(api_key=SYNTHETIC_KEY)
    assert rec.requests == [] and wire.requests == []
    assert asyncio.run(cp.messages.create(**PARAMS)).content[0].text == "ok"
    wire.one_federated_request()
    # the model-args factory (the provider's own client path) builds the same class
    m = get_model("anthropic/claude-sonnet-5-wif-synthetic-r14-copy", http_client=wire.async_api())
    mc = m.api.client
    assert type(mc) is vault.federated_sdk_class(real_cls) and mc.credentials is creds
    mcp = mc.copy()
    assert type(mcp) is type(mc) and isinstance(mcp.messages, w._LatchedMessages) and vault.client_credential_latch(mcp) is w._LATCH


def test_under_op_and_env_a_vault_client_is_the_sdks_own_class_and_its_copies_are_the_sdks_own(monkeypatch):
    """The keyed modes are untouched: the client is exactly ``anthropic.Anthropic`` (``AsyncAnthropic``), and ``with_options`` /
    ``copy`` / ``with_middleware`` are the SDK's own -- an ``http_client`` on a copy is taken as handed, as before (the SDK's copy with
    the caller's transport under it sends the org's key where the SDK's own copy would)."""
    for name in list(os.environ):
        if name.startswith(vault.WIF_SCRUB_PREFIX):
            monkeypatch.delenv(name, raising=False)
    for mode, var in (("op", "ANTHROPIC_API_KEY_LOWPRIO"), ("env", "ANTHROPIC_API_KEY")):
        monkeypatch.setenv(vault.SECRETS_MODE_ENV, mode)
        monkeypatch.setenv(var, SYNTHETIC_KEY)
        c = vault.anthropic_client("lowprio", max_retries=2)
        a = vault.anthropic_async_client("lowprio", max_retries=2)
        assert type(c) is anthropic.Anthropic and type(a) is anthropic.AsyncAnthropic
        assert c.max_retries == 2 and vault.federated_retry_budget(c) is None
        rec = _Record()
        cp = c.with_options(http_client=httpx.Client(transport=httpx.MockTransport(rec)))
        assert type(cp) is anthropic.Anthropic and type(c.copy(timeout=1.0)) is anthropic.Anthropic
        assert type(c.with_middleware()) is anthropic.Anthropic
        assert cp.messages.create(**PARAMS).content[0].text == "ok"
        assert len(rec.requests) == 1 and rec.requests[0]["x-api-key"] == SYNTHETIC_KEY and rec.requests[0]["host"] == "api.anthropic.com"
        monkeypatch.delenv(var)


def test_the_constructors_build_the_federated_class_and_no_tree_code_copies_a_vault_client():
    """The pins: the vault's constructor and both inspect factories build through ``vault.federated_sdk_class``; the no-copy sweep of
    the tree (``tests/test_wif_federated_destination_and_sdk_retries.py``) still holds -- a copy is guarded now, and still nothing in
    the tree needs one."""
    vsrc = (REPO / "src/sisteer/vault.py").read_text()
    assert "cls = federated_sdk_class(anthropic.AsyncAnthropic if async_client else anthropic.Anthropic)" in vsrc
    assert "client = cls(credentials=creds, base_url=pinned, max_retries=0, timeout=timeout, **extra)" in vsrc
    assert "def federated_sdk_class(" in vsrc and "def federated_copy(" in vsrc and "class FederatedCopyRefused(" in vsrc
    wsrc = (REPO / "src/sisteer/eval/anthropic_wif.py").read_text()
    assert "fed_cls = vault.federated_sdk_class(real_cls)" in wsrc
    assert "client = fed_cls(*args, credentials=creds, max_retries=0, **kwargs)" in wsrc
    assert "client = fed_cls(credentials=creds, base_url=pinned_url, max_retries=0, **extra)" in wsrc
    assert "vault.FEDERATED_DRESS_ATTR" in wsrc
    assert vault.federated_sdk_class(anthropic.Anthropic) is vault.federated_sdk_class(anthropic.Anthropic)       # built once per class
    assert vault.federated_sdk_class(vault.federated_sdk_class(anthropic.Anthropic)) is vault.federated_sdk_class(anthropic.Anthropic)
    with pytest.raises(TypeError, match="not a class"):
        vault.federated_sdk_class(lambda **kw: None)                                   # a stand-in that is not a class: loud, never wrapped
