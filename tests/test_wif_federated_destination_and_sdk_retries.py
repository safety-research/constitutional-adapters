"""A federated client's destination is the pin BY CONSTRUCTION, its SDK retries run ABOVE the latch, and the keyed modes are
byte-identical to the base.

CRITICAL -- the constructors accepted a non-Anthropic destination. ``vault.anthropic_client(..., base_url="https://other")`` under
wif forwarded the URL to ``anthropic_client_wif``, which built a credentialed SDK client against it: the exchanged bearer and every
request's contents could reach that host. ``install_inspect_wif(base_url=...)`` had the same seam. ``pinned_base_url`` guarded the
readers of an OUTSIDE value (an environment variable, a config key), never the constructor itself. Now
``vault.pin_federated_destination`` is consulted FIRST at both federated construction boundaries -- the vault's constructor (so both
public doors under wif) and the inspect install plus both of its factories -- and refuses BY NAME an in-process ``base_url`` or an
``http_client`` whose own base URL names another host, before any id is read or any client built. The test seam is a TRANSPORT:
an ``http_client`` over a mock transport, the request leaving the SDK addressed to the pinned host through it.

MAJOR -- the SDK's own retries re-sent below every latch. ``judge_sonnet`` (SDK ``max_retries=1``), ``judge_overrefusal`` (1), the
misalign ``PoolClient`` (1), ``build_probe_set`` (2), ``ClaudeJudge`` (2), ``grokmt_analyze`` (3), ``AnthropicRewriter`` (8) and the
inspect provider's clients each checked their latch in their OWN loop, while the SDK slept and re-sent underneath: a worker B inside
an SDK backoff when worker A's request met the run's first 401 / 403 woke and issued another request on the rejected credential.
Round 11 closed that on the live grader's path alone. Now the rule is the ONE constructor's: under federation the SDK client is built
with ``max_retries=0`` and the count asked for is the budget of ``vault.LatchedRetryMessages`` (``federate_sdk_retries``), which
replays the SDK's own policy -- the same statuses, counts and delays -- but checks the client's latch before EVERY attempt. Every
driver's robustness contract is unchanged; the sleeper wakes into the typed stop. Each two-worker test below holds B in that
backoff (``time.sleep`` / ``asyncio.sleep`` patched to a wait on A's rejection being issued and held) and asserts exactly TWO
requests on the wire -- B's 429 and A's 401 -- never a third. On fd06740ae every one of them saw the third request.

MAJOR -- the keyed modes had lost their retry contract. Round 11's ``LiveStrongREJECTGrader`` refused a non-zero ``max_retries`` and
built its client with 0 in op / env too, against the stated byte-identity of the keyed modes. The base's default (8), clamp (2) and
client build (``make_client(org, max_retries=min(n, 2))``) are back in every mode; where the count LIVES is the vault's business by
mode. The byte-identity is proven here: the vault's keyed branches hand the SDK exactly the base's arguments, the drivers' SDK retry
literals are the base's (``e30939041``), and a prebuilt client with retries is taken as handed under op / env (and refused by name
under wif, where it cannot have come from the vault). The round-10 wire harness (``tests/test_wif_inherited_static_auth.py``) is
reused: the real SDK over mock transports, no network."""
from __future__ import annotations

import asyncio
import concurrent.futures
import importlib.util
import inspect as _inspect
import json
import os
import re
import subprocess
import sys
import threading
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
    STATIC_KEY,
    _Wire,
    _clean,
    httpx,
)

PIN = vault.ANTHROPIC_BASE_URL
OTHER_HOSTS = ("http://127.0.0.1:1/double", "https://another-host.example", "http://localhost:9/judge-mock")
WAKE_BOUND_S = 3.0
_REAL_SLEEP = time.sleep                     # the harness's own stagger never goes through a patched time.sleep
SYNTHETIC_KEY = "synthetic-placeholder-not-a-key"
#: the base e30939041's SDK retry literals, per driver: the keyed modes' contract (what each driver asks the vault for; under op /
#: env the SDK's own count, under wif the latched layer's budget). A changed literal fails here until the table says why.
KEYED_SDK_RETRY_LITERALS = {
    "scripts/judge_sonnet.py": ("make_client(org, max_retries=1)",),
    "scripts/judge_overrefusal.py": ('make_client("lowprio", max_retries=1)',),
    "scripts/sdfbase/full22/misalign/misalign_judge.py": ("max_retries=1, timeout=JUDGE_TIMEOUT_S",),
    "scripts/sdfbase/build_probe_set.py": ("MAX_RETRIES = 2", "max_retries=MAX_RETRIES"),
    "src/sisteer/capeval/judging.py": ("MAX_RETRIES = 2", "max_retries=self.max_retries"),
    "scripts/grokmt_analyze.py": ("vault.anthropic_client(JUDGE_ORG, max_retries=3)",),
    "src/sisteer/eval/attacks/llm_rewriter.py": ("max_retries: int = 8", "max_retries=max_retries"),
    "scripts/judge_batch.py": ("vault.anthropic_client(self.org, max_retries=5)",),
    "scripts/sdfbase/full22/judge/batch_judge_fill.py": ("max_retries=2, timeout=judging.REQUEST_TIMEOUT_S",),
    "scripts/capeval_judged_run.py": ("max_retries: int = 1",),
    "scripts/alphasteer_f22/grade_alphasteer_f22.py": ("make_client(org, max_retries=0)",),
    "scripts/grade_bon_live.py": ("max_retries: int = 0",),
    "src/sisteer/eval/api_keys.py": ("max_retries: int = 5",),
    "src/sisteer/eval/attacks/grading_live.py": ("max_retries: int = 8", "self._sdk_max_retries = min(int(max_retries), 2)",
                                                  "make_client(self.org, max_retries=self._sdk_max_retries)"),
    "src/sisteer/vault.py": ("def anthropic_client(org: str = \"lowprio\", *, max_retries: int = 5",
                             "def anthropic_async_client(org: str = \"lowprio\", *, max_retries: int = 5",
                             "def anthropic_client_wif(org: str = \"lowprio\", *, max_retries: int = 5"),
}


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


def _error_body(status: int) -> dict:
    kind = {401: "authentication_error", 403: "permission_error", 429: "rate_limit_error", 400: "invalid_request_error",
            404: "not_found_error", 529: "overloaded_error"}.get(status, "api_error")
    return {"type": "error", "error": {"type": kind, "message": "synthetic"}}


class _ScriptedApi:
    """The API end of the wire with the answers SCRIPTED BY CALL ORDER under a lock: call k answers ``script[k-1]`` (a status, or
    ``(status, headers)``; 200 is a minimal message), a call past the script answers the LAST entry again (an extra request a
    retrying layer makes lands here and is counted). ``issued`` is set when a 401 / 403 has been answered, for the sleeper;
    ``requests`` records host, path and the auth header of every request (never a body)."""

    def __init__(self, script):
        self.script, self.requests, self._lock = list(script), [], threading.Lock()
        self.issued = threading.Event()

    def __call__(self, request):
        with self._lock:
            n = len(self.requests) + 1
            self.requests.append({"host": request.url.host, "path": request.url.path,
                                  "authorization": request.headers.get("Authorization"), "x-api-key": request.headers.get("X-Api-Key")})
        entry = self.script[min(n, len(self.script)) - 1]
        status, headers = (entry if isinstance(entry, tuple) else (entry, {}))
        if status == 200:
            body = MESSAGE if not request.url.path.endswith("/count_tokens") else {"input_tokens": 7}
            return httpx.Response(200, json=body, headers=headers, request=request)
        if status in (401, 403):
            self.issued.set()
        return httpx.Response(status, json=_error_body(status), headers=headers, request=request)

    def sync(self):
        return httpx.Client(transport=httpx.MockTransport(self))

    def async_(self):
        return httpx.AsyncClient(transport=httpx.MockTransport(self))

    def two_federated_requests(self):
        """Exactly TWO requests left the process, both to the pinned host with the exchanged bearer and no static header: B's (the
        429) and A's (the 401); the third -- the retry below the latch -- never happened."""
        assert len(self.requests) == 2, f"{len(self.requests)} request(s) on the wire (a retry below the latch): {self.requests}"
        for r in self.requests:
            assert r["host"] == "api.anthropic.com" and r["authorization"] == f"Bearer {EXCHANGED}" and r["x-api-key"] is None, r


def _route_through(monkeypatch, api: _ScriptedApi) -> list:
    """Every client a DRIVER builds through the vault's two doors leaves through ``api``: the doors are wrapped to pass the test
    seam (an ``http_client`` over the mock transport against the pinned host) and nothing else about the build changes. Returns the
    clients built. (On a tree whose doors have no ``http_client`` seam the transport is put on a ``with_options`` copy, the stamp
    restored, so the test measures the SDK's retries there rather than failing at the seam.)"""
    built: list = []
    real_sync, real_async = vault.anthropic_client, vault.anthropic_async_client

    def _build(real, org, kw, http_client):
        try:
            return real(org, http_client=http_client, **kw)
        except TypeError as e:
            if "http_client" not in str(e):
                raise
            c = real(org, **kw)
            copy = c.with_options(http_client=http_client)
            setattr(copy, vault.POOL_STAMP_ATTR, vault.client_pool_stamp(c))
            return copy

    def sync_door(org="lowprio", **kw):
        c = _build(real_sync, org, kw, api.sync())
        built.append(c)
        return c

    def async_door(org="lowprio", **kw):
        c = _build(real_async, org, kw, api.async_())
        built.append(c)
        return c

    monkeypatch.setattr(vault, "anthropic_client", sync_door)
    monkeypatch.setattr(vault, "anthropic_async_client", async_door)
    return built


def _client_latch_held(built: list):
    def held() -> bool:
        for c in built:
            latch = vault.client_credential_latch(c)
            if latch is not None and latch.held is not None:
                return True
        return False
    return held


def _sleeper(api: _ScriptedApi, held, slept: list):
    """The backoff worker B sleeps in, made deterministic: wait until the rejection has been ISSUED to a sibling and then until it
    is HELD (``held()``), bounded, then return -- B wakes exactly when a retry layer would re-send. Built BEFORE ``time.sleep`` is
    patched: it keeps the real one. Patching ``time.sleep`` reaches the SDK's own retry sleep and the vault layer's alike."""
    real_sleep = time.sleep

    def sleep(s):
        slept.append(s)
        api.issued.wait(WAKE_BOUND_S)
        t0 = time.monotonic()
        while not held() and time.monotonic() - t0 < WAKE_BOUND_S:
            real_sleep(0.005)
    return sleep


def _async_sleeper(api: _ScriptedApi, held, slept: list):
    real_sleep = asyncio.sleep

    async def sleep(s):
        slept.append(s)
        t0 = time.monotonic()
        while not (api.issued.is_set() and held()) and time.monotonic() - t0 < WAKE_BOUND_S:
            await real_sleep(0.005)
    return sleep


def _two_workers(fn_a, fn_b):
    """Run B first (its request is the script's first: the 429), A a moment later (the 401); return both outcomes (a result or the
    exception). The scripted API orders the answers by ARRIVAL, so B's head start makes the roles deterministic."""
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


class _Recorder:
    """A stand-in for the SDK's client class: records the constructor's kwargs; carries what a keyed client carries."""
    instances: list = []

    def __init__(self, **kw):
        self.kw = dict(kw)
        self.max_retries = kw.get("max_retries")
        self.base_url = kw.get("base_url")
        self.messages = object()                 # the SDK's own resource stands in; a wrapper would replace it
        type(self).instances.append(self)


# ======================================================================================================== THE CRITICAL


def test_under_wif_a_base_url_naming_another_host_is_refused_by_name_before_anything_is_built(wire, monkeypatch):
    """The in-process ``base_url=`` used to be 'each constructor's test seam': under wif it moved the federated bearer to the host it
    named. Now both public doors and the direct constructor refuse it BY NAME -- naming the argument, the override, the mode and
    the pin -- before the ids are read, before the SDK's constructor runs, before any token is minted."""
    def boom(**kw):
        raise AssertionError(f"the SDK's constructor ran for a refused destination: {sorted(kw)}")

    def no_ids(*a, **k):
        raise AssertionError("the ids file was read for a refused destination")

    with pytest.MonkeyPatch.context() as mp:            # the booms live only here; the wire fixture's patches stay
        mp.setattr(anthropic, "Anthropic", boom)
        mp.setattr(anthropic, "AsyncAnthropic", boom)
        mp.setattr(vault, "wif_credentials", no_ids)
        for door in (vault.anthropic_client, vault.anthropic_async_client, vault.anthropic_client_wif):
            for other in OTHER_HOSTS:
                with pytest.raises(vault.BaseUrlRefused) as ei:
                    door("lowprio", base_url=other)
                msg = str(ei.value)
                assert f"base_url={other!r}" in msg and PIN in msg and f"{vault.SECRETS_MODE_ENV}=wif" in msg and "mock transport" in msg, msg
                assert isinstance(ei.value, vault.VaultSecretError)
        # the direct constructor IS the wif path whatever the switch says: under op it refuses the same
        mp.setenv(vault.SECRETS_MODE_ENV, "op")
        with pytest.raises(vault.BaseUrlRefused, match="base_url="):
            vault.anthropic_client_wif("lowprio", base_url=OTHER_HOSTS[0], token_provider=wire.provider)
    assert wire.provider_calls == 0
    # the pin's own spelling, a trailing slash aside, builds (the real constructor again)
    for spelling in (PIN, PIN + "/"):
        c = vault.anthropic_client("lowprio", base_url=spelling)
        assert str(c.base_url).rstrip("/") == PIN and c.api_key is None


def test_the_test_seam_is_a_transport_and_the_request_leaves_for_the_pinned_host_through_it(wire):
    """What replaces the URL seam: an ``http_client`` over a mock transport. The client is pinned, keyless, federated; its one request
    goes out addressed to api.anthropic.com with the EXCHANGED bearer and arrives at the transport -- sync and async doors alike."""
    c = vault.anthropic_client("lowprio", max_retries=0, http_client=wire.sync_api())
    _clean(c)
    assert vault.client_pool_stamp(c)["auth"] == "wif"
    r = c.messages.create(**PARAMS)
    assert r.content[0].text == "ok"
    wire.one_federated_request()
    a = vault.anthropic_async_client("lowprio", max_retries=0, http_client=wire.async_api())
    _clean(a)
    assert asyncio.run(a.messages.create(**PARAMS)).content[0].text == "ok"
    assert wire.requests[1]["host"] == "api.anthropic.com" and wire.requests[1]["authorization"] == f"Bearer {EXCHANGED}"


def test_an_http_client_whose_own_base_url_names_another_host_is_refused_and_a_static_header_on_it_too(wire):
    """The transport seam is not a second URL seam: an httpx client whose base URL names another host is refused by name (the pin's
    own spelling on it passes); one carrying a static X-Api-Key / Authorization header is refused as round 11's rule says."""
    for other in OTHER_HOSTS:
        with pytest.raises(vault.BaseUrlRefused) as ei:
            vault.anthropic_client("lowprio", http_client=httpx.Client(base_url=other, transport=httpx.MockTransport(wire.api)))
        assert "http_client.base_url=" in str(ei.value) and other in str(ei.value) and PIN in str(ei.value)   # httpx spells it with a trailing slash
        with pytest.raises(vault.BaseUrlRefused, match="http_client.base_url="):
            vault.anthropic_async_client("lowprio", http_client=httpx.AsyncClient(base_url=other, transport=httpx.MockTransport(wire.api)))
    assert wire.provider_calls == 0 and wire.requests == []
    ok = vault.anthropic_client("lowprio", max_retries=0, http_client=httpx.Client(base_url=PIN + "/", transport=httpx.MockTransport(wire.api)))
    ok.messages.create(**PARAMS)
    wire.one_federated_request()
    with pytest.raises(vault.StaticAuthRefused, match="(?i)http_client.headers: x-api-key"):      # httpx lower-cases header names
        vault.anthropic_client("lowprio", http_client=httpx.Client(headers={"X-Api-Key": STATIC_KEY}, transport=httpx.MockTransport(wire.api)))
    assert len(wire.requests) == 1


def test_the_inspect_install_refuses_a_non_pinned_destination_before_installing_and_in_both_factories(wire, monkeypatch):
    """``install_inspect_wif(base_url=<other host>)`` installed a federated provider whose every client carried the bearer to that
    host. Now it refuses by name before touching the provider (nothing installed, no latch); the pin's own spelling installs; an
    ``http_client`` model argument whose own base URL names another host is refused in the bypassed provider factory (``get_model``)
    and in the rebound backstop alike; and the clean seam -- an ``http_client`` over the mock transport -- federates to the pin."""
    pytest.importorskip("inspect_ai")
    from inspect_ai.model import get_model
    from inspect_ai.model._providers import anthropic as prov
    from sisteer.eval import anthropic_wif as w
    real_cls = prov.AsyncAnthropic
    assert getattr(real_cls, "__sisteer_wif_real__", None) is None, "a stale install leaked from another test"
    for other in OTHER_HOSTS:
        with pytest.raises(vault.BaseUrlRefused) as ei:
            w.install_inspect_wif("lowprio", base_url=other)
        assert f"base_url={other!r}" in str(ei.value) and "install_inspect_wif" in str(ei.value)
        assert prov.AsyncAnthropic is real_cls and w.credential_rejected() is None and w.inspect_wif_base_url() is None
    try:
        creds = w.install_inspect_wif("lowprio", base_url=PIN + "/")                     # the pin's own spelling
        assert w.inspect_wif_base_url() == PIN and prov.AsyncAnthropic is not real_cls
        for other in OTHER_HOSTS:
            with pytest.raises(vault.BaseUrlRefused, match="http_client.base_url="):
                get_model("anthropic/claude-sonnet-5-wif-synthetic-r12-dest",
                          http_client=httpx.AsyncClient(base_url=other, transport=httpx.MockTransport(wire.api)))
            with pytest.raises(vault.BaseUrlRefused, match="http_client.base_url="):
                prov.AsyncAnthropic(api_key=w.WIF_SENTINEL_KEY, http_client=httpx.AsyncClient(base_url=other, transport=httpx.MockTransport(wire.api)))
        assert wire.requests == [] and wire.provider_calls == 0
        m = get_model("anthropic/claude-sonnet-5-wif-synthetic-r12-seam", http_client=wire.async_api())
        c = m.api.client
        assert isinstance(c, real_cls) and c.credentials is creds and str(c.base_url).rstrip("/") == PIN and c.api_key is None
        assert asyncio.run(c.messages.create(**PARAMS)).content[0].text == "ok"
        wire.one_federated_request()
    finally:
        assert w.uninstall_inspect_wif() is True
    assert prov.AsyncAnthropic is real_cls


def test_under_op_and_env_an_explicit_base_url_still_wins_and_nothing_is_wrapped(monkeypatch):
    """The keyed modes keep the override exactly as before: the SDK is handed the caller's URL, its own retry count, and no layer."""
    monkeypatch.setattr(anthropic, "Anthropic", _Recorder)
    _Recorder.instances.clear()
    for mode, var in (("op", "ANTHROPIC_API_KEY_LOWPRIO"), ("env", "ANTHROPIC_API_KEY")):
        monkeypatch.setenv(vault.SECRETS_MODE_ENV, mode)
        monkeypatch.setenv(var, SYNTHETIC_KEY)
        c = vault.anthropic_client("lowprio", base_url=OTHER_HOSTS[0])
        assert c.kw == {"api_key": SYNTHETIC_KEY, "base_url": OTHER_HOSTS[0], "max_retries": 5, "timeout": vault.REQUEST_TIMEOUT_S}
        assert vault.federated_retry_budget(c) is None and type(c.messages) is object
        monkeypatch.delenv(var)


def test_no_tree_code_copies_a_vault_client_and_the_constructor_hands_the_sdk_the_pin_with_retries_off():
    """Nothing in the tree copies a vault client (the corpus generator's batch creator copies ITS OWN client, whose copies are its own
    member's federated ones since round 14). Since round 14 a copy of a federated client IS federated (``vault.federated_copy``,
    ``tests/test_wif_federated_client_copies.py``), so the sweep is no longer what keeps a copy safe -- it stays because a copy that
    nobody needs is a seam nobody watches. And the federated constructors' SDK calls are pinned textually: the federated class, the pin,
    the SDK's retries off."""
    out = subprocess.run(["git", "ls-files", "-z", "--", "src/", "scripts/", "reproduce/"], cwd=REPO, capture_output=True, text=True, check=True).stdout
    copiers = []
    for rel in sorted(p for p in out.split("\0") if p.endswith(".py") and "/tests/" not in p and not p.startswith("legacy/")):
        src = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"\.with_options\(|\.copy\(\s*\*\*|client\.copy\(", src):
            copiers.append(f"{rel}:{src.count(chr(10), 0, m.start()) + 1}")
    assert copiers == [], f"a copy of a vault client drops its stamp, latch and latched retries: {copiers}"
    vsrc = (REPO / "src/sisteer/vault.py").read_text()
    assert "client = cls(credentials=creds, base_url=pinned, max_retries=0, timeout=timeout, **extra)" in vsrc
    assert "pinned = pin_federated_destination(base_url, http_client=http_client, subject=subject)" in vsrc
    assert "federate_sdk_retries(client, max_retries=max_retries)" in vsrc
    wsrc = (REPO / "src/sisteer/eval/anthropic_wif.py").read_text()
    assert wsrc.count("vault.pin_federated_destination(") == 3                     # the install, the backstop factory, the model-args factory
    assert "fed_cls = vault.federated_sdk_class(real_cls)" in wsrc                     # round 14: the vault's federated subclass of the provider's
    assert "client = fed_cls(*args, credentials=creds, max_retries=0, **kwargs)" in wsrc            # the backstop factory
    assert "client = fed_cls(credentials=creds, base_url=pinned_url, max_retries=0, **extra)" in wsrc   # the model-args factory
    assert wsrc.count("vault.federate_sdk_retries(") == 1


# ========================================================================================================= THE MAJORS: the layer


def test_under_wif_every_vault_door_builds_the_sdk_with_retries_off_and_the_count_asked_for_is_the_layers_budget(wire, monkeypatch):
    monkeypatch.setattr(vault, "google_identity_token", wire.provider)
    from sisteer.eval import api_keys
    import grade_bon_live
    cases = [(vault.anthropic_client("lowprio", max_retries=1), 1, vault.LatchedRetryMessages),
             (vault.anthropic_async_client("lowprio", max_retries=2), 2, vault.AsyncLatchedRetryMessages),
             (vault.anthropic_client_wif("highprio", max_retries=8), 8, vault.LatchedRetryMessages),
             (vault.anthropic_client("batch"), 5, vault.LatchedRetryMessages),
             (api_keys.make_client("lowprio", max_retries=1), 1, vault.LatchedRetryMessages),
             (grade_bon_live.make_client("lowprio"), 0, vault.LatchedRetryMessages)]
    for c, budget, cls in cases:
        assert c.max_retries == 0, (type(c).__name__, c.max_retries)              # the SDK's own retries: OFF
        assert isinstance(c.messages, cls) and c.messages.max_retries == budget      # the count asked for: the layer's
        assert vault.federated_retry_budget(c) == budget
        assert type(c.messages.batches).__name__.endswith("LatchedRetryBatches") and c.messages.batches.max_retries == budget
        assert c.messages.with_raw_response is not None                              # other attributes forward to the SDK's resource
    # the layer refuses to sit over a retrying SDK client (two layers, the lower below the latch)
    retrying = anthropic.Anthropic(api_key=SYNTHETIC_KEY, base_url=PIN, max_retries=2)
    with pytest.raises(ValueError, match="carries max_retries=2"):
        vault.federate_sdk_retries(retrying, max_retries=2)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 408, 409, 413, 429, 500, 502, 503, 529])
def test_the_layers_retry_decision_is_the_installed_sdks_own(status):
    """``vault.sdk_would_retry`` says exactly what the installed SDK's ``_should_retry`` says for the same response -- for every
    status the drivers meet, with and without the ``x-should-retry`` header -- and a connection error is retried as the SDK does.
    A future SDK that changes its policy fails here, which is the signal to re-read the replay."""
    sdk = anthropic.Anthropic(api_key=SYNTHETIC_KEY, base_url=PIN, max_retries=0)
    req = httpx.Request("POST", PIN + "/v1/messages")
    for headers in ({}, {"x-should-retry": "true"}, {"x-should-retry": "false"}):
        resp = httpx.Response(status, request=req, headers=headers, json=_error_body(status))
        exc = anthropic.APIStatusError(f"Error code: {status}", response=resp, body=_error_body(status))
        exc.status_code = status
        assert vault.sdk_would_retry(exc)[0] is sdk._should_retry(resp), (status, headers)
    conn = anthropic.APIConnectionError(request=req)
    assert vault.sdk_would_retry(conn) == (True, None)
    wrapped = RuntimeError("wrapped")
    wrapped.__cause__ = conn
    assert vault.sdk_would_retry(wrapped)[0] is True                               # the __cause__ chain, as the SDK walks it
    assert vault.sdk_would_retry(ValueError("not a wire error")) == (False, None)


def test_the_layer_replays_the_sdks_delays_counts_and_statuses_above_the_latch(wire, monkeypatch):
    """Against the real SDK over a scripted wire, ``time.sleep`` recorded: a 429 with ``retry-after: 3`` sleeps 3 s then succeeds; a
    500 sleeps one SDK backoff (0.375 .. 0.5 s) then succeeds; ``retry-after-ms: 250`` is 0.25 s; a retry-after over 60 s is the
    backoff instead; a 408 and a 409 are retried; ``x-should-retry: false`` on a 500 is not; a 400 is not; a connection error is;
    the budget holds (two retries, three requests, then the SDK's last error); a 401 is never retried and is raised as the SDK
    raised it, held on the client's latch, and the next call is the typed stop without a request."""
    slept: list = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))

    def client(script, budget=2):
        api = _ScriptedApi(script)
        return api, vault.anthropic_client("lowprio", max_retries=budget, http_client=api.sync())

    api, c = client([(429, {"retry-after": "3"}), 200])
    assert c.messages.create(**PARAMS).content[0].text == "ok" and len(api.requests) == 2 and slept == [3.0]
    slept.clear()
    api, c = client([500, 200])
    c.messages.create(**PARAMS)
    assert len(api.requests) == 2 and len(slept) == 1 and 0.375 <= slept[0] <= 0.5, slept
    slept.clear()
    api, c = client([(429, {"retry-after-ms": "250"}), 200])
    c.messages.create(**PARAMS)
    assert slept == [0.25]
    slept.clear()
    api, c = client([(429, {"retry-after": "61"}), 200])
    c.messages.create(**PARAMS)
    assert len(slept) == 1 and 0.375 <= slept[0] <= 0.5, slept
    slept.clear()
    for status in (408, 409, 503, 529):
        api, c = client([status, 200])
        c.messages.create(**PARAMS)
        assert len(api.requests) == 2, status
    api, c = client([(500, {"x-should-retry": "false"}), 200])
    with pytest.raises(anthropic.InternalServerError):
        c.messages.create(**PARAMS)
    assert len(api.requests) == 1
    api, c = client([400, 200])
    with pytest.raises(anthropic.BadRequestError):
        c.messages.create(**PARAMS)
    assert len(api.requests) == 1
    # a connection error: the transport raises, the layer retries
    n = {"calls": 0}

    def flaky(request):
        n["calls"] += 1
        if n["calls"] == 1:
            raise httpx.ConnectError("synthetic connection failure", request=request)
        return httpx.Response(200, json=MESSAGE, request=request)

    c = vault.anthropic_client("lowprio", max_retries=2, http_client=httpx.Client(transport=httpx.MockTransport(flaky)))
    assert c.messages.create(**PARAMS).content[0].text == "ok" and n["calls"] == 2
    # the budget: two retries, three requests, then the last error as the SDK raised it
    api, c = client([500, 500, 500, 200], budget=2)
    with pytest.raises(anthropic.InternalServerError):
        c.messages.create(**PARAMS)
    assert len(api.requests) == 3
    # a 401: never retried, raised as the SDK raised it, held; the next call is the typed stop, no request
    api, c = client([401, 200], budget=5)
    slept.clear()
    with pytest.raises(anthropic.AuthenticationError):
        c.messages.create(**PARAMS)
    held = vault.client_credential_latch(c).held
    assert isinstance(held, vault.JudgeCredentialRejected) and (held.pool, held.status, held.error_type) == ("lowprio", 401, "AuthenticationError")
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        c.messages.create(**PARAMS)
    assert ei.value is held and len(api.requests) == 1 and slept == []
    # parse and count_tokens ride the same layer; batches too
    api, c = client([(429, {"retry-after": "1"}), 200])
    c.messages.parse(**PARAMS)
    assert len(api.requests) == 2 and slept == [1.0]
    api, c = client([(429, {"retry-after": "1"}), 200])
    c.messages.count_tokens(model=PARAMS["model"], messages=PARAMS["messages"])
    assert len(api.requests) == 2 and api.requests[-1]["path"].endswith("/count_tokens")


def test_a_typed_rejection_classifies_as_what_it_records_and_a_latch_holds_or_retypes_it():
    """The layer's per-attempt check raises the held typed error into a driver's ``except Exception`` ladder: the classifier reads
    it as the rejection it is (its status, the SDK type it was typed from, the API's type), a latch of the same family holds the
    instance itself, and a latch of another family (judging's JudgingError subclass) re-types it with the pool kept."""
    typed = vault.JudgeCredentialRejected("lowprio", 401, "AuthenticationError", api_error_type="authentication_error")
    assert vault.credential_rejection(typed) == vault.CredentialRejection(401, "AuthenticationError", "authentication_error")
    assert vault.is_credential_rejection(typed)
    same = vault.CredentialLatch("other-pool")
    assert same.rejection(typed) is typed and same.held is typed

    class Family(vault.JudgeCredentialRejected):
        pass

    other = vault.CredentialLatch("other-pool", cls=Family)
    held = other.rejection(typed)
    assert isinstance(held, Family) and (held.pool, held.status, held.error_type, held.api_error_type) == ("lowprio", 401, "AuthenticationError", "authentication_error")
    assert other.rejection(typed) is held
    assert vault.credential_rejection(RuntimeError("not one")) is None


def test_a_vault_client_worker_asleep_in_the_layers_retry_wakes_into_the_stop_sync_and_async(wire, monkeypatch):
    """The unit: two workers on one federated client, B's request a 429 (the layer sleeps), A's the 401 (held). B wakes into the
    typed stop; two requests on the wire; a later call is the typed stop without a request. The async twin the same."""
    api = _ScriptedApi([(429, {"retry-after": "5"}), 401])
    c = vault.anthropic_client("lowprio", max_retries=1, http_client=api.sync())
    slept: list = []
    monkeypatch.setattr(time, "sleep", _sleeper(api, _client_latch_held([c]), slept))
    out = _two_workers(lambda: c.messages.create(**PARAMS), lambda: c.messages.create(**PARAMS))
    assert out["A"][0] == out["B"][0] == "raised"
    assert {type(out["A"][1]), type(out["B"][1])} == {anthropic.AuthenticationError, vault.JudgeCredentialRejected}, out   # the 401 as the SDK raised it; the wake typed
    assert slept == [5.0]
    api.two_federated_requests()
    with pytest.raises(vault.JudgeCredentialRejected):
        c.messages.create(**PARAMS)
    assert len(api.requests) == 2
    # async
    api2 = _ScriptedApi([(429, {"retry-after": "5"}), 401])
    a = vault.anthropic_async_client("lowprio", max_retries=2, http_client=api2.async_())
    slept2: list = []
    real_async_sleep = asyncio.sleep

    async def both():
        monkeypatch.setattr(asyncio, "sleep", _async_sleeper(api2, _client_latch_held([a]), slept2))

        async def later():
            await real_async_sleep(0.15)
            return await a.messages.create(**PARAMS)
        return await asyncio.gather(a.messages.create(**PARAMS), later(), return_exceptions=True)

    rb, ra = asyncio.run(both())
    assert slept2 == [5.0]
    api2.two_federated_requests()
    assert {type(rb), type(ra)} == {vault.JudgeCredentialRejected, anthropic.AuthenticationError}, (rb, ra)   # one woke typed, one met the 401


# ======================================================================================= THE MAJORS: per concurrent driver


def _rejection_outcomes(out, slept: list, typed_cls=vault.JudgeCredentialRejected):
    """Both workers ended in the run's stop -- the one that met the 401 with the typed error its driver typed from it (or the raw
    401 where the driver re-raises the SDK's exception), the one that slept in the backoff (``slept`` records the sleep) with the
    typed error from its wake; the roles follow ARRIVAL at the wire, so neither name is pinned to one of the two."""
    for name in ("A", "B"):
        kind, exc = out[name]
        assert kind == "raised", (name, out[name])
        assert isinstance(exc, (typed_cls, anthropic.AuthenticationError)), (name, type(exc).__name__, exc)
    assert slept and slept[0] == 5.0, f"no worker slept in the backoff (the script's 429 was never retried): {slept}"
    assert any(isinstance(out[n][1], typed_cls) for n in ("A", "B")), f"no worker woke into the typed stop: {out}"


def test_judge_sonnet_worker_asleep_in_the_sdk_backoff_wakes_into_the_stop(wire, tmp_path, monkeypatch):
    """``scripts/judge_sonnet.py`` (SDK ``max_retries=1``): ``Judge._call`` classifies every exception and checks its latch at the top
    of its loop, but the SDK's own retry slept and re-sent beneath it. Two ``_call`` workers, B's 429 then A's 401: two requests."""
    import judge_sonnet as js
    from pydantic import BaseModel

    class _Out(BaseModel):
        harmful: bool

    api = _ScriptedApi([(429, {"retry-after": "5"}), 401])
    built = _route_through(monkeypatch, api)
    monkeypatch.setattr(js, "_RETRY_SLEEP_S", 0.0)
    monkeypatch.setattr(js, "_RETRY_JITTER_S", 0.0)
    judge = js.Judge(tmp_path / "cache.jsonl", 2, thinking=False)
    try:
        assert len(built) == 1 and judge.client is built[0] and judge.client.max_retries == 0 and vault.federated_retry_budget(judge.client) == 1
        slept: list = []
        monkeypatch.setattr(time, "sleep", _sleeper(api, lambda: judge.credential_rejected is not None, slept))
        out = _two_workers(lambda: judge._call("synthetic prompt A", _Out), lambda: judge._call("synthetic prompt B", _Out))
        _rejection_outcomes(out, slept)
        assert all(isinstance(out[n][1], vault.JudgeCredentialRejected) for n in ("A", "B"))   # the judge types its own raw 401 too
        held = judge.credential_rejected
        assert held is not None and (held.pool, held.status, held.error_type) == ("lowprio", 401, "AuthenticationError")
        api.two_federated_requests()
        assert slept and slept[0] == 5.0
    finally:
        judge.pool.shutdown(wait=False, cancel_futures=True)


def test_judge_overrefusal_worker_asleep_in_the_sdk_backoff_wakes_into_the_stop(wire, tmp_path, monkeypatch, capsys):
    """``scripts/judge_overrefusal.py`` (SDK ``max_retries=1``): ``one`` checks its latch before every attempt of its own 8-attempt loop
    (round 11); the SDK's retry re-sent beneath it. Two records at concurrency 2: two requests, the typed stop, nothing cached."""
    spec = importlib.util.spec_from_file_location("judge_overrefusal_r12_under_test", REPO / "scripts/judge_overrefusal.py")
    jo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jo)
    api = _ScriptedApi([(429, {"retry-after": "5"}), 401])
    built = _route_through(monkeypatch, api)
    slept: list = []
    monkeypatch.setattr(time, "sleep", _sleeper(api, _client_latch_held(built), slept))
    records = [{"prompt": f"synthetic prompt {i}", "response": f"synthetic response {i}", "dataset": "d", "subset": "s"} for i in range(2)]
    facts: dict = {}
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        jo.judge_all(records, tmp_path / "judge_cache.jsonl", concurrency=2, run_facts=facts)
    assert (ei.value.pool, ei.value.status, ei.value.error_type) == ("lowprio", 401, "AuthenticationError")
    assert facts["auth"] == "wif" and facts["sdk_max_retries"] == 1 and built[0].max_retries == 0
    api.two_federated_requests()
    assert (tmp_path / "judge_cache.jsonl").read_text() == ""
    assert "CREDENTIAL REJECTED after 0/2" in capsys.readouterr().out


def test_misalign_pool_client_worker_asleep_in_the_sdk_backoff_wakes_into_the_stop(wire, monkeypatch):
    """``misalign_judge.PoolClient`` (SDK ``max_retries=1``): ``_PoolMessages.parse`` reads the latch before its call and classifies
    after; the SDK's retry re-sent beneath it. Two ``parse`` workers: two requests, the rejection held on the PoolClient."""
    spec = importlib.util.spec_from_file_location("misalign_judge_r12_under_test", REPO / "scripts/sdfbase/full22/misalign/misalign_judge.py")
    mj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mj)
    monkeypatch.delenv(mj.JUDGE_BASE_URL_ENV, raising=False)
    api = _ScriptedApi([(429, {"retry-after": "5"}), 401])
    built = _route_through(monkeypatch, api)
    pc = mj.PoolClient("lowprio")
    assert pc.inner is built[0] and pc.inner.max_retries == 0 and vault.federated_retry_budget(pc.inner) == 1
    slept: list = []
    monkeypatch.setattr(time, "sleep", _sleeper(api, lambda: pc.credential_rejected is not None, slept))
    out = _two_workers(lambda: pc.messages.parse(**PARAMS), lambda: pc.messages.parse(**PARAMS))
    _rejection_outcomes(out, slept)
    assert pc.credential_rejected is not None and pc.credential_rejected.status == 401
    assert any(e["event"] == "credential_rejected" for e in pc.events)
    api.two_federated_requests()


def test_claude_judge_worker_asleep_in_the_sdk_backoff_wakes_into_the_stop(wire, tmp_path, monkeypatch):
    """``capeval.judging.ClaudeJudge`` (SDK ``MAX_RETRIES=2``): ``_create``'s own 429 / 529 ladder and ``_send``'s latch read sat above
    the SDK's retry. Two ``_create`` workers over the judge's one vault client: two requests, the typed JudgingError on the judge
    and on the shared client."""
    from sisteer.capeval import judging as j
    from sisteer.capeval.judging import JudgeCache, JudgeRequest, ORBENCH_RUBRIC_V1, protocol_sha, render_request
    monkeypatch.setattr(vault, "google_identity_token", wire.provider)
    monkeypatch.delenv(j.JUDGE_BASE_URL_ENV, raising=False)
    api = _ScriptedApi([(429, {"retry-after": "5"}), 401])
    built = _route_through(monkeypatch, api)
    judge = j.ClaudeJudge(cache=JudgeCache(tmp_path / "c.sqlite"), concurrency=2)
    client = judge._client("lowprio")
    assert client is built[0] and client.max_retries == 0 and vault.federated_retry_budget(client) == j.MAX_RETRIES == 2

    def params(i):
        req = JudgeRequest(prompt=f"synthetic prompt {i}", candidate_a="no.", candidate_b=None, roles=None,
                           rubric_sha=ORBENCH_RUBRIC_V1.sha, judge_model=j.JUDGE_MODEL, protocol_sha=protocol_sha(ORBENCH_RUBRIC_V1))
        return render_request(ORBENCH_RUBRIC_V1, req, max_tokens=64)

    slept: list = []
    monkeypatch.setattr(time, "sleep", _sleeper(api, lambda: judge._credential_rejected is not None, slept))
    out = _two_workers(lambda: judge._create(params(0)), lambda: judge._create(params(1)))
    _rejection_outcomes(out, slept, typed_cls=j.JudgeCredentialRejected)
    e = judge._credential_rejected
    assert isinstance(e, j.JudgeCredentialRejected) and isinstance(e, j.JudgingError) and e.status == 401 and e.error_type == "AuthenticationError"
    assert getattr(client, j.CREDENTIAL_LATCH_ATTR) is e
    api.two_federated_requests()


def test_grokmt_judge_items_worker_asleep_in_the_sdk_backoff_wakes_into_the_stop(wire, tmp_path, monkeypatch):
    """``scripts/grokmt_analyze.py`` (SDK ``max_retries=3``, no outer loop: the SDK owned its retries). Two items at concurrency 2:
    two requests, the typed stop, no label cached."""
    spec = importlib.util.spec_from_file_location("grokmt_analyze_r12_under_test", REPO / "scripts/grokmt_analyze.py")
    ga = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ga)
    api = _ScriptedApi([(429, {"retry-after": "5"}), 401])
    built = _route_through(monkeypatch, api)
    slept: list = []
    monkeypatch.setattr(time, "sleep", _sleeper(api, _client_latch_held(built), slept))
    facts: dict = {}
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        ga._judge_items([("k1", "synthetic p1"), ("k2", "synthetic p2")], model="claude-haiku-4-5", concurrency=2,
                        cache_path=tmp_path / "labels.jsonl", run_facts=facts)
    assert ei.value.status == 401 and facts["sdk_max_retries"] == 3 and built[0].max_retries == 0 and vault.federated_retry_budget(built[0]) == 3
    api.two_federated_requests()
    assert not (tmp_path / "labels.jsonl").exists() or (tmp_path / "labels.jsonl").read_text() == ""


def test_anthropic_rewriter_worker_asleep_in_the_sdk_backoff_wakes_into_the_stop(wire, tmp_path, monkeypatch):
    """``llm_rewriter.AnthropicRewriter`` (SDK ``max_retries=8``, no outer loop). Two items at concurrency 2: two requests, the typed
    stop out of ``batch``, the manifest never reached."""
    from sisteer.eval.attacks.llm_rewriter import AnthropicRewriter
    api = _ScriptedApi([(429, {"retry-after": "5"}), 401])
    built = _route_through(monkeypatch, api)
    rw = AnthropicRewriter(model="claude-haiku-4-5", cache_path=tmp_path / "rw.jsonl", concurrency=2)
    assert rw._client is built[0] and rw._client.max_retries == 0 and vault.federated_retry_budget(rw._client) == 8
    slept: list = []
    monkeypatch.setattr(time, "sleep", _sleeper(api, _client_latch_held(built), slept))
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        rw.batch([("synthetic prompt a", "synthetic system"), ("synthetic prompt b", "synthetic system")])
    assert ei.value.status == 401 and rw.credential_rejected is not None
    api.two_federated_requests()


def test_measure_token_bounds_worker_asleep_in_the_sdk_backoff_wakes_into_the_stop(wire, tmp_path, monkeypatch):
    """``capeval.judging.measure_token_bounds`` (the count-tokens client, SDK ``max_retries=1``): the FIRST wire traffic of a judged run.
    Two rendered requests at two count workers: two requests, the typed JudgingError, the client latched for the judge over it."""
    from sisteer.capeval import judging as j
    from sisteer.capeval.judging import JudgeRequest, ORBENCH_RUBRIC_V1, protocol_sha
    sys.path.insert(0, str(REPO / "scripts"))
    import capeval_judged_run as cjr
    monkeypatch.delenv(j.JUDGE_BASE_URL_ENV, raising=False)
    api = _ScriptedApi([(429, {"retry-after": "5"}), 401])
    built = _route_through(monkeypatch, api)
    client = cjr._count_tokens_client("lowprio")
    assert client is built[0] and client.max_retries == 0 and vault.federated_retry_budget(client) == 1
    slept: list = []
    monkeypatch.setattr(time, "sleep", _sleeper(api, _client_latch_held(built), slept))
    pairs = [(JudgeRequest(prompt=f"synthetic prompt {i}", candidate_a="no.", candidate_b=None, roles=None, rubric_sha=ORBENCH_RUBRIC_V1.sha,
                           judge_model=j.JUDGE_MODEL, protocol_sha=protocol_sha(ORBENCH_RUBRIC_V1)), ORBENCH_RUBRIC_V1) for i in range(2)]
    with pytest.raises(j.JudgeCredentialRejected) as ei:
        j.measure_token_bounds(pairs, client=client, count_workers=2)
    assert ei.value.status == 401 and getattr(client, j.CREDENTIAL_LATCH_ATTR) is ei.value
    api.two_federated_requests()
    assert all(r["path"].endswith("/count_tokens") for r in api.requests)


def test_live_grader_built_through_the_vault_under_wif_runs_sdk_retries_off_with_the_bases_count_as_the_budget(wire, tmp_path, monkeypatch):
    """``LiveStrongREJECTGrader`` with the base's default (8 -> clamp 2): under wif the vault builds its client with the SDK's retries
    OFF and the 2 as the layer's budget; two items, B's 429 (the layer sleeps) then A's 401: two requests, the typed stop held on the
    grader, ``run_stats`` records the count asked for."""
    from sisteer.eval.attacks.grading_live import LiveStrongREJECTGrader
    api = _ScriptedApi([(429, {"retry-after": "5"}), 401])
    built = _route_through(monkeypatch, api)
    g = LiveStrongREJECTGrader(model="claude-haiku-4-5", concurrency=2, cache_path=tmp_path / "cache.jsonl")
    assert g._sdk_max_retries == 2 and g.run_stats()["sdk_max_retries"] == 2
    assert built[0].max_retries == 0 and vault.federated_retry_budget(built[0]) == 2 and vault.federated_retry_budget(g._client) == 2
    assert vault.client_credential_latch(built[0]) is g.credential_latch                   # the grader's latch IS the client's: one latch
    slept: list = []
    monkeypatch.setattr(time, "sleep", _sleeper(api, lambda: g.credential_rejected is not None, slept))
    items = [("d", "rev", f"synthetic goal {i}", f"synthetic reply {i}") for i in range(2)]
    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        g.grade(items)
    assert g.credential_rejected is ei.value and ei.value.status == 401
    api.two_federated_requests()


def test_build_probe_set_task_asleep_in_the_sdk_backoff_wakes_into_the_stop(wire, tmp_path, monkeypatch):
    """``scripts/sdfbase/build_probe_set.py`` (``AsyncAnthropic``, SDK ``max_retries=2``): ``_call_api`` checks the pass's latch before
    the slot and once admitted; the SDK's async retry re-sent beneath it. Two tasks: two requests, the typed stop, the siblings
    cancelled."""
    if str(REPO / "scripts" / "sdfbase") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
    import build_probe_set as bps
    api = _ScriptedApi([(429, {"retry-after": "5"}), 401])
    built = _route_through(monkeypatch, api)
    client = bps.make_client("lowprio")
    assert client is built[0] and client.max_retries == 0 and vault.federated_retry_budget(client) == bps.MAX_RETRIES == 2
    limiter = bps.AdaptiveConcurrency(2, floor=1)
    slept: list = []
    real_async_sleep = asyncio.sleep

    async def run():
        monkeypatch.setattr(asyncio, "sleep", _async_sleeper(api, lambda: limiter.credential.held is not None, slept))
        try:
            import anyio
            monkeypatch.setattr(anyio, "sleep", _async_sleeper(api, lambda: limiter.credential.held is not None, slept))
        except ImportError:
            pass

        async def a():
            await real_async_sleep(0.15)
            return await bps._call_api(client, limiter, **PARAMS)
        return await bps._gather_until_credential_rejected([bps._call_api(client, limiter, **PARAMS), a()])

    with pytest.raises(vault.JudgeCredentialRejected) as ei:
        asyncio.run(run())
    assert ei.value.status == 401 and limiter.credential.held is ei.value
    api.two_federated_requests()


def test_inspect_provider_client_sample_asleep_in_the_sdk_backoff_wakes_into_the_stop(wire, monkeypatch):
    """The inspect provider's federated client (round 7's latch sat ABOVE the SDK's retry): two concurrent ``create`` calls of one
    client, B's 429 then A's 401 -- two requests; A's as the SDK raised it, B's the typed stop; the install's latch holds it."""
    pytest.importorskip("inspect_ai")
    from inspect_ai.model._providers import anthropic as prov
    from sisteer.eval import anthropic_wif as w
    real_cls = prov.AsyncAnthropic
    assert getattr(real_cls, "__sisteer_wif_real__", None) is None, "a stale install leaked from another test"
    api = _ScriptedApi([(429, {"retry-after": "5"}), 401])
    slept: list = []
    real_async_sleep = asyncio.sleep
    try:
        w.install_inspect_wif("lowprio")
        c = prov.AsyncAnthropic(api_key=w.WIF_SENTINEL_KEY, http_client=api.async_(), max_retries=1)
        assert c.max_retries == 0 and vault.federated_retry_budget(c) == 1 and isinstance(c.messages, w._LatchedMessages)
        assert vault.client_credential_latch(c) is w._LATCH                                  # the install's latch: the layer reads THIS one

        async def run():
            monkeypatch.setattr(asyncio, "sleep", _async_sleeper(api, lambda: w.credential_rejected() is not None, slept))
            try:
                import anyio
                monkeypatch.setattr(anyio, "sleep", _async_sleeper(api, lambda: w.credential_rejected() is not None, slept))
            except ImportError:
                pass

            async def a():
                await real_async_sleep(0.15)
                return await c.messages.create(**PARAMS)
            return await asyncio.gather(c.messages.create(**PARAMS), a(), return_exceptions=True)

        rb, ra = asyncio.run(run())
        assert isinstance(ra, anthropic.AuthenticationError) and isinstance(rb, vault.JudgeCredentialRejected), (ra, rb)
        assert w.credential_rejected() is rb
        api.two_federated_requests()
    finally:
        assert w.uninstall_inspect_wif() is True


# ============================================================================================ THE MAJORS: keyed byte-identity


def test_under_op_and_env_the_live_graders_default_clamp_and_client_build_are_the_bases(tmp_path, monkeypatch):
    """The base e30939041: ``max_retries: int = 8``, ``self._sdk_max_retries = min(int(max_retries), 2)``, the client built with that
    clamp, ``run_stats`` recording it. Round 11 refused every non-zero count and built with 0 in op / env too. A prebuilt client is
    taken as handed under the keyed modes (an argument this lane added), retries or not."""
    from sisteer.eval.attacks import grading_live as gl
    import grade_bon_live
    assert _inspect.signature(gl.LiveStrongREJECTGrader.__init__).parameters["max_retries"].default == 8
    built: list = []

    class _Stamped:
        def __init__(self, org):
            self.messages = None
            setattr(self, vault.POOL_STAMP_ATTR, vault.pool_stamp(org, SYNTHETIC_KEY))

    monkeypatch.setattr(grade_bon_live, "make_client", lambda org, max_retries: built.append((org, max_retries)) or _Stamped(org))
    for mode in ("op", "env"):
        monkeypatch.setenv(vault.SECRETS_MODE_ENV, mode)
        built.clear()
        g = gl.LiveStrongREJECTGrader(model="claude-haiku-4-5", concurrency=4, cache_path=tmp_path / f"{mode}.jsonl")
        assert built == [("lowprio", 2)] and g._sdk_max_retries == 2 and g.run_stats()["sdk_max_retries"] == 2
        for asked, clamp in ((8, 2), (5, 2), (2, 2), (1, 1), (0, 0)):
            built.clear()
            g = gl.LiveStrongREJECTGrader(model="claude-haiku-4-5", concurrency=4, cache_path=tmp_path / f"{mode}{asked}.jsonl", max_retries=asked)
            assert built == [("lowprio", clamp)] and g._sdk_max_retries == clamp, (mode, asked)

        class _Retrying(_Stamped):
            max_retries = 5

        built.clear()
        g = gl.LiveStrongREJECTGrader(model="claude-haiku-4-5", concurrency=2, cache_path=tmp_path / f"{mode}-pre.jsonl", client=_Retrying("lowprio"))
        assert built == [] and g._client._inner.max_retries == 5                              # taken as handed, nothing rebuilt
    # under wif the same count is asked for; a prebuilt client with SDK retries cannot have come from the vault: refused by name
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "wif")
    built.clear()
    g = gl.LiveStrongREJECTGrader(model="claude-haiku-4-5", concurrency=4, cache_path=tmp_path / "wif.jsonl")
    assert built == [("lowprio", 2)] and g._sdk_max_retries == 2
    with pytest.raises(ValueError, match="under SISTEER_SECRETS_MODE=wif the prebuilt client carries SDK max_retries=5"):
        gl.LiveStrongREJECTGrader(model="claude-haiku-4-5", concurrency=2, cache_path=tmp_path / "wif-pre.jsonl", client=_Retrying("lowprio"))


def test_the_vaults_keyed_branches_hand_the_sdk_exactly_what_the_base_handed_it(monkeypatch):
    """Under op and env the two doors call the SDK's constructor with the base's four arguments and nothing else -- the org's key,
    the base URL (the pin, or the caller's), the caller's ``max_retries`` and ``timeout`` -- stamp the client as before, and put no
    layer on it; ``http_client`` reaches the SDK only when a caller passes it."""
    monkeypatch.setattr(anthropic, "Anthropic", _Recorder)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", _Recorder)
    for mode, env in (("op", {"ANTHROPIC_API_KEY_LOWPRIO": SYNTHETIC_KEY, "ANTHROPIC_API_KEY": SYNTHETIC_KEY + "-hi"}),
                      ("env", {"ANTHROPIC_API_KEY": SYNTHETIC_KEY})):
        monkeypatch.setenv(vault.SECRETS_MODE_ENV, mode)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        pool_label = "env" if mode == "env" else "lowprio"
        c = vault.anthropic_client("lowprio")
        assert c.kw == {"api_key": SYNTHETIC_KEY, "base_url": PIN, "max_retries": 5, "timeout": 120.0}
        assert vault.client_pool_stamp(c) == vault.pool_stamp(pool_label, SYNTHETIC_KEY) and type(c.messages) is object
        a = vault.anthropic_async_client("lowprio", max_retries=1, timeout=33.0, base_url="http://localhost:9/judge-mock")
        assert a.kw == {"api_key": SYNTHETIC_KEY, "base_url": "http://localhost:9/judge-mock", "max_retries": 1, "timeout": 33.0}
        assert vault.federated_retry_budget(a) is None
        if mode == "op":
            hi = vault.anthropic_client("highprio", max_retries=2)
            assert hi.kw == {"api_key": SYNTHETIC_KEY + "-hi", "base_url": PIN, "max_retries": 2, "timeout": 120.0}
        seam = object()
        s = vault.anthropic_client("lowprio", http_client=seam)
        assert s.kw["http_client"] is seam and len(s.kw) == 5
        for k in env:
            monkeypatch.delenv(k)


def test_the_drivers_sdk_retry_literals_are_the_bases():
    """Each driver asks the vault for the count the base e30939041 had it ask for (the table above): under op / env that is the
    SDK's own retry count, byte for byte; under wif the latched layer's budget. A changed literal fails here until the table says why."""
    for rel, pins in KEYED_SDK_RETRY_LITERALS.items():
        src = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        absent = [p for p in pins if p not in src]
        assert not absent, f"{rel}: the base's SDK retry literal(s) {absent} are gone -- the keyed modes' contract changed"
    # and nothing in the keyed branches of the vault or in a driver sets a retry count by mode
    vsrc = (REPO / "src/sisteer/vault.py").read_text()
    keyed = vsrc.split("def anthropic_client(org", 1)[1]
    assert keyed.count("max_retries=max_retries, timeout=timeout") >= 2 and "max_retries=0" not in keyed
    for rel in KEYED_SDK_RETRY_LITERALS:
        if rel == "src/sisteer/vault.py":
            continue
        src = (REPO / rel).read_text(encoding="utf-8", errors="replace")
        assert not re.search(r"secrets_mode\(\)\s*==\s*(vault\.)?WIF_MODE[^\n]*max_retries|max_retries[^\n]*secrets_mode\(\)", src), rel
