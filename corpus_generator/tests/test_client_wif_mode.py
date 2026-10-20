"""The wif secrets mode (the front door's mode since the org moved to workload identity federation):
under ``SISTEER_SECRETS_MODE=wif`` the clients carry the SDK's ``WorkloadIdentityCredentials`` over the pool's ids file --
the batch pool's for kind ``batch``, the loprio pool's for ``smoke`` / ``live`` -- with NO API key: ``read_key`` refuses by
name, the secret-manager CLI is never called, ``ANTHROPIC_API_KEY`` is never read (the SDK consults no credential variable
once ``credentials=`` is passed; ``api_key`` stays None), the base URL stays pinned, ``with_options`` keeps the credentials
(batch.py's ``max_retries=0`` creator), an absent / empty / incomplete ids file refuses by name (never a fallback to another
file -- a similarly named ids file exists EMPTY beside the real one), and guard 3 refuses a kind whose ids are the
highprio pool's. op and env are unchanged (test_client_env_mode.py, test_client_guard.py). Fakes ride the ``token_provider=``
seam; no network."""

from __future__ import annotations

import hashlib
from pathlib import Path

import anthropic
import pytest

from corpus_generator import client as C
from corpus_generator.client import (
    API_BASE_URL,
    ClientGuardError,
    build_async_client,
    build_client,
    read_key,
    secrets_mode,
    wif_credentials,
    wif_ids,
    wif_pool_stamp,
)

IDS = {"ANTHROPIC_FEDERATION_RULE_ID_GCP": "fr_11111111", "ANTHROPIC_ORGANIZATION_ID": "org_22222222",
       "ANTHROPIC_SERVICE_ACCOUNT_ID": "sa_33333333", "ANTHROPIC_WORKSPACE_ID": "ws_44444444"}
#: a value with NO credential shape; the assertion is that the variable is never consulted
SENTINEL = "env-value-" + "never-read-" * 2


def _ids(suffix: str) -> dict:
    return {k: f"{v}{suffix}" for k, v in IDS.items()}


def _write(path: Path, ids: dict, comment: str = "") -> None:
    path.write_text(comment + "".join(f"{k}={v}\n" for k, v in ids.items()))


def _ids_dir(tmp_path: Path, monkeypatch) -> Path:
    d = tmp_path / "anthropic"
    d.mkdir()
    _write(d / C.WIF_FILES["smoke"], _ids(""), "# the lowprio pool\n")       # the pools' files, as the client names them
    _write(d / C.WIF_FILES["batch"], _ids("-batch"))
    _write(d / C.HIGHPRIO_WIF_FILE, _ids("-hi"))
    (d / "wif-lowprio.env").write_text("")
    monkeypatch.setenv(C.WIF_ENV_DIR_ENV, str(d))
    return d


@pytest.fixture
def wif(tmp_path, monkeypatch):
    """wif mode with every credential door nailed shut: the secret-manager CLI is off PATH and its runner raises, the env key
    is a sentinel that must never be read, ANTHROPIC_BASE_URL names a gateway that must never be inherited."""
    d = _ids_dir(tmp_path, monkeypatch)
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "wif")
    monkeypatch.setenv("ANTHROPIC_API_KEY", SENTINEL)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:1/session-gateway")
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    (tmp_path / "empty-bin").mkdir()

    def boom(*a, **k):  # pragma: no cover - the assertion IS the test
        raise AssertionError(f"the secret-manager CLI must never run under wif: {a}")

    monkeypatch.setattr(C.subprocess, "run", boom)
    return d


def _creds_ids(creds) -> dict:
    """The ids the SDK's credentials object was built on (its private fields; the SDK exposes no reader)."""
    return {"ANTHROPIC_FEDERATION_RULE_ID_GCP": creds._federation_rule_id, "ANTHROPIC_ORGANIZATION_ID": creds._organization_id,
            "ANTHROPIC_SERVICE_ACCOUNT_ID": creds._service_account_id, "ANTHROPIC_WORKSPACE_ID": creds._workspace_id}


def _assert_federated(client, ids: dict, *, async_client: bool = False):
    assert isinstance(client, anthropic.AsyncAnthropic if async_client else anthropic.Anthropic)
    assert client.api_key is None and client.auth_token is None, "a wif client holds no API key -- the env was never read"
    assert isinstance(client.credentials, anthropic.WorkloadIdentityCredentials)
    assert _creds_ids(client.credentials) == ids
    assert str(client.base_url).rstrip("/") == API_BASE_URL, "base URL stays pinned under wif"
    # round 15: the SDK is handed the member's own httpx client over the pin guard, never its default client
    own = getattr(client, "_client")
    assert type(own) is (C.FederatedAsyncHttpxClient if async_client else C.FederatedHttpxClient), type(own)
    assert type(own._transport) is C.PinnedTransport and str(own.base_url).rstrip("/") == API_BASE_URL


def test_wif_mode_federates_every_kind_on_its_own_pools_ids_keyless(wif):
    assert secrets_mode() == "wif"
    provider = lambda: "google-id-token-stub"  # noqa: E731
    batch = build_client("batch", token_provider=provider)
    _assert_federated(batch, _ids("-batch"))
    smoke = build_client("smoke", token_provider=provider)
    _assert_federated(smoke, _ids(""))
    live = build_async_client("live", token_provider=provider)
    _assert_federated(live, _ids(""), async_client=True)
    # round 14: under wif the SDK's OWN retries are off on every client (it re-sent the first 401 of a federated client with a fresh
    # exchange); the count asked for -- build_client's default 4, build_async_client's 0 -- is the member's latched retry layer's budget
    assert live.max_retries == 0 and batch.max_retries == 0
    assert C.federated_retry_budget(batch) == 4 and C.federated_retry_budget(live) == 0 and C.federated_retry_budget(smoke) == 4
    # the identity token is the injected provider's, fetched by the SDK at exchange time, never by this module at build time
    assert batch.credentials._identity_token_provider is provider
    # batch.py's creator: with_options keeps the credentials and the pin (a double-submit guard, not a credential change) -- a federated
    # copy since round 14, its layer at budget 0
    creator = batch.with_options(max_retries=0)
    assert creator.max_retries == 0 and creator.api_key is None and C.federated_retry_budget(creator) == 0 and type(creator) is type(batch)
    assert isinstance(creator.credentials, anthropic.WorkloadIdentityCredentials) and str(creator.base_url).rstrip("/") == API_BASE_URL
    # the keyless stamp: the pool's name, auth wif, the file's digest -- never a token, never an id
    st = wif_pool_stamp("batch")
    assert st == {"pool": "batch", "auth": "wif", "ids_sha12": hashlib.sha256((wif / C.WIF_FILES["batch"]).read_bytes()).hexdigest()[:12]}
    assert wif_pool_stamp("live")["pool"] == wif_pool_stamp("smoke")["pool"] == "lowprio"
    assert "org_22222222" not in str(st) and "google-id-token-stub" not in str(st)


def test_wif_mode_has_no_key_to_read_and_never_consults_the_env_key_or_the_vault(wif, monkeypatch):
    for kind in ("batch", "smoke", "live"):
        with pytest.raises(ClientGuardError, match="NO API key") as ei:
            read_key(kind)
        assert "SISTEER_SECRETS_MODE=wif" in str(ei.value) and "never the secret-manager CLI" in str(ei.value) and SENTINEL not in str(ei.value)
    # the env key is a sentinel that must never be read: with every read of it made to fail, the clients still build
    real_get = C.os.environ.get

    def guarded_get(key, default=None):
        assert key != "ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY read under wif"
        return real_get(key, default)

    monkeypatch.setattr(C.os.environ, "get", guarded_get, raising=False)
    client = build_client("smoke", token_provider=lambda: "t")
    assert client.api_key is None
    # the ids file is the ONE file read: wif_ids reads it whole and keeps the four ids, comments and blanks skipped
    assert wif_ids("smoke") == _ids("") and wif_ids("batch") == _ids("-batch")


def test_an_absent_empty_or_incomplete_ids_file_refuses_by_name_and_never_falls_back(wif):
    (wif / C.WIF_FILES["batch"]).unlink()
    with pytest.raises(ClientGuardError, match="does not exist") as ei:
        build_client("batch", token_provider=lambda: "t")
    assert str(wif / C.WIF_FILES["batch"]) in str(ei.value) and "batch pool" in str(ei.value)
    # smoke / live read the loprio pool's ids file; an EMPTY one refuses by name, no other file is tried
    (wif / C.WIF_FILES["smoke"]).write_text("")
    with pytest.raises(ClientGuardError, match="is empty") as ei:
        build_client("smoke", token_provider=lambda: "t")
    assert C.WIF_FILES["smoke"] in str(ei.value) and "each pool reads only its own ids file" in str(ei.value)
    _write(wif / C.WIF_FILES["smoke"], {k: v for k, v in IDS.items() if k != "ANTHROPIC_WORKSPACE_ID"})
    with pytest.raises(ClientGuardError, match="incomplete") as ei:
        build_async_client("live", token_provider=lambda: "t")
    assert "ANTHROPIC_WORKSPACE_ID" in str(ei.value)
    with pytest.raises(ClientGuardError, match="unknown client kind"):
        wif_ids("highprio")  # type: ignore[arg-type]


def test_guard_3_under_wif_refuses_a_kind_whose_ids_are_the_highprio_pools(wif):
    _write(wif / C.WIF_FILES["batch"], _ids("-hi"))                                  # the batch file now carries the highprio identity
    with pytest.raises(ClientGuardError, match="HIGHPRIO") as ei:
        wif_credentials("batch", token_provider=lambda: "t")
    assert "never runs on highprio" in str(ei.value) and C.HIGHPRIO_WIF_FILE in str(ei.value)
    with pytest.raises(ClientGuardError, match="HIGHPRIO"):
        build_client("batch", token_provider=lambda: "t")
    # the lowprio file is untouched: smoke / live still build; an absent or incomplete highprio file disables the comparison,
    # exactly as the op-mode guard skips an empty highprio item
    build_client("smoke", token_provider=lambda: "t")
    (wif / C.HIGHPRIO_WIF_FILE).write_text("ANTHROPIC_ORGANIZATION_ID=org_22222222-hi\n")
    build_client("batch", token_provider=lambda: "t")
    (wif / C.HIGHPRIO_WIF_FILE).unlink()
    build_client("batch", token_provider=lambda: "t")


def test_the_default_token_provider_is_the_google_identity_token_and_the_mode_names_a_fourth_value(wif, monkeypatch):
    seen = []
    monkeypatch.setattr(C, "google_identity_token", lambda: seen.append(1) or "google-id-token")
    client = build_client("batch")
    assert client.credentials._identity_token_provider is C.google_identity_token and seen == []   # fetched at exchange, not at build
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "federation")
    with pytest.raises(ClientGuardError, match="must be one of 'op', 'env' or 'wif'"):
        build_client("batch")


def test_under_wif_a_modal_container_refuses_by_name_before_any_construction_unless_a_token_provider_is_injected(wif, monkeypatch):
    """Unlike ``sisteer.vault.wif_credentials``, this member's constructor built the credentials inside a
    Modal container and let the first request fail late on the GCP identity-token route. Now both constructors (sync and async) refuse
    by name there, BEFORE the ids file is read; an injected ``token_provider`` (the future Modal constructor) passes."""
    monkeypatch.setenv(C.MODAL_MARKER_ENV, "ta-synthetic")
    assert C.MODAL_MARKER_ENV == "MODAL_TASK_ID" and C.on_modal() is True and C.on_modal({}) is False
    for build, kind in ((build_client, "batch"), (build_async_client, "live"), (build_client, "smoke")):
        with pytest.raises(ClientGuardError, match="inside a Modal container") as ei:
            build(kind)
        msg = str(ei.value)
        assert "MODAL_TASK_ID" in msg and f"the {C.WIF_POOLS[kind]} pool" in msg and "refusing before any request" in msg
        assert "ANTHROPIC_FEDERATION_RULE_ID_GCP" in msg and "SISTEER_SECRETS_MODE=wif" in msg
    with pytest.raises(ClientGuardError, match="inside a Modal container"):
        wif_credentials("batch")
    # the ids file is not even read: an absent file is not what refuses here
    (wif / C.WIF_FILES["batch"]).unlink()
    with pytest.raises(ClientGuardError, match="inside a Modal container"):
        build_client("batch")
    # an injected provider passes the Modal gate (and then the ids file matters again, as before)
    with pytest.raises(ClientGuardError, match="does not exist"):
        build_client("batch", token_provider=lambda: "t")
    _assert_federated(build_client("smoke", token_provider=lambda: "t"), _ids(""))
    _assert_federated(build_async_client("live", token_provider=lambda: "t"), _ids(""), async_client=True)
    # off Modal the default provider builds as before
    monkeypatch.delenv(C.MODAL_MARKER_ENV)
    _assert_federated(build_client("smoke"), _ids(""))
    _assert_federated(build_async_client("live"), _ids(""), async_client=True)
