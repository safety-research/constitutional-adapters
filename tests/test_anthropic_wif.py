"""sisteer.eval.anthropic_wif: the Petri legs' inspect_ai Anthropic provider routed through workload identity federation on the
VAULT's credentials -- no API key, the provider's own api-key requirement met with a sentinel the federated factory discards
before the SDK sees it, the originals back on uninstall, the leg's env switch, the drive script's wif mode (no secret-manager
CLI, a present key refused, the ids-file digest attested) and the Modal refusal. No network: the token provider is a stub and
the SDK's constructor fetches nothing."""
from __future__ import annotations

import hashlib
import sys
from collections.abc import MutableMapping
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from sisteer import vault  # noqa: E402
from sisteer.eval import anthropic_wif as w  # noqa: E402

IDS = {"ANTHROPIC_FEDERATION_RULE_ID_GCP": "fr_gcp_55555555", "ANTHROPIC_ORGANIZATION_ID": "org_11111111",
       "ANTHROPIC_SERVICE_ACCOUNT_ID": "sa_33333333", "ANTHROPIC_WORKSPACE_ID": "wrkspc_22222222"}


def _env_dir(tmp_path, monkeypatch) -> Path:
    d = tmp_path / "anthropic"; d.mkdir(parents=True, exist_ok=True)
    for stem in ("loprio", "hiprio", "batch"):
        (d / f"wif-{stem}.env").write_text("# synthetic ids (TESTHOST)\n" + "".join(f"{k}={v}-{stem}\n" for k, v in IDS.items()))
    (d / "wif-lowprio.env").write_text("")           # the stray empty file the maintainer's machine carries
    monkeypatch.setenv(vault.WIF_ENV_DIR_ENV, str(d))
    monkeypatch.delenv(vault.MODAL_MARKER_ENV, raising=False)
    return d


def _stub_token():
    return "stub-identity-token"


def test_org_aliases_and_the_leg_switch_parse_by_name():
    assert w.org_name("lowprio") == "lowprio" and w.org_name("loprio") == "lowprio" and w.org_name("hiprio") == "highprio" and w.org_name("BATCH") == "batch"
    with pytest.raises(ValueError, match="unknown Claude API org"):
        w.org_name("medprio")
    assert w.parse_auth_switch(None) is None and w.parse_auth_switch("  ") is None
    assert w.parse_auth_switch("wif:lowprio") == "lowprio" and w.parse_auth_switch("wif:loprio") == "lowprio" and w.parse_auth_switch("wif:highprio") == "highprio"
    with pytest.raises(ValueError, match="expected wif:<org>"):
        w.parse_auth_switch("key:lowprio")
    with pytest.raises(ValueError, match="expected wif:<org>"):
        w.parse_auth_switch("wif:")


def test_the_install_builds_the_vaults_credentials_and_refuses_a_missing_file_or_a_modal_container_by_name(tmp_path, monkeypatch):
    pytest.importorskip("inspect_ai")
    import anthropic
    d = _env_dir(tmp_path, monkeypatch)
    try:
        creds = w.install_inspect_wif("loprio", token_provider=_stub_token)
        assert isinstance(creds, anthropic.WorkloadIdentityCredentials) and w.inspect_wif_org() == "lowprio"
        assert creds._federation_rule_id == IDS["ANTHROPIC_FEDERATION_RULE_ID_GCP"] + "-loprio" and creds._organization_id == IDS["ANTHROPIC_ORGANIZATION_ID"] + "-loprio"
    finally:
        w.uninstall_inspect_wif()
    hi = Path(vault.wif_env_path("highprio")); hi.write_text("")                    # the highprio pool's ids file, as the vault names it
    with pytest.raises(vault.PoolUnavailable, match="is empty") as ei:
        w.install_inspect_wif("highprio", token_provider=_stub_token)
    assert f"{hi.name} is empty" in str(ei.value)
    assert w.inspect_wif_org() is None                                            # a refused install leaves nothing installed
    monkeypatch.setenv(vault.MODAL_MARKER_ENV, "ta-synthetic")                     # a Modal container: the GCP id-token path cannot work there
    with pytest.raises(vault.PoolUnavailable, match="Modal container"):
        w.install_inspect_wif("lowprio")
    assert w.install_inspect_wif("lowprio", token_provider=_stub_token) is not None   # an injected provider (the future Modal constructor) passes
    assert w.uninstall_inspect_wif() is True


def test_the_inspect_provider_builds_federated_clients_after_the_install_and_the_originals_return_on_uninstall(tmp_path, monkeypatch):
    pytest.importorskip("inspect_ai")
    import anthropic
    from inspect_ai.model import get_model
    from inspect_ai.model._providers import anthropic as prov
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)      # the precondition federation needs: no static key in this process
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    _env_dir(tmp_path, monkeypatch); real_cls = prov.AsyncAnthropic; real_init = prov.AnthropicAPI.__init__
    assert getattr(real_cls, "__sisteer_wif_real__", None) is None, "a stale install leaked from another test"
    try:
        creds = w.install_inspect_wif("loprio", token_provider=_stub_token)
        assert w.inspect_wif_org() == "lowprio"
        creds2 = w.install_inspect_wif("lowprio", token_provider=_stub_token)   # idempotent: replaces, never stacks
        assert prov.AsyncAnthropic.__sisteer_wif_real__ is real_cls and prov.AnthropicAPI.__init__.__sisteer_wif_real__ is real_init
        m = get_model("anthropic/claude-sonnet-5-wif-synthetic-a")
        client = m.api.client
        assert isinstance(client, real_cls) and client.credentials is creds2 and creds2 is not creds
        assert client.api_key is None and client.auth_token is None and str(client.base_url).startswith("https://api.anthropic.com")
        assert m.api.api_key == w.WIF_SENTINEL_KEY            # the provider's own requirement, met by the sentinel the factory discarded
        # a key passed explicitly by a caller is discarded too: federation, never a static credential
        m2 = get_model("anthropic/claude-sonnet-5-wif-synthetic-b", api_key="-".join(("not", "a", "credential")))
        assert m2.api.client.api_key is None and m2.api.client.credentials is creds2
    finally:
        assert w.uninstall_inspect_wif() is True
    assert prov.AsyncAnthropic is real_cls and prov.AnthropicAPI.__init__ is real_init and w.inspect_wif_org() is None
    assert w.uninstall_inspect_wif() is False


def test_the_petri_leg_installs_federation_only_when_its_env_switch_names_an_org(monkeypatch):
    """The MODE is authoritative (round 1 on 7778694d0: with SISTEER_SECRETS_MODE=wif and the switch absent the leg returned None and inspect's
    provider read its own ANTHROPIC_API_KEY -- a direct petri_run.py or a resumed leg outside the drive shell ran keyed under a mode that said
    federation): under wif the switch is REQUIRED (absent: refused by name before any get_model); under op / env it must be ABSENT (present: a
    contradiction, refused); the key modes without a switch are the key path, byte for byte."""
    from sisteer.capeval.tiers import petri_runner as pr
    calls = []
    monkeypatch.setattr(w, "install_inspect_wif", lambda org, **kw: calls.append(org))
    assert pr.WIF_AUTH_ENV == w.WIF_AUTH_ENV == "SISTEER_ANTHROPIC_AUTH"
    # the key modes: no switch -> the key path, nothing installed (the default mode is op)
    monkeypatch.delenv(pr.WIF_AUTH_ENV, raising=False)
    for mode in (None, "op", "env"):
        if mode is None:
            monkeypatch.delenv(vault.SECRETS_MODE_ENV, raising=False)
        else:
            monkeypatch.setenv(vault.SECRETS_MODE_ENV, mode)
        assert pr.install_wif_if_requested() is None and calls == []
    # wif: the switch names the org (the vault's spelling, or lane 06's alias of the same pool)
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "wif")
    monkeypatch.setenv(pr.WIF_AUTH_ENV, "wif:lowprio")
    assert pr.install_wif_if_requested() == "lowprio" and calls == ["lowprio"]
    monkeypatch.setenv(pr.WIF_AUTH_ENV, "wif:loprio")                                  # lane 06's spelling is the same pool
    assert pr.install_wif_if_requested() == "lowprio" and calls == ["lowprio", "lowprio"]
    monkeypatch.setenv(pr.WIF_AUTH_ENV, "sk-ant-not-a-mode")
    with pytest.raises(pr.PetriRunnerError, match="expected wif:<org>"):
        pr.install_wif_if_requested()
    monkeypatch.setenv(pr.WIF_AUTH_ENV, "wif:")
    with pytest.raises(pr.PetriRunnerError):
        pr.install_wif_if_requested()
    # wif WITHOUT the switch: refused by name, nothing installed -- never a fall-through to the provider's ANTHROPIC_API_KEY read
    monkeypatch.delenv(pr.WIF_AUTH_ENV, raising=False)
    with pytest.raises(pr.PetriRunnerError, match="SISTEER_SECRETS_MODE=wif but SISTEER_ANTHROPIC_AUTH is unset") as ei:
        pr.install_wif_if_requested()
    assert "ANTHROPIC_API_KEY" in str(ei.value) and "F22_AUDITOR_ORG" in str(ei.value) and "refusing before the first get_model" in str(ei.value)
    assert calls == ["lowprio", "lowprio"]
    # a switch under a key mode: a contradiction, refused -- which credential did the caller mean?
    monkeypatch.setenv(pr.WIF_AUTH_ENV, "wif:highprio")
    for mode in ("op", "env"):
        monkeypatch.setenv(vault.SECRETS_MODE_ENV, mode)
        with pytest.raises(pr.PetriRunnerError, match=f"under SISTEER_SECRETS_MODE={mode}: federation is the wif mode's"):
            pr.install_wif_if_requested()
    assert calls == ["lowprio", "lowprio"]
    # an unknown mode refuses with the vault's words, whatever the switch
    monkeypatch.setenv(vault.SECRETS_MODE_ENV, "keychain")
    with pytest.raises(pr.PetriRunnerError, match="must be one of"):
        pr.install_wif_if_requested()
    # the resolver itself, over an explicit environment (no process state)
    assert w.leg_org({"SISTEER_SECRETS_MODE": "wif", "SISTEER_ANTHROPIC_AUTH": "wif:batch"}) == "batch"
    assert w.leg_org({}) is None and w.leg_org({"SISTEER_SECRETS_MODE": "env"}) is None
    with pytest.raises(ValueError, match="is unset"):
        w.leg_org({"SISTEER_SECRETS_MODE": "wif"})
    with pytest.raises(ValueError, match="contradiction"):
        w.leg_org({"SISTEER_ANTHROPIC_AUTH": "wif:lowprio"})
    src = (REPO / "src" / "sisteer" / "capeval" / "tiers" / "petri_runner.py").read_text()
    body = src.split("def _inspect_driver(", 1)[1]
    assert body.index("install_wif_if_requested()") < body.index("get_model(")      # before the first get_model: the provider builds its client in its constructor


def test_the_federated_provider_client_is_pinned_to_the_api_host_whatever_the_environment_says(tmp_path, monkeypatch):
    """The factory dropped the static credentials but forwarded the provider's `base_url` -- which inspect resolves from
    ANTHROPIC_BASE_URL (an environment may export a local gateway there) or INSPECT_EVAL_MODEL_BASE_URL -- so a federated get_model built a
    keyless client aimed at the gateway. Now the client's base URL is the vault's pin (sisteer.vault.ANTHROPIC_BASE_URL, exactly
    anthropic_client_wif's), the provider's ANTHROPIC_AUTH_TOKEN branch leaves no token and no oauth beta header (the provider reads its
    per-request betas off the client's default headers), an inherited ANTHROPIC_API_KEY never reaches the SDK, and nothing moves the pin -- the
    install's own `base_url=` naming another host is refused by name."""
    pytest.importorskip("inspect_ai")
    from inspect_ai.model import get_model
    from inspect_ai.model._providers import anthropic as prov
    _env_dir(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:1/mock-gateway")           # the inherited gateway
    monkeypatch.setenv("INSPECT_EVAL_MODEL_BASE_URL", "http://127.0.0.1:2/inspect-default")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "-".join(("not", "a", "token")))            # the provider's OAuth bearer branch
    monkeypatch.setenv("ANTHROPIC_API_KEY", "-".join(("placeholder", "not", "a", "key")))
    real_cls = prov.AsyncAnthropic
    assert getattr(real_cls, "__sisteer_wif_real__", None) is None, "a stale install leaked from another test"
    try:
        creds = w.install_inspect_wif("lowprio", token_provider=_stub_token)
        assert w.inspect_wif_base_url() == vault.ANTHROPIC_BASE_URL == "https://api.anthropic.com"
        m = get_model("anthropic/claude-sonnet-5-wif-synthetic-url-a")
        c = m.api.client
        assert isinstance(c, real_cls) and c.credentials is creds
        assert str(c.base_url).rstrip("/") == vault.ANTHROPIC_BASE_URL                      # the pin, not the gateway, not inspect's default
        assert c.api_key is None and c.auth_token is None
        assert not any(k.lower() == "anthropic-beta" for k in getattr(c, "_custom_headers", {}) or {})   # no oauth-2025-04-20 beta rides along
        assert m.api._client_default_betas() == []
        # a model-level base_url (inspect's --model-base-url) is discarded the same way: the federated client talks to the pinned host only
        m2 = get_model("anthropic/claude-sonnet-5-wif-synthetic-url-b", base_url="http://127.0.0.1:3/explicit")
        assert str(m2.api.client.base_url).rstrip("/") == vault.ANTHROPIC_BASE_URL and m2.api.client.credentials is creds
        # round 12 on fd06740ae: the install's own base_url= moves NOTHING either -- a host other than the pin is refused by name
        # before anything is installed (the previous install stands), the pin's own spelling installs; a test injects an
        # http_client over a mock transport through get_model's model args (tests/test_wif_federated_destination_and_sdk_retries.py)
        with pytest.raises(vault.BaseUrlRefused, match="base_url='http://127.0.0.1:4/service-double'"):
            w.install_inspect_wif("lowprio", token_provider=_stub_token, base_url="http://127.0.0.1:4/service-double")
        assert w.inspect_wif_base_url() == vault.ANTHROPIC_BASE_URL
        w.install_inspect_wif("lowprio", token_provider=_stub_token, base_url=vault.ANTHROPIC_BASE_URL + "/")
        m3 = get_model("anthropic/claude-sonnet-5-wif-synthetic-url-c")
        assert str(m3.api.client.base_url).rstrip("/") == vault.ANTHROPIC_BASE_URL and w.inspect_wif_base_url() == vault.ANTHROPIC_BASE_URL
        assert m3.api.client.api_key is None and m3.api.client.auth_token is None
    finally:
        assert w.uninstall_inspect_wif() is True
    assert prov.AsyncAnthropic is real_cls and w.inspect_wif_base_url() is None


class _RecordingEnviron(MutableMapping):
    """``os.environ`` with every key READ recorded (``get`` / ``[]`` / ``in``), everything forwarded to the real mapping."""

    def __init__(self, inner):
        self.inner, self.reads = inner, []

    def __getitem__(self, k):
        self.reads.append(k); return self.inner[k]

    def __setitem__(self, k, v):
        self.inner[k] = v

    def __delitem__(self, k):
        del self.inner[k]

    def __iter__(self):
        return iter(self.inner)

    def __len__(self):
        return len(self.inner)

    def __contains__(self, k):
        self.reads.append(k); return k in self.inner

    def copy(self):
        return dict(self.inner)


#: the variables a federated client must never be built from -- nor READ while being built (the machine rule of 2026-10-07)
CREDENTIAL_OR_URL_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")


def test_the_federated_provider_builds_its_client_without_reading_any_credential_or_url_variable_and_leaves_the_other_services_to_the_provider(
        tmp_path, monkeypatch):
    """The provider's own `_create_client` read ANTHROPIC_AUTH_TOKEN (its OAuth bearer branch) and
    ANTHROPIC_BASE_URL (model_base_url) BEFORE the federated factory discarded them -- the client came out right, but an inherited
    bearer had been read and handed through a constructor under a mode that forbids any ANTHROPIC_* credential read. Now the install
    replaces the provider's client factory for the plain API: the federated client is built directly from the vault's credentials and
    the pin, and none of the three variables is consulted at all during get_model (measured through a recording os.environ: before
    this, ANTHROPIC_AUTH_TOKEN and ANTHROPIC_BASE_URL were read); bedrock / vertex / azure names keep the provider's own factory."""
    pytest.importorskip("inspect_ai")
    import os
    from inspect_ai.model import get_model
    from inspect_ai.model._providers import anthropic as prov
    _env_dir(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "-".join(("not", "a", "token")))            # the provider's OAuth bearer branch
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:1/mock-gateway")           # the inherited gateway
    monkeypatch.setenv("ANTHROPIC_API_KEY", "-".join(("placeholder", "not", "a", "key")))
    real_cls, real_create = prov.AsyncAnthropic, prov.AnthropicAPI._create_client
    assert getattr(real_create, "__sisteer_wif_real__", None) is None, "a stale install leaked from another test"
    delegated = []

    def recording_create(self):                       # stands in for the provider's own factory: what the OTHER services must reach
        delegated.append(self.model_name)
        return SimpleNamespace(messages=None, _client=None)
    monkeypatch.setattr(prov.AnthropicAPI, "_create_client", recording_create)
    try:
        creds = w.install_inspect_wif("lowprio", token_provider=_stub_token)
        assert prov.AnthropicAPI._create_client.__sisteer_wif_real__ is recording_create
        real_environ = os.environ
        rec = _RecordingEnviron(real_environ)
        os.environ = rec
        try:
            m = get_model("anthropic/claude-sonnet-5-wif-synthetic-noread")
        finally:
            os.environ = real_environ
        c = m.api.client
        assert isinstance(c, real_cls) and c.credentials is creds and c.api_key is None and c.auth_token is None
        assert str(c.base_url).rstrip("/") == vault.ANTHROPIC_BASE_URL and isinstance(c.messages, w._LatchedMessages)
        read = sorted({k for k in rec.reads if k in CREDENTIAL_OR_URL_VARS})
        assert read == [], f"the provider read {read} while building a federated client"
        assert delegated == []                                                   # the plain API never reaches the provider's factory
        # the other services: the provider's own factory, untouched
        prov.AnthropicAPI._create_client(SimpleNamespace(model_name="bedrock/anthropic.claude-synthetic", model_args={}))
        assert delegated == ["bedrock/anthropic.claude-synthetic"]
    finally:
        assert w.uninstall_inspect_wif() is True
    assert prov.AnthropicAPI._create_client is recording_create and prov.AsyncAnthropic is real_cls


def test_the_drive_script_carries_the_wif_secrets_mode_without_a_key_and_the_old_modes_unchanged():
    s = (REPO / "scripts" / "sdfbase" / "full22" / "petri_drive_full22.sh").read_text()
    assert 'if [ "$SECRETS_MODE" = wif ]; then\n  SECRET_CMD=""' in s                                  # no secret-manager CLI under wif
    assert '[ -z "${ANTHROPIC_API_KEY+x}" ] ||' in s                                                   # a present key refuses: it would shadow federation
    assert 'export SISTEER_ANTHROPIC_AUTH="wif:$F22_AUDITOR_ORG"' in s                                 # the leg switch, the vault's org spelling
    assert "from sisteer.vault import PoolUnavailable, wif_ids_sha12" in s                              # the digest comes from the vault, the ONE ids reader
    assert 'AUDITOR_ATTEST="auditor_auth=wif:$F22_AUDITOR_ORG ids_sha12=$AUDITOR_IDS_SHA12"' in s       # the federated attestation
    assert 'AUDITOR_KEY_SHA12=$(printf %s "$ANTHROPIC_API_KEY" | sha256sum | cut -c1-12); AUDITOR_ATTEST="auditor_key_sha12=$AUDITOR_KEY_SHA12"' in s   # the key modes' attestation, unchanged in content
    assert "auditor_org=$F22_AUDITOR_ORG $AUDITOR_ATTEST secrets_mode=$SECRETS_MODE" in s              # op / env log `auditor_org=<o> auditor_key_sha12=<h> secrets_mode=<m>` byte for byte
    assert 'ANTHROPIC_API_KEY="$("$SECRET_CMD" "$F22_AUDITOR_ORG")"' in s                                # the vault path is intact
    assert "sha256sum < " not in s.split("SECRETS_MODE=", 1)[1].split("THE ATTESTATION", 1)[0]            # no second reader of the ids file beside the vault


def test_the_federated_stamp_matches_what_the_drive_attests(tmp_path, monkeypatch):
    """The digest the drive attests (`ids_sha12=`), the vault's pool stamp and the daemon's expected value are ONE number: sha256 of the org's ids
    file, first 12 hex -- so the three never disagree by construction."""
    d = _env_dir(tmp_path, monkeypatch)
    digest = hashlib.sha256(Path(vault.wif_env_path("lowprio")).read_bytes()).hexdigest()[:12]
    assert vault.wif_ids_sha12("lowprio") == digest == vault.wif_pool_stamp("lowprio")["ids_sha12"]
    assert vault.wif_org_of_ids_sha12(digest) == "lowprio" and vault.wif_org_of_ids_sha12("0" * 12) is None
