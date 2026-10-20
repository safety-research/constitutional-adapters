"""sisteer.vault's TWO secrets modes — the Python half.

env mode (`SISTEER_SECRETS_MODE=env`): `anthropic_api_key(org)` / `anthropic_client(org)` resolve EVERY org role
(lowprio / highprio / batch) to the single `ANTHROPIC_API_KEY`, required by name (a typed `PoolUnavailable` when unset),
never a file, never the secret-manager CLI (absent from PATH here); `openrouter_api_key()` reads `OPENROUTER_API_KEY`; the
pool stamp records `pool = "env"` + the key's sha12. op mode is byte-for-byte unchanged: a missing pool still refuses by
name and never hops to another pool's key (tests/test_vault_pools.py pins that side)."""
from __future__ import annotations

import hashlib
import os

import pytest

from sisteer import vault


@pytest.fixture
def env_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "env")
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))          # the secret-manager CLI is ABSENT
    for v in ("ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY_LOWPRIO", "ANTHROPIC_API_KEY_BATCH", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(v, raising=False)


@pytest.mark.parametrize("org", ["lowprio", "highprio", "batch"])
def test_every_org_role_resolves_to_the_one_key(env_mode, monkeypatch, org):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-env-key")
    assert vault.anthropic_api_key(org) == "synthetic-env-key"


def test_env_mode_ignores_the_op_mode_pool_variables(env_mode, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY_LOWPRIO", "the-op-mode-pool-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "the-env-key")
    assert vault.anthropic_api_key("lowprio") == "the-env-key"          # one key = every role; *_LOWPRIO is op-mode only


def test_env_mode_refuses_by_name_when_unset(env_mode):
    with pytest.raises(vault.PoolUnavailable) as ei:
        vault.anthropic_api_key("lowprio")
    assert "ANTHROPIC_API_KEY must be set" in str(ei.value) and ei.value.pool == "lowprio"
    assert "never the secret-manager CLI" in str(ei.value)          # says what it never does


def test_env_mode_never_calls_the_secret_cli(env_mode, monkeypatch):
    calls = []
    monkeypatch.setattr(vault.subprocess, "check_output", lambda *a, **k: calls.append(a) or "leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    vault.anthropic_api_key("batch"); vault.anthropic_api_key("highprio")
    assert calls == []


def test_openrouter_env_mode(env_mode, monkeypatch):
    with pytest.raises(vault.VaultSecretError):
        vault.openrouter_api_key()
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
    assert vault.openrouter_api_key() == "or-key"


def test_client_stamp_records_the_env_contract(env_mode, monkeypatch):
    pytest.importorskip("anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-env-key")
    c = vault.anthropic_client("lowprio")
    st = vault.client_pool_stamp(c)
    assert st == {"pool": "env", "key_sha256_12": hashlib.sha256(b"synthetic-env-key").hexdigest()[:12]}
    assert str(c.base_url).rstrip("/") == vault.ANTHROPIC_BASE_URL


def test_unknown_mode_is_a_typed_refusal(monkeypatch):
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "vault")
    with pytest.raises(vault.VaultSecretError):
        vault.anthropic_api_key("lowprio")


def test_op_mode_is_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv("SISTEER_SECRETS_MODE", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "highprio-only")
    monkeypatch.delenv("ANTHROPIC_API_KEY_LOWPRIO", raising=False)
    with pytest.raises(vault.PoolUnavailable):          # lowprio never hops to ANTHROPIC_API_KEY under op mode
        vault.anthropic_api_key("lowprio")
    assert vault.anthropic_api_key("highprio") == "highprio-only"
