"""The env-only secrets mode (front-door contract):
under ``SISTEER_SECRETS_MODE=env`` the ONE documented ``ANTHROPIC_API_KEY``
serves EVERY client kind (batch / smoke / live), the secret-manager CLI is
never called, no file is read, and a missing variable refuses by name; under
``op`` (the default) the vault path and the highprio refusal stand exactly as
before. Neither mode has a key FILE (paper-companion-repo local patch of
see ../README.md): fakes ride the ``runner=`` seam."""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from corpus_generator import client as C
from corpus_generator.client import (
    API_BASE_URL,
    ClientGuardError,
    build_async_client,
    build_client,
    read_key,
    secrets_mode,
)


@pytest.fixture
def no_secret_cli(monkeypatch, tmp_path):
    """The secret-manager CLI off PATH and a runner that fails the test if called (the seam's default resolves to
    ``subprocess.run`` at CALL time, so the patch is what any constructor would reach)."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))
    (tmp_path / "empty-bin").mkdir()

    def boom(*a, **k):  # pragma: no cover - the assertion IS the test
        raise AssertionError(f"the secret-manager CLI must not be called in env mode: {a}")

    monkeypatch.setattr(C.subprocess, "run", boom)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    return tmp_path


def test_env_mode_one_key_serves_every_kind(monkeypatch, no_secret_cli):
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "env")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "FAKE-ENV-MODE-KEY-999")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:9999/some-gateway")
    assert secrets_mode() == "env"
    for kind in ("batch", "smoke", "live"):
        assert read_key(kind) == "FAKE-ENV-MODE-KEY-999"
    client = build_client("batch")
    assert client.api_key == "FAKE-ENV-MODE-KEY-999"
    assert str(client.base_url).rstrip("/") == API_BASE_URL, "base URL stays pinned in env mode"
    aclient = build_async_client("live")
    assert aclient.api_key == "FAKE-ENV-MODE-KEY-999"


def test_env_mode_reads_no_file_and_no_vault(monkeypatch, no_secret_cli):
    """No file is ever read in env mode — and there is no file to point the client at: ``read_key`` takes no ``key_dir``
    (FD 17). With every file read made to fail, the env key still answers."""
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "env")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "FAKE-ENV-MODE-KEY-999")
    assert "key_dir" not in inspect.signature(read_key).parameters

    def no_file_reads(self, *a, **k):  # pragma: no cover - the assertion IS the test
        raise AssertionError(f"read_key opened a file: {self}")

    monkeypatch.setattr(Path, "read_text", no_file_reads)
    monkeypatch.setattr(Path, "open", no_file_reads)
    assert read_key("batch") == "FAKE-ENV-MODE-KEY-999"


def test_env_mode_missing_variable_refuses_by_name(monkeypatch, no_secret_cli):
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "env")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ClientGuardError, match="ANTHROPIC_API_KEY"):
        read_key("batch")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    with pytest.raises(ClientGuardError, match="ANTHROPIC_API_KEY"):
        build_client("smoke")


def test_unknown_mode_refuses(monkeypatch):
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "vault")
    with pytest.raises(ClientGuardError, match="SISTEER_SECRETS_MODE"):
        read_key("batch")


def test_op_mode_is_the_default_and_unchanged(monkeypatch, tmp_path):
    """No switch set -> op mode: the vault answers, the env key is ignored, the highprio refusal stands (the pre-existing
    guarantees, restated here so a change to the default is a test failure, not a silent reroute)."""
    monkeypatch.delenv("SISTEER_SECRETS_MODE", raising=False)
    assert secrets_mode() == "op"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "FAKE-ENV-KEY-MUST-BE-IGNORED-IN-OP-MODE")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv("SISTEER_SECRET_CMD", "fake-secret-cmd")
    monkeypatch.setattr(C.shutil, "which", lambda name: "/usr/local/bin/fake-secret-cmd")
    answers = {"batch": "FAKE-BATCH-KEY-111\n", "highprio": "FAKE-HIGHPRIO-KEY-000\n"}
    monkeypatch.setattr(C.subprocess, "run", lambda argv, capture_output, text: NS(returncode=0, stdout=answers[argv[1]], stderr=""))
    assert read_key("batch") == "FAKE-BATCH-KEY-111"
    answers["batch"] = answers["highprio"]
    with pytest.raises(ClientGuardError, match="HIGHPRIO"):
        read_key("batch")


def test_op_mode_vault_items_follow_the_front_door_config_names():
    """The vault item names are the SISTEER_OP_ITEM_ANTHROPIC_* keys' values
    (defaults batch / lowprio / highprio) — a site naming its items differently
    configures, never edits."""
    assert C.VAULT_ITEMS["batch"] == os.environ.get("SISTEER_OP_ITEM_ANTHROPIC_BATCH", "batch")
    assert C.VAULT_ITEMS["smoke"] == C.VAULT_ITEMS["live"] == os.environ.get("SISTEER_OP_ITEM_ANTHROPIC_LOWPRIO", "lowprio")
    assert C.HIGHPRIO_VAULT_ITEM == os.environ.get("SISTEER_OP_ITEM_ANTHROPIC_HIGHPRIO", "highprio")
