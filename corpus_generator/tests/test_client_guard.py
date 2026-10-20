"""Env-hijack + highprio guards on the single client constructor.

Fakes reach ``read_key`` through its ``runner=`` seam — a stand-in for the
secret-manager CLI subprocess — never through a key FILE: the client has no
key-file path at all (
spec FD 17 "two secrets modes, never a third"; see ../README.md)."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from corpus_generator import client as C
from corpus_generator.client import API_BASE_URL, ClientGuardError, build_client, read_key

FAKE = {"batch": "FAKE-BATCH-KEY-111\n", "lowprio": "FAKE-LOWPRIO-KEY-222\n", "highprio": "FAKE-HIGHPRIO-KEY-000\n"}


@pytest.fixture
def vault(monkeypatch, tmp_path):
    """A fake secret-manager CLI: on PATH (``shutil.which`` patched) and answering from ``answers`` (``subprocess.run`` patched,
    which the seam's call-time default resolves to). HOME is an empty tmp dir, so nothing real is reachable; the mode
    switch and ``ANTHROPIC_API_KEY`` are scrubbed unless a test sets them."""
    monkeypatch.delenv("SISTEER_SECRETS_MODE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setenv(C.SECRET_CMD_ENV, "fake-secret-cmd")                                # the site's CLI, by setting
    monkeypatch.setattr(C.shutil, "which", lambda name: "/usr/local/bin/fake-secret-cmd" if name == "fake-secret-cmd" else None)
    answers = dict(FAKE)
    calls: list[list[str]] = []

    def runner(argv, capture_output, text):
        assert argv[0] == "fake-secret-cmd" and capture_output and text
        calls.append(list(argv))
        return NS(returncode=0, stdout=answers[argv[1]], stderr="")

    monkeypatch.setattr(C.subprocess, "run", runner)
    return NS(answers=answers, calls=calls, home=tmp_path)


def test_env_hijack_ignored(vault, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "FAKE-SENTINEL-ENV-KEY")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://localhost:9999/session-gateway")
    client = build_client("batch")
    assert client.api_key == "FAKE-BATCH-KEY-111"
    assert str(client.base_url).rstrip("/") == API_BASE_URL
    smoke = build_client("smoke")
    assert smoke.api_key == "FAKE-LOWPRIO-KEY-222"
    # each kind's item, then the highprio item for the refusal check — from the vault, both times
    assert [c[1] for c in vault.calls] == ["batch", "highprio", "lowprio", "highprio"]


def test_highprio_contents_refused(vault):
    vault.answers["batch"] = FAKE["highprio"]
    with pytest.raises(ClientGuardError, match="HIGHPRIO"):
        build_client("batch")


def test_highprio_refusal_applies_to_smoke_too(vault):
    vault.answers["lowprio"] = FAKE["highprio"].strip()
    with pytest.raises(ClientGuardError, match="HIGHPRIO"):
        build_client("smoke")


def test_op_mode_has_no_key_file_path(vault, monkeypatch):
    """Op mode is the vault and nothing else. Structurally — no
    ``key_dir`` parameter on any constructor, no key-file name table in the module — and behaviourally: with every file read
    made to fail, ``read_key`` still answers from the vault, so no file is consulted before it."""
    for fn in (read_key, build_client, C.build_async_client, C.vault_secret):
        assert "key_dir" not in inspect.signature(fn).parameters, fn.__name__
    assert not [n for n in vars(C) if "KEY_FILE" in n.upper()], "a key-file name table is a third secrets path"

    def no_file_reads(self, *a, **k):  # pragma: no cover - the assertion IS the test
        raise AssertionError(f"read_key opened a file: {self}")

    monkeypatch.setattr(Path, "read_text", no_file_reads)
    monkeypatch.setattr(Path, "open", no_file_reads)
    assert read_key("batch") == "FAKE-BATCH-KEY-111"
    assert [c[1] for c in vault.calls] == ["batch", "highprio"]


def test_vault_failure_and_absence_are_loud(vault, monkeypatch):
    monkeypatch.setattr(C.subprocess, "run", lambda argv, capture_output, text: NS(returncode=1, stdout="", stderr="no session"))
    with pytest.raises(ClientGuardError, match="failed"):
        read_key("batch")
    monkeypatch.setattr(C.shutil, "which", lambda name: None)
    with pytest.raises(ClientGuardError, match="not on PATH"):
        read_key("batch")


def test_empty_vault_value_errors(vault):
    vault.answers["batch"] = "\n"
    with pytest.raises(ClientGuardError, match="empty"):
        read_key("batch")


def test_keys_are_stripped(vault):
    assert read_key("batch") == "FAKE-BATCH-KEY-111"
    assert read_key("live") == "FAKE-LOWPRIO-KEY-222"


def test_live_http_client_uses_a_plain_ssl_context_not_truststore():
    """2026-09-03: the SDK's default httpx2 transport verifies through
    truststore, whose per-handshake context configuration races across
    httpcore2's worker threads under a 200-wide burst (glibc heap corruption,
    three launch crashes). The live runner's transport must be a plain stdlib
    ssl context with limits sized for the wave."""
    import ssl

    from corpus_generator.client import live_http_client, live_ssl_context

    ctx = live_ssl_context()
    assert type(ctx) is ssl.SSLContext and type(ctx).__module__ == "ssl"
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    client = live_http_client(timeout=30.0)
    pool = client._transport._pool
    assert type(pool._ssl_context).__module__ == "ssl"
    assert pool._max_connections >= 200 and pool._max_keepalive_connections >= 200
