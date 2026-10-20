"""op mode's secret-manager CLI is a site setting, ``SISTEER_SECRET_CMD``, with no default command.

The Python vault, the shell library and the grading daemon's key-hash pipeline each run exactly the command the setting
names, with the vault item as its one argument, and each refuses by name while the setting is unset (a message names the
setting, never the command it holds). A detached unit receives the CLI's absolute path, a value and never a key, under op
and under wif (whose non-Anthropic secrets resolve as under op), and its generated key line runs that command. The CLI
here is a fake that records its arguments and prints a placeholder; nothing reaches a real vault."""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))
from sisteer import vault  # noqa: E402

LIB = REPO / "reproduce" / "lib" / "secrets.sh"
VALUE = "synthetic-item-value"


def _cli(tmp_path: Path, value: str = VALUE) -> Path:
    """A fake secret-manager CLI under ``tmp_path/bin``: appends its arguments to ``tmp_path/cli.log``, prints ``value``."""
    log = tmp_path / "cli.log"
    f = tmp_path / "bin" / "fake-secret-cmd"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(f'#!/bin/sh\necho "$*" >> {log}\nprintf "%s" {value}\n')
    f.chmod(0o755)
    return f


def _calls(tmp_path: Path) -> list:
    log = tmp_path / "cli.log"
    return log.read_text().splitlines() if log.exists() else []


def _base_env(tmp_path: Path) -> dict:
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SISTEER_", "ANTHROPIC_", "OPENROUTER_", "HF_"))}
    e["HOME"] = str(tmp_path / "home")
    return e


def _bash(script: str, env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=60)


# --------------------------------------------------------------------------------------------------------- the Python vault
def test_the_vault_refuses_by_name_while_the_setting_is_unset_and_runs_nothing(monkeypatch):
    monkeypatch.delenv("SISTEER_SECRET_CMD", raising=False)
    monkeypatch.delenv("SISTEER_SECRETS_MODE", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(vault.subprocess, "check_output", lambda *a, **k: pytest.fail(f"a command ran: {a}"))
    with pytest.raises(vault.VaultSecretError, match="SISTEER_SECRET_CMD is not set") as ei:
        vault.openrouter_api_key()
    assert "OpenRouter API Key" in str(ei.value) and "config.env" in str(ei.value)
    with pytest.raises(vault.PoolUnavailable, match="SISTEER_SECRET_CMD is not set"):
        vault.anthropic_api_key("lowprio")
    monkeypatch.setenv("SISTEER_SECRET_CMD", "   ")                       # blank is unset
    with pytest.raises(vault.VaultSecretError, match="SISTEER_SECRET_CMD is not set"):
        vault.secret_cmd()


def test_the_vault_runs_the_named_command_with_the_item_as_its_one_argument(tmp_path, monkeypatch):
    cli = _cli(tmp_path)
    monkeypatch.delenv("SISTEER_SECRETS_MODE", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("SISTEER_SECRET_CMD", str(cli))
    assert vault.secret_cmd() == str(cli) and vault.secret_cmd({"SISTEER_SECRET_CMD": "other"}) == "other"
    assert vault.openrouter_api_key() == VALUE and _calls(tmp_path) == ["OpenRouter API Key"]
    # a bare name resolves on PATH
    monkeypatch.setenv("PATH", f"{cli.parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("SISTEER_SECRET_CMD", cli.name)
    assert vault.resolve_secret(env="OPENROUTER_API_KEY", item="item two") == VALUE and _calls(tmp_path)[-1] == "item two"
    # a name that is no executable refuses by the setting's name, never by the value it holds
    monkeypatch.setenv("SISTEER_SECRET_CMD", str(tmp_path / "absent-cli"))
    with pytest.raises(vault.VaultSecretError, match="the secret-manager CLI that SISTEER_SECRET_CMD names is not an executable") as ei:
        vault.resolve_secret(env="OPENROUTER_API_KEY", item="item three")
    assert str(tmp_path) not in str(ei.value)


def test_the_daemons_key_hash_pipeline_runs_the_command_its_drive_environment_names(tmp_path):
    """``secret_sha12`` hashes the item in a shell pipeline under ANOTHER process's environment (the one the grading daemon hands
    a Petri drive): the command comes from that environment's setting, and an environment without it is a reason, not a guess."""
    cli = _cli(tmp_path, value="fake-vault-item")
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home")}
    sha12, why = vault.secret_sha12("lowprio", env=env)
    assert sha12 is None and "SISTEER_SECRET_CMD is not set" in why and _calls(tmp_path) == []
    sha12, why = vault.secret_sha12("lowprio", env=dict(env, SISTEER_SECRET_CMD=str(cli)))
    assert why is None and sha12 == hashlib.sha256(b"fake-vault-item").hexdigest()[:12] and _calls(tmp_path) == ["lowprio"]
    sha12, why = vault.secret_sha12("highprio", env=dict(env, SISTEER_SECRET_CMD=cli.name, PATH=f"{cli.parent}{os.pathsep}{env['PATH']}"))
    assert why is None and sha12 == hashlib.sha256(b"fake-vault-item").hexdigest()[:12] and _calls(tmp_path)[-1] == "highprio"
    sha12, why = vault.secret_sha12("lowprio", env=dict(env, SISTEER_SECRET_CMD=str(tmp_path / "absent-cli")))
    assert sha12 is None and why == "SISTEER_SECRET_CMD does not name an executable (not found, or not on PATH)" and _calls(tmp_path)[-1] == "highprio"
    sha12, why = vault.secret_sha12("lowprio", env=dict(env, SISTEER_SECRET_CMD=str(_cli(tmp_path / "fails", value=""))))
    assert sha12 is None and why.startswith("SISTEER_SECRET_CMD lowprio failed") and str(tmp_path) not in why     # an empty value


# --------------------------------------------------------------------------------------------------------- the shell library
def test_the_shell_library_refuses_by_name_while_the_setting_is_unset(tmp_path):
    e = dict(_base_env(tmp_path), SISTEER_SECRETS_MODE="op")
    r = _bash(f"source {LIB}; secret_anthropic lowprio", e)
    assert r.returncode == 1 and r.stdout == "" and "SISTEER_SECRET_CMD is not set" in r.stderr
    r = _bash(f"source {LIB}; secrets_unit_flags", e)                         # nothing to carry: the unit refuses by name
    assert r.returncode == 0 and r.stdout == ""
    r = _bash(f"source {LIB}; secrets_op_bin", dict(e, SISTEER_SECRET_CMD=str(tmp_path / "absent-cli")))
    assert r.returncode == 1 and "SISTEER_SECRET_CMD does not name an executable" in r.stderr and str(tmp_path) not in r.stderr


def test_the_shell_library_runs_the_named_command_and_carries_its_path_into_a_unit(tmp_path):
    cli = _cli(tmp_path)
    e = dict(_base_env(tmp_path), SISTEER_SECRETS_MODE="op", SISTEER_SECRET_CMD=str(cli))
    r = _bash(f"source {LIB}; secret_anthropic highprio", e)
    assert r.returncode == 0 and r.stdout == VALUE and _calls(tmp_path) == ["highprio"]
    assert _bash(f"source {LIB}; secrets_unit_flags", e).stdout == f"--setenv=SISTEER_SECRET_CMD={cli}"
    # a bare name is carried as the absolute path it resolves to (a unit's PATH may lack the CLI's directory)
    bare = dict(e, SISTEER_SECRET_CMD=cli.name, PATH=f"{cli.parent}:{e['PATH']}")
    assert _bash(f"source {LIB}; secrets_unit_flags", bare).stdout == f"--setenv=SISTEER_SECRET_CMD={cli}"
    # under wif the path rides beside the mode: federation covers the Claude API only
    wif = dict(e, SISTEER_SECRETS_MODE="wif")
    assert _bash(f"source {LIB}; secrets_unit_flags", wif).stdout == f"--setenv=SISTEER_SECRETS_MODE=wif --setenv=SISTEER_SECRET_CMD={cli}"
    # the generated key line runs the CLI inside the unit, from the unit's own environment
    line = _bash(f"source {LIB}; secrets_unit_key_line lowprio", e).stdout
    run = _bash(line + '; printf "%s" "$ANTHROPIC_API_KEY_LOWPRIO"', dict(_base_env(tmp_path), SISTEER_SECRET_CMD=str(cli)))
    assert run.returncode == 0 and run.stdout == VALUE and _calls(tmp_path)[-1] == "lowprio"
    # ... and names the setting when that environment lacks it (the key stays empty, the judge's vault then refuses by name)
    bad = _bash(line + '; printf "[%s]" "${ANTHROPIC_API_KEY_LOWPRIO-unset}"', _base_env(tmp_path))
    assert bad.stdout == "[]" and "SISTEER_SECRET_CMD must name the secret-manager CLI" in bad.stderr
