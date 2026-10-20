"""op mode's secret-manager CLI is the site setting ``SISTEER_SECRET_CMD``, with no default command: the client runs exactly
the command the setting names, with the vault item as its one argument, and refuses by name while the setting is unset
(nothing runs). The pool -> ids-file stems are the vault's (``sisteer.vault.WIF_ORG_FILES``), carried here as a twin because
this member imports no sisteer, and pinned equal to the vault's module beside it. Fakes ride the ``runner=`` seam."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from corpus_generator import client as C

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def op_mode(monkeypatch):
    monkeypatch.delenv("SISTEER_SECRETS_MODE", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv(C.SECRET_CMD_ENV, raising=False)


def test_op_mode_refuses_by_name_while_the_setting_is_unset_and_runs_nothing(op_mode, monkeypatch):
    def runner(*a, **k):  # pragma: no cover - the assertion IS the test
        raise AssertionError(f"a command ran: {a}")

    for kind in ("batch", "smoke", "live"):
        with pytest.raises(C.ClientGuardError, match="SISTEER_SECRET_CMD is not set") as ei:
            C.read_key(kind, runner=runner)
        assert "config.env" in str(ei.value)
    monkeypatch.setenv(C.SECRET_CMD_ENV, "  ")                               # blank is unset
    with pytest.raises(C.ClientGuardError, match="SISTEER_SECRET_CMD is not set"):
        C.vault_secret("batch", runner=runner)


def test_op_mode_runs_the_named_command_with_the_item_as_its_one_argument(op_mode, monkeypatch, tmp_path):
    cli = tmp_path / "fake-secret-cmd"
    cli.write_text("#!/bin/sh\nexit 0\n")
    cli.chmod(0o755)
    monkeypatch.setenv(C.SECRET_CMD_ENV, str(cli))
    answers = {C.VAULT_ITEMS["batch"]: "FAKE-BATCH-KEY\n", C.HIGHPRIO_VAULT_ITEM: "FAKE-HIGHPRIO-KEY\n"}
    calls: list = []

    def runner(argv, capture_output, text):
        calls.append(list(argv))
        return NS(returncode=0, stdout=answers[argv[1]], stderr="")

    assert C.read_key("batch", runner=runner) == "FAKE-BATCH-KEY"
    assert calls == [[str(cli), C.VAULT_ITEMS["batch"]], [str(cli), C.HIGHPRIO_VAULT_ITEM]]
    # a name that is no executable refuses by the setting's name, never by the value it holds
    monkeypatch.setenv(C.SECRET_CMD_ENV, str(tmp_path / "absent-cli"))
    with pytest.raises(C.ClientGuardError, match="SISTEER_SECRET_CMD names is not an executable") as ei:
        C.read_key("batch", runner=runner)
    assert str(tmp_path) not in str(ei.value)


def test_the_ids_file_names_follow_the_vaults_stems():
    path = REPO / "src" / "sisteer" / "vault.py"
    if not path.is_file():
        pytest.skip("the vault's module is not beside this member (installed outside the repository)")
    sys.path.insert(0, str(REPO / "src"))
    try:
        vault = importlib.import_module("sisteer.vault")
    finally:
        sys.path.remove(str(REPO / "src"))
    assert C.WIF_POOL_STEMS == vault.WIF_ORG_FILES
    assert C.WIF_FILES == {kind: f"wif-{vault.WIF_ORG_FILES[pool]}.env" for kind, pool in C.WIF_POOLS.items()}
    assert C.HIGHPRIO_WIF_FILE == f"wif-{vault.WIF_ORG_FILES['highprio']}.env"
    assert C.SECRET_CMD_ENV == vault.SECRET_CMD_ENV
