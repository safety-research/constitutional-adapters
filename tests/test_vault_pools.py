"""Pool resolution through ``sisteer.vault``: a request for one Anthropic pool
is satisfied ONLY by that pool's own environment variable or its own vault
item; a pool with neither RAISES :class:`sisteer.vault.PoolUnavailable`; and
every client the vault builds carries a POOL STAMP a run can record.

WHY. Before the vault, ``eval/api_keys.make_client("lowprio")`` read a key
file, and when the file was absent (it was deleted on purpose on 2026-09-04
for the vault-only rule) it built a bare SDK client. The SDK then read
``ANTHROPIC_API_KEY`` from the shell — the HIGH PRIO key — so every caller
that asked for the shared Low Prio pool got the priority pool and was told
nothing. The routing rule (on 429s throttle down, never hop pools) was never
in force on those paths: the helper hopped by construction, before anyone
could throttle. These tests pin the three properties that make that
impossible to reintroduce: per-pool variables, a typed refusal instead of a
fallback, and a stamp so the pool a run used is RECORDED rather than inferred
from which shell launched it.

CPU-only, no network, no vault: the secret-manager CLI is a stub that answers
from a dict, and the placeholder strings are not keys.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import types

import pytest

from sisteer import vault

LOW, HIGH, BATCH = "ORG-KEY-LOWPRIO", "ORG-KEY-HIGHPRIO", "ORG-KEY-BATCH"
KEYS = {"lowprio": LOW, "highprio": HIGH, "batch": BATCH}


@pytest.fixture(autouse=True)
def _no_pool_env(monkeypatch):
    for env, _item in vault.ANTHROPIC_ORGS.values():
        monkeypatch.delenv(env, raising=False)


def _vault_answers(monkeypatch, answers: dict):
    """The secret-manager CLI for ``<item>`` -> ``answers[item]``; an item not
    in the dict is the CLI's non-zero exit (item missing / session expired)."""
    def fake_check_output(cmd, *a, **k):
        item = cmd[1]
        if item not in answers:
            raise subprocess.CalledProcessError(
                1, cmd, stderr="[ERROR] item not found in the vault")
        return answers[item]

    monkeypatch.setattr(vault.subprocess, "check_output", fake_check_output)
    monkeypatch.setenv("SISTEER_SECRET_CMD", "fake-secret-cmd")       # op mode runs the CLI the site setting names


# ------------------------------------------------------- the hop is closed


def test_lowprio_with_only_the_highprio_variable_set_raises_never_hops(monkeypatch):
    """THE REGRESSION. ``ANTHROPIC_API_KEY`` holds the High Prio key and the
    vault has nothing for ``lowprio``: the request must RAISE — a typed,
    named refusal — not return the High Prio key."""
    monkeypatch.setenv(vault.ANTHROPIC_ORGS["highprio"][0], HIGH)
    _vault_answers(monkeypatch, {})
    with pytest.raises(vault.PoolUnavailable) as ei:
        vault.anthropic_api_key("lowprio")
    assert ei.value.pool == "lowprio"
    assert isinstance(ei.value, vault.VaultSecretError)   # old handlers still catch it
    msg = str(ei.value)
    assert vault.ANTHROPIC_ORGS["lowprio"][0] in msg and "'lowprio'" in msg
    assert HIGH not in msg                                # never carries a value
    with pytest.raises(vault.PoolUnavailable):
        vault.anthropic_client("lowprio")
    with pytest.raises(vault.PoolUnavailable):
        vault.anthropic_async_client("lowprio")


def test_vault_answering_empty_for_lowprio_raises(monkeypatch):
    monkeypatch.setenv(vault.ANTHROPIC_ORGS["highprio"][0], HIGH)
    _vault_answers(monkeypatch, {"lowprio": "\n"})
    with pytest.raises(vault.PoolUnavailable) as ei:
        vault.anthropic_api_key("lowprio")
    assert ei.value.pool == "lowprio" and "empty" in str(ei.value)


@pytest.mark.parametrize("pool", sorted(vault.ANTHROPIC_ORGS))
def test_each_pool_reads_only_its_own_variable(monkeypatch, pool):
    """All three variables set: each pool returns ITS key. Then unset only
    this pool's variable while the other two stay set: this pool raises and
    neither of the others' keys is borrowed or named."""
    for p, (env, _item) in vault.ANTHROPIC_ORGS.items():
        monkeypatch.setenv(env, KEYS[p])
    _vault_answers(monkeypatch, {})
    assert vault.anthropic_api_key(pool) == KEYS[pool]
    monkeypatch.delenv(vault.ANTHROPIC_ORGS[pool][0])
    with pytest.raises(vault.PoolUnavailable) as ei:
        vault.anthropic_api_key(pool)
    assert ei.value.pool == pool
    for other in KEYS.values():
        assert other not in str(ei.value)


def test_a_pool_falls_to_its_own_vault_item_never_anothers(monkeypatch):
    monkeypatch.setenv(vault.ANTHROPIC_ORGS["highprio"][0], HIGH)
    _vault_answers(monkeypatch, {"lowprio": LOW + "\n", "highprio": "WRONG"})
    assert vault.anthropic_api_key("lowprio") == LOW


def test_unknown_pool_is_a_value_error_not_a_guess():
    with pytest.raises(ValueError, match="org must be one of"):
        vault.anthropic_api_key("midprio")


# ------------------------------------------------------------- the stamp


def test_pool_stamp_shape_and_never_the_key():
    st = vault.pool_stamp("lowprio", LOW)
    assert st == {"pool": "lowprio",
                  "key_sha256_12": hashlib.sha256(LOW.encode()).hexdigest()[:12]}
    assert len(st["key_sha256_12"]) == 12
    int(st["key_sha256_12"], 16)                          # hex
    assert LOW not in json.dumps(st)
    assert vault.pool_stamp("lowprio", LOW) != vault.pool_stamp("highprio", HIGH)
    with pytest.raises(ValueError):
        vault.pool_stamp("lowprio", "")


@pytest.mark.parametrize("build", [vault.anthropic_client, vault.anthropic_async_client])
def test_vault_built_clients_carry_the_stamp_and_the_pins(monkeypatch, build):
    monkeypatch.setenv(vault.ANTHROPIC_ORGS["lowprio"][0], LOW)
    monkeypatch.setenv(vault.ANTHROPIC_ORGS["highprio"][0], HIGH)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8787/t/deadbeef")
    c = build("lowprio", max_retries=0, timeout=7.5)
    assert vault.client_pool_stamp(c) == vault.pool_stamp("lowprio", LOW)
    assert c.api_key == LOW
    assert str(c.base_url).rstrip("/") == vault.ANTHROPIC_BASE_URL   # pinned
    assert c.timeout == 7.5 and c.max_retries == 0
    h = build("highprio")
    assert vault.client_pool_stamp(h) == vault.pool_stamp("highprio", HIGH)


def test_client_pool_stamp_refuses_a_client_the_vault_did_not_build():
    """A run record must say which pool it used or fail loudly — never
    write an unknown. Half a stamp is no stamp."""
    for c in (object(), types.SimpleNamespace(messages=None),
              types.SimpleNamespace(sisteer_pool_stamp={"pool": "lowprio"}),
              types.SimpleNamespace(sisteer_pool_stamp="lowprio")):
        with pytest.raises(ValueError, match="no pool stamp"):
            vault.client_pool_stamp(c)


def test_client_pool_stamp_returns_a_copy(monkeypatch):
    monkeypatch.setenv(vault.ANTHROPIC_ORGS["lowprio"][0], LOW)
    c = vault.anthropic_client("lowprio")
    st = vault.client_pool_stamp(c)
    st["pool"] = "tampered"
    assert vault.client_pool_stamp(c)["pool"] == "lowprio"


def test_stamp_reads_through_the_throttled_wrapper(monkeypatch):
    """``LiveStrongREJECTGrader`` wraps its client in ``ThrottledClient``;
    the stamp must be readable through the wrapper's attribute forwarding."""
    from sisteer.throttle import AdaptiveGate, ThrottledClient

    monkeypatch.setenv(vault.ANTHROPIC_ORGS["lowprio"][0], LOW)
    wrapped = ThrottledClient(vault.anthropic_client("lowprio"), AdaptiveGate(4))
    assert vault.client_pool_stamp(wrapped) == vault.pool_stamp("lowprio", LOW)
