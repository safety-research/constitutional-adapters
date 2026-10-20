"""``sisteer.wandb_entity`` — the ONE W&B routing validator: refuse on an
unset/empty WANDB_ENTITY, refuse a key whose default entity is not that
entity, retry API errors a bounded number of times and then refuse
(never degrade); ``wandb_init_checked`` passes the validated entity
explicitly and applies the same bound to ``init``.
"""

from __future__ import annotations

import sys
import types

import pytest

import sisteer.wandb_entity as we


def _fake(monkeypatch, default_entity, *, api_fail=0, init_fail=0):
    fake = types.ModuleType("wandb")
    calls = {"api": 0, "init": [], "sleeps": []}

    class _Api:
        def __init__(self, timeout=None):
            calls["api"] += 1
            calls.setdefault("timeouts", []).append(timeout)
            if calls["api"] <= api_fail:
                raise ConnectionError(f"down #{calls['api']}")

        @property
        def default_entity(self):
            return default_entity

    def init(**kw):
        calls["init"].append(kw)
        if len(calls["init"]) <= init_fail:
            raise OSError("init down")
        return types.SimpleNamespace(id="run", log=lambda *a, **k: None)

    fake.Api = _Api
    fake.init = init
    monkeypatch.setitem(sys.modules, "wandb", fake)
    monkeypatch.setattr(we, "_sleep", lambda s: calls["sleeps"].append(s))
    return calls


ENT = "example-entity"


def test_unset_or_empty_entity_refuses_without_touching_the_api(monkeypatch):
    calls = _fake(monkeypatch, ENT)
    for value in (None, "", "   ", "\t"):
        if value is None:
            monkeypatch.delenv("WANDB_ENTITY", raising=False)
        else:
            monkeypatch.setenv("WANDB_ENTITY", value)
        with pytest.raises(we.WandbEntityError, match="WANDB_ENTITY is unset"):
            we.require_wandb_entity()
    assert calls["api"] == 0
    # the hint (the Modal wrappers name the secret to mount) rides along
    with pytest.raises(we.WandbEntityError, match="mount the 'x' secret"):
        we.require_wandb_entity(hint="mount the 'x' secret")
    assert issubclass(we.WandbEntityError, RuntimeError)


def test_foreign_key_refuses(monkeypatch):
    calls = _fake(monkeypatch, "someone-else-none")
    monkeypatch.setenv("WANDB_ENTITY", ENT)
    with pytest.raises(we.WandbEntityError, match="foreign key") as ei:
        we.require_wandb_entity()
    assert "someone-else-none" in str(ei.value) and ENT in str(ei.value)
    assert calls["api"] == 1 and calls["sleeps"] == []


def test_ok_returns_the_entity_and_retries_transient_errors_within_bound(
        monkeypatch):
    calls = _fake(monkeypatch, ENT, api_fail=2)
    monkeypatch.setenv("WANDB_ENTITY", ENT)
    assert we.require_wandb_entity() == ENT
    assert calls["api"] == 3                      # two failures, then verified
    assert calls["sleeps"] == [2.0, 4.0]          # backoff between tries only
    assert calls["timeouts"] == [60.0, 60.0, 60.0]
    # a clean key needs one call and no sleep
    calls = _fake(monkeypatch, ENT)
    assert we.require_wandb_entity(timeout=5) == ENT
    assert calls["api"] == 1 and calls["sleeps"] == [] and calls["timeouts"] == [5]


def test_api_errors_past_the_bound_refuse_not_degrade(monkeypatch):
    calls = _fake(monkeypatch, ENT, api_fail=99)
    monkeypatch.setenv("WANDB_ENTITY", ENT)
    with pytest.raises(we.WandbEntityError, match="after 3 tries") as ei:
        we.require_wandb_entity()
    assert "ConnectionError: down #3" in str(ei.value)
    assert calls["api"] == 3 and calls["sleeps"] == [2.0, 4.0]
    calls = _fake(monkeypatch, ENT, api_fail=99)
    with pytest.raises(we.WandbEntityError, match="after 1 tries"):
        we.require_wandb_entity(tries=1)
    assert calls["api"] == 1 and calls["sleeps"] == []


def test_env_override_is_honoured(monkeypatch):
    calls = _fake(monkeypatch, "team-x")
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    assert we.require_wandb_entity(env={"WANDB_ENTITY": "team-x"}) == "team-x"
    assert calls["api"] == 1


def test_wandb_init_checked_passes_the_validated_entity_and_retries(monkeypatch):
    calls = _fake(monkeypatch, ENT, api_fail=1, init_fail=1)
    monkeypatch.setenv("WANDB_ENTITY", ENT)
    run = we.wandb_init_checked(project="p", name="n", config={"a": 1},
                                tags=["t"])
    assert run.id == "run"
    assert calls["api"] == 2 and len(calls["init"]) == 2
    kw = calls["init"][-1]
    assert kw == {"project": "p", "entity": ENT, "name": "n",
                  "config": {"a": 1}, "tags": ["t"]}
    assert calls["sleeps"] == [2.0, 2.0]          # one api retry, one init retry


def test_wandb_init_checked_refuses_when_init_keeps_failing(monkeypatch):
    calls = _fake(monkeypatch, ENT, init_fail=99)
    monkeypatch.setenv("WANDB_ENTITY", ENT)
    with pytest.raises(we.WandbEntityError, match="failed 3 times") as ei:
        we.wandb_init_checked(project="p", name="n")
    assert "OSError: init down" in str(ei.value)
    assert len(calls["init"]) == 3
    # and a misrouted key never reaches init at all
    calls = _fake(monkeypatch, "someone-else-none")
    with pytest.raises(we.WandbEntityError, match="foreign key"):
        we.wandb_init_checked(project="p", name="n")
    assert calls["init"] == []
