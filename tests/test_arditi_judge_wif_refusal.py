"""The Arditi OR-Bench grader's judge is an INDIRECT reacher of Claude: ``scripts/arditi/grade_arditi_orbench.py``
launches the pinned ``scripts/served_capability_judge.py`` of a SEPARATE checkout by path, with the Low-Prio KEY in the child's
environment (``harness_env``: every inherited ``ANTHROPIC_*`` dropped, ``ANTHROPIC_API_KEY=<the key>``). The AST inventory
(``tests/test_wif_driver_inventory.py``) cannot see a subprocess reacher, so its behaviour under ``SISTEER_SECRETS_MODE=wif`` is pinned
here: the frozen driver reads a key and nothing else, there is no key under federation, and ``_vault_lowprio`` -> ``vault.anthropic_api_key``
REFUSES BY NAME (the vault's ``PoolUnavailable``, surfaced as the grader's ``Refuse``) BEFORE the private cache is seeded and BEFORE the
child is launched -- the same frozen-driver refusal the sweep records for ``grade_alphasteer_controls.py``; never a federated client
handed to a driver that cannot use it, never the shell's ``ANTHROPIC_API_KEY`` read in its place.

The script's two lane-local helper modules (``grade_arditi_static``, ``battery_curves``) are not on this tree (they live on the Arditi
lane's branch; the grader landed on main without them -- a pre-existing condition of the base, recorded in the sweep's residuals), so
they are stubbed here to the names the import line binds; nothing under test lives in them."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/arditi/grade_arditi_orbench.py"
PER_ORG = "synthetic-per-org-key-not-a-key"
ONE_KEY = "synthetic-highprio-key-not-a-key"


@pytest.fixture(scope="module")
def grader():
    """The script loaded by path, its two absent lane-local helpers stubbed to the names its import lines bind."""
    gas = types.ModuleType("grade_arditi_static")

    class Refuse(RuntimeError):
        pass

    gas.Refuse = Refuse; gas.DETECTORS = (); gas._detector_view = lambda *a, **k: None; gas.degeneration_block = lambda *a, **k: {}; gas.ts = lambda: "t"
    bc = types.ModuleType("battery_curves"); bc.GRADED_BY_LABEL = {}; bc.GRADED_AT_CAP = 0.5; bc.GATE_AT_CAP = 1.0
    saved = {k: sys.modules.get(k) for k in ("grade_arditi_static", "battery_curves")}
    sys.modules["grade_arditi_static"], sys.modules["battery_curves"] = gas, bc
    try:
        spec = importlib.util.spec_from_file_location("grade_arditi_orbench_under_test", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    assert mod.Refuse is Refuse and mod.JUDGE_ORG == "lowprio" and mod.JUDGE_DRIVER == "scripts/served_capability_judge.py"
    return mod


@pytest.mark.parametrize("dry_run", [False, True])
def test_under_wif_the_pinned_judge_driver_is_refused_by_name_before_the_cache_is_seeded_or_the_child_launched(grader, tmp_path, monkeypatch, dry_run):
    mod = grader
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "wif")
    monkeypatch.setenv("ANTHROPIC_API_KEY_LOWPRIO", PER_ORG)     # present in the shell, never read under wif
    monkeypatch.setenv("ANTHROPIC_API_KEY", ONE_KEY)
    launched: list = []
    monkeypatch.setattr(mod, "verify_driver", lambda tree: {"tree": str(tree)})                 # the instrument pin is not the subject
    monkeypatch.setattr(mod, "seed_private_cache", lambda *a, **k: launched.append(("seed", k)))
    monkeypatch.setattr(mod, "_run_logged", lambda *a, **k: launched.append(("run", k.get("what"))))
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **k: launched.append(("subprocess", a)))
    log = tmp_path / "judge.log"
    with pytest.raises(mod.Refuse) as ei:
        mod.invoke_harness_judge(tmp_path / "tag", 2048, judge_tree=tmp_path / "tree", private_cache=tmp_path / "cache.sqlite",
                                 log=log, concurrency=4, dry_run=dry_run)
    msg = str(ei.value)
    assert "no key for the pinned judge driver's lowprio pool" in msg and "SISTEER_SECRETS_MODE=wif" in msg
    assert "workload identity federation" in msg and "nothing reads ANTHROPIC_API_KEY*" in msg
    assert PER_ORG not in msg and ONE_KEY not in msg                                             # a refusal names variables, never values
    assert launched == [] and not log.exists() and not (tmp_path / "cache.sqlite").exists()     # before the seed, before the child


def test_the_refusal_is_the_modes_not_a_missing_key_and_the_childs_environment_carries_only_the_pools_key(grader, monkeypatch, tmp_path):
    """Under op the SAME shell resolves the pool's OWN variable (never the High-Prio ANTHROPIC_API_KEY) and the child's environment is
    built on it with every inherited ANTHROPIC_* dropped -- the frozen driver's contract; so what wif refuses is the key itself."""
    mod = grader
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "op")
    monkeypatch.setenv("ANTHROPIC_API_KEY_LOWPRIO", PER_ORG)
    monkeypatch.setenv("ANTHROPIC_API_KEY", ONE_KEY)
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "synthetic-oauth-bearer-not-a-token")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:1/session-gateway")
    key = mod._vault_lowprio()
    assert key == PER_ORG
    env = mod.harness_env(key, tmp_path / "cache.sqlite")
    assert env["ANTHROPIC_API_KEY"] == PER_ORG and env["CAPEVAL_JUDGE_CACHE"] == str(tmp_path / "cache.sqlite")
    assert [k for k in env if k.startswith("ANTHROPIC_")] == ["ANTHROPIC_API_KEY"]                 # the bearer, the gateway, the other keys: dropped
    # and the order of the refusal is the source's: the key is resolved before the seed and the launch in invoke_harness_judge
    src = SCRIPT.read_text()
    body = src.split("def invoke_harness_judge(", 1)[1].split("\ndef ", 1)[0]
    assert body.index("_vault_lowprio()") < body.index("seed_private_cache(") < body.index("_run_logged(cmd")
