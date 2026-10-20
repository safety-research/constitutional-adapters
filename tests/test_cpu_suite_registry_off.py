"""The CPU-suite gate's containers spool the results registry into a THROWAWAY directory.

The gate runs the repo's test files inside a Modal container whose volume carries a provisioned ``/vol/registry``; the
tests' own smoke fits carry a default-on registry emitter, so left alone the gate is a PRODUCER: 879 ``class_b`` object
records from tiny smoke models reached the live record spool. The first answer, ``SISTEER_REGISTRY_DISABLED=1`` in
both places the container env is composed, over-shot: that switch makes EVERY emitter method a no-op
(``sisteer.registry.emit.RecordWriter``), so 12 registry tests that emit into their own tmp spool failed inside the gate
while passing everywhere else. A fresh per-run spool under the container's tmp replaces the switch: emission
stays ON, ``emit.spool_root`` returns ``$SISTEER_REGISTRY_SPOOL`` before it probes the volume, and the runner removes
the directory once pytest is gone, whatever its rc. Pinned here: neither env site carries the switch; ``suite_env``
mints the spool and drops every knob that could re-disable or re-route emission; under the composed env the hooks
resolve the throwaway spool over a provisioned host (the Modal volume probe itself left with the Modal transport, so a
probe-rooted ``vol/registry`` alone now resolves nothing); and ``run_pytest`` removes it on
exit AND on timeout."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPOOL_ENV = "SISTEER_REGISTRY_SPOOL"
#: the knobs the old switch sat among; every one of them is popped by ``suite_env`` (REGISTRY_ENV_POPPED)
KNOBS = {"SISTEER_REGISTRY_DISABLED": "1", "REGISTRY_EMIT": "0", "SISTEER_REGISTRY_PUSH": "/nonexistent/push_records.sh"}


def _prune_test_module():
    spec = importlib.util.spec_from_file_location("_cpu_suite_prune_test", REPO / "tests/test_cpu_suite_prune.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cs():
    return _prune_test_module()._load()


def test_the_gate_subprocess_env_mints_a_fresh_throwaway_spool_under_the_given_root(cs, tmp_path):
    base = {"PATH": "/usr/bin", SPOOL_ENV: "/vol/registry/records", **KNOBS}
    env = cs.suite_env(base, tmp_path / "tree", spool_root=tmp_path)
    spool = Path(env[SPOOL_ENV])
    assert spool.is_dir(), "suite_env must CREATE the spool (an emitter given a missing directory makes it, anywhere)"
    assert spool.parent == tmp_path and spool.name.startswith(cs.REGISTRY_SPOOL_PREFIX)
    assert env[SPOOL_ENV] != base[SPOOL_ENV], "an inherited spool (the volume's) is never kept"
    # fresh per run: a second composition is a second directory
    again = Path(cs.suite_env(base, tmp_path / "tree", spool_root=tmp_path)[SPOOL_ENV])
    assert again.is_dir() and again != spool
    # the old switch is GONE, and so is every other knob that could re-disable or re-route emission
    for name in KNOBS:
        assert name not in env, name
    assert set(KNOBS) == set(cs.REGISTRY_ENV_POPPED)
    assert not any(k == "SISTEER_REGISTRY_DISABLED" for k in env), "D132: no global off switch in the gate env"


def test_the_default_spool_root_is_the_containers_tmp(cs, tmp_path):
    env = cs.suite_env({"PATH": "/usr/bin"}, tmp_path / "tree")
    spool = Path(env[SPOOL_ENV])
    try:
        assert spool.is_dir()
        assert spool.parent == Path(tempfile.gettempdir())
    finally:
        spool.rmdir()


def test_the_gate_image_env_carries_no_registry_switch():
    src = (REPO / "scripts/modal/cpu_suite.py").read_text()
    m = re.search(r"\.env\(\{(.*?)\}\)", src, re.S)
    assert m, "cpu_image has no .env({...}) call"
    assert "SISTEER_REGISTRY" not in m.group(1), "the image env must steer the registry through suite_env, never image-wide"
    assert "REGISTRY_EMIT" not in m.group(1)


def _resolve_under(env: dict) -> dict:
    """What the registry hooks resolve in a SUBPROCESS carrying exactly ``env`` (the gate's shape: PYTHONPATH from
    ``suite_env`` points at the tree's ``src``)."""
    code = ("import json, sisteer.registry.hooks as H, sisteer.registry.emit as E; "
            "print(json.dumps({'resolve_spool': H.resolve_spool(), 'enabled': H.emission_enabled(), "
            "'spool_root': E.spool_root()}))")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_under_the_composed_env_the_hooks_resolve_the_throwaway_spool_not_the_provisioned_host(cs, tmp_path):
    """The goal kept: a host whose (probe-rooted) ``~/registry`` IS provisioned -- the shape of a host with a live
    spool -- and the composed env still spools to the throwaway directory with emission ON. (The provisioned host of this
    test was once the Modal container's ``/vol/registry``; that probe left with the Modal transport, so a
    probe-rooted ``vol/registry`` now resolves nothing -- pinned below.)"""
    host = tmp_path / "host"
    (host / "vol" / "registry").mkdir(parents=True)
    home = tmp_path / "home"
    home.mkdir()
    home_registry = host / str(home).lstrip("/") / "registry"        # emit.probe_path("~/registry") under the probe root
    home_registry.mkdir(parents=True)
    base = {"PATH": os.environ.get("PATH", ""), "HOME": str(home), "SISTEER_REGISTRY_PROBE_ROOT": str(host), **KNOBS}
    env = cs.suite_env(base, REPO, spool_root=tmp_path)   # tree=REPO: PYTHONPATH -> this checkout's src
    got = _resolve_under(env)
    assert got["enabled"] is True, got
    assert got["resolve_spool"] == env[SPOOL_ENV] == got["spool_root"], got
    # control: the SAME env minus the throwaway spool would have found the provisioned host -- the explicit spool is
    # the only thing between the smoke fits and the live records (emit.spool_root: the env value returns first)
    control = dict(env)
    del control[SPOOL_ENV]
    ctl = _resolve_under(control)
    assert ctl["resolve_spool"] == str(home_registry / "records"), ctl
    assert ctl["enabled"] is True
    # the removed probe (C10): a probe-rooted vol/registry alone is NOT a registry host -- nothing resolves, emission is off
    import shutil
    shutil.rmtree(home_registry)
    vol_only = _resolve_under(control)
    assert vol_only == {"resolve_spool": None, "enabled": False, "spool_root": None}, vol_only


def test_run_pytest_removes_the_throwaway_spool_whatever_the_rc(cs, tmp_path):
    env = cs.suite_env({"PATH": os.environ.get("PATH", "")}, tmp_path, spool_root=tmp_path)
    spool = Path(env[SPOOL_ENV])
    writer = ("import os, sys; p = os.path.join(os.environ['SISTEER_REGISTRY_SPOOL'], 'c', 'r.jsonl'); "
              "os.makedirs(os.path.dirname(p)); open(p, 'w').write('x'); sys.exit(3)")
    rc, timed_out = cs.run_pytest([sys.executable, "-c", writer], tmp_path, env, tmp_path / "pytest.log", 60)
    assert (rc, timed_out) == (3, False)
    assert not spool.exists(), "records the smoke fits spooled must not outlive the run"


def test_run_pytest_removes_the_throwaway_spool_on_timeout(cs, tmp_path):
    env = cs.suite_env({"PATH": os.environ.get("PATH", "")}, tmp_path, spool_root=tmp_path)
    spool = Path(env[SPOOL_ENV])
    rc, timed_out = cs.run_pytest([sys.executable, "-c", "import time; time.sleep(60)"], tmp_path, env,
                                  tmp_path / "pytest.log", 1)
    assert timed_out is True and rc != 0
    assert not spool.exists()
