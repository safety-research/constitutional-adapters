"""No test resolves a LIVE registry spool by default (2026-10-02).

``sisteer.registry.emit.spool_root`` probes fixed host paths — the per-user shared roots, the Modal volume, ``~/registry`` — and on the
grading host ``~/registry`` IS the live registry root (the harvester folds its ``records/`` on a timer). A test that opted emission back on
(``monkeypatch.delenv(emit.DISABLED_ENV)``) and ran ``emit_cli register --from-config`` WITHOUT ``--spool`` wrote two synthetic ``run``
records straight into it, and they were in the database within the minute (two records withdrawn). ``tests/conftest.py`` now wears two belts
for the whole session, in-process and in every child a test spawns: every probe is pointed at an EMPTY host (``SISTEER_REGISTRY_PROBE_ROOT``),
so no fixed host path can resolve, and ``SISTEER_REGISTRY_SPOOL`` names a hermetic spool UNDER that host, so "a spool resolves" stays true for
the opt-in tests whose hooks ask the resolver (2 of 45 files went quiet with the probe root alone). A spool a test or a harness names still
wins; a test that deletes the spool variable gets "no spool resolves". Pinned here: the host holds nothing but the suite spool; the
resolution is the suite spool in-process and in a child, never the live root; a child CLI without ``--spool`` writes under the host and
nowhere else; an explicit spool wins; with the spool variable deleted nothing resolves and ``register`` refuses (exit 2); and the known-bad
shape, hermetically: with both belts off and a provisioned home registry the resolver finds it — what the fixture prevents."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from sisteer.registry import emit, emit_cli, synth

REPO = Path(__file__).resolve().parents[1]
LIVE_ROOT = os.path.realpath(os.path.join(os.path.expanduser("~/registry"), "records"))


def _suite_host() -> str:
    root = os.environ.get(emit.PROBE_ROOT_ENV, "")
    spool = os.environ.get(emit.SPOOL_ENV, "")
    if not root or not spool or not spool.startswith(root):
        pytest.skip(f"the caller named its own spool / probe host ({emit.SPOOL_ENV}={spool!r}, {emit.PROBE_ROOT_ENV}={root!r}): the session fixture yields to it")
    return root


def test_the_suite_points_every_probe_at_an_empty_host_and_names_a_hermetic_spool_under_it():
    root = _suite_host()
    assert os.path.isdir(root) and os.listdir(root) == ["suite-spool"], os.listdir(root)                   # nothing provisioned on the host but the suite spool
    spool = os.environ[emit.SPOOL_ENV]
    assert spool == os.path.join(root, "suite-spool", "records") and os.path.isdir(spool)
    home_reg = emit.probe_path(emit.HOME_REGISTRY)
    assert home_reg.startswith(root) and not os.path.isdir(home_reg)                                       # ~/registry is looked up UNDER the empty host
    assert os.path.realpath(home_reg) != os.path.realpath(os.path.expanduser(emit.HOME_REGISTRY))


def test_the_resolution_is_the_suite_spool_never_the_live_root_and_nothing_resolves_once_the_variable_is_deleted(tmp_path, monkeypatch):
    root = _suite_host()
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)                                                   # the opt-in shape that leaked
    monkeypatch.delenv("REGISTRY_EMIT", raising=False)
    suite_spool = os.environ[emit.SPOOL_ENV]
    assert emit.spool_root() == suite_spool == emit_cli._spool(None)
    assert os.path.realpath(emit_cli._spool(None)).startswith(os.path.realpath(root)) and os.path.realpath(emit_cli._spool(None)) != LIVE_ROOT
    # an explicit spool wins over the suite's, as it always did
    monkeypatch.setenv(emit.SPOOL_ENV, str(tmp_path / "named-spool"))
    assert emit_cli._spool(None) == str(tmp_path / "named-spool") and emit_cli._spool(str(tmp_path / "flag")) == str(tmp_path / "flag")
    # the spool variable deleted: the first belt — nothing resolves, whatever the host would offer
    monkeypatch.delenv(emit.SPOOL_ENV)
    assert emit.spool_root() is None and emit_cli._spool(None) is None
    # the known-bad shape, hermetically: both belts off and a provisioned "home registry" -> the resolver finds it (what the fixture prevents); the Modal
    # volume probe this once also parked is gone with the Modal transport
    monkeypatch.delenv(emit.PROBE_ROOT_ENV)
    fake_home_reg = tmp_path / "home" / "registry"; (fake_home_reg / "records").mkdir(parents=True)
    monkeypatch.setattr(emit, "HOME_REGISTRY", str(fake_home_reg))
    monkeypatch.setattr(emit, "SHARED_BASES", ())
    assert emit_cli._spool(None) == str(fake_home_reg / "records")
    # the probe root back: the same provisioned home registry is invisible again
    monkeypatch.setenv(emit.PROBE_ROOT_ENV, str(tmp_path / "empty-host"))
    assert emit_cli._spool(None) is None


def test_a_child_without_a_spool_flag_writes_under_the_suite_host_and_nowhere_else_and_refuses_once_the_variable_is_deleted(tmp_path):
    root = _suite_host()
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    env.pop(emit.DISABLED_ENV, None); env.pop("REGISTRY_EMIT", None)                                      # a child that opted emission on
    out = subprocess.run([sys.executable, "-c", "from sisteer.registry import emit_cli; print(emit_cli._spool(None))"],
                         env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and out.stdout.strip() == os.environ[emit.SPOOL_ENV], out.stdout + out.stderr
    # the real CLI over the real seeds, NO --spool: every record path lands under the suite host — never the live root
    co = tmp_path / "checkout"; (co / "configs").mkdir(parents=True); os.symlink(REPO / "configs" / "registry", co / "configs" / "registry")
    cfg = co / "configs" / "sdfbase_full22_qwen35_4b.yaml"; cfg.write_text("explicit_cells: [none]\n")

    def register(job: str) -> list:
        return [sys.executable, "-m", "sisteer.registry.emit_cli", "register", "--from-config", str(cfg), "--model-key", "qwen35_4b", "--evals", "pasttense",
                "--serving-path", "lora_native", "--campaign", "sdfbase_full22", "--job-id", job, "--pending-dir", str(tmp_path / "p"), "--code-sha", synth.code_sha("a"),
                "--substrate", "cluster_a", "--no-push"]
    r = subprocess.run(register("990001"), env=env, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and (tmp_path / "p" / "990001.json").exists(), r.stdout + r.stderr
    written = [p for p in r.stdout.splitlines() if p.strip()]
    assert written and all(os.path.realpath(p).startswith(os.path.realpath(root)) for p in written), written   # the record paths the CLI printed
    assert not any(os.path.realpath(p).startswith(LIVE_ROOT) for p in written)
    # the spool variable deleted in the child: the first belt — nothing resolves, the CLI refuses (exit 2) and writes nothing
    env2 = dict(env); env2.pop(emit.SPOOL_ENV, None)
    r2 = subprocess.run(register("990002"), env=env2, capture_output=True, text=True, timeout=300)
    assert r2.returncode == 2 and "no spool resolves" in r2.stderr and not (tmp_path / "p" / "990002.json").exists(), r2.stdout + r2.stderr


def test_the_fixture_is_the_sessions_and_yields_to_the_callers_host():
    src = (REPO / "tests" / "conftest.py").read_text()
    i = src.index("def _no_live_registry_spool_for_the_suite(tmp_path_factory):")
    assert src[:i].rstrip().endswith('@pytest.fixture(scope="session", autouse=True)')
    body = src[i:i + 3600]
    assert 'if os.environ.get(spool_var) or os.environ.get(probe_var):' in body and 'os.environ[probe_var] = str(host)' in body
    assert 'os.environ[spool_var] = str(suite_spool)' in body and 'suite_spool = host / "suite-spool" / "records"' in body
    assert "os.environ.pop(probe_var, None)" in body and "os.environ.pop(spool_var, None)" in body and 'tmp_path_factory.mktemp("no-live-spool-host")' in body
