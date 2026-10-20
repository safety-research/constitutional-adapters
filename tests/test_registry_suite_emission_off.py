"""The test suite never emits into a real results-registry spool (2026-09-18).

The one switch (``sisteer.registry.hooks.emission_enabled``) is ON wherever a spool resolves, and the suite's gate
container of that time carried a provisioned spool of its own — so every fit a test ran through the fit writers'
hooks registered its tiny-model object into the LIVE log: 2,727 ``object`` records (855 run directories, 3,168
files) blocked on ``hf-internal-testing/tiny-random-LlamaForCausalLM``, each stamped ``scripts/fit_class_b.py``,
``fit_layers_json [1]``, ``campaign class_b``. The log is append-only, so they stay blocked rather than deleted. Two
belts close the leak, pinned here: ``tests/conftest.py`` sets ``SISTEER_REGISTRY_DISABLED=1`` for the whole session
unless the caller set the variable, and the Slurm gate's ``suite_env`` (``scripts/registry/landing/gate_lib.py``, its
own since the gate became Slurm-only) sets it in the gate's environment. Known-bad first: the pre-fix default on a
host with a provisioned spool (a grading-host-shaped ``~/registry``, under a probe root; the gate container's volume
probe left with the Modal transport) writes the very records that leaked; the suite default writes nothing; the
opt-in writes to the fixture spool only.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from sisteer.registry import emit, hooks
from sisteer.registry import records as R

REPO = Path(__file__).resolve().parents[1]
TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
TRUTHY = ("1", "true", "yes", "on")


def test_suite_default_is_emission_off():
    """What every test in this session sees: the switch variable set by ``tests/conftest.py`` and emission off."""
    raw = os.environ.get(emit.DISABLED_ENV)
    assert raw is not None, f"tests/conftest.py's session fixture did not set {emit.DISABLED_ENV}"
    if raw.strip().lower() not in TRUTHY:
        pytest.skip(f"{emit.DISABLED_ENV}={raw!r}: the caller opted this run into emission")
    assert hooks.emission_enabled() is False
    assert emit.RecordWriter("/nonexistent/spool", "class_b", "object__0000", {"script": "t"}).disabled


def test_conftest_default_yields_to_the_callers_setting(tmp_path):
    """``SISTEER_REGISTRY_DISABLED`` set by the caller (here ``=0``) stands: the session fixture does not overwrite it,
    so a run deliberately opted into emission stays opted in — the default test above then skips rather than fails."""
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    env[emit.DISABLED_ENV] = "0"
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(REPO),
                          f"{Path(__file__)}::test_suite_default_is_emission_off", "-rs"],
                         cwd=str(REPO), env=env, capture_output=True, text=True, timeout=600)
    # the repo's addopts already carry -q, so the child prints no "1 skipped" tally: read the -rs reason line
    assert "SKIPPED [1]" in out.stdout and "opted this run into emission" in out.stdout, out.stdout + out.stderr


def _class_b_alphasteer_dir(tmp_path: Path) -> Path:
    """An object directory shaped like ``scripts/fit_class_b.py``'s alphasteer output on the tiny model: the artifact
    plus the ``fit_meta.json`` / ``fit_provenance.json`` / ``meta.json`` sidecars the hook reads."""
    d = tmp_path / "objects" / "alphasteer"
    d.mkdir(parents=True)
    (d / "object.pt").write_bytes(b"synthetic alphasteer object bytes " * 32)
    sha = hashlib.sha256((d / "object.pt").read_bytes()).hexdigest()
    (d / "fit_meta.json").write_text(json.dumps({"arm": "alphasteer", "layers": [1], "seed": 0, "object_sha256": sha}, indent=1))
    (d / "fit_provenance.json").write_text(json.dumps({
        "resolved_model_id": TINY_LLAMA, "resolved_revision": "a" * 40, "chat_template_hash": "0" * 16,
        "writer": "fit_class_b.py", "positions_at_fit": "response"}, indent=1))
    (d / "meta.json").write_text(json.dumps({"arm": "alphasteer", "layers": [1], "model": TINY_LLAMA}, indent=1))
    return d


def _fit_class_b():
    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    import fit_class_b  # noqa: E402 — the real writer (torch import; the suite pays it in test_fit_class_b anyway)
    return fit_class_b


def _records_under(spool: Path):
    return [rec for p in R.iter_record_files(str(spool)) for _, rec, _ in R.read_record_file(p)]


def test_fit_registration_on_a_provisioned_host_leaks_without_the_default_and_not_with_it(tmp_path, monkeypatch, capsys):
    """Known-bad first. A host with a provisioned spool (``~/registry`` under the probe root, nothing else -- the shape of the
    grading host; the 2026-09-17 leak came through a gate container's own provisioned volume, a probe that left with the Modal
    transport) and the smoke fit's own registration path, ``fit_class_b._registry_object``: with the pre-fix suite environment
    (the switch variable unset) it writes the ``object`` + ``artifact`` records that reached the live log — ``object_kind
    alphasteer``, ``fit_model_id`` the tiny model, ``fit_layers_json [1]``, ``campaign class_b``, producer
    ``scripts/fit_class_b.py`` on the host's substrate. With the suite default (``SISTEER_REGISTRY_DISABLED=1``) the same call
    writes nothing there; with the opt-in (variable unset, a fixture spool named) it writes to the fixture spool and nothing
    else."""
    host = tmp_path / "host"
    monkeypatch.setenv(emit.PROBE_ROOT_ENV, str(host))
    for var in (emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, emit.SUBSTRATE_ENV, hooks.EMIT_ENV, hooks.RUN_IDENTITY_ENV,
                "REGISTRY_CAMPAIGN", "SLURM_CLUSTER_NAME", "SLURM_JOB_ID", "RUNPOD_POD_ID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("USER", "nobody-synthetic")
    monkeypatch.setenv("HOME", str(host / "home"))
    home_registry = Path(emit.probe_path(emit.HOME_REGISTRY))       # <probe root>/<$HOME>/registry: the grading host's spool, relocated
    home_registry.mkdir(parents=True)
    live = home_registry / "records"
    obj_dir = _class_b_alphasteer_dir(tmp_path)
    fcb = _fit_class_b()

    # known-bad: the pre-fix suite left the variable unset, so rule 4 applies and this host's spool resolves
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    assert emit.spool_root() == str(live) and emit.substrate_from_env() == "grading_host" and hooks.emission_enabled()
    fcb._registry_object({}, "alphasteer", obj_dir)
    leaked = _records_under(live)
    assert leaked, "the smoke fit registered nothing on a host where the pre-fix suite leaked; stderr: " + capsys.readouterr().err
    assert [r["record_type"] for r in leaked] == ["object", "artifact"]
    o = leaked[0]["payload"]
    assert o["object_kind"] == "alphasteer" and o["fit_model_id"] == TINY_LLAMA and o["fit_layers_json"] == [1]
    assert o["campaign"] == "class_b" and o["object_role"] == "safety" and o["fit_objective"] == "alphasteer"
    assert leaked[0]["producer"]["script"] == "scripts/fit_class_b.py" and leaked[0]["producer"]["substrate"] == "grading_host"

    # the fix: the suite default (tests/conftest.py) and the gate's environment (gate_lib.suite_env)
    shutil.rmtree(live)
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    assert not hooks.emission_enabled()
    fcb._registry_object({}, "alphasteer", obj_dir)
    assert not live.exists(), sorted(str(p) for p in live.rglob("*"))
    assert "[registry]" not in capsys.readouterr().err  # off is silent: a no-op, not a failure

    # the opt-in a registry test makes: the variable unset AND its own spool — the fixture spool fills, the host's stays empty
    monkeypatch.delenv(emit.DISABLED_ENV)
    fixture_spool = tmp_path / "fixture-spool"
    monkeypatch.setenv(emit.SPOOL_ENV, str(fixture_spool))
    fcb._registry_object({}, "alphasteer", obj_dir)
    kept = _records_under(fixture_spool)
    assert [r["record_type"] for r in kept] == ["object", "artifact"] and kept[0]["payload"]["object_id"] == o["object_id"]
    assert all(os.path.dirname(p).endswith(f"object__{o['object_id'][:16]}") and str(fixture_spool / "class_b") in p
               for p in R.iter_record_files(str(fixture_spool)))
    assert not live.exists()
