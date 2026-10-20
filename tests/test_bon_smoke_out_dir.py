"""The integration smoke's output root is FRESH per run.

The registry export is non-idempotent by design (lens_steer._write_blob
chmods its blobs 0440 — an immutable snapshot under a running server), so a
reused output directory is poisoned by any prior attempt: job 213880 failed
late and left 0440 blobs; job 213947 then died at _write_blob in 17s. The
smoke therefore keys its out dir on the attempt (SLURM_JOB_ID, else a UTC
timestamp) so no prior state can ever collide."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_SMOKE = (Path(__file__).resolve().parents[1] / "scripts" /
          "petri_bon_smoke_integration.py")


def _load_smoke():
    spec = importlib.util.spec_from_file_location(
        "petri_bon_smoke_integration", _SMOKE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_out_dir_is_per_job_and_never_the_shared_root(monkeypatch):
    smoke = _load_smoke()
    monkeypatch.setenv("SLURM_JOB_ID", "213880")
    a = smoke.run_out_dir()
    monkeypatch.setenv("SLURM_JOB_ID", "213947")
    b = smoke.run_out_dir()
    assert a != b                                   # one dir per attempt
    assert a.name == "run_213880" and b.name == "run_213947"
    shared_root = smoke.SHARED / "petri" / "serving" / "bon_smoke_integration"
    assert a.parent == shared_root and a != shared_root


def test_out_dir_timestamp_fallback_without_slurm(monkeypatch):
    smoke = _load_smoke()
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    d = smoke.run_out_dir()
    assert d.name.startswith("run_") and d.name != "run_"
    # a pod/local run still nests under the smoke's own root
    assert d.parent.name == "bon_smoke_integration"
