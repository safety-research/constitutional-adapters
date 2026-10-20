"""_tsplit_det_wrapper (P4 proof): the proof-only determinism wrapper
must set every knob BEFORE handing control to the wrapped script, pass argv
through unchanged, and fail loudly on a bad mode — the knobs are what make
the A/A control (and therefore the A/B env-parity verdict) interpretable,
so a silently-skipped knob would void the criterion. Runs on CPU torch."""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "scripts" / "_tsplit_det_wrapper.py"

PROBE = """\
import json, os, sys
import torch
json.dump({
    "argv": sys.argv,
    "deterministic": torch.are_deterministic_algorithms_enabled(),
    "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
    "cudnn_benchmark": torch.backends.cudnn.benchmark,
    "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
    "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    "cublas_ws": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
}, open(sys.argv[-1], "w"))
"""


def _run_wrapper(tmp_path, env_extra=None, args=None):
    probe = tmp_path / "probe.py"
    probe.write_text(PROBE)
    dump = tmp_path / "dump.json"
    env = {k: v for k, v in os.environ.items()
           if k not in ("TSPLIT_DET", "CUBLAS_WORKSPACE_CONFIG")}
    env.update(env_extra or {})
    argv = [sys.executable, str(WRAPPER), str(probe),
            *(args or []), str(dump)]
    proc = subprocess.run(argv, capture_output=True, text=True,
                          cwd=str(tmp_path), env=env)
    return proc, (json.loads(dump.read_text()) if dump.is_file() else None)


def test_strict_mode_sets_every_knob_and_passes_argv(tmp_path):
    proc, state = _run_wrapper(tmp_path, args=["--alpha", "2"])
    assert proc.returncode == 0, proc.stderr
    assert "[tsplit-det] determinism knobs ON (mode=strict)" in proc.stdout
    assert state["deterministic"] is True
    assert state["warn_only"] is False
    assert state["cudnn_benchmark"] is False
    assert state["matmul_tf32"] is False
    assert state["cudnn_tf32"] is False
    assert state["cublas_ws"] == ":4096:8"
    # argv[0] is the wrapped script, exactly as if invoked directly
    assert state["argv"][0].endswith("probe.py")
    assert state["argv"][1:3] == ["--alpha", "2"]


def test_warn_mode_is_explicit_and_warn_only(tmp_path):
    proc, state = _run_wrapper(tmp_path, env_extra={"TSPLIT_DET": "warn"})
    assert proc.returncode == 0, proc.stderr
    assert "mode=warn" in proc.stdout   # the fallback must SAY so
    assert state["deterministic"] is True
    assert state["warn_only"] is True


def test_unknown_mode_is_fatal(tmp_path):
    proc, _ = _run_wrapper(tmp_path, env_extra={"TSPLIT_DET": "bogus"})
    assert proc.returncode != 0
    assert "unknown TSPLIT_DET" in proc.stderr


def test_no_target_is_usage_error():
    proc = subprocess.run([sys.executable, str(WRAPPER)],
                          capture_output=True, text=True)
    assert proc.returncode != 0
    assert "usage:" in proc.stderr