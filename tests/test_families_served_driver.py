"""Pure logic of the served-wave driver's render proof (Qwen3.5
onboarding, 2026-08-30): the HTTP half of the thinking-off chain —
template-hash equality is necessary, the served token count is the
proof of which regime actually rendered."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location(
    "_families_served", ROOT / "scripts" /
    "petri_families_served_integration.py")
fam = importlib.util.module_from_spec(_spec)
sys.modules["_families_served"] = fam
_spec.loader.exec_module(fam)


def test_render_verdict_passes_thinking_off():
    assert fam.thinking_render_verdict(207, 207, 200) is None


def test_render_verdict_names_a_thinking_on_serve():
    v = fam.thinking_render_verdict(200, 207, 200)
    assert v and "thinking-ON" in v and "wrong boundary" in v


def test_render_verdict_names_a_third_render():
    v = fam.thinking_render_verdict(999, 207, 200)
    assert v and "NEITHER" in v


def test_render_verdict_silent_without_a_discriminating_template():
    """No thinking switch (dense models), or coinciding renders: nothing
    to prove, and nothing that can false-fail a llama/gemma wave."""
    assert fam.thinking_render_verdict(123, 207, None) is None
    assert fam.thinking_render_verdict(123, 207, 207) is None


def test_run_out_dir_wave_override_enables_resume(monkeypatch, tmp_path):
    """WAVE_OUT_DIR points a resubmit at a prior job's out dir so BoN and the
    families resume rather than regenerating ~16.5k rows (jobs 272640/272642
    died on a hardcoded 90-min subprocess cap; the resume path is how a
    corrected resubmit reuses the surviving BoN leg)."""
    monkeypatch.setenv("WAVE_OUT_DIR", str(tmp_path / "run_prior"))
    monkeypatch.setenv("SLURM_JOB_ID", "999999")  # must be IGNORED when override set
    assert fam.run_out_dir() == tmp_path / "run_prior"
    monkeypatch.delenv("WAVE_OUT_DIR")
    # without the override, the SLURM job id keys a fresh dir
    assert fam.run_out_dir().name == "run_999999"
