"""Every front-door stage answers --help (exit 0) and documents the common flags."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPRO, STAGES, assert_no_tokens  # noqa: E402

REQUIRED_IN_HELP = ("--dry-run", "--config", "--help")


@pytest.mark.parametrize("stage", STAGES)
def test_stage_help_exits_zero_and_names_the_common_flags(stage):
    r = subprocess.run(["bash", str(REPRO / stage), "--help"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, (stage, r.stdout[-800:], r.stderr[-800:])
    out = r.stdout + r.stderr
    for flag in REQUIRED_IN_HELP:
        assert flag in out, (stage, flag)
    if stage not in ("00_setup.sh", "01_make_corpora.sh", "10_figures.sh"):
        assert "--model" in out, stage
    assert_no_tokens(out, stage)


def test_the_stage_list_is_numbered_and_unique():
    nums = [s[:2] for s in STAGES]
    assert nums == sorted(nums) and len(set(nums)) == len(nums), STAGES
    for owned in ("00_setup.sh", "05_stage_roster.sh", "06_eval_waves.sh", "07_bcb_bfcl.sh", "09_dose_ladder.sh"):
        assert owned in STAGES, STAGES
