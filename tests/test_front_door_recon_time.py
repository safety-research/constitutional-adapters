"""03_hpo.sh --phase recon sizes its walltime from the site (SISTEER_TIME_RECON), default 03:00:00 (paper-companion-repo stage E,
2026-09-30): a reconnaissance rung trains min(one epoch of the trial pool, 150 optimizer steps) at the folded effective batch, so
for a large model the five rungs run ~11 h and the paper models' 3 h, previously hard-coded, would TIMEOUT a job that has no resume.
The other three sbatch budgets already come from the site config; this pins the recon's the same way, for both protocols' classes."""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPRO, VENV_PY, run_stage  # noqa: E402

MODEL = "qwen35_4b_demo"      # the front-door model (models.front_door_keys()); its fits config is the unique pin of its fit checkpoint


def _fd6_overrides(root: Path) -> Path:
    """An overrides file in the FD 6 protocol (reconnaissance clip, one warm-up level, the LoRA architecture search): the protocol
    under which 03_hpo.sh has a recon phase at all — under the paper's defaults the clip is a SEARCHED axis and 03 says `recon: not applicable`."""
    p = root / "fd6_test.env"
    p.write_text("OVERRIDE_ID=fd6_test_v1\nOVERRIDE_INIT_SCHEME=random_lecun_1e-3\nOVERRIDE_LORA_ARCH_SEARCH=1\nOVERRIDE_LORA_RANK=''\nOVERRIDE_LORA_MODULE_SET=''\n"
                 "OVERRIDE_HPO_WARMUP_FRAC=0.15\nOVERRIDE_HPO_CLIP_MODE=reconnaissance\nOVERRIDE_HPO_CLIP_PERCENTILE=95\nOVERRIDE_HPO_TRIALS=16\n"
                 "OVERRIDE_HPO_RECENTRE_TRIALS=4\nOVERRIDE_REFUSAL_STEER_ALPHA=0.3\nOVERRIDE_EFFECTIVE_BATCH=64\n")
    return p


def _site(root: Path, extra: str = "") -> Path:
    OVERRIDES = _fd6_overrides(root)                                 # the FD 6 protocol (reconnaissance clip) — the recon applies
    site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n"
                 f"SISTEER_RUN_OVERRIDES={OVERRIDES}\n{extra}")
    return site


def _recon_lines(out: str) -> list[str]:
    return [ln for ln in out.splitlines() if "sbatch" in ln and "HPO_JOB_MODE=recon" in ln]


def test_example_site_recon_time_is_the_paper_default(tmp_path):
    assert re.search(r"^SISTEER_TIME_RECON=03:00:00\b", (REPRO / "site.example" / "config.env").read_text(), re.M), \
        "site.example/config.env documents SISTEER_TIME_RECON at the paper models' 03:00:00"
    r = run_stage("03_hpo.sh", "--dry-run", "--object-class", "relu", "--phase", "recon", "--model", MODEL, site=_site(tmp_path), timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    lines = _recon_lines(out)
    assert len(lines) == 1, out[-2500:]
    assert "--time=03:00:00" in lines[0], lines[0]


@pytest.mark.parametrize("cls", ["relu", "lora"])
def test_site_sizes_the_recon_walltime(tmp_path, cls):
    r = run_stage("03_hpo.sh", "--dry-run", "--object-class", cls, "--phase", "recon", "--model", MODEL,
                  site=_site(tmp_path, "SISTEER_TIME_RECON=1-00:00:00\n"), timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-2500:]
    lines = _recon_lines(out)
    assert len(lines) == 1, out[-2500:]
    assert "--time=1-00:00:00" in lines[0], lines[0]
    assert "--time=03:00:00" not in lines[0]
    assert "--array=0-4" in lines[0], lines[0]                       # ONE array, one rung per task; the ladder has 5 rungs
