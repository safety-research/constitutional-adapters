"""03_hpo.sh declares the RECON submission's registration contract `none` — and only that one.
The recon rungs run scripts/sbatch/hpo_trial.sbatch under HPO_JOB_MODE=recon and write only their own record
dir, so a dead recon array is re-minted at sacct-terminal alone; the same sbatch's TRIAL mode emits fit_trial and keeps the funnel's
`registry` contract; the serving probe is kind `other` and derives `none` from its script. Executed end to end: the stage's dry run
against the neutral site prints, per submission, the contract the funnel line carries."""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, REPRO, VENV_PY, run_stage  # noqa: E402

MODEL = "qwen35_4b_demo"      # the front-door model (models.front_door_keys()); its fits config is the unique pin of its fit checkpoint


def _fd6_overrides(root: Path) -> Path:
    """An overrides file in the FD 6 protocol (reconnaissance clip, one warm-up level, the LoRA architecture search): the protocol
    under which 03_hpo.sh has a recon phase at all — under the paper's defaults the clip is a SEARCHED axis and 03 says `recon: not applicable`."""
    p = root / "fd6_test.env"
    p.write_text("OVERRIDE_ID=fd6_test_v1\nOVERRIDE_INIT_SCHEME=random_lecun_1e-3\nOVERRIDE_LORA_ARCH_SEARCH=1\nOVERRIDE_LORA_RANK=''\nOVERRIDE_LORA_MODULE_SET=''\n"
                 "OVERRIDE_HPO_WARMUP_FRAC=0.15\nOVERRIDE_HPO_CLIP_MODE=reconnaissance\nOVERRIDE_HPO_CLIP_PERCENTILE=95\nOVERRIDE_HPO_TRIALS=16\n"
                 "OVERRIDE_HPO_RECENTRE_TRIALS=4\nOVERRIDE_REFUSAL_STEER_ALPHA=0.3\nOVERRIDE_EFFECTIVE_BATCH=64\n")
    return p


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    root = tmp_path_factory.mktemp("fd03"); site = root / "site"
    shutil.copytree(REPRO / "site.example", site)
    with open(site / "config.env", "a") as fh:
        fh.write(f"\nSISTEER_STATE_DIR={root / 'state'}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={root / 'registry'}\n")
    return site


def _contracts(stdout: str) -> dict:
    """unit -> (registers, kind, script) from the dry run's `registration contract:` lines."""
    out = {}
    for m in re.finditer(r"^registration contract: (\S+) registers=(\S+) \(kind (\S+), (\S+)\)$", stdout, re.M):
        out[m.group(1)] = (m.group(2), m.group(3), m.group(4))
    return out


def test_the_recon_submission_declares_none_and_the_trial_submission_does_not(site):
    # the recon exists only under the FD 6 protocol's overrides (under the paper's defaults the clip is a SEARCHED axis and 03 says
    # `recon: not applicable`), so one dry run takes an FD 6 overrides file and one the paper's defaults
    r = run_stage("03_hpo.sh", "--dry-run", "--object-class", "relu", "--phase", "all", "--model", MODEL,
                  "--overrides", str(_fd6_overrides(site.parent)), site=site, timeout=900)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    c = _contracts(r.stdout)
    recon = {u: v for u, v in c.items() if "recon" in u}
    assert recon and all(v == ("none", "fit", "scripts/sbatch/hpo_trial.sbatch") for v in recon.values()), (recon, r.stdout[-1500:])
    assert all(v[0] != "none" for u, v in c.items() if "recon" not in u), c                    # nothing else declares none
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("DRY-RUN: ssh") and "hpo_trial.sbatch" in ln]
    assert lines and any("HPO_JOB_MODE=recon" in ln for ln in lines), lines[:3]
    # the paper's defaults: no recon; the TRIAL submission of the same sbatch leaves the contract to the funnel (registry)
    r2 = run_stage("03_hpo.sh", "--dry-run", "--object-class", "relu", "--phase", "all", "--model", MODEL, site=site, timeout=900)
    assert r2.returncode == 0 and "recon: not applicable" in r2.stdout, r2.stdout[-2000:] + r2.stderr[-2000:]
    c2 = _contracts(r2.stdout)
    trials = {u: v for u, v in c2.items() if v[2] == "scripts/sbatch/hpo_trial.sbatch"}
    assert trials and all(v == ("derived", "fit", "scripts/sbatch/hpo_trial.sbatch") for v in trials.values()), (trials, r2.stdout[-1500:])
    assert not any("recon" in u for u in c2)
    assert all("HPO_JOB_MODE=recon" not in ln for ln in r2.stdout.splitlines() if ln.startswith("DRY-RUN: ssh"))


def test_the_source_marks_only_the_recon_call_site_none():
    src = (REPO / "reproduce" / "03_hpo.sh").read_text()
    calls = [ln for ln in src.splitlines() if re.match(r"\s*submit \"\$\(unit ", ln)]
    recon = [ln for ln in calls if "$(unit recon)" in ln]
    assert len(recon) == 1 and "--registers none" in recon[0] and 'HPO_JOB_MODE=recon' in recon[0], recon
    assert all("--registers" not in ln for ln in calls if "$(unit recon)" not in ln), [ln for ln in calls if "--registers" in ln]
    assert '--registers) registers="${2:?--registers needs registry|none}"; shift 2;;' in src
    assert '${registers:+--registers "$registers"} --wave-sha "$SHA"' in src
