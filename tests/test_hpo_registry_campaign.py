"""03_hpo.sh's registry campaign: the resolved value is SISTEER_REGISTRY_CAMPAIGN, else REGISTRY_CAMPAIGN, else
sdfbase_full22 — the funnel's `two_si` only when a caller names it — and job_env hands THAT value to the trial / fit job on both branches, so
the job's emitter (registry_emission.campaign_from_env / train_two_si_objects: REGISTRY_CAMPAIGN) never files a paper-frame object under
two_si by default. Executed under bash (the rule: run the form, do not string-match it)."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
HPO = REPO / "reproduce" / "03_hpo.sh"


def _campaign_line() -> str:
    lines = [ln for ln in HPO.read_text().splitlines() if ln.startswith("CAMPAIGN=")]
    assert len(lines) == 1, lines
    return lines[0]


def _job_env_fn() -> str:
    src = HPO.read_text()
    return src[src.index("job_env(){"):src.index("submit(){")]


def _bash(script: str, env: dict[str, str]) -> str:
    base = {k: v for k, v in os.environ.items() if k not in ("REGISTRY_CAMPAIGN", "SISTEER_REGISTRY_CAMPAIGN")}
    r = subprocess.run(["bash", "-c", f"set -u\n{script}"], capture_output=True, text=True, env={**base, **env}, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.strip()


def test_the_campaign_resolves_site_then_env_then_sdfbase_full22_and_two_si_only_when_named():
    line = _campaign_line()
    run = lambda env: _bash(f'{line}\necho "$CAMPAIGN"', env)
    assert run({}) == "sdfbase_full22"
    assert run({"REGISTRY_CAMPAIGN": "alpha"}) == "alpha"
    assert run({"SISTEER_REGISTRY_CAMPAIGN": "beta", "REGISTRY_CAMPAIGN": "alpha"}) == "beta", "the site's value wins"
    assert run({"REGISTRY_CAMPAIGN": "two_si"}) == "two_si", "named explicitly it is allowed"
    assert ":-two_si" not in HPO.read_text(), "no two_si fallback anywhere in 03_hpo.sh"


def test_job_env_hands_the_resolved_campaign_to_the_job_on_both_branches():
    fn = _job_env_fn()
    stubs = ('slurm_job_env(){ echo "SISTEER_SHARED_ROOT=/s SISTEER_CLUSTER_LABEL=$1"; }\n'
             'cluster_var(){ echo "lab-$2"; }\ncluster_fit_env_name(){ echo venv; }\nCLUSTER=cluster_a; SHARED_ROOT=/s\n')
    line = _campaign_line()
    slurm = _bash(f"{stubs}{fn}\n{line}\nHAVE_SLURM_LIB=1; job_env", {})
    local = _bash(f"{stubs}{fn}\n{line}\nHAVE_SLURM_LIB=0; job_env", {})
    for out in (slurm, local):
        assert re.search(r"(^|\s)REGISTRY_CAMPAIGN=sdfbase_full22(\s|$)", out), out
        assert "two_si" not in out, out
    assert slurm.startswith("SISTEER_SHARED_ROOT=/s SISTEER_CLUSTER_LABEL=cluster_a "), "the Slurm branch keeps the library's env and appends"
    named = _bash(f"{stubs}{fn}\n{line}\nHAVE_SLURM_LIB=1; job_env", {"SISTEER_REGISTRY_CAMPAIGN": "two_si"})
    assert "REGISTRY_CAMPAIGN=two_si" in named, "named by the caller, the job files under it"
