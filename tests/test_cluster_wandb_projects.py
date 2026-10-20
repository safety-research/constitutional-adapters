"""The W&B projects of the hand-submitted cluster jobs: `scripts/measure_train_budget.py`'s budget figures and the AlphaSteer screen /
Llama gate sbatch. They named their projects as literals. A compute node has no site config, so the site's per-purpose keys
(SISTEER_WANDB_PROJECT_BUDGET / _ALPHASTEER) reach the job through the cluster's hf_site.env, which `reproduce/00_setup.sh --what
helpers` renders and the installed hf_env.sh exports (tests/test_helpers_parity.py covers those two). Unset: the budget figures log to
the neutral `sisteer-budget`, the AlphaSteer scripts leave W&B off (their own default)."""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import measure_train_budget as mtb  # noqa: E402

AS_SBATCH = ("scripts/sbatch/alphasteer_fit_f22.sbatch", "scripts/sbatch/alphasteer_llama_gate.sbatch")
W_AND_B_KEYS = ("SISTEER_WANDB_PROJECT_BUDGET", "SISTEER_WANDB_PROJECT_ALPHASTEER", "WANDB_PROJECT", "SISTEER_WANDB_PROJECT",
                "WANDB_DISABLED", "SISTEER_SITE_DIR")


@pytest.fixture
def no_site(tmp_path, monkeypatch):
    """No site config and no W&B key in the environment: a compute node's view."""
    empty = tmp_path / "config.env"
    empty.write_text("")
    monkeypatch.setenv("SISTEER_CONFIG", str(empty))
    for k in W_AND_B_KEYS:
        monkeypatch.delenv(k, raising=False)
    return empty


@pytest.fixture
def fake_wandb(monkeypatch):
    calls: dict = {"init": [], "log": [], "finish": 0}

    class Run:
        def log(self, d):
            calls["log"].append(d)

        def finish(self):
            calls["finish"] += 1

    def init(**kw):
        calls["init"].append(kw)
        if calls.get("fail"):
            raise RuntimeError("no network")
        return Run()

    mod = types.SimpleNamespace(init=init, Image=lambda p: ("image", p))
    monkeypatch.setitem(sys.modules, "wandb", mod)
    return calls


def test_the_budget_figures_project_comes_from_the_purpose_key_then_the_environment_then_the_neutral_name(no_site, fake_wandb, tmp_path,
                                                                                                         monkeypatch, capsys):
    assert mtb.WANDB_PURPOSE == "budget"
    paths = {"peak GB": tmp_path / "final_budget_peak_gb.png"}
    mtb._log_wandb_figures(paths)                                            # nothing set: the neutral name
    monkeypatch.setenv("SISTEER_WANDB_PROJECT", "site-proj")
    mtb._log_wandb_figures(paths)
    monkeypatch.setenv("WANDB_PROJECT", "job-proj")
    mtb._log_wandb_figures(paths)
    monkeypatch.setenv("SISTEER_WANDB_PROJECT_BUDGET", "budget-proj")      # the purpose key first (hf_env.sh exports it on a node)
    mtb._log_wandb_figures(paths)
    assert [c["project"] for c in fake_wandb["init"]] == ["sisteer-budget", "site-proj", "job-proj", "budget-proj"]
    assert all(c["job_type"] == "budget" for c in fake_wandb["init"]) and fake_wandb["finish"] == 4
    assert fake_wandb["log"][0] == {"budget/final_budget_peak_gb": ("image", str(paths["peak GB"]))}
    # W&B off on request; a failing init warns and never fails the emit
    monkeypatch.setenv("WANDB_DISABLED", "true")
    mtb._log_wandb_figures(paths)
    assert len(fake_wandb["init"]) == 4
    monkeypatch.delenv("WANDB_DISABLED")
    fake_wandb["fail"] = True
    mtb._log_wandb_figures(paths)
    assert "[budget] WARNING: wandb figure logging failed: no network" in capsys.readouterr().err
    src = (REPO / "scripts/measure_train_budget.py").read_text()
    assert not re.search(r"wandb\.init\(project=\"", src)                   # no literal project left


def _as_line(rel: str) -> str:
    lines = [line for line in (REPO / rel).read_text().splitlines() if line.startswith("AS_WANDB_PROJECT=")]
    assert len(lines) == 1, (rel, lines)
    return lines[0]


@pytest.mark.parametrize("rel", AS_SBATCH)
def test_the_alphasteer_sbatch_take_the_project_from_the_purpose_key_then_the_job_env_else_w_and_b_off(rel):
    text = (REPO / rel).read_text()
    assert "--wandb-project alphasteer" not in text and '--wandb-project "$AS_WANDB_PROJECT"' in text
    # read after s2_init, which sources hf_env.sh (the purpose key's carrier on a compute node)
    assert text.index("s2_init ") < text.index("AS_WANDB_PROJECT=") < text.index('--wandb-project "$AS_WANDB_PROJECT"')
    line = _as_line(rel)
    cases = [({}, ""),
             ({"SISTEER_WANDB_PROJECT": "site-proj"}, "site-proj"),
             ({"SISTEER_WANDB_PROJECT": "site-proj", "WANDB_PROJECT": "job-proj"}, "job-proj"),
             ({"SISTEER_WANDB_PROJECT": "site-proj", "WANDB_PROJECT": "job-proj", "SISTEER_WANDB_PROJECT_ALPHASTEER": "as-proj"}, "as-proj")]
    for env, want in cases:
        r = subprocess.run(["bash", "-c", f'set -u; {line}; printf "%s" "$AS_WANDB_PROJECT"'], capture_output=True, text=True,
                           env={"PATH": os.environ["PATH"], **env})
        assert r.returncode == 0 and r.stdout == want, (rel, env, r.stdout, r.stderr)


def test_a_rendered_hf_site_env_gives_the_hand_submitted_jobs_the_sites_projects(tmp_path):
    """End to end on a node's view (no site config): the installed hf_env.sh beside a rendered hf_site.env, then the AlphaSteer sbatch's
    project line and the budget figures' resolver in that job environment."""
    helpers = tmp_path / "shared" / "scripts"
    helpers.mkdir(parents=True)
    (helpers / "hf_env.sh").write_text((REPO / "reproduce/lib/hf_env.sh").read_text())
    (helpers / "hf_site.env").write_text(f"HF_SITE_HF_HOME={tmp_path / 'models'}\nHF_SITE_SHARED_ROOT={tmp_path / 'shared'}\n"
                                         "HF_SITE_WANDB_PROJECT_BUDGET=budget-proj\nHF_SITE_WANDB_PROJECT_ALPHASTEER=as-proj\n")
    home = tmp_path / "home"
    home.mkdir()
    py = ("import measure_train_budget as m; from sisteer.wandb_entity import wandb_project; print(wandb_project(m.WANDB_PURPOSE))")
    script = (f"set -euo pipefail; source {shlex.quote(str(helpers / 'hf_env.sh'))}; {_as_line(AS_SBATCH[0])}; "
              f'echo "$AS_WANDB_PROJECT"; {shlex.quote(sys.executable)} -c {shlex.quote(py)}')
    env = {"PATH": os.environ["PATH"], "USER": "nobody-here", "HOME": str(home),
           "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO / "scripts")]), "HF_HUB_OFFLINE": "1"}
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.split() == ["as-proj", "budget-proj"], r.stdout
