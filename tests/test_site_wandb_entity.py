"""The W&B entity is a SITE value with no default: the readers outside the registry take it from the site
(``WANDB_ENTITY``, else the site config's ``SISTEER_WANDB_ENTITY``) and never name an account of their own.

* ``scripts/fst_figs.py``: the destination is ``--wandb-entity`` / ``--wandb-project``, else ``WANDB_*``, else
  ``SISTEER_WANDB_*``; W&B logging with either missing refuses by name before the analysis is even read.
* ``scripts/sbatch/sdfbase_fit_cell.sbatch``: the job logs offline under the site's entity; with none in the job
  environment W&B is off for the job (``WANDB_MODE=disabled``) and the job log says so.
* ``scripts/sbatch/hpo_trial.sbatch``: a trial's offline run is filed under the site's entity when the job env carries one.
Synthetic values only (``example-entity``); W&B is never contacted.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import fst_figs as ff  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPRO, write_site  # noqa: E402

FIT = REPO / "scripts" / "sbatch" / "sdfbase_fit_cell.sbatch"
HPO = REPO / "scripts" / "sbatch" / "hpo_trial.sbatch"
ENT = "example-entity"


@pytest.fixture
def no_site_wandb(monkeypatch):
    for n in ff.WANDB_ENTITY_ENV + ff.WANDB_PROJECT_ENV:
        monkeypatch.delenv(n, raising=False)


def test_fst_figs_refuses_wandb_without_an_entity_before_reading_anything(tmp_path, monkeypatch, no_site_wandb):
    logged = []
    monkeypatch.setattr(ff, "log_wandb", lambda *a, **k: logged.append(k))
    monkeypatch.setenv("WANDB_PROJECT", "example-project")
    with pytest.raises(SystemExit) as ei:
        ff.main(["--analysis", str(tmp_path / "absent.json"), "--figs-dir", str(tmp_path / "figs"), "--run", "r1"])
    msg = str(ei.value)
    assert "needs an entity:" in msg and "WANDB_ENTITY" in msg and "SISTEER_WANDB_ENTITY" in msg and "--no-wandb" in msg
    assert logged == [] and not (tmp_path / "figs").exists()


def test_fst_figs_destination_order_flag_then_wandb_env_then_site_config(no_site_wandb):
    with pytest.raises(SystemExit, match="entity and a project"):
        ff.wandb_destination(None, None, env={})
    assert ff.wandb_destination(None, None, env={"SISTEER_WANDB_ENTITY": ENT, "SISTEER_WANDB_PROJECT": "p-site"}) == (ENT, "p-site")
    env = {"WANDB_ENTITY": "e-env", "SISTEER_WANDB_ENTITY": ENT, "WANDB_PROJECT": "p-env", "SISTEER_WANDB_PROJECT": "p-site"}
    assert ff.wandb_destination(None, None, env=env) == ("e-env", "p-env")
    assert ff.wandb_destination("e-flag", "p-flag", env=env) == ("e-flag", "p-flag")
    with pytest.raises(SystemExit, match="needs a project"):
        ff.wandb_destination(ENT, "  ", env={"WANDB_PROJECT": ""})


def test_fst_figs_parser_carries_no_default_destination(no_site_wandb):
    args = ff.build_parser().parse_args(["--analysis", "a", "--figs-dir", "f", "--run", "r"])
    assert args.wandb_entity is None and args.wandb_project is None
    src = (REPO / "scripts" / "fst_figs.py").read_text()
    assert not re.search(r"--wandb-(entity|project)\", default=\"", src)


def _fit_wandb_block():
    """The fit job's W&B lines, from the entity export to the end of its on/off switch."""
    lines = FIT.read_text().splitlines()
    i = next(n for n, ln in enumerate(lines) if ln.startswith("export WANDB_ENTITY="))
    j = next(n for n in range(i, len(lines)) if lines[n] == "fi")
    return "\n".join(lines[i:j + 1])


def _run_block(env):
    script = 'log() { echo "LOG $*"; }\n' + _fit_wandb_block() + '\necho "MODE=$WANDB_MODE ENTITY=$WANDB_ENTITY PROJECT=$WANDB_PROJECT"\n'
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", **env}, timeout=30)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_fit_sbatch_takes_the_site_entity_and_turns_wandb_off_without_one():
    out = _run_block({"SISTEER_WANDB_ENTITY": ENT, "SISTEER_WANDB_PROJECT": "p-site"})
    assert f"MODE=offline ENTITY={ENT} PROJECT=p-site" in out and "LOG" not in out
    out = _run_block({"WANDB_ENTITY": "e-env", "SISTEER_WANDB_ENTITY": ENT})
    assert "MODE=offline ENTITY=e-env PROJECT=sisteer-fits" in out              # the neutral default (a fit config's own project still wins)
    out = _run_block({})
    assert "MODE=disabled ENTITY= " in out and "LOG W&B off: no WANDB_ENTITY / SISTEER_WANDB_ENTITY" in out


def _hpo_entity(env):
    """The entity a trial's run is filed under: the hpo sbatch's WANDB_ENTITY line run under `set -eu`."""
    line = next(ln for ln in HPO.read_text().splitlines() if "WANDB_ENTITY=" in ln and not ln.lstrip().startswith("#"))
    r = subprocess.run(["bash", "-euc", line + '\necho "ENTITY=${WANDB_ENTITY:-}"'], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", **env}, timeout=30)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_hpo_trial_files_its_offline_run_under_the_site_entity():
    assert _hpo_entity({"SISTEER_WANDB_ENTITY": ENT}) == f"ENTITY={ENT}"
    assert _hpo_entity({"WANDB_ENTITY": "e-env", "SISTEER_WANDB_ENTITY": ENT}) == "ENTITY=e-env"
    assert _hpo_entity({}) == "ENTITY="


def _job_env(tmp_path, extra_config: str) -> str:
    site = write_site(tmp_path, extra_config=extra_config)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SISTEER_")}
    env["SISTEER_STATE_DIR"] = str(tmp_path / "state")
    r = subprocess.run(["bash", "-c", f"source {REPRO}/lib/common.sh; source {REPRO}/lib/slurm.sh; REPRO_CONFIG={site}/config.env; "
                        "repro_load_config; slurm_job_env cluster_a"], capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_job_env_carries_the_site_wandb_values_only_when_the_site_names_an_entity(tmp_path):
    out = _job_env(tmp_path / "on", f"SISTEER_WANDB_ENTITY={ENT}\nSISTEER_WANDB_PROJECT=example-project\n")
    assert out.startswith("SISTEER_SHARED_ROOT=") and out.endswith(f" SISTEER_WANDB_ENTITY={ENT} SISTEER_WANDB_PROJECT=example-project"), out
    out = _job_env(tmp_path / "off", "")
    assert "WANDB" not in out and out.startswith("SISTEER_SHARED_ROOT="), out
