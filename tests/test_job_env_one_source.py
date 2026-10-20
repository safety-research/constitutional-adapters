"""A cluster job's environment is composed in ONE place: lib/submit_twin.py Coordinator.job_env, the env prefix of every
live sbatch line the funnel composes. The shell face, lib/slurm.sh slurm_job_env (which 03_hpo.sh's dry-run printer uses),
prints that same environment through `submit_twin.py job-env`, so a dry run shows exactly what a live job carries. When the
site names a W&B entity, its entity and project ride that environment, so a fit job logs under the site's entity instead of
running with W&B off. `job-env --export` prints the same values for a job submitted by hand.

A fit job a launcher script submits outside the funnel takes the W&B part of that environment alone (`job-env --wandb`, the same
function job_env calls): scripts/stage_llama_fits.sh prints it on each submission, with the two roots its job refuses without
(the cluster's shared root and project directory, as job_env names them), and its job files runs under the site's entity.
scripts/two_si_overnight.py renders the same keys into its fit scripts (tests/test_overnight.py).

The live side goes through the funnel to a stub sbatch that records the environment it was started with. The test strips
every SISTEER_ / WANDB_ / REGISTRY_ variable from the funnel's own environment, so a value the stub records can only have come
from the line's env prefix, as on a real cluster, where ssh carries nothing else.
Synthetic site (cluster_a / cluster_b, a fake Slurm under the test's temp dir); nothing contacts a cluster or W&B."""
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, REPRO, job_ids, make_registry, write_fake_slurm, write_site  # noqa: E402

SHA = "e" * 40
MODEL = "qwen35_4b_demo"
ENT, PROJ = "example-entity", "example-project"
WANDB_KEYS = ("SISTEER_WANDB_ENTITY", "SISTEER_WANDB_PROJECT")
#: what the funnel adds to a job's env per submission (the submission's own clock), and what the test itself hands the funnel
NOT_JOB_ENV = ("SISTEER_UNIT_SUBMITTED_UTC", "SISTEER_NO_WATCHER")


def _clean_env(**extra) -> dict:
    """The caller's environment without any site, job or W&B variable: nothing can reach a job except through its env prefix."""
    e = {k: v for k, v in os.environ.items() if not k.startswith(("SISTEER_", "WANDB_", "REGISTRY_", "SDFBASE_", "F22_", "CLUSTER_"))}
    e.update(extra)
    return e


def _pairs(text: str) -> dict:
    return dict(tok.split("=", 1) for tok in shlex.split(text.strip()))


class World:
    """One site whose cluster_a runs a fake Slurm (`CLUSTER_SSH=local:<dir>`) and a clean checkout dir at SHA."""

    def __init__(self, tmp: Path, extra_config: str = ""):
        self.tmp = tmp
        self.fake = tmp / "fake_cluster_a"
        self.bin = write_fake_slurm(tmp / "bin")
        self.site = write_site(tmp, ssh={"cluster_a": f"local:{self.fake}"}, extra_config=extra_config)
        self.config = self.site / "config.env"
        root = subprocess.run([sys.executable, str(LIB / "clusters.py"), "--site-dir", str(self.site), "--get", "cluster_a",
                               "CLUSTER_EVAL_REPO_ROOT"], capture_output=True, text=True, check=True).stdout.strip()
        co = Path(root) / SHA / "scripts" / "sbatch"
        co.mkdir(parents=True)
        (co / "hpo_trial.sbatch").write_text("#!/bin/bash\nf22_attempt_wait   # sisteer-attempt-wait-hook\n")
        make_registry(tmp / "registry")

    def cli(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(LIB / "submit_twin.py"), "--config", str(self.config), *args],
                              capture_output=True, text=True, env=_clean_env(), timeout=120)

    def shell_job_env(self, label: str = "cluster_a") -> str:
        """lib/slurm.sh slurm_job_env after repro_load_config: the function 03_hpo.sh's dry-run printer calls."""
        r = subprocess.run(["bash", "-c", f"source {REPRO}/lib/common.sh; source {REPRO}/lib/slurm.sh; REPRO_CONFIG={self.config}; "
                            f"repro_load_config; slurm_job_env {label}"], capture_output=True, text=True,
                           env=_clean_env(SISTEER_STATE_DIR=str(self.tmp / "state")), timeout=180)
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout

    def dry_run_03_env(self) -> dict:
        """The env prefix of 03_hpo.sh's dry-run sbatch line for the relu trials (the printer's own composition)."""
        r = subprocess.run(["bash", str(REPRO / "03_hpo.sh"), "--dry-run", "--object-class", "relu", "--phase", "all", "--model", MODEL,
                            "--config", str(self.config)], capture_output=True, text=True,
                           env=_clean_env(SISTEER_STATE_DIR=str(self.tmp / "state")), cwd=str(REPO), timeout=600)
        assert r.returncode == 0, (r.stdout + r.stderr)[-2500:]
        lines = [ln for ln in r.stdout.splitlines() if ln.startswith("DRY-RUN: ssh ") and "hpo_trial.sbatch" in ln and " sbatch " in ln]
        assert lines, r.stdout[-2500:]
        remote = shlex.split(lines[0])[3]
        toks = remote.split()
        return dict(t.split("=", 1) for t in toks[toks.index("&&") + 1:toks.index("sbatch")])

    def live_job_env(self) -> dict:
        """What the stub sbatch recorded for one live submission through the funnel."""
        r = subprocess.run([sys.executable, str(LIB / "submit_twin.py"), "--config", str(self.config), "--model", "m", "submit",
                            "--unit", "u", "--lwid", "m/jobenv/u", "--script", "scripts/sbatch/hpo_trial.sbatch", "--name", "jobenv-u",
                            "--args", "x", "--clusters", "cluster_a", "--kind", "target", "--wave-sha", SHA],
                           capture_output=True, text=True, env=_clean_env(PATH=f"{self.bin}:{os.environ['PATH']}", SISTEER_NO_WATCHER="1"),
                           timeout=180)
        assert r.returncode == 0 and "submitted cluster_a " in r.stdout, r.stdout + r.stderr
        jid = job_ids(self.fake)[-1]
        rec = dict(ln.split("=", 1) for ln in (self.fake / "jobs" / f"{jid}.env").read_text().splitlines() if "=" in ln)
        return {k: v for k, v in rec.items() if k not in NOT_JOB_ENV}


def test_a_live_submission_carries_the_site_wandb_values_and_exactly_the_dry_run_job_env(tmp_path):
    w = World(tmp_path, extra_config=f"SISTEER_WANDB_ENTITY={ENT}\nSISTEER_WANDB_PROJECT={PROJ}\n")
    live = w.live_job_env()
    # the finding: the funnel's live line carries the site's W&B destination (sdfbase_fit_cell.sbatch turns W&B off without it)
    assert {k: live.get(k) for k in WANDB_KEYS} == {"SISTEER_WANDB_ENTITY": ENT, "SISTEER_WANDB_PROJECT": PROJ}, live
    shell = w.shell_job_env()
    dry = _pairs(shell)
    assert {k: dry.get(k) for k in WANDB_KEYS} == {"SISTEER_WANDB_ENTITY": ENT, "SISTEER_WANDB_PROJECT": PROJ}, dry
    # ONE list: the dry-run face and the live job carry the same variables with the same values, key for key
    assert live == dry, {"live only": sorted(live.keys() - dry.keys()), "dry only": sorted(dry.keys() - live.keys()),
                         "differ": sorted(k for k in live.keys() & dry.keys() if live[k] != dry[k])}
    # the shell face prints the funnel's own `job-env`, byte for byte
    cli = w.cli("job-env", "--cluster", "cluster_a")
    assert cli.returncode == 0 and cli.stdout.strip() == shell.strip(), cli.stdout + cli.stderr
    # 03_hpo.sh's dry-run line (its printer composes the prefix from slurm_job_env, then states the stage's campaign) shows them all
    printed = w.dry_run_03_env()
    assert {k: printed.get(k) for k in dry} == dry, {k: (printed.get(k), v) for k, v in dry.items() if printed.get(k) != v}


def test_no_wandb_value_rides_when_the_site_names_no_entity(tmp_path):
    for sub, extra in (("none", ""), ("project_only", f"SISTEER_WANDB_PROJECT={PROJ}\n"), ("empty", "SISTEER_WANDB_ENTITY=''\n")):
        w = World(tmp_path / sub, extra_config=extra)
        live, dry = w.live_job_env(), _pairs(w.shell_job_env())
        assert not any(k in live or k in dry for k in WANDB_KEYS), (sub, live, dry)
        assert live == dry, sub


def test_job_env_reads_the_site_alone_creates_nothing_and_refuses_an_unknown_cluster(tmp_path):
    w = World(tmp_path, extra_config=f"SISTEER_WANDB_ENTITY={ENT}\nSISTEER_WANDB_PROJECT={PROJ}\n")
    state = tmp_path / "state"
    assert not state.exists()
    r = w.cli("job-env", "--cluster", "cluster_b")                 # no --model: the job env is the site's, not a model's
    assert r.returncode == 0, r.stderr
    env = _pairs(r.stdout)
    assert env["SISTEER_CLUSTER_LABEL"] == "cluster_b" and env["SISTEER_SHARED_ROOT"] == str(tmp_path / "shared_cluster_b")
    assert env["SISTEER_WANDB_ENTITY"] == ENT and env["SISTEER_WANDB_PROJECT"] == PROJ
    assert not state.exists()                                       # nothing written: read as a dry run reads the site
    # --export: the lines a job submitted by hand evals before sbatch, carrying the same values
    x = w.cli("job-env", "--cluster", "cluster_b", "--export")
    assert x.returncode == 0 and all(ln.startswith("export ") for ln in x.stdout.splitlines()), x.stdout
    names = " ".join(f'"${{{k}}}"' for k in env)
    probe = subprocess.run(["bash", "-c", f'eval "$1"; printf "%s\\n" {names}', "_", x.stdout], capture_output=True, text=True,
                           env=_clean_env(), timeout=30)
    assert probe.stdout.splitlines() == list(env.values())
    bad = w.cli("job-env", "--cluster", "cluster_z")
    assert bad.returncode == 3 and "cluster_z" in bad.stderr and bad.stdout == ""
    # every other command still needs a model
    nomodel = w.cli("binding", "--lwid", "m/jobenv/u")
    assert nomodel.returncode == 2 and "--model" in nomodel.stderr


def _function_body(path: Path, name: str) -> str:
    lines = path.read_text().splitlines()
    i = next(n for n, ln in enumerate(lines) if re.match(rf"{name}\(\)\s*\{{", ln))
    j = next(n for n in range(i, len(lines)) if lines[n] == "}")
    return "\n".join(lines[i:j + 1])


def test_the_shell_face_keeps_no_list_of_its_own():
    body = _function_body(REPRO / "lib" / "slurm.sh", "slurm_job_env")
    assert "job-env" in body and "submit_twin.py" in body, body
    assert not re.search(r"\b(SISTEER|REGISTRY|SDFBASE)_[A-Z_]+=", body), body


def test_the_hand_submission_roots_agree_with_the_job_env(tmp_path):
    """clusters.py --job-paths prints the cluster's directory roots for a job submitted by hand; each value is the job env's."""
    helper = subprocess.run([sys.executable, str(LIB / "clusters.py"), "--help"], capture_output=True, text=True, timeout=60)
    if "--job-paths" not in helper.stdout:
        pytest.skip("this checkout's clusters.py has no --job-paths helper")
    w = World(tmp_path)
    r = subprocess.run([sys.executable, str(LIB / "clusters.py"), "--site-dir", str(w.site), "--job-paths", "cluster_b"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    roots = dict(ln[len("export "):].split("=", 1) for ln in r.stdout.splitlines())
    env = _pairs(w.cli("job-env", "--cluster", "cluster_b").stdout)
    assert roots and {k: shlex.split(v)[0] for k, v in roots.items()} == {k: env.get(k) for k in roots}


def test_job_env_wandb_prints_the_wandb_part_of_the_job_env_alone(tmp_path):
    """`job-env --wandb`: the site's W&B destination alone (site_wandb_env, the function job_env takes it from), for a fit job a
    launcher script submits outside the funnel. Empty when the site names no entity; it needs no cluster."""
    w = World(tmp_path / "e", extra_config=f"SISTEER_WANDB_ENTITY={ENT}\nSISTEER_WANDB_PROJECT={PROJ}\n")
    r = w.cli("job-env", "--wandb")
    assert r.returncode == 0, r.stderr
    full = _pairs(w.cli("job-env", "--cluster", "cluster_a").stdout)
    assert _pairs(r.stdout) == {k: full[k] for k in WANDB_KEYS} == {"SISTEER_WANDB_ENTITY": ENT, "SISTEER_WANDB_PROJECT": PROJ}
    x = w.cli("job-env", "--wandb", "--export")
    assert x.returncode == 0 and x.stdout.splitlines() == [f"export SISTEER_WANDB_ENTITY={ENT}", f"export SISTEER_WANDB_PROJECT={PROJ}"]
    for sub, extra in (("none", ""), ("project_only", f"SISTEER_WANDB_PROJECT={PROJ}\n")):
        r = World(tmp_path / sub, extra_config=extra).cli("job-env", "--wandb")
        assert r.returncode == 0 and r.stdout.strip() == "", (sub, r.stdout, r.stderr)
    both = w.cli("job-env", "--wandb", "--cluster", "cluster_a")      # one or the other
    assert both.returncode == 2 and "not allowed with" in both.stderr


LLAMA_STAGER = REPO / "scripts" / "stage_llama_fits.sh"
LLAMA_JOB = REPO / "scripts" / "sbatch" / "twosi_fits_llama.sbatch"
#: the roots the llama fit job refuses without; the launcher reads both from the cluster's block, as the front door's job env does
LLAMA_JOB_ROOTS = ("SISTEER_SHARED_ROOT", "SISTEER_PROJECT_ROOT")


def _llama_job_requires() -> set:
    """Every setting the llama fit job refuses by name without (`${SISTEER_X:?...}` in a code line)."""
    code = [ln for ln in LLAMA_JOB.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    return {m for ln in code for m in re.findall(r"\$\{(SISTEER_[A-Z0-9_]+):\?", ln)}


def _llama_stager_world(tmp: Path):
    """A clean git checkout holding the llama fit launcher and the funnel library (the launcher refuses a dirty tree and prints
    its lines from HEAD), with src/ beside reproduce/ as in the repository (the site reader loads src/sisteer/site.py by path
    from there), plus stub ssh / rsync / mirror tool that record their calls and touch nothing."""
    import shutil
    repo = tmp / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy2(LLAMA_STAGER, repo / "scripts" / LLAMA_STAGER.name)
    shutil.copytree(LIB, repo / "reproduce" / "lib", ignore=shutil.ignore_patterns("__pycache__"))
    (repo / "src").symlink_to(REPO / "src", target_is_directory=True)
    (repo / "configs").mkdir()
    for f in ("two_si_llama_fits.yaml", "two_si_llama_fits_dp3.yaml"):
        (repo / "configs" / f).write_text("model: example\n")
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "core.hooksPath=/dev/null"]
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-qm", "init"], check=True)
    home, bindir, calls = tmp / "home", tmp / "bin", tmp / "calls.log"
    (home / "scripts").mkdir(parents=True); bindir.mkdir()
    stub = f'#!/bin/bash\necho "$(basename "$0") $*" >> {calls}\n'
    (home / "scripts" / "mirror-repos.sh").write_text(stub + "exit 0\n")
    (bindir / "rsync").write_text(stub + "exit 0\n")
    (bindir / "ssh").write_text(stub + 'case "$2" in echo*) echo /stub/shared;; esac\nexit 0\n')
    for f in (home / "scripts" / "mirror-repos.sh", bindir / "rsync", bindir / "ssh"):
        f.chmod(0o755)
    (bindir / "python3").symlink_to(sys.executable)
    return repo, home, bindir, calls


def _llama_submissions(tmp: Path, config: Path | None) -> tuple[subprocess.CompletedProcess, list, Path]:
    repo, home, bindir, calls = _llama_stager_world(tmp)
    env = _clean_env(PATH=f"{bindir}:{os.environ['PATH']}", HOME=str(home), HOST="login-a")
    if config is not None:
        env["SISTEER_CONFIG"] = str(config)
    r = subprocess.run(["bash", str(repo / "scripts" / LLAMA_STAGER.name)], capture_output=True, text=True, env=env, timeout=180)
    subs = []
    for ln in r.stdout.splitlines():
        if ln.startswith("ssh login-a ") and LLAMA_JOB.name in ln:
            argv = shlex.split(ln)
            assert len(argv) == 3, ln                                   # ssh, the host, ONE remote command word
            toks = shlex.split(argv[2])
            i = toks.index("sbatch")
            subs.append((dict(t.split("=", 1) for t in toks[:i]), toks[i:]))
    return r, subs, calls


def test_the_llama_fit_launcher_prints_the_site_wandb_destination_on_each_submission(tmp_path):
    """scripts/stage_llama_fits.sh prints the submissions of the llama fit waves. Their job loads configs whose wandb_project is
    $WANDB_PROJECT, so each printed line carries the site's W&B destination, the same keys and values as the front door's job env
    (`job-env --wandb`); without them the fits run with W&B off. The roots the job refuses without ride the line too, the
    cluster's shared root and project directory with the front door's values. The project here holds a space, so the value is
    quoted inside the one remote command word ssh is given."""
    proj = "example project"
    w = World(tmp_path / "site_e", extra_config=f"SISTEER_WANDB_ENTITY={ENT}\nSISTEER_WANDB_PROJECT='{proj}'\n")
    front = _pairs(w.cli("job-env", "--cluster", "cluster_a").stdout)
    r, subs, _ = _llama_submissions(tmp_path / "e", w.config)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2500:]
    assert len(subs) == 2, r.stdout[-2500:]                             # the response wave and the token_faithful wave
    for env, cmd in subs:
        assert {k: env.get(k) for k in WANDB_KEYS} == {k: front[k] for k in WANDB_KEYS} == {
            "SISTEER_WANDB_ENTITY": ENT, "SISTEER_WANDB_PROJECT": proj}, env
        assert {k: env.get(k) for k in LLAMA_JOB_ROOTS} == {k: front[k] for k in LLAMA_JOB_ROOTS}, env
        assert _llama_job_requires() <= set(env), sorted(_llama_job_requires() - set(env))     # the job gets past every refusal
        assert cmd[-1].startswith(front["SISTEER_PROJECT_ROOT"] + "/checkouts/") and cmd[-1].endswith(f"/scripts/sbatch/{LLAMA_JOB.name}"), cmd
        # this launcher's job takes the W&B part of the front door's env and two of its roots, none of its registry settings
        assert not any(k.startswith(("REGISTRY_", "SISTEER_REGISTRY_")) for k in env), env
    # a site with no entity: the lines carry the job's two roots and nothing else
    bare = World(tmp_path / "site_n", extra_config=f"SISTEER_WANDB_PROJECT={PROJ}\n")
    bfront = _pairs(bare.cli("job-env", "--cluster", "cluster_a").stdout)
    r, subs, _ = _llama_submissions(tmp_path / "n", bare.config)
    assert r.returncode == 0 and len(subs) == 2, (r.stdout + r.stderr)[-2500:]
    assert all(env == {k: bfront[k] for k in LLAMA_JOB_ROOTS} for env, _ in subs), subs


def test_the_llama_fit_launcher_refuses_without_the_site_config_before_touching_anything(tmp_path):
    r, subs, calls = _llama_submissions(tmp_path, None)
    assert r.returncode != 0 and "SISTEER_CONFIG" in r.stderr and subs == [], r.stdout + r.stderr
    assert not calls.exists() or calls.read_text() == "", calls.read_text()    # no mirror, ssh or rsync ran


def _llama_job_entity(env: dict) -> str:
    """The entity the llama fit job files its offline runs under: its WANDB_ENTITY line run under `set -eu`."""
    lines = [ln for ln in LLAMA_JOB.read_text().splitlines() if "WANDB_ENTITY=" in ln and not ln.lstrip().startswith("#")]
    assert len(lines) == 1, f"the llama fit job maps no site entity onto WANDB_ENTITY: {lines}"
    r = subprocess.run(["bash", "-euc", lines[0] + '\necho "ENTITY=${WANDB_ENTITY:-}"'], capture_output=True, text=True,
                       env={"PATH": "/usr/bin:/bin", **env}, timeout=30)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_the_llama_fit_job_files_its_runs_under_the_site_entity():
    assert _llama_job_entity({"SISTEER_WANDB_ENTITY": ENT}) == f"ENTITY={ENT}"
    assert _llama_job_entity({"WANDB_ENTITY": "e-env", "SISTEER_WANDB_ENTITY": ENT}) == "ENTITY=e-env"
    assert _llama_job_entity({}) == "ENTITY="
