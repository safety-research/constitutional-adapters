"""The cluster's directory roots reach every cluster job through the job environment: the project directory
(CLUSTER_PROJECT_ROOT -> SISTEER_PROJECT_ROOT, `<root>/<dir>` and the per-experiment `<root>-<suffix>` directories), the
per-commit checkouts (CLUSTER_EVAL_REPO_ROOT -> SISTEER_EVAL_REPO_ROOT), the per-commit Petri checkouts
(CLUSTER_PETRI_REPO_ROOT -> SISTEER_PETRI_REPO_ROOT) and the cluster's clone of this repository (CLUSTER_REPO_MIRROR ->
SISTEER_REPO_MIRROR). Both faces of the front door (lib/slurm.sh slurm_job_env, submit_twin.job_env) carry the four from the
cluster's site block, `clusters.py --job-paths` prints them for a job submitted by hand, and the cluster job scripts refuse by
name, with one message per variable, when one is missing. Synthetic site (cluster_a / cluster_b); no real host or path."""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, REPRO, write_site  # noqa: E402

ROOTS = (("SISTEER_PROJECT_ROOT", "CLUSTER_PROJECT_ROOT"), ("SISTEER_EVAL_REPO_ROOT", "CLUSTER_EVAL_REPO_ROOT"),
         ("SISTEER_PETRI_REPO_ROOT", "CLUSTER_PETRI_REPO_ROOT"), ("SISTEER_REPO_MIRROR", "CLUSTER_REPO_MIRROR"))
REFUSAL = {
    "SISTEER_PROJECT_ROOT": "set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)",
    "SISTEER_EVAL_REPO_ROOT": "set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)",
    "SISTEER_PETRI_REPO_ROOT": "set SISTEER_PETRI_REPO_ROOT, the root of the per-commit Petri checkouts (the front door exports it from the site config)",
    "SISTEER_REPO_MIRROR": "set SISTEER_REPO_MIRROR, the cluster clone of this repository (the front door exports it from the site config)",
}


def _site(tmp_path: Path) -> Path:
    """Two clusters whose blocks name none of the project or Petri roots (the defaults apply)."""
    return write_site(tmp_path)


def _override_b(site: Path) -> None:
    """cluster_b names its project directory and Petri root itself."""
    blk = site / "clusters" / "cluster_b.env"
    blk.write_text(blk.read_text() + "CLUSTER_PROJECT_ROOT='$SHARED/proj'\nCLUSTER_PETRI_REPO_ROOT='$SHARED/petri'\n")


def _clusters(site: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(LIB / "clusters.py"), "--site-dir", str(site), *args],
                          capture_output=True, text=True, timeout=60)


def _block(site: Path, label: str) -> dict:
    out = {}
    for _env, key in ROOTS:
        g = _clusters(site, "--get", label, key)
        assert g.returncode == 0, g.stderr
        out[key] = g.stdout.strip()
    return out


def test_project_root_defaults_under_the_shared_root_and_a_block_overrides_it(tmp_path):
    site = _site(tmp_path)
    _override_b(site)
    a, b = _block(site, "cluster_a"), _block(site, "cluster_b")
    assert a["CLUSTER_PROJECT_ROOT"] == f"{tmp_path}/shared_cluster_a/sisteer"
    assert a["CLUSTER_PETRI_REPO_ROOT"] == f"{tmp_path}/shared_cluster_a/sisteer-petri-repo"
    assert b["CLUSTER_PROJECT_ROOT"] == f"{tmp_path}/shared_cluster_b/proj"
    assert b["CLUSTER_PETRI_REPO_ROOT"] == f"{tmp_path}/shared_cluster_b/petri"


def test_job_paths_prints_the_four_roots_as_exports_and_refuses_an_unknown_cluster(tmp_path):
    site = _site(tmp_path)
    _override_b(site)
    want = _block(site, "cluster_b")
    r = _clusters(site, "--job-paths", "cluster_b")
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == [f"export {env}={shlex.quote(want[key])}" for env, key in ROOTS]
    # the lines are what a hand submission evals before sbatch
    probe = subprocess.run(["bash", "-c", 'eval "$1"; printf "%s|%s|%s|%s" "$SISTEER_PROJECT_ROOT" "$SISTEER_EVAL_REPO_ROOT" '
                            '"$SISTEER_PETRI_REPO_ROOT" "$SISTEER_REPO_MIRROR"', "_", r.stdout], capture_output=True, text=True)
    assert probe.stdout == "|".join(want[key] for _env, key in ROOTS)
    bad = _clusters(site, "--job-paths", "cluster_z")
    assert bad.returncode == 3 and "cluster_z" in bad.stderr and bad.stdout == ""


def _shell_job_env(site: Path, label: str) -> dict:
    e = dict(os.environ, SISTEER_STATE_DIR=str(site.parent / "state"))
    r = subprocess.run(["bash", "-c", f"source {REPRO}/lib/common.sh; source {REPRO}/lib/slurm.sh; REPRO_CONFIG={site}/config.env; "
                        f"repro_load_config; slurm_job_env {label}"], capture_output=True, text=True, env=e, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    return dict(tok.split("=", 1) for tok in shlex.split(r.stdout.strip()))


def _python_job_env(site: Path, label: str) -> dict:
    probe = ("import json, sys; sys.path.insert(0, sys.argv[1]); import submit_twin as T; s = T.Site(sys.argv[2], True); "
             "print(json.dumps(T.Coordinator(s, 'm').job_env(sys.argv[3])))")
    r = subprocess.run([sys.executable, "-c", probe, str(LIB), str(site / "config.env"), label], capture_output=True, text=True,
                       env=dict(os.environ, SISTEER_STATE_DIR=str(site.parent / "state")), timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_both_faces_of_the_front_door_carry_the_four_roots_from_the_block(tmp_path):
    site = _site(tmp_path)
    _override_b(site)
    for label in ("cluster_a", "cluster_b"):
        want = {env: _block(site, label)[key] for env, key in ROOTS}
        shell, python = _shell_job_env(site, label), _python_job_env(site, label)
        assert {k: shell.get(k) for k in want} == want, (label, shell)
        assert {k: python.get(k) for k in want} == want, (label, python)


_REFUSE = re.compile(r"\$\{\{?(SISTEER_(?:PROJECT_ROOT|EVAL_REPO_ROOT|PETRI_REPO_ROOT|REPO_MIRROR)):\?([^}]*)\}")


def _job_scripts():
    for p in sorted((REPO / "scripts").rglob("*")):
        if p.is_file() and p.suffix in (".sh", ".sbatch", ".py", ".md"):
            yield p


def test_every_refusal_of_a_directory_root_says_the_one_message_for_its_variable():
    seen = {env: 0 for env in REFUSAL}
    wrong = []
    for p in _job_scripts():
        for m in _REFUSE.finditer(p.read_text(errors="replace")):
            seen[m.group(1)] += 1
            if m.group(2) != REFUSAL[m.group(1)]:
                wrong.append(f"{p.relative_to(REPO)}: {m.group(1)}: {m.group(2)!r}")
    assert wrong == []
    assert all(seen.values()), seen      # the scan found the scripts that read each root


def test_every_plain_use_of_a_directory_root_sits_in_a_script_that_refuses_or_tests_it():
    unguarded = []
    for p in _job_scripts():
        if p.suffix not in (".sh", ".sbatch"):
            continue
        code = "\n".join(ln for ln in p.read_text(errors="replace").splitlines() if not ln.lstrip().startswith("#"))
        for env in REFUSAL:
            plain = re.search(r"\$\{?" + env + r"(?![A-Za-z0-9_])(?!:[?-])", code)
            guarded = "${" + env + ":?" in code or "${" + env + ":-" in code
            if plain and not guarded:
                unguarded.append(f"{p.relative_to(REPO)}: {env}")
    assert unguarded == []
