"""The cluster job scripts hand the shared root to the driver they start.

Two kinds of driver resolve the cluster's shared root through the site, as ``sisteer.site.expand_path("$SHARED")``: the
job's SISTEER_SHARED_ROOT (or CLUSTER_SHARED_ROOT), else the site config's cluster block. The Petri and BoN drivers do it
when they are imported and refuse at entry when it does not resolve. The jailbreak drivers do it in ``_private_root``
when a run is given no ``--private-root``, and take the grading host's default when it does not resolve. A compute node
carries no site config, so the job script has to put the root into the driver's environment. It does that by taking SHARED
from the job's SISTEER_SHARED_ROOT, an inherited variable that srun passes on, or by exporting the root before the driver
starts. A job script that only assigns a shell variable leaves a Petri driver with the placeholder form and sends a
jailbreak run to the grading host's directory on the node's local disk.

For every job script under scripts/ or reproduce/ that starts such a driver (a jailbreak driver only when the command
gives no ``--private-root``), the test runs the script's own root lines (its ``set`` options, its SHARED assignment and
any export of the root, in their order) in bash from a home with no site config. It then imports the driver in a child
process of that shell, as srun would start it, and compares the root the driver resolved with the shell's SHARED. Without
a root in the job environment the script either refuses by name before the driver starts or exports a root of its own,
which the driver then resolves.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: a Petri or BoN driver: the module-level line by which it resolves the shared root when it is imported
_MODULE_ROOT = re.compile(r'^\s*SHARED = _site\.expand_path\("\$SHARED"\)\s*$', re.M)
#: a jailbreak driver: its `_private_root` function (up to the next top-level definition), which consults the site
_PRIVATE_ROOT_DEF = re.compile(r"^def _private_root\(.*?(?=^(?:def|class) |\Z)", re.M | re.S)
_SITE_SHARED = '_site.expand_path("$SHARED")'
#: a job-script line that starts a driver under scripts/ (`uv run --no-sync python scripts/x.py`, `"$VENV/bin/python" scripts/x.py`)
_STARTS_DRIVER = re.compile(r'(?:^|[\s/"])python[0-9.]*"?\s+(?:-\S+\s+)*scripts/(\w+)\.py\b')
#: the root lines a job script runs before its driver, at the top level of the script
_ROOT_LINE = re.compile(r"^(?:set\s+-\S+.*|(?:export\s+)?SHARED=.*|export\s+(?:SISTEER|CLUSTER)_SHARED_ROOT\b.*)$")

#: the drivers this check must find, by kind, and the job scripts that start them, so it never passes by finding nothing
EXPECTED_DRIVERS = {
    "bon_negprobe": "module", "petri_bon_fanout": "module", "petri_bon_pathA": "module", "petri_bon_prompted": "module",
    "petri_bon_prompted_remeasure": "module", "petri_bon_smoke_integration": "module", "petri_bon_speedup": "module",
    "petri_cache_isolation_gate": "module", "petri_families_prompted": "module",
    "petri_families_served_integration": "module", "jailbreak_run": "private_root", "build_safety_defense": "private_root",
}
EXPECTED_JOBS = {
    ("scripts/sbatch/bon_negprobe.sbatch", "bon_negprobe"),
    ("scripts/sbatch/petri_bon_fanout.sbatch", "petri_bon_fanout"),
    ("scripts/sbatch/petri_bon_pathA.sbatch", "petri_bon_pathA"),
    ("scripts/sbatch/petri_bon_prompted.sbatch", "petri_bon_prompted"),
    ("scripts/sbatch/petri_bon_smoke_integration.sbatch", "petri_bon_smoke_integration"),
    ("scripts/sbatch/petri_bon_speedup.sbatch", "petri_bon_speedup"),
    ("scripts/sbatch/petri_cache_isolation_gate.sbatch", "petri_cache_isolation_gate"),
    ("scripts/sbatch/petri_families_prompted.sbatch", "petri_families_prompted"),
    ("scripts/sbatch/petri_families_served_integration.sbatch", "petri_families_served_integration"),
    ("scripts/sbatch/qwen_wave.sbatch", "petri_families_served_integration"),
    ("scripts/pod_run_bon_smoke.sh", "petri_bon_smoke_integration"),
    ("scripts/sbatch/jailbreak.sbatch", "jailbreak_run"),
    ("scripts/sbatch/jailbreak_attacks.sbatch", "jailbreak_run"),
    ("scripts/sbatch/jailbreak_gcg.sbatch", "jailbreak_run"),
}

#: what the srun child does: import the driver as a module, never as __main__, and print the root it resolved
_CHILD = """\
import importlib.util, sys
spec = importlib.util.spec_from_file_location("driver_under_test", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
root = mod.SHARED if sys.argv[2] == "module" else mod._private_root({}, None)
print("DRIVER_ROOT=" + str(root))
"""


def _kind(text: str):
    if _MODULE_ROOT.search(text):
        return "module"
    m = _PRIVATE_ROOT_DEF.search(text)
    return "private_root" if m and _SITE_SHARED in m.group(0) else None


def _drivers() -> dict:
    """{driver: kind} for every script under scripts/ that resolves the shared root through the site."""
    out = {}
    for p in (REPO / "scripts").glob("*.py"):
        kind = _kind(p.read_text(encoding="utf-8"))
        if kind:
            out[p.stem] = kind
    return out


def _commands(path: Path):
    """(line index, the whole command with its continuation lines) for each non-comment line of ``path``."""
    lines = path.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("#"):
            continue
        cmd, j = line, i
        while cmd.rstrip().endswith("\\") and j + 1 < len(lines):
            j += 1
            cmd = cmd.rstrip()[:-1] + " " + lines[j]
        yield i, cmd


def _jobs() -> list:
    """(job script, driver, kind) for every command under scripts/ or reproduce/ that starts a driver resolving the root
    through the site; a jailbreak driver's command counts only when it gives no --private-root."""
    drivers = _drivers()
    found = set()
    for top in ("scripts", "reproduce"):
        for path in sorted((REPO / top).rglob("*")):
            if path.suffix not in (".sbatch", ".sh") or not path.is_file():
                continue
            for _, cmd in _commands(path):
                for name in _STARTS_DRIVER.findall(cmd):
                    kind = drivers.get(name)
                    if kind is None or (kind == "private_root" and "--private-root" in cmd):
                        continue
                    found.add((path.relative_to(REPO).as_posix(), name, kind))
    return sorted(found)


def _root_lines(job: str, driver: str) -> list:
    """The job script's top-level root lines before the first line that starts ``driver``, in their order."""
    out = []
    for line in (REPO / job).read_text(encoding="utf-8").splitlines():
        if not line.lstrip().startswith("#") and driver in _STARTS_DRIVER.findall(line):
            return out
        if _ROOT_LINE.match(line):
            out.append(line)
    raise AssertionError(f"{job} never starts scripts/{driver}.py")


def _run(job: str, driver: str, kind: str, tmp_path: Path, root) -> subprocess.CompletedProcess:
    home = tmp_path / "home"   # no .config/sisteer under it: a compute node's site, which is none
    home.mkdir(exist_ok=True)
    child = tmp_path / "child.py"
    child.write_text(_CHILD, encoding="utf-8")
    lines = _root_lines(job, driver)
    assert any(re.match(r"(?:export\s+)?(?:SHARED|SISTEER_SHARED_ROOT|CLUSTER_SHARED_ROOT)=", ln) for ln in lines), \
        f"{job} sets no shared root before it starts scripts/{driver}.py (its root lines: {lines})"
    script = "\n".join(lines + ['printf "JOB_SHARED=%s\\n" "${SHARED-}"', '"$PY" "$CHILD" "$DRIVER" "$KIND"']) + "\n"
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(home), "USER": "jobuser", "LOGNAME": "jobuser",
           "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(REPO / "src"),
           "PY": sys.executable, "CHILD": str(child), "DRIVER": str(REPO / "scripts" / f"{driver}.py"), "KIND": kind}
    if root is not None:
        env["SISTEER_SHARED_ROOT"] = root
    return subprocess.run(["bash", "-c", script], cwd=str(REPO), env=env, capture_output=True, text=True, timeout=300)


def _value(out: str, key: str):
    got = [ln.split("=", 1)[1] for ln in out.splitlines() if ln.startswith(key + "=")]
    return got[-1] if got else None


def _expected(job_root, kind: str):
    """What the driver must resolve for a job script whose SHARED is ``job_root``."""
    if job_root is None:
        return None
    return job_root if kind == "module" else str(Path(job_root) / "exp" / "jailbreak")


def test_the_check_finds_every_driver_and_job_script():
    drivers = _drivers()
    assert {d: drivers.get(d) for d in EXPECTED_DRIVERS} == EXPECTED_DRIVERS
    jobs = {(job, driver) for job, driver, _ in _jobs()}
    assert EXPECTED_JOBS <= jobs, sorted(EXPECTED_JOBS - jobs)


@pytest.mark.parametrize("job,driver,kind", _jobs(), ids=lambda v: Path(v).name)
def test_the_driver_resolves_the_job_scripts_root_with_no_site_config(job, driver, kind, tmp_path):
    root = tmp_path / "shared-root"
    root.mkdir()   # a jailbreak driver takes the root only when it names a directory
    r = _run(job, driver, kind, tmp_path, str(root))
    assert r.returncode == 0, f"{job}: rc {r.returncode}\n{r.stdout}\n{r.stderr}"
    job_root, driver_root = _value(r.stdout, "JOB_SHARED"), _value(r.stdout, "DRIVER_ROOT")
    assert driver_root == _expected(job_root, kind), (
        f"{job} starts scripts/{driver}.py, which resolved {driver_root!r} where the job script's SHARED is "
        f"{job_root!r}: the root must reach the driver's environment\n{r.stderr}")
    assert job_root == str(root), f"{job} does not take SHARED from the job's SISTEER_SHARED_ROOT (it has {job_root!r})"


@pytest.mark.parametrize("job,driver,kind", _jobs(), ids=lambda v: Path(v).name)
def test_without_a_root_the_job_script_refuses_by_name_or_hands_its_own(job, driver, kind, tmp_path):
    r = _run(job, driver, kind, tmp_path, None)
    driver_root = _value(r.stdout, "DRIVER_ROOT")
    if r.returncode != 0 and driver_root is None:
        assert "SISTEER_SHARED_ROOT" in r.stderr, f"{job} stopped without naming SISTEER_SHARED_ROOT:\n{r.stderr}"
        return
    assert r.returncode == 0, f"{job}: rc {r.returncode}\n{r.stdout}\n{r.stderr}"
    assert driver_root and "$" not in driver_root, f"{job}: the driver kept the placeholder form {driver_root!r}"
    assert driver_root == _expected(_value(r.stdout, "JOB_SHARED"), kind), r.stdout
