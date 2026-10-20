"""sdfbase_fit_env.sbatch's host knobs (qwen-base-seed-replicates O7, 2026-09-16):
the interpreter (positional 2 > SDFBASE_PY > the shared 3.12.11 > the highest
cpython-3.12.* under $SHARED/.uv-pythons), CUDA_HOME from the environment with
nvcc asserted only for a source build, and SDFBASE_KERNEL_SOURCE=source|wheel
(`wheel` = the same two pinned kernel versions from the prebuilt cu128 wheels in
$SDFBASE_WHEELS). Every knob defaults to today's cluster A invocation: the dry
run below records the exact `uv` argv sequence build_env issues under a stubbed
`uv` and pins the default to the sequence the script issued before the knobs
existed. Cluster B's three measured blockers (w4_prep §5b: no 3.12.11, no CUDA
12.8 toolkit, nvcc 13 refusing the source builds) are each exercised."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

from test_sdfbase_slurm_sbatch import _extract_function, _hard_rules

REPO = Path(__file__).resolve().parents[1]
ENV = REPO / "scripts" / "sbatch" / "sdfbase_fit_env.sbatch"
DEFAULT_PY_REL = ".uv-pythons/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12"
CC1D_WHL = "causal_conv1d-1.6.2.post1-cp312-cp312-linux_x86_64.whl"
MAMBA_WHL = "mamba_ssm-2.3.2.post1-cp312-cp312-linux_x86_64.whl"

UV_STUB = """#!/bin/bash
printf '%s\\n' "uv $* [CC1D=${CAUSAL_CONV1D_FORCE_BUILD:-} MAMBA=${MAMBA_FORCE_BUILD:-}]" >> "$STUB_LOG"
case "$1" in
  venv) d="${@: -1}"; mkdir -p "$d/bin"; cp "$STUB_PY" "$d/bin/python"; chmod +x "$d/bin/python" ;;
  pip) [ "$2" = freeze ] && echo "stub==0.0" ;;
esac
exit 0
"""
PY_STUB = """#!/bin/bash
printf '%s\\n' "python $*" >> "$STUB_LOG"
[ "${1:-}" = "-" ] && cat >/dev/null
exit 0
"""


def _knob_block(txt: str) -> str:
    """The script's own knob lines: from the env-dir NAME through the kernel-source
    case (everything between the SHARED detection and MAX_JOBS)."""
    start = txt.index('export SDFBASE_FIT_ENV_NAME=')
    end = txt.index('export MAX_JOBS=')
    return txt[start:end]


def _clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if k not in ("CUDA_HOME", "SDFBASE_KERNEL_SOURCE", "SDFBASE_WHEELS", "SDFBASE_PY",
                        "SDFBASE_FIT_ENV_NAME", "SDFBASE_FIT_ENV", "SDFBASE_BUILD_DIR",
                        "CAUSAL_CONV1D_FORCE_BUILD", "MAMBA_FORCE_BUILD")}
    env.update(extra)
    return env


def _exe(path: Path, body: str = "#!/bin/bash\nexit 0\n") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    path.chmod(0o755)
    return path


def _resolve(shared: Path, *positional: str, **env: str) -> subprocess.CompletedProcess:
    """Run the knob block under bash with SHARED = a fixture dir; stdout is the resolved
    SDFBASE_PY|SDFBASE_PY_FROM|CUDA_HOME|SDFBASE_KERNEL_SOURCE|SDFBASE_WHEELS."""
    prog = (f'SHARED={shared}\n{_knob_block(ENV.read_text())}\n'
            'printf "%s|%s|%s|%s|%s" "$SDFBASE_PY" "$SDFBASE_PY_FROM" "$CUDA_HOME" '
            '"$SDFBASE_KERNEL_SOURCE" "$SDFBASE_WHEELS"')
    return subprocess.run(["bash", "-c", prog, "_", *positional], capture_output=True,
                          text=True, env=_clean_env(**env))


def test_parses_keeps_hygiene_and_todays_default_strings():
    subprocess.run(["bash", "-n", str(ENV)], check=True)
    txt = ENV.read_text()
    _hard_rules(txt)
    assert not re.search(r"^#SBATCH --gres", txt, re.M)
    # the knobs' defaults ARE today's values, spelled the same
    assert 'SDFBASE_PY_DEFAULT="$SHARED/.uv-pythons/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12"' in txt
    assert 'export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"' in txt
    assert 'export SDFBASE_KERNEL_SOURCE="${SDFBASE_KERNEL_SOURCE:-source}"' in txt
    assert 'export SDFBASE_WHEELS="${SDFBASE_WHEELS:-$SHARED/wheels}"' in txt
    # the interpreter fallback never installs a python and never touches the
    # minor-version link peers resolve through
    code = "\n".join(l for l in txt.splitlines() if not l.lstrip().startswith("#"))
    assert "uv python install" not in code and not re.search(r"^\s*ln ", code, re.M)
    assert 'cpython-3.12.*-linux-x86_64-gnu/bin/python3.12' in txt
    # wheel mode: same two pins, offline, deps already resolved by steps 1-2
    for s in ('uvpi --no-index --find-links "$SDFBASE_WHEELS" --no-deps causal-conv1d==1.6.2.post1',
              'uvpi --no-index --find-links "$SDFBASE_WHEELS" --no-deps mamba-ssm==2.3.2.post1',
              f'CC1D_WHL="$SDFBASE_WHEELS/{CC1D_WHL}"', f'MAMBA_WHL="$SDFBASE_WHEELS/{MAMBA_WHL}"',
              'want = {"causal-conv1d": "1.6.2.post1", "mamba-ssm": "2.3.2.post1"}'):
        assert s in txt, s


def test_interpreter_resolution_order(tmp_path):
    shared = tmp_path / "S"
    default = _exe(shared / DEFAULT_PY_REL)
    # today's cluster A: the shared 3.12.11 exists -> that path, nothing else consulted
    r = _resolve(shared)
    assert r.returncode == 0, r.stderr
    py, frm, cuda, ks, wheels = r.stdout.split("|")
    assert (py, frm, cuda, ks, wheels) == (str(default), "shared default", "/usr/local/cuda-12.8",
                                           "source", f"{shared}/wheels")
    # SDFBASE_PY (implicit export) beats the default; positional 2 beats both
    r = _resolve(shared, SDFBASE_PY="/opt/py/bin/python3.12")
    assert r.stdout.split("|")[:2] == ["/opt/py/bin/python3.12", "SDFBASE_PY (environment)"]
    r = _resolve(shared, "sdfbase-fit-tf515", "/pos/bin/python3.12", SDFBASE_PY="/opt/py/bin/python3.12")
    assert r.stdout.split("|")[:2] == ["/pos/bin/python3.12", "positional 2"]


def test_interpreter_fallback_is_the_highest_patch_never_the_minor_link(tmp_path):
    """Cluster B: no 3.12.11; .uv-pythons holds 3.12.13 and the uv minor-version link.
    The fallback picks the highest patch by version order (13 > 9 > 4, not lexical) and
    never the link (which peers' strength-dose-fit resolves through)."""
    shared = tmp_path / "S"
    for patch in ("9", "13", "4"):
        _exe(shared / f".uv-pythons/cpython-3.12.{patch}-linux-x86_64-gnu/bin/python3.12")
    (shared / ".uv-pythons/cpython-3.12-linux-x86_64-gnu").symlink_to(
        shared / ".uv-pythons/cpython-3.12.13-linux-x86_64-gnu")
    r = _resolve(shared)
    assert r.returncode == 0, r.stderr
    py, frm = r.stdout.split("|")[:2]
    assert py == str(shared / ".uv-pythons/cpython-3.12.13-linux-x86_64-gnu/bin/python3.12")
    assert frm.startswith("highest cpython-3.12.* under") and "shared default absent" in frm
    # the link is still what it was: never re-pointed
    assert os.readlink(shared / ".uv-pythons/cpython-3.12-linux-x86_64-gnu") == \
        str(shared / ".uv-pythons/cpython-3.12.13-linux-x86_64-gnu")
    # nothing at all -> empty (the preflight then refuses; see the dry run)
    r = _resolve(tmp_path / "empty")
    assert r.returncode == 0 and r.stdout.split("|")[0] == ""


def test_cuda_home_and_kernel_source_knobs(tmp_path):
    shared = tmp_path / "S"
    _exe(shared / DEFAULT_PY_REL)
    r = _resolve(shared, CUDA_HOME="/usr/local/cuda")
    assert r.stdout.split("|")[2] == "/usr/local/cuda"
    r = _resolve(shared, SDFBASE_KERNEL_SOURCE="wheel", SDFBASE_WHEELS="/elsewhere/wheels")
    assert r.stdout.split("|")[3:] == ["wheel", "/elsewhere/wheels"]
    r = _resolve(shared, SDFBASE_KERNEL_SOURCE="carried")
    assert r.returncode == 3 and "FATAL SDFBASE_KERNEL_SOURCE must be source|wheel" in r.stdout


# --------------------------------------------------------------------------- #
# dry run: the script's own knob block + its own build_env under a stubbed uv   #
# --------------------------------------------------------------------------- #

def _dry_run(tmp_path: Path, *, nvcc: bool, wheels: tuple[str, ...] = (), default_py: bool = True,
             positional: tuple[str, ...] = (), **env: str):
    shared = tmp_path / "S"
    if default_py:
        _exe(shared / DEFAULT_PY_REL)
    name = positional[0] if positional else "sdfbase-fit"
    (shared / "envs" / f"{name}.build").mkdir(parents=True)
    (shared / "envs" / f"{name}.build" / "requirements.txt").write_text("torch==2.11.0+cu128\n")
    cuda = tmp_path / "cuda"
    if nvcc:
        _exe(cuda / "bin" / "nvcc")
    wdir = tmp_path / "wheels"
    wdir.mkdir()
    for w in wheels:
        (wdir / w).write_bytes(b"PK\x05\x06" + bytes(18))
    binp = tmp_path / "bin"
    _exe(binp / "uv", UV_STUB)
    stub_py = _exe(tmp_path / "python.stub", PY_STUB)
    log = tmp_path / "stub.log"
    txt = ENV.read_text()
    prog = (f'SHARED={shared}\n{_knob_block(txt)}\nexport MAX_JOBS=8\n'
            f'{_extract_function(txt, "build_env")}\nbuild_env')
    r = subprocess.run(["bash", "-c", prog, "_", *positional], capture_output=True, text=True,
                       env=_clean_env(PATH=f"{binp}:{os.environ['PATH']}", STUB_LOG=str(log),
                                      STUB_PY=str(stub_py), CUDA_HOME=str(cuda),
                                      SDFBASE_WHEELS=str(wdir), **env))
    lines = log.read_text().splitlines() if log.exists() else []
    return r, [l for l in lines if l.startswith("uv ")], [l for l in lines if l.startswith("python ")], shared


def _todays_sequence(shared: Path, name: str = "sdfbase-fit") -> list[str]:
    py = shared / DEFAULT_PY_REL
    venv = shared / "envs" / name
    pb = venv / "bin" / "python"
    build = shared / "envs" / f"{name}.build"
    idle = "[CC1D= MAMBA=]"
    return [
        f"uv venv --python {py} {venv} {idle}",
        f"uv pip install --no-config --python {pb} -r {build}/requirements.txt "
        f"--extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match {idle}",
        f"uv pip install --no-config --python {pb} triton==3.7.1 fla-core==0.5.2 flash-linear-attention==0.5.2 ninja==1.13.0 {idle}",
        f"uv pip install --no-config --python {pb} --no-build-isolation causal-conv1d==1.6.2.post1 [CC1D=TRUE MAMBA=]",
        f"uv pip install --no-config --python {pb} --no-deps triton==3.7.1 {idle}",
        f"uv pip install --no-config --python {pb} --no-build-isolation --no-deps mamba-ssm==2.3.2.post1 [CC1D= MAMBA=TRUE]",
        f"uv pip install --no-config --python {pb} --no-deps triton==3.7.1 {idle}",
        f"uv pip freeze --python {pb} {idle}",
    ]


def test_default_dry_run_is_todays_cluster_a_sequence(tmp_path):
    """No knob set, nvcc present: the uv calls build_env issues are exactly the ones the
    script issued before the knobs existed (forced source builds, the FORCE_BUILD env on
    the two kernel installs only), each install now with --no-config (inert from the
    documented submission directory; it keeps a submission from inside a checkout from
    reading the project's [tool.uv] table), three python invocations, BUILD-COMPLETE written."""
    r, uv, py, shared = _dry_run(tmp_path, nvcc=True)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert uv == _todays_sequence(shared)
    assert all(l.startswith("uv pip install --no-config ") for l in uv if l.startswith("uv pip install"))
    assert len(py) == 3 and py[0].startswith("python -c import triton") and py[2] == "python -"
    assert "interpreter from: shared default; CUDA_HOME=" in r.stdout
    assert (shared / "envs" / "sdfbase-fit" / "BUILD-COMPLETE").read_text().split()[1]
    assert r.stdout.rstrip().splitlines()[-1].startswith("[env] BUILD-COMPLETE ")
    # the name knob still routes the build (O6)
    r, uv, _, shared = _dry_run(tmp_path / "b", nvcc=True, positional=("sdfbase-fit-tf515",))
    assert r.returncode == 0 and uv == _todays_sequence(shared, "sdfbase-fit-tf515")


def test_source_mode_still_needs_nvcc_and_the_interpreter(tmp_path):
    r, uv, _, _ = _dry_run(tmp_path, nvcc=False)
    assert r.returncode == 3 and "FATAL nvcc missing at" in r.stdout and uv == []
    r, uv, _, _ = _dry_run(tmp_path / "b", nvcc=True, default_py=False)
    assert r.returncode == 3 and "FATAL shared interpreter missing:" in r.stdout and uv == []
    assert "highest cpython-3.12.* under" in r.stdout    # the log names the failed source


def test_wheel_mode_installs_the_same_pins_offline_without_nvcc(tmp_path):
    """Cluster B: CUDA 13 toolkit (no nvcc 12.8), 3.12.13 only, SDFBASE_KERNEL_SOURCE=wheel.
    Steps 1-2 and 5-6 are today's; steps 3-4 become --no-index --find-links --no-deps of
    the SAME two pins with no FORCE_BUILD in the environment; the wheels' sha256 and the
    kernel-version check are added, everything else is unchanged."""
    r, uv, py, shared = _dry_run(tmp_path, nvcc=False, wheels=(CC1D_WHL, MAMBA_WHL), default_py=False,
                                 SDFBASE_KERNEL_SOURCE="wheel",
                                 SDFBASE_PY=str(_exe(tmp_path / "py313" / "bin" / "python3.12")))
    assert r.returncode == 0, (r.stdout, r.stderr)
    today = _todays_sequence(shared)
    wdir = tmp_path / "wheels"
    pb = shared / "envs" / "sdfbase-fit" / "bin" / "python"
    assert uv[0] == today[0].replace(str(shared / DEFAULT_PY_REL), str(tmp_path / "py313" / "bin" / "python3.12"))
    assert uv[1:3] == today[1:3] and uv[4] == today[4] and uv[6:] == today[6:]
    assert uv[3] == f"uv pip install --no-config --python {pb} --no-index --find-links {wdir} --no-deps causal-conv1d==1.6.2.post1 [CC1D= MAMBA=]"
    assert uv[5] == f"uv pip install --no-config --python {pb} --no-index --find-links {wdir} --no-deps mamba-ssm==2.3.2.post1 [CC1D= MAMBA=]"
    assert not any("--no-build-isolation" in l or "=TRUE" in l for l in uv)
    assert len(py) == 4 and py[3] == "python -"          # the kernel-version / .so-hash step
    assert "kernel wheels (sha256):" in r.stdout and f"{wdir}/{MAMBA_WHL}" in r.stdout
    assert "interpreter from: SDFBASE_PY (environment); CUDA_HOME=" in r.stdout
    assert (shared / "envs" / "sdfbase-fit" / "BUILD-COMPLETE").exists()
    # a wheelhouse missing either pinned wheel refuses BEFORE any uv call
    r, uv, _, _ = _dry_run(tmp_path / "b", nvcc=False, wheels=(CC1D_WHL,), SDFBASE_KERNEL_SOURCE="wheel")
    assert r.returncode == 3 and f"FATAL prebuilt cu128 wheel missing: " in r.stdout
    assert MAMBA_WHL in r.stdout and uv == []
    # a bogus knob value refuses in the knob block itself
    r, uv, _, _ = _dry_run(tmp_path / "c", nvcc=True, SDFBASE_KERNEL_SOURCE="carried")
    assert r.returncode == 3 and uv == []
