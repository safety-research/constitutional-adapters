"""scripts/sbatch/cuda_toolkit.sh -- the one place the Mamba-kernel source build
picks its CUDA toolkit (s2_init's mamba branch and mamba_build.sbatch both
source it). The lockfile's torch is 2.11.0+cu128 and torch.utils.cpp_extension
raises on a toolkit of another MAJOR version, so the selection must VALIDATE
nvcc's release rather than trust a path: the 2026-10-12 review found the
unconditional `CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}` naming cluster B's
CUDA 13 toolkit, and a cold-cache fit there died inside uv's build.

Every test runs the real helper under bash with its absolute /usr/local probes
repointed at tmp fakes whose bin/nvcc answers `--version` like the real one.
"""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HELPER = REPO / "scripts" / "sbatch" / "cuda_toolkit.sh"


def _toolkit(root: Path, release: str) -> Path:
    (root / "bin").mkdir(parents=True, exist_ok=True)
    nvcc = root / "bin" / "nvcc"
    nvcc.write_text("#!/bin/sh\necho 'nvcc: NVIDIA (R) Cuda compiler driver'\n"
                    "echo 'Copyright (c) 2005-2025 NVIDIA Corporation'\n"
                    f"echo 'Cuda compilation tools, release {release}, V{release}.93'\n"
                    f"echo 'Build cuda_{release}.r{release}/compiler.35583870_0'\n")
    nvcc.chmod(0o755)
    return root


@pytest.fixture
def world(tmp_path):
    """The repointed helper plus a tmp /usr/local and a tmp $SHARED."""
    usr_local = tmp_path / "usr" / "local"
    usr_local.mkdir(parents=True)
    helper = tmp_path / "cuda_toolkit.sh"
    helper.write_text(HELPER.read_text().replace("/usr/local/cuda", f"{usr_local}/cuda"))
    shared = tmp_path / "shared"
    shared.mkdir()
    return helper, usr_local, shared


def _select(helper: Path, **env):
    """Source the helper, run s2_cuda12_select, report rc + CUDA_HOME + release + PATH head."""
    base = {k: v for k, v in os.environ.items() if k not in ("CUDA_HOME", "SHARED", "S2_CUDA_SHARED_TOOLKIT", "S2_CUDA_MAJOR")}
    base.update(env)
    prog = (f'set -u; source "{helper}"; s2_cuda12_select; rc=$?; '
            'printf "RC=%s\\nCUDA_HOME=%s\\nREL=%s\\nPATH0=%s\\n" "$rc" "${CUDA_HOME:-unset}" "${S2_CUDA_RELEASE:-unset}" "${PATH%%:*}"')
    r = subprocess.run(["bash", "-c", prog], capture_output=True, text=True, env=base, timeout=30)
    got = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
    return got, r.stderr


def test_cluster_a_system_toolkit_is_selected_and_put_first_on_path(world):
    helper, usr_local, shared = world
    _toolkit(usr_local / "cuda-12.8", "12.8")
    (usr_local / "cuda").symlink_to(usr_local / "cuda-12.8")
    got, err = _select(helper, SHARED=str(shared))
    assert got["RC"] == "0", err
    assert got["CUDA_HOME"] == f"{usr_local}/cuda-12.8"          # the versioned dir wins over the symlink
    assert got["REL"] == "12.8"
    assert got["PATH0"] == f"{usr_local}/cuda-12.8/bin"            # compute nodes ship nvcc OFF PATH
    assert err == ""


def test_the_cuda_symlink_is_accepted_when_it_is_cuda_12(world):
    helper, usr_local, shared = world
    _toolkit(usr_local / "cuda-12.6", "12.6")
    (usr_local / "cuda").symlink_to(usr_local / "cuda-12.6")
    got, err = _select(helper)
    assert got["RC"] == "0", err
    assert got["CUDA_HOME"] == f"{usr_local}/cuda" and got["REL"] == "12.6"


def test_cluster_b_cuda_13_only_is_refused_with_every_probe_named(world):
    """The review's scenario: /usr/local/cuda -> cuda-13.0, nothing else."""
    helper, usr_local, shared = world
    _toolkit(usr_local / "cuda-13.0", "13.0")
    (usr_local / "cuda").symlink_to(usr_local / "cuda-13.0")
    got, err = _select(helper, SHARED=str(shared))
    assert got["RC"] == "1"
    assert got["CUDA_HOME"] == "unset" and got["REL"] == "unset"
    assert "no CUDA 12 toolkit" in err and "cu128" in err
    for probe in (f"{usr_local}/cuda-12.8: absent", f"{usr_local}/cuda: release 13.0",
                  f"{usr_local}/cuda-12: absent", f"{shared}/cuda-12.8: absent"):
        assert probe in err, probe


def test_the_warm_ups_shared_install_is_a_candidate_after_the_system_dirs(world):
    """Once mamba_build.sbatch has installed $SHARED/cuda-12.8 on a CUDA-13
    cluster, every later fit selects it -- and a CUDA 12 system toolkit, when
    one exists, still comes first (the install is the fallback, not a shadow)."""
    helper, usr_local, shared = world
    _toolkit(usr_local / "cuda-13.0", "13.0")
    (usr_local / "cuda").symlink_to(usr_local / "cuda-13.0")
    _toolkit(shared / "cuda-12.8", "12.8")
    got, err = _select(helper, SHARED=str(shared))
    assert got["RC"] == "0", err
    assert got["CUDA_HOME"] == f"{shared}/cuda-12.8" and got["PATH0"] == f"{shared}/cuda-12.8/bin"
    # S2_CUDA_SHARED_TOOLKIT names the install path explicitly (the warmer's seam)
    other = _toolkit(shared / "elsewhere" / "cuda", "12.8")
    got, err = _select(helper, SHARED=str(shared / "nope"), S2_CUDA_SHARED_TOOLKIT=str(other))
    assert got["RC"] == "0" and got["CUDA_HOME"] == str(other)
    # with no SHARED and no explicit path, there is simply no fourth candidate:
    # the three system probes and nothing under the (unset) shared root
    got, err = _select(helper)
    probed = err.split("probed:")[1]
    assert got["RC"] == "1" and f"{usr_local}/cuda-12.8: absent" in probed
    assert str(shared) not in probed and probed.count(": ") == 3
    # a system CUDA 12 outranks the shared install
    _toolkit(usr_local / "cuda-12.8", "12.8")
    got, err = _select(helper, SHARED=str(shared))
    assert got["CUDA_HOME"] == f"{usr_local}/cuda-12.8"


def test_a_callers_cuda_home_is_honoured_when_valid_and_refused_when_not(world):
    helper, usr_local, shared = world
    _toolkit(usr_local / "cuda-12.8", "12.8")
    own = _toolkit(shared / "opt" / "cuda-12.4", "12.4")
    got, err = _select(helper, CUDA_HOME=str(own), SHARED=str(shared))
    assert got["RC"] == "0" and got["CUDA_HOME"] == str(own) and got["REL"] == "12.4"
    assert got["PATH0"] == f"{own}/bin"
    # a CUDA 13 CUDA_HOME is refused, NOT silently replaced by the valid system one
    bad = _toolkit(shared / "opt" / "cuda-13.0", "13.0")
    got, err = _select(helper, CUDA_HOME=str(bad), SHARED=str(shared))
    assert got["RC"] == "1" and "refusing" in err and "release" not in got.get("REL", "unset")
    assert got["CUDA_HOME"] == str(bad)                            # left as the caller set it
    # a CUDA_HOME with no runnable nvcc is refused too
    empty = shared / "opt" / "empty"
    empty.mkdir(parents=True)
    got, err = _select(helper, CUDA_HOME=str(empty))
    assert got["RC"] == "1" and "none runnable" in err


def test_a_dir_without_a_runnable_nvcc_is_skipped_not_selected(world):
    helper, usr_local, shared = world
    (usr_local / "cuda-12.8" / "bin").mkdir(parents=True)        # a toolkit dir with no nvcc at all
    _toolkit(usr_local / "cuda-12", "12.8")
    got, err = _select(helper)
    assert got["RC"] == "0" and got["CUDA_HOME"] == f"{usr_local}/cuda-12"
    (usr_local / "cuda-12" / "bin" / "nvcc").unlink()
    got, err = _select(helper)
    assert got["RC"] == "1" and f"{usr_local}/cuda-12.8: no runnable bin/nvcc" in err


def test_s2_cuda_release_parses_nvcc_output(world):
    helper, usr_local, shared = world
    _toolkit(usr_local / "cuda-12.8", "12.8")
    prog = (f'source "{helper}"; s2_cuda_release "{usr_local}/cuda-12.8"; echo "rc=$?"; '
            f's2_cuda_release "{usr_local}/nothing"; echo "rc=$?"')
    r = subprocess.run(["bash", "-c", prog], capture_output=True, text=True, timeout=30)
    assert r.stdout.splitlines() == ["12.8", "rc=0", "rc=1"]


def test_the_helper_only_defines_functions_and_is_shell_clean():
    """Sourcing it must have no side effect beyond defining S2_CUDA_MAJOR and
    the two functions: s2_init sources it inside a running job."""
    subprocess.run(["bash", "-n", str(HELPER)], check=True)
    prog = (f'set -u; before=$(declare -p | wc -l); source "{HELPER}"; '
            'echo "MAJOR=$S2_CUDA_MAJOR"; declare -F s2_cuda12_select s2_cuda_release; '
            'echo "CUDA_HOME=${CUDA_HOME:-unset}"')
    env = {k: v for k, v in os.environ.items() if k != "CUDA_HOME"}
    r = subprocess.run(["bash", "-c", prog], capture_output=True, text=True, env=env, timeout=30)
    assert r.returncode == 0, r.stderr
    assert r.stdout.splitlines() == ["MAJOR=12", "s2_cuda12_select", "s2_cuda_release", "CUDA_HOME=unset"]
