"""``scripts/modal/tinycap_images.py`` — the Nemotron image variant:
the two pinned kernels built from source with ``--no-build-isolation --no-deps`` and the two
``*_FORCE_BUILD`` variables, the import smoke that fails the BUILD on a moved stack, the layer inserted
UNDER the launcher's mount layer (Modal refuses a build step after ``add_local_*``) with the mount
re-applied, the pins equal to the cluster overlay's, and the in-container smoke's record shape.
No Modal call is made here; the image objects are fakes with Modal's ``deps()`` shape.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def ti():
    spec = importlib.util.spec_from_file_location("_tinycap_images_under_test", REPO / "scripts/modal/tinycap_images.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_tinycap_images_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_image_nemotron_run_commands_carry_exactly_the_two_pins_with_the_build_discipline(ti):
    cmds = ti.nemotron_build_commands()
    assert len(cmds) == 3
    conv, ssm, smoke = cmds
    assert conv.endswith("pip install --no-build-isolation --no-deps causal-conv1d==1.6.2.post1")
    assert ssm.endswith("pip install --no-build-isolation --no-deps mamba-ssm==2.3.2.post1")
    assert "CAUSAL_CONV1D_FORCE_BUILD=TRUE" in conv and "MAMBA_FORCE_BUILD=TRUE" in ssm
    for c in (conv, ssm):
        assert "CUDA_HOME=/usr/local/cuda" in c and "TORCH_CUDA_ARCH_LIST='9.0;10.0'" in c and "MAX_JOBS=8" in c
    # exactly the two specs, each pinned to an exact release (no floating spec anywhere)
    specs = re.findall(r"(?:causal-conv1d|mamba-ssm)==[\w.]+", conv + " " + ssm)
    assert sorted(specs) == sorted(ti.NEMOTRON_KERNEL_SPECS) == ["causal-conv1d==1.6.2.post1", "mamba-ssm==2.3.2.post1"]
    assert ti.image_install_specs_nemotron() == list(ti.NEMOTRON_KERNEL_SPECS)
    # the smoke is the build's last step: stack unchanged + the four entry points import
    assert smoke.startswith('python -c "') and smoke.endswith('"') and '"' not in smoke[len('python -c "'):-1]
    body = ti.kernel_smoke_source()
    assert "assert torch.__version__ == '2.11.0+cu129'" in body
    assert "assert vllm.__version__.startswith('0.25.1')" in body
    assert "assert mamba_ssm.__version__ == '2.3.2.post1'" in body
    assert "assert causal_conv1d.__version__ == '1.6.2.post1'" in body
    for mod, name in ti.KERNEL_IMPORTS:
        assert f"from {mod} import {name}" in body
    assert body.rstrip().endswith(")") and ti.SMOKE_OK_MARKER in body


def test_pins_equal_the_cluster_overlay_requirements(ti):
    lines = [ln.strip() for ln in (REPO / ti.NEMOTRON_KERNEL_REQUIREMENTS).read_text().splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    assert sorted(lines) == sorted(ti.NEMOTRON_KERNEL_SPECS)


class _FakeMount:
    pass


class _FakeImage:
    """Modal's shape for the tests: ``deps()`` -> the objects this layer was built on; a mount layer's
    deps are ``[base Image, Mount]``, a build layer's ``[base Image]``."""

    def __init__(self, name, deps=()):
        self.name, self._deps_list, self.calls = name, list(deps), []

    def deps(self):
        return list(self._deps_list)

    def run_commands(self, *cmds, gpu=None, **kw):
        img = _FakeImage(f"{self.name}+run", deps=[self])
        img.calls = self.calls + [("run_commands", cmds, gpu)]
        return img

    def add_local_python_source(self, *pkgs, **kw):
        img = _FakeImage(f"{self.name}+mount", deps=[self, _FakeMount()])
        img.calls = self.calls + [("add_local_python_source", pkgs)]
        return img


def test_the_kernel_layer_is_inserted_under_the_launchers_mount_layer_and_the_mount_reapplied(ti):
    build = _FakeImage("build", deps=[_FakeImage("root")])
    launcher_image = build.add_local_python_source("sisteer")       # the launcher's shape
    assert ti.build_image_of(build) is build                          # mount-free: unchanged
    assert ti.build_image_of(launcher_image) is build                 # the mount layer is peeled
    img = ti.nemotron_image(launcher_image)
    kinds = [c[0] for c in img.calls]
    assert kinds == ["run_commands", "add_local_python_source"]
    run = img.calls[0]
    assert list(run[1]) == ti.nemotron_build_commands() and run[2] is None    # gpu=None: a CPU build step
    assert img.calls[1][1] == ("sisteer",) == ti.LAUNCHER_PYTHON_SOURCE
    # the launcher's ONE mount is exactly that call
    src = (REPO / "scripts/modal/tinycap_served.py").read_text(encoding="utf-8")
    assert src.count('.add_local_python_source("sisteer")') == 1
    assert len(re.findall(r"\.add_local_(?:python_source|dir|file)\(", src)) == \
        src.count("copy=True") + 1                                   # every other add_local_* is copy=True
    # two stacked mounts peel to the same build image
    twice = launcher_image.add_local_python_source("other")
    assert ti.build_image_of(twice) is build
    with pytest.raises(RuntimeError, match="mount-free build image"):
        ti.build_image_of(_FakeImage("weird", deps=[_FakeMount()]))


def test_the_smoke_records_a_moved_or_missing_stack_as_not_ok(ti):
    """On this CPU box (no cu129 torch, no vLLM) the smoke must come back ok=False with the failure
    RECORDED, never raise — the record is what the build report shows."""
    rec = ti.nemotron_kernel_smoke(contract=None)
    assert rec["ok"] is False
    assert set(rec) >= {"ok", "versions", "kernel_imports", "packages_report", "static_discovery"}
    # the contract-aware discovery is best-effort and recorded either way
    rec2 = ti.nemotron_kernel_smoke(contract={"roles": {}})
    assert rec2["ok"] is False


def test_build_requires_a_named_app(ti):
    with pytest.raises(SystemExit, match="--app-name is required"):
        ti.build_and_smoke("")
    assert ti.main(["--print-commands"]) == 0
    assert ti.main([]) == 2
