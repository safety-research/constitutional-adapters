#!/usr/bin/env python3
"""The NEMOTRON IMAGE VARIANT for the tinycap Modal launcher: a
Nemotron / NemotronH / Mamba-hybrid model NEVER runs without the fused Mamba kernels — ``mamba-ssm`` +
``causal-conv1d`` are installed BEFORE its first job, and the quiet pure-torch SSM fallback is refused
by attestation, never inferred from a slow job.

``image_nemotron`` = the launcher's ONE serving image (``scripts/modal/tinycap_served.py:image`` —
cuda 12.9.1-devel, torch 2.11.0+cu129, vLLM 0.25.1+cu129, vllm-lens 1.2.1, transformers 5.14.1,
the hashed client venv) PLUS the two pinned kernel packages built FROM SOURCE against that torch
(no wheel exists at torch 2.11 / cu129 for either; ``*_FORCE_BUILD=TRUE`` makes the source build the
explicit, deterministic path instead of a 404'ing wheel probe), ``--no-build-isolation`` (the venv's
own torch is the build's torch), ``--no-deps`` (mamba-ssm 2.3.2.post1's Mamba3-only tilelang / tvm /
quack chain is not needed for NemotronH's Mamba2 path and would drag unpinned packages in; every
dependency the Mamba2 path needs — torch, triton, einops, packaging, ninja — is already pinned in the
base), then an IMPORT SMOKE that fails the BUILD if the stack moved (torch / vLLM versions asserted
unchanged) or a kernel entry point does not import. A first build is a 20–40 min nvcc compile; Modal
caches the layer.

Two ways to consume this module:

* the launcher: ``import tinycap_images; image_nemotron = tinycap_images.nemotron_image(image)``
  right after its ``image`` definition (a function taking the base image — the launcher imports this
  module, so this module must not import the launcher at module load: that is a cycle);
  ``image_nemotron`` is ALSO exported lazily (module ``__getattr__``) for ``from tinycap_images import
  image_nemotron`` from any process where the launcher is importable and not mid-import;
* the build / smoke: ``python scripts/modal/tinycap_images.py --build --app-name <name>`` builds the
  image and runs :func:`nemotron_kernel_smoke` in ONE CPU container (``gpu=None``: importing the
  compiled extensions needs the libraries, not a device) — versions, the four kernel entry points, and
  the static backend discovery against ``configs/capeval/mamba_backend_contract.json`` when vLLM's
  model module imports on CPU (recorded either way). The app name is REQUIRED (every Modal run is
  named; ``contracts.app_name_for``-style descriptive names) and the function is SERIALIZED, so the
  container never re-imports this module or the launcher.

Nothing here launches a GPU function.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

HERE = Path(__file__).resolve().parent


def _repo_local() -> Path:
    for parent in HERE.parents:
        if (parent / "src" / "sisteer").is_dir() and (parent / "scripts" / "modal").is_dir():
            return parent
    return HERE.parent


REPO_LOCAL = _repo_local()

#: the two fused-kernel packages, at the fit env's pins (both > 1 week old; the cluster overlay
#: ``configs/capeval/vllm-requirements-nemotron.txt`` carries the same two lines)
NEMOTRON_KERNEL_SPECS: Tuple[str, str] = ("causal-conv1d==1.6.2.post1", "mamba-ssm==2.3.2.post1")
NEMOTRON_KERNEL_REQUIREMENTS = "configs/capeval/vllm-requirements-nemotron.txt"
#: what the image's stack must STILL be after the two builds (asserted in the smoke)
EXPECTED_TORCH = "2.11.0+cu129"
EXPECTED_VLLM_PREFIX = "0.25.1"
EXPECTED_VERSIONS: Mapping[str, str] = {"mamba_ssm": "2.3.2.post1", "causal_conv1d": "1.6.2.post1"}
#: the source build's environment: nvcc from the cuda-devel base, Hopper + Blackwell, bounded parallelism
BUILD_ENV = "CUDA_HOME=/usr/local/cuda CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST='9.0;10.0' MAX_JOBS=8"
FORCE_BUILD_VAR: Mapping[str, str] = {"causal-conv1d": "CAUSAL_CONV1D_FORCE_BUILD=TRUE",
                                      "mamba-ssm": "MAMBA_FORCE_BUILD=TRUE"}
PIP_FLAGS = "--no-build-isolation --no-deps"
#: the four fused entry points the smoke imports (the letter of the rule; vLLM serves on its OWN
#: Triton kernels — the served-path attestation is ``sisteer.capeval.serving.mamba_backend``)
KERNEL_IMPORTS: Tuple[Tuple[str, str], ...] = (
    ("causal_conv1d", "causal_conv1d_fn"),
    ("causal_conv1d", "causal_conv1d_update"),
    ("mamba_ssm.ops.triton.ssd_combined", "mamba_chunk_scan_combined"),
    ("mamba_ssm.ops.triton.selective_state_update", "selective_state_update"),
)
SMOKE_OK_MARKER = "NEMOTRON IMAGE OK"


def kernel_smoke_source() -> str:
    """The ``python -c`` body run at the END of the image build: the stack must not have moved and the
    kernels must import — a regressed stack fails the BUILD, never the first cell."""
    imports = "; ".join(f"from {mod} import {name}" for mod, name in KERNEL_IMPORTS)
    return (
        "import torch, vllm, mamba_ssm, causal_conv1d; "
        f"assert torch.__version__ == '{EXPECTED_TORCH}', torch.__version__; "
        f"assert vllm.__version__.startswith('{EXPECTED_VLLM_PREFIX}'), vllm.__version__; "
        f"assert mamba_ssm.__version__ == '{EXPECTED_VERSIONS['mamba_ssm']}', mamba_ssm.__version__; "
        f"assert causal_conv1d.__version__ == '{EXPECTED_VERSIONS['causal_conv1d']}', causal_conv1d.__version__; "
        f"{imports}; "
        f"print('{SMOKE_OK_MARKER}', mamba_ssm.__version__, causal_conv1d.__version__, torch.__version__, vllm.__version__)"
    )


def nemotron_build_commands() -> List[str]:
    """The ``run_commands`` strings, in order: causal-conv1d, then mamba-ssm, then the smoke."""
    cmds = []
    for spec in NEMOTRON_KERNEL_SPECS:
        name = spec.split("==", 1)[0]
        cmds.append(f"{BUILD_ENV} {FORCE_BUILD_VAR[name]} pip install {PIP_FLAGS} {spec}")
    cmds.append(f'python -c "{kernel_smoke_source()}"')
    return cmds


#: the launcher's ONE mount layer (``image.add_local_python_source("sisteer")``: the launcher
#: process's own ``sisteer`` at /root; pinned against the launcher's source by test) — re-applied
#: on top of the kernel layer, because a build step cannot follow a mount layer
LAUNCHER_PYTHON_SOURCE: Tuple[str, ...] = ("sisteer",)


def build_image_of(image: Any) -> Any:
    """The BUILD image under ``image``: ``image`` itself when it carries no mount layers, else the
    mount-free image its ``add_local_*`` layers were added to. Modal refuses a build step after an
    ``add_local_*`` mount layer, and the launcher's ``image`` ends in ``add_local_python_source
    ("sisteer")`` — so the kernel layer must be inserted BEFORE that mount and the mount re-applied
    after it (:func:`nemotron_image`). TRANSITIONAL: reads the pre-mount image off the mount layer's
    dependency list (``_Image._deps``, modal 1.5.4 — the same client pin the launcher's allocation
    reader holds); the launcher exporting a mount-free ``image_build`` makes this a no-op."""
    deps = image_deps(image)
    if not any(type(d).__name__.endswith("Mount") for d in deps):
        return image                      # a build image: no add_local_* layer on top
    for dep in deps:
        if type(dep).__name__.endswith("Image"):
            return build_image_of(dep)    # the image the mount layer was added to
    raise RuntimeError("cannot find the mount-free build image under an image with add_local_* layers "
                       "(modal internals changed?) — have the launcher export its mount-free image_build")


def image_deps(image: Any) -> List[Any]:
    """An image's dependency objects (``Image.deps()``: a list, or a callable returning one — modal
    1.5.4 exposes the mount layer as ``[<base Image>, <Mount>]``; a build layer as ``[<base Image>]``)."""
    deps = getattr(image, "deps", None)
    if deps is None:
        return []
    got = deps() if callable(deps) else deps
    got = got() if callable(got) else got
    return list(got or [])


def nemotron_image(base: Any, *, python_source: Tuple[str, ...] = LAUNCHER_PYTHON_SOURCE) -> Any:
    """``image_nemotron`` from the launcher's image: the two source builds + the smoke as ONE cached
    layer group on a CPU build machine (``gpu=None`` — nvcc needs no device), inserted under the
    launcher's mount layer(s), which are re-applied (``python_source``) so the launcher process finds
    its ``sisteer`` in the Nemotron container exactly as in the base one."""
    img = build_image_of(base).run_commands(*nemotron_build_commands(), gpu=None)
    for pkg in python_source:
        img = img.add_local_python_source(pkg)
    return img


def image_install_specs_nemotron() -> List[str]:
    """Every requirement string the Nemotron layer installs (for the launcher's pin audit)."""
    return list(NEMOTRON_KERNEL_SPECS)


def base_image() -> Any:
    """The launcher's serving image, imported lazily (the launcher imports THIS module)."""
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import tinycap_served  # the launcher; requires TINYCAP_APP_NAME in a local process
    return tinycap_served.image


def __getattr__(name: str) -> Any:
    if name == "image_nemotron":
        img = nemotron_image(base_image())
        globals()["image_nemotron"] = img
        return img
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- the in-container smoke (serialized: self-contained, imports inside) --------------------------

def nemotron_kernel_smoke(contract: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Runs INSIDE image_nemotron on CPU: versions + the four kernel entry points (``ok`` iff every one
    imports at the expected versions), plus — recorded, never the pass criterion — the pip-half
    ``packages_report`` and the static backend discovery against the contract when vLLM's NemotronH
    model module imports without a GPU."""
    import importlib
    rec: Dict[str, Any] = {"ok": False, "python": sys.executable, "versions": {}, "kernel_imports": {},
                           "packages_report": None, "static_discovery": None}
    try:
        import torch
        import vllm
        # plain str: torch.__version__ is a TorchVersion (str subclass) that needs torch to unpickle
        rec["versions"] = {"torch": str(torch.__version__), "vllm": str(vllm.__version__)}
        for mod in ("mamba_ssm", "causal_conv1d", "triton", "transformers"):
            try:
                rec["versions"][mod] = str(importlib.import_module(mod).__version__)
            except Exception as e:  # noqa: BLE001
                rec["versions"][mod] = f"IMPORT FAILED: {type(e).__name__}: {e}"
        for mod, name in KERNEL_IMPORTS:
            try:
                obj = getattr(importlib.import_module(mod), name)
                rec["kernel_imports"][f"{mod}.{name}"] = {"ok": callable(obj), "module": str(getattr(obj, "__module__", None))}
            except Exception as e:  # noqa: BLE001
                rec["kernel_imports"][f"{mod}.{name}"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        rec["ok"] = (rec["versions"].get("torch") == EXPECTED_TORCH
                     and str(rec["versions"].get("vllm", "")).startswith(EXPECTED_VLLM_PREFIX)
                     and all(rec["versions"].get(k) == v for k, v in EXPECTED_VERSIONS.items())
                     and all(v.get("ok") for v in rec["kernel_imports"].values()))
    except Exception as e:  # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
    try:
        from sisteer.capeval.serving import mamba_backend as mb
        rec["packages_report"] = mb.packages_report()
        if contract is not None:
            att = mb.static_attest(contract)
            rec["static_discovery"] = att
    except Exception as e:  # noqa: BLE001 — sisteer may not be importable here; recorded
        rec["static_discovery_error"] = f"{type(e).__name__}: {e}"
    return rec


def build_and_smoke(app_name: str, *, contract_path: Optional[Path] = None, timeout_s: int = 3600) -> Dict[str, Any]:
    """Build ``image_nemotron`` (CPU-side; Modal caches the layer) and run the smoke in one CPU
    container under a NAMED ephemeral app. Never a GPU function."""
    if not app_name or not app_name.strip():
        raise SystemExit("--app-name is required: every Modal run is named (build the name like "
                         "tinycap-f22-nemo-image-<sha7>)")
    import modal   # after the cheap refusal: the modal client is not installed in every test venv
    os.environ.setdefault("TINYCAP_APP_NAME", app_name)   # the launcher's import refuses unnamed local imports
    contract = None
    if contract_path is not None and Path(contract_path).is_file():
        contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    img = nemotron_image(base_image())
    app = modal.App(app_name)
    fn = app.function(image=img, timeout=timeout_s, cpu=4, memory=16384, serialized=True)(nemotron_kernel_smoke)
    with modal.enable_output(), app.run():
        return fn.remote(contract)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true", help="build image_nemotron and run the CPU kernel smoke")
    ap.add_argument("--app-name", default="", help="the Modal app name for the build run (REQUIRED with --build)")
    ap.add_argument("--contract", default=str(REPO_LOCAL / "configs/capeval/mamba_backend_contract.json"),
                    help="the reviewed backend contract handed to the smoke's static discovery (recorded)")
    ap.add_argument("--print-commands", action="store_true", help="print the image's run_commands and exit")
    ap.add_argument("--out", default=None, help="write the smoke record here as JSON")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    if a.print_commands:
        for c in nemotron_build_commands():
            print(c)
        return 0
    if not a.build:
        build_parser().print_help()
        return 2
    rec = build_and_smoke(a.app_name, contract_path=Path(a.contract) if a.contract else None)
    text = json.dumps(rec, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if rec.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
