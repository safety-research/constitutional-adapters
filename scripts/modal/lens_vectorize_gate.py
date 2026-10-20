"""lens_vectorize_gate on Modal — the five-family GPU fidelity (bit-identity +
bounded reduction tolerance) + timing gate for the vectorized lens apply
(scripts/lens_vectorize_gate.py; read its docstring for what is proved and how
verdicts are decided), ported from scripts/sbatch/lens_vectorize_gate.sbatch
after the Slurm clusters went away (2026-09-04). The sbatch is the
specification of this file: everything it set up is reproduced here, and every
departure is named under WHAT IS WEAKER at the bottom.

WHAT RUNS. One H200 container per family (the sbatch's one array task per
family, fanned out with ``.spawn()`` so the families run concurrently). Inside
it the gate itself runs unchanged — the candidate tree's
``scripts/lens_vectorize_gate.py`` with ``--phase gate`` — standing the
production server up per served-op case THREE times through the tree-local
``scripts/lens_gate_standup.py``: the REFERENCE from a separate checkout of
the base commit (``--reference-sha``, 09be969 by default: the whole old
implementation, worker extension included), the CANDIDATE from the gated
tree, and the CANDIDATE+SHADOW (``SISTEER_LENS_SHADOW_COMPARE=1``, set by the
gate per standup, never here), driving identical greedy waves at B=1 (hard)
and the ``--batches`` ladder (evidence), the in-worker shadow compare held to
THE REDUCTION CONTRACT under the gate's default reduction mode (stacked; this
file never sets ``GATE_REDUCTION_MODE`` or ``SISTEER_LENS_STACKED_REDUCTIONS``
— the gate SETS the latter per candidate standup itself and any inherited value
is scrubbed here exactly as the sbatch ``unset`` it), every standup at
production ``VLLM_BATCH_INVARIANT=0`` (``--batch-invariant 0``, the variable
unset in the environment as the sbatch did).

TREES. The gate REQUIRES two git checkouts (``tree_facts`` / the helper's
``verify_tree``: ``git rev-parse HEAD`` == the expected sha, ``git diff
--quiet HEAD --`` clean). There is no cluster mirror to clone from any more
and the repo is private (no credentials go on Modal), so ``--stage`` builds a
SHALLOW clone of each sha from the LOCAL repo (``git fetch --depth 1 <sha>``
into a fresh ``git init``; HEAD detached at the sha; ~37 MB with its pack),
tars it, pushes the tarball over the Modal client (``Volume.batch_upload``),
and ``unpack_tree`` extracts it into ``/vol/repos/sisteer-<sha>`` and verifies
HEAD + cleanness ON the volume before committing. The earlier ``/vol/repos``
trees (other lanes') carry no ``.git`` and are left alone; a same-sha dir that
already exists is verified, never overwritten (``--restage`` to replace).
Each family container copies both trees to LOCAL disk first (``/root/trees``)
— git on a shared volume view would rewrite the index under five concurrent
containers, and symlinked staging (the gate's ``stage_cell`` / multigroup
dir) needs a real filesystem — and the gate verifies the copies again.

IMAGES. Two, for the transformers split the sbatch selects per MODEL:
``configs/capeval/geom-sweep-requirements-tf5141.txt`` (transformers==5.14.1)
for gemma4e4b — 5.15.0 + vLLM 0.25.1 raises
AmbiguousGlobalPerLayerAttributeError at APIServer startup for gemma-4 — and
``configs/capeval/geom-sweep-requirements.txt`` (the lane's lock, 5.15.0,
which Nemotron-3-Nano NEEDS) for the other four. Both are built by the
sbatch's certified command: uv 0.12.1 installs the
selected geom-sweep requirements file under the lane's
``--exclude-newer 2026-08-03`` cutoff and ``unsafe-best-match`` index policy.
Those files include the same sha256-pinned vLLM 0.25.1+cu129 wheel, cu129
Torch 2.11.0 stack, and vllm-lens 1.2.1; the non-Gemma image adds the required
exact Transformers 5.15.0 pin because its requirements file inherits that
version from the old project environment rather than spelling it out. That
release postdates the lane's general cutoff, so only Transformers gets an
``--exclude-newer-package`` exception through 2026-08-11; every transitive
package remains under the 2026-08-03 boundary. This uses the reviewed dependency
inputs directly instead of maintaining a second hand-copied dependency list in
the Modal launcher.

INPUTS ON THE VOLUME (``twosi-fits`` at ``/vol``, the repo's Modal
convention): the HF cache at ``/vol/hf`` (``HF_HUB_OFFLINE=1`` in the gate;
``--weights`` snapshot-downloads any family whose pinned revision is missing,
reporting what was already present), the scale-12 object bank at
``/vol/objects/scale-12/<family>`` (READ-ONLY; llama8b / gemma4e4b / qwen35_4b
hold psr_kl_all + lora_r4 x 3 SIs, glm46vflash / nemotron3nano4b have no bank
— their cases are synthetic, labelled so in the gate's own report), the
candsi_refit per-family bank at ``/vol/candsi_refit/newmodels/anchor_bank``
(present, empty today) and the stage3-wave extra dirs keyed llama8b exactly as
the sbatch passed them (absent on the volume today; the gate skips a missing
dir and refuses an unattributable object). The gate mints a synthetic object
for every case no real object covers, and refuses a fitted object whose
fit-time record names another family — unchanged.

OUTPUTS. The gate writes to container-local disk (``/root/lens-gate-work``)
and this file copies ``report.md`` + ``lens_vectorize_gate.json`` + ``RESULT``
+ ``gate.log`` + every standup's ``standup.log`` / ``vllm_server.log`` /
``endpoint.json`` (never ``server_auth.json``) to
``/vol/lens-gate/<sha>/<model>[/subset/<slug>]/`` — a ``--cases`` subset
lands under ``subset/`` by the gate's own rule (``subset_slug``), so it can
never overwrite the family's full report — every 5 minutes while the gate
runs (a container killed at the timeout still leaves its partial JSON) and
once at the end. ``--go`` waits for every family and mirrors those files to
``$FIGS_DIR/lens-gate-stacked/<model>[/subset/<slug>]/`` on the grading host;
``--harvest <state.json>`` re-attaches by call id (the state file ``--go``
writes under ``~/.local/state/lens-gate/``) after a detached run.

RUN (the app/run name is REQUIRED. Modal 1.5 calls the CLI option
``--name``/``-n``; the entrypoint verifies it equals
``lens-gate-stacked-<shortsha>`` so an unnamed run is refused):

  ~/venvs/modal/bin/python -m modal run -n lens-gate-stacked-<shortsha> scripts/modal/lens_vectorize_gate.py --stage --sha <sha>
  ~/venvs/modal/bin/python -m modal run -n lens-gate-stacked-<shortsha> scripts/modal/lens_vectorize_gate.py --weights --sha <sha>
  ~/venvs/modal/bin/python -m modal run -n lens-gate-stacked-<shortsha> scripts/modal/lens_vectorize_gate.py --pf --sha <sha>
  ~/venvs/modal/bin/python -m modal run -n lens-gate-stacked-<shortsha> scripts/modal/lens_vectorize_gate.py --go --sha <sha> \
      --models llama8b --batches 8 --measure-s 5 \
      --cases psr:relu_gated_vector:gemv:response:skipbos0:both,vector_all:add_vector:vec:all:skipbos1:both
  ~/venvs/modal/bin/python -m modal run --detach -n lens-gate-stacked-<shortsha> scripts/modal/lens_vectorize_gate.py --go --sha <sha>
  ~/venvs/modal/bin/python -m modal run -n lens-gate-stacked-<shortsha> scripts/modal/lens_vectorize_gate.py \
      --harvest ~/.local/state/lens-gate/<app>-<ts>.json

Seams (the sbatch's body-env seams, as flags): ``--cases``, ``--batches``,
``--timing-batches``, ``--measure-s``, ``--legacy-ab``, ``--reduction-mode``
(passed to the gate as its flag; the default stays the gate's), ``--models``,
``--ref-sha``, and the served geometry ``--max-num-seqs`` / ``--max-model-len``
(passed to the gate exactly like ``--timing-batches``; the defaults equal the
gate's own — 64 x 4096 — so the certification run is unchanged; a TIMING-ONLY
row can run at ``--max-model-len 2048 --max-num-seqs 256 --timing-batches
8,32,64,128,256`` to size a wave). Both are recorded in the state file and in
the ``[lens-gate] app=...`` line. Not ported: the sbatch's ``GATE_BI0_PROBE`` (``--bi0-probe``)
and ``GATE_BATCH_INVARIANT=1`` opt-in experiments — this file always runs the
production regime.

CANARY (the smallest run that exercises every moving part — one family, one
real-object case and one synthetic case, the reduced ladder above): the
``--go`` line with ``--models llama8b --batches 8 --measure-s 5`` and the two
cases shown. The ``psr ... :response:`` case is the one llama8b's scale-12
bank actually covers (``objects: real``); ``vector_all`` has no fitted object
anywhere (``objects: synthetic``). It stands up the passthrough plus three
servers per case (reference / candidate / candidate+shadow), so it proves the
staging, both trees, the weights, the bank, the shadow compare, the sync and
the mirror in ~16 minutes on one H200. Its exit is 3 BY CONSTRUCTION (a
subset run is never a certification); the proof is both cases PASS with a
clean shadow contract in ``report.md``.

WHAT IS WEAKER THAN THE SLURM GATE (the verdict's evidence, honestly):
  * different hardware lineage: Modal's H200s + driver, not cluster A's; the
    reduction tolerance's measured concentration bounds came from cluster
    runs. The contract is hardware-agnostic (bf16 ulps of the edit), but a
    kernel-variance fraction measured here is a new sample, not a repeat.
  * the reference and candidate trees are shallow clones pushed from the
    grading host rather than clones of the cluster mirror; the sha and cleanness
    checks are the same, the provenance chain is one hop shorter.
  * no scale-12 bank for glm46vflash / nemotron3nano4b on the volume (the
    cluster's are gone): those two families are gated on synthetic objects
    only, where the cluster run might have served real fits.
  * the candsi_refit / stage3-wave real objects are absent, so llama8b's
    rank-1 LoRA and every candsi case fall to synthetic here.
  * a container killed at the Modal timeout has no SIGTERM grace: the
    5-minute sync is what survives, not a final flush.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path

import modal

#: ``modal run -n lens-gate-stacked-<shortsha>`` is REQUIRED. Modal calls
#: this the run ``--name``; it is the app name shown in the dashboard. The
#: local entrypoint verifies the exact descriptive name after resolving SHA,
#: so omitting ``-n`` cannot silently reuse a stale app identity.
DEFAULT_APP_NAME = "lens-vectorize-gate"
app = modal.App(DEFAULT_APP_NAME)
VOL = modal.Volume.from_name("twosi-fits", create_if_missing=False)
VOL_PATH = "/vol"
REPOS = f"{VOL_PATH}/repos"                       # staged trees: sisteer-<sha> (git checkouts here)
HF_HOME = f"{VOL_PATH}/hf"
OBJECTS_ROOT = f"{VOL_PATH}/objects/scale-12"      # READ-ONLY shared bank, <family>/<cell>/object.pt
CANDSI_ROOT = f"{VOL_PATH}/candsi_refit/newmodels/anchor_bank"   # <family>/<cand>__<arm>/ (empty today)
#: the sbatch's extra dirs, keyed llama8b — absent on the volume today; the
#: gate skips a missing dir and never serves an unattributable object
EXTRA_OBJECTS = {f"{VOL_PATH}/objects/stage3-wave/A": "llama8b",
                 f"{VOL_PATH}/objects/stage3-wave/C": "llama8b"}
RESULTS_ROOT = f"{VOL_PATH}/lens-gate"             # <sha>/<model>[/subset/<slug>]/
WORK_ROOT = "/root/lens-gate-work"                  # container-local: the gate's --results-root
TREES_ROOT = "/root/trees"                          # container-local copies of the staged trees
DEFAULT_REFERENCE_SHA = "09be969fb676db26a532917e7eb1673de40605f5"   # lens_vectorize_gate.DEFAULT_REFERENCE_SHA
MIRROR_ROOT = Path.home() / "figs" / "lens-gate-stacked"
STATE_DIR = Path.home() / ".local" / "state" / "lens-gate"
#: how often the running gate's report / JSON are copied to the volume
SYNC_EVERY_S = 300
#: the three switches the sbatch unset before the gate (the gate sets the
#: shadow / stacked ones PER STANDUP itself; batch invariance is the flag)
SCRUB_ENV = ("SISTEER_LENS_PER_ROW_APPLY", "SISTEER_LENS_SHADOW_COMPARE",
             "SISTEER_LENS_STACKED_REDUCTIONS", "VLLM_BATCH_INVARIANT", "GATE_REDUCTION_MODE")

# --- the families: ONE canonical table, lens_vectorize_gate.MODELS, mirrored ----
# (pinned equal by tests/test_modal_gates.py; the order is MODEL_ORDER)
MODELS = {
    "llama8b": dict(hf="NousResearch/Meta-Llama-3.1-8B-Instruct",
                    rev="d10aef7999a2b5ba950ab3974312feeedbfe0b77", cfg="configs/two_si.yaml"),
    "gemma4e4b": dict(hf="google/gemma-4-E4B-it", rev="ee0ef6023621cff504d758262d4e04895a5af4a2",
                      cfg="configs/two_si_gemma4e4b_reduced.yaml"),
    "qwen35_4b": dict(hf="Qwen/Qwen3.5-4B", rev="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                      cfg="configs/two_si_qwen35_4b.yaml"),
    "glm46vflash": dict(hf="zai-org/GLM-4.6V-Flash", rev="411bb4d77144a3f03accbf4b780f5acb8b7cde4e",
                        cfg="configs/two_si_glm46vflash.yaml"),
    "nemotron3nano4b": dict(hf="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                            rev="dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
                            cfg="configs/two_si_nemotron3nano4b.yaml"),
}
MODEL_ORDER = tuple(MODELS)
#: the sbatch's per-MODEL requirements selection: gemma-4 keys -> the
#: transformers==5.14.1 variant, everything else the lane's lock (5.15.0)
TF_VARIANT = {m: ("tf5141" if m in ("gemma4e4b", "gemma4e2b") else "tf5150") for m in MODELS}
REQS_FILE = {"tf5141": "configs/capeval/geom-sweep-requirements-tf5141.txt",
             "tf5150": "configs/capeval/geom-sweep-requirements.txt"}
TRANSFORMERS_PIN = {"tf5141": "5.14.1", "tf5150": "5.15.0"}

# --- the serving images: the pinned vLLM stack, cu129 --------------------------
try:
    LOCAL_REPO = Path(__file__).resolve().parents[2]
except IndexError:  # Modal re-imports this module at a flat /root path
    LOCAL_REPO = Path("/repo")
VLLM_REQUIREMENTS = LOCAL_REPO / "configs/capeval/vllm-requirements.txt"
UV_VERSION = "0.12.1"
SERVING_EXCLUDE_NEWER = "2026-08-03T00:00:00Z"
# Transformers 5.15.0 was released after the pinned stack's general cutoff;
# keep the cutoff for every other package and grant only this exact pin enough
# publication headroom. 5.14.1 predates the general cutoff.
TRANSFORMERS_EXCLUDE_NEWER = {"tf5141": None, "tf5150": "2026-08-11T00:00:00Z"}
SERVING_REQS_ROOT = "/opt/sisteer-capeval-reqs"


def _serving_image(variant: str) -> modal.Image:
    """Build from the exact requirements input and uv command used by Slurm.

    The outer geom-sweep file includes ``vllm-requirements.txt`` by relative
    path, so both files are copied together. Transformers 5.15.0 is explicit
    here because the default outer file inherited it from the old project
    environment; Modal has no surrounding synced environment to inherit.
    """
    if variant not in REQS_FILE:
        raise ValueError(f"unknown serving-image variant {variant!r}")
    tf_pin = TRANSFORMERS_PIN[variant]
    req_path = LOCAL_REPO / REQS_FILE[variant]
    remote_vllm = f"{SERVING_REQS_ROOT}/{VLLM_REQUIREMENTS.name}"
    remote_outer = f"{SERVING_REQS_ROOT}/{req_path.name}"
    tf_cutoff = TRANSFORMERS_EXCLUDE_NEWER[variant]
    tf_cutoff_arg = (f"--exclude-newer-package transformers={tf_cutoff} "
                     if tf_cutoff else "")
    install = (
        "uv pip install --python $(command -v python) "
        "--index-strategy unsafe-best-match "
        f"--exclude-newer {SERVING_EXCLUDE_NEWER} "
        f"{tf_cutoff_arg}-r {remote_outer} transformers=={tf_pin}"
    )
    return (
        modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
        .apt_install("git", "build-essential")
        .env({"CUDA_HOME": "/usr/local/cuda"})
        # Bootstrap only the pinned uv executable with pip; uv resolves and
        # installs the serving environment itself from the certified files.
        .pip_install(f"uv=={UV_VERSION}")
        .add_local_file(VLLM_REQUIREMENTS, remote_vllm, copy=True)
        .add_local_file(req_path, remote_outer, copy=True)
        .run_commands(
            f"uv --version | grep -E '^uv {UV_VERSION.replace('.', '[.]')}( |$)'",
            install,
            "git config --global --add safe.directory '*'",
            "python -c \"import torch, vllm, transformers, aiohttp, yaml, matplotlib; "
            "assert torch.__version__ == '2.11.0+cu129', torch.__version__; "
            "assert vllm.__version__.startswith('0.25.1'), vllm.__version__; "
            f"assert transformers.__version__ == '{tf_pin}', transformers.__version__; "
            "print('IMAGE OK', vllm.__version__, torch.__version__, transformers.__version__)\"",
        )
        .env({"HF_HOME": HF_HOME, "HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    )


image_tf5150 = _serving_image("tf5150")
image_tf5141 = _serving_image("tf5141")
IMAGES = {"tf5150": image_tf5150, "tf5141": image_tf5141}


# --------------------------------------------------------------------------- #
# helpers (pure enough to unit-test; nothing here imports modal at call time)   #
# --------------------------------------------------------------------------- #


def volume_client_path(path: str) -> str:
    """Translate a mounted ``/vol/...`` path to the Volume client's ``/...``."""
    if path == VOL_PATH:
        return "/"
    prefix = f"{VOL_PATH}/"
    if not path.startswith(prefix):
        raise ValueError(f"path is outside {VOL_PATH}: {path}")
    return path[len(VOL_PATH):]


def inventory_real_sources(cases: dict) -> dict[str, list[str]]:
    """Real artifact paths by case; ``adapter-probe`` is synthetic coverage."""
    return {
        case: [src for src in (entry.get("sources") or []) if src != "adapter-probe"]
        for case, entry in cases.items()
        if any(src != "adapter-probe" for src in (entry.get("sources") or []))
    }


def resolve_models(models: str | list | None) -> list:
    """The family keys a run covers, in MODEL_ORDER; an unknown key refuses
    (never a silent skip of a misspelled family)."""
    if isinstance(models, str):
        sel = [m for m in models.split(",") if m.strip()]
    else:
        sel = list(models or [])
    unknown = [m for m in sel if m not in MODELS]
    if unknown:
        raise SystemExit(f"[lens-gate] unknown family key(s) {unknown}; known: {list(MODELS)}")
    sel = sel or list(MODELS)
    return [m for m in MODEL_ORDER if m in sel]


def subset_slug(cases: list) -> str:
    """lens_vectorize_gate.subset_slug, byte for byte (the sbatch mirrored it
    with ``tr ':,' '_+'``): a ``--cases`` subset writes under
    ``<model>/subset/<slug>/`` so it can never overwrite the full report."""
    return "+".join(str(c).replace(":", "_") for c in cases)


def results_rel(model: str, cases: list | None) -> str:
    """The gate's results dir relative to its --results-root (main():
    ``<model>`` or ``<model>/subset/<slug>``)."""
    return f"{model}/subset/{subset_slug(cases)}" if cases else model


#: the gate's own served-geometry defaults (scripts/lens_vectorize_gate.py's
#: ``--max-num-seqs`` / ``--max-model-len``); tests/test_modal_gates.py pins these
#: equal to the gate's parser so the certification run's geometry cannot drift
DEFAULT_MAX_NUM_SEQS = 64
DEFAULT_MAX_MODEL_LEN = 4096


def gate_argv(python: str, cand: str, ref: str, *, model: str, ref_sha: str, results_root: str,
              cases: list | None, batches: str, timing_batches: str, measure_s: float,
              legacy_ab: bool, reduction_mode: str | None, ready_timeout_s: int,
              max_num_seqs: int = DEFAULT_MAX_NUM_SEQS, max_model_len: int = DEFAULT_MAX_MODEL_LEN) -> list:
    """The sbatch's ``srun ... lens_vectorize_gate.py`` line, as argv: the
    candidate tree's gate script, --phase gate, both trees, the volume's
    object roots, the extra dirs keyed llama8b, --batch-invariant 0 (the
    production regime), the ladder seams, and the served geometry
    (``--max-num-seqs`` / ``--max-model-len``, always passed — the defaults are
    the gate's own). ``--reduction-mode`` is passed ONLY when the caller gave
    one — otherwise the gate's default stands."""
    argv = [python, f"{cand}/scripts/lens_vectorize_gate.py",
            "--model", model, "--phase", "gate", "--repo-dir", cand,
            "--reference-repo", ref, "--reference-sha", ref_sha,
            "--results-root", results_root, "--objects-root", OBJECTS_ROOT,
            "--candsi-root", CANDSI_ROOT]
    for d, key in EXTRA_OBJECTS.items():
        argv += ["--extra-objects-dir", f"{d}={key}"]
    if cases:
        argv += ["--cases", ",".join(cases)]
    if legacy_ab:
        argv += ["--legacy-ab"]
    if reduction_mode:
        argv += ["--reduction-mode", reduction_mode]
    argv += ["--batch-invariant", "0", "--batches", batches, "--timing-batches", timing_batches,
             "--measure-s", str(measure_s), "--gpu-mem-util", "0.90",
             "--ready-timeout-s", str(int(ready_timeout_s)),
             "--max-num-seqs", str(int(max_num_seqs)), "--max-model-len", str(int(max_model_len))]
    return argv


def launch_state(*, run_name: str, sha: str, ref_sha: str, case_list: list | None, batches: str,
                 timing_batches: str, max_num_seqs: int, max_model_len: int, models: list,
                 calls: dict) -> dict:
    """The ``--go`` state file (``~/.local/state/lens-gate/<app>-<ts>.json``):
    what was launched, incl. the served geometry, and the call ids ``--harvest``
    re-attaches by."""
    return {"app": run_name, "sha": sha, "ref_sha": ref_sha, "cases": case_list, "batches": batches,
            "timing_batches": timing_batches, "max_num_seqs": int(max_num_seqs),
            "max_model_len": int(max_model_len), "models": list(models), "calls": dict(calls),
            "results_root": f"{RESULTS_ROOT}/{sha}"}


def banner_line(*, run_name: str, app_id: str, sha: str, ref_sha: str, models: list, case_list: list | None,
                batches: str, timing_batches: str, measure_s: float, max_num_seqs: int,
                max_model_len: int) -> str:
    """The ``[lens-gate] app=...`` line ``--go`` prints: every seam the run was
    launched with, the served geometry included."""
    return (f"[lens-gate] app={run_name} app_id={app_id} sha={sha[:12]} ref={ref_sha[:12]} models={models} "
            f"cases={case_list or 'ALL'} batches={batches} timing_batches={timing_batches} measure_s={measure_s} "
            f"max_num_seqs={int(max_num_seqs)} max_model_len={int(max_model_len)}")


def gate_env(base: dict, cand: str) -> dict:
    """The gate's environment: the sbatch's ``unset`` of the per-row / shadow /
    stacked / batch-invariance switches (the gate sets what it needs per
    standup), no GATE_REDUCTION_MODE (the default stands unless passed as a
    flag), PYTHONPATH naming the candidate tree only."""
    env = dict(base)
    for k in SCRUB_ENV:
        env.pop(k, None)
    env["PYTHONPATH"] = f"{cand}/src:{cand}/scripts"
    env.setdefault("HF_HOME", HF_HOME)
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def snapshot_rel(hf: str, rev: str) -> str:
    """lens_vectorize_gate.shell_env_lines' GATE_HF_SNAPSHOT_REL."""
    return f"hub/models--{hf.replace('/', '--')}/snapshots/{rev}"


def _git(repo: str, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          timeout=120).stdout.strip()


def verify_checkout(repo: str, sha: str) -> dict:
    """``{"ok", "head", "clean", "why"}`` — the gate's own tree facts: HEAD
    equals the sha and no tracked file is modified."""
    head = _git(repo, "rev-parse", "HEAD")
    clean = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=repo,
                           capture_output=True).returncode == 0 if head else False
    ok = bool(head) and head == sha and clean
    why = None if ok else (f"not a git checkout" if not head else
                           f"HEAD {head} != {sha}" if head != sha else "modified tracked files")
    return {"ok": ok, "head": head or None, "clean": clean, "why": why}


def summarize_gate_json(doc: dict) -> dict:
    """The per-family summary the entrypoint prints: exit code, complete,
    every case's verdict + object label (real vs synthetic) + failed hard
    checks, and the inventory notes (foreign / unadaptable real artifacts)."""
    cases = []
    for c in doc.get("cases") or []:
        checks = c.get("checks") or {}
        failed = sorted(k for k, v in checks.items()
                        if isinstance(v, dict) and v.get("ok") is False)
        none_ = sorted(k for k, v in checks.items()
                       if isinstance(v, dict) and v.get("ok") is None)
        pure = ((c.get("verdicts") or {}).get("pure_B1") or {}).get("verdict")
        cases.append({"case": c.get("case_id"), "objects": c.get("objects"),
                      "verdict": c.get("verdict") or pure, "pure_B1": pure,
                      "error": (c.get("error") or "")[:200] or None,
                      "checks_failed": failed, "checks_none": none_})
    real = sum(1 for c in cases if str(c.get("objects") or "").startswith("real"))
    synth = sum(1 for c in cases if str(c.get("objects") or "").startswith("synthetic"))
    return {"exit_code": doc.get("exit_code"), "complete": doc.get("complete"),
            "n_cases": len(cases), "n_real_objects": real, "n_synthetic_objects": synth,
            "uncovered": doc.get("uncovered_cases"), "tree_error": doc.get("tree_error"),
            "passthrough_error": doc.get("passthrough_error"),
            "inventory_notes": doc.get("inventory_notes"), "cases": cases}


# --------------------------------------------------------------------------- #
# staging: the two trees, the weights                                          #
# --------------------------------------------------------------------------- #


def local_repo_root() -> Path:
    """The git repo this file lives in (the worktree's COMMON dir's parent —
    every worktree shares its objects, so any sha reachable anywhere in the
    repo can be fetched from it)."""
    here = Path(__file__).resolve().parent
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            cwd=str(here), capture_output=True, text=True, check=True).stdout.strip()
    return Path(common).parent


def shallow_clone_tarball(sha: str, dest_dir: Path) -> Path:
    """A SHALLOW standalone clone of ``sha`` from the local repo (``git init``
    + ``fetch --depth 1`` of the sha — allowed by the local transport with
    ``uploadpack.allowAnySHA1InWant``), HEAD detached at the sha, verified,
    tarred (gzip) as ``dest_dir/sisteer-<sha>.tar.gz`` whose single top-level
    entry is ``sisteer-<sha>/``."""
    src = local_repo_root()
    if not _git(str(src), "cat-file", "-t", sha) == "commit":
        raise SystemExit(f"[lens-gate] {sha} is not a commit in {src}")
    work = Path(tempfile.mkdtemp(prefix="lens-gate-stage-"))
    tree = work / f"sisteer-{sha}"
    tree.mkdir()
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    subprocess.run(["git", "-c", "uploadpack.allowAnySHA1InWant=true", "-C", str(tree),
                    "fetch", "-q", "--depth", "1", str(src), sha], check=True)
    subprocess.run(["git", "-C", str(tree), "checkout", "-q", "--detach", "FETCH_HEAD"], check=True)
    v = verify_checkout(str(tree), sha)
    if not v["ok"]:
        raise SystemExit(f"[lens-gate] shallow clone of {sha} failed verification: {v}")
    tar_path = dest_dir / f"sisteer-{sha}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(str(tree), arcname=f"sisteer-{sha}")
    shutil.rmtree(work, ignore_errors=True)
    return tar_path


@app.function(image=image_tf5150, volumes={VOL_PATH: VOL}, timeout=1800, cpu=4)
def unpack_tree(sha: str, force: bool = False) -> dict:
    """Extract ``/vol/repos/sisteer-<sha>.tar.gz`` (pushed by --stage) into
    ``/vol/repos/sisteer-<sha>``, verify the checkout ON the volume (HEAD ==
    sha, clean), delete the tarball, commit. An existing dir is verified and
    kept unless ``force``."""
    VOL.reload()
    dst = Path(REPOS) / f"sisteer-{sha}"
    tar_path = Path(REPOS) / f"sisteer-{sha}.tar.gz"
    out = {"sha": sha, "tree": str(dst)}
    if dst.exists() and not force:
        v = verify_checkout(str(dst), sha)
        out.update(existing=True, verify=v)
        if tar_path.exists():
            tar_path.unlink()
            VOL.commit()
        if not v["ok"]:
            out["error"] = (f"{dst} exists but does not verify ({v['why']}) — a tree from another "
                            "lane without .git, or the wrong sha; --restage replaces it")
        return out
    if not tar_path.is_file():
        out["error"] = f"no tarball at {tar_path} — run --stage first"
        return out
    tmp = Path(REPOS) / f".unpack-{sha}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(str(tmp), filter="fully_trusted")
    extracted = tmp / f"sisteer-{sha}"
    if not (extracted / ".git").exists():
        out["error"] = f"tarball carries no .git under sisteer-{sha}/"
        shutil.rmtree(tmp, ignore_errors=True)
        return out
    if dst.exists():
        shutil.rmtree(dst)
    extracted.rename(dst)
    shutil.rmtree(tmp, ignore_errors=True)
    tar_path.unlink()
    v = verify_checkout(str(dst), sha)
    out.update(existing=False, verify=v)
    if not v["ok"]:
        out["error"] = f"unpacked tree does not verify: {v['why']}"
    VOL.commit()
    return out


@app.function(image=image_tf5150, volumes={VOL_PATH: VOL}, timeout=3 * 3600, cpu=8,
              secrets=[modal.Secret.from_name("hf-token")])
def stage_weights(models: list) -> dict:
    """Per family: report the pinned snapshot PRESENT (config.json under the
    pinned revision, the sbatch's guard) or snapshot_download it (network on
    here only; the workspace hf-token secret for gated repos). Commits only
    when something was downloaded."""
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import snapshot_download
    VOL.reload()
    out, downloaded = {}, False
    for m in models:
        spec = MODELS[m]
        snap = Path(HF_HOME) / snapshot_rel(spec["hf"], spec["rev"])
        if (snap / "config.json").is_file():
            out[m] = {"status": "present", "snapshot": str(snap)}
            continue
        t0 = time.time()
        try:
            p = snapshot_download(spec["hf"], revision=spec["rev"], local_files_only=False)
            ok = (Path(p) / "config.json").is_file()
            out[m] = {"status": "downloaded" if ok else "download-incomplete", "snapshot": p,
                      "seconds": round(time.time() - t0, 1)}
            downloaded = downloaded or ok
        except Exception as e:  # noqa: BLE001 — reported per family, never a silent skip
            out[m] = {"status": "FAILED", "error": f"{type(e).__name__}: {e}"[:500]}
    if downloaded:
        VOL.commit()
    return out


# --------------------------------------------------------------------------- #
# preflight (CPU, per family on its own image): trees, weights, banks, inventory#
# --------------------------------------------------------------------------- #


def _preflight_family(model: str, sha: str, ref_sha: str) -> dict:
    VOL.reload()
    spec = MODELS[model]
    rep = {"model": model, "variant": TF_VARIANT[model], "ok": True, "checks": {}}

    def chk(name, fn):
        try:
            rep["checks"][name] = {"ok": True, **(fn() or {})}
        except BaseException as e:  # noqa: BLE001
            rep["checks"][name] = {"ok": False, "err": f"{type(e).__name__}: {e}"[:800]}
            rep["ok"] = False

    import importlib.metadata as md
    chk("image_pins", lambda: {"transformers": md.version("transformers"), "vllm": md.version("vllm"),
                               "torch": md.version("torch"), "vllm_lens": md.version("vllm-lens"),
                               "expected_transformers": TRANSFORMERS_PIN[TF_VARIANT[model]]})
    if md.version("transformers") != TRANSFORMERS_PIN[TF_VARIANT[model]]:
        rep["checks"]["image_pins"]["ok"] = False
        rep["ok"] = False

    cand, ref = f"{REPOS}/sisteer-{sha}", f"{REPOS}/sisteer-{ref_sha}"

    def trees():
        if sha == ref_sha:
            raise AssertionError(f"reference sha equals the candidate sha {sha} — nothing to gate")
        vc, vr = verify_checkout(cand, sha), verify_checkout(ref, ref_sha)
        if not vc["ok"]:
            raise AssertionError(f"candidate {cand}: {vc['why']}")
        if not vr["ok"]:
            raise AssertionError(f"reference {ref}: {vr['why']}")
        for rel in ("scripts/lens_vectorize_gate.py", "scripts/lens_gate_standup.py", spec["cfg"],
                    "configs/capeval/manifests/petri_twosi.yaml"):
            if not Path(cand, rel).is_file():
                raise AssertionError(f"candidate tree lacks {rel}")
        if not Path(ref, "src/sisteer/capeval/serving/vllm_server.py").is_file():
            raise AssertionError("reference tree lacks the serving stack")
        return {"candidate": vc, "reference": vr}
    chk("trees", trees)

    def snapshot():
        snap = Path(HF_HOME) / snapshot_rel(spec["hf"], spec["rev"])
        if not (snap / "config.json").is_file():
            raise AssertionError(f"pinned snapshot {snap} has no config.json — run --weights")
        return {"snapshot": str(snap)}
    chk("pinned_snapshot", snapshot)

    def banks():
        bank = Path(OBJECTS_ROOT) / model
        cells = sorted(p.name for p in bank.iterdir() if (p / "object.pt").is_file()) if bank.is_dir() else []
        candsi = Path(CANDSI_ROOT) / model
        cand_cells = sorted(p.name for p in candsi.iterdir() if p.is_dir()) if candsi.is_dir() else None
        extra = {d: (sorted(p.name for p in Path(d).iterdir()) if Path(d).is_dir() else None)
                 for d in EXTRA_OBJECTS}
        return {"scale12_bank": str(bank), "scale12_cells": cells,
                "scale12_present": bool(cells),
                "candsi_dir": str(candsi), "candsi_cells": cand_cells, "extra_dirs": extra}
    chk("object_banks", banks)

    def env_phase():
        r = subprocess.run([sys.executable, f"{cand}/scripts/lens_vectorize_gate.py", "--model", model,
                            "--phase", "env", "--repo-dir", cand],
                           capture_output=True, text=True, timeout=600, env=gate_env(os.environ, cand))
        if r.returncode != 0:
            raise AssertionError(f"--phase env rc={r.returncode}: {r.stderr[-800:]}")
        lines = dict(l.split("=", 1) for l in r.stdout.splitlines() if "=" in l)
        return {"env": lines}
    chk("gate_phase_env", env_phase)

    def inventory():
        argv = [sys.executable, f"{cand}/scripts/lens_vectorize_gate.py", "--model", model,
                "--phase", "inventory", "--repo-dir", cand, "--objects-root", OBJECTS_ROOT,
                "--candsi-root", CANDSI_ROOT]
        for d, key in EXTRA_OBJECTS.items():
            argv += ["--extra-objects-dir", f"{d}={key}"]
        r = subprocess.run(argv, capture_output=True, text=True, timeout=1200,
                           env=gate_env(os.environ, cand))
        if r.returncode != 0:
            raise AssertionError(f"--phase inventory rc={r.returncode}: {r.stderr[-1200:]}")
        txt = r.stdout
        doc = json.loads(txt[txt.find("{"):])
        cases = doc.get("cases") or {}
        real_sources = inventory_real_sources(cases)
        real = sorted(real_sources)
        return {"n_cases": len(cases), "mandatory": len(doc.get("mandatory") or []),
                "cases_with_real_sources": real,
                "real_sources_by_case": real_sources,
                "cases_synthetic_only": sorted(k for k in cases if k not in real),
                "timing_measured": doc.get("timing_measured"),
                "unadaptable_real_artifacts": doc.get("unadaptable_real_artifacts"),
                "foreign_real_artifacts": doc.get("foreign_real_artifacts"),
                "tokenizer_note": r.stderr.strip()[-300:] or None}
    chk("gate_phase_inventory", inventory)
    return rep


@app.function(image=image_tf5150, volumes={VOL_PATH: VOL}, timeout=1800, cpu=8)
def preflight_tf5150(model: str, sha: str, ref_sha: str) -> dict:
    return _preflight_family(model, sha, ref_sha)


@app.function(image=image_tf5141, volumes={VOL_PATH: VOL}, timeout=1800, cpu=8)
def preflight_tf5141(model: str, sha: str, ref_sha: str) -> dict:
    return _preflight_family(model, sha, ref_sha)


PREFLIGHT = {"tf5150": preflight_tf5150, "tf5141": preflight_tf5141}


# --------------------------------------------------------------------------- #
# the gate, one family per H200                                                #
# --------------------------------------------------------------------------- #


def _copy_tree(sha: str) -> str:
    """The staged tree copied to local disk and verified there."""
    src, dst = Path(REPOS) / f"sisteer-{sha}", Path(TREES_ROOT) / f"sisteer-{sha}"
    if not (src / ".git").exists():
        raise RuntimeError(f"{src} is not a git checkout (no .git) — run --stage for {sha}")
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(src, dst, symlinks=True)
    v = verify_checkout(str(dst), sha)
    if not v["ok"]:
        raise RuntimeError(f"local copy of {sha} does not verify: {v['why']}")
    return str(dst)


def _sync_results(work_dir: Path, vol_dir: Path, *, final: bool) -> list:
    """Copy the gate's report / JSON (+ the logs and every standup's
    non-secret files at the end) into the volume dir and commit."""
    copied = []
    vol_dir.mkdir(parents=True, exist_ok=True)
    names = ["report.md", "lens_vectorize_gate.json", "gate.log", "RESULT"]
    for n in names:
        p = work_dir / n
        if p.is_file():
            shutil.copy2(p, vol_dir / n)
            copied.append(n)
    if final:
        su_root = work_dir / "standups"
        if su_root.is_dir():
            for su in sorted(su_root.iterdir()):
                for n in ("standup.log", "vllm_server.log", "endpoint.json", "two_si_gate.yaml"):
                    p = su / n
                    if p.is_file():
                        d = vol_dir / "standups" / su.name
                        d.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, d / n)
                        copied.append(f"standups/{su.name}/{n}")
    VOL.commit()
    return copied


def _gate_family(model: str, sha: str, ref_sha: str, cases: list | None, batches: str,
                 timing_batches: str, measure_s: float, legacy_ab: bool,
                 reduction_mode: str | None, ready_timeout_s: int, run_name: str,
                 max_num_seqs: int = DEFAULT_MAX_NUM_SEQS, max_model_len: int = DEFAULT_MAX_MODEL_LEN) -> dict:
    T0 = time.time()
    VOL.reload()
    spec = MODELS[model]
    rel = results_rel(model, cases)
    vol_dir = Path(RESULTS_ROOT) / sha / rel
    work_dir = Path(WORK_ROOT) / rel
    rec = {"model": model, "sha": sha, "ref_sha": ref_sha, "variant": TF_VARIANT[model],
           "cases": cases, "batches": batches, "timing_batches": timing_batches,
           "max_num_seqs": int(max_num_seqs), "max_model_len": int(max_model_len),
           "results_dir": str(vol_dir), "app": run_name}
    import importlib.metadata as md
    rec["pins"] = {"transformers": md.version("transformers"), "vllm": md.version("vllm"),
                   "torch": md.version("torch")}
    if rec["pins"]["transformers"] != TRANSFORMERS_PIN[TF_VARIANT[model]]:
        rec["status"] = "WRONG_IMAGE"
        rec["error"] = (f"{model} needs transformers {TRANSFORMERS_PIN[TF_VARIANT[model]]}, "
                        f"the container has {rec['pins']['transformers']}")
        return rec
    try:
        if sha == ref_sha:
            raise RuntimeError(f"reference sha equals the candidate sha {sha} — nothing to gate")
        cand, ref = _copy_tree(sha), _copy_tree(ref_sha)
        snap = Path(HF_HOME) / snapshot_rel(spec["hf"], spec["rev"])
        if not (snap / "config.json").is_file():
            raise RuntimeError(f"pinned snapshot {snap} has no config.json — run --weights first")
        bank = Path(OBJECTS_ROOT) / model
        rec["scale12_bank_cells"] = (sorted(p.name for p in bank.iterdir() if (p / "object.pt").is_file())
                                     if bank.is_dir() else [])
    except Exception as e:  # noqa: BLE001
        rec["status"] = "SETUP_FAILED"
        rec["error"] = f"{type(e).__name__}: {e}"[:1500]
        return rec

    work_dir.mkdir(parents=True, exist_ok=True)
    argv = gate_argv(sys.executable, cand, ref, model=model, ref_sha=ref_sha, results_root=WORK_ROOT,
                     cases=cases, batches=batches, timing_batches=timing_batches, measure_s=measure_s,
                     legacy_ab=legacy_ab, reduction_mode=reduction_mode,
                     ready_timeout_s=ready_timeout_s, max_num_seqs=max_num_seqs, max_model_len=max_model_len)
    env = gate_env(os.environ, cand)
    rec["argv"] = argv
    print(f"[lens-gate] {model} @ {sha[:12]} vs {ref_sha[:12]} ({TF_VARIANT[model]}: transformers "
          f"{rec['pins']['transformers']}, vllm {rec['pins']['vllm']}); bank cells "
          f"{rec['scale12_bank_cells']}; results -> {vol_dir}", flush=True)
    print("[lens-gate] " + " ".join(argv), flush=True)

    stop = threading.Event()

    def _periodic():
        while not stop.wait(SYNC_EVERY_S):
            try:
                _sync_results(work_dir, vol_dir, final=False)
            except Exception as e:  # noqa: BLE001 — a sync failure never stops the gate
                print(f"[lens-gate] periodic sync failed: {type(e).__name__}: {e}", flush=True)
    threading.Thread(target=_periodic, name="sync", daemon=True).start()

    log = work_dir / "gate.log"
    rc = None
    with open(log, "ab") as lf:
        proc = subprocess.Popen(argv, env=env, cwd=cand, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        for line in proc.stdout:
            lf.write(line)
            lf.flush()
            sys.stdout.write(line.decode("utf-8", "replace"))
            sys.stdout.flush()
        rc = proc.wait()
    stop.set()
    state = "COMPLETED" if rc == 0 else "FAILED"
    (work_dir / "RESULT").write_text(
        f"{state} exit={rc} elapsed_s={int(time.time() - T0)} app={run_name} model={model} "
        f"sha={sha} ref={ref_sha} variant={TF_VARIANT[model]} cases={','.join(cases) if cases else 'ALL'} "
        f"timing_batches={timing_batches} max_num_seqs={int(max_num_seqs)} max_model_len={int(max_model_len)}\n")
    rec["gate_rc"] = rc
    rec["elapsed_s"] = round(time.time() - T0, 1)
    jp = work_dir / "lens_vectorize_gate.json"
    if jp.is_file():
        try:
            rec["summary"] = summarize_gate_json(json.loads(jp.read_text()))
        except Exception as e:  # noqa: BLE001
            rec["summary_error"] = f"{type(e).__name__}: {e}"
    else:
        rec["summary_error"] = "the gate wrote no lens_vectorize_gate.json"
        rec["gate_tail"] = log.read_text(errors="replace")[-3000:] if log.is_file() else None
    try:
        rec["synced"] = len(_sync_results(work_dir, vol_dir, final=True))
    except Exception as e:  # noqa: BLE001
        rec["sync_error"] = f"{type(e).__name__}: {e}"
    rec["status"] = "DONE" if rc == 0 else ("GATE_FAIL" if rc == 2 else
                                            "INCONCLUSIVE" if rc == 3 else f"EXIT_{rc}")
    return rec


#: 8 h was the Slurm budget (2-3.5 h observed per family); 9 h leaves the
#: tree copies and the final sync inside the container's clock
GATE_TIMEOUT_S = 9 * 3600


def _run_named_family(model: str, sha: str, ref_sha: str, cases: list | None, batches: str,
                      timing_batches: str, measure_s: float, legacy_ab: bool,
                      reduction_mode: str | None, ready_timeout_s: int,
                      run_name: str, max_num_seqs: int = DEFAULT_MAX_NUM_SEQS,
                      max_model_len: int = DEFAULT_MAX_MODEL_LEN) -> dict:
    rec = _gate_family(model, sha, ref_sha, cases, batches, timing_batches, measure_s,
                       legacy_ab, reduction_mode, ready_timeout_s, run_name,
                       max_num_seqs=max_num_seqs, max_model_len=max_model_len)
    print("[lens-gate] REC", json.dumps({k: v for k, v in rec.items() if k != "argv"}, default=str))
    return rec


# One named Modal function per family: dashboard rows and failures identify the
# family directly, while the image split stays exact (gemma at tf5141, four at
# tf5150). Each invocation owns one H200, matching one former array task.
@app.function(image=image_tf5150, volumes={VOL_PATH: VOL}, gpu="H200", cpu=16, memory=131072,
              timeout=GATE_TIMEOUT_S, retries=0)
def gate_llama8b(sha: str, ref_sha: str, cases: list | None, batches: str,
                  timing_batches: str, measure_s: float, legacy_ab: bool,
                  reduction_mode: str | None, ready_timeout_s: int, run_name: str,
                  max_num_seqs: int = DEFAULT_MAX_NUM_SEQS, max_model_len: int = DEFAULT_MAX_MODEL_LEN) -> dict:
    return _run_named_family("llama8b", sha, ref_sha, cases, batches, timing_batches,
                             measure_s, legacy_ab, reduction_mode, ready_timeout_s, run_name,
                             max_num_seqs=max_num_seqs, max_model_len=max_model_len)


@app.function(image=image_tf5141, volumes={VOL_PATH: VOL}, gpu="H200", cpu=16, memory=131072,
              timeout=GATE_TIMEOUT_S, retries=0)
def gate_gemma4e4b(sha: str, ref_sha: str, cases: list | None, batches: str,
                    timing_batches: str, measure_s: float, legacy_ab: bool,
                    reduction_mode: str | None, ready_timeout_s: int, run_name: str,
                    max_num_seqs: int = DEFAULT_MAX_NUM_SEQS, max_model_len: int = DEFAULT_MAX_MODEL_LEN) -> dict:
    return _run_named_family("gemma4e4b", sha, ref_sha, cases, batches, timing_batches,
                             measure_s, legacy_ab, reduction_mode, ready_timeout_s, run_name,
                             max_num_seqs=max_num_seqs, max_model_len=max_model_len)


@app.function(image=image_tf5150, volumes={VOL_PATH: VOL}, gpu="H200", cpu=16, memory=131072,
              timeout=GATE_TIMEOUT_S, retries=0)
def gate_qwen35_4b(sha: str, ref_sha: str, cases: list | None, batches: str,
                   timing_batches: str, measure_s: float, legacy_ab: bool,
                   reduction_mode: str | None, ready_timeout_s: int, run_name: str,
                   max_num_seqs: int = DEFAULT_MAX_NUM_SEQS, max_model_len: int = DEFAULT_MAX_MODEL_LEN) -> dict:
    return _run_named_family("qwen35_4b", sha, ref_sha, cases, batches, timing_batches,
                             measure_s, legacy_ab, reduction_mode, ready_timeout_s, run_name,
                             max_num_seqs=max_num_seqs, max_model_len=max_model_len)


@app.function(image=image_tf5150, volumes={VOL_PATH: VOL}, gpu="H200", cpu=16, memory=131072,
              timeout=GATE_TIMEOUT_S, retries=0)
def gate_glm46vflash(sha: str, ref_sha: str, cases: list | None, batches: str,
                     timing_batches: str, measure_s: float, legacy_ab: bool,
                     reduction_mode: str | None, ready_timeout_s: int, run_name: str,
                     max_num_seqs: int = DEFAULT_MAX_NUM_SEQS, max_model_len: int = DEFAULT_MAX_MODEL_LEN) -> dict:
    return _run_named_family("glm46vflash", sha, ref_sha, cases, batches, timing_batches,
                             measure_s, legacy_ab, reduction_mode, ready_timeout_s, run_name,
                             max_num_seqs=max_num_seqs, max_model_len=max_model_len)


@app.function(image=image_tf5150, volumes={VOL_PATH: VOL}, gpu="H200", cpu=16, memory=131072,
              timeout=GATE_TIMEOUT_S, retries=0)
def gate_nemotron3nano4b(sha: str, ref_sha: str, cases: list | None, batches: str,
                         timing_batches: str, measure_s: float, legacy_ab: bool,
                         reduction_mode: str | None, ready_timeout_s: int, run_name: str,
                         max_num_seqs: int = DEFAULT_MAX_NUM_SEQS, max_model_len: int = DEFAULT_MAX_MODEL_LEN) -> dict:
    return _run_named_family("nemotron3nano4b", sha, ref_sha, cases, batches, timing_batches,
                             measure_s, legacy_ab, reduction_mode, ready_timeout_s, run_name,
                             max_num_seqs=max_num_seqs, max_model_len=max_model_len)


GATE_FN = {
    "llama8b": gate_llama8b,
    "gemma4e4b": gate_gemma4e4b,
    "qwen35_4b": gate_qwen35_4b,
    "glm46vflash": gate_glm46vflash,
    "nemotron3nano4b": gate_nemotron3nano4b,
}


# --------------------------------------------------------------------------- #
# local side: mirror + state                                                   #
# --------------------------------------------------------------------------- #


def _vol_read(path: str) -> bytes:
    return b"".join(VOL.read_file(path))


def mirror_family(sha: str, model: str, cases: list | None, mirror_root: Path) -> dict:
    """Pull ``/vol/lens-gate/<sha>/<rel>/{report.md, JSON, RESULT, gate.log}``
    (+ every standup's logs) to ``<mirror_root>/<rel>/`` on the grading host."""
    rel = results_rel(model, cases)
    src = f"{volume_client_path(RESULTS_ROOT)}/{sha}/{rel}"
    dst = mirror_root / rel
    dst.mkdir(parents=True, exist_ok=True)
    pulled = []
    try:
        entries = VOL.listdir(src, recursive=True)
    except Exception as e:  # noqa: BLE001
        return {"model": model, "dst": str(dst), "error": f"{type(e).__name__}: {e}"}
    for e in entries:
        # Modal's client returns full Volume paths without a leading slash,
        # while its read/list APIs accept canonical slash-prefixed paths.
        p = "/" + e.path.lstrip("/")
        if not p.startswith(src + "/"):
            continue
        rel_p = p[len(src) + 1:]
        entry_type = getattr(getattr(e, "type", None), "name", "")
        if entry_type == "DIRECTORY" or p.endswith("/") or rel_p.endswith("server_auth.json"):
            continue
        try:
            data = _vol_read(p)
        except Exception:  # noqa: BLE001 — a directory entry, or a file mid-write
            continue
        out = dst / rel_p
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        pulled.append(rel_p)
    return {"model": model, "dst": str(dst), "pulled": pulled}


def modal_exit_code(records: list[dict]) -> int:
    """Aggregate per-family gate outcomes with the reviewed gate's 0/2/3
    semantics: any hard failure wins (2), otherwise any inconclusive/missing
    result is 3, else 0. A setup or image failure is hard, never a successful
    Modal command that merely printed an error."""
    hard = False
    incomplete = False
    for rec in records:
        rc = rec.get("gate_rc")
        status = rec.get("status")
        if rc == 2 or status in {"GATE_FAIL", "SETUP_FAILED", "WRONG_IMAGE"}:
            hard = True
        elif rc != 0 or status != "DONE":
            incomplete = True
    return 2 if hard else 3 if incomplete else 0


def _print_summary(rec: dict) -> None:
    s = rec.get("summary") or {}
    head = (f"[lens-gate] {rec.get('model')}: status={rec.get('status')} gate_rc={rec.get('gate_rc')} "
            f"elapsed={rec.get('elapsed_s')}s complete={s.get('complete')} cases={s.get('n_cases')} "
            f"real={s.get('n_real_objects')} synthetic={s.get('n_synthetic_objects')} "
            f"results={rec.get('results_dir')}")
    print(head, flush=True)
    if rec.get("error"):
        print(f"           error: {rec['error']}", flush=True)
    for c in s.get("cases") or []:
        flag = "" if not c.get("checks_failed") else f"  checks_failed={c['checks_failed']}"
        err = "" if not c.get("error") else f"  error={c['error']}"
        print(f"           {c.get('verdict') or '?':22} {c.get('objects') or '?':40} {c.get('case')}{flag}{err}",
              flush=True)


@app.local_entrypoint()
def main(sha: str = "", ref_sha: str = DEFAULT_REFERENCE_SHA, models: str = "",
         stage: bool = False, weights: bool = False, pf: bool = False, go: bool = False,
         harvest: str = "", cases: str = "", batches: str = "1,8,32", timing_batches: str = "8,32,64",
         measure_s: float = 45.0, legacy_ab: bool = False, reduction_mode: str = "",
         ready_timeout_s: int = 2700, restage: bool = False, no_mirror: bool = False,
         mirror_root: str = "", max_num_seqs: int = 64, max_model_len: int = 4096):
    mroot = Path(mirror_root).expanduser() if mirror_root else MIRROR_ROOT
    sel = resolve_models(models)
    case_list = [c.strip() for c in cases.split(",") if c.strip()] or None
    if harvest:
        st = json.loads(Path(harvest).expanduser().read_text())
        recs = []
        for m, cid in st["calls"].items():
            print(f"[lens-gate] harvesting {m} from call {cid} ...", flush=True)
            rec = modal.FunctionCall.from_id(cid).get()
            recs.append(rec)
            _print_summary(rec)
            if not no_mirror:
                print(json.dumps(mirror_family(st["sha"], m, st.get("cases"), mroot), indent=1))
        code = modal_exit_code(recs)
        if code:
            raise SystemExit(code)
        return
    if not sha:
        sha = _git(str(Path(__file__).resolve().parent), "rev-parse", "HEAD")
        print(f"[lens-gate] --sha not given: using the local HEAD {sha} (the STAGED tree at that sha "
              "is what runs; local edits never travel)", flush=True)
    if len(sha) != 40 or len(ref_sha) != 40:
        raise SystemExit("[lens-gate] pass FULL 40-hex shas (--sha / --ref-sha)")
    if sha == ref_sha:
        raise SystemExit(f"[lens-gate] --sha equals --ref-sha {sha}: nothing to gate")
    expected_name = f"lens-gate-stacked-{sha[:7]}"
    run_name = app.description or DEFAULT_APP_NAME
    if run_name != expected_name:
        raise SystemExit(f"[lens-gate] run with `modal run -n {expected_name} ...`; got app name "
                         f"{run_name!r}. A descriptive explicit app name is mandatory so Modal "
                         "cannot route this run under a stale identity")
    print(banner_line(run_name=run_name, app_id=app.app_id, sha=sha, ref_sha=ref_sha, models=sel,
                      case_list=case_list, batches=batches, timing_batches=timing_batches, measure_s=measure_s,
                      max_num_seqs=max_num_seqs, max_model_len=max_model_len), flush=True)

    if stage:
        with tempfile.TemporaryDirectory(prefix="lens-gate-tars-") as td:
            for s in (sha, ref_sha):
                tar = shallow_clone_tarball(s, Path(td))
                print(f"[lens-gate] pushing {tar.name} ({tar.stat().st_size // 1_000_000} MB) -> "
                      f"{REPOS}/", flush=True)
                with VOL.batch_upload(force=True) as b:
                    b.put_file(str(tar), f"{volume_client_path(REPOS)}/{tar.name}")
                r = unpack_tree.remote(s, force=restage)
                print(json.dumps(r, indent=1))
                if r.get("error"):
                    raise SystemExit(f"[lens-gate] staging {s} failed: {r['error']}")
        if not (weights or pf or go):
            return
    if weights:
        r = stage_weights.remote(sel)
        print(json.dumps(r, indent=1))
        bad = [m for m, v in r.items() if v.get("status") not in ("present", "downloaded")]
        if bad:
            raise SystemExit(f"[lens-gate] weights missing for {bad}")
        if not (pf or go):
            return
    if pf:
        calls = {m: PREFLIGHT[TF_VARIANT[m]].spawn(m, sha, ref_sha) for m in sel}
        reps = {m: c.get() for m, c in calls.items()}
        print(json.dumps(reps, indent=1, default=str))
        bad = [m for m, r in reps.items() if not r.get("ok")]
        if bad:
            raise SystemExit(f"[lens-gate] PREFLIGHT FAILED for {bad} — fix before any GPU")
        if not go:
            return
    if not go:
        raise SystemExit("pass --stage / --weights / --pf / --go / --harvest <state.json>")

    calls = {}
    for m in sel:
        fn = GATE_FN[m]
        calls[m] = fn.spawn(sha, ref_sha, case_list, batches, timing_batches, float(measure_s),
                            bool(legacy_ab), reduction_mode or None, int(ready_timeout_s), run_name,
                            int(max_num_seqs), int(max_model_len))
        print(f"[lens-gate] spawned {m} ({TF_VARIANT[m]}): call {calls[m].object_id}", flush=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path = STATE_DIR / f"{run_name}-{time.strftime('%Y%m%dT%H%M%S')}.json"
    state_path.write_text(json.dumps(launch_state(
        run_name=run_name, sha=sha, ref_sha=ref_sha, case_list=case_list, batches=batches,
        timing_batches=timing_batches, max_num_seqs=max_num_seqs, max_model_len=max_model_len, models=sel,
        calls={m: c.object_id for m, c in calls.items()}), indent=1))
    print(f"[lens-gate] state -> {state_path} (re-attach with --harvest {state_path})", flush=True)
    records = []
    for m, c in calls.items():
        rec = c.get()
        records.append(rec)
        _print_summary(rec)
        if not no_mirror:
            print(json.dumps(mirror_family(sha, m, case_list, mroot), indent=1))
    code = modal_exit_code(records)
    print(f"[lens-gate] Modal aggregate exit={code} (same 0/2/3 meaning as the gate)", flush=True)
    if code:
        raise SystemExit(code)
