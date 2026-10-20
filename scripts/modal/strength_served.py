"""strength-served: the strength dose-ladder legs on Modal H200s.

ONE LEG = ONE `target_sampled.sbatch` INVOCATION, UNMODIFIED. The Slurm clusters were
unavailable (cluster B queue, cluster A frozen), so this app runs the SAME instrument
-- `scripts/sdfbase/full22/sampled/target_sampled.sbatch` at the wave sha, sourcing the
same `common_sampled.sh` / `common.sh`, standing up through the same
`scripts/sdfbase/mt_serving_up.py`, driving with the same `served_capability_run.py` /
`jailbreak_run.py`, folding the same ladder -- inside one H200 container per
(key, view, phase). Nothing under `scripts/sdfbase/`, `src/` or the configs is edited or
forked: the port is an ENVIRONMENT the sbatch already knows how to run in.

HOW THE PATHS ARE MADE FAITHFUL. `common.sh f22_init` derives everything from ONE root:
`$SHARED` (the shared storage root, resolved per cluster), then
`$SHARED/scripts/hf_env.sh`, `$SHARED/sisteer-eval-repo/<sha>`,
`$SHARED/exp/sdfbase_full22`, `$SHARED/exp/objects/sdfbase/full22/...`,
`$HOME/venvs/...`. The container builds `$CROOT` (CONTAINER_USER = "strength", so no
recorded path can be mistaken for a cluster user's; `$CHOME` is its home) as a
container-local tree whose leaves point at the volume:

  $CROOT/scripts/hf_env.sh              written here: HF_HOME=/vol/hf (the read-only shared cache)
  $CROOT/sisteer-eval-repo/<sha>        a COPY of /vol/repos/sisteer-<sha> (the cpu_suite staging; git-verified)
  $CROOT/exp/objects                    a COPY of /vol/strength/objects (the bank, symlinks preserved)
  $CROOT/exp/sdfbase_full22             container-local WAVE ROOT, seeded from and synced to
                                        /vol/strength/exp/sdfbase_full22_modal  (a DISTINCT root: never a cluster store)
  $CROOT/exp/heldout_corpora -> /vol/heldout_corpora
  $CROOT/.cache/uv -> /opt/uv-cache     the image's pre-warmed uv cache (the SAME uv.lock / requirements)
  $CHOME/venvs/sisteer-petri-vllm-<h>   built IN THE IMAGE by f22_serving_venv's exact command + sentinel,
                                        so the job finds "serving venv already bootstrapped" (as a warm node does)
  $CHOME/venvs/f22-<sha12>              built AT RUN TIME by f22_driver_venv itself (`uv sync --frozen --extra cu128`
                                        in the staged tree, from the warm cache), so sisteer resolves INSIDE the sha tree

The sbatch's only Slurm dependencies are naming/logging: `$SLURM_JOB_ID` (in the serving
dir name; `set -u` makes it required), `$SLURM_JOB_NAME` (FIGS_DIR / RESULT), and
`$CUDA_VISIBLE_DEVICES` (`TARGET_GPU="$CUDA_VISIBLE_DEVICES"`, required under `set -u`;
Modal gives one device = 0). No `srun`/`scontrol`/`scancel` runs anywhere on the path
(grepped). Two labels are therefore path-derived, not hardware-derived, and are stated
rather than edited: `standup_jobs.jsonl` records `cluster: "cluster_b"` (f22_init's
branch for cluster B's shared root) and `job_id: modal-<task id>`; `node` is the
container's hostname. Every store root and the serving dir get an ADDITIVE
`modal_evidence.json` (app, function call id, task id, hostname, GPU name/UUID/driver,
image, argv, the path map) so the record builder can see the hardware axis.

Volume writes are container-local first, then shipped incrementally (changed files only)
and COMMITTED every `sync_every_s` and at exit -- the grokmt/fst precedent: a torn
`events.jsonl` on a volume mount, or two containers appending to one, is exactly what the
cluster's shared filesystem never had to solve. Two legs of the same (TAG, trial) share
`ladders/<TAG>_t<T>/events.jsonl` on the cluster; here the fan-out SERIALISES them
(`--legs` never runs two legs of one (TAG, trial) at once) and each leg seeds its local
ladder dir from the volume first.

Run (the grading host; `-n strength-served-<sha7>` is REQUIRED, the cpu_suite rule):

  M=~/venvs/modal/bin/python; SHA=f09206a3d5f7c96eb4a37eee871e4612278fb58d; S7=${SHA:0:7}
  $M -m modal run -n strength-served-$S7 scripts/modal/strength_served.py --sha $SHA --build        # the GPU image (~25 min first time)
  $M -m modal run -n strength-served-$S7 scripts/modal/strength_served.py --sha $SHA --stage --stage-bank --stage-tools
  $M -m modal run -n strength-served-$S7 scripts/modal/strength_served.py --sha $SHA --pf --key nemotron3nano4b --view d0p25
  $M -m modal run --detach -n strength-served-$S7 scripts/modal/strength_served.py --sha $SHA --go \\
      --key nemotron3nano4b --view d0p25 --phase controls_t0 --trial 0
  $M -m modal run -n strength-served-$S7 scripts/modal/strength_served.py --sha $SHA --watch      # DONE|FAILED lines -> legs.log
  $M -m modal run -n strength-served-$S7 scripts/modal/strength_served.py --sha $SHA --pull nemotron3nano4b_d0p25 --pull-to <dir>
  # Phase 2 fan-out: one `key view phase trial` per line
  $M -m modal run --detach -n strength-served-$S7 scripts/modal/strength_served.py --sha $SHA --legs legs.txt --max-concurrent 6
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from pathlib import Path

import modal

DEFAULT_APP_NAME = "strength-served"
app = modal.App(DEFAULT_APP_NAME)
VOL_NAME = "twosi-fits"
VOL = modal.Volume.from_name(VOL_NAME, create_if_missing=False)
VOL_PATH = "/vol"
REPOS = f"{VOL_PATH}/repos"                       # the shared tree staging (cpu_suite / grokmt convention)
STRENGTH = f"{VOL_PATH}/strength"                 # everything this lane WRITES lives under here
BANK_VOL = f"{STRENGTH}/objects/sdfbase/full22"   # == $SHARED/exp/objects/sdfbase/full22 (roster bank_root)
WAVE_VOL = f"{STRENGTH}/exp/sdfbase_full22_modal"  # the DISTINCT wave root (never sdfbase_full22)
LOGS_VOL = f"{STRENGTH}/exp/logs_modal"
FIGS_VOL = f"{STRENGTH}/exp/figs_modal"
HF_VOL = f"{VOL_PATH}/hf"                         # READ-ONLY shared HF cache (models + pinned datasets)
CORPORA_VOL = f"{VOL_PATH}/heldout_corpora"       # READ-ONLY (renellm / multilingual replay corpora)
CONTAINER_USER = "strength"
SHARED_LOCAL = f"/workspace/{CONTAINER_USER}"     # f22_init: [ -d /workspace/$USER ] -> SHARED, CLUSTER=cluster_b
VENVS_LOCAL = f"/home/{CONTAINER_USER}/venvs"
UV_CACHE = "/opt/uv-cache"
UV_PYTHONS = "/opt/uv-pythons"
UV_PIN = "0.12.1"
PY_VERSION = "3.12"
SBATCH_REL = "scripts/sdfbase/full22/sampled/target_sampled.sbatch"
VLLM_REQS_REL = "configs/capeval/vllm-requirements.txt"
PINS_REL = "configs/sdfbase_full22_pins.json"
ROSTER_REL = "configs/sdfbase_full22_roster.yaml"
DATASET_PINS_REL = "configs/capeval/pins.yaml"
DEPENDENCY_FILES = ("pyproject.toml", "uv.lock", VLLM_REQS_REL)
#: LADDER_RULES.md section 2: the classifier's rep_ratio source, sha-pinned by common_sampled.sh
REP_RATIO_SHA256 = "152d180376dd3ce9d6e054354b8bd4a6ddcf6ffd9d4635ab708398d32c5f33ae"
#: the grading host's stagings live under its staging directory ($STAGE, sisteer.site); the staging steps expand them
#: (_staged), so importing this module in a container needs no site config
REP_RATIO_LOCAL = "$STAGE/sdfbase/fullscale/full22_join_scores.py"
TOOLS_VOL = f"{WAVE_VOL}/sampled/tools"
#: the grading host staging of the bank subtree pulled from the cluster (symlinks preserved)
BANK_LOCAL = "$STAGE/strength/bank"
STATE_DIR = Path.home() / ".local" / "state" / "strength-modal"
CALLS_FILE = STATE_DIR / "calls.jsonl"
LEGS_LOG = STATE_DIR / "legs.log"
WAVE_ROOT_DEFAULT = "sdfbase_full22_modal"
WAVE_ROOT = WAVE_ROOT_DEFAULT


def set_wave_root(name: str) -> None:
    """Rebind the wave's volume universe (stores, logs, figures, tools, the local ledger) to `<STRENGTH>/exp/<name>`.
    The default keeps every historical path byte for byte (`sdfbase_full22_modal`, `logs_modal`, `figs_modal`,
    `legs.log`); any other name gets its own `logs_<name>` / `figs_<name>` / `legs_<name>.log`. Called locally from
    the CLI and FIRST THING inside every remote function from its `wave_root` argument (module globals do not
    travel to the container). The base->instruct sweep runs under `sdfbase_full22_native` — the same TAGs as
    the lens ladders on llama8b would otherwise land in one store dir and be read as one instrument."""
    global WAVE_ROOT, WAVE_VOL, LOGS_VOL, FIGS_VOL, TOOLS_VOL, LEGS_LOG
    if not re.fullmatch(r"[A-Za-z0-9_]+", name or ""):
        raise SystemExit(f"[strength] --wave-root must match [A-Za-z0-9_]+, got {name!r}")
    WAVE_ROOT = name
    WAVE_VOL = f"{STRENGTH}/exp/{name}"
    suffix = "modal" if name == WAVE_ROOT_DEFAULT else name
    LOGS_VOL = f"{STRENGTH}/exp/logs_{suffix}"
    FIGS_VOL = f"{STRENGTH}/exp/figs_{suffix}"
    TOOLS_VOL = f"{WAVE_VOL}/sampled/tools"
    LEGS_LOG = STATE_DIR / ("legs.log" if name == WAVE_ROOT_DEFAULT else f"legs_{name}.log")
#: submit_sampled.py's job-name abbreviations: f22s-<TAG>-<abbr><T>
PHASE_ABBR = {"static_base": "sb", "static_esc": "se", "controls": "ctl",
              "ladder_extend": "le", "capability": "cap"}
#: the sbatch header: --cpus-per-task=32 --mem=200G --time=16:00:00
LEG_CPU = 32
LEG_MEMORY_MIB = 196608
LEG_TIMEOUT_S = 16 * 3600
EVIDENCE_NAME = "modal_evidence.json"

try:
    LOCAL_REPO = Path(__file__).resolve().parents[2]
except IndexError:  # Modal re-imports this module flat at /root/strength_served.py
    LOCAL_REPO = Path("/repo")

if modal.is_local():
    # the cpu_suite staging helpers (shallow standalone checkout -> tarball; exact-HEAD + clean
    # verification) are reused, never re-implemented; cpu_suite's import-time dependency-hash
    # guard also holds at this sha (checked before this file was written)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cpu_suite import shallow_clone_tarball, verify_checkout  # noqa: E402


# --------------------------------------------------------------------------- #
# small pure helpers (shared by the local side and the container)              #
# --------------------------------------------------------------------------- #

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def volume_client_path(path: str) -> str:
    """Translate a mounted ``/vol/...`` path to the Volume client's ``/...``."""
    if path == VOL_PATH:
        return "/"
    prefix = f"{VOL_PATH}/"
    if not path.startswith(prefix):
        raise ValueError(f"path is outside {VOL_PATH}: {path}")
    return path[len(VOL_PATH):]


def _git(repo, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-c", f"safe.directory={Path(repo).resolve()}", *args],
                          # 900 s: `git diff --quiet HEAD` over the volume-mounted checkout took > 120 s on 2026-09-15 (the repo grew;
                          # the volume is network I/O) and the preflight refused a tree that had verified clean at stage time
                          cwd=str(repo), capture_output=True, text=True, timeout=900)


def checkout_facts(repo, sha: str) -> dict:
    """Exact HEAD and clean tracked files (the cpu_suite ``verify_checkout`` facts), usable in
    the container where cpu_suite is not importable."""
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    clean = bool(head) and _git(repo, "diff", "--quiet", "HEAD", "--").returncode == 0
    ok = bool(head) and head == sha and clean
    why = None if ok else ("not a git checkout" if not head else
                           f"HEAD {head} != {sha}" if head != sha else "modified tracked files")
    return {"ok": ok, "head": head or None, "clean": clean, "why": why}


def tag_of(key: str, view: str) -> str:
    """common.sh f22_resolve_config: TAG = KEY for the main view, KEY_<view> otherwise."""
    return key if view == "main" else f"{key}_{view}"


def parse_phase(phase: str) -> tuple[str, int, list[str]]:
    """common_sampled.sh f22s_parse_phase, read-only: (kind, trial, families)."""
    spec, _, fams = phase.partition(":")
    for kind in PHASE_ABBR:
        if spec.startswith(kind + "_t") and spec[len(kind) + 2:] in ("0", "1"):
            return kind, int(spec[-1]), [f for f in fams.split(",") if f]
    raise ValueError(f"unknown phase {phase!r} (static_base_t<T>[:fam,...] | static_esc_t<T>[:fam,...] | "
                     "controls_t<T> | ladder_extend_t<T> | capability_t<T>; T in 0,1)")


def job_name_of(tag: str, phase: str) -> str:
    """submit_sampled.py's f22s-<TAG>-<abbr><T>, with a family subset appended."""
    kind, trial, fams = parse_phase(phase)
    name = f"f22s-{tag}-{PHASE_ABBR[kind]}{trial}"
    return name + (("-" + "-".join(fams)) if fams else "")


def expected_app_name(sha: str) -> str:
    return f"strength-served-{sha[:7]}"


SERVING_PATHS = ("lens", "lora_native")


def serving_path_pin(repo_dir, want: str = "lens") -> dict:
    """The ladder's serving ENGINE is the LENS o_proj site on every rung the record carries (the legs ran on trees that had
    no other engine). Main's native-serving default (2026-09-14) would stand a LoRA cell up NATIVELY unless the
    standup is told otherwise, so a leg composed from a tree that knows `--serving-path` pins the engine the caller names
    (`want`: the lens by default — the ladders' engine — or `lora_native` for the base->instruct sweep) through
    `F22_SERVING_PATH` — the EXPLICIT-pin channel `common.sh f22_select_serving_path` honours: an explicit value
    wins, is selected and RECORDED in f22_init, and f22_standup passes `--serving-path` from that selection with the record
    beside it. NOT through `F22_STANDUP_EXTRA` (the first port's channel, 2026-09-16): that hook is appended to
    mt_serving_up.py's argv AFTER --serving-path / --serving-path-selection, so the standup's parser took the later value
    and load_serving_path_selection refused the record ("selected 'lora_native' but this standup serves --serving-path
    lens") — the pin never launched — and common.sh now refuses a --serving-path element there BY NAME at f22_init. A tree
    that knows the flag but whose common.sh does not bind F22_SERVING_PATH (f22_bind_engine) has no channel to carry the pin: REFUSED
    loudly (never the old hook, never both). A tree that predates the flag is lens-only by construction and gets no pin.
    The verdict is recorded in the leg's evidence either way."""
    repo_dir = Path(repo_dir)
    if want not in SERVING_PATHS:
        return {"serving_path": None, "env": {}, "why": f"REFUSE: --serving-path {want!r} is not one of {SERVING_PATHS}"}
    up = repo_dir / "scripts" / "sdfbase" / "mt_serving_up.py"
    common = repo_dir / "scripts" / "sdfbase" / "full22" / "common.sh"
    knows_flag = up.is_file() and '"--serving-path"' in up.read_text(errors="replace")
    common_text = common.read_text(errors="replace") if common.is_file() else ""
    binds_engine = "F22_SERVING_PATH" in common_text and "f22_bind_engine" in common_text
    if want == "lora_native":
        # the NATIVE engine (the base->instruct sweep): the whole view byte for byte through common.sh's explicit
        # F22_SERVING_PATH (it wins over the kind-aware default and binds TARGET_ENGINE=lora_native for every driver)
        if knows_flag and binds_engine:
            return {"serving_path": "lora_native", "env": {"F22_SERVING_PATH": "lora_native"},
                    "why": "the tree binds an explicit F22_SERVING_PATH: the native LoRA engine for the whole view"}
        return {"serving_path": None, "env": {},
                "why": "REFUSE: the native engine needs a tree that knows --serving-path AND binds F22_SERVING_PATH (main since #138)"}
    if knows_flag and binds_engine:
        # main's tree (2026-09-15) SELECTS the path kind-aware in f22_init unless F22_SERVING_PATH is explicit — an argv-level
        # pin through the standup argv hook arrives after that selection and does not bind TARGET_ENGINE (a "lens" leg
        # then ran natively), so the pin is the explicit variable, the whole view byte for byte
        return {"serving_path": "lens", "env": {"F22_SERVING_PATH": "lens"},
                "why": "the tree binds an explicit F22_SERVING_PATH: pinned to the ladder's engine, the lens, for the whole view "
                       "(selected and recorded in f22_init)"}
    if knows_flag:
        # THE ONE CHANNEL (2026-09-16): a tree that knows the flag but binds no F22_SERVING_PATH has nothing that
        # can carry the pin — the standup argv hook is not one (appended after --serving-path / --serving-path-selection the
        # standup's parser takes the later value and load_serving_path_selection refuses the record, and common.sh refuses a
        # --serving-path element there by name at f22_init), so REFUSE loudly rather than guess an engine
        return {"serving_path": None, "env": {},
                "why": "REFUSE: the tree knows --serving-path but its common.sh does not bind F22_SERVING_PATH — no explicit-pin "
                       "channel carries the lens pin (the standup argv hook is not one: the parser takes it after --serving-path "
                       "and common.sh refuses a --serving-path element there by name; never used)"}
    return {"serving_path": "lens", "env": {}, "why": "the tree predates --serving-path: lens-only by construction, nothing to pin"}


def dependency_hashes(tree) -> dict:
    return {name: sha256_file(Path(tree) / name) for name in DEPENDENCY_FILES}


def serving_venv_names(reqs_sha256: str) -> tuple[str, str, str]:
    """common.sh f22_serving_venv: venv `/home/$USER/venvs/sisteer-petri-vllm-<sha[:8]>`,
    sentinel `.<basename>.vllm-bootstrap` holding sha[:16]."""
    venv = f"{VENVS_LOCAL}/sisteer-petri-vllm-{reqs_sha256[:8]}"
    sentinel = f"{VENVS_LOCAL}/.sisteer-petri-vllm-{reqs_sha256[:8]}.vllm-bootstrap"
    return venv, sentinel, reqs_sha256[:16]


# --------------------------------------------------------------------------- #
# the image: the sbatch's two venv recipes, pre-warmed from THIS tree's pins   #
# --------------------------------------------------------------------------- #

if modal.is_local():
    LOCAL_DEP_HASHES = dependency_hashes(LOCAL_REPO)
    _REQS_SHA = LOCAL_DEP_HASHES[VLLM_REQS_REL]
else:
    LOCAL_DEP_HASHES = {}
    _REQS_SHA = "0" * 64
SERVE_VENV, SERVE_SENTINEL, SERVE_SENTINEL_VALUE = serving_venv_names(_REQS_SHA)
BUILD = "/opt/strength-build"

# f22_serving_venv's install line, byte for byte (common.sh), against the SAME requirements file
SERVING_VENV_CMD = (
    f"mkdir -p {VENVS_LOCAL} && uv venv --python {PY_VERSION} {SERVE_VENV} && "
    f"uv pip install --python {SERVE_VENV}/bin/python --index-strategy unsafe-best-match "
    f"--exclude-newer 2026-08-03T00:00:00Z -r {BUILD}/vllm-requirements.txt && "
    f"echo {SERVE_SENTINEL_VALUE} > {SERVE_SENTINEL}"
)
# f22_driver_venv's sync line (common.sh) against a stub project carrying the SAME pyproject + lock, so
# every wheel (and hatchling) is in the cache; the venv itself is thrown away -- the job rebuilds it in
# the staged tree so `sisteer` resolves inside the sha directory, as the cluster's guard requires
DRIVER_PREWARM_CMD = (
    f"mkdir -p {BUILD}/proj/src/sisteer && printf '__version__ = \"0.1.0\"\\n' > {BUILD}/proj/src/sisteer/__init__.py && "
    f"cd {BUILD}/proj && UV_PROJECT_ENVIRONMENT=/opt/prewarm-driver uv sync --frozen --extra cu128 && "
    "/opt/prewarm-driver/bin/python -c 'import torch, pyrit, yaml; print(\"driver prewarm OK\", torch.__version__, pyrit.__version__)' && "
    "rm -rf /opt/prewarm-driver"
)
def serving_venv_check_cmd(venv: str, sentinel: str) -> str:
    """The shell line proving the serving venv: the three pinned versions import, `vllm` is on its
    bin/, the sentinel exists. Used by the image build (local names) and by image_check / run_leg
    in the container (names derived from the requirements sha they are GIVEN -- the container's
    re-import of this module cannot see the local tree, so module constants are placeholders there)."""
    return (
        f"{venv}/bin/python -c \"import vllm, torch, transformers; "
        "assert vllm.__version__.startswith('0.25.1'), vllm.__version__; "
        "assert torch.__version__ == '2.11.0+cu129', torch.__version__; "
        "assert transformers.__version__ == '5.14.1', transformers.__version__; "
        f"print('SERVING VENV OK', vllm.__version__, torch.__version__)\" && test -x {venv}/bin/vllm && cat {sentinel}"
    )


SERVING_VENV_CHECK = serving_venv_check_cmd(SERVE_VENV, SERVE_SENTINEL)

# the CPU side (tree/bank staging, preflight) needs git + yaml only; a slim image keeps staging off the
# GPU image's ~25-minute first build, which `--build` runs in parallel
cpu_image = (
    modal.Image.debian_slim(python_version=PY_VERSION)
    .apt_install("git")
    .pip_install("pyyaml")
    .run_commands("git config --global --add safe.directory '*'")
    .env({"USER": CONTAINER_USER, "LOGNAME": CONTAINER_USER})
)

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python=PY_VERSION)
    .apt_install("git", "build-essential", "curl", "unixodbc", "util-linux", "procps")
    .env({"CUDA_HOME": "/usr/local/cuda", "USER": CONTAINER_USER, "LOGNAME": CONTAINER_USER,
          "UV_CACHE_DIR": UV_CACHE, "UV_LINK_MODE": "copy", "UV_PYTHON_INSTALL_DIR": UV_PYTHONS,
          "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1"})
    .pip_install(f"uv=={UV_PIN}", "pyyaml")
    .run_commands(f"uv python install {PY_VERSION}",
                  "git config --global --add safe.directory '*'")
    .add_local_file(str(LOCAL_REPO / "pyproject.toml"), f"{BUILD}/proj/pyproject.toml", copy=True)
    .add_local_file(str(LOCAL_REPO / "uv.lock"), f"{BUILD}/proj/uv.lock", copy=True)
    .add_local_file(str(LOCAL_REPO / VLLM_REQS_REL), f"{BUILD}/vllm-requirements.txt", copy=True)
    .run_commands(DRIVER_PREWARM_CMD)
    .run_commands(SERVING_VENV_CMD, SERVING_VENV_CHECK)
)


# --------------------------------------------------------------------------- #
# staging: the tree (cpu_suite convention) and the bank                        #
# --------------------------------------------------------------------------- #

@app.function(image=image, timeout=1800, cpu=2, memory=8192)
def image_check(reqs_sha: str) -> dict:
    """Build (or reuse) the GPU image and prove its two venv facts: the serving venv at
    f22_serving_venv's path (derived from the GIVEN requirements sha) with its sentinel, and a
    warm uv cache for f22_driver_venv. Runs without a GPU, so nvidia-smi is absent here by design."""
    venv, sentinel, value = serving_venv_names(reqs_sha)
    r = subprocess.run(["bash", "-c", serving_venv_check_cmd(venv, sentinel)], capture_output=True, text=True, timeout=600)
    cache_entries = sum(1 for _ in Path(UV_CACHE).iterdir()) if Path(UV_CACHE).is_dir() else 0
    sentinel_value = Path(sentinel).read_text().strip() if Path(sentinel).is_file() else None
    return {"serving_venv": venv, "sentinel": sentinel_value, "sentinel_expected": value,
            "check_rc": r.returncode, "check_out": (r.stdout + r.stderr)[-1500:], "uv_cache_entries": cache_entries,
            "uv": subprocess.run(["uv", "--version"], capture_output=True, text=True).stdout.strip(),
            "managed_pythons": sorted(p.name for p in Path(UV_PYTHONS).iterdir()) if Path(UV_PYTHONS).is_dir() else [],
            "tools": {t: shutil.which(t) for t in ("nvidia-smi", "curl", "flock", "git", "bash", "hostname", "uv")},
            "ok": r.returncode == 0 and cache_entries > 0 and sentinel_value == value}


@app.function(image=cpu_image, volumes={VOL_PATH: VOL}, timeout=1800, cpu=4, memory=16384)
def unpack_tree(sha: str, force: bool = False) -> dict:
    """Install a client-uploaded shallow checkout tarball under ``/vol/repos`` (cpu_suite's
    unpack_tree, on this app so the two apps never share a Function)."""
    VOL.reload()
    dst = Path(REPOS) / f"sisteer-{sha}"
    tar_path = Path(REPOS) / f"sisteer-{sha}.tar.gz"
    out = {"sha": sha, "tree": str(dst)}
    if dst.exists() and not force:
        checked = checkout_facts(dst, sha)
        out.update(existing=True, verify=checked)
        if tar_path.exists():
            tar_path.unlink()
            VOL.commit()
        if not checked["ok"]:
            out["error"] = f"existing checkout does not verify: {checked['why']} (use --restage)"
        return out
    if not tar_path.is_file():
        return {**out, "error": f"no uploaded tarball at {tar_path}"}
    tmp = Path(REPOS) / f".strength-unpack-{sha}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(tmp, filter="fully_trusted")
        extracted = tmp / f"sisteer-{sha}"
        if not (extracted / ".git").exists():
            return {**out, "error": "uploaded checkout has no .git directory"}
        if dst.exists():
            shutil.rmtree(dst)
        extracted.rename(dst)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    tar_path.unlink(missing_ok=True)
    checked = checkout_facts(dst, sha)
    out.update(existing=False, verify=checked)
    if not checked["ok"]:
        out["error"] = f"unpacked checkout does not verify: {checked['why']}"
    VOL.commit()
    return out


def bank_entries(bank_root) -> list[dict]:
    """Every view entry under ``<bank>/<target>/cells_view/<view>/`` as recorded by its
    VIEW_MANIFEST.json, with the on-disk facts beside it."""
    out = []
    for vm in sorted(Path(bank_root).glob("*/cells_view/*/VIEW_MANIFEST.json")):
        m = json.loads(vm.read_text())
        for e in m["entries"]:
            p = vm.parent / e["cell_id"]
            rec = {"target": m["target"], "view": m["view"], "cell_id": e["cell_id"],
                   "staged_id": e["staged_id"], "manifest_sha256": e["object_sha256"],
                   "is_symlink": p.is_symlink(), "link_target": os.readlink(p) if p.is_symlink() else None,
                   "resolves": (p / "object.pt").is_file()}
            rec["object_sha256"] = sha256_file(p / "object.pt") if rec["resolves"] else None
            rec["sha_ok"] = rec["object_sha256"] == e["object_sha256"]
            out.append(rec)
    return out


@app.function(image=cpu_image, volumes={VOL_PATH: VOL}, timeout=1800, cpu=4, memory=16384)
def unpack_bank(tar_name: str, sha: str, pins: dict) -> dict:
    """Extract the uploaded bank tarball (``<target>/cells/<staged_id>/``, ``<target>/cells_view/<view>/``
    with its relative symlinks, ``staging_manifest.json``) into ``/vol/strength/objects/sdfbase/full22/``,
    replacing entries of the same name, then verify EVERY view entry: the symlink resolves inside the
    bank, object.pt hashes to the VIEW_MANIFEST's sha, and that sha equals the staged tree's pin."""
    VOL.reload()
    bank = Path(BANK_VOL)
    tar_path = Path(STRENGTH) / tar_name
    if not tar_path.is_file():
        return {"error": f"no uploaded tarball at {tar_path}"}
    tmp = Path(STRENGTH) / f".unpack-bank-{int(time.time())}"
    tmp.mkdir(parents=True)
    moved = []
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(tmp, filter="fully_trusted")
        for target_dir in sorted(p for p in tmp.iterdir() if p.is_dir()):
            for kind in ("cells", "cells_view"):
                src_kind = target_dir / kind
                if not src_kind.is_dir():
                    continue
                for entry in sorted(src_kind.iterdir()):
                    dst = bank / target_dir.name / kind / entry.name
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.is_symlink() or dst.exists():
                        shutil.rmtree(dst) if dst.is_dir() and not dst.is_symlink() else dst.unlink()
                    shutil.move(str(entry), str(dst))
                    moved.append(str(dst.relative_to(bank)))
        sm = tmp / "staging_manifest.json"
        if sm.is_file():
            bank.mkdir(parents=True, exist_ok=True)
            shutil.move(str(sm), str(bank / "staging_manifest.json"))
            moved.append("staging_manifest.json")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        tar_path.unlink(missing_ok=True)
    entries = bank_entries(bank)
    tree_pins_path = Path(REPOS) / f"sisteer-{sha}" / PINS_REL
    tree_pins = json.loads(tree_pins_path.read_text()) if tree_pins_path.is_file() else None
    problems = []
    for e in entries:
        pin = (pins.get(e["target"]) or {}).get(e["staged_id"]) or {}
        e["pin_sha256"] = pin.get("object_sha256")
        e["pin_ok"] = e["pin_sha256"] == e["manifest_sha256"] == e["object_sha256"]
        if tree_pins is not None:
            tp = (tree_pins.get(e["target"]) or {}).get(e["staged_id"]) or {}
            e["tree_pin_ok"] = tp.get("object_sha256") == e["object_sha256"]
        if not (e["is_symlink"] and e["resolves"] and e["sha_ok"] and e["pin_ok"]
                and e.get("tree_pin_ok", True)):
            problems.append({k: e[k] for k in ("target", "view", "cell_id", "is_symlink", "resolves",
                                               "sha_ok", "pin_ok", "tree_pin_ok") if k in e})
    VOL.commit()
    return {"bank": str(bank), "n_moved": len(moved), "n_entries": len(entries),
            "n_views": len({(e["target"], e["view"]) for e in entries}),
            "tree_pins_checked": tree_pins is not None, "problems": problems, "ok": not problems}


@app.function(image=cpu_image, volumes={VOL_PATH: VOL}, timeout=900, cpu=2, memory=8192)
def preflight(sha: str, key: str, view: str, phase: str, expected_dep_hashes: dict, wave_root: str = WAVE_ROOT_DEFAULT) -> dict:
    set_wave_root(wave_root)
    """Every non-GPU precondition of one leg, read from the volume: the staged tree (exact HEAD,
    clean, the image's dependency files), the bank view for (key, view) sha-verified against the
    tree's pins and complete against the roster, the served model snapshot at the config's
    revision, the pinned corpora the drivers read offline, the heldout corpus, the rep_ratio tool.
    A missing input is reported by name; nothing is downloaded."""
    import yaml
    VOL.reload()
    rep = {"sha": sha, "key": key, "view": view, "phase": phase, "ok": True, "checks": {}}

    def chk(name, fn):
        try:
            rep["checks"][name] = {"ok": True, **(fn() or {})}
        except BaseException as e:  # noqa: BLE001 -- every failure named, none hidden
            rep["checks"][name] = {"ok": False, "err": f"{type(e).__name__}: {str(e)[:600]}"}
            rep["ok"] = False

    tree = Path(REPOS) / f"sisteer-{sha}"

    def tree_ok():
        facts = checkout_facts(tree, sha)
        assert facts["ok"], facts
        got = dependency_hashes(tree)
        assert got == expected_dep_hashes, f"staged dependency files differ from the image's: {got} != {expected_dep_hashes}"
        assert (tree / SBATCH_REL).is_file(), SBATCH_REL
        return {"tree": str(tree), **facts, "dependency_hashes": got}
    chk("tree_staged_clean_and_image_matches", tree_ok)

    tag = tag_of(key, view)
    kind, trial, fams = parse_phase(phase)
    cfg_path = tree / f"configs/sdfbase_full22_{tag}_sampled_t{trial}.yaml"
    cfg = {}

    def configs():
        nonlocal cfg
        for p in (cfg_path, tree / f"configs/sdfbase_full22_{tag}_sampled_t{trial}_cap8192.yaml",
                  tree / f"configs/sdfbase_full22_{tag}_sampled_t{trial}_cap16384.yaml",
                  tree / f"configs/sdfbase_full22_{tag}.yaml",
                  tree / f"configs/capeval/manifests/petri_full22_{tag}.yaml"):
            assert p.is_file(), f"missing {p.relative_to(tree)}"
        cfg = yaml.safe_load(cfg_path.read_text())
        return {"config": cfg_path.name, "model": cfg["model"], "model_revision": cfg["model_revision"],
                "explicit_cells": cfg["explicit_cells"], "tag": tag}
    chk("configs_present", configs)

    def bank_view():
        entries = [e for e in bank_entries(BANK_VOL) if e["target"] == key and e["view"] == view]
        assert entries, f"no view {key}/{view} under {BANK_VOL}"
        pins = json.loads((tree / PINS_REL).read_text()).get(key, {})
        bad = [e for e in entries if not (e["is_symlink"] and e["resolves"] and e["sha_ok"]
                                          and (pins.get(e["staged_id"]) or {}).get("object_sha256") == e["object_sha256"])]
        assert not bad, bad
        sys.path.insert(0, str(tree / "scripts/sdfbase/full22"))
        from f22_roster import staged_cells  # noqa: PLC0415 -- the tree's own reader
        roster = yaml.safe_load((tree / ROSTER_REL).read_text())
        want = staged_cells(roster, key, view)
        have = {e["cell_id"]: e["staged_id"] for e in entries}
        assert have == dict(want), f"view {have} != roster staged cells {dict(want)}"
        return {"n_entries": len(entries), "cells": have,
                "shas": {e["cell_id"]: e["object_sha256"][:12] for e in entries}}
    chk("bank_view_sha_verified_and_complete", bank_view)

    def model_snapshot():
        assert cfg, "config not loaded"
        snap = Path(HF_VOL) / "hub" / f"models--{cfg['model'].replace('/', '--')}" / "snapshots" / cfg["model_revision"]
        assert snap.is_dir(), f"no snapshot at {snap}"
        files = [p for p in snap.iterdir()]
        st = [p for p in files if p.name.endswith(".safetensors")]
        assert st, f"no safetensors under {snap}"
        assert (snap / "config.json").is_file() and (snap / "tokenizer_config.json").is_file(), sorted(p.name for p in files)
        return {"snapshot": str(snap), "n_files": len(files), "n_safetensors": len(st),
                "gb": round(sum(p.resolve().stat().st_size for p in st) / 1e9, 2)}
    chk("model_snapshot_at_pinned_revision", model_snapshot)

    def corpora():
        pins = yaml.safe_load((tree / DATASET_PINS_REL).read_text())["datasets"]
        found = {}
        for name in ("xstest_v2", "or_bench"):
            e = pins[name]
            snap = Path(HF_VOL) / "hub" / f"datasets--{e['repo_id'].replace('/', '--')}" / "snapshots" / e["revision"]
            files = sorted(str(p.relative_to(snap)) for p in snap.rglob("*") if p.is_file())
            assert files, f"{name}: no files at {snap}"
            found[name] = {"repo_id": e["repo_id"], "revision": e["revision"][:12], "files": files}
        # the static families' goal sets (jailbreak_data.py pins) and renellm's replay corpus -- the
        # constants read by regex from the staged file (importing sisteer.eval would pull torch)
        import re
        src = (tree / "src/sisteer/eval/jailbreak_data.py").read_text()

        def const(name):
            m = re.search(rf'^{name}\s*=\s*"([^"]+)"', src, re.M)
            assert m, f"{name} not found in jailbreak_data.py"
            return m.group(1)
        jbb_harmful = re.search(r'"harmful":\s*"([^"]+)"', src).group(1)
        for label, repo, rev, file in (("harmbench", const("HARMBENCH_REPO"), const("HARMBENCH_REVISION"), None),
                                       ("jbb", const("JBB_REPO"), const("JBB_REVISION"), jbb_harmful),
                                       ("xstest_paul", const("XSTEST_REPO"), const("XSTEST_REVISION"), const("XSTEST_FILE"))):
            snap = Path(HF_VOL) / "hub" / f"datasets--{repo.replace('/', '--')}" / "snapshots" / rev
            assert snap.is_dir(), f"{label}: no snapshot {snap}"
            if file:
                assert (snap / file).is_file(), f"{label}: {file} missing under {snap}"
            found[label] = {"repo_id": repo, "revision": rev[:12], "present": True}
        for f in ("renellm.jsonl", "renellm.jsonl.manifest.json"):
            assert (Path(CORPORA_VOL) / f).is_file(), f"heldout corpus {f} missing under {CORPORA_VOL}"
        found["heldout_corpora"] = sorted(p.name for p in Path(CORPORA_VOL).iterdir())
        return found
    chk("corpora_pinned_snapshots_present", corpora)

    def tools():
        p = Path(TOOLS_VOL) / "full22_join_scores.py"
        assert p.is_file(), f"rep_ratio source missing at {p} (--stage-tools)"
        got = sha256_file(p)
        assert got == REP_RATIO_SHA256, f"{p} sha {got} != pinned {REP_RATIO_SHA256}"
        return {"path": str(p), "sha256": got[:16]}
    chk("rep_ratio_tool_pinned", tools)

    def venv_names():
        # the GPU image's venv facts themselves are proven by `--build` (image_check) and re-asserted
        # at the start of run_leg; here only the NAME derivation against the staged requirements
        h = dependency_hashes(tree)[VLLM_REQS_REL]
        venv, sentinel, value = serving_venv_names(h)
        return {"serving_venv": venv, "sentinel": sentinel, "sentinel_value": value}
    chk("serving_venv_name_from_staged_requirements", venv_names)

    prior = sorted(str(p.relative_to(WAVE_VOL)) for p in Path(WAVE_VOL).glob(f"sampled/*/{tag}_t*")) if Path(WAVE_VOL).is_dir() else []
    rep["prior_modal_stores_for_tag"] = prior
    return rep


@app.function(image=cpu_image, gpu="H200", timeout=600, cpu=2, memory=8192)
def gpu_probe(hold_s: int = 60) -> dict:
    """Phase 0 (d): one short H200 container -- when N of these are spawned together, the overlap of
    their [start, end] windows measures how many H200 containers the workspace runs at once."""
    t0 = time.time()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    time.sleep(hold_s)
    return {"task_id": os.environ.get("MODAL_TASK_ID"), "gpu": gpu, "start": t0, "end": time.time()}


def probe_concurrency(n: int, hold_s: int = 60) -> dict:
    calls = [gpu_probe.spawn(hold_s) for _ in range(n)]
    t_spawn = time.time()
    results = [c.get() for c in calls]
    starts = sorted(r["start"] for r in results)
    ends = sorted(r["end"] for r in results)
    # the maximum number of probes alive at any instant
    peak, alive, i, j = 0, 0, 0, 0
    while i < len(starts):
        if j < len(ends) and ends[j] <= starts[i]:
            alive -= 1; j += 1
        else:
            alive += 1; i += 1; peak = max(peak, alive)
    rep = {"n_spawned": n, "hold_s": hold_s, "peak_concurrent": peak,
           "first_start_after_spawn_s": round(starts[0] - t_spawn, 1),
           "last_start_after_spawn_s": round(starts[-1] - t_spawn, 1),
           "distinct_gpus": len({r["gpu"] for r in results}), "gpus": sorted({r["gpu"] for r in results})[:3]}
    print(json.dumps(rep, indent=1))
    return rep


# --------------------------------------------------------------------------- #
# the leg                                                                      #
# --------------------------------------------------------------------------- #

class TreeSyncer:
    """Incremental copy of a local tree to a volume path: a file is re-shipped only when its
    (size, mtime_ns) differs from what THIS process last shipped (the grokmt RunSyncer)."""

    def __init__(self, src, dest) -> None:
        self.src, self.dest = Path(src), Path(dest)
        self.seen: dict[str, tuple[int, int]] = {}

    def mark_seen(self) -> None:
        """Record the current signatures WITHOUT shipping (seeded files came FROM the volume)."""
        for p in self.src.rglob("*"):
            if p.is_file() and not p.is_symlink():
                st = p.stat()
                self.seen[str(p.relative_to(self.src))] = (st.st_size, st.st_mtime_ns)

    def copy_changed(self) -> int:
        n = 0
        if not self.src.is_dir():
            return 0
        for p in self.src.rglob("*"):
            if p.is_symlink() or not p.is_file():
                continue
            try:
                st = p.stat()
            except FileNotFoundError:
                continue
            rel = str(p.relative_to(self.src))
            sig = (st.st_size, st.st_mtime_ns)
            if self.seen.get(rel) == sig:
                continue
            q = self.dest / rel
            q.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copyfile(p, q)
            except FileNotFoundError:
                continue
            self.seen[rel] = sig
            n += 1
        return n


def _sync_all(syncers, lock, label: str) -> None:
    with lock:
        try:
            n = sum(s.copy_changed() for s in syncers)
            VOL.commit()
            print(f"[sync {label}] {n} file(s) committed", flush=True)
        except Exception as e:  # noqa: BLE001 -- a sync failure is logged, the leg goes on; the final sync re-ships
            print(f"[sync {label}] WARNING: {type(e).__name__}: {e}", flush=True)


def _gpu_facts() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=30).stdout.strip()
        name, uuid, drv, mem = [s.strip() for s in out.split(",", 3)]
        return {"name": name, "uuid": uuid, "driver_version": drv, "memory_total": mem}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def _seed_from_volume(wave_local: Path, tag: str) -> list[str]:
    """Copy the volume's prior stores of this TAG (both trials) into the local wave root, so a
    later phase reads them and the shared ladder events file accumulates across legs."""
    wave_vol = Path(WAVE_VOL)
    if not wave_vol.is_dir():
        return []
    seeded = []
    globs = [f"sampled/ladders/{tag}_t*", f"sampled/controls_cap*/{tag}_t*",
             f"sampled/gsm8k_thinking_on_cap*/{tag}_t*", f"sampled/mmlu/{tag}_t*",
             f"transcripts_results/sdfbase_full22_{tag}_sampled_t*", f"sampled/serving/*_{tag}_*",
             "sampled/tools"]
    for g in globs:
        for src in sorted(wave_vol.glob(g)):
            if not src.is_dir():
                continue
            dst = wave_local / src.relative_to(wave_vol)
            shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)
            seeded.append(str(src.relative_to(wave_vol)))
    return seeded


def _hf_env_sh() -> str:
    """The `$SHARED/scripts/hf_env.sh` f22_init requires: the cluster helper's contract (HF_HOME +
    a per-user token path that does not exist), pointed at the volume's read-only cache."""
    return (
        "#!/bin/bash\n"
        "# strength-served (Modal): the per-cluster helper f22_init sources. HF_HOME is the volume's shared,\n"
        "# read-only cache (models + pinned datasets); no token exists here, on purpose.\n"
        f"export HF_HOME={HF_VOL}\n"
        f"export HF_TOKEN_PATH={SHARED_LOCAL}/hf_token\n"
        f"export HUGGINGFACE_TOKEN_PATH={SHARED_LOCAL}/hf_token\n"
    )


def parse_leg_env(spec: str) -> dict:
    """``--leg-env``: extra environment for every leg of a launch, ``KEY=VALUE`` pairs separated by ``;``
    (a value may hold commas and ``=``; the first ``=`` splits). Meant for the tree's argv hooks — e.g.
    ``F22_STANDUP_EXTRA=--max-num-seqs=48`` clamps the escalation standup's width to a CUDA-graph capture
    size. Applied AFTER the serving-path pin and recorded in the leg's evidence; an empty spec is no-op."""
    out: dict = {}
    for part in [x.strip() for x in str(spec or "").split(";") if x.strip()]:
        if "=" not in part:
            raise SystemExit(f"[strength] --leg-env: {part!r} is not KEY=VALUE")
        k, v = part.split("=", 1)
        if not k.strip() or not k.strip().isidentifier():
            raise SystemExit(f"[strength] --leg-env: {k!r} is not an environment variable name")
        out[k.strip()] = v
    return out


def require_cluster_label(leg_env: str) -> None:
    """Refuse a launch, before any GPU, whose ``--leg-env`` does not name ``SISTEER_CLUSTER_LABEL``. The wave shell records
    the label as the run's substrate and resolves each arm's servability by it, and this app has no site config to read it
    from, so a leg without it would stop inside its paid GPU container. (The shared root and the checkouts' root the shell
    also reads are the assembled container tree's own, set in the leg; --leg-env may still override them.)"""
    if not parse_leg_env(leg_env).get("SISTEER_CLUSTER_LABEL"):
        raise SystemExit("[strength] --leg-env must name SISTEER_CLUSTER_LABEL=<label> (the site's label of the cluster "
                         "whose roster and arms this leg mirrors, as the front door passes it from the site config); "
                         "refusing before any GPU")


def local_wave_dir_name(serving_path: str) -> str:
    """The directory the TREE writes its wave root under `$SHARED/exp/` — main's common.sh (#138) derives it from the serving
    path: `sdfbase_full22` on the lens engine, `sdfbase_full22_native` on the native engine (a tree that predates the split
    always writes `sdfbase_full22`, and is lens-only). The port's syncer must watch THIS directory or every native store is
    lost with the container — measured 2026-09-15: 36 native legs synced logs but no stores."""
    return "sdfbase_full22" if serving_path == "lens" else "sdfbase_full22_native"


def _build_shared(wave_sha: str, tag: str, serving_path: str = "lens") -> dict:
    """Assemble /workspace/<user> so `f22_init` resolves every path unmodified."""
    shared = Path(SHARED_LOCAL)
    shutil.rmtree(shared, ignore_errors=True)
    (shared / "scripts").mkdir(parents=True)
    (shared / "scripts" / "hf_env.sh").write_text(_hf_env_sh())
    # the tree: a COPY of the staged checkout (git-verified below), at the path the sha-guard reads
    tree_src = Path(REPOS) / f"sisteer-{wave_sha}"
    repo_dir = shared / "si-steering-eval-repo" / wave_sha
    if not (tree_src / ".git").exists():
        raise RuntimeError(f"{tree_src} is not a staged checkout (run --stage)")
    shutil.copytree(tree_src, repo_dir, symlinks=True)
    facts = checkout_facts(repo_dir, wave_sha)
    if not facts["ok"]:
        raise RuntimeError(f"copied tree does not verify: {facts}")
    # the bank: a COPY (symlinks preserved) so make_bank_view --check and its lock run on local disk
    exp = shared / "exp"
    exp.mkdir()
    shutil.copytree(Path(STRENGTH) / "objects", exp / "objects", symlinks=True)
    # the wave root: container-local, named as the TREE names it for this serving path (local_wave_dir_name), seeded from the
    # volume's DISTINCT root for this wave (WAVE_VOL) and synced back to it
    wave_local = exp / local_wave_dir_name(serving_path)
    wave_local.mkdir()
    seeded = _seed_from_volume(wave_local, tag)
    os.symlink(CORPORA_VOL, exp / "heldout_corpora", target_is_directory=True)
    (exp / "figs").mkdir()
    (exp / "logs").mkdir()
    (shared / ".cache").mkdir()
    os.symlink(UV_CACHE, shared / ".cache" / "uv", target_is_directory=True)
    os.symlink(UV_PYTHONS, shared / ".uv-pythons", target_is_directory=True)
    Path(VENVS_LOCAL).mkdir(parents=True, exist_ok=True)
    return {"repo_dir": str(repo_dir), "wave_local": str(wave_local), "tree": facts, "seeded": seeded,
            "path_map": {
                f"{SHARED_LOCAL}/si-steering-eval-repo/{wave_sha}": f"copy of {tree_src}",
                f"{SHARED_LOCAL}/exp/objects": f"copy of {STRENGTH}/objects",
                f"{SHARED_LOCAL}/exp/{local_wave_dir_name(serving_path)}": f"container-local (the tree's wave root for serving path {serving_path}); synced to {WAVE_VOL}",
                f"{SHARED_LOCAL}/exp/logs": f"container-local; synced to {LOGS_VOL}",
                f"{SHARED_LOCAL}/exp/figs": f"container-local; synced to {FIGS_VOL}",
                f"{SHARED_LOCAL}/exp/heldout_corpora": f"-> {CORPORA_VOL}",
                f"{SHARED_LOCAL}/scripts/hf_env.sh": f"written: HF_HOME={HF_VOL}",
                f"{SHARED_LOCAL}/.cache/uv": f"-> {UV_CACHE} (image, pre-warmed)",
                f"{VENVS_LOCAL}": "image (serving venv + sentinel) / built by f22_driver_venv at run time",
            }}


def _controls_summary(wave_local: Path, tag: str, trial: int) -> dict:
    """Per rung root of this (TAG, trial): cell -> control -> {n_total, n_at_cap} from gen_stats.json."""
    out = {}
    for root in sorted(wave_local.glob(f"sampled/controls_cap*/{tag}_t{trial}")):
        cells = {}
        for cd in sorted((root / "cells").glob("*")):
            try:
                cid = json.loads((cd / "manifest.json").read_text()).get("cell_id")
                gs = json.loads((cd / "gen_stats.json").read_text())
            except (OSError, ValueError):
                continue
            cells[cid] = {c: {"n_total": v.get("n_total"), "n_at_cap": v.get("n_at_cap")}
                          for c, v in gs.items() if isinstance(v, dict)}
        out[root.parent.name] = cells
    return out


@app.function(image=image, volumes={VOL_PATH: VOL}, gpu="H200", cpu=LEG_CPU, memory=LEG_MEMORY_MIB,
              timeout=LEG_TIMEOUT_S, retries=0)
def run_leg(key: str, view: str, phase: str, trial: int, wave_sha: str, nshards: int = 9,
            cells: str = "", app_name: str = "", expected_dep_hashes: dict | None = None,
            sync_every_s: int = 60, max_hours: float = 15.5, serving_path: str = "lens",
            wave_root: str = WAVE_ROOT_DEFAULT, leg_env: str = "") -> dict:
    """One (key, view, phase) leg: build the cluster-shaped tree, run target_sampled.sbatch
    under bash with the three Slurm variables it reads, sync the store to the volume."""
    set_wave_root(wave_root)
    t0 = time.time()
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    kind, phase_trial, fams = parse_phase(phase)
    if phase_trial != int(trial):
        raise ValueError(f"--trial {trial} disagrees with the phase's _t{phase_trial} ({phase})")
    tag = tag_of(key, view)
    job_name = job_name_of(tag, phase)
    task_id = os.environ.get("MODAL_TASK_ID", "")
    call_id = modal.current_function_call_id() or ""
    job_id = "modal-" + (task_id.replace("ta-", "") or call_id.replace("fc-", ""))[:16].lower()
    hostname = socket.gethostname()
    VOL.reload()
    print(f"[leg] {tag} {phase} sha={wave_sha[:12]} job_id={job_id} task={task_id} call={call_id} host={hostname}", flush=True)
    # the image's half of the two venv recipes, before any tree is built (f22_serving_venv would
    # otherwise rebuild vLLM at run time -- faithful, but 10 minutes of GPU idle per leg). The names
    # derive from the requirements sha the caller passed: module constants are placeholders here.
    if not expected_dep_hashes or VLLM_REQS_REL not in expected_dep_hashes:
        return {"status": "NO_DEPENDENCY_HASHES", "tag": tag, "phase": phase,
                "error": "run_leg needs expected_dep_hashes (the launcher passes its tree's three file hashes)"}
    serve_venv, serve_sentinel, serve_value = serving_venv_names(expected_dep_hashes[VLLM_REQS_REL])
    if not (Path(serve_venv, "bin", "vllm").exists() and Path(serve_sentinel).is_file()
            and Path(serve_sentinel).read_text().strip() == serve_value
            and Path(UV_CACHE).is_dir() and any(Path(UV_CACHE).iterdir())):
        return {"status": "IMAGE_UNPREPARED", "tag": tag, "phase": phase,
                "error": f"serving venv {serve_venv} / sentinel {serve_sentinel} / uv cache {UV_CACHE} not as the image build promised"}

    built = _build_shared(wave_sha, tag, serving_path)
    repo_dir = Path(built["repo_dir"])
    wave_local = Path(built["wave_local"])
    if expected_dep_hashes:
        got = dependency_hashes(repo_dir)
        if got != expected_dep_hashes:
            return {"status": "DEPENDENCY_MISMATCH", "tag": tag, "phase": phase, "expected": expected_dep_hashes,
                    "staged": got, "error": "staged pyproject.toml/uv.lock/vllm-requirements differ from the image's"}

    logs_local = Path(SHARED_LOCAL) / "exp" / "logs"
    figs_local = Path(SHARED_LOCAL) / "exp" / "figs"
    log_path = logs_local / f"{job_name}_{job_id}.out"          # the cluster's %x_%j.out
    syncers = [TreeSyncer(wave_local, WAVE_VOL), TreeSyncer(logs_local, LOGS_VOL), TreeSyncer(figs_local, FIGS_VOL)]
    syncers[0].mark_seen()                                       # the seeded files came from the volume
    lock = threading.Lock()
    stop = threading.Event()

    def sync_loop():
        while not stop.wait(sync_every_s):
            _sync_all(syncers, lock, "periodic")
    threading.Thread(target=sync_loop, daemon=True, name="tree-sync").start()

    env = dict(os.environ)
    for k in ("HF_HOME", "HF_HUB_OFFLINE", "UV_PROJECT_ENVIRONMENT", "VIRTUAL_ENV", "PYTHONPATH"):
        env.pop(k, None)
    env.update({"USER": CONTAINER_USER, "LOGNAME": CONTAINER_USER, "HOME": "/root",
                "CUDA_VISIBLE_DEVICES": "0",                  # Slurm's export for a 1-GPU job; the sbatch READS it
                "SLURM_JOB_ID": job_id, "SLURM_JOB_NAME": job_name})
    pin = serving_path_pin(repo_dir, serving_path)
    if pin["why"].startswith("REFUSE"):
        return {"status": "SERVING_PATH_UNPINNABLE", "tag": tag, "phase": phase, "error": pin["why"]}
    env.update(pin["env"])                                        # F22_SERVING_PATH=<engine> for a native-default tree (read by f22_init)
    # the wave shell reads the shared root and the checkouts' root from the job env (on a cluster, the front door's site
    # values); here they are the assembled container tree's own, and --leg-env (applied next) still wins
    env.setdefault("SISTEER_SHARED_ROOT", SHARED_LOCAL)
    env.setdefault("SISTEER_EVAL_REPO_ROOT", str(Path(repo_dir).parent))
    extra_env = parse_leg_env(leg_env)                            # The launch's per-leg env (argv hooks), after the pin
    env.update(extra_env)
    print(f"[leg] serving path: {pin['serving_path']} — {pin['why']}", flush=True)
    argv = ["bash", str(repo_dir / SBATCH_REL), key, phase, wave_sha, str(int(nshards)), cells or "", view]
    evidence = {
        "schema": "strength_served/modal_evidence/1", "hardware": "modal",
        "modal": {"app_name": app_name, "function_call_id": call_id, "task_id": task_id, "hostname": hostname,
                  "image_id": os.environ.get("MODAL_IMAGE_ID"), "gpu": _gpu_facts(),
                  "cpu": LEG_CPU, "memory_mib": LEG_MEMORY_MIB, "timeout_s": LEG_TIMEOUT_S,
                  "serving_venv": serve_venv, "serving_venv_sentinel": serve_value, "uv_cache": UV_CACHE},
        "wave_sha": wave_sha, "key": key, "view": view, "tag": tag, "phase": phase, "trial": int(trial),
        "nshards": int(nshards), "cells_arg": cells or "", "argv": argv,
        "env_overrides": {k: env[k] for k in ("USER", "CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID", "SLURM_JOB_NAME") + tuple(pin["env"]) + tuple(extra_env)},
        "leg_env": extra_env,
        "serving_path_pin": pin,
        "path_map": built["path_map"], "tree": built["tree"], "seeded_from_volume": built["seeded"],
        "labels_note": ("standup_jobs.jsonl `cluster` reads 'cluster_b' because common.sh f22_init derives it "
                        "from the /workspace/$USER path, and `job_id`/`node` are this record's SLURM_JOB_ID / "
                        "the container hostname; the hardware is the Modal H200 named under `modal.gpu`"),
        "wave_root_on_volume": WAVE_VOL, "wave_root": WAVE_ROOT, "started_at": started_at,
    }
    print("[leg] argv:", " ".join(argv), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as fh:
        proc = subprocess.Popen(argv, cwd=str(repo_dir), env=env, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        deadline = t0 + max_hours * 3600
        rc = None
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if time.time() > deadline:
                print(f"[leg] max_hours {max_hours} reached: SIGTERM to the job's process group (the sbatch traps it)", flush=True)
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    rc = proc.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    rc = proc.wait()
                evidence["killed_at_max_hours"] = True
                break
            time.sleep(15)
    stop.set()
    elapsed = round(time.time() - t0, 1)
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    evidence.update(rc=rc, elapsed_s=elapsed, finished_at=finished_at,
                    job_log=f"{LOGS_VOL}/{log_path.name}")
    result_file = figs_local / job_name / "RESULT"
    evidence["RESULT"] = result_file.read_text().strip().splitlines()[-3:] if result_file.is_file() else None

    # the additive evidence: the serving dir + every store root this leg's job id / trial owns
    serve_dirs = sorted(wave_local.glob(f"sampled/serving/*_{tag}_{job_id}"))
    store_roots = sorted(wave_local.glob(f"sampled/controls_cap*/{tag}_t{trial}")) \
        + sorted(wave_local.glob(f"sampled/gsm8k_thinking_on_cap*/{tag}_t{trial}")) \
        + sorted(wave_local.glob(f"sampled/mmlu/{tag}_t{trial}")) \
        + sorted(wave_local.glob(f"transcripts_results/sdfbase_full22_{tag}_sampled_t{trial}*")) \
        + sorted(wave_local.glob(f"sampled/ladders/{tag}_t{trial}"))
    written = []
    for d in serve_dirs + store_roots:
        if d.is_dir():
            p = d / EVIDENCE_NAME
            # a store root touched by several legs keeps one record per leg (append-only list)
            prior = json.loads(p.read_text()) if p.is_file() else []
            prior = prior if isinstance(prior, list) else [prior]
            prior.append(evidence)
            p.write_text(json.dumps(prior, indent=1, default=str))
            written.append(str(d.relative_to(wave_local)))
    _sync_all(syncers, lock, "final")
    summary = {
        "status": "DONE" if rc == 0 else "FAILED", "rc": rc, "elapsed_s": elapsed, "tag": tag, "key": key,
        "view": view, "phase": phase, "trial": int(trial), "job_id": job_id, "job_name": job_name,
        "task_id": task_id, "function_call_id": call_id, "hostname": hostname, "gpu": evidence["modal"]["gpu"],
        "job_log": evidence["job_log"], "serve_dirs": [f"{WAVE_VOL}/{d.relative_to(wave_local)}" for d in serve_dirs],
        "store_roots": [f"{WAVE_VOL}/{d.relative_to(wave_local)}" for d in store_roots],
        "evidence_written_to": written, "RESULT": evidence["RESULT"],
        "controls": _controls_summary(wave_local, tag, int(trial)) if kind == "controls" else None,
        "seeded_from_volume": built["seeded"],
    }
    tail = log_path.read_text(errors="replace")[-6000:] if log_path.is_file() else ""
    print("[leg] SUMMARY", json.dumps(summary, default=str), flush=True)
    print("[leg] LOG TAIL\n" + tail, flush=True)
    summary["log_tail"] = tail[-3000:]
    return summary


# --------------------------------------------------------------------------- #
# local side: staging, preflight, launch, watch, fan-out, pull                 #
# --------------------------------------------------------------------------- #

def _record_call(rec: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with CALLS_FILE.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _load_calls() -> list[dict]:
    if not CALLS_FILE.is_file():
        return []
    out = []
    for line in CALLS_FILE.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def _leg_line(summary: dict, *, error: str | None = None) -> str:
    status = summary.get("status", "FAILED") if error is None else "FAILED"
    el = summary.get("elapsed_s")
    el_s = f"{int(el // 60)}m{int(el % 60):02d}s" if isinstance(el, (int, float)) else "?"
    line = f"{status} {summary.get('tag', '?')} {summary.get('phase', '?')} {el_s} job={summary.get('job_id', '?')} rc={summary.get('rc')}"
    if error:
        line += f" error={error[:200]!r}"
    return line


def _append_leg_log(line: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    with LEGS_LOG.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {line}\n")
    print(line, flush=True)


def _poll_call(call_id: str):
    """(done, summary, error) for one spawned call, without blocking."""
    from modal.exception import OutputExpiredError
    fc = modal.FunctionCall.from_id(call_id)
    try:
        return True, fc.get(timeout=0), None
    except OutputExpiredError as e:      # a TimeoutError subclass, but TERMINAL: the result is gone
        return True, None, f"OutputExpiredError: {e} (read the leg's job log on the volume)"
    except TimeoutError:
        return False, None, None
    except Exception as e:  # noqa: BLE001 -- the remote exception IS the leg's failure
        return True, None, f"{type(e).__name__}: {e}"


def do_stage_tree(sha: str, restage: bool) -> dict:
    with tempfile.TemporaryDirectory(prefix="strength-tree-") as td:
        tar = shallow_clone_tarball(sha, Path(td))
        print(f"[stage] pushing {tar.name} ({tar.stat().st_size // 1_000_000} MB) -> {REPOS}/", flush=True)
        with VOL.batch_upload(force=True) as batch:
            batch.put_file(str(tar), f"{volume_client_path(REPOS)}/{tar.name}")
        staged = unpack_tree.remote(sha, force=restage)
    print(json.dumps(staged, indent=1))
    if staged.get("error"):
        raise SystemExit(f"[stage] tree staging failed: {staged['error']}")
    return staged


def _staged(spec: str) -> Path:
    """A staging path on the grading host: ``spec`` with ``$STAGE`` expanded through sisteer.site (refused by name when the
    staging directory is unset)."""
    if str(LOCAL_REPO / "src") not in sys.path:
        sys.path.insert(0, str(LOCAL_REPO / "src"))
    from sisteer import site as _site
    return _site.expand_path(spec)


def do_stage_bank(sha: str) -> dict:
    bank_local = _staged(BANK_LOCAL)
    if not bank_local.is_dir():
        raise SystemExit(f"[stage-bank] no local bank staging at {bank_local}")
    pins = json.loads((LOCAL_REPO / PINS_REL).read_text())
    local_entries = bank_entries(bank_local)
    bad = [e for e in local_entries if not (e["is_symlink"] and e["resolves"] and e["sha_ok"]
                                            and (pins.get(e["target"]) or {}).get(e["staged_id"], {}).get("object_sha256") == e["object_sha256"])]
    if bad:
        raise SystemExit(f"[stage-bank] local bank refuses before upload: {bad}")
    print(f"[stage-bank] local bank OK: {len(local_entries)} entries in "
          f"{len({(e['target'], e['view']) for e in local_entries})} views, every sha == VIEW_MANIFEST == pin", flush=True)
    tar_name = f"bank-{int(time.time())}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="strength-bank-") as td:
        tar = Path(td) / tar_name
        with tarfile.open(tar, "w:gz") as tf:
            for p in sorted(bank_local.iterdir()):
                tf.add(p, arcname=p.name)      # symlinks stay symlinks (dereference=False)
        print(f"[stage-bank] pushing {tar.name} ({tar.stat().st_size // 1_000_000} MB) -> {STRENGTH}/", flush=True)
        with VOL.batch_upload(force=True) as batch:
            batch.put_file(str(tar), f"{volume_client_path(STRENGTH)}/{tar_name}")
    rep = unpack_bank.remote(tar_name, sha, pins)
    print(json.dumps(rep, indent=1))
    if not rep.get("ok"):
        raise SystemExit("[stage-bank] the bank on the volume does not verify")
    return rep


def do_stage_tools() -> None:
    rep_ratio_local = _staged(REP_RATIO_LOCAL)
    if not rep_ratio_local.is_file():
        raise SystemExit(f"[stage-tools] no rep_ratio source at {rep_ratio_local}")
    got = sha256_file(rep_ratio_local)
    if got != REP_RATIO_SHA256:
        raise SystemExit(f"[stage-tools] {rep_ratio_local} sha {got} != pinned {REP_RATIO_SHA256}")
    with VOL.batch_upload(force=True) as batch:
        batch.put_file(str(rep_ratio_local), f"{volume_client_path(TOOLS_VOL)}/full22_join_scores.py")
    print(f"[stage-tools] {rep_ratio_local.name} -> {TOOLS_VOL}/ (sha {got[:16]})", flush=True)


def spawn_leg(sha: str, app_name: str, key: str, view: str, phase: str, trial: int, nshards: int,
              cells: str, sync_every_s: int, max_hours: float, serving_path: str = "lens", leg_env: str = "") -> str:
    parse_phase(phase)
    if serving_path not in SERVING_PATHS:
        raise SystemExit(f"[strength] --serving-path must be one of {SERVING_PATHS}, got {serving_path!r}")
    fc = run_leg.spawn(key, view, phase, int(trial), sha, int(nshards), cells, app_name, LOCAL_DEP_HASHES,
                       int(sync_every_s), float(max_hours), serving_path, WAVE_ROOT, leg_env)
    rec = {"call_id": fc.object_id, "app": app_name, "sha": sha, "key": key, "view": view, "tag": tag_of(key, view),
           "phase": phase, "trial": int(trial), "serving_path": serving_path, "wave_root": WAVE_ROOT,
           "spawned_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _record_call(rec)
    _append_leg_log(f"SPAWNED {rec['tag']} {phase} call={fc.object_id} serving_path={serving_path}")
    return fc.object_id


def watch_calls(poll_s: int = 60) -> None:
    """Poll every recorded, not-yet-reported call; append one DONE|FAILED line per leg to legs.log."""
    reported = set()
    if LEGS_LOG.is_file():
        for line in LEGS_LOG.read_text().splitlines():
            if " call=" in line and (" DONE " in line or " FAILED " in line):
                reported.add(line.rsplit("call=", 1)[-1].split()[0])
    while True:
        pending = [c for c in _load_calls() if c["call_id"] not in reported
                   and c.get("wave_root", WAVE_ROOT_DEFAULT) == WAVE_ROOT]       # one ledger per wave root
        if not pending:
            print("[watch] nothing pending", flush=True)
            return
        for c in pending:
            done, summary, err = _poll_call(c["call_id"])
            if not done:
                continue
            summary = summary or {"tag": c["tag"], "phase": c["phase"]}
            _append_leg_log(_leg_line(summary, error=err) + f" call={c['call_id']}")
            reported.add(c["call_id"])
        time.sleep(poll_s)


def run_legs(sha: str, app_name: str, legs_file: str, max_concurrent: int, nshards: int, sync_every_s: int,
             max_hours: float, poll_s: int = 60, serving_path: str = "lens", leg_env: str = "") -> None:
    """Phase-2 fan-out: `key view phase trial` per line; at most max_concurrent containers; never two
    legs of one (TAG, trial) at once (they share ladders/<TAG>_t<T>/events.jsonl on the cluster)."""
    legs = []
    for line in Path(legs_file).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        key, view, phase, trial = line.split()
        parse_phase(phase)
        legs.append({"key": key, "view": view, "phase": phase, "trial": int(trial), "tag": tag_of(key, view)})
    running: dict[str, dict] = {}
    pending = list(legs)
    print(f"[legs] {len(legs)} legs, <= {max_concurrent} concurrent", flush=True)
    while pending or running:
        busy = {(r["tag"], r["trial"]) for r in running.values()}
        for leg in list(pending):
            if len(running) >= max_concurrent:
                break
            if (leg["tag"], leg["trial"]) in busy:
                continue
            cid = spawn_leg(sha, app_name, leg["key"], leg["view"], leg["phase"], leg["trial"], nshards, "",
                            sync_every_s, max_hours, serving_path=serving_path, leg_env=leg_env)
            running[cid] = {**leg, "t0": time.time()}
            busy.add((leg["tag"], leg["trial"]))
            pending.remove(leg)
        time.sleep(poll_s)
        for cid, leg in list(running.items()):
            done, summary, err = _poll_call(cid)
            if done:
                summary = summary or {"tag": leg["tag"], "phase": leg["phase"], "elapsed_s": time.time() - leg["t0"]}
                _append_leg_log(_leg_line(summary, error=err) + f" call={cid}")
                del running[cid]
    print("[legs] all legs finished", flush=True)


def pull_tag(tag: str, dest: str) -> None:
    """Fetch every store, ladder, serving dir and log of one TAG from the volume, preserving layout
    (via the modal CLI's recursive `volume get`)."""
    dest_p = Path(dest)
    dest_p.mkdir(parents=True, exist_ok=True)
    cli = [sys.executable, "-m", "modal", "volume"]
    ls = subprocess.run([*cli, "ls", VOL_NAME, volume_client_path(WAVE_VOL), "--json"], capture_output=True, text=True)
    if ls.returncode != 0:
        raise SystemExit(f"[pull] cannot list {WAVE_VOL}: {ls.stderr[-500:]}")
    wanted = []
    for sub in ("sampled/controls_cap512", "sampled/controls_cap2048", "sampled/controls_cap8192", "sampled/controls_cap16384",
                "sampled/gsm8k_thinking_on_cap8192", "sampled/gsm8k_thinking_on_cap16384", "sampled/mmlu",
                "sampled/ladders", "sampled/serving", "transcripts_results"):
        r = subprocess.run([*cli, "ls", VOL_NAME, f"{volume_client_path(WAVE_VOL)}/{sub}", "--json"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            continue
        for e in json.loads(r.stdout or "[]"):
            name = Path(e.get("filename") or "").name   # `modal volume ls --json` keys: filename/type/created_modified/size
            if name.startswith(f"{tag}_t") or f"_{tag}_" in name or name.startswith(f"sdfbase_full22_{tag}_sampled_t"):
                wanted.append(f"{sub}/{name}")
    for rel in wanted:
        out = dest_p / WAVE_ROOT / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"[pull] {rel} -> {out}", flush=True)
        subprocess.run([*cli, "get", "--force", VOL_NAME, f"{volume_client_path(WAVE_VOL)}/{rel}", str(out.parent)], check=False)
    logs_out = dest_p / "logs_modal"
    logs_out.mkdir(exist_ok=True)
    r = subprocess.run([*cli, "ls", VOL_NAME, volume_client_path(LOGS_VOL), "--json"], capture_output=True, text=True)
    for e in json.loads(r.stdout or "[]") if r.returncode == 0 else []:
        name = Path(e.get("filename") or "").name   # `modal volume ls --json` keys: filename/type/created_modified/size
        if f"-{tag}-" in name:
            subprocess.run([*cli, "get", "--force", VOL_NAME, f"{volume_client_path(LOGS_VOL)}/{name}", str(logs_out)], check=False)
    print(f"[pull] done: {len(wanted)} store dirs under {dest_p}", flush=True)


@app.local_entrypoint()
def main(sha: str = "", build: bool = False, stage: bool = False, restage: bool = False, stage_bank: bool = False,
         stage_tools: bool = False, pf: bool = False, go: bool = False, key: str = "", view: str = "",
         phase: str = "", trial: int = -1, nshards: int = 9, cells: str = "", wait: bool = False,
         watch: bool = False, legs: str = "", max_concurrent: int = 4, pull: str = "", pull_to: str = "",
         sync_every_s: int = 60, max_hours: float = 15.5, probe_concurrency_n: int = 0, serving_path: str = "lens",
         wave_root: str = WAVE_ROOT_DEFAULT, leg_env: str = ""):
    if len(sha) != 40:
        raise SystemExit("[strength] pass the full 40-hex --sha of the wave tree")
    set_wave_root(wave_root)
    print(f"[strength] wave root {WAVE_ROOT}: stores {WAVE_VOL}, logs {LOGS_VOL}, ledger {LEGS_LOG}", flush=True)
    want = expected_app_name(sha)
    run_name = app.description or DEFAULT_APP_NAME
    if run_name != want:
        raise SystemExit(f"[strength] run with `modal run -n {want} ...`; got app name {run_name!r}")
    print(f"[strength] app={run_name} sha={sha} local_dep_hashes={ {k: v[:12] for k, v in LOCAL_DEP_HASHES.items()} }", flush=True)
    if probe_concurrency_n:
        probe_concurrency(int(probe_concurrency_n))
    if build:
        rep = image_check.remote(LOCAL_DEP_HASHES[VLLM_REQS_REL])
        print(json.dumps(rep, indent=1, default=str))
        if not rep.get("ok"):
            raise SystemExit("[strength] the GPU image's venv check failed")
    if stage:
        do_stage_tree(sha, restage)
    if stage_bank:
        do_stage_bank(sha)
    if stage_tools:
        do_stage_tools()
    if pf:
        if not (key and view):
            raise SystemExit("[strength] --pf needs --key and --view (and --phase, default controls_t0)")
        rep = preflight.remote(sha, key, view, phase or "controls_t0", LOCAL_DEP_HASHES, wave_root)
        print(json.dumps(rep, indent=1, default=str))
        if not rep.get("ok"):
            raise SystemExit("[strength] PREFLIGHT FAILED -- fix every item above before any GPU")
    if go:
        if not (key and view and phase) or trial not in (0, 1):
            raise SystemExit("[strength] --go needs --key --view --phase and --trial 0|1")
        require_cluster_label(leg_env)
        cid = spawn_leg(sha, run_name, key, view, phase, trial, nshards, cells, sync_every_s, max_hours, serving_path, leg_env)
        print(f"[strength] spawned {cid} for {tag_of(key, view)} {phase}; `--watch` reports it to {LEGS_LOG}", flush=True)
        if wait:
            done, summary, err = False, None, None
            while not done:
                time.sleep(60)
                done, summary, err = _poll_call(cid)
            summary = summary or {"tag": tag_of(key, view), "phase": phase}
            _append_leg_log(_leg_line(summary, error=err) + f" call={cid}")
            print(json.dumps({k: v for k, v in summary.items() if k != "log_tail"}, indent=1, default=str))
            if err or summary.get("status") != "DONE":
                raise SystemExit(f"[strength] leg did not finish DONE: {err or summary.get('status')}")
    if legs:
        require_cluster_label(leg_env)
        run_legs(sha, run_name, legs, int(max_concurrent), nshards, sync_every_s, max_hours, serving_path=serving_path, leg_env=leg_env)
    if watch:
        watch_calls()
    if pull:
        if not pull_to:   # the default destination is under the grade root, $GRADE; an unset one refuses by name
            if str(LOCAL_REPO / "src") not in sys.path:
                sys.path.insert(0, str(LOCAL_REPO / "src"))
            from sisteer import site as _site
            pull_to = str(_site.expand_path("$GRADE/strength/modal_stores", key="--pull-to"))
        pull_tag(pull, pull_to)


if __name__ == "__main__":
    raise SystemExit("run with `python -m modal run -n strength-served-<sha7> scripts/modal/strength_served.py ...`")
