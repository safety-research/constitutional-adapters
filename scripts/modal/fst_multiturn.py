"""fst-multiturn: experiment 2 of the fit-support-transfer spec — multi-turn
steering under three serve settings — on Modal.

One 2xH200 container reproduces ``scripts/modal/grokmt_served.py``'s topology
(GPU 0 = the campaign's abliterated Qwen3.6-27B attacker on vLLM at
``--max-model-len 49152``; GPU 1 = the gemma-4-E4B-it target served by
``capctrl_serving_up.py`` + the lens-routed RoutingProxy), with the drivers
(``scripts/jailbreak_run.py generate --target-engine served --attack crescendo``)
running IN the container on loopback. ONE standup serves every cell of the run
as its own virtual model:

  none                                  the unsteered baseline (petri__base)
  safety_refusal__psr_kl_all            setting (1): the last assistant turn (response mask)
  safety_refusal__psr_kl_all@all_turns  setting (2): every assistant turn (response_all_turns)
  safety_refusal__psr_kl_all@all_positions  setting (3): every position, attacker turns included

The two ``@variants`` are HASHED serve declarations (``serve_declaration.py``,
WS-0): three DISTINCT intervention hashes over ONE artifact, attested per cell
in ``lens_serve_declarations``; the drivers' served target REFUSES a variant
whose declaration the proxy does not attest (``served_target.py``). The
standup runs ``--lens-salt-policy request_unique_all`` and
``SISTEER_LENS_STACKED_REDUCTIONS=1``, so every lens leg (base, both variants,
the ``none__lens0`` twin) mints request-unique salts BY DECLARATION and the
settings differ in the MASK only (an attested fact).

THE RUN IDENTITY. The
inherited machinery binds retained work to the ATTACKER only, so ``--go``
first canonicalises ``fst_run_identity.json`` — gated sha, the goal set under
THREE names (``goal_ids`` the ordered list, ``goal_ids_file_sha256`` the digest
of the ids file's ORIGINAL bytes, ``goal_ids_sha256`` the canonical ordered-ids
digest), ``subshards`` plus the full goal -> shard assignment the drivers
slice (``goal_shards``), cell set, per-cell intervention hash / artifact sha /
declaration / positions, the artifact digest map, reduction mode (SET, not read: the env var
is exported explicitly and ``stacked`` canonicalised), salt policy, geometry,
attack settings — and compares it BYTE FOR BYTE with the record on the volume
BEFORE anything is seeded or overwritten: equal -> proceed (a retry of an
identity-only run dir re-enters), different -> REFUSE naming the differing
fields (a changed target is a NEW experiment: use a new ``--run``), absent ->
write it (0440, temp + os.replace). ``seed_from_previous`` runs only when a
driver tree exists under the run. After standup the ATTESTED values
(``lens_serve_declarations``, ``lens_reduction_mode``, ``lens_salt_policy``,
the routing table's per-cell ``intervention_hash``,
``object_artifact_sha256_by_cell``) must equal the record's, or no driver
starts.

RESUME. ``--resume --run <same>`` under a NEW app name seeds from the volume
copy: completed goals AND render-refused goals (``refused:<code>``, terminal)
are never regenerated; ``error:*`` goals are retried
(``static_common.load_covered_goals``). The run tree is synced + committed on
every progress change. **A run that already holds a driver tree REQUIRES
``--resume``** (code review S-C1): a same-run relaunch without it would start
from an EMPTY local tree, regenerate every goal and let the sync overwrite a
terminal ``refused:*`` record with a stochastic success — so it is refused
(``NEEDS_RESUME``) before anything is seeded, stood up or synced. Only an
identity-only run dir (no driver tree: a standup or attacker failure before
the first driver) admits the plain retry.

THE ATTEMPT LEASE (code review S-C2, r3, AC-2, D36). Two concurrent ``--resume
--run <same>`` launches (different permitted app-name suffixes) would both seed,
both stand up and both sync the same durable paths, last writer wins per file.
So before anything is seeded or stood up the attempt takes the run's entry in
the Modal Dict ``fst-multiturn-leases`` — the AUTHORITY for ownership — with
ONE atomic ``put(run_id, record, skip_if_exists=True)``. A volume file cannot do
this job (separate mount snapshots + last-writer-wins commits let two creators
each read back their own record), so ``attempt.lease`` on the volume is only
the human-readable RECORD, written best-effort after the put. An EXISTING entry
refuses the launch (``LEASE_HELD``) whether its heartbeat is fresh or stale:
there is NO automatic takeover — re-read + pop + put are three operations, and
two contenders judging the same stale owner could both win, while a displaced
owner's unconditional heartbeat put could restore it (D36) — so **overlapping
resumptions are not supported**, and a stale lease needs deliberate cleanup by
the operator (the message quotes it: inspect the entry, confirm the prior app
is dead, pop the entry, remove the record). The run loop refreshes the
heartbeat every ``LEASE_HEARTBEAT_S`` (60 s) — get, verify the owner, put, read
back — and every sync is FENCED by the same owner check immediately before it:
another owner -> ``LEASE_LOST`` (stop, sync nothing more); an unreadable Dict
-> ``LEASE_INDETERMINATE`` (nothing written, stop, sync nothing more). The
sixty seconds is a heartbeat interval, never a deadline: a check is only as
prompt as the loop that runs it, and a writer that passed its check can still
publish after a manual reassignment — which is why the operator must confirm
the prior attempt is STOPPED before clearing its lease. Uncertainty fails
CLOSED. ``release`` re-checks the owner, stamps the file with the outcome and
commits that, THEN deletes the entry; a released run is free at once. The
release needs POSITIVE authorization (D36 #3 and its closing review): the
lease is released only once the attempt's FINAL data sync has SUCCEEDED
(copied AND committed) — never merely because no failure was recorded. A
failed final sync leaves the lease HELD and the run reports
``FINAL_SYNC_FAILED`` loudly, so no later launch resumes from a tree that was
never durably committed; destination files are copied in place, so a failed
copy can leave a truncated file there, and only a later sync that copies it
whole and commits proves the destination complete. An unexpected exception
after data copying has begun stops every writer (the drivers, the target, the
attacker), attempts ONE owned final sync, and releases only if that commits;
otherwise the lease stays held and the exception propagates. The ONLY exits
that release without a committed final sync are failures before this
attempt's first data copy — the local scaffolding writes, the attacker and
target spawns, and the standup and readiness waits when no sync has been
reached — because by then this attempt has put nothing under the run dir but
the identity file and the lease record: the identity-only state a refused
launch leaves, which the plain retry admits. What resume PRESERVES is exactly
the committed state — committed terminal rows (done and ``refused:*``) are
never regenerated. Work a driver finished locally after the last committed
sync is not preserved: it may be REPEATED on resume, and repeating it can
change the sample — a repeated goal is a fresh stochastic attack (a lost
refusal need not refuse again) — even though it overwrites no committed
measurement. The 15 s driver poll and the 120 s ``sync_every_s`` are cadence,
not a bound on how much can be lost: the run waits up to 300 s for the target
teardown before its final copy with no periodic sync during that wait, and
blocking I/O stretches any interval further without any failed commit.

Isolation: app ``fst-multiturn`` (``-n fst-multiturn-<sha8>-<run>`` REQUIRED);
writes ONLY under ``/vol/fst/`` (the staged tree, the run outputs, and — only
when ``--stage-attacker`` is asked for — an attacker HF cache).
``/vol/objects/scale-12`` and ``/vol/grokmt/hf_attacker`` (the attacker copy
``grokmt_served.py --stage`` made; read when ``/vol/fst/hf_attacker`` is
absent) are READ-ONLY inputs. NO credential is used: the attacker is a public
model on a per-job vLLM, the grader (haiku, Low Prio via ``sisteer.vault``)
runs on the grading host afterwards (``scripts/grokmt_grade.py``).

Run (the grading host; the sha is the INTEGRATION tip)::

  M=~/venvs/modal/bin/python; SHA=<fst-integration tip>; S8=${SHA:0:8}
  $M -m modal run -n fst-multiturn-$S8-stage scripts/modal/fst_multiturn.py --stage --sha $SHA
  $M -m modal run -n fst-multiturn-$S8-pf    scripts/modal/fst_multiturn.py --pf --sha $SHA
  $M -m modal run --detach -n fst-multiturn-$S8-r1 scripts/modal/fst_multiturn.py \\
      --go --sha $SHA --run r1        # --ids defaults to the tracked configs/fst/fst_exp2_goal_ids.txt
  $M -m modal run -n fst-multiturn-$S8-r1 scripts/modal/fst_multiturn.py --harvest ~/.local/state/fst-multiturn/r1.json
  # a resume: SAME --run, a NEW app name suffix
  $M -m modal run --detach -n fst-multiturn-$S8-r1-a2 scripts/modal/fst_multiturn.py \\
      --go --sha $SHA --run r1 --resume
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path

import modal

DEFAULT_APP_NAME = "fst-multiturn"
app = modal.App(DEFAULT_APP_NAME)
VOL_NAME = "twosi-fits"
VOL = modal.Volume.from_name(VOL_NAME, create_if_missing=False)
VOL_PATH = "/vol"
MY = f"{VOL_PATH}/fst"
REPOS = f"{MY}/repos"
GEN = f"{MY}/gen"
ATTACKER_HF = f"{MY}/hf_attacker"
#: READ-ONLY: the attacker snapshot grokmt_served.py --stage put on the volume;
#: read only when MY's own copy is absent (nothing is ever written there)
ATTACKER_HF_SHARED = f"{VOL_PATH}/grokmt/hf_attacker"
OBJECTS_X1 = f"{VOL_PATH}/objects/scale-12/gemma4e4b"      # READ-ONLY shared bank
CFG_REL = "configs/two_si_gemma4e4b_reduced.yaml"
BASE_CELL = "safety_refusal__psr_kl_all"
CONSTITUTION_CELL = "si_constitution__psr_kl_all"
VARIANT_ALL_TURNS = "all_turns"
VARIANT_ALL_POSITIONS = "all_positions"


def cell_spellings(base: str) -> tuple[str, str, str]:
    """The three settings of one fitted base: (1) the base at its recorded
    response mask, (2) ``@all_turns``, (3) ``@all_positions``."""
    return (base, f"{base}@{VARIANT_ALL_TURNS}", f"{base}@{VARIANT_ALL_POSITIONS}")


DEFAULT_CELLS = ("none",) + cell_spellings(BASE_CELL)
ATTACKS = ("crescendo",)
ARMS = ("qwen",)
ATTACKER_MODEL = "huihui-ai/Huihui-Qwen3.6-27B-abliterated"
ATTACKER_SNAPSHOT = "27502c8717fd5a2f8c0c77188c10c243fd4f672e"
ATTACKER_MAX_MODEL_LEN = 49152
PRIVATE_RUN = "two_si_v1"   # cfg['private_run'] — the wave's layout
#: the geometry-table eval the served target IS: the per-attack p99 entry
#: (9984 x identity S) — the pooled 8704 was sized for all three attacks
TARGET_EVAL = "multiturn_attacks:crescendo"
TARGET_KEY = "gemma4e4b"
#: the standup knobs the identity binds
LENS_SALT_POLICY = "request_unique_all"
REDUCTION_MODE = "stacked"
STACKED_REDUCTIONS_ENV = "SISTEER_LENS_STACKED_REDUCTIONS"
STACKED_REDUCTIONS_VALUE = "1"
IDENTITY_FILE = "fst_run_identity.json"
#: the TRACKED goal-id file (scripts/fst_exp2_goals.py writes it; the pair is
#: pinned by tests/test_fst_exp2_goals.py) — the default of --ids, relative to
#: the repo root, so the identity's goal_ids_file_sha256 (the digest of the
#: file's ORIGINAL bytes, header comments included) points at a tracked artifact
TRACKED_IDS_REL = "configs/fst/fst_exp2_goal_ids.txt"
IDENTITY_SCHEMA = "fst_run_identity/1"
#: the container-local run tree the drivers write and RunSyncer ships to GEN
RUN_LOCAL_ROOT = "/tmp/fst_run"
#: the per-run attempt lease (code review S-C2): one file in the run dir
LEASE_FILE = "attempt.lease"
LEASE_SCHEMA = "fst_attempt_lease/1"
#: a heartbeat at least this old may be taken over (the prior call not confirmed running)
LEASE_STALE_S = 15 * 60
#: the run loop refreshes the heartbeat at least this often
LEASE_HEARTBEAT_S = 60
#: the Modal Dict that is the AUTHORITY for run ownership (code review AC-2):
#: one entry per run id, created by an ATOMIC put(skip_if_exists=True)
LEASE_DICT_NAME = "fst-multiturn-leases"
DEFAULT_SUBSHARDS = 5
CONTAINER_TIMEOUT_S = 6 * 3600
STATE_DIR = Path.home() / ".local" / "state" / "fst-multiturn"


# The image's LAST layer, and lens_vectorize_gate.py's line, byte for byte: that
# script stages the same way and verifies. The staged tree is unpacked by root
# from a tarball carrying this machine's uid, so git sees "dubious ownership" and
# refuses every command in it. Two lines of defence, on purpose: ``_git_run``
# carries its own ``-c safe.directory=<tree>`` per call (the fix that is testable locally
# and survives an image rebuild), and this global config covers the git the
# STAGED TREE's own code runs outside ``_git`` — sisteer's
# ``run_identity.tree_sha`` (``git rev-parse HEAD`` for the record's
# writer_tree_sha), which would otherwise silently record no sha. A module-level
# constant, consumed by the chain below, so the test that pins it reads the
# module's own data and then watches the chain issue exactly this — under a
# ``modal`` of its own choosing, whatever imported this module first.
IMAGE_GIT_SAFE_DIRECTORY_CMD = "git config --global --add safe.directory '*'"

# --- the image: grokmt_served.py's proven cu129 serving stack + pyrit, verbatim ---
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04",
                              add_python="3.12")
    .apt_install("git", "build-essential")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "https://github.com/vllm-project/vllm/releases/download/v0.25.1/"
        "vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
        "#sha256=9e206f370c934a2d4b6b1f05d3d09708d344e05d80260189ef19f60755709431",
        "torch==2.11.0+cu129", "torchvision==0.26.0+cu129",
        "torchaudio==2.11.0+cu129", "vllm-lens==1.2.1",
        "pandas", "pyarrow", "aiohttp", "pyyaml", "numpy",
        "huggingface_hub", "ninja",
        extra_index_url="https://download.pytorch.org/whl/cu129",
    )
    .pip_install("transformers==5.14.1", "tokenizers==0.22.2",
                 extra_options="--no-deps --force-reinstall")
    .run_commands(
        "python -c \"import torch, vllm, transformers, aiohttp, yaml; "
        "assert torch.__version__ == '2.11.0+cu129', torch.__version__; "
        "assert vllm.__version__.startswith('0.25.1'), vllm.__version__; "
        "print('IMAGE OK', vllm.__version__, torch.__version__)\"",
        "which vllm",
    )
    .env({"HF_HOME": f"{VOL_PATH}/hf", "HF_HUB_OFFLINE": "1",
          "TOKENIZERS_PARALLELISM": "false",
          "SISTEER_HELDOUT_CORPUS_DIR": f"{VOL_PATH}/heldout_corpora"})
    .apt_install("unixodbc")
    .pip_install("pyrit==1.0.1", "requests")
    .run_commands(
        "python -c \"import pyrit, vllm, torch, transformers, requests; "
        "assert transformers.__version__ == '5.14.1', transformers.__version__; "
        "assert vllm.__version__.startswith('0.25.1'), vllm.__version__; "
        "assert torch.__version__ == '2.11.0+cu129', torch.__version__; "
        "print('PYRIT OK', pyrit.__version__)\"",
    )
    # The global safe.directory (see ``IMAGE_GIT_SAFE_DIRECTORY_CMD`` above for
    # why): its own layer, appended last, so every cached layer above it stays
    # cached.
    .run_commands(IMAGE_GIT_SAFE_DIRECTORY_CMD)
)


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _http_ok(url: str, timeout: float = 5.0) -> bool:
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return 200 <= r.status < 300
    except Exception:  # noqa: BLE001
        return False


class HealthWaitFailed(RuntimeError):
    """A server died or never answered ``/health`` — the health wait's OWN
    failure type, so its handler can never catch anything else (code review
    AC-1: a bare ``except RuntimeError`` there swallowed a lease failure raised
    by ``on_tick`` and synced over another owner's files)."""


def _wait_health(url: str, proc: subprocess.Popen, timeout_s: float,
                 what: str, log: Path, on_tick=None) -> float:
    """``on_tick`` (the lease heartbeat) runs once per poll; whatever it raises
    passes straight through — only this wait's own failures are
    :class:`HealthWaitFailed`."""
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if on_tick is not None:
            on_tick()
        if proc.poll() is not None:
            tail = log.read_text(errors="replace")[-3000:] if log.exists() else ""
            raise HealthWaitFailed(f"{what} died (rc={proc.returncode}) before "
                                   f"healthy:\n{tail}")
        if _http_ok(url):
            return time.time() - t0
        time.sleep(5)
    raise HealthWaitFailed(f"{what} not healthy after {timeout_s}s")


def _git_run(repo: str | Path, *args: str) -> subprocess.CompletedProcess:
    """``git <args>`` in ``repo`` with stdout AND stderr captured. Every call
    carries the repo's own ``safe.directory``: the staged tree is tarred here
    (``tarfile`` records this machine's uid on every entry) and unpacked by the
    container's root, which chowns the files back to that uid — a repository
    owned by another user, and git refuses EVERY command in one (``fatal:
    detected dubious ownership``, rc 128, empty stdout). Until 2026-09-07 that
    empty stdout read as "not a git checkout" and blocked staging. The per-call
    flag needs nothing from the image (which also sets it globally, see
    ``image``), so a rebuilt or foreign image cannot bring the refusal back;
    the resolved path (not ``*``) trusts only the tree this script addresses."""
    repo = Path(repo)
    return subprocess.run(["git", "-c", f"safe.directory={repo.resolve()}", *args],
                          cwd=str(repo), capture_output=True, text=True, timeout=120)


def _git(repo: str | Path, *args: str) -> str:
    return _git_run(repo, *args).stdout.strip()


def _git_stderr(p: subprocess.CompletedProcess) -> str:
    """git's stderr as one line for a ``why`` (its hint lines folded in)."""
    return " ".join(p.stderr.split()) or "no output"


def _attacker_snapshot_dir() -> str:
    """The attacker snapshot: MY's own copy when present (``--stage-attacker``),
    else the READ-ONLY copy grokmt_served.py staged."""
    rel = f"models--{ATTACKER_MODEL.replace('/', '--')}/snapshots/{ATTACKER_SNAPSHOT}"
    own = f"{ATTACKER_HF}/{rel}"
    if Path(own).is_dir():
        return own
    return f"{ATTACKER_HF_SHARED}/{rel}"


def _tree(sha: str) -> str:
    return f"{REPOS}/sisteer-{sha}"


def _env(tree: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tree}/src:{tree}/scripts"
    env["TT_TWO_SI_DEFENSES_DIR"] = OBJECTS_X1
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def _ensure_sisteer_importable(tree=None) -> None:
    """Put a ``sisteer`` on ``sys.path``: the STAGED tree's (in-container) when
    given, else the local checkout's (at ``modal run`` time) when nothing is
    importable yet. Every sisteer import in this script runs after this."""
    if tree and f"{tree}/src" not in sys.path:
        sys.path.insert(0, f"{tree}/src")
    try:
        import sisteer  # noqa: F401
    except ImportError:
        local_src = str(Path(__file__).resolve().parents[2] / "src")
        if local_src not in sys.path:
            sys.path.insert(0, local_src)


def _geometry_module(tree):
    """The per-eval geometry table from the STAGED tree (in-container), else
    whichever sisteer is importable (the local checkout at ``modal run`` time)."""
    _ensure_sisteer_importable(tree)
    from sisteer.capeval.serving import geometry
    return geometry


def _served_target_module(tree=None):
    """``sisteer.eval.attacks.served_target`` — the ONE block of pinned
    attestation keys this script reads (from the staged tree in-container)."""
    _ensure_sisteer_importable(tree)
    from sisteer.eval.attacks import served_target
    return served_target


def resolve_target_geometry(tree, *, target_window: int = 0, target_seqs: int = 0,
                            allow_over_budget: bool = False, target_key: str = TARGET_KEY):
    """The served target's launch geometry: ``multiturn_attacks:crescendo`` with
    S sized from ``target_key``'s measured budget unless overridden (0 = unset)."""
    g = _geometry_module(tree)
    return g.resolve_geometry(TARGET_EVAL, max_model_len=target_window or None,
                              max_num_seqs=target_seqs or None,
                              allow_over_budget=allow_over_budget,
                              model_key=target_key or None)


def parse_cells(cells: str, constitution: bool = False) -> list[str]:
    """The ordered, de-duplicated cell list: ``cells`` (comma-separated; empty
    = :data:`DEFAULT_CELLS`) plus the three constitution spellings under
    ``--constitution``. Every id must parse under the serve-variant grammar
    (``serve_declaration.split_variant``) — an unknown variant refuses here."""
    out = [c.strip() for c in (cells or "").split(",") if c.strip()] or list(DEFAULT_CELLS)
    if constitution:
        out += [c for c in cell_spellings(CONSTITUTION_CELL) if c not in out]
    if len(set(out)) != len(out):
        raise SystemExit(f"[fst] duplicate cell ids in {out}")
    _ensure_sisteer_importable()
    from sisteer.capeval.serving.serve_declaration import ServeDeclarationError, split_variant
    for c in out:
        try:
            split_variant(c)
        except ServeDeclarationError as e:
            raise SystemExit(f"[fst] --cells: {e}")
    return out


def fitted_bases(cells: list[str]) -> list[str]:
    """The distinct fitted BASE cells (``none`` excluded), in first-seen order."""
    _ensure_sisteer_importable()
    from sisteer.capeval.serving.serve_declaration import split_variant
    seen: list[str] = []
    for c in cells:
        base, _v = split_variant(c)
        if base != "none" and base not in seen:
            seen.append(base)
    return seen


def expected_app_name(sha: str, run: str) -> str:
    return f"fst-multiturn-{sha[:8]}-{run}"


def check_app_name(run_name: str, *, sha: str, run: str = "") -> None:
    """The entrypoint REQUIRES ``-n fst-multiturn-<sha8>-<run>[-<suffix>]``
    (the lens gate's rule): a resumed attempt runs under a NEW suffix so a
    ``modal app stop`` aimed at one attempt can never hit another, and an
    unnamed run cannot inherit a stale identity. Without ``--run`` (``--stage``
    / ``--pf``) any ``fst-multiturn-<sha8>-<label>`` is accepted."""
    prefix = f"fst-multiturn-{sha[:8]}-"
    if run:
        ok = run_name == expected_app_name(sha, run) or \
            run_name.startswith(expected_app_name(sha, run) + "-")
        want = f"{expected_app_name(sha, run)}[-<attempt suffix>]"
    else:
        ok = run_name.startswith(prefix) and len(run_name) > len(prefix)
        want = f"{prefix}<label>"
    if not ok:
        raise SystemExit(f"[fst] run with `modal run -n {want} ...`; got app name "
                         f"{run_name!r}. The explicit descriptive app name is mandatory")


# --------------------------------------------------------------------------- #
# staging: the tree by sha from the LOCAL object store (lens gate / cpu_suite) #
# --------------------------------------------------------------------------- #

def local_repo_root() -> Path:
    """The git repo this file lives in (the worktree's COMMON dir's parent —
    every worktree shares its objects, so any sha reachable anywhere in the
    repo can be fetched from it)."""
    here = Path(__file__).resolve().parent
    common = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
                            cwd=str(here), capture_output=True, text=True, check=True).stdout.strip()
    return Path(common).parent


def verify_checkout(repo: str | Path, sha: str) -> dict:
    """Exact HEAD and clean tracked files — the same tree facts as the gates.
    When a git command fails outright, ``why`` carries git's own stderr (rc
    included) rather than a guess: "not a git checkout" hid the container's
    dubious-ownership refusal for a whole staging round."""
    rp = _git_run(repo, "rev-parse", "HEAD")
    head = rp.stdout.strip() if rp.returncode == 0 else ""
    df = _git_run(repo, "diff", "--quiet", "HEAD", "--") if head else None
    clean = df is not None and df.returncode == 0         # 0 clean, 1 dirty, else an error
    ok = bool(head) and head == sha and clean
    if ok:
        why = None
    elif not head:
        why = f"git rev-parse HEAD failed (rc {rp.returncode}): {_git_stderr(rp)}"
    elif head != sha:
        why = f"HEAD {head} != {sha}"
    elif df.returncode == 1:
        why = "modified tracked files"
    else:
        why = f"git diff --quiet HEAD failed (rc {df.returncode}): {_git_stderr(df)}"
    return {"ok": ok, "head": head or None, "clean": clean, "why": why}


def shallow_clone_tarball(sha: str, dest_dir: Path) -> Path:
    """A SHALLOW standalone clone of ``sha`` from the local repo, HEAD detached
    at the sha, verified, tarred as ``dest_dir/sisteer-<sha>.tar.gz``."""
    src = local_repo_root()
    if _git(src, "cat-file", "-t", sha) != "commit":
        raise SystemExit(f"[fst] {sha} is not a commit in {src}")
    work = Path(tempfile.mkdtemp(prefix="fst-stage-"))
    tree = work / f"sisteer-{sha}"
    try:
        tree.mkdir()
        subprocess.run(["git", "init", "-q", str(tree)], check=True)
        subprocess.run(["git", "-c", "uploadpack.allowAnySHA1InWant=true", "-C", str(tree),
                        "fetch", "-q", "--depth", "1", str(src), sha], check=True)
        subprocess.run(["git", "-C", str(tree), "checkout", "-q", "--detach", "FETCH_HEAD"],
                       check=True)
        v = verify_checkout(tree, sha)
        if not v["ok"]:
            raise SystemExit(f"[fst] shallow clone of {sha} failed verification: {v}")
        tar_path = dest_dir / f"sisteer-{sha}.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tf:
            tf.add(str(tree), arcname=f"sisteer-{sha}")
        return tar_path
    finally:
        shutil.rmtree(work, ignore_errors=True)


def volume_client_path(path: str) -> str:
    prefix = f"{VOL_PATH}/"
    if not path.startswith(prefix):
        raise ValueError(f"path is outside {VOL_PATH}: {path}")
    return path[len(VOL_PATH):]


@app.function(image=image, volumes={VOL_PATH: VOL}, timeout=1800, cpu=4, memory=16384)
def unpack_tree(sha: str, force: bool = False) -> dict:
    """Extract ``/vol/fst/repos/sisteer-<sha>.tar.gz`` (pushed by --stage) into
    ``/vol/fst/repos/sisteer-<sha>``, verify ON the volume, delete the tarball,
    commit. An existing dir is verified and kept unless ``force``."""
    VOL.reload()
    dst = Path(REPOS) / f"sisteer-{sha}"
    tar_path = Path(REPOS) / f"sisteer-{sha}.tar.gz"
    out = {"sha": sha, "tree": str(dst)}
    if dst.exists() and not force:
        v = verify_checkout(dst, sha)
        out.update(existing=True, verify=v)
        if tar_path.exists():
            tar_path.unlink()
            VOL.commit()
        if not v["ok"]:
            out["error"] = f"{dst} exists but does not verify ({v['why']}); --restage replaces it"
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
    v = verify_checkout(dst, sha)
    out.update(existing=False, verify=v)
    if not v["ok"]:
        out["error"] = f"unpacked tree does not verify: {v['why']}"
    VOL.commit()
    return out


@app.function(image=image, volumes={VOL_PATH: VOL}, timeout=3 * 3600, cpu=8, memory=32768)
def stage_attacker() -> dict:
    """The 27B attacker (public model) into MY's own HF cache — only when the
    grokmt copy is not on the volume (``--stage-attacker``)."""
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import snapshot_download
    t0 = time.time()
    path = snapshot_download(ATTACKER_MODEL, revision=ATTACKER_SNAPSHOT,
                             cache_dir=ATTACKER_HF, local_files_only=False, max_workers=16)
    VOL.commit()
    files = [f for f in Path(path).rglob("*") if f.is_file()]
    return {"path": path, "n_files": len(files),
            "gb": round(sum(f.stat().st_size for f in files) / 1e9, 1),
            "s": round(time.time() - t0)}


# --------------------------------------------------------------------------- #
# the run identity: canonical, compared before anything is touched             #
# --------------------------------------------------------------------------- #

class RunIdentityMismatch(RuntimeError):
    """``fst_run_identity.json`` on the volume differs from this launch's —
    a changed target is a NEW experiment, never a resume."""


class RunNeedsResume(RuntimeError):
    """The run dir already holds a driver tree and ``--resume`` was not given
    (code review S-C1): proceeding would regenerate every goal from an empty
    local tree and overwrite terminal ``refused:*`` records on sync."""


def canonical_json_bytes(record: dict) -> bytes:
    """Canonical JSON: sorted keys, no whitespace variance, ASCII-escaped."""
    return json.dumps(record, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True).encode("utf-8")


def identity_diff(old: dict, new: dict, _prefix: str = "", _depth: int = 0) -> list[str]:
    """The names of the differing fields, descending into dict-valued fields
    two levels deep (``per_cell.<harness id>.<field>``), in sorted order."""
    out: list[str] = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k, "<absent>"), new.get(k, "<absent>")
        if a == b:
            continue
        if _depth < 2 and isinstance(a, dict) and isinstance(b, dict):
            out += identity_diff(a, b, f"{_prefix}{k}.", _depth + 1)
        else:
            out.append(f"{_prefix}{k}")
    return out


def parse_goal_ids(text: str) -> list[str]:
    """The ids of an ids file's text: one per line, ``#`` comments and blank
    lines skipped, whitespace stripped — the rule ``jailbreak_run.
    _select_goal_ids`` applies to the run's copy, so what the identity binds
    is what every driver replays."""
    return [ln.strip() for ln in text.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]


def goal_ids_bindings(goal_ids: list[str], goal_ids_file_bytes: bytes) -> dict:
    """The THREE goal bindings of the identity, each under its own name
    (code review E1/B2): ``goal_ids`` — the ordered list; ``goal_ids_file_sha256``
    — the sha256 of the ids file's ORIGINAL bytes (the tracked
    ``configs/fst/fst_exp2_goal_ids.txt`` carries a ``#`` header, so this is
    the digest the selection record pins and the analyzer compares against the
    file it is handed); ``goal_ids_sha256`` — the sha256 of the canonical
    ordered ids (``"\\n".join(ids) + "\\n"``), the reconstruction-independent
    digest. Refuses when the bytes do not parse to exactly ``goal_ids``: the
    list and the file must be the same goal set, never two reconstructions."""
    parsed = parse_goal_ids(goal_ids_file_bytes.decode("utf-8"))
    if parsed != list(goal_ids):
        raise ValueError(f"goal_ids {list(goal_ids)[:4]}... (n={len(goal_ids)}) are not what the "
                         f"ids file's bytes parse to: {parsed[:4]}... (n={len(parsed)})")
    canonical = ("\n".join(goal_ids) + "\n").encode("utf-8")
    return {"goal_ids": list(goal_ids),
            "goal_ids_file_sha256": hashlib.sha256(goal_ids_file_bytes).hexdigest(),
            "goal_ids_sha256": hashlib.sha256(canonical).hexdigest()}


def shard_assignment(goal_ids: list[str], subshards: int) -> list[list[str]]:
    """The complete goal -> shard assignment, in shard order: shard ``k`` runs
    ``goal_ids[k::subshards]`` — exactly the slice ``jailbreak_run`` takes
    (``goals[args.shard::args.nshards]`` after ``_select_goal_ids`` restores
    FILE order), so the identity binds what the drivers will actually own
    (code review B1: a changed count re-partitions goals, and a terminally
    refused goal moving to a shard that cannot see its record would be
    re-attempted and could overwrite its refusal on merge)."""
    if int(subshards) < 1:
        raise ValueError(f"subshards must be >= 1, got {subshards}")
    return [list(goal_ids[k::int(subshards)]) for k in range(int(subshards))]


def build_run_identity(*, sha: str, goal_ids: list[str], goal_ids_file_bytes: bytes,
                       cells: list[str], subshards: int,
                       per_cell: dict, control_legs: dict, digests_by_cell: dict,
                       geom_dict: dict, attack: dict, target: dict,
                       reduction_mode: str = REDUCTION_MODE,
                       lens_salt_policy: str = LENS_SALT_POLICY) -> dict:
    """The record, as a plain dict — ``canonical_json_bytes`` of it
    is what the volume holds. ``goal_ids_file_bytes`` are the ORIGINAL bytes of
    the ids file the launcher read (:func:`goal_ids_bindings`); ``subshards``
    and the derived :func:`shard_assignment` bind the goal partition.
    ``per_cell`` is keyed by HARNESS id in its variant spelling;
    ``digests_by_cell`` is the ``object_artifact_sha256_by_cell`` map the
    standup is held to (fitted lens cells only)."""
    return {
        "schema": IDENTITY_SCHEMA,
        "gated_sha": sha,
        **goal_ids_bindings(goal_ids, goal_ids_file_bytes),
        "subshards": int(subshards),
        "goal_shards": shard_assignment(goal_ids, subshards),
        "cells": list(cells),
        "per_cell": {hid: dict(v) for hid, v in per_cell.items()},
        "control_legs": {hid: dict(v) for hid, v in control_legs.items()},
        "object_artifact_sha256_by_cell": dict(digests_by_cell),
        "reduction_mode": reduction_mode,
        "lens_salt_policy": lens_salt_policy,
        "geometry": dict(geom_dict),
        "attack": dict(attack),
        "target": dict(target),
    }


def ensure_run_identity(run_dir: Path, record: dict, *, commit=None, write: bool = True) -> str:
    """Idempotent on an EQUAL identity: compare ``run_dir/fst_run_identity.json``
    byte for byte with ``canonical_json_bytes(record)`` — equal -> ``"equal"``
    (the file is left as is), different -> :class:`RunIdentityMismatch` BEFORE
    anything under ``run_dir`` is touched, absent -> write it (0440, temp +
    ``os.replace``) and return ``"written"``. With ``write=False`` (the
    compare-only pass a launch makes BEFORE it holds the run's lease) an
    absent file returns ``"absent"`` and nothing under ``run_dir`` is
    created — the identity is written only by the owner."""
    run_dir = Path(run_dir)
    p = run_dir / IDENTITY_FILE
    data = canonical_json_bytes(record)
    if p.exists():
        existing = p.read_bytes()
        if existing == data:
            return "equal"
        try:
            old = json.loads(existing.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            old = None
        diffs = identity_diff(old, record) if isinstance(old, dict) else ["<unparseable existing record>"]
        raise RunIdentityMismatch(
            f"{p} records a DIFFERENT run identity than this launch's — differing fields: "
            f"{diffs}. A changed target/goal set/cell set/declaration/digest/salt policy/"
            "reduction mode/geometry/attack setting is a NEW experiment, not a resume: "
            "use a NEW --run. Nothing under the run was touched.")
    if not write:
        return "absent"
    run_dir.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(data)
    try:
        os.chmod(tmp, 0o440)
    except OSError:
        pass
    os.replace(tmp, p)
    if commit is not None:
        commit()
    return "written"


# --------------------------------------------------------------------------- #
# the attempt lease (code review S-C2): one file, one acquire/beat/release       #
# --------------------------------------------------------------------------- #

class AttemptLeaseError(Exception):
    """Base of every lease failure — deliberately NOT a ``RuntimeError``, so no
    ``except RuntimeError`` on the run path can swallow one and go on to sync
    (code review AC-1). Only the run body's own lease handler catches these."""


class AttemptLeaseHeld(AttemptLeaseError):
    """Another attempt owns the run (a live heartbeat, a stale one whose call is
    still running, a lost create/commit race, or an unreadable record)."""


class AttemptLeaseLost(AttemptLeaseError):
    """This attempt's lease was taken over: its heartbeat found another owner
    (before the heartbeat, or at the read-back after its commit)."""


class AttemptLeaseIndeterminate(AttemptLeaseError):
    """A heartbeat's volume reload failed, so whether this attempt still owns
    the run cannot be determined: fail CLOSED — nothing written or committed,
    the lease dropped, the run stops without syncing anything more."""


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(epoch))


def lease_record(*, run_id: str, owner: str, owner_kind: str, app: str, now: float) -> dict:
    """The lease as a plain dict (``canonical_json_bytes`` of it is the file)."""
    return {"schema": LEASE_SCHEMA, "run_id": run_id, "owner": owner, "owner_kind": owner_kind,
            "app": app, "started_at": _iso(now), "started_epoch": float(now),
            "heartbeat_at": _iso(now), "heartbeat_epoch": float(now),
            "released_at": None, "outcome": None}


def owner_of() -> tuple[str, str]:
    """``(owner, kind)``: the Modal function-call id of this container when
    there is one (``modal.current_function_call_id``), else a fresh uuid."""
    fn = getattr(modal, "current_function_call_id", None)
    try:
        cid = fn() if callable(fn) else None
    except Exception:  # noqa: BLE001
        cid = None
    if cid:
        return str(cid), "modal_function_call"
    return uuid.uuid4().hex, "uuid"


def _lease_dict():
    """The named Dict holding every run's attempt lease (created on first use)."""
    return modal.Dict.from_name(LEASE_DICT_NAME, create_if_missing=True)


def _lease_remedy(run_id: str, run_dir: Path) -> str:
    """The MANUAL release of a run's lease — the ONLY way an existing entry is
    ever cleared by anyone but its owner: inspect the Dict entry, and,
    once the prior attempt is confirmed dead, pop it (the authority) and remove
    the volume record."""
    path = Path(run_dir) / LEASE_FILE
    pop = (f"python -c 'import modal; print(modal.Dict.from_name(\"{LEASE_DICT_NAME}\")"
           f".pop(\"{run_id}\"))'")
    head = (f"inspect the entry with `modal dict get {LEASE_DICT_NAME} {run_id}` and, once you have "
            "confirmed by hand that the prior attempt is dead (`modal app list`; `modal app stop <app>`), "
            f"delete it with `{pop}`")
    if str(path).startswith(f"{VOL_PATH}/"):
        rel = volume_client_path(str(path))
        return f"{head} and remove the record with `modal volume rm {VOL_NAME} {rel}`"
    return f"{head} and remove {path}"


def _lease_summary(rec) -> str:
    if not isinstance(rec, dict):
        return repr(rec)
    return (f"owner {rec.get('owner')} ({rec.get('owner_kind')}), app {rec.get('app')!r}, started "
            f"{rec.get('started_at')}, heartbeat {rec.get('heartbeat_at')}, released "
            f"{rec.get('released_at')}, outcome {rec.get('outcome')!r}")


class AttemptLease:
    """EXCLUSIVE ownership of one run for one attempt (code review S-C2 / AC-2 /
    D36). The AUTHORITY is the run's entry in the Modal Dict
    :data:`LEASE_DICT_NAME` (key = run id, value = :func:`lease_record`);
    ``attempt.lease`` on the volume is the human-readable record, written
    best-effort after every successful Dict write. ``acquire()``: ONE atomic
    ``put(skip_if_exists=True)`` — created -> held; ANY existing entry ->
    :class:`AttemptLeaseHeld`, fresh or stale alike, naming the holder and the
    manual remedy. A stale entry is never taken over automatically: re-read +
    pop + put cannot be made atomic on a Dict without compare-and-swap, so two
    contenders could both win and a displaced owner could restore itself (D36)
    — overlapping resumptions are not supported. ``verify_owner()`` is the FENCE
    run before every durable write and by every heartbeat: another owner or no
    entry -> :class:`AttemptLeaseLost`; an unreadable Dict ->
    :class:`AttemptLeaseIndeterminate` — both drop the lease with nothing
    written. ``beat()`` refreshes the heartbeat every :data:`LEASE_HEARTBEAT_S`
    (a heartbeat interval, never a deadline): verify, put, read back and
    compare. ``release(outcome)`` re-checks the owner, stamps the file with the
    outcome (committed, best-effort) and THEN deletes the entry (D36 #3), never
    raises, and is a no-op when the lease is not held. The store, the file-layer ``commit`` and the clock are injectable
    for tests; the defaults are the named Dict and ``VOL.commit``. Neither a
    ``flock`` (``run_identity.OutDirLock``, kernel-held on one mount) nor a
    volume file (separate snapshots, last-writer-wins) can be exclusive across
    containers — the Dict's atomic put can."""

    def __init__(self, run_dir: Path, *, run_id: str, app: str, owner: str | None = None,
                 owner_kind: str | None = None, store=None, commit=None, clock=time.time) -> None:
        self.run_dir = Path(run_dir)
        self.run_id, self.app = run_id, app
        if owner is None:
            owner, owner_kind = owner_of()
        self.owner, self.owner_kind = owner, owner_kind or "uuid"
        self._store = store
        self.commit = commit if commit is not None else VOL.commit
        self.clock = clock
        self.held = False
        self.lost = False
        self.indeterminate = False
        self.file_errors = 0
        self._last_beat = 0.0

    @property
    def store(self):
        if self._store is None:
            self._store = _lease_dict()
        return self._store

    @property
    def path(self) -> Path:
        return self.run_dir / LEASE_FILE

    def _where(self) -> str:
        return f"run {self.run_id!r} (Dict {LEASE_DICT_NAME!r}; record {self.path})"

    @staticmethod
    def _parse(value):
        """The record, ``None`` when absent; ``False`` when not a lease record."""
        if value is None:
            return None
        if not isinstance(value, dict) or value.get("schema") != LEASE_SCHEMA \
                or not isinstance(value.get("owner"), str) \
                or not isinstance(value.get("heartbeat_epoch"), (int, float)):
            return False
        return value

    def _indeterminate(self, what: str, e: Exception) -> AttemptLeaseIndeterminate:
        self.held, self.indeterminate = False, True
        return AttemptLeaseIndeterminate(
            f"{self._where()}: the lease Dict {what} ({type(e).__name__}: {e}), so whether this attempt "
            f"({self.owner}) owns the run cannot be determined. Failing closed: nothing written, the lease "
            "dropped, the entry and the record left as they are; stopping without syncing anything more.")

    def _read(self, when: str):
        """The run's entry (``None`` when absent); an unreadable Dict is INDETERMINATE."""
        try:
            return self.store.get(self.run_id)
        except Exception as e:  # noqa: BLE001
            raise self._indeterminate(f"could not be read {when}", e) from e

    def _put_new(self, rec: dict, when: str) -> bool:
        """The ATOMIC create: True when this call created the entry."""
        try:
            return bool(self.store.put(self.run_id, rec, skip_if_exists=True))
        except Exception as e:  # noqa: BLE001
            raise self._indeterminate(f"could not be written {when}", e) from e

    def _write_file(self, rec: dict) -> None:
        """The human-readable record on the volume — best-effort: the Dict entry
        is the authority, so a failure here is counted and printed, never raised."""
        try:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(f"{LEASE_FILE}.tmp-{self.owner}")
            tmp.write_bytes(canonical_json_bytes(rec))
            os.replace(tmp, self.path)
            self.commit()
        except Exception as e:  # noqa: BLE001
            self.file_errors += 1
            print(f"[lease] WARNING: the record {self.path} could not be written ({type(e).__name__}: "
                  f"{e}); the Dict entry is the authority", flush=True)

    def holder(self) -> dict | None:
        rec = self._parse(self._read("at inspection"))
        return rec if isinstance(rec, dict) else None

    def _acquired(self, rec: dict) -> "AttemptLease":
        self.held, self.lost = True, False
        self._last_beat = float(rec["heartbeat_epoch"])
        self._write_file(rec)
        return self

    def acquire(self) -> "AttemptLease":
        rec = lease_record(run_id=self.run_id, owner=self.owner, owner_kind=self.owner_kind,
                           app=self.app, now=self.clock())
        if self._put_new(rec, "at acquisition"):
            return self._acquired(rec)
        raw = self._read("at acquisition")
        incumbent = self._parse(raw)
        if incumbent is False:
            raise AttemptLeaseHeld(
                f"{self._where()} holds an unparseable entry or one of another schema than "
                f"{LEASE_SCHEMA!r} ({raw!r:.200}); refusing to guess whether an attempt is live. To release "
                f"it by hand once you are certain none is: {_lease_remedy(self.run_id, self.run_dir)}.")
        if incumbent is None:
            # released between this attempt's put and its read: one more atomic try
            if self._put_new(rec, "at acquisition"):
                return self._acquired(rec)
            raise AttemptLeaseHeld(
                f"{self._where()}: another launch won the race for the entry: "
                f"{_lease_summary(self.holder())}. Nothing under the run was touched.")
        age = self.clock() - float(incumbent["heartbeat_epoch"])
        if age < LEASE_STALE_S:
            raise AttemptLeaseHeld(
                f"run {self.run_dir} is owned by a LIVE attempt: {_lease_summary(incumbent)} — its "
                f"heartbeat is {age:.0f}s old (< {LEASE_STALE_S}s). Two attempts syncing one run "
                "would race its progress/response files, last writer wins. Wait for it to finish (or "
                "stop it). A lease is not taken over automatically, even once stale: overlapping "
                "resumptions are not supported. If the attempt is dead, release it by hand — "
                f"{_lease_remedy(self.run_id, self.run_dir)}. Nothing under the run was touched.")
        raise AttemptLeaseHeld(
            f"run {self.run_dir} is owned by an attempt whose heartbeat is STALE ({age:.0f}s old, >= "
            f"{LEASE_STALE_S}s): {_lease_summary(incumbent)}. It is not taken over automatically: "
            "overlapping resumptions are not supported, and a stale lease needs deliberate cleanup — "
            f"{_lease_remedy(self.run_id, self.run_dir)}. Nothing under the run was touched.")

    def verify_owner(self) -> dict:
        """THE FENCE (code review AC-2), run immediately before every durable
        write and by every heartbeat: the entry must be this attempt's."""
        if not self.held:
            raise AttemptLeaseLost(f"{self._where()} is not held by this attempt ({self.owner})")
        cur = self._parse(self._read("at the ownership check"))
        if not isinstance(cur, dict) or cur.get("owner") != self.owner:
            self.held, self.lost = False, True
            raise AttemptLeaseLost(
                f"{self._where()} is no longer this attempt's ({self.owner}): "
                f"{'no entry' if cur is None else _lease_summary(cur)} — another launch took the run "
                "over; stopping without syncing anything more")
        return cur

    def refresh(self) -> None:
        """A fresh heartbeat: verify the owner, put, read back and compare."""
        cur = self.verify_owner()
        now = self.clock()
        cur["heartbeat_at"], cur["heartbeat_epoch"] = _iso(now), float(now)
        try:
            self.store.put(self.run_id, cur)
        except Exception as e:  # noqa: BLE001
            raise self._indeterminate("could not be written at the heartbeat", e) from e
        back = self._parse(self._read("at the heartbeat read-back"))
        if back != cur:
            self.held, self.lost = False, True
            raise AttemptLeaseLost(
                f"{self._where()}: after this attempt's ({self.owner}) heartbeat the entry is another "
                f"attempt's ({_lease_summary(back)}) — another launch took the run over; stopping "
                "without syncing anything more")
        self._last_beat = now
        self._write_file(cur)

    def beat(self) -> None:
        """Refresh when :data:`LEASE_HEARTBEAT_S` has passed since the last one."""
        if self.clock() - self._last_beat >= LEASE_HEARTBEAT_S:
            self.refresh()

    def release(self, outcome) -> bool:
        """Re-check the owner, DELETE the entry and stamp the record with
        ``released_at`` + ``outcome`` (True); False when not held, not ours any
        more, or the Dict could not be read/updated (nothing deleted then).
        Never raises: a release failure must not mask the run's own outcome."""
        if not self.held:
            return False
        try:
            cur = self._parse(self.store.get(self.run_id))
            if not isinstance(cur, dict) or cur.get("owner") != self.owner:
                print(f"[lease] {self._where()} is not this attempt's any more "
                      f"({'no entry' if cur is None else _lease_summary(cur)}); leaving it", flush=True)
                return False
            now = self.clock()
            cur.update(heartbeat_at=_iso(now), heartbeat_epoch=float(now), released_at=_iso(now),
                       outcome=None if outcome is None else str(outcome))
            self._write_file(cur)                 # the record and its commit FIRST ...
            self.store.pop(self.run_id, None)     # ... then the entry goes: nothing lands after "free"
            return True
        except Exception as e:  # noqa: BLE001
            print(f"[lease] WARNING: release failed ({type(e).__name__}: {e}); nothing deleted", flush=True)
            return False
        finally:
            self.held = False

def write_run_ids_file(run_local: Path, goal_ids_file_bytes: bytes) -> Path:
    """The run's own copy of the ids file — what every driver's
    ``--goal-ids-file`` reads — is the ORIGINAL bytes, never a reconstruction,
    so its sha256 IS the identity's ``goal_ids_file_sha256``."""
    run_local = Path(run_local)
    run_local.mkdir(parents=True, exist_ok=True)
    p = run_local / "goal_ids.txt"
    p.write_bytes(goal_ids_file_bytes)
    return p


def has_driver_tree(dest: Path, arms=ARMS) -> bool:
    """Whether a previous attempt left a driver tree to seed from
    (``seed_from_previous`` runs only then; an identity-only run dir restarts
    the pre-driver setup unseeded and files no ``attempts/NN/`` entry)."""
    return any((Path(dest) / a).is_dir() for a in arms)


def _needs_resume_message(dest: Path, arms) -> str:
    return (f"{dest} already holds a driver tree from a previous attempt (arms "
            f"{[a for a in arms if (Path(dest) / a).is_dir()]}); relaunching it without --resume "
            "would start from an EMPTY local tree, regenerate every goal and let the sync "
            "overwrite terminal refused:* records with stochastic successes. Pass --resume (under "
            "a NEW -n app-name suffix) to continue this run, or use a NEW --run. Nothing under the "
            "run was touched.")


def run_prelude(dest: Path, run_local: Path, record: dict, *, resume: bool,
                seed_fn=None, commit=None, arms=ARMS, lease_fn=None, reload_fn=None) -> dict:
    """THE ORDER the setup must follow, as one function: (1) the identity is
    COMPARED (:func:`ensure_run_identity`) BEFORE anything else — a mismatch
    raises :class:`RunIdentityMismatch` with nothing under ``dest`` touched and
    ``seed_fn`` never called; when no lease is involved it is also written here;
    (2) a run dir that already holds a driver tree (:func:`has_driver_tree`)
    REQUIRES ``--resume`` — without it :class:`RunNeedsResume` is raised, again
    with nothing touched (an identity-only dir admits the plain retry); (3) the
    attempt lease is taken (``lease_fn(dest)``) BEFORE anything is seeded or
    written — its refusal propagates with ``seed_fn`` never called; (4) holding
    the lease, the volume is RELOADED (``reload_fn``) and the identity
    compare-or-write and the driver-tree check are RE-RUN against that fresh
    snapshot (the previous owner's last commit — a terminal refusal, a whole
    driver tree, even another identity — may have landed after the first
    snapshot; a refusal here releases the lease); (5) only then, and only under
    ``--resume`` and only when a driver tree exists, the run is seeded from the
    fresh snapshot (``seed_fn(dest, run_local, arms)``); an identity-only run
    dir restarts the setup unseeded. Returns ``{"identity": "written"|"equal",
    "seed": report | None, "note": str | None, "lease": what lease_fn returned
    | None}``."""
    seed_fn = seed_fn or seed_from_previous
    dest = Path(dest)
    status = ensure_run_identity(dest, record, commit=commit, write=lease_fn is None)
    tree = has_driver_tree(dest, arms)
    if tree and not resume:
        raise RunNeedsResume(_needs_resume_message(dest, arms))
    out: dict = {"identity": status, "seed": None, "note": None, "lease": None}
    if lease_fn is not None:
        out["lease"] = lease_fn(dest)
        try:
            if reload_fn is not None:
                reload_fn()
            status = ensure_run_identity(dest, record, commit=commit)   # compare, or WRITE — as the owner
            tree = has_driver_tree(dest, arms)
            if tree and not resume:
                raise RunNeedsResume(_needs_resume_message(dest, arms))
        except BaseException as e:
            out["lease"].release("IDENTITY_MISMATCH" if isinstance(e, RunIdentityMismatch)
                                 else "NEEDS_RESUME" if isinstance(e, RunNeedsResume)
                                 else f"PRELUDE_FAILED:{type(e).__name__}")
            raise
        out["identity"] = status
    if not resume:
        return out
    try:
        if tree:
            Path(run_local).mkdir(parents=True, exist_ok=True)
            out["seed"] = seed_fn(dest, Path(run_local), list(arms))
        else:
            out["note"] = ("identity-only run dir: no driver tree to seed from; restarting "
                           "setup unseeded (no attempts/NN entry filed)")
    except BaseException as e:
        if out["lease"] is not None:          # the lease this prelude took goes back
            out["lease"].release(f"SEED_FAILED:{type(e).__name__}")
        raise
    return out

def check_attestation(record: dict, endpoint: dict, route_hash_by_cell: dict,
                      *, st=None) -> list[str]:
    """Mismatches between the identity record and what the standup ATTESTS
    (``endpoint.json`` + the routing table): [] = ok, else the named problems.
    The existing pre-driver check, against the RECORD rather than argv."""
    st = st or _served_target_module()
    problems: list[str] = []
    expected_decl = {hid: pc["serve_declaration"] for hid, pc in record["per_cell"].items()
                     if pc.get("serve_declaration")}
    got_decl = endpoint.get(st.ATTEST_SERVE_DECLARATIONS_KEY)
    err = endpoint.get(st.ATTEST_SERVE_DECLARATIONS_ERROR_KEY)
    if expected_decl or got_decl:
        if err:
            problems.append(f"{st.ATTEST_SERVE_DECLARATIONS_ERROR_KEY}={err!r}")
        if not isinstance(got_decl, dict) or dict(got_decl) != expected_decl:
            problems.append(f"{st.ATTEST_SERVE_DECLARATIONS_KEY}: attested {got_decl!r} != "
                            f"record {expected_decl!r}")
    for hid, pc in record["per_cell"].items():
        if not pc.get("serve_declaration") and isinstance(got_decl, dict) and hid in got_decl:
            problems.append(f"undeclared cell {hid!r} appears in the attested declarations")
    mode = endpoint.get("lens_reduction_mode")
    if mode != record["reduction_mode"]:
        problems.append(f"lens_reduction_mode: attested {mode!r} != record "
                        f"{record['reduction_mode']!r} ({endpoint.get('lens_reduction_mode_error')})")
    pol = endpoint.get(st.ATTEST_LENS_SALT_POLICY_KEY)
    if pol != record["lens_salt_policy"]:
        problems.append(f"{st.ATTEST_LENS_SALT_POLICY_KEY}: attested {pol!r} != record "
                        f"{record['lens_salt_policy']!r}")
    dig = endpoint.get(st.ATTEST_OBJECT_SHA256_BY_CELL_KEY)
    if dig != record["object_artifact_sha256_by_cell"]:
        problems.append(f"{st.ATTEST_OBJECT_SHA256_BY_CELL_KEY}: attested {dig!r} != record "
                        f"{record['object_artifact_sha256_by_cell']!r}")
    for hid, pc in record["per_cell"].items():
        if route_hash_by_cell.get(hid) != pc["intervention_hash"]:
            problems.append(f"routing table intervention_hash[{hid!r}] = "
                            f"{route_hash_by_cell.get(hid)!r} != record {pc['intervention_hash']!r}")
    for hid, cl in record.get("control_legs", {}).items():
        if route_hash_by_cell.get(hid) != cl["intervention_hash"]:
            problems.append(f"routing table intervention_hash[{hid!r}] = "
                            f"{route_hash_by_cell.get(hid)!r} != record {cl['intervention_hash']!r}")
    return problems


def resolve_cells(tree: str, cells: list[str]) -> dict:
    """IN-CONTAINER, CPU: enumerate the run's cells over the STAGED tree exactly
    as the standup will (registry + hashed specs), and read per cell: the
    harness id, the intervention hash, the served ``positions`` and the serve
    declaration (from the resolved spec), the artifact path + sha256. Asserts
    the fitted blobs' ``positions == "response"`` and header length (the
    fit/serve preflight), DISTINCT hashes across the spellings of one base, and
    the ``none__lens0`` twin. Returns ``{per_cell, control_legs,
    digests_by_cell, digests_by_tt, target}``."""
    import dataclasses
    import torch
    import yaml
    from transformers import AutoConfig, AutoTokenizer

    sys.path[:0] = [f"{tree}/scripts", f"{tree}/src"]
    os.environ["TT_TWO_SI_DEFENSES_DIR"] = OBJECTS_X1
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.petri import registry_adapter as ra
    from sisteer.capeval.petri.registry_adapter import traintest_cell_source
    from sisteer.capeval.serving import serve_declaration as sd
    from sisteer.chat_markers import generation_header_ids
    from sisteer.eval.defense_cell import sha256_file

    cfg_path = Path(f"{tree}/{CFG_REL}")
    cfg = yaml.safe_load(cfg_path.read_text())
    model, rev = cfg["model"], cfg["model_revision"]
    hids = {c: ra.harness_cell_id(c) for c in cells}
    tok = AutoTokenizer.from_pretrained(model, revision=rev)
    tsha = cm.sha256_hex16(tok.chat_template.encode())
    mcfg = AutoConfig.from_pretrained(model, revision=rev)
    hidden = getattr(mcfg, "hidden_size", None) or mcfg.text_config.hidden_size
    spec0 = cm.load_petri_spec(Path(tree) / "configs/capeval/manifests/petri_twosi.yaml")
    spec_cells = ["none"] + [hids[c] for c in cells if c != "none"]
    spec = dataclasses.replace(
        spec0, target_model=model, target_revision=rev, serving_mode="interleaved",
        cells=spec_cells,
        serving_launch_flags=[f for f in spec0.serving_launch_flags if f != "--enable-lora"],
        scenario_bundle_hash="ab" * 8, serving_impl_commit="fedcba9876" * 4,
        chat_template_hash=tsha)
    src = traintest_cell_source(spec, repo_root=Path(tree), config_path=cfg_path,
                                defenses_dir=Path(OBJECTS_X1))
    bf = {k: f"# {k}\n".encode() for k in cm.petri_expected_behavior_keys(spec)}
    if spec.prompt_provenance.startswith("override@"):
        from sisteer.capeval.tiers.petri_runner import resolve_prompt_files
        prompt_files = resolve_prompt_files(
            spec, override_root=Path(tree) / "configs/capeval/petri/prompts")
    else:
        prompt_files = {"auditor": b"# a\n", "judge": b"# j\n"}
    cs = dc.enumerate_cells(spec, src, behavior_files=bf, prompt_files=prompt_files)
    if cs.absent:
        raise RuntimeError(f"absent cells: {[(a.cell_id, a.reason) for a in cs.absent]}")
    by_id = {c.cell_id: c for c in cs.canonical}
    genlen = len(generation_header_ids(tok))
    per_cell: dict = {}
    digests_by_cell: dict = {}
    digests_by_tt: dict = {}
    for c in cells:
        hid = hids[c]
        cell = by_id[hid]
        entry = src.get(hid)
        spec_r = dict(entry.resolved_spec or {}) if entry is not None else {}
        decl = sd.declarations_of(spec_r) if spec_r else {}
        rec = {"tt_cell_id": c, "virtual_model": cell.virtual_model_name,
               "intervention_hash": cell.intervention_hash,
               "serving_identity": cell.serving_identity,
               "positions": spec_r.get(sd.POSITIONS_KEY),
               "serve_declaration": decl, "artifact_sha256": None,
               "recorded_fit_positions": None, "response_header_len": None}
        if cell.object_path:
            sha = sha256_file(cell.object_path)
            blob = torch.load(cell.object_path, map_location="cpu", weights_only=True)
            fit_pos, H = blob.get("positions"), blob.get("response_header_len")
            if fit_pos != "response" or H != genlen:
                raise RuntimeError(f"{c}: fitted blob positions={fit_pos!r} H={H} vs the "
                                   f"template's header {genlen} — the fit/serve guard would refuse")
            rec.update(artifact_sha256=sha, recorded_fit_positions=fit_pos, response_header_len=H)
            digests_by_cell[hid] = sha
            digests_by_tt[c] = sha
        per_cell[hid] = rec
    # DISTINCT hashes across the spellings of one base
    by_base: dict = {}
    for c in cells:
        base, _v = sd.split_variant(c)
        by_base.setdefault(base, []).append(per_cell[hids[c]]["intervention_hash"])
    for base, hs in by_base.items():
        if len(set(hs)) != len(hs):
            raise RuntimeError(f"{base}: its spellings do not have distinct intervention hashes: {hs}")
    controls = {c.cell_id: {"intervention_hash": c.intervention_hash, "control_of": c.control_of,
                            "virtual_model": c.virtual_model_name} for c in cs.controls}
    if "none__lens0" not in controls:
        raise RuntimeError(f"no none__lens0 twin among the control legs: {sorted(controls)}")
    target = {"model": model, "revision": rev, "config": CFG_REL, "chat_template_sha16": tsha,
              "hidden_size": int(hidden), "generation_header_len": genlen}
    return {"per_cell": per_cell, "control_legs": controls, "digests_by_cell": digests_by_cell,
            "digests_by_tt": digests_by_tt, "target": target,
            "vmodels": sorted(c.virtual_model_name for c in cs.all_served())}


def attack_settings(cfg: dict, *, jr) -> dict:
    """The attack settings the identity binds, from the staged
    config exactly as the drivers will read them."""
    from sisteer.eval.attacks import multiturn as mt
    samp = (cfg.get("attacks", {}).get("attacker_sampling", {}) or {}).get("crescendo") \
        or jr._DEFAULT_ATTACKER_SAMPLING["crescendo"]
    return {"name": "crescendo", "max_turns": mt.CRESCENDO_MAX_TURNS,
            "max_backtracks": mt.CRESCENDO_MAX_BACKTRACKS,
            "attacker_temperature": float(samp["temperature"]),
            "attacker_top_p": float(samp["top_p"]),
            "target_temperature": 0.0,
            "target_max_new_tokens": int(jr._target_max_new_tokens(cfg)),
            "attacker_max_new_tokens": int(jr._attacker_max_new_tokens(cfg)),
            "forward_cap": jr._forward_cap(cfg),
            "attacker_model": ATTACKER_MODEL, "attacker_revision": ATTACKER_SNAPSHOT,
            "attacker_max_model_len": ATTACKER_MAX_MODEL_LEN}


def geometry_record(geom) -> dict:
    return {"eval": TARGET_EVAL, "model_key": TARGET_KEY,
            "max_model_len": int(geom.max_model_len), "max_num_seqs": int(geom.max_num_seqs),
            "window_source": geom.window_source, "seqs_source": geom.seqs_source}


# --------------------------------------------------------------------------- #
# run tree sync / resume (grokmt_served.py's, with the refused count)          #
# --------------------------------------------------------------------------- #

class RunSyncer:
    """Incremental copy of the run tree to the volume (a file is re-copied only
    when its (size, mtime_ns) changed); ``VOL.commit()`` is what makes a synced
    goal survive a container kill."""

    def __init__(self, src: Path, dest: str) -> None:
        self.src, self.dest = Path(src), Path(dest)
        self.seen: dict[str, tuple[int, int]] = {}

    def copy_changed(self) -> int:
        n = 0
        for p in self.src.rglob("*"):
            if not p.is_file():
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


def _sync(syncer: RunSyncer) -> bool:
    """Copy the changed files and COMMIT the volume; True only when both
    succeeded (a failure is printed, never raised — the caller decides what a
    failed sync means; for the run's final sync it means the lease stays
    held)."""
    try:
        n = syncer.copy_changed()
        VOL.commit()
        print(f"[sync] {n} file(s) committed to {syncer.dest}", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[sync] WARNING: {type(e).__name__}: {e}", flush=True)
        return False


def _fenced_sync(lease: AttemptLease, syncer: RunSyncer) -> bool:
    """EVERY durable write is fenced (code review AC-2): the Dict owner is
    re-verified immediately before the sync — another owner ->
    :class:`AttemptLeaseLost`, an unreadable Dict ->
    :class:`AttemptLeaseIndeterminate` — and nothing is copied or committed
    then; the run body's lease handler stops the run. Returns whether the sync
    was committed."""
    lease.verify_owner()
    return _sync(syncer)


ATTEMPT_ROOT_DIRS = ("srv", "driver_logs")
ATTEMPT_ROOT_FILES = ("run_provenance.json", "summary.json", "driver_cmds.json", "goal_ids.txt",
                      "attestation_check.json")
TIMED_JSONL = ("attacker_calls.jsonl", "target_timing.jsonl", "scorer_calls.jsonl")
ERRORED_PREFIX = "error:"
REFUSED_PREFIX = "refused:"


def progress_counts(progress_path: Path) -> dict:
    """``{'done', 'errored', 'refused', 'rows'}`` for one driver's
    ``progress.jsonl``: ``done`` = distinct goals with a terminal row,
    ``errored`` = those whose LAST row is ``error:*`` (retried on resume),
    ``refused`` = those whose LAST row is ``refused:*`` (TERMINAL — covered,
    never retried, never counted in ``errored``), ``rows`` = raw line count."""
    last: dict[str, str] = {}
    rows = 0
    if progress_path.exists():
        for line in progress_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                gid = rec["goal_id"]
            except (json.JSONDecodeError, KeyError):
                continue
            rows += 1
            last[gid] = str(rec.get("outcome", ""))
    errored = sum(1 for o in last.values() if o.startswith(ERRORED_PREFIX))
    refused = sum(1 for o in last.values() if o.startswith(REFUSED_PREFIX))
    return {"done": len(last), "errored": errored, "refused": refused, "rows": rows}


def sqlite_intact(path: Path) -> bool:
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            row = con.execute("PRAGMA integrity_check").fetchone()
            return bool(row) and row[0] == "ok"
        finally:
            con.close()
    except sqlite3.Error:
        return False


def _iso_to_epoch(s):
    if not s:
        return None
    import calendar
    try:
        return float(calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")))
    except ValueError:
        return None


def seed_from_previous(dest: Path, run_local: Path, arms: list[str]) -> dict:
    """Seed ``run_local`` from ``dest`` (the volume copy of an interrupted
    attempt) so the drivers RESUME: every arm's driver tree is copied (a torn
    ``memory.db`` set aside), the previous attempt's standup/driver logs are
    filed under ``attempts/<NN>/``, and the report carries per-driver counts
    (``refused`` its own count) and the previous attempt's time window. Raises
    if ``dest`` holds no prior driver tree — the caller checks
    :func:`has_driver_tree` first (an identity-only run dir is NOT seeded)."""
    dest, run_local = Path(dest), Path(run_local)
    if not has_driver_tree(dest, arms):
        raise RuntimeError(f"--resume: no prior driver tree under {dest} for arms {arms}")
    prior = sorted(int(p.name) for p in (dest / "attempts").glob("[0-9][0-9]")
                   if p.is_dir()) if (dest / "attempts").is_dir() else []
    prev_attempt = (max(prior) if prior else 0) + 1
    this_attempt = prev_attempt + 1
    rep: dict = {"attempt": this_attempt, "previous_attempt": prev_attempt,
                 "seeded": {}, "torn_memory_dbs": [], "covered_total": 0,
                 "refused_total": 0, "errored_to_retry_total": 0}
    last_t = 0.0
    for arm in arms:
        if not (dest / arm).is_dir():
            continue
        shutil.copytree(dest / arm, run_local / arm, dirs_exist_ok=True)
        for pj in sorted((run_local / arm).glob(f"{PRIVATE_RUN}/*/*/shard*/progress.jsonl")):
            key = f"{arm}/" + "/".join(pj.relative_to(run_local / arm / PRIVATE_RUN).parts[:-1])
            c = progress_counts(pj)
            rep["seeded"][key] = c
            rep["covered_total"] += c["done"] - c["errored"]      # refused goals ARE covered
            rep["refused_total"] += c["refused"]
            rep["errored_to_retry_total"] += c["errored"]
        for sd_ in (run_local / arm).glob(f"{PRIVATE_RUN}/*/*/shard*"):
            for name in TIMED_JSONL:
                f = sd_ / name
                if f.exists():
                    for line in f.read_text(errors="replace").splitlines():
                        try:
                            last_t = max(last_t, float(json.loads(line).get("t_start") or 0))
                        except (json.JSONDecodeError, ValueError, AttributeError):
                            continue
        for db in (run_local / arm).rglob("memory.db"):
            if not sqlite_intact(db):
                torn = db.with_name(f"memory.db.torn-attempt{prev_attempt:02d}")
                db.rename(torn)
                rep["torn_memory_dbs"].append(str(db.relative_to(run_local)))
    filed = run_local / "attempts" / f"{prev_attempt:02d}"
    filed.mkdir(parents=True, exist_ok=True)
    for name in ATTEMPT_ROOT_DIRS:
        if (dest / name).is_dir():
            shutil.copytree(dest / name, filed / name, dirs_exist_ok=True)
    for name in ATTEMPT_ROOT_FILES:
        if (dest / name).is_file():
            shutil.copyfile(dest / name, filed / name)
    prev_prov = {}
    if (dest / "run_provenance.json").is_file():
        try:
            prev_prov = json.loads((dest / "run_provenance.json").read_text())
        except json.JSONDecodeError:
            prev_prov = {}
    attempts = list(prev_prov.get("attempts") or [])
    if not attempts:
        attempts = [{"attempt": 1, "started_at": prev_prov.get("started_at"),
                     "started_at_epoch": _iso_to_epoch(prev_prov.get("started_at"))}]
    attempts[-1]["ended_at_epoch"] = last_t or None
    rep["attempts"] = attempts
    rep["previous_window_epoch"] = [attempts[-1].get("started_at_epoch"), last_t or None]
    return rep


# --------------------------------------------------------------------------- #
# the attacker (grokmt_served.py's per-job vLLM, verbatim)                     #
# --------------------------------------------------------------------------- #

def _attacker_serving_options():
    from sisteer.eval.attacks import attacker_serving as serving
    opts, sources = serving.resolve()
    return serving, opts, sources


def _attacker_serving_manifest_after_health(serving, opts, sources, proc, port: int,
                                            log: Path, cmd: list[str]) -> None:
    import urllib.request
    manifest = log.with_suffix(".serving.json")
    while proc.poll() is None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if 200 <= r.status < 300:
                    break
        except Exception:  # noqa: BLE001
            pass
        time.sleep(3)
    if proc.poll() is not None:
        return
    try:
        engine = serving.parse_engine_log(log.read_text(errors="replace"))
        serving.write_manifest(
            manifest, opts=opts, sources=sources, engine=engine,
            model=_attacker_snapshot_dir(), served_name=ATTACKER_MODEL,
            endpoint=f"http://127.0.0.1:{port}", log_path=str(log),
            argv=opts.vllm_argv(model_role="attacker"),
            extra={"gpu": "0", "standup_cmd": cmd, "snapshot": ATTACKER_SNAPSHOT})
        print(f"[attacker] serving manifest verified: {manifest}", flush=True)
    except serving.AttackerServingError as exc:
        msg = f"[attacker] FATAL: engine contradicts the requested serving options: {exc}"
        print(msg, flush=True)
        with log.open("a") as fh:
            fh.write(msg + "\n")
        log.with_suffix(".serving.error").write_text(msg + "\n")
        proc.kill()


class AttackerServingUnverified(RuntimeError):
    """The attacker answered /health but no VERIFIED serving manifest followed."""


def _wait_attacker_serving_manifest(log: Path, proc, timeout_s: float = 600.0,
                                    poll_s: float = 2.0, on_tick=None) -> Path:
    """``on_tick`` (the lease heartbeat) runs once per poll."""
    manifest = log.with_suffix(".serving.json")
    error = log.with_suffix(".serving.error")
    deadline = time.time() + timeout_s
    while True:
        if on_tick is not None:
            on_tick()
        if manifest.exists():
            return manifest
        if error.exists():
            raise AttackerServingUnverified(error.read_text(errors="replace").strip()[-4000:])
        if proc is not None and proc.poll() is not None:
            raise AttackerServingUnverified(
                f"attacker vLLM exited (rc={proc.returncode}) before its serving manifest was written")
        if time.time() > deadline:
            raise AttackerServingUnverified(
                f"no verified attacker serving manifest at {manifest} after {timeout_s:.0f}s")
        time.sleep(poll_s)


def _start_attacker(port: int, log: Path) -> subprocess.Popen:
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "0"
    serving, opts, sources = _attacker_serving_options()
    cmd = ["vllm", "serve", _attacker_snapshot_dir(),
           "--served-model-name", ATTACKER_MODEL, "--port", str(port),
           "--host", "127.0.0.1", "--gpu-memory-utilization", "0.85",
           "--max-model-len", str(ATTACKER_MAX_MODEL_LEN),
           *opts.vllm_argv(model_role="attacker")]
    print("[attacker]", " ".join(cmd), flush=True)
    print("[attacker] serving options (configs/attacker_serving.json + flags):",
          json.dumps(opts.as_dict()), flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=log.open("w"), stderr=subprocess.STDOUT)
    threading.Thread(target=_attacker_serving_manifest_after_health,
                     args=(serving, opts, sources, proc, port, log, cmd),
                     daemon=True, name="attacker-serving-manifest").start()
    return proc


# --------------------------------------------------------------------------- #
# the target standup and the drivers                                           #
# --------------------------------------------------------------------------- #

def _geometry_argv(geom) -> list[str]:
    """The geometry part of the standup argv (grokmt_served's rule): dimensions
    travel BY SOURCE — a derived factor is re-derived by the standup from
    ``--eval`` / ``--model-key``; only a recorded OVERRIDE is passed explicitly."""
    argv: list = []
    if geom.window_source == "override":
        argv += ["--max-model-len", str(geom.max_model_len)]
    if geom.seqs_source == "override":
        argv += ["--max-num-seqs", str(geom.max_num_seqs)]
    if geom.eval:
        argv += ["--eval", str(geom.eval)]
    else:
        argv = ["--max-model-len", str(geom.max_model_len),
                "--max-num-seqs", str(geom.max_num_seqs)]
    if geom.model_key:
        argv += ["--model-key", str(geom.model_key)]
    if getattr(geom, "overridden", False) and (geom.over_budget or not geom.is_identity):
        argv += ["--allow-over-budget"]
    return argv


def _target_standup_cmd(tree: str, srv: Path, stop: Path, geom, *, cells: list[str],
                        digests_by_tt: dict) -> list[str]:
    """The standup argv: ``--cell`` per cell (``none`` included; the
    twin is derived), ``--expect-object-sha256 <tt_cell>=<sha>`` for EVERY
    fitted spelling (variants of one base share its artifact's sha, keyed by
    each spelling), the crescendo geometry entry, the model key, and
    ``--lens-salt-policy request_unique_all``."""
    cmd = [sys.executable, f"{tree}/scripts/capctrl_serving_up.py",
           "--repo-dir", tree, "--config", f"{tree}/{CFG_REL}"]
    for c in cells:
        cmd += ["--cell", c]
    for tt, sha in digests_by_tt.items():
        cmd += ["--expect-object-sha256", f"{tt}={sha}"]
    cmd += ["--defenses-dir", OBJECTS_X1, "--out", str(srv), "--stop-file", str(stop),
            *_geometry_argv(geom),
            "--lens-salt-policy", LENS_SALT_POLICY,
            "--gpu-mem-util", "0.90", "--ready-timeout-s", "2700",
            # TODO(native-serving-default §8.6): this launcher is not a capability / tinycap surface, so it keeps the
            # lens standup EXPLICITLY (the standup's own default is the native engine since 2026-09-13; an
            # argv without the flag would silently stand up the other engine). The native reading it lacks:
            # endpoint.json.engine / /v1/models petri_proxy.engine.kind held to the requested path, and
            # `lens_reduction_mode` read as not_applicable under lora_native (tinycap_served.py has both).
            "--serving-path", "lens"]
    return cmd


def _standup_env(tree: str, gpu: str = "1") -> dict:
    """The standup's environment: the reduction mode SET explicitly (the
    worker reads it once at import and only attests afterwards), the
    target GPU."""
    env = _env(tree)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env[STACKED_REDUCTIONS_ENV] = STACKED_REDUCTIONS_VALUE
    return env


def _start_target(tree: str, srv: Path, geom, *, cells, digests_by_tt,
                  gpu: str = "1") -> tuple[subprocess.Popen, Path]:
    stop = srv / "STOP"
    cmd = _target_standup_cmd(tree, srv, stop, geom, cells=cells, digests_by_tt=digests_by_tt)
    print("[target]", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=_standup_env(tree, gpu),
                            stdout=(srv / "standup.log").open("w"), stderr=subprocess.STDOUT)
    return proc, stop


def _driver_cmd(tree: str, *, cell: str, sub: int, subshards: int, ids_path: str,
                private_root: str, proxy_url: str, rt: str, attacker_port: int,
                target_window: int, attacker_serving: str | None,
                attack: str = "crescendo") -> list[str]:
    """One driver: ``--positions response`` for the BASE fitted cell
    ONLY (the wave's `_gen` cells); a variant carries none (the variant IS
    the positions — jailbreak_run refuses the pair); ``none`` carries none.
    Never ``--allow-errored-goals`` at generate."""
    _ensure_sisteer_importable()
    from sisteer.capeval.serving.serve_declaration import split_variant
    base, variant = split_variant(cell)
    cmd = [sys.executable, f"{tree}/scripts/jailbreak_run.py", "generate",
           "--config", f"{tree}/{CFG_REL}", "--attack", attack, "--defense", cell,
           "--confirmatory", "--goal-ids-file", ids_path,
           "--shard", str(sub), "--nshards", str(subshards),
           "--private-root", private_root,
           "--target-engine", "served", "--target-proxy-url", proxy_url,
           "--target-routing-table", rt,
           "--target-max-model-len", str(int(target_window)),
           # THE TARGET DECODE IS PINNED GREEDY: this experiment's IDENTITY RECORD binds
           # attack_settings()["target_temperature"] = 0.0 and refuses a run whose settings differ,
           # so the driver must actually send greedy. jailbreak_run's default became "sample at the
           # served model's own generation config" on 2026-09-11; without this flag the record would
           # claim greedy while the run sampled -- the record lying, which is worse than a refusal.
           "--target-greedy",
           "--attacker-engine", "vllm", "--attacker-endpoint", f"http://127.0.0.1:{attacker_port}"]
    if base != "none" and variant is None:
        cmd += ["--positions", "response"]
    if attacker_serving:
        cmd += ["--attacker-serving", str(attacker_serving)]
    return cmd


def _kill(proc) -> None:
    if proc is not None and proc.poll() is None:
        proc.kill()


def _provenance(*, run_id, sha, geom, goal_ids, cells, attempts, now_iso, resume, summary,
                record) -> dict:
    return {
        "run_id": run_id, "tree_sha": sha, "config": CFG_REL, "cells": list(cells),
        "attacks": list(ATTACKS), "arms": list(ARMS),
        "object_dir": OBJECTS_X1,
        "run_identity": record,
        "attacker_qwen": {"model": ATTACKER_MODEL, "snapshot": ATTACKER_SNAPSHOT,
                          "max_model_len": ATTACKER_MAX_MODEL_LEN, "thinking": "off"},
        "target_geometry": geom.as_dict(),
        "goal_ids": list(goal_ids),
        "topology": ("1 container, 2xH200: GPU0 attacker vLLM, GPU1 target vLLM+lens+"
                     "RoutingProxy (one standup, every cell a virtual model), drivers on loopback"),
        "started_at": attempts[0].get("started_at") or now_iso,
        "attempts": attempts,
        "resume": ({k: v for k, v in summary["resume"].items() if k != "attempts"}
                   if resume and "resume" in summary else None)}


def _run_exp2_impl(sha: str, goal_ids: list[str], run_id: str, cells: list[str],
                   subshards: int = DEFAULT_SUBSHARDS, sync_every_s: int = 120,
                   resume: bool = False, target_window: int = 0, target_seqs: int = 0,
                   allow_over_budget: bool = False, *, ids_file_bytes: bytes | None = None,
                   app_name: str = "") -> dict:
    if not isinstance(ids_file_bytes, (bytes, bytearray)) or not ids_file_bytes:
        # the identity hashes the ORIGINAL ids file, never a reconstruction (E1/B2)
        raise RuntimeError("[fst] the launcher must pass the ids file's bytes (ids_file_bytes)")
    ids_file_bytes = bytes(ids_file_bytes)
    tree = _tree(sha)
    assert Path(f"{tree}/src/sisteer").is_dir(), tree
    sys.path[:0] = [f"{tree}/scripts", f"{tree}/src"]
    import yaml
    import jailbreak_run as jr
    st = _served_target_module(tree)
    geom = resolve_target_geometry(tree, target_window=target_window, target_seqs=target_seqs,
                                   allow_over_budget=allow_over_budget)
    print(geom.describe(), flush=True)
    cfg = yaml.safe_load(open(f"{tree}/{CFG_REL}"))
    dest = Path(f"{GEN}/{run_id}")
    run_local = Path(f"{RUN_LOCAL_ROOT}/{run_id}")
    summary: dict = {"run_id": run_id, "sha": sha, "cells": cells, "attacks": list(ATTACKS),
                     "subshards": subshards, "n_goals": len(goal_ids), "timeline": {},
                     "target_geometry": geom.as_dict(), "lens_salt_policy": LENS_SALT_POLICY,
                     "reduction_mode": REDUCTION_MODE}
    T0 = time.time()

    def mark(k):
        summary["timeline"][k] = round(time.time() - T0, 1)
        print(f"[t+{summary['timeline'][k]:.0f}s] {k}", flush=True)

    # 0. THE IDENTITY, before anything is seeded or overwritten; this first snapshot
    #    serves the cell resolution and the pre-lease compare — the prelude reloads AGAIN once the
    #    lease is held and seeds from that
    VOL.reload()
    resolved = resolve_cells(tree, cells)
    record = build_run_identity(
        sha=sha, goal_ids=goal_ids, goal_ids_file_bytes=ids_file_bytes, cells=cells,
        subshards=subshards, per_cell=resolved["per_cell"],
        control_legs=resolved["control_legs"], digests_by_cell=resolved["digests_by_cell"],
        geom_dict=geometry_record(geom), attack=attack_settings(cfg, jr=jr),
        target=resolved["target"])
    try:
        pre = run_prelude(
            dest, run_local, record, resume=resume, commit=VOL.commit,
            lease_fn=lambda d: AttemptLease(d, run_id=run_id, app=app_name, commit=VOL.commit).acquire(),
            reload_fn=VOL.reload)             # the seed comes from the POST-acquire snapshot
    except RunIdentityMismatch as e:
        summary["status"] = "IDENTITY_MISMATCH"
        summary["identity_err"] = str(e)[-4000:]
        print(f"[fst] REFUSED: {summary['identity_err']}", flush=True)
        return summary          # nothing under the run was touched, nothing written
    except RunNeedsResume as e:
        summary["status"] = "NEEDS_RESUME"
        summary["needs_resume_err"] = str(e)[-4000:]
        print(f"[fst] REFUSED: {summary['needs_resume_err']}", flush=True)
        return summary          # S-C1: a populated run is only ever continued with --resume
    except AttemptLeaseHeld as e:
        summary["status"] = "LEASE_HELD"
        summary["lease_err"] = str(e)[-4000:]
        print(f"[fst] REFUSED: {summary['lease_err']}", flush=True)
        return summary          # S-C2: another attempt owns the run; nothing seeded or stood up
    except AttemptLeaseIndeterminate as e:
        summary["status"] = "LEASE_INDETERMINATE"
        summary["lease_err"] = str(e)[-4000:]
        print(f"[fst] REFUSED: {summary['lease_err']}", flush=True)
        return summary          # the lease Dict could not be read/written: nothing seeded or stood up
    lease = pre["lease"]
    summary["lease"] = {"dict": LEASE_DICT_NAME, "file": str(lease.path), "owner": lease.owner,
                        "owner_kind": lease.owner_kind, "app": app_name}
    att_proc = tgt_proc = stop = None
    procs: dict[str, subprocess.Popen] = {}
    # the release decision: POSITIVE authorization only. `final_committed`
    # is set by the ONE event that frees the run — a final sync that copied AND committed;
    # `data_write_begun` is set before this attempt's first copy under dest, so an exit that cannot
    # show a committed final sync after that point keeps the lease
    sync_state = {"data_write_begun": False, "final_committed": False}
    summary["sync_failures"] = 0

    def stop_writers() -> None:
        """Every process of this attempt that writes under run_local: the
        drivers, then the target (STOP + kill), then the attacker."""
        for p in procs.values():
            if p.poll() is None:
                p.kill()
        if stop is not None:
            stop.touch()
        _kill(tgt_proc)
        _kill(att_proc)

    try:
        status = pre["identity"]
        summary["identity"] = {"file": str(dest / IDENTITY_FILE), "status": status,
                               "sha256": hashlib.sha256(canonical_json_bytes(record)).hexdigest()}
        print(f"[fst] run identity {status}: {summary['identity']['sha256']}", flush=True)
        mark(f"identity_{status}")

        run_local.mkdir(parents=True, exist_ok=True)
        attempts: list[dict] = []
        if resume:
            seed = pre["seed"]
            if seed is not None:
                summary["resume"] = seed
                attempts = seed["attempts"]
                print(f"[resume] attempt {seed['attempt']}: seeded {seed['covered_total']} covered "
                      f"goal(s) ({seed['refused_total']} refused, terminal), "
                      f"{seed['errored_to_retry_total']} errored goal(s) to retry, "
                      f"{len(seed['torn_memory_dbs'])} torn memory.db set aside", flush=True)
                for k, c in sorted(seed["seeded"].items()):
                    print(f"[resume]   {k}: done={c['done']} errored={c['errored']} "
                          f"refused={c['refused']} rows={c['rows']}", flush=True)
                mark(f"resumed_attempt_{seed['attempt']}")
            else:
                summary["resume"] = {"seeded": False, "reason": pre["note"]}
                print(f"[resume] {pre['note']}", flush=True)
        now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        attempts.append({"attempt": len(attempts) + 1, "started_at": now_iso,
                         "started_at_epoch": time.time()})
        ids_path = str(write_run_ids_file(run_local, ids_file_bytes))   # the ORIGINAL bytes
        (run_local / IDENTITY_FILE).write_bytes(canonical_json_bytes(record))
        srv = run_local / "srv"
        srv.mkdir(exist_ok=True)
        syncer = RunSyncer(run_local, str(dest))

        def data_sync() -> bool:
            """ONE fenced copy + commit of the run tree. Marks data copying as
            BEGUN before the copy starts (a failed copy can leave a truncated
            destination file, so from here on only a committed final sync may
            authorize the release); returns whether the sync committed."""
            sync_state["data_write_begun"] = True
            return _fenced_sync(lease, syncer)

        def final_sync() -> None:
            """The attempt's LAST durable write: its SUCCESS is the one
            event that authorizes the release — the LATEST final sync's, so a
            later one that fails withdraws it; otherwise the lease stays HELD,
            loudly."""
            sync_state["final_committed"] = False
            if data_sync():
                sync_state["final_committed"] = True
                return
            summary["status_before_final_sync"] = summary.get("status")
            summary["status"] = "FINAL_SYNC_FAILED"
            print(f"[fst] FATAL: FINAL_SYNC_FAILED — the final sync did not commit, so the run tree on the volume may be "
                  f"incomplete, so the lease stays HELD (owner {lease.owner}, Dict {LEASE_DICT_NAME!r}, "
                  f"run {run_id!r}) and no later launch may resume this run until it is inspected. Once "
                  f"this attempt is confirmed dead, release by hand: {_lease_remedy(run_id, dest)}",
                  flush=True)

        # 1. the attacker on GPU 0 (background)
        attacker_port = _free_port()
        att_log = srv / "attacker_vllm.log"
        att_proc = _start_attacker(attacker_port, att_log)
        mark("attacker_spawned")
        (run_local / "run_provenance.json").write_text(json.dumps(_provenance(
            run_id=run_id, sha=sha, geom=geom, goal_ids=goal_ids, cells=cells, attempts=attempts,
            now_iso=now_iso, resume=resume, summary=summary, record=record), indent=2))

        # 2. the target standup on GPU 1
        tgt_proc, stop = _start_target(tree, srv, geom, cells=cells,
                                       digests_by_tt=resolved["digests_by_tt"], gpu="1")
        ep = srv / "endpoint.json"
        t2 = time.time()
        while not ep.exists():
            lease.beat()
            if tgt_proc.poll() is not None:
                summary["status"] = "TARGET_STANDUP_FAILED"
                summary["standup_tail"] = (srv / "standup.log").read_text(errors="replace")[-4000:]
                vl = srv / "vllm_server.log"
                summary["vllm_tail"] = vl.read_text(errors="replace")[-4000:] if vl.exists() else None
                _kill(att_proc); final_sync(); return summary
            if time.time() - t2 > 3000:
                summary["status"] = "TARGET_STANDUP_TIMEOUT"; _kill(att_proc); final_sync()
                return summary
            time.sleep(5)
        endpoint = json.loads(ep.read_text())
        proxy_url, rt = endpoint["proxy_url"], endpoint["routing_table"]
        summary["target_standup_s"] = round(time.time() - t2, 1)
        summary["proxy_url"] = proxy_url
        mark("target_up")

        # 3. the ATTESTED values against the RECORD — no drivers otherwise
        from sisteer.capeval.serving.proxy import load_routing_table
        routes = list(load_routing_table(Path(rt))["routes"])
        route_hash = {r.cell_id: r.intervention_hash for r in routes}
        problems = check_attestation(record, endpoint, route_hash, st=st)
        (run_local / "attestation_check.json").write_text(json.dumps(
            {"problems": problems, "endpoint": endpoint,
             "routes": {r.cell_id: r.intervention_hash for r in routes}}, indent=2, default=str))
        if problems:
            summary["status"] = "IDENTITY_ATTESTATION_MISMATCH"
            summary["attestation_problems"] = problems
            print("[fst] FATAL: the standup does not attest the run identity:\n  - "
                  + "\n  - ".join(problems), flush=True)
            stop.touch(); _kill(att_proc); final_sync(); return summary
        mark("attestation_ok")

        # 4. attacker healthy + its verified serving manifest
        try:
            summary["attacker_ready_s"] = round(_wait_health(
                f"http://127.0.0.1:{attacker_port}/health", att_proc, 3000, "attacker vLLM", att_log,
                on_tick=lease.beat), 1)
        except HealthWaitFailed as e:        # the wait's OWN failures only; a lease failure passes (AC-1)
            summary["status"] = "ATTACKER_FAILED"; summary["attacker_err"] = str(e)[-4000:]
            stop.touch(); final_sync(); return summary
        try:
            serving_manifest = _wait_attacker_serving_manifest(att_log, att_proc, on_tick=lease.beat)
        except AttackerServingUnverified as e:
            summary["status"] = "ATTACKER_SERVING_UNVERIFIED"; summary["attacker_err"] = str(e)[-4000:]
            stop.touch(); _kill(att_proc); final_sync(); return summary
        summary["attacker_serving_manifest"] = str(serving_manifest)
        summary["attacker_serving"] = json.loads(serving_manifest.read_text()).get("requested")
        mark("attacker_up")
        if not data_sync():                          # intermediate: counted, the final sync decides
            summary["sync_failures"] += 1

        # 5. the drivers: every (cell x subshard), concurrently
        env = _env(tree)
        cmds: dict[str, list[str]] = {}
        for attack in ATTACKS:
            for cell in cells:
                for sub in range(subshards):
                    key = f"qwen/{attack}/{cell}/shard{sub:02d}"
                    log = run_local / "driver_logs" / f"{key.replace('/', '__')}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    cmd = _driver_cmd(tree, cell=cell, sub=sub, subshards=subshards, ids_path=ids_path,
                                      private_root=str(run_local / "qwen"), proxy_url=proxy_url, rt=rt,
                                      attacker_port=attacker_port, target_window=geom.max_model_len,
                                      attacker_serving=str(serving_manifest), attack=attack)
                    cmds[key] = cmd
                    procs[key] = subprocess.Popen(cmd, env=env, stdout=log.open("w"),
                                                  stderr=subprocess.STDOUT)
        (run_local / "driver_cmds.json").write_text(json.dumps(cmds, indent=1))
        mark(f"drivers_spawned_{len(procs)}")

        def progress() -> dict:
            stt = {}
            for key in procs:
                arm, attack, cell, shard = key.split("/")
                pj = run_local / arm / PRIVATE_RUN / attack / cell / shard / "progress.jsonl"
                c = progress_counts(pj)
                stt[key] = {"goals_done": c["done"], "errored": c["errored"], "refused": c["refused"],
                            "rows": c["rows"], "rc": procs[key].poll()}
            return stt

        last_sync = time.time(); last_print = 0.0; last_rows = -1
        while any(p.poll() is None for p in procs.values()):
            lease.beat()                     # AttemptLeaseLost -> LEASE_LOST below, nothing synced
            if tgt_proc.poll() is not None:
                print("[fst] FATAL: target standup died mid-run", flush=True)
                summary["status"] = "TARGET_DIED"; break
            if att_proc.poll() is not None:
                print("[fst] FATAL: attacker vLLM died mid-run", flush=True)
                summary["status"] = "ATTACKER_DIED"; break
            stt = progress()
            rows = sum(v["rows"] for v in stt.values())
            if time.time() - last_print > 180:
                done = sum(v["goals_done"] for v in stt.values())
                err = sum(v["errored"] for v in stt.values())
                ref = sum(v["refused"] for v in stt.values())
                fin = sum(1 for v in stt.values() if v["rc"] is not None)
                bad = [k for k, v in stt.items() if v["rc"] not in (None, 0)]
                print(f"[t+{time.time() - T0:.0f}s] goals_done={done} errored={err} refused={ref} "
                      f"rows={rows} drivers_finished={fin}/{len(procs)} nonzero_rc={bad}", flush=True)
                last_print = time.time()
            if rows != last_rows or time.time() - last_sync > sync_every_s:
                if not data_sync():                  # intermediate: counted, the final sync decides
                    summary["sync_failures"] += 1
                last_sync = time.time(); last_rows = rows
            time.sleep(15)
        for p in procs.values():
            if p.poll() is None:
                p.kill()
        summary["drivers"] = progress()
        summary["drivers_wall_s"] = round(
            time.time() - T0 - summary["timeline"].get(f"drivers_spawned_{len(procs)}", 0), 1)
        summary.setdefault("status", "DONE" if all(v["rc"] == 0 for v in summary["drivers"].values())
                           else "DRIVER_ERRORS")
        summary["refused_total"] = sum(v["refused"] for v in summary["drivers"].values())
        summary["errored_total"] = sum(v["errored"] for v in summary["drivers"].values())
        mark("drivers_done")

        # 6. teardown on every path + final sync
        stop.touch()
        try:
            tgt_proc.wait(timeout=300)
        except subprocess.TimeoutExpired:
            tgt_proc.kill()
        _kill(att_proc)
        summary["total_wall_s"] = round(time.time() - T0, 1)
        (run_local / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        final_sync()
        print("[fst] SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "drivers"},
                                         default=str), flush=True)
        return summary
    except (AttemptLeaseLost, AttemptLeaseIndeterminate) as e:
        summary["status"] = "LEASE_LOST" if isinstance(e, AttemptLeaseLost) else "LEASE_INDETERMINATE"
        summary["lease_err"] = str(e)[-4000:]
        print(f"[fst] FATAL: {summary['lease_err']}", flush=True)
        stop_writers()
        return summary          # NO sync: the run belongs (or may belong) to another attempt
    except BaseException as e:
        summary.setdefault("status", f"EXCEPTION:{type(e).__name__}")
        summary["exception"] = f"{type(e).__name__}: {e}"[-4000:]
        stop_writers()
        if sync_state["data_write_begun"]:
            # data copying has begun: with every writer stopped, ONE owned final sync decides the
            # release (it commits -> released in the finally; it fails -> FINAL_SYNC_FAILED, held).
            # The exception propagates either way.
            try:
                (run_local / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
            except Exception as e2:  # noqa: BLE001
                print(f"[fst] WARNING: summary.json not written ({type(e2).__name__}: {e2})", flush=True)
            try:
                final_sync()
            except AttemptLeaseError as e2:      # the fence refused: the run is (or may be) another's
                summary["lease_err"] = str(e2)[-4000:]
                print(f"[fst] FATAL: {summary['lease_err']}", flush=True)
        raise
    finally:
        # the lease is released on POSITIVE authorization only — a final sync that
        # copied and committed — or on a failure before this attempt's first data copy (the named
        # pre-data set: anything up to the first data_sync(), when nothing of this attempt's is
        # under dest but the identity file and the lease record — the identity-only state a
        # refused launch leaves, which the plain retry admits). Every other exit keeps it held.
        outcome = summary.get("status")
        if sync_state["final_committed"]:
            summary["lease"]["released"] = lease.release(outcome)
            summary["lease"]["held_after_exit"] = False
            summary["lease"]["release_basis"] = "final_sync_committed"
        elif not sync_state["data_write_begun"]:
            summary["lease"]["released"] = lease.release(outcome)
            summary["lease"]["held_after_exit"] = False
            summary["lease"]["release_basis"] = "no_data_written"
            print(f"[fst] lease released after {outcome}: this attempt copied no data under {dest} "
                  "(the failure came before its first sync)", flush=True)
        else:
            summary["lease"]["released"] = False
            summary["lease"]["held_after_exit"] = lease.held
            summary["lease"]["release_basis"] = None
            if lease.held and outcome != "FINAL_SYNC_FAILED":    # final_sync() already said it
                print(f"[fst] FATAL: exiting ({outcome}) after data copying began with no committed "
                      f"final sync — the lease stays HELD (owner {lease.owner}, Dict {LEASE_DICT_NAME!r}, "
                      f"run {run_id!r}) and no later launch may resume this run until it is inspected. "
                      f"Once this attempt is confirmed dead, release by hand: {_lease_remedy(run_id, dest)}",
                      flush=True)
        summary["lease"]["indeterminate"] = lease.indeterminate
        summary["lease"]["file_errors"] = lease.file_errors


@app.function(image=image, volumes={VOL_PATH: VOL}, gpu="H200:2", cpu=16,
              memory=131072, timeout=CONTAINER_TIMEOUT_S)
def run_exp2(sha: str, goal_ids: list[str], run_id: str, cells: list[str],
             subshards: int = DEFAULT_SUBSHARDS, sync_every_s: int = 120, resume: bool = False,
             target_window: int = 0, target_seqs: int = 0, allow_over_budget: bool = False,
             *, ids_file_bytes: bytes | None = None, app_name: str = "") -> dict:
    """The 2xH200 topology: GPU0 attacker, GPU1 the one standup serving every
    cell. ``ids_file_bytes`` are the ORIGINAL bytes of the launcher's ids file
    (required: the identity hashes them and the run's copy IS them);
    ``app_name`` is the launcher's ``-n`` name, recorded in the attempt lease."""
    try:
        return _run_exp2_impl(sha, goal_ids, run_id, cells, subshards=subshards,
                              sync_every_s=sync_every_s, resume=resume,
                              target_window=target_window, target_seqs=target_seqs,
                              allow_over_budget=allow_over_budget, ids_file_bytes=ids_file_bytes,
                              app_name=app_name)
    finally:
        # idle-GPU rule: nothing of ours may outlive the function
        subprocess.run(["pkill", "-f", "capctrl_serving_up.py"], check=False)
        subprocess.run(["pkill", "-f", "vllm serve"], check=False)


# --------------------------------------------------------------------------- #
# preflight: every non-GPU precondition, in the container, on CPU              #
# --------------------------------------------------------------------------- #

#: Placeholders for the preflight's driver-argv check: ``jailbreak_run.main()``
#: touches none of these paths before dispatching to ``cmd_generate`` (only
#: ``--config`` is read), so they need not exist. A test rebuilds the SAME
#: command from them to pin what the check hands ``main()``.
PREFLIGHT_DRIVER_PLACEHOLDERS = dict(
    sub=0, subshards=5, ids_path="/tmp/ids", private_root="/tmp/x",
    proxy_url="http://127.0.0.1:1/v1", rt="/tmp/rt.json", attacker_port=2,
    target_window=9984, attacker_serving=None)


def driver_argv_check(tree: str, cell: str) -> dict:
    """The preflight's driver-argv check, one cell: the REAL ``_driver_cmd``
    for ``cell`` handed to the tree's ``jailbreak_run.main()`` exactly as a
    driver process would see it, then the resolution the real ``cmd_generate``
    performs first — so the cell is ACCEPTED and RESOLVED to ``split_variant``,
    not merely parsed.

    The seam: a driver is launched as ``[interpreter, script, *args]`` and
    sees ``sys.argv == [script, *args] == cmd[1:]``; ``main()`` reads
    ``sys.argv`` through ``ap.parse_args()``, so that is what is installed
    (the standup check strips the same two for capctrl's ``parse_args(argv)``).
    Until 2026-09-07 this check kept the script path from ``cmd[1:]`` AND
    prepended a fake argv[0], so the path sat in argv[1] — argparse's
    subcommand slot — and died with ``invalid choice`` (SystemExit 2). The
    check could never pass; the preflight had never run end to end.

    ``cmd_generate`` is replaced by a capture for the call, so nothing loads a
    model or contacts a pool; what it would have done first —
    ``_check_defense_id(args, _valid_defense_ids(cfg), served=…)`` with its
    own derivation of ``served`` — is then run on the captured args, and a
    variant beside ``--positions``, an unknown base, or a variant the harness
    grammar refuses raises here exactly as it would in the driver."""
    import jailbreak_run as jr
    cmd = _driver_cmd(tree, cell=cell, **PREFLIGHT_DRIVER_PLACEHOLDERS)
    from sisteer.capeval.serving.serve_declaration import split_variant
    base, variant = split_variant(cell)
    cap: dict = {}
    orig_generate, orig_argv = jr.cmd_generate, sys.argv
    jr.cmd_generate = lambda a, c: cap.update(args=a, cfg=c)
    try:
        sys.argv = cmd[1:]                     # the driver process's own argv
        jr.main()
    finally:
        jr.cmd_generate, sys.argv = orig_generate, orig_argv
    assert cap, "jailbreak_run.main() returned without dispatching to cmd_generate"
    a, cfg = cap["args"], cap["cfg"]
    assert a.cmd == "generate" and a.defense == cell and a.target_engine == "served", vars(a)
    # No private copy of the positions rule here: the driver's own validator
    # below is the ONE verdict on a variant beside --positions (the builder's
    # literal is pinned by test_driver_argv_pins).
    served = getattr(a, "target_engine", "hf") == "served"        # cmd_generate's derivation
    resolved = jr._check_defense_id(a, jr._valid_defense_ids(cfg), served=served)
    assert resolved == (base, variant), (cell, resolved, (base, variant))
    return {"argv": cmd[1:], "defense": a.defense, "positions": a.positions,
            "resolved": {"base": base, "variant": variant}}


@app.function(image=image, volumes={VOL_PATH: VOL}, timeout=1800, cpu=4, memory=16384)
def preflight(sha: str, goal_ids: list[str], cells: list[str], ids_file_bytes: bytes | None = None,
              subshards: int = DEFAULT_SUBSHARDS) -> dict:
    tree = _tree(sha)
    sys.path[:0] = [f"{tree}/scripts", f"{tree}/src"]
    os.environ["TT_TWO_SI_DEFENSES_DIR"] = OBJECTS_X1
    rep = {"sha": sha[:8], "ok": True, "cells": cells, "checks": {}}

    def chk(name, fn):
        try:
            rep["checks"][name] = {"ok": True, **(fn() or {})}
        except BaseException as e:  # noqa: BLE001
            import traceback
            rep["checks"][name] = {"ok": False, "err": f"{type(e).__name__}: {e}",
                                   "tb": traceback.format_exc()[-1500:]}
            rep["ok"] = False

    def imports():
        import pyrit, requests, torch, transformers, vllm  # noqa: F401
        import sisteer
        assert sisteer.__file__.startswith(tree), sisteer.__file__
        from sisteer.eval.attacks import served_target  # noqa: F401
        return {"pyrit": pyrit.__version__, "vllm": vllm.__version__,
                "transformers": transformers.__version__, "torch": str(torch.__version__)}
    chk("imports", imports)

    def vocabulary():
        """The staged sha carries the serve-declaration vocabulary AND the
        client's pinned attestation keys — a pre-landing tree refuses."""
        from sisteer.capeval.serving import serve_declaration as sd
        from sisteer.eval.attacks import served_target as st_
        assert sd.EXPERIMENT_FIT_SERVE_PAIRS and VARIANT_ALL_TURNS in sd.SERVE_VARIANTS
        assert VARIANT_ALL_POSITIONS in sd.SERVE_VARIANTS
        for c in cells:
            sd.split_variant(c)
        return {"variants": sorted(sd.SERVE_VARIANTS), "experiments": list(sd.SERVE_EXPERIMENTS),
                "attest_keys": [st_.ATTEST_SERVE_DECLARATIONS_KEY, st_.ATTEST_LENS_SALT_POLICY_KEY,
                                st_.ATTEST_OBJECT_SHA256_BY_CELL_KEY],
                "refusal_codes": list(st_.ALL_TURNS_REFUSAL_CODES)}
    chk("serve_declaration_vocabulary_on_staged_sha", vocabulary)

    def capctrl_argv():
        """capctrl_serving_up accepts repeated --cell, per-cell digests and
        --lens-salt-policy (an older tree refuses here, before any GPU)."""
        import capctrl_serving_up as cs
        ap = cs.build_parser()
        fake = {c: "0" * 64 for c in cells if c != "none"}
        argv = _target_standup_cmd(tree, Path("/tmp/srv"), Path("/tmp/srv/STOP"),
                                   resolve_target_geometry(tree), cells=cells, digests_by_tt=fake)[2:]
        ns = ap.parse_args(argv)
        assert list(ns.cell) == list(cells), ns.cell
        assert getattr(ns, "lens_salt_policy") == LENS_SALT_POLICY
        return {"argv_tail": argv[-8:], "n_cell": len(ns.cell)}
    chk("capctrl_accepts_repeated_cell_and_salt_policy", capctrl_argv)

    def driver_argv():
        """Every cell of the run through ``driver_argv_check`` (module-level so
        a test drives it against a local checkout): the real argv, the tree's
        real ``jailbreak_run``, and its real defense resolution."""
        return {"per_cell": {c: driver_argv_check(tree, c) for c in cells}}
    chk("driver_argv_accepts_variant_cell", driver_argv)

    def goals():
        import jailbreak_run as jr
        import fst_exp2_goals as fg
        cfg = jr._load_cfg(f"{tree}/{CFG_REL}")
        gs = jr._eval_goals(cfg, confirmatory=True)
        full_ids = [g.goal_id for g in gs]
        s15 = fg.s15_subset(full_ids)
        assert goal_ids, "no goal ids"
        missing = [g for g in goal_ids if g not in set(s15)]
        assert not missing, f"ids outside the §15 87: {missing[:5]}"
        assert len(set(goal_ids)) == len(goal_ids), "duplicate goal ids"
        # the drivers' slice of the run's copy == the identity's shard assignment
        p = Path("/tmp/fst_goal_ids.txt")
        if ids_file_bytes:
            p.write_bytes(ids_file_bytes)           # the ORIGINAL bytes, as the run copies them
        else:
            p.write_text("\n".join(goal_ids) + "\n")
        sel = jr._select_goal_ids(gs, str(p))
        assert [g.goal_id for g in sel] == list(goal_ids), "the drivers would replay another list"
        shards = shard_assignment(goal_ids, subshards)
        assert [[g.goal_id for g in sel[k::subshards]] for k in range(subshards)] == shards
        tracked = Path(tree) / TRACKED_IDS_REL
        tracked_ids = _read_ids(str(tracked)) if tracked.is_file() else None
        tracked_sha = hashlib.sha256(tracked.read_bytes()).hexdigest() if tracked.is_file() else None
        ids_sha = hashlib.sha256(ids_file_bytes).hexdigest() if ids_file_bytes else None
        return {"n": len(goal_ids), "n_s15": len(s15), "n_confirmatory": len(full_ids),
                "subshards": subshards, "subshard_sizes": [len(s) for s in shards],
                "forward_cap": jr._forward_cap(cfg),
                "target_max_new_tokens": jr._target_max_new_tokens(cfg),
                "tracked_ids_in_staged_tree": str(tracked) if tracked_ids is not None else None,
                "equals_tracked": (tracked_ids == goal_ids) if tracked_ids is not None else None,
                "ids_file_sha256": ids_sha, "tracked_file_sha256": tracked_sha,
                "equals_tracked_bytes": ((ids_sha == tracked_sha)
                                         if ids_sha and tracked_sha else None)}
    chk("goals_subset_of_s15_subset_of_confirmatory", goals)

    def cells_and_objects():
        r = resolve_cells(tree, cells)
        return {"per_cell": r["per_cell"], "control_legs": r["control_legs"],
                "object_artifact_sha256_by_cell": r["digests_by_cell"], "vmodels": r["vmodels"],
                "target": r["target"]}
    chk("cells_enumerate_distinct_hashes_and_digests", cells_and_objects)

    def attacker_files():
        d = Path(_attacker_snapshot_dir())
        stf = list(d.glob("*.safetensors"))
        assert d.is_dir() and stf and (d / "config.json").exists(), (
            f"attacker snapshot missing at {d}: run --stage-attacker (or grokmt_served.py --stage)")
        return {"snapshot_dir": str(d), "n_safetensors": len(stf),
                "gb": round(sum(f.stat().st_size for f in stf) / 1e9, 1)}
    chk("qwen_attacker_staged", attacker_files)

    def writable():
        d = Path(f"{GEN}/_preflight"); d.mkdir(parents=True, exist_ok=True)
        (d / "probe").write_text("ok"); (d / "probe").unlink()
        assert Path(f"{tree}/scripts/capctrl_serving_up.py").exists()
        return {"gen_root": GEN}
    chk("out_root_writable_and_scripts_present", writable)

    VOL.commit()
    print("[preflight]", json.dumps(rep, indent=1, default=str))
    return rep


# --------------------------------------------------------------------------- #
# local entrypoint                                                             #
# --------------------------------------------------------------------------- #

def _read_ids(ids: str) -> list[str]:
    return parse_goal_ids(Path(ids).read_text())


def default_ids_path() -> Path:
    """The tracked ids file in THIS checkout (the launch machine's copy of
    ``TRACKED_IDS_REL``); the staged tree carries the same bytes at the gated
    sha, which the preflight reports against."""
    return Path(__file__).resolve().parents[2] / TRACKED_IDS_REL


@app.local_entrypoint()
def main(sha: str = "", ids: str = "", run: str = "", stage: bool = False,
         stage_attacker: bool = False, pf: bool = False, go: bool = False, harvest: str = "",
         cells: str = "", constitution: bool = False, subshards: int = DEFAULT_SUBSHARDS,
         resume: bool = False, sync_every_s: int = 120, target_window: int = 0,
         target_seqs: int = 0, allow_over_budget: bool = False, restage: bool = False):
    if harvest:
        stt = json.loads(Path(harvest).expanduser().read_text())
        print(f"[fst] harvesting run {stt['run']} from call {stt['call_id']} ...", flush=True)
        rec = modal.FunctionCall.from_id(stt["call_id"]).get()
        print(json.dumps({k: v for k, v in rec.items() if k != "drivers"}, indent=1, default=str))
        if rec.get("status") != "DONE":
            raise SystemExit(f"[fst] run {stt['run']}: status {rec.get('status')}")
        return
    if not sha or len(sha) != 40:
        raise SystemExit("[fst] pass the FULL 40-hex --sha of the gated integration tip")
    run_name = app.description or DEFAULT_APP_NAME
    check_app_name(run_name, sha=sha, run=run if go else "")
    cell_list = parse_cells(cells, constitution)
    print(f"[fst] app={run_name} sha={sha} cells={cell_list}", flush=True)
    if stage:
        with tempfile.TemporaryDirectory(prefix="fst-tar-") as td:
            tar = shallow_clone_tarball(sha, Path(td))
            print(f"[fst] pushing {tar.name} ({tar.stat().st_size // 1_000_000} MB) -> {REPOS}/",
                  flush=True)
            with VOL.batch_upload(force=True) as b:
                b.put_file(str(tar), f"{volume_client_path(REPOS)}/{tar.name}")
            r = unpack_tree.remote(sha, force=restage)
            print(json.dumps(r, indent=1))
            if r.get("error"):
                raise SystemExit(f"[fst] staging failed: {r['error']}")
    if stage_attacker:
        print(json.dumps(stage_attacker_fn_remote(), indent=1))
    if not (pf or go):
        if stage or stage_attacker:
            return
        raise SystemExit("[fst] pass --stage / --stage-attacker / --pf / --go / --harvest <state>")
    ids = ids or str(default_ids_path())
    if not Path(ids).is_file():
        raise SystemExit(f"[fst] --ids {ids}: no such file (the tracked default is "
                         f"{TRACKED_IDS_REL}; scripts/fst_exp2_goals.py writes it)")
    ids_bytes = Path(ids).read_bytes()      # the ORIGINAL bytes: hashed AND copied as-is (E1/B2)
    goal_ids = parse_goal_ids(ids_bytes.decode("utf-8"))
    if not goal_ids:
        raise SystemExit(f"[fst] --ids {ids}: no goal ids")
    if len(set(goal_ids)) != len(goal_ids):
        raise SystemExit(f"[fst] --ids {ids}: duplicate goal ids")
    if subshards < 1:
        raise SystemExit(f"[fst] --subshards must be >= 1, got {subshards}")
    print(f"[fst] goal ids: {len(goal_ids)} from {ids} "
          f"(sha256 {hashlib.sha256(ids_bytes).hexdigest()[:16]}); subshards={subshards}", flush=True)
    if pf:
        r = preflight.remote(sha, goal_ids, cell_list, ids_bytes, subshards)
        print(json.dumps(r, indent=1, default=str))
        if not r.get("ok"):
            raise SystemExit("[fst] PREFLIGHT FAILED — fix all above before any GPU")
        if not go:
            return
    if not run:
        raise SystemExit("pass --run <run_id>")
    geometry = _geometry_module(None)
    try:
        geom = resolve_target_geometry(None, target_window=target_window, target_seqs=target_seqs,
                                       allow_over_budget=allow_over_budget)
    except (geometry.GeometryOverBudget, geometry.UnknownModelKey) as e:
        raise SystemExit(f"[fst] refusing the target geometry: {e}")
    print(geom.describe(), flush=True)
    print(f"[fst] launching run {run} @ {sha[:8]} cells={cell_list} subshards={subshards} "
          f"resume={resume} app={run_name} target={geom.max_model_len}x{geom.max_num_seqs} "
          f"salt_policy={LENS_SALT_POLICY} reduction={REDUCTION_MODE}", flush=True)
    call = run_exp2.spawn(sha, goal_ids, run, cell_list, subshards, sync_every_s, resume,
                          target_window, target_seqs, allow_over_budget,
                          ids_file_bytes=ids_bytes, app_name=run_name)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    state_path = STATE_DIR / f"{run}.json"
    state_path.write_text(json.dumps({
        "app": run_name, "sha": sha, "run": run, "call_id": call.object_id,
        "cells": cell_list, "subshards": subshards, "resume": resume, "ids": str(ids),
        "ids_file_sha256": hashlib.sha256(ids_bytes).hexdigest(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    print(f"[fst] spawned call {call.object_id}; state -> {state_path} "
          f"(re-attach with --harvest {state_path})", flush=True)
    s = call.get()
    print(json.dumps({k: v for k, v in s.items() if k != "drivers"}, indent=1, default=str))
    if s.get("status") != "DONE":
        raise SystemExit(f"[fst] run {run}: status {s.get('status')}")


def stage_attacker_fn_remote() -> dict:
    return stage_attacker.remote()
