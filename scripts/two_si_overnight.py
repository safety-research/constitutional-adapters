"""two_si_overnight.py — the bounded overnight orchestrator for the two-SI
chain. Runs UNATTENDED on the grading host and exits cleanly; it never
polls indefinitely and it never touches anything that is not its own.

The chain, in order (each stage resumable, the whole script safe to re-run):

  1. wait      — poll sacct (over ssh) until the named generate-pos cluster
                 job (`tt-twosi-posfull`, or --generate-job-id) is terminal.
  2. pull      — rsync the generated UNGRADED constitution rows back from the
                 cluster's mode-0700 private dir to the grading host private root
                 (modes forced 0700/0600).
  3. grade-pos — on the grading host, where the Low-Prio grader key lives.
                 Wide by default (200, the repo's live-grader norm); the
                 stage PRINTS the ready-to-post coordination line (when the
                 site names a channel, SISTEER_COORDINATION_CHANNEL) into the
                 log and PROCEEDS (print-and-proceed — the post is a courtesy,
                 never a gate; 429 protection is the grader's same-org
                 backoff).
  4. manifest  — fold the graded rows in and freeze the counts-only manifest.
                 If the regime comes out `native_per_si` the run STOPS unless
                 --allow-native: an unattended run must not silently fit the
                 confounded comparison (it must be labelled as such, and the
                 overnight intent is the matched pool).
  5. emit-pairs— the frozen manifest -> per-SI {prompt,pos,neg} jsonls
                 (build_two_si_pool.py emit-pairs, piece D).
  6. push+fit  — rsync the pairs to the cluster private dir, then submit ONE
                 fitting job per SI (train_two_si_objects.py train: one model
                 load, lockstep-shared teacher) and monitor them to completion.
  7. fit-b     — submit the ONE SI-agnostic Class B job (fit_class_b.py:
                 the config's class_b_arms — antidote, alphasteer, and the
                 caa_static_r matched CAA floor since 2026-08-21
                 (circuit_breaker left the registry entirely 2026-08-20) —
                 each fit once on the AlphaSteer authors' 14k/2k content
                 lists with no SI) and monitor it. INDEPENDENT of steps 1-6
                 on purpose: the two classes share no inputs, so a Class A
                 failure — any stage of it, the regime gate included — never
                 skips the Class B fit, and a Class B failure never touches
                 Class A. Both halves are always attempted; either failing
                 still makes the whole run exit 1.
  8. report    — a structured JSONL progress log is appended the whole way
                 (every submission, verdict, retry and exclusion), so a reader
                 can reconstruct exactly what happened without re-deriving it.

Safety/robustness properties (why this is safe to leave alone):

- the nodes the site excludes on the run's cluster (``--exclude-nodes``, else the
  ``--exclude`` list in that cluster block's CLUSTER_SBATCH_EXTRA: GPU-leak nodes,
  where Slurm and the driver disagree about GPU state and Slurm fails the JOB not
  the NODE) are excluded on EVERY sbatch, and the exclusion list is logged on every
  submission. Any node that produces a busy-GPU failure is ADDED to the list
  for the remainder of the run. Nodes are only ever recorded and excluded —
  this script never kills a process and never drains a node.
- The busy-GPU failure (FAILED in ~40s–2min with ``CUDA error: ...busy
  or unavailable`` / ``AcceleratorError``) is RETRYABLE: detected from the
  job's own stderr PLUS a short-elapsed gate, resubmitted with the failing
  node excluded, bounded by --max-attempts. A real code failure (no
  signature, or a long-elapsed death) is TERMINAL and is never retried —
  retrying a bug five times just burns five allocations. NODE_FAIL (the
  variant where slurmd itself dies) is likewise retryable.
- Hard-bounded: a wall-clock deadline (--deadline-hours, default 10) after
  which it stops submitting/polling and exits with a clear summary of what
  finished, what is still running (left running — never cancelled), and what
  never started.
- Re-entrant: generation/grading/fitting are each already resumable, and a
  re-run adopts a COMPLETED/RUNNING fit job by name instead of resubmitting
  (names are `tt-twosi-fit-<si>-<run-tag>` and `tt-twosi-fitb-<run-tag>`, so
  a re-run with the same --run-tag resumes and a new tag is a new session).
  On top of that, fit_class_b.py itself skips any arm whose meta.json
  completion sentinel exists, so even a fresh Class B submission never redoes
  a finished arm. A pid lockfile refuses two live instances of the same run
  tag.
- Both fit classes write their compiled objects to ONE absolute,
  commit-independent dir derived from --remote-root: never inside the
  per-commit checkout, where the next code change (new sha, new checkout,
  empty objects dir) would silently void resume and refit every object.
- sacct verdicts are recycled-id-proof: this cluster REUSES job ids, and a
  bare `sacct -j <id>` once returned an unrelated months-old job (the n1 run
  filed a false "TIMEOUT" verdict, citing a nonexistent array-style log, off
  exactly such a row). Every sacct query is therefore scoped to this user and
  the 7-day submit window (-u/-S) and takes only the -X main-job row.

Exit codes: 0 = chain complete; 1 = terminal stage failure (details in the
log); 3 = deadline expired with work pending; 4 = another live instance
holds the lock.

Everything that touches the outside world (ssh, sbatch, sacct, rsync, the
local pipeline subcommands) goes through ONE injectable runner, so the whole
control flow — including the retry-on-busy path and the deadline — is
unit-testable without a cluster (tests/test_overnight.py).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

import build_two_si_pool as bp  # pairs_filename / PAIRS_META_NAME (one spelling)

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# GPU-leak node constants                                                     #
# --------------------------------------------------------------------------- #

# GPU-leak nodes: Slurm books their GPUs idle while the driver holds them busy,
# marks each arriving JOB failed rather than the NODE, so they stay schedulable
# and kill every next arrival. They are excluded on EVERY submission until an
# admin drains and clears them. The site names them per cluster (the --exclude
# list in the cluster block's CLUSTER_SBATCH_EXTRA, the flag the front door adds
# to every job there), so a list never reaches a cluster whose scheduler does
# not know those nodes; nodes that fail busy DURING a run are added to the
# per-run copy.


def _site_blocks() -> dict:
    """Every cluster block of the site config this host uses (reproduce/lib/clusters.py load_addressable: each block under the
    site's clusters/ directory, listed in SISTEER_CLUSTERS or not); empty when this host has no site config file."""
    lib = str(REPO / "reproduce" / "lib")
    if lib not in sys.path:
        sys.path.append(lib)
    import clusters as CL  # noqa: E402 (the site's cluster blocks)
    cfg = CL.default_config_path()
    try:
        return CL.load_addressable(os.path.dirname(cfg), CL.site_config_file(cfg)) if os.path.isfile(cfg) else {}
    except SystemExit as exc:      # the reader's refusals are SystemExit
        raise StageFailed(f"the site config {cfg} could not be read: {exc}") from None


def site_block_for(shared_root: str) -> dict:
    """The site's cluster block whose CLUSTER_SHARED_ROOT is ``shared_root``; empty when no block names that root."""
    for block in _site_blocks().values():
        if block.get("CLUSTER_SHARED_ROOT", "").rstrip("/") == shared_root.rstrip("/"):
            return block
    return {}


def site_exclude_nodes(shared_root: str) -> list[str]:
    """The nodes the site excludes on the cluster whose shared root this is: the --exclude list (``--exclude=a,b``,
    ``--exclude a,b`` or ``-x a,b``) in that block's CLUSTER_SBATCH_EXTRA; empty when it names none."""
    toks = shlex.split(site_block_for(shared_root).get("CLUSTER_SBATCH_EXTRA", ""))
    for i, tok in enumerate(toks):
        if tok.startswith("--exclude="):
            return [n for n in tok.split("=", 1)[1].split(",") if n]
        if tok in ("--exclude", "-x") and i + 1 < len(toks):
            return [n for n in toks[i + 1].split(",") if n]
    return []


def default_remote_root() -> str:
    """The default cluster private root: <the site's shared root>/exp/jailbreak, the shared root resolved by sisteer.site
    ($SHARED: the environment's SISTEER_SHARED_ROOT, else the primary cluster block's CLUSTER_SHARED_ROOT). Refused by name
    when the site names none."""
    from sisteer import site
    try:
        return f"{site.expand('$SHARED').rstrip('/')}/exp/jailbreak"
    except (site.PlaceholderError, site.SiteConfigError) as exc:
        raise StageFailed(f"--remote-root not given and the site names no shared root: {exc}") from None
# The busy-GPU signature in a failed job's stderr/stdout: dies in
# ~40s–2min at torch.cuda.set_device with one of these. Gated by the
# short-elapsed check in classify_failure so a long-running job that happens
# to print "AcceleratorError" late is never misclassified as the leak.
BUSY_GPU_SIGNATURE = re.compile(
    r"busy or unavailable|AcceleratorError", re.IGNORECASE)

# sacct states after which a job will never run again.
TERMINAL_STATES = frozenset({
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
    "OUT_OF_MEMORY", "BOOT_FAIL", "DEADLINE", "PREEMPTED", "REVOKED"})

SACCT_FORMAT = "JobID,JobName,State,Elapsed,ExitCode,NodeList"

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


class StageFailed(Exception):
    """A terminal failure: the chain cannot continue (exit 1)."""


class DeadlineExpired(Exception):
    """The wall-clock bound was hit: stop submitting/polling (exit 3)."""


def default_runner(argv: list[str], timeout: float | None = None):
    """The one seam to the outside world: run ``argv``, capture text output.
    Tests replace this with a fake; nothing else in the module shells out."""
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)


def derive_qos(shared_root: str) -> str:
    """The cluster's non-preemptible QoS: CLUSTER_QOS of the site block whose
    CLUSTER_SHARED_ROOT is the run's shared root (the QoS the front door
    submits that cluster's jobs at). ONE derivation feeds both the rendered
    ``#SBATCH --qos`` header and every submit-time ``--qos`` flag, so the
    two can never disagree. A root no block describes fails loudly —
    submitting at a QoS the cluster does not define would reject every job."""
    qos = site_block_for(shared_root).get("CLUSTER_QOS", "")
    if qos:
        return qos
    raise StageFailed(
        f"cannot derive a QoS from shared root {shared_root!r}: no site cluster "
        "block names that root with a CLUSTER_QOS -- describe the cluster in "
        "the site config, or pass --qos explicitly")


def elapsed_seconds(elapsed: str) -> int:
    """sacct Elapsed -> seconds. Accepts D-HH:MM:SS, HH:MM:SS and MM:SS."""
    if not elapsed:
        return 0
    days = 0
    if "-" in elapsed:
        d, elapsed = elapsed.split("-", 1)
        days = int(d)
    parts = [int(p) for p in elapsed.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, s = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + s


def _fit_script_text(*, name: str, purpose: str, command: str,
                     time_limit: str, logdir: str, store_dir: str,
                     objects_dir: str, sha: str, shared_root: str,
                     qos: str = "high", gpus: int = 1,
                     project_root: str | None = None,
                     wandb_env: dict[str, str] | None = None) -> str:
    """The ONE sbatch template every fitting job this orchestrator submits is
    rendered from — Class A (per-SI, ``fit_sbatch_script``) and Class B
    (SI-agnostic, ``fit_class_b_sbatch_script``) differ only in ``purpose``
    (the header comment) and ``command`` (the srun payload). One mechanism ON
    PURPOSE, so a new job class cannot forget any of the hard-won parts:

    - qos parameterised per cluster (``derive_qos`` — a QoS name hardcoded
      for one cluster is rejected outright by the other cluster's
      scheduler), 1 GPU, 16 cpus, 128G (NEVER --mem=0), a right-sized
      --time, --open-mode=append, --no-requeue, --signal=B:TERM@120.
    - logs into the mode-0700 private _logs dir (a crash can print attack
      text), pre-created by the orchestrator (Slurm does not create log
      parents).
    - the pin guard: run from the IMMUTABLE commit-addressed checkout,
      never the shared mutable clone — and source hf_env.sh DIRECTLY rather
      than stage2_common.sh, whose s2_init hard-fails at job start whenever
      anything else moves the shared clone's HEAD off its EXPECTED_SHA (that
      aborted real jobs in seconds for reasons unrelated to the job).
    - W&B fully disabled and TT_TWO_SI_STORE_DIR pinned to shared storage
      (no store resolution may fall back to node-local /home).
    - the compiled-objects dir pinned to the ONE absolute, commit-independent
      location: passed to the driver via --out AND exported as
      TT_TWO_SI_DEFENSES_DIR, so writers and readers can never split across
      per-commit checkouts — the checkout's own out/ is emptied by every new
      sha, which would silently void resume.
    - env NEVER passes via sbatch --export (it crashes slurmd on this
      cluster) — everything is in the script body.
    - the committed lock in the job's flavor: ``uv sync --frozen --extra
      cu128`` with its own rc handling and no unfrozen fallback — the same
      rules the sbatch flavor audit (tests/test_torch_flavors.py) holds the
      checked-in job scripts to, applied to the rendered text by
      tests/test_overnight.py; torch lives only in the flavor extras, so a
      flavor-less sync would hand the job a venv without it.
    - the unattended-window RESULT sentinel: an EXIT trap (separate from the
      SIGTERM/SIGINT/SIGQUIT process-group cleanup, which it must not
      disturb), installed before the pin guard and the sync so EVERY exit of
      the script records one, writes a one-line outcome (state, exit code,
      elapsed seconds, job id, node) to $FIGS_DIR/RESULT and mirrors the
      figures dir to the grading host — every step best-effort, so a failed
      sentinel write can never change a finished job's exit code.

    ``gpus`` scales the allocation for the parallel Class A driver
    (data_parallel.gpus in the config — the driver schedules per-arm
    process groups itself, so the srun payload stays a plain single-task
    python invocation, never torchrun): cpus at 16/GPU, mem at 128G/GPU
    (never --mem=0; 8 GPUs = 1024G, inside the ~128-250G/GPU norm), plus
    the cluster's NCCL env for the multi-GPU case.

    ``wandb_env`` is the site's W&B destination (``resolve_wandb_env``),
    rendered into the body because a bare sbatch carries no job env.

    Rendered as TEXT so tests can assert every required line without a
    cluster.
    """
    nccl_env = "" if gpus <= 1 else """
# Multi-GPU (the parallel fit driver's per-arm process groups): the
# cluster's NCCL conventions. Single-node collectives ride NVLink; the
# socket interface is left to the site's nccl.conf (an exported
# NCCL_SOCKET_IFNAME would override it).
export NCCL_NVLS_ENABLE=0
"""
    # these jobs go out with a bare sbatch, outside the front door's job env, so the project directory their pinned
    # checkout lives under is rendered into the script itself
    project_export = ("" if not project_root else
                      f"export SISTEER_PROJECT_ROOT={shlex.quote(project_root)}   # the project directory (checkouts/<sha>)\n")
    # and so is the site's W&B destination, the keys the front door's job env carries: without them the config's
    # $WANDB_PROJECT has no value on the node and the fit runs with W&B off
    wandb_export = "" if not wandb_env else (
        "# the site's W&B destination (the keys the front door's job env carries): the config's $WANDB_PROJECT\n"
        "# expands to its project, and the run is filed under its entity unless the job already has a WANDB_ENTITY\n"
        + "".join(f"export {k}={shlex.quote(v)}\n" for k, v in wandb_env.items())
        + '[ -n "${WANDB_ENTITY:-}" ] || [ -z "${SISTEER_WANDB_ENTITY:-}" ] || export WANDB_ENTITY="$SISTEER_WANDB_ENTITY"\n')
    return f"""#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --partition=general
#SBATCH --qos={qos}
#SBATCH --gres=gpu:{gpus}
#SBATCH --cpus-per-task={16 * gpus}
#SBATCH --mem={128 * gpus}G
#SBATCH --time={time_limit}
#SBATCH --output={logdir}/%x_%j.out
#SBATCH --error={logdir}/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120

# Generated by scripts/two_si_overnight.py — {purpose}
set -uo pipefail
SHARED={shared_root}
{project_export}
# Unattended-window RESULT sentinel, installed FIRST so EVERY exit of this
# script records an outcome — a failed pin guard or a failed sync below as
# much as the workload (a SEPARATE EXIT trap — the signal trap further down
# keeps its process-group cleanup job untouched): on every exit, write a
# one-line outcome (state, exit code, elapsed seconds, job id, node) next to
# the figures and push both to the grading host mirror, so the result is visible on
# the tailnet with no session watching. Best-effort THROUGHOUT (2>/dev/null
# placed BEFORE the redirect so even a failed redirect stays silent, || true,
# and no `exit` inside the trap): a failed sentinel write or mirror must never
# change a finished job's exit code.
export FIGS_DIR="$SHARED/exp/figs/{name}"
mkdir -p "$FIGS_DIR"
result_sentinel(){{
  rc=$?
  state=COMPLETED
  [ "$rc" -ne 0 ] && state=FAILED
  echo "$state exit=$rc elapsed_s=$SECONDS job=${{SLURM_JOB_ID:-unknown}} node=${{SLURMD_NODENAME:-$(hostname)}}" \\
    2>/dev/null > "$FIGS_DIR/RESULT" || true
  "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" >/dev/null 2>&1 || true
}}
trap result_sentinel EXIT

# Pin guard: run EXACTLY the commit this job was submitted for
# (--comment=sha:{sha}).
PINNED_SHA={sha}
# D7: jobs run from an IMMUTABLE commit-addressed checkout, NEVER the shared
# mutable clone ($SISTEER_PROJECT_ROOT/repo). Pointing at the shared clone is
# what broke jobs 148749/148750: it sat at a different commit that lacked
# scripts/train_two_si_objects.py entirely (exit 2 in 8s), and moving its HEAD
# to fix that would mutate a tree peers' running jobs execute from.
REPO="${{SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}}/checkouts/$PINNED_SHA"
if [ ! -d "$REPO" ]; then
  echo "FATAL: no immutable checkout at $REPO -- create it before submitting (D7)." >&2
  exit 1
fi
cd "$REPO"
ACTUAL_SHA=$(git rev-parse HEAD 2>/dev/null)
if [ "$ACTUAL_SHA" != "$PINNED_SHA" ]; then
  echo "FATAL: checkout HEAD $ACTUAL_SHA != pinned $PINNED_SHA." >&2
  exit 1
fi

# The proven env (canary 148354 + full run 148506 both COMPLETED with this):
# source hf_env.sh directly rather than stage2_common, whose s2_init enforces
# EXPECTED_SHA against the SHARED clone and would fight the pinned checkout.
source "$SHARED/scripts/hf_env.sh"
export HF_HUB_OFFLINE=1
export UV_PROJECT_ENVIRONMENT=/home/$USER/venvs/sisteer-attackgen
export UV_CACHE_DIR=$SHARED/.cache/uv UV_LINK_MODE=copy
# The committed lock in the job's flavor, never a re-resolve. torch lives
# ONLY in the flavor extras (tests/test_torch_flavors.py), so a flavor-less
# sync installs none and strips it from an existing venv: the sync names
# cu128, the wheel the cluster jobs assert. A failed frozen sync fails the
# job here, loudly (no errexit in this script, so the handler is explicit;
# the EXIT trap above records the outcome).
uv sync --frozen --extra cu128 --quiet || {{ echo "FATAL: uv sync --frozen --extra cu128 failed in $UV_PROJECT_ENVIRONMENT -- the lock is never re-resolved on a job node" >&2; exit 1; }}
cleanup(){{ kill -TERM -$$ 2>/dev/null; wait; }}
trap cleanup SIGTERM SIGINT SIGQUIT

# Metrics-only LIVE W&B for the fit jobs (spec two-si-stage3, acceptance
# check 5): WANDB_DISABLED stays UNSET so the driver streams numeric
# scalars live, but console capture stays OFF — this job's stdout prints
# raw attack/pair text on error paths, and W&B console capture would
# upload it. The driver (train_two_si_objects/fit_class_b logging) filters
# every payload to numeric scalars and mirrors each step to a per-SI
# metrics jsonl beside the objects, so a W&B outage cannot lose the
# curves. TRANSCRIPT jobs keep full containment (WANDB_DISABLED=true) —
# this is the fit body only.
export WANDB_CONSOLE=off
{wandb_export}# LIVE streaming additionally needs a W&B key ON THE NODE. It must be
# INHERITED from the submitting shell (which exports it from the vault), not
# read from shared storage: this script used to cat a key file under the
# shared root, and on
# a MooseFS shared root chmod is silently ignored, so that file was
# world-readable to every other user of the cluster. No credential is parked
# on shared storage. Absent a key, wandb_fit_run degrades LOUDLY to console +
# per-step jsonl and the fit itself never blocks on W&B.
# Cluster convention for every training sbatch — and the 156649 OOM sat on
# 3.57 GiB of reserved-but-unallocated fragmentation this flag avoids.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
{nccl_env}
# D44: never let a store_dir resolution fall back to node-local /home.
export TT_TWO_SI_STORE_DIR={shlex.quote(store_dir)}
# D58: compiled objects live in ONE commit-independent dir (also passed to
# the driver via --out) — never the checkout's out/, which the next code
# change would orphan.
export TT_TWO_SI_DEFENSES_DIR={shlex.quote(objects_dir)}

{command}
exit $?
"""


def fit_sbatch_script(*, name: str, si: str, pairs_remote: str, config: str,
                      time_limit: str, logdir: str, store_dir: str,
                      objects_dir: str, sha: str, shared_root: str,
                      qos: str = "high", gpus: int = 1,
                      arms: list[str] | None = None,
                      project_root: str | None = None,
                      wandb_env: dict[str, str] | None = None) -> str:
    """The Class A fitting job for one SI: train_two_si_objects.py train —
    one model load per rank for the arms of this SI. With ``gpus`` > 1 the
    driver itself schedules per-arm process groups over the allocation
    (row-sharded: the optimization is unchanged vs single-GPU); the srun
    payload stays a plain single-task python invocation either way.
    ``arms`` restricts the job to an explicit shard of the arm list (the
    --fit-arm-shards split — several jobs per SI for more total width).
    Every convention lives in ``_fit_script_text`` (shared with Class B)."""
    purpose = """the per-SI Class A fitting job (D51):
# ONE model load per rank for the arms of this SI; gpus>1 = the parallel
# driver (per-arm process groups, row-sharded batches — optimization
# unchanged). Logs go to the mode-0700 private _logs dir, the
# pairs jsonl lives in the mode-0700 private pairs dir — attack text never
# lands anywhere world-readable (JB4)."""
    arms_flag = ""
    if arms:
        arms_flag = f" \\\n  --arms {shlex.quote(','.join(arms))}"
    command = f"""srun uv run --no-sync python scripts/train_two_si_objects.py train \\
  --config {shlex.quote(config)} \\
  --si {shlex.quote(si)} \\
  --pairs {shlex.quote(pairs_remote)} \\
  --out {shlex.quote(objects_dir)}{arms_flag}"""
    return _fit_script_text(
        name=name, purpose=purpose, command=command, time_limit=time_limit,
        logdir=logdir, store_dir=store_dir, objects_dir=objects_dir, sha=sha,
        shared_root=shared_root, qos=qos, gpus=gpus, project_root=project_root,
        wandb_env=wandb_env)


def fit_class_b_sbatch_script(*, name: str, config: str, time_limit: str,
                              logdir: str, store_dir: str, objects_dir: str,
                              sha: str, shared_root: str,
                              qos: str = "high",
                              project_root: str | None = None,
                              wandb_env: dict[str, str] | None = None) -> str:
    """The Class B fitting job: fit_class_b.py fits the config's
    class_b_arms (antidote, alphasteer, and caa_static_r since 2026-08-21 —
    circuit_breaker left the REGISTRY entirely 2026-08-20; its
    implementation, load_object hydration, and fitted object survive) ONCE
    each on the AlphaSteer authors' 14k/2k
    prompt lists (configs/two_si.yaml class_b block), saving every object
    plus its meta.json completion sentinel the instant it finishes and
    skipping arms whose sentinel already exists — so a resubmission never
    redoes a finished arm.

    TT_TWO_SI_STORE_DIR (exported in the BODY by ``_fit_script_text`` — never
    via sbatch --export, hard rule 1) is kept on every submission: no Class B
    arm reads the store any more, but the export is harmless and keeps the
    fit-a/fit-b script templates identical.

    Deliberately NOT scripts/sbatch/fit_class_b.sbatch: that companion runs
    on top of stage2_common.sh sourced from the shared MUTABLE clone, whose
    s2_init aborts the job at start whenever another lane moves that clone's
    HEAD off its EXPECTED_SHA — a failure this lane already hit. The shared
    template here sources hf_env.sh directly and pins the immutable
    commit-addressed checkout instead."""
    purpose = """the SI-agnostic Class B fitting job:
# fit_class_b.py fits the config's class_b_arms once each on
# harmful/benign CONTENT (no system instruction), writing each object plus
# its meta.json completion sentinel to the defenses dir the moment it
# finishes. Logs go to the mode-0700 private _logs dir (a crash can print
# store/attack text, JB4). No job array: one set of objects total."""
    command = f"""srun uv run --no-sync python scripts/fit_class_b.py \\
  --config {shlex.quote(config)} \\
  --out {shlex.quote(objects_dir)}"""
    return _fit_script_text(
        name=name, purpose=purpose, command=command, time_limit=time_limit,
        logdir=logdir, store_dir=store_dir, objects_dir=objects_dir, sha=sha,
        shared_root=shared_root, qos=qos, project_root=project_root,
        wandb_env=wandb_env)


def resolve_project_root(shared_root: str, explicit: str | None) -> str:
    """The cluster project directory the fit jobs' immutable checkouts live under (<root>/checkouts/<sha>):
    --remote-project, else CLUSTER_PROJECT_ROOT of the site block whose CLUSTER_SHARED_ROOT is the run's shared root
    (reproduce/lib/clusters.py; every block under the site's clusters/ directory, listed in SISTEER_CLUSTERS or not, the
    listed ones first). Refused by name when neither gives it."""
    if explicit:
        return explicit.rstrip("/")
    lib = str(REPO / "reproduce" / "lib")
    if lib not in sys.path:
        sys.path.append(lib)
    import clusters as CL  # noqa: E402 (the site's cluster blocks)
    cfg = CL.default_config_path()
    try:
        blocks = CL.load_addressable(os.path.dirname(cfg), CL.site_config_file(cfg)) if os.path.isfile(cfg) else {}
    except SystemExit as exc:      # the reader's refusals are SystemExit
        raise StageFailed(f"--remote-project not given and the site config {cfg} could not be read: {exc}") from None
    for block in blocks.values():
        if block.get("CLUSTER_SHARED_ROOT", "").rstrip("/") == shared_root.rstrip("/") and block.get("CLUSTER_PROJECT_ROOT"):
            return block["CLUSTER_PROJECT_ROOT"].rstrip("/")
    raise StageFailed(f"no project directory for the shared root {shared_root}: pass --remote-project, or describe that "
                      "cluster (CLUSTER_SHARED_ROOT, CLUSTER_PROJECT_ROOT) in the site config")


def resolve_wandb_env() -> dict[str, str]:
    """The site's W&B destination for the fit jobs: the keys the front door's job env carries (lib/submit_twin.py
    site_wandb_env, the same function), read from the site config this host uses (reproduce/lib/clusters.py
    default_config_path). Empty when the site names no entity, or when this host has no site config file."""
    lib = str(REPO / "reproduce" / "lib")
    if lib not in sys.path:
        sys.path.append(lib)
    import clusters as CL  # noqa: E402 (the site's cluster blocks)
    cfg = CL.default_config_path()
    if not os.path.isfile(cfg):
        return {}
    saved = list(sys.path)
    try:
        import submit_twin as SUB  # noqa: E402 (the funnel's site reader and its W&B keys)
    finally:
        sys.path[:] = saved          # the funnel puts its own directory first; this process keeps its search order
    try:
        return SUB.site_wandb_env(SUB.Site(cfg, True))
    except SystemExit as exc:      # the site readers' refusals are SystemExit
        raise StageFailed(f"the site config {cfg} could not be read for its W&B destination: {exc}") from None


@dataclass
class JobState:
    """One monitored cluster job (the generate job, or one SI's fit job)."""

    label: str                       # log label ("generate", "fit:<si>")
    name: str                        # slurm job name
    job_id: str | None = None        # None -> needs (re)submission
    attempts: int = 0                # submissions counted against the cap
    resubmit: object = None          # callable -> job_id, or None (cannot)
    done: bool = False
    failed_reason: str | None = None
    row: dict | None = None          # last sacct row seen

    @property
    def finished(self) -> bool:
        return self.done or self.failed_reason is not None


class OvernightChain:
    def __init__(self, args, *, runner=default_runner, clock=time.monotonic,
                 sleep=None):
        self.args = args
        self.runner = runner
        self.clock = clock
        self.sleep = sleep if sleep is not None else time.sleep
        self.deadline = clock() + args.deadline_hours * 3600.0

        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = REPO / args.config
        import yaml
        self.cfg = yaml.safe_load(cfg_path.read_text())
        self.si_ids: list[str] = list(self.cfg["si_ids"])
        self.constitution_run = self.cfg["pool"].get(
            "constitution_run", "two_si_pos_constitution")
        self.pairs_run = self.cfg["pool"].get("pairs_run", "two_si_pairs")

        self.private_root = Path(args.private_root).expanduser() \
            if args.private_root else self._default_private_root()
        self.remote_root = (args.remote_root or default_remote_root()).rstrip("/")
        self.shared_root = args.remote_shared or str(
            Path(self.remote_root).parents[1])
        # the project directory the fit jobs' pinned checkouts live under, resolved before any work so an unattended
        # run refuses up front rather than at the fit stage
        self.project_root = resolve_project_root(self.shared_root, getattr(args, "remote_project", None))
        # the site's W&B destination for the fit scripts, read up front the same way
        self.wandb_env = resolve_wandb_env()
        # the login host every ssh / rsync goes to: --cluster-host, else CLUSTER_SSH of the site block whose shared root is
        # the run's; refused by name up front when neither names one
        self.cluster_host = args.cluster_host or site_block_for(self.shared_root).get("CLUSTER_SSH", "")
        if not self.cluster_host:
            raise StageFailed(f"no login host for the shared root {self.shared_root}: pass --cluster-host, or describe that "
                              "cluster (CLUSTER_SHARED_ROOT, CLUSTER_SSH) in the site config")
        # QoS is the cluster's (see derive_qos) unless overridden; ONE
        # value feeds the rendered #SBATCH header and every submit-time
        # --qos, so they can never disagree.
        self.qos = args.qos or derive_qos(self.shared_root)
        # per-run copy of the exclusion list (--exclude-nodes, else the
        # site's list for this cluster); grows on busy-GPU failures.
        self.excluded: list[str] = (
            [n for n in args.exclude_nodes.split(",") if n]
            if args.exclude_nodes is not None else site_exclude_nodes(self.shared_root))
        self.remote_store_dir = args.remote_store_dir or \
            f"{self.remote_root}/two_si_store"
        # ONE absolute, commit-independent objects dir for BOTH fit
        # classes, derived from the remote root — NEVER inside the per-commit
        # checkout, where any code change (new sha -> new checkout) would
        # silently void resume and refit every object.
        self.remote_objects_dir = args.remote_objects_dir or \
            f"{self.remote_root}/two_si_defenses/cells"
        self.remote_logdir = f"{self.remote_root}/_logs"
        self.remote_jobdir = f"{self.remote_root}/_overnight"
        self.remote_pairs_dir = f"{self.remote_root}/{self.pairs_run}"
        self.local_pairs_dir = self.private_root / self.pairs_run

        log_dir = self.private_root / "two_si_overnight"
        self._mkdir_private(log_dir)
        self.log_path = Path(args.log_file) if args.log_file else \
            log_dir / f"{args.run_tag}.log.jsonl"
        self.lock_path = log_dir / f"{args.run_tag}.lock"
        self._seq = 0
        self._sha: str | None = args.sha or None
        self.stage_status: dict[str, str] = {}
        self.jobs: list[JobState] = []

    # ----------------------------------------------------------------- utils

    @staticmethod
    def _default_private_root() -> Path:
        from sisteer.eval.jailbreak_artifacts import grading_host_private_root
        return grading_host_private_root()

    @staticmethod
    def _mkdir_private(path: Path) -> None:
        prev = os.umask(0o077)
        try:
            path.mkdir(parents=True, exist_ok=True)
        finally:
            os.umask(prev)
        try:
            path.chmod(0o700)
        except OSError:
            pass

    def log(self, event: str, **fields) -> None:
        """Append one structured record to the JSONL progress log AND echo a
        human line — the log is the report (task step 7): every submission,
        verdict, retry and exclusion lands here as it happens."""
        rec = {"seq": self._seq,
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "event": event, **fields}
        self._seq += 1
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        detail = " ".join(f"{k}={v}" for k, v in fields.items())
        print(f"[overnight] {event} {detail}".rstrip(), flush=True)

    def remaining(self) -> float:
        return self.deadline - self.clock()

    def _check_deadline(self, where: str) -> None:
        if self.remaining() <= 0:
            raise DeadlineExpired(where)

    def _sleep_poll(self) -> None:
        self.sleep(min(self.args.poll_seconds, max(1.0, self.remaining())))

    def _run(self, argv: list[str], *, what: str, timeout: float | None = None):
        try:
            cp = self.runner(argv, timeout=timeout)
        except subprocess.TimeoutExpired:
            self.log("command_timeout", what=what, argv=argv)
            self._check_deadline(what)          # past deadline -> exit 3 path
            raise StageFailed(f"{what}: command timed out: {argv}")
        return cp

    def _ssh(self, command: str, *, what: str, timeout: float = 300.0):
        argv = ["ssh", "-o", "BatchMode=yes", self.cluster_host, command]
        return self._run(argv, what=what, timeout=timeout)

    # ----------------------------------------------------------------- slurm

    def _sacct(self, *, job_id: str | None = None,
               name: str | None = None) -> list[dict]:
        # Job ids RECYCLE on this cluster: a bare `sacct -j <id>` can return
        # an unrelated months-old job — the n1 run filed a false "TIMEOUT
        # (exit 0:0)" verdict, citing a nonexistent array-style log, off
        # exactly such a recycled row. EVERY query is therefore scoped to
        # THIS user and a recent submit window (the same 7-day window the
        # by-name adoption already used, so explicitly-passed and adopted
        # job ids from earlier days stay visible), and -X keeps only the
        # main-job row — a stale or foreign row can never become a verdict.
        scope = '-u "$USER" -S now-7days'
        if job_id is not None:
            cmd = (f"sacct -n -P -X {scope} -j {shlex.quote(str(job_id))} "
                   f"--format={SACCT_FORMAT}")
        else:
            cmd = (f"sacct -n -P -X {scope} --name={shlex.quote(name)} "
                   f"--format={SACCT_FORMAT}")
        cp = self._ssh(cmd, what="sacct")
        rows = []
        for line in (cp.stdout or "").splitlines():
            parts = line.strip().split("|")
            if len(parts) < 6 or not parts[0]:
                continue
            rows.append({"job_id": parts[0], "name": parts[1],
                         "state": parts[2].split()[0], "elapsed": parts[3],
                         "exit_code": parts[4], "node": parts[5]})
        return rows

    def _latest_by_name(self, name: str) -> dict | None:
        def key(row):
            m = re.match(r"(\d+)", row["job_id"])
            return int(m.group(1)) if m else -1
        rows = [r for r in self._sacct(name=name) if r["name"] == name]
        return max(rows, key=key) if rows else None

    def _job_output_tail(self, name: str, job_id: str) -> str:
        """The tail of the job's stderr+stdout (the lane's %x_%j log
        convention) for signature detection. Unreadable logs -> empty string,
        which classifies as NOT the busy signature (i.e. terminal — the safe
        direction: an unreadable log must not license unbounded retries)."""
        err = shlex.quote(f"{self.remote_logdir}/{name}_{job_id}.err")
        out = shlex.quote(f"{self.remote_logdir}/{name}_{job_id}.out")
        cp = self._ssh(f"tail -c 20000 {err} 2>/dev/null; "
                       f"tail -c 20000 {out} 2>/dev/null",
                       what="job-log-tail")
        return cp.stdout or ""

    def classify_failure(self, row: dict, output_tail: str) -> str:
        """'completed' | 'retryable' | 'terminal' for a TERMINAL sacct row.

        Retryable is EXACTLY the GPU leak: NODE_FAIL (the variant where
        slurmd itself dies), or FAILED with the busy-GPU signature in the
        job's own output AND a short elapsed time (the leak kills jobs in
        ~40s–2min; --busy-elapsed-max gates it). Everything else — a real
        code failure, CANCELLED (someone chose to stop it), TIMEOUT, OOM —
        is terminal and never retried."""
        state = row["state"]
        if state == "COMPLETED":
            return "completed"
        if state == "NODE_FAIL":
            return "retryable"
        if state == "FAILED":
            short = elapsed_seconds(row["elapsed"]) <= self.args.busy_elapsed_max
            if short and BUSY_GPU_SIGNATURE.search(output_tail or ""):
                return "retryable"
        return "terminal"

    def _record_bad_node(self, node: str | None) -> None:
        """Record + exclude — NEVER kill/drain (that is an admin's call)."""
        if not node or node.lower().startswith("none"):
            return
        if node not in self.excluded:
            self.excluded.append(node)
            self.log("node_excluded", node=node, excluded=list(self.excluded))

    def _submit(self, script_remote: str, *, name: str) -> str:
        """sbatch with EVERY required submit-time flag; logs the exclusion
        list on every submission. Bounded submit retries (transient
        slurmctld hiccups), then StageFailed."""
        sha = self._pinned_sha()
        exclude_flag = (f"--exclude={','.join(self.excluded)} "
                        if self.excluded else "")
        cmd = (f"sbatch {exclude_flag}--qos={self.qos} "
               f"--comment=sha:{sha} --job-name={shlex.quote(name)} "
               f"{shlex.quote(script_remote)}")
        for attempt in range(1, 4):
            self._check_deadline(f"submitting {name}")
            self.log("sbatch_submit", name=name, script=script_remote,
                     qos=self.qos, sha=sha, exclude=list(self.excluded),
                     submit_attempt=attempt)
            cp = self._ssh(cmd, what=f"sbatch {name}")
            m = re.search(r"Submitted batch job (\d+)", cp.stdout or "")
            if cp.returncode == 0 and m:
                job_id = m.group(1)
                self.log("sbatch_accepted", name=name, job_id=job_id)
                return job_id
            self.log("sbatch_rejected", name=name, rc=cp.returncode,
                     stderr=(cp.stderr or "")[-500:])
            if attempt < 3:
                self.sleep(60)
        raise StageFailed(f"sbatch for {name} rejected 3 times")

    def _pinned_sha(self) -> str:
        """The commit every submission pins (--comment=sha:<...> + the
        in-script guard): the cluster clone's EXPECTED_SHA — the same value
        stage2_common's s2_init enforces, so submission and guard agree."""
        if self._sha:
            return self._sha
        # Pin to THIS worktree's HEAD — the commit whose code we actually want
        # run. Reading the cluster clone's EXPECTED_SHA (the previous behaviour)
        # pins to whatever a DIFFERENT lane last synced, which is how
        # 148749/148750 were dispatched at a commit lacking the fit script.
        cp = self._run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                       what="resolve this worktree's HEAD", timeout=60)
        sha = (cp.stdout or "").strip()
        if not _SHA_RE.match(sha):
            raise StageFailed(
                f"cannot resolve this worktree's HEAD (got {sha!r})")
        self._sha = sha
        return sha

    # -------------------------------------------------------------- monitor

    def _monitor(self, jobs: list[JobState]) -> None:
        """Poll the given jobs to a terminal state, applying the bounded
        retry-with-exclusion policy. Returns when every job is done or
        failed; raises DeadlineExpired at the wall-clock bound (jobs are left
        RUNNING — never cancelled — and named in the final summary)."""
        while True:
            self._check_deadline("monitoring jobs")
            for st in jobs:
                if st.finished:
                    continue
                if st.job_id is None:                      # needs resubmission
                    if st.resubmit is None:
                        st.failed_reason = (
                            "retryable failure but no resubmit command known "
                            "(pass --generate-script to let the orchestrator "
                            "resubmit the generate job)")
                        self.log("job_failed", label=st.label,
                                 reason=st.failed_reason)
                        continue
                    if st.attempts >= self.args.max_attempts:
                        st.failed_reason = (
                            f"attempt cap {self.args.max_attempts} reached")
                        self.log("job_failed", label=st.label,
                                 reason=st.failed_reason)
                        continue
                    st.attempts += 1
                    st.job_id = st.resubmit()
                    self.log("job_resubmitted", label=st.label, name=st.name,
                             job_id=st.job_id, attempt=st.attempts)
                    continue
                rows = self._sacct(job_id=st.job_id)
                row = rows[0] if rows else None
                if row is None or row["state"] not in TERMINAL_STATES:
                    continue                                # pending/running
                st.row = row
                tail = "" if row["state"] == "COMPLETED" else \
                    self._job_output_tail(st.name, st.job_id)
                verdict = self.classify_failure(row, tail)
                self.log("job_terminal", label=st.label, job_id=st.job_id,
                         state=row["state"], elapsed=row["elapsed"],
                         node=row["node"], verdict=verdict,
                         attempt=max(st.attempts, 1))
                if verdict == "completed":
                    st.done = True
                elif verdict == "retryable":
                    self._record_bad_node(row["node"])
                    st.job_id = None                        # resubmit next pass
                else:
                    st.failed_reason = (
                        f"terminal {row['state']} (exit {row['exit_code']}) — "
                        "not the busy-GPU signature, so NOT retried; see "
                        f"{self.remote_logdir}/{st.name}_{row['job_id']}.err")
                    self.log("job_failed", label=st.label,
                             reason=st.failed_reason)
            if all(st.finished for st in jobs):
                return
            self._sleep_poll()

    # --------------------------------------------------------------- stages

    def stage_wait_generate(self) -> None:
        self._check_deadline("stage wait-generate")
        name = self.args.generate_name
        job_id = self.args.generate_job_id
        if not job_id:
            row = self._latest_by_name(name)
            if row is None:
                # No such job on record: generation may have finished outside
                # the sacct window, or never ran. Proceed — the manifest's
                # regime gate (stage 4) stops the run before any fit could
                # train on an incomplete pool.
                self.log("generate_job_not_found", name=name,
                         note="proceeding; the regime gate protects fitting")
                self.stage_status["wait-generate"] = "skipped (no job found)"
                return
            job_id = row["job_id"]
        resub = None
        if self.args.generate_script:
            script = self.args.generate_script
            resub = lambda: self._submit(script, name=name)  # noqa: E731
        st = JobState(label="generate", name=name, job_id=str(job_id),
                      attempts=1, resubmit=resub)
        self.jobs.append(st)
        self.log("stage", stage="wait-generate", name=name, job_id=st.job_id)
        self._monitor([st])
        if st.failed_reason:
            raise StageFailed(f"generate job {name}: {st.failed_reason}")
        self.stage_status["wait-generate"] = "completed"

    def _rsync(self, src: str, dst: str, *, what: str) -> bool:
        argv = ["rsync", "-az", "--chmod=D700,F600", src, dst]
        for attempt in range(1, 4):
            self._check_deadline(what)
            cp = self._run(argv, what=what,
                           timeout=max(60.0, min(3600.0, self.remaining())))
            self.log("rsync", what=what, src=src, dst=dst, rc=cp.returncode,
                     attempt=attempt)
            if cp.returncode == 0:
                return True
            if attempt < 3:
                self.sleep(30)
        return False

    def stage_pull_rows(self) -> None:
        self._check_deadline("stage pull")
        local = self.private_root / self.constitution_run
        self._mkdir_private(self.private_root)
        self._mkdir_private(local)
        src = f"{self.cluster_host}:{self.remote_root}/{self.constitution_run}/"
        ok = self._rsync(src, str(local) + "/", what="pull constitution rows")
        if not ok:
            # A failed pull with rows already local (an earlier pull) is
            # resumable; a failed pull with NOTHING local is terminal.
            if any(local.glob("*.jsonl")):
                self.log("pull_failed_but_local_rows_exist", local=str(local))
                self.stage_status["pull"] = "stale (rsync failed; using local)"
                return
            raise StageFailed(
                f"rsync of {src} failed 3x and no local rows exist — nothing "
                "to grade. Check the generate job's output dir.")
        self.stage_status["pull"] = "completed"

    def _pipeline(self, subcmd: str, *extra: str) -> None:
        """One build_two_si_pool.py subcommand, locally, through the runner.
        A nonzero exit is a REAL code/data failure: terminal, never retried."""
        self._check_deadline(f"stage {subcmd}")
        argv = [sys.executable, str(REPO / "scripts" / "build_two_si_pool.py"),
                subcmd, "--config", self.args.config,
                "--private-root", str(self.private_root), *extra]
        cp = self._run(argv, what=subcmd, timeout=max(60.0, self.remaining()))
        self.log("pipeline", subcmd=subcmd, rc=cp.returncode,
                 stdout_tail=(cp.stdout or "")[-2000:])
        if cp.returncode != 0:
            raise StageFailed(
                f"{subcmd} failed (rc={cp.returncode}): "
                f"{(cp.stderr or cp.stdout or '')[-2000:]}")
        self.stage_status[subcmd] = "completed"

    def stage_grade(self) -> None:
        # Coordination courtesy line (print-and-proceed): when the site names
        # a coordination channel (SISTEER_COORDINATION_CHANNEL), printed into
        # BOTH the console and the structured log so the operator can post it
        # there — the stage never waits on the post going out.
        from sisteer import coordination
        channel = coordination.channel()
        model_id = (self.cfg.get("grader") or {}).get("model",
                                                      "unknown-model")
        line = (f"{self.args.concurrency} {model_id} lowprio ~1h "
                "(two-SI overnight grade-pos, live StrongREJECT judgments)")
        if channel:
            print(f"[coordination] ready to post to {channel} "
                  "(grading proceeds now):\n"
                  f"  {line}", flush=True)
            self.log("coordination_post", line=line)
        self._pipeline("grade-pos", "--concurrency", str(self.args.concurrency))

    def stage_manifest(self) -> dict:
        self._pipeline("manifest")
        mpath = Path(self.cfg["pool_manifest"])
        if not mpath.is_absolute():
            mpath = REPO / mpath
        manifest = json.loads(mpath.read_text())
        self.log("manifest_frozen", regime=manifest["regime"],
                 pool_sizes=manifest["pool_sizes"],
                 missing={si: c["n_missing"]
                          for si, c in manifest["coverage"].items()},
                 manifest_sha256=manifest["manifest_sha256"])
        if manifest["regime"] != "matched_intersection" \
                and not self.args.allow_native:
            raise StageFailed(
                f"manifest regime is {manifest['regime']} — the matched "
                "intersection did not materialise, so an UNATTENDED run stops "
                "here rather than silently fitting the confounded comparison "
                "(D26). Inspect the coverage gaps, or re-run with "
                "--allow-native to fit the labelled native pools.")
        return manifest

    def stage_emit_pairs(self) -> dict:
        self._pipeline("emit-pairs")
        sidecar = json.loads(
            (self.local_pairs_dir / bp.PAIRS_META_NAME).read_text())
        self.log("pairs_emitted", regime=sidecar["regime"],
                 pool_sizes=sidecar["pool_sizes"],
                 n_ungraded_pos_skipped=sidecar["n_ungraded_pos_skipped"])
        return sidecar

    def stage_push_pairs(self) -> None:
        self._check_deadline("stage push-pairs")
        # Pre-create every remote dir at mode 0700 (Slurm does NOT create log
        # parents, and the pairs/text dirs must never exist world-readable).
        dirs = " ".join(shlex.quote(d) for d in (
            self.remote_root, self.remote_logdir, self.remote_jobdir,
            self.remote_pairs_dir))
        cp = self._ssh(f"install -d -m 700 {dirs}", what="prepare remote dirs")
        if cp.returncode != 0:
            raise StageFailed(
                f"could not create remote 0700 dirs: {(cp.stderr or '')[-500:]}")
        ok = self._rsync(str(self.local_pairs_dir) + "/",
                         f"{self.cluster_host}:{self.remote_pairs_dir}/",
                         what="push pairs")
        if not ok:
            raise StageFailed("rsync push of the pairs dir failed 3x")
        self.stage_status["push-pairs"] = "completed"

    def _fit_arm_shards(self) -> list[list[str] | None]:
        """The per-SI job split: ``[None]`` (one job, every arm — the
        default) or ``--fit-arm-shards N`` balanced shards of the Class A
        arm list (longest-first by the driver's static cost table, config
        order preserved within a shard). Several jobs per SI buy more
        TOTAL width when one node's GPUs are not enough — each job still
        runs the same parallel driver on its own allocation, and the
        objects dir + resume-by-skip are shared, so shards never collide
        (disjoint arms) and a re-run only fits what is missing."""
        n = int(getattr(self.args, "fit_arm_shards", 1) or 1)
        if n <= 1:
            return [None]
        import train_two_si_objects as tt

        arms = tt.default_class_a_arms(self.cfg)
        order = sorted(arms, key=lambda a: (-tt._ARM_COST.get(
            a, tt._DEFAULT_ARM_COST), a))
        shards: list[list[str]] = [[] for _ in range(n)]
        loads = [0.0] * n
        for a in order:
            i = loads.index(min(loads))
            shards[i].append(a)
            loads[i] += tt._ARM_COST.get(a, tt._DEFAULT_ARM_COST)
        return [[a for a in arms if a in s] for s in shards if s]

    def stage_fit(self, sidecar: dict) -> None:
        self._check_deadline("stage fit")
        fit_jobs: list[JobState] = []
        shard_lists = self._fit_arm_shards()
        for si in self.si_ids:
            if sidecar["pool_sizes"].get(si, 0) <= 0:
                raise StageFailed(
                    f"emit-pairs produced 0 pairs for {si} — refusing to "
                    "submit a fit job with an empty pool.")
            for j, shard in enumerate(shard_lists):
                suffix = f"-p{j}" if len(shard_lists) > 1 else ""
                name = f"tt-twosi-fit-{si}{suffix}-{self.args.run_tag}"
                label = f"fit:{si}{suffix}"
                existing = self._latest_by_name(name)
                if existing and existing["state"] == "COMPLETED":
                    # Re-entry: this fit already finished — never redo it.
                    self.log("fit_already_completed", si=si, name=name,
                             job_id=existing["job_id"])
                    st = JobState(label=label, name=name,
                                  job_id=existing["job_id"], attempts=1,
                                  done=True, row=existing)
                    fit_jobs.append(st)
                    self.jobs.append(st)
                    continue
                script_remote = f"{self.remote_jobdir}/{name}.sbatch"
                submit = self._make_fit_submitter(si, name, script_remote,
                                                  arms=shard)
                if existing and existing["state"] not in TERMINAL_STATES:
                    # Re-entry: adopt the live job, never double-submit.
                    self.log("fit_adopted_running", si=si, name=name,
                             job_id=existing["job_id"],
                             state=existing["state"])
                    st = JobState(label=label, name=name,
                                  job_id=existing["job_id"], attempts=1,
                                  resubmit=submit)
                else:
                    st = JobState(label=label, name=name, job_id=submit(),
                                  attempts=1, resubmit=submit)
                fit_jobs.append(st)
                self.jobs.append(st)
        self._monitor(fit_jobs)
        failed = [st for st in fit_jobs if st.failed_reason]
        if failed:
            raise StageFailed("; ".join(
                f"{st.label}: {st.failed_reason}" for st in failed))
        self.stage_status["fit"] = "completed"

    def _make_submitter(self, name: str, script_remote: str, render):
        """The ship-and-submit closure shared by EVERY fitting job (Class A
        and Class B): render the sbatch text (lazily, at submit time), keep a
        local copy in the private log dir, rsync it to the cluster, sbatch
        it. One mechanism, so both classes dodge the same traps the same
        way."""
        def submit() -> str:
            local = self.private_root / "two_si_overnight" / f"{name}.sbatch"
            self._mkdir_private(local.parent)
            local.write_text(render())
            if not self._rsync(str(local), f"{self.cluster_host}:{script_remote}",
                               what=f"ship {name}.sbatch"):
                raise StageFailed(f"could not ship {name}.sbatch to the cluster")
            return self._submit(script_remote, name=name)
        return submit

    def _make_fit_submitter(self, si: str, name: str, script_remote: str,
                            arms: list[str] | None = None):
        def render() -> str:
            pairs_remote = f"{self.remote_pairs_dir}/{bp.pairs_filename(si)}"
            gpus = int((self.cfg.get("data_parallel") or {})
                       .get("gpus", 1) or 1)
            return fit_sbatch_script(
                name=name, si=si, pairs_remote=pairs_remote,
                config=self.args.config, time_limit=self.args.fit_time,
                logdir=self.remote_logdir, store_dir=self.remote_store_dir,
                objects_dir=self.remote_objects_dir,
                sha=self._pinned_sha(), shared_root=self.shared_root,
                qos=self.qos, gpus=gpus, arms=arms, project_root=self.project_root,
                wandb_env=self.wandb_env)
        return self._make_submitter(name, script_remote, render)

    def _make_fit_b_submitter(self, name: str, script_remote: str):
        def render() -> str:
            return fit_class_b_sbatch_script(
                name=name, config=self.args.config,
                time_limit=self.args.fit_b_time, logdir=self.remote_logdir,
                store_dir=self.remote_store_dir,
                objects_dir=self.remote_objects_dir, sha=self._pinned_sha(),
                shared_root=self.shared_root, qos=self.qos,
                project_root=self.project_root, wandb_env=self.wandb_env)
        return self._make_submitter(name, script_remote, render)

    def stage_fit_class_b(self) -> None:
        """Step 7: the SI-agnostic Class B fits — ONE job running
        fit_class_b.py (the config's class_b_arms, each fit once
        on content, no SI), through the same generated-script + monitor +
        bounded-retry mechanism as the Class A fits. Re-entry adopts a
        COMPLETED job by name (never redone) or a RUNNING one (never
        double-submitted); on top of that, fit_class_b.py itself skips arms
        whose meta.json sentinel exists, so even a resubmission only fits
        what is missing."""
        self._check_deadline("stage fit-b")
        name = f"tt-twosi-fitb-{self.args.run_tag}"
        existing = self._latest_by_name(name)
        if existing and existing["state"] == "COMPLETED":
            # Re-entry: the Class B fit already finished — never redo it.
            self.log("fit_b_already_completed", name=name,
                     job_id=existing["job_id"])
            self.jobs.append(JobState(label="fit-b", name=name,
                                      job_id=existing["job_id"], attempts=1,
                                      done=True, row=existing))
            self.stage_status["fit-b"] = "completed"
            return
        script_remote = f"{self.remote_jobdir}/{name}.sbatch"
        submit = self._make_fit_b_submitter(name, script_remote)
        if existing and existing["state"] not in TERMINAL_STATES:
            # Re-entry: adopt the live job instead of double-submitting.
            self.log("fit_b_adopted_running", name=name,
                     job_id=existing["job_id"], state=existing["state"])
            st = JobState(label="fit-b", name=name,
                          job_id=existing["job_id"], attempts=1,
                          resubmit=submit)
        else:
            # Class B must be submittable even when the Class A chain died
            # before stage_push_pairs created the remote dirs (the classes
            # share no inputs) — make its own, idempotently, at mode 0700
            # (the job's logs can print store text, JB4).
            dirs = " ".join(shlex.quote(d) for d in (
                self.remote_root, self.remote_logdir, self.remote_jobdir))
            cp = self._ssh(f"install -d -m 700 {dirs}",
                           what="prepare remote dirs (fit-b)")
            if cp.returncode != 0:
                raise StageFailed(
                    "fit-b: could not create remote 0700 dirs: "
                    f"{(cp.stderr or '')[-500:]}")
            st = JobState(label="fit-b", name=name, job_id=submit(),
                          attempts=1, resubmit=submit)
        self.jobs.append(st)
        self._monitor([st])
        if st.failed_reason:
            raise StageFailed(f"fit-b job {name}: {st.failed_reason}")
        self.stage_status["fit-b"] = "completed"

    # ------------------------------------------------------------------ run

    def _acquire_lock(self) -> bool:
        if self.lock_path.exists():
            try:
                pid = int(self.lock_path.read_text().strip())
                os.kill(pid, 0)                    # signal 0: existence probe
            except (ValueError, ProcessLookupError):
                pass                               # stale -> take over
            except PermissionError:
                return False                       # alive under another uid
            else:
                return False                       # alive -> refuse
        self.lock_path.write_text(str(os.getpid()))
        return True

    def _summary(self, outcome: str) -> None:
        jobs = [{"label": st.label, "name": st.name, "job_id": st.job_id,
                 "attempts": st.attempts, "done": st.done,
                 "failed_reason": st.failed_reason,
                 "state": st.row["state"] if st.row else "unknown"}
                for st in self.jobs]
        still_running = [j for j in jobs
                         if not j["done"] and j["failed_reason"] is None]
        self.log("summary", outcome=outcome, stages=self.stage_status,
                 jobs=jobs, still_running=[j["name"] for j in still_running],
                 excluded_nodes=list(self.excluded),
                 log_file=str(self.log_path))
        if still_running:
            print("[overnight] NOTE: jobs left RUNNING (never cancelled): "
                  + ", ".join(f"{j['name']}({j['job_id']})"
                              for j in still_running), flush=True)

    def run(self) -> int:
        if not self._acquire_lock():
            print(f"[overnight] another live instance holds {self.lock_path} "
                  "— refusing to run twice concurrently.", flush=True)
            return 4
        try:
            self.log("start", run_tag=self.args.run_tag,
                     deadline_hours=self.args.deadline_hours,
                     exclude_nodes=list(self.excluded),
                     cluster_host=self.cluster_host,
                     remote_root=self.remote_root,
                     remote_objects_dir=self.remote_objects_dir,
                     private_root=str(self.private_root),
                     config=self.args.config)
            # Steps 1-6 (the Class A chain: pool -> per-SI fits) and step 7
            # (the SI-agnostic Class B fit) share NO inputs, so a terminal
            # failure in either half must never skip the other: both are
            # always attempted, their failures are combined, and any failure
            # still exits 1. DeadlineExpired is deliberately NOT caught here
            # — the wall-clock bound aborts everything (exit 3), as before.
            errors: list[str] = []
            try:
                self.stage_wait_generate()
                self.stage_pull_rows()
                self.stage_grade()
                self.stage_manifest()
                sidecar = self.stage_emit_pairs()
                self.stage_push_pairs()
                self.stage_fit(sidecar)
            except StageFailed as e:
                errors.append(str(e))
                self.log("class_a_chain_failed", error=str(e),
                         note="Class B still runs — the classes share no inputs")
            try:
                self.stage_fit_class_b()
            except StageFailed as e:
                errors.append(str(e))
                self.log("class_b_failed", error=str(e),
                         note="Class A unaffected — the classes share no inputs")
            if errors:
                raise StageFailed("; ".join(errors))
            self._summary("complete")
            return 0
        except DeadlineExpired as e:
            self.stage_status["deadline_hit_during"] = str(e)
            self._summary("deadline_expired")
            return 3
        except StageFailed as e:
            self.stage_status["failed"] = str(e)
            self.log("stage_failed", error=str(e))
            self._summary("failed")
            return 1
        finally:
            try:
                self.lock_path.unlink()
            except OSError:
                pass


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/two_si.yaml")
    ap.add_argument("--cluster-host", default=None,
                    help="ssh alias of the cluster's login host (default: "
                         "CLUSTER_SSH of the site block whose shared root is "
                         "the run's)")
    ap.add_argument("--remote-root", default=None,
                    help="cluster private root (mode-0700; default: <the "
                         "site's shared root>/exp/jailbreak). Point a run only "
                         "at storage that honours modes: on a filesystem that "
                         "IGNORES chmod everything is world-readable to "
                         "cluster users — only with that exposure accepted")
    ap.add_argument("--remote-project", default=None,
                    help="the cluster project directory the fit jobs' pinned checkouts live under (<dir>/checkouts/<sha>); "
                         "default: CLUSTER_PROJECT_ROOT of the site block whose shared root is the run's")
    ap.add_argument("--remote-shared", default=None,
                    help="cluster shared root (default: derived from "
                         "--remote-root, its grandparent directory)")
    ap.add_argument("--qos", default=None,
                    help="submit-time QoS for every sbatch, also rendered "
                         "into the generated headers (default: CLUSTER_QOS "
                         "of the site block whose shared root is the run's)")
    ap.add_argument("--exclude-nodes", default=None,
                    help="comma list of nodes excluded on every sbatch "
                         "(default: the --exclude list in the site block's "
                         "CLUSTER_SBATCH_EXTRA; an empty value excludes none); "
                         "nodes that fail busy are added during the run")
    ap.add_argument("--remote-store-dir", default=None,
                    help="TT_TWO_SI_STORE_DIR for fit jobs (default "
                         "<remote-root>/two_si_store — the D43 shipped subset)")
    ap.add_argument("--remote-objects-dir", default=None,
                    help="ABSOLUTE compiled-objects dir passed to BOTH fit "
                         "drivers via --out and exported as "
                         "TT_TWO_SI_DEFENSES_DIR (default "
                         "<remote-root>/two_si_defenses/cells). "
                         "Commit-independent ON PURPOSE (D58): inside the "
                         "per-commit checkout, any code change would "
                         "silently void resume and refit every object")
    ap.add_argument("--private-root", default=None,
                    help="the grading host private root (default: the "
                         "site's SISTEER_JAILBREAK_PRIVATE_ROOT, else "
                         "$STATE/sisteer-jailbreak)")
    ap.add_argument("--generate-name", default="tt-twosi-posfull",
                    help="job name of the generate-pos cluster job to wait on")
    ap.add_argument("--generate-job-id", default=None,
                    help="explicit job id (overrides --generate-name discovery)")
    ap.add_argument("--generate-script", default=None,
                    help="CLUSTER-side sbatch script path for the generate "
                         "job, enabling resubmission if it dies with the "
                         "busy-GPU signature; without it a generate failure "
                         "is terminal")
    ap.add_argument("--run-tag", default=time.strftime("%Y%m%d"),
                    help="session tag baked into every tt-twosi-* job name; "
                         "re-running with the SAME tag resumes (adopts "
                         "completed/running jobs), a new tag is a new session")
    ap.add_argument("--deadline-hours", type=float, default=10.0,
                    help="hard wall-clock bound: after this the orchestrator "
                         "stops submitting/polling and exits with a summary "
                         "(running jobs are left running, never cancelled)")
    ap.add_argument("--poll-seconds", type=float, default=120.0)
    ap.add_argument("--max-attempts", type=int, default=5,
                    help="total submissions per job for RETRYABLE (busy-GPU/"
                         "NODE_FAIL) failures; real failures never retry")
    ap.add_argument("--busy-elapsed-max", type=float, default=600.0,
                    help="a FAILED job only counts as the busy-GPU leak if "
                         "it died within this many seconds (D51: the leak "
                         "kills in ~40s-2min) AND its log carries the "
                         "signature")
    ap.add_argument("--concurrency", type=int, default=200,
                    help="grade-pos live grader concurrency (Low-Prio). "
                         "Default 200, the repo's live-grader norm: the "
                         "grade stage prints the ready-to-post "
                         "coordination line (when the site names a channel) "
                         "and proceeds (print-and-proceed — never held on the "
                         "post; the grader backs off on the same org on "
                         "429s)")
    ap.add_argument("--fit-time", default="0-12:00:00",
                    help="--time for each fit job. A first-run guess (~2x "
                         "headroom target); right-size it from the first "
                         "completed fit's sacct Elapsed")
    ap.add_argument("--fit-arm-shards", type=int, default=1,
                    help="split each SI's Class A arms into N jobs (balanced "
                         "by the driver's static cost table) for more TOTAL "
                         "width than one node's GPUs — e.g. 2 shards x "
                         "data_parallel.gpus=8 = 16 GPUs per SI. Objects dir "
                         "and resume are shared; the shards' arm sets are "
                         "disjoint, so they never collide.")
    ap.add_argument("--fit-b-time", default="0-04:00:00",
                    help="--time for the single Class B fit job "
                         "(fit_class_b.py: antidote/alphasteer/caa_static_r "
                         "are closed-form; the 16k-prompt capture forwards "
                         "over the authors' lists dominate — caa_static_r "
                         "adds one more such pass, ten layers in one "
                         "forward). Matches the companion sbatch's "
                         "estimate; right-size it from the first completed "
                         "run's sacct Elapsed")
    ap.add_argument("--allow-native", action="store_true",
                    help="proceed to fitting even if the manifest regime is "
                         "native_per_si (default: stop — an unattended run "
                         "must not silently fit the confounded comparison)")
    ap.add_argument("--sha", default=None,
                    help="pinned commit for --comment=sha: and the in-script "
                         "guard (default: the cluster clone's EXPECTED_SHA)")
    ap.add_argument("--log-file", default=None,
                    help="structured JSONL progress log (default "
                         "<private-root>/two_si_overnight/<run-tag>.log.jsonl)")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    sys.exit(OvernightChain(args).run())


if __name__ == "__main__":
    main()
