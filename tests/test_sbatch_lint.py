"""Lint for the capeval-controls sbatch surface (extends the
existing sbatch-hygiene pattern to the new scripts).

Static text checks only — the cluster hard rules that have each caused real
multi-user outages (no ``--export=``, no ``--mem=0``, never ``dev`` in
sbatch, no manual ``CUDA_VISIBLE_DEVICES``), the lane conventions
(session-unique ``evals_*`` name, qos=high, append logs, stage2_common
scaffolding, preemption-tolerant resume), and the §6 own-checkout rule: an
``evals_*`` job must run from its OWN evals checkout under $SHARED via
S2_ROOT and must NEVER reference the shared repo checkout directly (other
live jobs own that one; repointing it would run wrong code under them).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SBATCH_DIR = REPO / "scripts" / "sbatch"

# The sbatch files THIS lane owns (extend as evals_* jobs are added), each
# with the OWN-checkout line it must carry: evals_budget runs from the seam
# lane's checkout (S2_ROOT); evals_cell_controls needs BOTH lanes' code, so
# it runs from the integration checkout via capeval_common.sh
# (CAPEVAL_REPO_DIR -> S2_REPO_DIR, EXPECTED_SHA inside the checkout).
EVALS_SBATCH = {
    "evals_budget.sbatch": (
        'S2_ROOT="${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-evals"',
        r'source "\$S2_ROOT/repo/scripts/sbatch/stage2_common\.sh"'),
    "evals_cell_controls.sbatch": (
        'CAPEVAL_REPO_DIR="${CAPEVAL_REPO_DIR:-${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-evals-repo}"',
        r'source "\$CAPEVAL_REPO_DIR/scripts/sbatch/capeval_common\.sh"'),
}


def _read(name: str) -> str:
    return (SBATCH_DIR / name).read_text()


def _hard_rules(txt: str) -> None:
    # the cluster hard rules (each has caused a real outage)
    assert not re.search(r"^#SBATCH --export", txt, re.M)     # rule 1
    assert not re.search(r"^#SBATCH --mem=0", txt, re.M)      # rule 2
    assert "CUDA_VISIBLE_DEVICES" not in txt                  # rule 3
    assert not re.search(r"^#SBATCH --qos=dev", txt, re.M)    # dev is srun-only


def test_evals_budget_sbatch_hygiene():
    txt = _read("evals_budget.sbatch")
    _hard_rules(txt)
    # header block
    assert re.search(r"^#SBATCH --job-name=evals_budget\b", txt, re.M)
    assert "#SBATCH --partition=general" in txt
    assert "#SBATCH --qos=high" in txt
    assert "#SBATCH --gres=gpu:1" in txt
    assert "#SBATCH --mem=128G" in txt
    assert re.search(r"^#SBATCH --time=0-06:00:00", txt, re.M)
    assert "#SBATCH --open-mode=append" in txt
    assert "#SBATCH --no-requeue" in txt
    assert "#SBATCH --signal=B:TERM@120" in txt
    # scaffolding: hf_env.sh + traps + figure mirror come from
    # stage2_common.sh; NCCL_NVLS_ENABLE kept explicit per the standing checklist
    assert "stage2_common.sh" in txt and "s2_init" in txt and "s2_run" in txt
    assert "NCCL_SOCKET_IFNAME=" not in txt   # the socket interface is the site's (its nccl.conf), never exported
    # preemption-tolerant resume: the fingerprinted trials JSONL on shared
    # storage, and the workload driven through the resumable `run` command
    assert "trials.jsonl" in txt
    assert "measure_train_budget.py run" in txt
    assert "$FIGS_DIR" in txt


def test_evals_cell_controls_sbatch_hygiene():
    txt = _read("evals_cell_controls.sbatch")
    _hard_rules(txt)
    # header block (§6: single task, 1xH200, ~4h x2 padding)
    assert re.search(r"^#SBATCH --job-name=evals_cell_controls\b", txt, re.M)
    assert "#SBATCH --partition=general" in txt
    assert "#SBATCH --qos=high" in txt
    assert "#SBATCH --gres=gpu:1" in txt
    assert "#SBATCH --mem=128G" in txt
    assert re.search(r"^#SBATCH --time=0-08:00:00", txt, re.M)
    assert "#SBATCH --open-mode=append" in txt
    assert "#SBATCH --no-requeue" in txt
    assert "#SBATCH --signal=B:TERM@120" in txt
    assert "s2_init" in txt and "s2_run" in txt
    assert "NCCL_SOCKET_IFNAME=" not in txt   # the socket interface is the site's (its nccl.conf), never exported
    # the §4.1 equivalence GATE runs BEFORE any generation (r1 #2) — assert
    # the ORDER in the script body, not just presence
    gate = txt.index("test_cell_controls_integration.py")
    generate = txt.index("cell_controls_run.py generate")
    assert gate < generate
    assert "SISTEER_EQUIV_CELL" in txt and "SISTEER_EQUIV_DEFENSES_DIR" in txt
    # the default cells: the real compiled object + the prompted
    # coverage cell, iterated in ONE job
    assert ("CELLS:-safety_refusal__vector_all "
            "prompted__safety_refusal") in txt
    # the handed-over defenses dir travels via the driver's --defenses-dir
    assert "--defenses-dir" in txt
    assert "exp/jailbreak/two_si_defenses/cells" in txt
    # the store root lives on SHARED storage, never inside the checkout
    assert 'OUT_ROOT="${OUT_ROOT:-$SHARED/exp/training/' in txt
    # smoke seam
    assert "LIMIT" in txt


def test_evals_sbatch_never_touch_the_shared_checkout():
    """§6: evals_* jobs run from their OWN checkout — evals_budget via
    S2_ROOT, evals_cell_controls via CAPEVAL_REPO_DIR/capeval_common.sh.
    Referencing the shared repo checkout would mean repointing the checkout
    other live jobs run from."""
    for name, (checkout_line, source_re) in EVALS_SBATCH.items():
        txt = _read(name)
        assert checkout_line in txt, name
        assert re.search(source_re, txt), name
        assert not re.search(r"\$\{?SISTEER_PROJECT_ROOT(?::\?[^}]*)?\}?/repo", txt), name
        assert not re.search(r"\$\{?SISTEER_PROJECT_ROOT(?::\?[^}]*)?\}?/EXPECTED_SHA", txt), name


# --------------------------------------------------------------------------- #
# the shared root (cluster A vs cluster B)                                     #
# --------------------------------------------------------------------------- #

# Every file that used to hardcode a single shared root now names it one of
# two ways, and the test reads which from the file itself:
#   * the site form: SHARED is the job environment's SISTEER_SHARED_ROOT,
#     which the front door exports from the cluster's site block; a job
#     without it refuses by name (there is no built-in root);
#   * a per-account probe over the cluster roots the file lists
#     (`if [ -d "<root>/$USER" ]; then ... elif ...; fi`), never the bare
#     mountpoint — cluster B compute nodes carry an empty root-owned stub at
#     cluster A's shared root path (verified 2026-08-20, job 22450), so a
#     bare `[ -d ]` test on it misdetects them as cluster A and every derived
#     path lands on an unwritable root. A probing wrapper is submit-side (it
#     derives QOS and fails loudly off-cluster); a probing job file falls back
#     to the first root off-cluster.
# The roots a probe lists are read from its block and repointed at tmp
# fakes, so this test names no site's storage.
# mamba_build.sbatch and its wrapper joined on 2026-10-12, when a review
# found the once-per-cluster Mamba-kernel cache warm-up hardcoding
# cluster A's shared root, so on cluster B it looked for the checkout,
# hf_env.sh and the uv cache under a root that does not exist.
DETECTED_FILES = [
    "stage2_common.sh",
    "capeval_common.sh",
    "evals_cell_controls.sbatch",
    "submit_evals_cell_controls.sh",
    "mamba_build.sbatch",
    "submit_mamba_build.sh",
]
_SITE_FORM = 'SHARED="${SISTEER_SHARED_ROOT:?'
_PROBE_RE = re.compile(
    r'^(?P<i>[ \t]*)if \[ -d "/[^"$]+/\$USER" \]; then\n'
    r".*?^(?P=i)fi$", re.M | re.S)
_PROBED_ROOT_RE = re.compile(r'\[ -d "(/[^"$]+)/\$USER" \]')
_QOS_RE = re.compile(r'(?:S2_CLUSTER_QOS|QOS)="(?:\$\{QOS:-)?([A-Za-z0-9_-]+)\}?"')


def _probe_block(txt: str):
    """The file's per-account probe block and the roots it lists, in order; None when the file has none."""
    m = _PROBE_RE.search(txt)
    return (m.group(0), _PROBED_ROOT_RE.findall(m.group(0))) if m else None


def _site_block(txt: str) -> str:
    """The site form: the SISTEER_SHARED_ROOT line, with the QoS assignments (QOS= / S2_CLUSTER_QOS=) right after it."""
    lines = txt[txt.rfind("\n", 0, txt.index(_SITE_FORM)) + 1:].split("\n")
    block = [lines[0]]
    for line in lines[1:]:
        if not re.match(r"\s*(?:S2_CLUSTER_QOS|QOS)=", line):
            break
        block.append(line)
    return "\n".join(block) + "\n"


def _run_block(block: str, env_extra: dict) -> tuple:
    script = (block + "\n"
              'echo "RESOLVED_SHARED=${SHARED:-}"\n'
              'echo "RESOLVED_QOS=${S2_CLUSTER_QOS:-${QOS:-}}"\n')
    env = {k: v for k, v in os.environ.items() if k not in ("SISTEER_SHARED_ROOT", "SISTEER_CLUSTER_QOS", "QOS")}
    env.update(USER="TESTUSER", **env_extra)
    proc = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=30)
    got = dict(line.split("=", 1) for line in proc.stdout.splitlines()
               if line.startswith("RESOLVED_"))
    return proc.returncode, got, proc.stderr


def _run_detection(block: str, roots: list, tmp_path: Path, present: list) -> tuple:
    """RUN the file's ACTUAL probe block with each listed root repointed at tmp_path/r<i> — asserting resolution, not
    just presence. ``present`` names what exists: (i, True) the root's per-user dir, (i, False) the bare root alone.
    Returns (returncode, {var: value})."""
    fakes = [tmp_path / f"r{i}" for i in range(len(roots))]
    # longest root first, through placeholders, so a root that prefixes another is never rewritten inside it
    for i in sorted(range(len(roots)), key=lambda i: -len(roots[i])):
        block = block.replace(roots[i] + "/", f"@@ROOT{i}@@/")
    for i, fake in enumerate(fakes):
        block = block.replace(f"@@ROOT{i}@@/", f"{fake}/")
    tmp_path.mkdir(parents=True, exist_ok=True)   # the no-mount case
    for i, with_user in present:
        (fakes[i] / "TESTUSER" if with_user else fakes[i]).mkdir(parents=True, exist_ok=True)
    rc, got, _ = _run_block(block, {})
    return rc, got, fakes


@pytest.mark.parametrize("name", DETECTED_FILES)
def test_shared_root_resolves_on_both_clusters(name, tmp_path):
    txt = _read(name)
    probe = _probe_block(txt)
    if probe is None:
        # the site form: the job environment's root, else a refusal that names the setting
        assert _SITE_FORM in txt, f"{name}: neither the site form nor a per-account probe"
        block = _site_block(txt)
        site = {"SISTEER_SHARED_ROOT": str(tmp_path / "site-root")}
        if "SISTEER_CLUSTER_QOS" in block:
            # the cluster's QoS is the job environment's too; QOS= in the body env overrides it
            site["SISTEER_CLUSTER_QOS"] = "site-qos"
        rc, got, _ = _run_block(block, site)
        assert rc == 0 and got["RESOLVED_SHARED"] == str(tmp_path / "site-root")
        if "SISTEER_CLUSTER_QOS" in block:
            assert got["RESOLVED_QOS"] == "site-qos"
            rc, got, _ = _run_block(block, {**site, "QOS": "body-qos"})
            assert rc == 0 and got["RESOLVED_QOS"] == "body-qos"
        rc, got, err = _run_block(block, {})
        assert rc != 0 and "SISTEER_SHARED_ROOT" in err
        assert "RESOLVED_SHARED" not in got
        return
    block, roots = probe
    assert len(roots) >= 2, (name, roots)
    qos = _QOS_RE.findall(block)
    sets_qos = len(qos) >= 2
    if sets_qos:
        assert qos[0] != qos[1], qos       # the QoS namespace differs per cluster
    # cluster A: the per-user dir under its shared root exists (and wins
    # even with cluster B's root and its user dir present too)
    rc, got, fakes = _run_detection(block, roots, tmp_path / "a", [(0, True), (1, True)])
    assert rc == 0
    assert got["RESOLVED_SHARED"] == str(fakes[0] / "TESTUSER")
    if sets_qos:
        assert got["RESOLVED_QOS"] == qos[0]
    # cluster B login node: only cluster B's root (with the user dir) exists
    rc, got, fakes = _run_detection(block, roots, tmp_path / "b", [(1, True)])
    assert rc == 0
    assert got["RESOLVED_SHARED"] == str(fakes[1] / "TESTUSER")
    if sets_qos:
        assert got["RESOLVED_QOS"] == qos[1]
    # cluster B COMPUTE node: a bare mountpoint stub for cluster A's shared
    # root with NO user dir under it (the 2026-08-20 job-22450 failure) —
    # the per-user probe must still resolve cluster B, never the stub.
    rc, got, fakes = _run_detection(block, roots, tmp_path / "stub", [(0, False), (1, True)])
    assert rc == 0
    assert got["RESOLVED_SHARED"] == str(fakes[1] / "TESTUSER")
    if sets_qos:
        assert got["RESOLVED_QOS"] == qos[1]
    # Off-cluster (neither user dir): a submit wrapper fails LOUDLY (it is
    # about to sbatch); a probing job file keeps the first root.
    rc, got, fakes = _run_detection(block, roots, tmp_path / "none", [])
    if name.startswith("submit_"):
        assert rc != 0
    else:
        assert rc == 0
        assert got["RESOLVED_SHARED"] == str(fakes[0] / "TESTUSER")


def test_stage2_common_qos_and_result_are_cluster_ready():
    """The cluster-B port pins on stage2_common.sh: the shared root is the
    site's (SISTEER_SHARED_ROOT, never guessed from which storage paths
    exist) and the self-escalation QoS is the job's own ESCALATE_QOS, never
    derived (not from a storage probe, so a cluster B preemption never
    resubmits at a QoS that exists only on cluster A, and not from the QoS
    the job was submitted at), the
    resubmit carries the predecessor's RESOLVED log path (most headers
    hardcode an --output under cluster A's root whose parent does not
    exist on cluster B — and Slurm never creates log parents, so that
    resubmit would never start), and s2_finish writes the one-line RESULT
    sentinel next to the figures before mirroring them."""
    txt = _read("stage2_common.sh")
    assert _SITE_FORM in txt and _probe_block(txt) is None
    assert 'ESCALATE_QOS="${ESCALATE_QOS:?' in txt and "SISTEER_CLUSTER_QOS" not in txt
    escalate = txt.split("s2_escalate()", 1)[1].split("\ns2_", 1)[0]
    assert "StdOut=" in escalate and '--output="$stdout_path"' in escalate
    finish = txt.split("s2_finish()", 1)[1].split("\ns2_", 1)[0]
    assert '"$FIGS_DIR/RESULT"' in finish
    for field in ("exit=$rc", "elapsed_s=$SECONDS",
                  "job=${SLURM_JOB_ID:-unknown}", "node=${SLURMD_NODENAME:-"):
        assert field in finish, field
    # the sentinel + mirror are best-effort: no bare `exit` in the trap
    active = [l.strip() for l in finish.splitlines()
              if l.strip() and not l.strip().startswith("#")]
    assert not any(l.startswith("exit") for l in active)


def test_s2_init_corrects_hf_env_misdetection_on_cluster_b(tmp_path):
    """The INSTALLED hf_env.sh probes the bare mountpoint of cluster A's shared
    storage, so on a cluster B compute node (the mountpoint stub — see the cluster
    detection tests above) it exports cluster A's HF paths and every model
    load dies offline. s2_init must re-point HF at the per-user cluster B
    cache whenever its own per-user probe resolved cluster B. RUN the
    correction block: a cluster-A-shaped HF_HOME under a cluster B SHARED
    is rewritten; a cluster A SHARED leaves it untouched."""
    txt = _read("stage2_common.sh")
    start = txt.index('source "$SHARED/scripts/hf_env.sh"')
    block = txt[start:txt.index("export HF_HUB_OFFLINE=1")]
    block = "\n".join(l for l in block.splitlines()
                      if not l.lstrip().startswith(("source ", "#")))
    run = lambda shared, hf: subprocess.run(  # noqa: E731
        ["bash", "-c",
         f'SHARED="{shared}"; export HF_HOME="{hf}"; {block}\n'
         'echo "GOT=$HF_HOME"'],
        capture_output=True, text=True,
        env=dict(os.environ, USER="TESTUSER"), timeout=30)
    # Cluster B SHARED (the per-user root the block compares with) + a
    # misdetected cluster A HF_HOME -> corrected
    other_hf = "/shared-a/hf_cache"            # any HF_HOME but cluster B's own cache
    proc = run("/workspace/TESTUSER", other_hf)
    assert proc.returncode == 0
    assert "GOT=/workspace/TESTUSER/hf_cache" in proc.stdout
    # Cluster A SHARED -> untouched
    proc = run("/shared-a/TESTUSER", other_hf)
    assert proc.returncode == 0
    assert f"GOT={other_hf}" in proc.stdout


def test_submit_wrapper_creates_log_parent_and_overrides_header():
    """The submit wrapper exists because #SBATCH headers cannot branch per
    cluster: it derives the QoS, creates the log dir BEFORE sbatch (Slurm
    never creates log parents; a missing one means the job never starts),
    and overrides --output — and it must never smuggle in the
    cluster-killer flags."""
    txt = _read("submit_evals_cell_controls.sh")
    active = [l for l in txt.splitlines()
              if l.strip() and not l.lstrip().startswith("#")]
    assert not any("--export" in l for l in active)           # hard rule 1
    assert not any("--mem" in l for l in active)              # header owns memory
    assert 'mkdir -p "$LOGDIR"' in txt
    assert txt.index('mkdir -p "$LOGDIR"') < txt.index("exec sbatch")
    assert '--qos="$QOS"' in txt
    assert '--output="$LOGDIR/%x_%j.out"' in txt
    assert "evals_cell_controls.sbatch" in txt
    # the QoS: one default per probed cluster (they differ), or the site's
    defaults = re.findall(r'^\s*QOS="\$\{QOS:-(.*)\}"\s*$', txt, re.M)
    if _probe_block(txt) is not None:
        assert len(set(defaults)) >= 2, defaults
    else:
        assert any("SISTEER_" in d for d in defaults), defaults


def test_evals_cell_controls_paths_are_shared_root_relative():
    """No body path may hardcode a per-user shared root: everything the job
    touches goes through $SHARED, the job environment's SISTEER_SHARED_ROOT
    (the header --output is the documented exception — headers cannot
    branch, the submit wrapper overrides it). /home/$USER is node-local
    disk (the venvs), not a shared root."""
    txt = _read("evals_cell_controls.sbatch")
    body = [l for l in txt.splitlines()
            if not l.lstrip().startswith("#") and l.strip()]
    per_user_root = re.compile(r'(?<![\w$}])/(?!home/)[^\s"/$]+(?:/[^\s"/$]+)*/(?:\$USER|\$\{USER\}|%u)\b')
    offenders = [l for l in body if per_user_root.search(l)]
    assert offenders == [], offenders
    assert _SITE_FORM in txt
    assert 'DEFENSES_DIR="${DEFENSES_DIR:-$SHARED/exp/jailbreak/' in txt


def test_stage2_common_s2_root_is_default_preserving():
    """The S2_ROOT seam must keep the existing scripts byte-identical in
    behavior: the default expands to the historical checkout under $SHARED,
    and no code path (comments aside) hardcodes the old root anymore."""
    txt = _read("stage2_common.sh")
    # the UNIFIED seam: S2_ROOT primary, S2_REPO_DIR derived (capeval §3.3
    # keeps overriding it for a checkout not named `repo`)
    assert ('S2_ROOT="${S2_ROOT:-${S2_REPO_ROOT:-${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}}}"'
            in txt)  # S2_REPO_ROOT honored as the same seam's alias
    assert 'S2_REPO_ROOT="${S2_REPO_ROOT:-$S2_ROOT}"' in txt
    assert 'S2_REPO_DIR="${S2_REPO_DIR:-$S2_ROOT/repo}"' in txt
    # every functional reference goes through the seam now
    code_lines = [ln for ln in txt.splitlines()
                  if not ln.lstrip().startswith("#")]
    offenders = [ln for ln in code_lines if "SISTEER_PROJECT_ROOT" in ln
                 and "S2_ROOT:-" not in ln]
    assert offenders == [], offenders
    # the uv env: explicit S2_UV_ENV wins, else root-named — different roots
    # can never share a venv, and nothing set = the historical name
    assert ('S2_UV_ENV="${S2_UV_ENV:-/home/$USER/venvs/'
            '$(basename "$S2_ROOT")}"') in txt
    assert 'export UV_PROJECT_ENVIRONMENT="$S2_UV_ENV"' in txt
    # the SHA gate + cd go through S2_REPO_DIR (root-derived by default;
    # <checkout>/EXPECTED_SHA when the checkout is not named `repo`)
    assert '"${S2_REPO_DIR%/repo}/EXPECTED_SHA"' in txt
    assert 'git -C "$S2_REPO_DIR" rev-parse HEAD' in txt
    assert 'cd "$S2_REPO_DIR"' in txt
    assert 'WANDB_DIR="$S2_ROOT/wandb"' in txt
