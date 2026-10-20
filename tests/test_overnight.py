"""two_si_overnight.py — the bounded overnight orchestrator, unit-tested
WITHOUT a cluster: every ssh/sbatch/sacct/rsync/local-pipeline call goes
through one injectable runner, faked here by a tiny scripted World. CPU-only,
offline, synthetic strings only (TESTHOST, invented job names/ids).

What the suite proves — the properties that make the script safe at 3am:

- every sbatch carries the required flags, `--exclude` always includes the
  known leak nodes, and the exclusion list is LOGGED on every submission;
- the busy-GPU failure (short elapsed + the CUDA-busy/AcceleratorError
  signature) is retried with the failing node added to the exclusion, while a
  REAL code failure is terminal after ONE submission — never retried;
- retries are bounded by --max-attempts;
- the wall-clock deadline ends every polling loop: the orchestrator exits
  with a summary and leaves running jobs running (no scancel, ever);
- re-entry adopts an already-COMPLETED fit job instead of resubmitting;
- the native_per_si regime STOPS the run before any Class A fit;
- the Class B fit (fit_class_b.py, SI-agnostic) is submitted AFTER the Class A
  fits through the SAME generated-script mechanism — TT_TWO_SI_STORE_DIR
  exported in the script BODY, no sbatch --export anywhere, no
  stage2_common.sh — and NEITHER class's failure skips the other (they share
  no inputs);
- BOTH rendered fit scripts point their driver at the ONE absolute,
  commit-independent objects dir (--out + a body-level TT_TWO_SI_DEFENSES_DIR
  export), never a path inside the per-commit checkouts/ tree;
- every sacct query — by id and by name — is scoped to this user and a
  submit window (-u/-S, with -X), so a RECYCLED job id's stale row (another
  user, months old) can never be adopted as this run's verdict;
- the full chain runs the stages in order and the structured log records it;
- the pid lockfile refuses a concurrent second instance;
- nothing destructive (scancel / scontrol update / drain) is EVER issued.
"""

from __future__ import annotations

import json
import types
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import two_si_overnight as ov  # noqa: E402

SIS = ("safety_refusal", "si_constitution")
SHA = "a" * 40
HOST = "TESTHOST"
RROOT = "/shared/TESTUSER/exp/jailbreak"
RPROJECT = "/shared/TESTUSER/proj"
# The derived default — absolute, under the private remote root, never
# inside the per-commit checkouts/ tree.
OBJDIR = f"{RROOT}/two_si_defenses/cells"


@pytest.fixture(autouse=True)
def synthetic_site(tmp_path, monkeypatch):
    """The site config the orchestrator reads its cluster facts from: cluster A
    (shared root /shared/TESTUSER, QoS high, two GPU-leak nodes excluded) and
    cluster B (/workspace/TESTUSER, QoS high-b, no exclusions). Synthetic
    values only; a test that writes its own site sets SISTEER_CONFIG again."""
    site = tmp_path / "synthetic-site"
    (site / "clusters").mkdir(parents=True)
    (site / "config.env").write_text('SISTEER_CLUSTERS="one two"\nSISTEER_PRIMARY_CLUSTER=one\n')
    for label, ssh, root, qos, extra in (("one", "login-0", "/shared/TESTUSER", "high", "--exclude=compute-16,compute-18"),
                                         ("two", "login-1", "/workspace/TESTUSER", "high-b", "")):
        (site / "clusters" / f"{label}.env").write_text(
            f"CLUSTER_SSH={ssh}\nCLUSTER_SHARED_ROOT={root}\nCLUSTER_PARTITION=p\nCLUSTER_QOS={qos}\n"
            f"CLUSTER_SUBSTRATE=slurm\nCLUSTER_LABEL={label}\nCLUSTER_SBATCH_EXTRA='{extra}'\n")
    monkeypatch.setenv("SISTEER_CONFIG", str(site / "config.env"))
    for k in ("SISTEER_SHARED_ROOT", "CLUSTER_SHARED_ROOT", "SISTEER_CLUSTER_LABEL", "SISTEER_WANDB_ENTITY", "SISTEER_WANDB_PROJECT"):
        monkeypatch.delenv(k, raising=False)
    return site


def cp(argv, rc=0, out="", err=""):
    return subprocess.CompletedProcess(argv, rc, out, err)


def contains(*subs):
    def pred(argv):
        joined = " ".join(map(str, argv))
        return all(s in joined for s in subs)
    return pred


class FakeRunner:
    """Scripted stand-in for the one outside-world seam. Unhandled commands
    FAIL the test — the orchestrator may only do what the scenario scripted."""

    def __init__(self):
        self.calls: list[list[str]] = []
        self.handlers: list[tuple] = []

    def add(self, pred, responder):
        self.handlers.append((pred, responder))

    def add_local_head(self, sha):
        """The pinned sha now comes from THIS worktree's HEAD (not the cluster
        clone's EXPECTED_SHA), so every fixture must answer that git call."""
        self.add(lambda a: a[:1] == ["git"] and "rev-parse" in a,
                 lambda a: types.SimpleNamespace(returncode=0, stdout=sha + "\n", stderr=""))

    def __call__(self, argv, timeout=None):
        argv = list(map(str, argv))
        self.calls.append(argv)
        for pred, responder in self.handlers:
            if pred(argv):
                return responder(argv)
        raise AssertionError(f"unhandled command: {' '.join(argv)}")

    def joined_calls(self):
        return [" ".join(c) for c in self.calls]


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


BUSY_TAIL = ("torch.AcceleratorError: CUDA error: CUDA-capable device(s) "
             "is/are busy or unavailable")
REAL_TAIL = 'Traceback (most recent call last):\n  ...\nKeyError: "pos"'


class World:
    """A tiny fake cluster + local pipeline. Job fates per name are scripted:
    each sbatch submission for a name pops the next fate."""

    def __init__(self, runner, priv, manifest_path, *,
                 regime="matched_intersection", generate=None,
                 fit_fates=None, preexisting=(), n_ungraded=0, recycled=None):
        self.runner = runner
        self.priv = priv
        self.manifest_path = manifest_path
        self.regime = regime
        self.n_ungraded = n_ungraded
        self.recycled = dict(recycled or {})   # jid -> a stale foreign row
        self.fit_fates = {k: list(v) for k, v in (fit_fates or {}).items()}
        self.jobs: dict[str, dict] = {}
        self.next_id = 1000
        self.busy_node_n = 6
        self.sbatch_cmds: list[str] = []

        if generate is not None:
            self.jobs["900"] = dict(name="tt-twosi-posfull", **generate)
        for row in preexisting:
            self.jobs[row["job_id"]] = {k: v for k, v in row.items()
                                        if k != "job_id"}

        r = runner
        r.add(contains("install -d -m 700"), lambda a: cp(a))
        # the pinned sha now comes from THIS worktree's HEAD
        r.add(lambda a: a[:1] == ["git"] and "rev-parse" in a,
              lambda a: cp(a, out=SHA + "\n"))
        r.add(contains("sacct", "-j "), self._sacct_by_id)
        r.add(contains("sacct", "--name="), self._sacct_by_name)
        # "sbatch --" (not "sbatch --exclude"): a cluster B-target run
        # deliberately submits with NO --exclude (the known leak nodes are
        # cluster A names), and rsync argv mentioning "<name>.sbatch"
        # must not match.
        r.add(contains("sbatch --"), self._sbatch)
        r.add(contains("tail -c"), self._tail)
        r.add(lambda a: a[0] == "rsync", lambda a: cp(a))
        r.add(contains("build_two_si_pool.py", "grade-pos"), lambda a: cp(a))
        r.add(contains("build_two_si_pool.py", "manifest"), self._manifest)
        r.add(contains("build_two_si_pool.py", "emit-pairs"), self._emit)

    # -- slurm ----------------------------------------------------------------

    def _row(self, jid, spec):
        return (f"{jid}|{spec['name']}|{spec['state']}|{spec['elapsed']}|"
                f"{spec.get('exit', '1:0')}|{spec['node']}")

    def _sacct_by_id(self, argv):
        joined = " ".join(argv)
        jid = re.search(r"-j (\d+)", joined).group(1)
        spec = self.jobs.get(jid)
        if spec:
            return cp(argv, out=self._row(jid, spec) + "\n")
        # a RECYCLED id (an unrelated months-old job): real sacct only
        # returns it to an UNSCOPED query — `-u "$USER" -S <window>` filters
        # it out, which is exactly what the orchestrator must rely on.
        row = self.recycled.get(jid)
        if row and not ('-u "$USER"' in joined and "-S " in joined):
            return cp(argv, out=row + "\n")
        return cp(argv, out="")

    def _sacct_by_name(self, argv):
        name = re.search(r"--name=(\S+)", " ".join(argv)).group(1)
        rows = [self._row(jid, spec) for jid, spec in self.jobs.items()
                if spec["name"] == name]
        return cp(argv, out="\n".join(rows) + ("\n" if rows else ""))

    def _sbatch(self, argv):
        joined = " ".join(argv)
        self.sbatch_cmds.append(joined)
        name = re.search(r"--job-name=(\S+)", joined).group(1)
        self.next_id += 1
        jid = str(self.next_id)
        fates = self.fit_fates.get(name)
        fate = fates.pop(0) if fates else "COMPLETED"
        if fate == "COMPLETED":
            spec = dict(state="COMPLETED", elapsed="01:00:00", exit="0:0",
                        node="compute-31", tail="")
        elif fate == "BUSY":
            self.busy_node_n += 1
            spec = dict(state="FAILED", elapsed="00:01:30",
                        node=f"compute-{self.busy_node_n}", tail=BUSY_TAIL)
        elif fate == "BUSY_LONG":
            spec = dict(state="FAILED", elapsed="02:00:00", node="compute-31",
                        tail=BUSY_TAIL)
        elif fate == "REALFAIL":
            spec = dict(state="FAILED", elapsed="00:01:00", node="compute-31",
                        tail=REAL_TAIL)
        elif fate == "NODE_FAIL":
            spec = dict(state="NODE_FAIL", elapsed="00:00:50", node="compute-16",
                        tail="")
        else:  # pragma: no cover - scenario bug
            raise AssertionError(f"unknown fate {fate}")
        self.jobs[jid] = dict(name=name, **spec)
        return cp(argv, out=f"Submitted batch job {jid}\n")

    def _tail(self, argv):
        m = re.search(r"_(\d+)\.err", " ".join(argv))
        spec = self.jobs.get(m.group(1)) if m else None
        return cp(argv, out=spec.get("tail", "") if spec else "")

    # -- local pipeline ---------------------------------------------------------

    def _manifest(self, argv):
        m = {"regime": self.regime,
             "pool_sizes": {si: 2 for si in SIS},
             "coverage": {si: {"n_missing": 0} for si in SIS},
             "manifest_sha256": "ab" * 32}
        self.manifest_path.write_text(json.dumps(m))
        return cp(argv, out=f"[two-si-pool] REGIME: {self.regime}\n")

    def _emit(self, argv):
        d = self.priv / "two_si_pairs"
        d.mkdir(parents=True, exist_ok=True)
        for si in SIS:
            (d / f"pairs_{si}.jsonl").write_text(
                '{"prompt": "SYNTHETIC p", "pos": "SYNTHETIC y", '
                '"neg": "SYNTHETIC n"}\n')
        (d / "pairs_meta.json").write_text(json.dumps({
            "regime": self.regime, "pool_sizes": {si: 2 for si in SIS},
            "n_ungraded_pos_skipped": self.n_ungraded}))
        return cp(argv, out="[emit-pairs] REGIME\n")


GEN_OK = dict(state="COMPLETED", elapsed="01:00:00", exit="0:0",
              node="compute-31", tail="")
GEN_RUNNING = dict(state="RUNNING", elapsed="00:10:00", exit="0:0",
                   node="compute-31", tail="")
GEN_BUSY = dict(state="FAILED", elapsed="00:01:40", exit="1:0",
                node="compute-9", tail=BUSY_TAIL)


def make_orch(tmp_path, *, extra_args=(), world_kw=None, deadline_hours=10.0):
    tmp_path = Path(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = {"si_ids": list(SIS),
           "pool": {"constitution_run": "two_si_pos_constitution"},
           "pool_manifest": str(tmp_path / "pool_manifest.json")}
    cfg_path = tmp_path / "two_si_test.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    priv = tmp_path / "priv"

    args = ov.build_parser().parse_args([
        "--config", str(cfg_path),
        "--private-root", str(priv),
        "--cluster-host", HOST,
        "--remote-root", RROOT,
        "--remote-project", RPROJECT,
        "--run-tag", "t1",
        "--poll-seconds", "30",
        "--deadline-hours", str(deadline_hours),
        "--sha", SHA,
        *extra_args,
    ])
    runner = FakeRunner()
    world = World(runner, priv, tmp_path / "pool_manifest.json",
                  **(world_kw or {"generate": GEN_OK}))
    clock = FakeClock()
    orch = ov.OvernightChain(args, runner=runner, clock=clock, sleep=clock.sleep)
    return orch, runner, world, clock


def read_log(orch):
    return [json.loads(l) for l in orch.log_path.read_text().splitlines()]


def assert_nothing_destructive(runner):
    """The standing rule: record + exclude, NEVER kill or drain."""
    for joined in runner.joined_calls():
        assert "scancel" not in joined
        assert "scontrol update" not in joined
        assert "drain" not in joined.lower()


# --------------------------------------------------------------------------- #
# the rendered fit sbatch script                                                #
# --------------------------------------------------------------------------- #


def test_generated_fit_scripts_carry_the_project_directory_their_bare_sbatch_needs(tmp_path, monkeypatch):
    """The orchestrator submits its fit jobs with a bare sbatch, outside the front door's job env, so the project
    directory the pinned checkout lives under is rendered into the script; it refuses up front when it has none."""
    kw = dict(name="n", config="c.yaml", time_limit="1:00:00", logdir=f"{RROOT}/_logs", store_dir=f"{RROOT}/s",
              objects_dir=OBJDIR, sha=SHA, shared_root="/shared/TESTUSER")
    with_root = ov.fit_class_b_sbatch_script(**kw, project_root=RPROJECT)
    assert f"\nexport SISTEER_PROJECT_ROOT={RPROJECT}   # the project directory (checkouts/<sha>)\n" in with_root
    assert with_root.index("export SISTEER_PROJECT_ROOT=") < with_root.index('REPO="${SISTEER_PROJECT_ROOT:?')
    assert "export SISTEER_PROJECT_ROOT" not in ov.fit_class_b_sbatch_script(**kw)
    # the site block whose shared root is the run's names the directory; anything else refuses by name
    site = tmp_path / "site"; (site / "clusters").mkdir(parents=True)
    (site / "config.env").write_text('SISTEER_CLUSTERS="one"\nSISTEER_PRIMARY_CLUSTER=one\n')
    (site / "clusters" / "one.env").write_text("CLUSTER_SSH=login-one\nCLUSTER_SHARED_ROOT=/shared/one\nCLUSTER_PARTITION=p\n"
                                               "CLUSTER_QOS=q\nCLUSTER_SUBSTRATE=slurm\nCLUSTER_LABEL=one\n"
                                               "CLUSTER_PROJECT_ROOT='$SHARED/proj'\n")
    monkeypatch.setenv("SISTEER_CONFIG", str(site / "config.env"))
    assert ov.resolve_project_root("/shared/one/", None) == "/shared/one/proj"
    assert ov.resolve_project_root("/shared/one", "/elsewhere/p/") == "/elsewhere/p"
    with pytest.raises(ov.StageFailed, match="--remote-project"):
        ov.resolve_project_root("/shared/two", None)


def test_the_project_directory_of_a_cluster_the_site_does_not_fan_out_to_comes_from_its_block_file(tmp_path, monkeypatch):
    """A site that submits to one cluster by default (SISTEER_CLUSTERS="one") can still describe a second in
    clusters/two.env; a run on the second cluster's shared root takes its project directory from that block. A shared root
    no block carries still refuses by name."""
    site = tmp_path / "site"; (site / "clusters").mkdir(parents=True)
    (site / "config.env").write_text('SISTEER_CLUSTERS="one"\nSISTEER_PRIMARY_CLUSTER=one\n')
    for label, root, proj in (("one", "/shared/one", "proj"), ("two", "/shared/two", "proj-two")):
        (site / "clusters" / f"{label}.env").write_text(
            f"CLUSTER_SSH=login-{label}\nCLUSTER_SHARED_ROOT={root}\nCLUSTER_PARTITION=p\nCLUSTER_QOS=q\n"
            f"CLUSTER_SUBSTRATE=slurm\nCLUSTER_LABEL={label}\nCLUSTER_PROJECT_ROOT='$SHARED/{proj}'\n")
    monkeypatch.setenv("SISTEER_CONFIG", str(site / "config.env"))
    assert ov.resolve_project_root("/shared/two/", None) == "/shared/two/proj-two"
    assert ov.resolve_project_root("/shared/one", None) == "/shared/one/proj"
    with pytest.raises(ov.StageFailed, match="no project directory for the shared root /shared/three"):
        ov.resolve_project_root("/shared/three", None)


def _rendered_wandb(text: str, env: dict) -> str:
    """The W&B lines of a rendered fit script run under `set -eu`: the entity and project the job's fit would log under."""
    lines = [ln for ln in text.splitlines() if "WANDB" in ln and not ln.lstrip().startswith("#")]
    r = subprocess.run(["bash", "-euc", "\n".join(lines) + '\necho "ENTITY=${WANDB_ENTITY:-} PROJECT=${SISTEER_WANDB_PROJECT:-}"'],
                       capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", **env}, timeout=30)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_generated_fit_scripts_carry_the_site_wandb_destination_the_front_door_job_env_carries(tmp_path, monkeypatch):
    """The fit jobs go out with a bare sbatch, so the site's W&B destination (lib/submit_twin.py site_wandb_env, the keys the
    front door's job env carries) is rendered into each script: the config's $WANDB_PROJECT expands on the node and the run is
    filed under the site's entity (without them the fit runs with W&B off). With no entity in the site nothing is rendered.
    Synthetic site and values; nothing contacts a cluster or W&B."""
    import shlex
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from front_door_testlib import LIB, write_site
    for n in ("SISTEER_WANDB_ENTITY", "SISTEER_WANDB_PROJECT", "WANDB_ENTITY", "WANDB_PROJECT"):
        monkeypatch.delenv(n, raising=False)
    site = write_site(tmp_path / "w", extra_config="SISTEER_WANDB_ENTITY=example-entity\nSISTEER_WANDB_PROJECT=example-project\n")
    monkeypatch.setenv("SISTEER_CONFIG", str(site / "config.env"))
    env = ov.resolve_wandb_env()
    # the front door's own job env for a cluster of that site names the same destination, key for key
    r = subprocess.run([sys.executable, str(LIB / "submit_twin.py"), "--config", str(site / "config.env"), "job-env", "--cluster",
                        "cluster_a"], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    front = dict(t.split("=", 1) for t in shlex.split(r.stdout))
    assert env == {k: front.get(k) for k in env} == {"SISTEER_WANDB_ENTITY": "example-entity", "SISTEER_WANDB_PROJECT": "example-project"}
    # the orchestrator reads it up front and renders it into the Class A and the Class B script alike
    orch, *_ = make_orch(tmp_path / "o", extra_args=["--qos", "high"])      # this site describes no cluster at the run's root
    assert orch.wandb_env == env
    orch._make_submitter = lambda name, script_remote, render: render
    texts = [orch._make_fit_submitter(SIS[0], "n-a", f"{RROOT}/n-a.sbatch")(), orch._make_fit_b_submitter("n-b", f"{RROOT}/n-b.sbatch")()]
    for text in texts:
        assert "export SISTEER_WANDB_ENTITY=example-entity\nexport SISTEER_WANDB_PROJECT=example-project\n" in text
        assert _rendered_wandb(text, {}) == "ENTITY=example-entity PROJECT=example-project"
        assert _rendered_wandb(text, {"WANDB_ENTITY": "e-env"}) == "ENTITY=e-env PROJECT=example-project"   # the job's own wins
    # a site with no entity: nothing resolved, nothing rendered, the script as before
    bare = write_site(tmp_path / "n", extra_config="SISTEER_WANDB_PROJECT=example-project\n")
    monkeypatch.setenv("SISTEER_CONFIG", str(bare / "config.env"))
    assert ov.resolve_wandb_env() == {}
    orch2, *_ = make_orch(tmp_path / "o2", extra_args=["--qos", "high"])
    orch2._make_submitter = lambda name, script_remote, render: render
    text = orch2._make_fit_submitter(SIS[0], "n-a", f"{RROOT}/n-a.sbatch")()
    assert "SISTEER_WANDB" not in text and "WANDB_ENTITY" not in text


def test_fit_sbatch_script_carries_every_required_flag():
    text = ov.fit_sbatch_script(
        name="tt-twosi-fit-safety_refusal-t1", si="safety_refusal",
        pairs_remote=f"{RROOT}/two_si_pairs/pairs_safety_refusal.jsonl",
        config="configs/two_si.yaml", time_limit="0-12:00:00",
        logdir=f"{RROOT}/_logs", store_dir=f"{RROOT}/two_si_store",
        objects_dir=OBJDIR, sha=SHA, shared_root="/shared/TESTUSER")
    for required in (
            "#SBATCH --job-name=tt-twosi-fit-safety_refusal-t1",
            "#SBATCH --partition=general",
            "#SBATCH --qos=high",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=16",
            "#SBATCH --mem=128G",
            "#SBATCH --time=0-12:00:00",
            f"#SBATCH --output={RROOT}/_logs/%x_%j.out",
            f"#SBATCH --error={RROOT}/_logs/%x_%j.err",
            "#SBATCH --open-mode=append",
            "#SBATCH --no-requeue",
            "#SBATCH --signal=B:TERM@120",
            # hf_env.sh + HF_HUB_OFFLINE=1 + uv sync --frozen + the
            # the proven canary env (hf_env + per-worktree venv), NOT s2_init,
            # plus this script's own pin guard against the wrong checkout:
            f"PINNED_SHA={SHA}",
            # run from the IMMUTABLE checkout, never the shared clone
            'REPO="${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}/checkouts/$PINNED_SHA"',
            "rev-parse HEAD",
            # metrics-only live W&B:
            # WANDB_DISABLED deliberately UNSET so the driver streams
            # scalars, console capture OFF (error paths print attack/pair
            # text) — plus the store override:
            "export WANDB_CONSOLE=off",
            # LIVE streaming needs a key ON THE NODE (cluster /home is
            # node-local): the optional per-user key file beside hf_token,
            # a no-op when absent (the driver then degrades loudly to
            # console + per-step jsonl):

            'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"',
            f"export TT_TWO_SI_STORE_DIR={RROOT}/two_si_store",
            "hf_env.sh",                      # the proven canary env
            "venvs/sisteer-attackgen",    # per-worktree venv
            # the commit-independent objects dir, via the driver's
            # --out AND the body-level env export readers resolve through:
            f"--out {OBJDIR}",
            f"export TT_TWO_SI_DEFENSES_DIR={OBJDIR}",
            # the actual work:
            "train_two_si_objects.py train",
            "--si safety_refusal",
            f"--pairs {RROOT}/two_si_pairs/pairs_safety_refusal.jsonl",
            # the unattended-window RESULT sentinel: an EXIT trap (separate
            # from the signal-trap cleanup) writes the one-line outcome next
            # to the figures and mirrors both:
            'export FIGS_DIR="$SHARED/exp/figs/tt-twosi-fit-safety_refusal-t1"',
            '> "$FIGS_DIR/RESULT" || true',
            'mirror-figs.sh" "$FIGS_DIR"',
            "trap result_sentinel EXIT",
            "trap cleanup SIGTERM SIGINT SIGQUIT",   # the cleanup trap survives
            # the committed lock in the job's flavor (torch lives only in the extras), with its own rc handling (no errexit here):
            'uv sync --frozen --extra cu128 --quiet || { echo "FATAL: uv sync --frozen --extra cu128 failed',
    ):
        assert required in text, f"missing from fit sbatch script: {required}"
    # the two cluster-killers can never appear (hard rules 1 and 2):
    assert "--export" not in text
    assert "--mem=0" not in text
    # no unfrozen fallback: `|| uv sync` re-resolves the lock on a job node (the sbatch flavor audit's rule; the audit itself
    # runs over the rendered text in test_rendered_fit_scripts_pass_the_sbatch_flavor_audit)
    assert "|| uv sync" not in text
    # the RESULT sentinel is armed before the first path that can exit the script (the pin guard, then the sync), so a job
    # that dies there still leaves its outcome line
    assert text.index("trap result_sentinel EXIT") < text.index('REPO="${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}/checkouts/$PINNED_SHA"') < text.index("uv sync --frozen")
    # live metrics: the fit body must NOT disable W&B (transcript jobs keep
    # their own WANDB_DISABLED=true containment; the driver streams numeric
    # scalars only, with console capture off — asserted present above)
    assert "export WANDB_DISABLED" not in text


def test_rendered_fit_scripts_pass_the_sbatch_flavor_audit(tmp_path):
    """The sbatch flavor audit (tests/test_torch_flavors.py: every active `uv sync` frozen + cu128 + rc-handled with no unfrozen
    fallback, every `uv run` --no-sync) reads scripts/sbatch/ and never this Python template — so its rules are applied to the
    RENDERED text of both fit classes here, through the audit's own lexer. The two mutations prove the coverage is live: a
    flavor-less sync and an unfrozen fallback must each be flagged."""
    from test_torch_flavors import _flavor_violations
    rendered = tmp_path / "rendered"; rendered.mkdir()
    a = _render_fit()
    b = ov.fit_class_b_sbatch_script(
        name="tt-twosi-fitb-t1", config="configs/two_si.yaml", time_limit="0-04:00:00", logdir=f"{RROOT}/_logs",
        store_dir=f"{RROOT}/two_si_store", objects_dir=OBJDIR, sha=SHA, shared_root="/shared/TESTUSER")
    (rendered / "fit_a.sbatch").write_text(a)
    (rendered / "fit_b.sbatch").write_text(b)
    assert _flavor_violations(rendered) == []
    # mutation 1: the flavor dropped from the sync (the pre-fix template) -> "flavor-less uv sync"
    mutated = tmp_path / "flavorless"; mutated.mkdir()
    (mutated / "fit_a.sbatch").write_text(a.replace("uv sync --frozen --extra cu128 --quiet", "uv sync --frozen --quiet", 1))
    assert any("flavor-less uv sync" in v for v in _flavor_violations(mutated)), _flavor_violations(mutated)
    # mutation 2: the unfrozen fallback back in place of the handler -> the frozen-sync rule
    mutated = tmp_path / "fallback"; mutated.mkdir()
    (mutated / "fit_a.sbatch").write_text(re.sub(r"\|\| \{ echo \"FATAL: uv sync[^\n]*", "|| uv sync --quiet", a, count=1))
    assert any("unfrozen `|| uv sync` fallback" in v for v in _flavor_violations(mutated)), _flavor_violations(mutated)


# --------------------------------------------------------------------------- #
# cluster plumbing: QoS derivation + the RESULT sentinel                        #
# --------------------------------------------------------------------------- #


def _render_fit(**overrides):
    kw = dict(
        name="tt-twosi-fit-safety_refusal-t1", si="safety_refusal",
        pairs_remote=f"{RROOT}/two_si_pairs/pairs_safety_refusal.jsonl",
        config="configs/two_si.yaml", time_limit="0-12:00:00",
        logdir=f"{RROOT}/_logs", store_dir=f"{RROOT}/two_si_store",
        objects_dir=OBJDIR, sha=SHA, shared_root="/shared/TESTUSER")
    kw.update(overrides)
    return ov.fit_sbatch_script(**kw)


def test_derive_qos_maps_each_cluster_root():
    """ONE derivation feeds the rendered header and every submit-time
    --qos: the QoS of the site block whose shared root is the run's (each
    cluster names its own; a fixed `high` would be rejected by a scheduler
    that calls it something else), and a root no block describes fails
    LOUDLY instead of submitting at a QoS the cluster does not define."""
    assert ov.derive_qos("/shared/TESTUSER") == "high"
    assert ov.derive_qos("/workspace/TESTUSER/") == "high-b"
    with pytest.raises(ov.StageFailed, match="cannot derive a QoS"):
        ov.derive_qos("/scratch/TESTUSER")


def test_fit_scripts_render_the_given_qos():
    """The generated headers are QoS-parameterised — both fit classes."""
    a = _render_fit(shared_root="/workspace/TESTUSER", qos="high-b")
    assert "#SBATCH --qos=high-b" in a
    assert "#SBATCH --qos=high\n" not in a
    b = ov.fit_class_b_sbatch_script(
        name="tt-twosi-fitb-t1", config="configs/two_si.yaml",
        time_limit="0-04:00:00", logdir=f"{RROOT}/_logs",
        store_dir=f"{RROOT}/two_si_store", objects_dir=OBJDIR, sha=SHA,
        shared_root="/workspace/TESTUSER", qos="high-b")
    assert "#SBATCH --qos=high-b" in b
    # unset -> the historical default, byte-identical
    assert "#SBATCH --qos=high\n" in _render_fit()


def test_result_sentinel_executes_and_never_changes_the_exit_code(tmp_path):
    """RUN the rendered RESULT-sentinel block (not just grep for it): on a
    failing job it writes the one-line outcome (state, exit code, elapsed
    seconds, job id, node) next to the figures WITHOUT changing the job's
    exit code; on success it records COMPLETED; and when even the sentinel
    write itself fails (unwritable figs root), the exit code STILL
    survives — a failed sentinel must never re-fail a finished job."""
    text = _render_fit(shared_root=str(tmp_path))
    start = text.index("export FIGS_DIR=")
    end = text.index("trap result_sentinel EXIT")
    block = "set -uo pipefail\n" + text[start:end] + "trap result_sentinel EXIT\n"
    env = dict(os.environ, SHARED=str(tmp_path), SLURM_JOB_ID="12345",
               SLURMD_NODENAME="TESTNODE")
    result_path = (tmp_path / "exp" / "figs"
                   / "tt-twosi-fit-safety_refusal-t1" / "RESULT")

    proc = subprocess.run(["bash", "-c", block + "exit 7\n"],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 7, proc.stderr          # exit code preserved
    line = result_path.read_text().strip()
    assert line.startswith("FAILED exit=7 elapsed_s=")
    assert "job=12345" in line and "node=TESTNODE" in line
    assert len(line.splitlines()) == 1                # ONE line, exactly

    proc = subprocess.run(["bash", "-c", block + "exit 0\n"],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 0
    assert result_path.read_text().startswith("COMPLETED exit=0 ")

    # sentinel write failure: figs root unwritable -> mkdir and the
    # redirect both fail, exit code STILL preserved, nothing raised
    ro = tmp_path / "ro"
    ro.mkdir()
    ro.chmod(0o555)
    proc = subprocess.run(["bash", "-c", block + "exit 7\n"],
                          capture_output=True, text=True,
                          env=dict(env, SHARED=str(ro)))
    assert proc.returncode == 7
    ro.chmod(0o755)                     # let pytest clean tmp_path up


CHRROOT = "/workspace/TESTUSER/exp/jailbreak"


def test_cluster_b_root_takes_its_block_qos_and_no_excludes(tmp_path):
    """A remote root on cluster B's shared root flips every submission to
    that block's QoS, renders the SAME QoS into the generated headers, and
    seeds NO node exclusions — the site names them for cluster A only, and
    sbatch rejects a --exclude naming nodes the cluster does not have."""
    orch, runner, world, _c = make_orch(
        tmp_path, extra_args=["--remote-root", CHRROOT])
    assert orch.qos == "high-b"
    assert orch.shared_root == "/workspace/TESTUSER"
    assert orch.excluded == []
    assert orch.run() == 0
    assert len(world.sbatch_cmds) == 3
    for cmd in world.sbatch_cmds:
        assert "--qos=high-b" in cmd
        assert "--qos=high " not in cmd
        assert "--exclude" not in cmd
    fitb_text = (orch.private_root / "two_si_overnight"
                 / "tt-twosi-fitb-t1.sbatch").read_text()
    assert "#SBATCH --qos=high-b" in fitb_text
    assert "SHARED=/workspace/TESTUSER\n" in fitb_text
    for e in read_log(orch):
        if e["event"] == "sbatch_submit":
            assert e["qos"] == "high-b"
            assert e["exclude"] == []
    assert_nothing_destructive(runner)


def test_the_site_supplies_the_login_host_root_and_exclusions(tmp_path):
    """With no --cluster-host and no --remote-root, the login host is the
    CLUSTER_SSH of the block whose shared root is the run's, the private root
    is the primary cluster's shared root + /exp/jailbreak, and the exclusion
    list is that block's --exclude list; --exclude-nodes replaces it."""
    args = ov.build_parser().parse_args(["--private-root", str(tmp_path / "priv"), "--remote-project", RPROJECT, "--sha", SHA])
    orch = ov.OvernightChain(args)
    assert orch.cluster_host == "login-0" and orch.remote_root == RROOT and orch.shared_root == "/shared/TESTUSER"
    assert orch.qos == "high" and orch.excluded == ["compute-16", "compute-18"]
    args = ov.build_parser().parse_args(["--private-root", str(tmp_path / "priv"), "--remote-project", RPROJECT, "--sha", SHA,
                                         "--run-tag", "t2", "--exclude-nodes", "compute-3"])
    assert ov.OvernightChain(args).excluded == ["compute-3"]
    args = ov.build_parser().parse_args(["--private-root", str(tmp_path / "priv"), "--remote-project", RPROJECT, "--sha", SHA,
                                         "--run-tag", "t3", "--exclude-nodes", ""])
    assert ov.OvernightChain(args).excluded == []


def test_a_root_no_block_describes_needs_its_login_host(tmp_path):
    args = ov.build_parser().parse_args(["--private-root", str(tmp_path / "priv"), "--remote-project", RPROJECT, "--sha", SHA,
                                         "--remote-root", "/scratch/TESTUSER/exp/jailbreak", "--qos", "q"])
    with pytest.raises(ov.StageFailed, match="--cluster-host"):
        ov.OvernightChain(args)


@pytest.mark.parametrize("extra,nodes", [("--exclude=compute-1,compute-2", ["compute-1", "compute-2"]),
                                         ("--account=a --exclude compute-5", ["compute-5"]), ("-x compute-6", ["compute-6"]),
                                         ("", [])])
def test_the_exclusion_list_reads_every_spelling_of_the_flag(synthetic_site, extra, nodes):
    (synthetic_site / "clusters" / "one.env").write_text(
        "CLUSTER_SSH=login-0\nCLUSTER_SHARED_ROOT=/shared/TESTUSER\nCLUSTER_PARTITION=p\nCLUSTER_QOS=high\n"
        f"CLUSTER_SUBSTRATE=slurm\nCLUSTER_LABEL=one\nCLUSTER_SBATCH_EXTRA='{extra}'\n")
    assert ov.site_exclude_nodes("/shared/TESTUSER") == nodes


def test_no_site_root_refuses_the_default_remote_root(tmp_path, monkeypatch):
    monkeypatch.setenv("SISTEER_CONFIG", str(tmp_path / "absent" / "config.env"))
    with pytest.raises(ov.StageFailed, match="--remote-root"):
        ov.default_remote_root()


def test_explicit_qos_flag_overrides_the_derived_default(tmp_path):
    orch, _r, world, _c = make_orch(tmp_path, extra_args=["--qos", "low"])
    assert orch.qos == "low"
    assert orch.run() == 0
    assert world.sbatch_cmds and all("--qos=low" in c
                                     for c in world.sbatch_cmds)


# --------------------------------------------------------------------------- #
# elapsed parsing + failure classification (the retry gate)                     #
# --------------------------------------------------------------------------- #


def test_elapsed_seconds_parses_all_sacct_shapes():
    assert ov.elapsed_seconds("00:01:30") == 90
    assert ov.elapsed_seconds("1-02:00:00") == 93600
    assert ov.elapsed_seconds("05:00") == 300
    assert ov.elapsed_seconds("") == 0


def test_classify_failure_gates_retry_on_signature_plus_short_elapsed(tmp_path):
    orch, *_ = make_orch(tmp_path)
    row = lambda state, elapsed: {"state": state, "elapsed": elapsed}  # noqa: E731
    assert orch.classify_failure(row("COMPLETED", "01:00:00"), "") == "completed"
    # the GPU leak: fails fast WITH the signature -> retryable
    assert orch.classify_failure(row("FAILED", "00:01:30"),
                                 BUSY_TAIL) == "retryable"
    # slurmd death variant -> retryable regardless of the log
    assert orch.classify_failure(row("NODE_FAIL", "00:00:50"), "") == "retryable"
    # a REAL code failure fails fast WITHOUT the signature -> terminal
    assert orch.classify_failure(row("FAILED", "00:01:00"),
                                 REAL_TAIL) == "terminal"
    # signature text after a LONG run is not the leak -> terminal
    assert orch.classify_failure(row("FAILED", "02:00:00"),
                                 BUSY_TAIL) == "terminal"
    # deliberate stops / walltime are never auto-retried
    assert orch.classify_failure(row("CANCELLED", "00:10:00"), "") == "terminal"
    assert orch.classify_failure(row("TIMEOUT", "12:00:01"), "") == "terminal"
    # an UNREADABLE log can never license a retry (empty tail, no signature)
    assert orch.classify_failure(row("FAILED", "00:01:30"), "") == "terminal"


# --------------------------------------------------------------------------- #
# the full chain, happy path                                                    #
# --------------------------------------------------------------------------- #


def test_full_chain_happy_path(tmp_path):
    orch, runner, world, _clock = make_orch(tmp_path)
    assert orch.run() == 0

    # one fit submission per SI plus ONE Class B job, every one with the
    # required submit-time flags
    assert len(world.sbatch_cmds) == 3
    for cmd in world.sbatch_cmds:
        assert "--exclude=compute-16,compute-18" in cmd
        assert "--qos=high" in cmd
        assert f"--comment=sha:{SHA}" in cmd
        assert "--job-name=tt-twosi-fit" in cmd
        assert "--export" not in cmd                       # hard rule 1
    names = {re.search(r"--job-name=(\S+)", c).group(1)
             for c in world.sbatch_cmds}
    assert names == ({f"tt-twosi-fit-{si}-t1" for si in SIS}
                     | {"tt-twosi-fitb-t1"})
    # step 7 comes after step 6: the Class B job is submitted last
    assert "tt-twosi-fitb-t1" in world.sbatch_cmds[-1]
    # the ACTUAL shipped Class B script (the local copy of what went to the
    # cluster) is the safe one: store exported in the body, no --export, no
    # stage2_common.sh, invoking fit_class_b.py
    fitb_text = (orch.private_root / "two_si_overnight"
                 / "tt-twosi-fitb-t1.sbatch").read_text()
    assert f"export TT_TWO_SI_STORE_DIR={RROOT}/two_si_store" in fitb_text
    assert "--export" not in fitb_text
    assert "stage2_common.sh" not in fitb_text
    assert "fit_class_b.py" in fitb_text

    # the structured log records the chain in order, ending complete
    events = read_log(orch)
    order = [e["event"] for e in events]
    for a, b in [("start", "sbatch_submit"),
                 ("manifest_frozen", "pairs_emitted"),
                 ("pairs_emitted", "sbatch_submit"),
                 ("sbatch_submit", "summary")]:
        assert order.index(a) < order.index(b), f"{a} not before {b}: {order}"
    grade_i = next(i for i, e in enumerate(events)
                   if e["event"] == "pipeline" and e["subcmd"] == "grade-pos")
    manifest_i = next(i for i, e in enumerate(events)
                      if e["event"] == "pipeline" and e["subcmd"] == "manifest")
    emit_i = next(i for i, e in enumerate(events)
                  if e["event"] == "pipeline" and e["subcmd"] == "emit-pairs")
    assert grade_i < manifest_i < emit_i
    summary = events[-1]
    assert summary["event"] == "summary" and summary["outcome"] == "complete"
    assert summary["stages"]["fit"] == "completed"
    assert summary["stages"]["fit-b"] == "completed"
    assert summary["still_running"] == []
    # every submission logged the exclusion list
    for e in events:
        if e["event"] == "sbatch_submit":
            assert e["exclude"] == ["compute-16", "compute-18"]
            assert e["qos"] == "high"
    assert_nothing_destructive(runner)


def test_ungraded_skip_count_reaches_the_log(tmp_path):
    orch, _r, _w, _c = make_orch(
        tmp_path, world_kw={"generate": GEN_OK, "n_ungraded": 7})
    assert orch.run() == 0
    emitted = next(e for e in read_log(orch) if e["event"] == "pairs_emitted")
    assert emitted["n_ungraded_pos_skipped"] == 7


def test_stage_grade_is_wide_by_default_and_prints_coordination(tmp_path,
                                                                capsys, monkeypatch):
    """The unattended launcher's real grade-pos default is 200 (the repo's
    live-grader norm), and the coordination courtesy line is PRINTED into
    the console AND the structured log while grading proceeds in the SAME
    call — print-and-proceed, never a gate. (The old default of 4 quietly
    treated the post as a gate, which the standing rule forbids.)"""
    orch, runner, _w, _c = make_orch(tmp_path)
    assert orch.args.concurrency == 200            # build_parser default
    monkeypatch.setenv("SISTEER_COORDINATION_CHANNEL", "#example-api-coordination")
    orch.stage_grade()

    out = capsys.readouterr().out
    assert "[coordination] ready to post to #example-api-coordination" in out and "lowprio" in out
    grade_call = next(c for c in runner.joined_calls() if "grade-pos" in c)
    assert "--concurrency 200" in grade_call       # the default reaches grade-pos

    events = read_log(orch)
    coord_i = next(i for i, e in enumerate(events)
                   if e["event"] == "coordination_post")
    grade_i = next(i for i, e in enumerate(events)
                   if e["event"] == "pipeline" and e["subcmd"] == "grade-pos")
    assert "lowprio" in events[coord_i]["line"]
    assert coord_i < grade_i                       # printed, then proceeded
    assert orch.stage_status["grade-pos"] == "completed"


def test_stage_grade_without_a_coordination_channel_prints_no_line(tmp_path, capsys, monkeypatch):
    """The channel is site config: a site that names none gets no coordination line or event, and grading proceeds."""
    orch, runner, _w, _c = make_orch(tmp_path)
    monkeypatch.delenv("SISTEER_COORDINATION_CHANNEL", raising=False)
    orch.stage_grade()
    assert "[coordination]" not in capsys.readouterr().out
    assert not any(e["event"] == "coordination_post" for e in read_log(orch))
    assert any("grade-pos" in c for c in runner.joined_calls()) and orch.stage_status["grade-pos"] == "completed"


# --------------------------------------------------------------------------- #
# busy-GPU retry: bounded, node-excluding, and distinguished from real bugs     #
# --------------------------------------------------------------------------- #


def test_busy_gpu_fit_failure_is_resubmitted_with_node_excluded(tmp_path):
    name = f"tt-twosi-fit-{SIS[0]}-t1"
    orch, runner, world, _c = make_orch(
        tmp_path, world_kw={"generate": GEN_OK,
                            "fit_fates": {name: ["BUSY", "COMPLETED"]}})
    assert orch.run() == 0

    cmds = [c for c in world.sbatch_cmds if name in c]
    assert len(cmds) == 2                       # first try + one retry
    assert "--exclude=compute-16,compute-18" in cmds[0]
    # the failing node joined the exclusion for the resubmission
    assert re.search(r"--exclude=\S*compute-7", cmds[1])
    events = read_log(orch)
    assert any(e["event"] == "node_excluded" and e["node"] == "compute-7"
               for e in events)
    assert any(e["event"] == "job_terminal" and e["verdict"] == "retryable"
               for e in events)
    assert_nothing_destructive(runner)


def test_node_fail_is_retryable_too(tmp_path):
    """The node-failure variant: slurmd dies, the job ends NODE_FAIL with no
    stderr to inspect — still a retry-with-exclusion, not a stop."""
    name = f"tt-twosi-fit-{SIS[1]}-t1"
    orch, _r, world, _c = make_orch(
        tmp_path, world_kw={"generate": GEN_OK,
                            "fit_fates": {name: ["NODE_FAIL", "COMPLETED"]}})
    assert orch.run() == 0
    assert len([c for c in world.sbatch_cmds if name in c]) == 2


def test_real_code_failure_is_terminal_after_one_submission(tmp_path):
    """A genuine bug must NOT be retried five times: no busy signature ->
    terminal on the first failure."""
    name = f"tt-twosi-fit-{SIS[0]}-t1"
    orch, runner, world, _c = make_orch(
        tmp_path, world_kw={"generate": GEN_OK,
                            "fit_fates": {name: ["REALFAIL"]}})
    assert orch.run() == 1
    assert len([c for c in world.sbatch_cmds if name in c]) == 1   # no retry
    events = read_log(orch)
    failed = next(e for e in events if e["event"] == "job_failed")
    assert "NOT retried" in failed["reason"]
    assert_nothing_destructive(runner)


def test_busy_signature_after_long_elapsed_is_not_retried(tmp_path):
    """The elapsed gate: the leak kills in 40s–2min; the same words in the
    log of a job that ran 2h are a different disease."""
    name = f"tt-twosi-fit-{SIS[0]}-t1"
    orch, _r, world, _c = make_orch(
        tmp_path, world_kw={"generate": GEN_OK,
                            "fit_fates": {name: ["BUSY_LONG"]}})
    assert orch.run() == 1
    assert len([c for c in world.sbatch_cmds if name in c]) == 1


def test_retries_are_bounded_by_max_attempts(tmp_path):
    name = f"tt-twosi-fit-{SIS[0]}-t1"
    orch, runner, world, _c = make_orch(
        tmp_path, extra_args=("--max-attempts", "3"),
        world_kw={"generate": GEN_OK,
                  "fit_fates": {name: ["BUSY"] * 10}})
    assert orch.run() == 1
    assert len([c for c in world.sbatch_cmds if name in c]) == 3   # the cap
    events = read_log(orch)
    assert any(e["event"] == "job_failed" and "attempt cap" in e["reason"]
               for e in events)
    assert_nothing_destructive(runner)


# --------------------------------------------------------------------------- #
# the generate stage                                                            #
# --------------------------------------------------------------------------- #


def test_generate_busy_failure_without_script_is_terminal(tmp_path):
    orch, _r, world, _c = make_orch(
        tmp_path, world_kw={"generate": GEN_BUSY})
    assert orch.run() == 1
    # no Class A fit was ever submitted — but the INDEPENDENT Class B fit
    # still ran (generation feeds only the Class A pool, not Class B)
    assert not any("--job-name=tt-twosi-fit-" in c for c in world.sbatch_cmds)
    assert any("--job-name=tt-twosi-fitb-t1" in c for c in world.sbatch_cmds)
    events = read_log(orch)
    failed = next(e for e in events if e["event"] == "job_failed")
    assert "no resubmit command" in failed["reason"]
    assert events[-1]["stages"]["fit-b"] == "completed"


def test_generate_busy_failure_with_script_is_resubmitted(tmp_path):
    script = f"{RROOT}/_overnight/genpos.sbatch"
    orch, _r, world, _c = make_orch(
        tmp_path, extra_args=("--generate-script", script),
        world_kw={"generate": GEN_BUSY,
                  "fit_fates": {"tt-twosi-posfull": ["COMPLETED"]}})
    assert orch.run() == 0
    gen_cmds = [c for c in world.sbatch_cmds if "tt-twosi-posfull" in c]
    assert len(gen_cmds) == 1 and script in gen_cmds[0]
    # the node the failed generate died on joined the exclusion
    assert re.search(r"--exclude=\S*compute-9", gen_cmds[0])


def test_missing_generate_job_proceeds_behind_the_regime_gate(tmp_path):
    """No job on record: generation may have finished outside the sacct
    window. The run proceeds — the regime gate protects fitting."""
    orch, _r, _w, _c = make_orch(tmp_path, world_kw={"generate": None})
    assert orch.run() == 0
    assert any(e["event"] == "generate_job_not_found" for e in read_log(orch))


# --------------------------------------------------------------------------- #
# deadline: every loop is bounded                                               #
# --------------------------------------------------------------------------- #


def test_deadline_ends_the_generate_wait_with_a_summary(tmp_path):
    orch, runner, world, clock = make_orch(
        tmp_path, deadline_hours=0.05,             # 180 s, poll every 30 s
        world_kw={"generate": GEN_RUNNING})        # never finishes
    assert orch.run() == 3
    assert world.sbatch_cmds == []                 # stopped submitting
    events = read_log(orch)
    summary = events[-1]
    assert summary["event"] == "summary"
    assert summary["outcome"] == "deadline_expired"
    # the still-running generate job is NAMED, and was left running
    assert "tt-twosi-posfull" in summary["still_running"]
    assert clock.t >= 180                          # the loop really ran out
    assert_nothing_destructive(runner)


def test_deadline_during_fit_monitoring_leaves_jobs_running(tmp_path):
    # fits submit fine but never reach a terminal state
    orch, runner, world, _clock = make_orch(
        tmp_path, deadline_hours=0.05, world_kw={"generate": GEN_OK})
    # freeze both fit jobs in RUNNING forever
    real_sbatch = world._sbatch

    def running_sbatch(argv):
        out = real_sbatch(argv)
        jid = re.search(r"job (\d+)", out.stdout).group(1)
        world.jobs[jid].update(state="RUNNING")
        return out

    runner.handlers = [(p, running_sbatch if r == world._sbatch else r)
                       for p, r in runner.handlers]
    assert orch.run() == 3
    summary = read_log(orch)[-1]
    assert summary["outcome"] == "deadline_expired"
    assert len(summary["still_running"]) == 2      # both fits named, not killed
    assert_nothing_destructive(runner)


# --------------------------------------------------------------------------- #
# re-entry: never redo finished work, never double-submit                       #
# --------------------------------------------------------------------------- #


def test_rerun_adopts_completed_fit_jobs_instead_of_resubmitting(tmp_path):
    pre = [{"job_id": str(2000 + i),
            "name": f"tt-twosi-fit-{si}-t1", "state": "COMPLETED",
            "elapsed": "01:00:00", "exit": "0:0", "node": "compute-31",
            "tail": ""}
           for i, si in enumerate(SIS)]
    pre.append({"job_id": "2002", "name": "tt-twosi-fitb-t1",
                "state": "COMPLETED", "elapsed": "01:00:00", "exit": "0:0",
                "node": "compute-31", "tail": ""})
    orch, _r, world, _c = make_orch(
        tmp_path, world_kw={"generate": GEN_OK, "preexisting": pre})
    assert orch.run() == 0
    assert world.sbatch_cmds == []                 # nothing resubmitted
    events = read_log(orch)
    assert sum(e["event"] == "fit_already_completed" for e in events) == 2
    assert sum(e["event"] == "fit_b_already_completed" for e in events) == 1


def test_rerun_adopts_a_running_fit_job(tmp_path):
    """A live job from a previous invocation is monitored, not duplicated."""
    pre = [{"job_id": "2100", "name": f"tt-twosi-fit-{SIS[0]}-t1",
            "state": "RUNNING", "elapsed": "00:30:00", "exit": "0:0",
            "node": "compute-31", "tail": ""}]
    orch, runner, world, _c = make_orch(
        tmp_path, world_kw={"generate": GEN_OK, "preexisting": pre})

    # let the adopted job complete on the second poll
    polls = {"n": 0}
    real = world._sacct_by_id

    def flipping(argv):
        if "-j 2100" in " ".join(argv):
            polls["n"] += 1
            if polls["n"] >= 2:
                world.jobs["2100"]["state"] = "COMPLETED"
        return real(argv)

    runner.handlers = [(p, flipping if r == world._sacct_by_id else r)
                       for p, r in runner.handlers]
    assert orch.run() == 0
    # only the other SI and the Class B job were submitted; the adopted job
    # got no resubmission
    assert len(world.sbatch_cmds) == 2
    assert not any(f"tt-twosi-fit-{SIS[0]}-t1" in c for c in world.sbatch_cmds)
    assert any(e["event"] == "fit_adopted_running" for e in read_log(orch))


# --------------------------------------------------------------------------- #
# the regime gate                                                               #
# --------------------------------------------------------------------------- #


def test_native_regime_stops_class_a_but_still_fits_class_b(tmp_path):
    orch, runner, world, _c = make_orch(
        tmp_path, world_kw={"generate": GEN_OK, "regime": "native_per_si"})
    assert orch.run() == 1
    # no Class A fit was submitted — the confounded comparison is never fitted
    assert not any("--job-name=tt-twosi-fit-" in c for c in world.sbatch_cmds)
    assert not any("emit-pairs" in j for j in runner.joined_calls())
    events = read_log(orch)
    failed = next(e for e in events if e["event"] == "stage_failed")
    assert "native_per_si" in failed["error"]
    assert "--allow-native" in failed["error"]
    # ...but Class B has no pool input, so its fit still ran to completion
    assert any("--job-name=tt-twosi-fitb-t1" in c for c in world.sbatch_cmds)
    assert events[-1]["stages"]["fit-b"] == "completed"


def test_allow_native_proceeds_to_fitting(tmp_path):
    orch, _r, world, _c = make_orch(
        tmp_path, extra_args=("--allow-native",),
        world_kw={"generate": GEN_OK, "regime": "native_per_si"})
    assert orch.run() == 0
    assert len(world.sbatch_cmds) == 3             # 2 Class A + 1 Class B


# --------------------------------------------------------------------------- #
# lockfile: safe to run twice                                                   #
# --------------------------------------------------------------------------- #


def test_lockfile_refuses_a_live_concurrent_instance(tmp_path):
    orch, runner, _w, _c = make_orch(tmp_path)
    orch.lock_path.write_text(str(os.getpid()))    # a LIVE pid holds the lock
    assert orch.run() == 4
    assert runner.calls == []                      # nothing was touched


def test_stale_lockfile_is_taken_over(tmp_path):
    orch, _r, _w, _c = make_orch(tmp_path)
    p = subprocess.Popen(["true"])
    p.wait()                                       # a DEAD pid
    orch.lock_path.write_text(str(p.pid))
    assert orch.run() == 0
    assert not orch.lock_path.exists()             # released on exit


# --------------------------------------------------------------------------- #
# pinned sha                                                                     #
# --------------------------------------------------------------------------- #


def test_pinned_sha_read_from_local_worktree_head_and_validated(tmp_path):
    orch, runner, _w, _c = make_orch(tmp_path)
    orch._sha = None                               # force the remote read
    assert orch._pinned_sha() == SHA
    assert any("rev-parse" in j for j in runner.joined_calls())

    orch2, runner2, _w2, _c2 = make_orch(tmp_path / "b")
    orch2._sha = None
    runner2.handlers.insert(
        0, (lambda a: a[:1] == ["git"] and "rev-parse" in a,
            lambda a: cp(a, out="not-a-sha\n")))
    with pytest.raises(ov.StageFailed, match="HEAD"):
        orch2._pinned_sha()


# --------------------------------------------------------------------------- #
# Class B (step 7): the rendered script, independence from Class A, resume      #
# --------------------------------------------------------------------------- #


FITB = "tt-twosi-fitb-t1"


def test_fit_class_b_sbatch_script_carries_every_required_flag():
    text = ov.fit_class_b_sbatch_script(
        name=FITB, config="configs/two_si.yaml", time_limit="0-04:00:00",
        logdir=f"{RROOT}/_logs", store_dir=f"{RROOT}/two_si_store",
        objects_dir=OBJDIR, sha=SHA, shared_root="/shared/TESTUSER")
    for required in (
            f"#SBATCH --job-name={FITB}",
            "#SBATCH --partition=general",
            "#SBATCH --qos=high",
            "#SBATCH --gres=gpu:1",
            "#SBATCH --cpus-per-task=16",
            "#SBATCH --mem=128G",
            "#SBATCH --time=0-04:00:00",
            f"#SBATCH --output={RROOT}/_logs/%x_%j.out",
            f"#SBATCH --error={RROOT}/_logs/%x_%j.err",
            "#SBATCH --open-mode=append",
            "#SBATCH --no-requeue",
            "#SBATCH --signal=B:TERM@120",
            # the pin guard against the shared mutable clone:
            f"PINNED_SHA={SHA}",
            'REPO="${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}/checkouts/$PINNED_SHA"',
            "rev-parse HEAD",
            # metrics-only live W&B for the fit body (console capture
            # off; WANDB_DISABLED deliberately unset — transcript jobs
            # keep their own full containment):
            "export WANDB_CONSOLE=off",
            # the optional node-side key file (shared template — see the
            # Class A test):

            # allocator hygiene (the 156649 OOM's fragmentation reserve):
            'export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"',
            # circuit_breaker's store, exported in the BODY (hard rule 1 —
            # NEVER via sbatch --export):
            f"export TT_TWO_SI_STORE_DIR={RROOT}/two_si_store",
            # objects to the commit-independent dir, flag + env:
            f"--out {OBJDIR}",
            f"export TT_TWO_SI_DEFENSES_DIR={OBJDIR}",
            "hf_env.sh",                      # the proven canary env
            "venvs/sisteer-attackgen",    # per-worktree venv
            # the actual work:
            "fit_class_b.py",
            "--config configs/two_si.yaml",
            # the unattended-window RESULT sentinel (shared template):
            f'export FIGS_DIR="$SHARED/exp/figs/{FITB}"',
            '> "$FIGS_DIR/RESULT" || true',
            "trap result_sentinel EXIT",
    ):
        assert required in text, f"missing from Class B sbatch script: {required}"
    # live metrics: the fit body must NOT disable W&B (see the Class A test)
    assert "export WANDB_DISABLED" not in text
    # the cluster-killers and the shared-clone trap can never appear:
    assert "--export" not in text                  # hard rule 1
    assert "--mem=0" not in text                   # hard rule 2
    assert "CUDA_VISIBLE_DEVICES" not in text      # hard rule 3
    assert "--array" not in text                   # one job, never an array
    assert "stage2_common.sh" not in text          # the EXPECTED_SHA abort trap
    # the ONLY sourced file is hf_env.sh, directly — never stage2_common
    sourced = [l.strip() for l in text.splitlines()
               if l.strip().startswith("source ")]
    assert sourced == ['source "$SHARED/scripts/hf_env.sh"']
    # Class B is SI-agnostic: no per-SI plumbing may leak into the payload
    # (the shared template's pin comment may NAME train_two_si_objects.py,
    # but nothing may RUN it here)
    assert "python scripts/fit_class_b.py" in text
    assert "python scripts/train_two_si_objects.py" not in text
    assert "--si " not in text and "--pairs" not in text


def test_class_a_fit_failure_still_attempts_class_b(tmp_path):
    """The two classes share no inputs: a terminal Class A fit failure exits 1
    but must NOT skip the Class B fit."""
    name = f"tt-twosi-fit-{SIS[0]}-t1"
    orch, runner, world, _c = make_orch(
        tmp_path, world_kw={"generate": GEN_OK,
                            "fit_fates": {name: ["REALFAIL"]}})
    assert orch.run() == 1                         # the failure still exits 1
    # ...but Class B was submitted and completed anyway
    assert any(f"--job-name={FITB}" in c for c in world.sbatch_cmds)
    events = read_log(orch)
    assert any(e["event"] == "class_a_chain_failed" for e in events)
    summary = events[-1]
    assert summary["outcome"] == "failed"
    assert summary["stages"]["fit-b"] == "completed"
    assert_nothing_destructive(runner)


def test_class_b_failure_does_not_touch_class_a(tmp_path):
    """...and vice versa: a terminal Class B failure exits 1 while the Class A
    fits stand completed, untouched."""
    orch, runner, world, _c = make_orch(
        tmp_path, world_kw={"generate": GEN_OK,
                            "fit_fates": {FITB: ["REALFAIL"]}})
    assert orch.run() == 1
    # both Class A fits were submitted and completed
    assert len([c for c in world.sbatch_cmds
                if "--job-name=tt-twosi-fit-" in c]) == 2
    events = read_log(orch)
    summary = events[-1]
    assert summary["outcome"] == "failed"
    assert summary["stages"]["fit"] == "completed"     # Class A unaffected
    assert "fit-b" not in summary["stages"]
    assert any(e["event"] == "class_b_failed" for e in events)
    assert any(e["event"] == "job_failed" and e.get("label") == "fit-b"
               for e in events)
    assert_nothing_destructive(runner)


def test_class_b_busy_gpu_failure_is_resubmitted_with_node_excluded(tmp_path):
    """The Class B job rides the same bounded retry-with-exclusion policy as
    every other submission."""
    orch, runner, world, _c = make_orch(
        tmp_path, world_kw={"generate": GEN_OK,
                            "fit_fates": {FITB: ["BUSY", "COMPLETED"]}})
    assert orch.run() == 0
    cmds = [c for c in world.sbatch_cmds if FITB in c]
    assert len(cmds) == 2                          # first try + one retry
    assert "--exclude=compute-16,compute-18" in cmds[0]
    assert re.search(r"--exclude=\S*compute-7", cmds[1])
    assert_nothing_destructive(runner)


def test_rerun_adopts_completed_class_b_job(tmp_path):
    """Resume: a COMPLETED Class B job from a previous invocation (its objects
    already written) is adopted by name — the step is skipped, never redone."""
    pre = [{"job_id": "2200", "name": FITB, "state": "COMPLETED",
            "elapsed": "01:00:00", "exit": "0:0", "node": "compute-31",
            "tail": ""}]
    orch, _r, world, _c = make_orch(
        tmp_path, world_kw={"generate": GEN_OK, "preexisting": pre})
    assert orch.run() == 0
    assert not any(FITB in c for c in world.sbatch_cmds)
    events = read_log(orch)
    assert any(e["event"] == "fit_b_already_completed" for e in events)
    assert events[-1]["stages"]["fit-b"] == "completed"


def test_rerun_adopts_a_running_class_b_job(tmp_path):
    """A live Class B job from a previous invocation is monitored, never
    double-submitted."""
    pre = [{"job_id": "2300", "name": FITB, "state": "RUNNING",
            "elapsed": "00:30:00", "exit": "0:0", "node": "compute-31",
            "tail": ""}]
    orch, runner, world, _c = make_orch(
        tmp_path, world_kw={"generate": GEN_OK, "preexisting": pre})

    # let the adopted job complete on the second poll
    polls = {"n": 0}
    real = world._sacct_by_id

    def flipping(argv):
        if "-j 2300" in " ".join(argv):
            polls["n"] += 1
            if polls["n"] >= 2:
                world.jobs["2300"]["state"] = "COMPLETED"
        return real(argv)

    runner.handlers = [(p, flipping if r == world._sacct_by_id else r)
                       for p, r in runner.handlers]
    assert orch.run() == 0
    # only the two Class A fits were submitted; the adopted Class B got none
    assert len(world.sbatch_cmds) == 2
    assert not any(FITB in c for c in world.sbatch_cmds)
    assert any(e["event"] == "fit_b_adopted_running" for e in read_log(orch))


# --------------------------------------------------------------------------- #
# the objects dir is absolute and commit-independent                             #
# --------------------------------------------------------------------------- #


def test_rendered_scripts_carry_absolute_objects_dir_outside_checkouts(tmp_path):
    """Compiled objects land in ONE absolute dir derived from the remote
    root — NEVER inside the per-commit checkout, where any code change (new
    sha -> new checkout) would silently void resume and refit every object.
    Asserted on the ACTUAL shipped scripts (the local copies of what went to
    the cluster), for BOTH fit classes."""
    orch, _r, _w, _c = make_orch(tmp_path)
    assert orch.run() == 0
    shipped = sorted((orch.private_root / "two_si_overnight").glob("*.sbatch"))
    assert {p.name for p in shipped} == (
        {f"tt-twosi-fit-{si}-t1.sbatch" for si in SIS}
        | {"tt-twosi-fitb-t1.sbatch"})
    for p in shipped:
        text = p.read_text()
        m = re.search(r"--out (\S+)", text)
        assert m, f"{p.name}: no --out objects dir passed to the fit driver"
        out_dir = m.group(1)
        assert out_dir.startswith("/"), \
            f"{p.name}: objects dir is not absolute: {out_dir}"
        assert "checkouts/" not in out_dir, \
            f"{p.name}: objects dir is inside the per-commit checkout: {out_dir}"
        assert out_dir == OBJDIR
        # readers resolve the SAME dir through the body-exported env var, so
        # writer and reader can never split across checkouts
        env = re.search(r"export TT_TWO_SI_DEFENSES_DIR=(\S+)", text)
        assert env and env.group(1) == out_dir


def test_remote_objects_dir_flag_overrides_the_derived_default(tmp_path):
    orch, _r, _w, _c = make_orch(
        tmp_path, extra_args=("--remote-objects-dir", "/elsewhere/objects"))
    assert orch.run() == 0
    shipped = list((orch.private_root / "two_si_overnight").glob("*.sbatch"))
    assert len(shipped) == 3
    for p in shipped:
        text = p.read_text()
        assert "--out /elsewhere/objects" in text
        assert "export TT_TWO_SI_DEFENSES_DIR=/elsewhere/objects" in text


# --------------------------------------------------------------------------- #
# sacct scoping: a recycled job id can never become a verdict                   #
# --------------------------------------------------------------------------- #


def test_every_sacct_query_is_scoped_to_user_and_window(tmp_path):
    """Job ids recycle on this cluster, so an UNSCOPED sacct can return an
    unrelated months-old job. Every query — by id and by name — must carry
    the owning user + submit-window scope and -X (main-job row only)."""
    orch, runner, _w, _c = make_orch(tmp_path)
    assert orch.run() == 0
    sacct_calls = [j for j in runner.joined_calls() if "sacct -n" in j]
    assert sacct_calls, "the run must have polled sacct"
    for joined in sacct_calls:
        assert '-u "$USER"' in joined, f"sacct not scoped to the user: {joined}"
        assert "-S now-7days" in joined, f"sacct without a window: {joined}"
        assert "-X" in joined, f"sacct without -X (main-job row): {joined}"
    # both query forms were exercised and scoped
    assert any("--name=" in j for j in sacct_calls)
    assert any(" -j " in j for j in sacct_calls)


def test_recycled_job_id_row_is_never_adopted_as_a_verdict(tmp_path):
    """The n1 failure: a bare `sacct -j 148750` returned a RECYCLED id's row —
    an unrelated January array job — and its 'TIMEOUT (exit 0:0)' was filed
    as the generate job's verdict, citing a log file that never existed. The
    scoped query makes the stale row invisible, so the orchestrator keeps
    polling (and eventually hits its deadline) rather than adopting a false
    terminal verdict."""
    recycled = {"148750":
                "148750_0|SYNTH_other_project|TIMEOUT|00:02:08|0:0|compute-3"}
    orch, runner, world, _c = make_orch(
        tmp_path, deadline_hours=0.05,             # bounded: 180 s, poll 30 s
        extra_args=("--generate-job-id", "148750"),
        world_kw={"generate": None, "recycled": recycled})
    assert orch.run() == 3                         # deadline, NOT a verdict
    events = read_log(orch)
    # no terminal verdict and no failure were ever filed off the stale row
    assert not any(e["event"] == "job_terminal" for e in events)
    assert not any(e["event"] == "job_failed" for e in events)
    summary = events[-1]
    assert summary["outcome"] == "deadline_expired"
    assert "tt-twosi-posfull" in summary["still_running"]
    # and the id polls really were scoped (what made the row invisible)
    id_polls = [j for j in runner.joined_calls()
                if "sacct -n" in j and " -j " in j]
    assert id_polls
    assert all('-u "$USER"' in j and "-S now-7days" in j for j in id_polls)
    assert_nothing_destructive(runner)
