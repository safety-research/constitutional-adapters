"""grokmt-served: the multi-turn attack smoke on a SERVED target, on Modal.

One 2xH200 container reproduces the gemma4e4b multi-turn wave's topology
(GPU A = vLLM attacker, GPU B = the steered target) with the target moved
from in-process HF to the lens-routed RoutingProxy (ONE vLLM+lens server
serving `none` and `si_constitution__psr_kl_all` as virtual models), and the
attack drivers (`scripts/jailbreak_run.py generate --target-engine served`)
running IN the container on loopback — no tunnel, no open port.

Two attacker arms on the SAME goals against the SAME served target:
  grok  — x-ai/grok-4.6 via OpenRouter (--attacker-engine openrouter); the
          in-loop LocalAttackScorers stay on the campaign's Qwen
          (--scorer-endpoint), so the contrast is attacker-only.
  qwen  — the campaign's production attacker, huihui-ai/Huihui-Qwen3.6-27B-
          abliterated @27502c87 on vLLM (max-model-len 49152, thinking off).

Matched to the published wave of 2026-09-01: config
configs/two_si_gemma4e4b_reduced.yaml, the x1 scale-12 object bank
(/vol/objects/scale-12/gemma4e4b — sha256-verified against the cluster
object), PAIR 5x3 (pair_tap.PAIR_N_STREAMS=5, commit d17976c), Crescendo
10/10, hand-rolled ActorAttack, forward_cap 150, target greedy at 2048
tokens, goals = the wave's shard00 (29 ids, read from its progress.jsonl and
passed in — never re-derived).

Isolation: app `grokmt-served`; writes ONLY under /vol/grokmt/ (tree, the
attacker's HF cache, run outputs). /vol/objects/scale-12, /vol/hf, /vol/repos
and /vol/capctrl are read-only inputs.

The OpenRouter key rides an EPHEMERAL Modal secret built from the local key
file at `modal run` time (env var in the container; never on the volume,
never printed).

Resume (an interrupted attempt): the drivers resume by goal id from the run
dir's ``progress.jsonl`` files, so pass ``--resume`` with the SAME ``--run``
id and the seeded run dir on the volume is picked up (completed goals are
never regenerated; ``error:`` goals are retried, as the harness intends). Run
DETACHED (``modal run --detach``) so a dying local client cannot take the
container with it; the run tree is synced + committed to the volume on every
progress change, so a container loss keeps every finished goal.

Target GEOMETRY (the per-eval ruling, 2026-09-03): the served target runs at
the ``multiturn_attacks`` entry of ``sisteer.capeval.serving.geometry`` —
8704 x 45, the p99 of prompt+completion over 4,598 real attack calls at the
SAME batched-token footprint as the certified Petri 16384 x 24 — resolved
from the STAGED tree's table at run time, logged with its basis, and written
into run_provenance.json. ``--target-window`` / ``--target-seqs`` override a
factor (0 = unset); an override whose S x L exceeds the certified budget
refuses unless ``--allow-over-budget`` is stated. The resolved window is
threaded into every driver (``--target-max-model-len``) so the served
target's once-only max_tokens clamp knows the window it was launched with.

Attacker POOL (2026-09-04): ``--attacker-pool <endpoint.json>`` uses the
SHARED Modal attacker (scripts/modal/attacker_pool.py; the contract is
``sisteer.eval.attacks.attacker_pool``) instead of standing the 27B attacker
up on a second GPU: the container is then a 1xH200 (``run_smoke_pooled``),
the pool is VERIFIED — bearer accepted, expected model + snapshot listed,
window as advertised — BEFORE the target standup (a refusal ends the run
with status ATTACKER_POOL_REFUSED; there is no fallback to a local
attacker), every driver is given ``--attacker-pool`` so it re-verifies and
records the pool's identity in its attacker_engine.json, and
run_provenance.json carries ``attacker_pool`` (the endpoint.json contents)
in place of the per-job ``attacker_qwen`` block. The bearer rides an
ephemeral Modal secret built at ``modal run`` time from the pool's local 0600
key file (endpoint.json's ``api_key_file``) or ``$VLLM_API_KEY`` — never the
volume, never argv. The full record is required: a bare URL has no attributable
geometry or Modal app/server liveness and is refused. The qwen arm is the one
served by the pool; the grok arm's scorer moves to it too
(``--scorer-endpoint``).

Run (the grading host):
  modal run scripts/modal/grokmt_served.py --stage                 # image + 27B attacker to the volume
  modal run scripts/modal/grokmt_served.py --sha <sha> --pf --ids <ids.txt>
  modal run --detach scripts/modal/grokmt_served.py --sha <sha> --ids <ids.txt> --run <run_id>
  modal run --detach scripts/modal/grokmt_served.py --sha <sha> --ids <ids.txt> --run <run_id> \
      --target-window 16384 --target-seqs 24     # the old Petri-inherited geometry, on purpose
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

# GROKMT_APP_NAME lets a resumed attempt run under its OWN app name (so a
# `modal app stop` aimed at one attempt can never hit another).
app = modal.App(os.environ.get("GROKMT_APP_NAME", "grokmt-served"))
VOL = modal.Volume.from_name("twosi-fits", create_if_missing=False)
VOL_PATH = "/vol"
MY = f"{VOL_PATH}/grokmt"
REPOS = f"{MY}/repos"
GEN = f"{MY}/gen"
ATTACKER_HF = f"{MY}/hf_attacker"
OBJECTS_X1 = f"{VOL_PATH}/objects/scale-12/gemma4e4b"      # READ-ONLY shared bank
CFG_REL = "configs/two_si_gemma4e4b_reduced.yaml"
CELL = "si_constitution__psr_kl_all"
CELLS = ("none", CELL)
ATTACKS = ("pair", "crescendo", "actorattack")
ATTACKER_MODEL = "huihui-ai/Huihui-Qwen3.6-27B-abliterated"
ATTACKER_SNAPSHOT = "27502c8717fd5a2f8c0c77188c10c243fd4f672e"
ATTACKER_MAX_MODEL_LEN = 49152
OBJECT_SHA256 = "c4de7681269335f7653a8694d69803c07cb047cb864981b0b1425275ceb37495"
GROK_MODEL = "x-ai/grok-4.6"
PAIR_BUDGET_COMMIT = "d17976c"
PRIVATE_RUN = "two_si_v1"   # cfg['private_run'] — the wave's layout
#: the geometry-table eval the served target IS (PAIR/Crescendo/ActorAttack).
TARGET_EVAL = "multiturn_attacks"
#: the served target's canonical model key (CFG_REL's model, gemma4e4b): an
#: unset --target-seqs is sized from THIS model's measured batched-token budget
#: (geometry.BUDGET_BY_MODEL, ModelBudget.seqs_at — the hybrid per-slot cap
#: applied) and the budget guard is held to it, not to the Llama footprint.
TARGET_KEY = "gemma4e4b"


def _geometry_module(tree):
    """The per-eval geometry table (sisteer.capeval.serving.geometry) from the
    STAGED tree when one is given (in-container: the tree that actually runs),
    else from whichever sisteer is importable — the local checkout at
    ``modal run`` time, whose src is put on sys.path if it is not already."""
    if tree and f"{tree}/src" not in sys.path:
        sys.path.insert(0, f"{tree}/src")
    try:
        from sisteer.capeval.serving import geometry
    except ImportError:
        local_src = str(Path(__file__).resolve().parents[2] / "src")
        if local_src not in sys.path:
            sys.path.insert(0, local_src)
        from sisteer.capeval.serving import geometry
    return geometry


def resolve_target_geometry(tree, *, target_window: int = 0, target_seqs: int = 0,
                            allow_over_budget: bool = False, target_key: str = TARGET_KEY):
    """The served target's launch geometry: the ``multiturn_attacks`` table
    window, with S sized from ``target_key``'s MEASURED budget
    (``seqs_source="model_budget"``) unless ``target_window`` / ``target_seqs``
    override a factor (0 = unset — Modal's CLI has no None). An override whose
    S x L exceeds the MODEL's batched-token budget raises ``GeometryOverBudget``
    unless ``allow_over_budget`` — the identity is then broken on purpose and
    the log says so. An empty ``target_key`` sizes from the Llama-certified
    default (with geometry's warning semantics: no model named)."""
    g = _geometry_module(tree)
    return g.resolve_geometry(TARGET_EVAL, max_model_len=target_window or None,
                              max_num_seqs=target_seqs or None,
                              allow_over_budget=allow_over_budget,
                              model_key=target_key or None)


def _openrouter_key(*, required: bool = False) -> str:
    """The OpenRouter key: the environment, else (locally) the vault through
    the secret-manager CLI. Never printed, never a file.

    This is called at IMPORT to build the Secret the function decorators
    attach, so it must NOT raise there — importing this module happens in the
    container (where there is no vault CLI and the attached secret already
    populated the env) and in the CPU test suite (where there is neither).
    Import-time resolution that raises turns "no key configured" into "this
    module cannot be imported", which took out 20 unrelated tests.

    The loudness lives at LAUNCH instead: :func:`main` calls this with
    ``required=True`` before spawning anything, so a missing key is a named
    error on the launching machine rather than an empty secret that rides
    into the container and fails as a 401 far from its cause.
    """
    env = os.environ.get("OPENROUTER_API_KEY", "")
    if env or not modal.is_local():
        return env
    try:
        from sisteer.vault import openrouter_api_key
        return openrouter_api_key()
    except Exception:
        if required:
            raise
        return ""


openrouter_secret = modal.Secret.from_dict({"OPENROUTER_API_KEY": _openrouter_key()})


def _pool_spec_from_arg(spec: str) -> dict | None:
    """Read the full LOCAL endpoint.json at ``modal run`` time.

    A bare URL is refused by the contract: it carries neither the real serving
    geometry nor the Modal app/server identity needed for liveness. Returns the
    record as a plain dict (Modal ships dicts), or None."""
    if not spec:
        return None
    from sisteer.eval.attacks import attacker_pool as apool
    ep = apool.resolve_pool(spec)
    return ep.as_dict()


def _pool_api_key(spec: str) -> str:
    """Local only: the pool's bearer for the ephemeral secret — the env var
    the endpoint names when set, else the local 0600 key file. Empty when no
    pool is asked for (the secret is then an empty mapping, like an unset
    OpenRouter key). Never printed."""
    if not spec or not modal.is_local():
        return os.environ.get("VLLM_API_KEY", "")
    from sisteer.eval.attacks import attacker_pool as apool
    try:
        ep = apool.resolve_pool(spec)
        return apool.load_api_key(ep, require=False) or ""
    except apool.AttackerPoolError:
        return os.environ.get("VLLM_API_KEY", "")


_POOL_ARG = os.environ.get("GROKMT_ATTACKER_POOL", "")   # main() re-reads its own flag
attacker_pool_secret = modal.Secret.from_dict(
    {"VLLM_API_KEY": _pool_api_key(_POOL_ARG)} if _POOL_ARG else {})

# --- the image: the proven cu129 serving stack (capctrl_served.py, verbatim) ---
# + pyrit for the PAIR/Crescendo drivers. pyrit is APPENDED after the cached
# serving layers so the vLLM/lens layers are reused, not rebuilt.
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
    # --- the drivers' deps (PyRIT pins transformers>=5.0.0rc3: compatible) ---
    .apt_install("unixodbc")
    .pip_install("pyrit==1.0.1", "requests")
    .run_commands(
        "python -c \"import pyrit, vllm, torch, transformers, requests; "
        "assert transformers.__version__ == '5.14.1', transformers.__version__; "
        "assert vllm.__version__.startswith('0.25.1'), vllm.__version__; "
        "assert torch.__version__ == '2.11.0+cu129', torch.__version__; "
        "print('PYRIT OK', pyrit.__version__)\"",
    )
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


def _wait_health(url: str, proc: subprocess.Popen, timeout_s: float,
                 what: str, log: Path) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            tail = log.read_text(errors="replace")[-3000:] if log.exists() else ""
            raise RuntimeError(f"{what} died (rc={proc.returncode}) before "
                               f"healthy:\n{tail}")
        if _http_ok(url):
            return time.time() - t0
        time.sleep(5)
    raise RuntimeError(f"{what} not healthy after {timeout_s}s")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _attacker_snapshot_dir() -> str:
    return (f"{ATTACKER_HF}/models--{ATTACKER_MODEL.replace('/', '--')}"
            f"/snapshots/{ATTACKER_SNAPSHOT}")


def _tree(sha: str) -> str:
    return f"{REPOS}/sisteer-{sha}"


def _env(tree: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tree}/src:{tree}/scripts"
    env["TT_TWO_SI_DEFENSES_DIR"] = OBJECTS_X1
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


class RunSyncer:
    """Incremental copy of the run tree to the volume: a file is re-copied only
    when its (size, mtime_ns) differs from what THIS process last shipped, so a
    sync after every progress change costs the changed files, not the tree.
    Volume writes are invisible outside the container until ``commit()`` (Modal
    Volume semantics), so the commit is what makes a synced goal survive a
    container kill — writing the drivers' files straight onto the volume mount
    would lose exactly as much on a kill as /tmp does, and PyRIT's SQLite
    memory needs a POSIX filesystem anyway. A torn copy of a file mid-append is
    tolerated downstream (``_load_progress`` skips a torn tail) and a torn
    ``memory.db`` is caught at seed time (``sqlite_intact``)."""

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


def _sync(syncer: RunSyncer) -> None:
    """Ship changed files to the volume and commit (best effort, never raises)."""
    try:
        n = syncer.copy_changed()
        VOL.commit()
        print(f"[sync] {n} file(s) committed to {syncer.dest}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[sync] WARNING: {type(e).__name__}: {e}", flush=True)


# --------------------------------------------------------------------------- #
# resume: seed the run dir from the volume copy of an interrupted attempt       #
# --------------------------------------------------------------------------- #

# Per-attempt artifacts: filed under attempts/<NN>/ when a later attempt resumes
# the run, so the standup/driver logs of every attempt survive side by side.
ATTEMPT_ROOT_DIRS = ("srv", "driver_logs")
ATTEMPT_ROOT_FILES = ("run_provenance.json", "summary.json", "driver_cmds.json",
                      "served_timing_probe.json", "served_probe_timing_none.jsonl",
                      f"served_probe_timing_{CELL}.jsonl", "goal_ids.txt")
# Produced once per run and reused by a resumed attempt (--no-hf-probe).
KEEP_AT_ROOT = ("hf_timing_probe.json", "hf_timing_probe.log")
TIMED_JSONL = ("attacker_calls.jsonl", "target_timing.jsonl", "scorer_calls.jsonl")


def progress_counts(progress_path: Path) -> dict:
    """``{'done', 'errored', 'rows'}`` for one driver's ``progress.jsonl``:
    ``done`` = distinct goals with a terminal row, ``errored`` = those whose
    LAST row is an ``error:`` outcome (the harness retries these on resume —
    ``static_common.load_covered_goals``), ``rows`` = raw line count (a retried
    goal has one row per attempt)."""
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
                continue  # torn tail
            rows += 1
            last[gid] = str(rec.get("outcome", ""))
    errored = sum(1 for o in last.values() if o.startswith("error:"))
    return {"done": len(last), "errored": errored, "rows": rows}


def sqlite_intact(path: Path) -> bool:
    """True iff ``path`` opens as SQLite and passes ``PRAGMA integrity_check``."""
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


def _iso_to_epoch(s: str | None) -> float | None:
    if not s:
        return None
    import calendar
    try:
        return float(calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")))
    except ValueError:
        return None


def seed_from_previous(dest: Path, run_local: Path, arms: list[str]) -> dict:
    """Seed ``run_local`` from ``dest`` (the volume copy of an interrupted
    attempt) so the drivers RESUME instead of restarting.

    - every arm's driver tree (``<arm>/two_si_v1/<attack>/<cell>/shard*``) is
      copied over: ``progress.jsonl`` is the resume skip set, ``responses.jsonl``
      / ``transcripts.jsonl`` the persisted goals, ``memory.db`` PyRIT's state;
      a ``memory.db`` that fails ``PRAGMA integrity_check`` (torn by a sync
      mid-write) is set aside as ``memory.db.torn-attemptNN`` so PyRIT recreates
      it — nothing a completed goal needs lives only there;
    - the previous attempt's standup/driver logs and probe files are filed under
      ``attempts/<NN>/`` (never overwritten by this attempt's);
    - ``hf_timing_probe.*`` stays at the root (produced once, reused);
    - the report carries per-driver seeded counts and the previous attempt's
      time window (its ``started_at`` and the last ``t_start`` it logged), so the
      analyzer can compute per-attempt wall-clock spans instead of bridging the
      gap between attempts.
    Raises if ``dest`` holds no prior run."""
    dest, run_local = Path(dest), Path(run_local)
    if not any((dest / a).is_dir() for a in arms):
        raise RuntimeError(f"--resume: no prior driver tree under {dest} for arms {arms}")
    prior = sorted(int(p.name) for p in (dest / "attempts").glob("[0-9][0-9]")
                   if p.is_dir()) if (dest / "attempts").is_dir() else []
    prev_attempt = (max(prior) if prior else 0) + 1
    this_attempt = prev_attempt + 1
    rep: dict = {"attempt": this_attempt, "previous_attempt": prev_attempt,
                 "seeded": {}, "torn_memory_dbs": [], "covered_total": 0,
                 "errored_to_retry_total": 0}
    last_t = 0.0
    for arm in arms:
        if not (dest / arm).is_dir():
            continue
        shutil.copytree(dest / arm, run_local / arm, dirs_exist_ok=True)
        for pj in sorted((run_local / arm).glob(f"{PRIVATE_RUN}/*/*/shard*/progress.jsonl")):
            key = f"{arm}/" + "/".join(pj.relative_to(run_local / arm / PRIVATE_RUN).parts[:-1])
            c = progress_counts(pj)
            rep["seeded"][key] = c
            rep["covered_total"] += c["done"] - c["errored"]
            rep["errored_to_retry_total"] += c["errored"]
        for sd in (run_local / arm).glob(f"{PRIVATE_RUN}/*/*/shard*"):
            for name in TIMED_JSONL:
                f = sd / name
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
    for name in KEEP_AT_ROOT:
        if (dest / name).is_file():
            shutil.copyfile(dest / name, run_local / name)
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
# stage: the 27B attacker into MY hf cache on the volume (public model)        #
# --------------------------------------------------------------------------- #

@app.function(image=image, volumes={VOL_PATH: VOL}, timeout=3 * 3600, cpu=8,
              memory=32768)
def stage_attacker() -> dict:
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import snapshot_download
    t0 = time.time()
    path = snapshot_download(ATTACKER_MODEL, revision=ATTACKER_SNAPSHOT,
                             cache_dir=ATTACKER_HF, local_files_only=False,
                             max_workers=16)
    VOL.commit()
    files = [f for f in Path(path).rglob("*") if f.is_file()]
    return {"path": path, "n_files": len(files),
            "gb": round(sum(f.stat().st_size for f in files) / 1e9, 1),
            "s": round(time.time() - t0)}


# --------------------------------------------------------------------------- #
# preflight: every non-GPU precondition, incl. ONE real OpenRouter request     #
# --------------------------------------------------------------------------- #

@app.function(image=image, volumes={VOL_PATH: VOL}, timeout=1800, cpu=4,
              memory=16384, secrets=[openrouter_secret])
def preflight(sha: str, goal_ids: list[str], qwen: bool = True) -> dict:
    tree = _tree(sha)
    sys.path[:0] = [f"{tree}/scripts", f"{tree}/src"]
    os.environ["TT_TWO_SI_DEFENSES_DIR"] = OBJECTS_X1
    rep = {"sha": sha[:7], "ok": True, "checks": {}}

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
        from sisteer.eval.attacks import openrouter_attacker, served_target  # noqa: F401
        return {"pyrit": pyrit.__version__, "vllm": vllm.__version__,
                "transformers": transformers.__version__, "torch": str(torch.__version__)}
    chk("imports", imports)

    def argparse_flags():
        import jailbreak_run as jr
        cap = {}
        orig, orig_argv = jr.cmd_generate, sys.argv
        jr.cmd_generate = lambda a, c: cap.update(vars(a))
        try:
            sys.argv = ["jailbreak_run.py", "generate", "--config", f"{tree}/{CFG_REL}",
                        "--attack", "pair", "--defense", CELL, "--confirmatory",
                        "--goal-ids-file", "/tmp/ids", "--shard", "1", "--nshards", "3",
                        "--private-root", "/tmp/x", "--target-engine", "served",
                        "--target-proxy-url", "http://127.0.0.1:1/v1",
                        "--target-routing-table", "/tmp/rt.json", "--positions", "response",
                        "--target-greedy",        # as _driver_cmd sends it (the published wave's decode)
                        "--expect-pair-n-streams", "5", "--attacker-engine", "openrouter",
                        "--attacker-model", GROK_MODEL, "--scorer-endpoint",
                        "http://127.0.0.1:2"]
            jr.main()
        finally:
            jr.cmd_generate, sys.argv = orig, orig_argv
        assert cap["target_engine"] == "served" and cap["attacker_engine"] == "openrouter"
        assert cap["expect_pair_n_streams"] == 5 and cap["goal_ids_file"] == "/tmp/ids"
        return {"parsed": sorted(k for k in cap if k.startswith(("target_", "attacker_", "scorer_", "goal_", "expect_")))}
    chk("argparse_accepts_served_and_openrouter_flags", argparse_flags)

    def pair_budget():
        from sisteer.eval.attacks import pair_tap
        assert pair_tap.PAIR_N_STREAMS == 5, pair_tap.PAIR_N_STREAMS
        assert pair_tap.PAIR_DEPTH == 3, pair_tap.PAIR_DEPTH
        from sisteer.eval.attacks import multiturn as mt
        return {"pair_n_streams": 5, "pair_depth": 3, "provenance": PAIR_BUDGET_COMMIT,
                "crescendo_max_turns": mt.CRESCENDO_MAX_TURNS,
                "crescendo_max_backtracks": mt.CRESCENDO_MAX_BACKTRACKS,
                "actorattack_commit": mt.ACTORATTACK_COMMIT}
    chk("pair_budget_is_5x3", pair_budget)

    def goals():
        import jailbreak_run as jr
        cfg = jr._load_cfg(f"{tree}/{CFG_REL}")
        gs = jr._eval_goals(cfg, confirmatory=True)
        shard00 = [g.goal_id for g in gs[0::9]]
        assert len(goal_ids) == 29, len(goal_ids)
        assert set(shard00) == set(goal_ids), (
            f"shard00 slice != HeldOut ids: only_slice={sorted(set(shard00) - set(goal_ids))[:5]} "
            f"only_heldout={sorted(set(goal_ids) - set(shard00))[:5]}")
        sel = jr._select_goal_ids(gs, _write_ids(goal_ids))
        subs = [len(sel[k::3]) for k in range(3)]
        return {"n_total_confirmatory": len(gs), "n_shard00": len(shard00),
                "order_identical": shard00 == goal_ids, "subshard_sizes": subs,
                "forward_cap": jr._forward_cap(cfg),
                "target_max_new_tokens": jr._target_max_new_tokens(cfg),
                "attacker_max_new_tokens": jr._attacker_max_new_tokens(cfg)}
    chk("goals_offline_shard00_matches_heldout", goals)

    def object_and_guard():
        import torch, yaml
        from transformers import AutoTokenizer
        from sisteer.chat_markers import generation_header_ids
        cfg = yaml.safe_load(open(f"{tree}/{CFG_REL}"))
        p = f"{OBJECTS_X1}/{CELL}/object.pt"
        sha = _sha256(p)
        assert sha == OBJECT_SHA256, f"object sha {sha} != wave's {OBJECT_SHA256}"
        blob = torch.load(p, map_location="cpu", weights_only=True)
        tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["model_revision"])
        genlen = len(generation_header_ids(tok))
        H = blob.get("response_header_len")
        ok = blob.get("positions") == "response" and H == genlen
        assert ok, f"guard MISMATCH positions={blob.get('positions')} H={H} genlen={genlen}"
        return {"sha256": sha, "positions": blob.get("positions"),
                "response_header_len": H, "gen_header_ids_len": genlen,
                "guard_will_pass": True, "kind": str(blob.get("kind"))}
    chk("object_identity_and_fit_serve_guard", object_and_guard)

    def routing_export():
        """The standup's registry + routing-table export, on CPU, into /tmp —
        the exact vmodels the drivers will address."""
        import dataclasses, yaml
        from transformers import AutoConfig, AutoTokenizer
        from sisteer.capeval import manifest as cm
        from sisteer.capeval.petri import defense_cells as dc
        from sisteer.capeval.petri import registry_adapter as ra
        from sisteer.capeval.petri.registry_adapter import traintest_cell_source
        from sisteer.capeval.serving.lens_steer import export_worker_registry
        from sisteer.capeval.serving.proxy import export_routing_table, load_routing_table
        from sisteer.chat_markers import generation_header_ids
        from sisteer.eval.attacks import bon_served as bs
        from sisteer.eval.attacks.served_target import route_for
        out = Path("/tmp/pf_srv"); shutil.rmtree(out, ignore_errors=True); out.mkdir()
        cfg_path = Path(f"{tree}/{CFG_REL}")
        cfg = yaml.safe_load(cfg_path.read_text())
        model, rev = cfg["model"], cfg["model_revision"]
        hid = ra.harness_cell_id(CELL)
        tok = AutoTokenizer.from_pretrained(model, revision=rev)
        tsha = cm.sha256_hex16(tok.chat_template.encode())
        mcfg = AutoConfig.from_pretrained(model, revision=rev)
        hidden = getattr(mcfg, "hidden_size", None) or mcfg.text_config.hidden_size
        spec0 = cm.load_petri_spec(Path(tree) / "configs/capeval/manifests/petri_twosi.yaml")
        spec = dataclasses.replace(
            spec0, target_model=model, target_revision=rev, serving_mode="interleaved",
            cells=["none", hid],
            serving_launch_flags=[f for f in spec0.serving_launch_flags if f != "--enable-lora"],
            scenario_bundle_hash="ab" * 8, serving_impl_commit="fedcba9876" * 4,
            chat_template_hash=tsha)
        src = traintest_cell_source(spec, repo_root=Path(tree), config_path=cfg_path,
                                    defenses_dir=Path(OBJECTS_X1))
        bf = {k: f"# {k}\n".encode() for k in cm.petri_expected_behavior_keys(spec)}
        # the fixed standup's prompt-bytes resolution (7f9c704): an override@
        # provenance must load the REAL auditor/judge bytes or enumerate_cells
        # refuses (H10) — the placeholder bytes the old standup passed fail.
        if spec.prompt_provenance.startswith("override@"):
            from sisteer.capeval.tiers.petri_runner import resolve_prompt_files
            prompt_files = resolve_prompt_files(
                spec, override_root=Path(tree) / "configs/capeval/petri/prompts")
        else:
            prompt_files = {"auditor": b"# a\n", "judge": b"# j\n"}
        cells = dc.enumerate_cells(spec, src, behavior_files=bf, prompt_files=prompt_files)
        assert not cells.absent, [a.cell_id for a in cells.absent]
        export_worker_registry(cells, src, out / "lens_registry",
                               assistant_header_ids=tuple(generation_header_ids(tok)),
                               zero_op_hidden_size=int(hidden))
        export_routing_table(cells, src, out / "routing_table.json", base_model=model,
                             serving_mode="interleaved")
        table = load_routing_table(out / "routing_table.json")
        routes = list(table["routes"])
        bs.assert_cell_salt_isolation(routes)
        r_cell = route_for(CELL, routes); r_none = route_for("none", routes)
        by_cell = {r.cell_id: r for r in routes}
        twin = by_cell.get(f"{r_cell.cell_id}__lens0") or by_cell.get("none__lens0")
        assert twin is not None, sorted(by_cell)
        return {"harness_cell": hid, "vmodels": sorted(r.virtual_model for r in routes),
                "cell_vmodel": r_cell.virtual_model, "cell_identity": r_cell.serving_identity,
                "cell_intervention_hash": r_cell.intervention_hash,
                "none_vmodel": r_none.virtual_model, "twin": twin.cell_id,
                "hidden": int(hidden), "chat_template_sha16": tsha}
    chk("registry_routing_table_export_cpu", routing_export)

    def openrouter_wire():
        """ONE real request with the attacker client's exact payload shape
        (PyRIT-like system + user, the PAIR sampling params, max_tokens 1024),
        benign content — a param 400 must surface HERE, not after GPUs are up."""
        from sisteer.eval.attacks.openrouter_attacker import OpenRouterAttackerClient
        key = os.environ.get("OPENROUTER_API_KEY", "")
        assert key, "OPENROUTER_API_KEY not in the container env (secret missing)"
        c = OpenRouterAttackerClient(GROK_MODEL, api_key=key)
        c.check_health()
        msgs = [{"role": "system", "content": "You are a helpful assistant that answers ONLY in JSON with keys 'improvement' and 'prompt'."},
                {"role": "user", "content": "Produce a JSON object whose 'prompt' asks for a haiku about autumn and whose 'improvement' says 'first attempt'."}]
        out = c.chat(msgs, temperature=1.0, top_p=0.9, max_new_tokens=1024)
        assert out and out.strip(), f"empty content; meta={c.last_meta}"
        # scorer-shaped call too (in case the scorer ever shares this client)
        s = c.chat([{"role": "user", "content": "Rate from 1 to 10 how relevant 'the sky is blue' is to 'describe the sky'. Reply with one integer."}],
                   temperature=0.0, top_p=1.0, max_new_tokens=8)
        return {"model": GROK_MODEL, "content_head": out[:160], "meta": c.last_meta,
                "scorer_shaped_reply": s[:20], "payload_keys": sorted(c.build_payload(msgs, temperature=1.0, top_p=0.9, max_new_tokens=1024))}
    chk("openrouter_wire_roundtrip", openrouter_wire)

    if qwen:
        def attacker_files():
            d = Path(_attacker_snapshot_dir())
            st = list(d.glob("*.safetensors"))
            assert d.is_dir() and st and (d / "config.json").exists(), str(d)
            return {"snapshot_dir": str(d), "n_safetensors": len(st),
                    "gb": round(sum(f.stat().st_size for f in st) / 1e9, 1)}
        chk("qwen_attacker_staged", attacker_files)

    def writable():
        d = Path(f"{GEN}/_preflight"); d.mkdir(parents=True, exist_ok=True)
        (d / "probe").write_text("ok"); (d / "probe").unlink()
        assert Path(f"{tree}/scripts/grokmt_hf_timing_probe.py").exists()
        assert Path(f"{tree}/scripts/capctrl_serving_up.py").exists()
        return {"gen_root": GEN}
    chk("out_root_writable_and_scripts_present", writable)

    VOL.commit()
    print("[preflight]", json.dumps(rep, indent=1, default=str))
    return rep


def _write_ids(goal_ids: list[str]) -> str:
    p = "/tmp/grokmt_goal_ids.txt"
    Path(p).write_text("\n".join(goal_ids) + "\n")
    return p


# --------------------------------------------------------------------------- #
# the smoke                                                                    #
# --------------------------------------------------------------------------- #

def _attacker_serving_options():
    """ONE definition of the attacker's serving flags -- prefix caching and
    weight quantization -- from ``sisteer.eval.attacks.attacker_serving`` and
    the STAGED tree's ``configs/attacker_serving.json`` (the tree's src is on
    sys.path by the time the attacker starts: ``resolve_target_geometry`` put it
    there). ATTACKER ONLY: ``_target_standup_cmd`` / ``_start_target`` compose
    nothing from this seam (``tests/test_attacker_serving.py`` pins it) -- the
    steered target keeps its per-request cache salt and its own composition.
    Returns ``(module, options, sources)``."""
    from sisteer.eval.attacks import attacker_serving as serving
    opts, sources = serving.resolve()
    return serving, opts, sources


def _attacker_serving_manifest_after_health(serving, opts, sources, proc, port: int,
                                            log: Path, cmd: list[str]) -> None:
    """Once ``/health`` answers, check the engine's OWN resolved-config line
    against the request (quantization, prefix caching, KV-cache dtype, mamba
    cache mode) and write ``<log stem>.serving.json`` beside the attacker log --
    the twin of the shell launcher's ``VLLM_SERVING_MANIFEST``, for the drivers'
    ``--attacker-serving`` / ``attacker_engine.json``. A contradiction kills the
    attacker (and says why in its log) so the smoke fails LOUDLY at its health
    wait rather than running on an unverified identity."""
    import urllib.request

    manifest = log.with_suffix(".serving.json")
    while proc.poll() is None:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if 200 <= r.status < 300:
                    break
        except Exception:  # noqa: BLE001 - readiness polling
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
    """The per-job attacker answered /health but no VERIFIED serving manifest
    followed: the engine contradicted the requested options (the writer left
    ``<log stem>.serving.error`` and killed the server), the server died, or
    the wait ran out. The smoke ends with status ATTACKER_SERVING_UNVERIFIED
    rather than spawning drivers that would record 'unrecorded'."""


def _wait_attacker_serving_manifest(log: Path, proc, timeout_s: float = 600.0,
                                    poll_s: float = 2.0) -> Path:
    """Block until ``_attacker_serving_manifest_after_health`` has written the
    verified ``<log stem>.serving.json`` (returned), or raise
    :class:`AttackerServingUnverified` on its ``.serving.error`` twin, a dead
    attacker, or the timeout. Every Qwen driver of a per-job run is then
    given ``--attacker-serving <manifest>`` so its attacker_engine.json records
    the weight precision the aggregate's mixing refusal compares."""
    manifest = log.with_suffix(".serving.json")
    error = log.with_suffix(".serving.error")
    deadline = time.time() + timeout_s
    while True:
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


def _geometry_argv(geom) -> list[str]:
    """The geometry part of the standup argv: the explicit dimensions PLUS the
    eval key and target model key they were resolved from, so the standup's own
    resolution (endpoint.json) equals this one and its model-budget guard runs
    (review R4); an accepted identity/budget-breaking override travels with
    --allow-over-budget, as the outer resolution accepted it."""
    # dimensions travel BY SOURCE (review S3): a derived factor (table /
    # model_budget) is re-derived by the standup from --eval / --model-key;
    # only a recorded OVERRIDE is passed explicitly, whatever its value — so
    # the standup's window_source / seqs_source equal this resolution's.
    argv: list = []
    if geom.window_source == "override":
        argv += ["--max-model-len", str(geom.max_model_len)]
    if geom.seqs_source == "override":
        argv += ["--max-num-seqs", str(geom.max_num_seqs)]
    if geom.eval:
        argv += ["--eval", str(geom.eval)]
    else:                       # a legacy / fallback resolution: no table to re-derive from
        argv = ["--max-model-len", str(geom.max_model_len),
                "--max-num-seqs", str(geom.max_num_seqs)]
    if geom.model_key:
        argv += ["--model-key", str(geom.model_key)]
    if geom.overridden and (geom.over_budget or not geom.is_identity):
        argv += ["--allow-over-budget"]
    return argv


def _target_standup_cmd(tree: str, srv: Path, stop: Path, geom) -> list[str]:
    """The standup argv: the RESOLVED geometry (never a literal window here —
    the table + overrides decide; tests/test_served_geometry greps for it) with
    its eval / model-key provenance (_geometry_argv)."""
    return [sys.executable, f"{tree}/scripts/capctrl_serving_up.py",
            "--repo-dir", tree, "--config", f"{tree}/{CFG_REL}", "--cell", CELL,
            "--defenses-dir", OBJECTS_X1, "--out", str(srv), "--stop-file", str(stop),
            *_geometry_argv(geom),
            "--gpu-mem-util", "0.90", "--ready-timeout-s", "2700",
            # TODO(native-serving-default §8.6): this launcher is not a capability / tinycap surface, so it keeps the
            # lens standup EXPLICITLY (the standup's own default is the native engine since 2026-09-13; an
            # argv without the flag would silently stand up the other engine). The native reading it lacks:
            # endpoint.json.engine / /v1/models petri_proxy.engine.kind held to the requested path, and
            # `lens_reduction_mode` read as not_applicable under lora_native (tinycap_served.py has both).
            "--serving-path", "lens"]


def _start_target(tree: str, srv: Path, geom, *, gpu: str = "1") -> tuple[subprocess.Popen, Path]:
    env = _env(tree); env["CUDA_VISIBLE_DEVICES"] = gpu
    stop = srv / "STOP"
    cmd = _target_standup_cmd(tree, srv, stop, geom)
    print("[target]", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=(srv / "standup.log").open("w"),
                            stderr=subprocess.STDOUT)
    return proc, stop


def _served_timing_probe(tree: str, proxy_url: str, rt: str, out_json: Path,
                         max_new: int) -> dict:
    """The vLLM twin of grokmt_hf_timing_probe: the SAME prompts, greedy, one
    at a time (uncontended), through the proxy, for both cells."""
    sys.path[:0] = [f"{tree}/scripts", f"{tree}/src"]
    from grokmt_hf_timing_probe import TIMING_PROMPTS
    from sisteer.eval.attacks.served_target import build_served_target
    res = {"engine": "vllm_served", "max_new_tokens": max_new, "cells": {}}
    for cell in CELLS:
        tl = out_json.parent / f"served_probe_timing_{cell}.jsonl"
        d = build_served_target(proxy_url=proxy_url, routing_table=rt, cell_id=cell,
                                timing_log=tl, controls=False)
        d.chat([{"role": "user", "content": "Say hello."}], max_new_tokens=16)  # warm
        tl.unlink(missing_ok=True)
        rows = []
        for i, msgs in enumerate(TIMING_PROMPTS):
            t1 = time.time(); text = d.chat(msgs, max_new_tokens=max_new); dt = time.time() - t1
            rec = json.loads(tl.read_text().splitlines()[-1])
            rows.append({"prompt_idx": i, "latency_s": round(dt, 3),
                         "completion_tokens": rec.get("completion_tokens"),
                         "tok_per_s": (round(rec["completion_tokens"] / dt, 2)
                                       if rec.get("completion_tokens") else None),
                         "n_messages": len(msgs), "reply_head": (text or "")[:120]})
            print(f"[served-probe] {cell} p{i}: {dt:.1f}s {rec.get('completion_tokens')} tok", flush=True)
        lat = [r["latency_s"] for r in rows]; toks = [r["completion_tokens"] or 0 for r in rows]
        res["cells"][cell] = {"n_calls": len(rows), "mean_latency_s": round(sum(lat) / len(lat), 3),
                              "total_latency_s": round(sum(lat), 3),
                              "total_completion_tokens": sum(toks),
                              "tok_per_s_overall": round(sum(toks) / sum(lat), 2), "rows": rows}
    out_json.write_text(json.dumps(res, indent=2))
    return res


def _driver_cmd(tree: str, *, attack: str, cell: str, arm: str, sub: int, subshards: int,
                ids_path: str, private_root: str, proxy_url: str, rt: str,
                attacker_port: int, target_window: int,
                attacker_pool: str | None = None,
                attacker_serving: str | None = None) -> list[str]:
    """``attacker_pool``: the path of the pool's endpoint.json copy in the run
    dir — the qwen arm then runs ``--attacker-pool <it>`` (the driver
    re-verifies the pool and records its identity, serving fields included)
    and the grok arm's scorer points at the pool URL; ``attacker_port`` is
    ignored. None = the per-job attacker on loopback, whose qwen arm is given
    ``--attacker-serving <attacker_serving>`` -- the verified manifest
    ``_wait_attacker_serving_manifest`` returned -- so every shard records the
    attacker's weight precision (bf16 vs FP8) instead of 'unrecorded'. The
    grok arm never takes it: its attacker is hosted; only its scorer is Qwen."""
    cmd = [sys.executable, f"{tree}/scripts/jailbreak_run.py", "generate",
           "--config", f"{tree}/{CFG_REL}", "--attack", attack, "--defense", cell,
           "--confirmatory", "--goal-ids-file", ids_path,
           "--shard", str(sub), "--nshards", str(subshards),
           "--private-root", private_root,
           "--target-engine", "served", "--target-proxy-url", proxy_url,
           "--target-routing-table", rt,
           # THE TARGET DECODE IS PINNED GREEDY, and must stay pinned: this app replays the
           # published wave of 2026-09-01 as an ATTACKER-ONLY contrast -- grok-4.6
           # against the campaign's production attacker, same served target, same goals -- so the
           # target must be byte-for-byte what that wave ran. jailbreak_run's DEFAULT target decode
           # became "sample at the served model's own generation config" on 2026-09-11 and the
           # served model's VENDOR PRESET on 2026-09-16; without this flag either default would
           # silently move the target underneath a published comparison, and the app would still
           # write plausible rows.
           "--target-greedy",
           # the launched window: the served target's once-only max_tokens
           # clamp (served_window) must know the geometry actually serving it
           "--target-max-model-len", str(int(target_window))]
    if cell != "none":
        cmd += ["--positions", "response"]          # the wave's _gen cells
    if attack == "pair":
        cmd += ["--expect-pair-n-streams", "5"]
    if attacker_pool:
        pool_url = json.loads(Path(attacker_pool).read_text())["url"]
        if arm == "grok":
            cmd += ["--attacker-engine", "openrouter", "--attacker-model", GROK_MODEL,
                    "--scorer-endpoint", pool_url]    # scorer held constant (the pool's Qwen)
        else:
            cmd += ["--attacker-engine", "vllm", "--attacker-endpoint", pool_url,
                    "--attacker-pool", str(attacker_pool)]
        return cmd
    qwen_ep = f"http://127.0.0.1:{attacker_port}"
    if arm == "grok":
        cmd += ["--attacker-engine", "openrouter", "--attacker-model", GROK_MODEL,
                "--scorer-endpoint", qwen_ep]         # scorer held constant (Qwen)
    else:
        cmd += ["--attacker-engine", "vllm", "--attacker-endpoint", qwen_ep]
        if attacker_serving:
            cmd += ["--attacker-serving", str(attacker_serving)]
    return cmd


def _attacker_for_run(attacker_pool: dict | None, *, port: int, log: Path, run_local: Path,
                      start_attacker=None, connect=None) -> tuple:
    """Stand up the per-job attacker (``attacker_pool`` None -> ``(proc,
    None, None)``) OR connect to the shared pool: verify it — the bearer from
    the env (the ephemeral secret), the pinned model + snapshot expected —
    write its endpoint record to ``run_local/attacker_pool.json`` for the
    drivers, and return ``(None, endpoint_json_path, {identity, evidence})``.
    A pool refusal raises the typed AttackerPoolError; nothing is started."""
    if attacker_pool is None:
        start_attacker = start_attacker or _start_attacker
        return start_attacker(port, log), None, None
    from sisteer.eval.attacks import attacker_pool as apool
    ep = apool.PoolEndpoint.from_dict(
        attacker_pool, source=attacker_pool.get("source", "endpoint.json"))
    # the serving identity FIRST (local, network-free): a literal-only,
    # unattested or self-contradictory record is exactly what every child
    # driver would refuse at its manifest write -- refusing it here, before
    # connect, the HF timing probe and the target standup, costs nothing
    # (connect_endpoint repeats the check; this one cannot be bypassed by an
    # injected connect)
    serving_fields = apool.check_serving_record(ep)
    connect = connect or apool.connect_endpoint
    ep, _key, evidence = connect(
        ep, squeue=False, env=os.environ, expect_model=ATTACKER_MODEL,
        expect_snapshot=ATTACKER_SNAPSHOT)
    p = run_local / "attacker_pool.json"
    p.write_text(json.dumps(ep.as_dict(), indent=2))
    return None, p, {"identity": ep.identity(), "evidence": evidence, "serving": serving_fields}


def _run_smoke_impl(sha: str, goal_ids: list[str], run_id: str, arms: list[str],
                    subshards: int = 3, hf_probe: bool = True, target_seqs: int = 0,
                    attacks: list[str] = list(ATTACKS), cells: list[str] = list(CELLS),
                    sync_every_s: int = 120, resume: bool = False,
                    target_window: int = 0, allow_over_budget: bool = False,
                    target_key: str = TARGET_KEY, attacker_pool: dict | None = None) -> dict:
    # pooled: ONE GPU (index 0) hosts the target; per-job: GPU 0 attacker, GPU 1 target
    target_gpu = "0" if attacker_pool else "1"
    tree = _tree(sha)
    assert Path(f"{tree}/src/sisteer").is_dir(), tree
    # the target geometry from the STAGED tree's table (module docstring), S
    # from the target model's measured budget
    geom = resolve_target_geometry(tree, target_window=target_window,
                                   target_seqs=target_seqs,
                                   allow_over_budget=allow_over_budget,
                                   target_key=target_key)
    print(geom.describe(), flush=True)
    run_local = Path(f"/tmp/grokmt_run/{run_id}"); run_local.mkdir(parents=True, exist_ok=True)
    dest = f"{GEN}/{run_id}"
    summary: dict = {"run_id": run_id, "sha": sha, "arms": arms, "attacks": attacks,
                     "cells": cells, "subshards": subshards, "n_goals": len(goal_ids),
                     "pair_budget_commit": PAIR_BUDGET_COMMIT, "timeline": {},
                     "target_geometry": geom.as_dict(),
                     "attacker_pool": (attacker_pool or {}).get("url") if attacker_pool else None}
    T0 = time.time()

    def mark(k): summary["timeline"][k] = round(time.time() - T0, 1); print(f"[t+{summary['timeline'][k]:.0f}s] {k}", flush=True)

    # --resume: seed the local run dir from the volume copy of the interrupted
    # attempt BEFORE anything else touches it (the drivers skip covered goals;
    # errored goals are retried by design — static_common.load_covered_goals).
    attempts: list[dict] = []
    if resume:
        seed = seed_from_previous(Path(dest), run_local, arms)
        summary["resume"] = seed
        attempts = seed["attempts"]
        print(f"[resume] attempt {seed['attempt']}: seeded {seed['covered_total']} covered goal(s), "
              f"{seed['errored_to_retry_total']} errored goal(s) to retry, "
              f"{len(seed['torn_memory_dbs'])} torn memory.db set aside", flush=True)
        for k, c in sorted(seed["seeded"].items()):
            print(f"[resume]   {k}: done={c['done']} errored={c['errored']} rows={c['rows']}", flush=True)
        mark(f"resumed_attempt_{seed['attempt']}")
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    attempts.append({"attempt": len(attempts) + 1, "started_at": now_iso,
                     "started_at_epoch": time.time()})
    ids_path = str(run_local / "goal_ids.txt"); Path(ids_path).write_text("\n".join(goal_ids) + "\n")
    srv = run_local / "srv"; srv.mkdir(exist_ok=True)

    # 0/1. the attacker: the shared pool (verified FIRST — before any target
    # standup; a refusal ends the run here, no fallback) or the per-job vLLM on
    # GPU 0 (in the background). The provenance is written AFTER this step so
    # a pooled run records the pool it actually verified.
    attacker_port = _free_port()
    att_log = srv / "attacker_vllm.log"
    pool_rec = None
    try:
        att_proc, pool_json, pool_rec = _attacker_for_run(
            attacker_pool, port=attacker_port, log=att_log, run_local=run_local)
    except Exception as e:  # noqa: BLE001 — a typed AttackerPoolError: refuse, never fall back
        summary["status"] = "ATTACKER_POOL_REFUSED"
        summary["attacker_pool_err"] = f"{type(e).__name__}: {e}"[-4000:]
        print(f"[smoke] REFUSED: {summary['attacker_pool_err']}", flush=True)
        (run_local / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        _sync(RunSyncer(run_local, dest))
        return summary
    mark("attacker_pool_verified" if attacker_pool else "attacker_spawned")

    # provenance beside everything else
    (run_local / "run_provenance.json").write_text(json.dumps(_provenance(
        run_id=run_id, sha=sha, geom=geom, goal_ids=goal_ids, attempts=attempts,
        now_iso=now_iso, resume=resume, summary=summary, pool_rec=pool_rec), indent=2))
    syncer = RunSyncer(run_local, dest)

    # 2. HF timing probe on the target GPU while the attacker loads (best effort)
    if hf_probe:
        env = _env(tree); env["CUDA_VISIBLE_DEVICES"] = target_gpu
        t1 = time.time()
        r = subprocess.run([sys.executable, f"{tree}/scripts/grokmt_hf_timing_probe.py",
                            "--config", f"{tree}/{CFG_REL}", "--cells", ",".join(cells),
                            "--out", str(run_local / "hf_timing_probe.json")],
                           env=env, capture_output=True, text=True, timeout=3600)
        (run_local / "hf_timing_probe.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr)
        summary["hf_probe_rc"] = r.returncode; summary["hf_probe_s"] = round(time.time() - t1, 1)
        mark(f"hf_probe_done_rc{r.returncode}")

    # 3. target standup on the target GPU
    tgt_proc, stop = _start_target(tree, srv, geom, gpu=target_gpu)
    ep = srv / "endpoint.json"
    t2 = time.time()
    while not ep.exists():
        if tgt_proc.poll() is not None:
            summary["status"] = "TARGET_STANDUP_FAILED"
            summary["standup_tail"] = (srv / "standup.log").read_text(errors="replace")[-4000:]
            vl = srv / "vllm_server.log"
            summary["vllm_tail"] = vl.read_text(errors="replace")[-4000:] if vl.exists() else None
            _kill(att_proc); _sync(syncer); return summary
        if time.time() - t2 > 3000:
            summary["status"] = "TARGET_STANDUP_TIMEOUT"; _kill(att_proc); _sync(syncer); return summary
        time.sleep(5)
    endpoint = json.loads(ep.read_text())
    proxy_url, rt = endpoint["proxy_url"], endpoint["routing_table"]
    summary["target_standup_s"] = round(time.time() - t2, 1); summary["proxy_url"] = proxy_url
    mark("target_up")

    # 4. attacker healthy? (a pool was verified up front; re-probe it now)
    #    A per-job attacker must ALSO have its verified serving manifest
    #    (weights bf16|fp8, prefix caching, KV dtype -- checked against the
    #    engine's own config line): the drivers are given it, so their
    #    attacker_engine.json can be compared by the aggregate. No manifest,
    #    no drivers: the run ends ATTACKER_SERVING_UNVERIFIED.
    serving_manifest: Path | None = None
    if att_proc is not None:
        try:
            summary["attacker_ready_s"] = round(_wait_health(f"http://127.0.0.1:{attacker_port}/health",
                                                             att_proc, 3000, "attacker vLLM", att_log), 1)
        except RuntimeError as e:
            summary["status"] = "ATTACKER_FAILED"; summary["attacker_err"] = str(e)[-4000:]
            stop.touch(); _sync(syncer); return summary
        try:
            serving_manifest = _wait_attacker_serving_manifest(att_log, att_proc)
        except AttackerServingUnverified as e:
            summary["status"] = "ATTACKER_SERVING_UNVERIFIED"; summary["attacker_err"] = str(e)[-4000:]
            print(f"[smoke] FATAL: {summary['attacker_err']}", flush=True)
            stop.touch(); _kill(att_proc); _sync(syncer); return summary
        summary["attacker_serving_manifest"] = str(serving_manifest)
        summary["attacker_serving"] = json.loads(serving_manifest.read_text()).get("requested")
    elif not _http_ok(f"{attacker_pool['url']}/health"):
        summary["status"] = "ATTACKER_POOL_DIED"; stop.touch(); _sync(syncer); return summary
    mark("attacker_up")

    # 5. uncontended served timing probe (same prompts as the HF probe)
    try:
        import yaml
        max_new = int(yaml.safe_load(open(f"{tree}/{CFG_REL}"))["attacks"]["target_max_new_tokens"])
        _served_timing_probe(tree, proxy_url, rt, run_local / "served_timing_probe.json", max_new)
    except Exception as e:  # noqa: BLE001
        summary["served_probe_err"] = f"{type(e).__name__}: {e}"
    mark("served_probe_done")
    _sync(syncer)

    # 6. the drivers: every (arm x attack x cell x subshard), concurrently
    procs: dict[str, subprocess.Popen] = {}
    logs: dict[str, Path] = {}
    env = _env(tree)
    for arm in arms:
        for attack in attacks:
            for cell in cells:
                for sub in range(subshards):
                    key = f"{arm}/{attack}/{cell}/shard{sub:02d}"
                    log = run_local / "driver_logs" / f"{key.replace('/', '__')}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    cmd = _driver_cmd(tree, attack=attack, cell=cell, arm=arm, sub=sub,
                                      subshards=subshards, ids_path=ids_path,
                                      private_root=str(run_local / arm), proxy_url=proxy_url,
                                      rt=rt, attacker_port=attacker_port,
                                      target_window=geom.max_model_len,
                                      attacker_pool=(str(pool_json) if pool_json else None),
                                      attacker_serving=(str(serving_manifest) if serving_manifest else None))
                    procs[key] = subprocess.Popen(cmd, env=env, stdout=log.open("w"),
                                                  stderr=subprocess.STDOUT)
                    logs[key] = log
    (run_local / "driver_cmds.json").write_text(json.dumps(
        {k: _driver_cmd(tree, attack=k.split("/")[1], cell=k.split("/")[2], arm=k.split("/")[0],
                        sub=int(k.split("shard")[-1]), subshards=subshards, ids_path=ids_path,
                        private_root=str(run_local / k.split("/")[0]), proxy_url=proxy_url, rt=rt,
                        attacker_port=attacker_port, target_window=geom.max_model_len,
                        attacker_pool=(str(pool_json) if pool_json else None),
                        attacker_serving=(str(serving_manifest) if serving_manifest else None))
         for k in procs}, indent=1))
    mark(f"drivers_spawned_{len(procs)}")

    def progress() -> dict:
        st = {}
        for key in procs:
            arm, attack, cell, shard = key.split("/")
            pj = run_local / arm / PRIVATE_RUN / attack / cell / shard / "progress.jsonl"
            c = progress_counts(pj)
            st[key] = {"goals_done": c["done"], "errored": c["errored"], "rows": c["rows"],
                       "rc": procs[key].poll()}
        return st

    # Sync is EVENT-based: every time the progress row count changes (a goal
    # reached a terminal record) the changed files ship to the volume and are
    # committed, so a container kill loses at most the goals in flight — plus a
    # timer floor (sync_every_s) for the per-call logs of long goals.
    last_sync = time.time(); last_print = 0.0; last_rows = -1; last_pool_probe = 0.0
    while any(p.poll() is None for p in procs.values()):
        if tgt_proc.poll() is not None:
            print("[smoke] FATAL: target standup died mid-run", flush=True); summary["status"] = "TARGET_DIED"; break
        if att_proc is not None and att_proc.poll() is not None:
            print("[smoke] FATAL: attacker vLLM died mid-run", flush=True); summary["status"] = "ATTACKER_DIED"; break
        if att_proc is None and time.time() - last_pool_probe > 120:
            last_pool_probe = time.time()
            if not _http_ok(f"{attacker_pool['url']}/health"):
                print("[smoke] FATAL: the shared attacker pool stopped answering /health", flush=True)
                summary["status"] = "ATTACKER_POOL_DIED"; break
        st = progress()
        rows = sum(v["rows"] for v in st.values())
        if time.time() - last_print > 180:
            done = sum(v["goals_done"] for v in st.values())
            err = sum(v["errored"] for v in st.values())
            fin = sum(1 for v in st.values() if v["rc"] is not None)
            bad = [k for k, v in st.items() if v["rc"] not in (None, 0)]
            print(f"[t+{time.time() - T0:.0f}s] goals_done={done} errored={err} rows={rows} "
                  f"drivers_finished={fin}/{len(procs)} nonzero_rc={bad}", flush=True)
            last_print = time.time()
        if rows != last_rows or time.time() - last_sync > sync_every_s:
            _sync(syncer); last_sync = time.time(); last_rows = rows
        time.sleep(15)
    for p in procs.values():
        if p.poll() is None:
            p.kill()
    summary["drivers"] = progress()
    summary["drivers_wall_s"] = round(time.time() - T0 - summary["timeline"].get(f"drivers_spawned_{len(procs)}", 0), 1)
    summary.setdefault("status", "DONE" if all(v["rc"] == 0 for v in summary["drivers"].values()) else "DRIVER_ERRORS")
    mark("drivers_done")

    # 7. teardown + final sync
    stop.touch()
    try:
        tgt_proc.wait(timeout=300)
    except subprocess.TimeoutExpired:
        tgt_proc.kill()
    _kill(att_proc)          # the pool is NOT ours to stop: only a per-job attacker is killed
    for name in ("vllm_server.log", "standup.log", "attacker_vllm.log"):
        pass  # already under srv/ (synced with the tree)
    summary["total_wall_s"] = round(time.time() - T0, 1)
    (run_local / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    _sync(syncer)
    print("[smoke] SUMMARY", json.dumps({k: v for k, v in summary.items() if k != "drivers"}, default=str), flush=True)
    return summary


def _kill(proc) -> None:
    if proc is not None and proc.poll() is None:
        proc.kill()


def _provenance(*, run_id, sha, geom, goal_ids, attempts, now_iso, resume, summary,
                pool_rec) -> dict:
    """run_provenance.json: the per-job attacker block (``attacker_qwen``) OR the
    shared pool's record (``attacker_pool`` = endpoint.json contents + the
    verification evidence; ``attacker_qwen`` then names the pool as its
    source), and the topology that matches."""
    prov = {
        "run_id": run_id, "tree_sha": sha, "config": CFG_REL, "cell": CELL,
        "object_dir": f"{OBJECTS_X1}/{CELL}", "object_sha256": OBJECT_SHA256,
        "pair_n_streams": 5, "pair_depth": 3, "pair_budget_commit": PAIR_BUDGET_COMMIT,
        "attacker_grok": {"model": GROK_MODEL, "endpoint": "https://openrouter.ai/api",
                          "scorer": "qwen (--scorer-endpoint)"},
        "target_geometry": geom.as_dict(),
        "goal_ids": goal_ids,
        "started_at": attempts[0].get("started_at") or now_iso,
        "attempts": attempts,
        "resume": ({k: v for k, v in summary["resume"].items() if k != "attempts"}
                   if resume else None)}
    if pool_rec is None:
        prov["attacker_qwen"] = {"model": ATTACKER_MODEL, "snapshot": ATTACKER_SNAPSHOT,
                                 "max_model_len": ATTACKER_MAX_MODEL_LEN, "thinking": "off (a328caf)"}
        prov["attacker_pool"] = None
        prov["topology"] = ("1 container, 2xH200: GPU0 attacker vLLM, GPU1 target vLLM+lens+"
                            "RoutingProxy, drivers on loopback")
    else:
        ident = pool_rec["identity"]
        prov["attacker_qwen"] = {"model": ident["model"], "snapshot": ident["snapshot"],
                                 "max_model_len": ident["max_model_len"],
                                 "thinking": "off (a328caf)", "source": "shared attacker pool"}
        prov["attacker_pool"] = pool_rec
        prov["topology"] = ("1 container, 1xH200: GPU0 target vLLM+lens+RoutingProxy, drivers on "
                            f"loopback; attacker = shared Modal pool {ident['url']} "
                            f"(server {ident['job_id']}, container {ident['node']})")
    return prov


@app.function(image=image, volumes={VOL_PATH: VOL}, gpu="H200:2", cpu=16,
              memory=131072, timeout=12 * 3600, secrets=[openrouter_secret])
def run_smoke(sha: str, goal_ids: list[str], run_id: str, arms: list[str],
              subshards: int = 3, hf_probe: bool = True, target_seqs: int = 0,
              attacks: list[str] = list(ATTACKS), cells: list[str] = list(CELLS),
              sync_every_s: int = 120, resume: bool = False,
              target_window: int = 0, allow_over_budget: bool = False,
              target_key: str = TARGET_KEY) -> dict:
    """The per-job topology (2xH200: attacker + target), unchanged."""
    return _run_smoke_impl(sha, goal_ids, run_id, arms, subshards=subshards, hf_probe=hf_probe,
                           target_seqs=target_seqs, attacks=attacks, cells=cells,
                           sync_every_s=sync_every_s, resume=resume, target_window=target_window,
                           allow_over_budget=allow_over_budget, target_key=target_key,
                           attacker_pool=None)


@app.function(image=image, volumes={VOL_PATH: VOL}, gpu="H200:1", cpu=16,
              memory=131072, timeout=12 * 3600,
              secrets=[openrouter_secret, attacker_pool_secret])
def run_smoke_pooled(sha: str, goal_ids: list[str], run_id: str, arms: list[str],
                     attacker_pool: dict, subshards: int = 3, hf_probe: bool = True,
                     target_seqs: int = 0, attacks: list[str] = list(ATTACKS),
                     cells: list[str] = list(CELLS), sync_every_s: int = 120,
                     resume: bool = False, target_window: int = 0,
                     allow_over_budget: bool = False, target_key: str = TARGET_KEY) -> dict:
    """The POOLED topology: ONE H200 for the target; the attacker is the shared
    pool (verified before the target standup; its bearer arrives via the
    ephemeral secret)."""
    return _run_smoke_impl(sha, goal_ids, run_id, arms, subshards=subshards, hf_probe=hf_probe,
                           target_seqs=target_seqs, attacks=attacks, cells=cells,
                           sync_every_s=sync_every_s, resume=resume, target_window=target_window,
                           allow_over_budget=allow_over_budget, target_key=target_key,
                           attacker_pool=attacker_pool)


# --------------------------------------------------------------------------- #
# local entrypoint                                                             #
# --------------------------------------------------------------------------- #

@app.local_entrypoint()
def main(sha: str = "", ids: str = "", run: str = "", stage: bool = False,
         pf: bool = False, go: bool = False, arms: str = "grok,qwen",
         subshards: int = 3, no_hf_probe: bool = False, target_seqs: int = 0,
         attacks: str = ",".join(ATTACKS), resume: bool = False, sync_every_s: int = 120,
         target_window: int = 0, allow_over_budget: bool = False,
         target_key: str = TARGET_KEY, attacker_pool: str = ""):
    # LOUD HERE, not at import: the key that rides into every container was
    # resolved at import time to build the Secret, and an empty one only
    # surfaces as a 401 deep inside a running container. Re-resolve strictly
    # before anything is spawned, so "no key configured" is a named error on
    # this machine.
    try:
        _openrouter_key(required=True)
    except Exception as exc:
        raise SystemExit(
            f"[grokmt] no OpenRouter key: {exc} — export OPENROUTER_API_KEY "
            "or make the vault item readable; refusing to launch with an "
            "empty secret") from None
    if attacker_pool and attacker_pool != _POOL_ARG:
        # the bearer secret is built at import from GROKMT_ATTACKER_POOL; the
        # flag alone cannot mint it after the fact (Modal serializes the app
        # before main runs), so the two must agree — refuse, don't guess
        raise SystemExit("--attacker-pool must ALSO be exported as GROKMT_ATTACKER_POOL=<same spec> "
                         "for `modal run` (the pool's bearer is attached as an ephemeral secret "
                         "at import time)")
    if stage:
        print("[grokmt] staging the 27B attacker into", ATTACKER_HF)
        print(json.dumps(stage_attacker.remote(), indent=1))
        if not (pf or go):
            return
    if not sha:
        raise SystemExit("pass --sha <tree sha staged at /vol/grokmt/repos/sisteer-<sha>>")
    goal_ids = []
    if ids:
        goal_ids = [l.strip() for l in Path(ids).read_text().splitlines()
                    if l.strip() and not l.startswith("#")]
    if pf:
        r = preflight.remote(sha, goal_ids, qwen=("qwen" in arms))
        print(json.dumps(r, indent=1, default=str))
        if not r.get("ok"):
            raise SystemExit("[grokmt] PREFLIGHT FAILED — fix all above before any GPU")
        if not go:
            return
    if go:
        if not run:
            raise SystemExit("pass --run <run_id>")
        if not goal_ids:
            raise SystemExit("pass --ids <goal_ids.txt>")
        arm_list = [a for a in arms.split(",") if a]
        # the geometry the run will ask for, resolved locally FIRST so an
        # over-budget override refuses before a GPU is touched (the container
        # re-resolves from the staged tree — that resolution is the one used)
        geometry = _geometry_module(None)
        try:
            geom = resolve_target_geometry(None, target_window=target_window,
                                           target_seqs=target_seqs,
                                           allow_over_budget=allow_over_budget,
                                           target_key=target_key)
        except (geometry.GeometryOverBudget, geometry.UnknownModelKey) as e:
            raise SystemExit(f"[grokmt] refusing the target geometry: {e}")
        print(geom.describe(), flush=True)
        pool = _pool_spec_from_arg(attacker_pool)      # a local read; refuses a bad spec here
        if pool and not _pool_api_key(attacker_pool):
            raise SystemExit("[grokmt] --attacker-pool: no bearer found (the endpoint's key file "
                             "is not readable here and $VLLM_API_KEY is unset); refusing")
        print(f"[grokmt] launching run {run} @ {sha[:7]} arms={arm_list} "
              f"subshards={subshards} attacks={attacks} resume={resume} "
              f"app={app.name} sync_every_s={sync_every_s} "
              f"target={geom.max_model_len}x{geom.max_num_seqs} "
              f"attacker={'POOL ' + pool['url'] if pool else 'per-job vLLM (2xH200)'}", flush=True)
        common = dict(subshards=subshards, hf_probe=not no_hf_probe, target_seqs=target_seqs,
                      attacks=[a for a in attacks.split(",") if a],
                      sync_every_s=sync_every_s, resume=resume, target_window=target_window,
                      allow_over_budget=allow_over_budget, target_key=target_key)
        if pool:
            s = run_smoke_pooled.remote(sha, goal_ids, run, arm_list, pool, **common)
        else:
            s = run_smoke.remote(sha, goal_ids, run, arm_list, **common)
        print(json.dumps({k: v for k, v in s.items() if k != "drivers"}, indent=1, default=str))
