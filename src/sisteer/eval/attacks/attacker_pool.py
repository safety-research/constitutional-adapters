"""The SHARED ATTACKER POOL: one Modal Server running vLLM for the unsteered
27B attacker, serving many attack drivers over HTTPS.

Why. The served multi-turn lane (scripts/modal/grokmt_served.py) historically
stood the attacker up PER JOB on its own GPU, where
it idles at a mean of ~0.6 requests in flight: a PyRIT driver makes ONE
attacker call at a time and spends most of each turn waiting on the target.
A pool turns N attacker GPUs into one, kept busy by every driver at once.

This module is the pool's CONTRACT, shared by the three sides:

- the pool server (``scripts/modal/attacker_pool.py``) sizes its geometry
  here (:func:`resolve_pool_geometry` — chunked prefill at a bounded batch
  with the KV budget covering ``S x L`` on one H200, refusing a geometry
  that does not fit; the no-chunk identity ``max_num_batched_tokens = S x L``
  is modelled too and is unfittable for this attacker at any useful S — see
  :data:`PROFILE_ACTIVATION_BYTES_PER_TOKEN`), mints a per-launch bearer
  token (:func:`mint_api_key`, written 0600 by :func:`write_api_key`), and
  advertises ``endpoint.json`` (:class:`PoolEndpoint`) in its pool dir;
- a consumer (``scripts/jailbreak_run.py --attacker-pool``, the grokmt
  launcher, the canary) reads that file (:func:`read_endpoint`), takes the
  token from the pool's key file or the env (:func:`load_api_key`), and
  VERIFIES the pool before it touches a target GPU (:func:`connect`:
  the recorded Modal Server object still resolves to the same URL, ``/health``
  is up, and ``/v1/models`` refuses an unauthenticated call then, with the
  bearer, lists the expected model at
  the expected snapshot and window). Every failure is a typed
  :class:`AttackerPoolError` — a LOUD refusal, never a fallback to a local
  attacker or to HF (the codebase rule: a silent fallback mislabels a run);
- the run manifest records :meth:`PoolEndpoint.identity` (model + snapshot
  + geometry + job) so results stay attributable to the attacker that made
  them.

Security posture (Modal). The Modal Server URL is public at the transport
layer so other Modal apps can reach it; the bearer token is the application
guard (vLLM's stock ``VLLM_API_KEY`` middleware over ``/v1*`` — the same
variable and client-side header the serving code uses). The token never
appears on argv or in endpoint.json. A launcher on the grading host keeps it
in a mode-0600 state file and attaches it as an ephemeral Modal Secret;
consumers refuse a local key file that is group/world-readable.
Unauthenticated paths are vLLM's liveness endpoints and ``/metrics``
(aggregate counters, no content).

Deliberately dependency-light (stdlib + ``requests`` at call time): it is
imported from the vLLM serving venv (no torch, no sisteer deps) via
PYTHONPATH as well as from the drivers' venv.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
import stat
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# constants                                                                     #
# --------------------------------------------------------------------------- #

SCHEMA_VERSION = 2
ENDPOINT_FILE = "endpoint.json"
API_KEY_FILE = "api_key"
#: the env var the server AND every client agree on — vLLM's stock guard reads
#: it (``vllm serve`` without ``--api-key`` on argv), and it equals
#: ``sisteer.capeval.serving.vllm_auth.TOKEN_ENV`` (a test pins the equality).
API_KEY_ENV = "VLLM_API_KEY"
#: the env name a DRIVER container carries the attacker-pool bearer under -- DISTINCT from
#: :data:`API_KEY_ENV`, which in a driver container is the TARGET proxy's bearer (when any). The pool
#: app renders ONE bearer under ``$XDG_RUNTIME_DIR`` and delivers it to the POOL containers as
#: ``VLLM_API_KEY`` and to the DRIVER containers under this name, by the same Modal-secret transport;
#: ``jailbreak_run._vllm_api_key`` reads it ONLY for the attacker endpoint (``--attacker-endpoint``),
#: never for a scorer endpoint or a per-job server. ``scripts/modal/seedrep_served.py`` cannot import
#: this module inside its container (only the launch plan ships beside it), so it spells the same
#: literal; ``tests/test_seedrep_fw_pool_bearer.py`` pins the equality, as the ``vllm_auth.TOKEN_ENV``
#: pin above does for :data:`API_KEY_ENV`.
POOL_BEARER_ENV = "SEEDREP_ATTACKER_POOL_KEY"
#: the sentinel a pool watches for a deliberate shutdown (``touch <POOL_DIR>/STOP``)
STOP_FILE = "STOP"

#: the pinned attacker (== pyrit_adapters.ATTACKER_MODEL / ATTACKER_SNAPSHOT; a test pins it)
ATTACKER_MODEL = "huihui-ai/Huihui-Qwen3.6-27B-abliterated"
ATTACKER_SNAPSHOT = "27502c8717fd5a2f8c0c77188c10c243fd4f672e"

#: the pool's DEFAULT window: measured on the attacker's own calls (attacker +
#: in-loop scorer roles, PAIR + Crescendo + ActorAttack, 22,661 calls from the
#: 2026-09-04 served qwen35_4b multi-turn wave, 135 shards x 3 attacks) with
#: the sizing rule of ``sisteer.capeval.serving.geometry``: p99 of prompt +
#: completion = 18,973 for the attacker role -> next 256-multiple 19,200
#: (clamp rate at L 1.44% < 5%, so the p99_total branch; 0.87% of calls exceed
#: it — Crescendo's 10-turn tail, whose own window would be 20,480). The
#: scorer role is 768. The measured basis — both roles' call counts, p99 and
#: max, the clamp rate and the share over L, the 20,480 and 24,576
#: alternatives — is the window-sizing table of :data:`MEASUREMENT_DOC`
#: (produced by scripts/attacker_pool_profile.py); a test pins the literals
#: below to that table.
DEFAULT_MAX_MODEL_LEN = 19200
MEASURED_ATTACKER_P99_TOTAL = 18973
MEASURED_ATTACKER_P99_PROMPT = 18718
MEASURED_ATTACKER_MAX_TOTAL = 23752
MEASURED_ATTACKER_N_CALLS = 10147
MEASURED_CLAMP_RATE_AT_L = 0.0144
MEASURED_FRAC_OVER_L = 0.0087
MEASURED_ON = "2026-09-04"
#: the doc whose window-sizing table carries the measured basis of the literals above
MEASUREMENT_DOC = "docs/attacker_pool.md"
#: Crescendo alone (the longest of the three): its own rule window
CRESCENDO_MAX_MODEL_LEN = 20480
#: covers every observed attacker call (max total 23,752): the zero-tail alternative
COVER_MAX_MAX_MODEL_LEN = 24576
DEFAULT_GPU_MEMORY_UTILIZATION = 0.90
#: requests are refused above this many sequences however much would fit
#: (vLLM's own scheduler ceiling for a sane request queue)
MAX_SEQS_CEILING = 512

REQUIRED_KEYS: Tuple[str, ...] = (
    "url", "model", "snapshot", "max_model_len", "max_num_seqs",
    "api_key_env_name", "job_id", "node", "started_at",
)
_URL_RE = re.compile(r"^https?://[^/\s]+(:\d+)?/?$")
#: docker-bridge (172.16/12) and loopback ranges are never routable cross-node;
#: a pool that advertised `hostname -I | awk '{print $1}'` = the bridge
#: address served ZERO requests.
_NON_ROUTABLE = re.compile(r"^(127\.|172\.(1[6-9]|2\d|3[01])\.|0\.0\.0\.0$)")


# --------------------------------------------------------------------------- #
# errors: every one a refusal                                                   #
# --------------------------------------------------------------------------- #

class AttackerPoolError(RuntimeError):
    """A shared attacker pool could not be used. Never caught to fall back:
    a run that asked for the pool either uses THAT pool or does not run."""

    hint = ("NOT falling back to a per-job attacker or to HF: the run asked "
            "for the pool, and a silent substitute would mislabel its results.")

    def __init__(self, msg: str) -> None:
        super().__init__(f"{msg}\n{self.hint}")


class PoolAbsent(AttackerPoolError):
    """No endpoint.json (pool not up / never started / already torn down)."""


class PoolStale(AttackerPoolError):
    """endpoint.json names a server invocation that is no longer live."""


class PoolUnreachable(AttackerPoolError):
    """The advertised URL does not answer /health (network or server down)."""


class PoolUnauthorized(AttackerPoolError):
    """The bearer we hold was refused (401/403) on an authenticated route."""


class PoolUnguarded(AttackerPoolError):
    """The pool answered an UNauthenticated /v1/models — its guard is off.
    Refused: a public Modal URL without the bearer guard is not the contract."""


class PoolMismatch(AttackerPoolError):
    """The live server is not the advertised attacker (model / snapshot /
    window differ from endpoint.json or from what the run expects)."""


class PoolGeometryUnfit(AttackerPoolError):
    """The requested S x L does not fit the attacker's memory on one H200."""


class PoolUnattested(AttackerPoolError):
    """The record's serving identity cannot be copied: it carries no
    engine-attested ``serving`` block at all (stale, stripped or malformed), it
    DECLARES a precision without one (the first deployed pool's literal-only
    record), or the block is unattested, incomplete, or contradicts its own
    recorded engine line. Checked FIRST on every consumer
    path -- before any liveness lookup, health probe, bearer check, HF timing
    probe or target standup -- so a refusal costs nothing."""


class PoolKeyUnreadable(AttackerPoolError):
    """The key file is missing, or readable by others (guard void)."""


# --------------------------------------------------------------------------- #
# endpoint.json                                                                 #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PoolEndpoint:
    """What a live pool advertises. ``url`` is the server root
    (``http://<routable ip>:<port>``); ``model`` the served name the client
    sends; ``snapshot`` the HF snapshot sha the served path resolves to;
    ``max_model_len`` / ``max_num_seqs`` / ``max_num_batched_tokens`` the
    launched geometry; ``api_key_env_name`` the variable a consumer exports
    the bearer under (``api_key_file`` is where the pool wrote it, 0600);
    ``job_id`` / ``node`` / ``started_at`` remain the stable identity fields
    (on Modal they carry the server invocation/object id and container id).
    ``extra["liveness"]`` carries the authoritative runtime-specific identity;
    the remaining extra fields hold vLLM version, KV coverage and log paths."""

    url: str
    model: str
    snapshot: str
    max_model_len: int
    max_num_seqs: int
    api_key_env_name: str
    job_id: str
    node: str
    started_at: str
    max_num_batched_tokens: Optional[int] = None
    api_key_file: Optional[str] = None
    gpu_memory_utilization: Optional[float] = None
    prefix_caching: bool = True
    pool_name: Optional[str] = None
    pool_dir: Optional[str] = None
    schema_version: int = SCHEMA_VERSION
    source: str = "endpoint.json"
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], *, source: str = "endpoint.json") -> "PoolEndpoint":
        if not isinstance(d, dict):
            raise PoolAbsent(f"endpoint record is not a JSON object ({type(d).__name__})")
        missing = [k for k in REQUIRED_KEYS if k not in d or d[k] in (None, "")]
        if missing:
            raise PoolAbsent(f"endpoint record is missing required key(s) {missing}; "
                             f"present: {sorted(d)}")
        url = str(d["url"]).rstrip("/")
        if not _URL_RE.match(url):
            raise PoolAbsent(f"endpoint url {url!r} is not http(s)://host[:port]")
        host = url.split("//", 1)[1].split(":")[0]
        if _NON_ROUTABLE.match(host):
            raise PoolAbsent(f"endpoint url {url!r} is not routable cross-node "
                             "(loopback or docker-bridge address)")
        for k in ("max_model_len", "max_num_seqs"):
            v = d[k]
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise PoolAbsent(f"endpoint {k} must be a positive int, got {v!r}")
        known = {f.name for f in cls.__dataclass_fields__.values()} - {"extra", "source"}
        extra = {k: v for k, v in d.items() if k not in known and k != "source"}
        kw = {k: v for k, v in d.items() if k in known}
        kw["url"] = url
        kw["job_id"] = str(d["job_id"])
        mnbt = d.get("max_num_batched_tokens")
        kw["max_num_batched_tokens"] = int(mnbt) if mnbt is not None else None
        # a record round-tripped through as_dict() carries its own source
        return cls(source=str(d.get("source") or source), extra=extra, **kw)

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra")
        d.update(extra)
        return d

    def identity(self) -> Dict[str, Any]:
        """The attributable subset a run manifest records: WHICH attacker,
        at WHAT geometry, from WHICH job — enough to tell two pools apart
        and to match a transcript to the server that produced it."""
        return {
            "model": self.model, "snapshot": self.snapshot,
            "max_model_len": self.max_model_len, "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "prefix_caching": self.prefix_caching,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "url": self.url, "job_id": self.job_id, "node": self.node,
            "started_at": self.started_at, "pool_name": self.pool_name,
            "pool_dir": self.pool_dir, "source": self.source,
            "liveness": self.extra.get("liveness"),
            # the engine's deployment shape (F-M): cards, allocator config, engine version -- what a certificate must keep
            "gpu": self.extra.get("gpu"), "tensor_parallel_size": self.extra.get("tensor_parallel_size"),
            "pytorch_cuda_alloc_conf": self.extra.get("pytorch_cuda_alloc_conf"),
            "vllm_version": self.extra.get("vllm_version"),
        }


def endpoint_path(pool: str | os.PathLike) -> Path:
    """``<POOL_DIR>/endpoint.json`` for a directory; the file itself when given."""
    p = Path(pool)
    return p if p.suffix == ".json" else p / ENDPOINT_FILE


def is_url(spec: str) -> bool:
    return bool(spec) and spec.startswith(("http://", "https://"))


def read_endpoint(pool: str | os.PathLike) -> PoolEndpoint:
    """Read and validate a pool's endpoint.json (:class:`PoolAbsent` when it
    is missing, torn, or malformed)."""
    p = endpoint_path(pool)
    if not p.is_file():
        raise PoolAbsent(f"no attacker pool endpoint at {p} — the pool is not up "
                         "(deploy scripts/modal/attacker_pool.py and wait for "
                         "endpoint.json), or the pool directory/file is wrong")
    try:
        d = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise PoolAbsent(f"attacker pool endpoint {p} is unreadable/torn: "
                         f"{type(e).__name__}: {e}") from e
    ep = PoolEndpoint.from_dict(d)
    if ep.pool_dir is None and p.suffix == ".json":
        ep = PoolEndpoint(**{**asdict(ep), "pool_dir": str(p.parent)})
    return ep


def endpoint_from_url(url: str, *, model: str = ATTACKER_MODEL,
                      snapshot: str = ATTACKER_SNAPSHOT) -> PoolEndpoint:
    """Refuse a bare URL: it cannot carry attributable geometry or liveness.

    The former URL path fabricated ``max_model_len=1`` / ``max_num_seqs=1`` and
    then skipped the live window check. That made the manifest claim an identity
    no server had. A caller must copy/fetch the pool's endpoint.json instead;
    :func:`verify_pool` checks its real window against the live model card."""
    del model, snapshot
    raise PoolAbsent(
        f"bare attacker pool URL {url!r} is not an endpoint identity. Pass the "
        "pool's endpoint.json (or a directory containing it); URL-only specs "
        "cannot prove model snapshot, geometry, bearer source, or Modal liveness")


def resolve_pool(spec: str) -> PoolEndpoint:
    """``--attacker-pool <POOL_DIR | endpoint.json>`` -> the full endpoint."""
    if is_url(spec):
        return endpoint_from_url(spec)
    return read_endpoint(spec)


# --------------------------------------------------------------------------- #
# the bearer                                                                    #
# --------------------------------------------------------------------------- #

def mint_api_key() -> str:
    """A per-launch random token — the serving code's recipe
    (``vllm_server.build_server_launch``: ``secrets.token_urlsafe(32)``)."""
    return secrets.token_urlsafe(32)


def _mode_is_private(path: Path) -> bool:
    st = path.stat()
    return (st.st_mode & (stat.S_IRWXG | stat.S_IRWXO)) == 0


def write_api_key(path: str | os.PathLike, key: str) -> Path:
    """Write the token to ``path`` with mode 0600 and refuse otherwise.

    A world-readable key voids the bearer guard. The Modal launcher uses this
    for its grading-host-only state file; old cluster records used the same contract."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        p.unlink()
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(fd, key.encode("utf-8"))
    finally:
        os.close(fd)
    os.chmod(p, 0o600)
    if not _mode_is_private(p):
        mode = oct(p.stat().st_mode & 0o777)
        p.unlink()
        raise PoolKeyUnreadable(
            f"{p}: the filesystem left the key file at mode {mode} after chmod 600; "
            "refusing to park a credential in world-readable storage")
    return p


def load_api_key(ep: PoolEndpoint, *, env: Optional[Dict[str, str]] = None,
                 require: bool = True) -> Optional[str]:
    """The bearer for ``ep``: the env var it names when set (a launcher may
    ship the token that way — Modal secret, exported var), else the pool's
    key file, which must be private (0600) or the read is REFUSED. ``None``
    only when ``require`` is False and neither source has it."""
    env = os.environ if env is None else env
    v = env.get(ep.api_key_env_name or API_KEY_ENV, "")
    if v:
        return v
    kf = ep.api_key_file
    if not kf and ep.pool_dir:
        kf = str(Path(ep.pool_dir) / API_KEY_FILE)
    if kf:
        p = Path(kf)
        if p.is_file():
            if not _mode_is_private(p):
                raise PoolKeyUnreadable(
                    f"attacker pool key file {p} is mode {oct(p.stat().st_mode & 0o777)} "
                    "(readable by group/others): the pool's guard is void; refusing")
            key = p.read_text().strip()
            if key:
                return key
        elif require:
            raise PoolKeyUnreadable(
                f"attacker pool key file {p} is missing and ${ep.api_key_env_name} is "
                "unset — the pool advertises a bearer-guarded endpoint; without the "
                "token every request would 401")
    if require:
        raise PoolKeyUnreadable(
            f"no bearer for the attacker pool: ${ep.api_key_env_name} is unset and the "
            "endpoint names no readable key file")
    return None


# --------------------------------------------------------------------------- #
# liveness + verification                                                       #
# --------------------------------------------------------------------------- #

def squeue_state(job_id: str, *, runner: Optional[Callable] = None) -> Optional[str]:
    """Legacy Slurm state lookup, retained only for the retired sbatch record.

    The clusters this path targeted no longer exist. New consumers never call
    this unless ``squeue=True`` is requested explicitly for an old record."""
    runner = runner or subprocess.run
    try:
        r = runner(["squeue", "-h", "-j", str(job_id), "-o", "%T"],
                   capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        raise PoolStale(f"cannot query legacy squeue for pool job {job_id}: "
                        f"{type(e).__name__}: {e}") from e
    out = (r.stdout or "").strip().splitlines()
    return out[0].strip() if out else None


def check_job_live(ep: PoolEndpoint, *, runner: Optional[Callable] = None) -> str:
    """Legacy refusal unless an old endpoint's Slurm job is RUNNING."""
    if ep.job_id in ("", "unknown"):
        raise PoolStale("the legacy pool endpoint names no Slurm job id")
    state = squeue_state(ep.job_id, runner=runner)
    if state != "RUNNING":
        raise PoolStale(
            f"legacy attacker pool job {ep.job_id} (node {ep.node}, started "
            f"{ep.started_at}) is {state or 'not in squeue'}: endpoint.json is stale")
    return state


def liveness_kind(ep: PoolEndpoint) -> str:
    """The endpoint's declared authoritative liveness mechanism."""
    rec = ep.extra.get("liveness")
    if isinstance(rec, dict) and rec.get("kind"):
        return str(rec["kind"])
    return "legacy_slurm"


def check_modal_live(ep: PoolEndpoint, *, lookup: Optional[Callable] = None,
                     app_lookup: Optional[Callable] = None,
                     call_live: Optional[Callable] = None) -> Dict[str, Any]:
    """Prove the recorded Modal app, call and Server still own this URL.

    The three callables are test seams. At runtime they resolve
    ``modal.Server.from_name``, ``modal.App.lookup`` and poll the recorded
    ``modal.FunctionCall``. A healthy URL alone is insufficient: after a
    redeploy, a stale endpoint could otherwise describe an old invocation
    while traffic reaches a new one."""
    rec = ep.extra.get("liveness")
    if not isinstance(rec, dict) or rec.get("kind") != "modal":
        raise PoolStale("endpoint.json carries no Modal liveness identity")
    need = ("app_name", "app_id", "server_name", "server_id")
    missing = [k for k in need if not rec.get(k)]
    if missing:
        raise PoolStale(f"endpoint.json Modal liveness is missing {missing}")
    if lookup is None:
        try:
            import modal

            lookup = lambda app_name, server_name: modal.Server.from_name(  # noqa: E731
                app_name, server_name)
            app_lookup = app_lookup or modal.App.lookup

            if call_live is None:
                def call_live(call_id: str) -> bool:
                    call = modal.FunctionCall.from_id(call_id)
                    try:
                        call.get(timeout=0)
                    except modal.exception.TimeoutError:
                        return True
                    return False
        except Exception as e:  # noqa: BLE001
            raise PoolStale(f"cannot import Modal to verify the pool identity: "
                            f"{type(e).__name__}: {e}") from e
    try:
        server = lookup(str(rec["app_name"]), str(rec["server_name"]))
        hydrate = getattr(server, "hydrate", None)
        if hydrate is not None:
            hydrate()
        actual_id = str(getattr(server, "object_id", "") or "")
        actual_url = str(server.get_url() or "").rstrip("/")
    except Exception as e:  # noqa: BLE001
        raise PoolStale(
            f"Modal cannot resolve deployed server {rec['app_name']}/{rec['server_name']}: "
            f"{type(e).__name__}: {e}") from e
    if actual_id != str(rec["server_id"]):
        raise PoolStale(
            f"Modal server {rec['app_name']}/{rec['server_name']} now has object id "
            f"{actual_id!r}, not endpoint.json's {rec['server_id']!r}; the record is stale")
    if actual_url != ep.url.rstrip("/"):
        raise PoolStale(
            f"Modal server {rec['app_name']}/{rec['server_name']} now resolves to "
            f"{actual_url!r}, not endpoint.json's {ep.url!r}; the record is stale")

    actual_app_id = str(rec["app_id"] or "")
    if app_lookup is not None:
        try:
            actual_app = app_lookup(str(rec["app_name"]))
            actual_app_id = str(getattr(actual_app, "app_id", "") or "")
        except Exception as e:  # noqa: BLE001
            raise PoolStale(
                f"Modal cannot resolve deployed app {rec['app_name']}: "
                f"{type(e).__name__}: {e}") from e
        if actual_app_id != str(rec["app_id"]):
            raise PoolStale(
                f"Modal app {rec['app_name']} now has id {actual_app_id!r}, not "
                f"endpoint.json's {rec['app_id']!r}; the record is stale")

    call_id = str(rec.get("call_id") or "")
    call_state = "not-recorded"
    if call_id and call_live is not None:
        try:
            is_live = bool(call_live(call_id))
        except Exception as e:  # noqa: BLE001
            raise PoolStale(
                f"Modal cannot poll recorded server call {call_id}: "
                f"{type(e).__name__}: {e}") from e
        if not is_live:
            raise PoolStale(
                f"Modal server call {call_id} has completed; endpoint.json is stale")
        call_state = "running"
    elif call_id:
        call_state = "unchecked-test-seam"

    return {"kind": "modal", "state": "deployed", "app_name": rec["app_name"],
            "server_name": rec["server_name"], "server_id": actual_id,
            "app_id": actual_app_id, "call_id": rec.get("call_id"),
            "call_state": call_state, "container_id": rec.get("container_id"),
            "url": actual_url}


def _get(session, url: str, *, headers: Optional[dict], timeout: float):
    try:
        return session.get(url, headers=headers, timeout=timeout)
    except Exception as e:  # noqa: BLE001 — any transport failure is unreachable
        raise PoolUnreachable(f"GET {url} failed: {type(e).__name__}: {e}") from e


def verify_pool(ep: PoolEndpoint, api_key: str, *, session=None, timeout: float = 15.0,
                expect_model: Optional[str] = None,
                expect_snapshot: Optional[str] = None) -> Dict[str, Any]:
    """The live checks, in order — each a typed refusal:

    1. ``GET /health`` answers 2xx (:class:`PoolUnreachable` — the first thing
       a cross-node consumer proves is that the routable address is reachable);
    2. ``GET /v1/models`` WITHOUT a bearer is refused with 401/403
       (:class:`PoolUnguarded` otherwise — the guard is the application-level
       isolation on the public Modal URL);
    3. ``GET /v1/models`` WITH the bearer is 2xx (:class:`PoolUnauthorized` on
       401/403) and lists ``expect_model`` (default the endpoint's model); the
       card's ``root`` (the path ``vllm serve`` was given) contains
       ``expect_snapshot`` (default the endpoint's), and its real
       ``max_model_len`` equals the advertised window (:class:`PoolMismatch`
       on any of these).
    Returns the evidence (statuses, the listed ids, root, window)."""
    if session is None:
        import requests
        session = requests.Session()
    base = ep.url.rstrip("/")
    ev: Dict[str, Any] = {"url": base, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    r = _get(session, f"{base}/health", headers=None, timeout=timeout)
    ev["health_status"] = st = getattr(r, "status_code", None)
    if st is None or not (200 <= st < 300):
        raise PoolUnreachable(f"attacker pool {base}/health answered {st} (expected 2xx): the "
                              "server is down, still loading, or the address is not routable "
                              "from this node")
    r = _get(session, f"{base}/v1/models", headers={"Authorization": None}, timeout=timeout)
    ev["unauthenticated_models_status"] = st = getattr(r, "status_code", None)
    if st is None or st not in (401, 403):
        raise PoolUnguarded(f"attacker pool {base}/v1/models answered {st} to an UNauthenticated "
                            "request (expected 401/403): the bearer guard is not enforced — "
                            "refusing to use an open attacker at a public Modal URL")
    r = _get(session, f"{base}/v1/models", headers={"Authorization": f"Bearer {api_key}"},
             timeout=timeout)
    ev["authenticated_models_status"] = st = getattr(r, "status_code", None)
    if st in (401, 403):
        raise PoolUnauthorized(f"attacker pool {base}/v1/models refused our bearer ({st}): the "
                               f"token in ${ep.api_key_env_name} / the pool key file is not this "
                               "pool's — a stale key from an earlier pool, or the wrong pool dir")
    if st is None or not (200 <= st < 300):
        raise PoolUnreachable(f"attacker pool {base}/v1/models answered {st} with the bearer "
                              "(expected 2xx)")
    try:
        data = r.json()
        cards = list(data.get("data") or [])
    except Exception as e:  # noqa: BLE001
        raise PoolMismatch(f"attacker pool {base}/v1/models returned a non-model-list body: "
                           f"{type(e).__name__}: {e}") from e
    ids = [str(c.get("id")) for c in cards]
    ev["models"] = ids
    want_model = expect_model or ep.model
    card = next((c for c in cards if str(c.get("id")) == want_model), None)
    if card is None:
        raise PoolMismatch(f"attacker pool {base} serves {ids}, not the expected attacker "
                           f"{want_model!r}")
    root = str(card.get("root") or "")
    ev["root"] = root
    want_snap = expect_snapshot or ep.snapshot
    if want_snap and want_snap not in root:
        raise PoolMismatch(f"attacker pool {base} serves {want_model!r} from root {root!r}, "
                           f"which does not resolve to the expected snapshot {want_snap} — "
                           "a different revision of the attacker, refusing")
    L = card.get("max_model_len")
    ev["max_model_len"] = L
    if L is None:
        raise PoolMismatch(f"attacker pool {base} model card omits max_model_len; cannot verify "
                           f"the advertised window {ep.max_model_len}")
    if int(L) != int(ep.max_model_len):
        raise PoolMismatch(f"attacker pool {base} reports max_model_len {L} but endpoint.json "
                           f"advertises {ep.max_model_len}: the file does not describe the "
                           "live server (a restarted pool over a stale file?)")
    ev["ok"] = True
    return ev


def check_serving_record(ep: PoolEndpoint) -> Dict[str, Any]:
    """Validate the record's serving identity the way the run driver will
    consume it (``attacker_serving.pool_driver_fields``: only an engine-attested
    block is copied, re-verified; a bare declaration and a record with no block
    at all are refused -- a shared pool must attest) and return those driver
    fields as evidence. A local, network-free check; raises
    :class:`PoolUnattested`."""
    from . import attacker_serving as serving

    try:
        return serving.pool_driver_fields(
            ep.extra.get("serving"),
            declared_quantization=ep.extra.get("attacker_quantization"),
            url=ep.url)
    except serving.AttackerServingError as exc:
        raise PoolUnattested(f"pool {ep.url} serving identity refused: {exc}") from exc


def connect_endpoint(ep: PoolEndpoint, *, session=None, squeue: bool = False,
                     runner: Optional[Callable] = None,
                     modal_lookup: Optional[Callable] = None,
                     modal_app_lookup: Optional[Callable] = None,
                     modal_call_live: Optional[Callable] = None,
                     env: Optional[Dict[str, str]] = None,
                     expect_model: Optional[str] = None,
                     expect_snapshot: Optional[str] = None,
                     pool_spec: Optional[str] = None) -> Tuple[PoolEndpoint, str, Dict[str, Any]]:
    """Check one already-resolved endpoint through the shared consumer path.

    Modal identity is always checked. ``squeue`` is an explicit opt-in for an
    old Slurm record only; it defaults off so copying endpoint.json into a Modal
    container can never shell out to a nonexistent Slurm client."""
    ev: Dict[str, Any] = {"pool_spec": pool_spec, "source": ep.source}
    # the serving identity first: it is local and free, and a record the driver
    # would refuse at its manifest write must not cost a liveness lookup, a
    # health probe, an HF timing probe or a target standup before that refusal
    ev["serving"] = check_serving_record(ep)
    kind = liveness_kind(ep)
    if kind == "modal":
        ev["liveness"] = check_modal_live(
            ep, lookup=modal_lookup, app_lookup=modal_app_lookup,
            call_live=modal_call_live)
        ev["job_state"] = "modal:deployed"  # compatibility for old manifest readers
    elif squeue:
        ev["job_state"] = check_job_live(ep, runner=runner)
        ev["liveness"] = {"kind": "legacy_slurm", "state": ev["job_state"]}
    else:
        ev["job_state"] = "unchecked-legacy"
        ev["liveness"] = {"kind": "legacy_slurm", "state": "unchecked"}
    key = load_api_key(ep, env=env)
    ev.update(verify_pool(ep, key, session=session, expect_model=expect_model,
                          expect_snapshot=expect_snapshot))
    return ep, key, ev


def connect(spec: str, *, session=None, squeue: bool = False,
            runner: Optional[Callable] = None,
            modal_lookup: Optional[Callable] = None,
            modal_app_lookup: Optional[Callable] = None,
            modal_call_live: Optional[Callable] = None,
            env: Optional[Dict[str, str]] = None,
            expect_model: Optional[str] = None,
            expect_snapshot: Optional[str] = None) -> Tuple[PoolEndpoint, str, Dict[str, Any]]:
    """Resolve + check liveness + bearer + live identity before target work."""
    ep = resolve_pool(spec)
    return connect_endpoint(
        ep, session=session, squeue=squeue, runner=runner,
        modal_lookup=modal_lookup, modal_app_lookup=modal_app_lookup,
        modal_call_live=modal_call_live, env=env,
        expect_model=expect_model, expect_snapshot=expect_snapshot,
        pool_spec=spec)


# --------------------------------------------------------------------------- #
# sizing: the attacker's memory on one H200 (chunked prefill, or the identity)  #
# --------------------------------------------------------------------------- #

GIB = 2 ** 30
#: the H200's usable device memory as vLLM sees it (geometry.py's Llama-8B basis
#: note: "on a 139.81 GiB card")
H200_BYTES = int(139.81 * GIB)
#: CUDA context + graphs + sampler workspace not covered by the terms below
RESERVE_BYTES = 2 * GIB
#: the startup profile's ACTIVATION peak per batched token, MEASURED (a lower
#: bound) from the 2026-09-04 Modal deploy of the identity geometry (dc065dd:
#: eager, 710,400 batched tokens, multimodal profiling on): torch held
#: 137.04 GiB including 51.08 GiB of weights when a further 23.04 GiB MLP
#: allocation (the act_fn output, 710,400 x 17,408 x bf16) failed ->
#: (137.04 - 51.08 + 23.04) GiB / 710,400 = 164,749 bytes per token. The first
#: model counted ONE MLP intermediate (23 GiB); the live forward holds gate_up
#: (2x that), the activation, the residual streams and the encoder-cache
#: profile at once. Text-only serving (every modality limit 0) removes the
#: encoder share, so this OVER-estimates the text-only peak: safe for sizing,
#: and the serve log's coverage line remains the fact.
PROFILE_ACTIVATION_BYTES_PER_TOKEN = 164_749


@dataclass(frozen=True)
class AttackerArch:
    """The pieces of the attacker's config.json the memory model needs. The
    pinned attacker is a Qwen3.5-family HYBRID (``qwen3_5``): 64 layers with
    ``full_attention_interval`` 4 -> 16 full-attention layers holding a paged
    KV cache, 48 gated-deltanet layers holding a FIXED recurrent state per
    sequence slot (fp32, ``mamba_ssm_dtype``) — so ``max_num_seqs`` itself
    costs memory, exactly the hybrid limitation geometry.py's ``ModelBudget``
    states for qwen35_9b / qwen35_4b."""

    name: str
    weights_bytes: int
    n_layers: int
    n_full_attention: int
    num_key_value_heads: int
    head_dim: int
    linear_num_key_heads: int
    linear_key_head_dim: int
    linear_num_value_heads: int
    linear_value_head_dim: int
    linear_conv_kernel_dim: int
    intermediate_size: int
    vocab_size: int
    kv_dtype_bytes: int = 2          # bf16 KV
    state_dtype_bytes: int = 4       # fp32 recurrent state (mamba_ssm_dtype)

    @property
    def n_linear(self) -> int:
        return self.n_layers - self.n_full_attention

    @property
    def kv_bytes_per_token(self) -> int:
        """K + V for every full-attention layer, per token."""
        return 2 * self.n_full_attention * self.num_key_value_heads * self.head_dim * self.kv_dtype_bytes

    @property
    def state_bytes_per_slot(self) -> int:
        """The recurrent + conv state every gated-deltanet layer pre-allocates
        per sequence slot (independent of L)."""
        rec = (self.linear_num_value_heads * self.linear_key_head_dim
               * self.linear_value_head_dim * self.state_dtype_bytes)
        conv_dim = 2 * self.linear_num_key_heads * self.linear_key_head_dim \
            + self.linear_num_value_heads * self.linear_value_head_dim
        conv = (self.linear_conv_kernel_dim - 1) * conv_dim * self.kv_dtype_bytes
        return self.n_linear * (rec + conv)

    def profile_activation_bytes(self, batched_tokens: int, seqs: int) -> int:
        """vLLM's profiling forward runs ``max_num_batched_tokens`` tokens through
        the model before it sizes the KV cache. Its peak is the MEASURED
        :data:`PROFILE_ACTIVATION_BYTES_PER_TOKEN` per batched token (the MLP's
        gate_up + activation intermediates, the residual streams and the
        encoder-cache profile held at once — 4.7x the single
        ``intermediate_size x bf16`` term the first model used, which is why
        the 710,400-token identity OOMed live) plus one fp32 logits row per
        sequence. Under the identity this term is ~2.4x the KV it protects;
        chunked prefill at 16,384 tokens shrinks it to ~2.5 GiB."""
        return batched_tokens * PROFILE_ACTIVATION_BYTES_PER_TOKEN \
            + seqs * self.vocab_size * 4


#: huihui-ai/Huihui-Qwen3.6-27B-abliterated @27502c87 — config.json (text_config)
#: + the snapshot's safetensors total (55,562,855,904 bytes, 15 shards, incl.
#: the vision tower vLLM also loads; model.safetensors.index.json total_size).
QWEN36_27B = AttackerArch(
    name=ATTACKER_MODEL, weights_bytes=55_562_855_904,
    n_layers=64, n_full_attention=16, num_key_value_heads=4, head_dim=256,
    linear_num_key_heads=16, linear_key_head_dim=128,
    linear_num_value_heads=48, linear_value_head_dim=128, linear_conv_kernel_dim=4,
    intermediate_size=17_408, vocab_size=248_320,
)


@dataclass(frozen=True)
class PoolGeometry:
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    chunked_prefill: bool
    gpu_memory_utilization: float
    fits: bool
    budget_bytes: int
    weights_bytes: int
    kv_bytes: int
    state_bytes: int
    activation_bytes: int
    reserve_bytes: int
    headroom_bytes: int
    max_seqs_fitting: int
    seqs_source: str
    arch: str
    note: str = ""
    #: the KV rule this geometry was fit under. ``kv_seq_tokens`` None = the IDENTITY (KV covers S x L:
    #: zero preemption whatever the requests look like, ``kv_rule == "identity"``); an int = the KV
    #: covers S sequences of that many tokens at once AND one full-window sequence, under the named
    #: rule -- ``"peak_mean"`` (the seedrep pools: S = the production peak, the KV checked at the MEAN
    #: sequence, vLLM's scheduler queuing / preempting on the tail) or ``"p99_fit"`` (the KV holds S
    #: p99-long sequences; recorded BESIDE peak_mean, never the bound)
    kv_seq_tokens: Optional[int] = None
    kv_rule: str = "identity"

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for k in ("budget_bytes", "weights_bytes", "kv_bytes", "state_bytes",
                  "activation_bytes", "reserve_bytes", "headroom_bytes"):
            d[k.replace("_bytes", "_gib")] = round(d[k] / GIB, 2)
        return d

    def describe(self) -> str:
        g = lambda b: f"{b / GIB:.1f} GiB"  # noqa: E731
        return (f"attacker pool geometry: max_model_len={self.max_model_len} "
                f"max_num_seqs={self.max_num_seqs} ({self.seqs_source}) "
                f"max_num_batched_tokens={self.max_num_batched_tokens} "
                f"({'chunked prefill' if self.chunked_prefill else 'no-chunk identity S x L'}); "
                f"budget {g(self.budget_bytes)} @util {self.gpu_memory_utilization} = weights "
                f"{g(self.weights_bytes)} + KV {g(self.kv_bytes)} + recurrent state "
                f"{g(self.state_bytes)} + profile activations {g(self.activation_bytes)} + reserve "
                f"{g(self.reserve_bytes)} -> headroom {g(self.headroom_bytes)} "
                f"({'FITS' if self.fits else 'DOES NOT FIT'}; largest fitting S at this window: "
                f"{self.max_seqs_fitting}; KV rule {self.kv_rule}"
                f"{'' if self.kv_seq_tokens is None else f' at {self.kv_seq_tokens:,} tokens/sequence'})"
                f"{(' — ' + self.note) if self.note else ''}")


def kv_tokens_needed(L: int, S: int, kv_seq_tokens: Optional[int] = None) -> int:
    """The KV tokens the pool must hold at ``S`` slots: ``S x L`` under the identity, else
    ``max(S x kv_seq_tokens, L)`` (S sequences of the given length -- the measured mean under
    ``peak_mean``, the p99 under ``p99_fit`` -- at once, and never less than ONE full-window
    sequence: a single request may still reach L)."""
    if kv_seq_tokens is None:
        return S * L
    return max(S * int(kv_seq_tokens), L)


def _resolve_kv_rule(kv_seq_tokens: Optional[int], kv_rule: Optional[str], L: int) -> Tuple[Optional[int], str]:
    """Validate the (kv_seq_tokens, kv_rule) pair: the identity takes no sequence length and no other
    name; a sequence length needs a rule NAME (never a default guessed for it) and must fit the window."""
    if kv_seq_tokens is None:
        if kv_rule not in (None, "identity"):
            raise ValueError(f"kv_rule {kv_rule!r} needs kv_seq_tokens (the identity is the only rule without one)")
        return None, "identity"
    if isinstance(kv_seq_tokens, bool) or not isinstance(kv_seq_tokens, int) or kv_seq_tokens < 1:
        raise ValueError(f"kv_seq_tokens must be a positive int, got {kv_seq_tokens!r}")
    if kv_seq_tokens > L:
        raise ValueError(f"kv_seq_tokens {kv_seq_tokens} exceeds the window {L}: a sequence the window cannot hold "
                         "means the window, not S, is wrong")
    if not isinstance(kv_rule, str) or not kv_rule.strip() or kv_rule == "identity":
        raise ValueError(f"a KV rule at {kv_seq_tokens} tokens/sequence must be NAMED (e.g. 'peak_mean', 'p99_fit'), got {kv_rule!r}")
    return int(kv_seq_tokens), kv_rule


def _need(arch: AttackerArch, L: int, S: int, batched: int,
          reserve_bytes: int = RESERVE_BYTES, kv_seq_tokens: Optional[int] = None) -> Tuple[int, int, int, int]:
    kv = kv_tokens_needed(L, S, kv_seq_tokens) * arch.kv_bytes_per_token
    state = S * arch.state_bytes_per_slot
    act = arch.profile_activation_bytes(batched, S)
    total = arch.weights_bytes + kv + state + act + reserve_bytes
    return kv, state, act, total


def fit(L: int, S: int, *, util: float = DEFAULT_GPU_MEMORY_UTILIZATION,
        chunked_prefill: bool = False, chunk_tokens: int = 16_384,
        arch: AttackerArch = QWEN36_27B, reserve_bytes: int = RESERVE_BYTES,
        kv_seq_tokens: Optional[int] = None, kv_rule: Optional[str] = None) -> PoolGeometry:
    """The memory model at ``S x L``: does the attacker fit one H200 at
    ``util`` with the whole KV footprint covered (no preemption)? Under the
    no-chunking identity ``max_num_batched_tokens = S x L`` the profiling
    forward runs the whole product; with ``chunked_prefill`` it runs
    ``chunk_tokens`` (KV must still cover S x L for zero preemption).
    ``reserve_bytes`` is the fixed non-KV term (default :data:`RESERVE_BYTES`,
    the one-card CUDA context + graphs + sampler workspace); a caller serving
    the model over several cards passes the larger per-card reserve its
    communicators need, and the geometry records the value it was fit with.
    ``kv_seq_tokens`` + ``kv_rule`` (the one-card seedrep pools) switch the KV term from the identity
    ``S x L`` to :func:`kv_tokens_needed`: the KV must hold ``S`` sequences of the given MEASURED
    length at once (and one full-window sequence) -- the identity over-provisions a pool whose
    requests are far shorter than its window (the crescendo attacker's mean prompt is ~7 % of its
    65,536 window) and is what forced the seedrep pools onto two cards. ``"peak_mean"`` (S = the
    production peak, the KV checked at the mean sequence, vLLM's scheduler queuing / preempting on
    the tail) and ``"p99_fit"`` (S p99-long sequences at once) are the two named rules. The geometry
    records the rule and the length it was fit under."""
    if not (isinstance(L, int) and L > 0 and isinstance(S, int) and S > 0):
        raise ValueError(f"max_model_len and max_num_seqs must be positive ints, got {L!r}, {S!r}")
    if L % 16:
        raise ValueError(f"max_model_len {L} is not a multiple of the vLLM block size 16")
    if not (isinstance(reserve_bytes, int) and reserve_bytes >= 0):
        raise ValueError(f"reserve_bytes must be a non-negative int, got {reserve_bytes!r}")
    kv_seq_tokens, kv_rule_name = _resolve_kv_rule(kv_seq_tokens, kv_rule, L)
    budget = int(util * H200_BYTES)
    batched = min(chunk_tokens, S * L) if chunked_prefill else S * L
    kv, state, act, total = _need(arch, L, S, batched, reserve_bytes, kv_seq_tokens)
    fits = total <= budget
    # the largest S that fits at this window under the same rule (monotone in S)
    s_max = 0
    for s in range(1, MAX_SEQS_CEILING + 1):
        b = min(chunk_tokens, s * L) if chunked_prefill else s * L
        if _need(arch, L, s, b, reserve_bytes, kv_seq_tokens)[3] <= budget:
            s_max = s
        else:
            break
    return PoolGeometry(
        max_model_len=L, max_num_seqs=S, max_num_batched_tokens=batched,
        chunked_prefill=chunked_prefill, gpu_memory_utilization=util, fits=fits,
        budget_bytes=budget, weights_bytes=arch.weights_bytes, kv_bytes=kv, state_bytes=state,
        activation_bytes=act, reserve_bytes=reserve_bytes, headroom_bytes=budget - total,
        max_seqs_fitting=s_max, seqs_source="explicit", arch=arch.name,
        kv_seq_tokens=kv_seq_tokens, kv_rule=kv_rule_name)


def resolve_pool_geometry(max_model_len: int = DEFAULT_MAX_MODEL_LEN,
                          max_num_seqs: Optional[int] = None, *,
                          util: float = DEFAULT_GPU_MEMORY_UTILIZATION,
                          chunked_prefill: bool = False, chunk_tokens: int = 16_384,
                          arch: AttackerArch = QWEN36_27B,
                          reserve_bytes: int = RESERVE_BYTES,
                          kv_seq_tokens: Optional[int] = None, kv_rule: Optional[str] = None) -> PoolGeometry:
    """The pool's launch geometry. ``max_num_seqs`` unset -> the LARGEST S the
    memory model fits at the window under the KV rule (``seqs_source="kv_fit"``; the
    rule itself is ``geometry.kv_rule``); an explicit S that does not fit is REFUSED
    (:class:`PoolGeometryUnfit`), naming the rule and the largest S that would. An
    explicit S above :data:`MAX_SEQS_CEILING` is refused too. ``reserve_bytes``,
    ``kv_seq_tokens`` and ``kv_rule`` as in :func:`fit`."""
    kw = dict(util=util, chunked_prefill=chunked_prefill, chunk_tokens=chunk_tokens, arch=arch,
              reserve_bytes=reserve_bytes, kv_seq_tokens=kv_seq_tokens, kv_rule=kv_rule)
    if max_num_seqs is None:
        probe = fit(max_model_len, 1, **kw)
        if probe.max_seqs_fitting < 1:
            raise PoolGeometryUnfit(probe.describe() + " — not even one sequence fits")
        g = fit(max_model_len, probe.max_seqs_fitting, **kw)
        return PoolGeometry(**{**asdict(g), "seqs_source": "kv_fit"})
    if max_num_seqs > MAX_SEQS_CEILING:
        raise PoolGeometryUnfit(f"max_num_seqs {max_num_seqs} exceeds the {MAX_SEQS_CEILING} ceiling")
    g = fit(max_model_len, max_num_seqs, **kw)
    if not g.fits:
        rule = ("S x L identity" if g.kv_seq_tokens is None
                else f"{g.kv_rule} rule (S x {g.kv_seq_tokens} sequence tokens, and one full window)")
        raise PoolGeometryUnfit(
            g.describe() + f" — refusing under the {rule}: set max_num_seqs <= {g.max_seqs_fitting} at this "
            "window, or lower the window, or run chunked prefill at a bounded batch "
            "(the pool's mode) for a larger S")
    return g


# --------------------------------------------------------------------------- #
# the vLLM serve log: coverage attestation + in-flight statistics               #
# --------------------------------------------------------------------------- #

_KV_RE = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens")
_CONC_RE = re.compile(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)x")
_STATS_RE = re.compile(
    r"(?P<ts>\d\d-\d\d \d\d:\d\d:\d\d).*?"
    r"Avg prompt throughput:\s*(?P<prompt_tps>[\d.]+)\s*tokens/s,\s*"
    r"Avg generation throughput:\s*(?P<gen_tps>[\d.]+)\s*tokens/s,\s*"
    r"Running:\s*(?P<running>\d+)\s*reqs,\s*Waiting:\s*(?P<waiting>\d+)\s*reqs"
    r"(?:.*?GPU KV cache usage:\s*(?P<kv_pct>[\d.]+)%)?"
    r"(?:.*?Prefix cache hit rate:\s*(?P<prefix_pct>[\d.]+)%)?")


def kv_coverage_from_log(text: str) -> Dict[str, Any]:
    """vLLM's own statement of what it allocated: ``GPU KV cache size: N
    tokens`` and ``Maximum concurrency for L tokens per request: Xx`` (for a
    hybrid model the latter is the allocator's real slot count, the number
    geometry.py's hybrid budgets are read from)."""
    out: Dict[str, Any] = {"kv_tokens_reported": None, "max_concurrency_reported": None,
                           "concurrency_window": None}
    m = _KV_RE.search(text)
    if m:
        out["kv_tokens_reported"] = int(m.group(1).replace(",", ""))
    m = _CONC_RE.search(text)
    if m:
        out["concurrency_window"] = int(m.group(1).replace(",", ""))
        out["max_concurrency_reported"] = float(m.group(2))
    return out


def live_kv_tokens(cov: Dict[str, Any], *, max_model_len: int) -> Optional[int]:
    """The KV tokens the live engine allocated: ``GPU KV cache size: N tokens`` when the log states
    it, else the concurrency line x the window (the same fact); ``None`` when neither is present."""
    kv_tokens = cov.get("kv_tokens_reported")
    if isinstance(kv_tokens, int) and not isinstance(kv_tokens, bool):
        return int(kv_tokens)
    conc = cov.get("max_concurrency_reported")
    return int(math.floor(conc * max_model_len)) if conc is not None else None


def coverage_fraction(cov: Dict[str, Any], *, max_num_seqs: int, max_model_len: int,
                      seq_tokens: int) -> Optional[float]:
    """``live KV tokens / (S x seq_tokens, never below one window)`` -- how much of S sequences of
    ``seq_tokens`` the live allocation holds at once (>= 1: all of them). The seedrep pools record
    it at the p99 sequence BESIDE the ``peak_mean`` check: the number to read next
    to ``vllm:num_requests_waiting`` / ``vllm:num_preemptions_total`` when the tail queues."""
    if isinstance(seq_tokens, bool) or not isinstance(seq_tokens, int) or seq_tokens < 1:
        raise ValueError(f"seq_tokens must be a positive int, got {seq_tokens!r}")
    live = live_kv_tokens(cov, max_model_len=max_model_len)
    if live is None:
        return None
    return round(live / kv_tokens_needed(max_model_len, max_num_seqs, seq_tokens), 4)


def coverage_problem(cov: Dict[str, Any], *, max_num_seqs: int, max_model_len: int,
                     kv_seq_tokens: Optional[int] = None, kv_rule: Optional[str] = None) -> Optional[str]:
    """``None`` when the live allocation covers the launched geometry under its KV
    rule -- ``S x L`` (the identity: no preemption at the launched S whatever the
    requests), or, with ``kv_seq_tokens`` + ``kv_rule``, ``max(S x kv_seq_tokens, L)``
    tokens (S sequences of that length at once, and one full-window one) -- else the
    shortfall, named. The live figure is vLLM's ``Maximum concurrency for L tokens per
    request: Xx`` line (= KV tokens / L; ``GPU KV cache size: N tokens`` when it is
    present is the same fact and is preferred under a named rule)."""
    conc = cov.get("max_concurrency_reported")
    if conc is None:
        return "the serve log states no 'Maximum concurrency' line (server not ready, or a vLLM " \
               "that logs differently) — coverage unattested"
    if cov.get("concurrency_window") not in (None, max_model_len):
        return (f"the serve log's concurrency line is for {cov['concurrency_window']} tokens per "
                f"request, not the launched window {max_model_len}")
    try:
        seq, rule = _resolve_kv_rule(kv_seq_tokens, kv_rule, max_model_len)
    except ValueError as exc:
        return f"the KV rule cannot be checked: {exc}"
    if seq is None:
        if conc + 1e-9 < max_num_seqs:
            return (f"vLLM reports room for {conc}x sequences of {max_model_len} tokens, below the "
                    f"launched max_num_seqs {max_num_seqs}: the identity does not hold live (the memory "
                    "model over-estimated the headroom); relaunch at S <= "
                    f"{math.floor(conc)} or lower the window")
        return None
    live = live_kv_tokens(cov, max_model_len=max_model_len)
    need = kv_tokens_needed(max_model_len, max_num_seqs, seq)
    if live < need:
        s_fit = max(0, live // seq) if live >= max_model_len else 0
        return (f"vLLM reports {live:,} KV tokens ({conc}x {max_model_len}), below the {need:,} the {rule} rule needs "
                f"for max_num_seqs {max_num_seqs} x {seq} sequence tokens (never below one full "
                f"window): the fit does not hold live; relaunch at S <= {s_fit} or lower the window")
    return None


def log_stats(text: str, *, since: Optional[str] = None, until: Optional[str] = None) -> Dict[str, Any]:
    """The pool's in-flight profile from vLLM's periodic stats lines
    (``Running: N reqs, Waiting: M reqs, ... Prefix cache hit rate: P%``),
    optionally restricted to log timestamps (``MM-DD HH:MM:SS``, vLLM's format)
    between ``since`` and ``until`` inclusive — the canary's window."""
    rows = []
    for m in _STATS_RE.finditer(text):
        ts = m.group("ts")
        if since and ts < since:
            continue
        if until and ts > until:
            continue
        rows.append({k: (float(v) if v is not None else None)
                     for k, v in m.groupdict().items() if k != "ts"} | {"ts": ts})
    if not rows:
        return {"n_stats_lines": 0}
    run = [r["running"] for r in rows]
    wait = [r["waiting"] for r in rows]
    gen = [r["gen_tps"] for r in rows]
    active = [r for r in rows if r["running"] > 0]
    pref = [r["prefix_pct"] for r in rows if r.get("prefix_pct") is not None]
    kv = [r["kv_pct"] for r in rows if r.get("kv_pct") is not None]
    return {
        "n_stats_lines": len(rows), "first_ts": rows[0]["ts"], "last_ts": rows[-1]["ts"],
        "running_mean": round(sum(run) / len(run), 2), "running_max": max(run),
        "running_mean_while_active": (round(sum(r["running"] for r in active) / len(active), 2)
                                      if active else 0.0),
        "waiting_mean": round(sum(wait) / len(wait), 2), "waiting_max": max(wait),
        "gen_tps_mean": round(sum(gen) / len(gen), 1), "gen_tps_max": max(gen),
        "gen_tps_mean_while_active": (round(sum(r["gen_tps"] for r in active) / len(active), 1)
                                      if active else 0.0),
        "prefix_cache_hit_rate_last": (pref[-1] if pref else None),
        "kv_usage_pct_max": (max(kv) if kv else None),
    }


# --------------------------------------------------------------------------- #
# routable address                                                              #
# --------------------------------------------------------------------------- #

def routable_problem(ip: str) -> Optional[str]:
    """Why ``ip`` must not be advertised (loopback / docker bridge), else None."""
    if not ip or _NON_ROUTABLE.match(ip):
        return (f"{ip!r} is not routable cross-node: loopback or the docker-bridge range "
                "(172.16/12). Advertise the address the node NAME resolves to (`getent hosts "
                "$(hostname)`, on the cluster's internal fabric Slurm's NodeAddr uses), never "
                "`hostname -I | awk '{print $1}'` — that picked the bridge on 2026-08-30 and the "
                "pool served zero requests")
    return None


# --------------------------------------------------------------------------- #
# CLI (the sbatch scripts call these)                                           #
# --------------------------------------------------------------------------- #

def _cli_size(a) -> int:
    try:
        g = resolve_pool_geometry(a.max_model_len, a.max_num_seqs, util=a.util,
                                  chunked_prefill=a.chunked_prefill, chunk_tokens=a.chunk_tokens)
    except (PoolGeometryUnfit, ValueError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    if a.sh:
        print(f"POOL_MAX_MODEL_LEN={g.max_model_len}")
        print(f"POOL_MAX_NUM_SEQS={g.max_num_seqs}")
        print(f"POOL_MAX_NUM_BATCHED_TOKENS={g.max_num_batched_tokens}")
        print(f"POOL_SEQS_SOURCE={g.seqs_source}")
        print(f"POOL_GEOMETRY_JSON='{json.dumps(g.as_dict())}'")
    else:
        print(g.describe())
        print(json.dumps(g.as_dict(), indent=2))
    return 0


def _cli_verify(a) -> int:
    try:
        ep, _key, ev = connect(a.pool, squeue=a.squeue, expect_model=a.expect_model,
                               expect_snapshot=a.expect_snapshot)
    except AttackerPoolError as e:
        print(f"REFUSED ({type(e).__name__}): {e}", file=sys.stderr)
        return 3
    out = {"endpoint": ep.identity(), "evidence": ev}
    print(json.dumps(out, indent=2))
    return 0


def _cli_wait(a) -> int:
    """Block until endpoint.json appears and verifies (the consumer's
    'wait for the pool' step), or time out."""
    t0 = time.time()
    last = ""
    while time.time() - t0 < a.timeout:
        try:
            ep, _key, ev = connect(a.pool, squeue=a.squeue)
            print(json.dumps({"endpoint": ep.identity(), "evidence": ev}, indent=2))
            return 0
        except AttackerPoolError as e:
            msg = f"{type(e).__name__}: {str(e).splitlines()[0]}"
            if msg != last:
                print(f"[wait] {msg}", file=sys.stderr)
                last = msg
            if isinstance(e, (PoolStale, PoolUnguarded, PoolUnauthorized, PoolMismatch)):
                print("REFUSED: not a transient condition; giving up", file=sys.stderr)
                return 3
        time.sleep(a.poll)
    print(f"REFUSED: pool at {a.pool} did not come up within {a.timeout}s", file=sys.stderr)
    return 4


def _cli_coverage(a) -> int:
    text = Path(a.log).read_text(errors="replace")
    cov = kv_coverage_from_log(text)
    problem = coverage_problem(cov, max_num_seqs=a.max_num_seqs, max_model_len=a.max_model_len)
    print(json.dumps({**cov, "problem": problem}, indent=2))
    return 0 if problem is None else 3


def _cli_logstats(a) -> int:
    text = Path(a.log).read_text(errors="replace")
    print(json.dumps(log_stats(text, since=a.since, until=a.until), indent=2))
    return 0


def _cli_routable(a) -> int:
    p = routable_problem(a.ip)
    if p:
        print(f"REFUSED: {p}", file=sys.stderr)
        return 3
    print(a.ip)
    return 0


def _cli_mint(a) -> int:
    key = mint_api_key()
    try:
        write_api_key(a.out, key)
    except PoolKeyUnreadable as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    print(a.out)          # the PATH, never the key
    return 0


def _cli_endpoint(a) -> int:
    """Write endpoint.json atomically from the fields the pool sbatch knows."""
    d = {
        "schema_version": SCHEMA_VERSION, "pool_name": a.pool_name, "pool_dir": a.pool_dir,
        "url": a.url, "model": a.model, "snapshot": a.snapshot, "model_path": a.model_path,
        "max_model_len": a.max_model_len, "max_num_seqs": a.max_num_seqs,
        "max_num_batched_tokens": a.max_num_batched_tokens,
        "gpu_memory_utilization": a.util, "prefix_caching": not a.no_prefix_caching,
        "api_key_env_name": API_KEY_ENV, "api_key_file": a.api_key_file,
        "job_id": a.job_id, "node": a.node, "node_ip": a.node_ip,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "serve_log": a.serve_log, "repo_sha": a.repo_sha,
    }
    if a.geometry_json:
        d["geometry"] = json.loads(a.geometry_json)
    if a.coverage_json:
        d["kv_coverage"] = json.loads(a.coverage_json)
    if getattr(a, "serving_manifest", None):
        # the launcher's ENGINE-VERIFIED serving manifest (scripts/vllm_attacker_serve.sh
        # prints VLLM_SERVING_MANIFEST=): the record's precision / prefix-cache
        # fields are DERIVED from it, never declared here; a consumer refuses a
        # record without the attested block
        from . import attacker_serving as serving
        try:
            manifest = serving.read_manifest(a.serving_manifest)
            attested = serving.attested_serving_record(manifest, manifest_path=str(a.serving_manifest))
        except serving.AttackerServingError as exc:
            raise PoolAbsent(f"--serving-manifest {a.serving_manifest}: {exc}") from exc
        d["attacker_quantization"] = attested["attacker_quantization"]
        d["prefix_caching"] = attested["prefix_caching"]
        d[serving.SERVING_RECORD_KEY] = attested
    for kv in a.extra or []:
        k, _, v = kv.partition("=")
        d[k] = v
    ep = PoolEndpoint.from_dict(d)       # validates before anything is written
    out = Path(a.pool_dir) / ENDPOINT_FILE
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ep.as_dict(), indent=2))
    tmp.rename(out)
    print(str(out))
    return 0


def _cli_live_job(a) -> int:
    """Exit 0 iff endpoint.json in the pool dir names a RUNNING job (the
    refuse-to-overwrite check); 1 when there is no file; 2 when it is stale."""
    p = endpoint_path(a.pool)
    if not p.is_file():
        return 1
    try:
        ep = read_endpoint(p)
        check_job_live(ep)
    except PoolStale as e:
        print(str(e).splitlines()[0])
        return 2
    except PoolAbsent as e:
        print(str(e).splitlines()[0])
        return 2
    print(f"LIVE job {ep.job_id} on {ep.node} at {ep.url}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="attacker_pool", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("size", help="resolve/refuse the pool geometry (memory model)")
    s.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    s.add_argument("--max-num-seqs", type=int, default=None)
    s.add_argument("--util", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    s.add_argument("--chunked-prefill", action="store_true")
    s.add_argument("--chunk-tokens", type=int, default=16_384)
    s.add_argument("--sh", action="store_true", help="print shell assignments")
    s.set_defaults(fn=_cli_size)
    v = sub.add_parser("verify", help="liveness + bearer + identity checks against a pool")
    v.add_argument("--pool", required=True, help="POOL_DIR or endpoint.json")
    v.add_argument("--squeue", action="store_true", help="legacy only: check an old Slurm job")
    v.add_argument("--expect-model", default=None)
    v.add_argument("--expect-snapshot", default=None)
    v.set_defaults(fn=_cli_verify)
    w = sub.add_parser("wait", help="wait for a pool to advertise + verify")
    w.add_argument("--pool", required=True)
    w.add_argument("--timeout", type=float, default=3600)
    w.add_argument("--poll", type=float, default=20)
    w.add_argument("--squeue", action="store_true", help="legacy only: check an old Slurm job")
    w.set_defaults(fn=_cli_wait)
    c = sub.add_parser("coverage", help="attest S x L coverage from the serve log")
    c.add_argument("--log", required=True)
    c.add_argument("--max-num-seqs", type=int, required=True)
    c.add_argument("--max-model-len", type=int, required=True)
    c.set_defaults(fn=_cli_coverage)
    ls = sub.add_parser("logstats", help="in-flight / throughput stats from the serve log")
    ls.add_argument("--log", required=True)
    ls.add_argument("--since", default=None, help="vLLM log timestamp 'MM-DD HH:MM:SS'")
    ls.add_argument("--until", default=None)
    ls.set_defaults(fn=_cli_logstats)
    r = sub.add_parser("routable", help="refuse a loopback / docker-bridge address")
    r.add_argument("ip")
    r.set_defaults(fn=_cli_routable)
    m = sub.add_parser("mint", help="mint the bearer into a 0600 file (prints the path)")
    m.add_argument("--out", required=True)
    m.set_defaults(fn=_cli_mint)
    e = sub.add_parser("endpoint", help="write endpoint.json atomically")
    for name in ("pool-name", "pool-dir", "url", "model", "snapshot", "model-path",
                 "api-key-file", "job-id", "node", "node-ip", "serve-log", "repo-sha"):
        e.add_argument(f"--{name}", required=name not in ("model-path", "serve-log", "repo-sha"),
                       default=None)
    e.add_argument("--max-model-len", type=int, required=True)
    e.add_argument("--max-num-seqs", type=int, required=True)
    e.add_argument("--max-num-batched-tokens", type=int, required=True)
    e.add_argument("--util", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION)
    e.add_argument("--no-prefix-caching", action="store_true")
    e.add_argument("--geometry-json", default=None)
    e.add_argument("--coverage-json", default=None)
    e.add_argument("--serving-manifest", default=None,
                   help="the launcher's engine-verified serving manifest (VLLM_SERVING_MANIFEST=); "
                        "the record's attacker_quantization / prefix_caching / serving fields are "
                        "derived from it (a consumer refuses a record that lacks the attested block)")
    e.add_argument("--extra", action="append", default=None, help="k=v informational fields")
    e.set_defaults(fn=_cli_endpoint)
    lj = sub.add_parser("live-job", help="0 = a RUNNING pool owns endpoint.json; 1 = none; 2 = stale")
    lj.add_argument("--pool", required=True)
    lj.set_defaults(fn=_cli_live_job)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
