"""One shared Qwen3.6-27B attacker on a persistent Modal Server.

The pool is a deployed ``modal.Server``, not a per-run Function: vLLM already
handles concurrent HTTP requests, and one fixed H200 must serve up to the
measured 37-request geometry without Modal scaling a second copy.  The public
Modal URL is application-guarded by vLLM's ``VLLM_API_KEY`` bearer.

Every CLI invocation is explicitly named.  Modal 1.5 calls the app-name flag
``--name`` (not ``--app-name``):

  # rotate the local 0600 bearer, stage the existing snapshot under /vol/hf
  ATTACKER_POOL_ROTATE_KEY=1 ~/venvs/modal/bin/python -m modal run \
    --name attacker-pool-qwen36-27b-stage \
    scripts/modal/attacker_pool.py::tools.main --action stage

  # deploy only the persistent one-H200 server app
  ~/venvs/modal/bin/python -m modal deploy \
    --name attacker-pool-qwen36-27b scripts/modal/attacker_pool.py::app

  # utility app: no local run instantiates the H200 Server declaration
  ~/venvs/modal/bin/python -m modal run \
    --name attacker-pool-qwen36-27b-endpoint \
    scripts/modal/attacker_pool.py::tools.main --action endpoint
  ~/venvs/modal/bin/python -m modal run \
    --name attacker-pool-qwen36-27b-canary \
    scripts/modal/attacker_pool.py::tools.main --action canary

The canary is CPU-only and compares Modal container ids before doing any HTTP
work.  Same-container evidence is a refusal, never a green canary.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path

import modal

APP_NAME = "attacker-pool-qwen36-27b"
SERVER_NAME = "qwen36-27b-abliterated-vllm"
POOL_NAME = "qwen36-27b-l19200-s37"
VOL = modal.Volume.from_name("twosi-fits", create_if_missing=False)
VOL_PATH = "/vol"
POOL_DIR = f"{VOL_PATH}/attacker-pool/{POOL_NAME}"
SERVE_LOG = f"{POOL_DIR}/serve.log"
ENDPOINT_PATH = f"{POOL_DIR}/endpoint.json"
PORT = 8000

ATTACKER_MODEL = "huihui-ai/Huihui-Qwen3.6-27B-abliterated"
ATTACKER_SNAPSHOT = "27502c8717fd5a2f8c0c77188c10c243fd4f672e"
MODEL_REPO_DIR = f"{VOL_PATH}/hf/hub/models--huihui-ai--Huihui-Qwen3.6-27B-abliterated"
MODEL_PATH = f"{MODEL_REPO_DIR}/snapshots/{ATTACKER_SNAPSHOT}"
LEGACY_MODEL_REPO_DIR = (
    f"{VOL_PATH}/grokmt/hf_attacker/"
    "models--huihui-ai--Huihui-Qwen3.6-27B-abliterated"
)
MAX_MODEL_LEN = 19_200
MAX_NUM_SEQS = 37
# CHUNKED prefill at a bounded batch. The no-chunk identity
# (max_num_batched_tokens = S x L = 710,400) is unfittable for this bf16 hybrid
# 27B on one H200: the 2026-09-04 deploy of dc065dd OOMed inside the startup
# profile (137.84 GiB in use with 51.08 GiB of weights, a further 23.04 GiB MLP
# allocation failing) -- the profile forward at 710,400 tokens alone peaks
# above 109 GiB, more than any weight offload could give back, so the offload
# variant (8e68019, never deployed) could not have served either. The per-job
# attacker whose 22,661 calls sized L ran vLLM's default chunked prefill, so
# chunked IS the attacker's measured regime; the identity never was. KV still
# covers S x L (the live coverage gate below is the fact), bf16 weights stay
# exact, and torch.compile + CUDA graphs stay on for decode speed.
MAX_NUM_BATCHED_TOKENS = 16_384
CHUNKED_PREFILL = True
GPU_MEMORY_UTILIZATION = 0.90
# Text-only: the attacker never sees an image. With every modality limit 0
# vLLM loads no vision modules and profiles no encoder cache (the prior
# deploy profiled 37 maximum-size images on top of the 710,400-token forward).
LIMIT_MM_PER_PROMPT = '{"image": 0, "video": 0}'
API_KEY_ENV = "VLLM_API_KEY"
# The REQUESTED serving options (weights bf16, prefix caching on). The vLLM flags
# are composed from them through sisteer.eval.attacks.attacker_serving -- the same
# seam as the per-job attacker -- and after readiness the engine's own resolved
# config line is verified against them (_attest_serving); the endpoint record
# advertises what that line SAID, never these literals, and a contradiction
# terminates the pool instead of advertising.
POOL_QUANTIZATION = "none"
POOL_PREFIX_CACHING = True
SERVING_MANIFEST = f"{POOL_DIR}/serving.json"

# Modal mounts a source file flat at /root/<name>.py in remote containers.
# Resolve checkout paths only on the grading host; the baked source exists at the
# destination path when the module is imported remotely.
if modal.is_local():
    REPO = Path(__file__).resolve().parents[2]
    SOURCE_DIR = REPO / "src" / "sisteer"
else:
    REPO = Path("/root")
    SOURCE_DIR = Path("/root/sisteer")


def _repo_sha() -> str:
    if not modal.is_local():
        return os.environ.get("ATTACKER_POOL_REPO_SHA", "unknown")
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001 - informational provenance only
        return os.environ.get("ATTACKER_POOL_REPO_SHA", "unknown")


SOURCE_SHA = _repo_sha()


def _local_state_dir() -> Path:
    """The grading host's pool directory: the site's SISTEER_ATTACKER_POOL_STATE_DIR, else
    ``$STATE/sisteer/attacker-pool-modal``, else ``~/.local/state/sisteer/attacker-pool-modal``. Grading host only:
    the Modal client's interpreter carries no project package, so the checkout's ``src`` goes on the path first."""
    import sys

    src = str(REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from sisteer import site

    return site.state_path("SISTEER_ATTACKER_POOL_STATE_DIR", "sisteer", "attacker-pool-modal")


def _local_key_path() -> Path:
    return _local_state_dir() / "api_key"


def _local_endpoint_path() -> Path:
    return _local_state_dir() / "endpoint.json"


def _load_local_api_key() -> str:
    """Read or mint the grading-host-only bearer; never print or put it on argv."""
    if not modal.is_local():
        return os.environ.get(API_KEY_ENV, "")
    p = _local_key_path()
    rotate = os.environ.get("ATTACKER_POOL_ROTATE_KEY") == "1"
    if rotate and p.exists():
        p.unlink()
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(fd, secrets.token_urlsafe(32).encode())
        finally:
            os.close(fd)
        os.chmod(p, 0o600)
    mode = p.stat().st_mode & 0o777
    if mode & 0o077:
        raise RuntimeError(f"refusing bearer file {p}: mode {oct(mode)} is not private")
    key = p.read_text().strip()
    if not key:
        raise RuntimeError(f"refusing empty bearer file {p}")
    return key


pool_secret = modal.Secret.from_dict({API_KEY_ENV: _load_local_api_key()})

# Same proven cu129/vLLM stack as grokmt_served.py.  Only pinned packages are
# added here; the source tree is copied so the child vLLM process can import the
# shared contract helpers from /root/sisteer.
pool_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "build-essential")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "https://github.com/vllm-project/vllm/releases/download/v0.25.1/"
        "vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
        "#sha256=9e206f370c934a2d4b6b1f05d3d09708d344e05d80260189ef19f60755709431",
        "torch==2.11.0+cu129",
        "torchvision==0.26.0+cu129",
        "torchaudio==2.11.0+cu129",
        "requests==2.34.2",
        "ninja==1.13.0",
        extra_index_url="https://download.pytorch.org/whl/cu129",
    )
    .pip_install(
        "transformers==5.14.1",
        "tokenizers==0.22.2",
        extra_options="--no-deps --force-reinstall",
    )
    .add_local_dir(SOURCE_DIR, "/root/sisteer", copy=True)
    .env(
        {
            "PYTHONPATH": "/root",
            "HF_HOME": f"{VOL_PATH}/hf",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "ATTACKER_POOL_REPO_SHA": SOURCE_SHA,
        }
    )
)

client_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("requests==2.34.2", "huggingface-hub==1.16.1")
    .add_local_dir(SOURCE_DIR, "/root/sisteer", copy=True)
    .env({
        "PYTHONPATH": "/root",
        "HF_HOME": f"{VOL_PATH}/hf",
        "ATTACKER_POOL_REPO_SHA": SOURCE_SHA,
    })
)

# Keep the production Server in its own app. Running stage/endpoint/canary
# through ``tools`` must never instantiate a second min_containers=1 H200.
app = modal.App(APP_NAME)
tools = modal.App(f"{APP_NAME}-tools")


def _http_status(url: str, *, key: str | None = None, timeout: float = 10.0) -> int:
    import requests

    headers = {"Authorization": f"Bearer {key}"} if key else None
    try:
        return int(requests.get(url, headers=headers, timeout=timeout).status_code)
    except Exception:  # noqa: BLE001 - caller turns 0 into a refusal
        return 0


def _wait_ready(proc: subprocess.Popen, log_path: Path, timeout_s: float = 3600) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = log_path.read_text(errors="replace")[-50_000:] if log_path.exists() else ""
            VOL.commit()
            raise RuntimeError(f"vLLM exited rc={proc.returncode} before readiness:\n{tail}")
        if _http_status(f"http://127.0.0.1:{PORT}/health") == 200:
            return
        time.sleep(5)
    raise RuntimeError(f"vLLM did not become healthy within {timeout_s:.0f}s")


def _attest_coverage(log_text: str, *, chunked_prefill: bool = CHUNKED_PREFILL) -> dict:
    """Return vLLM's coverage fact or refuse, in either prefill mode."""
    from sisteer.eval.attacks import attacker_pool as contract

    cov = contract.kv_coverage_from_log(log_text)
    problem = contract.coverage_problem(
        cov, max_num_seqs=MAX_NUM_SEQS, max_model_len=MAX_MODEL_LEN
    )
    if problem:
        mode = "chunked" if chunked_prefill else "no-chunk"
        raise RuntimeError(
            f"{mode} pool coverage is not certified: {problem}. Refusing to advertise"
        )
    return cov


def _attest_serving(log_text: str, opts, *, manifest_path: str | Path, log_path: str,
                    endpoint: str, model: str = MODEL_PATH, served_name: str = ATTACKER_MODEL,
                    extra: dict | None = None) -> tuple[dict, dict]:
    """Verify the live engine against the REQUESTED serving options and return
    ``(manifest, record_fields)``: the verified manifest (also written to
    ``manifest_path``, the pool's twin of the per-job ``<log>.serving.json``)
    and the endpoint-record fields DERIVED from the engine line
    (``attacker_quantization``, ``prefix_caching``, ``serving``).  A contradiction
    -- FP8 where bf16 was asked, cache off where on was asked, a non-auto KV
    cache dtype, a missing align mode -- raises, and the caller terminates the
    pool rather than advertise a false identity."""
    from sisteer.eval.attacks import attacker_serving as serving

    engine = serving.parse_engine_log(log_text)
    problems = serving.verify_engine(opts, engine)
    if problems:
        raise RuntimeError(
            "attacker pool engine contradicts the requested serving options: "
            + "; ".join(problems) + ". Refusing to advertise")
    manifest = serving.write_manifest(
        manifest_path, opts=opts, sources={"quantization": "pool_constant",
                                           "prefix_caching": "pool_constant"},
        engine=engine, model=model, served_name=served_name, endpoint=endpoint,
        log_path=log_path, argv=opts.vllm_argv(model_role="attacker"), extra=extra)
    attested = serving.attested_serving_record(manifest, manifest_path=str(manifest_path))
    fields = {
        "attacker_quantization": attested["attacker_quantization"],
        "prefix_caching": attested["prefix_caching"],
        serving.SERVING_RECORD_KEY: attested,
    }
    return manifest, fields


def _wait_endpoint_file(timeout_s: float = 600) -> dict:
    """Wait for a complete Volume-published endpoint record."""
    from sisteer.eval.attacks import attacker_pool as contract

    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            VOL.reload()
            record = json.loads(Path(ENDPOINT_PATH).read_text())
            contract.PoolEndpoint.from_dict(record)
            return record
        except (OSError, json.JSONDecodeError, contract.AttackerPoolError) as exc:
            last_error = exc
            time.sleep(5)
    detail = f": {type(last_error).__name__}: {last_error}" if last_error else ""
    raise RuntimeError(f"endpoint.json was not published within {timeout_s:.0f}s{detail}")


def _assert_separate_container(endpoint: dict, canary_container_id: str) -> str:
    """Return the pool container id, or refuse before the canary sends traffic."""
    live = endpoint.get("liveness")
    pool_container = live.get("container_id") if isinstance(live, dict) else None
    if not canary_container_id:
        raise RuntimeError("canary has no MODAL_TASK_ID; container separation cannot be proved")
    if not pool_container:
        raise RuntimeError("endpoint.json has no pool container id; separation cannot be proved")
    if str(pool_container) == str(canary_container_id):
        raise RuntimeError(
            f"canary container {canary_container_id} is the pool container; refusing same-container green"
        )
    return str(pool_container)


def _server_identity() -> tuple[str, str]:
    """The Server's object id + public URL from inside its own container. The
    deployed-name lookup is the fallback when the in-container handle is not
    hydrated; empty strings mean unknown, and the caller refuses to advertise."""
    sid, url = "", ""
    try:
        sid = str(AttackerPoolServer.object_id or "")
        url = str(AttackerPoolServer.get_url() or "")
    except Exception as exc:  # noqa: BLE001 - fall through to the lookup
        print(f"[pool] in-container server handle unavailable: {type(exc).__name__}: {exc}", flush=True)
    if not (sid and url):
        try:
            srv = modal.Server.from_name(APP_NAME, SERVER_NAME)
            srv.hydrate()
            sid = sid or str(srv.object_id or "")
            url = url or str(srv.get_url() or "")
        except Exception as exc:  # noqa: BLE001 - the caller refuses on empty
            print(f"[pool] deployed-name server lookup failed: {type(exc).__name__}: {exc}", flush=True)
    return sid, url.rstrip("/")


def _app_identity() -> str:
    aid = ""
    try:
        aid = str(app.app_id or "")
    except Exception:  # noqa: BLE001
        aid = ""
    if not aid:
        try:
            aid = str(modal.App.lookup(APP_NAME).app_id or "")
        except Exception as exc:  # noqa: BLE001 - the caller refuses on empty
            print(f"[pool] deployed-name app lookup failed: {type(exc).__name__}: {exc}", flush=True)
    return aid


def _call_id() -> str | None:
    """A Server has no per-input call; record one only when Modal supplies it."""
    try:
        cid = modal.current_function_call_id()
    except Exception:  # noqa: BLE001
        return None
    return str(cid) if cid else None


def _write_endpoint(record: dict) -> None:
    from sisteer.eval.attacks import attacker_pool as contract

    # Validate the complete record before the atomic publish.
    contract.PoolEndpoint.from_dict(record)
    out = Path(ENDPOINT_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2))
    os.replace(tmp, out)
    VOL.commit()


@tools.function(image=client_image, volumes={VOL_PATH: VOL}, timeout=3 * 3600, cpu=8, memory=32768)
def stage_weights() -> dict:
    """Make the already-staged public snapshot available at /vol/hf/hub.

    The earlier per-job launcher staged the exact snapshot elsewhere on the
    same Volume.  A Volume symlink avoids a second 55.6 GB copy.  If that source
    is absent, snapshot_download populates the standard /vol/hf cache instead.
    """
    dst = Path(MODEL_REPO_DIR)
    snap = Path(MODEL_PATH)
    legacy = Path(LEGACY_MODEL_REPO_DIR)
    legacy_snap = legacy / "snapshots" / ATTACKER_SNAPSHOT
    if snap.is_dir() and (snap / "config.json").is_file():
        return {"ok": True, "method": "already_present", "path": str(snap)}
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        raise RuntimeError(f"partial/non-matching model cache already exists at {dst}; refusing to replace it")
    if legacy_snap.is_dir() and (legacy_snap / "config.json").is_file():
        os.symlink(str(legacy), str(dst), target_is_directory=True)
        VOL.commit()
        if not (snap / "config.json").is_file():
            raise RuntimeError("Modal Volume did not preserve the /vol/hf attacker-cache symlink")
        return {"ok": True, "method": "symlink_existing_volume_snapshot", "path": str(snap)}

    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        ATTACKER_MODEL,
        revision=ATTACKER_SNAPSHOT,
        cache_dir=f"{VOL_PATH}/hf/hub",
        local_files_only=False,
        max_workers=16,
    )
    VOL.commit()
    if Path(path).resolve() != snap.resolve():
        raise RuntimeError(f"snapshot_download resolved {path}, expected {snap}")
    return {"ok": True, "method": "snapshot_download", "path": str(snap)}


@app.server(
    name=SERVER_NAME,
    image=pool_image,
    volumes={VOL_PATH: VOL},
    secrets=[pool_secret],
    gpu="H200",
    cpu=16,
    memory=131072,
    port=PORT,
    target_concurrency=MAX_NUM_SEQS,
    min_containers=1,
    max_containers=1,
    startup_timeout=3600,
    unauthenticated=True,
    exit_grace_period=60,
)
class AttackerPoolServer:
    @modal.enter()
    def start(self) -> None:
        from sisteer.eval.attacks import attacker_pool as contract

        for tool in ("vllm", "ninja"):
            if not shutil.which(tool):
                raise RuntimeError(f"required runtime tool {tool!r} is not on PATH")

        geom = contract.resolve_pool_geometry(
            MAX_MODEL_LEN,
            MAX_NUM_SEQS,
            util=GPU_MEMORY_UTILIZATION,
            chunked_prefill=CHUNKED_PREFILL,
            chunk_tokens=MAX_NUM_BATCHED_TOKENS,
        )
        if (geom.max_model_len, geom.max_num_seqs, geom.max_num_batched_tokens) != (
            MAX_MODEL_LEN,
            MAX_NUM_SEQS,
            MAX_NUM_BATCHED_TOKENS,
        ):
            raise RuntimeError(f"resolved geometry drifted: {geom.describe()}")
        model = Path(MODEL_PATH)
        if not (model / "config.json").is_file():
            raise RuntimeError(
                f"attacker snapshot absent at {MODEL_PATH}; run the explicitly named stage command first"
            )
        key = os.environ.get(API_KEY_ENV, "")
        if not key:
            raise RuntimeError(f"{API_KEY_ENV} secret is absent/empty")

        from sisteer.eval.attacks import attacker_serving as serving

        pool_dir = Path(POOL_DIR)
        pool_dir.mkdir(parents=True, exist_ok=True)
        Path(ENDPOINT_PATH).unlink(missing_ok=True)
        Path(SERVING_MANIFEST).unlink(missing_ok=True)
        VOL.commit()
        log_path = Path(SERVE_LOG)
        self._log_file = log_path.open("w", buffering=1)
        opts = serving.AttackerServingOptions(POOL_QUANTIZATION, POOL_PREFIX_CACHING)
        cmd = [
            "vllm",
            "serve",
            MODEL_PATH,
            "--served-model-name",
            ATTACKER_MODEL,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--gpu-memory-utilization",
            str(GPU_MEMORY_UTILIZATION),
            "--max-model-len",
            str(MAX_MODEL_LEN),
            "--max-num-seqs",
            str(MAX_NUM_SEQS),
            "--max-num-batched-tokens",
            str(MAX_NUM_BATCHED_TOKENS),
            "--enable-chunked-prefill",
            # weights + prefix cache from the one seam (adds --mamba-cache-mode align,
            # the mode vLLM resolves for this model anyway; verified below)
            *opts.vllm_argv(model_role="attacker"),
            "--limit-mm-per-prompt",
            LIMIT_MM_PER_PROMPT,
        ]
        print(
            f"[pool] launching {ATTACKER_MODEL}@{ATTACKER_SNAPSHOT[:8]} "
            f"L={MAX_MODEL_LEN} S={MAX_NUM_SEQS} batched={MAX_NUM_BATCHED_TOKENS} "
            f"chunked_prefill={CHUNKED_PREFILL} serving={json.dumps(opts.as_dict())} text_only=1",
            flush=True,
        )
        self._proc = subprocess.Popen(cmd, stdout=self._log_file, stderr=subprocess.STDOUT)
        _wait_ready(self._proc, log_path)

        # The memory model is only an estimate.  vLLM's post-readiness statement
        # is the fact, and ANY missing/short coverage is fatal in both modes.
        deadline = time.time() + 60
        coverage = None
        last_error = None
        while time.time() < deadline:
            try:
                coverage = _attest_coverage(
                    log_path.read_text(errors="replace"),
                    chunked_prefill=CHUNKED_PREFILL,
                )
                break
            except RuntimeError as exc:
                last_error = exc
                time.sleep(2)
        if coverage is None:
            self._terminate()
            raise last_error or RuntimeError("vLLM coverage attestation missing")

        # The serving identity is a fact of the engine line too: verify it
        # against the request and DERIVE what is advertised from it. A
        # contradiction kills the pool, as the per-job standup does.
        try:
            _, serving_fields = _attest_serving(
                log_path.read_text(errors="replace"), opts,
                manifest_path=SERVING_MANIFEST, log_path=SERVE_LOG,
                endpoint=f"http://127.0.0.1:{PORT}",
                extra={"pool_name": POOL_NAME, "snapshot": ATTACKER_SNAPSHOT})
        except Exception:
            self._terminate()
            raise
        print(f"[pool] serving identity attested from the engine line: "
              f"{json.dumps({k: v for k, v in serving_fields.items() if k != 'serving'})}",
              flush=True)

        unauth = _http_status(f"http://127.0.0.1:{PORT}/v1/models")
        auth = _http_status(f"http://127.0.0.1:{PORT}/v1/models", key=key)
        if unauth not in (401, 403) or auth != 200:
            self._terminate()
            raise RuntimeError(
                f"bearer guard failed: unauthenticated /v1/models={unauth}, authenticated={auth}"
            )

        server_id, server_url = _server_identity()
        app_id = _app_identity()
        call_id = _call_id()
        container_id = os.environ.get("MODAL_TASK_ID", "")
        if not (server_id and server_url and app_id and container_id):
            self._terminate()
            raise RuntimeError(
                "Modal omitted server/app/url/container identity; refusing an unattributable endpoint"
            )
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        liveness = {
            "kind": "modal",
            "app_name": APP_NAME,
            "app_id": app_id,
            "server_name": SERVER_NAME,
            "server_id": server_id,
            "call_id": call_id,
            "container_id": container_id,
        }
        record = {
            "schema_version": contract.SCHEMA_VERSION,
            "source": "endpoint.json",
            "pool_name": POOL_NAME,
            "pool_dir": POOL_DIR,
            "url": server_url,
            "model": ATTACKER_MODEL,
            "snapshot": ATTACKER_SNAPSHOT,
            "model_path": MODEL_PATH,
            "max_model_len": MAX_MODEL_LEN,
            "max_num_seqs": MAX_NUM_SEQS,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
            "cpu_offload_gb": 0,
            "enforce_eager": False,
            "limit_mm_per_prompt": LIMIT_MM_PER_PROMPT,
            # attacker_quantization / prefix_caching / serving: DERIVED from the
            # verified engine line by _attest_serving, never literals
            **serving_fields,
            "api_key_env_name": API_KEY_ENV,
            "api_key_file": None,
            # Stable schema fields: Modal invocation/container identity.
            "job_id": call_id or server_id,
            "node": container_id,
            "started_at": started,
            "serve_log": SERVE_LOG,
            "repo_sha": SOURCE_SHA,
            "runtime": "modal_server",
            "liveness": liveness,
            "geometry": geom.as_dict(),
            "kv_coverage": coverage,
            "chunked_prefill": CHUNKED_PREFILL,
            "vllm_version": "0.25.1+cu129",
        }
        _write_endpoint(record)
        print(
            f"[pool] ADVERTISED app={APP_NAME} app_id={app_id} server_id={server_id} "
            f"call_id={call_id} container_id={container_id} url={server_url}",
            flush=True,
        )
        print(
            "[pool] coverage: "
            f"window={coverage['concurrency_window']} "
            f"max_concurrency={coverage['max_concurrency_reported']}x "
            f"kv_tokens={coverage['kv_tokens_reported']}",
            flush=True,
        )

        self._stop = threading.Event()
        self._heartbeat = threading.Thread(target=self._commit_loop, daemon=True)
        self._heartbeat.start()

    def _commit_loop(self) -> None:
        while not self._stop.wait(10):
            if self._proc.poll() is not None:
                print(f"[pool] FATAL: vLLM exited rc={self._proc.returncode}", flush=True)
                Path(ENDPOINT_PATH).unlink(missing_ok=True)
                VOL.commit()
                return
            # Makes current vLLM stats (including prefix-cache hit rate) visible
            # to the separate canary container.
            VOL.commit()

    def _terminate(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)

    @modal.exit()
    def stop(self) -> None:
        stop = getattr(self, "_stop", None)
        if stop is not None:
            stop.set()
        endpoint = Path(ENDPOINT_PATH)
        if endpoint.is_file():
            try:
                rec = json.loads(endpoint.read_text())
            except Exception:  # noqa: BLE001
                rec = {}
            if (rec.get("liveness") or {}).get("container_id") == os.environ.get("MODAL_TASK_ID"):
                endpoint.unlink(missing_ok=True)
        self._terminate()
        log_file = getattr(self, "_log_file", None)
        if log_file is not None:
            log_file.close()
        VOL.commit()


@tools.function(
    image=client_image,
    volumes={VOL_PATH: VOL},
    secrets=[pool_secret],
    timeout=900,
    cpu=4,
    memory=8192,
)
def canary(n_calls: int = 4, concurrency: int = 1, max_new_tokens: int = 16) -> dict:
    """Prove a separate Modal container can reach and use the guarded pool.

    ``concurrency`` > 1 fires the calls in parallel threads (one client each)
    so the pool's own stats lines show requests in flight; the default is the
    sequential handful."""
    from sisteer.eval.attacks import attacker_pool as contract
    from sisteer.eval.attacks.vllm_attacker import VLLMAttackerClient

    canary_container = os.environ.get("MODAL_TASK_ID", "")
    deadline = time.time() + 600
    last_stale: Exception | None = None
    while True:
        endpoint = _wait_endpoint_file(max(1, deadline - time.time()))

        # FIRST assertion. No liveness lookup, health check, or attacker call
        # occurs if separation is absent or same-container.
        pool_container = _assert_separate_container(endpoint, canary_container)
        ep = contract.PoolEndpoint.from_dict(endpoint)
        try:
            ep, key, evidence = contract.connect_endpoint(
                ep,
                squeue=False,
                env=os.environ,
                expect_model=ATTACKER_MODEL,
                expect_snapshot=ATTACKER_SNAPSHOT,
            )
            break
        except contract.PoolStale as exc:
            last_stale = exc
            if time.time() >= deadline:
                raise RuntimeError(
                    f"no current deployed endpoint appeared within 600s: {last_stale}"
                ) from exc
            time.sleep(5)
    shared_prefix = (
        "This is a benign transport canary. Read the repeated context and answer the final "
        "request briefly. "
        + "shared-prefix-token " * 2500
    )

    def one(i: int) -> dict:
        client = VLLMAttackerClient(ep.url, ep.model, api_key=key, timeout=600, max_retries=1)
        t0 = time.time()
        text = client.chat(
            [
                {"role": "system", "content": shared_prefix},
                {"role": "user", "content": f"Reply with the integer {i}, then list every integer "
                                            f"from {i} to {i + 600}, comma separated."},
            ],
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=int(max_new_tokens),
        )
        return {"index": i, "latency_s": round(time.time() - t0, 3), "reply_chars": len(text or "")}

    started = time.strftime("%m-%d %H:%M:%S", time.gmtime())
    t_all = time.time()
    if int(concurrency) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=int(concurrency)) as ex:
            calls = list(ex.map(one, range(int(n_calls))))
    else:
        calls = [one(i) for i in range(int(n_calls))]
    wall_s = round(time.time() - t_all, 3)

    stats = {"n_stats_lines": 0}
    deadline = time.time() + 90
    while time.time() < deadline:
        time.sleep(10)
        VOL.reload()
        log_path = Path(str(endpoint.get("serve_log") or SERVE_LOG))
        if log_path.is_file():
            stats = contract.log_stats(log_path.read_text(errors="replace"), since=started)
        if stats.get("n_stats_lines") and stats.get("prefix_cache_hit_rate_last") is not None:
            break
    return {
        "ok": True,
        "separate_container": True,
        "pool_container_id": pool_container,
        "canary_container_id": canary_container,
        "modal_app_id": (endpoint.get("liveness") or {}).get("app_id"),
        "modal_call_id": (endpoint.get("liveness") or {}).get("call_id"),
        "modal_server_id": (endpoint.get("liveness") or {}).get("server_id"),
        "endpoint": ep.identity(),
        "verification": evidence,
        "concurrency": int(concurrency),
        "max_new_tokens": int(max_new_tokens),
        "wall_s": wall_s,
        "latency_mean_s": round(sum(c["latency_s"] for c in calls) / max(1, len(calls)), 3),
        "latency_max_s": max((c["latency_s"] for c in calls), default=0.0),
        "calls": calls,
        "pool_logstats": stats,
    }


@tools.function(image=client_image, volumes={VOL_PATH: VOL}, timeout=900)
def endpoint_record() -> dict:
    from sisteer.eval.attacks import attacker_pool as contract

    deadline = time.time() + 600
    last_stale: Exception | None = None
    while time.time() < deadline:
        record = _wait_endpoint_file(max(1, deadline - time.time()))
        ep = contract.PoolEndpoint.from_dict(record)
        try:
            contract.check_modal_live(ep)
            return record
        except contract.PoolStale as exc:
            last_stale = exc
            time.sleep(5)
    raise RuntimeError(f"no current deployed endpoint appeared within 600s: {last_stale}")


@tools.local_entrypoint()
def main(action: str = "canary", n_calls: int = 4, endpoint_out: str = "",
         concurrency: int = 1, max_new_tokens: int = 16) -> None:
    if action == "stage":
        print(json.dumps(stage_weights.remote(), indent=2))
        return
    if action == "canary":
        print(json.dumps(canary.remote(n_calls, concurrency, max_new_tokens), indent=2))
        return
    if action == "endpoint":
        rec = endpoint_record.remote()
        out = Path(endpoint_out).expanduser() if endpoint_out else _local_endpoint_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        rec["api_key_file"] = str(_local_key_path())
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec, indent=2))
        os.replace(tmp, out)
        print(out)
        return
    raise SystemExit("--action must be stage, endpoint, or canary")
