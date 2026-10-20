#!/usr/bin/env python
"""Measure what the lens serving path costs a LoRA cell: the CUDA-graph
(``--enforce-eager``) penalty and native multi-LoRA's own cost over the
passthrough floor, per model, on one GPU, with bare ``vllm serve``.

WHY (2026-09-11). Every LoRA cell of the grid is served through the lens
hook path (``lens_adapters.OP_LORA_OPROJ``): the o_proj hook's cost over an
EAGER passthrough floor is measured (docs/lens_stacked_reductions_scoping.md
§7.2, 1.4-1.8x per step at B=64 in stacked mode), but the eager penalty
itself is not measured anywhere in the repo, and vllm-lens forces
``enforce_eager`` process-wide whenever it is loaded, so no production launch
has ever run with CUDA graphs. Native multi-LoRA (``--enable-lora``) is wired
end to end (vllm_server / proxy / lora_export) and never used in production;
it is the path that would pay neither the hook nor (on a LoRA-only server
without the plugin) the eager cost. This script puts numbers on both.

ARMS (each a fresh ``vllm serve`` on the same GPU, same scheduler geometry):

  pt_eager_plugin     passthrough, ``--enforce-eager``, vllm-lens plugin
                      LOADED — today's production floor (the lens server
                      with no steering selected)
  pt_eager_noplugin   passthrough, ``--enforce-eager``, plugin kept out
                      (``VLLM_PLUGINS=""``) — separates the plugin's
                      presence from eager mode
  pt_graphs           passthrough, CUDA graphs ON (no ``--enforce-eager``),
                      plugin out — what a plugin-free server decodes at
  lora_graphs         a rank-4 o_proj adapter on every attention layer,
                      served natively (``--enable-lora --lora-modules``),
                      CUDA graphs ON, plugin out — the candidate path
  lora_eager_noplugin the same adapter under ``--enforce-eager`` — isolates
                      native LoRA's own cost from the graph win

THE ADAPTER IS SYNTHETIC. Its A/B tensors are random (seeded), shaped like
the grid's ``lora_all`` objects (rank 4, one adapter per attention-layer
o_proj: every layer on gemma-4, the four attention blocks of Nemotron-H).
Throughput does not depend on the weights, only on the module set and the
rank, so this measures the path, not any object. A greedy probe per arm is
recorded so a LoRA arm whose output equals passthrough (an adapter that
silently did not apply) is visible in the results — ``lora_bites``.

MEASUREMENT. Closed loop: B concurrent clients, each re-issuing a chat
completion of EXACTLY ``max_tokens`` new tokens (``min_tokens`` +
``ignore_eos``), so every step of the window decodes B rows. After a warmup,
the window's generated-token count is read from the server's own
``/metrics`` counter (``vllm:generation_tokens_total``), so request
boundaries do not bias it; the client-side count is a cross-check. The
running-request gauge is sampled to attest that B was actually reached, and
``nvidia-smi`` utilization is sampled for SM duty. ms/step = B / (tok/s).

Nothing here touches the lens registry, the proxy, or any lane's store. It
runs inside the pinned serving venv on a GPU node (the sbatch beside it).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: the two targets, pinned as the lanes pin them (scripts/derive_model_config.py)
MODELS: Dict[str, Dict[str, Any]] = {
    "gemma4e4b": {
        "model": "google/gemma-4-E4B-it",
        "revision": "ee0ef6023621cff504d758262d4e04895a5af4a2",
        # the VLM wrapper's decoder sits under language_model (models._LAYER_PATHS)
        "decoder_prefix": "model.language_model.layers",
        "oproj_site": "self_attn.o_proj",
        # dense: every block carries attention; prefix caching ON in production
        "hybrid": False,
    },
    "nemotron3nano4b": {
        "model": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "revision": "dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
        "decoder_prefix": "model.layers",
        "oproj_site": "mixer.o_proj",
        # NemotronH: 4 attention blocks among 38 Mamba2/MLP blocks (the '*'
        # entries of hybrid_override_pattern); prefix caching structurally
        # OFF on this stack (vllm_server.hybrid_recurrent_model)
        "hybrid": True,
    },
}

ADAPTER_NAME = "bench_lora_r4"
PROMPT = (
    "You are helping a colleague prepare a short internal report. Summarize, "
    "in a few paragraphs, the trade-offs between eager kernel launches and "
    "captured execution graphs for autoregressive decoding on a modern GPU, "
    "covering launch overhead, memory-bandwidth limits, batch size, and the "
    "kinds of runtime hooks that cannot survive graph capture. Then list the "
    "measurements you would run to decide between the two for a small model "
    "served at moderate concurrency, and say what each measurement would "
    "rule in or out."
)

# scheduler pins copied from sisteer.capeval.serving.vllm_server (the no-chunking
# identity + the explicit synchronous scheduler), restated here so this script
# needs nothing from the package inside the serving venv
NO_CHUNK_PINS = ("--long-prefill-token-threshold", "0",
                 "--max-num-partial-prefills", "1",
                 "--max-long-partial-prefills", "1")
SYNC_SCHEDULING_FLAG = "--no-async-scheduling"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# --------------------------------------------------------------------------- #
# the synthetic rank-4 o_proj adapter                                          #
# --------------------------------------------------------------------------- #

def load_hf_config(model: str, revision: str) -> Dict[str, Any]:
    """The pinned snapshot's config.json from the LOCAL cache (offline)."""
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(model, "config.json", revision=revision,
                        local_files_only=True)
    return json.loads(Path(p).read_text())


def attention_layers(key: str, cfg: Dict[str, Any]) -> List[int]:
    if key == "nemotron3nano4b":
        pat = cfg["hybrid_override_pattern"]
        layers = [i for i, ch in enumerate(pat) if ch == "*"]
        if len(layers) != 4:
            raise RuntimeError(f"expected 4 attention blocks in {pat!r}, "
                               f"got {layers}")
        return layers
    text = cfg.get("text_config", cfg)
    return list(range(int(text["num_hidden_layers"])))


def oproj_shape(cfg: Dict[str, Any]) -> Tuple[int, int]:
    """(in_features, out_features) of o_proj: heads x head_dim -> hidden."""
    text = cfg.get("text_config", cfg)
    heads = int(text["num_attention_heads"])
    head_dim = int(text.get("head_dim") or text["hidden_size"] // heads)
    return heads * head_dim, int(text["hidden_size"])


def export_random_adapter(out_dir: Path, *, key: str, spec: Dict[str, Any],
                          rank: int, seed: int, a_std: float = 0.02,
                          b_std: float = 0.2) -> Dict[str, Any]:
    """A PEFT-layout adapter dir (adapter_config.json + safetensors) with
    seeded random A/B on every attention layer's o_proj — the module set of
    the grid's lora_all objects, the same layout lora_export writes.

    ``b_std`` sizes the perturbation: with A at 0.02 the read ``A x`` has std
    ~0.02 * sqrt(in_features) per unit of x (~1.4 on Nemotron's 5120-wide
    o_proj input), so B at 0.2 writes a ~0.6x-of-x edit into the o_proj output
    — large enough that the greedy probe MUST diverge from passthrough if the
    adapter applies. The first pass used 0.02 for both and its probe did not
    diverge (an edit ~20x smaller), which left ``lora_bites`` uninformative."""
    import torch
    from safetensors.torch import save_file

    cfg = load_hf_config(spec["model"], spec["revision"])
    layers = attention_layers(key, cfg)
    in_f, out_f = oproj_shape(cfg)
    g = torch.Generator().manual_seed(seed)
    tensors = {}
    modules = []
    for i in layers:
        mod = f"{spec['decoder_prefix']}.{i}.{spec['oproj_site']}"
        modules.append(mod)
        tensors[f"base_model.model.{mod}.lora_A.weight"] = (
            torch.randn((rank, in_f), generator=g) * a_std)
        tensors[f"base_model.model.{mod}.lora_B.weight"] = (
            torch.randn((out_f, rank), generator=g) * b_std)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": spec["model"],
        "r": rank,
        "lora_alpha": float(rank),  # scaling alpha/r = 1
        "lora_dropout": 0.0,
        "bias": "none",
        "fan_in_fan_out": False,
        "target_modules": ["o_proj"],
        "task_type": "CAUSAL_LM",
        "bench_synthetic_adapter": True,
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(config, indent=2))
    save_file({k: v.contiguous() for k, v in tensors.items()},
              str(out_dir / "adapter_model.safetensors"))
    # verify the module names against a weight-free skeleton when transformers
    # can build one (informational: the live vLLM load is the real check)
    skeleton_ok: Optional[bool] = None
    skeleton_err = None
    try:
        from transformers import AutoConfig
        import transformers
        hf_cfg = AutoConfig.from_pretrained(spec["model"], revision=spec["revision"],
                                            local_files_only=True)
        arch = (hf_cfg.architectures or [""])[0]
        cls = getattr(transformers, arch)
        with torch.device("meta"):
            sk = cls._from_config(hf_cfg) if hasattr(cls, "_from_config") \
                else cls(hf_cfg)
        names = {n for n, _ in sk.named_modules()}
        missing = [m for m in modules if m not in names]
        skeleton_ok = not missing
        if missing:
            skeleton_err = f"{len(missing)} adapter module names absent from the " \
                           f"skeleton, e.g. {missing[:2]}"
    except Exception as e:  # noqa: BLE001 — informational only
        skeleton_err = f"{type(e).__name__}: {e}"
    return {"adapter_dir": str(out_dir), "rank": rank, "n_modules": len(modules),
            "a_std": a_std, "b_std": b_std,
            "layers": layers, "in_features": in_f, "out_features": out_f,
            "modules_head": modules[:2], "skeleton_ok": skeleton_ok,
            "skeleton_err": skeleton_err}


# --------------------------------------------------------------------------- #
# server lifecycle                                                             #
# --------------------------------------------------------------------------- #

@dataclass
class Arm:
    name: str
    eager: bool
    plugin: bool           # vllm-lens plugin loaded (production) or kept out
    lora: bool
    #: per-arm ``--max-lora-rank`` (None -> the CLI's, the composer's 64 ceiling).
    #: vLLM pads every adapter's LoRA matmul to this rank, so a rank-4 adapter
    #: under a 64 cap does 16x the LoRA FLOPs it needs; the r8 arm shows the
    #: cost at the smallest legal cap above the adapter's rank.
    max_lora_rank: Optional[int] = None


ARMS: Dict[str, Arm] = {a.name: a for a in (
    Arm("pt_eager_plugin", eager=True, plugin=True, lora=False),
    Arm("pt_eager_noplugin", eager=True, plugin=False, lora=False),
    Arm("pt_graphs", eager=False, plugin=False, lora=False),
    Arm("lora_graphs", eager=False, plugin=False, lora=True),
    Arm("lora_graphs_r8", eager=False, plugin=False, lora=True, max_lora_rank=8),
    Arm("lora_eager_noplugin", eager=True, plugin=False, lora=True),
)}


def compose_argv(arm: Arm, spec: Dict[str, Any], *, port: int, max_model_len: int,
                 max_num_seqs: int, gpu_mem_util: float, adapter_dir: Optional[Path],
                 max_lora_rank: int, prefix_caching: bool) -> List[str]:
    cmd = ["vllm", "serve", spec["model"],
           "--revision", spec["revision"],
           "--dtype", "bfloat16",
           "--host", "127.0.0.1", "--port", str(port),
           "--max-model-len", str(max_model_len),
           "--max-num-seqs", str(max_num_seqs),
           "--max-num-batched-tokens", str(max_num_seqs * max_model_len),
           *NO_CHUNK_PINS, SYNC_SCHEDULING_FLAG,
           "--gpu-memory-utilization", str(gpu_mem_util),
           "--seed", "0",
           "--no-enable-auto-tool-choice",
           "--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching",
           ]
    if arm.eager:
        cmd.append("--enforce-eager")
    if arm.lora:
        if adapter_dir is None:
            raise RuntimeError("lora arm without an adapter")
        cmd += ["--enable-lora", "--max-loras", "1",
                "--max-lora-rank", str(arm.max_lora_rank or max_lora_rank),
                "--lora-modules", f"{ADAPTER_NAME}={adapter_dir}"]
    return cmd


def http_json(url: str, body: Optional[dict] = None, timeout: float = 30.0) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def http_text(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode()


def wait_ready(base: str, proc: subprocess.Popen, timeout_s: float) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm serve exited early with rc={proc.returncode}")
        try:
            http_json(base + "/v1/models", timeout=5)
            return
        except Exception:  # noqa: BLE001
            time.sleep(3)
    raise RuntimeError(f"server not ready after {timeout_s:.0f}s")


def stop_server(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=30)


def gpu_index() -> Optional[str]:
    """The physical index Slurm handed us (never set; only read)."""
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    first = v.split(",")[0].strip()
    return first if first.isdigit() else None


def gpu_mem_used_mib() -> Optional[int]:
    idx = gpu_index()
    cmd = ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    if idx is not None:
        cmd += ["-i", idx]
    try:
        out = subprocess.check_output(cmd, text=True, timeout=20).strip().splitlines()
        return int(out[0])
    except Exception:  # noqa: BLE001
        return None


def wait_gpu_free(threshold_mib: int = 6000, timeout_s: float = 120) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        used = gpu_mem_used_mib()
        if used is None or used < threshold_mib:
            return
        time.sleep(3)
    log(f"WARNING: GPU still holds {gpu_mem_used_mib()} MiB after {timeout_s:.0f}s")


def server_log_facts(path: Path) -> Dict[str, Any]:
    """Attest from the engine's own log what mode it resolved — the exact
    lines vLLM 0.25.1 emits (read off the first pass's logs): a graphs launch
    logs ``Graph capturing finished in N secs``; an eager launch logs
    ``Enforce eager set, disabling torch.compile and CUDAGraphs``; an adapter
    load logs ``Loaded new LoRA adapter: name '<name>'``. (The first pass's
    looser ``cudagraph`` regex matched the config dump in every log.)"""
    txt = path.read_text(errors="replace") if path.exists() else ""
    return {
        "plugin_loaded": bool(re.search(r"Loading plugin activations|vllm_lens", txt)),
        "cuda_graphs": bool(re.search(r"Graph capturing finished in", txt)),
        "enforce_eager_true": bool(re.search(r"Enforce eager set, disabling", txt)),
        "lora_loaded": bool(re.search(
            rf"Loaded new LoRA adapter: name '{re.escape(ADAPTER_NAME)}'", txt)),
        "log_bytes": len(txt),
    }


# --------------------------------------------------------------------------- #
# metrics + load                                                               #
# --------------------------------------------------------------------------- #

METRIC_RE = re.compile(r"^(vllm:[a-z_]+)(\{[^}]*\})?\s+([0-9.eE+-]+)\s*$", re.M)


def read_metrics(base: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    try:
        txt = http_text(base + "/metrics")
    except Exception:  # noqa: BLE001
        return out
    for name, _labels, val in METRIC_RE.findall(txt):
        try:
            out[name] = out.get(name, 0.0) + float(val)
        except ValueError:
            pass
    return out


class UtilSampler:
    """1 Hz nvidia-smi utilization on our device, for the window."""

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self.path: Optional[Path] = None

    def start(self, path: Path) -> None:
        idx = gpu_index()
        cmd = ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu,memory.used",
               "--format=csv,noheader,nounits", "-lms", "1000"]
        if idx is not None:
            cmd += ["-i", idx]
        self.path = path
        self.proc = subprocess.Popen(cmd, stdout=path.open("w"), stderr=subprocess.DEVNULL)

    def stop(self) -> Dict[str, Any]:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        utils: List[float] = []
        if self.path and self.path.exists():
            for line in self.path.read_text().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[1].replace(".", "", 1).isdigit():
                    utils.append(float(parts[1]))
        return {"sm_util_mean": statistics.fmean(utils) if utils else None,
                "sm_util_n": len(utils)}


@dataclass
class Point:
    arm: str
    B: int
    rep: int
    max_tokens: int
    window_s: float
    server_tokens: Optional[float]
    server_tok_s: Optional[float]
    ms_per_step: Optional[float]
    client_tokens: int
    client_tok_s: float
    requests_done: int
    running_mean: Optional[float]
    running_max: Optional[float]
    waiting_max: Optional[float]
    latency_p50_s: Optional[float]
    latency_p95_s: Optional[float]
    sm_util_mean: Optional[float]
    errors: int
    error_sample: Optional[str] = None


async def closed_loop(base: str, model: str, *, B: int, max_tokens: int,
                      warmup_s: float, window_s: float, sample_s: float = 2.0
                      ) -> Tuple[Dict[str, Any], List[Dict[str, float]]]:
    import aiohttp

    body = {"model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens, "min_tokens": max_tokens,
            "ignore_eos": True, "temperature": 0.0, "stream": False}
    stop = asyncio.Event()
    done: List[Tuple[float, float, int]] = []   # (t_end, latency, completion_tokens)
    errors: List[str] = []
    gauges: List[Dict[str, float]] = []

    async def worker(session: aiohttp.ClientSession) -> None:
        while not stop.is_set():
            t0 = time.monotonic()
            try:
                async with session.post(base + "/v1/chat/completions", json=body,
                                        timeout=aiohttp.ClientTimeout(total=600)) as r:
                    if r.status != 200:
                        errors.append(f"HTTP {r.status}: {(await r.text())[:200]}")
                        await asyncio.sleep(1)
                        continue
                    j = await r.json()
                toks = int(j.get("usage", {}).get("completion_tokens", 0))
                done.append((time.monotonic(), time.monotonic() - t0, toks))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{type(e).__name__}: {e}"[:200])
                await asyncio.sleep(1)

    async def sampler() -> None:
        loop = asyncio.get_running_loop()
        while not stop.is_set():
            m = await loop.run_in_executor(None, read_metrics, base)
            gauges.append({"t": time.monotonic(),
                           "running": m.get("vllm:num_requests_running"),
                           "waiting": m.get("vllm:num_requests_waiting"),
                           "gen_total": m.get("vllm:generation_tokens_total")})
            await asyncio.sleep(sample_s)

    connector = aiohttp.TCPConnector(limit=B + 8)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [asyncio.create_task(worker(session)) for _ in range(B)]
        stask = asyncio.create_task(sampler())
        await asyncio.sleep(warmup_s)
        m0 = read_metrics(base)
        t_start = time.monotonic()
        await asyncio.sleep(window_s)
        m1 = read_metrics(base)
        t_end = time.monotonic()
        stop.set()
        # let in-flight requests drain so the server is quiet before teardown
        await asyncio.wait(tasks, timeout=600)
        stask.cancel()
        for t in tasks:
            t.cancel()

    win = [(te, lat, tk) for te, lat, tk in done if t_start <= te <= t_end]
    w = t_end - t_start
    g0, g1 = m0.get("vllm:generation_tokens_total"), m1.get("vllm:generation_tokens_total")
    server_tokens = (g1 - g0) if (g0 is not None and g1 is not None) else None
    gw = [g for g in gauges if t_start <= g["t"] <= t_end]
    running = [g["running"] for g in gw if g.get("running") is not None]
    waiting = [g["waiting"] for g in gw if g.get("waiting") is not None]
    lats = sorted(lat for _, lat, _ in win)
    stats = {
        "window_s": w,
        "server_tokens": server_tokens,
        "server_tok_s": (server_tokens / w) if server_tokens is not None else None,
        "client_tokens": sum(tk for _, _, tk in win),
        "client_tok_s": sum(tk for _, _, tk in win) / w,
        "requests_done": len(win),
        "running_mean": statistics.fmean(running) if running else None,
        "running_max": max(running) if running else None,
        "waiting_max": max(waiting) if waiting else None,
        "latency_p50_s": lats[len(lats) // 2] if lats else None,
        "latency_p95_s": lats[int(len(lats) * 0.95)] if lats else None,
        "errors": len(errors),
        "error_sample": errors[0] if errors else None,
    }
    return stats, gw


def greedy_probe(base: str, model: str, n: int = 32) -> str:
    j = http_json(base + "/v1/chat/completions",
                  {"model": model, "messages": [{"role": "user", "content": PROMPT}],
                   "max_tokens": n, "temperature": 0.0}, timeout=300)
    return j["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-key", required=True, choices=sorted(MODELS))
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--arms", default=",".join(ARMS),
                    help=f"comma list from {sorted(ARMS)}")
    ap.add_argument("--batches", default="8,32,64")
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--warmup-s", type=float, default=20.0)
    ap.add_argument("--window-s", type=float, default=60.0)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-num-seqs", type=int, default=64)
    ap.add_argument("--gpu-mem-util", type=float, default=0.9)
    ap.add_argument("--max-lora-rank", type=int, default=64,
                    help="the composer's ceiling (vllm_server: max_rank = 64)")
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--adapter-b-std", type=float, default=0.2,
                    help="std of the synthetic adapter's B (A is 0.02); see export_random_adapter")
    ap.add_argument("--repeats", type=int, default=1,
                    help="measurement windows per (arm, B), back to back — a noise estimate")
    ap.add_argument("--ready-timeout-s", type=float, default=1500.0)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    key = args.model_key
    spec = MODELS[key]
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    arms = [ARMS[a] for a in args.arms.split(",") if a]
    batches = [int(b) for b in args.batches.split(",") if b]
    if max(batches) > args.max_num_seqs:
        raise SystemExit(f"a batch {max(batches)} exceeds --max-num-seqs "
                         f"{args.max_num_seqs}: the client would queue, not batch")
    prefix_caching = not spec["hybrid"]

    facts: Dict[str, Any] = {
        "model_key": key, "spec": spec, "argv": sys.argv,
        "geometry": {"max_model_len": args.max_model_len,
                     "max_num_seqs": args.max_num_seqs,
                     "max_num_batched_tokens": args.max_model_len * args.max_num_seqs,
                     "prefix_caching": prefix_caching},
        "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": sys.executable,
    }
    try:
        import vllm, torch  # noqa: E401
        facts["vllm_version"] = vllm.__version__
        facts["torch_version"] = torch.__version__
        facts["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as e:  # noqa: BLE001
        facts["version_error"] = repr(e)
    try:
        facts["driver"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=20).strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        pass

    log(f"model {key} = {spec['model']} @ {spec['revision'][:12]}; arms "
        f"{[a.name for a in arms]}; B {batches}; geometry "
        f"{args.max_num_seqs} x {args.max_model_len}")
    adapter_dir = out / "adapter"
    facts["adapter"] = export_random_adapter(adapter_dir, key=key, spec=spec,
                                             rank=args.rank, seed=args.seed,
                                             b_std=args.adapter_b_std)
    facts["repeats"] = args.repeats
    log(f"adapter: {facts['adapter']}")

    points: List[Point] = []
    arm_records: Dict[str, Any] = {}
    probes: Dict[str, str] = {}
    failures: List[str] = []
    for arm in arms:
        adir = out / arm.name
        adir.mkdir(exist_ok=True)
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        cmd = compose_argv(arm, spec, port=port, max_model_len=args.max_model_len,
                           max_num_seqs=args.max_num_seqs, gpu_mem_util=args.gpu_mem_util,
                           adapter_dir=adapter_dir if arm.lora else None,
                           max_lora_rank=args.max_lora_rank, prefix_caching=prefix_caching)
        env = dict(os.environ)
        if not arm.plugin:
            # an EMPTY allow-list: vllm's envs splits it to [''] and no plugin
            # name matches, so vllm-lens (the general plugin that forces
            # enforce_eager process-wide) is not imported
            env["VLLM_PLUGINS"] = ""
        else:
            env.pop("VLLM_PLUGINS", None)
        rec: Dict[str, Any] = {"cmd": cmd, "VLLM_PLUGINS": env.get("VLLM_PLUGINS"),
                               "eager": arm.eager, "plugin": arm.plugin, "lora": arm.lora}
        (adir / "cmd.json").write_text(json.dumps(rec, indent=2))
        log(f"=== arm {arm.name}: {' '.join(cmd)}  [VLLM_PLUGINS={env.get('VLLM_PLUGINS')!r}]")
        wait_gpu_free()
        slog = adir / "vllm_server.log"
        t_launch = time.time()
        proc = subprocess.Popen(cmd, env=env, stdout=slog.open("w"),
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            wait_ready(base, proc, args.ready_timeout_s)
            rec["ready_s"] = time.time() - t_launch
            log(f"ready in {rec['ready_s']:.0f}s")
            model_name = ADAPTER_NAME if arm.lora else spec["model"]
            served = [m["id"] for m in http_json(base + "/v1/models")["data"]]
            rec["served_models"] = served
            if model_name not in served:
                raise RuntimeError(f"{model_name!r} not in served models {served}")
            probes[arm.name] = greedy_probe(base, model_name)
            rec["probe_head"] = probes[arm.name][:120]
            for B, rep in ((B, r) for B in batches for r in range(args.repeats)):
                sampler = UtilSampler()
                sampler.start(adir / f"util_B{B}_r{rep}.csv")
                stats, gauges = asyncio.run(closed_loop(
                    base, model_name, B=B, max_tokens=args.max_tokens,
                    warmup_s=args.warmup_s, window_s=args.window_s))
                util = sampler.stop()
                (adir / f"gauges_B{B}_r{rep}.json").write_text(json.dumps(gauges))
                tok_s = stats["server_tok_s"]
                pt = Point(arm=arm.name, B=B, rep=rep, max_tokens=args.max_tokens,
                           window_s=stats["window_s"], server_tokens=stats["server_tokens"],
                           server_tok_s=tok_s,
                           ms_per_step=(1000.0 * B / tok_s) if tok_s else None,
                           client_tokens=stats["client_tokens"],
                           client_tok_s=stats["client_tok_s"],
                           requests_done=stats["requests_done"],
                           running_mean=stats["running_mean"], running_max=stats["running_max"],
                           waiting_max=stats["waiting_max"],
                           latency_p50_s=stats["latency_p50_s"],
                           latency_p95_s=stats["latency_p95_s"],
                           sm_util_mean=util["sm_util_mean"], errors=stats["errors"],
                           error_sample=stats["error_sample"])
                points.append(pt)
                log(f"  B={B:>3} r{rep}: {tok_s if tok_s is None else round(tok_s, 1)} tok/s "
                    f"(client {pt.client_tok_s:.1f}), {pt.ms_per_step and round(pt.ms_per_step, 1)} "
                    f"ms/step, running mean {pt.running_mean}, SM {pt.sm_util_mean}, "
                    f"errors {pt.errors}")
                # checkpoint after every point so a killed job still leaves data
                (out / "results.json").write_text(json.dumps(
                    {"facts": facts, "arms": arm_records | {arm.name: rec},
                     "points": [asdict(p) for p in points], "probes": probes,
                     "failures": failures, "complete": False}, indent=2))
        except Exception as e:  # noqa: BLE001
            tail = slog.read_text(errors="replace")[-3000:] if slog.exists() else ""
            rec["error"] = f"{type(e).__name__}: {e}"
            failures.append(f"{arm.name}: {rec['error']}")
            log(f"ARM FAILED {arm.name}: {rec['error']}\n--- server log tail ---\n{tail}")
        finally:
            stop_server(proc)
            rec["log_facts"] = server_log_facts(slog)
            arm_records[arm.name] = rec
            wait_gpu_free()

    # did the adapter bite? (a LoRA probe equal to the passthrough probe means
    # the adapter did not apply — the silent failure this records)
    pt_ref = probes.get("pt_graphs") or probes.get("pt_eager_noplugin") or probes.get("pt_eager_plugin")
    lora_bites = {a: (probes[a] != pt_ref) for a in probes if a.startswith("lora")} \
        if pt_ref is not None else {}
    result = {"facts": facts, "arms": arm_records, "points": [asdict(p) for p in points],
              "probes": probes, "lora_bites": lora_bites, "failures": failures,
              "complete": True}
    (out / "results.json").write_text(json.dumps(result, indent=2))

    # the human-readable table
    lines = [f"# {key}: eager vs CUDA graphs vs native LoRA ({facts.get('gpu_name')}, "
             f"vllm {facts.get('vllm_version')})", "",
             "| arm | B | rep | tok/s | ms/step | running mean | SM % | errors |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for p in points:
        lines.append(f"| {p.arm} | {p.B} | {p.rep} | {p.server_tok_s and round(p.server_tok_s, 1)} | "
                     f"{p.ms_per_step and round(p.ms_per_step, 1)} | "
                     f"{p.running_mean and round(p.running_mean, 1)} | "
                     f"{p.sm_util_mean and round(p.sm_util_mean, 1)} | {p.errors} |")
    lines += ["", f"lora_bites: {lora_bites}", f"failures: {failures}"]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    if failures:
        log(f"{len(failures)} arm(s) FAILED — a step that cannot run is a failure")
        return 1
    if not points:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
