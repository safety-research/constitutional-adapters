#!/usr/bin/env python3
"""The NEMOTRON FUSED-KERNEL BACKEND ATTESTATION (the standing Nemotron rule: no quiet
pure-torch SSM path in the serving process).

Driven by ``nemo_backend_probe.sbatch`` on a GPU node with the Nemotron serving-venv variant
(``f22_serving_venv nemotron``: the campaign's pinned vLLM stack + ``mamba-ssm`` / ``causal-conv1d`` at the fit
env's pins). Three sub-commands, in the sbatch's order:

  packages      the venv imports ``mamba_ssm`` + ``causal_conv1d`` (versions recorded) — the rule's letter;
  dispatch      the STATIC dispatch map of the code the served NemotronH model resolves to: vLLM's
                ``nemotron_h`` model module -> its Mamba mixer class -> the kernel symbols its forward path names
                (``causal_conv1d_fn`` / ``causal_conv1d_update`` / ``mamba_chunk_scan_combined`` /
                ``selective_state_update``), each recorded with the module it resolves to and whether that module
                is one of vLLM's own fused (Triton / custom-op) kernel modules or the pip packages'; any pure-torch
                fallback symbol reachable from the mixer is recorded — the rule's intent. Also asks the LIVE
                workers ``sisteer_mamba_backend_report`` when a standup is up (the RPC the plan names; on this tree
                the worker does not answer it — recorded as such, never fatal);
  duty          the measured SM duty + tok/s of a 64-sequence x 512-token decode burst through the live standup,
                sampled by ``nvidia-smi dmon -s u`` at 1 Hz on the job's GPU — run TWICE by the sbatch, on the same
                GPU in the same job: first against a same-size hybrid PEER served on the campaign's base venv at
                Nemotron's served concurrency (the reference), then against Nemotron on the variant venv;
  attest        folds the records into ``mamba_backend_attestation.json``: ADMISSION = fused on every Mamba symbol
                AND Nemotron's mean SM duty >= DUTY_RATIO (0.5) x the peer's AND its tok/s >= DUTY_RATIO x the
                peer's, both bursts identical (n_seqs, max_tokens, GPU). The served path runs EAGER (vllm-lens forces
                --enforce-eager; CUDA graphs off), so every model's decode is launch-bound and an ABSOLUTE duty floor
                calibrates nothing: the first probe measured 0.28 on Nemotron against a 0.90 floor with vLLM's Triton
                kernels attested. A pure-torch SSM path is 9-60x slower, so half the peer's rate is a wide
                margin that still catches it. ``common.sh f22_model_facts`` requires this file
                (F22_REQUIRE_NEMO_BACKEND=1) for every Nemotron math500 job, RE-DERIVES the rule from the recorded
                numbers, and keys it to the serving venv's requirements sha, so a rebuilt venv needs a fresh attestation.

Honest gap, recorded in the file: without a worker RPC the per-LAYER dispatch of the LIVE engine is inferred
from the resolved code the engine imports, not read from the running worker; the RPC (``lens_worker.py``, not
this lane's surface) is what would close it, and this probe consumes it the moment it exists.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

#: the kernel symbols a NemotronH Mamba2 mixer's forward path dispatches through (vLLM's MambaMixer2)
KERNEL_SYMBOLS = ("causal_conv1d_fn", "causal_conv1d_update", "mamba_chunk_scan_combined", "selective_state_update")
#: module prefixes that ARE fused kernels (vLLM's own Triton / custom-op implementations, or the pip packages')
FUSED_MODULE_PREFIXES = ("vllm.model_executor.layers.mamba.ops", "vllm._custom_ops", "vllm._C",
                         "mamba_ssm.ops", "causal_conv1d")
#: names that would mean a pure-torch path
FALLBACK_MARKERS = ("naive", "torch_fallback", "reference", "fallback", "_ref", "slow_forward")
RPC_METHOD = "sisteer_mamba_backend_report"
#: admission: Nemotron's duty AND tok/s >= this fraction of the same-GPU peer's under the identical burst
DUTY_RATIO = 0.5
ATTESTATION_FILENAME = "mamba_backend_attestation.json"


def _write(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
    tmp.replace(path)


def cmd_packages(a) -> int:
    rec: dict[str, Any] = {"python": sys.executable, "ok": False}
    try:
        import causal_conv1d  # type: ignore
        import mamba_ssm  # type: ignore
        rec.update(mamba_ssm_version=getattr(mamba_ssm, "__version__", None),
                   causal_conv1d_version=getattr(causal_conv1d, "__version__", None),
                   mamba_ssm_file=getattr(mamba_ssm, "__file__", None),
                   causal_conv1d_file=getattr(causal_conv1d, "__file__", None))
        from causal_conv1d import causal_conv1d_fn  # noqa: F401
        from mamba_ssm.ops.triton.ssd_combined import (
            mamba_chunk_scan_combined,  # noqa: F401
        )
        rec["ok"] = True
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    try:
        import vllm  # type: ignore
        rec["vllm_version"] = getattr(vllm, "__version__", None)
    except Exception as e:
        rec["vllm_error"] = f"{type(e).__name__}: {e}"
    _write(Path(a.out) / "packages.json", rec)
    print(json.dumps(rec, indent=1, default=str))
    return 0 if rec["ok"] else 1


def _classify(symbol: str, obj: Any) -> dict[str, Any]:
    mod = getattr(obj, "__module__", None) or ""
    fused = any(mod.startswith(p) for p in FUSED_MODULE_PREFIXES)
    src_hint = None
    try:
        src = inspect.getsource(obj)
        src_hint = {"triton_jit_call": ("[grid]" in src or "triton" in src.lower()),
                    "custom_op": ("torch.ops" in src or "_custom_ops" in src),
                    "fallback_marker": any(m in src.lower() for m in FALLBACK_MARKERS)}
    except (OSError, TypeError):
        pass
    return {"symbol": symbol, "module": mod, "qualname": getattr(obj, "__qualname__", None),
            "fused_module": fused, "source_hints": src_hint}


def static_dispatch_map() -> dict[str, Any]:
    """The kernel symbols vLLM's NemotronH Mamba mixer resolves at import — the dispatch a worker in this venv
    would run. Records the model module, the mixer class, each symbol's module and fused-ness, and any
    fallback-marked symbol reachable from the mixer module's namespace."""
    rec: dict[str, Any] = {"model_module": None, "mixer_module": None, "mixer_class": None, "symbols": [],
                           "fallback_symbols": [], "ok": False}
    try:
        model_mod = importlib.import_module("vllm.model_executor.models.nemotron_h")
        rec["model_module"] = model_mod.__file__
    except Exception as e:
        rec["error"] = f"cannot import vllm.model_executor.models.nemotron_h: {type(e).__name__}: {e}"
        return rec
    try:
        mixer_mod = importlib.import_module("vllm.model_executor.layers.mamba.mamba_mixer2")
    except Exception:
        rec["error"] = "cannot import vLLM's Mamba2 mixer module"
        return rec
    rec["mixer_module"] = mixer_mod.__file__
    for name, obj in vars(mixer_mod).items():
        if inspect.isclass(obj) and "Mixer" in name and "Mamba" in name:
            rec["mixer_class"] = f"{obj.__module__}.{obj.__qualname__}"
    for sym in KERNEL_SYMBOLS:
        obj = getattr(mixer_mod, sym, None)
        if obj is None:
            rec["symbols"].append({"symbol": sym, "module": None, "fused_module": False, "absent": True})
            continue
        rec["symbols"].append(_classify(sym, obj))
    for name, obj in vars(mixer_mod).items():
        if callable(obj) and any(m in name.lower() for m in FALLBACK_MARKERS):
            rec["fallback_symbols"].append({"name": name, "module": getattr(obj, "__module__", None)})
    present = [s for s in rec["symbols"] if not s.get("absent")]
    rec["ok"] = bool(present) and all(s["fused_module"] for s in present) and not rec["fallback_symbols"]
    rec["nemotron_h_uses_mixer"] = any(
        rec["mixer_class"] and rec["mixer_class"].rsplit(".", 1)[-1] in line
        for line in Path(model_mod.__file__).read_text(encoding="utf-8", errors="replace").splitlines()) if rec["mixer_class"] else None
    return rec


def live_worker_report(endpoint_json: str | None) -> dict[str, Any]:
    """Ask the live workers for the plan's RPC over the standup's authenticated collective_rpc route (the same
    route mt_serving_up uses for mrope). Absent RPC / no standup => recorded, never fatal."""
    if not endpoint_json or not Path(endpoint_json).is_file():
        return {"asked": False, "why": "no endpoint.json — no live standup to ask"}
    try:
        ep = json.loads(Path(endpoint_json).read_text())
        att = json.loads(Path(ep["steering_attestation"]).read_text())
        base_url = f"http://127.0.0.1:{ep['vllm_port']}"
        token = (att.get("launch") or {}).get("api_key") or ep.get("vllm_api_key")
        body = json.dumps({"method": RPC_METHOD, "args": []}).encode()
        req = urllib.request.Request(f"{base_url}/collective_rpc", data=body,
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": f"Bearer {token}"} if token else {})}, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode())
        return {"asked": True, "rpc_method": RPC_METHOD, "reports": payload.get("results")}
    except Exception as e:
        return {"asked": True, "rpc_method": RPC_METHOD, "reports": None,
                "why": f"{type(e).__name__}: {str(e)[:300]} (the worker on this tree may not implement {RPC_METHOD}; "
                       "lens_worker.py is outside lane B's surface — the static dispatch map stands in)"}


def cmd_dispatch(a) -> int:
    rec = {"static": static_dispatch_map(), "live_worker": live_worker_report(a.endpoint_json)}
    _write(Path(a.out) / "dispatch.json", rec)
    print(json.dumps(rec, indent=1, default=str))
    return 0 if rec["static"]["ok"] else 1


class _Dmon:
    """``nvidia-smi dmon -s u -d 1`` on the job's GPU, parsed to SM% samples."""

    def __init__(self, gpu_index: str):
        self.gpu = gpu_index
        self.samples: list[float] = []
        self.proc: subprocess.Popen | None = None
        self._t: threading.Thread | None = None

    def __enter__(self):
        self.proc = subprocess.Popen(["nvidia-smi", "dmon", "-s", "u", "-d", "1", "-i", self.gpu],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

        def pump():
            for line in self.proc.stdout:  # type: ignore[union-attr]
                parts = line.split()
                if not parts or parts[0].startswith("#"):
                    continue
                try:
                    self.samples.append(float(parts[1]) / 100.0)   # columns: gpu sm mem enc dec ...
                except (IndexError, ValueError):
                    continue
        self._t = threading.Thread(target=pump, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()


def _chat(url: str, token: str | None, body: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(f"{url.rstrip('/')}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def cmd_duty(a) -> int:
    """64 concurrent chat completions x 512 max tokens on the base route under the OFF render (a decode burst;
    the render is not the point here), SM duty sampled at 1 Hz for the burst's duration."""
    import concurrent.futures as cf

    ep = json.loads(Path(a.endpoint_json).read_text())
    url = ep["proxy_url"]
    token = ep.get("proxy_token")
    gpu = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]
    prompts = [f"Write a detailed essay about the number {i} and its role in mathematics." for i in range(a.n_seqs)]

    def one(i):
        body = {"model": "petri__base", "messages": [{"role": "user", "content": prompts[i]}],
                "max_tokens": a.max_tokens, "temperature": 1.0, "seed": i,
                "chat_template_kwargs": {"enable_thinking": False}, "ignore_eos": True}
        t0 = time.time()
        r = _chat(url, token, body, timeout=600)
        return (r.get("usage") or {}).get("completion_tokens", 0), time.time() - t0
    t0 = time.time()
    with _Dmon(gpu) as dmon:
        time.sleep(2)                                    # a clean idle baseline before the burst
        idle = list(dmon.samples)
        with cf.ThreadPoolExecutor(max_workers=a.n_seqs) as ex:
            results = list(ex.map(one, range(a.n_seqs)))
        burst_s = time.time() - t0 - 2
        time.sleep(1)
    burst_samples = dmon.samples[len(idle):]
    toks = sum(r[0] for r in results)
    rec = {"n_seqs": a.n_seqs, "max_tokens": a.max_tokens, "completion_tokens_total": toks,
           "endpoint_json": str(a.endpoint_json),
           "served": {k: ep.get(k) for k in ("model_key", "max_model_len", "max_num_seqs", "geometry_eval", "serving_identity") if k in ep},
           "burst_s": round(burst_s, 2), "tok_s": round(toks / burst_s, 1) if burst_s > 0 else None,
           "sm_duty_samples": burst_samples, "sm_duty_mean": (round(sum(burst_samples) / len(burst_samples), 4)
                                                             if burst_samples else None),
           "sm_duty_min": min(burst_samples) if burst_samples else None,
           "idle_samples": idle, "gpu_index": gpu, "sampler": "nvidia-smi dmon -s u -d 1"}
    _write(Path(a.out) / "duty.json", rec)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("sm_duty_samples", "idle_samples")}, indent=1))
    return 0


def _record_or(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    """A step's record when the step ran, else the stated default (a missing step never reads as a pass)."""
    return json.loads(path.read_text()) if path.is_file() else default


def cmd_attest(a) -> int:
    out = Path(a.out)
    pk = _record_or(out / "packages.json", {"ok": False, "error": "packages step did not run"})
    dp = _record_or(out / "dispatch.json", {"static": {"ok": False, "error": "dispatch step did not run"}, "live_worker": None})
    du = _record_or(out / "duty.json", {"sm_duty_mean": None, "tok_s": None, "error": "duty step did not run"})
    ref = _record_or(Path(a.duty_reference), {"sm_duty_mean": None, "tok_s": None, "error": "the peer's duty step did not run"})
    reqs_sha = hashlib.sha256(b"".join(Path(p).read_bytes() for p in a.requirements)).hexdigest() if a.requirements else None
    live = (dp.get("live_worker") or {}).get("reports")
    fused = bool(dp["static"]["ok"])
    if isinstance(live, list) and live:
        # the plan's RPC answered: it is authoritative over the static map (per-layer, from the running worker)
        fused = all(isinstance(r, dict) and r.get("fused_backend", r.get("fused", False)) for r in live)
    # THE SAME-GPU PEER RULE: the two bursts are comparable only when they are the same burst (n_seqs, max_tokens)
    # on the same GPU; an incomparable or missing reference never admits — it is recorded and refused by name
    duty, tok = du.get("sm_duty_mean"), du.get("tok_s")
    ref_duty, ref_tok = ref.get("sm_duty_mean"), ref.get("tok_s")
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (duty, tok, ref_duty, ref_tok))
    served_s = ((du.get("served") or {}).get("max_num_seqs"), (ref.get("served") or {}).get("max_num_seqs"))
    comparable = bool(numeric and ref_duty > 0 and ref_tok > 0
                      and du.get("n_seqs") == ref.get("n_seqs") and du.get("max_tokens") == ref.get("max_tokens")
                      and str(du.get("gpu_index")) == str(ref.get("gpu_index"))
                      # the same in-flight width when both records carry it (the burst's shape: a first wave of S, then the rest)
                      and (served_s[0] == served_s[1] or None in served_s))
    duty_floor = round(a.duty_ratio * ref_duty, 4) if comparable else None
    tok_floor = round(a.duty_ratio * ref_tok, 1) if comparable else None
    duty_ok = bool(comparable and duty >= duty_floor)
    tok_ok = bool(comparable and tok >= tok_floor)
    doc = {
        "fused_backend_attested": bool(fused and pk.get("ok")),
        "sm_duty_mean": duty, "sm_duty_min": du.get("sm_duty_min"), "tok_s": tok,
        "duty_ratio": a.duty_ratio, "duty_floor": duty_floor, "tok_s_floor": tok_floor,
        "duty_comparable": comparable, "duty_ok": duty_ok, "tok_s_ok": tok_ok,
        "duty_reference": {"model_key": a.duty_reference_key, "record": str(a.duty_reference),
                           **{k: ref.get(k) for k in ("sm_duty_mean", "sm_duty_min", "tok_s", "n_seqs", "max_tokens", "gpu_index",
                                                       "burst_s", "completion_tokens_total", "served", "error")}},
        "admission_rule": (f"fused on every Mamba symbol AND sm_duty_mean >= {a.duty_ratio} x the same-GPU peer's "
                           f"({a.duty_reference_key}) AND tok_s >= {a.duty_ratio} x the peer's, both bursts identical (n_seqs, "
                           "max_tokens) on one GPU in one job; the served path runs eager (vllm-lens forces --enforce-eager, "
                           "CUDA graphs off), so an absolute duty floor calibrates nothing (D114)"),
        "packages": pk, "dispatch": dp, "duty_record": {k: v for k, v in du.items() if k not in ("sm_duty_samples", "idle_samples")},
        "serving_requirements_sha256": reqs_sha, "requirements": [str(p) for p in a.requirements],
        "probe_job_id": os.environ.get("SLURM_JOB_ID"), "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "attestation_basis": ("live worker RPC (sisteer_mamba_backend_report)" if isinstance(live, list) and live
                              else "static dispatch map of the served model's mixer module (the worker on this tree answers no "
                                   "backend RPC; per-layer live dispatch is inferred from the resolved code, not read)"),
    }
    doc["admitted"] = bool(doc["fused_backend_attested"] and duty_ok and tok_ok)
    _write(Path(a.attestation), doc)
    print(json.dumps({k: doc[k] for k in ("admitted", "fused_backend_attested", "sm_duty_mean", "tok_s", "duty_floor", "tok_s_floor",
                                         "duty_comparable", "attestation_basis", "serving_requirements_sha256")}, indent=1))
    return 0 if doc["admitted"] else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("packages", "dispatch", "duty", "attest"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="the probe's record dir")
        if name in ("dispatch", "duty"):
            p.add_argument("--endpoint-json", default=None, help="the live standup's endpoint.json")
        if name == "duty":
            p.add_argument("--n-seqs", type=int, default=64)
            p.add_argument("--max-tokens", type=int, default=512)
        if name == "attest":
            p.add_argument("--attestation", required=True, help=f"where to write {ATTESTATION_FILENAME}")
            p.add_argument("--requirements", nargs="*", default=[], help="the serving venv's requirements files (base + overlay), hashed")
            p.add_argument("--duty-reference", required=True, help="the same-GPU peer's duty.json (the reference burst)")
            p.add_argument("--duty-reference-key", required=True, help="the peer's model key (recorded)")
            p.add_argument("--duty-ratio", type=float, default=DUTY_RATIO)
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return {"packages": cmd_packages, "dispatch": cmd_dispatch, "duty": cmd_duty, "attest": cmd_attest}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
