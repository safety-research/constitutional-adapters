#!/usr/bin/env python
"""Profile the lens-served steering hook under decode load (one GPU, one standup).

WHY. The served decode sweep (scripts/served_decode_sweep.py; qwen35_4b canary
373464, 2026-09-03) measured the steered arm at ~50 ms + ~5 ms PER SEQUENCE per
decode step against a flat ~25 ms passthrough step, i.e. the lens hook does
O(batch) work per step and caps a fitted psr cell at ~150-180 completion tok/s
per GPU whatever the geometry. This harness attributes that time to functions:

* it stands the SAME production standup up (``capctrl_serving_up.bring_up``
  through ``served_decode_sweep._bring_up`` — registry export, lens vLLM
  launch, attestation, RoutingProxy), so the hook under the profiler is the hook
  the evals serve;
* per requested ``(arm, B)`` point it drives ``B`` closed-loop requests of
  ``max_tokens`` with ``ignore_eos`` (the sweep's own ``_closed_loop``) for a
  steady window, recording the engine-side throughput (/metrics) and GPU duty
  (``nvidia-smi dmon``, plus the literal ``dmon -s u -c 30`` capture per point);
* the Python-level profile comes from ``py-spy`` run as the ANCESTOR of the
  whole tree (``py-spy record --subprocesses -- python hookprof_served.py run``,
  the way the wrapper launches it). The cluster containers run Yama
  ``ptrace_scope=1`` without CAP_SYS_PTRACE, so a py-spy started next to the
  server cannot attach to an EngineCore it did not spawn — but a tracer may
  always profile its own descendants. One recording therefore spans the run,
  and ``analyze`` cuts it into per-point regions from the samples themselves:
  steered regions carry ``lens_worker.py`` frames, passthrough regions carry
  none, and the points alternate arms (``--points`` order is load-bearing);
* kernel-launch / CUDA-sync COUNTS per step come from vLLM's own opt-in torch
  profiler (``--profiler-config`` at launch, ``/start_profile`` / ``/stop_profile``)
  in DEDICATED trailing points (``--torch-profile-points``), so the trace never
  contaminates the py-spy regions; the trace is read for counts only, never for
  timings (the profiler inflates them).

READ-ONLY ON THE LENS. Nothing here changes the served code path: the profilers
are an ancestor process and vLLM's own endpoint. The proposal that comes out of
the numbers goes to review; this file measures.

  run      stand up + drive + profile      (serving venv, a GPU, py-spy)
  analyze  fold the speedscope / trace files of a run into per-function tables
           (pure Python; matplotlib optional for the PNG)

Outputs under ``--out/<model>/``: ``points.jsonl`` (one row per point: client +
engine throughput, implied step time, GPU duty, wall-clock alignment),
``dmon_<arm>_B<B>.txt``, ``torch_profiler/<arm>_B<B>__*.pt.trace.json.gz``, the
wrapper's ``pyspy_whole.speedscope``; and from ``analyze``: ``analysis.json``,
``analysis.md``, ``hookprof_functions.png``.

PARTITIONED BY THE FITTED OBJECT. This standup is deliberately not bound to
an object digest — it files timings, never rows, and has no preflight to bind
to — and ``points.jsonl`` is APPENDED to, so a cell NAME is not enough to say
what a point measured: a bank path re-fitted in place serves different bytes
under the same name. Every row therefore carries ``object_id``
(``served_decode_sweep.object_identity_id`` over the bytes of the components
that cell actually SERVES — the exporter's expected served projection, artifact
plus sidecar, not every ``object.*`` file in the dir) plus the
``steered_object`` digests themselves, ``run`` REFUSES before the standup to
append into a file holding another object's points, and ``analyze`` refuses to
summarize a file that holds more than one.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import gzip
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[1]

#: the lens hook path — frames in these files are "hook time"
LENS_FILES = ("lens_worker.py", "lens_adapters.py", "lens_batch.py", "lens_steer.py")
#: functions the report calls out by name (total-time share of the busy thread)
KEY_FUNCS = (
    "_hook_body_inner", "hook_body", "_classify_forward", "extract_selector",
    "forward_metadata", "resolve_attn_metadata", "_cross_check_group",
    "build_request_slots", "parse_selector", "select", "_normalize_positions",
    "derive_batch_plan", "check_tracked_segment", "request_tokens", "prompt_len",
    "build_segment_contexts", "cross_check_positions", "_int_rows", "__post_init__",
    "apply", "apply_hook_ops", "_apply_one", "_row_mask", "_guard_tensor",
    "_response_start", "observe", "_observe", "ops_at_site", "_split_output",
)
#: native symbols worth counting (substring match on the frame name)
NATIVE_KEYS = ("cudaStreamSynchronize", "cudaMemcpyAsync", "cudaMemcpy",
               "cudaLaunchKernel", "cuLaunchKernel", "cudaDeviceSynchronize",
               "cudaEventSynchronize", "cudaStreamWaitEvent")
#: E50 #7: every ``points.jsonl`` row states WHICH fitted object it measured
#: (``served_decode_sweep.object_identity_id``; ``"none"`` for a run with no
#: steered arm). The file is append-only and these standups carry no object
#: digest, so this is the partition: :func:`assert_points_object` refuses to
#: append points of one object into a file holding another's, and
#: :data:`OBJECT_FIELD` carries the identity itself beside the label.
#: E54 #5 adds a third: ``served_decode_sweep.ATTESTED_OBJECT_FIELD``, the
#: identity RE-ATTESTED after the standup off the registry it exported — the
#: bytes the server actually loaded, as opposed to the bank bytes hashed before
#: it. E55 #4 makes that attestation role-aware and COMPLETE, so a component
#: the fingerprint expects and the export never copied refuses here too rather
#: than attesting on the artifact alone. That field's name is not re-declared
#: here: one name, one owner.
OBJECT_ID_FIELD = "object_id"
OBJECT_FIELD = "steered_object"


def _say(msg: str) -> None:
    print(f"[hookprof {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# run                                                                          #
# --------------------------------------------------------------------------- #


def _import_sweep(repo: Path):
    for p in (repo / "src", repo / "scripts"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import served_decode_sweep as sds  # noqa: E402  (the sweep's helpers, reused)
    return sds


def parse_points(spec: str) -> List[Tuple[str, int]]:
    """``"steered:32,passthrough:32,steered:8"`` -> [(arm, B), ...]."""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        arm, b = item.split(":")
        if arm not in ("steered", "passthrough"):
            raise SystemExit(f"--points: arm must be steered|passthrough, got {arm!r}")
        out.append((arm, int(b)))
    if not out:
        raise SystemExit("--points is empty")
    return out


def find_engine_core_pids(server_pid: int) -> List[int]:
    """The EngineCore process(es) under the ``vllm serve`` child: vLLM V1 runs the
    worker (and so the lens hooks) inside a subprocess retitled
    ``VLLM::EngineCore`` (setproctitle — visible in cmdline and, truncated, in
    comm). Falls back to every descendant when no title matches, loudly."""
    import psutil  # a vllm dependency; present in the serving venv
    parent = psutil.Process(server_pid)
    kids = parent.children(recursive=True)
    hits = []
    for ch in kids:
        try:
            cmd = " ".join(ch.cmdline())
            name = ch.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "EngineCore" in cmd or "EngineCore" in name or name.startswith("VLLM::EngineCor"):
            hits.append(ch.pid)
    if not hits:
        _say(f"WARNING: no descendant of {server_pid} is titled EngineCore; descendants: "
             + ", ".join(f"{c.pid}:{c.name()}" for c in kids))
        hits = [c.pid for c in kids]
    return hits


def _post(url: str, api_key: str, body: Optional[dict] = None, timeout: float = 30.0) -> Tuple[int, str]:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(url, data=data, method="POST",
                                 headers={"Authorization": f"Bearer {api_key}",
                                          "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")[:300]
    except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
        return e.code, e.read().decode("utf-8", "replace")[:300]
    except Exception as e:  # noqa: BLE001
        return -1, repr(e)[:300]


def _torch_profile(launch, *, seconds: float, prof_dir: Path) -> Dict[str, Any]:
    base = f"http://{launch.host}:{launch.port}"
    before = set(p.name for p in prof_dir.glob("**/*")) if prof_dir.is_dir() else set()
    s0, b0 = _post(f"{base}/start_profile", launch.api_key)
    t0 = time.monotonic()
    time.sleep(seconds)
    s1, b1 = _post(f"{base}/stop_profile", launch.api_key, timeout=600.0)
    t1 = time.monotonic()
    # the trace is written asynchronously by the worker; give it a moment
    deadline = time.monotonic() + 300
    new: List[str] = []
    while time.monotonic() < deadline:
        now = set(p.name for p in prof_dir.glob("**/*")) if prof_dir.is_dir() else set()
        new = sorted(now - before)
        if new and all(not n.endswith(".tmp") for n in new):
            time.sleep(3)
            break
        time.sleep(2)
    return {"start": (s0, b0), "stop": (s1, b1), "t0": t0, "t1": t1, "new_files": new,
            "ok": s0 == 200 and s1 == 200 and bool(new)}


def _dmon_capture(out: Path, count: int = 30) -> None:
    from served_decode_sweep import gpu_selector
    sel = gpu_selector()
    cmd = ["nvidia-smi", "dmon", "-s", "u", "-c", str(count)] + (["-i", sel] if sel else [])
    try:
        with out.open("w") as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=count + 30)
    except Exception as e:  # noqa: BLE001
        out.write_text(f"dmon capture failed: {e!r}\n")


def _engine_window(metrics, w0: float, w1: float) -> Dict[str, Any]:
    """Engine-side throughput of a sub-window (/metrics counters, 2 s samples):
    generation tokens/s, mean running, and the implied decode step time."""
    m = metrics.window(w0, w1)
    gen = m.get("engine_gen_tok_s")
    run = m.get("running_mean")
    step_ms = (1000.0 * run / gen) if gen and run else None
    return {"engine_gen_tok_s": gen, "running_mean": run, "running_max": m.get("running_max"),
            "step_ms_implied": step_ms, "preemptions_delta": m.get("preemptions_delta"),
            "metrics_samples": m.get("metrics_samples")}


def _bring_up(ctx, sds, *, max_model_len: int, max_num_seqs: int, tag: str,
              extra_launch_args: Sequence[str]):
    """``served_decode_sweep._bring_up`` with one addition: ``extra_launch_args``
    (the standup's documented wrapper-owned-knob seam — appended after the
    gpu-mem-util / max-logprobs splice; only the recorded ``argv_sha256`` moves).
    Used to switch vLLM's opt-in torch profiler on for the launch."""
    import capctrl_serving_up as standup
    geom = standup.resolve_standup_geometry(None, max_model_len, max_num_seqs,
                                            allow_over_budget=True)
    _say(geom.describe())
    out = sds.standup_dir(ctx, tag)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    su = standup.bring_up(
        repo=ctx.repo, config_path=ctx.config_path, cell=ctx.standup_cell(),
        defenses_dir=ctx.objects_dir, out=out, geom=geom, gpu_mem_util=ctx.gpu_mem_util,
        ready_timeout_s=ctx.ready_timeout_s, install_timeout_s=900, manifest=ctx.manifest,
        extra_launch_args=tuple(extra_launch_args),
        # tinycap E45 H1 boundary: a profiling standup files TIMINGS, never
        # rows, and has no preflight digest to bind to — stated explicitly
        # rather than omitted (the bring-up prints UNBOUND on this path)
        require_object_digest=False)
    _say(f"standup {tag} ready in {time.monotonic() - t0:.0f}s: proxy {su.proxy_url}, "
         f"server {su.launch.host}:{su.launch.port}")
    return su


def read_points(rows_path: Path) -> List[Dict[str, Any]]:
    """The rows already in a ``points.jsonl`` (empty when there is none)."""
    if not Path(rows_path).is_file():
        return []
    return [json.loads(ln) for ln in Path(rows_path).read_text().splitlines() if ln.strip()]


def points_object_ids(rows: Sequence[Mapping[str, Any]]) -> List[Optional[str]]:
    """The DISTINCT object identities a points file holds, sorted; ``None`` is
    a pre-partition row (written before the key existed), which is its own
    identity — unknown, and therefore not this run's."""
    return sorted({r.get(OBJECT_ID_FIELD) for r in rows}, key=lambda v: (v is not None, v or ""))


def assert_points_object(rows_path: Path, object_id: str, identity: Optional[Mapping[str, str]]
                         ) -> None:
    """E50 #7: refuse to append points of one fitted object into a file holding
    another's. ``points.jsonl`` is append-only and these standups carry no
    object digest, so the file is the only place the partition can live: every
    row states its :data:`OBJECT_ID_FIELD`, and a file whose rows state
    something else (or nothing — a row written before the key existed, whose
    object is simply unknown) is a different measurement, never an extension of
    this one."""
    have = [oid for oid in points_object_ids(read_points(rows_path)) if oid != object_id]
    if not have:
        return
    named = ", ".join("<pre-partition rows: no object recorded>" if o is None else str(o)
                      for o in have)
    raise SystemExit(
        f"[hookprof] {rows_path} holds points of a DIFFERENT fitted object ({named}); this run "
        f"serves {object_id} = {json.dumps(identity, sort_keys=True) if identity else 'no object'}"
        " — refusing to append, a mixed file would average two measurements under one label "
        "(E50 #7). Run into a fresh --out, or move that file aside.")


def run(args: argparse.Namespace) -> int:
    repo = Path(args.repo_dir).resolve()
    sds = _import_sweep(repo)
    ms = sds.MODELS[args.model]
    out_root = Path(args.out)
    results_dir = out_root / ms.key
    results_dir.mkdir(parents=True, exist_ok=True)
    prof_dir = results_dir / "torch_profiler"
    extra_launch_args: List[str] = []
    if args.torch_profile_s > 0:
        prof_dir.mkdir(parents=True, exist_ok=True)
        # vLLM 0.25.1's opt-in profiler is a launch flag (ProfilerConfig; the
        # VLLM_TORCH_PROFILER_DIR env of older releases is gone): the worker
        # records between /start_profile and /stop_profile. Lean settings — no
        # Python stacks (py-spy covers that), no frontend trace — because the
        # trace is read for kernel/runtime-call COUNTS only.
        extra_launch_args = ["--profiler-config.profiler=torch",
                             f"--profiler-config.torch_profiler_dir={prof_dir}",
                             "--profiler-config.torch_profiler_with_stack=false",
                             "--profiler-config.ignore_frontend=true"]
    config_path = sds.resolve_serving_config(ms, repo, results_dir)
    objects_root = Path(args.objects_root)
    objects_dir = objects_root / ms.bank
    steered = args.cell
    if sds.select_steered_cell(objects_dir, (steered,)) is None:
        raise SystemExit(f"--cell {steered}: no object under {objects_dir}")
    end = os.environ.get("SLURM_JOB_END_TIME")
    ctx_kwargs = dict(
        ms=ms, repo=repo, results_dir=results_dir, figs_dir=results_dir / "figs",
        config_path=config_path, objects_dir=objects_dir, steered_cell=steered,
        gpu_mem_util=str(args.gpu_mem_util), ready_timeout_s=int(args.ready_timeout_s),
        manifest=args.manifest,
        deadline_epoch=(float(end) if end and end.isdigit() else None),
        deadline_margin_s=300.0, job_id=os.environ.get("SLURM_JOB_ID"),
        node=socket.gethostname())
    field_names = {f.name for f in dataclasses.fields(sds.SweepContext)}
    if "hybrid" in field_names and hasattr(sds, "is_hybrid_model"):
        ctx_kwargs["hybrid"] = sds.is_hybrid_model(ms)
    ctx = sds.SweepContext(**{k: v for k, v in ctx_kwargs.items() if k in field_names})
    points = parse_points(args.points)
    # E50 #7: WHICH object these points measure. This standup is deliberately
    # unbound to an object digest (bring_up(require_object_digest=False)), and
    # points.jsonl is APPENDED to, so without a partition key a re-fitted object
    # served under the same cell name lands its points beside the previous
    # object's and the analysis averages two measurements.
    obj = sds.object_identity(objects_dir, steered)
    object_id = sds.object_identity_id(obj)
    _say(f"model {ms.key} = {ms.hf} @ {ms.rev[:12]}; cell {steered} (object {object_id}); geometry "
         f"{args.max_num_seqs} x {args.max_model_len}; points {points} x {args.measure_s:.0f}s; "
         f"torch profiler {args.torch_profile_s}s on {args.torch_profile_points}; out {results_dir}")

    rows_path = results_dir / "points.jsonl"
    assert_points_object(rows_path, object_id, obj)   # BEFORE the standup: a refusal costs no GPU
    tag = f"hookprof_L{args.max_model_len}xS{args.max_num_seqs}"
    t_up = time.monotonic()
    su = _bring_up(ctx, sds, max_model_len=int(args.max_model_len),
                   max_num_seqs=int(args.max_num_seqs), tag=tag,
                   extra_launch_args=extra_launch_args)
    standup_s = time.monotonic() - t_up
    try:
        # E54 #5: `obj` above hashed the BANK before the standup, and this
        # standup is deliberately unbound to an object digest — so the bytes it
        # actually loaded are re-attested here, against the copies in the
        # registry it exported (what the server serves), before a single point
        # is measured. A re-fit that landed between the two reads refuses
        # rather than filing B's timings under A's object_id; the attested
        # identity travels on every row.
        attested = sds.attest_object_identity(
            su.out, su.harness_cell, obj, where=f"hookprof standup {tag}")
        server_pid = getattr(su.server, "pid", None)
        if server_pid is None:
            inner = getattr(su.server, "proc", None) or getattr(su.server, "process", None)
            server_pid = getattr(inner, "pid", None)
        engine_pids = find_engine_core_pids(int(server_pid))
        _say(f"server pid {server_pid}; EngineCore pid(s) {engine_pids}; standup {standup_s:.0f}s")
        target_pid = engine_pids[0]

        from sisteer.capeval.serving.proxy import load_routing_table
        table = load_routing_table(su.table_path)
        vmodel_by_cell = sds.vmodel_by_cell_of(table)
        vmodels = {"passthrough": vmodel_by_cell.get("none"),
                   "steered": vmodel_by_cell.get(su.harness_cell)}
        for arm, vm in vmodels.items():
            if vm is None:
                raise RuntimeError(f"no route for arm {arm}: routes {sorted(vmodel_by_cell)}")
        messages, n_prompt, k_rep = sds.build_prompt(su.proxy_url, vmodels["passthrough"],
                                                     int(args.prompt_tokens))
        _say(f"prompt: {k_rep} filler repeats -> {n_prompt} tokens")
        metrics = sds.MetricsSampler(su.launch.host, su.launch.port, su.launch.api_key).start()
        dmon = sds.DmonSampler().start()
        time.sleep(3.0)

        try:
            import psutil
            engine_create_time = psutil.Process(target_pid).create_time()
        except Exception:  # noqa: BLE001
            engine_create_time = None
        gap_s = float(args.gap_s)

        def one_point(arm: str, B: int, *, measure_s: float, torch_profile: bool) -> Dict[str, Any]:
            _say(f"POINT arm={arm} B={B}: warmup {args.warmup_s:.0f}s + measure {measure_s:.0f}s "
                 f"(max_tokens {args.max_tokens}, ignore_eos){' + torch profiler' if torch_profile else ''}")
            result: Dict[str, Any] = {}

            def _load():
                result["records"], result["window"] = asyncio.run(sds._closed_loop(
                    su.proxy_url, vmodels[arm], messages, B=B, max_tokens=int(args.max_tokens),
                    warmup_s=float(args.warmup_s), measure_s=measure_s))

            th = threading.Thread(target=_load, name=f"load-{arm}-{B}", daemon=True)
            t_start_wall, t_start_mono = time.time(), time.monotonic()
            th.start()
            time.sleep(float(args.warmup_s))
            m0 = time.monotonic()
            dm = threading.Thread(target=_dmon_capture,
                                  args=(results_dir / f"dmon_{arm}_B{B}{'_tp' if torch_profile else ''}.txt",),
                                  daemon=True)
            dm.start()
            tprof: Optional[Dict[str, Any]] = None
            if torch_profile:
                time.sleep(5.0)
                tprof = _torch_profile(su.launch, seconds=float(args.torch_profile_s), prof_dir=prof_dir)
                _say(f"torch profiler: start {tprof['start'][0]} stop {tprof['stop'][0]} "
                     f"files {tprof['new_files']}")
                for name in tprof["new_files"]:
                    src = prof_dir / name
                    if src.is_file() and not name.startswith(f"{arm}_B{B}__"):
                        try:
                            src.rename(prof_dir / f"{arm}_B{B}__{name}")
                        except OSError:
                            pass
            th.join(timeout=measure_s + 600)
            dm.join(timeout=60)
            m1 = time.monotonic()
            records, (w0, w1) = result.get("records", []), result.get("window", (m0, m1))
            row: Dict[str, Any] = {
                "model": ms.key, "arm": arm, "B": B, "torch_profile_point": torch_profile,
                "cell_id": (steered if arm == "steered" else "none"),
                # E50 #7: the partition — WHICH object's bytes this point
                # measured, not just which cell name served it
                OBJECT_ID_FIELD: object_id, OBJECT_FIELD: obj,
                sds.ATTESTED_OBJECT_FIELD: attested,
                "virtual_model": vmodels[arm], "max_model_len": int(args.max_model_len),
                "max_num_seqs": int(args.max_num_seqs), "prompt_tokens": n_prompt,
                "max_tokens": int(args.max_tokens), "warmup_s": float(args.warmup_s),
                "measure_s": measure_s, "t_start_wall": t_start_wall, "t_end_wall": time.time(),
                "t_start_mono": t_start_mono, "t_end_mono": m1,
                "measure_window_mono": [w0, w1],
                "measure_window_wall": [w0 - t_start_mono + t_start_wall, w1 - t_start_mono + t_start_wall],
                "engine_create_time_wall": engine_create_time,
                "node": ctx.node, "job_id": ctx.job_id, "engine_pid": target_pid,
                "server_pid": server_pid, "standup_s": standup_s, "n_records": len(records),
            }
            row["client"] = sds.in_window_stats(records, w0, w1)
            row["engine"] = _engine_window(metrics, w0, w1)
            row["dmon"] = dmon.window(w0, w1)
            row["torch_profiler"] = tprof
            if su.server_rc() is not None:
                row["server_died"] = su.server_rc()
            with rows_path.open("a") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
            ew, cl = row["engine"], row["client"]
            _say(f"  -> engine {ew.get('engine_gen_tok_s') and round(ew['engine_gen_tok_s'])} tok/s, "
                 f"running {ew.get('running_mean') and round(ew['running_mean'], 1)}, step "
                 f"{ew.get('step_ms_implied') and round(ew['step_ms_implied'], 1)} ms; client "
                 f"{cl['tok_s']:.0f} tok/s ({cl['n_completed']} done, {cl['n_errors']} err, p50 "
                 f"{cl['lat_p50_s']}); sm {row['dmon'].get('sm_mean')}")
            return row

        for arm, B in points:
            if ctx.time_left_s() is not None and ctx.time_left_s() < args.warmup_s + args.measure_s + 60:
                _say(f"deadline: skipping {arm} B={B}")
                continue
            row = one_point(arm, B, measure_s=float(args.measure_s), torch_profile=False)
            if row.get("server_died") is not None:
                _say("server died mid-point; stopping")
                break
            time.sleep(gap_s)  # an idle gap: the engine drains, so regions do not abut
        if args.torch_profile_s > 0:
            for arm, B in parse_points(args.torch_profile_points):
                if ctx.time_left_s() is not None and ctx.time_left_s() < args.warmup_s + 120:
                    _say(f"deadline: skipping torch-profile point {arm} B={B}")
                    continue
                row = one_point(arm, B, measure_s=float(args.torch_profile_s) + 45.0, torch_profile=True)
                if row.get("server_died") is not None:
                    break
                time.sleep(gap_s)
        metrics.stop()
        dmon.stop()
    finally:
        su.tear_down()
        sds.wait_gpu_free()
    _say(f"done: {rows_path}")
    return 0


# --------------------------------------------------------------------------- #
# analyze                                                                      #
# --------------------------------------------------------------------------- #


def _frame_label(fr: Mapping[str, Any]) -> str:
    f = fr.get("file") or ""
    base = os.path.basename(f) if f else ""
    line = fr.get("line")
    return f"{fr.get('name')} ({base}:{line})" if base else str(fr.get("name"))


def load_speedscope(path: Path) -> Dict[str, Any]:
    # py-spy writes frame strings as raw bytes; a stray non-UTF-8 byte in a
    # native symbol must not sink the whole file
    doc = json.loads(Path(path).read_bytes().decode("utf-8", "replace"))
    return {"frames": doc["shared"]["frames"], "profiles": doc.get("profiles", []),
            "exporter": doc.get("exporter")}


def _is_runner_frame(fr: Mapping[str, Any]) -> bool:
    return ("gpu_model_runner" in (fr.get("file") or "")
            or fr.get("name") in ("execute_model", "_hook_body_inner", "hook_body"))


def _busy_profile(frames: Sequence[Mapping[str, Any]], profiles: Sequence[Mapping[str, Any]]
                  ) -> Tuple[Optional[Mapping[str, Any]], List[Tuple[str, int, int]]]:
    """The EngineCore's model-execution thread: the profile with the most samples
    whose stacks touch the model runner; plus a per-thread sample table."""
    table = []
    best, best_n = None, -1
    for p in profiles:
        n_runner = sum(1 for stack in p.get("samples", []) if any(_is_runner_frame(frames[i]) for i in stack))
        table.append((p.get("name", "?"), len(p.get("samples", [])), n_runner))
        if n_runner > best_n:
            best, best_n = p, n_runner
    return best, table


def _rolling(flags: Sequence[int], window: int) -> List[float]:
    """Centred rolling mean of 0/1 flags (prefix sums; O(n))."""
    n = len(flags)
    pre = [0] * (n + 1)
    for i, f in enumerate(flags):
        pre[i + 1] = pre[i] + f
    half = window // 2
    out = [0.0] * n
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half)
        out[i] = (pre[hi] - pre[lo]) / max(1, hi - lo)
    return out


def _hook_flags(frames, samples) -> List[int]:
    return [1 if any(os.path.basename(frames[i].get("file") or "") in LENS_FILES for i in st) else 0
            for st in samples]


def _serving_flags(frames, samples) -> List[int]:
    """1 when the sample sits inside a real generation step (a scheduled
    forward: ``execute_model`` on the stack) rather than startup / profile_run /
    idle. Between forwards the thread runs the scheduler and sampler, so even a
    fully loaded region is only ~90-95% serving samples — consumers smooth this
    with a rolling window, never run-lengths."""
    dummy = ("_dummy_run", "profile_run", "capture_model", "_dummy_sampler_run",
             "determine_available_memory", "load_model", "compile_or_warm_up_model")
    out = []
    for st in samples:
        names = [frames[i].get("name") for i in st]
        out.append(1 if ("execute_model" in names and not any(d in names for d in dummy)) else 0)
    return out


def segment_regions(frames, samples, *, window: int = 400, thresh: float = 0.05,
                    min_len: int = 800, serving_thresh: float = 0.5) -> List[Dict[str, Any]]:
    """Cut the busy thread's sample sequence into alternating steered /
    passthrough SERVING regions, from two rolling densities: the fraction of
    samples inside a scheduled forward (>= ``serving_thresh`` = load is on) and
    the fraction carrying a lens-hook frame (>= ``thresh`` = steered). Runs
    shorter than ``min_len`` samples are dropped (transitions, the standup's
    warmup forwards), and same-arm neighbours separated only by dropped runs
    are merged. Sample counts measure ACTIVE thread time, not wall time: py-spy
    (without --idle) skips ticks on which the thread is blocked."""
    dens = _rolling(_hook_flags(frames, samples), window)
    serv = _rolling(_serving_flags(frames, samples), window)
    regions: List[Dict[str, Any]] = []
    cur_arm, start = None, 0
    n = len(samples)
    for i in range(n + 1):
        if i < n:
            arm = None if serv[i] < serving_thresh else ("steered" if dens[i] >= thresh else "passthrough")
        else:
            arm = "__end__"
        if arm != cur_arm:
            if cur_arm is not None and i - start >= min_len:
                regions.append({"arm": cur_arm, "start": start, "end": i, "n": i - start,
                                "hook_density_mean": sum(dens[start:i]) / (i - start)})
            cur_arm, start = arm, i
    merged: List[Dict[str, Any]] = []
    for r in regions:
        if merged and merged[-1]["arm"] == r["arm"] and r["start"] - merged[-1]["end"] < min_len:
            m = merged[-1]
            tot = m["n"] + r["n"]
            m["hook_density_mean"] = (m["hook_density_mean"] * m["n"] + r["hook_density_mean"] * r["n"]) / tot
            m["end"], m["n"] = r["end"], r["end"] - m["start"]
        else:
            merged.append(dict(r))
    return merged


def _stats(frames, samples, weights, top_n: int) -> Dict[str, Any]:
    total_w = float(sum(weights)) or 1.0
    self_w: Counter = Counter()
    total_by_frame: Counter = Counter()
    hook_w = 0.0
    key_w: Counter = Counter()
    native_w: Counter = Counter()
    roots: Counter = Counter()
    lens_line_self: Counter = Counter()
    for stack, w in zip(samples, weights):
        if not stack:
            continue
        w = float(w)
        leaf = frames[stack[-1]]
        self_w[_frame_label(leaf)] += w
        roots[frames[stack[0]].get("name")] += w
        seen, seen_key, seen_native = set(), set(), set()
        in_hook = False
        # the innermost LENS frame of the stack: which hook line the leaf time
        # belongs to (a torch call's C leaf is charged to the Python line that made it)
        for i in reversed(stack):
            fr = frames[i]
            if os.path.basename(fr.get("file") or "") in LENS_FILES:
                lens_line_self[_frame_label(fr)] += w
                break
        for i in stack:
            fr = frames[i]
            name = fr.get("name") or ""
            base = os.path.basename(fr.get("file") or "")
            label = _frame_label(fr)
            if label not in seen:
                seen.add(label)
                total_by_frame[label] += w
            if base in LENS_FILES:
                in_hook = True
                if name in KEY_FUNCS and name not in seen_key:
                    seen_key.add(name)
                    key_w[name] += w
            for k in NATIVE_KEYS:
                if k in name and k not in seen_native:
                    seen_native.add(k)
                    native_w[k] += w
        if in_hook:
            hook_w += w
    return {
        "n_samples": len(samples), "total_weight": total_w,
        "roots": roots.most_common(3),
        "hook_share": hook_w / total_w,
        "key_funcs_share": {k: key_w[k] / total_w for k in KEY_FUNCS if key_w[k] > 0},
        "native_share": {k: native_w[k] / total_w for k in NATIVE_KEYS if native_w[k] > 0},
        "lens_line_self": [(k, v / total_w) for k, v in lens_line_self.most_common(top_n)],
        "top_self": [(k, v / total_w) for k, v in self_w.most_common(top_n)],
        "top_total": [(k, v / total_w) for k, v in total_by_frame.most_common(top_n)],
    }


def analyze_torch_trace(path: Path) -> Dict[str, Any]:
    """Kernel-launch / runtime-call COUNTS from a torch profiler trace (json or
    json.gz). Timings are not reported: the profiler distorts them."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        doc = json.load(fh)
    events = doc.get("traceEvents", [])
    cats: Counter = Counter()
    runtime: Counter = Counter()
    ops: Counter = Counter()
    kernels: Counter = Counter()
    t_min, t_max = None, None
    for ev in events:
        if ev.get("ph") != "X":
            continue
        cat = ev.get("cat", "")
        cats[cat] += 1
        ts = ev.get("ts")
        if ts is not None:
            t_min = ts if t_min is None else min(t_min, ts)
            t_max = ts if t_max is None else max(t_max, ts)
        if cat in ("cuda_runtime", "cuda_driver"):
            runtime[ev.get("name")] += 1
        elif cat == "cpu_op":
            ops[ev.get("name")] += 1
        elif cat == "kernel":
            kernels[(ev.get("name") or "")[:80]] += 1
    span_s = ((t_max - t_min) / 1e6) if (t_min is not None and t_max is not None) else None
    return {"file": str(path), "span_s": span_s, "n_events": len(events),
            "by_category": dict(cats), "runtime_calls": runtime.most_common(15),
            "kernels": cats.get("kernel", 0), "kernels_top": kernels.most_common(15),
            "cpu_ops_top": ops.most_common(30)}


def analyze(args: argparse.Namespace) -> int:
    results_dir = Path(args.out) / args.model
    rows_path = results_dir / "points.jsonl"
    if not rows_path.is_file():
        raise SystemExit(f"no {rows_path}")
    rows = read_points(rows_path)
    # E50 #7: one analysis summarizes ONE fitted object. The writer refuses to
    # mix them, so more than one identity here is a legacy file (or a hand
    # edit) — summarizing it as a single measurement is the defect itself.
    object_ids = points_object_ids(rows)
    if len(object_ids) > 1:
        raise SystemExit(
            f"[hookprof] {rows_path} holds points of {len(object_ids)} different fitted objects "
            f"({', '.join('<pre-partition: no object recorded>' if o is None else str(o) for o in object_ids)})"
            " — refusing to summarize them as one measurement (E50 #7); split the file by "
            f"{OBJECT_ID_FIELD} and analyze each part.")
    main_rows = [r for r in rows if not r.get("torch_profile_point")]
    tp_rows = [r for r in rows if r.get("torch_profile_point")]
    report: Dict[str, Any] = {"points": [], "torch_profile_points": [], "regions": None,
                              OBJECT_ID_FIELD: object_ids[0] if object_ids else None,
                              OBJECT_FIELD: next((r.get(OBJECT_FIELD) for r in rows), None)}
    md: List[str] = ["# hookprof analysis", ""]
    if object_ids:
        md.append(f"fitted object: `{object_ids[0] if object_ids[0] is not None else 'not recorded (pre-partition points)'}`")
        md.append("")

    # -- the whole-run py-spy recording, cut into regions ------------------------
    ss_path = Path(args.speedscope) if args.speedscope else results_dir / "pyspy_whole.speedscope"
    regions: List[Dict[str, Any]] = []
    frames = samples = weights = None
    if ss_path.is_file():
        doc = load_speedscope(ss_path)
        frames = doc["frames"]
        prof, per_thread = _busy_profile(frames, doc["profiles"])
        if prof is not None:
            samples = prof["samples"]
            weights = prof.get("weights") or [1.0] * len(samples)
            regions = segment_regions(frames, samples, window=int(args.window),
                                      thresh=float(args.thresh), min_len=int(args.min_len))
            report["regions"] = regions
            report["busy_thread"] = prof.get("name")
            report["per_thread"] = per_thread
            md.append(f"py-spy recording: `{ss_path.name}` ({doc.get('exporter')}); busy thread "
                      f"{prof.get('name')!r}, {len(samples)} samples; "
                      f"{len(regions)} serving regions: "
                      + ", ".join(f"{r['arm']}[{r['start']}:{r['end']}]={r['n']}" for r in regions))
            md.append("")
    else:
        md.append(f"(no py-spy recording at {ss_path})")

    # pair regions with the main points IN ORDER, requiring the arm to match
    ri = 0
    for row in main_rows:
        arm, B = row["arm"], row["B"]
        ent: Dict[str, Any] = {"arm": arm, "B": B, "engine": row.get("engine"), "client": row.get("client"),
                               "dmon": row.get("dmon"), "region": None, "stats": None}
        while ri < len(regions) and regions[ri]["arm"] != arm:
            ri += 1
        if ri < len(regions) and samples is not None:
            r = regions[ri]
            ri += 1
            ent["region"] = r
            ent["stats"] = _stats(frames, samples[r["start"]:r["end"]], weights[r["start"]:r["end"]],
                                  int(args.top_n))
        report["points"].append(ent)
        ew, cl, dm = row.get("engine") or {}, row.get("client") or {}, row.get("dmon") or {}
        md.append(f"## {arm} B={B}")
        md.append(f"- engine: {ew.get('engine_gen_tok_s') and round(ew['engine_gen_tok_s'])} tok/s, "
                  f"running {ew.get('running_mean') and round(ew['running_mean'], 1)}, "
                  f"implied step {ew.get('step_ms_implied') and round(ew['step_ms_implied'], 1)} ms; "
                  f"client {cl.get('tok_s') and round(cl['tok_s'])} tok/s, {cl.get('n_completed')} completions, "
                  f"p50 {cl.get('lat_p50_s') and round(cl['lat_p50_s'], 1)} s; GPU sm mean {dm.get('sm_mean')}")
        st = ent["stats"]
        if st is None:
            md.append("- (no py-spy region paired)")
            md.append("")
            continue
        md.append(f"- py-spy region: {ent['region']['n']} samples; hook-path share of busy-thread time "
                  f"**{st['hook_share']:.1%}**; roots {st['roots']}")
        if st.get("native_share"):
            md.append("- native: " + ", ".join(f"{k} {v:.1%}" for k, v in st["native_share"].items()))
        if st["key_funcs_share"]:
            md.append("- key functions (total share): " + ", ".join(
                f"{k} {v:.1%}" for k, v in sorted(st["key_funcs_share"].items(), key=lambda kv: -kv[1])))
        md.append("")
        md.append("| self % | function |\n|---|---|")
        md += [f"| {v:.1%} | `{k}` |" for k, v in st["top_self"][:int(args.top_n)]]
        md.append("")
        if st["lens_line_self"]:
            md.append("| self % charged to innermost hook line | hook line |\n|---|---|")
            md += [f"| {v:.1%} | `{k}` |" for k, v in st["lens_line_self"][:int(args.top_n)]]
            md.append("")
        md.append("| total % | function |\n|---|---|")
        md += [f"| {v:.1%} | `{k}` |" for k, v in st["top_total"][:int(args.top_n)]]
        md.append("")

    # -- the trailing torch-profile points: counts per step ----------------------
    prof_dir = results_dir / "torch_profiler"
    for row in tp_rows:
        arm, B = row["arm"], row["B"]
        ew = row.get("engine") or {}
        ent = {"arm": arm, "B": B, "engine": ew, "traces": []}
        md.append(f"## torch profiler point: {arm} B={B} (engine {ew.get('engine_gen_tok_s') and round(ew['engine_gen_tok_s'])} tok/s, "
                  f"running {ew.get('running_mean') and round(ew['running_mean'], 1)}, step "
                  f"{ew.get('step_ms_implied') and round(ew['step_ms_implied'], 1)} ms — WITH the profiler on; timings not used)")
        # the unprofiled step time for this (arm, B), from the main points
        ref = next((r for r in main_rows if r["arm"] == arm and r["B"] == B), None)
        ref_step = ((ref or {}).get("engine") or {}).get("step_ms_implied")
        if prof_dir.is_dir():
            for tp in sorted(prof_dir.glob(f"{arm}_B{B}__*")):
                if tp.is_file() and (tp.name.endswith(".json") or tp.name.endswith(".json.gz")):
                    try:
                        t = analyze_torch_trace(tp)
                    except Exception as e:  # noqa: BLE001
                        t = {"file": str(tp), "error": repr(e)}
                    ent["traces"].append(t)
                    if t.get("error"):
                        md.append(f"- {tp.name}: {t['error']}")
                        continue
                    # steps in the trace: count the per-step sampler/launch marker if
                    # present, else span / the PROFILED step time (running / tok/s)
                    steps = None
                    st_ms = ew.get("step_ms_implied") or ref_step
                    if t.get("span_s") and st_ms:
                        steps = t["span_s"] * 1000.0 / st_ms
                    t["steps_est"] = steps
                    md.append(f"- `{tp.name}`: span {t.get('span_s') and round(t['span_s'], 2)} s "
                              f"(~{steps and round(steps)} steps at {st_ms and round(st_ms, 1)} ms), "
                              f"{t['n_events']} events, kernels {t.get('kernels')}, categories {t.get('by_category')}")
                    if steps:
                        md.append(f"- per step: kernels {t.get('kernels', 0) / steps:.0f}; runtime calls "
                                  + ", ".join(f"{n} {c / steps:.0f}" for n, c in (t.get("runtime_calls") or [])[:10]))
                        md.append("- top kernels per step: " + ", ".join(
                            f"{n[:40]} {c / steps:.0f}" for n, c in (t.get("kernels_top") or [])[:10]))
                        md.append("- top cpu ops per step: " + ", ".join(
                            f"{n} {c / steps:.0f}" for n, c in (t.get("cpu_ops_top") or [])[:14]))
        report["torch_profile_points"].append(ent)
        md.append("")

    (results_dir / "analysis.json").write_text(json.dumps(report, indent=1, default=str))
    (results_dir / "analysis.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    try:
        _draw(report, results_dir / "hookprof_functions.png")
        mb = results_dir / "microbench.json"
        if mb.is_file():
            _draw_decomposition(report, json.loads(mb.read_text()), results_dir / "hookprof_step_decomposition.png")
    except Exception as e:  # noqa: BLE001
        print(f"[analyze] figure skipped: {e!r}")
    return 0


#: the dataviz skill's validated categorical order (light surface), used in slot order
PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
TEXT_PRIMARY, TEXT_SECONDARY, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"

#: hook lines grouped by what they do (report vocabulary); (file, line) -> group
LINE_GROUPS = (
    ("gate gemv + bias  (hidden.float() @ w + b)", (("lens_adapters.py", 3056),)),
    ("relu / scale / outer product  (eff*g)·v", (("lens_adapters.py", 3058), ("lens_adapters.py", 3059),
                                                 ("functional.py", None))),
    ("mask: torch.tensor(positions) H2D sync", (("lens_adapters.py", 2918),)),
    ("mask: torch.ones + return", (("lens_adapters.py", 2941), ("lens_adapters.py", 2946),
                                   ("lens_adapters.py", 3060))),
    ("mask.any() D2H sync", (("lens_adapters.py", 3062),)),
    ("mask multiply + cast + add", (("lens_adapters.py", 3064), ("lens_adapters.py", 3065))),
    ("slot loop: delta / clone / new_hidden[rows] write", (("lens_worker.py", 1875), ("lens_worker.py", 1877),
                                                           ("lens_worker.py", 1878), ("lens_worker.py", 1838))),
    ("slot loop: registry.apply call + op scans", (("lens_worker.py", 1840), ("lens_adapters.py", 3014),
                                                   ("lens_adapters.py", 2984), ("lens_adapters.py", None),
                                                   ("lens_steer.py", None))),
    ("classify: token flatten (2 full list copies / request)", (("lens_worker.py", 972), ("lens_worker.py", 664),
                                                                ("lens_worker.py", 673))),
    ("classify: GPU syncs (torch.equal, .tolist)", (("lens_worker.py", 503), ("lens_batch.py", 158))),
    ("classify: dataclasses + validation (Segment/HookContext/RequestSlot)", (("lens_batch.py", None),
                                                                              ("lens_worker.py", None))),
    ("SelectionRecord._observe", (("lens_worker.py", 1312), ("lens_worker.py", None))),
)


def _group_shares(stats: Mapping[str, Any]) -> Dict[str, float]:
    """Fold the innermost-hook-line self shares into LINE_GROUPS (first match
    wins; (file, None) is a wildcard for the file). Samples whose innermost hook
    frame is a lens file but matches no explicit line fall to the file wildcard."""
    out = {g: 0.0 for g, _ in LINE_GROUPS}
    for label, share in stats.get("lens_line_self", []):
        # label = "func (file.py:line)"
        try:
            loc = label.rsplit("(", 1)[1].rstrip(")")
            fname, line = loc.rsplit(":", 1)
            line = int(line) if line != "None" else None
        except (IndexError, ValueError):
            continue
        for g, keys in LINE_GROUPS:
            if any(k[0] == fname and k[1] == line for k in keys):
                out[g] += share
                break
        else:
            for g, keys in LINE_GROUPS:
                if any(k[0] == fname and k[1] is None for k in keys):
                    out[g] += share
                    break
    return out


def _draw(report: Mapping[str, Any], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = [p for p in report["points"] if p.get("stats") and p["arm"] == "steered"]
    if not pts:
        return
    groups = [g for g, _ in LINE_GROUPS]
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=140, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    n = len(pts)
    h = 0.8 / n
    ys = list(range(len(groups)))
    for j, p in enumerate(pts):
        shares = _group_shares(p["stats"])
        vals = [100.0 * shares[g] for g in groups]
        yy = [y + (j - (n - 1) / 2) * h for y in ys]
        bars = ax.barh(yy, vals, height=h * 0.92, color=PALETTE[j], label=f"steered B={p['B']}  "
                       f"(hook = {100 * p['stats']['hook_share']:.0f}% of thread time)")
        for y, v in zip(yy, vals):
            if v >= 0.8:
                ax.text(v + 0.4, y, f"{v:.1f}%", va="center", ha="left", fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(ys)
    ax.set_yticklabels(groups, fontsize=8.5, color=TEXT_PRIMARY)
    ax.invert_yaxis()
    ax.set_xlabel("% of EngineCore busy-thread samples, charged to the innermost lens line (py-spy, nonblocking)",
                  fontsize=9, color=TEXT_SECONDARY)
    fig.suptitle("Steered decode step by innermost hook line  (qwen35_4b, psr_kl_all: 32 hooked "
                 "layers, positions=response)", fontsize=10, color=TEXT_PRIMARY, x=0.02, ha="left")
    ax.grid(True, axis="x", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)
    ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"[analyze] wrote {out}")


def _draw_decomposition(report: Mapping[str, Any], micro: Mapping[str, Any], out: Path) -> None:
    """Measured step time per point next to the microbenchmarked components
    (production registry, same GPU): the apply loop as written, the per-layer
    classification x32, and the unattributed remainder; plus the projected step
    after the fix (passthrough + the vectorized per-layer apply + one
    classification)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by = {(p["arm"], p["B"]): (p.get("engine") or {}).get("step_ms_implied") for p in report["points"]}
    Bs = sorted({B for (arm, B) in by if arm == "steered" and by[("passthrough", B)] and by[("steered", B)]})
    if not Bs:
        return
    labels, stacks = [], []
    comps = ("model forward (= passthrough step)", "apply loop as written (microbench)",
             "classification x32 layers (microbench)", "remainder (sync-wait, hidden+residual, observe, ...)")
    for B in Bs:
        pt, st = by[("passthrough", B)], by[("steered", B)]
        apply = micro.get(f"slot_loop_per_step_32layers_B{B}_ms")
        cls = micro.get(f"classify_per_step_32layers_B{B}_ms")
        if apply is None or cls is None:
            continue
        rem = max(0.0, st - pt - apply - cls)
        labels.append(f"steered B={B}\nmeasured {st:.0f} ms")
        stacks.append((pt, apply, cls, rem))
        vec = micro.get(f"batched_per_step_32layers_B{B}_ms") or micro.get("batched_per_step_32layers_B32_ms")
        cls1 = (micro.get(f"classify_one_layer_B{B}_us") or 0) / 1000.0
        labels.append(f"projected after fix B={B}\n~{pt + vec + cls1:.0f} ms")
        stacks.append((pt, vec, cls1, 0.0))
    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=140, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = list(range(len(labels)))
    bottom = [0.0] * len(labels)
    for k, name in enumerate(comps):
        vals = [s[k] for s in stacks]
        ax.bar(x, vals, bottom=bottom, width=0.62, color=PALETTE[k], label=name, edgecolor=SURFACE, linewidth=2)
        for xi, (b0, v) in enumerate(zip(bottom, vals)):
            if v >= 8:
                ax.text(xi, b0 + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=8, color="#ffffff")
        bottom = [b0 + v for b0, v in zip(bottom, vals)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, color=TEXT_PRIMARY)
    ax.set_ylabel("decode step time, ms", fontsize=9, color=TEXT_SECONDARY)
    ax.set_title("Steered step time: measured vs its microbenchmarked parts, and the projected fix",
                 fontsize=10.5, color=TEXT_PRIMARY, loc="left")
    ax.grid(True, axis="y", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8.5)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"[analyze] wrote {out}")


# --------------------------------------------------------------------------- #


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="stand up, drive, profile")
    r.add_argument("--model", default="qwen35_4b")
    r.add_argument("--repo-dir", default=str(REPO))
    r.add_argument("--out", required=True, help="results root; per-model dir beneath it")
    r.add_argument("--objects-root", required=True)
    r.add_argument("--cell", default="si_constitution__psr_kl_all")
    r.add_argument("--max-model-len", type=int, default=7424)
    r.add_argument("--max-num-seqs", type=int, default=64)
    r.add_argument("--gpu-mem-util", default="0.90")
    r.add_argument("--ready-timeout-s", type=int, default=1500)
    r.add_argument("--manifest", default="configs/capeval/manifests/petri_twosi.yaml")
    r.add_argument("--points", default="steered:32,passthrough:32,steered:8,passthrough:8")
    r.add_argument("--prompt-tokens", type=int, default=1000)
    r.add_argument("--max-tokens", type=int, default=512)
    r.add_argument("--warmup-s", type=float, default=20.0)
    r.add_argument("--measure-s", type=float, default=60.0, help="steady-state window per point")
    r.add_argument("--gap-s", type=float, default=12.0, help="idle gap between points")
    r.add_argument("--torch-profile-s", type=float, default=2.0, help="0 disables the trailing points")
    r.add_argument("--torch-profile-points", default="steered:32,passthrough:32",
                   help="dedicated trailing points that run vLLM's torch profiler (counts only)")
    r.set_defaults(fn=run)
    a = sub.add_parser("analyze", help="fold a run's profiles into tables (CPU only)")
    a.add_argument("--model", default="qwen35_4b")
    a.add_argument("--out", required=True)
    a.add_argument("--top-n", type=int, default=25)
    a.add_argument("--speedscope", default=None, help="default <out>/<model>/pyspy_whole.speedscope")
    a.add_argument("--window", type=int, default=400, help="rolling window (samples) for hook density")
    a.add_argument("--thresh", type=float, default=0.05, help="hook density that marks a steered region")
    a.add_argument("--min-len", type=int, default=800, help="shortest region kept (samples)")
    a.set_defaults(fn=analyze)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
