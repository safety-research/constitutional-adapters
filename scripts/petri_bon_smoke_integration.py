#!/usr/bin/env python
"""Integration smoke: prove unified_sweep_run's SERVED BoN path completes and
stamps correctly on hardware, end-to-end, against a REAL campaign object.

This closes the one gap Path A's bespoke driver left: `unified_sweep_run`'s
served integration (serving-mirror resolution, cell-id translation, sidecar
stamping, ProxyClientEngine transport) had only ever been CPU-unit-tested.
Throughput is irrelevant here — Path A owns the ratio. This proves the path the
13-arm wave would actually run through completes and writes a correct
`engine=vllm_served` sidecar.

Object: swp/fitserve_A2resp/defenses/safety_refusal__psr_kl_all (sha 5bb02dee,
NOT denylisted, provenance train_budget_meta[verified], NousResearch@d10aef7) —
a CAMPAIGN object, resolved by the sweep driver all day. NOT the stage-2
two_si_defenses/cells store.

One GPU: the vLLM server is capped at 0.4 gpu-memory-utilization so
`unified_sweep_run`'s own HF model load (it builds the defense even for
--bon-engine served) fits alongside it. Server + routing proxy stood up here;
`unified_sweep_run generate --bon-engine served` run as a subprocess against the
loopback proxy. Tiny: 2 goals x n_max 8.

Outcome: a served sidecar with engine=vllm_served + a timings file => exit 0
(SMOKE PASS). Any failure => nonzero.
"""
from __future__ import annotations
import asyncio, dataclasses, json, os, socket, subprocess, sys, threading, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

NOUS = "NousResearch/Meta-Llama-3.1-8B-Instruct"
NREV = "d10aef7999a2b5ba950ab3974312feeedbfe0b77"
from sisteer import site as _site  # noqa: E402
#: the cluster's shared root (sisteer.site: the job's SISTEER_SHARED_ROOT, else the site config's cluster block). Where neither
#: sets it (an import off the cluster) it keeps its placeholder form, and a run refuses by name before it reads or writes there.
try:
    SHARED = _site.expand_path("$SHARED")
except (_site.PlaceholderError, _site.SiteConfigError):
    SHARED = Path("$SHARED")
CAMPAIGN_DDIR = SHARED / "exp/swp/fitserve_A2resp/defenses"
TT_CELL = "safety_refusal__psr_kl_all"
HID = "si_safety__psr_kl_all"


def run_out_dir() -> Path:
    """A FRESH per-run output root — never a reused one. The registry export
    writes deliberately READ-ONLY artifact snapshots (lens_steer._write_blob
    chmods 0440: immutability under a running server, not to be relaxed), so
    the export is NON-IDEMPOTENT: a reused directory is poisoned by any prior
    attempt. Exactly that happened on cluster A — job 213880 failed late (the
    defenses-dir bug) and left 0440 blobs under the shared
    bon_smoke_integration/ root; job 213947 then died in 17s at _write_blob
    (PermissionError) before reaching anything else. Key the dir on
    SLURM_JOB_ID when present (one dir per attempt, findable from the job id),
    else a UTC timestamp (pod/local runs)."""
    tag = os.environ.get("SLURM_JOB_ID") or time.strftime(
        "%Y%m%dT%H%M%SZ", time.gmtime())
    return SHARED / "petri" / "serving" / "bon_smoke_integration" / f"run_{tag}"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return int(s.getsockname()[1])


def _start_proxy_thread(app, host, port):
    from aiohttp import web
    ready = threading.Event(); loop = asyncio.new_event_loop()
    def _run():
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(app); loop.run_until_complete(runner.setup())
        loop.run_until_complete(web.TCPSite(runner, host, port).start())
        ready.set(); loop.run_forever()
    threading.Thread(target=_run, name="proxy", daemon=True).start()
    if not ready.wait(timeout=30):
        raise RuntimeError("proxy thread not ready")
    return loop


def main():
    out = run_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[smoke] out dir (fresh per-run): {out}")
    rec = {"status": "started", "object_dir": str(CAMPAIGN_DDIR / TT_CELL),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "node": socket.gethostname()}

    import dataclasses as dc_
    from transformers import AutoTokenizer
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.petri.registry_adapter import traintest_cell_source
    from sisteer.capeval.serving import vllm_server as vs
    from sisteer.capeval.serving.lens_steer import export_worker_registry
    from sisteer.capeval.serving.proxy import (HTTPBackend, RoutingProxy,
                                               export_routing_table, make_app)
    from sisteer.chat_markers import generation_header_ids

    tok = AutoTokenizer.from_pretrained(NOUS, revision=NREV)
    template_path = out / "chat_template.jinja"
    template_path.write_bytes(tok.chat_template.encode())
    tsha = cm.sha256_hex16(tok.chat_template.encode())

    spec0 = cm.load_petri_spec(REPO_ROOT / "configs/capeval/manifests/petri_twosi.yaml")
    spec = dc_.replace(
        spec0, target_model=NOUS, target_revision=NREV,
        serving_mode="interleaved", cells=["none", HID],
        serving_launch_flags=[f for f in spec0.serving_launch_flags if f != "--enable-lora"],
        scenario_bundle_hash="ab" * 8, serving_impl_commit="fedcba9876" * 4,
        chat_template_hash=tsha)
    src = traintest_cell_source(spec, repo_root=REPO_ROOT, defenses_dir=CAMPAIGN_DDIR)
    bf = {k: f"# {k}\n".encode() for k in cm.petri_expected_behavior_keys(spec)}
    pf = {"auditor": b"# a\n", "judge": b"# j\n"}
    cells = dc.enumerate_cells(spec, src, behavior_files=bf, prompt_files=pf)
    if cells.absent:
        raise RuntimeError(f"absent: {[a.cell_id for a in cells.absent]}")
    reg_dir = out / "lens_registry"
    export_worker_registry(cells, src, reg_dir,
                           assistant_header_ids=tuple(generation_header_ids(tok)))
    table_path = out / "routing_table.json"
    export_routing_table(cells, src, table_path, base_model=NOUS, serving_mode="interleaved")
    print("[smoke] export OK")

    launch = vs.build_server_launch(
        spec, port=free_port(), lens_registry_dir=reg_dir,
        chat_template_path=template_path, max_model_len=2048, max_num_seqs=8)
    # ONE GPU: cap vLLM util so unified_sweep_run's own HF model load fits
    # alongside (append-only to argv; the attestation checks scheduler geometry,
    # not memory, so this cannot break it).
    launch = dc_.replace(launch, cmd=tuple(launch.cmd) + ("--gpu-memory-utilization", "0.4"))
    server = vs.launch_server(launch, log_path=out / "vllm_server.log",
                              ready_timeout_s=1800, install_timeout_s=900)
    proxy_loop = None
    try:
        from sisteer.capeval.serving.proxy import load_routing_table
        table = load_routing_table(table_path)
        proxy = RoutingProxy(table["routes"],
                             HTTPBackend(launch.base_url, api_key=launch.api_key),
                             base_model=NOUS, serving_mode="interleaved",
                             prompted_prefixes=(), exact_replay_strategy=None)
        proxy_port = free_port()
        proxy_loop = _start_proxy_thread(make_app(proxy), "127.0.0.1", proxy_port)
        proxy_url = f"http://127.0.0.1:{proxy_port}/v1"
        print(f"[smoke] proxy up at {proxy_url}")

        # conditions TSV: one row, bon only, campaign cell.
        tsv = out / "conditions_smoke.tsv"
        cols = "# label\tcell\tcoeff\tmask\tstatic\tadaptive\tbenign\ttarget_engine\tattacker_engine\tserving_path\tmirror_override\theld_out"
        row = f"psr_smoke\t{TT_CELL}\t\tall\tbon\t-\t-\thf\tvllm\thf\t\t-"
        tsv.write_text(cols + "\n" + row + "\n")

        sweep_out = out / "sweep_out"
        cmd = [sys.executable, str(REPO_ROOT / "scripts/unified_sweep_run.py"), "generate",
               "--conditions", str(tsv), "--config", "configs/two_si.yaml",
               "--out-root", str(sweep_out), "--defenses-dir", str(CAMPAIGN_DDIR),
               "--task-id", "0", "--bon-engine", "served",
               "--bon-proxy-url", proxy_url, "--bon-routing-table", str(table_path),
               "--bon-n-max", "8", "--max-goals", "2", "--bon-max-concurrency", "8"]
        rec["sweep_cmd"] = cmd
        print("[smoke] running unified_sweep_run --bon-engine served ...")
        # the petri-vllm venv is a requirements-file venv (sisteer NOT installed),
        # so the child needs src (and scripts) on PYTHONPATH — sys.executable is
        # already the venv python. This is the 212216/213880 ModuleNotFoundError.
        child_env = dict(os.environ)
        child_env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts"),
             child_env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        r = subprocess.run(cmd, cwd=str(REPO_ROOT), env=child_env,
                           capture_output=True, text=True, timeout=3600)
        (out / "sweep_stdout.log").write_text(r.stdout)
        (out / "sweep_stderr.log").write_text(r.stderr)
        rec["sweep_returncode"] = r.returncode
        print(f"[smoke] unified_sweep_run rc={r.returncode}")
        if r.returncode != 0:
            raise RuntimeError(f"unified_sweep_run served exited {r.returncode} "
                               f"(see sweep_stderr.log): {r.stderr[-1500:]}")

        # verify: a served responses file + a sidecar stamping engine=vllm_served
        resp = list(sweep_out.rglob("bon_responses.jsonl"))
        sidecars = list(sweep_out.rglob("*.json"))
        served_rows = 0
        if resp:
            for line in resp[0].read_text().splitlines():
                if line.strip() and json.loads(line).get("engine") == "vllm_served":
                    served_rows += 1
        stamped = []
        for sc in sidecars:
            try:
                d = json.loads(sc.read_text())
            except Exception:
                continue
            blob = json.dumps(d)
            if "vllm_served" in blob or "proxy_vllm" in blob:
                stamped.append(sc.name)
        timings = list(sweep_out.rglob("timings_*.json"))
        rec["served_rows"] = served_rows
        rec["responses_files"] = [str(p) for p in resp]
        rec["served_stamped_sidecars"] = stamped
        rec["timings_files"] = [str(p) for p in timings]
        static_bon = None
        for t in timings:
            td = json.loads(t.read_text()).get("timings_s", {})
            if "static/bon" in td:
                static_bon = td["static/bon"]
        rec["timings_static_bon"] = static_bon

        ok = served_rows > 0 and bool(stamped)
        rec["status"] = "SMOKE_PASS" if ok else "SMOKE_FAIL"
        (out / "smoke_record.json").write_text(json.dumps(rec, indent=2, default=str))
        print(json.dumps({k: rec[k] for k in
                          ("status", "served_rows", "served_stamped_sidecars",
                           "timings_static_bon", "sweep_returncode")}, indent=2, default=str))
        if not ok:
            raise RuntimeError(f"smoke did not confirm served stamping: "
                               f"served_rows={served_rows} stamped={stamped}")
    finally:
        if proxy_loop is not None:
            try:
                proxy_loop.call_soon_threadsafe(proxy_loop.stop)
            except Exception:
                pass
        try:
            vs._shutdown_child(server)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    _site.require_expanded(str(SHARED), key="the cluster's shared root (SISTEER_SHARED_ROOT)")
    sys.exit(main())
