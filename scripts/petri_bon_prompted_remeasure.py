#!/usr/bin/env python
"""The case where served CAN win: a long-SI PROMPTED (unhooked) cell, where the
~5.9k SI prefix genuinely caches (cell-stable salt) and HF pays its worst tax
(re-prefilling ~5.9k tokens on every generation).

Path A/fanout measured a short-prompt LENS cell — no caching by design, no
prefill tax for HF — i.e. the one cell type where served cannot win. This is the
complement: `prompted__constitution` (~5872-token SI prefix, verified), UNHOOKED
so `RoutingProxy.cache_salt_for(lens=False)` gives the CELL-STABLE salt and the
prefix caches across every request in the cell.

Same structure as the fanout job so it pairs with the lens arm: three served
transports (throttled HTTP / fixed async HTTP / in-process over-proxy) + the
matched HF baseline at --bon-batch-size 64, same 8 goals x n_max 64.

MECHANISM UNDER TEST -> MEASURED: each served transport records the cache-hit
rate (usage.prompt_tokens_details.cached_tokens / prompt_tokens). For a prompted
cell this should be high (the SI prefix is reused); for the lens cell it was ~0.
Without it a null result is uninterpretable. The prefix is WARMED once before
each transport's fan-out (one request, outside the timed section) so we measure
steady-state cached throughput and avoid a 64-wide cold-prefill stampede.

No steering object at all (base + system prompt), so no object-resolution or
denylist machinery applies. NousResearch@d10aef7, interleaved, caching ON,
max_model_len 8192 (the 5.9k prefix needs it), max_num_seqs 64.
"""
from __future__ import annotations
import asyncio, dataclasses, gc, json, os, socket, sys, threading, time
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
DDIR = SHARED / "exp/jailbreak/two_si_defenses/cells"   # only needed so the adapter builds; prompted uses no object
PROMPTED = "prompted__constitution"
PVMODEL = "petri__" + PROMPTED


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return int(s.getsockname()[1])


def _proxy_thread(app, host, port):
    from aiohttp import web
    ready = threading.Event(); loop = asyncio.new_event_loop()
    def _run():
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(app); loop.run_until_complete(runner.setup())
        loop.run_until_complete(web.TCPSite(runner, host, port).start())
        ready.set(); loop.run_forever()
    threading.Thread(target=_run, daemon=True).start()
    if not ready.wait(timeout=30):
        raise RuntimeError("proxy not ready")
    return loop


class Counting:
    """Async achat wrapper: records max concurrent in-flight AND accumulates
    prompt/cached tokens so cache-hit rate is measured, not assumed."""
    def __init__(self, inner):
        self._inner = inner
        self.inflight = self.max_inflight = 0
        self.sum_prompt = self.sum_cached = self.n = 0
    async def __call__(self, payload):
        self.inflight += 1; self.max_inflight = max(self.max_inflight, self.inflight)
        try:
            resp = await self._inner(payload)
            u = resp.get("usage") or {}
            self.sum_prompt += int(u.get("prompt_tokens") or 0)
            d = u.get("prompt_tokens_details") or {}
            self.sum_cached += int(d.get("cached_tokens") or 0)
            self.n += 1
            return resp
        finally:
            self.inflight -= 1
    def stats(self):
        return {"max_inflight": self.max_inflight, "n": self.n,
                "sum_prompt_tokens": self.sum_prompt, "sum_cached_tokens": self.sum_cached,
                "cache_hit_rate": (self.sum_cached / self.sum_prompt) if self.sum_prompt else None}


def main():
    out = SHARED / "petri/serving/bon_prompted_remeasure"
    out.mkdir(parents=True, exist_ok=True)
    rec = {"status": "started", "cell": PROMPTED, "n_max": 64, "max_concurrency": 64,
           "server_max_num_seqs": 64, "max_model_len": 8192,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "node": socket.gethostname()}

    import jailbreak_run as jr
    from transformers import AutoTokenizer
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.petri.registry_adapter import traintest_cell_source
    from sisteer.capeval.serving import vllm_server as vs
    from sisteer.capeval.serving.lens_steer import export_worker_registry
    from sisteer.capeval.serving.proxy import (HTTPBackend, ProxyClientEngine,
                                               RoutingProxy, export_routing_table,
                                               load_routing_table, make_app)
    from sisteer.eval.attacks.bon_served import ServedDefense, bon_generate_served
    from sisteer.chat_markers import generation_header_ids

    cfg = jr._load_cfg("configs/two_si.yaml")
    goals = jr._eval_goals(cfg, confirmatory=True)[:8]
    n_gen = len(goals) * 64
    rec["n_goals"] = len(goals); rec["n_gen_per_arm"] = n_gen

    tok = AutoTokenizer.from_pretrained(NOUS, revision=NREV)
    tpath = out / "chat_template.jinja"; tpath.write_bytes(tok.chat_template.encode())
    spec0 = cm.load_petri_spec(REPO_ROOT / "configs/capeval/manifests/petri_twosi.yaml")
    spec = dataclasses.replace(
        spec0, target_model=NOUS, target_revision=NREV, serving_mode="interleaved",
        cells=["none", PROMPTED],
        serving_launch_flags=[f for f in spec0.serving_launch_flags if f != "--enable-lora"],
        scenario_bundle_hash="ab" * 8, serving_impl_commit="fedcba9876" * 4,
        chat_template_hash=cm.sha256_hex16(tok.chat_template.encode()))
    src = traintest_cell_source(spec, repo_root=REPO_ROOT, defenses_dir=DDIR)
    si_text = bytes(src.get(PROMPTED).si_prefix).decode("utf-8")
    rec["si_prefix_chars"] = len(si_text)
    bf = {k: f"# {k}\n".encode() for k in cm.petri_expected_behavior_keys(spec)}
    cells = dc.enumerate_cells(spec, src, behavior_files=bf,
                              prompt_files={"auditor": b"# a\n", "judge": b"# j\n"})
    reg_dir = out / "lens_registry"
    export_worker_registry(cells, src, reg_dir,
                           assistant_header_ids=tuple(generation_header_ids(tok)),
                           zero_op_hidden_size=4096)
    tbl = out / "routing_table.json"
    export_routing_table(cells, src, tbl, base_model=NOUS, serving_mode="interleaved")
    routes = load_routing_table(tbl)["routes"]

    server = None; proxy_loop = None
    try:
        launch = vs.build_server_launch(spec, port=free_port(), lens_registry_dir=reg_dir,
                                        chat_template_path=tpath, max_model_len=8192,
                                        max_num_seqs=64)
        # vllm 0.25.1 gates usage.prompt_tokens_details behind this server flag
        # (cli_args.py: enable_prompt_tokens_details=False by default); without
        # it cached_tokens is absent and the client-side rate reads 0.0.
        launch = dataclasses.replace(launch, cmd=tuple(launch.cmd) + ("--enable-prompt-tokens-details",))
        rec["prefix_caching_composed"] = launch.prefix_caching
        server = vs.launch_server(launch, log_path=out / "vllm_server.log",
                                  ready_timeout_s=1800, install_timeout_s=900)
        proxy = RoutingProxy(routes, HTTPBackend(launch.base_url, api_key=launch.api_key),
                             base_model=NOUS, serving_mode="interleaved",
                             prompted_prefixes=load_routing_table(tbl)["prompted_prefixes"],
                             exact_replay_strategy=None)
        pport = free_port()
        proxy_loop = _proxy_thread(make_app(proxy), "127.0.0.1", pport)
        proxy_url = f"http://127.0.0.1:{pport}/v1"
        client = ProxyClientEngine(proxy_url, identity=cm.serving_identity("proxy_vllm", "passthrough", "interleaved"))
        client.load(require_virtual_models=[PVMODEL])

        def warm():
            # one cold request to populate the ~5.9k prefix in the cell-stable
            # cache namespace, so the fan-out below measures cached throughput.
            client.chat_completions({"model": PVMODEL,
                "messages": [{"role": "system", "content": si_text},
                             {"role": "user", "content": "Say OK."}],
                "temperature": 0.0, "max_tokens": 4, "seed": 0})

        import aiohttp
        results = {}

        def _served(tag, achat_counter):
            warm()
            d = out / tag; d.mkdir(exist_ok=True)
            defense = ServedDefense(achat_counter, virtual_model=PVMODEL,
                                    cell_id=PROMPTED, system=si_text)
            t = time.time()
            bon_generate_served(defense, goals, out_dir=d, routes=routes, n_max=64,
                                sigma=0.4, trial=0, max_new_tokens=512,
                                max_concurrency=64, steering_arms=(), steering_probe=None)
            sec = time.time() - t
            results[tag] = {"seconds": sec, "per_gen_s": sec / n_gen, **achat_counter.stats()}
            print(f"[prompted] {tag}: {sec:.1f}s ({sec/n_gen:.4f} s/gen) "
                  f"max_inflight={achat_counter.max_inflight} "
                  f"cache_hit_rate={results[tag]['cache_hit_rate']}")

        # 1. throttled HTTP (Path A transport): to_thread over the sync client
        _served("served_throttled_http",
                Counting(lambda p: asyncio.to_thread(client.chat_completions, dict(p))))
        # 2. fixed async HTTP: true 64-wide
        sess = {"s": None}
        async def fixed_inner(p):
            if sess["s"] is None:
                sess["s"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600))
            async with sess["s"].post(proxy_url + "/chat/completions", json=dict(p),
                                      headers={"Authorization": "Bearer petri-proxy"}) as r:
                b = await r.text()
                if r.status != 200:
                    raise RuntimeError(f"proxy {r.status}: {b[:400]}")
                return json.loads(b)
        _served("served_fixed_http", Counting(fixed_inner))
        # 3. in-process over-proxy (ceiling)
        _served("served_over_proxy", Counting(lambda p: proxy.handle_chat_completions(dict(p))))
        rec["served"] = results
        (out / "bon_prompted_remeasure_record.json").write_text(json.dumps(rec, indent=2, default=str))
    except BaseException as e:
        rec["status"] = "failed"; rec["error"] = f"{type(e).__name__}: {e}"
        (out / "bon_prompted_remeasure_failed.json").write_text(json.dumps(rec, indent=2, default=str))
        raise
    finally:
        if proxy_loop is not None:
            try: proxy_loop.call_soon_threadsafe(proxy_loop.stop)
            except Exception: pass
        if server is not None:
            try: vs._shutdown_child(server)
            except Exception: pass
            gc.collect()

    rec["status"] = "completed"
    (out / "bon_prompted_remeasure_record.json").write_text(json.dumps(rec, indent=2, default=str))
    print(json.dumps({"served": rec["served"], "hf": rec.get("hf"),
                      "ratios_hf_over_served": rec.get("ratios_hf_over_served")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    _site.require_expanded(str(SHARED), key="the cluster's shared root (SISTEER_SHARED_ROOT)")
    sys.exit(main())
