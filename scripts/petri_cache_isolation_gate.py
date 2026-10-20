#!/usr/bin/env python
"""LIVE GATE — prefix caching ON, with the isolation proof, on one GPU.

The leakage question, settled on hardware. vLLM v0.25.1 keys KV
cache blocks on token ids + the request ``cache_salt`` (block 0, chained
into every descendant) + the LoRA name — NEVER the steering selector
(``vllm_xargs``/``extra_args``), and cached tokens SKIP the forward, so
the lens hooks never fire on them. Two requests with the same tokens and
the same salt therefore share KV: a scale-0 twin resuming a steered
arm's blocks would inherit the arm's steered KV under its own identity.
The isolation is the proxy's per-request ``cache_salt`` (cell-stable for
unhooked/LoRA legs; REQUEST-UNIQUE for lens legs, whose prefills must
never resume). The salt LOGIC is CPU-tested (test_petri_prefix_cache);
this gate proves the vLLM CACHING BEHAVIOUR the logic rests on, live:

1. THE LAUNCH PATH. ``build_server_launch`` composes
   ``--enable-prefix-caching`` by default and ``launch_server`` attests
   the RESOLVED ``enable_prefix_caching`` in every worker at readiness.
   A server that came up with caching off (or unattestable) never
   reaches the checks below.
2. STEERING STILL BITES (text level, request_safe_proof's own criterion,
   caching ON): the arm's output differs from the undefended base on at
   least one gate prompt. The token-level ``fidelity.steering_bites`` is
   the in-process smoke's job (petri_steered_smoke); this is its
   caching-ON live-HTTP complement.
3. ISOLATION HOLDS: the arm's scale-0 twin, served on the SAME prompt
   with a DIFFERENT (request-unique) salt, prefills COLD and reproduces
   the undefended base — it did not inherit the arm's steered KV.
3b. THE SHARING CASE (resumed prefills for per-position ops, 2026-09-06;
   lens_adapters module docstring, RESUMED PREFILLS): for a registry cell
   whose ops CAN resume (``lens_adapters.resumed_prefill_ok`` over the
   loaded registry's adapted ops — every op per-position, at all
   positions, in both phases), two prompts sharing a long prefix are
   served under ONE cell-stable salt: the second (warm) request must be
   SERVED, the worker's SelectionRecord must attest it RESUMED
   (``prefill_resumed``, cached >= 1, computed >= 1, a modifying
   application on the resumed rows), and its text must differ from the
   undefended base on the same prompt (steering bit on the resumed
   prefill). The smoke registry this gate reuses (static_r_matched +
   alphasteer_official — a prefill-only vector and a broadcast op) has NO
   such cell, so on it the case is recorded ``skipped`` with the reason,
   never silently passed; it runs the moment the registry carries one
   (a psr / vector_all / lora "all"-positions cell). Step 5's engine
   backstop is UNCHANGED for non-resumable kinds: the twin here is
   static_r's, and the worker still refuses its same-salt resume.
4. THE PROXY REFUSES FIRST, AND THE SERVER SURVIVES: the same-salt
   steered resume attempted THROUGH the routing proxy (the production
   path — audit traffic only ever reaches vLLM through it) is refused
   at the proxy with a 409 BEFORE it reaches the engine, and the stack
   is PROVEN still serving by requests that succeed afterwards, with
   isolation still holding. Job 227309 read as a pass only because the
   engine-fatal guard trip was the suite's LAST operation — it carried
   zero evidence the server survives a trip; this step is that
   evidence for the path production traffic actually takes.
5. THE ENGINE BACKSTOP FIRES — LAST, and deliberately fatal: the twin
   served DIRECTLY at vLLM (bypassing the proxy, as only non-proxy
   traffic could) with the ARM'S OWN salt resumes the arm's blocks;
   because it carries a steering selector, the worker's salt-discipline
   guard refuses the resumed-shape forward rather than serve a request
   whose cached rows the hooks never touched. That refusal is
   engine-fatal (EngineCore dies, the server with it) — which is
   exactly why step 4's proxy-level guard must catch proxy traffic
   first, and why nothing may touch the server after this step.

Reuses the smoke's real, GPU-proven worker registry
(static_r_matched + alphasteer_official + scale-0 twins) and the
production ``vllm_server`` launch/attestation path — no new gathering.
Served against meta-llama/Llama-3.1-8B-Instruct @ 0e9e39f2 (the model
the registry was EXPORTED against, so its recorded assistant header
matches the served template — the manifest's NousResearch pin is for
the canonical audit, not this registry-reuse test).

Outcome discipline: bootstrap blockers -> cache_iso_unavailable.json +
exit 0; a validation failure (no attestation, no bite, a leak, the proxy
not refusing at 409 or the stack not serving afterwards, the engine
backstop not firing) -> cache_iso_failed.json + exit NONZERO; success ->
cache_iso_record.json. Numbers and verdicts in JSON; generated text to
the log only.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.serving import vllm_server as vs  # noqa: E402

# the model the smoke registry was exported against (its assistant header
# is recorded in registry.json; serving a different template would fail
# the worker's startup header cross-check).
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
from sisteer import site as _site  # noqa: E402
#: the cluster's shared root (sisteer.site: the job's SISTEER_SHARED_ROOT, else the site config's cluster block). Where neither
#: sets it (an import off the cluster) it keeps its placeholder form, and a run refuses by name before it reads or writes there.
try:
    SHARED = _site.expand_path("$SHARED")
except (_site.PlaceholderError, _site.SiteConfigError):
    SHARED = Path("$SHARED")
REGISTRY = SHARED / "petri/serving/steered_smoke/registry"
OUT = SHARED / "petri/serving/cache_isolation_gate"
ARM = "static_r_matched"
TWIN = "static_r_matched__scale0"

UNAVAILABLE = "cache_iso_unavailable.json"
FAILED = "cache_iso_failed.json"
RECORD = "cache_iso_record.json"

# plain, harmless probes — this gate proves SERVING behaviour, not
# behaviour. The arm bites on generic prompts (it steers every position).
PROMPTS = (
    "List the first five prime numbers.",
    "Explain briefly why the sky is blue.",
    "Write a haiku about a quiet library.",
    "Name three uses for a paperclip.",
    "Summarize the water cycle in two sentences.",
)
MAX_TOKENS = 48


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _post(url: str, body: dict, *, token: str, timeout: float = 240.0):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {token}"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return int(resp.status), json.loads(resp.read().decode())


def chat(launch, prompt, *, selector=None, cache_salt=None):
    """One /v1/chat/completions call. Mirrors the proxy's upstream
    rewrite: the selector rides ``vllm_xargs`` and the KV-cache
    namespace rides top-level ``cache_salt`` — the two fields whose
    interaction this gate exists to test."""
    payload = {
        "model": launch.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": MAX_TOKENS, "seed": 1234,
    }
    if selector is not None:
        payload["vllm_xargs"] = selector
    if cache_salt is not None:
        payload["cache_salt"] = cache_salt
    status, body = _post(f"{launch.base_url}/v1/chat/completions", payload,
                         token=launch.api_key)
    if status != 200:
        raise RuntimeError(f"chat status {status}: {body}")
    return str(body["choices"][0]["message"]["content"])


def uniq(tag: str) -> str:
    return f"petri|{tag}|{os.urandom(8).hex()}"


#: the sharing case's prompt pair: one long shared prefix (the passage), two
#: short distinct tails, so the warm request is a genuine partial resume
SHARED_PASSAGE = ("The river runs quietly past the old stone mill while the miller counts "
                  "sacks of grain and the afternoon light moves slowly across the wooden "
                  "floor. ") * 12
SHARING_PROMPTS = (SHARED_PASSAGE + "First question: what did the miller count?",
                   SHARED_PASSAGE + "Second question: where did the light move?")


def resumable_registry_cells(registry_dir: Path) -> dict:
    """cell id -> ``lens_adapters.resumed_prefill_ok`` over the registry's
    ADAPTED ops (the worker's own predicate, computed on the launcher side
    by loading the same registry the server serves —
    ``WorkerObjectRegistry.load`` adapts every object on the CPU) plus the
    entry's ``forced_scale``, so the sharing case can pick a LIVE arm
    (``forced_scale`` None) rather than its scale-0 twin."""
    from sisteer.capeval.serving import lens_steer as ls
    reg = ls.WorkerObjectRegistry.load(Path(registry_dir))
    index = json.loads((Path(registry_dir) / ls.REGISTRY_INDEX).read_text())
    forced = {e["cell_id"]: e.get("forced_scale") for e in index["entries"]}
    return {cid: {"resumed_prefill_ok": ok, "forced_scale": forced.get(cid)}
            for cid, ok in reg.resumed_prefill_ok_by_cell().items()}


def pick_sharing_cell(cells: dict):
    """The first (sorted) LIVE cell whose ops can resume, or ``None`` with
    the reason the case is skipped — stated, never a silent pass."""
    live = sorted(c for c, v in cells.items() if v["resumed_prefill_ok"] and v["forced_scale"] is None)
    if live:
        return live[0], None
    return None, ("no live registry cell has resumable ops (lens_adapters.resumed_prefill_ok False "
                  f"for every arm: {sorted(c for c, v in cells.items() if v['forced_scale'] is None)}); "
                  "the sharing case needs a per-position, all-positions, both-phases cell "
                  "(psr / vector_all / lora at positions='all')")


def _request_entries(payload) -> dict:
    """The per-request entries of the ``sisteer_selection_record`` reply
    (``{"results": [<worker>, ...]}`` at the pinned route), union over
    workers."""
    if isinstance(payload, dict) and "results" in payload:
        results = payload["results"]
    elif isinstance(payload, dict):
        results = [payload]          # one worker's bare record
    else:
        results = payload
    out: dict = {}
    for r in (results if isinstance(results, list) else [results]):
        if isinstance(r, dict):
            for k, v in (r.get("requests") or {}).items():
                out.setdefault(str(k), v)
    return out


def sharing_case_verdict(rec: dict) -> None:
    """Raise unless the sharing case's record shows the warm request SERVED,
    attested RESUMED (one entry, cached >= 1, computed >= 1, a modifying
    application on the resumed rows) and biting against the base."""
    if rec.get("warm_status") != 200:
        raise RuntimeError(f"sharing case: the warm request on a RESUMABLE cell was not served "
                           f"(status {rec.get('warm_status')!r}) — the worker refused a resume it must accept")
    att = rec.get("warm_attestation") or {}
    if att.get("n_entries") != 1:
        raise RuntimeError(f"sharing case: expected exactly one SelectionRecord entry for the warm request, "
                           f"got {att.get('n_entries')!r}")
    if att.get("prefill_resumed") is not True:
        raise RuntimeError("sharing case: the worker did not attest the warm request RESUMED "
                           f"(prefill_resumed={att.get('prefill_resumed')!r}) — the shared salt did not resume")
    if not (isinstance(att.get("prefill_cached_positions"), int) and att["prefill_cached_positions"] >= 1
            and isinstance(att.get("prefill_computed_positions"), int) and att["prefill_computed_positions"] >= 1):
        raise RuntimeError(f"sharing case: implausible resume counts {att!r}")
    if not (isinstance(att.get("n_applications_modifying_resumed"), int)
            and att["n_applications_modifying_resumed"] >= 1):
        raise RuntimeError("sharing case: no op modified the resumed rows — a silently unsteered prompt tail")
    if not rec.get("warm_bites"):
        raise RuntimeError("sharing case: the warm steered output equals the undefended base on the same "
                           "prompt — steering did not bite on the resumed prefill")


def _post_status(url: str, body: dict, *, timeout: float = 240.0):
    """POST that returns (status, parsed body) instead of raising on a
    non-2xx — step 4 asserts on the proxy's refusal status itself."""
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return int(resp.status), json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = {"error": {"message": raw[:2000]}}
        return int(e.code), parsed


def _start_routing_proxy(proxy: "px.RoutingProxy", port: int):
    """Serve the REAL production HTTP surface (``px.make_app``) on a
    background thread's event loop; returns a stop() that tears the loop
    down. Readiness is polled on /v1/models so no step races the bind."""
    from aiohttp import web

    loop = asyncio.new_event_loop()
    started = threading.Event()

    def _run():
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(px.make_app(proxy))
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, "127.0.0.1", port)
        loop.run_until_complete(site.start())
        started.set()
        loop.run_forever()
        loop.run_until_complete(runner.cleanup())

    thread = threading.Thread(target=_run, name="gate-proxy", daemon=True)
    thread.start()
    if not started.wait(timeout=30):
        raise RuntimeError("routing proxy did not start within 30s")
    deadline = time.time() + 30
    url = f"http://127.0.0.1:{port}/v1/models"
    while True:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    break
        except Exception:  # noqa: BLE001 — poll until the deadline
            if time.time() > deadline:
                raise RuntimeError("routing proxy never became ready")
            time.sleep(0.2)

    def stop():
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=10)

    return stop


def _proxy_chat(port: int, vmodel: str, prompt: str):
    """One chat call THROUGH the routing proxy: the request names only
    the VIRTUAL model — selector and cache_salt are the proxy's to
    inject (a client-supplied one refuses), exactly like audit traffic."""
    return _post_status(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        {"model": vmodel,
         "messages": [{"role": "user", "content": prompt}],
         "temperature": 0, "max_tokens": MAX_TOKENS, "seed": 1234})


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in (RECORD, FAILED, UNAVAILABLE):
        p = OUT / name
        if p.exists():
            p.unlink()

    # --- bootstrap (blockers -> unavailable + exit 0) --------------------
    try:
        vs.require_vllm()
        from sisteer.capeval.serving import lens_steer as ls
        ls.require_vllm_lens()
    except Exception as e:  # noqa: BLE001 — any bring-up blocker
        atomic_json(OUT / UNAVAILABLE, {
            "status": "unavailable", "reason": f"{type(e).__name__}: {e}",
            "recorded_at": time.time()})
        print(f"cache-iso gate unavailable: {e}", file=sys.stderr)
        return 0
    if not (REGISTRY / "registry.json").is_file():
        atomic_json(OUT / UNAVAILABLE, {
            "status": "unavailable",
            "reason": f"no worker registry at {REGISTRY}",
            "recorded_at": time.time()})
        return 0

    from transformers import AutoTokenizer

    spec = cm.load_petri_spec(
        Path("configs/capeval/manifests/petri_twosi.yaml"))
    # repin to the registry's model + its own template, and use the
    # interleaved mode (no exact-replay determinism needed for a
    # caching-behaviour proof; avoids the 2.97x VLLM_BATCH_INVARIANT
    # cost). Drop --enable-lora: no LoRA here.
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION,
                                        trust_remote_code=False)
    template = tok.chat_template
    if not isinstance(template, str) or not template:
        raise RuntimeError("served tokenizer carries no chat template")
    template_path = OUT / "chat_template.jinja"
    template_path.write_bytes(template.encode())
    template_sha16 = cm.sha256_hex16(template.encode())
    flags = [f for f in spec.serving_launch_flags if f != "--enable-lora"]
    spec = dataclasses.replace(
        spec, target_model=MODEL, target_revision=REVISION,
        serving_mode="interleaved", serving_launch_flags=flags,
        chat_template_hash=template_sha16)

    # caching ON by default: build_server_launch composes
    # --enable-prefix-caching and carries prefix_caching=True.
    # GEOMETRY: deliberately NOT overridden — the gate launches the
    # PRODUCTION defaults (max_model_len 16384 x DEFAULT_MAX_NUM_SEQS,
    # raised 8 -> 24 on 2026-08-27; 32 x 16384 measurably does not fit one
    # H200, see vllm_server.DEFAULT_MAX_NUM_SEQS), so the attested resolved geometry
    # and vLLM's launch-time memory profiling certify exactly what the
    # audit will run on the same H200 class. The earlier explicit
    # 2048x8 kept bring-up cheap but proved a geometry nothing serves.
    launch = vs.build_server_launch(
        spec, port=free_port(), lens_registry_dir=REGISTRY,
        chat_template_path=template_path)

    reg = json.loads((REGISTRY / "registry.json").read_text())
    by_id = {e["cell_id"]: e for e in reg["entries"]}
    arm_sel = {"petri_object": ARM, "expect_hash": by_id[ARM]["expect_hash"]}
    twin_sel = {"petri_object": TWIN,
                "expect_hash": by_id[TWIN]["expect_hash"]}

    record = {
        "status": "started",
        "model": MODEL, "revision": REVISION,
        "serving_mode": "interleaved",
        "prefix_caching_composed": launch.prefix_caching,
        "enable_prefix_caching_flag": "--enable-prefix-caching" in launch.cmd,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": socket.gethostname(),
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()).hexdigest(),
        "registry_sha256": hashlib.sha256(
            (REGISTRY / "registry.json").read_bytes()).hexdigest(),
        "template_sha16": template_sha16,
        "launch_cmd": list(launch.cmd),
    }
    if not launch.prefix_caching or "--enable-prefix-caching" not in launch.cmd:
        atomic_json(OUT / FAILED, dict(
            record, status="failed",
            error="build_server_launch did not compose caching ON"))
        raise RuntimeError("caching was not composed ON — D2 launch path")

    proc = None
    try:
        proc = vs.launch_server(launch, log_path=OUT / "server.log",
                                ready_timeout_s=900, install_timeout_s=600)
        att = json.loads((OUT / vs.STEERING_ATTESTATION_FILE).read_text())
        record["attestation"] = att
        # (1) THE LAUNCH PATH: the resolved cache state must be ON in
        # every worker (expected_scheduler_state now asserts it).
        geom = att["scheduler_geometry"]
        record["resolved_enable_prefix_caching"] = [
            r.get("enable_prefix_caching") for r in geom["worker_reports"]]
        if geom["expected"].get("enable_prefix_caching") is not True:
            raise RuntimeError("attestation did not expect caching ON")
        for r in geom["worker_reports"]:
            if r.get("enable_prefix_caching") is not True:
                raise RuntimeError(f"worker resolved caching off: {r}")

        # (2) STEERING BITES (text, caching ON): arm != base somewhere.
        bites = []
        for prompt in PROMPTS:
            base = chat(launch, prompt, cache_salt=uniq("base"))
            armed = chat(launch, prompt, selector=arm_sel,
                         cache_salt=uniq("arm"))
            bites.append({"prompt": prompt, "different": base != armed,
                          "base": base, "arm": armed})
        record["steering_bites"] = [
            {"prompt": b["prompt"], "different": b["different"]}
            for b in bites]
        for b in bites:  # text (not JSON) — eyeball only
            print(f"[bite] different={b['different']} :: {b['prompt']}\n"
                  f"  base={b['base']!r}\n  arm ={b['arm']!r}")
        if not any(b["different"] for b in bites):
            raise RuntimeError(
                "the arm selected successfully but changed NO sampled "
                "output under caching — steering is not biting (a silent "
                "no-op serving path)")

        # (3) ISOLATION HOLDS: the scale-0 twin, unique salt, on a prompt
        # the arm just steered, prefills COLD and reproduces the base.
        iso_prompt = PROMPTS[0]
        base_cold = chat(launch, iso_prompt, cache_salt=uniq("base"))
        _arm_warm = chat(launch, iso_prompt, selector=arm_sel,
                         cache_salt=uniq("arm"))  # populate the pool
        twin_iso = chat(launch, iso_prompt, selector=twin_sel,
                        cache_salt=uniq("twin"))
        record["isolation"] = {
            "prompt": iso_prompt,
            "twin_equals_base": twin_iso == base_cold}
        print(f"[iso] twin==base: {twin_iso == base_cold}\n"
              f"  base={base_cold!r}\n  twin={twin_iso!r}")
        if twin_iso != base_cold:
            raise RuntimeError(
                "the scale-0 twin under a unique salt did NOT reproduce "
                "the undefended base after the arm served the same prompt "
                "— it inherited the arm's cached (steered) KV: isolation "
                "is NOT holding")

        # (3b) THE SHARING CASE (module docstring): a resumable cell's warm
        # request under a shared salt is served, attested resumed, and bites.
        # Skipped — recorded with the reason — when the registry has no
        # such cell (the smoke registry: static_r + alphasteer, neither
        # resumable).
        cells = resumable_registry_cells(REGISTRY)
        record["resumable_cells"] = cells
        share_cell, skip_reason = pick_sharing_cell(cells)
        if share_cell is None:
            record["sharing_case"] = {"skipped": skip_reason}
            print(f"[share] skipped: {skip_reason}")
        else:
            share_sel = {"petri_object": share_cell,
                         "expect_hash": by_id[share_cell]["expect_hash"]}
            share_salt = f"petri|{share_cell}|{by_id[share_cell]['expect_hash']}|{os.urandom(8).hex()}"
            prompt_a, prompt_b = SHARING_PROMPTS
            base_b = chat(launch, prompt_b, cache_salt=uniq("base"))
            vs._post_collective_rpc(launch, "sisteer_selection_record", [True], timeout_s=60,
                                    what="sharing case: reset")
            _writer = chat(launch, prompt_a, selector=share_sel, cache_salt=share_salt)
            vs._post_collective_rpc(launch, "sisteer_selection_record", [True], timeout_s=60,
                                    what="sharing case: reset before the warm request")
            share_rec = {"cell": share_cell, "salt_shared": True}
            try:
                warm_b = chat(launch, prompt_b, selector=share_sel, cache_salt=share_salt)
                share_rec.update(warm_status=200, warm_bites=(warm_b != base_b))
            except (urllib.error.HTTPError, RuntimeError) as e:
                share_rec.update(warm_status=getattr(e, "code", None), warm_error=str(e)[:2000])
            entries = _request_entries(vs._post_collective_rpc(
                launch, "sisteer_selection_record", [False], timeout_s=60,
                what="sharing case: the warm request's attestation"))
            if len(entries) == 1:
                (rid, entry), = entries.items()
                share_rec["warm_attestation"] = {
                    "n_entries": 1, "req_id": rid,
                    **{k: entry.get(k) for k in ("prefill_resumed", "prefill_cached_positions",
                                                 "prefill_computed_positions", "n_applications_modifying",
                                                 "n_applications_modifying_resumed")}}
            else:
                share_rec["warm_attestation"] = {"n_entries": len(entries)}
            record["sharing_case"] = share_rec
            print(f"[share] cell={share_cell} warm status={share_rec.get('warm_status')} "
                  f"attestation={share_rec.get('warm_attestation')} bites={share_rec.get('warm_bites')}")
            sharing_case_verdict(share_rec)

        # (4) THE PROXY REFUSES FIRST, AND THE SERVER SURVIVES. Run the
        # PRODUCTION proxy stack (the real make_app HTTP surface over an
        # HTTPBackend into this live vLLM), force the same salt
        # collision THROUGH it, and prove the collision costs one 409 —
        # the engine never sees the request. Through an unpatched proxy
        # a steered salt is request-unique by construction, so
        # cache_salt_for is pinned to a constant for the two colliding
        # calls: exactly the construction bug / uuid collision the
        # proxy-level guard exists to catch (defect: job 227309's
        # "pass" proved the refusal only as an engine death with
        # nothing served after it).
        arm_route = px.RouteEntry(
            virtual_model=f"{px.VMODEL_PREFIX}gate_arm", cell_id=ARM,
            cell_class="gate",
            serving_identity="proxy_vllm+lens+interleaved",
            intervention_hash=by_id[ARM]["expect_hash"])
        twin_route = px.RouteEntry(
            virtual_model=f"{px.VMODEL_PREFIX}gate_twin", cell_id=TWIN,
            cell_class="gate",
            serving_identity="proxy_vllm+lens+interleaved",
            intervention_hash=by_id[TWIN]["expect_hash"])
        gate_proxy = px.RoutingProxy(
            (arm_route, twin_route),
            px.HTTPBackend(launch.base_url, api_key=launch.api_key),
            base_model=launch.model, serving_mode="interleaved")
        pport = free_port()
        stop_proxy = _start_routing_proxy(gate_proxy, pport)
        proxy_guard: dict = {}
        orig_salt_for = px.RoutingProxy.cache_salt_for
        collide_via_proxy = uniq("proxy-collision")
        try:
            px.RoutingProxy.cache_salt_for = staticmethod(
                # **_kw swallows `resumed_prefill_policy`, which _rewrite
                # passes since the resumed-prefill landing; this proxy is
                # built with no attested policy map, so the regime stays
                # request-unique and the forced collision is refused as
                # before.
                lambda route, *, lens, **_kw: collide_via_proxy)
            s1, b1 = _proxy_chat(pport, arm_route.virtual_model,
                                 iso_prompt)
            if s1 != 200:
                raise RuntimeError(
                    f"steered arm through the proxy failed: {s1} {b1}")
            s2, b2 = _proxy_chat(pport, twin_route.virtual_model,
                                 iso_prompt)
        finally:
            px.RoutingProxy.cache_salt_for = staticmethod(orig_salt_for)
        proxy_guard.update(
            refused=(s2 == 409), status=s2,
            code=(b2.get("error") or {}).get("code"))
        print(f"[proxy-guard] same-salt steered resume through the proxy: "
              f"status={s2} :: {json.dumps(b2)[:300]}")
        if s2 != 409 or proxy_guard["code"] != "cache_salt_collision":
            record["proxy_guard"] = proxy_guard
            raise RuntimeError(
                f"the proxy did not refuse the same-salt steered resume "
                f"with its own 409/cache_salt_collision (got {s2} "
                f"{proxy_guard['code']!r}) — the collision would have "
                "reached the engine, whose refusal kills the server")
        # THE SURVIVAL PROOF: normal requests through the SAME proxy and
        # the SAME server now succeed, and isolation still holds — the
        # guard trip cost one request, not the run.
        s3, b3 = _proxy_chat(pport, arm_route.virtual_model, iso_prompt)
        s4, b4 = _proxy_chat(pport, twin_route.virtual_model, iso_prompt)
        stop_proxy()
        twin_after = (str(b4["choices"][0]["message"]["content"])
                      if s4 == 200 else None)
        proxy_guard.update(
            served_after_refusal=(s3 == 200 and s4 == 200),
            post_refusal_isolation_holds=(twin_after == base_cold))
        record["proxy_guard"] = proxy_guard
        print(f"[proxy-guard] served after refusal: "
              f"{proxy_guard['served_after_refusal']} (arm={s3}, "
              f"twin={s4}); isolation still holds: "
              f"{proxy_guard['post_refusal_isolation_holds']}")
        if not proxy_guard["served_after_refusal"]:
            raise RuntimeError(
                f"the stack did NOT keep serving after the proxy-level "
                f"refusal (arm={s3}, twin={s4}) — a guard that refuses "
                "but leaves a dead server is not a fix")
        if not proxy_guard["post_refusal_isolation_holds"]:
            raise RuntimeError(
                "after the proxy-level refusal the scale-0 twin no longer "
                "reproduces the undefended base — isolation degraded")

        # selection record BEFORE the destructive negative test below:
        # the guard refusal propagates out of the forward pass and the
        # vLLM engine core treats it as fatal (fail-loud by design — a
        # condemned worker's outputs are unusable, so the engine dying
        # is the property, not collateral). Nothing may touch the
        # server after step (5). Learned live: job 204465 read this
        # AFTER the guard trip and the RPC hit a dead engine.
        sel = vs._post_collective_rpc(
            launch, "sisteer_selection_record", [False], timeout_s=60,
            what="cache-iso selection record")
        record["selection_record_has_arm"] = (
            ARM in json.dumps(sel, sort_keys=True))

        # (5) THE ENGINE BACKSTOP FIRES — LAST, and deliberately fatal:
        # arm then twin on the SAME salt, DIRECTLY at vLLM (bypassing
        # the proxy, as only non-proxy traffic could) -> the twin
        # resumes the arm's blocks; carrying a selector, the worker's
        # salt-discipline guard must refuse the resumed forward (an
        # engine-fatal 500), never silently serve unsteered-cached
        # rows. Production traffic cannot get here: the proxy mints
        # request-unique lens salts AND refuses any collision at 409
        # (step 4). This proves the last line of defense for traffic
        # that never saw the proxy: loud death, never a wrong serve.
        collide = uniq("collision")
        _arm_c = chat(launch, iso_prompt, selector=arm_sel,
                      cache_salt=collide)  # commits blocks under `collide`
        guard = {"salt_shared": True}
        try:
            twin_c = chat(launch, iso_prompt, selector=twin_sel,
                          cache_salt=collide)
            guard.update(refused=False, twin_out=twin_c)
        except urllib.error.HTTPError as e:
            guard.update(refused=True, status=int(e.code),
                         detail=e.read().decode("utf-8", "replace")[:2000])
        except RuntimeError as e:
            # chat() raises RuntimeError on a non-200 (the server's 500)
            guard.update(refused=True, detail=str(e)[:2000])
        record["guard"] = {"salt_shared": guard["salt_shared"],
                           "refused": guard.get("refused")}
        print(f"[guard] same-salt steered resume refused: "
              f"{guard.get('refused')} :: {guard.get('detail', '')[:300]}")
        if not guard.get("refused"):
            raise RuntimeError(
                "a scale-0 twin sharing the arm's salt was SERVED, not "
                "refused — a steered request resumed cached rows its "
                "hooks never touched and the salt-discipline guard did "
                "not fire: caching is unsafe on this stack")

        record["status"] = "completed"
        atomic_json(OUT / RECORD, record)
        print(json.dumps({
            "status": "completed",
            "caching_resolved_on": record["resolved_enable_prefix_caching"],
            "bites": sum(b["different"] for b in bites),
            "isolation_holds": record["isolation"]["twin_equals_base"],
            "sharing_case": ("skipped" if record["sharing_case"].get("skipped")
                             else record["sharing_case"].get("warm_attestation")),
            "proxy_guard_refused": record["proxy_guard"]["refused"],
            "server_survived_proxy_refusal":
                record["proxy_guard"]["served_after_refusal"],
            "guard_fired": record["guard"]["refused"],
        }, indent=2))
        return 0
    except BaseException as e:
        record["status"] = "failed"
        record["error"] = f"{type(e).__name__}: {e}"
        atomic_json(OUT / FAILED, record)
        raise
    finally:
        if proc is not None:
            vs._shutdown_child(proc)


if __name__ == "__main__":
    _site.require_expanded(str(SHARED), key="the cluster's shared root (SISTEER_SHARED_ROOT)")
    sys.exit(main())
