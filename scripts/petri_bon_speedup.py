#!/usr/bin/env python
"""Path B: prove served BoN runs end-to-end and is verifiably STEERED, on ONE
steered cell, and measure its wall-clock against an OBJECT-MATCHED HF baseline.

This is the "just confirm vLLM works" check (small, quick, no exact
equivalence). It reuses the GPU-proven steered-smoke worker registry
(``static_r_matched`` + its ``static_r_matched__scale0`` twin), which already
carries the correct 4-token Llama-3.1 assistant header and against which
today's serving/caching gates preflighted clean — so there is ZERO
registry-export risk here.

THREE deliverables, in this order (the first two are PRIMARY and written
durably before anything destructive runs):

  1. SERVED BoN, verifiably steered. The steered cell is served THROUGH the
     RoutingProxy (in-process; routing + per-cell cache_salt + prefix
     verification are the proxy's, never a bare vLLM URL). Before any timing,
     ``assert_steering_bites`` runs the SAME serving path and refuses on an
     inert (silently-unhooked) arm — that refusal IS the "does vLLM work"
     confirmation. ``bon_generate_served`` also runs ``assert_cell_salt_isolation``
     over the route set. The served wall-clock is written to
     ``timings["static/bon"]``, the shared timing instrument, regardless of
     anything downstream.

  2. An OBJECT-MATCHED HF timing arm. The IDENTICAL intervention (the matched
     CAA vector at scale 0.05, served in-process via ``build_defense(mode=
     "caa_static_r", object_path=<the same object.pt the smoke registry was
     derived from>)``) driven through the production HF ``bon_generate`` at the
     production BoN width (``--bon-batch-size == n_max``). Because both engines
     serve the SAME object at the SAME scale on the SAME base model/template
     (meta-llama), the ratio is an ENGINE ratio modulo bf16 numerics + the
     scale-fold path — NOT object-conflated. Best-effort: any failure here is
     recorded and the served deliverables still stand.

  3. A CELL-STABLE cache-reuse PROBE (diagnostic, destructive, runs LAST).
     Serves the steered cell TWICE under a forced CELL-STABLE cache_salt
     (direct to the vLLM server, bypassing the proxy's request-unique lens
     salt for this probe only — the shipped salt policy is NOT changed) so the
     second request is a cache HIT, and measures whether steering survives the
     reuse. This tests the peer's hypothesis that a lens leg could safely
     reuse its OWN steered cached blocks. NB the shipped worker carries a
     salt-discipline guard (``lens_worker._classify_forward``) that refuses ANY
     selector-carrying RESUMED prefill when caching is on; if it fires, the
     second request returns an engine-fatal 500 rather than text, and THAT is
     recorded as the verdict (the guard, not just the salt policy, blocks
     lens-cell caching). Records a MAGNITUDE comparison, not a nonzero check:
     cache-hit-vs-twin distance against cold-vs-twin distance, so partial
     steering loss is caught, plus whether the cache-hit output is bit-identical
     to the cold one (DATA, never asserted — attention kernels are not
     batch-shape invariant, so a difference need not mean steering loss).

Outcome discipline: a served-phase failure (engine won't come up steered,
bites refuses, BoN raises) writes ``bon_speedup_failed.json`` and exits
NONZERO — that is the headline finding, and the watcher must wake on it. The
HF arm and the cache-reuse probe are wrapped so neither can lose the served
result or fail the job. Numbers/verdicts go to JSON; generated text goes to
the log only.

MODEL: meta-llama/Llama-3.1-8B-Instruct @ 0e9e39f2 (the mirror the smoke
registry was exported against; its 43-token template with the 25-token default
system block is used consistently by BOTH arms, so they stay comparable).
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:  # sisteer package
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT / "scripts") not in sys.path:  # jailbreak_run leaf helpers
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

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
#: the HF arm's object is the one the steered smoke's static_r arm serves, so it is named by the smoke's own setting (the
#: environment, then the site config; a value may spell $SHARED) when --static-r-object does not name it. No default: a run
#: with the HF arm and neither refuses by name before it writes anything.
STATIC_R_OBJECT_KEY = "SISTEER_PETRI_SMOKE_STATIC_R_OBJECT"
ARM = "static_r_matched"
TWIN = "static_r_matched__scale0"
STATIC_R_SCALE = 0.05

# plain, harmless probe prompts (this proves SERVING, not behaviour) — the
# same set the sweep's served probe and the cache-iso gate use.
PROBE_PROMPTS = (
    "List the first five prime numbers.",
    "Explain briefly why the sky is blue.",
    "Write a haiku about a quiet library.",
    "Name three uses for a paperclip.",
    "Summarize the water cycle in two sentences.",
)
PROBE_MAX_TOKENS = 48

FAILED = "bon_speedup_failed.json"
RECORD = "bon_speedup_record.json"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _tokdist(a, b) -> int:
    """Positional token/byte disagreement count + length gap — a deterministic
    distance between two id lists (greedy decoding makes it attributable)."""
    n = min(len(a), len(b))
    return sum(1 for i in range(n) if a[i] != b[i]) + abs(len(a) - len(b))


def static_r_object(given=None) -> Path:
    """The HF arm's object: ``given`` (the flag), else the setting STATIC_R_OBJECT_KEY, expanded; SystemExit naming both when
    neither names it or the value does not expand."""
    try:
        value = given or _site.setting(STATIC_R_OBJECT_KEY)
        if not value:
            raise SystemExit(f"REFUSED: --static-r-object is not given and {STATIC_R_OBJECT_KEY} is not set: name the "
                             "static_r_matched object.pt the HF arm serves (the site config's key, or the job environment's), "
                             "or pass --skip-hf")
        return _site.expand_path(value, key="--static-r-object" if given else STATIC_R_OBJECT_KEY)
    except (_site.PlaceholderError, _site.SiteConfigError) as exc:
        raise SystemExit(f"REFUSED: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", default="configs/two_si.yaml")
    ap.add_argument("--max-goals", type=int, default=8,
                    help="a pure PREFIX of the confirmatory goal set — fine "
                         "for a timing ratio (both arms take the same prefix)")
    ap.add_argument("--n-max", type=int, default=64,
                    help="BoN width; production screens at 64")
    ap.add_argument("--max-concurrency", type=int, default=64,
                    help="served fan-out ceiling; matches vLLM --max-num-seqs")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--sigma", type=float, default=0.4)
    ap.add_argument("--skip-hf", action="store_true",
                    help="run only the served arm + probes (no HF baseline)")
    ap.add_argument("--static-r-object", default=None,
                    help=f"the HF arm's object.pt (default: the setting {STATIC_R_OBJECT_KEY})")
    args = ap.parse_args()
    static_r = None if args.skip_hf else static_r_object(args.static_r_object)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in (RECORD, FAILED):
        p = out / name
        if p.exists():
            p.unlink()

    record: dict = {
        "status": "started",
        "model": MODEL, "revision": REVISION,
        "registry": str(REGISTRY),
        "static_r_object": str(static_r) if static_r else None,
        "arm": ARM, "twin": TWIN, "static_r_scale": STATIC_R_SCALE,
        "n_max": args.n_max, "max_concurrency": args.max_concurrency,
        "max_new_tokens": args.max_new_tokens, "sigma": args.sigma,
        "max_goals": args.max_goals,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "node": socket.gethostname(),
    }

    # --- heavy imports (after sys.path) ---------------------------------------
    import jailbreak_run as jr
    from transformers import AutoTokenizer
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.serving import vllm_server as vs
    from sisteer.capeval.serving.proxy import (HTTPBackend, RouteEntry,
                                               RoutingProxy)
    from sisteer.eval.attacks.bon_served import (ServedDefense,
                                                 assert_steering_bites,
                                                 bon_generate_served)

    cfg = jr._load_cfg(args.config)
    goals = jr._eval_goals(cfg, confirmatory=True)[: args.max_goals]
    n_gen = len(goals) * args.n_max
    record["n_goals"] = len(goals)
    record["n_gen_per_arm"] = n_gen
    print(f"[bon-speedup] goals={len(goals)} n_max={args.n_max} "
          f"-> {n_gen} generations/arm")

    server = None
    try:
        # ============ SERVED PHASE (primary) ==============================
        tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION,
                                            trust_remote_code=False)
        template = tok.chat_template
        if not isinstance(template, str) or not template:
            raise RuntimeError("served tokenizer carries no chat template")
        template_path = out / "chat_template.jinja"
        template_path.write_bytes(template.encode())
        template_sha16 = cm.sha256_hex16(template.encode())

        spec = cm.load_petri_spec(
            REPO_ROOT / "configs/capeval/manifests/petri_twosi.yaml")
        flags = [f for f in spec.serving_launch_flags if f != "--enable-lora"]
        # interleaved (no exact-replay determinism needed for a timing/plumbing
        # check — avoids the ~3x VLLM_BATCH_INVARIANT cost); repin to the
        # registry's own model + template.
        spec = dataclasses.replace(
            spec, target_model=MODEL, target_revision=REVISION,
            serving_mode="interleaved", serving_launch_flags=flags,
            chat_template_hash=template_sha16)

        # caching ON by default (build_server_launch prefix_caching=True) +
        # --enforce-eager composed; max_num_seqs sized to the fan-out so the
        # scheduler holds the whole concurrent batch with no chunking.
        launch = vs.build_server_launch(
            spec, port=free_port(), lens_registry_dir=REGISTRY,
            chat_template_path=template_path,
            max_model_len=2048, max_num_seqs=args.max_concurrency)
        record["prefix_caching_composed"] = launch.prefix_caching
        record["enforce_eager"] = "--enforce-eager" in launch.cmd

        server = vs.launch_server(
            launch, log_path=out / "vllm_server.log",
            ready_timeout_s=1800, install_timeout_s=900)
        att_path = out / vs.STEERING_ATTESTATION_FILE
        if att_path.is_file():
            record["attestation"] = json.loads(att_path.read_text())

        reg = json.loads((REGISTRY / "registry.json").read_text())
        by_id = {e["cell_id"]: e for e in reg["entries"]}
        ident = cm.serving_identity("proxy_vllm", "lens", "interleaved")
        routes = (
            RouteEntry(virtual_model="petri__" + ARM, cell_id=ARM,
                       cell_class="class_b", serving_identity=ident,
                       intervention_hash=by_id[ARM]["expect_hash"]),
            RouteEntry(virtual_model="petri__" + TWIN, cell_id=TWIN,
                       cell_class="control", serving_identity=ident,
                       intervention_hash=by_id[TWIN]["expect_hash"],
                       control_of=ARM),
        )
        proxy = RoutingProxy(
            routes, HTTPBackend(launch.base_url, api_key=launch.api_key),
            base_model=MODEL, serving_mode="interleaved",
            prompted_prefixes=(), exact_replay_strategy=None)

        # steering-bites probe THROUGH the proxy (same serving path the run
        # uses); greedy, request-unique salt owned by the proxy.
        import asyncio

        def sync_chat(payload):
            return asyncio.run(proxy.handle_chat_completions(dict(payload)))

        def greedy(model_name, prompt):
            r = sync_chat({"model": model_name,
                           "messages": [{"role": "user", "content": prompt}],
                           "temperature": 0.0,
                           "max_tokens": PROBE_MAX_TOKENS, "seed": 0})
            return list(r["choices"][0]["message"]["content"].encode("utf-8"))

        def probe(_arm):
            steered = [greedy("petri__" + ARM, p) for p in PROBE_PROMPTS]
            scale0 = [greedy("petri__" + TWIN, p) for p in PROBE_PROMPTS]
            return steered, scale0

        bites_diag = assert_steering_bites([ARM], probe)  # raises on inert
        record["bites"] = {"passed": True, "diag": bites_diag}
        print("[bon-speedup] steering_bites PASSED (vLLM is steered)")

        # served BoN timing (bon_generate_served re-runs salt-isolation AND
        # the bites probe internally before generating).
        served_out = out / "served"
        served = ServedDefense.over_proxy(
            proxy, virtual_model="petri__" + ARM, cell_id=ARM, system=None)
        t0 = time.time()
        bon_generate_served(
            served, goals, out_dir=served_out, routes=routes,
            n_max=args.n_max, sigma=args.sigma, trial=0,
            max_new_tokens=args.max_new_tokens,
            max_concurrency=args.max_concurrency,
            steering_arms=(ARM,), steering_probe=probe)
        served_seconds = time.time() - t0
        record["served"] = {
            "seconds": served_seconds, "n_gen": n_gen,
            "per_gen_s": served_seconds / n_gen,
            "engine": "vllm_served"}
        print(f"[bon-speedup] SERVED BoN: {served_seconds:.1f}s for "
              f"{n_gen} generations ({served_seconds / n_gen:.3f} s/gen)")

        # DURABLE WRITE of the primary deliverables BEFORE anything destructive.
        atomic_json(out / "timings_served.json", {
            "condition": "static_r_matched_pathB",
            "engine": "vllm_served",
            "timings_s": {"static/bon": served_seconds},
            "n_gen": n_gen})
        record["status"] = "served_done"
        atomic_json(out / RECORD, record)

        # ============ CELL-STABLE CACHE-REUSE PROBE (destructive, LAST) ====
        record["cache_reuse_probe"] = _cache_reuse_probe(launch, by_id, tok)
        atomic_json(out / RECORD, record)

    except BaseException as e:
        record["status"] = "failed"
        record["error"] = f"{type(e).__name__}: {e}"
        atomic_json(out / FAILED, record)
        raise
    finally:
        if server is not None:
            try:
                vs._shutdown_child(server)
            except Exception:
                pass
            gc.collect()

    # ============ HF PHASE (best-effort, object-MATCHED engine ratio) ======
    if not args.skip_hf:
        try:
            import torch  # noqa: F401
            from sisteer.models import load_model_and_tokenizer
            from sisteer.eval.jailbreak import build_defense
            from sisteer.eval.attacks.bon import bon_generate

            model, tok2 = load_model_and_tokenizer(
                MODEL, revision=REVISION, dtype="bfloat16",
                attn_implementation="sdpa")
            defense = build_defense(
                model, tok2, mode="caa_static_r",
                object_path=str(static_r), scale=STATIC_R_SCALE,
                positions="all")
            hf_out = out / "hf"
            t1 = time.time()
            bon_generate(defense, goals, out_dir=hf_out, n_max=args.n_max,
                         sigma=args.sigma, trial=0,
                         max_new_tokens=args.max_new_tokens,
                         batch_size=args.n_max)
            hf_seconds = time.time() - t1
            served_seconds = record["served"]["seconds"]
            record["hf"] = {
                "seconds": hf_seconds, "per_gen_s": hf_seconds / n_gen,
                "object_matched": True, "mode": "caa_static_r",
                "scale": STATIC_R_SCALE, "bon_batch_size": args.n_max,
                "engine": "hf"}
            record["ratio"] = {
                "hf_over_served": hf_seconds / served_seconds,
                "label": ("object-MATCHED (same matched-CAA @0.05, same "
                          "meta-llama base/template, both engines) — an ENGINE "
                          "ratio modulo bf16 numerics + the served scale-fold "
                          "path; NOT object-conflated")}
            print(f"[bon-speedup] HF BoN: {hf_seconds:.1f}s  "
                  f"ratio hf/served = {hf_seconds / served_seconds:.2f}x")
        except Exception as e:
            record["hf"] = {
                "error": f"{type(e).__name__}: {e}",
                "note": ("served result + bites + timing stand; the "
                         "object-matched ratio is deferred to Path A")}
            print(f"[bon-speedup] HF arm FAILED (best-effort): {e}")

    record["status"] = "completed"
    atomic_json(out / RECORD, record)
    print(json.dumps({k: record.get(k) for k in
                      ("served", "hf", "ratio", "bites", "cache_reuse_probe")},
                     indent=2, default=str))
    return 0


def _cache_reuse_probe(launch, by_id, tok) -> dict:
    """Serve the steered cell TWICE under a forced CELL-STABLE salt (direct to
    the vLLM server, bypassing the proxy's request-unique lens salt for this
    probe ONLY — the shipped policy is unchanged) and measure whether steering
    survives the reuse. Greedy/deterministic. Never raises: records whichever
    happens (a served cache-hit, or the salt-discipline guard's engine-fatal
    500 refusal), because it runs LAST and the engine may not survive it."""
    arm_sel = {"petri_object": ARM, "expect_hash": by_id[ARM]["expect_hash"]}
    twin_sel = {"petri_object": TWIN, "expect_hash": by_id[TWIN]["expect_hash"]}
    # long enough that several FULL 16-token blocks are cacheable, so the
    # reuse is a genuine hit rather than a sub-block miss.
    prompt = ("Please answer thoroughly and precisely. Walk through the water "
              "cycle one stage at a time: evaporation from oceans and lakes, "
              "transpiration from plants, condensation into clouds, "
              "precipitation as rain or snow, and collection in rivers and "
              "aquifers. Then explain how energy from the sun drives the "
              "whole loop, and why the total amount of water stays roughly "
              "constant over time.")
    stable = f"petri|{ARM}|{by_id[ARM]['expect_hash']}|cellstable-reuse-probe"

    def vchat(selector, cache_salt, max_tokens=96):
        payload = {"model": launch.model,
                   "messages": [{"role": "user", "content": prompt}],
                   "temperature": 0.0, "max_tokens": max_tokens, "seed": 0}
        if selector is not None:
            payload["vllm_xargs"] = selector
        if cache_salt is not None:
            payload["cache_salt"] = cache_salt
        req = urllib.request.Request(
            f"{launch.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {launch.api_key}"},
            method="POST")
        with urllib.request.urlopen(req, timeout=300) as resp:
            return int(resp.status), json.loads(resp.read().decode())

    def toks(body):
        text = body["choices"][0]["message"]["content"]
        # token-level ids (the shape the comparison expects); greedy
        # decode makes the re-tokenization comparison attributable.
        return list(tok(text, add_special_tokens=False)["input_ids"])

    def cached_tokens(body):
        u = body.get("usage") or {}
        d = u.get("prompt_tokens_details") or {}
        return d.get("cached_tokens")

    rec: dict = {
        "salt": "cell-stable (forced for this probe only; shipped policy "
                "unchanged)",
        "decoding": "greedy (temperature 0, seed 0)",
        "prompt_chars": len(prompt)}
    try:
        # 1) COLD: arm under the cell-stable salt (first use -> cold prefill).
        _, b_cold = vchat(arm_sel, stable)
        cold = toks(b_cold)
        rec["cold_cached_tokens"] = cached_tokens(b_cold)
        # 2) TWIN: scale-0, request-unique salt (cold), base-equivalent.
        _, b_twin = vchat(twin_sel, f"petri|{TWIN}|uniq|{os.urandom(6).hex()}")
        twin = toks(b_twin)
        cold_vs_twin = _tokdist(cold, twin)
        rec["cold_vs_twin_distance"] = cold_vs_twin
        # 3) CACHE HIT: SAME arm, SAME stable salt, SAME prompt -> a resumed
        #    (cache-hit) prefill. The shipped salt-discipline guard may refuse
        #    this with an engine-fatal 500; if so, that IS the verdict.
        try:
            _, b_hit = vchat(arm_sel, stable)
            hit = toks(b_hit)
            ch = cached_tokens(b_hit)
            cachehit_vs_twin = _tokdist(hit, twin)
            rec.update(
                cachehit_served=True,
                cachehit_cached_tokens=ch,
                cache_hit_confirmed=bool(ch and ch > 0),
                differs_from_twin=(hit != twin),
                cachehit_vs_twin_distance=cachehit_vs_twin,
                cachehit_equals_cold=(hit == cold))
            if not (ch and ch > 0):
                rec["verdict"] = "no_cache_hit_vacuous"
            elif hit == twin:
                rec["verdict"] = "FAIL_reuse_collapsed_to_base"
            elif cachehit_vs_twin < 0.5 * max(1, cold_vs_twin):
                # still differs from twin, but MUCH less than cold did:
                # partial steering loss on the reused blocks.
                rec["verdict"] = "FAIL_partial_steering_loss"
            else:
                rec["verdict"] = "reuse_still_bites"
        except urllib.error.HTTPError as e:
            rec.update(cachehit_served=False, guard_fired=True,
                       guard_status=int(e.code),
                       guard_detail=e.read().decode("utf-8", "replace")[:800],
                       verdict="guard_refused_resumed_selector")
        except Exception as e:  # engine died mid-request from the guard
            rec.update(cachehit_served=False, guard_fired=True,
                       guard_detail=f"{type(e).__name__}: {e}",
                       verdict="guard_refused_resumed_selector")
    except Exception as e:
        rec.update(error=f"{type(e).__name__}: {e}", verdict="probe_error")
    print(f"[bon-speedup] cache-reuse probe verdict: {rec.get('verdict')} "
          f"(cachehit_cached_tokens={rec.get('cachehit_cached_tokens')}, "
          f"cachehit_vs_twin={rec.get('cachehit_vs_twin_distance')}, "
          f"cold_vs_twin={rec.get('cold_vs_twin_distance')})")
    return rec


if __name__ == "__main__":
    _site.require_expanded(str(SHARED), key="the cluster's shared root (SISTEER_SHARED_ROOT)")
    sys.exit(main())
