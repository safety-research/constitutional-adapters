#!/usr/bin/env python
"""Engine A/B for the NON-BoN attack families on the long-SI PROMPTED cell.

BoN's served-vs-HF win on ``prompted__si_constitution`` (the ~5.9k-token SI) is
already measured (87.6x; petri_bon_prompted.py). BoN is one member of a
six-family composite; the OTHER five — artprompt, codeattack, drattack,
pasttense, renellm — are now the MAJORITY of a long-SI condition's remaining
generation cost and had never been measured on either engine. This driver
measures all five (plus grok420, reported SEPARATELY — a live family that is
NOT part of the composite) on BOTH engines, same cell, same pinned
goal set, so the per-family HF/served ratio is attributable to the engine and
the summed per-condition total is a real planning number, not an extrapolation.

WHAT IS AND ISN'T VARIED. This is a deliberate engine A/B, so mixing engines
across the two arms is the POINT (the no-mixing rule is about pooling results
into one figure, not about a timing contrast). Everything else is held: the
same cell (base weights + the constitution SI as a system turn — UNHOOKED, no
steering object), the same 112-goal pinned held-out set the matrix_fill
reference used, the same deterministic per-family variant transforms, the same
max_new_tokens (512), GREEDY decoding on both arms (HF do_sample=False; served
temperature=0), and the same family runners (``get_attack(fam)`` over
``run_static``). Only the defense's generate PATH differs:

  HF arm      build_defense(mode='prompted', system=SI, positions='all',
              batch_size=8) — the reference's exact config (engine_stamp
              batch_size=8, target_engine=hf). Re-prefills the ~5.9k SI on every
              batch; no KV reuse. This arm reproduces the matrix_fill numbers
              (pasttense 837.2s / codeattack 403.6s / renellm 248.5s on
              prompt_con) as its own sanity check.

  served arm  a ServedStaticDefense whose .generate(prompts) fans the whole
              family out CONCURRENTLY (max_concurrency=64) through the SAME
              RoutingProxy + prefix-caching vLLM server the served BoN path uses
              — never a bare vLLM URL (the proxy owns per-cell cache_salt
              isolation and SI-prefix verification). The cell is UNHOOKED, so
              its salt is cell-stable and the ~5.9k SI prefix caches across
              every request in the cell (warmed once before the timed loop).

assert_steering_bites is DELIBERATELY NOT called: a prompted cell is unhooked
(base + system prompt, no steering object), so there is no arm to probe and no
bites verdict to report (the proxy's per-request SI-prefix hash check is the
analogous guard). Report is per-family (HF s/gen, served s/gen, ratio) plus the
summed per-condition total each way. grok420 is reported on its own line.

Serving stack mirrors petri_bon_prompted.py verbatim: NousResearch@d10aef7,
interleaved, prefix caching ON. The GEOMETRY is the ``families_prompted``
entry of the per-eval table (sisteer.capeval.serving.geometry — the table is
the only place the numbers live; the entry is MEASURED over the prompted
long-SI arms at the 512 cap, docs/served_geometry_p99.md), with the
WAVE_MAX_MODEL_LEN / WAVE_MAX_NUM_SEQS env overrides winning. The old 10240
(not BoN's 8192) rationale — the static families' variants are longer than
BoN's goal+noise augmentations (codeattack code frames, grok420 templates,
renellm nested scenarios), and the ~6k SI + longest variant + 512 output must
fit — is what the measurement now sizes exactly: p99 prompt+completion 7,284,
max 7,305.
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
# The PETRI serving-cell name for the long-SI prompted cell. defense_cells.py
# documents the alias: prompted__constitution <-> the sweep's
# prompted__si_constitution (same base weights + constitution SI, no object).
PROMPTED = "prompted__constitution"
PVMODEL = "petri__" + PROMPTED

# The composite's five non-BoN members (composite ASR = mean over these + BoN);
# grok420 is measured too but reported SEPARATELY (not in the composite).
COMPOSITE_FAMILIES = ("artprompt", "codeattack", "drattack", "pasttense", "renellm")
EXTRA_FAMILIES = ("grok420",)
ALL_FAMILIES = COMPOSITE_FAMILIES + EXTRA_FAMILIES

MAX_NEW_TOKENS = 512
HF_BATCH_SIZE = 8            # the reference engine_stamp value (target_engine=hf)
# GEOMETRY from the per-eval table (module docstring); WAVE_* env wins.
from sisteer.capeval.serving.geometry import geometry_for as _geometry_for
from sisteer.capeval.serving.geometry import model_key_for as _model_key_for
from sisteer.capeval.serving.geometry import resolve_geometry as _resolve_geometry

GEOMETRY = _geometry_for("families_prompted")
# THE PER-MODEL BUDGET (2026-09-04): the served model's key (WAVE_MODEL_KEY, else
# derived from the pinned NOUS id) sizes the default S from its MEASURED budget
# (ModelBudget.seqs_at, hybrid cap applied); WAVE_* overrides still win and the
# budget guard is the model's.
MODEL_KEY = os.environ.get("WAVE_MODEL_KEY", "") or _model_key_for(NOUS)
_DEFAULT_GEOMETRY = _resolve_geometry("families_prompted", model_key=MODEL_KEY)
MAX_MODEL_LEN = int(os.environ.get("WAVE_MAX_MODEL_LEN", str(_DEFAULT_GEOMETRY.max_model_len)))
MAX_NUM_SEQS = int(os.environ.get("WAVE_MAX_NUM_SEQS", str(_DEFAULT_GEOMETRY.max_num_seqs)))
SERVED_MAX_CONCURRENCY = int(os.environ.get("WAVE_MAX_CONCURRENCY",
                                            str(MAX_NUM_SEQS)))  # == server max_num_seqs


def resolved_geometry():
    """The launch geometry as a ``ResolvedGeometry`` (table default vs WAVE_*
    override per factor, budget position, basis) for the log + record. A
    WAVE_* override that breaks the identity or the budget REFUSES unless
    ``WAVE_ALLOW_OVER_BUDGET=1`` (same rule as the served-families wave)."""
    from sisteer.capeval.serving.geometry import resolve_geometry
    return resolve_geometry(
        "families_prompted",
        max_model_len=MAX_MODEL_LEN if "WAVE_MAX_MODEL_LEN" in os.environ else None,
        max_num_seqs=MAX_NUM_SEQS if "WAVE_MAX_NUM_SEQS" in os.environ else None,
        allow_over_budget=os.environ.get("WAVE_ALLOW_OVER_BUDGET", "").strip()
        not in ("", "0"),
        strict_identity=True, model_key=MODEL_KEY)


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


def _load_manifest_goals(cfg, goal_ids_path):
    """The EXACT pinned held-out set the matrix_fill reference used: filter the
    FULL 300-goal universe (HarmBench standard test+val, copyright excluded, +
    JBB harmful) to the manifest's goal_ids, in manifest order. Refuses on any
    missing id (never silently a different item set). Same logic as the sweep's
    _load_manifest_goals/_pin_goal_set on the matrixfill lineage."""
    from sisteer.eval.jailbreak_data import load_harmbench, load_jbb
    hb = cfg["datasets"]["harmbench"]; jbb = cfg["datasets"]["jbb"]
    goals = []
    for split in ("test", "val"):
        goals += load_harmbench(split=split, revision=hb["revision"],
                                functional=hb["eval_functional"],
                                exclude_copyright=hb["exclude_copyright"])
    goals += load_jbb("harmful", revision=jbb["revision"])
    wanted = json.loads(Path(goal_ids_path).read_text())
    ids = wanted["goal_ids"] if isinstance(wanted, dict) else list(wanted)
    by = {g.goal_id: g for g in goals}
    missing = [i for i in ids if i not in by]
    if missing:
        raise SystemExit(
            f"[families] goal manifest {goal_ids_path}: {len(missing)} of "
            f"{len(ids)} ids absent from the pooled set (first: {missing[:5]}) "
            "— refusing rather than running a different item set.")
    return [by[i] for i in ids]


# ServedStaticDefense now lives in sisteer.eval.attacks.bon_served (the ONE
# implementation the sweep driver shares). This timing driver builds it in
# record-mode (raise_on_error=False) so one overflow cannot nuke a whole
# family's single chunk during a measurement — it surfaces n_errors > 0 itself
# (below) rather than reporting a skewed number as clean. The sweep uses the
# default raise-on-error mode instead (a short/errored store must fail loudly).


def _count_rows(responses_path: Path) -> int:
    if not responses_path.exists():
        return 0
    n = 0
    for line in responses_path.read_text().splitlines():
        if line.strip():
            n += 1
    return n


def _run_family(defense, fam, goals, out_dir):
    """Run one family through get_attack over the given defense, timing it and
    counting the rows actually generated (renellm covers only the frozen
    corpus's 259 ids, so it runs a strict subset of the 112 goals)."""
    from sisteer.eval.attacks import get_attack
    fdir = out_dir / fam
    fdir.mkdir(parents=True, exist_ok=True)
    t = time.time()
    summary = get_attack(fam)(defense, goals, out_dir=fdir,
                              max_new_tokens=MAX_NEW_TOKENS, seed=0)
    sec = time.time() - t
    n_rows = _count_rows(fdir / "responses.jsonl")
    rec = {"seconds": sec, "n_gen": n_rows,
           "per_gen_s": (sec / n_rows) if n_rows else None,
           "n_goals_run": summary.get("n_goals_run") if isinstance(summary, dict) else None}
    if isinstance(summary, dict) and summary.get("skipped"):
        # a family that skipped (e.g. renellm with no frozen corpus) generated nothing; the
        # record carries it, and main() refuses to call the run completed
        rec["skipped"] = True
        rec["skip_reason"] = summary.get("reason")
    return rec


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal-ids", required=True,
                    help="the pinned held-out goal_manifest.json (112 ids)")
    ap.add_argument("--defenses-dir",
                    default=str(SHARED / "exp/swp/matrix_fill/defenses"),
                    help="two-SI cells dir (prompted uses no object; needed so "
                         "the cell source builds)")
    ap.add_argument("--out", default=str(SHARED / "petri/serving/families_prompted"))
    ap.add_argument("--families", default=",".join(ALL_FAMILIES))
    args = ap.parse_args()

    families = tuple(f.strip() for f in args.families.split(",") if f.strip())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    DDIR = Path(args.defenses_dir)

    geom = resolved_geometry()
    print(geom.describe(), flush=True)
    rec = {"status": "started", "cell": PROMPTED, "families": list(families),
           "max_new_tokens": MAX_NEW_TOKENS, "hf_batch_size": HF_BATCH_SIZE,
           "served_max_concurrency": SERVED_MAX_CONCURRENCY,
           "max_model_len": MAX_MODEL_LEN, "max_num_seqs": MAX_NUM_SEQS,
           "model_key": MODEL_KEY, "geometry": geom.as_dict(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "node": socket.gethostname(),
           "heldout_corpus_dir": os.environ.get("SISTEER_HELDOUT_CORPUS_DIR")}

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
    from sisteer.chat_markers import generation_header_ids

    cfg = jr._load_cfg("configs/two_si.yaml")
    goals = _load_manifest_goals(cfg, args.goal_ids)
    rec["n_goals"] = len(goals)
    print(f"[families] pinned goal set: {len(goals)} ids from {args.goal_ids}")

    tok = AutoTokenizer.from_pretrained(NOUS, revision=NREV)
    # STAGED ARTIFACTS (review, third pass): the chat template, lens registry
    # and routing table are generated into a temporary dir INSIDE the out dir
    # and installed by rename only AFTER the run identity is verified — a
    # refused resume must leave the completed run's artifacts byte-identical.
    from sisteer.capeval.serving import run_identity as ri
    # THE OUT-DIR LOCK (review, fourth pass): exclusive ownership of this out
    # dir from before the artifacts are staged / the sidecar is written until
    # the process ends (generation + the final verification) — a concurrent
    # launch on the same out dir refuses with exit 2 naming the holder instead
    # of racing the sidecar or resuming the same stores.
    import atexit
    out_lock = ri.OutDirLock(out, phase="families").acquire()
    atexit.register(out_lock.release)
    stage = ri.stage_dir(out)
    tpath = stage / "chat_template.jinja"; tpath.write_bytes(tok.chat_template.encode())
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
    print(f"[families] SI prefix: {len(si_text)} chars")
    bf = {k: f"# {k}\n".encode() for k in cm.petri_expected_behavior_keys(spec)}
    cells = dc.enumerate_cells(spec, src, behavior_files=bf,
                              prompt_files={"auditor": b"# a\n", "judge": b"# j\n"})
    reg_dir = stage / "lens_registry"
    export_worker_registry(cells, src, reg_dir,
                           assistant_header_ids=tuple(generation_header_ids(tok)),
                           zero_op_hidden_size=4096)
    tbl = stage / "routing_table.json"
    export_routing_table(cells, src, tbl, base_model=NOUS, serving_mode="interleaved")

    # THE RUN IDENTITY (reviews R1 + S1): written on a fresh out dir, VERIFIED on
    # a reused one BEFORE any family resumes — the served cells' ACTUAL
    # intervention hashes (the routing table just exported) and every replayed
    # frozen corpus's content hash are part of it, so a refit / promotion in
    # place or a corpus rebuilt in place is refused, never relabeled; the
    # geometry and run stamps are exported so every served row carries them.
    from sisteer.capeval.serving import run_identity as ri
    try:
        identity = ri.RunIdentity(
            model=NOUS, revision=NREV, tree_sha=ri.tree_sha(REPO_ROOT), model_key=MODEL_KEY,
            max_model_len=MAX_MODEL_LEN, max_num_seqs=MAX_NUM_SEQS,
            client_concurrency=SERVED_MAX_CONCURRENCY,
            budget=geom.budget_tokens, budget_source=geom.budget_source,
            serving_versions=ri.serving_versions(),
            sampling={"max_new_tokens": MAX_NEW_TOKENS, "seed": 0, "hf_batch_size": HF_BATCH_SIZE},
            input_fingerprints={"goal_ids_sha16": ri.file_sha16(args.goal_ids),
                                "families": list(families), "cell": PROMPTED,
                                "defenses_dir": str(DDIR),
                                "config_sha16": ri.file_sha16(REPO_ROOT / "configs/two_si.yaml"),
                                "intervention_hashes": ri.intervention_hashes(tbl, ["none", PROMPTED]),
                                "frozen_corpora": ri.frozen_corpus_hashes(families),
                                "corpus_dir": str(ri.default_corpus_dir())})
        sidecar = ri.write_or_verify(out, identity)
    except ri.RunIdentityMismatch as e:   # a corpus/manifest disagreement included
        ri.discard_stage(stage)            # the out dir's artifacts stay exactly as they were
        out_lock.release()
        raise SystemExit(f"[families] {e}")
    # identity verified: install the generated artifacts atomically (rename)
    rec["artifacts_install"] = ri.install_staged(stage, out, ["chat_template.jinja", "lens_registry", "routing_table.json"])
    tpath = out / "chat_template.jinja"
    reg_dir = out / "lens_registry"
    tbl = out / "routing_table.json"
    rec["run_identity"] = identity.as_dict()
    rec["run_identity_file"] = str(sidecar)
    rec["geometry_id"] = ri.stamp_env(identity)
    rec["run_id"] = identity.run_id
    print(f"[families] run identity {identity.run_id} (geometry {identity.geometry_id}; "
          f"interventions {identity.input_fingerprints['intervention_hashes']}; frozen corpora "
          f"{identity.input_fingerprints['frozen_corpora']}) at {sidecar}", flush=True)

    # ---- SERVED arm (server first; torn down before the HF model loads) ----
    served_results = {}
    server = None; proxy_loop = None
    try:
        launch = vs.build_server_launch(spec, port=free_port(), lens_registry_dir=reg_dir,
                                        chat_template_path=tpath,
                                        max_model_len=MAX_MODEL_LEN,
                                        max_num_seqs=MAX_NUM_SEQS)
        rec["prefix_caching_composed"] = launch.prefix_caching
        server = vs.launch_server(launch, log_path=out / "vllm_server.log",
                                  ready_timeout_s=1800, install_timeout_s=900)
        proxy = RoutingProxy(load_routing_table(tbl)["routes"],
                             HTTPBackend(launch.base_url, api_key=launch.api_key),
                             base_model=NOUS, serving_mode="interleaved",
                             prompted_prefixes=load_routing_table(tbl)["prompted_prefixes"],
                             exact_replay_strategy=None)
        pport = free_port()
        proxy_loop = _proxy_thread(make_app(proxy), "127.0.0.1", pport)
        proxy_url = f"http://127.0.0.1:{pport}/v1"
        client = ProxyClientEngine(proxy_url, identity=cm.serving_identity(
            "proxy_vllm", "passthrough", "interleaved"))
        client.load(require_virtual_models=[PVMODEL])

        # Warm the ~5.9k SI prefix ONCE (cell-stable salt -> caches for every
        # family in the cell), so the timed loop measures steady-state cached
        # throughput, not a cold-prefill stampede on the first family.
        client.chat_completions({"model": PVMODEL,
            "messages": [{"role": "system", "content": si_text},
                         {"role": "user", "content": "Say OK."}],
            "temperature": 0.0, "max_tokens": 4, "seed": 0})

        import aiohttp
        # ONE event loop + ONE aiohttp session for the whole served arm (see
        # ServedStaticDefense): reused across every family so nothing touches a
        # session bound to an already-closed loop.
        served_loop = asyncio.new_event_loop()
        sess = {"s": None}
        async def _post(path, payload):
            if sess["s"] is None:
                sess["s"] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600))
            async with sess["s"].post(proxy_url + path, json=dict(payload),
                                      headers={"Authorization": "Bearer petri-proxy"}) as r:
                b = await r.text()
                if r.status != 200:
                    raise RuntimeError(f"proxy {r.status}: {b[:400]}")
                return json.loads(b)

        async def achat(payload):
            return await _post("/chat/completions", payload)

        async def atokenize(payload):
            # the proxy's read-only /tokenize (served_window floor-only resolution)
            return await _post("/tokenize", payload)

        from sisteer.eval.attacks.bon_served import (ServedDefense,
                                                     ServedStaticDefense)
        sdir = out / "served"
        try:
            for fam in families:
                sd = ServedDefense(achat, virtual_model=PVMODEL,
                                   cell_id=PROMPTED, system=si_text,
                                   max_model_len=MAX_MODEL_LEN, atokenize=atokenize)
                # record-mode (raise_on_error=False): timing driver, so an
                # overflow is recorded in n_errors (surfaced below), never
                # allowed to nuke a whole family's single chunk mid-measurement.
                defense = ServedStaticDefense(
                    sd, loop=served_loop, max_concurrency=SERVED_MAX_CONCURRENCY,
                    raise_on_error=False)
                r = _run_family(defense, fam, goals, sdir)
                r["n_errors"] = defense.n_errors
                r["max_inflight"] = defense.max_inflight
                served_results[fam] = r
                print(f"[families] served/{fam}: {r['seconds']:.1f}s "
                      f"({r['per_gen_s']} s/gen, n_gen={r['n_gen']}, "
                      f"max_inflight={r['max_inflight']}, n_errors={r['n_errors']})")
        finally:
            if sess["s"] is not None:
                try: served_loop.run_until_complete(sess["s"].close())
                except Exception: pass
            served_loop.close()
        rec["served"] = served_results
        (out / "families_prompted_record.json").write_text(
            json.dumps(rec, indent=2, default=str))
    except BaseException as e:
        rec["status"] = "served_failed"; rec["error"] = f"{type(e).__name__}: {e}"
        rec["served"] = served_results
        (out / "families_prompted_record.json").write_text(
            json.dumps(rec, indent=2, default=str))
        # do NOT re-raise: still run the HF arm so a served hiccup doesn't cost
        # the whole job. The error is recorded and surfaced.
        print(f"[families] SERVED ARM ERROR (continuing to HF): {rec['error']}")
    finally:
        if proxy_loop is not None:
            try: proxy_loop.call_soon_threadsafe(proxy_loop.stop)
            except Exception: pass
        if server is not None:
            try: vs._shutdown_child(server)
            except Exception: pass
            gc.collect()

    # ---- HF arm (base + SI system, re-prefilling the ~5.9k SI every batch) ----
    hf_results = {}
    try:
        import torch  # noqa
        from sisteer.models import load_model_and_tokenizer
        from sisteer.eval.jailbreak import build_defense
        model, tok2 = load_model_and_tokenizer(NOUS, revision=NREV, dtype="bfloat16",
                                               attn_implementation="sdpa")
        hdir = out / "hf"
        for fam in families:
            defense = build_defense(model, tok2, mode="prompted", system=si_text,
                                    positions="all", batch_size=HF_BATCH_SIZE)
            r = _run_family(defense, fam, goals, hdir)
            hf_results[fam] = r
            print(f"[families] hf/{fam}: {r['seconds']:.1f}s "
                  f"({r['per_gen_s']} s/gen, n_gen={r['n_gen']})")
        rec["hf"] = hf_results
    except Exception as e:
        rec["hf"] = {"error": f"{type(e).__name__}: {e}", **hf_results}
        print(f"[families] HF ARM ERROR: {type(e).__name__}: {e}")

    # ---- ratios + per-condition totals ----
    ratios = {}
    for fam in families:
        s = served_results.get(fam, {}).get("seconds")
        h = hf_results.get(fam, {}).get("seconds")
        if s and h:
            ratios[fam] = {"hf_over_served_walltime": h / s,
                           "hf_per_gen_s": hf_results[fam].get("per_gen_s"),
                           "served_per_gen_s": served_results[fam].get("per_gen_s")}
    rec["ratios"] = ratios

    def _sum(engine_results, fams):
        tot = 0.0; ok = True
        for f in fams:
            v = engine_results.get(f, {}).get("seconds")
            if v is None:
                ok = False
            else:
                tot += v
        return tot, ok
    comp_hf, comp_hf_ok = _sum(hf_results, COMPOSITE_FAMILIES)
    comp_sv, comp_sv_ok = _sum(served_results, COMPOSITE_FAMILIES)
    rec["condition_totals_composite_nonbon"] = {
        "hf_seconds": comp_hf, "hf_complete": comp_hf_ok,
        "served_seconds": comp_sv, "served_complete": comp_sv_ok,
        "hf_over_served": (comp_hf / comp_sv) if comp_sv else None,
        "families": list(COMPOSITE_FAMILIES),
        "note": "sum over the FIVE non-BoN composite families; BoN measured "
                "separately (HF 17804.1s / served 0.0344 s/gen on this cell)."}

    skipped = sorted({f for res in (hf_results, served_results)
                      for f, r in res.items() if isinstance(r, dict) and r.get("skipped")})
    rec["skipped_families"] = skipped
    rec["status"] = "incomplete_skipped" if skipped else "completed"
    (out / "families_prompted_record.json").write_text(
        json.dumps(rec, indent=2, default=str))
    if skipped:
        reasons = {f: (hf_results.get(f) or served_results.get(f) or {}).get("skip_reason") for f in skipped}
        raise SystemExit(f"REFUSED: requested families generated nothing: {reasons}; the record says "
                         f"status=incomplete_skipped ({out / 'families_prompted_record.json'})")

    # ---- human-readable table ----
    def _fmt(x, spec):
        return format(x, spec) if isinstance(x, (int, float)) else "NA"
    print("\n===== MEASURED: non-BoN families on prompted__si_constitution "
          "(long SI ~5.9k tok) =====")
    header = ("family".ljust(12) + "HF s/gen".rjust(10) + "served s/gen".rjust(14)
              + "ratio".rjust(9) + "n_gen".rjust(8) + "HF wall".rjust(10)
              + "served wall".rjust(12))
    print(header)
    for fam in families:
        h = hf_results.get(fam, {}); s = served_results.get(fam, {})
        hpg = h.get("per_gen_s"); spg = s.get("per_gen_s")
        hw = h.get("seconds"); sw = s.get("seconds")
        ratio = (hw / sw) if (hw and sw) else None
        tag = "  (composite)" if fam in COMPOSITE_FAMILIES else "  (SEPARATE)"
        row = (fam.ljust(12)
               + _fmt(hpg, ".4f").rjust(10)
               + _fmt(spg, ".4f").rjust(14)
               + (_fmt(ratio, ".1f") + ("x" if isinstance(ratio, float) else "")).rjust(9)
               + str(s.get("n_gen", "?")).rjust(8)
               + _fmt(hw, ".1f").rjust(10)
               + _fmt(sw, ".1f").rjust(12)
               + tag)
        print(row)
    if comp_sv:
        print(f"\nCOMPOSITE per-condition total (5 non-BoN families): "
              f"HF {comp_hf:.1f}s vs served {comp_sv:.1f}s -> {comp_hf/comp_sv:.1f}x")
    else:
        print("\nCOMPOSITE per-condition total: served incomplete")
    print(json.dumps({"served": served_results, "hf": hf_results,
                      "condition_totals": rec["condition_totals_composite_nonbon"]},
                     indent=2, default=str))
    return 0


if __name__ == "__main__":
    _site.require_expanded(str(SHARED), key="the cluster's shared root (SISTEER_SHARED_ROOT)")
    sys.exit(main())
