#!/usr/bin/env python
"""Integration validation: unified_sweep_run's FULLY SERVED path on ONE long-SI
condition, end-to-end on hardware.

The BoN-only served path already had an integration smoke
(petri_bon_smoke_integration.py). This one proves the thing the families wiring
added: with ``--gen-engine served`` the WHOLE fixed-prompt leg — BoN AND the
non-BoN static/held-out families — generates through the routing proxy over the
vLLM server, so a served sweep is fully served, not BoN-only.

Cell: ``prompted__si_constitution`` (harness ``prompted__constitution``, the
~5.9k-token SI) — an UNHOOKED baseline, so prefix caching applies and NO
steering-bites gate runs (there is no arm); the salt-isolation gate still runs.
Families: bon + pasttense (static) and artprompt, codeattack, drattack, renellm
(held-out) — the reference driver's measured composite members, minus grok420
(not on this branch's sweep sets). Full confirmatory goal set, bon n_max 64
(the reference's budget) so the wall-clock is comparable to the measured
~5.4 min for a fully served long-SI condition.

One GPU: with the fully-served fix, unified_sweep_run does NOT load an HF model
on a fully-served run (it is dead weight — every family serves through the
proxy), so the vLLM server has the card to itself (gpu_memory_utilization=0.9).

PASS = the sweep exits 0 (every served family's assert_static_store_complete
passed, so per-family row counts equal the deterministic HF-reference counts by
construction) AND every family's responses carry engine=vllm_served AND every
family's engine sidecar records the proxy_vllm serving identity AND there is not
one ``__SERVED_ERROR__`` placeholder anywhere. The per-family row counts and the
summed served wall-clock are recorded for the report.
"""
from __future__ import annotations
import asyncio
import dataclasses
import json
import os
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# EVERY per-condition value below is env-overridable so one driver can run a
# WAVE (2026-08-28). Defaults are the original single-cell integration run, so
# an un-parameterised invocation reproduces it byte-for-byte.
#
# WAVE_MASK is the one that can silently destroy a run: these objects record
# their fit regime in the blob (`positions`), and serving a RESPONSE-fit psr
# object at `all` is a measured model-destroyer -- 65.7%/26.7% degenerate
# against a 0.5% undefended baseline. The default stays "all" because the
# reference cell is `prompted__*` (no object, nothing to mis-serve); every
# fitted cell MUST pass WAVE_MASK=generated, which unified_sweep_run maps to
# positions="response" (assistant-header tail + all generated, so the forward
# emitting token 1 is steered).
_E = os.environ.get

NOUS = _E("WAVE_MODEL", "NousResearch/Meta-Llama-3.1-8B-Instruct")
NREV = _E("WAVE_REV", "d10aef7999a2b5ba950ab3974312feeedbfe0b77")
from sisteer import site as _site  # noqa: E402
#: the cluster's shared root (sisteer.site: the job's SISTEER_SHARED_ROOT, else the site config's cluster block). Where neither
#: sets it (an import off the cluster) it keeps its placeholder form, and a run refuses by name before it reads or writes there.
try:
    SHARED = _site.expand_path("$SHARED")
except (_site.PlaceholderError, _site.SiteConfigError):
    SHARED = Path("$SHARED")
# prompted uses no object, but the cell SOURCE still builds against a defenses
# dir (the matrix_fill campaign dir the reference used).
DDIR = Path(_E("WAVE_DDIR", str(SHARED / "exp/swp/matrix_fill/defenses")))
TT_CELL = _E("WAVE_CELL", "prompted__si_constitution")
PROMPTED = _E("WAVE_PROMPTED", "prompted__constitution")   # the harness id
STATIC = _E("WAVE_STATIC", "bon,pasttense")
HELD_OUT = _E("WAVE_HELD_OUT", "artprompt,codeattack,drattack,renellm")
MASK = _E("WAVE_MASK", "all")
CONFIG = _E("WAVE_CONFIG", "configs/two_si.yaml")
# GEOMETRY: the defaults are the ``families_served`` entry of the per-eval
# table (sisteer.capeval.serving.geometry — no served eval inherits a
# window by accident; the table is the only place the numbers live, and
# the entry is MEASURED over the single-shot pools at the 4,096 cap,
# docs/served_geometry_p99.md). The WAVE_* env overrides still win
# (qwen_wave.sbatch sets 16384 x 32 for the hybrid). The client fan-out
# follows the launched width unless WAVE_MAX_CONCURRENCY says otherwise.
from sisteer.capeval.serving.geometry import geometry_for as _geometry_for
from sisteer.capeval.serving.geometry import model_key_for as _model_key_for
from sisteer.capeval.serving.geometry import resolve_geometry as _resolve_geometry

GEOMETRY = _geometry_for("families_served")
# THE PER-MODEL BUDGET (2026-09-04): the wave's model key — WAVE_MODEL_KEY, else
# derived from WAVE_MODEL's HF id — sizes the default S from that model's
# MEASURED batched-token budget (geometry.BUDGET_BY_MODEL, ModelBudget.seqs_at
# with the hybrid per-slot cap); the WAVE_* overrides still win and the budget
# guard is held to the model's budget. An unknown model id falls to the
# Llama-certified default (model_key_for warns).
MODEL_KEY = _E("WAVE_MODEL_KEY", "") or _model_key_for(NOUS)
_DEFAULT_GEOMETRY = _resolve_geometry("families_served", model_key=MODEL_KEY)
MAX_MODEL_LEN = int(_E("WAVE_MAX_MODEL_LEN", str(_DEFAULT_GEOMETRY.max_model_len)))
MAX_NUM_SEQS = int(_E("WAVE_MAX_NUM_SEQS", str(_DEFAULT_GEOMETRY.max_num_seqs)))
BON_N_MAX = int(_E("WAVE_BON_N_MAX", "64"))
MAX_CONCURRENCY = int(_E("WAVE_MAX_CONCURRENCY", str(MAX_NUM_SEQS)))


def resolved_geometry():
    """The wave's launch geometry as a ``ResolvedGeometry`` — table default vs
    WAVE_* override per factor, budget position, basis — for the startup log
    and the run record. A WAVE_* override that breaks the identity or the
    certified budget REFUSES (GeometryOverBudget / GeometryIdentityBroken)
    unless ``WAVE_ALLOW_OVER_BUDGET=1`` is exported alongside it — the second
    adversarial review found qwen_wave.sbatch exporting a literal 16384 x 32
    (524,288 tokens, 33% over the footprint) through this seam unchecked."""
    from sisteer.capeval.serving.geometry import resolve_geometry
    return resolve_geometry(
        "families_served",
        max_model_len=MAX_MODEL_LEN if "WAVE_MAX_MODEL_LEN" in os.environ else None,
        max_num_seqs=MAX_NUM_SEQS if "WAVE_MAX_NUM_SEQS" in os.environ else None,
        allow_over_budget=os.environ.get("WAVE_ALLOW_OVER_BUDGET", "").strip()
        not in ("", "0"),
        strict_identity=True, model_key=MODEL_KEY)

if MASK not in ("all", "generated"):
    raise SystemExit(f"WAVE_MASK must be 'all' or 'generated', got {MASK!r}")
# The no-chunking identity is a CORRECTNESS requirement (the plan lattice
# refuses chunk shapes), so it is asserted here rather than assumed.
_BUDGET = MAX_NUM_SEQS * MAX_MODEL_LEN


def run_out_dir() -> Path:
    # WAVE_OUT_DIR lets a resubmit RESUME into a prior job's out dir: BoN
    # (_load_covered, keyed on goal/sample/trial) and the static/held-out
    # families (static_common load_covered_goals + filter_responses_to_covered,
    # which also drops a SIGKILL-torn tail) both resume per-unit, so a reused
    # dir skips the ~16.5k already-generated BoN rows and the covered family
    # goals rather than regenerating them. Absent it, one fresh dir per job.
    override = os.environ.get("WAVE_OUT_DIR")
    if override:
        return Path(override)
    tag = os.environ.get("SLURM_JOB_ID") or time.strftime(
        "%Y%m%dT%H%M%SZ", time.gmtime())
    return SHARED / "petri" / "serving" / "families_served_integration" / f"run_{tag}"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _start_proxy_thread(app, host, port):
    from aiohttp import web
    ready = threading.Event()
    loop = asyncio.new_event_loop()

    def _run():
        asyncio.set_event_loop(loop)
        runner = web.AppRunner(app)
        loop.run_until_complete(runner.setup())
        loop.run_until_complete(web.TCPSite(runner, host, port).start())
        ready.set()
        loop.run_forever()
    threading.Thread(target=_run, name="proxy", daemon=True).start()
    if not ready.wait(timeout=30):
        raise RuntimeError("proxy thread not ready")
    return loop


def _count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def thinking_render_verdict(served: int, n_off: int, n_on):
    """None when the served prompt length proves the fit's render regime;
    else the problem, named. Only DISCRIMINATING templates assert: when
    the template has no thinking switch (or the two renders coincide)
    there is nothing to prove and nothing that can false-fail a dense
    wave. For a reasoning template (Qwen3.5) this is the end-to-end HTTP
    half of the thinking-off proof: hash equality of the template file
    is necessary but not sufficient — only the served token count shows
    which regime the server actually rendered."""
    if n_on is None or n_on == n_off:
        return None
    if served == n_off:
        return None
    if served == n_on:
        return (f"served prompt_tokens {served} equals the thinking-ON "
                f"render ({n_on}), not the fit's thinking-OFF render "
                f"({n_off}) — the server is rendering thinking-ON and every "
                "response mask would sit at the wrong boundary")
    return (f"served prompt_tokens {served} matches NEITHER local render "
            f"(off={n_off}, on={n_on}) — the served template/render "
            "disagrees with the pinned local one")


def _served_rows_and_errors(path: Path):
    """(rows, served_tagged_rows, error_placeholders) for a responses file."""
    rows = served = errs = 0
    if not path.exists():
        return 0, 0, 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows += 1
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("engine") == "vllm_served":
            served += 1
        if "__SERVED_ERROR__" in str(rec.get("response", "")):
            errs += 1
    return rows, served, errs


def main():
    out = run_out_dir()
    out.mkdir(parents=True, exist_ok=True)
    print(f"[fam-served] out dir (fresh per-run): {out}")
    geom = resolved_geometry()
    print(geom.describe(), flush=True)
    rec = {"status": "started", "cell": TT_CELL,
           "static": STATIC, "held_out": HELD_OUT,
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
           "node": socket.gethostname(),
           "heldout_corpus_dir": os.environ.get("SISTEER_HELDOUT_CORPUS_DIR"),
           "model_key": MODEL_KEY,
           "geometry": geom.as_dict(), "max_concurrency": MAX_CONCURRENCY}

    from transformers import AutoTokenizer
    from sisteer.capeval import manifest as cm
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.petri.registry_adapter import traintest_cell_source
    from sisteer.capeval.serving import vllm_server as vs
    from sisteer.capeval.serving.lens_steer import export_worker_registry
    from sisteer.capeval.serving.proxy import (HTTPBackend, RoutingProxy,
                                               export_routing_table,
                                               load_routing_table, make_app)
    from sisteer.chat_markers import generation_header_ids

    from transformers import AutoConfig

    from sisteer.models import pin_reasoning_template_off
    tok = AutoTokenizer.from_pretrained(NOUS, revision=NREV)
    # reasoning templates (Qwen3.5) render thinking-OFF at EVERY seam —
    # the registry's assistant header ids derive from this tokenizer, and
    # unpinned they would be the thinking-ON header, which matches
    # neither the fits (H=7 on the closed <think></think>) nor the
    # served render (build_server_launch pins the engine OFF).
    rec["thinking_pin_applied"] = pin_reasoning_template_off(tok)
    # STAGED ARTIFACTS (review, third pass): generated into a temporary dir
    # INSIDE the out dir, installed by rename only AFTER the run identity is
    # verified — a refused WAVE_OUT_DIR resume leaves the completed run's
    # chat template / registry / routing table byte-identical.
    from sisteer.capeval.serving import run_identity as ri
    # THE OUT-DIR LOCK (review, fourth pass): exclusive ownership of this out
    # dir from before the artifacts are staged / the sidecar is written until
    # the process ends (generation + the final verification) — a concurrent
    # launch on the same out dir refuses with exit 2 naming the holder instead
    # of racing the sidecar or resuming the same stores.
    import atexit
    out_lock = ri.OutDirLock(out, phase="fam-served").acquire()
    atexit.register(out_lock.release)
    stage = ri.stage_dir(out)
    tpath = stage / "chat_template.jinja"
    tpath.write_bytes(tok.chat_template.encode())
    tsha = cm.sha256_hex16(tok.chat_template.encode())
    # the zero-op (noop lens) cell must carry the SERVED model's hidden
    # size — 4096 was a llama hardcode that silently fit the 9B and would
    # mis-shape the 4B (hidden 2560).
    _cfg = AutoConfig.from_pretrained(NOUS, revision=NREV)
    hidden = int(getattr(getattr(_cfg, "text_config", _cfg), "hidden_size"))
    rec["zero_op_hidden_size"] = hidden

    spec0 = cm.load_petri_spec(REPO_ROOT / "configs/capeval/manifests/petri_twosi.yaml")
    spec = dataclasses.replace(
        spec0, target_model=NOUS, target_revision=NREV,
        serving_mode="interleaved", cells=["none", PROMPTED],
        serving_launch_flags=[f for f in spec0.serving_launch_flags
                              if f != "--enable-lora"],
        scenario_bundle_hash="ab" * 8, serving_impl_commit="fedcba9876" * 4,
        chat_template_hash=tsha)
    # the SAME per-condition config governs the cell source and the sweep
    # (WAVE_CONFIG): the default two_si.yaml pins the llama model, and the
    # adapter refuses a config/spec model disagreement — so a qwen wave
    # must name its own config here, not just at the sweep.
    src = traintest_cell_source(spec, repo_root=REPO_ROOT,
                                config_path=REPO_ROOT / CONFIG,
                                defenses_dir=DDIR)
    bf = {k: f"# {k}\n".encode() for k in cm.petri_expected_behavior_keys(spec)}
    cells = dc.enumerate_cells(spec, src, behavior_files=bf,
                              prompt_files={"auditor": b"# a\n", "judge": b"# j\n"})
    if cells.absent:
        raise RuntimeError(f"absent: {[a.cell_id for a in cells.absent]}")
    reg_dir = stage / "lens_registry"
    export_worker_registry(
        cells, src, reg_dir,
        assistant_header_ids=tuple(generation_header_ids(tok)),
        zero_op_hidden_size=hidden)
    table_path = stage / "routing_table.json"
    export_routing_table(cells, src, table_path, base_model=NOUS,
                         serving_mode="interleaved")
    print("[fam-served] registry + routing table exported (staged)")

    # THE RUN IDENTITY (reviews R1 + S1): a WAVE_OUT_DIR resume is honored only
    # when the out dir's recorded identity equals this run's — model, revision,
    # tree, model key, L x S, client fan-out, budget, serving stack, sampling,
    # the config / goal inputs, the served cells' ACTUAL intervention hashes
    # (from the routing table just exported: a refit in place under the same
    # defenses dir is a different run) and every replayed frozen corpus's
    # content hash (a corpus rebuilt in place is a different run). A mismatch,
    # or a pre-sidecar dir holding generations, is refused BEFORE the server
    # comes up and before unified_sweep_run resumes a single unit. The geometry
    # and run stamps are exported so every served row (BoN + static) carries
    # them.
    from sisteer.capeval.serving import run_identity as ri
    try:
        families_run = [f for f in (STATIC + "," + HELD_OUT).split(",") if f]
        identity = ri.RunIdentity(
            model=NOUS, revision=NREV, tree_sha=ri.tree_sha(REPO_ROOT), model_key=MODEL_KEY,
            max_model_len=MAX_MODEL_LEN, max_num_seqs=MAX_NUM_SEQS,
            client_concurrency=MAX_CONCURRENCY,
            budget=geom.budget_tokens, budget_source=geom.budget_source,
            serving_versions=ri.serving_versions(),
            sampling={"bon_n_max": BON_N_MAX, "mask": MASK},
            input_fingerprints={"config": CONFIG, "config_sha16": ri.file_sha16(REPO_ROOT / CONFIG),
                                "static": STATIC, "held_out": HELD_OUT, "cell": TT_CELL,
                                "harness_cell": PROMPTED, "defenses_dir": str(DDIR),
                                "intervention_hashes": ri.intervention_hashes(table_path,
                                                                              ["none", PROMPTED]),
                                "frozen_corpora": ri.frozen_corpus_hashes(families_run),
                                "corpus_dir": str(ri.default_corpus_dir())})
        sidecar = ri.write_or_verify(out, identity)
    except ri.RunIdentityMismatch as e:   # a corpus/manifest disagreement included
        ri.discard_stage(stage)            # the out dir's artifacts stay exactly as they were
        out_lock.release()
        raise SystemExit(f"[fam-served] {e}")
    # identity verified: install the generated artifacts atomically (rename)
    rec["artifacts_install"] = ri.install_staged(stage, out, ["chat_template.jinja", "lens_registry", "routing_table.json"])
    tpath = out / "chat_template.jinja"
    reg_dir = out / "lens_registry"
    table_path = out / "routing_table.json"
    rec["run_identity"] = identity.as_dict()
    rec["run_identity_file"] = str(sidecar)
    rec["geometry_id"] = ri.stamp_env(identity)
    rec["run_id"] = identity.run_id
    print(f"[fam-served] run identity {identity.run_id} (geometry {identity.geometry_id}; "
          f"interventions {identity.input_fingerprints['intervention_hashes']}; frozen corpora "
          f"{identity.input_fingerprints['frozen_corpora']}) at {sidecar}", flush=True)

    launch = vs.build_server_launch(
        spec, port=free_port(), lens_registry_dir=reg_dir,
        chat_template_path=tpath, max_model_len=MAX_MODEL_LEN,
        max_num_seqs=MAX_NUM_SEQS)
    # With the fully-served fix, unified_sweep_run does NOT load the HF target
    # (it is dead weight when every family serves through the proxy), so the
    # vLLM server has the card to itself and utilization goes back UP. 0.9 is the
    # normal dedicated-GPU value; the sweep subprocess only holds a tokenizer + a
    # small CUDA context, well within the ~14 GiB left. (0.4 broke it the other
    # way — job 227219 negative KV; 0.8 was a stopgap while the HF model still
    # contended — job 227485 OOM.) Do NOT touch the identity
    # max_num_batched_tokens = max_num_seqs * max_model_len.
    launch = dataclasses.replace(
        launch, cmd=tuple(launch.cmd) + ("--gpu-memory-utilization", "0.9"))
    server = vs.launch_server(launch, log_path=out / "vllm_server.log",
                              ready_timeout_s=1800, install_timeout_s=900)
    proxy_loop = None
    try:
        table = load_routing_table(table_path)
        proxy = RoutingProxy(
            table["routes"],
            HTTPBackend(launch.base_url, api_key=launch.api_key),
            base_model=NOUS, serving_mode="interleaved",
            prompted_prefixes=table["prompted_prefixes"],
            exact_replay_strategy=None)
        proxy_port = free_port()
        proxy_loop = _start_proxy_thread(make_app(proxy), "127.0.0.1", proxy_port)
        proxy_url = f"http://127.0.0.1:{proxy_port}/v1"
        print(f"[fam-served] proxy up at {proxy_url}")

        # THE SERVED-RENDER PROOF (thinking_render_verdict's docstring):
        # one request through the wave's OWN path (proxy -> vLLM chat
        # template application), before any generation rides on it.
        from sisteer.models import apply_chat_template_folding
        probe_msgs = [{"role": "user", "content": "Render-regime probe."}]
        s_off = apply_chat_template_folding(tok, probe_msgs,
                                            add_generation_prompt=True)
        n_off = len(tok(s_off, add_special_tokens=False)["input_ids"])
        n_on = None
        if "enable_thinking" in (getattr(tok, "chat_template", None) or ""):
            s_on = tok.apply_chat_template(
                probe_msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=True)
            n_on = len(tok(s_on, add_special_tokens=False)["input_ids"])
        none_route = next(r for r in table["routes"] if r.cell_id == "none")
        body = json.dumps({"model": none_route.virtual_model,
                           "messages": probe_msgs,
                           "max_tokens": 1, "temperature": 0.0}).encode()
        req = urllib.request.Request(
            f"{proxy_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            served_usage = json.loads(resp.read())["usage"]
        served_pt = int(served_usage["prompt_tokens"])
        rec["thinking_render_proof"] = {
            "served_prompt_tokens": served_pt, "local_off": n_off,
            "local_on": n_on}
        problem = thinking_render_verdict(served_pt, n_off, n_on)
        if problem is not None:
            raise RuntimeError(f"served-render proof FAILED: {problem}")
        print(f"[fam-served] served-render proof: prompt_tokens={served_pt} "
              f"(off={n_off}, on={n_on})")

        tsv = out / "conditions.tsv"
        cols = ("# label\tcell\tcoeff\tmask\tstatic\tadaptive\tbenign\t"
                "target_engine\tattacker_engine\tserving_path\tmirror_override\t"
                "held_out")
        row = (f"fam_val\t{TT_CELL}\t\t{MASK}\t{STATIC}\t-\t-\tvllm\tnone\thf\t\t"
               f"{HELD_OUT}")
        tsv.write_text(cols + "\n" + row + "\n")

        sweep_out = out / "sweep_out"
        cmd = [sys.executable, str(REPO_ROOT / "scripts/unified_sweep_run.py"),
               "generate", "--conditions", str(tsv),
               "--config", CONFIG,
               "--out-root", str(sweep_out), "--defenses-dir", str(DDIR),
               "--task-id", "0", "--gen-engine", "served",
               "--bon-proxy-url", proxy_url,
               "--bon-routing-table", str(table_path),
               "--bon-n-max", str(BON_N_MAX),
               "--bon-max-concurrency", str(MAX_CONCURRENCY),
               # the launched window, so the sweep's served clients clamp
               # max_tokens against the geometry actually serving them
               "--served-max-model-len", str(MAX_MODEL_LEN)]
        rec["sweep_cmd"] = cmd
        print("[fam-served] running unified_sweep_run --gen-engine served ...")
        child_env = dict(os.environ)
        child_env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts"),
             child_env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
        t0 = time.time()
        import subprocess
        # NO hardcoded subprocess cap. A constant here (was 5400s = 90min)
        # competes with the Slurm --time (8h) as a SECOND, invisible deadline,
        # and because BoN runtime scales INVERSELY with defense strength it
        # preferentially killed the slow baselines (none / weak SIs) mid-wave
        # more than 5h before the wall (jobs 272640/272642, 2026-08-30). The
        # single bound is --time, enforced by #SBATCH --signal=B:TERM@120 + the
        # script's cleanup trap; the watcher's stall detection catches a true
        # hang. WAVE_SWEEP_TIMEOUT_S can set a walltime-derived cap if ever
        # wanted, but it must never be a constant tighter than --time.
        _to = os.environ.get("WAVE_SWEEP_TIMEOUT_S")
        sweep_timeout = float(_to) if _to else None
        r = subprocess.run(cmd, cwd=str(REPO_ROOT), env=child_env,
                           capture_output=True, text=True,
                           timeout=sweep_timeout)
        rec["sweep_wall_s"] = time.time() - t0
        (out / "sweep_stdout.log").write_text(r.stdout)
        (out / "sweep_stderr.log").write_text(r.stderr)
        rec["sweep_returncode"] = r.returncode
        print(f"[fam-served] unified_sweep_run rc={r.returncode} "
              f"(wall {rec['sweep_wall_s']:.1f}s)")
        if r.returncode != 0:
            raise RuntimeError(
                f"unified_sweep_run --gen-engine served exited {r.returncode} "
                f"(see sweep_stderr.log): {r.stderr[-1800:]}")

        # ---- verify: per-family responses + engine sidecars + zero errors --- #
        # BoN uses the legacy per-cell file; the other families the shard layout.
        per_family = {}
        total_err = 0
        for resp in sorted(sweep_out.rglob("*responses.jsonl")):
            # BoN legacy path: <private_run>/<cell>/bon_responses.jsonl.
            # Non-BoN SHARD path: <private_run>/<FAMILY>/<cell>/shardNN/
            # responses.jsonl — the family is THREE parents up (shard, cell,
            # family). resp.parent.parent is the CELL (prompted__si_constitution),
            # which collapsed every non-BoN family into ONE key and made this
            # checker report them absent though all five wrote full stores.
            fam = ("bon" if resp.name == "bon_responses.jsonl"
                   else resp.parent.parent.parent.name)
            rows, served, errs = _served_rows_and_errors(resp)
            per_family[fam] = {"rows": rows, "served_tagged": served,
                               "errors": errs, "path": str(resp)}
            total_err += errs
        rec["per_family"] = per_family
        rec["served_error_placeholders"] = total_err

        # engine sidecars: every family must record the proxy_vllm identity.
        sidecar_engines = {}
        for sc in sorted(sweep_out.rglob("engine_stamps/*/*/*.json")):
            try:
                d = json.loads(sc.read_text())
            except Exception:
                continue
            ident = d.get("serving_identity") or ""
            blob = json.dumps(d)
            sidecar_engines[f"{d.get('leg')}/{d.get('member')}"] = {
                "serving_identity": ident,
                "proxy_vllm": ("proxy_vllm" in blob),
                "target_engine": (d.get("engine_stamp") or {}).get("target_engine")}
        rec["sidecar_engines"] = sidecar_engines

        timings = list(sweep_out.rglob("timings_*.json"))
        served_wall = None
        if timings:
            td = json.loads(timings[0].read_text()).get("timings_s", {})
            served_wall = sum(v for k, v in td.items()
                              if k.startswith(("static/", "held_out/")))
            rec["leg_timings_s"] = td
        rec["served_leg_wall_s"] = served_wall

        # PASS predicate
        want_static = [f for f in STATIC.split(",")]
        want_held = [f for f in HELD_OUT.split(",")]
        want = want_static + want_held
        every_present = all(per_family.get(f, {}).get("rows", 0) > 0 for f in want)
        # non-bon families must be engine-tagged on every row; bon carries its own
        # engine field too (bon_served). renellm may cover a subset of goals.
        nonbon_tagged = all(
            per_family.get(f, {}).get("served_tagged", 0)
            == per_family.get(f, {}).get("rows", -1)
            for f in want_static[1:] + want_held)   # skip bon (index 0)
        every_sidecar_proxy = (bool(sidecar_engines)
                               and all(v["proxy_vllm"]
                                       for v in sidecar_engines.values()))
        ok = (every_present and nonbon_tagged and total_err == 0
              and every_sidecar_proxy and r.returncode == 0)
        rec["checks"] = {
            "every_family_nonempty": every_present,
            "nonbon_rows_all_engine_tagged": nonbon_tagged,
            "zero_error_placeholders": total_err == 0,
            "every_sidecar_records_proxy_vllm": every_sidecar_proxy,
            "sweep_rc_zero": r.returncode == 0}
        rec["status"] = "SMOKE_PASS" if ok else "SMOKE_FAIL"
        (out / "smoke_record.json").write_text(json.dumps(rec, indent=2,
                                                          default=str))
        print(json.dumps({k: rec[k] for k in
                          ("status", "per_family", "sidecar_engines",
                           "served_leg_wall_s", "served_error_placeholders",
                           "sweep_wall_s", "checks")}, indent=2, default=str))
        if not ok:
            raise RuntimeError(f"fully-served validation FAILED: {rec['checks']}")
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
