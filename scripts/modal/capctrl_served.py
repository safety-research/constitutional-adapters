"""SERVED capability controls for the scale-2 (x2) psr_kl cells, on Modal H200s.

Phase 1 of moving capability eval off the HF-only path onto the SERVED (steered
vLLM + loopback RoutingProxy + lens) path the jailbreak/BoN evals already use.
Ports the same cluster serving stack (image = the proven cu129 vllm-0.25.1 /
vllm-lens-1.2.1 / transformers-5.14.1 venv, EXACTLY the truncregen
image), one H200 per cell, and runs scripts/served_capability_run.py against the
live proxy: GSM8K + XSTest + OR-Bench generated on the served path.

Phase 2 (--mmlu): mmlu_letter SERVED. The driver scores it teacher-forced
through the proxy — assistant seed "Answer:" as a continued final message
declared via petri_response_prefill (the proxy gate that re-locates the lens
response mask to header+seed+generated, so the scored token-1 distribution is
masked BY CONSTRUCTION), one greedy token, letter logprobs read from the raw
top-K (return_tokens_as_token_ids). The standup is the IN-TREE
capctrl_serving_up.py — scale12_serving_up.py plus --max-logprobs 64 (the
engine default 20 is the letter-read ceiling otherwise); /vol/repos stays
untouched. The loglikelihood positive control (steered vs scale-0 twin vs
twin-repeat letter logprobs; fidelity.check_logprob_steering_bites) gates
every cell and its verdict rides evidence.json.

Cells (steering strength x2 baked into the object = 2x directions, served coeff
1.0 — verified ratio 2.000 vs x1): {llama8b, gemma4e4b} x {safety_refusal,
si_constitution}, arm psr_kl_all. 4 cells.

Isolation (hard constraints):
  * app name capctrl-served (NOT truncregen-fitted-served-4096 / bon-throughput-
    sweep / twosi-generate).
  * writes ONLY under /vol/capctrl/. /vol/objects/scale-12 + /vol/repos are
    READ-ONLY shared inputs and untouched.

Serving tree: MY sha, staged to /vol/capctrl/repos/sisteer-<sha> — it descends
from the published psr serving tree (436ea7b psrmaskfix is in its ancestry, so
it serves the recorded response mask identically) AND carries the two ported
guards (fail-loud fit/serve position-regime guard + one post-load arm hash),
which the published 436ea7b serving tree LACKS. The guard is a positive control
that can FAIL: a mis-served object refuses at standup.

Run:
  modal run scripts/modal/capctrl_served.py --spike     # llama8b safety_refusal only
  modal run scripts/modal/capctrl_served.py --fanout    # all 4 cells
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import modal

app = modal.App("capctrl-served")
VOL = modal.Volume.from_name("twosi-fits", create_if_missing=False)
VOL_PATH = "/vol"

# --- the ONE image: the cluster serving venv 6b72fc94 stack, cu129 --------------
# Copied verbatim from truncregen_fitted_fanout.py (the proven served image).
image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04",
                              add_python="3.12")
    .apt_install("git", "build-essential")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install(
        "https://github.com/vllm-project/vllm/releases/download/v0.25.1/"
        "vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
        "#sha256=9e206f370c934a2d4b6b1f05d3d09708d344e05d80260189ef19f60755709431",
        "torch==2.11.0+cu129", "torchvision==0.26.0+cu129",
        "torchaudio==2.11.0+cu129", "vllm-lens==1.2.1",
        "pandas", "pyarrow", "aiohttp", "pyyaml", "numpy",
        "huggingface_hub", "ninja",
        extra_index_url="https://download.pytorch.org/whl/cu129",
    )
    .pip_install("transformers==5.14.1", "tokenizers==0.22.2",
                 extra_options="--no-deps --force-reinstall")
    .run_commands(
        "python -c \"import torch, vllm, transformers, aiohttp, yaml; "
        "assert torch.__version__ == '2.11.0+cu129', torch.__version__; "
        "assert vllm.__version__.startswith('0.25.1'), vllm.__version__; "
        "print('IMAGE OK', vllm.__version__, torch.__version__)\"",
        "which vllm",
    )
    .env({"HF_HOME": f"{VOL_PATH}/hf", "HF_HUB_OFFLINE": "1",
          "TOKENIZERS_PARALLELISM": "false",
          "SISTEER_HELDOUT_CORPUS_DIR": f"{VOL_PATH}/heldout_corpora"})
)

# --- my staged tree + x2 bank ---------------------------------------------------
# The tree sha is a RUN PARAMETER (modal run ... --sha <sha>): passed through to
# every cell so the container never depends on a local env var or a placeholder.
CAP_REPOS = f"{VOL_PATH}/capctrl/repos"
CAP_OBJECTS = f"{VOL_PATH}/capctrl/objects/scale-12-x2"
CAP_GEN = f"{VOL_PATH}/capctrl/gen"

# The registered served models; the standup drives every one through the
# same API. llama8b / gemma4e4b are non-thinking templates. glm46vflash
# (GLM-4.6V-Flash, a dense text-only-served VLM) carries the enable_thinking
# switch: build_server_launch pins the served render OFF
# (--default-chat-template-kwargs), and every offline prediction in
# preflight() renders under that same pin (_pinned_tok) — an unpinned
# tokenizer would predict the thinking-ON header/seeded render, a
# different world from the one the server serves. The serving GEOMETRY
# (max_model_len x max_num_seqs) is the `capability` entry of the per-eval
# table, sisteer.capeval.serving.geometry: every served eval resolves its
# own window from that one table, never a literal in a launcher, and the
# entry is MEASURED — the p99 of prompt+completion over the capability
# corpora, docs/served_geometry_p99.md — see cell_geometry(). A row may
# override a factor with `window=` / `seqs=` (model-specific: GLM's KV is
# tiny, 2 KV heads x 128 x 40 layers ~= 40 KiB/token, so a wider row would
# over-cover), and an override whose S x L exceeds the certified budget
# refuses unless the row also says allow_over_budget=True; readiness
# attests the resolved coverage anyway. A model registered here has NO x2
# bank until its fits land: preflight then fails loudly on that model (the
# standing 'GPUs last' gate), and `--models a,b` scopes a run to the models
# whose banks exist.
MODELS = {
    "llama8b": dict(hf="NousResearch/Meta-Llama-3.1-8B-Instruct",
                    rev="d10aef7999a2b5ba950ab3974312feeedbfe0b77",
                    cfg="configs/two_si.yaml"),                 # in-tree
    "gemma4e4b": dict(hf="google/gemma-4-E4B-it",
                      rev="ee0ef6023621cff504d758262d4e04895a5af4a2",
                      cfg="configs/two_si_gemma4e4b_reduced.yaml"),  # in-tree
    "glm46vflash": dict(hf="zai-org/GLM-4.6V-Flash",
                        rev="411bb4d77144a3f03accbf4b780f5acb8b7cde4e",
                        cfg="configs/two_si_glm46vflash.yaml"),  # in-tree
    # Nemotron-3-Nano-4B (nemotron_h): a Mamba2/MLP/attention hybrid — only
    # 4 of 42 blocks hold KV (8 KV heads x 128 x 4 layers x bf16 ~= 16 KiB/
    # token) plus per-request Mamba2 state pages, so the lane default 24 is
    # over-covered on the KV side and the state pages are what readiness
    # must attest (kv_group_report). Its ChatML template carries the
    # enable_thinking switch (pinned OFF like GLM: header <|im_start|> ass
    # istant \n <think> </think>, H=6); build_server_launch composes prefix
    # caching OFF for it (hybrid_recurrent_model) — no launch knob here.
    "nemotron3nano4b": dict(hf="nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
                            rev="dfaf35de3e30f1867dd8dbc38a7fc9fb52d3914f",
                            cfg="configs/two_si_nemotron3nano4b.yaml"),  # in-tree
}
CELLS = ("safety_refusal__psr_kl_all", "si_constitution__psr_kl_all")
#: the geometry-table eval this lane IS (served GSM8K/MMLU/XSTest/OR-Bench).
CAPABILITY_EVAL = "capability"


# --- the serving PATH and the ENGINE, by value -----------
# The same flag and readings as the tinycap launcher (scripts/modal/tinycap_served.py): the
# standup is told which engine to serve on EVERY launch (its own default is native since this
# landing, so a lens launch that omitted the flag would silently stand up the other engine);
# endpoint.json's `engine` is read FIRST by the reduction-mode resolver (`not_applicable` under
# the native engine, no lens key consulted) and held to the requested path after the bring-up.
# By value, like the copy above: the read site runs with the staged tree first on sys.path.
SERVING_PATH_FLAG = "--serving-path"
SERVING_PATH_LORA_NATIVE = "lora_native"
SERVING_PATH_LENS = "lens"
SERVING_PATHS = (SERVING_PATH_LORA_NATIVE, SERVING_PATH_LENS)
#: the ENTRY-POINT default (main): native — the same constant as the tinycap launcher's and
#: registry_adapter.DEFAULT_STANDUP_SERVING_PATH; this wave's registered CELLS are PSR objects,
#: which refuse under it by name at plan time — `--serving-path lens` runs them as before
SERVING_PATH_DEFAULT = SERVING_PATH_LORA_NATIVE
ENDPOINT_ENGINE_KEY = "engine"
ENGINE_LORA_NATIVE = "lora_native"
ENGINE_LENS = "lens"
ENGINES = (ENGINE_LENS, ENGINE_LORA_NATIVE)
REDUCTION_MODE_NOT_APPLICABLE = "not_applicable"
#: the row-facing spelling the native core writes beside `engine` (contracts.SERVED_ENGINE_FIELD)
SERVED_ENGINE_FIELD = "serving_engine"
REC_SERVING_ENGINE_MISMATCH = "SERVING_ENGINE_MISMATCH"
#: §1.9: the adapter bank root — the SAME volume ("twosi-fits") the tinycap launcher mounts at /vol,
#: so the two Modal launchers share one content-addressed bank; exported into every container and
#: passed explicitly as --adapter-bank <root>/<model key> under native
ADAPTER_BANK_ROOT_ENV = "SISTEER_ADAPTER_BANK_ROOT"
ADAPTER_BANK_ROOT = f"{VOL_PATH}/tinycap/multilora_adapters"
ADAPTER_BANK_FLAG = "--adapter-bank"


def check_serving_path_arg(path) -> str:
    if path not in SERVING_PATHS:
        raise ValueError(f"{SERVING_PATH_FLAG} must be one of {list(SERVING_PATHS)}, got {path!r}")
    return str(path)


def engine_of_serving_path(path) -> str:
    return ENGINE_LORA_NATIVE if check_serving_path_arg(path) == SERVING_PATH_LORA_NATIVE else ENGINE_LENS


def endpoint_engine(endpoint) -> str:
    """The engine a standup's endpoint record attests: `engine` when stated, ``lens`` when the key
    is absent (every lens standup before the native landing: lens by era); an unknown value refuses."""
    if not isinstance(endpoint, Mapping):
        raise ValueError(f"endpoint record is not a mapping ({type(endpoint).__name__})")
    engine = endpoint.get(ENDPOINT_ENGINE_KEY)
    if isinstance(engine, Mapping):
        engine = engine.get("kind")        # the /v1/models identity block spells the engine as a block
    row_engine = endpoint.get(SERVED_ENGINE_FIELD)
    if engine is not None and row_engine is not None and engine != row_engine:
        raise ValueError(f"endpoint.json states {ENDPOINT_ENGINE_KEY}={engine!r} and {SERVED_ENGINE_FIELD}="
                         f"{row_engine!r} — two spellings of one engine that disagree; a torn record")
    engine = engine if engine is not None else row_engine
    if engine is None:
        return ENGINE_LENS
    if engine not in ENGINES:
        raise ValueError(f"endpoint.json {ENDPOINT_ENGINE_KEY}={engine!r} is not one of {list(ENGINES)}")
    return str(engine)


def proxy_engine_kind(endpoint, *, timeout_s: float = 30.0):
    """The LIVE proxy's /v1/models -> petri_proxy.engine.kind: ``(kind, None)`` or ``(None, why)``."""
    import urllib.request
    url = str(endpoint.get("proxy_url") or "").rstrip("/") if isinstance(endpoint, Mapping) else ""
    if not url:
        return None, "endpoint.json names no proxy_url"
    req = urllib.request.Request(f"{url}/models")
    token = endpoint.get("proxy_token")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            listing = json.loads(resp.read().decode())
    except Exception as e:  # noqa: BLE001 — refused by the caller, never read around
        return None, f"GET {url}/models failed: {type(e).__name__}: {e}"
    meta = listing.get("petri_proxy") if isinstance(listing, Mapping) else None
    eng = meta.get("engine") if isinstance(meta, Mapping) else None
    if not isinstance(eng, Mapping) or not isinstance(eng.get("kind"), str):
        return None, f"/v1/models petri_proxy.engine is absent or malformed: {str(eng)[:200]}"
    return str(eng["kind"]), None


def _served_engine_refusal(endpoint, *, serving_path: str, rec: dict, ep_path, stop, standup, commit=None):
    """native-serving-default: the engine reading FIRST — endpoint.json's `engine` must be the one
    ``serving_path`` asks for, and under native the live proxy's /v1/models engine block must agree.
    Returns None when the cell may proceed (``rec`` carries the engine), else the REFUSAL record —
    the same exit shape as ``_resolve_served_reduction_mode`` (typed status, stop file, teardown,
    commit). Imports nothing from the tree."""
    want = engine_of_serving_path(serving_path)
    try:
        served = endpoint_engine(endpoint)
    except ValueError as e:
        why = f"standup record {ep_path}: {e}"
    else:
        rec["served_engine"] = served
        why = None
        if served != want:
            why = (f"standup record {ep_path} attests {ENDPOINT_ENGINE_KEY}={served!r} but {SERVING_PATH_FLAG} "
                   f"{serving_path} expects {want!r} — the standup did not serve the engine the launch asked for")
        elif want == ENGINE_LORA_NATIVE:
            kind, err = proxy_engine_kind(endpoint)
            rec["proxy_engine_kind"] = kind
            if kind != ENGINE_LORA_NATIVE:
                why = (f"the live proxy's /v1/models petri_proxy.engine.kind is {kind!r} but the standup attests "
                       f"{ENDPOINT_ENGINE_KEY}={served!r}" + (f" ({err})" if err else ""))
    if why is None:
        return None
    rec["status"] = REC_SERVING_ENGINE_MISMATCH
    rec["serving_engine_error"] = why
    print(f"[capctrl] REFUSING {rec.get('model')}/{rec.get('cell')}: {why}", flush=True)
    teardown_error = _teardown_standup(standup, stop)
    if teardown_error:
        rec["teardown_error"] = teardown_error
    try:
        (commit or VOL.commit)()
    except Exception as ce:
        rec["commit_error"] = f"{type(ce).__name__}: {ce}"
    return rec


def native_cell_refusal(cell: str):
    """Why ``cell`` cannot be planned under the native default — None when it can. This wave's
    cells are ``<si>__<arm>``; the arm's object kind (registry_adapter.ARM_OBJECT_KINDS) is judged
    through defense_cells' disposition (native_serving_disposition when the tree has it, else the
    backstop native_lora_unservable_reason). Local-side, before any Modal call."""
    from sisteer.capeval.petri import defense_cells as dc
    from sisteer.capeval.petri import registry_adapter as ra
    if cell == "none" or cell.startswith("prompted__"):
        return None
    arm = cell.rsplit("__", 1)[-1]
    kind = ra.ARM_OBJECT_KINDS.get(arm) or ra.ARM_OBJECT_KINDS.get(cell)
    if not kind:
        return f"cell {cell!r}: arm {arm!r} has no object kind in registry_adapter.ARM_OBJECT_KINDS"
    fn = getattr(dc, "native_serving_disposition", None)
    if callable(fn):
        verdict, reason = fn(kind, None)
    else:
        reason = dc.native_lora_unservable_reason(kind, None)
        verdict = "serve" if reason is None else "skip"
    if verdict == "serve":
        return None
    return (f"cell {cell!r} (kind {kind!r}) cannot be served on the native multi-LoRA engine — {reason}; pass "
            f"{SERVING_PATH_FLAG} {SERVING_PATH_LENS} to run it on the lens standup")




def _geometry_module(tree=None):
    """The per-eval geometry table (sisteer.capeval.serving.geometry) from the
    STAGED tree when one is given (in-container: the tree that runs), else
    from whichever sisteer is importable — the local checkout at ``modal
    run`` time, whose src is put on sys.path if it is not already."""
    if tree and f"{tree}/src" not in sys.path:
        sys.path.insert(0, f"{tree}/src")
    try:
        from sisteer.capeval.serving import geometry
    except ImportError:
        local_src = str(Path(__file__).resolve().parents[2] / "src")
        if local_src not in sys.path:
            sys.path.insert(0, local_src)
        from sisteer.capeval.serving import geometry
    return geometry


def cell_geometry(spec: dict, tree=None):
    """One cell's served geometry: the ``capability`` table window, with S
    sized from the cell's MODEL's measured batched-token budget
    (``spec["model"]`` — the canonical key every ``cell_specs`` row carries;
    ``geometry.BUDGET_BY_MODEL`` / ``ModelBudget.seqs_at``, the hybrid per-slot
    cap applied; ``seqs_source="model_budget"``) unless the registry row
    overrides ``window`` / ``seqs``. An override whose S x L exceeds the
    MODEL's budget raises ``GeometryOverBudget`` unless the row states
    ``allow_over_budget=True``. A spec without ``model`` (a bare registry row)
    sizes from the Llama-certified default, as before the per-model budgets."""
    g = _geometry_module(tree)
    return g.resolve_geometry(CAPABILITY_EVAL,
                              max_model_len=spec.get("window"),
                              max_num_seqs=spec.get("seqs"),
                              allow_over_budget=bool(spec.get("allow_over_budget", False)),
                              model_key=spec.get("model"))


def resolve_models(models=None) -> list:
    """The model slugs a run/preflight covers: every registered one by
    default, or an explicit subset — an unknown slug refuses (never a
    silent skip of a misspelled model)."""
    sel = list(models) if models else list(MODELS)
    unknown = [m for m in sel if m not in MODELS]
    if unknown:
        raise SystemExit(f"[capctrl] unknown model slug(s) {unknown}; "
                         f"registered: {sorted(MODELS)}")
    return sel


def cell_specs(models=None):
    out = []
    for model in resolve_models(models):
        for cell in CELLS:
            out.append({"model": model, "cell": cell, **MODELS[model]})
    return out


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0)); return int(s.getsockname()[1])


def _geometry_argv(geom) -> list:
    """The geometry part of a standup argv: the explicit dimensions PLUS the
    eval key and model key the geometry was resolved from, so the standup's own
    resolution — the endpoint.json provenance — equals this one (review R4: with
    the dimensions alone the inner resolver recorded eval=None, model_key=None,
    the Llama budget and over_budget=True, and its model-budget guard never
    ran). An accepted override that breaks the identity or the budget travels
    with --allow-over-budget, exactly as the outer resolution accepted it."""
    # dimensions travel BY SOURCE (review S3): a derived factor (table /
    # model_budget) is re-derived by the standup from --eval / --model-key;
    # only a recorded OVERRIDE is passed explicitly, whatever its value — so
    # the standup's window_source / seqs_source equal this resolution's.
    argv: list = []
    if geom.window_source == "override":
        argv += ["--max-model-len", str(geom.max_model_len)]
    if geom.seqs_source == "override":
        argv += ["--max-num-seqs", str(geom.max_num_seqs)]
    if geom.eval:
        argv += ["--eval", str(geom.eval)]
    else:                       # a legacy / fallback resolution: no table to re-derive from
        argv = ["--max-model-len", str(geom.max_model_len),
                "--max-num-seqs", str(geom.max_num_seqs)]
    if geom.model_key:
        argv += ["--model-key", str(geom.model_key)]
    if geom.overridden and (geom.over_budget or not geom.is_identity):
        argv += ["--allow-over-budget"]
    return argv


def _standup_cmd(tree, cfg, cell, ddir, srv, stop, geom, top_logprobs, *,
                 serving_path: str = SERVING_PATH_LENS, adapter_bank=None) -> list:
    """The standup argv with the RESOLVED geometry (never a literal window —
    the table + per-model overrides decide; tests/test_served_geometry greps
    this file for one) and its eval / model-key provenance (_geometry_argv).

    native-serving-default (§1.7 / §1.9): ``--serving-path <path>`` on EVERY standup (the library
    default here is lens, so a direct call composes today's argv plus the stated path; ``main``
    threads its own default, native), and ``--adapter-bank <bank>`` under native only (required
    there, refused on a lens standup)."""
    serving_path = check_serving_path_arg(serving_path)
    if serving_path == SERVING_PATH_LORA_NATIVE and not adapter_bank:
        raise ValueError(f"a {SERVING_PATH_LORA_NATIVE} standup needs its adapter bank ({ADAPTER_BANK_FLAG})")
    if serving_path == SERVING_PATH_LENS and adapter_bank:
        raise ValueError(f"{ADAPTER_BANK_FLAG} {adapter_bank!r} was offered to a lens standup — a wiring error")
    return [sys.executable, f"{tree}/scripts/capctrl_serving_up.py",
            "--repo-dir", tree, "--config", cfg, "--cell", cell,
            "--defenses-dir", ddir, "--out", str(srv), "--stop-file", str(stop),
            *_geometry_argv(geom),
            "--max-logprobs", str(int(top_logprobs)),
            "--gpu-mem-util", "0.90", "--ready-timeout-s", "2700",
            SERVING_PATH_FLAG, serving_path,
            *([ADAPTER_BANK_FLAG, str(adapter_bank)] if adapter_bank else [])]


#: The Phase-1 generation controls this launcher runs by default. gsm8k LEFT
#: it with PW3: the bespoke inline leg is retired and `--controls gsm8k` now
#: names the STANDARD Inspect gsm8k, a tinycap control that needs `--thinking`
#: and the tinycap standup (scripts/modal/tinycap_served.py drives it).
PHASE1_CONTROLS = "xstest,or_bench_hard_1k"


# --- the served REDUCTION MODE, resolved BY THIS FILE ---------------------------
# The read site (in _run_cell) runs AFTER the H200 bring-up has been paid for and
# BEFORE the block that can tear the standup down, and by then the STAGED
# CANDIDATE tree's src is FIRST on sys.path (cell_geometry put it there). An
# import from that tree at the read site is therefore a crash the cell cannot
# recover from: on any pre-landing sha the symbol does not exist, so the cell
# dies with ImportError having written no status record, never touching the stop
# file, leaving the standup alive until Modal reaps the container — and the one
# reading that makes a pre-landing sha runnable at all (a record stating NEITHER
# key is a pre-landing standup, which served per_row) would be unreachable at the
# only site that needs it.
#
# So the rule ships HERE, in the file Modal mounts, as a pure function: a
# deliberate SECOND COPY of sisteer.capeval.serving.lens_worker.
# endpoint_reduction_mode, pinned to it case by case over the whole record
# vocabulary by tests/test_lens_reduction_mode_attestation.py section (h). Do not
# replace it with an import of that resolver, and keep the two in step.
LENS_REDUCTION_MODE_KEY = "lens_reduction_mode"
LENS_REDUCTION_MODE_ERROR_KEY = "lens_reduction_mode_error"
REDUCTION_MODE_STACKED = "stacked"
REDUCTION_MODE_PER_ROW = "per_row"
#: the mode vocabulary, by value (lens_adapters.REDUCTION_MODES)
REDUCTION_MODES = (REDUCTION_MODE_STACKED, REDUCTION_MODE_PER_ROW)
#: how long a teardown waits for the standup to exit before killing it
STANDUP_TEARDOWN_TIMEOUT_S = 300

class ReductionModeUnattested(RuntimeError):
    """A standup endpoint record states no reduction mode AND says why it could
    not determine one — the driver-side twin of lens_worker's exception of the
    same name, raised instead of falling back to the pre-landing default."""


def resolve_reduction_mode(endpoint, *, source: str = "standup endpoint record") -> tuple:
    """``(mode, provenance)`` — the reduction mode a consumer must read out of a
    standup endpoint record, and where that reading came from. Three readings,
    and only three:

    * ``lens_reduction_mode`` NAMED (a member of :data:`REDUCTION_MODES`) —
      that mode, provenance ``"attested"``;
    * NEITHER ``lens_reduction_mode`` nor ``lens_reduction_mode_error`` stated
      — a PRE-LANDING standup (every server from before the reduction-mode
      attestation landed), which served ``per_row``; provenance
      ``"pre-landing-default"``, recorded so a store can be told apart from one
      whose server actually attested per_row. A null mode with NO reason is the
      same reading: the record stated nothing either way.
    * ``lens_reduction_mode`` null WITH a stated reason —
      :exc:`ReductionModeUnattested` quoting it. A post-landing standup ALWAYS
      writes the key, and writes null plus a reason when its workers attest
      nothing or DISAGREE, so defaulting there would record ``per_row`` as the
      served identity of a server whose arithmetic nobody could establish.

    An unknown value and an unreadable record refuse the same way.

    The ENGINE is read FIRST, in all three copies, the way
    ``lens_worker.endpoint_reduction_mode`` reads it — ``engine`` as the bare kind or the
    ``/v1/models`` block's ``{"kind": …}``, the row-facing ``serving_engine`` beside it (a
    disagreeing pair is torn and refuses); ``lora_native`` -> ``(not_applicable, "engine")`` with
    no lens key consulted (a native record that STATES one is torn and refuses); ``engine`` absent
    or ``lens`` -> the three readings above, byte for byte; any other value refuses."""
    if not isinstance(endpoint, Mapping):
        raise ReductionModeUnattested(
            f"{source}: not a mapping ({type(endpoint).__name__}) — no "
            f"{LENS_REDUCTION_MODE_KEY} can be read from it")
    stated = endpoint.get(LENS_REDUCTION_MODE_KEY)
    engine = endpoint.get(ENDPOINT_ENGINE_KEY)
    if isinstance(engine, Mapping):
        engine = engine.get("kind")        # the /v1/models identity block spells the engine as a block
    row_engine = endpoint.get(SERVED_ENGINE_FIELD)
    if engine is not None and row_engine is not None and engine != row_engine:
        raise ReductionModeUnattested(
            f"{source}: states {ENDPOINT_ENGINE_KEY}={engine!r} and {SERVED_ENGINE_FIELD}={row_engine!r} — two "
            "spellings of one engine that disagree; a torn record, refusing to read either")
    engine = engine if engine is not None else row_engine
    if engine == ENGINE_LORA_NATIVE:
        if stated is not None:
            raise ReductionModeUnattested(
                f"{source}: a native multi-LoRA standup ({ENDPOINT_ENGINE_KEY}={ENGINE_LORA_NATIVE!r}) states "
                f"{LENS_REDUCTION_MODE_KEY}={stated!r} — it has no lens hook to attest a reduction mode for; "
                "a torn record, refusing rather than reading either fact")
        return REDUCTION_MODE_NOT_APPLICABLE, "engine"
    if engine is not None and engine != ENGINE_LENS:
        raise ReductionModeUnattested(
            f"{source}: {ENDPOINT_ENGINE_KEY}={engine!r} is neither {ENGINE_LENS!r} nor {ENGINE_LORA_NATIVE!r} — "
            "no reduction mode can be read for an unknown engine")
    why_none = endpoint.get(LENS_REDUCTION_MODE_ERROR_KEY)
    if stated is None:
        if why_none:
            raise ReductionModeUnattested(
                f"{source}: {LENS_REDUCTION_MODE_KEY} is null and "
                f"{LENS_REDUCTION_MODE_ERROR_KEY} is {why_none!r} — the standup could not "
                "determine the lens reduction mode its own workers served, so this is not a "
                "pre-landing server (which states neither key and means "
                f"{REDUCTION_MODE_PER_ROW!r}); refusing rather than recording the default "
                "as served identity")
        return REDUCTION_MODE_PER_ROW, "pre-landing-default"
    if stated not in REDUCTION_MODES:
        raise ReductionModeUnattested(
            f"{source}: {LENS_REDUCTION_MODE_KEY}={stated!r} is not one of {REDUCTION_MODES}")
    return stated, "attested"


def _teardown_standup(standup, stop, timeout: int = STANDUP_TEARDOWN_TIMEOUT_S):
    """Stop the standup: the stop file, then the wait, then a kill if it outlives
    the wait. Returns ``None``, or why the teardown could not complete — a
    teardown that fails must neither crash the cell nor displace the reason the
    cell is being refused."""
    errs = []
    try:
        stop.touch()
    except Exception as e:
        errs.append(f"stop file: {type(e).__name__}: {e}")
    try:
        standup.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            standup.kill()
        except Exception as e:
            errs.append(f"kill: {type(e).__name__}: {e}")
    except Exception as e:
        errs.append(f"wait: {type(e).__name__}: {e}")
    return "; ".join(errs) or None


def _resolve_served_reduction_mode(endpoint, *, rec: dict, stop, standup, ep_path,
                                   commit=None):
    """Record the served reduction mode on the cell, or refuse the cell and tear
    the standup down. Returns ``None`` when the cell may proceed (``rec`` carries
    the mode and its provenance) and the REFUSAL RECORD otherwise.

    The whole resolution — including anything that can raise — sits in this one
    scope, so every failure takes the same exit: a typed status record, the stop
    file, a bounded wait on the standup then a kill, a volume commit. Nothing
    here may escape as a crash: the cell has already spent the GPU bring-up, and
    a crash leaves the standup running with no record of why. ``commit`` is the
    test seam for the volume commit."""
    try:
        mode, provenance = resolve_reduction_mode(endpoint,
                                                  source=f"standup record {ep_path}")
        rec[LENS_REDUCTION_MODE_KEY] = mode
        rec["lens_reduction_mode_source"] = provenance
        return None
    except Exception as e:
        why = str(e) or repr(e)
        rec["status"] = "REDUCTION_MODE_UNATTESTED"
        rec["reduction_mode_error"] = why if isinstance(e, ReductionModeUnattested) else (
            f"{type(e).__name__}: {why} (resolving the served reduction mode from "
            f"standup record {ep_path})")
        print(f"[capctrl] REFUSING {rec.get('model')}/{rec.get('cell')}: "
              f"{rec['reduction_mode_error']}", flush=True)
        teardown_error = _teardown_standup(standup, stop)
        if teardown_error:
            rec["teardown_error"] = teardown_error
        try:
            (commit or VOL.commit)()
        except Exception as ce:
            rec["commit_error"] = f"{type(ce).__name__}: {ce}"
        return rec


def _run_cell(spec: dict, limit: int, sha: str,
              controls: str = PHASE1_CONTROLS,
              out_suffix: str = "", top_logprobs: int = 64,
              serving_path: str = SERVING_PATH_LENS) -> dict:
    """Stand up the steered server for one cell from MY tree + the x2 bank, run
    the served capability driver against the proxy, tear down, gather evidence.
    ``controls``/``out_suffix`` select the wave: the Phase-1 generation trio
    (default) or the Phase-2 served-MMLU loglikelihood leg
    (``controls="mmlu"``, ``out_suffix="__mmlu"`` — separate out dir, so the
    two waves' stores never share a resume identity).
    """
    model, cell = spec["model"], spec["cell"]
    tree = f"{CAP_REPOS}/sisteer-{sha}"
    if not Path(f"{tree}/src/sisteer").is_dir():
        return {"model": model, "cell": cell, "status": "NO_TREE",
                "tree": tree}
    ddir = f"{CAP_OBJECTS}/{model}"
    out_dir = Path(f"{CAP_GEN}/{model}__{cell}{out_suffix}")
    import shutil as _sh
    _sh.rmtree(out_dir, ignore_errors=True)          # export is non-idempotent
    out_dir.mkdir(parents=True, exist_ok=True)
    srv = Path(f"/tmp/srv_{model}_{cell}"); srv.mkdir(parents=True, exist_ok=True)
    stop = srv / "STOP"
    cfg = spec["cfg"]
    if not Path(cfg).is_absolute():
        cfg = f"{tree}/{cfg}"
    serving_path = check_serving_path_arg(serving_path)
    native = serving_path == SERVING_PATH_LORA_NATIVE
    rec = {"model": model, "cell": cell, "sha": sha[:7], "cap_x2": True,
           "out": str(out_dir), "serving_path": serving_path, "serving_engine": engine_of_serving_path(serving_path)}
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{tree}/src:{tree}/scripts"
    env[ADAPTER_BANK_ROOT_ENV] = ADAPTER_BANK_ROOT          # §1.9: the bank root, in every container
    # the served geometry from the STAGED tree's table, S from THIS model's
    # measured budget (+ any per-model override in MODELS) — logged with its
    # basis and budget, recorded on the cell (geometry.as_dict carries
    # model_key / budget / seqs_source)
    geom = cell_geometry(spec, tree)
    print(geom.describe(), flush=True)
    rec["geometry"] = geom.as_dict()

    # --- stand up (the IN-TREE capctrl_serving_up.py — scale12_serving_up
    # plus --max-logprobs 64, the served letter-read ceiling; /vol/repos'
    # shared copy stays untouched)
    t0 = time.time()
    standup = subprocess.Popen(
        _standup_cmd(tree, cfg, cell, ddir, srv, stop, geom, top_logprobs,
                     serving_path=serving_path,
                     adapter_bank=(f"{ADAPTER_BANK_ROOT}/{model}" if native else None)),
        env=env, stdout=(srv / "standup.log").open("w"),
        stderr=subprocess.STDOUT)
    ep = srv / "endpoint.json"
    for _ in range(900):                              # <= 75 min bring-up
        if ep.exists() or standup.poll() is not None:
            break
        time.sleep(5)
    if not ep.exists():
        rec["status"] = "STANDUP_FAILED"
        rec["standup_tail"] = (srv / "standup.log").read_text()[-3000:]
        vl = srv / "vllm_server.log"
        rec["vllm_tail"] = vl.read_text()[-3000:] if vl.exists() else None
        VOL.commit()
        return rec
    endpoint = json.loads(ep.read_text())
    rec["standup_s"] = round(time.time() - t0, 1)
    rec["proxy_url"] = endpoint["proxy_url"]
    # THE SERVED REDUCTION MODE, resolved by THIS FILE (resolve_reduction_mode
    # above) and never imported from the staged candidate tree, whose src is
    # first on sys.path here and which on a pre-landing sha has no such symbol:
    # a record that states NEITHER key is a pre-landing standup and means
    # per_row, but one whose lens_reduction_mode is null WITH a
    # lens_reduction_mode_error could not determine what its own workers served
    # — that must never be read as the per_row default and recorded as this
    # store's served identity. The whole resolution lives in one scope whose
    # every failure refuses the cell, touches the stop file and tears the server
    # down rather than spending the GPU on a store nobody can label.
    # native-serving-default (§1.7): the ENGINE first — endpoint.json's `engine` held to the
    # requested path (and, under native, the live proxy's engine block), the same exit shape
    refused = _served_engine_refusal(endpoint, serving_path=serving_path, rec=rec, ep_path=ep,
                                     stop=stop, standup=standup)
    if refused is not None:
        return refused
    refused = _resolve_served_reduction_mode(endpoint, rec=rec, stop=stop,
                                             standup=standup, ep_path=ep)
    if refused is not None:
        return refused

    # --- the served capability driver ----------------------------------------
    # No --steer-positions: the driver arms the response mask itself from the
    # object's own fit leg (the x2 psr blob carries positions="response" +
    # response_header_len=4, verified), matching how the object was fit.
    cmd = [sys.executable, f"{tree}/scripts/served_capability_run.py", "generate",
           "--proxy-url", endpoint["proxy_url"],
           "--routing-table", endpoint["routing_table"],
           "--cell", cell, "--config", cfg, "--defenses-dir", ddir,
           "--out-root", str(out_dir),
           "--controls", controls,
           "--top-logprobs", str(int(top_logprobs)),
           # the client fan-out follows the launched width (S), by construction
           "--limit", str(limit), "--concurrency", str(geom.max_num_seqs),
           # judge pinned EXPLICITLY: the x1 comparison stores record
           # claude-sonnet-5; inheriting judging.JUDGE_MODEL would let a
           # lineage change silently move these numbers.
           "--judge-model", "claude-sonnet-5",
           "--vllm-log", str(srv / "vllm_server.log")]
    t1 = time.time()
    r = subprocess.run(cmd, cwd=tree, env=env, capture_output=True, text=True,
                       timeout=3 * 3600)
    rec["driver_s"] = round(time.time() - t1, 1)
    rec["driver_rc"] = r.returncode
    (out_dir / "driver_stdout.log").write_text(r.stdout)
    (out_dir / "driver_stderr.log").write_text(r.stderr)
    rec["steering_bites"] = ("steering-bites verified" in r.stdout)
    rec["gate_lines"] = [l for l in r.stdout.splitlines()
                         if any(k in l for k in ("steering-bites", "salt",
                                                 "GSM8K acc", "mmlu", "mask",
                                                 "LOGLIKELIHOOD", "logprob",
                                                 "DECODE-ONLY", "refus"))][:16]

    if r.returncode != 0:
        # a driver failure's root cause often lives in the ENGINE log
        # (an in-worker refusal reaches the client as a bare 500), and
        # the srv dir dies with the container — persist both logs.
        for name in ("vllm_server.log", "standup.log"):
            src_log = srv / name
            if src_log.exists():
                (out_dir / name).write_text(src_log.read_text(
                    errors="replace")[-2_000_000:])

    # --- teardown -------------------------------------------------------------
    teardown_error = _teardown_standup(standup, stop)
    if teardown_error:
        rec["teardown_error"] = teardown_error
    ev = out_dir / "evidence.json"
    rec["evidence"] = json.loads(ev.read_text()) if ev.exists() else None
    # steering_bites: authoritative from evidence.json (the stdout grep was a
    # false-negative — the driver prints it differently than it records it).
    sb = ((rec.get("evidence") or {}).get("steering_bites") or {})
    rec["steering_bites"] = bool(sb.get("verified"))
    lb = ((rec.get("evidence") or {}).get("logprob_steering_bites") or {})
    if lb:
        rec["logprob_steering_bites"] = bool(lb.get("verified"))
    VOL.commit()
    return rec


@app.function(image=image, volumes={VOL_PATH: VOL}, gpu="H200",
              timeout=4 * 3600, max_containers=4)
def run_cell(spec: dict, limit: int, sha: str,
             controls: str = PHASE1_CONTROLS,
             out_suffix: str = "", top_logprobs: int = 64,
             serving_path: str = SERVING_PATH_LENS) -> dict:
    rec = _run_cell(spec, limit, sha, controls, out_suffix, top_logprobs, serving_path=serving_path)
    print("[capctrl] REC", json.dumps({k: v for k, v in rec.items()
                                       if k != "evidence"}))
    return rec


@app.function(image=image, volumes={VOL_PATH: VOL}, timeout=1800)
def stage_corpora(sha: str) -> dict:
    """Populate /vol/hf with the 3 benign corpora (gsm8k_platinum, xstest_v2,
    or_bench) at their PINNED revisions, so the offline eval loaders find them.
    Network ON here only (HF_HUB_OFFLINE popped); the identical snapshot_download
    the loader uses (repo_type='dataset', revision, allow_patterns), just with
    local_files_only=False. One-time; commits to the volume."""
    import os as _os, yaml
    _os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import snapshot_download
    pins = yaml.safe_load(open(
        f"{CAP_REPOS}/sisteer-{sha}/configs/capeval/pins.yaml"))
    ds = pins.get("datasets", pins)
    out = {}
    for name in ("gsm8k_platinum", "xstest_v2", "or_bench"):
        e = ds[name]
        path = snapshot_download(e["repo_id"], repo_type="dataset",
                                 revision=e["revision"],
                                 allow_patterns=e.get("patterns") or None,
                                 local_files_only=False)
        out[name] = {"repo_id": e["repo_id"], "rev": e["revision"][:12],
                     "path": path}
        print(f"[stage] {name}: {e['repo_id']}@{e['revision'][:12]} -> {path}")
    # gsm8k and mmlu are loaded via datasets.load_dataset (NOT the corpora
    # snapshot path), which needs its own arrow cache under HF_HOME/datasets
    # — populate it with the IDENTICAL calls parity.gsm8k_parity_rows /
    # parity.mmlu_parity_rows make (network on here).
    from datasets import load_dataset
    g = ds["gsm8k_platinum"]
    d = load_dataset("madrylab/gsm8k-platinum", "main", split="test",
                     revision=g["revision"])
    out["gsm8k_load_dataset"] = {"n": len(d)}
    print(f"[stage] gsm8k load_dataset arrow cache: n={len(d)}")
    m = ds["mmlu"]
    dm = load_dataset("cais/mmlu", "all", split="validation",
                      revision=m["revision"])
    out["mmlu_load_dataset"] = {"n": len(dm), "rev": m["revision"][:12]}
    print(f"[stage] mmlu load_dataset arrow cache: n={len(dm)}")
    VOL.commit()
    return out


@app.function(image=image, volumes={VOL_PATH: VOL}, timeout=1800)
def preflight(sha: str, limit: int = 100, models=None) -> dict:
    """CPU-only: verify EVERY non-GPU precondition the driver + guard touch,
    for every cell of the selected models (all registered by default),
    before any server exists (the standing 'GPUs last' gate).
    Checks: driver argparse accepts the module's exact flags; config/object/pins
    paths; corpora load OFFLINE + item-subset sha; object provenance
    (positions/header_len); tokenizer + generation_header_ids length == the
    object's response_header_len (the fit/serve guard's cross-check, PREDICTED
    offline); out-root writability. Every per-model prediction renders under
    the SAME thinking regime the server composes (_pinned_tok)."""
    import sys, json, importlib
    from pathlib import Path
    tree = f"{CAP_REPOS}/sisteer-{sha}"
    sys.path.insert(0, f"{tree}/scripts"); sys.path.insert(0, f"{tree}/src")
    models = resolve_models(models)
    rep = {"sha": sha[:7], "ok": True, "models": list(models), "checks": {}}
    def chk(name, fn):
        try:
            rep["checks"][name] = {"ok": True, **(fn() or {})}
        except BaseException as e:
            import traceback
            rep["checks"][name] = {"ok": False, "err": f"{type(e).__name__}: {e}"}
            rep["ok"] = False

    import os as _os
    _os.environ["CAPEVAL_PINS"] = f"{tree}/configs/capeval/pins.yaml"
    import served_capability_run as srun
    from sisteer.capeval.tiers import judged
    from sisteer.capeval import corpora as _corpora

    def _pinned_tok(model):
        """The model's tokenizer under the SAME render regime the server
        composes: build_server_launch pins --default-chat-template-kwargs
        {"enable_thinking": false} whenever the pinned template carries
        the switch (GLM-4.6V: the header becomes
        <|assistant|> \\n <think> </think> \\n, H=5, and '/nothink' is
        appended to the user turn). An unpinned tokenizer renders the
        thinking-ON regime — a different header and a different seeded
        render — so every offline prediction below would describe a world
        the server never serves. The identical seam every render site
        uses (models.pin_reasoning_template_off); a no-op on plain
        templates (llama, gemma-4)."""
        from transformers import AutoTokenizer
        from sisteer.models import pin_reasoning_template_off
        tok = AutoTokenizer.from_pretrained(MODELS[model]["hf"],
                                            revision=MODELS[model]["rev"])
        pin_reasoning_template_off(tok)
        return tok

    def cli():
        cap = {}
        orig = srun.cmd_generate
        srun.cmd_generate = lambda a: cap.update(vars(a))
        try:
            for model in models:
                cfg = MODELS[model]["cfg"]
                seqs = cell_geometry({"model": model, **MODELS[model]}, tree).max_num_seqs
                srun.main(["generate", "--proxy-url", "http://x/v1",
                    "--routing-table", "/tmp/rt.json", "--cell",
                    "safety_refusal__psr_kl_all", "--config", f"{tree}/{cfg}",
                    "--defenses-dir", f"{CAP_OBJECTS}/{model}", "--out-root",
                    "/tmp/pf", "--controls", PHASE1_CONTROLS,
                    "--limit", str(limit), "--concurrency", str(seqs),
                    "--judge-model", "claude-sonnet-5", "--vllm-log", "/tmp/v"])
        finally:
            srun.cmd_generate = orig
        return {"parsed_controls": cap.get("controls"), "judge": cap.get("judge_model")}
    chk("argparse_accepts_module_flags", cli)

    EXP = {"or_bench_hard_1k": "9c2cc05bfa0f236d", "xstest_v2": "c72f4da328aca094"}
    def _sub(rows, lim, seed=0):
        import random
        if not lim or lim >= len(rows): return rows
        o = list(range(len(rows))); random.Random(seed).shuffle(o)
        return [rows[i] for i in sorted(o[:lim])]
    def corpora():
        out = {}
        pins = _corpora.load_pins()
        from sisteer.capeval.tiers import parity
        bad = []
        # each corpus independently — enumerate the FULL set in one sweep
        try:
            gr = _sub(parity.gsm8k_parity_rows(pins), limit)
            out["gsm8k"] = {"n": len(gr), "ok": True}
        except BaseException as e:
            out["gsm8k"] = {"ok": False, "err": f"{type(e).__name__}: {e}"}
            bad.append("gsm8k")
        for task, loader in (("or_bench_hard_1k", judged.or_bench_rows),
                             ("xstest_v2", judged.xstest_rows)):
            try:
                rows = _sub(loader(pins), limit)
                sha_ = judged.rows_content_sha(rows)
                out[task] = {"n": len(rows), "sha": sha_,
                             "matches_HF": sha_ == EXP[task]}
                if sha_ != EXP[task]:
                    bad.append(f"{task}:sha")
            except BaseException as e:
                out[task] = {"ok": False, "err": f"{type(e).__name__}: {e}"}
                bad.append(task)
        if bad:
            raise AssertionError(f"corpora failing: {bad} (details above)")
        return out
    chk("corpora_offline_and_item_sha", corpora)

    def objects_and_guard():
        import torch
        from sisteer.chat_markers import generation_header_ids
        out = {}
        toks = {}
        for model in models:
            # the PINNED render: the guard compares the object's recorded
            # thinking-off H against the header the server will derive
            toks[model] = len(generation_header_ids(_pinned_tok(model)))
        for model in models:
            for cell in CELLS:
                blob = torch.load(f"{CAP_OBJECTS}/{model}/{cell}/object.pt",
                                  map_location="cpu", weights_only=True)
                H = blob.get("response_header_len")
                genlen = toks[model]
                ok = (blob.get("positions") == "response" and H == genlen)
                out[f"{model}/{cell}"] = {"positions": blob.get("positions"),
                    "response_header_len": H, "gen_header_ids_len": genlen,
                    "guard_will_pass": ok}
                if not ok:
                    raise AssertionError(f"{model}/{cell}: guard MISMATCH H={H} genlen={genlen} pos={blob.get('positions')}")
        return out
    chk("object_provenance_and_guard_prediction", objects_and_guard)

    def writable():
        d = Path(f"{CAP_GEN}/_preflight"); d.mkdir(parents=True, exist_ok=True)
        (d / "probe").write_text("ok"); (d / "probe").unlink()
        return {"out_root": str(d)}
    chk("out_root_writable", writable)

    # ---- Phase 2 (served MMLU loglikelihood) preconditions ----------------
    def vllm_scoring_surface():
        """The pinned wheel's request surface the mmlu leg relies on —
        knowable offline, so a drifted pin fails HERE, not at standup."""
        import inspect
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest as R)
        f = R.model_fields
        need = ("continue_final_message", "add_generation_prompt",
                "logprobs", "top_logprobs", "return_tokens_as_token_ids")
        missing = [k for k in need if k not in f]
        if missing:
            raise AssertionError(f"pinned vLLM chat protocol lacks {missing}")
        import dataclasses as _dc
        from vllm.config import ModelConfig
        flds = {x.name: getattr(x.default, "default", x.default)
                for x in _dc.fields(ModelConfig)}
        if flds.get("logprobs_mode") != "raw_logprobs":
            raise AssertionError(
                f"logprobs_mode default is {flds.get('logprobs_mode')!r}, "
                "not raw_logprobs — returned logprobs would not be the "
                "log-softmax of the raw logits the HF reference reads")
        # the chat render tokenizes with add_special_tokens=False (no
        # double BOS) — the fact that makes string render parity imply
        # token-id parity with the HF side's encode_kwargs.
        from vllm.renderers import base as rb
        src = inspect.getsource(rb.BaseRenderer.default_chat_tok_params.func)
        if "add_special_tokens=False" not in src:
            raise AssertionError(
                "pinned vLLM chat tokenization no longer pins "
                "add_special_tokens=False — render parity would not imply "
                "token parity; re-verify before serving")
        # str(): a dataclass default here can be a pydantic FieldInfo,
        # which the LOCAL modal venv cannot unpickle (bitten 2026-09-01).
        return {"max_logprobs_default": str(flds.get("max_logprobs")),
                "logprobs_mode": str(flds.get("logprobs_mode"))}
    chk("phase2_vllm_scoring_surface", vllm_scoring_surface)

    def mmlu_render_parity():
        """Per model: the SERVER-side seeded render (apply_chat_template
        with continue_final_message — the exact callable under the pinned
        renderer) must reproduce the HF instrument's
        build_chat_prompt + "Answer:" byte-for-byte on real pinned rows;
        the header ids must be locatable at the HF scorer's own boundary
        (the served mask's rfind == parity's _rfind_subseq); the answer
        letters must be single tokens."""
        from sisteer.capeval import corpora as _corpora
        from sisteer.capeval.serving import lens_adapters as la
        from sisteer.capeval.tiers import parity
        from sisteer.chat_markers import generation_header_ids
        from sisteer.eval.capability_ext import _letter_token_ids, _letters
        from sisteer.generate import encode_kwargs
        import served_capability_run as srun
        pins = _corpora.load_pins()
        rows = parity.mmlu_parity_rows(pins)[:5]
        out = {}
        for model in models:
            tok = _pinned_tok(model)
            # the server renders with the composed --default-chat-template-
            # kwargs; the local seeded render must carry the identical
            # kwargs or it predicts a different string (GLM: no '/nothink',
            # a thinking-ON header). Empty on plain templates.
            render_kw = dict(getattr(tok, "_sisteer_template_kwargs", {})
                             or {})
            header = [int(t) for t in generation_header_ids(tok)]
            letter_ids = [int(t) for t in _letter_token_ids(tok, _letters(4))]
            for row in rows:
                want = parity._mmlu_prompt(tok, row, system=None)
                msgs = [{"role": "user",
                         "content": srun._mmlu_user_text(row)},
                        {"role": "assistant",
                         "content": srun.MMLU_ANSWER_SEED}]
                got = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False,
                    continue_final_message=True, **render_kw)
                if got != want:
                    raise AssertionError(
                        f"{model} {row['item_id']}: seeded chat render != "
                        f"instrument render\nserved: {got[-200:]!r}\n"
                        f"instr.: {want[-200:]!r}")
                ids = list(tok(want, **encode_kwargs(tok, want))["input_ids"])
                start = la._rfind_header(ids, header)
                if start < 0:
                    raise AssertionError(
                        f"{model} {row['item_id']}: header ids {header} not "
                        "locatable in the seeded render's ids — boundary "
                        "retokenization; the served mask would refuse")
                if start != parity._rfind_subseq(ids, header):
                    raise AssertionError(
                        f"{model} {row['item_id']}: served rfind boundary "
                        f"{start} != parity._rfind_subseq — the two scorers "
                        "would mask different regions")
                seed_ids = ids[start + len(header):]
                if not seed_ids:
                    raise AssertionError(
                        f"{model} {row['item_id']}: no seed tokens after "
                        "the header — the render lost the 'Answer:' seed")
            out[model] = {"header_len": len(header), "letter_ids": letter_ids,
                          "rows_checked": len(rows)}
        return out
    chk("phase2_mmlu_render_parity", mmlu_render_parity)

    def worker_seam():
        """CPU sanity of the flag's whole worker-side path: selector parse,
        slot threading, seeded boundary location, mask coverage of the
        scored (final prefill) row, and the undeclared-seed refusal."""
        import torch
        from sisteer.capeval.serving import lens_adapters as la
        from sisteer.capeval.serving import lens_steer as ls
        # BOTH wire forms: vllm_xargs lax-coerces JSON true -> int 1
        for wire in (True, 1):
            sel = ls.parse_selector({"petri_object": "c",
                                     "expect_hash": "h",
                                     "response_prefill": wire})
            assert sel["response_prefill"] is True
        header = (7, 8, 9)
        ids = (1, 2, 3, 7, 8, 9, 4, 5)   # header + 2-token seed
        ctx = la.HookContext("prefill", tuple(range(len(ids))), ids,
                             response_prefill=True)
        op = la.HookOp(op="add_vector", layer=0, positions="response",
                       header_ids=header,
                       vector=torch.zeros(4, dtype=torch.float32))
        start = la._response_start(op, ctx, len(ids))
        assert start == 3, start
        mask = la._row_mask(op, ctx, torch.zeros(len(ids), 4))
        assert bool(mask[-1]) and bool(mask[3]) and not bool(mask[2])
        ctx0 = la.HookContext("prefill", tuple(range(len(ids))), ids)
        try:
            la._response_start(op, ctx0, len(ids))
            raise AssertionError("undeclared seeded render did NOT refuse")
        except la.AdapterError:
            pass
        return {"located_start": start}
    chk("phase2_worker_seam", worker_seam)

    def wire_roundtrip():
        """The scoring payload's FULL wire round trip through the PINNED
        pydantic request model — the check whose absence cost the first
        spike a GPU bring-up (2026-09-01): the proxy injected a bool into
        vllm_xargs, the pinned frontend's dict[str, str|int|float]
        lax-coerced it to int 1, and the worker's strict-bool parse
        refused INSIDE the forward, killing the EngineCore. Knowable
        offline; now known offline. Builds the driver's exact payload,
        applies the proxy's exact injection, validates through
        ChatCompletionRequest, and parses the selector the worker would
        actually receive."""
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest)
        from sisteer.capeval.serving import lens_steer as ls
        from sisteer.capeval.serving import lens_worker as lwk
        from sisteer.capeval.serving.proxy import (RESPONSE_PREFILL_KEY,
                                                   UPSTREAM_EXTRA_ARGS_KEY)
        payload = srun._mmlu_scoring_payload(
            "petri__si_safety__psr_kl_all", "Which letter? Reply A-D.",
            system=None, top_logprobs=64, response_prefill=True)
        assert payload[RESPONSE_PREFILL_KEY] is True
        # the proxy's rewrite, verbatim in effect: client key popped,
        # selector injected with the transport-int fact, salt stamped
        upstream = dict(payload)
        upstream.pop(RESPONSE_PREFILL_KEY)
        upstream["model"] = MODELS["llama8b"]["hf"]
        upstream[UPSTREAM_EXTRA_ARGS_KEY] = {
            "petri_object": "si_safety__psr_kl_all",
            "expect_hash": "ab" * 8, "response_prefill": 1}
        upstream["cache_salt"] = "petri|preflight|roundtrip"
        req = ChatCompletionRequest.model_validate(upstream)
        wire = dict(req.vllm_xargs)
        sel = lwk.extract_selector(wire)
        if sel is None or sel.get("response_prefill") is not True:
            raise AssertionError(
                f"wire selector did not round-trip: vllm_xargs={wire!r} "
                f"-> parsed {sel!r}")
        if req.continue_final_message is not True                 or req.add_generation_prompt is not False                 or req.top_logprobs != 64 or not req.logprobs                 or req.return_tokens_as_token_ids is not True:
            raise AssertionError(
                "scoring knobs did not survive validation: "
                f"cfm={req.continue_final_message} "
                f"agp={req.add_generation_prompt} "
                f"top={req.top_logprobs} lp={req.logprobs} "
                f"ids={req.return_tokens_as_token_ids}")
        return {"wire_response_prefill": repr(wire.get("response_prefill")),
                "parsed": True}
    chk("phase2_wire_roundtrip", wire_roundtrip)

    def agent_loop_wire_roundtrip():
        """The multi-turn landing's analogue of phase2_wire_roundtrip: a
        Petri agent-loop turn (a tool result awaiting the model, in the
        OpenAI wire form with STRING tool-call arguments), declared by the
        Petri-facing entry (proxy.declare_agent_loop), rewritten by the
        proxy's exact injection (client key popped, selector fact
        agent_loop=1), validated through the PINNED ChatCompletionRequest
        (whose vllm_xargs is dict[str, str|int|float] — the lax coercion
        that turned a bool into int 1 and killed the first Phase-2
        bring-up), and parsed by the worker's extractor as exactly True.
        Knowable offline; known offline."""
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest)
        from sisteer.capeval.serving import lens_worker as lwk
        from sisteer.capeval.serving.proxy import (
            AGENT_LOOP_KEY, AGENT_LOOP_SHAPE_TOOL_RESULT,
            UPSTREAM_EXTRA_ARGS_KEY, agent_loop_shape, declare_agent_loop)
        tools = [{"type": "function", "function": {
            "name": "get_weather", "description": "Get weather",
            "parameters": {"type": "object",
                           "properties": {"city": {"type": "string"}},
                           "required": ["city"]}}}]
        body = {
            "model": "petri__si_safety__psr_kl_all",
            "messages": [
                {"role": "user", "content": "Please check the weather."},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "call1", "type": "function",
                                 "function": {
                                     "name": "get_weather",
                                     "arguments": json.dumps(
                                         {"city": "Boston"})}}]},
                {"role": "tool", "tool_call_id": "call1",
                 "content": "Sunny, 72F."}],
            "tools": tools, "max_tokens": 64, "temperature": 0.0}
        declared = declare_agent_loop(body, lens_route=True)
        assert declared[AGENT_LOOP_KEY] is True
        assert agent_loop_shape(declared) == AGENT_LOOP_SHAPE_TOOL_RESULT
        # the proxy's rewrite, verbatim in effect: client key popped,
        # selector injected with the transport-int fact, salt stamped
        upstream = dict(declared)
        upstream.pop(AGENT_LOOP_KEY)
        upstream["model"] = MODELS["llama8b"]["hf"]
        upstream[UPSTREAM_EXTRA_ARGS_KEY] = {
            "petri_object": "si_safety__psr_kl_all",
            "expect_hash": "ab" * 8, "agent_loop": 1}
        upstream["cache_salt"] = "petri|preflight|agentloop"
        req = ChatCompletionRequest.model_validate(upstream)
        wire = dict(req.vllm_xargs)
        sel = lwk.extract_selector(wire)
        if sel is None or sel.get("agent_loop") is not True:
            raise AssertionError(
                f"wire selector did not round-trip: vllm_xargs={wire!r} "
                f"-> parsed {sel!r}")
        if "response_prefill" in sel:
            raise AssertionError(
                f"an agent-loop turn parsed a response_prefill fact: {sel!r}")
        if req.continue_final_message or req.add_generation_prompt is not True:
            raise AssertionError(
                "agent-loop render knobs did not survive validation: "
                f"cfm={req.continue_final_message} "
                f"agp={req.add_generation_prompt}")
        if not req.tools:
            raise AssertionError("the tool list did not survive validation")
        return {"wire_agent_loop": repr(wire.get("agent_loop")),
                "parsed": True}
    chk("phase2_agent_loop_wire_roundtrip", agent_loop_wire_roundtrip)

    def mmlu_cli():
        cap = {}
        orig = srun.cmd_generate
        srun.cmd_generate = lambda a: cap.update(vars(a))
        try:
            srun.main(["generate", "--proxy-url", "http://x/v1",
                "--routing-table", "/tmp/rt.json", "--cell",
                "safety_refusal__psr_kl_all", "--config",
                f"{tree}/configs/two_si.yaml",
                "--defenses-dir", f"{CAP_OBJECTS}/llama8b", "--out-root",
                "/tmp/pf", "--controls", "mmlu", "--limit", str(limit),
                "--concurrency", str(cell_geometry({"model": "llama8b", **MODELS["llama8b"]},
                                                   tree).max_num_seqs),
                "--top-logprobs", "64",
                "--judge-model", "claude-sonnet-5"])
        finally:
            srun.cmd_generate = orig
        return {"controls": cap.get("controls"),
                "top_logprobs": cap.get("top_logprobs")}
    chk("phase2_mmlu_cli", mmlu_cli)

    def standup_flag():
        """The in-tree standup must exist and accept --max-logprobs (the
        letter-read ceiling is a LAUNCH property; a stale tree would
        stand up an engine-default-20 server and every top-64 request
        would 400)."""
        text = Path(f"{tree}/scripts/capctrl_serving_up.py").read_text()
        if "--max-logprobs" not in text:
            raise AssertionError("capctrl_serving_up.py lacks --max-logprobs")
        if SERVING_PATH_FLAG not in text:
            # native-serving-default: a pre-flip tree cannot be told which engine to serve
            raise AssertionError(f"capctrl_serving_up.py lacks {SERVING_PATH_FLAG} on this tree (a pre-flip sha)")
        return {"standup": "capctrl_serving_up.py", "flags": ["--max-logprobs", SERVING_PATH_FLAG]}
    chk("phase2_standup_max_logprobs", standup_flag)

    VOL.commit()
    print("[preflight]", json.dumps(rep, indent=1, default=str))
    return rep


@app.local_entrypoint()
def main(spike: bool = False, fanout: bool = False, stage: bool = False,
         pf: bool = False, limit: int = 100, sha: str = "", skip: str = "",
         mmlu: bool = False, top_logprobs: int = 64, models: str = "",
         serving_path: str = SERVING_PATH_DEFAULT):
    if not sha:
        raise SystemExit("pass --sha <tree sha staged at /vol/capctrl/repos/>")
    # native-serving-default (§1.7): the serving path — native by default, `lens` on request. This
    # wave's CELLS are PSR objects: under the default they refuse HERE, by name, before any Modal call
    try:
        serving_path = check_serving_path_arg(serving_path)
    except ValueError as e:
        raise SystemExit(f"[capctrl] {e}") from None
    if serving_path == SERVING_PATH_LORA_NATIVE and (spike or fanout):
        refusals = [why for why in (native_cell_refusal(c) for c in CELLS) if why]
        if refusals:
            raise SystemExit("[capctrl] " + "; ".join(refusals))
    # --models a,b scopes preflight + fanout to these registered slugs
    # (default: every registered model — a model whose x2 bank has not
    # landed fails preflight loudly rather than being skipped)
    sel = resolve_models([m for m in models.split(",") if m] or None)
    if stage:
        print("[capctrl] staging benign corpora into /vol/hf ...")
        print(json.dumps(stage_corpora.remote(sha), indent=1))
        if not (spike or fanout or pf):
            return
    if pf:
        r = preflight.remote(sha, limit, sel)
        print(json.dumps(r, indent=1, default=str))
        if not r.get("ok"):
            raise SystemExit("[capctrl] PREFLIGHT FAILED — fix all above before any GPU")
        if not (spike or fanout):
            return
    if not (spike or fanout):
        raise SystemExit("pass --spike (llama8b safety_refusal) or --fanout "
                         "(every selected model x CELLS; --models to scope)")
    specs = cell_specs(sel)
    if spike:
        specs = [s for s in specs
                 if s["model"] == "llama8b" and s["cell"] == "safety_refusal__psr_kl_all"]
    if skip:  # "model:cell,model:cell" pairs already done (e.g. the spike)
        drop = {tuple(x.split(":", 1)) for x in skip.split(",") if x}
        specs = [s for s in specs if (s["model"], s["cell"]) not in drop]
    controls = "mmlu" if mmlu else PHASE1_CONTROLS
    out_suffix = "__mmlu" if mmlu else ""
    print(f"[capctrl] launching {len(specs)} cell(s) @ sha {sha[:7]}, "
          f"limit {limit}, controls {controls}")
    print(f"[capctrl] serving path: {serving_path} ({SERVING_PATH_FLAG}; engine {engine_of_serving_path(serving_path)})")
    recs = list(run_cell.map(specs, kwargs={"limit": limit, "sha": sha,
                                            "controls": controls,
                                            "out_suffix": out_suffix,
                                            "top_logprobs": top_logprobs,
                                            "serving_path": serving_path}))
    ok = sum(1 for r in recs
             if r.get("driver_rc") == 0 and r.get("steering_bites")
             and (not mmlu or r.get("logprob_steering_bites")))
    print(f"[capctrl] {ok}/{len(recs)} cells OK (rc=0 + steering-bites"
          + (" + logprob-bites)" if mmlu else ")"))
    for r in recs:
        print(json.dumps({k: v for k, v in r.items() if k != "evidence"}))
