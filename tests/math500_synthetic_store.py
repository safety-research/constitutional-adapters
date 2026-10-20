"""SYNTHETIC stores for the MATH-500 thinking-ON analysis tests
(``scripts/tinycap_math500_thinking.py``) and for a manual figure render.

Placeholder sha, ``TESTHOST``, invented item ids — never real recorded data.
Every store is written through the REAL writers and record builders of this
tree: ``completion.finish_task`` files the summary, the ``outcomes/<task>.json``
record and the done marker; the certification records are the PRODUCER's own
helpers' outputs — ``contracts.render_certified`` (predicate (A)),
``contracts.lens_position_attested`` over a producer-shaped standup record
(predicate (B′)), ``contracts.lens_on_bite_verdict`` (predicate (C)),
``contracts.transfer_provenance_block``; the sampling block is
``contracts.sampling_identity`` and every row's digest ``contracts.sampling_ref``;
every request-evidence record is ``contracts.request_evidence_record`` over the
body the driver sends (one per ATTEMPTED call: the bites probe legs, the render
probes, the lens-ON bite, the task calls — probes carry no task / sample, as
the driver's recorder writes them); every ledger row is
``usage_ledger.ledger_row``. The blocks the driver assembles inline with no
helper (``steering_bites``, ``resume_attempts``, the route block) mirror
``scripts/served_capability_run.py`` key for key — so the reader and the
producer cannot drift apart without a red test (fixtures built THROUGH the
producer, never the reader's own shapes).

Modelled on ``tests/oos_synthetic_store.py`` (imported for nothing: this
experiment's legs are model x cell x sampling mode, not cell x arm). A helper
module rather than a test: the figure smoke imports it from a venv without
pytest.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from sisteer.capeval.serving import usage_ledger as ul
from sisteer.capeval.serving.usage_ledger import ledger_row
from sisteer.capeval.tiers import inspect_served as isv
from sisteer.capeval.tinycap import completion
from sisteer.capeval.tinycap import contracts as C

REPO = Path(__file__).resolve().parents[1]

#: the placeholder STORE pin (the run's WAVE_SHA) and the producer tree sha
FULL_SHA = "5555555566666666777777778888888899999999"
SHA7 = FULL_SHA[:7]
TASK = C.inspect_task("math")
N = 500
IDS = [f"math_{i:08x}" for i in range(N)]
CAP = C.MATH500_MAX_TOKENS_DEFAULT       # the one-pass cap (16,384) — the store follows the constant, never a literal

#: THE ONE ENUMERATED MATRIX (``scripts/sdfbase/math500/cells.py``) and the pinned
#: sampling rows (``configs/sdfbase/math500/sampling_rows.yaml``), READ here — the builder can only write a leg the
#: matrix names, under the row the run config pins, so the wrapper, the gates and the analysis cannot disagree
_CELLS_PY = REPO / "scripts" / "sdfbase" / "math500" / "cells.py"
_ROWS_YAML = REPO / "configs" / "sdfbase" / "math500" / "sampling_rows.yaml"


def _cells_module():
    spec = importlib.util.spec_from_file_location("m5t_cells_matrix_for_tests", _CELLS_PY)
    mod = importlib.util.module_from_spec(spec)
    # register BEFORE executing: cells.py defines dataclasses, and dataclass processing
    # resolves the class module through sys.modules (an unregistered module is None)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


CELLS_MODULE = _cells_module()
M5T_CELLS: list[dict[str, Any]] = [dict(r) for r in CELLS_MODULE.M5T_CELLS]
MODEL_KEYS: tuple[str, ...] = tuple(CELLS_MODULE.MODEL_KEYS)
FAMILY_BY_KEY: Mapping[str, str] = dict(CELLS_MODULE.FAMILY_BY_MODEL)
#: MATH-500's own served window (capability:gen_long_math, 17,664), read from the matrix module the wrapper runs
MAX_MODEL_LEN = int(CELLS_MODULE.GEOMETRY_L)
CELL_BARE = C.ROUTE_NONE_CELL
CELL_PROMPTED = C.CELL_PROMPTED_SAFETY
CELL_OBJECT = "constv2-mix-full__lora_r4_ce_minus_ctl"
CELLS: tuple[str, ...] = (CELL_BARE, CELL_PROMPTED, CELL_OBJECT)
assert CELLS == tuple(CELLS_MODULE.CELLS)
MODE_PRIMARY, MODE_GENERAL, MODE_BATTERY = "thinking_card", "general", "battery"
assert (MODE_PRIMARY, MODE_GENERAL, MODE_BATTERY) == tuple(CELLS_MODULE.SAMPLING_MODES)
#: which (model, mode) triples the matrix carries, derived from it
MODES_BY_MODEL: Mapping[str, tuple[str, ...]] = {
    key: tuple(mode for mode in (MODE_PRIMARY, MODE_GENERAL, MODE_BATTERY)
               if any(r["model_key"] == key and r["sampling_mode"] == mode for r in M5T_CELLS))
    for key in MODEL_KEYS}


def load_sampling_rows_yaml(path: Path = _ROWS_YAML) -> dict[str, dict[str, dict[str, Any]]]:
    """``{mode: {model_key: row}}`` from the run config's ``rows: {model_key: {mode: row}}``."""
    import yaml
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for key, by_mode in doc["rows"].items():
        for mode, row in by_mode.items():
            out.setdefault(str(mode), {})[str(key)] = dict(row)
    return out


#: the literal sampling rows (primary + general; battery), mode -> model key -> row, from the run config
SAMPLING_ROWS: Mapping[str, Mapping[str, Mapping[str, Any]]] = load_sampling_rows_yaml()
ROW_NUMERIC_FIELDS: tuple[str, ...] = tuple(CELLS_MODULE.SAMPLING_PARAM_KEYS)


def _fact_row(family: str) -> dict[str, Any]:
    """The family's render fact as the analysis' ``--render-facts`` table carries
    it: the CONTRACTS' measured tuples and split mode (the producer's
    ``render_certified`` compares against exactly these, so the fixture can be
    built through it), with the certification status the fixture ASSUMES
    (``certified`` — G-PARSER's output stands in; the contracts' own status is
    the producer's copy and is recorded, not asserted, by the reader)."""
    fact = C.FAMILY_RENDER_FACTS[family]
    row = dict(fact._asdict())
    row["off_ids"], row["on_ids"] = list(fact.off_ids), list(fact.on_ids)
    row["evidence_status"] = C.RENDER_EVIDENCE_CERTIFIED
    row["evidence"] = "synthetic: certified for the fixture (G-PARSER's output stands in)"
    return row


#: the family render facts: the contracts' header id tuples,
#: split mode and server parser, under a synthetic CERTIFIED status
#: (a family the CONTRACTS carry no fact for has no synthetic leg either — the `llama` rows are HELD on this tree for
#: exactly that reason, and a held row is never a store, so the fixture never needs its fact)
RENDER_FACTS: Mapping[str, Mapping[str, Any]] = {fam: _fact_row(fam) for fam in sorted(set(FAMILY_BY_KEY.values()))
                                                 if fam in C.FAMILY_RENDER_FACTS}

#: the reader constants the analysis exports — the producer's own names where the
#: producer exports them (pinned equal to the script by test)
REQUEST_EVIDENCE_FILENAME = C.REQUEST_EVIDENCE_FILENAME
EVIDENCE_RENDER_CERTIFIED = "render_certified"           # the driver's evidence.json key for contracts.render_certified()
EVIDENCE_RENDER_PROBE = "render_probe"                    # the driver's per-arm probe completions beside it
EVIDENCE_LENS_POSITION = "lens_position_attested"         # the driver's key for contracts.lens_position_attested()
EVIDENCE_LENS_ON_BITE = C.LENS_ON_BITE_KEY
MANIFEST_KEY_TRANSFER_PROVENANCE = C.MANIFEST_KEY_TRANSFER_PROVENANCE
MANIFEST_KEY_OOS_EXPERIMENT = C.MANIFEST_KEY_AUTHORISED_EXPERIMENT
MANIFEST_KEY_RESUME_ATTEMPTS = C.MANIFEST_KEY_RESUME_ATTEMPTS
MANIFEST_KEY_BINDING = "binding"                          # the driver's MANIFEST_KEY_BINDING (a driver-module constant)
ROUTE_KEY_LENS_SERVED_POSITIONS = C.LENS_SERVED_POSITIONS_KEY
IDENTITY_KEY_VERSION = C.IDENTITY_KEY_IDENTITY_VERSION
IDENTITY_KEY_THINKING_HEADER_IDS = C.IDENTITY_KEY_THINKING_HEADER_IDS
#: the driver's inline record of every run of a store (``resume_attempts``), one entry per run, the first included;
#: ``completed`` false until the run passes every gate (FC4 (3): written BEFORE the probes, so a preflight that refuses
#: leaves its attempt — and its labelled probe rows — attributable)
RESUME_ATTEMPT_FIELDS: tuple[str, ...] = ("ts", "tree_sha", "attempt", "completed")

# --------------------------------------------------------------------------- the standup / launch record (code review C2)
#: the standup record beside every served job, as a read-only pull of the cluster's ``serving/`` relocates it under the gen
#: root (``<gen_root>/serving/<jobdir>/math500/``): the manifest's ``routing_table`` names a synthetic ABSOLUTE cluster
#: wave root (no host, no home path — placeholders only); the analysis reads only its trailing
#: ``serving/<jobdir>/math500/routing_table.json`` and resolves the directory under the leg's gen root.
CLUSTER_WAVE_ROOT = "/cluster-shared/exp/math500_thinking_on_synthetic"
STANDUP_DIR_PARENT, STANDUP_TASK_SUBDIR = "serving", "math500"
STANDUP_ROUTING_TABLE, STANDUP_JOBS, STANDUP_ATTESTATION, LAUNCH_LOG = ("routing_table.json", "standup_jobs.jsonl",
                                                                        "steering_attestation.json", "vllm_server.log")
#: the standup's SERVING-CONFIG provenance record (mt_serving_up.SERVING_CONFIG_RECORD_FILE): the
#: sha256 of the config + manifest THAT leg served. A leg built WITHOUT it is the earlier shape, which the analysis must
#: still read (`serving_config_record: absent`, a null digest) — `serving_config=None` builds that leg.
STANDUP_SERVING_CONFIG = "serving_config.json"
#: the NATIVE (lora_native) engine's launch record — a lora-native standup writes it INSTEAD of the lens workers' attestation
#: (`serving_attestation.SERVING_ATTESTATION_FILE`, pinned equal by test); the analysis selects it by the manifest's
#: `route.serving_engine`, never by which file exists, and reads it through the STRICT reader
STANDUP_SERVING_ATTESTATION = "serving_attestation.json"
STANDUP_JOB_BASE, LAUNCH_PORT_BASE = 480000, 30000
#: the LAUNCH as the engine echoes it (``vllm serve``'s ``api_utils.py:273 non-default args: {...}`` — every CLI arg that
#: differs from its default), per model: the composer's per-MODEL choices (``vllm_server.build_server_launch``: prefix
#: caching off on a hybrid-recurrent target; the VLM's multimodal limits) — constant across the arms of one model, which is
#: what the analysis compares; dtype the composer's two-SI pin
LAUNCH_DTYPE = "bfloat16"
# llama8b (the Llama paper-rows lane's no-switch target, GSM8K only): prefix caching on (a dense Llama; the lane's standups), no extra args
LAUNCH_PREFIX_CACHING_BY_KEY: Mapping[str, bool] = {"gemma4e4b": True, "glm46vflash": True, "qwen35_4b": False, "nemotron3nano4b": False, "llama8b": True}
LAUNCH_EXTRA_ARGS_BY_KEY: Mapping[str, Mapping[str, Any]] = {"qwen35_4b": {"limit_mm_per_prompt": {"image": 0, "video": 0}}}
#: the ONE extension slot, per engine: the lens plugin's worker extension and the plugin-free native one
#: (``vllm_server.build_server_launch``; read off a real lora-native standup's echo, 2026-09-16)
LENS_WORKER_EXTENSION = "sisteer.capeval.serving.lens_worker.SisteerWorkerExtension"
NATIVE_WORKER_EXTENSION = "sisteer.capeval.serving.native_worker.NativeWorkerExtension"
#: the NATIVE engine's LoRA flags as the real echo carries them (rank-1 adapters, every adapter resident, o_proj only)
NATIVE_LORA_TARGET_MODULES: tuple[str, ...] = ("o_proj",)
NATIVE_MAX_LORA_RANK = 1
#: the snapshot's generation_config.json the engine reports it overrode its defaults with (the ``model.py`` WARNING the
#: recorder reads); None = the snapshot ships none (the engine's neutral defaults govern omitted keys)
GENERATION_CONFIG_BY_KEY: Mapping[str, Mapping[str, Any] | None] = {"gemma4e4b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95},
                                                                     "qwen35_4b": {"temperature": 0.7, "top_k": 20, "top_p": 0.8},
                                                                     "nemotron3nano4b": None, "glm46vflash": None,
                                                                       # the Nous Llama-3.1-8B-Instruct card row (generation_config.json: do_sample, 0.6 / 0.9), the no-switch family
                                                                       "llama8b": {"temperature": 0.6, "top_p": 0.9}
}


def standup_job_id(model_key: str, cell: str, mode: str, *, tree_sha: str) -> str:
    """The synthetic Slurm job id of a leg's standup — one per (model, cell, mode, producer tree)."""
    return str(STANDUP_JOB_BASE + int(hashlib.sha256(f"{model_key}|{cell}|{mode}|{tree_sha}".encode()).hexdigest()[:6], 16) % 90000)


def standup_jobdir(model_key: str, cell: str, mode: str, job: str) -> str:
    """``m5t_<model>_<cell slug>_<mode>_<job>`` — the launcher's standup directory name."""
    return f"m5t_{model_key}_{cell_slug(cell)}_{mode}_{job}"


def launch_port(job: str) -> int:
    """The port a synthetic standup bound its server to — ONE rule for the attestation and the log's echo (both stores)."""
    return LAUNCH_PORT_BASE + int(job) % 20000


def cluster_routing_table_path(jobdir: str) -> str:
    """The ABSOLUTE cluster path the manifest's ``routing_table`` / ``binding.routing_table`` name."""
    return f"{CLUSTER_WAVE_ROOT}/{STANDUP_DIR_PARENT}/{jobdir}/{STANDUP_TASK_SUBDIR}/{STANDUP_ROUTING_TABLE}"


def served_seqs(model_key: str, max_model_len: int) -> int:
    """The S the standup serves ``model_key`` at window L: the MODEL budget's S under the resolver's own rule
    (``sisteer.capeval.serving.geometry.ModelBudget.seqs_at``: ``budget // L``, the hybrid caps applied) — what the
    analysis' arm contract holds every leg to. The store's legs are served at it, the client width equals
    it, and the launch echo carries it, exactly as the driver records a production leg."""
    from sisteer.capeval.serving import geometry as SG
    return int(SG.budget_for(model_key).entry.seqs_at(int(max_model_len)))


def geometry_block(model_key: str, *, eval_key: str, max_model_len: int) -> dict[str, Any]:
    """The manifest's ``geometry`` block in the PRODUCER's shape: the standup's ``ResolvedGeometry.as_dict()`` (eval, the
    window from the table entry, S from the model budget, ``batched_tokens``, ``budget`` + its source, ``identity`` — the
    resolver's own attestation that S is the budget's S at L) as the driver copies it off endpoint.json, plus the
    ``pass`` field (null: no measurement pass)."""
    from sisteer.capeval.serving import geometry as SG
    budget = SG.budget_for(model_key)
    resolved = SG.ResolvedGeometry(eval=eval_key, max_model_len=int(max_model_len), max_num_seqs=served_seqs(model_key, max_model_len),
                                   window_source="table", seqs_source="model_budget", entry=None, model_key=model_key, budget=budget)
    return {**resolved.as_dict(), C.GEOMETRY_PASS_FIELD: None}


@dataclass
class LoRAModulePath:
    """vLLM's ``--lora-modules`` entry as the pinned engine echoes it at startup: a dataclass whose ``repr`` —
    ``LoRAModulePath(name='…', path='…', base_model_name=None, is_3d_lora_weight=False)`` — is what sits inside the native
    echo's ``lora_modules`` list (read off a real lora-native standup's log, 2026-09-16; field ORDER as echoed). The
    fixture renders the echo with ``repr()``, so a native log here is byte-shaped like the real one — NOT a Python literal,
    which is exactly what the analysis' strict echo reader must admit for this one constructor and refuse for any other."""
    name: str
    path: str
    base_model_name: str | None = None
    is_3d_lora_weight: bool = False


def launch_echo_args(model_key: str, *, revision: str, port: int, max_model_len: int, max_num_seqs: int, reasoning_parser: str | None,
                     log_requests: bool = False, jobdir: str = "job", native: bool = False,
                     lora_modules: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The ``non-default args`` dict the API server echoes at startup for THIS model's composed launch (key order the
    server's): the per-model choices above plus the per-job instance values (port, template path, geometry). ``native``
    composes the lora_native engine's echo as a real native standup's log carries it: the ``lora_modules`` list of
    :class:`LoRAModulePath` (``{serving name: adapter dir}``, in ``--lora-modules`` order) right after the model tag, the
    native worker extension, NO ``enforce_eager`` (graphs on is the default the echo omits), ``enable_lora`` and the LoRA
    config flags (``max_loras`` / ``max_cpu_loras`` = the adapter count, ``max_lora_rank`` 1, ``lora_target_modules``)."""
    hf, _rev = model_pin(model_key)
    args: dict[str, Any] = {"model_tag": hf}
    if native:
        modules = dict(lora_modules or {})
        if not modules:
            raise ValueError("a native launch echo names at least one --lora-modules entry")
        args["lora_modules"] = [LoRAModulePath(name=n, path=str(d)) for n, d in modules.items()]
    args.update({"chat_template": f"{CLUSTER_WAVE_ROOT}/{STANDUP_DIR_PARENT}/{jobdir}/{STANDUP_TASK_SUBDIR}/chat_template.jinja",
                 "chat_template_content_format": "string", "default_chat_template_kwargs": {"enable_thinking": False},
                 "host": "127.0.0.1", "port": int(port), "middleware": ["sisteer.capeval.serving.vllm_auth.SisteerTokenAuthMiddleware"],
                 "model": hf, "dtype": LAUNCH_DTYPE, "revision": revision, "max_model_len": int(max_model_len)})
    if not native:
        args["enforce_eager"] = True
    if reasoning_parser:
        args["reasoning_parser"] = reasoning_parser
    args.update({"worker_extension_cls": NATIVE_WORKER_EXTENSION if native else LENS_WORKER_EXTENSION, "gpu_memory_utilization": 0.9,
                 "enable_prefix_caching": LAUNCH_PREFIX_CACHING_BY_KEY[model_key], **LAUNCH_EXTRA_ARGS_BY_KEY.get(model_key, {})})
    if native:
        n = len(args["lora_modules"])
        args.update({"enable_lora": True, "max_loras": n, "max_lora_rank": NATIVE_MAX_LORA_RANK, "max_cpu_loras": n,
                     "lora_target_modules": list(NATIVE_LORA_TARGET_MODULES)})
    args.update({"max_num_batched_tokens": int(max_model_len) * int(max_num_seqs), "max_num_seqs": int(max_num_seqs), "async_scheduling": False})
    if log_requests:
        args["enable_log_requests"] = True
    return args


def launch_log_text(args: Mapping[str, Any], *, generation_config: Mapping[str, Any] | None, log_requests: bool = False) -> str:
    """A ``vllm_server.log`` in the pinned server's shape: the startup echo of the launch (``non-default args``), the
    request-logging attestation line when the flag is set, and the generation-config override WARNING when the snapshot
    ships one (its absence is the engine's positive evidence that none governs — the recorder's reading). A native echo's
    ``lora_modules`` render through :class:`LoRAModulePath`'s repr, as the real log carries them."""
    lines = [f"(APIServer pid=1) INFO 01-01 00:00:00 [api_utils.py:273] non-default args: {dict(args)!r}"]
    if log_requests:
        lines.append("(APIServer pid=1) INFO 01-01 00:00:01 [request_logger.py:60] `--enable-log-requests` is set but the minimum log level is "
                     "higher than DEBUG. Only limited information will be logged for each request.")
    lines.append("(EngineCore pid=2) INFO 01-01 00:00:02 [core.py:114] Initializing a V1 LLM engine (v0.25.1) with config: "
                 f"model={args.get('model')!r}, dtype=torch.{args.get('dtype')}, max_seq_len={args.get('max_model_len')}")
    if generation_config is not None:
        lines.append("(APIServer pid=1) WARNING 01-01 00:00:03 [model.py:1528] Default vLLM sampling parameters have been overridden by the "
                     f"model's `generation_config.json`: `{dict(generation_config)!r}`. If this is not intended, please relaunch vLLM instance "
                     "with `--generation-config vllm`.")
    return "\n".join(lines) + "\n"


#: the native engine's CUDA-graph facts a strict record carries (a graphs-on engine: `enforce_eager` False, a non-NONE mode)
NATIVE_CUDAGRAPH_MODE = "FULL_AND_PIECEWISE"
NATIVE_CUDAGRAPH_CAPTURE_SIZES: tuple[int, ...] = (1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88)
#: the adapter dirs a synthetic native standup binds its serving names to (the manifest's cluster wave root, `adapters/<model>/`)
NATIVE_ADAPTER_ROOT = f"{CLUSTER_WAVE_ROOT}/adapters"


def native_serving_attestation_doc(*, job: str, port: int, model: str, revision: str, argv_sha256: str, nonce: str,
                                   adapters: Mapping[str, Mapping[str, Any]], max_model_len: int, max_num_seqs: int,
                                   n_layers: int = 32, hostname: str = "TESTHOST", host: str = "127.0.0.1",
                                   attested_at: float = 0.0, server_pid: int = 1) -> dict[str, Any]:
    """A STRICT-VALID ``serving_attestation.json`` (schema 2) — exactly ``serving_attestation.REQUIRED_TOP_KEYS_V2``, every
    block keyed as the strict reader ``read_serving_attestation`` requires (``kind`` lora_native, ``name_binding.asserted``
    True, ``engine.enforce_eager`` False with a non-NONE CUDA-graph mode, the LoRA config, the execution modules with their
    strides, the schema-2 adapter keys, the worker adapters, an ``id_name_map`` consistent with the adapters'
    ``lora_int_id``, a disarmed residual slot) — with synthetic values. ``adapters`` is ``{serving name: {"lora_int_id",
    "r", "dir", "petri_intervention_hash", "petri_zero_adapter", "petri_source_kind"}}`` in ``--lora-modules`` order; the
    binding fields the analysis reads (``slurm_job_id``, ``port``, ``argv_sha256``) sit at the top level as in the real
    record. ONE builder for every fixture that writes a native record (the GSM8K synthetic store, the checked-in native
    card fixture), so the tests' native records are the record the strict reader admits — pinned by test."""
    def _h(tag: str) -> str:
        return hashlib.sha256(f"{tag}|{job}".encode()).hexdigest()
    module_names = [f"model.layers.{i}.self_attn.o_proj" for i in range(int(n_layers))]
    module_keys = [f"L{i}.self_attn.o_proj" for i in range(int(n_layers))]
    stride = 4096
    adapters_block: dict[str, Any] = {}
    worker_adapters: list[dict[str, Any]] = []
    for name, spec in adapters.items():
        zero = bool(spec.get("petri_zero_adapter"))
        adapters_block[name] = {"dir": str(spec["dir"]), "config_sha256": _h(f"config|{name}"), "weights_sha256": _h(f"weights|{name}"),
                                "fingerprint_bf16": _h(f"fp|{name}"), "r": int(spec.get("r", 1)), "lora_int_id": int(spec["lora_int_id"]),
                                "module_keys": list(module_keys), "petri_intervention_hash": spec.get("petri_intervention_hash"),
                                "petri_zero_adapter": zero, "models_root": str(spec["dir"]),
                                "petri_source_kind": spec.get("petri_source_kind"), "residual_fingerprint": spec.get("residual_fingerprint"),
                                "position_mode": spec.get("position_mode")}
        worker_adapters.append({"id": int(spec["lora_int_id"]), "rank": int(spec.get("r", 1)), "n_modules": len(module_keys),
                                "module_keys": list(module_keys), "tensor_dtype": "torch.bfloat16", "fingerprint_bf16": _h(f"fp|{name}"),
                                "execution_bindings": {m: m for m in module_names}})
    id_name_map = {str(b["lora_int_id"]): n for n, b in adapters_block.items()}
    n = len(adapters_block)
    return {"kind": "lora_native", "schema_version": 2, "model": model, "revision": revision, "attested_at": float(attested_at),
            "server_pid": int(server_pid), "hostname": hostname, "slurm_job_id": str(job), "argv_sha256": argv_sha256, "host": host,
            "port": int(port), "nonce": nonce,
            "expected": {"lora_loaded_count": n, "max_model_len": int(max_model_len), "max_num_seqs": int(max_num_seqs),
                         "max_num_batched_tokens": int(max_model_len) * int(max_num_seqs), "enforce_eager": False,
                         "cudagraph_mode_is_none": False, "lens_plugin_imported": False},
            "expected_native_lora": {"n_adapters": n, "adapters": {name: {"lora_int_id": b["lora_int_id"], "fingerprint_bf16": b["fingerprint_bf16"],
                                                                          "r": b["r"], "module_keys": list(module_keys)}
                                                                   for name, b in adapters_block.items()}},
            "scheduler_geometry": {"n_workers": 1, "rpc_method": "sisteer_scheduler_config",
                                   "worker_reports": [{"max_model_len": int(max_model_len), "max_num_seqs": int(max_num_seqs),
                                                       "max_num_batched_tokens": int(max_model_len) * int(max_num_seqs)}]},
            "engine": {"enforce_eager": False, "cudagraph_mode": NATIVE_CUDAGRAPH_MODE, "cudagraph_specialize_lora": True,
                       "cudagraph_capture_sizes": list(NATIVE_CUDAGRAPH_CAPTURE_SIZES), "max_cudagraph_capture_size": max(NATIVE_CUDAGRAPH_CAPTURE_SIZES),
                       "lens_plugin_imported": False, "enable_prefix_caching": True},
            "lora_config": {"max_loras": n, "max_lora_rank": NATIVE_MAX_LORA_RANK, "max_cpu_loras": n, "lora_dtype": "torch.bfloat16",
                            "fully_sharded_loras": False, "target_modules": None},
            "adapters": adapters_block, "worker_adapters": [worker_adapters], "id_name_map": id_name_map,
            "name_binding": {"asserted": True, "basis": "composed --lora-modules order == frontend lora_id_counter order; per-name fingerprint + "
                                                        "rank + module_keys + /v1/models root all held (synthetic)"},
            "execution_modules": {"module_keys": list(module_keys), "module_names": list(module_names), "n_modules": len(module_names), "other": [],
                                  "strides": {m: {"in": stride, "out": stride} for m in module_names}, "stride_max": stride},
            "residual_slot": {"armed": False, "rpc_method": "sisteer_residual_slot_report", "n_workers": 1,
                              "expected": {"armed": False, "adapters": {}}, "worker_reports": [{"armed": False}]}}


def native_adapter_specs(model_key: str, cell_name: str, *, intervention_hash: str, source_kind: str | None = None,
                         zero_name: str = "none__lora0") -> dict[str, dict[str, Any]]:
    """The two adapters a synthetic native standup serves, in ``--lora-modules`` order: the engine's ZERO adapter (the
    control route, id 1) and the served cell's (id 2) — ``{name: spec}`` for :func:`native_serving_attestation_doc` and
    ``{name: dir}`` (``native_lora_modules``) for the echo."""
    root = f"{NATIVE_ADAPTER_ROOT}/{model_key}"
    return {zero_name: {"lora_int_id": 1, "r": 1, "dir": f"{root}/{zero_name}", "petri_intervention_hash": None, "petri_zero_adapter": True,
                        "petri_source_kind": None},
            cell_name: {"lora_int_id": 2, "r": 1, "dir": f"{root}/{cell_name}", "petri_intervention_hash": intervention_hash,
                        "petri_zero_adapter": False, "petri_source_kind": source_kind}}


def native_lora_modules(specs: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    """``{serving name: adapter dir}`` in ``--lora-modules`` order, for :func:`launch_echo_args` ``native=True``."""
    return {name: str(spec["dir"]) for name, spec in specs.items()}


def write_launch_record(gen_root: Path, jobdir: str, *, job: str, port: int, args: Mapping[str, Any], generation_config: Mapping[str, Any] | None,
                        tree_sha: str, model_key: str, max_model_len: int, max_num_seqs: int, log_requests: bool = False,
                        serving_config: Mapping[str, Any] | None = None, native: bool = False,
                        adapters: Mapping[str, Mapping[str, Any]] | None = None) -> Path:
    """The standup record under ``<gen_root>/serving/<jobdir>/math500/``: the ENGINE's launch record bound to the Slurm job
    and the bound port — the lens workers' attestation, or with ``native`` the strict-valid native serving attestation
    (:func:`native_serving_attestation_doc` over ``adapters``, :func:`native_adapter_specs`' shape) INSTEAD of it, as a
    lora-native standup writes exactly one engine's record — ``standup_jobs.jsonl`` binding the job to the tree, a minimal
    routing table, and the server log carrying the launch echo (:func:`launch_log_text`; pass ``args`` from
    ``launch_echo_args(native=True, ...)`` for a native leg). Returns the directory."""
    hf, rev = model_pin(model_key)
    d = Path(gen_root) / STANDUP_DIR_PARENT / jobdir / STANDUP_TASK_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    argv_sha256 = hashlib.sha256(f"synthetic argv {jobdir}".encode()).hexdigest()
    nonce = hashlib.sha256(f"nonce|{jobdir}".encode()).hexdigest()[:32]
    if native:
        if not adapters:
            raise ValueError("a native launch record binds at least one adapter (native_adapter_specs)")
        attestation = native_serving_attestation_doc(job=str(job), port=int(port), model=hf, revision=rev, argv_sha256=argv_sha256, nonce=nonce,
                                                     adapters=adapters, max_model_len=int(max_model_len), max_num_seqs=int(max_num_seqs))
        (d / STANDUP_SERVING_ATTESTATION).write_text(json.dumps(attestation, indent=1), encoding="utf-8")
    else:
        attestation = {"slurm_job_id": str(job), "hostname": "TESTHOST", "host": "127.0.0.1", "port": int(port), "model": hf, "n_workers": 1,
                       "attested_at": 0.0, "argv_sha256": argv_sha256[:16],
                       "nonce": nonce, "registry_index_sha16": None,
                       "scheduler_geometry": {"max_model_len": int(max_model_len), "max_num_seqs": int(max_num_seqs)}}
        (d / STANDUP_ATTESTATION).write_text(json.dumps(attestation, indent=1), encoding="utf-8")
    jobs = [{"job_id": str(job), "node": "TESTHOST", "key": model_key, "view": "main", "mode": "attacks", "eval": "capability:gen_long_math",
             "sha": tree_sha, "lens_reduction_mode": ("stacked" if not native else None), "max_num_seqs": int(max_num_seqs), "max_model_len": int(max_model_len),
             "cluster": "synthetic", "written_at": "2026-01-01T00:00:00Z", **({"engine": "lora_native", "serving_path": "lora_native"} if native else {})}]
    (d / STANDUP_JOBS).write_text("".join(json.dumps(r) + "\n" for r in jobs), encoding="utf-8")
    (d / STANDUP_ROUTING_TABLE).write_text(json.dumps({"base_model": hf, "base_revision": rev, "serving_mode": "interleaved", "version": 1}, indent=1),
                                           encoding="utf-8")
    (d / LAUNCH_LOG).write_text(launch_log_text(args, generation_config=generation_config, log_requests=log_requests), encoding="utf-8")
    if serving_config is not None:
        (d / STANDUP_SERVING_CONFIG).write_text(json.dumps(dict(serving_config), indent=1), encoding="utf-8")
    return d


#: the primary scorer of the row task, as the runner names it (``inspect_served.PRIMARY_SCORER``)
TASK_NAME = TASK.removeprefix(C.INSPECT_TASK_PREFIX)
PRIMARY_SCORER: str = isv.PRIMARY_SCORER[TASK_NAME]
#: the per-item shapes THIS tree's writer admits. The producer's ``outcome_record()`` stamps
#: ``sampling_ref`` on every item of an identity-v2 run; the harness fix names the admitted shapes in
#: ``contracts.TASK_OUTCOME_ITEM_FIELD_SETS`` (v1 without the stamp, v2 with it). A tree before it has only
#: ``TASK_OUTCOME_ITEM_FIELDS`` — the analysis tests key their xfail on exactly this lookup, so the requirement
#: flips on by itself when the harness fix merges
OUTCOME_ITEM_FIELD_SETS_NAME = "TASK_OUTCOME_ITEM_FIELD_SETS"
OUTCOME_ITEM_FIELD_SETS: tuple[tuple[str, ...], ...] = tuple(
    tuple(fs) for fs in getattr(C, OUTCOME_ITEM_FIELD_SETS_NAME, (C.TASK_OUTCOME_ITEM_FIELDS,)))
STAMPED_OUTCOME_ITEMS_ADMITTED: bool = any(C.SAMPLING_REF_FIELD in fs for fs in OUTCOME_ITEM_FIELD_SETS)
#: ``build_store(outcome_sampling_ref=...)``: AUTO stamps the record iff this tree's writer admits the stamped
#: shape (the fixture mirrors the producer the tree can run to completion); ALWAYS stamps regardless (the
#: producer -> ``finish_task`` -> analysis test, which xfails until the harness fix lands); ``None`` writes the v1
#: record; any other string is stamped verbatim (a negative test's wrong digest)
OUTCOME_STAMP_AUTO = "auto"
#: ``write_inspect_log(max_tokens=...)``: the completion cap the log's PLAN records (``log.plan.config.max_tokens``
#: — what Inspect writes at log start from the eval-level ``max_tokens`` the runner passes ``eval_async``, and what the
#: producer's selection holds a retained log to). STORE (default) records the store's own summary cap, as the
#: producer's log of that store does; an int records that cap (a copied-in log generated at another cap); ``None``
#: records no cap at all (a plan carrying a bare ``GenerateConfig()`` — an earlier or foreign writer's log).
LOG_CAP_STORE = "store"
OUTCOME_STAMP_ALWAYS = "always"
#: the steering-bites probe: ``unified_sweep_run._SERVED_PROBE_PROMPTS`` has five prompts, sent on BOTH legs
N_BITES_PROMPTS = 5
BITES_PROBE_MAX_TOKENS = 48
RENDER_PROBE_MAX_TOKENS = 64
LENS_ON_BITE_MAX_TOKENS = 256

PROXY_URL = "http://127.0.0.1:19000"
LEDGER_URL = "http://127.0.0.1:19001"
PROMPT_BASE = 400
DECODE_TOK_S = 20.0
TS0 = "2026-01-01T00:00:00Z"
#: the client's request timeout — one same-cap re-send starts this long after the request it replaces
REDRAW_TIMEOUT_S = 600
#: an `error_retries` message of each class the analysis recognises (the timeout ladder's is the real one's shape)
FAILED_ATTEMPT_TIMEOUT_MESSAGE = "RetryError(<Future at 0x7f00deadbeef state=finished raised APITimeoutError>)"
FAILED_ATTEMPT_REFUSAL_MESSAGE = ("tinycap provider: inspect/math item 'math_00000000' sample 0: the response carries neither "
                                  "content nor reasoning under the composed reasoning parser 'glm45' (stop_reason='stop', "
                                  "output_tokens=12) — cannot tell an empty answer from text the parser lost; refusing to score it")
#: a provider refusal raised BEFORE any request (`collapse_system_messages`): the attempt made no call
FAILED_ATTEMPT_PRE_WIRE_MESSAGE = "tinycap provider: empty message list — nothing to send"


def _ts_plus(ts: str, seconds: int) -> str:
    """``ts`` (the fixture's ``...Z`` ISO form) advanced by ``seconds``."""
    import datetime as _dt
    base = _dt.datetime.fromisoformat(ts)
    return (base + _dt.timedelta(seconds=seconds)).strftime("%Y-%m-%dT%H:%M:%SZ")


def cell_slug(cell: str) -> str:
    """The out-root directory token of a cell (``__`` -> ``--``)."""
    return cell.replace("__", "--")


def _hash8(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:8]


@cache
def _pins_doc() -> Mapping[str, Any]:
    return json.loads((REPO / "configs" / "sdfbase_full22_pins.json").read_text(encoding="utf-8"))


def object_pin(model_key: str) -> str:
    """The fitted object's artifact digest from the battery's PIN file (an
    untouched, checked-in input the analysis reads by the same path)."""
    return str(_pins_doc()[model_key][f"derived__{CELL_OBJECT}"]["object_sha256"])


def staged_object_pin(model_key: str, staged_id: str) -> str:
    """Any staged object's digest from the pin file (the eight-arm extension's
    ``base__<cell>`` LoRAs beside the published ``derived__…_minus_ctl``)."""
    return str(_pins_doc()[model_key][staged_id]["object_sha256"])


#: the sha256[:16] of each prompted cell's SI text as the driver files it (``resolved_config.system_sha``) — the
#: analysis pins the same two digests (``PROMPTED_SI_PREFIX_SHA16``), held equal by test
SI_PREFIX_SHA16: Mapping[str, str] = {"safety_refusal": "ac3a58f61d53cf24", "si_constitution": "21e06f482a6559b3"}
#: the served chat-template digest per family the fixture's manifests carry (``provenance.serving.chat_template_hash``)
CHAT_TEMPLATE_HASH_BY_FAMILY: Mapping[str, str] = {"gemma4": "0a2c8073c878ab1d", "qwen3_5": "a4aee8afcf2e0711",
                                                   "nemotron_h": "ab7813c3abdd9cb6", "glm4v": "5a5a5a5a5a5a5a5a"}


def si_of(cell: str) -> str | None:
    """The SI the driver files as ``provenance.si_id``: a prompted cell's SI, a fitted cell's corpus (the token before
    ``__``), none on the bare route."""
    if cell.startswith(C.PROMPTED_CELL_PREFIX):
        return cell[len(C.PROMPTED_CELL_PREFIX):]
    if cell == CELL_BARE:
        return None
    return cell.split("__")[0]


def serving_identity_of(fitted: bool) -> str:
    """The route's serving identity as the driver files it (``route.serving_identity``): the lens route on a fitted cell,
    passthrough otherwise — the per-row ARM CONTRACT reads it (plan P5 (i′))."""
    return "proxy_vllm+lens+interleaved" if fitted else "proxy_vllm+passthrough+interleaved"


def fake_git_for(known, diffs, *, diff_args):
    """A stand-in for an analysis' ``_git_output`` (both eight-arm test modules use it): ``rev-parse --verify --quiet
    <ref>^{commit}`` resolves a known sha or a unique prefix; the diff enumeration (``diff_args`` = the module's
    ``GIT_DIFF_NAMES_ARGS``) returns the configured file list NUL-terminated; anything else is an unexpected call."""

    def fake(repo_root, *args):
        if args[:3] == ("rev-parse", "--verify", "--quiet"):
            ref = args[3].removesuffix("^{commit}")
            hits = [sha for sha in known if sha.startswith(ref)]
            if len(hits) != 1:
                raise subprocess.CalledProcessError(1, ["git", *args])
            return hits[0] + "\n"
        if args[:len(diff_args)] == tuple(diff_args) and args[-1] == "--":
            return "".join(f + "\0" for f in diffs[(args[-3], args[-2])])
        raise AssertionError(f"unexpected git call {args}")

    return fake


@cache
def model_pin(model_key: str) -> tuple[str, str]:
    """``(hf id, revision)`` from the battery's own target config (top-level
    ``model:`` / ``model_revision:``), the same file the analysis reads — a
    checked-in input, read once per process."""
    text = (REPO / "configs" / f"sdfbase_full22_{model_key}.yaml").read_text(encoding="utf-8")
    hf = re.search(r"^model:\s*(\S+)\s*$", text, re.MULTILINE)
    rev = re.search(r"^model_revision:\s*(\S+)\s*$", text, re.MULTILINE)
    assert hf and rev, model_key
    return hf.group(1), rev.group(1)


@cache
def vmodel_of(cell: str) -> str:
    return C.cell_spellings(cell, declared=False).vmodel


@cache
def harness_id_of(cell: str) -> str:
    return C.cell_spellings(cell, declared=False).harness


def is_fitted(cell: str) -> bool:
    return C.is_lens_cell(cell)


def matrix_row(model_key: str, cell: str, mode: str) -> dict[str, Any]:
    """The matrix's row for a (model, cell, mode) — a triple the matrix does not
    name refuses (the module's CLI-shaped ``SystemExit`` surfaces as a ``KeyError``)."""
    try:
        return dict(CELLS_MODULE.row_for(model_key, cell, mode))
    except SystemExit as e:
        raise KeyError(str(e)) from None


def tuple_name(model_key: str, mode: str) -> str:
    return str(CELLS_MODULE.tuple_name(model_key, mode))


def cells_matrix() -> list[dict[str, Any]]:
    """The PUBLISHED matrix's 24 rows (``M5T_CELLS`` restricted to the published wave), copied — the design the
    published-shape tests run on. A tree whose ``cells.py`` appends extension rows (wave ``ext1``, ...; the eight-arm
    extension) keeps this at the published 24: those rows are :func:`ext_rows`' business."""
    return [dict(r) for r in M5T_CELLS if str(r.get("wave") or "published") == "published"]


def _write_json(path: Path, doc: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return path


def write_cells_json(path: Path, rows: Sequence[Mapping[str, Any]] | None = None) -> Path:
    return _write_json(path, list(rows if rows is not None else cells_matrix()))


def write_sampling_rows(path: Path, rows: Mapping[str, Mapping[str, Mapping[str, Any]]] | None = None) -> Path:
    return _write_json(path, rows if rows is not None else SAMPLING_ROWS)


def write_render_facts(path: Path, facts: Mapping[str, Mapping[str, Any]] | None = None) -> Path:
    return _write_json(path, facts if facts is not None else RENDER_FACTS)


def sampling_row_of(row: Mapping[str, Any]) -> C.SamplingRow:
    """The run config's row as the contracts' ``SamplingRow`` (what the driver
    resolves and hashes)."""
    return C.SamplingRow(temperature=float(row["temperature"]),
                         top_p=(None if row.get("top_p") is None else float(row["top_p"])),
                         top_k=(None if row.get("top_k") is None else int(row["top_k"])),
                         source=str(row.get("source") or ""),
                         **{f: (None if row.get(f) is None else float(row[f])) for f in C.SAMPLING_OPTIONAL_ROW_FIELDS})


def sampling_block_of(model_key: str, mode: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """``manifest["sampling"]`` exactly as the driver's ``_sampling_block`` builds
    it: the hashed identity (``contracts.sampling_identity``) plus the mode,
    the model key and the row's source prose."""
    srow = sampling_row_of(row)
    block = {"mode": mode, "model_key": model_key, **C.sampling_identity(srow, {TASK: 1}), "source": srow.source}
    assert set(block) == set(C.sampling_block_fields(srow))
    return block


def expected_sampling_ref(model_key: str, mode: str,
                          sampling_rows: Mapping[str, Mapping[str, Mapping[str, Any]]] = SAMPLING_ROWS) -> str:
    """The producer's digest of the leg's sampling block (``contracts.sampling_ref``)
    — what every row and, on an identity-v2 run, every outcome item carries."""
    return C.sampling_ref(sampling_block_of(model_key, mode, sampling_rows[mode][model_key]))


def default_outcome_stamp(model_key: str, mode: str) -> str | None:
    """What ``build_store`` stamps on the outcome items by default: the digest iff
    this tree's writer admits the stamped shape, else nothing (the v1 record)."""
    return expected_sampling_ref(model_key, mode) if STAMPED_OUTCOME_ITEMS_ADMITTED else None


def _resolve_outcome_stamp(arg: str | None, digest: str) -> str | None:
    if arg == OUTCOME_STAMP_AUTO:
        return digest if STAMPED_OUTCOME_ITEMS_ADMITTED else None
    if arg == OUTCOME_STAMP_ALWAYS:
        return digest
    return arg


def task_plan(row: Mapping[str, Any], ids: Sequence[str], fact: Mapping[str, Any]) -> isv.TaskPlan:
    """The runner's plan for this experiment's MATH-500 pass, as ``outcome_record``
    and ``think_split_counts`` read it: the row task and its selection, k = 1, the
    cap, the primary scorer, the frozen full count, the sampling row, the ON arm,
    no retry, the family's split mode."""
    srow = sampling_row_of(row)
    return isv.TaskPlan(row_task=TASK, name=TASK_NAME, task=None, sample_ids=list(ids), samples_per_item=1, limit=None,
                        max_tokens=CAP, primary_scorer=PRIMARY_SCORER, expected_n=len(ids), n_run=len(ids),
                        temperature=srow.temperature, top_p=srow.top_p, top_k=srow.top_k, item_set=None,
                        thinking=C.THINKING_ON, retry_max_tokens=0, grader_max_tokens=None,
                        think_split_mode=fact["split_mode"], family=fact["family"],
                        render_evidence_status=fact["evidence_status"],
                        **{f: getattr(srow, f) for f in C.SAMPLING_OPTIONAL_ROW_FIELDS})


def out_root_of(gen_root: Path, model_key: str, cell: str, mode: str) -> Path:
    return Path(gen_root) / "controls" / model_key / cell_slug(cell) / mode


def resume_attempts_of(n_attempts: int, tree_sha: str = FULL_SHA, incomplete: Iterable[int] = ()) -> list[dict[str, Any]]:
    """The driver's ``resume_attempts`` record after ``n_attempts`` runs of a
    store (the first run included): ``{ts, tree_sha, attempt, completed}`` per
    run; ``incomplete`` names the 0-based runs whose preflight refused
    (``completed`` false — the entry is on record, the run sent probes only)."""
    off = set(incomplete)
    return [{"ts": f"2026-01-01T{k:02d}:00:00Z", "tree_sha": tree_sha, "attempt": k + 1, "completed": k not in off}
            for k in range(n_attempts)]


def standup_endpoint(model_key: str, cell: str, positions: Sequence[str] = ("all",),
                     policy: str = "always", *, pin: str | None = None) -> dict[str, Any]:
    """The standup record (``endpoint.json`` / the routing table's identity block)
    predicate (B′) reads for the object cell: the empty declarations map with its
    null error, the served-positions map, the object digest twice, the attested
    resumed-prefill policy — the keys ``contracts.lens_position_attested`` names.
    ``pin`` (eight-arm extension): the cell's OWN object digest; default the
    published minus_ctl object's."""
    harness, pin = harness_id_of(cell), (pin or object_pin(model_key))
    return {C.PROXY_IDENTITY_SERVE_DECLARATIONS: {}, C.PROXY_IDENTITY_SERVE_DECLARATIONS_ERROR: None,
            C.LENS_SERVED_POSITIONS_KEY: {harness: list(positions)},
            C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256: pin,
            C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {harness: pin},
            C.ENDPOINT_KEY_RESUMED_PREFILL_POLICY_CELLS: {harness: policy}}


def think_shaped_completion(fact: Mapping[str, Any], answer: str, *, finish_reason: str = "stop") -> dict[str, Any]:
    """A completion carrying the family's think SHAPE (``render_certified`` rule
    3 / ``lens_on_bite_verdict``): ``record_only`` a reasoning channel,
    ``implicit_open`` the prompt-opened block closed by the close tag,
    ``explicit_tags`` the model-emitted open + close tags."""
    if fact["split_mode"] == C.THINK_SPLIT_RECORD_ONLY:
        return {"content": answer, "reasoning": "reasoning", "finish_reason": finish_reason}
    if fact["split_mode"] == C.THINK_SPLIT_IMPLICIT_OPEN:
        return {"content": f"reasoning{fact['think_close']}{answer}", "reasoning": None, "finish_reason": finish_reason}
    return {"content": f"{fact['think_open']}reasoning{fact['think_close']}{answer}", "reasoning": None,
            "finish_reason": finish_reason}


def render_certified_record(family: str, fact: Mapping[str, Any]) -> dict[str, Any]:
    """Predicate (A)'s record THROUGH ``contracts.render_certified`` — the live
    tuples, both probe regimes, the ON probe's think shape."""
    return C.render_certified(family=family, header_ids_by_arm={"off": list(fact["off_ids"]), "on": list(fact["on_ids"])},
                              measured_live=True,
                              probe_regimes={C.THINKING_ON: C.RENDER_REGIME_THINKING_ON,
                                             C.THINKING_OFF: C.RENDER_REGIME_THINKING_OFF},
                              on_completion=think_shaped_completion(fact, "Blue."))


def render_probe_record(fact: Mapping[str, Any]) -> dict[str, Any]:
    """The driver's ``render_probe`` block: the (A) probe completion per arm."""
    return {C.THINKING_OFF: {"content": "Blue.", "reasoning": None, "finish_reason": "stop",
                             "render_regime": C.RENDER_REGIME_THINKING_OFF},
            C.THINKING_ON: {**think_shaped_completion(fact, "Blue."), "render_regime": C.RENDER_REGIME_THINKING_ON}}


def lens_position_record(model_key: str, cell: str, *, pin: str | None = None) -> dict[str, Any]:
    """Predicate (B′)'s record THROUGH ``contracts.lens_position_attested`` over
    the producer-shaped standup record (``pin``: the cell's own object digest)."""
    pin = pin or object_pin(model_key)
    return C.lens_position_attested(standup_endpoint(model_key, cell, pin=pin), cell=cell, expected_sha256=pin,
                                    model_key=model_key)


def lens_on_bite_record(cell: str, fact: Mapping[str, Any]) -> dict[str, Any]:
    """Predicate (C)'s record THROUGH ``contracts.lens_on_bite_verdict`` plus the
    two vmodels the driver's ``_lens_on_bite`` appends."""
    lens = {**think_shaped_completion(fact, "Rayleigh scattering."), "render_regime": C.RENDER_REGIME_THINKING_ON}
    base = {**think_shaped_completion(fact, "Because of scattering."), "render_regime": C.RENDER_REGIME_THINKING_ON}
    rec = C.lens_on_bite_verdict(lens_reply=lens, base_reply=base, family=fact["family"])
    rec["lens_vmodel"], rec["base_vmodel"] = vmodel_of(cell), vmodel_of(CELL_BARE)
    return rec


def steering_bites_record(cell: str, *, n_differing: int = 4) -> dict[str, Any]:
    """The driver's ``steering_bites`` block on a lens leg (assembled inline in
    ``_served_evidence``; no contracts helper): the twin, the arm, the probe
    counts, the fidelity diag, the OFF render the probe was sent under,
    the body it added, the per-leg stamps."""
    arm = cell.split("__")[-1]
    stamps = {"salt_regime": "cell-stable", "resumed_policy": "always", "render_regime": C.RENDER_REGIME_THINKING_OFF}
    return {C.OOS_EVIDENCE_BITES_APPLICABLE: True, C.OOS_EVIDENCE_BITES_VERIFIED: True,
            C.OOS_EVIDENCE_BITES_TWIN: C.TWIN_CELL_ID, "arm": arm, "n_probes": N_BITES_PROMPTS,
            C.OOS_EVIDENCE_BITES_N_DIFFERING: n_differing,
            "diag": {"arms": {arm: {"n": N_BITES_PROMPTS, "n_differing": n_differing,
                                    "first_divergences": [[i, 3] for i in range(n_differing)]}}},
            C.OOS_EVIDENCE_BITES_PROBE_RENDER_ARM: C.THINKING_OFF,
            "probe_extra_body": C.render_fragment(lens=True, arm=C.THINKING_OFF, header_located=False),
            "stamps_by_leg": {"steered": [dict(stamps) for _ in range(N_BITES_PROMPTS)],
                              "scale0": [dict(stamps) for _ in range(N_BITES_PROMPTS)]},
            "stamp_errors_by_leg": {"steered": [], "scale0": []}}


def transfer_provenance_record(family: str, fact: Mapping[str, Any]) -> dict[str, Any]:
    """The transfer provenance block THROUGH ``contracts.transfer_provenance_block``
    with the driver's arguments for a document-fit object served all-position
    thinking-ON."""
    return C.transfer_provenance_block(fit_kind="document_fit", fit_template_hash=None, fit_response_header_len=None,
                                       served_positions=["all"], served_render_arm=C.THINKING_ON,
                                       served_header_ids_by_arm={"off": list(fact["off_ids"]), "on": list(fact["on_ids"])},
                                       family=family)


@dataclass
class Attempt:
    """ONE HTTP attempt as the two producer records see it — the request-evidence
    record the driver's writer files from the wire bytes + headers
    (``contracts.request_evidence_record``) and the usage-ledger row the ledger
    writes for the same bytes (``usage_ledger.ledger_row``), bound by the minted
    ``X-Tinycap-Request-Id`` and the wire digest."""

    record: dict[str, Any]
    row: dict[str, Any]


def wire_bytes(body: Mapping[str, Any]) -> bytes:
    """The SDK's compact JSON — the exact bytes both records digest."""
    return json.dumps(dict(body), separators=(",", ":")).encode("utf-8")


def attempt(body: Mapping[str, Any], *, task: str, sample: str, ts: str, request_id: str,
            prompt_tokens: int, completion_tokens: int, finish_reason: str = "stop") -> Attempt:
    """One attempt from its wire body and labels: the record THROUGH
    ``contracts.request_evidence_record`` over the bytes and the headers the
    driver / provider set (task, sample, request id); the ledger row THROUGH
    ``usage_ledger.ledger_row`` over the same bytes (``body_facts`` for the
    body-derived columns, the identity columns from the header and the digest)."""
    raw = wire_bytes(body)
    headers = {C.HEADER_TASK: task, C.HEADER_SAMPLE: sample, C.HEADER_REQUEST_ID: request_id}
    rec = C.request_evidence_record(raw, headers=headers, ts=ts)
    row = dict.fromkeys(C.LEDGER_ROW_FIELDS)
    row.update(ts=ts, task=task, sample=sample, has_tools=False, n_tools=0, rejected=False, status=200, error_code=None,
               prompt_tokens_pre=prompt_tokens, usage_prompt_tokens=prompt_tokens, usage_completion_tokens=completion_tokens,
               finish_reason=finish_reason, latency_s=round(completion_tokens / DECODE_TOK_S, 3), tokenize_mismatch=False,
               error_class=C.LEDGER_ERROR_CLASS_NONE, request_id=request_id, body_sha256=C.wire_body_sha256(raw))
    row.update(ul.body_facts(body))
    return Attempt(rec, ledger_row(**row))


def _request_id(*parts: Any) -> str:
    """A deterministic 32-hex attempt id (the driver mints uuid4 hex; the fixture
    derives it from the attempt's own coordinates so a store rebuilds byte for byte)."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:32]


def _probe_body(vmodel: str, prompt: str, *, arm: str, lens: bool, max_tokens: int) -> dict[str, Any]:
    return {"model": vmodel, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0,
            "max_tokens": max_tokens, "seed": 0, **C.render_fragment(lens=lens, arm=arm, header_located=False)}


def probe_attempt(vmodel: str, *, kind: str, n: int, arm: str, lens: bool, prompt: str = "Reply with one word.",
                  max_tokens: int = RENDER_PROBE_MAX_TOKENS, ts: str = TS0, salt: str = "") -> Attempt:
    """One LABELLED probe attempt as the driver's ``_RequestEvidence.labelled``
    wrapper sends it (FC4 (2)): ``X-Tinycap-Task: probe/<kind>``,
    ``X-Tinycap-Sample: <vmodel>/<n>`` (n per kind and route, from 0), a fresh id."""
    label = C.probe_task_label(kind)
    return attempt(_probe_body(vmodel, prompt, arm=arm, lens=lens, max_tokens=max_tokens), task=label,
                   sample=f"{vmodel}/{n}", ts=ts, request_id=_request_id(salt, ts, label, vmodel, n),
                   prompt_tokens=30, completion_tokens=8)


def probe_attempts_one_run(cell: str, *, fitted: bool, ts: str = TS0, salt: str = "") -> list[Attempt]:
    """Every probe attempt ONE run of the driver sends, in the driver's order,
    each a ledger row AND a record (under identity version 2 the probes travel
    THROUGH the ledger): the bites probe on both legs (five prompts each,
    thinking-OFF — fitted only), predicate (A)'s OFF and ON render probes
    on the cell's route, predicate (C)'s lens-ON bite on the lens route and the
    base route (fitted only). ``n`` counts per (kind, route) from 0, as the
    driver's wrapper does, restarting on every run."""
    vm = vmodel_of(cell)
    out: list[Attempt] = []
    if fitted:
        for target in (vm, C.twin_vmodel()):
            for i in range(N_BITES_PROMPTS):
                out.append(probe_attempt(target, kind=C.PROBE_KIND_BITES, n=i, arm=C.THINKING_OFF, lens=True,
                                         prompt=f"probe prompt {i}", max_tokens=BITES_PROBE_MAX_TOKENS, ts=ts, salt=salt))
    for i, arm in enumerate((C.THINKING_OFF, C.THINKING_ON)):
        out.append(probe_attempt(vm, kind=C.PROBE_KIND_RENDER, n=i, arm=arm, lens=fitted, ts=ts, salt=salt))
    if fitted:
        for target in (vm, vmodel_of(CELL_BARE)):
            out.append(probe_attempt(target, kind=C.PROBE_KIND_LENS_ON_BITE, n=0, arm=C.THINKING_ON, lens=True,
                                     prompt="In one short paragraph, why is the sky blue?", max_tokens=LENS_ON_BITE_MAX_TOKENS,
                                     ts=ts, salt=salt))
    return out


def probe_records_one_attempt(cell: str, *, fitted: bool, ts: str = TS0) -> list[dict[str, Any]]:
    """The request-evidence PROBE records of one run (:func:`probe_attempts_one_run`)."""
    return [a.record for a in probe_attempts_one_run(cell, fitted=fitted, ts=ts)]


def task_body(cell: str, *, fitted: bool, row: Mapping[str, Any], item_id: str, seed: int) -> dict[str, Any]:
    """The Inspect provider's wire body for one task call: the route, the item's
    prompt, the cap, the sampling row's numbers (``top_k`` and the optional
    terms ride the body — the provider puts them in ``extra_body``), the seed,
    the ON render fragment."""
    body: dict[str, Any] = {"model": vmodel_of(cell), "messages": [{"role": "user", "content": f"MATH-500 problem {item_id}"}],
                            "max_tokens": CAP, "temperature": float(row["temperature"]), "seed": seed}
    if row.get("top_p") is not None:
        body["top_p"] = float(row["top_p"])
    if row.get("top_k") is not None:
        body["top_k"] = int(row["top_k"])
    for f in C.SAMPLING_OPTIONAL_ROW_FIELDS:
        if row.get(f) is not None:
            body[f] = float(row[f])
    body.update(C.render_fragment(lens=fitted, arm=C.THINKING_ON, header_located=False))
    return body


def task_attempt(cell: str, *, fitted: bool, row: Mapping[str, Any], item: Mapping[str, Any], ts: str = TS0,
                 salt: str = "") -> Attempt:
    """One task call's attempt: the record over the provider's wire body under
    the tinycap labels (``X-Tinycap-Task`` the row task, ``X-Tinycap-Sample``
    ``<item_id>/<sample_idx>``) and the ledger row the ledger writes for it."""
    sample = C.sample_header_value(item["item_id"], item["sample_idx"])
    return attempt(task_body(cell, fitted=fitted, row=row, item_id=str(item["item_id"]), seed=int(item["seed"])),
                   task=TASK, sample=sample, ts=ts, request_id=_request_id(salt, ts, TASK, sample),
                   prompt_tokens=int(item["prompt_tokens"]), completion_tokens=int(item["completion_tokens"]),
                   finish_reason="length" if item["at_cap"] else "stop")


def build_store(gen_root: Path, *, model_key: str, cell: str, mode: str = MODE_PRIMARY,
                correct: Iterable[int], unscored: Iterable[int] = (), not_at_cap: Iterable[int] = (),
                stray: Iterable[int] = (), capped_scored: Iterable[int] = (), no_think: Iterable[int] = (),
                ids: Sequence[str] = IDS, sha: str = SHA7, tree_sha: str = FULL_SHA,
                outcomes_tree_sha: str | None = None,
                outcome_sampling_ref: str | None = OUTCOME_STAMP_AUTO,
                render_facts: Mapping[str, Mapping[str, Any]] = RENDER_FACTS,
                sampling_rows: Mapping[str, Mapping[str, Mapping[str, Any]]] = SAMPLING_ROWS,
                write_rows: bool = True, ledger: bool = True, request_evidence: bool = True,
                evidence: bool = True, stamps: Mapping[str, Any] | None = None,
                summary_edit: Callable[[dict[str, Any]], None] | None = None,
                header_edit: Callable[[dict[str, Any]], None] | None = None,
                manifest_edit: Callable[[dict[str, Any]], None] | None = None,
                evidence_edit: Callable[[dict[str, Any]], None] | None = None,
                ledger_edit: Callable[[list[dict[str, Any]]], None] | None = None,
                request_evidence_edit: Callable[[list[dict[str, Any]]], None] | None = None,
                attempts_edit: Callable[[list[Attempt]], None] | None = None,
                incomplete_runs: Iterable[int] = (),
                rows_edit: Callable[[list[dict[str, Any]]], None] | None = None,
                completion_tokens: Mapping[int, int] | None = None,
                texts: Mapping[int, tuple[str | None, str | None]] | None = None,
                resume_rows: int = 0, resumes: int | None = None,
                redraws: Mapping[int, int] | None = None, redraw_late_completion: bool = False,
                resume_attempts: Sequence[Mapping[str, Any]] | None = None,
                limits: Mapping[str, int] | None = None, n_expected_full: int | None = None,
                out_root: Path | None = None, row: Mapping[str, Any] | None = None, object_sha: str | None = None,
                max_model_len: int | None = None, launch_record: bool = True,
                launch_edit: Callable[[dict[str, Any]], None] | None = None, log_requests: bool = False,
                serving_config_record: bool = True, serving_config_sha256: str | None = None) -> Path:
    """One leg's store under ``<gen_root>/controls/<model>/<cell slug>/<mode>/``
    (or, for a row carrying a wave-isolated ``out_root`` — the eight-arm
    extension — under ``<gen_root>/<row out_root>``). ``row`` is the matrix row
    to build under when the tree's ``cells.py`` does not name the leg (the
    extension's rows, :func:`ext_rows`); ``object_sha`` the fitted cell's OWN
    object digest (default the row's ``object_sha256``, else the published
    minus_ctl pin); ``max_model_len`` the served window (default the row's
    ``geometry_L``, else the published 17,664).

    ``correct`` / ``unscored`` / ``not_at_cap`` / ``stray`` / ``capped_scored`` /
    ``no_think`` are item INDICES into ``ids``: an unscored item's think block
    never closed (at the cap unless in ``not_at_cap``: case (b)); a
    ``capped_scored`` item closed its block, then ran to the cap in the answer
    and IS scored (case (c)); a ``no_think`` item (GLM's ``explicit_tags`` only)
    emitted no think tag at all — the whole completion is the answer, scored,
    ``think_block_present False``; a ``stray`` item carries a literal open tag
    inside a prompt-opened block. On gemma (``record_only``) an unscored item
    is a reasoning-only cap-bound output: ``content_empty_with_reasoning``.
    ``texts[j] = (content, reasoning)`` sets a draw's two channels verbatim (the
    default channels are one-token stand-ins); every SCORED draw's channels are
    filed on its row's ``extra_json`` as ``content`` / ``reasoning``, exactly as
    the producer's ``ItemResult.extra`` files them — an unscored draw has no row.

    A RESUME, as the driver leaves it: ``resumes`` extra runs of the store
    (default one when ``resume_rows`` is set, else none), each recorded on the
    manifest's ``resume_attempts`` (``resume_attempts_of``), each sending its
    own probe attempts; ``resume_rows`` names how many samples (the first that
    many) the resume re-sent — one more attempt (a ledger row AND a record, a
    fresh request id) per re-sent sample. ``incomplete_runs`` (0-based) are runs
    whose preflight refused: recorded ``completed: false``, probes only.
    ``resume_attempts`` overrides the manifest's record verbatim (a negative
    test's off-shape record).

    ``redraws[j] = n``: the harness RE-SENT item ``j`` ``n`` extra times
    at the same cap inside the first run — the client gave up on a request the
    upstream finished (the openai SDK's default read timeout), so the ledger
    carries ``n`` earlier HTTP 200 rows for the sample (each a record AND a
    ledger row with a fresh request id, at the cap, ``finish_reason length``,
    each started one client timeout after the last) before the final draw's
    row; no run is recorded for them. The final draw is the one the outcome
    record files. ``redraw_late_completion`` appends the abandoned requests'
    rows AFTER the final draw's (the ledger appends on completion: an
    abandoned at-cap request finishing after its replacement lands last) while
    their ``ts`` stays earlier — the shape the GLM cells show.

    Every attempted call is ONE ``Attempt`` (a record and a ledger row from the
    same wire bytes, bound by the request id) in the driver's order per run —
    probes first, then the task calls; ``attempts_edit`` edits that joint list,
    ``request_evidence_edit`` / ``ledger_edit`` the split lists after it. Under
    identity version 2 the probes travel through the ledger, labelled
    ``probe/<kind>`` with sample ``<vmodel>/<n>``.
    ``outcomes_tree_sha`` (default ``tree_sha``) is the outcomes record's own
    producer sha.

    The outcome record is the PRODUCER's: the per-draw facts become
    ``inspect_served.ItemResult`` objects (what ``results_from_log`` yields), the
    summary's population / quality / split counts are the runner's own
    derivations over them (``attempted_counts`` / ``task_quality`` /
    ``think_split_counts``) and the record is ``outcome_record()`` over the same
    list, stamped per ``outcome_sampling_ref`` (:data:`OUTCOME_STAMP_AUTO`) —
    a hand-assembled record hid the writer / contracts mismatch that
    refuses every stamped v2 record. Returns the cell dir."""
    matrix = dict(row) if row is not None else matrix_row(model_key, cell, mode)     # refuses a leg the matrix does not name
    family = FAMILY_BY_KEY[model_key]
    fact = render_facts[family]
    row = sampling_rows[mode][model_key]
    fitted = bool(matrix["fitted"])
    assert fitted == is_fitted(cell)
    correct, unscored, not_at_cap = set(correct), set(unscored), set(not_at_cap)
    stray, no_think = set(stray), set(no_think)
    capped_scored = set(capped_scored) - unscored
    length_of = dict(completion_tokens or {})
    if out_root is not None:
        root = Path(out_root)
    elif matrix.get("out_root") and not str(matrix["out_root"]).startswith("controls/"):
        root = Path(gen_root) / str(matrix["out_root"])            # a wave-isolated out root (controls_<wave>/...)
    else:
        root = out_root_of(gen_root, model_key, cell, mode)
    served_window = int(max_model_len or matrix.get("geometry_L") or MAX_MODEL_LEN)
    vmodel = vmodel_of(cell)
    hf, rev = model_pin(model_key)
    ihash = _hash8(f"{model_key}|{cell}|ihash")
    chash = _hash8(f"{model_key}|{cell}|{mode}|{CAP}|chash")
    key = f"{cell_slug(cell)}__controls__t0__a0__{ihash}__{chash}"
    cell_dir = root / "cells" / key
    cell_dir.mkdir(parents=True, exist_ok=True)
    record_only = fact["split_mode"] == C.THINK_SPLIT_RECORD_ONLY
    n_resumes = int(resumes) if resumes is not None else (1 if resume_rows else 0)
    n_attempts = 1 + n_resumes
    incomplete = {int(k) for k in incomplete_runs}
    assert 0 not in incomplete, "the first run wrote the store; it completed"
    resume_ts = [f"2026-01-01T{k + 1:02d}:00:00Z" for k in range(n_resumes)]

    # the sampling block (contracts.sampling_identity) and its digest first: the producer stamps the digest on
    # every row and, on an identity-v2 run, on every outcome item
    sampling_block = sampling_block_of(model_key, mode, row)
    sampling_ref = C.sampling_ref(sampling_block)
    stamp = _resolve_outcome_stamp(outcome_sampling_ref, sampling_ref)

    # every draw's FINAL attempt as the runner's `results_from_log` shapes it; the record below is the producer's
    # `outcome_record()` over this list, never a hand-assembled one
    results: list[isv.ItemResult] = []
    channels = dict(texts or {})
    for j, item_id in enumerate(ids):
        is_unscored = j in unscored
        at_cap = (is_unscored and j not in not_at_cap) or j in capped_scored
        value = float("nan") if is_unscored else (1.0 if j in correct else 0.0)
        content, reasoning = channels.get(j, ("" if (record_only and is_unscored) else "7",
                                              "reasoning" if record_only else None))
        results.append(isv.ItemResult(
            item_id=item_id, value=value, sample_idx=0, seed=1000 + j, answer=None,
            prompt_tokens=PROMPT_BASE + (j % 160),
            completion_tokens=CAP if at_cap else length_of.get(j, 1500 + (j * 7) % 900),
            paired_id=None, scores={PRIMARY_SCORER: value},
            content=content, reasoning=reasoning,
            content_empty_with_reasoning=bool(reasoning) and not content,
            stop_reason=isv.STOP_REASON_AT_CAP if at_cap else "stop",
            unparseable=None if is_unscored else False, attempt=1, max_tokens_used=CAP,
            think_block_present=j not in no_think, think_block_unterminated=is_unscored, stray_open_tag=j in stray))
    plan = task_plan(row, ids, fact)
    counts = isv.attempted_counts(results)
    split_counts = isv.think_split_counts(results, plan=plan)
    n_scored = counts["n_scored"]
    n_correct = sum(1 for r in results if not r.unscored and r.value == 1.0)
    stamps = dict(stamps) if stamps is not None else dict(zip(C.EXPECTED_STAMP_TRIPLE_FIELDS, matrix["expected_stamps"]))
    summary: dict[str, Any] = {
        C.METRIC_ACC: n_correct / n_scored if n_scored else None, C.METRIC_N: counts["n"], "n_requests": len(ids),
        "n_rejected": 0, "n_http_rejected_retried": 0,
        "samples_per_item": 1, "task": TASK, "item_set": None,
        "framework": C.INSPECT_TASK_VERSION, "engine": "vllm", "route": vmodel,
        "thinking": C.THINKING_ON, "render": C.THINKING_RENDER_NATIVE, "wall_clock_s": 3600.0,
        "logs_reused": 0, "reasoning_parser": fact["server_parser"],
        "n_content_empty_with_reasoning": sum(1 for r in results if r.content_empty_with_reasoning),
        "n_scored": n_scored, "n_unscored": counts["n_unscored"],
        "n_scored_samples": counts["n_scored_samples"], "n_unscored_samples": counts["n_unscored_samples"],
        # the runner's own quality and split derivations over the same results (U5-5; plan S2.6)
        **isv.task_quality(results),
        **split_counts,
        "max_tokens": CAP, "temperature": row["temperature"], "top_p": row.get("top_p"), "top_k": row.get("top_k"),
        "grader_max_tokens": None,
    }
    for f in C.EXPECTED_STAMP_TRIPLE_FIELDS:
        summary[f] = stamps.get(f)
        summary[C.STAMP_REGIME_COUNT_FIELD[f]] = stamps.get(C.STAMP_REGIME_COUNT_FIELD[f], 1)
    if summary_edit:
        summary_edit(summary)
    # the PRODUCER's record over the same results: its header copies the (edited) summary's counts, as the
    # runner's does, so `finish_task` verifies by construction; the stamp rides every item when set
    header = isv.outcome_record(results, plan=plan, task=TASK, thinking=C.THINKING_ON, summary=summary,
                                split_counts={k: summary[k] for k in split_counts},
                                producer_tree_sha=tree_sha if outcomes_tree_sha is None else outcomes_tree_sha,
                                sampling_ref=stamp)
    if header_edit:
        header_edit(header)
    items = header["items"]

    # the served concurrency is the model budget's S at the served window (the arm contract holds every leg to
    # its row's resolved geometry — window, S AND the client width — and the launch echo is reconciled with it)
    served_s = served_seqs(model_key, served_window)
    geometry_eval = str(matrix.get("geometry_eval") or CELLS_MODULE.geometry_for(cell)[0])
    served = {"reasoning_parser": fact["server_parser"], "tool_calls": False, "max_model_len": served_window,
              "max_num_seqs": served_s, "concurrency": served_s, C.AGENT_LOOP_ENTRY_KEY: False,
              C.REDUCTION_MODE_FIELD: "stacked"}
    tinycap: dict[str, Any] = {
        "tasks": [TASK], "pass": 1, "samples": 1, "served": served,
        "thinking": {"mode": C.THINKING_ON, "by_task": {TASK: C.THINKING_ON}},
        C.IDENTITY_KEY_THINKING_BY_TASK: {TASK: C.THINKING_ON},
        "versions": {"tinycap": C.TINYCAP_VERSION}, "limits": {}, "frameworks": {"inspect_ai": "0.3.260"},
        IDENTITY_KEY_VERSION: C.TINYCAP_IDENTITY_VERSION_2,
        C.IDENTITY_KEY_MAX_TOKENS_BY_TASK: {TASK: CAP},
        C.IDENTITY_KEY_MATH500_RETRY_MAX_TOKENS: 0,
        # the hashed sampling identity (contracts.sampling_identity): the three nucleus knobs ALWAYS (None when the
        # row sets none) and the optional terms only when the row names them
        **{k: sampling_block[k] for k in C.SAMPLING_IDENTITY_FIELDS},
        **{f: sampling_block[f] for f in C.SAMPLING_OPTIONAL_IDENTITY_FIELDS if f in sampling_block},
        IDENTITY_KEY_THINKING_HEADER_IDS: {"off": list(fact["off_ids"]), "on": list(fact["on_ids"])},
    }
    if fitted:
        tinycap[C.IDENTITY_KEY_OUT_OF_SUPPORT_THINKING] = True
    pin = (object_sha or matrix.get("object_sha256") or object_pin(model_key)) if fitted else None
    harness = harness_id_of(cell)
    served_positions = ["all"] if fitted else None
    # the served route class and the SI provenance as the driver files them (the eight-arm analysis' per-row ARM
    # CONTRACT reads them, plan P5 (i′)): the lens serving identity on a fitted cell, passthrough otherwise; the SI the
    # request carried (`provenance.si_id`) and the prompted prefix digest (`resolved_config.system_sha`)
    serving_identity = serving_identity_of(fitted)
    si_id = si_of(cell)
    system_sha = SI_PREFIX_SHA16.get(str(si_id)) if cell.startswith(C.PROMPTED_CELL_PREFIX) else None
    manifest: dict[str, Any] = {
        "cell_id": cell, "harness_cell_id": harness, "virtual_model": vmodel,
        "intervention_hash": ihash + "0" * 56, "harness_config_hash": chash + "0" * 56,
        "model_id": hf, "model_revision": rev, "engine": "vllm", "serving_path": "lens" if fitted else "passthrough",
        "serving_identity": serving_identity,
        C.MANIFEST_KEY_PRODUCER: dict(zip(C.PRODUCER_BLOCK_FIELDS, (tree_sha, C.PRODUCER_DRIVER_SERVED))),
        C.MANIFEST_KEY_CONTRACT_VERSION: C.CONTRACT_VERSION,
        C.MANIFEST_KEY_PROVENANCE: {"cell_id": cell, "class": "A" if fitted else "baseline",
                                    "artifact_sha256": pin, "intervention_hash": ihash + "0" * 56,
                                    "model": hf, "model_revision": rev, "si_id": si_id,
                                    "serving": {"chat_template_hash": CHAT_TEMPLATE_HASH_BY_FAMILY[family], "engine": "vllm",
                                                "requested_model": hf, "requested_revision": rev, "resolved_model_id": hf,
                                                "resolved_revision": rev, "serving_identity": serving_identity, "virtual_model": vmodel}},
        C.MANIFEST_KEY_TINYCAP: tinycap,
        "resolved_config": {"tinycap": tinycap, "cell_id": cell, "model_id": hf, "model_revision": rev, "system_sha": system_sha,
                            "serving": {"serving_identity": serving_identity, "virtual_model": vmodel}},
        C.MANIFEST_KEY_ROUTE: {"virtual_model": vmodel, "grader_virtual_model": None, "tool_calls": False,
                               "serving_identity": serving_identity,
                               C.REDUCTION_MODE_FIELD: "stacked",
                               C.ROUTE_KEY_SERVE_DECLARATIONS: {},
                               C.ENDPOINT_KEY_OBJECT_ARTIFACT_SHA256_BY_CELL: {harness: pin} if fitted else {},
                               C.ENDPOINT_KEY_LENS_SALT_POLICY: C.LENS_SALT_POLICY_ATTESTED,
                               C.ROUTE_KEY_RESUMED_PREFILL_POLICY: stamps.get("resumed_policy"),
                               # the SELECTED cell's served modes (the driver copies `served_positions`: a list on
                               # a lens cell, null on the base / prompted route)
                               ROUTE_KEY_LENS_SERVED_POSITIONS: served_positions},
        C.MANIFEST_KEY_THINKING: {"enable_thinking": True, "render": C.THINKING_RENDER_NATIVE,
                                  "reasoning_parser": fact["server_parser"], "mode": C.THINKING_ON,
                                  "by_task": {TASK: C.THINKING_ON}},
        C.MANIFEST_KEY_GEOMETRY: geometry_block(model_key, eval_key=geometry_eval, max_model_len=served_window),
        C.MANIFEST_KEY_CONCURRENCY: served_s,
        C.MANIFEST_KEY_SAMPLING: sampling_block,
        C.MANIFEST_KEY_USAGE_LEDGER: str(root / C.LEDGER_FILENAME) if ledger else None,
        "ledger_url": LEDGER_URL,
        # the standup record's cluster path (code review C2): the analysis resolves serving/<jobdir>/math500/ under the gen root
        "routing_table": cluster_routing_table_path(standup_jobdir(model_key, cell, mode, standup_job_id(model_key, cell, mode, tree_sha=tree_sha))),
        "row_constants": {"slurm_job_id": standup_job_id(model_key, cell, mode, tree_sha=tree_sha), "engine": "vllm", "model_id": hf,
                          "model_revision": rev, "kind": "cell", "attempt": 0},
        MANIFEST_KEY_BINDING: {"proxy_url": PROXY_URL,
                               "routing_table": cluster_routing_table_path(standup_jobdir(model_key, cell, mode, standup_job_id(model_key, cell, mode, tree_sha=tree_sha))),
                               "ledger_url": LEDGER_URL, "ledger_upstream": PROXY_URL,
                               "ledger_endpoint": {"ledger_url": LEDGER_URL, "upstream": PROXY_URL}},
        # the identity-version-2 experiment record, unhashed
        C.MANIFEST_KEY_RENDER_FACT_EVIDENCE: {k: fact.get(k) for k in ("family", "off_ids", "on_ids", "split_mode",
                                                                       "server_parser", "evidence_status")},
        MANIFEST_KEY_OOS_EXPERIMENT: matrix["tuple_name"] if fitted else None,
        "support": C.support_label(fitted=fitted, out_of_support=fitted),
        "header_located": False if fitted else None,
        C.LENS_SERVED_POSITIONS_KEY: served_positions,
        MANIFEST_KEY_RESUME_ATTEMPTS: (list(resume_attempts) if resume_attempts is not None
                                       else resume_attempts_of(n_attempts, tree_sha, incomplete)),
        C.MANIFEST_KEY_REQUEST_EVIDENCE: str(root / REQUEST_EVIDENCE_FILENAME),
        "ledger_thinking_refusals": {},
        "host": "TESTHOST",
    }
    if fitted:
        manifest[MANIFEST_KEY_TRANSFER_PROVENANCE] = transfer_provenance_record(family, fact)
    if manifest_edit:
        manifest_edit(manifest)
    (cell_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    if write_rows:
        rows = []
        channels_of = {res.item_id: (res.content, res.reasoning) for res in results}
        for r in items:
            if not r["scored"]:
                continue
            content, reasoning = channels_of[r["item_id"]]
            extra = {"sample_idx": 0, "seed": r["seed"], "answer": "7", "prompt_tokens": r["prompt_tokens"],
                     "completion_tokens": r["completion_tokens"], "stop_reason": r["stop_reason"],
                     "unparseable": False, "unscored": False, "attempt": 1, "max_tokens_used": CAP,
                     # both channels, as the producer's ItemResult.extra files them (the analysis' loop rule reads them)
                     "content": content, "reasoning": reasoning,
                     "think_block_present": r["think_block_present"], "think_block_unterminated": False,
                     C.SAMPLING_REF_FIELD: sampling_ref}
            rows.append({"task": TASK, "item_id": r["item_id"], "metric": C.METRIC_ACC, "value": r["value"],
                         "engine": "vllm", "extra_json": json.dumps(extra),
                         "harness_config_hash": chash + "0" * 56, "intervention_hash": ihash + "0" * 56})
        if rows_edit:
            rows_edit(rows)
        lines = [json.dumps(r) for r in rows]
        lines.append(json.dumps({"task": TASK, "item_id": C.AGG_ITEM_ID, "metric": C.METRIC_ACC_AGGREGATE,
                                 "value": summary[C.METRIC_ACC], "engine": "vllm", "extra_json": "{}"}))
        lines.append(json.dumps({"task": TASK, "item_id": C.AGG_ITEM_ID, "metric": C.METRIC_N,
                                 "value": float(len(ids)), "engine": "vllm", "extra_json": None}))
        (cell_dir / "rows.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    completion.finish_task(cell_dir, task=TASK, n_items=len(ids),
                           n_expected_full=(len(ids) if n_expected_full is None else n_expected_full),
                           limits=limits, n_rows=n_scored, n_requests=len(ids), n_rejected=0,
                           wall_clock_s=3600.0, summary=summary, outcomes=header)

    # every attempted call, in the driver's order per run: the probes (preflight), then the task calls; a resume run
    # sends its probes and re-sends `resume_rows` samples (new attempts, new ids); an incomplete run sends probes only.
    # ONE inventory feeds both files: each Attempt is a record AND a ledger row over the same bytes
    attempts: list[Attempt] = []
    attempts += probe_attempts_one_run(cell, fitted=fitted, ts=TS0, salt=key)
    for j, r in enumerate(items):
        # the harness's same-cap re-sends of this item (the client's retries, each an attempt of its own with a fresh
        # id, HTTP 200 at the cap, started one client timeout apart), then the final draw the record files — the ledger
        # stamps `ts` at request START; the file order is COMPLETION order: chronological by default, and under
        # `redraw_late_completion` the abandoned requests land AFTER the replacement that finished first
        n_redraw = int((redraws or {}).get(j, 0))
        phantoms = [task_attempt(cell, fitted=fitted, row=row, item={**r, "completion_tokens": CAP, "at_cap": True},
                                 ts=_ts_plus(TS0, REDRAW_TIMEOUT_S * n), salt=f"{key}|redraw{n}") for n in range(n_redraw)]
        final = task_attempt(cell, fitted=fitted, row=row, item=r, ts=_ts_plus(TS0, REDRAW_TIMEOUT_S * n_redraw), salt=key)
        attempts += ([final, *phantoms] if redraw_late_completion else [*phantoms, final])
    resent = items[:resume_rows]
    for k, ts in enumerate(resume_ts):
        attempts += probe_attempts_one_run(cell, fitted=fitted, ts=ts, salt=key)
        if k + 1 in incomplete:
            continue                # a preflight that refused: on record, probes only
        if resent:
            attempts += [task_attempt(cell, fitted=fitted, row=row, item=r, ts=ts, salt=key) for r in resent]
            resent = []
    if resent:                      # re-sent samples with NO resume run recorded (a negative test): extra attempts of run 1
        attempts += [task_attempt(cell, fitted=fitted, row=row, item=r, ts="2026-01-01T01:00:00Z", salt=key) for r in resent]
    if attempts_edit:
        attempts_edit(attempts)
    recs: list[dict[str, Any]] = [a.record for a in attempts]
    lrows: list[dict[str, Any]] = [a.row for a in attempts]
    if request_evidence:
        if request_evidence_edit:
            request_evidence_edit(recs)
        (root / REQUEST_EVIDENCE_FILENAME).write_text("\n".join(json.dumps(r, sort_keys=True) for r in recs) + "\n",
                                                      encoding="utf-8")

    if evidence:
        ev: dict[str, Any] = {
            "driver": "served_capability_run", "cell_id": cell, "harness_cell_id": harness, "virtual_model": vmodel,
            "engine": "vllm", "mask": "generated" if fitted else None,
            C.OOS_EVIDENCE_BITES_BLOCK: (steering_bites_record(cell) if fitted
                                         else {C.OOS_EVIDENCE_BITES_APPLICABLE: False,
                                               "reason": f"route {harness!r} carries no steering selector (serving path is "
                                                         "not lens) — no steered arm to probe"}),
            EVIDENCE_RENDER_CERTIFIED: render_certified_record(family, fact),
            EVIDENCE_RENDER_PROBE: render_probe_record(fact),
            "ledger_thinking_refusals": {},
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(0)),
        }
        if fitted:
            ev[EVIDENCE_LENS_POSITION] = lens_position_record(model_key, cell, pin=pin)
            ev[EVIDENCE_LENS_ON_BITE] = lens_on_bite_record(cell, fact)
            ev["request_evidence_gate"] = {"n_records": len(recs), "n_ledger_rows": len(lrows), "n_filed_items": n_scored,
                                           "problem": None}
        if evidence_edit:
            evidence_edit(ev)
        blob = json.dumps(ev, indent=1)
        (cell_dir / "evidence.json").write_text(blob, encoding="utf-8")
        (root / "evidence.json").write_text(blob, encoding="utf-8")

    if ledger:
        # the ledger fronts EVERY call under identity version 2 — the Inspect task traffic (the provider's tinycap
        # labels) and the driver's labelled probes — one row per attempt, the identity columns on each
        if ledger_edit:
            ledger_edit(lrows)
        assert all(set(r) == set(C.LEDGER_ROW_FIELDS) for r in lrows)
        (root / C.LEDGER_FILENAME).write_text("\n".join(json.dumps(r) for r in lrows) + "\n", encoding="utf-8")
    if launch_record:
        # the standup / launch record beside the job (code review C2): the attestation bound to the Slurm job and its port, and
        # the server log carrying the engine's echo of the composed launch; `launch_edit` mutates the echoed args (a negative test)
        job = standup_job_id(model_key, cell, mode, tree_sha=tree_sha)
        jobdir = standup_jobdir(model_key, cell, mode, job)
        port = launch_port(job)
        args = launch_echo_args(model_key, revision=rev, port=port, max_model_len=served_window, max_num_seqs=served_s,
                                reasoning_parser=fact["server_parser"], log_requests=log_requests, jobdir=jobdir)
        if launch_edit:
            launch_edit(args)
        # the serving-config provenance beside them — the sha256 of the config THIS leg served, so the
        # instrument fork (an object selected on one tree's serving config, evaluated on another's) is visible per leg.
        # `serving_config_record=False` builds the EARLIER shape a published leg has, which must still read.
        cfg_rel = f"configs/sdfbase/math500/sdfbase_full22_{model_key}.math500.yaml"
        sc = None if not serving_config_record else {
            "schema": "serving_config_record/1", "config": cfg_rel,
            "serving_config_sha256": (serving_config_sha256
                                      or hashlib.sha256(f"{cfg_rel}|{tree_sha}".encode()).hexdigest()),
            "manifest": f"configs/capeval/manifests/petri_full22_{model_key}.math500.yaml",
            "manifest_sha256": hashlib.sha256(f"manifest|{model_key}|{tree_sha}".encode()).hexdigest(),
            "model_key": model_key, "model": hf, "model_revision": rev, "written_at": "2026-01-01T00:00:00Z"}
        write_launch_record(gen_root, jobdir, job=job, port=port, args=args, generation_config=GENERATION_CONFIG_BY_KEY.get(model_key),
                            tree_sha=tree_sha, model_key=model_key, max_model_len=served_window, max_num_seqs=served_s,
                            log_requests=log_requests, serving_config=sc)
    return cell_dir


def write_inspect_log(cell_dir: Path, *, name: str, items: Sequence[Mapping[str, Any]] | None = None,
                      vmodel: str | None = None, sample_idx: int = 0, n_samples: int | None = None,
                      cancelled: bool = False, value_edit: Callable[[str, float], float] | None = None,
                      mtime: float | None = None, max_tokens: int | str | None = LOG_CAP_STORE,
                      texts: Mapping[str, tuple[str | None, str | None]] | None = None,
                      model_retries: Mapping[str, Sequence[int]] | None = None,
                      failed_attempts: Mapping[str, int | Sequence[str]] | None = None,
                      raw_bodies: Mapping[str, tuple[str | None, str | None]] | None = None) -> Path:
    """A REAL Inspect ``.eval`` log of the store's MATH-500 pass, written through
    ``inspect_ai.log.write_eval_log`` into the producer's log dir
    (``contracts.inspect_log_dir``) in the shape the runner reads back
    (``inspect_served.results_from_log``): one ``EvalSample`` per outcome item —
    its id, the primary scorer's letter (or inspect's UNSCORED sentinel) with the
    think-split metadata, the provider's stop reason, the target's usage — under
    the PLAN Inspect records at log start (``inspect_served.log_max_tokens``
    reads it): the eval-level ``GenerateConfig`` the runner passes ``eval_async``,
    the cap and the store's sampling row, merged over the task's own config.
    ``max_tokens`` is that recorded cap (:data:`LOG_CAP_STORE`: the store's own,
    as the producer's log of it carries; an int: a log generated at another cap;
    ``None``: a plan recording no cap). ``items`` defaults to the store's own
    outcome record. ``n_samples`` keeps the
    first that many (an abandoned run's log); ``cancelled`` writes what an
    INTERRUPTED eval leaves — status ``cancelled``, the last kept sample carrying
    inspect's backend cancellation error, no score — the log a later successful
    resume sits beside. ``value_edit(item_id, value)`` perturbs a scored value (a
    log that disagrees with the record). ``mtime`` pins the file time so the
    selection order among siblings is deterministic. ``texts[item_id] =
    (content, reasoning)`` writes that sample's output in inspect's split shape
    (a ``ContentReasoning`` part before the ``ContentText`` when ``reasoning`` is
    not None — the shape ``think_split.output_channels`` reads back); the default
    output is the one-token stand-in. Every sample carries its final attempt's
    ``ModelEvent`` s as Inspect logs them: ONE event with no retries by default;
    ``model_retries[item_id] = [r1, r2, ...]`` writes that many events, each with
    its ``retries`` count, every event but the last carrying Inspect's timeout
    error (a model call that exhausted its retries); ``failed_attempts[item_id]``
    lists earlier sample attempts on ``error_retries`` — an int ``n`` writes
    ``n`` timeout-ladder failures (:data:`FAILED_ATTEMPT_TIMEOUT_MESSAGE`), a
    sequence of strings writes those messages verbatim (a provider refusal,
    :data:`FAILED_ATTEMPT_REFUSAL_MESSAGE`, or anything else; the ledger
    cross-check classifies each). ``raw_bodies[item_id] = (raw_content,
    finish_reason)`` puts the provider's RAW response on the sample's
    final model event as ``ModelEvent.call.response`` in the OpenAI chat shape
    (``choices[0].message.content`` / ``finish_reason``) — the un-lifted body the
    analysis classifies an unterminated-not-at-cap draw from; a sample without
    one carries no ``call`` (as a log whose provider recorded none). Needs
    inspect_ai (the figure venv has none: callers ``importorskip`` it). Returns
    the path."""
    from inspect_ai._util.content import ContentReasoning, ContentText
    from inspect_ai.event import ModelEvent
    from inspect_ai.log import (
        EvalConfig,
        EvalDataset,
        EvalLog,
        EvalPlan,
        EvalPlanStep,
        EvalResults,
        EvalSample,
        EvalSpec,
        EvalStats,
        write_eval_log,
    )
    from inspect_ai.log._log import EvalError, EvalRetryError
    from inspect_ai.model import (
        ChatCompletionChoice,
        ChatMessageAssistant,
        ChatMessageUser,
        GenerateConfig,
        ModelCall,
        ModelOutput,
        ModelUsage,
    )
    from inspect_ai.scorer import Score

    cell_dir = Path(cell_dir)
    manifest = completion.read_store_manifest(cell_dir)
    summary = completion.read_task_summary(cell_dir, TASK)
    if items is None:
        items = completion.read_task_outcomes(cell_dir, TASK)["items"]
    if vmodel is None:
        vmodel = str(manifest[C.MANIFEST_KEY_ROUTE]["virtual_model"])
    model = C.inspect_model_name(vmodel)
    # the plan Inspect writes at log start: `plan.config` is the eval-level GenerateConfig the runner passes
    # `eval_async` (`_eval` in inspect_served: the cap, then the row's temperature / top_p / top_k and the optional
    # presence penalty) merged over the task's own config — `log.plan.config.max_tokens` is what `reusable_log`
    # holds a retained log to, so a fixture log that carried a bare plan could never exercise that predicate
    gen: dict[str, Any] = {}
    if max_tokens == LOG_CAP_STORE:
        max_tokens = int(summary["max_tokens"])
    if max_tokens is not None:
        gen["max_tokens"] = int(max_tokens)
    if summary.get("temperature") is not None:
        gen["temperature"] = float(summary["temperature"])
    if summary.get("top_p") is not None:
        gen["top_p"] = float(summary["top_p"])
    if summary.get("top_k") is not None:
        gen["top_k"] = int(summary["top_k"])
    penalty = (manifest.get(C.MANIFEST_KEY_SAMPLING) or {}).get(C.IDENTITY_KEY_PRESENCE_PENALTY)
    if penalty is not None:
        gen["presence_penalty"] = float(penalty)
    plan = EvalPlan(name="plan", steps=[EvalPlanStep(solver="generate", params={})], config=GenerateConfig(**gen))
    kept = list(items)[: (len(items) if n_samples is None else int(n_samples))]
    samples = []
    for k, it in enumerate(kept):
        last_cancelled = cancelled and k == len(kept) - 1
        value: Any = float("nan")
        if it["scored"]:
            value = float(it["value"]) if value_edit is None else float(value_edit(str(it["item_id"]), float(it["value"])))
            value = "C" if value == 1.0 else "I"          # MATH's sympy scorer files CORRECT / INCORRECT letters
        usage = ModelUsage(input_tokens=int(it["prompt_tokens"]), output_tokens=int(it["completion_tokens"]))
        if texts and str(it["item_id"]) in texts:
            content, reasoning = texts[str(it["item_id"])]
            message_content: Any = (content or "") if reasoning is None else \
                [ContentReasoning(reasoning=reasoning), ContentText(text=content or "")]
        else:
            message_content = "" if it["content_empty_with_reasoning"] else "<think>reasoning</think>7"
        output = ModelOutput(model=model, usage=usage,
                             choices=[ChatCompletionChoice(message=ChatMessageAssistant(content=message_content),
                                                           stop_reason=it["stop_reason"])])
        meta = {f: bool(it[f]) for f in ("think_block_present", "think_block_unterminated", "stray_open_tag")}
        # the final attempt's model events (one per generate call Inspect made; the last is the one that returned
        # `output`, every earlier one exhausted its model-call retries on the client's request timeout)
        retries = list((model_retries or {}).get(str(it["item_id"]), [0]))
        raw = (raw_bodies or {}).get(str(it["item_id"]))
        call = None
        if raw is not None:
            raw_content, finish_reason = raw
            call = ModelCall(request={"model": vmodel, "messages": [{"role": "user", "content": "a MATH-500 problem"}]},
                             response={"id": "chatcmpl-fixture", "object": "chat.completion", "model": vmodel,
                                       "choices": [{"index": 0, "finish_reason": finish_reason,
                                                    "message": {"role": "assistant", "content": raw_content}}]})
        events = []
        for k, r in enumerate(retries):
            last = k == len(retries) - 1
            events.append(ModelEvent(
                model=model, input=[ChatMessageUser(content="a MATH-500 problem")], tools=[], tool_choice="auto",
                config=GenerateConfig(**gen), output=output if last else ModelOutput.from_content(model, ""),
                retries=int(r) or None, error=None if last else "Request timed out.", call=call if last else None))
        failed = (failed_attempts or {}).get(str(it["item_id"]), 0)
        messages = [FAILED_ATTEMPT_TIMEOUT_MESSAGE] * int(failed) if isinstance(failed, int) else [str(m) for m in failed]
        samples.append(EvalSample(
            id=str(it["item_id"]), epoch=1, input="a MATH-500 problem", target="7", output=output,
            scores=None if last_cancelled else {PRIMARY_SCORER: Score(value=value, metadata=meta)},
            model_usage={model: usage}, events=events,
            error_retries=[EvalRetryError(message=m, traceback="", traceback_ansi="") for m in messages],
            error=EvalError(message="CancelledError()", traceback="", traceback_ansi="") if last_cancelled else None))
    spec = EvalSpec(created="2026-01-01T00:00:00+00:00", task=TASK_NAME, model=model, config=EvalConfig(),
                    dataset=EvalDataset(name=TASK_NAME, samples=len(items)))
    log = EvalLog(eval=spec, plan=plan, status="cancelled" if cancelled else "success", stats=EvalStats(), samples=samples,
                  results=EvalResults(total_samples=len(items), completed_samples=len(kept) - (1 if cancelled else 0)),
                  error=EvalError(message="eval cancelled", traceback="", traceback_ansi="") if cancelled else None)
    log_dir = C.inspect_log_dir(cell_dir, TASK, sample_idx)
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{name}_{TASK_NAME}_{_hash8(name)}.eval"
    write_eval_log(log, path)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


# -- the fixture the tests share ------------------------------------------------

#: per (model key, cell) the correct-item set under the PRIMARY row: bare 300 /
#: prompted 298 / object 260 of 500 on every model. object - bare = -8.0 pp with
#: 40 bare-only correct items and 0 object-only (one-way: resolved, "lower");
#: prompted - bare = -0.4 pp with 20 bare-only and 18 prompted-only items (a
#: two-way discordance whose CI covers zero: "not resolved")
BARE_CORRECT = set(range(300))
PROMPTED_CORRECT = set(range(280)) | set(range(300, 318))
OBJECT_CORRECT = set(range(260))
CORRECT_BY_CELL: Mapping[str, set[int]] = {CELL_BARE: BARE_CORRECT, CELL_PROMPTED: PROMPTED_CORRECT,
                                           CELL_OBJECT: OBJECT_CORRECT}
#: a sensitivity row adds this many correct items to EVERY cell (the first
#: incorrect items in id order): general +10 (+2.0 pp), battery +5 (+1.0 pp)
SENSITIVITY_SHIFT: Mapping[str, int] = {MODE_GENERAL: 10, MODE_BATTERY: 5}
LAST4 = {496, 497, 498, 499}


#: the eight-arm extension's further arms (plan P5; `cells.EXT_CELLS` spellings) and their correct-item sets:
#: prompted constitution 296 / LoRA ctl2 302 / LoRA mix 290 of 500 — three DISTINCT readings beside the published three
#: the extension's VOCABULARY is read off the producer (the launcher's `cells.py`) — the cell ids, the extension cells and
#: models, the staged ids, the wave names and their out-root prefixes, the long-SI window — so no re-typed literal can drift
#: from the matrix; the store's own DERIVATIONS below (route, budget class, geometry by cell) stay its own, so the pin
#: test against `M5T_CELLS` still checks something. The steer stubs have no producer row yet and stay the store's.
CELL_PROMPTED_CONST, CELL_LORA_CTL2, CELL_LORA_MIX = CELLS_MODULE.CELL_PROMPTED_CONST, CELLS_MODULE.CELL_LORA_CTL2, CELLS_MODULE.CELL_LORA_MIX
CELL_ALPHASTEER, CELL_ARDITI = "alphasteer", "arditi"
EXT_CELLS: tuple[str, ...] = tuple(CELLS_MODULE.EXT_CELLS)
STEER_CELLS: tuple[str, ...] = (CELL_ALPHASTEER, CELL_ARDITI)
EXT_MODELS: tuple[str, ...] = tuple(CELLS_MODULE.EXT_MODELS)
STAGED_ID_BY_CELL: Mapping[str, str] = dict(CELLS_MODULE.STAGED_ID_BY_CELL)
WAVE_PUBLISHED, WAVE_EXT1, WAVE_STEER = CELLS_MODULE.WAVE_PUBLISHED, CELLS_MODULE.WAVE_EXT1, "steer"
STATUS_COMPLETE, STATUS_LAUNCH, STATUS_HELD, STATUS_DEFERRED = "complete", "launch", "held", "deferred"
OUT_ROOT_PREFIX_BY_WAVE: Mapping[str, str] = {**CELLS_MODULE.OUT_ROOT_PREFIX_BY_WAVE, WAVE_STEER: "controls_steer"}
#: the constitution rows' larger served window (contracts' `capability:gen_long_math_longsi`, plan F4) vs the published 17,664
MAX_MODEL_LEN_LONGSI = int(CELLS_MODULE.GEOMETRY_BY_CLASS[CELLS_MODULE.PROMPT_CLASS_LONG_SI][1])
#: the producer's own literals (the launcher's `cells.py`): the prompted SIs' full digests, the per-model cluster constraint,
#: the extension rows' one `--time` class — read, never re-typed, so the synthetic rows cannot drift from the matrix
SI_SHA256_BY_CELL: Mapping[str, str] = dict(CELLS_MODULE.SI_SHA256_BY_CELL)
CLUSTER_CONSTRAINT_BY_MODEL: Mapping[str, str] = dict(CELLS_MODULE.CLUSTER_CONSTRAINT_BY_MODEL)
TIME_EXT: str = str(CELLS_MODULE.TIME_EXT)
DEFERRED_REASON = "deferred: the AlphaSteer / Arditi rung waits for the measured minimal-over-refusal dose (D2/D3/D8/D14)"
#: E3: the it-doc row family's vocabulary, read off the producer (the launcher's `cells.py`): the wave, the `it` view, the object
#: source, the models and the three LoRA arms; the lane / sidecar / staged-id spellings are the admission record's (re-typed here
#: as the synthetic rows' own literals — the pin test against `M5T_CELLS` holds the KEY SET, the values are synthetic)
WAVE_ITDOC, VIEW_MAIN, VIEW_IT = CELLS_MODULE.WAVE_ITDOC, CELLS_MODULE.VIEW_MAIN, CELLS_MODULE.VIEW_IT
OBJECT_SOURCE_ITDOC = CELLS_MODULE.OBJECT_SOURCE_ITDOC
ITDOC_MODELS: tuple[str, ...] = tuple(CELLS_MODULE.ITDOC_MODELS)
ITDOC_CELLS: tuple[str, ...] = tuple(CELLS_MODULE.ITDOC_CELLS)
ITDOC_LANE = "itdoc_eb64"
ITDOC_HELD_REASON = "held (synthetic): the it-doc object is harvested, not staged; no bank id"
#: SYNTHETIC qualification prose (invented; the real rows' sentences live in the admission record, never here)
ITDOC_QUALIFICATION_BY_MODEL: Mapping[str, str] = {
    "gemma4e4b": "SYNTHETIC gemma qualification: the constitution fit is budget-limited (best_epoch 9 of 9); the control early-stopped at 2 of 4",
    "qwen35_4b": "SYNTHETIC qwen qualification: the constitution fit is budget-limited (best_epoch 16 of 16); the control early-stopped at 1 of 3",
}
ITDOC_QUALIFICATION_PATTERN = ("SYNTHETIC cross-model pattern: on every it-doc pair the constitution fit ran out its budget while the control "
                               "early-stopped; a difference of the pair inherits the asymmetry")
#: the it-doc legs' correct-item sets (distinct from the main-view arms' so an `it` contrast reads apart from its main twin)
CORRECT_BY_CELL_ITDOC: Mapping[str, set[int]] = {CELL_LORA_CTL2: set(range(298)), CELL_LORA_MIX: set(range(306)), CELL_OBJECT: set(range(294))}
CORRECT_BY_CELL_EXT: Mapping[str, set[int]] = {CELL_PROMPTED_CONST: set(range(276)) | set(range(300, 320)),
                                               CELL_LORA_CTL2: set(range(302)), CELL_LORA_MIX: set(range(290)),
                                               CELL_ALPHASTEER: set(range(285)), CELL_ARDITI: set(range(280))}


def design_rows(published: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """The published 24 rows in the EIGHT-ARM matrix's shape (the launcher's `cells.py` after E1, A5 / A12): every
    published key byte-identical, plus the extension keys the producer stamps — `wave: published`, `status: launch`
    (published-ness is the WAVE, `cells.COMPLETED_WAVES`, never a status), `view: main`, the served geometry the published
    legs ran at, the staged id of the fitted cell, the prompted SI's digest, the model's cluster constraint. On this tree
    it equals :func:`cells_matrix` key for key (test-pinned); given the 23-key PUBLISHED fixture rows it adds the keys."""
    out = []
    for r in (published if published is not None else cells_matrix()):
        row = dict(r)
        row.update(wave=WAVE_PUBLISHED, status=STATUS_LAUNCH, thinking_mode=C.THINKING_ON, view="main",
                   geometry_eval="capability:gen_long_math", geometry_L=MAX_MODEL_LEN, budget_class=str(r["cell"]) if not r["fitted"] else "none",
                   staged_id=STAGED_ID_BY_CELL.get(str(r["cell"])), si_sha256=SI_SHA256_BY_CELL.get(str(r["cell"])), log_requests=False,
                   cluster_constraint=CLUSTER_CONSTRAINT_BY_MODEL.get(str(r["model_key"]), "any"), extension=False)
        out.append(row)
    return out


def ext_rows(start: int | None = None, *, models: Sequence[str] = EXT_MODELS, cells: Sequence[str] = EXT_CELLS,
             wave: str = WAVE_EXT1, status: str = STATUS_LAUNCH, tree_sha: str | None = None) -> list[dict[str, Any]]:
    """The extension rows (plan P2 / §4.1: rows 24-32 of E1's matrix — per model the prompted constitution cell and the
    two further LoRAs, primary sampling row, thinking ON), wave-isolated under ``controls_<wave>/``, each fitted row
    pinned to its OWN staged object (`base__<cell>` of the pin file), the constitution rows at the long-SI window.
    Indexed from the end of the PUBLISHED wave by default (row 24 follows the 24 published rows whatever the tree's
    `M5T_CELLS` has grown to); with the defaults these are the launcher's rows 24-32 key for key (test-pinned)."""
    start = len(cells_matrix()) if start is None else start
    out: list[dict[str, Any]] = []
    for key in models:
        template = matrix_row(key, CELL_OBJECT, MODE_PRIMARY)
        for cell in cells:
            fitted = is_fitted(cell)
            staged = STAGED_ID_BY_CELL.get(cell)
            row = {
                "index": start + len(out), "job_id": f"m5t-{key}-{CELLS_MODULE.cell_short(cell)}", "model_key": key, "cell": cell,
                "cell_slug": cell_slug(cell), "harness_cell": harness_id_of(cell), "sampling_mode": MODE_PRIMARY,
                "sampling": dict(SAMPLING_ROWS[MODE_PRIMARY][key]), "fitted": fitted,
                "tuple_name": C.m5t_ext_tuple_name(key) if fitted else None,
                "route_kind": "lens" if fitted else "prompted", "served_positions_expected": "all" if fitted else None,
                "expected_stamps": list(template["expected_stamps"]) if fitted else list(matrix_row(key, CELL_PROMPTED, MODE_PRIMARY)["expected_stamps"]),
                "twin_expected_stamps": list(template.get("twin_expected_stamps") or []) if fitted else None,
                "out_root": f"{OUT_ROOT_PREFIX_BY_WAVE[wave]}/{key}/{cell_slug(cell)}/{MODE_PRIMARY}",
                "time_limit": TIME_EXT, "family": FAMILY_BY_KEY[key], "reasoning_parser": template.get("reasoning_parser"),
                "gpu_mem_util": template.get("gpu_mem_util"), "cluster_preference": template.get("cluster_preference"),
                "smoke_diagnostic_out": f"diagnostic/{FAMILY_BY_KEY[key]}", "split_mode_candidate": template.get("split_mode_candidate"),
                "object_sha256": staged_object_pin(key, staged) if staged else None,
                "wave": wave, "status": status, "thinking_mode": C.THINKING_ON, "view": "main",
                "geometry_eval": "capability:gen_long_math_longsi" if cell == CELL_PROMPTED_CONST else "capability:gen_long_math",
                "geometry_L": MAX_MODEL_LEN_LONGSI if cell == CELL_PROMPTED_CONST else MAX_MODEL_LEN,
                "budget_class": cell if not fitted else "none", "staged_id": staged,
                "si_sha256": SI_SHA256_BY_CELL.get(cell),
                "log_requests": True, "cluster_constraint": CLUSTER_CONSTRAINT_BY_MODEL.get(key, "any"), "extension": True,
            }
            if tree_sha is not None:
                row["tree_sha"] = tree_sha
            out.append(row)
    return out


def steer_rows(start: int, *, models: Sequence[str] = EXT_MODELS, status: str = STATUS_DEFERRED) -> list[dict[str, Any]]:
    """The AlphaSteer / Arditi stubs: design-only rows (`status: deferred` by default), never collected —
    no object pin yet (the admission record is empty), the external object source named."""
    out: list[dict[str, Any]] = []
    for key in models:
        template = matrix_row(key, CELL_OBJECT, MODE_PRIMARY)
        for cell in STEER_CELLS:
            out.append({
                "index": start + len(out), "job_id": f"m5t-{key}-{cell}", "model_key": key, "cell": cell, "cell_slug": cell,
                "harness_cell": cell, "sampling_mode": MODE_PRIMARY, "sampling": dict(SAMPLING_ROWS[MODE_PRIMARY][key]),
                "fitted": True, "tuple_name": None, "route_kind": "lens", "served_positions_expected": "all",
                "expected_stamps": None, "twin_expected_stamps": None,
                "out_root": f"{OUT_ROOT_PREFIX_BY_WAVE[WAVE_STEER]}/{key}/{cell}/{MODE_PRIMARY}", "time_limit": TIME_EXT,
                "family": FAMILY_BY_KEY[key], "reasoning_parser": template.get("reasoning_parser"), "gpu_mem_util": template.get("gpu_mem_util"),
                "cluster_preference": template.get("cluster_preference"), "smoke_diagnostic_out": f"diagnostic/{FAMILY_BY_KEY[key]}",
                "split_mode_candidate": template.get("split_mode_candidate"), "object_sha256": None,
                "wave": WAVE_STEER, "status": status, "thinking_mode": C.THINKING_ON, "view": "main",
                "geometry_eval": "capability:gen_long_math", "geometry_L": MAX_MODEL_LEN, "budget_class": "none", "staged_id": None,
                "si_sha256": None, "log_requests": True, "cluster_constraint": CLUSTER_CONSTRAINT_BY_MODEL.get(key, "any"), "extension": True,
                "object_source": "external",
                "deferred_reason": DEFERRED_REASON if status == STATUS_DEFERRED else None,
                "held_reason": None if status != STATUS_HELD else "held: no admission record (defence evidence missing)",
            })
    return out


def itdoc_pin(model_key: str, cell: str) -> str:
    """A SYNTHETIC 64-hex object digest for an it-doc leg (the admission record's pin stands in the real matrix)."""
    return hashlib.sha256(f"itdoc|{model_key}|{cell}".encode()).hexdigest()


def itdoc_rows(start: int, *, models: Sequence[str] = ITDOC_MODELS, cells: Sequence[str] = ITDOC_CELLS,
               launch: Iterable[tuple[str, str]] = (), pinned: Iterable[tuple[str, str]] | None = None) -> list[dict[str, Any]]:
    """The E3 it-doc rows in the launcher's shape (`cells.py` rows 47-52): per model the three LoRA arms of the `it` VIEW, primary
    sampling row, thinking ON, wave-isolated under ``controls_itdoc/``, `object_source: itdoc`, the lane, the object kind and its
    sidecar, the SYNTHETIC qualification (the model's sentence and the cross-model pattern) — every row `held` with a reason
    unless its (model, cell) is in ``launch``. A row's ``object_sha256`` is :func:`itdoc_pin` when its (model, cell) is in
    ``pinned`` (default: the launch rows — a hold carries no placeholder), else None. Each `it` row shares (model, cell, mode)
    with the main-view row of its cell: the analysis keys it apart by its view."""
    launch = set(launch)
    pinned = set(launch) if pinned is None else set(pinned)
    out: list[dict[str, Any]] = []
    for key in models:
        template = matrix_row(key, CELL_OBJECT, MODE_PRIMARY)
        for cell in cells:
            derived = cell.endswith("_minus_ctl")
            row = {
                "index": start + len(out), "job_id": f"m5t-{key}-it-{CELLS_MODULE.cell_short(cell)}", "model_key": key, "cell": cell,
                "cell_slug": cell_slug(cell), "harness_cell": harness_id_of(cell), "sampling_mode": MODE_PRIMARY,
                "sampling": dict(SAMPLING_ROWS[MODE_PRIMARY][key]), "fitted": True, "tuple_name": C.m5t_ext_tuple_name(key),
                "route_kind": "lens", "served_positions_expected": "all",
                "expected_stamps": list(template["expected_stamps"]), "twin_expected_stamps": list(template.get("twin_expected_stamps") or []),
                "out_root": f"{OUT_ROOT_PREFIX_BY_WAVE[WAVE_ITDOC]}/{key}/{cell_slug(cell)}/{MODE_PRIMARY}",
                "time_limit": TIME_EXT, "family": FAMILY_BY_KEY[key], "reasoning_parser": template.get("reasoning_parser"),
                "gpu_mem_util": template.get("gpu_mem_util"), "cluster_preference": template.get("cluster_preference"),
                "smoke_diagnostic_out": f"diagnostic/{FAMILY_BY_KEY[key]}", "split_mode_candidate": template.get("split_mode_candidate"),
                "object_sha256": itdoc_pin(key, cell) if (key, cell) in pinned else None,
                "wave": WAVE_ITDOC, "status": STATUS_LAUNCH if (key, cell) in launch else STATUS_HELD, "thinking_mode": C.THINKING_ON,
                "view": VIEW_IT, "geometry_eval": "capability:gen_long_math", "geometry_L": MAX_MODEL_LEN, "budget_class": "none",
                "staged_id": f"{'derived_' if derived else ''}itdoc64__{cell}", "si_sha256": None, "log_requests": True,
                "cluster_constraint": CLUSTER_CONSTRAINT_BY_MODEL.get(key, "any"), "extension": True,
                # the it-doc keys (the launcher's ITDOC_KEYS)
                "object_source": OBJECT_SOURCE_ITDOC, "lane": ITDOC_LANE, "object_kind": "derived" if derived else "fitted",
                "object_generation": "batch64",          # the record lane's objects are the batch-64 re-fits (cells.py stamps every it row)
                "bank_sidecar": "derived_meta.json" if derived else "train_budget_meta.json", "bank_view": None, "bank_view_rel": None,
                "bank_cell_dir": None, "parents_expected": None, "superseded": None, "stale_objects": {},
                "qualification": ITDOC_QUALIFICATION_BY_MODEL.get(key), "qualification_pattern": ITDOC_QUALIFICATION_PATTERN,
                "fit_record": None, "admission_notes": [],
                "held_reason": None if (key, cell) in launch else ITDOC_HELD_REASON,
            }
            out.append(row)
    return out


def eight_arm_matrix(*, with_steer: bool = True) -> list[dict[str, Any]]:
    """The fixture's matrix: the 24 published rows (`launch`, wave published — the launcher's own rows),
    the 9 extension rows (`launch`, wave ext1 — rows 24-32, the launcher's) and — by default —
    the 6 steering stubs (`deferred`, 33-38)."""
    rows = design_rows() + ext_rows()
    if with_steer:
        rows += steer_rows(len(rows))
    return rows


def default_kwargs(model_key: str, cell: str, mode: str, view: str = VIEW_MAIN) -> dict[str, Any]:
    shift = SENSITIVITY_SHIFT.get(mode, 0)
    base = CORRECT_BY_CELL_ITDOC[cell] if view == VIEW_IT else (CORRECT_BY_CELL.get(cell) or CORRECT_BY_CELL_EXT[cell])
    correct = set(base) | set(sorted(set(range(N)) - base)[:shift])
    kw: dict[str, Any] = {"correct": correct}
    if FAMILY_BY_KEY[model_key] == "glm4v":
        kw["no_think"] = {40, 41}          # GLM emits no tag on a couple of items: whole completion = answer
    return kw


def build_all(gen_root: Path, *, overrides: Mapping[tuple[str, str, str], Mapping[str, Any]] | None = None,
              rows: Sequence[Mapping[str, Any]] | None = None, **common: Any) -> dict[str, Path]:
    """Every leg of the matrix (24 by default); ``overrides[(model, cell, mode)]``
    replaces / extends that leg's ``build_store`` kwargs. A row of a NON-main
    view (E3) is keyed ``<model>/<cell>/<mode>@<view>`` in the result and takes
    ``overrides[(model, cell, mode, view)]`` — its main-view twin's triple never
    reaches it."""
    out: dict[str, Path] = {}
    for r in (rows if rows is not None else cells_matrix()):
        key, cell, mode = r["model_key"], r["cell"], r["sampling_mode"]
        # the analysis' keying rule: a row's view keys it apart on the LoRA arms; an external arm's `view` is its bank directory
        view = VIEW_MAIN if cell in STEER_CELLS else str(r.get("view") or VIEW_MAIN)
        if str(r.get("status") or STATUS_LAUNCH) not in (STATUS_LAUNCH, STATUS_COMPLETE):
            continue                # a held / deferred design row is never a store
        over = (overrides or {}).get((key, cell, mode, view), {}) if view != VIEW_MAIN else (overrides or {}).get((key, cell, mode), {})
        kw = {**default_kwargs(key, cell, mode, view), **common, **over}
        if rows is not None and r.get("wave") is not None:
            kw.setdefault("row", r)         # an eight-arm matrix row: build under its own out root / pin / window
        leg = f"{key}/{cell}/{mode}" + ("" if view == VIEW_MAIN else f"@{view}")
        out[leg] = build_store(gen_root, model_key=key, cell=cell, mode=mode, **kw)
    return out
