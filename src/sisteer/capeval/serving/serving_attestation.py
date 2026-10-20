"""``serving_attestation.json`` — the lora-native readiness contract's
record, its strict reader, and the pure per-NAME binding checks the
launcher runs before writing it.

WHAT THE FILE CLAIMS. A lora-native server passed readiness only after
three things held, per worker and per SERVING NAME:

1. the RESOLVED scheduler geometry, KV coverage, prefix caching, engine
   mode (``enforce_eager is False``, a CUDA-graph mode other than
   ``NONE``, no ``vllm_lens`` module imported) and LoRA config equal what
   the composer composed (``vllm_server.expected_scheduler_state`` under
   ``launch.lora_native``, checked by the unchanged
   ``validate_scheduler_reports``);
2. ``/v1/models`` lists the base model and EVERY adapter name exactly
   once, and each adapter card's ``root`` resolves to the directory the
   launch bound to that name (:func:`native_models_problems` — the
   frontend writes ``root=lora.path`` @0.25.1, so this is the frontend's
   own name -> path binding, checked per name: two names with swapped
   roots refuse);
3. for EVERY name, the worker report at that name's expected
   ``lora_int_id`` carries the fingerprint, rank and canonical module
   keys the launcher derived from that name's directory, every
   fingerprinted key names a module carrying a LoRA execution wrapper,
   and — canonical agreement not being application — every such key is
   backed by an EXACT-NAME binding: the worker reports, per adapter and
   per recognised execution module, the factor key vLLM's own lookup selects
   (``execution_bindings``; ``LoRAModel.get_lora`` is a plain
   ``loras.get(module_name)`` and ``activate_adapter`` calls ``reset_lora``
   on a miss, ``model_manager.py`` @0.25.1), and a fingerprinted layer
   whose wrapper selects nothing refuses (a factor loaded under an
   extra ``model.`` prefix canonicalised, fingerprinted and covered
   identically while the engine applied nothing on that layer)
   (:func:`native_lora_report_problems`). The expected id map is
   ``{name: i + 1}`` over the composer's ONE sorted ``--lora-modules``
   flag, because the frontend assigns ``lora_int_id = lora_id_counter.
   inc(1)`` in ``static_lora_modules`` order and ``LoRAParserAction``
   preserves argv order (``entrypoints/openai/models/serving.py``,
   ``cli_args.py`` @0.25.1). It is an ASSERTED expectation, fail-closed:
   a frontend that numbered differently fails step 3 for at least one
   name (distinct adapters have distinct fingerprints — the zero adapter's
   can never equal a genuine one's), and readiness refuses rather than
   attesting a wrong map. A permutation of two same-shaped adapters
   between ids (fingerprint multiset unchanged) therefore refuses naming
   both names and both ids; the earlier multiset comparison is gone.

The chain a request rides is then closed end to end: the proxy sets
``upstream["model"] = route.cell_id`` -> the frontend maps that name to
its ``lora_int_id`` (asserted against the composed order) -> the worker
applies that id's weights (asserted against the fingerprint expected
from the directory bound to that name, whose canonical module keys pin
layer index + projection).

THE READER IS STRICT. :func:`read_serving_attestation` requires every
top-level key, ``kind == "lora_native"``, ``schema_version == 1``,
``name_binding.asserted is True`` and ``engine.enforce_eager is False``;
anything else is a :class:`ServingAttestationError`. The lens path's
``steering_attestation.json`` and its readers are untouched — this is a
SEPARATE file with a discriminator, never a second shape under the old
name (three lens readers parse that path with lens expectations).
:func:`engine_identity` is the fixed sub-shape the routing proxy is
handed (``RoutingProxy`` started with ``engine_attestation=``; the two
are pinned equal).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

SERVING_ATTESTATION_FILE = "serving_attestation.json"
SERVING_ATTESTATION_KIND = "lora_native"
#: the two PEFT files every adapter directory carries — the ONE spelling the
#: exporter (``lora_export.ADAPTER_CONFIG`` / ``ADAPTER_WEIGHTS``), the worker
#: fingerprint (``worker_reports``) and the routing-table block (``proxy.
#: ADAPTER_CONFIG_FILE`` / ``ADAPTER_WEIGHTS_FILE``) all bind to; this module
#: is a stdlib-only leaf, so every one of them can import it.
ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_WEIGHTS_FILE = "adapter_model.safetensors"
SERVING_ATTESTATION_SCHEMA_VERSION = 2
#: the schema versions the strict reader accepts: 1 (every LoRA-only record
#: written before native-psr-serving) and 2 (adds the RESIDUAL SLOT block —
#: :data:`RESIDUAL_SLOT_KEY` — and the per-adapter residual identity in
#: :data:`ADAPTER_KEYS`). A v1 record reads as "no residual slot"
#: (:func:`engine_identity` -> ``residual_slot.armed False``, the per-adapter
#: residual fields ``None``); a v2 record without the block refuses.
READABLE_SCHEMA_VERSIONS = (1, 2)
#: THE RESIDUAL SLOT: the worker-side op
#: ``torch.ops.vllm.sisteer_residual_slot`` that serves PSR's gated write and
#: the plain steering-vector add at the decoder block output, routed by the
#: LoRA slot of a rank-1 ZERO o_proj carrier whose ``adapter_model.safetensors``
#: carries the payload. Its readiness RPC, the ONE op name, the attestation
#: block's key and its keys.
RESIDUAL_SLOT_KEY = "residual_slot"
RESIDUAL_SLOT_RPC_METHOD = "sisteer_residual_slot_report"
RESIDUAL_SLOT_OP_NAME = "vllm.sisteer_residual_slot"
RESIDUAL_SLOT_KEYS = ("armed", "rpc_method", "n_workers", "expected", "worker_reports")
#: THE LORA KERNEL INDEX CEILING (gemma certification canary, job 488932,
#: 2026-09-12). The pinned vLLM 0.25.1 LoRA Triton kernels
#: (``vllm/lora/ops/triton_ops/lora_shrink_op.py``, ``lora_expand_op.py``)
#: compute their pointer offsets in int32 — only the fp8 / MoE variants cast
#: to int64 — and ``kernel_warmup -> flashinfer_autotune -> _dummy_run`` pushes
#: a FULL-BUDGET batch (``max_num_batched_tokens = max_num_seqs x
#: max_model_len`` rows) through them. Rows x the widest adapted module's
#: in/out features above 2^31-1 overflows the offset and the engine core
#: dies at warmup with ``CUDA error: an illegal memory access was
#: encountered`` (588,544 x 4096 = 2.41e9 on gemma-4-E4B, whose eight
#: 512-wide global-attention heads give a 4096-wide o_proj input; an earlier
#: pass ran the same model + adapter + graphs + autotune clean at 262,144 x
#: 4096 = 1.07e9). The lens legs serve the same 589k-token geometry because
#: they never run those kernels. So: the composer REFUSES a lora-native
#: geometry over the ceiling (never clamps); the standup and the
#: certification driver compute the same ceiling BEFORE the first attempt
#: and lower S to ``ceiling // L`` under :data:`LORA_KERNEL_CEILING_REASON`;
#: the attestation records both numbers (additive keys, below).
LORA_KERNEL_INDEX_LIMIT = 2**31 - 1
LORA_KERNEL_CEILING_REASON = "lora_kernel_index_ceiling"
LORA_KERNEL_CEILING_KEY = "lora_kernel_index_ceiling"
LORA_MAX_MODULE_DIM_KEY = "lora_max_module_dim"
#: THE ADAPTED CLASS: the module suffixes the
#: composed ``--lora-target-modules`` restricts the LoRA execution wrappers
#: to. With no restriction vLLM 0.25.1 wraps EVERY linear projection and
#: dispatches the LoRA kernels on each wrapper whenever the batch carries
#: LoRA requests (an adapter without that module's factors merely gets
#: ``reset_lora()``), so the int32 index ceiling above would have to be
#: sized from the WIDEST wrapper — gemma-4-E4B's packed ``gate_up_proj``
#: output stride is 20,480, five times the adapted o_proj's 4,096. The
#: launch composes the class off the registered adapters, readiness asserts
#: the resolved config equals it and holds every EXECUTING wrapper's stride
#: to :data:`LORA_MAX_MODULE_DIM_KEY` (:func:`execution_stride_problems`);
#: the attestation records the class (additive top-level key) and the
#: executing-stride maximum (``execution_modules.stride_max``).
LORA_TARGET_MODULES_KEY = "lora_target_modules"
#: THE LORA ROUTING RECORDER's three ``collective_rpc`` methods on the native
#: worker extension (``native_worker.NativeWorkerExtension``): arm,
#: observe, disarm. Named HERE (the stdlib-only leaf) because both the
#: launcher (``vllm_server``, over the launch's authenticated RPC route) and
#: the routing proxy (its allowlisted pass-through for the gate canary, which
#: attaches to a running standup and holds no launch token) need them.
LORA_ROUTING_ARM_RPC_METHOD = "sisteer_arm_lora_routing_recorder"
LORA_ROUTING_OBSERVE_RPC_METHOD = "sisteer_lora_routing_observations"
LORA_ROUTING_DISARM_RPC_METHOD = "sisteer_disarm_lora_routing_recorder"
LORA_ROUTING_RPC_METHODS = (LORA_ROUTING_ARM_RPC_METHOD, LORA_ROUTING_OBSERVE_RPC_METHOD,
                            LORA_ROUTING_DISARM_RPC_METHOD)
#: the selection control's structural verdict label (``positive_controls.json``
#: ``controls.selection_honored.verdict``; the admission holds it)
SELECTION_VERDICT_ROUTING_ATTESTED = "routing_attested"
#: THE ENGINE's per-request id suffix at the pinned vLLM 0.25.1 (canary #4
#: job 489533 read 21/21 requests "missing"):
#: ``AsyncLLM.add_request`` (``v1/engine/async_llm.py:368``) calls
#: ``InputProcessor.assign_request_id`` (``v1/engine/input_processor.py:223``),
#: which moves the client-visible id to ``external_req_id`` and sets
#: ``request_id = f"{external_req_id}-{random_uuid():.8}"`` — ``-`` plus 8
#: lowercase hex — exactly ONCE, unless ``VLLM_DISABLE_REQUEST_ID_RANDOMIZATION``
#: (never set on our launches: production parity). The chat completion's ``id``
#: is the EXTERNAL id; every id the scheduler and the runners see is the
#: INTERNAL one. ``scripts/lens_vectorize_gate.py`` pins the same contract for
#: the lens path's SelectionRecord (``is_engine_id_of``); a test holds the two
#: equal so the contract has one meaning in the tree.
ENGINE_REQUEST_ID_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}\Z")
#: THE BOUNDARY-REFUSAL CONTROL's one probe body (O49): the proxy admits
#: the control's marker ONLY beside EXACTLY these messages and kwargs — a single user turn under
#: ``add_generation_prompt: false``, greedy, four tokens — so no other body (a ``[user, assistant,
#: user]`` transcript whose earlier assistant header the locator would find, a different prompt) can
#: ride the exception. The proxy and the standup read the same constants; the marker's VALUE is a
#: per-standup nonce the proxy mints, so a client outside the standup cannot present it.
BOUNDARY_REFUSAL_CONTROL_PROMPT = ("boundary-refusal control: this single user turn is rendered without the "
                                   "assistant header; the worker must ledger it and the proxy must refuse it.")
BOUNDARY_REFUSAL_CONTROL_MESSAGES = ({"role": "user", "content": BOUNDARY_REFUSAL_CONTROL_PROMPT},)
BOUNDARY_REFUSAL_CONTROL_KWARGS = {"max_tokens": 4, "temperature": 0, "add_generation_prompt": False}


def is_engine_id_of(engine_id: Any, external_id: Any) -> bool:
    """Whether engine request id ``engine_id`` is THIS response's request: the
    completion ``id`` itself (randomization disabled / an offline record), or
    that id plus exactly ONE engine suffix (:data:`ENGINE_REQUEST_ID_SUFFIX_RE`
    — the delimiter, the width and the alphabet as the runtime mints them;
    nothing before, nothing after). A different request's id — one merely
    sharing a prefix included — another delimiter, a second suffix or a
    parallel-sampling child (``<i>_<id>-…``) is not. An empty or missing id
    on either side matches nothing."""
    if engine_id is None or external_id is None:
        return False
    r, x = str(engine_id), str(external_id)
    if not r or not x:
        return False
    if r == x:
        return True
    return r.startswith(x) and ENGINE_REQUEST_ID_SUFFIX_RE.fullmatch(r, len(x)) is not None


def external_request_id_candidates(engine_id: Any) -> Tuple[str, ...]:
    """The external ids an engine id CAN belong to under :func:`is_engine_id_of`:
    the id itself and, when it wears exactly the engine suffix, the id with
    that suffix stripped. Lets a join over thousands of engine ids run in one
    pass (look each candidate up in the expected set) instead of every pair."""
    if engine_id is None:
        return ()
    r = str(engine_id)
    if not r:
        return ()
    m = ENGINE_REQUEST_ID_SUFFIX_RE.search(r)
    if m is not None and m.start() > 0:
        return (r, r[:m.start()])
    return (r,)


#: THE TWO MODEL RUNNERS the pinned vLLM 0.25.1 GPU worker constructs
#: (``v1/worker/gpu_worker.py:384-398``: ``use_v2_model_runner`` picks
#: ``vllm.v1.worker.gpu.model_runner.GPUModelRunner`` (V2, ``self.req_states`` +
#: ``self.lora_state``, no ``input_batch``) else
#: ``vllm.v1.worker.gpu_model_runner.GPUModelRunner`` (V1, ``self.input_batch``)).
#: Canary #4 (job 489533) ran the V2 runner. The routing recorder reads its
#: PRIMARY evidence from the ``scheduler_output`` both consume and cross-checks
#: against the runner's own state by kind; any other class is an unpinned
#: engine — recorded here, refused by the check and by readiness.
MODEL_RUNNER_CLASSES = {
    "vllm.v1.worker.gpu_model_runner.GPUModelRunner": "v1",
    "vllm.v1.worker.gpu.model_runner.GPUModelRunner": "v2",
}
MODEL_RUNNER_KINDS = tuple(sorted(set(MODEL_RUNNER_CLASSES.values())))
#: the additive attestation key recording which runner served (class + kind)
MODEL_RUNNER_KEY = "model_runner"
#: the basis the file records for its ASSERTED id -> name map.
NAME_BINDING_BASIS = (
    "composed --lora-modules order == frontend lora_id_counter order "
    "(entrypoints/openai/models/serving.py init_static_loras -> "
    "load_lora_adapter, lora_id_counter.inc(1) @0.25.1); per-name "
    "fingerprint + rank + module_keys + /v1/models root all held")
#: the bf16 tensor dtype every worker adapter tensor must report.
EXPECTED_TENSOR_DTYPE = "torch.bfloat16"

REQUIRED_TOP_KEYS = (
    "kind", "schema_version", "model", "revision", "attested_at",
    "server_pid", "hostname", "slurm_job_id", "argv_sha256", "host", "port",
    "nonce", "expected", "expected_native_lora", "scheduler_geometry",
    "engine", "lora_config", "adapters", "worker_adapters", "id_name_map",
    "name_binding", "execution_modules")
#: a schema-2 record carries the residual slot block as well
REQUIRED_TOP_KEYS_V2 = REQUIRED_TOP_KEYS + (RESIDUAL_SLOT_KEY,)
#: ``cudagraph_capture_sizes`` / ``max_cudagraph_capture_size`` are the
#: RESOLVED capture surface (code-review finding, 2026-09-11): the
#: dispatcher runs any batch above the capture max EAGER while the mode
#: stays non-NONE, so both are attested and held to ``max_num_seqs``
#: (:func:`cudagraph_coverage_problems`).
ENGINE_KEYS = ("enforce_eager", "cudagraph_mode", "cudagraph_specialize_lora",
               "cudagraph_capture_sizes", "max_cudagraph_capture_size",
               "lens_plugin_imported", "enable_prefix_caching")
#: THE FULL CAPTURE SURFACE (2026-09-16, additive): ``cudagraph_full_capture_
#: sizes`` / ``cudagraph_full_capture_max`` are the token counts of the
#: dispatcher's FULL keys (``model_runner.cudagraph_dispatcher.cudagraph_keys
#: [FULL]``, reported by ``worker_reports.cudagraph_capture_coverage_report``)
#: — the batch sizes that actually dispatch to a FULL graph. vLLM captures
#: FULL decode graphs only for capture sizes ``<= max_num_seqs``, so a
#: default list that skips S leaves the widest decode batches PIECEWISE with
#: attention eager while ``cudagraph_capture_sizes`` / the max read fine (the
#: S=51 collapse: max 96, FULL largest 48). The builder LIFTS them from the
#: report (a report without them is not a lora-native report); readiness holds
#: the max to ``max_num_seqs`` (:func:`cudagraph_coverage_problems`); the
#: strict reader TOLERATES their absence on a record written before this
#: landing (schema unchanged — additive keys, never a required-key bump) and
#: refuses a malformed value when present. ``engine_identity`` (the proxy's
#: exact key set) is unchanged: the surface is a readiness fact, not routing
#: identity.
ENGINE_CAPTURE_COVERAGE_KEYS = ("cudagraph_full_capture_sizes", "cudagraph_full_capture_max")
#: ``target_modules`` is the resolved deployment-time LoRA restriction
#: (``--lora-target-modules``); asserted ``None`` by the launcher.
LORA_CONFIG_KEYS = ("max_loras", "max_lora_rank", "max_cpu_loras",
                    "lora_dtype", "fully_sharded_loras", "target_modules")
ADAPTER_KEYS_V1 = ("dir", "config_sha256", "weights_sha256", "fingerprint_bf16",
                   "r", "lora_int_id", "module_keys", "petri_intervention_hash",
                   "petri_zero_adapter", "models_root")
#: schema 2 adds the adapter's OBJECT KIND (``petri_source_kind``; ``None`` on
#: the zero adapter) and its RESIDUAL identity — ``residual_fingerprint`` (the
#: payload's sha256 beside the zero carrier; ``None`` for a plain LoRA) and
#: ``position_mode`` (``None`` for a plain LoRA). Every residual entry on one
#: base shares the SAME carrier ``fingerprint_bf16`` (a rank-1 zero pair), so
#: adapter identity is the PAIR ``(fingerprint_bf16, residual_fingerprint)``.
ADAPTER_KEYS = ADAPTER_KEYS_V1 + ("petri_source_kind", "residual_fingerprint", "position_mode")
#: ``execution_bindings`` is the per-adapter
#: ``{execution_module_name: factor_key | None}`` map vLLM's own exact-name
#: lookup resolves — the fact that closes canonical-key agreement to actual
#: application.
WORKER_ADAPTER_KEYS = ("id", "rank", "n_modules", "module_keys",
                       "tensor_dtype", "fingerprint_bf16", "execution_bindings")
#: the EXECUTION side of the adapters (``worker_reports.
#: lora_execution_modules_report``): the canonical keys AND the exact
#: registry names of the model's wrapped recognised modules, the registry size,
#: and any wrapped module of another family by raw name. Every adapter's
#: fingerprinted module keys are held to ``module_keys`` per worker, and
#: every one of them to a real binding on a module in ``module_names``.
#: ``strides`` is ``{exact_name: {"in": input_size, "out":
#: sum(output_slices)}}`` for EVERY execution wrapper — o_proj and any other
#: family — and ``stride_max`` the largest of those; readiness holds it to
#: the composed ``lora_max_module_dim``.
EXECUTION_MODULES_KEYS = ("module_keys", "module_names", "n_modules", "other",
                          "strides", "stride_max")
#: the report keys the launcher lifts into ``lora_config`` /
#: ``execution_modules`` (``engine`` lifts :data:`ENGINE_KEYS` by name).
_REPORT_LORA_KEYS = {"max_loras": "lora_max_loras",
                     "max_lora_rank": "lora_max_lora_rank",
                     "max_cpu_loras": "lora_max_cpu_loras",
                     "lora_dtype": "lora_dtype",
                     "fully_sharded_loras": "lora_fully_sharded",
                     "target_modules": "lora_target_modules"}
_REPORT_EXECUTION_KEYS = {"module_keys": "lora_execution_module_keys",
                          "module_names": "lora_execution_module_names",
                          "n_modules": "lora_execution_modules_n",
                          "other": "lora_execution_modules_other",
                          "strides": "lora_execution_module_strides",
                          "stride_max": "lora_execution_module_stride_max"}


class ServingAttestationError(RuntimeError):
    """A malformed, foreign, or non-asserting attestation — refused,
    never read as a weaker one."""


def sha256_bytes(data: bytes) -> str:
    """64-hex sha256 over ``data``."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    """64-hex sha256 over a file's bytes (the adapter digests' definition:
    the ONE reading the adapter bank, the routing-table block, the proxy's
    start-time re-hash, the gate's object inputs and the certification
    records all use), streamed in 1 MiB chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, obj: Any, *,
                      default: Optional[Callable[[Any], Any]] = None) -> Path:
    """Publish ``obj`` at ``path`` as indented, key-sorted JSON so a reader
    sees the whole document or none: written to a sibling temp name,
    fsynced, then ``os.replace``d. The ONE writer behind the attestations,
    the adapter bank's ``adapter_digests.json``, the lora-native standup's
    records and every certification record; ``default`` is
    ``json.dumps``'s."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, indent=2, sort_keys=True, default=default))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    return path


# --- (2) the frontend's own name -> path binding, per name -------------------------


def native_models_problems(body: Any, *, model: str,
                           adapter_dirs: Mapping[str, str]) -> List[str]:
    """Problems with a ``/v1/models`` body for a lora-native launch;
    ``[]`` when it lists ``model`` and every adapter name EXACTLY once
    and every adapter card's ``root`` resolves to ``adapter_dirs[name]``.
    The frontend registers every static adapter BEFORE the port opens
    (``init_app_state`` awaits ``init_static_loras`` ahead of
    ``serve_http`` @0.25.1), so a 200 that lacks a card is a real
    failure, not a race."""
    if not isinstance(body, Mapping) or not isinstance(body.get("data"), list):
        return ["/v1/models body is not a model list"]
    cards = [m for m in body["data"] if isinstance(m, Mapping)]
    ids = [m.get("id") for m in cards]
    problems: List[str] = []
    n_base = ids.count(model)
    if n_base != 1:
        problems.append(
            f"base model {model!r} listed {n_base} times in /v1/models, "
            "expected exactly once")
    for name in sorted(adapter_dirs):
        n = ids.count(name)
        if n != 1:
            problems.append(
                f"adapter {name!r} listed {n} times in /v1/models, expected "
                "exactly once (the frontend registers every static adapter "
                "before it serves, so a missing card means the adapter did "
                "not load; a duplicate means two registrations share the "
                "name)")
            continue
        card = next(m for m in cards if m.get("id") == name)
        root = card.get("root")
        if not isinstance(root, str) or not root:
            problems.append(
                f"adapter {name!r}: its /v1/models card carries no root "
                "path, so the frontend's name -> path binding cannot be "
                "checked")
            continue
        want = Path(adapter_dirs[name]).resolve()
        got = Path(root).resolve()
        if got != want:
            problems.append(
                f"adapter {name!r}: /v1/models root {root!r} resolves to "
                f"{got}, expected {want} — the frontend's name -> path "
                "binding does not match the composed one")
    return problems


def models_roots(body: Any, names: Sequence[str]) -> Dict[str, str]:
    """``{name: root}`` off a ``/v1/models`` body that already passed
    :func:`native_models_problems`."""
    cards = [m for m in body["data"] if isinstance(m, Mapping)]
    out: Dict[str, str] = {}
    for name in names:
        card = next(m for m in cards if m.get("id") == name)
        out[name] = str(card["root"])
    return out


# --- (3) the launcher's expectation and the per-name binding check -----------------


#: the safetensors key prefix of a RESIDUAL-slot payload beside an adapter's
#: LoRA factors (``native_residual.PAYLOAD_PREFIX`` spells the same string —
#: pinned equal by test) and the fields it may carry
RESIDUAL_KEY_PREFIX = "sisteer_residual."
#: (``gate_w2`` / ``gate_b2`` / ``direction2`` are THE PAIRED WRITE's second side; pinned equal
#: to ``native_residual.RESIDUAL_FIELDS`` by test)
RESIDUAL_KEY_FIELDS = ("gate_w", "gate_b", "direction", "gate_w2", "gate_b2", "direction2", "vector")
#: the two key classes :func:`classify_adapter_key` returns
ADAPTER_KEY_LORA = "lora"
ADAPTER_KEY_RESIDUAL = "residual"


def classify_adapter_key(key: str) -> Tuple[str, Any, str]:
    """THE ONE CLASSIFIER of an adapter file's safetensors keys: every bank
    reader used to parse EVERY key as a
    LoRA factor, so a residual payload beside the rank-1 zero pair refused
    the whole entry. Returns ``("lora", module_name, "A"|"B")`` for a plain
    ``<module>.lora_A.weight`` / ``.lora_B.weight`` key (the admitted exports'
    shape; whether the module is a RECOGNISED adapted module —
    :func:`module_key_of` — is the consumer's check, as today; the routed
    experts' PEFT ``ParamWrapper`` layout is the same key shape under the
    module names ``...mlp.experts.base_layer`` (gate_up) and ``...mlp.experts``
    (down), contract §3), or ``("residual", layer, field)`` for
    ``sisteer_residual.layers.<l>.<field>`` with ``l`` a non-negative int and
    ``field`` one of :data:`RESIDUAL_KEY_FIELDS`. Anything else refuses naming
    the key: an unknown field, a missing / non-integer layer, a residual key
    of another shape, or a key that is neither."""
    if not isinstance(key, str) or not key:
        raise ServingAttestationError(f"adapter key must be a non-empty string, got {key!r}")
    if key.startswith(RESIDUAL_KEY_PREFIX):
        parts = key[len(RESIDUAL_KEY_PREFIX):].split(".")
        if len(parts) != 3 or parts[0] != "layers":
            raise ServingAttestationError(
                f"safetensors key {key!r} is not a residual payload key of the form "
                f"{RESIDUAL_KEY_PREFIX}layers.<layer>.<field>")
        if not parts[1].isdigit():
            raise ServingAttestationError(
                f"safetensors key {key!r}: layer {parts[1]!r} is not a non-negative integer")
        if parts[2] not in RESIDUAL_KEY_FIELDS:
            raise ServingAttestationError(
                f"safetensors key {key!r}: field {parts[2]!r} is not one of "
                f"{list(RESIDUAL_KEY_FIELDS)}")
        return ADAPTER_KEY_RESIDUAL, int(parts[1]), parts[2]
    for suffix, which in ((".lora_A.weight", "A"), (".lora_B.weight", "B")):
        if key.endswith(suffix) and len(key) > len(suffix):
            return ADAPTER_KEY_LORA, key[:-len(suffix)], which
    raise ServingAttestationError(
        f"safetensors key {key!r} is neither a lora_A/lora_B weight nor a "
        f"{RESIDUAL_KEY_PREFIX}* payload key — the admitted exports write only those")


# --- the adapted-module vocabulary ---------------------------------------------------
#
# The native attestation used to recognise ONE module family (a decoder layer's
# attention o_proj). The module sets of the companion repo adapt q/k/v/o, the
# dense / shared-expert MLP projections and the routed experts, and vLLM
# 0.25.1 EXECUTES several of those under a different module than the file
# names them: q/k/v under the packed ``qkv_proj`` wrapper
# (``MergedQKVParallelLinearWithLoRA``), gate/up under the packed
# ``gate_up_proj`` wrapper (``MergedColumnParallelLinearWithLoRA``), and BOTH
# routed-expert projections under the one ``...mlp.experts``
# ``FusedMoE3DWithLoRA`` wrapper (``model_manager.py:720-800, 800-880``
# @0.25.1). So there are two vocabularies, related by a fixed map:
#
# * LOGICAL keys — one per adapted projection, what the fingerprint hashes and
#   what an adapter's ``module_keys`` list: ``L<N>.self_attn.{q,k,v,o}_proj``
#   (NemotronH: ``mixer.*``), ``L<N>.mlp.{gate,up,down}_proj`` (dense),
#   ``L<N>.mlp.shared_expert.{gate,up,down}_proj`` (MoE shared expert),
#   ``L<N>.mlp.experts.gate_up_proj`` / ``L<N>.mlp.experts.down_proj`` (the
#   two 3-D routed-expert projections).
# * EXECUTION keys — one per wrapper in the engine's registry: the logical key
#   itself for a plain projection, ``L<N>.self_attn.qkv_proj`` /
#   ``L<N>.mlp[.shared_expert].gate_up_proj`` for the packed wrappers,
#   ``L<N>.mlp.experts`` for the MoE wrapper.
#
# :func:`execution_carrier_key` is the map (logical -> execution) and
# :func:`carried_keys` its inverse over a key list; :func:`module_key_of`
# canonicalises a module NAME in any of the spellings the two sides use (the
# file's HF/PEFT ``base_model.model.model[.language_model].layers.N...``, vLLM's
# post-mapper ``language_model.model.layers.N...`` / ``model.layers.N...``).
# Everything here is string algebra on purpose: this module is the stdlib-only
# leaf ``worker_reports`` imports at load.

#: the decoder block's FFN attribute names the module vocabulary admits — ONE
#: tuple for the fitter's capability probe (``extraction.lora_modules._ffn_layout``
#: reads it from here) and for the canonical keys below, so an architecture the
#: fitter adapts (a ``feed_forward`` / ``block_sparse_moe`` block) is never
#: refused at readiness by a fingerprint that only knew ``mlp``. Each may
#: carry a ``shared_expert`` sub-module and/or a 3-D ``experts`` wrapper.
FFN_PARENTS: Tuple[str, ...] = ("mlp", "feed_forward", "block_sparse_moe", "mixer")
_FFN_PARENT_RE = "|".join(re.escape(p) for p in FFN_PARENTS)
#: a module name's site path after ``layers.<N>.`` — the recognised adapted
#: modules and nothing else (a router ``gate``, ``shared_expert_gate``,
#: ``conv1d``, the GDN ``in_proj_*`` / ``out_proj`` are never guessed into a key)
MODULE_NAME_RE = re.compile(
    r"(?:^|\.)layers\.(?P<layer>\d+)\."
    r"(?P<site>"
    r"(?:self_attn|mixer)\.(?:q_proj|k_proj|v_proj|o_proj|qkv_proj)"
    r"|(?:" + _FFN_PARENT_RE + r")(?:\.shared_expert)?\.(?:gate_proj|up_proj|down_proj|gate_up_proj)"
    r"|(?:" + _FFN_PARENT_RE + r")\.experts(?:\.base_layer)?"
    r")$")
#: a canonical key: ``L<N>.<site>``
MODULE_KEY_RE = re.compile(r"^L(?P<layer>\d+)\.(?P<site>[A-Za-z_][A-Za-z0-9_.]*)$")
#: vLLM's packed wrappers and their adapter-visible components IN SLICE ORDER
#: (``packed_modules_mapping`` of every llama-style model in 0.25.1: gemma-4,
#: Qwen3.5 ``qwen3_5.py:278-289``, NemotronH; a ``PackedLoRALayerWeights``'s
#: list slots and a wrapper's ``output_slices`` follow this order)
PACKED_COMPONENTS: Dict[str, Tuple[str, ...]] = {
    "qkv_proj": ("q_proj", "k_proj", "v_proj"),
    "gate_up_proj": ("gate_proj", "up_proj"),
}
#: the two logical projections of a routed-experts (``FusedMoE3DWithLoRA``)
#: wrapper, in the order vLLM's ``_stack_moe_lora_weights`` leaves them in the
#: wrapper entry's list-valued factors (``[gate_up, down]``)
EXPERTS_PROJECTIONS: Tuple[str, str] = ("gate_up_proj", "down_proj")
#: the leaf of an experts wrapper's module name and of its canonical key
EXPERTS_LEAF = "experts"
#: PEFT's ``ParamWrapper`` nesting: ``<experts>.base_layer`` is the FIRST
#: wrapped parameter (gate_up_proj), ``<experts>`` itself the second (down_proj)
#: — ``homework/05`` §1d; ``model_manager.py:835-843`` hard-codes the same
EXPERTS_BASE_LAYER_SUFFIX = ".base_layer"
#: the module kinds :func:`module_kind_of_key` returns
MODULE_KIND_LINEAR = "linear"                 # one projection, one 2-D factor pair
MODULE_KIND_PACKED = "packed"                 # qkv_proj / gate_up_proj: a wrapper over components
MODULE_KIND_EXPERTS = "experts"               # the MoE wrapper: both 3-D projections
MODULE_KIND_EXPERTS_PROJECTION = "experts_projection"   # one 3-D routed-expert projection


def module_key_of(module_name: Any) -> Optional[str]:
    """The canonical LOGICAL-or-EXECUTION key of a module name, or ``None``
    when the name is not a recognised adapted module. ``...layers.N.<site>``
    -> ``L<N>.<site>`` for every site :data:`MODULE_NAME_RE` admits, in any
    prefix spelling; the ONE respelling is PEFT's routed-experts nesting:
    ``...layers.N.mlp.experts.base_layer`` (the gate_up ``ParamWrapper``) ->
    ``L<N>.mlp.experts.gate_up_proj``. ``...layers.N.mlp.experts`` keeps the
    wrapper key ``L<N>.mlp.experts``: in a FILE it is the down projection
    (contract §3), on a worker after ``_stack_moe_lora_weights`` it carries
    both projections as list-valued factors — the fingerprint decides by the
    value's shape, the key names the module."""
    if not isinstance(module_name, str) or not module_name:
        return None
    m = MODULE_NAME_RE.search(module_name)
    if m is None:
        return None
    site = m.group("site")
    if site.endswith(EXPERTS_BASE_LAYER_SUFFIX):
        site = site[:-len(EXPERTS_BASE_LAYER_SUFFIX)] + "." + EXPERTS_PROJECTIONS[0]
    return f"L{int(m.group('layer'))}.{site}"


def parse_module_key(key: str) -> Tuple[int, str, str]:
    """``(layer, site_prefix, leaf)`` of a canonical key: ``L3.mlp.experts``
    -> ``(3, "mlp", "experts")``, ``L3.mlp.experts.gate_up_proj`` ->
    ``(3, "mlp.experts", "gate_up_proj")``, ``L3.self_attn.o_proj`` ->
    ``(3, "self_attn", "o_proj")``. Refuses a string that is not a key."""
    m = MODULE_KEY_RE.match(key) if isinstance(key, str) else None
    if m is None:
        raise ServingAttestationError(f"{key!r} is not a canonical module key (L<N>.<site>)")
    site = m.group("site")
    prefix, _, leaf = site.rpartition(".")
    return int(m.group("layer")), prefix, leaf


def _kind_of_parts(prefix: str, leaf: str) -> str:
    """:func:`module_kind_of_key` over an already-parsed ``(prefix, leaf)``
    (the helpers below parse once and classify from the parts)."""
    if leaf == EXPERTS_LEAF:
        return MODULE_KIND_EXPERTS
    if prefix.endswith("." + EXPERTS_LEAF) and leaf in EXPERTS_PROJECTIONS:
        return MODULE_KIND_EXPERTS_PROJECTION
    if leaf in PACKED_COMPONENTS:
        return MODULE_KIND_PACKED
    return MODULE_KIND_LINEAR


def module_kind_of_key(key: str) -> str:
    """Which of the four kinds a canonical key names (module constants)."""
    _, prefix, leaf = parse_module_key(key)
    return _kind_of_parts(prefix, leaf)


def packed_component_keys(packed_key: str) -> Tuple[str, ...]:
    """The logical component keys of a packed execution key, in slice order:
    ``L3.self_attn.qkv_proj`` -> ``(L3.self_attn.q_proj, L3.self_attn.k_proj,
    L3.self_attn.v_proj)``. Refuses a key that is not a packed wrapper's."""
    layer, prefix, leaf = parse_module_key(packed_key)
    if _kind_of_parts(prefix, leaf) != MODULE_KIND_PACKED:
        raise ServingAttestationError(f"{packed_key!r} is not a packed wrapper key")
    return tuple(f"L{layer}.{prefix}.{c}" for c in PACKED_COMPONENTS[leaf])


def experts_projection_keys(experts_key: str) -> Tuple[str, str]:
    """The two logical projection keys of an experts wrapper key, in vLLM's
    list order: ``L3.mlp.experts`` -> ``(L3.mlp.experts.gate_up_proj,
    L3.mlp.experts.down_proj)``."""
    if module_kind_of_key(experts_key) != MODULE_KIND_EXPERTS:
        raise ServingAttestationError(f"{experts_key!r} is not an experts wrapper key")
    return tuple(f"{experts_key}.{p}" for p in EXPERTS_PROJECTIONS)  # type: ignore[return-value]


def execution_carrier_key(key: str) -> str:
    """The EXECUTION key vLLM applies a logical key under: a q/k/v component
    -> its ``qkv_proj`` wrapper, a gate/up component -> its ``gate_up_proj``
    wrapper, an experts projection -> the ``...mlp.experts`` wrapper, every
    other key (o_proj, down_proj, a packed key, an experts key) itself. Pure
    key algebra — whether the engine actually packs is what the worker's
    execution registry reports; :func:`carried_keys` accepts both worlds."""
    layer, prefix, leaf = parse_module_key(key)
    if _kind_of_parts(prefix, leaf) == MODULE_KIND_EXPERTS_PROJECTION:
        return f"L{layer}.{prefix}"
    for packed, components in PACKED_COMPONENTS.items():
        if leaf in components:
            return f"L{layer}.{prefix}.{packed}"
    return key


def carried_keys(execution_key: str, keys: Sequence[str]) -> List[str]:
    """The logical keys among ``keys`` that the wrapper at ``execution_key``
    executes: a key equal to it (an unpacked engine, or o_proj / down_proj)
    or one whose :func:`execution_carrier_key` is it (a packed component, an
    experts projection). Sorted."""
    out = []
    for k in keys:
        try:
            if k == execution_key or execution_carrier_key(k) == execution_key:
                out.append(k)
        except ServingAttestationError:
            continue
    return sorted(out)


def _carriers_of(keys: Sequence[str]) -> Dict[str, Optional[str]]:
    """``{logical key: its execution carrier}`` over an adapter's fingerprinted
    keys, computed ONCE per adapter (``None`` for a string that is not a
    canonical key — it then carries nothing, as :func:`carried_keys` treats
    it). The per-worker checks below are dict lookups over this map instead
    of a regex parse per (key × execution module) pair."""
    out: Dict[str, Optional[str]] = {}
    for k in keys:
        try:
            out[k] = execution_carrier_key(k)
        except ServingAttestationError:
            out[k] = None
    return out


def _by_value(mapping: Mapping[str, str]) -> Dict[str, List[str]]:
    """The inverse index of a ``{name: canonical key}`` map: ``{key: [names]}``."""
    out: Dict[str, List[str]] = {}
    for name, key in mapping.items():
        out.setdefault(key, []).append(name)
    return out


def is_in_target_class(module_name: str, targets: Sequence[str]) -> bool:
    """vLLM's ``is_in_target_modules`` (``lora/utils.py:271-311`` @0.25.1)
    over the composed ``--lora-target-modules`` class, with the harness's
    :data:`PACKED_COMPONENTS` standing in for the model's
    ``packed_modules_mapping``: the module's last dot-component or full name
    is in the class; or it is a packed parent one of whose children is in the
    class (``q_proj`` in the class creates the ``qkv_proj`` wrapper); or it is
    a child whose packed parent is in the class. The MoE wrapper's suffix is
    ``experts``, which the composer adds to the class whenever an adapter
    carries expert factors (contract §3)."""
    target_set = {str(t) for t in targets}
    suffix = module_name.rsplit(".", 1)[-1]
    if suffix in target_set or module_name in target_set:
        return True
    children = PACKED_COMPONENTS.get(suffix)
    if children and any(c in target_set for c in children):
        return True
    return any(suffix in children and parent in target_set
               for parent, children in PACKED_COMPONENTS.items())


def lora_kernel_index_ceiling(max_module_dim: int) -> int:
    """The batched-token ceiling of the pinned LoRA Triton kernels for
    adapters whose widest adapted module has ``max_module_dim`` in/out
    features: ``(2^31 - 1) // D`` rows (module constants)."""
    if isinstance(max_module_dim, bool) or not isinstance(max_module_dim, int) \
            or max_module_dim < 1:
        raise ServingAttestationError(
            f"max_module_dim must be a positive int, got {max_module_dim!r}")
    return LORA_KERNEL_INDEX_LIMIT // int(max_module_dim)


def _factor_other_axis(shape: Sequence[int], r: int, *, which: str, module: str,
                       adapter_dir: Path) -> int:
    """The non-rank axis of one LoRA factor (``worker_reports._oriented``'s
    rule on the header shape alone): exactly one axis equal to ``r``
    decides; both or neither refuses rather than guessing."""
    if len(shape) != 2:
        raise ServingAttestationError(
            f"{adapter_dir}: module {module!r} lora_{which} has {len(shape)} dims, expected 2")
    d0, d1 = int(shape[0]), int(shape[1])
    if d0 == r and d1 == r:
        raise ServingAttestationError(
            f"{adapter_dir}: module {module!r} lora_{which} has shape {tuple(shape)} with BOTH "
            f"axes equal to r={r}: the orientation is ambiguous; refusing to guess")
    if d0 == r:
        return d1
    if d1 == r:
        return d0
    raise ServingAttestationError(
        f"{adapter_dir}: module {module!r} lora_{which} has shape {tuple(shape)}; neither axis "
        f"equals r={r} — not a rank-{r} factor")


def _is_experts_module_name(module: str) -> bool:
    """True for a FILE module name of a routed-experts pair (``...mlp.experts``
    = down, ``...mlp.experts.base_layer`` = gate_up; contract §3)."""
    key = module_key_of(module)
    return key is not None and module_kind_of_key(key) in (MODULE_KIND_EXPERTS,
                                                            MODULE_KIND_EXPERTS_PROJECTION)


def adapter_module_dims(adapter_dir: Path) -> Dict[str, Tuple[int, int]]:
    """Every adapted module of the PEFT adapter at ``adapter_dir`` ->
    ``(in_features, out_features)``, read off the safetensors HEADER (no
    tensor is loaded), keyed by the file's module name. A LINEAR factor pair
    is oriented by the config's ``r`` exactly as the fingerprint orients it
    (``lora_A`` is ``[r, d_in]``, ``lora_B`` ``[d_out, r]``, in either stored
    orientation; a shape that matches ``r`` on neither or both axes refuses).
    A ROUTED-EXPERTS pair (``...mlp.experts.base_layer`` = gate_up,
    ``...mlp.experts`` = down) is the flat PEFT ``ParamWrapper`` layout
    ``lora_A [E*r_e, d_in]`` / ``lora_B [d_out, E*r_e]`` whose rank axis is
    ``E*r_e``, NOT ``r`` — sized by its own shape (``d_in = A.shape[1]``,
    ``d_out = B.shape[0]``, the two rank axes must agree), so gate_up reports
    ``(d, 2I)`` and down ``(I, d)``. These are per-MODULE widths; the int32
    kernel index ceiling is sized from the EXECUTING wrapper's stride, which
    for a packed wrapper is the SUM of its components' outputs —
    :func:`execution_widths` / :func:`max_module_dim` do that grouping.
    Refusals are :class:`ServingAttestationError` naming the module."""
    from safetensors import safe_open  # transformers-stack dep, lazy
    from .worker_reports import (FingerprintError, _split_peft_key, is_residual_payload_key,
                                 read_adapter_config)

    adapter_dir = Path(adapter_dir)
    try:
        cfg = read_adapter_config(adapter_dir)
    except FingerprintError as e:
        raise ServingAttestationError(str(e)) from e
    r = int(cfg["r"])
    weights = adapter_dir / ADAPTER_WEIGHTS_FILE
    if not weights.is_file():
        raise ServingAttestationError(
            f"{adapter_dir} has no {ADAPTER_WEIGHTS_FILE} — no factors to size")
    shapes: Dict[str, Dict[str, Tuple[int, ...]]] = {}
    with safe_open(str(weights), framework="pt") as fh:
        for key in fh.keys():
            try:
                if is_residual_payload_key(key):
                    continue      # the payload is not a factor; the carrier pair sizes the kernel
                module, which = _split_peft_key(key)
            except (FingerprintError, ServingAttestationError) as e:
                raise ServingAttestationError(str(e)) from e
            slot = shapes.setdefault(module, {})
            if which in slot:
                raise ServingAttestationError(
                    f"{adapter_dir}: module {module!r} carries two lora_{which} tensors")
            slot[which] = tuple(int(d) for d in fh.get_slice(key).get_shape())
    dims: Dict[str, Tuple[int, int]] = {}
    for module, ab in sorted(shapes.items()):
        if set(ab) != {"A", "B"}:
            raise ServingAttestationError(
                f"{adapter_dir}: module {module!r} has lora_{sorted(ab)} only; both factors are "
                "needed to size it")
        if _is_experts_module_name(module):
            a, b = ab["A"], ab["B"]
            if len(a) != 2 or len(b) != 2:
                raise ServingAttestationError(
                    f"{adapter_dir}: routed-experts module {module!r} has lora_A {a} / lora_B {b}; "
                    "the flat PEFT ParamWrapper layout is 2-D ([E*r_e, d_in] / [d_out, E*r_e])")
            if a[0] != b[1] or a[0] < 1:
                raise ServingAttestationError(
                    f"{adapter_dir}: routed-experts module {module!r} lora_A {a} and lora_B {b} do "
                    "not share the E*r_e rank axis (A.shape[0] must equal B.shape[1]); not a flat "
                    "[E*r_e, d_in] / [d_out, E*r_e] pair")
            dims[module] = (int(a[1]), int(b[0]))
            continue
        d_in = _factor_other_axis(ab["A"], r, which="A", module=module, adapter_dir=adapter_dir)
        d_out = _factor_other_axis(ab["B"], r, which="B", module=module, adapter_dir=adapter_dir)
        dims[module] = (d_in, d_out)
    if not dims:
        raise ServingAttestationError(
            f"{adapter_dir}: {ADAPTER_WEIGHTS_FILE} carries no LoRA factors")
    return dims


def execution_widths(module_dims: Mapping[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
    """The EXECUTING wrappers' ``(in, out)`` widths implied by one adapter's
    per-module dims (:func:`adapter_module_dims`): components vLLM packs
    into one wrapper are merged — the ``in`` widths must agree (one input),
    the ``out`` widths SUM (the packed wrapper's output stride is
    ``sum(output_slices)``; ``qwen3_next.py:258-276`` + ``worker_reports``'
    stride record) — under the wrapper's name in the same spelling as the
    components (their prefix + the packed leaf, e.g. ``...layers.3.self_attn.
    qkv_proj``); every other module is its own wrapper under its own name.
    The routed-experts pairs are left as the two GEMMs the MoE wrapper runs
    (``(d, 2I)`` and ``(I, d)``). A module name the vocabulary does not
    recognise stands as its own wrapper at its own dims: this function only
    needs to know which modules PACK, an unrecognised name packs with nothing,
    and whether the name is an admitted module is the fingerprint's refusal
    (``worker_reports.canonical_module_key``), not the width arithmetic's —
    the geometry tests fake ``adapter_module_dims`` with bare ``{"o_proj":
    (D, D)}`` and must keep sizing. Note the width of a packed wrapper is
    exact only when the file carries EVERY component (the admitted sets adapt
    q, k, v together); a q-only adapter would size 8192 where the wrapper
    executes 9216 — and readiness then refuses it against the worker's own
    stride (:func:`execution_stride_problems`), loudly rather than silently."""
    out: Dict[str, Tuple[int, int]] = {}
    groups: Dict[str, List[Tuple[str, int, int]]] = {}
    for module, (d_in, d_out) in sorted(module_dims.items()):
        key = module_key_of(module)
        if key is None:
            out[module] = (int(d_in), int(d_out))      # packs with nothing: its own wrapper
            continue
        carrier = execution_carrier_key(key)
        if carrier == key or module_kind_of_key(key) == MODULE_KIND_EXPERTS_PROJECTION:
            out[module] = (int(d_in), int(d_out))
            continue
        _, _, packed_leaf = parse_module_key(carrier)
        name = module.rsplit(".", 1)[0] + "." + packed_leaf
        groups.setdefault(name, []).append((module, int(d_in), int(d_out)))
    for name, members in groups.items():
        ins = {d_in for _, d_in, _ in members}
        if len(ins) != 1:
            raise ServingAttestationError(
                f"packed wrapper {name!r}: its components {[m for m, _, _ in members]} disagree on "
                f"the input width ({sorted(ins)}); one wrapper has one input — refusing")
        out[name] = (ins.pop(), sum(d_out for _, _, d_out in members))
    return dict(sorted(out.items()))


def adapter_execution_widths(adapter_dir: Path) -> Dict[str, Tuple[int, int]]:
    """:func:`execution_widths` over :func:`adapter_module_dims` of one
    adapter directory — the composer's input for the kernel index ceiling
    when the adapted class packs (q/k/v, gate/up)."""
    return execution_widths(adapter_module_dims(adapter_dir))


def max_module_dim(dims_by_adapter: Mapping[str, Mapping[str, Tuple[int, int]]]
                   ) -> Tuple[int, str]:
    """``(D, "<adapter>:<wrapper>")`` — the widest EXECUTING stride over every
    registered adapter (the D the kernels index rows against): each
    adapter's per-module dims are grouped into the wrappers vLLM executes
    them under (:func:`execution_widths`; a plain o_proj adapter is unchanged
    — one module, one wrapper, the label is the file's module name), then the
    widest in/out over all of them wins."""
    best: Optional[Tuple[int, str]] = None
    for name in sorted(dims_by_adapter):
        for module, (d_in, d_out) in sorted(execution_widths(dims_by_adapter[name]).items()):
            d = max(int(d_in), int(d_out))
            if best is None or d > best[0]:
                best = (d, f"{name}:{module}")
    if best is None:
        raise ServingAttestationError("max_module_dim over no adapted modules")
    return best


def expected_native_lora_state_for(names: Sequence[str],
                                   adapter_dirs: Mapping[str, str]
                                   ) -> Dict[str, Any]:
    """``{"n_adapters": N, "adapters": {name: {"dir", "fingerprint_bf16",
    "module_keys", "r", "lora_int_id"}}}`` — what EVERY worker must
    report, derived by the launcher from the adapter directories alone.
    ``names`` is the composer's sorted ``--lora-modules`` order, so
    ``lora_int_id = position + 1`` (the frontend's counter order, module
    docstring). Fingerprint refusals (``worker_reports.FingerprintError``)
    propagate: a directory that cannot be fingerprinted cannot be
    attested."""
    from sisteer.capeval.serving import worker_reports as wr

    names = tuple(names)
    if not names:
        raise ServingAttestationError(
            "a lora-native launch with no adapter names has nothing to "
            "attest")
    if len(set(names)) != len(names):
        raise ServingAttestationError(
            f"duplicate adapter names {sorted(names)!r}")
    if set(names) != set(adapter_dirs):
        raise ServingAttestationError(
            f"adapter names {sorted(names)!r} and directories "
            f"{sorted(adapter_dirs)!r} disagree")
    from sisteer.capeval.serving import native_residual as nr

    adapters: Dict[str, Dict[str, Any]] = {}
    for i, name in enumerate(names):
        d = Path(adapter_dirs[name])
        fingerprint, keys = wr.expected_adapter_fingerprint_bf16(d)
        cfg = wr.read_adapter_config(d)
        try:
            payload = nr.read_residual_payload(d)
        except nr.ResidualSlotError as e:
            raise ServingAttestationError(f"{d}: {e}") from e
        adapters[name] = {
            "dir": str(d),
            "fingerprint_bf16": fingerprint,
            "module_keys": list(keys),
            "r": int(cfg["r"]),
            "lora_int_id": i + 1,
            # the residual identity beside the carrier's: None for a plain LoRA
            "residual_fingerprint": (payload.fingerprint if payload is not None else None),
        }
    return {"n_adapters": len(names), "adapters": adapters}


def expected_residual_slot_state(names: Sequence[str], adapter_dirs: Mapping[str, str]
                                 ) -> Dict[str, Any]:
    """What EVERY worker's ``sisteer_residual_slot_report`` must show, derived by
    the launcher from the adapter directories alone: ``armed`` iff any
    registered adapter carries a residual payload
    (the composer sets the launch env flag on the same fact); the payloads'
    shared ``n_layers_model`` / ``hidden_size`` (they must agree — one base);
    the ONE op name; per adapter ``{lora_int_id, residual_fingerprint, kind,
    op, position_mode, layers, boundaries}`` (the residual fields ``None`` for
    a plain LoRA). ``names`` is the composer's sorted ``--lora-modules`` order."""
    from sisteer.capeval.serving import native_residual as nr

    names = tuple(names)
    if len(set(names)) != len(names):
        raise ServingAttestationError(f"duplicate adapter names {sorted(names)!r}")
    if set(names) != set(adapter_dirs):
        raise ServingAttestationError(
            f"adapter names {sorted(names)!r} and directories {sorted(adapter_dirs)!r} disagree")
    adapters: Dict[str, Dict[str, Any]] = {}
    n_layers: set = set()
    hidden: set = set()
    for i, name in enumerate(names):
        d = Path(adapter_dirs[name])
        try:
            payload = nr.read_residual_payload(d)
        except nr.ResidualSlotError as e:
            raise ServingAttestationError(f"{d}: {e}") from e
        if payload is None:
            adapters[name] = {"lora_int_id": i + 1, "residual_fingerprint": None, "kind": None,
                              "op": None, "position_mode": None, "layers": None,
                              "boundaries": None}
            continue
        adapters[name] = {"lora_int_id": i + 1, "residual_fingerprint": payload.fingerprint,
                          "kind": payload.kind, "op": payload.op,
                          "position_mode": payload.position_mode,
                          "layers": list(payload.layers), "boundaries": list(payload.boundaries)}
        n_layers.add(int(payload.n_layers_model))
        hidden.add(int(payload.hidden_size))
    if len(n_layers) > 1 or len(hidden) > 1:
        raise ServingAttestationError(
            f"the registered residual payloads disagree on the model geometry (n_layers_model "
            f"{sorted(n_layers)}, hidden_size {sorted(hidden)}) — they were not exported for one base")
    # THE PAIRED WRITE: iff a registered payload is the paired op the composer sets the paired
    # launch flag and the bank must carry its second-gate buffers (readiness refuses otherwise)
    paired = any(info.get("op") == nr.OP_RELU_GATED_VECTOR_PAIR for info in adapters.values())
    return {"armed": bool(n_layers), "paired": paired, "op_name": RESIDUAL_SLOT_OP_NAME,
            "n_layers_model": (next(iter(n_layers)) if n_layers else None),
            "hidden_size": (next(iter(hidden)) if hidden else None),
            "adapters": adapters}


def residual_slot_problems(results: Any, expected: Mapping[str, Any]) -> List[str]:
    """The residual slot's readiness assertions over every worker's
    ``sisteer_residual_slot_report`` against :func:`expected_residual_slot_state`:
    ``armed`` equals the expectation on every
    worker; the ledger is empty and not faulted; UNARMED: no slot loaded;
    ARMED: the op is registered under the ONE name, the runner is bound, every
    patched decoder class reports a wrapped forward, the hooked block count
    equals the model's, and per adapter the loaded slot at its ``lora_int_id``
    carries the expected payload fingerprint / kind / mode / layers /
    boundaries — a plain LoRA has no slot, and no slot is loaded for a name the
    expectation does not carry. Returns the problems; ``[]`` = every assertion
    held for every worker."""
    if not isinstance(results, list) or not results:
        return [f"residual-slot RPC returned {results!r} — expected one report PER WORKER; a "
                "server whose residual slot cannot be read must not pass readiness"]
    armed = bool(expected["armed"])
    by_name: Mapping[str, Mapping[str, Any]] = expected["adapters"]
    with_payload = {str(info["lora_int_id"]): name for name, info in by_name.items()
                    if info.get("residual_fingerprint") is not None}
    problems: List[str] = []
    for i, rep in enumerate(results):
        if not isinstance(rep, Mapping):
            problems.append(f"worker {i}: non-report result {rep!r}")
            continue
        if rep.get("armed") is not armed:
            problems.append(
                f"worker {i}: residual_slot.armed = {rep.get('armed')!r}, expected {armed} (the launch "
                f"env {'carries' if armed else 'does not carry'} the residual-slot flag because "
                f"{'a' if armed else 'no'} registered adapter carries a payload)")
            continue
        if rep.get("ledger_fault") is not False:
            problems.append(f"worker {i}: residual_slot.ledger_fault = {rep.get('ledger_fault')!r} at readiness")
        if rep.get("ledger_size") != 0:
            problems.append(f"worker {i}: residual_slot.ledger_size = {rep.get('ledger_size')!r} at readiness "
                            "(no request has run; an unclaimed failure here is a wiring defect)")
        slots = rep.get("slots")
        if not isinstance(slots, Mapping):
            problems.append(f"worker {i}: residual_slot.slots is {slots!r}, not a mapping")
            continue
        if not armed:
            if slots:
                problems.append(f"worker {i}: {len(slots)} residual slot(s) loaded on an UNARMED worker")
            continue
        if rep.get("op_name") != RESIDUAL_SLOT_OP_NAME:
            problems.append(f"worker {i}: residual_slot.op_name = {rep.get('op_name')!r}, expected "
                            f"{RESIDUAL_SLOT_OP_NAME!r}")
        if rep.get("op_registered") is not True:
            problems.append(f"worker {i}: the residual op is not registered (op_registered="
                            f"{rep.get('op_registered')!r})")
        if rep.get("runner_bound") is not True:
            problems.append(f"worker {i}: the residual bank is not bound to the model runner "
                            f"(runner_bound={rep.get('runner_bound')!r}; fact 11)")
        # THE PAIRED WRITE: the bank's second-gate buffers exist iff a registered payload is the paired
        # op (the composer's flag and the adapters must agree; a paired payload on an unarmed bank also
        # refuses at activation, but a disagreement is a wiring defect and readiness names it)
        bank_rep = rep.get("bank")
        paired_armed = bank_rep.get("paired_armed") if isinstance(bank_rep, Mapping) else None
        want_paired = bool(expected.get("paired", False))
        if want_paired and paired_armed is not True:
            problems.append(
                f"worker {i}: a registered payload is the paired op but the bank carries no second-gate "
                f"buffers (bank.paired_armed={paired_armed!r}) — the launch env lacks the paired flag the "
                "composer sets on that fact; the paired cell could only be served as one gate; refusing")
        elif paired_armed is True and not want_paired:
            problems.append(
                f"worker {i}: the bank allocated second-gate buffers (bank.paired_armed=True) but no registered "
                "payload is the paired op — the launch env carries the paired flag on a fact the adapters do "
                "not show; the composer and the bank disagree")
        n_model = rep.get("n_layers_model")
        if n_model != expected.get("n_layers_model"):
            problems.append(f"worker {i}: residual_slot.n_layers_model = {n_model!r}, the payloads were "
                            f"exported for {expected.get('n_layers_model')!r} decoder blocks")
        if rep.get("n_layers_hooked") != n_model:
            problems.append(f"worker {i}: {rep.get('n_layers_hooked')!r} decoder block(s) hooked of "
                            f"{n_model!r} — a decoder-layer class is not wrapped (or constructed without "
                            "a boundary)")
        # stage 2 (native-psr-serving step 7): the per-request FACT table must not have overflowed and no
        # serving fault may be recorded at readiness — a worker that already refused a request at the cap
        # (`overflowed`, latching `overflow`) or judged a request unservable before any traffic is a wiring
        # defect, never a clean start; facts RECLAIMED for finished requests are the healthy lifecycle
        ft = rep.get("fact_table")
        if isinstance(ft, Mapping):
            if ft.get("overflow"):
                problems.append(f"worker {i}: residual_slot.fact_table overflowed at readiness ({ft})")
            # the ENGINE-ID CONTRACT pin: the proxy claims a failure by the
            # completion id; the worker ledgers it under the scheduler's id, which at vLLM 0.25.1 is
            # that id plus ONE `-<8 hex>` suffix (ENGINE_REQUEST_ID_SUFFIX_RE). An admitted id WITHOUT
            # the suffix means the minting changed (a vLLM bump, VLLM_DISABLE_REQUEST_ID_RANDOMIZATION)
            # and no failure under it could ever be claimed — refused before the first steered request
            if int(ft.get("off_contract") or 0):
                problems.append(f"worker {i}: residual_slot.fact_table admitted {int(ft['off_contract'])} request "
                                "id(s) OFF the pinned engine-id contract (no `-<8 hex>` engine suffix: vLLM 0.25.1 "
                                "InputProcessor.assign_request_id — a vLLM bump or VLLM_DISABLE_REQUEST_ID_RANDOMIZATION?) "
                                f"— the proxy claims failures by the completion id, so such a failure could never be "
                                f"claimed ({ft})")
            # the table's two counters are SEPARATE by name: `reclaimed` counts facts released for
            # FINISHED requests (RequestFactTable.drop over finished_req_ids — the healthy path every
            # readiness probe drives to S; a refusal on it killed four jobs), `overflowed` counts requests
            # REFUSED at the cap (finished ones were NOT reclaimed) and latches `overflow` — refused above;
            # a fact lost before judgement is a ledgered failure — refused below.
        if int(rep.get("n_faults") or 0):
            problems.append(f"worker {i}: residual_slot records {rep.get('n_faults')} serving fault(s) at readiness: "
                            f"{str(rep.get('faults'))[:300]}")
        classes = rep.get("patched_classes")
        if not isinstance(classes, Mapping) or not classes:
            problems.append(f"worker {i}: residual_slot.patched_classes is empty or malformed ({classes!r})")
        else:
            for cls, entry in classes.items():
                if not isinstance(entry, Mapping) or entry.get("forward_wrapped") is not True:
                    problems.append(f"worker {i}: decoder class {cls!r} reports no wrapped forward ({entry!r})")
        for name in sorted(by_name):
            info = by_name[name]
            key = str(info["lora_int_id"])
            got = slots.get(key)
            if info.get("residual_fingerprint") is None:
                if got is not None:
                    problems.append(f"worker {i}: adapter {name!r} (lora_int_id {key}) carries no residual "
                                    f"payload but the worker loaded slot {got!r} for it")
                continue
            if not isinstance(got, Mapping):
                problems.append(f"worker {i}: adapter {name!r} (lora_int_id {key}) has no loaded residual slot")
                continue
            for field, want in (("fingerprint", info["residual_fingerprint"]), ("kind", info["kind"]),
                                ("op", info["op"]), ("mode", info["position_mode"]),
                                ("layers", list(info["layers"])), ("boundaries", list(info["boundaries"]))):
                if got.get(field) != want:
                    problems.append(f"worker {i}: adapter {name!r} (lora_int_id {key}) residual slot "
                                    f"{field} = {got.get(field)!r}, expected {want!r}")
        stray = sorted(set(slots) - set(with_payload))
        if stray:
            problems.append(f"worker {i}: residual slot(s) loaded for lora_int_id(s) {stray} that no "
                            "registered adapter's payload accounts for")
    return problems


def _validate_residual_slot_key(rs: Any, *, where: str) -> None:
    """The :data:`RESIDUAL_SLOT_KEY` block's shape (schema 2)."""
    if not isinstance(rs, Mapping) or any(k not in rs for k in RESIDUAL_SLOT_KEYS):
        raise ServingAttestationError(
            f"{where}: {RESIDUAL_SLOT_KEY} block lacks keys "
            f"{[k for k in RESIDUAL_SLOT_KEYS if not isinstance(rs, Mapping) or k not in rs]}")
    if not isinstance(rs["armed"], bool):
        raise ServingAttestationError(f"{where}: {RESIDUAL_SLOT_KEY}.armed must be a bool, got {rs['armed']!r}")
    if rs["rpc_method"] != RESIDUAL_SLOT_RPC_METHOD:
        raise ServingAttestationError(
            f"{where}: {RESIDUAL_SLOT_KEY}.rpc_method={rs['rpc_method']!r}, expected {RESIDUAL_SLOT_RPC_METHOD!r}")
    n = rs["n_workers"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ServingAttestationError(f"{where}: {RESIDUAL_SLOT_KEY}.n_workers={n!r} is not a positive int")
    exp = rs["expected"]
    if not isinstance(exp, Mapping) or not isinstance(exp.get("armed"), bool) \
            or not isinstance(exp.get("adapters"), Mapping) or exp["armed"] != rs["armed"]:
        raise ServingAttestationError(
            f"{where}: {RESIDUAL_SLOT_KEY}.expected malformed or disagreeing with armed ({exp!r})")
    reps = rs["worker_reports"]
    if not isinstance(reps, list) or not reps or not all(
            isinstance(r, Mapping) and r.get("armed") is rs["armed"] for r in reps):
        raise ServingAttestationError(
            f"{where}: {RESIDUAL_SLOT_KEY}.worker_reports must be a non-empty list of reports whose "
            f"armed equals {rs['armed']!r}")


def native_lora_report_problems(results: Any, expected: Mapping[str, Any]
                                ) -> List[str]:
    """The per-NAME binding assertions over every worker's report
    (module docstring, step 3): the loaded id set is exactly ``{1..N}``
    with ``lora_loaded_count == N``; then for EVERY expected name, the
    report at that name's ``lora_int_id`` carries the expected
    fingerprint, rank and module keys and a bf16 tensor dtype. A
    fingerprint that belongs to ANOTHER expected name is named as such
    (the permutation case names both names and both ids). Returns the
    problems; ``[]`` means every assertion held for every worker."""
    if not isinstance(results, list) or not results:
        return [f"lora-native RPC returned {results!r} — expected one "
                "resolved report PER WORKER; a server whose adapters cannot "
                "be read must not pass readiness"]
    n = int(expected["n_adapters"])
    by_name: Mapping[str, Mapping[str, Any]] = expected["adapters"]
    # adapter identity is the PAIR (carrier fingerprint, residual fingerprint): every
    # residual entry on one base shares the same rank-1 zero carrier
    fp_owner = {(info["fingerprint_bf16"], info.get("residual_fingerprint")): name
                for name, info in by_name.items()}
    problems: List[str] = []
    for i, rep in enumerate(results):
        if not isinstance(rep, Mapping):
            problems.append(f"worker {i}: non-report result {rep!r}")
            continue
        adapters = rep.get("lora_adapters")
        if not isinstance(adapters, list):
            problems.append(
                f"worker {i}: lora_adapters MISSING from the report "
                f"(got {adapters!r}) — the loaded adapters cannot be read")
            continue
        count = rep.get("lora_loaded_count")
        if count != n:
            problems.append(
                f"worker {i}: lora_loaded_count = {count!r}, expected {n} "
                "(every registered adapter is loaded at start)")
        ids = [a.get("id") for a in adapters if isinstance(a, Mapping)]
        if len(ids) != len(adapters) or sorted(ids) != list(range(1, n + 1)):
            problems.append(
                f"worker {i}: loaded adapter ids {sorted(ids, key=str)} != "
                f"the expected {list(range(1, n + 1))} (one id per composed "
                "--lora-modules entry, in order)")
        by_id = {a.get("id"): a for a in adapters if isinstance(a, Mapping)}
        for name in sorted(by_name):
            info = by_name[name]
            want_id = info["lora_int_id"]
            got = by_id.get(want_id)
            if got is None:
                problems.append(
                    f"worker {i}: adapter {name!r} (lora_int_id {want_id}) "
                    "has no report at that id")
                continue
            fp = got.get("fingerprint_bf16")
            rfp = got.get("residual_fingerprint")
            want_rfp = info.get("residual_fingerprint")
            if fp != info["fingerprint_bf16"] or rfp != want_rfp:
                other = fp_owner.get((fp, rfp))
                tail = ""
                if other is not None and other != name:
                    tail = (f" — that (fingerprint_bf16, residual_fingerprint) pair is the one "
                            f"expected for adapter {other!r} (lora_int_id "
                            f"{by_name[other]['lora_int_id']}): the worker's "
                            "ids do not bind to the composed names")
                problems.append(
                    f"worker {i}: adapter {name!r} (lora_int_id {want_id}) "
                    f"reports fingerprint_bf16 {fp!r} / residual_fingerprint {rfp!r}, expected "
                    f"{info['fingerprint_bf16']!r} / {want_rfp!r}{tail}")
            try:
                rank = int(got.get("rank"))
            except (TypeError, ValueError):
                rank = None
            if rank != int(info["r"]):
                problems.append(
                    f"worker {i}: adapter {name!r} (lora_int_id {want_id}) "
                    f"reports rank {got.get('rank')!r}, expected {info['r']}")
            keys = got.get("module_keys")
            if not isinstance(keys, list) or list(keys) != list(info["module_keys"]):
                problems.append(
                    f"worker {i}: adapter {name!r} (lora_int_id {want_id}) "
                    f"reports module_keys {keys!r}, expected "
                    f"{list(info['module_keys'])!r}")
            dtype = got.get("tensor_dtype")
            if dtype != EXPECTED_TENSOR_DTYPE:
                problems.append(
                    f"worker {i}: adapter {name!r} (lora_int_id {want_id}) "
                    f"reports tensor_dtype {dtype!r}, expected "
                    f"{EXPECTED_TENSOR_DTYPE!r} (the fingerprint is defined "
                    "on bf16 bits)")
        # THE EXECUTION SET (code-review finding, 2026-09-11): the
        # fingerprint above hashes the CPU-side LoRAModel.loras factors,
        # which vLLM loads for every supported module regardless of
        # LoRAConfig.target_modules; the weights are APPLIED only where the
        # model carries a LoRA execution wrapper (model_manager.modules,
        # filled from _match_target_modules @0.25.1). Every fingerprinted
        # module key must therefore name a registered execution module, or
        # the attested intervention is applied on a subset of its layers.
        exec_keys = rep.get("lora_execution_module_keys")
        if not isinstance(exec_keys, list) or not all(
                isinstance(k, str) for k in exec_keys):
            problems.append(
                f"worker {i}: lora_execution_module_keys MISSING from the "
                f"report (got {exec_keys!r}) — the modules carrying LoRA "
                "execution cannot be read, so the adapters cannot be held to "
                "them")
            continue
        # a logical key is COVERED when some execution key carries it: itself
        # (o_proj, down_proj, an unpacked engine) or the packed / MoE wrapper
        # vLLM executes it under (q/k/v -> qkv_proj, gate/up -> gate_up_proj,
        # experts.{gate_up,down}_proj -> experts; carried_keys / the module
        # vocabulary above)
        exec_set = set(exec_keys)
        for name in sorted(by_name):
            info = by_name[name]
            carrier_of = _carriers_of(info["module_keys"])
            missing = sorted(k for k in info["module_keys"]
                             if k not in exec_set and carrier_of[k] not in exec_set)
            if missing:
                shown = missing[:8] + (["..."] if len(missing) > 8 else [])
                problems.append(
                    f"worker {i}: adapter {name!r} (lora_int_id "
                    f"{info['lora_int_id']}): {len(missing)} of "
                    f"{len(info['module_keys'])} fingerprinted module(s) "
                    f"{shown} have NO LoRA execution layer on the model (the "
                    f"worker wraps {len(exec_set)} recognised module(s)) — a "
                    "--lora-target-modules restriction or a wrapper miss "
                    "would apply only a SUBSET of the attested intervention "
                    "under this name; refusing")
        # THE EXACT-NAME BINDING. The canonical keys above unify the HF/PEFT
        # and post-mapper spellings so the launcher (files) and the worker
        # (loaded factors) can compare fingerprints — but vLLM APPLIES by
        # exact post-mapper name: activate_adapter iterates the execution
        # registry and looks each module's weights up with
        # LoRAModel.get_lora(module_name) == loras.get(module_name)
        # (model_manager.py @0.25.1), reset_lora() on a miss. A factor
        # loaded under `language_model.model.model.layers.0.self_attn.
        # o_proj` against a wrapper at `language_model.model.layers.0.
        # self_attn.o_proj` canonicalises to the same L0 key, passes the
        # fingerprint and the coverage check above, and is never applied. So
        # the worker reports, per adapter, which factor its OWN lookup
        # selects for every recognised execution module, and every
        # fingerprinted key WITH an execution layer must be backed by a real
        # binding on it; a selection that is not this adapter's factor, a
        # factor selected by two modules, or a bound module outside the
        # fingerprinted set refuses too (both directions). A packed wrapper
        # (qkv_proj, gate_up_proj) or the MoE wrapper (experts) binds ONE
        # packed factor for SEVERAL logical keys — the relation is
        # carried_keys, not equality (the coverage is exact by NAME, never
        # weakened). A key with NO execution layer at all is the coverage
        # problem already named above, not repeated here.
        # Lazy import: worker_reports pulls torch, which this stdlib-only leaf
        # must not import at module load.
        from sisteer.capeval.serving.worker_reports import (
            FingerprintError, canonical_module_key)
        exec_names = rep.get("lora_execution_module_names")
        if not isinstance(exec_names, list) or not all(
                isinstance(m, str) for m in exec_names):
            problems.append(
                f"worker {i}: lora_execution_module_names MISSING from the "
                f"report (got {exec_names!r}) — the EXACT names of the modules "
                "carrying LoRA execution cannot be read, so no factor can be "
                "held to the module vLLM applies it on")
            continue
        try:
            exec_key_of = {m: canonical_module_key(m) for m in exec_names}
        except FingerprintError as e:
            problems.append(
                f"worker {i}: lora_execution_module_names carries a name that "
                f"is not a recognised adapted module ({e}) — refusing to bind to it")
            continue
        if sorted(exec_key_of.values()) != sorted(exec_keys):
            problems.append(
                f"worker {i}: lora_execution_module_names canonicalise to "
                f"{sorted(exec_key_of.values())} but lora_execution_module_keys "
                f"is {sorted(exec_keys)} — the report's exact and canonical "
                "spellings of the execution set disagree; refusing")
        by_exec_key = _by_value(exec_key_of)
        for name in sorted(by_name):
            info = by_name[name]
            want_id = info["lora_int_id"]
            got = by_id.get(want_id)
            if got is None:
                continue                       # already named above
            head = f"worker {i}: adapter {name!r} (lora_int_id {want_id})"
            factors = got.get("raw_module_names")
            if not isinstance(factors, list) or not all(
                    isinstance(f, str) for f in factors):
                problems.append(
                    f"{head}: raw_module_names MISSING from the report (got "
                    f"{factors!r}) — the factors' exact names cannot be read, "
                    "so they cannot be held to the execution modules")
                continue
            bindings = got.get("execution_bindings")
            if not isinstance(bindings, Mapping) or not all(
                    isinstance(m, str) and (f is None or isinstance(f, str))
                    for m, f in bindings.items()):
                problems.append(
                    f"{head}: execution_bindings MISSING from the report (got "
                    f"{bindings!r}) — which factor vLLM's exact-name lookup "
                    "selects for each execution module cannot be read; the "
                    "adapter cannot be attested as applied")
                continue
            if set(bindings) != set(exec_names):
                problems.append(
                    f"{head}: execution_bindings name {sorted(bindings)} but the "
                    f"execution modules are {sorted(exec_names)} — the bindings "
                    "were not resolved over the execution registry; refusing")
                continue
            try:
                factor_key = {f: canonical_module_key(f) for f in factors}
            except FingerprintError as e:
                problems.append(
                    f"{head}: raw_module_names carries a name that is not a "
                    f"recognised adapted module ({e}); refusing")
                continue
            # a fingerprinted key WITH an execution layer but NO exact binding
            # on it — the finding. The execution modules that carry a logical
            # key: those whose key equals it or is its packed / MoE carrier
            # (carried_keys' relation, as two index lookups per key).
            carrier_of = _carriers_of(info["module_keys"])
            by_factor_key = _by_value(factor_key)
            no_exact: List[str] = []
            for key in info["module_keys"]:
                mods = sorted({*by_exec_key.get(key, ()), *by_exec_key.get(carrier_of[key], ())})
                if not mods:
                    continue                   # the coverage problem owns it
                if not any(bindings.get(m) is not None for m in mods):
                    fs = sorted({*by_factor_key.get(key, ()), *by_factor_key.get(carrier_of[key], ())})
                    no_exact.append(f"{key}: factor {fs} vs execution {mods}")
            if no_exact:
                shown = no_exact[:8] + (["..."] if len(no_exact) > 8 else [])
                problems.append(
                    f"{head}: {len(no_exact)} of {len(info['module_keys'])} "
                    "fingerprinted module(s) have a LoRA execution layer at the "
                    "same canonical key but NO exact-name binding on it — "
                    "vLLM's activate_adapter looks each execution module's "
                    "weights up by EXACT post-mapper name (LoRAModel.get_lora, "
                    "model_manager.py @0.25.1) and reset_lora()s on a miss, so "
                    "those layers of the attested intervention would never be "
                    f"applied: {shown}; refusing")
            # every selection is one of THIS adapter's factors, selected once
            selected = [f for f in bindings.values() if f is not None]
            stray = sorted(set(selected) - set(factors))
            if stray:
                problems.append(
                    f"{head}: execution_bindings select {stray}, which are not "
                    f"among this adapter's {len(factors)} loaded factors — not "
                    "the pinned lookup; refusing")
            dup = sorted({f for f in selected if selected.count(f) > 1})
            if dup:
                problems.append(
                    f"{head}: factor(s) {dup} are selected by more than one "
                    f"execution module ({sorted(m for m, f in bindings.items() if f in dup)}) "
                    "— one factor, one module; refusing")
            # both directions: the modules that bind this adapter's factors
            # carry keys INSIDE its fingerprinted set (a module that binds
            # nothing is a layer this adapter does not adapt — the unrestricted
            # engine wraps every projection, a genuine export covers its fit
            # layers only); a packed / MoE wrapper is inside when it carries
            # ANY fingerprinted component
            covered = set(info["module_keys"]) | {c for c in carrier_of.values() if c is not None}
            extra = sorted({exec_key_of[m] for m, f in bindings.items()
                            if f is not None and exec_key_of[m] not in covered})
            if extra:
                pairs = sorted(f"{m} -> {f}" for m, f in bindings.items()
                               if f is not None and exec_key_of[m] in extra)
                problems.append(
                    f"{head}: execution module(s) at {extra} bind a factor of "
                    f"this adapter ({pairs}) although its fingerprinted "
                    f"module_keys are {list(info['module_keys'])} — the applied "
                    "set exceeds the attested one; refusing")
    return problems


# --- (4) the CUDA-graph capture surface covers the decode concurrency --------------


def execution_stride_problems(results: Any, *, target_modules: Sequence[str],
                              max_module_dim: int) -> List[str]:
    """THE EXECUTING-STRIDE CHECK: over every
    worker's report, (1) every module carrying a LoRA execution wrapper —
    the recognised set ``lora_execution_module_names`` AND every other
    family in ``lora_execution_modules_other`` — is inside the composed
    ``target_modules`` class by vLLM's own rule (``is_in_target_modules``
    @0.25.1: the last dot-component or the full name is in the class, or a
    packed parent one of whose children is — ``q_proj`` in the class creates
    the ``qkv_proj`` wrapper; :func:`is_in_target_class`), so no wrapper
    exists outside the adapted class; (2) ``lora_execution_module_strides`` names every
    one of them with positive ``in`` / ``out`` strides; (3) the widest
    stride equals the reported ``lora_execution_module_stride_max`` and
    is ``<= max_module_dim`` — the D the composer sized the int32 kernel
    index ceiling from. A wider wrapper means the pinned kernels index
    ``rows x`` THAT stride (``lora_expand_op``: ``ram * output_d0_stride``;
    ``lora_shrink_op``: the input stride) and the real ceiling is
    ``(2^31-1) // stride``, not the composed one. Returns the problems;
    ``[]`` means every wrapper on every worker stays inside the class and
    under D."""
    if not isinstance(results, list) or not results:
        return [f"lora-native RPC returned {results!r} — expected one "
                "resolved report PER WORKER; a server whose LoRA wrappers "
                "cannot be read must not pass readiness"]
    targets = {str(t) for t in target_modules}
    if not targets:
        return ["no composed --lora-target-modules class to hold the executing "
                "wrappers to — refusing"]
    if isinstance(max_module_dim, bool) or not isinstance(max_module_dim, int) \
            or max_module_dim < 1:
        return [f"lora_max_module_dim={max_module_dim!r} is not a positive int — "
                "the executing strides cannot be held to a ceiling; refusing"]
    problems: List[str] = []
    for i, rep in enumerate(results):
        if not isinstance(rep, Mapping):
            problems.append(f"worker {i}: non-report result {rep!r}")
            continue
        names = rep.get("lora_execution_module_names")
        other = rep.get("lora_execution_modules_other")
        if not (isinstance(names, list) and all(isinstance(m, str) for m in names)
                and isinstance(other, list) and all(isinstance(m, str) for m in other)):
            problems.append(
                f"worker {i}: lora_execution_module_names / lora_execution_modules_other "
                f"MISSING from the report (got {names!r} / {other!r}) — the modules carrying "
                "LoRA execution cannot be read, so their strides cannot be held to the "
                "kernel index ceiling")
            continue
        executing = sorted(set(names) | set(other))
        outside = sorted(m for m in executing if not is_in_target_class(m, targets))
        if outside:
            shown = outside[:8] + (["..."] if len(outside) > 8 else [])
            problems.append(
                f"worker {i}: {len(outside)} execution module(s) carry a LoRA wrapper whose "
                f"suffix is NOT in the composed --lora-target-modules {sorted(targets)}: "
                f"{shown} — the restriction to the adapted class did not take, so the LoRA "
                "kernels run at those wrappers' strides (reset_lora zeroes the weights; the "
                "shrink/expand still dispatch whenever the batch carries LoRA requests) and the "
                f"int32 index ceiling sized from D={max_module_dim:,} no longer bounds them; "
                "refusing")
        strides = rep.get("lora_execution_module_strides")
        ok_shape = isinstance(strides, Mapping) and all(
            isinstance(m, str) and isinstance(v, Mapping)
            and all(k in v and not isinstance(v[k], bool) and isinstance(v[k], int)
                    and v[k] >= 1 for k in ("in", "out"))
            for m, v in strides.items())
        if not ok_shape:
            problems.append(
                f"worker {i}: lora_execution_module_strides MISSING from the report (got "
                f"{strides!r}) — each execution wrapper's (in, out) stride cannot be read, so "
                "the kernel index ceiling cannot be held to what actually executes")
            continue
        if set(strides) != set(executing):
            problems.append(
                f"worker {i}: lora_execution_module_strides names {sorted(strides)} but the "
                f"execution modules are {executing} — the strides were not read over the "
                "execution registry; refusing")
            continue
        widest_mod, widest = max(((m, max(int(v["in"]), int(v["out"])))
                                  for m, v in strides.items()), key=lambda t: (t[1], t[0]))
        reported = rep.get("lora_execution_module_stride_max")
        if reported != widest:
            problems.append(
                f"worker {i}: lora_execution_module_stride_max {reported!r} != the widest "
                f"reported stride {widest:,} ({widest_mod!r}) — the report disagrees with "
                "itself; refusing")
        if widest > int(max_module_dim):
            v = strides[widest_mod]
            problems.append(
                f"worker {i}: execution module {widest_mod!r} carries a LoRA wrapper with "
                f"stride {widest:,} (in {int(v['in']):,} x out {int(v['out']):,}) > the composed "
                f"{LORA_MAX_MODULE_DIM_KEY} {max_module_dim:,} the kernel index ceiling was "
                "sized from — the pinned vLLM 0.25.1 LoRA Triton kernels index rows x THIS "
                "stride in int32 (lora_expand_op: ram * output_d0_stride; lora_shrink_op: the "
                f"input stride), so the real ceiling is (2^31-1) // {widest:,} = "
                f"{LORA_KERNEL_INDEX_LIMIT // widest:,} batched tokens, not the composed "
                f"{LORA_KERNEL_INDEX_LIMIT // int(max_module_dim):,}; refusing (gemma "
                "certification canary job 488932; review round 3)")
    return problems


def cudagraph_coverage_problems(results: Any, *, max_num_seqs: int,
                                composed_capture_sizes: Optional[Sequence[int]] = None
                                ) -> List[str]:
    """Problems with every worker's RESOLVED CUDA-graph capture surface
    against the composed decode concurrency (code-review finding,
    2026-09-11); ``[]`` when every worker covers it.

    ``enforce_eager False`` and a non-NONE ``cudagraph_mode`` do not by
    themselves mean decode runs under graphs: ``CudagraphDispatcher.
    dispatch()`` (``v1/cudagraph_dispatcher.py`` @0.25.1) returns
    ``CUDAGraphMode.NONE`` for any batch whose ``num_tokens`` exceeds
    ``compilation_config.max_cudagraph_capture_size``, and pads a smaller
    batch up to the nearest entry of ``cudagraph_capture_sizes``. A
    uniform decode step at the composed width schedules ``max_num_seqs``
    tokens, so coverage needs ``max_cudagraph_capture_size >= max_num_seqs``
    AND ``max(cudagraph_capture_sizes) >= max_num_seqs`` (vLLM's own
    ``post_init_cudagraph_sizes`` asserts the two agree once resolved; a
    disagreement here is surface drift and refuses too). vLLM's DEFAULT
    max is ``min(2 * max_num_seqs, 512)`` (``config/vllm.py`` @0.25.1), so
    a composed width above 512 is refused here by name until the composer
    composes a larger capture max deliberately — the alternative is an
    attestation that says graphs-on while every full-width decode step
    runs eager.

    THE FULL SURFACE (2026-09-16; Modal H200, vLLM 0.25.1, Qwen3.5-4B
    native). The two checks above passed on a launch whose decode had
    collapsed 6x: S=51 on vLLM's default list ``[1, 2, 4, 8, ..., 96]``
    (max 96 >= 51, largest 96 >= 51) — but vLLM captures FULL decode
    graphs ONLY for capture sizes ``<= max_num_seqs * uniform_decode_query_
    len`` (``initialize_cudagraph_keys``), so the FULL surface stopped at
    48 and every 49-51 sequence step padded to 56, found no FULL key and
    ran the PIECEWISE graph with attention eager (``FULL=18 (largest=48)``
    in the log; 58 -> 9 tok/s/seq). So each report must also carry the
    dispatcher's FULL keys — ``cudagraph_full_capture_sizes`` (their token
    counts) and ``cudagraph_full_capture_max`` — plus
    ``uniform_decode_query_len``, and coverage needs ``cudagraph_full_
    capture_max >= max_num_seqs * uniform_decode_query_len``: exactly
    "S is a captured size" under a separate decode routine
    (FULL_AND_PIECEWISE / FULL_DECODE_ONLY), and implied by the max check
    under plain FULL (every capture size has a FULL key). A report with NO
    FULL key (``None`` — PIECEWISE-only, or a mode that never captured)
    refuses by name too: no FULL graph covers any decode batch, so the
    graphs-on record would describe an engine that runs attention eager
    on every step (a mode without FULL decode graphs is a call to make
    explicitly, never a quiet pass). ``composed_capture_sizes`` (the
    composer's list, when the launch carries one) must every one appear in
    the resolved list — vLLM takes an explicit list verbatim, so a missing
    entry means the engine did not run the composed surface."""
    if not isinstance(results, list) or not results:
        return [f"lora-native RPC returned {results!r} — expected one "
                "resolved report PER WORKER; a server whose capture surface "
                "cannot be read must not pass readiness"]
    S = int(max_num_seqs)
    problems: List[str] = []
    for i, rep in enumerate(results):
        if not isinstance(rep, Mapping):
            problems.append(f"worker {i}: non-report result {rep!r}")
            continue
        for key in ("cudagraph_capture_sizes", "max_cudagraph_capture_size"):
            if key not in rep:
                problems.append(
                    f"worker {i}: {key} MISSING from the resolved report "
                    "(config surface drift) — the CUDA-graph capture surface "
                    "cannot be verified")
        if any(k not in rep for k in ("cudagraph_capture_sizes",
                                      "max_cudagraph_capture_size")):
            continue
        max_size = rep["max_cudagraph_capture_size"]
        sizes = rep["cudagraph_capture_sizes"]
        if (isinstance(max_size, bool) or not isinstance(max_size, int)):
            problems.append(
                f"worker {i}: max_cudagraph_capture_size resolved to "
                f"{max_size!r}, not an int — under graphs vLLM resolves it "
                "to min(2 * max_num_seqs, 512) or the composed value; an "
                "unresolved capture max means the engine is not the "
                "graphs-on engine the attestation would claim")
            continue
        if (not isinstance(sizes, list) or not sizes or not all(
                isinstance(x, int) and not isinstance(x, bool) for x in sizes)):
            problems.append(
                f"worker {i}: cudagraph_capture_sizes resolved to {sizes!r} — "
                "expected a non-empty list of ints (empty is what vLLM sets "
                "under enforce_eager)")
            continue
        largest = max(sizes)
        if max_size < S:
            problems.append(
                f"worker {i}: max_cudagraph_capture_size {max_size} < the "
                f"composed max_num_seqs {S} — the dispatcher returns "
                "CUDAGraphMode.NONE for every decode batch above the capture "
                "max, so production decode at the served width would run "
                "EAGER under a graphs-on attestation (a manifest "
                "--max-cudagraph-capture-size is wrapper-owned; a width "
                "above vLLM's default cap of 512 needs a deliberately "
                "composed capture max)")
        if largest < S:
            problems.append(
                f"worker {i}: the largest cudagraph_capture_size {largest} "
                f"< the composed max_num_seqs {S} — no captured graph covers "
                "a full-width decode batch, which the dispatcher then runs "
                "eager")
        if largest != max_size:
            problems.append(
                f"worker {i}: max(cudagraph_capture_sizes) {largest} != "
                f"max_cudagraph_capture_size {max_size} — vLLM's "
                "post_init_cudagraph_sizes asserts these agree once "
                "resolved; the capture surface is not the pinned one")
        if composed_capture_sizes is not None:
            missing = sorted(int(x) for x in composed_capture_sizes if int(x) not in sizes)
            if missing:
                problems.append(
                    f"worker {i}: the composed --cudagraph-capture-sizes entries {missing} are "
                    f"MISSING from the resolved cudagraph_capture_sizes {sizes!r} — vLLM 0.25.1 "
                    "takes an explicit list verbatim (de-duplicated, sorted, filtered only "
                    "against max_num_batched_tokens), so the engine did not run the composed "
                    "capture surface")
        # THE FULL SURFACE (docstring): the dispatcher's FULL keys, held to S
        full_keys = ("cudagraph_full_capture_sizes", "cudagraph_full_capture_max",
                     "uniform_decode_query_len")
        missing_full = [k for k in full_keys if k not in rep]
        if missing_full:
            problems.append(
                f"worker {i}: {missing_full} MISSING from the resolved report — the FULL "
                "CUDA-graph capture surface (the dispatcher's FULL keys) cannot be verified, "
                "so whether a full-width decode step runs under a FULL graph is unknown; a "
                "worker extension that does not report it must not pass readiness")
            continue
        full_sizes = rep["cudagraph_full_capture_sizes"]
        full_max = rep["cudagraph_full_capture_max"]
        qlen = rep["uniform_decode_query_len"]
        if isinstance(qlen, bool) or not isinstance(qlen, int) or qlen < 1:
            problems.append(
                f"worker {i}: uniform_decode_query_len resolved to {qlen!r}, not a positive "
                "int — the decode step's token count per sequence cannot be read")
            continue
        if (not isinstance(full_sizes, list) or not all(
                isinstance(x, int) and not isinstance(x, bool) for x in full_sizes)):
            problems.append(
                f"worker {i}: cudagraph_full_capture_sizes resolved to {full_sizes!r} — "
                "expected a list of ints (the token counts of the dispatcher's FULL keys)")
            continue
        if full_max is not None and (isinstance(full_max, bool) or not isinstance(full_max, int)):
            problems.append(
                f"worker {i}: cudagraph_full_capture_max resolved to {full_max!r}, not an int "
                "or None")
            continue
        if (full_max is None) != (not full_sizes) or (full_sizes and max(full_sizes) != full_max):
            problems.append(
                f"worker {i}: cudagraph_full_capture_max {full_max!r} disagrees with "
                f"cudagraph_full_capture_sizes {full_sizes!r} — two readings of one dispatcher "
                "disagree; the FULL surface is not the reported one")
            continue
        outside = sorted(x for x in full_sizes if x not in sizes)
        if outside:
            problems.append(
                f"worker {i}: FULL capture sizes {outside} are not in cudagraph_capture_sizes "
                f"{sizes!r} — the dispatcher's FULL keys come from the capture list "
                "(initialize_cudagraph_keys @0.25.1); the surface is not the reported one")
        # WHICH runner surface answered (V1 dispatcher / V2 manager — Slurm 521717 ran gemma
        # on V2 and the V1-only reader refused every request): recorded, and held to the
        # class kind model_runner_report attested when that kind is pinned
        surface = rep.get("model_runner_surface")
        if surface not in ("v1", "v2"):
            problems.append(
                f"worker {i}: model_runner_surface resolved to {surface!r}, not 'v1' | 'v2' — the "
                "report does not say which runner's graph bookkeeping it read (V1 "
                "cudagraph_dispatcher.cudagraph_keys or V2 cudagraph_manager.graphs); a worker "
                "extension that does not say must not pass readiness")
        kind = rep.get("model_runner_kind")
        if surface in ("v1", "v2") and kind in ("v1", "v2") and kind != surface:
            problems.append(
                f"worker {i}: the FULL surface was read off the {surface} runner surface but the "
                f"model runner class is the {kind} runner ({rep.get('model_runner_class')!r}) — "
                "two readings of one worker disagree; the surface is not the runner's")
        need = S * qlen
        if full_max is None:
            problems.append(
                f"worker {i}: NO FULL CUDA-graph capture size — the dispatcher holds no FULL key "
                f"under cudagraph_mode {rep.get('cudagraph_mode')!r}, so every decode batch "
                "(the full-width one at max_num_seqs "
                f"{S} included) dispatches PIECEWISE with attention eager or runs eager "
                "outright; a graphs-on attestation would misdescribe production decode. A mode "
                "without FULL decode graphs is a deliberate call, never a quiet pass")
        elif full_max < need:
            problems.append(
                f"worker {i}: the largest FULL CUDA-graph capture size {full_max} < the composed "
                f"max_num_seqs {S}" + (f" x uniform_decode_query_len {qlen} = {need} tokens"
                                       if qlen != 1 else "")
                + " — vLLM captures FULL decode graphs only for capture sizes <= max_num_seqs "
                "(v1/cudagraph_dispatcher.py initialize_cudagraph_keys @0.25.1), so max_num_seqs "
                "itself must be a captured size; decode batches of "
                f"{full_max // qlen + 1}..{S} sequences pad to the next PIECEWISE size and run "
                "the splitting ops (attention) eager — the 2026-09-16 Modal H200 collapse "
                "(S=51 on the default list, FULL largest=48, 58 -> 9 tok/s/seq). The composer's "
                "cudagraph_capture_sizes_for(S) puts S in the list; a launch that skipped it "
                "must not pass readiness")
    return problems


# --- the record -------------------------------------------------------------------


#: THE SERVING VENV every serving record names: the venv the serving PROCESS runs in
#: (``sys.prefix``), its name's variant + hash, the identity common.sh's f22_serving_venv exported (variant, the
#: requirements sha256 the name hashes), and the fused-kernel packages the venv can import — so a reader of an
#: attestation / endpoint / cells_served record knows which venv served, before or after a variant landed.
SERVING_VENV_KEY = "serving_venv"
SERVING_VENV_ENV, SERVING_VENV_VARIANT_ENV, SERVING_VENV_REQS_SHA_ENV = (
    "SISTEER_SERVING_VENV", "SISTEER_SERVING_VENV_VARIANT", "SISTEER_SERVING_REQS_SHA256")
#: THE ENGINE IDENTITY: the engine is venv-independent — vLLM 0.25.1 vendors its causal_conv1d Triton kernels and its fla
#: ops (byte-identical across the base / nemotron / gdn venvs), and the overlays reach only the HF+PEFT reference — so what makes two
#: served numerics paths the SAME is {the vllm version, the BASE requirements digest, the flashinfer version} (FlashInfer's GDN prefill
#: kernel is the one environment-dependent engine dispatch). The admission gate compares exactly this block; the venv's variant and
#: full digest stay as the HF-reference provenance.
ENGINE_IDENTITY_KEY = "engine_identity"
ENGINE_IDENTITY_FIELDS: Tuple[str, ...] = ("vllm_version", "base_requirements_sha256", "flashinfer_version")
SERVING_BASE_REQS_SHA_ENV = "SISTEER_SERVING_BASE_REQS_SHA256"
FLASHINFER_DISTS: Tuple[str, ...] = ("flashinfer-python", "flashinfer")
SERVING_VENV_NAME_PREFIX = "sisteer-petri-vllm"
#: the kernel packages whose presence distinguishes the variants (GDN: fla + causal_conv1d; Nemotron: mamba_ssm + causal_conv1d)
SERVING_VENV_KERNEL_PACKAGES = (("fla", "flash-linear-attention"), ("causal_conv1d", "causal-conv1d"), ("mamba_ssm", "mamba-ssm"))


def parse_serving_venv_name(name: str) -> Dict[str, Any]:
    """``sisteer-petri-vllm[-<variant>]-<hash8>`` -> ``{"campaign_venv": bool, "variant": str|None, "hash8": str|None}``;
    any other name is ``campaign_venv: False`` with both None (a developer venv, a Modal image)."""
    name = str(name or "")
    if not name.startswith(SERVING_VENV_NAME_PREFIX + "-"):
        return {"campaign_venv": False, "variant": None, "hash8": None}
    rest = name[len(SERVING_VENV_NAME_PREFIX) + 1:]
    parts = rest.rsplit("-", 1)
    if len(parts) == 1:
        return {"campaign_venv": True, "variant": None, "hash8": parts[0] or None}
    variant, h = parts
    return {"campaign_venv": True, "variant": variant or None, "hash8": h or None}


def serving_venv_facts(*, prefix: Optional[str] = None, environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The venv facts of THIS process (the standup / the vLLM server both run in the serving venv): the prefix and
    its parsed name; the identity common.sh exported (``None`` when the process was not stood up through
    f22_serving_venv); the kernel packages importable here with their installed versions. Stdlib only, never raises."""
    import importlib.util
    import importlib.metadata as _md
    import os as _os
    import sys as _sys

    env = _os.environ if environ is None else environ
    pfx = str(prefix if prefix is not None else _sys.prefix)
    name = _os.path.basename(pfx.rstrip("/"))
    parsed = parse_serving_venv_name(name)
    kernels: Dict[str, Any] = {}
    for module, dist in SERVING_VENV_KERNEL_PACKAGES:
        try:
            importable = importlib.util.find_spec(module) is not None
        except Exception:  # noqa: BLE001 — a broken package is "not importable", recorded, never fatal
            importable = False
        try:
            version = _md.version(dist)
        except Exception:  # noqa: BLE001
            version = None
        kernels[module] = {"importable": bool(importable), "version": version}
    exported_variant = env.get(SERVING_VENV_VARIANT_ENV)

    def _version(*dists: str) -> Optional[str]:
        for dist in dists:
            try:
                return _md.version(dist)
            except Exception:  # noqa: BLE001
                continue
        return None
    engine_identity = {"vllm_version": _version("vllm"),
                       "base_requirements_sha256": env.get(SERVING_BASE_REQS_SHA_ENV) or None,
                       "flashinfer_version": _version(*FLASHINFER_DISTS)}
    return {
        "prefix": pfx, "name": name, **parsed,
        ENGINE_IDENTITY_KEY: engine_identity,
        "exported": {"venv": env.get(SERVING_VENV_ENV), "variant": (exported_variant if exported_variant else None),
                     "requirements_sha256": env.get(SERVING_VENV_REQS_SHA_ENV)},
        "exported_matches_prefix": (env.get(SERVING_VENV_ENV) is None or
                                    _os.path.realpath(str(env.get(SERVING_VENV_ENV))) == _os.path.realpath(pfx)),
        "kernels": kernels,
    }


def build_serving_attestation(*, model: str, revision: Any,
                              bindings: Mapping[str, Any],
                              expected: Mapping[str, Any],
                              expected_native_lora: Mapping[str, Any],
                              scheduler_geometry: Mapping[str, Any],
                              models_body: Any,
                              kernel_ceiling: Optional[Mapping[str, Any]] = None,
                              model_runner: Optional[Mapping[str, Any]] = None,
                              residual_slot: Optional[Mapping[str, Any]] = None,
                              ) -> Dict[str, Any]:
    """Assemble the file body (module docstring's schema) from the parts
    the launcher validated: the launch bindings (pid, host, argv hash,
    nonce, ...), the composed expectation, the per-name expectation, the
    validated worker reports, and the ``/v1/models`` body. The ``engine``
    and ``lora_config`` blocks are lifted from worker 0's report (every
    asserted key already equals the expectation in every worker; the
    recorded ones — ``cudagraph_mode``, ``lora_dtype`` — ride along per
    worker in ``scheduler_geometry.worker_reports``)."""
    from sisteer.capeval.serving import worker_reports as wr

    reports = scheduler_geometry.get("worker_reports")
    if not isinstance(reports, list) or not reports:
        raise ServingAttestationError(
            "no validated worker reports — nothing to attest")
    rep0 = reports[0]
    try:
        # + the FULL capture surface (ENGINE_CAPTURE_COVERAGE_KEYS): lifted
        # by name like the rest; a report without it is not a lora-native one
        engine = {k: rep0[k] for k in ENGINE_KEYS + ENGINE_CAPTURE_COVERAGE_KEYS}
        lora_config = {k: rep0[src] for k, src in _REPORT_LORA_KEYS.items()}
        execution_modules = {k: rep0[src]
                             for k, src in _REPORT_EXECUTION_KEYS.items()}
    except KeyError as e:
        raise ServingAttestationError(
            f"worker report lacks {e.args[0]!r} — not a lora-native report") \
            from e
    # schema 2: the residual slot block is REQUIRED — the
    # launcher verified it at readiness (vllm_server._verify_native_lora_state)
    if residual_slot is None:
        raise ServingAttestationError(
            f"a schema-{SERVING_ATTESTATION_SCHEMA_VERSION} attestation carries the "
            f"{RESIDUAL_SLOT_KEY} block; the launcher verifies the residual slot at readiness and "
            "hands it here — refusing to write a record without it")
    _validate_residual_slot_key(residual_slot, where="the composed launch")
    residual_adapters = residual_slot["expected"]["adapters"]
    names = sorted(expected_native_lora["adapters"])
    roots = models_roots(models_body, names)
    adapters: Dict[str, Dict[str, Any]] = {}
    for name in names:
        info = expected_native_lora["adapters"][name]
        d = Path(info["dir"])
        cfg = wr.read_adapter_config(d)
        rinfo = residual_adapters.get(name)
        if not isinstance(rinfo, Mapping):
            raise ServingAttestationError(
                f"{RESIDUAL_SLOT_KEY}.expected.adapters lacks {name!r} — the residual expectation "
                "was not computed over the composed adapter set")
        if rinfo.get("residual_fingerprint") != info.get("residual_fingerprint"):
            raise ServingAttestationError(
                f"adapter {name!r}: the LoRA expectation's residual_fingerprint "
                f"{info.get('residual_fingerprint')!r} != the residual expectation's "
                f"{rinfo.get('residual_fingerprint')!r} — two readings of one directory disagree")
        adapters[name] = {
            "dir": str(d),
            "config_sha256": sha256_file(d / ADAPTER_CONFIG_FILE),
            "weights_sha256": sha256_file(d / ADAPTER_WEIGHTS_FILE),
            "fingerprint_bf16": info["fingerprint_bf16"],
            "r": int(info["r"]),
            "lora_int_id": int(info["lora_int_id"]),
            "module_keys": list(info["module_keys"]),
            "petri_intervention_hash": cfg.get("petri_intervention_hash"),
            "petri_zero_adapter": bool(cfg.get("petri_zero_adapter", False)),
            "models_root": roots[name],
            "petri_source_kind": cfg.get("petri_source_kind"),
            "residual_fingerprint": rinfo.get("residual_fingerprint"),
            "position_mode": rinfo.get("position_mode"),
        }
    worker_adapters = [
        [{k: a[k] for k in WORKER_ADAPTER_KEYS} for a in rep["lora_adapters"]]
        for rep in reports]
    id_name_map = {str(info["lora_int_id"]): name
                   for name, info in expected_native_lora["adapters"].items()}
    body: Dict[str, Any] = {
        "kind": SERVING_ATTESTATION_KIND,
        "schema_version": SERVING_ATTESTATION_SCHEMA_VERSION,
        "model": model,
        "revision": revision,
        "attested_at": time.time(),
    }
    body.update(dict(bindings))
    body.update({
        SERVING_VENV_KEY: serving_venv_facts(),   # the venv this server runs in
        "expected": dict(expected),
        "expected_native_lora": dict(expected_native_lora),
        "scheduler_geometry": dict(scheduler_geometry),
        "engine": engine,
        "lora_config": lora_config,
        "adapters": adapters,
        "worker_adapters": worker_adapters,
        "id_name_map": id_name_map,
        "name_binding": {"asserted": True, "basis": NAME_BINDING_BASIS},
        "execution_modules": execution_modules,
        RESIDUAL_SLOT_KEY: dict(residual_slot),
    })
    # THE LORA KERNEL INDEX CEILING the composer held the geometry to
    # (module constants) — additive keys, written when the launch carried them
    if model_runner is not None:
        _validate_model_runner_key(model_runner, where="the composed launch")
        body[MODEL_RUNNER_KEY] = {"class": str(model_runner["class"]),
                                  "kind": str(model_runner["kind"])}
    if kernel_ceiling is not None:
        ceiling = kernel_ceiling.get(LORA_KERNEL_CEILING_KEY)
        dim = kernel_ceiling.get(LORA_MAX_MODULE_DIM_KEY)
        if ceiling is not None or dim is not None:
            _validate_kernel_ceiling_keys(ceiling, dim, where="the composed launch")
            body[LORA_KERNEL_CEILING_KEY] = int(ceiling)
            body[LORA_MAX_MODULE_DIM_KEY] = int(dim)
            # the belt behind execution_stride_problems: a record whose
            # executing-stride maximum exceeds the D its ceiling was sized
            # from is never written
            if int(execution_modules["stride_max"]) > int(dim):
                raise ServingAttestationError(
                    f"the composed launch: execution_modules.stride_max="
                    f"{execution_modules['stride_max']} > {LORA_MAX_MODULE_DIM_KEY}={dim} — "
                    "an executing LoRA wrapper is wider than the kernel index ceiling's D; "
                    "refusing to write a record that attests a ceiling the kernels exceed")
        targets = kernel_ceiling.get(LORA_TARGET_MODULES_KEY)
        if targets:
            _validate_target_modules_key(targets, where="the composed launch")
            body[LORA_TARGET_MODULES_KEY] = sorted(str(t) for t in targets)
    missing = [k for k in REQUIRED_TOP_KEYS_V2 if k not in body]
    if missing:
        raise ServingAttestationError(
            f"attestation body lacks {missing} — refusing to write a record "
            "the strict reader would refuse")
    return body


def _validate_kernel_ceiling_keys(ceiling: Any, dim: Any, *, where: str) -> None:
    """Both additive ceiling keys present, positive ints, and the ceiling
    equal to ``(2^31 - 1) // dim`` — one cannot ride without the other."""
    for name, v in ((LORA_KERNEL_CEILING_KEY, ceiling), (LORA_MAX_MODULE_DIM_KEY, dim)):
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ServingAttestationError(f"{where}: {name}={v!r} is not a positive int")
    if int(ceiling) != LORA_KERNEL_INDEX_LIMIT // int(dim):
        raise ServingAttestationError(
            f"{where}: {LORA_KERNEL_CEILING_KEY}={ceiling} != (2^31-1) // {LORA_MAX_MODULE_DIM_KEY}"
            f"={dim} ({LORA_KERNEL_INDEX_LIMIT // int(dim)})")


def _validate_model_runner_key(mr: Any, *, where: str) -> None:
    """The additive runner key: ``{"class": <one of MODEL_RUNNER_CLASSES>,
    "kind": its kind}`` — an unpinned runner class is never written or read
    as attested."""
    if (not isinstance(mr, Mapping) or not isinstance(mr.get("class"), str)
            or mr["class"] not in MODEL_RUNNER_CLASSES
            or mr.get("kind") != MODEL_RUNNER_CLASSES[mr["class"]]):
        raise ServingAttestationError(
            f"{where}: {MODEL_RUNNER_KEY}={mr!r} is not one of the pinned GPU model runners "
            f"{sorted(MODEL_RUNNER_CLASSES)} with its kind — an unknown runner is an unpinned "
            "engine")


def _validate_target_modules_key(targets: Any, *, where: str) -> None:
    """The additive adapted-class key: a non-empty list of non-empty str."""
    if (not isinstance(targets, list) or not targets
            or not all(isinstance(t, str) and t for t in targets)):
        raise ServingAttestationError(
            f"{where}: {LORA_TARGET_MODULES_KEY}={targets!r} is not a non-empty list of "
            "module suffixes")


def _validate_engine_capture_coverage_keys(engine: Mapping[str, Any], *, where: str) -> None:
    """:data:`ENGINE_CAPTURE_COVERAGE_KEYS` when present: ``cudagraph_full_
    capture_sizes`` a list of ints, ``cudagraph_full_capture_max`` an int or
    None, the two agreeing (max of the list, None iff empty). Either key
    present without the other refuses — half a surface is a drift, not a
    pre-landing record."""
    present = [k for k in ENGINE_CAPTURE_COVERAGE_KEYS if k in engine]
    if not present:
        return
    if len(present) != len(ENGINE_CAPTURE_COVERAGE_KEYS):
        raise ServingAttestationError(
            f"{where}: engine block carries {present} without "
            f"{[k for k in ENGINE_CAPTURE_COVERAGE_KEYS if k not in engine]} — half a FULL "
            "capture surface is a reader drift, not a record written before the keys existed")
    sizes = engine["cudagraph_full_capture_sizes"]
    mx = engine["cudagraph_full_capture_max"]
    if (not isinstance(sizes, list)
            or not all(isinstance(x, int) and not isinstance(x, bool) for x in sizes)):
        raise ServingAttestationError(
            f"{where}: engine.cudagraph_full_capture_sizes={sizes!r} is not a list of ints")
    if mx is not None and (isinstance(mx, bool) or not isinstance(mx, int)):
        raise ServingAttestationError(
            f"{where}: engine.cudagraph_full_capture_max={mx!r} is not an int or None")
    if (mx is None) != (not sizes) or (sizes and max(sizes) != mx):
        raise ServingAttestationError(
            f"{where}: engine.cudagraph_full_capture_max={mx!r} disagrees with "
            f"engine.cudagraph_full_capture_sizes={sizes!r}")


def read_serving_attestation(path: Path) -> Dict[str, Any]:
    """The strict reader: every key of the schema present, ``kind ==
    "lora_native"``, ``schema_version`` one of :data:`READABLE_SCHEMA_VERSIONS`
    (a schema-2 record must carry the :data:`RESIDUAL_SLOT_KEY` block and the
    schema-2 adapter keys; a schema-1 record — every certification written
    before native-psr-serving — reads unchanged), ``name_binding.asserted is
    True``, ``engine.enforce_eager is False``, the adapter and
    worker-adapter blocks fully keyed, ``id_name_map`` consistent with
    the adapters' ``lora_int_id``. Anything else refuses — a
    ``steering_attestation.json`` handed here, an older or foreign file,
    or a record whose binding was inferred rather than asserted."""
    path = Path(path)
    if not path.is_file():
        raise ServingAttestationError(f"no serving attestation at {path}")
    try:
        att = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ServingAttestationError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(att, Mapping):
        raise ServingAttestationError(f"{path} is not a JSON object")
    missing = [k for k in REQUIRED_TOP_KEYS if k not in att]
    if missing:
        raise ServingAttestationError(
            f"{path} lacks required keys {missing} — not a "
            f"{SERVING_ATTESTATION_KIND} serving attestation")
    if att["kind"] != SERVING_ATTESTATION_KIND:
        raise ServingAttestationError(
            f"{path}: kind={att['kind']!r}, expected "
            f"{SERVING_ATTESTATION_KIND!r}")
    version = att["schema_version"]
    if version not in READABLE_SCHEMA_VERSIONS:
        raise ServingAttestationError(
            f"{path}: schema_version={version!r}, this reader "
            f"understands {list(READABLE_SCHEMA_VERSIONS)}")
    v2 = version == 2
    if v2:
        if RESIDUAL_SLOT_KEY not in att:
            raise ServingAttestationError(
                f"{path}: a schema-2 attestation lacks the {RESIDUAL_SLOT_KEY} block — the residual "
                "slot's readiness verdict is part of the record since native-psr-serving")
        _validate_residual_slot_key(att[RESIDUAL_SLOT_KEY], where=str(path))
    adapter_keys = ADAPTER_KEYS if v2 else ADAPTER_KEYS_V1
    nb = att["name_binding"]
    if not isinstance(nb, Mapping) or nb.get("asserted") is not True:
        raise ServingAttestationError(
            f"{path}: name_binding.asserted is not true "
            f"({nb!r}) — an id -> name map that was inferred rather than "
            "asserted per name is not an attestation")
    engine = att["engine"]
    if not isinstance(engine, Mapping) or any(k not in engine for k in ENGINE_KEYS):
        raise ServingAttestationError(
            f"{path}: engine block lacks keys "
            f"{[k for k in ENGINE_KEYS if not isinstance(engine, Mapping) or k not in engine]}")
    if engine["enforce_eager"] is not False:
        raise ServingAttestationError(
            f"{path}: engine.enforce_eager={engine['enforce_eager']!r} — a "
            "lora_native attestation records a graphs-on engine")
    # the FULL capture surface (ENGINE_CAPTURE_COVERAGE_KEYS, 2026-09-16) is
    # ADDITIVE: absent on every record written before it (tolerated — the
    # schema is unchanged), typed when present (a malformed value refuses)
    _validate_engine_capture_coverage_keys(engine, where=str(path))
    lc = att["lora_config"]
    if not isinstance(lc, Mapping) or any(k not in lc for k in LORA_CONFIG_KEYS):
        raise ServingAttestationError(f"{path}: lora_config block malformed ({lc!r})")
    em = att["execution_modules"]
    if (not isinstance(em, Mapping)
            or any(k not in em for k in EXECUTION_MODULES_KEYS)
            or not isinstance(em["module_keys"], list) or not em["module_keys"]
            or not isinstance(em["module_names"], list) or not em["module_names"]
            or not all(isinstance(m, str) for m in em["module_names"])
            or not isinstance(em["strides"], Mapping)
            or isinstance(em["stride_max"], bool)
            or not isinstance(em["stride_max"], int) or em["stride_max"] < 1):
        raise ServingAttestationError(
            f"{path}: execution_modules block malformed ({em!r}) — a lora_native "
            "attestation records the model modules that carry LoRA execution: "
            "canonical keys, exact names, and each wrapper's (in, out) stride with "
            "their maximum")
    adapters = att["adapters"]
    if not isinstance(adapters, Mapping) or not adapters:
        raise ServingAttestationError(f"{path}: adapters block is empty or malformed")
    for name, block in adapters.items():
        if not isinstance(block, Mapping) or any(k not in block for k in adapter_keys):
            raise ServingAttestationError(
                f"{path}: adapter {name!r} block lacks keys "
                f"{[k for k in adapter_keys if not isinstance(block, Mapping) or k not in block]}")
    if LORA_KERNEL_CEILING_KEY in att or LORA_MAX_MODULE_DIM_KEY in att:
        _validate_kernel_ceiling_keys(att.get(LORA_KERNEL_CEILING_KEY),
                                      att.get(LORA_MAX_MODULE_DIM_KEY), where=str(path))
        if int(em["stride_max"]) > int(att[LORA_MAX_MODULE_DIM_KEY]):
            raise ServingAttestationError(
                f"{path}: execution_modules.stride_max={em['stride_max']} > "
                f"{LORA_MAX_MODULE_DIM_KEY}={att[LORA_MAX_MODULE_DIM_KEY]} — the record "
                "attests a kernel index ceiling an executing LoRA wrapper exceeds")
    if LORA_TARGET_MODULES_KEY in att:
        _validate_target_modules_key(att[LORA_TARGET_MODULES_KEY], where=str(path))
    if MODEL_RUNNER_KEY in att:
        _validate_model_runner_key(att[MODEL_RUNNER_KEY], where=str(path))
    id_map = att["id_name_map"]
    want_map = {str(b["lora_int_id"]): n for n, b in adapters.items()}
    if not isinstance(id_map, Mapping) or dict(id_map) != want_map:
        raise ServingAttestationError(
            f"{path}: id_name_map {id_map!r} disagrees with the adapters' "
            f"lora_int_id ({want_map!r})")
    wa = att["worker_adapters"]
    if not isinstance(wa, list) or not wa or not all(
            isinstance(per_worker, list) and all(
                isinstance(a, Mapping) and all(k in a for k in WORKER_ADAPTER_KEYS)
                for a in per_worker)
            for per_worker in wa):
        raise ServingAttestationError(
            f"{path}: worker_adapters block malformed")
    return dict(att)


def engine_identity(att: Mapping[str, Any]) -> Dict[str, Any]:
    """The fixed sub-shape the routing proxy validates and echoes
    (``RoutingProxy`` started with ``engine_attestation=``): kind, eager state,
    graph mode, the LoRA config, and per name the two file digests, the
    fingerprint, the asserted ``lora_int_id``, the
    ``petri_intervention_hash`` the loaded adapter's config carries (``None``
    on the zero adapter) and the config's ``petri_zero_adapter`` flag. Built
    from a record that already passed :func:`read_serving_attestation`. The
    hash is in the identity because the proxy holds it to every genuine
    lora route's block: with the digests alone, a regenerated table whose
    block claimed intervention B over the attested adapter of object A
    (digests copied over) started and stamped rows with B's hash while A's
    weights served. The ZERO FLAG is in the identity too: the proxy used to
    exempt a route from that hash check on the TABLE's
    ``petri_zero_adapter`` alone, so a table naming a genuine adapter
    ``none__lora0`` (its digests copied over) started and returned genuine
    steering stamped as the no-op control; now the attested flag must agree
    with the block's before any exemption. The key order is
    ``proxy.ENGINE_ADAPTER_KEYS``, pinned equal.

    Since native-psr-serving (schema 2): ``residual_slot: {"armed": bool,
    "n_workers": int}`` — whether the worker served the residual slot, and how
    many workers attested (the proxy binds every failure claim's per-worker
    result list to that count; absent on a schema-1 record) — and per adapter
    its object KIND (``petri_source_kind``), the payload's
    ``residual_fingerprint`` and its ``position_mode`` (all ``None`` for a plain
    LoRA and on a schema-1 record, whose engine had no slot: ``armed False``).
    The top-level key order is ``proxy.ENGINE_ATTESTATION_KEYS``."""
    rs = att.get(RESIDUAL_SLOT_KEY)
    return {
        "kind": att["kind"],
        "enforce_eager": att["engine"]["enforce_eager"],
        "cudagraph_mode": att["engine"]["cudagraph_mode"],
        "lora_config": dict(att["lora_config"]),
        # + the attested WORKER COUNT (schema 2 records it): the proxy holds every failure claim's
        # per-worker result list to exactly this many entries
        # (read_serving_attestation already refused a schema-2 record whose n_workers is not a positive
        # int, so an ARMED identity always carries the count the proxy requires)
        RESIDUAL_SLOT_KEY: ({"armed": bool(rs["armed"]),
                             **({"n_workers": int(rs["n_workers"])} if "n_workers" in rs else {})}
                            if isinstance(rs, Mapping) else {"armed": False}),
        "adapters": {
            name: {"config_sha256": b["config_sha256"],
                   "weights_sha256": b["weights_sha256"],
                   "fingerprint_bf16": b["fingerprint_bf16"],
                   "lora_int_id": b["lora_int_id"],
                   "petri_intervention_hash": b["petri_intervention_hash"],
                   "petri_zero_adapter": b["petri_zero_adapter"],
                   "petri_source_kind": b.get("petri_source_kind"),
                   "residual_fingerprint": b.get("residual_fingerprint"),
                   "position_mode": b.get("position_mode")}
            for name, b in att["adapters"].items()},
    }
