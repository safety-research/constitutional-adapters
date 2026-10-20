"""Plugin-free worker reports: what a vLLM worker can attest about its
RESOLVED engine state, importable with neither vllm-lens nor vllm at
module import.

WHY THIS MODULE EXISTS. The lens
path's worker extension (``lens_worker.SisteerWorkerExtension``)
SUBCLASSES vllm-lens's extension, so the module that defines it cannot
be imported inside a plugin-free process — and a lora-native server is
exactly that: ``VLLM_PLUGINS=""`` keeps vllm-lens out so CUDA graphs
stay on. The pure report functions the ``sisteer_scheduler_config`` RPC
is built from are therefore moved HERE, verbatim, and ``lens_worker``
imports them back by name (``lens_worker.LensWorkerError is
worker_reports.LensWorkerError``; every ``lw.X`` reference and the RPC
body's literal ``out.update(batch_invariant_report())`` line are
unchanged). ``native_worker.NativeWorkerExtension`` composes the SAME
functions plus the new ones below, so both serving modes attest through
one code path per fact.

NEW FOR THE LORA-NATIVE MODE:

- :func:`engine_mode_report` — the RESOLVED eager/graph state
  (``model_config.enforce_eager``, ``compilation_config.cudagraph_mode``
  — ``config/vllm.py`` @0.25.1 forces ``CUDAGraphMode.NONE`` iff
  ``enforce_eager``), ``cudagraph_specialize_lora``, and whether any
  ``vllm_lens`` module got imported into the worker process anyway.
  Nothing anywhere attested eager/graph state before this: a launch that
  composed ``--no-enforce-eager`` and got eager regardless (a plugin
  loaded) passed every check the stack had.
- :func:`lora_config_report` — the RESOLVED ``LoRAConfig`` (max_loras,
  max_lora_rank, max_cpu_loras, lora_dtype, fully_sharded_loras).
- THE ADAPTER FINGERPRINT (:func:`adapter_fingerprint_bf16` on the worker
  side over the loaded ``LoRALayerWeights``; :func:`expected_adapter_
  fingerprint_bf16` on the launcher side over the adapter directory).
  The launcher's expectation and the worker's report must be EQUAL per
  serving name, which is what binds ``route.cell_id`` -> the frontend's
  ``lora_int_id`` -> the weights the worker actually applies.
  Design points, each load-bearing:

  * bf16, via a ``torch.int16`` BYTE VIEW. vLLM casts every adapter
    tensor exactly once at load (``LoRAModel.from_lora_tensors``:
    ``tensor.to(device=..., dtype=lora_dtype)``; ``lora_dtype "auto"`` is
    the model dtype, bf16 here), so the launcher applies the same single
    fp32 -> bf16 cast to the file's factors and both sides hash the bf16
    bits. ``.numpy()`` on a bf16 tensor raises ``TypeError: Got
    unsupported ScalarType BFloat16`` on torch 2.11 CPU (reproduced on
    the borrowed venv), so :func:`bf16_bytes` REINTERPRETS the storage as
    int16 — same bytes, no conversion. Repeated identical casts are
    byte-identical (``bf16 -> bf16`` is the identity and ``bf16 -> fp32 ->
    bf16`` round-trips exactly), so a content fingerprint cannot, and
    must not, count casts; it detects CONTENT (one flipped mantissa bit
    changes it).
  * CANONICAL MODULE KEYS INSIDE THE HASH. Every module is keyed
    ``L<layer>.<site>.o_proj`` (:func:`canonical_module_key`, from either
    the HF/PEFT spelling or vLLM's post-``hf_to_vllm_mapper`` spelling),
    and the per-module digests are hashed SORTED BY KEY — so swapping two
    same-shaped layers' tensors changes the fingerprint, while module
    order, storage orientation and device do not. A name the regex does
    not match REFUSES (no guessing).
  * REFUSALS, never tolerances. A packed module
    (``PackedLoRALayerWeights`` — q/k/v or gate/up fused, list-valued
    ``lora_a``/``lora_b``) has no defined fingerprint and is not an
    admitted export kind here. A non-unit scaling (``lora_alpha != r``)
    is refused because ``LoRALayerWeights.optimize()`` folds ``alpha/r``
    INTO ``lora_b`` in place at load, so the worker's bytes would no
    longer be the file's cast bytes; the admitted kinds (``lora_all``,
    ``lora_subtraction``, the zero adapter) all export ``alpha == r``.
    Ambiguous orientation (a dim equal to ``r`` on both axes) refuses.
    Non-bf16 tensors refuse (the resolved ``lora_dtype`` is GPU-verify;
    a drift fails readiness by name instead of hashing the wrong bits).
- :func:`loaded_adapters_report` — every loaded adapter's id, rank,
  canonical module keys, raw names + shapes (so a refusal is diagnosable
  from the attestation log alone), fingerprint, and its EXACT-NAME
  EXECUTION BINDINGS: for every recognised module carrying a LoRA execution
  wrapper, the factor key vLLM's OWN lookup selects for it
  (``LoRAModelManager._get_lora_layer_weights`` -> ``LoRAModel.get_lora``
  -> ``loras.get(module_name)`` @0.25.1, the exact dict lookup
  ``activate_adapter`` runs per registered module, ``reset_lora`` on a
  miss) or ``None``. Canonical keys unify the HF/PEFT and post-mapper
  spellings for the fingerprint, but vLLM applies by exact post-mapper
  name — so a factor loaded as ``language_model.model.model.layers.0...``
  against a wrapper at ``language_model.model.layers.0...`` canonicalises
  identically, fingerprints identically, and is never applied. The
  launcher holds every fingerprinted key to a REAL binding on its
  execution module (``serving_attestation.native_lora_report_problems``).
  Read off ``model_runner.lora_manager._adapter_manager`` (the attribute
  ``worker_manager.py`` itself uses @0.25.1; GPU-verify — a name drift
  refuses readiness naming the attribute).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import (Any, Callable, Dict, List, Mapping, NamedTuple, Optional,
                    Sequence, Tuple)

import torch

#: the two PEFT files every adapter directory carries — the ONE spelling in
#: ``serving_attestation`` (a stdlib-only leaf, so the worker process stays
#: import-light; ``lora_export`` writes the same names).
from sisteer.capeval.serving.serving_attestation import (  # noqa: E402,F401
    ADAPTER_CONFIG_FILE, ADAPTER_WEIGHTS_FILE, MODEL_RUNNER_CLASSES)
from sisteer.capeval.serving.serving_attestation import (  # noqa: E402
    EXPERTS_PROJECTIONS, MODULE_KIND_EXPERTS, MODULE_KIND_EXPERTS_PROJECTION,
    MODULE_KIND_LINEAR, MODULE_KIND_PACKED, PACKED_COMPONENTS, execution_carrier_key,
    experts_projection_keys, module_key_of, module_kind_of_key, packed_component_keys)


class LensWorkerError(RuntimeError):
    """A worker-wiring refusal (structure drift, condemned run, bad
    install) — loud, never a silent no-op (vllm-lens's own documented
    failure mode is silently capturing nothing, so every seam here
    refuses instead of degrading)."""


# --- the resolved scheduler geometry (launch-time silent-clamp protection) ---------


#: the resolved SCHEDULER-OWNED int fields the launcher verifies against
#: the invariant it composed (vllm_server's no-chunking sizing + pins) —
#: every one a stored attribute of vllm/config/scheduler.py's
#: SchedulerConfig at the pinned 0.25.1 tag. max_model_len is
#: DELIBERATELY absent: at the tag it is an ``InitVar`` on
#: SchedulerConfig — consumed by __post_init__, NOT stored — and its
#: docstring says "This is stored in the ModelConfig"; reading it off
#: scheduler_config would AttributeError on the real stack (a fake-only
#: attribute the CPU tests pin against, test_petri_lens_worker).
SCHEDULER_CONFIG_FIELDS = (
    "max_num_seqs",
    "max_num_batched_tokens",
    # the rest of the no-chunking invariant: the threshold CHUNKS long
    # prompts regardless of the budget, and >1 partial prefills silently
    # auto-set it (SchedulerConfig.__post_init__ at the tag):
    "long_prefill_token_threshold",
    "max_num_partial_prefills",
    "max_long_partial_prefills",
)


def scheduler_config_report(vllm_config: Any) -> Dict[str, Any]:
    """The worker's RESOLVED scheduler state, as plain scalars — what
    the ``sisteer_scheduler_config`` RPC returns so the LAUNCHER can
    verify the whole no-chunking invariant (sizing, partial-prefill
    pins, effective scheduled-token ceiling, no speculative decoding);
    argv alone proves nothing about the resolved engine config. Reads
    mirror the REAL pinned 0.25.1 layout exactly:

    - scheduler-owned values off ``vllm_config.scheduler_config``
      (:data:`SCHEDULER_CONFIG_FIELDS`);
    - ``max_model_len`` off ``vllm_config.model_config`` — its canonical
      home; SchedulerConfig's is an InitVar and NOT a runtime attribute
      at the tag (constant's note above);
    - ``max_num_scheduled_tokens`` (a real, usually-None SchedulerConfig
      field: "usually equal to max_num_batched_tokens, but can be
      smaller ... such as speculative decoding") resolved to its
      EFFECTIVE value — the budget when None;
    - the speculative state off ``vllm_config.speculative_config``
      (None = disabled), which covers every enable route including the
      speculators-format-model auto-enable that never touches a flag;
    - the EFFECTIVE scheduler class, resolved exactly the way the
      pinned ``get_scheduler_cls`` resolves it (stored None -> the
      stock V1 Scheduler, or AsyncScheduler under async_scheduling;
      anything else = a custom class, reported as its dotted path) —
      every other attested knob is INTERPRETED by this class, so the
      invariant means nothing under a custom scheduler.

    Refuses loudly on any surface drift rather than reporting a guess —
    an unverifiable invariant must fail the launch, never pass it."""
    sched = getattr(vllm_config, "scheduler_config", None)
    if sched is None:
        raise LensWorkerError(
            "vllm_config carries no scheduler_config — not the pinned "
            "vLLM config surface; the resolved scheduler geometry cannot "
            "be verified, so the launch must not pass readiness")
    out: Dict[str, Any] = {}
    for name in SCHEDULER_CONFIG_FIELDS:
        if not hasattr(sched, name):
            raise LensWorkerError(
                f"scheduler_config has no {name!r} — not the pinned vLLM "
                "SchedulerConfig surface; refusing to report a geometry "
                "that cannot be read")
        try:
            out[name] = int(getattr(sched, name))
        except (TypeError, ValueError) as e:
            raise LensWorkerError(
                f"scheduler_config.{name} is not an integer "
                f"({getattr(sched, name)!r}) — refusing to report an "
                "unreadable geometry") from e
    # max_model_len: the ModelConfig is its canonical home (the pinned
    # SchedulerConfig only borrows it as an InitVar and stores nothing).
    model = getattr(vllm_config, "model_config", None)
    if model is None or not hasattr(model, "max_model_len"):
        raise LensWorkerError(
            "vllm_config carries no model_config.max_model_len — not the "
            "pinned vLLM config surface (SchedulerConfig's max_model_len "
            "is an InitVar and never stored, so there is no fallback to "
            "read); refusing to report a geometry that cannot be read")
    try:
        out["max_model_len"] = int(model.max_model_len)
    except (TypeError, ValueError) as e:
        raise LensWorkerError(
            f"model_config.max_model_len is not an integer "
            f"({model.max_model_len!r}) — refusing to report an "
            "unreadable geometry") from e
    # the EFFECTIVE per-step scheduled-token ceiling: the real 0.25.1
    # field defaults to None (= the budget) and diverges only under
    # token-appending features (spec decode) — its absence is drift.
    if not hasattr(sched, "max_num_scheduled_tokens"):
        raise LensWorkerError(
            "scheduler_config has no 'max_num_scheduled_tokens' — not "
            "the pinned vLLM SchedulerConfig surface; the effective "
            "per-step ceiling cannot be verified, so the launch must "
            "not pass readiness")
    mnst = sched.max_num_scheduled_tokens
    try:
        out["effective_max_num_scheduled_tokens"] = (
            out["max_num_batched_tokens"] if mnst is None else int(mnst))
    except (TypeError, ValueError) as e:
        raise LensWorkerError(
            f"scheduler_config.max_num_scheduled_tokens is neither None "
            f"nor an integer ({mnst!r}) — refusing to report an "
            "unreadable ceiling") from e
    # the speculative state: None = disabled. The ATTRIBUTE must exist —
    # "cannot prove no spec decode" is a refusal, never a pass.
    if not hasattr(vllm_config, "speculative_config"):
        raise LensWorkerError(
            "vllm_config carries no speculative_config attribute — not "
            "the pinned vLLM config surface; a worker that cannot prove "
            "speculative decoding is OFF must not pass readiness "
            "(spec decode schedules q>1 decode steps, which the plan "
            "lattice refuses)")
    spec = vllm_config.speculative_config
    out["speculative"] = (
        None if spec is None else
        f"{type(spec).__name__}(num_speculative_tokens="
        f"{getattr(spec, 'num_speculative_tokens', '?')})")
    # the EFFECTIVE scheduler class, mirroring the pinned
    # SchedulerConfig.get_scheduler_cls exactly (read at the 0.25.1
    # tag): a stored None resolves to the stock V1 Scheduler — or to
    # AsyncScheduler iff async_scheduling is truthy — and anything else
    # is a CUSTOM class (string qualname or the class object itself),
    # reported as its dotted path so the launcher can hold it to the
    # pinned default. Both attributes are REQUIRED: a scheduler state
    # that cannot be read cannot be vouched for. NB the installed
    # 0.25.1 resolved async_scheduling TRUTHY with no flag on argv
    # (job 194568 — omission is not a pin), which this attestation
    # correctly refused; the composer now pins --no-async-scheduling
    # explicitly (vllm_server.SYNC_SCHEDULING_FLAG) and this report
    # remains the runtime half that catches any such drift.
    for attr in ("scheduler_cls", "async_scheduling"):
        if not hasattr(sched, attr):
            raise LensWorkerError(
                f"scheduler_config has no {attr!r} — not the pinned vLLM "
                "SchedulerConfig surface; the resolved scheduler class "
                "cannot be verified, so the launch must not pass "
                "readiness")
    raw_cls = sched.scheduler_cls
    if raw_cls is None:
        out["scheduler_cls"] = (
            "vllm.v1.core.sched.async_scheduler.AsyncScheduler"
            if sched.async_scheduling else
            "vllm.v1.core.sched.scheduler.Scheduler")
    elif isinstance(raw_cls, str):
        out["scheduler_cls"] = raw_cls
    else:
        out["scheduler_cls"] = (
            f"{getattr(raw_cls, '__module__', '?')}."
            f"{getattr(raw_cls, '__qualname__', repr(raw_cls))}")
    # informational, never asserted: V1 forces chunked prefill ON; the
    # no-chunking guarantee is the SIZING + pins, not this flag.
    flag = getattr(sched, "enable_chunked_prefill", None)
    out["enable_chunked_prefill"] = None if flag is None else bool(flag)
    # the RESOLVED prefix-caching state, ASSERTED by the launcher against
    # the composed choice (vllm_server.expected_scheduler_state): with
    # caching on, correctness rests on the proxy's per-request cache_salt
    # discipline, so a launch whose cache state cannot be read (or that
    # silently flipped) must not pass readiness.
    cache = getattr(vllm_config, "cache_config", None)
    if cache is None or not hasattr(cache, "enable_prefix_caching"):
        raise LensWorkerError(
            "vllm_config carries no cache_config.enable_prefix_caching — "
            "not the pinned vLLM config surface; the resolved cache state "
            "cannot be verified, so the launch must not pass readiness")
    out["enable_prefix_caching"] = bool(cache.enable_prefix_caching)
    # the RESOLVED Mamba SSM-state cache dtype — cache_config.
    # mamba_ssm_cache_dtype at the pinned 0.25.1 (a plain string: "auto"
    # unless a launch composed one; the engine sizes its MambaSpec pages
    # from it). RECORDED for every launch, ASSERTED by the launcher only
    # when it composed the opt-in (vllm_server.expected_scheduler_state
    # adds the key iff launch.mamba_ssm_cache_dtype is set), so a dense
    # launch attests nothing new, and a config surface that lacks the
    # attribute reports None — which the launcher's comparison refuses
    # exactly on the launches that composed the dtype. GPU-verified
    # 2026-09-02 on Nemotron-3-Nano-4B (vllm 0.25.1+cu129): with float32
    # composed, the attested MambaSpec page was 3,997,696 B == the fp32
    # ssm state (96 x 80 x 128 x 4 B) + bf16 conv state, padded 0.18% to
    # the attention page — the composed dtype is what the engine resolved.
    mdt = getattr(cache, "mamba_ssm_cache_dtype", None)
    out["mamba_ssm_cache_dtype"] = None if mdt is None else str(mdt)
    # the RESOLVED KV cache size: gpu_worker.initialize_cache wrote
    # num_gpu_blocks onto cache_config after memory profiling — the
    # authoritative record of whether the composed geometry is actually
    # KV-BACKED. max_num_seqs is a scheduler CEILING, not a grant: vLLM
    # runs a 48 x 16384 geometry over a 146,118-token KV cache (18.6%
    # coverage) without a single error anywhere — it schedules ~9
    # sequences and PREEMPTS the rest, silently delivering a fifth of
    # the requested width (the scale-12 gemma wave, job 232411; the
    # activation term, not KV, is what binds on gemma-4's geometry).
    # The launcher holds these to the composed budget at readiness
    # (vllm_server.validate_scheduler_reports), mirroring the smoke's
    # kv-coverage gate — the wave path previously had no equivalent.
    for name in ("num_gpu_blocks", "block_size"):
        val = getattr(cache, name, None)
        if isinstance(val, bool) or not isinstance(val, int) or val <= 0:
            raise LensWorkerError(
                f"cache_config.{name} is {val!r} — the resolved KV cache "
                "size cannot be read, so whether the composed geometry is "
                "KV-backed cannot be verified; a geometry that silently "
                "delivers a fraction of its requested concurrency must "
                "refuse at readiness, so the launch must not pass it")
        out[f"kv_{name}"] = int(val)
    return out


def kv_group_report(vllm_config: Any, model_runner: Any,
                    max_concurrency_fn: Optional[Callable[[Any, Any], Any]]
                    = None) -> Dict[str, Any]:
    """The RESOLVED KV-cache GROUP structure plus the allocator's own
    max-concurrency verdict — the hybrid-correct half of the KV-coverage
    invariant, merged into the ``sisteer_scheduler_config`` payload next
    to :func:`scheduler_config_report`'s scalars.

    WHY: ``num_gpu_blocks * block_size`` is token capacity only for a
    UNIFORM cache. On a hybrid model (Qwen3.5: full-attention +
    gated-delta-net groups) the scheduler block size is an LCM over the
    groups' block sizes and a Mamba/GDN group's pages hold per-sequence
    STATE, not token spans — so the uniform formula silently misstates
    coverage, in either direction, with every check still green. The
    authoritative number is vLLM's own
    ``get_max_concurrency_for_kv_cache_config`` (the same computation
    the engine logs as "Maximum concurrency for N tokens per request:
    X.XXx", whose own comment says sourcing capacity from it "handles
    hybrid layouts correctly") — read, never re-derived, per the
    authoritative-sources rule.

    GPU-VERIFY (all against the pinned 0.25.1 source, not executed
    here): ``model_runner.kv_cache_config`` is set on the V1 GPU runner
    by ``initialize_kv_cache`` and used throughout the tag's
    ``gpu_model_runner.py``; ``KVCacheConfig.kv_cache_groups`` is a list
    of ``KVCacheGroupSpec`` (``layer_names`` + ``kv_cache_spec``); every
    ``KVCacheSpec`` declares ``block_size`` and a ``page_size_bytes``
    property. Refusals are for UNREADABLE structure or a spec this
    report cannot account for — never for "hybrid detected": a hybrid
    with healthy coverage must pass, that being the point of the whole
    stack.

    ``max_concurrency_fn`` is injectable so every branch runs on CPU
    under tests; the RPC leaves it ``None`` and the real vLLM function
    is imported lazily (import failure = refusal, never a guess)."""
    kv_cfg = getattr(model_runner, "kv_cache_config", None)
    if kv_cfg is None:
        raise LensWorkerError(
            "the model runner carries no kv_cache_config — not the "
            "pinned V1 runner surface; the resolved KV-cache group "
            "structure cannot be read, so whether the composed geometry "
            "is actually KV-backed cannot be verified; the launch must "
            "not pass readiness")
    groups = getattr(kv_cfg, "kv_cache_groups", None)
    if not groups:
        raise LensWorkerError(
            f"kv_cache_config carries no KV cache groups ({groups!r}) — "
            "an attention-free or unrecognized cache layout whose "
            "coverage this report cannot certify; the launch must not "
            "pass readiness")
    out_groups: List[Dict[str, Any]] = []
    for i, g in enumerate(groups):
        spec = getattr(g, "kv_cache_spec", None)
        layer_names = getattr(g, "layer_names", None)
        try:
            out_groups.append({
                "spec": type(spec).__name__,
                "block_size": int(spec.block_size),
                "num_layers": len(layer_names),
                "page_size_bytes": int(spec.page_size_bytes),
            })
        except Exception as e:
            raise LensWorkerError(
                f"KV cache group {i} carries a spec this report cannot "
                f"account for ({type(spec).__name__}; "
                f"{type(e).__name__}: {e}) — coverage cannot be "
                "certified, so the launch must not pass readiness") from e
    if max_concurrency_fn is None:
        try:
            from vllm.v1.core.kv_cache_utils import (
                get_max_concurrency_for_kv_cache_config as
                max_concurrency_fn)
        except Exception as e:
            raise LensWorkerError(
                "vllm.v1.core.kv_cache_utils."
                "get_max_concurrency_for_kv_cache_config could not be "
                f"imported ({type(e).__name__}: {e}) — the allocator's "
                "own coverage verdict is unavailable and this report "
                "must not substitute a guess; the launch must not pass "
                "readiness") from e
    try:
        mc = float(max_concurrency_fn(vllm_config, kv_cfg))
    except Exception as e:
        raise LensWorkerError(
            "get_max_concurrency_for_kv_cache_config refused this "
            f"config ({type(e).__name__}: {e}) — a geometry whose KV "
            "coverage the allocator itself cannot compute must not pass "
            "readiness") from e
    if not math.isfinite(mc) or mc <= 0:
        raise LensWorkerError(
            f"the allocator reports a max concurrency of {mc!r} — not a "
            "usable coverage verdict; the launch must not pass "
            "readiness")
    return {"kv_num_cache_groups": len(out_groups),
            "kv_cache_groups": out_groups,
            "kv_max_concurrency": mc}


# --- the resolved batch-invariance state ---------------------------------------


#: vLLM versions whose ``vllm.model_executor.layers.batch_invariant`` module
#: exposes NO public reader of the active batch-invariance state and whose
#: PRIVATE module flag ``_batch_invariant_MODE`` was VERIFIED (2026-09-04,
#: against the installed ``vllm-0.25.1+cu129`` wheel this repository pins —
#: configs/capeval/vllm-requirements.txt) to be that state: ``False`` at
#: import, set ``True`` exactly once by ``enable_batch_invariant_mode()``,
#: which ``init_batch_invariance()`` calls iff ``envs.VLLM_BATCH_INVARIANT``
#: — and ``init_batch_invariance()`` runs from
#: ``vllm/v1/worker/gpu_worker.py::init_worker_distributed_environment``,
#: i.e. inside ``Worker.init_device()``, BEFORE ``load_model()``, so the flag
#: is settled by the time any post-load RPC reads it. The module's public
#: surface at that version is exactly ``enable_batch_invariant_mode``,
#: ``override_envs_for_invariance``, ``init_batch_invariance`` (plus the
#: kernels); there is no ``vllm_is_batch_invariant``. Reading a private flag
#: is accepted ONLY on a version listed here — any other version without the
#: public reader is an explicit error (the gate then refuses the standup),
#: never a guess: re-verify the surface at the new pin and extend this tuple.
BATCH_INVARIANT_MODE_ATTR_VERSIONS = ("0.25.1",)
#: the public reader newer vLLMs expose; preferred whenever present
BATCH_INVARIANT_PUBLIC_READER = "vllm_is_batch_invariant"
BATCH_INVARIANT_MODE_ATTR = "_batch_invariant_MODE"


def batch_invariant_active(bi_mod: Any, version: Any) -> Tuple[bool, str]:
    """The ACTIVE runtime batch-invariance state of ``bi_mod`` (vLLM's
    ``batch_invariant`` module), version-guarded: ``(active, surface)``.

    1. a callable :data:`BATCH_INVARIANT_PUBLIC_READER` on the module (newer
       vLLM) — used, whatever the version;
    2. else the private :data:`BATCH_INVARIANT_MODE_ATTR`, accepted only when
       it is a real ``bool`` AND ``version`` is one of
       :data:`BATCH_INVARIANT_MODE_ATTR_VERSIONS` (the pinned 0.25.1, where the
       flag's meaning was verified — see that constant);
    3. else raises ``LensWorkerError`` naming what the module does carry, so
       an unknown surface is recorded as an error and REFUSED by the gate,
       never read as "not active"."""
    reader = getattr(bi_mod, BATCH_INVARIANT_PUBLIC_READER, None)
    if callable(reader):
        val = reader()
        if not isinstance(val, bool):
            raise LensWorkerError(
                f"{BATCH_INVARIANT_PUBLIC_READER}() returned {val!r}, not a bool")
        return val, f"{BATCH_INVARIANT_PUBLIC_READER}()"
    ver = str(version) if version is not None else None
    if hasattr(bi_mod, BATCH_INVARIANT_MODE_ATTR):
        if ver not in BATCH_INVARIANT_MODE_ATTR_VERSIONS:
            raise LensWorkerError(
                f"vllm {ver!r} exposes no {BATCH_INVARIANT_PUBLIC_READER}() and its "
                f"private {BATCH_INVARIANT_MODE_ATTR} flag was verified to carry the "
                f"active state only at {list(BATCH_INVARIANT_MODE_ATTR_VERSIONS)}; "
                "refusing to read an unverified private flag (re-verify the "
                "surface at this pin and extend BATCH_INVARIANT_MODE_ATTR_VERSIONS)")
        val = getattr(bi_mod, BATCH_INVARIANT_MODE_ATTR)
        if not isinstance(val, bool):
            raise LensWorkerError(
                f"{BATCH_INVARIANT_MODE_ATTR} is {val!r}, not a bool")
        return val, f"{BATCH_INVARIANT_MODE_ATTR} (vllm {ver})"
    names = sorted(n for n in dir(bi_mod) if "batch_invariant" in n.lower())
    raise LensWorkerError(
        f"vllm {ver!r}: the batch_invariant module carries neither "
        f"{BATCH_INVARIANT_PUBLIC_READER}() nor {BATCH_INVARIANT_MODE_ATTR} "
        f"(batch-invariant names present: {names}); the active state is unreadable")


def batch_invariant_report(envs_mod: Any = None, bi_mod: Any = None,
                           version: Any = None) -> Dict[str, Any]:
    """The worker's RESOLVED batch-invariance state, for the scheduler /
    geometry attestation (review #5, item 4): ``vllm.envs.VLLM_BATCH_INVARIANT``
    (the env as vLLM parsed it — the value the kernel dispatch in
    ``layernorm.py`` / ``linear.py`` reads live) and the ACTIVE runtime state
    (:func:`batch_invariant_active` — the aten-override mode
    ``init_batch_invariance`` installs at ``init_device``), with
    ``batch_invariant_surface`` naming WHICH reader answered. Either surface
    missing is recorded as None with the error, never guessed; the launcher's
    intent is compared against these by the gate (and any reader of the
    attestation file), which REFUSES a report carrying an error. Modules and
    the version are injectable for the CPU tests; the real ones are imported
    lazily (vLLM is absent on this box)."""
    import importlib
    out: Dict[str, Any] = {"batch_invariant_env": None, "batch_invariant_active": None,
                           "batch_invariant_surface": None, "batch_invariant_error": None}
    errors = []
    try:
        envs = envs_mod if envs_mod is not None else importlib.import_module("vllm.envs")
        env = getattr(envs, "VLLM_BATCH_INVARIANT")
        if not isinstance(env, bool):
            raise LensWorkerError(f"vllm.envs.VLLM_BATCH_INVARIANT is {env!r}, not a bool")
        out["batch_invariant_env"] = env
    except Exception as e:  # noqa: BLE001 — recorded
        errors.append(f"env: {type(e).__name__}: {e}")
    try:
        bi = bi_mod if bi_mod is not None else importlib.import_module(
            "vllm.model_executor.layers.batch_invariant")
        ver = version if version is not None else getattr(
            importlib.import_module("vllm"), "__version__", None)
        active, surface = batch_invariant_active(bi, ver)
        out["batch_invariant_active"] = active
        out["batch_invariant_surface"] = surface
    except Exception as e:  # noqa: BLE001 — recorded
        errors.append(f"active: {type(e).__name__}: {e}")
    if errors:
        out["batch_invariant_error"] = "; ".join(errors)
    return out



# --- the RESOLVED engine mode + LoRA config (the lora-native attestation) ----------


def engine_mode_report(vllm_config: Any, *,
                       sys_modules: Optional[Mapping[str, Any]] = None
                       ) -> Dict[str, Any]:
    """The worker's RESOLVED eager/graph state, as plain scalars:

    - ``enforce_eager`` off ``vllm_config.model_config`` (its canonical
      home at 0.25.1: ``config/vllm.py`` reads exactly this attribute to
      force ``CUDAGraphMode.NONE`` and ``CompilationMode.NONE``);
    - ``cudagraph_mode``: the RESOLVED ``compilation_config.cudagraph_mode``
      enum member's NAME (``FULL_AND_PIECEWISE`` / ``FULL`` / ``PIECEWISE``
      / ``FULL_DECODE_ONLY`` / ``NONE`` — vLLM resolves the default per
      platform, so the launcher RECORDS the name and ASSERTS only
      ``cudagraph_mode_is_none is False``);
    - ``cudagraph_specialize_lora``: recorded (default True at the tag —
      a passthrough-only batch then runs no LoRA ops);
    - ``lens_plugin_imported``: True iff ANY ``vllm_lens`` module is
      present in ``sys_modules`` (the plugin's ``_activations_plugin``
      included). A lora-native worker must show False: the plugin
      forces eager process-wide, so its mere presence means the launch
      env did not keep it out. ``sys_modules`` is injectable for the CPU
      tests; the real report reads ``sys.modules``.

    Any missing attribute is a ``LensWorkerError`` (config-surface drift
    — a state that cannot be read must fail readiness, never pass it)."""
    model = getattr(vllm_config, "model_config", None)
    if model is None or not hasattr(model, "enforce_eager"):
        raise LensWorkerError(
            "vllm_config carries no model_config.enforce_eager — not the "
            "pinned vLLM config surface (config/vllm.py @0.25.1 reads "
            "model_config.enforce_eager to force CUDAGraphMode.NONE); the "
            "engine mode cannot be verified, so the launch must not pass "
            "readiness")
    eager = model.enforce_eager
    if not isinstance(eager, bool):
        raise LensWorkerError(
            f"model_config.enforce_eager is {eager!r}, not a bool — "
            "refusing to report an unreadable engine mode")
    comp = getattr(vllm_config, "compilation_config", None)
    if comp is None or not hasattr(comp, "cudagraph_mode"):
        raise LensWorkerError(
            "vllm_config carries no compilation_config.cudagraph_mode — "
            "not the pinned vLLM config surface; the resolved CUDA-graph "
            "mode cannot be verified, so the launch must not pass "
            "readiness")
    mode = comp.cudagraph_mode
    mode_name = getattr(mode, "name", None)
    if not isinstance(mode_name, str) or not mode_name:
        raise LensWorkerError(
            f"compilation_config.cudagraph_mode is {mode!r}, not a "
            "CUDAGraphMode enum member (no .name) — refusing to report an "
            "unreadable graph mode")
    if not hasattr(comp, "cudagraph_specialize_lora"):
        raise LensWorkerError(
            "compilation_config has no 'cudagraph_specialize_lora' — not "
            "the pinned vLLM CompilationConfig surface; refusing to report "
            "an unreadable graph configuration")
    specialize = comp.cudagraph_specialize_lora
    if not isinstance(specialize, bool):
        raise LensWorkerError(
            f"compilation_config.cudagraph_specialize_lora is "
            f"{specialize!r}, not a bool")
    # THE CAPTURE SURFACE (code-review finding, 2026-09-11): a non-NONE
    # mode alone does not mean decode runs under graphs. The dispatcher
    # returns CUDAGraphMode.NONE for ANY batch whose token count exceeds
    # compilation_config.max_cudagraph_capture_size (v1/cudagraph_dispatcher
    # .py dispatch() @0.25.1), so a `--max-cudagraph-capture-size 1` (or a
    # `--cudagraph-capture-sizes` list that stops short of max_num_seqs)
    # keeps enforce_eager False and a FULL_AND_PIECEWISE mode while every
    # production decode step runs eager. Both RESOLVED values are reported
    # here and held to the composed max_num_seqs by the launcher
    # (serving_attestation.cudagraph_coverage_problems). Under eager vLLM
    # sets max 0 and sizes [] (config/vllm.py:1325); unresolved is None.
    for attr in ("cudagraph_capture_sizes", "max_cudagraph_capture_size"):
        if not hasattr(comp, attr):
            raise LensWorkerError(
                f"compilation_config has no {attr!r} — not the pinned vLLM "
                "CompilationConfig surface (config/compilation.py @0.25.1); "
                "the CUDA-graph capture surface cannot be verified, so the "
                "launch must not pass readiness")
    sizes = comp.cudagraph_capture_sizes
    if sizes is None:
        sizes_out: Optional[List[int]] = None
    elif (isinstance(sizes, (list, tuple))
          and all(isinstance(x, int) and not isinstance(x, bool) for x in sizes)):
        sizes_out = sorted(int(x) for x in sizes)
    else:
        raise LensWorkerError(
            f"compilation_config.cudagraph_capture_sizes is {sizes!r}, not a "
            "list of ints — refusing to report an unreadable capture surface")
    max_size = comp.max_cudagraph_capture_size
    if max_size is None:
        max_out: Optional[int] = None
    elif isinstance(max_size, int) and not isinstance(max_size, bool):
        max_out = int(max_size)
    else:
        raise LensWorkerError(
            f"compilation_config.max_cudagraph_capture_size is {max_size!r}, "
            "not an int — refusing to report an unreadable capture surface")
    mods = sys.modules if sys_modules is None else sys_modules
    lens_mods = sorted(m for m in list(mods)
                       if m == "vllm_lens" or m.startswith("vllm_lens."))
    return {
        "enforce_eager": eager,
        "cudagraph_mode": mode_name,
        "cudagraph_mode_is_none": mode_name == "NONE",
        "cudagraph_specialize_lora": specialize,
        "cudagraph_capture_sizes": sizes_out,
        "max_cudagraph_capture_size": max_out,
        "lens_plugin_imported": bool(lens_mods),
    }


#: the ``CUDAGraphMode`` member whose dispatcher keys are the FULL graphs
#: (``vllm.config.CUDAGraphMode.FULL``, matched by ``.name`` so the report
#: needs no vllm import)
CUDAGRAPH_FULL_MODE_NAME = "FULL"


def cudagraph_capture_coverage_report(model_runner: Any) -> Dict[str, Any]:
    """The worker's RESOLVED FULL CUDA-graph capture surface, read off the GPU
    model runner's OWN graph bookkeeping — WHICHEVER of the two 0.25.1 runners
    this worker constructed (both are named ``GPUModelRunner``;
    ``VllmConfig.use_v2_model_runner`` picks V2 for a generate-type, non-hybrid,
    non-attention-free dense model — gemma4e4b, llama8b — and V1 for the hybrids
    — qwen35_4b, nemotron; ``model_runner_report`` records the class):

    - V1 (``v1/worker/gpu_model_runner.py``): ``model_runner.cudagraph_dispatcher.
      cudagraph_keys[CUDAGraphMode.FULL]`` ("the keys stored in dispatcher are the
      only source of truth for valid cudagraphs that can be dispatched at
      runtime"), the ``BatchDescriptor`` keys' ``num_tokens``;
    - V2 (``v1/worker/gpu/model_runner.py``): ``model_runner.cudagraph_manager.
      graphs`` (``CudaGraphManager``, ``v1/worker/gpu/cudagraph_utils.py``: the
      CAPTURED graphs keyed by ``BatchExecutionDescriptor(cg_mode, num_tokens,
      num_reqs, uniform_token_count, num_active_loras)``; ``dispatch()`` replays
      only from it once ``_graphs_captured``), the descriptors whose ``cg_mode``
      is FULL, their ``num_tokens``. V2 plans its FULL decode descriptors under
      the same rule as V1 (``_init_candidates``: ``rounded_num_tokens <=
      max_num_reqs * decode_query_len``), so the same S-in-the-list guarantee
      applies.

    As plain scalars, the same on both:

    - ``cudagraph_full_capture_sizes``: the sorted distinct FULL token counts
      (``[]`` when the mode captured no FULL graph);
    - ``cudagraph_full_capture_max``: the largest of them, ``None`` when none;
    - ``uniform_decode_query_len``: tokens per sequence of a uniform decode step
      (V1 ``dispatcher.uniform_decode_query_len`` / V2 ``manager.decode_query_len``;
      1 on every composed launch);
    - ``cudagraph_dispatcher_mode``: the mode the runner resolved and captured
      under (V1 ``dispatcher.cudagraph_mode`` / V2 ``manager.cudagraph_mode``);
    - ``model_runner_surface``: ``"v1"`` / ``"v2"`` — WHICH surface answered
      (the launcher holds it to ``model_runner_report``'s class kind);
    - ``cudagraph_lora_capture_cases``: V2's LoRA specialisation cases
      (``manager.lora_capture_cases``, the analogue of V1's
      ``cudagraph_specialize_lora``); ``None`` on V1.

    WHY THE RUNNER AND NOT THE CONFIG (2026-09-16, Modal H200, Qwen3.5-4B): the
    config's ``cudagraph_capture_sizes`` / ``max_cudagraph_capture_size`` are the
    MIXED (piecewise) surface; FULL decode graphs are captured only for capture
    sizes ``<= max_num_seqs * decode_query_len``, so with ``--max-num-seqs 51``
    on the default list the FULL surface stopped at 48 while the config read
    ``max 96``. WHY BOTH RUNNERS (Slurm 521717, gemma4e4b, 2026-09-16): the first
    cut read V1's dispatcher only and refused every request on the V2 runner
    ("GPUModelRunner has no 'cudagraph_dispatcher'") while vLLM had captured
    FULL 34/34 (17 sizes <= 107 x 2 LoRA cases). The launcher holds
    ``cudagraph_full_capture_max`` to ``max_num_seqs``
    (``serving_attestation.cudagraph_coverage_problems``).

    Any missing surface is a ``LensWorkerError`` — a runner with NEITHER
    ``cudagraph_dispatcher`` nor ``cudagraph_manager`` (or with both), a V1
    dispatcher whose keys are not initialised, a V2 manager that is ``None`` or
    whose graphs are not captured, a key without an int ``num_tokens``: a FULL
    surface that cannot be read must fail readiness, never pass it. The RPC runs
    after the server is up, i.e. after capture, so the captured state is the
    healthy one."""
    if model_runner is None:
        raise LensWorkerError(
            "no model_runner on the worker — the CUDA-graph FULL capture surface "
            "cannot be read (neither a V1 dispatcher nor a V2 manager), so the FULL "
            "surface cannot be verified; the launch must not pass readiness")
    has_dispatcher = hasattr(model_runner, "cudagraph_dispatcher")
    has_manager = hasattr(model_runner, "cudagraph_manager")
    runner = type(model_runner).__name__
    if has_dispatcher and has_manager:
        raise LensWorkerError(
            f"{runner} carries BOTH 'cudagraph_dispatcher' (the V1 surface) and "
            "'cudagraph_manager' (the V2 surface) — not either pinned GPU model runner "
            "(v1/worker/gpu_model_runner.py, v1/worker/gpu/model_runner.py @0.25.1); "
            "refusing to pick one")
    if not has_dispatcher and not has_manager:
        raise LensWorkerError(
            f"{runner} has neither 'cudagraph_dispatcher' (V1: v1/worker/gpu_model_runner.py) "
            "nor 'cudagraph_manager' (V2: v1/worker/gpu/model_runner.py) — not a pinned GPU "
            "model runner surface @0.25.1; the FULL capture surface cannot be verified, so the "
            "launch must not pass readiness")
    sizes = set()
    lora_cases: Optional[List[int]] = None
    if has_dispatcher:
        surface = "v1"
        disp = model_runner.cudagraph_dispatcher
        if disp is None:
            raise LensWorkerError(
                "cudagraph_dispatcher is None — the V1 runner has not constructed its "
                "dispatcher; the FULL capture surface is unknown")
        if getattr(disp, "keys_initialized", None) is not True:
            raise LensWorkerError(
                "cudagraph_dispatcher.keys_initialized is not True — the dispatcher has not "
                "resolved its capture keys (initialize_cudagraph_keys has not run), so the FULL "
                "capture surface is unknown; refusing to report it as empty")
        keys = getattr(disp, "cudagraph_keys", None)
        if not isinstance(keys, Mapping):
            raise LensWorkerError(
                f"cudagraph_dispatcher.cudagraph_keys is {type(keys).__name__}, not a mapping "
                "{CUDAGraphMode: set[BatchDescriptor]} — not the pinned dispatcher surface")
        full = None
        for mode, descs in keys.items():
            if getattr(mode, "name", None) == CUDAGRAPH_FULL_MODE_NAME:
                full = descs
                break
        if full is None:
            raise LensWorkerError(
                f"cudagraph_dispatcher.cudagraph_keys carries no {CUDAGRAPH_FULL_MODE_NAME} entry "
                f"(modes present: {[getattr(m, 'name', m) for m in keys]}) — not the pinned "
                "dispatcher surface (it holds a PIECEWISE and a FULL set from construction)")
        try:
            descs = list(full)
        except TypeError as e:
            raise LensWorkerError(
                f"cudagraph_keys[FULL] is not iterable ({type(full).__name__})") from e
        for d in descs:
            nt = getattr(d, "num_tokens", None)
            if isinstance(nt, bool) or not isinstance(nt, int):
                raise LensWorkerError(
                    f"a FULL cudagraph key {d!r} carries num_tokens={nt!r}, not an int — not the "
                    "pinned BatchDescriptor surface")
            sizes.add(int(nt))
        owner, owner_name = disp, "cudagraph_dispatcher"
        qlen = getattr(disp, "uniform_decode_query_len", None)
        qlen_name = "uniform_decode_query_len"
    else:
        surface = "v2"
        mgr = model_runner.cudagraph_manager
        if mgr is None:
            raise LensWorkerError(
                "cudagraph_manager is None — the V2 runner has not initialised its CUDA-graph "
                "manager (initialize_cudagraph_capture has not run), so the FULL capture surface "
                "is unknown; refusing to report it as empty")
        if getattr(mgr, "_graphs_captured", None) is not True:
            raise LensWorkerError(
                "cudagraph_manager._graphs_captured is not True — the V2 manager has not captured "
                "its graphs (dispatch() replays nothing before that), so the FULL capture surface "
                "is unknown; refusing to report it as empty")
        graphs = getattr(mgr, "graphs", None)
        if not isinstance(graphs, Mapping):
            raise LensWorkerError(
                f"cudagraph_manager.graphs is {type(graphs).__name__}, not a mapping "
                "{BatchExecutionDescriptor: CUDAGraph} — not the pinned V2 manager surface "
                "(v1/worker/gpu/cudagraph_utils.py @0.25.1)")
        for d in graphs:
            mode = getattr(getattr(d, "cg_mode", None), "name", None)
            if not isinstance(mode, str) or not mode:
                raise LensWorkerError(
                    f"a captured V2 graph key {d!r} carries no cg_mode member — not the pinned "
                    "BatchExecutionDescriptor surface")
            if mode != CUDAGRAPH_FULL_MODE_NAME:
                continue
            nt = getattr(d, "num_tokens", None)
            if isinstance(nt, bool) or not isinstance(nt, int):
                raise LensWorkerError(
                    f"a FULL V2 graph key {d!r} carries num_tokens={nt!r}, not an int — not the "
                    "pinned BatchExecutionDescriptor surface")
            sizes.add(int(nt))
        cases = getattr(mgr, "lora_capture_cases", None)
        if (not isinstance(cases, (list, tuple)) or not cases
                or not all(isinstance(c, int) and not isinstance(c, bool) for c in cases)):
            raise LensWorkerError(
                f"cudagraph_manager.lora_capture_cases is {cases!r}, not a non-empty list of ints "
                "— not the pinned V2 manager surface")
        lora_cases = [int(c) for c in cases]
        owner, owner_name = mgr, "cudagraph_manager"
        qlen = getattr(mgr, "decode_query_len", None)
        qlen_name = "decode_query_len"
    if isinstance(qlen, bool) or not isinstance(qlen, int) or qlen < 1:
        raise LensWorkerError(
            f"{owner_name}.{qlen_name} is {qlen!r}, not a positive int — the decode step's "
            "tokens per sequence cannot be read")
    dmode = getattr(getattr(owner, "cudagraph_mode", None), "name", None)
    if not isinstance(dmode, str) or not dmode:
        raise LensWorkerError(
            f"{owner_name}.cudagraph_mode is not a CUDAGraphMode member (no .name) — the "
            "resolved capture mode cannot be read")
    full_sizes = sorted(sizes)
    return {
        "cudagraph_full_capture_sizes": full_sizes,
        "cudagraph_full_capture_max": (full_sizes[-1] if full_sizes else None),
        "uniform_decode_query_len": int(qlen),
        "cudagraph_dispatcher_mode": dmode,
        "model_runner_surface": surface,
        "cudagraph_lora_capture_cases": lora_cases,
    }


def lora_config_report(vllm_config: Any) -> Dict[str, Any]:
    """The worker's RESOLVED ``LoRAConfig`` (``vllm_config.lora_config``
    @0.25.1: ``max_loras``, ``max_lora_rank``, ``max_cpu_loras`` — set to
    ``max_loras`` by the config's own validator when None —
    ``lora_dtype`` — the model dtype under ``"auto"``, reported as its
    string — and ``fully_sharded_loras``). ``lora_config is None`` means
    LoRA is not enabled in this engine: a lora-native server whose every
    adapter name would 404, refused by name. Missing attributes refuse
    as config-surface drift."""
    if not hasattr(vllm_config, "lora_config"):
        raise LensWorkerError(
            "vllm_config carries no lora_config attribute — not the pinned "
            "vLLM config surface; the LoRA configuration cannot be "
            "verified, so the launch must not pass readiness")
    lc = vllm_config.lora_config
    if lc is None:
        raise LensWorkerError(
            "vllm_config.lora_config is None — a lora-native server with no "
            "LoRA config (--enable-lora absent or ignored) would 404 every "
            "adapter name; the launch must not pass readiness")
    out: Dict[str, Any] = {}
    for attr, key in (("max_loras", "lora_max_loras"),
                      ("max_lora_rank", "lora_max_lora_rank"),
                      ("max_cpu_loras", "lora_max_cpu_loras")):
        if not hasattr(lc, attr):
            raise LensWorkerError(
                f"lora_config has no {attr!r} — not the pinned vLLM "
                "LoRAConfig surface; refusing to report an unreadable LoRA "
                "configuration")
        val = getattr(lc, attr)
        if isinstance(val, bool) or not isinstance(val, int):
            raise LensWorkerError(
                f"lora_config.{attr} is {val!r}, not an int — refusing to "
                "report an unreadable LoRA configuration")
        out[key] = int(val)
    for attr in ("lora_dtype", "fully_sharded_loras"):
        if not hasattr(lc, attr):
            raise LensWorkerError(
                f"lora_config has no {attr!r} — not the pinned vLLM "
                "LoRAConfig surface; refusing to report an unreadable LoRA "
                "configuration")
    out["lora_dtype"] = str(lc.lora_dtype)
    sharded = lc.fully_sharded_loras
    if not isinstance(sharded, bool):
        raise LensWorkerError(
            f"lora_config.fully_sharded_loras is {sharded!r}, not a bool")
    out["lora_fully_sharded"] = sharded
    # THE DEPLOYMENT-TIME RESTRICTION:
    # LoRAConfig.target_modules (--lora-target-modules @0.25.1) filters
    # which model modules get a LoRA EXECUTION wrapper (model_manager.
    # _match_target_modules -> is_in_target_modules) but does NOT filter
    # the CPU-side `loras` dict the fingerprint hashes (worker_manager.
    # _load_adapter checks against supported_lora_modules only). Reported
    # as a plain None-or-sorted-list so the launcher can assert it EQUALS the
    # composed --lora-target-modules class (the composer restricts the
    # wrappers to the adapted suffixes; None would mean every linear
    # projection wrapped) — the manifest may not name it (wrapper-owned).
    if not hasattr(lc, "target_modules"):
        raise LensWorkerError(
            "lora_config has no 'target_modules' — not the pinned vLLM "
            "LoRAConfig surface (config/lora.py @0.25.1); the deployment-time "
            "LoRA restriction cannot be verified, so the launch must not pass "
            "readiness")
    targets = lc.target_modules
    if targets is None:
        out["lora_target_modules"] = None
    elif (isinstance(targets, (list, tuple))
          and all(isinstance(t, str) for t in targets)):
        out["lora_target_modules"] = sorted(str(t) for t in targets)
    else:
        raise LensWorkerError(
            f"lora_config.target_modules is {targets!r}, not None or a list "
            "of str — refusing to report an unreadable LoRA restriction")
    return out


# --- the adapter fingerprint (launcher expectation == worker report, per name) -----


class FingerprintError(LensWorkerError):
    """A fingerprint refusal — packed module, folded scaling, an
    unrecognised module name, a non-bf16 tensor, an ambiguous shape.
    On the worker it fails the RPC (and so readiness) by name; on the
    launcher it refuses the launch at compose time."""


def canonical_module_key(module_name: str) -> str:
    """``"L<N>.<site>"`` for any spelling ``serving_attestation.module_key_of``
    recognises — the key the per-module digests are sorted and hashed under,
    so layer index and projection are INSIDE the fingerprint on both sides.
    Every adapted module of the companion repo's module sets canonicalises:
    ``L<N>.self_attn.{q,k,v,o}_proj`` (NemotronH ``mixer.*``),
    ``L<N>.mlp.{gate,up,down}_proj``, ``L<N>.mlp.shared_expert.{gate,up,down}_proj``,
    the routed experts' ``L<N>.mlp.experts.gate_up_proj`` (the file's / PEFT's
    ``...mlp.experts.base_layer``) and ``L<N>.mlp.experts`` (the wrapper: the
    file's down projection, the worker's stacked pair — the fingerprint
    un-packs it by the value's shape); and the vLLM PACKED wrappers
    ``L<N>.self_attn.qkv_proj`` / ``L<N>.mlp[.shared_expert].gate_up_proj``,
    which the fingerprint un-packs into their components in slice order.
    Accepted prefixes: HF/PEFT ``base_model.model.model[.language_model].
    layers.N``, vLLM's post-mapper ``language_model.model.layers.N`` /
    ``model.layers.N``. A name outside the vocabulary is a
    :class:`FingerprintError` naming the module — a router ``gate``, a
    ``conv1d``, a name without a layer index, a module spelling this module
    has never seen — never guessed into a key. The o_proj keys are the ones
    the paper's standups were certified on and are byte-identical."""
    if not isinstance(module_name, str) or not module_name:
        raise FingerprintError(
            f"module name {module_name!r} is not a non-empty string")
    key = module_key_of(module_name)
    if key is None:
        raise FingerprintError(
            f"module {module_name!r} is not a recognised adapted module (expected "
            "'...layers.<N>.<site>' with site one of (self_attn|mixer).{q,k,v,o,qkv}_proj, "
            "(mlp|mixer|mlp.shared_expert).{gate,up,down,gate_up}_proj, "
            "mlp.experts[.base_layer]) — the native fingerprint never guesses a "
            "canonical key")
    return key


def bf16_bytes(t: Any) -> bytes:
    """The raw bytes of a bf16 tensor, by REINTERPRETATION: ``t`` must
    already be ``torch.bfloat16`` (a cast here would hide a dtype drift
    the fingerprint exists to catch), and its storage is viewed as
    ``torch.int16`` before ``.numpy()`` — ``.numpy()`` on a bf16 tensor
    raises ``TypeError: Got unsupported ScalarType BFloat16`` on torch 2.11
    CPU. ``2 * numel`` bytes, little-endian (asserted once: both sides run
    on x86-64 here)."""
    if not isinstance(t, torch.Tensor):
        raise FingerprintError(
            f"fingerprint input is {type(t).__name__}, not a tensor")
    if t.dtype is not torch.bfloat16:
        raise FingerprintError(
            f"fingerprint is defined on bf16, got {t.dtype}")
    if sys.byteorder != "little":
        raise FingerprintError(
            f"fingerprint bytes are defined little-endian; this host is "
            f"{sys.byteorder!r}")
    return (t.detach().cpu().contiguous().view(torch.int16)
            .numpy().tobytes())


class ModuleWeights(NamedTuple):
    """One adapted module's LoRA factors as the worker holds them
    (``LoRALayerWeights`` @0.25.1: ``lora_a``, ``lora_b``, ``rank``,
    ``lora_alpha``, ``is_packed``; a ``PackedLoRALayerWeights`` — fused
    q/k/v or gate/up — has LIST-valued factors with ``None`` in an unadapted
    slot, ``is_packed=True``, ``lora_alpha=0`` and the per-component alphas
    in ``lora_alphas``; a stacked routed-experts entry has list-valued 3-D
    factors ``[gate_up, down]`` and ``is_packed=False``,
    ``model_manager.py:800-880``) or as the launcher reconstructs them from
    the file (one fp32 -> bf16 cast, ``is_packed=False``, the experts' flat
    factors reshaped to 3-D exactly as vLLM does). ``rank`` / ``lora_alpha``
    are the TOP-LEVEL ``r`` / ``lora_alpha`` on every entry (vLLM's
    ``LoRALayerWeights.from_config``); the physical rank comes from the
    tensor shapes."""

    lora_a: Any
    lora_b: Any
    rank: int
    lora_alpha: float
    is_packed: bool
    lora_alphas: Any = None


def _oriented(t: Any, rank: int, *, which: str, name: str) -> torch.Tensor:
    """``lora_a`` as ``[r, d_in]`` / ``lora_b`` as ``[d_out, r]``, from
    either stored orientation, with ``r`` known: a dim equal to ``r`` on
    exactly one axis decides; on both axes (or neither) the orientation
    is undecidable and REFUSES rather than guessing."""
    if not isinstance(t, torch.Tensor):
        raise FingerprintError(
            f"module {name!r} lora_{which} is {type(t).__name__}, not a "
            "tensor")
    if t.dim() != 2:
        raise FingerprintError(
            f"module {name!r} lora_{which} has {t.dim()} dims, expected 2")
    d0, d1 = int(t.shape[0]), int(t.shape[1])
    if d0 == rank and d1 == rank:
        raise FingerprintError(
            f"module {name!r} lora_{which} has shape {tuple(t.shape)} with "
            f"BOTH axes equal to r={rank}: the orientation is ambiguous; "
            "refusing to guess")
    if which == "A":  # want [r, d_in]
        if d0 == rank:
            return t
        if d1 == rank:
            return t.t()
    else:  # want [d_out, r]
        if d1 == rank:
            return t
        if d0 == rank:
            return t.t()
    raise FingerprintError(
        f"module {name!r} lora_{which} has shape {tuple(t.shape)}; neither "
        f"axis equals r={rank} — not a rank-{rank} factor")


def _check_unit_scaling(alpha: Any, rank: int, *, name: str, what: str = "module") -> None:
    """``alpha == rank`` (scaling exactly 1) or a :class:`FingerprintError`:
    ``LoRALayerWeights.optimize()`` folds ``alpha/r`` into ``lora_b`` in place
    at load (``PackedLoRALayerWeights.pack`` per component, before packing),
    so the fingerprint is defined only at scaling 1."""
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or float(alpha) != float(rank)):
        raise FingerprintError(
            f"non-unit LoRA scaling alpha={alpha!r} r={rank} on {what} "
            f"{name!r}: vLLM's LoRALayerWeights.optimize() folds alpha/r "
            "into lora_b in place at load; the fingerprint is defined "
            "only at scaling 1")


def _linear_digest(key: str, a: Any, b: Any, rank: int, *, name: str) -> bytes:
    """THE o_proj RECIPE, unchanged since the paper's standups were certified
    on it: orient by shape with ``r`` known (refuses ambiguity), then
    ``sha256(key + "\\0" + "A[r,in];B[out,r]" + "\\0" + bf16 bytes of A and B)``.
    Used for EVERY 2-D linear factor — o_proj, a packed component, a dense
    or shared-expert MLP projection — under its own logical key. For a
    linear factor the physical rank read off the shape must EQUAL the
    top-level ``r`` (every admitted export's invariant; the difference
    adapter doubles both); a factor matching ``r`` on neither axis refuses
    by name rather than being sized at either."""
    a = _oriented(a, rank, which="A", name=name)
    b = _oriented(b, rank, which="B", name=name)
    header = (f"A[{a.shape[0]},{a.shape[1]}];"
              f"B[{b.shape[0]},{b.shape[1]}]").encode("utf-8") + b"\0"
    return hashlib.sha256(
        key.encode("utf-8") + b"\0" + header + bf16_bytes(a) + bf16_bytes(b)).digest()


def _experts_digest(key: str, a: Any, b: Any, *, name: str
                    ) -> Tuple[bytes, Tuple[int, int, int, int]]:
    """One 3-D routed-expert projection: ``A [E, r_e, in]`` / ``B [E, out,
    r_e]`` — the form vLLM's ``_stack_moe_lora_weights`` leaves
    (``model_manager.py:857-880`` @0.25.1) and the launcher reproduces from
    the flat file. The PHYSICAL per-expert rank is ``A.shape[1]`` (must
    equal ``B.shape[2]``), E is ``A.shape[0]`` (must equal ``B.shape[0]``);
    the top-level ``r`` plays no part (it is the attention / shared-expert
    rank, and the flat file's rank axis is ``E*r_e``). A 2-D tensor here is
    the UNSTACKED flat layout: on a worker that means no ``FusedMoE3DWithLoRA``
    wrapper claimed the module (the expert adaptation would never be
    applied); on the launcher the reshape did not happen — refused by name
    either way. ``digest = sha256(key + "\\0" + "A3[E,r,in];B3[E,out,r]" +
    "\\0" + bf16 bytes of A and B)``; returns ``(digest, (E, r_e, in, out))``."""
    for which, t in (("A", a), ("B", b)):
        if not isinstance(t, torch.Tensor):
            raise FingerprintError(
                f"experts module {name!r} lora_{which} is {type(t).__name__}, not a tensor")
        if t.dim() == 2:
            raise FingerprintError(
                f"experts module {name!r} lora_{which} is a FLAT 2-D tensor {tuple(t.shape)} "
                "([E*r_e, in] / [out, E*r_e]): not stacked into vLLM's 3-D [E, r_e, in] / "
                "[E, out, r_e] form — on a worker no FusedMoE3DWithLoRA wrapper claimed this "
                "module (_stack_moe_lora_weights never ran, so the expert adaptation would "
                "never be applied); refusing")
        if t.dim() != 3:
            raise FingerprintError(
                f"experts module {name!r} lora_{which} has {t.dim()} dims, expected 3")
    E, r_e, d_in = (int(x) for x in a.shape)
    E_b, d_out, r_b = (int(x) for x in b.shape)
    if E != E_b or E < 1:
        raise FingerprintError(
            f"experts module {name!r}: lora_A {tuple(a.shape)} and lora_B {tuple(b.shape)} "
            f"disagree on the expert count (A.shape[0]={E}, B.shape[0]={E_b}); refusing")
    if r_e != r_b or r_e < 1:
        raise FingerprintError(
            f"experts module {name!r}: lora_A {tuple(a.shape)} and lora_B {tuple(b.shape)} "
            f"disagree on the per-expert rank (A.shape[1]={r_e}, B.shape[2]={r_b}); refusing")
    if d_in < 1 or d_out < 1:
        raise FingerprintError(
            f"experts module {name!r}: degenerate shapes lora_A {tuple(a.shape)} / lora_B "
            f"{tuple(b.shape)}")
    header = f"A3[{E},{r_e},{d_in}];B3[{E},{d_out},{r_e}]".encode("utf-8") + b"\0"
    digest = hashlib.sha256(
        key.encode("utf-8") + b"\0" + header + bf16_bytes(a) + bf16_bytes(b)).digest()
    return digest, (E, r_e, d_in, d_out)


def _is_list_valued(w: ModuleWeights) -> bool:
    return isinstance(w.lora_a, (list, tuple)) or isinstance(w.lora_b, (list, tuple))


def adapter_fingerprint_bf16(modules: Mapping[str, ModuleWeights]
                             ) -> Tuple[str, Tuple[str, ...]]:
    """``(fingerprint, sorted canonical LOGICAL keys)`` over one adapter's
    modules — the worker's loaded ``LoRAModel.loras`` (after vLLM's packing
    and MoE stacking) or the launcher's reconstruction from the file. Per
    module, by the kind its canonical key names (``serving_attestation.
    module_kind_of_key``):

    * LINEAR (o_proj, down_proj, an unpacked q/k/v or gate/up): REFUSE a
      packed / list-valued entry (vLLM does not pack this site); REFUSE
      ``lora_alpha != rank``; digest by the o_proj recipe
      (:func:`_linear_digest`) under the module's own key. Byte-identical to
      the certified o_proj fingerprint.
    * PACKED (``qkv_proj`` / ``gate_up_proj`` — ``PackedLoRALayerWeights``,
      ``model_manager.py:730-791``): list-valued factors, one slot per
      component in slice order (``PACKED_COMPONENTS``); an absent component
      is ``None`` in BOTH slots; each PRESENT component's OWN
      ``lora_alphas[i] == rank`` (``pack()`` calls ``optimize()`` per
      component; the packed entry's top-level ``lora_alpha`` is 0 and
      ignored); each component digests as a linear factor under ITS
      component key (``L<N>.self_attn.q_proj`` ...). A plain tensor pair
      under a packed site refuses (not a state vLLM produces).
    * EXPERTS (``...mlp.experts``): list-valued ``[gate_up, down]`` 3-D
      factors (the worker after ``_stack_moe_lora_weights``) -> two digests
      under ``L<N>.mlp.experts.gate_up_proj`` / ``.down_proj``
      (:func:`_experts_digest`); or a single 3-D pair (the launcher's
      reshaped file down projection) -> the down digest. ``None`` in a slot,
      or a flat 2-D tensor, refuses by name.
    * EXPERTS PROJECTION (``...mlp.experts.base_layer`` -> gate_up): a single
      3-D pair -> the gate_up digest. vLLM's stacker leaves this entry in
      ``loras`` pointing at the SAME reshaped tensors the wrapper entry's
      list holds, so the one admitted key collision is this alias with an
      EQUAL digest; any other collision refuses.

    Then every experts layer must carry BOTH projections (vLLM asserts both
    at load, ``model_manager.py:840-843``; a gate_up-only adapter is
    silently a no-op) with agreeing hidden size, or the fingerprint refuses
    naming the layer and the missing projection. The fingerprint is sha256
    over the digests SORTED BY KEY: layer identity and projection are inside
    it; module order, storage orientation, packing and device are not."""
    if not isinstance(modules, Mapping) or not modules:
        raise FingerprintError(
            "an adapter with no LoRA modules has no fingerprint — refusing "
            "to attest an empty adapter")
    digests: Dict[str, bytes] = {}
    sources: Dict[str, str] = {}
    source_kind: Dict[str, str] = {}
    experts_shapes: Dict[str, Dict[str, Tuple[str, Tuple[int, int, int, int]]]] = {}

    def _record(key: str, digest: bytes, name: str, kind: str) -> None:
        if key in digests:
            alias = (module_kind_of_key(key) == MODULE_KIND_EXPERTS_PROJECTION
                     and key.endswith("." + EXPERTS_PROJECTIONS[0])
                     and {source_kind[key], kind} == {MODULE_KIND_EXPERTS,
                                                      MODULE_KIND_EXPERTS_PROJECTION})
            if alias and digests[key] == digest:
                return          # vLLM's own leftover alias of the stacked gate_up tensors
            raise FingerprintError(
                f"two modules canonicalise to {key!r}: {sources[key]!r} and "
                f"{name!r}" + (" with DIFFERENT factors" if alias else "")
                + " — an adapter cannot adapt one projection twice")
        digests[key] = digest
        sources[key] = name
        source_kind[key] = kind

    for name, w in modules.items():
        if not isinstance(w, ModuleWeights):
            raise FingerprintError(
                f"module {name!r}: expected ModuleWeights, got "
                f"{type(w).__name__}")
        rank = w.rank
        if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
            raise FingerprintError(
                f"module {name!r}: rank={rank!r} is not a positive int")
        key = canonical_module_key(name)
        kind = module_kind_of_key(key)
        if kind == MODULE_KIND_LINEAR:
            if w.is_packed or _is_list_valued(w):
                raise FingerprintError(
                    f"packed module {name!r}: PackedLoRALayerWeights (list-valued "
                    f"factors) on {key}, a site vLLM does not pack (its packed "
                    f"wrappers are {sorted(PACKED_COMPONENTS)}) — not a state the "
                    "engine produces; refusing")
            _check_unit_scaling(w.lora_alpha, rank, name=name)
            _record(key, _linear_digest(key, w.lora_a, w.lora_b, rank, name=name), name, kind)
            continue
        if kind == MODULE_KIND_PACKED:
            components = packed_component_keys(key)
            leaf = key.rsplit(".", 1)[-1]
            if not (isinstance(w.lora_a, (list, tuple)) and isinstance(w.lora_b, (list, tuple))):
                raise FingerprintError(
                    f"module {name!r}: {leaf} is a packed wrapper site — vLLM's "
                    "PackedLoRALayerWeights carries LIST-valued factors, one slot per "
                    f"component {list(PACKED_COMPONENTS[leaf])}; a plain tensor pair under "
                    "this name is not a state the engine produces; refusing")
            if not w.is_packed:
                raise FingerprintError(
                    f"module {name!r}: list-valued factors under the packed site {leaf} "
                    "with is_packed=False — not PackedLoRALayerWeights; refusing")
            if len(w.lora_a) != len(components) or len(w.lora_b) != len(components):
                raise FingerprintError(
                    f"packed module {name!r}: {len(w.lora_a)} lora_a / {len(w.lora_b)} lora_b "
                    f"slots, expected {len(components)} ({list(PACKED_COMPONENTS[leaf])})")
            alphas = w.lora_alphas
            if not (isinstance(alphas, (list, tuple)) and len(alphas) == len(components)):
                raise FingerprintError(
                    f"packed module {name!r} carries no per-component lora_alphas list of "
                    f"{len(components)} (got {alphas!r}) — PackedLoRALayerWeights.lora_alphas "
                    "@0.25.1 is where each component's scaling lives (the packed entry's own "
                    "lora_alpha is 0); the unit-scaling check cannot be made, refusing")
            present = 0
            for i, comp_key in enumerate(components):
                a_i, b_i = w.lora_a[i], w.lora_b[i]
                comp = comp_key.rsplit(".", 1)[-1]
                if a_i is None and b_i is None:
                    continue                    # an unadapted component (pack() keeps None)
                if a_i is None or b_i is None:
                    raise FingerprintError(
                        f"packed module {name!r} component {comp!r} (slot {i}): lora_a is "
                        f"{'None' if a_i is None else 'set'} but lora_b is "
                        f"{'None' if b_i is None else 'set'} — half a factor; refusing")
                _check_unit_scaling(alphas[i], rank, name=name,
                                    what=f"packed component {comp!r} of")
                _record(comp_key,
                        _linear_digest(comp_key, a_i, b_i, rank, name=f"{name}[{comp}]"),
                        f"{name}[{comp}]", MODULE_KIND_LINEAR)
                present += 1
            if not present:
                raise FingerprintError(
                    f"packed module {name!r}: every component slot is None — nothing to "
                    "fingerprint")
            continue
        # the routed experts: top-level unit scaling (optimize() folds alpha/r
        # into the flat lora_b BEFORE _stack_moe_lora_weights), then 3-D digests
        _check_unit_scaling(w.lora_alpha, rank, name=name)
        if kind == MODULE_KIND_EXPERTS_PROJECTION:
            if _is_list_valued(w):
                raise FingerprintError(
                    f"experts projection {name!r}: list-valued factors under the gate_up "
                    "(base_layer) entry — vLLM leaves the two reshaped tensors there, not a "
                    "list; refusing")
            digest, shape = _experts_digest(key, w.lora_a, w.lora_b, name=name)
            _record(key, digest, name, kind)
            experts_shapes.setdefault(execution_carrier_key(key), {})[
                EXPERTS_PROJECTIONS[0]] = (name, shape)
            continue
        assert kind == MODULE_KIND_EXPERTS
        gate_up_key, down_key = experts_projection_keys(key)
        if _is_list_valued(w):
            if not (isinstance(w.lora_a, (list, tuple)) and isinstance(w.lora_b, (list, tuple))
                    and len(w.lora_a) == 2 and len(w.lora_b) == 2):
                raise FingerprintError(
                    f"experts module {name!r}: expected list-valued factors with exactly 2 "
                    "slots [gate_up, down] (vLLM's _stack_moe_lora_weights), got "
                    f"{_shape_or_none(w.lora_a)} / {_shape_or_none(w.lora_b)}; refusing")
            for i, (proj, pkey) in enumerate(zip(EXPERTS_PROJECTIONS, (gate_up_key, down_key))):
                a_i, b_i = w.lora_a[i], w.lora_b[i]
                if a_i is None or b_i is None:
                    raise FingerprintError(
                        f"experts module {name!r}: the {proj} slot ({i}) is None — a missing "
                        "expert binding: vLLM asserts both gate_up (…experts.base_layer) and "
                        "down (…experts) factors at load (model_manager.py:840-843 @0.25.1); "
                        "refusing")
                digest, shape = _experts_digest(pkey, a_i, b_i, name=f"{name}[{proj}]")
                _record(pkey, digest, f"{name}[{proj}]", kind)
                experts_shapes.setdefault(key, {})[proj] = (f"{name}[{proj}]", shape)
        else:
            digest, shape = _experts_digest(down_key, w.lora_a, w.lora_b, name=name)
            _record(down_key, digest, name, kind)
            experts_shapes.setdefault(key, {})[EXPERTS_PROJECTIONS[1]] = (name, shape)
    # every experts layer carries BOTH projections, on one hidden size
    for wrapper_key in sorted(experts_shapes):
        have = experts_shapes[wrapper_key]
        missing = [p for p in EXPERTS_PROJECTIONS if p not in have]
        if missing:
            present_names = sorted(n for n, _ in have.values())
            raise FingerprintError(
                f"experts module {wrapper_key} has {sorted(have)} ({present_names}) but no "
                f"{missing} factor — a missing expert binding: vLLM's _stack_moe_lora_weights "
                "asserts both gate_up (…mlp.experts.base_layer) and down (…mlp.experts) are "
                "present (model_manager.py:840-843 @0.25.1); a gate_up-only adapter is "
                "silently a no-op and a down-only adapter fails at load; refusing")
        (gu_name, (E_gu, _, d_gu, _)), (dn_name, (E_dn, _, _, d_dn)) = (
            have[EXPERTS_PROJECTIONS[0]], have[EXPERTS_PROJECTIONS[1]])
        if E_gu != E_dn:
            raise FingerprintError(
                f"experts module {wrapper_key}: gate_up ({gu_name!r}) has E={E_gu} experts but "
                f"down ({dn_name!r}) has E={E_dn}; refusing")
        if d_gu != d_dn:
            raise FingerprintError(
                f"experts module {wrapper_key}: gate_up ({gu_name!r}) input width {d_gu} != "
                f"down ({dn_name!r}) output width {d_dn} — the two projections of one MoE "
                "block share the hidden size; refusing")
    h = hashlib.sha256()
    for key in sorted(digests):
        h.update(key.encode("utf-8"))
        h.update(b"\0")
        h.update(digests[key])
    return h.hexdigest(), tuple(sorted(digests))


def read_adapter_config(adapter_dir: Path) -> Dict[str, Any]:
    """The adapter directory's ``adapter_config.json`` as a dict, with
    ``r`` validated as a positive int. Missing or unparseable REFUSES
    (:class:`FingerprintError`) — vLLM could not register such an
    adapter either."""
    path = Path(adapter_dir) / ADAPTER_CONFIG_FILE
    if not path.is_file():
        raise FingerprintError(
            f"{Path(adapter_dir)} has no {ADAPTER_CONFIG_FILE} — not a PEFT "
            "adapter directory; nothing to fingerprint")
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise FingerprintError(
            f"{path} is not valid JSON: {e}") from e
    if not isinstance(cfg, Mapping):
        raise FingerprintError(f"{path} is not a JSON object")
    r = cfg.get("r")
    if isinstance(r, bool) or not isinstance(r, int) or r < 1:
        raise FingerprintError(
            f"{path}: r={r!r} is not a positive int")
    return dict(cfg)


def _split_peft_key(key: str) -> Tuple[str, str]:
    """``(module_name, "A"|"B")`` from a PEFT safetensors key; any other
    key shape refuses — the admitted exports write plain
    ``<module>.lora_A.weight`` / ``<module>.lora_B.weight`` pairs only. A
    RESIDUAL payload key (``sisteer_residual.*``) is not a factor: callers
    skip it through :func:`is_residual_payload_key` before asking."""
    from .serving_attestation import (ADAPTER_KEY_LORA, ServingAttestationError,
                                      classify_adapter_key)

    try:
        cls, module, which = classify_adapter_key(key)
    except ServingAttestationError as e:
        raise FingerprintError(
            f"safetensors key {key!r} is not a lora_A/lora_B weight — the "
            f"native fingerprint covers plain <module>.lora_A/lora_B.weight factor pairs "
            f"only ({e})") from e
    if cls != ADAPTER_KEY_LORA:
        raise FingerprintError(
            f"safetensors key {key!r} is a residual-slot payload key, not a LoRA factor — "
            "the LoRA fingerprint covers the factor pairs; the payload has its own "
            "fingerprint (native_residual.residual_fingerprint)")
    return module, which


def is_residual_payload_key(key: str) -> bool:
    """True for a ``sisteer_residual.*`` payload key (native-psr-serving §1.4:
    the residual kinds' bank entries carry the payload BESIDE the rank-1 zero
    LoRA pair; the LoRA readers skip it and the residual readers own it)."""
    from .serving_attestation import ADAPTER_KEY_RESIDUAL, classify_adapter_key

    return classify_adapter_key(key)[0] == ADAPTER_KEY_RESIDUAL


#: adapter_config.json keys the launcher reads for the routed experts (contract §3)
CONFIG_TARGET_PARAMETERS = "target_parameters"
CONFIG_RANK_PATTERN = "rank_pattern"
CONFIG_PETRI_TARGETS = "petri_targets"
CONFIG_PETRI_PER_EXPERT_RANK = "petri_per_expert_rank"


def _peft_param_path(module: str, projection: str) -> str:
    """The HF parameter path PEFT matched a ``rank_pattern`` key against for a
    routed-experts factor: ``base_model.model.<pfx>.layers.N.mlp.experts[.base_layer]``
    -> ``<pfx>.layers.N.mlp.experts.<gate_up_proj|down_proj>`` (PEFT strips
    the ``base_model.model.`` prefix it writes; ``get_pattern_key`` then
    does ``re.match(r"(.*\\.)?(<key>)$", path)``)."""
    base = module[:-len(".base_layer")] if module.endswith(".base_layer") else module
    if base.startswith("base_model.model."):
        base = base[len("base_model.model."):]
    return f"{base}.{projection}"


def _file_num_experts(cfg: Mapping[str, Any], experts: Mapping[str, Tuple[str, torch.Tensor]],
                      adapter_dir: Path) -> int:
    """E, the expert count the flat ``[E*r_e, d]`` factors reshape by, from
    ``adapter_config.json`` — in this precedence, the FIRST source that
    resolves wins (they are not cross-checked against each other on purpose:
    ``petri_targets`` carries E itself with no rank provenance, while the
    rank sources may name the parent object's r_e or the file's physical
    one for a difference adapter): (1) ``petri_targets[*].num_experts`` of
    the ``kind == "experts"`` entries (all must agree); (2) a ``rank_pattern``
    entry matching an experts parameter path by PEFT's own rule -> the
    PHYSICAL r_e (PEFT builds the file's shapes from it), E = A.shape[0] //
    r_e; (3) ``petri_per_expert_rank`` (``{"routed_expert": r_e}`` or an
    int) -> the same division. Every experts factor's rank axis must be a
    multiple of E. None resolving refuses: E cannot be guessed."""
    def _from_rank(r_e: Any, source: str) -> int:
        if isinstance(r_e, bool) or not isinstance(r_e, int) or r_e < 1:
            raise FingerprintError(
                f"{adapter_dir}: {source} gives per-expert rank {r_e!r}, not a positive int")
        Es = set()
        for module, (_, a) in experts.items():
            rows = int(a.shape[0])
            if rows % r_e:
                raise FingerprintError(
                    f"{adapter_dir}: experts module {module!r} lora_A has {rows} rows, not a "
                    f"multiple of the per-expert rank {r_e} from {source}; cannot reshape to "
                    "[E, r_e, d_in]")
            Es.add(rows // r_e)
        if len(Es) != 1:
            raise FingerprintError(
                f"{adapter_dir}: the experts modules imply different expert counts "
                f"{sorted(Es)} at per-expert rank {r_e} ({source}); refusing")
        return Es.pop()

    targets = cfg.get(CONFIG_PETRI_TARGETS)
    if isinstance(targets, Mapping):
        Es = set()
        for name, info in targets.items():
            if isinstance(info, Mapping) and info.get("kind") == "experts":
                E = info.get("num_experts")
                if isinstance(E, bool) or not isinstance(E, int) or E < 1:
                    raise FingerprintError(
                        f"{adapter_dir}: {CONFIG_PETRI_TARGETS}[{name!r}] is an experts target "
                        f"with num_experts={E!r}, not a positive int")
                Es.add(E)
        if len(Es) > 1:
            raise FingerprintError(
                f"{adapter_dir}: {CONFIG_PETRI_TARGETS} experts targets disagree on num_experts "
                f"{sorted(Es)}; refusing")
        if Es:
            E = Es.pop()
            for module, (_, a) in experts.items():
                if int(a.shape[0]) % E:
                    raise FingerprintError(
                        f"{adapter_dir}: experts module {module!r} lora_A has {int(a.shape[0])} "
                        f"rows, not a multiple of num_experts={E} ({CONFIG_PETRI_TARGETS}); "
                        "cannot reshape to [E, r_e, d_in]")
            return E
    pattern = cfg.get(CONFIG_RANK_PATTERN)
    if isinstance(pattern, Mapping) and pattern:
        ranks = set()
        for module, (projection, _) in experts.items():
            path = _peft_param_path(module, projection)
            for pat, r_e in pattern.items():
                try:
                    hit = re.match(rf"(.*\.)?({pat})$", path) is not None
                except re.error as e:
                    raise FingerprintError(
                        f"{adapter_dir}: {CONFIG_RANK_PATTERN} key {pat!r} is not a valid "
                        f"regex ({e})") from e
                if hit:
                    ranks.add(r_e)
                    break
        if len(ranks) > 1:
            raise FingerprintError(
                f"{adapter_dir}: {CONFIG_RANK_PATTERN} gives different per-expert ranks "
                f"{sorted(ranks, key=str)} across the experts modules; refusing")
        if ranks:
            return _from_rank(ranks.pop(), CONFIG_RANK_PATTERN)
    per = cfg.get(CONFIG_PETRI_PER_EXPERT_RANK)
    if isinstance(per, Mapping) and "routed_expert" in per:
        return _from_rank(per["routed_expert"], f"{CONFIG_PETRI_PER_EXPERT_RANK}.routed_expert")
    if isinstance(per, int) and not isinstance(per, bool):
        return _from_rank(per, CONFIG_PETRI_PER_EXPERT_RANK)
    raise FingerprintError(
        f"{adapter_dir}: cannot pin the expert count E — {ADAPTER_CONFIG_FILE} carries none of "
        f"{CONFIG_PETRI_TARGETS}[*].num_experts (kind 'experts'), a {CONFIG_RANK_PATTERN} entry "
        f"matching the experts parameters, or {CONFIG_PETRI_PER_EXPERT_RANK}.routed_expert; the "
        "flat [E*r_e, d] factors cannot be reshaped into vLLM's [E, r_e, d] without E; refusing")


def reshape_flat_experts(a: torch.Tensor, b: torch.Tensor, num_experts: int
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
    """The flat PEFT ``ParamWrapper`` pair -> vLLM's 3-D form, EXACTLY as
    ``_stack_moe_lora_weights`` does it (``model_manager.py:857-880``
    @0.25.1, EP off): ``A [E*r_e, in].reshape(E, -1, in)`` -> ``[E, r_e, in]``;
    ``B [out, E*r_e].reshape(out, -1, E).permute(2, 0, 1)`` -> ``[E, out, r_e]``
    (expert-minor columns — the layout contract §3's exporter writes with
    ``B.permute(1, 2, 0).reshape(out, r_e*E)``). Contiguous, as vLLM's."""
    E = int(num_experts)
    a3 = a.reshape(E, -1, int(a.shape[-1])).contiguous()
    b3 = b.reshape(int(b.shape[0]), -1, E).permute(2, 0, 1).contiguous()
    return a3, b3


def expected_adapter_fingerprint_bf16(adapter_dir: Path
                                      ) -> Tuple[str, Tuple[str, ...]]:
    """THE LAUNCHER SIDE of the contract: the fingerprint the worker MUST
    report for the adapter at ``adapter_dir``, derived independently from
    the files, for EVERY admitted module class:

    * ``adapter_config.json`` declares ``lora_alpha == r`` (scaling 1
      everywhere — vLLM applies ONE ``alpha/r`` to every module) and a
      non-empty ``target_modules`` list of str whose UNIQUE contents equal
      the short HF module names the file's LINEAR factors adapt (PEFT
      semantics: ``["o_proj"]``, ``[..., "gate_proj", "up_proj", "down_proj"]``
      reaching the shared expert; the zero adapter's un-deduplicated
      ``["o_proj", "o_proj", ...]`` is admitted; a list naming a module the
      file does not carry, or missing one it does, refuses).
    * Every safetensors key is a plain ``<module>.lora_A/B.weight`` pair
      (residual ``sisteer_residual.*`` payload keys are skipped — the payload
      has its own fingerprint), fp32 in the file, cast ONCE to bf16 exactly
      as ``LoRAModel.from_lora_tensors`` does, with no scaling. Every module
      name canonicalises (:func:`canonical_module_key`).
    * The routed experts (``...mlp.experts.base_layer`` = gate_up,
      ``...mlp.experts`` = down; contract §3) require ``target_parameters``
      naming both ``…experts.gate_up_proj`` and ``…experts.down_proj``, BOTH
      pairs per layer, and E from the config (:func:`_file_num_experts`);
      the flat factors are reshaped to 3-D exactly as vLLM does
      (:func:`reshape_flat_experts`) so the digest equals the worker's
      over its stacked tensors. Per-module physical ranks come from the
      shapes (the experts' ``r_e = rows // E``; the linear factors' must
      equal ``r``).

    Then the same :func:`adapter_fingerprint_bf16` as the worker, over the
    same logical keys."""
    adapter_dir = Path(adapter_dir)
    cfg = read_adapter_config(adapter_dir)
    r = int(cfg["r"])
    alpha = cfg.get("lora_alpha")
    if (isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or float(alpha) != float(r)):
        raise FingerprintError(
            f"{adapter_dir}: non-unit LoRA scaling lora_alpha={alpha!r} r={r}: "
            "vLLM's LoRALayerWeights.optimize() folds alpha/r into lora_b in "
            "place at load; the fingerprint is defined only at scaling 1")
    targets = cfg.get("target_modules")
    if (not isinstance(targets, list) or not targets
            or not all(isinstance(m, str) for m in targets)):
        raise FingerprintError(
            f"{adapter_dir}: target_modules={targets!r}; the native fingerprint admits "
            "adapters whose target_modules is a non-empty list of str naming the short HF "
            "module names the file's linear factors adapt (PEFT semantics; the zero "
            "adapter's un-deduplicated ['o_proj', 'o_proj', ...] is admitted)")
    weights = adapter_dir / ADAPTER_WEIGHTS_FILE
    if not weights.is_file():
        raise FingerprintError(
            f"{adapter_dir} has no {ADAPTER_WEIGHTS_FILE} — nothing to "
            "fingerprint")
    from safetensors.torch import load_file  # transformers-stack dep

    tensors = load_file(str(weights))
    grouped: Dict[str, Dict[str, torch.Tensor]] = {}
    for key in sorted(tensors):
        if is_residual_payload_key(key):
            continue          # the residual payload has its own fingerprint
        module, which = _split_peft_key(key)
        t = tensors[key]
        if t.dtype is not torch.float32:
            raise FingerprintError(
                f"{adapter_dir}: {key} is {t.dtype}; the admitted exports "
                "write float32 factors and the contract is exactly one "
                "fp32 -> bf16 cast — refusing to fingerprint a file whose "
                "cast history is unknown")
        slot = grouped.setdefault(module, {})
        if which in slot:
            raise FingerprintError(
                f"{adapter_dir}: module {module!r} carries two lora_{which} "
                "tensors")
        slot[which] = t
    if not grouped:
        raise FingerprintError(
            f"{adapter_dir}: {ADAPTER_WEIGHTS_FILE} carries no LoRA factors")
    linear: Dict[str, Dict[str, torch.Tensor]] = {}
    experts: Dict[str, Tuple[str, torch.Tensor]] = {}       # module -> (projection, flat A)
    linear_suffixes = set()
    for module, ab in grouped.items():
        if set(ab) != {"A", "B"}:
            raise FingerprintError(
                f"{adapter_dir}: module {module!r} has lora_{sorted(ab)} "
                "only; the fingerprint needs both lora_A and lora_B")
        key = canonical_module_key(module)
        kind = module_kind_of_key(key)
        if kind == MODULE_KIND_PACKED:
            raise FingerprintError(
                f"{adapter_dir}: module {module!r} names vLLM's packed wrapper "
                f"{key.rsplit('.', 1)[-1]} — a PEFT file carries the components "
                f"({list(PACKED_COMPONENTS[key.rsplit('.', 1)[-1]])}), the engine packs them; "
                "refusing")
        if kind == MODULE_KIND_EXPERTS_PROJECTION:
            experts[module] = (EXPERTS_PROJECTIONS[0], ab["A"])
        elif kind == MODULE_KIND_EXPERTS:
            experts[module] = (EXPERTS_PROJECTIONS[1], ab["A"])
        else:
            linear[module] = ab
            linear_suffixes.add(module.rsplit(".", 1)[-1])
    if set(targets) != linear_suffixes:
        raise FingerprintError(
            f"{adapter_dir}: target_modules={targets!r}; its unique contents must equal the "
            f"short module names the file's linear factors adapt ({sorted(linear_suffixes)}) "
            "— PEFT semantics (the routed-expert pairs are declared by target_parameters, "
            "not target_modules; the zero adapter's un-deduplicated list is admitted); "
            "refusing")
    params = cfg.get(CONFIG_TARGET_PARAMETERS)
    names_experts = (isinstance(params, list)
                     and any(isinstance(t, str) and ".experts." in t + "." for t in params))
    if experts:
        ok = (isinstance(params, list) and all(isinstance(t, str) for t in params)
              and all(any(t.endswith(f"experts.{proj}") for t in params)
                      for proj in EXPERTS_PROJECTIONS))
        if not ok:
            raise FingerprintError(
                f"{adapter_dir}: the file carries routed-expert factors "
                f"({sorted(experts)}) but {CONFIG_TARGET_PARAMETERS}={params!r} does not name "
                f"both …experts.{EXPERTS_PROJECTIONS[0]} and …experts.{EXPERTS_PROJECTIONS[1]} "
                "(contract §3: PEFT reaches the experts' nn.Parameters through "
                "target_parameters); refusing")
        E = _file_num_experts(cfg, experts, adapter_dir)
    elif names_experts:
        raise FingerprintError(
            f"{adapter_dir}: {CONFIG_TARGET_PARAMETERS}={params!r} declares routed-expert "
            "parameters but the file carries no experts factors; refusing")
    modules: Dict[str, ModuleWeights] = {}
    for module, ab in linear.items():
        modules[module] = ModuleWeights(
            lora_a=ab["A"].to(torch.bfloat16),
            lora_b=ab["B"].to(torch.bfloat16),
            rank=r, lora_alpha=float(alpha), is_packed=False)
    for module in experts:
        ab = grouped[module]
        a, b = ab["A"], ab["B"]
        if a.dim() != 2 or b.dim() != 2 or int(a.shape[0]) != int(b.shape[1]):
            raise FingerprintError(
                f"{adapter_dir}: experts module {module!r} lora_A {tuple(a.shape)} / lora_B "
                f"{tuple(b.shape)} is not the flat PEFT layout [E*r_e, d_in] / [d_out, E*r_e] "
                "(2-D, sharing the rank axis); refusing")
        a3, b3 = reshape_flat_experts(a.to(torch.bfloat16), b.to(torch.bfloat16), E)
        modules[module] = ModuleWeights(
            lora_a=a3, lora_b=b3, rank=r, lora_alpha=float(alpha), is_packed=False)
    return adapter_fingerprint_bf16(modules)


def _shape_or_none(t: Any) -> Any:
    if isinstance(t, torch.Tensor):
        return [int(d) for d in t.shape]
    if isinstance(t, (list, tuple)):
        return [_shape_or_none(x) for x in t]
    return None


def _lora_managers_of(model_runner: Any) -> Tuple[Any, Any]:
    """``(model_runner.lora_manager, lora_manager._adapter_manager)`` — the
    two pinned 0.25.1 surfaces every adapter report reads (the
    ``WorkerLoRAManager`` set by ``LoRAModelRunnerMixin.load_lora_model``
    under ``--enable-lora``, and the ``LoRAModelManager`` it wraps, the
    attribute ``worker_manager.py`` itself uses). Any absence refuses by
    name: a state that cannot be read must fail readiness."""
    if model_runner is None:
        raise LensWorkerError(
            "the worker carries no model_runner — not the pinned V1 worker "
            "surface; the loaded adapters cannot be read, so the launch "
            "must not pass readiness")
    lm = getattr(model_runner, "lora_manager", None)
    if lm is None:
        raise LensWorkerError(
            "the model runner carries no lora_manager "
            "(LoRAModelRunnerMixin.load_lora_model sets it under "
            "--enable-lora) — a lora-native server with no LoRA manager has "
            "nothing to attest; the launch must not pass readiness")
    am = getattr(lm, "_adapter_manager", None)
    if am is None:
        raise LensWorkerError(
            "lora_manager carries no _adapter_manager — not the pinned "
            "WorkerLoRAManager surface (vllm/lora/worker_manager.py "
            "@0.25.1); the loaded adapters cannot be read")
    return lm, am


def _execution_registry(am: Any) -> Mapping[str, Any]:
    """``_adapter_manager.modules`` — the ``{module_name: BaseLayerWithLoRA}``
    registry ``LoRAModelManager._create_lora_modules`` fills
    (``register_module``, once per wrapped module; ``model_manager.py``
    @0.25.1), keyed by the EXACT ``named_modules`` name ``activate_adapter``
    later looks each adapter up under. Not a mapping, or empty, refuses by
    name (a lora-native server with no execution layer applies nothing
    under every adapter name)."""
    modules = getattr(am, "modules", None)
    if not isinstance(modules, Mapping):
        raise LensWorkerError(
            f"_adapter_manager.modules is {type(modules).__name__}, not the "
            "pinned LoRAModelManager module registry (vllm/lora/model_manager"
            ".py @0.25.1: {module_name: BaseLayerWithLoRA}) — the modules "
            "carrying LoRA execution cannot be read, so the launch must not "
            "pass readiness")
    if not modules:
        raise LensWorkerError(
            "_adapter_manager.modules is EMPTY — no model module carries a "
            "LoRA execution layer, so every adapter name would serve base; "
            "the launch must not pass readiness")
    return modules


#: the four stacked LoRA buffers of a ``FusedMoE3DWithLoRA`` wrapper
#: (``vllm/lora/layers/fused_moe.py`` @0.25.1: each a tuple of ONE tensor for
#: the 3-D class — ``_w13_slices = 1`` — shaped ``[max_loras, E_local, R, d]``,
#: ``[max_loras, E_local, 2I, R]``, ``[max_loras, E_local, R, I]``,
#: ``[max_loras, E_local, d, R]``, ``R = max_lora_rank``)
MOE_STACKED_ATTRS = ("w13_lora_a_stacked", "w13_lora_b_stacked",
                     "w2_lora_a_stacked", "w2_lora_b_stacked")
#: the stacked LoRA buffers of a ``BaseLinearLayerWithLoRA`` wrapper (a tuple
#: per attribute, one tensor per output slice; ``base_linear.py`` /
#: ``column_parallel_linear.py`` @0.25.1)
LINEAR_STACKED_ATTRS = ("lora_a_stacked", "lora_b_stacked")


def _stacked_tensors(wrapper: Any, attr: str, *, mod: str) -> Optional[List[torch.Tensor]]:
    """``getattr(wrapper, attr)`` as a list of tensors (a tuple / list of
    tensors, or one tensor), ``None`` when the attribute is absent; any
    other value is surface drift refused by name."""
    v = getattr(wrapper, attr, None)
    if v is None:
        return None
    if isinstance(v, torch.Tensor):
        return [v]
    if isinstance(v, (list, tuple)) and v and all(isinstance(t, torch.Tensor) for t in v):
        return list(v)
    raise LensWorkerError(
        f"execution module {mod!r}: {attr} is {type(v).__name__}, not a tuple of tensors — "
        "not the pinned vLLM 0.25.1 LoRA wrapper surface; the LoRA buffers cannot be read")


def _buffer_bytes(tensors: List[torch.Tensor]) -> int:
    return int(sum(int(t.numel()) * int(t.element_size()) for t in tensors))


def _moe_wrapper_surfaces(mod: str, wrapper: Any
                          ) -> Tuple[Dict[str, int], Dict[str, Any], int]:
    """A ``FusedMoE3DWithLoRA`` wrapper's ``(stride record, geometry record,
    LoRA buffer bytes)`` read off its four stacked buffers:
    ``w13_lora_a_stacked[0]`` is ``[max_loras, E, R, d]``,
    ``w13_lora_b_stacked[0]`` ``[max_loras, E, 2I, R]``, ``w2_lora_a_stacked[0]``
    ``[max_loras, E, R, I]``, ``w2_lora_b_stacked[0]`` ``[max_loras, E, d, R]``
    — cross-checked against each other and, where the wrapper exposes them,
    against ``hidden_size`` / ``intermediate_size_per_partition`` /
    ``local_num_experts``. The stride record is the WIDEST input and output
    stride the wrapper's two GEMMs execute (``in = max(d, I)``, ``out =
    max(2I, d)``), so the launcher's ceiling arithmetic bounds both; the
    geometry record carries every shape so the probe can assert the
    expected-bytes law (``110 MiB × R`` per slot at 40 × 256 experts). One
    w13 slice only: two slices is the 2-D ``FusedMoEWithLoRA`` layout, not
    the 3-D one the admitted exports load through."""
    bufs: Dict[str, List[torch.Tensor]] = {}
    for attr in MOE_STACKED_ATTRS:
        got = _stacked_tensors(wrapper, attr, mod=mod)
        if got is None:
            raise LensWorkerError(
                f"execution module {mod!r}: its MoE LoRA wrapper has no {attr} — not the "
                "pinned FusedMoE3DWithLoRA surface (vllm/lora/layers/fused_moe.py @0.25.1); "
                "the executing expert strides cannot be read, so the launch must not pass "
                "readiness")
        bufs[attr] = got
    if any(len(v) != 1 for v in bufs.values()):
        raise LensWorkerError(
            f"execution module {mod!r}: MoE LoRA buffers with "
            f"{[len(v) for v in bufs.values()]} slices — the 3-D FusedMoE3DWithLoRA layout "
            "has exactly one w13 slice (the 2-D FusedMoEWithLoRA has two); refusing")
    w13a, w13b, w2a, w2b = (bufs[a][0] for a in MOE_STACKED_ATTRS)
    for attr, t in zip(MOE_STACKED_ATTRS, (w13a, w13b, w2a, w2b)):
        if t.dim() != 4:
            raise LensWorkerError(
                f"execution module {mod!r}: {attr}[0] has shape {tuple(t.shape)}, expected 4-D "
                "[max_loras, E, ., .]")
    max_loras, E, R, d = (int(x) for x in w13a.shape)
    n_b, E_b, two_I, R_b = (int(x) for x in w13b.shape)
    n_c, E_c, R_c, I = (int(x) for x in w2a.shape)
    n_d, E_d, d_d, R_d = (int(x) for x in w2b.shape)
    if not (max_loras == n_b == n_c == n_d and E == E_b == E_c == E_d
            and R == R_b == R_c == R_d and d == d_d and two_I == 2 * I):
        raise LensWorkerError(
            f"execution module {mod!r}: MoE LoRA buffers disagree — w13_a {tuple(w13a.shape)}, "
            f"w13_b {tuple(w13b.shape)}, w2_a {tuple(w2a.shape)}, w2_b {tuple(w2b.shape)}; "
            "expected [n, E, R, d] / [n, E, 2I, R] / [n, E, R, I] / [n, E, d, R]; refusing")
    if min(max_loras, E, R, d, I) < 1:
        raise LensWorkerError(
            f"execution module {mod!r}: degenerate MoE LoRA buffer shape {tuple(w13a.shape)}")
    for attr, want in (("hidden_size", d), ("intermediate_size_per_partition", I),
                       ("local_num_experts", E)):
        have = getattr(wrapper, attr, None)
        if have is None:
            continue
        if isinstance(have, bool) or not isinstance(have, int) or int(have) != want:
            raise LensWorkerError(
                f"execution module {mod!r}: wrapper.{attr}={have!r} but its stacked LoRA "
                f"buffers say {want}; the surface disagrees with itself, refusing")
    record = {
        "num_experts_local": E, "max_lora_rank": R, "max_loras": max_loras,
        "hidden_size": d, "intermediate_size_per_partition": I,
        "w13_in": d, "w13_out": two_I, "w2_in": I, "w2_out": d,
    }
    for attr in MOE_STACKED_ATTRS:
        record[attr] = [int(x) for x in bufs[attr][0].shape]
    stride = {"in": max(d, I), "out": max(two_I, d)}
    return stride, record, _buffer_bytes([t for v in bufs.values() for t in v])


def lora_execution_modules_report(model_runner: Any) -> Dict[str, Any]:
    """The model modules that actually CARRY a LoRA execution layer on this
    worker — ``lora_manager._adapter_manager.modules``, the ``{module_name:
    BaseLayerWithLoRA}`` registry ``LoRAModelManager._create_lora_modules``
    fills with exactly the modules that passed ``_match_target_modules``
    (``model_manager.py`` @0.25.1: ``register_module`` is called once per
    wrapped module, and ``activate_adapter`` iterates THIS dict to
    ``set_lora`` the weights). It is the execution-side complement of
    :func:`loaded_adapters_report`: that report hashes the CPU-side
    ``LoRAModel.loras`` factors, which ``--lora-target-modules`` does NOT
    filter, so a registered adapter can carry every fitted layer while only
    a subset of layers has a wrapper to apply it. The launcher asserts that
    every fingerprinted logical key of every adapter is CARRIED by a module
    in this set (``serving_attestation.native_lora_report_problems`` via
    ``carried_keys``: a q_proj key by the ``qkv_proj`` wrapper, an
    ``experts.*`` key by the ``experts`` wrapper).

    ``lora_execution_module_keys`` are the canonical EXECUTION keys of every
    RECOGNISED wrapper (``L<N>.self_attn.o_proj``, the packed
    ``L<N>.self_attn.qkv_proj`` / ``L<N>.mlp[.shared_expert].gate_up_proj``,
    the MoE ``L<N>.mlp.experts``; ``serving_attestation.module_key_of``) and
    ``lora_execution_module_names`` their EXACT registry names — the name
    ``activate_adapter`` looks each adapter's factor up under, since
    canonical agreement is not application; wrapped modules the vocabulary
    does NOT recognise (a router ``gate``, the GDN ``in_proj_*``,
    ``conv1d``) ride along by raw name in ``lora_execution_modules_other``
    (the launcher composes ``--lora-target-modules`` to the adapted class,
    so a wrapper outside that class is refused at readiness by
    ``execution_stride_problems``). ``lora_execution_module_strides`` is,
    for EVERY wrapper of every family, ``{"in", "out"}``: a linear
    wrapper's ``input_size`` and ``sum(output_slices)`` off the
    ``BaseLinearLayerWithLoRA`` surface (``base_linear.py`` @0.25.1: one
    slice for a plain projection, the packed widths for ``gate_up_proj`` /
    ``qkv_proj`` — the sum is the expand kernel's ``output_d0_stride``), a
    ``FusedMoE3DWithLoRA`` wrapper's widest GEMM strides off its stacked
    buffers (:func:`_moe_wrapper_surfaces`; it exposes no ``input_size`` /
    ``output_slices``), and ``lora_execution_module_stride_max`` the widest
    of them: the number the launcher holds to the D its int32 kernel index
    ceiling was sized from (a wrapper with neither surface: refused by name
    — GPU-verify). ``lora_execution_moe_wrappers`` carries each MoE
    wrapper's full buffer geometry. ``lora_execution_module_lora_buffer_bytes``
    is Σ ``numel × element_size`` over every wrapper's stacked LoRA buffers
    (the linear wrappers' ``lora_a_stacked`` / ``lora_b_stacked``, the MoE
    wrapper's four) — the probe asserts the expected-bytes law against it;
    ``..._by_module`` is the per-wrapper split and
    ``lora_execution_module_lora_buffer_unread`` names any linear wrapper
    that exposed no stacked buffers (the total covers the read ones only, so
    a consumer trusts it iff this list is empty — never empty on a real
    engine, where ``create_lora_weights`` allocates them). An empty registry
    refuses: a lora-native server with no execution layer applies nothing
    under every adapter name."""
    _, am = _lora_managers_of(model_runner)
    modules = _execution_registry(am)
    keys: List[str] = []
    names: List[str] = []
    other: List[str] = []
    strides: Dict[str, Dict[str, int]] = {}
    moe: Dict[str, Dict[str, Any]] = {}
    nbytes: Dict[str, int] = {}
    unread: List[str] = []
    for name, wrapper in modules.items():
        mod = str(name)
        key = module_key_of(mod)
        if key is not None:
            keys.append(key)
            names.append(mod)
        else:
            other.append(mod)
        if getattr(wrapper, MOE_STACKED_ATTRS[0], None) is not None:
            stride, record, b = _moe_wrapper_surfaces(mod, wrapper)
            strides[mod] = stride
            moe[mod] = record
            nbytes[mod] = b
            continue
        in_f = getattr(wrapper, "input_size", None)
        if isinstance(in_f, bool) or not isinstance(in_f, int) or in_f < 1:
            raise LensWorkerError(
                f"execution module {mod!r}: its LoRA wrapper has no positive int "
                f"input_size (got {in_f!r}) — not the pinned BaseLinearLayerWithLoRA "
                "surface (vllm/lora/layers/base_linear.py @0.25.1) nor the FusedMoE3DWithLoRA "
                "one (no w13_lora_a_stacked); the executing stride cannot be read, so the "
                "launch must not pass readiness")
        slices = getattr(wrapper, "output_slices", None)
        if (not isinstance(slices, (tuple, list)) or not slices
                or any(isinstance(d, bool) or not isinstance(d, int) or d < 1
                       for d in slices)):
            raise LensWorkerError(
                f"execution module {mod!r}: its LoRA wrapper has no non-empty tuple of "
                f"positive int output_slices (got {slices!r}) — not the pinned "
                "BaseLinearLayerWithLoRA surface (vllm/lora/layers/base_linear.py "
                "@0.25.1); the executing output stride cannot be read, so the launch "
                "must not pass readiness")
        strides[mod] = {"in": int(in_f), "out": int(sum(int(d) for d in slices))}
        bufs = [_stacked_tensors(wrapper, attr, mod=mod) for attr in LINEAR_STACKED_ATTRS]
        if any(b is None for b in bufs):
            unread.append(mod)
        else:
            nbytes[mod] = _buffer_bytes([t for b in bufs for t in b])  # type: ignore[union-attr]
    return {
        "lora_execution_module_keys": sorted(keys),
        "lora_execution_module_names": sorted(names),
        "lora_execution_modules_n": len(modules),
        "lora_execution_modules_other": sorted(other),
        "lora_execution_module_strides": dict(sorted(strides.items())),
        "lora_execution_module_stride_max": max(
            max(v["in"], v["out"]) for v in strides.values()),
        "lora_execution_moe_wrappers": dict(sorted(moe.items())),
        "lora_execution_module_lora_buffer_bytes": int(sum(nbytes.values())),
        "lora_execution_module_lora_buffer_bytes_by_module": dict(sorted(nbytes.items())),
        "lora_execution_module_lora_buffer_unread": sorted(unread),
    }


def _execution_bindings(am: Any, lora: Any, loras: Mapping[str, Any]
                        ) -> Dict[str, Optional[str]]:
    """``{execution_module_name: factor_key | None}`` over every RECOGNISED
    module in the execution registry (``serving_attestation.module_key_of``:
    the o_proj / down_proj wrappers, the packed ``qkv_proj`` / ``gate_up_proj``
    wrappers — which bind ONE packed factor for their components — and the
    MoE ``experts`` wrapper; an unrecognised wrapper is outside the class and
    refused elsewhere), resolved by vLLM's OWN lookup —
    ``_adapter_manager._get_lora_layer_weights(lora_model, module_name)``,
    exactly what ``activate_adapter`` calls per registered module
    (``model_manager.py`` @0.25.1; ``LoRAModel.get_lora`` is
    ``self.loras.get(module_name)``, plus the pooling-model ``model.``
    prefix retry inside that method). The returned weights object is
    mapped back to its key in ``loras`` BY IDENTITY, so the report says
    which factor the engine will actually apply on each module, and a
    lookup that returns something not among this adapter's factors is a
    surface drift refused by name. Missing lookup method: refused by name
    (GPU-verify) — the binding cannot be attested otherwise."""
    lookup = getattr(am, "_get_lora_layer_weights", None)
    if not callable(lookup):
        raise LensWorkerError(
            "_adapter_manager has no callable _get_lora_layer_weights — not "
            "the pinned LoRAModelManager surface (vllm/lora/model_manager.py "
            "@0.25.1: the exact-name lookup activate_adapter applies weights "
            "through); which factor each execution module would apply cannot "
            "be read, so the launch must not pass readiness")
    key_of = {id(w): str(name) for name, w in loras.items()}
    out: Dict[str, Optional[str]] = {}
    for name in _execution_registry(am):
        mod = str(name)
        if module_key_of(mod) is None:
            continue
        try:
            w = lookup(lora, mod)
        except Exception as e:  # noqa: BLE001 — surface drift, refused by name
            raise LensWorkerError(
                f"_get_lora_layer_weights(adapter id {getattr(lora, 'id', '?')}, "
                f"{mod!r}) raised {type(e).__name__}: {e} — the exact-name "
                "binding cannot be read") from e
        if w is None:
            out[mod] = None
            continue
        key = key_of.get(id(w))
        if key is None:
            raise LensWorkerError(
                f"_get_lora_layer_weights(adapter id {getattr(lora, 'id', '?')}, "
                f"{mod!r}) returned weights that are not among this adapter's "
                f"{len(loras)} loaded factors — not the pinned lookup "
                "(LoRAModel.get_lora over LoRAModel.loras @0.25.1); refusing")
        out[mod] = key
    return dict(sorted(out.items()))


def loaded_adapters_report(model_runner: Any, *,
                           fingerprint_fn: Callable[
                               [Mapping[str, ModuleWeights]],
                               Tuple[str, Tuple[str, ...]]]
                           = adapter_fingerprint_bf16) -> Dict[str, Any]:
    """Every LoRA adapter this worker has LOADED, fingerprinted: reads
    ``model_runner.lora_manager`` (set by ``LoRAModelRunnerMixin.
    load_lora_model`` under ``--enable-lora``), ``lora_manager.
    list_adapters()`` (ids) and ``lora_manager._adapter_manager.
    get_adapter(id)`` -> ``LoRAModel`` (``id``, ``rank``, ``loras:
    {module_name: LoRALayerWeights}``; it carries NO adapter name — the
    name<->id binding is the frontend's and is asserted by the
    launcher). Adapters are loaded and activated at server start
    (``init_static_loras`` @0.25.1), so the report is non-empty at
    readiness. Raw module names and shapes ride along so a fingerprint
    refusal is diagnosable from the attestation log alone, and
    ``execution_bindings`` (:func:`_execution_bindings`) records, per
    recognised execution module, the factor key vLLM's own exact-name lookup
    selects for it — the fact the launcher holds every fingerprinted
    module to. Every surface read here is GPU-verify: a missing
    attribute refuses by name. ``fingerprint_fn`` is injectable for the
    CPU tests."""
    lm, am = _lora_managers_of(model_runner)
    try:
        ids = sorted(int(i) for i in lm.list_adapters())
    except Exception as e:  # noqa: BLE001 — surface drift, refused by name
        raise LensWorkerError(
            f"lora_manager.list_adapters() could not be read "
            f"({type(e).__name__}: {e})") from e
    entries: List[Dict[str, Any]] = []
    for lora_id in ids:
        lora = am.get_adapter(lora_id)
        if lora is None:
            raise LensWorkerError(
                f"adapter id {lora_id} is listed but _adapter_manager."
                "get_adapter returned None — the loaded set cannot be read")
        for attr in ("rank", "loras"):
            if not hasattr(lora, attr):
                raise LensWorkerError(
                    f"adapter id {lora_id}: LoRAModel has no {attr!r} — not "
                    "the pinned vllm/lora/lora_model.py surface @0.25.1")
        loras = lora.loras
        if not isinstance(loras, Mapping) or not loras:
            raise LensWorkerError(
                f"adapter id {lora_id}: loras is {loras!r} — an adapter with "
                "no modules cannot be attested")
        modules: Dict[str, ModuleWeights] = {}
        raw_shapes: Dict[str, Any] = {}
        dtypes = set()
        for name, w in loras.items():
            for attr in ("lora_a", "lora_b", "rank", "lora_alpha",
                         "is_packed"):
                if not hasattr(w, attr):
                    raise LensWorkerError(
                        f"adapter id {lora_id} module {name!r}: "
                        f"LoRALayerWeights has no {attr!r} — not the pinned "
                        "vllm/lora/lora_weights.py surface @0.25.1")
            modules[str(name)] = ModuleWeights(
                lora_a=w.lora_a, lora_b=w.lora_b, rank=w.rank,
                lora_alpha=w.lora_alpha, is_packed=bool(w.is_packed),
                # PackedLoRALayerWeights.lora_alphas: the per-component alphas of a
                # fused q/k/v or gate/up entry (absent on a plain LoRALayerWeights)
                lora_alphas=getattr(w, "lora_alphas", None))
            raw_shapes[str(name)] = {"lora_a": _shape_or_none(w.lora_a),
                                     "lora_b": _shape_or_none(w.lora_b)}
            for t in (w.lora_a, w.lora_b):
                for x in (t if isinstance(t, (list, tuple)) else (t,)):
                    if isinstance(x, torch.Tensor):
                        dtypes.add(str(x.dtype))
        fingerprint, keys = fingerprint_fn(modules)
        if len(dtypes) != 1:
            raise LensWorkerError(
                f"adapter id {lora_id}: tensor dtypes {sorted(dtypes)} — "
                "expected exactly one")
        # the RESIDUAL identity beside the carrier's: the payload
        # ``native_residual``'s loader patch attached to the LoRAModel, None for a plain LoRA
        residual = getattr(lora, "sisteer_residual", None)
        entries.append({
            "id": int(lora_id),
            "rank": int(lora.rank),
            "n_modules": len(keys),
            "module_keys": list(keys),
            "raw_module_names": sorted(str(n) for n in loras),
            "raw_shapes": raw_shapes,
            "tensor_dtype": dtypes.pop(),
            "fingerprint_bf16": fingerprint,
            "residual_fingerprint": getattr(residual, "fingerprint", None),
            "execution_bindings": _execution_bindings(am, lora, loras),
        })
    return {"lora_loaded_count": len(entries), "lora_adapters": entries}


# --- the residual slot ---------------------------------


def residual_slot_report(model_runner: Any) -> Dict[str, Any]:
    """THE RESIDUAL SLOT's readiness facts (``native_residual``): ``armed`` and
    ``op_name`` (``vllm.sisteer_residual_slot`` when armed), the ``installs``
    count, ``patched_classes`` (per decoder class: ``forward`` / ``__init__``
    ``wrapped`` or ``inherits:<parent>``, plus ``*_wrapped`` bools),
    ``machinery`` (the LoRA-path methods), ``n_layers_hooked`` (distinct
    decoder blocks that constructed with a boundary and a wrapped forward) and
    ``hooked_boundaries``, ``op_registered``, ``stage2_position_classes``; then
    the bank: ``bank`` (geometry + counters, ``None`` when no LoRA manager has
    been constructed), ``slots`` ({``lora_int_id``: {slot, kind, op, layers,
    boundaries, mode, header_len, skip_bos, coeff, fingerprint}}),
    ``n_layers_model`` (the bank's boundary count), ``runner_bound`` (the bank's
    runner IS ``model_runner``), ``ledger_size`` / ``ledger_fault`` /
    ``ledger_cap``; and the STAGE-2 facts:
    ``header_ids`` / ``header_len`` (the ONE assistant header every
    restricted-mode payload named, cross-checked at activation; ``None`` when
    no restricted slot has loaded), ``restricted_slots``, ``fact_table``
    ({size, cap, overflow, admitted, overflowed, reclaimed, off_contract, warmup} of the live-request
    fact table — ``reclaimed`` counts the facts released for FINISHED requests,
    the healthy lifecycle a readiness probe of S requests drives to S and never
    a refusal; ``overflowed`` counts the requests refused at the cap and
    ``overflow`` latches on the first — a readiness refusal; ``off_contract`` counts
    admitted ids without the pinned ``-<8 hex>`` engine suffix — a readiness
    refusal too: the proxy could never claim a failure under such an id), ``n_faults`` / ``faults``
    (runner-state disagreements and missing facts, verbatim up to
    ``MAX_FAULTS_KEPT``; a non-zero count is a readiness refusal),
    ``position_modes`` (the served vocabulary with its slot codes) and the
    runner wrappers under ``machinery`` (``<module>.GPUModelRunner.execute_model``
    for both runners, ``LoRAModelRunnerMixin.set_active_loras``,
    ``LoraState.make_lora_inputs``). Facts only, never a refusal: an unarmed
    worker reports ``armed: False`` with an empty slot table, and the
    launcher's validator (``serving_attestation.residual_slot_problems``, step
    3) decides whether that matches what it composed."""
    from sisteer.capeval.serving import native_residual as nr

    rep = nr.PATCH_REPORT
    out: Dict[str, Any] = {
        "armed": bool(rep.armed), "op_name": rep.op_name, "installs": int(rep.installs),
        "patched_classes": {k: dict(v) for k, v in rep.patched_classes.items()},
        "machinery": dict(rep.machinery),
        "n_layers_hooked": len(rep.hooked_boundaries()),
        "hooked_boundaries": rep.hooked_boundaries(),
        "op_registered": nr.op_registered(),
        "stage2_position_classes": bool(nr.STAGE2_POSITION_CLASSES),
        "position_modes": dict(nr.MODE_CODES),
        "error": rep.error,
    }
    b = nr.bank()
    if b is None:
        out.update(bank=None, slots={}, n_layers_model=None, runner_bound=False,
                   ledger_size=0, ledger_fault=False, ledger_cap=nr.REQUEST_FAILURES_CAP,
                   header_ids=None, header_len=None, restricted_slots=0, fact_table=None,
                   n_faults=0, faults=[])
        return out
    bank_report = b.report()
    out.update(bank=bank_report, slots=b.slots_report(), n_layers_model=int(b.n_boundaries),
               runner_bound=(model_runner is not None and b.runner is model_runner),
               **b.ledger.report(),
               header_ids=bank_report["header_ids"], header_len=bank_report["header_len"],
               restricted_slots=bank_report["restricted_slots"],
               fact_table=bank_report["fact_table"],
               n_faults=bank_report["n_faults"], faults=bank_report["faults"])
    return out


# --- which GPU model runner this worker constructed (canary 489533) ----


def model_runner_class_name(model_runner: Any) -> str:
    cls = type(model_runner)
    return f"{cls.__module__}.{cls.__qualname__}"


def model_runner_report(model_runner: Any) -> Dict[str, Any]:
    """``model_runner_class`` (module + qualname) and ``model_runner_kind`` —
    ``v1`` / ``v2`` per ``serving_attestation.MODEL_RUNNER_CLASSES``, else
    ``unknown``. GPU-VERIFY: ``gpu_worker.py:384-398`` @0.25.1 constructs one
    of exactly two classes off ``vllm_config.use_v2_model_runner``; canary #4
    (job 489533) ran the V2 one, whose state lives in ``req_states`` /
    ``lora_state`` rather than an ``input_batch``. The report never refuses
    on the kind (the launcher's validator does, by name); a missing runner
    refuses here."""
    if model_runner is None:
        raise LensWorkerError(
            "the worker carries no model_runner — not the pinned V1 worker surface; "
            "which model runner serves cannot be attested")
    name = model_runner_class_name(model_runner)
    return {"model_runner_class": name,
            "model_runner_kind": MODEL_RUNNER_CLASSES.get(name, "unknown")}


# --- the per-request LoRA ROUTING RECORDER (gemma canaries 489308, 489533) -------

#: the recorder's request-id memory bound: past it ``overflow`` is set and the
#: per-step list keeps growing to its own bound (the control disarms right
#: after its batches, so a real run never approaches either; a run that does
#: is refused by the check, never silently truncated).
LORA_ROUTING_RECORDER_CAP = 50_000
#: the per-step scheduled-request-set list's bound (``steps_overflow``)
LORA_ROUTING_STEPS_CAP = 50_000
#: how many per-step read failures / conflicting bindings are kept verbatim
_LORA_ROUTING_MAX_ERRORS = 8


#: the EXPECTED operational refusals of the three recorder RPCs — returned to
#: the client as PAYLOADS, never raised out of the extension method: the
#: pinned 0.25.1 ``MultiprocExecutor.collective_rpc`` reads the per-worker
#: response queues sequentially and raises on the FIRST worker FAILURE
#: without draining the later workers' replies, and the queues carry no RPC
#: identifier — so an exception from worker 0 leaves worker 1's reply queued,
#: and the OWNER's next valid call reads that stale reply as its own. A
#: refusal that rides as a SUCCESS payload keeps every queue aligned; the
#: CLIENT reader collects every worker's payload and then refuses. Unexpected
#: internal errors (a bug, a surface drift) still raise.
LORA_ROUTING_REFUSALS = ("already_armed", "epoch_mismatch", "bad_epoch", "not_armed")


class LoraRoutingRefusal(LensWorkerError):
    """An EXPECTED refusal of a recorder operation (:data:`LORA_ROUTING_REFUSALS`),
    raised inside the recorder and converted to a payload by the extension
    method (never escaping it); ``str()`` keeps the ``<refusal>: <detail>``
    text so in-process callers read the same words the client will."""

    def __init__(self, refusal: str, detail: str, *, live_epoch: Optional[str] = None):
        super().__init__(f"{refusal}: {detail}")
        self.refusal = refusal
        self.detail = detail
        self.live_epoch = live_epoch

    def payload(self, *, runner_kind: Optional[str] = None,
                runner_class: Optional[str] = None) -> Dict[str, Any]:
        return {"ok": False, "refusal": self.refusal, "detail": self.detail,
                "live_epoch": self.live_epoch, "runner_kind": runner_kind,
                "runner_class": runner_class}


def decode_epoch_arg(epoch: Any) -> str:
    """The ``epoch`` argument of the arm / observe / disarm RPCs as it arrives
    over the pinned release's ``/collective_rpc`` route (string args only): a
    non-empty string, else refused by name (``bad_epoch``) — an unnamed epoch
    must never read as 'the current one'."""
    if not isinstance(epoch, str) or not epoch:
        raise LoraRoutingRefusal(
            "bad_epoch",
            f"epoch must be a non-empty string (the id the control minted), got "
            f"{type(epoch).__name__} — refusing an unnamed arm/observations/disarm request")
    return epoch


def _lora_binding(lora_request: Any) -> Tuple[int, Optional[str]]:
    """``(lora_int_id, lora_name)`` off a ``LoRARequest`` (``vllm/lora/request.py``
    @0.25.1: ``lora_name: str``, ``lora_int_id: int`` >= 1), ``(0, None)`` for
    no LoRA."""
    if lora_request is None:
        return 0, None
    lid = getattr(lora_request, "lora_int_id", None)
    name = getattr(lora_request, "lora_name", None)
    return (int(lid) if isinstance(lid, int) and not isinstance(lid, bool) else -1,
            str(name) if isinstance(name, str) else None)


class LoraRoutingRecorder:
    """Records, per ENGINE request id, the LoRA the engine bound the request to
    — the structural evidence behind the lora-native standup's
    ``selection_honored`` control — one EPOCH at a time.

    WHY (gemma canary #3, job 489308): the control used to compare each
    mixed-batch row's OUTPUT with a same-size single-selection reference row;
    the standup submits its rows as N concurrent HTTP requests admitted as
    they arrive, so two submissions of "the same N requests" are never the
    same step-by-step computation at BI=0 — the zero adapter differed from
    base on one prompt of a 33-row batch while agreeing bit for bit one
    request at a time. The verdict is the engine's own bookkeeping instead.

    WHAT IT READS (GPU-VERIFY — every surface read out of the pinned
    vllm 0.25.1 tree, pinned here by V1- and V2-shaped fake runners; a live
    drift is recorded per step as a read failure, never raised into the
    engine step, and the check refuses on any recorded failure):

    * ``model_runner.execute_model(scheduler_output, intermediate_tensors=
      None)`` — the per-step forward of BOTH runners (V1
      ``v1/worker/gpu_model_runner.py:4070``, V2
      ``v1/worker/gpu/model_runner.py:1114``; the worker calls it through the
      INSTANCE, ``gpu_worker.py:1015``, so an instance attribute wraps it).
    * PRIMARY, runner-agnostic, read BEFORE the original call off the
      ``scheduler_output`` both runners consume (``v1/core/sched/output.py``):
      ``scheduled_new_reqs: list[NewRequestData]`` (``.req_id: str`` :32,
      ``.lora_request: LoRARequest | None`` :39) binds every request admitted
      this step on first sight; ``num_scheduled_tokens: dict[str, int]`` (:193)
      is exactly the set of requests scheduled THIS step — appended to the
      epoch's step list, from which the check computes "mixed" over the
      control's OWN requests (canary #4: a recorder-side mixed
      count could be inherited from a neighbour's traffic).
    * SECONDARY cross-check AFTER the call, dispatched on the runner's class
      (``serving_attestation.MODEL_RUNNER_CLASSES``): V1 -> ``input_batch``
      (re-read every step: ``gpu_model_runner.py:7056`` re-creates it;
      ``num_reqs`` :1087, ``req_ids`` :305 with ``None`` for a freed slot,
      ``request_lora_mapping`` :246, ``lora_id_to_lora_request`` :248); V2 ->
      ``req_states.req_id_to_index`` (``gpu/states.py:26``) with
      ``lora_state.lora_ids[idx]`` (``gpu/lora_utils.py:74,84``; ``NO_LORA_ID
      = 0`` :17) and ``lora_state.lora_requests.get(rid).lora_name`` (:77,:83)
      — V2's ``execute_model`` runs ``finish_requests`` then ``add_requests``
      inside the step (:1125-1127, :745, :766-795), so after the call the
      state holds this step's admissions minus its finishes. Any disagreement
      with the primary binding goes under ``inconsistent``; an UNKNOWN runner
      class gets no secondary and is recorded as ``runner_kind = "unknown"``,
      which the check refuses (an unpinned engine).

    EPOCHS: the CLIENT mints the epoch id (uuid4 hex) and passes it to
    ``arm(epoch)``; every worker ADOPTS it and RESETS every record.
    Arming the epoch that is already live is idempotent; arming while a
    DIFFERENT epoch is live refuses naming it (``already_armed``);
    ``observations_payload(epoch)`` / ``disarm(epoch)`` refuse an epoch that
    is not the live one (``epoch_mismatch``), so a control can never read a
    neighbour's — or an earlier control's — evidence as its own. The client
    mints because a PER-WORKER mint (the ceba8874 shape) made a
    multi-worker launch's arm payloads disagree: the launcher's reader
    refused before the control's try/finally, and every worker stayed armed
    with no single epoch able to disarm them.

    WHAT IT RECORDS: ``engine_rid -> {lora_int_id, lora_name}`` on first sight
    (the engine id wears ``-<8 hex>`` after the client's id — the check joins
    them under ``serving_attestation.is_engine_id_of``); a later different
    binding goes under ``inconsistent`` and never overwrites; the per-step
    scheduled sets; memory bounds (:data:`LORA_ROUTING_RECORDER_CAP`,
    :data:`LORA_ROUTING_STEPS_CAP`) -> ``overflow`` / ``steps_overflow``."""

    def __init__(self, model_runner: Any, *, cap: int = LORA_ROUTING_RECORDER_CAP,
                 steps_cap: int = LORA_ROUTING_STEPS_CAP):
        if model_runner is None:
            raise LensWorkerError(
                "the worker carries no model_runner — not the pinned V1 worker surface; "
                "the LoRA routing recorder has nothing to wrap")
        self._runner = model_runner
        self._cap = int(cap)
        self._steps_cap = int(steps_cap)
        self._original: Any = None
        self._wrapper: Any = None
        self._original_was_instance_attr = False
        self.runner_class = model_runner_class_name(model_runner)
        self.runner_kind = MODEL_RUNNER_CLASSES.get(self.runner_class, "unknown")
        self.armed = False
        self.epoch: Optional[str] = None
        self._reset()

    def _reset(self) -> None:
        self.observations: Dict[str, Dict[str, Any]] = {}
        self.inconsistent: Dict[str, List[Dict[str, Any]]] = {}
        self.steps: List[List[str]] = []
        self.n_steps = 0
        self.steps_overflow = False
        self.overflow = False
        self.n_snapshot_errors = 0
        self.errors: List[str] = []

    # -- arm / disarm ------------------------------------------------------------

    def arm(self, epoch: str) -> Dict[str, Any]:
        """Adopt the CLIENT's epoch id: idempotent when it is already the live
        one; refuse while a DIFFERENT epoch is armed; else reset every record,
        take the epoch and wrap ``execute_model``."""
        epoch = decode_epoch_arg(epoch)
        if self.armed:
            if epoch == self.epoch:
                return {"ok": True, "armed": True, "epoch": self.epoch, "already_armed": True,
                        "runner_class": self.runner_class, "runner_kind": self.runner_kind}
            raise LoraRoutingRefusal(
                "already_armed",
                f"the LoRA routing recorder is armed by epoch {self.epoch} — disarm it first; "
                "two controls may not share one epoch's evidence", live_epoch=self.epoch)
        original = getattr(self._runner, "execute_model", None)
        if not callable(original):
            raise LensWorkerError(
                "model_runner has no callable execute_model — not the pinned V1/V2 GPU model "
                "runner surface @0.25.1; the routing recorder cannot arm")
        self._reset()
        self.epoch = epoch
        self._original_was_instance_attr = "execute_model" in vars(self._runner)
        self._original = original
        recorder = self

        def sisteer_lora_routing_wrapped_execute_model(*args: Any, **kwargs: Any) -> Any:
            recorder._pre_step(args, kwargs)
            out = original(*args, **kwargs)
            recorder._post_step()
            return out

        sisteer_lora_routing_wrapped_execute_model.__wrapped__ = original  # type: ignore[attr-defined]
        self._wrapper = sisteer_lora_routing_wrapped_execute_model
        self._runner.execute_model = self._wrapper
        self.armed = True
        return {"ok": True, "armed": True, "epoch": self.epoch, "already_armed": False,
                "runner_class": self.runner_class, "runner_kind": self.runner_kind}

    def _check_epoch(self, epoch: str, what: str) -> None:
        if self.epoch is None or epoch != self.epoch:
            raise LoraRoutingRefusal(
                "epoch_mismatch",
                f"{what} asked for epoch {epoch!r} but the recorder's live epoch is "
                f"{self.epoch!r} — another control's (or no) evidence; refusing",
                live_epoch=self.epoch)

    def disarm(self, epoch: str) -> Dict[str, Any]:
        """Restore the original ``execute_model`` for THIS epoch (idempotent for
        the same epoch); returns the final summary (no per-request map)."""
        self._check_epoch(epoch, "disarm")
        was_armed = self.armed
        restored = False
        if self.armed:
            inst = vars(self._runner)
            if inst.get("execute_model") is self._wrapper:
                if self._original_was_instance_attr:
                    inst["execute_model"] = self._original
                else:
                    del inst["execute_model"]
                restored = True
            else:
                # someone re-bound execute_model after we armed: leave THEIR binding
                # alone and say so rather than clobbering it
                self._note_error("disarm: execute_model is no longer this recorder's wrapper; "
                                 "left the current binding in place")
            self.armed = False
        return {**self.summary(), "disarmed": True, "was_armed": was_armed,
                "restored": restored}

    # -- the per-step reads -------------------------------------------------------

    def _note_error(self, text: str) -> None:
        self.n_snapshot_errors += 1
        if len(self.errors) < _LORA_ROUTING_MAX_ERRORS:
            self.errors.append(text[:300])

    def _pre_step(self, args: Tuple[Any, ...], kwargs: Mapping[str, Any]) -> None:
        """PRIMARY: bind this step's new requests and record its scheduled set
        off the ``scheduler_output`` argument. NEVER raises into the step."""
        try:
            so = kwargs["scheduler_output"] if "scheduler_output" in kwargs else args[0]
            self.record_scheduler_output(so)
        except Exception as e:  # noqa: BLE001 — recorded, the step must complete
            self._note_error(f"pre-step: {type(e).__name__}: {e}")

    def record_scheduler_output(self, scheduler_output: Any) -> None:
        """One step's primary evidence (public for the tests)."""
        for nr in list(scheduler_output.scheduled_new_reqs):
            lid, name = _lora_binding(getattr(nr, "lora_request", None))
            self._record(str(nr.req_id), lid, name)
        scheduled = scheduler_output.num_scheduled_tokens
        ids = sorted(str(r) for r in scheduled.keys())
        self.n_steps += 1
        if len(self.steps) >= self._steps_cap:
            self.steps_overflow = True
        else:
            self.steps.append(ids)

    def _post_step(self) -> None:
        """SECONDARY: cross-check the runner's own state by kind. NEVER raises."""
        try:
            if self.runner_kind == "v1":
                self.snapshot_input_batch(self._runner.input_batch)
            elif self.runner_kind == "v2":
                self.snapshot_v2_state(self._runner.req_states, self._runner.lora_state)
            # unknown: no secondary; the check refuses on runner_kind
        except Exception as e:  # noqa: BLE001 — recorded, the step must complete
            self._note_error(f"post-step ({self.runner_kind}): {type(e).__name__}: {e}")

    def snapshot_input_batch(self, ib: Any) -> None:
        """V1: one step's bookkeeping off an ``InputBatch``."""
        n = int(ib.num_reqs)
        req_ids = ib.req_ids
        mapping = ib.request_lora_mapping
        id_to_req = ib.lora_id_to_lora_request
        for idx in range(n):
            rid = req_ids[idx] if idx < len(req_ids) else None
            if rid is None:
                continue
            lora_id = int(mapping[idx])
            name: Optional[str] = None
            if lora_id != 0:
                lreq = id_to_req.get(lora_id) if isinstance(id_to_req, Mapping) else None
                lname = getattr(lreq, "lora_name", None) if lreq is not None else None
                name = str(lname) if isinstance(lname, str) else None
            self._record(str(rid), lora_id, name)

    def snapshot_v2_state(self, req_states: Any, lora_state: Any) -> None:
        """V2: one step's bookkeeping off ``RequestState`` + ``LoraState``."""
        lora_ids = lora_state.lora_ids
        lora_requests = lora_state.lora_requests
        for rid, idx in list(req_states.req_id_to_index.items()):
            lora_id = int(lora_ids[int(idx)])
            lreq = lora_requests.get(rid) if isinstance(lora_requests, Mapping) else None
            lname = getattr(lreq, "lora_name", None) if lreq is not None else None
            name = str(lname) if isinstance(lname, str) else None
            self._record(str(rid), lora_id, name)

    def _record(self, rid: str, lora_id: int, name: Optional[str]) -> None:
        prev = self.observations.get(rid)
        if prev is None:
            if len(self.observations) >= self._cap:
                self.overflow = True
                return
            self.observations[rid] = {"lora_int_id": lora_id, "lora_name": name}
            return
        if (prev["lora_int_id"], prev["lora_name"]) != (lora_id, name):
            seen = self.inconsistent.setdefault(rid, [dict(prev)])
            entry = {"lora_int_id": lora_id, "lora_name": name}
            if entry not in seen and len(seen) < _LORA_ROUTING_MAX_ERRORS:
                seen.append(entry)

    # -- the payloads -----------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        return {"ok": True, "armed": self.armed, "epoch": self.epoch, "runner_class": self.runner_class,
                "runner_kind": self.runner_kind, "n_steps": self.n_steps,
                "steps_overflow": self.steps_overflow, "overflow": self.overflow,
                "n_observed": len(self.observations), "n_inconsistent": len(self.inconsistent),
                "snapshot_errors": self.n_snapshot_errors, "errors": list(self.errors),
                "cap": self._cap, "steps_cap": self._steps_cap}

    def observations_payload(self, epoch: str) -> Dict[str, Any]:
        """JSON-able: THE WHOLE EPOCH — ``observations`` (engine id -> binding),
        ``inconsistent``, ``steps`` (one sorted engine-id list per step), the
        counters and the recorder's own health. The join to the control's
        external ids happens in the check."""
        self._check_epoch(epoch, "observations")
        return {**self.summary(),
                "observations": {r: dict(v) for r, v in self.observations.items()},
                "inconsistent": {r: [dict(e) for e in v] for r, v in self.inconsistent.items()},
                "steps": [list(s) for s in self.steps]}
