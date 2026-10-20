"""The RESIDUAL SLOT: PSR's gated write and the plain steering-vector add at the
decoder block OUTPUT, served as one more adapter slot of the native (plugin-free,
CUDA-graphs-ON) vLLM multi-LoRA engine.

WHAT IT IS. A fitted PSR is, per decoder block l, ``z <- z + relu(w_l . z + b_l)
* d_l`` on the block-output residual stream z; a plain steering vector is
``z <- z + v_l``. Neither can ride vLLM's punica LoRA kernels (no bias, no
nonlinearity, an o_proj INPUT site), so the arithmetic is its OWN custom
op, ``torch.ops.vllm.sisteer_residual_slot`` (fact 12: registered into vLLM's
own library, so that fully qualified name is the ONE name everywhere), called
from a wrapper around each decoder layer's ``forward``. What it BORROWS from the
LoRA machinery is the routing: every residual cell is exported as a rank-1 ZERO
o_proj LoRA whose ``adapter_model.safetensors`` also carries the payload under
the ``sisteer_residual.`` prefix (``lora_export`` step 2), so vLLM allocates it a
slot, a ``lora_int_id``, a ``/v1/models`` card and the per-token adapter index;
this module fills a static per-slot bank from the payload at activation and
reads the per-token slot index the LoRA path computes.

THE FOUR CLASS PATCHES (installed at IMPORT when the launch env carries
``SISTEER_NATIVE_RESIDUAL_SLOT=1`` — the composer sets it iff a served adapter
carries a payload, so a LoRA-only standup's graph is byte-for-byte the certified
one; the import happens in the worker process through ``--worker-extension-cls``
BEFORE ``load_model``, fact 6):

1. ``LoRAModel.from_local_checkpoint`` — adds ``"sisteer_residual."`` to
   ``skip_prefixes`` (the loader's own kwarg, honoured in both the unexpected-
   module check and the tensor walk, ``lora/lora_model.py`` @0.25.1), calls the
   original, then reads the payload + ``petri_residual_slot`` config block off
   the same files (:func:`read_residual_payload`) and attaches it as
   ``lora_model.sisteer_residual`` (``None`` for a plain LoRA / the capture
   dummies).
2. ``LoRAModelManager.__init__`` — creates the :class:`ResidualSlotBank`
   (published as the module global ``_BANK``); ``activate_adapter`` copies the
   payload into the slot vLLM chose (``lora_index_to_id.index(lora_id)``) AFTER
   the original; ``_deactivate_adapter`` captures the slot BEFORE the original
   erases it and zeroes it (idempotent: the LRU ``_on_remove`` callback, a
   repeated deactivate, and ``remove_all_adapters`` — which resets
   ``lora_index_to_id`` wholesale, so THAT wrapper zeroes every slot first).
3. ``PunicaWrapperBase._update_base_metadata`` — after the original, the
   DEDICATED ``slot_by_token`` buffer is refilled: ``fill_(-1)`` then the first
   ``n`` entries copied from ``_token_lora_indices`` — on EVERY mapping change,
   base-only batches included (vLLM's own kernel metadata returns
   before copying the mapping when every token is unadapted, so it is STALE on a
   base-only batch and would steer ``none``). Pad rows (beyond ``n``) and base
   tokens read -1 by construction.
4. Decoder-layer ``forward`` — wrapped ONLY on classes that DEFINE ``forward``
   in their own ``__dict__`` (``Qwen3_5DecoderLayer`` defines only
   ``__init__`` and inherits the parent's forward — wrapping "both" wrote the
   PSR twice per block). Every original is snapshotted BEFORE any wrapper is
   installed, every wrapper carries ``__sisteer_wrapped__``, and a second
   install is a no-op. The wrapper calls the original, then
   ``hidden = op(hidden, residual_or_none, positions, boundary)``; the residual
   passes through untouched. ``boundary`` is bound at CONSTRUCTION by a patched
   ``__init__``: the instance's own ``layer_idx`` where the class sets one
   (gemma4, Qwen3Next, Qwen3.5), the ``layer_idx`` argument or
   ``extract_layer_index(prefix)`` otherwise (Llama; NemotronH — whose four
   layer classes store NO ``layer_idx``, verified on the wheel) — all
   present sources must agree; neither present -> the first forward refuses
   naming the class.
5. ``LoRAModelRunnerMixin.load_lora_model`` — binds the runner to the bank AFTER
   the original returns (fact 11: the runner is constructed in ``init_device``
   BEFORE the LoRA manager exists inside ``load_model``).

THE ARITHMETIC (:func:`residual_slot_forward`, the op's eager impl and the CPU
reference; the lens's write-back reproduced — ``lens_worker._apply_slots_block``
``delta = applied - true; new_hidden = hidden + delta`` with ``true = hidden +
residual`` in the model dtype)::

    idx   = slot_by_token[:T]; live = idx >= 0; idx_c = idx.clamp(min=0)
    sel   = live & active[idx_c, l] [& ~(skip_bos[idx_c] & positions == 0)]
    true  = hidden if residual is None else hidden + residual        # model dtype
    g     = relu((true.float() * gate_w[idx_c, l]).sum(-1) + gate_b[idx_c, l])
    add32 = (g * sel)[:, None] * direction[idx_c, l]                 # fp32
    applied = where(gated[idx_c], true + add32.to(model), true + vector[idx_c, l])
    delta = applied - true;  hidden' = where(sel, hidden + delta, hidden)

Unsteered rows are BIT-identical to the unpatched pair (the ``noop_certificate``
control). Steered rows reproduce the lens form exactly GIVEN the gate: the gate
itself is under a NUMERICAL CONTRACT, not an exactness claim (fact 14): a
batched/gathered fp32 gate is not bit-identical to the references' per-vector
matvec ``x.float() @ w`` — ``|g - g_ref| <= 4 * d * eps32 * ||x|| * ||w||`` per
row (:func:`gate_contract_bound`), the measured maximum reported by the tests,
and the bf16 rounding of ``add.to(model)`` can flip where the two gates differ
(the flip rate is measured, never assumed). ``-1`` never indexes a slot (an
unclamped gather would wrap to slot S-1 and steer ``none``).

THE PAIRED WRITE (relu-differenced-native-qwen; kind ``psr_paired_write``, op
``relu_gated_vector_pair``): ``extraction/paired_write.PSRPairedWriteObject``'s
``V(z) = relu(w1.z + b1) d1 - relu(w2.z + b2) d2`` with BOTH gates reading the
SAME pre-write state ``true`` (``application_order = shared_pre_write_state``),
served as a SECOND GATE TERM of the same slot: the payload carries the ``neg``
side under ``gate_w2 / gate_b2 / direction2`` beside the ``pos`` side, both
parents' object shas and the two terms' signed coefficients ``(+1, -1)`` in the
block (so ``residual_fingerprint`` covers both sides and both parents), the bank
allocates ``gate_w2 / gate_b2 / direction2 / paired`` ONLY when the launch env
carries ``SISTEER_NATIVE_RESIDUAL_SLOT_PAIRED=1`` (the composer sets it iff a
served payload is the paired op; a paired payload on an unarmed bank REFUSES),
and the op body forms the fp32 difference and casts it ONCE — one read, one
write. On an unarmed bank the body is the single-gate one verbatim; on an armed
bank an unpaired slot's rows are selected by ``torch.where`` and keep their
bits (:func:`_residual_slot_forward_rows` says why the where-form and not the
zero-row form). The lens has no adapter for the kind; only this slot serves it.

THE CONSTANT KINDS fold the object's baked ``coeff`` (``static_r``'s
``base_scale``; 1.0 for ``vector`` / ``vector_all`` / ``diffmean``) into the
model-dtype ``vector`` buffer at load in the lens's order (``addition = vec *
eff`` on the model-dtype vector, ``lens_adapters._apply_one`` OP_ADD_VECTOR), so
the per-token add is ``true + addition`` exactly as the lens writes it.

THE POSITION CLASS (stage 2). A slot serves ONE of the
``POSITION_MODES`` (``all`` / ``response`` / ``prompt_last`` / ``prefill_all``,
an int code per slot in ``mode[S]``); the three restricted modes need two
PER-REQUEST facts the op cannot see — the prompt length and the response
boundary — which three more wrappers carry into two per-token buffers beside
``slot_by_token``:

6. ``execute_model(scheduler_output)`` on BOTH GPU model runners
   (``serving_attestation.MODEL_RUNNER_CLASSES``; the routing recorder's
   precedent, which wraps the INSTANCE and composes with this class-level
   wrapper): BEFORE the original, every ``scheduled_new_reqs`` entry yields
   ``prompt_len = len(prompt_token_ids)`` and ``response_start = rfind(
   prompt_token_ids, header_ids)`` — the LAST header occurrence, the lens's
   declared semantics (``lens_adapters.py`` THE RESPONSE MASK;
   ``span_locator.rfind``): header + any seed + every generated row —
   into the request FACT TABLE (:class:`RequestFactTable`, bounded like the
   recorder: a cap and an ``overflow`` fact); ``finished_req_ids`` drop from
   the fact table only. ``header_ids`` are the served payloads'
   ``response_header_ids``, cross-checked EQUAL at slot activation (a
   disagreement refuses the activation; a ``BOUNDARY_MODES`` payload without
   them refuses — the exporter derives them for the ``response`` mode only,
   and ``prompt_last`` / ``prefill_all`` need no boundary). A header that
   occurs nowhere in a request routed to a ``response`` slot makes the
   request a SERVING FAILURE (fact 10):
   ``response_start = UNLOCATED``, every row of it UNSTEERED, and a ledger
   entry ``{req_id, lora_int_id, cell, error_type, prompt_len, header_len,
   detail}`` for the proxy to claim over ``sisteer_take_request_failure``
   before the completion is returned. The same shape ledgers, on EVERY
   restricted mode (their masks need the prompt length), a fact-table
   overflow, a missing fact and a runner-state disagreement — EVERY ledgered
   request is served unsteered, never the reverse. A request on an
   all-positions slot, a plain LoRA or no LoRA never needs a boundary and
   never ledgers. AFTER the original the wrapper judges the requests whose
   adapter was not yet in a slot at admission (the step activates it).
7. V1 ``LoRAModelRunnerMixin.set_active_loras(input_batch, num_scheduled_
   tokens, ...)`` / V2 ``LoraState.make_lora_inputs(req_ids, idx_mapping,
   num_scheduled_tokens)`` — after the original, ``fill_(SENTINEL)`` then
   ``copy_`` of ``np.repeat(values_by_req, num_scheduled_tokens)`` into
   ``prompt_len_by_token`` / ``response_start_by_token`` on EVERY step
   (``slot_by_token``'s discipline; a mapping the LoRA path skips as
   unchanged may still belong to other requests), cross-checked against the
   runner's own state (V1 ``input_batch.num_prompt_tokens`` + ``request_lora_
   mapping``; V2 ``req_states.prompt_len.np[idx_mapping]`` + ``lora_state.
   lora_ids``, reached through ``_BANK.runner``, fact 11): a disagreement is a
   recorded FAULT and ledgers the request.

In the op (:func:`position_class_selection`, per row, graph-safe, chunked
trivially): ``response = positions >= response_start``; ``prompt_last =
positions == prompt_len - 1``; ``prefill_all = positions < prompt_len``;
``all`` = every row; a ``response`` row whose ``response_start`` is
``UNLOCATED``, and any restricted-mode row whose facts are the SENTINEL, is
UNSTEERED — the where-form keeps its bits. ``positions`` are absolute (valid across prefix-cache resume
and chunking, fact 1). A forward that carries NO positions while a
restricted-mode or ``skip_bos`` slot is loaded refuses (a class it cannot
evaluate is never served as ``all``).

:data:`STAGE2_POSITION_CLASSES` admits the restricted modes at load (pinned
equal to ``defense_cells.NATIVE_STAGE2_ARMED`` by the constants-agree test);
the per-request failure LEDGER is :class:`RequestFailureLedger` (the lens's
fail-closed rule, fact 15: entries leave ONLY by being claimed; at the cap it
LATCHES ``ledger_fault`` and every later claim returns a fault payload).

IMPORT CONTRACT. Importable with no vllm: the patches are installed only by
:func:`install`, which the env flag triggers at import; with the flag unset
``PATCH_REPORT.armed is False`` and nothing is touched. With the flag SET and
vllm not importable the import RAISES — a worker asked to arm the slot must
never come up unarmed. The op can also be registered on a vllm-less process
(:func:`ensure_op_registered` falls back to a fragment library of the same
``vllm`` namespace) so the CPU tests exercise the exact registered op
through its fake and eager impls.
"""

import hashlib
import inspect
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import (Any, Callable, Dict, List, Mapping, NamedTuple, Optional,
                    Sequence, Tuple)

import numpy as np
import torch
import torch.nn.functional as F

from .span_locator import rfind as _rfind  # torch-free, the ONE last-occurrence rule
from .serving_attestation import (ENGINE_REQUEST_ID_SUFFIX_RE,          # the ONE engine-id contract in the tree
                                  external_request_id_candidates)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# vocabulary
# ---------------------------------------------------------------------------

#: the launch-env flag the composer sets iff a served adapter carries a payload
ENV_FLAG = "SISTEER_NATIVE_RESIDUAL_SLOT"
#: THE PAIRED WRITE's launch-env flag (relu-differenced-native-qwen): the composer sets it iff
#: a served adapter's payload is the PAIRED op (``vllm_server._residual_slot_paired``), and the
#: bank then allocates its second-gate buffers AT CONSTRUCTION — before any graph is captured
#: — and the op body takes the two-term path. Unset, the bank allocates nothing extra and the
#: body is the single-gate one byte for byte (THE PAIRED WRITE, below); a paired payload
#: activated on such a bank REFUSES rather than serving one gate under the cell's name.
ENV_FLAG_PAIRED = "SISTEER_NATIVE_RESIDUAL_SLOT_PAIRED"
#: the op's name inside vLLM's ``vllm`` library — ``torch.ops.vllm.<OP_NAME>``
OP_NAME = "sisteer_residual_slot"
#: the ONE fully qualified name the wrapper, the report and the tests use
OP_QUALNAME = f"vllm.{OP_NAME}"
#: the safetensors key prefix of the payload beside the rank-1 zero LoRA pair
PAYLOAD_PREFIX = "sisteer_residual."
#: the ``adapter_config.json`` block describing the payload
CONFIG_BLOCK = "petri_residual_slot"
RESIDUAL_SLOT_SCHEMA = 1
#: the object kinds the slot serves (plan: "residual kind"); ``psr_paired_write`` is the
#: PAIRED write (relu-differenced-native-qwen), pinned equal to
#: ``defense_cells.RESIDUAL_SLOT_KINDS`` / ``lora_export.RESIDUAL_EXPORT_KINDS`` by test
RESIDUAL_SLOT_KINDS = ("psr", "vector", "vector_all", "diffmean", "static_r", "psr_paired_write")
#: the ONE-gate kinds (``OP_RELU_GATED_VECTOR``)
GATED_KINDS = ("psr",)
#: the TWO-gate kind (``OP_RELU_GATED_VECTOR_PAIR``): ``extraction.paired_write.
#: PSRPairedWriteObject.kind``, pinned equal by test
PAIRED_KINDS = ("psr_paired_write",)
CONSTANT_KINDS = ("vector", "vector_all", "diffmean", "static_r")
#: the three ops — the first and last spelled as ``lens_adapters.OP_RELU_GATED_VECTOR`` /
#: ``OP_ADD_VECTOR`` (pinned equal by test; this module imports no lens code); the PAIR op is
#: this slot's own (the lens has no adapter for the paired kind)
OP_RELU_GATED_VECTOR = "relu_gated_vector"
OP_RELU_GATED_VECTOR_PAIR = "relu_gated_vector_pair"
OP_ADD_VECTOR = "add_vector"
RESIDUAL_OPS = (OP_RELU_GATED_VECTOR, OP_RELU_GATED_VECTOR_PAIR, OP_ADD_VECTOR)
#: the payload fields per op: the paired op carries GATED_FIELDS (the ``pos`` side) AND
#: PAIRED_FIELDS (the ``neg`` side) for every layer
GATED_FIELDS = ("gate_w", "gate_b", "direction")
PAIRED_FIELDS = ("gate_w2", "gate_b2", "direction2")
CONSTANT_FIELDS = ("vector",)
RESIDUAL_FIELDS = GATED_FIELDS + PAIRED_FIELDS + CONSTANT_FIELDS
#: the paired write's declared application order — ``extraction.paired_write.
#: APPLICATION_ORDER`` spelled here (pinned equal by test; this module imports no
#: extraction code): BOTH gates read the identical pre-write layer output, one combined add
PAIRED_APPLICATION_ORDER = "shared_pre_write_state"
#: the two terms' signed coefficients, ``(pos, neg)`` — ``PSRPairedWriteObject.terms``' own
#: ``[(+1.0, pos), (-1.0, neg)]``. The kernel computes exactly ``+g1 d1 - g2 d2`` at route
#: dose 1.0 and never rescales a side, so a payload declaring anything else is refused: the
#: coefficients ride in the block as a FACT about the served function, not as a knob.
PAIRED_TERM_COEFFS = (1.0, -1.0)
#: the native position modes (``NATIVE_POSITION_MODES``)
POSITION_MODES = ("all", "response", "prompt_last", "prefill_all")
MODE_CODES = {m: i for i, m in enumerate(POSITION_MODES)}
MODE_ALL, MODE_RESPONSE, MODE_PROMPT_LAST, MODE_PREFILL_ALL = (MODE_CODES[m] for m in POSITION_MODES)
#: the modes whose rows need the per-request facts (everything but ``all``)
RESTRICTED_MODES = tuple(m for m in POSITION_MODES if m != "all")
#: the modes whose mask needs the RESPONSE BOUNDARY (the lens's header-located
#: mask): ``response`` alone — ``prompt_last`` / ``prefill_all`` read the prompt
#: length only (the lens's ``last`` / prefill-``all`` never consult the header),
#: so an unlocated header fails a request on THESE modes and no other
BOUNDARY_MODES = ("response",)
#: the per-token fact buffers' fill (``prompt_len_by_token`` /
#: ``response_start_by_token``): this row belongs to no request the fact table
#: knows — a pad row, a row past the mapping, a request whose facts are gone.
#: Never a valid prompt length or boundary; a restricted-mode row reading it is
#: UNSTEERED.
POSITION_FACT_SENTINEL = -1
#: the response boundary of a request whose header could not be located (the
#: header ids occur nowhere in its prompt, no header ids are known, or the
#: prompt ids are absent): every restricted-mode row of it is UNSTEERED and the
#: request is LEDGERED (fact 10). Distinct from the SENTINEL so a buffer read
#: tells "no facts" from "facts, boundary unlocated".
UNLOCATED = -2
#: the ledger's ``error_type`` vocabulary (the proxy reads the payload's keys,
#: never these values, but they name the failure in the refusal text)
ERROR_BOUNDARY_UNLOCATED = "ResponseBoundaryUnlocated"
ERROR_FACTS_OVERFLOW = "RequestFactsOverflow"
ERROR_FACTS_MISSING = "RequestFactsMissing"
ERROR_PROMPT_LEN_DISAGREES = "PromptLenDisagreement"
ERROR_LORA_ID_DISAGREES = "LoraIdDisagreement"
#: the keys of EVERY ledger entry as RECORDED (``req_id`` is the ENGINE id the
#: scheduler handed the worker; ``external_req_id`` the completion id it belongs to)
LEDGER_ENTRY_KEYS = ("req_id", "external_req_id", "lora_int_id", "cell", "error_type", "prompt_len",
                     "header_len", "detail")
#: ``take_request_failure``'s non-fault payload: the first claimed entry plus EVERY
#: engine id the claim popped (one for ``n == 1``; a parallel-sampling child per choice)
CLAIM_KEYS = LEDGER_ENTRY_KEYS + ("claimed_engine_ids",)
#: THE ENGINE's request-id contract at the pinned vLLM 0.25.1 (the lens's precedent
#: ``serving_attestation.ENGINE_REQUEST_ID_SUFFIX_RE``): the
#: OpenAI layer mints the completion ``id`` (the EXTERNAL id), ``AsyncLLM.add_request``
#: (``v1/engine/async_llm.py:368``) calls ``InputProcessor.assign_request_id``
#: (``v1/engine/input_processor.py:223-240``), which keeps the external id under
#: ``external_req_id`` and sets ``request_id = f"{external_req_id}-{random_uuid():.8}"``
#: (``-`` + 8 lowercase hex, ``utils.random_uuid`` is hex) — unconditionally unless
#: ``VLLM_DISABLE_REQUEST_ID_RANDOMIZATION`` (never set on our launches); for ``n > 1``
#: ``ParentRequest.get_child_info`` (``v1/engine/parallel_sampling.py:92``) mints each
#: child ``f"{index}_{parent.request_id}"`` from the ALREADY-suffixed parent id. Every
#: id the scheduler and the runners see — ``scheduled_new_reqs[*].req_id``,
#: ``finished_req_ids``, the runner's batch — is the ENGINE id; the proxy claims a
#: failure by the completion ``id``. The ledger therefore indexes every entry under
#: the external ids its engine id CAN belong to (:func:`external_request_ids`) and a
#: claim by the external id pops them ALL; readiness refuses a worker that admitted
#: an id OFF this contract (``fact_table.off_contract``), so a vLLM bump that changes
#: the minting fails readiness before the first steered request.
ENGINE_CHILD_PREFIX_RE = re.compile(r"\A\d+_")
#: THE ENGINE'S OWN WARM-UP REQUESTS at the pinned vLLM 0.25.1: the V2 runner's warm-up
#: (``v1/worker/gpu/warmup.py:224`` ``req_ids = [f"_warmup_{i}_" for i in range(num_reqs)]``,
#: built into ``NewRequestData.from_request`` — they DO arrive through ``scheduled_new_reqs``,
#: ``num_reqs`` = ``max_num_seqs``) and the profiling dummy run (``gpu/model_runner.py:535``
#: ``_dummy_req_{i}``, an empty SchedulerOutput — batch rows only). They never pass
#: ``InputProcessor.assign_request_id`` (no OpenAI-layer id, no completion to claim), so they
#: are TABLED (their rows need facts) but counted under ``warmup``, never ``admitted``, and never
#: held to the engine-id contract. The pattern is PINNED: a vLLM rename makes the renamed rows
#: read as off-contract client requests and readiness refuses loudly (C5'' 513982 read 107 of
#: them as off-contract under the first cut of the pin and refused a healthy engine).
ENGINE_WARMUP_ID_RE = re.compile(r"\A(_warmup_\d+_|_dummy_req_\d+)\Z")


def external_request_ids(engine_id: Any) -> Tuple[str, ...]:
    """The external ids engine id ``engine_id`` CAN belong to, most specific first:
    the id itself (randomization disabled / an offline record), the id minus exactly
    one engine suffix, and — for a parallel-sampling child ``<i>_<parent>`` — the
    parent forms of the same two. Empty for an empty/missing id."""
    if engine_id is None:
        return ()
    r = str(engine_id)
    if not r:
        return ()
    out: List[str] = list(external_request_id_candidates(r))
    parent = ENGINE_CHILD_PREFIX_RE.sub("", r, count=1)
    if parent and parent != r:
        out += [c for c in external_request_id_candidates(parent) if c not in out]
    return tuple(out)


def engine_id_on_contract(engine_id: Any) -> bool:
    """Whether ``engine_id`` wears EXACTLY the pinned engine suffix (after an optional
    parallel-sampling child prefix) — what every id the scheduler hands the worker
    looks like at vLLM 0.25.1 with randomization on."""
    r = ENGINE_CHILD_PREFIX_RE.sub("", str(engine_id or ""), count=1)
    m = ENGINE_REQUEST_ID_SUFFIX_RE.search(r)
    return m is not None and m.start() > 0
#: the request FACT table's bound (the recorder's discipline,
#: ``worker_reports.LORA_ROUTING_RECORDER_CAP``): the table holds LIVE
#: requests only (finished ones drop), so its size is bounded by the
#: scheduler's running set (hundreds) and the cap is a leak detector — past
#: it ``overflow`` latches, the overflowing request is ledgered when its
#: adapter is restricted-mode, and readiness refuses the fact.
REQUEST_FACTS_CAP = 16_384
#: how many faults (runner-state disagreements, missing facts, unbound
#: reads) the bank keeps verbatim for the report; the count is unbounded
MAX_FAULTS_KEPT = 16
#: the dtype policies a payload declares — the gated write computes its gate in
#: fp32 and casts the addition to the model dtype ONCE; the constant add is in
#: the model dtype throughout
DTYPE_POLICY_GATED = "gate_fp32_cast_once"
DTYPE_POLICY_CONSTANT = "model_dtype_add"
DTYPE_POLICIES = {OP_RELU_GATED_VECTOR: DTYPE_POLICY_GATED,
                  OP_RELU_GATED_VECTOR_PAIR: DTYPE_POLICY_GATED,   # both gates fp32, the DIFFERENCE cast once
                  OP_ADD_VECTOR: DTYPE_POLICY_CONSTANT}
#: stage 2 LANDED: every ``POSITION_MODES`` entry may be
#: loaded into a slot; the per-token position class and the request-scoped
#: response-boundary failure serve the restricted ones. Pinned equal to
#: ``defense_cells.NATIVE_STAGE2_ARMED`` (torch-free) by the constants test.
STAGE2_POSITION_CLASSES = True
#: the lens's ledger bound (``lens_worker.REQUEST_FAILURES_CAP``)
REQUEST_FAILURES_CAP = 1024
#: the marker every installed wrapper carries
WRAP_MARK = "__sisteer_wrapped__"
#: the gate contract's constant (fact 14): ``|g - g_ref| <= 4 d eps32 |x| |w|``
GATE_CONTRACT_FACTOR = 4.0
EPS32 = float(torch.finfo(torch.float32).eps)

#: the wheel's decoder-layer classes, by module and name (fact 1 / §1.3 item 5).
#: Every class except ``Qwen3_5DecoderLayer`` DEFINES ``forward``; Qwen3.5
#: inherits Qwen3Next's and is listed so its ``__init__`` is patched (for the
#: boundary) and its presence REPORTED.
VLLM_DECODER_LAYER_CLASSES: Tuple[Tuple[str, str], ...] = (
    ("vllm.model_executor.models.llama", "LlamaDecoderLayer"),
    ("vllm.model_executor.models.gemma4", "Gemma4DecoderLayer"),
    ("vllm.model_executor.models.qwen3_next", "Qwen3NextDecoderLayer"),
    ("vllm.model_executor.models.qwen3_5", "Qwen3_5DecoderLayer"),
    ("vllm.model_executor.models.nemotron_h", "NemotronHMLPDecoderLayer"),
    ("vllm.model_executor.models.nemotron_h", "NemotronHMoEDecoderLayer"),
    ("vllm.model_executor.models.nemotron_h", "NemotronHMambaDecoderLayer"),
    ("vllm.model_executor.models.nemotron_h", "NemotronHAttentionDecoderLayer"),
)


class ResidualSlotError(RuntimeError):
    """A refusal of this module: a state that cannot be read, a payload that
    is not what the config block says, a patch that cannot be installed."""


def op_for_kind(kind: str) -> str:
    """The op a residual kind serves through (the ONE kind -> op rule the validator and the
    exporter share): one gate for ``psr``, two for ``psr_paired_write``, the constant add for
    the vector kinds; a kind outside :data:`RESIDUAL_SLOT_KINDS` refuses."""
    if kind in GATED_KINDS:
        return OP_RELU_GATED_VECTOR
    if kind in PAIRED_KINDS:
        return OP_RELU_GATED_VECTOR_PAIR
    if kind in CONSTANT_KINDS:
        return OP_ADD_VECTOR
    raise ResidualSlotError(f"kind {kind!r} is not a residual kind {list(RESIDUAL_SLOT_KINDS)}")


def _paired_from_env() -> bool:
    """Whether this worker was launched for a PAIRED payload (:data:`ENV_FLAG_PAIRED`) — read
    ONCE, at bank construction, so the allocation and the op body's path are fixed before the
    first forward (and therefore before any CUDA graph is captured)."""
    return os.environ.get(ENV_FLAG_PAIRED) == "1"


# ---------------------------------------------------------------------------
# the arithmetic — pure functions (the op's eager impl and the CPU reference)
# ---------------------------------------------------------------------------


class SlotBuffers(NamedTuple):
    """The static per-slot buffers the op reads (all on one device). ``S``
    slots, ``L`` boundaries (decoder blocks), ``d`` hidden.

    The last four fields are THE PAIRED WRITE's (relu-differenced-native-qwen) and default
    to ``None``: a bank that was not armed for a paired payload (:data:`ENV_FLAG_PAIRED`)
    never allocates them, so a LoRA-only / single-gate standup's memory and its captured
    graph are the certified ones; the op body branches on ``gate_w2 is None`` — a HOST fact
    fixed at construction, never a tensor read — and takes the single-gate path verbatim."""

    slot_by_token: torch.Tensor   # [max_tokens] int64; -1 = unadapted / pad
    active: torch.Tensor          # [S, L] bool — slot s writes at boundary l
    gated: torch.Tensor           # [S] bool — relu-gated (PSR, paired) vs constant add
    skip_bos: torch.Tensor        # [S] bool — position 0 excluded
    gate_w: torch.Tensor          # [S, L, d] fp32
    gate_b: torch.Tensor          # [S, L] fp32
    direction: torch.Tensor       # [S, L, d] fp32
    vector: torch.Tensor          # [S, L, d] model dtype (coeff folded)
    mode: torch.Tensor            # [S] int8 — MODE_CODES (0 = all; a cleared slot reads all)
    prompt_len_by_token: torch.Tensor      # [max_tokens] int64; POSITION_FACT_SENTINEL = no facts
    response_start_by_token: torch.Tensor  # [max_tokens] int64; >= 0 located, UNLOCATED, or SENTINEL
    paired: Optional[torch.Tensor] = None      # [S] bool — slot s carries a second gate term
    gate_w2: Optional[torch.Tensor] = None     # [S, L, d] fp32 — the neg side's gate (zero on unpaired slots)
    gate_b2: Optional[torch.Tensor] = None     # [S, L] fp32
    direction2: Optional[torch.Tensor] = None  # [S, L, d] fp32


def position_class_selection(mode_tok: torch.Tensor, pos: Optional[torch.Tensor],
                             prompt_len_tok: torch.Tensor, response_start_tok: torch.Tensor
                             ) -> torch.Tensor:
    """THE POSITION CLASS per row, over the per-row slot mode code
    ``mode_tok`` ``[T]``, the absolute positions ``pos`` ``[T]`` (or ``None``)
    and the per-row request facts ``prompt_len_tok`` / ``response_start_tok``
    ``[T]``: ``all`` rows are always in; ``response`` rows iff ``pos >=
    response_start`` (the header, any seed and every generated row — the
    lens's THE RESPONSE MASK); ``prompt_last`` rows iff ``pos == prompt_len -
    1`` (the lens's ``positions="last"`` in prefill; no decode row); ``prefill_all``
    rows iff ``pos < prompt_len`` (the lens's ``positions="all"`` in prefill).
    A ``response`` row whose boundary is not located (``response_start < 0``:
    ``UNLOCATED`` or the SENTINEL) is OUT — its request was ledgered (fact
    10); a restricted-mode row without facts (``prompt_len`` the SENTINEL: a
    pad row, a failed request) is OUT. With ``pos`` ``None`` every
    restricted-mode row is OUT (the bank refuses that forward before it gets
    here; this is the pure function's own fail-closed shape). Elementwise, no
    host reads: graph-capturable and chunked trivially."""
    restricted = mode_tok != MODE_ALL
    if pos is None:
        return ~restricted
    has_facts = prompt_len_tok >= 0
    located = response_start_tok >= 0
    return (~restricted
            | ((mode_tok == MODE_RESPONSE) & located & (pos >= response_start_tok))
            | ((mode_tok == MODE_PROMPT_LAST) & has_facts & (pos == prompt_len_tok - 1))
            | ((mode_tok == MODE_PREFILL_ALL) & has_facts & (pos < prompt_len_tok)))


def gate_contract_bound(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """The per-row bound of the gate contract: ``4 * d * eps32 * ||x_t||_2 *
    ||w_t||_2`` for ``x`` ``[T, d]`` and ``w`` ``[d]`` or ``[T, d]`` (fp32)."""
    x32 = x.float()
    w32 = w.float()
    if w32.dim() == 1:
        w32 = w32.unsqueeze(0).expand_as(x32)
    d = x32.shape[-1]
    return GATE_CONTRACT_FACTOR * d * EPS32 * x32.norm(dim=-1) * w32.norm(dim=-1)


def writeback(hidden: torch.Tensor, true: torch.Tensor, applied: torch.Tensor,
              sel: torch.Tensor) -> torch.Tensor:
    """THE LENS'S WRITE-BACK, given the applied stream: ``delta = applied -
    true; hidden' = hidden + delta`` on the selected rows
    (``lens_worker._apply_slots_block``: ``delta = applied - true_rows;
    new_hidden[rows] = new_hidden[rows] + delta``), the other rows the SAME
    bits as ``hidden`` (``torch.where``, never ``+ 0``: a signed zero must
    not move). ``hidden``, ``true``, ``applied`` share the model dtype;
    ``sel`` is ``[T]`` bool."""
    delta = applied - true
    new = hidden + delta
    return torch.where(sel.unsqueeze(-1), new, hidden)


#: THE ROW CHUNK of the op's eager body (GPU proof job 508688, 2026-09-15): vLLM's memory
#: profile runs ONE forward at ``max_num_batched_tokens`` — under the no-chunking invariant
#: that is S x L (471,808 rows for llama8b at 97 x 4864) — and every ``[T, d]`` temporary of
#: the arithmetic below (the fp32 gate gather, the direction gather, the fp32 stream, the
#: add, the two applied streams, the where) is then a multi-GiB tensor; their PEAK is what
#: the profiler subtracts from the KV budget. Unchunked, the llama8b PSR certification lost
#: 16 GiB of KV (61.9 -> 45.7 GiB) and its standup refused at 77 of 97 seqs; nemotron lost
#: 15.6 GiB. The body therefore runs in row chunks of this size: the arithmetic per row is
#: unchanged, the ``where`` no-op stays bit-exact, and a captured CUDA graph (T <= the
#: capture size, far below this) takes the single-chunk path exactly as before. The gate's
#: per-row reduction may pick another kernel at another T (fact 14): that is already the
#: stated error contract, never an exactness claim.
RESIDUAL_FORWARD_CHUNK_ROWS = 8192


def residual_slot_forward(hidden: torch.Tensor, residual: Optional[torch.Tensor],
                          positions: Optional[torch.Tensor], boundary: int,
                          buffers: SlotBuffers) -> torch.Tensor:
    """The op body (module docstring, THE ARITHMETIC) over ``hidden`` ``[T, d]``
    in the model dtype, the paired ``residual`` (``None`` on gemma4, whose
    block folds the residual in) and the decoder block ``boundary``. Pure:
    no host reads of tensor DATA (graph-capturable); shapes only. Rows are
    processed in chunks of :data:`RESIDUAL_FORWARD_CHUNK_ROWS` (one chunk for
    every captured shape; the profile run's S x L rows in many) so the
    temporaries' peak is bounded whatever ``T`` is."""
    if hidden.dim() != 2:
        raise ResidualSlotError(
            f"residual slot: hidden must be [num_tokens, hidden], got shape "
            f"{tuple(hidden.shape)}")
    T = hidden.shape[0]
    if T > buffers.slot_by_token.numel():
        raise ResidualSlotError(
            f"residual slot: the forward carries {T} rows but slot_by_token holds "
            f"{buffers.slot_by_token.numel()} — a batch wider than the LoRA mapping "
            "buffer; refusing rather than reading rows the mapping never covered")
    if residual is not None and (residual.shape != hidden.shape
                                 or residual.dtype != hidden.dtype):
        raise ResidualSlotError(
            f"residual slot: residual {tuple(residual.shape)}/{residual.dtype} does "
            f"not pair with hidden {tuple(hidden.shape)}/{hidden.dtype}")
    if buffers.vector.dtype != hidden.dtype:
        raise ResidualSlotError(
            f"residual slot: the bank's vector buffer is {buffers.vector.dtype} but "
            f"the model runs {hidden.dtype} — the constant add must be in the model "
            "dtype (the lens keeps the vector in the model dtype); refusing")
    idx = buffers.slot_by_token[:T]
    pos: Optional[torch.Tensor] = None
    if positions is not None:
        pos = positions
        if pos.dim() == 2:      # M-RoPE layout: the temporal index is row 0
            pos = pos[0]
        if pos.shape[0] != T:
            raise ResidualSlotError(
                f"residual slot: positions carries {pos.shape[0]} entries for {T} rows")
    pl = buffers.prompt_len_by_token[:T]
    rs = buffers.response_start_by_token[:T]
    chunk = int(RESIDUAL_FORWARD_CHUNK_ROWS)
    if T <= chunk:
        return _residual_slot_forward_rows(hidden, residual, pos, boundary, buffers, idx, pl, rs)
    out = torch.empty_like(hidden)
    for start in range(0, T, chunk):
        sl = slice(start, min(T, start + chunk))
        out[sl] = _residual_slot_forward_rows(
            hidden[sl], None if residual is None else residual[sl],
            None if pos is None else pos[sl], boundary, buffers, idx[sl], pl[sl], rs[sl])
    return out


def _residual_slot_forward_rows(hidden: torch.Tensor, residual: Optional[torch.Tensor],
                                pos: Optional[torch.Tensor], boundary: int,
                                buffers: SlotBuffers, idx: torch.Tensor,
                                prompt_len_tok: torch.Tensor, response_start_tok: torch.Tensor
                                ) -> torch.Tensor:
    """THE ARITHMETIC over one row chunk: ``hidden`` / ``residual`` ``[T, d]``,
    ``pos`` ``[T]`` or None, ``idx`` the chunk's ``slot_by_token`` rows,
    ``prompt_len_tok`` / ``response_start_tok`` the chunk's per-token facts.

    THE PAIRED WRITE (relu-differenced-native-qwen; ``extraction/paired_write.py``'s
    ``V(z) = u_pos(z) - u_neg(z)`` with ``application_order = shared_pre_write_state``):
    when the bank is armed (``buffers.gate_w2 is not None``) a SECOND gate term is read off
    the same ``true`` — the pre-write state BOTH gates see, the point of the paired object —
    ``g2 = relu(true . w2 + b2)`` and the fp32 add becomes ``(g sel) d - (g2 sel) d2``:
    the difference is formed in fp32, cast to the model dtype ONCE and written through the
    ONE ``writeback`` below, exactly as the object's hook writes ``hidden + (u_pos -
    u_neg).to(dtype)``. The subtraction ``a - b`` is bit-identical to the object's
    ``a + (-1.0 * g2)[..., None] * d2`` (a sign flip and a negation are exact in IEEE
    arithmetic), so the two differ only by the gate contract (fact 14) per TERM.

    BYTE-IDENTITY OF UNPAIRED SLOTS, and the mechanism chosen. Two mechanisms were open:
    rely on the unpaired slots' zero ``w2 / b2 / d2`` rows (``g2 = relu(0) = 0`` and ``a -
    0 = a``), or SELECT with ``torch.where``. The zero-row route is exact for every finite
    ``a`` EXCEPT a signed zero: ``0 * d2`` is ``-0.0`` wherever ``d2 < 0`` and ``-0.0 -
    (-0.0)`` is ``+0.0``, so a closed gate's ``-0.0`` add (``0 * d`` with ``d < 0``) would
    flip sign on an unpaired row and then move a ``true`` of ``-0.0`` — a real, if rare,
    bit change on a slot that promised none. The where-form is chosen: for a row whose slot
    is not paired ``torch.where`` returns the ORIGINAL ``add32`` tensor's bits, and the
    single-gate arithmetic before it is the untouched line-for-line body of the unarmed
    path, so an unpaired slot on an armed bank writes byte-for-byte what it writes on an
    unarmed one (the tests pin both against the pre-change body). The unarmed path itself
    is the pre-change body verbatim: no second gather, no where, no extra temporary — a
    standup without a paired payload keeps its certified graph and memory."""
    live = idx >= 0
    idx_c = idx.clamp(min=0)                      # -1 never indexes a slot
    sel = live & buffers.active[idx_c, boundary]  # [T]
    if pos is not None:
        sel = sel & ~(buffers.skip_bos[idx_c] & (pos == 0))
    sel = sel & position_class_selection(buffers.mode[idx_c], pos, prompt_len_tok,
                                         response_start_tok)
    true = hidden if residual is None else hidden + residual
    w_tok = buffers.gate_w[idx_c, boundary]       # [T, d] fp32
    b_tok = buffers.gate_b[idx_c, boundary]       # [T]
    d_tok = buffers.direction[idx_c, boundary]    # [T, d] fp32
    v_tok = buffers.vector[idx_c, boundary]       # [T, d] model dtype
    g = F.relu((true.float() * w_tok).sum(-1) + b_tok)          # CONTRACT (fact 14)
    add32 = (g * sel.to(torch.float32)).unsqueeze(-1) * d_tok   # exact per element
    # ^ the single-gate expression and its temporaries VERBATIM as certified: nothing hoisted
    #   before it, nothing shared with the branch below. The op body runs inside CUDA-graph
    #   capture as one opaque op, so the unarmed path's DISPATCH SEQUENCE (not only its bytes)
    #   is the captured graph; a probe caught a hoisted mask cast reordering it,
    #   and tests/test_native_paired_write_slot.py pins the op sequence against this body.
    if buffers.gate_w2 is not None:                             # THE PAIRED WRITE (armed banks only)
        w2_tok = buffers.gate_w2[idx_c, boundary]               # [T, d] fp32
        b2_tok = buffers.gate_b2[idx_c, boundary]               # [T]
        d2_tok = buffers.direction2[idx_c, boundary]            # [T, d] fp32
        g2 = F.relu((true.float() * w2_tok).sum(-1) + b2_tok)  # the SAME pre-write state; CONTRACT per term
        add32_2 = (g2 * sel.to(torch.float32)).unsqueeze(-1) * d2_tok   # the second term's own mask cast
        add32 = torch.where(buffers.paired[idx_c].unsqueeze(-1), add32 - add32_2, add32)
    applied_gated = true + add32.to(hidden.dtype)               # gated_vector_write: ONE cast, ONE write
    applied_const = true + v_tok                                # out + addition
    applied = torch.where(buffers.gated[idx_c].unsqueeze(-1),
                          applied_gated, applied_const)
    return writeback(hidden, true, applied, sel)


# ---------------------------------------------------------------------------
# the payload: what a bank entry carries beside its rank-1 zero LoRA pair
# ---------------------------------------------------------------------------


def _int_list(name: str, raw: Any, *, minimum: int = 0) -> Tuple[int, ...]:
    if not isinstance(raw, (list, tuple)):
        raise ResidualSlotError(f"{CONFIG_BLOCK}.{name} must be a list, got {raw!r}")
    out: List[int] = []
    for v in raw:
        if isinstance(v, bool) or not isinstance(v, int) or v < minimum:
            raise ResidualSlotError(
                f"{CONFIG_BLOCK}.{name} entry {v!r} is not an int >= {minimum}")
        out.append(int(v))
    return tuple(out)


def residual_fingerprint(block: Mapping[str, Any],
                         tensors: Mapping[str, torch.Tensor]) -> str:
    """sha256 over the canonical JSON of the config block and every payload
    tensor's key, dtype, shape and raw bytes in sorted key order — the
    ``residual_fingerprint`` the attestation compares."""
    h = hashlib.sha256()
    h.update(json.dumps(dict(block), sort_keys=True, separators=(",", ":"),
                        ensure_ascii=True).encode("utf-8"))
    for key in sorted(tensors):
        t = tensors[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(json.dumps(list(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


@dataclass(frozen=True)
class ResidualPayload:
    """One validated payload: the config block's facts plus the per-layer
    tensors keyed by FIT layer (``layers[i]`` writes at ``boundaries[i]``)."""

    kind: str
    op: str
    layers: Tuple[int, ...]
    boundaries: Tuple[int, ...]
    position_mode: str
    response_header_len: Optional[int]
    response_header_ids: Tuple[int, ...]
    skip_bos: bool
    coeff: float
    hidden_size: int
    n_layers_model: int
    anchor_layer: int
    dtype_policy: str
    gate_w: Dict[int, torch.Tensor] = field(default_factory=dict)
    gate_b: Dict[int, torch.Tensor] = field(default_factory=dict)
    direction: Dict[int, torch.Tensor] = field(default_factory=dict)
    vector: Dict[int, torch.Tensor] = field(default_factory=dict)
    fingerprint: str = ""
    # -- THE PAIRED WRITE (op OP_RELU_GATED_VECTOR_PAIR): the ``neg`` side, both parents' shas
    #    and the two terms' signed coefficients; all unset on every other op --
    gate_w2: Dict[int, torch.Tensor] = field(default_factory=dict)
    gate_b2: Dict[int, torch.Tensor] = field(default_factory=dict)
    direction2: Dict[int, torch.Tensor] = field(default_factory=dict)
    parent_pos_sha256: Optional[str] = None
    parent_neg_sha256: Optional[str] = None
    parent_pos_coeff: Optional[float] = None
    parent_neg_coeff: Optional[float] = None

    @property
    def gated(self) -> bool:
        """A relu-gated write — one gate or two (the constant kinds are the other class)."""
        return self.op in (OP_RELU_GATED_VECTOR, OP_RELU_GATED_VECTOR_PAIR)

    @property
    def paired(self) -> bool:
        return self.op == OP_RELU_GATED_VECTOR_PAIR

    def block(self) -> Dict[str, Any]:
        """The ``petri_residual_slot`` config block, canonical. A single-gate / constant block
        carries EXACTLY the pre-paired key set (every certified payload's fingerprint hashes
        it, so a paired key may never appear there); a paired block carries the four paired
        keys besides — both parent shas and both term coefficients — which is how the
        fingerprint covers the parents' identities."""
        out = {
            "schema": RESIDUAL_SLOT_SCHEMA, "kind": self.kind, "op": self.op,
            "layers": list(self.layers), "boundaries": list(self.boundaries),
            "position_mode": self.position_mode,
            "response_header_len": self.response_header_len,
            "response_header_ids": list(self.response_header_ids),
            "skip_bos": self.skip_bos, "coeff": self.coeff,
            "hidden_size": self.hidden_size, "n_layers_model": self.n_layers_model,
            "anchor_layer": self.anchor_layer, "dtype_policy": self.dtype_policy,
        }
        if self.paired:
            out.update(parent_pos_sha256=self.parent_pos_sha256, parent_neg_sha256=self.parent_neg_sha256,
                       parent_pos_coeff=self.parent_pos_coeff, parent_neg_coeff=self.parent_neg_coeff)
        return out

    def tensors(self) -> Dict[str, torch.Tensor]:
        """The payload as safetensors entries ``sisteer_residual.layers.<l>.<field>``."""
        out: Dict[str, torch.Tensor] = {}
        for l in self.layers:
            if self.gated:
                out[f"{PAYLOAD_PREFIX}layers.{l}.gate_w"] = self.gate_w[l]
                out[f"{PAYLOAD_PREFIX}layers.{l}.gate_b"] = self.gate_b[l]
                out[f"{PAYLOAD_PREFIX}layers.{l}.direction"] = self.direction[l]
                if self.paired:
                    out[f"{PAYLOAD_PREFIX}layers.{l}.gate_w2"] = self.gate_w2[l]
                    out[f"{PAYLOAD_PREFIX}layers.{l}.gate_b2"] = self.gate_b2[l]
                    out[f"{PAYLOAD_PREFIX}layers.{l}.direction2"] = self.direction2[l]
            else:
                out[f"{PAYLOAD_PREFIX}layers.{l}.vector"] = self.vector[l]
        return out

    def summary(self) -> Dict[str, Any]:
        """The per-slot facts the report carries."""
        out = {"kind": self.kind, "op": self.op, "layers": list(self.layers),
               "boundaries": list(self.boundaries), "mode": self.position_mode,
               "header_len": self.response_header_len, "skip_bos": self.skip_bos,
               "coeff": self.coeff, "fingerprint": self.fingerprint}
        if self.paired:
            out.update(parent_pos_sha256=self.parent_pos_sha256, parent_neg_sha256=self.parent_neg_sha256)
        return out


_BLOCK_KEYS = frozenset({"schema", "kind", "op", "layers", "boundaries", "position_mode",
                         "response_header_len", "response_header_ids", "skip_bos",
                         "coeff", "hidden_size", "n_layers_model", "anchor_layer",
                         "dtype_policy"})
#: the keys a PAIRED block carries BESIDES ``_BLOCK_KEYS`` (and no other block may carry)
_PAIRED_BLOCK_KEYS = frozenset({"parent_pos_sha256", "parent_neg_sha256",
                                "parent_pos_coeff", "parent_neg_coeff"})
_SHA256_RE = re.compile(r"\A[0-9a-f]{64}\Z")


def is_sha256_hex(value: Any) -> bool:
    """Whether ``value`` is a lowercase 64-hex sha256 digest (the parent shas' shape)."""
    return isinstance(value, str) and _SHA256_RE.match(value) is not None


def _validated_block(block: Any) -> Dict[str, Any]:
    if not isinstance(block, Mapping):
        raise ResidualSlotError(f"{CONFIG_BLOCK} must be a JSON object, got {type(block).__name__}")
    keys = set(block)
    paired = block.get("op") == OP_RELU_GATED_VECTOR_PAIR
    want_keys = (_BLOCK_KEYS | _PAIRED_BLOCK_KEYS) if paired else _BLOCK_KEYS
    if keys != want_keys:
        raise ResidualSlotError(
            f"{CONFIG_BLOCK} keys {sorted(keys)} != {sorted(want_keys)} — the block is not "
            f"one this module wrote for op {block.get('op')!r} (schema drift"
            + ("; the paired keys belong to the paired op alone" if keys & _PAIRED_BLOCK_KEYS and not paired
               else "; a paired block carries both parent shas and both term coefficients" if paired else "")
            + "); refusing")
    if block["schema"] != RESIDUAL_SLOT_SCHEMA:
        raise ResidualSlotError(
            f"{CONFIG_BLOCK}.schema={block['schema']!r}; this module reads schema "
            f"{RESIDUAL_SLOT_SCHEMA}")
    kind = block["kind"]
    if kind not in RESIDUAL_SLOT_KINDS:
        raise ResidualSlotError(
            f"{CONFIG_BLOCK}.kind={kind!r} is not a residual kind {list(RESIDUAL_SLOT_KINDS)}")
    op = block["op"]
    if op not in RESIDUAL_OPS:
        raise ResidualSlotError(f"{CONFIG_BLOCK}.op={op!r} is not one of {list(RESIDUAL_OPS)}")
    want_op = op_for_kind(kind)
    if op != want_op:
        raise ResidualSlotError(
            f"{CONFIG_BLOCK}: kind {kind!r} serves through op {want_op!r}, the block says {op!r}"
            + (" — a paired payload's kind is 'psr_paired_write' and nothing else" if paired else ""))
    if paired:
        shas = (block["parent_pos_sha256"], block["parent_neg_sha256"])
        for name, sha in zip(("parent_pos_sha256", "parent_neg_sha256"), shas):
            if not isinstance(sha, str) or not _SHA256_RE.match(sha):
                raise ResidualSlotError(
                    f"{CONFIG_BLOCK}.{name}={sha!r} is not a lowercase 64-hex sha256 — a paired payload "
                    "names BOTH parents' object bytes; refusing")
        if shas[0] == shas[1]:
            raise ResidualSlotError(
                f"{CONFIG_BLOCK}: parent_pos_sha256 == parent_neg_sha256 ({shas[0][:12]}...) — both sides name "
                "the same parent, whose paired write is identically zero; a no-op under a cell's name is "
                "never served; refusing")
        coeffs = (block["parent_pos_coeff"], block["parent_neg_coeff"])
        for name, c in zip(("parent_pos_coeff", "parent_neg_coeff"), coeffs):
            if isinstance(c, bool) or not isinstance(c, (int, float)) or c != c:
                raise ResidualSlotError(f"{CONFIG_BLOCK}.{name}={c!r} is not a finite number")
        if tuple(float(c) for c in coeffs) != PAIRED_TERM_COEFFS:
            raise ResidualSlotError(
                f"{CONFIG_BLOCK}: parent (pos, neg) coeffs {coeffs} != {PAIRED_TERM_COEFFS} — the slot computes "
                "exactly +g1 d1 - g2 d2 at route dose 1.0 and never rescales one side (the paired object's "
                "own terms); a payload declaring another composition is a different intervention; refusing")
    if block["dtype_policy"] != DTYPE_POLICIES[op]:
        raise ResidualSlotError(
            f"{CONFIG_BLOCK}.dtype_policy={block['dtype_policy']!r}; op {op!r} is served under "
            f"{DTYPE_POLICIES[op]!r} — a payload under another policy is a different intervention")
    layers = _int_list("layers", block["layers"])
    boundaries = _int_list("boundaries", block["boundaries"])
    if not layers:
        raise ResidualSlotError(f"{CONFIG_BLOCK}.layers is empty — nothing to serve")
    if len(set(layers)) != len(layers):
        raise ResidualSlotError(f"{CONFIG_BLOCK}.layers carries a duplicate: {list(layers)}")
    if len(boundaries) != len(layers):
        raise ResidualSlotError(
            f"{CONFIG_BLOCK}.boundaries has {len(boundaries)} entries for {len(layers)} layers")
    if len(set(boundaries)) != len(boundaries):
        raise ResidualSlotError(
            f"{CONFIG_BLOCK}.boundaries carries a duplicate: {list(boundaries)} — two writes "
            "at one block output would sum, not serve the object")
    for name in ("hidden_size", "n_layers_model"):
        v = block[name]
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ResidualSlotError(f"{CONFIG_BLOCK}.{name}={v!r} is not a positive int")
    anchor = block["anchor_layer"]
    if isinstance(anchor, bool) or not isinstance(anchor, int) or anchor < 0 \
            or anchor >= block["n_layers_model"]:
        raise ResidualSlotError(
            f"{CONFIG_BLOCK}.anchor_layer={anchor!r} is outside [0, {block['n_layers_model']})")
    bad = [b for b in boundaries if b >= block["n_layers_model"]]
    if bad:
        raise ResidualSlotError(
            f"{CONFIG_BLOCK}.boundaries {bad} are outside the model's {block['n_layers_model']} blocks")
    mode = block["position_mode"]
    if mode not in POSITION_MODES:
        raise ResidualSlotError(
            f"{CONFIG_BLOCK}.position_mode={mode!r} is not one of {list(POSITION_MODES)}")
    hl = block["response_header_len"]
    if mode == "response":
        if isinstance(hl, bool) or not isinstance(hl, int) or hl < 1:
            raise ResidualSlotError(
                f"{CONFIG_BLOCK}: position_mode 'response' needs response_header_len >= 1, "
                f"got {hl!r}")
    elif hl is not None:
        raise ResidualSlotError(
            f"{CONFIG_BLOCK}: response_header_len={hl!r} on position_mode {mode!r} — the "
            "header width belongs to the response mode alone")
    header_ids = _int_list("response_header_ids", block["response_header_ids"])
    if header_ids and hl is not None and len(header_ids) != hl:
        raise ResidualSlotError(
            f"{CONFIG_BLOCK}: response_header_ids carries {len(header_ids)} ids but "
            f"response_header_len is {hl} — the width and the ids must name one header")
    if not isinstance(block["skip_bos"], bool):
        raise ResidualSlotError(f"{CONFIG_BLOCK}.skip_bos must be a bool, got {block['skip_bos']!r}")
    coeff = block["coeff"]
    if isinstance(coeff, bool) or not isinstance(coeff, (int, float)) \
            or coeff != coeff or coeff in (float("inf"), float("-inf")):
        raise ResidualSlotError(f"{CONFIG_BLOCK}.coeff={coeff!r} is not a finite number")
    if op in (OP_RELU_GATED_VECTOR, OP_RELU_GATED_VECTOR_PAIR) and float(coeff) != 1.0:
        raise ResidualSlotError(
            f"{CONFIG_BLOCK}: a gated (PSR) payload carries coeff={coeff!r}; the PSR dose is "
            "baked into ||direction|| and the slot applies it at coeff 1.0 exactly — a baked "
            "coeff would be a different intervention; refusing")
    if op in (OP_RELU_GATED_VECTOR, OP_RELU_GATED_VECTOR_PAIR) and block["skip_bos"]:
        raise ResidualSlotError(
            f"{CONFIG_BLOCK}: a gated (PSR) payload with skip_bos=True — the lens's psr "
            "adapter never skips BOS (the gate is learned); refusing")
    return dict(block)


def payload_from_block_and_tensors(block: Any, tensors: Mapping[str, torch.Tensor]
                                   ) -> ResidualPayload:
    """Validate a config block against its payload tensors (both directions:
    every declared layer has exactly its op's fields, every tensor names a
    declared layer and a field of that op, fp32, the right shape, finite)
    and return the :class:`ResidualPayload`. The exporter validates what it
    writes with this; the loader validates what it reads."""
    from .serving_attestation import (ADAPTER_KEY_RESIDUAL, ServingAttestationError,
                                      classify_adapter_key)

    blk = _validated_block(block)
    op = blk["op"]
    fields = {OP_RELU_GATED_VECTOR: GATED_FIELDS,
              OP_RELU_GATED_VECTOR_PAIR: GATED_FIELDS + PAIRED_FIELDS,   # a paired op with ONE side present refuses below
              OP_ADD_VECTOR: CONSTANT_FIELDS}[op]
    d = int(blk["hidden_size"])
    declared = set(blk["layers"])
    seen: Dict[int, Dict[str, torch.Tensor]] = {l: {} for l in declared}
    for key in sorted(tensors):
        try:
            cls, layer, fld = classify_adapter_key(key)
        except ServingAttestationError as e:
            raise ResidualSlotError(str(e)) from e
        if cls != ADAPTER_KEY_RESIDUAL:
            raise ResidualSlotError(
                f"payload tensor {key!r} is not a {PAYLOAD_PREFIX}* key")
        if layer not in declared:
            raise ResidualSlotError(
                f"payload tensor {key!r} names layer {layer}, which {CONFIG_BLOCK}.layers "
                f"{sorted(declared)} does not declare")
        if fld not in fields:
            raise ResidualSlotError(
                f"payload tensor {key!r}: field {fld!r} is not one of op {op!r}'s "
                f"{list(fields)}")
        t = tensors[key]
        if not isinstance(t, torch.Tensor):
            raise ResidualSlotError(f"payload {key!r} is not a tensor")
        if t.dtype != torch.float32:
            raise ResidualSlotError(
                f"payload {key!r} is {t.dtype}; the payload is written fp32 (the objects' own "
                "save dtype) and cast under the declared policy at load — refusing a "
                "pre-cast payload")
        want_shape = (1,) if fld in ("gate_b", "gate_b2") else (d,)
        if tuple(t.shape) != want_shape:
            raise ResidualSlotError(
                f"payload {key!r} has shape {tuple(t.shape)}, expected {want_shape}")
        if not torch.isfinite(t).all():
            raise ResidualSlotError(f"payload {key!r} carries non-finite values")
        seen[layer][fld] = t.detach().clone()
    for l in sorted(declared):
        missing = [f for f in fields if f not in seen[l]]
        if missing:
            raise ResidualSlotError(
                f"payload lacks {missing} for declared layer {l} (op {op!r})")
    fp = residual_fingerprint(blk, tensors)
    return ResidualPayload(
        kind=blk["kind"], op=op, layers=tuple(blk["layers"]),
        boundaries=tuple(blk["boundaries"]), position_mode=blk["position_mode"],
        response_header_len=blk["response_header_len"],
        response_header_ids=tuple(blk["response_header_ids"]),
        skip_bos=bool(blk["skip_bos"]), coeff=float(blk["coeff"]),
        hidden_size=d, n_layers_model=int(blk["n_layers_model"]),
        anchor_layer=int(blk["anchor_layer"]), dtype_policy=blk["dtype_policy"],
        gate_w={l: seen[l]["gate_w"] for l in declared if "gate_w" in seen[l]},
        gate_b={l: seen[l]["gate_b"] for l in declared if "gate_b" in seen[l]},
        direction={l: seen[l]["direction"] for l in declared if "direction" in seen[l]},
        vector={l: seen[l]["vector"] for l in declared if "vector" in seen[l]},
        fingerprint=fp,
        gate_w2={l: seen[l]["gate_w2"] for l in declared if "gate_w2" in seen[l]},
        gate_b2={l: seen[l]["gate_b2"] for l in declared if "gate_b2" in seen[l]},
        direction2={l: seen[l]["direction2"] for l in declared if "direction2" in seen[l]},
        parent_pos_sha256=blk.get("parent_pos_sha256"), parent_neg_sha256=blk.get("parent_neg_sha256"),
        parent_pos_coeff=(None if blk.get("parent_pos_coeff") is None else float(blk["parent_pos_coeff"])),
        parent_neg_coeff=(None if blk.get("parent_neg_coeff") is None else float(blk["parent_neg_coeff"])))


def read_residual_payload(adapter_dir: Path) -> Optional[ResidualPayload]:
    """The payload of the bank entry at ``adapter_dir``: ``None`` when the
    entry carries neither a ``petri_residual_slot`` block nor any
    ``sisteer_residual.*`` tensor (a plain LoRA); a refusal when it carries
    one without the other; else the validated :class:`ResidualPayload`.
    Reads the safetensors HEADER for the key set and loads only the payload
    tensors (the LoRA factors are vLLM's to load)."""
    from safetensors import safe_open  # transformers-stack dep, lazy

    from .serving_attestation import (ADAPTER_CONFIG_FILE, ADAPTER_KEY_RESIDUAL,
                                      ADAPTER_WEIGHTS_FILE, ServingAttestationError,
                                      classify_adapter_key)

    d = Path(adapter_dir)
    cfg_path, weights = d / ADAPTER_CONFIG_FILE, d / ADAPTER_WEIGHTS_FILE
    if not cfg_path.is_file() or not weights.is_file():
        raise ResidualSlotError(f"{d}: not a PEFT adapter dir (missing {ADAPTER_CONFIG_FILE} or "
                                f"{ADAPTER_WEIGHTS_FILE})")
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ResidualSlotError(f"{cfg_path}: unparseable JSON: {e}") from e
    if not isinstance(cfg, Mapping):
        raise ResidualSlotError(f"{cfg_path} is not a JSON object")
    block = cfg.get(CONFIG_BLOCK)
    tensors: Dict[str, torch.Tensor] = {}
    with safe_open(str(weights), framework="pt") as fh:
        for key in fh.keys():
            try:
                cls = classify_adapter_key(key)[0]
            except ServingAttestationError as e:
                raise ResidualSlotError(f"{d}: {e}") from e
            if cls == ADAPTER_KEY_RESIDUAL:
                tensors[key] = fh.get_tensor(key)
    if block is None and not tensors:
        return None
    if block is None:
        raise ResidualSlotError(
            f"{d}: {len(tensors)} {PAYLOAD_PREFIX}* tensor(s) but no {CONFIG_BLOCK} block in "
            f"{ADAPTER_CONFIG_FILE} — a payload without its declaration; refusing")
    if not tensors:
        raise ResidualSlotError(
            f"{d}: a {CONFIG_BLOCK} block but no {PAYLOAD_PREFIX}* tensor — a declaration "
            "without its payload; refusing")
    try:
        return payload_from_block_and_tensors(block, tensors)
    except ResidualSlotError as e:
        raise ResidualSlotError(f"{d}: {e}") from e


# ---------------------------------------------------------------------------
# the per-request failure ledger (fact 15; written into by stage 2)
# ---------------------------------------------------------------------------


class RequestFailureLedger:
    """The lens's per-request steering-failure ledger, ported: entries leave
    ONLY by being claimed (:meth:`claim`); nothing evicts or expires one. At
    ``cap`` unclaimed entries a new failure LATCHES ``fault`` — from then on
    every claim, for ANY request id, returns the fault payload, so no
    post-check on a restricted-mode route can read "no failure" again (the
    lens condemns the worker instead; the latch keeps the engine serving
    while making the affected class of responses unreturnable — plan rev 4)."""

    FAULT = "ledger_overflow"

    def __init__(self, cap: int = REQUEST_FAILURES_CAP):
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ResidualSlotError(f"ledger cap must be a positive int, got {cap!r}")
        self.cap = cap
        self._entries: Dict[str, Dict[str, Any]] = {}          # by ENGINE id
        self._by_external: Dict[str, set] = {}                 # external-id candidate -> engine ids
        self.failed_ids: set = set()                            # engine ids
        self.fault = False

    def __len__(self) -> int:
        return len(self._entries)

    def record(self, req_id: str, **facts: Any) -> None:
        """Record the failure of ENGINE request ``req_id`` (the scheduler's id — the
        completion id plus the engine suffix, ``<i>_``-prefixed for a parallel-sampling
        child), indexed under every external id it can belong to."""
        rid = str(req_id)
        self.failed_ids.add(rid)
        if rid in self._entries:
            return
        if len(self._entries) >= self.cap:
            if not self.fault:
                log.error("residual slot: the per-request failure ledger holds %d unclaimed "
                          "entries (cap %d) — nothing is claiming request failures; LATCHING "
                          "ledger_fault: every claim now returns the fault payload",
                          len(self._entries), self.cap)
            self.fault = True
            return
        candidates = external_request_ids(rid)
        self._entries[rid] = {"req_id": rid, "external_req_id": candidates[-1], **facts}
        for c in candidates:
            self._by_external.setdefault(c, set()).add(rid)

    def claim(self, req_id: Any) -> Optional[Dict[str, Any]]:
        """Claim (pop) EVERY failure that belongs to ``req_id`` — the completion id the
        proxy reads off the response, or an engine id — and return the first with
        ``claimed_engine_ids`` naming all of them; ``None`` when none is ledgered. A
        failed parallel-sampling child condemns the whole completion (its choices
        are one response)."""
        if self.fault:
            return {"fault": self.FAULT, "unclaimed": len(self._entries), "cap": self.cap}
        x = str(req_id)
        ids = [rid for rid in self._by_external.get(x, ()) if rid in self._entries]
        if x in self._entries and x not in ids:
            ids.insert(0, x)
        if not ids:
            return None
        ids.sort(key=lambda rid: (rid != x, rid))
        first: Optional[Dict[str, Any]] = None
        for rid in ids:
            entry = self._entries.pop(rid)
            for c in external_request_ids(rid):
                bucket = self._by_external.get(c)
                if bucket is not None:
                    bucket.discard(rid)
                    if not bucket:
                        del self._by_external[c]
            first = first or entry
        return {**first, "claimed_engine_ids": ids}

    def report(self) -> Dict[str, Any]:
        return {"ledger_size": len(self._entries), "ledger_fault": bool(self.fault),
                "ledger_cap": self.cap}


# ---------------------------------------------------------------------------
# the per-request FACTS (stage 2): prompt length + response boundary
# ---------------------------------------------------------------------------


@dataclass
class RequestFacts:
    """One admitted request's facts: ``prompt_len`` (``POSITION_FACT_SENTINEL``
    when the prompt ids were absent), ``response_start`` (the LAST header
    occurrence, else ``UNLOCATED`` with ``boundary_detail`` saying why), its
    LoRA binding; then the judgement state — ``mode`` (the residual slot's
    position mode once its adapter is resident, ``""`` for a plain LoRA / no
    LoRA), ``failure`` (``(error_type, detail)`` naming a HARD failure:
    overflow, missing facts, a runner-state disagreement — these fail the
    request on every restricted mode; an unlocated boundary fails it on
    ``BOUNDARY_MODES`` only, judged at ledger time) and ``ledgered``."""

    req_id: str
    lora_int_id: int
    lora_name: Optional[str]
    prompt_len: int
    response_start: int
    boundary_detail: Optional[str] = None
    mode: Optional[str] = None
    failure: Optional[Tuple[str, str]] = None
    ledgered: bool = False

    def fail(self, error_type: str, detail: str) -> None:
        """Mark the request unsteerable (the FIRST failure names it; later
        ones do not overwrite) and take its facts out of the class masks."""
        if self.failure is None:
            self.failure = (error_type, detail)
        self.response_start = UNLOCATED
        self.prompt_len = POSITION_FACT_SENTINEL

    @property
    def judged(self) -> bool:
        return self.mode is not None


class RequestFactTable:
    """The LIVE requests' facts keyed by engine request id: admitted from
    ``scheduled_new_reqs`` (a re-admission — a request resumed from
    preemption on V2, an aborted id resubmitted — overwrites), RECLAIMED on
    ``finished_req_ids`` (``drop``; the counter is ``reclaimed`` — the
    healthy lifecycle, which a readiness probe of S requests drives to S).
    Bounded like the routing recorder: past ``cap`` a new request is REFUSED
    (``admit`` returns False), ``overflow`` latches and ``overflowed`` counts
    the refusals — the caller ledgers the request when its adapter is
    restricted-mode, so the overflow can never steer wrong rows silently;
    readiness refuses ``overflow`` (never ``reclaimed``)."""

    def __init__(self, cap: int = REQUEST_FACTS_CAP):
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 1:
            raise ResidualSlotError(f"fact table cap must be a positive int, got {cap!r}")
        self.cap = cap
        self._by_id: Dict[str, RequestFacts] = {}
        self.overflow = False
        self.admitted = 0
        self.overflowed = 0
        self.reclaimed = 0
        self.off_contract = 0        # CLIENT scheduler admissions without the engine suffix (readiness refuses any)
        self.warmup = 0              # the engine's own warm-up / profiling rows (ENGINE_WARMUP_ID_RE), tabled, never "admitted"

    def __len__(self) -> int:
        return len(self._by_id)

    def get(self, req_id: Any) -> Optional[RequestFacts]:
        return self._by_id.get(str(req_id))

    def admit(self, facts: RequestFacts, *, warmup: bool = False) -> bool:
        if facts.req_id not in self._by_id and len(self._by_id) >= self.cap:
            if not self.overflow:
                log.error("residual slot: the request fact table holds %d live entries (cap %d) "
                          "— finished requests are not being dropped; latching overflow",
                          len(self._by_id), self.cap)
            self.overflow = True
            self.overflowed += 1
            return False
        self._by_id[facts.req_id] = facts
        if warmup:
            self.warmup += 1
        else:
            self.admitted += 1
        return True

    def note_scheduler_admission(self, req_id: Any) -> None:
        """The ENGINE-ID CONTRACT pin, for a request the SCHEDULER admitted
        (``scheduled_new_reqs`` — a client request the OpenAI layer minted an id
        for): count it OFF the contract when it wears no engine suffix. Never
        for the facts :meth:`RequestSlotBank.expand_token_facts` inserts for
        batch rows the scheduler never announced — the engine's warm-up /
        profiling rows (``_dummy_run`` @0.25.1 builds ``SchedulerOutput.
        make_empty()`` over ``_dummy_req_<i>`` ids with no ``scheduled_new_reqs``;
        C5'' 513982 read 107 of them at readiness and refused a healthy engine)."""
        if not engine_id_on_contract(req_id):
            if not self.off_contract:
                log.error("residual slot: scheduler request id %r wears no engine suffix (vLLM 0.25.1 mints "
                          "<external>-<8 hex>; VLLM_DISABLE_REQUEST_ID_RANDOMIZATION or a vLLM bump?) — "
                          "a failure ledgered under it could never be claimed by the completion id; "
                          "readiness refuses this", req_id)
            self.off_contract += 1

    def drop(self, req_ids: Any) -> int:
        n = 0
        for rid in list(req_ids):
            if self._by_id.pop(str(rid), None) is not None:
                n += 1
        self.reclaimed += n
        return n

    def report(self) -> Dict[str, Any]:
        return {"size": len(self._by_id), "cap": self.cap, "overflow": bool(self.overflow),
                "admitted": self.admitted, "overflowed": self.overflowed, "reclaimed": self.reclaimed,
                "off_contract": self.off_contract, "warmup": self.warmup}


def _lora_binding(lora_request: Any) -> Tuple[int, Optional[str]]:
    """``(lora_int_id, lora_name)`` off a ``LoRARequest`` (``vllm/lora/request.py``
    @0.25.1), ``(0, None)`` for no LoRA — the routing recorder's reading."""
    if lora_request is None:
        return 0, None
    lid = getattr(lora_request, "lora_int_id", None)
    name = getattr(lora_request, "lora_name", None)
    return (int(lid) if isinstance(lid, int) and not isinstance(lid, bool) else -1,
            str(name) if isinstance(name, str) else None)


# ---------------------------------------------------------------------------
# the bank
# ---------------------------------------------------------------------------


@dataclass
class SlotRecord:
    lora_int_id: int
    payload: ResidualPayload


class ResidualSlotBank:
    """The static per-slot buffers (:class:`SlotBuffers`) plus the host-side
    facts: which ``lora_int_id`` sits in which slot with which payload, the
    bound runner, the ledger, and two counters the readiness probe reads
    (mapping updates, op invocations — the eager impl increments the latter,
    so a report taken after a forward shows the op RAN)."""

    def __init__(self, *, n_slots: int, n_boundaries: int, hidden_size: int,
                 max_tokens: int, device: Any, model_dtype: torch.dtype,
                 paired: bool = False):
        for name, v in (("n_slots", n_slots), ("n_boundaries", n_boundaries),
                        ("hidden_size", hidden_size), ("max_tokens", max_tokens)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ResidualSlotError(f"ResidualSlotBank: {name} must be a positive int, got {v!r}")
        if not isinstance(model_dtype, torch.dtype) or not model_dtype.is_floating_point:
            raise ResidualSlotError(f"ResidualSlotBank: model_dtype must be a float dtype, got {model_dtype!r}")
        if not isinstance(paired, bool):
            raise ResidualSlotError(f"ResidualSlotBank: paired must be a bool, got {paired!r}")
        S, L, d = n_slots, n_boundaries, hidden_size
        self.n_slots, self.n_boundaries, self.hidden_size = S, L, d
        self.max_tokens, self.device, self.model_dtype = max_tokens, device, model_dtype
        # THE PAIRED WRITE's buffers exist iff the worker was launched for a paired payload
        # (ENV_FLAG_PAIRED -> `paired`): allocated HERE, once, before the first forward, so the
        # op body's path (SlotBuffers docstring) is fixed before any graph is captured. An
        # unarmed bank allocates exactly what it allocated before the paired write existed.
        paired_bufs: Dict[str, Optional[torch.Tensor]] = dict(paired=None, gate_w2=None, gate_b2=None, direction2=None)
        if paired:
            paired_bufs = dict(paired=torch.zeros(S, dtype=torch.bool, device=device),
                               gate_w2=torch.zeros(S, L, d, dtype=torch.float32, device=device),
                               gate_b2=torch.zeros(S, L, dtype=torch.float32, device=device),
                               direction2=torch.zeros(S, L, d, dtype=torch.float32, device=device))
        self.buffers = SlotBuffers(
            slot_by_token=torch.full((max_tokens,), -1, dtype=torch.int64, device=device),
            active=torch.zeros(S, L, dtype=torch.bool, device=device),
            gated=torch.zeros(S, dtype=torch.bool, device=device),
            skip_bos=torch.zeros(S, dtype=torch.bool, device=device),
            gate_w=torch.zeros(S, L, d, dtype=torch.float32, device=device),
            gate_b=torch.zeros(S, L, dtype=torch.float32, device=device),
            direction=torch.zeros(S, L, d, dtype=torch.float32, device=device),
            vector=torch.zeros(S, L, d, dtype=model_dtype, device=device),
            mode=torch.zeros(S, dtype=torch.int8, device=device),
            prompt_len_by_token=torch.full((max_tokens,), POSITION_FACT_SENTINEL,
                                           dtype=torch.int64, device=device),
            response_start_by_token=torch.full((max_tokens,), POSITION_FACT_SENTINEL,
                                               dtype=torch.int64, device=device),
            **paired_bufs)
        self.records: List[Optional[SlotRecord]] = [None] * S
        self.runner: Any = None
        self.ledger = RequestFailureLedger()
        self.mapping_updates = 0
        self.op_invocations = 0
        # -- stage 2 --
        #: the ONE assistant-header id sequence every restricted-mode payload names
        #: (cross-checked at activation; sticky for the process: one base, one template)
        self.header_ids: Optional[Tuple[int, ...]] = None
        self.facts = RequestFactTable()
        #: requests admitted while their adapter was not yet in a slot (judged post-step)
        self._pending: Dict[str, RequestFacts] = {}
        self.token_fact_updates = 0
        self.steps_seen = 0
        self.faults: List[str] = []
        self.n_faults = 0
        self._n_restricted = 0
        self._n_skip_bos = 0
        self._n_paired = 0

    @property
    def paired_armed(self) -> bool:
        """Whether this bank carries the second-gate buffers (THE PAIRED WRITE)."""
        return self.buffers.gate_w2 is not None

    # -- slots -----------------------------------------------------------------------

    def _check_index(self, index: Any) -> int:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.n_slots:
            raise ResidualSlotError(f"slot index {index!r} outside [0, {self.n_slots})")
        return index

    def clear_slot(self, index: int) -> bool:
        """Zero slot ``index`` (buffers + record). True iff it held a payload.
        Idempotent."""
        index = self._check_index(index)
        b = self.buffers
        b.active[index].zero_()
        b.gated[index] = False
        b.skip_bos[index] = False
        b.mode[index] = MODE_ALL
        b.gate_w[index].zero_()
        b.gate_b[index].zero_()
        b.direction[index].zero_()
        b.vector[index].zero_()
        if b.gate_w2 is not None:                 # the paired buffers, when this bank carries them
            b.paired[index] = False
            b.gate_w2[index].zero_()
            b.gate_b2[index].zero_()
            b.direction2[index].zero_()
        held = self.records[index] is not None
        self.records[index] = None
        self._recount()
        return held

    def _recount(self) -> None:
        recs = [r for r in self.records if r is not None]
        self._n_restricted = sum(r.payload.position_mode != "all" for r in recs)
        self._n_skip_bos = sum(bool(r.payload.skip_bos) for r in recs)
        self._n_paired = sum(bool(r.payload.paired) for r in recs)

    @property
    def restricted_slots(self) -> int:
        return self._n_restricted

    @property
    def paired_slots(self) -> int:
        return self._n_paired

    def _adopt_header_ids(self, ids: Tuple[int, ...], *, index: int, lora_int_id: int) -> None:
        if self.header_ids is None:
            self.header_ids = tuple(ids)
        elif self.header_ids != tuple(ids):
            raise ResidualSlotError(
                f"slot {index} (lora_int_id {lora_int_id}): response_header_ids {list(ids)} "
                f"disagree with the header ids this worker already serves {list(self.header_ids)} "
                "— two chat templates in one worker; the response boundary is located against ONE "
                "header; refusing the activation")

    def clear_all(self) -> int:
        return sum(int(self.clear_slot(i)) for i in range(self.n_slots))

    def load_slot(self, index: int, payload: Optional[ResidualPayload], *,
                  lora_int_id: int) -> None:
        """Fill slot ``index`` from ``payload`` (``None`` -> zeros, inactive:
        a plain LoRA or a capture dummy). The buffers are zeroed first so a
        slot never carries a previous tenant's bytes."""
        index = self._check_index(index)
        self.clear_slot(index)   # first: a refused load leaves an EMPTY slot, never another tenant's
        if payload is None:
            return
        if payload.hidden_size != self.hidden_size:
            raise ResidualSlotError(
                f"slot {index} (lora_int_id {lora_int_id}): payload hidden_size "
                f"{payload.hidden_size} != the model's {self.hidden_size}")
        if payload.n_layers_model != self.n_boundaries:
            raise ResidualSlotError(
                f"slot {index} (lora_int_id {lora_int_id}): payload n_layers_model "
                f"{payload.n_layers_model} != the model's {self.n_boundaries} decoder blocks")
        mode = payload.position_mode
        if mode not in POSITION_MODES:
            raise ResidualSlotError(
                f"slot {index} (lora_int_id {lora_int_id}): position mode {mode!r} is not one "
                f"of {list(POSITION_MODES)}")
        if mode != "all" and not STAGE2_POSITION_CLASSES:
            raise ResidualSlotError(
                f"slot {index} (lora_int_id {lora_int_id}): position mode "
                f"{mode!r} needs the stage-2 position class, which this "
                "worker does not carry (STAGE2_POSITION_CLASSES is False) — the disposition "
                "should have skipped this cell; refusing rather than serving it at every position")
        header_ids = tuple(payload.response_header_ids)
        if mode in BOUNDARY_MODES and not header_ids:
            raise ResidualSlotError(
                f"slot {index} (lora_int_id {lora_int_id}): position mode {mode!r} needs the "
                "assistant header's token ids (response_header_ids) to locate every request's "
                "response boundary, and the payload carries none — the exporter must derive them "
                "from the served template; refusing rather than serving a boundary it cannot locate")
        if header_ids:
            self._adopt_header_ids(header_ids, index=index, lora_int_id=int(lora_int_id))
        if payload.paired and not self.paired_armed:
            raise ResidualSlotError(
                f"slot {index} (lora_int_id {lora_int_id}): a PAIRED payload (op {payload.op!r}, kind "
                f"{payload.kind!r}) on a bank without second-gate buffers — this worker was launched "
                f"without {ENV_FLAG_PAIRED}=1 (the composer sets it iff a served adapter's payload is the "
                "paired op), so the op body is the single-gate one and would serve ONE gate under the "
                "cell's name; refusing the activation")
        b = self.buffers
        for l, bd in zip(payload.layers, payload.boundaries):
            if payload.gated:
                b.gate_w[index, bd].copy_(payload.gate_w[l].to(self.device))
                b.gate_b[index, bd] = payload.gate_b[l].reshape(()).to(self.device)
                b.direction[index, bd].copy_(payload.direction[l].to(self.device))
                if payload.paired:
                    b.gate_w2[index, bd].copy_(payload.gate_w2[l].to(self.device))
                    b.gate_b2[index, bd] = payload.gate_b2[l].reshape(()).to(self.device)
                    b.direction2[index, bd].copy_(payload.direction2[l].to(self.device))
            else:
                # the lens's order: the vector is kept in the MODEL dtype and the
                # baked coeff multiplies it there (``addition = vec * eff``)
                vec = payload.vector[l].to(self.device).to(self.model_dtype)
                b.vector[index, bd].copy_(vec * float(payload.coeff))
            b.active[index, bd] = True
        b.gated[index] = bool(payload.gated)
        b.skip_bos[index] = bool(payload.skip_bos)
        b.mode[index] = MODE_CODES[mode]
        if b.paired is not None:
            b.paired[index] = bool(payload.paired)
        self.records[index] = SlotRecord(lora_int_id=int(lora_int_id), payload=payload)
        self._recount()

    def record_for_lora(self, lora_int_id: int) -> Optional[SlotRecord]:
        """The residual slot record bound to ``lora_int_id`` (``None``: a plain
        LoRA, no LoRA, or an adapter not resident in any slot)."""
        for rec in self.records:
            if rec is not None and rec.lora_int_id == lora_int_id:
                return rec
        return None

    # -- routing ---------------------------------------------------------------------

    def update_mapping(self, token_lora_indices: torch.Tensor, n: int) -> None:
        """Refill ``slot_by_token`` from the punica base indices (item 3):
        every row -1, then the first ``n`` copied. Called after EVERY
        ``_update_base_metadata`` — base-only batches included."""
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ResidualSlotError(f"update_mapping: n={n!r} is not a non-negative int")
        if n > self.max_tokens:
            raise ResidualSlotError(
                f"update_mapping: mapping length {n} exceeds slot_by_token's {self.max_tokens}")
        sbt = self.buffers.slot_by_token
        sbt.fill_(-1)
        if n:
            sbt[:n].copy_(token_lora_indices[:n])
        self.mapping_updates += 1

    def bind_runner(self, runner: Any) -> None:
        if runner is None:
            raise ResidualSlotError("bind_runner: runner is None")
        if self.runner is None:
            self.runner = runner
        elif self.runner is not runner:
            raise ResidualSlotError(
                "bind_runner: the bank is already bound to another model runner — one runner "
                "per worker process; refusing to rebind")

    def _require_runner(self, where: str) -> Any:
        if self.runner is None:
            raise ResidualSlotError(
                f"{where}: the bank is bound to no model runner — load_lora_model's wrapper "
                "binds it (fact 11) and must run before any step; refusing")
        return self.runner

    # -- the per-request facts (stage 2) --------------------------------------------

    def record_fault(self, text: str) -> None:
        """A bookkeeping fault (a runner-state disagreement, a scheduled request
        with no facts, a foreign runner): counted, the first few kept verbatim
        for the report; readiness refuses a non-zero count."""
        self.n_faults += 1
        if len(self.faults) < MAX_FAULTS_KEPT:
            self.faults.append(str(text)[:400])
        log.error("residual slot FAULT: %s", text)

    def locate_response_start(self, prompt_token_ids: Any) -> Tuple[int, Optional[str]]:
        """``(response_start, detail)``: the LAST occurrence of the served
        header ids in ``prompt_token_ids`` (``span_locator.rfind`` — header +
        any seed + every generated row, the lens's declared semantics),
        else ``UNLOCATED`` with the reason."""
        if prompt_token_ids is None:
            return UNLOCATED, "the request carries no prompt_token_ids (prompt embeds?)"
        if self.header_ids is None:
            return UNLOCATED, ("no header ids are known to this worker: no restricted-mode "
                               "slot has been activated")
        start = _rfind(prompt_token_ids, self.header_ids)
        if start < 0:
            return UNLOCATED, (f"the assistant header ids {list(self.header_ids)} occur nowhere "
                               f"in the {len(prompt_token_ids)}-token prompt")
        return int(start), None

    def admit_request(self, new_req: Any) -> RequestFacts:
        """Admit one ``scheduled_new_reqs`` entry (``NewRequestData`` @0.25.1:
        ``req_id``, ``prompt_token_ids``, ``lora_request``): compute its facts,
        table them, judge them against the slot its adapter sits in (or defer
        the judgement to :meth:`post_step` when the adapter is not resident
        yet — the step activates it)."""
        rid = str(new_req.req_id)
        lid, name = _lora_binding(getattr(new_req, "lora_request", None))
        ids = getattr(new_req, "prompt_token_ids", None)
        prompt_len = int(len(ids)) if ids is not None else POSITION_FACT_SENTINEL
        start, detail = self.locate_response_start(ids)
        facts = RequestFacts(req_id=rid, lora_int_id=lid, lora_name=name,
                             prompt_len=prompt_len, response_start=start, boundary_detail=detail)
        warmup = ENGINE_WARMUP_ID_RE.fullmatch(rid) is not None      # the engine's own rows: tabled, never a client request
        admitted = self.facts.admit(facts, warmup=warmup)
        if admitted and not warmup:
            self.facts.note_scheduler_admission(rid)      # the engine-id contract pin (tabled CLIENT admissions only)
        if not admitted:
            facts.fail(ERROR_FACTS_OVERFLOW,
                       f"the request fact table holds {len(self.facts)} live entries (cap "
                       f"{self.facts.cap}); this request's facts could not be kept")
            self.record_fault(f"fact table overflow admitting request {rid!r} (lora_int_id {lid})")
        self._pending.pop(rid, None)
        if not self._judge(facts):
            self._pending[rid] = facts
        return facts

    def _judge(self, facts: RequestFacts) -> bool:
        """Bind ``facts`` to its adapter's position mode and LEDGER it when the
        mode is restricted and the request carries a failure. True once
        judged; False while the adapter is not resident in any slot yet."""
        if facts.lora_int_id <= 0:
            facts.mode = ""                      # no LoRA: nothing to serve, nothing to judge
            return True
        rec = self.record_for_lora(facts.lora_int_id)
        if rec is None:
            return False
        facts.mode = rec.payload.position_mode
        self._ledger_if_failed(facts)
        return True

    def _ledger_if_failed(self, facts: RequestFacts) -> None:
        """LEDGER ``facts`` when its slot's mode is restricted and the request
        cannot be served as that mode asks: a hard ``failure`` (every restricted
        mode), or an unlocated boundary on a ``BOUNDARY_MODES`` slot. Once."""
        if facts.ledgered or facts.mode not in RESTRICTED_MODES:
            return
        if facts.failure is not None:
            error_type, detail = facts.failure
        elif facts.mode in BOUNDARY_MODES and facts.response_start == UNLOCATED:
            error_type, detail = ERROR_BOUNDARY_UNLOCATED, (facts.boundary_detail or "")
        else:
            return
        self.ledger.record(
            facts.req_id, lora_int_id=int(facts.lora_int_id), cell=facts.lora_name,
            error_type=error_type,
            prompt_len=(None if facts.prompt_len < 0 else int(facts.prompt_len)),
            header_len=(None if self.header_ids is None else len(self.header_ids)),
            detail=detail)
        facts.ledgered = True
        log.error("residual slot: request %s (lora_int_id %d, cell %r, mode %r) FAILED %s: %s — "
                  "served unsteered; its completion must be refused at the proxy "
                  "(sisteer_take_request_failure)", facts.req_id, facts.lora_int_id,
                  facts.lora_name, facts.mode, error_type, detail)

    def pre_step(self, scheduler_output: Any) -> None:
        """The ``execute_model`` pre-wrapper's body: drop the finished requests'
        facts (the ledger keeps its entries), admit this step's new requests.
        Preempted requests are NOT dropped (V1 resumes them as cached requests
        with the same prompt; V2 re-admits them through ``scheduled_new_reqs``,
        which overwrites)."""
        self.steps_seen += 1
        finished = getattr(scheduler_output, "finished_req_ids", None) or ()
        if finished:
            self.facts.drop(finished)
            for rid in finished:
                self._pending.pop(str(rid), None)
        for entry in list(scheduler_output.scheduled_new_reqs):
            self.admit_request(entry)

    def post_step(self) -> None:
        """After the original ``execute_model``: every adapter this step routed
        to is resident now, so a request admitted before its slot existed is
        judged here (a plain LoRA — no residual record — is judged as such)."""
        for facts in self._pending.values():
            if not self._judge(facts):
                # still no record after the step that activated its adapter: a plain LoRA
                facts.mode = ""
        self._pending.clear()

    def expand_token_facts(self, req_ids: Sequence[Any], lora_ids: Any, num_scheduled_tokens: Any,
                           runner_prompt_lens: Optional[Any] = None) -> int:
        """The PER-TOKEN expansion, called on EVERY step by the
        runner-specific mapping wrapper with the batch's request ids IN BATCH
        ORDER, the runner's own per-request LoRA ids and prompt lengths (the
        cross-check), and the per-request scheduled token counts:
        ``fill_(SENTINEL)`` then ``copy_`` of ``np.repeat(values_by_req,
        num_scheduled_tokens)`` into ``prompt_len_by_token`` /
        ``response_start_by_token``. A request the runner schedules that the
        fact table does not know, or whose runner-side prompt length / LoRA id
        disagrees with its facts, is a recorded FAULT and is FAILED (unsteered
        on a restricted-mode slot, ledgered). Returns the token count."""
        ids = [str(r) for r in req_ids]
        n = len(ids)
        nst = np.asarray(num_scheduled_tokens, dtype=np.int64).reshape(-1)
        if nst.shape[0] < n:
            raise ResidualSlotError(
                f"expand_token_facts: {n} requests but {nst.shape[0]} scheduled-token counts")
        nst = nst[:n]
        lids = np.asarray(lora_ids).reshape(-1)[:n] if lora_ids is not None else None
        rpl = (np.asarray(runner_prompt_lens).reshape(-1)[:n]
               if runner_prompt_lens is not None else None)
        pl = np.full(n, POSITION_FACT_SENTINEL, dtype=np.int64)
        rs = np.full(n, POSITION_FACT_SENTINEL, dtype=np.int64)
        for i, rid in enumerate(ids):
            facts = self.facts.get(rid)
            lid_runner = int(lids[i]) if lids is not None else None
            if facts is None:
                # scheduled but not in the table (never admitted through scheduled_new_reqs, or
                # refused by the cap): a drift, recorded EVERY step it recurs. Fail it closed on
                # whatever adapter the runner says it wears — a transient judgement, never a
                # table write (the table's cap is what may have refused it).
                self.record_fault(f"request {rid!r} is scheduled but has no facts (not admitted "
                                  f"through scheduled_new_reqs, or refused by the cap); runner "
                                  f"lora_int_id={lid_runner}")
                facts = RequestFacts(req_id=rid, lora_int_id=(lid_runner or 0), lora_name=None,
                                     prompt_len=POSITION_FACT_SENTINEL, response_start=UNLOCATED)
                facts.fail(ERROR_FACTS_MISSING, "the request was scheduled without facts (never "
                                                "admitted through scheduled_new_reqs, or refused by "
                                                "the fact table's cap)")
                self._judge(facts)
                continue
            if lid_runner is not None and lid_runner != facts.lora_int_id:
                self.record_fault(f"request {rid!r}: the runner binds lora_int_id {lid_runner} but "
                                  f"the facts were admitted under {facts.lora_int_id}")
                facts.fail(ERROR_LORA_ID_DISAGREES,
                           f"runner lora_int_id {lid_runner} != admitted {facts.lora_int_id}")
                facts.lora_int_id = lid_runner        # judge it on the adapter it actually wears
                facts.mode = None
                self._judge(facts)
            if rpl is not None and facts.prompt_len >= 0 and int(rpl[i]) != facts.prompt_len:
                self.record_fault(f"request {rid!r}: the runner's prompt length {int(rpl[i])} != "
                                  f"the admitted {facts.prompt_len}")
                facts.fail(ERROR_PROMPT_LEN_DISAGREES,
                           f"runner prompt_len {int(rpl[i])} != admitted {facts.prompt_len}")
                self._ledger_if_failed(facts)
            if not facts.judged:
                self._judge(facts)
            pl[i] = facts.prompt_len
            rs[i] = facts.response_start
        T = int(nst.sum()) if n else 0
        if T > self.max_tokens:
            raise ResidualSlotError(
                f"expand_token_facts: {T} scheduled tokens exceed the buffers' {self.max_tokens}")
        b = self.buffers
        b.prompt_len_by_token.fill_(POSITION_FACT_SENTINEL)
        b.response_start_by_token.fill_(POSITION_FACT_SENTINEL)
        if T:
            b.prompt_len_by_token[:T].copy_(torch.from_numpy(np.repeat(pl, nst)))
            b.response_start_by_token[:T].copy_(torch.from_numpy(np.repeat(rs, nst)))
        self.token_fact_updates += 1
        return T

    # -- the op body -----------------------------------------------------------------

    def apply(self, hidden: torch.Tensor, residual: Optional[torch.Tensor],
              positions: Optional[torch.Tensor], boundary: int) -> torch.Tensor:
        if isinstance(boundary, bool) or not isinstance(boundary, int) \
                or not 0 <= boundary < self.n_boundaries:
            raise ResidualSlotError(
                f"residual slot: boundary {boundary!r} outside the model's "
                f"{self.n_boundaries} blocks")
        if hidden.dim() == 2 and hidden.shape[1] != self.hidden_size:
            raise ResidualSlotError(
                f"residual slot: hidden width {hidden.shape[1]} != the bank's {self.hidden_size}")
        if positions is None and (self._n_restricted or self._n_skip_bos):
            # a host fact (the layer's call shape), not a tensor read: graph-safe
            raise ResidualSlotError(
                f"residual slot: the forward at boundary {boundary} carries no positions while "
                f"{self._n_restricted} restricted-mode and {self._n_skip_bos} skip_bos slot(s) are "
                "loaded — a position class it cannot evaluate is never served as 'all'; refusing")
        return residual_slot_forward(hidden, residual, positions, boundary, self.buffers)

    # -- facts -----------------------------------------------------------------------

    def slots_report(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for i, rec in enumerate(self.records):
            if rec is not None:
                out[str(rec.lora_int_id)] = {"slot": i, **rec.payload.summary()}
        return out

    def report(self) -> Dict[str, Any]:
        return {
            "n_slots": self.n_slots, "n_boundaries": self.n_boundaries,
            "hidden_size": self.hidden_size, "max_tokens": self.max_tokens,
            "model_dtype": str(self.model_dtype), "device": str(self.device),
            "slots_loaded": sum(r is not None for r in self.records),
            "mapping_updates": self.mapping_updates, "op_invocations": self.op_invocations,
            "runner_bound": self.runner is not None,
            # THE PAIRED WRITE: whether the second-gate buffers exist and how many slots use them
            "paired_armed": self.paired_armed, "paired_slots": self._n_paired,
            **self.ledger.report(),
            # stage 2
            "header_ids": (None if self.header_ids is None else list(self.header_ids)),
            "header_len": (None if self.header_ids is None else len(self.header_ids)),
            "restricted_slots": self._n_restricted,
            "fact_table": self.facts.report(),
            "pending_judgements": len(self._pending),
            "token_fact_updates": self.token_fact_updates,
            "steps_seen": self.steps_seen,
            "n_faults": self.n_faults, "faults": list(self.faults),
        }


#: the ONE bank of this worker process (created by the manager patch)
_BANK: Optional[ResidualSlotBank] = None


def bank() -> Optional[ResidualSlotBank]:
    return _BANK


def _publish_bank(b: ResidualSlotBank) -> None:
    global _BANK
    if _BANK is not None:
        raise ResidualSlotError(
            "a second ResidualSlotBank is being created in this process — vLLM builds ONE "
            "LoRAModelManager per worker; refusing to shadow the live bank")
    _BANK = b


def _require_bank(where: str) -> ResidualSlotBank:
    if _BANK is None:
        raise ResidualSlotError(
            f"{where}: no ResidualSlotBank exists — the LoRAModelManager patch did not run "
            "before this call (fact 11: the bank is created by the manager's constructor "
            "inside load_lora_model; nothing may read it earlier)")
    return _BANK


def take_request_failure(req_id: Any) -> Optional[Dict[str, Any]]:
    """The sixth RPC's body: claim-and-clear the ledger entry for
    ``req_id`` (``None`` when none); the fault payload for EVERY id once the
    ledger latched. An armed worker with no bank answers a fault too — a
    claim that cannot be judged must never read as "no failure"."""
    if _BANK is None:
        if PATCH_REPORT.armed:
            return {"fault": "bank_missing", "req_id": str(req_id)}
        return None
    return _BANK.ledger.claim(req_id)


# ---------------------------------------------------------------------------
# the custom op
# ---------------------------------------------------------------------------


def _op_impl(hidden: torch.Tensor, residual: Optional[torch.Tensor],
             positions: Optional[torch.Tensor], boundary: int) -> torch.Tensor:
    """The eager impl of ``torch.ops.vllm.sisteer_residual_slot``: the bank's
    :meth:`ResidualSlotBank.apply` over the live buffers. Runs under torch.
    compile as one opaque node and inside CUDA-graph capture (no host reads
    of tensor data)."""
    b = _BANK
    if b is None:
        raise ResidualSlotError(
            f"{OP_QUALNAME} fired with no ResidualSlotBank — a decoder layer ran before the "
            "LoRA manager was constructed (fact 11) or the manager patch is not installed")
    b.op_invocations += 1
    return b.apply(hidden, residual, positions, boundary)


def _op_fake(hidden: torch.Tensor, residual: Optional[torch.Tensor],
             positions: Optional[torch.Tensor], boundary: int) -> torch.Tensor:
    return torch.empty_like(hidden)


_FALLBACK_LIB: Any = None


def op_registered() -> bool:
    try:
        getattr(torch.ops.vllm, OP_NAME)
    except (AttributeError, RuntimeError):
        return False
    return True


def ensure_op_registered() -> Any:
    """Register ``torch.ops.vllm.sisteer_residual_slot`` once and return the
    op packet. With vllm importable: through ``vllm.utils.torch_utils.
    direct_register_custom_op`` (fact 12: into vLLM's own ``vllm`` library,
    dispatched to the platform's key, fake impl registered). Without vllm
    (the CPU tests): a FRAGMENT library of the SAME namespace held by
    this module, CPU dispatch — so the name, the schema and the fake impl
    are the ones the worker will carry."""
    global _FALLBACK_LIB
    if op_registered():
        return getattr(torch.ops.vllm, OP_NAME)
    try:
        from vllm.utils.torch_utils import direct_register_custom_op
    except ImportError:
        direct_register_custom_op = None
    if direct_register_custom_op is not None:
        direct_register_custom_op(op_name=OP_NAME, op_func=_op_impl, mutates_args=[],
                                  fake_impl=_op_fake)
    else:
        from torch.library import Library, infer_schema

        lib = Library("vllm", "FRAGMENT")
        schema = infer_schema(_op_impl, mutates_args=())
        lib.define(OP_NAME + schema)
        lib.impl(OP_NAME, _op_impl, dispatch_key="CPU")
        lib._register_fake(OP_NAME, _op_fake)
        _FALLBACK_LIB = lib
    if not op_registered():
        raise ResidualSlotError(f"registering {OP_QUALNAME} did not make it resolvable")
    return getattr(torch.ops.vllm, OP_NAME)


# ---------------------------------------------------------------------------
# the class patches
# ---------------------------------------------------------------------------


@dataclass
class PatchTargets:
    """The classes the patches land on — vLLM's (:func:`vllm_patch_targets`)
    or a test's fakes carrying the wheel's signatures."""

    lora_model_cls: type
    manager_cls: type
    punica_cls: type
    runner_mixin_cls: type
    decoder_layer_classes: Tuple[type, ...]
    #: ``(manager, bound constructor arguments) -> (hidden_size, n_layers,
    #: model_dtype)``, read at manager construction (the wheel's manager keeps
    #: only parts of the ``vllm_config`` it receives, so the wrapper binds the
    #: original's arguments by signature and hands them over)
    model_info: Callable[[Any, Mapping[str, Any]], Tuple[int, int, torch.dtype]]
    #: the op the forward wrappers call (``torch.ops.vllm.sisteer_residual_slot``)
    op: Any
    #: stage 2 (§1.8): the GPU model runner classes whose ``execute_model`` takes
    #: the pre-wrapper (both of ``serving_attestation.MODEL_RUNNER_CLASSES``)
    runner_classes: Tuple[type, ...] = ()
    #: stage 2: V2's ``LoraState`` (its ``make_lora_inputs`` takes the per-token
    #: expansion); V1's rides on ``runner_mixin_cls.set_active_loras``
    lora_state_cls: Optional[type] = None


@dataclass
class PatchReport:
    armed: bool = False
    op_name: Optional[str] = None
    installs: int = 0
    patched_classes: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    machinery: Dict[str, str] = field(default_factory=dict)
    #: ``(class qualname, boundary, forward_wrapped)`` per constructed decoder layer
    instances: List[Tuple[str, int, bool]] = field(default_factory=list)
    error: Optional[str] = None

    def hooked_boundaries(self) -> List[int]:
        return sorted({b for _, b, wrapped in self.instances if wrapped})

    def as_dict(self) -> Dict[str, Any]:
        return {"armed": self.armed, "op_name": self.op_name, "installs": self.installs,
                "patched_classes": dict(self.patched_classes),
                "machinery": dict(self.machinery),
                "n_layers_hooked": len(self.hooked_boundaries()),
                "instances": [list(t) for t in self.instances], "error": self.error}


PATCH_REPORT = PatchReport()


def _qual(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _is_wrapped(attr: Any) -> bool:
    fn = getattr(attr, "__func__", attr)
    return bool(getattr(fn, WRAP_MARK, False))


def _mark(fn: Callable, orig: Any) -> Callable:
    setattr(fn, WRAP_MARK, True)
    fn._sisteer_orig = orig  # type: ignore[attr-defined]
    return fn


def layer_index_from_prefix(prefix: Any) -> Optional[int]:
    """``vllm.model_executor.models.utils.extract_layer_index``'s rule — the
    ONE integer among the dotted components (``model.layers.12`` -> 12);
    ``None`` when the prefix carries no integer or more than one."""
    if not isinstance(prefix, str) or not prefix:
        return None
    ints = []
    for part in prefix.split("."):
        try:
            ints.append(int(part))
        except ValueError:
            continue
    return ints[0] if len(ints) == 1 else None


def _positions_index(sig: inspect.Signature) -> Optional[int]:
    """The positional index (after ``self``) of a ``positions`` parameter."""
    params = [p for p in list(sig.parameters.values())[1:]
              if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    for i, p in enumerate(params):
        if p.name == "positions":
            return i
    return None


def _make_forward_wrapper(orig: Callable, cls_name: str, positions_index: Optional[int],
                          op: Any) -> Callable:
    def forward(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        boundary = getattr(self, "_sisteer_boundary", None)
        if boundary is None:
            boundary = getattr(self, "layer_idx", None)
        if isinstance(boundary, bool) or not isinstance(boundary, int):
            raise ResidualSlotError(
                f"residual slot: {cls_name} instance carries neither a patched boundary nor "
                "an int layer_idx — the layer was constructed before the patch was installed, "
                "or its constructor received no prefix; refusing at the first forward")
        positions = kwargs.get("positions")
        if positions is None and positions_index is not None and len(args) > positions_index:
            positions = args[positions_index]
        if isinstance(out, tuple):
            hidden, residual = out[0], (out[1] if len(out) > 1 else None)
            new_hidden = op(hidden, residual, positions, boundary)
            return (new_hidden,) + tuple(out[1:])
        return op(out, None, positions, boundary)

    forward.__name__ = "forward"
    forward.__qualname__ = f"{cls_name}.forward"
    forward.__doc__ = getattr(orig, "__doc__", None)
    return _mark(forward, orig)


def _make_init_wrapper(orig: Callable, cls: type, report: PatchReport) -> Callable:
    sig = inspect.signature(orig)
    cls_name = _qual(cls)

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        try:
            bound = sig.bind_partial(self, *args, **kwargs)
            arguments = bound.arguments
        except TypeError:
            arguments = {}
        candidates: Dict[str, int] = {}
        own = getattr(self, "layer_idx", None)
        if isinstance(own, int) and not isinstance(own, bool):
            candidates["layer_idx attribute"] = own
        arg = arguments.get("layer_idx")
        if isinstance(arg, int) and not isinstance(arg, bool):
            candidates["layer_idx argument"] = arg
        from_prefix = layer_index_from_prefix(arguments.get("prefix"))
        if from_prefix is not None:
            candidates["prefix"] = from_prefix
        if not candidates:
            # not a refusal here: the FORWARD refuses naming the class (a layer that
            # never runs a forward — a skeleton — may be constructed without one)
            return
        if len(set(candidates.values())) != 1:
            raise ResidualSlotError(
                f"residual slot: {cls_name}: the boundary sources disagree ({candidates}); "
                "refusing to pick one")
        boundary = next(iter(candidates.values()))
        object.__setattr__(self, "_sisteer_boundary", int(boundary))
        report.instances.append((cls_name, int(boundary),
                                 _is_wrapped(getattr(type(self), "forward", None))))

    __init__.__name__ = "__init__"
    __init__.__qualname__ = f"{cls_name}.__init__"
    return _mark(__init__, orig)


def _make_from_local_checkpoint(orig_func: Callable) -> classmethod:
    def from_local_checkpoint(cls, lora_dir, expected_lora_modules, peft_helper, *args,
                              skip_prefixes=None, **kwargs):
        prefixes = list(skip_prefixes or [])
        if PAYLOAD_PREFIX not in prefixes:
            prefixes.append(PAYLOAD_PREFIX)
        lora = orig_func(cls, lora_dir, expected_lora_modules, peft_helper, *args,
                         skip_prefixes=prefixes, **kwargs)
        lora.sisteer_residual = read_residual_payload(Path(lora_dir))
        return lora

    from_local_checkpoint.__name__ = "from_local_checkpoint"
    return classmethod(_mark(from_local_checkpoint, orig_func))


def _make_manager_init(orig: Callable, model_info: Callable) -> Callable:
    sig = inspect.signature(orig)

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        try:
            arguments = dict(sig.bind_partial(self, *args, **kwargs).arguments)
        except TypeError:
            arguments = {}
        d, n_layers, dtype = model_info(self, arguments)
        b = ResidualSlotBank(n_slots=int(self.lora_slots), n_boundaries=int(n_layers),
                             hidden_size=int(d), max_tokens=int(self.max_num_batched_tokens),
                             device=self.device, model_dtype=dtype,
                             paired=_paired_from_env())    # THE PAIRED WRITE: decided before any forward
        self.sisteer_bank = b
        _publish_bank(b)

    return _mark(__init__, orig)


def _make_activate_adapter(orig: Callable) -> Callable:
    def activate_adapter(self, lora_id):
        activated = orig(self, lora_id)
        if activated:
            index = self.lora_index_to_id.index(lora_id)
            lora_model = self._registered_adapters[lora_id]
            payload = getattr(lora_model, "sisteer_residual", None)
            self.sisteer_bank.load_slot(index, payload, lora_int_id=int(lora_id))
        return activated

    return _mark(activate_adapter, orig)


def _make_deactivate_adapter(orig: Callable) -> Callable:
    def _deactivate_adapter(self, lora_id):
        # capture the slot BEFORE the original erases lora_index_to_id[index]
        try:
            index = self.lora_index_to_id.index(lora_id)
        except ValueError:
            index = None
        if index is not None:
            self.sisteer_bank.clear_slot(index)
        return orig(self, lora_id)

    return _mark(_deactivate_adapter, orig)


def _make_remove_all_adapters(orig: Callable) -> Callable:
    def remove_all_adapters(self):
        # the original resets lora_index_to_id wholesale BEFORE clearing the
        # active set, so the per-slot capture above would find nothing: zero
        # every slot first
        self.sisteer_bank.clear_all()
        return orig(self)

    return _mark(remove_all_adapters, orig)


def _make_update_base_metadata(orig: Callable) -> Callable:
    def _update_base_metadata(self, mapping, lora_index_to_id, max_loras, vocab_size):
        out = orig(self, mapping, lora_index_to_id, max_loras, vocab_size)
        b = _require_bank("PunicaWrapperBase._update_base_metadata")
        n = int(self.indices_len[0])
        b.update_mapping(self._token_lora_indices, n)
        return out

    return _mark(_update_base_metadata, orig)


def _make_load_lora_model(orig: Callable) -> Callable:
    def load_lora_model(self, model, vllm_config, device):
        out = orig(self, model, vllm_config, device)
        _require_bank("LoRAModelRunnerMixin.load_lora_model").bind_runner(self)
        return out

    return _mark(load_lora_model, orig)


def _make_execute_model(orig: Callable, cls_name: str) -> Callable:
    """The CLASS-LEVEL pre-wrapper on ``execute_model(scheduler_output, ...)``
    of BOTH GPU model runners (§1.8 item 6): before the original, the bank
    drops the finished requests' facts and admits the new ones off the
    ``scheduler_output`` (the routing recorder reads the same object off the
    same argument — its INSTANCE wrapper composes with this one); after it,
    the deferred judgements. A dummy run (V2 ``dummy_run=True``: profile,
    capture) or a call with no scheduler output touches nothing. Refuses
    loudly with no bank or an unbound runner (fact 11)."""
    where = f"{cls_name}.execute_model"

    def execute_model(self, *args, **kwargs):
        b = _require_bank(where)
        b._require_runner(where)
        so = kwargs["scheduler_output"] if "scheduler_output" in kwargs else (args[0] if args else None)
        live = so is not None and not bool(kwargs.get("dummy_run", False))
        if live:
            if b.runner is not self:
                b.record_fault(f"{where}: a step ran on a runner that is not the one the bank bound")
            b.pre_step(so)
        out = orig(self, *args, **kwargs)
        if live:
            b.post_step()
        return out

    execute_model.__name__ = "execute_model"
    execute_model.__qualname__ = f"{cls_name}.execute_model"
    return _mark(execute_model, orig)


def _make_set_active_loras(orig: Callable) -> Callable:
    """V1's per-token expansion (§1.8 item 7): ``LoRAModelRunnerMixin.
    set_active_loras(input_batch, num_scheduled_tokens, ...)`` — after the
    original (which activates this step's adapters and, through punica,
    refills ``slot_by_token``), the batch's request ids / LoRA ids / prompt
    lengths off the ``InputBatch`` (``req_ids``, ``request_lora_mapping``,
    ``num_prompt_tokens``; ``gpu_input_batch.py`` @0.25.1) expand into the
    per-token fact buffers."""
    where = "LoRAModelRunnerMixin.set_active_loras"

    def set_active_loras(self, input_batch, num_scheduled_tokens, *args, **kwargs):
        out = orig(self, input_batch, num_scheduled_tokens, *args, **kwargs)
        b = _require_bank(where)
        b._require_runner(where)
        n = int(input_batch.num_reqs)
        b.expand_token_facts(list(input_batch.req_ids)[:n], input_batch.request_lora_mapping[:n],
                             num_scheduled_tokens, input_batch.num_prompt_tokens[:n])
        return out

    set_active_loras.__name__ = "set_active_loras"
    return _mark(set_active_loras, orig)


def _make_make_lora_inputs(orig: Callable) -> Callable:
    """V2's per-token expansion (§1.8 item 7): ``LoraState.make_lora_inputs(
    req_ids, idx_mapping, num_scheduled_tokens)`` (``gpu/lora_utils.py``
    @0.25.1, called from ``execute_model`` right before ``_set_active_loras``)
    — after the original, the batch's request ids expand into the per-token
    fact buffers, cross-checked against ``lora_ids[idx_mapping]`` and the
    bound runner's ``req_states.prompt_len.np[idx_mapping]`` (fact 11)."""
    where = "LoraState.make_lora_inputs"

    def make_lora_inputs(self, req_ids, idx_mapping, num_scheduled_tokens):
        out = orig(self, req_ids, idx_mapping, num_scheduled_tokens)
        b = _require_bank(where)
        runner = b._require_runner(where)
        idx = np.asarray(idx_mapping)
        b.expand_token_facts(req_ids, self.lora_ids[idx], num_scheduled_tokens,
                             runner.req_states.prompt_len.np[idx])
        return out

    make_lora_inputs.__name__ = "make_lora_inputs"
    return _mark(make_lora_inputs, orig)


def install_patches(targets: PatchTargets, report: Optional[PatchReport] = None) -> PatchReport:
    """Install the class patches on ``targets``. IDEMPOTENT: every original is
    snapshotted off the class's OWN ``__dict__`` before any wrapper lands; a
    method already carrying :data:`WRAP_MARK` is left alone; a class that
    does not DEFINE a method (inherits it) is not wrapped for it and is so
    recorded. Returns the report (``patched_classes`` per decoder class,
    ``machinery`` per LoRA-path method)."""
    rep = report if report is not None else PatchReport()
    rep.installs += 1
    plan: List[Tuple[type, str, Any, Callable[[Any], Any], str]] = []

    def want(cls: type, name: str, make: Callable[[Any], Any], *, required: bool,
             key: Optional[str] = None) -> None:
        own = cls.__dict__.get(name)
        key = key or f"{cls.__qualname__}.{name}"
        if own is None:
            if required:
                raise ResidualSlotError(
                    f"residual slot: {_qual(cls)} does not define {name!r} in its own __dict__ "
                    "— not the pinned 0.25.1 surface; refusing to patch an inherited method")
            rep.machinery.setdefault(key, "inherits")
            return
        if _is_wrapped(own):
            rep.machinery[key] = "wrapped"
            return
        plan.append((cls, name, own, make, key))

    # 1. snapshot every original first (nothing installed yet)
    want(targets.lora_model_cls, "from_local_checkpoint",
         lambda o: _make_from_local_checkpoint(o.__func__), required=True)
    want(targets.manager_cls, "__init__",
         lambda o: _make_manager_init(o, targets.model_info), required=True)
    want(targets.manager_cls, "activate_adapter", _make_activate_adapter, required=True)
    want(targets.manager_cls, "_deactivate_adapter", _make_deactivate_adapter, required=True)
    want(targets.manager_cls, "remove_all_adapters", _make_remove_all_adapters, required=True)
    want(targets.punica_cls, "_update_base_metadata", _make_update_base_metadata, required=True)
    want(targets.runner_mixin_cls, "load_lora_model", _make_load_lora_model, required=True)
    # stage 2 (§1.8): the two runners' execute_model (keyed by MODULE-qualified name — both
    # wheel classes are called GPUModelRunner), V1's set_active_loras, V2's make_lora_inputs
    for rcls in targets.runner_classes:
        want(rcls, "execute_model", (lambda o, c=rcls: _make_execute_model(o, _qual(c))),
             required=True, key=f"{_qual(rcls)}.execute_model")
    want(targets.runner_mixin_cls, "set_active_loras", _make_set_active_loras, required=True)
    if targets.lora_state_cls is not None:
        want(targets.lora_state_cls, "make_lora_inputs", _make_make_lora_inputs, required=True)
    else:
        rep.machinery.setdefault("LoraState.make_lora_inputs", "absent")
    layer_plan: List[Tuple[type, str, Any]] = []
    for cls in targets.decoder_layer_classes:
        entry = rep.patched_classes.setdefault(_qual(cls), {})
        for name in ("forward", "__init__"):
            own = cls.__dict__.get(name)
            if own is None:
                # inherited: name the class it comes from
                owner = next((k for k in cls.__mro__[1:] if name in k.__dict__), None)
                entry[name] = f"inherits:{owner.__qualname__ if owner else '?'}"
                continue
            if _is_wrapped(own):
                entry[name] = "wrapped"
                continue
            layer_plan.append((cls, name, own))
    # 2. install
    for cls, name, own, make, key in plan:
        setattr(cls, name, make(own))
        rep.machinery[key] = "wrapped"
    for cls, name, own in layer_plan:
        entry = rep.patched_classes[_qual(cls)]
        if name == "forward":
            setattr(cls, name, _make_forward_wrapper(
                own, _qual(cls), _positions_index(inspect.signature(own)), targets.op))
        else:
            setattr(cls, name, _make_init_wrapper(own, cls, rep))
        entry[name] = "wrapped"
    # 3. the report states the END state — every `*_wrapped` bit is read HERE, off the live MRO,
    #    after every install: a class that INHERITS its forward / __init__ (Qwen3_5 from Qwen3Next)
    #    read False at plan time, BEFORE the parent's method was wrapped — the wheel's first install
    #    reported Qwen3_5DecoderLayer forward_wrapped False and readiness refused (GPU proof 507491,
    #    2026-09-15).
    for cls in targets.decoder_layer_classes:
        entry = rep.patched_classes[_qual(cls)]
        for name in ("forward", "__init__"):
            entry[f"{name}_wrapped"] = _is_wrapped(getattr(cls, name, None))
    # every listed class must END UP with a wrapped forward somewhere in its MRO
    for cls in targets.decoder_layer_classes:
        if not _is_wrapped(getattr(cls, "forward", None)):
            raise ResidualSlotError(
                f"residual slot: {_qual(cls)} has no wrapped forward after install")
    return rep


def vllm_patch_targets() -> PatchTargets:
    """The pinned wheel's classes (fact 3, 6, 11; raises ResidualSlotError
    when vllm is not importable or a class is missing)."""
    import importlib

    try:
        from vllm.lora.lora_model import LoRAModel
        from vllm.lora.model_manager import LoRAModelManager
        from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase
        from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner as _RunnerV1
        from vllm.v1.worker.gpu.model_runner import GPUModelRunner as _RunnerV2
        from vllm.v1.worker.gpu.lora_utils import LoraState
    except ImportError as e:
        raise ResidualSlotError(
            f"residual slot: vllm is not importable here ({e}); the class patches cannot be "
            "installed") from e
    from .serving_attestation import MODEL_RUNNER_CLASSES

    runners = (_RunnerV1, _RunnerV2)
    for rcls in runners:
        if _qual(rcls) not in MODEL_RUNNER_CLASSES:
            raise ResidualSlotError(
                f"residual slot: {_qual(rcls)} is not one of the pinned GPU model runners "
                f"{sorted(MODEL_RUNNER_CLASSES)} — not the pinned 0.25.1 surface")
    layers: List[type] = []
    for mod, name in VLLM_DECODER_LAYER_CLASSES:
        try:
            m = importlib.import_module(mod)
            layers.append(getattr(m, name))
        except (ImportError, AttributeError) as e:
            raise ResidualSlotError(
                f"residual slot: {mod}.{name} is not on this vllm ({e}) — not the pinned "
                "0.25.1 surface") from e

    def model_info(manager: Any, arguments: Mapping[str, Any]) -> Tuple[int, int, torch.dtype]:
        vc = arguments.get("vllm_config")
        if vc is None:
            raise ResidualSlotError(
                "residual slot: LoRAModelManager.__init__ received no vllm_config to size the "
                "bank from — not the pinned 0.25.1 surface (create_lora_manager passes it)")
        mc = vc.model_config
        text = mc.hf_text_config
        return int(mc.get_hidden_size()), int(text.num_hidden_layers), mc.dtype

    return PatchTargets(lora_model_cls=LoRAModel, manager_cls=LoRAModelManager,
                        punica_cls=PunicaWrapperBase, runner_mixin_cls=LoRAModelRunnerMixin,
                        decoder_layer_classes=tuple(layers), model_info=model_info,
                        op=ensure_op_registered(), runner_classes=runners,
                        lora_state_cls=LoraState)


def install(targets: Optional[PatchTargets] = None) -> PatchReport:
    """Arm the slot: register the op, install the patches (idempotent),
    publish :data:`PATCH_REPORT`. ``targets`` defaults to the pinned wheel's."""
    global PATCH_REPORT
    if targets is None:
        targets = vllm_patch_targets()
    rep = install_patches(targets, PATCH_REPORT if PATCH_REPORT.armed else None)
    rep.armed = True
    rep.op_name = OP_QUALNAME
    PATCH_REPORT = rep
    log.info("sisteer native_residual: residual slot ARMED (%s; %d decoder classes)",
             OP_QUALNAME, len(targets.decoder_layer_classes))
    return rep


def reset_for_tests() -> None:
    """Forget the live bank and the report (the CPU tests build fresh fakes
    per case; the class patches on those fakes die with them)."""
    global _BANK, PATCH_REPORT
    _BANK = None
    PATCH_REPORT = PatchReport()


def _arm_from_env() -> PatchReport:
    if os.environ.get(ENV_FLAG) != "1":
        return PatchReport(armed=False)
    return install()


PATCH_REPORT = _arm_from_env()
